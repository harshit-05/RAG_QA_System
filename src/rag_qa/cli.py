"""Interactive query REPL.

A thin client over :func:`rag_qa.answering.stream_answer` (SRS §8.2, DEC-14): the CLI,
the Phase 2 HTTP API and the evaluation harness consume the same answer stream, never
three code paths. Answers stream token by token; sources are known before the first
token and are printed after the answer, numbered as the prompt numbered them.

**One event loop per session.** The session runs on one :class:`asyncio.Runner`, so
ChatOllama's async connection pool stays on one loop, and each answer is one
``runner.run()``. Between answers the default SIGINT handler is back, so ``input()``
behaves as it always did.

**Ctrl-C (DEC-14).** During an answer, the Runner's own SIGINT handling cancels the
answer's task; the stream to Ollama is closed before ``run()`` returns, so generation
stops. "(answer interrupted…)" is printed and the prompt returns. A second Ctrl-C while
that answer is still unwinding quits the session. Ctrl-C at the prompt, end of input
(piped stdin), ``exit`` or ``quit`` end the session too. However it ends, the model's
HTTP clients are closed on the session's loop.

**Staying alive (S1-4, ISS-06).** One failed answer — Ollama stopped, a timeout, a
model error — prints the error and returns to the prompt instead of ending the
session. A connection error from an Ollama model names Ollama, with a hint.

**Line editing (S2-8).** Importing :mod:`readline` gives ``input()`` line editing and an
in-session history: an arrow key moves the cursor instead of reaching the model as
``^[[A``.

**Starting (S2-8).** :func:`start` refuses an unusable index from its files alone, then
asks whether the language model's server answers (:func:`rag_qa.components.preflight`,
the check ``rag-serve`` runs), then loads the models. A failure in any of them, or a
Ctrl-C while they load, is one message and exit 2, not a traceback.

``rag-query`` exit codes: **0** the session ended normally; **2** it could not start
(configuration problem, no index yet, an index this config cannot use, the language
model's server not answering, a model that would not load, Ctrl-C while loading, or a
command-line usage error). An index it cannot use is a v0.2 one, or one built with another
embedder (DEC-17): it says to re-run ``rag-ingest``, before loading any model.
"""

import argparse
import asyncio
import readline  # noqa: F401  # for its side effect: input() gains line editing and history
import sys
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel

from rag_qa.answering import AnswerEvent, SourceRef, Sources, Token, stream_answer
from rag_qa.chain import NoIndexError, QueryPipeline, build_query_pipeline, check_index
from rag_qa.components import OLLAMA_HINT, preflight
from rag_qa.config import ConfigError, load_config
from rag_qa.schema import RagConfig
from rag_qa.settings import ENV_CONFIG

GREEN, YELLOW, BLUE, RED, DIM, RESET = (
    "\033[92m", "\033[93m", "\033[94m", "\033[91m", "\033[2m", "\033[0m"
)

EXIT_OK = 0
EXIT_CANNOT_START = 2

#: Shown under a model that would not load. Offline, a model not in the Hugging Face cache
#: yet fails with an ``OSError``: the embedder, or since S2-4 the reranker.
HF_CACHE_HINT = (
    "If this machine is offline, a model may be missing from the Hugging Face cache: run "
    "rag-query once with network to fetch it."
)


def question_prompt() -> str:
    """The question prompt, in blue.

    When ``input()`` reads through readline (stdin and stdout both a terminal), the colour
    codes are marked as taking no width, ``\\001``…``\\002``; otherwise readline counts
    them, and the cursor lands in the wrong place once a line is edited or wraps. Anywhere
    else ``input()`` writes the prompt itself, and the markers would be stray bytes.

    The prompt ends on its space, after the reset: libedit, the readline of uv's Python,
    drops a marked sequence that ends a prompt, which left the typed question blue.
    """
    if sys.stdin.isatty() and sys.stdout.isatty():
        return f"\001{BLUE}\002Question:\001{RESET}\002 "
    return f"{BLUE}Question:{RESET} "


def print_sources(sources: list[SourceRef]) -> None:
    """List retrieved chunks with the same [n] numbering the model saw in its prompt."""
    if not sources:
        return
    print(f"\n{DIM}Sources:")
    for ref in sources:
        print(f"  [{ref.n}] {ref.citation}")
    print(RESET, end="")


async def render(events: AsyncIterator[AnswerEvent]) -> None:
    """Print one answer to the terminal as it streams, then its sources.

    The ``Answer:`` header waits for the first token, so an answer that fails or is
    interrupted before any text arrives prints no empty header above its error. An answer
    that completes with no text at all still gets one.
    """
    sources: list[SourceRef] = []
    answering = False
    async for event in events:
        if isinstance(event, Sources):
            sources = event.sources
            continue
        if not answering:  # the first Token, or Done for an answer with no text
            print(f"\n{GREEN}Answer:{RESET}")
            answering = True
        if isinstance(event, Token):
            print(event.text, end="", flush=True)
    print()
    print_sources(sources)


#: httpx's errors for a server that cannot be reached or that dropped the connection.
#: Not its read, write and pool timeouts: those mean a slow server or a busy client, and
#: "Is Ollama running?" would point the wrong way (S2-8's review).
_UNREACHABLE = {"ConnectError", "ConnectTimeout", "ReadError", "WriteError", "RemoteProtocolError"}


def _connection_failed(error: Exception) -> bool:
    """Whether ``error`` is a refused or dropped connection: the builtin
    ``ConnectionError``, or one of httpx's :data:`_UNREACHABLE`, which ChatOllama's stream
    raises unwrapped (``ConnectError: All connection attempts failed``). httpx is matched
    by name rather than imported: it comes with ollama, and is not a dependency of ours."""
    return isinstance(error, ConnectionError) or any(
        kind.__module__ == "httpx" and kind.__name__ in _UNREACHABLE
        for kind in type(error).__mro__
    )


def describe_failure(error: Exception, llm: BaseChatModel) -> str:
    """A failed answer's error line. A connection error from an Ollama model names Ollama
    and adds :data:`~rag_qa.components.OLLAMA_HINT`: httpx's own text does not say what it
    could not reach."""
    text = f"{type(error).__name__}: {error}"
    if _connection_failed(error) and type(llm).__module__.startswith("langchain_ollama"):
        return f"the connection to Ollama failed ({text})\n{OLLAMA_HINT}"
    return text


class _Answer:
    """One answer on the session's loop, with its task kept for :meth:`unwinding`.

    ``Runner.run()`` creates the task itself; it is captured from inside, so it is
    ``None`` only when a Ctrl-C cancelled the task before it first ran.
    """

    def __init__(self, pipeline: QueryPipeline, question: str) -> None:
        self.pipeline = pipeline
        self.question = question
        self.task: asyncio.Task[Any] | None = None

    async def run(self) -> None:
        self.task = asyncio.current_task()
        await render(stream_answer(self.pipeline, self.question))

    def unwinding(self) -> bool:
        """Whether a second Ctrl-C interrupted the cancelled answer before it finished.

        The first Ctrl-C cancels the task and ``run()`` returns once it has unwound, so
        the task is cancelled. The second raises ``KeyboardInterrupt`` at once, from
        inside the loop: the task is then still pending, or (when the raise landed in
        its own step) finished with that ``KeyboardInterrupt``, which reading it here
        also marks as retrieved.
        """
        task = self.task
        if task is None:
            return False
        if not task.done():
            return True
        return not task.cancelled() and isinstance(task.exception(), KeyboardInterrupt)


def ask(runner: asyncio.Runner, pipeline: QueryPipeline, question: str) -> bool:
    """Answer one question on the session's loop. ``False`` means quit (a second Ctrl-C)."""
    answer = _Answer(pipeline, question)
    try:
        runner.run(answer.run())
    # run() raises KeyboardInterrupt after a Ctrl-C, or CancelledError when the task's
    # cancel count does not return to zero (Python 3.12 runners.py); both mean the same.
    except (KeyboardInterrupt, asyncio.CancelledError):
        if answer.unwinding():
            return False
        print(f"\n{YELLOW}(answer interrupted; Ctrl-C again at the prompt quits){RESET}")
    # Deliberately broad (ISS-06): this is the REPL's boundary, and whatever
    # failed — the Ollama client, the retriever, the model — the right response
    # is the same: say so and keep the session. KeyboardInterrupt and SystemExit
    # are not Exceptions, so quitting still works.
    except Exception as e:  # noqa: BLE001
        print(f"\n{RED}Error: {describe_failure(e, pipeline.llm)}{RESET}")
        print("That question was not answered; ask again or try another.")
    return True


async def _settle(tasks: set[asyncio.Task[Any]]) -> None:
    await asyncio.gather(*tasks, return_exceptions=True)


def close_session(runner: asyncio.Runner, pipeline: QueryPipeline) -> None:
    """Close the model's HTTP clients on the session's loop, then the loop itself.

    After a second Ctrl-C the interrupted answer is still pending on the loop. It is
    cancelled before the loop runs again, so it unwinds as a cancel instead of
    resuming with the ``KeyboardInterrupt`` and raising it a second time, and it has
    finished before the clients close. ``runner.close()`` then finds nothing left.
    """
    try:
        leftovers = asyncio.all_tasks(runner.get_loop())
        for task in leftovers:
            task.cancel()
        if leftovers:
            runner.run(_settle(leftovers))
        runner.run(pipeline.aclose())
    except KeyboardInterrupt:
        # Ctrl-C while the clients close: the session is already over, so leave quietly
        # rather than print a traceback. The sockets go with the process.
        print(f"\n{YELLOW}(closing interrupted){RESET}")
    # Best effort, for the same reason: the user has already left, and an error while
    # closing (httpx's, say) must not turn a clean exit into a traceback and exit 1.
    except Exception as e:  # noqa: BLE001
        print(f"\n{YELLOW}(could not close the model's connections cleanly: "
              f"{type(e).__name__}){RESET}")
    finally:
        runner.close()


def repl(pipeline: QueryPipeline, read: Callable[[str], str] | None = None) -> int:
    """Ask questions until the user leaves; one failed answer never ends the session.

    ``read`` defaults to ``input``, looked up at call time (not bound as a default
    argument) so a patched ``input`` is honoured; tests pass a scripted one.
    """
    read = read or input
    print(f"\n{GREEN}System is ready. Ask questions.{RESET}")
    print(f"{YELLOW}Type 'exit' or 'quit' to end.{RESET}")

    runner = asyncio.Runner()
    try:
        while True:
            print()  # not in the prompt: readline is safest with a one-line prompt
            try:
                query = read(question_prompt()).strip()
            except (EOFError, KeyboardInterrupt):
                # End of piped input, or Ctrl-C at the prompt: a normal way to leave,
                # not an error worth a traceback.
                print("\nExiting...")
                return EXIT_OK
            if query.lower() in ("exit", "quit"):
                print("Exiting...")
                return EXIT_OK
            if not query:
                continue

            print("Searching for an answer...")
            if not ask(runner, pipeline, query):
                print("\nExiting...")
                return EXIT_OK
    finally:
        close_session(runner, pipeline)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-query",
        description="Ask questions about the indexed corpus in an interactive session.",
        epilog="Exit status: 0 session ended normally; 2 could not start "
        "(configuration error, no index yet, an index this config cannot use, the language "
        "model's server not answering, a model that would not load, Ctrl-C while loading, "
        "or usage error).",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help=f"config file to use (default: ${ENV_CONFIG}, then ./config.yaml)",
    )
    return parser


def start(config: RagConfig) -> QueryPipeline | str:
    """The session's pipeline, or why the session cannot start (exit 2).

    Cheapest first, so a refusal never waits on a later step:

    1. the index, from its files alone (:class:`NoIndexError`, or its
       ``IncompatibleIndexError``): otherwise a missing index would surface as FAISS's
       "could not open ... for reading", and one from another embedder would answer with
       wrong chunks;
    2. the language model's server (:func:`~rag_qa.components.preflight`): Ollama down, or
       a model it lacks, before the embedder and the reranker take seconds to load;
    3. the models, in :func:`build_query_pipeline`.

    Only the start is guarded here; once the session runs, :func:`ask` handles each
    failed answer (ISS-06). A Ctrl-C propagates, for :func:`main` to report.
    """
    try:
        check_index(config)  # build_query_pipeline checks again; a few small reads
    except NoIndexError as e:
        return str(e)
    # The manifest or the link could not be read. Apart from the model catch below, so that
    # its Hugging Face hint is not shown for a problem with the index (S2-8's review).
    except OSError as e:
        return f"cannot read the index: {type(e).__name__}: {e}"
    try:
        problem = preflight(config)
        if problem is not None:
            return problem
        return build_query_pipeline(config)
    # From the checks that need the embedder: the dimension, and an index that will not
    # open.
    except NoIndexError as e:
        return str(e)
    # Deliberately broad, as preflight is: whatever stops the models loading (a model
    # missing from the HF cache offline, Ollama gone since the check) stops the session
    # before it starts, and deserves a reason rather than a traceback.
    except Exception as e:  # noqa: BLE001
        hint = f"\n{HF_CACHE_HINT}" if isinstance(e, OSError) else ""
        return f"cannot load the models: {type(e).__name__}: {e}{hint}"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``rag-query``). Returns the exit status; see the module docstring."""
    args = build_parser().parse_args(argv)  # --help and usage errors exit here, before any work

    print("--- Initializing the Q&A system ---")
    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_CANNOT_START

    print("Loading vector store and models...")
    try:
        started = start(config)
    # Ctrl-C while the models load: the user chose to leave, so no traceback. During the
    # start-up check's ping, its asyncio.run raises KeyboardInterrupt, or CancelledError as
    # ask() describes; both mean the same.
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nInterrupted while loading; the session did not start.", file=sys.stderr)
        return EXIT_CANNOT_START
    if isinstance(started, str):
        print(f"Error: {started}", file=sys.stderr)
        return EXIT_CANNOT_START
    return repl(started)


if __name__ == "__main__":
    sys.exit(main())
