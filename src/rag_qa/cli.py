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

**Starting (S2-8, S3-2).** :func:`start` asks Qdrant first, then refuses an unusable index
from its metadata alone, then asks whether the language model's server answers
(:func:`rag_qa.components.preflight`, the check ``rag-serve`` runs), then loads the
models. A failure in any of them, or a Ctrl-C while they load, is one message and exit 2,
not a traceback. Once the session runs, the alias is resolved again before every question
(DEC-19), so an ingest made meanwhile is answered from, and a session never outlives the
generation it started on.

**The hosted fallback (S3-6, DEC-21).** With ``pipeline.query.fallback_llm`` set, Ollama
down at the start is a warning, not exit 2, and the fallback answers until Ollama is back:
"(answered by the hosted fallback, …)" follows each such answer, whose question and sources
left this machine. Without one, a stopped Ollama stops the start, as before.

**Its log (NFR-10, S3-7).** Each complete answer logs one line on stderr, after its sources:
the model that answered, the times and the token counts, never the question or the answer
(:func:`rag_qa.observability.logs.configure_logging`; JSON when stderr is not a terminal).

``rag-query`` exit codes: **0** the session ended normally; **2** it could not start
(configuration problem, Qdrant unusable, no index yet, an index this config cannot use,
the language model's server not answering, a model that would not load, Ctrl-C while
loading, or a command-line usage error). An index it cannot use is one built with another
embedder (DEC-17): it says to re-run ``rag-ingest``, before loading any model.
"""

import argparse
import asyncio
import difflib
import logging
import readline  # noqa: F401  # for its side effect: input() gains line editing and history
import sys
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from rag_qa import resilience
from rag_qa.answering import (
    AnswerEvent,
    Done,
    SourceRef,
    Sources,
    Token,
    answered_line,
    stream_answer,
)
from rag_qa.chain import (
    IncompatibleIndexError,
    NoIndexError,
    QueryPipeline,
    build_query_pipeline,
    check_index,
)
from rag_qa.components import OLLAMA_HINT, build_store, preflight, raised_by_fallback
from rag_qa.config import ConfigError, load_config, resolve_config_path
from rag_qa.manifest import Manifest
from rag_qa.observability.logs import LogSettingsError, configure_logging
from rag_qa.resilience import BreakerOpenError
from rag_qa.schema import RagConfig
from rag_qa.settings import ENV_CONFIG, ENV_FALLBACK_API_KEY
from rag_qa.vectorstore import StoreError, StoreUnavailableError

logger = logging.getLogger(__name__)

GREEN, YELLOW, BLUE, RED, DIM, RESET = (
    "\033[92m",
    "\033[93m",
    "\033[94m",
    "\033[91m",
    "\033[2m",
    "\033[0m",
)

EXIT_OK = 0
EXIT_CANNOT_START = 2

#: Shown under a model that would not load. Offline, a model not in the Hugging Face cache
#: yet fails with an ``OSError``: the embedder, or since S2-4 the reranker.
HF_CACHE_HINT = (
    "If this machine is offline, a model may be missing from the Hugging Face cache: run "
    "rag-query once with network to fetch it."
)

#: Shown under "no index" while v0.3's FAISS folder is still beside the config (DEC-19).
LEGACY_INDEX_HINT = (
    "v0.3's FAISS index in {folder} is no longer read: since 1.0 the index lives in "
    'Qdrant. Run rag-ingest to build it there, then delete {folder} (README, "Upgrading '
    'from v0.3").'
)


def legacy_index_hint(config_dir: Path | None) -> str | None:
    """:data:`LEGACY_INDEX_HINT` when v0.3's ``vectorstore/db_faiss*`` is beside the config
    file, where v0.3's default ``paths.vector_store`` put it; ``None`` otherwise."""
    folder = None if config_dir is None else config_dir / "vectorstore"
    if folder is None or not any(folder.glob("db_faiss*")):
        return None
    return LEGACY_INDEX_HINT.format(folder=folder)


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


async def render(events: AsyncIterator[AnswerEvent], *, fallback: str | None = None) -> None:
    """Print one answer to the terminal as it streams, then its sources.

    The ``Answer:`` header waits for the first token, so an answer that fails or is
    interrupted before any text arrives prints no empty header above its error. An answer
    that completes with no text at all still gets one. When the hosted fallback ``fallback``
    answered, a line under the answer says so (S3-6): its question and sources left this
    machine. A complete answer then logs one line (S3-7).
    """
    sources: list[SourceRef] = []
    answering = False
    done: Done | None = None
    async for event in events:
        if isinstance(event, Sources):
            sources = event.sources
            continue
        if not answering:  # the first Token, or Done for an answer with no text
            print(f"\n{GREEN}Answer:{RESET}")
            answering = True
        if isinstance(event, Token):
            print(event.text, end="", flush=True)
        elif isinstance(event, Done):
            done = event
    print()
    if fallback is not None and done is not None and done.model == fallback:
        print(f"{DIM}(answered by the hosted fallback, {fallback}){RESET}")
    print_sources(sources)
    if done is not None:
        sys.stdout.flush()  # the answer and its sources first, then the log's line on stderr
        logger.info(answered_line(done), extra=asdict(done))


def _connection_failed(error: Exception) -> bool:
    """Whether ``error`` is a refused or dropped connection
    (:data:`~rag_qa.resilience.CONNECTION`): the builtin ``ConnectionError``, or httpx's
    errors for a server it could not reach or that dropped the connection, which ChatOllama's
    stream raises unwrapped (``ConnectError: All connection attempts failed``). Not the read,
    write and pool timeouts: those mean a slow server or a busy client, and "Is Ollama
    running?" would point the wrong way (S2-8's review)."""
    return resilience.classify(error) == resilience.CONNECTION


def describe_failure(error: Exception, llm: BaseChatModel) -> str:
    """A failed answer's error line. A connection error from an Ollama model names Ollama
    and adds :data:`~rag_qa.components.OLLAMA_HINT`: httpx's own text does not say what it
    could not reach. An open breaker says which service, and when it is tried again; the
    hosted fallback's errors say they are its (S3-6)."""
    if isinstance(error, StoreError):  # Qdrant's: worded for the user, the hint included
        return str(error)
    if isinstance(error, BreakerOpenError):
        return f"{error}\n{OLLAMA_HINT}" if error.service == "Ollama" else str(error)
    text = f"{type(error).__name__}: {error}"
    if raised_by_fallback(error):
        refused = getattr(error, "status_code", None) in (401, 403)
        check = f"\nCheck {ENV_FALLBACK_API_KEY}." if refused else ""
        return f"the hosted fallback failed ({text}){check}"
    if _connection_failed(error) and type(llm).__module__.startswith("langchain_ollama"):
        return f"the connection to Ollama failed ({text})\n{OLLAMA_HINT}"
    return text


class _Answer:
    """One answer on the session's loop, with its task kept for :meth:`unwinding`.

    ``Runner.run()`` creates the task itself; it is captured from inside, so it is
    ``None`` only when a Ctrl-C cancelled the task before it first ran.
    """

    def __init__(self, pipeline: QueryPipeline, question: str, sources: Sequence[str] = ()) -> None:
        self.pipeline = pipeline
        self.question = question
        self.sources = tuple(sources)
        self.task: asyncio.Task[Any] | None = None

    async def run(self) -> None:
        self.task = asyncio.current_task()
        pipeline = self.pipeline
        fallback = pipeline.fallback_ref if pipeline.fallback_llm is not None else None
        await render(
            stream_answer(pipeline, self.question, sources=self.sources), fallback=fallback
        )

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


def ask(
    runner: asyncio.Runner,
    pipeline: QueryPipeline,
    question: str,
    sources: Sequence[str] = (),
) -> bool:
    """Answer one question on the session's loop, from ``sources`` only when there are any.
    ``False`` means quit (a second Ctrl-C)."""
    answer = _Answer(pipeline, question, sources)
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
        print(
            f"\n{YELLOW}(could not close the model's connections cleanly: "
            f"{type(e).__name__}){RESET}"
        )
    finally:
        runner.close()


def repl(
    pipeline: QueryPipeline,
    read: Callable[[str], str] | None = None,
    *,
    sources: Sequence[str] = (),
) -> int:
    """Ask questions until the user leaves; one failed answer never ends the session.

    ``read`` defaults to ``input``, looked up at call time (not bound as a default
    argument) so a patched ``input`` is honoured; tests pass a scripted one. With
    ``sources``, every answer comes from those corpus documents only (``--source``).
    """
    read = read or input
    print(f"\n{GREEN}System is ready. Ask questions.{RESET}")
    if sources:
        print(f"{YELLOW}Answering from {', '.join(sources)} only.{RESET}")
    if pipeline.fallback_llm is not None:
        print(
            f"{YELLOW}When {pipeline.llm_breaker.name} cannot answer, the hosted fallback "
            f"{pipeline.fallback_ref} does, and the question and its sources leave this "
            f"machine.{RESET}"
        )
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
            if not ask(runner, pipeline, query, sources):
                print("\nExiting...")
                return EXIT_OK
    finally:
        close_session(runner, pipeline)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-query",
        description="Ask questions about the indexed corpus in an interactive session.",
        epilog="Exit status: 0 session ended normally; 2 could not start "
        "(configuration error, Qdrant unusable, no index yet, an index this config cannot "
        "use, the language model's server not answering, a model that would not load, "
        "Ctrl-C while loading, or usage error).",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help=f"config file to use (default: ${ENV_CONFIG}, then ./config.yaml)",
    )
    parser.add_argument(
        "--source",
        metavar="PATH",
        action="append",
        default=[],
        dest="sources",
        help="answer from this corpus document only, named by its path relative to the "
        "corpus (e.g. report.pdf, or sub/notes.md); repeat it for several. It must be in the "
        "index",
    )
    return parser


def unknown_sources(sources: Sequence[str], manifest: Manifest) -> str | None:
    """Why ``rag-query`` cannot answer from ``sources``, or ``None`` when the index holds
    every one: each is named, with the nearest document it does hold when one is close."""
    unknown = [source for source in sources if source not in manifest.documents]
    if not unknown:
        return None
    named = []
    for source in unknown:
        near = difflib.get_close_matches(source, list(manifest.documents), n=1)
        named.append(f"{source!r}" + (f" (did you mean {near[0]!r}?)" if near else ""))
    return (
        f"--source names {'a document' if len(unknown) == 1 else 'documents'} not in the index: "
        f"{', '.join(named)}. A source is a path relative to the corpus, as rag-ingest indexed "
        f"it, such as {next(iter(sorted(manifest.documents)), 'report.pdf')!r}."
    )


def start(
    config: RagConfig, config_dir: Path | None = None, *, sources: Sequence[str] = ()
) -> QueryPipeline | str:
    """The session's pipeline, or why the session cannot start (exit 2).

    Cheapest first, so a refusal never waits on a later step:

    1. Qdrant (:meth:`~rag_qa.vectorstore.QdrantStore.check`), before anything else
       (DEC-19);
    2. the index, from its metadata alone (:class:`NoIndexError`, or its
       ``IncompatibleIndexError``): one from another embedder would answer with wrong
       chunks. "No index" mentions v0.3's FAISS folder when it is still beside the config
       file, in ``config_dir``. Then ``sources``, ``--source``'s documents, against its
       manifest (:func:`unknown_sources`);
    3. the language model's server (:func:`~rag_qa.components.preflight`): Ollama down, or
       a model it lacks, before the embedder and the reranker take seconds to load. With a
       hosted fallback configured, Ollama down is a warning, printed here, and the session
       starts (DEC-21); the fallback must be buildable, its key set;
    4. the models, in :func:`build_query_pipeline`, the fallback included.

    Only the start is guarded here; once the session runs, :func:`ask` handles each
    failed answer (ISS-06). A Ctrl-C propagates, for :func:`main` to report.
    """
    try:
        store = build_store(config)
        store.check()
    except (ConfigError, StoreUnavailableError) as e:
        return str(e)
    try:
        # build_query_pipeline checks again; a few reads
        _, manifest = check_index(config, store=store)
    except IncompatibleIndexError as e:
        return str(e)
    except NoIndexError as e:
        hint = legacy_index_hint(config_dir)
        return str(e) if hint is None else f"{e}\n{hint}"
    # Qdrant's answer could not be used. Apart from the model catch below, so that its
    # Hugging Face hint is not shown for a problem with the index (S2-8's review).
    except StoreError as e:
        return f"cannot read the index: {e}"
    unknown = unknown_sources(sources, manifest)
    if unknown is not None:
        return unknown
    try:
        check = preflight(config)
        if check.problem is not None:
            return check.problem
        if check.warning is not None:
            print(f"Warning: {check.warning}", file=sys.stderr)
        return build_query_pipeline(config, store=store, fallback=True)
    # From the check that needs the embedder, the dimension; and Qdrant gone since.
    except (NoIndexError, StoreError) as e:
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
    try:
        configure_logging()
    except LogSettingsError as e:
        print(f"Error: invalid log settings: {e}", file=sys.stderr)
        return EXIT_CANNOT_START

    print("--- Initializing the Q&A system ---")
    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_CANNOT_START

    print("Loading vector store and models...")
    sources = tuple(dict.fromkeys(args.sources))  # a source given twice counts once
    try:
        started = start(config, resolve_config_path(args.config).resolve().parent, sources=sources)
    # Ctrl-C while the models load: the user chose to leave, so no traceback. During the
    # start-up check's ping, its asyncio.run raises KeyboardInterrupt, or CancelledError as
    # ask() describes; both mean the same.
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nInterrupted while loading; the session did not start.", file=sys.stderr)
        return EXIT_CANNOT_START
    if isinstance(started, str):
        print(f"Error: {started}", file=sys.stderr)
        return EXIT_CANNOT_START
    return repl(started, sources=sources)


if __name__ == "__main__":
    sys.exit(main())
