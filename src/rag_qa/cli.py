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
session.

``rag-query`` exit codes: **0** the session ended normally; **2** it could not start
(configuration problem, no index yet, or a command-line usage error).
"""

import argparse
import asyncio
import sys
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from rag_qa.answering import AnswerEvent, Done, SourceRef, Sources, Token, stream_answer
from rag_qa.chain import QueryPipeline, build_query_pipeline
from rag_qa.config import ConfigError, load_config
from rag_qa.settings import ENV_CONFIG
from rag_qa.vectorstore import store_exists

GREEN, YELLOW, BLUE, RED, DIM, RESET = (
    "\033[92m", "\033[93m", "\033[94m", "\033[91m", "\033[2m", "\033[0m"
)

EXIT_OK = 0
EXIT_CANNOT_START = 2


def print_sources(sources: list[SourceRef]) -> None:
    """List retrieved chunks with the same [n] numbering the model saw in its prompt."""
    if not sources:
        return
    print(f"\n{DIM}Sources:")
    for ref in sources:
        print(f"  [{ref.n}] {ref.citation}")
    print(RESET, end="")


async def render(events: AsyncIterator[AnswerEvent]) -> None:
    """Print one answer to the terminal as it streams, then its sources."""
    sources: list[SourceRef] = []
    print(f"\n{GREEN}Answer:{RESET}")
    async for event in events:
        match event:
            case Sources():
                sources = event.sources
            case Token():
                print(event.text, end="", flush=True)
            case Done():
                pass
    print()
    print_sources(sources)


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
        print(f"\n{RED}Error: {type(e).__name__}: {e}{RESET}")
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
            try:
                query = read(f"\n{BLUE}Question: {RESET}").strip()
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
        "(configuration error, no index yet, or usage error).",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help=f"config file to use (default: ${ENV_CONFIG}, then ./config.yaml)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``rag-query``). Returns the exit status; see the module docstring."""
    args = build_parser().parse_args(argv)  # --help and usage errors exit here, before any work

    print("--- Initializing the Q&A system ---")
    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_CANNOT_START
    if not store_exists(config.paths.vector_store):
        # Otherwise this surfaces as FAISS's "could not open ... for reading".
        print(
            f"Error: no index at '{config.paths.vector_store}'. Run rag-ingest first "
            f"to build it from the corpus.",
            file=sys.stderr,
        )
        return EXIT_CANNOT_START

    print("Loading vector store and models...")
    pipeline = build_query_pipeline(config)
    return repl(pipeline)


if __name__ == "__main__":
    sys.exit(main())
