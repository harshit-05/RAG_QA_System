"""Interactive query REPL.

A thin client over :func:`rag_qa.chain.build_rag_chain` (SRS §8.2): the CLI and the
Phase 2 HTTP API are two front ends onto the same chain object, never two code paths.

Answers stream token by token. On CPU a full answer takes a minute or more, and the
chain emits the retrieved context before the first token, so sources are known
before the answer finishes — the same order the Phase 2 SSE endpoint will use.

**Staying alive (S1-4, ISS-06).** One failed answer — Ollama stopped, a timeout, a
model error — prints the error and returns to the prompt instead of ending the
session. Ctrl-C while an answer streams abandons that answer; Ctrl-C at the prompt,
end of input (piped stdin), ``exit`` or ``quit`` end the session cleanly.

``rag-query`` exit codes: **0** the session ended normally; **2** it could not start
(configuration problem, no index yet, or a command-line usage error).
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import Any

from rag_qa.chain import build_rag_chain, citation
from rag_qa.config import ConfigError, load_config
from rag_qa.settings import ENV_CONFIG
from rag_qa.vectorstore import store_exists

GREEN, YELLOW, BLUE, RED, DIM, RESET = (
    "\033[92m", "\033[93m", "\033[94m", "\033[91m", "\033[2m", "\033[0m"
)

EXIT_OK = 0
EXIT_CANNOT_START = 2


def print_sources(context: list) -> None:
    """List retrieved chunks with the same [n] numbering the model saw in its prompt."""
    if not context:
        return
    print(f"\n{DIM}Sources:")
    for i, doc in enumerate(context, 1):
        print(f"  [{i}] {citation(doc)}")
    print(RESET, end="")


def answer(chain: Any, question: str) -> None:
    """Stream one answer to the terminal, then print its sources."""
    context = []
    print(f"\n{GREEN}Answer:{RESET}")
    for chunk in chain.stream({"question": question}):
        if "context" in chunk:
            context = chunk["context"]
        if "answer" in chunk:
            print(chunk["answer"], end="", flush=True)
    print()
    print_sources(context)


def repl(chain: Any, read: Callable[[str], str] | None = None) -> int:
    """Ask questions until the user leaves; one failed answer never ends the session.

    ``read`` defaults to ``input``, looked up at call time (not bound as a default
    argument) so a patched ``input`` is honoured; tests pass a scripted one.
    """
    read = read or input
    print(f"\n{GREEN}System is ready. Ask questions.{RESET}")
    print(f"{YELLOW}Type 'exit' or 'quit' to end.{RESET}")

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
        try:
            answer(chain, query)
        except KeyboardInterrupt:
            print(f"\n{YELLOW}(answer interrupted; Ctrl-C again at the prompt quits){RESET}")
        # Deliberately broad (ISS-06): this is the REPL's boundary, and whatever
        # failed — the Ollama client, the retriever, the model — the right response
        # is the same: say so and keep the session. KeyboardInterrupt and SystemExit
        # are not Exceptions, so quitting still works.
        except Exception as e:  # noqa: BLE001
            print(f"\n{RED}Error: {type(e).__name__}: {e}{RESET}")
            print("That question was not answered; ask again or try another.")


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
    chain = build_rag_chain(config)
    return repl(chain)


if __name__ == "__main__":
    sys.exit(main())
