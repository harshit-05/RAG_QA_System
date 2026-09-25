"""Interactive query REPL.

A thin client over :func:`rag_qa.chain.build_rag_chain` (SRS §8.2): the CLI and the
Phase 2 HTTP API are two front ends onto the same chain object, never two code paths.

Answers stream token by token. On CPU a full answer takes a minute or more, and the
chain emits the retrieved context before the first token, so sources are known
before the answer finishes — the same order the Phase 2 SSE endpoint will use.
"""

from rag_qa.chain import build_rag_chain, citation
from rag_qa.config import load_config

GREEN, YELLOW, BLUE, DIM, RESET = "\033[92m", "\033[93m", "\033[94m", "\033[2m", "\033[0m"


def print_sources(context):
    """List retrieved chunks with the same [n] numbering the model saw in its prompt."""
    if not context:
        return
    print(f"\n{DIM}Sources:")
    for i, doc in enumerate(context, 1):
        print(f"  [{i}] {citation(doc)}")
    print(RESET, end="")


def answer(chain, question):
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


def main():
    """Main function to run the query interface."""
    print("--- Initializing the Q&A system ---")
    config = load_config()
    print("Loading vector store and models...")
    chain = build_rag_chain(config)

    print(f"\n{GREEN}System is ready. Ask questions.{RESET}")
    print(f"{YELLOW}Type 'exit' or 'quit' to end.{RESET}")

    while True:
        query = input(f"\n{BLUE}Question: {RESET}").strip()
        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        if not query:
            continue

        print("Searching for an answer...")
        answer(chain, query)


if __name__ == "__main__":
    main()
