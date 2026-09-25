"""RAGAs evaluation harness.

Phase 2 work (FR-7): the golden dataset is not committed yet and the RAGAs judge
still defaults to OpenAI rather than the local Ollama model (ISS-15). This module is
carried across the S0-3 restructuring unchanged in behaviour, with its former
module-level body wrapped in :func:`main` so that importing it no longer launches an
evaluation run.

Its third-party imports live inside :func:`main` because ragas, datasets and pandas
ship in the optional ``eval`` extra, which a default ``uv sync`` does not install.
"""


def main():
    """Run the RAGAs evaluation over the committed golden dataset."""
    import pandas as pd
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from rag_qa.chain import build_rag_chain
    from rag_qa.config import load_config

    print("Building RAG chain for evaluation...")
    rag_chain = build_rag_chain(load_config())

    # Load evaluation questions
    questions = []
    ground_truths = []
    df = pd.read_json("eval_dataset.jsonl", lines=True)
    questions.extend(df["question"].tolist())
    ground_truths.extend(df["ground_truth"].tolist())

    # Run the pipeline on all questions
    answers = []
    contexts = []
    for query in questions:
        result = rag_chain.invoke({"question": query})
        answers.append(result["answer"])
        contexts.append([doc.page_content for doc in result["context"]])

    # Create a dataset for RAGAs
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    # Evaluate and print the report
    print("\n--- Running RAGAs Evaluation ---")
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
    )

    print("\n--- Evaluation Report ---")
    print(result)


if __name__ == "__main__":
    main()
