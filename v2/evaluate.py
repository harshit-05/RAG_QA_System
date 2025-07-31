#evaluate.py

import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset
from pipeline_builder import build_rag_chain

print("Building RAG chain for evaluation...")
rag_chain = build_rag_chain()

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
    result = rag_chain.invoke({"query": query})
    answers.append(result["result"])
    contexts.append([doc.page_content for doc in result["source_documents"]])

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
