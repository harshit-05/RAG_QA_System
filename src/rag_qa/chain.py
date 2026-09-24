"""Query chain construction.

Still the legacy ``RetrievalQA`` shape from v2; S0-5 replaces it with the LCEL
composition in ARCHITECTURE.md §0.4 and changes the signature to take a loaded
config rather than a path.
"""

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from rag_qa.config import load_config
from rag_qa.registry import build_object, resolve_ref
from rag_qa.vectorstore import open_store


def build_rag_chain(config_path="config.yaml"):
    """Builds the entire RAG chain from a YAML config file."""
    config = load_config(config_path)

    query_pipeline_config = config["pipeline"]["query"]

    llm = build_object(resolve_ref(config, query_pipeline_config["llm"]))
    embeddings = build_object(resolve_ref(config, config["pipeline"]["ingestion"]["embedder"]))
    print("Loading vector store...")
    db = open_store(embeddings, config)
    base_retriever = db.as_retriever(**resolve_ref(config, query_pipeline_config["retriever"]))
    final_retriever = base_retriever
    if "reranker" in query_pipeline_config:
        print("Building re-ranker...")
        reranker_config = resolve_ref(config, query_pipeline_config["reranker"])
        reranker_config["base_retriever"] = base_retriever
        final_retriever = build_object(reranker_config)

    prompt = PromptTemplate(template=query_pipeline_config["prompt"], input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=final_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    print("RAG chain built successfully.")
    return qa_chain
