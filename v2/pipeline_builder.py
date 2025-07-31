#pipeline_builder.py

import os
import yaml
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from importlib import import_module #helps in dynamically importing classes from strings
from langchain_groq import ChatGroq

def import_from_string(dotted_path):
    """Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed."""

    try:
        module_path, class_name = dotted_path.rsplit('.', 1)
        module = import_module(module_path)
        return getattr(module, class_name)
    except (ValueError, AttributeError, ImportError) as e:
        raise ImportError(f"Could not import {dotted_path}") from e

def build_object(config_dict):
    """Recursively build objects from a config dictionary."""
    if "_target_" in config_dict:
        class_to_instantiate = import_from_string(config_dict["_target_"])
        args = {key: build_object(value) for key, value in config_dict.items() if key != "_target_"}
        return class_to_instantiate(**args)
    elif isinstance(config_dict, dict):
        return {key: build_object(value) for key, value in config_dict.items()}
    elif isinstance(config_dict, list):
        return [build_object(item) for item in config_dict]
    else:
        return config_dict

def get_component_from_path(config, path_str):
    """Navigate a dot-separated path in the config dictionary."""
    keys = path_str.split('.')
    value = config
    for key in keys:
        value = value[key]
    return value

def build_rag_chain(config_path="config.yaml"):
    """Builds the entire RAG chain from a YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    query_pipeline_config = config["pipeline"]["query"]

    llm = _build_object(_get_component_from_path(config, query_pipeline_config["llm"]))
    embeddings = _build_object(_get_component_from_path(config, config["pipeline"]["ingestion"]["embedder"]))
    
    print("Loading vector store...")
    db = FAISS.load_local(config["vector_store_path"], embeddings, allow_dangerous_deserialization=True)
    
    base_retriever = db.as_retriever(**_get_component_from_path(config, query_pipeline_config["retriever"]))
    
    final_retriever = base_retriever
    if "reranker" in query_pipeline_config:
        print("Building re-ranker...")
        reranker_config = _get_component_from_path(config, query_pipeline_config["reranker"])
        reranker_config["base_retriever"] = base_retriever
        final_retriever = _build_object(reranker_config)

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
    """
    #1. building the main components
    llm_path = config["pipeline"]["llm"]
    llm_config = get_component_from_path(config, llm_path)
    llm = build_object(llm_config)
    
    embedding_path = config["pipeline"]["embedding"]
    embedding_config = get_component_from_path(config, embedding_path)
    embeddings = build_object(embedding_config)

    #2. loading the vector store
    print("Loading vector store...")
    db = FAISS.load_local(
        config["vector_store_path"],
        embeddings,
        allow_dangerous_deserialization=True
    )

    #3. Building the retriever
    retriever_config_path = config["pipeline"]["retriever"]
    retriever_config = get_component_from_path(config, retriever_config_path)
    base_retriever = db.as_retriever(**retriever_config)

    #4. Checking for a Re-ranker
    final_retriever = base_retriever
    if "reranker" in config["pipeline"]:
        print("Building re-ranker...")
        reranker_path = config["pipeline"]["reranker"]
        reranker_config = get_component_from_path(config, reranker_path)
        # We need to inject the base_retriever into the reranker config
        reranker_config["base_retriever"] = base_retriever
        final_retriever = build_object(reranker_config)

    #5. Building the final QA chain
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=final_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    print("RAG chain built successfully.")
    return qa_chain
"""
# You'll need to define your prompt template here or load it from the config
#PROMPT_TEMPLATE = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

