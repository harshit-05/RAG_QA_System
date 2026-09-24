"""Corpus ingestion: load, chunk, embed, persist.

Depends on the registry, config and vector store only. It must never import
:mod:`rag_qa.chain` (ARCHITECTURE.md §0.2) — in v2 the dependency ran the wrong way,
which made ingestion drag the whole query stack in with it.
"""

import os
import sys

from rag_qa.config import load_config
from rag_qa.registry import build_object, import_from_string, resolve_ref
from rag_qa.vectorstore import create_store


def load_documents(config):
    """Dynamically loads documents using loaders defined in the config."""
    all_docs = []
    data_path = config["data_path"]

    loader_configs = config["components"]["loaders"].values()

    print(f"Loading documents from '{data_path}'...")
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        file_ext = os.path.splitext(filename)[1].lower()

        loader_found = False
        for loader_config in loader_configs:
            if file_ext in loader_config.get("extensions", []):
                try:
                    loader_class_name = loader_config["_target_"].split('.')[-1]
                    print(f"  - Loading {filename} with {loader_class_name}")
                    LoaderClass = import_from_string(loader_config["_target_"])
                    loader = LoaderClass(file_path)
                    docs = loader.load()
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"    Error loading {filename}: {e}")
                loader_found = True
                break
        if not loader_found:
            print(f"  - Skipped {filename} (no loader configured for this file type)")
    return all_docs


def main(config_path="config.yaml"):
    """Builds the vector store from the ingestion pipeline defined in the config."""
    print("--- Starting Document Ingestion Engine ---")
    config = load_config(config_path)

    ingestion_config = config["pipeline"]["ingestion"]

    documents = load_documents(config)
    if not documents:
        print("Error: No documents were loaded. Exiting.")
        sys.exit(1)
    print(f"\nLoaded a total of {len(documents)} document pages/sections.")

    splitter_config = resolve_ref(config, ingestion_config["splitter"])
    text_splitter = build_object(splitter_config)
    chunks = text_splitter.split_documents(documents)
    print(f"Split content into {len(chunks)} chunks.")

    embedder_config = resolve_ref(config, ingestion_config["embedder"])
    embeddings = build_object(embedder_config)

    print("Creating and saving the FAISS vector store...")
    create_store(chunks, embeddings, config)

    print(f"--- Ingestion Complete. Vector store saved at '{config['vector_store_path']}' ---")


if __name__ == "__main__":
    main()
