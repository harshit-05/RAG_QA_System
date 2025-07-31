# file_processor.py
import os
import sys
import yaml
from langchain_community.vectorstores import FAISS
from pipeline_builder import _build_object, _get_component_from_path

def load_documents(config):
    """Dynamically loads documents using loaders defined in the config."""
    all_docs = []
    data_path = config["data_path"]
    ingestion_config = config["pipeline"]["ingestion"]
    
    loader_paths = ingestion_config["loaders"]
    loader_configs = [_get_component_from_path(config, path) for path in loader_paths]

    print(f"Loading documents from '{data_path}'...")
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        for loader_config in loader_configs:
            loader_class_name = loader_config["_target_"].split('.')[-1]
            if file_ext[1:] in loader_class_name.lower():
                try:
                    print(f"  - Loading {filename} with {loader_class_name}")
                    loader = _build_object(loader_config)
                    docs = loader.load(file_path)
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"    Error loading {filename}: {e}")
                break
    return all_docs

def main(config_path="config.yaml"):
    """Builds the vector store from the ingestion pipeline defined in the config."""
    print("--- Starting Document Ingestion Engine ---")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    ingestion_config = config["pipeline"]["ingestion"]

    documents = load_documents(config)
    if not documents:
        print("Error: No documents were loaded. Exiting.")
        sys.exit(1)
    print(f"\nLoaded a total of {len(documents)} document pages/sections.")
    
    text_splitter = _build_object(_get_component_from_path(config, ingestion_config["splitter"]))
    chunks = text_splitter.split_documents(documents)
    print(f"Split content into {len(chunks)} chunks.")
    
    embeddings = _build_object(_get_component_from_path(config, ingestion_config["embedder"]))
    
    print("Creating and saving the FAISS vector store...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(config["vector_store_path"])
    
    print(f"--- Ingestion Complete. Vector store saved at '{config['vector_store_path']}' ---")

if __name__ == "__main__":
    main()
