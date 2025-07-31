# docx-processor.py

import os
import sys
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    DirectoryLoader,
#    ParquetLoader, #not able to import it from the langchain_community
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import config

# mapping from file extension to loader class
LOADER_MAPPING = {
    ".pdf": {"loader": PyPDFLoader, "args": {}},
    ".txt": {"loader": TextLoader, "args": {}},
    ".docx": {"loader": Docx2txtLoader, "args": {}},
#    ".parquet":{"loader": ParquetLoader, "args": {"content_column": config.PARQUET_CONTENT_COLUMN}},
}

def load_documents(folder_path: str) -> list:
    """loads all docs from specified folder, using diff loaders for diff file types."""
    print(f"Loading documents from {folder_path}...")
    all_docs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext in LOADER_MAPPING:
            try:
                """
                loader_info = LOADER_MAPPING[file_ext]
                loader_class = loader_info["loader"]
                loader_args = loader_info["args"]
                """
                loader_class = LOADER_MAPPING[file_ext]["loader"]

                #loader = loader_class(file_path, **loader_args)
                loader = loader_class(file_path)
                docs = loader.load()
                all_docs.extend(docs)
                print(f"  - Loaded {filename} ({len(docs)} documents)")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"  - Skipped {filename} (unsupported file type)")

    return all_docs

# main fn to run the doc ingestion process
def main():
    print("--- Starting Document Ingestion ---")

    if not os.path.exists(config.DOCS_PATH):
        print(f"Error: The documents directory '{config.DOCS_PATH}' was not found.")
        sys.exit(1)

    # 1.loading the docs
    documents = load_documents(config.DOCS_PATH)
    if not documents:
        print("Error: No documents were loaded. Exiting.")
        sys.exit(1)
    print(f"\nLoaded a total of {len(documents)} document pages/sections.")

    # 2. Splitting documents in chunks
    print("Splitting documents into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split content into {len(texts)} chunks.")

    # 3.creating embeddings
    print(f"Loading embedding model: '{config.EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name = config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    # 4. create and save FAISS vector store
    print("Creating and Saving the FAISS vector store...")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(config.DB_PATH)

    print(f"--- Ingestion Complete. Vector store at '{config.DB_PATH}' ---")

if __name__ == "__main__":
    main()
