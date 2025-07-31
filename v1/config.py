# config.py

# --- Paths ---
DOCS_PATH = "/home/cybernyx/my_project/RAG_System/docs"
DB_PATH = "/home/cybernyx/my_project/RAG_System/vectorstore/db_faiss"

# since I am pulling qwen2:7b from ollama, model path is no longer required
# but if you want to use any other model, you can put the .gguf files in models/ directory
# and uncomment it
# MODEL_PATH = "models/"

# --- Embedding model ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- LLM ---
# specifying the ollama model name directly
OLLAMA_MODEL = "qwen2:7b"

# for example if you are using qwen1.5-1.8b-chat.Q4_K_M.gguf
# we will use the following snippets
"""
MODEL_TYPE = "qwen"
LLM_CONFIG = {
    "max_new_tokens":512,
    "temperature":0.7,
    "context_length":4096,
}
"""

# --- RAG ---
# this snippet defines how many relative chunks are sent to the LLM
K_RETRIEVED_CHUNKS = 4

# --- parquet file configuration ---
# PARQUET_CONTENT_COLUMN = "0000_from"

# --- prompt tempelate ---
PROMPT_TEMPLATE = """
Use the following pieces of information to answer the user's question.
The documents are about military missions. Be precise and use the provided context.
If you don't know the answer, just say that you don't know, don't try to make 
up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
