# main.py

import os
import sys
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import  HuggingFaceEmbeddings
#from langchain_community.llms import CTransformers
from langchain_community.llms import Ollama 
#swapping CTransformers for ollama as i will be pulling qwen2:7b from ollama

import config #importing our configuration from config.py

def create_qa_chain():
    """Sets up and return the Retrieval QA chain."""
    if not os.path.exists(config.DB_PATH):
        print(f"Error: Vector DB not found at {config.DB_PATH}. Please run docs_process.py first.")
        sys.exit(1)
    """
    if not os.path.exists(config.MODEL_PATH):
        print(f"Error: LLM model not found at {config.MODEL_PATH}.")
        sys.exit(1)
    """
    # loading the embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name = config.EMBEDDING_MODEL,
        model_kwargs = {"device":"cpu"}
    )

    # loading the FAISS vector store
    print("Loading vector store...")
    db = FAISS.load_local(
        config.DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True # needed for local FAISS loading
    )

    # creating the retriever
    retriever = db.as_retriever(search_kwargs={"k": config.K_RETRIEVED_CHUNKS})

    # prompt creation from the template
    prompt = PromptTemplate(
        template=config.PROMPT_TEMPLATE, input_variables=["context","question"]
    )

   # Loading the LLM from Ollama
    print(f"Loading Ollama model: '{config.OLLAMA_MODEL}'...")
    llm = Ollama(model=config.OLLAMA_MODEL) # <-- This is the new, simpler way
    
    # Creating the Retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    
    return qa_chain

def main():
    """Main function to run the query interface."""
    print("--- Initializing Q&A System ---")
    qa_chain = create_qa_chain()
    print("\n\033[92mSystem is ready. Ask questions about your mission documents.\033[0m")
    print("\033[93mType 'exit' or 'quit' to end.\033[0m")
    
    while True:
        query = input("\n\033[94mQuestion: \033[0m").strip()
        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        if not query:
            continue

        print("Searching for an answer...")
        result = qa_chain.invoke({"query": query})
        
        print("\n\033[92mAnswer:\033[0m")
        print(result["result"])

if __name__ == "__main__":
    main()
