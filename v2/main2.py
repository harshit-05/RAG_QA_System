# main2.py

from pipeline_builder import build_rag_chain

def main():
    """Main function to run the query interface."""
    print("--- Initializin the Q&A system from config.yaml ---")
    qa_chain = build_rag_chain()

    print("\n\033[92mSystem is ready. Ask questions.\033[0m")
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
