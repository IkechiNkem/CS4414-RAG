def main():
    print("Initializing RAG Pipeline...")
    import encode
    import llm_generation
    import vector_db
    print("Pipeline Intitialized\n")
    while True:
        prompt = input("Enter prompt: ")
        K = input("How many relavant documents: ")
        if (K.isnumeric() and (0 <= int(K) <= vector_db.DOC_COUNT)):
            K = int(K)
            encoded_prompt = encode.encode(prompt)
            documents = []
            if K > 0:
                _, I = vector_db.search(encoded_prompt, K)
                documents = [vector_db.retrieve_document(i) for i in I[0]]
            print("\nRetrieved the following documents:\n")
            for i, d in enumerate(documents):
                print(f"Document {i+1}: {d}\n")
            augmented_prompt = llm_generation.augment(prompt, documents)
            response = llm_generation.respond(augmented_prompt)
            print()
            print("LLM Response: ")
            print(response)
            print()
        else:
            print(f"Please enter a number between 0 and {vector_db.DOC_COUNT} when selecting relavent documents.")
    
if __name__ == "__main__":
    main()