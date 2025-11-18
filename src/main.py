import encode
import llm_generation
import vector_db

def main():
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
            print(f"Passing the following docs: {documents}")
            augmented_prompt = llm_generation.augment(prompt, documents)
            print("Thinking...")
            response = llm_generation.respond(augmented_prompt)
            print("Response:")
            print(response)
        else:
            print(f"Please enter a number between 0 and {vector_db.DOC_COUNT} when selecting relavent documents.")
    

if __name__ == "__main__":
    main()