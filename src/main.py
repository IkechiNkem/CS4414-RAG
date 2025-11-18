import encode
import llm_generation
import vector_db

def main():
    while True:
        prompt = input("Enter prompt: ")
        k = input("How many relavant documets: ")
        if (not k.isnumeric() or (0 <= int(k) <= vector_db.DOC_COUNT)):
            k = int(k)
            encoded_prompt = encode.encode(prompt)
            print(len(encoded_prompt))
            print(encoded_prompt.shape)
            _, I = vector_db.search(encoded_prompt, k)
            print(I[0])
            # documents = [vector_db.retrieve_document(i) for i in I[0]]
        else:
            print(f"Please enter a number between 0 and {vector_db.DOC_COUNT}")
    

if __name__ == "__main__":
    main()