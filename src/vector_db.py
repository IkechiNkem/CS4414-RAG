import faiss
import numpy as np
import json
from  llama_cpp import Llama

with open("data/preprocessed_documents.json", "r") as f:
        preprocessed = json.load(f)
        
d = 768
nb = preprocessed[-1]["id"] + 1
batch_size = 1

vectors = [doc["embedding"] for doc in preprocessed]
doc_matrix = np.array(vectors).reshape(nb,d)

index = faiss.IndexFlatL2(d)
print(f"FAISS index Trained Status: {index.is_trained}")
index.add(doc_matrix)
print(f"FAISS index has {index.ntotal} embeddings in its search space (should be 10000)")

query_embedding = np.array(preprocessed[42]["embedding"]).reshape(batch_size, d)
D, I = index.search(query_embedding, k = 5)
D = np.array(D)
I = np.array(I)

query = {}
query["id"] = 42
query["text"] = preprocessed[42]["text"]
print(f"Query:\nId: {query["id"]}\nText: {query["text"]}")
print("------------------------------------------------------")
print()
print("[")
for idx in range(I.size):
    if idx < (I.size - 1):
        print(f"id: {I[0,idx]}\nDistance: {D[0,idx]}\nText: {preprocessed[idx]["text"]},\n")
    else:
        print(f"id: {I[0,idx]}\nDistance: {D[0,idx]}\nText: {preprocessed[idx]["text"]}")
print("]")