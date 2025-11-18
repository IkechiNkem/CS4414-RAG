import faiss
import numpy as np
import json

with open("data/preprocessed_documents.json", "r") as f:
        preprocessed = json.load(f)      
DIM = 768
DOC_COUNT = len(preprocessed)
documents = {}
vectors = []

for doc in preprocessed:
    vectors.append(doc["embedding"])
    documents[doc["id"]] = doc["text"]
    
doc_matrix = np.array(vectors).reshape(DOC_COUNT,DIM)
index = faiss.IndexFlatL2(DIM)
index.add(doc_matrix)

def search(query_embedding, top_k):
    # query_embedding = query_embedding.reshape(1, DIM)
    D, I = index.search(query_embedding, k = top_k)
    D = np.array(D)
    I = np.array(I)
    return (D, I)
def retrieve_document(id):
    return documents[id]