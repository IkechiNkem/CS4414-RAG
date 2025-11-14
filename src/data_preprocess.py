from llama_cpp import Llama
from multiprocessing import Pool
import json
import time


# Each worker loads the model once at startup
embedder = None

def init_worker():
    global embedder
    embedder = Llama(
        model_path="model/bge-base-en-v1.5-f32.gguf",
        n_gpu_layers=-1,
        n_threads = 2,
        embedding=True,
        verbose=False
    )

def embed_doc(doc):
    doc["embedding"] = embedder.create_embedding(doc["text"])["data"][0]["embedding"]
    return doc  # must return updated doc


if __name__ == "__main__":
    print("-- Opening 'data/documents.json' --")
    with open("data/documents.json", "r") as f:
        docs = json.load(f)
    print("-- Opened 'data/documents.json' and created JSON Object --")
    
    
    print("-- Pooling embedding models across 8 processes --")
    with Pool(processes=6, initializer=init_worker) as p:
        processed_docs = p.map(embed_doc, docs)
    print("-- Embedding models pooled and embeddings created --")

    print("-- Creating 'data/preprocessed_documents.json' --")
    with open("data/preprocessed_documents.json", "w") as f:
        json.dump(processed_docs, f, indent=2)
    print("-- Embeddings complete and saved at 'data/preprocessed_documents.json' --")