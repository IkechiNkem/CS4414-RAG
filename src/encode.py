from llama_cpp import Llama
import numpy as np
import json

BATCH_SIZE = 10
embedder = Llama(
    model_path="model/bge-base-en-v1.5-f32.gguf",
    n_gpu_layers=-1,
    n_batch=BATCH_SIZE,
    embedding=True,
    logits_all=False,
)

if __name__ == "__main__":
    print("-- Opening 'data/queries.json' --")
    with open("data/queries.json", "r") as f:
        queries = json.load(f)
    print("-- Loaded queries JSON --")
    print("-- Starting Query Embedding Process --")
    texts = [query["text"] for query in queries]
    all_embeddings = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        all_embeddings.extend(embedder.embed(batch_texts))
        
    for query, emb in zip(queries, all_embeddings):
        query["embedding"] = np.array(emb)
        print(query)
        exit()
    print("-- Query Embedding Complete --")