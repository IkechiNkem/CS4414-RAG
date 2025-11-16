import llama_cpp
from llama_cpp import Llama
import json
import math

BATCH_SIZE = 50
embedder = Llama(
    model_path="model/bge-base-en-v1.5-f32.gguf",
    n_gpu_layers=-1,
    n_ctx=512,
    embedding=True,
    verbose=False,
    n_batch=BATCH_SIZE,
    logits_all=False,
    pooling_type=llama_cpp.LLAMA_POOLING_TYPE_MEAN
)


def embed_batch(texts):
    result = embedder.create_embedding(texts)
    return [entry["embedding"] for entry in result["data"]]

if __name__ == "__main__":
    print("-- Opening 'data/documents.json' --")
    with open("data/documents.json", "r") as f:
        docs = json.load(f)
    print("-- Loaded documents JSON --")

    print("-- Starting Embedding Process --")
    texts = [doc["text"] for doc in docs]
    all_embeddings = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        all_embeddings.extend(embed_batch(batch_texts))
    for doc, emb in zip(docs, all_embeddings):
        doc["embedding"] = emb
    print("-- Embedding Complete --")

    print("-- Saving to 'data/preprocessed_documents.json' --")
    with open("data/preprocessed_documents.json", "w") as f:
        json.dump(docs, f, indent=2)
    print("-- Saved Successfully --")