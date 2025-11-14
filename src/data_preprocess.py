from llama_cpp import Llama
import json

embedder = Llama(
    model_path="model/bge-base-en-v1.5-f32.gguf",
    n_gpu_layers=-1,
    embedding=True,
    verbose=False
)

with open("data/documents.json", "r") as f:
    docs = json.load(f)

for doc in docs:
    doc["embedding"] = embedder.create_embedding(doc["text"])["data"][0]["embedding"]

with open("data/emb.json", "w") as f:
    json.dump(docs, f, indent=2)