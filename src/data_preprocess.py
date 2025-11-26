from llama_cpp import Llama
import json

embedder = Llama(
    model_path="models/bge-base-en-v1.5-f32.gguf",
    n_gpu_layers=-1,
    embedding=True,
    logits_all=False,
    no_perf=True,
    verbose=False
)
if __name__ == "__main__":
    print("-- Opening 'data/documents.json' --")
    with open("data/documents.json", "r") as f:
        docs = json.load(f)
    print("-- Loaded documents JSON --")

    print("-- Starting Embedding Process --")
    for doc in docs:
        doc["embedding"] = embedder.embed(doc["text"])
    print("-- Embedding Complete --")

    print("-- Saving to 'data/preprocessed_documents.json' --")
    with open("data/preprocessed_documents.json", "w") as f:
        json.dump(docs, f, indent=2)
    print("-- Saved Successfully --")