from llama_cpp import Llama
import json

BATCH_SIZE = 50
embedder = Llama(
    model_path="models/bge-base-en-v1.5-f32.gguf",
    n_gpu_layers=-1,
    embedding=True,
    n_batch=BATCH_SIZE,
    logits_all=False,
)


if __name__ == "__main__":
    print("-- Opening 'data/documents.json' --")
    with open("data/documents.json", "r") as f:
        docs = json.load(f)
    print("-- Loaded documents JSON --")

    print("-- Starting Embedding Process --")
    texts = [doc["text"] for doc in docs]
    all_embeddings = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        all_embeddings.extend(embedder.embed(batch_texts))
        
    for doc, emb in zip(docs, all_embeddings):
        doc["embedding"] = emb
    print("-- Embedding Complete --")

    print("-- Saving to 'data/preprocessed_documents.json' --")
    with open("data/preprocessed_documents.json", "w") as f:
        json.dump(docs, f, indent=2)
    print("-- Saved Successfully --")