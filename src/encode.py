from llama_cpp import Llama
import numpy as np

BATCH_SIZE = 50
embedder = Llama(
    model_path="models/bge-base-en-v1.5-f32.gguf",
    n_gpu_layers=-1,
    embedding=True,
    logits_all=False,
    no_perf=True,
    n_batch=BATCH_SIZE,
    verbose=False
)
def encode(query):
    return np.array(embedder.embed(query)).reshape(768,)