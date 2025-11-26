from llama_cpp import Llama

llm = Llama(
    model_path="models/qwen2-1_5b-instruct-q4_0.gguf",
    n_gpu_layers=-1,
    n_ctx=32768,
    verbose=False,
    no_perf=True
)
def augment(prompt, documents):
    return prompt + " Top documents:" + ". ".join(documents)

def respond(augmented_prompt):
    res = llm(augmented_prompt, max_tokens=512)
    return res["choices"][0]["text"].strip()