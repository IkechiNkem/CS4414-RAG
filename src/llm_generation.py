from llama_cpp import Llama

llm = Llama(
    model_path="models/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=2048,
    verbose=False,
    no_perf=True
)
def augment(prompt, documents):
    return prompt + " Top documents:" + "".join(documents)

def respond(augmented_prompt):
    res = llm(augmented_prompt, max_tokens=512)
    return res["choices"][0]["text"]