from llama_cpp import Llama

llm = Llama(
    model_path="models/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf",
    n_gpu_layers=-1,
)
def augment(prompt, documents):
    return prompt + " Top documents:" + "".join(documents)

def respond(augmented_prompt):
    res = llm.create_completion(augmented_prompt)
    print(res)