Folder and file structure:

├── data/
│   ├── documents.json
│   ├── preprocessed_documents.json
│   └── queries.json
├── models/
│   ├── bge-base-en-v1.5-f32.gguf
│   ├── Llama-3.2-3B-Instruct-Q4_K_M.gguf
│   ├── qwen2-1_5b-instruct-q4_0.gguf
│   └── tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf
├── src/
│   ├── data_preprocess.py
│   ├── encode.py
│   ├── llm_generation.py
│   ├── main.py
│   └── vector_db.py