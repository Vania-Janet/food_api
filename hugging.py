from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

# Push your dataset files
upload_folder(folder_path="/Users/vania/Documents/MLProject/RAG/FAISS_recetas", repo_id="vania-janet/data-ML-Project", repo_type="dataset")
