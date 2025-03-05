# IMPORTANT: Before running this script run: export HF_TOKEN=<your_token>
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-3.2-1B",
    repo_type="model",
    local_dir="./llama3",
    allow_patterns=["original/*"]  # Downloads only files inside "original/"
)