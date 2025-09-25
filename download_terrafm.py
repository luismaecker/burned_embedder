from huggingface_hub import snapshot_download

# Download to a specific folder in your project
repo_path = snapshot_download(
    repo_id="MBZUAI/TerraFM", 
    local_dir="./terrafm_models"
)
print(f"Downloaded to: {repo_path}")