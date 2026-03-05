from huggingface_hub import hf_hub_download
import os

REPO_ID = "stabilityai/stable-audio-open-1.0"
LOCAL_DIR = "./checkpoints/sao_original"

os.makedirs(LOCAL_DIR, exist_ok=True)

files = ["model.safetensors", "model_config.json"]

for file in files:
    print(f"Downloading {file}...")
    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=file,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )
        print(f"✅ {file} downloaded successfully.")
    except Exception as e:
        print(f"❌ Error downloading {file}: {e}")
        print("Make sure you have accepted the terms on the Hugging Face model page!")

print(f"\n📂 Files are located in: {os.path.abspath(LOCAL_DIR)}")
