import sys
import time
from huggingface_hub import hf_hub_download

print("Starting download of AlexWortega/openjev qwen3.5-4b-nli/model.safetensors...")
t0 = time.time()
path = hf_hub_download(
    repo_id="AlexWortega/openjev",
    filename="model.safetensors",
    subfolder="qwen3.5-4b-nli"
)
elapsed = time.time() - t0
print(f"Downloaded model to {path} in {elapsed:.1f}s")
