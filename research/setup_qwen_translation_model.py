"""One-time download of the local multilingual translation trial model."""

import os
from pathlib import Path

root = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(root / "translation_data" / "hf_cache")
from huggingface_hub import snapshot_download

destination = root / "models" / "qwen3-1.7b"
snapshot_download(
    repo_id="Qwen/Qwen3-1.7B",
    local_dir=destination,
    allow_patterns=["*.safetensors", "*.json", "*.model"],
    max_workers=2,
)
print(f"Ready: {destination}")
