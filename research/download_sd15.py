"""Download only the fp16 components needed for the inpainting probe."""
from pathlib import Path
from huggingface_hub import snapshot_download

here = Path(__file__).parent
path = snapshot_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-inpainting",
    local_dir=here / "models" / "sd15-inpainting",
    allow_patterns=[
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "text_encoder/*.json", "text_encoder/*.fp16.safetensors",
        "unet/*.json", "unet/*.fp16.safetensors",
        "vae/*.json", "vae/*.fp16.safetensors",
    ],
    max_workers=3,
)
print(path)
