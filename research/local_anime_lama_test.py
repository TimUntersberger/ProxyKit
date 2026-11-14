"""Illustration-trained LaMa local inference; mask-only composite."""

from pathlib import Path
from time import perf_counter
import hashlib
import numpy as np
import torch
from PIL import Image, ImageFilter

HERE = Path(__file__).parent
MODEL = HERE / "models" / "anime-manga-big-lama.pt"
SOURCE = Path(r"C:\Users\timun\Downloads\OP18-085.webp")
MASK = HERE / "approx_mask.png"

md5 = hashlib.md5(MODEL.read_bytes()).hexdigest()
print("model_bytes", MODEL.stat().st_size, "md5", md5)
if md5 != "29f284f36a0a510bcacf39ecf4c4d54f":
    raise RuntimeError("Unexpected model hash")

torch.set_num_threads(4)
model = torch.jit.load(str(MODEL), map_location="cpu").eval()

base_mask = Image.open(MASK).convert("L").point(lambda x: 255 if x > 12 else 0)
full_mask = base_mask.filter(ImageFilter.MaxFilter(11))
inner_mask = base_mask.filter(ImageFilter.MinFilter(9))
ring_mask = Image.fromarray(np.where(np.asarray(full_mask) > np.asarray(inner_mask), 255, 0).astype("uint8"))

for label, source_path, mask_image in (
    ("direct", SOURCE, full_mask),
    ("hybrid", HERE / "inverse_0.65.png", ring_mask),
):
    source = Image.open(source_path).convert("RGB")
    source_array = np.asarray(source, dtype=np.uint8)
    mask = np.asarray(mask_image) > 0
    h, w = mask.shape
    pad_h = (-h) % 8
    pad_w = (-w) % 8
    image_padded = np.pad(source_array, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    mask_padded = np.pad(mask.astype("float32"), ((0, pad_h), (0, pad_w)), mode="constant")
    image_tensor = torch.from_numpy(np.transpose(image_padded.astype("float32") / 255.0, (2, 0, 1))[None])
    mask_tensor = torch.from_numpy(mask_padded[None, None])

    start = perf_counter()
    with torch.no_grad():
        output = model(image_tensor, mask_tensor)
    print(label, "inference_seconds", round(perf_counter() - start, 2))
    generated = np.clip(np.transpose(output[0].numpy(), (1, 2, 0))[:h, :w] * 255.0, 0, 255).astype("uint8")
    result = source_array.copy()
    result[mask] = generated[mask]
    Image.fromarray(result).save(HERE / f"local_anime_lama_{label}.png")
    mask_image.save(HERE / f"local_anime_lama_{label}_mask.png")
    print(label, "masked_pixels", int(mask.sum()), "outside_mask_changed", int(np.any(result[~mask] != source_array[~mask])))
