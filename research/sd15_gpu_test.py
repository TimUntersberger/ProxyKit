"""Local SD1.5 inpainting on a card crop; save only masked pixels back."""
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageFilter
import torch
from diffusers import StableDiffusionInpaintPipeline

here = Path(__file__).parent
source = Image.open(r"C:\Users\timun\Downloads\OP18-085.webp").convert("RGB")
dewhite = Image.open(here / "opacity_adaptive_0.72.png").convert("RGB")
letter = Image.open(here / "shared_edge_binary.png").convert("L")
shifted = Image.new("L", source.size, 0)
shifted.paste(letter, (0, 3))
full = shifted.filter(ImageFilter.MaxFilter(7)).point(lambda v: 255 if v else 0)
edge = Image.open(here / "edge_guided_mask_2.png").convert("L")
box = (44, 185, 556, 697)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    str(here / "models" / "sd15-inpainting"),
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False,
).to("cuda")
pipe.set_progress_bar_config(disable=True)
print("GPU model ready", round(torch.cuda.max_memory_reserved() / 2**30, 2), "GiB", flush=True)

prompt = "detailed anime trading card illustration of a three headed crimson and purple hellhound, snarling face, sharp white teeth, red fur, dark blue magical smoke, matching surrounding art"
negative = "text, letters, logo, watermark, typography, signature, blurry, extra face, extra mouth"
for name, base, mask, seed, strength in [
    ("full_original_101", source, full, 101, .99),
    ("full_dewhite_202", dewhite, full, 202, .99),
    ("edge_dewhite_303", dewhite, edge, 303, .99),
    ("full_original_404", source, full, 404, .99),
    ("full_original_505", source, full, 505, .85),
]:
    if (here / f"sd15_{name}.png").exists():
        continue
    begin = time.perf_counter()
    generated = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=base.crop(box),
        mask_image=mask.crop(box),
        guidance_scale=7.5,
        num_inference_steps=25,
        strength=strength,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0]
    result = np.asarray(base).copy()
    piece = np.asarray(generated)
    m = np.asarray(mask.crop(box)) > 0
    region = result[box[1]:box[3], box[0]:box[2]]
    region[m] = piece[m]
    Image.fromarray(result).save(here / f"sd15_{name}.png")
    print(name, "seconds", round(time.perf_counter()-begin, 1), "mask", int(m.sum()), "vram_gib", round(torch.cuda.max_memory_reserved()/2**30,2), flush=True)
