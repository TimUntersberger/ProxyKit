"""Research probe: local diffusion candidates confined to an existing repair mask."""
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

here = Path(__file__).parent
run = here / 'runs/batch_OP13-012_SP_eb9b91ee'
base = Image.open(run / 'repaired_palette_1x.png').convert('RGB')
source = Image.open(r'C:\Users\timun\Downloads\batch_OP13-012_SP.png').convert('RGB')
mask = Image.open(run / 'repaired_1x_mask.png').convert('L')
mask_array = np.asarray(mask) > 0
ys, xs = np.nonzero(mask_array)
side = 512
cx = (int(xs.min()) + int(xs.max())) // 2
cy = (int(ys.min()) + int(ys.max())) // 2
left = max(0, min(cx - side // 2, base.width - side))
top = max(0, min(cy - side // 2, base.height - side))
box = (left, top, left + side, top + side)
crop = base.crop(box)
mask_crop = mask.crop(box)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    str(here / 'models/sd15-inpainting'), torch_dtype=torch.float16,
    variant='fp16', use_safetensors=True, safety_checker=None,
    feature_extractor=None, requires_safety_checker=False).to('cuda')
pipe.set_progress_bar_config(disable=True)
preserved = pipe.image_processor.preprocess(crop).to('cuda', dtype=torch.float16)
prompt = ('faithful detailed trading card illustration, elegant woman in pink dress, '
          'intricate blue and gold floral ornament, continuous thin gold scrollwork, '
          'preserve original composition and line positions')
negative = 'text, letters, watermark, blurry lines, distorted hands, extra objects'

for strength, seed in ((.35, 101), (.55, 101), (.55, 202)):
    start = time.perf_counter()
    with torch.inference_mode():
        generated = pipe(
            prompt=prompt, negative_prompt=negative, image=crop,
            mask_image=mask_crop, masked_image_latents=preserved,
            guidance_scale=4.0, num_inference_steps=30, strength=strength,
            generator=torch.Generator(device='cuda').manual_seed(seed)).images[0]
    result = np.asarray(base).copy()
    region = result[top:top + side, left:left + side]
    selection = np.asarray(mask_crop) > 0
    region[selection] = np.asarray(generated)[selection]
    result[~mask_array] = np.asarray(source)[~mask_array]
    output = here / f'vivi_local_inpaint_{strength:.2f}_{seed}.png'
    Image.fromarray(result).save(output)
    changed_outside = np.count_nonzero(np.any(result != np.asarray(source), axis=2) & ~mask_array)
    print(output.name, 'seconds', round(time.perf_counter() - start, 2),
          'outside_mask_changes', changed_outside, flush=True)
