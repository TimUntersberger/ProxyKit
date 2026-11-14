"""Test SDXL near its training resolution on the same 10-pixel outline ring."""
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionXLInpaintPipeline

here = Path(__file__).parent
run = here / 'runs/batch_OP13-012_SP_eb9b91ee'
base = np.asarray(Image.open(run / 'repaired_palette_1x.png').convert('RGB'))
glyph = np.asarray(Image.open(run / 'repaired_1x_mask.png').convert('L')) > 0
box = (0, 190, 640, 702)
crop = Image.fromarray(base).crop(box)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
outer = cv2.dilate(glyph.astype(np.uint8), kernel) > 0
inner = cv2.erode(glyph.astype(np.uint8), kernel) > 0
selection = outer & ~inner
mask_crop = Image.fromarray((selection.astype(np.uint8) * 255)).crop(box)
large_crop = crop.resize((960, 768), Image.Resampling.LANCZOS)
large_mask = mask_crop.resize((960, 768), Image.Resampling.NEAREST)

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    here / 'models/sdxl-inpainting', torch_dtype=torch.float16,
    variant='fp16', use_safetensors=True, local_files_only=True)
pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(disable=True)
prompt = ('faithful flat-color illustrated trading card, precise continuous line art, '
          'intricate ornament, preserve existing figure, composition, and colors')
negative = 'text, letters, watermark, blurry lines, broken ornament, altered anatomy, new objects'
start = time.perf_counter()
with torch.inference_mode():
    generated = pipe(prompt=prompt, negative_prompt=negative, image=large_crop,
                     mask_image=large_mask, width=960, height=768,
                     guidance_scale=5, num_inference_steps=28, strength=.55,
                     generator=torch.Generator(device='cuda').manual_seed(101)).images[0]
generated.save(here / 'vivi_sdxl_outline10_960x768_generated.png')
down = generated.resize((640, 512), Image.Resampling.LANCZOS)
candidate = base.copy()
region = candidate[box[1]:box[3], box[0]:box[2]]
local = np.asarray(mask_crop) > 0
region[local] = np.asarray(down)[local]
distance = cv2.distanceTransform(selection.astype(np.uint8), cv2.DIST_L2, 3)
weight = .8 * np.clip(distance / 5, 0, 1) * selection
blended = np.uint8(np.clip(base.astype(np.float32) * (1 - weight[..., None]) +
                           candidate.astype(np.float32) * weight[..., None], 0, 255))
blended[~selection] = base[~selection]
path = here / 'vivi_sdxl_outline10_highres_1x.png'
Image.fromarray(blended).save(path)
print(path.name, 'seconds', round(time.perf_counter() - start, 2),
      'outside_mask', int(np.count_nonzero(np.any(blended != base, axis=2) & ~selection)),
      'peak_reserved_GiB', round(torch.cuda.max_memory_reserved() / 2**30, 2), flush=True)
