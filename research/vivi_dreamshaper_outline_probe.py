"""Compare illustration-tuned DreamShaper inpainting on Vivi's outline mask."""
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

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

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    here / 'models/dreamshaper-7-inpainting', torch_dtype=torch.float16,
    variant='fp16', use_safetensors=True, local_files_only=True,
    safety_checker=None, feature_extractor=None, requires_safety_checker=False).to('cuda')
pipe.set_progress_bar_config(disable=True)
prompt = ('faithful flat-color illustrated trading card, precise continuous line art, '
          'intricate ornament, preserve existing figure, composition, and colors')
negative = 'text, letters, watermark, blurry lines, broken ornament, altered anatomy, new objects'

for strength in (.55, .75):
    start = time.perf_counter()
    with torch.inference_mode():
        generated = pipe(prompt=prompt, negative_prompt=negative, image=crop,
                         mask_image=mask_crop, width=640, height=512,
                         guidance_scale=4, num_inference_steps=30, strength=strength,
                         generator=torch.Generator(device='cuda').manual_seed(101)).images[0]
    candidate = base.copy()
    region = candidate[box[1]:box[3], box[0]:box[2]]
    local = np.asarray(mask_crop) > 0
    region[local] = np.asarray(generated)[local]
    distance = cv2.distanceTransform(selection.astype(np.uint8), cv2.DIST_L2, 3)
    weight = .8 * np.clip(distance / 5, 0, 1) * selection
    blended = np.uint8(np.clip(base.astype(np.float32) * (1 - weight[..., None]) +
                               candidate.astype(np.float32) * weight[..., None], 0, 255))
    blended[~selection] = base[~selection]
    path = here / f'vivi_dreamshaper_outline10_{strength:.2f}_1x.png'
    Image.fromarray(blended).save(path)
    print(path.name, 'seconds', round(time.perf_counter() - start, 2),
          'outside_mask', int(np.count_nonzero(np.any(blended != base, axis=2) & ~selection)),
          'peak_reserved_GiB', round(torch.cuda.max_memory_reserved() / 2**30, 2), flush=True)
