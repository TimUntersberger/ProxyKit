"""Compare local inpainting border widths with fixed crop, prompt, seed and strength."""
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
source = np.asarray(Image.open(r'C:\Users\timun\Downloads\batch_OP13-012_SP.png').convert('RGB'))
full_mask = np.asarray(Image.open(run / 'repaired_1x_mask.png').convert('L')) > 0
roi = np.zeros_like(full_mask)
roi[435:520, 0:230] = True  # Same user-selected scroll band for every candidate.
box = (0, 250, 512, 762)
crop = Image.fromarray(base).crop(box)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    str(here / 'models/sd15-inpainting'), torch_dtype=torch.float16,
    variant='fp16', use_safetensors=True, safety_checker=None,
    feature_extractor=None, requires_safety_checker=False).to('cuda')
pipe.set_progress_bar_config(disable=True)
preserved = pipe.image_processor.preprocess(crop).to('cuda', dtype=torch.float16)
prompt = ('faithful flat-color trading card illustration, precise symmetrical blue and gold '
          'filigree ornament with continuous curled gold lines, preserve existing geometry')
negative = 'text, letters, watermark, blurry lines, broken scrolls, new objects'

for border in (1, 2, 4, 5, 8, 10):
    start = time.perf_counter()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border + 1, 2 * border + 1))
    expanded = cv2.dilate(full_mask.astype(np.uint8), kernel) > 0
    selection = expanded & roi
    mask_image = Image.fromarray((selection.astype(np.uint8) * 255)).crop(box)
    with torch.inference_mode():
        generated = pipe(prompt=prompt, negative_prompt=negative, image=crop,
                         mask_image=mask_image, masked_image_latents=preserved,
                         guidance_scale=4, num_inference_steps=30, strength=.55,
                         generator=torch.Generator(device='cuda').manual_seed(101)).images[0]
    raw = base.copy()
    region = raw[box[1]:box[3], box[0]:box[2]]
    local_select = np.asarray(mask_image) > 0
    region[local_select] = np.asarray(generated)[local_select]
    raw[~expanded] = source[~expanded]
    Image.fromarray(raw).save(here / f'vivi_local_scroll_border{border}_0.55_101.png')

    distance = cv2.distanceTransform(selection.astype(np.uint8), cv2.DIST_L2, 3)
    weight = .8 * np.clip(distance / 5, 0, 1) * selection
    feathered = np.uint8(np.clip(base.astype(np.float32) * (1 - weight[..., None]) +
                                  raw.astype(np.float32) * weight[..., None], 0, 255))
    feathered[~expanded] = source[~expanded]
    Image.fromarray(feathered).save(here / f'vivi_scroll_border{border}_feathered_1x.png')
    print('border', border, 'selection', int(selection.sum()),
          'outside_expanded', int(np.count_nonzero(np.any(feathered != source, axis=2) & ~expanded)),
          'seconds', round(time.perf_counter() - start, 2), flush=True)
