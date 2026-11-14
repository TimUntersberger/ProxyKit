"""Local SD1.5 experiment covering the full SAMPLE width with a 5-pixel border."""
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
mask = np.asarray(Image.open(run / 'repaired_1x_mask.png').convert('L')) > 0
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
expanded = cv2.dilate(mask.astype(np.uint8), kernel) > 0
box = (0, 190, 640, 702)
crop = Image.fromarray(base).crop(box)
mask_crop = Image.fromarray((expanded.astype(np.uint8) * 255)).crop(box)
crop.save(here / 'vivi_sd15_full_width_input_640x512.png')
mask_crop.save(here / 'vivi_sd15_full_width_border5_mask_640x512.png')

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    str(here / 'models/sd15-inpainting'), torch_dtype=torch.float16,
    variant='fp16', use_safetensors=True, safety_checker=None,
    feature_extractor=None, requires_safety_checker=False).to('cuda')
pipe.set_progress_bar_config(disable=True)
preserved = pipe.image_processor.preprocess(crop).to('cuda', dtype=torch.float16)
prompt = ('faithful flat-color illustrated trading card, precise continuous line art, '
          'intricate ornament, preserve existing figure, composition, and colors')
negative = 'text, letters, watermark, blurry lines, broken ornament, altered anatomy, new objects'

for strength in (.45, .55, .70):
    start = time.perf_counter()
    with torch.inference_mode():
        generated = pipe(prompt=prompt, negative_prompt=negative, image=crop,
                         mask_image=mask_crop, masked_image_latents=preserved,
                         width=640, height=512,
                         guidance_scale=4, num_inference_steps=30, strength=strength,
                         generator=torch.Generator(device='cuda').manual_seed(101)).images[0]
    candidate = base.copy()
    region = candidate[box[1]:box[3], box[0]:box[2]]
    selection = np.asarray(mask_crop) > 0
    region[selection] = np.asarray(generated)[selection]
    candidate[~expanded] = source[~expanded]
    Image.fromarray(candidate).save(here / f'vivi_full_width_border5_raw_{strength:.2f}.png')
    distance = cv2.distanceTransform(expanded.astype(np.uint8), cv2.DIST_L2, 3)
    weight = .8 * np.clip(distance / 5, 0, 1) * expanded
    blended = np.uint8(np.clip(base.astype(np.float32) * (1 - weight[..., None]) +
                               candidate.astype(np.float32) * weight[..., None], 0, 255))
    blended[~expanded] = source[~expanded]
    path = here / f'vivi_full_width_border5_blended_{strength:.2f}.png'
    Image.fromarray(blended).save(path)
    print(path.name, 'seconds', round(time.perf_counter() - start, 2),
          'outside_mask', int(np.count_nonzero(np.any(blended != source, axis=2) & ~expanded)),
          flush=True)
