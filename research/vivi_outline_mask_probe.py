"""Compare outline-only and filled glyph masks with proper masked conditioning."""
from pathlib import Path
import time
from argparse import ArgumentParser

import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

parser = ArgumentParser()
parser.add_argument('--outline-width', type=int,
                    help='test only this outline width in original-image pixels')
args = parser.parse_args()
if args.outline_width is not None and not 1 <= args.outline_width <= 24:
    parser.error('--outline-width must be between 1 and 24')

here = Path(__file__).parent
run = here / 'runs/batch_OP13-012_SP_eb9b91ee'
base = np.asarray(Image.open(run / 'repaired_palette_1x.png').convert('RGB'))
source = np.asarray(Image.open(r'C:\Users\timun\Downloads\batch_OP13-012_SP.png').convert('RGB'))
glyph = np.asarray(Image.open(run / 'repaired_1x_mask.png').convert('L')) > 0
box = (0, 190, 640, 702)
crop = Image.fromarray(base).crop(box)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    str(here / 'models/sd15-inpainting'), torch_dtype=torch.float16,
    variant='fp16', use_safetensors=True, safety_checker=None,
    feature_extractor=None, requires_safety_checker=False).to('cuda')
pipe.set_progress_bar_config(disable=True)
prompt = ('faithful flat-color illustrated trading card, precise continuous line art, '
          'intricate ornament, preserve existing figure, composition, and colors')
negative = 'text, letters, watermark, blurry lines, broken ornament, altered anatomy, new objects'

variants = [(f'outline{args.outline_width}', args.outline_width)] if args.outline_width else [
    ('outline3', 3), ('outline6', 6), ('filled5', 5)]
for label, radius in variants:
    start = time.perf_counter()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * radius + 1, 2 * radius + 1))
    outer = cv2.dilate(glyph.astype(np.uint8), kernel) > 0
    if label.startswith('outline'):
        inner = cv2.erode(glyph.astype(np.uint8), kernel) > 0
        selection = outer & ~inner
    else:
        selection = outer
    mask_crop = Image.fromarray((selection.astype(np.uint8) * 255)).crop(box)
    mask_crop.save(here / f'vivi_sd15_{label}_mask_640x512.png')
    with torch.inference_mode():
        generated = pipe(prompt=prompt, negative_prompt=negative, image=crop,
                         mask_image=mask_crop, width=640, height=512,
                         guidance_scale=4, num_inference_steps=30, strength=.55,
                         generator=torch.Generator(device='cuda').manual_seed(101)).images[0]
    candidate = base.copy()
    region = candidate[box[1]:box[3], box[0]:box[2]]
    local_selection = np.asarray(mask_crop) > 0
    region[local_selection] = np.asarray(generated)[local_selection]
    candidate[~selection] = base[~selection]
    Image.fromarray(candidate).save(here / f'vivi_{label}_proper_raw.png')
    distance = cv2.distanceTransform(selection.astype(np.uint8), cv2.DIST_L2, 3)
    weight = .8 * np.clip(distance / 5, 0, 1) * selection
    blended = np.uint8(np.clip(base.astype(np.float32) * (1 - weight[..., None]) +
                               candidate.astype(np.float32) * weight[..., None], 0, 255))
    blended[~selection] = base[~selection]
    path = here / f'vivi_{label}_proper_blended_1x.png'
    Image.fromarray(blended).save(path)
    print(label, 'mask_pixels', int(selection.sum()), 'seconds', round(time.perf_counter() - start, 2),
          'outside_selection_vs_base', int(np.count_nonzero(np.any(blended != base, axis=2) & ~selection)),
          flush=True)
