"""Research probe: locally inpaint a user-selected ornament region only."""
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
import cv2
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--border', type=int, default=0, help='extra pixels around the existing repair mask')
parser.add_argument('--strength', type=float, default=.55)
parser.add_argument('--seed', type=int, default=101)
args = parser.parse_args()
if not 0 <= args.border <= 16:
    parser.error('--border must be from 0 to 16')

here = Path(__file__).parent
run = here / 'runs/batch_OP13-012_SP_eb9b91ee'
base = Image.open(run / 'repaired_palette_1x.png').convert('RGB')
source = Image.open(r'C:\Users\timun\Downloads\batch_OP13-012_SP.png').convert('RGB')
full_mask = np.asarray(Image.open(run / 'repaired_1x_mask.png').convert('L')) > 0
# The ROI is an optional user selection, not an anatomy coordinate in the method.
roi = (0, 435, 230, 520)
mask = np.zeros_like(full_mask)
if args.border:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * args.border + 1, 2 * args.border + 1))
    expanded = cv2.dilate(full_mask.astype(np.uint8), kernel) > 0
else:
    expanded = full_mask
mask[roi[1]:roi[3], roi[0]:roi[2]] = expanded[roi[1]:roi[3], roi[0]:roi[2]]
crop_box = (0, 250, 512, 762)
crop = base.crop(crop_box)
mask_crop = Image.fromarray((mask.astype(np.uint8) * 255)).crop(crop_box)
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    str(here / 'models/sd15-inpainting'), torch_dtype=torch.float16,
    variant='fp16', use_safetensors=True, safety_checker=None,
    feature_extractor=None, requires_safety_checker=False).to('cuda')
pipe.set_progress_bar_config(disable=True)
preserved = pipe.image_processor.preprocess(crop).to('cuda', dtype=torch.float16)
prompt = ('faithful flat-color trading card illustration, precise symmetrical blue and gold '
          'filigree ornament with continuous curled gold lines, preserve existing geometry')
negative = 'text, letters, watermark, blurry lines, broken scrolls, new objects'
for strength, seed in ((args.strength, args.seed),):
    start = time.perf_counter()
    with torch.inference_mode():
        generated = pipe(prompt=prompt, negative_prompt=negative, image=crop,
                         mask_image=mask_crop, masked_image_latents=preserved,
                         guidance_scale=4, num_inference_steps=30, strength=strength,
                         generator=torch.Generator(device='cuda').manual_seed(seed)).images[0]
    result = np.asarray(base).copy()
    region = result[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
    selection = np.asarray(mask_crop) > 0
    region[selection] = np.asarray(generated)[selection]
    result[~expanded] = np.asarray(source)[~expanded]
    path = here / f'vivi_local_scroll_border{args.border}_{strength:.2f}_{seed}.png'
    Image.fromarray(result).save(path)
    print(path.name, round(time.perf_counter() - start, 2),
          'outside_mask', int(np.count_nonzero(np.any(result != np.asarray(source), axis=2) & ~expanded)),
          flush=True)
