"""Local BrushNet comparison on Vivi using the authors' released checkpoint."""
from pathlib import Path
import sys
import time
from argparse import ArgumentParser

import cv2
import huggingface_hub
import numpy as np
from PIL import Image
import torch

parser = ArgumentParser()
parser.add_argument('--variant', choices=['segmentation', 'random'], default='segmentation')
args = parser.parse_args()

# BrushNet's bundled Diffusers 0.27 imports the old hub function. The model
# and code are loaded from local paths below, so this import shim is unused for downloads.
huggingface_hub.cached_download = huggingface_hub.hf_hub_download
here = Path(__file__).parent
sys.path.insert(0, str(here / 'brushnet_src/src'))
from diffusers import BrushNetModel, StableDiffusionBrushNetPipeline, UniPCMultistepScheduler

run = here / 'runs/batch_OP13-012_SP_eb9b91ee'
base = np.asarray(Image.open(run / 'repaired_palette_1x.png').convert('RGB'))
glyph = np.asarray(Image.open(run / 'repaired_1x_mask.png').convert('L')) > 0
box = (0, 190, 640, 702)
crop = base[box[1]:box[3], box[0]:box[2]]
brushnet_path = (here / 'models/brushnet_random/random_mask_brushnet_ckpt'
                 if args.variant == 'random' else
                 here / 'models/brushnet_official/data/ckpt/segmentation_mask_brushnet_ckpt')
brushnet = BrushNetModel.from_pretrained(
    brushnet_path, torch_dtype=torch.float16, local_files_only=True)
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    here / 'models/sd15-base', brushnet=brushnet, torch_dtype=torch.float16,
    variant='fp16', local_files_only=True, low_cpu_mem_usage=False,
    safety_checker=None, feature_extractor=None, requires_safety_checker=False)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(disable=True)
prompt = ('faithful flat-color illustrated trading card, precise continuous line art, '
          'intricate ornament, preserve existing figure, composition, and colors')
negative = 'text, letters, watermark, blurry lines, broken ornament, altered anatomy, new objects'

cases = (('outline10', 10),) if args.variant == 'random' else (('outline10', 10), ('filled5', 5))
for label, radius in cases:
    start = time.perf_counter()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * radius + 1, 2 * radius + 1))
    outer = cv2.dilate(glyph.astype(np.uint8), kernel) > 0
    inner = cv2.erode(glyph.astype(np.uint8), kernel) > 0
    selection = outer & ~inner if label.startswith('outline') else outer
    local_mask = selection[box[1]:box[3], box[0]:box[2]]
    conditioning = crop.copy()
    conditioning[local_mask] = 0
    mask_rgb = np.repeat((local_mask.astype(np.uint8) * 255)[:, :, None], 3, axis=2)
    with torch.inference_mode():
        generated = pipe(
            prompt=prompt, negative_prompt=negative,
            image=Image.fromarray(conditioning), mask=Image.fromarray(mask_rgb),
            width=640, height=512, guidance_scale=5,
            num_inference_steps=30, brushnet_conditioning_scale=1.0,
            generator=torch.Generator(device='cuda').manual_seed(101)).images[0]
    candidate = base.copy()
    region = candidate[box[1]:box[3], box[0]:box[2]]
    region[local_mask] = np.asarray(generated)[local_mask]
    distance = cv2.distanceTransform(selection.astype(np.uint8), cv2.DIST_L2, 3)
    weight = .8 * np.clip(distance / 5, 0, 1) * selection
    blended = np.uint8(np.clip(base.astype(np.float32) * (1 - weight[..., None]) +
                               candidate.astype(np.float32) * weight[..., None], 0, 255))
    blended[~selection] = base[~selection]
    path = here / f'vivi_brushnet_{args.variant}_{label}_1x.png'
    Image.fromarray(blended).save(path)
    print(path.name, 'seconds', round(time.perf_counter() - start, 2),
          'outside_mask', int(np.count_nonzero(np.any(blended != base, axis=2) & ~selection)),
          'peak_reserved_GiB', round(torch.cuda.max_memory_reserved() / 2**30, 2), flush=True)
