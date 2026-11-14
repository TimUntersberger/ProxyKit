"""Test whether BrushNet repeats SAMPLE because the model mask has letter shapes."""
from pathlib import Path
import sys
import time

import cv2
import huggingface_hub
import numpy as np
from PIL import Image
import torch

huggingface_hub.cached_download = huggingface_hub.hf_hub_download
here = Path(__file__).parent
sys.path.insert(0, str(here / 'brushnet_src/src'))
from diffusers import BrushNetModel, StableDiffusionBrushNetPipeline, UniPCMultistepScheduler

run = here / 'runs/batch_OP13-012_SP_eb9b91ee'
base = np.asarray(Image.open(run / 'repaired_palette_1x.png').convert('RGB'))
glyph = np.asarray(Image.open(run / 'repaired_1x_mask.png').convert('L')) > 0
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
paste_area = cv2.dilate(glyph.astype(np.uint8), kernel) > 0
box = (0, 190, 640, 702)
crop = base[box[1]:box[3], box[0]:box[2]]

brushnet = BrushNetModel.from_pretrained(
    here / 'models/brushnet_random/random_mask_brushnet_ckpt',
    torch_dtype=torch.float16, local_files_only=True)
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

for label, region_box in (('scroll', (0, 435, 230, 520)),
                          ('full_band', (50, 368, 590, 525))):
    start = time.perf_counter()
    model_mask = np.zeros(glyph.shape, np.uint8)
    x0, y0, x1, y1 = region_box
    model_mask[y0:y1, x0:x1] = 1
    local_mask = model_mask[box[1]:box[3], box[0]:box[2]] > 0
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
    paste = paste_area & (model_mask > 0)
    region = candidate[box[1]:box[3], box[0]:box[2]]
    local_paste = paste[box[1]:box[3], box[0]:box[2]]
    region[local_paste] = np.asarray(generated)[local_paste]
    distance = cv2.distanceTransform(paste.astype(np.uint8), cv2.DIST_L2, 3)
    weight = .8 * np.clip(distance / 5, 0, 1) * paste
    blended = np.uint8(np.clip(base.astype(np.float32) * (1 - weight[..., None]) +
                               candidate.astype(np.float32) * weight[..., None], 0, 255))
    blended[~paste] = base[~paste]
    path = here / f'vivi_brushnet_rect_{label}_1x.png'
    Image.fromarray(blended).save(path)
    print(path.name, 'seconds', round(time.perf_counter() - start, 2),
          'outside_paste', int(np.count_nonzero(np.any(blended != base, axis=2) & ~paste)), flush=True)
