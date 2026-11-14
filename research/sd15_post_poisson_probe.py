"""Low-strength local diffusion experiment on a source-only repair.

Research probe only: generated pixels are saved solely inside the supplied
repair mask, and the prompt contains no image-specific anatomy labels.
"""
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

here=Path(__file__).parent
base=Image.open(here/'one_image_result_texture.png').convert('RGB')
mask=Image.open(here/'one_image_result_texture_mask.png').convert('L')
mask_array=np.asarray(mask)>0
ys,xs=np.nonzero(mask_array)
side=512
cx=(int(xs.min())+int(xs.max()))//2
cy=(int(ys.min())+int(ys.max()))//2
left=max(0,min(cx-side//2,base.width-side))
top=max(0,min(cy-side//2,base.height-side))
box=(left,top,left+side,top+side)
crop=base.crop(box)
mask_crop=mask.crop(box)

pipe=StableDiffusionInpaintPipeline.from_pretrained(
    str(here/'models/sd15-inpainting'),torch_dtype=torch.float16,
    variant='fp16',use_safetensors=True,safety_checker=None,
    feature_extractor=None,requires_safety_checker=False).to('cuda')
pipe.set_progress_bar_config(disable=True)
preserved=pipe.image_processor.preprocess(crop).to('cuda',dtype=torch.float16)
prompt='faithful detailed illustration, preserve the existing composition, shapes, colors, and fine lines'
negative='text, letters, watermark, extra objects, altered anatomy, blurry'
for strength in (.15,.25,.35):
    start=time.perf_counter()
    generated=pipe(prompt=prompt,negative_prompt=negative,image=crop,
                   mask_image=mask_crop,masked_image_latents=preserved,
                   guidance_scale=3.0,num_inference_steps=30,strength=strength,
                   generator=torch.Generator(device='cuda').manual_seed(101)).images[0]
    result=np.asarray(base).copy()
    region=result[box[1]:box[3],box[0]:box[2]]
    selection=np.asarray(mask_crop)>0
    region[selection]=np.asarray(generated)[selection]
    path=here/f'OP18-085_sd15_post_poisson_{strength:.2f}.png'
    Image.fromarray(result).save(path)
    print(strength,'seconds',round(time.perf_counter()-start,2),'output',path.name,flush=True)
