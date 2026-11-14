"""Refine the de-whitened source while passing its detail into masked conditioning."""
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageFilter
import torch
from diffusers import StableDiffusionInpaintPipeline

here=Path(__file__).parent
base=Image.open(here/'opacity_adaptive_0.72.png').convert('RGB')
letter=Image.open(here/'shared_edge_binary.png').convert('L')
mask=Image.new('L',base.size,0);mask.paste(letter,(0,3))
mask=mask.filter(ImageFilter.MaxFilter(7)).point(lambda v:255 if v else 0)
box=(44,185,556,697)
crop=base.crop(box)
mask_crop=mask.crop(box)
pipe=StableDiffusionInpaintPipeline.from_pretrained(
    str(here/'models/sd15-inpainting'),torch_dtype=torch.float16,
    variant='fp16',use_safetensors=True,safety_checker=None,
    feature_extractor=None,requires_safety_checker=False).to('cuda')
pipe.set_progress_bar_config(disable=True)
preserved=pipe.image_processor.preprocess(crop).to('cuda',dtype=torch.float16)
prompt='detailed anime trading card illustration of a three headed crimson and purple hellhound, sharp white teeth, red fur, dark blue magical smoke, matching surrounding art'
negative='text, letters, logo, watermark, typography, extra face, extra mouth, blurry'
for strength in (.25,.45,.65):
    begin=time.perf_counter()
    gen=pipe(prompt=prompt,negative_prompt=negative,image=crop,mask_image=mask_crop,
             masked_image_latents=preserved,guidance_scale=5.0,
             num_inference_steps=30,strength=strength,
             generator=torch.Generator(device='cuda').manual_seed(101)).images[0]
    result=np.asarray(base).copy(); region=result[box[1]:box[3],box[0]:box[2]]
    m=np.asarray(mask_crop)>0; region[m]=np.asarray(gen)[m]
    Image.fromarray(result).save(here/f'sd15_conditioned_{strength:.2f}.png')
    print(strength,round(time.perf_counter()-begin,2),flush=True)
