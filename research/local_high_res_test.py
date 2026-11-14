"""Check whether a higher-resolution copy improves masked local inpainting."""

from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from PIL import Image, ImageFilter

HERE = Path(__file__).parent
source = Image.open(HERE / "samples" / "OP18-085 Cerberus.jpg").convert("RGB")
mask = Image.open(HERE / "approx_mask.png").convert("L").resize(source.size, Image.Resampling.BICUBIC)
mask = mask.point(lambda x: 255 if x > 12 else 0).filter(ImageFilter.MaxFilter(13))
source_arr = np.asarray(source, dtype=np.uint8)
mask_arr = np.asarray(mask) > 0
h,w = mask_arr.shape
pad_h,pad_w = (-h)%8,(-w)%8
rgb_padded = np.pad(source_arr, ((0,pad_h),(0,pad_w),(0,0)), mode="reflect")
mask_padded = np.pad(mask_arr.astype("float32"), ((0,pad_h),(0,pad_w)), mode="constant")
torch.set_num_threads(4)
model = torch.jit.load(str(HERE / "models" / "anime-manga-big-lama.pt"),map_location="cpu").eval()
start = perf_counter()
with torch.no_grad():
    result = model(torch.from_numpy(np.transpose(rgb_padded.astype("float32")/255,(2,0,1))[None]),torch.from_numpy(mask_padded[None,None]))
print('inference_seconds',round(perf_counter()-start,2))
generated = np.clip(np.transpose(result[0].numpy(),(1,2,0))[:h,:w]*255,0,255).astype('uint8')
composite = source_arr.copy()
composite[mask_arr] = generated[mask_arr]
out = Image.fromarray(composite)
out.save(HERE / 'local_anime_lama_highres.png')
out.resize((600,838),Image.Resampling.LANCZOS).save(HERE / 'local_anime_lama_highres_downscaled.png')
print('outside_changed',int(np.any(composite[~mask_arr]!=source_arr[~mask_arr])))
