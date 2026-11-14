"""Run official SLBR weights on the provided card only."""
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import cv2
import numpy as np
from PIL import Image
import torch
import skimage.measure
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
skimage.measure.compare_psnr=peak_signal_noise_ratio
skimage.measure.compare_ssim=structural_similarity

here=Path(__file__).parent
sys.path.insert(0,str(here/'models/slbr_src'))
from src.networks.resunet import SLBR

args=SimpleNamespace(bg_mode='res_mask',mask_mode='res',sim_metric='cos',
                     k_center=2,project_mode='simple',use_refine=True,
                     k_refine=3,k_skip_stage=3)
model=SLBR(args,shared_depth=1,blocks=3,long_skip=True)
checkpoint=torch.load(here/'models/slbr_model_best.pth.tar',map_location='cpu',weights_only=True)
print('checkpoint',checkpoint['nets'],checkpoint['epoch'],flush=True)
model.load_state_dict(checkpoint['state_dict'],strict=True)
model=model.eval().cuda()

source=np.asarray(Image.open(r'C:\Users\timun\Downloads\OP18-085.webp').convert('RGB'))
box=(44,185,556,697)
crop=source[box[1]:box[3],box[0]:box[2]]
for resolution in (256,512):
    small=cv2.resize(crop,(resolution,resolution),interpolation=cv2.INTER_AREA)
    tensor=torch.from_numpy(small.transpose(2,0,1).copy()).float()[None].cuda()/255
    begin=time.perf_counter()
    with torch.inference_mode():
        images,masks,_=model(tensor)
    print(resolution,'seconds',round(time.perf_counter()-begin,2),'mask',tuple(m.shape for m in masks),flush=True)
    pred=images[0][0].detach().cpu().numpy().transpose(1,2,0)
    mask=masks[0][0,0].detach().cpu().numpy()
    pred=cv2.resize(pred,(512,512),interpolation=cv2.INTER_CUBIC)
    mask=cv2.resize(mask,(512,512),interpolation=cv2.INTER_LINEAR)
    out=source.copy()
    region=out[box[1]:box[3],box[0]:box[2]].astype(np.float32)/255
    fixed=pred*mask[:,:,None]+region*(1-mask[:,:,None])
    out[box[1]:box[3],box[0]:box[2]]=np.clip(fixed*255,0,255).astype('uint8')
    Image.fromarray(out).save(here/f'slbr_{resolution}_composite.png')
    Image.fromarray(np.clip(mask*255,0,255).astype('uint8')).save(here/f'slbr_{resolution}_mask_crop.png')
    out=source.copy();out[box[1]:box[3],box[0]:box[2]]=np.clip(pred*255,0,255).astype('uint8')
    Image.fromarray(out).save(here/f'slbr_{resolution}_raw.png')
