"""Estimate varying white-overlay strength from an interpolated color baseline."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "deps"))
import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
source = np.asarray(Image.open(Path(r"C:\Users\timun\Downloads\OP18-085.webp")).convert("RGB"), dtype=np.float32)
shared = np.asarray(Image.open(HERE / "shared_edge_binary.png").convert("L")) > 0
mask = cv2.warpAffine(shared.astype("uint8"),np.float32([[1,0,0],[0,1,3]]),(600,838),flags=cv2.INTER_NEAREST)
mask = cv2.dilate(mask,np.ones((3,3),"uint8"))
baseline = cv2.inpaint(source.astype("uint8")[:,:,::-1].copy(),mask*255,10,cv2.INPAINT_TELEA)[:,:,::-1].astype("float32")
baseline = cv2.GaussianBlur(baseline,(0,0),5)

ratio = (source-baseline)/np.maximum(255-baseline,10)
ratio = np.where(baseline<225,ratio,np.nan)
local_alpha = np.nanmedian(ratio,axis=2)
local_alpha = np.nan_to_num(local_alpha,nan=.65)
local_alpha = np.clip(local_alpha,.45,.78)
local_alpha = cv2.GaussianBlur(local_alpha,(0,0),5)
Image.fromarray(np.clip(local_alpha*mask*255,0,255).astype("uint8")).save(HERE / "local_estimated_alpha.png")

soft_mask = cv2.GaussianBlur(mask.astype("float32"),(0,0),.8)
cap = np.clip((source.min(axis=2)-8)/255,0,.9)
for strength in (.5,1.0):
    proposed = .65 + strength*(local_alpha-.65)
    alpha = np.minimum(proposed*soft_mask,cap)
    result = np.clip((source-alpha[:,:,None]*255)/(1-alpha[:,:,None]),0,255).astype("uint8")
    Image.fromarray(result).save(HERE / f"local_alpha_inverse_{strength:.1f}.png")
    border = np.asarray(Image.open(HERE / "edge_guided_mask_1.png").convert("L"))
    inpainted = cv2.inpaint(result[:,:,::-1].copy(),border,2,cv2.INPAINT_TELEA)[:,:,::-1]
    result[border>0] = inpainted[border>0]
    Image.fromarray(result).save(HERE / f"local_alpha_repaired_{strength:.1f}.png")
