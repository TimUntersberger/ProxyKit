"""Estimate edge-only overlay opacity from neighboring recovered colors."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "deps"))
import cv2
import numpy as np
from PIL import Image

here = Path(__file__).parent
src = np.asarray(Image.open(r"C:\Users\timun\Downloads\OP18-085.webp").convert("RGB"), np.float32)
base = np.asarray(Image.open(here / "opacity_adaptive_0.72.png").convert("RGB"), np.float32)
letters0 = np.asarray(Image.open(here / "shared_edge_binary.png").convert("L")) > 0
letters = cv2.warpAffine(letters0.astype("uint8"),np.float32([[1,0,0],[0,1,3]]),(600,838),flags=cv2.INTER_NEAREST)
di = cv2.distanceTransform(letters,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
do = cv2.distanceTransform(1-letters,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
sd = di-do
gx = cv2.Sobel(cv2.GaussianBlur(sd,(0,0),1),cv2.CV_32F,1,0)
gy = cv2.Sobel(cv2.GaussianBlur(sd,(0,0),1),cv2.CV_32F,0,1)
m = np.maximum(np.hypot(gx,gy),.001)
nx,ny = gx/m,gy/m
y,x = np.mgrid[0:838,0:600].astype(np.float32)
cap = np.clip((src.min(axis=2)-8)/255,0,.9)
for radius in (2,3,4):
    offset = radius+2
    inner = cv2.remap(base,x+nx*offset,y+ny*offset,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    outer = cv2.remap(base,x-nx*offset,y-ny*offset,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    weight = np.clip((sd+offset)/(2*offset),0,1)[:,:,None]
    expected = outer*(1-weight)+inner*weight
    vec = 255-expected
    a = np.sum((src-expected)*vec,axis=2)/np.maximum(np.sum(vec*vec,axis=2),100)
    a = np.clip(a,0,.72)
    a = cv2.GaussianBlur(a,(0,0),1.1)
    a = np.minimum(a,cap)
    inverse = np.clip((src-a[:,:,None]*255)/(1-a[:,:,None]),0,255)
    band = np.abs(sd)<=radius
    out = base.copy()
    out[band]=inverse[band]
    Image.fromarray(out.astype('uint8')).save(here/f"outline_alpha_fit_{radius}.png")
    Image.fromarray((a*255).astype('uint8')).save(here/f"outline_alpha_fit_{radius}_matte.png")
    print(radius,round(float(a[band].mean()),3))
