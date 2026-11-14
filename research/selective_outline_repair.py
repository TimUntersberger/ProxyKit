"""Mark visible stroke fragments and inpaint only those pixels."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'deps'))
import cv2
import numpy as np
from PIL import Image

here=Path(__file__).parent
src=np.asarray(Image.open(here/'opacity_adaptive_0.72.png').convert('RGB'))
binary=np.asarray(Image.open(here/'shared_edge_binary.png').convert('L'))>0
letters=cv2.warpAffine(binary.astype('uint8'),np.float32([[1,0,0],[0,1,3]]),(600,838),flags=cv2.INTER_NEAREST)
di=cv2.distanceTransform(letters,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
do=cv2.distanceTransform(1-letters,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
sd=di-do
gx=cv2.Sobel(cv2.GaussianBlur(sd,(0,0),1),cv2.CV_32F,1,0)
gy=cv2.Sobel(cv2.GaussianBlur(sd,(0,0),1),cv2.CV_32F,0,1)
gm=np.maximum(np.hypot(gx,gy),.001)
nx,ny=gx/gm,gy/gm
lum=cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
median=cv2.medianBlur(lum,7)
resid=lum.astype(np.float32)-median.astype(np.float32)
ix=cv2.Sobel(lum,cv2.CV_32F,1,0,ksize=3)/8
iy=cv2.Sobel(lum,cv2.CV_32F,0,1,ksize=3)/8
normal=np.abs(ix*nx+iy*ny)
for threshold in (8,14,20):
    # Strong, thin local extrema along the expected stroke edge.
    band=np.abs(sd)<=3
    marked=band & ((np.abs(resid)>threshold)&(normal>threshold/2))
    marked=cv2.morphologyEx(marked.astype('uint8'),cv2.MORPH_CLOSE,np.ones((3,3),'uint8'))
    marked=(marked>0)&band
    mask=(marked*255).astype('uint8')
    result=cv2.inpaint(src[:,:,::-1].copy(),mask,2,cv2.INPAINT_TELEA)[:,:,::-1]
    Image.fromarray(mask).save(here/f'selective_outline_mask_{threshold}.png')
    Image.fromarray(result).save(here/f'selective_outline_repair_{threshold}.png')
    print(threshold,int(marked.sum()))
