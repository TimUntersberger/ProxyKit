"""Replace narrow letter outlines using pixels on both sides of each edge."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "deps"))
import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
base = np.asarray(Image.open(HERE / "opacity_adaptive_0.72.png").convert("RGB"),dtype=np.float32)
binary = np.asarray(Image.open(HERE / "shared_edge_binary.png").convert("L")) > 0
letters = cv2.warpAffine(binary.astype("uint8"),np.float32([[1,0,0],[0,1,3]]),(600,838),flags=cv2.INTER_NEAREST)
inside_distance = cv2.distanceTransform(letters,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
outside_distance = cv2.distanceTransform(1-letters,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
signed = inside_distance - outside_distance
smooth = cv2.GaussianBlur(signed,(0,0),1.0)
gx = cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3)
gy = cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3)
mag = np.maximum(np.sqrt(gx*gx+gy*gy),.001)
nx,ny = gx/mag,gy/mag
yy,xx = np.mgrid[0:838,0:600].astype(np.float32)

for radius in (2.0,3.0,4.0):
    offset = radius + 3.0
    inner = cv2.remap(base,xx+nx*offset,yy+ny*offset,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    outer = cv2.remap(base,xx-nx*offset,yy-ny*offset,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    t = np.clip((signed+radius)/(2*radius),0,1)[:,:,None]
    filled = outer*(1-t)+inner*t
    band = np.abs(signed)<=radius
    result = base.copy()
    result[band] = filled[band]
    Image.fromarray(np.clip(result,0,255).astype('uint8')).save(HERE / f"normal_repair_{int(radius)}.png")
    Image.fromarray((band.astype('uint8')*255)).save(HERE / f"normal_repair_{int(radius)}_mask.png")
    print(radius,int(band.sum()))
