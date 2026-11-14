"""Test estimating a narrow border from a shared watermark edge map."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "deps"))
import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
original = np.asarray(Image.open(Path(r"C:\Users\timun\Downloads\OP18-085.webp")).convert("RGB"))
q25 = np.asarray(Image.open(HERE / "shared_percentile_25.png").convert("RGB"))
gray = cv2.cvtColor(q25, cv2.COLOR_RGB2GRAY)
font = np.asarray(Image.open(HERE / "approx_mask.png").convert("L"))
prior = cv2.dilate((font > 5).astype("uint8"), np.ones((9,9),"uint8"))
shared = ((gray > 120) & (prior > 0)).astype("uint8")
shared = cv2.morphologyEx(shared, cv2.MORPH_CLOSE, np.ones((3,3),"uint8"))
shared[:345] = 0
shared[498:] = 0
Image.fromarray(shared * 255).save(HERE / "shared_edge_binary.png")

edge = cv2.morphologyEx(shared, cv2.MORPH_GRADIENT, np.ones((3,3),"uint8"))
target_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
gx = cv2.Sobel(target_gray, cv2.CV_32F, 1, 0)
gy = cv2.Sobel(target_gray, cv2.CV_32F, 0, 1)
gradient = np.sqrt(gx*gx + gy*gy)
scores = []
for dx in range(-5,6):
    for dy in range(-5,8):
        shifted = np.zeros_like(edge)
        matrix = np.float32([[1,0,dx],[0,1,dy]])
        shifted = cv2.warpAffine(edge, matrix, (600,838), flags=cv2.INTER_NEAREST)
        scores.append((float((gradient * shifted).sum() / max(shifted.sum(),1)),dx,dy))
score,dx,dy = max(scores)
print("best_registration",dx,dy,round(score,2))
shared = cv2.warpAffine(shared,np.float32([[1,0,dx],[0,1,dy]]),(600,838),flags=cv2.INTER_NEAREST)

# Approximate inverse compositing with a softened shared alpha shape.
soft = cv2.GaussianBlur(shared.astype("float32"),(0,0),.8)
alpha = np.clip(.65 * soft,0,.9)
inverse = np.clip((original.astype("float32")-alpha[:,:,None]*255)/(1-alpha[:,:,None]),0,255).astype("uint8")
Image.fromarray(inverse).save(HERE / "edge_guided_inverse.png")

for radius in (1,2,3):
    k = 2*radius+1
    outer = cv2.dilate(shared,np.ones((k,k),"uint8"))
    inner = cv2.erode(shared,np.ones((k,k),"uint8"))
    border = ((outer > 0) & (inner == 0)).astype("uint8")*255
    Image.fromarray(border).save(HERE / f"edge_guided_mask_{radius}.png")
    filled = cv2.inpaint(inverse[:,:,::-1].copy(),border,2,cv2.INPAINT_TELEA)[:,:,::-1]
    result = inverse.copy()
    result[border>0] = filled[border>0]
    Image.fromarray(result).save(HERE / f"edge_guided_repair_{radius}.png")
