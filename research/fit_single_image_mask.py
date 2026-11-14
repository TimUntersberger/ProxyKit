"""Register an approximate SAMPLE font shape to gradients in one image."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'deps'))
import cv2
import numpy as np
from scipy.optimize import differential_evolution
from PIL import Image

here=Path(__file__).parent
source=np.asarray(Image.open(r'C:\Users\timun\Downloads\OP18-085.webp').convert('RGB'))
base=np.asarray(Image.open(here/'approx_mask.png').convert('L'),np.float32)/255
gray=cv2.cvtColor(source,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
gray=cv2.GaussianBlur(gray,(0,0),.7)
ix=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
iy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
sl=np.s_[342:500,45:555]
cx,cy=300,422

def render(p):
    dx,dy,sx,sy=p
    matrix=np.float32([[sx,0,dx+cx*(1-sx)],[0,sy,dy+cy*(1-sy)]])
    m=cv2.warpAffine(base,matrix,(600,838),flags=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(m[sl],(0,0),.55)

def objective(p):
    m=render(p)
    mx=cv2.Sobel(m,cv2.CV_32F,1,0,ksize=3)
    my=cv2.Sobel(m,cv2.CV_32F,0,1,ksize=3)
    mag=np.hypot(mx,my)
    alignment=float((mx*ix[sl]+my*iy[sl]).sum()/max(mag.sum(),1))
    # Weakly prefer the measured initial size and position unless gradients disagree.
    prior=.002*(p[0]**2+p[1]**2)+.3*((p[2]-1)**2+(p[3]-1)**2)
    return -alignment+prior

initial=(0,0,1,1)
print('initial',objective(initial),flush=True)
opt=differential_evolution(objective,[(-7,7),(-7,7),(.96,1.04),(.95,1.05)],
                           maxiter=28,popsize=8,seed=12,polish=True,tol=.0005)
print('best',opt.x,'objective',opt.fun,'evaluations',opt.nfev,flush=True)
dx,dy,sx,sy=opt.x
matrix=np.float32([[sx,0,dx+cx*(1-sx)],[0,sy,dy+cy*(1-sy)]])
aligned=cv2.warpAffine(base,matrix,(600,838),flags=cv2.INTER_LINEAR)
Image.fromarray((aligned*255).astype('uint8')).save(here/'single_image_fit_mask.png')
edge=cv2.morphologyEx((aligned>.5).astype('uint8'),cv2.MORPH_GRADIENT,np.ones((3,3),'uint8'))>0
preview=source.copy();preview[edge]=[0,255,0]
Image.fromarray(preview).save(here/'single_image_fit_preview.png')
