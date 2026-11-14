"""Fit a global colored translucent overlay from one image's boundary gradients."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'deps'))
import cv2
import numpy as np
from scipy.optimize import differential_evolution
from PIL import Image

here=Path(__file__).parent
source=np.asarray(Image.open(r'C:\Users\timun\Downloads\OP18-085.webp').convert('RGB'),np.float32)/255
mask=np.asarray(Image.open(here/'single_image_fit_mask.png').convert('L'),np.float32)/255
sl=np.s_[343:499,46:554]
I=source[sl]; M=cv2.GaussianBlur(mask[sl],(0,0),.5)
mx=cv2.Sobel(M,cv2.CV_32F,1,0,ksize=3); my=cv2.Sobel(M,cv2.CV_32F,0,1,ksize=3)
mag=np.hypot(mx,my)
nx=mx/np.maximum(mag,.01);ny=my/np.maximum(mag,.01)
edge=(mag>.1)

def objective(p):
    a=p[0]; F=np.array(p[1:],np.float32)
    alpha=a*M
    raw=((I-alpha[:,:,None]*F)/(1-alpha[:,:,None])).astype(np.float32)
    gx=cv2.Sobel(raw,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(raw,cv2.CV_32F,0,1,ksize=3)
    directional=gx*nx[:,:,None]+gy*ny[:,:,None]
    robust=np.sqrt(directional[edge]**2+.04**2).mean()
    negative=np.minimum(raw,0)
    clip=(negative*negative).mean()*6
    prior=.005*np.sum((F-1)**2)
    return float(robust+clip+prior)

print('white baseline',objective([.65,1,1,1]),flush=True)
opt=differential_evolution(objective,[(.35,.82),(.65,1.05),(.65,1.05),(.65,1.05)],
                           seed=3,popsize=8,maxiter=35,tol=.001,polish=True)
print('fitted',opt.x,'objective',opt.fun,'evals',opt.nfev,flush=True)
a=opt.x[0];F=np.array(opt.x[1:],np.float32)
fullM=cv2.GaussianBlur(mask,(0,0),.5)
alpha=a*fullM
raw=(source-alpha[:,:,None]*F)/(1-alpha[:,:,None])
Image.fromarray((np.clip(raw,0,1)*255).astype('uint8')).save(here/'single_image_color_fit.png')
