"""Experimental color propagation within connected image-color regions."""
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

p=ArgumentParser()
p.add_argument('--image',required=True)
p.add_argument('--reconstruction',required=True)
p.add_argument('--mask',required=True)
p.add_argument('--output',required=True)
p.add_argument('--colors',type=int,default=32)
args=p.parse_args()

src=np.asarray(Image.open(args.image).convert('RGB'))
result=np.asarray(Image.open(args.reconstruction).convert('RGB'))
mask=np.asarray(Image.open(args.mask).convert('L'))>0
guide=cv2.inpaint(src,mask.astype('uint8')*255,5,cv2.INPAINT_TELEA)
guide_lab=cv2.cvtColor(guide,cv2.COLOR_RGB2Lab)
src_lab=cv2.cvtColor(src,cv2.COLOR_RGB2Lab)
yy,xx=np.indices(mask.shape)
sample=(~mask)&(xx%4==0)&(yy%4==0)
training=src_lab[sample].astype('float32')
_,_,centers=cv2.kmeans(training,args.colors,None,
                        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,.3),
                        2,cv2.KMEANS_PP_CENTERS)
labels=cKDTree(centers).query(guide_lab.reshape(-1,3),workers=-1)[1].reshape(mask.shape)

gray=cv2.GaussianBlur(cv2.cvtColor(result,cv2.COLOR_RGB2GRAY).astype('float32')/255,(0,0),1)
mag=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3),
                  cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3))
flat=np.clip((.10-cv2.GaussianBlur(mag,(0,0),4))/.07,0,1)
edges=cv2.Canny(result,50,100)>0
distance=cv2.distanceTransform((~edges).astype('uint8'),cv2.DIST_L2,3)
safe=np.clip((distance-3)/6,0,1)
output=result.astype('float32').copy()
count=0
for color in range(args.colors):
    n,components=cv2.connectedComponents((labels==color).astype('uint8'))
    candidates=np.unique(components[mask & (labels==color)])
    for component in candidates[candidates>0]:
        region=components==component
        outside=region&~mask
        inside=region&mask
        if outside.sum()<100 or inside.sum()<30:
            continue
        donor=src[outside].astype('float32')
        if np.median(np.std(donor,axis=0))>28:
            continue
        color_value=np.median(donor,axis=0)
        weight=.8*flat[inside]*safe[inside]
        output[inside]=output[inside]*(1-weight[:,None])+color_value*weight[:,None]
        count+=int((weight>.1).sum())
Image.fromarray(np.uint8(np.clip(output,0,255))).save(Path(args.output))
print('color_regions',args.colors,'blended_pixels',count)
