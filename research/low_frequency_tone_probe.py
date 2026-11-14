"""Experimental low-frequency tone correction inside an existing repair mask."""
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

p=ArgumentParser()
p.add_argument('--image',required=True)
p.add_argument('--reconstruction',required=True)
p.add_argument('--mask',required=True)
p.add_argument('--anchors')
p.add_argument('--output',required=True)
p.add_argument('--sigma',type=float,default=20)
p.add_argument('--strength',type=float,default=.5)
p.add_argument('--max-correction',type=float,default=35)
args=p.parse_args()

source=np.asarray(Image.open(args.image).convert('RGB'))
result=np.asarray(Image.open(args.reconstruction).convert('RGB'))
mask=np.asarray(Image.open(args.mask).convert('L'))>0
if source.shape!=result.shape or source.shape[:2]!=mask.shape:
    p.error('dimensions must match')
guide=cv2.inpaint(source,mask.astype('uint8')*255,5,cv2.INPAINT_TELEA)
low_result=cv2.GaussianBlur(result.astype('float32'),(0,0),args.sigma)
low_guide=cv2.GaussianBlur(guide.astype('float32'),(0,0),args.sigma)
correction=np.clip(low_guide-low_result,-args.max_correction,args.max_correction)
distance=cv2.distanceTransform(mask.astype('uint8'),cv2.DIST_L2,3)
weight=args.strength*np.clip(distance/12,0,1)*mask
if args.anchors:
    anchor=np.asarray(Image.open(args.anchors).convert('L'))>0
    protected=cv2.GaussianBlur(anchor.astype('float32'),(0,0),8)
    weight*=1-np.clip(protected*5,0,1)
output=np.clip(result.astype('float32')+weight[:,:,None]*correction,0,255).astype('uint8')
output[~mask]=result[~mask]
Image.fromarray(output).save(Path(args.output))
print('mean_absolute_correction',round(float(np.abs(output.astype('float32')-result)[mask].mean()),2))
