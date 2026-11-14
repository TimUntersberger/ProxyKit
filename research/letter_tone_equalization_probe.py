"""Experimental color-jump correction for connected glyph components."""
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

p=ArgumentParser()
p.add_argument('--image',required=True)
p.add_argument('--mask',required=True)
p.add_argument('--output',required=True)
p.add_argument('--strength',type=float,default=1.0)
args=p.parse_args()

base=np.asarray(Image.open(args.image).convert('RGB'),np.float32)
unknown=np.asarray(Image.open(args.mask).convert('L'))>0
shape=cv2.erode(unknown.astype('uint8'),np.ones((5,5),'uint8'))>0
inside=cv2.distanceTransform(shape.astype('uint8'),cv2.DIST_L2,3)
outside=cv2.distanceTransform((~shape).astype('uint8'),cv2.DIST_L2,3)
inner_ring=(inside>=2)&(inside<=6)
outer_ring=(outside>=2)&(outside<=6)

def local_color(region):
    weight=cv2.GaussianBlur(region.astype('float32'),(0,0),8)
    color=cv2.GaussianBlur(base*region[:,:,None],(0,0),8)
    return color/np.maximum(weight[:,:,None],1e-4)

jump=local_color(inner_ring)-local_color(outer_ring)
count,labels,stats,_=cv2.connectedComponentsWithStats(shape.astype('uint8'))
result=base.copy()
for label in range(1,count):
    if stats[label,cv2.CC_STAT_AREA]<100:
        continue
    edge=(labels==label)&(inside>=2)&(inside<=4)
    color_jump=np.clip(np.median(jump[edge],axis=0),-12,12)
    region=labels==label
    fade=np.clip(inside[region]/4,0,1)*args.strength
    result[region]-=fade[:,None]*color_jump
    print('component',label,'jump',np.round(color_jump,2).tolist())
result=np.uint8(np.clip(result,0,255))
result[~unknown]=base[~unknown].astype('uint8')
Image.fromarray(result).save(Path(args.output))
