"""Keep an upscaled source unchanged outside a feathered repair region."""
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

p=ArgumentParser()
p.add_argument('--original-upscaled',required=True)
p.add_argument('--candidate',required=True)
p.add_argument('--small-mask',required=True)
p.add_argument('--other-mask')
p.add_argument('--output',required=True)
p.add_argument('--dilate',type=int,default=20,help='extra repair margin at output size')
p.add_argument('--feather',type=float,default=16,help='blend width at output size')
args=p.parse_args()

source=np.asarray(Image.open(args.original_upscaled).convert('RGB'))
candidate=np.asarray(Image.open(args.candidate).convert('RGB'))
small=np.asarray(Image.open(args.small_mask).convert('L'))>0
if source.shape!=candidate.shape:
    p.error('upscaled source and candidate dimensions must match')
if args.dilate<0 or args.feather<=0:
    p.error('--dilate must be nonnegative and --feather must be positive')
mask=cv2.resize(small.astype('uint8'),(source.shape[1],source.shape[0]),
                interpolation=cv2.INTER_NEAREST)>0
if args.other_mask:
    other=np.asarray(Image.open(args.other_mask).convert('L'))>0
    if other.shape!=mask.shape:
        p.error('--other-mask dimensions must match the upscaled image')
    mask|=other
if args.dilate:
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (2*args.dilate+1,2*args.dilate+1))
    mask=cv2.dilate(mask.astype('uint8'),kernel)>0
distance=cv2.distanceTransform(mask.astype('uint8'),cv2.DIST_L2,3)
blend=np.clip(distance/args.feather,0,1)*mask
output=np.uint8(np.clip(source.astype('float32')*(1-blend[:,:,None])+
                        candidate.astype('float32')*blend[:,:,None],0,255))
output[~mask]=source[~mask]
path=Path(args.output)
path.parent.mkdir(parents=True,exist_ok=True)
Image.fromarray(output).save(path)
Image.fromarray(np.uint8(mask*255)).save(path.with_name(path.stem+'_mask.png'))
print('repair_mask_pixels',int(mask.sum()),'changed_outside',
      int(np.count_nonzero(np.any(output!=source,axis=2)&~mask)))
