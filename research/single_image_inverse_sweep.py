"""Inverse-blend a mask fitted solely to the supplied image."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'deps'))
import cv2
import numpy as np
from PIL import Image,ImageDraw

here=Path(__file__).parent
src=np.asarray(Image.open(r'C:\Users\timun\Downloads\OP18-085.webp').convert('RGB'),np.float32)/255
mask=np.asarray(Image.open(here/'single_image_fit_mask.png').convert('L'),np.float32)/255
binary=mask>.5
inner=cv2.erode(binary.astype('uint8'),np.ones((5,5),'uint8'))>0
outer=cv2.dilate(binary.astype('uint8'),np.ones((5,5),'uint8'))>0
band=outer&~inner
cap=np.clip(src.min(2)-8/255,0,.9)
results=[]
for sigma in (.45,.8,1.2):
    soft=cv2.GaussianBlur(mask,(0,0),sigma)
    for peak in (.50,.55,.60,.65,.70,.75):
        alpha=np.minimum(peak*soft,cap)
        result=np.clip((src-alpha[:,:,None])/(1-alpha[:,:,None]),0,1)
        lum=cv2.cvtColor(result.astype('float32'),cv2.COLOR_RGB2GRAY)
        gx=cv2.Sobel(lum,cv2.CV_32F,1,0,ksize=3)
        gy=cv2.Sobel(lum,cv2.CV_32F,0,1,ksize=3)
        mx=cv2.Sobel(soft,cv2.CV_32F,1,0,ksize=3)
        my=cv2.Sobel(soft,cv2.CV_32F,0,1,ksize=3)
        dot=(gx*mx+gy*my)/np.maximum(np.hypot(mx,my),.01)
        edge_mean=float(dot[band].mean())
        edge_abs=float(np.abs(dot[band]).mean())
        results.append((sigma,peak,edge_mean,edge_abs,result))
        print(sigma,peak,round(edge_mean,3),round(edge_abs,3))
        Image.fromarray((result*255).astype('uint8')).save(here/f'single_inverse_{sigma:.2f}_{peak:.2f}.png')
montage=Image.new('RGB',(1020,310*len(results)))
for i,(sigma,peak,mean,ab,res) in enumerate(results):
    tile=Image.fromarray((res*255).astype('uint8')).crop((45,345,555,500)).resize((1020,310))
    ImageDraw.Draw(tile).text((5,5),f'sigma={sigma} alpha={peak} edge={mean:.3f}',fill='yellow')
    montage.paste(tile,(0,310*i))
montage.save(here/'single_inverse_sweep.png')
