"""Preserve compact chromatic highlights while solving a watermark band."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'deps'))
import cv2
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import cg
from PIL import Image

here=Path(__file__).parent
src_u8=np.asarray(Image.open(r'C:\Users\timun\Downloads\OP18-085.webp').convert('RGB'))
image=src_u8.astype(np.float32)/255
matte=np.asarray(Image.open(here/'single_image_fit_mask.png').convert('L'),np.float32)/255
shape=matte>.5
unknown=cv2.dilate(shape.astype('uint8'),np.ones((5,5),'uint8'))>0
core=cv2.erode(shape.astype('uint8'),np.ones((7,7),'uint8'))>0

# A compact bright feature has a local luminance peak, chroma unlike its
# neighborhood, and negative curvature in both image directions. Long font
# strokes tend to have curvature in only one direction.
lab=cv2.cvtColor(image,cv2.COLOR_RGB2Lab)
chroma=np.linalg.norm(lab[:,:,1:]-cv2.GaussianBlur(lab,(0,0),4)[:,:,1:],axis=2)
lum=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
peak=lum-cv2.GaussianBlur(lum,(0,0),3)
smooth=cv2.GaussianBlur(lum,(0,0),1)
xx=cv2.Sobel(smooth,cv2.CV_32F,2,0,ksize=3)
yy=cv2.Sobel(smooth,cv2.CV_32F,0,2,ksize=3)
xy=cv2.Sobel(smooth,cv2.CV_32F,1,1,ksize=3)
curvature=(xx+yy+np.sqrt((xx-yy)**2+4*xy**2))/2
candidate=(unknown&(matte>.25)&(chroma>14)&(peak>.09)&(lum>.55)&(curvature<-.04)).astype('uint8')
n_labels,labels,stats,_=cv2.connectedComponentsWithStats(candidate)
accepted=[i for i in range(1,n_labels) if 2<=stats[i,cv2.CC_STAT_AREA]<=100]
salient=np.isin(labels,accepted).astype('uint8')
weight=cv2.GaussianBlur(salient.astype(np.float32),(0,0),1)
weight*=unknown
Image.fromarray((salient*255)).save(here/'poisson_feature_anchor_mask.png')

ys,xs=np.nonzero(unknown);n=len(ys)
index=np.full(unknown.shape,-1,np.int32);index[ys,xs]=np.arange(n)
rows=[];cols=[];data=[]
boundary=np.zeros((n,3),np.float32);gradient=np.zeros((n,3),np.float32)
for dy,dx in ((0,1),(0,-1),(1,0),(-1,0)):
    yq=np.clip(ys+dy,0,837);xq=np.clip(xs+dx,0,599)
    qi=index[yq,xq];inside=qi>=0
    rows.extend(np.nonzero(inside)[0].tolist());cols.extend(qi[inside].tolist());data.extend([-1]*int(inside.sum()))
    boundary[~inside]+=image[yq[~inside],xq[~inside]]
    keep=core[ys,xs]&core[yq,xq]
    gradient[keep]+=image[ys[keep],xs[keep]]-image[yq[keep],xq[keep]]
rows.extend(np.arange(n).tolist());cols.extend(np.arange(n).tolist());data.extend([4]*n)
matrix=coo_matrix((data,(rows,cols)),shape=(n,n)).tocsr()
alpha=.6*matte
target=np.clip((image-alpha[:,:,None])/(1-alpha[:,:,None]),0,1)
base_rhs=boundary+gradient/.4
for lam in (2,8,24):
    w=lam*weight[ys,xs]
    mat=matrix+diags(w)
    rhs=base_rhs+w[:,None]*target[ys,xs]
    result=image.copy()
    for channel in range(3):
        x,info=cg(mat,rhs[:,channel],rtol=1e-5,maxiter=2500)
        if info:raise RuntimeError(info)
        result[ys,xs,channel]=x
    out=np.clip(result*255,0,255).astype('uint8')
    # Original bytes outside the mask are kept exactly.
    out[~unknown]=src_u8[~unknown]
    Image.fromarray(out).save(here/f'poisson_feature_anchor_{lam}.png')
    print('lambda',lam,'salient',int(salient.sum()),'eye_pixel',out[371,426],flush=True)
