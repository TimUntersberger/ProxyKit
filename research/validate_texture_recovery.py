"""Test texture recovery on unrelated images and imperfect mask registration."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'deps'))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import cg
from skimage import data as skdata

here=Path(__file__).parent
font=ImageFont.truetype(r'C:\Windows\Fonts\impact.ttf',140)
text=Image.new('L',(650,150));draw=ImageDraw.Draw(text)
bbox=draw.textbbox((0,0),'SAMPLE',font=font)
draw.text((-bbox[0],-bbox[1]),'SAMPLE',font=font,fill=255)
glyph=text.crop((0,0,bbox[2]-bbox[0],bbox[3]-bbox[1])).resize((410,108))

def solve(observed,matte,lam,texture,opacity):
    image=observed.astype(np.float32)/255
    shape=matte>.5
    unknown=cv2.dilate(shape.astype('uint8'),np.ones((5,5),'uint8'))>0
    core=cv2.erode(shape.astype('uint8'),np.ones((7,7),'uint8'))>0
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
    weight=cv2.GaussianBlur(salient.astype(np.float32),(0,0),1)*unknown
    ys,xs=np.nonzero(unknown);n=len(ys);h,w=unknown.shape
    index=np.full((h,w),-1,np.int32);index[ys,xs]=np.arange(n)
    rows=[];cols=[];values=[]
    boundary=np.zeros((n,3),np.float32);gradient=np.zeros((n,3),np.float32)
    for dy,dx in ((0,1),(0,-1),(1,0),(-1,0)):
        yq=np.clip(ys+dy,0,h-1);xq=np.clip(xs+dx,0,w-1)
        qi=index[yq,xq];inside=qi>=0
        rows.extend(np.nonzero(inside)[0].tolist());cols.extend(qi[inside].tolist());values.extend([-1]*int(inside.sum()))
        boundary[~inside]+=image[yq[~inside],xq[~inside]]
        keep=core[ys,xs]&core[yq,xq]
        gradient[keep]+=image[ys[keep],xs[keep]]-image[yq[keep],xq[keep]]
    rows.extend(np.arange(n).tolist());cols.extend(np.arange(n).tolist());values.extend([4]*n)
    matrix=coo_matrix((values,(rows,cols)),shape=(n,n)).tocsr()
    wgt=lam*weight[ys,xs]
    matrix=matrix+diags(wgt)
    alpha=opacity*matte
    target=np.clip((image-alpha[:,:,None])/(1-alpha[:,:,None]),0,1)
    rhs=boundary+gradient/(1-opacity)+wgt[:,None]*target[ys,xs]
    result=image.copy()
    for c in range(3):
        x,info=cg(matrix,rhs[:,c],rtol=1e-5,maxiter=2500)
        if info:raise RuntimeError(info)
        result[ys,xs,c]=x
    if texture:
        distance=cv2.distanceTransform(shape.astype('uint8'),cv2.DIST_L2,3)
        trust=.75*np.clip((distance-2)/3,0,1)*unknown
        trust*=1-np.clip(weight*5,0,1)
        result=result*(1-trust[:,:,None])+target*trust[:,:,None]
    return np.clip(result*255,0,255).astype('uint8'),int(salient.sum()),unknown

samples={'astronaut':skdata.astronaut(),'chelsea':skdata.chelsea(),'coffee':skdata.coffee()}
for name,source in samples.items():
    clean=cv2.resize(source,(512,512),interpolation=cv2.INTER_AREA)
    mask=Image.new('L',(512,512));mask.paste(glyph,(51,185))
    true_matte=np.asarray(mask,np.float32)/255
    observed=np.clip(clean.astype(np.float32)*(1-.7*true_matte[:,:,None])+255*.7*true_matte[:,:,None],0,255).astype('uint8')
    encoded=cv2.imencode('.webp',cv2.cvtColor(observed,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_WEBP_QUALITY,80])[1]
    observed=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    estimated=cv2.warpAffine(true_matte,np.float32([[1,0,.3],[0,1,0]]),(512,512))
    for opacity in (.68,.70,.72):
        baseline,_,unknown=solve(observed,estimated,8,False,opacity)
        recovered,_,_=solve(observed,estimated,8,True,opacity)
        err0=float(np.abs(baseline[unknown].astype('float32')-clean[unknown]).mean())
        err1=float(np.abs(recovered[unknown].astype('float32')-clean[unknown]).mean())
        print(name,'opacity',opacity,'MAE',round(err0,2),round(err1,2),flush=True)
        if opacity==.70:
            Image.fromarray(recovered).save(here/f'holdout_{name}_texture.png')
