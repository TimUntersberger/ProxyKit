"""Find the narrowest unreliable border band for source-gradient recovery."""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).parent/'deps'))
import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg
from PIL import Image

here=Path(__file__).parent
image=np.asarray(Image.open(r'C:\Users\timun\Downloads\OP18-085.webp').convert('RGB'),np.float32)/255
shape=np.asarray(Image.open(here/'single_image_fit_mask.png').convert('L'))>127
for inner_radius,outer_radius in ((1,1),(2,1),(2,2),(3,2)):
    inner_kernel=np.ones((2*inner_radius+1,)*2,'uint8')
    outer_kernel=np.ones((2*outer_radius+1,)*2,'uint8')
    unknown=cv2.dilate(shape.astype('uint8'),outer_kernel)>0
    core=cv2.erode(shape.astype('uint8'),inner_kernel)>0
    ys,xs=np.nonzero(unknown);n=len(ys)
    index=np.full(unknown.shape,-1,np.int32);index[ys,xs]=np.arange(n)
    rows=[];cols=[];data=[]
    rhs_boundary=np.zeros((n,3),np.float32)
    rhs_gradient=np.zeros((n,3),np.float32)
    for dy,dx in ((0,1),(0,-1),(1,0),(-1,0)):
        yq=np.clip(ys+dy,0,837);xq=np.clip(xs+dx,0,599)
        qi=index[yq,xq];inside=qi>=0
        rows.extend(np.nonzero(inside)[0].tolist());cols.extend(qi[inside].tolist());data.extend([-1]*int(inside.sum()))
        rhs_boundary[~inside]+=image[yq[~inside],xq[~inside]]
        keep=core[ys,xs]&core[yq,xq]
        rhs_gradient[keep]+=image[ys[keep],xs[keep]]-image[yq[keep],xq[keep]]
    rows.extend(np.arange(n).tolist());cols.extend(np.arange(n).tolist());data.extend([4]*n)
    matrix=coo_matrix((data,(rows,cols)),shape=(n,n)).tocsr()
    rhs=rhs_boundary+rhs_gradient/.4
    result=image.copy();begin=time.perf_counter()
    for channel in range(3):
        x,info=cg(matrix,rhs[:,channel],rtol=1e-5,maxiter=2500)
        if info:raise RuntimeError(info)
        result[ys,xs,channel]=x
    path=here/f'poisson_band_in{inner_radius}_out{outer_radius}.png'
    Image.fromarray((np.clip(result,0,1)*255).astype('uint8')).save(path)
    print(inner_radius,outer_radius,n,round(time.perf_counter()-begin,2),flush=True)
