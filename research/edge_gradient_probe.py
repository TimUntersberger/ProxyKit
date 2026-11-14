"""Source-only fidelity probe for a known translucent word watermark.

The rough word, font, and box are supplied for this sample. Registration and
color reconstruction use only the input image; no reference artwork or model.
"""
from argparse import ArgumentParser
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import differential_evolution
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import cg

parser=ArgumentParser()
parser.add_argument('--image',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--word',default='SAMPLE')
parser.add_argument('--box',default='51,355,551,488',help='rough left,top,right,bottom')
parser.add_argument('--auto-box',action='store_true',
                    help='search the whole image for the supplied word before fine alignment')
parser.add_argument('--font',default=r'C:\Windows\Fonts\impact.ttf')
parser.add_argument('--opacity',type=float,default=.60)
parser.add_argument('--matte-file',help='optional precomputed grayscale watermark matte')
parser.add_argument('--save-matte',action='store_true',help='save the aligned matte beside the output')
parser.add_argument('--feature-anchor-strength',type=float,default=8.0)
parser.add_argument('--texture-recovery',action='store_true',
                    help='blend inverse-overlay pixels into safe letter interiors')
parser.add_argument('--edge-gradient-weight',type=float,default=0.,
                    help='experimental weight for inverse-composited gradients outside the safe glyph core')
parser.add_argument('--edge-clearance',type=float,default=0.,
                    help='suppress inverse gradients this many pixels around glyph boundaries')
parser.add_argument('--texture-cap',type=float,default=.75,
                    help='maximum blend of direct inverse color into safe stroke interiors')
parser.add_argument('--texture-fade-start',type=float,default=2.,
                    help='distance in pixels from the glyph edge before direct inverse blending starts')
parser.add_argument('--core-size',type=int,default=7,
                    help='odd erosion kernel used for safe unmodified source gradients')
args=parser.parse_args()
src_u8=np.asarray(Image.open(args.image).convert('RGB'))
h,w=src_u8.shape[:2]
if args.auto_box:
    from detect_watermark_box import detect_box
    detected,score=detect_box(args.image,args.word,args.font)
    if score<.5:
        parser.error(f'no confident {args.word} match (score {score:.3f}); supply --box')
    args.box=','.join(map(str,detected))
    print('detected_box',args.box,'score',round(score,4))
left,top,right,bottom=[int(v) for v in args.box.split(',')]
font=ImageFont.truetype(args.font,190)
text_bbox=ImageDraw.Draw(Image.new('L',(1,1))).textbbox((0,0),args.word,font=font)
text=Image.new('L',(text_bbox[2]-text_bbox[0],text_bbox[3]-text_bbox[1]))
ImageDraw.Draw(text).text((-text_bbox[0],-text_bbox[1]),args.word,font=font,fill=255)
glyph=text.resize((right-left,bottom-top),Image.Resampling.BICUBIC)
rough=Image.new('L',(w,h));rough.paste(glyph,(left,top))
rough=np.asarray(rough,np.float32)/255

gray=cv2.GaussianBlur(cv2.cvtColor(src_u8,cv2.COLOR_RGB2GRAY).astype(np.float32)/255,(0,0),.7)
ix=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
iy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
margin=12
sl=np.s_[max(0,top-margin):min(h,bottom+margin),max(0,left-margin):min(w,right+margin)]
cx,cy=(left+right)/2,(top+bottom)/2
def render(p):
    dx,dy,sx,sy=p
    matrix=np.float32([[sx,0,dx+cx*(1-sx)],[0,sy,dy+cy*(1-sy)]])
    return cv2.warpAffine(rough,matrix,(w,h),flags=cv2.INTER_LINEAR)
def objective(p):
    m=cv2.GaussianBlur(render(p)[sl],(0,0),.55)
    mx=cv2.Sobel(m,cv2.CV_32F,1,0,ksize=3)
    my=cv2.Sobel(m,cv2.CV_32F,0,1,ksize=3)
    mag=np.hypot(mx,my)
    align=(mx*ix[sl]+my*iy[sl]).sum()/max(mag.sum(),1)
    return float(-align+.002*(p[0]**2+p[1]**2)+.3*((p[2]-1)**2+(p[3]-1)**2))
start=time.perf_counter()
opt=differential_evolution(objective,[(-7,7),(-7,7),(.96,1.04),(.95,1.05)],
                           maxiter=28,popsize=8,seed=12,polish=True,tol=.0005)
matte=render(opt.x)
if args.matte_file:
    supplied=np.asarray(Image.open(args.matte_file).convert('L'),np.float32)/255
    if supplied.shape!=(h,w):
        parser.error('--matte-file dimensions must match the input image')
    matte=supplied
if args.save_matte:
    Image.fromarray(np.rint(matte*255).astype('uint8')).save(
        Path(args.output).with_name(Path(args.output).stem+'_matte.png'))
shape=matte>.5
unknown=cv2.dilate(shape.astype('uint8'),np.ones((5,5),'uint8'))>0
core=cv2.erode(shape.astype('uint8'),np.ones((args.core_size,args.core_size),'uint8'))>0
inside_distance=cv2.distanceTransform(shape.astype('uint8'),cv2.DIST_L2,3)
outside_distance=cv2.distanceTransform((~shape).astype('uint8'),cv2.DIST_L2,3)
clearance=np.where(shape,inside_distance,outside_distance)
ys,xs=np.nonzero(unknown);n=len(ys)
index=np.full((h,w),-1,np.int32);index[ys,xs]=np.arange(n)
image=src_u8.astype(np.float32)/255
alpha=args.opacity*matte
inverse=np.clip((image-alpha[:,:,None])/(1-alpha[:,:,None]),0,1)

# Preserve compact, locally distinctive bright features visible through the
# overlay. Negative curvature in both directions rejects long watermark edges.
# This uses only image statistics, not coordinates or object labels.
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
rows=[];cols=[];data=[]
boundary=np.zeros((n,3),np.float32);gradient=np.zeros((n,3),np.float32)
for dy,dx in ((0,1),(0,-1),(1,0),(-1,0)):
    yq=np.clip(ys+dy,0,h-1);xq=np.clip(xs+dx,0,w-1)
    qi=index[yq,xq];inside=qi>=0
    rows.extend(np.nonzero(inside)[0].tolist());cols.extend(qi[inside].tolist());data.extend([-1]*int(inside.sum()))
    boundary[~inside]+=image[yq[~inside],xq[~inside]]
    keep=core[ys,xs]&core[yq,xq]
    gradient[keep]+=(image[ys[keep],xs[keep]]-image[yq[keep],xq[keep]])/(1-args.opacity)
    edge=inside&~keep
    if args.edge_gradient_weight:
        # Trust the analytical inverse only fractionally near uncertain glyph
        # edges; high matte derivatives are especially registration-sensitive.
        delta=np.abs(matte[ys[edge],xs[edge]]-matte[yq[edge],xq[edge]])
        confidence=args.edge_gradient_weight*np.clip(1-2*delta,0,1)
        if args.edge_clearance:
            depth=np.minimum(clearance[ys[edge],xs[edge]],clearance[yq[edge],xq[edge]])
            confidence*=np.clip((depth-args.edge_clearance)/1.5,0,1)
        gradient[edge]+=confidence[:,None]*(inverse[ys[edge],xs[edge]]-inverse[yq[edge],xq[edge]])
rows.extend(np.arange(n).tolist());cols.extend(np.arange(n).tolist());data.extend([4]*n)
matrix=coo_matrix((data,(rows,cols)),shape=(n,n)).tocsr()
rhs=boundary+gradient
anchor_weight=args.feature_anchor_strength*weight[ys,xs]
matrix=matrix+diags(anchor_weight)
anchor=inverse
rhs+=anchor_weight[:,None]*anchor[ys,xs]
result=image.copy()
for channel in range(3):
    values,info=cg(matrix,rhs[:,channel],rtol=1e-5,maxiter=2500)
    if info:raise RuntimeError(f'Poisson solver failed: {info}')
    result[ys,xs,channel]=values
if args.texture_recovery:
    # The direct inverse keeps input texture but is sensitive to imperfect
    # watermark registration. Trust it only several pixels into each glyph.
    distance=cv2.distanceTransform(shape.astype('uint8'),cv2.DIST_L2,3)
    trust=args.texture_cap*np.clip((distance-args.texture_fade_start)/3,0,1)*unknown
    trust*=1-np.clip(weight*5*min(args.feature_anchor_strength/8,1),0,1)
    result=result*(1-trust[:,:,None])+anchor*trust[:,:,None]
output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
finished=(np.clip(result,0,1)*255).astype('uint8')
finished[~unknown]=src_u8[~unknown]
Image.fromarray(finished).save(output)
Image.fromarray((unknown*255).astype('uint8')).save(output.with_name(output.stem+'_mask.png'))
Image.fromarray((salient*255)).save(output.with_name(output.stem+'_anchors.png'))
print('alignment',opt.x,'changed_pixels',n,'feature_anchors',int(salient.sum()),'seconds',round(time.perf_counter()-start,2))
