"""One-image synthetic-overlay training probe for a conservative alpha estimator."""
from pathlib import Path
import random,time
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

here=Path(__file__).parent
torch.manual_seed(17);random.seed(17);np.random.seed(17)
source=np.asarray(Image.open(r'C:\Users\timun\Downloads\OP18-085.webp').convert('RGB'),np.float32)/255
glyph=np.asarray(Image.open(here/'approx_mask.png').convert('L'),np.float32)[355:488,51:551]/255
art=source[35:340,45:555]

class Block(nn.Module):
    def __init__(self,a,b):
        super().__init__();self.seq=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(True),nn.Conv2d(b,b,3,padding=1),nn.ReLU(True))
    def forward(self,x):return self.seq(x)
class Net(nn.Module):
    def __init__(self):
        super().__init__();self.a=Block(4,24);self.b=Block(24,48);self.c=Block(48,96);self.u=Block(144,48);self.v=Block(72,24);self.out=nn.Conv2d(24,1,1)
    def forward(self,x):
        a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2))
        u=self.u(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear'),b],1))
        v=self.v(torch.cat([F.interpolate(u,size=a.shape[-2:],mode='bilinear'),a],1))
        return torch.sigmoid(self.out(v))*.85

def batch(n=12,s=96):
    clean=[];observed=[];prior=[];truth=[]
    for _ in range(n):
        y=random.randint(0,art.shape[0]-s);x=random.randint(0,art.shape[1]-s)
        bg=art[y:y+s,x:x+s]
        gy=random.randint(0,glyph.shape[0]-s);gx=random.randint(0,glyph.shape[1]-s)
        shape=glyph[gy:gy+s,gx:gx+s].copy()
        shape=cv2.GaussianBlur(shape,(0,0),random.uniform(.45,1.2))
        opacity=random.uniform(.38,.76)
        alpha=opacity*shape
        mark=bg*(1-alpha[:,:,None])+alpha[:,:,None]
        mark=np.clip(mark+np.random.normal(0,.004,mark.shape),0,1)
        k=random.choice([1,3,5,7]);rough=(shape>.2).astype('uint8')
        if k>1:
            op=random.choice([cv2.dilate,cv2.erode]);rough=op(rough,np.ones((k,k),'uint8'))
        rough=np.roll(rough,(random.randint(-2,2),random.randint(-2,2)),(0,1)).astype('float32')
        clean.append(bg);observed.append(mark);prior.append(rough);truth.append(alpha)
    inp=np.concatenate([np.stack(observed),np.stack(prior)[...,None]],axis=3)
    return (torch.from_numpy(inp.transpose(0,3,1,2)).float().cuda(),
            torch.from_numpy(np.stack(truth)[:,None]).float().cuda(),
            torch.from_numpy(np.stack(clean).transpose(0,3,1,2)).float().cuda())

net=Net().cuda();opt=torch.optim.Adam(net.parameters(),lr=.001)
begin=time.perf_counter()
for step in range(401):
    inp,alpha,bg=batch();pred=net(inp)
    restored=((inp[:,:3]-pred)/(1-pred)).clamp(0,1)
    loss=(pred-alpha).abs().mean()+.5*(restored-bg).abs().mean()
    opt.zero_grad();loss.backward();opt.step()
    if step%100==0:print('step',step,'loss',round(float(loss),4),'seconds',round(time.perf_counter()-begin,1),flush=True)
torch.save(net.state_dict(),here/'models/self_supervised_alpha.pt')

net.eval();rough=np.asarray(Image.open(here/'approx_mask.png').convert('L'),np.float32)/255
rough=cv2.dilate((rough>.05).astype('uint8'),np.ones((7,7),'uint8')).astype(np.float32)
with torch.inference_mode():
    inp=np.concatenate([source,rough[:,:,None]],2)
    tensor=torch.from_numpy(inp.transpose(2,0,1).copy())[None].float().cuda()
    # Pad to multiples of four for the down/up path.
    tensor=F.pad(tensor,(0,0,0,2),mode='replicate')
    learned=net(tensor)[0,0,:838].cpu().numpy()
alpha=np.minimum(learned*rough,np.clip(source.min(2)-8/255,0,.9))
restored=np.clip((source-alpha[:,:,None])/(1-alpha[:,:,None]),0,1)
Image.fromarray((alpha*255).astype('uint8')).save(here/'self_supervised_alpha_matte.png')
Image.fromarray((restored*255).astype('uint8')).save(here/'self_supervised_alpha_result.png')
