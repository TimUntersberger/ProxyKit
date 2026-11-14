"""Compare local MI-GAN inpainting masks on the sample card."""
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).parent/'deps'))
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

here=Path(__file__).parent
source=np.asarray(Image.open(r'C:\Users\timun\Downloads\OP18-085.webp').convert('RGB'))
dewhite=np.asarray(Image.open(here/'opacity_adaptive_0.72.png').convert('RGB'))
letter0=np.asarray(Image.open(here/'shared_edge_binary.png').convert('L'))>0
letter=cv2.warpAffine(letter0.astype('uint8'),np.float32([[1,0,0],[0,1,3]]),(600,838),flags=cv2.INTER_NEAREST)
full=cv2.dilate(letter,np.ones((5,5),'uint8'))>0
edge=np.asarray(Image.open(here/'edge_guided_mask_2.png').convert('L'))>0
select=np.asarray(Image.open(here/'selective_outline_mask_8.png').convert('L'))>0

session=ort.InferenceSession(str(here/'models/migan_pipeline_v2.onnx'),providers=['CPUExecutionProvider'])
print('inputs',[(v.name,v.shape,v.type) for v in session.get_inputs()])
print('outputs',[(v.name,v.shape,v.type) for v in session.get_outputs()])
for name,base,mask in [('full_original',source,full),('full_dewhite',dewhite,full),('edge_dewhite',dewhite,edge),('selective_dewhite',dewhite,select)]:
    image=np.ascontiguousarray(base.transpose(2,0,1)[None])
    keep=np.ascontiguousarray((~mask).astype('uint8')[None,None]*255)
    begin=time.perf_counter()
    output=session.run(None,{'image':image,'mask':keep})[0]
    elapsed=time.perf_counter()-begin
    result=output[0].transpose(1,2,0)
    # Preserve all untouched pixels, regardless of graph compositing details.
    result[~mask]=base[~mask]
    Image.fromarray(result).save(here/f'migan_{name}.png')
    print(name,'mask',int(mask.sum()),'seconds',round(elapsed,2),'outside_changed',int(np.any(result[~mask]!=base[~mask])))
