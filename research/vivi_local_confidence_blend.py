"""Research probe: transfer only low-detail local model corrections into source repair."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

here = Path(__file__).parent
run = here / 'runs/batch_OP13-012_SP_eb9b91ee'
source = np.asarray(Image.open(r'C:\Users\timun\Downloads\batch_OP13-012_SP.png').convert('RGB'))
base = np.asarray(Image.open(run / 'repaired_palette_1x.png').convert('RGB'))
generated = np.asarray(Image.open(here / 'vivi_local_inpaint_0.35_101.png').convert('RGB'))
mask = np.asarray(Image.open(run / 'repaired_1x_mask.png').convert('L')) > 0
matte = np.asarray(Image.open(run / 'repaired_1x_matte.png').convert('L'), np.float32) / 255

def gradient(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3),
                    cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)) / 8

base_grad = gradient(base)
gen_grad = gradient(generated)
color_delta = np.linalg.norm(base.astype(np.float32) - generated.astype(np.float32), axis=2)
distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
confidence = (np.clip((22 - base_grad) / 12, 0, 1) *
              np.clip((22 - gen_grad) / 12, 0, 1) *
              np.clip((45 - color_delta) / 30, 0, 1) *
              np.clip((matte - .15) / .35, 0, 1) *
              np.clip((distance - 1) / 4, 0, 1) * mask)
for amount in (.4, .7):
    weight = amount * confidence
    result = np.uint8(np.clip(base.astype(np.float32) * (1 - weight[..., None]) +
                              generated.astype(np.float32) * weight[..., None], 0, 255))
    result[~mask] = source[~mask]
    path = here / f'vivi_local_confidence_{amount:.1f}.png'
    Image.fromarray(result).save(path)
    print(path.name, 'changed', int(np.count_nonzero(np.any(result != base, axis=2))),
          'outside_mask', int(np.count_nonzero(np.any(result != source, axis=2) & ~mask)))
