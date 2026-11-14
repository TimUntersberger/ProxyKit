"""Feather a local inpaint candidate within a selected repair region."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

here = Path(__file__).parent
run = here / 'runs/batch_OP13-012_SP_eb9b91ee'
base = np.asarray(Image.open(run / 'repaired_palette_1x.png').convert('RGB'), np.float32)
source = np.asarray(Image.open(r'C:\Users\timun\Downloads\batch_OP13-012_SP.png').convert('RGB'))
mask = np.asarray(Image.open(run / 'repaired_1x_mask.png').convert('L')) > 0
roi = np.zeros_like(mask)
roi[435:520, 0:230] = True  # User-selected scroll band for this research comparison.
feather = cv2.distanceTransform(roi.astype('uint8'), cv2.DIST_L2, 3)
feather = np.clip(feather / 6, 0, 1) * mask

for name, amount in [('0.55_101', .8), ('0.75_202', .7)]:
    candidate = np.asarray(Image.open(here / f'vivi_local_scroll_inpaint_{name}.png').convert('RGB'), np.float32)
    weight = amount * feather
    result = np.uint8(np.clip(base * (1 - weight[..., None]) +
                              candidate * weight[..., None], 0, 255))
    result[~mask] = source[~mask]
    output = here / f'vivi_scroll_local_feathered_{name}.png'
    Image.fromarray(result).save(output)
    print(output.name, 'changed', int(np.count_nonzero(np.any(result != base, axis=2))),
          'outside_mask', int(np.count_nonzero(np.any(result != source, axis=2) & ~mask)))
