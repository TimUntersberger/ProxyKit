"""Research comparison: composite cloud edit through the exact local 10px mask."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

here = Path(__file__).parent
run = here / 'runs/batch_OP13-012_SP_eb9b91ee'
base = np.asarray(Image.open(run / 'repaired_palette_1x.png').convert('RGB'), np.float32)
cloud = np.asarray(Image.open(here / 'vivi_cloud_same_input_raw.png').convert('RGB').resize(
    (640, 512), Image.Resampling.LANCZOS), np.float32)
local_mask = np.asarray(Image.open(here / 'vivi_sd15_outline10_mask_640x512.png').convert('L')) > 0
box = (0, 190, 640, 702)
mask = np.zeros(base.shape[:2], bool)
mask[box[1]:box[3], box[0]:box[2]] = local_mask
candidate = base.copy()
candidate[box[1]:box[3], box[0]:box[2]] = cloud
distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
feather = np.clip(distance / 5, 0, 1) * mask

for strength in (.5, 1.0):
    weight = strength * feather
    result = np.uint8(np.clip(base * (1 - weight[..., None]) +
                              candidate * weight[..., None], 0, 255))
    result[~mask] = base[~mask].astype(np.uint8)
    path = here / f'vivi_cloud_same_mask_{strength:.1f}_1x.png'
    Image.fromarray(result).save(path)
    print(path.name, 'changed', int(np.count_nonzero(np.any(result != base, axis=2))),
          'outside_mask', int(np.count_nonzero(np.any(result != base, axis=2) & ~mask)))
