"""Prototype: confine a generated Vivi edit to the existing SAMPLE repair mask."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
base = np.asarray(Image.open(HERE / 'runs/batch_OP13-012_SP_eb9b91ee/repaired_palette_1x.png').convert('RGB'))
source = np.asarray(Image.open(r'C:\Users\timun\Downloads\batch_OP13-012_SP.png').convert('RGB'))
ai = np.asarray(Image.open(HERE / 'vivi_ai_edit_demo.png').convert('RGB').resize(
    (base.shape[1], base.shape[0]), Image.Resampling.LANCZOS))
mask = np.asarray(Image.open(HERE / 'runs/batch_OP13-012_SP_eb9b91ee/repaired_1x_mask.png').convert('L')) > 0
distance = cv2.distanceTransform(mask.astype('uint8'), cv2.DIST_L2, 3)
feather = np.clip(distance / 4, 0, 1) * mask
for strength in (.5, 1.0):
    blend = strength * feather
    result = np.uint8(np.clip(base.astype(np.float32) * (1 - blend[:, :, None]) +
                              ai.astype(np.float32) * blend[:, :, None], 0, 255))
    result[~mask] = source[~mask]
    path = HERE / f'vivi_ai_hybrid_{strength}_1x.png'
    Image.fromarray(result).save(path)
    print(path.name, 'changed_outside_mask', int(np.count_nonzero(np.any(result != source, axis=2) & ~mask)))
