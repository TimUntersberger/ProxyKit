"""Feather local border-mask experiments into the source-based Vivi repair."""
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
roi[435:520, 0:230] = True  # Selected scroll band for this research probe.

for border in (3, 6):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border + 1, 2 * border + 1))
    expanded = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    selection = expanded & roi
    distance = cv2.distanceTransform(selection.astype(np.uint8), cv2.DIST_L2, 3)
    weight = .8 * np.clip(distance / 5, 0, 1) * selection
    candidate = np.asarray(Image.open(here / f'vivi_local_scroll_border{border}_0.55_101.png').convert('RGB'), np.float32)
    result = np.uint8(np.clip(base * (1 - weight[..., None]) +
                              candidate * weight[..., None], 0, 255))
    result[~expanded] = source[~expanded]
    path = here / f'vivi_scroll_border{border}_feathered_1x.png'
    Image.fromarray(result).save(path)
    print(path.name, 'changed_outside_expanded',
          int(np.count_nonzero(np.any(result != source, axis=2) & ~expanded)))
