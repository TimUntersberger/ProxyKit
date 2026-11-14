"""Paste a full-width local candidate only into the existing repair mask plus border."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

here = Path(__file__).parent
base = np.asarray(Image.open(here / 'batch_OP13-012_SP_sharp_palette_workflow.png').convert('RGB'), np.float32)
candidate = np.asarray(Image.open(here / 'vivi_full_width_border5_blended_0.45_2x.png').convert('RGB'), np.float32)
small = np.asarray(Image.open(here / 'runs/batch_OP13-012_SP_eb9b91ee/repaired_1x_mask.png').convert('L')) > 0
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
expanded = cv2.dilate(small.astype(np.uint8), kernel) > 0
selection = cv2.resize(expanded.astype(np.uint8), (base.shape[1], base.shape[0]),
                       interpolation=cv2.INTER_NEAREST) > 0
distance = cv2.distanceTransform(selection.astype(np.uint8), cv2.DIST_L2, 3)
weight = np.clip(distance / 10, 0, 1) * selection
result = np.uint8(np.clip(base * (1 - weight[..., None]) + candidate * weight[..., None], 0, 255))
path = here / 'batch_OP13-012_SP_full_width_border5_2x_preview.png'
Image.fromarray(result).save(path)
print(path.name, 'changed', int(np.count_nonzero(np.any(result != base, axis=2))),
      'outside_expanded_mask', int(np.count_nonzero(np.any(result != base, axis=2) & ~selection)))
