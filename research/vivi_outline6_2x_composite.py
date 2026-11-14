"""Full-card 2x preview of the corrected 6-pixel outline-only inpaint result."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

here = Path(__file__).parent
base = np.asarray(Image.open(here / 'batch_OP13-012_SP_sharp_palette_workflow.png').convert('RGB'), np.float32)
candidate = np.asarray(Image.open(here / 'vivi_outline6_proper_blended_2x.png').convert('RGB'), np.float32)
glyph = np.asarray(Image.open(here / 'runs/batch_OP13-012_SP_eb9b91ee/repaired_1x_mask.png').convert('L')) > 0
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
outer = cv2.dilate(glyph.astype(np.uint8), kernel) > 0
inner = cv2.erode(glyph.astype(np.uint8), kernel) > 0
ring = outer & ~inner
selection = cv2.resize(ring.astype(np.uint8), (base.shape[1], base.shape[0]),
                       interpolation=cv2.INTER_NEAREST) > 0
distance = cv2.distanceTransform(selection.astype(np.uint8), cv2.DIST_L2, 3)
weight = np.clip(distance / 8, 0, 1) * selection
result = np.uint8(np.clip(base * (1 - weight[..., None]) + candidate * weight[..., None], 0, 255))
path = here / 'batch_OP13-012_SP_outline6_full_2x.png'
Image.fromarray(result).save(path)
print(path.name, 'dimensions', result.shape[1], result.shape[0],
      'changed', int(np.count_nonzero(np.any(result != base, axis=2))),
      'outside_outline', int(np.count_nonzero(np.any(result != base, axis=2) & ~selection)))
