"""Keep original upscaled card outside the exact 10px outline comparison mask."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

here = Path(__file__).parent
base = np.asarray(Image.open(here / 'batch_OP13-012_SP_sharp_palette_workflow.png').convert('RGB'), np.float32)
glyph = np.asarray(Image.open(here / 'runs/batch_OP13-012_SP_eb9b91ee/repaired_1x_mask.png').convert('L')) > 0
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
outer = cv2.dilate(glyph.astype(np.uint8), kernel) > 0
inner = cv2.erode(glyph.astype(np.uint8), kernel) > 0
ring = outer & ~inner
selection = cv2.resize(ring.astype(np.uint8), (base.shape[1], base.shape[0]),
                       interpolation=cv2.INTER_NEAREST) > 0
distance = cv2.distanceTransform(selection.astype(np.uint8), cv2.DIST_L2, 3)
feather = np.clip(distance / 8, 0, 1) * selection

for strength in (.5, 1.0):
    candidate = np.asarray(Image.open(here / f'vivi_cloud_same_mask_{strength:.1f}_2x.png').convert('RGB'), np.float32)
    result = np.uint8(np.clip(base * (1 - feather[..., None]) +
                              candidate * feather[..., None], 0, 255))
    result[~selection] = base[~selection].astype(np.uint8)
    path = here / f'batch_OP13-012_SP_cloud_same_mask_{strength:.1f}_full_2x.png'
    Image.fromarray(result).save(path)
    print(path.name, 'changed', int(np.count_nonzero(np.any(result != base, axis=2))),
          'outside_outline', int(np.count_nonzero(np.any(result != base, axis=2) & ~selection)))
