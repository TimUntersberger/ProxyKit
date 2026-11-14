"""Composite a focused 2x local inpaint experiment into the existing workflow output."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--candidate', default='vivi_scroll_local_feathered_0.55_101_2x.png')
parser.add_argument('--border', type=int, default=0)
parser.add_argument('--output', default='batch_OP13-012_SP_local_scroll_2x_preview.png')
args = parser.parse_args()

here = Path(__file__).parent
baseline = np.asarray(Image.open(here / 'batch_OP13-012_SP_sharp_palette_workflow.png').convert('RGB'), np.float32)
candidate = np.asarray(Image.open(here / args.candidate).convert('RGB'), np.float32)
small_mask = np.asarray(Image.open(here / 'runs/batch_OP13-012_SP_eb9b91ee/repaired_1x_mask.png').convert('L')) > 0
if args.border:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * args.border + 1, 2 * args.border + 1))
    small_mask = cv2.dilate(small_mask.astype(np.uint8), kernel) > 0
mask = cv2.resize(small_mask.astype(np.uint8), (baseline.shape[1], baseline.shape[0]),
                  interpolation=cv2.INTER_NEAREST) > 0
roi = np.zeros_like(mask)
roi[870:1040, 0:460] = True
selection = mask & roi
distance = cv2.distanceTransform(selection.astype(np.uint8), cv2.DIST_L2, 3)
blend = np.clip(distance / 12, 0, 1) * selection
result = np.uint8(np.clip(baseline * (1 - blend[..., None]) +
                          candidate * blend[..., None], 0, 255))
path = here / args.output
Image.fromarray(result).save(path)
print(path.name, 'changed', int(np.count_nonzero(np.any(result != baseline, axis=2))),
      'outside_local_mask', int(np.count_nonzero(np.any(result != baseline, axis=2) & ~selection)))
