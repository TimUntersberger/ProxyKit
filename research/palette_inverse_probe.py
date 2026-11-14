"""Experimental inverse-overlay color snapping in flat, palette-like regions.

Uses only the supplied image's unmarked pixels as color candidates.  The
existing reconstruction supplies structure; this adjusts low-gradient areas
inside known watermark strokes when the forward overlay predicts the source.
"""
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

p = ArgumentParser()
p.add_argument('--image', required=True)
p.add_argument('--reconstruction', required=True)
p.add_argument('--matte', required=True)
p.add_argument('--mask', required=True)
p.add_argument('--output', required=True)
p.add_argument('--opacity', type=float, default=.70)
p.add_argument('--strength', type=float, default=.6)
p.add_argument('--spatial-weight', type=float, default=.15)
p.add_argument('--all-interior', action='store_true',
               help='also snap colors on detailed strokes, not just flat fills')
args = p.parse_args()

source = np.asarray(Image.open(args.image).convert('RGB'))
result = np.asarray(Image.open(args.reconstruction).convert('RGB'))
matte = np.asarray(Image.open(args.matte).convert('L'), np.float32) / 255
mask = np.asarray(Image.open(args.mask).convert('L')) > 0
if source.shape != result.shape or source.shape[:2] != matte.shape or matte.shape != mask.shape:
    p.error('all images must have the same dimensions')

# Frequent unmarked colors are plausible original fills; reject isolated
# colors from antialiased lines or compression noise.
unmarked = source[~mask]
colors, counts = np.unique(unmarked.reshape(-1, 3), axis=0, return_counts=True)
palette = colors[counts >= 30].astype(np.float32)
if len(palette) < 8:
    p.error('not enough recurring unmarked colors for a palette estimate')

height, width = matte.shape
yy, xx = np.indices((height, width))
outside = (~mask).astype('uint8')
distance_from_mask = cv2.distanceTransform(outside, cv2.DIST_L2, 3)
# For each palette color, locate actual nearby unmarked donors of that color.
donor = (~mask) & (distance_from_mask < 120) & ((xx % 3) == 0) & ((yy % 3) == 0)
donor_rgb = source[donor].astype(np.float32)
if len(donor_rgb) == 0:
    p.error('no unmarked donor pixels')
palette_tree = cKDTree(palette)
nearest_palette = palette_tree.query(donor_rgb)[1]
donor_good = np.linalg.norm(donor_rgb - palette[nearest_palette], axis=1) < 1
dy, dx = np.nonzero(donor)
dy, dx = dy[donor_good], dx[donor_good]
donor_rgb = donor_rgb[donor_good]

alpha = args.opacity * matte
inverse = np.clip((source.astype(np.float32) - 255 * alpha[:, :, None]) /
                  np.maximum(1 - alpha[:, :, None], .05), 0, 255)

gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
local_gradient = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 2)
flat = np.clip((.13 - local_gradient) / .09, 0, 1)
edge = cv2.Canny(result, 50, 100) > 0
edge_distance = cv2.distanceTransform((~edge).astype('uint8'), cv2.DIST_L2, 3)
edge_safety = np.clip((edge_distance - 2) / 5, 0, 1)
inside = cv2.distanceTransform((matte > .5).astype('uint8'), cv2.DIST_L2, 3)
interior = np.clip((inside - 3) / 4, 0, 1)
region_confidence = interior if args.all_interior else flat * edge_safety * interior
eligible = mask & (matte > .95) & (region_confidence > .05)
ty, tx = np.nonzero(eligible)

scale = args.spatial_weight
donor_features = np.column_stack((donor_rgb, dx * scale, dy * scale))
target_features = np.column_stack((inverse[ty, tx], tx * scale, ty * scale))
tree = cKDTree(donor_features)
_, matches = tree.query(target_features, k=4, workers=-1)
choices = donor_rgb[matches]
predicted = (1 - alpha[ty, tx, None, None]) * choices + 255 * alpha[ty, tx, None, None]
observed = source[ty, tx, None, :].astype(np.float32)
forward_error = np.linalg.norm(predicted - observed, axis=2)
spatial_distance = np.hypot(dx[matches] - tx[:, None], dy[matches] - ty[:, None])
cost = forward_error + scale * spatial_distance
pick = np.argmin(cost, axis=1)
chosen = choices[np.arange(len(ty)), pick]
fit = forward_error[np.arange(len(ty)), pick]
confidence = np.clip((24 - fit) / 16, 0, 1)
weight = args.strength * region_confidence[ty, tx] * confidence

output = result.copy().astype(np.float32)
output[ty, tx] = output[ty, tx] * (1 - weight[:, None]) + chosen * weight[:, None]
output = np.clip(output, 0, 255).astype('uint8')
output[~mask] = source[~mask]
Image.fromarray(output).save(Path(args.output))
print('palette_colors', len(palette), 'donors', len(donor_rgb),
      'eligible', len(ty), 'blended_gt_10pct', int((weight > .1).sum()),
      'changed_outside', int(np.count_nonzero(np.any(output != source, axis=2) & ~mask)))
