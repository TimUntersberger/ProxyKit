"""Experimental flat-fill repair using only unmasked pixels of the input.

This does not alter detected edges or pixels outside the watermark mask.
It is a diagnostic prototype, not the default reconstruction method.
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
p.add_argument('--mask', required=True)
p.add_argument('--output', required=True)
p.add_argument('--spatial-weight', type=float, default=.2)
p.add_argument('--strength', type=float, default=.8)
args = p.parse_args()

source = np.asarray(Image.open(args.image).convert('RGB'))
result = np.asarray(Image.open(args.reconstruction).convert('RGB'))
mask = np.asarray(Image.open(args.mask).convert('L')) > 0
if source.shape != result.shape or source.shape[:2] != mask.shape:
    p.error('dimensions must match')

guide = cv2.inpaint(source, mask.astype('uint8') * 255, 5, cv2.INPAINT_TELEA)
gray = cv2.GaussianBlur(cv2.cvtColor(result, cv2.COLOR_RGB2GRAY).astype('float32') / 255,
                        (0, 0), 1)
mag = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                    cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
local = cv2.GaussianBlur(mag, (0, 0), 4)
flatness = np.clip((.10 - local) / .07, 0, 1)
edges = cv2.Canny(result, 50, 100) > 0
edge_distance = cv2.distanceTransform((~edges).astype('uint8'), cv2.DIST_L2, 3)
edge_safety = np.clip((edge_distance - 3) / 6, 0, 1)
target_weight = flatness * edge_safety * mask

# Restrict donors to nearby unmasked, locally flat pixels. The stride keeps
# the nearest-neighbor search small without tying it to image content.
distance_to_mask = cv2.distanceTransform((~mask).astype('uint8'), cv2.DIST_L2, 3)
donor = (~mask) & (distance_to_mask < 140) & (local < .08)
yy, xx = np.indices(mask.shape)
donor &= (xx % 2 == 0) & (yy % 2 == 0)
dy, dx = np.nonzero(donor)
ty, tx = np.nonzero(target_weight > .05)
if len(dx) == 0 or len(tx) == 0:
    raise RuntimeError('no suitable flat source or target pixels')

scale = args.spatial_weight
donor_features = np.column_stack((dx * scale, dy * scale, source[dy, dx].astype('float32')))
target_features = np.column_stack((tx * scale, ty * scale, guide[ty, tx].astype('float32')))
_, indexes = cKDTree(donor_features).query(target_features, k=4, workers=-1)
donor_colors = source[dy[indexes], dx[indexes]].astype('float32')
chosen = np.median(donor_colors, axis=1)
color_gap = np.linalg.norm(chosen - guide[ty, tx].astype('float32'), axis=1)
confidence = np.clip((50 - color_gap) / 35, 0, 1)
blend = (args.strength * target_weight[ty, tx] * confidence)[:, None]

output = result.copy()
output[ty, tx] = np.clip(result[ty, tx] * (1 - blend) + chosen * blend, 0, 255)
Image.fromarray(output).save(Path(args.output))
print('targets', len(tx), 'donors', len(dx), 'mean_blend', round(float(blend.mean()), 3))
