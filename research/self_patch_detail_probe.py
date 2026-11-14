"""Conservative same-image exemplar detail transfer after watermark removal."""
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
p.add_argument('--strength', type=float, default=.5)
args = p.parse_args()
source = np.asarray(Image.open(args.image).convert('RGB'), np.uint8)
base = np.asarray(Image.open(args.reconstruction).convert('RGB'), np.uint8)
mask = np.asarray(Image.open(args.mask).convert('L')) > 0
if source.shape != base.shape or mask.shape != source.shape[:2]:
    p.error('image sizes must match')
h, w = mask.shape
gray_source = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32)
gray_base = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.float32)
safe = cv2.erode((~mask).astype('uint8'), np.ones((9, 9), 'uint8')) > 0
yy, xx = np.indices(mask.shape)
donor = safe & (xx % 3 == 0) & (yy % 3 == 0)
donor[:5] = donor[-5:] = False
donor[:, :5] = donor[:, -5:] = False
dy, dx = np.nonzero(donor)
interior = cv2.distanceTransform(mask.astype('uint8'), cv2.DIST_L2, 3)
base_grad = cv2.magnitude(cv2.Sobel(gray_base, cv2.CV_32F, 1, 0, 3),
                          cv2.Sobel(gray_base, cv2.CV_32F, 0, 1, 3))
target = mask & (interior > 3) & (base_grad > 25) & (xx % 2 == 0) & (yy % 2 == 0)
ty, tx = np.nonzero(target)
offsets = (-4, 0, 4)
def features(gray, rgb, ys, xs):
    small = np.stack([gray[ys + oy, xs + ox] for oy in offsets for ox in offsets], axis=1)
    # Local shape and the center color, balancing geometry and appearance.
    return np.concatenate(((small - small[:, 4:5]) / 36,
                           rgb[ys, xs].astype(np.float32) / 80), axis=1).astype(np.float32)

donor_features = features(gray_source, source, dy, dx)
target_features = features(gray_base, base, ty, tx)
tree = cKDTree(donor_features)
_, neighbors = tree.query(target_features, k=8, workers=-1)
weights = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16
donor_low = cv2.filter2D(source.astype(np.float32), -1, weights)
base_low = cv2.filter2D(base.astype(np.float32), -1, weights)
best = np.full(len(ty), np.inf, np.float32)
chosen = np.zeros((len(ty), 3), np.float32)
for rank in range(neighbors.shape[1]):
    iy = dy[neighbors[:, rank]]
    ix = dx[neighbors[:, rank]]
    # Verify the surrounding 9x9 texture; center can differ because the
    # reconstructed target has lower contrast than a clean donor.
    cost = np.zeros(len(ty), np.float32)
    for oy in (-4, -2, 0, 2, 4):
        for ox in (-4, -2, 0, 2, 4):
            a = gray_source[iy + oy, ix + ox] - gray_source[iy, ix]
            b = gray_base[ty + oy, tx + ox] - gray_base[ty, tx]
            cost += np.abs(a - b)
    cost /= 25
    better = cost < best
    best[better] = cost[better]
    chosen[better] = source[iy[better], ix[better]].astype(np.float32) - donor_low[iy[better], ix[better]]

confidence = np.clip((12 - best) / 8, 0, 1)
detail = np.clip(chosen, -25, 25)
result = base.astype(np.float32)
blend = args.strength * confidence
result[ty, tx] += blend[:, None] * detail
result = np.clip(result, 0, 255).astype('uint8')
result[~mask] = source[~mask]
Image.fromarray(result).save(Path(args.output))
print('donors', len(dx), 'targets', len(tx), 'accepted', int((confidence > .1).sum()),
      'changed_outside', int(np.count_nonzero(np.any(result != source, axis=2) & ~mask)))
