"""Experimental second-pass matte estimate from one watermarked input and its first-pass result."""
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

parser = ArgumentParser()
parser.add_argument('--image', required=True)
parser.add_argument('--first-pass', required=True)
parser.add_argument('--matte', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--strength', type=float, default=0.5)
args = parser.parse_args()

observed = np.asarray(Image.open(args.image).convert('RGB'), np.float32) / 255
background = np.asarray(Image.open(args.first_pass).convert('RGB'), np.float32) / 255
matte = np.asarray(Image.open(args.matte).convert('L'), np.float32) / 255
if background.shape != observed.shape or matte.shape != observed.shape[:2]:
    parser.error('all image dimensions must match')

shape = matte > .5
core = cv2.erode((matte > .9).astype('uint8'), np.ones((9, 9), 'uint8')) > 0
denominator = ((1 - background) ** 2).sum(axis=2)
alpha = ((observed - background) * (1 - background)).sum(axis=2) / np.maximum(denominator, 1e-4)
good = core & (denominator > .12) & (observed.min(axis=2) > .65)
opacity = float(np.median(alpha[good]))

# The white-overlay model predicts the same opacity in each color channel.
# Discount edge pixels where that prediction disagrees across channels.
channel_alpha = (observed - background) / np.maximum(1 - background, .08)
channel_spread = channel_alpha.max(axis=2) - channel_alpha.min(axis=2)
confidence = np.clip((.12 - channel_spread) / .09, 0, 1)
confidence *= np.clip(denominator / .18, 0, 1)

fit = np.clip(alpha / opacity, 0, 1)
fit = cv2.GaussianBlur(fit.astype('float32'), (0, 0), .7)
edge_band = (cv2.dilate(shape.astype('uint8'), np.ones((5, 5), 'uint8')) > 0) & (
    cv2.erode(shape.astype('uint8'), np.ones((5, 5), 'uint8')) == 0)
mix = args.strength * confidence * edge_band
refined = matte * (1 - mix) + fit * mix
output = Path(args.output)
Image.fromarray(np.rint(np.clip(refined, 0, 1) * 255).astype('uint8')).save(output)
print('estimated_opacity', round(opacity, 4), 'edge_pixels', int(edge_band.sum()),
      'mean_matte_change', round(float(np.abs(refined - matte)[edge_band].mean()), 4))
