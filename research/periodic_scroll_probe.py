"""Experimental periodic-pattern transfer inside a user-selected scroll band."""
from argparse import ArgumentParser
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

p = ArgumentParser()
p.add_argument('--image', required=True)
p.add_argument('--reconstruction', required=True)
p.add_argument('--matte', required=True)
p.add_argument('--mask', required=True)
p.add_argument('--roi', default='0,440,225,520', help='left,top,right,bottom')
p.add_argument('--output', required=True)
p.add_argument('--strength', type=float, default=.5)
p.add_argument('--color-tolerance', type=float, default=55.,
               help='maximum low-frequency RGB difference for a repeated donor')
args = p.parse_args()
src = np.asarray(Image.open(args.image).convert('RGB'), np.float32)
base = np.asarray(Image.open(args.reconstruction).convert('RGB'), np.float32)
matte = np.asarray(Image.open(args.matte).convert('L'), np.float32) / 255
mask = np.asarray(Image.open(args.mask).convert('L')) > 0
left, top, right, bottom = map(int, args.roi.split(','))
if src.shape != base.shape or matte.shape != mask.shape or src.shape[:2] != mask.shape:
    p.error('image dimensions must match')
if not (0 <= left < right <= src.shape[1] and 0 <= top < bottom <= src.shape[0]):
    p.error('ROI must be within image')

region = base[top:bottom, left:right]
gray = cv2.cvtColor(region.astype('uint8'), cv2.COLOR_RGB2GRAY).astype(np.float32)
edge = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3),
                cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3))
min_period = max(12, (right-left)//5)
max_period = min((right-left)//2, 120)
scores = []
for shift in range(min_period, max_period):
    a, b = edge[:, :-shift], edge[:, shift:]
    score = float(np.mean(a*b) / (np.sqrt(np.mean(a*a)*np.mean(b*b)) + 1e-6))
    scores.append((score, shift))
score, period = max(scores)
if score < .55:
    p.error(f'no confident periodic pattern in ROI (best score {score:.3f})')

output = base.copy()
weight_sum = np.zeros(mask.shape, np.float32)
color_sum = np.zeros_like(base)
yy, xx = np.indices(mask.shape)
roi_mask = (xx >= left) & (xx < right) & (yy >= top) & (yy < bottom) & mask
for shift in (-2*period, -period, period, 2*period):
    donor_x = xx + shift
    valid = roi_mask & (donor_x >= left) & (donor_x < right)
    y, x = np.nonzero(valid)
    donor_x = x + shift
    donor_matte = matte[y, donor_x]
    target_matte = matte[y, x]
    gain = np.clip((target_matte - donor_matte - .15) / .65, 0, 1)
    # Match low-frequency local color to avoid copying a different ornament.
    target_low = cv2.GaussianBlur(base, (0, 0), 2)[y, x]
    donor_color = np.where((donor_matte < .05)[:, None], src[y, donor_x], base[y, donor_x])
    donor_low = cv2.GaussianBlur(base, (0, 0), 2)[y, donor_x]
    similarity = np.clip((args.color_tolerance - np.linalg.norm(target_low-donor_low, axis=1)) /
                         (args.color_tolerance * .65), 0, 1)
    weight = gain * similarity
    color_sum[y, x] += weight[:, None] * donor_color
    weight_sum[y, x] += weight
selection = roi_mask & (weight_sum > .05)
donor_average = color_sum[selection] / weight_sum[selection][:, None]
blend = args.strength * np.clip(weight_sum[selection], 0, 1)
output[selection] = output[selection]*(1-blend[:, None]) + donor_average*blend[:, None]
output = np.clip(output, 0, 255).astype('uint8')
output[~mask] = src[~mask].astype('uint8')
Image.fromarray(output).save(Path(args.output))
print('period', period, 'score', round(score, 3), 'transferred', int(selection.sum()),
      'clean_donor_coverage', round(float(np.mean(np.min(np.stack([np.roll(matte[top:bottom,left:right],s,axis=1)
                             for s in (-2*period,-period,0,period,2*period)]),axis=0)<.05)),3))
