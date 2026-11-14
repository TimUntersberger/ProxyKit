"""Experimental whole-image search for a known translucent word watermark.

Searches for the glyph's oriented edges in image gradients. The detected box
is still approximate; one_image_poisson.py performs fine alignment afterward.
"""
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def detect_box(image_path, word='SAMPLE', font_path=None):
    source = np.asarray(Image.open(image_path).convert('RGB'))
    h, w = source.shape[:2]
    scale = 600 / w
    small = cv2.resize(source, (600, round(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype('float32') / 255,
                            (0, 0), .7)
    ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    best = None
    font_paths = [font_path] if font_path else [
        r'C:\Windows\Fonts\impact.ttf', r'C:\Windows\Fonts\arial.ttf',
        r'C:\Windows\Fonts\arialbd.ttf']
    for candidate_font in font_paths:
        font = ImageFont.truetype(candidate_font, 190)
        bbox = ImageDraw.Draw(Image.new('L', (1, 1))).textbbox((0, 0), word, font=font)
        glyph = Image.new('L', (bbox[2]-bbox[0], bbox[3]-bbox[1]))
        ImageDraw.Draw(glyph).text((-bbox[0], -bbox[1]), word, font=font, fill=255)
        glyph = np.asarray(glyph, np.float32) / 255
        for aspect in (3.5, 3.8, 4.1, 4.5, 5.0, 5.5, 6.0):
            for width in range(330, 571, 10):
              height = round(width / aspect)
              if height >= gray.shape[0]:
                  continue
              matte = cv2.resize(glyph, (width, height), interpolation=cv2.INTER_CUBIC)
              matte = cv2.GaussianBlur(matte, (0, 0), .55)
              mx = cv2.Sobel(matte, cv2.CV_32F, 1, 0, ksize=3)
              my = cv2.Sobel(matte, cv2.CV_32F, 0, 1, ksize=3)
              score = (cv2.matchTemplate(ix, mx, cv2.TM_CCORR) +
                       cv2.matchTemplate(iy, my, cv2.TM_CCORR)) / max(np.hypot(mx, my).sum(), 1)
              # The supplied card watermark is centered over the illustration.
              # Exclude rules text near the lower edge, which otherwise dominates
              # the gradient match on cards with a faint watermark.
              rows = np.arange(score.shape[0])
              centered = ((rows + height / 2) / gray.shape[0] >= .35) & (
                  (rows + height / 2) / gray.shape[0] <= .65)
              score[~centered] = -np.inf
              _, maximum, _, location = cv2.minMaxLoc(score)
              if best is None or maximum > best[0]:
                  best = (maximum, location[0], location[1], width, height, candidate_font)
    if best is None:
        raise RuntimeError('no box candidates fit this image')
    _, x, y, bw, bh, matched_font = best
    detected = (round(x / scale), round(y / scale),
                round((x + bw) / scale), round((y + bh) / scale))
    return detected, float(best[0]), matched_font


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--word', default='SAMPLE')
    p.add_argument('--font')
    args = p.parse_args()
    box, score, font = detect_box(Path(args.image), args.word, args.font)
    print(','.join(map(str, box)))
    print('score', round(score, 5))
    print('font', font)
