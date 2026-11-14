"""Controlled watermark tests on clean regions of the two supplied cards.

The clean images here are made only from unmarked regions of the supplied
images. This is a diagnostic, not an alternative source for either result.
"""
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
SOURCE_DIR = Path(r'C:\Users\timun\Downloads')
OUT = HERE / 'diagnostics'
OUT.mkdir(exist_ok=True)
BOX = (51, 355, 551, 488)
FONT = ImageFont.truetype(r'C:\Windows\Fonts\impact.ttf', 190)
bb = ImageDraw.Draw(Image.new('L', (1, 1))).textbbox((0, 0), 'SAMPLE', font=FONT)
glyph = Image.new('L', (bb[2]-bb[0], bb[3]-bb[1]))
ImageDraw.Draw(glyph).text((-bb[0], -bb[1]), 'SAMPLE', font=FONT, fill=255)
glyph = glyph.resize((BOX[2]-BOX[0], BOX[3]-BOX[1]), Image.Resampling.BICUBIC)
mask_image = Image.new('L', (600, 838))
mask_image.paste(glyph, BOX[:2])
matte = np.asarray(mask_image, np.float32) / 255

rows = []
for card in ('021', '085'):
    source = np.asarray(Image.open(SOURCE_DIR / f'OP18-{card}.webp').convert('RGB'))
    # The watermark starts near y=355. Stretch only the clean art above it
    # into a test canvas so no pixels from the marked band enter the target.
    clean = cv2.resize(source[20:340], (600, 838), interpolation=cv2.INTER_LINEAR)
    Image.fromarray(clean).save(OUT / f'{card}_clean.png')
    composited = np.rint(clean.astype(np.float32) * (1-.7*matte[:,:,None]) +
                          255*.7*matte[:,:,None]).clip(0, 255).astype('uint8')
    for label, quality, shift in (('raw_exact', None, 0),
                                  ('webp80_exact', 80, 0),
                                  ('webp80_shift', 80, .3),
                                  ('webp80_shift1', 80, 1),
                                  ('webp80_shift2', 80, 2),
                                  ('webp60_exact', 60, 0)):
        observed = composited.copy()
        if quality is not None:
            encoded = cv2.imencode('.webp', cv2.cvtColor(observed, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_WEBP_QUALITY, quality])[1]
            observed = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        estimated = cv2.warpAffine(matte, np.float32([[1, 0, shift], [0, 1, shift]]),
                                   (600, 838), flags=cv2.INTER_LINEAR)
        prefix = f'{card}_{label}'
        input_path = OUT / f'{prefix}_input.png'
        matte_path = OUT / f'{prefix}_matte.png'
        output_path = OUT / f'{prefix}_result.png'
        Image.fromarray(observed).save(input_path)
        Image.fromarray(np.rint(estimated*255).astype('uint8')).save(matte_path)
        if not output_path.exists():
            subprocess.run([sys.executable, str(HERE/'one_image_poisson.py'),
                            '--image', str(input_path), '--output', str(output_path),
                            '--opacity', '.70', '--texture-recovery',
                            '--matte-file', str(matte_path)], check=True, capture_output=True)
        result = np.asarray(Image.open(output_path).convert('RGB'))
        region = matte > .01
        core = cv2.erode((matte > .9).astype('uint8'), np.ones((9,9),'uint8')) > 0
        err = np.abs(result.astype(np.float32)-clean.astype(np.float32)).mean(axis=2)
        rows.append((card, label, round(float(err[region].mean()), 2),
                     round(float(err[core].mean()), 2)))

print('card condition mean_absolute_error_region mean_absolute_error_core')
for row in rows:
    print(*row)
