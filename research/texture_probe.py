"""Compare source-only texture recovery variants against the current output."""
from pathlib import Path
import runpy
import sys

import cv2
import numpy as np
from PIL import Image

here = Path(__file__).parent
sys.argv = [
    str(here / 'one_image_poisson.py'),
    '--image', r'C:\Users\timun\Downloads\OP18-085.webp',
    '--output', str(here / 'texture_probe_baseline.png'),
    '--opacity', '.70',
]
s = runpy.run_path(str(here / 'one_image_poisson.py'))
image = s['image']
matte = s['matte']
unknown = s['unknown']
baseline = s['finished'].astype(np.float32) / 255
anchors = s['weight']

alpha = .7 * matte
inverse = np.clip((image - alpha[:, :, None]) / (1 - alpha[:, :, None]), 0, 1)

# Distance into each letter supplies a soft confidence ramp. A direct inverse
# preserves every input pixel, but near edges registration errors can print the
# letter contour into the result.
shape = (matte > .5).astype('uint8')
distance = cv2.distanceTransform(shape, cv2.DIST_L2, 3)
for start, end in ((1, 3), (2, 5), (3, 7)):
    trust = np.clip((distance - start) / (end - start), 0, 1)
    trust = trust * unknown * (1 - np.clip(anchors * 5, 0, 1))
    output = baseline * (1 - trust[:, :, None]) + inverse * trust[:, :, None]
    output[~unknown] = image[~unknown]
    Image.fromarray((np.clip(output, 0, 1) * 255).astype('uint8')).save(
        here / f'texture_probe_inverse_{start}_{end}.png'
    )

# Also test a conservative blend that never fully uses the direct inverse.
trust = .5 * np.clip((distance - 2) / 4, 0, 1) * unknown
output = baseline * (1 - trust[:, :, None]) + inverse * trust[:, :, None]
output[~unknown] = image[~unknown]
Image.fromarray((np.clip(output, 0, 1) * 255).astype('uint8')).save(
    here / 'texture_probe_inverse_half.png'
)

# Carry only high-frequency texture across the skipped edge band. The low
# frequency color remains the Poisson result and no original watermark edge
# gradients are brought back.
core = s['core']
band = unknown & ~core
band_u8 = band.astype('uint8')
for sigma in (1.2, 2.0):
    low = cv2.GaussianBlur(baseline, (0, 0), sigma)
    detail = baseline - low
    repaired = np.empty_like(detail)
    for c in range(3):
        repaired[:, :, c] = cv2.inpaint(
            detail[:, :, c].astype('float32'), band_u8, 3, cv2.INPAINT_TELEA
        )
    for strength in (.7, 1.3):
        out = baseline.copy()
        out[band] += strength * repaired[band]
        out[~unknown] = image[~unknown]
        Image.fromarray((np.clip(out, 0, 1) * 255).astype('uint8')).save(
            here / f'texture_probe_highpass_{sigma}_{strength}.png'
        )
