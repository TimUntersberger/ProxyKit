"""Measure loss of strong edges under a synthetic indexed watermark."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
source = np.asarray(Image.open(r'C:\Users\timun\Downloads\batch_OP13-012_SP.png').convert('RGB'))
matte = np.asarray(Image.open(HERE / 'batch_OP13-012_SP_edge_baseline_1x_matte.png').convert('L'))
core = matte > 240
def gradient(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3),
                    cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3))

for name, start in (('upper', 0), ('lower', 560)):
    clean = np.tile(source[start:start+300], (3, 1, 1))[:source.shape[0]]
    marked = np.asarray(Image.open(HERE / 'diagnostics' / f'vivi_{name}_palette_marked.png').convert('RGB'))
    recovered = np.asarray(Image.open(HERE / 'diagnostics' / f'vivi_{name}_core3_palette.png').convert('RGB'))
    g0, g1, g2 = map(gradient, (clean, marked, recovered))
    strong = core & (g0 > 100)
    lost = strong & (g1 < .35 * g0)
    print(name, 'strong_edges', int(strong.sum()),
          'heavily_attenuated_pct', round(100 * float(lost.sum() / strong.sum()), 1),
          'median_marked_ratio', round(float(np.median(g1[strong] / g0[strong])), 2),
          'median_recovered_ratio', round(float(np.median(g2[strong] / g0[strong])), 2))
