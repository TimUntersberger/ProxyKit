"""Two-region synthetic check of palette-aware inverse recovery for indexed art."""
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
OUT = HERE / 'diagnostics'
source = np.asarray(Image.open(r'C:\Users\timun\Downloads\batch_OP13-012_SP.png').convert('RGB'))
matte_path = HERE / 'batch_OP13-012_SP_edge_baseline_1x_matte.png'
matte = np.asarray(Image.open(matte_path).convert('L'), np.float32) / 255
region = matte > .01
for name, start in (('upper', 0), ('lower', 560)):
    clean = np.tile(source[start:start+300], (3, 1, 1))[:source.shape[0]]
    observed = np.rint(clean.astype(np.float32) * (1 - .7 * matte[:, :, None]) +
                        255 * .7 * matte[:, :, None]).clip(0, 255).astype('uint8')
    observed = np.asarray(Image.fromarray(observed).quantize(
        colors=127, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).convert('RGB'))
    input_path = OUT / f'vivi_{name}_palette_marked.png'
    baseline_path = OUT / f'vivi_{name}_palette_baseline.png'
    Image.fromarray(observed).save(input_path)
    subprocess.run([sys.executable, str(HERE / 'one_image_poisson.py'),
                    '--image', str(input_path), '--matte-file', str(matte_path),
                    '--output', str(baseline_path), '--opacity', '.70',
                    '--texture-recovery', '--edge-gradient-weight', '.35',
                    '--edge-clearance', '.5'], check=True, capture_output=True)
    results = [('baseline', baseline_path)]
    for strength in (.5, 1.0):
        candidate_path = OUT / f'vivi_{name}_palette_all_{strength}.png'
        subprocess.run([sys.executable, str(HERE / 'palette_inverse_probe.py'),
                        '--image', str(input_path), '--reconstruction', str(baseline_path),
                        '--matte', str(matte_path),
                        '--mask', str(OUT / f'vivi_{name}_palette_baseline_mask.png'),
                        '--output', str(candidate_path), '--strength', str(strength),
                        '--all-interior'], check=True, capture_output=True)
        results.append((str(strength), candidate_path))
    for label, path in results:
        result = np.asarray(Image.open(path).convert('RGB'), np.float32)
        error = np.abs(result - clean.astype(np.float32)).mean(axis=2)
        print(name, label, round(float(error[region].mean()), 3))
