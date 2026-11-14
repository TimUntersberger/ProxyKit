"""Ground-truth check for phase-aligned transfer on a synthetic repeated strip."""
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
roi = np.s_[447:505, 0:225]
clean = np.tile(source[:300], (3, 1, 1))[:source.shape[0]].copy()
tile = source[250:308, 0:69]
clean[roi] = np.tile(tile, (1, 4, 1))[:, :225]
marked = np.rint(clean.astype(np.float32)*(1-.7*matte[:,:,None]) +
                  255*.7*matte[:,:,None]).clip(0,255).astype('uint8')
marked = np.asarray(Image.fromarray(marked).quantize(
    colors=127, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).convert('RGB'))
Image.fromarray(clean).save(OUT / 'periodic_clean.png')
Image.fromarray(marked).save(OUT / 'periodic_marked.png')
baseline = OUT / 'periodic_baseline.png'
subprocess.run([sys.executable, str(HERE/'one_image_poisson.py'),
                '--image', str(OUT/'periodic_marked.png'), '--matte-file', str(matte_path),
                '--output', str(baseline), '--opacity', '.70', '--texture-recovery',
                '--edge-gradient-weight', '.35', '--edge-clearance', '.5', '--core-size', '3'],
               check=True, capture_output=True)
palette = OUT / 'periodic_palette.png'
subprocess.run([sys.executable, str(HERE/'palette_inverse_probe.py'),
                '--image', str(OUT/'periodic_marked.png'), '--reconstruction', str(baseline),
                '--matte', str(matte_path), '--mask', str(OUT/'periodic_baseline_mask.png'),
                '--output', str(palette), '--strength', '1.0', '--spatial-weight', '0.4',
                '--all-interior'], check=True, capture_output=True)
for strength in (0, .5, .8, 1.0):
    if strength:
        result = OUT / f'periodic_transfer_{strength}.png'
        subprocess.run([sys.executable, str(HERE/'periodic_scroll_probe.py'),
                        '--image', str(OUT/'periodic_marked.png'),
                        '--reconstruction', str(palette), '--matte', str(matte_path),
                        '--mask', str(OUT/'periodic_baseline_mask.png'),
                        '--roi', '0,447,225,505', '--output', str(result),
                        '--strength', str(strength)], check=True, capture_output=True)
    else:
        result = palette
    reconstructed = np.asarray(Image.open(result).convert('RGB'),np.float32)
    error = np.abs(reconstructed[roi] - clean[roi].astype(np.float32)).mean()
    print('strength', strength, 'roi_mae', round(float(error),3))
