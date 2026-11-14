"""Narrow non-AI repair after adaptive opacity reversal."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "deps"))
import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
for opacity in ("0.72", "0.76"):
    source = np.asarray(Image.open(HERE / f"opacity_adaptive_{opacity}.png").convert("RGB"))
    for width in (1, 2):
        mask = np.asarray(Image.open(HERE / f"edge_guided_mask_{width}.png").convert("L"))
        restored = cv2.inpaint(source[:,:,::-1].copy(),mask,2,cv2.INPAINT_TELEA)[:,:,::-1]
        result = source.copy()
        result[mask > 0] = restored[mask > 0]
        Image.fromarray(result).save(HERE / f"opacity_adaptive_{opacity}_repair_{width}.png")
