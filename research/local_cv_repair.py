"""Local classical inpainting test on the shared-template inverse."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "deps"))
import cv2
import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).parent
source = np.asarray(Image.open(HERE / "shared_shift_+0_+2.png").convert("RGB"), dtype=np.uint8)
alpha = Image.open(HERE / "shared_alpha_matte.png").convert("L")
binary = alpha.point(lambda x: 255 if x > 64 else 0)
shifted = Image.new("L", binary.size, 0)
shifted.paste(binary, (0, 2))

for width in (3, 5, 9):
    outer = shifted.filter(ImageFilter.MaxFilter(width))
    inner = shifted.filter(ImageFilter.MinFilter(width))
    mask = np.where(np.asarray(outer) > np.asarray(inner), 255, 0).astype("uint8")
    Image.fromarray(mask).save(HERE / f"cv_ring_{width}.png")
    for name, method in (("telea", cv2.INPAINT_TELEA), ("ns", cv2.INPAINT_NS)):
        result = cv2.inpaint(source[:,:,::-1].copy(), mask, 3, method)[:,:,::-1]
        composite = source.copy()
        composite[mask > 0] = result[mask > 0]
        Image.fromarray(composite).save(HERE / f"cv_{name}_ring_{width}.png")
        print(name,width,"masked",int((mask>0).sum()),"outside_changed",int(np.any(source[mask==0]!=composite[mask==0])))
