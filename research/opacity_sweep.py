"""Sweep stronger white-overlay inversion with the aligned shared letter shape."""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).parent
source = np.asarray(Image.open(Path(r"C:\Users\timun\Downloads\OP18-085.webp")).convert("RGB"), dtype=np.float32)
shared = Image.open(HERE / "shared_edge_binary.png").convert("L")
shifted = Image.new("L", shared.size, 0)
shifted.paste(shared, (0, 3))
mask = np.asarray(shifted.filter(ImageFilter.GaussianBlur(.8)), dtype=np.float32) / 255
inside = np.asarray(shifted) > 0
tiles = []
for amount in (.60, .65, .70, .75, .80):
    alpha = np.clip(mask * amount, 0, .95)
    raw = (source - alpha[:,:,None] * 255) / (1 - alpha[:,:,None])
    clipped = np.clip(raw, 0, 255).astype("uint8")
    Image.fromarray(clipped).save(HERE / f"opacity_{amount:.2f}.png")
    clip_percent = float(np.any(raw[inside] < 0, axis=1).mean() * 100)
    print(amount, "masked pixels with clipped channel", round(clip_percent, 1), "%")
    crop = Image.fromarray(clipped).crop((45,345,555,500)).resize((1020,310))
    ImageDraw.Draw(crop).text((8,8),f"alpha {amount:.2f}; clipped {clip_percent:.1f}%",fill=(255,255,0))
    tiles.append(crop)
montage = Image.new("RGB", (1020,310*len(tiles)))
for i,tile in enumerate(tiles):
    montage.paste(tile,(0,i*310))
montage.save(HERE / "opacity_sweep.png")

# An exploratory pixel-wise opacity cap: a white overlay cannot contribute
# more than the darkest observed color channel without clipping that channel.
for peak in (.72, .76, .80):
    cap = np.clip((source.min(axis=2) - 8) / 255, 0, .9)
    alpha = np.minimum(mask * peak, cap)
    raw = (source - alpha[:,:,None] * 255) / (1-alpha[:,:,None])
    Image.fromarray(np.clip(raw,0,255).astype("uint8")).save(HERE / f"opacity_adaptive_{peak:.2f}.png")
