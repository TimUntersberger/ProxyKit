"""Simple multi-image estimate of alpha and watermark color; research probe."""

from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).parent
ids = {"003", "010", "047", "063", "066", "085", "086", "088", "089", "091", "092", "101", "106", "112"}
files = [p for p in sorted((HERE / "samples" / "refs").glob("*.jpg")) if p.stem[-3:] in ids]
stack = np.stack([np.asarray(Image.open(p).convert("RGB").resize((600,838), Image.Resampling.LANCZOS), dtype=np.float32) for p in files])
mean = stack.mean(axis=0)
sd = stack.std(axis=0).mean(axis=2)

font = Image.open(HERE / "approx_mask.png").convert("L").point(lambda x: 255 if x > 5 else 0)
coarse = np.asarray(font.filter(ImageFilter.MaxFilter(31)), dtype=np.float32) / 255.0
valid = 1 - coarse

def weighted_blur(values, radius):
    weight = np.asarray(Image.fromarray((valid * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255
    if values.ndim == 2:
        numerator = np.asarray(Image.fromarray(np.clip(values * valid, 0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
        return numerator / np.maximum(weight, .001)
    channels = []
    for c in range(values.shape[2]):
        numerator = np.asarray(Image.fromarray(np.clip(values[:,:,c] * valid, 0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
        channels.append(numerator / np.maximum(weight, .001))
    return np.stack(channels, axis=2)

sd_clean = weighted_blur(sd, 30)
mean_clean = weighted_blur(mean, 30)
alpha = np.clip(1 - sd / np.maximum(sd_clean, 1), 0, .9) * coarse
alpha = np.asarray(Image.fromarray(np.clip(alpha * 255, 0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(.8)), dtype=np.float32) / 255
alpha[alpha < .12] = 0
alpha[:345] = 0
alpha[498:] = 0
alpha[:, :45] = 0
alpha[:, 555:] = 0

watermark = np.clip((mean - (1 - alpha[...,None]) * mean_clean) / np.maximum(alpha[...,None], .1), 0, 255)
watermark = np.asarray(Image.fromarray(watermark.astype("uint8")).filter(ImageFilter.GaussianBlur(1)), dtype=np.float32)
Image.fromarray(np.clip(alpha * 255, 0, 255).astype("uint8")).save(HERE / "shared_variance_alpha.png")
Image.fromarray(watermark.astype("uint8")).save(HERE / "shared_variance_color.png")

source = np.asarray(Image.open(Path(r"C:\Users\timun\Downloads\OP18-085.webp")).convert("RGB"), dtype=np.float32)
for gain in (.8, 1.0, 1.2):
    a = np.clip(alpha * gain, 0, .9)
    restored = np.clip((source - a[...,None] * watermark) / (1-a[...,None]), 0, 255).astype("uint8")
    Image.fromarray(restored).save(HERE / f"shared_variance_inverse_{gain:.1f}.png")
print('alpha quantiles',np.quantile(alpha[alpha>.1],[.1,.5,.9]).round(3).tolist())
