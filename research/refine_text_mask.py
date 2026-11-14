"""Experimental automatic ink mask from OCR character regions and color enrichment."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("detections", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    data = json.loads(args.detections.read_text(encoding="utf-8"))
    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    # Coarse LAB bins collect foreground colors even with antialiasing.
    bins = ((lab[:, :, 0].astype(np.int32) // 16) * 256 +
            (lab[:, :, 1].astype(np.int32) // 16) * 16 +
            (lab[:, :, 2].astype(np.int32) // 16))
    output = np.zeros((height, width), dtype=np.uint8)
    for region in data["regions"]:
        chars = region.get("characters", [])
        if not chars:
            continue
        points = np.concatenate([np.asarray(c["polygon"], dtype=np.int32) for c in chars])
        x0 = max(0, int(points[:, 0].min()) - 14)
        x1 = min(width, int(points[:, 0].max()) + 15)
        y0 = max(0, int(points[:, 1].min()) - 14)
        y1 = min(height, int(points[:, 1].max()) + 15)
        char_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        for char in chars:
            polygon = np.asarray(char["polygon"], dtype=np.int32) - [x0, y0]
            cv2.fillPoly(char_mask, [polygon], 1)
        nearby = cv2.dilate(char_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        background = (nearby != 0) & (char_mask == 0)
        if background.sum() < 20:
            continue
        colors = bins[y0:y1, x0:x1]
        foreground_hist = np.bincount(colors[char_mask != 0], minlength=4096).astype(np.float64)
        background_hist = np.bincount(colors[background], minlength=4096).astype(np.float64)
        foreground_rate = (foreground_hist + 1) / (foreground_hist.sum() + 4096)
        background_rate = (background_hist + 1) / (background_hist.sum() + 4096)
        enriched = (foreground_rate / background_rate > 1.8) & (foreground_hist >= 5)
        selected = (enriched[colors] & (char_mask != 0)).astype(np.uint8) * 255
        # Small holes and antialiased rims belong to the same printed stroke.
        selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        selected = cv2.dilate(selected, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        output[y0:y1, x0:x1] = np.maximum(output[y0:y1, x0:x1], selected)
    Image.fromarray(output).save(args.output)
    overlay = image.copy()
    overlay[output != 0] = (255, 0, 0)
    preview = np.clip(image * 0.55 + overlay * 0.45, 0, 255).astype(np.uint8)
    preview_path = args.output.with_name(args.output.stem + "_preview.png")
    Image.fromarray(preview).save(preview_path)
    print(f"Mask pixels: {int((output != 0).sum())}; saved {args.output} and {preview_path}")


if __name__ == "__main__":
    main()
