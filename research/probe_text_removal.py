"""Experimental baseline: fill OCR polygons with the local 512px LaMa model.

This deliberately uses the OCR polygons as-is to show why a tighter text mask
matters. It is not the final text-removal algorithm.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


def starts(length: int, tile: int = 512, stride: int = 384) -> list[int]:
    values = list(range(0, max(1, length - tile + 1), stride))
    end = max(0, length - tile)
    if values[-1] != end:
        values.append(end)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("detections", type=Path, help="JSON from highlight_card_text.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--character-boxes", action="store_true", help="Use OCR character polygons instead of full line polygons")
    parser.add_argument("--mask-image", type=Path, help="Use a prepared grayscale mask instead of detection polygons")
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    height, width = image.shape[:2]
    data = json.loads(args.detections.read_text(encoding="utf-8"))
    if args.mask_image:
        mask = np.asarray(Image.open(args.mask_image).convert("L"), dtype=np.uint8)
        if mask.shape != (height, width):
            parser.error("Prepared mask dimensions do not match the image")
        mask = (mask > 127).astype(np.uint8) * 255
    else:
        mask = np.zeros((height, width), dtype=np.uint8)
        for item in data["regions"]:
            polygons = ([char["polygon"] for char in item.get("characters", [])]
                        if args.character_boxes else [item["polygon"]])
            if not polygons:
                polygons = [item["polygon"]]
            for polygon in polygons:
                cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask_path = args.output.with_name(args.output.stem + "_mask.png")
    Image.fromarray(mask).save(mask_path)

    model = Path(__file__).resolve().parent / "models" / "inpainting_lama_2025jan.onnx"
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    sums = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    ramp = np.maximum(0.1, np.hanning(512).astype(np.float32))
    window = (ramp[:, None] * ramp[None, :])[:, :, None]
    count = 0
    for y in starts(height):
        for x in starts(width):
            patch_mask = mask[y:y + 512, x:x + 512]
            if not patch_mask.any():
                continue
            patch = image[y:y + 512, x:x + 512]
            if patch.shape[:2] != (512, 512):
                raise ValueError("This baseline requires an image at least 512x512")
            inputs = {
                "image": np.transpose(patch.astype(np.float32) / 255.0, (2, 0, 1))[None],
                "mask": (patch_mask.astype(np.float32) / 255.0)[None, None],
            }
            prediction = session.run(None, inputs)[0][0].transpose(1, 2, 0)
            if count == 0:
                print(f"Raw model range: {prediction.min():.3f} to {prediction.max():.3f}", flush=True)
            if prediction.max() > 2:
                prediction /= 255.0
            sums[y:y + 512, x:x + 512] += prediction * window
            weights[y:y + 512, x:x + 512] += window
            count += 1
    candidate = np.clip(sums / np.maximum(weights, 1e-6) * 255, 0, 255).astype(np.uint8)
    result = image.copy()
    result[mask != 0] = candidate[mask != 0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(args.output)
    print(f"Processed {count} patches; saved {args.output} and {mask_path}")
    print(f"Changed pixels outside mask: {np.count_nonzero(np.any(result[mask == 0] != image[mask == 0], axis=1))}")


if __name__ == "__main__":
    main()
