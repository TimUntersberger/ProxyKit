"""Research-only FLUX text removal over automatically grouped OCR regions."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import Flux2KleinPipeline
from sdnq import SDNQConfig  # noqa: F401 - registers quantized loading


def limits(region: dict) -> tuple[int, int, int, int]:
    points = np.asarray(region["polygon"])
    return int(points[:, 0].min()), int(points[:, 1].min()), int(points[:, 0].max()), int(points[:, 1].max())


def group_regions(regions: list[dict]) -> list[list[dict]]:
    groups = []
    for region in sorted(regions, key=lambda item: limits(item)[1]):
        box = limits(region)
        if groups and box[1] - max(limits(item)[3] for item in groups[-1]) < 110:
            groups[-1].append(region)
        else:
            groups.append([region])
    return groups


def crop_positions(group: list[dict], width: int, height: int) -> list[tuple[int, int]]:
    x0 = min(limits(item)[0] for item in group)
    x1 = max(limits(item)[2] for item in group)
    y0 = min(limits(item)[1] for item in group)
    y1 = max(limits(item)[3] for item in group)
    if y1 - y0 <= 440:
        ys = [int(np.clip((y0 + y1) / 2 - 256, 0, height - 512))]
    else:
        first_y = int(np.clip(y0 - 40, 0, height - 512))
        last_y = int(np.clip(y1 - 472, 0, height - 512))
        count_y = max(2, math.ceil((last_y - first_y) / 260) + 1)
        ys = [int(round(value)) for value in np.linspace(first_y, last_y, count_y)]
    if x1 - x0 <= 560:
        x = int(np.clip((x0 + x1) / 2 - 320, 0, width - 640))
        return [(x, y) for y in ys]
    first = int(np.clip(x0 - 45, 0, width - 640))
    last = int(np.clip(x1 - 595, 0, width - 640))
    n = max(2, math.ceil((last - first) / 320) + 1)
    return [(int(round(x)), y) for y in ys for x in np.linspace(first, last, n)]


def split_simple_background_regions(source: np.ndarray, regions: list[dict]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Estimate smooth card graphics from nearby clean pixels."""
    height, width = source.shape[:2]
    fill_mask = np.zeros((height, width), dtype=np.uint8)
    fill_rgb = np.zeros(source.shape, dtype=np.uint8)
    remaining = []
    for region in regions:
        polygon = np.asarray(region["polygon"], dtype=np.int32)
        text = (region.get("text") or "").strip()
        x0, y0, x1, y1 = limits(region)
        compact_top_emblem = (len(text) < 8 and y0 < height * 0.15
                              and not (text.isdigit() and (x1 - x0) * (y1 - y0) > 2000))
        local = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(local, [polygon], 1)
        ring = (cv2.dilate(local, np.ones((31, 31), dtype=np.uint8)) != 0) & (local == 0)
        surrounding = source[ring]
        white_fraction = float(np.mean(np.min(surrounding, axis=1) >= 235)) if len(surrounding) else 0.0
        if compact_top_emblem and white_fraction < 0.90:
            print(f"Preserving compact top emblem: {text}", flush=True)
            continue
        if len(text) < 8 and (x1 - x0) * (y1 - y0) < 1500 and white_fraction < 0.90:
            print(f"Preserving small detailed label: {text}", flush=True)
            continue
        if len(surrounding) < 50:
            remaining.append(region)
            continue
        median = np.median(surrounding, axis=0)
        mad = np.median(np.abs(surrounding - median), axis=0)
        close_fraction = float(np.mean(np.max(np.abs(surrounding - median), axis=1) < 25))
        simple = (white_fraction >= 0.80 or
                  (np.min(median) > 220 and np.max(mad) < 8 and close_fraction > 0.55) or
                  (np.max(mad) < 15 and close_fraction > 0.70) or
                  (np.mean(median) < 160 and np.max(mad) < 25 and
                   close_fraction > 0.58 and y1 - y0 < 40))
        if not simple and y0 < height * 0.15 and text.isdigit():
            print(f"Preserving power number over detailed artwork: {text}", flush=True)
            continue
        if not simple:
            remaining.append(region)
            continue
        expanded = cv2.dilate(local, np.ones((9, 9), dtype=np.uint8)) != 0
        # Fit a color plane to clean surrounding pixels. This retains gentle
        # gradients across nameplates without copying letters into the fill.
        clean = ring & (np.max(np.abs(source.astype(np.float32) - median), axis=2) < 35)
        yy, xx = np.nonzero(clean)
        if len(xx) >= 30:
            design = np.stack((xx / max(width, 1), yy / max(height, 1), np.ones_like(xx)), axis=1)
            coefficients = np.linalg.lstsq(design, source[clean].astype(np.float32), rcond=None)[0]
            fy, fx = np.nonzero(expanded)
            fill = np.stack((fx / max(width, 1), fy / max(height, 1), np.ones_like(fx)), axis=1) @ coefficients
            fill_rgb[fy, fx] = np.clip(fill, 0, 255).astype(np.uint8)
        else:
            fill_rgb[expanded] = np.clip(median, 0, 255).astype(np.uint8)
        fill_mask[expanded] = 1
        print(f"Smooth background fill: {text}", flush=True)
    return remaining, fill_mask, fill_rgb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("detections", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = np.asarray(Image.open(args.image).convert("RGB"))
    original_height, original_width = source.shape[:2]
    if original_width < 640 or original_height < 512:
        source = cv2.copyMakeBorder(source, 0, max(0, 512 - original_height),
                                    0, max(0, 640 - original_width), cv2.BORDER_REPLICATE)
    height, width = source.shape[:2]
    regions = json.loads(args.detections.read_text(encoding="utf-8"))["regions"]
    if not regions:
        parser.error("No high-confidence text was detected. Try a clearer or larger source image.")
    model_regions, simple_mask, simple_rgb = split_simple_background_regions(source, regions)
    groups = group_regions(model_regions)
    jobs = [(index, group, x, y)
            for index, group in enumerate(groups)
            for x, y in crop_positions(group, width, height)]
    print(f"{len(regions)} OCR regions, {len(groups)} repaint bands, {len(jobs)} model crops", flush=True)
    pipe = None
    if jobs:
        model = Path(__file__).resolve().parent / "models" / "flux2-klein-4b-sdnq"
        pipe = Flux2KleinPipeline.from_pretrained(model, torch_dtype=torch.bfloat16,
                                                   local_files_only=True, low_cpu_mem_usage=True)
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
    sums = np.zeros(source.shape, dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    full_mask = np.zeros((height, width), dtype=np.uint8)
    output_dir = args.output.with_name(args.output.stem + "_crops")
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = (
        "Remove all printed words, letters, numerals, and their outlines from "
        "this illustrated card crop. Reconstruct the uninterrupted underlying "
        "illustration, clothing, color fields, cloud shapes, ornamental lines, "
        "and border pattern. Keep the original composition and flat-color art "
        "style. No text or lettering should remain."
    )
    for job_index, (group_index, group, x, y) in enumerate(jobs):
        group_mask = np.zeros((height, width), dtype=np.uint8)
        for region in group:
            cv2.fillPoly(group_mask, [np.asarray(region["polygon"], np.int32)], 255)
        group_mask = cv2.dilate(group_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        full_mask = np.maximum(full_mask, group_mask)
        crop = Image.fromarray(source[y:y + 512, x:x + 640])
        print(f"Crop {job_index + 1}/{len(jobs)}: band {group_index + 1}, x={x}, y={y}", flush=True)
        with torch.inference_mode():
            generated = pipe(
                image=crop, prompt=prompt, height=512, width=640,
                num_inference_steps=4, guidance_scale=1.0,
                generator=torch.Generator(device="cuda").manual_seed(101 + group_index),
            ).images[0]
        generated.save(output_dir / f"crop_{job_index + 1:02d}.png")
        prediction = np.asarray(generated, dtype=np.float32)
        selected = group_mask[y:y + 512, x:x + 640] > 0
        # Prefer the middle of overlapping crops, where the model sees context
        # on both sides of a text line.
        xx = np.minimum(np.arange(640), np.arange(639, -1, -1))
        yy = np.minimum(np.arange(512), np.arange(511, -1, -1))
        center = (0.2 + np.minimum(xx[None, :], yy[:, None]) / 200).clip(0.2, 1)
        weight = (selected * center).astype(np.float32)[..., None]
        sums[y:y + 512, x:x + 640] += prediction * weight
        weights[y:y + 512, x:x + 640] += weight
    candidate = np.clip(sums / np.maximum(weights, 1e-6), 0, 255)
    active = (full_mask > 0) & (weights[:, :, 0] > 0)
    distance = cv2.distanceTransform(active.astype(np.uint8), cv2.DIST_L2, 3)
    blend = np.clip(distance / 6, 0, 1)[..., None]
    result = source.copy()
    mixed = np.uint8(np.clip(source * (1 - blend) + candidate * blend, 0, 255))
    result[active] = mixed[active]
    simple_distance = cv2.distanceTransform(simple_mask, cv2.DIST_L2, 3)
    simple_blend = np.clip(simple_distance / 4, 0, 1)[..., None]
    simple_mixed = np.uint8(np.clip(result * (1 - simple_blend) + simple_rgb * simple_blend, 0, 255))
    result[simple_mask != 0] = simple_mixed[simple_mask != 0]
    full_mask = np.maximum(full_mask, simple_mask * 255)
    Image.fromarray(result[:original_height, :original_width]).save(args.output)
    Image.fromarray(full_mask[:original_height, :original_width]).save(args.output.with_name(args.output.stem + "_mask.png"))
    print(f"Saved {args.output}; outside-mask changes: "
          f"{np.count_nonzero(np.any(result != source, axis=2) & (full_mask == 0))}", flush=True)


if __name__ == "__main__":
    main()
