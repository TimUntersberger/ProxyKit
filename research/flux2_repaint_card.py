"""Research-only FLUX.2 repaint of an existing source-repaired card.

The model receives one 640x512 crop centered on the fitted watermark mask.
Only the fitted glyph plus a 10-pixel border is pasted into the card.
"""

import argparse
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import Flux2KleinPipeline
from sdnq import SDNQConfig  # Registers the checkpoint's 4-bit loader.


HERE = Path(__file__).resolve().parent
UPSCAYL = Path(r"C:\Program Files\Upscayl\resources\bin\upscayl-bin.exe")
MODELS = Path(r"C:\Program Files\Upscayl\resources\models")
PROMPT = (
    "Remove the faint translucent SAMPLE watermark remnants from this "
    "illustrated trading card. Reconstruct the existing artwork across the "
    "mark by continuing nearby shapes, outlines, colors, shading, and texture. "
    "Keep the characters' anatomy, objects, line style, composition, and "
    "legitimate printed card text faithful to the reference."
)
PAIRED_PROMPT = (
    "Edit the first image to remove faint translucent SAMPLE watermark remnants. "
    "The second image is a guide with the letter-edge residue softened. "
    "Use the first image for exact objects, anatomy, outlines, colors, and "
    "legitimate printed text; use the second only to identify residue to remove. "
    "Continue nearby artwork through those marks without adding or removing objects."
)


def condition_reference(repaired: np.ndarray, glyph: np.ndarray,
                        mode: str) -> np.ndarray:
    if mode == "repaired":
        return repaired
    if mode == "edge-inpaint":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        outer = cv2.dilate(glyph.astype(np.uint8), kernel) > 0
        inner = cv2.erode(glyph.astype(np.uint8), kernel) > 0
        band = np.uint8(outer & ~inner) * 255
        filled = cv2.inpaint(repaired, band, 5, cv2.INPAINT_TELEA)
        alpha = cv2.GaussianBlur(band.astype(np.float32) / 255, (0, 0), 2)
    elif mode == "band-blur":
        ys, xs = np.where(glyph)
        band = np.zeros(glyph.shape, np.float32)
        margin = 8
        band[max(0, ys.min() - margin):min(glyph.shape[0], ys.max() + margin + 1),
             max(0, xs.min() - margin):min(glyph.shape[1], xs.max() + margin + 1)] = 1
        filled = cv2.GaussianBlur(repaired, (0, 0), 10)
        alpha = cv2.GaussianBlur(band, (0, 0), 5)
    else:
        raise ValueError(f"Unknown conditioning mode: {mode}")
    return np.uint8(np.clip(
        repaired.astype(np.float32) * (1 - alpha[..., None])
        + filled.astype(np.float32) * alpha[..., None], 0, 255,
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repaired", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--baseline-2x", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--conditioning-mode",
                        choices=("repaired", "edge-inpaint", "band-blur",
                                 "paired-edge-inpaint"),
                        default="repaired")
    args = parser.parse_args()

    repaired = np.asarray(Image.open(args.repaired).convert("RGB"))
    glyph = np.asarray(Image.open(args.mask).convert("L")) > 0
    baseline = np.asarray(Image.open(args.baseline_2x).convert("RGB"))
    height, width = repaired.shape[:2]
    if glyph.shape != (height, width) or baseline.shape[:2] != (height * 2, width * 2):
        parser.error("Repaired image, mask, and 2x baseline dimensions do not match")
    if height < 512 or width > 640:
        parser.error("This probe expects an image at least 512 high and at most 640 wide")

    ys, _ = np.where(glyph)
    if len(ys) == 0:
        parser.error("Watermark mask is empty")
    y0 = int(np.clip(round((ys.min() + ys.max() + 1) / 2 - 256), 0, height - 512))
    left_pad = (640 - width) // 2
    right_pad = 640 - width - left_pad
    paired = args.conditioning_mode == "paired-edge-inpaint"
    conditioned = condition_reference(
        repaired, glyph, "edge-inpaint" if paired else args.conditioning_mode)
    crop = np.pad(conditioned[y0:y0 + 512],
                  ((0, 0), (left_pad, right_pad), (0, 0)), mode="edge")
    if paired:
        faithful_crop = np.pad(repaired[y0:y0 + 512],
                               ((0, 0), (left_pad, right_pad), (0, 0)),
                               mode="edge")
        references = [Image.fromarray(faithful_crop), Image.fromarray(crop)]
    else:
        references = Image.fromarray(crop)

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_path = prefix.with_name(prefix.name + "_raw_crop.png")
    small_path = prefix.with_name(prefix.name + "_repaint_1x.png")
    upscaled_path = prefix.with_name(prefix.name + "_repaint_upscaled_2x.png")
    output_path = prefix.with_name(prefix.name + "_full_2x.png")
    condition_path = prefix.with_name(prefix.name + "_condition_crop.png")
    for path in (raw_path, small_path, upscaled_path, output_path, condition_path):
        if path.exists():
            parser.error(f"Output already exists: {path}")

    started = time.perf_counter()
    if args.conditioning_mode != "repaired":
        Image.fromarray(crop).save(condition_path)
    print("Loading Klein 4B", "crop", (left_pad, y0, 640, y0 + 512), flush=True)
    pipe = Flux2KleinPipeline.from_pretrained(
        HERE / "models/flux2-klein-4b-sdnq", torch_dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True,
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        generated = pipe(
            image=references, prompt=PAIRED_PROMPT if paired else PROMPT,
            height=512, width=640,
            num_inference_steps=4, guidance_scale=1.0,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images[0]
    generated.save(raw_path)
    raw = np.asarray(generated.convert("RGB"))[:, left_pad:left_pad + width]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    selection = cv2.dilate(glyph.astype(np.uint8), kernel) > 0
    distance = cv2.distanceTransform(selection.astype(np.uint8), cv2.DIST_L2, 3)
    weight = np.clip(distance / 12, 0, 1) * selection
    candidate = repaired.copy()
    candidate[y0:y0 + 512] = raw
    small = np.uint8(np.clip(
        repaired.astype(np.float32) * (1 - weight[..., None])
        + candidate.astype(np.float32) * weight[..., None], 0, 255,
    ))
    small[~selection] = repaired[~selection]
    Image.fromarray(small).save(small_path)

    subprocess.run([
        str(UPSCAYL), "-i", str(small_path), "-o", str(upscaled_path),
        "-m", str(MODELS), "-n", "upscayl-standard-4x", "-s", "2", "-f", "png",
    ], check=True, stdout=subprocess.DEVNULL)
    upscaled = np.asarray(Image.open(upscaled_path).convert("RGB"))
    selection_2x = cv2.resize(selection.astype(np.uint8),
                              (baseline.shape[1], baseline.shape[0]),
                              interpolation=cv2.INTER_NEAREST) > 0
    distance_2x = cv2.distanceTransform(selection_2x.astype(np.uint8), cv2.DIST_L2, 3)
    weight_2x = np.clip(distance_2x / 16, 0, 1) * selection_2x
    full = np.uint8(np.clip(
        baseline.astype(np.float32) * (1 - weight_2x[..., None])
        + upscaled.astype(np.float32) * weight_2x[..., None], 0, 255,
    ))
    full[~selection_2x] = baseline[~selection_2x]
    Image.fromarray(full).save(output_path)
    print(output_path, "seconds", round(time.perf_counter() - started, 1),
          "selected_1x", int(selection.sum()),
          "changed_2x", int(np.count_nonzero(np.any(full != baseline, axis=2))),
          "outside", int(np.count_nonzero(np.any(full != baseline, axis=2) & ~selection_2x)),
          "peak_GiB", round(torch.cuda.max_memory_reserved() / 2**30, 2), flush=True)


if __name__ == "__main__":
    main()
