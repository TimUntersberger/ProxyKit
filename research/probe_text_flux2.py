"""Research probe: local FLUX.2 Klein on a detected rules-text crop."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import Flux2KleinPipeline
from sdnq import SDNQConfig  # noqa: F401 - registers quantized loading


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("detections", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--single-image", action="store_true", help="Prompt from the image alone, without a mask reference")
    args = parser.parse_args()
    base = np.asarray(Image.open(args.image).convert("RGB"))
    mask = np.asarray(Image.open(args.mask).convert("L"))
    if mask.shape != base.shape[:2]:
        parser.error("Mask dimensions do not match image")
    regions = json.loads(args.detections.read_text(encoding="utf-8"))["regions"]
    longest = max(regions, key=lambda item: len(item.get("text") or ""))
    points = np.asarray(longest["polygon"])
    height, width = base.shape[:2]
    x = int(np.clip(points[:, 0].mean() - 320, 0, width - 640))
    y = int(np.clip(points[:, 1].mean() - 256, 0, height - 512))
    crop = Image.fromarray(base[y:y + 512, x:x + 640])
    crop_mask = Image.fromarray(mask[y:y + 512, x:x + 640])
    model = Path(__file__).resolve().parent / "models" / "flux2-klein-4b-sdnq"
    print(f"Crop: x={x}, y={y}, 640x512. Loading FLUX.2 Klein", flush=True)
    pipe = Flux2KleinPipeline.from_pretrained(
        model, torch_dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=True,
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    if args.single_image:
        prompt = (
            "Remove all printed words and letters crossing the lower half of this "
            "illustrated card crop. Show the underlying uninterrupted pink dress, "
            "blue background, gold ornament, and pale cloud. No lettering or "
            "typography should remain in those four horizontal rows. Preserve "
            "the existing illustration and its colors."
        )
        references = crop
    else:
        prompt = (
            "Edit the first image. The second image is an edit map: white marks "
            "the printed rules text to erase. Remove the small black-and-white "
            "letters and their outlines. Continue the existing pink dress folds, "
            "blue background, flower patterns, gold lines, and pale cloud shapes "
            "under the letters. Preserve all other card artwork faithfully. "
            "Do not add any new text."
        )
        references = [crop, crop_mask.convert("RGB")]
    with torch.inference_mode():
        generated = pipe(
            image=references, prompt=prompt,
            height=512, width=640, num_inference_steps=4, guidance_scale=1.0,
            generator=torch.Generator(device="cuda").manual_seed(101),
        ).images[0]
    raw_path = args.output.with_name(args.output.stem + "_raw_crop.png")
    generated.save(raw_path)
    crop_mask.save(args.output.with_name(args.output.stem + "_mask_crop.png"))
    candidate = np.asarray(generated)
    result = base.copy()
    selection = mask[y:y + 512, x:x + 640] > 0
    distance = cv2.distanceTransform(selection.astype(np.uint8), cv2.DIST_L2, 3)
    weight = np.clip(distance / 4, 0, 1)[..., None]
    original = result[y:y + 512, x:x + 640].astype(np.float32)
    blended = np.clip(original * (1 - weight) + candidate * weight, 0, 255).astype(np.uint8)
    result[y:y + 512, x:x + 640][selection] = blended[selection]
    Image.fromarray(result).save(args.output)
    print(f"Saved {args.output} and {raw_path}", flush=True)


if __name__ == "__main__":
    main()
