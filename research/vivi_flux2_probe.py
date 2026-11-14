"""Compare local FLUX.2 Klein editing with the established Vivi outline repair.

Research-only Python script; the planned desktop app does not use Python.
"""

import argparse
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import Flux2KleinPipeline
from sdnq import SDNQConfig  # Registers SDNQ loading in Diffusers and Transformers.


HERE = Path(__file__).resolve().parent
BOX = (0, 190, 640, 702)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-reference", action="store_true")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--source", choices=("repaired", "original"), default="repaired")
    parser.add_argument("--generic-prompt", action="store_true")
    args = parser.parse_args()

    model = HERE / "models/flux2-klein-4b-sdnq"
    if args.source == "original":
        crop = Image.open(r"C:\Users\timun\Downloads\batch_OP13-012_SP.png").convert("RGB").crop(BOX)
    else:
        crop = Image.open(HERE / "vivi_sd15_full_width_input_640x512.png").convert("RGB")
    mask = Image.open(HERE / "vivi_sd15_outline10_mask_640x512.png").convert("L")
    mask_pixels = np.asarray(mask) > 0
    base = np.asarray(Image.open(
        HERE / "runs/batch_OP13-012_SP_eb9b91ee/repaired_palette_1x.png"
    ).convert("RGB"))

    if args.mask_reference:
        references = [crop, mask.convert("RGB")]
        prompt = (
            "Edit the first image. The second image is an edit map: white areas "
            "mark the only places to repaint; black areas indicate original "
            "artwork to retain. Remove the faint translucent SAMPLE lettering "
            "and blurred letter contours through the center. Continue the "
            "existing gold scroll lines, flower ornament, pink dress folds, "
            "and hand precisely across those narrow regions. Keep the card's "
            "composition, shapes, colors, and printed text faithful to the first image."
        )
        suffix = "maskref"
    else:
        references = crop
        if args.source == "original":
            prompt = (
                "Remove the large translucent white SAMPLE watermark across the "
                "middle of this illustrated card. Reconstruct the gold scrollwork, "
                "flower ornament, pink dress folds, hand, and blue background as "
                "they continue behind the letters. Match the original flat colors "
                "and line art. Preserve the composition, character anatomy, and "
                "printed card text."
            )
        else:
            prompt = (
                "Remove the faint translucent SAMPLE watermark remnants and blurred "
                "letter outlines across the middle of this illustrated card. "
                "Reconnect the existing gold scrollwork and flower ornament, and "
                "continue the pink dress folds and hand lines faithfully. Preserve "
                "the original composition, character anatomy, flat colors, and "
                "printed card text."
            )
        suffix = "single"
    if args.source == "original":
        suffix += "_original"
    if args.generic_prompt:
        prompt = (
            "Remove the remaining translucent watermark marks from this image. "
            "Reconstruct the obscured content by continuing nearby shapes, "
            "contours, colors, and texture. Keep the image's existing style, "
            "subject anatomy, layout, and legitimate printed details faithful "
            "to the reference."
        )
        suffix += "_generic"
    if args.steps != 4:
        suffix += f"_steps{args.steps}"
    if args.seed != 101:
        suffix += f"_seed{args.seed}"

    started = time.perf_counter()
    print("Loading FLUX.2 Klein 4B SDNQ 4-bit", flush=True)
    pipe = Flux2KleinPipeline.from_pretrained(
        model, torch_dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=True,
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    print("Generating", round(time.perf_counter() - started, 1), "seconds after start", flush=True)
    with torch.inference_mode():
        generated = pipe(
            image=references, prompt=prompt, height=512, width=640,
            num_inference_steps=args.steps, guidance_scale=1.0,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images[0]

    raw_path = HERE / f"vivi_flux2_klein_{suffix}_raw.png"
    generated.save(raw_path)
    if generated.size != crop.size:
        generated = generated.resize(crop.size, Image.Resampling.LANCZOS)

    edit = base.copy()
    region = edit[BOX[1]:BOX[3], BOX[0]:BOX[2]]
    region[mask_pixels] = np.asarray(generated)[mask_pixels]
    selection = np.zeros(base.shape[:2], dtype=bool)
    selection[BOX[1]:BOX[3], BOX[0]:BOX[2]] = mask_pixels
    distance = cv2.distanceTransform(selection.astype(np.uint8), cv2.DIST_L2, 3)
    weight = np.clip(distance / 5, 0, 1) * selection
    result = np.uint8(np.clip(
        base.astype(np.float32) * (1 - weight[..., None])
        + edit.astype(np.float32) * weight[..., None], 0, 255,
    ))
    result[~selection] = base[~selection]
    result_path = HERE / f"vivi_flux2_klein_{suffix}_1x.png"
    Image.fromarray(result).save(result_path)
    print(raw_path.name, result_path.name,
          "seconds", round(time.perf_counter() - started, 1),
          "outside_mask", int(np.count_nonzero(np.any(result != base, axis=2) & ~selection)),
          "peak_reserved_GiB", round(torch.cuda.max_memory_reserved() / 2**30, 2),
          flush=True)


if __name__ == "__main__":
    main()
