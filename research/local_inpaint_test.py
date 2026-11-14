"""Local, mask-constrained LaMa feasibility test. Not application code."""

from pathlib import Path
from time import perf_counter
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).parent / "deps"))
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter

HERE = Path(__file__).parent
MODEL = HERE / "models" / "inpainting_lama_2025jan.onnx"
SOURCE = Path(r"C:\Users\timun\Downloads\OP18-085.webp")
MASK = HERE / "approx_mask.png"
INVERSE = HERE / "inverse_0.65.png"
EXPECTED_SHA256 = "7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af81841b40adcf74fdf2"

digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
if digest != EXPECTED_SHA256:
    raise RuntimeError(f"Unexpected model SHA256: {digest}")

options = ort.SessionOptions()
options.log_severity_level = 3
options.intra_op_num_threads = 4
session = ort.InferenceSession(str(MODEL), sess_options=options, providers=["CPUExecutionProvider"])

base_mask = Image.open(MASK).convert("L")
full_mask = base_mask.point(lambda x: 255 if x > 12 else 0).filter(ImageFilter.MaxFilter(11))
inner_mask = base_mask.point(lambda x: 255 if x > 12 else 0).filter(ImageFilter.MinFilter(9))
ring_mask = Image.fromarray(
    np.where(np.asarray(full_mask) > np.asarray(inner_mask), 255, 0).astype("uint8")
)
shared_alpha = Image.open(HERE / "shared_alpha_matte.png").convert("L")
shared_binary = shared_alpha.point(lambda x: 255 if x > 64 else 0)
shifted_shared_binary = Image.new("L", shared_binary.size, 0)
shifted_shared_binary.paste(shared_binary, (0, 2))
shared_outer = shifted_shared_binary.filter(ImageFilter.MaxFilter(11))
shared_inner = shifted_shared_binary.filter(ImageFilter.MinFilter(9))
shared_ring = Image.fromarray(np.where(np.asarray(shared_outer) > np.asarray(shared_inner), 255, 0).astype("uint8"))

# A 512x512 crop preserves the original pixel scale for the fixed-size ONNX model.
left, top = 44, 150
box = (left, top, left + 512, top + 512)

for label, source_path, mask_image in (
    ("direct", SOURCE, full_mask),
    ("hybrid", INVERSE, ring_mask),
    ("shared_hybrid", HERE / "shared_shift_+0_+2.png", shared_ring),
):
    source = Image.open(source_path).convert("RGB")
    crop = np.asarray(source.crop(box), dtype=np.uint8)
    crop_mask = np.asarray(mask_image.crop(box), dtype=np.uint8) > 0
    image_input = np.transpose(crop[:, :, ::-1].astype("float32") / 255.0, (2, 0, 1))[None]
    mask_input = crop_mask.astype("float32")[None, None]
    start = perf_counter()
    output = session.run(None, {"image": image_input, "mask": mask_input})[0]
    elapsed = perf_counter() - start
    print(label, "inference_seconds", round(elapsed, 2), "model_output_range", (float(output.min()), float(output.max())))
    restored_crop = np.clip(np.transpose(output[0], (1, 2, 0))[:, :, ::-1], 0, 255).astype("uint8")

    # The app would retain original pixels everywhere the mask is zero.
    composite = crop.copy()
    composite[crop_mask] = restored_crop[crop_mask]
    result = source.copy()
    result.paste(Image.fromarray(composite), (left, top))
    result.save(HERE / f"local_lama_{label}.png")
    mask_image.save(HERE / f"local_lama_{label}_mask.png")
    original_pixels = np.asarray(source)
    result_pixels = np.asarray(result)
    full_mask_array = np.asarray(mask_image) > 0
    print(label, "masked_pixels", int(full_mask_array.sum()), "outside_mask_changed", int(np.any(original_pixels[~full_mask_array] != result_pixels[~full_mask_array])))
