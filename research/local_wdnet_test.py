"""Run official WDNet pretrained weights locally, with no network inference."""

import ast
from pathlib import Path
from time import perf_counter
import sys
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).parent
SRC = HERE / "models" / "wdnet_src"
sys.path.insert(0, str(SRC))
import unet_parts

# The repository mixes training dependencies into WDNet.py. Load only its
# generator class so the inference probe needs no torchvision/tensorboard.
tree = ast.parse((SRC / "WDNet.py").read_text())
generator_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "generator")
namespace = {"torch": torch, "nn": torch.nn, **{k: v for k, v in vars(unet_parts).items() if not k.startswith("__")}}
exec(compile(ast.Module(body=[generator_node], type_ignores=[]), str(SRC / "WDNet.py"), "exec"), namespace)
Generator = namespace["generator"]

torch.set_num_threads(4)
model = Generator(3, 3).eval()
weights = torch.load(HERE / "models" / "wdnet" / "WDNet_G.pkl", map_location="cpu", weights_only=True)
model.load_state_dict(weights)

source = Image.open(Path(r"C:\Users\timun\Downloads\OP18-085.webp")).convert("RGB")
left, top = 44, 150
crop = np.asarray(source.crop((left, top, left + 512, top + 512)), dtype=np.uint8)
input_tensor = torch.from_numpy(np.transpose(crop.astype("float32") / 255.0, (2, 0, 1))[None])
start = perf_counter()
with torch.no_grad():
    prediction, model_mask, alpha, watermark_color, provisional = model(input_tensor)
print("inference_seconds", round(perf_counter() - start, 2))
print("predicted_mask_quantiles", np.quantile(model_mask.numpy(), [.5,.9,.99]).round(3).tolist())

pred = np.clip(np.transpose(prediction[0].numpy(), (1, 2, 0)) * 255, 0, 255).astype("uint8")
mask = np.clip(model_mask[0, 0].numpy() * 255, 0, 255).astype("uint8")
Image.fromarray(mask).save(HERE / "wdnet_predicted_mask_crop.png")
Image.fromarray(pred).save(HERE / "wdnet_prediction_crop.png")

# Restore only the card's artwork band to avoid changing rules or border text.
result = np.asarray(source).copy()
region = pred[195:350, 0:512]
result[top + 195:top + 350, left:left + 512] = region
Image.fromarray(result).save(HERE / "local_wdnet.png")
