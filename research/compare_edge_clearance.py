"""Build review crops for the baseline, edge-gradient, and clearance variants."""
from pathlib import Path
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
cases = {
    "cerberus": (
        ["OP18-085_remove_then_upscale_composite.png",
         "OP18-085_edge_gradient_035_composite.png",
         "OP18-085_edge_clear_05_composite.png"],
        (0, 680, 1200, 1100),
    ),
    "vivi": (
        ["batch_OP13-012_SP_remove_then_upscale.png",
         "batch_OP13-012_SP_edge_035_composite.png",
         "batch_OP13-012_SP_edge_clear_05_composite.png"],
        (0, 690, 1280, 1110),
    ),
}
for label, (files, box) in cases.items():
    width, height = box[2] - box[0], box[3] - box[1]
    sheet = Image.new("RGB", (3 * width, height + 25), "white")
    draw = ImageDraw.Draw(sheet)
    for i, filename in enumerate(files):
        sheet.paste(Image.open(HERE / filename).convert("RGB").crop(box),
                    (i * width, 25))
        draw.text((i * width + 8, 5), filename, fill="black")
    sheet.save(HERE / f"edge_clear_05_{label}_2x_comparison.png")
