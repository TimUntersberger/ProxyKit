"""Extract reviewable card-effect OCR lines from a detection manifest.

This does not rewrite or correct OCR text. The source image and per-line
confidence remain in the output so mistakes can be reviewed in the UI.
"""

import argparse
import json
import re
import statistics
from pathlib import Path

from PIL import Image


def bounds(polygon):
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def extract(manifest, top_fraction=0.55, bottom_fraction=0.84):
    source = Path(manifest["source"])
    with Image.open(source) as image:
        width, height = image.size
    lines = []
    for region in manifest["regions"]:
        if region.get("kind") != "recognized" or region.get("rotation") != 0:
            continue
        value = (region.get("text") or "").strip()
        if not value:
            continue
        left, top, right, bottom = bounds(region["polygon"])
        center = (top + bottom) / 2
        if not top_fraction * height <= center <= bottom_fraction * height:
            continue
        if left > width * 0.9:
            continue
        lines.append({
            "text": value,
            "confidence": region.get("score"),
            "box": [left, top, right, bottom],
        })
    lines.sort(key=lambda line: ((line["box"][1] + line["box"][3]) / 2,
                                 line["box"][0]))
    body_heights = [line["box"][3] - line["box"][1] for line in lines
                    if len(line["text"]) >= 8 and
                    line["box"][2] - line["box"][0] >= width * 0.2]
    typical_height = statistics.median(body_heights) if body_heights else 0
    for line in lines:
        line_height = line["box"][3] - line["box"][1]
        japanese_kana = any("\u3040" <= char <= "\u30ff" for char in line["text"])
        if re.fullmatch(r"\d", line["text"]):
            line["role"] = "uncertain"
        elif line["text"].strip("[]【】〈〉<> ") in {"トリガー", "KO時", "K.O.時", "登場時", "起動メイン", "アタック時"}:
            line["role"] = "effect"
        elif japanese_kana and len(line["text"]) < 12 and typical_height and line_height < typical_height * 0.65:
            line["role"] = "ruby"
        else:
            line["role"] = "effect"
    rows = []
    for line in lines:
        if line["role"] != "effect":
            continue
        center = (line["box"][1] + line["box"][3]) / 2
        line_height = line["box"][3] - line["box"][1]
        if rows and abs(center - rows[-1]["center"]) <= min(line_height, rows[-1]["height"]) * 0.45:
            rows[-1]["parts"].append(line)
        else:
            rows.append({"center": center, "height": line_height, "parts": [line]})
    draft_rows = []
    for row in rows:
        parts = sorted(row["parts"], key=lambda part: part["box"][0])
        draft_rows.append(" ".join(part["text"] for part in parts))
    return {
        "source": str(source),
        "image_size": [width, height],
        "effect_zone": [top_fraction, bottom_fraction],
        "lines": lines,
        "draft_text": "\n".join(draft_rows),
        "review_required": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top", type=float, default=0.55)
    parser.add_argument("--bottom", type=float, default=0.84)
    args = parser.parse_args()
    if not 0 <= args.top < args.bottom <= 1:
        parser.error("effect-zone fractions must satisfy 0 <= top < bottom <= 1")
    data = extract(json.loads(args.manifest.read_text(encoding="utf-8")),
                   args.top, args.bottom)
    output = args.output or args.manifest.with_name(args.manifest.stem + "_effects.json")
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(data['lines'])} effect lines: {output}")
    print(data["draft_text"])


if __name__ == "__main__":
    main()
