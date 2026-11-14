"""Automatically locate text and draw its detected regions over a card image.

Research helper only. It does not erase text or create a repair mask.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rapidocr import RapidOCR


def rotated(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def to_original(box: np.ndarray, angle: int, scale: float, width: int, height: int) -> np.ndarray:
    points = np.asarray(box, dtype=np.float32) / scale
    x, y = points[:, 0].copy(), points[:, 1].copy()
    if angle == 90:
        points[:, 0], points[:, 1] = y, height - 1 - x
    elif angle == 180:
        points[:, 0], points[:, 1] = width - 1 - x, height - 1 - y
    elif angle == 270:
        points[:, 0], points[:, 1] = width - 1 - y, x
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return points


def bounds(points: np.ndarray) -> tuple[float, float, float, float]:
    return (float(points[:, 0].min()), float(points[:, 1].min()),
            float(points[:, 0].max()), float(points[:, 1].max()))


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = bounds(a)
    bx0, by0, bx1, by1 = bounds(b)
    common = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    return common / max(1.0, area_a + area_b - common)


def trim_character_heights(characters: list[dict], edges: np.ndarray) -> None:
    """Find a common ink band for a recognized horizontal text line."""
    # Long lines provide enough repeated letter evidence to distinguish the
    # ink band from nearby borders. Short labels and numbers do not.
    if len(characters) < 20:
        return
    height, width = edges.shape
    x0 = max(0, int(min(c["points"][:, 0].min() for c in characters)))
    y0 = max(0, int(min(c["points"][:, 1].min() for c in characters)))
    x1 = min(width, int(max(c["points"][:, 0].max() for c in characters)) + 1)
    y1 = min(height, int(max(c["points"][:, 1].max() for c in characters)) + 1)
    line_height = y1 - y0
    if line_height < 16:
        return
    region = np.zeros((line_height, x1 - x0), dtype=np.uint8)
    for char in characters:
        polygon = np.rint(char["points"] - np.array([x0, y0])).astype(np.int32)
        cv2.fillPoly(region, [polygon], 1)
    rows = ((edges[y0:y1, x0:x1] != 0) * region).sum(axis=1).astype(np.float32)
    rows = np.convolve(rows, np.array([1, 2, 3, 2, 1], np.float32) / 9, mode="same")
    cutoff = max(float(np.median(rows)) * 1.3, float(rows.max()) * 0.30)
    strong = np.flatnonzero(rows > cutoff)
    if len(strong) == 0:
        return
    margin = max(2, round(line_height * 0.06))
    top = max(y0, y0 + int(strong[0]) - margin)
    bottom = min(y1 - 1, y0 + int(strong[-1]) + margin)
    # A drastic change usually means decoration or a border dominated the
    # projection, so leave the OCR geometry alone.
    if bottom - top < line_height * 0.45 or bottom - top > line_height * 0.96:
        return
    for char in characters:
        char["points"][:, 1] = np.clip(char["points"][:, 1], top, bottom)


def detect(image: np.ndarray, threshold: float, rotations: tuple[int, ...], tight_boxes: bool) -> list[dict]:
    height, width = image.shape[:2]
    edges = cv2.Canny(image, 80, 160) if tight_boxes else None
    engine = RapidOCR(params={
        "Global.use_cls": False,
        "Global.log_level": "error",
        "Det.box_thresh": threshold,
    })
    findings = []
    # A second, larger pass helps on small source images. Large images already
    # contain enough pixels, and doubling them would mostly add cost.
    scales = (1.0, 2.0) if max(width, height) < 1200 else (1.0,)
    for scale in scales:
        scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else image
        for angle in rotations:
            view = rotated(scaled, angle)
            recognized = engine(view, use_rec=True, use_cls=False, box_thresh=threshold,
                                return_word_box=True, return_single_char_box=True)
            detected = engine(view, use_rec=False, use_cls=False, box_thresh=threshold)
            print(f"scale {scale:g}, rotation {angle:3d}: "
                  f"{len(recognized.boxes) if recognized.boxes is not None else 0} OCR, "
                  f"{len(detected.boxes) if detected.boxes is not None else 0} detector regions", flush=True)
            for result, kind in ((detected, "candidate"), (recognized, "recognized")):
                boxes = [] if result.boxes is None else result.boxes
                scores = [] if result.scores is None else result.scores
                texts = getattr(result, "txts", None)
                for i, box in enumerate(boxes):
                    points = to_original(box, angle, scale, width, height)
                    if cv2.contourArea(points.astype(np.float32)) < 8:
                        continue
                    if angle != 0:
                        x0, y0, x1, y1 = bounds(points)
                        if y1 - y0 < (x1 - x0) * 1.4:
                            continue
                    score = float(scores[i]) if i < len(scores) else None
                    is_confirmed = kind == "recognized" and score is not None and score >= 0.8
                    characters = []
                    if is_confirmed and recognized.word_results and i < len(recognized.word_results):
                        for char, confidence, char_box in recognized.word_results[i]:
                            char_points = to_original(char_box, angle, scale, width, height)
                            characters.append({
                                "text": str(char),
                                "score": float(confidence),
                                "points": char_points,
                            })
                    if tight_boxes and angle == 0 and characters:
                        trim_character_heights(characters, edges)
                    findings.append({
                        "points": points,
                        "score": score,
                        "kind": "recognized" if is_confirmed else "candidate",
                        "text": str(texts[i]) if is_confirmed and texts is not None else None,
                        "characters": characters,
                        "rotation": angle,
                        "scale": scale,
                    })
    # Keep the highest scoring overlapping region. Partial overlaps remain
    # visible because they may be separate words or a missed piece of a line.
    findings.sort(key=lambda item: (item["kind"] == "recognized", item["score"] or 0), reverse=True)
    unique = []
    for item in findings:
        if all(overlap(item["points"], prior["points"]) < 0.55 for prior in unique):
            unique.append(item)
    return unique


def highlight(image: np.ndarray, findings: list[dict], tight_boxes: bool) -> np.ndarray:
    overlay = image.copy()
    for item in findings:
        color = (20, 35, 255) if item["kind"] == "recognized" else (0, 180, 255)
        polygons = [char["points"] for char in item["characters"]] if tight_boxes and item["characters"] else [item["points"]]
        for points in polygons:
            cv2.fillPoly(overlay, [np.rint(points).astype(np.int32)], color)
    result = cv2.addWeighted(overlay, 0.22, image, 0.78, 0)
    thickness = max(2, round(min(image.shape[:2]) / 450))
    for item in findings:
        color = (0, 0, 255) if item["kind"] == "recognized" else (0, 165, 255)
        polygons = [char["points"] for char in item["characters"]] if tight_boxes and item["characters"] else [item["points"]]
        for points in polygons:
            cv2.polylines(result, [np.rint(points).astype(np.int32)], True, color, thickness, cv2.LINE_AA)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Highlight automatically detected text in a card image.")
    parser.add_argument("image", type=Path, help="Input PNG, JPEG, or WebP")
    parser.add_argument("--output", type=Path, help="Output PNG; default is <image>_text_highlighted.png beside input")
    parser.add_argument("--threshold", type=float, default=0.4, help="Detector confidence cutoff (default: 0.4)")
    parser.add_argument("--horizontal-only", action="store_true", help="Run only the upright pass; skip 90/270 degree scans")
    parser.add_argument("--recognized-only", action="store_true", help="Hide orange detector candidates")
    parser.add_argument("--tight-boxes", action="store_true", help="Draw individual OCR character boxes for recognized text")
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    output = args.output or args.image.with_name(f"{args.image.stem}_text_highlighted.png")
    source = np.asarray(Image.open(args.image).convert("RGB"))[:, :, ::-1].copy()
    findings = detect(source, args.threshold, (0,) if args.horizontal_only else (0, 90, 270), args.tight_boxes)
    if args.recognized_only:
        findings = [item for item in findings if item["kind"] == "recognized"]
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(highlight(source, findings, args.tight_boxes), cv2.COLOR_BGR2RGB)).save(output)
    manifest = output.with_suffix(".json")
    manifest.write_text(json.dumps({
        "source": str(args.image.resolve()),
        "regions": [{
            "polygon": np.rint(item["points"]).astype(int).tolist(),
            "score": item["score"],
            "kind": item["kind"],
            "text": item["text"],
            "characters": [{
                "text": char["text"],
                "score": char["score"],
                "polygon": np.rint(char["points"]).astype(int).tolist(),
            } for char in item["characters"]],
            "rotation": item["rotation"],
            "scale": item["scale"],
        } for item in findings],
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(findings)} highlighted regions: {output}")
    print(f"Saved detection coordinates: {manifest}")


if __name__ == "__main__":
    main()
