#!/usr/bin/env python3
"""Run a pretrained fashion human parser locally over the image catalog.

This is inference, not fine-tuning. It produces reviewable masks/crops and
measurements so a later human-labeled training set can be built responsibly.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path.home() / "Desktop" / "fashionArchive"
DEFAULT_CATALOG = ROOT / "output" / "image_catalog.csv"
DEFAULT_OUTPUT = ROOT / "output" / "segmentation"
MODEL = "fashn-ai/fashn-human-parser"

# Category IDs from the model card. Categories are merged only for convenient review.
CLASS_LABELS = {
    0: "background", 1: "face", 2: "hair", 3: "top", 4: "dress", 5: "skirt",
    6: "pants", 7: "belt", 8: "bag", 9: "hat", 10: "scarf", 11: "glasses",
    12: "arms", 13: "hands", 14: "legs", 15: "feet", 16: "torso", 17: "jewelry",
}
GROUPS = {
    "top": [3], "dress": [4], "bottom": [5, 6], "bag": [8], "hat": [9],
    "accessory": [7, 10, 11, 17],
}
COLORS = {
    "top": (237, 22, 137), "dress": (224, 65, 108), "bottom": (65, 108, 209),
    "bag": (178, 194, 67), "hat": (245, 151, 48), "accessory": (125, 81, 181),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="Optional small-run cap; 0 means all images")
    parser.add_argument("--incremental", action="store_true", help="Skip images whose source size and modification time have not changed")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    crops_dir, overlays_dir = output / "crops", output / "overlays"
    crops_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    with args.catalog.open(encoding="utf-8-sig", newline="") as f:
        catalog = list(csv.DictReader(f))
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL).eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    result_path = output / "segmentation_results.json"
    existing = {}
    if args.incremental and result_path.is_file():
        try:
            existing = {r["image_id"]: r for r in json.loads(result_path.read_text(encoding="utf-8")).get("results", [])}
        except (OSError, ValueError, KeyError):
            existing = {}
    pending = []
    for row in catalog:
        path = (source / row["relative_path"]).resolve()
        prior = existing.get(row["image_id"])
        if source not in path.parents or not path.is_file():
            print(f"Skipping unavailable image: {row['relative_path']}")
            continue
        signature = f"{path.stat().st_size}:{path.stat().st_mtime_ns}"
        # Older result files predate signatures. In incremental mode, keep those
        # existing image results instead of unexpectedly re-running the archive.
        if prior and (not prior.get("source_signature") or prior.get("source_signature") == signature):
            continue
        row["source_signature"] = signature
        pending.append(row)
    if args.limit:
        pending = pending[:args.limit]

    results, flat = [], []

    for ix, row in enumerate(pending, 1):
        image_id = row["image_id"]
        path = (source / row["relative_path"]).resolve()
        if source not in path.parents:
            continue
        with Image.open(path) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        # Bound inference memory for large phone photos on small hosted instances.
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        w, h = image.size
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False).softmax(dim=1)[0]
        pred = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)
        prob_np = probs.cpu().numpy()

        overlay = np.asarray(image, dtype=np.float32).copy()
        detections = []
        for group, class_ids in GROUPS.items():
            mask = np.isin(pred, class_ids)
            area = int(mask.sum())
            threshold = max(80, int(w * h * 0.00025))
            if area < threshold:
                continue
            ys, xs = np.where(mask)
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            conf = float(np.mean([prob_np[c][mask & (pred == c)].mean() for c in class_ids if np.any(mask & (pred == c))]))
            # Transparent crop retains only the predicted region for visual review.
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., :3] = np.asarray(image)
            rgba[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
            crop_file = f"{image_id}_{group}.png"
            Image.fromarray(rgba[y0:y1, x0:x1]).save(crops_dir / crop_file, optimize=True)
            color = np.asarray(COLORS[group], dtype=np.float32)
            overlay[mask] = overlay[mask] * 0.48 + color * 0.52
            detection = {
                "category": group,
                "classes": [CLASS_LABELS[c] for c in class_ids if np.any(pred == c)],
                "pixel_area": area,
                "area_ratio": round(area / (w * h), 5),
                "mean_pixel_confidence": round(conf, 4),
                "mean_rgb": [round(float(v), 1) for v in np.asarray(image, dtype=np.float32)[mask].mean(axis=0)],
                "bbox_xyxy": [x0, y0, x1, y1],
                "crop_file": f"crops/{crop_file}",
            }
            detections.append(detection)
            flat.append({"image_id": image_id, "source_title": row.get("source_title", ""),
                "record_type": row.get("record_type", ""), **detection})

        overlay_file = f"{image_id}_overlay.jpg"
        Image.fromarray(np.uint8(np.clip(overlay, 0, 255))).save(overlays_dir / overlay_file, quality=88, optimize=True)
        results.append({"image_id": image_id, "source_signature": row.get("source_signature", ""), "relative_path": row["relative_path"],
            "image_mean_rgb": [round(float(v)) for v in np.asarray(image, dtype=np.float32).mean(axis=(0, 1))],
            "source_title": row.get("source_title", ""), "record_type": row.get("record_type", ""),
            "overlay_file": f"overlays/{overlay_file}", "detections": detections})
        print(f"[{ix}/{len(catalog)}] {row['relative_path']} — {len(detections)} garment groups")

    if args.incremental:
        by_id = dict(existing)
        by_id.update({r["image_id"]: r for r in results})
        results = list(by_id.values())
        flat = [{"image_id": r["image_id"], "source_title": r.get("source_title", ""),
            "record_type": r.get("record_type", ""), **d} for r in results for d in r["detections"]]
    temporary_result = result_path.with_suffix('.tmp')
    temporary_result.write_text(json.dumps({
        "model": MODEL, "model_is_pretrained": True, "image_count": len(results),
        "category_counts": {g: sum(any(d["category"] == g for d in r["detections"]) for r in results) for g in GROUPS},
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_result.replace(result_path)
    with (output / "segmentations.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["image_id", "source_title", "record_type", "category", "classes", "pixel_area", "area_ratio", "mean_pixel_confidence", "mean_rgb", "bbox_xyxy", "crop_file"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in flat:
            r["classes"] = ",".join(r["classes"])
            r["mean_rgb"] = json.dumps(r.get("mean_rgb", []))
            r["bbox_xyxy"] = json.dumps(r["bbox_xyxy"])
            writer.writerow(r)
    print(f"Processed {len(pending)} images; saved {len(results)} total image results and {len(flat)} detections to {output}")


if __name__ == "__main__":
    main()
