#!/usr/bin/env python3
"""
create_dataset.py — Build a COCO Panoptic dataset from Label Studio exports.

Supports two split strategies:
  --split random    : 80/10/10 random split across all tiles
  --split location  : assign entire locations to train/val/test via split_locations.yaml

Output structure:
  Dataset_V{n}/
  ├── train/
  │   ├── images/          JPEG tiles
  │   ├── panoptic/        PNG — COCO panoptic encoding (R+G*256+B*256^2 = segment_id)
  │   └── panoptic_viz/    PNG — semantic colors per class (human-readable)
  ├── val/   (same structure)
  ├── test/  (same structure)
  ├── panoptic_train.json
  ├── panoptic_val.json
  ├── panoptic_test.json
  └── dataset_info.json

Panoptic mask encoding (COCO standard):
  segment_id = R + G*256 + B*256^2
  0 = void/ignore (unannotated pixels)

Usage examples:
  python create_dataset.py --version 1 --split random
  python create_dataset.py --version 2 --split location
  python create_dataset.py --version 3 --split location --split-config my_split.yaml
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from label_studio_sdk.converter.brush import decode_rle as _ls_decode_rle

# ── Class colors (RGB) — shared with gemini_segmentation_cloud.py ───────────
# Used only for the panoptic_viz/ visualization masks; NOT for the COCO masks.
CLASS_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "buildings":               (66,  47,  47),
    "fuel_infrastructure":     (241, 196,  15),
    "military_vehicles":       (231,  76,  60),
    "communication_and_radar": (233,  30,  99),
    "roads_and_tracks":        (127, 140, 141),
    "forest":                  ( 32, 151,  40),
    "perimeter_structures":    ( 52, 152, 219),
}
VOID_COLOR_RGB: tuple[int, int, int] = (0, 0, 0)  # unannotated pixels

# ── Default paths (relative to this script) ────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_LABELSTUDIO_JSON = (
    PROJECT_ROOT / "src" / "dataset" / "export_from_label-studio" / "dataset_v1.json"
)
DEFAULT_CLASSES_YAML = PROJECT_ROOT / "GRDINO" / "config" / "classes.yaml"
DEFAULT_IMAGES_ROOT = PROJECT_ROOT / "src" / "images" / "google_maps_web"
DEFAULT_SPLIT_CONFIG = SCRIPT_DIR / "split_locations.yaml"


# ── Class catalog ───────────────────────────────────────────────────────────

def load_categories(yaml_path: Path) -> list[dict]:
    """Load COCO-style categories from GRDINO classes.yaml."""
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    categories = []
    for idx, pass_cfg in enumerate(cfg["passes"], start=1):
        categories.append({
            "id": idx,
            "name": pass_cfg["name"],
            "supercategory": "military_facility",
            "isthing": 0 if pass_cfg.get("is_stuff", False) else 1,
        })
    return categories


# ── Label Studio path helpers ───────────────────────────────────────────────

def ls_url_to_path(ls_url: str, images_root: Path) -> Path:
    """Convert a Label Studio image URL to an absolute filesystem path.

    Label Studio URL format:
      /data/local-files/?d=images/google_maps_web/Location/tile.jpeg
    """
    if "?d=" in ls_url:
        rel = ls_url.split("?d=", 1)[1]
    else:
        rel = ls_url.lstrip("/")

    parts = Path(rel).parts
    try:
        idx = list(parts).index("google_maps_web")
        sub = Path(*parts[idx + 1:])
    except ValueError:
        sub = Path(rel)

    return images_root / sub


def get_location(ls_url: str) -> str:
    """Extract location name (folder) from a Label Studio image URL."""
    if "?d=" in ls_url:
        rel = ls_url.split("?d=", 1)[1]
    else:
        rel = ls_url
    parts = Path(rel).parts
    try:
        idx = list(parts).index("google_maps_web")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return "unknown"


# ── Annotation decoders ─────────────────────────────────────────────────────

def decode_brush_rle(rle: list[int], width: int, height: int) -> np.ndarray:
    """Decode a Label Studio brush RLE to a binary (H, W) uint8 mask.

    The RLE is a binary-packed format from @thi.ng/rle-pack (NOT alternating counts).
    The first 4 bytes encode the total element count (width * height * 4 RGBA channels).
    The SDK's decode_rle decompresses the flat RGBA array; we extract the alpha channel.
    """
    try:
        flat_rgba = _ls_decode_rle(rle)
        alpha = np.reshape(flat_rgba, [height, width, 4])[:, :, 3]
        return (alpha > 0).astype(np.uint8)
    except Exception:
        return np.zeros((height, width), dtype=np.uint8)


def polygon_pct_to_mask(points_pct: list, width: int, height: int) -> np.ndarray:
    """Convert Label Studio polygon (percentage coords) to binary (H, W) mask."""
    pts = np.array(
        [[p[0] / 100.0 * width, p[1] / 100.0 * height] for p in points_pct],
        dtype=np.float32,
    ).reshape((-1, 1, 2)).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


# ── Panoptic mask builder ───────────────────────────────────────────────────

def build_panoptic_mask(
    annotation_results: list,
    width: int,
    height: int,
    category_by_name: dict,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Convert Label Studio annotation results to COCO panoptic mask + viz mask + segments_info.

    Priority: stuff classes are drawn first, thing instances on top.
    Void pixels (no annotation) remain 0.

    Returns:
        panoptic_rgb : (H, W, 3) uint8 — segment ID encoded as R+G*256+B*256^2 (COCO standard)
        viz_rgb      : (H, W, 3) uint8 — semantic color visualization (CLASS_COLORS_RGB)
        segments_info: list of dicts ready for COCO panoptic JSON
    """
    # Accumulated stuff masks (merged per class) and individual thing masks
    stuff_masks: dict[int, np.ndarray] = defaultdict(
        lambda: np.zeros((height, width), dtype=np.uint8)
    )
    thing_instances: list[tuple[int, np.ndarray]] = []  # (category_id, mask)

    for ann in annotation_results:
        ann_type = ann.get("type", "")
        value = ann.get("value", {})

        if ann_type == "polygonlabels":
            label = value.get("polygonlabels", [None])[0]
            cat = category_by_name.get(label)
            if cat is None:
                continue
            if "points" in value:
                mask = polygon_pct_to_mask(value["points"], width, height)
            elif "rle" in value:
                # Label Studio occasionally stores a brush stroke with type=polygonlabels
                rle = value.get("rle", [])
                if not rle:
                    continue
                orig_w = ann.get("original_width", width)
                orig_h = ann.get("original_height", height)
                mask = decode_brush_rle(rle, orig_w, orig_h)
                if mask.shape != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            else:
                continue

        elif ann_type == "brushlabels":
            label = value.get("brushlabels", [None])[0]
            cat = category_by_name.get(label)
            if cat is None:
                continue
            rle = value.get("rle", [])
            if not rle:
                continue
            orig_w = ann.get("original_width", width)
            orig_h = ann.get("original_height", height)
            mask = decode_brush_rle(rle, orig_w, orig_h)
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        else:
            # rectanglelabels and other unsupported types are skipped
            continue

        if not np.any(mask):
            continue

        if cat["isthing"]:
            thing_instances.append((cat["id"], mask.astype(np.uint8)))
        else:
            stuff_masks[cat["id"]] = np.logical_or(
                stuff_masks[cat["id"]], mask
            ).astype(np.uint8)

    # ── Assign segment IDs ────────────────────────────────────────────────
    # ID 0 is reserved for void. Stuff first, things on top.
    panoptic = np.zeros((height, width), dtype=np.int32)
    next_id = 1
    seg_order: list[tuple[int, int, np.ndarray]] = []  # (seg_id, cat_id, mask)

    for cat_id, smask in stuff_masks.items():
        if np.any(smask):
            seg_order.append((next_id, cat_id, smask))
            next_id += 1

    for cat_id, tmask in thing_instances:
        seg_order.append((next_id, cat_id, tmask))
        next_id += 1

    for seg_id, _cat_id, mask in seg_order:
        panoptic[mask > 0] = seg_id

    # ── Build segments_info from final panoptic ───────────────────────────
    segments_info: list[dict] = []
    seen_cats: set[int] = set()

    for seg_id, cat_id, _ in seg_order:
        ys, xs = np.where(panoptic == seg_id)
        area = int(len(xs))
        if area == 0:
            continue  # fully overwritten by a higher-priority segment
        bbox = [
            int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1),
        ]
        segments_info.append({
            "id": seg_id,
            "category_id": cat_id,
            "iscrowd": 0,
            "area": area,
            "bbox": bbox,
        })
        seen_cats.add(cat_id)

    # ── Encode segment IDs as RGB (COCO panoptic standard) ───────────────
    panoptic_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    panoptic_rgb[:, :, 0] = panoptic & 0xFF
    panoptic_rgb[:, :, 1] = (panoptic >> 8) & 0xFF
    panoptic_rgb[:, :, 2] = (panoptic >> 16) & 0xFF

    # ── Build semantic-color visualization mask ───────────────────────────
    # Build seg_id → category_name lookup from segments_info
    id_to_catname: dict[int, str] = {}
    for seg in segments_info:
        cat_name = next(
            (k for k, v in category_by_name.items() if v["id"] == seg["category_id"]),
            None,
        )
        if cat_name:
            id_to_catname[seg["id"]] = cat_name

    viz_rgb = np.full((height, width, 3), VOID_COLOR_RGB, dtype=np.uint8)
    for seg_id, cat_name in id_to_catname.items():
        color = CLASS_COLORS_RGB.get(cat_name, VOID_COLOR_RGB)
        viz_rgb[panoptic == seg_id] = color

    return panoptic_rgb, viz_rgb, segments_info


# ── Split strategies ────────────────────────────────────────────────────────

def split_by_location(items: list, split_cfg: dict) -> dict[str, list]:
    """Assign items to train/val/test based on their location folder."""
    loc_to_split: dict[str, str] = {}
    for split_name, locations in split_cfg.items():
        for loc in (locations or []):
            loc_to_split[loc] = split_name

    result: dict[str, list] = {"train": [], "val": [], "test": []}
    unassigned: list[str] = []
    for item in items:
        loc = get_location(item["data"]["image"])
        split = loc_to_split.get(loc)
        if split is None:
            unassigned.append(loc)
            result["train"].append(item)
        else:
            result[split].append(item)

    if unassigned:
        unique = sorted(set(unassigned))
        print(
            f"  [WARN] {len(unassigned)} image(s) from unassigned locations "
            f"({', '.join(unique)}) → assigned to train",
            flush=True,
        )
    return result


def split_random(
    items: list, ratios: tuple = (0.8, 0.1, 0.1), seed: int = 42
) -> dict[str, list]:
    """Random 80/10/10 split (or any custom ratio) of individual tiles."""
    rng = random.Random(seed)
    shuffled = items.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train: n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


# ── Per-split COCO JSON builder ─────────────────────────────────────────────

def process_split(
    split_name: str,
    items: list,
    categories: list[dict],
    images_out: Path,
    panoptic_out: Path,
    panoptic_viz_out: Path,
    images_root: Path,
    version: int,
) -> dict:
    """Process all images in one split and return the COCO panoptic JSON dict."""
    category_by_name = {c["name"]: c for c in categories}

    coco: dict = {
        "info": {
            "description": f"Military Facility Panoptic Segmentation Dataset V{version}",
            "version": str(version),
            "year": datetime.now().year,
            "contributor": "TFM",
            "date_created": datetime.now().strftime("%Y/%m/%d"),
        },
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": [],
    }

    skipped = 0
    for image_id, item in enumerate(items, start=1):
        ls_url = item["data"]["image"]
        img_path = ls_url_to_path(ls_url, images_root)

        if not img_path.exists():
            print(f"    [SKIP] Not found: {img_path.name}", flush=True)
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"    [SKIP] Cannot read: {img_path.name}", flush=True)
            skipped += 1
            continue

        h, w = img.shape[:2]
        fname = img_path.name
        mask_fname = img_path.stem + "_panoptic.png"

        # Copy image to split/images/
        shutil.copy2(img_path, images_out / fname)

        # Build panoptic mask + visualization
        results = (item["annotations"][0]["result"]
                   if item.get("annotations") else [])
        panoptic_rgb, viz_rgb, segments_info = build_panoptic_mask(
            results, w, h, category_by_name
        )

        # panoptic/ — COCO encoding (PNG; JPEG would corrupt segment IDs)
        cv2.imwrite(str(panoptic_out / mask_fname), panoptic_rgb)

        # panoptic_viz/ — semantic colors (human-readable)
        viz_bgr = cv2.cvtColor(viz_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(panoptic_viz_out / mask_fname), viz_bgr)

        coco["images"].append({
            "id": image_id,
            "file_name": fname,
            "height": h,
            "width": w,
        })
        coco["annotations"].append({
            "image_id": image_id,
            "file_name": mask_fname,
            "segments_info": segments_info,
        })

        n_segs = len(segments_info)
        print(f"    {fname}  →  {n_segs} segment(s)", flush=True)

    if skipped:
        print(f"    [WARN] Skipped {skipped} image(s) in {split_name}", flush=True)

    return coco


# ── Entry point ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a COCO Panoptic dataset from a Label Studio export.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--version", type=int, default=1,
        help="Dataset version number. Output goes to Dataset_V{version}/ (default: 1)",
    )
    p.add_argument(
        "--split", choices=["random", "location"], default="random",
        help="Split strategy: 'random' (80/10/10) or 'location' (uses split_locations.yaml)",
    )
    p.add_argument(
        "--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG,
        metavar="YAML",
        help="YAML file mapping locations → train/val/test (only used with --split location)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for --split random (default: 42)",
    )
    p.add_argument(
        "--labelstudio-json", type=Path, default=DEFAULT_LABELSTUDIO_JSON,
        metavar="JSON",
        help="Path to the Label Studio export JSON",
    )
    p.add_argument(
        "--images-root", type=Path, default=DEFAULT_IMAGES_ROOT,
        metavar="DIR",
        help="Root directory containing location sub-folders with tile images",
    )
    p.add_argument(
        "--classes-yaml", type=Path, default=DEFAULT_CLASSES_YAML,
        metavar="YAML",
        help="Path to GRDINO classes.yaml for category definitions",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        metavar="DIR",
        help="Override output directory (default: ../Dataset_V{version}/ next to this script)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir or (PROJECT_ROOT / f"Dataset_V{args.version}")

    print(f"\n{'='*60}")
    print(f"  Dataset Creator — COCO Panoptic")
    print(f"  Version  : V{args.version}")
    print(f"  Split    : {args.split}")
    print(f"  Output   : {output_dir}")
    print(f"{'='*60}\n")

    # ── Validate inputs ───────────────────────────────────────────────────
    for path, label in [
        (args.labelstudio_json, "Label Studio JSON"),
        (args.classes_yaml, "classes.yaml"),
        (args.images_root, "images root"),
    ]:
        if not path.exists():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    if args.split == "location" and not args.split_config.exists():
        print(
            f"[ERROR] Split config not found: {args.split_config}\n"
            f"        Create it or use --split random",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Load categories ───────────────────────────────────────────────────
    categories = load_categories(args.classes_yaml)
    print(f"Categories ({len(categories)}):")
    for c in categories:
        kind = "thing" if c["isthing"] else "stuff"
        print(f"  [{c['id']:2d}] {c['name']:<28s} ({kind})")

    # ── Load Label Studio export ──────────────────────────────────────────
    print(f"\nLoading annotations from {args.labelstudio_json.name} …")
    with open(args.labelstudio_json, encoding="utf-8") as f:
        ls_data: list = json.load(f)
    print(f"  {len(ls_data)} annotated image(s) loaded")

    # ── Split ─────────────────────────────────────────────────────────────
    print(f"\nSplitting ({args.split}) …")
    if args.split == "location":
        with open(args.split_config, encoding="utf-8") as f:
            split_cfg = yaml.safe_load(f)
        splits = split_by_location(ls_data, split_cfg)
    else:
        splits = split_random(ls_data, seed=args.seed)

    for s, items in splits.items():
        print(f"  {s:5s}: {len(items):3d} image(s)")

    # ── Create output directories ─────────────────────────────────────────
    for split_name in ("train", "val", "test"):
        (output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "panoptic").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "panoptic_viz").mkdir(parents=True, exist_ok=True)

    # ── Process each split ────────────────────────────────────────────────
    coco_splits: dict[str, dict] = {}
    for split_name, items in splits.items():
        print(f"\n[{split_name.upper()}] Processing {len(items)} image(s) …")
        coco = process_split(
            split_name=split_name,
            items=items,
            categories=categories,
            images_out=output_dir / split_name / "images",
            panoptic_out=output_dir / split_name / "panoptic",
            panoptic_viz_out=output_dir / split_name / "panoptic_viz",
            images_root=args.images_root,
            version=args.version,
        )
        coco_splits[split_name] = coco

        json_path = output_dir / f"panoptic_{split_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)
        print(f"  Saved → {json_path.name}")

    # ── dataset_info.json — summary metadata ──────────────────────────────
    stats = {}
    for s, coco in coco_splits.items():
        n_imgs = len(coco["images"])
        n_segs = sum(len(a["segments_info"]) for a in coco["annotations"])
        cat_counts: dict[str, int] = defaultdict(int)
        for ann in coco["annotations"]:
            for seg in ann["segments_info"]:
                cat_name = next(
                    c["name"] for c in categories if c["id"] == seg["category_id"]
                )
                cat_counts[cat_name] += 1
        stats[s] = {
            "images": n_imgs,
            "segments": n_segs,
            "segments_per_category": dict(sorted(cat_counts.items())),
        }

    dataset_info = {
        "version": args.version,
        "split_mode": args.split,
        "split_config": str(args.split_config) if args.split == "location" else None,
        "random_seed": args.seed if args.split == "random" else None,
        "categories": categories,
        "stats": stats,
        "created": datetime.now().isoformat(),
        "source": {
            "labelstudio_json": str(args.labelstudio_json),
            "images_root": str(args.images_root),
        },
    }
    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Done! Dataset_V{args.version} ready at:")
    print(f"  {output_dir}")
    print(f"{'='*60}")
    total_imgs = sum(s["images"] for s in stats.values())
    total_segs = sum(s["segments"] for s in stats.values())
    print(f"\n  Split    Images  Segments")
    print(f"  ─────────────────────────")
    for s, st in stats.items():
        print(f"  {s:5s}    {st['images']:6d}  {st['segments']:8d}")
    print(f"  ─────────────────────────")
    print(f"  total    {total_imgs:6d}  {total_segs:8d}\n")


if __name__ == "__main__":
    main()
