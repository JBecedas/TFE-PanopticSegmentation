"""
pipeline_gdino_sam.py
=====================
Multi-pass Pipeline: GroundingDINO → Outlier Rejection → SAM → Label Studio JSON

Architecture:
  1. Load passes from classes.yaml (same dir as this script)
  2. Read tiles from each ROI in TILES_ROOT
  3. For each tile, run N passes of GroundingDINO (one per class group):
     - Each pass runs ONE GroundingDINO call PER PROMPT (not all joined together)
     - In-pass IoU-NMS removes cross-prompt duplicates  (uses nms_threshold)
  4. Outlier Rejection per pass (independent params per class)
  5. Merge all pass detections
  6. SAM inference (box prompt) on merged boxes
  7. Output:
     - Visualization image with all boxes + masks   (outputs/visuals/)
     - Unified Label Studio JSON                    (outputs/import_to_labelstudio.json)
     - log.md with detailed statistics

Calibration mode (threshold tuning, no SAM needed):
  python pipeline_gdino_sam.py --calibrate
  python pipeline_gdino_sam.py --calibrate --class military_vehicles
  python pipeline_gdino_sam.py --calibrate --images /path/to/test/images --n-samples 10
"""

import os, sys, json, uuid, yaml, cv2, argparse, re
import numpy as np
from pathlib import Path
from datetime import datetime

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
CONFIG_PATH = Path("c:/TFM/GRDINO/config/classes.yaml")  # can be overridden with --config  
TILES_ROOT  = Path("c:/TFM/src/images/google_maps_web")
OUTPUT_VIS  = Path("c:/TFM/GRDINO/outputs/visuals")
OUTPUT_JSON = Path("c:/TFM/GRDINO/outputs")

SAM_CHECKPOINT = Path("c:/TFM/Labeling/scripts/sam_vit_h_4b8939.pth")
SAM_MODEL_TYPE = "vit_h"
GDINO_WEIGHTS  = Path("c:/TFM/GRDINO/weights/groundingdino_swint_ogc.pth")
GDINO_CONFIG   = None  # auto-detected from package if None

# ─── Conditional imports ──────────────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from groundingdino.util.inference import load_model, load_image, predict
    HAS_GDINO = True
except ImportError:
    HAS_GDINO = False

try:
    from segment_anything import sam_model_registry, SamPredictor
    HAS_SAM = True
except ImportError:
    HAS_SAM = False


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        print(f"[ERROR] Config not found: {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    passes = cfg.get("passes", [])
    print(f"[CONFIG] {len(passes)} passes loaded from {CONFIG_PATH.name}:")
    for p in passes:
        n = len(p.get("text_prompts", []))
        oc = "ON" if p.get("outlier_rejection", {}).get("enabled", True) else "OFF"
        print(f"  • {p['name']:<30} {n:>2} prompts  "
              f"box={p.get('box_threshold', 0.35)}  "
              f"text={p.get('text_threshold', 0.25)}  "
              f"nms={p.get('nms_threshold', 0.5)}  outlier={oc}")
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
#  MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def _find_gdino_config() -> str:
    import groundingdino
    pkg_dir = Path(groundingdino.__file__).parent
    candidates = list(pkg_dir.rglob("GroundingDINO_SwinT_OGC.py"))
    if not candidates:
        candidates = [c for c in pkg_dir.rglob("*.py")
                      if "SwinT" in c.name or "config" in c.name.lower()]
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError("GroundingDINO config file not found in package.")


def load_models(device: str):
    if not HAS_GDINO:
        raise ImportError("groundingdino not installed.")
    cfg_path = GDINO_CONFIG or _find_gdino_config()
    print(f"[GDINO] Config : {cfg_path}")
    print(f"[GDINO] Weights: {GDINO_WEIGHTS}")
    gdino_model = load_model(cfg_path, str(GDINO_WEIGHTS), device=device)

    if not HAS_SAM:
        raise ImportError("segment_anything not installed.")
    print(f"[SAM]  Checkpoint: {SAM_CHECKPOINT}")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device)
    return gdino_model, SamPredictor(sam)


def load_gdino_only(device: str):
    """Load only GroundingDINO (used in calibration mode — SAM not needed)."""
    if not HAS_GDINO:
        raise ImportError("groundingdino not installed.")
    cfg_path = GDINO_CONFIG or _find_gdino_config()
    print(f"[GDINO] Config : {cfg_path}")
    print(f"[GDINO] Weights: {GDINO_WEIGHTS}")
    return load_model(cfg_path, str(GDINO_WEIGHTS), device=device)


def load_sam_only(device: str):
    """Load only SAM (used when GroundingDINO is not needed, e.g. --from-annotations)."""
    if not HAS_SAM:
        raise ImportError("segment_anything not installed.")
    print(f"[SAM]  Checkpoint: {SAM_CHECKPOINT}")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device)
    return SamPredictor(sam)


# ═══════════════════════════════════════════════════════════════════════════════
#  IoU-NMS
# ═══════════════════════════════════════════════════════════════════════════════

def iou_nms(
    boxes: "torch.Tensor",
    scores: "torch.Tensor",
    iou_threshold: float = 0.5,
) -> "torch.Tensor":
    """
    IoU-based Non-Maximum Suppression.

    Args:
        boxes         : [N, 4] tensor in xyxy pixel coords
        scores        : [N] confidence tensor
        iou_threshold : boxes with IoU > this with a higher-scored box are suppressed

    Returns:
        1D LongTensor of kept indices (sorted by descending score).
    """
    if len(boxes) == 0:
        return torch.zeros(0, dtype=torch.long)

    order = scores.argsort(descending=True)
    keep  = []

    while len(order) > 0:
        idx = order[0].item()
        keep.append(idx)
        if len(order) == 1:
            break

        rest   = order[1:]
        b      = boxes[idx]
        others = boxes[rest]

        ix1 = torch.maximum(b[0], others[:, 0])
        iy1 = torch.maximum(b[1], others[:, 1])
        ix2 = torch.minimum(b[2], others[:, 2])
        iy2 = torch.minimum(b[3], others[:, 3])

        inter    = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        area_b   = (b[2] - b[0]) * (b[3] - b[1])
        area_o   = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
        iou      = inter / (area_b + area_o - inter).clamp(min=1e-6)

        order = rest[iou < iou_threshold]

    return torch.tensor(keep, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-PROMPT GDINO INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def run_gdino_per_prompt(
    gdino_model,
    image_transformed: "torch.Tensor",
    prompts: list,
    box_th: float,
    text_th: float,
    nms_th: float,
    img_w: int,
    img_h: int,
    device: str,
):
    """
    Runs GroundingDINO ONCE PER PROMPT (rather than all prompts joined in one call).
    After collecting all raw detections, applies in-pass IoU-NMS using nms_th
    to remove cross-prompt duplicates before returning.

    Why per-prompt?
      - Each GDINO call is focused on a single concept → less cross-concept noise.
      - nms_threshold (previously unused) now correctly de-duplicates overlapping
        boxes that were detected by different prompts of the same class.

    Args:
        gdino_model        : loaded GroundingDINO model
        image_transformed  : preprocessed image tensor (from load_image)
        prompts            : list of text prompts for this pass
        box_th / text_th   : GroundingDINO detection thresholds
        nms_th             : IoU threshold for in-pass NMS
        img_w / img_h      : image dimensions in pixels
        device             : "cuda" or "cpu"

    Returns:
        boxes_xyxy : torch.Tensor [N, 4]  absolute pixel coords (xyxy)
        logits     : torch.Tensor [N]
        phrases    : list[str]            matched text from caption
        n_raw      : int                  total detections before in-pass NMS
    """
    all_boxes, all_logits, all_phrases = [], [], []

    for prompt in prompts:
        caption = prompt.strip()
        if not caption.endswith("."):
            caption += " ."

        boxes, logits, phrases = predict(
            model=gdino_model,
            image=image_transformed,
            caption=caption,
            box_threshold=box_th,
            text_threshold=text_th,
            device=device,
        )
        if len(boxes) == 0:
            continue

        # Convert normalized cxcywh → absolute pixel xyxy
        xyxy = torch.zeros_like(boxes)
        xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * img_w
        xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * img_h
        xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * img_w
        xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * img_h

        all_boxes.append(xyxy)
        all_logits.append(logits)
        all_phrases.extend(phrases)

    if not all_boxes:
        empty = torch.zeros((0, 4))
        return empty, torch.zeros(0), [], 0

    merged_boxes  = torch.cat(all_boxes,  dim=0)
    merged_logits = torch.cat(all_logits, dim=0)
    n_raw = len(merged_boxes)

    # In-pass NMS: remove boxes that overlap across different prompts
    if n_raw > 1:
        keep          = iou_nms(merged_boxes, merged_logits, iou_threshold=nms_th)
        merged_boxes  = merged_boxes[keep]
        merged_logits = merged_logits[keep]
        phrases_out   = [all_phrases[i] for i in keep.tolist()]
    else:
        phrases_out = all_phrases

    return merged_boxes, merged_logits, phrases_out, n_raw


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTLIER REJECTION
# ═══════════════════════════════════════════════════════════════════════════════

def reject_outliers(
    boxes, logits, phrases,
    img_w, img_h,
    min_area_ratio=0.0005, max_area_ratio=0.5,
    min_aspect=0.1,        max_aspect=10.0,
    min_confidence=0.30,
):
    """Filters anomalous bounding boxes by area ratio, aspect ratio and confidence."""
    keep      = []
    img_area  = img_w * img_h

    for i, (box, logit) in enumerate(zip(boxes, logits)):
        x1, y1, x2, y2 = box
        bw, bh  = x2 - x1, y2 - y1
        ratio   = (bw * bh) / img_area
        aspect  = bw / max(bh, 1e-6)

        if ratio   < min_area_ratio or ratio   > max_area_ratio:
            continue
        if aspect  < min_aspect     or aspect  > max_aspect:
            continue
        if float(logit) < min_confidence:
            continue
        keep.append(i)

    if not keep:
        return boxes[:0], logits[:0], []

    n_rej = len(boxes) - len(keep)
    if n_rej > 0:
        print(f"      [OUTLIER] Rejected {n_rej}/{len(boxes)} boxes")
    return boxes[keep], logits[keep], [phrases[i] for i in keep]


# ═══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate_classes(
    passes_cfg:      list,
    gdino_model,
    device:          str,
    images_dir:      Path = None,
    target_class:    str  = None,
    n_sample_images: int  = 5,
    box_th_values:   list = None,
    text_th_values:  list = None,
    output_dir:      Path = None,
) -> None:
    """
    Statistical calibration of GroundingDINO thresholds per class pass.

    Runs two analyses per class:
      1. Per-prompt statistics — which prompts fire and how often (current config).
      2. Threshold grid (box_th × text_th) — total detections per combination
         using all prompts joined (fast approximation).

    Outputs a Markdown report in output_dir that you can open in any Markdown
    viewer to compare combinations and decide on final threshold values.

    Args:
        passes_cfg      : list of pass configs from classes.yaml
        gdino_model     : loaded GroundingDINO model
        device          : "cuda" or "cpu"
        images_dir      : directory with sample images; defaults to TILES_ROOT
        target_class    : if set, calibrate only that pass (by name)
        n_sample_images : max number of images to sample
        box_th_values   : box_threshold values to grid-search
        text_th_values  : text_threshold values to grid-search
        output_dir      : where to write the report; defaults to outputs/calibration/

    Usage examples:
        # All classes, 5 random tiles from production:
        python pipeline_gdino_sam.py --calibrate

        # One class, dedicated test images:
        python pipeline_gdino_sam.py --calibrate \\
            --class military_vehicles \\
            --images /path/to/test_images \\
            --n-samples 10
    """
    import random

    if box_th_values is None:
        box_th_values = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25]
    if text_th_values is None:
        text_th_values = [0.10, 0.15, 0.20, 0.25, 0.30]
    if output_dir is None:
        output_dir = OUTPUT_JSON / "calibration"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect sample images ─────────────────────────────────────────────────
    root = Path(images_dir) if images_dir else TILES_ROOT
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    all_imgs = [p for p in root.rglob("*") if p.suffix.lower() in exts]

    if not all_imgs:
        print(f"[CALIB] No images found under {root}")
        print(f"[CALIB] Use --images <dir> to point to a directory with test images.")
        return

    random.shuffle(all_imgs)
    samples = all_imgs[:n_sample_images]
    print(f"\n[CALIB] {len(samples)} sample images (from {len(all_imgs)} found in {root})")
    print(f"[CALIB] Report will be saved to: {output_dir}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report    = []
    r         = report.append

    r(f"# GroundingDINO Calibration Report — {timestamp}\n")
    r(f"- **Sample images:** {len(samples)} of {len(all_imgs)} available")
    r(f"- **Source directory:** `{root}`")
    r(f"- **box_threshold values:** {box_th_values}")
    r(f"- **text_threshold values:** {text_th_values}\n")
    r("---\n")
    r("> **How to read this report**")
    r("> - Section 1 shows which individual prompts generate detections with your current config.")
    r(">   Prompts with 0 detections can be removed or their thresholds lowered.")
    r("> - Section 2 is a grid search: higher numbers = more detections.")
    r(">   Too high → likely false positives. Too low → misses. Target a stable plateau.")
    r("> - The `◀ current` marker shows your current config in the grid.\n")
    r("---\n")

    for pass_cfg in passes_cfg:
        pass_name = pass_cfg["name"]
        if target_class and pass_name != target_class:
            continue

        prompts     = pass_cfg.get("text_prompts", [])
        cur_bth     = pass_cfg.get("box_threshold",  0.35)
        cur_tth     = pass_cfg.get("text_threshold",  0.25)
        nms_th      = pass_cfg.get("nms_threshold",   0.50)
        oc          = pass_cfg.get("outlier_rejection", {})

        print(f"  Calibrating: {pass_name}  ({len(prompts)} prompts) …")
        r(f"## Class: `{pass_name}`")
        r(f"*Current config → box_th=`{cur_bth}`, text_th=`{cur_tth}`, nms_th=`{nms_th}`*\n")

        # ── 1. Per-prompt statistics ──────────────────────────────────────────
        r("### 1. Per-prompt statistics (current thresholds)\n")
        r("| Prompt | Detections | Mean score | Std score | Mean bbox area (%) |")
        r("|:-------|:----------:|:----------:|:---------:|:-----------------:|")

        for prompt in prompts:
            caption = prompt.strip()
            if not caption.endswith("."):
                caption += " ."

            p_dets, p_scores, p_areas = [], [], []
            for img_path in samples:
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue
                h, w   = img_bgr.shape[:2]
                try:
                    _, img_t = load_image(str(img_path))
                    boxes, logits, _ = predict(
                        model=gdino_model, image=img_t,
                        caption=caption,
                        box_threshold=cur_bth,
                        text_threshold=cur_tth,
                        device=device,
                    )
                    for box, logit in zip(boxes, logits):
                        bw = box[2] * w
                        bh = box[3] * h
                        p_scores.append(float(logit))
                        p_areas.append((bw * bh) / (w * h) * 100)
                    p_dets.append(len(boxes))
                except Exception as exc:
                    print(f"    [WARN] {img_path.name}: {exc}")

            n_det  = sum(p_dets)
            s_mean = np.mean(p_scores) if p_scores else 0.0
            s_std  = np.std(p_scores)  if p_scores else 0.0
            a_mean = np.mean(p_areas)  if p_areas  else 0.0
            label  = (prompt[:48] + "…") if len(prompt) > 48 else prompt
            r(f"| `{label}` | {n_det} | {s_mean:.3f} | {s_std:.3f} | {a_mean:.3f}% |")

        r("")

        # ── 2. Threshold grid (all prompts joined for speed) ──────────────────
        joined_caption = (" . ".join(p.strip().rstrip(".")
                                     for p in prompts) + " .")

        r("### 2. Threshold grid — total detections over all sample images\n")
        r("*All prompts joined in one call (fast mode). Use as a directional guide.*\n")

        header = "| box_th \\ text_th |" + "".join(f" **{t}** |" for t in text_th_values)
        sep    = "|:----------------:|" + "".join(":---:|" for _ in text_th_values)
        r(header)
        r(sep)

        for bth in box_th_values:
            cells = [f"| **{bth}** |"]
            for tth in text_th_values:
                total = 0
                for img_path in samples:
                    if cv2.imread(str(img_path)) is None:
                        continue
                    try:
                        _, img_t = load_image(str(img_path))
                        boxes, _, _ = predict(
                            model=gdino_model, image=img_t,
                            caption=joined_caption,
                            box_threshold=bth,
                            text_threshold=tth,
                            device=device,
                        )
                        total += len(boxes)
                    except Exception:
                        pass
                is_current = (abs(bth - cur_bth) < 1e-9 and
                              abs(tth - cur_tth) < 1e-9)
                marker = " ◀ current" if is_current else ""
                cells.append(f" {total}{marker} |")
            r("".join(cells))

        r("")

        # ── 3. Outlier rejection params ───────────────────────────────────────
        if oc.get("enabled", True):
            r("### 3. Outlier rejection (current config)\n")
            r("| Parameter | Value |")
            r("|:----------|:-----:|")
            for k in ["min_area_ratio", "max_area_ratio", "min_aspect",
                      "max_aspect", "min_confidence"]:
                r(f"| `{k}` | {oc.get(k, '—')} |")
            r("")

        r("---\n")

    # ── Write report ──────────────────────────────────────────────────────────
    report_path = output_dir / f"calibration_{timestamp}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"\n[CALIB] Report saved: {report_path}")
    print(f"[CALIB] Open with a Markdown viewer (VS Code, Typora, etc.)")


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTO-CALIBRATION (automatic parameter optimisation)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_sample_images(root: Path, n_sample: int) -> list:
    """Returns list of (img_tensor, width, height, path) for up to n_sample images."""
    import random
    exts    = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    all_imgs = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    if not all_imgs:
        return []
    random.shuffle(all_imgs)
    loaded = []
    for p in all_imgs[:n_sample]:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        try:
            _, img_t = load_image(str(p))
            loaded.append((img_t, w, h, p))
        except Exception as exc:
            print(f"  [WARN] {p.name}: {exc}")
    return loaded


def _eval_config(
    gdino_model, samples: list,
    caption: str, box_th: float, text_th: float, nms_th: float,
    device: str,
) -> tuple:
    """
    Evaluates one (box_th, text_th, nms_th) combination across sample images.
    Returns (mean_confidence, n_detections, stats_list).
    Each stats_list entry: {"area_ratio", "aspect", "confidence"}.
    """
    all_scores, all_stats = [], []
    for img_t, w, h, _ in samples:
        try:
            boxes, logits, _ = predict(
                model=gdino_model, image=img_t,
                caption=caption, box_threshold=box_th,
                text_threshold=text_th, device=device,
            )
            if len(boxes) == 0:
                continue
            # normalised cxcywh → absolute pixel xyxy
            xyxy = torch.zeros_like(boxes)
            xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
            xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
            xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
            xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h
            if len(xyxy) > 1:
                keep   = iou_nms(xyxy, logits, iou_threshold=nms_th)
                xyxy   = xyxy[keep]
                logits = logits[keep]
            for box, logit in zip(xyxy, logits):
                x1, y1, x2, y2 = box.tolist()
                bw, bh = x2 - x1, y2 - y1
                all_scores.append(float(logit))
                all_stats.append({
                    "area_ratio": (bw * bh) / (w * h),
                    "aspect":     bw / max(bh, 1e-6),
                    "confidence": float(logit),
                })
        except Exception:
            pass
    n = len(all_scores)
    return (float(np.mean(all_scores)) if all_scores else 0.0), n, all_stats


def _grid_search(
    gdino_model, samples: list, caption: str,
    device: str, min_det: int,
    box_values: list, text_values: list, nms_th: float,
    label: str = "grid",
) -> dict:
    """
    Grid search over box_th × text_th maximising mean confidence.
    Returns {"box_th", "text_th", "mean_conf", "n_det"}.
    """
    best  = {"box_th": box_values[0], "text_th": text_values[0],
             "mean_conf": 0.0, "n_det": 0}
    total = len(box_values) * len(text_values)
    done  = 0
    for box_th in box_values:
        for text_th in text_values:
            done += 1
            mc, nd, _ = _eval_config(
                gdino_model, samples, caption, box_th, text_th, nms_th, device,
            )
            print(f"    [{label} {done:>3}/{total}] box={box_th:.3f} text={text_th:.3f}"
                  f" → conf={mc:.4f} n={nd:>4}", end="\r")
            if nd >= min_det and mc > best["mean_conf"]:
                best = {"box_th": box_th, "text_th": text_th,
                        "mean_conf": mc, "n_det": nd}
    print()   # newline after \r progress
    return best


def _search_nms_threshold(
    gdino_model, samples: list, caption: str,
    box_th: float, text_th: float,
    device: str, min_det: int, nms_values: list,
) -> float:
    """Sweep nms_threshold and return the value that maximises mean confidence."""
    best_nms, best_mc = nms_values[len(nms_values) // 2], 0.0
    for nms_th in nms_values:
        mc, nd, _ = _eval_config(
            gdino_model, samples, caption, box_th, text_th, nms_th, device,
        )
        print(f"    nms={nms_th:.2f} → conf={mc:.4f}  n={nd}")
        if nd >= min_det and mc > best_mc:
            best_mc, best_nms = mc, nms_th
    return best_nms


def _fit_outlier_params(stats: list, pct_low: float = 2.0, pct_high: float = 98.0) -> dict:
    """
    Fit outlier-rejection bounds from the detection distribution.
    Clips the tails at pct_low / pct_high percentiles so that extreme
    detections (noise) are removed while the core distribution is kept.
    """
    if not stats:
        return {}
    areas   = [s["area_ratio"]  for s in stats]
    aspects = [s["aspect"]      for s in stats]
    confs   = [s["confidence"]  for s in stats]
    p = np.percentile
    return {
        "min_area_ratio": round(float(max(1e-7, p(areas,   pct_low))),  7),
        "max_area_ratio": round(float(min(0.99,  p(areas,   pct_high))), 5),
        "min_aspect":     round(float(max(0.05,  p(aspects, pct_low))),  3),
        "max_aspect":     round(float(min(30.0,  p(aspects, pct_high))), 3),
        "min_confidence": round(float(max(0.01,  p(confs,   pct_low))),  4),
    }


def _patch_yaml_value(yaml_text: str, field: str, new_val) -> str:
    """Replace a YAML numeric field value in-place, preserving inline comments."""
    new_str = f"{new_val:.7g}" if isinstance(new_val, float) else str(new_val)
    pat = rf"^(\s*{re.escape(field)}\s*:\s*)[\d.eE+\-]+(\s*(?:#.*)?)$"
    return re.sub(pat, rf"\g<1>{new_str}\g<2>", yaml_text, flags=re.MULTILINE)


def _update_pass_in_yaml(yaml_text: str, pass_name: str, params: dict) -> str:
    """
    Apply params dict to the named pass block in the YAML text.
    Uses regex so inline comments are preserved; does not require ruamel.yaml.
    """
    m = re.search(rf"(?m)(- name:\s*{re.escape(pass_name)})", yaml_text)
    if not m:
        print(f"  [WARN] Pass '{pass_name}' not found in YAML — skipping.")
        return yaml_text
    after = yaml_text[m.end():]
    nxt   = re.search(r"\n  - name:", after)
    end   = (m.end() + nxt.start()) if nxt else len(yaml_text)
    block = yaml_text[m.start():end]
    for field, val in params.items():
        block = _patch_yaml_value(block, field, val)
    return yaml_text[:m.start()] + block + yaml_text[end:]


def auto_calibrate_classes(
    passes_cfg:      list,
    gdino_model,
    device:          str,
    images_dir:      Path = None,
    target_class:    str  = None,
    n_sample_images: int  = 5,
    output_dir:      Path = None,
    config_path:     Path = None,
    min_det_total:   int  = None,
) -> None:
    """
    Automatic calibration that maximises mean detection confidence per pass.

    Algorithm (4 stages per pass):
      1. Coarse grid  — box_threshold × text_threshold sweep (joined prompts)
      2. Fine grid    — ±0.04 around the coarse optimum (step 0.02)
      3. NMS sweep    — fix box/text, search nms_threshold
      4. Outlier fit  — collect detections at optimal thresholds,
                        derive min/max area_ratio, aspect, confidence
                        from the empirical distribution (percentile-based)

    Outputs:
      - Markdown report in output_dir
      - Backup of classes.yaml (classes.yaml.bak)
      - Updated classes.yaml with optimised parameters (comments preserved)

    Usage:
        python pipeline_gdino_sam.py --auto-calibrate
        python pipeline_gdino_sam.py --auto-calibrate --class military_vehicles \\
            --images /path/to/test_images --n-samples 15
    """
    COARSE_BOX  = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35]
    COARSE_TEXT = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    NMS_RANGE   = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    if output_dir is None:
        output_dir = OUTPUT_JSON / "calibration"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root    = Path(images_dir) if images_dir else TILES_ROOT
    samples = _load_sample_images(root, n_sample_images)

    if not samples:
        print(f"[AUTO-CALIB] No images found under {root}")
        print(f"[AUTO-CALIB] Use --images <dir> to specify a directory with test images.")
        return

    if min_det_total is None:
        min_det_total = max(1, len(samples) // 2)

    print(f"\n[AUTO-CALIB] {len(samples)} sample images | "
          f"min_det={min_det_total} | device={device.upper()}")
    if device == "cpu":
        print(f"[AUTO-CALIB] WARNING: running on CPU — calibration will be slow. "
              f"Use a GPU for best results.")

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report: list = []
    r            = report.append
    best_params: dict = {}
    yaml_path    = Path(config_path) if config_path else CONFIG_PATH

    r(f"# Auto-Calibration Report — {timestamp}\n")
    r(f"- **Objective**: maximise mean detection confidence (favours precision over recall)")
    r(f"- **Sample images**: {len(samples)} from `{root}`")
    r(f"- **Minimum detections required**: {min_det_total}")
    r(f"- **Coarse box_th grid**: {COARSE_BOX}")
    r(f"- **Coarse text_th grid**: {COARSE_TEXT}")
    r(f"- **NMS search range**: {NMS_RANGE}\n")
    r("> **Algorithm**: coarse grid (box×text) → fine grid (±0.04) → NMS sweep "
      "> → outlier bounds fitted from detection distribution\n")
    r("---\n")

    for pass_cfg in passes_cfg:
        pass_name = pass_cfg["name"]
        if target_class and pass_name != target_class:
            continue

        prompts = pass_cfg.get("text_prompts", [])
        cur_box = pass_cfg.get("box_threshold",  0.35)
        cur_txt = pass_cfg.get("text_threshold", 0.25)
        cur_nms = pass_cfg.get("nms_threshold",  0.50)
        cur_oc  = pass_cfg.get("outlier_rejection", {})
        caption = " . ".join(p.strip().rstrip(".") for p in prompts) + " ."

        print(f"\n{'─'*60}")
        print(f"  Pass: {pass_name}  ({len(prompts)} prompts)")
        print(f"  Baseline: box={cur_box}  text={cur_txt}  nms={cur_nms}")
        print(f"{'─'*60}")

        r(f"## Pass: `{pass_name}`")
        r(f"*Baseline: box_th=`{cur_box}`, text_th=`{cur_txt}`, nms_th=`{cur_nms}`*\n")

        # Baseline score
        mc_base, nd_base, _ = _eval_config(
            gdino_model, samples, caption, cur_box, cur_txt, cur_nms, device,
        )
        print(f"  Baseline → conf={mc_base:.4f}  n={nd_base}")
        r(f"**Baseline** → mean_conf=`{mc_base:.4f}`, detections=`{nd_base}`\n")

        # Stage 1: coarse grid
        n_coarse = len(COARSE_BOX) * len(COARSE_TEXT)
        print(f"\n  [1/4] Coarse grid ({n_coarse} combinations) …")
        coarse = _grid_search(
            gdino_model, samples, caption, device, min_det_total,
            COARSE_BOX, COARSE_TEXT, nms_th=cur_nms, label="coarse",
        )
        print(f"  Coarse best → box={coarse['box_th']}  text={coarse['text_th']}"
              f"  conf={coarse['mean_conf']:.4f}  n={coarse['n_det']}")

        # Stage 2: fine grid (±0.04 around coarse best, step 0.02)
        print(f"\n  [2/4] Fine grid (±0.04 around coarse best) …")
        cx, ct   = coarse["box_th"], coarse["text_th"]
        fine_box  = sorted({round(cx + i * 0.02, 3)
                             for i in range(-2, 3) if 0.01 <= cx + i * 0.02 <= 0.95})
        fine_text = sorted({round(ct + i * 0.02, 3)
                             for i in range(-2, 3) if 0.01 <= ct + i * 0.02 <= 0.95})
        fine = _grid_search(
            gdino_model, samples, caption, device, min_det_total,
            fine_box, fine_text, nms_th=cur_nms, label="fine ",
        )
        print(f"  Fine best  → box={fine['box_th']}  text={fine['text_th']}"
              f"  conf={fine['mean_conf']:.4f}  n={fine['n_det']}")

        if fine["mean_conf"] >= coarse["mean_conf"]:
            opt_box, opt_text, opt_conf = fine["box_th"], fine["text_th"], fine["mean_conf"]
        else:
            opt_box, opt_text, opt_conf = coarse["box_th"], coarse["text_th"], coarse["mean_conf"]

        if opt_conf == 0.0:
            print(f"  [WARN] No valid configuration found — keeping baseline parameters.")
            opt_box, opt_text, opt_conf = cur_box, cur_txt, mc_base

        # Stage 3: NMS sweep
        print(f"\n  [3/4] NMS sweep (box={opt_box}, text={opt_text}) …")
        opt_nms = _search_nms_threshold(
            gdino_model, samples, caption,
            opt_box, opt_text, device, min_det_total, NMS_RANGE,
        )
        print(f"  Optimal NMS → {opt_nms}")

        # Stage 4: collect detections → fit outlier bounds
        print(f"\n  [4/4] Collecting detections for outlier param fitting …")
        _, n_final, det_stats = _eval_config(
            gdino_model, samples, caption, opt_box, opt_text, opt_nms, device,
        )
        print(f"  {n_final} detections collected")

        oc_fit = _fit_outlier_params(det_stats)
        if oc_fit:
            print(f"  Fitted outlier params: {oc_fit}")
        else:
            print(f"  [WARN] No detections for fitting — keeping current outlier params.")
            oc_fit = {k: cur_oc[k] for k in
                      ["min_area_ratio", "max_area_ratio",
                       "min_aspect", "max_aspect", "min_confidence"]
                      if k in cur_oc}

        best_params[pass_name] = {
            "box_threshold":  opt_box,
            "text_threshold": opt_text,
            "nms_threshold":  opt_nms,
            "outlier_rejection": oc_fit,
        }

        # Build report table
        r("### Optimised parameters\n")
        r("| Parameter | Baseline | Optimised |")
        r("|:----------|:--------:|:---------:|")
        r(f"| box_threshold  | {cur_box}  | **{opt_box}**  |")
        r(f"| text_threshold | {cur_txt}  | **{opt_text}** |")
        r(f"| nms_threshold  | {cur_nms}  | **{opt_nms}**  |")
        for k in ["min_area_ratio", "max_area_ratio",
                  "min_aspect", "max_aspect", "min_confidence"]:
            r(f"| {k} | {cur_oc.get(k, '—')} | **{oc_fit.get(k, '—')}** |")

        gain = opt_conf - mc_base
        r(f"\n**Result**: mean_conf `{mc_base:.4f}` → `{opt_conf:.4f}` "
          f"({'%+.4f' % gain}) | final detections at optimal params: `{n_final}`\n")
        r("---\n")

    # ── Save report ────────────────────────────────────────────────────────────
    report_path = output_dir / f"auto_calibration_{timestamp}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\n[AUTO-CALIB] Report → {report_path}")

    if not best_params:
        print("[AUTO-CALIB] No passes calibrated — YAML unchanged.")
        return

    # ── Update classes.yaml (backup first) ────────────────────────────────────
    yaml_text   = yaml_path.read_text(encoding="utf-8")
    backup_path = yaml_path.with_suffix(".yaml.bak")
    backup_path.write_text(yaml_text, encoding="utf-8")
    print(f"[AUTO-CALIB] Backup  → {backup_path}")

    for pass_name, params in best_params.items():
        flat = {
            "box_threshold":  params["box_threshold"],
            "text_threshold": params["text_threshold"],
            "nms_threshold":  params["nms_threshold"],
            **params.get("outlier_rejection", {}),
        }
        yaml_text = _update_pass_in_yaml(yaml_text, pass_name, flat)

    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"[AUTO-CALIB] Updated → {yaml_path}")
    print(f"[AUTO-CALIB] Done. Open the report for the full summary.")


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def mask_to_polygon(bool_mask: np.ndarray) -> list | None:
    binary = (bool_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(largest, True)
    approx  = cv2.approxPolyDP(largest, epsilon, True)
    return approx.reshape(-1, 2).tolist() if len(approx) >= 3 else None


CLASS_COLORS = [
    (0, 255, 0),   (255, 0, 0),   (0, 0, 255),   (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0),  (255, 128, 0),
    (128, 0, 255), (0, 128, 255), (255, 128, 128),(128, 255, 128),
    (128, 128, 255),(200, 100, 50),(50, 200, 100), (100, 50, 200),
    (200, 200, 50), (50, 200, 200),(200, 50, 200), (100, 100, 100),
]

# Per-class colors matching the Label Studio label configuration (stored as BGR for OpenCV).
LABEL_COLORS: dict[str, tuple] = {
    "buildings":               (113, 204,  46),   # #2ECC71
    "fuel_infrastructure":     ( 15, 196, 241),   # #F1C40F
    "military_vehicles":       ( 60,  76, 231),   # #E74C3C
    "tanks":                   (173,  68, 142),   # #8E44AD
    "roads_and_tracks":        (141, 140, 127),   # #7F8C8D
    "perimeter_structures":    (219, 152,  52),   # #3498DB
    "communication_and_radar": ( 99,  30, 233),   # #E91E63
}


def _class_color(name: str, fallback_idx: int = 0) -> tuple:
    """Return the BGR color for a class name, falling back to CLASS_COLORS by index."""
    return LABEL_COLORS.get(name, CLASS_COLORS[fallback_idx % len(CLASS_COLORS)])


def draw_results(image_bgr, boxes_xyxy, phrases, masks, pass_labels, pass_color_map):
    vis = image_bgr.copy()
    for i, (box, phrase, plabel) in enumerate(zip(boxes_xyxy, phrases, pass_labels)):
        color = pass_color_map.get(plabel, (0, 255, 0))
        x1, y1, x2, y2 = map(int, box)
        if i < len(masks) and masks[i] is not None:
            overlay = vis.copy()
            overlay[masks[i]] = (
                overlay[masks[i]] * 0.5 + np.array(color) * 0.5
            ).astype(np.uint8)
            vis = overlay
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{plabel}: {phrase}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return vis


def build_ls_result(polygon_pts, label, w, h, score):
    ls_points = [
        [round((pt[0] / w) * 100, 4), round((pt[1] / h) * 100, 4)]
        for pt in polygon_pts
    ]
    return {
        "original_width": w, "original_height": h, "image_rotation": 0,
        "value": {"points": ls_points, "polygonlabels": [label]},
        "id": str(uuid.uuid4())[:10],
        "from_name": "label", "to_name": "image",
        "type": "polygonlabels", "score": float(score),
    }


def build_image_url(file_path: str) -> str:
    raw    = str(Path(file_path).resolve()).replace("\\", "/")
    marker = "google_maps_web/"
    idx    = raw.find(marker)
    rel    = raw[idx + len(marker):] if idx != -1 else raw.split("/")[-1]
    return f"/data/local-files/?d=images/google_maps_web/{rel}"


def _ls_url_to_path(url: str) -> Path | None:
    """
    Resolve a Label Studio file URL to an absolute disk path.

    URL format : /data/local-files/?d=images/google_maps_web/<roi>/<file>
    Disk path  : TILES_ROOT / <roi> / <file>

    Falls back to a recursive filename search in TILES_ROOT when the
    google_maps_web marker is absent (e.g. absolute-path exports).
    """
    try:
        d_value = url.split("?d=")[-1]          # images/google_maps_web/roi/file.jpg
        marker  = "google_maps_web/"
        idx     = d_value.find(marker)
        if idx != -1:
            rel = d_value[idx + len(marker):]   # roi/file.jpg
            return TILES_ROOT / rel
        # fallback: search by filename
        fname   = d_value.split("/")[-1]
        matches = list(TILES_ROOT.rglob(fname))
        return matches[0] if matches else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  SAM FROM MANUAL ANNOTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_sam_from_annotations(
    ls_export_json:   Path,
    sam_predictor,
    output_vis_dir:   Path = None,
    output_json_path: Path = None,
) -> None:
    """
    Inverse pipeline: Label Studio manual BBoxes → SAM masks → LS import JSON.

    Reads a Label Studio export JSON (object-detection project, RectangleLabels),
    runs SAM on every bounding box, and writes:
      - Visualisation images (boxes + masks)  →  outputs/visuals/from_annotations/
      - Label Studio import JSON (PolygonLabels) →  outputs/import_sam_from_annotations.json

    Workflow:
      1. Annotate bounding boxes in Label Studio (RectangleLabels, same class names
         as classes.yaml).
      2. Export → JSON (Label Studio native format).
      3. python pipeline_gdino_sam.py --from-annotations <export.json>
      4. Import the generated JSON into Label Studio to review/correct masks.

    Args:
        ls_export_json   : path to the Label Studio export JSON
        sam_predictor    : loaded SamPredictor instance
        output_vis_dir   : where to write visualisation images (default: visuals/from_annotations/)
        output_json_path : where to write the LS import JSON
    """
    if output_vis_dir is None:
        output_vis_dir = OUTPUT_VIS / "from_annotations"
    if output_json_path is None:
        output_json_path = OUTPUT_JSON / "import_sam_from_annotations.json"

    output_vis_dir   = Path(output_vis_dir)
    output_json_path = Path(output_json_path)
    output_vis_dir.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(ls_export_json, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    print(f"\n[SAM-ANNOT] {len(tasks)} task(s) loaded from {ls_export_json.name}")

    # ── Collect all label names to build a consistent colour map ─────────────
    all_labels: set = set()
    for task in tasks:
        for ann in task.get("annotations", []):
            for res in ann.get("result", []):
                if res.get("type") == "rectanglelabels":
                    all_labels.update(res["value"].get("rectanglelabels", []))

    pass_color_map = {
        lbl: _class_color(lbl, i)
        for i, lbl in enumerate(sorted(all_labels))
    }
    print(f"[SAM-ANNOT] Labels found: {sorted(all_labels)}")

    ls_output_tasks = []
    total_boxes     = 0
    total_masks     = 0
    skipped_tasks   = 0

    for task_idx, task in enumerate(tasks):
        image_url = task.get("data", {}).get("image", "")
        img_path  = _ls_url_to_path(image_url)

        if img_path is None or not img_path.exists():
            print(f"\n  [SKIP {task_idx + 1}/{len(tasks)}] Cannot resolve image: {image_url}")
            skipped_tasks += 1
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"\n  [SKIP {task_idx + 1}/{len(tasks)}] Cannot read: {img_path.name}")
            skipped_tasks += 1
            continue

        img_h, img_w = image_bgr.shape[:2]
        print(f"\n  [{task_idx + 1}/{len(tasks)}] {img_path.name}  ({img_w}×{img_h})")

        # Use the first annotation (highest priority / most recent in LS exports)
        annotations = task.get("annotations", [])
        if not annotations:
            print(f"    [SKIP] No annotations in this task.")
            skipped_tasks += 1
            continue

        # Collect rectanglelabels from the annotation
        boxes_info = []
        for res in annotations[0].get("result", []):
            if res.get("type") != "rectanglelabels":
                continue
            val   = res["value"]
            label = (val.get("rectanglelabels") or ["unknown"])[0]

            if val.get("rotation", 0) != 0:
                print(f"    [WARN] Rotated bbox for '{label}' — "
                      f"rotation ignored, using axis-aligned bounds.")

            # LS stores coords as percentage of image dimensions
            x1 = (val["x"]     / 100) * img_w
            y1 = (val["y"]     / 100) * img_h
            x2 = x1 + (val["width"]  / 100) * img_w
            y2 = y1 + (val["height"] / 100) * img_h

            # Clamp to image bounds
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(img_w), x2), min(float(img_h), y2)

            boxes_info.append({
                "label": label,
                "box":   [x1, y1, x2, y2],
            })

        if not boxes_info:
            print(f"    [SKIP] No rectanglelabels found in annotation.")
            skipped_tasks += 1
            continue

        print(f"    {len(boxes_info)} bbox(es): {[b['label'] for b in boxes_info]}")
        total_boxes += len(boxes_info)

        # ── SAM: one prediction per bounding box ──────────────────────────────
        sam_predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        masks_list, polygons_list, labels_list, boxes_xyxy = [], [], [], []

        for bi in boxes_info:
            box_np = np.array(bi["box"], dtype=np.float32)
            sm, _, _ = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_np,
                multimask_output=False,
            )
            mask    = sm[0]           # bool ndarray (H, W)
            polygon = mask_to_polygon(mask)

            masks_list.append(mask)
            polygons_list.append(polygon)
            labels_list.append(bi["label"])
            boxes_xyxy.append(bi["box"])

            status = (f"polygon OK ({len(polygon)} pts)"
                      if polygon is not None else "no contour — mask empty?")
            print(f"    [{bi['label']}]  box={[round(v,1) for v in bi['box']]}  SAM → {status}")

        # ── Visualisation ─────────────────────────────────────────────────────
        vis_img = draw_results(
            image_bgr,
            boxes_xyxy,       # plain Python lists — draw_results uses map(int, box)
            labels_list,      # "phrases" column: show the class label
            masks_list,
            labels_list,
            pass_color_map,
        )
        vis_path = output_vis_dir / f"{img_path.stem}_sam_result.jpg"
        cv2.imwrite(str(vis_path), vis_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # ── Build LS import task (PolygonLabels only) ─────────────────────────
        ls_results = []
        for bi, polygon in zip(boxes_info, polygons_list):
            if polygon is None:
                continue
            ls_results.append(build_ls_result(
                polygon, bi["label"],
                img_w, img_h,
                score=1.0,    # manual annotation origin → treat as high confidence
            ))
            total_masks += 1

        if ls_results:
            ls_output_tasks.append({
                "data":        {"image": image_url},
                "predictions": [{
                    "model_version": "manual_bbox_SAM_v1",
                    "result":        ls_results,
                }],
            })

    # ── Save JSON ──────────────────────────────────────────────────────────────
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(ls_output_tasks, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"  SAM-FROM-ANNOTATIONS COMPLETED")
    print(f"  Tasks processed  : {len(tasks) - skipped_tasks} / {len(tasks)}")
    print(f"  Skipped          : {skipped_tasks}")
    print(f"  Bounding boxes   : {total_boxes}")
    print(f"  Masks generated  : {total_masks}")
    print(f"  Label Studio JSON: {output_json_path.resolve()}")
    print(f"  Visualisations   : {output_vis_dir.resolve()}")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════════════════
#  LOG
# ═══════════════════════════════════════════════════════════════════════════════

def _write_log_md(roi_stats, timestamp, passes_cfg, total_det, total_tasks):
    log_path = OUTPUT_JSON / "log.md"
    L = []
    a = L.append

    a(f"# Pipeline Log — {timestamp}\n")
    a("## Pass configuration\n")
    for p in passes_cfg:
        oc = p.get("outlier_rejection", {})
        a(f"### `{p['name']}`\n")
        a("| Param | Value |")
        a("|---|---|")
        a(f"| prompts | {len(p['text_prompts'])} |")
        a(f"| box_threshold | {p.get('box_threshold', 0.35)} |")
        a(f"| text_threshold | {p.get('text_threshold', 0.25)} |")
        a(f"| nms_threshold | {p.get('nms_threshold', 0.5)} |")
        a(f"| outlier | {'ON' if oc.get('enabled', True) else 'OFF'} |")
        if oc.get("enabled", True):
            for k in ["min_area_ratio", "max_area_ratio", "min_aspect",
                      "max_aspect", "min_confidence"]:
                a(f"| {k} | {oc.get(k, '—')} |")
        a("")

    g_raw = sum(r["raw"]      for r in roi_stats)
    g_rej = sum(r["rejected"] for r in roi_stats)
    g_fin = sum(r["final"]    for r in roi_stats)
    g_no  = sum(len(r["no_det"]) for r in roi_stats)

    a("## Global summary\n")
    a("| Metric | Value |")
    a("|---|---|")
    a(f"| ROIs | {len(roi_stats)} |")
    a(f"| Raw detections (pre-NMS) | {g_raw} |")
    a(f"| Rejected (NMS + outlier) | {g_rej} |")
    a(f"| Final detections | {g_fin} |")
    a(f"| Rejection rate | {g_rej / max(g_raw, 1) * 100:.1f}% |")
    a(f"| Tiles without detections | {g_no} |")
    a(f"| Label Studio tasks | {total_tasks} |")
    a("")

    for roi in roi_stats:
        a(f"---\n## ROI: `{roi['name']}`\n")
        a("| Metric | Value |")
        a("|---|---|")
        a(f"| Tiles | {roi['tiles']} |")
        a(f"| Raw | {roi['raw']} |")
        a(f"| Rejected | {roi['rejected']} |")
        a(f"| Final | {roi['final']} |")
        a("")

        if roi["tile_details"]:
            a("### Per-image detail\n")
            a("| Image | Raw | Rejected | Final | Passes with detections |")
            a("|---|:---:|:---:|:---:|---|")
            for td in roi["tile_details"]:
                a(f"| `{td['tile']}` | {td['raw']} | {td['rej']} | "
                  f"{td['fin']} | {td['passes']} |")
            a("")

        if roi["no_det"]:
            a("### Tiles without final detections\n")
            a("| Image | Reason |")
            a("|---|---|")
            for name, reason in roi["no_det"]:
                a(f"| `{name}` | {reason} |")
            a("")

    all_no = [(r["name"], n, m) for r in roi_stats for n, m in r["no_det"]]
    if all_no:
        a("---\n## All tiles without detection\n")
        a("| ROI | Image | Reason |")
        a("|---|---|---|")
        for rn, tn, m in all_no:
            a(f"| `{rn}` | `{tn}` | {m} |")
        a("")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  [LOG]  {log_path.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    print("=" * 70)
    print("  MULTI-PASS PIPELINE: GroundingDINO → Outlier → SAM → Label Studio")
    print("  Mode: one GDINO call per prompt, in-pass IoU-NMS")
    print("=" * 70)

    cfg    = load_config()
    passes = cfg["passes"]

    if not HAS_TORCH:
        print("[ERROR] PyTorch not installed."); sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device.upper()}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    gdino_model, sam_predictor = load_models(device)

    pass_color_map = {
        p["name"]: _class_color(p["name"], i)
        for i, p in enumerate(passes)
    }

    OUTPUT_VIS.mkdir(parents=True, exist_ok=True)
    roi_dirs = sorted([d for d in TILES_ROOT.iterdir() if d.is_dir()])
    if not roi_dirs:
        print(f"[ERROR] No ROI subdirectories in {TILES_ROOT}")
        sys.exit(1)
    print(f"\n[TILES] {len(roi_dirs)} ROIs: {[d.name for d in roi_dirs]}")

    all_ls_tasks     = []
    total_detections = 0
    log_roi_stats    = []
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for roi_dir in roi_dirs:
        tiles = sorted([
            f for f in roi_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")
        ])
        if not tiles:
            continue

        print(f"\n{'─'*60}")
        print(f"  ROI: {roi_dir.name}  ({len(tiles)} tiles)")
        print(f"{'─'*60}")

        vis_roi = OUTPUT_VIS / roi_dir.name
        vis_roi.mkdir(parents=True, exist_ok=True)

        roi_stat = {
            "name": roi_dir.name, "tiles": len(tiles),
            "raw": 0, "rejected": 0, "final": 0,
            "tile_details": [], "no_det": [],
        }

        for tile_path in tiles:
            print(f"\n  → {tile_path.name}")

            image_bgr = cv2.imread(str(tile_path))
            if image_bgr is None:
                print(f"    [SKIP] Cannot read image")
                roi_stat["no_det"].append((tile_path.name, "cannot read image"))
                continue
            h, w = image_bgr.shape[:2]

            _, image_transformed = load_image(str(tile_path))

            # Accumulators for ALL passes on this tile
            all_boxes, all_logits, all_phrases, all_pass_labels = [], [], [], []
            tile_raw, tile_rej = 0, 0
            passes_with_det    = []

            # ── MULTI-PASS: one class per pass, one prompt per GDINO call ─────
            for p_cfg in passes:
                pass_name  = p_cfg["name"]
                prompts    = p_cfg.get("text_prompts", [])
                box_th     = p_cfg.get("box_threshold",  0.35)
                text_th    = p_cfg.get("text_threshold",  0.25)
                nms_th     = p_cfg.get("nms_threshold",   0.50)
                oc         = p_cfg.get("outlier_rejection", {})
                oc_enabled = oc.get("enabled", True)

                # One GDINO call per prompt → aggregate → in-pass NMS
                ba, logits, phrases, n_raw = run_gdino_per_prompt(
                    gdino_model, image_transformed,
                    prompts, box_th, text_th, nms_th,
                    w, h, device,
                )

                if n_raw == 0:
                    continue

                n_after_nms = len(ba)
                print(f"    [{pass_name}] {n_raw} raw → {n_after_nms} after NMS",
                      end="")

                # Outlier rejection
                if oc_enabled:
                    bf, lf, pf = reject_outliers(
                        ba, logits, phrases, w, h,
                        min_area_ratio=oc.get("min_area_ratio",  0.0005),
                        max_area_ratio=oc.get("max_area_ratio",  0.5),
                        min_aspect    =oc.get("min_aspect",      0.1),
                        max_aspect    =oc.get("max_aspect",      10.0),
                        min_confidence=oc.get("min_confidence",  0.30),
                    )
                else:
                    bf, lf, pf = ba, logits, phrases

                n_filt  = len(bf)
                tile_raw += n_raw
                tile_rej += (n_raw - n_filt)
                print(f" → {n_filt} valid")

                if n_filt > 0:
                    valid = [(b, l, p) for b, l, p in zip(bf, lf, pf) if p.strip()]
                    if valid:
                        all_boxes.append(torch.stack([v[0] for v in valid]))
                        all_logits.append(torch.stack([v[1] for v in valid]))
                        all_phrases.extend([v[2] for v in valid])
                        all_pass_labels.extend([pass_name] * len(valid))
                        passes_with_det.append(pass_name)

            # ── Merge all passes ──────────────────────────────────────────────
            tile_final = len(all_phrases)
            roi_stat["raw"]      += tile_raw
            roi_stat["rejected"] += tile_rej
            roi_stat["final"]    += tile_final

            roi_stat["tile_details"].append({
                "tile":   tile_path.name,
                "raw":    tile_raw,
                "rej":    tile_rej,
                "fin":    tile_final,
                "passes": ", ".join(passes_with_det) if passes_with_det else "—",
            })

            if tile_final == 0:
                reason = ("no detections in any pass" if tile_raw == 0
                          else "all rejected by NMS or outlier filter")
                print(f"    [INFO] {reason}")
                roi_stat["no_det"].append((tile_path.name, reason))
                continue

            merged_boxes  = torch.cat(all_boxes,  dim=0)
            merged_logits = torch.cat(all_logits, dim=0)
            print(f"    [MERGED] {tile_final} detections across "
                  f"{len(passes_with_det)} pass(es): {passes_with_det}")

            # ── SAM ──────────────────────────────────────────────────────────
            sam_predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            masks_list, polygons_list = [], []
            for box in merged_boxes:
                sm, _, _ = sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box.cpu().numpy(),
                    multimask_output=False,
                )
                masks_list.append(sm[0])
                polygons_list.append(mask_to_polygon(sm[0]))

            total_detections += tile_final

            # ── Visualization ─────────────────────────────────────────────────
            vis_img = draw_results(
                image_bgr, merged_boxes, all_phrases,
                masks_list, all_pass_labels, pass_color_map,
            )
            cv2.imwrite(
                str(vis_roi / f"{tile_path.stem}_result.jpg"),
                vis_img, [cv2.IMWRITE_JPEG_QUALITY, 90],
            )

            # ── Label Studio JSON ─────────────────────────────────────────────
            results = []
            for box, logit, plabel, polygon in zip(
                merged_boxes, merged_logits, all_pass_labels, polygons_list
            ):
                if polygon is not None:
                    results.append(build_ls_result(polygon, plabel, w, h, logit))

            if results:
                all_ls_tasks.append({
                    "data": {"image": build_image_url(str(tile_path))},
                    "predictions": [{
                        "model_version": "GroundingDINO_SAM_multipass_v2",
                        "result": results,
                    }],
                })

        log_roi_stats.append(roi_stat)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    OUTPUT_JSON.mkdir(parents=True, exist_ok=True)
    out_json = OUTPUT_JSON / "import_to_labelstudio.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_ls_tasks, f, indent=2, ensure_ascii=False)

    _write_log_md(log_roi_stats, run_ts, passes, total_detections, len(all_ls_tasks))

    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETED")
    print(f"  Total detections : {total_detections}")
    print(f"  Label Studio tasks: {len(all_ls_tasks)}")
    print(f"  JSON : {out_json.resolve()}")
    print(f"  Log  : {(OUTPUT_JSON / 'log.md').resolve()}")
    print(f"  Vis  : {OUTPUT_VIS.resolve()}")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    parser = argparse.ArgumentParser(
        description="GroundingDINO + SAM multi-pass labeling pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Production run (GroundingDINO + SAM + Label Studio JSON):
  python pipeline_gdino_sam.py

  # Calibrate all classes on 5 random production tiles:
  python pipeline_gdino_sam.py --calibrate

  # Calibrate one class on dedicated test images:
  python pipeline_gdino_sam.py --calibrate \\
      --class military_vehicles \\
      --images /path/to/test_images \\
      --n-samples 10
        """,
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Run calibration mode: threshold grid search + per-prompt stats. "
             "No SAM, no Label Studio output.",
    )
    parser.add_argument(
        "--auto-calibrate", action="store_true",
        help="Automatically optimise box_threshold, text_threshold, nms_threshold "
             "and outlier_rejection params by maximising mean detection confidence. "
             "Updates classes.yaml in place (backup created). No SAM, no Label Studio output.",
    )
    parser.add_argument(
        "--from-annotations", dest="annotations_json", default=None, metavar="JSON",
        help="Path to a Label Studio export JSON (RectangleLabels, object-detection project). "
             "Runs SAM on each bounding box and produces PolygonLabels + visualisations. "
             "No GroundingDINO needed.",
    )
    parser.add_argument(
        "--class", dest="target_class", default=None, metavar="CLASS_NAME",
        help="Calibrate only this pass (e.g. military_vehicles). "
             "Default: all passes.",
    )
    parser.add_argument(
        "--images", dest="images_dir", default=None, metavar="DIR",
        help="Directory with sample images for calibration. "
             "Default: production TILES_ROOT (random sample).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=5, metavar="N",
        help="Number of sample images to use in calibration (default: 5).",
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", default=None, metavar="DIR",
        help="Output directory for calibration report "
             "(default: outputs/calibration/).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.calibrate:
        print("=" * 70)
        print("  CALIBRATION MODE — GroundingDINO threshold & prompt tuning")
        print("=" * 70)

        if not HAS_TORCH:
            print("[ERROR] PyTorch not installed."); sys.exit(1)

        cfg    = load_config()
        passes = cfg["passes"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DEVICE] {device.upper()}")
        if device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

        gdino_model = load_gdino_only(device)

        calibrate_classes(
            passes_cfg      = passes,
            gdino_model     = gdino_model,
            device          = device,
            images_dir      = Path(args.images_dir) if args.images_dir else None,
            target_class    = args.target_class,
            n_sample_images = args.n_samples,
            output_dir      = Path(args.output_dir) if args.output_dir else None,
        )

    elif args.auto_calibrate:
        print("=" * 70)
        print("  AUTO-CALIBRATION — maximising mean detection confidence")
        print("=" * 70)

        if not HAS_TORCH:
            print("[ERROR] PyTorch not installed."); sys.exit(1)

        cfg    = load_config()
        passes = cfg["passes"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DEVICE] {device.upper()}")
        if device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

        gdino_model = load_gdino_only(device)

        auto_calibrate_classes(
            passes_cfg      = passes,
            gdino_model     = gdino_model,
            device          = device,
            images_dir      = Path(args.images_dir) if args.images_dir else None,
            target_class    = args.target_class,
            n_sample_images = args.n_samples,
            output_dir      = Path(args.output_dir) if args.output_dir else None,
            config_path     = CONFIG_PATH,
        )

    elif args.annotations_json:
        print("=" * 70)
        print("  SAM FROM ANNOTATIONS — manual BBoxes → SAM masks → LS import")
        print("=" * 70)

        json_path = Path(args.annotations_json)
        if not json_path.exists():
            print(f"[ERROR] File not found: {json_path}"); sys.exit(1)

        if not HAS_TORCH:
            print("[ERROR] PyTorch not installed."); sys.exit(1)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DEVICE] {device.upper()}")
        if device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

        sam_predictor = load_sam_only(device)

        run_sam_from_annotations(
            ls_export_json = json_path,
            sam_predictor  = sam_predictor,
        )

    else:
        run_pipeline()
