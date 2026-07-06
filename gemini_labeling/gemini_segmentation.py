#!/usr/bin/env python3
"""
gemini_segmentation.py  v2
==========================
Processes aerial/satellite tiles using Gemini Flash IMAGE GENERATION:
  1. Asks Gemini to generate a COLORED SEGMENTATION MASK image (not polygon coords).
  2. Extracts per-class contours from the mask with OpenCV.
  3. Builds Label Studio JSON (polygon + bbox for thing classes).
  4. Saves 512×512 PNG masks.

Classes (matching pipeline_gdino_sam.py LABEL_COLORS):
  THING — polygon mask + bounding-box polygon in JSON + painted in PNG:
    buildings, fuel_infrastructure, military_vehicles, communication_and_radar
  STUFF — painted in PNG only, no Label Studio annotations:
    roads_and_tracks, forest, perimeter_structures

Outputs:
  - c:/TFM/gemini_labeling/outputs/Gemini_to_label_studio.json
  - c:/TFM/gemini_labeling/outputs/masks/<roi>/<stem>_mask.png        (overlay)
  - c:/TFM/gemini_labeling/outputs/masks/<roi>/<stem>_mask_pure.png   (pure class colors)

Usage:
    python gemini_segmentation.py                          # all images
    python gemini_segmentation.py --roi Cabo_Noval         # single ROI
    python gemini_segmentation.py --dry-run                # list images only
    python gemini_segmentation.py --model gemini-2.5-flash
"""

import json
import uuid
import time
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np

# ─── API Key ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = "<YOUR_GEMINI_API_KEY>"  # https://aistudio.google.com/apikey

# ─── Model ────────────────────────────────────────────────────────────────────
# gemini-2.0-flash supports image output via response_modalities=["IMAGE"]
# gemini-2.5-flash  is more capable (better segmentation quality)
GEMINI_MODEL = "gemini-2.0-flash"

# ─── Paths ────────────────────────────────────────────────────────────────────
TILES_ROOT   = Path("c:/TFM/src/images/google_maps_web")
OUTPUT_DIR   = Path("c:/TFM/gemini_labeling/outputs")
OUTPUT_JSON  = OUTPUT_DIR / "Gemini_to_label_studio.json"
OUTPUT_MASKS = OUTPUT_DIR / "masks"

IMAGE_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
REQUEST_DELAY_SEC  = 2.0   # seconds between API calls (rate-limit safety)
MAX_RETRIES        = 3

# ── Contour filtering ─────────────────────────────────────────────────────────
# Minimum area of a contour as a fraction of the total image area.
# Below this → noise / ignore.
MIN_CONTOUR_AREA_FRAC = 0.0005   # 0.05 % of 512×512 ≈ 131 px²

# Color-matching tolerance when scanning the returned mask image for each class.
# Gemini may not return exact RGB values, so we accept ±TOLERANCE per channel.
COLOR_TOLERANCE = 30

# ══════════════════════════════════════════════════════════════════════════════
#  CLASS DEFINITIONS  (colors in RGB — same values as pipeline_gdino_sam.py)
#
#  Pipeline stores BGR; RGB equivalents:
#    buildings               ( 47, 47,  66) BGR → ( 66,  47,  47) RGB  #422f2f
#    fuel_infrastructure     ( 15,196, 241) BGR → (241, 196,  15) RGB  #F1C40F
#    military_vehicles       ( 60, 76, 231) BGR → (231,  76,  60) RGB  #E74C3C
#    communication_and_radar ( 99, 30, 233) BGR → (233,  30,  99) RGB  #E91E63
#    roads_and_tracks        (141,140, 127) BGR → (127, 140, 141) RGB  #7F8C8D
#    forest                  ( 40,151,  32) BGR → ( 32, 151,  40) RGB  #209728
#    perimeter_structures    (219,152,  52) BGR → ( 52, 152, 219) RGB  #3498DB
# ══════════════════════════════════════════════════════════════════════════════

THING_CLASSES: dict[str, dict] = {
    "buildings":               {"color_rgb": (66,  47,  47),  "hex": "#422f2f"},
    "fuel_infrastructure":     {"color_rgb": (241, 196, 15),  "hex": "#F1C40F"},
    "military_vehicles":       {"color_rgb": (231, 76,  60),  "hex": "#E74C3C"},
    "communication_and_radar": {"color_rgb": (233, 30,  99),  "hex": "#E91E63"},
}

STUFF_CLASSES: dict[str, dict] = {
    "roads_and_tracks":     {"color_rgb": (127, 140, 141), "hex": "#7F8C8D"},
    "forest":               {"color_rgb": (32,  151, 40),  "hex": "#209728"},
    "perimeter_structures": {"color_rgb": (52,  152, 219), "hex": "#3498DB"},
}

ALL_CLASSES: dict[str, dict] = {**THING_CLASSES, **STUFF_CLASSES}


# ══════════════════════════════════════════════════════════════════════════════
#  MASK-GENERATION PROMPT
#  We ask Gemini to OUTPUT AN IMAGE (the segmentation mask), not text/JSON.
# ══════════════════════════════════════════════════════════════════════════════

_COLOR_SPEC = "\n".join(
    f"  - {name}: RGB({c['color_rgb'][0]}, {c['color_rgb'][1]}, {c['color_rgb'][2]})  {c['hex']}"
    for name, c in ALL_CLASSES.items()
)

MASK_GEN_PROMPT = f"""
You are a semantic segmentation AI for aerial/satellite imagery.

Analyze this image of a military or industrial installation and generate a
PURE SEMANTIC SEGMENTATION MASK — output an IMAGE, not text.

Requirements for the output image:
  - Same dimensions as the input (512×512 pixels)
  - Every pixel painted with a single flat class color (no gradients, no blending)
  - Background / uncategorized areas: pure BLACK (0, 0, 0)

Use EXACTLY these RGB colors for each class:
{_COLOR_SPEC}
  - background: RGB(0, 0, 0)  #000000

Class descriptions:
  - buildings: flat-roof structures, warehouses, hangars — any building footprint
  - fuel_infrastructure: cylindrical or circular storage tanks, fuel depots
  - military_vehicles: tanks, APCs, armored vehicles, military trucks (viewed from above)
  - communication_and_radar: communication towers, antennas, radar dishes
  - roads_and_tracks: paved roads, dirt roads, access tracks, vehicle paths
  - forest: woodland, tree clusters, dense vegetation patches
  - perimeter_structures: security fences, perimeter walls, compound boundaries

Rules:
  - Paint EVERY visible pixel with the appropriate class color.
  - Segment the ENTIRE image — leave nothing unclassified except truly empty ground.
  - Use PURE, FLAT colors — no anti-aliasing at class boundaries.
  - Output ONLY the segmentation mask image. No text. No labels. No overlays.
"""


# ══════════════════════════════════════════════════════════════════════════════
#  GEMINI CLIENT
# ══════════════════════════════════════════════════════════════════════════════

def _init_gemini_client():
    """
    Initialises the Gemini client.
    Prefers the newer google-genai SDK; falls back to google-generativeai.
    Install: pip install google-genai
    """
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
        return client, types, "google-genai"
    except ImportError:
        pass
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        return genai, None, "google-generativeai"
    except ImportError:
        raise ImportError(
            "\nNeither 'google-genai' nor 'google-generativeai' is installed.\n"
            "Install:  pip install google-genai\n"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  GEMINI CALL — IMAGE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def _decode_image_part(part) -> np.ndarray | None:
    """
    Decode an inline_data image part from Gemini's response into a numpy array.
    Returns an RGB array (H, W, 3) or None on failure.
    """
    try:
        data = part.inline_data.data
        if isinstance(data, str):
            import base64
            data = base64.b64decode(data)
        arr = np.frombuffer(data, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def _call_new_sdk_image(client, types, image_path: Path) -> np.ndarray | None:
    """
    Calls Gemini with response_modalities=["IMAGE"] to generate a
    colored segmentation mask. Returns an RGB numpy array (512×512).
    """
    with open(image_path, "rb") as fh:
        raw = fh.read()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".tif": "image/tiff", ".tiff": "image/tiff"}
    mime = mime_map.get(image_path.suffix.lower(), "image/jpeg")

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=raw, mime_type=mime),
            MASK_GEN_PROMPT,
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            temperature=0.1,
        ),
    )

    for part in resp.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data.mime_type.startswith("image/"):
            img = _decode_image_part(part)
            if img is not None:
                return cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
    return None


def _call_old_sdk_image(genai_module, image_path: Path) -> np.ndarray | None:
    """Fallback using google-generativeai SDK."""
    try:
        import PIL.Image
    except ImportError:
        raise ImportError("Pillow required for google-generativeai: pip install Pillow")

    model = genai_module.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config=genai_module.GenerationConfig(temperature=0.1),
    )
    pil_img = PIL.Image.open(str(image_path))
    resp = model.generate_content([pil_img, MASK_GEN_PROMPT])

    for part in resp.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data.mime_type.startswith("image/"):
            img = _decode_image_part(part)
            if img is not None:
                return cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
    return None


def call_gemini_for_mask(client, types_mod, sdk_name: str, image_path: Path) -> np.ndarray | None:
    """
    Asks Gemini to generate a colored segmentation mask image.
    Returns an RGB numpy array (512×512×3) or None on repeated failure.
    """
    for attempt in range(MAX_RETRIES):
        try:
            if sdk_name == "google-genai":
                result = _call_new_sdk_image(client, types_mod, image_path)
            else:
                result = _call_old_sdk_image(client, image_path)

            if result is not None:
                return result
            raise ValueError("No image part returned by Gemini")

        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt * 2
                logging.warning(f"  [RETRY {attempt + 1}/{MAX_RETRIES}] {exc} — waiting {wait}s")
                time.sleep(wait)
            else:
                logging.error(f"  [FAIL] {image_path.name}: {exc}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  MASK → CONTOURS → LABEL STUDIO POLYGONS
# ══════════════════════════════════════════════════════════════════════════════

def _extract_binary_mask(mask_rgb: np.ndarray, class_name: str) -> np.ndarray:
    """
    Returns a binary mask for one class by matching pixels to its RGB color
    within ±COLOR_TOLERANCE per channel.
    """
    target = np.array(ALL_CLASSES[class_name]["color_rgb"], dtype=np.int32)
    diff   = mask_rgb.astype(np.int32) - target
    dist   = np.abs(diff).max(axis=2)          # Chebyshev distance per pixel
    return (dist <= COLOR_TOLERANCE).astype(np.uint8) * 255


def _get_contours(binary: np.ndarray, min_area: float) -> list:
    """Find external contours above min_area, after morphological cleanup."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean  = cv2.morphologyEx(clean,  cv2.MORPH_OPEN,  kernel, iterations=1)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def _contour_to_polygon_pct(contour, size: int = 512) -> list | None:
    """
    Approximate a contour as a polygon and return vertices as [x%, y%] pairs.
    Uses a small epsilon to keep enough detail without over-smoothing.
    """
    perimeter = cv2.arcLength(contour, True)
    epsilon   = 0.004 * perimeter                # smaller = more points = more detail
    approx    = cv2.approxPolyDP(contour, epsilon, True)
    pts       = approx.reshape(-1, 2)
    if len(pts) < 3:
        return None
    return [
        [round(float(p[0]) / size * 100, 4), round(float(p[1]) / size * 100, 4)]
        for p in pts
    ]


def _contour_to_bbox_pct(contour, size: int = 512) -> list:
    """Return 4-point bounding-box polygon from a contour in % coords."""
    x, y, w, h = cv2.boundingRect(contour)
    xmn = round(x / size * 100, 4)
    ymn = round(y / size * 100, 4)
    xmx = round((x + w) / size * 100, 4)
    ymx = round((y + h) / size * 100, 4)
    return [[xmn, ymn], [xmx, ymn], [xmx, ymx], [xmn, ymx]]


def _ls_entry(points: list, label: str, img_w: int, img_h: int, score: float = 1.0) -> dict:
    return {
        "original_width":  img_w,
        "original_height": img_h,
        "image_rotation":  0,
        "value": {"points": points, "polygonlabels": [label]},
        "id":        str(uuid.uuid4())[:10],
        "from_name": "label",
        "to_name":   "image",
        "type":      "polygonlabels",
        "score":     float(score),
    }


def mask_to_ls_annotations(
    mask_rgb: np.ndarray,
    img_w: int,
    img_h: int,
    size: int = 512,
) -> list:
    """
    Processes the colored mask image to produce Label Studio annotations.

    For THING classes:
      - One entry per contour: the mask polygon (from OpenCV contour)
      - One entry per contour: the bounding box as a 4-point rectangle polygon

    For STUFF classes:
      - Nothing added to the JSON (only painted in the PNG).
    """
    ls_results: list = []
    min_area = size * size * MIN_CONTOUR_AREA_FRAC

    for class_name in THING_CLASSES:
        binary   = _extract_binary_mask(mask_rgb, class_name)
        contours = _get_contours(binary, min_area)
        logging.debug(f"    {class_name}: {len(contours)} contour(s)")

        for contour in contours:
            poly = _contour_to_polygon_pct(contour, size)
            if poly and len(poly) >= 3:
                ls_results.append(_ls_entry(poly, class_name, img_w, img_h, 1.0))

            bbox_poly = _contour_to_bbox_pct(contour, size)
            ls_results.append(_ls_entry(bbox_poly, class_name, img_w, img_h, 1.0))

    return ls_results


# ══════════════════════════════════════════════════════════════════════════════
#  PNG OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def save_mask_pngs(
    image_path: Path,
    mask_rgb:   np.ndarray,
    out_path:   Path,
    size:       int = 512,
):
    """
    Saves two 512×512 PNGs:
      <stem>_mask_pure.png  — pure segmentation mask (class colors on black)
      <stem>_mask.png       — original image blended with the mask overlay
    """
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pure segmentation mask
    pure_path = out_path.parent / (out_path.stem + "_pure.png")
    cv2.imwrite(str(pure_path), mask_bgr)

    # Overlay on original image
    img = cv2.imread(str(image_path))
    if img is not None:
        img     = cv2.resize(img, (size, size))
        overlay = cv2.addWeighted(img, 0.55, mask_bgr, 0.45, 0)
        cv2.imwrite(str(out_path), overlay)


# ══════════════════════════════════════════════════════════════════════════════
#  LABEL STUDIO URL
# ══════════════════════════════════════════════════════════════════════════════

def build_ls_url(image_path: Path) -> str:
    raw    = str(image_path.resolve()).replace("\\", "/")
    marker = "google_maps_web/"
    idx    = raw.find(marker)
    if idx != -1:
        return f"/data/local-files/?d=images/google_maps_web/{raw[idx + len(marker):]}"
    return f"/data/local-files/?d=images/google_maps_web/{image_path.name}"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(roi_filter: str | None = None, dry_run: bool = False):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    # ── Discover images ───────────────────────────────────────────────────────
    all_images = sorted(
        p for p in TILES_ROOT.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if roi_filter:
        all_images = [p for p in all_images
                      if roi_filter.lower() in p.parent.name.lower()]

    log.info(f"Found {len(all_images)} image(s) in {TILES_ROOT}")
    if not all_images:
        log.error("No images found. Check TILES_ROOT.")
        return

    if dry_run:
        log.info("[DRY RUN] Would process:")
        for p in all_images[:15]:
            log.info(f"  {p.parent.name}/{p.name}")
        if len(all_images) > 15:
            log.info(f"  … and {len(all_images) - 15} more")
        return

    # ── Init Gemini ───────────────────────────────────────────────────────────
    log.info(f"Initialising Gemini (model={GEMINI_MODEL}) …")
    client, types_mod, sdk_name = _init_gemini_client()
    log.info(f"SDK: {sdk_name}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MASKS.mkdir(parents=True, exist_ok=True)

    # ── Per-image loop ────────────────────────────────────────────────────────
    ls_tasks: list = []
    n_proc = 0
    n_skip = 0

    for idx, img_path in enumerate(all_images, 1):
        roi = img_path.parent.name
        log.info(f"[{idx:>3}/{len(all_images)}] {roi}/{img_path.name}")

        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("  Cannot read image — skipping")
            n_skip += 1
            continue
        img_h, img_w = img.shape[:2]

        # ── Ask Gemini to generate colored segmentation mask ─────────────────
        mask_rgb = call_gemini_for_mask(client, types_mod, sdk_name, img_path)
        if mask_rgb is None:
            log.warning("  Gemini returned no mask — skipping")
            n_skip += 1
            continue

        # ── Extract Label Studio annotations from mask ────────────────────────
        ls_results = mask_to_ls_annotations(mask_rgb, img_w, img_h)
        log.info(f"  → {len(ls_results)} LS annotations")

        if ls_results:
            ls_tasks.append({
                "data": {"image": build_ls_url(img_path)},
                "predictions": [{
                    "model_version": f"gemini_{GEMINI_MODEL.replace('-', '_')}_v2",
                    "result":        ls_results,
                }],
            })

        # ── Save PNG masks ────────────────────────────────────────────────────
        mask_path = OUTPUT_MASKS / roi / f"{img_path.stem}_mask.png"
        save_mask_pngs(img_path, mask_rgb, mask_path)

        n_proc += 1
        if idx < len(all_images):
            time.sleep(REQUEST_DELAY_SEC)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(ls_tasks, fh, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    sep = "=" * 60
    log.info(sep)
    log.info("  GEMINI SEGMENTATION COMPLETE")
    log.info(f"  Processed          : {n_proc} / {len(all_images)}")
    log.info(f"  Skipped            : {n_skip}")
    log.info(f"  Tasks with results : {len(ls_tasks)}")
    log.info(f"  JSON  → {OUTPUT_JSON}")
    log.info(f"  Masks → {OUTPUT_MASKS}")
    log.info(sep)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gemini Flash image-generation segmentation → Label Studio JSON + mask PNGs"
    )
    parser.add_argument("--roi",     default=None,
                        help="Filter by ROI folder name (partial match, e.g. Cabo_Noval)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List images without making API calls")
    parser.add_argument("--model",   default=GEMINI_MODEL,
                        help=f"Gemini model (default: {GEMINI_MODEL})")
    parser.add_argument("--delay",   type=float, default=REQUEST_DELAY_SEC,
                        help=f"Seconds between API calls (default: {REQUEST_DELAY_SEC})")
    args = parser.parse_args()

    GEMINI_MODEL      = args.model
    REQUEST_DELAY_SEC = args.delay

    run(roi_filter=args.roi, dry_run=args.dry_run)
