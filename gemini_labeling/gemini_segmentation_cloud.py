#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  gemini_segmentation_cloud.py  v2  —  Colab / Vertex AI / local            ║
# ║                                                                              ║
# ║  PROBLEM SOLVED: "Image generation is not available in your country"        ║
# ║  → Run this script from Google Colab (requests originate from Google's      ║
# ║    US servers) or Vertex AI (us-central1) to bypass the restriction.        ║
# ║                                                                              ║
# ║  ENVIRONMENTS SUPPORTED:                                                     ║
# ║    1. Google Colab   — recommended, free, no GCP billing needed             ║
# ║    2. Vertex AI      — requires GCP project + billing enabled               ║
# ║    3. Local          — only works if image generation is available for you   ║
# ║                                                                              ║
# ║  HOW TO USE IN COLAB:                                                        ║
# ║    a) Upload this file to Google Drive or paste into a Colab cell            ║
# ║    b) Upload your images to Google Drive:                                    ║
# ║         My Drive/TFM/src/images/google_maps_web/<roi>/<tiles>.jpeg          ║
# ║    c) Set GEMINI_API_KEY below                                               ║
# ║    d) Run — outputs are saved back to Google Drive                           ║
# ║                                                                              ║
# ║  HOW TO USE WITH VERTEX AI:                                                  ║
# ║    a) Set USE_VERTEX_AI = True                                               ║
# ║    b) Set GCP_PROJECT_ID and GCP_LOCATION                                   ║
# ║    c) Authenticate: gcloud auth application-default login                    ║
# ║       or in Colab: from google.colab import auth; auth.authenticate_user()  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ══════════════════════════════════════════════════════════════════════════════
#  ★  CONFIGURATION — edit these values before running  ★
# ══════════════════════════════════════════════════════════════════════════════

# ── API key (AI Studio — https://aistudio.google.com/apikey) ─────────────────
GEMINI_API_KEY = "<YOUR_GEMINI_API_KEY>"  # https://aistudio.google.com/apikey

# ── Model ─────────────────────────────────────────────────────────────────────
# Models that support image OUTPUT via response_modalities=["TEXT","IMAGE"]:
#   gemini-2.5-flash-image   ← nano banana: fast, cost-effective
#   gemini-3.1-flash-image   ← latest (Jun 2026), best quality + speed
#   gemini-3-pro-image       ← professional grade, slower/costlier
# Models that do NOT support image output (text only):
#   gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-*  → will error
GEMINI_MODEL = "gemini-3.1-flash-image"   # swap to gemini-3.1-flash-image for latest

# ── Vertex AI (set True to use GCP Vertex AI instead of AI Studio) ────────────
USE_VERTEX_AI  = False
GCP_PROJECT_ID = ""           # your GCP project ID, e.g. "my-project-123"
GCP_LOCATION   = "us-central1"

# ── Image source ──────────────────────────────────────────────────────────────
# Colab / Drive path (relative to "My Drive"):
DRIVE_IMAGE_DIR  = "TFM/src/images/google_maps_web"
DRIVE_OUTPUT_DIR = "TFM/gemini_labeling/outputs"

# Local path (used when NOT in Colab):
LOCAL_IMAGE_DIR  = "c:/TFM/src/images/google_maps_web"
LOCAL_OUTPUT_DIR = "c:/TFM/gemini_labeling/outputs"

# ── ROI filter (None = all; "Cabo_Noval" = only that folder) ─────────────────
ROI_FILTER = None

# ── Rate limiting ─────────────────────────────────────────────────────────────
REQUEST_DELAY_SEC = 3.0   # seconds between API calls
MAX_RETRIES       = 3

# ── Mask / color settings ─────────────────────────────────────────────────────
MIN_CONTOUR_AREA_FRAC = 0.0005   # ignore regions < 0.05% of image area
COLOR_TOLERANCE       = 30       # ±tolerance per channel when matching class colors
POLYGON_EPSILON       = 0.001    # approxPolyDP: fraction of perimeter kept as epsilon
                                 # lower → more points, higher accuracy (0.001 = 0.1%)
                                 # higher → fewer points, coarser polygons (0.004 = 0.4%)


# ══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT DETECTION & SETUP
# ══════════════════════════════════════════════════════════════════════════════

import sys, subprocess

def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# Detect Google Colab
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Install dependencies if needed
for pkg in ["google-genai", "opencv-python-headless" if IN_COLAB else "opencv-python"]:
    try:
        if "opencv" in pkg:
            import cv2  # noqa: F401
        else:
            from google import genai  # noqa: F401
    except ImportError:
        print(f"[SETUP] Installing {pkg} …")
        _install(pkg)

# Mount Google Drive (Colab only)
if IN_COLAB:
    from google.colab import drive
    if not __import__("os").path.exists("/content/drive/MyDrive"):
        print("[SETUP] Mounting Google Drive …")
        drive.mount("/content/drive")
    else:
        print("[SETUP] Google Drive already mounted.")

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTS & PATHS
# ══════════════════════════════════════════════════════════════════════════════

import json
import uuid
import time
import argparse
from pathlib import Path

import cv2
import numpy as np

# Resolve working paths
if IN_COLAB:
    DRIVE_ROOT   = Path("/content/drive/MyDrive")
    TILES_ROOT   = DRIVE_ROOT / DRIVE_IMAGE_DIR
    OUTPUT_DIR   = DRIVE_ROOT / DRIVE_OUTPUT_DIR
else:
    TILES_ROOT   = Path(LOCAL_IMAGE_DIR)
    OUTPUT_DIR   = Path(LOCAL_OUTPUT_DIR)

OUTPUT_JSON  = OUTPUT_DIR / "Gemini_to_label_studio.json"
OUTPUT_MASKS = OUTPUT_DIR / "masks"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

print(f"[PATHS] Images : {TILES_ROOT}")
print(f"[PATHS] Output : {OUTPUT_DIR}")
print(f"[ENV]   Colab  : {IN_COLAB}")
print(f"[ENV]   Vertex : {USE_VERTEX_AI}")
print(f"[MODEL] {GEMINI_MODEL}")

# ══════════════════════════════════════════════════════════════════════════════
#  CLASS DEFINITIONS  (RGB colors — matching pipeline_gdino_sam.py LABEL_COLORS)
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
#  MASK-GENERATION PROMPT  (Gemini outputs an IMAGE, not JSON/text)
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

Use EXACTLY these RGB colors:
{_COLOR_SPEC}
  - background: RGB(0, 0, 0)  #000000

Class descriptions:
  - buildings: flat-roof structures, warehouses, hangars — any building footprint
  - fuel_infrastructure: cylindrical/circular storage tanks, fuel depots
  - military_vehicles: tanks, APCs, armored vehicles, military trucks (aerial view)
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
#  GEMINI CLIENT  (AI Studio or Vertex AI)
# ══════════════════════════════════════════════════════════════════════════════

def _init_client():
    """
    Returns (client, types) for the google-genai SDK.
    Supports both AI Studio (API key) and Vertex AI (GCP auth).
    """
    from google import genai
    from google.genai import types

    if USE_VERTEX_AI:
        if not GCP_PROJECT_ID:
            raise ValueError(
                "GCP_PROJECT_ID must be set when USE_VERTEX_AI = True.\n"
                "In Colab also run:  from google.colab import auth; auth.authenticate_user()"
            )
        print(f"[AUTH] Using Vertex AI — project={GCP_PROJECT_ID}  location={GCP_LOCATION}")
        client = genai.Client(
            vertexai=True,
            project=GCP_PROJECT_ID,
            location=GCP_LOCATION,
        )
    else:
        print("[AUTH] Using AI Studio API key")
        client = genai.Client(api_key=GEMINI_API_KEY)

    return client, types


# ══════════════════════════════════════════════════════════════════════════════
#  GEMINI CALL — REQUEST IMAGE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def _decode_image_part(part) -> "np.ndarray | None":
    """Decode an inline_data image part → RGB numpy array."""
    try:
        data = part.inline_data.data
        if isinstance(data, str):
            import base64
            data = base64.b64decode(data)
        arr     = np.frombuffer(data, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def call_gemini_for_mask(client, types, image_path: Path) -> "np.ndarray | None":
    """
    Asks Gemini (GEMINI_MODEL) to generate a colored segmentation mask.
    Returns an RGB numpy array (512×512×3) or None on failure.

    Use models that support response_modalities=["TEXT","IMAGE"]:
      gemini-2.5-flash-image   ← nano banana, recommended
      gemini-3.1-flash-image   ← latest (Jun 2026)
      gemini-3-pro-image       ← professional grade
    """
    with open(image_path, "rb") as fh:
        raw = fh.read()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".tif": "image/tiff", ".tiff": "image/tiff"}
    mime = mime_map.get(image_path.suffix.lower(), "image/jpeg")

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=raw, mime_type=mime),
                    MASK_GEN_PROMPT,
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    temperature=0.1,
                ),
            )
            for part in resp.candidates[0].content.parts:
                if hasattr(part, "inline_data") and \
                        part.inline_data.mime_type.startswith("image/"):
                    img = _decode_image_part(part)
                    if img is not None:
                        return cv2.resize(img, (512, 512),
                                          interpolation=cv2.INTER_NEAREST)
            raise ValueError("No image part in Gemini response")

        except Exception as exc:
            msg = str(exc)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                print(f"  [RATE LIMIT] waiting 60s …")
                time.sleep(60)
            elif attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt * 3
                print(f"  [RETRY {attempt + 1}/{MAX_RETRIES}] {exc} — waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"  [FAIL] {image_path.name}: {exc}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  MASK → CONTOURS → LABEL STUDIO PREDICTIONS  (polygonlabels format)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_binary_mask(mask_rgb: np.ndarray, class_name: str) -> np.ndarray:
    """Return 0/255 uint8 array where pixels match the class color (±COLOR_TOLERANCE)."""
    target = np.array(ALL_CLASSES[class_name]["color_rgb"], dtype=np.int32)
    dist   = np.abs(mask_rgb.astype(np.int32) - target).max(axis=2)
    return (dist <= COLOR_TOLERANCE).astype(np.uint8) * 255


def _get_contours(binary: np.ndarray, min_area: float,
                  is_thing: bool = False) -> list:
    if is_thing:
        # Thing classes (vehicles, buildings, …): skip MORPH_CLOSE so that nearby
        # instances are NOT merged into a single blob. Only OPEN to remove pixel noise.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    else:
        # Stuff classes (forest, roads, …): CLOSE fills small holes/gaps inside large
        # continuous regions; OPEN removes isolated specks.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        clean  = cv2.morphologyEx(clean,  cv2.MORPH_OPEN,  kernel, iterations=1)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def _contour_to_polygon_pct(contour, size: int = 512) -> "list | None":
    epsilon = POLYGON_EPSILON * cv2.arcLength(contour, True)
    approx  = cv2.approxPolyDP(contour, epsilon, True)
    pts     = approx.reshape(-1, 2)
    if len(pts) < 3:
        return None
    return [[round(float(p[0]) / size * 100, 4),
             round(float(p[1]) / size * 100, 4)] for p in pts]


def _contour_to_bbox_pct(contour, size: int = 512) -> list:
    x, y, w, h = cv2.boundingRect(contour)
    mn = lambda v: round(v / size * 100, 4)
    xmn, ymn = mn(x), mn(y)
    xmx, ymx = mn(x + w), mn(y + h)
    return [[xmn, ymn], [xmx, ymn], [xmx, ymx], [xmn, ymx]]


def _ls_entry(points, label, img_w, img_h, score=1.0) -> dict:
    return {
        "original_width": img_w, "original_height": img_h, "image_rotation": 0,
        "value": {"points": points, "polygonlabels": [label]},
        "id": str(uuid.uuid4())[:10],
        "from_name": "label", "to_name": "image",
        "type": "polygonlabels", "score": float(score),
    }


def mask_to_ls_annotations(mask_rgb: np.ndarray,
                            img_w: int, img_h: int, size: int = 512) -> list:
    ls_results = []
    min_area   = size * size * MIN_CONTOUR_AREA_FRAC

    # Thing classes: contour polygon + bounding box, no merging of nearby instances
    for class_name in THING_CLASSES:
        binary   = _extract_binary_mask(mask_rgb, class_name)
        contours = _get_contours(binary, min_area, is_thing=True)
        for contour in contours:
            poly = _contour_to_polygon_pct(contour, size)
            if poly:
                ls_results.append(_ls_entry(poly, class_name, img_w, img_h))
            ls_results.append(
                _ls_entry(_contour_to_bbox_pct(contour, size), class_name, img_w, img_h)
            )

    # Stuff classes: contour polygon only (no bounding box), fill gaps in large regions
    for class_name in STUFF_CLASSES:
        binary   = _extract_binary_mask(mask_rgb, class_name)
        contours = _get_contours(binary, min_area, is_thing=False)
        for contour in contours:
            poly = _contour_to_polygon_pct(contour, size)
            if poly:
                ls_results.append(_ls_entry(poly, class_name, img_w, img_h))

    return ls_results


# ══════════════════════════════════════════════════════════════════════════════
#  PNG OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def save_mask_pngs(image_path: Path, mask_rgb: np.ndarray,
                   out_path: Path, size: int = 512):
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pure class-color mask
    cv2.imwrite(str(out_path.parent / (out_path.stem + "_pure.png")), mask_bgr)

    # Overlay on original
    img = cv2.imread(str(image_path))
    if img is not None:
        img = cv2.resize(img, (size, size))
        cv2.imwrite(str(out_path), cv2.addWeighted(img, 0.55, mask_bgr, 0.45, 0))


# ══════════════════════════════════════════════════════════════════════════════
#  LABEL STUDIO URL
# ══════════════════════════════════════════════════════════════════════════════

def build_ls_url(image_path: Path) -> str:
    raw    = str(image_path.resolve()).replace("\\", "/")
    marker = "google_maps_web/"
    idx    = raw.find(marker)
    return (f"/data/local-files/?d=images/google_maps_web/{raw[idx + len(marker):]}"
            if idx != -1
            else f"/data/local-files/?d=images/google_maps_web/{image_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(roi_filter=None):
    def _log(msg): print(msg, flush=True)

    # Discover images
    all_images = sorted(
        p for p in TILES_ROOT.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if roi_filter:
        all_images = [p for p in all_images
                      if roi_filter.lower() in p.parent.name.lower()]

    _log(f"[SCAN] TILES_ROOT = {TILES_ROOT}")
    _log(f"[SCAN] Images found: {len(all_images)}")
    if not all_images:
        _log(f"[ERROR] No images found in {TILES_ROOT} — check the path and Drive mount.")
        return

    # Init Gemini
    client, types = _init_client()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MASKS.mkdir(parents=True, exist_ok=True)

    ls_tasks  = []
    n_proc = n_skip = 0

    # Load existing JSON to allow resuming interrupted runs
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON) as fh:
            ls_tasks = json.load(fh)
        done_urls = {t["data"]["image"] for t in ls_tasks}
        _log(f"[RESUME] {len(done_urls)} images already done — skipping them")
    else:
        done_urls = set()

    for idx, img_path in enumerate(all_images, 1):
        roi = img_path.parent.name
        url = build_ls_url(img_path)

        if url in done_urls:
            _log(f"[{idx:>3}/{len(all_images)}] SKIP {img_path.name}")
            n_proc += 1
            continue

        _log(f"[{idx:>3}/{len(all_images)}] {roi}/{img_path.name}")

        img = cv2.imread(str(img_path))
        if img is None:
            _log(f"  [WARN] Cannot read image — skipping")
            n_skip += 1
            continue
        img_h, img_w = img.shape[:2]

        mask_rgb = call_gemini_for_mask(client, types, img_path)
        if mask_rgb is None:
            n_skip += 1
            continue

        ls_results = mask_to_ls_annotations(mask_rgb, img_w, img_h)
        _log(f"  → {len(ls_results)} LS annotations")

        if ls_results:
            ls_tasks.append({
                "data": {"image": url},
                "predictions": [{
                    "model_version": f"gemini_{GEMINI_MODEL.replace('-','_')}_v2",
                    "result": ls_results,
                }],
            })

        mask_path = OUTPUT_MASKS / roi / f"{img_path.stem}_mask.png"
        save_mask_pngs(img_path, mask_rgb, mask_path)
        n_proc += 1

        # Save JSON after every image (allows resuming if interrupted)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
            json.dump(ls_tasks, fh, indent=2, ensure_ascii=False)

        if idx < len(all_images):
            time.sleep(REQUEST_DELAY_SEC)

    sep = "=" * 60
    print(sep, flush=True)
    print(f"  COMPLETE — {n_proc} processed, {n_skip} skipped", flush=True)
    print(f"  JSON  → {OUTPUT_JSON}", flush=True)
    print(f"  Masks → {OUTPUT_MASKS}", flush=True)
    print(sep, flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if IN_COLAB:
        run(roi_filter=ROI_FILTER)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--roi",   default=ROI_FILTER)
        parser.add_argument("--model", default=GEMINI_MODEL)
        parser.add_argument("--delay", type=float, default=REQUEST_DELAY_SEC)
        args = parser.parse_args()
        GEMINI_MODEL      = args.model
        REQUEST_DELAY_SEC = args.delay
        run(roi_filter=args.roi)
