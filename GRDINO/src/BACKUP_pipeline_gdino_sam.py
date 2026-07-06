"""
pipeline_gdino_sam.py
=====================
Multi-pass Pipeline: GroundingDINO → Outlier Rejection → SAM → Label Studio JSON

Flujo:
  1. Carga pasadas (passes) desde config/classes.yaml
  2. Lee tiles de cada ROI en src/images/google_maps_web/
  3. Por cada tile, ejecuta N pasadas de GroundingDINO (una por grupo de clases)
  4. Outlier Rejection por pasada (con parámetros independientes)
  5. Fusiona todas las detecciones de todas las pasadas
  6. Inferencia SAM (box prompt) sobre las cajas fusionadas
  7. Salida:
     - Imagen con TODAS las bounding boxes + máscaras  (outputs/visuals/)
     - JSON unificado para Label Studio                (outputs/import_to_labelstudio.json)
     - log.md con estadísticas detalladas
"""

import os, sys, json, uuid, yaml, cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# ─── Rutas ───────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
CONFIG_PATH  = BASE_DIR / "config" / "classes.yaml"
TILES_ROOT   = Path("c:/TFM/src/images/google_maps_web")
OUTPUT_VIS   = BASE_DIR / "outputs" / "visuals"
OUTPUT_JSON  = BASE_DIR / "outputs"

SAM_CHECKPOINT = Path("c:/TFM/Labeling/scripts/sam_vit_h_4b8939.pth")
SAM_MODEL_TYPE = "vit_h"
GDINO_WEIGHTS = BASE_DIR / "weights" / "groundingdino_swint_ogc.pth"
GDINO_CONFIG  = None

# ─── Importaciones condicionales ─────────────────────────────────────────────
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
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    """Lee classes.yaml con formato multi-pass."""
    if not CONFIG_PATH.exists():
        print(f"[ERROR] No se encuentra {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    passes = cfg.get("passes", [])
    print(f"[CONFIG] {len(passes)} pasadas configuradas:")
    for p in passes:
        n_prompts = len(p.get("text_prompts", []))
        outlier_st = "ON" if p.get("outlier_rejection", {}).get("enabled", True) else "OFF"
        print(f"  • {p['name']}  ({n_prompts} prompts, "
              f"box={p.get('box_threshold', 0.35)}, "
              f"text={p.get('text_threshold', 0.25)}, "
              f"outlier={outlier_st})")
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
#  MODELOS
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
    raise FileNotFoundError("No se encontró la config de GroundingDINO.")


def load_models(device: str):
    if not HAS_GDINO:
        raise ImportError("groundingdino no instalado.")
    cfg_path = GDINO_CONFIG or _find_gdino_config()
    print(f"[GDINO] Config: {cfg_path}")
    print(f"[GDINO] Weights: {GDINO_WEIGHTS}")
    gdino_model = load_model(cfg_path, str(GDINO_WEIGHTS), device=device)

    if not HAS_SAM:
        raise ImportError("segment_anything no instalado.")
    print(f"[SAM] Checkpoint: {SAM_CHECKPOINT}")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device)
    return gdino_model, SamPredictor(sam)


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTLIER REJECTION
# ═══════════════════════════════════════════════════════════════════════════════

def reject_outliers(boxes, logits, phrases, img_w, img_h,
                    min_area_ratio=0.0005, max_area_ratio=0.5,
                    min_aspect=0.1, max_aspect=10.0,
                    min_confidence=0.30):
    """Filtra bounding boxes anómalas. Devuelve (boxes, logits, phrases) filtrados."""
    keep = []
    img_area = img_w * img_h
    for i, (box, logit) in enumerate(zip(boxes, logits)):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        ratio = (bw * bh) / img_area
        aspect = bw / max(bh, 1e-6)
        if ratio < min_area_ratio or ratio > max_area_ratio:
            continue
        if aspect < min_aspect or aspect > max_aspect:
            continue
        if float(logit) < min_confidence:
            continue
        keep.append(i)

    if not keep:
        return boxes[:0], logits[:0], []
    n_rej = len(boxes) - len(keep)
    if n_rej > 0:
        print(f"      [OUTLIER] Rechazadas {n_rej}/{len(boxes)} cajas")
    return boxes[keep], logits[keep], [phrases[i] for i in keep]


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════

def mask_to_polygon(bool_mask: np.ndarray) -> list | None:
    binary = (bool_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return approx.reshape(-1, 2).tolist() if len(approx) >= 3 else None


CLASS_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (128, 0, 255), (0, 128, 255), (255, 128, 128), (128, 255, 128),
    (128, 128, 255), (200, 100, 50), (50, 200, 100), (100, 50, 200),
    (200, 200, 50), (50, 200, 200), (200, 50, 200), (100, 100, 100),
]


def draw_results(image_bgr, boxes_xyxy, phrases, masks, pass_labels, pass_color_map):
    """Dibuja bounding boxes + máscaras. pass_labels indica el pass_name de cada detección."""
    vis = image_bgr.copy()
    for i, (box, phrase, plabel) in enumerate(zip(boxes_xyxy, phrases, pass_labels)):
        color = pass_color_map.get(plabel, (0, 255, 0))
        x1, y1, x2, y2 = map(int, box)
        if i < len(masks) and masks[i] is not None:
            overlay = vis.copy()
            overlay[masks[i]] = (overlay[masks[i]] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            vis = overlay
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{plabel}: {phrase}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return vis


def build_ls_result(polygon_pts, phrase, w, h, score):
    ls_points = [[round((pt[0]/w)*100, 4), round((pt[1]/h)*100, 4)] for pt in polygon_pts]
    return {
        "original_width": w, "original_height": h, "image_rotation": 0,
        "value": {"points": ls_points, "polygonlabels": [phrase]},
        "id": str(uuid.uuid4())[:10],
        "from_name": "label", "to_name": "image",
        "type": "polygonlabels", "score": float(score),
    }


def build_image_url(file_path: str) -> str:
    raw = str(Path(file_path).resolve()).replace("\\", "/")
    marker = "google_maps_web/"
    idx = raw.find(marker)
    rel = raw[idx + len(marker):] if idx != -1 else raw.split("/")[-1]
    return f"/data/local-files/?d=images/google_maps_web/{rel}"


# ═══════════════════════════════════════════════════════════════════════════════
#  LOG GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _write_log_md(roi_stats, timestamp, passes_cfg, total_det, total_tasks):
    log_path = OUTPUT_JSON / "log.md"
    L = []
    a = L.append

    a(f"# Pipeline Log — {timestamp}\n")
    a(f"## Configuración de pasadas\n")
    for p in passes_cfg:
        oc = p.get("outlier_rejection", {})
        a(f"### `{p['name']}`\n")
        a(f"| Param | Valor |")
        a(f"|---|---|")
        a(f"| prompts | {len(p['text_prompts'])} |")
        a(f"| box_threshold | {p.get('box_threshold', 0.35)} |")
        a(f"| text_threshold | {p.get('text_threshold', 0.25)} |")
        a(f"| outlier | {'ON' if oc.get('enabled', True) else 'OFF'} |")
        if oc.get("enabled", True):
            for k in ["min_area_ratio","max_area_ratio","min_aspect","max_aspect","min_confidence"]:
                a(f"| {k} | {oc.get(k, '—')} |")
        a("")

    g_raw = sum(r["raw"] for r in roi_stats)
    g_rej = sum(r["rejected"] for r in roi_stats)
    g_fin = sum(r["final"] for r in roi_stats)
    g_no = sum(len(r["no_det"]) for r in roi_stats)

    a("## Resumen Global\n")
    a("| Métrica | Valor |")
    a("|---|---|")
    a(f"| ROIs | {len(roi_stats)} |")
    a(f"| Detecciones brutas | {g_raw} |")
    a(f"| Outliers rechazados | {g_rej} |")
    a(f"| Detecciones finales | {g_fin} |")
    a(f"| Tasa rechazo | {g_rej/max(g_raw,1)*100:.1f}% |")
    a(f"| Tiles sin detección | {g_no} |")
    a(f"| Tareas Label Studio | {total_tasks} |")
    a("")

    for roi in roi_stats:
        a(f"---\n## ROI: `{roi['name']}`\n")
        a("| Métrica | Valor |")
        a("|---|---|")
        a(f"| Tiles | {roi['tiles']} |")
        a(f"| Brutas | {roi['raw']} |")
        a(f"| Rechazadas | {roi['rejected']} |")
        a(f"| Finales | {roi['final']} |")
        a("")

        if roi["tile_details"]:
            a("### Detalle por imagen\n")
            a("| Imagen | Brutas | Rechazadas | Finales | Pasadas con detección |")
            a("|---|:---:|:---:|:---:|---|")
            for td in roi["tile_details"]:
                a(f"| `{td['tile']}` | {td['raw']} | {td['rej']} | {td['fin']} | {td['passes']} |")
            a("")

        if roi["no_det"]:
            a("### ⚠️ Imágenes sin detecciones finales\n")
            a("| Imagen | Motivo |")
            a("|---|---|")
            for name, reason in roi["no_det"]:
                a(f"| `{name}` | {reason} |")
            a("")

    all_no = [(r["name"], n, m) for r in roi_stats for n, m in r["no_det"]]
    if all_no:
        a("---\n## 📋 Completo: tiles sin detección\n")
        a("| ROI | Imagen | Motivo |")
        a("|---|---|---|")
        for rn, tn, m in all_no:
            a(f"| `{rn}` | `{tn}` | {m} |")
        a("")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n  [LOG] {log_path.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL — MULTI-PASS
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    print("=" * 70)
    print("  PIPELINE MULTI-PASS: GroundingDINO → Outlier → SAM → Label Studio")
    print("=" * 70)

    cfg = load_config()
    passes = cfg["passes"]

    if not HAS_TORCH:
        print("[ERROR] PyTorch no instalado."); sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device.upper()}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    gdino_model, sam_predictor = load_models(device)

    # Asignar un color por pass
    pass_color_map = {}
    for i, p in enumerate(passes):
        pass_color_map[p["name"]] = CLASS_COLORS[i % len(CLASS_COLORS)]

    OUTPUT_VIS.mkdir(parents=True, exist_ok=True)
    roi_dirs = sorted([d for d in TILES_ROOT.iterdir() if d.is_dir()])
    if not roi_dirs:
        print(f"[ERROR] No ROIs en {TILES_ROOT}"); sys.exit(1)
    print(f"\n[TILES] {len(roi_dirs)} ROIs: {[d.name for d in roi_dirs]}")

    all_ls_tasks = []
    total_detections = 0
    log_roi_stats = []
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for roi_dir in roi_dirs:
        tiles = sorted([f for f in roi_dir.iterdir()
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")])
        if not tiles:
            continue

        print(f"\n{'─'*50}")
        print(f"  ROI: {roi_dir.name}  ({len(tiles)} tiles)")
        print(f"{'─'*50}")

        vis_roi = OUTPUT_VIS / roi_dir.name
        vis_roi.mkdir(parents=True, exist_ok=True)

        roi_stat = {"name": roi_dir.name, "tiles": len(tiles),
                    "raw": 0, "rejected": 0, "final": 0,
                    "tile_details": [], "no_det": []}

        for tile_path in tiles:
            print(f"\n  → {tile_path.name}")

            image_bgr = cv2.imread(str(tile_path))
            if image_bgr is None:
                print(f"    [SKIP] No se pudo leer")
                roi_stat["no_det"].append((tile_path.name, "no se pudo leer"))
                continue
            h, w = image_bgr.shape[:2]

            image_source, image_transformed = load_image(str(tile_path))

            # Acumuladores para TODAS las pasadas de este tile
            all_boxes, all_logits, all_phrases, all_pass_labels = [], [], [], []
            tile_raw, tile_rej = 0, 0
            passes_with_det = []

            # ── MULTI-PASS ──
            for p_cfg in passes:
                pass_name = p_cfg["name"]
                prompts = p_cfg.get("text_prompts", [])
                box_th = p_cfg.get("box_threshold", 0.35)
                text_th = p_cfg.get("text_threshold", 0.25)
                oc = p_cfg.get("outlier_rejection", {})
                oc_enabled = oc.get("enabled", True)

                text_prompt = " . ".join(prompts) + " ."

                boxes, logits, phrases = predict(
                    model=gdino_model, image=image_transformed,
                    caption=text_prompt, box_threshold=box_th,
                    text_threshold=text_th, device=device,
                )

                n_raw = len(boxes)
                if n_raw == 0:
                    continue

                # Convertir a píxeles
                ba = boxes.clone()
                ba[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
                ba[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
                ba[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
                ba[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h

                print(f"    [{pass_name}] {n_raw} brutas", end="")

                if oc_enabled:
                    bf, lf, pf = reject_outliers(
                        ba, logits, phrases, w, h,
                        min_area_ratio=oc.get("min_area_ratio", 0.0005),
                        max_area_ratio=oc.get("max_area_ratio", 0.5),
                        min_aspect=oc.get("min_aspect", 0.1),
                        max_aspect=oc.get("max_aspect", 10.0),
                        min_confidence=oc.get("min_confidence", 0.30),
                    )
                else:
                    bf, lf, pf = ba, logits, phrases

                n_filt = len(bf)
                tile_raw += n_raw
                tile_rej += (n_raw - n_filt)
                print(f" → {n_filt} válidas")

                if n_filt > 0:
                    # Filtrar detecciones con frase vacía
                    valid = [(b, l, p) for b, l, p in zip(bf, lf, pf) if p.strip()]
                    if valid:
                        v_boxes = torch.stack([v[0] for v in valid])
                        v_logits = torch.stack([v[1] for v in valid])
                        v_phrases = [v[2] for v in valid]
                        passes_with_det.append(pass_name)
                        all_boxes.append(v_boxes)
                        all_logits.append(v_logits)
                        all_phrases.extend(v_phrases)
                        all_pass_labels.extend([pass_name] * len(valid))
                        n_filt = len(valid)

            # ── Fusionar resultados de todas las pasadas ──
            tile_final = len(all_phrases)
            roi_stat["raw"] += tile_raw
            roi_stat["rejected"] += tile_rej
            roi_stat["final"] += tile_final

            roi_stat["tile_details"].append({
                "tile": tile_path.name, "raw": tile_raw,
                "rej": tile_rej, "fin": tile_final,
                "passes": ", ".join(passes_with_det) if passes_with_det else "—",
            })

            if tile_final == 0:
                reason = "sin detecciones en ninguna pasada" if tile_raw == 0 \
                    else "todas rechazadas por outlier rejection"
                print(f"    [INFO] {reason}")
                roi_stat["no_det"].append((tile_path.name, reason))
                continue

            merged_boxes = torch.cat(all_boxes, dim=0)
            merged_logits = torch.cat(all_logits, dim=0)

            print(f"    [MERGED] {tile_final} detecciones totales de "
                  f"{len(passes_with_det)} pasada(s)")

            # ── SAM ──
            sam_predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            masks_list, polygons_list = [], []
            for box in merged_boxes:
                sm, _, _ = sam_predictor.predict(
                    point_coords=None, point_labels=None,
                    box=box.cpu().numpy(), multimask_output=False)
                masks_list.append(sm[0])
                polygons_list.append(mask_to_polygon(sm[0]))

            total_detections += tile_final

            # ── VISUALIZACIÓN (una imagen con todo) ──
            vis_img = draw_results(image_bgr, merged_boxes, all_phrases,
                                   masks_list, all_pass_labels, pass_color_map)
            cv2.imwrite(str(vis_roi / f"{tile_path.stem}_result.jpg"),
                        vis_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # ── LABEL STUDIO JSON (un task por tile, con todas las detecciones) ──
            # Usa pass_name como label (no la frase bruta de GDINO)
            results = []
            for box, logit, plabel, polygon in zip(
                merged_boxes, merged_logits, all_pass_labels, polygons_list
            ):
                if polygon is not None:
                    results.append(build_ls_result(polygon, plabel, w, h, logit))

            if results:
                all_ls_tasks.append({
                    "data": {"image": build_image_url(str(tile_path))},
                    "predictions": [{"model_version": "GroundingDINO_SAM_multipass",
                                     "result": results}]
                })

        log_roi_stats.append(roi_stat)

    # ── Guardar JSON ──
    out_json = OUTPUT_JSON / "import_to_labelstudio.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_ls_tasks, f, indent=2, ensure_ascii=False)

    # ── Generar log.md ──
    _write_log_md(log_roi_stats, run_ts, passes, total_detections, len(all_ls_tasks))

    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETADO")
    print(f"  Detecciones totales: {total_detections}")
    print(f"  Tareas Label Studio: {len(all_ls_tasks)}")
    print(f"  JSON: {out_json.resolve()}")
    print(f"  Log:  {(OUTPUT_JSON / 'log.md').resolve()}")
    print(f"  Vis:  {OUTPUT_VIS.resolve()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_pipeline()
