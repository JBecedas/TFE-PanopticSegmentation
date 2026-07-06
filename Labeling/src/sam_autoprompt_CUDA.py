import os
import json
import numpy as np
import cv2
from pathlib import Path

try:
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    HAS_SAM = True
except ImportError:
    HAS_SAM = False

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "data" / "dataset_index.json"
MASK_DIR = BASE_DIR / "data" / "masks"
MASK_DIR.mkdir(parents=True, exist_ok=True)

SAM_CHECKPOINT = BASE_DIR / "scripts" / "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"


def assert_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "\n❌ CUDA NO DISPONIBLE\n"
            "Soluciones:\n"
            "1. Instala PyTorch con CUDA\n"
            "2. Verifica drivers NVIDIA\n"
            "3. Ejecuta: nvidia-smi\n"
        )

    print(f"✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
    print(f"🔥 VRAM total: {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)} GB")


def load_sam_pipeline():
    if not HAS_SAM:
        raise ImportError("Falta torch o segment-anything")

    if not SAM_CHECKPOINT.exists():
        raise FileNotFoundError(f"Falta checkpoint: {SAM_CHECKPOINT}")

    assert_cuda()

    device = torch.device("cuda")

    print("🚀 Cargando SAM en GPU...")

    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device=device)
    sam.eval()

    print(f"📍 Modelo en: {next(sam.parameters()).device}")

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    return mask_generator


def process_masks_to_polygons(sam_output):
    polygons_data = []

    sorted_masks = sorted(sam_output, key=lambda x: x['area'], reverse=True)

    for mask_dict in sorted_masks:
        bool_mask = mask_dict['segmentation']

        binary_img = (bool_mask.astype(np.uint8)) * 255

        contours, _ = cv2.findContours(
            binary_img,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE  # más rápido que TC89
        )

        if not contours:
            continue

        largest_contour = max(contours, key=cv2.contourArea)

        epsilon = 0.002 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(approx) < 3:
            continue

        flat_poly = approx.reshape(-1, 2).tolist()

        polygons_data.append({
            "area": float(mask_dict['area']),
            "bbox": mask_dict['bbox'],
            "predicted_iou": float(mask_dict['predicted_iou']),
            "polygon": flat_poly
        })

    return polygons_data


def run_auto_prompt():
    print("----- SAM Auto-Prompter (CUDA MODE) -----")

    if not INDEX_PATH.exists():
        raise FileNotFoundError("Falta dataset_index.json")

    with open(INDEX_PATH, 'r') as f:
        dataset = json.load(f)

    generator = load_sam_pipeline()

    print("🔥 Iniciando inferencia en GPU...")

    for idx, item in enumerate(dataset):
        if item['processed_sam']:
            continue

        image_path = item['file_path']
        print(f"[{idx+1}/{len(dataset)}] {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ⚡ Inferencia optimizada
        with torch.inference_mode(), torch.cuda.amp.autocast():
            masks = generator.generate(image)

        polygons = process_masks_to_polygons(masks)

        out_path = MASK_DIR / f"{item['id']}_sam.json"
        with open(out_path, 'w') as f:
            json.dump(polygons, f)

        item['processed_sam'] = True

    with open(INDEX_PATH, 'w') as f:
        json.dump(dataset, f, indent=4)

    print("\n✅ Procesamiento completado en GPU")


if __name__ == "__main__":
    run_auto_prompt()