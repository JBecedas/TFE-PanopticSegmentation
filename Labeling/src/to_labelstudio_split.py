"""
to_labelstudio_split.py
-----------------------
Versión alternativa de to_labelstudio.py que genera un JSON independiente
por cada ROI (región de interés), produciendo archivos manejables que se
pueden importar individualmente en Label Studio.

Salida:  exports/split/<ROI_name>.json
"""

import json
import cv2
import uuid
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "data" / "dataset_index.json"
MASK_DIR = BASE_DIR / "data" / "masks"
EXPORTS_DIR = BASE_DIR / "exports"
SPLIT_DIR = EXPORTS_DIR / "split"

def build_image_url(file_path: str) -> str:
    """Convierte ruta absoluta Windows a URL de Label Studio Local Storage."""
    raw_path = str(Path(file_path).resolve()).replace("\\", "/")
    marker = "google_maps_web/"
    idx = raw_path.find(marker)
    if idx != -1:
        relative_part = raw_path[idx + len(marker):]
    else:
        relative_part = raw_path.split("/")[-1]
    return f"/data/local-files/?d=images/google_maps_web/{relative_part}"


def build_task(item: dict, sam_polygons: list, w: int, h: int) -> dict:
    """Construye una tarea de Label Studio a partir de un item del dataset."""
    results = []
    for poly_obj in sam_polygons:
        poly_points = poly_obj['polygon']

        ls_points = []
        for point in poly_points:
            px = round((point[0] / w) * 100.0, 4)
            py = round((point[1] / h) * 100.0, 4)
            ls_points.append([px, py])

        region_id = str(uuid.uuid4())[:10]

        results.append({
            "original_width": w,
            "original_height": h,
            "image_rotation": 0,
            "value": {
                "points": ls_points,
                "polygonlabels": ["Revisame"]
            },
            "id": region_id,
            "from_name": "label",
            "to_name": "image",
            "type": "polygonlabels",
            "score": poly_obj.get('predicted_iou', 0.9)
        })

    return {
        "data": {
            "image": build_image_url(item['file_path'])
        },
        "predictions": [{
            "model_version": "SAM_Auto",
            "result": results
        }]
    }


def generate_split_json():
    print("----- Formateando para Label Studio (SPLIT por ROI) -----")
    if not INDEX_PATH.exists():
        print("Corre primero discovery y sam_autoprompt")
        return

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    with open(INDEX_PATH, 'r') as f:
        dataset = json.load(f)

    # Agrupar items por ROI
    roi_groups = defaultdict(list)
    for item in dataset:
        roi_groups[item['roi']].append(item)

    total_tasks = 0
    total_files = 0

    for roi_name, items in sorted(roi_groups.items()):
        tasks = []
        for item in items:
            mask_file = MASK_DIR / f"{item['id']}_sam.json"
            if not mask_file.exists():
                continue

            with open(mask_file, 'r') as f:
                sam_polygons = json.load(f)

            img = cv2.imread(item['file_path'])
            if img is None:
                continue
            h, w = img.shape[:2]

            tasks.append(build_task(item, sam_polygons, w, h))

        if not tasks:
            continue

        out_path = SPLIT_DIR / f"{roi_name}.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)

        total_files += 1
        total_tasks += len(tasks)
        print(f"  {roi_name}: {len(tasks)} tareas → {out_path.name}")

    print(f"\n✓ {total_files} archivos generados en {SPLIT_DIR.resolve()}")
    print(f"  Total de tareas: {total_tasks}")
    print("Importa cada JSON individualmente en Label Studio.")


if __name__ == "__main__":
    generate_split_json()
