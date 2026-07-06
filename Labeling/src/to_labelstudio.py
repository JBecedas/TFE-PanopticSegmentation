import json
import cv2
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "data" / "dataset_index.json"
MASK_DIR = BASE_DIR / "data" / "masks"
EXPORTS_DIR = BASE_DIR / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)

# URL Root Asumida. Cuando montes imágenes locales en Label Studio Server local:
# Normalmente usarás LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT e importarás.
# Aquí formateamos un JSON que se puede subir en "Import Tasks" en UI de LS.
# LS buscará mapear $data.image

def generate_ls_json():
    print("----- Formateando para Label Studio -----")
    if not INDEX_PATH.exists():
        print("Corre primero discovery y sam_autoprompt")
        return

    with open(INDEX_PATH, 'r') as f:
        dataset = json.load(f)

    ls_tasks = []

    for item in dataset:
        mask_file = MASK_DIR / f"{item['id']}_sam.json"
        if not mask_file.exists():
            continue
            
        with open(mask_file, 'r') as f:
            sam_polygons = json.load(f)
            
        # Determinar dimensiones de la imagen
        # cv2 es super eficiente leyendo cabeceras sin cargar matriz si no es necesario
        # o simplemente cargandola
        img = cv2.imread(item['file_path'])
        if img is None: continue
        h, w = img.shape[:2]
        
        # Construir estructura unificada "Prediction"
        results = []
        for poly_obj in sam_polygons:
            poly_points = poly_obj['polygon']
            
            # Label Studio requiere que las coordenadas vengan en porcentaje (0-100) del ancho y alto
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
                    "polygonlabels": ["Revisame"] # Clase genérica predefinida para que el editor cambie su color
                },
                "id": region_id,
                "from_name": "label",
                "to_name": "image",
                "type": "polygonlabels",
                "score": poly_obj.get('predicted_iou', 0.9)
            })
            
        # Añadir al formato general de LS: Una Tarea incluye data.image y su array de predictions
        # Construimos la URL para Label Studio Local Storage:
        #   /data/local-files/?d=images/google_maps_web/<ROI>/<archivo>
        raw_path = str(Path(item['file_path']).resolve()).replace("\\", "/")
        # Extraer la parte relativa a partir de "google_maps_web/"
        marker = "google_maps_web/"
        idx = raw_path.find(marker)
        if idx != -1:
            relative_part = raw_path[idx + len(marker):]
        else:
            relative_part = raw_path.split("/")[-1]  # fallback: solo el nombre del archivo
        ls_image_url = f"/data/local-files/?d=images/google_maps_web/{relative_part}"
        
        ls_tasks.append({
            "data": {
                "image": ls_image_url
            },
            "predictions": [{
                "model_version": "SAM_Auto",
                "result": results
            }]
        })
        
    out_json = EXPORTS_DIR / "import_to_labelstudio.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(ls_tasks, f, indent=2, ensure_ascii=False)
        
    print(f"Éxito! Importa el archivo {out_json.resolve()} en tu proyecto de Label Studio.")
    print("Recuerda que Label Studio debe tener acceso de lectura (Local Storage) a donde están tus imágenes.")

if __name__ == "__main__":
    generate_ls_json()
