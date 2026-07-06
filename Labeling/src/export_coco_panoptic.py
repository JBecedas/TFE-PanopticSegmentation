import json
import yaml
import cv2
import numpy as np
import os
import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ONTOLOGY_PATH = BASE_DIR / "config" / "ontology.yaml"
# Asumimos que tú desde la Interfaz Gráfica de Label Studio le diste "Exportar -> JSON"
# y lo pegaste en la ruta Exports:
LS_EXPORT = BASE_DIR / "exports" / "project_export.json" 

COCO_OUTPUT_DIR = BASE_DIR / "exports" / "coco_panoptic"
COCO_IMG_DIR = COCO_OUTPUT_DIR / "images"
COCO_MASKS_DIR = COCO_OUTPUT_DIR / "panoptic_masks"

for d in [COCO_OUTPUT_DIR, COCO_IMG_DIR, COCO_MASKS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def load_ontology():
    with open(ONTOLOGY_PATH, 'r') as f:
        config = yaml.safe_load(f)
    categories = config.get("categories", [])
    # Mapa rápido de nombre a ID y atributos
    cat_map = {c["name"].lower(): c for c in categories}
    return categories, cat_map

def id_to_rgb(id_num):
    """
    Convierte un segment_id numerico (1, 2, 3...) a color RGB [R, G, B]
    Formula COCO Panoptic estándar: R + G*256 + B*256^2 = id
    """
    r = id_num % 256
    g = (id_num // 256) % 256
    b = (id_num // (256 * 256)) % 256
    return [r, g, b]

def rgb_to_id(r, g, b):
    return r + g * 256 + b * 256 * 256

def construct_panoptic():
    print("----- Transformando LS Export a COCO Panoptic -----")
    if not LS_EXPORT.exists():
        print(f"Buscando el archivo {LS_EXPORT.resolve()} pero no existe.")
        print("Tuerce hacia tu UI de Label Studio -> Export -> JSON y guardalo ahí.")
        return

    categories, cat_map = load_ontology()
    
    with open(LS_EXPORT, 'r') as f:
        ls_data = json.load(f)

    # Base del JSON Arquitectura COCO
    panoptic_json = {
        "info": {
            "description": "Base Militar Panoptic Dataset",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "date_created": datetime.datetime.now().strftime("%Y/%m/%d")
        },
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": []
    }

    # Procesar imagen por imagen
    global_img_id = 0
    global_segment_id = 1
    
    for task in ls_data:
        global_img_id += 1
        
        # En el export de Label Studio, la ruta de la orginal suele venir en task['data']['image']
        # Si era un disco local de LS, te llegará estilo '/data/local-files/?d=/Users...jpg'
        # Haremos un parseo basico asumiendo que contiene la ruta
        raw_img_path = task['data'].get('image', '')
        if 'd=' in raw_img_path:
            clean_img_path = raw_img_path.split('d=')[1]
        else:
            clean_img_path = raw_img_path
            
        clean_img_path = Path(clean_img_path)
        img_name = clean_img_path.name
        
        # Leemos la dimension real
        # Si la ruta no existe por problemas de volumentes en LS, saltaremos fallback
        if clean_img_path.exists():
            img = cv2.imread(str(clean_img_path))
            h, w = img.shape[:2]
        else:
            print(f"[Warning] Imposible leer disco original en {clean_img_path}. Se usará dims hardcodeadas.")
            h, w = 512, 512 # Fallback
            
        panoptic_json["images"].append({
            "id": global_img_id,
            "width": w,
            "height": h,
            "file_name": img_name
        })
        
        # Preparación de la Máscara Panóptica (Array vacio RGB, default negro 0,0,0)
        panoptic_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        segments_info = []
        
        # Buscar "annotations" realizadas por el humano
        if not task.get('annotations'):
            continue
            
        annotation = task['annotations'][0] # Toma la primera validada
        results = annotation.get('result', [])
        
        # El algoritmo COCO Panoptic sugiere pintar primero el "Stuff" y luego el "Thing" 
        # porque Instance no debe ser arrollado por un macro-area
        stuff_results = []
        thing_results = []
        
        for res in results:
            if 'polygonlabels' not in res['value']: continue
            label = res['value']['polygonlabels'][0].lower()
            if label not in cat_map:
                continue
            if cat_map[label]['isthing'] == 1:
                thing_results.append(res)
            else:
                stuff_results.append(res)
                
        # 1. Dibujar STUFF (Si hay overlap, Thing sobrescribe)
        for res in stuff_results + thing_results:
            points_percent = res['value']['points']
            label = res['value']['polygonlabels'][0].lower()
            cat = cat_map[label]
            
            # Reconvertir porcentajes a pixels absolutos
            abs_points = []
            for pt in points_percent:
                x = int((pt[0] / 100.0) * w)
                y = int((pt[1] / 100.0) * h)
                abs_points.append([x, y])
            
            abs_points_arr = np.array([abs_points], dtype=np.int32)
            
            # Asignar un segment_id único por instancia en esta imagen
            seg_id = global_segment_id
            global_segment_id += 1
            
            color_rgb = id_to_rgb(seg_id)
            cv2.fillPoly(panoptic_mask, abs_points_arr, color_rgb)
            
            # Calcular Caja deliminadora (Bbox COCO es [x, y, ancho, alto])
            x_coords = [p[0] for p in abs_points]
            y_coords = [p[1] for p in abs_points]
            bbox = [min(x_coords), min(y_coords), max(x_coords)-min(x_coords), max(y_coords)-min(y_coords)]
            
            # Ojo: El área real la dictaría un algoritmo que quite lo superpuesto.
            # COCO tools utiliza librerías pycocotools. Aqui un area estimativa via cv2
            area = cv2.contourArea(abs_points_arr)
            
            segments_info.append({
                "id": seg_id, # Coincide con RGB en png
                "category_id": cat['id'],
                "iscrowd": 0,
                "bbox": bbox,
                "area": area
            })
            
        # Guardar en archivo PNG (Como OpenCV guarda en BGR, invertimos a RGB para asegurar
        # que R g y b calcen perfecto al hacer read del png despues per COCO logic)
        panoptic_mask_bgr = cv2.cvtColor(panoptic_mask, cv2.COLOR_RGB2BGR)
        png_mask_name = f"{img_name.split('.')[0]}_panoptic.png"
        cv2.imwrite(str(COCO_MASKS_DIR / png_mask_name), panoptic_mask_bgr)
        
        # Y Copiamos la imagen original a images/
        if clean_img_path.exists():
            dest_img = COCO_IMG_DIR / img_name
            cv2.imwrite(str(dest_img), img) # Copia simple en formato OpenCV
            
        # Push anotacion general
        panoptic_json["annotations"].append({
            "image_id": global_img_id,
            "file_name": png_mask_name,
            "segments_info": segments_info
        })
        
    final_json_path = COCO_OUTPUT_DIR / "panoptic_train.json"
    with open(final_json_path, 'w') as f:
        json.dump(panoptic_json, f, indent=4)
        
    print(f"[EXITO] COCO Panoptic exportado en {COCO_OUTPUT_DIR.resolve()}")
    print("Contiene la carpeta de imágenes, máscaras segmentadas unificadas RGB y su JSON COCO.")

if __name__ == "__main__":
    construct_panoptic()
