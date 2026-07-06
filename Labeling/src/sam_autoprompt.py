import os
import json
import numpy as np
import cv2
from pathlib import Path

# Placeholder / Try-catch para importar dependencias pesadas de IA
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

# Configuración del modelo SAM (Debe ser descargado manualmente: sam_vit_h_4b8939.pth)
SAM_CHECKPOINT = BASE_DIR / "scripts" / "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

def load_sam_pipeline():
    if not HAS_SAM:
        print("[ADVERTENCIA] No se ha detectado 'segment_anything' o 'torch'.")
        print("Para ejeuctar SAM en real usa: pip install torch torchvision opencv-python git+https://github.com/facebookresearch/segment-anything.git")
        return None

    if not SAM_CHECKPOINT.exists():
        print(f"[ADVERTENCIA] Falta el peso del modelo en {SAM_CHECKPOINT}.")
        print("Descárgalo con: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Fallback para Apple Silicon (M1/M2/M3)
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        device = "mps"
        
    print(f"Cargando Modelo SAM en dispositivo: {device.upper()}...")
    
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device=device)

    # El generador automático hace grillas predeterminadas de puntos para cubrir todo
    # Ajustamos params para no sacar polígonos microscópicos (min_mask_region_area)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Evitar ruido en suelo/sombras pequeñas
    )
    return mask_generator

def process_masks_to_polygons(sam_output):
    """
    Convierte la matriz booleana de SAM en coordenadas XY de contorno suavizado.
    """
    polygons_data = []
    
    # Ordenar por tamaño de área descendente. Útil en Label Studio para poner
    # 'Stuff' de fondo primero, y 'Things' apiladas encima
    sorted_masks = sorted(sam_output, key=(lambda x: x['area']), reverse=True)
    
    for mask_dict in sorted_masks:
        bool_mask = mask_dict['segmentation']
        
        # Convertir a imagen binaria para OpenCV
        binary_img = (bool_mask * 255).astype(np.uint8)
        
        # Extraer contornos externos
        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        
        if not contours:
            continue
            
        # Tomamos el contorno de mayor longitud para cada máscara
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Opcional: Suavizar o simplificar un poco el polígono para que JSON no pese 20MB ni congele UI
        epsilon = 0.002 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) < 3: # Un polígono requiere 3 vértices mínimo
            continue
            
        # Aplanar la estructura lista de tuplas a [[x,y], [x,y]]
        flat_poly = approx.reshape(-1, 2).tolist()
        
        polygons_data.append({
            "area": float(mask_dict['area']),
            "bbox": mask_dict['bbox'], # [x, y, w, h]
            "predicted_iou": float(mask_dict['predicted_iou']),
            "polygon": flat_poly
        })
        
    return polygons_data

def run_auto_prompt():
    print("----- Iniciando SAM Auto-Prompter -----")
    if not INDEX_PATH.exists():
        print("No hay index. Corre src/discovery.py primero.")
        return

    with open(INDEX_PATH, 'r') as f:
        dataset = json.load(f)

    generator = load_sam_pipeline()
    if not generator:
        print("[MOCK] Ejecución en Modo Seco. Sin IA, guardaría un polígono ficticio.")
        # MODO MOCK PARA QUE EL FLUJO NO ROMPA SI NO TIENE GPU
        for item in dataset:
            out_path = MASK_DIR / f"{item['id']}_sam.json"
            if item['processed_sam']: continue
            
            mock_data = [{
                "area": 2500, "bbox": [10, 10, 50, 50], "predicted_iou": 0.99,
                "polygon": [[10,10], [60,10], [60,60], [10,60]]
            }]
            with open(out_path, 'w') as f: json.dump(mock_data, f)
        return

    # MODO REAL CON INFERENCIA DE IA
    print("Comenzando inferencia...")
    for idx, item in enumerate(dataset):
        if item['processed_sam']:
            continue
            
        image_path = item['file_path']
        print(f"[{idx+1}/{len(dataset)}] Extracting {image_path}...")
        
        image = cv2.imread(image_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        masks = generator.generate(image)
        polygons = process_masks_to_polygons(masks)
        
        # Guarda diccionario local intermedio
        out_path = MASK_DIR / f"{item['id']}_sam.json"
        with open(out_path, 'w') as f:
            json.dump(polygons, f)
            
        item['processed_sam'] = True
        
    # Actualizar index general validando procesos
    with open(INDEX_PATH, 'w') as f:
         json.dump(dataset, f, indent=4)
         
    print("\nProcesamiento SAM Terminado!")

if __name__ == "__main__":
    run_auto_prompt()
