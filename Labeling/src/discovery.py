import os
import json
import glob
from pathlib import Path

# Configuración de Rutas Relativas (El pipeline se ejecutará desde TFM/Labeling)
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_PATH = BASE_DIR.parent / "src" / "images" / "google_maps_web"
INDEX_PATH = BASE_DIR / "data" / "dataset_index.json"

def discover_dataset():
    """Explora recursivamente las ROIs y cataloga todos los tiles"""
    if not IMAGES_PATH.exists():
        print(f"[ERROR] No se encontró el directorio de imágenes: {IMAGES_PATH}")
        return

    dataset = []
    
    # Cada subcarpeta en google_maps_web se considera una ROI
    roi_folders = [f for f in IMAGES_PATH.iterdir() if f.is_dir()]
    
    for roi in roi_folders:
        print(f"Descubriendo ROI: {roi.name}...")
        
        # Archivos principales y mosaicos (.jpeg, .tiff)
        extensions = ('*.jpeg', '*.jpg', '*.tiff', '*.tif', '*.png')
        images = []
        for ext in extensions:
            images.extend(roi.rglob(ext))
        
        for img_path in images:
            # Buscar metadata par si la hay
            meta_path = img_path.with_suffix('.json')
            has_meta = meta_path.exists()
            
            # Clasificar si es master o tile subgrid
            img_type = "main" if "_main" in img_path.name else "tile"
            
            record = {
                "id": f"{roi.name}_{img_path.stem}",
                "roi": roi.name,
                "type": img_type,
                "file_path": str(img_path.resolve()),
                "meta_path": str(meta_path.resolve()) if has_meta else None,
                "processed_sam": False,
                "label_studio_task_id": None
            }
            dataset.append(record)

    # Guardar indice centralizado
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, 'w') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"\n[Exito] ¡Descubrimiento completado! Se han indexado {len(dataset)} imágenes en total.")
    print(f"Archivo guardado en {INDEX_PATH.resolve()}")

if __name__ == "__main__":
    discover_dataset()
