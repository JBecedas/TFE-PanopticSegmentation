import os
import json
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS LOCALES ---
BASE_DIR = Path(__file__).resolve().parent.parent
# Ruta donde están tus imágenes físicamente en tu PC
IMAGES_PATH = BASE_DIR.parent / "src" / "images" / "google_maps_web"
INDEX_PATH = BASE_DIR / "data" / "dataset_index.json"

# --- CONFIGURACIÓN DE RUTA DESTINO (Colab/Drive) ---
COLAB_DRIVE_ROOT = "/content/drive/MyDrive/Colab_Notebooks/TFM/src/google_maps_web"

def discover_dataset():
    """Explora localmente respetando subcarpetas y mapea a rutas de Colab"""
    
    if not IMAGES_PATH.exists():
        print(f"[ERROR] No se encontró el directorio local: {IMAGES_PATH}")
        return

    dataset = []
    
    # Extensiones de imagen soportadas
    extensions = ('*.jpeg', '*.jpg', '*.tiff', '*.tif', '*.png')
    
    print(f"Escaneando archivos en: {IMAGES_PATH}")

    # Buscamos todos los archivos de imagen de forma recursiva
    all_images = []
    for ext in extensions:
        all_images.extend(IMAGES_PATH.rglob(ext))

    for img_path in all_images:
        # --- CAMBIO CLAVE: CÁLCULO DE RUTA RELATIVA ---
        # Esto extrae la parte de la ruta desde 'google_maps_web' hacia abajo
        # Ejemplo local: .../google_maps_web/Morocco_1/1_1/imagen.jpg
        # relative_path -> Morocco_1/1_1/imagen.jpg
        relative_path = img_path.relative_to(IMAGES_PATH)
        
        # Metadata (verificamos localmente si existe el .json par)
        meta_path_local = img_path.with_suffix('.json')
        has_meta = meta_path_local.exists()
        
        # Clasificación (main vs tile)
        img_type = "main" if "_main" in img_path.name else "tile"
        
        # --- CONSTRUCCIÓN DE RUTA REMOTA (Linux Style) ---
        # Unimos la raíz de Colab con la ruta relativa calculada
        # .as_posix() asegura que use siempre '/' incluso si estás en Windows
        remote_file_path = (Path(COLAB_DRIVE_ROOT) / relative_path).as_posix()
        
        if has_meta:
            relative_meta_path = meta_path_local.relative_to(IMAGES_PATH)
            remote_meta_path = (Path(COLAB_DRIVE_ROOT) / relative_meta_path).as_posix()
        else:
            remote_meta_path = None

        record = {
            "id": f"{img_path.stem}", # ID basado en el nombre del archivo
            "roi": img_path.parent.name, # Nombre de la carpeta inmediata
            "type": img_type,
            "file_path": remote_file_path,
            "meta_path": remote_meta_path,
            "processed_sam": False,
            "label_studio_task_id": None
        }
        dataset.append(record)

    # Guardar índice localmente
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
        
    print(f"\n[Éxito] Descubrimiento completado.")
    print(f"-> {len(dataset)} imágenes indexadas.")
    print(f"-> Estructura de carpetas preservada para Colab.")
    if dataset:
        print(f"-> Ejemplo de ruta generada: {dataset[0]['file_path']}")

if __name__ == "__main__":
    discover_dataset()