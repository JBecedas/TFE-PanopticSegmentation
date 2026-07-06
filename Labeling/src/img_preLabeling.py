import os
import json
from pathlib import Path

def generate_prelabeling_json():
    """
    Generates a Label Studio import JSON from a directory of images.
    Ignores files that contain 'main' in their names.
    """
    # Rutas relativas a la ubicación de este script
    script_dir = Path(__file__).resolve().parent
    labeling_dir = script_dir.parent
    root_dir = labeling_dir.parent  # Directorio raíz (TFM)
    
    images_dir = root_dir /"src" / "images" / "google_maps_web"
    output_file = labeling_dir / "exports" / "preLabeling.json"

    base_url = "/data/local-files/?d=images/google_maps_web/"
    tasks = []
    
    if not images_dir.exists() or not images_dir.is_dir():
        print(f"Error: El directorio {images_dir} no existe o no es un directorio.")
        return

    # Iterate over all files in the directory recursively
    for file_path in images_dir.rglob('*'):
        if file_path.is_file():
            # Skip files that contain "main" in the name
            if "main" in file_path.name.lower():
                continue
                
            # Filter by common image extensions
            if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']:
                continue

            # Calculate relative path to the images_dir
            try:
                rel_path = file_path.relative_to(images_dir)
            except ValueError:
                continue

            # Convert to POSIX path format for the URL (e.g., FOLDER/image.jpg)
            posix_rel_path = rel_path.as_posix()
            
            image_url = f"{base_url}{posix_rel_path}"
            
            task = {
                "data": {
                    "image": image_url
                },
                "predictions": []
            }
            tasks.append(task)
            
    # Write the JSON output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully generated {output_file} with {len(tasks)} images.")

if __name__ == "__main__":
    generate_prelabeling_json()
