cd ..# Pipeline de Panoptic Labeling

Orquestación automática para producir Dataset COCO Panoptic utilizando las máscaras calculadas por PyTorch vía Segment Anything (SAM) refindas gracias a la interfaz gráfica de Label Studio.

## Funciones
El proyecto implementa la lógica `Imagen -> Máscaras(SAM) -> Label Studio -> COCO Panoptic`.

## Estructura de Clases
La ontología semántica está dictada en `config/ontology.yaml`. Edita este fichero para separar entre cosas unitarias (`isthing: 1`, como _Carros_) y fluidos continuos (`isthing: 0`, como _Vegetación_). 

## Flujo de Trabajo

### 1. Descubrimiento Básico
Examina tu base de imágenes del `Google_Maps_Web`.
```bash
python3 src/discovery.py
```
> Resultado: Devuelve `data/dataset_index.json`.

### 2. Autosegmentador Universal (SAM)
Lee la arquitectura generada y para cada imagen crea siluetas vectorizadas de alta calidad.
```bash
# Recuerda instalar 'segment_anything' de Facebook y dejar el archivo .pth en /scripts
python3 src/sam_autoprompt.py
```
> Resultado: Genera perfiles y los acopia en la carpeta interna `/data/masks/`.

### 3. Puente a Label Studio
Traduce la geometría generada matemáticamente al Canvas gráfico % de tu Label Studio personal.
```bash
python3 src/to_labelstudio.py
```
# Versión monolítica optimizada
python src/to_labelstudio.py

# Versión dividida por ROI
python src/to_labelstudio_split.py

> Resultado: Se obtiene el documento unificado `/exports/import_to_labelstudio.json`. **Ve a tu aplicación Label Studio, crea un proyecto, y arrástralo en la opción de Import**. *Nota: Label Studio debe poder leer tus rutas `/src/images` localmente.*
### ruta Label Studio "/data/local-files/?d=images/google_maps_web/

### 4. Empaquetador COCO Panoptic
Una vez que en Label Studio el equipo designe el material como "Anotado" validando y pintando por los contornos, dale botón a **Export** eligiendo el formato `JSON` y pégalo devolviéndolo a esta carpeta bajo `/exports/project_export.json`.
Corrige el JSON final y conviértelo al formato Machine Learning validado:
```bash
python3 src/export_coco_panoptic.py
```
> Resultado: Finaliza creando `/exports/coco_panoptic/` poblado con **Imágenes Maestras**, las legendarias **Máscaras RGB** y el árbol formal JSON **panoptic_train.json**.
