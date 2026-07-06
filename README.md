# TFE — Segmentación Panóptica de Imágenes Satelitales

Trabajo Fin de Estudios sobre segmentación panóptica aplicada a imágenes satelitales de instalaciones militares, usando técnicas de visión por computador y modelos de IA generativa.

---

## Arquitectura general

El flujo de trabajo se organiza en tres pipelines encadenados:

```
Google Maps Web          ─►  src/dataset/dataset_V1   (tiles 512×512 px)
     │
     ▼
[Pipeline 1 — GRDINO]   ─►  GroundingDINO + SAM → anotaciones Label Studio
[Pipeline 2 — Labeling]  ─►  SAM ML Backend (Docker) + revisión manual
[Gemini Labeling]        ─►  Segmentación asistida por Gemini 1.5 Pro
     │
     ▼
Dataset_Creator          ─►  Exporta a formato COCO panóptico
     │
     ▼
[Pipeline 3 — Analizer]  ─►  Entrenamiento y evaluación (Mask2Former, DETR, PSM-DIQ)
```

---

## Directorios

### `src/`
Dataset principal en formato **COCO panóptico**.

- `src/dataset/dataset_V1/` — versión actual del dataset: splits train/val/test con imágenes JPEG y máscaras PNG + JSONs de anotación COCO.

### `Google_Maps_Web/`
Aplicación web (Vite + Express) para capturar tiles de **Google Maps Satellite** y guardarlos como imágenes 512×512 px.

- `backend/` — servidor Express que hace proxy de tiles de Google Maps
- `frontend/` — UI para seleccionar zona, zoom y guardar tiles

### `GRDINO/`
Pipeline **zero-shot** de segmentación: GroundingDINO para detección de bounding boxes → Outlier Rejection → SAM para segmentación a nivel píxel → exportación a Label Studio.

```bash
cd GRDINO
python src/pipeline_gdino_sam.py                          # modo producción
python src/pipeline_gdino_sam.py --calibrate              # calibración de umbrales
python src/pipeline_gdino_sam.py --auto-calibrate         # optimización automática
python src/pipeline_gdino_sam.py --evaluate --gt gt.json  # evaluación PQ/SQ/RQ
```

Requiere los pesos de GroundingDINO (`weights/groundingdino_swint_ogc.pth`, ~662 MB, no incluidos) y SAM ViT-H (`Labeling/scripts/sam_vit_h_4b8939.pth`, ~2.4 GB, no incluidos).

### `label-studio-ml/`
Backend de ML para **Label Studio** basado en **SAM ViT-H**, desplegado con Docker Compose.

```bash
cd label-studio-ml
cp .env.example .env      # rellenar LABEL_STUDIO_ACCESS_TOKEN
docker compose up --build # primera vez (~5-10 min)
docker compose up         # siguientes veces
```

El backend queda disponible en `http://localhost:9090`. Ver [label-studio-ml/README.md](label-studio-ml/README.md) para instrucciones completas de conexión.

### `Labeling/`
Scripts de apoyo al proceso de etiquetado manual en Label Studio.

- `src/` — pre-procesado, exportación y conversión de anotaciones (COCO, Label Studio JSON)
- `config/` — configuración de proyectos y clases
- `scripts/` — utilidades adicionales (autoprompt SAM, conversión de formatos)

### `gemini_labeling/`
Pipeline de segmentación asistido por **Gemini 1.5 Pro** (API de Google).

```bash
cd gemini_labeling
python gemini_segmentation.py        # inferencia local
python gemini_segmentation_cloud.py  # inferencia en Cloud Run
```

- `outputs/masks/GOLOSO_18/` — ejemplo de máscaras generadas (3 imágenes de muestra)
- `outputs/Gemini_to_label_studio_V2.json` — exportación a formato Label Studio

### `Dataset_Creator/`
Convierte exportaciones de Label Studio a dataset **COCO panóptico** con split train/val/test.

```bash
cd Dataset_Creator
python create_dataset.py --version 1 --split random      # split aleatorio 80/10/10
python create_dataset.py --version 1 --split location    # split por localización (editar split_locations.yaml)
```

### `IMGscale/`
Mejora de resolución de tiles usando **Real-ESRGAN** (GPU local) o **GPT-image-2** (OpenAI API).

```bash
cd IMGscale
python enhance_images.py --mode esrgan   # local, requiere GPU
python enhance_images.py --mode openai   # vía API (editar prompt_openai.txt)
```

Los pesos de RealESRGAN (`weights/RealESRGAN_x4plus.pth`, ~64 MB) no están incluidos; se descargan automáticamente al primer uso.

### `analizer/`
Análisis comparativo de modelos de segmentación panóptica: **Mask2Former**, **DETR-Panoptic** y **PSM-DIQ**.

- `comparative_analysis.ipynb` — notebook de evaluación y métricas (PQ, SQ, RQ, IoU)
- `config.yaml` — parámetros de evaluación
- `papers/` — artículos de referencia

Los checkpoints de los modelos (~164 MB – 2.4 GB) no están incluidos en el repo.

### `Viewer/`
Visor HTML estático para inspeccionar imágenes y resultados de segmentación.

```bash
# Abrir directamente en el navegador
start Viewer/index.html
```

---

## Requisitos del entorno

### Python (entorno virtual)

```bash
# Crear y activar entorno
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### Node.js (Google_Maps_Web)

```bash
cd Google_Maps_Web/backend && npm install
cd Google_Maps_Web/frontend && npm install
npm run dev   # servidor de desarrollo (Vite)
```

### Docker (label-studio-ml)
Requiere **Docker Desktop** con WSL2 y **NVIDIA Container Toolkit** para GPU.

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

## Pesos de modelos (no incluidos en el repo)

| Modelo | Fichero | Tamaño | Dónde descargarlo |
|--------|---------|--------|-------------------|
| SAM ViT-H | `Labeling/scripts/sam_vit_h_4b8939.pth` | 2.4 GB | [Meta AI](https://github.com/facebookresearch/segment-anything#model-checkpoints) |
| GroundingDINO SwIN-T | `GRDINO/weights/groundingdino_swint_ogc.pth` | 662 MB | [IDEA-Research](https://github.com/IDEA-Research/GroundingDINO) |
| RealESRGAN x4+ | `IMGscale/weights/RealESRGAN_x4plus.pth` | 64 MB | Descarga automática |
| Mask2Former | `analizer/checkpoints/mask2former_best.pth` | 181 MB | Entrenamiento propio |
| DETR-Panoptic | `analizer/checkpoints/detr-panoptic_best.pth` | 164 MB | Entrenamiento propio |

---

## Clases del dataset

El dataset incluye clases de segmentación de instalaciones militares:
edificios, vehículos, hangares, depósitos, zonas pavimentadas, vegetación y fondo.
La configuración completa de clases se encuentra en `GRDINO/config/classes.yaml`.

---

## Autor

Javier Martínez Becedas — [javiermartinezbecedas@gmail.com](mailto:javiermartinezbecedas@gmail.com)
