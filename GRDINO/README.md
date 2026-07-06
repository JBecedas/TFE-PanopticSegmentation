# GRDINO — Pipeline GroundingDINO → Outlier Rejection → SAM → Label Studio

## Llamadas CLI — Referencia rápida

```bash
# ── Modo producción: GroundingDINO + SAM + JSON para Label Studio ──────────
python pipeline_gdino_sam.py

# ── BBoxes manuales de LS → SAM → import_to_labelstudio.json ──────────────
#    (sin GroundingDINO: sólo segmenta lo que tú has anotado)
python pipeline_gdino_sam.py --boxes-json mi_export_label_studio.json

# ── Evaluación PQ/SQ/RQ/IoU (standalone, tras ejecutar el pipeline) ────────
python pipeline_gdino_sam.py --evaluate \
    --gt mi_gt_verificado.json \
    --pred outputs/import_to_labelstudio.json   # --pred es opcional

# ── Evaluación automática integrada en cualquier modo ──────────────────────
python pipeline_gdino_sam.py --gt mi_gt_verificado.json
python pipeline_gdino_sam.py --boxes-json anotaciones.json --gt mi_gt.json
python pipeline_gdino_sam.py --from-annotations anotaciones.json --gt mi_gt.json

# ── Calibración manual (grid de umbrales, sin SAM) ─────────────────────────
python pipeline_gdino_sam.py --calibrate
python pipeline_gdino_sam.py --calibrate --class military_vehicles \
    --images /path/to/test_images --n-samples 10

# ── Auto-calibración (optimización automática + actualiza classes.yaml) ─────
python pipeline_gdino_sam.py --auto-calibrate
python pipeline_gdino_sam.py --auto-calibrate --class buildings \
    --images /path/to/test_images --n-samples 15

# ── Solo SAM desde anotaciones manuales (salida separada) ──────────────────
python pipeline_gdino_sam.py --from-annotations export.json
```

---

## Índice de funciones — `pipeline_gdino_sam.py`

| Función | Sección | Descripción breve |
|---------|---------|-------------------|
| `load_config()` | CONFIG | Lee `classes.yaml` y muestra el resumen de passes |
| `_find_gdino_config()` | MODELS | Localiza el fichero de configuración de GroundingDINO dentro del paquete |
| `load_models()` | MODELS | Carga GroundingDINO + SAM |
| `load_gdino_only()` | MODELS | Carga solo GroundingDINO (modo calibración) |
| `load_sam_only()` | MODELS | Carga solo SAM (modo `--from-annotations`) |
| `iou_nms()` | IoU-NMS | Non-Maximum Suppression por IoU sobre tensores |
| `run_gdino_per_prompt()` | GDINO | Ejecuta una llamada GDINO por prompt y aplica NMS intra-pass |
| `reject_outliers()` | OUTLIER | Filtra cajas por área relativa, aspect ratio y confianza |
| `calibrate_classes()` | CALIBRATION | Informe estadístico: por-prompt + grid box×text |
| `_load_sample_images()` | AUTO-CALIB | Carga tensores de muestra una sola vez |
| `_eval_config()` | AUTO-CALIB | Evalúa una combinación (box_th, text_th, nms_th) |
| `_grid_search()` | AUTO-CALIB | Grid search box×text maximizando confianza media |
| `_search_nms_threshold()` | AUTO-CALIB | Barrido de nms_threshold con box/text fijados |
| `_fit_outlier_params()` | AUTO-CALIB | Ajusta bounds de outlier_rejection por percentiles |
| `_patch_yaml_value()` | AUTO-CALIB | Reemplaza valor numérico en YAML preservando comentarios |
| `_update_pass_in_yaml()` | AUTO-CALIB | Aplica dict de parámetros al bloque del pass en el YAML |
| `auto_calibrate_classes()` | AUTO-CALIB | Orquesta los 4 stages y actualiza `classes.yaml` |
| `_rotated_bbox_to_aabb()` | UTILITIES | Convierte bbox rotado de Label Studio a AABB para SAM |
| `mask_to_polygon()` | UTILITIES | Convierte máscara booleana a polígono (contorno más grande) |
| `_class_color()` | UTILITIES | Devuelve color BGR para una clase (con fallback) |
| `_get_stuff_classes()` | UTILITIES | Devuelve el set de clases marcadas `is_stuff: true` |
| `draw_results()` | UTILITIES | Dibuja máscaras y (para thing-classes) bounding boxes en imagen |
| `build_ls_result()` | UTILITIES | Construye una anotación PolygonLabel para Label Studio |
| `build_image_url()` | UTILITIES | Genera la URL de imagen en formato Label Studio |
| `_ls_url_to_path()` | UTILITIES | Resuelve una URL de Label Studio a ruta de disco |
| `run_sam_from_annotations()` | SAM-FROM-ANNOT | Pipeline inverso: BBoxes manuales → SAM → JSON LS |
| `_polygon_to_mask()` | EVAL | Rasteriza polígono LS (% coords) a máscara binaria |
| `_mask_iou()` | EVAL | IoU píxel a píxel entre dos máscaras binarias |
| `_greedy_match()` | EVAL | Emparejamiento greedy pred↔GT por IoU descendente |
| `_pq_from_parts()` | EVAL | Calcula PQ, SQ, RQ a partir de TP/FP/FN |
| `_load_ls_polygons()` | EVAL | Carga máscaras de un JSON de LS (export o pipeline) |
| `_write_eval_log_md()` | EVAL | Escribe `eval_log.md` con métricas por imagen y clase |
| `evaluate_predictions()` | EVAL | Función principal de evaluación PQ/SQ/RQ/IoU |
| `_write_log_md()` | LOG | Escribe `log.md` con estadísticas detalladas por ROI |
| `run_pipeline()` | MAIN PIPELINE | Pipeline completo: tiles → GDINO → outlier → SAM → LS |
| `_parse_args()` | CLI | Define y parsea los argumentos de línea de comandos |

---

## Visión general

```
┌─────────────┐     ┌────────────────────┐     ┌─────┐     ┌──────────────┐
│ Tiles .jpg  │ ──▸ │   GroundingDINO    │ ──▸ │ SAM │ ──▸ │ Label Studio │
│ por ROI     │     │  (detección + NLP) │     │     │     │    JSON      │
└─────────────┘     └────────────────────┘     └─────┘     └──────────────┘
                              │                   ▲
                              ▼                   │
                    ┌─────────────────────┐       │
                    │  Outlier Rejection  │───────┘
                    │  (filtrado cajas)   │
                    └─────────────────────┘
```

1. **GroundingDINO** recibe cada tile y un prompt de texto. Devuelve bounding boxes + logits + frases.
2. **Outlier Rejection** filtra detecciones anómalas antes de enviarlas a SAM.
3. **SAM** recibe las cajas filtradas como *box prompts* y genera máscaras de segmentación.
4. Se generan imágenes anotadas y un JSON importable a **Label Studio**.

---

## Clases: thing vs stuff

Las clases se dividen en dos categorías según el campo `is_stuff` de `classes.yaml`:

| Categoría | Descripción | Comportamiento |
|-----------|-------------|----------------|
| **thing** | Objetos contables con instancias individuales (vehículos, edificios…) | Bounding box + máscara en visualización; PolygonLabel en JSON |
| **stuff** | Regiones amorfas sin instancias discretas (carreteras, vegetación…) | Solo máscara en visualización (sin bbox ni etiqueta); PolygonLabel en JSON |

Para declarar una clase como stuff en `classes.yaml`:

```yaml
- name: roads_and_tracks
  is_stuff: true
  text_prompts:
    - "road."
  ...
```

### Clases actuales

| Clase | Tipo | Color |
|-------|------|-------|
| `military_vehicles` | thing | `#E74C3C` |
| `buildings` | thing | `#422f2f` |
| `fuel_infrastructure` | thing | `#F1C40F` |
| `perimeter_structures` | thing | `#3498DB` |
| `communication_and_radar` | thing | `#E91E63` |
| `roads_and_tracks` | **stuff** | `#7F8C8D` |
| `forest` | **stuff** | `#209728` |

---

## Modos de operación

### 1. Modo producción (predeterminado)

```bash
python pipeline_gdino_sam.py
```

Pipeline completo por cada tile en `TILES_ROOT/<ROI>/`:

1. Para cada pass en `classes.yaml`: una llamada GDINO por prompt → NMS intra-pass → Outlier Rejection
2. Merge de todos los passes
3. SAM con cada bounding box como prompt
4. Visualización en `outputs/visuals/<ROI>/`
5. JSON importable en `outputs/import_to_labelstudio.json`
6. Log estadístico en `outputs/log.md`

### 2. Modo calibración manual (`--calibrate`)

```bash
python pipeline_gdino_sam.py --calibrate [--class CLASS] [--images DIR] [--n-samples N]
```

Sin SAM. Genera un informe Markdown en `outputs/calibration/` con:

- **Sección 1**: estadísticas por prompt (detecciones, score medio, área media) con los umbrales actuales
- **Sección 2**: grid `box_threshold × text_threshold` — total de detecciones por combinación

Útil para decidir qué prompts mantener y qué umbrales elegir.

**Opciones:**

| Opción | Default | Descripción |
|--------|---------|-------------|
| `--class CLASS` | todas | Calibra solo ese pass |
| `--images DIR` | `TILES_ROOT` | Directorio con imágenes de prueba |
| `--n-samples N` | `5` | Número máximo de imágenes a muestrear |
| `--output-dir DIR` | `outputs/calibration/` | Dónde guardar el informe |

### 3. Auto-calibración (`--auto-calibrate`)

```bash
python pipeline_gdino_sam.py --auto-calibrate [--class CLASS] [--images DIR] [--n-samples N]
```

Sin SAM. Optimiza automáticamente los parámetros de cada pass y **actualiza `classes.yaml`** (se crea backup `.bak` primero).

**Algoritmo por pass (4 stages):**

1. **Grid coarse** — 10×7 = 70 combinaciones `box_th × text_th`
2. **Grid fino** — ±0.04 alrededor del óptimo coarse (paso 0.02)
3. **Barrido NMS** — 10 valores con `box/text` óptimos fijados
4. **Ajuste outlier** — percentiles p2/p98 sobre la distribución real de detecciones

Objetivo: maximizar la confianza media de detección (favorece precisión sobre recall).

### 4. BBoxes manuales → SAM (salida de pipeline, `--boxes-json`)

```bash
python pipeline_gdino_sam.py --boxes-json <export_label_studio.json>
```

**Flujo de trabajo recomendado:**

1. Anotar bounding boxes en Label Studio (proyecto RectangleLabels, mismos nombres de clase que `classes.yaml`)
2. Exportar → formato JSON nativo de Label Studio
3. Ejecutar con `--boxes-json`
4. Importar el JSON resultante en Label Studio para revisar / corregir las máscaras

**Diferencia clave respecto a `--from-annotations`:**

| | `--boxes-json` | `--from-annotations` |
|-|----------------|----------------------|
| Salida JSON | `import_to_labelstudio.json` (misma que pipeline completo) | `import_sam_from_annotations.json` |
| Salida visual | `outputs/visuals/from_boxes/` | `outputs/visuals/from_annotations/` |
| Uso típico | Reemplazar el pipeline completo con anotaciones propias | Exploración / prueba rápida de SAM |

GroundingDINO **no se carga** en este modo, así que arranca mucho más rápido y solo genera máscaras para los objetos que el usuario ha anotado explícitamente.

**Soporte de rotación:** Label Studio permite crear bounding boxes rotadas. El pipeline las convierte automáticamente al AABB mínimo que las contiene (SAM solo acepta prompts xyxy). La calidad de la máscara no se ve afectada.

**Comportamiento según tipo de clase:**

- **thing classes**: máscara + rectángulo en el visual; PolygonLabel en el JSON
- **stuff classes** (`is_stuff: true`): solo máscara (sin rectángulo ni etiqueta de texto); PolygonLabel en el JSON igualmente

### 5. Evaluación PQ (`--evaluate` / `--gt`)

```bash
# Standalone (tras ejecutar el pipeline):
python pipeline_gdino_sam.py --evaluate \
    --gt gt_verificado.json \
    [--pred outputs/import_to_labelstudio.json]

# Integrado: evalúa automáticamente al final de cualquier modo:
python pipeline_gdino_sam.py --gt gt_verificado.json
python pipeline_gdino_sam.py --boxes-json anotaciones.json --gt gt_verificado.json
```

**Ground truth**: export JSON de Label Studio con PolygonLabels verificadas manualmente. El formato debe coincidir con el que produce el pipeline (mismo proyecto LS, mismas URLs de imagen).

**Métricas calculadas (por clase y global):**

| Métrica | Definición |
|---------|-----------|
| **IoU** | Intersección / Unión píxel a píxel entre máscara pred y GT (solo TPs) |
| **SQ** | Segmentation Quality = media de IoU de pares emparejados (TPs) |
| **RQ** | Recognition Quality = TP / (TP + 0.5·FP + 0.5·FN) |
| **PQ** | Panoptic Quality = SQ × RQ |

**Regla de emparejamiento**: para cada clase, emparejamiento greedy por IoU descendente; un par es TP solo si IoU ≥ 0.5. Las máscaras se rasterizán a 1024×1024 en coordenadas porcentuales para evitar desajustes de dimensión.

**Salidas:**
- Resumen por pantalla (clases + global)
- `outputs/eval_log.md` con tablas por imagen, por clase y global

### 6. SAM desde anotaciones manuales (`--from-annotations`)

```bash
python pipeline_gdino_sam.py --from-annotations <export.json>
```

Igual que `--boxes-json` pero con salidas en rutas separadas. Útil cuando se quieren tener los resultados de SAM-desde-anotaciones y los del pipeline completo en paralelo sin sobrescribir nada.

---

## Outlier Rejection — Cómo funciona

### ¿Qué problema resuelve?

GroundingDINO es un detector open-vocabulary que inevitablemente genera **falsos positivos**:

- Cajas diminutas de pocos píxeles (ruido)
- Cajas gigantes que cubren casi toda la imagen
- Cajas con aspect ratios absurdos (líneas de 1px de alto)
- Detecciones con confianza muy baja

El módulo **Outlier Rejection** actúa como filtro de calidad entre GroundingDINO y SAM.

### Criterios de rechazo

Cada bounding box pasa por **3 filtros secuenciales**:

#### 1. Filtro de Área Relativa

```
ratio = (ancho_caja × alto_caja) / (ancho_imagen × alto_imagen)
```

| Parámetro        | Default  | Rechaza si...                               |
|------------------|----------|----------------------------------------------|
| `min_area_ratio` | `0.0005` | `ratio < 0.0005` → caja < 0.05% de la imagen |
| `max_area_ratio` | `0.5`    | `ratio > 0.5`   → caja > 50% de la imagen   |

#### 2. Filtro de Aspect Ratio

```
aspect = ancho_caja / alto_caja
```

| Parámetro    | Default | Rechaza si...                           |
|--------------|---------|------------------------------------------|
| `min_aspect` | `0.1`   | `aspect < 0.1` → 10× más alto que ancho |
| `max_aspect` | `10.0`  | `aspect > 10`  → 10× más ancho que alto |

#### 3. Filtro de Confianza

| Parámetro        | Default | Rechaza si...                    |
|------------------|---------|----------------------------------|
| `min_confidence` | `0.30`  | `logit < 0.30` → baja confianza |

### Flujo de decisión

```
Para cada caja detectada por GroundingDINO:
  │
  ├── ¿Área relativa < min_area_ratio?  ──▸ RECHAZADA
  ├── ¿Área relativa > max_area_ratio?  ──▸ RECHAZADA
  ├── ¿Aspect ratio < min_aspect?       ──▸ RECHAZADA
  ├── ¿Aspect ratio > max_aspect?       ──▸ RECHAZADA
  ├── ¿Confianza < min_confidence?      ──▸ RECHAZADA
  │
  └── Pasa todos los filtros            ──▸ ACEPTADA → enviada a SAM
```

### Activar / Desactivar

```yaml
outlier_rejection:
  enabled: true        # false para desactivar (todas las cajas pasan a SAM)
  min_area_ratio: 0.0005
  max_area_ratio: 0.5
  min_aspect: 0.1
  max_aspect: 10.0
  min_confidence: 0.30
```

---

## Estructura del directorio

```
GRDINO/
├── config/
│   └── classes.yaml              # Clases, umbrales y config de outlier rejection
├── outputs/
│   ├── visuals/                  # Imágenes anotadas con máscaras (y bbox para thing-classes)
│   │   └── from_annotations/    # Resultados del modo --from-annotations
│   ├── calibration/             # Informes de calibración (.md)
│   ├── import_to_labelstudio.json
│   └── log.md
├── src/
│   └── pipeline_gdino_sam.py    # Pipeline principal
├── weights/
│   └── groundingdino_swint_ogc.pth
└── README.md
```

## Ejecución

```bash
cd GRDINO/src
python pipeline_gdino_sam.py
```

Los tiles se leen de `c:/TFM/src/images/google_maps_web/<ROI_NAME>/` y los resultados se guardan en `GRDINO/outputs/`.

---

## `classes.yaml` — Formato completo

```yaml
passes:
  - name: military_vehicles       # identificador único del pass
    is_stuff: false               # (opcional, default false) true = stuff class
    text_prompts:
      - "military tank."          # cada entrada = una llamada GDINO independiente
      - "armored vehicle."

    box_threshold:  0.30          # umbral de confianza de detección (0–1)
    text_threshold: 0.04          # umbral de coincidencia texto-imagen (0–1)
    nms_threshold:  0.35          # IoU máximo tolerado entre cajas del mismo pass

    outlier_rejection:
      enabled: true
      min_area_ratio: 0.005       # fracción mínima del área de la imagen
      max_area_ratio: 0.88        # fracción máxima
      min_aspect:     0.39        # ancho/alto mínimo
      max_aspect:     1.92        # ancho/alto máximo
      min_confidence: 0.30        # confianza mínima post-detección
```
