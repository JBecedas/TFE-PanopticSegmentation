# Dataset Creator — Documentación

## Qué hace el script

`create_dataset.py` convierte las anotaciones exportadas de Label Studio en un dataset de
**segmentación panóptica** en formato COCO, listo para entrenar modelos como Mask2Former.

---

## Estructura de salida

```
Dataset_V{n}/
├── train/
│   ├── images/          ← tiles JPEG copiados tal cual
│   ├── panoptic/        ← máscaras PNG con encoding COCO (para entrenar)
│   └── panoptic_viz/    ← máscaras PNG con colores semánticos (para inspección visual)
├── val/                 ← misma estructura
├── test/                ← misma estructura
├── panoptic_train.json  ← JSON COCO panoptic del split train
├── panoptic_val.json
├── panoptic_test.json
└── dataset_info.json    ← resumen de clases, splits y estadísticas
```

---

## Por qué las máscaras de `panoptic/` parecen casi negras

El encoding que usa `panoptic/` es el **estándar obligatorio de COCO panoptic**:

```
segment_id = R + G×256 + B×256²
```

Cada píxel **no lleva el color de la clase**, sino un **número entero único** que identifica
el segmento al que pertenece. Como los IDs suelen ser números pequeños (1, 2, 3…), los píxeles
resultan casi negros visualmente.

El JSON (`panoptic_train.json`, etc.) es el que conecta cada `segment_id` con su `category_id`:

```json
{
  "annotations": [
    {
      "image_id": 1,
      "file_name": "tile_panoptic.png",
      "segments_info": [
        { "id": 1,  "category_id": 2, "area": 3240, "bbox": [10, 20, 50, 60] },
        { "id": 2,  "category_id": 2, "area": 1800, "bbox": [80, 30, 40, 50] },
        { "id": 3,  "category_id": 5, "area": 9100, "bbox": [0,  0, 512, 512] }
      ]
    }
  ]
}
```

En este ejemplo, los segmentos 1 y 2 son **dos instancias distintas** de la misma clase
(`category_id=2`, buildings). Sin este encoding no habría forma de distinguirlas.

### ¿Se puede cambiar?

**No, para Mask2Former**. El modelo espera exactamente este formato. Cambiar el encoding
rompería el entrenamiento.

---

## Diferencia entre `panoptic/` y `panoptic_viz/`

| Carpeta | Encoding | Para qué |
|---|---|---|
| `panoptic/` | `R + G×256 + B×256²` (COCO estándar) | Entrenar Mask2Former |
| `panoptic_viz/` | Color semántico por clase | Inspección visual humana |

### Colores usados en `panoptic_viz/`

Los mismos que define `gemini_segmentation_cloud.py`:

| Clase | Color RGB | Hex |
|---|---|---|
| buildings | (66, 47, 47) | `#422f2f` |
| fuel_infrastructure | (241, 196, 15) | `#F1C40F` |
| military_vehicles | (231, 76, 60) | `#E74C3C` |
| communication_and_radar | (233, 30, 99) | `#E91E63` |
| roads_and_tracks | (127, 140, 141) | `#7F8C8D` |
| forest | (32, 151, 40) | `#209728` |
| perimeter_structures | (52, 152, 219) | `#3498DB` |
| void (sin anotar) | (0, 0, 0) | `#000000` |

---

## Clases: things vs stuff

En segmentación panóptica hay dos tipos de clase:

| Tipo | Significado | Clases en este dataset |
|---|---|---|
| **thing** (`isthing=1`) | Objetos contables con instancias separadas | buildings, fuel_infrastructure, military_vehicles, communication_and_radar, perimeter_structures |
| **stuff** (`isthing=0`) | Regiones amorfas sin instancias individuales | roads_and_tracks, forest |

- Para las clases **thing**: cada polígono/máscara genera un segmento con ID único.
  Dos edificios = dos `segment_id` distintos, mismo `category_id`.
- Para las clases **stuff**: todos los píxeles de la misma clase en la imagen se funden
  en **un único segmento**. Toda la vegetación = un `segment_id`, un `category_id`.

Las clases y su tipo se leen automáticamente de `GRDINO/config/classes.yaml`
(campo `is_stuff: true` para stuff, ausente para things).

---

## Píxeles sin anotación (void)

Los píxeles que no están cubiertos por ninguna anotación se dejan como **void** (`segment_id=0`).
Mask2Former ignora estos píxeles al calcular la pérdida, lo que es útil cuando las anotaciones
son incompletas.

---

## Cómo se procesan las anotaciones de Label Studio

Label Studio exporta tres tipos de anotación. El script maneja dos de ellos:

### 1. `polygonlabels` (polígonos)

Las coordenadas vienen en **porcentaje** del tamaño de imagen (0–100). Se convierten a píxeles
absolutos y se rasterizan con `cv2.fillPoly`.

```
punto_pixel_x = punto_pct_x / 100 × width
punto_pixel_y = punto_pct_y / 100 × height
```

### 2. `brushlabels` (trazos de pincel — formato RLE)

Label Studio codifica los trazos de pincel como **Run-Length Encoding (RLE)**:

- Lista de enteros que alternan conteos de píxeles transparentes y opacos:
  `[n_bg, n_fg, n_bg, n_fg, ...]`
- El orden de los píxeles es **column-major** (columna 0 de arriba a abajo,
  luego columna 1, etc.) — igual que el SDK oficial de Label Studio.

Decodificación:

```python
flat = np.zeros(width * height)
pos = 0; is_fg = False
for count in rle:
    if is_fg:
        flat[pos:pos+count] = 1
    pos += count
    is_fg = not is_fg
mask = flat.reshape(width, height).T  # column-major → (H, W)
```

### 3. `rectanglelabels` (rectángulos)

Se ignoran. En el dataset actual no hay rectángulos sin polígono asociado
(Label Studio los generó como bounding boxes auxiliares junto a los polígonos).

---

## Prioridad al rasterizar (sin solapamiento de instancias)

En panóptica, **cada píxel pertenece a exactamente un segmento**. El orden de pintado es:

1. Clases **stuff** primero (forest, roads) — se funden en un único segmento por clase.
2. Clases **thing** encima — cada instancia sobreescribe lo que había debajo.

Esto garantiza que los objetos individuales (vehículos, edificios) no queden oscurecidos
por regiones de fondo como vegetación.

---

## Estrategias de split

### `--split random` (80/10/10)

Mezcla aleatoriamente las 101 imágenes (tiles) con semilla fija (`--seed 42` por defecto)
y las reparte 80 % train / 10 % val / 10 % test.

**Riesgo**: tiles del mismo emplazamiento pueden acabar en splits distintos, lo que puede
inflar las métricas (data leakage entre tiles adyacentes).

### `--split location` (por emplazamiento)

Cada emplazamiento (carpeta bajo `google_maps_web/`) va completo a un único split.
Se configura en `split_locations.yaml`:

```yaml
train:
  - Cabo_Noval
  - Constantina_radar
  - ...
val:
  - San_Clemente
test:
  - General_menacho_botoa
```

**Ventaja**: no hay leakage entre tiles del mismo sitio. Recomendado para imágenes de satélite
donde tiles adyacentes comparten contexto visual.

**Emplazamientos disponibles** (9 en total, 101 tiles):
`Cabo_Noval`, `Constantina_radar`, `Frasno_radar`, `General_menacho_botoa`,
`GOLOSO_18`, `Penas_del_chache_radar`, `Pozo_Canarias_radar`, `San_Clemente`, `Viator`

---

## Uso

```bash
# Split aleatorio 80/10/10, versión 1
python create_dataset.py --version 1 --split random

# Split por localización (editar split_locations.yaml primero)
python create_dataset.py --version 2 --split location

# Opciones avanzadas
python create_dataset.py --version 3 \
    --split location \
    --split-config mi_split.yaml \
    --labelstudio-json ruta/export.json \
    --output-dir D:/datasets/Dataset_V3
```

### Argumentos

| Argumento | Default | Descripción |
|---|---|---|
| `--version N` | `1` | Número de versión → carpeta `Dataset_VN/` |
| `--split` | `random` | `random` (80/10/10) o `location` |
| `--split-config` | `split_locations.yaml` | YAML de asignación de localizaciones |
| `--seed N` | `42` | Semilla aleatoria para `--split random` |
| `--labelstudio-json` | `src/dataset/export_from_label-studio/dataset_v1.json` | Export de Label Studio |
| `--images-root` | `src/images/google_maps_web/` | Carpeta raíz con las imágenes |
| `--classes-yaml` | `GRDINO/config/classes.yaml` | Definición de clases |
| `--output-dir` | `../Dataset_VN/` | Sobreescribir carpeta de salida |

---

## Dependencias

```
opencv-python
numpy
pyyaml
```

Ya disponibles en el `.venv` del proyecto.
