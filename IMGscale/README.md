# IMGscale — Mejora de imágenes con IA

Script para mejorar la calidad de los tiles 512×512 px de `src/images/google_maps_web`
usando Real-ESRGAN (local/GPU) u OpenAI gpt-image-2 (API).

---

## Estructura de archivos

```
IMGscale/
├── enhance_images.py          ← script principal
├── prompt_openai.txt          ← prompt editable para el modo openai
├── to_annotations_labelStudio.json  ← generado automáticamente tras procesar
├── images/
│   └── google_maps_web/       ← imágenes mejoradas (misma estructura que src)
│       └── <carpeta>/
│           └── <imagen>.jpeg
└── weights/
    └── RealESRGAN_x4plus.pth  ← descargado automáticamente (65 MB)
```

---

## Instalación

### Modo local (Real-ESRGAN — RECOMENDADO)

```cmd
rem 1. Activar entorno virtual
cd c:\TFM
.venv\Scripts\activate

rem 2. PyTorch con CUDA (RTX 4000 Ada → CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

rem 3. Resto de dependencias
pip install -r IMGscale\requirements_local.txt
```

### Modo OpenAI

```cmd
.venv\Scripts\activate
pip install -r IMGscale\requirements_api.txt
```

---

## Uso

### Modo local (GPU)

```cmd
python IMGscale\enhance_images.py --mode local
```

### Modo OpenAI — gpt-image-2

```cmd
rem Opción A: clave en variable de entorno
set OPENAI_API_KEY=sk-proj-...
python IMGscale\enhance_images.py --mode openai

rem Opción B: clave directa
python IMGscale\enhance_images.py --mode openai --openai-key sk-proj-...
```

### Opciones adicionales

| Flag | Descripción |
|------|-------------|
| `--dry-run` | Lista imágenes pendientes sin procesar |
| `--force` | Reprocesa aunque el archivo de salida ya exista |
| `--source <ruta>` | Cambia el directorio de imágenes fuente |
| `--output <ruta>` | Cambia el directorio de salida |

---

## Personalizar el prompt de OpenAI

Edita `IMGscale/prompt_openai.txt` con cualquier editor de texto.
El script lo lee en cada ejecución, por lo que no necesitas tocar el código.

**Ejemplos de prompts:**

```
# Enfoque en nitidez y eliminación de artefactos JPEG:
Enhance sharpness, remove JPEG artifacts, improve edge definition in this
aerial satellite image. Keep all structures identical.

# Enfoque en denoising:
Reduce noise and grain in this satellite photograph while preserving
all geographic details, roads, and building structures exactly.

# Para imágenes nocturnas/oscuras:
Improve brightness and contrast of this satellite aerial image.
Enhance visibility of structures. Preserve exact layout and geometry.
```

---

## Librería OpenAI Python — Referencia rápida

### Instalación

```cmd
pip install openai
```

### Versión mínima requerida

```
openai >= 1.30.0
```

### Autenticación

```python
from openai import OpenAI

client = OpenAI(api_key="sk-proj-...")
# O sin argumento si está definido OPENAI_API_KEY en el entorno
client = OpenAI()
```

### Endpoint usado: `images.edit`

El script usa `client.images.edit()` que permite enviar una imagen existente
y un prompt de instrucciones. El modelo genera una versión modificada.

```python
with open("imagen.jpeg", "rb") as f:
    img_bytes = f.read()

response = client.images.edit(
    model="gpt-image-2",
    image=("imagen.jpeg", img_bytes, "image/jpeg"),  # (nombre, bytes, mime)
    prompt="Enhance quality, sharpen details...",
    size="1024x1024",   # tamaños: "256x256", "512x512", "1024x1024"
    n=1,                # número de variantes a generar
)

# La respuesta contiene la imagen en base64
import base64
from PIL import Image
import io

b64 = response.data[0].b64_json
img = Image.open(io.BytesIO(base64.b64decode(b64)))
img.save("salida.jpeg", quality=95)
```

### Otros endpoints útiles

```python
# Generar imagen desde prompt (sin imagen base)
response = client.images.generate(
    model="gpt-image-2",
    prompt="Satellite view of a military base...",
    size="1024x1024",
    n=1,
)

# Crear variación de una imagen existente
response = client.images.create_variation(
    model="dall-e-2",   # solo disponible en dall-e-2
    image=open("imagen.png", "rb"),
    n=1,
    size="1024x1024",
)
```

### Tamaños soportados por gpt-image-2

| Tamaño | Uso |
|--------|-----|
| `1024x1024` | Cuadrado estándar |
| `1536x1024` | Apaisado |
| `1024x1536` | Vertical |
| `auto` | El modelo elige |

### Manejo de errores

```python
from openai import OpenAI, APIError, RateLimitError, AuthenticationError

client = OpenAI()

try:
    response = client.images.edit(...)
except AuthenticationError:
    print("API key inválida")
except RateLimitError:
    print("Límite de peticiones alcanzado, espera unos segundos")
except APIError as e:
    print(f"Error de API: {e}")
```

### Coste aproximado (gpt-image-2)

| Calidad | Precio por imagen |
|---------|-------------------|
| Standard 1024×1024 | ~$0.04 |
| HD 1024×1024 | ~$0.08 |

> Consulta precios actuales en: platform.openai.com/docs/pricing

---

## Salida Label Studio

Tras cada ejecución se genera/actualiza `to_annotations_labelStudio.json`
con todas las imágenes presentes en `images/google_maps_web/`.

**Formato de las rutas:**
```
/data/local-files/?d=images/google_maps_web/<carpeta>/<imagen>.jpeg
```

Esto asume que Label Studio tiene configurado como directorio raíz de
archivos locales: `c:\TFM\IMGscale\`

**Estructura del JSON:**
```json
[
  {
    "data": {
      "image": "/data/local-files/?d=images/google_maps_web/Goloso1/Goloso1_tile_r03_c03.jpeg"
    },
    "predictions": []
  }
]
```

Para importar en Label Studio: `Projects → Import → Upload Files → selecciona to_annotations_labelStudio.json`
