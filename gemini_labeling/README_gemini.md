# Gemini Segmentation → Label Studio

Pipeline que usa Gemini para generar máscaras de segmentación semántica sobre imágenes aéreas y las preimporta en Label Studio para revisión y edición manual.

---

## 1. Arrancar Label Studio con soporte de archivos locales

Label Studio necesita acceso a las imágenes locales. Arrancarlo **siempre** con estas variables de entorno:

```powershell
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT   = "C:\TFM\src"
label-studio start
```

> Si Label Studio ya está corriendo sin estas variables, reinícialo con ellas.  
> La variable `DOCUMENT_ROOT = C:\TFM\src` hace que la URL  
> `/data/local-files/?d=images/google_maps_web/roi/tile.jpeg`  
> resuelva a `C:\TFM\src\images\google_maps_web\roi\tile.jpeg`.

---

## 2. Crear el proyecto en Label Studio

1. Abre http://localhost:8080 y crea un nuevo proyecto.
2. Ve a **Settings → Labeling Interface → Code**.
3. Pega exactamente este XML y guarda:

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <PolygonLabels name="label" toName="image" strokeWidth="3" opacity="0.4" ellipsepoints="4">
    <Label value="buildings"               background="#422f2f"/>
    <Label value="fuel_infrastructure"     background="#F1C40F"/>
    <Label value="military_vehicles"       background="#E74C3C"/>
    <Label value="communication_and_radar" background="#E91E63"/>
    <Label value="roads_and_tracks"        background="#7F8C8D"/>
    <Label value="forest"                  background="#209728"/>
    <Label value="perimeter_structures"    background="#3498DB"/>
  </PolygonLabels>
</View>
```

**Crítico:** los atributos `name="label"` y `name="image"` deben ser exactamente esos valores — el JSON generado por el script usa `from_name: "label"` y `to_name: "image"` para enlazar las predicciones con estos tags.

Los colores `background` coinciden con los que usa el script en la máscara PNG de salida.

---

## 3. Ejecutar el script en Colab

```
gemini_segmentation_cloud.py
```

Genera dos outputs en `TFM/gemini_labeling/outputs/` (Google Drive):

| Archivo | Descripción |
|---------|-------------|
| `Gemini_to_label_studio.json` | Predicciones en formato Label Studio |
| `masks/<roi>/<tile>_mask.png` | Superposición máscara + imagen original |
| `masks/<roi>/<tile>_mask_pure.png` | Máscara de color puro (sin imagen) |

---

## 4. Importar en Label Studio

1. En tu proyecto, ve a **Import**.
2. Sube `Gemini_to_label_studio.json`.
3. Las tareas aparecerán en la lista con un icono de predicción (número en azul).

---

## 5. Ver y aceptar predicciones

Al abrir una tarea:

1. Los polígonos de Gemini aparecen en el panel **Predictions** (lado derecho).
2. Para convertirlos en anotación editable: haz clic en **▶ Accept** junto a la predicción.
3. Una vez aceptada, puedes editar vértices, mover polígonos o eliminarlos.
4. Guarda con **Submit**.

> Las predicciones son **read-only** hasta que las aceptas.  
> Después de aceptar, son anotaciones normales completamente editables.

---

## 6. Clases y colores

| Clase | Color hex | Predicción automática (JSON) | Máscara PNG |
|-------|-----------|------------------------------|-------------|
| `buildings` | `#422f2f` | ✅ polígono + bbox | ✅ |
| `fuel_infrastructure` | `#F1C40F` | ✅ polígono + bbox | ✅ |
| `military_vehicles` | `#E74C3C` | ✅ polígono + bbox | ✅ |
| `communication_and_radar` | `#E91E63` | ✅ polígono + bbox | ✅ |
| `roads_and_tracks` | `#7F8C8D` | ✅ solo polígono | ✅ |
| `forest` | `#209728` | ✅ solo polígono | ✅ |
| `perimeter_structures` | `#3498DB` | ✅ solo polígono | ✅ |

---

## 7. Mostrar clases stuff (forest, roads\_and\_tracks, perimeter\_structures) en Label Studio

### Situación actual

El script genera predicciones JSON **solo para las 4 clases thing**.  
Las 3 clases stuff (`forest`, `roads_and_tracks`, `perimeter_structures`) aparecen pintadas en los PNG de salida pero **no se precargan como predicciones en Label Studio**.

Las 3 clases sí están declaradas en el XML de configuración (`<Label value="forest" .../>`), así que puedes etiquetarlas **manualmente** con la herramienta de polígonos.

### Opción A — Etiquetado manual (sin cambios en el script)

El XML actual ya incluye las 3 clases stuff. Para etiquetarlas:

1. Abre la tarea en Label Studio.
2. Acepta las predicciones de Gemini (clases thing).
3. Selecciona la clase deseada (`forest`, etc.) en el panel de etiquetas.
4. Dibuja el polígono manualmente sobre la imagen.
5. Usa las máscaras PNG de salida como referencia visual (ábrelas en paralelo).

### Opción B — Predicciones automáticas para stuff ✅ (implementado)

El script genera polígonos automáticos para las 7 clases:
- **Thing classes**: polígono de contorno + bounding box
- **Stuff classes**: solo polígono de contorno (sin bounding box)

No se requiere ningún cambio en el XML de configuración de Label Studio.

---

## 8. Referencia rápida de rutas

| Recurso | Ruta local |
|---------|-----------|
| Imágenes fuente | `C:\TFM\src\images\google_maps_web\` |
| Script | `C:\TFM\gemini_labeling\gemini_segmentation_cloud.py` |
| JSON de salida | `C:\TFM\gemini_labeling\outputs\Gemini_to_label_studio.json` |
| Máscaras PNG | `C:\TFM\gemini_labeling\outputs\masks\` |
| Label Studio DB | `C:\Users\CTEET\AppData\Local\label-studio\label-studio\label_studio.sqlite3` |
