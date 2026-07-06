Quiero que me diseñes e implementes un pipeline completo, modular y automatizable para crear un dataset de segmentación panóptica de bases militares a partir de imágenes satelitales ya descargadas.

## Contexto del proyecto
Estoy haciendo un TFM sobre segmentación panóptica de bases militares en imágenes satelitales.

Ya tengo las imágenes organizadas en esta ruta local:

TFM/src/images/google_maps_web

Dentro de esa carpeta hay subcarpetas, y cada subcarpeta corresponde a una ROI distinta. Dentro de cada ROI hay tiles ya generados, normalmente imágenes de 512x512 px procedentes de Google Maps / Google Earth o una captura equivalente.

Las imágenes tienen resolución alta, suficiente para distinguir edificios, carreteras, vegetación, superficies pavimentadas y, en algunos casos, objetos pequeños.

Quiero automatizar al máximo la creación del dataset y dejar un flujo sólido para:
1. presegmentación automática con SAM
2. revisión y corrección manual
3. anotación semántica/panóptica
4. exportación final a formato COCO panoptic

---

## Objetivo general
Construye un sistema end-to-end que:
- recorra automáticamente todas las carpetas de ROI
- cargue todos los tiles
- genere presegmentaciones automáticas con SAM o SAM2 si es conveniente
- permita revisión y corrección manual con Label Studio
- si hace falta, cree una aplicación externa en React + Node.js para facilitar la anotación asistida y acelerar el trabajo
- clasifique y guarde el resultado final en formato COCO panoptic
- permita reanudar procesos, versionar anotaciones y regenerar salidas sin perder trabajo previo

---

## Ideas clave del pipeline que debes respetar
Quiero que sigas esta lógica de trabajo:

imagen → máscaras → anotaciones → COCO panoptic → entrenamiento

Y esta estrategia:

Google Maps image
   ↓
SAM (auto máscaras)
   ↓
Label Studio (corregir + etiquetar clases)
   ↓
Export COCO
   ↓
Convertir a COCO Panoptic

También quiero que tengas en cuenta esta explicación funcional, que resume exactamente cómo quiero trabajar con cada imagen:

- Tengo imágenes tipo 512×512 px con resolución alta, por ejemplo ~0.12 m/px.
- Eso es suficientemente bueno para segmentar edificios, carreteras, vegetación, zonas pavimentadas y detalles finos.
- Para una imagen, el pipeline correcto es:
  imagen → máscaras → anotaciones → COCO panoptic → entrenamiento
- Primero hay que definir clases “things” y “stuff”.
- Luego generar máscaras automáticamente con SAM en todo lo posible.
- Después corregir de forma manual lo que haga falta.
- El formato final NO es SAM; SAM solo ayuda a generar máscaras.
- El formato final debe ser COCO panoptic.
- La salida debe incluir:
  1. imagen original
  2. máscara panóptica
  3. JSON compatible con COCO panoptic
- La estrategia ideal es:
  Google Maps image → SAM → revisión/corrección → exportación COCO → conversión a COCO panoptic
- No quiero un sistema basado solo en bounding boxes.
- Necesito máscaras por píxel.
- El objetivo es escalar a cientos o miles de tiles.
- Para el fondo no hace falta una precisión quirúrgica, pero sí consistencia.
- Para objetos principales sí quiero precisión razonable.

---

## Clases a utilizar
Quiero que el sistema soporte estas clases iniciales y sea fácil ampliarlo después.

### Things (instancias)
- barracones
- depósitos de combustible
- carros de combate
- radares
- baterías antiaéreas
- hangares
- vehículo civil
- vehículo militar
- perímetro_base como instancia si detectas segmentos individualizables, si no como stuff especializado
- otras estructuras militares relevantes si el diseño lo permite

### Stuff (regiones)
- carretera
- vegetación
- terreno
- pavimento
- patio_de_armas
- zona_operativa
- sombra si lo consideras útil
- perímetro_base si resulta mejor tratarlo como clase tipo stuff lineal/delgada
- Sport Facility

### Regla importante
Distinguir explícitamente entre:
- vehículo civil
- vehículo militar (habitualmente verde oliva)

Además:
- intenta diferenciar el perímetro de la base militar
- suele aparecer como líneas rectas y delgadas
- puede corresponder a alambradas, muros o cerramientos
- quiero que el sistema intente asistir especialmente en esta clase aunque sea difícil

Otras zonas comunes que deben contemplarse:
- patios de armas
- hangares

---

## Requisitos funcionales
Quiero una solución robusta y práctica, no una demo superficial.

### 1. Descubrimiento automático del dataset
- recorrer TFM/src/images/google_maps_web
- detectar automáticamente cada ROI
- detectar todos los tiles válidos de imagen
- generar un índice del dataset
- guardar metadatos por ROI y por tile

### 2. Presegmentación automática
- integrar SAM o SAM2
- proponer máscaras automáticas por tile
- guardar máscaras intermedias
- priorizar objetos grandes y regiones claras:
  - edificios
  - carreteras
  - vegetación
  - pavimento
  - hangares
  - patios
- para objetos pequeños:
  - radares
  - carros
  - baterías
  - vehículos
  usar prompts, heurísticas o clasificación posterior si es necesario

### 3. Asistencia a anotación manual
- integrar Label Studio si es suficiente
- si Label Studio no cubre bien el caso, crear una app externa en React + Node.js que:
  - muestre la imagen
  - superponga máscaras SAM
  - permita aceptar/rechazar/corregir máscaras
  - permita asignar clase con atajos
  - permita fusionar, dividir y editar polígonos/máscaras
  - permita navegar por ROI y tile
  - permita versionado de cambios
  - permita exportar a COCO y COCO panoptic
- si desarrollas esta app, reutiliza componentes o ideas de Label Studio si es viable, pero no dependas de hacks frágiles

### 4. Sistema de etiquetado
- cada máscara debe poder convertirse en:
  - instancia thing con id único
  - o región stuff
- permitir relabeling posterior sin perder trazabilidad
- guardar clases en un archivo de configuración central
- soportar ontología editable

### 5. Exportación final
- exportar a formato COCO panoptic
- generar:
  - images/
  - panoptic masks PNG
  - annotations JSON
- incluir categories con isthing
- incluir segments_info
- generar ids consistentes y reproducibles
- dividir opcionalmente en train / val / test por ROI, no mezclando tiles de la misma ROI entre splits

### 6. Calidad y productividad
- soporte para checkpoints y reanudación
- logs claros
- scripts CLI reutilizables
- documentación de instalación y uso
- configuración por variables o archivos yaml/json
- posibilidad de correr en local con GPU si existe
- fallback CPU si no hay GPU

---

## Requisitos técnicos
Quiero que propongas e implementes la mejor arquitectura práctica para esto.

### Backend
Preferencia:
- Python para procesamiento, SAM, conversión, utilidades de dataset
- Node.js solo si hace falta para la app de anotación

### Frontend
Si hace falta interfaz personalizada:
- React
- visor de imagen con zoom y pan fluido
- overlay de máscaras
- edición de polígonos/máscaras
- asignación rápida de clases
- filtros por ROI, estado, clase, dificultad

### Estructura esperada
Propón una estructura de proyecto limpia, por ejemplo:
- scripts/
- src/
- backend/
- frontend/
- data/
- exports/
- configs/
- docs/

### Debes incluir
- pipeline de descubrimiento de imágenes
- pipeline de presegmentación SAM
- pipeline de preparación para anotación
- pipeline de exportación a COCO panoptic
- ejemplo de configuración
- comandos de ejecución
- README claro

---

## Heurísticas y lógica deseada
Quiero que pienses de forma inteligente para automatizar lo máximo posible.

### Heurísticas útiles
- edificios grandes y rectangulares con tejados homogéneos pueden ser barracones o hangares
- estructuras alargadas pueden ser barracones
- áreas amplias despejadas y pavimentadas pueden ser patio_de_armas o zona_operativa
- líneas finas rectas en perímetro pueden indicar perímetro_base
- vehículos verde oliva pueden sugerir vehículo militar
- vehículos de colores comunes o formas civiles pueden sugerir vehículo civil
- depósitos de combustible pueden aparecer como estructuras cilíndricas o agrupaciones específicas
- radar y batería antiaérea serán clases difíciles; permite marcarlas como candidatas o low-confidence para revisión humana

### Importante
No quiero falsas promesas.
Si una clase no puede segmentarse automáticamente con fiabilidad, diseña el sistema para:
- generar candidatos
- resaltar posibles regiones
- dejar la decisión final al anotador

---

## Qué quiero que entregues
Quiero que generes todo lo necesario para que el proyecto pueda arrancar ya.

### Entregables mínimos
1. Diseño de arquitectura
2. Árbol de archivos del proyecto
3. Código base funcional
4. Scripts CLI
5. Integración con SAM
6. Flujo de anotación manual
7. Exportación a COCO panoptic
8. README con instalación y uso
9. Propuesta de ontología editable
10. Ejemplo real sobre una carpeta ROI de muestra

### Si detectas que la mejor solución es híbrida
Haz una solución híbrida:
- Python para visión y dataset
- React/Node para anotación asistida
- Label Studio como apoyo si ayuda

---

## Criterios de calidad
- código limpio y modular
- tipado cuando tenga sentido
- comentarios útiles
- manejo de errores
- scripts reutilizables
- no crear algo bloqueado a una sola carpeta
- diseño escalable para cientos o miles de tiles
- dejar preparado el camino para entrenamiento posterior con Detectron2 o Mask2Former

---

## Muy importante
No me des solo una explicación. Quiero que construyas el proyecto o, si no puedes completarlo todo de una vez, que empieces por una primera versión funcional muy sólida con:
- exploración automática de carpetas
- inferencia SAM
- persistencia de máscaras/candidatos
- estructura preparada para anotación manual
- exportación preliminar a COCO panoptic

Empieza proponiendo la arquitectura y luego implementa el código inicial con una estructura real de archivos.
Centrate en la segmetnacion de mascaras e instancias. LAs imagenes las proporciono yo.
Ruta para guardar, que siga la arquitectura COCO