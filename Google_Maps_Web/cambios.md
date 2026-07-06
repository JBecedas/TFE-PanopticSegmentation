Quiero que modifiques y mejores la aplicación web ya existente desarrollada en React + JavaScript para Google Maps / Maps Tile API. No quiero una reescritura desde cero, sino una actualización del proyecto actual, revisando componentes, estado, lógica de eventos, exportación de imágenes y experiencia de uso.

Debes actuar como un desarrollador senior frontend/full-stack con capacidad de debugging real. Analiza el código actual, detecta los fallos y aplica las correcciones necesarias de forma funcional y consistente.

## Objetivo general
Corregir errores de interacción con el mapa, permitir control manual del zoom, revisar el guardado de la ROI, y añadir una nueva funcionalidad clave: al guardar una ROI, generar automáticamente una malla de subimágenes a zoom 20 que cubran exactamente la misma zona geográfica seleccionada.

---

## Cambios obligatorios a implementar

### 1. Zoom editable desde el input
Actualmente la aplicación muestra el zoom actual en un input o campo visual, pero ese valor debe ser también editable por el usuario.

Requisitos:
- El input que muestra el zoom actual debe permitir escritura manual
- Si el usuario introduce un nuevo zoom válido:
  - el mapa debe actualizarse inmediatamente a ese zoom
  - el valor mostrado debe sincronizarse con el mapa real
- Si el zoom cambia con el ratón o con controles del mapa, el input debe actualizarse también
- Debe haber sincronización bidireccional completa:
  - mapa → input
  - input → mapa
- Validar rango de zoom permitido
- Si el valor introducido no es válido, mostrar error o restaurar el valor correcto

---

### 2. Guardado sin marcas visuales no deseadas
Cuando se guarda una imagen:
- no debe aparecer la marca de agua de Google
- no deben aparecer textos tipo `Imagery ©2026 Airbus`, `Map data`, u otras atribuciones visuales incrustadas en la imagen final

Importante:
- Revisa la legalidad y viabilidad técnica real de esto
- No quiero una respuesta superficial
- Quiero que evalúes si la fuente de captura actual está incluyendo overlays de atribución
- Si no se puede eliminar legítimamente usando una determinada API, propón e implementa una alternativa técnica correcta y realista
- Si la API elegida obliga a incluir atribución visible, explícalo claramente y propone el flujo técnicamente más limpio posible
- No inventes capacidades inexistentes

---

### 3. Bug en el guardado de la ROI
Hay un fallo actual:
- cuando selecciono un rectángulo sobre el mapa
- al guardar, no se guarda exactamente esa imagen
- se guarda una imagen más pequeña o incorrecta

Quiero que:
- revises cuidadosamente la lógica de selección ROI
- revises conversión entre coordenadas de pantalla, coordenadas geográficas y dimensiones de exportación
- revises offsets, escalados, devicePixelRatio, tiles, canvas, overlays y cualquier recorte incorrecto
- localices el bug real
- lo corrijas
- la imagen guardada debe corresponder exactamente al área visible delimitada por el rectángulo seleccionado

Quiero que esto se trate como tarea explícita de debugging profundo.

---

### 4. Navegación libre por el mapa
Actualmente no puedo moverme libremente por el mapa con el ratón. La imagen queda como bloqueada o “pillada”.

Quiero que:
- se revise la interacción del mapa
- el mapa pueda desplazarse libremente con drag/pan
- también se pueda hacer zoom normal con rueda o controles si procede
- si existen modos de interacción como “selección ROI”, estos no deben romper permanentemente la navegación
- la selección de ROI debe activarse y desactivarse correctamente
- al salir del modo recorte, el mapa debe volver a modo navegación normal
- aunque yo cambie las coordenadas con inputs y salte a un lugar, luego debo poder seguir moviéndome por el mapa con libertad

---

## Funcionalidad clave nueva a implementar

### 5. Generación automática de malla de subimágenes a zoom 20
Este es el cambio más importante.

#### Comportamiento requerido
Cuando el usuario seleccione una ROI rectangular cualquiera:
- esa ROI tendrá unas coordenadas geográficas delimitadoras
- normalmente podrá haber sido seleccionada a cualquier zoom, por ejemplo zoom 16
- al pulsar “Guardar”, el sistema debe:
  1. guardar la imagen principal correspondiente a la ROI seleccionada
  2. generar automáticamente una malla de subimágenes a zoom 20 que cubran exactamente la misma superficie geográfica

#### Explicación funcional
Ejemplo:
- el usuario selecciona una ROI a zoom 16
- esa ROI define un rectángulo geográfico real, delimitado por 4 coordenadas
- al guardar, además de la imagen original, se debe crear una matriz de subcuadrículas a zoom 20
- estas subimágenes deben cubrir toda la misma área geográfica de la ROI original

#### Requisitos de la malla
- la malla puede ser `n x m`
- todas las subimágenes deben tener exactamente el mismo tamaño en píxeles
- todas deben generarse a zoom 20
- deben cubrir por completo la ROI original
- puede permitirse un pequeño margen sobrante por arriba/abajo/izquierda/derecha si eso facilita una malla regular
- esos márgenes deben calcularse automáticamente
- quiero que tú calcules el criterio óptimo para:
  - tamaño fijo de subimagen
  - número de filas y columnas
  - pequeños márgenes laterales y superior/inferior
- la prioridad es que:
  - todas las subimágenes tengan el mismo tamaño
  - todas estén alineadas
  - juntas cubran la ROI completa
  - el proceso sea reproducible y consistente

#### Lógica que debes implementar
1. Obtener las coordenadas geográficas exactas de la ROI seleccionada
2. Reproyectarlas o convertirlas al sistema necesario para cálculo en zoom 20
3. Calcular el bounding box real de esa ROI
4. Definir una cuadrícula regular de recortes a zoom 20
5. Generar todas las subimágenes necesarias para cubrir esa zona
6. Guardarlas con nombres coherentes
7. Generar metadatos tanto:
   - de la ROI principal
   - como de cada subimagen de la malla

#### Nomenclatura sugerida
Ejemplo:
- `roi_001_main.jpg`
- `roi_001_tile_r00_c00.jpg`
- `roi_001_tile_r00_c01.jpg`
- `roi_001_tile_r01_c00.jpg`

o equivalente, pero consistente.
- Guardar todas las imágenes en una carpeta específica para cada ROI
- El nombre de la carpeta debe ser el nombre de la ROI
- Añade un input para que el usuario pueda introducir el nombre de la ROI


---

## Metadatos de las subimágenes
Cada subimagen de la malla debe tener al menos:
- nombre de archivo
- fila y columna
- zoom = 20
- coordenadas del subrectángulo
- coordenadas del centro
- dimensiones en píxeles
- dimensiones reales aproximadas
- referencia a la ROI padre

Si el formato no permite incrustarlo todo cómodamente, usar JSON sidecar.

---

## Revisión técnica obligatoria
Quiero que revises el código existente para detectar problemas en:
- eventos de mouse
- overlays
- bloqueo de interacción
- sincronización de estado
- cálculos de zoom
- cálculo de m/px
- exportación de canvas o tiles
- recortes incorrectos
- uso de Maps Tile API y Maps Static API
- guardado de archivos
- naming y estructura de carpetas

---

## Decisiones técnicas
Si alguna parte requiere backend o ajuste del flujo actual:
- intégralo correctamente
- no dejes pseudocódigo
- no dejes funciones vacías
- documenta las decisiones

Si la solución actual mezcla mal frontend y backend, refactorízala.

---

## Entregables esperados
Quiero que devuelvas:
1. El código actualizado
2. Los archivos modificados completos
3. Explicación breve de cada corrección importante
4. Descripción del bug detectado en el guardado de ROI
5. Explicación del algoritmo usado para generar la malla zoom 20
6. Cualquier limitación real derivada de Google Maps APIs, claramente explicada
7. Instrucciones para probar:
   - navegación
   - edición de zoom
   - selección ROI
   - guardado normal
   - generación de submalla zoom 20

---

## Calidad exigida
- Solución funcional
- Código consistente con el proyecto actual
- Buen debugging
- Sin inventar capacidades imposibles
- Sin omitir la parte compleja de la malla
- Implementación real, no teórica

Haz la modificación completa del proyecto existente.