Quiero que desarrolles una aplicación web en React + JavaScript, lista para ejecutar en local, con una interfaz limpia y funcional, orientada a la descarga, visualización y recorte de imágenes satelitales de Google Maps.

## SKILLS

- @prompt_engineering-patterns
- @skill-gemini-google-maps-tool
- Google Maps Tile API
- Google Maps Static API

## Objetivo general

Construir una web que integre mapas de Google usando principalmente la Maps Tile API, y que además permita usar funcionalidades equivalentes a Maps Static para captura/exportación de imágenes. La aplicación debe servir para navegar por el mapa, centrar ubicaciones, visualizar zoom y resolución, seleccionar una ROI rectangular, exportar la imagen recortada con metadatos, y gestionar una pequeña galería de recortes guardados localmente.

 

---

 

## Stack y restricciones técnicas

- Frontend: React + JavaScript

- Estilo: CSS simple o CSS Modules, sin necesidad de frameworks pesados

- Estructura clara y mantenible

- Código modular, comentado y listo para ampliar

- La app debe ser ejecutable en local

- Debe incluir instrucciones de instalación y ejecución

- Usa Google Maps Tile API como base principal del mapa

- Si para ciertas funciones de captura/exportación es necesario apoyarse en Maps Static API, intégralo de forma coherente

- Gestiona las claves API mediante variables de entorno (Disponibles en src/data/credentials_google_maps.json)

- El proyecto debe incluir manejo de errores, validaciones y mensajes visuales para el usuario

 

---

 

## Diseño de la interfaz

La aplicación debe tener:

- Un mapa principal ocupando casi toda la pantalla

- Un menú vertical fijo en el lado izquierdo

- El menú lateral debe contener controles, información en tiempo real y la lista de imágenes guardadas

- La interfaz debe ser clara, funcional y pensada para escritorio

 

---

 

## Funcionalidades obligatorias

 

### 1. Integración del mapa

- Integrar el mismo mapa visual de Google Maps usando la Maps Tile API

- Permitir cargar el mapa y trabajar con funcionalidades compatibles con:

  - visualización mediante tiles

  - captura/exportación tipo Maps Static

- Debe existir un botón en el menú lateral para cargar/inicializar el mapa

 

### 2. Navegación e información en tiempo real

Mientras el usuario navega por el mapa, en el menú lateral debe visualizarse en tiempo real:

- zoom actual

- resolución estimada en m/px

- coordenadas del centro o del cursor

- tamaño visible aproximado de la vista si es posible

 

La resolución m/px debe calcularse correctamente según zoom y latitud.

 

### 3. Inputs de coordenadas

- Deben existir dos inputs:

  - latitud

  - longitud

- Cuando el usuario introduzca unas coordenadas válidas y confirme:

  - el mapa debe centrarse en ese punto

  - debe colocarse un PIN visible sobre esa ubicación

- El PIN debe poder:

  - ocultarse

  - eliminarse

  - volver a mostrarse si procede

 

### 4. Herramienta de recorte / ROI

- Debe existir un botón “Recorte de imagen” o similar

- Al activarlo, el usuario debe poder dibujar una región rectangular directamente sobre el mapa

- Durante el trazado del rectángulo, mientras arrastra el cursor, debe mostrarse dinámicamente:

  - tamaño en píxeles

  - tamaño real estimado en metros o kilómetros

- Ejemplo visual esperado:

  - `250x250 px / 25m x 25m`

  - o `512x512 px / 5.12km x 5.12km`

- La ROI debe quedar visible una vez seleccionada

- Debe poder cancelarse, reajustarse o eliminarse

 

### 5. Guardado de la ROI

- Una vez seleccionada la ROI, debe haber un botón para guardarla

- La imagen debe guardarse en:

  - `/src/images/google_maps_web`

- Si por limitaciones del navegador no se puede escribir directamente en esa ruta desde frontend puro, implementa la solución adecuada con backend ligero o Node/Express y explícalo

- El guardado debe generar también sus metadatos asociados

 

### 6. Formato de salida

Permitir seleccionar el formato de salida entre:

- JPEG

- TIFF

 

### 7. Metadatos de la imagen

Cada imagen guardada debe incluir metadatos asociados. Como mínimo:

- nombre de la imagen

- fecha y hora de creación

- coordenadas del polígono ROI

- coordenada del centro

- lugar / país

- proveedor o fuente de la imagen

- satélite o servicio proveedor, en la medida en que la API lo permita de forma realista

- resolución m/px

- zoom aplicado

- tamaño de imagen en píxeles

- tamaño real cubierto por la ROI

 

Si TIFF permite incrustar metadatos reales, hacerlo.  Además. generar un JSON sidecar junto a la imagen.

Si JPEG no permite guardar todos los metadatos de forma práctica, generar un JSON sidecar junto a la imagen.

 

### 8. Lista de imágenes guardadas

- En el menú lateral debe mostrarse una lista de las imágenes que se van guardando en el directorio

- Cada elemento de la lista debe mostrar al menos:

  - nombre

  - formato

  - fecha

  - resolución

- Al hacer clic sobre una imagen:

  - debe abrirse en una nueva pestaña del navegador

- Si existen metadatos asociados, también deberían poder consultarse

 

---

 

## Requisitos funcionales adicionales

- Validación de coordenadas de entrada

- Manejo de errores de API

- Manejo de errores si no hay ROI seleccionada

- Feedback visual cuando se guarda una imagen

- Indicador de carga si se está generando o descargando la imagen

- Soporte básico para múltiples capturas en una sesión

- Posibilidad de renombrar la imagen antes de guardar

 

---

 

## Consideraciones técnicas importantes

Quiero que evalúes con criterio técnico qué partes deben resolverse en frontend y cuáles en backend. 

Si guardar archivos en `/src/images/google_maps_web`, generar TIFF con metadatos, o listar ficheros del directorio requiere backend, entonces:

- crea una arquitectura con React en frontend y un backend ligero en Node.js/Express

- documenta claramente esa decisión

- implementa ambos lados

 

No quiero una solución ficticia o incompleta: quiero una solución funcional y coherente con las limitaciones reales del navegador y de las APIs de Google.

 

---

 

## Entregables esperados

Genera:

1. La estructura completa del proyecto

2. El código fuente de todos los archivos principales

3. Un README con:

   - requisitos

   - instalación

   - variables de entorno

   - cómo ejecutar frontend y backend

   - cómo configurar Google Maps Tile API / Maps Static API

4. Explicación breve de la arquitectura elegida

5. Comentarios en el código en las partes críticas

6. Una implementación visualmente usable, no solo un prototipo mínimo roto

 

---

 

## Calidad esperada

- Código limpio y modular

- Componentes React bien separados

- Estado manejado de forma ordenada

- Sin pseudocódigo

- Sin funciones vacías

- Sin asumir capacidades imposibles del navegador sin resolverlas correctamente

- Si alguna parte no puede hacerse exactamente como se pide por limitaciones reales de Google o del navegador, propón e implementa la alternativa más cercana y funcional, explicándolo claramente

 

---

 

## Extra deseable

Si puedes, añade:

- overlay del rectángulo con borde claro y etiqueta flotante

- botón para copiar coordenadas de la ROI

- mini panel de metadatos al seleccionar una imagen guardada

- soporte para recalcular resolución según latitud de forma precisa

 

Desarrolla la solución completa.