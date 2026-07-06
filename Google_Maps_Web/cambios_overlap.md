Modifica mi aplicación de extracción de tiles para añadir solape configurable entre tiles.

Contexto:
Estoy creando un dataset para segmentación panóptica de bases militares en imágenes satelitales. Actualmente genero tiles de una ROI seleccionada en Google Maps / Google Earth, normalmente de 512x512 px. Necesito mejorar el tiling añadiendo overlap para evitar que objetos importantes queden cortados en los bordes.

Objetivo:
Implementar tiling con overlap configurable.

Requisitos:
- FIX THIS: No se puede navegar por el mapa haciendo click y arrastrando. Para mover el mapa, se debe hacer scroll para alejar o acercar y para centrar en la ubicación deseada.
- Al presionar tecla "Escape" se debe cancelar la selección de ROI y limpiar el mapa.

1. Añadir un parámetro configurable llamado overlapPercent.
2. Valor por defecto: 20.
3. Debe permitirse cambiarlo desde la interfaz.
4. Para tileSize = 512 y overlapPercent = 20:
   - overlapPx = round(512 * 0.20) = 102 px
   - stridePx = tileSize - overlapPx = 410 px
5. El sistema debe generar los tiles avanzando por stridePx, no por tileSize.
6. Debe funcionar tanto horizontal como verticalmente.
7. Debe garantizar cobertura completa de la ROI, incluyendo bordes.
8. Si el último tile no encaja exactamente, debe generarse igualmente ajustado al borde de la ROI.
9. Cada tile debe guardar metadatos:
   - roiId
   - tileId
   - tileSize
   - overlapPercent
   - overlapPx
   - stridePx
   - row
   - col
   - pixelX
   - pixelY
   - bounds geográficos del tile
   - zoom
   - resolución estimada m/px si está disponible
10. Mantener compatibilidad con la estructura actual:
    TFM/src/images/google_maps_web/
       ROI_x/
          tile_000001.png
          tile_000002.png
          metadata.json

Estrategia recomendada:
- Usar tileSize = 512 px.
- Usar overlapPercent = 20% por defecto.
- Permitir valores entre 0 y 50%.
- Mostrar advertencia si overlapPercent > 30%, porque aumenta mucho el número de tiles.
- Mantener 20% como valor recomendado para segmentación panóptica.

Motivo:
El overlap es necesario porque en segmentación panóptica los objetos cortados en los bordes generan máscaras incompletas y errores en COCO panoptic. En bases militares, esto afecta especialmente a:
- vehículos militares
- vehículos civiles
- barracones
- hangares
- depósitos de combustible
- radares
- baterías antiaéreas
- perímetro de base militar
- carreteras
- patios de armas
- zonas pavimentadas

Implementación esperada:
- Actualizar frontend para incluir input de overlapPercent.
- Actualizar backend o lógica de generación de tiles para calcular stridePx.
- Actualizar la generación de la malla visual sobre el mapa.
- Actualizar el guardado de tiles.
- Actualizar metadata.json.
- Añadir logs claros indicando:
  tileSize, overlapPercent, overlapPx, stridePx, número total de filas, número total de columnas y número total de tiles.
- Evitar duplicados.
- No romper el flujo actual de selección de ROI.

Fórmula:
overlapPx = round(tileSize * overlapPercent / 100)
stridePx = tileSize - overlapPx

Ejemplo:
tileSize = 512
overlapPercent = 20
overlapPx = 102
stridePx = 410

Criterios de aceptación:
- Si selecciono una ROI, la app genera una malla con tiles solapados.
- Los tiles vecinos comparten aproximadamente el porcentaje indicado.
- La ROI queda cubierta completamente.
- Los tiles de borde no dejan zonas sin capturar.
- Los metadatos reflejan correctamente el overlap.
- Puedo cambiar el overlap desde la UI antes de guardar.