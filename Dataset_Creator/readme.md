Lo que se ha creado
Dataset_Creator/create_dataset.py
Script principal. Dos modos de uso:


# Split aleatorio 80/10/10
python create_dataset.py --version 1 --split random

# Split por localización (edita split_locations.yaml primero)
python create_dataset.py --version 1 --split location
Argumentos opcionales:

Argumento	Descripción
--version N	Número de versión → genera Dataset_VN/
--split random|location	Estrategia de split
--split-config PATH	YAML de localización alternativo
--seed N	Semilla aleatoria (default 42)
--output-dir PATH	Sobreescribir carpeta de salida
Dataset_Creator/split_locations.yaml
Config editable para el modo --split location. Mueve localizaciones entre train/val/test según necesites.

Estructura de salida (Dataset_V1/)

Dataset_V1/
├── train/images/          ← tiles JPEG copiados
├── train/panoptic/        ← PNGs con ID codificado en RGB
├── val/   (ídem)
├── test/  (ídem)
├── panoptic_train.json    ← COCO panoptic JSON
├── panoptic_val.json
├── panoptic_test.json
└── dataset_info.json      ← resumen de clases, splits y estadísticas
Notas técnicas
Polígonos → cv2.fillPoly (coordenadas % → píxeles absolutos)
Brush RLE → decode column-major (formato Label Studio SDK)
Prioridad: clases stuff (forest, roads) se rasterzan primero; things encima → cada píxel pertenece a exactamente un segmento
Píxeles sin anotación → ID=0 (void/ignore)
Dependencias: opencv-python, numpy, pyyaml (ya deberías tenerlas en el venv)


### Primera version
============================================================
  Done! Dataset_V1 ready at:
  C:\TFM\Dataset_V1
============================================================

  Split    Images  Segments
  ─────────────────────────
  train        61       913
  val          11       187
  test         29       388
  ─────────────────────────
  total       101      1488