# Producción normal:
python pipeline_gdino_sam.py

# Calibración todas las clases (5 tiles aleatorios de producción):
python pipeline_gdino_sam.py --calibrate

# Calibración clase específica con imágenes de prueba:
python pipeline_gdino_sam.py --calibrate \
    --class military_vehicles \
    --images /ruta/imagenes_test \
    --n-samples 10