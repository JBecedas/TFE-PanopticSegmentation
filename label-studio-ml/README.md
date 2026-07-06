# Label Studio ML Backend — SAM (Segment Anything Model)

ML backend con SAM ViT-H integrado en Label Studio via Docker Compose.

---

## Credenciales y configuración

| Parámetro | Valor |
|-----------|-------|
| Label Studio URL | http://localhost:8080 |
| Usuario | javiermbecedas@gmail.com |
| API Token | Ver `.env` (no incluido en el repo — usar `.env.example` como plantilla) |
| ML Backend URL | http://localhost:9090 |

> Token legacy (tabla `authtoken_token`). Label Studio 1.23.0 deshabilita estos tokens por defecto.
> Si el token deja de funcionar (error "legacy token authentication has been disabled"), ejecutar:
> ```powershell
> python C:\Users\CTEET\AppData\Local\Temp\fix_ls_token.py
> ```
> O bien con sqlite3:
> `UPDATE jwt_auth_jwtsettings SET legacy_api_tokens_enabled=1 WHERE organization_id=1;`
> BD: `C:\Users\CTEET\AppData\Local\label-studio\label-studio\label_studio.sqlite3`

---

## Arquitectura

```
Label Studio (nativo, :8080)
        ↕  API REST + token
SAM ML Backend (Docker, :9090)
  └── Pesos: C:\TFM\Labeling\scripts\sam_vit_h_4b8939.pth (2.4GB, ViT-H)
             montados en /app/models/ dentro del contenedor
```

Los mismos pesos son usados por GRDINO en `pipeline_gdino_sam.py:38`.

---

## Prerrequisitos

1. **Docker Desktop** con WSL2 backend (ya instalado)
2. **NVIDIA Container Toolkit** para GPU — verificar con:
   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```
3. **Label Studio corriendo** en http://localhost:8080

### Arrancar Label Studio con local files habilitado

Las imágenes están en `C:\TFM\src\images\`. Para que el ML backend pueda acceder a ellas,
Label Studio debe arrancarse con estas variables:

```powershell
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "C:\TFM\src"
label-studio start
```

---

## Primer uso (build + run)

```powershell
cd C:\TFM\label-studio-ml

# Primera vez: construye la imagen (~5-10 min, descarga dependencias de GitHub)
docker compose up --build

# Siguientes veces (imagen ya construida)
docker compose up
```

El backend estará listo cuando aparezca en los logs:
```
Uvicorn running on http://0.0.0.0:9090
```

---

## Conectar el ML backend a Label Studio

1. Abrir Label Studio → proyecto → **Settings → Model**
2. Clic en **Connect Model**
3. URL del modelo: `http://localhost:9090`
4. Guardar y verificar que el estado sea **Connected**

> Nota: el modelo SAM ViT-H tarda ~2s en generar el embedding de cada imagen.

---

## Variables de entorno (docker-compose)

| Variable | Valor configurado | Descripción |
|----------|-------------------|-------------|
| `SAM_CHOICE` | `SAM` | Modelo: SAM (ViT-H), MobileSAM, o ONNX |
| `VITH_CHECKPOINT` | `/app/models/sam_vit_h_4b8939.pth` | Ruta de pesos dentro del contenedor |
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU habilitada |
| `WORKERS` | `1` | Workers gunicorn |
| `THREADS` | `8` | Threads por worker |
| `MODEL_DIR` | `/data/models` | Directorio de estado del servidor |
| `LABEL_STUDIO_HOST` | `http://host.docker.internal:8080` | URL de LS desde dentro del contenedor |

---

## Parar y limpiar

```powershell
# Parar el backend
docker compose down

# Parar y borrar imagen (para reconstruir desde cero)
docker compose down --rmi all
```

---

## Troubleshooting

### Error: "GPU not available"
```powershell
# Verificar NVIDIA en Docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```
Si falla: instalar [NVIDIA Container Toolkit for WSL2](https://docs.nvidia.com/cuda/wsl-user-guide/).

### Error: "Connection refused" al conectar a Label Studio
- Verificar que LS esté corriendo en el 8080
- `host.docker.internal` resuelve automáticamente en Docker Desktop (Windows) — no hace falta configuración extra

### Error: pesos no encontrados
- Verificar que el fichero exista: `C:\TFM\Labeling\scripts\sam_vit_h_4b8939.pth`
- El volumen en docker-compose lo monta en `/app/models/` dentro del contenedor

### Ver logs en tiempo real
```powershell
docker compose logs -f
```
