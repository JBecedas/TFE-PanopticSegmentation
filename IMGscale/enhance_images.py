#!/usr/bin/env python3
"""
Mejora de calidad de imágenes satelitales 512x512 de google_maps_web.
Entrada: c:/TFM/src/images/google_maps_web/<carpeta>/<imagen>
Salida:  c:/TFM/IMGscale/images/google_maps_web/<carpeta>/<imagen>  (siempre 512x512)

Modos:
  local   -- Real-ESRGAN en GPU (RTX 4000 Ada). RECOMENDADO.
  openai  -- GPT-Image-2 via API OpenAI (prompt editable en prompt_openai.txt).
  gemini  -- Gemini 2.0 Flash image generation via API Google.

Uso:
  python enhance_images.py --mode local
  python enhance_images.py --mode openai --openai-key sk-...
  python enhance_images.py --mode gemini --gemini-key AIza...
  python enhance_images.py --dry-run             # solo lista imágenes
  python enhance_images.py --mode local --force  # reprocesa aunque existan

Tras procesar siempre genera/actualiza IMGscale/to_annotations_labelStudio.json
con todas las imágenes mejoradas presentes en la carpeta de salida.
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# ─── Configuración ─────────────────────────────────────────────────────────────
SOURCE_DIR       = Path(__file__).parent.parent / "src" / "images" / "google_maps_web"
OUTPUT_DIR       = Path(__file__).parent / "images" / "google_maps_web"
WEIGHTS_DIR      = Path(__file__).parent / "weights"
PROMPT_FILE      = Path(__file__).parent / "prompt_openai.txt"
LABELSTUDIO_JSON = Path(__file__).parent / "to_annotations_labelStudio.json"
TARGET_SIZE      = 512
IMG_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Prefijo de ruta que usa Label Studio (raíz = IMGscale/)
LS_PREFIX = "/data/local-files/?d="


# ─── Utilidades comunes ────────────────────────────────────────────────────────

def collect_images(source: Path) -> list:
    """Lista todas las imágenes en source manteniendo rutas relativas."""
    items = []
    for img in sorted(source.rglob("*")):
        if img.is_file() and img.suffix.lower() in IMG_EXTENSIONS:
            items.append((img, img.relative_to(source)))
    return items


def output_path_for(rel: Path, output_root: Path) -> Path:
    """Calcula ruta de salida y crea la carpeta si no existe."""
    out = output_root / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in (".jpg", ".jpeg"):
        out = out.with_suffix(".jpeg")
    return out


def load_prompt() -> str:
    """Lee el prompt de prompt_openai.txt. Si no existe, crea uno por defecto."""
    if not PROMPT_FILE.exists():
        default = (
            "Enhance the quality of this satellite aerial photograph: sharpen edges, "
            "reduce JPEG compression artifacts and noise, increase visual clarity and detail. "
            "Preserve the exact geographic structures, buildings, roads, and geometry. "
            "Do not add, remove, or modify any element. Output same composition."
        )
        PROMPT_FILE.write_text(default, encoding="utf-8")
        print(f"Creado prompt por defecto en: {PROMPT_FILE}")
    return PROMPT_FILE.read_text(encoding="utf-8").strip()


# ─── Label Studio JSON ─────────────────────────────────────────────────────────

def generate_labelstudio_json(output_root: Path) -> int:
    """
    Escanea output_root y genera to_annotations_labelStudio.json con todas
    las imágenes mejoradas. Ruta Label Studio: /data/local-files/?d=images/google_maps_web/...
    La raíz para las rutas relativas es IMGscale/ (parent de images/).
    Devuelve el número de entradas generadas.
    """
    ls_root = output_root.parent.parent  # IMGscale/
    entries = []
    for img in sorted(output_root.rglob("*")):
        if img.is_file() and img.suffix.lower() in IMG_EXTENSIONS:
            rel = img.relative_to(ls_root)          # images/google_maps_web/<carpeta>/<file>
            ls_path = LS_PREFIX + rel.as_posix()     # /data/local-files/?d=images/google_maps_web/...
            entries.append({"data": {"image": ls_path}, "predictions": []})

    LABELSTUDIO_JSON.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(entries)


# ─── Modo LOCAL: Real-ESRGAN ───────────────────────────────────────────────────

MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)


def _download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        print(f"\r  Descargando... {pct:.1f}% ({downloaded/1e6:.1f}/{total_size/1e6:.1f} MB)", end="")


def get_weights() -> Path:
    """Descarga pesos Real-ESRGAN si no existen."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = WEIGHTS_DIR / "RealESRGAN_x4plus.pth"
    if model_path.exists():
        return model_path
    print(f"Descargando pesos Real-ESRGAN x4plus (~65 MB) en {model_path}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path, _download_progress)
        print()
    except Exception as exc:
        print(f"\nERROR descargando pesos: {exc}")
        print(f"Descarga manual en: {MODEL_URL}")
        print(f"Guarda el archivo en: {model_path}")
        sys.exit(1)
    return model_path


def load_realesrgan():
    """Carga el modelo Real-ESRGAN en GPU si está disponible."""
    try:
        import torch
    except ImportError as e:
        print(f"ERROR importando torch: {e}")
        print("Instala:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError as e:
        print(f"ERROR importando basicsr: {e}")
        print("Instala:  pip install basicsr")
        print(f"Python usado: {sys.executable}")
        sys.exit(1)

    try:
        from realesrgan import RealESRGANer
    except ImportError as e:
        print(f"ERROR importando realesrgan: {e}")
        print("Instala:  pip install realesrgan facexlib gfpgan")
        print(f"Python usado: {sys.executable}")
        sys.exit(1)

    model_path = get_weights()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        print("ADVERTENCIA: CUDA no disponible. Procesado en CPU (lento).")
        print(f"Python: {sys.executable}")
        print("Verifica con:  nvidia-smi")

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path=str(model_path),
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=(device == "cuda"),
        device=device,
    )
    return upsampler


def enhance_local(img_path: Path, output_path: Path, upsampler) -> bool:
    """Upscale 4x con Real-ESRGAN, luego downscale Lanczos a 512x512."""
    import cv2

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  ERROR: no se puede leer {img_path.name}")
        return False

    enhanced, _ = upsampler.enhance(img, outscale=4)  # 512 → 2048
    final = cv2.resize(enhanced, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(output_path), final, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


# ─── Modo OPENAI: gpt-image-2 ─────────────────────────────────────────────────

def enhance_openai(img_path: Path, output_path: Path, client, prompt: str) -> bool:
    """
    Mejora con gpt-image-2 via OpenAI images.edit.
    El prompt se lee de prompt_openai.txt (editable entre ejecuciones).
    """
    from PIL import Image
    import io

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    response = client.images.edit(
        model="gpt-image-2",
        image=(img_path.name, img_bytes, "image/jpeg"),
        prompt=prompt,
        size="1024x1024",
        n=1,
    )

    b64 = response.data[0].b64_json
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    img.save(str(output_path), format="JPEG", quality=95)
    return True


# ─── Modo GEMINI: Imagen generation ──────────────────────────────────────────

GEMINI_PROMPT = (
    "Enhance the quality of this satellite aerial photograph. "
    "Sharpen edges and fine details, reduce compression noise and artifacts, "
    "improve visual clarity. Preserve the exact geographic structures, "
    "geometry, colors and layout. Do not alter the content."
)


def enhance_gemini(img_path: Path, output_path: Path, client) -> bool:
    """Mejora con Gemini 2.0 Flash image generation."""
    from PIL import Image
    import io
    from google.genai import types

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    suffix = img_path.suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"

    response = client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        contents=[
            types.Content(
                parts=[
                    types.Part(inline_data=types.Blob(mime_type=mime, data=img_bytes)),
                    types.Part(text=GEMINI_PROMPT),
                ]
            )
        ],
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )

    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
            img_out = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
            img_out = img_out.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
            img_out.save(str(output_path), format="JPEG", quality=95)
            return True

    print("  ERROR: Gemini no devolvió imagen en la respuesta.")
    return False


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Mejora imágenes google_maps_web con IA (local/OpenAI/Gemini)"
    )
    parser.add_argument(
        "--mode", choices=["local", "openai", "gemini"], default="local",
        help="Motor de mejora: local (Real-ESRGAN GPU), openai, gemini (default: local)"
    )
    parser.add_argument(
        "--source", type=Path, default=SOURCE_DIR,
        help=f"Directorio fuente (default: {SOURCE_DIR})"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR,
        help=f"Directorio de salida (default: {OUTPUT_DIR})"
    )
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"),
                        help="API key OpenAI (o variable de entorno OPENAI_API_KEY)")
    parser.add_argument("--gemini-key", default=os.getenv("GEMINI_API_KEY"),
                        help="API key Gemini (o variable de entorno GEMINI_API_KEY)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo lista imágenes que se procesarían, sin ejecutar")
    parser.add_argument("--force", action="store_true",
                        help="Reprocesa aunque el archivo de salida ya exista")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"ERROR: Directorio fuente no existe: {args.source}")
        sys.exit(1)

    images = collect_images(args.source)
    if not images:
        print(f"No se encontraron imágenes en: {args.source}")
        sys.exit(0)

    print(f"Modo       : {args.mode.upper()}")
    print(f"Fuente     : {args.source}")
    print(f"Salida     : {args.output}")
    print(f"Imágenes   : {len(images)}")
    print(f"Resolución : {TARGET_SIZE}×{TARGET_SIZE} px (salida)")
    print()

    if args.dry_run:
        for img, rel in images:
            out = output_path_for(rel, args.output)
            status = "EXISTS" if out.exists() else "PENDING"
            print(f"  [{status}] {rel}")
        return

    # ── Inicializar motor ──────────────────────────────────────────────────────
    if args.mode == "local":
        print("Cargando Real-ESRGAN...")
        enhancer = load_realesrgan()
        print("Modelo listo.\n")
        enhance_fn = lambda src, dst: enhance_local(src, dst, enhancer)

    elif args.mode == "openai":
        if not args.openai_key:
            print("ERROR: API key de OpenAI requerida.")
            print("Usa --openai-key <key>  o  set OPENAI_API_KEY=<key>")
            sys.exit(1)
        try:
            from openai import OpenAI
        except ImportError:
            print("ERROR: pip install openai pillow")
            sys.exit(1)
        client = OpenAI(api_key=args.openai_key)
        prompt = load_prompt()
        print(f"Prompt ({len(prompt)} chars): {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print()
        enhance_fn = lambda src, dst: enhance_openai(src, dst, client, prompt)

    elif args.mode == "gemini":
        if not args.gemini_key:
            print("ERROR: API key de Gemini requerida.")
            print("Usa --gemini-key <key>  o  set GEMINI_API_KEY=<key>")
            sys.exit(1)
        try:
            from google import genai
        except ImportError:
            print("ERROR: pip install google-genai pillow")
            sys.exit(1)
        client = genai.Client(api_key=args.gemini_key)
        enhance_fn = lambda src, dst: enhance_gemini(src, dst, client)

    # ── Procesar imágenes ──────────────────────────────────────────────────────
    ok = err = skipped = 0
    t_global = time.time()

    for i, (img_path, rel_path) in enumerate(images, 1):
        out_path = output_path_for(rel_path, args.output)
        prefix = f"[{i:>4}/{len(images)}]"

        if out_path.exists() and not args.force:
            print(f"{prefix} SKIP  {rel_path}")
            skipped += 1
            continue

        print(f"{prefix} {rel_path} ... ", end="", flush=True)
        t1 = time.time()
        try:
            success = enhance_fn(img_path, out_path)
            elapsed = time.time() - t1
            if success:
                print(f"OK ({elapsed:.1f}s)")
                ok += 1
            else:
                print("FALLO")
                err += 1
        except Exception as exc:
            print(f"ERROR — {exc}")
            err += 1

    # ── Resumen ────────────────────────────────────────────────────────────────
    total_time = time.time() - t_global
    print()
    print(f"{'─'*50}")
    print(f"OK: {ok}  |  Errores: {err}  |  Saltadas: {skipped}")
    print(f"Tiempo total: {total_time:.1f}s")
    if ok > 0:
        print(f"Velocidad media: {total_time / ok:.2f}s/imagen")
    print(f"Resultados en: {args.output}")

    # ── Generar JSON para Label Studio ─────────────────────────────────────────
    if args.output.exists():
        n = generate_labelstudio_json(args.output)
        print(f"Label Studio JSON: {LABELSTUDIO_JSON} ({n} entradas)")


if __name__ == "__main__":
    main()
