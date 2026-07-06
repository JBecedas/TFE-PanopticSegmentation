# Informe: Opciones de IA para Super-Resolución de Imágenes Satelitales

**Objetivo:** Mejorar calidad visual de tiles 512×512 px de Google Maps Web (imágenes aéreas)  
**Restricción:** Salida siempre 512×512 px (upscale interno 4× → 2048 → downscale Lanczos)  
**Hardware disponible:** NVIDIA RTX 4000 Ada (20 GB VRAM)  

---

## Resumen Ejecutivo

> **Recomendación: Opción A — Real-ESRGAN local (GPU)**  
> Es la única opción que hace super-resolución real, no regenera la imagen, es gratuita y preserva el contenido exacto. Crítico para imágenes con contenido geográfico/militar sensible.

---

## Análisis de Opciones

### Opción A — Real-ESRGAN Local (RTX 4000 Ada) ⭐ RECOMENDADA

| Criterio | Valoración |
|----------|-----------|
| Calidad | ★★★★★ — Super-resolución real, PSNR/SSIM más altos |
| Velocidad | ★★★★★ — ~0.3–1 s/imagen en RTX 4000 Ada (FP16) |
| Coste | ★★★★★ — Gratis tras instalar pesos (~65 MB) |
| Privacidad | ★★★★★ — Datos nunca salen del equipo |
| Fidelidad | ★★★★★ — Preserva geometría y estructuras exactas |

**Cómo funciona:**  
Red neuronal GAN entrenada específicamente para super-resolución en imágenes reales (no generativa). Upscale real 4× (512→2048) + downscale Lanczos a 512. El resultado tiene más detalle y menos artefactos JPEG que la original.

**Modelos disponibles:**
- `RealESRGAN_x4plus.pth` — General purpose, ideal para satelital (65 MB)
- `RealESRGAN_x4plus_denoised.pth` — Variante con denoising agresivo
- `realesr-general-x4v3.pth` — Más rápido, menor VRAM

**Instalación:**
```cmd
pip install realesrgan basicsr facexlib opencv-python tqdm
python IMGscale\enhance_images.py --mode local
```

---

### Opción B — OpenAI GPT-Image-1 (API de pago)

| Criterio | Valoración |
|----------|-----------|
| Calidad | ★★★☆☆ — Regenera la imagen, no super-resolución real |
| Velocidad | ★★☆☆☆ — 5–15 s/imagen (latencia API + transferencia) |
| Coste | ★★☆☆☆ — ~$0.02–0.06 por imagen (100 imgs ≈ $2–6) |
| Privacidad | ★★☆☆☆ — Imágenes se envían a servidores OpenAI |
| Fidelidad | ★★★☆☆ — Puede alterar detalles estructurales |

**Cómo funciona:**  
Usa el endpoint `images.edit` de `gpt-image-1`. Recibe la imagen y un prompt de mejora. **IMPORTANTE:** Este modelo es generativo (DALL-E), no un upscaler. Puede cambiar detalles de la imagen original. No apto si la fidelidad geográfica es crítica.

**Casos de uso válidos:** Cuando la apariencia visual importa más que la precisión geográfica.

**Instalación:**
```cmd
pip install openai pillow
set OPENAI_API_KEY=sk-...
python IMGscale\enhance_images.py --mode openai
```

---

### Opción C — Google Gemini Imagen 3 (API de pago)

| Criterio | Valoración |
|----------|-----------|
| Calidad | ★★★☆☆ — Regenera la imagen, no super-resolución real |
| Velocidad | ★★☆☆☆ — 5–20 s/imagen (latencia variable) |
| Coste | ★★★☆☆ — ~$0.02–0.04 por imagen (acceso incluido en Gemini Advanced) |
| Privacidad | ★★☆☆☆ — Imágenes enviadas a Google |
| Fidelidad | ★★★☆☆ — Similar limitación que GPT-Image-1 |

**Cómo funciona:**  
Usa `gemini-2.0-flash-preview-image-generation`. El modelo puede recibir una imagen y generar una versión modificada. También generativo, no super-resolución pura.

**Instalación:**
```cmd
pip install google-genai pillow
set GEMINI_API_KEY=AIza...
python IMGscale\enhance_images.py --mode gemini
```

---

### Opción D — Claude (Anthropic) — NO VIABLE

Claude puede analizar imágenes pero **no puede generar ni modificar imágenes**. No es una opción para este caso de uso.

---

### Opción E — Alternativas Locales Adicionales

| Herramienta | Calidad | Velocidad | Notas |
|-------------|---------|-----------|-------|
| **HAT (Hybrid Attention Transformer)** | ★★★★★ | ★★★☆☆ | Mejor calidad que ESRGAN, más lento |
| **SwinIR** | ★★★★☆ | ★★★☆☆ | Base de transformers, muy robusto |
| **Waifu2x** | ★★★☆☆ | ★★★★☆ | Más antiguo, funciona bien para 2× |
| **Topaz Gigapixel AI** | ★★★★★ | ★★★★☆ | Software comercial $99/año, GUI fácil |
| **StableSR** | ★★★★☆ | ★★☆☆☆ | Usa Stable Diffusion, más VRAM |

---

## Tabla Comparativa Final

| | Real-ESRGAN Local | OpenAI GPT-Image-1 | Google Gemini | HAT Local |
|--|:-:|:-:|:-:|:-:|
| Super-resolución real | ✅ | ❌ | ❌ | ✅ |
| Preserva estructuras | ✅ | ⚠️ | ⚠️ | ✅ |
| Privacidad total | ✅ | ❌ | ❌ | ✅ |
| Coste por 1000 imgs | $0 | ~$20–60 | ~$20–40 | $0 |
| Velocidad (RTX 4000) | ~1 s/img | ~10 s/img | ~15 s/img | ~3 s/img |
| Facilidad de uso | Alta | Alta | Alta | Media |
| **RECOMENDACIÓN** | ✅ **1ª opción** | 3ª opción | 4ª opción | 2ª opción |

---

## Conclusión

Para imágenes aéreas/satelitales, el procesado **local con Real-ESRGAN es claramente superior**:

1. **Calidad:** Los modelos de super-resolución dedicados superan a los generativos para preservar detalles geográficos
2. **Velocidad:** La RTX 4000 Ada procesa 1000 imágenes en ~15 minutos vs. ~3 horas por API
3. **Coste:** Gratis frente a $20–60 por cada 1000 imágenes
4. **Privacidad:** El contenido no sale del equipo (importante para imágenes de instalaciones militares/sensibles)
5. **Fidelidad:** Un upscaler no altera el contenido; un modelo generativo puede hacerlo

Las APIs online (ChatGPT/Gemini) se justifican solo si no se puede configurar el entorno Python/CUDA local.
