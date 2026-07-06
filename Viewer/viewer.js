/**
 * TIF VIEWER — viewer.js
 *
 * Prompt Engineering Patterns Applied:
 * ─────────────────────────────────────────────────────────────
 * • Role-Based System Design   : cada clase tiene una responsabilidad única
 * • Structured Output Pattern  : metadata normalizada con schema validado
 * • Chain-of-Thought Rendering : pipeline declarativo step-by-step
 * • Progressive Disclosure     : carga incremental con feedback de progreso
 * • Error Recovery + Fallback  : múltiples estrategias de decodificación
 * • Template Parameterisation  : file catalogue inyectado dinámicamente
 * ─────────────────────────────────────────────────────────────
 */

'use strict';

/* ═══════════════════════════════════════════════════════════
   CATALOGUE — dataset files relative to index.html location
   ═══════════════════════════════════════════════════════════ */
const CATALOGUE = {
  sentinel: [
    { name: '001_S-300PMU-2_2024-07-10.tif', size: '264 KB' },
    { name: '002_S-125_2024-10-31.tif', size: '192 KB' },
    { name: '003_S-300PMU-2_2024-07-23.tif', size: '268 KB' },
    { name: '004_2K12_garrison_2024-08-10.tif', size: '279 KB' },
    { name: '005_2K12_2024-07-07.tif', size: '229 KB' },
    { name: '006_S-300PMU-2_2024-10-31.tif', size: '235 KB' },
    { name: '007_S-125_2024-08-06.tif', size: '255 KB' },
    { name: '008_S-125_2024-07-10.tif', size: '282 KB' },
    { name: '009_S-125_2024-07-10.tif', size: '286 KB' },
    { name: '010_S-125_2024-11-04.tif', size: '288 KB' },
  ],
  gmaps: [
    { name: '001_S-300PMU-2_2023-10-12.tif', size: '3.1 MB' },
    { name: '002_S-125_2023-01-10.tif', size: '2.5 MB' },
    { name: '003_S-300PMU-2_2024-08-17.tif', size: '2.6 MB' },
    { name: '004_2K12_garrison_2024-06-06.tif', size: '3.1 MB' },
    { name: '005_2K12_2023-06-23.tif', size: '2.9 MB' },
    { name: '006_S-300PMU-2_2023-01-10.tif', size: '2.9 MB' },
    { name: '007_S-125_2023-08-02.tif', size: '2.7 MB' },
    { name: '008_S-125_2024-08-17.tif', size: '600 KB' },
    { name: '009_S-125_2023-10-12.tif', size: '2.9 MB' },
    { name: '010_S-125_2023-08-02.tif', size: '2.5 MB' },
  ]
};

/* Path prefixes relative to Viewer/index.html */
const SOURCE_PREFIX = {
  sentinel: '../src/images/Sentinel/',
  gmaps: '../src/images/google_earth/'
};

/* ═══════════════════════════════════════════════════════════
   STATE — single source of truth
   ═══════════════════════════════════════════════════════════ */
const State = {
  activeSource: 'sentinel',
  activeFile: null,
  // Raw decoded raster data
  raster: null,  // { width, height, bands, data[], dtype, meta }
  imageData: null,  // processed ImageData for current view settings
  // View transform
  zoom: 1.0,
  panX: 0,
  panY: 0,
  isPanning: false,
  panStart: { x: 0, y: 0, originX: 0, originY: 0 },
  // Adjustments
  channel: 'rgb',
  contrast: 1.0,
  brightness: 0,
};

/* ═══════════════════════════════════════════════════════════
   DOM REFS
   ═══════════════════════════════════════════════════════════ */
const $ = id => document.getElementById(id);

const DOM = {
  fileList: $('file-list'),
  canvasViewport: $('canvas-viewport'),
  canvas: $('tif-canvas'),
  emptyState: $('empty-state'),
  loadingOverlay: $('loading-overlay'),
  loadingText: $('loading-text'),
  progressBar: $('progress-bar'),
  metaPanel: $('metadata-panel'),
  metaGrid: $('meta-grid'),
  metaToggle: $('btn-meta-toggle'),
  statusbar: $('statusbar'),
  statusFilename: $('status-filename'),
  statusSize: $('status-size'),
  statusBands: $('status-bands'),
  statusCursor: $('status-cursor'),
  statusPixelVal: $('status-pixel-val'),
  zoomPct: $('zoom-pct'),
  contrastSlider: $('contrast-slider'),
  contrastVal: $('contrast-val'),
  brightnessSlider: $('brightness-slider'),
  brightnessVal: $('brightness-val'),
  channelSelect: $('channel-select'),
  dropZone: $('drop-zone'),
  fileInput: $('file-input'),
  badgeDot: document.querySelector('.badge-dot'),
  badgeLabel: $('badge-label'),
  pixelTooltip: $('pixel-tooltip'),
};

const ctx = DOM.canvas.getContext('2d', { willReadFrequently: true });

/* ═══════════════════════════════════════════════════════════
   CATALOGUE RENDERING — Pattern: Template Parameterisation
   ═══════════════════════════════════════════════════════════ */
function renderFileList(source) {
  const files = CATALOGUE[source];
  const iconClass = source === 'sentinel' ? 'sentinel' : 'gmaps';
  const iconLabel = source === 'sentinel' ? 'S2' : 'GM';

  DOM.fileList.innerHTML = files.map((f, i) => `
    <div class="file-item" data-source="${source}" data-filename="${f.name}"
         id="file-item-${source}-${i}" role="button" tabindex="0"
         aria-label="Cargar ${f.name}">
      <div class="file-icon ${iconClass}">${iconLabel}</div>
      <div class="file-info">
        <div class="file-name" title="${f.name}">${f.name}</div>
        <div class="file-size">${f.size}</div>
      </div>
    </div>
  `).join('');

  // Attach click listeners
  DOM.fileList.querySelectorAll('.file-item').forEach(el => {
    el.addEventListener('click', () => handleFileSelect(el.dataset.source, el.dataset.filename, el));
    el.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') el.click(); });
  });
}

/* Restore active highlight after re-render */
function restoreActiveHighlight() {
  if (!State.activeFile) return;
  const selector = `[data-filename="${CSS.escape(State.activeFile)}"][data-source="${State.activeSource}"]`;
  const el = DOM.fileList.querySelector(selector);
  if (el) el.classList.add('active');
}

/* ═══════════════════════════════════════════════════════════
   GeoTIFF LOADING — Pattern: Progressive Disclosure + Error Recovery
   ═══════════════════════════════════════════════════════════ */

/**
 * Load a GeoTIFF from a URL path.
 * Chain-of-thought pipeline:
 *  1. Validate & show loading state
 *  2. Fetch + parse with geotiff.js
 *  3. Read raster data from first image
 *  4. Extract structured metadata
 *  5. Render to canvas
 *  6. Update UI state
 */
async function loadGeoTIFF(url, filename, source) {
  showLoading('Fetcheando archivo GeoTIFF…', 5);

  try {
    // Step 1: Parse GeoTIFF
    setProgress(15, 'Parseando cabecera TIFF…');
    const tiff = await GeoTIFF.fromUrl(url, { allowFullFile: true });

    setProgress(30, 'Leyendo imagen…');
    const image = await tiff.getImage();

    // Step 2: Extract structured metadata (Structured Output Pattern)
    const meta = extractMetadata(image, filename, source);

    setProgress(50, `Decodificando ${meta.width}×${meta.height} px, ${meta.bands} banda(s)…`);

    // Step 3: Read raster data
    const rasterData = await image.readRasters({ interleave: false });

    setProgress(80, 'Procesando datos de banda(s)…');

    // Step 4: Store in State
    State.raster = {
      width: meta.width,
      height: meta.height,
      bands: meta.bands,
      data: Array.from({ length: meta.bands }, (_, i) => rasterData[i]),
      dtype: meta.dtype,
      meta,
    };
    State.activeFile = filename;
    State.activeSource = source;

    // Step 5: Reset adjustments for new file
    State.channel = DOM.channelSelect.value;
    State.contrast = parseFloat(DOM.contrastSlider.value);
    State.brightness = parseInt(DOM.brightnessSlider.value);

    setProgress(90, 'Renderizando…');

    // Step 6: Render
    renderToCanvas();
    fitToViewport();
    updateMetadataPanel(meta);
    updateBadge(source);
    updateStatusBar(meta);
    showCanvas();

    setProgress(100, 'Listo.');
    setTimeout(hideLoading, 300);

  } catch (err) {
    console.error('[TIFViewer] Error cargando GeoTIFF:', err);
    // Error Recovery Fallback: try as standard image via <img> element
    await fallbackImageLoad(url, filename, source, err);
  }
}

/**
 * Fallback: render the TIF as a regular <img> via browser native decoder.
 * Some TIFs (JPEG-compressed RGB) are supported natively by Safari/Chrome.
 */
async function fallbackImageLoad(url, filename, source, originalError) {
  setProgress(0, 'Intentando decodificación alternativa…');
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      setProgress(90, 'Renderizando (modo compatibilidad)…');
      DOM.canvas.width = img.naturalWidth;
      DOM.canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);

      const fakeMeta = {
        filename, source,
        width: img.naturalWidth,
        height: img.naturalHeight,
        bands: 'N/A (decodificación nativa)',
        dtype: 'uint8 (estimado)',
        nodata: '—',
        crs: '—',
        resolution: '—',
        originX: '—',
        originY: '—',
        pixelSizeX: '—',
        pixelSizeY: '—',
        fileSize: '—',
        note: 'Modo compatibilidad: metadatos geoespaciales no disponibles.',
      };

      // Store minimal raster info
      const imgData = ctx.getImageData(0, 0, img.naturalWidth, img.naturalHeight);
      State.raster = {
        width: img.naturalWidth, height: img.naturalHeight,
        bands: 3, data: null, dtype: 'uint8', meta: fakeMeta,
        nativeImageData: imgData,
      };
      State.activeFile = filename;
      State.activeSource = source;

      updateMetadataPanel(fakeMeta);
      updateBadge(source);
      updateStatusBar(fakeMeta);
      fitToViewport();
      showCanvas();
      setProgress(100, 'Listo (modo compatibilidad).');
      setTimeout(hideLoading, 400);
      resolve();
    };
    img.onerror = () => {
      hideLoading();
      showError(filename, originalError);
      resolve();
    };
    img.src = url;
  });
}

/* ═══════════════════════════════════════════════════════════
   METADATA EXTRACTION — Structured Output Pattern
   Schema: { filename, source, width, height, bands, dtype,
             nodata, crs, resolution, originX, originY,
             pixelSizeX, pixelSizeY, fileSize }
   ═══════════════════════════════════════════════════════════ */
function extractMetadata(image, filename, source) {
  // Safe accessor: geotiff.js v2 may return TypedArrays or plain arrays
  const safeGet = (fn, fallback) => { try { const r = fn(); return (r != null) ? r : fallback; } catch (_) { return fallback; } };
  const safeIdx = (fn, idx, fallback) => { try { const r = fn(); return (r != null && r[idx] != null) ? r[idx] : fallback; } catch (_) { return fallback; } };

  const fd = safeGet(() => image.getFileDirectory(), {});
  const gk = safeGet(() => image.getGeoKeys ? image.getGeoKeys() : {}, {});

  const width = image.getWidth();
  const height = image.getHeight();
  const bands = image.getSamplesPerPixel();

  // getBitsPerSample returns TypedArray in geotiff.js v2 — access by index, not destructuring
  const bpsRaw = safeGet(() => image.getBitsPerSample ? image.getBitsPerSample() : null, null);
  const bps = bpsRaw != null ? (bpsRaw[0] ?? bpsRaw) : 8;

  // Determine dtype from bits-per-sample and sample format
  const sampleFormat = (fd.SampleFormat && fd.SampleFormat[0]) ? fd.SampleFormat[0] : 1;
  let dtype = 'uint8';
  if (bps === 16) dtype = sampleFormat === 2 ? 'int16' : 'uint16';
  else if (bps === 32) dtype = sampleFormat === 3 ? 'float32' : (sampleFormat === 2 ? 'int32' : 'uint32');
  else if (bps === 64) dtype = 'float64';

  // Geospatial: getOrigin() → [x, y, z], getResolution() → [xRes, yRes, zRes]
  let realOriginX = '—', realOriginY = '—', realPSX = '—', realPSY = '—';
  const origin = safeGet(() => image.getOrigin ? image.getOrigin() : null, null);
  const res = safeGet(() => image.getResolution ? image.getResolution() : null, null);
  if (origin && origin[0] != null) { realOriginX = fmtCoord(origin[0]); realOriginY = fmtCoord(origin[1]); }
  if (res && res[0] != null) { realPSX = fmtNum(res[0]); realPSY = fmtNum(Math.abs(res[1])); }

  // CRS
  let crs = '—';
  if (gk.GeographicTypeGeoKey) crs = `EPSG:${gk.GeographicTypeGeoKey}`;
  if (gk.ProjectedCSTypeGeoKey) crs = `EPSG:${gk.ProjectedCSTypeGeoKey}`;

  // Resolution description
  let resolution = '—';
  if (realPSX !== '—') {
    resolution = crs.includes('4326') || crs.includes('4258')
      ? `${realPSX}° × ${realPSY}°`
      : `${realPSX} m/px × ${realPSY} m/px`;
  }

  // Nodata
  const nodata = fd.GDAL_NODATA ? String(fd.GDAL_NODATA).trim() : '—';

  // Compression
  const compMap = { 1: 'Sin compresión', 5: 'LZW', 6: 'JPEG (old)', 7: 'JPEG', 8: 'Deflate', 32773: 'PackBits' };
  const comp = fd.Compression ? (compMap[fd.Compression] || `ID:${fd.Compression}`) : '—';

  // Date from filename
  const source_label = source === 'sentinel' ? 'Sentinel-2' : (source === 'gmaps' ? 'Google Maps' : 'Archivo local');
  const dateMatch = filename.match(/(\d{4}-\d{2}-\d{2})/);
  const dateStr = dateMatch ? dateMatch[1] : '—';

  return {
    filename, source: source_label,
    width, height, bands, dtype, bps,
    nodata, crs, resolution,
    originX: realOriginX, originY: realOriginY,
    pixelSizeX: realPSX, pixelSizeY: realPSY,
    compression: comp,
    date: dateStr,
    fileSize: '—',
  };
}

/* ═══════════════════════════════════════════════════════════
   CANVAS RENDERING — Chain-of-Thought Pipeline
   Step 1: Determine active bands
   Step 2: Normalise/convert data range to [0,255]
   Step 3: Apply contrast + brightness
   Step 4: Write to ImageData
   Step 5: Draw to canvas
   ═══════════════════════════════════════════════════════════ */
function renderToCanvas() {
  const { raster, channel, contrast, brightness } = State;
  if (!raster) return;

  // If native image data (fallback mode), just apply adjustments
  if (raster.nativeImageData) {
    applyAdjustmentsToNativeImage(raster.nativeImageData);
    return;
  }

  const { width, height, bands, data, dtype } = raster;

  // Step 1: Determine which band indices to use
  let rBand, gBand, bBand, greyBand;
  const nb = bands;
  if (channel === 'rgb' || channel === 'gray') {
    if (nb >= 3) { rBand = 0; gBand = 1; bBand = 2; }
    else if (nb === 2) { rBand = 0; gBand = 0; bBand = 0; }
    else { rBand = 0; gBand = 0; bBand = 0; }
  } else if (channel === 'r') { rBand = gBand = bBand = 0; }
  else if (channel === 'g') { rBand = gBand = bBand = (nb > 1 ? 1 : 0); }
  else if (channel === 'b') { rBand = gBand = bBand = (nb > 2 ? 2 : 0); }

  // Step 2: Normalise range per band
  const normalize = (bandData) => {
    let min = Infinity, max = -Infinity;
    const len = bandData.length;
    for (let i = 0; i < len; i++) {
      const v = bandData[i];
      if (isNaN(v) || v === undefined) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min || 1;
    const out = new Float32Array(len);
    for (let i = 0; i < len; i++) out[i] = ((bandData[i] - min) / range) * 255;
    return { out, min, max };
  };

  const nR = normalize(data[rBand]);
  const nG = (gBand === rBand) ? nR : normalize(data[gBand]);
  const nB = (bBand === rBand) ? nR : (bBand === gBand ? nG : normalize(data[bBand]));

  // Step 3 & 4: Apply adjustments, write ImageData
  DOM.canvas.width = width;
  DOM.canvas.height = height;

  const imgData = ctx.createImageData(width, height);
  const buf = imgData.data;
  const half = width * height;

  for (let i = 0; i < half; i++) {
    let r = nR.out[i], g = nG.out[i], b = nB.out[i];

    // Contrast (pivot at 128)
    r = (r - 128) * contrast + 128;
    g = (g - 128) * contrast + 128;
    b = (b - 128) * contrast + 128;

    // Brightness
    r += brightness;
    g += brightness;
    b += brightness;

    if (channel === 'gray') { const lum = 0.299 * r + 0.587 * g + 0.114 * b; r = g = b = lum; }

    buf[i * 4] = clamp255(r);
    buf[i * 4 + 1] = clamp255(g);
    buf[i * 4 + 2] = clamp255(b);
    buf[i * 4 + 3] = 255;
  }

  // Step 5: Draw
  ctx.putImageData(imgData, 0, 0);
  State.imageData = imgData;
}

function applyAdjustmentsToNativeImage(srcData) {
  const { width, height } = State.raster;
  const { contrast, brightness, channel } = State;

  DOM.canvas.width = width;
  DOM.canvas.height = height;

  const imgData = ctx.createImageData(width, height);
  const src = srcData.data;
  const dst = imgData.data;

  for (let i = 0; i < src.length; i += 4) {
    let r = (src[i] - 128) * contrast + 128 + brightness;
    let g = (src[i + 1] - 128) * contrast + 128 + brightness;
    let b = (src[i + 2] - 128) * contrast + 128 + brightness;

    if (channel === 'gray') { const lum = 0.299 * r + 0.587 * g + 0.114 * b; r = g = b = lum; }
    else if (channel === 'r') { g = 0; b = 0; }
    else if (channel === 'g') { r = 0; b = 0; }
    else if (channel === 'b') { r = 0; g = 0; }

    dst[i] = clamp255(r);
    dst[i + 1] = clamp255(g);
    dst[i + 2] = clamp255(b);
    dst[i + 3] = src[i + 3];
  }
  ctx.putImageData(imgData, 0, 0);
  State.imageData = imgData;
}

/* ═══════════════════════════════════════════════════════════
   PAN & ZOOM
   ═══════════════════════════════════════════════════════════ */
function applyTransform() {
  DOM.canvas.style.transform = `translate(${State.panX}px, ${State.panY}px) scale(${State.zoom})`;
  DOM.zoomPct.textContent = `${Math.round(State.zoom * 100)}%`;
}

function fitToViewport() {
  if (!State.raster) return;
  const { width, height } = State.raster;
  const vp = DOM.canvasViewport.getBoundingClientRect();
  const padding = 40;
  const scaleX = (vp.width - padding) / width;
  const scaleY = (vp.height - padding) / height;
  State.zoom = Math.min(scaleX, scaleY, 1.0); // never upscale on fit
  State.panX = (vp.width - width * State.zoom) / 2;
  State.panY = (vp.height - height * State.zoom) / 2;
  applyTransform();
}

function zoom(factor, pivotX, pivotY) {
  const vp = DOM.canvasViewport.getBoundingClientRect();
  const cx = pivotX ?? vp.width / 2;
  const cy = pivotY ?? vp.height / 2;

  const newZoom = Math.min(Math.max(State.zoom * factor, 0.02), 50);
  const ratio = newZoom / State.zoom;
  State.panX = cx - ratio * (cx - State.panX);
  State.panY = cy - ratio * (cy - State.panY);
  State.zoom = newZoom;
  applyTransform();
}

/* ═══════════════════════════════════════════════════════════
   EVENT HANDLERS
   ═══════════════════════════════════════════════════════════ */

/* ── Source tabs ── */
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.remove('active');
      b.setAttribute('aria-selected', 'false');
    });
    btn.classList.add('active');
    btn.setAttribute('aria-selected', 'true');
    State.activeSource = btn.dataset.source;
    renderFileList(btn.dataset.source);
    restoreActiveHighlight();
  });
});

/* ── File select from catalogue ── */
async function handleFileSelect(source, filename, el) {
  // Highlight
  DOM.fileList.querySelectorAll('.file-item').forEach(e => e.classList.remove('active'));
  el.classList.add('active');

  const url = SOURCE_PREFIX[source] + filename;
  await loadGeoTIFF(url, filename, source);
}

/* ── Drop zone ── */
DOM.dropZone.addEventListener('dragover', e => { e.preventDefault(); DOM.dropZone.classList.add('dragover'); });
DOM.dropZone.addEventListener('dragleave', () => DOM.dropZone.classList.remove('dragover'));
DOM.dropZone.addEventListener('drop', async e => {
  e.preventDefault();
  DOM.dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) await loadFileFromBlob(file);
});
DOM.dropZone.addEventListener('click', () => DOM.fileInput.click());
DOM.dropZone.addEventListener('keydown', e => { if (e.key === 'Enter') DOM.fileInput.click(); });
DOM.fileInput.addEventListener('change', async e => {
  const file = e.target.files[0];
  if (file) await loadFileFromBlob(file);
});

async function loadFileFromBlob(file) {
  const url = URL.createObjectURL(file);
  const filename = file.name;
  showLoading('Cargando archivo local…', 5);
  await loadGeoTIFF(url, filename, 'custom');
  URL.revokeObjectURL(url);
  // Deselect catalogue items
  DOM.fileList.querySelectorAll('.file-item').forEach(e => e.classList.remove('active'));
}

/* ── Zoom buttons ── */
$('btn-zoom-in').addEventListener('click', () => zoom(1.25));
$('btn-zoom-out').addEventListener('click', () => zoom(0.8));
$('btn-zoom-fit').addEventListener('click', () => fitToViewport());
$('btn-zoom-100').addEventListener('click', () => {
  if (!State.raster) return;
  const vp = DOM.canvasViewport.getBoundingClientRect();
  State.zoom = 1.0;
  State.panX = (vp.width - State.raster.width) / 2;
  State.panY = (vp.height - State.raster.height) / 2;
  applyTransform();
});

/* ── Wheel zoom ── */
DOM.canvasViewport.addEventListener('wheel', e => {
  if (!State.raster) return;
  e.preventDefault();
  const rect = DOM.canvasViewport.getBoundingClientRect();
  const pivotX = e.clientX - rect.left;
  const pivotY = e.clientY - rect.top;
  const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
  zoom(factor, pivotX, pivotY);
}, { passive: false });

/* ── Pan (mouse drag) ── */
DOM.canvasViewport.addEventListener('mousedown', e => {
  if (e.button !== 0 || !State.raster) return;
  State.isPanning = true;
  State.panStart = { x: e.clientX, y: e.clientY, originX: State.panX, originY: State.panY };
  DOM.canvasViewport.style.cursor = 'grabbing';
});
window.addEventListener('mousemove', e => {
  if (!State.isPanning) { updateCursorInfo(e); return; }
  State.panX = State.panStart.originX + (e.clientX - State.panStart.x);
  State.panY = State.panStart.originY + (e.clientY - State.panStart.y);
  applyTransform();
});
window.addEventListener('mouseup', () => {
  State.isPanning = false;
  DOM.canvasViewport.style.cursor = '';
});

/* ── Touch pan ── */
let lastTouchX = 0, lastTouchY = 0;
DOM.canvasViewport.addEventListener('touchstart', e => {
  if (e.touches.length === 1) {
    lastTouchX = e.touches[0].clientX;
    lastTouchY = e.touches[0].clientY;
  }
}, { passive: true });
DOM.canvasViewport.addEventListener('touchmove', e => {
  if (e.touches.length === 1) {
    State.panX += e.touches[0].clientX - lastTouchX;
    State.panY += e.touches[0].clientY - lastTouchY;
    lastTouchX = e.touches[0].clientX;
    lastTouchY = e.touches[0].clientY;
    applyTransform();
  }
}, { passive: true });

/* ── Adjust controls ── */
DOM.channelSelect.addEventListener('change', () => {
  State.channel = DOM.channelSelect.value;
  if (State.raster) renderToCanvas();
});
DOM.contrastSlider.addEventListener('input', () => {
  State.contrast = parseFloat(DOM.contrastSlider.value);
  DOM.contrastVal.textContent = `${State.contrast.toFixed(1)}×`;
  if (State.raster) renderToCanvas();
});
DOM.brightnessSlider.addEventListener('input', () => {
  State.brightness = parseInt(DOM.brightnessSlider.value);
  DOM.brightnessVal.textContent = State.brightness >= 0 ? `+${State.brightness}` : `${State.brightness}`;
  if (State.raster) renderToCanvas();
});
$('btn-reset-adjust').addEventListener('click', () => {
  DOM.contrastSlider.value = '1';
  DOM.brightnessSlider.value = '0';
  DOM.channelSelect.value = 'rgb';
  State.contrast = 1.0;
  State.brightness = 0;
  State.channel = 'rgb';
  DOM.contrastVal.textContent = '1.0×';
  DOM.brightnessVal.textContent = '+0';
  if (State.raster) renderToCanvas();
});

/* ── Download PNG ── */
$('btn-download').addEventListener('click', () => {
  if (!State.raster) return;
  const link = document.createElement('a');
  const base = (State.activeFile || 'image').replace(/\.tiff?$/i, '');
  link.download = `${base}_${State.channel}_z${Math.round(State.zoom * 100)}.png`;
  link.href = DOM.canvas.toDataURL('image/png');
  link.click();
});

/* ── Fullscreen ── */
$('btn-fullscreen').addEventListener('click', () => {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen?.();
  } else {
    document.exitFullscreen?.();
  }
});

/* ── Metadata panel ── */
$('btn-meta-close').addEventListener('click', () => { DOM.metaPanel.classList.add('hidden'); DOM.metaToggle.classList.remove('hidden'); });
DOM.metaToggle.addEventListener('click', () => { DOM.metaPanel.classList.remove('hidden'); DOM.metaToggle.classList.add('hidden'); });

/* ── Pixel cursor info ── */
function updateCursorInfo(e) {
  if (!State.raster || !State.imageData) return;

  const vpRect = DOM.canvasViewport.getBoundingClientRect();
  const mx = e.clientX - vpRect.left;
  const my = e.clientY - vpRect.top;

  // Convert viewport coords → canvas pixel coords
  const px = Math.floor((mx - State.panX) / State.zoom);
  const py = Math.floor((my - State.panY) / State.zoom);

  const { width, height } = State.raster;
  if (px < 0 || py < 0 || px >= width || py >= height) {
    DOM.statusCursor.textContent = 'x:— y:—';
    DOM.statusPixelVal.textContent = 'val:—';
    DOM.pixelTooltip.classList.add('hidden');
    return;
  }

  DOM.statusCursor.textContent = `x:${px} y:${py}`;

  // Get pixel RGB from imageData
  const idx = (py * width + px) * 4;
  const d = State.imageData.data;
  const r = d[idx], g = d[idx + 1], b = d[idx + 2];

  // Get raw raster value
  let rawVals = '';
  if (State.raster.data) {
    const vals = State.raster.data.map((band, i) => {
      const v = band[py * width + px];
      return typeof v === 'number' ? v.toFixed(1) : '—';
    });
    rawVals = `raw:[${vals.slice(0, 3).join(',')}]`;
  }

  DOM.statusPixelVal.textContent = `rgb(${r},${g},${b}) ${rawVals}`;

  // Tooltip
  const tt = DOM.pixelTooltip;
  tt.textContent = `(${px}, ${py})  rgb(${r},${g},${b})${rawVals ? ' · ' + rawVals : ''}`;
  tt.style.left = `${e.clientX + 14}px`;
  tt.style.top = `${e.clientY - 28}px`;
  tt.classList.remove('hidden');
}

DOM.canvasViewport.addEventListener('mouseleave', () => {
  DOM.pixelTooltip.classList.add('hidden');
  DOM.statusCursor.textContent = 'x:— y:—';
  DOM.statusPixelVal.textContent = 'val:—';
});

/* ── Keyboard shortcuts ── */
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  switch (e.key) {
    case '+': case '=': zoom(1.25); break;
    case '-': case '_': zoom(0.8); break;
    case '0': fitToViewport(); break;
    case '1': {
      if (!State.raster) break;
      const vp = DOM.canvasViewport.getBoundingClientRect();
      State.zoom = 1; State.panX = (vp.width - State.raster.width) / 2; State.panY = (vp.height - State.raster.height) / 2; applyTransform();
      break;
    }
    case 'i': case 'I': {
      if (State.raster) { DOM.metaPanel.classList.toggle('hidden'); DOM.metaToggle.classList.toggle('hidden'); }
      break;
    }
  }
});

/* ── Resize ── */
window.addEventListener('resize', () => { if (State.raster) fitToViewport(); });

/* ═══════════════════════════════════════════════════════════
   METADATA PANEL — Structured Output display
   ═══════════════════════════════════════════════════════════ */
function updateMetadataPanel(meta) {
  const items = [
    { key: 'Nombre', val: meta.filename, full: true },
    { key: 'Fuente', val: meta.source },
    { key: 'Fecha', val: meta.date },
    { key: 'Resolución', val: `${meta.width} × ${meta.height} px` },
    { key: 'Bandas', val: meta.bands },
    { key: 'Tipo dato', val: meta.dtype },
    { key: 'Bits/muestra', val: meta.bps ?? '—' },
    { key: 'Nodata', val: meta.nodata },
    { key: 'CRS', val: meta.crs, accent: true },
    { key: 'Res. espacial', val: meta.resolution, accent: true },
    { key: 'Origen X', val: meta.originX },
    { key: 'Origen Y', val: meta.originY },
    { key: 'Compresión', val: meta.compression ?? '—' },
  ];

  if (meta.note) items.push({ key: '⚠ Nota', val: meta.note, full: true });

  DOM.metaGrid.innerHTML = items.map(({ key, val, accent, full }) => `
    <div class="meta-item${full ? ' full-width' : ''}">
      <div class="meta-key">${key}</div>
      <div class="meta-val${accent ? ' accent' : ''}">${val ?? '—'}</div>
    </div>
  `).join('');

  DOM.metaPanel.classList.remove('hidden');
  DOM.metaToggle.classList.add('hidden');
}

function updateBadge(source) {
  const labels = { sentinel: 'Sentinel-2', gmaps: 'Google Maps', custom: 'Archivo local' };
  DOM.badgeDot.className = 'badge-dot ' + source;
  DOM.badgeLabel.textContent = `${labels[source] || source}: ${State.activeFile || ''}`;
}

function updateStatusBar(meta) {
  DOM.statusFilename.textContent = meta.filename;
  DOM.statusSize.textContent = `${meta.width}×${meta.height}`;
  DOM.statusBands.textContent = `${meta.bands} banda(s)`;
  DOM.statusbar.classList.remove('hidden');
}

/* ═══════════════════════════════════════════════════════════
   UI HELPERS
   ═══════════════════════════════════════════════════════════ */
function showLoading(msg, progress) {
  DOM.emptyState.classList.add('hidden');
  DOM.canvas.classList.add('hidden');
  DOM.loadingOverlay.classList.remove('hidden');
  DOM.loadingText.textContent = msg;
  setProgress(progress, msg);
}
function setProgress(pct, msg) {
  DOM.progressBar.style.width = pct + '%';
  if (msg) DOM.loadingText.textContent = msg;
}
function hideLoading() { DOM.loadingOverlay.classList.add('hidden'); }

function showCanvas() {
  DOM.loadingOverlay.classList.add('hidden');
  DOM.emptyState.classList.add('hidden');
  DOM.canvas.classList.remove('hidden');
}

function showError(filename, err) {
  DOM.emptyState.classList.remove('hidden');
  DOM.emptyState.innerHTML = `
    <div class="empty-icon" style="color:var(--accent-rose)">
      <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
        <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
      </svg>
    </div>
    <h2 style="color:var(--accent-rose)">Error al cargar</h2>
    <p><strong>${filename}</strong><br><code style="font-size:11px;color:var(--text-muted)">${err?.message || err}</code></p>
    <p style="font-size:11px;color:var(--text-muted);margin-top:8px">
      Asegúrate de abrir el visualizador a través de un servidor local<br>
      (p.ej. <code>python3 -m http.server 8080</code> desde la carpeta TFM/).
    </p>
  `;
}

/* ── Number formatters ── */
function fmtCoord(n) { return typeof n === 'number' ? n.toFixed(6) : '—'; }
function fmtNum(n) { return typeof n === 'number' ? n.toFixed(4) : '—'; }
function clamp255(v) { return Math.max(0, Math.min(255, Math.round(v))); }

/* ═══════════════════════════════════════════════════════════
   INIT
   ═══════════════════════════════════════════════════════════ */
(function init() {
  renderFileList('sentinel');

  // Show viewer toolbar immediately (always visible)
  // Status bar and metadata only appear after a file is loaded (handled above)
})();
