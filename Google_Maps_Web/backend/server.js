import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import axios from 'axios';
import sharp from 'sharp';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3001;

// Define absolute paths relative to the current file
const CREDENTIALS_PATH = path.resolve(__dirname, '../../src/data/credentials_google_maps.json');
const IMAGES_DIR = path.resolve(__dirname, '../../src/images/google_maps_web');

app.use(cors());
app.use(express.json());
app.use('/images', express.static(IMAGES_DIR));

// Ensure images directory exists
if (!fs.existsSync(IMAGES_DIR)) {
  fs.mkdirSync(IMAGES_DIR, { recursive: true });
}

// 1. Endpoint to get frontend credentials
app.get('/api/credentials', (req, res) => {
  try {
    const rawData = fs.readFileSync(CREDENTIALS_PATH, 'utf8');
    const credentials = JSON.parse(rawData);
    // Send only the tile api key to frontend for the interactive map
    res.json({
      maps_tile_api_key: credentials.maps_tile_api_key,
    });
  } catch (error) {
    console.error('Error reading credentials:', error);
    res.status(500).json({ error: 'Could not load credentials' });
  }
});

// 2. Endpoint to list saved images
app.get('/api/images', (req, res) => {
  try {
    const files = fs.readdirSync(IMAGES_DIR);
    const imagesList = [];

    files.forEach(file => {
      const ext = path.extname(file).toLowerCase();
      if (ext === '.jpeg' || ext === '.jpg' || ext === '.tiff' || ext === '.tif') {
        const name = path.basename(file, ext);
        // Try to read sidecar JSON for metadata
        const metaPath = path.join(IMAGES_DIR, `${name}.json`);
        let metadata = {};
        if (fs.existsSync(metaPath)) {
          try {
            metadata = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
          } catch (e) {
            console.error(`Error reading metadata for ${file}`, e);
          }
        }

        const stat = fs.statSync(path.join(IMAGES_DIR, file));

        imagesList.push({
          filename: file,
          name: name,
          format: ext.replace('.', '').toUpperCase(),
          date: stat.mtime,
          metadata: metadata
        });
      }
    });

    // Sort by date descending
    imagesList.sort((a, b) => new Date(b.date) - new Date(a.date));
    res.json(imagesList);

  } catch (error) {
    console.error('Error listing images:', error);
    res.status(500).json({ error: 'Could not list images' });
  }
});

// --- Web Mercator Math Utilities ---
function latLngToPixelXY(lat, lng, zoom) {
  const tileSize = 256;
  const n = Math.pow(2, zoom);
  const x = ((lng + 180) / 360) * n * tileSize;
  const latRad = (lat * Math.PI) / 180;
  const y = ((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) * n * tileSize;
  return { x, y };
}

function pixelXYToLatLng(pixelX, pixelY, zoom) {
  const tileSize = 256;
  const n = Math.pow(2, zoom);
  const x = pixelX / tileSize;
  const y = pixelY / tileSize;

  const lng = (x / n) * 360 - 180;
  const nY = Math.PI - 2 * Math.PI * y / n;
  const lat = (180 / Math.PI) * Math.atan(0.5 * (Math.exp(nY) - Math.exp(-nY)));
  return { lat, lng };
}

async function fetchTile(x, y, z) {
  const url = `https://mt1.google.com/vt/lyrs=s&x=${x}&y=${y}&z=${z}`;
  const response = await axios({
    method: 'get',
    url: url,
    responseType: 'arraybuffer'
  });
  return Buffer.from(response.data, 'binary');
}

async function generateImageForBounds(bounds, zoom, outputFormat, outPath) {
  const { north, south, east, west } = bounds;
  
  // Calculate pixel coordinates for the bounding box
  const nwPix = latLngToPixelXY(north, west, zoom);
  const sePix = latLngToPixelXY(south, east, zoom);
  
  // Calculate tile ranges
  const tileSize = 256;
  const startX = Math.floor(nwPix.x / tileSize);
  const endX = Math.floor(sePix.x / tileSize);
  const startY = Math.floor(nwPix.y / tileSize);
  const endY = Math.floor(sePix.y / tileSize);

  // Total canvas dimensions
  const cols = endX - startX + 1;
  const rows = endY - startY + 1;
  const canvasWidth = cols * tileSize;
  const canvasHeight = rows * tileSize;

  // Fetch all tiles
  const tilePromises = [];
  for (let x = startX; x <= endX; x++) {
    for (let y = startY; y <= endY; y++) {
      tilePromises.push(
        fetchTile(x, y, zoom)
          .then(buffer => ({
            input: buffer,
            left: (x - startX) * tileSize,
            top: (y - startY) * tileSize
          }))
          .catch(err => {
            console.error(`Error fetching tile ${x},${y},${zoom}:`, err.message);
            return null;
          })
      );
    }
  }

  const tiles = (await Promise.all(tilePromises)).filter(t => t !== null);

  // Stitch tiles
  const stitchedBuffer = await sharp({
    create: {
      width: canvasWidth,
      height: canvasHeight,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    }
  })
  .composite(tiles)
  .png()
  .toBuffer();

  // Crop to exact ROI
  const left = Math.floor(nwPix.x - startX * tileSize);
  const top = Math.floor(nwPix.y - startY * tileSize);
  const width = Math.floor(sePix.x - nwPix.x);
  const height = Math.floor(sePix.y - nwPix.y);

  // Ensure dimensions are valid (> 0)
  if (width <= 0 || height <= 0) {
    throw new Error('Invalid crop dimensions');
  }

  const croppedBuffer = await sharp(stitchedBuffer)
    .extract({ left, top, width, height })
    [outputFormat.toLowerCase() === 'tiff' ? 'tiff' : 'jpeg']({ quality: 90 })
    .toBuffer();

  fs.writeFileSync(outPath, croppedBuffer);
  return { width, height };
}

// 3. Endpoint to capture and save an image
app.post('/api/capture', async (req, res) => {
  try {
    const {
      bounds,       // { north, south, east, west, center }
      zoom,
      format,       // "JPEG" or "TIFF"
      roiName,
      overlapPercent = 20,
      tileZoom,     // zoom level for sub-tiles (defaults to 20 if not provided)
      metadata    // additional custom metadata
    } = req.body;

    if (!bounds) {
      return res.status(400).json({ error: 'Bounds are required' });
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const baseFilename = roiName ? `${roiName}_main` : `capture_${timestamp}`;
    const fileExt = format.toLowerCase() === 'tiff' ? '.tiff' : '.jpeg';
    
    // Set up directories
    let targetDir = IMAGES_DIR;
    if (roiName) {
      targetDir = path.join(IMAGES_DIR, roiName);
      if (!fs.existsSync(targetDir)) {
        fs.mkdirSync(targetDir, { recursive: true });
      }
    }
    
    const filePath = path.join(targetDir, `${baseFilename}${fileExt}`);
    const metaPath = path.join(targetDir, `${baseFilename}.json`);

    // 1. Generate main image at selected zoom
    const { width, height } = await generateImageForBounds(bounds, zoom, format, filePath);

    const finalMetadata = {
      name: baseFilename,
      roi_name: roiName || null,
      date_created: new Date().toISOString(),
      center: bounds.center,
      bounds: bounds,
      provider: 'mt1.google.com (Direct Tiles)',
      map_type: 'satellite',
      zoom: zoom,
      pixel_size: `${width}x${height}`,
      ...metadata
    };
    fs.writeFileSync(metaPath, JSON.stringify(finalMetadata, null, 2), 'utf8');

    // 2. Generate sub-images at the requested tile zoom if roiName is provided
    let subImagesCount = 0;
    if (roiName) {
      const z20 = tileZoom || 20;
      const nwPix20 = latLngToPixelXY(bounds.north, bounds.west, z20);
      const sePix20 = latLngToPixelXY(bounds.south, bounds.east, z20);
      
      const width20 = sePix20.x - nwPix20.x;
      const height20 = sePix20.y - nwPix20.y;
      
      const subSize = 512; // 512x512 pixels per tile at zoom 20
      const overlapPx = Math.round(subSize * (overlapPercent / 100));
      const stridePx = subSize - overlapPx;
      
      const cols = width20 > subSize ? Math.ceil((width20 - subSize) / stridePx) + 1 : 1;
      const rows = height20 > subSize ? Math.ceil((height20 - subSize) / stridePx) + 1 : 1;
      
      console.log(`Generating grid: tileSize=${subSize}, overlapPercent=${overlapPercent}%, overlapPx=${overlapPx}, stridePx=${stridePx}, rows=${rows}, cols=${cols}`);
      
      const globalMetadataList = [];

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          let subStartX = nwPix20.x + c * stridePx;
          let subStartY = nwPix20.y + r * stridePx;
          
          // Clamp to boundaries if the tile goes out of bounds
          if (subStartX + subSize > sePix20.x) {
            subStartX = Math.max(nwPix20.x, sePix20.x - subSize);
          }
          if (subStartY + subSize > sePix20.y) {
            subStartY = Math.max(nwPix20.y, sePix20.y - subSize);
          }

          const subEndX = subStartX + subSize;
          const subEndY = subStartY + subSize;
          
          const subNwLatLng = pixelXYToLatLng(subStartX, subStartY, z20);
          const subSeLatLng = pixelXYToLatLng(subEndX, subEndY, z20);
          
          const subBounds = {
            north: subNwLatLng.lat,
            west: subNwLatLng.lng,
            south: subSeLatLng.lat,
            east: subSeLatLng.lng,
            center: {
              lat: (subNwLatLng.lat + subSeLatLng.lat) / 2,
              lng: (subNwLatLng.lng + subSeLatLng.lng) / 2
            }
          };
          
          const subFilename = `${roiName}_tile_r${String(r).padStart(2, '0')}_c${String(c).padStart(2, '0')}`;
          const subFilePath = path.join(targetDir, `${subFilename}${fileExt}`);
          const subMetaPath = path.join(targetDir, `${subFilename}.json`);
          
          const subDim = await generateImageForBounds(subBounds, z20, format, subFilePath);
          
          const subMetadata = {
            roiId: roiName,
            tileId: subFilename,
            tileSize: subSize,
            overlapPercent: overlapPercent,
            overlapPx: overlapPx,
            stridePx: stridePx,
            row: r,
            col: c,
            pixelX: subStartX,
            pixelY: subStartY,
            zoom: z20,
            date_created: new Date().toISOString(),
            bounds: subBounds,
            center: subBounds.center,
            pixel_size: `${subDim.width}x${subDim.height}`,
            parent_image: baseFilename
          };
          fs.writeFileSync(subMetaPath, JSON.stringify(subMetadata, null, 2), 'utf8');
          globalMetadataList.push(subMetadata);
          subImagesCount++;
        }
      }
      
      // Save global metadata
      const globalMetaPath = path.join(targetDir, 'metadata.json');
      fs.writeFileSync(globalMetaPath, JSON.stringify({
        roiId: roiName,
        totalTiles: subImagesCount,
        tileSize: subSize,
        overlapPercent: overlapPercent,
        overlapPx: overlapPx,
        stridePx: stridePx,
        rows: rows,
        cols: cols,
        tiles: globalMetadataList
      }, null, 2), 'utf8');
    }

    res.json({
      success: true,
      message: `Captured main image and ${subImagesCount} sub-images`,
      file: `${baseFilename}${fileExt}`,
      metadata: finalMetadata
    });

  } catch (error) {
    console.error('Capture error:', error);
    res.status(500).json({ error: error.message || 'Failed to capture image' });
  }
});

app.listen(PORT, () => {
  console.log(`Backend server running on http://localhost:${PORT}`);
});
