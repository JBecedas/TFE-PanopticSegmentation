import React, { useState } from 'react';
import axios from 'axios';
import { Navigation, MapPin, Crop, Download, Image as ImageIcon, Map as MapIcon, XSquare, Search } from 'lucide-react';
import './Sidebar.css';

export default function Sidebar({
  mapCenter, setMapCenter, mapZoom, setMapZoom, resolution, 
  roiBounds, roiDimensions,
  savedImages, refreshImages,
  isDrawingMode, setIsDrawingMode,
  pinLocation, setPinLocation, showPin, setShowPin
}) {

  const [inputLat, setInputLat] = useState(mapCenter.lat);
  const [inputLng, setInputLng] = useState(mapCenter.lng);
  const [roiName, setRoiName] = useState('');
  const [overlapPercent, setOverlapPercent] = useState(20);
  const [captureFormat, setCaptureFormat] = useState('JPEG');
  const [isSaving, setIsSaving] = useState(false);
  const [useCurrentZoom, setUseCurrentZoom] = useState(false);

  const tileZoomForExport = useCurrentZoom ? Math.round(mapZoom) : 20;

  const estimatedTileCount = (() => {
    if (!roiDimensions) return 0;
    const parts = roiDimensions.pixels.split('x').map(p => Math.abs(parseInt(p)));
    if (parts.length < 2 || isNaN(parts[0]) || isNaN(parts[1])) return 0;
    const zoomFactor = Math.pow(2, tileZoomForExport - mapZoom);
    const width = parts[0] * zoomFactor;
    const height = parts[1] * zoomFactor;
    const subSize = 512;
    const overlapPx = Math.round(subSize * (parseFloat(overlapPercent) / 100));
    const stridePx = Math.max(1, subSize - overlapPx);
    const cols = width > subSize ? Math.ceil((width - subSize) / stridePx) + 1 : 1;
    const rows = height > subSize ? Math.ceil((height - subSize) / stridePx) + 1 : 1;
    return rows * cols;
  })();

  const handleNavigate = () => {
    const lat = parseFloat(inputLat);
    const lng = parseFloat(inputLng);
    if (!isNaN(lat) && !isNaN(lng)) {
      const loc = { lat, lng };
      setMapCenter(loc);
      setPinLocation(loc);
      setShowPin(true);
    }
  };

  const handleTogglePin = () => {
    setShowPin(!showPin);
  };

  const clearPin = () => {
    setPinLocation(null);
  };

  const handleSaveCapture = async () => {
    if (!roiBounds || !roiDimensions) return;
    setIsSaving(true);
    try {
      const centerParams = `${roiBounds.center.lat},${roiBounds.center.lng}`;
      const pxArray = roiDimensions.pixels.split('x').map(p => parseInt(Math.abs(p)));
      const payload = {
        bounds: roiBounds,
        zoom: mapZoom,
        format: captureFormat,
        roiName: roiName,
        overlapPercent: parseFloat(overlapPercent),
        tileZoom: tileZoomForExport,
        metadata: {
          resolution_mpx: resolution
        }
      };
      
      const res = await axios.post('http://localhost:3001/api/capture', payload);
      if (res.data.success) {
        setIsDrawingMode(false); // Disable drawing after successful capture
        refreshImages();         // Refresh sidebar gallery
      }
    } catch (err) {
      console.error('Error saving capture', err);
      alert('Failed to save image. Check console for details.');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="sidebar glass-panel">
      <h1><MapIcon size={24} /> Maps Capture</h1>
      
      {/* MAP INFO */}
      <div className="section">
        <h2><Navigation size={16}/> View Info</h2>
        <div className="info-grid">
          <div className="info-item">
            <span className="info-label">Zoom</span>
            <input 
              type="number" 
              step="any" 
              value={Number(mapZoom).toFixed(1)} 
              onChange={e => {
                const val = parseFloat(e.target.value);
                if (!isNaN(val)) setMapZoom(val);
              }}
              style={{ width: '60px', background: 'rgba(255,255,255,0.1)', color: 'white', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '4px', padding: '2px 4px', textAlign: 'right' }}
            />
          </div>
          <div className="info-item">
            <span className="info-label">Resolution</span>
            <span className="info-val">{resolution.toFixed(2)} m/px</span>
          </div>
          <div className="info-item" style={{ gridColumn: 'span 2' }}>
            <span className="info-label">Current Center</span>
            <span className="info-val" style={{ fontSize: '0.8rem' }}>
              {mapCenter.lat.toFixed(6)}, {mapCenter.lng.toFixed(6)}
            </span>
          </div>
        </div>
      </div>

      {/* NAVIGATION */}
      <div className="section">
        <h2><Search size={16}/> Go To</h2>
        <div className="input-group">
          <div className="input-row">
            <span className="info-label" style={{ width: '40px' }}>LAT</span>
            <input type="number" step="any" value={inputLat} onChange={e => setInputLat(e.target.value)} />
          </div>
          <div className="input-row">
            <span className="info-label" style={{ width: '40px' }}>LNG</span>
            <input type="number" step="any" value={inputLng} onChange={e => setInputLng(e.target.value)} />
          </div>
          <button className="btn btn-primary" onClick={handleNavigate}>
            Find & Pin
          </button>
          
          {pinLocation && (
            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
               <button className="btn" onClick={handleTogglePin} style={{ flex: 1 }}>
                 {showPin ? 'Hide Pin' : 'Show Pin'}
               </button>
               <button className="btn" onClick={clearPin} style={{ background: 'rgba(239, 68, 68, 0.2)' }}>
                 <XSquare size={16} />
               </button>
            </div>
          )}
        </div>
      </div>

      {/* ROI TOOL */}
      <div className="section">
        <h2><Crop size={16}/> Image Capture ROI</h2>
        <button 
          className={`btn ${isDrawingMode ? 'btn-active' : ''}`}
          onClick={() => setIsDrawingMode(!isDrawingMode)}
        >
          {isDrawingMode ? 'Cancel Selection' : 'Draw Rectangle'}
        </button>
        
        {roiBounds && (
          <div className="roi-stats">
            <div className="roi-val">Size: {roiDimensions.pixels} px</div>
            <div className="roi-val" style={{ color: '#6ee7b7' }}>Real: {roiDimensions.meters}</div>
            
            <div className="save-controls" style={{ flexDirection: 'column', gap: '8px' }}>
              <input
                type="text"
                placeholder="ROI Name (e.g. area_1)"
                value={roiName}
                onChange={(e) => setRoiName(e.target.value)}
                style={{ width: '100%', padding: '6px', borderRadius: '4px', border: '1px solid rgba(255,255,255,0.2)', background: 'rgba(255,255,255,0.1)', color: 'white' }}
              />
              
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span className="info-label" style={{ flex: 1 }}>Overlap %</span>
                <input
                  type="number"
                  min="0"
                  max="50"
                  value={overlapPercent}
                  onChange={(e) => setOverlapPercent(e.target.value)}
                  style={{ width: '60px', padding: '4px', borderRadius: '4px', border: '1px solid rgba(255,255,255,0.2)', background: 'rgba(255,255,255,0.1)', color: 'white', textAlign: 'right' }}
                />
              </div>
              {parseFloat(overlapPercent) > 30 && (
                <div style={{ color: '#fbbf24', fontSize: '0.75rem', marginTop: '-4px' }}>
                  Warning: High overlap will significantly increase the number of generated tiles.
                </div>
              )}

              {/* Zoom tiles toggle */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span className="info-label">Zoom tiles</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginLeft: 'auto' }}>
                  <span style={{ fontSize: '0.75rem', color: !useCurrentZoom ? 'var(--text-main)' : 'var(--text-muted)', transition: 'color 0.2s' }}>
                    Zoom 20
                  </span>
                  <label className="toggle-switch">
                    <input
                      type="checkbox"
                      checked={useCurrentZoom}
                      onChange={e => setUseCurrentZoom(e.target.checked)}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                  <span style={{ fontSize: '0.75rem', color: useCurrentZoom ? 'var(--text-main)' : 'var(--text-muted)', transition: 'color 0.2s' }}>
                    Zoom {Math.round(mapZoom)}
                  </span>
                </div>
              </div>

              {/* Tile count estimate — only when a roiName is set */}
              {roiName && (
                <div style={{ fontSize: '0.75rem', color: estimatedTileCount > 100 ? '#fbbf24' : '#6ee7b7' }}>
                  ~{estimatedTileCount} tiles estimados
                  {estimatedTileCount > 100 && ' — Advertencia: número elevado de tiles.'}
                </div>
              )}

              <div style={{ display: 'flex', gap: '8px', width: '100%' }}>
                <select 
                  className="capture-format"
                  value={captureFormat} 
                  onChange={e => setCaptureFormat(e.target.value)}
                  style={{ flex: 1 }}
                >
                  <option value="JPEG">JPEG</option>
                  <option value="TIFF">TIFF (Geo-ready)</option>
                </select>
                <button 
                  className="btn btn-primary" 
                  onClick={handleSaveCapture}
                  disabled={isSaving}
                  style={{ flex: 1 }}
                >
                  <Download size={16} /> {isSaving ? 'Processing...' : 'Export Image'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* GALLERY */}
      <div className="section" style={{ flex: 1, overflowY: 'auto' }}>
        <h2><ImageIcon size={16}/> Saved Captures</h2>
        <div className="gallery-list">
          {savedImages.length === 0 ? (
            <span className="info-label">No images saved yet.</span>
          ) : (
            savedImages.map(img => (
              <div 
                className="gallery-item" 
                key={img.filename}
                onClick={() => window.open(`http://localhost:3001/images/${img.filename}`, '_blank')}
                title="Click to view image"
              >
                <div className="gallery-item-info">
                  <span className="gallery-item-name">{img.name}</span>
                  <span className="gallery-item-meta">{new Date(img.date).toLocaleString()} • {img.format}</span>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

    </div>
  );
}
