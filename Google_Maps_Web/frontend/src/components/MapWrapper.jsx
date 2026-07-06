import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Map, Marker, useMap, useMapsLibrary } from '@vis.gl/react-google-maps';

export default function MapWrapper({
  mapCenter, setMapCenter,
  mapZoom, setMapZoom,
  setResolution,
  roiBounds, setRoiBounds, setRoiDimensions,
  isDrawingMode, setIsDrawingMode,
  pinLocation, showPin
}) {
  const map = useMap();
  const geometryLibrary = useMapsLibrary('geometry');
  const [rectangleOverlay, setRectangleOverlay] = useState(null);

  // Refs for stable access inside closures without adding to effect deps
  const rectangleOverlayRef = useRef(null);
  const drawingRectRef = useRef(null);

  useEffect(() => { rectangleOverlayRef.current = rectangleOverlay; }, [rectangleOverlay]);

  const updateROIStats = useCallback((bounds, currentZoom) => {
    if (!bounds || !geometryLibrary) return;

    const ne = bounds.getNorthEast();
    const sw = bounds.getSouthWest();
    const nw = new google.maps.LatLng(ne.lat(), sw.lng());

    const widthMeters = geometryLibrary.spherical.computeDistanceBetween(nw, ne);
    const heightMeters = geometryLibrary.spherical.computeDistanceBetween(sw, nw);

    const centerLat = bounds.getCenter().lat();
    const z = currentZoom !== undefined ? currentZoom : (map ? map.getZoom() : 15);
    const res = (156543.03392 * Math.cos(centerLat * Math.PI / 180)) / Math.pow(2, z);

    const widthPx = Math.round(widthMeters / res);
    const heightPx = Math.round(heightMeters / res);

    const formatMeters = (m) => m > 1000 ? `${(m / 1000).toFixed(2)} km` : `${m.toFixed(1)} m`;

    setRoiDimensions({
      pixels: `${widthPx}x${heightPx}`,
      meters: `${formatMeters(widthMeters)} x ${formatMeters(heightMeters)}`
    });
  }, [map, geometryLibrary, setRoiDimensions]);

  // Escape key: cancel any active drawing or clear the selection
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key !== 'Escape') return;
      setIsDrawingMode(false);
      if (drawingRectRef.current) {
        drawingRectRef.current.setMap(null);
        drawingRectRef.current = null;
      }
      if (rectangleOverlayRef.current) {
        rectangleOverlayRef.current.setMap(null);
        setRectangleOverlay(null);
        setRoiBounds(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [setIsDrawingMode, setRoiBounds]);

  // Enable / disable map panning when drawing mode changes
  useEffect(() => {
    if (!map) return;
    if (isDrawingMode) {
      // Clear previous rectangle when starting a new draw
      if (rectangleOverlayRef.current) {
        rectangleOverlayRef.current.setMap(null);
        setRectangleOverlay(null);
        setRoiBounds(null);
      }
      // Lock panning so mouse drag draws instead of panning the map
      map.setOptions({ draggable: false, disableDoubleClickZoom: true });
    } else {
      map.setOptions({ draggable: true, disableDoubleClickZoom: false });
    }
  }, [map, isDrawingMode, setRoiBounds]);

  // Custom rectangle drawing using raw map mouse events (replaces deprecated DrawingManager)
  useEffect(() => {
    if (!map || !isDrawingMode) return;

    let startLatLng = null;

    const onMouseDown = (e) => {
      startLatLng = e.latLng;
      if (drawingRectRef.current) drawingRectRef.current.setMap(null);
      drawingRectRef.current = new google.maps.Rectangle({
        bounds: new google.maps.LatLngBounds(startLatLng, startLatLng),
        fillColor: '#4f46e5',
        fillOpacity: 0.2,
        strokeWeight: 2,
        strokeColor: '#4f46e5',
        editable: false,
        draggable: false,
        clickable: false,
        zIndex: 1,
        map,
      });
    };

    const onMouseMove = (e) => {
      if (!startLatLng || !drawingRectRef.current) return;
      drawingRectRef.current.setBounds(new google.maps.LatLngBounds(
        new google.maps.LatLng(
          Math.min(startLatLng.lat(), e.latLng.lat()),
          Math.min(startLatLng.lng(), e.latLng.lng())
        ),
        new google.maps.LatLng(
          Math.max(startLatLng.lat(), e.latLng.lat()),
          Math.max(startLatLng.lng(), e.latLng.lng())
        )
      ));
    };

    const onMouseUp = () => {
      if (!startLatLng || !drawingRectRef.current) return;

      const rect = drawingRectRef.current;
      const bounds = rect.getBounds();
      const ne = bounds.getNorthEast();
      const sw = bounds.getSouthWest();

      rect.setEditable(true);
      rect.setDraggable(true);
      rect.setOptions({ clickable: true });

      setRectangleOverlay(rect);
      setRoiBounds({
        north: ne.lat(),
        south: sw.lat(),
        east: ne.lng(),
        west: sw.lng(),
        center: { lat: bounds.getCenter().lat(), lng: bounds.getCenter().lng() }
      });
      updateROIStats(bounds);

      // Keep bounds in sync when user edits the rectangle
      rect.addListener('bounds_changed', () => {
        const b = rect.getBounds();
        const bNe = b.getNorthEast();
        const bSw = b.getSouthWest();
        setRoiBounds({
          north: bNe.lat(),
          south: bSw.lat(),
          east: bNe.lng(),
          west: bSw.lng(),
          center: { lat: b.getCenter().lat(), lng: b.getCenter().lng() }
        });
        updateROIStats(b);
      });

      startLatLng = null;
      drawingRectRef.current = null;
      setIsDrawingMode(false);
    };

    const mdL = google.maps.event.addListener(map, 'mousedown', onMouseDown);
    const mmL = google.maps.event.addListener(map, 'mousemove', onMouseMove);
    const muL = google.maps.event.addListener(map, 'mouseup', onMouseUp);

    return () => {
      google.maps.event.removeListener(mdL);
      google.maps.event.removeListener(mmL);
      google.maps.event.removeListener(muL);
      // Remove any unfinished drawing rectangle on cleanup
      if (drawingRectRef.current) {
        drawingRectRef.current.setMap(null);
        drawingRectRef.current = null;
      }
    };
  }, [map, isDrawingMode, updateROIStats, setRoiBounds, setIsDrawingMode]);

  return (
    <div style={{ width: '100%', height: '100%', cursor: isDrawingMode ? 'crosshair' : 'default' }}>
      <Map
        center={mapCenter}
        zoom={mapZoom}
        disableDefaultUI={false}
        mapTypeId={'satellite'}
        onCameraChanged={(ev) => {
          const z = ev.detail.zoom;
          const lat = ev.detail.center.lat;
          setMapCenter(ev.detail.center);
          setMapZoom(z);

          const res = (156543.03392 * Math.cos(lat * Math.PI / 180)) / Math.pow(2, z);
          setResolution(res);

          if (rectangleOverlayRef.current) {
            updateROIStats(rectangleOverlayRef.current.getBounds(), z);
          }
        }}
        style={{ width: '100%', height: '100%' }}
      >
        {pinLocation && showPin && (
          <Marker position={pinLocation} />
        )}
      </Map>
    </div>
  );
}
