import { useState, useEffect } from 'react';
import axios from 'axios';
import { APIProvider } from '@vis.gl/react-google-maps';
import Sidebar from './components/Sidebar';
import MapWrapper from './components/MapWrapper';
import './App.css';

function App() {
  const [apiKey, setApiKey] = useState(null);
  const [error, setError] = useState(null);

  // Map state
  const [mapCenter, setMapCenter] = useState({ lat: 40.4168, lng: -3.7038 }); // Default to Madrid
  const [mapZoom, setMapZoom] = useState(15);
  const [resolution, setResolution] = useState(0); 
  const [roiBounds, setRoiBounds] = useState(null);
  const [roiDimensions, setRoiDimensions] = useState({ pixels: '0x0', meters: '0x0' });
  const [savedImages, setSavedImages] = useState([]);
  const [isDrawingMode, setIsDrawingMode] = useState(false);

  // Pin state
  const [pinLocation, setPinLocation] = useState(null);
  const [showPin, setShowPin] = useState(true);

  // Fetch API key sequence
  useEffect(() => {
    const fetchCredentials = async () => {
      try {
        const res = await axios.get('http://localhost:3001/api/credentials');
        if (res.data.maps_tile_api_key) {
          setApiKey(res.data.maps_tile_api_key);
        } else {
          setError('Map Tile API Key is missing from credentials.');
        }
      } catch (err) {
        console.error(err);
        setError('Failed to connect to local backend to retrieve credentials. Make sure backend is running.');
      }
    };
    fetchCredentials();
  }, []);

  // Fetch Saved Images sequence
  const fetchSavedImages = async () => {
    try {
      const res = await axios.get('http://localhost:3001/api/images');
      setSavedImages(res.data);
    } catch (err) {
      console.error('Error fetching images', err);
    }
  };

  useEffect(() => {
    fetchSavedImages();
  }, []);

  if (error) {
    return (
      <div className="loading-overlay">
        <h2>Configuration Error</h2>
        <p style={{ color: 'var(--danger)', marginTop: '1rem', maxWidth: '600px', textAlign: 'center' }}>{error}</p>
      </div>
    );
  }

  if (!apiKey) {
    return (
      <div className="loading-overlay">
        <div className="spinner"></div>
        <h2>Loading map services...</h2>
      </div>
    );
  }

  return (
    <div className="app-container">
      <Sidebar 
        mapCenter={mapCenter}
        setMapCenter={setMapCenter}
        mapZoom={mapZoom}
        setMapZoom={setMapZoom}
        resolution={resolution}
        roiBounds={roiBounds}
        roiDimensions={roiDimensions}
        savedImages={savedImages}
        refreshImages={fetchSavedImages}
        isDrawingMode={isDrawingMode}
        setIsDrawingMode={setIsDrawingMode}
        pinLocation={pinLocation}
        setPinLocation={setPinLocation}
        showPin={showPin}
        setShowPin={setShowPin}
      />
      <div className="map-container">
        <APIProvider apiKey={apiKey}>
          <MapWrapper 
            mapCenter={mapCenter}
            setMapCenter={setMapCenter}
            mapZoom={mapZoom}
            setMapZoom={setMapZoom}
            setResolution={setResolution}
            roiBounds={roiBounds}
            setRoiBounds={setRoiBounds}
            setRoiDimensions={setRoiDimensions}
            isDrawingMode={isDrawingMode}
            setIsDrawingMode={setIsDrawingMode}
            pinLocation={pinLocation}
            showPin={showPin}
          />
        </APIProvider>
      </div>
    </div>
  );
}

export default App;
