# Google Maps Web Capture

Aplicación web React construida para explorar, delimitar gráficamente (ROI) y descargar recortes de imágenes satelitales de alta resolución interactuando con las APIs de Google Maps.

## Arquitectura

- **Frontend:** Aplicación de página única estilizada con glassmorphism moderno, construida con React, Vite y la librería `@vis.gl/react-google-maps`. Se responsabiliza de renderizar la **Map Tile API (JS API)**, mostrar el nivel de zoom y calcular la resolución dinámicamente, y proveer la herramienta de dibujo (DrawingManager) de la Región de Interés.
- **Backend:** Servidor ligero Node.js/Express (`backend/server.js`). Lee las credenciales y, al recibir una petición de descarga, utiliza la **Maps Static API** para renderizar un recuadro preciso del área solicitada. Procesa la imagen descargada a JPEG o TIFF utilizando `sharp`, e incorpora a su lado un archivo JSON complementario con todos los metadatos necesarios.

## Requisitos Previos

- Node.js (v18 o superior)
- NPM
- Claves API de Google Cloud activas (Maps JavaScript API y Maps Static API habilitadas).

## Configuración y Variables de Entorno

Asegúrate de configurar el archivo de credenciales en `src/data/credentials_google_maps.json` (rutas relativas apuntan un nivel arriba de `Google_Maps_Web`). Debe tener este formato:
```json
{
  "maps_static_api_key": "TU_API_KEY",
  "maps_tile_api_key": "TU_API_KEY"
}
```

## Instalación y Ejecución

Debes arrancar tanto el backend como el frontend.

### 1. Iniciar el Backend
El backend escucha en el puerto `:3001` y se comunicará con la carpeta `src/images/google_maps_web` para guardar tus descargas.
```bash
cd backend
npm install
npm run dev
```

### 2. Iniciar el Frontend
En otra terminal, corre la interfaz web desarrollada en Vite/React:
```bash
cd frontend
npm install
npm run dev
```
Abre tu navegador en `http://localhost:5173`.

## Decisiones de Diseño
- Debido a las restricciones de CORS de la URL en la red de Map Tile API cruda, es necesario usar la Maps JS API como lienzo.
- El servidor Node centraliza la interacción segura con la Static API y procesa el guardado en formato TIFF, el cual incluye los metadatos en un sidecar JSON que facilita mucho la labor en entornos de Big Data frente a embutir metadata en el propio archivo.
