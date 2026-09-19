import json
import ee
import geemap
from rasterio.plot import show
import rasterio
import matplotlib.pyplot as plt
import numpy as np

# 1. Autenticación con tu llave JSON
ruta_credenciales = r"C:\Users\USER\Desktop\Projects\agh-wildfire-9ac1bade4a91.json" 
gee_credentials = ee.ServiceAccountCredentials(
    "agh-wildfire-almeria@agh-wildfire.iam.gserviceaccount.com", 
    key_file=ruta_credenciales
)
ee.Initialize(credentials=gee_credentials, project='agh-wildfire')
print("✅ Conexión a Earth Engine exitosa.")

# 2. Cargar el área de interés (AOI) usando el archivo GeoJSON correcto
# Corrección de la ruta del archivo AOI
ruta_json_aoi = r"C:\Users\USER\Desktop\Projects\EMSR892-aois.geojson"

with open(ruta_json_aoi, 'r', encoding='utf-8') as f:
    geojson_data = json.load(f)

# Extraer y agrupar todas las geometrías en un FeatureCollection de GEE
features_ee = [ee.Feature(ee.Geometry(feature['geometry'])) for feature in geojson_data['features']]
roi_incendio = ee.FeatureCollection(features_ee)
print("✅ Área de Interés cargada correctamente desde el GeoJSON.")

# 3. Filtrar imágenes Sentinel-2 (L2A con corrección atmosférica)
fecha_inicio = '2026-06-01'
fecha_fin = '2026-08-31'

coleccion_s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(roi_incendio.geometry())
                .filterDate(fecha_inicio, fecha_fin)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)))

cantidad_imagenes = coleccion_s2.size().getInfo()
print(f"🛰️ Se encontraron {cantidad_imagenes} imágenes Sentinel-2 útiles en tu área y fechas.")

# 4. Procesamiento y Exportación para visualizar con herramientas locales (Rasterio / Matplotlib)
# Seleccionamos la primer imagen disponible de la colección filtrada
imagen_s2 = coleccion_s2.median().clip(roi_incendio.geometry())

# Seleccionamos las bandas RGB estándar de Sentinel-2 para color real: B4 (Rojo), B3 (Verde), B2 (Azul)
bandas_rgb = ['B4', 'B3', 'B2']
imagen_rgb = imagen_s2.select(bandas_rgb)

# Parámetros de visualización (escalado para reflectancia de Sentinel-2)
parametros_vis = {
    'min': 0,
    'max': 3000,
    'bands': ['B4', 'B3', 'B2']
}

# Descargar una pequeña miniatura local procesada en la nube para graficarla rápidamente con Matplotlib
ruta_imagen_local = "sentinel_rgb_gallardos.tif"
geemap.ee_export_image(
    imagen_rgb,
    filename=ruta_imagen_local,
    scale=20,
    region=roi_incendio.geometry().bounds(),
    file_per_band=False
)

print(f" Imagen exportada localmente como: {ruta_imagen_local}")


with rasterio.open(ruta_imagen_local) as src:
    # 1. Leer los datos de la imagen como un array de Numpy
    imagen = src.read()
    
    # 2. Normalizar los valores al rango [0.0, 1.0]
    # Dividimos entre 3000 (el máximo que definimos en los parámetros visuales de GEE)
    imagen_norm = imagen / 3000.0
    
    # 3. Recortar cualquier valor atípico que supere el 1.0 para evitar errores de Matplotlib
    imagen_norm = np.clip(imagen_norm, 0, 1)
    
    # 4. Graficar la imagen normalizada
    fig, ax = plt.subplots(figsize=(8, 8))
    # Pasamos src.transform para no perder las coordenadas geográficas en los ejes X e Y
    show(imagen_norm, ax=ax, transform=src.transform, title="Incendio Los Gallardos - Sentinel-2 (RGB)")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.show()