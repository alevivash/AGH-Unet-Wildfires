# Importacion y segmetancion

#IMPORTAR LIBRERIAS

import json
import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import matplotlib.colors as colors
from rasterio.plot import show
from rasterio.mask import mask
import geopandas as gpd

# Autentificacion Json
ruta_credenciales = r"C:\Users\USER\Desktop\Projects\agh-wildfire-9ac1bade4a91.json" 
gee_credentials = ee.ServiceAccountCredentials(
    "agh-wildfire-almeria@agh-wildfire.iam.gserviceaccount.com", 
    key_file=ruta_credenciales
)
ee.Initialize(credentials=gee_credentials, project='agh-wildfire')
print("✅ Conexión a Earth Engine exitosa.")

# Cargar el área de interés (AOI)
ruta_json_aoi = r"C:\Users\USER\Desktop\Projects\EMSR892-aois.geojson"
with open(ruta_json_aoi, 'r', encoding='utf-8') as f:
    geojson_data = json.load(f)

features_ee = [ee.Feature(ee.Geometry(feature['geometry'])) for feature in geojson_data['features']]
roi_incendio = ee.FeatureCollection(features_ee)

# Filtrar imágenes Sentinel-2
fecha_pre_inicio = '2026-06-01'
fecha_pre_fin = '2026-07-08'
fecha_post_inicio = '2026-07-20'
fecha_post_fin = '2026-08-31'

def calcular_nbr(imagen):
    nbr = imagen.normalizedDifference(['B8', 'B12']).rename('NBR')
    return imagen.addBands(nbr)

col_pre = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(roi_incendio.geometry())
                .filterDate(fecha_pre_inicio, fecha_pre_fin)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                .map(calcular_nbr))
img_pre_nbr = col_pre.median().select('NBR').clip(roi_incendio.geometry())

col_post = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(roi_incendio.geometry())
                .filterDate(fecha_post_inicio, fecha_post_fin)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                .map(calcular_nbr))
img_post_nbr = col_post.median().select('NBR').clip(roi_incendio.geometry())

img_dnbr = img_pre_nbr.subtract(img_post_nbr).rename('dNBR')

# --- NUEVO: Extraer imagen RGB (True Color) Post-Incendio para el fondo ---
img_post_rgb = col_post.median().select(['B4', 'B3', 'B2']).clip(roi_incendio.geometry())

# 4. Exportar el dNBR y el RGB localmente
ruta_imagen_dnbr = "sentinel_dnbr_gallardos.tif"
ruta_imagen_rgb = "sentinel_rgb_gallardos.tif"

print("📥 Descargando imagen dNBR...")
geemap.ee_export_image(img_dnbr, filename=ruta_imagen_dnbr, scale=20, region=roi_incendio.geometry().bounds(), file_per_band=False)

print("📥 Descargando imagen RGB de fondo...")
geemap.ee_export_image(img_post_rgb, filename=ruta_imagen_rgb, scale=20, region=roi_incendio.geometry().bounds(), file_per_band=False)

# Segmentación, Enmascarado y Visualización multi capa
gdf = gpd.read_file(ruta_json_aoi)
geometrias = [geom for geom in gdf.geometry]

fig, ax = plt.subplots(figsize=(10, 8))

# Mostrar el fondo satelital a color real (RGB)
with rasterio.open(ruta_imagen_rgb) as src_rgb:
    rgb = src_rgb.read()
    rgb = rgb / 3000.0 # Normalizamos la reflectancia para Matplotlib
    rgb = np.clip(rgb, 0, 1)
    show(rgb, ax=ax, transform=src_rgb.transform)

# Mostrar la superposición (Overlay) del dNBR
with rasterio.open(ruta_imagen_dnbr) as src_dnbr:
    imagen_enmascarada, transformacion_enmascarada = mask(src_dnbr, geometrias, crop=True, nodata=np.nan)
    imagen_dnbr = imagen_enmascarada[0] 
    
    # Ocultar valores nulos de los bordes
    imagen_dnbr = np.ma.masked_invalid(imagen_dnbr)
    
    # Ocultar la categoría "Sin cambio" (< 0.1) para que se vea el fondo satelital
    imagen_dnbr = np.ma.masked_where(imagen_dnbr < 0.1, imagen_dnbr)
    
    height, width = imagen_dnbr.shape
    new_bounds = rasterio.transform.array_bounds(height, width, transformacion_enmascarada)

    # Actualizamos los colores para que solo contengan las clases afectadas
    umbral_dnbr = [0.1, 0.27, 0.44, 0.66, 2.0]
    colores_severidad = ['#ffff00', '#ff930e', '#d41314', '#0906cc'] # Amarillo, Naranja, Rojo, Rojo Oscuro

    # > 0.1 LOW SEVERITY [0.1 a 0.27)
    # [0.27, 0.44) MODERATE - LOW SEVERITY
    # [0.44, 0.66) MODERATE - HIGH SEVERITY
    # [0.66, 2.0] HIGH SEVERITY
    
    cmap_discreto = colors.ListedColormap(colores_severidad)
    cmap_discreto.set_bad(color='white', alpha=0)
    norm_discreto = colors.BoundaryNorm(umbral_dnbr, cmap_discreto.N)

    # Proyectar la capa de severidad con un poco de transparencia (alpha=0.8)
    img_plot = ax.imshow(
        imagen_dnbr, 
        cmap=cmap_discreto, 
        norm=norm_discreto, 
        extent=[new_bounds[0], new_bounds[2], new_bounds[1], new_bounds[3]],
        alpha=0.85
    )
    
    cbar = fig.colorbar(img_plot, ax=ax, ticks=umbral_dnbr, shrink=0.7)
    cbar.set_label('Rangue dNBR / Severity')
    cbar.ax.set_yticklabels([
    'Low (0.1)', 
    'Moderate-low severity (0.27)', 
    'Moderate-high (0.44)', 
    'High (0.66)',
    'Extreme (2.0)'
    ])
ax.set_title("Severidad del Incendio (Los Gallardos)", fontsize=14, pad=15)
plt.show()