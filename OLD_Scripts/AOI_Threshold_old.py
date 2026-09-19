import json
import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import show

# 1. Autenticación con tu llave JSON
ruta_credenciales = r"C:\Users\USER\Desktop\Projects\agh-wildfire-9ac1bade4a91.json" 
gee_credentials = ee.ServiceAccountCredentials(
    "agh-wildfire-almeria@agh-wildfire.iam.gserviceaccount.com", 
    key_file=ruta_credenciales
)
ee.Initialize(credentials=gee_credentials, project='agh-wildfire')
print("✅ Conexión a Earth Engine exitosa.")

# 2. Cargar el área de interés (AOI) usando el archivo GeoJSON correcto
# ---> UBICACIÓN: Aquí debes verificar la ruta de tu archivo GeoJSON
ruta_json_aoi = r"C:\Users\USER\Desktop\Projects\EMSR892-aois.geojson"

with open(ruta_json_aoi, 'r', encoding='utf-8') as f:
    geojson_data = json.load(f)

# Extraer y agrupar todas las geometrías en un FeatureCollection de GEE
features_ee = [ee.Feature(ee.Geometry(feature['geometry'])) for feature in geojson_data['features']]
roi_incendio = ee.FeatureCollection(features_ee)
print("✅ Área de Interés cargada correctamente desde el GeoJSON.")

# 3. Filtrar imágenes Sentinel-2 (L2A con corrección atmosférica)
#se toma fechas de referencia y se arma un mosaico con las mejores imagenes
# ---> UBICACIÓN: Aquí puedes variar las fechas pre y post incendio
fecha_pre_inicio = '2026-06-01'
fecha_pre_fin = '2026-07-08' # Antes del incendio

fecha_post_inicio = '2026-07-20'
fecha_post_fin = '2026-08-31' # Después del incendio

# Función para calcular el NBR
def calcular_nbr(imagen):
    # NBR = (NIR - SWIR) / (NIR + SWIR) -> En Sentinel-2: (B8 - B12) / (B8 + B12)
    nbr = imagen.normalizedDifference(['B8', 'B12']).rename('NBR')
    return imagen.addBands(nbr)

# Obtener colecciones pre y post incendio, calcular NBR y obtener la mediana
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

# Calcular el dNBR (Diferencia del Normalized Burn Ratio)
# dNBR = PreFire_NBR - PostFire_NBR
img_dnbr = img_pre_nbr.subtract(img_post_nbr).rename('dNBR')
print("🔥 Índice dNBR calculado exitosamente.")

# 4. Procesamiento y Exportación para visualizar con herramientas locales
# Exportar la imagen dNBR localmente como GeoTIFF
ruta_imagen_local = "sentinel_dnbr_gallardos.tif"
geemap.ee_export_image(
    img_dnbr,
    filename=ruta_imagen_local,
    scale=20, # Resolución de la banda SWIR (B12)
    region=roi_incendio.geometry().bounds(),
    file_per_band=False
)
print(f"📥 Imagen exportada localmente como: {ruta_imagen_local}")

# 5. Segmentación por Umbral (Thresholding) y Visualización


with rasterio.open(ruta_imagen_local) as src:
    imagen_dnbr = src.read(1) # Leer la banda dNBR extraída de Google Earth Engine

    # --- LÓGICA DE UMBRAL (THRESHOLDING) ---
    # En el índice dNBR, los valores positivos más altos indican mayor severidad de quemadura.
    #el nbr se encuentra aprox entre -0.5 a +1.3
    # low Severity umbral nbr e [0.1,0.22]
    # moderate [0.22, 0.44]
    # severe [0.44, 1]

    
    umbral_dnbr = [-1.0,0.1,0.27,0.66,2.0]

    umbral_dnbr = 0.1

    #color para cada rango
    
    
    # Crea una máscara binaria (1 = área quemada, 0 = área no quemada)
    # astype(np.uint8) convierte los valores booleanos (True/False) en enteros (1/0)
    mascara_quemada = (imagen_dnbr > umbral_dnbr).astype(np.uint8)

    # --- VISUALIZACIÓN COMPARATIVA ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Mapa 1: El dNBR en su escala continua
    img_plot = show(imagen_dnbr, ax=ax1, transform=src.transform, cmap='YlOrRd', title="1. Índice dNBR (Continuo)")
    cbar = plt.colorbar(img_plot.get_images()[0], ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('dNBR')
    ax1.set_xlabel("Longitud")
    ax1.set_ylabel("Latitud")
    
    # Mapa 2: La máscara binaria segmentada
    # Usamos el mapa de color 'Greys' para visualizar el contraste estricto entre blanco y negro
    show(mascara_quemada, ax=ax2, transform=src.transform, cmap='Greys', title=f"2. Máscara Binaria (Umbral > {umbral_dnbr})")
    ax2.set_xlabel("Longitud")
    ax2.set_ylabel("Latitud")
    
    plt.tight_layout()
    plt.show()