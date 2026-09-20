import json
import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import matplotlib.colors as colors
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
    bounds = src.bounds #limites geograficoas


    # --- LÓGICA DE UMBRAL (THRESHOLDING) ---
    # En el índice dNBR, los valores positivos más altos indican mayor severidad de quemadura.
    #el nbr se encuentra aprox entre -0.5 a +1.3
    # low Severity umbral nbr e [0.1,0.22]
    # moderate [0.22, 0.44]
    # severe [0.44, 1]

    
    umbral_dnbr = [-1.0,0.1,0.27,0.44,2.0]

    #color para cada rango
    
    colores_severidad =  ['#55e019', '#ffff00', '#ff930e', '#d41314']
    
   # 3. Crear el mapa de colores discreto (estilo hipsométrico)
    cmap_discreto = colors.ListedColormap(colores_severidad)
    norm_discreto = colors.BoundaryNorm(umbral_dnbr, cmap_discreto.N)

    # 4. Visualización
    fig, ax = plt.subplots(figsize=(10, 8))

    
 # USAR ax.imshow: Mapea correctamente los colores y normalización usando la extensión geográfica
    img_plot = ax.imshow(
        imagen_dnbr, 
        cmap=cmap_discreto, 
        norm=norm_discreto, 
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top]
    )
    
    # Añadir barra de colores de referencia (opcional pero muy recomendada)
    cbar = fig.colorbar(img_plot, ax=ax, ticks=umbral_dnbr, shrink=0.7)
    cbar.set_label('Rango dNBR / Severidad')
    cbar.ax.set_yticklabels(['-1.0', 'Sin cambio (0.1)', 'Leve (0.27)', 'Moderado (0.66)', 'Grave (2.0)'])

    ax.set_title("Niveles de Severidad del Incendio (Los Gallardos)", fontsize=14, pad=15)
    plt.show()

    #cosas que hay que arreglar: ver que son esos puntos del norte
    #hacer que el mapa atras se vea transparente.