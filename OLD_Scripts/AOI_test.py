import geopandas as gpd

# Define la ruta exacta hacia tu archivo JSON del evento observado
ruta_json = "C:\Users\USER\Desktop\Projects\AGHEMSR892_AOI01_DEL_PRODUCT_areaOfInterestA_v1.json" 

# Cargar el archivo; GeoPandas asume el formato GeoJSON internamente
perimetro_incendio = gpd.read_file(ruta_json)

# Inspeccionar la tabla de atributos y el sistema de coordenadas
print(perimetro_incendio.head())
print(perimetro_incendio.crs)