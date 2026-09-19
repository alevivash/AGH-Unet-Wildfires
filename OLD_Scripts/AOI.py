import ee
import json
import geopandas as gpd

# Ruta local de tu nuevo archivo JSON descargado (la clave de la cuenta de servicio)
ruta_credenciales = "C:/Users/USER/Desktop/Projects/AGH/EMSR892_AOI01_DEL_PRODUCT_areaOfInteresta_v1.json" # <--- CAMBIA ESTO por tu ruta real

# Configura las credenciales usando tu correo de cuenta de servicio
gee_credentials = ee.ServiceAccountCredentials(
    "agh-wildfire-almeria@agh-wildfire.iam.gserviceaccount.com",  # Tu correo de servicio de la imagen
    key_file=ruta_credenciales
)

# Inicializa Earth Engine (Asegúrate de colocar el nombre de proyecto correcto de Google Cloud)
ee.Initialize(credentials=gee_credentials, project='agh-wildfire')

print("✅ Conectado exitosamente a Earth Engine")


"C:/Users/USER/Desktop/Projects/AGH/EMSR892_AOI01_DEL_PRODUCT_areaOfInteresta_v1.json"