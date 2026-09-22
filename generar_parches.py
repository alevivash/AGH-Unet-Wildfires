import numpy as np
import rasterio
import os

# 1. Crear directorios automáticamente
ruta_base = "dataset_unet"
os.makedirs(f"{ruta_base}/imagenes_dnbr", exist_ok=True)
os.makedirs(f"{ruta_base}/mascaras_binarias", exist_ok=True)

TAMANO_PARCHE = 256
ruta_imagen_local = "sentinel_dnbr_gallardos.tif" 

with rasterio.open(ruta_imagen_local) as src:
    imagen_completa = src.read(1) 
    alto, ancho = imagen_completa.shape
    contador = 0
    
    # 2. Recorrer la imagen y recortar bloques exactos
    for y in range(0, alto - TAMANO_PARCHE + 1, TAMANO_PARCHE):
        for x in range(0, ancho - TAMANO_PARCHE + 1, TAMANO_PARCHE):
            
            parche_input = imagen_completa[y:y+TAMANO_PARCHE, x:x+TAMANO_PARCHE]
            
            # Saltar parches vacíos
            if np.isnan(parche_input).all():
                continue
                
            # Generar la máscara binaria (> 0.1 es fuego)
            parche_mask = (parche_input > 0.1).astype(np.uint8)
            
            # Reemplazar valores NaN por 0
            parche_input = np.nan_to_num(parche_input, nan=0.0)
            
            # 3. Guardar los parches en formato .npy
            np.save(f"{ruta_base}/imagenes_dnbr/parche_{contador}.npy", parche_input)
            np.save(f"{ruta_base}/mascaras_binarias/parche_{contador}.npy", parche_mask)
            
            contador += 1

print(f"✅ ¡Listo! Se crearon las carpetas y se guardaron {contador} parches.")