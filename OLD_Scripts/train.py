import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Importamos la arquitectura U-Net desde tu script anterior
# (Asegúrate de guardar el código de U-Net en un archivo llamado 'unet_model.py')
from OLD_Scripts.unet_model import UNet

class WildfireDataset(Dataset):
    """
    Clase personalizada de PyTorch para leer tus parches de áreas quemadas.
    """
    def __init__(self, dir_imagenes, dir_mascaras):
        self.dir_imagenes = dir_imagenes
        self.dir_mascaras = dir_mascaras
        # Listamos todos los parches generados en la carpeta
        self.archivos = [f for f in os.listdir(dir_imagenes) if f.endswith('.npy')]

    def __len__(self):
        # Le dice a PyTorch cuántos parches tenemos en total
        return len(self.archivos)

    def __getitem__(self, idx):
        # Esta función carga un solo parche cada vez que PyTorch lo solicita
        nombre_archivo = self.archivos[idx]

        ruta_img = os.path.join(self.dir_imagenes, nombre_archivo)
        ruta_mask = os.path.join(self.dir_mascaras, nombre_archivo)

        # Cargar los arrays usando numpy
        imagen = np.load(ruta_img)
        mascara = np.load(ruta_mask)

        # PyTorch espera que los datos tengan la forma (Canales, Alto, Ancho)
        # Usamos unsqueeze(0) para añadir la dimensión del canal único (1, 256, 256)
        imagen_tensor = torch.from_numpy(imagen).float().unsqueeze(0) 
        mascara_tensor = torch.from_numpy(mascara).float().unsqueeze(0) 

        return imagen_tensor, mascara_tensor

# 1. Instanciar el Dataset apuntando a las carpetas que creaste antes
dataset = WildfireDataset(
    dir_imagenes="dataset_unet/imagenes_dnbr",
    dir_mascaras="dataset_unet/mascaras_binarias"
)

# 2. Crear el DataLoader
# 'batch_size=8' significa que el modelo procesará 8 parches al mismo tiempo
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

print(f"✅ DataLoader listo. Se cargarán {len(dataset)} parches en total.")

# Comprobación rápida para ver si los datos fluyen correctamente
imagenes_batch, mascaras_batch = next(iter(dataloader))
print(f"Formato del lote de imágenes (Inputs): {imagenes_batch.shape}")
print(f"Formato del lote de máscaras (Labels): {mascaras_batch.shape}")