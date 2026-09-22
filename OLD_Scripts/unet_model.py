import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Bloque fundamental de U-Net: 
    Consiste en dos capas convolucionales consecutivas (3x3), 
    cada una seguida de una normalización por lotes (BatchNorm2d) y una activación ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        """
        Arquitectura U-Net
        :param in_channels: Número de canales de tu imagen de entrada. 
                            (1 si usas solo el dNBR continuo, 3 si usas RGB).
        :param out_channels: Número de clases a predecir. (1 para una máscara binaria: quemado/no quemado).
        """
        super(UNet, self).__init__()

        # --- ENCODER (Ruta de contracción) ---
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- CUELLO DE BOTELLA (Bottleneck) ---
        self.bottleneck = DoubleConv(512, 1024)

        # --- DECODER (Ruta de expansión con Skip Connections) ---
        # Upsampling (Transposed Convolutions)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512) # 1024 porque concatenaremos (512 del upconv + 512 del enc4)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # --- CAPA DE SALIDA ---
        # Una convolución 1x1 para mapear los 64 canales finales a la clase requerida (quemado o no)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # --- Paso por el Encoder ---
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # --- Cuello de Botella ---
        b = self.bottleneck(p4)

        # --- Paso por el Decoder (con concatenación de Skip Connections) ---
        u4 = self.upconv4(b)
        u4 = torch.cat([u4, e4], dim=1) # Concatenamos la característica espacial e4 guardada del encoder
        d4 = self.dec4(u4)

        u3 = self.upconv3(d4)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.upconv2(d3)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.upconv1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        # Salida final (mascara predictiva de 256x256)
        out = self.out_conv(d1)
        return out

# Instanciar el modelo para verificar que funcione correctamente
if __name__ == "__main__":
    # Creamos un tensor aleatorio simulando un lote (batch) de 4 parches de entrada dNBR (1 canal, 256x256)
    imagen_prueba = torch.randn((4, 1, 256, 256))
    
    # Inicializamos nuestro modelo
    modelo = UNet(in_channels=1, out_channels=1)
    
    # Hacemos una pasada de prueba (Forward pass)
    prediccion = modelo(imagen_prueba)
    
    # Verificamos que la salida tenga exactamente las mismas dimensiones que requerimos
    print(f"Forma de la imagen de entrada: {imagen_prueba.shape}")
    print(f"Forma de la predicción de salida: {prediccion.shape}")