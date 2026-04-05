from PIL import Image #traiter les images
import numpy as np

image = Image.open(".tmp/thumbnails/_1IBFoaCOmA.jpg")
image.resize((224, 224)).save(".tmp/thumbnails_resized/_1IBFoaCOmA_224.jpg")

new_image = Image.open(".tmp/thumbnails_resized/_1IBFoaCOmA_224.jpg")   
print(f"{new_image.size}")

arr = np.array(new_image) / 255.0
print(f"{arr.shape}")
print(f"Valeurs min/max : {arr.min()} / {arr.max()}")
