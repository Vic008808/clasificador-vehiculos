import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image
import numpy as np
import os
import gdown

# --- Configuración ---
file_id = "1-4oGsq7zvIvdr_w4qj5QEHmkO4hCR9Iv"
model_path = "modelo_vehiculos.h5"
clases = ['Bus', 'Car', 'Truck', 'motorcycle']

# --- Descargar y cargar modelo ---
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(model_path):
        with st.spinner("Descargando modelo desde Google Drive..."):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)

    if not os.path.exists(model_path):
        raise FileNotFoundError("No se encontró el archivo del modelo.")

    return tf.keras.models.load_model(model_path)

model = download_and_load_model()

# --- Interfaz de usuario ---
# Nuevo código para la predicción de pesos 

# --- Pesos promedio por clase en kilogramos (ajustables) ---
peso_bus = 12000         # Bus típico urbano
peso_car = 1500          # Automóvil promedio
peso_truck = 25000       # Camión de carga mediano a pesado
peso_motorcycle = 250    # Motocicleta turismo

# Ruta base del dataset
pathDataset = "/kaggle/input/vehicle-type-recognition/Dataset"

# Clases disponibles
clases = ["Bus", "Car", "Truck", "motorcycle"]

# Número de imágenes por clase a seleccionar
n = 10

# Lista para almacenar los arrays de imágenes y etiquetas
imagenes_array = []
etiquetas = []

# Cargar imágenes
for clase in clases:
    ruta_clase = os.path.join(pathDataset, clase)
    imagenes = os.listdir(ruta_clase)

    seleccionadas = random.sample(imagenes, n)

    for img_nombre in seleccionadas:
        ruta_imagen = os.path.join(ruta_clase, img_nombre)

        try:
            img = Image.open(ruta_imagen).convert("RGB")
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            imagenes_array.append(img_array)
            etiquetas.append(clase)
        except Exception as e:
            print(f"Error cargando {ruta_imagen}: {e}")

for img_array in imagenes_array:
    pred = model.predict(np.expand_dims(img_array, axis=0), verbose=0)
    pred_idx = np.argmax(pred)
    clase_predicha = clases[pred_idx]
    conteo_por_clase[clase_predicha] += 1
    total_peso_estimado += pesos_clase[clase_predicha]

# --- Resultados ---
print("Conteo por clase predicha:")
for clase, conteo in conteo_por_clase.items():
    print(f"{clase}: {conteo} vehículos")

print(f"\nPeso total estimado en el puente por las imágenes analizadas: {total_peso_estimado:,} kg")


