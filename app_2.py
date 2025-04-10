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

# --- Pesos promedio por clase en kilogramos ---
pesos_clase = {
    'Bus': 12000,
    'Car': 1500,
    'Truck': 25000,
    'motorcycle': 250
}

st.title("Clasificador de vehículos y estimación de carga")

# --- Cargar imágenes desde el usuario ---
uploaded_files = st.file_uploader("Sube imágenes de vehículos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    conteo_por_clase = {clase: 0 for clase in clases}
    total_peso_estimado = 0

    for uploaded_file in uploaded_files:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            pred = model.predict(img_array, verbose=0)
            pred_idx = np.argmax(pred)
            clase_predicha = clases[pred_idx]

            conteo_por_clase[clase_predicha] += 1
            total_peso_estimado += pesos_clase[clase_predicha]

            st.image(img, caption=f"Predicción: {clase_predicha}", width=250)
        except Exception as e:
            st.warning(f"Error procesando imagen: {uploaded_file.name}. Detalle: {e}")

    # --- Mostrar resultados ---
    st.subheader("Conteo por clase predicha:")
    for clase, conteo in conteo_por_clase.items():
        st.write(f"{clase}: {conteo} vehículos")

    st.subheader("Peso total estimado:")
    st.write(f"**{total_peso_estimado:,} kg**")
else:
    st.info("Por favor, sube una o más imágenes para iniciar la predicción.")
