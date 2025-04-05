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
clases = ['auto', 'moto', 'pesado', 'otro']  # Ajusta según corresponda

# --- Descargar modelo si no existe ---
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(model_path):
        with st.spinner("Descargando modelo desde Google Drive..."):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
    # Cargar modelo
    model = tf.keras.models.load_model(model_path)
    return model

model = download_and_load_model()

# --- Interfaz de usuario ---
st.title("Clasificador de Vehículos")
st.write("Sube una imagen para predecir si es un auto, moto, o vehículo pesado")

uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Imagen cargada", use_column_width=True)

        if st.button("Clasificar"):
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            pred = model.predict(img_array)
            pred_idx = np.argmax(pred)
            confianza = round(100 * np.max(pred), 2)

            st.success(f"Predicción: **{clases[pred_idx]}** con {confianza}% de confianza.")
    except Exception as e:
        st.error(f"Ocurrió un error procesando la imagen: {e}")
