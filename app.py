import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Charger le modèle
model = keras.models.load_model("models/mobilenet_v2.keras")

st.title("PYTHIA — YouTube Virality Predictor")

# Inputs
title = st.text_input("Video title")
duration = st.number_input("Duration (seconds)", min_value=0)
hour = st.slider("Publishing hour", 0, 23, 12)
thumbnail = st.file_uploader("Thumbnail", type=["jpg", "png"])

# Prédiction
if st.button("Predict") and thumbnail is not None:
    image = Image.open(thumbnail).resize((224, 224))
    arr = preprocess_input(np.array(image).astype(np.float32))
    arr = np.expand_dims(arr, axis=0)
    
    prediction = model.predict(arr)[0][0]
    
    if prediction >= 0.5:
        st.success("VIRAL")
    else:
        st.error("NOT VIRAL")