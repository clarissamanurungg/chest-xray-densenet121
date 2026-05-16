import streamlit as st
import numpy as np
import cv2
import os
import urllib.request
import tensorflow as tf

# Ganti URL ini setelah upload .h5 ke GitHub Release baru
MODEL_URL = "https://github.com/clarissamanurungg/chest-xray-densenet121/releases/download/1.2/densenet121_xray.h5"
MODEL_PATH = "densenet121_xray.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Sedang mengunduh model AI dari GitHub... Mohon tunggu sebentar."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_my_model()

labels = [
    "Atelectasis", "Cardiomegaly", "Effusion",
    "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "No Finding"
]

IMG_SIZE = 224

st.title("Chest X-Ray Disease Classification")
st.write("Optimasi Akurasi Klasifikasi Multi-label Penyakit Paru Menggunakan DenseNet121")

uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded X-Ray", use_container_width=True)

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    img_preprocessed = tf.keras.applications.densenet.preprocess_input(img_resized)
    img_input = np.expand_dims(img_preprocessed, axis=0)

    with st.spinner("Analyzing X-Ray Image..."):
        prediction = model.predict(img_input)[0]

    st.subheader("Prediction Confidence")
    for i, prob in enumerate(prediction):
        st.write(f"{labels[i]} : {prob*100:.2f}%")
