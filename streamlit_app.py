import streamlit as st
import numpy as np
import cv2
import os
import urllib.request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

MODEL_PATH = "densenet121_xray.h5"
MODEL_URL = "https://github.com/clarissamanurungg/chest-xray-densenet121/releases/download/1.1/densenet121_xray.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Sedang mengunduh model AI dari GitHub... Mohon tunggu sebentar."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH, compile=False)

model = load_my_model()

# ======================
# LABELS
# ======================

labels = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "No Finding"
]

IMG_SIZE = 224

# ======================
# TITLE
# ======================

st.title("Chest X-Ray Disease Classification")

st.write(
    "Optimasi Akurasi Klasifikasi Multi-label "
    "Penyakit Paru Menggunakan DenseNet121"
)

# ======================
# UPLOAD IMAGE
# ======================

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg", "jpeg", "png"]
)

# ======================
# PREDICTION
# ======================

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, 1)

    st.image(
        img,
        caption="Uploaded X-Ray",
        use_container_width=True
    )

    # ======================
    # PREPROCESSING
    # ======================

    img = cv2.resize(img, (224,224))

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    # ======================
    # PREDICTION
    # ======================

    with st.spinner("Analyzing X-Ray Image..."):

        prediction = model.predict(img)[0]

    st.subheader("Prediction Confidence")

    for i, prob in enumerate(prediction):

        st.write(
            f"{labels[i]} : {prob*100:.2f}%"
        )
