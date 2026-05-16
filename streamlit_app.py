import streamlit as st
import numpy as np
import cv2
import gdown
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

MODEL_PATH = "densenet121_xray.h5"

if not os.path.exists(MODEL_PATH):
    file_id = "1saJIdXwX3loVpCoPoAFq2_k_mxGiP_rs"
    
    with st.spinner("Sedang mengunduh model AI..."):
        gdown.download(id=file_id, output=MODEL_PATH, quiet=False)

@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

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