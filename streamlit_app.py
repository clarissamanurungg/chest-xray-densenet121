import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

MODEL_PATH = "best_densenet121_model.h5"

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_my_model()

# FIX BUG 1: Label disesuaikan dengan training (bukan label NIH CheXNet)
labels = [
    "pneumonia",
    "bronchitis",
    "emphysema",
    "edema",
    "cardiomegaly",
    "fibrosis",
    "atelectasis",
    "effusion",
    "normal"
]

labels_display = {
    "pneumonia":    "Pneumonia",
    "bronchitis":   "Bronchitis",
    "emphysema":    "Emfisema",
    "edema":        "Edema Paru",
    "cardiomegaly": "Kardiomegali",
    "fibrosis":     "Fibrosis",
    "atelectasis":  "Atelektasis",
    "effusion":     "Efusi Pleura",
    "normal":       "Tidak Ada Kelainan (Normal)"
}

IMG_SIZE = 224
THRESHOLD = 0.5  # Threshold untuk multi-label classification

st.set_page_config(
    page_title="Chest X-Ray Disease Classification",
    page_icon="🫁",
    layout="centered"
)

st.title("Chest X-Ray Disease Classification")
st.write("Optimasi Akurasi Klasifikasi Multi-label Penyakit Paru Menggunakan DenseNet-121")

st.divider()

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg", "jpeg", "png"],
    help="Upload gambar X-Ray dada dalam format JPG atau PNG"
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Gagal membaca gambar. Pastikan file tidak rusak dan formatnya JPG/PNG.")
        st.stop()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded X-Ray", use_container_width=True)

    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    img_preprocessed = img_resized / 255.0
    img_input = np.expand_dims(img_preprocessed, axis=0)

    with st.spinner("Menganalisis gambar X-Ray..."):
        prediction = model.predict(img_input)[0]

    st.subheader("Hasil Prediksi")

    detected = [(labels[i], prediction[i]) for i in range(len(labels)) if prediction[i] >= THRESHOLD]

    if detected:
        st.markdown("**Temuan yang Terdeteksi:**")
        for label, prob in sorted(detected, key=lambda x: x[1], reverse=True):
            display_name = labels_display.get(label, label)
            if label == "normal":
                st.success(f"✅ {display_name} — {prob * 100:.1f}%")
            else:
                st.warning(f"⚠️ {display_name} — {prob * 100:.1f}%")
    else:
        top_idx = int(np.argmax(prediction))
        top_label = labels[top_idx]
        top_prob = prediction[top_idx]
        display_name = labels_display.get(top_label, top_label)
        if top_label == "normal":
            st.success(f"✅ {display_name} — {top_prob * 100:.1f}%")
        else:
            st.warning(f"⚠️ {display_name} — {top_prob * 100:.1f}%")

    st.divider()
    st.subheader("Confidence Score Semua Kelas")

    sorted_preds = sorted(
        enumerate(prediction), key=lambda x: x[1], reverse=True
    )
    for idx, prob in sorted_preds:
        label = labels[idx]
        display_name = labels_display.get(label, label)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(float(prob), text=display_name)
        with col2:
            st.write(f"{prob * 100:.2f}%")

    st.divider()
