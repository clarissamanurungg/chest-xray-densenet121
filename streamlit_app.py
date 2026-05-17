import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

MODEL_PATH = "best_densenet121_model.h5"

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_my_model()

# PENTING: Urutan label ini HARUS SAMA PERSIS dengan 
# urutan MultiLabelBinarizer.classes_ saat proses training!
labels = [
    'pneumonia', 'bronchitis', 'emphysema', 'edema', 'cardiomegaly', 
    'fibrosis', 'atelectasis', 'effusion', 'normal'
]

IMG_SIZE = 224

# ==========================================
# 2. TAMPILAN ANTARMUKA (UI) WEB
# ==========================================
st.set_page_config(page_title="Chest X-Ray Classifier", page_icon="🫁")
st.title("🫁 Chest X-Ray Disease Classification")
st.write("Optimasi Akurasi Klasifikasi Multi-label Penyakit Paru Menggunakan Arsitektur CNN DenseNet-121")

uploaded_file = st.file_uploader("Silakan unggah citra rontgen dada (Chest X-Ray)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1) # Format warna BGR bawaan OpenCV
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Gambar X-Ray yang diunggah", use_container_width=True)

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    img_preprocessed = img_resized / 255.0
    img_input = np.expand_dims(img_preprocessed, axis=0)

    with st.spinner("Memproses prediksi citra medis..."):
        prediction = model.predict(img_input)[0]

    with col2:
        st.subheader("Hasil Prediksi Multi-label")
        
        results = {}
        for i, prob in enumerate(prediction):
            results[labels[i].capitalize()] = prob * 100

        sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

        detected_diseases = [dis for dis, prob in sorted_results.items() if prob >= 50.0]
        
        if len(detected_diseases) > 0:
            st.warning(f"⚠️ **Potensi Temuan:** {', '.join(detected_diseases)}")
        else:
            st.success("✅ Tidak ditemukan potensi penyakit mayor yang melampaui batas ambang (threshold).")
        
        st.divider()

        for dis, prob in sorted_results.items():
            st.write(f"**{dis}** : {prob:.2f}%")
            st.progress(int(prob) if prob <= 100 else 100)
