import streamlit as st
import joblib

# 1. Konfigurasi Halaman Web
st.set_page_config(page_title="Prediksi SDG NLP", page_icon="🌍", layout="centered")

# 2. Memuat Model dan Vectorizer (Otak dan Kamus AI)
# Pastikan file .pkl berada di folder yang sama dengan app.py
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# 3. Tampilan Antarmuka Web
st.title("🌍 Klasifikasi Teks Perubahan Iklim ke Target SDGs")
st.write("Aplikasi Machine Learning ini mengklasifikasikan teks atau tweet terkait perubahan iklim ke dalam Tujuan Pembangunan Berkelanjutan (SDGs).")

# Kotak Input Teks dari Pengguna
user_input = st.text_area("Masukkan teks atau tweet berbahasa Inggris di sini:", height=150)

# 4. Logika Prediksi saat Tombol Ditekan
if st.button("Prediksi SDG"):
    if user_input.strip() == "":
        st.warning("Teks tidak boleh kosong. Silakan masukkan teks terlebih dahulu!")
    else:
        # a. Ubah teks input menjadi angka menggunakan vectorizer
        input_vector = vectorizer.transform([user_input])
        
        # b. Lakukan prediksi menggunakan model
        prediksi = model.predict(input_vector)[0]
        
        # c. Tampilkan Hasil
        st.success("Teks berhasil dianalisis!")
        st.markdown(f"### Kategori SDG: **{prediksi}**")
        
st.markdown("---")
st.caption("Proyek UAS Machine Learning | Deployment menggunakan Streamlit")