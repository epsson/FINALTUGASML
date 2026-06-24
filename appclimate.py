import streamlit as st
import joblib

# 1. Konfigurasi Halaman Web Streamlit
st.set_page_config(
    page_title="Deteksi Emosi Perubahan Iklim - KNN",
    page_icon="🎭",
    layout="centered"
)

# 2. Fungsi untuk Memuat Model dan Vectorizer (Aman dengan Cache)
@st.cache_resource
def load_model_dan_vectorizer():
    # Mengambil file .pkl hasil training KNN dari Google Colab
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

# Eksekusi pemuatan model
try:
    model, vectorizer = load_model_dan_vectorizer()
except Exception as e:
    st.error(f"Gagal memuat file acuan AI. Pastikan file 'model.pkl' dan 'vectorizer.pkl' versi KNN sudah diletakkan di folder yang sama dengan file script ini!")
    st.stop()

# 3. Tampilan Antarmuka Pengguna (User Interface)
st.title("🎭 Deteksi Emosi Publik terhadap Perubahan Iklim")
st.markdown("""
Aplikasi web berbasis **Machine Learning (NLP)** ini ditenagai oleh algoritma **K-Nearest Neighbors (KNN)**. 
Sistem akan menganalisis dan mendeteksi emosi atau sentimen di balik teks/tweet mengenai isu **Perubahan Iklim (Climate Change)**.
""")

st.info("💡 **Kategori Emosi yang Tersedia:** Ketakutan (Fear), Kemarahan (Anger), Optimisme (Optimism), atau Netral (Neutral).")

# Kotak Input Teks dari Pengguna
user_input = st.text_area("Masukkan tweet atau kalimat opini (dalam Bahasa Inggris):", 
                          height=150, 
                          placeholder="Contoh: Climate change is a massive threat to our future generations, we must act now...")

# 4. Logika Prediksi KNN Saat Tombol Ditekan
if st.button("Analisis Emosi Teks"):
    if user_input.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong! Silakan masukkan kalimat terlebih dahulu.")
    else:
        with st.spinner("Sedang memproses teks dengan model KNN..."):
            # a. Ubah teks input pengguna menjadi vektor angka
            input_vektor = vectorizer.transform([user_input])
            
            # b. Lakukan prediksi emosi menggunakan kecerdasan KNN
            hasil_prediksi = model.predict(input_vektor)[0]
            
            # c. Tampilkan hasil prediksi dengan visualisasi yang menarik
            st.success("🎉 Teks Berhasil Dianalisis!")
            
            # Modifikasi tampilan output box berdasarkan hasil emosi agar lebih interaktif
            if "Ketakutan" in hasil_prediksi:
                st.error(f"### Hasil Klasifikasi: **{hasil_prediksi}** 😨")
                st.markdown("*Teks ini mengindikasikan rasa cemas, panik, atau kekhawatiran mendalam (**Climate Anxiety**) terhadap dampak buruk kerusakan lingkungan.*")
            elif "Kemarahan" in hasil_prediksi:
                st.warning(f"### Hasil Klasifikasi: **{hasil_prediksi}** 🤬")
                st.markdown("*Teks ini mengindikasikan kemarahan, skeptisisme, atau ketidakpuasan terhadap kebijakan, pajak, atau propaganda isu iklim.*")
            elif "Optimisme" in hasil_prediksi:
                st.info(f"### Hasil Klasifikasi: **{hasil_prediksi}** 🌱")
                st.markdown("*Teks ini memuat harapan, solusi teknologi hijau, aksi nyata, atau pandangan positif terhadap pemulihan bumi.*")
            else:
                st.success(f"### Hasil Klasifikasi: **{hasil_prediksi}** 😐")
                st.markdown("*Teks berupa penyampaian berita objektif atau fakta umum tanpa adanya muatan emosi personal yang kuat.*")

st.markdown("---")
st.caption("Proyek UAS Machine Learning | Klasifikasi Teks NLP Berbasis KNN | Deployment via Streamlit Community Cloud")