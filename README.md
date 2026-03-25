# 🎵 Music Emotion Recognition (CNN + LSTM)

Aplikasi berbasis **Streamlit** untuk memprediksi emosi dari musik menggunakan **Deep Learning (CNN + LSTM)** berdasarkan representasi **Mel-Spectrogram**.

---

## 🚀 Demo Fitur

✨ Aplikasi ini mampu:
- 🎧 Upload file **MP3** atau input **link YouTube**
- 🎼 Mengubah audio menjadi **Mel-Spectrogram**
- 🤖 Memprediksi emosi musik berdasarkan model **CNN + LSTM**
- 📊 Visualisasi distribusi emosi (interactive chart)
- 🎯 Interpretasi emosi **adaptif & human-readable**
- 🎭 Analisis tambahan:
  - Music Personality
  - Rekomendasi aktivitas
  - Mood Meter (Valence & Arousal)

---

## 🧠 Konsep Utama

Model menggunakan pendekatan **Arousal - Valence Model**:

| Kuadran | Emosi |
|--------|------|
| Q1 | Happy / Excited |
| Q2 | Angry / Nervous |
| Q3 | Sad / Bored |
| Q4 | Calm / Relaxed |

---

## 🧬 Arsitektur Model

- **CNN (Convolutional Neural Network)**  
  → Ekstraksi pola dari Mel-Spectrogram  

- **LSTM (Long Short-Term Memory)**  
  → Menangkap dinamika temporal dari audio  

- Output: probabilitas 4 kelas emosi

---

## 📊 Dataset

- **DEAM (Database for Emotional Analysis in Music)**
- 1800+ lagu (45 detik)
- Label berdasarkan:
  - **Valence (positivity)**
  - **Arousal (energy)**

🔗 http://cvml.unige.ch/databases/DEAM/

---

## 🖥️ Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- Librosa
- Plotly
- yt-dlp

---

## ⚙️ Cara Menjalankan

### 1. Clone repo
git clone https://github.com/brynathn/music_emotion.git
cd music_emotion

### 2. Buat virtual environment
Buat virtual environment

### 3. Aktifkan venv
Windows
venv\Scripts\activate

Mac/Linux
source venv/bin/activate

### 4. Install dependencies
pip install -r requirements.txt

### 5. Run app
streamlit run streamlit_app.py

📈 Contoh Insight yang Dihasilkan
🎯 Emosi Dominan
🔥 Kombinasi emosi (misal: Happy + Calm)
🎭 Music Personality (Energetic, Reflective, dll)
🎯 Rekomendasi aktivitas (Gym, Study, Relax)
🎚️ Mood Meter:
    - Valence (positivity)
    - Arousal (energy)
    
⚠️ Catatan
File model (.keras) bisa besar → gunakan Git LFS jika perlu
Audio YouTube membutuhkan koneksi internet
Minimal durasi audio: 30 detik

👨‍💻 Author
Bryan Nathaniel
🎓 Universitas Bunda Mulia
📧 Email: bryan.nathaniel73@gmail.com
🌐 GitHub: https://github.com/brynathn
