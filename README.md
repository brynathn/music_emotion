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


## 📊 Result
<img width="1496" height="906" alt="image" src="https://github.com/user-attachments/assets/8c2d8d91-33c2-452e-8499-8b425a9641ae" />

<img width="1496" height="641" alt="image" src="https://github.com/user-attachments/assets/5673e21c-57c8-4cab-b322-4d92a8fe923f" />

<img width="1495" height="816" alt="image" src="https://github.com/user-attachments/assets/7c1c4e83-5521-4371-9ca6-e6aab8dd72e1" />

<img width="1492" height="739" alt="image" src="https://github.com/user-attachments/assets/196575ae-f9a0-42f2-a7d9-c743cbd1c0ef" />

<img width="1483" height="858" alt="image" src="https://github.com/user-attachments/assets/764f064d-b02d-4e9b-8492-c0550eebd7fb" />


👨‍💻 Author
Bryan Nathaniel
🎓 Universitas Bunda Mulia
📧 Email: bryan.nathaniel73@gmail.com
🌐 GitHub: https://github.com/brynathn
