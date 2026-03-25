<<<<<<< HEAD
🎵 Prediksi Emosi Musik Menggunakan CNN + LSTM
Aplikasi berbasis Streamlit untuk memprediksi emosi pada musik menggunakan model CNN + LSTM, berdasarkan representasi Mel-Spectrogram dari audio.

📌 Fitur Utama
🎧 Input file audio MP3 atau link YouTube

🎼 Konversi audio menjadi Mel-Spectrogram

🤖 Prediksi emosi musik berdasarkan kuadran Arousal-Valence

📊 Visualisasi hasil prediksi emosi dalam bentuk grafik

📂 Struktur Proyek
prediksi_emosi/
├── venv/                   # Virtual environment (tidak perlu diunggah ke GitHub)
├── streamlit_app.py        # Script utama Streamlit
├── utils.py                # Fungsi bantu: ekstraksi audio, prediksi, dll
├── best_model.h5           # Model CNN + LSTM hasil pelatihan
├── requirements.txt        # Daftar dependensi
└── README.md               # Dokumentasi proyek

🚀 Cara Menjalankan Aplikasi
1. Clone Repo dan Masuk ke Folder Proyek
git clone https://github.com/namakamu/prediksi_emosi.git
cd prediksi_emosi

2. Buat dan Aktifkan Virtual Environment
# Buat venv
python -m venv venv

# Aktifkan (Windows)
venv\Scripts\activate

# Aktifkan (Mac/Linux)
source venv/bin/activate

3. Install Dependensi
pip install -r requirements.txt

4. Jalankan Aplikasi Streamlit
streamlit run streamlit_app.py

🧠 Tentang Model
Model yang digunakan adalah kombinasi Convolutional Neural Network (CNN) dan Long Short-Term Memory (LSTM) untuk memproses fitur audio berbentuk Mel-Spectrogram. Output model berupa probabilitas dari 4 kuadran emosi:
| Kuadran | Emosi Utama             |
| ------- | ----------------------- |
| Q1      | Excited, Happy, Pleased |
| Q2      | Angry, Nervous, Annoyed |
| Q3      | Sad, Bored, Sleepy      |
| Q4      | Calm, Peaceful, Relaxed |

📚 Dataset
Dataset utama: DEAM (Database for Emotional Analysis in Music)
Sumber: http://cvml.unige.ch/databases/DEAM/

🛠 Teknologi
- Python 3.8+
- Streamlit
- TensorFlow / Keras
- Librosa
- Pytube
- Matplotlib

📧 Kontak
Dibuat oleh Bryan Nathaniel
📬 Email: bryan.nathaniel73@gmail.com
🌐 GitHub: github.com/brynathn
=======
# music_emotion
>>>>>>> ab5b74c425feac110cebde5e52cd1152538023a0
