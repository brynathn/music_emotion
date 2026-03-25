import streamlit as st
import pandas as pd
import tempfile
import os
import hashlib
from utils import (
    is_valid_youtube_url,
    download_youtube_audio,
    predict_multi_segment,
    plot_emotion_prediction_animated,
    check_audio_duration,
    extract_basic_audio_features,
    show_melspectrogram_image
)

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Prediksi Emosi Musik", layout="centered")

# =========================
# HELPER
# =========================
def file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

# cache berdasarkan konten file, bukan path
@st.cache_data(show_spinner=False)
def cached_audio_features(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(file_bytes)
        path = tmp.name

    result = extract_basic_audio_features(path)
    os.remove(path)
    return result


@st.cache_data(show_spinner=False)
def cached_prediction(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(file_bytes)
        path = tmp.name

    result = predict_multi_segment(path)
    os.remove(path)
    return result

def plot_mood_gauge(value, title):
    import plotly.graph_objects as go

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black", 'thickness': 0.2},
            'steps': [
                {'range': [0, 25], 'color': "#6495ED"},   # Sad
                {'range': [25, 50], 'color': "#90EE90"},  # Calm
                {'range': [50, 75], 'color': "#FFD700"},  # Happy
                {'range': [75, 100], 'color': "#FF6347"}  # Energetic
            ],
        }
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=0),
    )

    return fig

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["🎧 Prediksi Emosi", "ℹ️ Tentang Aplikasi"])

# =========================
# MAIN PAGE
# =========================
if page == "🎧 Prediksi Emosi":

    st.title("🎵 Prediksi Emosi Musik")
    st.markdown("Upload MP3 atau gunakan link YouTube.")

    input_option = st.radio("Sumber audio:", ("Upload MP3", "Link YouTube"))

    audio_bytes = None
    audio_path = None

    # =========================
    # UPLOAD FILE
    # =========================
    if input_option == "Upload MP3":
        uploaded_file = st.file_uploader("Upload file MP3", type=["mp3"])

        if uploaded_file:
            audio_bytes = uploaded_file.read()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(audio_bytes)
                audio_path = tmp.name

    # =========================
    # YOUTUBE
    # =========================
    else:
        youtube_url = st.text_input("Masukkan link YouTube")

        if youtube_url and is_valid_youtube_url(youtube_url):
            with st.spinner("Downloading audio..."):
                audio_path = download_youtube_audio(youtube_url)

            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

        elif youtube_url:
            st.warning("Link tidak valid")

    # =========================
    # PROCESS
    # =========================
    if audio_path:

        valid, duration = check_audio_duration(audio_path)

        if not valid:
            st.error(f"Audio terlalu pendek ({duration}s). Minimal 30 detik.")
            st.stop()

        st.audio(audio_path)

        # =========================
        # AUDIO INFO
        # =========================
        st.markdown("---")
        st.subheader("🎧 Informasi Audio")

        with st.spinner("Extracting features..."):
            features = cached_audio_features(audio_bytes)

        col1, col2, col3 = st.columns(3)
        col1.metric("Durasi", f"{features['Durasi (detik)']}s")
        col2.metric("Tempo", f"{features['Tempo (BPM)']} BPM")
        col3.metric("Loudness", f"{features['Loudness (RMS)']}")

        # =========================
        # MEL SPEC
        # =========================
        st.markdown("---")
        st.subheader("🌈 Mel-Spectrogram")

        fig = show_melspectrogram_image(audio_path)
        st.pyplot(fig)

        # =========================
        # PREDICTION
        # =========================
        st.markdown("---")
        st.subheader("🧠 Prediksi Emosi")

        with st.spinner("Predicting..."):
            prediction = cached_prediction(audio_bytes)

        fig = plot_emotion_prediction_animated(prediction)
        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # RESULT + SMART EXPLANATION
        # =========================
        emotion_labels = {
            "Q1": "Happy / Excited",
            "Q2": "Angry / Nervous",
            "Q3": "Sad / Bored",
            "Q4": "Calm / Relaxed"
        }

        emotion_explanations = {
            "Happy / Excited": "🎉 Energi tinggi, ceria, penuh semangat, dan uplifting.",
            "Angry / Nervous": "🔥 Intens, tegang, agresif, atau penuh tekanan emosional.",
            "Sad / Bored": "😔 Melankolis, lambat, reflektif, dan emosional.",
            "Calm / Relaxed": "🧘 Tenang, damai, stabil, dan menenangkan."
        }

        # urutkan dari terbesar
        sorted_pred = sorted(prediction.items(), key=lambda x: x[1], reverse=True)

        top1_key, top1_val = sorted_pred[0]
        top2_key, top2_val = sorted_pred[1]

        top1 = emotion_labels[top1_key]
        top2 = emotion_labels[top2_key]

        top1_pct = top1_val * 100
        top2_pct = top2_val * 100

        st.markdown(f"### 🎯 Emosi Dominan: **{top1} ({top1_pct:.2f}%)**")

        # =========================
        # ADAPTIVE EXPLANATION
        # =========================

        if top1_pct > 70:
            # sangat dominan
            st.markdown(f"""
            Musik ini **sangat kuat merepresentasikan emosi {top1}**.

            {emotion_explanations[top1]}

            👉 Artinya, karakter musik ini cenderung konsisten dan jelas secara emosional,
            tanpa banyak campuran emosi lain.
            """)

        elif abs(top1_pct - top2_pct) < 10:
            # 50-50 atau hampir seimbang
            st.markdown(f"""
            Musik ini memiliki **dua emosi yang seimbang**:

            - **{top1} ({top1_pct:.1f}%)**
            - **{top2} ({top2_pct:.1f}%)**

            💡 Artinya:
            Musik ini berada di **zona transisi emosi**, misalnya:
            - Lagu yang awalnya tenang lalu naik energi
            - Atau lagu bahagia tapi ada nuansa mellow

            👉 Kombinasi ini membuat musik terasa **lebih kompleks dan emosional secara dinamis**.
            """)

        else:
            # campuran dominan + secondary
            st.markdown(f"""
            Musik ini didominasi oleh **{top1} ({top1_pct:.1f}%)**  
            dengan nuansa tambahan **{top2} ({top2_pct:.1f}%)**

            💡 Interpretasi:
            - **{top1} → karakter utama musik**
            - **{top2} → lapisan emosi tambahan**

            👉 Artinya musik ini tidak satu dimensi, tetapi memiliki:
            - kedalaman emosi
            - variasi dinamika
            """)
        
        # =========================
        # 🎭 MUSIC PERSONALITY
        # =========================

        st.markdown("### 🎭 Music Personality")

        personality = []

        if top1 in ["Happy / Excited"]:
            personality.append("Energetic ⚡")
            personality.append("Uplifting 🌈")

        if top1 in ["Calm / Relaxed"]:
            personality.append("Peaceful 🌿")
            personality.append("Ambient 🌌")

        if top1 in ["Sad / Bored"]:
            personality.append("Emotional 💔")
            personality.append("Reflective 🪞")

        if top1 in ["Angry / Nervous"]:
            personality.append("Intense 🔥")
            personality.append("Aggressive ⚔️")

        # tambahan dari secondary emotion
        if top2 in ["Calm / Relaxed"]:
            personality.append("Balanced ⚖️")

        if top2 in ["Happy / Excited"]:
            personality.append("Optimistic ☀️")

        st.markdown(" | ".join(personality))


        # =========================
        # 🎯 COCOK UNTUK AKTIVITAS
        # =========================

        st.markdown("### 🎯 Cocok untuk Aktivitas")

        activities = []

        if "Calm / Relaxed" in [top1, top2]:
            activities += ["Belajar 📚", "Tidur 😴", "Meditasi 🧘"]

        if "Happy / Excited" in [top1, top2]:
            activities += ["Olahraga 🏃", "Gaming 🎮", "Road trip 🚗"]

        if "Sad / Bored" in [top1, top2]:
            activities += ["Refleksi diri 🌙", "Menulis ✍️", "Santai sendiri ☕"]

        if "Angry / Nervous" in [top1, top2]:
            activities += ["Fokus kerja 💻", "Gym 💪", "Release emosi 🎧"]

        # hapus duplikat
        activities = list(dict.fromkeys(activities))

        st.markdown(" | ".join(activities))


        # =========================
        # 🎚️ MOOD METER (VALENCE - AROUSAL)
        # =========================

        st.markdown("### 🎚️ Mood Meter")

        # mapping sederhana
        valence_map = {
            "Happy / Excited": 0.9,
            "Calm / Relaxed": 0.7,
            "Sad / Bored": 0.3,
            "Angry / Nervous": 0.4
        }

        arousal_map = {
            "Happy / Excited": 0.9,
            "Angry / Nervous": 0.85,
            "Calm / Relaxed": 0.3,
            "Sad / Bored": 0.2
        }

        # weighted average
        valence = sum(valence_map[emotion_labels[k]] * v for k, v in prediction.items())
        arousal = sum(arousal_map[emotion_labels[k]] * v for k, v in prediction.items())

        valence_pct = valence * 100
        arousal_pct = arousal * 100

        # tampilkan gauge
        col1, col2 = st.columns(2)

        with col1:
            fig_val = plot_mood_gauge(valence_pct, "😊 Valence (Positivity)")
            st.plotly_chart(fig_val, use_container_width=True)

        with col2:
            fig_aro = plot_mood_gauge(arousal_pct, "⚡ Arousal (Energy)")
            st.plotly_chart(fig_aro, use_container_width=True)

        # interpretasi mood meter
        if valence > 0.7 and arousal > 0.7:
            mood_desc = "🔥 Energetic & Happy (Party / Workout vibe)"
        elif valence > 0.6 and arousal < 0.4:
            mood_desc = "🌿 Calm & Positive (Relax / Study vibe)"
        elif valence < 0.4 and arousal < 0.4:
            mood_desc = "😔 Low Energy & Sad (Melancholic vibe)"
        elif valence < 0.5 and arousal > 0.7:
            mood_desc = "⚡ Tense & Intense (Angry / Focus vibe)"
        else:
            mood_desc = "🎭 Mixed Emotional State (Complex mood)"

        st.markdown(f"**Mood Insight:** {mood_desc}")

        # =========================
        # FULL DISTRIBUTION
        # =========================
        df = pd.DataFrame.from_dict(prediction, orient='index', columns=['Probabilitas'])
        df.index = df.index.map(emotion_labels)

        st.markdown("### 📊 Distribusi Emosi Lengkap")
        st.dataframe(df.sort_values(by="Probabilitas", ascending=False)
                    .style.format({"Probabilitas": "{:.2%}"}))

        # =========================
        # FOOTER (UPGRADED)
        # =========================
        st.markdown("---")
        st.markdown(f"""
        <div style='text-align: center; font-size: 14px; color: gray;'>

        🧠 <strong>Model Machine Learning by Bryan Nathaniel</strong><br>
        Universitas Bunda Mulia<br><br>

        📚 Dataset: <a href="https://cvml.unige.ch/databases/DEAM/" target="_blank">DEAM</a><br>
        🎧 Emosi dihitung berdasarkan <em>valence</em> & <em>arousal</em><br>

        ⚙️ Model: CNN + LSTM (Music Emotion Recognition)

        </div>
        """, unsafe_allow_html=True)