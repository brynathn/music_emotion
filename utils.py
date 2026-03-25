import numpy as np
import plotly.graph_objects as go
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from pathlib import Path
import re
import os
import soundfile as sf
import tempfile
import uuid
import time
from functools import lru_cache
from yt_dlp import YoutubeDL

@lru_cache(maxsize=1)
def load_emotion_model(model_path="best_model.keras"):
    return load_model(model_path)

def is_valid_youtube_url(url):
    pattern = r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+'
    return re.match(pattern, url) is not None

class MyLogger:
    def debug(self, msg): pass
    def warning(self, msg): print("WARNING:", msg)
    def error(self, msg): print("ERROR:", msg)

def download_youtube_audio(youtube_url, output_dir="downloads"):
    if not is_valid_youtube_url(youtube_url):
        raise ValueError("URL YouTube tidak valid")

    os.makedirs(output_dir, exist_ok=True)
    unique_id = str(uuid.uuid4())
    output_base = Path(output_dir) / f"{unique_id}.%(ext)s"

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(Path(output_dir) / f"{unique_id}.%(ext)s"),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False,
        'noplaylist': True,
        'verbose': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        },
        'logger': MyLogger(),
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            result = ydl.download([youtube_url])
            print("Download result:", result)
    except Exception as e:
        raise Exception(f"Gagal mengunduh dari YouTube: {e}")

    final_audio_path = Path(output_dir) / f"{unique_id}.mp3"

    for i in range(5):
        if final_audio_path.exists() and final_audio_path.stat().st_size > 0:
            break
        print(f"Menunggu file tersedia... ({i+1}/5)")
        time.sleep(1)

    if not final_audio_path.exists() or final_audio_path.stat().st_size == 0:
        raise Exception(f"Gagal mengunduh audio. File tidak ditemukan atau kosong: {final_audio_path}")

    return str(final_audio_path.resolve())

def extract_melspectrogram(
    audio_path,
    sr=22050,
    n_mels=128,
    n_fft=2048,
    hop_length=512,
    target_length=256,
    segment_duration=5,
    mean=-33.16832733154297,
    std=23.629505157470703
):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
    except Exception as e:
        raise Exception(f"Gagal memuat audio: {e}")

    segment_samples = sr * segment_duration
    positions = [0.1, 0.3, 0.5, 0.7, 0.9]

    segments = []

    for p in positions:
        if len(y) < segment_samples:
            seg = np.pad(y, (0, segment_samples - len(y)), mode="reflect")
        else:
            start = max(0, int((len(y) - segment_samples) * p))
            seg = y[start:start + segment_samples]

        if np.max(np.abs(seg)) < 1e-5:
            continue

        mel = librosa.feature.melspectrogram(
            y=seg,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        frames = mel_db.shape[1]
        if frames < target_length:
            pad = target_length - frames
            mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode="reflect")
        else:
            mel_db = mel_db[:, :target_length]

        mel_db = (mel_db - mean) / (std + 1e-8)

        mel_db = mel_db[np.newaxis, ..., np.newaxis]
        segments.append(mel_db)

    return segments


def predict_multi_segment(audio_path):
    model = load_emotion_model()

    segments = extract_melspectrogram(audio_path)

    # 🔥 kalau semua segment invalid
    if len(segments) == 0:
        raise Exception("Audio tidak valid untuk diprediksi")

    # 🔥 batch predict (lebih cepat)
    batch = np.vstack(segments)
    preds = model.predict(batch, verbose=0)

    final_pred = np.mean(preds, axis=0)

    kuadran = ["Q1", "Q2", "Q3", "Q4"]
    return dict(zip(kuadran, final_pred))


def plot_emotion_prediction(prediction_dict):
    emotion_labels = {
        "Q1": "Happy / Excited",
        "Q2": "Angry / Nervous",
        "Q3": "Sad / Bored",
        "Q4": "Calm / Relaxed"
    }

    labels = [emotion_labels[q] for q in prediction_dict.keys()]
    values = [round(pred * 100, 2) for pred in prediction_dict.values()]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, values, color=["#FFD700", "#FF6347", "#6495ED", "#90EE90"])

    ax.set_xlim([0, 100])
    ax.set_xlabel("Persentase (%)")
    ax.set_title("Prediksi Emosi Musik")

    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.2f}%', xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords="offset points", ha='left', va='center')

    return fig

def extract_basic_audio_features(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    loudness = librosa.feature.rms(y=y).mean()
    return {
        "Durasi (detik)": round(duration, 2),
        "Tempo (BPM)": round(float(tempo)),
        "Loudness (RMS)": round(float(loudness), 5)
    }

def show_melspectrogram_image(audio_path, duration=45, sr=22050, n_mels=128, 
                              n_fft=2048, hop_length=512):
    y, sr = librosa.load(audio_path, sr=sr, duration=duration)

    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)), mode='reflect')

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mel_db, sr=sr, hop_length=hop_length,
                                   x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-Spectrogram Lagu')
    return fig

def plot_emotion_prediction_animated(prediction_dict):
    import pandas as pd
    import plotly.graph_objects as go

    emotion_labels = {
        "Q1": "Happy / Excited",
        "Q2": "Angry / Nervous",
        "Q3": "Sad / Bored",
        "Q4": "Calm / Relaxed"
    }

    emotion_colors = {
        "Happy / Excited": "#FFD700",
        "Angry / Nervous": "#FF6347",
        "Sad / Bored": "#6495ED",
        "Calm / Relaxed": "#90EE90"
    }

    data = pd.DataFrame([
        {
            "Emotion": emotion_labels[k],
            "Probability": prediction_dict[k] * 100,
            "Color": emotion_colors[emotion_labels[k]]
        }
        for k in prediction_dict
    ])

    steps = 20
    frames = []

    for step in range(steps + 1):
        progress = step / steps
        frame_data = go.Bar(
            x=data["Probability"] * progress,
            y=data["Emotion"],
            orientation='h',
            marker_color=data["Color"],
            text=(data["Probability"] * progress).round(1).astype(str) + "%",
            textposition='outside'
        )
        frames.append(go.Frame(data=[frame_data]))

    initial_data = go.Bar(
        x=[0] * len(data),
        y=data["Emotion"],
        orientation='h',
        marker_color=data["Color"],
        text=["0%"] * len(data),
        textposition='outside'
    )

    fig = go.Figure(
        data=[initial_data],
        layout=go.Layout(
            title=dict(
                text="Distribusi Emosi Musik",
                x=0.5
            ),
            xaxis=dict(
                title="Presentase",
                range=[0, 105],
                ticksuffix="%",
                showgrid=True
            ),
            yaxis=dict(
                autorange="reversed",
                title="Emosi"
            ),
            margin=dict(l=120, r=40, t=60, b=100),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.5,
                    y=-0.25,
                    xanchor="center",
                    yanchor="top",
                    direction="left",
                    buttons=[dict(
                        label="▶️ Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 100, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                        }]
                    )],
                    font=dict(
                        size=16,
                        color="white"
                    ),
                    bgcolor="#1f77b4",
                    bordercolor="black",
                    borderwidth=1,
                    pad=dict(r=10, t=10, b=10, l=10)
                )
            ]
        ),
        frames=frames
    )

    return fig

def check_audio_duration(audio_path, min_duration=30):
    try:
        duration = librosa.get_duration(filename=audio_path)
        return duration >= min_duration, round(duration, 2)
    except Exception as e:
        return False, 0
