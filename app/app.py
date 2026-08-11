from __future__ import annotations

from pathlib import Path
import tempfile

import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st

from covid_rars.audio_io import load_audio
from covid_rars.preprocess import preprocess_for_features
from covid_rars.quality import quality_for_file
from covid_rars.spectrograms import log_mel_spectrogram


st.set_page_config(page_title="Respiratory Audio Screening Prototype", layout="wide")
st.title("Respiratory Audio Screening Prototype")
st.warning("Research prototype only. Not a clinical diagnostic tool.")

uploaded = st.file_uploader("Upload cough, breath, or speech audio", type=["wav", "flac", "mp3", "ogg", "webm", "m4a"])
modality = st.selectbox("Modality", ["cough", "breath", "speech"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    y, sr, original_sr = load_audio(tmp_path)
    processed, event_info = preprocess_for_features(y, sr, modality)
    spec = log_mel_spectrogram(processed, sr)[0]
    quality_row = quality_for_file(
        {
            "recording_id": "uploaded",
            "audio_path": tmp_path.as_posix(),
            "modality": modality,
        }
    )

    st.subheader("Audio Quality")
    st.json(
        {
            "original_sample_rate": original_sr,
            "duration_sec": round(float(len(y) / sr), 3),
            "quality_flag": quality_row["quality_flag"],
            "quality_reasons": quality_row["quality_reasons"],
            "event_start_sec": round(float(event_info["event_start_sec"]), 3),
            "event_end_sec": round(float(event_info["event_end_sec"]), 3),
            "segmentation_method": event_info["segmentation_method"],
        }
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Waveform")
        fig, ax = plt.subplots(figsize=(8, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.axvspan(event_info["event_start_sec"], event_info["event_end_sec"], color="red", alpha=0.2)
        ax.set_xlabel("Time (s)")
        st.pyplot(fig)

    with col2:
        st.subheader("Log-Mel Spectrogram")
        fig, ax = plt.subplots(figsize=(8, 3))
        img = librosa.display.specshow(spec, sr=sr, x_axis="time", y_axis="mel", ax=ax)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        st.pyplot(fig)

    st.subheader("Model Output")
    st.info("Model inference will be enabled after trained calibrated models are available.")
