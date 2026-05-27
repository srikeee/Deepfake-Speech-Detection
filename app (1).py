import streamlit as st
import os
import tempfile
from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from scripts.final_decision import predict_single_file

BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="Deepfake Speech Detector",
    page_icon="🎙️",
    layout="centered"
)

# ------------------------------
# Custom Styling
# ------------------------------
st.markdown("""
<style>
.big-card {
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}
.real-card {
    background-color: #e6f4ea;
    color: #1b5e20;
}
.fake-card {
    background-color: #fdecea;
    color: #b71c1c;
}
.metric-label {
    font-size: 16px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Header
# ------------------------------
st.title(" Deepfake Speech Detection System")
st.markdown("""
Multimodal Detection using:

- Wav2Vec2 Transformer  
- Mel Spectrogram CNN  
- MFCC + BiLSTM  
- Calibrated Threshold Fusion  
""")

st.divider()

# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:

    st.audio(uploaded_file, format="audio/wav")

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # ------------------------------
    # Spectrogram Preview
    # ------------------------------
    st.subheader("File Loaded Successfully!")

    y, sr = librosa.load(temp_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.specshow(log_mel, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title("Log-Mel Spectrogram")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel")
    st.pyplot(fig)

    st.divider()

    if st.button("Run Detection"):

        with st.spinner("Analyzing audio..."):
            result = predict_single_file(
                audio_path=temp_path,
                base_dir=str(BASE_DIR),
                strategy="calibrated"
            )

        label = result["decision_string"].split(" ")[0]
        confidence = result["confidence"]
        fused_score = result["final_score"]
        probs = result.get("per_model_probabilities", {})

        st.divider()

        # ------------------------------
        # Big Prediction Card
        # ------------------------------
        if label == "FAKE":
            st.markdown(
                f'<div class="big-card fake-card">🚨 FAKE SPEECH DETECTED</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="big-card real-card">✅ GENUINE SPEECH</div>',
                unsafe_allow_html=True
            )

        st.divider()

        # ------------------------------
        # Confidence & Score
        # ------------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="metric-label">Fused Score</div>', unsafe_allow_html=True)
            st.metric("Fused Score", f"{fused_score:.4f}", label_visibility="collapsed")

        with col2:
            st.markdown('<div class="metric-label">Confidence</div>', unsafe_allow_html=True)
            st.metric("Confidence", f"{confidence:.4f}", label_visibility="collapsed")

        st.progress(float(confidence))

        st.divider()

        st.markdown("---")
        st.subheader("Explainability Panel")

        probs = result.get("per_model_probabilities", {})
        fused_score = result["final_score"]
        confidence = result["confidence"]


        st.markdown("### 📊 Model Probabilities (P(fake))")

        for model_name, prob in probs.items():
            st.progress(prob)
            st.write(f"**{model_name.upper()}** → {prob:.4f}")

        st.markdown("### Model Agreement Analysis")

        fake_votes = sum(1 for p in probs.values() if p > 0.5)
        real_votes = len(probs) - fake_votes

        if fake_votes == len(probs):
            st.success("All models strongly agree: FAKE")
        elif real_votes == len(probs):
            st.success("All models strongly agree: REAL")
        elif abs(fake_votes - real_votes) == 1:
            st.warning("Models partially disagree (2 vs 1 split)")
        else:
            st.error("High disagreement between models")

        st.write(f"Fake Votes: {fake_votes} | Real Votes: {real_votes}")

        st.markdown("### Fusion Decision Logic")

        st.write(
            f"""
            • Final Fused Score = {fused_score:.4f}  
            • Decision Threshold ≈ 0.80 (calibrated)  
            • Since fused score {'>' if fused_score > 0.8 else '<'} threshold → **{result['final_label']}**
            """
        )


        st.markdown("### Confidence Interpretation")

        if confidence > 0.85:
            st.success("Very High Confidence Decision")
        elif confidence > 0.65:
            st.info("Moderate Confidence Decision")
        else:
            st.warning("Low Confidence — borderline case")

        st.write(f"Confidence Score: {confidence:.4f}")

else:
    st.info("Upload a .wav file to begin.")