"""
Extract features from augmented audio for three parallel model paths:
1. Wav2Vec2 - raw waveforms
2. Mel Spectrogram - log-mel features
3. MFCC - mel-frequency cepstral coefficients
"""

import os
import numpy as np
import torch
import torchaudio
import librosa
from pathlib import Path
from tqdm import tqdm


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


# def extract_wav2vec_features(audio_dir, output_dir, sr=16000, max_length=160000):
#     """
#     Extract waveform tensors for wav2vec2.
    
#     Args:
#         audio_dir: Path to augmented audio files
#         output_dir: Path to save waveform tensors
#         sr: Sample rate (16kHz for wav2vec2)
#         max_length: Maximum waveform length (10 seconds at 16kHz)
#     """
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
    
#     for label in ['real', 'fake']:
#         label_dir = os.path.join(audio_dir, label)
#         output_label_dir = os.path.join(output_dir, label)
#         Path(output_label_dir).mkdir(exist_ok=True)
        
#         if not os.path.exists(label_dir):
#             continue
        
#         files = [f for f in os.listdir(label_dir) if f.endswith('.wav')]
        
#         for filename in tqdm(files, desc=f"Wav2Vec {label}"):
#             audio_path = os.path.join(label_dir, filename)
            
#             # Load audio
#             waveform, sample_rate = torchaudio.load(audio_path)
            
#             # Resample if needed
#             if sample_rate != sr:
#                 resampler = torchaudio.transforms.Resample(sample_rate, sr)
#                 waveform = resampler(waveform)
            
#             # Convert to mono
#             if waveform.shape[0] > 1:
#                 waveform = torch.mean(waveform, dim=0, keepdim=True)
            
#             # Pad or truncate to max_length
#             if waveform.shape[1] < max_length:
#                 padding = max_length - waveform.shape[1]
#                 waveform = torch.nn.functional.pad(waveform, (0, padding))
#             else:
#                 waveform = waveform[:, :max_length]
            
#             # Save tensor
#             output_path = os.path.join(output_label_dir, filename.replace('.wav', '.pt'))
#             torch.save(waveform, output_path)


def extract_mel_features(audio_dir, output_dir, sr=16000, n_mels=128, n_fft=2048, hop_length=512):
    """
    Extract log-mel spectrograms for CNN.
    
    Args:
        audio_dir: Path to augmented audio files
        output_dir: Path to save mel features
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT size
        hop_length: Hop length for STFT
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for label in ['real', 'fake']:
        label_dir = os.path.join(audio_dir, label)
        output_label_dir = os.path.join(output_dir, label)
        Path(output_label_dir).mkdir(exist_ok=True)
        
        if not os.path.exists(label_dir):
            continue
        
        files = [f for f in os.listdir(label_dir) if f.endswith('.wav')]
        
        for filename in tqdm(files, desc=f"Mel {label}"):
            audio_path = os.path.join(label_dir, filename)
            
            # Load audio
            y, _ = librosa.load(audio_path, sr=sr, mono=True)
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length
            )
            
            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Save as numpy array
            output_path = os.path.join(output_label_dir, filename.replace('.wav', '.npy'))
            np.save(output_path, log_mel_spec)


def extract_mfcc_features(audio_dir, output_dir, sr=16000, n_mfcc=40, n_fft=2048, hop_length=512):
    """
    Extract MFCC features with delta and delta-delta.
    
    Args:
        audio_dir: Path to augmented audio files
        output_dir: Path to save MFCC features
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT size
        hop_length: Hop length for STFT
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for label in ['real', 'fake']:
        label_dir = os.path.join(audio_dir, label)
        output_label_dir = os.path.join(output_dir, label)
        Path(output_label_dir).mkdir(exist_ok=True)
        
        if not os.path.exists(label_dir):
            continue
        
        files = [f for f in os.listdir(label_dir) if f.endswith('.wav')]
        
        for filename in tqdm(files, desc=f"MFCC {label}"):
            audio_path = os.path.join(label_dir, filename)
            
            # Load audio
            y, _ = librosa.load(audio_path, sr=sr, mono=True)
            
            # Compute MFCC
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length
            )
            
            # Compute delta and delta-delta
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # Stack features
            features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
            
            # Save as numpy array
            output_path = os.path.join(output_label_dir, filename.replace('.wav', '.npy'))
            np.save(output_path, features)


def extract_all_features(augmented_dir, processed_dir, sr=16000, seed=42):
    """Extract all feature types from augmented audio."""
    set_seed(seed)
    
    print("\n=== Feature Extraction Started ===\n")
    
    # # Path 1: Wav2Vec2 waveforms
    # print("Extracting Wav2Vec2 waveforms...")
    # wav2vec_dir = os.path.join(processed_dir, 'wav2vec')
    # extract_wav2vec_features(augmented_dir, wav2vec_dir, sr)
    
    # Path 2: Mel spectrograms
    print("\nExtracting Mel spectrograms...")
    mel_dir = os.path.join(processed_dir, 'mel')
    extract_mel_features(augmented_dir, mel_dir, sr)
    
    # Path 3: MFCC features
    print("\nExtracting MFCC features...")
    mfcc_dir = os.path.join(processed_dir, 'mfcc')
    extract_mfcc_features(augmented_dir, mfcc_dir, sr)
    
    print("\n=== Feature Extraction Complete ===\n")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    AUGMENTED_DIR = BASE_DIR / "data" / "processed" / "augmented"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    
    extract_all_features(
        augmented_dir=str(AUGMENTED_DIR),
        processed_dir=str(PROCESSED_DIR),
        sr=16000,
        seed=42
    )