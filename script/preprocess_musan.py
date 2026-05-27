"""
MUSAN-based audio augmentation for deepfake detection.
Applies noise, music, and babble at various SNR levels.
"""

import os
import random
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from pathlib import Path


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def load_audio(path, sr=16000):
    """Load audio file and resample to target sample rate."""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def get_random_musan_file(musan_dir, category):
    """Randomly select a MUSAN file from given category (noise/music/speech)."""
    category_path = os.path.join(musan_dir, category)
    if not os.path.exists(category_path):
        return None
    
    files = []
    for root, _, filenames in os.walk(category_path):
        for f in filenames:
            if f.endswith('.wav'):
                files.append(os.path.join(root, f))
    
    return random.choice(files) if files else None


def add_noise(audio, noise_audio, snr_db):
    """Add noise to audio at specified SNR level."""
    # Ensure noise is at least as long as audio
    if len(noise_audio) < len(audio):
        repeats = int(np.ceil(len(audio) / len(noise_audio)))
        noise_audio = np.tile(noise_audio, repeats)
    
    # Randomly crop noise to match audio length
    start_idx = random.randint(0, len(noise_audio) - len(audio))
    noise_audio = noise_audio[start_idx:start_idx + len(audio)]
    
    # Calculate signal and noise power
    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(noise_audio ** 2)
    
    # Calculate required noise scaling factor
    snr_linear = 10 ** (snr_db / 10)
    scale = np.sqrt(signal_power / (noise_power * snr_linear))
    
    # Add scaled noise to signal
    augmented = audio + scale * noise_audio
    
    # Normalize to prevent clipping
    max_val = np.abs(augmented).max()
    if max_val > 1.0:
        augmented = augmented / max_val
    
    return augmented


def augment_audio(audio_path, musan_dir, sr=16000):
    """Apply MUSAN augmentation to a single audio file."""
    audio = load_audio(audio_path, sr)
    
    # Randomly select augmentation type
    aug_type = random.choice(['noise', 'music', 'speech', 'none'])
    
    if aug_type == 'none' or not os.path.exists(musan_dir):
        return audio  # Return original
    
    # Get random MUSAN file
    musan_file = get_random_musan_file(musan_dir, aug_type)
    if musan_file is None:
        return audio
    
    # Load MUSAN audio
    musan_audio = load_audio(musan_file, sr)
    
    # Random SNR between 0 and 15 dB
    snr_db = random.uniform(0, 15)
    
    # Apply augmentation
    augmented = add_noise(audio, musan_audio, snr_db)
    
    return augmented


def preprocess_dataset(raw_dir, musan_dir, output_dir, sr=16000, seed=42):
    """
    Preprocess entire dataset with MUSAN augmentation.
    
    Args:
        raw_dir: Path to data/raw/ containing real/ and fake/ subdirectories
        musan_dir: Path to data/musan/
        output_dir: Path to data/processed/augmented/
        sr: Target sample rate
        seed: Random seed for reproducibility
    """
    set_seed(seed)
    
    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    real_out = os.path.join(output_dir, 'real')
    fake_out = os.path.join(output_dir, 'fake')
    Path(real_out).mkdir(exist_ok=True)
    Path(fake_out).mkdir(exist_ok=True)
    
    print("Starting MUSAN-based preprocessing...")
    
    # Process real audio
    real_dir = os.path.join(raw_dir, 'real')
    if os.path.exists(real_dir):
        real_files = [f for f in os.listdir(real_dir) if f.endswith('.wav')]
        print(f"\nProcessing {len(real_files)} real audio files...")
        
        for filename in tqdm(real_files, desc="Real"):
            input_path = os.path.join(real_dir, filename)
            output_path = os.path.join(real_out, filename)
            
            augmented = augment_audio(input_path, musan_dir, sr)
            sf.write(output_path, augmented, sr)
    
    # Process fake audio
    fake_dir = os.path.join(raw_dir, 'fake')
    if os.path.exists(fake_dir):
        fake_files = [f for f in os.listdir(fake_dir) if f.endswith('.wav')]
        print(f"\nProcessing {len(fake_files)} fake audio files...")
        
        for filename in tqdm(fake_files, desc="Fake"):
            input_path = os.path.join(fake_dir, filename)
            output_path = os.path.join(fake_out, filename)
            
            augmented = augment_audio(input_path, musan_dir, sr)
            sf.write(output_path, augmented, sr)
    
    print(f"\nPreprocessing complete! Augmented files saved to {output_dir}")


if __name__ == "__main__":
    # Configuration
    BASE_DIR = Path(__file__).parent.parent
    RAW_DIR = BASE_DIR / "data" / "raw"
    MUSAN_DIR = BASE_DIR / "data" / "musan"
    OUTPUT_DIR = BASE_DIR / "data" / "processed" / "augmented"
    
    preprocess_dataset(
        raw_dir=str(RAW_DIR),
        musan_dir=str(MUSAN_DIR),
        output_dir=str(OUTPUT_DIR),
        sr=16000,
        seed=42
    )