"""
final_decision.py  — Improved Fusion Entry Point
==================================================

Drop-in replacement for your existing final_decision.py.
Integrates three improved fusion strategies from fusion_strategies.py.

USAGE (from main.py):
    from scripts.final_decision import run_final_decision
    run_final_decision(base_dir)

USAGE (standalone):
    python scripts/final_decision.py
"""

import os
import sys
import json
from unittest import result
import numpy as np
from pathlib import Path
import requests

# ------------------------------
# HuggingFace Model URLs
# ------------------------------
HF_URLS = {
    "wav2vec": "https://huggingface.co/Shyazam7/deepfake-speech-detection-models/resolve/main/wav2vec_best_model.pth",
    "mel_cnn": "https://huggingface.co/Shyazam7/deepfake-speech-detection-models/resolve/main/mel_cnn_best_model.pth",
    "mfcc": "https://huggingface.co/Shyazam7/deepfake-speech-detection-models/resolve/main/mfcc_best_model.pth",
}

def download_if_missing(url: str, save_path: Path):
    """
    Download model from HuggingFace if it does not exist.
    Uses streaming to support large files.
    """
    if save_path.exists():
        return

    print(f"\n⬇ Downloading model from HuggingFace...")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print("✅ Download complete.")

# Add scripts to path so fusion_strategies can be imported
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from fusion_strategies import (
    run_all_strategies,
    class_precision_weighted_fusion,
    calibrated_threshold_fusion,
    disagreement_dampened_fusion,
    find_optimal_threshold,
    _validate_results,
    _get_arrays,
)


def run_final_decision(base_dir: str, strategy: str = 'all'):
    """
    Load evaluation results, run improved fusion, and report final decisions.

    Args:
        base_dir:  project root (deepfake_audio_project/)
        strategy:  one of 'class_precision', 'calibrated', 'disagreement', 'all'
                   'all' runs all three and prints comparison (recommended)
    """
    base_dir = Path(base_dir)

    # ------------------------------------------------------------------ #
    # 1. Load results — re-run evaluate_models to get probabilities
    # ------------------------------------------------------------------ #
    print("\nLoading model predictions...")
    from scripts.evaluate_models import evaluate_all_models
    results = evaluate_all_models(str(base_dir), seed=42)

    # Verify at least one model has probabilities
    _validate_results(results)

    # ------------------------------------------------------------------ #
    # 2. Run fusion strategies
    # ------------------------------------------------------------------ #
    if strategy == 'all':
        all_strategy_results = run_all_strategies(results, verbose=True)

        # Pick the best strategy by balanced accuracy (real_recall + fake_recall)
        def balanced(s):
            return (s['real_recall'] + s['fake_recall']) / 2.0

        best_key = max(all_strategy_results, key=lambda k: balanced(all_strategy_results[k]))
        best = all_strategy_results[best_key]

        print(f"\n  → Best strategy by balanced accuracy: {best_key.upper()}")
        print(f"     Fusion accuracy: {best['fusion_accuracy']:.1f}%")
        print(f"     Real recall:     {best['real_recall']:.3f}")
        print(f"     Fake recall:     {best['fake_recall']:.3f}")

    elif strategy == 'class_precision':
        best = class_precision_weighted_fusion(results, threshold=0.5)
        all_strategy_results = {'class_precision_weighted': best}

    elif strategy == 'calibrated':
        best = calibrated_threshold_fusion(results)
        all_strategy_results = {'calibrated_threshold': best}

    elif strategy == 'disagreement':
        best = disagreement_dampened_fusion(results, threshold=0.5)
        all_strategy_results = {'disagreement_dampened': best}

    else:
        raise ValueError(f"Unknown strategy: {strategy}. "
                         "Choose: 'all', 'class_precision', 'calibrated', 'disagreement'")

    # ------------------------------------------------------------------ #
    # 3. Example per-sample decision output
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("  EXAMPLE SAMPLE DECISIONS (first 5 samples)")
    print("=" * 70)

    # Get sample results from the best strategy or first available
    if 'sample_results' in best:
        sample_results = best['sample_results']
        available = _validate_results(results)
        ground_truth_labels = results[available[0]]['labels']

        for idx in range(min(5, len(sample_results))):
            r = sample_results[idx]
            gt = "FAKE" if ground_truth_labels[idx] == 1 else "REAL"
            correct = "✓" if r['final_label'] == gt else "✗"
            print(f"  Sample {idx:>3}: {r['decision_string']:<30} "
                  f"GT: {gt}  {correct}")
    # ------------------------------------------------------------------ #
    # 4. Save results to JSON (keep sample_results for BEST strategy only)
    # ------------------------------------------------------------------ #
    models_dir = base_dir / 'models'
    models_dir.mkdir(exist_ok=True)

    save_data = {}

    for key, s in all_strategy_results.items():
        save_data[key] = {
            k: v for k, v in s.items()
            if k != 'sample_results'
        }
    
    # Get labels once (aligned with sample_results)
    available = _validate_results(results)
    labels = results[available[0]]['labels'][:len(best['sample_results'])]

    # 🔑 SAVE SAMPLE RESULTS FOR BEST STRATEGY ONLY
    save_data['best_strategy'] = best_key
    save_data['best_strategy_results'] = {
        'strategy': best_key,
        'fusion_accuracy': best['fusion_accuracy'],
        'real_recall': best['real_recall'],
        'fake_recall': best['fake_recall'],
        'calibrated_threshold': best.get('calibrated_threshold'),
        'labels': labels, 
        'sample_results': best.get('sample_results')  # ← THIS FIXES VISUALIZATION
    }

    out_path = models_dir / 'improved_fusion_results.json'
    with open(out_path, 'w') as f:
        json.dump(
            save_data,
            f,
            indent=2,
            default=lambda x: round(float(x), 6)
            if isinstance(x, (float, np.floating)) else x
        )

    print(f"\n  ✓ Fusion results saved to {out_path}")


def predict_single_file(
    audio_path: str,
    base_dir: str,
    strategy: str = 'calibrated',
    threshold: float = None,
) -> dict:
    """
    Run the full pipeline on a NEW audio file to get a REAL/FAKE prediction.

    This is the inference entry point. It:
      1. Extracts features from the audio file
      2. Loads trained model weights
      3. Computes P(fake) from each model
      4. Applies the chosen fusion strategy

    Args:
        audio_path:  path to a .wav file
        base_dir:    project root
        strategy:    fusion strategy to use
        threshold:   decision threshold (if None, auto-calibrate for 'calibrated')

    Returns:
        dict with final_label, final_score, confidence, decision_string,
             per_model_probabilities, diagnostics
    """
    import torch
    import torchaudio
    import librosa
    import numpy as np
    from pathlib import Path

    # Deferred imports to avoid circular dependencies
    from scripts.train_wav2vec import Wav2VecClassifier
    from scripts.train_mel_cnn import MelCNN
    from scripts.train_mfcc import MFCCBiLSTM

    base_dir = Path(base_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SR = 16000
    MAX_WAV_LEN = 160000
    MAX_FRAMES = 256

    # ------------------------------------------------------------------ #
    # Feature extraction from a single file
    # ------------------------------------------------------------------ #
    print(f"\nProcessing: {audio_path}")

    # Load audio
    y, _ = librosa.load(audio_path, sr=SR, mono=True)

    # --- Wav2Vec2 input ---
    waveform = torch.FloatTensor(y).unsqueeze(0)
    if waveform.shape[1] < MAX_WAV_LEN:
        waveform = torch.nn.functional.pad(waveform, (0, MAX_WAV_LEN - waveform.shape[1]))
    else:
        waveform = waveform[:, :MAX_WAV_LEN]
    wav_input = waveform.squeeze(0).to(device)  # shape: (160000,)

    # --- Mel spectrogram input ---
    mel_spec = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128, n_fft=2048, hop_length=512)
    log_mel  = librosa.power_to_db(mel_spec, ref=np.max)
    if log_mel.shape[1] < MAX_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, MAX_FRAMES - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :MAX_FRAMES]
    mel_input = torch.FloatTensor(log_mel).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,128,256)

    # --- MFCC input ---
    mfcc       = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=40, n_fft=2048, hop_length=512)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_d2    = librosa.feature.delta(mfcc, order=2)
    mfcc_feat  = np.concatenate([mfcc, mfcc_delta, mfcc_d2], axis=0).T  # (time, 120)
    if mfcc_feat.shape[0] < MAX_FRAMES:
        mfcc_feat = np.pad(mfcc_feat, ((0, MAX_FRAMES - mfcc_feat.shape[0]), (0, 0)))
    else:
        mfcc_feat = mfcc_feat[:MAX_FRAMES, :]
    mfcc_input = torch.FloatTensor(mfcc_feat).unsqueeze(0).to(device)  # (1, 256, 120)

    # ------------------------------------------------------------------ #
    # Model inference
    # ------------------------------------------------------------------ #
    probs = {}

    # Wav2Vec2
    wav2vec_path = base_dir / 'models' / 'wav2vec' / 'best_model.pth'

    download_if_missing(HF_URLS["wav2vec"], wav2vec_path)

    if wav2vec_path.exists():
        model = Wav2VecClassifier().to(device)
        model.load_state_dict(torch.load(wav2vec_path, map_location=device))
        model.eval()
        with torch.no_grad():
            logits = model(wav_input.unsqueeze(0))
            probs['wav2vec'] = float(torch.softmax(logits, dim=1)[0, 1].cpu())

    # Mel CNN
    mel_path = base_dir / 'models' / 'mel_cnn' / 'best_model.pth'

    download_if_missing(HF_URLS["mel_cnn"], mel_path)

    if mel_path.exists():
        model = MelCNN().to(device)
        model.load_state_dict(torch.load(mel_path, map_location=device))
        model.eval()
        with torch.no_grad():
            logits = model(mel_input)
            probs['mel_cnn'] = float(torch.softmax(logits, dim=1)[0, 1].cpu())

    # MFCC BiLSTM
    mfcc_path = base_dir / 'models' / 'mfcc' / 'best_model.pth'

    download_if_missing(HF_URLS["mfcc"], mfcc_path)

    if mfcc_path.exists():
        model = MFCCBiLSTM().to(device)
        model.load_state_dict(torch.load(mfcc_path, map_location=device))
        model.eval()
        with torch.no_grad():
            logits = model(mfcc_input)
            probs['mfcc'] = float(torch.softmax(logits, dim=1)[0, 1].cpu())

    # ------------------------------------------------------------------ #
    # Fusion using pre-computed validation accuracies
    # ------------------------------------------------------------------ #
    fusion_results_path = base_dir / 'models' / 'improved_fusion_results.json'
    if fusion_results_path.exists():
        with open(fusion_results_path) as f:
            saved = json.load(f)
    else:
        # Fallback: assume 90% accuracy for each model
        saved = {}

    # Build a minimal results dict for single-sample fusion
    accuracy_map = {
        'wav2vec': 85.0,
        'mel_cnn': 96.7,
        'mfcc':    94.2,
    }

    # Try to load saved accuracies
    for s_key in ['class_precision_weighted', 'calibrated_threshold', 'disagreement_dampened']:
        if s_key in saved:
            # These won't have per-model accuracies directly, use original model results
            break

    # Build synthetic single-sample results for fusion
    available_models = list(probs.keys())
    if not available_models:
        raise RuntimeError("No models produced predictions. Check model weights.")

    single_results = {
        k: {
            'accuracy':      accuracy_map.get(k, 90.0),
            'probabilities': [probs[k]],
            'labels':        [0],  # placeholder label (not used for single prediction)
        }
        for k in available_models
    }

    # Apply chosen strategy

    # Load calibrated threshold from saved fusion results
    saved_threshold = None
    if fusion_results_path.exists():
        if 'best_strategy_results' in saved:
            saved_threshold = saved['best_strategy_results'].get('calibrated_threshold')

    if strategy == 'class_precision':
        strategy = 'calibrated'

    if strategy == 'calibrated':
        # Use saved calibrated threshold if available
        cal_thresh = threshold if threshold is not None else saved_threshold
        if cal_thresh is None:
            cal_thresh = 0.5  # fallback only if absolutely nothing available

        result = calibrated_threshold_fusion(
            single_results,
            threshold=cal_thresh,
            sample_idx=0
        )

    elif strategy == 'disagreement':
        result = disagreement_dampened_fusion(
            single_results,
            threshold=threshold or 0.5,
            sample_idx=0
        )

    else:
        result = calibrated_threshold_fusion(
            single_results,
            threshold=saved_threshold or 0.5,
            sample_idx=0
        )


    # ------------------------------------------------------------------ #
    # Print final output
    # ------------------------------------------------------------------ #
    result['per_model_probabilities'] = probs
    result['strategy_used'] = strategy

    print("\n" + "=" * 50)
    print(f"  FINAL DECISION: {result['decision_string']}")
    print(f"  Fused score:    {result['final_score']:.4f}")
    print("=" * 50)

    return result


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    run_final_decision(str(BASE_DIR), strategy='all')