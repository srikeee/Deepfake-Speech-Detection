# Deepfake Audio Detection Pipeline

End-to-end system for detecting deepfake speech using multi-modal fusion of Wav2Vec2, Mel-CNN, and MFCC-BiLSTM models.

## Architecture

```
Input Audio → MUSAN Augmentation → Feature Extraction (3 paths)
                                   ├── Wav2Vec2 (waveform)
                                   ├── Mel-CNN (spectrogram)
                                   └── MFCC-BiLSTM (cepstral)
                                         ↓
                                   Model Training
                                         ↓
                                   Accuracy-Weighted Fusion
                                         ↓
                                   REAL / FAKE Decision
```

## Directory Structure

```
deepfake_audio_project/
├── data/
│   ├── raw/real/                # Original real audio (.wav)
│   ├── raw/fake/                # Original fake audio (.wav)
│   ├── musan/                   # MUSAN dataset
│   └── processed/               # Processed features
├── models/                      # Trained model weights
├── scripts/                     # Pipeline modules
└── main.py                      # Main orchestrator
```

## Setup

```bash
pip install -r requirements.txt
```

**Note:** Download MUSAN dataset from https://www.openslr.org/17/ and extract to `data/musan/`

## Usage

### Complete Pipeline

```bash
python main.py --all
```

### Individual Steps

```bash
# 1. Preprocessing
python main.py --preprocess

# 2. Feature extraction
python main.py --extract

# 3. Training (all models)
python main.py --train

# 4. Evaluation
python main.py --evaluate

# 5. Fusion
python main.py --fusion
```

### Custom Training

```bash
# Train specific models
python main.py --train --train-wav2vec --train-mel

# Adjust epochs
python main.py --train --epochs-wav2vec 15 --epochs-mel 25
```

## Models

1. **Wav2Vec2** - Pretrained transformer fine-tuned on waveforms
2. **Mel-CNN** - 4-layer CNN on log-mel spectrograms
3. **MFCC-BiLSTM** - Bidirectional LSTM on MFCC+Δ+ΔΔ features

## Fusion Method

Accuracy-weighted decision:

```
final_score = Σ(accuracy_i × probability_i) / Σ(accuracy_i)
prediction = FAKE if final_score ≥ 0.5 else REAL
```

## Results

Check `models/fusion_results.json` for:

- Individual model accuracies
- Fusion weights
- Final accuracy

## Requirements

- Python 3.8+
- PyTorch 2.0+
- GPU recommended (CUDA)

## Citation

Based on MUSAN augmentation and multi-modal fusion approaches from deepfake detection literature.
