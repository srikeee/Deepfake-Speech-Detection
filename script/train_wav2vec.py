"""
Train Wav2Vec2 model for deepfake detection.
Partial fine-tuning (upper layers only) with balanced dataset.
"""

import random
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model
from pathlib import Path
from tqdm import tqdm


# ------------------ Utilities ------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------ Dataset ------------------

class Wav2VecDataset(Dataset):
    """
    Balanced dataset with raw wav loading.
    Uses 2.5s audio segments.
    """

    def __init__(self, data_dir, target_sr=16000, max_duration=2.5):
        self.target_sr = target_sr
        self.max_len = int(target_sr * max_duration)

        real_samples, fake_samples = [], []

        for label_idx, label in enumerate(["real", "fake"]):
            label_dir = Path(data_dir) / label
            if not label_dir.exists():
                continue

            for wav_path in label_dir.glob("*.wav"):
                entry = {"path": wav_path, "label": label_idx}
                if label == "real":
                    real_samples.append(entry)
                else:
                    fake_samples.append(entry)

        if len(real_samples) == 0 or len(fake_samples) == 0:
            raise RuntimeError("Missing real or fake samples for wav2vec training.")

        # Balance dataset
        random.shuffle(fake_samples)
        fake_samples = fake_samples[:len(real_samples)]

        self.samples = real_samples + fake_samples
        random.shuffle(self.samples)

        print(f"[Wav2Vec] Balanced dataset size: {len(self.samples)} "
              f"(real={len(real_samples)}, fake={len(fake_samples)})")

    def __len__(self):
        return len(self.samples)

    def _pad_or_truncate(self, waveform):
        length = waveform.shape[-1]
        if length > self.max_len:
            waveform = waveform[:, :self.max_len]
        elif length < self.max_len:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_len - length))
        return waveform

    def __getitem__(self, idx):
        sample = self.samples[idx]
        waveform, sr = torchaudio.load(sample["path"])

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.target_sr:
            waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)

        waveform = self._pad_or_truncate(waveform)
        return waveform.squeeze(0), torch.tensor(sample["label"], dtype=torch.long)


# ------------------ Model ------------------

class Wav2VecClassifier(nn.Module):
    """Partial fine-tuning: upper transformer layers + classifier."""

    def __init__(self, pretrained="facebook/wav2vec2-base", num_classes=2):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained)

        # Freeze feature extractor
        for p in self.wav2vec2.feature_extractor.parameters():
            p.requires_grad = False

        # Freeze lower transformer layers
        for layer in self.wav2vec2.encoder.layers[:6]:
            for p in layer.parameters():
                p.requires_grad = False

        hidden = self.wav2vec2.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        out = self.wav2vec2(x).last_hidden_state
        pooled = out.mean(dim=1)
        return self.classifier(pooled)


# ------------------ Training ------------------

def train_wav2vec(data_dir, model_dir, epochs=3, batch_size=4, lr=1e-4, seed=42):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = Wav2VecDataset(data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = Wav2VecClassifier().to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        correct = total = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(model(x), dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = torch.argmax(model(x), dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Train={train_acc:.2f}%  Val={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Path(model_dir) / "best_model.pth")

    print(f"Best Wav2Vec Val Accuracy: {best_val_acc:.2f}%")
    return best_val_acc
