"""
Train CNN on mel spectrograms for deepfake detection.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MelDataset(Dataset):
    """Dataset for mel spectrogram features."""
    
    def __init__(self, data_dir, max_time_steps=256):
        self.samples = []
        self.max_time_steps = max_time_steps
        
        for label_idx, label in enumerate(['real', 'fake']):
            label_dir = os.path.join(data_dir, label)
            if not os.path.exists(label_dir):
                continue
            
            files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
            for filename in files:
                self.samples.append({
                    'path': os.path.join(label_dir, filename),
                    'label': label_idx
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        mel_spec = np.load(sample['path'])
        
        # Pad or truncate time dimension
        if mel_spec.shape[1] < self.max_time_steps:
            pad_width = self.max_time_steps - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec = mel_spec[:, :self.max_time_steps]
        
        # Add channel dimension and convert to tensor
        mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0)
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return mel_spec, label


class MelCNN(nn.Module):
    """CNN architecture for mel spectrogram classification."""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


def train_mel_cnn(data_dir, model_dir, epochs=15, batch_size=32, lr=1e-3, seed=42):
    """Train mel spectrogram CNN."""
    set_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = MelDataset(data_dir)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize model
    print("Initializing model...")
    model = MelCNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for mel_specs, labels in pbar:
            mel_specs, labels = mel_specs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(mel_specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for mel_specs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                mel_specs, labels = mel_specs.to(device), labels.to(device)
                
                outputs = model(mel_specs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))
            print(f"  → Best model saved (Val Acc: {val_acc:.2f}%)")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    
    return best_val_acc


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "processed" / "mel"
    MODEL_DIR = BASE_DIR / "models" / "mel_cnn"
    
    train_mel_cnn(
        data_dir=str(DATA_DIR),
        model_dir=str(MODEL_DIR),
        epochs=20,
        batch_size=32,
        lr=1e-3,
        seed=42
    )