"""
Train MFCC-based classifier (BiLSTM) for deepfake detection.
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


class MFCCDataset(Dataset):
    """Dataset for MFCC features."""
    
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
        mfcc = np.load(sample['path'])
        
        # Transpose to (time, features)
        mfcc = mfcc.T
        
        # Pad or truncate time dimension
        if mfcc.shape[0] < self.max_time_steps:
            pad_width = self.max_time_steps - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:self.max_time_steps, :]
        
        mfcc = torch.FloatTensor(mfcc)
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return mfcc, label


class MFCCBiLSTM(nn.Module):
    """BiLSTM architecture for MFCC feature classification."""
    
    def __init__(self, input_size=120, hidden_size=128, num_layers=2, num_classes=2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden states from both directions
        hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        
        # Classification
        output = self.classifier(hidden)
        return output


def train_mfcc(data_dir, model_dir, epochs=20, batch_size=32, lr=1e-3, seed=42):
    """Train MFCC BiLSTM classifier."""
    set_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = MFCCDataset(data_dir)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize model
    print("Initializing model...")
    model = MFCCBiLSTM(input_size=120).to(device)
    
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
        for mfcc, labels in pbar:
            mfcc, labels = mfcc.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(mfcc)
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
            for mfcc, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                mfcc, labels = mfcc.to(device), labels.to(device)
                
                outputs = model(mfcc)
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
    DATA_DIR = BASE_DIR / "data" / "processed" / "mfcc"
    MODEL_DIR = BASE_DIR / "models" / "mfcc"
    
    train_mfcc(
        data_dir=str(DATA_DIR),
        model_dir=str(MODEL_DIR),
        epochs=20,
        batch_size=32,
        lr=1e-3,
        seed=42
    )