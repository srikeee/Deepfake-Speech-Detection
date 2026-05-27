"""
Evaluate all three models and compute accuracies for fusion.
Wav2Vec2 is evaluated on raw audio (augmented wav files).
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import model architectures and datasets
import sys
sys.path.append(str(Path(__file__).parent))

from train_wav2vec import Wav2VecClassifier, Wav2VecDataset
from train_mel_cnn import MelCNN, MelDataset
from train_mfcc import MFCCBiLSTM, MFCCDataset


# ------------------ Generic Evaluation ------------------

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_probs.extend(probs[:, 1].cpu().numpy())  # P(fake)
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    return accuracy, np.array(all_probs), np.array(all_labels)


# ------------------ Main Evaluation ------------------

def evaluate_all_models(base_dir, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    results = {}

    # ====================================================
    # Wav2Vec2 Evaluation (RAW AUDIO)
    # ====================================================
    print("=" * 60)
    print("Evaluating Wav2Vec2 Model")
    print("=" * 60)

    wav2vec_data_dir = os.path.join(base_dir, "data", "processed", "augmented")
    wav2vec_model_path = os.path.join(base_dir, "models", "wav2vec", "best_model.pth")

    if os.path.exists(wav2vec_model_path):
        dataset = Wav2VecDataset(wav2vec_data_dir)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        _, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )

        dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=4, shuffle=False, num_workers=0
        )

        model = Wav2VecClassifier().to(device)
        model.load_state_dict(torch.load(wav2vec_model_path, map_location=device))

        acc, probs, labels = evaluate_model(model, dataloader, device)
        results["wav2vec"] = {
            "accuracy": acc,
            "probabilities": probs.tolist(),
            "labels": labels.tolist()
        }

        print(f"Wav2Vec2 Accuracy: {acc:.2f}%\n")
    else:
        print("Wav2Vec2 model not found. Skipping.\n")
        results["wav2vec"] = None

    # ====================================================
    # Mel CNN Evaluation
    # ====================================================
    print("=" * 60)
    print("Evaluating Mel CNN Model")
    print("=" * 60)

    mel_data_dir = os.path.join(base_dir, "data", "processed", "mel")
    mel_model_path = os.path.join(base_dir, "models", "mel_cnn", "best_model.pth")

    if os.path.exists(mel_model_path):
        dataset = MelDataset(mel_data_dir)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        _, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )

        dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=0
        )

        model = MelCNN().to(device)
        model.load_state_dict(torch.load(mel_model_path, map_location=device))

        acc, probs, labels = evaluate_model(model, dataloader, device)
        results["mel_cnn"] = {
            "accuracy": acc,
            "probabilities": probs.tolist(),
            "labels": labels.tolist()
        }

        print(f"Mel CNN Accuracy: {acc:.2f}%\n")
    else:
        print("Mel CNN model not found. Skipping.\n")
        results["mel_cnn"] = None

    # ====================================================
    # MFCC BiLSTM Evaluation
    # ====================================================
    print("=" * 60)
    print("Evaluating MFCC BiLSTM Model")
    print("=" * 60)

    mfcc_data_dir = os.path.join(base_dir, "data", "processed", "mfcc")
    mfcc_model_path = os.path.join(base_dir, "models", "mfcc", "best_model.pth")

    if os.path.exists(mfcc_model_path):
        dataset = MFCCDataset(mfcc_data_dir)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        _, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )

        dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=0
        )

        model = MFCCBiLSTM().to(device)
        model.load_state_dict(torch.load(mfcc_model_path, map_location=device))

        acc, probs, labels = evaluate_model(model, dataloader, device)
        results["mfcc"] = {
            "accuracy": acc,
            "probabilities": probs.tolist(),
            "labels": labels.tolist()
        }

        print(f"MFCC BiLSTM Accuracy: {acc:.2f}%\n")
    else:
        print("MFCC model not found. Skipping.\n")
        results["mfcc"] = None

    # ====================================================
    # Save summary (accuracies only)
    # ====================================================
    results_path = os.path.join(base_dir, "models", "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {k: None if v is None else {"accuracy": v["accuracy"]} for k, v in results.items()},
            f,
            indent=2
        )

    print("=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    for model_name, result in results.items():
        if result:
            print(f"{model_name.upper()}: {result['accuracy']:.2f}%")
        else:
            print(f"{model_name.upper()}: Not evaluated")

    return results


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    evaluate_all_models(str(BASE_DIR), seed=42)
