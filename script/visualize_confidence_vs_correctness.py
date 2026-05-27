"""
Confidence vs Correctness Scatter Plot
=====================================

Shows whether fusion confidence correlates with prediction correctness.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ------------------------------------------------------------------
# Fix Python path so imports work when run as a script
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "scripts"))


def visualize_confidence_vs_correctness(base_dir):
    base_dir = Path(base_dir)

    fusion_path = base_dir / "models" / "improved_fusion_results.json"
    if not fusion_path.exists():
        print("❌ improved_fusion_results.json not found. Run --fusion first.")
        return

    with open(fusion_path, "r") as f:
        data = json.load(f)

    best = data.get("best_strategy_results")
    if best is None:
        print("❌ best_strategy_results missing.")
        return

    samples = best.get("sample_results")
    if samples is None:
        print("❌ sample_results missing.")
        return

    confidences = np.array([s["confidence"] for s in samples])
    predictions = np.array([1 if s["final_label"] == "FAKE" else 0 for s in samples])

    # ------------------------------------------------------------------
    # Load ground-truth labels (aligned with fusion samples)
    # ------------------------------------------------------------------
    labels = np.array(best["labels"])

    correctness = (predictions == labels).astype(int)

    # ------------------------------------------------------------------
    # Scatter plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 5))

    plt.scatter(
        confidences[correctness == 1],
        correctness[correctness == 1],
        alpha=0.5,
        label="Correct",
        color="green",
    )

    plt.scatter(
        confidences[correctness == 0],
        correctness[correctness == 0],
        alpha=0.5,
        label="Incorrect",
        color="red",
    )

    plt.yticks([0, 1], ["Incorrect", "Correct"])
    plt.xlabel("Prediction Confidence")
    plt.ylabel("Correctness")
    plt.title("Confidence vs Correctness (Best Fusion Strategy)")
    plt.legend()
    plt.grid(alpha=0.3)

    output_path = base_dir / "models" / "confidence_vs_correctness.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"✓ Plot saved to: {output_path}")

    # ------------------------------------------------------------------
    # Numeric sanity check (very useful for viva)
    # ------------------------------------------------------------------
    high_conf = confidences >= 0.8
    if high_conf.any():
        high_conf_acc = correctness[high_conf].mean() * 100
        print(f"High-confidence accuracy (conf ≥ 0.8): {high_conf_acc:.2f}%")

    low_conf = confidences < 0.6
    if low_conf.any():
        low_conf_acc = correctness[low_conf].mean() * 100
        print(f"Low-confidence accuracy (conf < 0.6): {low_conf_acc:.2f}%")


if __name__ == "__main__":
    visualize_confidence_vs_correctness(BASE_DIR)
