"""
Visualization of fused deepfake scores and calibrated threshold.
Reads from improved_fusion_results.json (best strategy only).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def visualize_fusion_scores(base_dir):
    base_dir = Path(base_dir)

    fusion_path = base_dir / "models" / "improved_fusion_results.json"
    if not fusion_path.exists():
        print("❌ improved_fusion_results.json not found. Run --fusion first.")
        return

    with open(fusion_path, "r") as f:
        fusion_data = json.load(f)

    # ✅ NEW CORRECT LOCATION
    best = fusion_data.get("best_strategy_results")
    if best is None:
        print("❌ best_strategy_results missing in fusion file.")
        return

    samples = best.get("sample_results")
    threshold = best.get("calibrated_threshold")

    if samples is None:
        print("❌ sample_results missing inside best_strategy_results.")
        return

    scores = np.array([s["final_score"] for s in samples])
    labels = np.array([1 if s["final_label"] == "FAKE" else 0 for s in samples])

    real_scores = scores[labels == 0]
    fake_scores = scores[labels == 1]

    plt.figure(figsize=(10, 6))

    plt.hist(real_scores, bins=50, alpha=0.7, label="REAL", color="blue", density=True)
    plt.hist(fake_scores, bins=50, alpha=0.7, label="FAKE", color="red", density=True)

    plt.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Calibrated Threshold = {threshold:.3f}",
    )

    plt.xlabel("Fused Fake Probability Score")
    plt.ylabel("Density")
    plt.title("Fused Score Distribution (Best Strategy)")
    plt.legend()
    plt.grid(alpha=0.3)

    output_path = base_dir / "models" / "fusion_score_distribution.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"✓ Visualization saved to: {output_path}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    visualize_fusion_scores(BASE_DIR)
