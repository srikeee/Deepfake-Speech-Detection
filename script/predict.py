from pathlib import Path
from final_decision import predict_single_file

BASE_DIR = Path(__file__).parent.parent

if __name__ == "__main__":
    audio_path = input("Enter path to audio file: ").strip()

    result = predict_single_file(
        audio_path=audio_path,
        base_dir=str(BASE_DIR),
        strategy='calibrated'
    )

    print("\nPrediction Complete.")
