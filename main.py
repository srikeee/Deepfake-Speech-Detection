"""
Main pipeline orchestrator for deepfake audio detection.
Runs the complete pipeline from preprocessing to final decision.
"""

import sys
import argparse
from pathlib import Path

# Add scripts to path
sys.path.append(str(Path(__file__).parent / "scripts"))

from scripts.preprocess_musan import preprocess_dataset
from scripts.extract_features import extract_all_features
from scripts.train_wav2vec import train_wav2vec
from scripts.train_mel_cnn import train_mel_cnn
from scripts.train_mfcc import train_mfcc
from scripts.evaluate_models import evaluate_all_models
from scripts.final_decision import run_final_decision


def print_header(text):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def run_pipeline(args):
    """Execute the complete deepfake detection pipeline."""
    BASE_DIR = Path(__file__).parent
    
    print_header("DEEPFAKE AUDIO DETECTION PIPELINE")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Random Seed: {args.seed}\n")
    
    # Step 1: MUSAN Preprocessing
    if args.preprocess:
        print_header("STEP 1: MUSAN-BASED PREPROCESSING")
        preprocess_dataset(
            raw_dir=str(BASE_DIR / "data" / "raw"),
            musan_dir=str(BASE_DIR / "data" / "musan"),
            output_dir=str(BASE_DIR / "data" / "processed" / "augmented"),
            sr=16000,
            seed=args.seed
        )
    
    # Step 2: Feature Extraction
    if args.extract:
        print_header("STEP 2: FEATURE EXTRACTION (3 PARALLEL PATHS)")
        extract_all_features(
            augmented_dir=str(BASE_DIR / "data" / "processed" / "augmented"),
            processed_dir=str(BASE_DIR / "data" / "processed"),
            sr=16000,
            seed=args.seed
        )
    
    # Step 3: Model Training
    if args.train:
        print_header("STEP 3: MODEL TRAINING")
        
        # Train Wav2Vec2
        if args.train_wav2vec:
            print("\n--- Training Wav2Vec2 Model ---")
            train_wav2vec(
                data_dir=str(BASE_DIR / "data" / "processed" / "augmented"),
                model_dir=str(BASE_DIR / "models" / "wav2vec"),
                epochs=args.epochs_wav2vec,
                batch_size=8,
                lr=1e-4,
                seed=args.seed
            )
        
        # Train Mel CNN
        if args.train_mel:
            print("\n--- Training Mel CNN Model ---")
            train_mel_cnn(
                data_dir=str(BASE_DIR / "data" / "processed" / "mel"),
                model_dir=str(BASE_DIR / "models" / "mel_cnn"),
                epochs=args.epochs_mel,
                batch_size=32,
                lr=1e-3,
                seed=args.seed
            )
        
        # Train MFCC BiLSTM
        if args.train_mfcc:
            print("\n--- Training MFCC BiLSTM Model ---")
            train_mfcc(
                data_dir=str(BASE_DIR / "data" / "processed" / "mfcc"),
                model_dir=str(BASE_DIR / "models" / "mfcc"),
                epochs=args.epochs_mfcc,
                batch_size=32,
                lr=1e-3,
                seed=args.seed
            )
    
    # Step 4: Evaluation
    if args.evaluate:
        print_header("STEP 4: MODEL EVALUATION")
        evaluate_all_models(str(BASE_DIR), seed=args.seed)
    
    # Step 5: Final Decision Fusion
    if args.fusion:
        print_header("STEP 5: ACCURACY-WEIGHTED FUSION")
        run_final_decision(str(BASE_DIR))
    
    print_header("PIPELINE COMPLETE")
    print("✓ All steps executed successfully!")
    print(f"✓ Models saved in: {BASE_DIR / 'models'}")
    print(f"✓ Check fusion_results.json for final performance\n")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end deepfake audio detection pipeline"
    )
    
    # Pipeline control
    parser.add_argument('--preprocess', action='store_true', help='Run MUSAN preprocessing')
    parser.add_argument('--extract', action='store_true', help='Run feature extraction')
    parser.add_argument('--train', action='store_true', help='Run model training')
    parser.add_argument('--evaluate', action='store_true', help='Run model evaluation')
    parser.add_argument('--fusion', action='store_true', help='Run final fusion')
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    
    # Individual model training
    parser.add_argument('--train-wav2vec', action='store_true', help='Train Wav2Vec2 model')
    parser.add_argument('--train-mel', action='store_true', help='Train Mel CNN model')
    parser.add_argument('--train-mfcc', action='store_true', help='Train MFCC model')
    
    # Training parameters
    parser.add_argument('--epochs-wav2vec', type=int, default=3, help='Epochs for Wav2Vec2')
    parser.add_argument('--epochs-mel', type=int, default=15, help='Epochs for Mel CNN')
    parser.add_argument('--epochs-mfcc', type=int, default=15, help='Epochs for MFCC')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # If --all is specified, enable all steps
    if args.all:
        args.preprocess = True
        args.extract = True
        args.train = True
        args.train_wav2vec = True
        args.train_mel = True
        args.train_mfcc = True
        args.evaluate = True
        args.fusion = True
    
    # If --train is specified without specific models, train all
    if args.train and not any([args.train_wav2vec, args.train_mel, args.train_mfcc]):
        args.train_wav2vec = True
        args.train_mel = True
        args.train_mfcc = True
    
    # If no arguments, show help
    if not any([args.preprocess, args.extract, args.train, args.evaluate, args.fusion]):
        parser.print_help()
        return
    
    run_pipeline(args)


if __name__ == "__main__":
    main()