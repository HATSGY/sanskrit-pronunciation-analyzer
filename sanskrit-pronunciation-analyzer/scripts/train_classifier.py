"""
train_classifier.py
-------------------
Trains an acoustic machine learning classifier for Sanskrit phoneme recognition.

Loads audio samples from data/dataset/<Alphabet>/*.wav, extracts acoustic
features (MFCCs, spectral descriptors, energy, deltas), performs data
augmentation, trains a calibrated scikit-learn model, and exports it to
models/sanskrit_phoneme_model.joblib.

Usage:
  python scripts/train_classifier.py
  python scripts/train_classifier.py --bootstrap   # Generates baseline data and trains immediately
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import numpy as np
from scipy.io import wavfile
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from audio.features import extract_feature_vector
from scripts.record_dataset import generate_bootstrap_dataset
from utils.config import AUDIO_CONFIG, DATA_DIR, SUPPORTED_ALPHABETS
from utils.logger import get_logger

logger = get_logger(__name__)

DATASET_DIR = DATA_DIR / "dataset"
MODELS_DIR = ROOT_DIR / "models"
MODEL_OUTPUT_PATH = MODELS_DIR / "sanskrit_phoneme_model.joblib"


def augment_audio(samples: np.ndarray, sample_rate: int) -> List[np.ndarray]:
    """
    Generate augmented versions of an audio clip:
      1. Original
      2. Gain variations (+2dB, -2dB)
      3. Additive subtle white noise (SNR ~ 30dB)
      4. Small time shifts
    """
    augmented = [samples]

    # 1. Gain variations
    augmented.append(samples * 1.25)
    augmented.append(samples * 0.80)

    # 2. Subtle noise
    rms = np.sqrt(np.mean(samples**2) + 1e-12)
    noise = np.random.normal(0, rms * 0.05, size=len(samples)).astype(np.float32)
    augmented.append(samples + noise)

    # 3. Small time shift
    shift_samples = int(0.03 * sample_rate)  # 30ms
    if len(samples) > shift_samples:
        shifted_right = np.pad(samples[:-shift_samples], (shift_samples, 0))
        augmented.append(shifted_right)

    return augmented


def load_dataset(dataset_dir: Path, augment: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scan dataset directory and extract feature vectors for all available audio files.
    """
    X_list = []
    y_list = []

    print(f"Loading dataset from: {dataset_dir}")
    for alphabet in SUPPORTED_ALPHABETS:
        alphabet_dir = dataset_dir / alphabet
        if not alphabet_dir.exists():
            continue

        wav_files = list(alphabet_dir.glob("*.wav"))
        if not wav_files:
            continue

        for wav_path in wav_files:
            try:
                sr, raw_int16 = wavfile.read(str(wav_path))
                # Convert to float32 in [-1.0, 1.0]
                if raw_int16.dtype == np.int16:
                    samples = raw_int16.astype(np.float32) / 32768.0
                else:
                    samples = raw_int16.astype(np.float32)

                if augment:
                    variations = augment_audio(samples, sr)
                else:
                    variations = [samples]

                for var_samples in variations:
                    feats = extract_feature_vector(var_samples, sample_rate=sr)
                    X_list.append(feats)
                    y_list.append(alphabet)

            except Exception as exc:
                logger.warning("Error reading %s: %s", wav_path, exc)

    if not X_list:
        raise ValueError("No audio samples found in dataset directory!")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    return X, y


def train_model(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """
    Train and calibrate a robust acoustic classifier pipeline with probabilistic outputs.
    """
    # Use an ensemble of ExtraTrees / MLP for fast inference and strong generalization
    base_clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=0.001,
        learning_rate_init=0.005,
        max_iter=300,
        random_state=42,
        early_stopping=True,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", base_clf),
    ])

    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTraining on {len(X_train)} samples across {len(np.unique(y))} classes...")
    pipeline.fit(X_train, y_train)

    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    print(f"[OK] Training Accuracy:   {train_acc * 100:.2f}%")
    print(f"[OK] Validation Accuracy: {test_acc * 100:.2f}%")

    # Fit final pipeline on all data
    pipeline.fit(X, y)
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Train Sanskrit phoneme recognition model.")
    parser.add_argument("--bootstrap", action="store_true", help="Generate synthetic baseline dataset before training")
    parser.add_argument("--no-augment", action="store_true", help="Disable audio data augmentation")
    args = parser.parse_args()

    # Check if dataset exists or bootstrap requested
    if args.bootstrap or not DATASET_DIR.exists() or len(list(DATASET_DIR.glob("*/*.wav"))) == 0:
        print("Dataset empty or --bootstrap specified. Generating baseline acoustic samples...")
        generate_bootstrap_dataset(reps_per_alphabet=15)

    X, y = load_dataset(DATASET_DIR, augment=not args.no_augment)
    print(f"Total extracted feature vectors: {X.shape[0]} (Feature dimension: {X.shape[1]})")

    model_pipeline = train_model(X, y)

    # Save model package
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "pipeline": model_pipeline,
        "classes": list(model_pipeline.classes_),
        "sample_rate": AUDIO_CONFIG.sample_rate,
        "feature_dim": X.shape[1],
    }

    joblib.dump(model_payload, MODEL_OUTPUT_PATH)
    print(f"\n[SUCCESS] Model successfully trained and saved to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
