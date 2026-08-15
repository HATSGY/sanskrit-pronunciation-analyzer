"""
record_dataset.py
-----------------
Interactive CLI utility for recording and managing Sanskrit phoneme audio datasets.

Usage:
  1. Record your voice for all or specific alphabets:
     python scripts/record_dataset.py --record --reps 3

  2. Record a specific alphabet:
     python scripts/record_dataset.py --alphabet Ka --reps 5

  3. Generate bootstrap baseline acoustic data (for immediate training/testing):
     python scripts/record_dataset.py --bootstrap --reps 10
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from scipy.io import wavfile

# Adjust Python path so we can import project modules
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.config import AUDIO_CONFIG, DATA_DIR, SUPPORTED_ALPHABETS
from utils.logger import get_logger

try:
    import sounddevice as sd
except ImportError:
    sd = None

logger = get_logger(__name__)
DATASET_DIR = DATA_DIR / "dataset"


# Acoustic Formant / Characteristic Profiles for Sanskrit Phonemes
# (F1, F2, F3 frequencies in Hz, aspiration level, noise level, voicing)
PHONETIC_ACOUSTIC_PROFILES: Dict[str, Dict] = {
    # Guttural (Kanthya) - Back of throat
    "Ka":  {"f1": 650, "f2": 1400, "f3": 2400, "aspiration": 0.05, "noise": 0.10, "voiced": False, "burst": 0.15},
    "Kha": {"f1": 650, "f2": 1400, "f3": 2400, "aspiration": 0.35, "noise": 0.25, "voiced": False, "burst": 0.25},
    "Ga":  {"f1": 600, "f2": 1350, "f3": 2350, "aspiration": 0.05, "noise": 0.05, "voiced": True,  "burst": 0.10},
    "Gha": {"f1": 600, "f2": 1350, "f3": 2350, "aspiration": 0.30, "noise": 0.20, "voiced": True,  "burst": 0.20},
    "Nga": {"f1": 400, "f2": 1200, "f3": 2200, "aspiration": 0.02, "noise": 0.02, "voiced": True,  "nasal": 0.35},

    # Palatal (Talavya) - Hard palate
    "Cha": {"f1": 550, "f2": 2100, "f3": 2900, "aspiration": 0.15, "noise": 0.30, "voiced": False, "burst": 0.20},
    "Chha":{"f1": 550, "f2": 2100, "f3": 2900, "aspiration": 0.40, "noise": 0.45, "voiced": False, "burst": 0.30},
    "Ja":  {"f1": 500, "f2": 2000, "f3": 2800, "aspiration": 0.10, "noise": 0.15, "voiced": True,  "burst": 0.15},
    "Jha": {"f1": 500, "f2": 2000, "f3": 2800, "aspiration": 0.35, "noise": 0.35, "voiced": True,  "burst": 0.25},
    "Nya": {"f1": 350, "f2": 1900, "f3": 2700, "aspiration": 0.02, "noise": 0.02, "voiced": True,  "nasal": 0.35},

    # Retroflex (Murdhanya) - Roof of mouth / curled tongue
    "Ta":  {"f1": 520, "f2": 1800, "f3": 2550, "aspiration": 0.08, "noise": 0.12, "voiced": False, "burst": 0.20},
    "Tha": {"f1": 520, "f2": 1800, "f3": 2550, "aspiration": 0.38, "noise": 0.30, "voiced": False, "burst": 0.30},
    "Da":  {"f1": 480, "f2": 1750, "f3": 2500, "aspiration": 0.06, "noise": 0.08, "voiced": True,  "burst": 0.12},
    "Dha": {"f1": 480, "f2": 1750, "f3": 2500, "aspiration": 0.32, "noise": 0.22, "voiced": True,  "burst": 0.22},
    "Na":  {"f1": 380, "f2": 1600, "f3": 2400, "aspiration": 0.02, "noise": 0.02, "voiced": True,  "nasal": 0.30},

    # Labial (Oshthya) - Lips
    "Pa":  {"f1": 580, "f2": 950,  "f3": 2200, "aspiration": 0.06, "noise": 0.08, "voiced": False, "burst": 0.22},
    "Pha": {"f1": 580, "f2": 950,  "f3": 2200, "aspiration": 0.38, "noise": 0.32, "voiced": False, "burst": 0.30},
    "Ba":  {"f1": 520, "f2": 900,  "f3": 2150, "aspiration": 0.04, "noise": 0.05, "voiced": True,  "burst": 0.12},
    "Bha": {"f1": 520, "f2": 900,  "f3": 2150, "aspiration": 0.30, "noise": 0.22, "voiced": True,  "burst": 0.22},
    "Ma":  {"f1": 350, "f2": 850,  "f3": 2100, "aspiration": 0.02, "noise": 0.02, "voiced": True,  "nasal": 0.40},

    # Semi-vowels (Antahstha)
    "Ya":  {"f1": 320, "f2": 2200, "f3": 2900, "aspiration": 0.02, "noise": 0.03, "voiced": True,  "glide": True},
    "Ra":  {"f1": 450, "f2": 1300, "f3": 1700, "aspiration": 0.05, "noise": 0.08, "voiced": True,  "trill": True},
    "La":  {"f1": 420, "f2": 1100, "f3": 2600, "aspiration": 0.03, "noise": 0.04, "voiced": True,  "lateral": True},
    "Va":  {"f1": 380, "f2": 900,  "f3": 2300, "aspiration": 0.04, "noise": 0.05, "voiced": True,  "glide": True},

    # Sibilants & Aspirate (Ushmana)
    "Sha": {"f1": 400, "f2": 2000, "f3": 3200, "aspiration": 0.10, "noise": 0.60, "voiced": False, "fricative": True},
    "Sa":  {"f1": 350, "f2": 1700, "f3": 4500, "aspiration": 0.08, "noise": 0.70, "voiced": False, "fricative": True},
    "Ha":  {"f1": 700, "f2": 1400, "f3": 2500, "aspiration": 0.65, "noise": 0.40, "voiced": True,  "glottal": True},
}


def synthesize_phoneme_audio(alphabet: str, sample_rate: int = 16000, duration: float = 0.8, variation: float = 0.05) -> np.ndarray:
    """
    Synthesize an acoustic signal modeling formants, bursts, and aspiration
    for a given Sanskrit alphabet. Used for bootstrap dataset generation.
    """
    profile = PHONETIC_ACOUSTIC_PROFILES.get(
        alphabet,
        {"f1": 500, "f2": 1500, "f3": 2500, "aspiration": 0.1, "noise": 0.1, "voiced": True},
    )

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    rng = np.random.RandomState()

    # Apply random pitch and formant jitter per sample
    pitch = 120.0 * (1.0 + rng.uniform(-variation, variation))
    f1 = profile["f1"] * (1.0 + rng.uniform(-variation, variation))
    f2 = profile["f2"] * (1.0 + rng.uniform(-variation, variation))
    f3 = profile["f3"] * (1.0 + rng.uniform(-variation, variation))

    # Envelope: attack, sustain, decay
    attack_samples = int(0.08 * sample_rate)
    decay_samples = int(0.20 * sample_rate)
    sustain_samples = len(t) - attack_samples - decay_samples

    envelope = np.concatenate([
        np.linspace(0, 1.0, attack_samples),
        np.ones(max(0, sustain_samples)),
        np.linspace(1.0, 0, decay_samples),
    ])[: len(t)]

    # Glottal pulse / Voiced source
    if profile.get("voiced", True):
        # Harmonic excitation
        harmonics = np.zeros_like(t)
        for h in range(1, 20):
            freq = h * pitch
            if freq < sample_rate / 2:
                harmonics += (1.0 / h) * np.sin(2 * np.pi * freq * t + rng.uniform(0, np.pi))
        source = harmonics
    else:
        source = np.zeros_like(t)

    # Formant resonators
    def resonator(signal_in, f_center, bandwidth=100.0):
        w = 2 * np.pi * f_center / sample_rate
        r = np.exp(-np.pi * bandwidth / sample_rate)
        # 2nd order IIR bandpass resonator
        b = [1 - r]
        a = [1, -2 * r * np.cos(w), r * r]
        from scipy.signal import lfilter
        return lfilter(b, a, signal_in)

    vocal_tract = np.zeros_like(t)
    if profile.get("voiced", True):
        vocal_tract = (
            0.50 * resonator(source, f1, 100)
            + 0.30 * resonator(source, f2, 130)
            + 0.20 * resonator(source, f3, 160)
        )

    # Noise / Aspiration / Fricative excitation
    noise = rng.normal(0, 1, size=len(t))
    noise_filt = resonator(noise, f3, bandwidth=f3 * 0.4) * profile.get("noise", 0.1)

    # Plosive burst
    burst_amp = profile.get("burst", 0.0)
    burst = np.zeros_like(t)
    if burst_amp > 0:
        burst_len = int(0.025 * sample_rate)
        burst[:burst_len] = rng.normal(0, burst_amp, burst_len) * np.exp(-np.linspace(0, 5, burst_len))

    # Combine components
    signal_out = (vocal_tract + noise_filt + burst) * envelope

    # Normalization
    max_val = np.max(np.abs(signal_out)) + 1e-12
    signal_out = (signal_out / max_val) * 0.85

    return signal_out.astype(np.float32)


def generate_bootstrap_dataset(reps_per_alphabet: int = 15) -> int:
    """Generate baseline dataset for all 27 Sanskrit alphabets."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    total_saved = 0

    print(f"Generating {reps_per_alphabet} baseline acoustic samples per alphabet...")
    for alphabet in SUPPORTED_ALPHABETS:
        alphabet_dir = DATASET_DIR / alphabet
        alphabet_dir.mkdir(parents=True, exist_ok=True)

        for i in range(reps_per_alphabet):
            samples = synthesize_phoneme_audio(alphabet, sample_rate=AUDIO_CONFIG.sample_rate, variation=0.08)
            filename = alphabet_dir / f"sample_bootstrap_{i:03d}.wav"
            int16_pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
            wavfile.write(str(filename), AUDIO_CONFIG.sample_rate, int16_pcm)
            total_saved += 1

    print(f"[OK] Successfully generated {total_saved} samples in '{DATASET_DIR}'.")
    return total_saved


def record_interactive(alphabets: List[str], reps: int = 3, duration: float = 1.8) -> None:
    """Prompt user to record microphone samples interactively."""
    if sd is None:
        print("[ERROR] sounddevice is not available. Please install sounddevice to record.")
        return

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    sr = AUDIO_CONFIG.sample_rate

    print("\n" + "=" * 60)
    print("  SANSKRIT PHONEME DATASET RECORDER")
    print("=" * 60)
    print(f"Alphabets to record: {len(alphabets)}")
    print(f"Repetitions each:    {reps}")
    print(f"Duration per sample: {duration:.1f}s")
    print("Speak clearly as soon as you see '>>> RECORDING NOW...'\n")

    for alphabet in alphabets:
        alphabet_dir = DATASET_DIR / alphabet
        alphabet_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(alphabet_dir.glob("*.wav")))

        print(f"\n--- Alphabet: [ {alphabet} ] ({existing} samples currently saved) ---")
        for r in range(reps):
            input(f"Press [ENTER] when ready for repetition #{r+1} of '{alphabet}'...")
            print("  3...", end=" ", flush=True); time.sleep(0.5)
            print("2...", end=" ", flush=True); time.sleep(0.5)
            print("1...", end=" ", flush=True); time.sleep(0.5)
            print(f">>> RECORDING NOW: Pronounce '{alphabet}'!")

            rec = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
            sd.wait()
            print("  [Done recording]")

            samples = rec.flatten()
            timestamp = int(time.time())
            filename = alphabet_dir / f"sample_user_{timestamp}_{r+1}.wav"
            int16_pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
            wavfile.write(str(filename), sr, int16_pcm)
            print(f"  Saved to {filename.name}")

    print("\n[SUCCESS] Dataset collection completed! Run 'python scripts/train_classifier.py' to train.")


def main():
    parser = argparse.ArgumentParser(description="Record and generate dataset samples for Sanskrit alphabets.")
    parser.add_argument("--record", action="store_true", help="Launch interactive microphone recording")
    parser.add_argument("--bootstrap", action="store_true", help="Generate synthetic acoustic baseline dataset")
    parser.add_argument("--alphabet", type=str, default=None, help="Record only a specific alphabet (e.g. Ka)")
    parser.add_argument("--reps", type=int, default=3, help="Number of repetitions per alphabet (default: 3)")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration in seconds (default: 2.0s)")

    args = parser.parse_args()

    if args.bootstrap:
        generate_bootstrap_dataset(reps_per_alphabet=args.reps if args.reps > 3 else 15)
    elif args.record or args.alphabet:
        target_alphabets = [args.alphabet] if args.alphabet else SUPPORTED_ALPHABETS
        record_interactive(target_alphabets, reps=args.reps, duration=args.duration)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
