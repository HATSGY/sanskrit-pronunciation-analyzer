"""
test_app.py
-----------
Comprehensive end-to-end and unit test suite for the Sanskrit Pronunciation Analyzer:
  1. Articulation repository validation (all 27 alphabets mapped correctly)
  2. Acoustic feature extraction & VAD checks
  3. ML Phoneme Recognizer inference & confidence checks
  4. Factory backend registry verification
  5. Full GUI application lifecycle & mock recognition event dispatch
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# Adjust project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.features import extract_feature_vector, trim_silence
from audio.recorder import RecordingResult
from gui.main_window import SanskritPronunciationApp
from recognition.base_recognizer import RecognitionResult
from recognition.ml_recognizer import MLPhonemeRecognizer
from recognition.recognizer_factory import RecognizerFactory
from utils.articulation_repository import ArticulationRepository
from utils.config import AUDIO_CONFIG, SUPPORTED_ALPHABETS


class TestSanskritAnalyzer(unittest.TestCase):

    def test_01_articulation_repository(self):
        """Verify all 27 supported Sanskrit alphabets have valid articulation data."""
        repo = ArticulationRepository()
        all_data = repo.all_alphabets()
        self.assertEqual(len(all_data), 27)

        for alpha in SUPPORTED_ALPHABETS:
            info = repo.get_info(alpha)
            self.assertIsNotNone(info, f"Missing articulation info for alphabet '{alpha}'")
            self.assertTrue(bool(info.place), f"Empty place for '{alpha}'")
            self.assertTrue(bool(info.category), f"Empty category for '{alpha}'")
            self.assertIn(info.highlight_region, ["throat", "palate", "roof", "teeth", "lips"])

    def test_02_feature_extraction(self):
        """Verify feature extraction produces correct vector shape and handles silence."""
        sample_rate = 16000
        # 1. Sine wave signal
        t = np.linspace(0, 1.0, sample_rate)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        feats = extract_feature_vector(audio, sample_rate)
        self.assertIsInstance(feats, np.ndarray)
        self.assertEqual(len(feats), 115)
        self.assertFalse(np.isnan(feats).any(), "NaN found in feature vector")

        # 2. Silence trimming
        silence = np.zeros(sample_rate, dtype=np.float32)
        trimmed = trim_silence(silence, sample_rate)
        self.assertIsInstance(trimmed, np.ndarray)

    def test_03_ml_phoneme_recognizer(self):
        """Verify the trained ML model recognizes sample audio correctly."""
        recognizer = RecognizerFactory.create("ml")
        self.assertIsInstance(recognizer, MLPhonemeRecognizer)

        # Test on Ka sample
        ka_file = PROJECT_ROOT / "data" / "dataset" / "Ka" / "sample_bootstrap_000.wav"
        if ka_file.exists():
            sr, raw = wavfile.read(str(ka_file))
            samples = raw.astype(np.float32) / 32768.0
            res = recognizer.recognize(samples, sr)

            self.assertIsInstance(res, RecognitionResult)
            self.assertTrue(res.is_recognized)
            self.assertEqual(res.alphabet, "Ka")
            self.assertGreaterEqual(res.confidence, 0.70)

    def test_04_gui_lifecycle_and_mock_event(self):
        """Test full GUI instantiation, widget construction, and recognition handling."""
        app = SanskritPronunciationApp()
        app.update()  # Process pending GUI events

        # Simulate completed recognition for 'Cha' (Palatal)
        dummy_audio = np.random.uniform(-0.1, 0.1, 16000).astype(np.float32)
        rec_result = RecordingResult(samples=dummy_audio, sample_rate=16000)
        rec_result.duration_seconds = 1.0

        mock_recognition = RecognitionResult(alphabet="Cha", confidence=0.95, raw_transcript="Cha (prob: 95%)")
        app._on_recognition_complete(mock_recognition, rec_result)
        app.update()

        # Check Result Panel updated
        self.assertEqual(app.result_panel._detected_label.cget("text"), "Cha")
        self.assertEqual(app.result_panel._confidence_label.cget("text"), "95%")
        self.assertEqual(app.result_panel._place_label.cget("text"), "Hard Palate (Talavya)")

        # Check Articulation Widget highlighted region dot radius/color
        palate_dot = app.articulation_widget._highlight_dots["palate"]
        self.assertEqual(palate_dot.get_radius(), 0.05)

        # Test Clear button
        app._on_clear()
        app.update()
        self.assertEqual(app.result_panel._detected_label.cget("text"), "—")
        self.assertEqual(palate_dot.get_radius(), 0.035)

        # Destroy GUI cleanly
        app.destroy()


if __name__ == "__main__":
    unittest.main(verbosity=2)
