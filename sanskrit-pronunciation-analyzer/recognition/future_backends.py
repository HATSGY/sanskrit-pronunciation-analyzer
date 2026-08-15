"""
future_backends.py
-------------------
Stub implementations for more advanced recognition backends. These are
NOT wired up by default (see recognizer_factory.py) but demonstrate the
exact shape a future contributor should follow to plug in a real model.
Each class already satisfies the BaseRecognizer contract, so switching
`RECOGNITION_CONFIG.backend` to one of these (once implemented) requires
no GUI changes whatsoever.
"""

import numpy as np

from recognition.base_recognizer import BaseRecognizer, RecognitionResult
from utils.logger import get_logger

logger = get_logger(__name__)


class WhisperRecognizer(BaseRecognizer):
    """
    Placeholder for an OpenAI Whisper-based backend.

    Real implementation sketch:
        import whisper
        self._model = whisper.load_model("base")
        result = self._model.transcribe(audio_float32_array, fp16=False)
        text = result["text"]
        # then reuse KeywordRecognizer._match_alphabet-style fuzzy matching,
        # or a dedicated Sanskrit-alphabet classifier head.
    """

    name = "whisper"

    def recognize(self, samples: np.ndarray, sample_rate: int) -> RecognitionResult:
        logger.warning("WhisperRecognizer is a stub. Install/openai-whisper and implement recognize().")
        raise NotImplementedError("WhisperRecognizer has not been implemented yet.")


class VoskRecognizer(BaseRecognizer):
    """
    Placeholder for an offline Vosk-based backend (useful when no internet
    connection is available, unlike the default Google Web Speech engine).

    Real implementation sketch:
        from vosk import Model, KaldiRecognizer
        self._model = Model("path/to/vosk-model")
        rec = KaldiRecognizer(self._model, sample_rate)
        rec.AcceptWaveform(int16_pcm_bytes)
        result = json.loads(rec.Result())
    """

    name = "vosk"

    def recognize(self, samples: np.ndarray, sample_rate: int) -> RecognitionResult:
        logger.warning("VoskRecognizer is a stub. Download a Vosk model and implement recognize().")
        raise NotImplementedError("VoskRecognizer has not been implemented yet.")


class Wav2Vec2Recognizer(BaseRecognizer):
    """
    Placeholder for a HuggingFace Wav2Vec2-based backend, ideal for a
    fine-tuned Sanskrit-alphabet phoneme classifier.

    Real implementation sketch:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self._processor = Wav2Vec2Processor.from_pretrained("model-name")
        self._model = Wav2Vec2ForCTC.from_pretrained("model-name")
        inputs = self._processor(samples, sampling_rate=sample_rate, return_tensors="pt")
        logits = self._model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        text = self._processor.batch_decode(predicted_ids)
    """

    name = "wav2vec2"

    def recognize(self, samples: np.ndarray, sample_rate: int) -> RecognitionResult:
        logger.warning("Wav2Vec2Recognizer is a stub. Load a fine-tuned model and implement recognize().")
        raise NotImplementedError("Wav2Vec2Recognizer has not been implemented yet.")


class TensorFlowClassifierRecognizer(BaseRecognizer):
    """
    Placeholder for a custom TensorFlow/Keras model trained specifically
    to classify the 27 Sanskrit alphabets from MFCC/mel-spectrogram
    features extracted from short recordings - the "final" ML upgrade
    path for this project.

    Real implementation sketch:
        import tensorflow as tf
        self._model = tf.keras.models.load_model("models/sanskrit_cnn.h5")
        features = extract_mfcc(samples, sample_rate)   # shape matches model input
        probs = self._model.predict(features[np.newaxis, ...])[0]
        best_idx = int(np.argmax(probs))
        alphabet = SUPPORTED_ALPHABETS[best_idx]
        confidence = float(probs[best_idx])
    """

    name = "tensorflow"

    def recognize(self, samples: np.ndarray, sample_rate: int) -> RecognitionResult:
        logger.warning("TensorFlowClassifierRecognizer is a stub. Train/load a model and implement recognize().")
        raise NotImplementedError("TensorFlowClassifierRecognizer has not been implemented yet.")
