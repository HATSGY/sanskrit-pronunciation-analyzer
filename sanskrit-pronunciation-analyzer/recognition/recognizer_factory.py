"""
recognizer_factory.py
----------------------
Single point of construction for whichever recognition backend is
currently configured. The GUI depends only on `BaseRecognizer` and
calls `RecognizerFactory.create()` - it never imports a concrete
backend directly. This is what makes the "swap Whisper/Vosk/Wav2Vec2/
TensorFlow in later, with zero GUI changes" requirement possible.
"""

from typing import Dict, Type

from recognition.base_recognizer import BaseRecognizer
from recognition.keyword_recognizer import KeywordRecognizer
from recognition.ml_recognizer import MLPhonemeRecognizer
from recognition.future_backends import (
    WhisperRecognizer,
    VoskRecognizer,
    Wav2Vec2Recognizer,
    TensorFlowClassifierRecognizer,
)
from utils.config import RECOGNITION_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)

# Registry mapping a config string -> recognizer class.
# To add a new backend: implement it (see future_backends.py for the shape),
# then register it here. No other code needs to change.
BACKEND_REGISTRY: Dict[str, Type[BaseRecognizer]] = {
    "ml": MLPhonemeRecognizer,
    "phoneme": MLPhonemeRecognizer,
    "keyword": KeywordRecognizer,
    "whisper": WhisperRecognizer,
    "vosk": VoskRecognizer,
    "wav2vec2": Wav2Vec2Recognizer,
    "tensorflow": TensorFlowClassifierRecognizer,
}


class RecognizerFactory:
    """Builds the active BaseRecognizer implementation based on configuration."""

    @staticmethod
    def create(backend_name: str = RECOGNITION_CONFIG.backend) -> BaseRecognizer:
        backend_cls = BACKEND_REGISTRY.get(backend_name)
        if backend_cls is None:
            logger.warning("Unknown backend '%s'; falling back to 'keyword'.", backend_name)
            backend_cls = KeywordRecognizer

        logger.info("Instantiating recognition backend: %s", backend_cls.__name__)
        return backend_cls()
