"""
recorder.py
-----------
Handles microphone capture using `sounddevice`.

Design notes:
- Recording happens on a background thread driven by sounddevice's
  InputStream callback, so the GUI never blocks.
- Every incoming audio block is pushed onto a thread-safe `queue.Queue`
  so the GUI can poll it (via `tkinter.after`) to redraw the live waveform.
- The full recorded audio is accumulated separately so it can be handed
  to the recognition module once recording stops.
"""

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    import sounddevice as sd
except OSError as exc:  # pragma: no cover - depends on host audio backend
    sd = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from utils.config import AUDIO_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


class MicrophoneUnavailableError(RuntimeError):
    """Raised when the system has no usable microphone / audio backend."""


@dataclass
class RecordingResult:
    """Container for a completed recording."""
    samples: np.ndarray             # 1-D float32 array of recorded audio
    sample_rate: int
    duration_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        self.duration_seconds = len(self.samples) / float(self.sample_rate) if self.sample_rate else 0.0


class AudioRecorder:
    """
    Wraps a sounddevice InputStream to provide:
      - start()/stop() control of microphone recording
      - a live queue of small audio blocks for waveform visualization
      - the full accumulated recording once stopped
    """

    def __init__(
        self,
        sample_rate: int = AUDIO_CONFIG.sample_rate,
        channels: int = AUDIO_CONFIG.channels,
        block_size: int = AUDIO_CONFIG.block_size,
        dtype: str = AUDIO_CONFIG.dtype,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.dtype = dtype

        self._stream: Optional["sd.InputStream"] = None
        self._recorded_chunks: List[np.ndarray] = []
        self._is_recording = threading.Event()
        self._start_time: Optional[float] = None

        # Thread-safe queue of small numpy chunks, consumed by the GUI for live waveform drawing
        self.live_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_recording(self) -> bool:
        return self._is_recording.is_set()

    def start(self) -> None:
        """Start capturing audio from the default microphone."""
        if sd is None:
            logger.error("sounddevice backend unavailable: %s", _IMPORT_ERROR)
            raise MicrophoneUnavailableError(
                "No audio backend found. Please ensure a microphone is connected "
                "and PortAudio is installed."
            )

        if self.is_recording():
            logger.warning("start() called while already recording; ignoring.")
            return

        self._recorded_chunks.clear()
        # Drain any stale items from a previous session
        while not self.live_queue.empty():
            self.live_queue.get_nowait()

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.block_size,
                dtype=self.dtype,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_recording.set()
            self._start_time = time.time()
            logger.info("Recording started (sr=%d, channels=%d)", self.sample_rate, self.channels)
        except Exception as exc:
            logger.exception("Failed to start audio stream")
            raise MicrophoneUnavailableError(str(exc)) from exc

    def stop(self) -> RecordingResult:
        """Stop capturing and return the full recorded audio."""
        if not self.is_recording():
            logger.warning("stop() called while not recording.")
            return RecordingResult(samples=np.array([], dtype=np.float32), sample_rate=self.sample_rate)

        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            logger.exception("Error while stopping audio stream")
        finally:
            self._stream = None
            self._is_recording.clear()

        if self._recorded_chunks:
            full_audio = np.concatenate(self._recorded_chunks, axis=0).flatten().astype(np.float32)
        else:
            full_audio = np.array([], dtype=np.float32)

        elapsed = time.time() - self._start_time if self._start_time else 0.0
        logger.info("Recording stopped. Captured %.2fs of audio (%d samples).", elapsed, len(full_audio))

        return RecordingResult(samples=full_audio, sample_rate=self.sample_rate)

    # ------------------------------------------------------------------
    # Internal callback (runs on sounddevice's audio thread)
    # ------------------------------------------------------------------
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.debug("Audio callback status: %s", status)

        chunk = indata.copy()
        self._recorded_chunks.append(chunk)

        try:
            self.live_queue.put_nowait(chunk.flatten())
        except queue.Full:  # pragma: no cover - unbounded queue, defensive only
            pass

    @staticmethod
    def list_input_devices() -> List[str]:
        """Return human-readable names of available input devices (for diagnostics/logging)."""
        if sd is None:
            return []
        try:
            devices = sd.query_devices()
            return [d["name"] for d in devices if d.get("max_input_channels", 0) > 0]
        except Exception:
            logger.exception("Failed to query audio devices")
            return []
