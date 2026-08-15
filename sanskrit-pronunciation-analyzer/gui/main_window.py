"""
main_window.py
---------------
The main CustomTkinter application window. Wires together:
  - AudioRecorder (audio/recorder.py)          - microphone capture
  - RecognizerFactory (recognition/*)          - pluggable ASR backend
  - ArticulationRepository (utils/*)           - pronunciation data
  - WaveformWidget, ArticulationWidget, ResultPanel (gui/*) - visuals

Threading model:
  - Recording runs on sounddevice's own audio thread (see AudioRecorder).
  - The GUI polls `recorder.live_queue` every `waveform_update_ms` via
    `self.after(...)` to redraw the live waveform WITHOUT blocking Tk's
    main loop.
  - Recognition (which may involve a network call) runs on a separate
    `threading.Thread` so the UI never freezes; the result is marshalled
    back to the GUI thread via `self.after(0, ...)`.
"""

import queue
import threading
import time
from typing import Optional

import customtkinter as ctk

from audio.recorder import AudioRecorder, MicrophoneUnavailableError, RecordingResult
from gui.articulation_widget import ArticulationWidget
from gui.result_panel import ResultPanel
from gui.waveform_widget import WaveformWidget
from recognition.base_recognizer import RecognitionResult
from recognition.recognizer_factory import RecognizerFactory
from utils.articulation_repository import ArticulationRepository
from utils.config import AUDIO_CONFIG, UI_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


class SanskritPronunciationApp(ctk.CTk):
    """Root application window."""

    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode(UI_CONFIG.appearance_mode)
        ctk.set_default_color_theme(UI_CONFIG.color_theme)

        self.title(UI_CONFIG.window_title)
        self.geometry(UI_CONFIG.window_size)
        self.minsize(900, 620)

        # --- Core components -------------------------------------------------
        self.recorder = AudioRecorder()
        self.articulation_repo = ArticulationRepository()
        self.recognizer = RecognizerFactory.create()

        self._recording_start_time: Optional[float] = None
        self._live_poll_job: Optional[str] = None
        self._last_recording: Optional[RecordingResult] = None

        self._build_layout()
        self._update_mic_status(idle=True)

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_left_panel()
        self._build_right_panel()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(16, 8))
        header.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(
            header, text="Sanskrit Alphabet Pronunciation Analyzer",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        self.mic_status_label = ctk.CTkLabel(
            header, text="● Idle", font=ctk.CTkFont(size=14, weight="bold"), text_color="#8a8a8a"
        )
        self.mic_status_label.grid(row=0, column=1, sticky="e")

    def _build_left_panel(self) -> None:
        left = ctk.CTkFrame(self)
        left.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=10)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        waveform_title = ctk.CTkLabel(left, text="Live Waveform", font=ctk.CTkFont(size=15, weight="bold"))
        waveform_title.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 0))

        self.waveform_widget = WaveformWidget(left)
        self.waveform_widget.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        controls = ctk.CTkFrame(left, fg_color="transparent")
        controls.grid(row=2, column=0, sticky="ew", padx=12, pady=(4, 12))
        controls.grid_columnconfigure((0, 1, 2), weight=1)

        self.start_button = ctk.CTkButton(
            controls, text="● Start Recording", command=self._on_start_recording,
            fg_color="#2f9e44", hover_color="#268a3a", height=42,
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.stop_button = ctk.CTkButton(
            controls, text="■ Stop Recording", command=self._on_stop_recording,
            fg_color="#c92a2a", hover_color="#a52222", height=42, state="disabled",
        )
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=6)

        self.clear_button = ctk.CTkButton(
            controls, text="Clear", command=self._on_clear,
            fg_color="#4a4a4a", hover_color="#3a3a3a", height=42,
        )
        self.clear_button.grid(row=0, column=2, sticky="ew", padx=(6, 0))

    def _build_right_panel(self) -> None:
        right = ctk.CTkFrame(self)
        right.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=10)
        right.grid_rowconfigure(1, weight=0)
        right.grid_columnconfigure(0, weight=1)

        self.result_panel = ResultPanel(right)
        self.result_panel.grid(row=0, column=0, sticky="new")

        illustration_title = ctk.CTkLabel(
            right, text="Articulation Point", font=ctk.CTkFont(size=13, weight="bold"), text_color="#8a8a8a"
        )
        illustration_title.grid(row=1, column=0, sticky="w", padx=16, pady=(8, 0))

        self.articulation_widget = ArticulationWidget(right)
        self.articulation_widget.grid(row=2, column=0, sticky="nsew", padx=8, pady=8)
        right.grid_rowconfigure(2, weight=1)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    def _on_start_recording(self) -> None:
        try:
            self.recorder.start()
        except MicrophoneUnavailableError as exc:
            logger.error("Microphone unavailable: %s", exc)
            self.result_panel.show_message(f"Microphone error: {exc}")
            self._update_mic_status(error=True)
            return

        self._recording_start_time = time.time()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.clear_button.configure(state="disabled")
        self._update_mic_status(recording=True)
        self._poll_live_waveform()

        # Safety auto-stop after the configured max duration so users don't
        # accidentally leave the mic open indefinitely.
        self.after(int(AUDIO_CONFIG.record_seconds * 1000) + 4000, self._auto_stop_if_still_recording)

    def _auto_stop_if_still_recording(self) -> None:
        if self.recorder.is_recording():
            logger.info("Auto-stopping recording after max duration reached.")
            self._on_stop_recording()

    def _on_stop_recording(self) -> None:
        if self._live_poll_job is not None:
            self.after_cancel(self._live_poll_job)
            self._live_poll_job = None

        result = self.recorder.stop()
        self._last_recording = result

        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.clear_button.configure(state="normal")
        self._update_mic_status(idle=True)

        if result.samples.size == 0:
            self.result_panel.show_message("No audio captured. Please try again.")
            return

        self.waveform_widget.show_static_waveform(result.samples, result.sample_rate)
        self.result_panel.show_message("Analyzing pronunciation…")
        self._run_recognition_async(result)

    def _on_clear(self) -> None:
        self.waveform_widget.clear()
        self.result_panel.clear()
        self.articulation_widget.reset()
        self._last_recording = None
        self._update_mic_status(idle=True)

    # ------------------------------------------------------------------
    # Live waveform polling (runs on the GUI thread via `after`)
    # ------------------------------------------------------------------
    def _poll_live_waveform(self) -> None:
        try:
            while True:
                chunk = self.recorder.live_queue.get_nowait()
                self.waveform_widget.push_live_chunk(chunk)
        except queue.Empty:
            pass

        if self.recorder.is_recording():
            self._live_poll_job = self.after(UI_CONFIG.waveform_update_ms, self._poll_live_waveform)

    # ------------------------------------------------------------------
    # Recognition (runs on a background thread)
    # ------------------------------------------------------------------
    def _run_recognition_async(self, recording: RecordingResult) -> None:
        thread = threading.Thread(target=self._recognition_worker, args=(recording,), daemon=True)
        thread.start()

    def _recognition_worker(self, recording: RecordingResult) -> None:
        try:
            result = self.recognizer.recognize(recording.samples, recording.sample_rate)
        except Exception:
            logger.exception("Recognition backend raised an unexpected error")
            result = RecognitionResult(alphabet="", confidence=0.0)

        # Marshal back onto the GUI thread before touching any widgets
        self.after(0, self._on_recognition_complete, result, recording)

    def _on_recognition_complete(self, result: RecognitionResult, recording: RecordingResult) -> None:
        info = self.articulation_repo.get_info(result.alphabet) if result.is_recognized else None
        self.result_panel.show_result(result, info, recording.duration_seconds)

        if info is not None:
            self.articulation_widget.highlight(info.highlight_region)
        else:
            self.articulation_widget.reset()

    # ------------------------------------------------------------------
    # Status indicator
    # ------------------------------------------------------------------
    def _update_mic_status(self, idle: bool = False, recording: bool = False, error: bool = False) -> None:
        if error:
            self.mic_status_label.configure(text="● Error", text_color="#e03131")
        elif recording:
            self.mic_status_label.configure(text="● Recording…", text_color="#e03131")
        elif idle:
            self.mic_status_label.configure(text="● Idle", text_color="#8a8a8a")
