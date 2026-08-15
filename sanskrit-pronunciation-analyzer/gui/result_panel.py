"""
result_panel.py
----------------
Displays the outcome of a recognition pass:
  - Detected Alphabet
  - Confidence %
  - Place of Articulation (+ Sanskrit term / category)
  - Recording duration
"""

from typing import Optional

import customtkinter as ctk

from utils.articulation_repository import ArticulationInfo
from recognition.base_recognizer import RecognitionResult


class ResultPanel(ctk.CTkFrame):
    """Right-hand panel summarizing the most recent recognition result."""

    def __init__(self, master, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._build_widgets()

    def _build_widgets(self) -> None:
        self.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(self, text="Result", font=ctk.CTkFont(size=18, weight="bold"))
        title.grid(row=0, column=0, sticky="w", padx=16, pady=(16, 8))

        self._detected_label = self._make_field("Detected", "—", row=1)
        self._confidence_label = self._make_field("Confidence", "—", row=3)
        self._place_label = self._make_field("Place of Articulation", "—", row=5)
        self._category_label = self._make_field("Category", "—", row=7)
        self._duration_label = self._make_field("Recording Duration", "—", row=9)
        self._heard_label = self._make_field("Heard (raw transcript)", "—", row=11, value_font_size=14)

        self._status_label = ctk.CTkLabel(
            self, text="", font=ctk.CTkFont(size=13), text_color="#e07b39", wraplength=260, justify="left"
        )
        self._status_label.grid(row=13, column=0, sticky="w", padx=16, pady=(12, 16))

    def _make_field(self, caption: str, value: str, row: int, value_font_size: int = 20) -> ctk.CTkLabel:
        caption_label = ctk.CTkLabel(
            self, text=caption.upper(), font=ctk.CTkFont(size=11, weight="bold"), text_color="#8a8a8a"
        )
        caption_label.grid(row=row, column=0, sticky="w", padx=16, pady=(6, 0))

        value_label = ctk.CTkLabel(
            self, text=value, font=ctk.CTkFont(size=value_font_size, weight="bold"),
            wraplength=260, justify="left",
        )
        value_label.grid(row=row + 1, column=0, sticky="w", padx=16, pady=(0, 4))
        return value_label

    # ------------------------------------------------------------------
    # Public update methods
    # ------------------------------------------------------------------
    def show_result(self, result: RecognitionResult, info: Optional[ArticulationInfo], duration_seconds: float) -> None:
        """Populate the panel with a completed recognition result."""
        self._status_label.configure(text="")

        if not result.is_recognized or info is None:
            self._detected_label.configure(text="Not recognized", text_color="#e0e0e0")
            self._confidence_label.configure(text=f"{result.confidence_percent}%")
            self._place_label.configure(text="—")
            self._category_label.configure(text="—")
            self._duration_label.configure(text=f"{duration_seconds:.2f} s")
            self._heard_label.configure(text=result.raw_transcript or "(nothing transcribed)")

            if not result.raw_transcript:
                self._status_label.configure(
                    text="No transcript was produced at all. This usually means the speech-recognition "
                         "engine couldn't be reached (check your internet connection) or the "
                         "'SpeechRecognition' package isn't installed. Check logs/app.log for details."
                )
            else:
                self._status_label.configure(
                    text="Heard something, but it didn't confidently match a known alphabet. "
                         "Try speaking a single, clear syllable closer to the mic."
                )
            return

        self._detected_label.configure(text=result.alphabet, text_color="#33d17a")
        self._confidence_label.configure(text=f"{result.confidence_percent}%")
        self._place_label.configure(text=info.place)
        self._category_label.configure(text=info.category)
        self._duration_label.configure(text=f"{duration_seconds:.2f} s")
        self._heard_label.configure(text=result.raw_transcript or "—")

    def show_message(self, message: str) -> None:
        """Show a transient status message (e.g. errors) without altering results."""
        self._status_label.configure(text=message)

    def clear(self) -> None:
        """Reset all fields to their empty state."""
        self._detected_label.configure(text="—", text_color="#e0e0e0")
        self._confidence_label.configure(text="—")
        self._place_label.configure(text="—")
        self._category_label.configure(text="—")
        self._duration_label.configure(text="—")
        self._heard_label.configure(text="—")
        self._status_label.configure(text="")