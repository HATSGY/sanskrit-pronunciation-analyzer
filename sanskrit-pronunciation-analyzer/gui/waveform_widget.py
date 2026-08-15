"""
waveform_widget.py
-------------------
A CustomTkinter frame that embeds a matplotlib figure to draw:
  - a live, scrolling waveform while recording is in progress
  - a static waveform of the full recording once it has stopped

Kept as its own widget class so it can be reused/tested independently
of the rest of the GUI.
"""

from typing import Optional

import customtkinter as ctk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from utils.config import AUDIO_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)

# Dark theme colors to match the rest of the CustomTkinter UI
FIGURE_BG = "#1a1a1a"
AXES_BG = "#1f1f1f"
WAVE_COLOR = "#33d17a"
GRID_COLOR = "#3a3a3a"


class WaveformWidget(ctk.CTkFrame):
    """Large waveform display panel with live + static rendering modes."""

    def __init__(self, master, **kwargs) -> None:
        super().__init__(master, **kwargs)

        self._sample_rate = AUDIO_CONFIG.sample_rate
        max_points = int(AUDIO_CONFIG.waveform_history_seconds * self._sample_rate)
        # Rolling buffer used while live-recording (fixed size ring buffer of amplitude values)
        self._live_buffer = np.zeros(max_points, dtype=np.float32)
        self._max_points = max_points

        self._build_figure()

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------
    def _build_figure(self) -> None:
        self.figure = Figure(figsize=(7, 3), dpi=100, facecolor=FIGURE_BG)
        self.axes = self.figure.add_subplot(111)
        self._style_axes()

        (self._line,) = self.axes.plot([], [], color=WAVE_COLOR, linewidth=1.2)
        self.axes.set_ylim(-1.0, 1.0)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas.draw()

    def _style_axes(self) -> None:
        self.axes.set_facecolor(AXES_BG)
        self.axes.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.6)
        self.axes.set_xticks([])
        self.axes.tick_params(colors="#9a9a9a")
        for spine in self.axes.spines.values():
            spine.set_color(GRID_COLOR)

    # ------------------------------------------------------------------
    # Live updates (called repeatedly while recording)
    # ------------------------------------------------------------------
    def push_live_chunk(self, chunk: np.ndarray) -> None:
        """Append a new small audio chunk to the rolling buffer and redraw."""
        if chunk.size == 0:
            return

        n = len(chunk)
        if n >= self._max_points:
            self._live_buffer = chunk[-self._max_points:]
        else:
            self._live_buffer = np.roll(self._live_buffer, -n)
            self._live_buffer[-n:] = chunk

        self._redraw(self._live_buffer)

    # ------------------------------------------------------------------
    # Static rendering (called once after recording stops)
    # ------------------------------------------------------------------
    def show_static_waveform(self, samples: np.ndarray, sample_rate: int) -> None:
        """Render the complete recorded waveform and keep it visible."""
        if samples.size == 0:
            logger.warning("show_static_waveform called with empty samples.")
            return
        self._sample_rate = sample_rate
        self._redraw(samples)

    def clear(self) -> None:
        """Reset the waveform display to an empty/flat line."""
        self._live_buffer = np.zeros(self._max_points, dtype=np.float32)
        self._line.set_data([], [])
        self.axes.set_xlim(0, 1)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Internal drawing helper
    # ------------------------------------------------------------------
    def _redraw(self, samples: np.ndarray) -> None:
        x = np.linspace(0, len(samples) / max(self._sample_rate, 1), num=len(samples))
        self._line.set_data(x, samples)
        self.axes.set_xlim(0, max(x[-1], 0.01) if len(x) else 1)

        peak = float(np.max(np.abs(samples))) if samples.size else 1.0
        y_limit = max(peak * 1.2, 0.1)
        self.axes.set_ylim(-y_limit, y_limit)

        self.canvas.draw_idle()
