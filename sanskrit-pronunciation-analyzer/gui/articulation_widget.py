"""
articulation_widget.py
-----------------------
Optional enhancement: draws a simplified side-profile diagram of the
mouth/throat and highlights the region responsible for producing the
detected alphabet (throat, palate, roof of mouth, teeth, or lips).

Drawn procedurally with matplotlib patches - no external image files
required, so the project has zero binary asset dependencies.
"""

from typing import Dict

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Circle

from utils.logger import get_logger

logger = get_logger(__name__)

FIGURE_BG = "#1a1a1a"
AXES_BG = "#1f1f1f"
OUTLINE_COLOR = "#9a9a9a"
HIGHLIGHT_COLOR = "#33d17a"
DEFAULT_COLOR = "#3a3a3a"

# Approximate (x, y) anchor points for each articulation region on the
# stylised side-profile diagram drawn in _draw_base_profile().
REGION_ANCHORS: Dict[str, tuple] = {
    "throat": (0.74, 0.20),
    "palate": (0.40, 0.66),
    "roof": (0.58, 0.70),
    "teeth": (0.18, 0.42),
    "lips": (0.06, 0.36),
}

# Text label offset (dx, dy) per region, tuned so labels don't collide
# on the compact diagram.
REGION_LABEL_OFFSETS: Dict[str, tuple] = {
    "throat": (0.0, -0.10),
    "palate": (-0.03, -0.10),
    "roof": (0.10, 0.07),
    "teeth": (0.0, -0.10),
    "lips": (-0.02, -0.10),
}

REGION_LABELS: Dict[str, str] = {
    "throat": "Throat",
    "palate": "Hard Palate",
    "roof": "Roof of Mouth",
    "teeth": "Teeth",
    "lips": "Lips",
}


class ArticulationWidget(ctk.CTkFrame):
    """Side-profile head diagram with a highlightable articulation point."""

    def __init__(self, master, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._highlight_dots = {}
        self._build_figure()

    def _build_figure(self) -> None:
        self.figure = Figure(figsize=(3.4, 3.0), dpi=100, facecolor=FIGURE_BG)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_facecolor(AXES_BG)
        self.axes.set_xlim(0, 1)
        self.axes.set_ylim(0, 1)
        self.axes.axis("off")

        self._draw_base_profile()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas.draw()

    def _draw_base_profile(self) -> None:
        """Draw a simplified head/mouth outline using basic shapes."""
        # Head outline (rounded box standing in for a side-profile head)
        head = FancyBboxPatch(
            (0.05, 0.15), 0.75, 0.7,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=1.5, edgecolor=OUTLINE_COLOR, facecolor="none",
        )
        self.axes.add_patch(head)

        # Mouth cavity line
        self.axes.plot([0.10, 0.55], [0.40, 0.40], color=OUTLINE_COLOR, linewidth=1.0, alpha=0.6)

        # Region markers, dimmed by default; recolored on highlight()
        for region, (x, y) in REGION_ANCHORS.items():
            dot = Circle((x, y), 0.035, facecolor=DEFAULT_COLOR, edgecolor=OUTLINE_COLOR, linewidth=1.0, zorder=3)
            self.axes.add_patch(dot)
            dx, dy = REGION_LABEL_OFFSETS.get(region, (0.0, -0.09))
            self.axes.text(x + dx, y + dy, REGION_LABELS[region], color="#9a9a9a",
                            fontsize=7, ha="center", va="top")
            self._highlight_dots[region] = dot

    def highlight(self, region: str) -> None:
        """Highlight the given articulation region (e.g. 'throat', 'lips')."""
        if region not in self._highlight_dots:
            logger.warning("Unknown articulation region '%s'; nothing highlighted.", region)
            return

        for r, dot in self._highlight_dots.items():
            dot.set_facecolor(HIGHLIGHT_COLOR if r == region else DEFAULT_COLOR)
            dot.set_radius(0.05 if r == region else 0.035)

        self.canvas.draw_idle()

    def reset(self) -> None:
        """Clear any active highlight."""
        for dot in self._highlight_dots.values():
            dot.set_facecolor(DEFAULT_COLOR)
            dot.set_radius(0.035)
        self.canvas.draw_idle()
