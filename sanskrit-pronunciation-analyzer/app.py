"""
app.py
------
Entry point for the Sanskrit Alphabet Pronunciation Analyzer desktop app.

Run with:
    python app.py
"""

import sys

from gui.main_window import SanskritPronunciationApp
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    """Launch the CustomTkinter application. Returns a process exit code."""
    try:
        app = SanskritPronunciationApp()
        logger.info("Application window created. Starting main loop.")
        app.mainloop()
        return 0
    except Exception:
        logger.exception("Fatal error while running the application")
        return 1


if __name__ == "__main__":
    sys.exit(main())
