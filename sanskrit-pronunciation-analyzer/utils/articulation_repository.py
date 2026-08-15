"""
articulation_repository.py
---------------------------
Loads the Sanskrit place-of-articulation data from data/articulation.json
and exposes a simple, typed lookup API. Isolating this behind a repository
class means the GUI and recognition modules never touch raw JSON directly.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from utils.config import ARTICULATION_JSON_PATH
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ArticulationInfo:
    """Structured record describing how/where a Sanskrit alphabet is produced."""
    alphabet: str
    place: str
    sanskrit_term: str
    category: str
    highlight_region: str
    description: str


class ArticulationRepository:
    """
    Loads and serves articulation (place-of-pronunciation) data.

    Usage:
        repo = ArticulationRepository()
        info = repo.get_info("Ka")
    """

    def __init__(self, json_path: Path = ARTICULATION_JSON_PATH) -> None:
        self._json_path = json_path
        self._data: Dict[str, ArticulationInfo] = {}
        self._load()

    def _load(self) -> None:
        """Read the JSON file from disk and parse it into ArticulationInfo objects."""
        try:
            with open(self._json_path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)

            alphabets = raw.get("alphabets", {})
            for name, fields in alphabets.items():
                self._data[name] = ArticulationInfo(
                    alphabet=name,
                    place=fields.get("place", "Unknown"),
                    sanskrit_term=fields.get("sanskrit_term", ""),
                    category=fields.get("category", ""),
                    highlight_region=fields.get("highlight_region", "throat"),
                    description=fields.get("description", ""),
                )
            logger.info("Loaded articulation data for %d alphabets", len(self._data))

        except FileNotFoundError:
            logger.error("Articulation JSON not found at %s", self._json_path)
            raise
        except json.JSONDecodeError as exc:
            logger.error("Malformed articulation JSON: %s", exc)
            raise

    def get_info(self, alphabet: str) -> Optional[ArticulationInfo]:
        """Return articulation info for the given alphabet name, or None if unknown."""
        return self._data.get(alphabet)

    def all_alphabets(self) -> Dict[str, ArticulationInfo]:
        """Return the full mapping of alphabet -> ArticulationInfo."""
        return dict(self._data)
