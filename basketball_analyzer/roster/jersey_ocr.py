from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from basketball_analyzer.roster.models import Player, Roster


class JerseyOCR:
    """Reads jersey numbers from player bounding box images using EasyOCR."""

    def __init__(self) -> None:
        # Lazy-import so tests can mock it without the heavy GPU dependency
        import easyocr  # type: ignore
        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def read_number(self, player_crop: np.ndarray) -> int | None:
        """Extract a jersey number (1-99) from an upper-chest crop of a player bbox."""
        if player_crop.size == 0:
            return None
        h, w = player_crop.shape[:2]
        # Focus on the upper 40% of the crop where numbers typically appear
        chest = player_crop[: int(h * 0.40), :]
        if chest.size == 0:
            return None

        results = self._reader.readtext(chest, detail=0, allowlist="0123456789")
        for text in results:
            cleaned = re.sub(r"\D", "", text)
            if cleaned and 1 <= int(cleaned) <= 99:
                return int(cleaned)
        return None


class JerseyResolver:
    """
    Accumulates jersey number votes per track ID and resolves to a Player once
    the same number is seen jersey_vote_threshold times for a given track.
    """

    def __init__(self, roster: Roster | None, vote_threshold: int = 3) -> None:
        self._roster = roster
        self._vote_threshold = vote_threshold
        # track_id → {number: count}
        self._votes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # track_id → resolved Player (locked in)
        self._resolved: dict[int, Player] = {}
        # track_id → fallback label (used when no roster)
        self._labels: dict[int, str] = {}

    def vote(self, track_id: int, number: int | None) -> None:
        if number is None or track_id in self._resolved:
            return
        self._votes[track_id][number] += 1
        if self._votes[track_id][number] >= self._vote_threshold:
            self._lock(track_id, number)

    def _lock(self, track_id: int, number: int) -> None:
        if self._roster:
            player = self._roster.get_player_by_number(number)
            if player:
                self._resolved[track_id] = player
                return
        # No roster match — store jersey number as label
        self._labels[track_id] = f"#{number}"

    def get_player(self, track_id: int) -> Player | None:
        return self._resolved.get(track_id)

    def get_label(self, track_id: int) -> str:
        if track_id in self._resolved:
            return self._resolved[track_id].display_name
        return self._labels.get(track_id, f"Player {track_id}")

    def all_resolved(self) -> dict[int, Player]:
        return dict(self._resolved)
