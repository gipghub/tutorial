from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from basketball_analyzer.roster.jersey_ocr import JerseyOCR, JerseyResolver
from basketball_analyzer.roster.models import Player, Roster, Team


def make_roster() -> Roster:
    return Roster(
        teams=[Team(name="Home", short_name="HME")],
        players=[
            Player(number=23, name="Alice", position="SF", team="Home"),
            Player(number=5,  name="Bob",   position="PG", team="Home"),
        ],
    )


def test_resolver_locks_after_vote_threshold():
    roster = make_roster()
    resolver = JerseyResolver(roster, vote_threshold=3)

    resolver.vote(track_id=1, number=23)
    resolver.vote(track_id=1, number=23)
    assert resolver.get_player(1) is None  # not yet locked

    resolver.vote(track_id=1, number=23)
    player = resolver.get_player(1)
    assert player is not None
    assert player.name == "Alice"


def test_resolver_no_lock_on_conflicting_votes():
    roster = make_roster()
    resolver = JerseyResolver(roster, vote_threshold=3)

    resolver.vote(track_id=2, number=5)
    resolver.vote(track_id=2, number=23)  # conflict
    resolver.vote(track_id=2, number=5)
    # Only 2 votes for #5 — not enough
    assert resolver.get_player(2) is None


def test_resolver_label_without_roster():
    resolver = JerseyResolver(None, vote_threshold=2)
    resolver.vote(track_id=7, number=10)
    resolver.vote(track_id=7, number=10)
    label = resolver.get_label(7)
    assert "#10" in label


def test_resolver_fallback_label_for_unknown():
    resolver = JerseyResolver(None)
    label = resolver.get_label(42)
    assert "42" in label


def test_jersey_ocr_read_number_mocked():
    with patch("easyocr.Reader") as MockReader:
        instance = MockReader.return_value
        instance.readtext.return_value = ["23"]

        ocr = JerseyOCR.__new__(JerseyOCR)
        ocr._reader = instance

        crop = np.zeros((100, 60, 3), dtype=np.uint8)
        result = ocr.read_number(crop)
        assert result == 23


def test_jersey_ocr_ignores_out_of_range():
    with patch("easyocr.Reader") as MockReader:
        instance = MockReader.return_value
        instance.readtext.return_value = ["0"]  # 0 is out of range 1-99

        ocr = JerseyOCR.__new__(JerseyOCR)
        ocr._reader = instance

        crop = np.zeros((100, 60, 3), dtype=np.uint8)
        result = ocr.read_number(crop)
        assert result is None
