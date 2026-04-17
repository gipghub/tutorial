from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from basketball_analyzer.commentary.generator import CommentaryGenerator, _fmt
from basketball_analyzer.config import AnalyzerConfig
from basketball_analyzer.highlights.extractor import HighlightSegment
from basketball_analyzer.stats.calculator import GameStats, ShotEvent


def make_stats() -> GameStats:
    stats = GameStats(duration_sec=600.0, total_frames_analyzed=3600)
    stats.shot_attempts = [
        ShotEvent(30.0, 900, (0.82, 0.35), "right", player_label="Alice (SF #23)"),
        ShotEvent(90.0, 2700, (0.18, 0.35), "left",  player_label="Bob (PG #5)"),
    ]
    stats.possession_by_player = {"Alice (SF #23)": 120.0, "Bob (PG #5)": 80.0}
    stats.events_per_minute = 0.2
    return stats


def make_highlights() -> list[HighlightSegment]:
    return [
        HighlightSegment(28.0, 35.0, 7.5, "combined"),
        HighlightSegment(88.0, 96.0, 6.0, "combined"),
    ]


def test_format_time():
    assert _fmt(0.0) == "00:00"
    assert _fmt(65.0) == "01:05"
    assert _fmt(125.0) == "02:05"


def test_build_payload_valid_json():
    cfg = AnalyzerConfig()
    gen = CommentaryGenerator.__new__(CommentaryGenerator)
    gen.config = cfg
    payload = gen._build_payload(make_stats(), make_highlights())
    data = json.loads(payload)
    assert "shots" in data
    assert "highlight_segments" in data
    assert data["shot_attempts"] == 2


def test_call_claude_uses_cache_control():
    cfg = AnalyzerConfig()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Great game!")]
    mock_client.messages.create.return_value = mock_response

    gen = CommentaryGenerator.__new__(CommentaryGenerator)
    gen.config = cfg
    gen.client = mock_client

    result = gen._call_claude("Test prompt")

    call_kwargs = mock_client.messages.create.call_args.kwargs
    system = call_kwargs["system"]
    assert isinstance(system, list)
    assert any(
        block.get("cache_control", {}).get("type") == "ephemeral"
        for block in system
        if isinstance(block, dict)
    )
    assert result == "Great game!"
