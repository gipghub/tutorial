from __future__ import annotations

from collections import deque

import pytest

from basketball_analyzer.config import AnalyzerConfig
from basketball_analyzer.detection.tracker import BallTrajectory, BoundingBox, Detection, TrackFrame
from basketball_analyzer.highlights.extractor import HighlightExtractor, HighlightSegment


def make_frame(timestamp: float, ball_speed: float = 0.0, n_players: int = 0) -> TrackFrame:
    # Simulate a ball trajectory with the given speed (normalized)
    if ball_speed > 0:
        traj = BallTrajectory(
            positions=deque([(0.5, 0.5), (0.5 + ball_speed * 0.1, 0.5)]),
            timestamps=deque([timestamp - 0.1, timestamp]),
        )
    else:
        traj = BallTrajectory()

    ball = Detection(99, 32, 0.9, BoundingBox(100, 100, 120, 120), is_ball=True)
    players = [
        Detection(i, 0, 0.9, BoundingBox(10 * i, 10, 10 * i + 8, 50), is_player=True)
        for i in range(n_players)
    ]
    return TrackFrame(
        frame_index=int(timestamp * 30),
        timestamp_sec=timestamp,
        players=players,
        ball=ball,
        ball_trajectory=traj,
    )


def test_no_segments_from_boring_frames(config):
    ext = HighlightExtractor(config)
    for t in range(20):
        ext.score_frame(make_frame(float(t)))
    segments = ext.find_segments()
    assert segments == []


def test_shot_boosts_score(config):
    ext = HighlightExtractor(config)
    ext.register_shot(5.0)
    score = ext.score_frame(make_frame(5.0))
    assert score >= 5.0


def test_fast_ball_boosts_score(config):
    cfg = AnalyzerConfig(highlight_ball_speed_threshold=0.05)
    ext = HighlightExtractor(cfg)
    score = ext.score_frame(make_frame(1.0, ball_speed=0.2))
    assert score >= 3.0


def test_segment_detected_during_high_excitement(config):
    cfg = AnalyzerConfig(highlight_min_duration_sec=2.0, highlight_padding_sec=0.0)
    ext = HighlightExtractor(cfg)
    ext.register_shot(5.0)
    for t in range(20):
        ext.score_frame(make_frame(float(t), ball_speed=0.3))
    segments = ext.find_segments()
    assert len(segments) >= 1


def test_merge_adjacent_segments():
    cfg = AnalyzerConfig()
    ext = HighlightExtractor(cfg)
    segs = [
        HighlightSegment(0.0, 5.0, 3.0, "combined"),
        HighlightSegment(6.0, 10.0, 2.0, "combined"),  # 1s gap < 3s threshold
    ]
    merged = ext._merge_segments(segs, gap_threshold=3.0)
    assert len(merged) == 1
    assert merged[0].start_sec == 0.0
    assert merged[0].end_sec == 10.0


def test_separate_segments_not_merged():
    cfg = AnalyzerConfig()
    ext = HighlightExtractor(cfg)
    segs = [
        HighlightSegment(0.0, 5.0, 3.0, "combined"),
        HighlightSegment(10.0, 15.0, 2.0, "combined"),  # 5s gap > 3s threshold
    ]
    merged = ext._merge_segments(segs, gap_threshold=3.0)
    assert len(merged) == 2
