from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from basketball_analyzer.config import AnalyzerConfig
from basketball_analyzer.detection.tracker import BallTrajectory, BoundingBox, Detection, TrackFrame
from basketball_analyzer.stats.calculator import StatsCalculator


def make_frame(
    timestamp: float,
    ball_positions: list[tuple[float, float]],
    ball_timestamps: list[float],
    players: list[Detection] | None = None,
) -> TrackFrame:
    traj = BallTrajectory(
        positions=deque(ball_positions),
        timestamps=deque(ball_timestamps),
    )
    ball = Detection(
        track_id=99, class_id=32, confidence=0.9,
        bbox=BoundingBox(100, 100, 120, 120), is_ball=True,
    )
    return TrackFrame(
        frame_index=int(timestamp * 30),
        timestamp_sec=timestamp,
        players=players or [],
        ball=ball,
        ball_trajectory=traj,
    )


def test_shot_detected_near_hoop(config):
    calc = StatsCalculator(config)
    # Ball moving upward (vy < -0.02) and near hoop_right (0.82, 0.35)
    tf = make_frame(
        timestamp=5.0,
        ball_positions=[(0.82, 0.40), (0.82, 0.37), (0.82, 0.34)],
        ball_timestamps=[4.8, 4.9, 5.0],
    )
    calc.accumulate(tf)
    assert len(calc._pending_shots) == 1
    assert calc._pending_shots[0].toward_hoop == "right"


def test_shot_not_detected_during_cooldown(config):
    calc = StatsCalculator(config)
    tf1 = make_frame(
        timestamp=5.0,
        ball_positions=[(0.82, 0.40), (0.82, 0.37), (0.82, 0.34)],
        ball_timestamps=[4.8, 4.9, 5.0],
    )
    tf2 = make_frame(
        timestamp=6.0,  # within 3s cooldown
        ball_positions=[(0.82, 0.40), (0.82, 0.37), (0.82, 0.34)],
        ball_timestamps=[5.8, 5.9, 6.0],
    )
    calc.accumulate(tf1)
    calc.accumulate(tf2)
    assert len(calc._pending_shots) == 1  # still just one


def test_shot_not_detected_far_from_hoop(config):
    calc = StatsCalculator(config)
    # Ball moving upward but in center court — too far from any hoop
    tf = make_frame(
        timestamp=5.0,
        ball_positions=[(0.50, 0.60), (0.50, 0.57), (0.50, 0.54)],
        ball_timestamps=[4.8, 4.9, 5.0],
    )
    calc.accumulate(tf)
    assert len(calc._pending_shots) == 0


def test_finalize_produces_game_stats(config):
    calc = StatsCalculator(config)
    tf = make_frame(5.0, [(0.82, 0.40), (0.82, 0.37), (0.82, 0.34)], [4.8, 4.9, 5.0])
    calc.accumulate(tf)
    stats = calc.finalize(duration_sec=120.0, total_frames=600)
    assert stats.duration_sec == 120.0
    assert stats.total_frames_analyzed == 600
    assert isinstance(stats.combined_heatmap, type(np.zeros((50, 50))))


def test_heatmap_from_normalized_values_and_shape(config):
    calc = StatsCalculator(config)
    positions = [(0.1, 0.2), (0.5, 0.5), (0.9, 0.8)]
    grid = calc.build_heatmap_from_normalized(positions)
    assert grid.shape == (50, 50)
    assert grid.min() >= 0.0
    assert grid.max() <= 1.0


def test_possession_switches_correctly(config):
    calc = StatsCalculator(config)
    calc.set_label(1, "Alice")
    calc.set_label(2, "Bob")

    # Ball near player 1 at t=0
    player1 = Detection(1, 0, 0.9, BoundingBox(50, 50, 100, 200), is_player=True)
    player2 = Detection(2, 0, 0.9, BoundingBox(500, 50, 550, 200), is_player=True)
    ball = Detection(99, 32, 0.9, BoundingBox(55, 50, 75, 70), is_ball=True)
    traj = BallTrajectory(positions=deque([(0.1, 0.1)]), timestamps=deque([0.0]))
    tf1 = TrackFrame(0, 0.0, [player1, player2], ball, traj)
    calc.accumulate(tf1)

    # Ball near player 2 at t=5
    ball2 = Detection(99, 32, 0.9, BoundingBox(505, 50, 525, 70), is_ball=True)
    traj2 = BallTrajectory(positions=deque([(0.9, 0.1)]), timestamps=deque([5.0]))
    tf2 = TrackFrame(150, 5.0, [player1, player2], ball2, traj2)
    calc.accumulate(tf2)

    stats = calc.finalize(10.0, 300)
    assert "Alice" in stats.possession_by_player
