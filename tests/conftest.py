from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pytest

from basketball_analyzer.config import AnalyzerConfig
from basketball_analyzer.detection.tracker import (
    BallTrajectory,
    BoundingBox,
    Detection,
    TrackFrame,
)


@pytest.fixture
def config() -> AnalyzerConfig:
    return AnalyzerConfig()


@pytest.fixture
def sample_ball_trajectory() -> BallTrajectory:
    return BallTrajectory(
        positions=deque([(0.50, 0.42), (0.51, 0.40), (0.52, 0.36)]),
        timestamps=deque([0.0, 0.1, 0.2]),
    )


@pytest.fixture
def sample_track_frame(sample_ball_trajectory) -> TrackFrame:
    ball = Detection(
        track_id=99,
        class_id=32,
        confidence=0.90,
        bbox=BoundingBox(400, 280, 420, 300),
        is_ball=True,
    )
    players = [
        Detection(
            track_id=i,
            class_id=0,
            confidence=0.85,
            bbox=BoundingBox(100 * i, 200, 100 * i + 50, 350),
            is_player=True,
        )
        for i in range(1, 6)
    ]
    return TrackFrame(
        frame_index=50,
        timestamp_sec=1.67,
        players=players,
        ball=ball,
        ball_trajectory=sample_ball_trajectory,
    )
