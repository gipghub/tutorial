from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter  # type: ignore

from basketball_analyzer.config import AnalyzerConfig
from basketball_analyzer.detection.tracker import TrackFrame


@dataclass
class ShotEvent:
    timestamp_sec: float
    frame_index: int
    ball_position: tuple[float, float]  # normalized (x, y)
    toward_hoop: str  # "left" or "right"
    is_scored: bool = False
    player_label: str = "Unknown"


@dataclass
class GameStats:
    duration_sec: float = 0.0
    total_frames_analyzed: int = 0
    shot_attempts: list[ShotEvent] = field(default_factory=list)
    possession_by_player: dict[str, float] = field(default_factory=dict)
    player_heatmaps: dict[str, np.ndarray] = field(default_factory=dict)
    combined_heatmap: np.ndarray = field(
        default_factory=lambda: np.zeros((50, 50), dtype=float)
    )
    events_per_minute: float = 0.0


class StatsCalculator:
    GRID_SIZE = 50
    POSSESSION_PIXEL_THRESHOLD = 150

    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config
        self._shot_cooldown_sec = 3.0
        self._last_shot_time = -99.0
        self._possession_last_holder: str = ""
        self._possession_start: float = 0.0
        self._possession_time: dict[str, float] = {}
        self._player_positions: dict[str, list[tuple[float, float]]] = {}
        self._pending_shots: list[ShotEvent] = []
        # label resolver: track_id -> label (set by pipeline runner)
        self._labels: dict[int, str] = {}

    def set_label(self, track_id: int, label: str) -> None:
        self._labels[track_id] = label

    def _label(self, track_id: int) -> str:
        return self._labels.get(track_id, f"Player {track_id}")

    def accumulate(self, tf: TrackFrame) -> None:
        self._check_for_shot(tf)
        self._update_possession(tf)
        self._accumulate_positions(tf)

    def _check_for_shot(self, tf: TrackFrame) -> None:
        if tf.ball is None:
            return
        t = tf.timestamp_sec
        if t - self._last_shot_time < self._shot_cooldown_sec:
            return
        traj = tf.ball_trajectory
        if traj.velocity is None or len(traj.positions) < 2:
            return

        _, ball_vy = traj.velocity
        ball_nx, ball_ny = traj.positions[-1]

        # For tracking cameras (XBOTGO Falcon etc.) hoop screen coords shift as the
        # camera pans, so we skip proximity gating and rely solely on upward velocity.
        upward = ball_vy < self.config.shot_upward_velocity_threshold

        # Still compute left/right for the report, using whichever half the ball is in.
        toward = "left" if ball_nx < 0.5 else "right"

        if upward:
            self._last_shot_time = t
            shooter_label = "Unknown"
            if tf.players:
                bx, by = tf.ball.bbox.center
                closest = min(
                    tf.players,
                    key=lambda p: np.sqrt(
                        (p.bbox.center[0] - bx) ** 2 + (p.bbox.center[1] - by) ** 2
                    ),
                )
                shooter_label = self._label(closest.track_id)
            self._pending_shots.append(
                ShotEvent(
                    timestamp_sec=t,
                    frame_index=tf.frame_index,
                    ball_position=(ball_nx, ball_ny),
                    toward_hoop=toward,
                    player_label=shooter_label,
                )
            )

    def _update_possession(self, tf: TrackFrame) -> None:
        if tf.ball is None or not tf.players:
            return
        bx, by = tf.ball.bbox.center
        closest = min(
            tf.players,
            key=lambda p: np.sqrt(
                (p.bbox.center[0] - bx) ** 2 + (p.bbox.center[1] - by) ** 2
            ),
        )
        dist = np.sqrt(
            (closest.bbox.center[0] - bx) ** 2 + (closest.bbox.center[1] - by) ** 2
        )
        if dist > self.POSSESSION_PIXEL_THRESHOLD:
            return

        holder_label = self._label(closest.track_id)
        if holder_label != self._possession_last_holder:
            if self._possession_last_holder:
                held = tf.timestamp_sec - self._possession_start
                self._possession_time[self._possession_last_holder] = (
                    self._possession_time.get(self._possession_last_holder, 0.0) + held
                )
            self._possession_last_holder = holder_label
            self._possession_start = tf.timestamp_sec

    def _accumulate_positions(self, tf: TrackFrame) -> None:
        for player in tf.players:
            label = self._label(player.track_id)
            cx, cy = player.bbox.center
            if label not in self._player_positions:
                self._player_positions[label] = []
            self._player_positions[label].append((cx, cy))

    def finalize(self, duration_sec: float, total_frames: int) -> GameStats:
        stats = GameStats(
            duration_sec=duration_sec,
            total_frames_analyzed=total_frames,
            shot_attempts=self._pending_shots,
            possession_by_player=dict(self._possession_time),
        )
        for label, positions in self._player_positions.items():
            grid = self._build_heatmap(positions)
            stats.player_heatmaps[label] = grid
            stats.combined_heatmap = stats.combined_heatmap + grid

        if stats.combined_heatmap.max() > 0:
            stats.combined_heatmap /= stats.combined_heatmap.max()

        n_events = len(stats.shot_attempts)
        if duration_sec > 0:
            stats.events_per_minute = n_events / (duration_sec / 60.0)

        return stats

    def _build_heatmap(self, positions: list[tuple[float, float]]) -> np.ndarray:
        grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=float)
        for x, y in positions:
            gx = min(int(x / 20), self.GRID_SIZE - 1)
            gy = min(int(y / 20), self.GRID_SIZE - 1)
            grid[gy, gx] += 1.0
        if grid.max() > 0:
            grid = gaussian_filter(grid, sigma=1.5)
            grid /= grid.max()
        return grid

    def build_heatmap_from_normalized(
        self, positions_normalized: list[tuple[float, float]]
    ) -> np.ndarray:
        grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=float)
        for nx, ny in positions_normalized:
            gx = min(int(nx * self.GRID_SIZE), self.GRID_SIZE - 1)
            gy = min(int(ny * self.GRID_SIZE), self.GRID_SIZE - 1)
            grid[gy, gx] += 1.0
        if grid.max() > 0:
            grid = gaussian_filter(grid, sigma=1.5)
            grid /= grid.max()
        return grid
