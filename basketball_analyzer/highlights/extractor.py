from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from basketball_analyzer.config import AnalyzerConfig
from basketball_analyzer.detection.tracker import TrackFrame
from basketball_analyzer.video.processor import VideoProcessor


@dataclass
class HighlightSegment:
    start_sec: float
    end_sec: float
    peak_excitement: float
    event_type: str  # "fast_ball", "player_cluster", "shot", "combined"
    clip_path: Path | None = None
    narrated_clip_path: Path | None = None


class HighlightExtractor:
    """
    Excitement scoring per frame:
      - Ball speed above threshold: +3 pts
      - Player clustering (>=4 within 150px): +2 pts
      - Shot event within 2 sec: +5 pts

    Scores are smoothed over a 2-second rolling window then thresholded.
    """

    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config
        self._frame_scores: list[tuple[float, float]] = []  # (timestamp, score)
        self._shot_timestamps: set[float] = set()
        self._shot_players: list[tuple[float, str]] = []  # (timestamp, player_label)

    def register_shot(self, timestamp_sec: float, player_label: str = "") -> None:
        self._shot_timestamps.add(timestamp_sec)
        self._shot_players.append((timestamp_sec, player_label))

    def score_frame(self, tf: TrackFrame) -> float:
        score = 0.0
        cfg = self.config

        if tf.ball_trajectory.speed > cfg.highlight_ball_speed_threshold:
            score += 3.0

        positions = [p.bbox.center for p in tf.players]
        if len(positions) >= 2:
            if self._count_clustered(positions, radius=150.0) >= cfg.highlight_player_cluster_threshold:
                score += 2.0

        for shot_t in self._shot_timestamps:
            if abs(tf.timestamp_sec - shot_t) <= 2.0:
                score += 5.0
                break

        self._frame_scores.append((tf.timestamp_sec, score))
        return score

    def _count_clustered(self, positions: list[tuple[float, float]], radius: float) -> int:
        arr = np.array(positions)
        max_cluster = 0
        for pos in arr:
            dists = np.linalg.norm(arr - pos, axis=1)
            max_cluster = max(max_cluster, int(np.sum(dists < radius)))
        return max_cluster

    def find_segments(self) -> list[HighlightSegment]:
        if not self._frame_scores:
            return []

        timestamps = np.array([t for t, _ in self._frame_scores])
        scores = np.array([s for _, s in self._frame_scores])

        series = pd.Series(scores, index=pd.to_datetime(timestamps, unit="s"))
        smoothed = (
            series.rolling("2s", min_periods=1).mean().fillna(0).values
        )

        THRESHOLD = 1.5
        exciting = smoothed > THRESHOLD
        segments: list[HighlightSegment] = []
        in_seg = False
        seg_start = 0.0
        peak = 0.0

        for i, (t, is_exc) in enumerate(zip(timestamps, exciting)):
            if is_exc and not in_seg:
                seg_start = t
                peak = float(smoothed[i])
                in_seg = True
            elif in_seg:
                if is_exc:
                    peak = max(peak, float(smoothed[i]))
                else:
                    duration = timestamps[i - 1] - seg_start
                    if duration >= self.config.highlight_min_duration_sec:
                        segments.append(
                            HighlightSegment(
                                start_sec=max(
                                    0.0, seg_start - self.config.highlight_padding_sec
                                ),
                                end_sec=timestamps[i - 1] + self.config.highlight_padding_sec,
                                peak_excitement=peak,
                                event_type="combined",
                            )
                        )
                    in_seg = False

        if in_seg and len(timestamps) > 0:
            segments.append(
                HighlightSegment(
                    start_sec=max(0.0, seg_start - self.config.highlight_padding_sec),
                    end_sec=timestamps[-1] + self.config.highlight_padding_sec,
                    peak_excitement=peak,
                    event_type="combined",
                )
            )

        merged = self._merge_segments(segments, gap_threshold=3.0)

        filter_players = self.config.highlight_players
        if not filter_players:
            return merged

        # Keep only segments that contain a shot by one of the target players
        targets = [p.lower() for p in filter_players]
        filtered = []
        for seg in merged:
            for shot_t, label in self._shot_players:
                if seg.start_sec <= shot_t <= seg.end_sec:
                    if any(t in label.lower() for t in targets):
                        filtered.append(seg)
                        break
        return filtered

    def _merge_segments(
        self, segments: list[HighlightSegment], gap_threshold: float
    ) -> list[HighlightSegment]:
        if not segments:
            return []
        merged = [segments[0]]
        for seg in segments[1:]:
            if seg.start_sec - merged[-1].end_sec < gap_threshold:
                merged[-1] = HighlightSegment(
                    start_sec=merged[-1].start_sec,
                    end_sec=max(merged[-1].end_sec, seg.end_sec),
                    peak_excitement=max(merged[-1].peak_excitement, seg.peak_excitement),
                    event_type="combined",
                )
            else:
                merged.append(seg)
        return merged

    def extract_clips(
        self,
        segments: list[HighlightSegment],
        video_processor: VideoProcessor,
        output_dir: Path,
    ) -> list[HighlightSegment]:
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, seg in enumerate(segments):
            clip_path = output_dir / f"highlight_{i+1:02d}_{seg.start_sec:.0f}s.mp4"
            video_processor.extract_clip(seg.start_sec, seg.end_sec, clip_path)
            seg.clip_path = clip_path
        return segments
