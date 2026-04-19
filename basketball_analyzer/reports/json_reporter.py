from __future__ import annotations

import json
from pathlib import Path

from basketball_analyzer.config import AnalyzerConfig
from basketball_analyzer.highlights.extractor import HighlightSegment
from basketball_analyzer.stats.calculator import GameStats


class JsonReporter:
    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config

    def write(
        self,
        stats: GameStats,
        highlights: list[HighlightSegment],
        commentary: dict[str, str],
    ) -> Path:
        report = {
            "game": {
                "duration_sec": round(stats.duration_sec, 1),
                "total_frames_analyzed": stats.total_frames_analyzed,
                "events_per_minute": round(stats.events_per_minute, 2),
            },
            "shots": [
                {
                    "timestamp_sec": round(s.timestamp_sec, 1),
                    "timestamp": _fmt(s.timestamp_sec),
                    "toward_hoop": s.toward_hoop,
                    "scored": s.is_scored,
                    "player": s.player_label,
                    "ball_position": list(s.ball_position),
                }
                for s in stats.shot_attempts
            ],
            "possession": {
                label: round(sec, 1)
                for label, sec in sorted(
                    stats.possession_by_player.items(), key=lambda x: x[1], reverse=True
                )
            },
            "highlights": [
                {
                    "start_sec": round(h.start_sec, 1),
                    "end_sec": round(h.end_sec, 1),
                    "duration_sec": round(h.end_sec - h.start_sec, 1),
                    "peak_excitement": round(h.peak_excitement, 2),
                    "event_type": h.event_type,
                    "clip_path": str(h.clip_path) if h.clip_path else None,
                    "narrated_clip_path": (
                        str(h.narrated_clip_path) if h.narrated_clip_path else None
                    ),
                }
                for h in highlights
            ],
            "heatmap": {
                "combined": stats.combined_heatmap.tolist(),
            },
            "commentary": commentary,
        }

        out_path = self.config.output_dir / "report.json"
        out_path.write_text(json.dumps(report, indent=2))
        return out_path


def _fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"
