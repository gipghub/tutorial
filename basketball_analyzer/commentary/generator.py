from __future__ import annotations

import json
import os

from anthropic import Anthropic

from basketball_analyzer.config import AnalyzerConfig
from basketball_analyzer.highlights.extractor import HighlightSegment
from basketball_analyzer.stats.calculator import GameStats

SYSTEM_PROMPT = """You are an expert basketball sports commentator analyzing a game \
captured by an XBOTGO Falcon AI tracking camera. You receive structured game event data \
in JSON format and produce engaging, accurate sports commentary.

Your commentary should:
- Use natural, energetic sports broadcasting language
- Reference specific timestamps and player names when available
- Note patterns like fast breaks, defensive stops, and momentum shifts
- For play-by-play: describe each event in 1-2 sentences
- For the summary: provide a 3-5 paragraph narrative of the game's key moments

Format your play-by-play as a numbered list with timestamps.
Format the summary as flowing prose paragraphs.
Do not invent facts not supported by the data."""


class CommentaryGenerator:
    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def generate(
        self, stats: GameStats, highlights: list[HighlightSegment]
    ) -> dict[str, str]:
        payload = self._build_payload(stats, highlights)

        play_by_play = self._call_claude(
            f"Generate play-by-play commentary for these basketball game events:\n\n{payload}"
        )
        summary = self._call_claude(
            f"Generate a game summary based on these basketball game events:\n\n{payload}"
        )

        return {"play_by_play": play_by_play, "summary": summary}

    def _call_claude(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.config.claude_model,
            max_tokens=self.config.claude_max_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text  # type: ignore[union-attr]

    def _build_payload(self, stats: GameStats, highlights: list[HighlightSegment]) -> str:
        shots = [
            {
                "type": "shot_attempt",
                "timestamp": _fmt(s.timestamp_sec),
                "toward_hoop": s.toward_hoop,
                "player": s.player_label,
                "scored": s.is_scored,
            }
            for s in stats.shot_attempts
        ]

        segs = [
            {
                "type": "highlight",
                "start": _fmt(h.start_sec),
                "end": _fmt(h.end_sec),
                "duration_sec": round(h.end_sec - h.start_sec, 1),
                "excitement": round(h.peak_excitement, 2),
            }
            for h in highlights
        ]

        top_possession = sorted(
            stats.possession_by_player.items(), key=lambda x: x[1], reverse=True
        )[:5]

        payload = {
            "duration": _fmt(stats.duration_sec),
            "shot_attempts": len(stats.shot_attempts),
            "events_per_minute": round(stats.events_per_minute, 2),
            "top_possession_holders": [
                {"player": p, "seconds": round(s, 1)} for p, s in top_possession
            ],
            "shots": shots,
            "highlight_segments": segs,
        }
        return json.dumps(payload, indent=2)


def _fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"
