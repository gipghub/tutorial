from __future__ import annotations

import asyncio
from pathlib import Path

from basketball_analyzer.config import AnalyzerConfig
from basketball_analyzer.highlights.extractor import HighlightSegment


class NarrationGenerator:
    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config

    def generate_full(self, text: str, output_path: Path) -> Path:
        """Convert full commentary text to speech and save as MP3."""
        asyncio.run(self._save(text, output_path))
        return output_path

    def generate_for_highlights(
        self,
        highlights: list[HighlightSegment],
        play_by_play: str,
        output_dir: Path,
    ) -> list[Path]:
        """
        Split play-by-play commentary into per-highlight chunks and generate
        one MP3 per highlight segment.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        lines = [l.strip() for l in play_by_play.splitlines() if l.strip()]
        n = len(highlights)
        paths: list[Path] = []

        # Distribute commentary lines roughly evenly across highlights
        chunk_size = max(1, len(lines) // n) if n else len(lines)
        chunks = [
            " ".join(lines[i : i + chunk_size]) for i in range(0, len(lines), chunk_size)
        ]

        for i, seg in enumerate(highlights):
            text_chunk = chunks[i] if i < len(chunks) else f"Highlight number {i+1}."
            mp3_path = output_dir / f"highlight_{i+1:02d}_narration.mp3"
            asyncio.run(self._save(text_chunk, mp3_path))
            paths.append(mp3_path)

        return paths

    async def _save(self, text: str, output_path: Path) -> None:
        import edge_tts  # type: ignore
        communicate = edge_tts.Communicate(text, self.config.tts_voice)
        await communicate.save(str(output_path))
