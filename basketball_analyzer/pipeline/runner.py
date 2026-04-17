from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from basketball_analyzer.config import AnalyzerConfig
from basketball_analyzer.video.processor import VideoProcessor
from basketball_analyzer.video.composer import VideoComposer
from basketball_analyzer.detection.tracker import PlayerTracker
from basketball_analyzer.roster.loader import load_roster
from basketball_analyzer.roster.jersey_ocr import JerseyOCR, JerseyResolver
from basketball_analyzer.stats.calculator import StatsCalculator, GameStats
from basketball_analyzer.highlights.extractor import HighlightExtractor, HighlightSegment
from basketball_analyzer.commentary.generator import CommentaryGenerator
from basketball_analyzer.tts.narrator import NarrationGenerator
from basketball_analyzer.reports.json_reporter import JsonReporter
from basketball_analyzer.reports.html_reporter import HtmlReporter


class PipelineRunner:
    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config
        self.console = Console()

    def run(
        self,
        video_path: Path,
        roster_path: Path | None = None,
        extract_highlights: bool = True,
        generate_commentary: bool = True,
        html_report: bool = False,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> dict:
        cfg = self.config
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        self.console.print(f"[bold green]{cfg.app_name}[/bold green]")
        self.console.print(f"Video: {video_path}")

        roster = load_roster(roster_path)
        if roster:
            self.console.print(
                f"Roster: {len(roster.players)} players, {len(roster.teams)} teams"
            )

        # Stage 1: Video metadata
        video = VideoProcessor(video_path)
        self.console.print(
            f"Duration: {video.meta.duration_sec:.1f}s  "
            f"FPS: {video.meta.fps:.1f}  "
            f"Size: {video.meta.width}x{video.meta.height}"
        )

        # Stage 2: Detection + tracking + OCR
        tracker = PlayerTracker(cfg)
        stats_calc = StatsCalculator(cfg)
        highlight_ext = HighlightExtractor(cfg)

        jersey_ocr: JerseyOCR | None = None
        resolver = JerseyResolver(roster, vote_threshold=cfg.jersey_vote_threshold)

        total_sample = video.meta.total_frames // cfg.frame_sample_rate or 1
        processed = 0

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                f"Analyzing (every {cfg.frame_sample_rate}th frame)...",
                total=total_sample,
            )
            for frame in video.iter_frames(sample_rate=cfg.frame_sample_rate):
                tf = tracker.process_frame(frame)

                # Jersey OCR — lazy-init on first need
                for player in tf.players:
                    if tracker.should_run_ocr(player.track_id):
                        if jersey_ocr is None:
                            try:
                                jersey_ocr = JerseyOCR()
                            except Exception:
                                jersey_ocr = None  # easyocr unavailable; skip silently
                        if jersey_ocr:
                            crop = player.bbox.crop(frame.image)
                            number = jersey_ocr.read_number(crop)
                            resolver.vote(player.track_id, number)

                # Update labels in stats calculator
                for player in tf.players:
                    label = resolver.get_label(player.track_id)
                    stats_calc.set_label(player.track_id, label)

                stats_calc.accumulate(tf)
                highlight_ext.score_frame(tf)
                processed += 1
                progress.advance(task)

                if progress_callback:
                    progress_callback("analyzing", processed, total_sample)

        self.console.print(f"Processed {processed} frames")

        # Stage 3: Stats finalization
        stats = stats_calc.finalize(
            duration_sec=video.meta.duration_sec,
            total_frames=processed,
        )

        # Build per-player heatmaps from normalized history
        for track_id, history in tracker.player_history.items():
            label = resolver.get_label(track_id)
            positions_norm = [(x, y) for x, y, _ in history]
            if positions_norm:
                stats.player_heatmaps[label] = stats_calc.build_heatmap_from_normalized(
                    positions_norm
                )

        # Recompute combined heatmap from normalized data
        import numpy as np

        combined = np.zeros((StatsCalculator.GRID_SIZE, StatsCalculator.GRID_SIZE), dtype=float)
        for grid in stats.player_heatmaps.values():
            combined += grid
        if combined.max() > 0:
            combined /= combined.max()
        stats.combined_heatmap = combined

        self.console.print(
            f"Stats: {len(stats.shot_attempts)} shots, "
            f"{stats.events_per_minute:.1f} events/min"
        )

        # Stage 4: Highlights
        highlight_segments: list[HighlightSegment] = []
        if extract_highlights:
            for shot in stats.shot_attempts:
                highlight_ext.register_shot(shot.timestamp_sec)
            segments = highlight_ext.find_segments()
            clips_dir = cfg.output_dir / cfg.highlight_clips_dir
            highlight_segments = highlight_ext.extract_clips(segments, video, clips_dir)
            self.console.print(f"Highlights: {len(highlight_segments)} segments")

            # Apply watermark if configured
            if cfg.watermark_text and highlight_segments:
                composer = VideoComposer()
                for seg in highlight_segments:
                    if seg.clip_path and seg.clip_path.exists():
                        wm_path = seg.clip_path.with_stem(seg.clip_path.stem + "_wm")
                        VideoComposer.add_watermark(
                            seg.clip_path,
                            wm_path,
                            cfg.watermark_text,
                            cfg.watermark_position,
                            cfg.org_logo_path,
                        )
                        seg.clip_path.unlink()
                        wm_path.rename(seg.clip_path)

            if progress_callback:
                progress_callback("highlights_done", len(highlight_segments), len(highlight_segments))

        # Stage 5: Commentary
        commentary: dict[str, str] = {}
        if generate_commentary:
            try:
                gen = CommentaryGenerator(cfg)
                commentary = gen.generate(stats, highlight_segments)
                commentary_path = cfg.output_dir / "commentary.txt"
                commentary_path.write_text(
                    f"=== PLAY-BY-PLAY ===\n\n{commentary['play_by_play']}\n\n"
                    f"=== GAME SUMMARY ===\n\n{commentary['summary']}\n"
                )
                self.console.print(f"Commentary: {commentary_path}")
            except Exception as exc:
                self.console.print(f"[yellow]Commentary skipped: {exc}[/yellow]")

        # Stage 6: Voice-over
        if generate_commentary and commentary and highlight_segments:
            try:
                narrator = NarrationGenerator(cfg)
                mp3_paths = narrator.generate_for_highlights(
                    highlight_segments,
                    commentary.get("play_by_play", ""),
                    cfg.output_dir / cfg.highlight_clips_dir,
                )
                for seg, mp3_path in zip(highlight_segments, mp3_paths):
                    if seg.clip_path and seg.clip_path.exists() and mp3_path.exists():
                        narrated = seg.clip_path.with_stem(
                            seg.clip_path.stem + "_narrated"
                        )
                        VideoComposer.overlay_audio(seg.clip_path, mp3_path, narrated)
                        seg.narrated_clip_path = narrated
                self.console.print("Voice-over: applied to highlight clips")
            except Exception as exc:
                self.console.print(f"[yellow]Voice-over skipped: {exc}[/yellow]")

        # Stage 7: Reports
        reporter = JsonReporter(cfg)
        report_path = reporter.write(stats, highlight_segments, commentary)
        self.console.print(f"JSON report: {report_path}")

        html_path: Path | None = None
        if html_report:
            html_rep = HtmlReporter(cfg)
            html_path = html_rep.write(stats, highlight_segments, commentary, roster)
            self.console.print(f"HTML report: {html_path}")

        # Build highlights zip
        zip_path = self._zip_highlights(highlight_segments)

        self.console.print("[bold green]Done![/bold green]")

        if progress_callback:
            progress_callback("done", 1, 1)

        return {
            "stats": stats,
            "highlights": highlight_segments,
            "commentary": commentary,
            "report_path": report_path,
            "html_path": html_path,
            "zip_path": zip_path,
        }

    def _zip_highlights(self, segments: list[HighlightSegment]) -> Path | None:
        clips = [
            p
            for seg in segments
            for p in [seg.narrated_clip_path or seg.clip_path]
            if p and p.exists()
        ]
        if not clips:
            return None
        zip_path = self.config.output_dir / "highlights.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for clip in clips:
                zf.write(clip, clip.name)
        return zip_path
