"""Integration test: full pipeline on a 5-second synthetic video (no real model/API)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from basketball_analyzer.config import AnalyzerConfig
from tests.fixtures.generate_test_video import generate_test_video


@pytest.fixture(scope="module")
def test_video(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("video")
    return generate_test_video(tmp / "test_game.mp4", duration_sec=3.0, fps=15.0)


def _mock_yolo_results():
    """Return a minimal YOLO result mock with no detections."""
    result = MagicMock()
    result.boxes = MagicMock()
    result.boxes.id = None
    result.boxes.__iter__ = MagicMock(return_value=iter([]))
    result.boxes.__len__ = MagicMock(return_value=0)
    return [result]


@patch("basketball_analyzer.detection.tracker.YOLO")
def test_pipeline_produces_report(MockYOLO, test_video, tmp_path):
    MockYOLO.return_value.track.return_value = _mock_yolo_results()

    from basketball_analyzer.pipeline.runner import PipelineRunner

    cfg = AnalyzerConfig(
        output_dir=tmp_path / "output",
        frame_sample_rate=5,
        model_name="yolov8n.pt",
    )
    runner = PipelineRunner(cfg)
    result = runner.run(
        video_path=test_video,
        extract_highlights=False,
        generate_commentary=False,
        html_report=False,
    )

    report_path = Path(result["report_path"])
    assert report_path.exists(), "report.json was not created"

    report = json.loads(report_path.read_text())
    assert "game" in report
    assert report["game"]["duration_sec"] > 0
    assert "shots" in report
    assert "highlights" in report


@patch("basketball_analyzer.detection.tracker.YOLO")
def test_pipeline_duration_approx(MockYOLO, test_video, tmp_path):
    MockYOLO.return_value.track.return_value = _mock_yolo_results()

    from basketball_analyzer.pipeline.runner import PipelineRunner

    cfg = AnalyzerConfig(output_dir=tmp_path / "output2", frame_sample_rate=5)
    runner = PipelineRunner(cfg)
    result = runner.run(test_video, extract_highlights=False, generate_commentary=False)

    stats = result["stats"]
    assert abs(stats.duration_sec - 3.0) < 1.0
