"""Integration tests for the Flask web app using the test client."""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from basketball_analyzer.config import AnalyzerConfig
from basketball_analyzer.web.app import JOBS, create_app
from tests.fixtures.generate_test_video import generate_test_video


@pytest.fixture(scope="module")
def test_video_bytes(tmp_path_factory) -> bytes:
    tmp = tmp_path_factory.mktemp("webvideo")
    path = generate_test_video(tmp / "game.mp4", duration_sec=1.0, fps=10.0)
    return path.read_bytes()


@pytest.fixture
def client():
    cfg = AnalyzerConfig(output_dir=Path("/tmp/test_web_output"))
    app = create_app(cfg)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_index_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Basketball" in resp.data


def test_upload_missing_video_returns_400(client):
    resp = client.post("/upload", data={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data


def test_upload_valid_video_returns_job_id(client, test_video_bytes):
    with patch("basketball_analyzer.web.app._run_job"):
        resp = client.post(
            "/upload",
            data={"video": (io.BytesIO(test_video_bytes), "game.mp4")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "job_id" in data


def test_status_unknown_job_returns_404(client):
    resp = client.get("/status/nonexistent-job-id")
    assert resp.status_code == 404


def test_status_queued_job_returns_status(client, test_video_bytes):
    with patch("basketball_analyzer.web.app._run_job"):
        resp = client.post(
            "/upload",
            data={"video": (io.BytesIO(test_video_bytes), "game.mp4")},
            content_type="multipart/form-data",
        )
    job_id = resp.get_json()["job_id"]
    status_resp = client.get(f"/status/{job_id}")
    assert status_resp.status_code == 200
    data = status_resp.get_json()
    assert "status" in data
