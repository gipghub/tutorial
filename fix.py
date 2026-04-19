# -*- coding: utf-8 -*-
"""Run from C:\\Basketball_Analyzer to overwrite web/app.py and web/templates/index.html"""
from pathlib import Path

app_content = '''\
from __future__ import annotations

import json
import os
import threading
import uuid
import zipfile
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from basketball_analyzer.config import AnalyzerConfig
from basketball_analyzer.pipeline.runner import PipelineRunner
from basketball_analyzer.sharing.email_sender import EmailSender
from basketball_analyzer.sharing.share_link import generate_qr, generate_share_url

UPLOAD_FOLDER = Path("uploads")
JOBS: dict[str, dict] = {}  # job_id -> {status, config, result, error}
ALLOWED_VIDEO = {".mp4", ".mov", ".avi", ".mkv"}
ALLOWED_ROSTER = {".json", ".csv"}


def create_app(config: AnalyzerConfig | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4 GB

    UPLOAD_FOLDER.mkdir(exist_ok=True)

    @app.route("/")
    def index():
        return render_template("index.html", app_name=config.app_name if config else "Basketball Analyzer")

    @app.route("/upload", methods=["POST"])
    def upload():
        try:
            job_id = str(uuid.uuid4())
            job_dir = UPLOAD_FOLDER / job_id
            job_dir.mkdir()

            roster_path: Path | None = None

            local_video = request.form.get("local_video_path", "").strip()
            if local_video:
                video_path = Path(local_video)
                if not video_path.exists():
                    return jsonify({"error": f"File not found: {local_video}"}), 400
                if video_path.suffix.lower() not in ALLOWED_VIDEO:
                    return jsonify({"error": f"Unsupported format: {video_path.suffix}"}), 400
                local_roster = request.form.get("local_roster_path", "").strip()
                if local_roster and Path(local_roster).exists():
                    roster_path = Path(local_roster)
            else:
                video = request.files.get("video")
                roster = request.files.get("roster")
                if not video or not video.filename:
                    return jsonify({"error": "No video file provided"}), 400
                ext = Path(video.filename).suffix.lower()
                if ext not in ALLOWED_VIDEO:
                    return jsonify({"error": f"Unsupported video format: {ext}"}), 400
                video_path = job_dir / secure_filename(video.filename)
                video.save(str(video_path))
                if roster and roster.filename:
                    r_ext = Path(roster.filename).suffix.lower()
                    if r_ext in ALLOWED_ROSTER:
                        roster_path = job_dir / secure_filename(roster.filename)
                        roster.save(str(roster_path))

            cfg = config or AnalyzerConfig()
            cfg.output_dir = job_dir / "output"

            form = request.form
            cfg.frame_sample_rate = int(form.get("frame_rate", cfg.frame_sample_rate))
            cfg.team_home_name = form.get("team_home", cfg.team_home_name)
            cfg.team_away_name = form.get("team_away", cfg.team_away_name)
            cfg.watermark_text = form.get("watermark", cfg.watermark_text)

            JOBS[job_id] = {"status": "queued", "progress": 0, "total": 1, "result": None, "error": None}

            thread = threading.Thread(
                target=_run_job,
                args=(job_id, video_path, roster_path, cfg),
                daemon=True,
            )
            thread.start()

            return jsonify({"job_id": job_id})

        except Exception as exc:
            import traceback
            return jsonify({"error": str(exc), "detail": traceback.format_exc()}), 500

    @app.route("/status/<job_id>")
    def status(job_id: str):
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        return jsonify({
            "status": job["status"],
            "progress": job.get("progress", 0),
            "total": job.get("total", 1),
            "error": job.get("error"),
        })

    @app.route("/results/<job_id>")
    def results(job_id: str):
        job = JOBS.get(job_id)
        if not job or job["status"] != "done":
            return render_template("processing.html", job_id=job_id,
                                   app_name=config.app_name if config else "Basketball Analyzer")
        result = job["result"]
        share_url = generate_share_url(request.host.split(":")[0], job_id)
        return render_template(
            "results.html",
            job_id=job_id,
            result=result,
            share_url=share_url,
            app_name=config.app_name if config else "Basketball Analyzer",
        )

    @app.route("/share/<job_id>")
    def share(job_id: str):
        job = JOBS.get(job_id)
        if not job or job["status"] != "done":
            return render_template("processing.html", job_id=job_id,
                                   app_name=config.app_name if config else "Basketball Analyzer")
        result = job["result"]
        share_url = generate_share_url(request.host.split(":")[0], job_id)
        return render_template(
            "results.html",
            job_id=job_id,
            result=result,
            share_url=share_url,
            readonly=True,
            app_name=config.app_name if config else "Basketball Analyzer",
        )

    @app.route("/download/<job_id>/report.json")
    def download_report(job_id: str):
        job = JOBS.get(job_id)
        if not job or not job.get("result"):
            return jsonify({"error": "Not found"}), 404
        report_path = job["result"].get("report_path")
        if report_path and Path(report_path).exists():
            return send_file(report_path, as_attachment=True)
        return jsonify({"error": "Report not found"}), 404

    @app.route("/download/<job_id>/highlights.zip")
    def download_zip(job_id: str):
        job = JOBS.get(job_id)
        if not job or not job.get("result"):
            return jsonify({"error": "Not found"}), 404
        zip_path = job["result"].get("zip_path")
        if zip_path and Path(zip_path).exists():
            return send_file(zip_path, as_attachment=True, download_name="highlights.zip")
        return jsonify({"error": "Zip not found"}), 404

    @app.route("/highlights/<job_id>/<filename>")
    def serve_highlight(job_id: str, filename: str):
        job = JOBS.get(job_id)
        if not job or not job.get("result"):
            return jsonify({"error": "Not found"}), 404
        clips_dir = UPLOAD_FOLDER / job_id / "output" / "highlights"
        return send_from_directory(clips_dir, filename)

    @app.route("/api/send-email", methods=["POST"])
    def send_email():
        data = request.get_json() or {}
        job_id = data.get("job_id")
        addresses = data.get("addresses", [])
        if not job_id or not addresses:
            return jsonify({"error": "job_id and addresses required"}), 400
        job = JOBS.get(job_id)
        if not job or job["status"] != "done":
            return jsonify({"error": "Job not ready"}), 400

        result = job["result"]
        highlights = result.get("highlights", [])
        clips = [
            Path(h.narrated_clip_path or h.clip_path)
            for h in highlights
            if (h.narrated_clip_path or h.clip_path)
        ]
        share_url = generate_share_url(request.host.split(":")[0], job_id)
        commentary = result.get("commentary", {})
        excerpt = (commentary.get("summary", "")[:300] + "...") if commentary.get("summary") else ""

        try:
            sender = EmailSender()
            sender.send_highlights(
                to_addresses=addresses,
                game_title=data.get("game_title", "Basketball Game"),
                share_url=share_url,
                highlight_clips=clips,
                commentary_excerpt=excerpt,
            )
            return jsonify({"ok": True})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/upload-youtube/<job_id>", methods=["POST"])
    def upload_youtube(job_id: str):
        job = JOBS.get(job_id)
        if not job or job["status"] != "done":
            return jsonify({"error": "Job not ready"}), 400
        data = request.get_json() or {}
        privacy = data.get("privacy", "unlisted")
        game_title = data.get("game_title", "Basketball Game")

        try:
            from basketball_analyzer.youtube.uploader import YouTubeUploader
            uploader = YouTubeUploader()
            highlights = job["result"].get("highlights", [])
            urls = uploader.upload_highlights(highlights, game_title, privacy)
            return jsonify({"urls": urls})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app


def _run_job(
    job_id: str,
    video_path: Path,
    roster_path: Path | None,
    cfg: AnalyzerConfig,
) -> None:
    def on_progress(stage: str, done: int, total: int) -> None:
        JOBS[job_id]["status"] = stage
        JOBS[job_id]["progress"] = done
        JOBS[job_id]["total"] = total

    try:
        JOBS[job_id]["status"] = "running"
        runner = PipelineRunner(cfg)
        result = runner.run(
            video_path=video_path,
            roster_path=roster_path,
            extract_highlights=True,
            generate_commentary=bool(os.environ.get("ANTHROPIC_API_KEY")),
            html_report=True,
            progress_callback=on_progress,
        )
        result["report_path"] = str(result.get("report_path") or "")
        result["html_path"] = str(result.get("html_path") or "")
        result["zip_path"] = str(result.get("zip_path") or "")
        JOBS[job_id]["result"] = result
        JOBS[job_id]["status"] = "done"
    except Exception as exc:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(exc)
'''

index_content = '''\
{% extends "base.html" %}
{% block title %}{{ app_name }} - Upload{% endblock %}
{% block content %}
<div class="row justify-content-center">
  <div class="col-md-8">
    <h1 class="mb-4" style="color:#e94560;">Basketball Game Analyzer</h1>
    <p class="text-secondary mb-4">
      Upload a video recorded by your XBOTGO Falcon camera to get AI-powered stats,
      highlight clips with voice-over, player tracking, and sharable reports.
    </p>

    <div class="card p-4 mb-4">
      <ul class="nav nav-tabs mb-3" id="inputTabs">
        <li class="nav-item">
          <button class="nav-link active" id="localTab" type="button" onclick="switchTab(\'local\')">
            Use Local File Path <span class="badge bg-success ms-1">Recommended</span>
          </button>
        </li>
        <li class="nav-item">
          <button class="nav-link" id="uploadTab" type="button" onclick="switchTab(\'upload\')">
            Upload File
          </button>
        </li>
      </ul>

      <form id="uploadForm" enctype="multipart/form-data">

        <div id="localPanel">
          <div class="mb-3">
            <label class="form-label fw-bold">Video File Path <span class="text-danger">*</span></label>
            <input type="text" class="form-control" name="local_video_path" id="localVideoPath"
              placeholder="C:\\BB_Videos\\game.mp4">
            <div class="form-text text-secondary">Paste the full path to your video - no upload needed, works with large files</div>
          </div>
          <div class="mb-3">
            <label class="form-label fw-bold">Roster File Path <span class="text-secondary">(optional)</span></label>
            <input type="text" class="form-control" name="local_roster_path"
              placeholder="C:\\BB_Videos\\roster.json">
          </div>
        </div>

        <div id="uploadPanel" style="display:none;">
          <div class="mb-3">
            <label class="form-label fw-bold">Game Video <span class="text-danger">*</span></label>
            <input type="file" class="form-control" name="video" accept=".mp4,.mov,.avi,.mkv">
            <div class="form-text text-secondary">MP4, MOV, AVI, MKV - may fail for large files on Windows</div>
          </div>
          <div class="mb-3">
            <label class="form-label fw-bold">Roster File <span class="text-secondary">(optional)</span></label>
            <input type="file" class="form-control" name="roster" accept=".json,.csv">
          </div>
        </div>

        <div class="row mb-3">
          <div class="col-md-6 mb-3 mb-md-0">
            <label class="form-label">Home Team Name</label>
            <input type="text" class="form-control" name="team_home" placeholder="e.g. Lakers">
          </div>
          <div class="col-md-6">
            <label class="form-label">Away Team Name</label>
            <input type="text" class="form-control" name="team_away" placeholder="e.g. Celtics">
          </div>
        </div>

        <div class="row mb-3">
          <div class="col-md-6 mb-3 mb-md-0">
            <label class="form-label">Video Watermark Text</label>
            <input type="text" class="form-control" name="watermark" placeholder="e.g. 2025 Coach Smith">
          </div>
          <div class="col-md-6">
            <label class="form-label">Frame Sample Rate</label>
            <select class="form-select" name="frame_rate">
              <option value="3">Every 3rd frame (high accuracy)</option>
              <option value="5" selected>Every 5th frame (recommended)</option>
              <option value="10">Every 10th frame (fast)</option>
            </select>
          </div>
        </div>

        <button type="submit" class="btn btn-primary w-100 py-2" id="submitBtn">
          Analyze Game
        </button>
      </form>
    </div>
  </div>
</div>
{% endblock %}
{% block scripts %}
<script>
let activeTab = \'local\';

function switchTab(tab) {
  activeTab = tab;
  document.getElementById(\'localPanel\').style.display  = tab === \'local\'  ? \'\' : \'none\';
  document.getElementById(\'uploadPanel\').style.display = tab === \'upload\' ? \'\' : \'none\';
  document.getElementById(\'localTab\').classList.toggle(\'active\',  tab === \'local\');
  document.getElementById(\'uploadTab\').classList.toggle(\'active\', tab === \'upload\');
  document.getElementById(\'localVideoPath\').required  = (tab === \'local\');
}

document.getElementById(\'uploadForm\').addEventListener(\'submit\', async (e) => {
  e.preventDefault();
  const btn = document.getElementById(\'submitBtn\');
  btn.textContent = activeTab === \'local\' ? \'Starting...\' : \'Uploading...\';
  btn.disabled = true;

  const formData = new FormData(e.target);
  if (activeTab === \'upload\') {
    formData.set(\'local_video_path\', \'\');
    formData.set(\'local_roster_path\', \'\');
  }
  try {
    const resp = await fetch(\'/upload\', { method: \'POST\', body: formData });
    const text = await resp.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      const firstLine = text.replace(/<[^>]+>/g, \' \').replace(/\\s+/g, \' \').trim().slice(0, 300);
      alert(\'Server error:\\n\' + firstLine);
      btn.textContent = \'Analyze Game\';
      btn.disabled = false;
      return;
    }
    if (data.job_id) {
      window.location.href = \`/results/\${data.job_id}\`;
    } else {
      alert(\'Error: \' + (data.error || \'Unknown error\') + (data.detail ? \'\\n\\n\' + data.detail.slice(0, 500) : \'\'));
      btn.textContent = \'Analyze Game\';
      btn.disabled = false;
    }
  } catch (err) {
    alert(\'Failed: \' + err.message);
    btn.textContent = \'Analyze Game\';
    btn.disabled = false;
  }
});
</script>
{% endblock %}
'''

app_path = Path("basketball_analyzer/web/app.py")
html_path = Path("basketball_analyzer/web/templates/index.html")

if not app_path.parent.exists():
    print(f"ERROR: Directory not found: {app_path.parent}")
    print("Make sure you are running this from C:\\Basketball_Analyzer")
else:
    app_path.write_text(app_content, encoding="utf-8")
    print(f"+ Wrote {app_path}")

if not html_path.parent.exists():
    print(f"ERROR: Directory not found: {html_path.parent}")
else:
    html_path.write_text(index_content, encoding="utf-8")
    print(f"+ Wrote {html_path}")

print("\nDone! Restart the app:  python -m basketball_analyzer --web")
