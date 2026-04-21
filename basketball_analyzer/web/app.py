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
            raw_players = form.get("highlight_players", "").strip()
            cfg.highlight_players = [p.strip() for p in raw_players.split(",") if p.strip()]

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
        try:
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
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            return f"<pre style='color:red;padding:20px'>Results page error:\n{tb}</pre>", 500

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
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = result
    except Exception as exc:
        import traceback
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(exc)
        JOBS[job_id]["detail"] = traceback.format_exc()
