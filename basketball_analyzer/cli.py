from __future__ import annotations

from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

console = Console()


@click.command()
@click.argument("video_path", required=False, type=click.Path(path_type=Path))
@click.option("--output-dir", "-o", default="output", type=click.Path(), help="Output directory")
@click.option("--roster", "-r", default=None, type=click.Path(), help="Roster JSON or CSV file")
@click.option("--model", default="yolov8n.pt", help="YOLO model (yolov8n.pt or yolov8s.pt)")
@click.option("--frame-rate", "-f", default=5, type=int, help="Process every Nth frame")
@click.option("--device", default="cpu", help="Inference device: cpu or cuda")
@click.option("--team-home", default="", help="Home team name")
@click.option("--team-away", default="", help="Away team name")
@click.option("--watermark", default="", help="Watermark text on video clips")
@click.option("--confidence", default=0.35, type=float, help="YOLO confidence threshold")
@click.option("--no-highlights", is_flag=True, help="Skip highlight extraction")
@click.option("--no-commentary", is_flag=True, help="Skip Claude API commentary")
@click.option("--html-report", is_flag=True, help="Generate HTML report")
@click.option("--email", default="", help="Comma-separated email addresses to share highlights")
@click.option("--upload-youtube", is_flag=True, help="Upload highlights to YouTube after analysis")
@click.option("--youtube-auth", is_flag=True, help="Authenticate with YouTube (one-time setup)")
@click.option("--youtube-privacy", default="unlisted", type=click.Choice(["public", "unlisted", "private"]))
@click.option("--web", is_flag=True, help="Launch Flask web UI")
@click.option("--port", default=5000, type=int, help="Web UI port")
def main(
    video_path,
    output_dir,
    roster,
    model,
    frame_rate,
    device,
    team_home,
    team_away,
    watermark,
    confidence,
    no_highlights,
    no_commentary,
    html_report,
    email,
    upload_youtube,
    youtube_auth,
    youtube_privacy,
    web,
    port,
):
    """Basketball Game Analyzer — analyze XBOTGO Falcon camera footage."""

    if youtube_auth:
        from basketball_analyzer.youtube.uploader import authenticate
        authenticate()
        console.print("[green]YouTube authentication successful![/green]")
        return

    if web:
        from basketball_analyzer.config import AnalyzerConfig
        from basketball_analyzer.web.app import create_app

        cfg = AnalyzerConfig(
            model_name=model,
            frame_sample_rate=frame_rate,
            device=device,
            confidence_threshold=confidence,
            output_dir=Path(output_dir),
            team_home_name=team_home,
            team_away_name=team_away,
            watermark_text=watermark,
        )
        app = create_app(cfg)
        console.print(f"[green]Starting web UI at http://localhost:{port}[/green]")
        app.run(host="0.0.0.0", port=port, debug=False)
        return

    if not video_path:
        raise click.UsageError("VIDEO_PATH is required unless --web or --youtube-auth is used.")

    video_path = Path(video_path)
    if not video_path.exists():
        raise click.BadParameter(f"File not found: {video_path}", param_hint="VIDEO_PATH")

    from basketball_analyzer.config import AnalyzerConfig
    from basketball_analyzer.pipeline.runner import PipelineRunner

    cfg = AnalyzerConfig(
        model_name=model,
        frame_sample_rate=frame_rate,
        device=device,
        confidence_threshold=confidence,
        output_dir=Path(output_dir),
        team_home_name=team_home,
        team_away_name=team_away,
        watermark_text=watermark,
    )

    runner = PipelineRunner(cfg)
    result = runner.run(
        video_path=video_path,
        roster_path=Path(roster) if roster else None,
        extract_highlights=not no_highlights,
        generate_commentary=not no_commentary,
        html_report=html_report,
    )

    highlights = result.get("highlights", [])

    # YouTube upload
    if upload_youtube and highlights:
        game_title = f"{team_home} vs {team_away}" if team_home and team_away else video_path.stem
        try:
            from basketball_analyzer.youtube.uploader import YouTubeUploader
            uploader = YouTubeUploader()
            urls = uploader.upload_highlights(highlights, game_title, youtube_privacy)
            console.print("[green]YouTube URLs:[/green]")
            for url in urls:
                console.print(f"  {url}")
        except Exception as exc:
            console.print(f"[yellow]YouTube upload failed: {exc}[/yellow]")

    # Email sharing
    if email:
        addresses = [a.strip() for a in email.split(",") if a.strip()]
        clips = [
            Path(h.narrated_clip_path or h.clip_path)
            for h in highlights
            if (h.narrated_clip_path or h.clip_path)
        ]
        commentary = result.get("commentary", {})
        excerpt = commentary.get("summary", "")[:300]
        try:
            from basketball_analyzer.sharing.email_sender import EmailSender
            sender = EmailSender()
            sender.send_highlights(
                to_addresses=addresses,
                game_title=video_path.stem,
                share_url="",
                highlight_clips=clips,
                commentary_excerpt=excerpt,
            )
            console.print(f"[green]Email sent to {', '.join(addresses)}[/green]")
        except Exception as exc:
            console.print(f"[yellow]Email failed: {exc}[/yellow]")
