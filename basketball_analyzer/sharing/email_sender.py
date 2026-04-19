from __future__ import annotations

import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path


MAX_ATTACHMENT_BYTES = 24 * 1024 * 1024  # 24 MB total


class EmailSender:
    def __init__(self) -> None:
        self.host = os.environ.get("EMAIL_SMTP_HOST", "smtp.gmail.com")
        self.port = int(os.environ.get("EMAIL_SMTP_PORT", "587"))
        self.user = os.environ.get("EMAIL_USER", "")
        self.password = os.environ.get("EMAIL_PASSWORD", "")

    def send_highlights(
        self,
        to_addresses: list[str],
        game_title: str,
        share_url: str,
        highlight_clips: list[Path],
        commentary_excerpt: str = "",
        youtube_urls: list[str] | None = None,
    ) -> None:
        if not self.user or not self.password:
            raise EnvironmentError(
                "EMAIL_USER and EMAIL_PASSWORD env vars must be set to send email."
            )

        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"🏀 Game Highlights — {game_title}"
        msg["From"] = self.user
        msg["To"] = ", ".join(to_addresses)

        yt_section = ""
        if youtube_urls:
            yt_links = "\n".join(f"  • {u}" for u in youtube_urls)
            yt_section = f"\n\n🎬 Watch on YouTube:\n{yt_links}"

        body_text = (
            f"Hi!\n\n"
            f"Game highlights from {game_title} are ready.\n"
            f"View the full analysis: {share_url}{yt_section}\n\n"
            f"{commentary_excerpt}\n\n"
            f"Enjoy the game!\n"
        )
        msg.attach(MIMEText(body_text, "plain"))

        # Attach clips up to size limit
        total_bytes = 0
        for clip in highlight_clips:
            if not clip.exists():
                continue
            size = clip.stat().st_size
            if total_bytes + size > MAX_ATTACHMENT_BYTES:
                break
            with open(clip, "rb") as f:
                part = MIMEApplication(f.read(), Name=clip.name)
            part["Content-Disposition"] = f'attachment; filename="{clip.name}"'
            msg.attach(part)
            total_bytes += size

        with smtplib.SMTP(self.host, self.port) as server:
            server.ehlo()
            server.starttls()
            server.login(self.user, self.password)
            server.sendmail(self.user, to_addresses, msg.as_string())
