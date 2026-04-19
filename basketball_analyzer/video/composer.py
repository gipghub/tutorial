from __future__ import annotations

from pathlib import Path

import cv2
import ffmpeg  # type: ignore
import numpy as np


class VideoComposer:
    """Overlays audio onto video clips and applies watermark branding."""

    @staticmethod
    def overlay_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
        """Merge a video file with an audio file. Audio is truncated or padded to match video."""
        video_stream = ffmpeg.input(str(video_path))
        audio_stream = ffmpeg.input(str(audio_path))

        # Get video duration to truncate audio if needed
        probe = ffmpeg.probe(str(video_path))
        duration = float(probe["format"]["duration"])

        (
            ffmpeg.output(
                video_stream.video,
                audio_stream.audio,
                str(output_path),
                vcodec="copy",
                acodec="aac",
                audio_bitrate="128k",
                t=duration,
                shortest=None,
            )
            .overwrite_output()
            .run(quiet=True)
        )

    @staticmethod
    def add_watermark(
        video_path: Path,
        output_path: Path,
        text: str,
        position: str = "bottom-right",
        logo_path: Path | None = None,
    ) -> None:
        """Burn a text watermark (and optional logo) onto every frame of a clip."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        logo: np.ndarray | None = None
        if logo_path and logo_path.exists():
            logo = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
            if logo is not None:
                max_logo_h = h // 10
                scale = max_logo_h / logo.shape[0]
                logo = cv2.resize(logo, (int(logo.shape[1] * scale), max_logo_h))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, w / 1920)
        thickness = max(1, int(font_scale * 2))
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

        margin = 10
        positions = {
            "top-left": (margin, th + margin),
            "top-right": (w - tw - margin, th + margin),
            "bottom-left": (margin, h - margin),
            "bottom-right": (w - tw - margin, h - margin),
        }
        tx, ty = positions.get(position, positions["bottom-right"])

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                overlay = frame.copy()
                # Semi-transparent background behind text
                cv2.rectangle(
                    overlay, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (0, 0, 0), -1
                )
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                cv2.putText(frame, text, (tx, ty), font, font_scale, (255, 255, 255), thickness)

                if logo is not None:
                    lh, lw = logo.shape[:2]
                    lx = w - lw - margin
                    ly = margin
                    roi = frame[ly : ly + lh, lx : lx + lw]
                    if logo.shape[2] == 4:
                        alpha = logo[:, :, 3] / 255.0
                        for c in range(3):
                            roi[:, :, c] = (
                                alpha * logo[:, :, c] + (1 - alpha) * roi[:, :, c]
                            ).astype(np.uint8)
                    else:
                        frame[ly : ly + lh, lx : lx + lw] = logo

                out.write(frame)
        finally:
            cap.release()
            out.release()
