"""Generates a tiny synthetic basketball-like MP4 for integration tests."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def generate_test_video(output_path: Path, duration_sec: float = 5.0, fps: float = 30.0) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    w, h = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    total_frames = int(duration_sec * fps)
    ball_x, ball_y = w // 2, h // 2
    ball_vx, ball_vy = 3, -2

    player_positions = [
        [50, 180], [100, 180], [200, 180], [250, 180], [290, 100],
    ]

    for i in range(total_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (34, 139, 34)  # green court

        # Court lines
        cv2.rectangle(frame, (20, 20), (w - 20, h - 20), (255, 255, 255), 2)
        cv2.line(frame, (w // 2, 20), (w // 2, h - 20), (255, 255, 255), 1)

        # Move ball
        ball_x += ball_vx
        ball_y += ball_vy
        if ball_x <= 10 or ball_x >= w - 10:
            ball_vx = -ball_vx
        if ball_y <= 10 or ball_y >= h - 10:
            ball_vy = -ball_vy

        # Draw ball (orange)
        cv2.circle(frame, (int(ball_x), int(ball_y)), 8, (0, 128, 255), -1)

        # Draw players (blue rectangles)
        for px, py in player_positions:
            px = (px + i // 10) % (w - 30)
            cv2.rectangle(frame, (px, py - 30), (px + 20, py), (255, 50, 50), -1)
            # Jersey number
            cv2.putText(frame, "5", (px + 4, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out.write(frame)

    out.release()
    return output_path


if __name__ == "__main__":
    path = generate_test_video(Path("test_game.mp4"))
    print(f"Created: {path}")
