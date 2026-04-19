from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoMeta:
    path: Path
    fps: float
    total_frames: int
    width: int
    height: int
    duration_sec: float


@dataclass
class Frame:
    index: int
    timestamp_sec: float
    image: np.ndarray  # BGR HxWxC


class VideoProcessor:
    def __init__(self, video_path: Path) -> None:
        self.path = Path(video_path)
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.meta = VideoMeta(
            path=self.path,
            fps=fps,
            total_frames=total,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            duration_sec=total / fps,
        )
        cap.release()

    def iter_frames(self, sample_rate: int = 1) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self.path))
        frame_idx = 0
        try:
            while True:
                ret, image = cap.read()
                if not ret:
                    break
                if frame_idx % sample_rate == 0:
                    yield Frame(
                        index=frame_idx,
                        timestamp_sec=frame_idx / self.meta.fps,
                        image=image,
                    )
                frame_idx += 1
        finally:
            cap.release()

    def extract_clip(
        self,
        start_sec: float,
        end_sec: float,
        output_path: Path,
        codec: str = "mp4v",
    ) -> None:
        cap = cv2.VideoCapture(str(self.path))
        fps = self.meta.fps
        fourcc = cv2.VideoWriter_fourcc(*codec)
        start_frame = max(0, int(start_sec * fps))
        end_frame = min(self.meta.total_frames, int(end_sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        out = cv2.VideoWriter(
            str(output_path), fourcc, fps, (self.meta.width, self.meta.height)
        )
        try:
            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
        finally:
            cap.release()
            out.release()
