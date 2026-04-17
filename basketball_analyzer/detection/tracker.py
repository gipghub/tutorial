from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from basketball_analyzer.config import AnalyzerConfig
from basketball_analyzer.video.processor import Frame

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:  # allows import without ultralytics installed (e.g. unit tests)
    YOLO = None  # type: ignore


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def crop(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        x1 = max(0, int(self.x1))
        y1 = max(0, int(self.y1))
        x2 = min(w, int(self.x2))
        y2 = min(h, int(self.y2))
        return image[y1:y2, x1:x2]


@dataclass
class Detection:
    track_id: int
    class_id: int
    confidence: float
    bbox: BoundingBox
    is_ball: bool = False
    is_player: bool = False


@dataclass
class BallTrajectory:
    positions: deque = field(default_factory=lambda: deque(maxlen=10))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=10))

    @property
    def velocity(self) -> Optional[tuple[float, float]]:
        if len(self.positions) < 2:
            return None
        dx = self.positions[-1][0] - self.positions[-2][0]
        dy = self.positions[-1][1] - self.positions[-2][1]
        dt = self.timestamps[-1] - self.timestamps[-2]
        if dt == 0:
            return None
        return (dx / dt, dy / dt)

    @property
    def speed(self) -> float:
        v = self.velocity
        if v is None:
            return 0.0
        return float(np.sqrt(v[0] ** 2 + v[1] ** 2))


@dataclass
class TrackFrame:
    frame_index: int
    timestamp_sec: float
    players: list[Detection]
    ball: Optional[Detection]
    ball_trajectory: BallTrajectory


class PlayerTracker:
    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config
        if YOLO is None:
            raise ImportError("ultralytics is required: pip install ultralytics")
        self.model = YOLO(config.model_name)
        self._ball_trajectory = BallTrajectory()
        # track_id → list of (nx, ny, timestamp)
        self.player_history: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
        # frame counter per track_id for OCR throttling
        self._ocr_frame_counter: dict[int, int] = defaultdict(int)

    def process_frame(self, frame: Frame) -> TrackFrame:
        cfg = self.config
        results = self.model.track(
            source=frame.image,
            persist=True,
            classes=[cfg.person_class_id, cfg.sports_ball_class_id],
            conf=cfg.confidence_threshold,
            iou=cfg.iou_threshold,
            device=cfg.device,
            verbose=False,
        )

        detections = self._parse_results(results[0])
        players = [d for d in detections if d.is_player]
        ball_dets = [d for d in detections if d.is_ball]
        ball = ball_dets[0] if ball_dets else None

        h, w = frame.image.shape[:2]

        if ball:
            cx, cy = ball.bbox.center
            self._ball_trajectory.positions.append((cx / w, cy / h))
            self._ball_trajectory.timestamps.append(frame.timestamp_sec)

        for player in players:
            cx, cy = player.bbox.center
            self.player_history[player.track_id].append(
                (cx / w, cy / h, frame.timestamp_sec)
            )
            self._ocr_frame_counter[player.track_id] += 1

        return TrackFrame(
            frame_index=frame.index,
            timestamp_sec=frame.timestamp_sec,
            players=players,
            ball=ball,
            ball_trajectory=BallTrajectory(
                positions=deque(self._ball_trajectory.positions),
                timestamps=deque(self._ball_trajectory.timestamps),
            ),
        )

    def should_run_ocr(self, track_id: int) -> bool:
        count = self._ocr_frame_counter.get(track_id, 0)
        return count > 0 and count % self.config.jersey_ocr_every_n_frames == 0

    def _parse_results(self, result: object) -> list[Detection]:
        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        track_ids = (
            boxes.id.cpu().numpy().astype(int)
            if boxes.id is not None
            else [-1] * len(boxes)
        )
        for box, track_id in zip(boxes, track_ids):
            class_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            det = Detection(
                track_id=int(track_id),
                class_id=class_id,
                confidence=conf,
                bbox=BoundingBox(x1, y1, x2, y2),
                is_ball=(class_id == self.config.sports_ball_class_id),
                is_player=(class_id == self.config.person_class_id),
            )
            detections.append(det)
        return detections
