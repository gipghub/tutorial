from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AnalyzerConfig:
    # Video sampling
    frame_sample_rate: int = 5

    # YOLO model
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.20
    iou_threshold: float = 0.45
    device: str = "cpu"

    # COCO class IDs
    person_class_id: int = 0
    sports_ball_class_id: int = 32

    # Court zone geometry (normalized 0-1 relative to frame)
    hoop_left: tuple[float, float] = (0.18, 0.35)
    hoop_right: tuple[float, float] = (0.82, 0.35)
    paint_left: tuple[float, float, float, float] = (0.08, 0.25, 0.33, 0.75)
    paint_right: tuple[float, float, float, float] = (0.67, 0.25, 0.92, 0.75)
    three_point_radius: float = 0.28

    # Highlight detection thresholds
    highlight_ball_speed_threshold: float = 0.06   # normalized units/sec (lowered for tracking cameras)
    highlight_player_cluster_threshold: int = 3    # lowered: tracking cam often shows fewer players
    highlight_min_duration_sec: float = 2.0
    highlight_padding_sec: float = 2.0

    # Shot detection -- tracking camera mode: hoop proximity is NOT used because the
    # camera pans, so hoop screen coords are unpredictable.  Only upward ball velocity
    # is required (ball moving up = shot attempt).
    shot_upward_velocity_threshold: float = -0.01  # normalized y/sec (less strict)
    shot_proximity_to_hoop_threshold: float = 1.0  # effectively disabled (always true)

    # Jersey OCR
    jersey_ocr_every_n_frames: int = 15  # run OCR every N processed frames per player
    jersey_vote_threshold: int = 3       # votes needed to lock in a player identity

    # Output
    output_dir: Path = field(default_factory=lambda: Path("output"))
    highlight_clips_dir: str = "highlights"

    # Branding
    app_name: str = "Basketball Analyzer"
    team_home_name: str = ""
    team_away_name: str = ""
    org_logo_path: Path | None = None
    watermark_text: str = ""
    watermark_position: str = "bottom-right"  # top-left, top-right, bottom-left, bottom-right
    primary_color: str = "#e94560"

    # Claude API
    claude_model: str = "claude-sonnet-4-6"
    claude_max_tokens: int = 4096

    # TTS
    tts_voice: str = "en-US-GuyNeural"

    # Highlight player filter -- if non-empty, only clips containing a shot by one
    # of these players (substring match, case-insensitive) will be extracted.
    # Empty list = include all excitement events (default).
    highlight_players: list = field(default_factory=list)
