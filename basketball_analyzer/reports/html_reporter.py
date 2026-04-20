from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from jinja2 import Template

matplotlib.use("Agg")

from basketball_analyzer.config import AnalyzerConfig
from basketball_analyzer.highlights.extractor import HighlightSegment
from basketball_analyzer.roster.models import Roster
from basketball_analyzer.stats.calculator import GameStats

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ app_name }} - Game Report</title>
<link rel="manifest" href="/static/manifest.json">
<meta name="theme-color" content="{{ primary_color }}">
<meta name="apple-mobile-web-app-capable" content="yes">
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
  :root { --primary: {{ primary_color }}; }
  body { font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; }
  .navbar { background: var(--primary) !important; }
  .stat-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; text-align: center; }
  .stat-value { font-size: 2.5rem; font-weight: 700; color: var(--primary); }
  .stat-label { color: #8b949e; font-size: 0.85rem; }
  .heatmap-img { width: 100%; border-radius: 8px; }
  .highlight-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin-bottom: 12px; }
  .badge-excitement { background: var(--primary); }
  .commentary-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; white-space: pre-wrap; font-size: 0.9rem; line-height: 1.7; }
  .player-row td { color: #e6edf3; }
  table { color: #e6edf3; }
  .share-btn { margin: 4px; }
  .qr-img { max-width: 180px; }
  video { width: 100%; border-radius: 8px; margin-top: 8px; }
  h2, h3 { color: var(--primary); }
</style>
</head>
<body>
<nav class="navbar navbar-dark mb-4">
  <div class="container">
    <span class="navbar-brand fw-bold">{{ app_name }}</span>
    {% if team_home or team_away %}
    <span class="text-white">{{ team_home }} vs {{ team_away }}</span>
    {% endif %}
  </div>
</nav>

<div class="container pb-5">

  <div class="row g-3 mb-4">
    <div class="col-6 col-md-3">
      <div class="stat-card">
        <div class="stat-value">{{ duration_fmt }}</div>
        <div class="stat-label">Game Duration</div>
      </div>
    </div>
    <div class="col-6 col-md-3">
      <div class="stat-card">
        <div class="stat-value">{{ stats.shot_attempts | length }}</div>
        <div class="stat-label">Shot Attempts</div>
      </div>
    </div>
    <div class="col-6 col-md-3">
      <div class="stat-card">
        <div class="stat-value">{{ "%.1f" | format(stats.events_per_minute) }}</div>
        <div class="stat-label">Events / Min</div>
      </div>
    </div>
    <div class="col-6 col-md-3">
      <div class="stat-card">
        <div class="stat-value">{{ highlights | length }}</div>
        <div class="stat-label">Highlight Clips</div>
      </div>
    </div>
  </div>

  {% if roster_players %}
  <h2 class="mb-3">Player Stats</h2>
  <div class="table-responsive mb-4">
    <table class="table table-dark table-hover">
      <thead><tr>
        <th>#</th><th>Name</th><th>Team</th><th>Pos</th><th>Age</th>
        <th>Possession (s)</th><th>Shot Attempts</th>
      </tr></thead>
      <tbody>
      {% for p in roster_players %}
      <tr class="player-row">
        <td>{{ p.number }}</td>
        <td>{{ p.name }}</td>
        <td>{{ p.team }}</td>
        <td>{{ p.position }}</td>
        <td>{{ p.age or '-' }}</td>
        <td>{{ possession.get(p.display_name, 0) | round(1) }}</td>
        <td>{{ shot_counts.get(p.name, 0) }}</td>
      </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
  {% endif %}

  <h2 class="mb-3">Court Heatmap</h2>
  <div class="row mb-4">
    <div class="col-md-8">
      <img class="heatmap-img" src="data:image/png;base64,{{ heatmap_b64 }}" alt="Heatmap">
    </div>
  </div>

  {% if highlights %}
  <h2 class="mb-3">Highlight Clips</h2>
  {% for h in highlights %}
  <div class="highlight-card">
    <div class="d-flex justify-content-between align-items-center mb-2">
      <strong>Highlight {{ loop.index }}</strong>
      <span>{{ "%.0f" | format(h.start_sec) }}s - {{ "%.0f" | format(h.end_sec) }}s
        &nbsp;<span class="badge badge-excitement">{{ "%.1f" | format(h.peak_excitement) }}</span>
      </span>
    </div>
    {% set clip = h.narrated_clip_path or h.clip_path %}
    {% if clip %}
    <video controls preload="metadata">
      <source src="file://{{ clip }}" type="video/mp4">
      <a href="file://{{ clip }}">Download clip</a>
    </video>
    {% endif %}
  </div>
  {% endfor %}
  {% endif %}

  {% if commentary %}
  <h2 class="mb-3 mt-4">Play-by-Play</h2>
  <div class="commentary-box mb-4">{{ commentary.get('play_by_play', '') }}</div>

  <h2 class="mb-3">Game Summary</h2>
  <div class="commentary-box mb-4">{{ commentary.get('summary', '') }}</div>
  {% endif %}

  <h2 class="mb-3">Share</h2>
  <div class="d-flex flex-wrap align-items-center mb-2">
    <button class="btn btn-outline-light share-btn" onclick="navigator.clipboard.writeText(window.location.href)">Copy Link</button>
    <a class="btn btn-success share-btn" href="https://wa.me/?text={{ share_text }}">WhatsApp</a>
    <a class="btn btn-primary share-btn" href="https://www.facebook.com/sharer/sharer.php?u={{ share_url }}">Facebook</a>
    <a class="btn btn-info share-btn" href="sms:?body={{ share_text }}">SMS</a>
  </div>

  <p class="text-muted mt-5" style="font-size:0.75rem;">
    Generated by {{ app_name }}. Footage recorded on your own device is your original content.
    Ensure you have rights before uploading to public platforms.
  </p>
</div>
</body>
</html>"""


class HtmlReporter:
    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config

    def write(
        self,
        stats: GameStats,
        highlights: list[HighlightSegment],
        commentary: dict[str, str],
        roster: Roster | None = None,
        share_url: str = "",
    ) -> Path:
        heatmap_b64 = self._render_heatmap(stats)
        duration_m, duration_s = divmod(int(stats.duration_sec), 60)
        duration_fmt = f"{duration_m}:{duration_s:02d}"

        possession = stats.possession_by_player
        shot_counts: dict[str, int] = {}
        for shot in stats.shot_attempts:
            name = shot.player_label.split(" (")[0]
            shot_counts[name] = shot_counts.get(name, 0) + 1

        roster_players = roster.players if roster else []

        import urllib.parse
        share_text = urllib.parse.quote(
            f"Watch basketball highlights: {share_url}" if share_url else "Basketball highlights!"
        )

        html = Template(HTML_TEMPLATE).render(
            app_name=self.config.app_name,
            primary_color=self.config.primary_color,
            team_home=self.config.team_home_name,
            team_away=self.config.team_away_name,
            stats=stats,
            highlights=highlights,
            commentary=commentary,
            heatmap_b64=heatmap_b64,
            duration_fmt=duration_fmt,
            roster_players=roster_players,
            possession=possession,
            shot_counts=shot_counts,
            share_url=share_url,
            share_text=share_text,
        )

        out_path = self.config.output_dir / "report.html"
        out_path.write_text(html, encoding="utf-8")
        return out_path

    def _render_heatmap(self, stats: GameStats) -> str:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#161b22")
        ax.set_facecolor("#161b22")
        im = ax.imshow(
            stats.combined_heatmap,
            cmap="hot",
            interpolation="bilinear",
            aspect="auto",
            origin="upper",
        )
        ax.set_title("Player Movement Heatmap", color="white")
        ax.set_xlabel("Court Width ->", color="#8b949e")
        ax.set_ylabel("Court Length ->", color="#8b949e")
        ax.tick_params(colors="#8b949e")
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(colors="#8b949e")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100, facecolor="#161b22")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
