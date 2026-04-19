import csv
import json
from pathlib import Path

from basketball_analyzer.roster.models import Player, Roster, Team


def load_roster(path: Path | None) -> Roster | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Roster file not found: {path}")
    if path.suffix.lower() == ".json":
        return _load_json(path)
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    raise ValueError(f"Unsupported roster format: {path.suffix} (use .json or .csv)")


def _load_json(path: Path) -> Roster:
    data = json.loads(path.read_text())
    return Roster.model_validate(data)


def _load_csv(path: Path) -> Roster:
    players: list[Player] = []
    teams_seen: dict[str, Team] = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            team_name = row.get("team", "Unknown")
            if team_name not in teams_seen:
                teams_seen[team_name] = Team(
                    name=team_name,
                    short_name=team_name[:3].upper(),
                )
            age_raw = row.get("age", "")
            players.append(
                Player(
                    number=int(row["number"]),
                    name=row["name"],
                    position=row.get("position", "PG"),
                    age=int(age_raw) if age_raw.strip().isdigit() else None,
                    team=team_name,
                )
            )

    return Roster(teams=list(teams_seen.values()), players=players)
