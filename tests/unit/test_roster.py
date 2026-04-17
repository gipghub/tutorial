from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from basketball_analyzer.roster.loader import load_roster
from basketball_analyzer.roster.models import Player, Roster, Team


def test_player_position_validated():
    with pytest.raises(ValueError):
        Player(number=5, name="Test", position="QB", team="Home")


def test_player_position_uppercased():
    p = Player(number=5, name="Test", position="pg", team="Home")
    assert p.position == "PG"


def test_roster_get_player_by_number():
    roster = Roster(
        teams=[Team(name="A", short_name="AA")],
        players=[
            Player(number=23, name="Alice", position="SF", team="A"),
            Player(number=5,  name="Bob",   position="PG", team="A"),
        ],
    )
    p = roster.get_player_by_number(23)
    assert p is not None
    assert p.name == "Alice"


def test_roster_get_player_missing():
    roster = Roster()
    assert roster.get_player_by_number(99) is None


def test_load_roster_returns_none_for_none():
    assert load_roster(None) is None


def test_load_roster_json(tmp_path):
    data = {
        "teams": [{"name": "Home", "short_name": "HME"}],
        "players": [
            {"number": 1, "name": "Alice", "position": "PG", "team": "Home"}
        ],
    }
    p = tmp_path / "roster.json"
    p.write_text(json.dumps(data))
    roster = load_roster(p)
    assert roster is not None
    assert len(roster.players) == 1
    assert roster.players[0].name == "Alice"


def test_load_roster_csv(tmp_path):
    csv_content = "number,name,position,age,team\n5,Bob,SG,17,Away\n"
    p = tmp_path / "roster.csv"
    p.write_text(csv_content)
    roster = load_roster(p)
    assert roster is not None
    assert len(roster.players) == 1
    assert roster.players[0].name == "Bob"


def test_load_roster_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_roster(Path("/nonexistent/path.json"))


def test_player_display_name():
    p = Player(number=23, name="Alice", position="SF", age=17, team="Home")
    assert "Alice" in p.display_name
    assert "#23" in p.display_name
    assert "SF" in p.display_name
    assert "17" in p.display_name
