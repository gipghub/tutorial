from pydantic import BaseModel, field_validator


class Player(BaseModel):
    number: int
    name: str
    position: str  # PG, SG, SF, PF, C
    age: int | None = None
    team: str

    @field_validator("position")
    @classmethod
    def validate_position(cls, v: str) -> str:
        valid = {"PG", "SG", "SF", "PF", "C"}
        if v.upper() not in valid:
            raise ValueError(f"position must be one of {valid}")
        return v.upper()

    @property
    def display_name(self) -> str:
        age_str = f", age {self.age}" if self.age else ""
        return f"{self.name} ({self.position} #{self.number}{age_str})"


class Team(BaseModel):
    name: str
    short_name: str
    color_primary: str = "#1a3a6b"
    color_secondary: str = "#ffffff"


class Roster(BaseModel):
    teams: list[Team] = []
    players: list[Player] = []

    def get_player_by_number(self, number: int, team: str | None = None) -> Player | None:
        for p in self.players:
            if p.number == number:
                if team is None or p.team == team:
                    return p
        return None

    def get_team(self, name: str) -> Team | None:
        for t in self.teams:
            if t.name == name:
                return t
        return None

    def get_players_by_team(self, team_name: str) -> list[Player]:
        return [p for p in self.players if p.team == team_name]
