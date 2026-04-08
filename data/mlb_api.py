"""MLB Stats API — 당일 경기 스케줄, 라인업, 선발투수 조회."""

import requests
from datetime import date


MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# MLB team ID → Statcast 팀 코드 매핑
TEAM_ID_TO_CODE = {
    108: "LAA", 109: "AZ", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC", 119: "LAD", 120: "WSH", 121: "NYM", 133: "ATH",
    134: "PIT", 135: "SD", 136: "SEA", 137: "SF", 138: "STL",
    139: "TB", 140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}


def get_schedule(game_date: str | None = None) -> list[dict]:
    """당일 MLB 경기 목록 + 선발투수 + 라인업.

    Args:
        game_date: "YYYY-MM-DD" 형식. None이면 오늘.

    Returns:
        [
            {
                "game_id": 12345,
                "away_team": "NYY",
                "home_team": "LAD",
                "away_team_name": "New York Yankees",
                "home_team_name": "Los Angeles Dodgers",
                "away_pitcher": {"id": 543037, "name": "Gerrit Cole"},
                "home_pitcher": {"id": 607192, "name": "Tyler Glasnow"},
                "away_lineup": [{"id": 660271, "name": "Shohei Ohtani", "order": 1}, ...],
                "home_lineup": [...],
                "status": "Scheduled",
                "game_time": "19:05",
            },
            ...
        ]
    """
    if game_date is None:
        game_date = date.today().isoformat()

    url = f"{MLB_API_BASE}/schedule"
    params = {
        "date": game_date,
        "sportId": 1,
        "hydrate": "lineups,probablePitcher",
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    dates = data.get("dates", [])
    if not dates:
        return []

    games = []
    for g in dates[0].get("games", []):
        away = g["teams"]["away"]
        home = g["teams"]["home"]

        # 팀 코드 (Statcast 기준)
        away_abbr = TEAM_ID_TO_CODE.get(away["team"]["id"], away["team"]["name"][:3].upper())
        home_abbr = TEAM_ID_TO_CODE.get(home["team"]["id"], home["team"]["name"][:3].upper())

        # 선발투수
        away_sp = away.get("probablePitcher", {})
        home_sp = home.get("probablePitcher", {})

        # 라인업
        lineups = g.get("lineups", {})
        away_lineup = []
        for i, p in enumerate(lineups.get("awayPlayers", [])):
            away_lineup.append({
                "id": p["id"],
                "name": p["fullName"],
                "order": i + 1,
            })

        home_lineup = []
        for i, p in enumerate(lineups.get("homePlayers", [])):
            home_lineup.append({
                "id": p["id"],
                "name": p["fullName"],
                "order": i + 1,
            })

        # 게임 시간 (UTC → KST 변환)
        game_date_utc = g.get("gameDate", "")
        game_time = ""
        game_date_kst = ""
        if "T" in game_date_utc:
            from datetime import datetime, timezone, timedelta
            utc_dt = datetime.fromisoformat(game_date_utc.replace("Z", "+00:00"))
            kst_dt = utc_dt.astimezone(timezone(timedelta(hours=9)))
            game_time = kst_dt.strftime("%H:%M")
            game_date_kst = kst_dt.strftime("%Y-%m-%d")

        games.append({
            "game_id": g["gamePk"],
            "away_team": away_abbr,
            "home_team": home_abbr,
            "away_team_name": away["team"]["name"],
            "home_team_name": home["team"]["name"],
            "away_pitcher": {
                "id": away_sp.get("id"),
                "name": away_sp.get("fullName", "TBD"),
            } if away_sp else {"id": None, "name": "TBD"},
            "home_pitcher": {
                "id": home_sp.get("id"),
                "name": home_sp.get("fullName", "TBD"),
            } if home_sp else {"id": None, "name": "TBD"},
            "away_lineup": away_lineup,
            "home_lineup": home_lineup,
            "status": g.get("status", {}).get("detailedState", ""),
            "game_time": game_time,
            "game_date_kst": game_date_kst,
        })

    return games


def get_today_games() -> list[dict]:
    """오늘 경기 목록."""
    return get_schedule()


def get_current_rosters(season: int = 2026) -> dict:
    """전 팀 현재 로스터 조회 → {player_id: team_code} 매핑.

    Returns:
        {660271: "LAD", 571945: "WSH", ...}
    """
    # 전체 팀 목록
    resp = requests.get(f"{MLB_API_BASE}/teams", params={"sportId": 1, "season": season}, timeout=10)
    resp.raise_for_status()
    teams = resp.json().get("teams", [])

    roster_map = {}
    for team in teams:
        team_id = team["id"]
        team_code = TEAM_ID_TO_CODE.get(team_id, team["name"][:3].upper())

        try:
            r = requests.get(
                f"{MLB_API_BASE}/teams/{team_id}/roster",
                params={"rosterType": "active", "season": season},
                timeout=10,
            )
            r.raise_for_status()
            roster = r.json().get("roster", [])
            for p in roster:
                roster_map[p["person"]["id"]] = team_code
        except Exception:
            continue

    return roster_map
