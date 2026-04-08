"""The Odds API — MLB 배당 데이터 조회.

키1: bulk 조회 (풀게임 h2h/spreads/totals)
키2: 이벤트별 조회 (team_totals, F5 마켓)
"""

import os
import requests

# .env 로컬 개발 지원
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

# API 키는 환경변수에서만 (GitHub Secrets or .env)
ODDS_API_KEYS = [
    k for k in [
        os.environ.get("ODDS_API_KEY_PRIMARY"),
        os.environ.get("ODDS_API_KEY_BACKUP1"),
        os.environ.get("ODDS_API_KEY_BACKUP2"),
    ] if k
]

if not ODDS_API_KEYS:
    print("[WARN] No ODDS_API_KEY_* environment variables set. Odds fetching will fail.")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"


def _get_working_key(preferred_idx: int = 0) -> str:
    """크레딧 남은 키 반환."""
    for i in range(len(ODDS_API_KEYS)):
        idx = (preferred_idx + i) % len(ODDS_API_KEYS)
        key = ODDS_API_KEYS[idx]
        try:
            r = requests.get(f"{ODDS_API_BASE}/sports", params={"apiKey": key}, timeout=5)
            remaining = int(r.headers.get("x-requests-remaining", 0))
            if remaining > 0:
                return key
        except Exception:
            continue
    return ODDS_API_KEYS[0]


def _to_dec(american):
    """아메리칸 배당 → 소수점 배당."""
    if american is None:
        return None
    return round(1 + american / 100, 2) if american > 0 else round(1 + 100 / abs(american), 2)


def get_mlb_odds() -> list[dict]:
    """현재 MLB 경기 배당 (머니라인 + 핸디캡 + O/U). Bulk 조회."""
    resp = requests.get(f"{ODDS_API_BASE}/sports/baseball_mlb/odds", params={
        "apiKey": _get_working_key(0),
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
    }, timeout=15)
    resp.raise_for_status()

    games = []
    for g in resp.json():
        parsed = {
            "away_team": g["away_team"],
            "home_team": g["home_team"],
            "commence_time": g["commence_time"],
            "moneyline": {},
            "spread": {},
            "total": {},
        }

        for bm in g.get("bookmakers", [])[:1]:
            for market in bm.get("markets", []):
                if market["key"] == "h2h":
                    for o in market["outcomes"]:
                        if o["name"] == g["away_team"]:
                            parsed["moneyline"]["away"] = o["price"]
                        else:
                            parsed["moneyline"]["home"] = o["price"]
                elif market["key"] == "spreads":
                    for o in market["outcomes"]:
                        if o["name"] == g["away_team"]:
                            parsed["spread"]["away_line"] = o.get("point", 0)
                            parsed["spread"]["away_price"] = o["price"]
                        else:
                            parsed["spread"]["home_line"] = o.get("point", 0)
                            parsed["spread"]["home_price"] = o["price"]
                elif market["key"] == "totals":
                    for o in market["outcomes"]:
                        if o["name"] == "Over":
                            parsed["total"]["line"] = o.get("point", 8.5)
                            parsed["total"]["over_price"] = o["price"]
                        else:
                            parsed["total"]["under_price"] = o["price"]

        games.append(parsed)

    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"[odds] {len(games)} games loaded (credits remaining: {remaining})")
    return games


def get_mlb_events(future_only: bool = True) -> list[dict]:
    """MLB 이벤트 목록 (무료, 크레딧 소모 없음).

    Args:
        future_only: True면 아직 시작 안 한 경기만 반환 (live 경기는 배당이 잔여 기준으로 변해 제외).
    """
    resp = requests.get(f"{ODDS_API_BASE}/sports/baseball_mlb/events", params={
        "apiKey": _get_working_key(0),
    }, timeout=15)
    resp.raise_for_status()
    events = resp.json()
    if future_only:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        events = [
            e for e in events
            if datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) > now
        ]
    return events


def get_event_odds(event_id: str, markets: str = "team_totals") -> dict:
    """개별 이벤트 배당 조회 (1 credit).

    Args:
        event_id: 이벤트 ID
        markets: 쉼표 구분 마켓 키

    Returns:
        전체 응답 dict
    """
    resp = requests.get(
        f"{ODDS_API_BASE}/sports/baseball_mlb/events/{event_id}/odds",
        params={
            "apiKey": _get_working_key(0),
            # us + us2 (bovada/betonline/pinnacle 포함) — 커버리지 중요
            "regions": "us,us2",
            "markets": markets,
            "oddsFormat": "american",
        },
        timeout=10,
    )
    if resp.status_code != 200:
        return {}
    return resp.json()


def get_mlb_team_totals() -> list[dict]:
    """전 경기 팀 토탈 라인 조회.

    1) /events로 경기 목록 (무료)
    2) 경기별 /events/{id}/odds?markets=team_totals (1 credit/경기)

    Returns:
        [{
            "event_id": str,
            "away_team": str,
            "home_team": str,
            "away_total": {"line": 4.5, "over_price": -110, "under_price": -110},
            "home_total": {"line": 4.5, "over_price": -110, "under_price": -110},
        }, ...]
    """
    events = get_mlb_events()

    results = []
    remaining = "?"

    for ev in events:
        eid = ev["id"]
        try:
            data = get_event_odds(eid, "team_totals")
            if not data:
                continue

            parsed = {
                "event_id": eid,
                "away_team": ev["away_team"],
                "home_team": ev["home_team"],
                "commence_time": ev.get("commence_time", ""),
                "away_total": {},
                "home_total": {},
            }

            for bm in data.get("bookmakers", [])[:1]:
                for market in bm.get("markets", []):
                    if market["key"] == "team_totals":
                        for o in market["outcomes"]:
                            team = o.get("description", "")
                            side = "away_total" if team == ev["away_team"] else "home_total"
                            if o["name"] == "Over":
                                parsed[side]["line"] = o.get("point")
                                parsed[side]["over_price"] = o["price"]
                            else:
                                parsed[side]["under_price"] = o["price"]

            results.append(parsed)
        except Exception:
            continue

    print(f"[odds] team_totals: {len(results)} games loaded")
    return results


def get_f5_line_for_event(event_id: str) -> float | None:
    """단일 이벤트의 F5 메인 라인(.5 단위 배당차 최소) 조회. 1 credit."""
    data = get_event_odds(event_id, "totals_1st_5_innings")
    if not data:
        return None
    cands = {}
    for bm in data.get("bookmakers", []):
        for m in bm.get("markets", []):
            if m.get("key") != "totals_1st_5_innings":
                continue
            for o in m.get("outcomes", []):
                line = o.get("point")
                if line is None:
                    continue
                if line not in cands:
                    cands[line] = {"line": line}
                if o["name"] == "Over":
                    cands[line]["over_price"] = o["price"]
                elif o["name"] == "Under":
                    cands[line]["under_price"] = o["price"]
    main = _pick_balanced_line(list(cands.values()))
    return main.get("line") if main else None


def find_event_by_teams(away_team_name: str, home_team_name: str, kst_date: str) -> str | None:
    """팀 이름 + KST 날짜로 event_id 찾기. 무료 (이벤트 리스트 조회)."""
    from datetime import datetime, timezone, timedelta
    events = get_mlb_events(future_only=True)
    for e in events:
        if e["away_team"] != away_team_name or e["home_team"] != home_team_name:
            continue
        # KST 날짜 확인
        utc_dt = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        kst_dt = utc_dt.astimezone(timezone(timedelta(hours=9)))
        if kst_dt.strftime("%Y-%m-%d") == kst_date:
            return e["id"]
    return None


def _pick_balanced_line(candidates: list, half_only: bool = True) -> dict | None:
    """여러 (line, over_price, under_price) 후보 중 배당차 최소 = 메인 라인 선택.

    Args:
        candidates: [{"line": 4.5, "over_price": -110, "under_price": -110}, ...]
        half_only: True면 .5 단위 라인만 (push 방지). 정수 라인(4.0 등) 제외.

    Returns:
        배당차 최소 후보 dict 또는 None
    """
    scored = []
    for c in candidates:
        line = c.get("line")
        if line is None:
            continue
        # .5 단위 라인만 (4.0, 5.0 같은 push 가능 라인 제외)
        if half_only and (float(line) * 2) % 2 == 0:
            continue
        op = c.get("over_price")
        up = c.get("under_price")
        if op is None or up is None:
            continue
        od = _to_dec(op)
        ud = _to_dec(up)
        if od is None or ud is None:
            continue
        diff = abs(od - ud)
        scored.append((diff, c))
    if not scored:
        # .5 라인 없으면 정수 라인이라도
        if half_only:
            return _pick_balanced_line(candidates, half_only=False)
        return None
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def get_mlb_f5_and_team_totals() -> list[dict]:
    """F5 마켓 + 팀토탈 + 풀게임 토탈 한번에 조회 (1 credit/경기).

    각 마켓마다 모든 북메이커 × 모든 라인 후보를 수집한 뒤
    **배당차가 가장 작은 라인 = 메인 라인**을 선택한다.

    Returns:
        [{
            "event_id": str,
            "away_team": str,
            "home_team": str,
            "team_totals": {"away": {"line": 2.5, ...}, "home": {"line": 2.5, ...}},
            "f5_moneyline": {"away": -150, "home": 130, ...},
            "f5_total": {"line": 4.5, ...},
            "f5_spread": {"away_line": -0.5, ...},
            "full_game_total": {"line": 8.5, ...},
        }, ...]
    """
    events = get_mlb_events()

    # F5 total 마켓 하나만 요청 — 1 credit/event (필요 없는 팀토탈/머니라인/스프레드/풀게임 제거)
    markets = "totals_1st_5_innings"
    results = []

    for ev in events:
        eid = ev["id"]
        try:
            data = get_event_odds(eid, markets)
            if not data:
                continue

            parsed = {
                "event_id": eid,
                "away_team": ev["away_team"],
                "home_team": ev["home_team"],
                "commence_time": ev.get("commence_time", ""),
                "team_totals": {"away": {}, "home": {}},  # 하위 호환
                "f5_total": {},
            }

            # F5 total 라인 후보 수집 (모든 북메이커 × 모든 라인)
            f5_total_cands = {}  # line → {over_price, under_price}
            for bm in data.get("bookmakers", []):
                for market in bm.get("markets", []):
                    if market.get("key") != "totals_1st_5_innings":
                        continue
                    for o in market.get("outcomes", []):
                        line = o.get("point")
                        if line is None:
                            continue
                        if line not in f5_total_cands:
                            f5_total_cands[line] = {"line": line}
                        if o["name"] == "Over":
                            f5_total_cands[line]["over_price"] = o["price"]
                        elif o["name"] == "Under":
                            f5_total_cands[line]["under_price"] = o["price"]

            # 메인 라인 선택: 배당차 최소 (1.9 vs 1.9에 가장 가까운)
            f5_main = _pick_balanced_line(list(f5_total_cands.values()))
            if f5_main:
                parsed["f5_total"] = f5_main

            results.append(parsed)
        except Exception as e:
            print(f"[odds] event {eid} parse error: {e}")
            continue

    print(f"[odds] F5 totals: {len(results)} games ({len(events)} events)")
    return results
