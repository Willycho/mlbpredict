"""불펜 가용성 — 전일/이틀 전 투구 이력 기반 가용 여부 판단.

Statcast raw 데이터 또는 MLB API에서 최근 불펜 사용량을 추적하여
오늘 등판 가능 여부와 피로 패널티를 결정.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


# 가용성 규칙
AVAILABILITY_RULES = {
    "unavailable": {
        # 어제 30투구 이상 OR 2이닝 이상 → 오늘 못 나옴
        "yesterday_pitches": 30,
        "yesterday_innings": 2.0,
        # 연속 3일 등판 → 오늘 못 나옴
        "consecutive_days": 3,
    },
    "tired": {
        # 어제 15-29투구 → 오늘 가능하지만 피로 패널티
        "yesterday_pitches_min": 15,
        "k_penalty": 0.03,
        "bb_penalty": 0.03,
    },
    "back_to_back": {
        # 어제 + 그저께 연속 등판 → 피로 패널티
        "k_penalty": 0.02,
        "bb_penalty": 0.02,
    },
}


def get_reliever_usage(
    raw: pd.DataFrame,
    team: str,
    game_date: str,
    lookback_days: int = 3,
) -> dict:
    """팀 릴리버들의 최근 등판 이력.

    Returns:
        {pitcher_id: {
            "dates": ["2025-07-14", "2025-07-15"],
            "yesterday_pitches": 22,
            "yesterday_innings": 1.0,
            "consecutive_days": 2,
            "status": "available" | "tired" | "unavailable",
            "k_penalty": 0.0,
            "bb_penalty": 0.0,
        }}
    """
    cutoff = pd.Timestamp(game_date)
    start = cutoff - timedelta(days=lookback_days)
    yesterday = cutoff - timedelta(days=1)
    day_before = cutoff - timedelta(days=2)

    # 해당 팀 투수 데이터 (해당 기간)
    team_data = raw[
        (raw["game_date"] >= start)
        & (raw["game_date"] < cutoff)
        & (
            ((raw["inning_topbot"] == "Top") & (raw["home_team"] == team))
            | ((raw["inning_topbot"] == "Bot") & (raw["away_team"] == team))
        )
    ]

    # 선발투수 제외 (1이닝 첫 등판이 아닌 투수 = 릴리버)
    starters = team_data[team_data["inning"] == 1].groupby("game_pk")["pitcher"].first()
    starter_set = set(starters.values)

    reliever_data = team_data[~team_data["pitcher"].isin(starter_set)]

    usage = {}
    for pid, grp in reliever_data.groupby("pitcher"):
        dates_pitched = sorted(grp["game_date"].dt.date.unique())

        # 어제 투구 수/이닝
        yest_data = grp[grp["game_date"].dt.date == yesterday.date()]
        yest_pitches = len(yest_data)
        yest_innings = yest_data["inning"].nunique() if len(yest_data) > 0 else 0

        # 연속 등판일
        consecutive = 0
        check_date = yesterday
        while check_date.date() in [d for d in dates_pitched]:
            consecutive += 1
            check_date -= timedelta(days=1)

        # 가용 상태 판단
        rules = AVAILABILITY_RULES
        status = "available"
        k_pen = 0.0
        bb_pen = 0.0

        if (yest_pitches >= rules["unavailable"]["yesterday_pitches"]
                or yest_innings >= rules["unavailable"]["yesterday_innings"]
                or consecutive >= rules["unavailable"]["consecutive_days"]):
            status = "unavailable"
        elif yest_pitches >= rules["tired"]["yesterday_pitches_min"]:
            status = "tired"
            k_pen = rules["tired"]["k_penalty"]
            bb_pen = rules["tired"]["bb_penalty"]
        elif (yesterday.date() in dates_pitched and day_before.date() in dates_pitched):
            status = "tired"
            k_pen = rules["back_to_back"]["k_penalty"]
            bb_pen = rules["back_to_back"]["bb_penalty"]

        usage[int(pid)] = {
            "dates": [str(d) for d in dates_pitched],
            "yesterday_pitches": yest_pitches,
            "yesterday_innings": yest_innings,
            "consecutive_days": consecutive,
            "status": status,
            "k_penalty": k_pen,
            "bb_penalty": bb_pen,
        }

    return usage


def apply_bullpen_availability(
    bullpen_profiles: dict,
    reliever_usage: dict,
    team_pitchers_map: dict | None = None,
) -> dict:
    """불펜 프로필에 가용성 반영.

    unavailable 릴리버 → 해당 역할을 리그 평균으로 대체 (degraded)
    tired 릴리버 → K%↓, BB%↑ 패널티 적용

    Args:
        bullpen_profiles: {role: pitcher_dict}
        reliever_usage: get_reliever_usage() 반환값
        team_pitchers_map: {role: pitcher_id} 매핑 (있으면)

    Returns:
        adjusted bullpen_profiles
    """
    if not reliever_usage:
        return bullpen_profiles

    result = {}
    for role, pitcher in bullpen_profiles.items():
        p = dict(pitcher)
        pid = p.get("player_id")

        # pitcher_id가 숫자가 아닌 경우 (bp_TEX_closer 등) → 매핑 시도
        if isinstance(pid, str) and team_pitchers_map:
            pid = team_pitchers_map.get(role)

        if pid and pid in reliever_usage:
            info = reliever_usage[pid]
            if info["status"] == "unavailable":
                # degraded: K%↓10%, BB%↑10%
                p["k_rate"] = max(0.10, p.get("k_rate", 0.22) * 0.90)
                p["bb_rate"] = min(0.20, p.get("bb_rate", 0.08) * 1.10)
                p["_availability"] = "unavailable"
            elif info["status"] == "tired":
                p["k_rate"] = max(0.10, p.get("k_rate", 0.22) - info["k_penalty"])
                p["bb_rate"] = min(0.20, p.get("bb_rate", 0.08) + info["bb_penalty"])
                p["_availability"] = "tired"
            else:
                p["_availability"] = "available"
        else:
            p["_availability"] = "unknown"

        result[role] = p

    return result


def get_reliever_usage_from_api(team_code: str, game_date: str) -> dict:
    """MLB API에서 최근 불펜 사용 이력 (Statcast raw 없이 실시간용).

    Returns same format as get_reliever_usage().
    """
    import requests
    from datetime import datetime

    cutoff = datetime.strptime(game_date, "%Y-%m-%d")
    results = {}

    for days_back in range(1, 4):
        check_date = (cutoff - timedelta(days=days_back)).strftime("%Y-%m-%d")
        try:
            resp = requests.get(
                "https://statsapi.mlb.com/api/v1/schedule",
                params={
                    "sportId": 1, "gameType": "R", "date": check_date,
                    "hydrate": "pitchers",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            for d in data.get("dates", []):
                for g in d.get("games", []):
                    if g.get("status", {}).get("detailedState") != "Final":
                        continue

                    # 해당 팀 경기인지 확인
                    from data.mlb_api import TEAM_ID_TO_CODE
                    away_code = TEAM_ID_TO_CODE.get(g["teams"]["away"]["team"]["id"], "")
                    home_code = TEAM_ID_TO_CODE.get(g["teams"]["home"]["team"]["id"], "")

                    if team_code not in (away_code, home_code):
                        continue

                    # pitchers 정보에서 릴리버 추출
                    # (simplified — full implementation would parse boxscore)
                    pass

        except Exception:
            continue

    return results
