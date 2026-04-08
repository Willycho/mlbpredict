"""V3 Game Scoring Engine — 두 투수 비교 → 승패/언오버 판정.

Elo 스타일 승률 변환 + 기대 득점 모델.
"""

import os
import json
import numpy as np
from datetime import date

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_DIR, PARK_FACTORS, PARK_HR_FACTORS,
    V3_ELO_D, V3_BASELINE_TOTAL, V3_RUNS_PER_SCORE_POINT,
    V3_PITCHER_WEIGHT_IN_EFFECTIVE, V3_MATCHUP_WEIGHT_IN_EFFECTIVE,
    V3_OFFENSE_WEIGHT, V3_DURABILITY_WEIGHT,
    V3_GRADES,
)
from engine.pitcher_score import score_pitcher
from engine.matchup_score import score_matchup
from engine.weather import get_wind_forecast

import pandas as pd
from scipy import stats as sp_stats

# ============================================================
# 팀 타선 화력 / 투수 소화력 (캐시)
# ============================================================
_game_cache = {}


def _load_game_data():
    """타선/투수 데이터 로드 (한번만)."""
    if _game_cache:
        return

    batters = pd.read_parquet(os.path.join(DATA_DIR, "batters_processed.parquet"))
    pitchers = pd.read_parquet(os.path.join(DATA_DIR, "pitchers_processed.parquet"))
    _game_cache["batters"] = batters
    _game_cache["pitchers"] = pitchers

    # 팀별 타선 화력 퍼센타일
    all_teams = batters[batters["season"] == 2025]["team"].unique()
    team_ops = {}
    for t in all_teams:
        tb = batters[(batters["team"] == t) & (batters["season"] == 2025) & (batters["pa"] >= 50)]
        if tb.empty:
            team_ops[t] = 0.52
        else:
            team_ops[t] = float(np.average(tb["iso"] + tb["babip"] + tb["bb_rate"], weights=tb["pa"]))
    ops_arr = np.array(list(team_ops.values()))
    _game_cache["team_offense"] = {
        t: float(sp_stats.percentileofscore(ops_arr, v, kind="rank"))
        for t, v in team_ops.items()
    }

    # 투수 소화력 풀 (선발 GS>3)
    starters = pitchers[pitchers["gs"] > 3]
    dur_arr = (starters["tbf"] / starters["gs"]).values
    _game_cache["dur_pool"] = dur_arr


def get_team_offense_score(team: str) -> float:
    """팀 타선 화력 퍼센타일 (0-100)."""
    _load_game_data()
    return _game_cache["team_offense"].get(team, 50.0)


def get_pitcher_park_interaction(pitcher_id: int, home_team: str) -> float:
    """투수 유형(GB/FB) × 구장 HR팩터 상호작용.

    플라이볼 투수가 HR 친화 구장에서 던지면 불리 → 마이너스 보정.
    땅볼 투수는 HR 구장에서도 덜 영향 → 보너스.

    Returns:
        보정값 (-5 ~ +5). 양수 = 투수에게 유리.
    """
    _load_game_data()
    pitchers = _game_cache["pitchers"]
    p = pitchers[pitchers["player_id"] == pitcher_id]
    if p.empty:
        return 0.0

    latest = p[p["season"] == p["season"].max()].iloc[0]
    fb_rate = latest.get("fb_rate", 0.27)  # 리그 평균 ~0.27
    gb_rate = latest.get("gb_rate", 0.42)  # 리그 평균 ~0.42

    hr_pf = PARK_HR_FACTORS.get(home_team, 100)

    # 투수 유형 판정: fb_rate - gb_rate로 연속 스케일
    # fb_rate 높으면(+) = 플라이볼형, gb_rate 높으면(-) = 땅볼형
    fb_tendency = fb_rate - 0.27  # 리그 평균 대비 (+면 플라이볼 편향)

    # 구장 HR 편차
    hr_deviation = (hr_pf - 100) / 100  # COL: +0.12, MIA: -0.11

    # 상호작용: 플라이볼 투수 × HR 구장 = 불리 (음수)
    # fb_tendency +0.10 × hr_deviation +0.12 = -0.012 → 투수에게 불리
    interaction = -fb_tendency * hr_deviation * 80  # 스케일링

    return round(np.clip(interaction, -5, 5), 1)


def get_pitcher_durability_score(pitcher_id: int) -> float:
    """투수 이닝 소화력 퍼센타일 (0-100)."""
    _load_game_data()
    pitchers = _game_cache["pitchers"]
    p = pitchers[pitchers["player_id"] == pitcher_id]
    if p.empty:
        return 50.0
    latest = p[p["season"] == p["season"].max()].iloc[0]
    gs = latest.get("gs", 0)
    if gs <= 0:
        return 20.0  # 릴리버/오프너
    dur = latest["tbf"] / gs
    return float(sp_stats.percentileofscore(_game_cache["dur_pool"], dur, kind="rank"))


# ============================================================
# 승률 / 기대 득점 계산
# ============================================================
def compute_win_probability(
    pitcher_score_a: float,
    matchup_score_a: float,
    pitcher_score_b: float,
    matchup_score_b: float,
    offense_a: float = 50.0,
    offense_b: float = 50.0,
    durability_a: float = 50.0,
    durability_b: float = 50.0,
) -> float:
    """Elo 스타일 승률 변환.

    effective = pitcher * pw + matchup * mw + offense * ow + durability * dw
    P(A wins) = 1 / (1 + 10^((eff_B - eff_A) / D))

    Returns: P(A wins) 0.0~1.0
    """
    pw = V3_PITCHER_WEIGHT_IN_EFFECTIVE
    mw = V3_MATCHUP_WEIGHT_IN_EFFECTIVE
    ow = V3_OFFENSE_WEIGHT
    dw = V3_DURABILITY_WEIGHT

    eff_a = pitcher_score_a * pw + matchup_score_a * mw + offense_a * ow + durability_a * dw
    eff_b = pitcher_score_b * pw + matchup_score_b * mw + offense_b * ow + durability_b * dw

    prob_a = 1.0 / (1.0 + 10 ** ((eff_b - eff_a) / V3_ELO_D))
    return prob_a


def compute_expected_total(
    pitcher_score_a: float,
    pitcher_score_b: float,
    home_team: str,
    wind_adj: float = 0.0,
) -> float:
    """양 투수 점수 → 기대 총 득점.

    combined = score_a + score_b (0-200 범위, 평균 ~100)
    expected = baseline - (combined - 100) * runs_per_point
    파크팩터 + 풍향 보정 적용.
    """
    combined = pitcher_score_a + pitcher_score_b
    deviation = combined - 100.0

    expected = V3_BASELINE_TOTAL - deviation * V3_RUNS_PER_SCORE_POINT

    # 파크팩터
    pf = PARK_FACTORS.get(home_team, 100) / 100.0
    expected *= pf

    # 풍향 보정 (scale=1.5, 백테스트 최적)
    expected += wind_adj * 1.5

    return round(max(expected, 1.5), 1)  # F5 최소 1.5


# ============================================================
# 등급 판정
# ============================================================
def grade_prediction(pick_prob: float) -> dict:
    """승률 → 등급.

    Args:
        pick_prob: 픽 방향 확률 (0.5~1.0)

    Returns:
        {"grade": "sweet_spot", "label": "SWEET SPOT"}
    """
    for key in ["sweet_spot", "good", "lean", "overconf"]:
        info = V3_GRADES[key]
        if info["min_prob"] <= pick_prob < info["max_prob"]:
            return {"grade": key, "label": info["label"]}

    if pick_prob >= 0.75:
        return {"grade": "overconf", "label": V3_GRADES["overconf"]["label"]}

    return {"grade": "pass", "label": V3_GRADES["pass"]["label"]}


# ============================================================
# 단일 경기 예측
# ============================================================
def predict_game_v3(game_info: dict, cutoff_date: str = None) -> dict:
    """V3 경기 예측.

    Args:
        game_info: {
            'game_id': int,
            'away_team': str,
            'home_team': str,
            'away_pitcher': {'id': int, 'name': str},
            'home_pitcher': {'id': int, 'name': str},
            'away_lineup': [{'id': int}, ...] or [],
            'home_lineup': [{'id': int}, ...] or [],
        }
        cutoff_date: 최근 폼 기준 날짜

    Returns: full prediction dict
    """
    away_team = game_info["away_team"]
    home_team = game_info["home_team"]
    away_sp = game_info.get("away_pitcher", {})
    home_sp = game_info.get("home_pitcher", {})

    away_pid = away_sp.get("id")
    home_pid = home_sp.get("id")

    # 선발투수 없으면 스킵
    if not away_pid or not home_pid:
        return _tbd_result(game_info)

    # 라인업 ID 추출
    away_lineup_ids = [p["id"] for p in game_info.get("away_lineup", []) if p.get("id")]
    home_lineup_ids = [p["id"] for p in game_info.get("home_lineup", []) if p.get("id")]

    # 1. 투수 스코어
    away_p_score = score_pitcher(away_pid, cutoff_date=cutoff_date, is_home=False)
    home_p_score = score_pitcher(home_pid, cutoff_date=cutoff_date, is_home=True)

    # 데이터 없는 투수는 MLB API 이름이라도 표시
    if away_p_score["name"] == "Unknown" and away_sp.get("name"):
        away_p_score["name"] = away_sp["name"]
        away_p_score["team"] = away_team
    if home_p_score["name"] == "Unknown" and home_sp.get("name"):
        home_p_score["name"] = home_sp["name"]
        home_p_score["team"] = home_team

    # 2. 매치업 스코어
    # away 투수 vs home 타선
    away_matchup = score_matchup(
        away_pid, home_team,
        lineup_ids=home_lineup_ids or None,
    )
    # home 투수 vs away 타선
    home_matchup = score_matchup(
        home_pid, away_team,
        lineup_ids=away_lineup_ids or None,
    )

    # 3. 타선 화력 + 투수 소화력 + 투수유형×구장
    away_offense = get_team_offense_score(away_team)
    home_offense = get_team_offense_score(home_team)
    away_durability = get_pitcher_durability_score(away_pid)
    home_durability = get_pitcher_durability_score(home_pid)
    away_park_int = get_pitcher_park_interaction(away_pid, home_team)
    home_park_int = get_pitcher_park_interaction(home_pid, home_team)

    # 4. 승률
    pw = V3_PITCHER_WEIGHT_IN_EFFECTIVE
    mw = V3_MATCHUP_WEIGHT_IN_EFFECTIVE
    ow = V3_OFFENSE_WEIGHT
    dw = V3_DURABILITY_WEIGHT

    eff_away = (away_p_score["total_score"] * pw + away_matchup["matchup_score"] * mw
                + away_offense * ow + away_durability * dw)
    eff_home = (home_p_score["total_score"] * pw + home_matchup["matchup_score"] * mw
                + home_offense * ow + home_durability * dw)

    win_prob_away = compute_win_probability(
        away_p_score["total_score"], away_matchup["matchup_score"],
        home_p_score["total_score"], home_matchup["matchup_score"],
        away_offense, home_offense,
        away_durability, home_durability,
    )
    win_prob_home = 1.0 - win_prob_away

    # 4. 풍향 예보 + 투수유형×구장 + 기대 총 득점
    wind_data = get_wind_forecast(home_team, game_hour=19)
    wind_adj = wind_data["adj"] if wind_data else 0.0

    # park interaction: 음수 = 투수에게 불리 = 득점 증가
    park_int_adj = -(away_park_int + home_park_int) * 0.05

    expected_total = compute_expected_total(
        away_p_score["total_score"],
        home_p_score["total_score"],
        home_team,
        wind_adj=wind_adj + park_int_adj,
    )

    # 5. 팀 토탈 (각팀 예상 F5 득점)
    away_pitcher_score = away_p_score["total_score"]
    home_pitcher_score = home_p_score["total_score"]
    away_matchup_score = away_matchup["matchup_score"]
    home_matchup_score = home_matchup["matchup_score"]

    # away팀 득점 = home투수가 얼마나 나쁜지
    # home팀 득점 = away투수가 얼마나 나쁜지
    away_strength = (100 - home_pitcher_score) * 0.4 + (100 - home_matchup_score) * 0.6
    home_strength = (100 - away_pitcher_score) * 0.4 + (100 - away_matchup_score) * 0.6
    total_strength = away_strength + home_strength

    if total_strength > 0:
        away_exp_runs = expected_total * (away_strength / total_strength)
        home_exp_runs = expected_total * (home_strength / total_strength)
    else:
        away_exp_runs = expected_total / 2
        home_exp_runs = expected_total / 2

    # 팀 토탈 O/U
    # 배당사 팀토탈은 풀게임 기준 → F5 비율(0.555) 적용 후 .5 단위로 변환
    def to_half_line(val):
        return int(val) + 0.5

    raw_away_line = game_info.get("away_team_total_line")  # 풀게임 기준
    raw_home_line = game_info.get("home_team_total_line")  # 풀게임 기준

    if raw_away_line:
        away_team_line = to_half_line(raw_away_line * 0.555)  # F5 변환
    else:
        away_team_line = to_half_line(away_exp_runs)

    if raw_home_line:
        home_team_line = to_half_line(raw_home_line * 0.555)  # F5 변환
    else:
        home_team_line = to_half_line(home_exp_runs)

    away_tt = {
        "expected": round(away_exp_runs, 2),
        "line": away_team_line,
        "pick": "OVER" if away_exp_runs > away_team_line else "UNDER",
        "gap": round(abs(away_exp_runs - away_team_line), 2),
    }
    home_tt = {
        "expected": round(home_exp_runs, 2),
        "line": home_team_line,
        "pick": "OVER" if home_exp_runs > home_team_line else "UNDER",
        "gap": round(abs(home_exp_runs - home_team_line), 2),
    }

    # 픽 방향
    if win_prob_home > 0.5:
        pick = home_team
        pick_prob = win_prob_home
    else:
        pick = away_team
        pick_prob = win_prob_away

    # 6. 등급 (pick_prob 기반)
    grade_info = grade_prediction(pick_prob)

    # 7. Sweet Picks — ML/OU/팀토탈 중 확신 높은 픽 수집
    sweet_picks = []

    # 라인업 소스 판정
    a_src = away_matchup.get("lineup_source", "roster")
    h_src = home_matchup.get("lineup_source", "roster")
    a_n = away_matchup.get("n_batters", 0)
    h_n = home_matchup.get("n_batters", 0)
    if a_src == "lineup" and h_src == "lineup":
        lineup_tag = "confirmed"
    elif a_src == "lineup" or h_src == "lineup":
        lineup_tag = "partial"
    else:
        lineup_tag = "roster"

    sp_base = {"lineup": lineup_tag, "n_batters": f"{a_n}+{h_n}"}

    # ML sweet spot (65-75%)
    if 0.65 <= pick_prob < 0.75:
        sweet_picks.append({
            **sp_base,
            "type": "ML",
            "pick": f"F5 {pick}",
            "conf": round(pick_prob, 3),
            "detail": f"F5 {pick} ML",
        })

    # Game total O/U — 배당사 F5 라인 우선 사용
    f5_line = game_info.get("f5_total_line")
    if f5_line is None:
        # Fallback 1: 풀게임 totals × 5/9 (풀게임 O/U 라인)
        full_game_line = game_info.get("full_game_total_line")
        if full_game_line:
            f5_line = round((full_game_line * 5 / 9) * 2) / 2  # .5 단위
        else:
            f5_line = 4.5  # Fallback 2
    ou_gap = abs(expected_total - f5_line)
    if ou_gap >= 0.3:
        ou_dir = "OVER" if expected_total > f5_line else "UNDER"
        sweet_picks.append({
            **sp_base,
            "type": "OU",
            "pick": f"F5 {ou_dir} {f5_line}",
            "conf": round(min(ou_gap / 3.0 + 0.5, 0.95), 3),
            "detail": f"Expected {expected_total} vs line {f5_line}",
        })

    # Team totals
    for team, tt in [(away_team, away_tt), (home_team, home_tt)]:
        if tt["gap"] >= 0.3:
            sweet_picks.append({
                **sp_base,
                "type": "TT",
                "pick": f"{team} {tt['pick']} {tt['line']}",
                "conf": round(min(tt["gap"] / 2.0 + 0.5, 0.95), 3),
                "detail": f"Expected {tt['expected']}",
            })

    sweet_picks.sort(key=lambda x: x["conf"], reverse=True)

    return {
        "game_id": game_info.get("game_id"),
        "away_team": away_team,
        "home_team": home_team,
        "game_time_kst": game_info.get("game_time", ""),
        "game_date_kst": game_info.get("game_date_kst", ""),
        "away_pitcher": {
            "id": away_pid,
            "name": away_p_score["name"],
            "team": away_p_score["team"],
            "throws": away_p_score["throws"],
            "total_score": away_p_score["total_score"],
            "season_score": away_p_score["season_score"],
            "arsenal_score": away_p_score["arsenal_score"],
            "recent_form_score": away_p_score["recent_form_score"],
            "home_away_adj": away_p_score["home_away_adj"],
            "no_data": away_p_score.get("no_data", False),
        },
        "home_pitcher": {
            "id": home_pid,
            "name": home_p_score["name"],
            "team": home_p_score["team"],
            "throws": home_p_score["throws"],
            "total_score": home_p_score["total_score"],
            "season_score": home_p_score["season_score"],
            "arsenal_score": home_p_score["arsenal_score"],
            "recent_form_score": home_p_score["recent_form_score"],
            "home_away_adj": home_p_score["home_away_adj"],
            "no_data": home_p_score.get("no_data", False),
        },
        "away_matchup": {
            "matchup_score": away_matchup["matchup_score"],
            "arsenal_matchup": away_matchup["arsenal_matchup_score"],
            "h2h_bonus": away_matchup["h2h_bonus"],
            "lineup_source": away_matchup.get("lineup_source", "roster"),
            "n_batters": away_matchup.get("n_batters", 0),
        },
        "home_matchup": {
            "matchup_score": home_matchup["matchup_score"],
            "arsenal_matchup": home_matchup["arsenal_matchup_score"],
            "h2h_bonus": home_matchup["h2h_bonus"],
            "lineup_source": home_matchup.get("lineup_source", "roster"),
            "n_batters": home_matchup.get("n_batters", 0),
        },
        "effective_away": round(eff_away, 1),
        "effective_home": round(eff_home, 1),
        "win_prob_away": round(win_prob_away, 3),
        "win_prob_home": round(win_prob_home, 3),
        "pick": pick,
        "pick_prob": round(pick_prob, 3),
        "expected_total": expected_total,
        "away_team_total": away_tt,
        "home_team_total": home_tt,
        "sweet_picks": sweet_picks,
        "grade": grade_info["grade"],
        "grade_label": grade_info["label"],
        # 풍향 정보
        "wind": {
            "speed_mph": wind_data["speed_mph"] if wind_data else 0,
            "direction": wind_data["relative"] if wind_data else "none",
            "adj": wind_adj,
        } if wind_data else None,
        # 상세 (UI 렌더링용)
        "away_pitch_detail": away_matchup.get("pitch_detail", []),
        "home_pitch_detail": home_matchup.get("pitch_detail", []),
        "away_h2h_detail": away_matchup.get("h2h_detail", {}),
        "home_h2h_detail": home_matchup.get("h2h_detail", {}),
    }


def _tbd_result(game_info: dict) -> dict:
    """선발투수 미정 경기용 기본 결과."""
    return {
        "game_id": game_info.get("game_id"),
        "away_team": game_info.get("away_team", ""),
        "home_team": game_info.get("home_team", ""),
        "away_pitcher": {"name": "TBD", "total_score": None},
        "home_pitcher": {"name": "TBD", "total_score": None},
        "win_prob_away": None,
        "win_prob_home": None,
        "pick": None,
        "pick_prob": None,
        "expected_total": None,
        "grade": "pass",
        "grade_label": "TBD",
    }


# ============================================================
# 전체 경기 예측 (배치)
# ============================================================
def predict_all_games_v3(game_date: str, schedule: list[dict] = None, skip_odds_fetch: bool = False) -> list[dict]:
    """특정 날짜 전체 경기 V3 예측.

    Args:
        game_date: "YYYY-MM-DD"
        schedule: 미리 조회한 스케줄 (None이면 MLB API 호출)
        skip_odds_fetch: True면 F5 라인 API 재호출 안 함 (schedule에 미리 주입된 f5_total_line 사용)

    Returns: list of prediction dicts
    """
    if schedule is None:
        from data.mlb_api import get_schedule
        schedule = get_schedule(game_date)

    odds_map = {}
    if not skip_odds_fetch:
        try:
            from data.odds_api import get_mlb_f5_and_team_totals
            from config import TEAMS
            name_to_code = {v: k for k, v in TEAMS.items()}
            odds_data = get_mlb_f5_and_team_totals()
            for o in odds_data:
                away_code = name_to_code.get(o["away_team"])
                home_code = name_to_code.get(o["home_team"])
                if away_code and home_code:
                    key = f"{away_code}@{home_code}"
                    odds_map[key] = o
        except Exception as e:
            print(f"[odds] F5 fetch failed: {e}")

    results = []
    for game in schedule:
        # skip_odds_fetch=True면 game["f5_total_line"]가 이미 주입되어 있음
        if not skip_odds_fetch:
            key = f"{game.get('away_team', '')}@{game.get('home_team', '')}"
            odds = odds_map.get(key, {})
            game["f5_total_line"] = odds.get("f5_total", {}).get("line")
        game.setdefault("away_team_total_line", None)
        game.setdefault("home_team_total_line", None)

        pred = predict_game_v3(game, cutoff_date=game_date)
        results.append(pred)

    return results


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    import sys

    game_date = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()

    print(f"\n=== V3 Predictions for {game_date} ===\n")

    from data.mlb_api import get_schedule
    schedule = get_schedule(game_date)
    print(f"Found {len(schedule)} games\n")

    predictions = predict_all_games_v3(game_date, schedule)

    for pred in predictions:
        away = pred["away_team"]
        home = pred["home_team"]
        ap = pred["away_pitcher"]
        hp = pred["home_pitcher"]

        if pred["pick"] is None:
            print(f"  {away} @ {home} — TBD")
            continue

        print(f"  {away} @ {home}")
        print(f"    Away SP: {ap['name']} ({ap.get('total_score', 'N/A')})")
        print(f"    Home SP: {hp['name']} ({hp.get('total_score', 'N/A')})")
        print(f"    Win%: {away} {pred['win_prob_away']:.1%} | {home} {pred['win_prob_home']:.1%}")
        print(f"    Pick: {pred['pick']} ({pred['pick_prob']:.1%})")
        print(f"    Expected Total: {pred['expected_total']}")
        print(f"    Grade: {pred['grade_label']} ({pred['grade']})")
        print()
