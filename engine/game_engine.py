"""v2 Game Engine — 4티어 불펜 + 투구수 기반 피로.

v1 game.py 대체. 핵심 변경:
1. 4티어 불펜 (setup_early, setup_late, bridge, closer)
2. 투구수 기반 피로 (PA × 4.0 근사)
3. markov_inning 사용
"""

import numpy as np

from config import (
    MANFRED_RUNNER_INNING, PITCH_COUNT_PER_PA, FATIGUE_TIERS, BULLPEN_ROLES,
)
from engine.markov_inning import simulate_half_inning


def apply_fatigue_v2(pitcher: dict, pa_faced: int) -> dict:
    """투구수 기반 피로 보정.

    PA × 4.0 = 추정 투구수. 구간별 차등 패널티.
    """
    est_pitches = pa_faced * PITCH_COUNT_PER_PA

    mild = FATIGUE_TIERS["mild"]
    moderate = FATIGUE_TIERS["moderate"]

    if est_pitches < mild["threshold"]:
        return pitcher

    result = dict(pitcher)

    if est_pitches >= moderate["threshold"]:
        k_pen = moderate["k_penalty"]
        bb_pen = moderate["bb_penalty"]
    else:
        k_pen = mild["k_penalty"]
        bb_pen = mild["bb_penalty"]

    result["k_rate"] = max(0.05, result.get("k_rate", 0.22) - k_pen)
    result["bb_rate"] = min(0.25, result.get("bb_rate", 0.08) + bb_pen)

    return result


def should_pull_starter(pa_faced: int, inning: int) -> bool:
    """선발 교체 필요 여부."""
    est_pitches = pa_faced * PITCH_COUNT_PER_PA
    return est_pitches >= FATIGUE_TIERS["pull"]["threshold"]


def get_bullpen_pitcher(
    bullpen_profiles: dict,
    inning: int,
    score_diff: int,
    current_role_idx: int,
) -> tuple[dict, int]:
    """이닝/점수 상황에 따른 불펜 투수 선택.

    Args:
        bullpen_profiles: {role: pitcher_dict} 또는 리스트
        inning: 현재 이닝
        score_diff: 양수 = 리드, 음수 = 뒤짐
        current_role_idx: 현재까지 사용한 역할 인덱스

    Returns:
        (투수 dict, 업데이트된 role_idx)
    """
    if isinstance(bullpen_profiles, dict):
        roles = bullpen_profiles
    else:
        roles = {"setup_early": bullpen_profiles}

    # 이닝 기반 역할 선택
    if inning >= 9 and abs(score_diff) <= 3:
        target_role = "closer"
    elif inning >= 8:
        target_role = "bridge"
    elif inning >= 7:
        target_role = "setup_late"
    else:
        target_role = "setup_early"

    # 역할이 없으면 가장 가까운 역할 사용
    if target_role in roles:
        return roles[target_role], BULLPEN_ROLES.index(target_role)

    # fallback: 아무 역할이나
    for role in BULLPEN_ROLES:
        if role in roles:
            return roles[role], BULLPEN_ROLES.index(role)

    # 최종 fallback: 첫 번째 값
    first = next(iter(roles.values()))
    return first, 0


def simulate_game(
    away_lineup: list[dict],
    home_lineup: list[dict],
    away_starter: dict,
    home_starter: dict,
    away_bullpen: dict,
    home_bullpen: dict,
    league_avg: dict,
    home_team: str,
    rng: np.random.Generator = None,
    mode: str = "full",
    matchup_cache: dict = None,
    splits_cache: dict = None,
    count_pitch_mix=None,
    away_defense_bucket: str = "avg",
    home_defense_bucket: str = "avg",
    speed_cache: dict = None,
    tm=None,
) -> dict:
    """전체 게임 시뮬레이션.

    Returns:
        dict with: away_score, home_score, inning_scores,
                   away_batter_stats, home_batter_stats
    """
    if rng is None:
        rng = np.random.default_rng()

    max_innings = 5 if mode == "f5" else 15

    away_pos = 0
    home_pos = 0
    away_score = 0
    home_score = 0
    inning_scores = []

    away_batter_stats = {}
    home_batter_stats = {}

    # 투수 상태 추적
    away_pitcher = away_starter
    home_pitcher = home_starter
    away_pa_faced = 0
    home_pa_faced = 0
    away_is_starter = True
    home_is_starter = True
    away_role_idx = -1
    home_role_idx = -1

    for inning in range(1, max_innings + 1):
        ghost = (mode == "full" and inning >= MANFRED_RUNNER_INNING)

        # === 초 (Away 공격, Home 수비) ===
        # 투수 교체 체크
        if home_is_starter and should_pull_starter(home_pa_faced, inning):
            home_pitcher, home_role_idx = get_bullpen_pitcher(
                home_bullpen, inning, home_score - away_score, home_role_idx
            )
            home_is_starter = False
            home_pa_faced = 0

        # 피로 적용
        effective_home_p = home_pitcher
        if home_is_starter:
            effective_home_p = apply_fatigue_v2(home_pitcher, home_pa_faced)

        top_runs, away_pos = simulate_half_inning(
            lineup=away_lineup,
            lineup_pos=away_pos,
            pitcher=effective_home_p,
            league_avg=league_avg,
            home_team=home_team,
            is_batter_home=False,
            ghost_runner=ghost,
            matchup_cache=matchup_cache,
            batter_stats=away_batter_stats,
            splits_cache=splits_cache,
            count_pitch_mix=count_pitch_mix,
            defense_bucket=home_defense_bucket,
            speed_cache=speed_cache,
            tm=tm,
            rng=rng,
        )
        away_score += top_runs
        home_pa_faced += (away_pos - (away_pos - 3))  # 근사: 최소 3 PA

        # === 말 (Home 공격, Away 수비) ===
        # 9회 이후 홈팀 리드 시 종료
        if mode == "full" and inning >= 9 and home_score > away_score:
            inning_scores.append({"inning": inning, "away": top_runs, "home": 0})
            break

        # 투수 교체 체크
        if away_is_starter and should_pull_starter(away_pa_faced, inning):
            away_pitcher, away_role_idx = get_bullpen_pitcher(
                away_bullpen, inning, away_score - home_score, away_role_idx
            )
            away_is_starter = False
            away_pa_faced = 0

        effective_away_p = away_pitcher
        if away_is_starter:
            effective_away_p = apply_fatigue_v2(away_pitcher, away_pa_faced)

        bot_runs, home_pos = simulate_half_inning(
            lineup=home_lineup,
            lineup_pos=home_pos,
            pitcher=effective_away_p,
            league_avg=league_avg,
            home_team=home_team,
            is_batter_home=True,
            ghost_runner=ghost,
            matchup_cache=matchup_cache,
            batter_stats=home_batter_stats,
            splits_cache=splits_cache,
            count_pitch_mix=count_pitch_mix,
            defense_bucket=away_defense_bucket,
            speed_cache=speed_cache,
            tm=tm,
            rng=rng,
        )
        home_score += bot_runs
        away_pa_faced += 3  # 근사

        inning_scores.append({
            "inning": inning,
            "away": top_runs,
            "home": bot_runs,
        })

        # 9회 이후 승부 결정 시 종료
        if mode == "full" and inning >= 9 and away_score != home_score:
            break

    return {
        "away_score": away_score,
        "home_score": home_score,
        "inning_scores": inning_scores,
        "away_batter_stats": away_batter_stats,
        "home_batter_stats": home_batter_stats,
    }
