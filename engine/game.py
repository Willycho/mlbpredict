"""풀게임(9이닝+) 시뮬레이션 — F5/Full 모드 지원."""

import numpy as np

from engine.inning import simulate_half_inning
from config import PITCHER_STAMINA, MANFRED_RUNNER_INNING


def apply_fatigue(pitcher: dict, pa_faced: int) -> dict:
    """투수 피로도 적용."""
    limit = PITCHER_STAMINA["starter_pa_limit"]
    threshold = PITCHER_STAMINA["fatigue_threshold"]

    if pa_faced < limit * threshold:
        return pitcher

    fatigue = min(1.0, (pa_faced - limit * threshold) / (limit * (1 - threshold)))

    fatigued = dict(pitcher)
    fatigued["k_rate"] = pitcher["k_rate"] * (1 - PITCHER_STAMINA["fatigue_k_penalty"] * fatigue)
    fatigued["bb_rate"] = pitcher["bb_rate"] * (1 + PITCHER_STAMINA["fatigue_bb_penalty"] * fatigue)

    return fatigued


def simulate_game(
    away_lineup: list[dict],
    home_lineup: list[dict],
    away_pitcher: dict,
    home_pitcher: dict,
    away_bullpen: dict,
    home_bullpen: dict,
    league_avg: dict,
    home_team: str,
    rng: np.random.Generator,
    matchup_cache: dict | None = None,
    mode: str = "full",
    away_bullpen_tiered: dict | None = None,
    home_bullpen_tiered: dict | None = None,
    splits_cache: dict | None = None,
) -> dict:
    """한 경기를 시뮬레이션.

    Args:
        mode: "full" (9이닝) 또는 "f5" (5이닝)
        away_bullpen_tiered: {"closer": {...}, "setup": {...}} or None
        home_bullpen_tiered: {"closer": {...}, "setup": {...}} or None
    """
    away_score = 0
    home_score = 0
    away_pos = 0
    home_pos = 0

    current_home_p = dict(home_pitcher)
    current_away_p = dict(away_pitcher)
    home_p_pa = 0
    away_p_pa = 0
    home_p_is_starter = True
    away_p_is_starter = True
    # 불펜 단계 추적: "starter" → "setup" → "closer"
    home_bp_phase = "starter"
    away_bp_phase = "starter"

    inning_scores = []
    away_batter_stats = {}
    home_batter_stats = {}

    max_innings = 5 if mode == "f5" else 15

    for inning in range(1, max_innings + 1):
        ghost = inning >= MANFRED_RUNNER_INNING and mode == "full"

        # === 초 (원정팀 공격, 홈 투수가 던짐) ===
        if home_p_is_starter and home_p_pa >= PITCHER_STAMINA["starter_pa_limit"]:
            # 선발 → 셋업으로 교체
            if home_bullpen_tiered:
                current_home_p = dict(home_bullpen_tiered["setup"])
            else:
                current_home_p = dict(home_bullpen)
            home_p_is_starter = False
            home_bp_phase = "setup"

        # 8회 이상 + 리드 상황 → 클로저 투입
        if (not home_p_is_starter and home_bp_phase == "setup"
                and inning >= 8 and home_score > away_score
                and home_bullpen_tiered):
            current_home_p = dict(home_bullpen_tiered["closer"])
            home_bp_phase = "closer"

        # 9회 + 3점차 이내 리드 → 클로저
        if (not home_p_is_starter and home_bp_phase == "setup"
                and inning >= 9 and home_score > away_score
                and (home_score - away_score) <= 3
                and home_bullpen_tiered):
            current_home_p = dict(home_bullpen_tiered["closer"])
            home_bp_phase = "closer"

        effective_home_p = apply_fatigue(current_home_p, home_p_pa) if home_p_is_starter else current_home_p
        prev_pos = away_pos

        top_runs, away_pos = simulate_half_inning(
            away_lineup, away_pos, effective_home_p, league_avg, home_team, rng, ghost,
            matchup_cache=matchup_cache,
            batter_stats=away_batter_stats,
            splits_cache=splits_cache,
        )
        away_score += top_runs
        home_p_pa += away_pos - prev_pos

        # === 말 (홈팀 공격, 원정 투수가 던짐) ===
        if mode == "full" and inning >= 9 and home_score > away_score:
            inning_scores.append((top_runs, 0))
            break

        if away_p_is_starter and away_p_pa >= PITCHER_STAMINA["starter_pa_limit"]:
            if away_bullpen_tiered:
                current_away_p = dict(away_bullpen_tiered["setup"])
            else:
                current_away_p = dict(away_bullpen)
            away_p_is_starter = False
            away_bp_phase = "setup"

        # 클로저 투입 로직 (원정 투수)
        if (not away_p_is_starter and away_bp_phase == "setup"
                and inning >= 8 and away_score > home_score
                and away_bullpen_tiered):
            current_away_p = dict(away_bullpen_tiered["closer"])
            away_bp_phase = "closer"

        if (not away_p_is_starter and away_bp_phase == "setup"
                and inning >= 9 and away_score > home_score
                and (away_score - home_score) <= 3
                and away_bullpen_tiered):
            current_away_p = dict(away_bullpen_tiered["closer"])
            away_bp_phase = "closer"

        effective_away_p = apply_fatigue(current_away_p, away_p_pa) if away_p_is_starter else current_away_p
        prev_pos = home_pos

        bot_runs, home_pos = simulate_half_inning(
            home_lineup, home_pos, effective_away_p, league_avg, home_team, rng, ghost,
            matchup_cache=matchup_cache,
            batter_stats=home_batter_stats,
            splits_cache=splits_cache,
        )
        home_score += bot_runs
        away_p_pa += home_pos - prev_pos

        inning_scores.append((top_runs, bot_runs))

        if mode == "full":
            if inning >= 9 and home_score > away_score:
                break
            if inning >= 9 and home_score != away_score:
                break

    return {
        "away_score": away_score,
        "home_score": home_score,
        "inning_scores": inning_scores,
        "away_batter_stats": away_batter_stats,
        "home_batter_stats": home_batter_stats,
    }
