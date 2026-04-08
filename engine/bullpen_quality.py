"""불펜 2차 정밀화 — platoon/inherited runner/leverage 보정.

C++ sim 코어를 건드리지 않고, matchup cache에 넘기기 전
Python 레이어에서 불펜 확률 벡터를 보정.
"""

from config import PLATOON_MULTIPLIERS


# ============================================================
# 1. Reliever Platoon Adjustment
# ============================================================

def adjust_bullpen_platoon(bp_profile: dict, lineup: list[dict]) -> dict:
    """불펜 투수의 손잡이 vs 라인업 타자 손잡이 분포 기반 K%/HR% 보정.

    현재 platoon은 matchup 레벨에서 OFF이지만,
    불펜은 의도적으로 좌타자 잡으려고 좌완을 넣는 등 platoon이 실질적.

    Args:
        bp_profile: {"k_rate": ..., "bb_rate": ..., "throws": "L"|"R", ...}
        lineup: 9명 타자 리스트 [{"bats": "L"|"R", ...}, ...]

    Returns:
        adjusted bp_profile with platoon-weighted stats
    """
    throws = bp_profile.get("throws", "R")
    if not lineup:
        return bp_profile

    # Count same-hand vs opposite-hand batters in lineup
    same = sum(1 for b in lineup if b.get("bats", "R") == throws)
    opp = len(lineup) - same
    total = same + opp
    if total == 0:
        return bp_profile

    same_ratio = same / total
    opp_ratio = opp / total

    # Weighted platoon effect (subtle: +-2% instead of full 5-8%)
    # Relievers are selected FOR platoon advantage, so effect is smaller
    RELIEVER_PLATOON_SCALE = 0.40  # 40% of full platoon effect

    result = dict(bp_profile)
    for stat, same_mult, opp_mult in [
        ("k_rate", 1.05, 0.92),
        ("bb_rate", 0.95, 1.05),
        ("hr_rate", 0.90, 1.12),
    ]:
        if stat not in result:
            continue
        # Weighted multiplier
        weighted_mult = same_ratio * same_mult + opp_ratio * opp_mult
        # Scale down (relievers already selected for platoon advantage)
        adj_mult = 1.0 + (weighted_mult - 1.0) * RELIEVER_PLATOON_SCALE
        result[stat] = result[stat] * adj_mult

    return result


# ============================================================
# 2. Inherited Runner Penalty
# ============================================================

# MLB average: ~30% of inherited runners score
# Good relievers: ~25%, Bad relievers: ~35%
INHERITED_RUNNER_BASE_RATE = 0.30

def calc_inherited_runner_penalty(bp_profile: dict) -> float:
    """불펜 투수의 inherited runner 처리 능력 추정.

    K%가 높고 BB%가 낮으면 주자 상황에서 유리.
    Returns: penalty multiplier for BABIP (1.0 = neutral, >1.0 = worse).
    """
    k = bp_profile.get("k_rate", 0.22)
    bb = bp_profile.get("bb_rate", 0.08)

    # K-BB% as quality proxy
    k_bb = k - bb
    # League avg K-BB% for relievers: ~0.14
    # penalty = 1.0 for avg, >1.0 for bad, <1.0 for good
    penalty = 1.0 - (k_bb - 0.14) * 0.5  # +-7%p K-BB → +-3.5% BABIP
    return max(0.90, min(1.10, penalty))


def apply_inherited_runner_adj(bp_profile: dict) -> dict:
    """Inherited runner penalty를 BABIP에 반영."""
    penalty = calc_inherited_runner_penalty(bp_profile)
    result = dict(bp_profile)
    result["babip"] = result.get("babip", 0.300) * penalty
    result["_ir_penalty"] = round(penalty, 3)
    return result


# ============================================================
# 3. Leverage-aware Role Selection
# ============================================================

def get_leverage_role(inning: int, score_diff: int, outs: int, runners_on: bool) -> str:
    """상황별 최적 불펜 역할 선택.

    기존: 이닝 기반 고정
    개선: 점수차 + 이닝 + 주자 상황 반영

    Returns: role name (setup_early, setup_late, bridge, closer)
    """
    # Save situation: leading by 1-3, 9th+
    if inning >= 9 and 1 <= score_diff <= 3:
        return "closer"

    # High leverage: tie or 1-run game, 7th+
    if inning >= 7 and abs(score_diff) <= 1:
        if inning >= 8:
            return "bridge"
        return "setup_late"

    # Medium leverage: 2-3 run game, 7th+
    if inning >= 7 and abs(score_diff) <= 3:
        return "setup_late"

    # Low leverage / early relief
    return "setup_early"


# ============================================================
# 4. Combined bullpen enhancement
# ============================================================

def enhance_bullpen_profiles(
    bullpen: dict,
    opponent_lineup: list[dict] = None,
) -> dict:
    """불펜 프로필에 platoon + inherited runner 보정 적용.

    Args:
        bullpen: {role: pitcher_dict} (4 roles)
        opponent_lineup: 상대 타선 (platoon 계산용)

    Returns:
        enhanced bullpen dict
    """
    result = {}
    for role, pitcher in bullpen.items():
        p = dict(pitcher)

        # Platoon adjustment
        if opponent_lineup:
            p = adjust_bullpen_platoon(p, opponent_lineup)

        # Inherited runner adjustment
        p = apply_inherited_runner_adj(p)

        result[role] = p

    return result
