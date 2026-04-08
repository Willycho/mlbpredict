"""v2 Conditional Base-Out Markov Transition Engine.

v1 baserunning.py 대��. 핵심 변경:
1. 고정 확률 → Statcast 기반 조건부 전이 행렬
2. speed bucket / defense bucket 런타임 modifier
3. 30 관측 미만 셀 → v1 결정론적 로직 폴백
"""

import os
import numpy as np
import pandas as pd

from config import (
    V2_DATA_DIR, SPEED_BUCKETS, DEFENSE_BUCKETS,
    TRANSITION_MIN_OBS, BASE_ADVANCEMENT, LEAGUE_AVG_SPRINT_SPEED,
)


# ============================================================
# Transition Matrix loader
# ============================================================

_tm_cache = None


def load_transition_matrix(path: str = None) -> pd.DataFrame:
    """전이 행렬 로드 (캐시)."""
    global _tm_cache
    if _tm_cache is not None:
        return _tm_cache

    if path is None:
        path = os.path.join(V2_DATA_DIR, "transition_matrix.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Transition matrix not found: {path}")

    _tm_cache = pd.read_parquet(path)
    return _tm_cache


def clear_cache():
    global _tm_cache
    _tm_cache = None


# ============================================================
# Transition lookup
# ============================================================

def get_transition_distribution(
    tm: pd.DataFrame,
    base_before: str,
    outs_before: int,
    event_type: str,
) -> pd.DataFrame | None:
    """특정 (pre-state, event)의 전이 분포 조회.

    Returns:
        DataFrame with columns: base_after, outs_after, runs_scored, probability, n_obs
        None if no data.
    """
    mask = (
        (tm["base_before"] == base_before)
        & (tm["outs_before"] == outs_before)
        & (tm["event_type"] == event_type)
    )
    subset = tm[mask]
    if len(subset) == 0:
        return None
    return subset


# ============================================================
# Runtime modifiers
# ============================================================

def apply_speed_modifier(dist: pd.DataFrame, speed_bucket: str) -> pd.DataFrame:
    """주자 속도에 따른 전이 확률 보정.

    fast: 진루 확률 ↑ (더 많은 베이스 진루하는 전이에 가중)
    slow: 진루 확률 ↓
    """
    if speed_bucket == "avg":
        return dist

    modifier = SPEED_BUCKETS.get(speed_bucket, 1.0)
    dist = dist.copy()

    # "진루가 더 많은" 전이에 가중치 부여
    # 기준: base_after에 주자가 더 앞에 있거나 runs_scored가 더 높은 전이
    dist["advancement_score"] = (
        dist["runs_scored"] * 2.0
        + dist["base_after"].apply(_count_advanced_bases)
    )

    # 중앙값 기준으로 위/아래 분리
    median_score = dist["advancement_score"].median()
    high_adv = dist["advancement_score"] > median_score
    low_adv = dist["advancement_score"] <= median_score

    # modifier 적용
    dist.loc[high_adv, "probability"] *= modifier
    dist.loc[low_adv, "probability"] *= (2.0 - modifier)  # 반대 ��향 보정

    # 재정규화
    total = dist["probability"].sum()
    if total > 0:
        dist["probability"] /= total

    dist = dist.drop(columns=["advancement_score"])
    return dist


def apply_defense_modifier(dist: pd.DataFrame, defense_bucket: str) -> pd.DataFrame:
    """수비 수준에 따른 전이 확률 보정.

    good defense: 진루 ↓, 아웃 ↑
    poor defense: 진루 ↑, ��웃 ↓
    """
    if defense_bucket == "avg":
        return dist

    modifier = DEFENSE_BUCKETS.get(defense_bucket, 1.0)
    dist = dist.copy()

    # 아웃이 증가하는 전이 vs 안타/진루가 많은 전이
    is_out_increase = dist["outs_after"] > dist.iloc[0]["outs_after"] if len(dist) > 0 else pd.Series(dtype=bool)

    # good defense = modifier < 1.0 → 안타 전이 ↓, 아웃 전이 ↑
    # 여기서는 반대로: runs/진루가 많은 전이에 modifier 적용
    has_runs = dist["runs_scored"] > 0
    has_advancement = dist["base_after"].apply(_count_runners) > 0

    advancing = has_runs | has_advancement
    dist.loc[advancing, "probability"] *= modifier
    dist.loc[~advancing, "probability"] *= (2.0 - modifier)

    total = dist["probability"].sum()
    if total > 0:
        dist["probability"] /= total

    return dist


def _count_advanced_bases(base_after: str) -> int:
    """base_after 문자열에서 진루 점수 계산."""
    return sum(1 for c in str(base_after) if c == "1")


def _count_runners(base_after: str) -> int:
    return sum(1 for c in str(base_after) if c == "1")


# ============================================================
# V1 fallback (deterministic baserunning)
# ============================================================

def _v1_fallback(
    outcome: str,
    bases: list,
    outs: int,
    rng: np.random.Generator,
) -> tuple[str, int, int]:
    """v1 baserunning 로직 간소화 폴백.

    Returns: (base_after, outs_after, runs_scored)
    """
    on_1b, on_2b, on_3b = bases
    runs = 0
    new_bases = [False, False, False]

    if outcome == "HR":
        runs = 1 + sum(1 for b in bases if b)
        return "000", outs, runs

    elif outcome == "3B":
        runs = sum(1 for b in bases if b)
        new_bases[2] = True
        return _encode(new_bases), outs, runs

    elif outcome == "2B":
        if on_3b: runs += 1
        if on_2b: runs += 1
        if on_1b:
            if rng.random() < BASE_ADVANCEMENT["double"]["first_to_home"]:
                runs += 1
            else:
                new_bases[2] = True
        new_bases[1] = True
        return _encode(new_bases), outs, runs

    elif outcome == "1B":
        if on_3b: runs += 1
        if on_2b:
            if rng.random() < BASE_ADVANCEMENT["single"]["second_to_home"]:
                runs += 1
            else:
                new_bases[2] = True
        if on_1b:
            if rng.random() < BASE_ADVANCEMENT["single"]["first_to_third"]:
                new_bases[2] = True
            else:
                new_bases[1] = True
        new_bases[0] = True
        return _encode(new_bases), outs, runs

    elif outcome in ("BB", "HBP"):
        if on_1b and on_2b and on_3b:
            runs += 1
        new = [False, False, False]
        if on_1b:
            if on_2b:
                if on_3b:
                    runs += 1
                new[2] = True
            new[1] = True
        new[0] = True
        return _encode(new), outs, runs

    elif outcome == "K":
        return _encode([on_1b, on_2b, on_3b]), outs + 1, 0

    elif outcome == "GO":
        added_outs = 1
        if on_1b and outs < 2:
            if rng.random() < BASE_ADVANCEMENT["ground_out"]["double_play_rate"]:
                added_outs = 2
                new_bases = [False, on_2b, on_3b]
                if on_3b and outs < 2:
                    if rng.random() < BASE_ADVANCEMENT["ground_out"]["third_to_home_less2"]:
                        runs += 1
                        new_bases[2] = False
                return _encode(new_bases), outs + added_outs, runs
        if on_3b and outs < 2:
            if rng.random() < BASE_ADVANCEMENT["ground_out"]["third_to_home_less2"]:
                runs += 1
        return _encode([False, on_2b or on_1b, on_3b if runs == 0 else False]), outs + 1, runs

    elif outcome in ("FO", "LO"):
        added_outs = 1
        if on_3b and outs < 2 and outcome == "FO":
            if rng.random() < BASE_ADVANCEMENT["fly_out"]["third_to_home_sac"]:
                runs += 1
        return _encode([on_1b, on_2b, on_3b if runs == 0 else False]), outs + added_outs, runs

    elif outcome in ("FC", "ROE"):
        # FC: 타자 출루, 선행 주자 하나 아웃 (간소화)
        if outcome == "FC":
            new_bases[0] = True
            if on_1b:
                new_bases[1] = True
            return _encode(new_bases), outs + 1, 0
        else:  # ROE
            new_bases[0] = True
            if on_3b: runs += 1
            if on_2b: new_bases[2] = True
            if on_1b: new_bases[1] = True
            return _encode(new_bases), outs, runs

    elif outcome == "SAC":
        if on_3b and outs < 2:
            runs += 1
        new = [False, False, False]
        if on_1b: new[1] = True
        if on_2b: new[2] = True
        return _encode(new), outs + 1, runs

    return _encode([on_1b, on_2b, on_3b]), outs + 1, 0


def _encode(bases) -> str:
    return "".join("1" if b else "0" for b in bases)


# ============================================================
# Main resolve function
# ============================================================

def resolve_transition(
    base_before: str,
    outs_before: int,
    event_type: str,
    speed_bucket: str = "avg",
    defense_bucket: str = "avg",
    rng: np.random.Generator = None,
    tm: pd.DataFrame = None,
) -> tuple[str, int, int]:
    """base-out 전이 결정.

    Args:
        base_before: "000"~"111" (1B/2B/3B 유무)
        outs_before: 0, 1, 2
        event_type: K, BB, HBP, HR, 1B, 2B, 3B, GO, FO, LO, FC, ROE, SAC
        speed_bucket: fast/avg/slow
        defense_bucket: good/avg/poor
        rng: numpy random generator

    Returns:
        (base_after, outs_after, runs_scored)
    """
    if rng is None:
        rng = np.random.default_rng()

    if tm is None:
        try:
            tm = load_transition_matrix()
        except FileNotFoundError:
            # 전이 행렬 없으면 v1 폴백
            bases = [c == "1" for c in base_before]
            return _v1_fallback(event_type, bases, outs_before, rng)

    # 전이 분포 조회
    dist = get_transition_distribution(tm, base_before, outs_before, event_type)

    # 관측 부족 시 v1 폴백
    if dist is None or dist["n_obs"].sum() < TRANSITION_MIN_OBS:
        bases = [c == "1" for c in base_before]
        return _v1_fallback(event_type, bases, outs_before, rng)

    dist = dist.copy()

    # speed/defense modifier 적용
    dist = apply_speed_modifier(dist, speed_bucket)
    dist = apply_defense_modifier(dist, defense_bucket)

    # 샘플링
    probs = dist["probability"].values
    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if total <= 0:
        bases = [c == "1" for c in base_before]
        return _v1_fallback(event_type, bases, outs_before, rng)
    probs = probs / total

    idx = rng.choice(len(dist), p=probs)
    row = dist.iloc[idx]

    base_after = str(row["base_after"])
    outs_after = int(row["outs_after"])
    runs_scored = int(row["runs_scored"])

    return base_after, outs_after, runs_scored
