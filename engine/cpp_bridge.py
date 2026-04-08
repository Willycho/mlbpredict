"""Python <-> C++ bridge: 데이터 변환 + C++ 시뮬레이션 호출."""

import os
import numpy as np
import pandas as pd

from config import (
    V2_DATA_DIR, SPEED_BUCKETS, DEFENSE_BUCKETS,
    PROBABILITY_DAMPENING_ALPHA, TRANSITION_MIN_OBS,
    FATIGUE_TIERS, SHRINKAGE,
)

# Event type -> index mapping (must match sim_core.h enum)
EVENT_TO_IDX = {
    "K": 0, "BB": 1, "HBP": 2, "HR": 3,
    "1B": 4, "2B": 5, "3B": 6,
    "GO": 7, "FO": 8, "LO": 9,
    "FC": 10, "ROE": 11, "SAC": 12,
}

# Base state string -> index mapping
BASE_TO_IDX = {
    "000": 0, "100": 1, "010": 2, "110": 3,
    "001": 4, "101": 5, "011": 6, "111": 7,
}

SPEED_BUCKET_TO_IDX = {"slow": 0, "avg": 1, "fast": 2}
DEFENSE_BUCKET_TO_IDX = {"poor": 0, "avg": 1, "good": 2}


def prepare_transition_arrays(tm_df: pd.DataFrame):
    """Transition matrix DataFrame -> numpy arrays for C++.

    Returns:
        transition_indices: [N, 3] int32 (base_before, outs_before, event)
        transition_results: [N, 3] int32 (base_after, outs_after, runs_scored)
        transition_probs: [N] float32
        transition_obs: [N] int32
    """
    if tm_df is None or len(tm_df) == 0:
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int32),
        )

    n = len(tm_df)
    indices = np.zeros((n, 3), dtype=np.int32)
    results = np.zeros((n, 3), dtype=np.int32)
    probs = np.zeros(n, dtype=np.float32)
    obs = np.zeros(n, dtype=np.int32)

    for i, (_, row) in enumerate(tm_df.iterrows()):
        base_before = BASE_TO_IDX.get(str(row["base_before"]), 0)
        outs_before = int(row["outs_before"])
        event = EVENT_TO_IDX.get(str(row["event_type"]), 0)

        base_after = BASE_TO_IDX.get(str(row["base_after"]), 0)
        outs_after = min(int(row["outs_after"]), 3)
        runs = int(row["runs_scored"])

        indices[i] = [base_before, outs_before, event]
        results[i] = [base_after, outs_after, runs]
        probs[i] = float(row["probability"])
        obs[i] = int(row["n_obs"])

    return indices, results, probs, obs


def prepare_matchup_cache(
    away_lineup: list[dict],
    home_lineup: list[dict],
    matchup_cache: dict,
    away_bullpen: dict,
    home_bullpen: dict,
) -> np.ndarray:
    """매치업 캐시 -> [2, 9, 5, 13] float32 array.

    dim0: team (0=away, 1=home)
    dim1: batter index (0-8)
    dim2: pitcher index (0=opp_starter, 1-4=opp_bullpen roles)
    dim3: probs (0-4=stage1, 5-12=stage2)
    """
    cache = np.zeros((2, 9, 5, 13), dtype=np.float32)

    bullpen_roles = ["setup_early", "setup_late", "bridge", "closer"]

    for team_idx, (lineup, opp_bullpen) in enumerate([
        (away_lineup, home_bullpen),
        (home_lineup, away_bullpen),
    ]):
        for b_idx, batter in enumerate(lineup[:9]):
            batter_id = batter.get("player_id")
            if batter_id is None:
                continue

            # All pitcher indices this batter might face
            pitcher_keys = []
            # idx 0: opponent starter
            for key in matchup_cache:
                if key[0] == batter_id:
                    pitcher_keys.append(key)
                    break

            # If we have a cached matchup, use it
            for p_idx, key in enumerate(pitcher_keys[:1]):
                probs = matchup_cache.get(key)
                if probs is None:
                    continue
                _fill_probs(cache, team_idx, b_idx, 0, probs)

            # Bullpen: use cached matchup, or build from bullpen profile directly
            for role_idx, role in enumerate(bullpen_roles):
                bp = opp_bullpen.get(role)
                if bp is None:
                    continue
                bp_id = bp.get("player_id")
                bp_key = (batter_id, bp_id)
                bp_probs = matchup_cache.get(bp_key)
                if bp_probs is not None:
                    _fill_probs(cache, team_idx, b_idx, 1 + role_idx, bp_probs)
                else:
                    # Build from bullpen profile stats directly (no starter copy)
                    _fill_probs_from_raw(cache, team_idx, b_idx, 1 + role_idx, bp)

    return cache


def _fill_probs_from_raw(cache: np.ndarray, team: int, batter: int, pitcher: int, bp_profile: dict):
    """Build Stage1+Stage2 directly from bullpen pitcher raw stats (no matchup computation).

    This avoids copying starter probs when no cached matchup exists.
    Uses the reliever's own K%/BB%/HR%/BABIP to build a probability vector.
    """
    k = max(0.01, min(0.50, bp_profile.get("k_rate", 0.22)))
    bb = max(0.01, min(0.25, bp_profile.get("bb_rate", 0.08)))
    hbp = max(0.002, min(0.05, bp_profile.get("hbp_rate", 0.01)))
    hr = max(0.005, min(0.10, bp_profile.get("hr_rate", 0.03)))
    bip = max(0.10, 1.0 - k - bb - hbp - hr)
    total = k + bb + hbp + hr + bip

    cache[team, batter, pitcher, 0] = k / total
    cache[team, batter, pitcher, 1] = bb / total
    cache[team, batter, pitcher, 2] = hbp / total
    cache[team, batter, pitcher, 3] = hr / total
    cache[team, batter, pitcher, 4] = bip / total

    # Stage2: league-average BIP distribution (no batter-specific info available)
    cache[team, batter, pitcher, 5] = 0.155   # 1B
    cache[team, batter, pitcher, 6] = 0.045   # 2B
    cache[team, batter, pitcher, 7] = 0.005   # 3B
    cache[team, batter, pitcher, 8] = 0.400   # GO
    cache[team, batter, pitcher, 9] = 0.290   # FO
    cache[team, batter, pitcher, 10] = 0.070  # LO
    cache[team, batter, pitcher, 11] = 0.007  # FC
    cache[team, batter, pitcher, 12] = 0.012  # ROE


def _fill_probs(cache: np.ndarray, team: int, batter: int, pitcher: int, probs: dict):
    """Fill probability array from matchup dict."""
    s1 = probs.get("stage1", {})
    s2 = probs.get("stage2_bip", {})

    # Stage 1
    cache[team, batter, pitcher, 0] = s1.get("k_rate", 0.22)
    cache[team, batter, pitcher, 1] = s1.get("bb_rate", 0.08)
    cache[team, batter, pitcher, 2] = s1.get("hbp_rate", 0.01)
    cache[team, batter, pitcher, 3] = s1.get("hr_rate", 0.03)
    cache[team, batter, pitcher, 4] = s1.get("bip_rate", 0.66)

    # Stage 2
    cache[team, batter, pitcher, 5] = s2.get("1B", 0.15)
    cache[team, batter, pitcher, 6] = s2.get("2B", 0.04)
    cache[team, batter, pitcher, 7] = s2.get("3B", 0.005)
    cache[team, batter, pitcher, 8] = s2.get("GO", 0.42)
    cache[team, batter, pitcher, 9] = s2.get("FO", 0.28)
    cache[team, batter, pitcher, 10] = s2.get("LO", 0.07)
    cache[team, batter, pitcher, 11] = s2.get("FC", 0.007)
    cache[team, batter, pitcher, 12] = s2.get("ROE", 0.012)


def prepare_lineup_speeds(
    away_lineup: list[dict],
    home_lineup: list[dict],
    speed_cache: dict,
) -> np.ndarray:
    """[2, 9] int32: speed bucket per batter."""
    speeds = np.ones((2, 9), dtype=np.int32)  # default avg=1
    for team_idx, lineup in enumerate([away_lineup, home_lineup]):
        for b_idx, batter in enumerate(lineup[:9]):
            pid = batter.get("player_id")
            if pid and pid in speed_cache:
                speeds[team_idx, b_idx] = SPEED_BUCKET_TO_IDX.get(speed_cache[pid], 1)
    return speeds


def prepare_defense_buckets(away_def: str, home_def: str) -> np.ndarray:
    """[2] int32."""
    return np.array([
        DEFENSE_BUCKET_TO_IDX.get(away_def, 1),
        DEFENSE_BUCKET_TO_IDX.get(home_def, 1),
    ], dtype=np.int32)


def run_simulation_cpp_wrapper(
    matchup_cache_array: np.ndarray,
    transition_indices: np.ndarray,
    transition_results: np.ndarray,
    transition_probs: np.ndarray,
    transition_obs: np.ndarray,
    lineup_speeds: np.ndarray,
    defense_buckets: np.ndarray,
    n_sims: int = 200,
    max_innings: int = 15,
    base_seed: int = 42,
) -> dict:
    """C++ 시뮬레이션 실행 + 결과 집계."""
    from engine.cpp.sim_core import run_simulation_cpp

    alpha = PROBABILITY_DAMPENING_ALPHA

    scores = run_simulation_cpp(
        matchup_cache_array,
        None,  # transition_data (unused)
        transition_indices,
        transition_results,
        transition_probs,
        transition_obs,
        lineup_speeds,
        defense_buckets,
        n_sims=n_sims,
        max_innings=max_innings,
        dampening_alpha=alpha,
        min_transition_obs=TRANSITION_MIN_OBS,
        speed_fast_mod=1.10,
        speed_slow_mod=0.90,
        def_good_mod=0.95,
        def_poor_mod=1.05,
        fatigue_pull_threshold=float(FATIGUE_TIERS["pull"]["threshold"]),
        base_seed=base_seed,
    )

    away_scores = scores[:, 0].astype(float)
    home_scores = scores[:, 1].astype(float)
    total_runs = away_scores + home_scores

    # Win probability
    home_wins = (home_scores > away_scores).sum()
    away_wins = (away_scores > home_scores).sum()
    non_ties = home_wins + away_wins
    raw_home_prob = home_wins / non_ties if non_ties > 0 else 0.5
    home_prob = 0.50 + (raw_home_prob - 0.50) * alpha

    # Strong confidence cap: 65%+ dampened → extra squeeze
    # 200-game backtest: 65%+ zone has -18.7%p overconfidence
    STRONG_CAP = 0.62
    if home_prob > STRONG_CAP:
        home_prob = STRONG_CAP + (home_prob - STRONG_CAP) * 0.50
    elif home_prob < (1.0 - STRONG_CAP):
        home_prob = (1.0 - STRONG_CAP) - ((1.0 - STRONG_CAP) - home_prob) * 0.50

    # O/U
    ou_lines = {}
    for line in [x * 0.5 for x in range(11, 24)]:
        over = (total_runs > line).sum() / n_sims
        ou_lines[str(line)] = {"over": round(over, 4), "under": round(1 - over, 4)}

    return {
        "home_win_prob": round(home_prob, 4),
        "away_win_prob": round(1 - home_prob, 4),
        "raw_home_win_prob": round(raw_home_prob, 4),
        "avg_home_score": round(float(home_scores.mean()), 2),
        "avg_away_score": round(float(away_scores.mean()), 2),
        "avg_total": round(float(total_runs.mean()), 2),
        "over_under_lines": ou_lines,
        "n_simulations": n_sims,
        "engine_version": "v2.2-cpp",
    }
