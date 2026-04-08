"""v2 Monte Carlo Simulation Orchestrator.

v1 simulation.py 대체. 핵심 변경:
1. v2 엔진 컴포넌트 사용 (bayesian_matchup, multinomial_pa, markov_transition)
2. calibration 레이어 옵션
3. use_team_strength 플래그 (기본 OFF)
4. count-aware pitch mix
"""

import os
import time
import numpy as np
import pandas as pd

from config import (
    DEFAULT_N_SIMULATIONS, V2_DATA_DIR, DATA_DIR,
    PROBABILITY_DAMPENING_ALPHA,
)
from engine.bayesian_matchup import compute_matchup_v2
from engine.game_engine import simulate_game
from engine.markov_transition import load_transition_matrix, clear_cache
from engine.lineup import (
    get_team_lineup, get_league_avg,
    get_batter_pitch_arsenal, get_pitcher_pitch_arsenal,
    get_h2h_stats, get_batter_splits, get_pitcher_splits,
)


def _load_v2_data() -> dict:
    """v2 전용 데이터 로드."""
    data = {}

    # Count pitch mix
    path = os.path.join(V2_DATA_DIR, "count_pitch_mix.parquet")
    if os.path.exists(path):
        data["count_pitch_mix"] = pd.read_parquet(path)
    else:
        data["count_pitch_mix"] = None

    # Team defense
    path = os.path.join(V2_DATA_DIR, "team_defense.parquet")
    if os.path.exists(path):
        data["team_defense"] = pd.read_parquet(path)
    else:
        data["team_defense"] = None

    # Speed proxy
    path = os.path.join(V2_DATA_DIR, "speed_proxy.parquet")
    if os.path.exists(path):
        data["speed_proxy"] = pd.read_parquet(path)
    else:
        data["speed_proxy"] = None

    # Bullpen profiles
    path = os.path.join(V2_DATA_DIR, "bullpen_profiles.parquet")
    if os.path.exists(path):
        data["bullpen_profiles"] = pd.read_parquet(path)
    else:
        data["bullpen_profiles"] = None

    # Transition matrix
    try:
        data["transition_matrix"] = load_transition_matrix()
    except FileNotFoundError:
        data["transition_matrix"] = None

    return data


def _get_defense_bucket(team_defense: pd.DataFrame | None, team: str, season: int = 2025) -> str:
    """팀의 수비 버킷 조회."""
    if team_defense is None:
        return "avg"
    match = team_defense[
        (team_defense["team"] == team) & (team_defense["season"] == season)
    ]
    if len(match) == 0:
        # 가장 최근 시즌
        match = team_defense[team_defense["team"] == team]
    if len(match) == 0:
        return "avg"
    return match.iloc[-1]["defense_bucket"]


def _build_speed_cache(speed_proxy: pd.DataFrame | None, lineup: list[dict]) -> dict:
    """라인업 선수들의 speed bucket 캐시."""
    cache = {}
    if speed_proxy is None:
        return cache
    for batter in lineup:
        pid = batter.get("player_id")
        if pid is None:
            continue
        match = speed_proxy[speed_proxy["player_id"] == pid]
        if len(match) > 0:
            cache[pid] = match.iloc[0]["speed_bucket"]
        else:
            cache[pid] = "avg"
    return cache


def _build_bullpen_dict(
    bullpen_profiles: pd.DataFrame | None,
    team: str,
    season: int = 2025,
    fallback_bullpen=None,
) -> dict:
    """팀의 4역할 불펜 dict 구성."""
    if bullpen_profiles is None:
        if fallback_bullpen:
            return {"setup_early": fallback_bullpen}
        return {}

    match = bullpen_profiles[
        (bullpen_profiles["team"] == team) & (bullpen_profiles["season"] == season)
    ]
    if len(match) == 0:
        match = bullpen_profiles[bullpen_profiles["team"] == team]
    if len(match) == 0 and fallback_bullpen:
        return {"setup_early": fallback_bullpen}

    result = {}
    for _, row in match.iterrows():
        role = row["role"]
        result[role] = {
            "k_rate": row["k_rate"],
            "bb_rate": row["bb_rate"],
            "hr_rate": row.get("hr_rate", 0.03),
            "hbp_rate": row.get("hbp_rate", 0.01),
            "babip": row.get("babip", 0.300),
            "player_id": f"bullpen_{team}_{role}",
        }
    return result


def _precompute_matchup_cache(
    lineup: list[dict],
    pitcher: dict,
    league_avg: dict,
    home_team: str,
    is_batter_home: bool,
    count_pitch_mix=None,
    defense_bucket: str = "avg",
) -> dict:
    """라인업 전체의 매치업 사전 계산."""
    cache = {}
    pitcher_id = pitcher.get("player_id", "p")

    for batter in lineup:
        batter_id = batter.get("player_id")
        if batter_id is None:
            continue

        probs = compute_matchup_v2(
            batter=batter,
            pitcher=pitcher,
            league_avg=league_avg,
            home_team=home_team,
            is_batter_home=is_batter_home,
            batter_arsenal=batter.get("arsenal"),
            pitcher_arsenal=pitcher.get("arsenal"),
            h2h_stats=batter.get("h2h"),
            count_pitch_mix=count_pitch_mix,
            defense_bucket=defense_bucket,
        )
        cache[(batter_id, pitcher_id)] = probs

    return cache


def run_simulation(
    away_team: str,
    home_team: str,
    away_pitcher: dict,
    home_pitcher: dict,
    n_simulations: int = None,
    season: int = 2025,
    mode: str = "full",
    lineup_override: dict = None,
    use_team_strength: bool = False,
    count_aware: bool = True,
) -> dict:
    """v2 Monte Carlo 시뮬레이션 실행.

    Returns:
        dict with: home_win_prob, away_win_prob, avg_total,
                   avg_away_score, avg_home_score, over_under_lines,
                   score_distribution, simulation_time, ...
    """
    if n_simulations is None:
        n_simulations = DEFAULT_N_SIMULATIONS

    start_time = time.time()

    # 데이터 로드
    v2_data = _load_v2_data()
    league_avg = get_league_avg(season)

    # 라인업
    if lineup_override and "away" in lineup_override:
        away_lineup = lineup_override["away"]
    else:
        away_lineup = get_team_lineup(away_team, season)

    if lineup_override and "home" in lineup_override:
        home_lineup = lineup_override["home"]
    else:
        home_lineup = get_team_lineup(home_team, season)

    # 타자 arsenal, h2h 부착
    for lineup, team in [(away_lineup, away_team), (home_lineup, home_team)]:
        opp_pitcher = home_pitcher if team == away_team else away_pitcher
        for batter in lineup:
            pid = batter.get("player_id")
            if pid:
                batter["arsenal"] = get_batter_pitch_arsenal(pid)
                opp_pid = opp_pitcher.get("player_id")
                if opp_pid:
                    batter["h2h"] = get_h2h_stats(pid, opp_pid)

    # 투수 arsenal 부착
    for pitcher in [away_pitcher, home_pitcher]:
        pid = pitcher.get("player_id")
        if pid:
            pitcher["arsenal"] = get_pitcher_pitch_arsenal(pid)

    # Defense buckets
    away_def = _get_defense_bucket(v2_data["team_defense"], away_team, season)
    home_def = _get_defense_bucket(v2_data["team_defense"], home_team, season)

    # Speed caches
    away_speed = _build_speed_cache(v2_data["speed_proxy"], away_lineup)
    home_speed = _build_speed_cache(v2_data["speed_proxy"], home_lineup)
    speed_cache = {**away_speed, **home_speed}

    # Bullpen
    away_bullpen = _build_bullpen_dict(v2_data["bullpen_profiles"], away_team, season)
    home_bullpen = _build_bullpen_dict(v2_data["bullpen_profiles"], home_team, season)

    # Count pitch mix
    cpm = v2_data["count_pitch_mix"] if count_aware else None

    # Transition matrix
    tm = v2_data["transition_matrix"]

    # Matchup cache precompute
    matchup_cache = {}
    for lineup, pitcher, is_home in [
        (away_lineup, home_pitcher, False),
        (home_lineup, away_pitcher, True),
    ]:
        cache = _precompute_matchup_cache(
            lineup, pitcher, league_avg, home_team, is_home, cpm,
            home_def if not is_home else away_def,
        )
        matchup_cache.update(cache)

    # 불펜 매치업도 캐시
    for lineup, bullpen_dict, is_home in [
        (away_lineup, home_bullpen, False),
        (home_lineup, away_bullpen, True),
    ]:
        for role, bp in bullpen_dict.items():
            cache = _precompute_matchup_cache(
                lineup, bp, league_avg, home_team, is_home, cpm,
                home_def if not is_home else away_def,
            )
            matchup_cache.update(cache)

    # Monte Carlo 루프
    away_scores = []
    home_scores = []
    total_runs_list = []

    for i in range(n_simulations):
        sim_rng = np.random.default_rng(seed=i)

        result = simulate_game(
            away_lineup=away_lineup,
            home_lineup=home_lineup,
            away_starter=away_pitcher,
            home_starter=home_pitcher,
            away_bullpen=away_bullpen,
            home_bullpen=home_bullpen,
            league_avg=league_avg,
            home_team=home_team,
            rng=sim_rng,
            mode=mode,
            matchup_cache=matchup_cache,
            count_pitch_mix=cpm,
            away_defense_bucket=away_def,
            home_defense_bucket=home_def,
            speed_cache=speed_cache,
            tm=tm,
        )

        away_scores.append(result["away_score"])
        home_scores.append(result["home_score"])
        total_runs_list.append(result["away_score"] + result["home_score"])

    away_scores = np.array(away_scores)
    home_scores = np.array(home_scores)
    total_runs = np.array(total_runs_list)

    # 집계
    # 타이 제외 승률
    home_wins = (home_scores > away_scores).sum()
    away_wins = (away_scores > home_scores).sum()
    non_ties = home_wins + away_wins
    raw_home_win_prob = home_wins / non_ties if non_ties > 0 else 0.5
    raw_away_win_prob = away_wins / non_ties if non_ties > 0 else 0.5

    # Dampening: 0.50 방향으로 당기기
    alpha = PROBABILITY_DAMPENING_ALPHA
    home_win_prob = 0.50 + (raw_home_win_prob - 0.50) * alpha
    away_win_prob = 1.0 - home_win_prob

    # O/U 라인별 확률
    ou_lines = {}
    for line in [x * 0.5 for x in range(11, 24)]:  # 5.5 ~ 11.5
        over_pct = (total_runs > line).sum() / n_simulations
        ou_lines[str(line)] = {
            "over": round(over_pct, 4),
            "under": round(1 - over_pct, 4),
        }

    # 점수 분포
    score_dist = {}
    for total in range(0, 30):
        count = (total_runs == total).sum()
        if count > 0:
            score_dist[total] = count / n_simulations

    elapsed = time.time() - start_time

    return {
        "home_win_prob": round(home_win_prob, 4),
        "away_win_prob": round(away_win_prob, 4),
        "raw_home_win_prob": round(raw_home_win_prob, 4),
        "raw_away_win_prob": round(raw_away_win_prob, 4),
        "dampening_alpha": alpha,
        "avg_home_score": round(float(home_scores.mean()), 2),
        "avg_away_score": round(float(away_scores.mean()), 2),
        "avg_total": round(float(total_runs.mean()), 2),
        "median_total": round(float(np.median(total_runs)), 1),
        "std_total": round(float(total_runs.std()), 2),
        "over_under_lines": ou_lines,
        "score_distribution": score_dist,
        "n_simulations": n_simulations,
        "simulation_time": round(elapsed, 2),
        "mode": mode,
        "engine_version": "v2.1",
        "away_team": away_team,
        "home_team": home_team,
        "away_defense_bucket": away_def,
        "home_defense_bucket": home_def,
    }
