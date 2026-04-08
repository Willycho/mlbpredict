"""N회 몬테카를로 시뮬레이션 실행 및 결과 집계."""

import time
import numpy as np
import pandas as pd

from engine.game import simulate_game
from engine.matchup import compute_matchup_probs
from engine.lineup import (
    get_team_lineup, get_bullpen_avg, get_bullpen_tiered, get_league_avg,
    get_batter_splits, get_pitcher_splits,
)
from config import DEFAULT_N_SIMULATIONS, TEAMS


def _precompute_matchup_cache(
    away_lineup, home_lineup,
    away_pitcher, home_pitcher,
    away_bullpen, home_bullpen,
    league_avg, home_team,
) -> dict:
    """모든 타자-투수 매치업 확률을 사전 계산."""
    cache = {}
    all_pitchers = [
        ("away_sp", away_pitcher),
        ("away_bp", away_bullpen),
        ("home_sp", home_pitcher),
        ("home_bp", home_bullpen),
    ]
    for p_key, pitcher in all_pitchers:
        p_id = pitcher.get("player_id", p_key)
        lineup = home_lineup if p_key.startswith("away") else away_lineup
        for batter in lineup:
            b_id = batter.get("player_id", id(batter))
            key = (b_id, p_id)
            if key not in cache:
                cache[key] = compute_matchup_probs(batter, pitcher, league_avg, home_team)
    return cache


def _build_summary(
    away_team, home_team,
    away_pitcher, home_pitcher,
    away_lineup, home_lineup,
    league_avg, home_win_prob, avg_away, avg_home,
) -> str:
    """시뮬레이션 결과 서머리 텍스트 생성."""
    away_name = TEAMS.get(away_team, away_team)
    home_name = TEAMS.get(home_team, home_team)
    away_sp = away_pitcher.get("name", "Unknown")
    home_sp = home_pitcher.get("name", "Unknown")

    # 투수 스탯 요약
    away_k = away_pitcher.get("k_rate", 0)
    away_bb = away_pitcher.get("bb_rate", 0)
    home_k = home_pitcher.get("k_rate", 0)
    home_bb = home_pitcher.get("bb_rate", 0)

    # 타선 파워 (라인업 평균 HR%)
    away_hr_avg = np.mean([b.get("hr_rate", 0.03) for b in away_lineup[:9]])
    home_hr_avg = np.mean([b.get("hr_rate", 0.03) for b in home_lineup[:9]])

    # 승패 요인 분석
    factors = []

    # 투수 우위
    if home_k > away_k * 1.15:
        factors.append(f"{home_sp}의 탈삼진 능력(K% {home_k:.0%})이 {away_sp}({away_k:.0%})보다 우수")
    elif away_k > home_k * 1.15:
        factors.append(f"{away_sp}의 탈삼진 능력(K% {away_k:.0%})이 {home_sp}({home_k:.0%})보다 우수")

    # 타선 파워
    if home_hr_avg > away_hr_avg * 1.2:
        factors.append(f"{home_team} 타선의 장타력(avg HR% {home_hr_avg:.1%})이 {away_team}({away_hr_avg:.1%})보다 강함")
    elif away_hr_avg > home_hr_avg * 1.2:
        factors.append(f"{away_team} 타선의 장타력(avg HR% {away_hr_avg:.1%})이 {home_team}({home_hr_avg:.1%})보다 강함")

    # 제구력
    if away_bb > home_bb * 1.3:
        factors.append(f"{away_sp}의 제구 불안(BB% {away_bb:.0%})이 출루 허용 가능성 높임")
    elif home_bb > away_bb * 1.3:
        factors.append(f"{home_sp}의 제구 불안(BB% {home_bb:.0%})이 출루 허용 가능성 높임")

    # 홈어드밴티지
    factors.append(f"{home_team} 홈 경기 이점")

    win_pct = max(home_win_prob, 1 - home_win_prob) * 100

    # 승리 판정: 승률과 평균 점수가 일치하는 방향 사용
    if avg_home > avg_away:
        winner = home_team
        loser = away_team
    elif avg_away > avg_home:
        winner = away_team
        loser = home_team
    else:
        winner = home_team if home_win_prob > 0.5 else away_team
        loser = away_team if winner == home_team else home_team

    # 신뢰도
    if win_pct >= 60:
        confidence = "강력 추천"
    elif win_pct >= 55:
        confidence = "추천"
    else:
        confidence = "접전 예상"

    summary = f"{winner} 승리 예측 ({win_pct:.1f}%, {confidence}). "
    summary += f"예상 스코어 {away_team} {avg_away:.1f} - {home_team} {avg_home:.1f}. "

    if factors:
        summary += "주요 요인: " + "; ".join(factors[:3]) + "."

    return summary


def run_simulation(
    away_team: str,
    home_team: str,
    away_pitcher: dict,
    home_pitcher: dict,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    season: int | None = None,
    mode: str = "full",
    away_lineup_override: list[dict] | None = None,
    home_lineup_override: list[dict] | None = None,
) -> dict:
    """몬테카를로 시뮬레이션 실행.

    Args:
        mode: "full" (9이닝, 불펜 2티어) 또는 "f5" (5이닝, 선발만)
        away_lineup_override: MLB API에서 가져온 실제 라인업 (None이면 PA 기반 기본)
        home_lineup_override: 위와 동일
    """
    start_time = time.time()

    # 에러율 통일 (F5/Full 동일)
    import engine.plate_appearance as PA
    PA.ERROR_RATE = 0.08

    away_lineup = away_lineup_override if away_lineup_override else get_team_lineup(away_team, season)
    home_lineup = home_lineup_override if home_lineup_override else get_team_lineup(home_team, season)
    away_bullpen = get_bullpen_avg(away_team, season)
    home_bullpen = get_bullpen_avg(home_team, season)
    league_avg = get_league_avg(season)

    # 불펜 티어 (풀게임 모드에서만 사용)
    away_bp_tiered = get_bullpen_tiered(away_team, season) if mode == "full" else None
    home_bp_tiered = get_bullpen_tiered(home_team, season) if mode == "full" else None

    # 사전 매치업 캐시 — 불펜 티어도 포함
    all_pitchers_for_cache = [
        ("away_sp", away_pitcher),
        ("home_sp", home_pitcher),
        ("away_bp", away_bullpen),
        ("home_bp", home_bullpen),
    ]
    if away_bp_tiered:
        all_pitchers_for_cache.append(("away_closer", away_bp_tiered["closer"]))
        all_pitchers_for_cache.append(("away_setup", away_bp_tiered["setup"]))
    if home_bp_tiered:
        all_pitchers_for_cache.append(("home_closer", home_bp_tiered["closer"]))
        all_pitchers_for_cache.append(("home_setup", home_bp_tiered["setup"]))

    matchup_cache = {}
    for p_key, pitcher in all_pitchers_for_cache:
        p_id = pitcher.get("player_id", p_key)
        # away 투수 → 홈 타선이 상대 (타자 is_home=True)
        # home 투수 → 원정 타선이 상대 (타자 is_home=False)
        if p_key.startswith("away"):
            lineup = home_lineup
            is_batter_home = True
        else:
            lineup = away_lineup
            is_batter_home = False
        for batter in lineup:
            b_id = batter.get("player_id", id(batter))
            key = (b_id, p_id)
            if key not in matchup_cache:
                matchup_cache[key] = compute_matchup_probs(
                    batter, pitcher, league_avg, home_team,
                    is_batter_home=is_batter_home,
                    runner_state="empty",  # 기본, RISP는 실시간 보정
                )

    # RISP 스플릿 사전 캐시: {player_id: risp_dict or None}
    splits_cache = {}
    for batter in away_lineup + home_lineup:
        b_id = batter.get("player_id")
        if b_id and b_id not in splits_cache:
            risp = get_batter_splits(b_id).get("runners", {}).get("risp")
            splits_cache[b_id] = risp if risp and risp.get("n_pa", 0) >= 20 else None

    for p_key, pitcher in all_pitchers_for_cache:
        p_id = pitcher.get("player_id", p_key)
        if p_id and p_id not in splits_cache:
            risp = get_pitcher_splits(p_id).get("runners", {}).get("risp")
            splits_cache[p_id] = risp if risp and risp.get("n_pa", 0) >= 20 else None

    rng = np.random.default_rng()

    results = []
    all_inning_away = []
    all_inning_home = []

    # 타자별 누적 스탯 {batter_id: {pa: [], ab: [], h: [], ...}}
    stat_keys = ["pa", "ab", "r", "h", "2b", "3b", "hr", "rbi", "bb", "k", "hbp"]
    away_cumulative = {}  # batter_id -> {key: [values per sim]}
    home_cumulative = {}

    for _ in range(n_simulations):
        game_result = simulate_game(
            away_lineup, home_lineup,
            away_pitcher, home_pitcher,
            away_bullpen, home_bullpen,
            league_avg, home_team, rng,
            matchup_cache=matchup_cache,
            mode=mode,
            away_bullpen_tiered=away_bp_tiered,
            home_bullpen_tiered=home_bp_tiered,
            splits_cache=splits_cache,
        )
        a_score = game_result["away_score"]
        h_score = game_result["home_score"]
        results.append({
            "away_score": a_score,
            "home_score": h_score,
            "total_runs": a_score + h_score,
            "home_win": h_score > a_score,
            "away_win": a_score > h_score,
            "tie": a_score == h_score,
        })

        # 이닝별
        n_innings = 5 if mode == "f5" else 9
        inning_scores = game_result["inning_scores"]
        away_by_inning = [0] * n_innings
        home_by_inning = [0] * n_innings
        for i, (a, h) in enumerate(inning_scores):
            if i < n_innings:
                away_by_inning[i] = a
                home_by_inning[i] = h
        all_inning_away.append(away_by_inning)
        all_inning_home.append(home_by_inning)

        # 타자별 스탯 누적
        for batter_id, stats in game_result.get("away_batter_stats", {}).items():
            if batter_id not in away_cumulative:
                away_cumulative[batter_id] = {k: [] for k in stat_keys}
            for k in stat_keys:
                away_cumulative[batter_id][k].append(stats.get(k, 0))

        for batter_id, stats in game_result.get("home_batter_stats", {}).items():
            if batter_id not in home_cumulative:
                home_cumulative[batter_id] = {k: [] for k in stat_keys}
            for k in stat_keys:
                home_cumulative[batter_id][k].append(stats.get(k, 0))

    df = pd.DataFrame(results)
    elapsed = time.time() - start_time

    # 동점 제외하고 승률 계산 (F5 모드에서 동점 가능)
    decided = df[~df["tie"]]
    if len(decided) > 0:
        home_win_prob = float(decided["home_win"].mean())
    else:
        home_win_prob = 0.5
    tie_rate = float(df["tie"].mean())

    avg_total = float(df["total_runs"].mean())
    avg_away = float(df["away_score"].mean())
    avg_home = float(df["home_score"].mean())

    # Over/Under
    ou_lines = [5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5]
    over_under = {}
    for line in ou_lines:
        over_prob = float((df["total_runs"] > line).mean())
        over_under[str(line)] = {
            "over": round(over_prob, 4),
            "under": round(1 - over_prob, 4),
        }

    # 핸디캡 — 페이버릿(승률 높은 팀) 기준
    margin = df["home_score"] - df["away_score"]  # 양수=홈 우세
    if home_win_prob >= 0.5:
        fav_team = home_team
        dog_team = away_team
        fav_margin = margin  # 페이버릿-언더독 점수차
    else:
        fav_team = away_team
        dog_team = home_team
        fav_margin = -margin

    run_lines = {
        "favorite": fav_team,
        "underdog": dog_team,
        "fav_-1.5": round(float((fav_margin > 1.5).mean()), 4),
        "fav_-2.5": round(float((fav_margin > 2.5).mean()), 4),
        "dog_+1.5": round(float((fav_margin < -1.5).mean()), 4),
        "dog_+2.5": round(float((fav_margin < -2.5).mean()), 4),
    }

    # 점수 분포
    total_dist = df["total_runs"].value_counts().sort_index()
    score_distribution = {str(int(k)): round(float(v / n_simulations), 4) for k, v in total_dist.items()}

    # 최빈 스코어
    score_pairs = df.apply(lambda r: f"{int(r['away_score'])}-{int(r['home_score'])}", axis=1)
    top_scores = score_pairs.value_counts().head(10)
    most_likely_scores = [
        {"score": score, "prob": round(float(count / n_simulations), 4)}
        for score, count in top_scores.items()
    ]

    # 이닝별 평균 득점
    n_innings = 5 if mode == "f5" else 9
    away_inning_arr = np.array(all_inning_away)
    home_inning_arr = np.array(all_inning_home)
    inning_avg_away = [round(float(away_inning_arr[:, i].mean()), 3) for i in range(n_innings)]
    inning_avg_home = [round(float(home_inning_arr[:, i].mean()), 3) for i in range(n_innings)]

    inning_scoring_away = [round(float((away_inning_arr[:, i] > 0).mean()), 3) for i in range(n_innings)]
    inning_scoring_home = [round(float((home_inning_arr[:, i] > 0).mean()), 3) for i in range(n_innings)]

    # 타자별 기대 성적 (1경기 평균)
    def _aggregate_batter_stats(cumulative, lineup):
        """타자별 누적 → 1경기 평균."""
        batter_results = []
        # 라인업 순서대로
        for idx, batter in enumerate(lineup[:9]):
            b_id = batter.get("player_id")
            if b_id and b_id in cumulative:
                cum = cumulative[b_id]
                n = len(cum["pa"])
                entry = {
                    "order": idx + 1,
                    "name": batter.get("name", f"#{idx+1}"),
                    "player_id": b_id,
                }
                for k in stat_keys:
                    entry[k] = round(np.mean(cum[k]), 2) if n > 0 else 0
                # AVG 계산
                total_ab = sum(cum["ab"])
                total_h = sum(cum["h"])
                entry["avg"] = round(total_h / total_ab, 3) if total_ab > 0 else 0
                batter_results.append(entry)
        return batter_results

    away_batter_results = _aggregate_batter_stats(away_cumulative, away_lineup)
    home_batter_results = _aggregate_batter_stats(home_cumulative, home_lineup)

    # 서머리
    summary = _build_summary(
        away_team, home_team,
        away_pitcher, home_pitcher,
        away_lineup, home_lineup,
        league_avg, home_win_prob, avg_away, avg_home,
    )

    return {
        "away_team": away_team,
        "home_team": home_team,
        "away_pitcher": away_pitcher.get("name", "Unknown"),
        "home_pitcher": home_pitcher.get("name", "Unknown"),
        "away_pitcher_k": round(float(away_pitcher.get("k_rate", 0)), 4),
        "home_pitcher_k": round(float(home_pitcher.get("k_rate", 0)), 4),
        "n_simulations": n_simulations,
        "home_win_prob": round(home_win_prob, 4),
        "away_win_prob": round(1 - home_win_prob, 4),
        "tie_rate": round(tie_rate, 4),
        "avg_total_runs": round(avg_total, 2),
        "avg_away_score": round(avg_away, 2),
        "avg_home_score": round(avg_home, 2),
        "over_under": over_under,
        "run_lines": run_lines,
        "score_distribution": score_distribution,
        "most_likely_scores": most_likely_scores,
        "inning_avg": {
            "away": inning_avg_away,
            "home": inning_avg_home,
        },
        "inning_scoring_prob": {
            "away": inning_scoring_away,
            "home": inning_scoring_home,
        },
        "away_batters": away_batter_results,
        "home_batters": home_batter_results,
        "mode": mode,
        "n_innings": n_innings,
        "summary": summary,
        "simulation_time_seconds": round(elapsed, 2),
    }
