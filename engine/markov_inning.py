"""v2 Markov 기반 반이닝 시뮬레이터.

v1 inning.py 대체. 핵심 변경:
1. multinomial_pa + markov_transition 사용
2. 절대 상태(absolute state) 기반 전이
3. RISP 실시간 스플릿 반영
"""

import numpy as np

from engine.bayesian_matchup import compute_matchup_v2
from engine.multinomial_pa import resolve_pa
from engine.markov_transition import resolve_transition


def simulate_half_inning(
    lineup: list[dict],
    lineup_pos: int,
    pitcher: dict,
    league_avg: dict,
    home_team: str,
    is_batter_home: bool,
    ghost_runner: bool = False,
    matchup_cache: dict = None,
    batter_stats: dict = None,
    splits_cache: dict = None,
    count_pitch_mix=None,
    defense_bucket: str = "avg",
    speed_cache: dict = None,
    tm=None,
    rng: np.random.Generator = None,
) -> tuple[int, int]:
    """반���닝 시뮬레이션.

    Args:
        lineup: 9명 타자 dict 리스트
        lineup_pos: 현재 타순 인덱스
        pitcher: 투수 dict
        league_avg: 리��� 평균 dict
        home_team: 홈팀 코드
        is_batter_home: 타자가 홈팀인지
        ghost_runner: Manfred runner (True → 2루 주자)
        matchup_cache: 사전 계산된 매치업 캐시
        batter_stats: 타자별 누적 스탯 (시뮬 중 업데이트)
        splits_cache: 선수별 스플릿 캐시
        count_pitch_mix: count별 pitch mix DataFrame
        defense_bucket: 수비 버킷
        speed_cache: {player_id: speed_bucket}
        tm: transition matrix DataFrame
        rng: numpy random generator

    Returns:
        (이닝 득점, 업데이트된 lineup_pos)
    """
    if rng is None:
        rng = np.random.default_rng()

    runs = 0
    outs = 0

    # 초기 base-out state
    if ghost_runner:
        base_state = "010"  # 2루 주자
    else:
        base_state = "000"

    while outs < 3:
        batter_idx = lineup_pos % len(lineup)
        batter = lineup[batter_idx]
        batter_id = batter.get("player_id", batter_idx)

        # runner state 결정 (RISP 스플릿용)
        runner_state = _get_runner_state(base_state)

        # speed bucket ���정
        speed_bucket = "avg"
        if speed_cache:
            # 가장 앞에 있는 주자의 speed bucket 사용
            # (간소화: 전체 주자 평균 대신 리드 러너)
            speed_bucket = speed_cache.get(batter_id, "avg")

        # 매치업 확률 계산 (캐시 우선)
        cache_key = None
        if matchup_cache is not None:
            pitcher_id = pitcher.get("player_id", "p")
            cache_key = (batter_id, pitcher_id)

        if cache_key and cache_key in matchup_cache:
            probs = matchup_cache[cache_key]
        else:
            # 스플릿 데이터
            batter_splits = None
            pitcher_splits = None
            if splits_cache:
                batter_splits = splits_cache.get(batter_id)
                pitcher_splits = splits_cache.get(pitcher.get("player_id"))

            probs = compute_matchup_v2(
                batter=batter,
                pitcher=pitcher,
                league_avg=league_avg,
                home_team=home_team,
                is_batter_home=is_batter_home,
                count_bucket="first_pitch",  # 타석 시작은 항상 0-0
                batter_arsenal=batter.get("arsenal"),
                pitcher_arsenal=pitcher.get("arsenal"),
                h2h_stats=batter.get("h2h"),
                batter_splits=batter_splits,
                count_pitch_mix=count_pitch_mix,
                defense_bucket=defense_bucket,
                runner_state=runner_state,
            )

            if matchup_cache is not None and cache_key:
                matchup_cache[cache_key] = probs

        # 타석 결과 결정
        outcome = resolve_pa(probs, rng)

        # base-out 전이
        new_base, new_outs, scored = resolve_transition(
            base_before=base_state,
            outs_before=outs,
            event_type=outcome,
            speed_bucket=speed_bucket,
            defense_bucket=defense_bucket,
            rng=rng,
            tm=tm,
        )

        runs += scored
        outs = min(new_outs, 3)
        base_state = new_base

        # 타자 스탯 추적
        if batter_stats is not None:
            _update_batter_stats(batter_stats, batter_id, outcome, scored)

        lineup_pos += 1

    return runs, lineup_pos


def _get_runner_state(base_state: str) -> str:
    """base_state에서 runner_state 추출."""
    if base_state == "000":
        return "empty"
    # 2루 또는 3루에 주자 → risp
    if base_state[1] == "1" or base_state[2] == "1":
        return "risp"
    return "runners_on"


def _update_batter_stats(
    stats: dict,
    batter_id,
    outcome: str,
    runs_scored: int,
):
    """타자 시뮬레이션 스탯 업데이트."""
    if batter_id not in stats:
        stats[batter_id] = {
            "pa": 0, "ab": 0, "r": 0, "h": 0,
            "2b": 0, "3b": 0, "hr": 0, "rbi": 0,
            "bb": 0, "k": 0, "hbp": 0,
        }

    s = stats[batter_id]
    s["pa"] += 1

    hit_outcomes = {"1B", "2B", "3B", "HR"}
    ab_excluded = {"BB", "HBP", "SAC"}

    if outcome not in ab_excluded:
        s["ab"] += 1

    if outcome in hit_outcomes:
        s["h"] += 1

    if outcome == "2B":
        s["2b"] += 1
    elif outcome == "3B":
        s["3b"] += 1
    elif outcome == "HR":
        s["hr"] += 1
        s["r"] += 1
        s["rbi"] += runs_scored
    elif outcome == "BB":
        s["bb"] += 1
    elif outcome == "HBP":
        s["hbp"] += 1
    elif outcome == "K":
        s["k"] += 1

    if outcome != "HR" and runs_scored > 0:
        s["rbi"] += runs_scored
