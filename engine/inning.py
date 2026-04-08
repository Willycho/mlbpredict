"""반이닝 시뮬레이션."""

import numpy as np

from engine.matchup import compute_matchup_probs
from engine.plate_appearance import resolve_pa
from engine.baserunning import resolve_baserunning


def simulate_half_inning(
    lineup: list[dict],
    lineup_pos: int,
    pitcher: dict,
    league_avg: dict,
    home_team: str,
    rng: np.random.Generator,
    ghost_runner: bool = False,
    matchup_cache: dict | None = None,
    batter_stats: dict | None = None,
    splits_cache: dict | None = None,
) -> tuple[int, int]:
    """반이닝을 시뮬레이션한다.

    Args:
        batter_stats: {batter_id: {"ab":0, "r":0, "h":0, ...}} 누적 기록용 딕셔너리.
                      None이면 기록 안 함.

    Returns:
        (이닝 득점, 업데이트된 lineup_pos)
    """
    outs = 0
    runs = 0
    bases = [None, None, None]  # 1B, 2B, 3B
    # 각 주자가 어떤 타자인지 추적 (RBI 계산용)
    base_runners = [None, None, None]  # batter_id or None

    if ghost_runner:
        bases[1] = 27.0
        base_runners[1] = "ghost"

    pitcher_id = pitcher.get("player_id", id(pitcher))

    # 투수 RISP 스플릿 (사전 캐시에서)
    p_risp = splits_cache.get(pitcher_id) if splits_cache else None

    while outs < 3:
        batter = lineup[lineup_pos % 9]
        batter_id = batter.get("player_id", id(batter))

        cache_key = (batter_id, pitcher_id)
        if matchup_cache and cache_key in matchup_cache:
            probs = matchup_cache[cache_key]
        else:
            probs = compute_matchup_probs(batter, pitcher, league_avg, home_team)

        # RISP 실시간 보정: 2루 or 3루에 주자 있으면
        if bases[1] is not None or bases[2] is not None:
            probs = dict(probs)  # 복사

            # 타자 RISP 보정 (사전 캐시에서)
            b_risp = splits_cache.get(batter_id) if splits_cache else None
            if b_risp:
                for k in ["k_rate", "bb_rate", "hr_rate", "babip"]:
                    if k in b_risp and probs.get(k, 0) > 0:
                        ratio = max(0.75, min(1.3, b_risp[k] / probs[k]))
                        probs[k] = probs[k] * ratio

            # 투수 RISP 보정
            if p_risp:
                for k in ["k_rate", "bb_rate", "hr_rate", "babip"]:
                    if k in p_risp and probs.get(k, 0) > 0:
                        ratio = max(0.75, min(1.3, p_risp[k] / probs[k]))
                        probs[k] = probs[k] * ratio

            probs["bip_rate"] = 1.0 - probs["k_rate"] - probs["bb_rate"] - probs.get("hbp_rate", 0.01) - probs["hr_rate"]

        outcome = resolve_pa(probs, rng)

        # 주루 처리
        gdp_rate = batter.get("gdp_rate", 0.15)
        scored, new_bases, added_outs = resolve_baserunning(
            outcome, bases, outs, gdp_rate, rng
        )

        # 타자별 스탯 기록
        if batter_stats is not None:
            if batter_id not in batter_stats:
                batter_stats[batter_id] = {
                    "pa": 0, "ab": 0, "r": 0, "h": 0,
                    "2b": 0, "3b": 0, "hr": 0,
                    "rbi": 0, "bb": 0, "k": 0, "hbp": 0,
                }
            s = batter_stats[batter_id]
            s["pa"] += 1

            if outcome == "K":
                s["ab"] += 1
                s["k"] += 1
            elif outcome == "BB":
                s["bb"] += 1
            elif outcome == "HBP":
                s["hbp"] += 1
            elif outcome == "HR":
                s["ab"] += 1
                s["h"] += 1
                s["hr"] += 1
                s["rbi"] += scored  # 홈런 타점 = 본인 포함 주자 전원
            elif outcome == "1B":
                s["ab"] += 1
                s["h"] += 1
                s["rbi"] += scored
            elif outcome == "2B":
                s["ab"] += 1
                s["h"] += 1
                s["2b"] += 1
                s["rbi"] += scored
            elif outcome == "3B":
                s["ab"] += 1
                s["h"] += 1
                s["3b"] += 1
                s["rbi"] += scored
            else:
                # GO, FO, LO, PO
                s["ab"] += 1
                s["rbi"] += scored  # 희생플라이 타점 등

        # 득점한 주자의 R 기록
        if batter_stats is not None and scored > 0:
            # 주자들의 득점 — 정확한 주자 ID 추적은 복잡하므로
            # 타자 본인이 홈인한 경우 (HR)만 기록
            if outcome == "HR":
                s["r"] += 1

        runs += scored
        bases = new_bases
        outs += added_outs
        lineup_pos += 1

    return runs, lineup_pos
