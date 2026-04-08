"""v2 Bayesian Matchup Engine — shrinkage + count-aware pitch mix + 수비 보정.

v1 matchup.py 대체. 핵심 변경:
1. 모든 소표본 피처에 shrinkage 적용
2. count bucket별 pitch mix 반영
3. 팀 수비 OAA proxy 반영
4. 팀 승률 보정 기본 OFF
5. 2단계 확률 벡터 출력 (Stage1 + Stage2 BIP)
"""

import os
import json
import pandas as pd
import numpy as np

from config import (
    PLATOON_MULTIPLIERS, PARK_FACTORS, PARK_HR_FACTORS,
    SHRINKAGE, V2_DATA_DIR, DATA_DIR,
    BIP_FC_RATE, BIP_ROE_RATE,
    BABIP_BY_BATTED_BALL, HIT_TYPE_DISTRIBUTION,
    DEFENSE_BUCKETS,
)

# ============================================================
# Shrinkage
# ============================================================

def shrink(observed: float, prior: float, n: int, k: int) -> float:
    """Empirical Bayes shrinkage: n/(n+k) 가중 평균."""
    if n <= 0:
        return prior
    w = n / (n + k)
    return w * observed + (1 - w) * prior


def shrink_rate(rate: float, league_rate: float, n: int, k: int) -> float:
    """rate 값에 shrinkage 적용. clamp 포함."""
    result = shrink(rate, league_rate, n, k)
    return max(0.001, min(0.999, result))


# ============================================================
# Log5
# ============================================================

def log5(p_batter: float, p_pitcher: float, p_league: float) -> float:
    """Log5 공식."""
    if p_league <= 0 or p_league >= 1:
        return p_batter
    num = p_batter * p_pitcher / p_league
    den = num + (1 - p_batter) * (1 - p_pitcher) / (1 - p_league)
    if den <= 0:
        return p_batter
    return max(0.001, min(0.999, num / den))


# ============================================================
# Pitch-type weighted matchup
# ============================================================

def compute_pitch_weighted_stats(
    batter_arsenal: list[dict],
    pitcher_arsenal: list[dict],
    league_avg: dict,
    count_pitch_mix: pd.DataFrame | None = None,
    count_bucket: str = "first_pitch",
    batter_hand: str = "R",
) -> dict:
    """구종별 가중 Log5 매치업 계산.

    Args:
        batter_arsenal: 타자의 구종별 성적 리스트
        pitcher_arsenal: 투수의 구종별 성적 + usage 리스트
        league_avg: 리그 평균 dict
        count_pitch_mix: count별 pitch mix DataFrame (optional)
        count_bucket: 현재 count bucket
        batter_hand: 타자 손잡이

    Returns:
        dict: k_rate, bb_rate, hbp_rate, hr_rate, babip, iso, ld_rate, gb_rate, fb_rate
    """
    stat_keys = ["k_rate", "bb_rate", "hbp_rate", "hr_rate", "babip"]

    # 투수 pitch mix 결정
    pitcher_pitches = {}
    for p in pitcher_arsenal:
        pt = p.get("pitch_type", "")
        usage = p.get("usage", 0)
        if usage >= 0.02 and pt:
            pitcher_pitches[pt] = p

    # count-aware pitch mix 적용
    if count_pitch_mix is not None and len(count_pitch_mix) > 0:
        pitcher_id = pitcher_arsenal[0].get("pitcher_id") if pitcher_arsenal else None
        if pitcher_id is not None:
            cm = count_pitch_mix[
                (count_pitch_mix["pitcher_id"] == pitcher_id)
                & (count_pitch_mix["stand"] == batter_hand)
                & (count_pitch_mix["count_bucket"] == count_bucket)
            ]
            if len(cm) > 0:
                # count mix usage로 덮어쓰기
                for _, row in cm.iterrows():
                    pt = row["pitch_type"]
                    if pt in pitcher_pitches:
                        pitcher_pitches[pt] = dict(pitcher_pitches[pt])
                        pitcher_pitches[pt]["usage"] = row["usage_pct"]

    if not pitcher_pitches:
        # pitch data 없으면 리그 평균 반환
        return {k: league_avg.get(k, 0.0) for k in stat_keys}

    # 타자 구종별 성적 인덱싱
    batter_by_pitch = {}
    for b in batter_arsenal:
        pt = b.get("pitch_type", "")
        if pt:
            batter_by_pitch[pt] = b

    # usage 재정규화
    total_usage = sum(p.get("usage", 0) for p in pitcher_pitches.values())
    if total_usage <= 0:
        total_usage = 1.0

    weighted = {k: 0.0 for k in stat_keys}
    total_weight = 0.0

    for pt, p_stats in pitcher_pitches.items():
        usage = p_stats.get("usage", 0) / total_usage
        if usage <= 0:
            continue

        b_stats = batter_by_pitch.get(pt)

        for key in stat_keys:
            p_val = p_stats.get(key, league_avg.get(key, 0.0))
            l_val = league_avg.get(key, 0.0)

            if b_stats and b_stats.get("n_pa", 0) >= 5:
                b_val = b_stats.get(key, l_val)
                # 타자 구종별 스탯도 shrinkage
                b_val = shrink(b_val, l_val, b_stats.get("n_pa", 0), SHRINKAGE["batter_rate"])
                val = log5(b_val, p_val, l_val)
            else:
                val = p_val  # 타자 데이터 없으면 투수 스탯 사용

            weighted[key] += val * usage

        total_weight += usage

    if total_weight > 0:
        for key in stat_keys:
            weighted[key] /= total_weight

    # 타구 유형 비율 (구종 가중이 아닌 타자 전체 기준)
    for key in ["ld_rate", "gb_rate", "fb_rate", "iffb_rate", "iso"]:
        if batter_arsenal:
            # PA 가중 평균
            total_pa = sum(b.get("n_pa", 0) for b in batter_arsenal)
            if total_pa > 0:
                weighted[key] = sum(
                    b.get(key, league_avg.get(key, 0.0)) * b.get("n_pa", 0)
                    for b in batter_arsenal
                ) / total_pa
            else:
                weighted[key] = league_avg.get(key, 0.0)
        else:
            weighted[key] = league_avg.get(key, 0.0)

    return weighted


# ============================================================
# Split adjustments
# ============================================================

def apply_split_ratio(
    stats: dict,
    split_stats: dict | None,
    overall_stats: dict | None,
    n_pa: int,
    stat_keys: list[str] = None,
) -> dict:
    """스플릿 보정 (ratio 기반, shrinkage 적용).

    ratio = split_stat / overall_stat
    shrunk_ratio = shrink(ratio, 1.0, n_pa, k=80)
    adjusted = base * shrunk_ratio
    """
    if not split_stats or not overall_stats or n_pa < 5:
        return stats

    if stat_keys is None:
        stat_keys = ["k_rate", "bb_rate", "hbp_rate", "hr_rate", "babip"]

    result = dict(stats)
    k = SHRINKAGE["split_ratio"]

    for key in stat_keys:
        split_val = split_stats.get(key)
        overall_val = overall_stats.get(key)
        if split_val is None or overall_val is None or overall_val <= 0:
            continue

        raw_ratio = split_val / overall_val
        shrunk_ratio = shrink(raw_ratio, 1.0, n_pa, k)
        # 극단적 보정 방지: [0.75, 1.30]
        shrunk_ratio = max(0.75, min(1.30, shrunk_ratio))
        result[key] = result.get(key, 0) * shrunk_ratio

    return result


def apply_platoon(stats: dict, batter_hand: str, pitcher_hand: str) -> dict:
    """좌우 스플릿 보정 (모집단 수준)."""
    same = (batter_hand == pitcher_hand)
    key = "same_hand" if same else "opp_hand"
    multipliers = PLATOON_MULTIPLIERS[key]

    result = dict(stats)
    for stat, mult in multipliers.items():
        if stat in result:
            result[stat] = result[stat] * mult
    return result


def apply_park_factor(stats: dict, home_team: str) -> dict:
    """파크팩터 보정 (50% 강도)."""
    pf_runs = PARK_FACTORS.get(home_team, 100) / 100.0
    pf_hr = PARK_HR_FACTORS.get(home_team, 100) / 100.0

    # 50% 강도 (시즌 스탯에 이미 ~50% 반영됨)
    adj_runs = 1.0 + (pf_runs - 1.0) * 0.5
    adj_hr = 1.0 + (pf_hr - 1.0) * 0.5

    result = dict(stats)
    result["babip"] = result.get("babip", 0.300) * adj_runs
    result["hr_rate"] = result.get("hr_rate", 0.03) * adj_hr
    return result


def apply_defense(stats: dict, defense_bucket: str) -> dict:
    """팀 수비 보정 (BABIP에 적용)."""
    modifier = DEFENSE_BUCKETS.get(defense_bucket, 1.0)
    result = dict(stats)
    result["babip"] = result.get("babip", 0.300) * modifier
    return result


# ============================================================
# H2H blending
# ============================================================

def blend_h2h(
    matchup_stats: dict,
    h2h_stats: dict | None,
    n_h2h_pa: int,
) -> dict:
    """H2H 직접 대전 블렌딩 (shrinkage 적용).

    w = n / (n + 50)
    """
    if not h2h_stats or n_h2h_pa < 3:
        return matchup_stats

    k = SHRINKAGE["h2h"]
    result = dict(matchup_stats)

    for key in ["k_rate", "bb_rate", "hbp_rate", "hr_rate", "babip"]:
        h2h_val = h2h_stats.get(key)
        base_val = matchup_stats.get(key)
        if h2h_val is not None and base_val is not None:
            result[key] = shrink(h2h_val, base_val, n_h2h_pa, k)

    return result


# ============================================================
# Stage 2: BIP sub-probabilities
# ============================================================

def compute_bip_distribution(stats: dict) -> dict:
    """BIP 이벤트의 세부 확률 벡터 계산.

    target BABIP에 정확히 맞도록 스케일링:
    1. 타구유형별 base BABIP로 "자연" hit rate 계산
    2. target BABIP / natural hit rate = scale factor
    3. 각 유형의 BABIP에 scale factor 적용

    Returns:
        dict with keys: 1B, 2B, 3B, GO, FO, LO, FC, ROE
        합계 = 1.0
    """
    ld = stats.get("ld_rate", 0.21)
    gb = stats.get("gb_rate", 0.43)
    fb = stats.get("fb_rate", 0.27)
    iffb = stats.get("iffb_rate", 0.10)  # FB 내 비율

    babip = stats.get("babip", 0.300)
    iso = stats.get("iso", 0.150)

    # 타구유형 share 정규화 (합이 1.0이 되도록)
    fb_actual_raw = fb * (1 - iffb)
    iffb_actual_raw = fb * iffb
    type_total = ld + gb + fb_actual_raw + iffb_actual_raw
    if type_total > 0:
        ld = ld / type_total
        gb = gb / type_total
        fb_actual = fb_actual_raw / type_total
        iffb_actual = iffb_actual_raw / type_total
    else:
        ld, gb, fb_actual, iffb_actual = 0.21, 0.43, 0.306, 0.034

    # Step 1: 타구유형별 base BABIP로 "자연" BIP hit rate 계산
    natural_hit_rate = (
        ld * BABIP_BY_BATTED_BALL["ld"]
        + gb * BABIP_BY_BATTED_BALL["gb"]
        + fb_actual * BABIP_BY_BATTED_BALL["fb"]
        + iffb_actual * BABIP_BY_BATTED_BALL["iffb"]
    )

    # Step 2: target BABIP에 맞추기 위한 scale factor
    if natural_hit_rate > 0:
        scale = babip / natural_hit_rate
    else:
        scale = 1.0

    # Step 3: 각 유형별 보정된 BABIP
    ld_hit_prob = min(0.95, BABIP_BY_BATTED_BALL["ld"] * scale)
    gb_hit_prob = min(0.95, BABIP_BY_BATTED_BALL["gb"] * scale)
    fb_hit_prob = min(0.95, BABIP_BY_BATTED_BALL["fb"] * scale)
    iffb_hit_prob = BABIP_BY_BATTED_BALL["iffb"]  # IFFB는 거의 항상 아웃

    # ISO boost for hit type distribution
    iso_boost = (iso / 0.150) ** 0.5 if iso > 0 else 1.0

    # 각 타구별 안타 유형 분배
    def distribute_hits(batted_type: str, hit_prob: float, type_pct: float):
        dist = HIT_TYPE_DISTRIBUTION.get(batted_type, {"1B": 0.80, "2B": 0.15, "3B": 0.05})
        p_1b = dist["1B"]
        p_2b = dist["2B"] * iso_boost
        p_3b = dist["3B"] * iso_boost
        total = p_1b + p_2b + p_3b
        p_1b, p_2b, p_3b = p_1b/total, p_2b/total, p_3b/total

        hit_share = type_pct * hit_prob
        out_share = type_pct * (1 - hit_prob)

        return {
            "1B": hit_share * p_1b,
            "2B": hit_share * p_2b,
            "3B": hit_share * p_3b,
        }, out_share

    out_map = {"ld": "LO", "gb": "GO", "fb": "FO"}

    result = {"1B": 0, "2B": 0, "3B": 0, "GO": 0, "FO": 0, "LO": 0, "FC": 0, "ROE": 0}

    for batted_type, hit_prob, type_pct in [
        ("ld", ld_hit_prob, ld),
        ("gb", gb_hit_prob, gb),
        ("fb", fb_hit_prob, fb_actual),
    ]:
        hits, out_share = distribute_hits(batted_type, hit_prob, type_pct)
        for k, v in hits.items():
            result[k] += v
        result[out_map[batted_type]] += out_share

    # IFFB -> FO
    result["FO"] += iffb_actual * (1 - iffb_hit_prob)
    result["1B"] += iffb_actual * iffb_hit_prob

    # FC/ROE를 GO에서 차감
    total_before = sum(result.values())
    fc_amount = total_before * BIP_FC_RATE
    roe_amount = total_before * BIP_ROE_RATE

    result["FC"] = fc_amount
    result["ROE"] = roe_amount
    # GO에서 차감 (FC/ROE 대부분 ground ball 상황)
    result["GO"] = max(0.001, result["GO"] - fc_amount - roe_amount)

    # 정규화
    total = sum(result.values())
    if total > 0:
        result = {k: v / total for k, v in result.items()}

    return result


# ============================================================
# Main matchup computation
# ============================================================

def compute_matchup_v2(
    batter: dict,
    pitcher: dict,
    league_avg: dict,
    home_team: str,
    is_batter_home: bool,
    count_bucket: str = "first_pitch",
    batter_arsenal: list[dict] | None = None,
    pitcher_arsenal: list[dict] | None = None,
    h2h_stats: dict | None = None,
    batter_splits: dict | None = None,
    pitcher_splits: dict | None = None,
    count_pitch_mix: pd.DataFrame | None = None,
    defense_bucket: str = "avg",
    runner_state: str = "empty",
    use_team_strength: bool = False,
    # Ablation flags
    disable_platoon: bool = False,
    disable_park: bool = False,
    disable_h2h: bool = False,
    disable_era_fip: bool = False,
) -> dict:
    """v2 매치업 확률 벡터 생성.

    Returns:
        dict with:
        - stage1: {k_rate, bb_rate, hbp_rate, hr_rate, bip_rate}
        - stage2_bip: {1B, 2B, 3B, GO, FO, LO, FC, ROE}
        - raw stats for debugging
    """
    batter_hand = batter.get("bats", "R")
    pitcher_hand = pitcher.get("throws", "R")

    # 1. 타자/투수 base rate에 shrinkage 적용
    batter_n = batter.get("pa", 0)
    pitcher_n = pitcher.get("tbf", 0)

    base_stats = {}
    for key in ["k_rate", "bb_rate", "hbp_rate", "hr_rate", "babip"]:
        b_val = shrink_rate(
            batter.get(key, league_avg.get(key, 0.0)),
            league_avg.get(key, 0.0),
            batter_n,
            SHRINKAGE["batter_rate"] if key != "hr_rate" else SHRINKAGE["batter_hr"],
        )
        p_val = shrink_rate(
            pitcher.get(key, league_avg.get(key, 0.0)),
            league_avg.get(key, 0.0),
            pitcher_n,
            SHRINKAGE["pitcher_rate"] if key != "hr_rate" else SHRINKAGE["pitcher_hr"],
        )
        base_stats[key] = log5(b_val, p_val, league_avg.get(key, 0.0))

    # ISO, 타구 유형은 shrinkage 후 복사
    for key in ["iso", "ld_rate", "gb_rate", "fb_rate", "iffb_rate"]:
        base_stats[key] = batter.get(key, league_avg.get(key, 0.0))

    # 2. 구종별 가중 매치업 (count-aware)
    if batter_arsenal and pitcher_arsenal:
        pitch_weighted = compute_pitch_weighted_stats(
            batter_arsenal, pitcher_arsenal, league_avg,
            count_pitch_mix, count_bucket, batter_hand,
        )
        # pitch-weighted 결과와 base stats 블렌딩 (50/50)
        for key in ["k_rate", "bb_rate", "hbp_rate", "hr_rate", "babip"]:
            base_stats[key] = 0.5 * base_stats[key] + 0.5 * pitch_weighted.get(key, base_stats[key])
        for key in ["iso", "ld_rate", "gb_rate", "fb_rate", "iffb_rate"]:
            if key in pitch_weighted:
                base_stats[key] = pitch_weighted[key]

    # 3. H2H 블렌딩
    if h2h_stats and not disable_h2h:
        n_h2h = h2h_stats.get("n_pa", 0)
        base_stats = blend_h2h(base_stats, h2h_stats, n_h2h)

    # 4. 스플릿 보정
    # Platoon (모집단 수준)
    if not disable_platoon:
        base_stats = apply_platoon(base_stats, batter_hand, pitcher_hand)

    # 개인 스플릿 (있으면)
    if batter_splits:
        # Home/Away
        ha_key = "home" if is_batter_home else "away"
        ha_split = batter_splits.get("home_away", {}).get(ha_key)
        ha_overall = batter_splits.get("home_away", {}).get("overall")
        if ha_split:
            base_stats = apply_split_ratio(
                base_stats, ha_split, ha_overall or batter,
                ha_split.get("n_pa", 0),
            )

        # RISP
        if runner_state in ("runners_on", "risp"):
            risp_split = batter_splits.get("runners", {}).get("risp")
            risp_overall = batter_splits.get("runners", {}).get("overall")
            if risp_split:
                base_stats = apply_split_ratio(
                    base_stats, risp_split, risp_overall or batter,
                    risp_split.get("n_pa", 0),
                )

    # 5. 파크팩터
    if not disable_park:
        base_stats = apply_park_factor(base_stats, home_team)

    # 6. 수비 보정
    base_stats = apply_defense(base_stats, defense_bucket)

    # 7. 팀 승률 (기본 OFF)
    if use_team_strength:
        pct_path = os.path.join(DATA_DIR, "team_win_pct.json")
        if os.path.exists(pct_path):
            with open(pct_path) as f:
                pcts = json.load(f)
            batting_team = batter.get("team", "")
            bat_pct = pcts.get(batting_team, 0.500)
            team_adj = 1.0 + (bat_pct - 0.500) * 0.30  # 약한 보정
            base_stats["babip"] *= team_adj
            base_stats["hr_rate"] *= team_adj

    # 7.5. xwOBA/ERA-FIP 투수 퀄리티 보정
    #   ERA-FIP gap: 양수면 운 나쁨 (실제보다 더 좋은 투수), 음수면 운 좋음
    #   보정 방향: FIP < ERA → 투수가 과소평가됨 → K↑, HR↓ 약간 보정
    pitcher_era = pitcher.get("era")
    pitcher_fip = pitcher.get("fip")
    pitcher_xwoba = pitcher.get("xwoba")

    if pitcher_era is not None and pitcher_fip is not None and pitcher_fip > 0 and pitcher_n >= 50 and not disable_era_fip:
        era_fip_gap = pitcher_era - pitcher_fip  # 양수 = unlucky
        # 작은 보정: gap * 0.01 만큼 K%/BB%/HR% 조정
        adj = max(-0.03, min(0.03, era_fip_gap * 0.01))
        base_stats["k_rate"] = base_stats["k_rate"] + adj      # unlucky → K↑
        base_stats["bb_rate"] = base_stats["bb_rate"] - adj * 0.5  # unlucky → BB↓
        base_stats["hr_rate"] = base_stats["hr_rate"] - adj     # unlucky → HR↓
        # clamp
        base_stats["k_rate"] = max(0.05, min(0.45, base_stats["k_rate"]))
        base_stats["bb_rate"] = max(0.02, min(0.20, base_stats["bb_rate"]))
        base_stats["hr_rate"] = max(0.005, min(0.08, base_stats["hr_rate"]))

    # 8. Stage1 확률 벡터 구성
    k = max(0.01, min(0.50, base_stats["k_rate"]))
    bb = max(0.01, min(0.25, base_stats["bb_rate"]))
    hbp = max(0.002, min(0.05, base_stats["hbp_rate"]))
    hr = max(0.005, min(0.10, base_stats["hr_rate"]))

    # BIP = 나머지
    bip = max(0.10, 1.0 - k - bb - hbp - hr)

    # 정규화
    total = k + bb + hbp + hr + bip
    stage1 = {
        "k_rate": k / total,
        "bb_rate": bb / total,
        "hbp_rate": hbp / total,
        "hr_rate": hr / total,
        "bip_rate": bip / total,
    }

    # 9. Stage2 BIP 분포
    stage2_bip = compute_bip_distribution(base_stats)

    return {
        "stage1": stage1,
        "stage2_bip": stage2_bip,
        # 디버깅/호환용 flat dict
        "k_rate": stage1["k_rate"],
        "bb_rate": stage1["bb_rate"],
        "hbp_rate": stage1["hbp_rate"],
        "hr_rate": stage1["hr_rate"],
        "bip_rate": stage1["bip_rate"],
        "babip": base_stats.get("babip", 0.300),
        "iso": base_stats.get("iso", 0.150),
        "ld_rate": base_stats.get("ld_rate", 0.21),
        "gb_rate": base_stats.get("gb_rate", 0.43),
        "fb_rate": base_stats.get("fb_rate", 0.27),
        "iffb_rate": base_stats.get("iffb_rate", 0.10),
        "gdp_rate": batter.get("gdp_rate", 0.15),
    }
