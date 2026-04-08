"""v4 매치업 엔진 — 구종별 가중 + H2H + 스플릿 + ERA + 팀 승률."""

import json
import os
from config import PLATOON_MULTIPLIERS, PARK_FACTORS, PARK_HR_FACTORS, DATA_DIR
from engine.lineup import (
    get_batter_pitch_arsenal, get_pitcher_pitch_arsenal, get_h2h_stats,
    get_batter_splits, get_pitcher_splits,
)

# 팀 승률 캐시
_team_win_pct = None

def _get_team_win_pct() -> dict:
    global _team_win_pct
    if _team_win_pct is None:
        path = os.path.join(DATA_DIR, "team_win_pct.json")
        if os.path.exists(path):
            with open(path) as f:
                _team_win_pct = json.load(f)
        else:
            _team_win_pct = {}
    return _team_win_pct


def apply_era_fip_adjustment(stats: dict, pitcher: dict, league_avg: dict) -> dict:
    """ERA + FIP 결합 투수 퀄리티 보정.

    FIP(운 제거) 70% + ERA(결과 포함) 30% 가중 = 투수 진짜 실력 근사.
    """
    era = pitcher.get("era")
    fip = pitcher.get("fip")

    if (era is None or era <= 0) and (fip is None or fip <= 0):
        return stats

    LEAGUE_AVG_ERA = 4.2

    # FIP 우선, 없으면 ERA
    if fip and fip > 0 and era and era > 0:
        combined = fip * 0.7 + era * 0.3  # FIP 70%, ERA 30%
    elif fip and fip > 0:
        combined = fip
    else:
        combined = era

    quality_ratio = combined / LEAGUE_AVG_ERA
    # 보정 강도 35%
    adj = 1.0 + (quality_ratio - 1.0) * 0.35

    result = dict(stats)
    result["babip"] = result.get("babip", 0.300) * adj
    result["hr_rate"] = result.get("hr_rate", 0.03) * adj
    return result


def apply_xwoba_adjustment(stats: dict, batter_arsenal: list, pitcher_arsenal: list) -> dict:
    """xwOBA 기반 타구 퀄리티 보정.

    구종별 xwOBA가 있으면 BABIP/HR를 보정.
    xwOBA는 타구 속도+각도 기반이라 BABIP보다 정확.
    """
    # 타자의 전체 xwOBA 평균
    batter_xwoba = 0
    total_pa = 0
    for b in batter_arsenal:
        if b.get("xwoba", 0) > 0 and b.get("n_pa", 0) > 0:
            batter_xwoba += b["xwoba"] * b["n_pa"]
            total_pa += b["n_pa"]
    if total_pa > 0:
        batter_xwoba = batter_xwoba / total_pa
    else:
        return stats

    # 투수의 전체 xwOBA (피타)
    pitcher_xwoba = 0
    total_pa_p = 0
    for p in pitcher_arsenal:
        if p.get("xwoba", 0) > 0 and p.get("n_pa", 0) > 0:
            pitcher_xwoba += p["xwoba"] * p["n_pa"]
            total_pa_p += p["n_pa"]
    if total_pa_p > 0:
        pitcher_xwoba = pitcher_xwoba / total_pa_p

    LEAGUE_AVG_XWOBA = 0.315
    if batter_xwoba <= 0:
        return stats

    # 타자 xwOBA 비율: 리그 평균 대비 얼마나 좋은지
    xwoba_ratio = batter_xwoba / LEAGUE_AVG_XWOBA
    # 보정 강도 20% (BABIP 계산에 이미 반영된 부분과 중복 방지)
    adj = 1.0 + (xwoba_ratio - 1.0) * 0.20

    result = dict(stats)
    result["babip"] = result.get("babip", 0.300) * adj
    result["hr_rate"] = result.get("hr_rate", 0.03) * adj
    return result


def apply_team_strength(stats: dict, batting_team: str, pitching_team: str) -> dict:
    """팀 전력 차이 보정.

    강팀 타선 → 약간 상향, 약팀 타선 → 약간 하향.
    리그 평균 승률 .500 기준.
    """
    pcts = _get_team_win_pct()
    bat_pct = pcts.get(batting_team, 0.500)
    # pit_pct = pcts.get(pitching_team, 0.500)  # 투수팀 전력은 ERA로 이미 반영

    # 타선 전력 보정: .600 팀은 +6%, .400 팀은 -6%
    team_adj = 1.0 + (bat_pct - 0.500) * 0.60

    result = dict(stats)
    result["babip"] = result.get("babip", 0.300) * team_adj
    result["hr_rate"] = result.get("hr_rate", 0.03) * team_adj
    result["bb_rate"] = result.get("bb_rate", 0.08) * (1.0 + (bat_pct - 0.500) * 0.30)
    return result


def log5(p_batter: float, p_pitcher: float, p_league: float) -> float:
    """Log5 공식."""
    if p_league <= 0 or p_league >= 1:
        return p_batter
    num = p_batter * p_pitcher / p_league
    den = num + (1 - p_batter) * (1 - p_pitcher) / (1 - p_league)
    if den <= 0:
        return p_batter
    return max(0.001, min(0.999, num / den))


def apply_platoon(stats: dict, batter_hand: str, pitcher_hand: str,
                   batter_splits: dict | None = None, pitcher_splits: dict | None = None) -> dict:
    """좌우 스플릿 보정 — 개인 데이터 있으면 개인 스플릿, 없으면 모집단 평균."""
    result = dict(stats)
    stat_keys = ["k_rate", "bb_rate", "hr_rate", "babip"]

    # 타자의 개인 좌우 스플릿
    batter_platoon = None
    if batter_splits and "platoon" in batter_splits:
        batter_platoon = batter_splits["platoon"].get(pitcher_hand)

    # 투수의 개인 좌우 스플릿
    pitcher_platoon = None
    if pitcher_splits and "platoon" in pitcher_splits:
        pitcher_platoon = pitcher_splits["platoon"].get(batter_hand)

    if batter_platoon and batter_platoon.get("n_pa", 0) >= 30:
        # 개인 스플릿 적용: 해당 투수 손잡이 상대 성적 / 전체 성적 비율
        for key in stat_keys:
            if key in batter_platoon and result[key] > 0:
                # 스플릿 값과 기본 값의 비율로 보정
                split_val = batter_platoon[key]
                ratio = split_val / result[key] if result[key] > 0 else 1.0
                # 극단값 억제 (0.7~1.4 범위)
                ratio = max(0.7, min(1.4, ratio))
                result[key] = result[key] * ratio
    else:
        # 모집단 평균 사용
        if batter_hand == pitcher_hand:
            mult = PLATOON_MULTIPLIERS["same_hand"]
        else:
            mult = PLATOON_MULTIPLIERS["opp_hand"]
        for key in stat_keys:
            if key in result and key in mult:
                result[key] = result[key] * mult[key]

    return result


def apply_situation_splits(
    stats: dict,
    batter_id: int | None,
    pitcher_id: int | None,
    batter_splits: dict | None,
    pitcher_splits: dict | None,
    is_home: bool,
    runner_state: str,  # "empty", "runners_on", "risp"
    game_month: int = 7,  # 경기 월 (기본 7월)
) -> dict:
    """상황별 스플릿 보정.

    홈/원정: 스플릿 데이터가 충분하면 기본 스탯을 직접 대체.
    RISP: 비율 보정.
    """
    result = dict(stats)
    stat_keys = ["k_rate", "bb_rate", "hr_rate", "babip"]

    def _replace_with_split(split_data: dict | None, min_pa: int = 50, blend: float = 0.7):
        """스플릿 스탯으로 직접 대체 (blend 비율)."""
        if not split_data or split_data.get("n_pa", 0) < min_pa:
            return
        for key in stat_keys:
            if key in split_data:
                result[key] = result[key] * (1 - blend) + split_data[key] * blend

    def _apply_split_ratio(split_data: dict | None):
        """비율 보정 (RISP 등)."""
        if not split_data or split_data.get("n_pa", 0) < 20:
            return
        for key in stat_keys:
            if key in split_data and key in stats and stats[key] > 0:
                ratio = split_data[key] / stats[key]
                ratio = max(0.75, min(1.3, ratio))
                result[key] = result[key] * ratio

    # 타자 홈/원정 — 직접 대체 (쿠어스 등 파크 효과 이미 포함)
    if batter_splits and "home_away" in batter_splits:
        ha_key = "home" if is_home else "away"
        ha_split = batter_splits["home_away"].get(ha_key)
        _replace_with_split(ha_split, min_pa=50, blend=0.7)

    # 투수 홈/원정 — 직접 대체
    if pitcher_splits and "home_away" in pitcher_splits:
        p_ha_key = "away" if is_home else "home"
        p_ha_split = pitcher_splits["home_away"].get(p_ha_key)
        _replace_with_split(p_ha_split, min_pa=50, blend=0.5)

    # 타자 RISP — 비율 보정
    if batter_splits and "runners" in batter_splits and runner_state != "empty":
        runner_split = batter_splits["runners"].get(runner_state)
        _apply_split_ratio(runner_split)

    # 투수 RISP — 비율 보정
    if pitcher_splits and "runners" in pitcher_splits and runner_state != "empty":
        p_runner_split = pitcher_splits["runners"].get(runner_state)
        _apply_split_ratio(p_runner_split)

    # 월별 스플릿: 제외 (2~3시즌 표본으로는 노이즈가 큼)

    return result


def apply_park_factor(stats: dict, home_team: str, is_batter_home: bool = False) -> dict:
    """파크팩터 보정 — 50% 적용 (시즌 스탯에 이미 홈 효과 반절 포함).

    홈팀 타자: 시즌 스탯에 이미 쿠어스 효과 ~50% 반영됨 → 추가 보정 25%만
    원정 타자: 시즌 스탯에 쿠어스 효과 없음 → 추가 보정 75%
    """
    runs_raw = PARK_FACTORS.get(home_team, 100) / 100  # 예: 쿠어스 1.14
    hr_raw = PARK_HR_FACTORS.get(home_team, 100) / 100  # 예: 쿠어스 1.12

    # 중립(1.0)으로부터의 차이를 비율로 적용
    if is_batter_home:
        # 홈 타자: 홈/원정 스플릿이 이미 홈구장 효과 포함 → 파크팩터 미적용
        return dict(stats)
    else:
        # 원정 타자: 원정 스플릿 기반 → 상대 구장 파크팩터 풀적용
        park_strength = 1.0

    runs_factor = 1.0 + (runs_raw - 1.0) * park_strength
    hr_factor = 1.0 + (hr_raw - 1.0) * park_strength

    result = dict(stats)
    result["hr_rate"] = result["hr_rate"] * hr_factor
    babip_adj = (runs_factor - 1) * 0.3 + 1
    result["babip"] = result["babip"] * babip_adj
    return result


def _compute_pitch_type_stats(
    batter_arsenal: list[dict],
    pitcher_arsenal: list[dict],
    league_avg: dict,
) -> dict:
    """구종별 가중 매치업 확률 계산.

    투수의 구종 비율로 가중하여 각 구종에서의 타자 vs 투수 성적을 결합.

    예: 콜이 4-Seam 35%, Slider 30% 던지면
        → 오타니의 4-Seam 대응 K% × 0.35 + 오타니의 Slider 대응 K% × 0.30 + ...
    """
    # 투수 구종 → dict로 변환
    pitcher_by_type = {}
    for p in pitcher_arsenal:
        pitcher_by_type[p["pitch_type"]] = p

    # 타자 구종 → dict로 변환
    batter_by_type = {}
    for b in batter_arsenal:
        batter_by_type[b["pitch_type"]] = b

    # 결합할 스탯 키
    stat_keys = ["k_rate", "bb_rate", "hr_rate", "hbp_rate", "babip",
                 "ld_rate", "gb_rate", "fb_rate"]

    weighted_stats = {k: 0.0 for k in stat_keys}
    total_weight = 0.0

    for pitch_type, p_stats in pitcher_by_type.items():
        usage = p_stats.get("usage", 0)
        if usage < 0.02:  # 2% 미만 구종 무시
            continue

        b_stats = batter_by_type.get(pitch_type, None)

        for key in stat_keys:
            p_val = p_stats.get(key, league_avg.get(key, 0.1))

            if b_stats and b_stats.get("n_pa", 0) >= 3:
                b_val = b_stats.get(key, league_avg.get(key, 0.1))
                # 타자와 투수의 구종별 스탯을 Log5로 결합
                combined = log5(b_val, p_val, league_avg.get(key, 0.1))
            else:
                # 타자의 해당 구종 데이터 없으면 투수 구종 스탯만 사용
                combined = p_val

            weighted_stats[key] += combined * usage

        total_weight += usage

    # 정규화
    if total_weight > 0:
        for key in stat_keys:
            weighted_stats[key] /= total_weight

    return weighted_stats


def _blend_h2h(
    pitch_based_stats: dict,
    h2h: dict | None,
    batter_season: dict,
) -> dict:
    """H2H 직접 대전 기록을 구종 기반 스탯에 블렌딩.

    h2h_weight = min(0.5, h2h_pa / 100)
    표본 3PA → 3%, 50PA → 25%, 100+PA → 50%
    """
    if h2h is None or h2h.get("n_pa", 0) < 1:
        return pitch_based_stats

    h2h_pa = h2h["n_pa"]
    h2h_weight = min(0.5, h2h_pa / 100)

    blended = {}
    stat_keys = ["k_rate", "bb_rate", "hr_rate", "hbp_rate", "babip",
                 "ld_rate", "gb_rate", "fb_rate"]

    for key in stat_keys:
        base_val = pitch_based_stats.get(key, 0)
        h2h_val = h2h.get(key, base_val)
        blended[key] = h2h_weight * h2h_val + (1 - h2h_weight) * base_val

    # iso, iffb_rate 등은 pitch_based에서 가져옴
    for key in pitch_based_stats:
        if key not in blended:
            blended[key] = pitch_based_stats[key]

    # H2H iso 있으면 블렌딩
    if "iso" in h2h:
        base_iso = pitch_based_stats.get("iso", batter_season.get("iso", 0.150))
        blended["iso"] = h2h_weight * h2h["iso"] + (1 - h2h_weight) * base_iso

    return blended


def compute_matchup_probs(
    batter: dict,
    pitcher: dict,
    league_avg: dict,
    home_team: str,
    is_batter_home: bool = False,
    runner_state: str = "empty",
) -> dict:
    """v3 매치업 확률 계산 — 구종별 가중 + H2H + 스플릿.

    1. 투수의 구종 비율 × 각 구종에서의 타자 vs 투수 Log5
    2. H2H 직접 대전 있으면 표본 크기 비례 블렌딩
    3. 개인별 좌우 스플릿 (없으면 모집단)
    4. 홈/원정 + RISP 스플릿 보정
    5. 파크팩터 보정
    """
    batter_id = batter.get("player_id")
    pitcher_id = pitcher.get("player_id")

    # 구종별 데이터 로드
    batter_arsenal = get_batter_pitch_arsenal(batter_id) if batter_id else []
    pitcher_arsenal = get_pitcher_pitch_arsenal(pitcher_id) if pitcher_id else []

    if pitcher_arsenal and batter_arsenal:
        # v2: 구종별 가중 매치업
        pitch_stats = _compute_pitch_type_stats(batter_arsenal, pitcher_arsenal, league_avg)
    elif pitcher_arsenal:
        # 타자 구종 데이터 없으면 투수 구종 + 타자 시즌 평균 Log5
        pitch_stats = {}
        for key in ["k_rate", "bb_rate", "hr_rate", "hbp_rate", "babip",
                     "ld_rate", "gb_rate", "fb_rate"]:
            pitch_stats[key] = log5(
                batter.get(key, league_avg.get(key, 0.1)),
                pitcher.get(key, league_avg.get(key, 0.1)),
                league_avg.get(key, 0.1),
            )
    else:
        # 구종 데이터 없으면 v1 fallback (시즌 평균 Log5)
        pitch_stats = {}
        for key in ["k_rate", "bb_rate", "hr_rate", "hbp_rate", "babip",
                     "ld_rate", "gb_rate", "fb_rate"]:
            pitch_stats[key] = log5(
                batter.get(key, league_avg.get(key, 0.1)),
                pitcher.get(key, league_avg.get(key, 0.1)),
                league_avg.get(key, 0.1),
            )

    # ISO, iffb_rate, gdp_rate는 타자 시즌 스탯에서
    pitch_stats["iso"] = batter.get("iso", 0.150)
    pitch_stats["iffb_rate"] = batter.get("iffb_rate", 0.10)

    # H2H 블렌딩
    h2h = get_h2h_stats(batter_id, pitcher_id) if batter_id and pitcher_id else None
    probs = _blend_h2h(pitch_stats, h2h, batter)

    # 스플릿 데이터 로드
    batter_splits = get_batter_splits(batter_id) if batter_id else None
    pitcher_splits = get_pitcher_splits(pitcher_id) if pitcher_id else None

    # 좌우 스플릿 보정 (개인 데이터 우선)
    batter_hand = batter.get("bats", "R")
    pitcher_hand = pitcher.get("throws", "R")
    probs = apply_platoon(probs, batter_hand, pitcher_hand, batter_splits, pitcher_splits)

    # 홈/원정 + RISP 스플릿 보정
    probs = apply_situation_splits(
        probs, batter_id, pitcher_id,
        batter_splits, pitcher_splits,
        is_home=is_batter_home,
        runner_state=runner_state,
    )

    # 파크팩터 보정 (홈/원정 구분)
    probs = apply_park_factor(probs, home_team, is_batter_home=is_batter_home)

    # ERA+FIP 투수 퀄리티 보정
    probs = apply_era_fip_adjustment(probs, pitcher, league_avg)

    # xwOBA 타구 퀄리티 보정 (위에서 이미 로드한 arsenal 재사용)
    probs = apply_xwoba_adjustment(probs, batter_arsenal, pitcher_arsenal)

    # 팀 전력 보정
    batting_team = batter.get("current_team", batter.get("team", ""))
    probs = apply_team_strength(probs, batting_team, pitcher.get("current_team", pitcher.get("team", "")))

    # 현실적 범위 클램핑 — 구종별 소표본에서 극단값 억제
    probs["babip"] = max(0.180, min(0.400, probs.get("babip", 0.300)))
    probs["k_rate"] = max(0.05, min(0.45, probs.get("k_rate", 0.22)))
    probs["bb_rate"] = max(0.02, min(0.20, probs.get("bb_rate", 0.08)))
    probs["hr_rate"] = max(0.005, min(0.08, probs.get("hr_rate", 0.03)))
    probs["hbp_rate"] = max(0.002, min(0.03, probs.get("hbp_rate", 0.01)))

    # 타구 유형 비율 정규화
    batted_total = probs.get("ld_rate", 0.21) + probs.get("gb_rate", 0.43) + probs.get("fb_rate", 0.34)
    if batted_total > 0:
        probs["ld_rate"] = probs.get("ld_rate", 0.21) / batted_total
        probs["gb_rate"] = probs.get("gb_rate", 0.43) / batted_total
        probs["fb_rate"] = probs.get("fb_rate", 0.34) / batted_total

    # K + BB + HBP + HR이 1을 넘지 않도록
    total_non_bip = probs["k_rate"] + probs["bb_rate"] + probs["hbp_rate"] + probs["hr_rate"]
    if total_non_bip >= 0.95:
        scale = 0.95 / total_non_bip
        probs["k_rate"] *= scale
        probs["bb_rate"] *= scale
        probs["hbp_rate"] *= scale
        probs["hr_rate"] *= scale

    probs["bip_rate"] = 1.0 - probs["k_rate"] - probs["bb_rate"] - probs["hbp_rate"] - probs["hr_rate"]

    return probs
