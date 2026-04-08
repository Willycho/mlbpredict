"""타석(PA) 결과를 확률적으로 결정."""

import numpy as np

from config import BABIP_BY_BATTED_BALL, HIT_TYPE_DISTRIBUTION


ERROR_RATE = 0.08       # Full game (9이닝, 불펜 포함)
ERROR_RATE_F5 = 0.04   # F5 (5이닝, 선발만)

def resolve_pa(probs: dict, rng: np.random.Generator) -> str:
    """하나의 타석 결과를 랜덤으로 결정.

    Args:
        probs: compute_matchup_probs()의 결과
        rng: numpy random generator

    Returns:
        결과 문자열: "K", "BB", "HBP", "HR", "1B", "2B", "3B",
                    "GO" (ground out), "FO" (fly out), "LO" (line out), "PO" (popup out)
    """
    r = rng.random()

    # 확률 누적으로 결과 결정
    cum = 0.0

    # 삼진
    cum += probs["k_rate"]
    if r < cum:
        return "K"

    # 볼넷
    cum += probs["bb_rate"]
    if r < cum:
        return "BB"

    # 사구
    cum += probs["hbp_rate"]
    if r < cum:
        return "HBP"

    # 홈런
    cum += probs["hr_rate"]
    if r < cum:
        return "HR"

    # 여기부터 Ball In Play (BIP)
    # 타구 유형 결정
    r2 = rng.random()
    ld = probs["ld_rate"]
    gb = probs["gb_rate"]
    fb = probs["fb_rate"]
    # iffb는 fb 내에서의 비율
    iffb = probs.get("iffb_rate", 0.10)

    if r2 < ld:
        result = _resolve_batted_ball("ld", probs, rng)
    elif r2 < ld + gb:
        result = _resolve_batted_ball("gb", probs, rng)
    else:
        if rng.random() < iffb:
            result = "PO"
        else:
            result = _resolve_batted_ball("fb", probs, rng)

    # 에러/야수선택 보정: 아웃 결과 중 일부를 출루(1B)로 전환
    # MLB 실제 에러+야수선택+폭투 등 종합 출루율 ~8%
    if result in ("GO", "FO", "LO", "PO") and rng.random() < ERROR_RATE:
        result = "1B"

    return result


def _resolve_batted_ball(batted_type: str, probs: dict, rng: np.random.Generator) -> str:
    """타구가 안타인지 아웃인지, 안타라면 어떤 종류인지 결정."""
    # 타구 유형별 BABIP (개인 BABIP으로 보정)
    base_babip = BABIP_BY_BATTED_BALL[batted_type]
    # 개인 BABIP과 기본 BABIP의 비율로 보정
    league_babip = 0.300
    personal_babip = probs.get("babip", league_babip)
    babip_ratio = personal_babip / league_babip if league_babip > 0 else 1.0
    adjusted_babip = min(0.95, base_babip * babip_ratio)

    if rng.random() < adjusted_babip:
        # 안타! 종류 결정
        return _resolve_hit_type(batted_type, probs, rng)
    else:
        # 아웃
        out_map = {"ld": "LO", "gb": "GO", "fb": "FO"}
        return out_map[batted_type]


def _resolve_hit_type(batted_type: str, probs: dict, rng: np.random.Generator) -> str:
    """안타의 종류(1B/2B/3B) 결정."""
    dist = HIT_TYPE_DISTRIBUTION.get(batted_type, {"1B": 0.80, "2B": 0.15, "3B": 0.05})

    # ISO가 높으면 장타 비율 증가 (제곱근으로 완화)
    iso = probs.get("iso", 0.150)
    iso_boost = (iso / 0.150) ** 0.5  # 제곱근: 2.6x → 1.6x

    # ISO 보정 적용
    p_1b = dist["1B"]
    p_2b = dist["2B"] * iso_boost
    p_3b = dist["3B"] * iso_boost

    total = p_1b + p_2b + p_3b
    p_1b /= total
    p_2b /= total
    p_3b /= total

    r = rng.random()
    if r < p_1b:
        return "1B"
    elif r < p_1b + p_2b:
        return "2B"
    else:
        return "3B"
