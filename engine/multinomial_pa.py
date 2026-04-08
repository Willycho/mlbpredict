"""v2 2단계 Multinomial Plate Appearance Model.

v1 plate_appearance.py 대체. 핵심 변경:
1. Sequential threshold → Multinomial sampling
2. 8% error hack 제거 → FC/ROE 명시적 모델링
3. 2단계: Stage1(K/BB/HBP/HR/BIP) → Stage2(1B/2B/3B/GO/FO/LO/FC/ROE)
"""

import numpy as np


# Stage1 결과 목록
STAGE1_OUTCOMES = ["K", "BB", "HBP", "HR", "BIP"]

# Stage2 BIP 결과 목록
STAGE2_BIP_OUTCOMES = ["1B", "2B", "3B", "GO", "FO", "LO", "FC", "ROE"]


def resolve_pa(probs: dict, rng: np.random.Generator) -> str:
    """2단계 다범주 모델로 타석 결과 결정.

    Args:
        probs: bayesian_matchup.compute_matchup_v2()의 결과
               - stage1: {k_rate, bb_rate, hbp_rate, hr_rate, bip_rate}
               - stage2_bip: {1B, 2B, 3B, GO, FO, LO, FC, ROE}
        rng: numpy random generator

    Returns:
        결과 문자열: "K", "BB", "HBP", "HR", "1B", "2B", "3B",
                    "GO", "FO", "LO", "FC", "ROE"
    """
    stage1 = probs.get("stage1")
    if stage1 is None:
        # fallback: flat dict에서 stage1 구성
        stage1 = {
            "k_rate": probs.get("k_rate", 0.22),
            "bb_rate": probs.get("bb_rate", 0.08),
            "hbp_rate": probs.get("hbp_rate", 0.01),
            "hr_rate": probs.get("hr_rate", 0.03),
            "bip_rate": probs.get("bip_rate", 0.66),
        }

    # Stage 1: 5범주 다범주 샘플링
    p1 = np.array([
        stage1["k_rate"],
        stage1["bb_rate"],
        stage1["hbp_rate"],
        stage1["hr_rate"],
        stage1["bip_rate"],
    ])
    # 정규화 (부동소수점 오차 방지)
    p1 = np.maximum(p1, 0.0)
    p1_sum = p1.sum()
    if p1_sum <= 0:
        p1 = np.array([0.22, 0.08, 0.01, 0.03, 0.66])
        p1_sum = p1.sum()
    p1 = p1 / p1_sum

    outcome_idx = rng.choice(len(STAGE1_OUTCOMES), p=p1)
    outcome = STAGE1_OUTCOMES[outcome_idx]

    if outcome != "BIP":
        return outcome

    # Stage 2: BIP인 경우 8범주 다범주 샘플링
    stage2 = probs.get("stage2_bip")
    if stage2 is None:
        # fallback: 기본 BIP 분포
        stage2 = {
            "1B": 0.155, "2B": 0.045, "3B": 0.005,
            "GO": 0.420, "FO": 0.280, "LO": 0.070,
            "FC": 0.007, "ROE": 0.012,
        }

    p2 = np.array([stage2.get(o, 0.0) for o in STAGE2_BIP_OUTCOMES])
    p2 = np.maximum(p2, 0.0)
    p2_sum = p2.sum()
    if p2_sum <= 0:
        p2 = np.array([0.155, 0.045, 0.005, 0.420, 0.280, 0.070, 0.007, 0.012])
        p2_sum = p2.sum()
    p2 = p2 / p2_sum

    bip_idx = rng.choice(len(STAGE2_BIP_OUTCOMES), p=p2)
    return STAGE2_BIP_OUTCOMES[bip_idx]
