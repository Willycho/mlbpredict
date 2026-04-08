"""배당 edge 분석 v2 — odds bucket별 threshold + market divergence gate.

설계 원칙:
- 메인 확률(P_model)은 시뮬레이션 엔진이 독립 생성
- 시장 배당은 prior가 아닌 sanity check / veto gate로 사용
- 언더독일수록 edge threshold 높게 → 가짜 edge 방지
- 시장과 괴리가 크면 자동 추천 금지, 별도 로그로 원인 분석용
"""


# ============================================================
# Odds bucket 정의
# ============================================================

ODDS_BUCKETS = {
    # bucket_name: (min_ml, max_ml, min_edge, description)
    "heavy_fav":    (-999, -200, 0.08, "heavy fav (-200+): high threshold (overconf risk)"),
    "mild_fav":     (-200, -110, 0.05, "mild fav (-200 ~ -110)"),
    "pick":         (-110, +110, 0.06, "픽엠 구간 (-110 ~ +110)"),
    "mild_dog":     (+110, +200, 0.12, "mild underdog (+110~+200): elevated threshold"),
    "med_dog":      (+200, +300, 0.16, "mid underdog (+200~+300): high threshold"),
    "big_dog":      (+300, +500, 0.22, "big underdog (+300~+500): very high threshold"),
    "mega_dog":     (+500, +999, 999,  "초대형 언더독 (+500 이상) → 기본 PASS"),
}

# Market divergence 임계값
DIVERGENCE_THRESHOLDS = {
    "yellow": 0.15,   # 15%p 차이 → 주의
    "red": 0.20,      # 20%p 차이 → 추천 금지
    "extreme": 0.25,  # 25%p 차이 → 모델 오류 의심
}


# ============================================================
# 기본 계산 함수
# ============================================================

def american_to_implied(odds: int | float) -> float:
    """American odds → implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def american_to_decimal(odds: int | float) -> float:
    """American odds → decimal odds."""
    if odds > 0:
        return 1.0 + odds / 100.0
    else:
        return 1.0 + 100.0 / abs(odds)


def remove_vig(away_impl: float, home_impl: float) -> tuple[float, float]:
    """오버라운드 제거 → fair probability."""
    total = away_impl + home_impl
    if total == 0:
        return 0.5, 0.5
    return away_impl / total, home_impl / total


def calc_edge(model_prob: float, fair_prob: float) -> float:
    """edge = model_prob - fair_prob."""
    return model_prob - fair_prob


def calc_ev(model_prob: float, decimal_odds: float) -> float:
    """Expected Value per unit bet."""
    return model_prob * (decimal_odds - 1.0) - (1.0 - model_prob)


def kelly_fraction(model_prob: float, decimal_odds: float, fraction: float = 0.25) -> float:
    """Kelly criterion (fractional)."""
    if decimal_odds <= 1.0:
        return 0.0
    full = (model_prob * decimal_odds - 1.0) / (decimal_odds - 1.0)
    if full <= 0:
        return 0.0
    return full * fraction


# ============================================================
# Odds bucket 분류
# ============================================================

def classify_odds_bucket(ml: int | float) -> tuple[str, float]:
    """American ML → (bucket_name, min_edge_threshold).

    Returns:
        (bucket_name, required_min_edge)
    """
    for name, (lo, hi, min_edge, _desc) in ODDS_BUCKETS.items():
        if lo <= ml < hi:
            return name, min_edge
    # fallback
    if ml >= 500:
        return "mega_dog", 999
    return "pick", 0.06


# ============================================================
# Market divergence 감시
# ============================================================

def check_divergence(model_prob: float, fair_prob: float) -> dict:
    """모델 vs 시장 확률 괴리 수준 판단.

    Returns:
        {
            "gap": float,          # 절대값 차이
            "level": str,          # "ok" | "yellow" | "red" | "extreme"
            "veto": bool,          # True면 자동 추천 금지
            "reason": str,
        }
    """
    gap = abs(model_prob - fair_prob)

    if gap >= DIVERGENCE_THRESHOLDS["extreme"]:
        return {
            "gap": round(gap, 4),
            "level": "extreme",
            "veto": True,
            "reason": f"모델({model_prob:.0%}) vs 시장({fair_prob:.0%}) 차이 {gap:.0%} — 모델 오류 의심",
        }
    elif gap >= DIVERGENCE_THRESHOLDS["red"]:
        return {
            "gap": round(gap, 4),
            "level": "red",
            "veto": True,
            "reason": f"모델({model_prob:.0%}) vs 시장({fair_prob:.0%}) 차이 {gap:.0%} — 추천 금지",
        }
    elif gap >= DIVERGENCE_THRESHOLDS["yellow"]:
        return {
            "gap": round(gap, 4),
            "level": "yellow",
            "veto": False,
            "reason": f"모델({model_prob:.0%}) vs 시장({fair_prob:.0%}) 차이 {gap:.0%} — 주의",
        }
    else:
        return {
            "gap": round(gap, 4),
            "level": "ok",
            "veto": False,
            "reason": "",
        }


# ============================================================
# 등급 판정 (odds-bucket-aware)
# ============================================================

def grade_bet(edge: float, kelly: float, odds_bucket: str, divergence_veto: bool) -> str:
    """베팅 등급 — bucket별 threshold + divergence veto.

    mega_dog → 무조건 PASS
    divergence veto → 무조건 PASS (별도 FLAG 태그)
    나머지 → bucket별 min_edge 충족 시 등급 부여
    """
    if divergence_veto:
        return "FLAG"  # 추천 금지, 별도 분석 필요

    if odds_bucket == "mega_dog":
        return "PASS"

    # 중언더독 이상 + divergence yellow = PASS (모델 스프레드 한계)
    if odds_bucket in ("med_dog", "big_dog") and not divergence_veto:
        # divergence_veto가 False여도, 이 함수는 veto만 체크하므로
        # 별도로 edge threshold가 높게 잡혀 있어 자연스럽게 걸림
        pass

    _, min_edge = classify_odds_bucket(0)  # fallback
    for name, (_lo, _hi, me, _desc) in ODDS_BUCKETS.items():
        if name == odds_bucket:
            min_edge = me
            break

    if edge < min_edge:
        return "PASS"

    # min_edge를 통과한 경우, 강도에 따라 등급
    excess = edge - min_edge
    if excess <= 0:
        return "PASS"
    if excess >= 0.06 and kelly >= 0.03:
        return "A"
    elif excess >= 0.03 and kelly >= 0.02:
        return "B"
    elif excess > 0 and kelly >= 0.01:
        return "C"
    return "PASS"


# ============================================================
# 종합 분석
# ============================================================

def analyze_moneyline(model_home_prob: float, ml_home: int, ml_away: int,
                      kelly_frac: float = 0.25) -> dict:
    """경기 하나의 moneyline edge 전체 분석 (v2).

    Returns dict with:
        - 기본 edge/ev/kelly (양 사이드)
        - odds_bucket별 등급
        - divergence check
        - best_side / best_grade (veto 반영)
    """
    impl_home = american_to_implied(ml_home)
    impl_away = american_to_implied(ml_away)
    fair_away, fair_home = remove_vig(impl_away, impl_home)
    vig = impl_home + impl_away - 1.0

    model_away_prob = 1.0 - model_home_prob
    dec_home = american_to_decimal(ml_home)
    dec_away = american_to_decimal(ml_away)

    # Home side
    h_edge = calc_edge(model_home_prob, fair_home)
    h_ev = calc_ev(model_home_prob, dec_home)
    h_kelly = kelly_fraction(model_home_prob, dec_home, kelly_frac)
    h_bucket, _ = classify_odds_bucket(ml_home)
    h_div = check_divergence(model_home_prob, fair_home)
    h_grade = grade_bet(h_edge, h_kelly, h_bucket, h_div["veto"])

    # Away side
    a_edge = calc_edge(model_away_prob, fair_away)
    a_ev = calc_ev(model_away_prob, dec_away)
    a_kelly = kelly_fraction(model_away_prob, dec_away, kelly_frac)
    a_bucket, _ = classify_odds_bucket(ml_away)
    a_div = check_divergence(model_away_prob, fair_away)
    a_grade = grade_bet(a_edge, a_kelly, a_bucket, a_div["veto"])

    # Best side (edge 기준, 단 veto면 반대쪽 선택)
    if h_grade == "FLAG" and a_grade == "FLAG":
        best_side = "home" if h_edge >= a_edge else "away"
        best_edge = max(h_edge, a_edge)
        best_kelly = h_kelly if h_edge >= a_edge else a_kelly
        best_grade = "FLAG"
        best_ev = h_ev if h_edge >= a_edge else a_ev
    elif h_grade == "FLAG":
        best_side, best_edge, best_kelly, best_grade, best_ev = "away", a_edge, a_kelly, a_grade, a_ev
    elif a_grade == "FLAG":
        best_side, best_edge, best_kelly, best_grade, best_ev = "home", h_edge, h_kelly, h_grade, h_ev
    elif h_edge >= a_edge:
        best_side, best_edge, best_kelly, best_grade, best_ev = "home", h_edge, h_kelly, h_grade, h_ev
    else:
        best_side, best_edge, best_kelly, best_grade, best_ev = "away", a_edge, a_kelly, a_grade, a_ev

    # Strong confidence warning: model 65%+ on either side
    # 200-game backtest shows -18.7%p overconfidence in this zone
    HIGH_CONF_THRESHOLD = 0.62  # dampened space (raw ~0.68)
    if model_home_prob > HIGH_CONF_THRESHOLD or model_away_prob > HIGH_CONF_THRESHOLD:
        conf_side = "home" if model_home_prob > HIGH_CONF_THRESHOLD else "away"
        conf_prob = model_home_prob if conf_side == "home" else model_away_prob
        # Force yellow divergence at minimum for high-confidence picks
        if conf_side == "home" and h_div["level"] == "ok":
            h_div = {"gap": h_div["gap"], "level": "yellow", "veto": False,
                     "reason": f"HIGH CONF WARNING: model {conf_prob:.0%} on {conf_side} (overconf risk)"}
        elif conf_side == "away" and a_div["level"] == "ok":
            a_div = {"gap": a_div["gap"], "level": "yellow", "veto": False,
                     "reason": f"HIGH CONF WARNING: model {conf_prob:.0%} on {conf_side} (overconf risk)"}

    # Divergence summary
    div_level = max(h_div["level"], a_div["level"],
                    key=lambda x: ["ok", "yellow", "red", "extreme"].index(x))
    div_reasons = [r for r in [h_div["reason"], a_div["reason"]] if r]

    return {
        "fair_home": round(fair_home, 4),
        "fair_away": round(fair_away, 4),
        "vig": round(vig, 4),
        # Home
        "home_edge": round(h_edge, 4),
        "home_ev": round(h_ev, 4),
        "home_kelly": round(h_kelly, 4),
        "home_grade": h_grade,
        "home_bucket": h_bucket,
        # Away
        "away_edge": round(a_edge, 4),
        "away_ev": round(a_ev, 4),
        "away_kelly": round(a_kelly, 4),
        "away_grade": a_grade,
        "away_bucket": a_bucket,
        # Best
        "best_side": best_side,
        "best_edge": round(best_edge, 4),
        "best_kelly": round(best_kelly, 4),
        "best_grade": best_grade,
        "best_ev": round(best_ev, 4),
        # Divergence
        "divergence_level": div_level,
        "divergence_reasons": div_reasons,
    }
