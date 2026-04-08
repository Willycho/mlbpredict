"""최근 폼 (Rolling Window) — 14일/30일 성적 반영.

Statcast raw 데이터에서 특정 날짜 기준 최근 N일 성적을 계산.
시즌 평균과 블렌딩하여 "핫/콜드 스트릭" 반영.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def compute_recent_stats(
    raw: pd.DataFrame,
    player_id: int,
    cutoff_date: str,
    window_days: int = 14,
    role: str = "batter",
    min_pa: int = 10,
) -> dict | None:
    """특정 선수의 최근 N일 성적 계산.

    Args:
        raw: Statcast raw pitches DataFrame
        player_id: 선수 ID
        cutoff_date: 기준 날짜 (미포함)
        window_days: 윈도우 크기 (일)
        role: "batter" or "pitcher"
        min_pa: 최소 PA (미만이면 None 반환)

    Returns:
        dict with k_rate, bb_rate, hr_rate, babip, pa 또는 None
    """
    cutoff = pd.Timestamp(cutoff_date)
    start = cutoff - timedelta(days=window_days)

    id_col = "batter" if role == "batter" else "pitcher"
    player_data = raw[
        (raw[id_col] == player_id)
        & (raw["game_date"] >= start)
        & (raw["game_date"] < cutoff)
        & (raw["events"].notna())
    ]

    pa = len(player_data)
    if pa < min_pa:
        return None

    events = player_data["events"]
    k = (events == "strikeout").sum() + (events == "strikeout_double_play").sum()
    bb = (events == "walk").sum() + (events == "intent_walk").sum()
    hbp = (events == "hit_by_pitch").sum()
    hr = (events == "home_run").sum()
    single = (events == "single").sum()
    double = (events == "double").sum()
    triple = (events == "triple").sum()
    h = single + double + triple + hr

    bip = pa - k - bb - hbp - hr
    babip = (h - hr) / bip if bip > 0 else 0.300

    return {
        "k_rate": k / pa,
        "bb_rate": bb / pa,
        "hbp_rate": hbp / pa,
        "hr_rate": hr / pa,
        "babip": babip,
        "pa": pa,
        "window_days": window_days,
    }


def blend_recent_form(
    season_stats: dict,
    recent_14d: dict | None,
    recent_30d: dict | None,
    weight_14d: float = 0.15,
    weight_30d: float = 0.10,
) -> dict:
    """시즌 스탯에 최근 폼 블렌딩.

    최종 = season * (1 - w14 - w30) + recent_14d * w14 + recent_30d * w30

    14일이 없으면 30일 weight를 흡수. 둘 다 없으면 시즌 그대로.
    """
    result = dict(season_stats)
    stat_keys = ["k_rate", "bb_rate", "hbp_rate", "hr_rate", "babip"]

    w14 = weight_14d if recent_14d else 0.0
    w30 = weight_30d if recent_30d else 0.0

    # 둘 다 없으면 그대로 반환
    if w14 == 0 and w30 == 0:
        return result

    w_season = 1.0 - w14 - w30

    for key in stat_keys:
        base = season_stats.get(key)
        if base is None:
            continue

        val = base * w_season
        if recent_14d and key in recent_14d:
            val += recent_14d[key] * w14
        else:
            val += base * w14  # fallback to season
        if recent_30d and key in recent_30d:
            val += recent_30d[key] * w30
        else:
            val += base * w30

        result[key] = val

    return result


def build_recent_form_cache(
    raw: pd.DataFrame,
    player_ids: list[int],
    cutoff_date: str,
    role: str = "batter",
) -> dict:
    """여러 선수의 최근 폼을 한번에 계산 (배치용).

    Returns:
        {player_id: {"14d": {...}, "30d": {...}}, ...}
    """
    # game_date 보장
    if not pd.api.types.is_datetime64_any_dtype(raw["game_date"]):
        raw = raw.copy()
        raw["game_date"] = pd.to_datetime(raw["game_date"])

    cache = {}
    for pid in player_ids:
        r14 = compute_recent_stats(raw, pid, cutoff_date, 14, role, min_pa=10)
        r30 = compute_recent_stats(raw, pid, cutoff_date, 30, role, min_pa=20)
        if r14 or r30:
            cache[pid] = {"14d": r14, "30d": r30}

    return cache
