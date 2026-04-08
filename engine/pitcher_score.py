"""V3 Pitcher Scoring Engine — 투수 1명을 0-100으로 스코어링.

퍼센타일 기반 정규화. 시즌 성적(40%) + 구종 가치(30%) + 최근 폼(30%).
"""

import os
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_DIR, SEASONS,
    V3_MIN_TBF, V3_MIN_ARSENAL_PA,
    V3_PITCHER_SCORE_WEIGHTS, V3_SEASON_STAT_WEIGHTS,
    V3_ARSENAL_WEIGHTS, V3_SEASON_RECENCY_WEIGHTS,
    V3_HOME_AWAY_MAX_ADJ, V3_TBF_SHRINKAGE_K,
)
from engine.recent_form import compute_recent_stats

# ============================================================
# Module-level data cache
# ============================================================
_cache = {}


def _load_data():
    """Lazy-load all required parquet files."""
    if _cache:
        return

    # 투수 시즌 성적
    _cache["pitchers"] = pd.read_parquet(
        os.path.join(DATA_DIR, "pitchers_processed.parquet")
    )

    # 리그 평균
    _cache["league_avg"] = pd.read_parquet(
        os.path.join(DATA_DIR, "league_averages.parquet")
    )

    # 구종 데이터 (전 시즌 concat)
    arsenal_frames = []
    for year in SEASONS:
        path = os.path.join(DATA_DIR, f"pitch_arsenal_pitchers_{year}.parquet")
        if os.path.exists(path):
            arsenal_frames.append(pd.read_parquet(path))
    _cache["arsenal"] = pd.concat(arsenal_frames, ignore_index=True) if arsenal_frames else pd.DataFrame()

    # 홈/원정 스플릿
    splits_path = os.path.join(DATA_DIR, "splits_pitcher_home_away.parquet")
    if os.path.exists(splits_path):
        _cache["splits_ha"] = pd.read_parquet(splits_path)
    else:
        _cache["splits_ha"] = pd.DataFrame()

    # Statcast raw (최근 시즌만, 최근 폼용)
    latest_season = max(SEASONS)
    raw_path = os.path.join(DATA_DIR, f"statcast_raw_{latest_season}.parquet")
    if os.path.exists(raw_path):
        raw = pd.read_parquet(raw_path, columns=[
            "game_date", "pitcher", "batter", "events",
        ])
        raw["game_date"] = pd.to_datetime(raw["game_date"])
        _cache["raw"] = raw
    else:
        _cache["raw"] = None


def _get_league_avg(season: int) -> dict:
    """특정 시즌 리그 평균 반환."""
    _load_data()
    la = _cache["league_avg"]
    row = la[la["season"] == season]
    if row.empty:
        row = la.iloc[-1:]  # fallback: 마지막 시즌
    return row.iloc[0].to_dict()


# ============================================================
# FIP 계산
# ============================================================
def compute_fip(k_rate: float, bb_rate: float, hbp_rate: float,
                hr_rate: float, tbf: int, fip_constant: float = 3.10) -> float:
    """Rate stats + TBF로 FIP 계산.

    FIP = ((13*HR + 3*(BB+HBP) - 2*K) / IP) + cFIP
    IP ≈ (TBF - BB - HBP - HR) / 3
    """
    hr = hr_rate * tbf
    bb = bb_rate * tbf
    hbp = hbp_rate * tbf
    k = k_rate * tbf

    outs = tbf - bb - hbp - hr  # 근사: BIP 결과 = 아웃 (HR 제외 BIP 중)
    ip = max(outs / 3.0, 1.0)

    fip = ((13 * hr + 3 * (bb + hbp) - 2 * k) / ip) + fip_constant
    return max(fip, 0.0)


# ============================================================
# 퍼센타일 랭킹
# ============================================================
def percentile_rank(value: float, pool: np.ndarray, invert: bool = False) -> float:
    """풀 내 퍼센타일 (0-100). invert=True면 낮은 값이 높은 점수."""
    if len(pool) == 0:
        return 50.0
    if invert:
        return float(sp_stats.percentileofscore(pool, value, kind="rank"))
    else:
        return float(sp_stats.percentileofscore(pool, value, kind="rank"))


def _percentile_inverted(value: float, pool: np.ndarray) -> float:
    """낮을수록 좋은 스탯용 퍼센타일 (ERA, BB%, HR%, FIP)."""
    if len(pool) == 0:
        return 50.0
    # 낮은 값 = 높은 퍼센타일: 100 - percentile
    return 100.0 - float(sp_stats.percentileofscore(pool, value, kind="rank"))


# ============================================================
# 투수 풀 구성 (시즌 가중치 적용)
# ============================================================
def _build_pitcher_pool() -> pd.DataFrame:
    """V3 스코어링용 투수 풀 구성.

    전 시즌 데이터를 가중 병합. 최신 시즌 60%, 전년 30%, 2년전 10%.
    동일 투수가 여러 시즌 있으면 가중 평균.
    """
    _load_data()
    pitchers = _cache["pitchers"].copy()
    pitchers = pitchers[pitchers["tbf"] >= V3_MIN_TBF]

    # 시즌 가중치 추가
    pitchers["season_weight"] = pitchers["season"].map(V3_SEASON_RECENCY_WEIGHTS).fillna(0.0)
    pitchers = pitchers[pitchers["season_weight"] > 0]

    if pitchers.empty:
        return pitchers

    # FIP 계산
    pitchers["computed_fip"] = pitchers.apply(
        lambda r: compute_fip(r["k_rate"], r["bb_rate"], r["hbp_rate"],
                              r["hr_rate"], r["tbf"]),
        axis=1
    )

    # K-BB%
    pitchers["k_bb_pct"] = pitchers["k_rate"] - pitchers["bb_rate"]

    # 선발 투수만 (gs > 0) — 릴리버는 별도 평가 필요
    # 일단 전체 포함 (선발 비중 높은 투수가 자연스럽게 높은 TBF)
    return pitchers


def _aggregate_pitcher_seasons(pool: pd.DataFrame, pitcher_id: int) -> dict | None:
    """특정 투수의 다시즌 가중 평균."""
    rows = pool[pool["player_id"] == pitcher_id]
    if rows.empty:
        return None

    stats = ["k_rate", "bb_rate", "hbp_rate", "hr_rate", "babip", "era",
             "computed_fip", "k_bb_pct"]

    total_weight = (rows["season_weight"] * rows["tbf"]).sum()
    if total_weight == 0:
        return None

    result = {}
    for stat in stats:
        if stat not in rows.columns:
            continue
        result[stat] = (rows[stat] * rows["season_weight"] * rows["tbf"]).sum() / total_weight

    result["tbf"] = rows["tbf"].sum()
    result["player_id"] = pitcher_id
    result["name"] = rows.iloc[-1]["name"]  # 최신 시즌 이름
    result["team"] = rows.iloc[-1]["team"]
    result["throws"] = rows.iloc[-1]["throws"]

    return result


# ============================================================
# 풀 집계 캐시 (한번만 계산)
# ============================================================
def _build_aggregated_pool() -> pd.DataFrame:
    """전체 투수 풀의 다시즌 가중 평균 (캐시됨)."""
    if "agg_pool" in _cache:
        return _cache["agg_pool"]

    pool = _build_pitcher_pool()
    if pool.empty:
        _cache["agg_pool"] = pd.DataFrame()
        return _cache["agg_pool"]

    aggs = []
    for pid in pool["player_id"].unique():
        agg = _aggregate_pitcher_seasons(pool, pid)
        if agg:
            aggs.append(agg)

    _cache["agg_pool"] = pd.DataFrame(aggs) if aggs else pd.DataFrame()
    return _cache["agg_pool"]


# ============================================================
# Sub-score 1: 시즌 성적 (40%)
# ============================================================
def score_season_stats(pitcher_agg: dict, pool_df: pd.DataFrame) -> tuple[float, dict]:
    """시즌 성적 서브스코어 (0-100).

    각 스탯의 퍼센타일을 가중 평균.
    Returns: (score, breakdown_dict)
    """
    pool_stats = _build_aggregated_pool()

    if pool_stats.empty:
        return 50.0, {}

    breakdown = {}
    weighted_score = 0.0
    total_weight = 0.0

    for stat, weight in V3_SEASON_STAT_WEIGHTS.items():
        if stat not in pool_stats.columns or stat not in pitcher_agg:
            continue

        pool_values = pool_stats[stat].dropna().values
        pitcher_val = pitcher_agg[stat]

        # ERA, BB%, HR%, FIP는 낮을수록 좋음
        if stat in ("bb_rate", "hr_rate", "era", "computed_fip"):
            pctile = _percentile_inverted(pitcher_val, pool_values)
        else:
            pctile = percentile_rank(pitcher_val, pool_values)

        breakdown[stat] = round(pctile, 1)
        weighted_score += pctile * weight
        total_weight += weight

    if total_weight > 0:
        score = weighted_score / total_weight
    else:
        score = 50.0

    return round(score, 1), breakdown


# ============================================================
# Sub-score 2: 구종 가치 (30%)
# ============================================================
def _compute_pitcher_arsenal_metrics(pitcher_id: int) -> dict | None:
    """투수의 구종별 usage-weighted 집계."""
    _load_data()
    arsenal = _cache["arsenal"]
    if arsenal.empty:
        return None

    # 최신 시즌 우선
    pa = arsenal[arsenal["pitcher_id"] == pitcher_id].copy()
    if pa.empty:
        return None

    # 가장 최근 시즌 데이터 사용
    latest = pa["season"].max()
    pa = pa[pa["season"] == latest]

    # 최소 PA 필터
    pa = pa[pa["n_pa"] >= 5]
    if pa.empty:
        return None

    total_pitches = pa["n_pitches"].sum()
    if total_pitches == 0:
        return None

    # usage 재계산 (해당 시즌 내 비율)
    pa = pa.copy()
    pa["usage"] = pa["n_pitches"] / total_pitches

    # usage-weighted 평균
    w_whiff = (pa["whiff_rate"] * pa["usage"]).sum()
    w_xwoba = (pa["xwoba"] * pa["usage"]).sum()

    # 구종별 상세
    pitches = []
    for _, row in pa.iterrows():
        pitches.append({
            "pitch_type": row["pitch_type"],
            "usage": round(row["usage"], 3),
            "whiff_rate": round(row["whiff_rate"], 3),
            "xwoba": round(row["xwoba"], 3),
            "k_rate": round(row.get("k_rate", 0), 3),
            "n_pitches": int(row["n_pitches"]),
        })

    return {
        "weighted_whiff": w_whiff,
        "weighted_xwoba": w_xwoba,
        "pitches": sorted(pitches, key=lambda x: x["usage"], reverse=True),
    }


def _build_arsenal_pool() -> pd.DataFrame:
    """전 투수의 arsenal 집계 (퍼센타일 풀용, 캐시됨)."""
    if "arsenal_pool" in _cache:
        return _cache["arsenal_pool"]

    _load_data()
    arsenal = _cache["arsenal"]
    if arsenal.empty:
        return pd.DataFrame()

    # 투수별 최신 시즌의 usage-weighted 집계
    rows = []
    for pid in arsenal["pitcher_id"].unique():
        pa = arsenal[arsenal["pitcher_id"] == pid]
        latest = pa["season"].max()
        pa = pa[(pa["season"] == latest) & (pa["n_pa"] >= 5)]
        total = pa["n_pitches"].sum()
        if total == 0:
            continue
        usage = pa["n_pitches"] / total
        rows.append({
            "pitcher_id": pid,
            "weighted_whiff": (pa["whiff_rate"] * usage).sum(),
            "weighted_xwoba": (pa["xwoba"] * usage).sum(),
        })

    result = pd.DataFrame(rows) if rows else pd.DataFrame()
    _cache["arsenal_pool"] = result
    return result


def score_arsenal(pitcher_id: int) -> tuple[float, dict]:
    """구종 가치 서브스코어 (0-100).

    stuff (whiff_rate 기반) 50% + value (xwOBA 역전) 50%.
    Returns: (score, detail_dict)
    """
    metrics = _compute_pitcher_arsenal_metrics(pitcher_id)
    if metrics is None:
        return 50.0, {"pitches": [], "note": "arsenal data not available"}

    pool = _build_arsenal_pool()
    if pool.empty:
        return 50.0, {"pitches": metrics["pitches"]}

    # Stuff score: whiff_rate 퍼센타일 (높을수록 좋음)
    stuff_pctile = percentile_rank(
        metrics["weighted_whiff"],
        pool["weighted_whiff"].values
    )

    # Value score: xwOBA 역전 퍼센타일 (낮을수록 좋음)
    value_pctile = _percentile_inverted(
        metrics["weighted_xwoba"],
        pool["weighted_xwoba"].values
    )

    score = (stuff_pctile * V3_ARSENAL_WEIGHTS["stuff"]
             + value_pctile * V3_ARSENAL_WEIGHTS["value"])

    detail = {
        "stuff_score": round(stuff_pctile, 1),
        "value_score": round(value_pctile, 1),
        "weighted_whiff": round(metrics["weighted_whiff"], 3),
        "weighted_xwoba": round(metrics["weighted_xwoba"], 3),
        "pitches": metrics["pitches"],
    }

    return round(score, 1), detail


# ============================================================
# Sub-score 3: 최근 폼 (30%)
# ============================================================
def score_recent_form(pitcher_id: int, cutoff_date: str,
                      pool_df: pd.DataFrame) -> tuple[float | None, dict]:
    """최근 폼 서브스코어 (0-100).

    14일/30일 rolling window에서 K%, BB%, HR% 추출 → 퍼센타일.
    데이터 부족 시 None 반환 (가중치를 season+arsenal이 흡수).
    """
    _load_data()
    raw = _cache.get("raw")
    if raw is None or cutoff_date is None:
        return None, {"note": "no raw data or cutoff_date"}

    r14 = compute_recent_stats(raw, pitcher_id, cutoff_date, 14, role="pitcher", min_pa=8)
    r30 = compute_recent_stats(raw, pitcher_id, cutoff_date, 30, role="pitcher", min_pa=15)

    if r14 is None and r30 is None:
        return None, {"note": "insufficient recent data"}

    # 14일과 30일 블렌딩 (14일 가중치 60%, 30일 40%)
    if r14 and r30:
        k_rate = r14["k_rate"] * 0.6 + r30["k_rate"] * 0.4
        bb_rate = r14["bb_rate"] * 0.6 + r30["bb_rate"] * 0.4
        hr_rate = r14["hr_rate"] * 0.6 + r30["hr_rate"] * 0.4
    elif r14:
        k_rate, bb_rate, hr_rate = r14["k_rate"], r14["bb_rate"], r14["hr_rate"]
    else:
        k_rate, bb_rate, hr_rate = r30["k_rate"], r30["bb_rate"], r30["hr_rate"]

    # 풀에서 퍼센타일 계산 (캐시된 집계 풀 사용)
    pool_s = _build_aggregated_pool()

    if pool_s.empty:
        return None, {"note": "empty pool"}

    k_pctile = percentile_rank(k_rate, pool_s["k_rate"].values)
    bb_pctile = _percentile_inverted(bb_rate, pool_s["bb_rate"].values)
    hr_pctile = _percentile_inverted(hr_rate, pool_s["hr_rate"].values)

    # K% 40%, BB% 30%, HR% 30%
    score = k_pctile * 0.4 + bb_pctile * 0.3 + hr_pctile * 0.3

    detail = {
        "k_rate": round(k_rate, 3),
        "bb_rate": round(bb_rate, 3),
        "hr_rate": round(hr_rate, 3),
        "k_pctile": round(k_pctile, 1),
        "bb_pctile": round(bb_pctile, 1),
        "hr_pctile": round(hr_pctile, 1),
        "14d_pa": r14["pa"] if r14 else 0,
        "30d_pa": r30["pa"] if r30 else 0,
    }

    return round(score, 1), detail


# ============================================================
# 홈/원정 조정
# ============================================================
def _home_away_adjustment(pitcher_id: int, is_home: bool) -> float:
    """홈/원정 스플릿 기반 조정 (-5 ~ +5)."""
    _load_data()
    splits = _cache["splits_ha"]
    if splits.empty:
        return 0.0

    rows = splits[splits["pitcher"] == pitcher_id]
    if len(rows) < 2:
        return 0.0

    home_row = rows[rows["pitcher_home_away"] == "home"]
    away_row = rows[rows["pitcher_home_away"] == "away"]

    if home_row.empty or away_row.empty:
        return 0.0

    home_row = home_row.iloc[0]
    away_row = away_row.iloc[0]

    # K%-BB% 차이로 판단
    home_kbb = home_row["k_rate"] - home_row["bb_rate"]
    away_kbb = away_row["k_rate"] - away_row["bb_rate"]
    diff = home_kbb - away_kbb  # 양수 = 홈이 더 좋음

    # 최소 PA 확인
    if home_row["n_pa"] < 30 or away_row["n_pa"] < 30:
        diff *= 0.5  # 소표본이면 절반 반영

    # diff를 점수로 변환 (K-BB% 0.05 차이 ≈ 5점)
    adjustment = diff * 100  # 0.05 → 5
    adjustment = np.clip(adjustment, -V3_HOME_AWAY_MAX_ADJ, V3_HOME_AWAY_MAX_ADJ)

    # 홈이면 홈 보너스, 원정이면 반전
    if not is_home:
        adjustment = -adjustment

    return round(adjustment, 1)


# ============================================================
# 메인 스코어링 함수
# ============================================================
def score_pitcher(
    pitcher_id: int,
    cutoff_date: str = None,
    is_home: bool = None,
) -> dict:
    """투수 종합 스코어 (0-100).

    Args:
        pitcher_id: MLB 선수 ID
        cutoff_date: 최근 폼 기준 날짜 (None이면 최근 폼 생략)
        is_home: 홈 여부 (None이면 홈/원정 조정 생략)

    Returns:
        {
            'pitcher_id': int,
            'name': str,
            'team': str,
            'throws': str,
            'total_score': float (0-100),
            'season_score': float,
            'arsenal_score': float,
            'recent_form_score': float | None,
            'home_away_adj': float,
            'breakdown': {season: {...}, arsenal: {...}, recent: {...}},
        }
    """
    pool = _build_pitcher_pool()
    if pool.empty:
        return _empty_result(pitcher_id)

    # 투수 다시즌 가중 평균
    pitcher_agg = _aggregate_pitcher_seasons(pool, pitcher_id)
    if pitcher_agg is None:
        return _empty_result(pitcher_id)

    # Sub-scores
    season_score, season_detail = score_season_stats(pitcher_agg, pool)
    arsenal_score, arsenal_detail = score_arsenal(pitcher_id)
    recent_score, recent_detail = score_recent_form(pitcher_id, cutoff_date, pool)

    # 가중치 결정
    weights = dict(V3_PITCHER_SCORE_WEIGHTS)
    if recent_score is None:
        # 최근 폼 없으면 나머지가 흡수
        total_remaining = weights["season"] + weights["arsenal"]
        weights["season"] = weights["season"] / total_remaining
        weights["arsenal"] = weights["arsenal"] / total_remaining
        weights["recent_form"] = 0.0
        recent_component = 0.0
    else:
        recent_component = recent_score * weights["recent_form"]

    # 종합 점수
    total = (season_score * weights["season"]
             + arsenal_score * weights["arsenal"]
             + recent_component)

    # 홈/원정 조정
    ha_adj = 0.0
    if is_home is not None:
        ha_adj = _home_away_adjustment(pitcher_id, is_home)
        total = np.clip(total + ha_adj, 0, 100)

    return {
        "pitcher_id": pitcher_id,
        "name": pitcher_agg["name"],
        "team": pitcher_agg["team"],
        "throws": pitcher_agg["throws"],
        "total_score": round(total, 1),
        "season_score": round(season_score, 1),
        "arsenal_score": round(arsenal_score, 1),
        "recent_form_score": round(recent_score, 1) if recent_score is not None else None,
        "home_away_adj": ha_adj,
        "weights_used": {k: round(v, 2) for k, v in weights.items()},
        "breakdown": {
            "season": season_detail,
            "arsenal": arsenal_detail,
            "recent_form": recent_detail,
        },
    }


def _empty_result(pitcher_id: int) -> dict:
    return {
        "pitcher_id": pitcher_id,
        "name": "Unknown",
        "team": "N/A",
        "throws": "R",
        "total_score": 50.0,
        "season_score": 50.0,
        "arsenal_score": 50.0,
        "recent_form_score": None,
        "home_away_adj": 0.0,
        "no_data": True,
        "weights_used": {},
        "breakdown": {"season": {}, "arsenal": {}, "recent_form": {}},
    }


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    import sys

    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 669373  # Gerrit Cole default
    date = sys.argv[2] if len(sys.argv) > 2 else None

    result = score_pitcher(pid, cutoff_date=date, is_home=True)
    print(f"\n=== V3 Pitcher Score: {result['name']} ({result['team']}) ===")
    print(f"Total: {result['total_score']:.1f}")
    print(f"  Season:      {result['season_score']:.1f} (weight: {result['weights_used'].get('season', 0):.0%})")
    print(f"  Arsenal:     {result['arsenal_score']:.1f} (weight: {result['weights_used'].get('arsenal', 0):.0%})")
    rf = result['recent_form_score']
    print(f"  Recent Form: {f'{rf:.1f}' if rf else 'N/A'} (weight: {result['weights_used'].get('recent_form', 0):.0%})")
    print(f"  Home/Away:   {result['home_away_adj']:+.1f}")

    print(f"\n--- Season Breakdown ---")
    for stat, pctile in result["breakdown"]["season"].items():
        print(f"  {stat:15s}: {pctile:.1f}th percentile")

    print(f"\n--- Arsenal Detail ---")
    ad = result["breakdown"]["arsenal"]
    if "pitches" in ad:
        print(f"  Stuff: {ad.get('stuff_score', 'N/A')}  Value: {ad.get('value_score', 'N/A')}")
        for p in ad.get("pitches", []):
            print(f"  {p['pitch_type']:4s} usage={p['usage']:.1%} whiff={p['whiff_rate']:.1%} xwOBA={p['xwoba']:.3f}")
