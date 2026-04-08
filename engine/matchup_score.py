"""V3 Matchup Scoring Engine — A투수 vs B팀 타선 개인별 매치업.

라인업 확정 시: 9명 개인별 구종 매칭
라인업 미확정 시: 25인 로스터 타자 전체 평균

구종 매치업(70%) + H2H 맞대결(30%).
"""

import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_DIR, SEASONS,
    V3_H2H_MAX_BONUS, V3_H2H_MIN_PA,
)

# ============================================================
# Module-level data cache
# ============================================================
_cache = {}


def _load_data():
    """Lazy-load matchup-related parquet files."""
    if _cache:
        return

    # 타자 구종 대처 (전 시즌)
    batter_frames = []
    for year in SEASONS:
        path = os.path.join(DATA_DIR, f"pitch_arsenal_batters_{year}.parquet")
        if os.path.exists(path):
            batter_frames.append(pd.read_parquet(path))
    _cache["batter_arsenal"] = pd.concat(batter_frames, ignore_index=True) if batter_frames else pd.DataFrame()

    # 투수 구종
    pitcher_frames = []
    for year in SEASONS:
        path = os.path.join(DATA_DIR, f"pitch_arsenal_pitchers_{year}.parquet")
        if os.path.exists(path):
            pitcher_frames.append(pd.read_parquet(path))
    _cache["pitcher_arsenal"] = pd.concat(pitcher_frames, ignore_index=True) if pitcher_frames else pd.DataFrame()

    # H2H 맞대결 (전 시즌)
    h2h_frames = []
    for year in SEASONS:
        path = os.path.join(DATA_DIR, f"h2h_matchups_{year}.parquet")
        if os.path.exists(path):
            h2h_frames.append(pd.read_parquet(path))
    _cache["h2h"] = pd.concat(h2h_frames, ignore_index=True) if h2h_frames else pd.DataFrame()

    # 타자 시즌 성적 (팀 로스터 조회용)
    batters_path = os.path.join(DATA_DIR, "batters_processed.parquet")
    if os.path.exists(batters_path):
        _cache["batters"] = pd.read_parquet(batters_path)
    else:
        _cache["batters"] = pd.DataFrame()

    # 리그 평균 구종별 xwOBA
    if not _cache["pitcher_arsenal"].empty:
        pa = _cache["pitcher_arsenal"]
        league_pitch = pa.groupby("pitch_type").apply(
            lambda g: np.average(g["xwoba"], weights=g["n_pitches"])
            if g["n_pitches"].sum() > 0 else 0.320
        ).to_dict()
        _cache["league_pitch_xwoba"] = league_pitch
    else:
        _cache["league_pitch_xwoba"] = {}

    # 투수 vs 팀 통산 전적
    pvt_path = os.path.join(DATA_DIR, "pitcher_vs_team.parquet")
    if os.path.exists(pvt_path):
        _cache["pitcher_vs_team"] = pd.read_parquet(pvt_path)
    else:
        _cache["pitcher_vs_team"] = pd.DataFrame()

    # 초구 데이터
    pfp_path = os.path.join(DATA_DIR, "pitcher_first_pitch.parquet")
    bfp_path = os.path.join(DATA_DIR, "batter_first_pitch.parquet")
    if os.path.exists(pfp_path):
        pfp = pd.read_parquet(pfp_path)
        _cache["pitcher_fp"] = dict(zip(pfp["pitcher"], pfp["fp_strike_rate"]))
        _cache["lg_fp_strike"] = pfp["fp_strike_rate"].mean()
    else:
        _cache["pitcher_fp"] = {}
        _cache["lg_fp_strike"] = 0.596

    if os.path.exists(bfp_path):
        bfp = pd.read_parquet(bfp_path)
        _cache["batter_fp"] = dict(zip(bfp["batter"], bfp["fp_swing_rate"]))
        _cache["lg_fp_swing"] = bfp["fp_swing_rate"].mean()
    else:
        _cache["batter_fp"] = {}
        _cache["lg_fp_swing"] = 0.322


# ============================================================
# 타선 결정: 확정 라인업 or 로스터 전체
# ============================================================
def _resolve_lineup(team: str, lineup_ids: list[int] = None) -> tuple[list[int], str]:
    """타선 ID 리스트 + 소스 반환.

    Args:
        team: 팀 코드
        lineup_ids: 확정 라인업 ID (None이면 로스터)

    Returns:
        (batter_id_list, source_str)
        source_str: "lineup" (확정 9명) or "roster" (전체 로스터)
    """
    _load_data()
    batters = _cache["batters"]

    if lineup_ids and len(lineup_ids) >= 8:
        return lineup_ids, "lineup"

    # 로스터: 최신 시즌 해당 팀 타자 (PA 상위)
    s = max(SEASONS)
    team_batters = batters[(batters["team"] == team) & (batters["season"] == s)]
    if team_batters.empty:
        team_batters = batters[batters["team"] == team]

    if team_batters.empty:
        return [], "roster"

    # PA 기준 정렬 (상위 타자 = 실제 주전)
    roster = team_batters.sort_values("pa", ascending=False)["player_id"].unique().tolist()
    return roster, "roster"


# ============================================================
# 개인별 타자-구종 매칭
# ============================================================
def _get_batter_pitch_profile(batter_id: int) -> dict:
    """타자 1명의 구종별 xwOBA 프로필."""
    _load_data()
    ba = _cache["batter_arsenal"]
    if ba.empty:
        return {}

    bdata = ba[ba["batter_id"] == batter_id]
    if bdata.empty:
        return {}

    # 최신 시즌만
    latest = bdata["season"].max()
    bdata = bdata[bdata["season"] == latest]

    return {
        row["pitch_type"]: {
            "xwoba": row["xwoba"],
            "whiff_rate": row["whiff_rate"],
            "n_pa": row["n_pa"],
        }
        for _, row in bdata.iterrows()
    }


def compute_individual_matchups(
    pitcher_id: int,
    batter_ids: list[int],
) -> tuple[float, list[dict], list[dict]]:
    """투수 구종 vs 개인별 타자 매칭.

    각 타자에 대해 투수의 주요 구종별 advantage 계산.

    Returns:
        (score 0-100, pitch_summary, batter_details)
    """
    _load_data()
    pa = _cache["pitcher_arsenal"]
    league_xwoba = _cache.get("league_pitch_xwoba", {})

    if pa.empty or not batter_ids:
        return 50.0, [], []

    # 투수의 최신 시즌 구종
    pitcher_data = pa[pa["pitcher_id"] == pitcher_id]
    if pitcher_data.empty:
        return 50.0, [], []

    latest = pitcher_data["season"].max()
    pitcher_data = pitcher_data[pitcher_data["season"] == latest]

    total_pitches = pitcher_data["n_pitches"].sum()
    if total_pitches == 0:
        return 50.0, [], []

    # 투수 구종 정보
    pitcher_pitches = []
    for _, row in pitcher_data.iterrows():
        pitcher_pitches.append({
            "pitch_type": row["pitch_type"],
            "usage": row["n_pitches"] / total_pitches,
            "pitcher_xwoba": row["xwoba"],
            "n_pitches": int(row["n_pitches"]),
        })
    pitcher_pitches.sort(key=lambda x: x["usage"], reverse=True)

    # 각 타자별 매칭
    batters_df = _cache["batters"]
    batter_details = []
    all_advantages = []  # (advantage, weight) per batter

    for bid in batter_ids:
        profile = _get_batter_pitch_profile(bid)

        # 타자 이름 조회
        brow = batters_df[batters_df["player_id"] == bid]
        batter_name = brow.iloc[-1]["name"] if not brow.empty else str(bid)
        batter_pa = int(brow.iloc[-1]["pa"]) if not brow.empty else 0

        batter_matchup = []
        batter_total_adv = 0.0

        for pp in pitcher_pitches:
            pt = pp["pitch_type"]
            lg_xwoba = league_xwoba.get(pt, 0.320)
            pitcher_xwoba = pp["pitcher_xwoba"]
            usage = pp["usage"]

            # 타자의 이 구종 xwOBA
            batter_pitch = profile.get(pt)
            if batter_pitch and batter_pitch["n_pa"] >= 5:
                batter_xwoba = batter_pitch["xwoba"]
            else:
                batter_xwoba = lg_xwoba  # 데이터 없으면 리그 평균

            # advantage: (리그평균 - 타자xwOBA) + (리그평균 - 투수xwOBA) / 2
            batter_weak = lg_xwoba - batter_xwoba  # 양수 = 타자가 이 구종에 약함
            pitcher_strong = lg_xwoba - pitcher_xwoba  # 양수 = 투수가 이 구종 잘 던짐
            adv = (batter_weak + pitcher_strong) / 2

            batter_total_adv += adv * usage

            batter_matchup.append({
                "pitch_type": pt,
                "usage": round(usage, 3),
                "batter_xwoba": round(batter_xwoba, 3),
                "pitcher_xwoba": round(pitcher_xwoba, 3),
                "advantage": round(adv, 4),
            })

        # PA 기반 가중 (주전일수록 비중 높음)
        weight = max(batter_pa, 50)  # 최소 50으로 하한
        all_advantages.append((batter_total_adv, weight))

        batter_details.append({
            "batter_id": bid,
            "name": batter_name,
            "pa": batter_pa,
            "total_advantage": round(batter_total_adv, 4),
            "pitches": batter_matchup,
        })

    # PA 가중 평균 advantage
    if all_advantages:
        total_w = sum(w for _, w in all_advantages)
        avg_advantage = sum(a * w for a, w in all_advantages) / total_w if total_w > 0 else 0
    else:
        avg_advantage = 0

    # 0-100 스케일 변환
    score = 50.0 + avg_advantage * 1000
    score = np.clip(score, 0, 100)

    # 구종별 요약 (기존 pitch_detail 호환)
    pitch_summary = []
    for pp in pitcher_pitches:
        pt = pp["pitch_type"]
        lg_xwoba = league_xwoba.get(pt, 0.320)

        # 전체 타선의 이 구종 평균 xwOBA
        batter_xwobas = []
        batter_weights = []
        for bd in batter_details:
            for pm in bd["pitches"]:
                if pm["pitch_type"] == pt:
                    batter_xwobas.append(pm["batter_xwoba"])
                    batter_weights.append(max(bd["pa"], 50))
                    break

        if batter_xwobas:
            team_xwoba = np.average(batter_xwobas, weights=batter_weights)
        else:
            team_xwoba = lg_xwoba

        pitcher_xwoba = pp["pitcher_xwoba"]
        combined = ((lg_xwoba - team_xwoba) + (lg_xwoba - pitcher_xwoba)) / 2

        pitch_summary.append({
            "pitch_type": pt,
            "usage": round(pp["usage"], 3),
            "pitcher_xwoba": round(pitcher_xwoba, 3),
            "team_xwoba": round(team_xwoba, 3),
            "league_xwoba": round(lg_xwoba, 3),
            "advantage": round(combined, 4),
        })

    # 타자 details를 advantage 순 정렬 (투수에게 가장 유리한 타자 먼저)
    batter_details.sort(key=lambda x: x["total_advantage"], reverse=True)

    return round(score, 1), pitch_summary, batter_details


# ============================================================
# H2H 맞대결 보너스
# ============================================================
def compute_h2h_bonus(
    pitcher_id: int,
    opposing_team: str,
    lineup_ids: list[int] = None,
) -> tuple[float, dict]:
    """맞대결 기록 기반 보너스 (-10 ~ +10)."""
    _load_data()
    h2h = _cache["h2h"]
    batters = _cache["batters"]

    if h2h.empty:
        return 0.0, {"note": "no h2h data"}

    if lineup_ids is None:
        team_batters = batters[batters["team"] == opposing_team]
        lineup_ids = team_batters["player_id"].unique().tolist()

    if not lineup_ids:
        return 0.0, {"note": "no lineup data"}

    matchups = h2h[
        (h2h["pitcher_id"] == pitcher_id) &
        (h2h["batter_id"].isin(lineup_ids))
    ]

    total_pa = matchups["n_pa"].sum()
    if total_pa < V3_H2H_MIN_PA:
        return 0.0, {"note": f"insufficient h2h PA ({total_pa} < {V3_H2H_MIN_PA})"}

    w_k = np.average(matchups["k_rate"], weights=matchups["n_pa"])
    w_bb = np.average(matchups["bb_rate"], weights=matchups["n_pa"])
    w_hr = np.average(matchups["hr_rate"], weights=matchups["n_pa"])

    lg_k, lg_bb, lg_hr = 0.22, 0.08, 0.03
    k_diff = w_k - lg_k
    bb_diff = lg_bb - w_bb
    hr_diff = lg_hr - w_hr

    bonus = (k_diff * 30 + bb_diff * 30 + hr_diff * 50)
    bonus = np.clip(bonus, -V3_H2H_MAX_BONUS, V3_H2H_MAX_BONUS)

    confidence = min(total_pa / 50.0, 1.0)
    bonus *= confidence

    detail = {
        "total_pa": int(total_pa),
        "n_batters": matchups["batter_id"].nunique(),
        "w_k_rate": round(w_k, 3),
        "w_bb_rate": round(w_bb, 3),
        "w_hr_rate": round(w_hr, 3),
        "confidence": round(confidence, 2),
    }

    return round(bonus, 1), detail


# ============================================================
# 투수 vs 팀 통산 전적 보너스
# ============================================================
def compute_pitcher_vs_team_bonus(
    pitcher_id: int,
    opposing_team: str,
) -> tuple[float, dict]:
    """투수의 상대팀 통산 성적 기반 보너스 (-10 ~ +10).

    K%/BB%/HR%/AVG를 리그 평균과 비교.
    최소 30PA 이상이어야 반영.
    """
    _load_data()
    pvt = _cache.get("pitcher_vs_team")
    if pvt is None or pvt.empty:
        return 0.0, {"note": "no pitcher_vs_team data"}

    record = pvt[(pvt["pitcher"] == pitcher_id) & (pvt["opp_team"] == opposing_team)]
    if record.empty:
        return 0.0, {"note": "no career record vs team"}

    row = record.iloc[0]
    total_pa = int(row["n_pa"])
    if total_pa < 30:
        return 0.0, {"note": f"insufficient PA ({total_pa} < 30)"}

    # 리그 평균 대비
    lg_k, lg_bb, lg_hr, lg_avg = 0.22, 0.08, 0.03, 0.245

    k_diff = row["k_rate"] - lg_k          # 양수 = 삼진 잘 잡음
    bb_diff = lg_bb - row["bb_rate"]       # 양수 = 볼넷 적음
    hr_diff = lg_hr - row["hr_rate"]       # 양수 = 홈런 적음
    avg_diff = lg_avg - row["batting_avg"] # 양수 = 안타 적음

    bonus = (k_diff * 20 + bb_diff * 20 + hr_diff * 30 + avg_diff * 30)
    bonus = np.clip(bonus, -V3_H2H_MAX_BONUS, V3_H2H_MAX_BONUS)

    # PA 기반 신뢰도 (100PA에서 full confidence)
    confidence = min(total_pa / 100.0, 1.0)
    bonus *= confidence

    detail = {
        "n_pa": total_pa,
        "n_games": int(row["n_games"]),
        "k_rate": round(row["k_rate"], 3),
        "bb_rate": round(row["bb_rate"], 3),
        "hr_rate": round(row["hr_rate"], 3),
        "batting_avg": round(row["batting_avg"], 3),
        "confidence": round(confidence, 2),
    }

    return round(bonus, 1), detail


# ============================================================
# 초구 카운트 어드밴티지
# ============================================================
def compute_count_advantage(
    pitcher_id: int,
    opposing_team: str,
    lineup_ids: list[int] = None,
) -> tuple[float, dict]:
    """투수 초구 스트라이크율 vs 상대 타선 초구 스윙율 → 카운트 주도권.

    Returns:
        (advantage, detail)
        advantage: -3 ~ +3. 양수 = 투수가 카운트 주도.
    """
    _load_data()
    pfp_map = _cache.get("pitcher_fp", {})
    bfp_map = _cache.get("batter_fp", {})
    lg_strike = _cache.get("lg_fp_strike", 0.596)
    lg_swing = _cache.get("lg_fp_swing", 0.322)

    fp_strike = pfp_map.get(pitcher_id, lg_strike)

    # 상대 타선 평균 초구 스윙율
    if lineup_ids is None:
        batter_ids, _ = _resolve_lineup(opposing_team)
    else:
        batter_ids = lineup_ids

    if not batter_ids:
        return 0.0, {"note": "no lineup"}

    swings = [bfp_map.get(bid, lg_swing) for bid in batter_ids]
    team_swing = float(np.mean(swings))

    pitcher_edge = fp_strike - lg_strike  # +면 평균보다 초구 스트라이크 잘 던짐
    batter_aggr = team_swing - lg_swing   # +면 평균보다 적극적 스윙

    advantage = pitcher_edge * 15 + batter_aggr * 5
    advantage = float(np.clip(advantage, -3, 3))

    detail = {
        "fp_strike_rate": round(fp_strike, 3),
        "team_fp_swing_rate": round(team_swing, 3),
        "pitcher_edge": round(pitcher_edge, 3),
        "batter_aggr": round(batter_aggr, 3),
    }

    return round(advantage, 2), detail


# ============================================================
# 메인 매치업 스코어
# ============================================================
def score_matchup(
    pitcher_id: int,
    opposing_team: str,
    lineup_ids: list[int] = None,
) -> dict:
    """A투수 vs B팀 매치업 스코어 (개인별 매칭).

    라인업 확정 시: 9명 개인별 구종 매칭
    라인업 미확정 시: 로스터 전체 타자 평균

    Returns:
        {
            'matchup_score': float,
            'arsenal_matchup_score': float,
            'h2h_bonus': float,
            'lineup_source': 'lineup' | 'roster',
            'n_batters': int,
            'pitch_detail': [...],
            'batter_detail': [...],
            'h2h_detail': {...},
        }
    """
    # 1. 타선 결정
    batter_ids, lineup_source = _resolve_lineup(opposing_team, lineup_ids)

    # 2. 개인별 구종 매치업
    arsenal_score, pitch_detail, batter_detail = compute_individual_matchups(
        pitcher_id, batter_ids
    )

    # 3. H2H 보너스 (개별 타자)
    h2h_bonus, h2h_detail = compute_h2h_bonus(pitcher_id, opposing_team, batter_ids)

    # 4. 투수 vs 팀 통산 전적
    pvt_bonus, pvt_detail = compute_pitcher_vs_team_bonus(pitcher_id, opposing_team)

    # 5. 초구 카운트 어드밴티지
    count_adv, count_detail = compute_count_advantage(pitcher_id, opposing_team, batter_ids)

    # 6. 종합 (arsenal 60%, h2h 20%, pitcher_vs_team 20%)
    h2h_component = 50.0 + h2h_bonus * 2.5
    h2h_component = np.clip(h2h_component, 0, 100)

    pvt_component = 50.0 + pvt_bonus * 2.5
    pvt_component = np.clip(pvt_component, 0, 100)

    matchup_score = arsenal_score * 0.60 + h2h_component * 0.20 + pvt_component * 0.20
    matchup_score = np.clip(matchup_score, 0, 100)

    return {
        "matchup_score": round(matchup_score, 1),
        "arsenal_matchup_score": round(arsenal_score, 1),
        "h2h_bonus": h2h_bonus,
        "h2h_component": round(h2h_component, 1),
        "pvt_bonus": pvt_bonus,
        "pvt_component": round(pvt_component, 1),
        "count_advantage": count_adv,
        "lineup_source": lineup_source,
        "n_batters": len(batter_ids),
        "pitch_detail": pitch_detail,
        "batter_detail": batter_detail,
        "h2h_detail": h2h_detail,
        "pvt_detail": pvt_detail,
        "count_detail": count_detail,
    }


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    import sys

    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 669373
    team = sys.argv[2] if len(sys.argv) > 2 else "BOS"

    result = score_matchup(pid, team)
    print(f"\n=== V3 Matchup: Pitcher {pid} vs {team} ({result['lineup_source']}, {result['n_batters']} batters) ===")
    print(f"Matchup Score: {result['matchup_score']:.1f}")
    print(f"  Arsenal Matchup: {result['arsenal_matchup_score']:.1f}")
    print(f"  H2H Bonus: {result['h2h_bonus']:+.1f}")

    print(f"\n--- Pitch Summary ---")
    for p in result["pitch_detail"]:
        print(f"  {p['pitch_type']:4s} usage={p['usage']:.1%} "
              f"pitcher={p['pitcher_xwoba']:.3f} "
              f"team={p['team_xwoba']:.3f} "
              f"adv={p['advantage']:+.4f}")

    print(f"\n--- Batter Detail (top 5) ---")
    for b in result["batter_detail"][:5]:
        print(f"  {b['name']:20s} PA={b['pa']:>4d} adv={b['total_advantage']:+.4f}")
