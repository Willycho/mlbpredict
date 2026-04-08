"""Shrinkage/dampening 파라미터 최적화 — grid search + cross-validation.

config.py의 SHRINKAGE k값들과 PROBABILITY_DAMPENING_ALPHA를
walk-forward backtest Brier score 최소화 기준으로 탐색.

사용법:
    python scripts/optimize_params.py --mode alpha    # alpha만 탐색
    python scripts/optimize_params.py --mode shrinkage # k값 탐색
    python scripts/optimize_params.py --mode full     # 전체 탐색 (느림)
"""

import sys, os, json, time, argparse, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR, V2_DATA_DIR, SHRINKAGE, PROBABILITY_DAMPENING_ALPHA
from engine.lineup import (
    get_team_lineup, get_league_avg,
    get_batter_pitch_arsenal, get_pitcher_pitch_arsenal, get_h2h_stats,
)
from engine.bayesian_matchup import compute_matchup_v2
from engine.cpp_bridge import (
    prepare_transition_arrays, prepare_matchup_cache,
    prepare_defense_buckets,
    SPEED_BUCKET_TO_IDX,
)
from engine.cpp.sim_core import run_simulation_cpp
from engine.calibration import brier_score


def load_backtest_games(start: str, end: str, max_n: int = 100) -> pd.DataFrame:
    """백테스트용 경기 로드 (실제 라인업 + 결과 포함)."""
    raw = pd.read_parquet(os.path.join(DATA_DIR, "statcast_raw_2025.parquet"))
    raw["game_date"] = pd.to_datetime(raw["game_date"])

    gr = (raw[raw["events"].notna()].sort_values(["game_pk", "inning"], ascending=[True, False])
          .groupby("game_pk").first().reset_index())[["game_pk", "game_date", "home_team", "away_team", "home_score", "away_score"]]
    starters = raw[raw["inning"] == 1].groupby(["game_pk", "inning_topbot"])["pitcher"].first().reset_index()
    gr = gr.merge(starters[starters["inning_topbot"] == "Top"].rename(columns={"pitcher": "hsp_id"})[["game_pk", "hsp_id"]], on="game_pk", how="left")
    gr = gr.merge(starters[starters["inning_topbot"] == "Bot"].rename(columns={"pitcher": "asp_id"})[["game_pk", "asp_id"]], on="game_pk", how="left")

    games = gr[(gr["game_date"] >= start) & (gr["game_date"] <= end)
               & gr["hsp_id"].notna() & gr["asp_id"].notna()]
    return games.head(max_n), raw


def run_with_params(games: pd.DataFrame, raw: pd.DataFrame, alpha: float,
                    shrinkage_overrides: dict = None, n_sims: int = 200) -> float:
    """특정 파라미터로 배치 시뮬 → Brier score 반환."""
    import config
    original_alpha = config.PROBABILITY_DAMPENING_ALPHA
    original_shrinkage = dict(config.SHRINKAGE)

    # 파라미터 오버라이드
    config.PROBABILITY_DAMPENING_ALPHA = alpha
    if shrinkage_overrides:
        config.SHRINKAGE.update(shrinkage_overrides)

    # 데이터 로드
    league_avg = get_league_avg(2025)
    pitchers_df = pd.read_parquet(os.path.join(DATA_DIR, "pitchers_processed.parquet"))
    batters_df = pd.read_parquet(os.path.join(DATA_DIR, "batters_processed.parquet"))
    speed_df = pd.read_parquet(os.path.join(V2_DATA_DIR, "speed_proxy.parquet"))
    defense_df = pd.read_parquet(os.path.join(V2_DATA_DIR, "team_defense.parquet"))
    bp_df = pd.read_parquet(os.path.join(V2_DATA_DIR, "bullpen_profiles.parquet"))
    tm_df = pd.read_parquet(os.path.join(V2_DATA_DIR, "transition_matrix.parquet"))
    ti, tr, tp, to_ = prepare_transition_arrays(tm_df)

    def get_ps(pid):
        m = pitchers_df[pitchers_df["player_id"] == pid]
        if len(m) == 0: return None
        r = m.iloc[-1]
        return {"player_id": int(pid), "name": str(r.get("name", "")),
                "k_rate": r["k_rate"], "bb_rate": r["bb_rate"],
                "hbp_rate": r["hbp_rate"], "hr_rate": r["hr_rate"],
                "babip": r["babip"], "throws": str(r.get("throws", "R")),
                "tbf": int(r.get("tbf", 100))}

    def fb_p():
        return {"player_id": "avg", "throws": "R", "tbf": 500,
                "k_rate": league_avg["k_rate"], "bb_rate": league_avg["bb_rate"],
                "hbp_rate": league_avg["hbp_rate"], "hr_rate": league_avg["hr_rate"],
                "babip": league_avg["babip"]}

    def get_batter(pid):
        m = batters_df[batters_df["player_id"] == pid]
        if len(m) == 0: return None
        r = m.iloc[-1]
        return {"player_id": int(pid), "bats": str(r.get("bats", "R")), "pa": int(r.get("pa", 0)),
                "k_rate": r["k_rate"], "bb_rate": r["bb_rate"],
                "hbp_rate": r["hbp_rate"], "hr_rate": r["hr_rate"],
                "babip": r["babip"], "iso": r.get("iso", 0.15),
                "ld_rate": r.get("ld_rate", 0.21), "gb_rate": r.get("gb_rate", 0.43),
                "fb_rate": r.get("fb_rate", 0.34), "iffb_rate": r.get("iffb_rate", 0.10),
                "gdp_rate": r.get("gdp_rate", 0.15)}

    def get_actual_lineup(game_pk, is_home):
        topbot = "Bot" if is_home else "Top"
        gd = raw[(raw["game_pk"] == game_pk) & (raw["inning_topbot"] == topbot)]
        if len(gd) == 0: return None
        pa = gd[gd["events"].notna()].sort_values("at_bat_number")
        ids = pa.drop_duplicates(subset=["batter"], keep="first")["batter"].tolist()[:9]
        return ids if len(ids) >= 9 else None

    def get_def_bucket(team):
        m = defense_df[defense_df["team"] == team]
        return m.iloc[-1]["defense_bucket"] if len(m) > 0 else "avg"

    def get_bp_dict(team):
        m = bp_df[bp_df["team"] == team]
        result = {}
        for _, row in m.iterrows():
            result[row["role"]] = {
                "player_id": f"bp_{team}_{row['role']}",
                "k_rate": row["k_rate"], "bb_rate": row["bb_rate"],
                "hr_rate": row.get("hr_rate", 0.03), "hbp_rate": row.get("hbp_rate", 0.01),
                "babip": row.get("babip", 0.300), "throws": "R",
            }
        for role in ["setup_early", "setup_late", "bridge", "closer"]:
            if role not in result:
                result[role] = dict(fb_p(), player_id=f"bp_{team}_{role}")
        return result

    preds = []
    actuals = []

    for _, g in games.iterrows():
        try:
            away, home = g["away_team"], g["home_team"]
            game_pk = int(g["game_pk"])

            away_ids = get_actual_lineup(game_pk, False)
            home_ids = get_actual_lineup(game_pk, True)

            al = [get_batter(pid) for pid in (away_ids or [])]
            al = [b for b in al if b is not None]
            if len(al) < 9:
                al = get_team_lineup(away, 2025)
            hl = [get_batter(pid) for pid in (home_ids or [])]
            hl = [b for b in hl if b is not None]
            if len(hl) < 9:
                hl = get_team_lineup(home, 2025)
            if len(al) < 9 or len(hl) < 9:
                continue

            asp = get_ps(g["asp_id"]) or fb_p()
            hsp = get_ps(g["hsp_id"]) or fb_p()

            away_bp = get_bp_dict(away)
            home_bp = get_bp_dict(home)
            away_def = get_def_bucket(away)
            home_def = get_def_bucket(home)

            mc = {}
            for lineup, pitcher, is_home in [(al, hsp, False), (hl, asp, True)]:
                for batter in lineup:
                    pid = batter.get("player_id")
                    if not pid: continue
                    probs = compute_matchup_v2(
                        batter=batter, pitcher=pitcher, league_avg=league_avg,
                        home_team=home, is_batter_home=is_home,
                        defense_bucket=home_def if not is_home else away_def,
                    )
                    mc[(pid, pitcher.get("player_id"))] = probs

            for lineup, bpd, is_home in [(al, home_bp, False), (hl, away_bp, True)]:
                for role, bp in bpd.items():
                    for batter in lineup:
                        pid = batter.get("player_id")
                        if not pid: continue
                        probs = compute_matchup_v2(
                            batter=batter, pitcher=bp, league_avg=league_avg,
                            home_team=home, is_batter_home=is_home,
                            defense_bucket=home_def if not is_home else away_def,
                        )
                        mc[(pid, bp.get("player_id"))] = probs

            mc_arr = prepare_matchup_cache(al, hl, mc, away_bp, home_bp)
            speeds = np.ones((2, 9), dtype=np.int32)
            for ti2, lineup in enumerate([al, hl]):
                for bi, b in enumerate(lineup[:9]):
                    pid = b.get("player_id")
                    if pid:
                        m = speed_df[speed_df["player_id"] == pid]
                        if len(m) > 0:
                            speeds[ti2, bi] = SPEED_BUCKET_TO_IDX.get(m.iloc[0]["speed_bucket"], 1)
            def_arr = prepare_defense_buckets(away_def, home_def)

            scores = run_simulation_cpp(
                mc_arr, None, ti, tr, tp, to_, speeds, def_arr,
                n_sims=n_sims, base_seed=game_pk,
            )

            aw = scores[:, 0]; hw = scores[:, 1]
            home_wins = (hw > aw).sum()
            away_wins = (aw > hw).sum()
            nt = home_wins + away_wins
            raw_hp = home_wins / nt if nt > 0 else 0.5
            hp = 0.50 + (raw_hp - 0.50) * alpha

            preds.append(hp)
            actuals.append(1 if g["home_score"] > g["away_score"] else 0)
        except Exception:
            continue

    # 복원
    config.PROBABILITY_DAMPENING_ALPHA = original_alpha
    config.SHRINKAGE = original_shrinkage

    if len(preds) < 10:
        return 1.0  # 데이터 부족

    return brier_score(np.array(preds), np.array(actuals))


def optimize_alpha(games, raw, alphas=None):
    """Alpha grid search."""
    if alphas is None:
        alphas = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]

    results = []
    for a in alphas:
        t0 = time.time()
        brier = run_with_params(games, raw, alpha=a)
        elapsed = time.time() - t0
        results.append({"alpha": a, "brier": round(brier, 6), "time": round(elapsed, 1)})
        print(f"  alpha={a:.2f}  Brier={brier:.6f}  ({elapsed:.1f}s)", flush=True)

    results.sort(key=lambda x: x["brier"])
    print(f"\nBest: alpha={results[0]['alpha']:.2f} Brier={results[0]['brier']:.6f}")
    return results


def optimize_shrinkage(games, raw, alpha=0.65):
    """주요 shrinkage k값 grid search."""
    base = dict(SHRINKAGE)

    # batter_rate, pitcher_rate, h2h가 가장 영향 큼
    batter_rates = [200, 300, 350, 400, 500]
    pitcher_rates = [150, 200, 250, 300, 400]

    results = []
    for br in batter_rates:
        for pr in pitcher_rates:
            overrides = {"batter_rate": br, "pitcher_rate": pr}
            t0 = time.time()
            brier = run_with_params(games, raw, alpha=alpha, shrinkage_overrides=overrides)
            elapsed = time.time() - t0
            results.append({
                "batter_rate": br, "pitcher_rate": pr,
                "brier": round(brier, 6), "time": round(elapsed, 1),
            })
            print(f"  b_rate={br} p_rate={pr}  Brier={brier:.6f}  ({elapsed:.1f}s)", flush=True)

    results.sort(key=lambda x: x["brier"])
    best = results[0]
    print(f"\nBest: batter_rate={best['batter_rate']} pitcher_rate={best['pitcher_rate']} "
          f"Brier={best['brier']:.6f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["alpha", "shrinkage", "full"], default="alpha")
    parser.add_argument("--start", default="2025-07-01")
    parser.add_argument("--end", default="2025-07-31")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading games {args.start} ~ {args.end} (max {args.n})...", flush=True)
    games, raw = load_backtest_games(args.start, args.end, args.n)
    print(f"Loaded {len(games)} games", flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "backtest", "results")
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "alpha":
        results = optimize_alpha(games, raw)
        with open(os.path.join(out_dir, "optimize_alpha.json"), "w") as f:
            json.dump(results, f, indent=2)

    elif args.mode == "shrinkage":
        results = optimize_shrinkage(games, raw)
        with open(os.path.join(out_dir, "optimize_shrinkage.json"), "w") as f:
            json.dump(results, f, indent=2)

    elif args.mode == "full":
        print("\n=== Phase 1: Alpha ===", flush=True)
        alpha_results = optimize_alpha(games, raw)
        best_alpha = alpha_results[0]["alpha"]

        print(f"\n=== Phase 2: Shrinkage (alpha={best_alpha}) ===", flush=True)
        shrinkage_results = optimize_shrinkage(games, raw, alpha=best_alpha)

        combined = {
            "best_alpha": best_alpha,
            "best_batter_rate": shrinkage_results[0]["batter_rate"],
            "best_pitcher_rate": shrinkage_results[0]["pitcher_rate"],
            "best_brier": shrinkage_results[0]["brier"],
            "alpha_results": alpha_results,
            "shrinkage_results": shrinkage_results,
        }
        with open(os.path.join(out_dir, "optimize_full.json"), "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nFinal: alpha={best_alpha} "
              f"batter_rate={shrinkage_results[0]['batter_rate']} "
              f"pitcher_rate={shrinkage_results[0]['pitcher_rate']} "
              f"Brier={shrinkage_results[0]['brier']:.6f}")


if __name__ == "__main__":
    main()
