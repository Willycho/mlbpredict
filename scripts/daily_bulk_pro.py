"""Daily bulk Pro prediction (V2 MC + V3 consensus).

Pro 버전은 V3(matchup) + V2(Monte Carlo) 컨센서스 픽을 생성.
백테스트 기준 V2 AGREE 된 픽의 적중률이 V3 단독보다 높음 (~78.9% sweet spot).

실행 시점: 매일 KST 21:00 (bulk_pro.yml cron)
Storage: data/storage/predictions_pro/{kst_date}/
"""
import sys
import os
import json
import argparse
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mlb_api import get_schedule
from data.odds_api import get_mlb_f5_and_team_totals
from engine.game_score import predict_all_games_v3
from config import TEAMS


STORAGE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "storage", "predictions_pro"
)


def get_target_kst_date() -> str:
    now_kst = datetime.now(timezone(timedelta(hours=9)))
    if now_kst.hour >= 18:
        target = now_kst + timedelta(days=1)
    else:
        target = now_kst
    return target.strftime("%Y-%m-%d")


def fetch_kst_schedule(kst_date: str) -> list[dict]:
    kst_dt = datetime.strptime(kst_date, "%Y-%m-%d")
    us_dates = [
        (kst_dt - timedelta(days=1)).strftime("%Y-%m-%d"),
        kst_dt.strftime("%Y-%m-%d"),
    ]
    schedule = []
    for d in us_dates:
        for g in get_schedule(d):
            if g.get("game_date_kst") == kst_date:
                schedule.append(g)
    return schedule


def build_consensus(v3_preds: list[dict], v2_preds: list[dict]) -> tuple[list[dict], list[dict]]:
    """V3 + V2 → 컨센서스 픽 병합.

    Returns:
        (all_games, top_picks)
    """
    v2_map = {p["game_id"]: p for p in v2_preds}
    all_games = []
    top_picks = []

    for v3 in v3_preds:
        gid = v3.get("game_id")
        v2 = v2_map.get(gid)
        game = dict(v3)

        if v2 and v3.get("pick"):
            game["v2_pick"] = v2.get("pick")
            game["v2_conf"] = round(v2.get("conf", 0.0), 3)
            game["consensus"] = (v2.get("pick") == v3["pick"])

            v3_conf = v3.get("pick_prob", 0.5)
            if game["consensus"]:
                if 0.65 <= v3_conf < 0.75:
                    game["tier"] = "sweet_spot"
                elif 0.60 <= v3_conf < 0.75:
                    game["tier"] = "good"
                else:
                    game["tier"] = "consensus"
            else:
                game["tier"] = "disagree"

            v2_f5_total = v2.get("avg_total")
            v3_expected = v3.get("expected_total")
            if v2_f5_total and v3_expected:
                v2_ou = "OVER" if v2_f5_total > 4.5 else "UNDER"
                v3_ou = "OVER" if v3_expected > 4.5 else "UNDER"
                game["ou_consensus"] = (v2_ou == v3_ou)
                game["v2_f5_total"] = round(v2_f5_total, 1)
            else:
                game["ou_consensus"] = None
                game["v2_f5_total"] = None
        else:
            game["v2_pick"] = None
            game["v2_conf"] = None
            game["consensus"] = None
            game["ou_consensus"] = None
            game["v2_f5_total"] = None
            game["tier"] = "no_v2"

        all_games.append(game)

        # 탑픽 수집: V3 sweet_picks 중 V2 AGREE 된 것만
        base_info = {
            "game_id": gid,
            "away_team": v3.get("away_team"),
            "home_team": v3.get("home_team"),
            "away_pitcher": (v3.get("away_pitcher") or {}).get("name", "TBD"),
            "home_pitcher": (v3.get("home_pitcher") or {}).get("name", "TBD"),
            "tier": game.get("tier", "no_v2"),
            "v2_pick": game.get("v2_pick"),
            "v2_conf": game.get("v2_conf"),
        }

        for sp in v3.get("sweet_picks", []):
            if sp["type"] == "TT":
                continue
            # ML: V2 ML과 방향 일치 필요
            if sp["type"] == "ML" and not game.get("consensus", False):
                continue
            # OU: V2 F5 total과 방향 일치 필요
            if sp["type"] == "OU" and not game.get("ou_consensus", False):
                continue
            pick_entry = {**base_info, **sp}
            if sp["type"] == "ML":
                pick_entry["ml_consensus"] = True
            elif sp["type"] == "OU":
                pick_entry["ou_consensus"] = True
            top_picks.append(pick_entry)

    top_picks.sort(key=lambda x: x.get("conf", 0), reverse=True)
    return all_games, top_picks


def save_pro_predictions(kst_date: str, all_games: list[dict], top_picks: list[dict], f5_lines_map: dict):
    date_dir = os.path.join(STORAGE_ROOT, kst_date)
    os.makedirs(date_dir, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()

    manifest_games = []
    for g in all_games:
        gid = g.get("game_id")
        if not gid:
            continue

        away_t = g.get("away_team", "")
        home_t = g.get("home_team", "")
        f5_line = f5_lines_map.get(f"{away_t}@{home_t}")

        game_file = {
            "game_id": gid,
            "kst_date": kst_date,
            "bulk_fetched_at": now_iso,
            "last_refreshed_at": now_iso,
            "refresh_type": "bulk_initial",
            "f5_book_line": f5_line,
            "prediction": g,
            "locked_away_pitcher_id": (g.get("away_pitcher") or {}).get("id"),
            "locked_home_pitcher_id": (g.get("home_pitcher") or {}).get("id"),
        }
        with open(os.path.join(date_dir, f"{gid}.json"), "w", encoding="utf-8") as f:
            json.dump(game_file, f, ensure_ascii=False, indent=2)

        manifest_games.append({
            "game_id": gid,
            "away_team": away_t,
            "home_team": home_t,
            "game_time_kst": g.get("game_time_kst", ""),
            "game_date_kst": g.get("game_date_kst", kst_date),
            "away_pitcher": (g.get("away_pitcher") or {}).get("name", "TBD"),
            "home_pitcher": (g.get("home_pitcher") or {}).get("name", "TBD"),
            "pick": g.get("pick"),
            "pick_prob": g.get("pick_prob"),
            "expected_total": g.get("expected_total"),
            "v2_pick": g.get("v2_pick"),
            "v2_conf": g.get("v2_conf"),
            "consensus": g.get("consensus"),
            "ou_consensus": g.get("ou_consensus"),
            "tier": g.get("tier"),
            "f5_book_line": f5_line,
            "lineup_source": (g.get("away_matchup") or {}).get("lineup_source", "roster"),
            "last_refreshed_at": now_iso,
            "refresh_type": "bulk_initial",
        })

    manifest = {
        "kst_date": kst_date,
        "generated_at": now_iso,
        "total_games": len(manifest_games),
        "games": sorted(manifest_games, key=lambda g: g.get("game_time_kst", "") or "zzz"),
        "top_picks": top_picks[:5],
        "stats": {
            "consensus_count": sum(1 for g in all_games if g.get("consensus")),
            "ou_consensus_count": sum(1 for g in all_games if g.get("ou_consensus")),
            "sweet_spot_count": sum(1 for g in all_games if g.get("tier") == "sweet_spot"),
            "total_games": len(all_games),
        },
    }
    with open(os.path.join(date_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[pro] saved {len(manifest_games)} games, {len(top_picks)} consensus top picks")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="Target KST date YYYY-MM-DD")
    parser.add_argument("--sims", type=int, default=500, help="V2 Monte Carlo sims per game")
    args = parser.parse_args()

    kst_date = args.date or get_target_kst_date()
    print(f"[pro] target KST date: {kst_date}")

    # 1. MLB 스케줄
    schedule = fetch_kst_schedule(kst_date)
    if not schedule:
        print(f"[pro] no games found for {kst_date}")
        return
    print(f"[pro] {len(schedule)} games found")

    # 2. F5 배당사 라인
    odds_data = get_mlb_f5_and_team_totals()
    name_to_code = {v: k for k, v in TEAMS.items()}
    f5_lines_map = {}
    for o in odds_data:
        aw = name_to_code.get(o["away_team"])
        hm = name_to_code.get(o["home_team"])
        if not (aw and hm):
            continue
        line = o.get("f5_total", {}).get("line")
        if line is not None:
            f5_lines_map[f"{aw}@{hm}"] = line
    print(f"[pro] F5 lines: {len(f5_lines_map)}/{len(schedule)}")

    # 3. F5 라인 주입
    for game in schedule:
        key = f"{game.get('away_team','')}@{game.get('home_team','')}"
        game["f5_total_line"] = f5_lines_map.get(key)

    # 4. V3 predict
    api_date = (datetime.strptime(kst_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    v3_preds = predict_all_games_v3(api_date, schedule, skip_odds_fetch=True)
    print(f"[pro] V3 predictions: {len(v3_preds)}")

    # 5. V2 F5 MC simulation
    # V2 predict_f5_games는 자체 스케줄 조회하지만 KST 날짜는 2개 날짜 병합 필요
    from web.predict import predict_f5_games
    v2_preds = []
    for d in [api_date, kst_date]:
        try:
            v2_preds.extend(predict_f5_games(d, n_sims=args.sims, force=True))
        except Exception as e:
            print(f"[pro] V2 predict failed for {d}: {e}")
    # dedupe by game_id
    seen = set()
    v2_preds_unique = []
    for p in v2_preds:
        gid = p.get("game_id")
        if gid and gid not in seen:
            seen.add(gid)
            v2_preds_unique.append(p)
    print(f"[pro] V2 predictions: {len(v2_preds_unique)}")

    # 6. 컨센서스 병합
    all_games, top_picks = build_consensus(v3_preds, v2_preds_unique)

    # 7. 저장
    save_pro_predictions(kst_date, all_games, top_picks, f5_lines_map)

    print(f"[pro] DONE")


if __name__ == "__main__":
    main()
