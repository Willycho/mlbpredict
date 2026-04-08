"""V3 Pitcher-Centric Model — 웹 서비스용 래퍼.

V3 Free: 투수 매치업 F5 예측
V3 Pro: V2.3 MC 시뮬 + V3 컨센서스 (65-75% 스위트스팟)
"""
import sys, os, json, time
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mlb_api import get_schedule
from engine.game_score import predict_all_games_v3

PREDICTIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "predictions")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


def predict_games_v3(game_date: str, force: bool = False, kst: bool = False) -> list[dict]:
    """V3 전경기 예측 (캐시 포함).

    Args:
        game_date: "YYYY-MM-DD"
        force: 캐시 무시
        kst: True면 game_date를 KST 날짜로 간주하고, 해당 KST 날짜의 경기만 필터
    """
    cache_key = f"v3_kst_{game_date}" if kst else f"v3_{game_date}"
    cache_path = os.path.join(PREDICTIONS_DIR, f"{cache_key}.json")

    if not force and os.path.exists(cache_path):
        mtime = os.path.getmtime(cache_path)
        if time.time() - mtime < 3600:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

    if kst:
        # KST 날짜에 해당하는 미국 경기 조회 (전날 + 당일)
        from datetime import datetime, timedelta
        kst_date = datetime.strptime(game_date, "%Y-%m-%d")
        us_date = (kst_date - timedelta(days=1)).strftime("%Y-%m-%d")
        us_date2 = kst_date.strftime("%Y-%m-%d")

        schedule = []
        for d in [us_date, us_date2]:
            for g in get_schedule(d):
                if g.get("game_date_kst") == game_date:
                    schedule.append(g)

        api_date = us_date  # cutoff_date용
    else:
        schedule = get_schedule(game_date)
        api_date = game_date

    predictions = predict_all_games_v3(api_date, schedule)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    return predictions


def predict_consensus(game_date: str, force: bool = False, kst: bool = False) -> dict:
    """V3 Pro: V2 F5 + V3 컨센서스 예측.

    Args:
        game_date: "YYYY-MM-DD"
        force: 캐시 무시
        kst: True면 game_date를 KST 날짜로 간주
    """
    cache_key = f"v3pro_kst_{game_date}" if kst else f"v3pro_{game_date}"
    cache_path = os.path.join(PREDICTIONS_DIR, f"{cache_key}.json")

    if not force and os.path.exists(cache_path):
        mtime = os.path.getmtime(cache_path)
        if time.time() - mtime < 3600:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

    # V3 예측 (KST 전달)
    v3_preds = predict_games_v3(game_date, force=force, kst=kst)

    # V2 F5 예측 — KST일 때는 전날+당일 미국 날짜 모두 조회
    from web.predict import predict_f5_games
    v2_map = {}
    if kst:
        from datetime import datetime, timedelta
        kst_date = datetime.strptime(game_date, "%Y-%m-%d")
        for d in [(kst_date - timedelta(days=1)).strftime("%Y-%m-%d"), game_date]:
            for p in predict_f5_games(d, n_sims=500, force=force):
                v2_map[p["game_id"]] = p
    else:
        for p in predict_f5_games(game_date, n_sims=500, force=force):
            v2_map[p["game_id"]] = p

    # 컨센서스 분석
    all_games = []
    top_picks = []  # ML/OU/TT 통합 탑픽

    for v3 in v3_preds:
        gid = v3.get("game_id")
        v2 = v2_map.get(gid)

        game = dict(v3)

        if v2 and v3.get("pick"):
            game["v2_pick"] = v2["pick"]
            game["v2_conf"] = round(v2["conf"], 3)
            game["consensus"] = v2["pick"] == v3["pick"]

            if game["consensus"]:
                v3_conf = v3.get("pick_prob", 0.5)
                if 0.65 <= v3_conf < 0.75:
                    game["tier"] = "sweet_spot"
                elif 0.60 <= v3_conf < 0.75:
                    game["tier"] = "good"
                else:
                    game["tier"] = "consensus"
            else:
                game["tier"] = "disagree"

            # V2 F5 O/U도 비교
            v2_f5_total = v2.get("avg_total")
            v3_expected = v3.get("expected_total")
            if v2_f5_total and v3_expected:
                v2_ou = "OVER" if v2_f5_total > 4.5 else "UNDER"
                v3_ou = "OVER" if v3_expected > 4.5 else "UNDER"
                game["ou_consensus"] = v2_ou == v3_ou
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

        # 탑픽 수집 — V3 sweet_picks 중 컨센서스인 것 우선
        base_info = {
            "game_id": gid,
            "away_team": v3.get("away_team"),
            "home_team": v3.get("home_team"),
            "away_pitcher": v3.get("away_pitcher", {}).get("name", "TBD"),
            "home_pitcher": v3.get("home_pitcher", {}).get("name", "TBD"),
            "tier": game.get("tier", "no_v2"),
            "v2_pick": game.get("v2_pick"),
            "v2_conf": game.get("v2_conf"),
        }

        for sp in v3.get("sweet_picks", []):
            # TT는 Pro TEAM TOTAL 탭에서 별도 처리 → TOP PICKS에서 제외
            if sp["type"] == "TT":
                continue
            # Pro TOP PICKS는 컨센서스만: V2와 같은 방향일 때만
            if sp["type"] == "ML" and not game.get("consensus", False):
                continue
            if sp["type"] == "OU" and not game.get("ou_consensus", False):
                continue
            pick_entry = {**base_info, **sp}
            if sp["type"] == "ML":
                pick_entry["ml_consensus"] = True
            elif sp["type"] == "OU":
                pick_entry["ou_consensus"] = True
            top_picks.append(pick_entry)

    # 탑픽 정렬: 확신도 내림차순
    top_picks.sort(key=lambda x: x.get("conf", 0), reverse=True)

    # all_games를 tier 순 정렬
    tier_order = {"sweet_spot": 0, "good": 1, "consensus": 2, "disagree": 3, "no_v2": 4}
    all_games.sort(key=lambda x: (tier_order.get(x.get("tier"), 9), -(x.get("pick_prob") or 0)))

    result = {
        "all_games": all_games,
        "top_picks": top_picks[:5],  # TOP 5
        "all_picks": top_picks,
        "stats": {
            "total_games": len(v3_preds),
            "consensus_count": sum(1 for g in all_games if g.get("consensus")),
            "sweet_spot_count": sum(1 for g in all_games if g.get("tier") == "sweet_spot"),
            "disagree_count": sum(1 for g in all_games if g.get("consensus") is False),
            "ou_consensus_count": sum(1 for g in all_games if g.get("ou_consensus")),
        },
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
