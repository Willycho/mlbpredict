"""매일 실행 — 당일 경기 예측 생성 + 전날 결과 검증."""

import sys
import os
import json
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine.plate_appearance as PA
from engine.lineup import get_pitcher_by_id, clear_cache
from engine.simulation import run_simulation
from data.mlb_api import get_schedule
from data.odds_api import get_mlb_odds

PREDICTIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "predictions")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


def predict_today(game_date: str | None = None):
    """당일 경기 예측 생성 (하이브리드: v4 승패 + v7 O/U)."""
    if game_date is None:
        # 한국시간 기준 오늘 → MLB 날짜 (하루 전)
        game_date = (date.today() - timedelta(days=1)).isoformat()

    print(f"=== {game_date} 경기 예측 ===")
    games = get_schedule(game_date)
    if not games:
        print("경기 없음")
        return

    # 배당 데이터 가져오기
    try:
        odds_data = get_mlb_odds()
        print(f"배당 데이터 {len(odds_data)}경기 로드")
    except Exception as e:
        print(f"배당 로드 실패: {e}")
        odds_data = []

    # 팀명 매칭용 (Odds API는 풀네임, MLB API는 약어)
    def find_odds(away_team, home_team):
        for od in odds_data:
            if (away_team.lower() in od["away_team"].lower() or
                od["away_team"].lower().endswith(away_team.lower())):
                if (home_team.lower() in od["home_team"].lower() or
                    od["home_team"].lower().endswith(home_team.lower())):
                    return od
        # 팀코드로 한번 더 시도
        from config import TEAMS
        away_name = TEAMS.get(away_team, away_team).split()[-1].lower()
        home_name = TEAMS.get(home_team, home_team).split()[-1].lower()
        for od in odds_data:
            if away_name in od["away_team"].lower() and home_name in od["home_team"].lower():
                return od
        return None

    predictions = []

    for g in games:
        away_sp_id = g["away_pitcher"]["id"]
        home_sp_id = g["home_pitcher"]["id"]
        if not away_sp_id or not home_sp_id:
            continue

        away_sp = get_pitcher_by_id(away_sp_id)
        home_sp = get_pitcher_by_id(home_sp_id)
        if not away_sp or not home_sp:
            continue

        print(f"  {g['away_team']}@{g['home_team']} | {g['away_pitcher']['name']} vs {g['home_pitcher']['name']}")

        # Phase 1: v4 (에러 0%) — 승패 예측
        PA.ERROR_RATE = 0.0
        clear_cache()
        r_win = run_simulation(
            g["away_team"], g["home_team"], away_sp, home_sp,
            n_simulations=500, mode="full",
        )

        # Phase 2: v7 (에러 8%) — O/U 예측
        PA.ERROR_RATE = 0.08
        clear_cache()
        r_ou = run_simulation(
            g["away_team"], g["home_team"], away_sp, home_sp,
            n_simulations=500, mode="full",
        )

        # 승패 판정 (v4 기준)
        win_team = g["home_team"] if r_win["avg_home_score"] > r_win["avg_away_score"] else g["away_team"]
        win_conf = max(r_win["home_win_prob"], r_win["away_win_prob"])

        # 배당 데이터 매칭
        odds = find_odds(g["away_team"], g["home_team"])
        ou_line = 8.5  # 기본값
        ml_away = None
        ml_home = None
        spread_line = None
        if odds:
            ou_line = odds.get("total", {}).get("line", 8.5)
            ml_away = odds.get("moneyline", {}).get("away")
            ml_home = odds.get("moneyline", {}).get("home")
            spread_line = odds.get("spread", {}).get("home_line")

        # O/U 판정 (v7 기준, 실제 배당 라인)
        ou_line_str = str(ou_line)
        ou_data = r_ou["over_under"].get(ou_line_str, {})
        if not ou_data:
            # 정확한 라인이 없으면 가장 가까운 거
            available = list(r_ou["over_under"].keys())
            closest = min(available, key=lambda x: abs(float(x) - ou_line))
            ou_data = r_ou["over_under"].get(closest, {})
        ou_over_prob = ou_data.get("over", 0.5)

        # 시뮬 예측 총득점 vs 배당 라인으로도 판단
        sim_total = r_ou["avg_total_runs"]
        ou_pick = "OVER" if sim_total > ou_line else "UNDER"
        ou_conf = max(ou_over_prob, 1 - ou_over_prob)

        pred = {
            "game_id": g["game_id"],
            "date": game_date,
            "away_team": g["away_team"],
            "home_team": g["home_team"],
            "away_pitcher": g["away_pitcher"]["name"],
            "home_pitcher": g["home_pitcher"]["name"],
            # 승패 (v4)
            "win_pick": win_team,
            "win_conf": round(win_conf, 4),
            "win_avg_away": r_win["avg_away_score"],
            "win_avg_home": r_win["avg_home_score"],
            # O/U (v7, 실제 배당 라인 기준)
            "ou_line": ou_line,
            "ou_pick": ou_pick,
            "ou_conf": round(ou_conf, 4),
            "ou_predicted_total": round(sim_total, 2),
            "ou_over_prob": round(ou_over_prob, 4),
            # 배당 정보
            "odds_ml_away": ml_away,
            "odds_ml_home": ml_home,
            "odds_spread": spread_line,
            "odds_ou_line": ou_line,
            # 아직 결과 없음
            "actual_away_score": None,
            "actual_home_score": None,
            "actual_total": None,
            "win_hit": None,
            "ou_hit": None,
        }
        predictions.append(pred)

        # 추천 등급
        rec = ""
        if win_conf >= 0.65:
            rec += f"WIN:{win_team} "
        if ou_conf >= 0.65:
            rec += f"{ou_pick} {ou_line} "
        if rec:
            odds_str = f"ML:{ml_away}/{ml_home}" if ml_away else ""
            print(f"    → {rec}| 시뮬 {sim_total:.1f}점 vs 라인 {ou_line} | {odds_str}")
        else:
            print(f"    → 추천 없음 (접전) | 시뮬 {sim_total:.1f}점 vs 라인 {ou_line}")

    # 저장
    path = os.path.join(PREDICTIONS_DIR, f"{game_date}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"\n{len(predictions)}경기 예측 저장: {path}")

    # 에러율 복원
    PA.ERROR_RATE = 0.08
    return predictions


def verify_results(game_date: str):
    """과거 예측과 실제 결과 비교."""
    path = os.path.join(PREDICTIONS_DIR, f"{game_date}.json")
    if not os.path.exists(path):
        print(f"예측 파일 없음: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    # 실제 결과 가져오기
    import requests
    from data.mlb_api import TEAM_ID_TO_CODE
    resp = requests.get("https://statsapi.mlb.com/api/v1/schedule", params={
        "sportId": 1, "gameType": "R", "date": game_date,
        "hydrate": "linescore",
    }, timeout=10)
    data = resp.json()

    actual = {}
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("detailedState") != "Final":
                continue
            ls = g.get("linescore", {})
            t = ls.get("teams", {})
            actual[g["gamePk"]] = {
                "away_score": t.get("away", {}).get("runs", 0),
                "home_score": t.get("home", {}).get("runs", 0),
            }

    # 매칭
    win_total = win_hit = 0
    win_65_total = win_65_hit = 0
    ou_total = ou_hit = 0
    ou_65_total = ou_65_hit = 0

    for p in predictions:
        gid = p["game_id"]
        if gid not in actual:
            continue

        a = actual[gid]
        p["actual_away_score"] = a["away_score"]
        p["actual_home_score"] = a["home_score"]
        p["actual_total"] = a["away_score"] + a["home_score"]

        # 승패 검증
        actual_winner = p["home_team"] if a["home_score"] > a["away_score"] else p["away_team"]
        p["win_hit"] = p["win_pick"] == actual_winner
        win_total += 1
        if p["win_hit"]:
            win_hit += 1
        if p["win_conf"] >= 0.65:
            win_65_total += 1
            if p["win_hit"]:
                win_65_hit += 1

        # O/U 검증 (실제 배당 라인 기준)
        ou_line = p.get("ou_line", 8.5)
        actual_over = p["actual_total"] > ou_line
        p["ou_hit"] = (p["ou_pick"] == "OVER") == actual_over
        ou_total += 1
        if p["ou_hit"]:
            ou_hit += 1
        if p["ou_conf"] >= 0.65:
            ou_65_total += 1
            if p["ou_hit"]:
                ou_65_hit += 1

    # 저장 (결과 업데이트)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    # 출력
    print(f"\n=== {game_date} 결과 검증 ===")
    print(f"승패: {win_hit}/{win_total} ({win_hit/win_total*100:.0f}%)" if win_total else "승패: -")
    print(f"승패 65%+: {win_65_hit}/{win_65_total} ({win_65_hit/win_65_total*100:.0f}%)" if win_65_total else "승패 65%+: -")
    print(f"O/U 8.5: {ou_hit}/{ou_total} ({ou_hit/ou_total*100:.0f}%)" if ou_total else "O/U: -")
    print(f"O/U 65%+: {ou_65_hit}/{ou_65_total} ({ou_65_hit/ou_65_total*100:.0f}%)" if ou_65_total else "O/U 65%+: -")


def monthly_summary():
    """이번 달 전체 적중률 요약."""
    files = sorted(f for f in os.listdir(PREDICTIONS_DIR) if f.endswith(".json"))
    if not files:
        print("예측 데이터 없음")
        return

    all_win = []
    all_ou = []

    for f in files:
        with open(os.path.join(PREDICTIONS_DIR, f), "r", encoding="utf-8") as fp:
            preds = json.load(fp)
        for p in preds:
            if p.get("win_hit") is not None:
                all_win.append(p)
            if p.get("ou_hit") is not None:
                all_ou.append(p)

    print(f"=== 월간 요약 ({len(files)}일) ===")

    def show(label, data, key):
        total = len(data)
        hits = sum(1 for d in data if d[key])
        c65 = [d for d in data if d.get(f"{key.replace('_hit','_conf')}",0) >= 0.65 or d.get(f"{'win' if 'win' in key else 'ou'}_conf",0) >= 0.65]
        h65 = sum(1 for d in c65 if d[key])
        print(f"{label}: {hits}/{total} ({hits/total*100:.0f}%)" if total else f"{label}: -")
        print(f"  65%+: {h65}/{len(c65)} ({h65/len(c65)*100:.0f}%)" if c65 else "  65%+: -")

    show("승패", all_win, "win_hit")
    show("O/U 8.5", all_ou, "ou_hit")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "predict":
            game_date = sys.argv[2] if len(sys.argv) > 2 else None
            predict_today(game_date)
        elif cmd == "verify":
            verify_results(sys.argv[2])
        elif cmd == "summary":
            monthly_summary()
    else:
        print("Usage:")
        print("  python daily_predict.py predict [YYYY-MM-DD]  # 당일 예측")
        print("  python daily_predict.py verify YYYY-MM-DD     # 결과 검증")
        print("  python daily_predict.py summary               # 월간 요약")
