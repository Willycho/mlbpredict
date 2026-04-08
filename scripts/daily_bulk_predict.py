"""Daily bulk prediction script.

실행 시점: 매일 KST 21:00 (UTC 12:00)
목적: 다음 KST 날짜의 전 경기 F5 라인 + 기본 예측 (라인업 없이 로스터 기준)

Storage:
  data/storage/predictions/{kst_date}/
    manifest.json           # 경기 목록 + 상태
    {game_id}.json          # 경기별 상세 예측
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
    "data", "storage", "predictions"
)


def get_target_kst_date() -> str:
    """스크립트 실행 시점 기준 '다음 KST 경기 날짜'를 구함.

    예: KST 21:00에 실행 → 다음날(내일) 경기를 가져옴
    """
    now_kst = datetime.now(timezone(timedelta(hours=9)))
    # 21:00 이후 실행이면 다음날, 이전이면 오늘
    if now_kst.hour >= 18:
        target = now_kst + timedelta(days=1)
    else:
        target = now_kst
    return target.strftime("%Y-%m-%d")


def fetch_kst_schedule(kst_date: str) -> list[dict]:
    """KST 날짜 기준 경기 조회 (미국 날짜 전날+당일 둘 다 봐야 함)."""
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


def save_predictions(kst_date: str, predictions: list[dict], f5_lines_map: dict):
    """예측 결과를 storage에 저장."""
    date_dir = os.path.join(STORAGE_ROOT, kst_date)
    os.makedirs(date_dir, exist_ok=True)

    now_iso = datetime.now(timezone.utc).isoformat()

    # Individual game files
    manifest_games = []
    for p in predictions:
        gid = p.get("game_id")
        if not gid:
            continue

        away_t = p.get("away_team", "")
        home_t = p.get("home_team", "")
        f5_line = f5_lines_map.get(f"{away_t}@{home_t}")

        game_file = {
            "game_id": gid,
            "kst_date": kst_date,
            "bulk_fetched_at": now_iso,
            "last_refreshed_at": now_iso,
            "refresh_type": "bulk_initial",  # bulk_initial / per_game
            "f5_book_line": f5_line,  # 배당사 F5 라인 (없으면 None)
            "prediction": p,
            # 투수 기록 — per_game_refresh에서 투수 변경 감지용
            "locked_away_pitcher_id": (p.get("away_pitcher") or {}).get("id"),
            "locked_home_pitcher_id": (p.get("home_pitcher") or {}).get("id"),
        }
        with open(os.path.join(date_dir, f"{gid}.json"), "w", encoding="utf-8") as f:
            json.dump(game_file, f, ensure_ascii=False, indent=2)

        manifest_games.append({
            "game_id": gid,
            "away_team": away_t,
            "home_team": home_t,
            "game_time_kst": p.get("game_time_kst", ""),
            "game_date_kst": p.get("game_date_kst", kst_date),
            "away_pitcher": (p.get("away_pitcher") or {}).get("name", "TBD"),
            "home_pitcher": (p.get("home_pitcher") or {}).get("name", "TBD"),
            "pick": p.get("pick"),
            "pick_prob": p.get("pick_prob"),
            "expected_total": p.get("expected_total"),
            "f5_book_line": f5_line,
            "lineup_source": (p.get("away_matchup") or {}).get("lineup_source", "roster"),
            "last_refreshed_at": now_iso,
            "refresh_type": "bulk_initial",
        })

    manifest = {
        "kst_date": kst_date,
        "generated_at": now_iso,
        "total_games": len(manifest_games),
        "games": sorted(manifest_games, key=lambda g: g.get("game_time_kst", "")),
    }
    with open(os.path.join(date_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[bulk] saved {len(manifest_games)} games to {date_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="Target KST date YYYY-MM-DD (override auto)")
    args = parser.parse_args()

    kst_date = args.date or get_target_kst_date()
    print(f"[bulk] target KST date: {kst_date}")

    # 1. MLB 스케줄
    schedule = fetch_kst_schedule(kst_date)
    if not schedule:
        print(f"[bulk] no games found for {kst_date}")
        return
    print(f"[bulk] {len(schedule)} games found")

    # 2. F5 배당사 라인 (미래 경기만 자동 필터)
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
    print(f"[bulk] F5 lines fetched: {len(f5_lines_map)}/{len(schedule)}")

    # 3. 스케줄에 F5 라인 주입 (engine이 쓸 수 있도록)
    for game in schedule:
        key = f"{game.get('away_team','')}@{game.get('home_team','')}"
        game["f5_total_line"] = f5_lines_map.get(key)

    # 4. Predict (로스터 기준, 라인업 없이) — 이미 F5 라인 주입됐으니 재호출 스킵
    api_date = (datetime.strptime(kst_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    predictions = predict_all_games_v3(api_date, schedule, skip_odds_fetch=True)
    print(f"[bulk] {len(predictions)} predictions done")

    # 5. 저장
    save_predictions(kst_date, predictions, f5_lines_map)

    print(f"[bulk] DONE")


if __name__ == "__main__":
    main()
