"""Per-game refresh script.

실행 시점: GitHub Actions cron 15분마다
목적:
  - 임박 경기(시작 10~60분 전)만 스캔
  - MLB API로 라인업 재조회 (무료)
  - 저장된 투수와 다르면 F5 라인 재조회 (1 credit)
  - 해당 경기만 재예측 → 저장

Storage 업데이트 대상:
  data/storage/predictions/{kst_date}/{game_id}.json
  data/storage/predictions/{kst_date}/manifest.json
"""
import sys
import os
import json
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mlb_api import get_schedule
from data.odds_api import get_f5_line_for_event, find_event_by_teams
from engine.game_score import predict_all_games_v3
from config import TEAMS


STORAGE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "storage", "predictions"
)

# 경기 시작 몇 분 전부터 refresh 할지
REFRESH_WINDOW_MIN_BEFORE = 60  # 60분 전부터
REFRESH_WINDOW_MAX_BEFORE = 5   # 5분 전까지


def get_current_kst_date() -> str:
    return datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")


def parse_game_datetime(game_date_kst: str, game_time_kst: str) -> datetime | None:
    if not game_date_kst or not game_time_kst:
        return None
    try:
        dt = datetime.strptime(f"{game_date_kst} {game_time_kst}", "%Y-%m-%d %H:%M")
        return dt.replace(tzinfo=timezone(timedelta(hours=9)))
    except Exception:
        return None


def load_manifest(kst_date: str) -> dict | None:
    path = os.path.join(STORAGE_ROOT, kst_date, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_game_file(kst_date: str, game_id: int) -> dict | None:
    path = os.path.join(STORAGE_ROOT, kst_date, f"{game_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_game_file(kst_date: str, game_id: int, data: dict):
    path = os.path.join(STORAGE_ROOT, kst_date, f"{game_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_manifest_entry(kst_date: str, game_id: int, update: dict):
    manifest = load_manifest(kst_date)
    if not manifest:
        return
    for g in manifest.get("games", []):
        if g.get("game_id") == game_id:
            g.update(update)
            break
    manifest["last_modified_at"] = datetime.now(timezone.utc).isoformat()
    path = os.path.join(STORAGE_ROOT, kst_date, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def pick_games_to_refresh(kst_date: str) -> list[dict]:
    """refresh 대상 경기 선정: 시작 60분 ~ 5분 전."""
    manifest = load_manifest(kst_date)
    if not manifest:
        return []
    now = datetime.now(timezone.utc)
    window_max = now + timedelta(minutes=REFRESH_WINDOW_MIN_BEFORE)
    window_min = now + timedelta(minutes=REFRESH_WINDOW_MAX_BEFORE)
    targets = []
    for g in manifest.get("games", []):
        gdt = parse_game_datetime(g.get("game_date_kst", ""), g.get("game_time_kst", ""))
        if not gdt:
            continue
        gdt_utc = gdt.astimezone(timezone.utc)
        if window_min <= gdt_utc <= window_max:
            targets.append(g)
    return targets


def refresh_single_game(kst_date: str, game_meta: dict) -> bool:
    """한 경기만 refresh.

    Returns: True면 변경됨, False면 변경 없음
    """
    game_id = game_meta["game_id"]
    old_file = load_game_file(kst_date, game_id)
    if not old_file:
        print(f"  [skip] no existing file for {game_id}")
        return False

    # 1. MLB API로 최신 스케줄 + 라인업 재조회 (무료)
    us_date = (datetime.strptime(kst_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    schedule_candidates = []
    for d in [us_date, kst_date]:
        for g in get_schedule(d):
            if g.get("game_id") == game_id:
                schedule_candidates.append(g)
                break
    if not schedule_candidates:
        print(f"  [skip] game {game_id} not in MLB schedule")
        return False
    new_game = schedule_candidates[0]

    # 2. 투수 변경 감지
    old_aw_pid = old_file.get("locked_away_pitcher_id")
    old_hm_pid = old_file.get("locked_home_pitcher_id")
    new_aw_pid = (new_game.get("away_pitcher") or {}).get("id")
    new_hm_pid = (new_game.get("home_pitcher") or {}).get("id")

    pitcher_changed = (old_aw_pid != new_aw_pid) or (old_hm_pid != new_hm_pid)

    # 3. F5 라인 재조회는 투수 변경 시에만
    f5_line = old_file.get("f5_book_line")
    if pitcher_changed:
        print(f"  [pitcher changed] {game_id}: fetching new F5 line")
        event_id = find_event_by_teams(
            new_game.get("away_team_name", ""),
            new_game.get("home_team_name", ""),
            kst_date,
        )
        if event_id:
            new_line = get_f5_line_for_event(event_id)
            if new_line is not None:
                f5_line = new_line
                print(f"    → new F5 line: {new_line}")
    else:
        print(f"  [pitcher same] {game_id}: reusing F5 line={f5_line}")

    # 4. Predict (라인업 포함)
    new_game["f5_total_line"] = f5_line
    predictions = predict_all_games_v3(us_date, [new_game], skip_odds_fetch=True)
    if not predictions:
        print(f"  [skip] predict returned empty for {game_id}")
        return False
    new_pred = predictions[0]

    # 5. 파일 업데이트
    now_iso = datetime.now(timezone.utc).isoformat()
    old_file.update({
        "last_refreshed_at": now_iso,
        "refresh_type": "per_game",
        "f5_book_line": f5_line,
        "prediction": new_pred,
        "locked_away_pitcher_id": new_aw_pid,
        "locked_home_pitcher_id": new_hm_pid,
        "pitcher_changed_at_refresh": pitcher_changed,
    })
    save_game_file(kst_date, game_id, old_file)

    update_manifest_entry(kst_date, game_id, {
        "last_refreshed_at": now_iso,
        "refresh_type": "per_game",
        "pick": new_pred.get("pick"),
        "pick_prob": new_pred.get("pick_prob"),
        "expected_total": new_pred.get("expected_total"),
        "f5_book_line": f5_line,
        "away_pitcher": (new_pred.get("away_pitcher") or {}).get("name", "TBD"),
        "home_pitcher": (new_pred.get("home_pitcher") or {}).get("name", "TBD"),
        "lineup_source": (new_pred.get("away_matchup") or {}).get("lineup_source", "roster"),
    })

    print(f"  [ok] {game_id} refreshed")
    return True


def main():
    kst_date = get_current_kst_date()
    print(f"[refresh] KST date: {kst_date}")

    targets = pick_games_to_refresh(kst_date)
    print(f"[refresh] {len(targets)} games in window")

    if not targets:
        # 오늘 날짜 manifest 없으면 다음날도 확인 (자정 전후)
        tomorrow = (datetime.strptime(kst_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        tomorrow_targets = pick_games_to_refresh(tomorrow)
        if tomorrow_targets:
            print(f"[refresh] checking tomorrow ({tomorrow}): {len(tomorrow_targets)} in window")
            targets = tomorrow_targets
            kst_date = tomorrow

    changed = 0
    for t in targets:
        print(f"[{t['away_team']}@{t['home_team']} {t.get('game_time_kst','')}]")
        if refresh_single_game(kst_date, t):
            changed += 1

    print(f"[refresh] DONE, {changed} games updated")


if __name__ == "__main__":
    main()
