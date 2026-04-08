"""Per-game refresh Pro (V2+V3 consensus).

Pro 전용 refresh. V3 refresh 로직 + V2 MC 재실행 + 컨센서스 재계산.
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
    "data", "storage", "predictions_pro"
)

REFRESH_WINDOW_MIN_BEFORE = 60
REFRESH_WINDOW_MAX_BEFORE = 5


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

    # top_picks 재빌드 — 전체 game 파일 다시 읽어서 consensus만 추출
    top_picks = []
    for mg in manifest["games"]:
        gid = mg["game_id"]
        gf = load_game_file(kst_date, gid)
        if not gf:
            continue
        pred = gf.get("prediction", {})
        consensus = pred.get("consensus")
        ou_consensus = pred.get("ou_consensus")
        for sp in pred.get("sweet_picks", []):
            if sp["type"] == "TT":
                continue
            if sp["type"] == "ML" and not consensus:
                continue
            if sp["type"] == "OU" and not ou_consensus:
                continue
            entry = {
                "game_id": gid,
                "away_team": mg["away_team"],
                "home_team": mg["home_team"],
                "away_pitcher": mg.get("away_pitcher"),
                "home_pitcher": mg.get("home_pitcher"),
                "tier": pred.get("tier"),
                "v2_pick": pred.get("v2_pick"),
                "v2_conf": pred.get("v2_conf"),
                **sp,
            }
            if sp["type"] == "ML":
                entry["ml_consensus"] = True
            elif sp["type"] == "OU":
                entry["ou_consensus"] = True
            top_picks.append(entry)
    top_picks.sort(key=lambda x: x.get("conf", 0), reverse=True)
    manifest["top_picks"] = top_picks[:5]

    path = os.path.join(STORAGE_ROOT, kst_date, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def pick_games_to_refresh(kst_date: str) -> list[dict]:
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


def refresh_single_game_pro(kst_date: str, game_meta: dict) -> bool:
    game_id = game_meta["game_id"]
    old_file = load_game_file(kst_date, game_id)
    if not old_file:
        return False

    us_date = (datetime.strptime(kst_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    schedule_candidates = []
    for d in [us_date, kst_date]:
        for g in get_schedule(d):
            if g.get("game_id") == game_id:
                schedule_candidates.append(g)
                break
    if not schedule_candidates:
        return False
    new_game = schedule_candidates[0]

    old_aw_pid = old_file.get("locked_away_pitcher_id")
    old_hm_pid = old_file.get("locked_home_pitcher_id")
    new_aw_pid = (new_game.get("away_pitcher") or {}).get("id")
    new_hm_pid = (new_game.get("home_pitcher") or {}).get("id")
    pitcher_changed = (old_aw_pid != new_aw_pid) or (old_hm_pid != new_hm_pid)

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

    new_game["f5_total_line"] = f5_line

    # V3 predict
    v3_predictions = predict_all_games_v3(us_date, [new_game], skip_odds_fetch=True)
    if not v3_predictions:
        return False
    v3 = v3_predictions[0]

    # V2 MC predict (1 game only)
    from web.predict import predict_f5_games
    v2_preds = []
    for d in [us_date, kst_date]:
        try:
            v2_preds.extend(predict_f5_games(d, n_sims=500, force=True))
        except Exception as e:
            print(f"  [v2 error] {d}: {e}")
    v2_match = next((p for p in v2_preds if p.get("game_id") == game_id), None)

    # 컨센서스 병합
    new_pred = dict(v3)
    if v2_match and v3.get("pick"):
        new_pred["v2_pick"] = v2_match.get("pick")
        new_pred["v2_conf"] = round(v2_match.get("conf", 0.0), 3)
        new_pred["consensus"] = (v2_match.get("pick") == v3["pick"])

        v3_conf = v3.get("pick_prob", 0.5)
        if new_pred["consensus"]:
            if 0.65 <= v3_conf < 0.75:
                new_pred["tier"] = "sweet_spot"
            elif 0.60 <= v3_conf < 0.75:
                new_pred["tier"] = "good"
            else:
                new_pred["tier"] = "consensus"
        else:
            new_pred["tier"] = "disagree"

        v2_total = v2_match.get("avg_total")
        v3_total = v3.get("expected_total")
        if v2_total and v3_total:
            v2_ou = "OVER" if v2_total > 4.5 else "UNDER"
            v3_ou = "OVER" if v3_total > 4.5 else "UNDER"
            new_pred["ou_consensus"] = (v2_ou == v3_ou)
            new_pred["v2_f5_total"] = round(v2_total, 1)
        else:
            new_pred["ou_consensus"] = None
    else:
        new_pred["v2_pick"] = None
        new_pred["v2_conf"] = None
        new_pred["consensus"] = None
        new_pred["ou_consensus"] = None
        new_pred["tier"] = "no_v2"

    # 저장
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
        "v2_pick": new_pred.get("v2_pick"),
        "v2_conf": new_pred.get("v2_conf"),
        "consensus": new_pred.get("consensus"),
        "ou_consensus": new_pred.get("ou_consensus"),
        "tier": new_pred.get("tier"),
        "f5_book_line": f5_line,
        "away_pitcher": (new_pred.get("away_pitcher") or {}).get("name", "TBD"),
        "home_pitcher": (new_pred.get("home_pitcher") or {}).get("name", "TBD"),
        "lineup_source": (new_pred.get("away_matchup") or {}).get("lineup_source", "roster"),
    })

    print(f"  [ok] {game_id} refreshed (pro)")
    return True


def main():
    kst_date = get_current_kst_date()
    print(f"[pro-refresh] KST date: {kst_date}")

    targets = pick_games_to_refresh(kst_date)
    print(f"[pro-refresh] {len(targets)} games in window")

    if not targets:
        tomorrow = (datetime.strptime(kst_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        tomorrow_targets = pick_games_to_refresh(tomorrow)
        if tomorrow_targets:
            print(f"[pro-refresh] checking tomorrow ({tomorrow}): {len(tomorrow_targets)}")
            targets = tomorrow_targets
            kst_date = tomorrow

    changed = 0
    for t in targets:
        print(f"[{t['away_team']}@{t['home_team']} {t.get('game_time_kst','')}]")
        if refresh_single_game_pro(kst_date, t):
            changed += 1

    print(f"[pro-refresh] DONE, {changed} updated")


if __name__ == "__main__":
    main()
