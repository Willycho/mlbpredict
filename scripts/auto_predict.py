"""자동 예측 + 결과 저장 스케줄러.

KST 오전 8시: 당일 MLB 경기 예측 (라인업 반영)
KST 오후 2시: 전일 MLB 경기 결과 검증

Usage:
    python scripts/auto_predict.py              # 한번 실행 (예측 + 결과)
    python scripts/auto_predict.py --daemon      # 백그라운드 데몬 (스케줄 자동 실행)
    python scripts/auto_predict.py predict       # 예측만
    python scripts/auto_predict.py verify        # 결과만
"""
import sys, os, json, time, argparse
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KST = timezone(timedelta(hours=9))
PREDICTIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "predictions")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def log(msg):
    ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOG_DIR, "auto_predict.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_mlb_date():
    """현재 KST 기준 MLB 경기 날짜. 오후 2시 전이면 전날, 이후면 당일."""
    now = datetime.now(KST)
    if now.hour < 14:
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")
    return now.strftime("%Y-%m-%d")


def get_yesterday_mlb_date():
    """결과 검증용: 가장 최근 끝난 MLB 경기 날짜."""
    now = datetime.now(KST)
    if now.hour < 14:
        return (now - timedelta(days=2)).strftime("%Y-%m-%d")
    return (now - timedelta(days=1)).strftime("%Y-%m-%d")


def run_predict(date_str=None):
    """예측 실행."""
    from web.predict import predict_games, predict_f5_games

    if date_str is None:
        date_str = get_mlb_date()

    log(f"Predicting {date_str}...")
    try:
        # Full Game
        preds = predict_games(date_str, n_sims=2000, force=True)
        api_count = sum(1 for p in preds if p.get("lineup_src", "").startswith("API"))
        log(f"  Full Game: {len(preds)} games, {api_count} with API lineup")

        # F5
        f5_preds = predict_f5_games(date_str, n_sims=2000, force=True)
        log(f"  F5: {len(f5_preds)} games")

        # Edge picks summary
        edges = [p for p in preds if p.get("edge_grade") in ("A", "B", "C")]
        if edges:
            log(f"  Edge picks: {len(edges)}")
            for e in edges:
                side = e["home"] if e.get("edge_side") == "home" else e["away"]
                log(f"    {e['away']}@{e['home']} {e['edge_grade']} {side} edge+{e.get('edge_value',0)*100:.1f}%")

        return preds
    except Exception as e:
        log(f"  ERROR: {e}")
        return []


def run_verify(date_str=None):
    """결과 검증."""
    from web.predict import predict_games, get_results

    if date_str is None:
        date_str = get_yesterday_mlb_date()

    log(f"Verifying {date_str}...")
    try:
        preds = predict_games(date_str, n_sims=2000)
        results = get_results(date_str)

        win_total = win_hit = edge_total = edge_hit = 0
        for p in preds:
            r = results.get(p.get("game_id"))
            if not r:
                continue
            winner = r["winner"]
            if p["pick"] == winner:
                win_hit += 1
            win_total += 1

            grade = p.get("edge_grade", "PASS")
            if grade in ("A", "B", "C"):
                edge_total += 1
                edge_pick = p["home"] if p.get("edge_side") == "home" else p["away"]
                if edge_pick == winner:
                    edge_hit += 1

        log(f"  Win: {win_hit}/{win_total} ({win_hit/win_total*100:.0f}%)" if win_total else "  Win: no games")
        log(f"  Edge: {edge_hit}/{edge_total} ({edge_hit/edge_total*100:.0f}%)" if edge_total else "  Edge: no picks")

        # Save daily report
        report = {
            "date": date_str,
            "verified_at": datetime.now(KST).isoformat(),
            "win_total": win_total, "win_hit": win_hit,
            "edge_total": edge_total, "edge_hit": edge_hit,
        }
        report_path = os.path.join(PREDICTIONS_DIR, f"report_{date_str}.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report
    except Exception as e:
        log(f"  ERROR: {e}")
        return {}


def daemon():
    """백그라운드 데몬: 스케줄에 맞춰 자동 실행.

    KST 기준:
    - 새벽 1시: 예측 (대부분 라인업 발표됨, 경기 시작 전)
    - 오전 10시: 결과 검증 (서부 경기까지 대부분 종료)
    """
    log("Daemon started")
    log("  Schedule: KST 01:00 predict, KST 10:00 verify")
    last_predict = None
    last_verify = None

    while True:
        now = datetime.now(KST)
        today = now.strftime("%Y-%m-%d")
        hour = now.hour

        # 새벽 1시: 예측 (라인업 대부분 발표된 시점)
        if 0 <= hour <= 3 and last_predict != today:
            run_predict()
            last_predict = today

        # 오전 10시: 결과 검증 (서부 나이트게임까지 종료)
        if 9 <= hour <= 12 and last_verify != today:
            run_verify()
            last_verify = today

        # 5분마다 체크
        time.sleep(300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", nargs="?", default="both",
                        choices=["predict", "verify", "both", "daemon"])
    parser.add_argument("--date", default=None)
    args = parser.parse_args()

    if args.action == "daemon":
        daemon()
    elif args.action == "predict":
        run_predict(args.date)
    elif args.action == "verify":
        run_verify(args.date)
    else:
        run_predict(args.date)
        run_verify()


if __name__ == "__main__":
    main()
