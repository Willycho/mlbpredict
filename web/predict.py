"""v2.3 예측 엔진 — 웹 서비스용 래퍼.

test_live.py의 핵심 로직을 함수화.
결과를 predictions/web_{date}.json으로 캐시.
"""
import sys, os, json, time
from datetime import datetime, timedelta, timezone
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR, V2_DATA_DIR, PROBABILITY_DAMPENING_ALPHA, KELLY_FRACTION, TEAMS
from data.mlb_api import get_schedule
from data.odds_api import get_mlb_odds
from engine.edge import analyze_moneyline
from engine.lineup import (
    get_team_lineup, get_league_avg,
    get_batter_pitch_arsenal, get_pitcher_pitch_arsenal, get_h2h_stats,
)
from engine.bayesian_matchup import compute_matchup_v2
from engine.bullpen_quality import enhance_bullpen_profiles
from engine.cpp_bridge import (
    prepare_transition_arrays, prepare_matchup_cache,
    prepare_defense_buckets, SPEED_BUCKET_TO_IDX,
)
from engine.cpp.sim_core import run_simulation_cpp

PREDICTIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "predictions")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Lazy-loaded global data
_data_cache = {}


def _load_data():
    """Global data 한번만 로드."""
    if _data_cache:
        return _data_cache
    _data_cache["league_avg"] = get_league_avg(2025)
    _data_cache["pitchers_df"] = pd.read_parquet(os.path.join(DATA_DIR, "pitchers_processed.parquet"))
    _data_cache["batters_df"] = pd.read_parquet(os.path.join(DATA_DIR, "batters_processed.parquet"))
    _data_cache["speed_df"] = pd.read_parquet(os.path.join(V2_DATA_DIR, "speed_proxy.parquet"))
    _data_cache["defense_df"] = pd.read_parquet(os.path.join(V2_DATA_DIR, "team_defense.parquet"))
    _data_cache["bp_df"] = pd.read_parquet(os.path.join(V2_DATA_DIR, "bullpen_profiles.parquet"))
    tm_df = pd.read_parquet(os.path.join(V2_DATA_DIR, "transition_matrix.parquet"))
    _data_cache["tm_arrays"] = prepare_transition_arrays(tm_df)
    return _data_cache


def _get_pitcher(pid, pitchers_df, league_avg):
    if pid is None:
        return None
    m = pitchers_df[pitchers_df["player_id"] == pid]
    if len(m) == 0:
        # Try name-based search
        return None
    r = m.iloc[-1]
    s = {"player_id": int(pid), "name": str(r.get("name", "")),
         "k_rate": r["k_rate"], "bb_rate": r["bb_rate"],
         "hbp_rate": r["hbp_rate"], "hr_rate": r["hr_rate"],
         "babip": r["babip"], "throws": str(r.get("throws", "R")),
         "era": float(r.get("era", 4.2)), "fip": float(r.get("fip", 0)),
         "tbf": int(r.get("tbf", 100))}
    a = get_pitcher_pitch_arsenal(int(pid))
    if a:
        s["arsenal"] = a
    return s


def _get_pitcher_by_name(name, pitchers_df, league_avg):
    parts = name.strip().split()
    if not parts:
        return None
    last = parts[-1]
    m = pitchers_df[pitchers_df["name"].str.contains(last, case=False, na=False)]
    if len(m) == 0:
        return None
    r = m.iloc[-1]
    return _get_pitcher(int(r["player_id"]), pitchers_df, league_avg)


def _fallback_pitcher(league_avg):
    return {"player_id": "avg", "throws": "R", "tbf": 500,
            "k_rate": league_avg["k_rate"], "bb_rate": league_avg["bb_rate"],
            "hbp_rate": league_avg["hbp_rate"], "hr_rate": league_avg["hr_rate"],
            "babip": league_avg["babip"]}


def _get_batter(pid, batters_df):
    m = batters_df[batters_df["player_id"] == pid]
    if len(m) == 0:
        return None
    r = m.iloc[-1]
    return {"player_id": int(pid), "name": str(r.get("name", "")),
            "bats": str(r.get("bats", "R")), "pa": int(r.get("pa", 0)),
            "k_rate": r["k_rate"], "bb_rate": r["bb_rate"],
            "hbp_rate": r["hbp_rate"], "hr_rate": r["hr_rate"],
            "babip": r["babip"], "iso": r.get("iso", 0.15),
            "ld_rate": r.get("ld_rate", 0.21), "gb_rate": r.get("gb_rate", 0.43),
            "fb_rate": r.get("fb_rate", 0.34), "iffb_rate": r.get("iffb_rate", 0.10),
            "gdp_rate": r.get("gdp_rate", 0.15)}


def predict_games(date_str: str, n_sims: int = 2000, force: bool = False) -> list[dict]:
    """당일 전경기 v2.3 예측.

    Returns list of prediction dicts, one per game.
    Results cached to predictions/web_{date}.json.
    """
    cache_path = os.path.join(PREDICTIONS_DIR, f"web_{date_str}.json")
    if not force and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    data = _load_data()
    league_avg = data["league_avg"]
    pitchers_df = data["pitchers_df"]
    batters_df = data["batters_df"]
    speed_df = data["speed_df"]
    defense_df = data["defense_df"]
    bp_df = data["bp_df"]
    ti, tr, tp, to_ = data["tm_arrays"]
    alpha = PROBABILITY_DAMPENING_ALPHA

    # Team name matching for odds
    _name_to_code = {}
    for code, full in TEAMS.items():
        _name_to_code[full.split()[-1].lower()] = code

    # Schedule
    games = get_schedule(date_str)
    if not games:
        return []

    # Load previous cache for odds lockout
    prev_cache = {}
    prev_path = os.path.join(PREDICTIONS_DIR, f"web_{date_str}.json")
    if os.path.exists(prev_path):
        try:
            with open(prev_path, "r", encoding="utf-8") as f:
                for p in json.load(f):
                    if p.get("ml_home") is not None:
                        prev_cache[p.get("game_id")] = {
                            "ml_home": p["ml_home"],
                            "ml_away": p["ml_away"],
                        }
        except Exception:
            pass

    # Odds
    try:
        odds_data = get_mlb_odds()
    except Exception:
        odds_data = []

    def match_odds(away_code, home_code):
        for od in odds_data:
            oa = _name_to_code.get(od["away_team"].split()[-1].lower())
            oh = _name_to_code.get(od["home_team"].split()[-1].lower())
            if oa == away_code and oh == home_code:
                return od
        return None

    def is_game_locked(game):
        """경기 시작 30분 전부터 배당 갱신 차단."""
        game_time_str = game.get("game_time", "")
        if not game_time_str or len(game_time_str) < 5:
            return False
        try:
            # game_time is UTC "HH:MM" from MLB API gameDate
            game_dt = datetime.strptime(f"{date_str}T{game_time_str}:00Z", "%Y-%m-%dT%H:%M:%SZ")
            game_dt = game_dt.replace(tzinfo=timezone.utc)
            now_utc = datetime.now(timezone.utc)
            return now_utc >= game_dt - timedelta(minutes=30)
        except Exception:
            return False

    def get_def(team):
        m = defense_df[defense_df["team"] == team]
        return m.iloc[-1]["defense_bucket"] if len(m) > 0 else "avg"

    def get_bp(team):
        m = bp_df[bp_df["team"] == team]
        result = {}
        for _, row in m.iterrows():
            result[row["role"]] = {"player_id": f"bp_{team}_{row['role']}",
                "k_rate": row["k_rate"], "bb_rate": row["bb_rate"],
                "hr_rate": row.get("hr_rate", 0.03), "hbp_rate": row.get("hbp_rate", 0.01),
                "babip": row.get("babip", 0.300), "throws": "R"}
        for role in ["setup_early", "setup_late", "bridge", "closer"]:
            if role not in result:
                result[role] = dict(_fallback_pitcher(league_avg), player_id=f"bp_{team}_{role}")
        return result

    predictions = []

    for g in games:
        away = g["away_team"]
        home = g["home_team"]
        ap_info = g.get("away_pitcher", {})
        hp_info = g.get("home_pitcher", {})

        if ap_info.get("name", "TBD") == "TBD" or hp_info.get("name", "TBD") == "TBD":
            continue
        if g.get("status") == "Postponed":
            continue

        t0 = time.time()

        # Pitchers
        asp = (_get_pitcher(ap_info.get("id"), pitchers_df, league_avg)
               or _get_pitcher_by_name(ap_info.get("name", ""), pitchers_df, league_avg)
               or _fallback_pitcher(league_avg))
        hsp = (_get_pitcher(hp_info.get("id"), pitchers_df, league_avg)
               or _get_pitcher_by_name(hp_info.get("name", ""), pitchers_df, league_avg)
               or _fallback_pitcher(league_avg))

        # Lineups
        al_ids = [p["id"] for p in g.get("away_lineup", []) if p.get("id")]
        hl_ids = [p["id"] for p in g.get("home_lineup", []) if p.get("id")]

        # Build lineups: use API players, fill missing with league avg
        def build_lineup(player_ids, team):
            lineup = []
            n_api = 0
            for pid in player_ids[:9]:
                b = _get_batter(pid, batters_df)
                if b:
                    lineup.append(b)
                    n_api += 1
                else:
                    # Unknown player -> league average filler
                    lineup.append({
                        "player_id": int(pid), "name": f"#{pid}",
                        "bats": "R", "pa": 100,
                        "k_rate": league_avg["k_rate"], "bb_rate": league_avg["bb_rate"],
                        "hbp_rate": league_avg["hbp_rate"], "hr_rate": league_avg["hr_rate"],
                        "babip": league_avg["babip"], "iso": 0.150,
                        "ld_rate": 0.21, "gb_rate": 0.43, "fb_rate": 0.34,
                        "iffb_rate": 0.10, "gdp_rate": 0.15,
                    })
            return lineup, n_api

        if len(al_ids) >= 9:
            al, al_api = build_lineup(al_ids, away)
        else:
            al, al_api = [], 0
        if len(al) < 9:
            try:
                al = get_team_lineup(away, 2025)
                al_api = 0
            except Exception:
                continue

        if len(hl_ids) >= 9:
            hl, hl_api = build_lineup(hl_ids, home)
        else:
            hl, hl_api = [], 0
        if len(hl) < 9:
            try:
                hl = get_team_lineup(home, 2025)
                hl_api = 0
            except Exception:
                continue

        if len(al) < 9 or len(hl) < 9:
            continue

        # Lineup source tag
        total_api = al_api + hl_api
        if total_api == 18:
            lineup_src = "API"
        elif total_api >= 14:
            lineup_src = f"API({total_api}/18)"
        elif total_api > 0:
            lineup_src = f"PARTIAL({total_api}/18)"
        else:
            lineup_src = "default"

        # Arsenal + H2H
        asp_id = asp["player_id"] if isinstance(asp["player_id"], int) else None
        hsp_id = hsp["player_id"] if isinstance(hsp["player_id"], int) else None
        for b in al:
            pid = b.get("player_id")
            if pid:
                b["arsenal"] = get_batter_pitch_arsenal(pid)
                if hsp_id:
                    b["h2h"] = get_h2h_stats(pid, hsp_id)
        for b in hl:
            pid = b.get("player_id")
            if pid:
                b["arsenal"] = get_batter_pitch_arsenal(pid)
                if asp_id:
                    b["h2h"] = get_h2h_stats(pid, asp_id)

        # Bullpen
        away_bp = enhance_bullpen_profiles(get_bp(away), hl)
        home_bp = enhance_bullpen_profiles(get_bp(home), al)
        away_def = get_def(away)
        home_def = get_def(home)

        # Matchup
        mc = {}
        for lineup, pitcher, is_home in [(al, hsp, False), (hl, asp, True)]:
            for batter in lineup:
                pid = batter.get("player_id")
                if not pid:
                    continue
                probs = compute_matchup_v2(
                    batter=batter, pitcher=pitcher, league_avg=league_avg,
                    home_team=home, is_batter_home=is_home,
                    batter_arsenal=batter.get("arsenal"),
                    pitcher_arsenal=pitcher.get("arsenal"),
                    h2h_stats=batter.get("h2h"),
                    defense_bucket=home_def if not is_home else away_def,
                )
                mc[(pid, pitcher.get("player_id"))] = probs

        for lineup, bpd, is_home in [(al, home_bp, False), (hl, away_bp, True)]:
            for role, bp in bpd.items():
                for batter in lineup:
                    pid = batter.get("player_id")
                    if not pid:
                        continue
                    probs = compute_matchup_v2(
                        batter=batter, pitcher=bp, league_avg=league_avg,
                        home_team=home, is_batter_home=is_home,
                        defense_bucket=home_def if not is_home else away_def,
                    )
                    mc[(pid, bp.get("player_id"))] = probs

        # C++ sim
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
            n_sims=n_sims,
            base_seed=g.get("game_id", 0) or hash(f"{away}{home}{date_str}") % 2**32,
        )

        aw = scores[:, 0].astype(float)
        hw = scores[:, 1].astype(float)
        total = aw + hw
        hwin = (hw > aw).sum()
        awin = (aw > hw).sum()
        nt = hwin + awin
        raw_hp = hwin / nt if nt > 0 else 0.5
        hp = 0.50 + (raw_hp - 0.50) * alpha
        # Strong confidence cap
        if hp > 0.62:
            hp = 0.62 + (hp - 0.62) * 0.50
        elif hp < 0.38:
            hp = 0.38 - (0.38 - hp) * 0.50

        pick = home if hp >= 0.5 else away
        conf = max(hp, 1 - hp)

        # Edge — 경기 시작 30분 전부터는 캐시된 배당 사용
        game_id = g.get("game_id")
        locked = is_game_locked(g)
        odds_locked = False

        if locked and game_id in prev_cache:
            # 배당 잠금: 이전 캐시 사용
            ml_home = prev_cache[game_id]["ml_home"]
            ml_away = prev_cache[game_id]["ml_away"]
            odds_locked = True
        else:
            odds = match_odds(away, home)
            ml_home = odds["moneyline"]["home"] if odds and odds.get("moneyline", {}).get("home") else None
            ml_away = odds["moneyline"]["away"] if odds and odds.get("moneyline", {}).get("away") else None

        edge_info = {}
        if ml_home is not None and ml_away is not None:
            edge_info = analyze_moneyline(hp, ml_home, ml_away, KELLY_FRACTION)

        elapsed = time.time() - t0

        pred = {
            "game_id": g.get("game_id"),
            "date": date_str,
            "away": away,
            "home": home,
            "away_name": g.get("away_team_name", TEAMS.get(away, away)),
            "home_name": g.get("home_team_name", TEAMS.get(home, home)),
            "away_pitcher": ap_info.get("name", "TBD"),
            "home_pitcher": hp_info.get("name", "TBD"),
            "pick": pick,
            "conf": round(conf, 3),
            "home_prob": round(hp, 3),
            "raw_home_prob": round(raw_hp, 3),
            "avg_away": round(float(aw.mean()), 1),
            "avg_home": round(float(hw.mean()), 1),
            "avg_total": round(float(total.mean()), 1),
            "lineup_src": lineup_src,
            "status": g.get("status", ""),
            "game_time": g.get("game_time", ""),
            "sim_time": round(elapsed, 1),
        }

        # Odds & edge
        if ml_home is not None:
            pred["ml_home"] = ml_home
            pred["ml_away"] = ml_away
            pred["odds_locked"] = odds_locked
            if not odds_locked and odds:
                pred["ou_line"] = odds.get("total", {}).get("line")
        if edge_info:
            pred["edge_grade"] = edge_info["best_grade"]
            pred["edge_side"] = edge_info["best_side"]
            pred["edge_value"] = edge_info["best_edge"]
            pred["edge_kelly"] = edge_info["best_kelly"]
            pred["edge_ev"] = edge_info["best_ev"]
            pred["divergence"] = edge_info.get("divergence_level", "ok")
            pred["divergence_reasons"] = edge_info.get("divergence_reasons", [])

        predictions.append(pred)

    # Save cache
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    return predictions


# ============================================================
# F5 O/U grading
# ============================================================

# Wind/temp sensitive parks (no weather API yet → auto conservative)
F5_SENSITIVE_PARKS = {"COL", "CHC", "BOS", "CIN", "TEX"}

def grade_f5_ou(over_prob: float, sim_total: float, ou_line: float, home_team: str) -> str:
    """F5 O/U 등급. A/B만 추천, C는 관찰용.

    A: prob 60%+, gap 1.0+, 환경 민감 구장 제외
    B: prob 58%+, gap 0.7+
    C: 관찰용 (노출하되 추천 아님)
    """
    gap = abs(sim_total - ou_line)
    prob = max(over_prob, 1 - over_prob)
    direction = "OVER" if over_prob > 0.5 else "UNDER"

    if home_team in F5_SENSITIVE_PARKS:
        if prob >= 0.60 and gap >= 1.0:
            return "C"  # 민감 구장은 C까지만
        return "PASS"

    if prob >= 0.60 and gap >= 1.0:
        return "A"
    elif prob >= 0.58 and gap >= 0.7:
        return "B"
    elif prob >= 0.55 and gap >= 0.5:
        return "C"
    return "PASS"


def predict_f5_games(date_str: str, n_sims: int = 2000, force: bool = False) -> list[dict]:
    """F5 전용 예측. Full Game과 독립.

    max_innings=5로 시뮬, F5 ML + F5 O/U 별도 계산.
    """
    cache_path = os.path.join(PREDICTIONS_DIR, f"web_f5_{date_str}.json")
    if not force and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    data = _load_data()
    league_avg = data["league_avg"]
    pitchers_df = data["pitchers_df"]
    batters_df = data["batters_df"]
    speed_df = data["speed_df"]
    defense_df = data["defense_df"]
    bp_df = data["bp_df"]
    ti, tr, tp, to_ = data["tm_arrays"]
    alpha = PROBABILITY_DAMPENING_ALPHA

    _name_to_code = {}
    for code, full in TEAMS.items():
        _name_to_code[full.split()[-1].lower()] = code

    games = get_schedule(date_str)
    if not games:
        return []

    try:
        odds_data = get_mlb_odds()
    except Exception:
        odds_data = []

    def match_odds(away_code, home_code):
        for od in odds_data:
            oa = _name_to_code.get(od["away_team"].split()[-1].lower())
            oh = _name_to_code.get(od["home_team"].split()[-1].lower())
            if oa == away_code and oh == home_code:
                return od
        return None

    def get_def(team):
        m = defense_df[defense_df["team"] == team]
        return m.iloc[-1]["defense_bucket"] if len(m) > 0 else "avg"

    def get_bp(team):
        m = bp_df[bp_df["team"] == team]
        result = {}
        for _, row in m.iterrows():
            result[row["role"]] = {"player_id": f"bp_{team}_{row['role']}",
                "k_rate": row["k_rate"], "bb_rate": row["bb_rate"],
                "hr_rate": row.get("hr_rate", 0.03), "hbp_rate": row.get("hbp_rate", 0.01),
                "babip": row.get("babip", 0.300), "throws": "R"}
        for role in ["setup_early", "setup_late", "bridge", "closer"]:
            if role not in result:
                result[role] = dict(_fallback_pitcher(league_avg), player_id=f"bp_{team}_{role}")
        return result

    def build_lineup(player_ids, team):
        lineup = []
        n_api = 0
        for pid in player_ids[:9]:
            b = _get_batter(pid, batters_df)
            if b:
                lineup.append(b)
                n_api += 1
            else:
                lineup.append({
                    "player_id": int(pid), "name": f"#{pid}", "bats": "R", "pa": 100,
                    "k_rate": league_avg["k_rate"], "bb_rate": league_avg["bb_rate"],
                    "hbp_rate": league_avg["hbp_rate"], "hr_rate": league_avg["hr_rate"],
                    "babip": league_avg["babip"], "iso": 0.150,
                    "ld_rate": 0.21, "gb_rate": 0.43, "fb_rate": 0.34,
                    "iffb_rate": 0.10, "gdp_rate": 0.15,
                })
        return lineup, n_api

    predictions = []

    for g in games:
        away = g["away_team"]
        home = g["home_team"]
        ap_info = g.get("away_pitcher", {})
        hp_info = g.get("home_pitcher", {})

        if ap_info.get("name", "TBD") == "TBD" or hp_info.get("name", "TBD") == "TBD":
            continue
        if g.get("status") == "Postponed":
            continue

        t0 = time.time()

        asp = (_get_pitcher(ap_info.get("id"), pitchers_df, league_avg)
               or _get_pitcher_by_name(ap_info.get("name", ""), pitchers_df, league_avg)
               or _fallback_pitcher(league_avg))
        hsp = (_get_pitcher(hp_info.get("id"), pitchers_df, league_avg)
               or _get_pitcher_by_name(hp_info.get("name", ""), pitchers_df, league_avg)
               or _fallback_pitcher(league_avg))

        al_ids = [p["id"] for p in g.get("away_lineup", []) if p.get("id")]
        hl_ids = [p["id"] for p in g.get("home_lineup", []) if p.get("id")]

        if len(al_ids) >= 9:
            al, al_api = build_lineup(al_ids, away)
        else:
            al, al_api = [], 0
        if len(al) < 9:
            try:
                al = get_team_lineup(away, 2025)
                al_api = 0
            except Exception:
                continue

        if len(hl_ids) >= 9:
            hl, hl_api = build_lineup(hl_ids, home)
        else:
            hl, hl_api = [], 0
        if len(hl) < 9:
            try:
                hl = get_team_lineup(home, 2025)
                hl_api = 0
            except Exception:
                continue

        if len(al) < 9 or len(hl) < 9:
            continue

        total_api = al_api + hl_api
        if total_api == 18: lineup_src = "API"
        elif total_api >= 14: lineup_src = f"API({total_api}/18)"
        elif total_api > 0: lineup_src = f"PARTIAL({total_api}/18)"
        else: lineup_src = "default"

        asp_id = asp["player_id"] if isinstance(asp["player_id"], int) else None
        hsp_id = hsp["player_id"] if isinstance(hsp["player_id"], int) else None
        for b in al:
            pid = b.get("player_id")
            if pid:
                b["arsenal"] = get_batter_pitch_arsenal(pid)
                if hsp_id: b["h2h"] = get_h2h_stats(pid, hsp_id)
        for b in hl:
            pid = b.get("player_id")
            if pid:
                b["arsenal"] = get_batter_pitch_arsenal(pid)
                if asp_id: b["h2h"] = get_h2h_stats(pid, asp_id)

        away_bp = enhance_bullpen_profiles(get_bp(away), hl)
        home_bp = enhance_bullpen_profiles(get_bp(home), al)
        away_def, home_def = get_def(away), get_def(home)

        mc = {}
        for lineup, pitcher, is_home in [(al, hsp, False), (hl, asp, True)]:
            for batter in lineup:
                pid = batter.get("player_id")
                if not pid: continue
                probs = compute_matchup_v2(
                    batter=batter, pitcher=pitcher, league_avg=league_avg,
                    home_team=home, is_batter_home=is_home,
                    batter_arsenal=batter.get("arsenal"),
                    pitcher_arsenal=pitcher.get("arsenal"),
                    h2h_stats=batter.get("h2h"),
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

        # F5: max_innings=5
        scores = run_simulation_cpp(
            mc_arr, None, ti, tr, tp, to_, speeds, def_arr,
            n_sims=n_sims, max_innings=5,
            base_seed=g.get("game_id", 0) or hash(f"{away}{home}{date_str}f5") % 2**32,
        )

        aw = scores[:, 0].astype(float)
        hw = scores[:, 1].astype(float)
        total = aw + hw
        hwin = (hw > aw).sum()
        awin = (aw > hw).sum()
        ties = (hw == aw).sum()
        nt = hwin + awin
        raw_hp = hwin / nt if nt > 0 else 0.5
        hp = 0.50 + (raw_hp - 0.50) * alpha
        if hp > 0.62: hp = 0.62 + (hp - 0.62) * 0.50
        elif hp < 0.38: hp = 0.38 - (0.38 - hp) * 0.50

        pick = home if hp >= 0.5 else away
        conf = max(hp, 1 - hp)
        avg_total = float(total.mean())

        # F5 O/U at common lines
        f5_ou = {}
        for line in [3.5, 4.0, 4.5, 5.0, 5.5]:
            over_pct = float((total > line).mean())
            grade = grade_f5_ou(over_pct, avg_total, line, home)
            direction = "OVER" if over_pct > 0.5 else "UNDER"
            f5_ou[str(line)] = {
                "over_prob": round(over_pct, 3),
                "under_prob": round(1 - over_pct, 3),
                "direction": direction,
                "grade": grade,
            }

        # F5 ML edge
        odds = match_odds(away, home)
        edge_info = {}
        if odds and odds.get("moneyline", {}).get("home") and odds.get("moneyline", {}).get("away"):
            edge_info = analyze_moneyline(hp, odds["moneyline"]["home"], odds["moneyline"]["away"], KELLY_FRACTION)

        elapsed = time.time() - t0

        pred = {
            "game_id": g.get("game_id"),
            "date": date_str,
            "away": away, "home": home,
            "away_name": g.get("away_team_name", TEAMS.get(away, away)),
            "home_name": g.get("home_team_name", TEAMS.get(home, home)),
            "away_pitcher": ap_info.get("name", "TBD"),
            "home_pitcher": hp_info.get("name", "TBD"),
            "pick": pick, "conf": round(conf, 3),
            "home_prob": round(hp, 3),
            "avg_away": round(float(aw.mean()), 1),
            "avg_home": round(float(hw.mean()), 1),
            "avg_total": round(avg_total, 1),
            "tie_pct": round(float(ties / len(aw)), 3),
            "lineup_src": lineup_src,
            "status": g.get("status", ""),
            "game_time": g.get("game_time", ""),
            "sim_time": round(elapsed, 1),
            "f5_ou": f5_ou,
        }

        if odds:
            pred["ml_home"] = odds["moneyline"].get("home")
            pred["ml_away"] = odds["moneyline"].get("away")
        if edge_info:
            pred["edge_grade"] = edge_info["best_grade"]
            pred["edge_side"] = edge_info["best_side"]
            pred["edge_value"] = edge_info["best_edge"]
            pred["edge_kelly"] = edge_info["best_kelly"]
            pred["edge_ev"] = edge_info["best_ev"]
            pred["divergence"] = edge_info.get("divergence_level", "ok")

        predictions.append(pred)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    return predictions


def get_f5_results(date_str: str) -> dict:
    """F5 실제 결과."""
    import requests
    from data.mlb_api import TEAM_ID_TO_CODE

    raw_sc = pd.read_parquet(os.path.join(DATA_DIR, "statcast_raw_2025.parquet"))
    # For 2026 games, use MLB API linescore by inning
    resp = requests.get("https://statsapi.mlb.com/api/v1/schedule", params={
        "sportId": 1, "gameType": "R", "date": date_str,
        "hydrate": "linescore",
    }, timeout=10)
    data = resp.json()

    results = {}
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("detailedState") != "Final":
                continue
            game_pk = g["gamePk"]
            ls = g.get("linescore", {})
            innings = ls.get("innings", [])

            # Sum runs through 5 innings
            away_f5 = sum(inn.get("away", {}).get("runs", 0) for inn in innings[:5])
            home_f5 = sum(inn.get("home", {}).get("runs", 0) for inn in innings[:5])

            results[game_pk] = {
                "f5_away": away_f5,
                "f5_home": home_f5,
                "f5_total": away_f5 + home_f5,
                "f5_winner": TEAM_ID_TO_CODE.get(g["teams"]["home"]["team"]["id"], "???") if home_f5 > away_f5
                             else (TEAM_ID_TO_CODE.get(g["teams"]["away"]["team"]["id"], "???") if away_f5 > home_f5
                                   else "TIE"),
            }

    return results


def get_results(date_str: str) -> dict:
    """실제 결과 가져와서 적중 검증. gamePk 기반 (더블헤더 대응)."""
    import requests
    from data.mlb_api import TEAM_ID_TO_CODE

    resp = requests.get("https://statsapi.mlb.com/api/v1/schedule", params={
        "sportId": 1, "gameType": "R", "date": date_str,
        "hydrate": "linescore",
    }, timeout=10)
    data = resp.json()

    results = {}
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("detailedState") != "Final":
                continue
            ls = g.get("linescore", {}).get("teams", {})
            ar = ls.get("away", {}).get("runs", 0)
            hr = ls.get("home", {}).get("runs", 0)
            game_pk = g["gamePk"]
            results[game_pk] = {
                "away_score": ar,
                "home_score": hr,
                "total": ar + hr,
                "winner": TEAM_ID_TO_CODE.get(g["teams"]["home"]["team"]["id"], "???") if hr > ar
                          else TEAM_ID_TO_CODE.get(g["teams"]["away"]["team"]["id"], "???"),
            }

    return results
