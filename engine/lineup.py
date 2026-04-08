"""라인업 구성 및 불펜 평균 스탯 계산."""

import os
import pandas as pd
import numpy as np

from config import DATA_DIR, SEASONS


_batters_cache = None
_pitchers_cache = None
_batter_arsenal_cache = None
_pitcher_arsenal_cache = None
_h2h_cache = None
_splits_cache = {}  # {"splits_batter_home_away": DataFrame, ...}


def _load_batters() -> pd.DataFrame:
    global _batters_cache
    if _batters_cache is None:
        path = os.path.join(DATA_DIR, "batters_processed.parquet")
        _batters_cache = pd.read_parquet(path)
    return _batters_cache


def _load_pitchers() -> pd.DataFrame:
    global _pitchers_cache
    if _pitchers_cache is None:
        path = os.path.join(DATA_DIR, "pitchers_processed.parquet")
        _pitchers_cache = pd.read_parquet(path)
    return _pitchers_cache


def get_league_avg(season: int | None = None) -> dict:
    """리그 평균 로드."""
    path = os.path.join(DATA_DIR, "league_averages.parquet")
    df = pd.read_parquet(path)
    if season and season in df["season"].values:
        row = df[df["season"] == season].iloc[0]
    else:
        row = df.iloc[-1]  # 가장 최근 시즌

    return {
        "k_rate": row["k_rate"],
        "bb_rate": row["bb_rate"],
        "hbp_rate": row["hbp_rate"],
        "hr_rate": row["hr_rate"],
        "babip": row["babip"],
        "ld_rate": 0.21,
        "gb_rate": 0.43,
        "fb_rate": 0.34,
        "iffb_rate": 0.10,
    }


def get_lineup_by_ids(player_ids: list[int]) -> list[dict]:
    """MLB API에서 가져온 player_id 리스트로 라인업 구성.

    각 선수의 최신 시즌 Statcast 데이터를 매칭.
    데이터가 없는 선수는 리그 평균으로 대체.
    """
    batters = _load_batters()
    league_avg = get_league_avg()
    lineup = []

    for pid in player_ids:
        # 해당 선수의 모든 시즌 데이터에서 최신 것
        matches = batters[batters["player_id"] == pid].sort_values("season", ascending=False)

        if len(matches) > 0:
            lineup.append(matches.iloc[0].to_dict())
        else:
            # 데이터 없는 신인 등 → 리그 평균
            lineup.append({
                "player_id": pid,
                "name": f"Player #{pid}",
                "pa": 100,
                "k_rate": league_avg["k_rate"],
                "bb_rate": league_avg["bb_rate"],
                "hbp_rate": league_avg["hbp_rate"],
                "hr_rate": league_avg["hr_rate"],
                "babip": league_avg["babip"],
                "iso": 0.150,
                "ld_rate": league_avg["ld_rate"],
                "gb_rate": league_avg["gb_rate"],
                "fb_rate": league_avg["fb_rate"],
                "iffb_rate": league_avg.get("iffb_rate", 0.10),
                "bats": "R",
                "gdp_rate": 0.15,
            })

    return lineup


def get_team_lineup(team: str, season: int | None = None) -> list[dict]:
    """팀의 기본 라인업 (current_team 기준, PA 상위 9명)."""
    batters = _load_batters()

    if season is None:
        season = max(SEASONS)

    # current_team이 있으면 우선 사용 (로스터 업데이트 반영)
    team_col = "current_team" if "current_team" in batters.columns else "team"

    team_batters = batters[(batters[team_col] == team) & (batters["season"] == season)]

    if len(team_batters) < 9:
        for s in sorted(SEASONS, reverse=True):
            if s == season:
                continue
            more = batters[(batters[team_col] == team) & (batters["season"] == s)]
            team_batters = pd.concat([team_batters, more]).drop_duplicates(subset=["player_id"], keep="first")
            if len(team_batters) >= 9:
                break

    # PA 상위 9명
    team_batters = team_batters.nlargest(9, "pa")

    if len(team_batters) < 9:
        # 그래도 부족하면 리그평균 타자로 채움
        avg = get_league_avg(season)
        filler = {
            "name": "League Average",
            "team": team,
            "pa": 500,
            "k_rate": avg["k_rate"],
            "bb_rate": avg["bb_rate"],
            "hbp_rate": avg["hbp_rate"],
            "hr_rate": avg["hr_rate"],
            "babip": avg["babip"],
            "iso": 0.150,
            "ld_rate": avg["ld_rate"],
            "gb_rate": avg["gb_rate"],
            "fb_rate": avg["fb_rate"],
            "iffb_rate": avg["iffb_rate"],
            "bats": "R",
            "gdp_rate": 0.15,
        }
        lineup = team_batters.to_dict("records")
        while len(lineup) < 9:
            lineup.append(dict(filler))
        return lineup

    return team_batters.to_dict("records")


def get_team_pitchers(team: str, season: int | None = None) -> list[dict]:
    """팀의 선발투수 목록 (current_team 기준, GS > 0)."""
    pitchers = _load_pitchers()

    if season is None:
        season = max(SEASONS)

    team_col = "current_team" if "current_team" in pitchers.columns else "team"

    # 현재 로스터 기준으로 찾되, 모든 시즌 데이터에서 최신 우선
    team_p = pitchers[pitchers[team_col] == team]

    # 선수별 최신 시즌만 남기기
    team_p = team_p.sort_values("season", ascending=False).drop_duplicates("player_id", keep="first")

    starters = team_p[team_p["gs"] > 0].sort_values("gs", ascending=False)

    if len(starters) == 0:
        starters = team_p.nlargest(5, "tbf")

    return starters.to_dict("records")


def get_pitcher_by_id(player_id: int) -> dict | None:
    """player_id로 투수 찾기 (팀 무관, 최신 시즌 우선).

    트레이드 선수도 찾을 수 있음.
    """
    pitchers = _load_pitchers()
    matches = pitchers[pitchers["player_id"] == player_id]

    if len(matches) == 0:
        return None

    # 최신 시즌 우선
    matches = matches.sort_values("season", ascending=False)
    return matches.iloc[0].to_dict()


def _make_bullpen_avg(relievers: pd.DataFrame, name: str, season: int | None = None) -> dict:
    """릴리버 그룹의 TBF 가중평균 스탯."""
    total_tbf = relievers["tbf"].sum()
    if total_tbf == 0:
        avg = get_league_avg(season)
        return {
            "name": name,
            "k_rate": avg["k_rate"],
            "bb_rate": avg["bb_rate"],
            "hbp_rate": avg["hbp_rate"],
            "hr_rate": avg["hr_rate"],
            "babip": avg["babip"],
            "ld_rate": avg["ld_rate"],
            "gb_rate": avg["gb_rate"],
            "fb_rate": avg["fb_rate"],
            "iffb_rate": avg["iffb_rate"],
            "throws": "R",
        }

    def wavg(col):
        return (relievers[col] * relievers["tbf"]).sum() / total_tbf

    return {
        "name": name,
        "k_rate": wavg("k_rate"),
        "bb_rate": wavg("bb_rate"),
        "hbp_rate": wavg("hbp_rate"),
        "hr_rate": wavg("hr_rate"),
        "babip": wavg("babip"),
        "ld_rate": wavg("ld_rate"),
        "gb_rate": wavg("gb_rate"),
        "fb_rate": wavg("fb_rate"),
        "iffb_rate": wavg("iffb_rate"),
        "throws": "R",
    }


def get_bullpen_avg(team: str, season: int | None = None) -> dict:
    """팀 불펜 전체 평균 (current_team 기준)."""
    pitchers = _load_pitchers()
    if season is None:
        season = max(SEASONS)
    team_col = "current_team" if "current_team" in pitchers.columns else "team"
    team_p = pitchers[pitchers[team_col] == team]
    team_p = team_p.sort_values("season", ascending=False).drop_duplicates("player_id", keep="first")
    relievers = team_p[team_p["gs"] == 0]
    if len(relievers) == 0:
        relievers = team_p
    return _make_bullpen_avg(relievers, f"{team} Bullpen", season)


def get_bullpen_tiered(team: str, season: int | None = None) -> dict:
    """불펜을 클로저급/셋업급 2티어로 분리.

    - closer: K% 상위 25% 릴리버 (최소 TBF 30+)
    - setup: 나머지 릴리버

    Returns:
        {"closer": {...스탯...}, "setup": {...스탯...}}
    """
    pitchers = _load_pitchers()
    if season is None:
        season = max(SEASONS)
    team_col = "current_team" if "current_team" in pitchers.columns else "team"
    team_p = pitchers[pitchers[team_col] == team]
    team_p = team_p.sort_values("season", ascending=False).drop_duplicates("player_id", keep="first")
    relievers = team_p[(team_p["gs"] == 0) & (team_p["tbf"] >= 30)].copy()

    if len(relievers) < 2:
        # 릴리버가 너무 적으면 전체 평균으로
        bp = get_bullpen_avg(team, season)
        return {"closer": bp, "setup": bp}

    # K% 상위 25%를 클로저급으로 분류
    k_threshold = relievers["k_rate"].quantile(0.75)
    closers = relievers[relievers["k_rate"] >= k_threshold]
    setups = relievers[relievers["k_rate"] < k_threshold]

    if len(closers) == 0:
        closers = relievers.nlargest(2, "k_rate")
    if len(setups) == 0:
        setups = relievers

    return {
        "closer": _make_bullpen_avg(closers, f"{team} Closer", season),
        "setup": _make_bullpen_avg(setups, f"{team} Setup", season),
    }


def get_available_teams() -> list[dict]:
    """데이터가 있는 팀 목록 (current_team 기준)."""
    batters = _load_batters()
    team_col = "current_team" if "current_team" in batters.columns else "team"
    teams = batters[team_col].unique()
    from config import TEAMS
    result = []
    for code in sorted(teams):
        result.append({
            "code": code,
            "name": TEAMS.get(code, code),
        })
    return result


def _wavg_merge(df, group_cols, weight_col, avg_cols, first_cols):
    """PA 가중평균으로 멀티시즌 병합 (벡터 연산)."""
    # 가중합 계산
    for col in avg_cols:
        df[f"_w_{col}"] = df[col] * df[weight_col]

    agg_dict = {weight_col: "sum"}
    for col in avg_cols:
        agg_dict[f"_w_{col}"] = "sum"
    for col in first_cols:
        agg_dict[col] = "first"
    if "n_pitches" in df.columns and "n_pitches" not in group_cols:
        agg_dict["n_pitches"] = "sum"

    merged = df.groupby(group_cols, as_index=False).agg(agg_dict)

    # 가중평균 복원
    for col in avg_cols:
        merged[col] = merged[f"_w_{col}"] / merged[weight_col].clip(lower=1)
        merged.drop(columns=[f"_w_{col}"], inplace=True)

    return merged


def _load_batter_arsenal() -> pd.DataFrame:
    global _batter_arsenal_cache
    if _batter_arsenal_cache is None:
        frames = []
        for year in SEASONS:
            path = os.path.join(DATA_DIR, f"pitch_arsenal_batters_{year}.parquet")
            if os.path.exists(path):
                frames.append(pd.read_parquet(path))
        if frames:
            df = pd.concat(frames, ignore_index=True)
            _batter_arsenal_cache = _wavg_merge(
                df,
                group_cols=["batter_id", "pitch_type"],
                weight_col="n_pa",
                avg_cols=["swing_rate", "whiff_rate", "k_rate", "bb_rate", "hr_rate",
                          "hbp_rate", "babip", "ld_rate", "gb_rate", "fb_rate", "xwoba"],
                first_cols=["name", "bats"],
            )
        else:
            _batter_arsenal_cache = pd.DataFrame()
    return _batter_arsenal_cache


def _load_pitcher_arsenal() -> pd.DataFrame:
    global _pitcher_arsenal_cache
    if _pitcher_arsenal_cache is None:
        frames = []
        for year in SEASONS:
            path = os.path.join(DATA_DIR, f"pitch_arsenal_pitchers_{year}.parquet")
            if os.path.exists(path):
                frames.append(pd.read_parquet(path))
        if frames:
            df = pd.concat(frames, ignore_index=True)
            merged = _wavg_merge(
                df,
                group_cols=["pitcher_id", "pitch_type"],
                weight_col="n_pa",
                avg_cols=["whiff_rate", "k_rate", "bb_rate", "hr_rate",
                          "hbp_rate", "babip", "ld_rate", "gb_rate", "fb_rate", "xwoba"],
                first_cols=["throws", "team"],
            )
            # usage 재계산: 투수별 총 투구 대비 구종별 비율
            pitcher_totals = merged.groupby("pitcher_id")["n_pitches"].transform("sum")
            merged["usage"] = merged["n_pitches"] / pitcher_totals.clip(lower=1)
            _pitcher_arsenal_cache = merged
        else:
            _pitcher_arsenal_cache = pd.DataFrame()
    return _pitcher_arsenal_cache


def _load_h2h() -> pd.DataFrame:
    global _h2h_cache
    if _h2h_cache is None:
        frames = []
        for year in SEASONS:
            path = os.path.join(DATA_DIR, f"h2h_matchups_{year}.parquet")
            if os.path.exists(path):
                frames.append(pd.read_parquet(path))
        if frames:
            df = pd.concat(frames, ignore_index=True)
            _h2h_cache = _wavg_merge(
                df,
                group_cols=["batter_id", "pitcher_id"],
                weight_col="n_pa",
                avg_cols=["k_rate", "bb_rate", "hr_rate", "hbp_rate", "babip", "iso",
                          "ld_rate", "gb_rate", "fb_rate"],
                first_cols=["batter_name", "bats", "throws"],
            )
        else:
            _h2h_cache = pd.DataFrame()
    return _h2h_cache


def get_batter_pitch_arsenal(batter_id: int) -> list[dict]:
    """타자의 구종별 성적."""
    df = _load_batter_arsenal()
    if df.empty:
        return []
    rows = df[df["batter_id"] == batter_id]
    return rows.to_dict("records")


def get_pitcher_pitch_arsenal(pitcher_id: int) -> list[dict]:
    """투수의 구종 비율 + 구종별 피안타 성적."""
    df = _load_pitcher_arsenal()
    if df.empty:
        return []
    rows = df[df["pitcher_id"] == pitcher_id]
    return rows.to_dict("records")


def get_h2h_stats(batter_id: int, pitcher_id: int) -> dict | None:
    """특정 타자-투수 직접 대전 기록."""
    df = _load_h2h()
    if df.empty:
        return None
    row = df[(df["batter_id"] == batter_id) & (df["pitcher_id"] == pitcher_id)]
    if len(row) == 0:
        return None
    return row.iloc[0].to_dict()


def _load_split(name: str) -> pd.DataFrame:
    """스플릿 데이터 로드 (캐시)."""
    global _splits_cache
    if name not in _splits_cache:
        path = os.path.join(DATA_DIR, f"{name}.parquet")
        if os.path.exists(path):
            _splits_cache[name] = pd.read_parquet(path)
        else:
            _splits_cache[name] = pd.DataFrame()
    return _splits_cache[name]


def get_batter_splits(batter_id: int) -> dict:
    """타자의 모든 스플릿 데이터 조회.

    Returns:
        {
            "home_away": {"home": {...stats}, "away": {...stats}},
            "runners": {"empty": {...}, "runners_on": {...}, "risp": {...}},
            "platoon": {"L": {...}, "R": {...}},
            "count": {"ahead": {...}, "behind": {...}, ...},
        }
    """
    result = {}

    # 홈/원정
    df = _load_split("splits_batter_home_away")
    if not df.empty:
        rows = df[df["batter"] == batter_id]
        result["home_away"] = {r["batter_home_away"]: r.to_dict() for _, r in rows.iterrows()}

    # 주자 상황
    df = _load_split("splits_batter_runners")
    if not df.empty:
        rows = df[df["batter"] == batter_id]
        result["runners"] = {r["runner_state"]: r.to_dict() for _, r in rows.iterrows()}

    # 좌우 개인 스플릿
    df = _load_split("splits_batter_platoon")
    if not df.empty:
        rows = df[df["batter"] == batter_id]
        result["platoon"] = {r["p_throws"]: r.to_dict() for _, r in rows.iterrows()}

    # 카운트
    df = _load_split("splits_batter_count")
    if not df.empty:
        rows = df[df["batter"] == batter_id]
        result["count"] = {r["count_state"]: r.to_dict() for _, r in rows.iterrows()}

    # 월별
    df = _load_split("splits_batter_month")
    if not df.empty:
        rows = df[df["batter"] == batter_id]
        result["month"] = {int(r["month"]): r.to_dict() for _, r in rows.iterrows()}

    return result


def get_pitcher_splits(pitcher_id: int) -> dict:
    """투수의 모든 스플릿 데이터."""
    result = {}

    df = _load_split("splits_pitcher_home_away")
    if not df.empty:
        rows = df[df["pitcher"] == pitcher_id]
        result["home_away"] = {r["pitcher_home_away"]: r.to_dict() for _, r in rows.iterrows()}

    df = _load_split("splits_pitcher_runners")
    if not df.empty:
        rows = df[df["pitcher"] == pitcher_id]
        result["runners"] = {r["runner_state"]: r.to_dict() for _, r in rows.iterrows()}

    df = _load_split("splits_pitcher_platoon")
    if not df.empty:
        rows = df[df["pitcher"] == pitcher_id]
        result["platoon"] = {r["stand"]: r.to_dict() for _, r in rows.iterrows()}

    df = _load_split("splits_pitcher_count")
    if not df.empty:
        rows = df[df["pitcher"] == pitcher_id]
        result["count"] = {r["count_state"]: r.to_dict() for _, r in rows.iterrows()}

    df = _load_split("splits_pitcher_month")
    if not df.empty:
        rows = df[df["pitcher"] == pitcher_id]
        result["month"] = {int(r["month"]): r.to_dict() for _, r in rows.iterrows()}

    return result


def clear_cache():
    """캐시 클리어 (데이터 새로고침 시)."""
    global _batters_cache, _pitchers_cache, _batter_arsenal_cache, _pitcher_arsenal_cache, _h2h_cache
    _batters_cache = None
    _pitchers_cache = None
    _batter_arsenal_cache = None
    _pitcher_arsenal_cache = None
    _h2h_cache = None
