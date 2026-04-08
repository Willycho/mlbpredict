"""V3 ML Model — XGBoost 기반 F5 예측.

기존 피처들(투수스코어, 매치업, 타선화력, 소화력, pvt)을
ML 모델이 최적 가중치로 학습.
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR, SEASONS
from engine.pitcher_score import score_pitcher, _build_pitcher_pool
from engine.matchup_score import (
    _load_data as _load_matchup_data,
    compute_individual_matchups, compute_h2h_bonus,
    compute_pitcher_vs_team_bonus, _resolve_lineup,
)
from engine.game_score import get_team_offense_score, get_pitcher_durability_score

MODEL_PATH = os.path.join(DATA_DIR, "v3_xgb_model.pkl")


def extract_features_for_game(
    away_pitcher_id: int,
    home_pitcher_id: int,
    away_team: str,
    home_team: str,
) -> dict | None:
    """경기 1개에 대한 피처 추출."""
    try:
        ap = score_pitcher(away_pitcher_id, is_home=False)
        hp = score_pitcher(home_pitcher_id, is_home=True)

        a_bids, a_src = _resolve_lineup(home_team)
        h_bids, h_src = _resolve_lineup(away_team)

        a_arsenal, _, _ = compute_individual_matchups(away_pitcher_id, a_bids)
        h_arsenal, _, _ = compute_individual_matchups(home_pitcher_id, h_bids)

        a_h2h, _ = compute_h2h_bonus(away_pitcher_id, home_team, a_bids)
        h_h2h, _ = compute_h2h_bonus(home_pitcher_id, away_team, h_bids)

        a_pvt, _ = compute_pitcher_vs_team_bonus(away_pitcher_id, home_team)
        h_pvt, _ = compute_pitcher_vs_team_bonus(home_pitcher_id, away_team)

        ao = get_team_offense_score(away_team)
        ho = get_team_offense_score(home_team)
        ad = get_pitcher_durability_score(away_pitcher_id)
        hd = get_pitcher_durability_score(home_pitcher_id)

        return {
            # 투수 종합 점수
            "away_pitcher_score": ap["total_score"],
            "home_pitcher_score": hp["total_score"],
            "pitcher_gap": hp["total_score"] - ap["total_score"],
            # 서브스코어
            "away_season": ap["season_score"],
            "home_season": hp["season_score"],
            "away_arsenal": ap["arsenal_score"],
            "home_arsenal": hp["arsenal_score"],
            # 매치업
            "away_matchup_arsenal": a_arsenal,
            "home_matchup_arsenal": h_arsenal,
            "matchup_gap": h_arsenal - a_arsenal,
            # H2H
            "away_h2h": a_h2h,
            "home_h2h": h_h2h,
            # 투수 vs 팀 통산
            "away_pvt": a_pvt,
            "home_pvt": h_pvt,
            # 타선 화력
            "away_offense": ao,
            "home_offense": ho,
            "offense_gap": ho - ao,
            # 소화력
            "away_durability": ad,
            "home_durability": hd,
        }
    except Exception:
        return None


def build_training_data(seasons: list[int] = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """과거 시즌 데이터로 학습 데이터셋 구축.

    Returns:
        (features_df, ml_labels, ou_labels)
        ml_labels: 1 = home wins F5, 0 = away wins F5, -1 = tie
        ou_labels: 1 = over 4.5, 0 = under 4.5
    """
    if seasons is None:
        seasons = SEASONS

    all_features = []
    ml_labels = []
    ou_labels = []

    for year in seasons:
        raw_path = os.path.join(DATA_DIR, f"statcast_raw_{year}.parquet")
        if not os.path.exists(raw_path):
            continue

        raw = pd.read_parquet(raw_path, columns=[
            "game_pk", "game_date", "pitcher", "batter", "events",
            "home_team", "away_team", "inning", "inning_topbot",
            "post_bat_score", "bat_score",
        ])

        # 경기별 선발투수 + F5 결과 추출
        games = raw.groupby("game_pk")
        game_list = []

        for gpk, gdata in games:
            # F5 이닝만
            f5 = gdata[gdata["inning"] <= 5]
            if f5.empty:
                continue

            # 선발투수: 1이닝에 처음 등판한 투수
            inn1_top = f5[(f5["inning"] == 1) & (f5["inning_topbot"] == "Top")]
            inn1_bot = f5[(f5["inning"] == 1) & (f5["inning_topbot"] == "Bot")]

            if inn1_top.empty or inn1_bot.empty:
                continue

            home_sp = inn1_top["pitcher"].iloc[0]  # Top = away batting, pitcher is home
            away_sp = inn1_bot["pitcher"].iloc[0]  # Bot = home batting, pitcher is away

            home_team = gdata["home_team"].iloc[0]
            away_team = gdata["away_team"].iloc[0]

            # F5 득점 계산
            f5_events = f5[f5["events"].notna()]
            if f5_events.empty:
                continue

            # inning 5 끝의 점수로 계산
            last_row = f5.iloc[-1]
            # post_bat_score는 타자 팀 점수이므로...
            # 좀 더 정확하게: 5이닝까지의 run scoring events 세기
            f5_top = f5[f5["inning_topbot"] == "Top"]
            f5_bot = f5[f5["inning_topbot"] == "Bot"]

            # 각 이닝의 마지막 행에서 점수 추출
            away_runs = 0
            home_runs = 0
            for inn in range(1, 6):
                inn_top = f5_top[f5_top["inning"] == inn]
                inn_bot = f5_bot[f5_bot["inning"] == inn]
                if not inn_top.empty:
                    last = inn_top.iloc[-1]
                    away_runs = int(last["bat_score"])
                if not inn_bot.empty:
                    last = inn_bot.iloc[-1]
                    home_runs = int(last["bat_score"])

            game_list.append({
                "game_pk": gpk,
                "away_sp": away_sp,
                "home_sp": home_sp,
                "away_team": away_team,
                "home_team": home_team,
                "f5_away": away_runs,
                "f5_home": home_runs,
                "f5_total": away_runs + home_runs,
            })

        print(f"  {year}: {len(game_list)} games extracted")

        # 피처 추출
        for g in game_list:
            feats = extract_features_for_game(
                g["away_sp"], g["home_sp"], g["away_team"], g["home_team"]
            )
            if feats is None:
                continue

            all_features.append(feats)

            # ML label: home wins = 1, away wins = 0, tie = -1
            if g["f5_home"] > g["f5_away"]:
                ml_labels.append(1)
            elif g["f5_away"] > g["f5_home"]:
                ml_labels.append(0)
            else:
                ml_labels.append(-1)

            # OU label: over 4.5 = 1, under = 0
            ou_labels.append(1 if g["f5_total"] > 4.5 else 0)

    df = pd.DataFrame(all_features)
    return df, np.array(ml_labels), np.array(ou_labels)


def train_model(X: pd.DataFrame, y_ml: np.ndarray, y_ou: np.ndarray):
    """XGBoost 모델 학습.

    ML (승패)과 OU (언오버) 각각 학습.
    tie(-1)는 ML 학습에서 제외.
    """
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score

    # ML 모델 (tie 제외)
    mask = y_ml != -1
    X_ml = X[mask]
    y_ml_clean = y_ml[mask]

    ml_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    ml_scores = cross_val_score(ml_model, X_ml, y_ml_clean, cv=5, scoring="accuracy")
    print(f"ML CV accuracy: {ml_scores.mean():.3f} (+/- {ml_scores.std():.3f})")

    ml_model.fit(X_ml, y_ml_clean)

    # OU 모델
    ou_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    ou_scores = cross_val_score(ou_model, X, y_ou, cv=5, scoring="accuracy")
    print(f"OU CV accuracy: {ou_scores.mean():.3f} (+/- {ou_scores.std():.3f})")

    ou_model.fit(X, y_ou)

    # Feature importance
    print(f"\n=== ML Feature Importance ===")
    for name, imp in sorted(zip(X.columns, ml_model.feature_importances_), key=lambda x: -x[1]):
        if imp > 0.01:
            print(f"  {name:25s}: {imp:.3f}")

    print(f"\n=== OU Feature Importance ===")
    for name, imp in sorted(zip(X.columns, ou_model.feature_importances_), key=lambda x: -x[1]):
        if imp > 0.01:
            print(f"  {name:25s}: {imp:.3f}")

    # 저장
    model_data = {
        "ml_model": ml_model,
        "ou_model": ou_model,
        "feature_names": list(X.columns),
        "ml_cv": float(ml_scores.mean()),
        "ou_cv": float(ou_scores.mean()),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\nModel saved to {MODEL_PATH}")

    return model_data


if __name__ == "__main__":
    print("Building training data...")
    X, y_ml, y_ou = build_training_data()
    print(f"\nDataset: {len(X)} games, {X.shape[1]} features")
    print(f"ML: home_win={sum(y_ml==1)}, away_win={sum(y_ml==0)}, tie={sum(y_ml==-1)}")
    print(f"OU: over={sum(y_ou==1)}, under={sum(y_ou==0)}")

    print("\nTraining...")
    train_model(X, y_ml, y_ou)
