"""파크팩터, 리그 상수, 시뮬레이션 설정."""

import os

SEASONS = [2023, 2024, 2025]
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE_DIR, "data", "storage")

# 30개 MLB 팀 코드 → 풀네임 (Statcast 코드 기준)
TEAMS = {
    "AZ": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "ATH": "Oakland Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres",
    "SF": "San Francisco Giants",
    "SEA": "Seattle Mariners",
    "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSH": "Washington Nationals",
}

# 파크팩터 (runs factor, 100 = neutral)
# 출처: ESPN Park Factors 2022-2024 평균
PARK_FACTORS = {
    "AZ": 104,
    "ATL": 101,
    "BAL": 102,
    "BOS": 104,
    "CHC": 102,
    "CWS": 101,
    "CIN": 106,
    "CLE": 97,
    "COL": 114,
    "DET": 96,
    "HOU": 100,
    "KC": 100,
    "LAA": 98,
    "LAD": 97,
    "MIA": 94,
    "MIL": 102,
    "MIN": 101,
    "NYM": 97,
    "NYY": 104,
    "ATH": 96,
    "PHI": 103,
    "PIT": 96,
    "SD": 95,
    "SF": 96,
    "SEA": 96,
    "STL": 98,
    "TB": 97,
    "TEX": 103,
    "TOR": 101,
    "WSH": 100,
}

# HR 파크팩터 (별도, HR에 더 민감한 구장이 있음)
PARK_HR_FACTORS = {
    "AZ": 107,
    "ATL": 103,
    "BAL": 109,
    "BOS": 101,
    "CHC": 106,
    "CWS": 104,
    "CIN": 113,
    "CLE": 97,
    "COL": 112,
    "DET": 95,
    "HOU": 100,
    "KC": 97,
    "LAA": 99,
    "LAD": 95,
    "MIA": 89,
    "MIL": 107,
    "MIN": 103,
    "NYM": 96,
    "NYY": 112,
    "ATH": 93,
    "PHI": 108,
    "PIT": 91,
    "SD": 92,
    "SF": 93,
    "SEA": 94,
    "STL": 97,
    "TB": 95,
    "TEX": 106,
    "TOR": 103,
    "WSH": 100,
}

# 좌우 스플릿 보정 계수 (모집단 평균)
# v2.3: ablation 결과 platoon OFF가 Brier -0.003, Resolution +44% 개선
# 현재 OFF (all 1.0). 시즌 중반 재평가 시 HALF 검토 가능.
PLATOON_ENABLED = False
PLATOON_MULTIPLIERS = {
    "same_hand": {
        "k_rate": 1.0,
        "bb_rate": 1.0,
        "hr_rate": 1.0,
        "babip": 1.0,
    },
    "opp_hand": {
        "k_rate": 1.0,
        "bb_rate": 1.0,
        "hr_rate": 1.0,
        "babip": 1.0,
    },
}

# 타구 유형별 기본 BABIP (리그 평균 근사치)
BABIP_BY_BATTED_BALL = {
    "ld": 0.680,   # 라인드라이브
    "gb": 0.239,   # 땅볼
    "fb": 0.207,   # 플라이볼 (HR 제외)
    "iffb": 0.005, # 인필드 팝업
}

# 안타 유형 분배 (타구 유형별)
# 안타가 된 경우, 1B/2B/3B 비율
HIT_TYPE_DISTRIBUTION = {
    "gb": {"1B": 0.97, "2B": 0.03, "3B": 0.00},
    "ld": {"1B": 0.78, "2B": 0.20, "3B": 0.02},
    "fb": {"1B": 0.52, "2B": 0.35, "3B": 0.13},
}

# 주루 진루 확률 (기본값, 스프린트 스피드로 보정)
BASE_ADVANCEMENT = {
    "single": {
        "first_to_third": 0.29,  # 1루주자 → 3루
        "second_to_home": 0.59,  # 2루주자 → 홈
    },
    "double": {
        "first_to_home": 0.60,  # 1루주자 → 홈
    },
    "ground_out": {
        "double_play_rate": 0.12,     # 실제 MLB ~12%
        "second_to_third": 0.50,      # 2루주자 → 3루
        "third_to_home_less2": 0.55,  # 실제 MLB ~55%
    },
    "fly_out": {
        "third_to_home_sac": 0.45,  # 실제 MLB ~45%
    },
}

# 투수 피로 설정
PITCHER_STAMINA = {
    "starter_pa_limit": 27,         # 선발 대면 타자 수 상한 (~6이닝)
    "fatigue_threshold": 0.80,      # 80% 넘으면 피로 적용
    "fatigue_k_penalty": 0.05,      # K% 5% 감소
    "fatigue_bb_penalty": 0.05,     # BB% 5% 증가
}

# 기본 시뮬레이션 설정
DEFAULT_N_SIMULATIONS = 10000
MANFRED_RUNNER_INNING = 10  # 10회부터 2루 고스트 러너

# 리그 평균 스프린트 스피드 (ft/s)
LEAGUE_AVG_SPRINT_SPEED = 27.0


# ============================================================
# V2 Engine Config
# ============================================================

V2_DATA_DIR = os.path.join(DATA_DIR, "v2")

# Count buckets: (balls, strikes) → bucket name
COUNT_BUCKET_MAP = {
    (0, 0): "first_pitch",
    (1, 0): "hitter_ahead",
    (2, 0): "hitter_ahead",
    (3, 0): "hitter_ahead",
    (2, 1): "hitter_ahead",
    (3, 1): "hitter_ahead",
    (0, 1): "pitcher_ahead",
    (1, 1): "pitcher_ahead",
    (0, 2): "two_strike",
    (1, 2): "two_strike",
    (2, 2): "two_strike",
    (3, 2): "two_strike",
}

COUNT_BUCKETS = ["first_pitch", "hitter_ahead", "pitcher_ahead", "two_strike"]

# Shrinkage constants (k값): n/(n+k) 공식에서 사용
# v2.1: overconfidence 보정으로 k값 상향 조정
SHRINKAGE = {
    "batter_rate": 350,       # K%, BB% (v2.0: 200 → 350)
    "batter_hr": 500,         # HR% (v2.0: 300 → 500)
    "batter_babip": 600,      # BABIP (v2.0: 400 → 600)
    "pitcher_rate": 250,      # 투수 K%, BB% (v2.0: 150 → 250)
    "pitcher_hr": 500,        # 투수 HR% (v2.0: 350 → 500)
    "h2h": 80,                # H2H 직접 대전 (v2.0: 50 → 80)
    "split_ratio": 120,       # platoon/RISP/home-away 스플릿 (v2.0: 80 → 120)
    "count_pitch_mix": 50,    # count별 pitch mix
}

# Raw probability dampening: damped = 0.50 + (raw - 0.50) * alpha
# 1.0 = no dampening, 0.5 = max dampening
PROBABILITY_DAMPENING_ALPHA = 0.65

# Statcast events → v2 simplified event types
# field_out, force_out, double_play 등은 bb_type 기반으로 별도 분류
EVENT_MAP = {
    "strikeout": "K",
    "strikeout_double_play": "K",
    "walk": "BB",
    "intent_walk": "BB",
    "hit_by_pitch": "HBP",
    "home_run": "HR",
    "single": "1B",
    "double": "2B",
    "triple": "3B",
    "field_error": "ROE",
    "fielders_choice": "FC",
    "fielders_choice_out": "FC",
    "sac_bunt": "SAC",
    "sac_bunt_double_play": "SAC",
    "sac_fly": "FO",
    "sac_fly_double_play": "FO",
}

# bb_type 기반 아웃 분류 (EVENT_MAP에 없는 field_out/force_out/double_play 등)
BB_TYPE_OUT_MAP = {
    "ground_ball": "GO",
    "fly_ball": "FO",
    "line_drive": "LO",
    "popup": "FO",
}

# BIP 내 FC/ROE 비율 (Statcast 2023-2025 실측)
BIP_FC_RATE = 0.007    # ~0.7%
BIP_ROE_RATE = 0.012   # ~1.2%

# Speed proxy buckets
SPEED_BUCKETS = {
    "fast": 1.10,   # 진루 확률 +10%
    "avg": 1.00,
    "slow": 0.90,   # 진루 확률 -10%
}

# Defense buckets (BIP 안타 전이에 적용)
DEFENSE_BUCKETS = {
    "good": 0.95,    # BABIP -5%
    "avg": 1.00,
    "poor": 1.05,    # BABIP +5%
}

# Bullpen roles
BULLPEN_ROLES = ["setup_early", "setup_late", "bridge", "closer"]

# Pitch-count fatigue (PA × 4.0 근사 투구수)
PITCH_COUNT_PER_PA = 4.0
FATIGUE_TIERS = {
    "mild": {"threshold": 75, "k_penalty": 0.02, "bb_penalty": 0.02},
    "moderate": {"threshold": 90, "k_penalty": 0.05, "bb_penalty": 0.05},
    "pull": {"threshold": 100},  # 강제 교체
}

# Transition matrix fallback threshold
TRANSITION_MIN_OBS = 30

# ============================================================
# Edge / Kelly Config
# ============================================================
KELLY_FRACTION = 0.25          # Quarter-Kelly (보수적)
MIN_EDGE_THRESHOLD = 0.03      # 3% 미만 edge는 PASS

# ============================================================
# Recent Form Config
# ============================================================
# Strong Home cap: 65%+ raw probability에 추가 dampening
# 200경기 backtest에서 65%+ 구간이 -18.7%p overconfidence
# hard cap 대신 extra dampening: raw 65%+ → 추가 alpha 적용
STRONG_HOME_EXTRA_DAMP = 0.50   # 65%+ 구간에 추가 0.50 dampening
STRONG_HOME_THRESHOLD = 0.62    # dampened 기준 (raw 65% * 0.65 + 0.50*0.35 ≈ 0.60)

RECENT_FORM_WEIGHT_14D = 0.15  # 14일 최근 폼 가중치
RECENT_FORM_WEIGHT_30D = 0.10  # 30일 최근 폼 가중치
RECENT_FORM_MIN_PA_14D = 10    # 14일 최소 PA
RECENT_FORM_MIN_PA_30D = 20    # 30일 최소 PA

# ============================================================
# V3 Pitcher-Centric Model Config
# ============================================================
V3_MIN_TBF = 50                # 스코어링 풀 최소 TBF
V3_MIN_ARSENAL_PA = 20         # 구종별 최소 PA

# 투수 종합 스코어 가중치
V3_PITCHER_SCORE_WEIGHTS = {
    "season": 0.40,
    "arsenal": 0.30,
    "recent_form": 0.30,
}

# 시즌 스탯 세부 가중치
V3_SEASON_STAT_WEIGHTS = {
    "k_rate": 0.20,         # 높을수록 좋음
    "bb_rate": 0.20,        # 낮을수록 좋음 (역전)
    "k_bb_pct": 0.20,       # K%-BB% (높을수록 좋음)
    "hr_rate": 0.15,        # 낮을수록 좋음 (역전)
    "era": 0.15,            # 낮을수록 좋음 (역전)
    "computed_fip": 0.10,   # 낮을수록 좋음 (역전)
}

# 구종 평가 가중치
V3_ARSENAL_WEIGHTS = {
    "stuff": 0.50,          # whiff_rate 기반
    "value": 0.50,          # xwOBA 기반 (역전)
}

# 시즌 가중치 (퍼센타일 풀 구성)
V3_SEASON_RECENCY_WEIGHTS = {
    2025: 0.60,
    2024: 0.30,
    2023: 0.10,
}

# Elo 스타일 승률 변환
V3_ELO_D = 50.0               # 보수적 확률 (calibration 최적)

# 오버/언더 기대 득점 (F5 기준)
V3_BASELINE_TOTAL = 4.5        # 리그 평균 F5 총 득점
V3_RUNS_PER_SCORE_POINT = 0.02 # 평균 대비 1점당 득점 변화 (F5 스케일)

# 홈/원정 & H2H 조정 범위
V3_HOME_AWAY_MAX_ADJ = 5.0     # ±5점
V3_H2H_MAX_BONUS = 10.0       # ±10점
V3_H2H_MIN_PA = 10            # H2H 최소 PA

# 등급 기준 (pick probability 기반)
# 스위트스팟: 65-75% = 최적 구간 (backtest 80% 적중)
V3_GRADES = {
    "sweet_spot": {"min_prob": 0.65, "max_prob": 0.75, "label": "SWEET SPOT"},
    "good":       {"min_prob": 0.60, "max_prob": 0.65, "label": "GOOD"},
    "lean":       {"min_prob": 0.55, "max_prob": 0.60, "label": "LEAN"},
    "pass":       {"min_prob": 0.00, "max_prob": 0.55, "label": "PASS"},
    "overconf":   {"min_prob": 0.75, "max_prob": 1.00, "label": "HIGH CONF"},
}

# effective score 구성 (backtest 최적: M70/P30 + Off5/Dur5)
V3_PITCHER_WEIGHT_IN_EFFECTIVE = 0.30
V3_MATCHUP_WEIGHT_IN_EFFECTIVE = 0.70
V3_OFFENSE_WEIGHT = 0.05             # 상대 타선 화력
V3_DURABILITY_WEIGHT = 0.05          # 투수 이닝 소화력

# TBF 기반 신뢰도 shrinkage: score = raw * conf + 50 * (1 - conf)
# conf = tbf / (tbf + k), k=300이면 300TBF에서 50% 신뢰
V3_TBF_SHRINKAGE_K = 300
