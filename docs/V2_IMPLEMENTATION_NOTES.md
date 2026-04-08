# V2 Engine Implementation Notes

## v1 → v2 변경 요약

### 삭제된 것
- `ERROR_RATE = 0.08` 아웃→안타 강제 전환 hack
- 독립 Log5 순차 threshold (K→BB→HBP→HR→BIP)
- 고정 baserunning 확률 (first_to_third=0.29 등)
- 리그 평균 2티어 불펜 (K% 75th percentile 기준)
- 데이터 누수형 백테스트

### 추가된 것
- **Bayesian Shrinkage**: 모든 소표본 피처에 `n/(n+k)` 적용
- **2단계 Multinomial PA**: Stage1(K/BB/HBP/HR/BIP) → Stage2(1B/2B/3B/GO/FO/LO/FC/ROE)
- **Markov Base-Out Transition**: Statcast 기반 288셀 조건부 전이 행렬
- **Count-aware Pitch Mix**: 4 count bucket별 투수 구종 분포
- **팀 수비 보정**: xBA vs actual BABIP proxy, 3 bucket
- **Speed Proxy**: 3루타+도루+내야안타 기반 3 bucket
- **4티어 불펜**: setup_early/setup_late/bridge/closer
- **투구수 기반 피로**: PA×4.0 근사, 구간별 차등 패널티
- **Calibration Layer**: Isotonic + Platt + Ensemble
- **Walk-forward Backtest**: 시간순 데이터 분할, 미래 데이터 미사용

### 파일 매핑

| v1 | v2 | 변경 내용 |
|----|-----|----------|
| engine/matchup.py | engine/bayesian_matchup.py | Shrinkage + count-aware + 수비 |
| engine/plate_appearance.py | engine/multinomial_pa.py | 2단계 multinomial, error hack 제거 |
| engine/baserunning.py | engine/markov_transition.py | Statcast 전이 행렬 |
| engine/inning.py | engine/markov_inning.py | absolute state transition |
| engine/game.py | engine/game_engine.py | 4티어 불펜, pitch count fatigue |
| engine/simulation.py | engine/monte_carlo.py | v2 데이터 통합, calibration |
| (없음) | engine/calibration.py | Isotonic/Platt/ECE/Brier |

### 새 데이터 스크립트

| 파일 | 출력 | 목적 |
|------|------|------|
| data/build_count_pitch_mix.py | v2/count_pitch_mix.parquet | 투수별 count×hand별 구종 분포 |
| data/build_transition_matrix.py | v2/transition_matrix.parquet | 24-state Markov 전이 행렬 |
| data/build_run_expectancy.py | v2/run_expectancy.parquet | RE24 기대 득점 행렬 |
| data/build_team_defense.py | v2/team_defense.parquet, v2/speed_proxy.parquet | 수비+주루 proxy |
| data/build_bullpen_profiles.py | v2/bullpen_profiles.parquet | 4역할 불펜 프로필 |
| data/build_time_safe_features.py | v2/snapshots/{date}/ | 시점 고정 피처 스냅샷 |

## 미구현 / 추정 처리 목록

1. **Sprint Speed**: Statcast raw에 없어 proxy(3루타+내야안타) 사용. pybaseball sprint_speed로 개선 가능.
2. **OAA**: 외부 데이터 대신 xBA vs BABIP proxy. FanGraphs OAA로 교체 가능.
3. **Count-state PA 모델**: 첫 버전은 타석 시작(0-0)만 반영. 실제 pitch-by-pitch count 시뮬은 미구현.
4. **최근 폼(recent form)**: 미구현. 최근 14일/30일 rolling window 반영 필요.
5. **투수 개별 불펜**: 팀 평균 역할 프로필 사용. 개별 릴리버 스탯 매칭은 미구현.
6. **도루/견제**: 미모델링. Markov 전이에 암묵적으로 포함되지만 명시적 SB 이벤트는 없음.
7. **날씨/온도**: 미반영.
8. **Umpire zone**: 미반영.
9. **DH/NL 차이**: 2022 이후 universal DH로 무관.
10. **Calibrator**: 충분한 백테스트 데이터 후 학습 필요. 초기에는 raw probability 사용.

## Shrinkage 상수 표

| 피처 | k값 | Prior | 의미 |
|------|-----|-------|------|
| 타자 K%/BB% | 200 PA | 리그 평균 | 200PA에서 50% 반영 |
| 타자 HR% | 300 PA | 리그 평균 | 더 노이즈 큼 |
| 타자 BABIP | 400 PA | 리그 평균 | 가장 노이즈 큼 |
| 투수 K%/BB% | 150 TBF | 리그 평균 | 투수 K% 안정화 빠름 |
| 투수 HR% | 350 TBF | 리그 평균 | HR/FB 매우 노이즈 |
| H2H | 50 PA | 구종 가중 매치업 | 극소 표본 |
| 스플릿 ratio | 80 PA | 1.0 (무보정) | platoon/RISP/H-A |
| Count pitch mix | 50투구 | 시즌 전체 mix | 버킷당 |

## API 사용법

```bash
# v1 (기존)
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"away_team":"NYY","home_team":"LAD","away_pitcher_name":"Cole","home_pitcher_name":"Glasnow"}'

# v2 (신규)
curl -X POST http://localhost:8000/api/simulate/v2 \
  -H "Content-Type: application/json" \
  -d '{"away_team":"NYY","home_team":"LAD","away_pitcher_name":"Cole","home_pitcher_name":"Glasnow","use_team_strength":false,"count_aware":true}'
```

## 데이터 빌드 순서

```bash
cd baseball-sim

# 1. v2 데이터 빌드 (Statcast raw 필요)
python data/build_count_pitch_mix.py
python data/build_transition_matrix.py
python data/build_run_expectancy.py
python data/build_team_defense.py
python data/build_bullpen_profiles.py

# 2. (선택) Time-safe 스냅샷 빌드
python data/build_time_safe_features.py 2025-07-01

# 3. Walk-forward 백테스트
python backtest/walk_forward_backtest.py --start 2025-07-01 --end 2025-07-31 --sims 500

# 4. Calibration 평가
python backtest/evaluate_calibration.py backtest/results/2025-07-01_2025-07-31_v2.json
```
