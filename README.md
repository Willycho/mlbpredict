# MLB Predict — V3 Pitcher-Matchup Engine

MLB F5 (First 5 Innings) 예측 엔진. GitHub Actions 기반 자동 스케줄링.

## 구조

```
scripts/
  daily_bulk_predict.py     # 매일 KST 21:00, 다음날 전 경기 bulk 예측 (F5 라인 + 로스터 기준)
  per_game_refresh.py       # 15분마다, 경기 시작 5~60분 전 경기만 라인업+재예측

.github/workflows/
  bulk_predict.yml          # cron '0 12 * * *' (UTC = KST 21:00)
  per_game_refresh.yml      # cron '*/15 * * * *'

data/storage/predictions/
  {YYYY-MM-DD}/
    manifest.json           # 해당 날짜 전 경기 요약
    {game_id}.json          # 경기별 상세 예측

docs/
  index.html, style.css     # GitHub Pages (JSON 읽어서 표시)

engine/                     # V3 예측 엔진 (pitcher_score, matchup_score, game_score)
data/                       # MLB API, Odds API, statcast 처리 데이터
```

## 셋업 (최초 1회)

### 1. GitHub Secrets

Settings → Secrets and variables → Actions → New repository secret

- `ODDS_API_KEY_PRIMARY` — The Odds API 키 (필수)
- `ODDS_API_KEY_BACKUP1` — 백업 키 (선택)
- `ODDS_API_KEY_BACKUP2` — 백업 키 (선택)

### 2. GitHub Pages

Settings → Pages → Source: `Deploy from a branch` → Branch: `main`, Folder: `/docs`

배포 완료되면 `https://{username}.github.io/mlbpredict/`에서 조회 가능.

### 3. Workflow Permissions

Settings → Actions → General → Workflow permissions → **Read and write permissions** 체크 (봇이 commit push 하기 위함)

### 4. 워크플로우 활성화

Actions 탭에서 workflow enable. 수동 트리거 테스트:
- `Daily Bulk Predict` → Run workflow
- `Per-Game Refresh` → Run workflow

## 로컬 개발

`.env` 파일에 키 설정:

```bash
ODDS_API_KEY_PRIMARY=xxx
ODDS_API_KEY_BACKUP1=yyy
```

```bash
pip install -r requirements.txt
python scripts/daily_bulk_predict.py      # bulk 수동 실행
python scripts/per_game_refresh.py        # refresh 수동 실행
```

## 동작 원리

**Bulk (KST 21:00)**
1. 다음 KST 날짜 전 경기 조회 (MLB API)
2. 전 경기 F5 배당사 라인 fetch (1 credit/경기 × 미래 경기 수)
3. 라인업 없이(로스터 기준) 예측 실행
4. `data/storage/predictions/{date}/{game_id}.json` 저장 + `manifest.json`
5. Git commit + push

**Per-Game Refresh (15분마다)**
1. 오늘 날짜 manifest 로드
2. 경기 시작 5~60분 전 범위의 경기만 선정
3. 각 경기:
   - MLB API로 라인업 + 선발투수 재조회 (무료)
   - 저장된 투수와 다르면 → F5 라인 재조회 (1 credit)
   - 같으면 → F5 라인 재사용 (0 credit)
   - 재예측 → 파일 덮어쓰기
4. Git commit + push (변경 있을 때만)

**F5 라인 선정 규칙**
- `totals_1st_5_innings` 마켓, 여러 북메이커 × 여러 라인 후보 수집
- `.5` 단위 라인만 (push 방지)
- **배당차 최소 (Over/Under 소수점 배당의 차이)** = 메인 라인
- 이미 시작된 경기는 제외 (live 잔여 이닝 기준 라인 배제)

## Credit 관리

- Bulk 1회: 약 15-20 credit (미래 경기만)
- Refresh 15분마다: 보통 0-3 credit (투수 변경 없으면 0)
- 하루 예상: 30-50 credit
- 500 credit 키 1개로 10일 이상 운영 가능

## Free vs Pro

로컬 웹 UI (`web/app.py`)는 live 계산. GitHub Actions는 사전 계산 + 저장. docs/index.html은 저장된 JSON을 읽음.

## 엔진 상세

V3 Pitcher-Matchup Engine — investor-ready predictive model combining:
- 투수 시즌 성적 (40%)
- 구종 가치 matchup (30%)
- 최근 폼 (30%)
- 홈/원정 보정
- 타선 화력 + 소화력
- 파크팩터 + 풍향 (Open-Meteo API)

백테스트 (127경기, 2026 3/27~4/6):
- OU 4.5 (gap ≥ 0.3): **60.6%**
- 컨센서스 스위트스팟: **78.9%** (15/19)
