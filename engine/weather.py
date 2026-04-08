"""날씨/풍속 기반 동적 파크팩터 보정.

Open-Meteo API로 경기 시작 전 풍속/풍향 예보를 가져와서
구장 azimuth와 비교하여 Out/In/Cross 판정.
"""

import re
import math
import requests


# 구장 정보: team_code → (lat, lon, azimuth_deg, roof_type)
# azimuth: 홈→CF 방향 (0=북, 90=동, 180=남, 270=서)
STADIUMS = {
    "AZ":  (33.4453, -112.0667, 0,    "Retractable"),
    "ATL": (33.8907, -84.4676,  145,  "Open"),
    "BAL": (39.2838, -76.6217,  31,   "Open"),
    "BOS": (42.3465, -71.0974,  45,   "Open"),
    "CHC": (41.9482, -87.6555,  37,   "Open"),
    "CWS": (41.8300, -87.6342,  127,  "Open"),
    "CIN": (39.0974, -84.5066,  122,  "Open"),
    "CLE": (41.4959, -81.6853,  0,    "Open"),
    "COL": (39.7560, -104.9941, 4,    "Open"),
    "DET": (42.3391, -83.0487,  150,  "Open"),
    "HOU": (29.7570, -95.3555,  343,  "Retractable"),
    "KC":  (39.0516, -94.4805,  46,   "Open"),
    "LAA": (33.8002, -117.8824, 44,   "Open"),
    "LAD": (34.0737, -118.2405, 26,   "Open"),
    "MIA": (25.7780, -80.2195,  128,  "Retractable"),
    "MIL": (43.0284, -87.9710,  129,  "Retractable"),
    "MIN": (44.9818, -93.2779,  129,  "Open"),
    "NYM": (40.7575, -73.8456,  13,   "Open"),
    "NYY": (40.8292, -73.9265,  75,   "Open"),
    "ATH": (38.5799, -121.5125, 46,   "Open"),  # Sutter Health Park
    "PHI": (39.9054, -75.1672,  9,    "Open"),
    "PIT": (40.4469, -80.0058,  116,  "Open"),
    "SD":  (32.7079, -117.1573, 0,    "Open"),
    "SF":  (37.7784, -122.3894, 85,   "Open"),
    "SEA": (47.5913, -122.3325, 49,   "Retractable"),
    "STL": (38.6226, -90.1929,  62,   "Open"),
    "TB":  (27.7678, -82.6525,  359,  "Dome"),
    "TEX": (32.7473, -97.0818,  30,   "Retractable"),
    "TOR": (43.6416, -79.3892,  345,  "Retractable"),
    "WSH": (38.8729, -77.0075,  28,   "Open"),
}


def get_wind_forecast(team: str, game_hour: int = 19) -> dict | None:
    """Open-Meteo API로 구장 풍속/풍향 예보 조회.

    Args:
        team: 홈팀 코드
        game_hour: 경기 시작 시간 (24h, 로컬). 기본 19시(7PM).

    Returns:
        {"speed_mph": float, "direction_deg": float, "relative": str, "adj": float}
        또는 None (돔/에러)
    """
    stadium = STADIUMS.get(team)
    if not stadium:
        return None

    lat, lon, azimuth, roof = stadium

    # 돔 구장은 풍향 무의미
    if roof == "Dome":
        return {"speed_mph": 0, "direction_deg": 0, "relative": "dome", "adj": 0.0}

    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "wind_speed_10m,wind_direction_10m",
            "wind_speed_unit": "mph",
            "timezone": "auto",
            "forecast_days": 2,
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        speeds = hourly.get("wind_speed_10m", [])
        dirs = hourly.get("wind_direction_10m", [])

        # 오늘/내일 game_hour 시간대 찾기
        target_hour = f"{game_hour:02d}:00"
        for i, t in enumerate(times):
            if target_hour in t:
                speed = speeds[i]
                wind_dir = dirs[i]

                # 구장 azimuth와 풍향 비교
                relative, adj = compute_wind_relative(speed, wind_dir, azimuth)

                return {
                    "speed_mph": round(speed, 1),
                    "direction_deg": round(wind_dir, 0),
                    "azimuth": azimuth,
                    "relative": relative,
                    "adj": round(adj, 2),
                    "forecast_time": t,
                }

        return None
    except Exception:
        return None


def compute_wind_relative(speed_mph: float, wind_dir_deg: float, azimuth_deg: float) -> tuple[str, float]:
    """풍향을 구장 기준으로 변환.

    풍향(meteorological): 바람이 불어오는 방향 (270 = 서쪽에서 불어옴 = 동쪽으로 감)
    azimuth: 홈→CF 방향

    바람이 CF 쪽으로 불면 = Out (타자 유리)
    바람이 홈 쪽으로 불면 = In (투수 유리)

    Returns:
        (relative_str, adjustment_runs)
    """
    if speed_mph < 2:
        return "calm", 0.0

    # 바람이 가는 방향 = wind_dir + 180
    wind_going = (wind_dir_deg + 180) % 360

    # azimuth와의 각도 차이
    diff = (wind_going - azimuth_deg + 360) % 360
    if diff > 180:
        diff = 360 - diff

    # diff 해석:
    # 0-45도: 바람이 CF 방향으로 (Out) → 타자 유리
    # 135-180도: 바람이 홈 방향으로 (In) → 투수 유리
    # 45-135도: 크로스 바람

    # Out/In 강도: cos(diff)
    # cos(0)=1 (완전 Out), cos(90)=0 (cross), cos(180)=-1 (완전 In)
    cos_factor = math.cos(math.radians(diff))

    # 보정값: speed * cos_factor * scale
    # 10mph 완전 Out ≈ +0.3 runs
    scale = 0.03
    adj = speed_mph * cos_factor * scale

    # cap
    adj = max(min(adj, 0.8), -0.8)

    if cos_factor > 0.3:
        relative = "out"
    elif cos_factor < -0.3:
        relative = "in"
    else:
        relative = "cross"

    return relative, adj


def parse_wind(wind_str: str) -> dict:
    """MLB API 풍속 문자열 파싱 (레거시 호환)."""
    if not wind_str or wind_str == "?":
        return {"speed": 0, "direction": "calm", "target": None}

    speed_match = re.match(r"(\d+)\s*mph", wind_str)
    speed = int(speed_match.group(1)) if speed_match else 0
    wind_lower = wind_str.lower()

    if "calm" in wind_lower or "none" in wind_lower or speed == 0:
        return {"speed": speed, "direction": "calm", "target": None}
    if "out to" in wind_lower:
        target_match = re.search(r"out to\s+(\w+)", wind_lower)
        target = target_match.group(1).upper() if target_match else "CF"
        return {"speed": speed, "direction": "out", "target": target}
    if "in from" in wind_lower:
        target_match = re.search(r"in from\s+(\w+)", wind_lower)
        target = target_match.group(1).upper() if target_match else "CF"
        return {"speed": speed, "direction": "in", "target": target}
    if "l to r" in wind_lower or "r to l" in wind_lower:
        return {"speed": speed, "direction": "cross", "target": None}
    if "varies" in wind_lower:
        return {"speed": speed, "direction": "varies", "target": None}

    return {"speed": speed, "direction": "unknown", "target": None}


if __name__ == "__main__":
    print("=== Open-Meteo Wind Forecast Test ===\n")
    test_teams = ["NYY", "BOS", "COL", "SF", "TB", "LAD"]
    for team in test_teams:
        result = get_wind_forecast(team, game_hour=19)
        if result:
            s = STADIUMS[team]
            print(f"  {team} ({s[3]:12s}): {result['speed_mph']:>5.1f}mph "
                  f"dir={result['direction_deg']:>3.0f}deg "
                  f"azimuth={result.get('azimuth', '-')} "
                  f"-> {result['relative']:5s} adj={result['adj']:+.2f} runs")
        else:
            print(f"  {team}: no data")
