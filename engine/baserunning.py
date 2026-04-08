"""주루 상태 전이 — 타석 결과에 따른 베이스/아웃/득점 처리."""

import numpy as np

from config import BASE_ADVANCEMENT, LEAGUE_AVG_SPRINT_SPEED


def speed_modifier(sprint_speed: float | None) -> float:
    """스프린트 스피드 기반 진루 확률 보정 계수.

    빠른 주자(>27 ft/s)는 진루 확률 상승, 느린 주자는 하락.
    """
    if sprint_speed is None:
        return 1.0
    diff = sprint_speed - LEAGUE_AVG_SPRINT_SPEED
    # ft/s 차이 1당 약 3% 보정
    return 1.0 + diff * 0.03


def resolve_baserunning(
    outcome: str,
    bases: list,  # [1B runner, 2B runner, 3B runner] — None or sprint_speed float
    outs: int,
    gdp_rate: float,
    rng: np.random.Generator,
) -> tuple[int, list, int]:
    """타석 결과에 따른 주루 상태 전이.

    Args:
        outcome: PA 결과 ("1B", "2B", "3B", "HR", "BB", "HBP", "K", "GO", "FO", "LO", "PO")
        bases: [1루, 2루, 3루] 각 값은 None 또는 주자의 sprint_speed
        outs: 현재 아웃 카운트 (0, 1, 2)
        gdp_rate: 타자의 병살 확률
        rng: numpy random generator

    Returns:
        (득점 수, 새로운 bases, 추가 아웃 수)
    """
    runs = 0
    new_bases = [None, None, None]
    added_outs = 0

    on_1b, on_2b, on_3b = bases

    if outcome == "HR":
        # 모든 주자 + 타자 홈인
        runs = 1  # 타자
        if on_1b is not None:
            runs += 1
        if on_2b is not None:
            runs += 1
        if on_3b is not None:
            runs += 1

    elif outcome in ("1B",):
        # 싱글
        adv = BASE_ADVANCEMENT["single"]

        # 3루주자 → 홈
        if on_3b is not None:
            runs += 1

        # 2루주자 → 홈 or 3루
        if on_2b is not None:
            p_home = adv["second_to_home"] * speed_modifier(on_2b)
            if rng.random() < p_home:
                runs += 1
            else:
                new_bases[2] = on_2b  # 3루에 머무름

        # 1루주자 → 3루 or 2루
        if on_1b is not None:
            p_third = adv["first_to_third"] * speed_modifier(on_1b)
            if rng.random() < p_third:
                new_bases[2] = on_1b if new_bases[2] is None else new_bases[2]
                # 3루가 차있으면 2루에
                if new_bases[2] != on_1b:
                    new_bases[1] = on_1b
            else:
                new_bases[1] = on_1b

        # 타자 → 1루
        new_bases[0] = LEAGUE_AVG_SPRINT_SPEED  # 타자 속도 (기본값)

    elif outcome == "2B":
        # 더블
        if on_3b is not None:
            runs += 1
        if on_2b is not None:
            runs += 1

        # 1루주자 → 홈 or 3루
        if on_1b is not None:
            p_home = BASE_ADVANCEMENT["double"]["first_to_home"] * speed_modifier(on_1b)
            if rng.random() < p_home:
                runs += 1
            else:
                new_bases[2] = on_1b

        # 타자 → 2루
        new_bases[1] = LEAGUE_AVG_SPRINT_SPEED

    elif outcome == "3B":
        # 트리플
        if on_3b is not None:
            runs += 1
        if on_2b is not None:
            runs += 1
        if on_1b is not None:
            runs += 1
        new_bases[2] = LEAGUE_AVG_SPRINT_SPEED

    elif outcome in ("BB", "HBP"):
        # 볼넷/사구 — 포스 진루만
        if on_1b is not None and on_2b is not None and on_3b is not None:
            runs += 1  # 만루에서 밀어내기
            new_bases = [LEAGUE_AVG_SPRINT_SPEED, on_1b, on_2b]
        elif on_1b is not None and on_2b is not None:
            new_bases = [LEAGUE_AVG_SPRINT_SPEED, on_1b, on_2b]
        elif on_1b is not None:
            new_bases = [LEAGUE_AVG_SPRINT_SPEED, on_1b, on_3b]
        else:
            new_bases = [LEAGUE_AVG_SPRINT_SPEED, on_2b, on_3b]

    elif outcome == "K":
        added_outs = 1

    elif outcome == "GO":
        # 땅볼 아웃
        added_outs = 1

        # 병살 체크
        if on_1b is not None and outs < 2:
            if rng.random() < gdp_rate:
                added_outs = 2
                # 1루주자 아웃, 타자 아웃
                # 다른 주자는 진루
                if on_3b is not None and outs == 0:
                    # 3루주자 홈인 (병살 중에도)
                    p_home = BASE_ADVANCEMENT["ground_out"]["third_to_home_less2"]
                    if rng.random() < p_home:
                        runs += 1
                    else:
                        new_bases[2] = on_3b
                if on_2b is not None:
                    new_bases[2] = on_2b
                return runs, new_bases, added_outs

        # 병살 아닌 경우
        # 3루주자 → 홈 (2아웃 미만, 내야 땅볼)
        if on_3b is not None and outs < 2:
            p_home = BASE_ADVANCEMENT["ground_out"]["third_to_home_less2"]
            if rng.random() < p_home:
                runs += 1
            else:
                new_bases[2] = on_3b
        elif on_3b is not None:
            new_bases[2] = on_3b

        # 2루주자 → 3루
        if on_2b is not None:
            p_adv = BASE_ADVANCEMENT["ground_out"]["second_to_third"]
            if new_bases[2] is None and rng.random() < p_adv:
                new_bases[2] = on_2b
            else:
                new_bases[1] = on_2b

        # 1루주자 처리
        if on_1b is not None:
            # 타자가 아웃, 1루주자는 2루로 진루 (포스 진루)
            if new_bases[1] is None:
                new_bases[1] = on_1b
            # 2루가 차있으면 포스아웃으로 1루주자 제거 (이미 아웃 1 카운트됨)

    elif outcome == "FO":
        # 플라이 아웃
        added_outs = 1

        # 희생 플라이 (3루주자, 2아웃 미만)
        if on_3b is not None and outs < 2:
            p_sac = BASE_ADVANCEMENT["fly_out"]["third_to_home_sac"]
            if rng.random() < p_sac:
                runs += 1
            else:
                new_bases[2] = on_3b
        elif on_3b is not None:
            new_bases[2] = on_3b

        # 2루주자 태그업 → 3루 (2아웃 미만, 외야 플라이)
        if on_2b is not None and outs < 2 and new_bases[2] is None:
            if rng.random() < 0.45:  # 실제 태그업 성공률 ~45%
                new_bases[2] = on_2b
            else:
                new_bases[1] = on_2b
        elif on_2b is not None:
            new_bases[1] = on_2b

        # 1루주자는 유지 (태그업 진루 드묾)
        if on_1b is not None:
            new_bases[0] = on_1b

    elif outcome in ("LO", "PO"):
        # 라인아웃, 팝업 — 주자 이동 없음
        added_outs = 1
        new_bases = [on_1b, on_2b, on_3b]

    return runs, new_bases, added_outs
