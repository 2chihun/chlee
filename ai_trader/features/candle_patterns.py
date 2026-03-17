"""캔들스틱 패턴 인식 모듈

「현명한 당신의 주식투자 교과서」(박병창 저) Part 3 "봉의 해석"을 기반으로
한국 주식시장에서 사용되는 캔들스틱 패턴을 감지합니다.

패턴 값: 1(매수 신호), -1(매도 신호), 0(해당 없음)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _body(o: pd.Series, c: pd.Series) -> pd.Series:
    """실체 크기 (절댓값)"""
    return (c - o).abs()


def _upper_shadow(h: pd.Series, o: pd.Series, c: pd.Series) -> pd.Series:
    """윗꼬리 길이"""
    return h - pd.concat([o, c], axis=1).max(axis=1)


def _lower_shadow(l: pd.Series, o: pd.Series, c: pd.Series) -> pd.Series:
    """아랫꼬리 길이"""
    return pd.concat([o, c], axis=1).min(axis=1) - l


def _candle_range(h: pd.Series, l: pd.Series) -> pd.Series:
    """고가-저가 전체 범위"""
    return (h - l).replace(0, np.nan)


def _is_bullish(o: pd.Series, c: pd.Series) -> pd.Series:
    """양봉 여부"""
    return c > o


def _is_bearish(o: pd.Series, c: pd.Series) -> pd.Series:
    """음봉 여부"""
    return c < o


def _avg_body(o: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    """최근 N봉 평균 실체 크기"""
    return _body(o, c).rolling(window=period, min_periods=1).mean()


# ---------------------------------------------------------------------------
# 단일 봉 패턴
# ---------------------------------------------------------------------------

def _hammer(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """망치형 (Hammer) — 하락 추세 후 반전 매수 신호

    긴 아랫꼬리(실체의 2배 이상), 짧은 윗꼬리, 작은 실체.
    """
    body = _body(o, c)
    rng = _candle_range(h, l)
    lower = _lower_shadow(l, o, c)
    upper = _upper_shadow(h, o, c)

    cond = (
        (lower >= body * 2)
        & (upper <= body * 0.3)
        & (body > 0)
        & (body <= rng * 0.35)
    )
    return cond.astype(int)


def _inverted_hammer(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series
) -> pd.Series:
    """역망치형 (Inverted Hammer) — 하락 추세 후 반전 매수 신호

    긴 윗꼬리, 짧은 아랫꼬리, 작은 실체.
    """
    body = _body(o, c)
    rng = _candle_range(h, l)
    lower = _lower_shadow(l, o, c)
    upper = _upper_shadow(h, o, c)

    cond = (
        (upper >= body * 2)
        & (lower <= body * 0.3)
        & (body > 0)
        & (body <= rng * 0.35)
    )
    return cond.astype(int)


def _hanging_man(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series
) -> pd.Series:
    """교수형 (Hanging Man) — 상승 추세 후 반전 매도 신호

    형태는 망치형과 동일하나 상승 추세에서 출현.
    """
    body = _body(o, c)
    rng = _candle_range(h, l)
    lower = _lower_shadow(l, o, c)
    upper = _upper_shadow(h, o, c)

    cond = (
        (lower >= body * 2)
        & (upper <= body * 0.3)
        & (body > 0)
        & (body <= rng * 0.35)
    )
    return cond.astype(int)


def _shooting_star(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series
) -> pd.Series:
    """유성형 (Shooting Star) — 상승 추세 후 반전 매도 신호

    형태는 역망치형과 동일하나 상승 추세에서 출현.
    """
    body = _body(o, c)
    rng = _candle_range(h, l)
    lower = _lower_shadow(l, o, c)
    upper = _upper_shadow(h, o, c)

    cond = (
        (upper >= body * 2)
        & (lower <= body * 0.3)
        & (body > 0)
        & (body <= rng * 0.35)
    )
    return cond.astype(int)


def _doji(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """도지 (Doji) — 시가와 종가가 거의 같은 봉, 추세 전환 가능성

    실체가 전체 범위의 5% 이하.
    """
    body = _body(o, c)
    rng = _candle_range(h, l)
    cond = (body <= rng * 0.05) & (rng > 0)
    return cond.astype(int)


def _spinning_top(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series
) -> pd.Series:
    """팽이 (Spinning Top) — 작은 실체에 위아래 꼬리, 우유부단

    실체가 전체 범위의 5~30%, 위아래 꼬리 모두 실체 이상.
    """
    body = _body(o, c)
    rng = _candle_range(h, l)
    lower = _lower_shadow(l, o, c)
    upper = _upper_shadow(h, o, c)

    body_ratio = body / rng
    cond = (
        (body_ratio > 0.05)
        & (body_ratio <= 0.30)
        & (upper >= body)
        & (lower >= body)
    )
    return cond.astype(int)


def _long_white(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series
) -> pd.Series:
    """장대양봉 (Long White/Green) — 강한 매수세

    양봉이면서 실체가 전체 범위의 70% 이상.
    """
    body = _body(o, c)
    rng = _candle_range(h, l)
    cond = _is_bullish(o, c) & (body >= rng * 0.70)
    return cond.astype(int)


def _long_black(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series
) -> pd.Series:
    """장대음봉 (Long Black/Red) — 강한 매도세

    음봉이면서 실체가 전체 범위의 70% 이상.
    """
    body = _body(o, c)
    rng = _candle_range(h, l)
    cond = _is_bearish(o, c) & (body >= rng * 0.70)
    return cond.astype(int)


def _marubozu(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series
) -> pd.Series:
    """마루보즈 (Marubozu) — 꼬리 없는 장대봉

    실체가 전체 범위의 95% 이상 (양봉 +1, 음봉 -1).
    """
    body = _body(o, c)
    rng = _candle_range(h, l)
    is_marubozu = body >= rng * 0.95

    result = pd.Series(0, index=o.index)
    result[is_marubozu & _is_bullish(o, c)] = 1
    result[is_marubozu & _is_bearish(o, c)] = -1
    return result


# ---------------------------------------------------------------------------
# 2봉 패턴
# ---------------------------------------------------------------------------

def _bullish_engulfing(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series
) -> pd.Series:
    """상승장악형 (Bullish Engulfing) — 강한 매수 반전

    전일 음봉의 실체를 금일 양봉이 완전히 감싸는 형태.
    """
    prev_o, prev_c = o.shift(1), c.shift(1)

    cond = (
        _is_bearish(prev_o, prev_c)       # 전일 음봉
        & _is_bullish(o, c)                # 금일 양봉
        & (o <= prev_c)                    # 금일 시가 <= 전일 종가
        & (c >= prev_o)                    # 금일 종가 >= 전일 시가
        & (_body(o, c) > _body(prev_o, prev_c))  # 금일 실체 > 전일 실체
    )
    return cond.astype(int)


def _bearish_engulfing(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series
) -> pd.Series:
    """하락장악형 (Bearish Engulfing) — 강한 매도 반전

    전일 양봉의 실체를 금일 음봉이 완전히 감싸는 형태.
    """
    prev_o, prev_c = o.shift(1), c.shift(1)

    cond = (
        _is_bullish(prev_o, prev_c)        # 전일 양봉
        & _is_bearish(o, c)                 # 금일 음봉
        & (o >= prev_c)                     # 금일 시가 >= 전일 종가
        & (c <= prev_o)                     # 금일 종가 <= 전일 시가
        & (_body(o, c) > _body(prev_o, prev_c))  # 금일 실체 > 전일 실체
    )
    return cond.astype(int)


def _piercing(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series
) -> pd.Series:
    """관통형 (Piercing Line) — 하락 추세 후 매수 반전

    전일 장대음봉 후, 금일 양봉이 전일 실체의 50% 이상 관통.
    """
    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_mid = (prev_o + prev_c) / 2

    cond = (
        _is_bearish(prev_o, prev_c)        # 전일 음봉
        & _is_bullish(o, c)                 # 금일 양봉
        & (o < prev_c)                      # 금일 시가 < 전일 종가 (갭다운)
        & (c > prev_mid)                    # 금일 종가 > 전일 실체 중간
        & (c < prev_o)                      # 금일 종가 < 전일 시가 (완전 장악 아님)
    )
    return cond.astype(int)


def _dark_cloud(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series
) -> pd.Series:
    """먹구름형 (Dark Cloud Cover) — 상승 추세 후 매도 반전

    전일 장대양봉 후, 금일 음봉이 전일 실체의 50% 이상 침투.
    """
    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_mid = (prev_o + prev_c) / 2

    cond = (
        _is_bullish(prev_o, prev_c)         # 전일 양봉
        & _is_bearish(o, c)                  # 금일 음봉
        & (o > prev_c)                       # 금일 시가 > 전일 종가 (갭업)
        & (c < prev_mid)                     # 금일 종가 < 전일 실체 중간
        & (c > prev_o)                       # 금일 종가 > 전일 시가 (완전 장악 아님)
    )
    return cond.astype(int)


def _harami(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series
) -> pd.Series:
    """하라미 (Harami) — 추세 둔화 및 반전 가능성

    금일 실체가 전일 실체 안에 완전히 포함.
    상승 하라미(전일 음봉+금일 양봉) +1, 하락 하라미(전일 양봉+금일 음봉) -1.
    """
    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_body_hi = pd.concat([prev_o, prev_c], axis=1).max(axis=1)
    prev_body_lo = pd.concat([prev_o, prev_c], axis=1).min(axis=1)
    cur_body_hi = pd.concat([o, c], axis=1).max(axis=1)
    cur_body_lo = pd.concat([o, c], axis=1).min(axis=1)

    contained = (
        (cur_body_hi <= prev_body_hi)
        & (cur_body_lo >= prev_body_lo)
        & (_body(o, c) < _body(prev_o, prev_c))
    )

    result = pd.Series(0, index=o.index)
    # 상승 하라미: 전일 음봉 + 금일 양봉
    result[contained & _is_bearish(prev_o, prev_c) & _is_bullish(o, c)] = 1
    # 하락 하라미: 전일 양봉 + 금일 음봉
    result[contained & _is_bullish(prev_o, prev_c) & _is_bearish(o, c)] = -1
    return result


# ---------------------------------------------------------------------------
# 3봉 패턴
# ---------------------------------------------------------------------------

def _morning_star(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series
) -> pd.Series:
    """샛별형 (Morning Star) — 하락 추세 후 강한 반전 매수 신호

    1일차: 장대음봉, 2일차: 갭다운 후 작은 실체, 3일차: 장대양봉.
    """
    prev2_o, prev2_c = o.shift(2), c.shift(2)
    prev1_o, prev1_c = o.shift(1), c.shift(1)
    avg = _avg_body(o, c)

    body_2 = _body(prev2_o, prev2_c)
    body_1 = _body(prev1_o, prev1_c)
    body_0 = _body(o, c)
    prev2_mid = (prev2_o + prev2_c) / 2

    cond = (
        _is_bearish(prev2_o, prev2_c)      # 1일차 음봉
        & (body_2 > avg)                     # 1일차 장대봉
        & (body_1 < body_2 * 0.3)           # 2일차 작은 실체
        & _is_bullish(o, c)                  # 3일차 양봉
        & (body_0 > avg)                     # 3일차 장대봉
        & (c > prev2_mid)                    # 3일차 종가 > 1일차 실체 중간
    )
    return cond.astype(int)


def _evening_star(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series
) -> pd.Series:
    """석별형 (Evening Star) — 상승 추세 후 강한 반전 매도 신호

    1일차: 장대양봉, 2일차: 갭업 후 작은 실체, 3일차: 장대음봉.
    """
    prev2_o, prev2_c = o.shift(2), c.shift(2)
    prev1_o, prev1_c = o.shift(1), c.shift(1)
    avg = _avg_body(o, c)

    body_2 = _body(prev2_o, prev2_c)
    body_1 = _body(prev1_o, prev1_c)
    body_0 = _body(o, c)
    prev2_mid = (prev2_o + prev2_c) / 2

    cond = (
        _is_bullish(prev2_o, prev2_c)       # 1일차 양봉
        & (body_2 > avg)                     # 1일차 장대봉
        & (body_1 < body_2 * 0.3)           # 2일차 작은 실체
        & _is_bearish(o, c)                  # 3일차 음봉
        & (body_0 > avg)                     # 3일차 장대봉
        & (c < prev2_mid)                    # 3일차 종가 < 1일차 실체 중간
    )
    return cond.astype(int)


def _three_white_soldiers(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series
) -> pd.Series:
    """적삼병 (Three White Soldiers) — 강한 상승 신호

    연속 3개의 양봉. 각 봉의 종가가 이전 봉의 종가보다 높고,
    각 봉의 시가가 이전 봉의 실체 내에서 시작.
    """
    prev2_o, prev2_c = o.shift(2), c.shift(2)
    prev1_o, prev1_c = o.shift(1), c.shift(1)
    avg = _avg_body(o, c)

    cond = (
        # 3봉 모두 양봉
        _is_bullish(prev2_o, prev2_c)
        & _is_bullish(prev1_o, prev1_c)
        & _is_bullish(o, c)
        # 종가 연속 상승
        & (prev1_c > prev2_c)
        & (c > prev1_c)
        # 시가가 이전 봉 실체 내
        & (prev1_o >= prev2_o) & (prev1_o <= prev2_c)
        & (o >= prev1_o) & (o <= prev1_c)
        # 각 봉이 평균 이상의 실체
        & (_body(prev2_o, prev2_c) > avg * 0.5)
        & (_body(prev1_o, prev1_c) > avg * 0.5)
        & (_body(o, c) > avg * 0.5)
    )
    return cond.astype(int)


def _three_black_crows(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series
) -> pd.Series:
    """흑삼병 (Three Black Crows) — 강한 하락 신호

    연속 3개의 음봉. 각 봉의 종가가 이전 봉의 종가보다 낮고,
    각 봉의 시가가 이전 봉의 실체 내에서 시작.
    """
    prev2_o, prev2_c = o.shift(2), c.shift(2)
    prev1_o, prev1_c = o.shift(1), c.shift(1)
    avg = _avg_body(o, c)

    cond = (
        # 3봉 모두 음봉
        _is_bearish(prev2_o, prev2_c)
        & _is_bearish(prev1_o, prev1_c)
        & _is_bearish(o, c)
        # 종가 연속 하락
        & (prev1_c < prev2_c)
        & (c < prev1_c)
        # 시가가 이전 봉 실체 내
        & (prev1_o <= prev2_o) & (prev1_o >= prev2_c)
        & (o <= prev1_o) & (o >= prev1_c)
        # 각 봉이 평균 이상의 실체
        & (_body(prev2_o, prev2_c) > avg * 0.5)
        & (_body(prev1_o, prev1_c) > avg * 0.5)
        & (_body(o, c) > avg * 0.5)
    )
    return cond.astype(int)


# ---------------------------------------------------------------------------
# 추세 판단 헬퍼 (단일/교수/유성 등 추세 맥락 필요 패턴용)
# ---------------------------------------------------------------------------

def _trend(c: pd.Series, period: int = 5) -> pd.Series:
    """단기 추세 방향: +1(상승), -1(하락), 0(횡보)

    최근 period봉의 종가 이동평균과 현재가 비교.
    """
    ma = c.rolling(window=period, min_periods=1).mean()
    diff = c - ma
    threshold = c * 0.005  # 0.5% 이상 차이
    result = pd.Series(0, index=c.index)
    result[diff > threshold] = 1
    result[diff < -threshold] = -1
    return result


# ---------------------------------------------------------------------------
# 패턴 신뢰도
# ---------------------------------------------------------------------------

PATTERN_CONFIG: dict[str, dict] = {
    # 단일 봉
    "hammer":           {"signal": 1,  "confidence": 0.55},
    "inverted_hammer":  {"signal": 1,  "confidence": 0.50},
    "hanging_man":      {"signal": -1, "confidence": 0.55},
    "shooting_star":    {"signal": -1, "confidence": 0.55},
    "doji":             {"signal": 0,  "confidence": 0.40},
    "spinning_top":     {"signal": 0,  "confidence": 0.35},
    "long_white":       {"signal": 1,  "confidence": 0.60},
    "long_black":       {"signal": -1, "confidence": 0.60},
    "marubozu":         {"signal": 0,  "confidence": 0.65},  # 방향은 값으로 결정
    # 2봉
    "bullish_engulfing": {"signal": 1,  "confidence": 0.75},
    "bearish_engulfing": {"signal": -1, "confidence": 0.75},
    "piercing":          {"signal": 1,  "confidence": 0.65},
    "dark_cloud":        {"signal": -1, "confidence": 0.65},
    "harami":            {"signal": 0,  "confidence": 0.55},  # 방향은 값으로 결정
    # 3봉
    "morning_star":      {"signal": 1,  "confidence": 0.80},
    "evening_star":      {"signal": -1, "confidence": 0.80},
    "three_soldiers":    {"signal": 1,  "confidence": 0.85},
    "three_crows":       {"signal": -1, "confidence": 0.85},
}


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def detect_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame에 캔들 패턴 컬럼을 추가합니다.

    추가되는 컬럼은 ``cdl_<패턴명>`` 형식이며 값은 1(매수), -1(매도), 0(해당 없음).
    추세 맥락이 필요한 패턴(망치형/교수형/역망치형/유성형)은 추세를 고려하여
    방향 신호를 부여합니다.

    Args:
        df: ``open``, ``high``, ``low``, ``close`` 컬럼이 필요한 DataFrame.

    Returns:
        캔들 패턴 컬럼이 추가된 DataFrame (원본 복사).
    """
    result = df.copy()
    o = result["open"].astype(float)
    h = result["high"].astype(float)
    l = result["low"].astype(float)
    c = result["close"].astype(float)

    tr = _trend(c)

    # --- 단일 봉 ---
    # 망치형: 하락 추세에서만 매수 신호
    hammer_shape = _hammer(o, h, l, c)
    result["cdl_hammer"] = (hammer_shape * (tr == -1).astype(int)).clip(-1, 1)

    # 역망치형: 하락 추세에서만 매수 신호
    inv_hammer_shape = _inverted_hammer(o, h, l, c)
    result["cdl_inverted_hammer"] = (
        inv_hammer_shape * (tr == -1).astype(int)
    ).clip(-1, 1)

    # 교수형: 상승 추세에서만 매도 신호
    hanging_shape = _hanging_man(o, h, l, c)
    result["cdl_hanging_man"] = (
        -hanging_shape * (tr == 1).astype(int)
    ).clip(-1, 1)

    # 유성형: 상승 추세에서만 매도 신호
    star_shape = _shooting_star(o, h, l, c)
    result["cdl_shooting_star"] = (
        -star_shape * (tr == 1).astype(int)
    ).clip(-1, 1)

    # 도지
    result["cdl_doji"] = _doji(o, h, l, c)

    # 팽이
    result["cdl_spinning_top"] = _spinning_top(o, h, l, c)

    # 장대양봉
    result["cdl_long_white"] = _long_white(o, h, l, c)

    # 장대음봉
    result["cdl_long_black"] = -_long_black(o, h, l, c)

    # 마루보즈
    result["cdl_marubozu"] = _marubozu(o, h, l, c)

    # --- 2봉 ---
    result["cdl_bullish_engulfing"] = _bullish_engulfing(o, h, l, c)
    result["cdl_bearish_engulfing"] = -_bearish_engulfing(o, h, l, c)
    result["cdl_piercing"] = _piercing(o, h, l, c)
    result["cdl_dark_cloud"] = -_dark_cloud(o, h, l, c)
    result["cdl_harami"] = _harami(o, h, l, c)

    # --- 3봉 ---
    result["cdl_morning_star"] = _morning_star(o, h, l, c)
    result["cdl_evening_star"] = -_evening_star(o, h, l, c)
    result["cdl_three_soldiers"] = _three_white_soldiers(o, h, l, c)
    result["cdl_three_crows"] = -_three_black_crows(o, h, l, c)

    return result


def detect_candle_groups(df: pd.DataFrame) -> pd.DataFrame:
    """캔들군 분석 — 캔들마스터 방식

    여러 봉을 묶어서 해석하는 캔들군 패턴을 감지합니다.

    감지 패턴:
    - 꼬리군: 이중 꼬리군(2개 긴 위꼬리), 다중 꼬리군(3개+ 긴 위꼬리),
              후퇴 꼬리군(고가가 서서히 낮아지는)
    - 상기캔: 저가가 점진적으로 상승하는 캔들 그룹
    - 수렴캔: 고저 범위가 점차 좁아지는 캔들 그룹
    - 후퇴캔: 고가가 점진적으로 하락하는 캔들 그룹

    Args:
        df: ``open``, ``high``, ``low``, ``close`` 컬럼이 필요한 DataFrame.

    Returns:
        캔들군 패턴 컬럼이 추가된 DataFrame (원본 복사).
        추가 컬럼: cdl_group_double_tail, cdl_group_multi_tail,
                   cdl_group_retreat_tail, cdl_group_rising_low,
                   cdl_group_converging, cdl_group_retreat_high
    """
    result = df.copy()
    o = result["open"].astype(float)
    h = result["high"].astype(float)
    l = result["low"].astype(float)
    c = result["close"].astype(float)

    rng = _candle_range(h, l)
    upper = _upper_shadow(h, o, c)
    body = _body(o, c)

    # 긴 위꼬리 판단: 위꼬리가 실체의 1.5배 이상이고 전체 범위의 40% 이상
    long_upper = (upper >= body * 1.5) & (upper >= rng * 0.4) & (rng > 0)

    # --- 이중 꼬리군: 연속 2개 긴 위꼬리 ---
    result["cdl_group_double_tail"] = (
        long_upper & long_upper.shift(1)
    ).fillna(False).astype(int)

    # --- 다중 꼬리군: 연속 3개+ 긴 위꼬리 ---
    result["cdl_group_multi_tail"] = (
        long_upper & long_upper.shift(1) & long_upper.shift(2)
    ).fillna(False).astype(int)

    # --- 후퇴 꼬리군: 3봉 연속 고가가 낮아지는 (고가 하락) ---
    retreating_highs = (
        (h < h.shift(1)) & (h.shift(1) < h.shift(2))
    )
    result["cdl_group_retreat_tail"] = (
        retreating_highs & (long_upper | long_upper.shift(1))
    ).fillna(False).astype(int)

    # --- 상기캔: 3봉 연속 저가가 점진적으로 상승 ---
    result["cdl_group_rising_low"] = (
        (l > l.shift(1)) & (l.shift(1) > l.shift(2))
    ).fillna(False).astype(int)

    # --- 수렴캔: 3봉 연속 고저 범위가 줄어듦 ---
    result["cdl_group_converging"] = (
        (rng < rng.shift(1)) & (rng.shift(1) < rng.shift(2)) & (rng > 0)
    ).fillna(False).astype(int)

    # --- 후퇴캔: 3봉 연속 고가가 점진적으로 하락 ---
    result["cdl_group_retreat_high"] = (
        (h < h.shift(1)) & (h.shift(1) < h.shift(2))
    ).fillna(False).astype(int)

    return result


def get_candle_group_signal(df: pd.DataFrame) -> pd.DataFrame:
    """캔들군 신호 종합 — 캔들마스터 방식

    캔들군 패턴을 종합하여 매수/매도 신호를 산출합니다.

    - 꼬리군 + 파동 위치가 후반부면 매수 신호 강화
    - 상기캔은 상승 전조 (매수 신호)
    - 수렴캔은 돌파 직전 신호 (매수 준비)
    - 후퇴캔/후퇴꼬리군은 약세 신호 (매도)

    Args:
        df: ``detect_candle_groups`` 를 거친 DataFrame.
            캔들군 컬럼이 없으면 내부에서 자동 호출합니다.

    Returns:
        candle_group_signal 컬럼이 추가된 DataFrame.
        1=매수, -1=매도, 0=중립
    """
    # 캔들군 컬럼이 없으면 자동 생성
    group_cols = [col for col in df.columns if col.startswith("cdl_group_")]
    if not group_cols:
        df = detect_candle_groups(df)

    result = df.copy()
    signal = pd.Series(0, index=result.index, dtype=int)

    # 매수 신호 가산
    # 상기캔 (저가 상승): 상승 전조 → +1
    if "cdl_group_rising_low" in result.columns:
        signal += result["cdl_group_rising_low"].fillna(0).astype(int)

    # 수렴캔: 돌파 직전 → +1
    if "cdl_group_converging" in result.columns:
        signal += result["cdl_group_converging"].fillna(0).astype(int)

    # 이중/다중 꼬리군: 저항선 형성 (매도 압력) → -1
    if "cdl_group_double_tail" in result.columns:
        signal -= result["cdl_group_double_tail"].fillna(0).astype(int)
    if "cdl_group_multi_tail" in result.columns:
        signal -= result["cdl_group_multi_tail"].fillna(0).astype(int)

    # 후퇴캔/후퇴꼬리군: 약세 → -1
    if "cdl_group_retreat_high" in result.columns:
        signal -= result["cdl_group_retreat_high"].fillna(0).astype(int)
    if "cdl_group_retreat_tail" in result.columns:
        signal -= result["cdl_group_retreat_tail"].fillna(0).astype(int)

    # 범위 클리핑: -1, 0, 1
    result["candle_group_signal"] = signal.clip(-1, 1)
    return result


def get_pattern_signal(df: pd.DataFrame) -> dict:
    """최근 봉(마지막 행)의 캔들 패턴 신호를 반환합니다.

    Args:
        df: :func:`detect_candle_patterns` 를 거친 DataFrame.
            아직 패턴 컬럼이 없으면 내부에서 자동 호출합니다.

    Returns:
        dict with keys:
            - ``direction``: ``"bullish"`` / ``"bearish"`` / ``"neutral"``
            - ``confidence``: 0.0 ~ 1.0 (가장 강한 패턴 기준)
            - ``patterns``: 감지된 패턴 리스트
              ``[{"name": str, "signal": int, "confidence": float}, ...]``
    """
    cdl_cols = [col for col in df.columns if col.startswith("cdl_")]
    if not cdl_cols:
        df = detect_candle_patterns(df)
        cdl_cols = [col for col in df.columns if col.startswith("cdl_")]

    last = df.iloc[-1]
    detected: list[dict] = []

    for col in cdl_cols:
        val = int(last[col])
        if val == 0:
            continue
        pattern_name = col.replace("cdl_", "")
        conf = PATTERN_CONFIG.get(pattern_name, {}).get("confidence", 0.5)
        detected.append({
            "name": pattern_name,
            "signal": val,
            "confidence": conf,
        })

    if not detected:
        return {
            "direction": "neutral",
            "confidence": 0.0,
            "patterns": [],
        }

    # 가장 높은 신뢰도 패턴 기준으로 종합 방향 결정
    detected.sort(key=lambda x: x["confidence"], reverse=True)
    top = detected[0]

    # 가중 합산으로 최종 방향 판단
    weighted_sum = sum(p["signal"] * p["confidence"] for p in detected)
    total_conf = sum(p["confidence"] for p in detected)
    avg_signal = weighted_sum / total_conf if total_conf else 0.0

    if avg_signal > 0.1:
        direction = "bullish"
    elif avg_signal < -0.1:
        direction = "bearish"
    else:
        direction = "neutral"

    return {
        "direction": direction,
        "confidence": round(top["confidence"], 2),
        "patterns": detected,
    }
