"""기술적 지표 계산 모듈

RSI, MACD, 볼린저밴드, VWAP, 이동평균선 등 단타/스윙 전략에 필요한 지표
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """단순이동평균"""
    return series.rolling(window=period, min_periods=1).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """지수이동평균"""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Relative Strength Index)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD (Moving Average Convergence Divergence)

    Returns:
        (macd_line, signal_line, histogram)
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    """볼린저밴드

    Returns:
        (upper, middle, lower, bandwidth, pct_b)
    """
    middle = sma(series, period)
    std = series.rolling(window=period, min_periods=1).std()
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    bandwidth = (upper - lower) / middle * 100
    pct_b = (series - lower) / (upper - lower)
    return upper, middle, lower, bandwidth, pct_b


def vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP (Volume Weighted Average Price)

    Args:
        df: 'high', 'low', 'close', 'volume' 컬럼이 필요한 DataFrame
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
    cumulative_vol = df["volume"].cumsum()
    return cumulative_tp_vol / cumulative_vol.replace(0, np.nan)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR (Average True Range)"""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=1).mean()


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """스토캐스틱 오실레이터

    Returns:
        (k_line, d_line)
    """
    lowest_low = df["low"].rolling(window=k_period, min_periods=1).min()
    highest_high = df["high"].rolling(window=k_period, min_periods=1).max()
    denom = highest_high - lowest_low
    k_line = 100 * (df["close"] - lowest_low) / denom.replace(0, np.nan)
    d_line = sma(k_line, d_period)
    return k_line, d_line


def obv(df: pd.DataFrame) -> pd.Series:
    """OBV (On Balance Volume)"""
    direction = np.sign(df["close"].diff())
    direction.iat[0] = 0
    return (direction * df["volume"]).cumsum()


def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """거래량 비율 (현재 거래량 / 평균 거래량)"""
    avg_vol = df["volume"].rolling(window=period, min_periods=1).mean()
    return df["volume"] / avg_vol.replace(0, np.nan)


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """윌리엄스 %R"""
    highest = df["high"].rolling(window=period, min_periods=1).max()
    lowest = df["low"].rolling(window=period, min_periods=1).min()
    denom = highest - lowest
    return -100 * (highest - df["close"]) / denom.replace(0, np.nan)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """CCI (Commodity Channel Index)"""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    mean_dev = typical.rolling(window=period, min_periods=1).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    sma_tp = sma(typical, period)
    return (typical - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """MFI (Money Flow Index) — 거래량 가중 RSI"""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = typical * df["volume"]
    direction = typical.diff()
    pos_flow = money_flow.where(direction > 0, 0.0)
    neg_flow = money_flow.where(direction < 0, 0.0)
    pos_sum = pos_flow.rolling(window=period, min_periods=1).sum()
    neg_sum = neg_flow.rolling(window=period, min_periods=1).sum()
    ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))


def support_resistance(df: pd.DataFrame, window: int = 20, tolerance: float = 0.02):
    """지지/저항 레벨 탐지

    Returns:
        (support_levels, resistance_levels) — 가격 레벨 리스트
    """
    highs = df["high"].values
    lows = df["low"].values
    supports, resistances = [], []

    for i in range(window, len(df) - window):
        # 저항선: 로컬 최고점
        if highs[i] == max(highs[i - window:i + window + 1]):
            resistances.append(float(highs[i]))
        # 지지선: 로컬 최저점
        if lows[i] == min(lows[i - window:i + window + 1]):
            supports.append(float(lows[i]))

    # 근접 레벨 합치기
    supports = _merge_levels(supports, tolerance)
    resistances = _merge_levels(resistances, tolerance)
    return supports, resistances


def _merge_levels(levels: list[float], tolerance: float) -> list[float]:
    """근접한 가격 레벨을 합칩니다."""
    if not levels:
        return []
    levels = sorted(levels)
    merged = [levels[0]]
    for lvl in levels[1:]:
        if abs(lvl - merged[-1]) / merged[-1] < tolerance:
            merged[-1] = (merged[-1] + lvl) / 2
        else:
            merged.append(lvl)
    return merged


def add_execution_strength(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """체결강도 = 매수체결량/매도체결량 * 100
    도서 p.88: "거래량이 대량으로 터져야 하며, 연속 체결로 주가의 체결강도가 강해져야 한다"
    간이 계산: 양봉 거래량 누적 / 음봉 거래량 누적 * 100 (period 기간)
    """
    result = df.copy()
    is_bullish = result["close"] >= result["open"]
    bull_vol = result["volume"].where(is_bullish, 0.0)
    bear_vol = result["volume"].where(~is_bullish, 0.0)
    bull_sum = bull_vol.rolling(window=period, min_periods=1).sum()
    bear_sum = bear_vol.rolling(window=period, min_periods=1).sum()
    result["execution_strength"] = bull_sum / bear_sum.replace(0, np.nan) * 100
    return result


def add_volume_spike(df: pd.DataFrame, threshold: float = 2.5) -> pd.DataFrame:
    """거래량 급증 감지 - 평균 대비 threshold 배 이상이면 1
    도서: 대량 거래 발생 시 추세 전환 가능성"""
    result = df.copy()
    avg_vol = result["volume"].rolling(window=20, min_periods=1).mean()
    result["volume_spike"] = (result["volume"] > avg_vol * threshold).astype(int)
    return result


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame에 모든 기술적 지표를 추가합니다.

    Args:
        df: 'open', 'high', 'low', 'close', 'volume' 컬럼이 필요

    Returns:
        지표 컬럼이 추가된 DataFrame
    """
    result = df.copy()
    close = result["close"].astype(float)

    # 이동평균선
    for p in [5, 10, 20, 60]:
        result[f"sma_{p}"] = sma(close, p)
        result[f"ema_{p}"] = ema(close, p)

    # RSI
    result["rsi_14"] = rsi(close, 14)
    result["rsi_9"] = rsi(close, 9)

    # MACD
    result["macd"], result["macd_signal"], result["macd_hist"] = macd(close)

    # 볼린저밴드
    result["bb_upper"], result["bb_mid"], result["bb_lower"], \
        result["bb_bandwidth"], result["bb_pctb"] = bollinger_bands(close)

    # VWAP
    result["vwap"] = vwap(result)

    # ATR
    result["atr_14"] = atr(result, 14)

    # 스토캐스틱
    result["stoch_k"], result["stoch_d"] = stochastic(result)

    # OBV
    result["obv"] = obv(result)

    # 거래량 비율
    result["vol_ratio"] = volume_ratio(result)

    # 윌리엄스 %R
    result["williams_r"] = williams_r(result)

    # CCI
    result["cci"] = cci(result)

    # MFI
    result["mfi"] = mfi(result)

    # 체결강도
    result = add_execution_strength(result)

    # 거래량 급증 감지
    result = add_volume_spike(result)

    return result
