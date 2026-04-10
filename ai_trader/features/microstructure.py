"""Microstructural Features (López de Prado Ch19)

시장 미시구조 지표를 통해 유동성 독성과 정보 비대칭을 측정합니다:

  1. VPIN (Volume-Synchronized Probability of Informed Trading)
     → 정보거래자 활동 비율 추정
     → Flash Crash 사전 경고 지표

  2. Kyle's Lambda
     → 가격 영향도 (1주문당 가격 변동)
     → 유동성 비용 측정

  3. Amihud Illiquidity
     → |수익률| / 거래대금
     → 비유동성 측정

실시간 모니터링 또는 전략 필터로 활용합니다.
유동성이 나쁜 시점에는 포지션 크기를 줄이거나 진입을 보류합니다.
"""

import numpy as np
import pandas as pd
from typing import Optional
from loguru import logger


def compute_vpin(
    close: pd.Series,
    volume: pd.Series,
    n_buckets: int = 50,
    window: int = 50,
) -> pd.Series:
    """VPIN (Volume-Synchronized Probability of Informed Trading).

    거래량 버킷으로 동기화한 후, 매수/매도 거래량 불균형을
    측정하여 정보거래 확률을 추정합니다.

    Args:
        close: 종가 Series
        volume: 거래량 Series
        n_buckets: VPIN 계산 버킷 수 (기본 50)
        window: 이동 윈도우 크기

    Returns:
        VPIN Series (0~1, 높을수록 정보거래 활발)
    """
    if len(close) < window + 10:
        return pd.Series(np.nan, index=close.index)

    # 수익률 기반 매수/매도 거래량 추정 (Bulk Classification)
    returns = close.pct_change()
    vol_std = returns.rolling(window=20, min_periods=5).std()

    # 정규 CDF로 매수 비율 추정
    from scipy import stats as sp_stats
    z_scores = returns / vol_std.replace(0, np.nan)
    z_scores = z_scores.fillna(0).clip(-5, 5)
    buy_pct = z_scores.apply(sp_stats.norm.cdf)

    # 매수/매도 거래량
    v_buy = volume * buy_pct
    v_sell = volume * (1 - buy_pct)

    # 거래량 버킷 분할 및 VPIN 계산
    total_volume = volume.sum()
    bucket_size = total_volume / n_buckets if n_buckets > 0 else volume.mean()

    if bucket_size <= 0:
        return pd.Series(np.nan, index=close.index)

    # 이동 윈도우 기반 VPIN (간소화)
    abs_imbalance = (v_buy - v_sell).abs()
    vol_sum = volume.rolling(window=window, min_periods=window // 2).sum()
    vpin = abs_imbalance.rolling(window=window, min_periods=window // 2).sum() / \
           vol_sum.replace(0, np.nan)

    vpin = vpin.clip(0, 1)
    return vpin


def kyles_lambda(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Kyle's Lambda (가격 영향도).

    회귀: Δp = λ * signed_volume + ε
    λ이 높을수록 유동성이 나쁨 (주문 하나가 가격에 큰 영향)

    Args:
        close: 종가 Series
        volume: 거래량 Series
        window: 회귀 윈도우

    Returns:
        Lambda Series (높을수록 비유동적)
    """
    returns = close.pct_change()

    # 부호 거래량: 수익률 부호 × 거래량
    signed_vol = np.sign(returns) * volume

    # 이동 윈도우 회귀
    lam = pd.Series(np.nan, index=close.index)

    for i in range(window, len(close)):
        y = returns.iloc[i - window + 1: i + 1].values
        x = signed_vol.iloc[i - window + 1: i + 1].values

        # NaN 제거
        mask = ~(np.isnan(y) | np.isnan(x))
        y_clean = y[mask]
        x_clean = x[mask]

        if len(y_clean) < 10 or np.std(x_clean) == 0:
            continue

        # OLS: y = a + λx
        x_mat = np.column_stack([np.ones(len(x_clean)), x_clean])
        try:
            beta = np.linalg.lstsq(x_mat, y_clean, rcond=None)[0]
            lam.iloc[i] = abs(beta[1])  # λ 절대값
        except Exception:
            continue

    return lam


def amihud_illiquidity(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Amihud 비유동성 비율.

    ILLIQ = |수익률| / 거래대금
    높을수록 비유동적 (적은 거래대금으로 큰 가격 변동)

    Args:
        close: 종가 Series
        volume: 거래량 Series
        window: 이동 평균 윈도우

    Returns:
        비유동성 Series (높을수록 비유동적)
    """
    returns = close.pct_change()
    dollar_volume = close * volume

    # 0 거래대금 처리
    dollar_volume = dollar_volume.replace(0, np.nan)

    illiq = returns.abs() / dollar_volume
    # 이동 평균 (스무딩)
    illiq_smooth = illiq.rolling(window=window, min_periods=window // 2).mean()

    return illiq_smooth


def compute_spread_estimator(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Corwin-Schultz 스프레드 추정기.

    고가/저가만으로 bid-ask spread를 추정합니다.
    실시간 호가 데이터 없이도 유동성을 추정할 수 있습니다.

    Args:
        high: 고가 Series
        low: 저가 Series
        window: 이동 평균 윈도우

    Returns:
        추정 스프레드 (%) Series
    """
    # Beckers (1983) 추정
    log_hl = np.log(high / low)
    log_hl_sq = log_hl ** 2

    # 2일 고저 비율
    beta = log_hl_sq.rolling(2).sum()
    gamma = (np.log(high.rolling(2).max() / low.rolling(2).min())) ** 2

    # 스프레드 추정
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    spread = spread.clip(0, None)  # 음수 제거

    return spread.rolling(window=window, min_periods=window // 2).mean() * 100


def microstructure_report(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """미시구조 지표 종합 리포트.

    OHLCV DataFrame을 받아 모든 미시구조 지표를 계산합니다.

    Args:
        df: OHLCV DataFrame (close, high, low, volume 필수)
        window: 이동 윈도우

    Returns:
        미시구조 지표가 추가된 DataFrame
    """
    result = df.copy()

    # VPIN
    if "close" in df.columns and "volume" in df.columns:
        result["vpin"] = compute_vpin(df["close"], df["volume"], window=window)

    # Kyle's Lambda
    if "close" in df.columns and "volume" in df.columns:
        result["kyles_lambda"] = kyles_lambda(df["close"], df["volume"], window=window)

    # Amihud
    if "close" in df.columns and "volume" in df.columns:
        result["amihud"] = amihud_illiquidity(df["close"], df["volume"], window=window)

    # Spread 추정
    if "high" in df.columns and "low" in df.columns:
        result["spread_est"] = compute_spread_estimator(df["high"], df["low"], window=window)

    # 유동성 위험 점수 (종합)
    risk_cols = [c for c in ["vpin", "kyles_lambda", "amihud"] if c in result.columns]
    if risk_cols:
        # 각 지표를 0~1로 정규화 후 평균
        for col in risk_cols:
            valid = result[col].dropna()
            if len(valid) > 0:
                col_min = valid.quantile(0.05)
                col_max = valid.quantile(0.95)
                if col_max > col_min:
                    result[f"{col}_norm"] = ((result[col] - col_min) / (col_max - col_min)).clip(0, 1)

        norm_cols = [c for c in result.columns if c.endswith("_norm")]
        if norm_cols:
            result["liquidity_risk"] = result[norm_cols].mean(axis=1)

    return result
