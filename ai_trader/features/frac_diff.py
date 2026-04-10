"""Fractionally Differentiated Features (López de Prado Ch5)

일반 차분(d=1)은 시계열의 기억(memory)을 완전히 파괴하고,
원본(d=0)은 비정상성(non-stationarity)을 유지합니다.

분수 차분(0 < d < 1)은 두 목표를 동시에 달성합니다:
  - 정상성 확보 (ADF 테스트 통과)
  - 기억 최대 보존 (원본과의 상관 유지)

최적 d ≈ 0.35~0.45가 대부분의 금융 시계열에서 관측됩니다.

핵심 알고리즘: FFD (Fixed-Width Window Fractional Differentiation)
  - 가중치가 threshold 이하로 떨어지면 절단 → 계산 효율화
  - Expanding Window 방식보다 안정적인 결과
"""

import numpy as np
import pandas as pd
from typing import Optional
from loguru import logger


def _get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """FFD 가중치를 생성합니다.

    이항급수 가중치: w_k = -w_{k-1} * (d - k + 1) / k
    |w_k| < threshold이면 절단합니다.

    Args:
        d: 분수 차분 차수 (0 < d < 1)
        threshold: 가중치 절단 임계값

    Returns:
        가중치 배열 (w_0=1, w_1, w_2, ...)
    """
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
        if k > 10000:  # 안전장치
            break
    return np.array(weights[::-1])  # 역순: 과거→현재


def frac_diff_ffd(
    series: pd.Series,
    d: float = 0.35,
    threshold: float = 1e-5,
) -> pd.Series:
    """Fixed-Width Window Fractional Differentiation.

    López de Prado Snippet 5.3 기반 구현입니다.

    Args:
        series: 원본 시계열 (종가 등)
        d: 분수 차분 차수 (기본 0.35)
        threshold: 가중치 절단 임계값

    Returns:
        분수 차분된 시계열 (앞쪽 NaN 포함)

    Example:
        >>> close = pd.Series(...)
        >>> fd = frac_diff_ffd(close, d=0.35)
    """
    weights = _get_weights_ffd(d, threshold)
    width = len(weights)
    n = len(series)

    if width > n:
        logger.warning(
            "FFD 가중치 길이({})가 데이터 길이({})보다 큽니다. d를 줄이거나 threshold를 높여주세요.",
            width, n
        )
        return pd.Series(np.nan, index=series.index)

    # 분수 차분 적용
    result = pd.Series(np.nan, index=series.index)
    values = series.values

    for i in range(width - 1, n):
        window = values[i - width + 1: i + 1]
        result.iloc[i] = np.dot(weights, window)

    return result


def find_optimal_d(
    series: pd.Series,
    d_range: tuple[float, float] = (0.0, 1.0),
    d_step: float = 0.05,
    threshold: float = 1e-5,
    adf_pvalue: float = 0.05,
    min_corr: float = 0.90,
) -> dict:
    """최적 분수 차분 차수 d를 자동 탐색합니다.

    ADF 테스트로 정상성을 확보하면서,
    원본과의 상관을 최대한 보존하는 최소 d를 찾습니다.

    Args:
        series: 원본 시계열
        d_range: 탐색 범위 (기본 0.0~1.0)
        d_step: 탐색 간격 (기본 0.05)
        threshold: FFD 가중치 임계값
        adf_pvalue: ADF 테스트 유의수준
        min_corr: 최소 상관계수

    Returns:
        dict with keys:
            optimal_d: 최적 d값
            adf_stat: ADF 통계량
            adf_pvalue: ADF p-value
            correlation: 원본과의 상관
            results: 전체 탐색 결과 리스트
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        logger.warning("statsmodels가 설치되지 않았습니다. 기본 d=0.35를 반환합니다.")
        return {
            "optimal_d": 0.35,
            "adf_stat": None,
            "adf_pvalue": None,
            "correlation": None,
            "results": [],
        }

    results = []
    optimal = None

    d_values = np.arange(d_range[0], d_range[1] + d_step, d_step)

    for d in d_values:
        d = round(d, 3)
        if d == 0:
            continue

        fd = frac_diff_ffd(series, d=d, threshold=threshold)
        valid = fd.dropna()

        if len(valid) < 30:
            continue

        # ADF 테스트
        try:
            adf_result = adfuller(valid, maxlag=1)
            p_val = adf_result[1]
        except Exception:
            continue

        # 원본과의 상관 (유효 구간만)
        common_idx = valid.index.intersection(series.index)
        corr = float(valid.loc[common_idx].corr(series.loc[common_idx]))

        entry = {
            "d": d,
            "adf_stat": float(adf_result[0]),
            "adf_pvalue": float(p_val),
            "correlation": corr,
            "valid_samples": len(valid),
        }
        results.append(entry)

        # 최적 d: ADF 통과 + 상관 최대인 최소 d
        if p_val < adf_pvalue and corr >= min_corr:
            if optimal is None or d < optimal["optimal_d"]:
                optimal = {
                    "optimal_d": d,
                    "adf_stat": float(adf_result[0]),
                    "adf_pvalue": float(p_val),
                    "correlation": corr,
                }

    if optimal is None:
        # ADF를 통과하는 최소 d (상관 조건 완화)
        for r in results:
            if r["adf_pvalue"] < adf_pvalue:
                optimal = {
                    "optimal_d": r["d"],
                    "adf_stat": r["adf_stat"],
                    "adf_pvalue": r["adf_pvalue"],
                    "correlation": r["correlation"],
                }
                break

    if optimal is None:
        optimal = {
            "optimal_d": 0.35,
            "adf_stat": None,
            "adf_pvalue": None,
            "correlation": None,
        }
        logger.warning("최적 d를 찾지 못했습니다. 기본값 0.35를 사용합니다.")

    optimal["results"] = results
    return optimal


def apply_frac_diff_to_ohlcv(
    df: pd.DataFrame,
    d: float = 0.35,
    threshold: float = 1e-5,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """OHLCV DataFrame의 가격 컬럼에 분수 차분을 적용합니다.

    Args:
        df: OHLCV DataFrame
        d: 분수 차분 차수
        threshold: FFD 가중치 임계값
        columns: 차분할 컬럼 (기본: close만)

    Returns:
        분수 차분 컬럼이 추가된 DataFrame
    """
    if columns is None:
        columns = ["close"]

    result = df.copy()

    for col in columns:
        if col not in df.columns:
            logger.warning("컬럼 '{}'이 DataFrame에 없습니다.", col)
            continue

        fd_col = f"{col}_fd{d}"
        result[fd_col] = frac_diff_ffd(df[col], d=d, threshold=threshold)

    return result
