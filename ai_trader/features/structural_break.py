"""Structural Break Tests (López de Prado Ch17)

구조적 변화(레짐 전환)를 통계적으로 감지합니다:

  1. CUSUM Test: 누적합 기반 — 평균 이탈 감지
     → 이벤트 샘플링 트리거로 사용 (Ch2와 연계)

  2. SADF (Supremum ADF): 폭발적 행태 감지
     → 버블/붕괴 시점 탐지 (기존 bubble_detector.py 보완)

  3. Chow-Type Breakpoint: 구조적 단절점 검정
     → 전략 파라미터 리셋 시점 결정

기존 bubble_detector.py(조상철 도서)는 규칙 기반 버블 감지인 반면,
이 모듈은 통계적 검정을 통한 과학적 감지를 제공합니다.
"""

import numpy as np
import pandas as pd
from typing import Optional
from loguru import logger


def cusum_test(
    series: pd.Series,
    threshold: float = 5.0,
) -> pd.DataFrame:
    """CUSUM 검정 (Cumulative Sum Test).

    평균 회귀 가정 하에서, 누적 이탈이 임계값을 초과하면
    구조적 변화를 감지합니다.

    Args:
        series: 시계열 (수익률 등)
        threshold: 감지 임계값 (표준편차 단위)

    Returns:
        DataFrame with columns:
            cusum_pos: 양(+) 방향 누적합
            cusum_neg: 음(-) 방향 누적합
            break_up: 상방 이탈 감지 (bool)
            break_down: 하방 이탈 감지 (bool)
    """
    mean = series.mean()
    std = series.std()

    if std == 0:
        return pd.DataFrame(index=series.index)

    # 표준화
    z = (series - mean) / std

    cusum_pos = pd.Series(0.0, index=series.index)
    cusum_neg = pd.Series(0.0, index=series.index)
    break_up = pd.Series(False, index=series.index)
    break_down = pd.Series(False, index=series.index)

    s_pos = 0.0
    s_neg = 0.0

    for i in range(len(z)):
        s_pos = max(0, s_pos + z.iloc[i])
        s_neg = min(0, s_neg + z.iloc[i])

        cusum_pos.iloc[i] = s_pos
        cusum_neg.iloc[i] = s_neg

        if s_pos > threshold:
            break_up.iloc[i] = True
            s_pos = 0.0  # 리셋
        if s_neg < -threshold:
            break_down.iloc[i] = True
            s_neg = 0.0  # 리셋

    return pd.DataFrame({
        "cusum_pos": cusum_pos,
        "cusum_neg": cusum_neg,
        "break_up": break_up,
        "break_down": break_down,
    }, index=series.index)


def sadf_test(
    log_prices: pd.Series,
    min_window: int = 20,
    max_lag: int = 1,
) -> pd.DataFrame:
    """SADF (Supremum Augmented Dickey-Fuller) 검정.

    Phillips, Shi, and Yu (2015)의 SADF 검정을 구현합니다.
    각 시점에서 ADF 통계량의 최대값을 계산하여
    폭발적(explosive) 행태를 감지합니다.

    Args:
        log_prices: 로그 가격 Series
        min_window: 최소 회귀 윈도우 크기 (기본 20)
        max_lag: ADF 검정 최대 래그

    Returns:
        DataFrame with columns:
            sadf: 각 시점의 SADF 통계량
            is_explosive: 폭발적 행태 감지 여부
    """
    n = len(log_prices)
    sadf_values = pd.Series(np.nan, index=log_prices.index)

    # 임계값 (Monte Carlo 시뮬레이션 근사)
    # 99% 수준: 대략 1.5~2.0 (샘플 크기에 따라)
    critical_value = 1.5

    for end in range(min_window + 10, n):
        max_adf = -np.inf

        for start in range(0, end - min_window + 1):
            window = log_prices.iloc[start: end + 1]
            adf_stat = _adf_regression(window, max_lag)

            if adf_stat is not None and adf_stat > max_adf:
                max_adf = adf_stat

        if max_adf > -np.inf:
            sadf_values.iloc[end] = max_adf

    return pd.DataFrame({
        "sadf": sadf_values,
        "is_explosive": sadf_values > critical_value,
    }, index=log_prices.index)


def _adf_regression(series: pd.Series, max_lag: int = 1) -> Optional[float]:
    """ADF 회귀의 t-통계량을 반환합니다.

    Δy_t = α + β*y_{t-1} + Σγ_i*Δy_{t-i} + ε_t
    H0: β=0 (단위근) vs H1: β<0 (정상) 또는 β>0 (폭발)

    Returns:
        β의 t-통계량 (None if error)
    """
    try:
        y = series.values
        n = len(y)
        if n < max_lag + 3:
            return None

        dy = np.diff(y)
        y_lag = y[max_lag:-1]
        dy_dep = dy[max_lag:]

        # 회귀 행렬 구성: [상수, y_{t-1}, Δy_{t-1}, ...]
        T = len(dy_dep)
        X = np.ones((T, 2 + max_lag))
        X[:, 1] = y_lag[:T]

        for lag in range(1, max_lag + 1):
            X[:, 1 + lag] = dy[max_lag - lag: max_lag - lag + T]

        # OLS
        XtX = X.T @ X
        det = np.linalg.det(XtX)
        if abs(det) < 1e-12:
            return None

        XtX_inv = np.linalg.inv(XtX)
        beta = XtX_inv @ (X.T @ dy_dep)

        # 잔차
        resid = dy_dep - X @ beta
        denom = T - len(beta)
        if denom <= 0:
            return None
        sigma2 = float(resid @ resid) / denom

        if sigma2 <= 0:
            return None

        # β (y_{t-1}의 계수)의 t-통계량
        se_beta = np.sqrt(sigma2 * XtX_inv[1, 1])
        if se_beta <= 0:
            return None

        t_stat = beta[1] / se_beta
        return float(t_stat)

    except Exception:
        return None


def chow_test(
    series: pd.Series,
    breakpoint_idx: int,
) -> dict:
    """Chow 검정 (구조적 단절 검정).

    주어진 시점에서 모델 파라미터가 변했는지 검정합니다.

    Args:
        series: 시계열
        breakpoint_idx: 단절점 위치 (정수 인덱스)

    Returns:
        dict with keys: f_stat, p_value, is_break
    """
    try:
        from scipy import stats as sp_stats
    except ImportError:
        return {"f_stat": None, "p_value": None, "is_break": False}

    y = series.values
    n = len(y)

    if breakpoint_idx < 5 or breakpoint_idx > n - 5:
        return {"f_stat": None, "p_value": None, "is_break": False}

    # 전체, 전반, 후반 회귀
    def _ols_sse(segment):
        T = len(segment)
        if T < 3:
            return None, 0
        X = np.column_stack([np.ones(T), np.arange(T)])
        try:
            beta = np.linalg.lstsq(X, segment, rcond=None)[0]
            resid = segment - X @ beta
            return float(resid @ resid), len(beta)
        except Exception:
            return None, 0

    sse_total, k = _ols_sse(y)
    sse_1, _ = _ols_sse(y[:breakpoint_idx])
    sse_2, _ = _ols_sse(y[breakpoint_idx:])

    if sse_total is None or sse_1 is None or sse_2 is None:
        return {"f_stat": None, "p_value": None, "is_break": False}

    sse_unrestricted = sse_1 + sse_2
    df1 = k
    df2 = n - 2 * k

    if df2 <= 0 or sse_unrestricted <= 0:
        return {"f_stat": None, "p_value": None, "is_break": False}

    f_stat = ((sse_total - sse_unrestricted) / df1) / (sse_unrestricted / df2)
    p_value = 1 - sp_stats.f.cdf(f_stat, df1, df2)

    return {
        "f_stat": float(f_stat),
        "p_value": float(p_value),
        "is_break": p_value < 0.05,
    }


def detect_regime_changes(
    close: pd.Series,
    cusum_threshold: float = 3.0,
    sadf_min_window: int = 30,
) -> pd.DataFrame:
    """레짐 변화를 종합 감지합니다.

    CUSUM + SADF를 결합하여 구조적 변화 시점을 식별합니다.

    Args:
        close: 종가 Series
        cusum_threshold: CUSUM 임계값
        sadf_min_window: SADF 최소 윈도우

    Returns:
        DataFrame with columns:
            cusum_break: CUSUM 감지 (up/down/none)
            sadf_explosive: SADF 폭발 감지 (bool)
            regime_change: 종합 레짐 변화 신호
    """
    returns = close.pct_change().dropna()

    # CUSUM
    cusum_result = cusum_test(returns, threshold=cusum_threshold)

    # SADF
    log_prices = np.log(close)
    sadf_result = sadf_test(log_prices, min_window=sadf_min_window)

    # 종합
    result = pd.DataFrame(index=close.index)

    if len(cusum_result) > 0:
        result["cusum_break"] = "none"
        if "break_up" in cusum_result.columns:
            up_idx = cusum_result.index[cusum_result["break_up"]]
            common_up = result.index.intersection(up_idx)
            result.loc[common_up, "cusum_break"] = "up"
        if "break_down" in cusum_result.columns:
            dn_idx = cusum_result.index[cusum_result["break_down"]]
            common_dn = result.index.intersection(dn_idx)
            result.loc[common_dn, "cusum_break"] = "down"

    if len(sadf_result) > 0:
        common_sadf = result.index.intersection(sadf_result.index)
        result.loc[common_sadf, "sadf_explosive"] = sadf_result.loc[common_sadf, "is_explosive"].values
        result.loc[common_sadf, "sadf_stat"] = sadf_result.loc[common_sadf, "sadf"].values

    # 종합 레짐 변화: CUSUM 이탈 또는 SADF 폭발
    result["regime_change"] = False
    if "cusum_break" in result.columns:
        result.loc[result["cusum_break"] != "none", "regime_change"] = True
    if "sadf_explosive" in result.columns:
        result.loc[result["sadf_explosive"] == True, "regime_change"] = True

    return result
