"""Entropy Features (López de Prado Ch18)

시장 예측 가능성과 정보 내용을 엔트로피로 측정합니다:

  1. Shannon Entropy: 가격 변동 패턴의 불확실성
     → 높으면 예측 어려움 (random), 낮으면 패턴 존재

  2. Lempel-Ziv Complexity: 시계열 압축률
     → 높으면 복잡한 시장 (예측 어려움)
     → 낮으면 반복 패턴 (전략 기회)

  3. Approximate Entropy (ApEn): 시계열 규칙성
     → 낮으면 규칙적 (추세 전략 유리)
     → 높으면 불규칙 (평균 회귀 또는 관망)

  4. Plug-in Entropy: 이산화된 수익률의 Shannon 엔트로피

활용:
  - 전략 선택기: 엔트로피 낮음 → 추세추종, 높음 → 평균회귀
  - 포지션 크기: 엔트로피 높을 때 포지션 축소
"""

import numpy as np
import pandas as pd
from typing import Optional
from loguru import logger


def shannon_entropy(
    series: pd.Series,
    n_bins: int = 20,
) -> float:
    """Shannon 엔트로피를 계산합니다.

    Args:
        series: 시계열 (수익률 등)
        n_bins: 히스토그램 빈 수

    Returns:
        Shannon entropy (bits), 높을수록 불확실
    """
    valid = series.dropna()
    if len(valid) < 10:
        return float("nan")

    # 히스토그램으로 확률 분포 추정
    counts, _ = np.histogram(valid, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]

    # H = -Σ p * log2(p)
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


def lempel_ziv_complexity(
    series: pd.Series,
    threshold: float = 0.0,
) -> float:
    """Lempel-Ziv 복잡도를 계산합니다.

    시계열을 이진 문자열로 변환 후 LZ76 알고리즘으로
    복잡도를 측정합니다.

    Args:
        series: 시계열 (수익률 등)
        threshold: 이진 변환 임계값 (기본 0 = 부호 기준)

    Returns:
        정규화된 LZ 복잡도 (0~1, 1=완전 랜덤)
    """
    valid = series.dropna()
    if len(valid) < 10:
        return float("nan")

    # 이진 변환
    binary = "".join(["1" if x > threshold else "0" for x in valid])

    # LZ76 알고리즘
    n = len(binary)
    complexity = _lz76_count(binary)

    # 정규화: 랜덤 이진 문자열의 기대 복잡도 = n / log2(n)
    if n > 1:
        normalized = complexity * np.log2(n) / n
    else:
        normalized = 0.0

    return float(min(normalized, 1.0))


def _lz76_count(s: str) -> int:
    """Lempel-Ziv 76 복잡도 카운트."""
    n = len(s)
    if n == 0:
        return 0

    i = 0
    c = 1  # 첫 문자는 항상 새 패턴
    l = 1  # 현재 매칭 길이
    k = 1

    while k + l <= n:
        # s[0:k]에서 s[k:k+l]과 매칭되는 부분 탐색
        if s[k: k + l] in s[i: k + l - 1]:
            l += 1
        else:
            c += 1
            k += l
            l = 1

    return c


def approximate_entropy(
    series: pd.Series,
    m: int = 2,
    r: Optional[float] = None,
) -> float:
    """근사 엔트로피 (Approximate Entropy, ApEn).

    시계열의 규칙성/예측 가능성을 측정합니다.
    낮으면 규칙적, 높으면 불규칙.

    Args:
        series: 시계열
        m: 패턴 길이 (기본 2)
        r: 유사성 임계값 (기본: 0.2 * std)

    Returns:
        ApEn 값 (높을수록 불규칙)
    """
    valid = series.dropna().values
    n = len(valid)

    if n < m + 10:
        return float("nan")

    if r is None:
        r = 0.2 * np.std(valid)

    if r <= 0:
        return float("nan")

    def _phi(m_val):
        """m-길이 패턴의 유사 패턴 비율."""
        templates = np.array([valid[i: i + m_val] for i in range(n - m_val + 1)])
        count = np.zeros(len(templates))

        for i, t in enumerate(templates):
            # 체비셰프 거리로 유사성 판단
            diffs = np.max(np.abs(templates - t), axis=1)
            count[i] = np.sum(diffs <= r)

        # 로그 평균
        count = count / len(templates)
        return np.mean(np.log(count[count > 0]))

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    return float(phi_m - phi_m1)


def plugin_entropy(
    series: pd.Series,
    n_bins: int = 10,
    window: int = 50,
) -> pd.Series:
    """이동 윈도우 Plug-in 엔트로피.

    이산화된 수익률의 Shannon 엔트로피를 이동 윈도우로 계산합니다.

    Args:
        series: 시계열 (수익률)
        n_bins: 이산화 빈 수
        window: 이동 윈도우

    Returns:
        이동 엔트로피 Series
    """
    result = pd.Series(np.nan, index=series.index)

    for i in range(window, len(series)):
        chunk = series.iloc[i - window: i].dropna()
        if len(chunk) >= 10:
            result.iloc[i] = shannon_entropy(chunk, n_bins=n_bins)

    return result


def entropy_features(
    close: pd.Series,
    window: int = 50,
    n_bins: int = 10,
) -> pd.DataFrame:
    """엔트로피 기반 피처를 종합 생성합니다.

    Args:
        close: 종가 Series
        window: 이동 윈도우
        n_bins: 이산화 빈 수

    Returns:
        DataFrame with entropy features
    """
    returns = close.pct_change().dropna()

    result = pd.DataFrame(index=close.index)

    # 이동 Shannon 엔트로피
    result["shannon_entropy"] = plugin_entropy(returns, n_bins=n_bins, window=window)

    # 이동 Lempel-Ziv
    lz = pd.Series(np.nan, index=close.index)
    for i in range(window, len(returns)):
        chunk = returns.iloc[i - window: i]
        lz.iloc[i + 1] = lempel_ziv_complexity(chunk)  # +1: returns 오프셋
    result["lz_complexity"] = lz

    # 이동 ApEn
    apen = pd.Series(np.nan, index=close.index)
    for i in range(window, len(returns)):
        chunk = returns.iloc[i - window: i]
        apen.iloc[i + 1] = approximate_entropy(chunk)
    result["approx_entropy"] = apen

    # 전략 선택 신호: 엔트로피 낮음 → 추세추종(1), 높음 → 평균회귀(-1)
    if result["shannon_entropy"].notna().sum() > 10:
        median_ent = result["shannon_entropy"].median()
        result["strategy_signal"] = np.where(
            result["shannon_entropy"] < median_ent, 1, -1
        )
        result.loc[result["shannon_entropy"].isna(), "strategy_signal"] = 0

    return result
