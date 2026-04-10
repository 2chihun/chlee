"""HRP Portfolio Optimizer (López de Prado Ch16)

Hierarchical Risk Parity (HRP)는 전통적 Mean-Variance 최적화의
핵심 문제점을 해결합니다:

  1. 공분산 행렬 역행렬 불요 → 수치 안정성
  2. 트리 클러스터링 → 자산 간 계층 구조 활용
  3. 재귀적 이분법 → 분산 기반 가중치 배분

기존 risk/manager.py의 균등 배분 대비:
  - 상관관계가 높은 자산군 내 가중치 자동 분산
  - 변동성이 높은 자산 가중치 자동 감소
  - 행렬 역행렬 필요 없어 안정적

워크플로우:
  수익률 행렬 → 상관/공분산 → 트리 클러스터링 →
  준대각화 → 재귀 이분법 → 최종 가중치
"""

import numpy as np
import pandas as pd
from typing import Optional
from loguru import logger


def _correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """상관 행렬을 거리 행렬로 변환합니다.

    d(i,j) = sqrt(0.5 * (1 - ρ(i,j)))

    Args:
        corr: 상관 행렬

    Returns:
        거리 행렬
    """
    dist = ((1 - corr) / 2.0) ** 0.5
    return dist


def _tree_clustering(dist: pd.DataFrame) -> np.ndarray:
    """계층적 트리 클러스터링.

    Args:
        dist: 거리 행렬

    Returns:
        scipy linkage 행렬
    """
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage

    # 대각선 제거 + 대칭 보장
    dist_clean = dist.copy()
    np.fill_diagonal(dist_clean.values, 0)
    condensed = squareform(dist_clean.values)

    # Ward's method
    link = linkage(condensed, method="single")
    return link


def _quasi_diag(link: np.ndarray) -> list[int]:
    """준대각화: linkage를 정렬된 리프 인덱스로 변환합니다.

    클러스터 트리를 탐색하여 유사한 자산이 인접하도록
    인덱스를 재배열합니다.

    Args:
        link: scipy linkage 행렬

    Returns:
        정렬된 인덱스 리스트
    """
    link = link.astype(int)
    n = link[-1, 3]  # 전체 리프 수

    sort_idx = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = n

    while sort_idx.max() >= num_items:
        sort_idx.index = range(0, sort_idx.shape[0] * 2, 2)
        df0 = sort_idx[sort_idx >= num_items]
        i = df0.index
        j = df0.values - num_items

        sort_idx[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_idx = pd.concat([sort_idx, df1])
        sort_idx = sort_idx.sort_index()
        sort_idx.index = range(sort_idx.shape[0])

    return sort_idx.astype(int).tolist()


def _recursive_bisection(
    cov: pd.DataFrame,
    sorted_idx: list[int],
) -> pd.Series:
    """재귀적 이분법으로 가중치를 배분합니다.

    각 클러스터 쌍에 대해 역분산 비율로 가중치를 나눕니다.

    Args:
        cov: 공분산 행렬 (sorted_idx 순서)
        sorted_idx: 준대각화된 인덱스

    Returns:
        자산별 가중치 Series (합 = 1)
    """
    weights = pd.Series(1.0, index=sorted_idx)
    cluster_items = [sorted_idx]

    while len(cluster_items) > 0:
        new_clusters = []
        for sub_cluster in cluster_items:
            if len(sub_cluster) <= 1:
                continue

            # 이분할
            mid = len(sub_cluster) // 2
            left = sub_cluster[:mid]
            right = sub_cluster[mid:]

            # 각 클러스터의 분산 계산
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)

            # 역분산 비율
            total_var = var_left + var_right
            if total_var > 0:
                alpha = 1 - var_left / total_var  # left 가중치
            else:
                alpha = 0.5

            # 가중치 갱신
            weights[left] *= alpha
            weights[right] *= (1 - alpha)

            if len(left) > 1:
                new_clusters.append(left)
            if len(right) > 1:
                new_clusters.append(right)

        cluster_items = new_clusters

    return weights


def _cluster_variance(cov: pd.DataFrame, items: list[int]) -> float:
    """클러스터의 분산을 계산합니다.

    역분산 포트폴리오의 분산을 반환합니다.
    """
    sub_cov = cov.iloc[items, items]
    ivp = _inverse_variance_portfolio(sub_cov)
    var = float(ivp @ sub_cov @ ivp)
    return var


def _inverse_variance_portfolio(cov: pd.DataFrame) -> np.ndarray:
    """역분산 포트폴리오 가중치."""
    diag = np.diag(cov.values)
    diag = np.where(diag > 0, diag, 1e-10)
    ivp = 1.0 / diag
    ivp = ivp / ivp.sum()
    return ivp


def hrp_portfolio(
    returns: pd.DataFrame,
    cov_method: str = "standard",
) -> pd.Series:
    """HRP 포트폴리오 최적화.

    수익률 DataFrame을 입력받아 각 자산의 최적 가중치를 반환합니다.

    Args:
        returns: 자산 수익률 DataFrame (columns = 자산명)
        cov_method: 공분산 추정 방법
            "standard": 표본 공분산
            "shrinkage": Ledoit-Wolf 수축 추정

    Returns:
        가중치 Series (합 = 1, 자산명 인덱스)
    """
    n_assets = returns.shape[1]
    asset_names = list(returns.columns)

    if n_assets < 2:
        return pd.Series(1.0, index=asset_names)

    # 상관/공분산 행렬
    corr = returns.corr()
    if cov_method == "shrinkage":
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(returns.dropna())
            cov = pd.DataFrame(lw.covariance_, index=asset_names, columns=asset_names)
        except ImportError:
            cov = returns.cov()
    else:
        cov = returns.cov()

    # NaN 처리
    corr = corr.fillna(0)
    cov = cov.fillna(0)
    np.fill_diagonal(corr.values, 1.0)

    # 1) 거리 행렬
    dist = _correlation_distance(corr)

    # 2) 트리 클러스터링
    link = _tree_clustering(dist)

    # 3) 준대각화
    sorted_idx = _quasi_diag(link)

    # 4) 재귀적 이분법
    weights_by_idx = _recursive_bisection(cov, sorted_idx)

    # 인덱스 → 자산명 매핑
    weights = pd.Series(0.0, index=asset_names)
    for idx, w in weights_by_idx.items():
        if idx < len(asset_names):
            weights[asset_names[idx]] = w

    # 정규화
    total = weights.sum()
    if total > 0:
        weights = weights / total

    return weights


def compare_portfolios(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.03,
) -> pd.DataFrame:
    """HRP vs 균등배분 vs 역분산 포트폴리오를 비교합니다.

    Args:
        returns: 자산 수익률 DataFrame
        risk_free_rate: 무위험 수익률 (연)

    Returns:
        비교 결과 DataFrame
    """
    n_assets = returns.shape[1]
    results = {}

    # 1) 균등 배분
    ew = pd.Series(1.0 / n_assets, index=returns.columns)
    results["equal_weight"] = _portfolio_stats(returns, ew, risk_free_rate)

    # 2) 역분산
    cov = returns.cov()
    diag = np.diag(cov.values)
    diag = np.where(diag > 0, diag, 1e-10)
    ivp = 1.0 / diag
    ivp = ivp / ivp.sum()
    ivp_weights = pd.Series(ivp, index=returns.columns)
    results["inverse_variance"] = _portfolio_stats(returns, ivp_weights, risk_free_rate)

    # 3) HRP
    hrp_weights = hrp_portfolio(returns)
    results["hrp"] = _portfolio_stats(returns, hrp_weights, risk_free_rate)

    comparison = pd.DataFrame(results).T
    comparison.index.name = "portfolio"
    return comparison


def _portfolio_stats(
    returns: pd.DataFrame,
    weights: pd.Series,
    risk_free_rate: float,
) -> dict:
    """포트폴리오 성과 통계."""
    port_returns = (returns * weights).sum(axis=1)
    ann_return = float(port_returns.mean() * 252)
    ann_vol = float(port_returns.std() * np.sqrt(252))
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0

    # MDD
    cum = (1 + port_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    mdd = float(dd.min() * 100)

    return {
        "annual_return": round(ann_return * 100, 2),
        "annual_volatility": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(mdd, 2),
        "weights_concentration": round(float((weights ** 2).sum()), 4),
    }
