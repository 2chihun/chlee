"""Sample Weights (López de Prado Ch4)

금융 시계열은 IID가 아닙니다. 라벨이 시간적으로 겹치기 때문에:
  - 동시 활성 라벨이 많은 구간의 관측은 정보량이 낮음
  - 이를 보정하지 않으면 과적합 발생

핵심 개념:
  1. Concurrent Labels: 동시에 활성인 라벨 수 계산
  2. Uniqueness: 각 관측의 고유성 (= 1 / 동시 라벨 수)
  3. Sample Weight: 평균 고유성에 기반한 가중치
  4. Sequential Bootstrap: 비중복 샘플링

이 모듈의 가중치는 ML 학습 시 sample_weight로 전달됩니다.
"""

import numpy as np
import pandas as pd
from typing import Optional
from loguru import logger


def count_concurrent_events(
    t1: pd.Series,
    close_idx: pd.DatetimeIndex,
    molecule: Optional[list] = None,
) -> pd.Series:
    """각 시점에서 동시 활성 라벨 수를 계산합니다.

    Args:
        t1: 각 관측의 라벨 종료 시점 Series (index=시작, value=종료)
        close_idx: 전체 시계열 인덱스 (DatetimeIndex)
        molecule: 처리할 관측 부분집합 (병렬처리용)

    Returns:
        각 시점의 동시 활성 라벨 수 Series
    """
    if molecule is None:
        molecule = t1.index.tolist()

    # 시간 범위 내의 인덱스만 사용
    t1_filtered = t1.loc[molecule].dropna()

    # 전체 기간에 대한 동시 이벤트 수
    count = pd.Series(0, index=close_idx, dtype=int)

    for start, end in t1_filtered.items():
        # start ~ end 사이의 모든 시점에 +1
        mask = (close_idx >= start) & (close_idx <= end)
        count[mask] += 1

    return count


def compute_average_uniqueness(
    t1: pd.Series,
    close_idx: pd.DatetimeIndex,
    molecule: Optional[list] = None,
) -> pd.Series:
    """각 관측의 평균 고유성을 계산합니다.

    고유성 = 1 / 동시활성라벨수
    평균 고유성 = 라벨 기간 동안의 평균(고유성)

    Args:
        t1: 라벨 종료 시점 Series
        close_idx: 전체 시계열 인덱스
        molecule: 처리할 관측 부분집합

    Returns:
        각 관측의 평균 고유성 (0~1)
    """
    if molecule is None:
        molecule = t1.index.tolist()

    # 동시 이벤트 수
    num_conc = count_concurrent_events(t1, close_idx, molecule)

    # 각 관측의 평균 고유성
    uniqueness = pd.Series(0.0, index=molecule)

    for obs_start in molecule:
        if obs_start not in t1.index or pd.isna(t1.at[obs_start]):
            continue

        obs_end = t1.at[obs_start]
        mask = (close_idx >= obs_start) & (close_idx <= obs_end)
        concurrent = num_conc[mask]

        if len(concurrent) == 0 or concurrent.sum() == 0:
            continue

        # 고유성 = 1/concurrent, 평균
        u = (1.0 / concurrent).mean()
        uniqueness.at[obs_start] = float(u)

    return uniqueness


def get_sample_weights(
    t1: pd.Series,
    close_idx: pd.DatetimeIndex,
    returns: Optional[pd.Series] = None,
) -> pd.Series:
    """샘플 가중치를 계산합니다.

    가중치 = 평균 고유성 × |수익률| (수익률 제공 시)

    Args:
        t1: 라벨 종료 시점 Series
        close_idx: 전체 시계열 인덱스
        returns: 각 관측의 수익률 (선택, 절대값 사용)

    Returns:
        샘플 가중치 Series (합 = 1로 정규화)
    """
    uniqueness = compute_average_uniqueness(t1, close_idx)

    if returns is not None:
        # 수익률 크기에 비례한 가중치
        common = uniqueness.index.intersection(returns.index)
        weights = uniqueness.loc[common] * returns.loc[common].abs()
    else:
        weights = uniqueness

    # 0 가중치 → 최소값으로 대체
    weights = weights.replace(0, np.nan)
    if weights.notna().sum() > 0:
        weights = weights.fillna(weights.min())

    # 정규화 (합 = 관측 수)
    total = weights.sum()
    if total > 0:
        weights = weights / total * len(weights)

    return weights


def sequential_bootstrap(
    t1: pd.Series,
    close_idx: pd.DatetimeIndex,
    n_samples: Optional[int] = None,
    random_state: int = 42,
) -> list[int]:
    """순차 부트스트랩 (Sequential Bootstrap).

    일반 부트스트랩은 중복 정보가 높은 샘플을 과다 추출합니다.
    순차 부트스트랩은 각 단계에서 평균 고유성이 높은 샘플을
    우선적으로 선택합니다.

    Args:
        t1: 라벨 종료 시점 Series
        close_idx: 전체 시계열 인덱스
        n_samples: 추출할 샘플 수 (기본: 원본 크기)
        random_state: 난수 시드

    Returns:
        선택된 인덱스 위치 리스트 (정수)
    """
    rng = np.random.RandomState(random_state)
    n_obs = len(t1)

    if n_samples is None:
        n_samples = n_obs

    # 인디케이터 행렬: (시점 × 관측) — 각 관측이 활성인 시점 표시
    # 메모리 효율을 위해 dict of sets로 구현
    obs_active = {}  # obs_idx → set of time indices
    time_to_idx = {t: i for i, t in enumerate(close_idx)}

    for obs_idx, (start, end) in enumerate(zip(t1.index, t1.values)):
        if pd.isna(end):
            continue
        active_times = set()
        for t in close_idx:
            if t >= start and t <= end:
                active_times.add(time_to_idx.get(t, -1))
        active_times.discard(-1)
        obs_active[obs_idx] = active_times

    selected = []
    selected_active = set()  # 이미 선택된 샘플이 커버하는 시점

    for _ in range(n_samples):
        # 각 관측의 평균 고유성 계산 (현재 선택된 것 고려)
        probs = np.zeros(n_obs)

        for obs_idx, active in obs_active.items():
            if not active:
                continue
            # 겹치는 시점에서의 밀도
            overlap = active & selected_active
            if len(active) == 0:
                continue
            uniqueness = 1.0 - len(overlap) / len(active)
            probs[obs_idx] = max(uniqueness, 1e-10)

        total = probs.sum()
        if total <= 0:
            # 모든 관측이 비활성 → 균등 확률
            probs = np.ones(n_obs) / n_obs
        else:
            probs = probs / total

        # 확률에 따라 샘플 선택
        chosen = rng.choice(n_obs, p=probs)
        selected.append(chosen)

        # 선택된 관측의 활성 시점 추가
        if chosen in obs_active:
            selected_active |= obs_active[chosen]

    return selected
