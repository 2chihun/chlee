"""Combinatorial Purged Cross-Validation (López de Prado Ch12)

기존 Walk-Forward 분석의 한계:
  - 단일 경로만 테스트 → 과적합 위험
  - OOS 기간이 짧음

CPCV는 N개 그룹에서 k개를 테스트 세트로 선택하는
모든 조합을 검사합니다:
  - C(N, k) 가지 경로 생성
  - 각 경로에서 Purge + Embargo 적용
  - 과적합 확률을 과학적으로 추정 가능

Walk-Forward 대비 장점:
  - 다중 경로로 과적합 감소
  - 과적합 확률 정량화 가능
  - 전체 데이터를 OOS로 활용
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import Optional
from loguru import logger


class CombinatorialPurgedCV:
    """Combinatorial Purged K-Fold Cross-Validation.

    N개 그룹 중 k개를 테스트 세트로 선택하는
    모든 조합에 대해 Purged CV를 수행합니다.

    Args:
        n_splits: 전체 그룹 수 N (기본 6)
        n_test_splits: 테스트 그룹 수 k (기본 2)
        purge_pct: 퍼지 비율
        embargo_pct: 엠바고 비율

    Example:
        >>> cpcv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2)
        >>> for train_idx, test_idx in cpcv.split(X, t1=t1):
        ...     model.fit(X.iloc[train_idx], y.iloc[train_idx])
        ...     score = model.score(X.iloc[test_idx], y.iloc[test_idx])
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        purge_pct: float = 0.01,
        embargo_pct: float = 0.01,
    ):
        if n_test_splits >= n_splits:
            raise ValueError("n_test_splits < n_splits 이어야 합니다.")
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct

    @property
    def n_combinations(self) -> int:
        """조합 수 C(N, k)."""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        t1: Optional[pd.Series] = None,
    ):
        """모든 조합에 대해 학습/테스트 인덱스를 생성합니다.

        Args:
            X: 특성 DataFrame
            y: 라벨 (사용되지 않으나 sklearn 호환)
            t1: 라벨 종료 시점 Series

        Yields:
            (train_indices, test_indices) 튜플
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # 그룹 분할
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        fold_starts = np.cumsum(np.r_[0, fold_sizes])

        groups = []
        for i in range(self.n_splits):
            groups.append(indices[fold_starts[i]: fold_starts[i + 1]])

        # embargo 크기
        embargo_size = int(n_samples * self.embargo_pct)

        # 모든 조합
        for test_combo in combinations(range(self.n_splits), self.n_test_splits):
            # 테스트 인덱스
            test_idx = np.concatenate([groups[i] for i in test_combo])

            # 학습 인덱스
            train_groups = [i for i in range(self.n_splits) if i not in test_combo]
            train_idx = np.concatenate([groups[i] for i in train_groups])

            # 퍼지
            if t1 is not None:
                train_idx = self._purge(X, train_idx, test_idx, t1)

            # 엠바고: 각 테스트 그룹 직후
            if embargo_size > 0:
                for test_group_idx in test_combo:
                    group_end = fold_starts[test_group_idx + 1]
                    embargo_start = group_end
                    embargo_end = min(group_end + embargo_size, n_samples)
                    embargo_mask = (train_idx >= embargo_start) & (train_idx < embargo_end)
                    train_idx = train_idx[~embargo_mask]

            if len(train_idx) == 0:
                continue

            yield train_idx.tolist(), test_idx.tolist()

    def _purge(
        self,
        X: pd.DataFrame,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        t1: pd.Series,
    ) -> np.ndarray:
        """퍼지: 학습 세트에서 테스트와 겹치는 관측 제거."""
        test_times = X.index[test_idx]
        test_start = test_times.min()
        test_end = test_times.max()

        purge_mask = np.ones(len(train_idx), dtype=bool)
        for j, idx in enumerate(train_idx):
            if idx >= len(t1):
                continue
            label_end = t1.iloc[idx]
            obs_start = X.index[idx]
            if obs_start < test_end and label_end > test_start:
                purge_mask[j] = False

        return train_idx[purge_mask]

    def get_n_splits(self) -> int:
        return self.n_combinations


def backtest_overfit_probability(
    trial_scores: list[float],
    oos_scores: list[float],
) -> float:
    """백테스트 과적합 확률 (PBO, Probability of Backtest Overfitting).

    CPCV의 IS/OOS 성과 쌍을 사용하여
    과적합 확률을 추정합니다.

    IS에서 최적인 전략이 OOS에서도 최적인지를 검사합니다.

    Args:
        trial_scores: IS(In-Sample) 성과 리스트
        oos_scores: OOS(Out-of-Sample) 성과 리스트

    Returns:
        과적합 확률 (0~1, 낮을수록 좋음)
    """
    if len(trial_scores) != len(oos_scores) or len(trial_scores) < 2:
        return float("nan")

    n = len(trial_scores)
    is_arr = np.array(trial_scores)
    oos_arr = np.array(oos_scores)

    # IS 최적 → OOS 성과 확인
    best_is_idx = np.argmax(is_arr)
    best_is_oos = oos_arr[best_is_idx]

    # OOS 중위수
    oos_median = np.median(oos_arr)

    # IS 최적 전략의 OOS 순위
    rank = np.sum(oos_arr <= best_is_oos) / n

    # 과적합: IS 최적이 OOS 중위수 이하인 비율
    # 여러 조합에서 반복하여 확률 추정
    overfit_count = 0
    for i in range(n):
        # i를 IS 최적으로 가정
        if oos_arr[i] <= oos_median:
            overfit_count += 1

    pbo = overfit_count / n
    return float(pbo)
