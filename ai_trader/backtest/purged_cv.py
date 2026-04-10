"""Purged K-Fold Cross-Validation (López de Prado Ch7)

시계열 ML에서 정보 유출(data leakage)을 방지하기 위해:
  1. Purging: 학습/검증 세트 간 라벨이 겹치는 관측을 제거
  2. Embargo: 퍼지 후 추가 보호 기간(gap)을 적용

일반 K-Fold는 시계열에서 미래 정보가 학습 세트로 유입되므로
과적합(overfitting)을 심각하게 유발합니다.
"""

import numpy as np
import pandas as pd
from typing import Optional


class PurgedKFoldCV:
    """Purged K-Fold Cross-Validation.

    시계열 라벨의 시작/끝 시점(t1)을 고려하여
    학습-검증 분할 시 정보 유출을 방지합니다.

    Args:
        n_splits: K-Fold 분할 수 (기본 5)
        purge_pct: 퍼지 비율 — 전체 기간 대비 제거 기간 (기본 1%)
        embargo_pct: 엠바고 비율 — 전체 기간 대비 보호 기간 (기본 1%)

    Example:
        >>> cv = PurgedKFoldCV(n_splits=5, purge_pct=0.01, embargo_pct=0.01)
        >>> for train_idx, test_idx in cv.split(X, t1=t1_series):
        ...     model.fit(X.iloc[train_idx], y.iloc[train_idx])
        ...     score = model.score(X.iloc[test_idx], y.iloc[test_idx])
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_pct: float = 0.01,
        embargo_pct: float = 0.01,
    ):
        if n_splits < 2:
            raise ValueError("n_splits은 2 이상이어야 합니다.")
        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        t1: Optional[pd.Series] = None,
    ):
        """퍼지된 학습/검증 인덱스를 생성합니다.

        Args:
            X: 특성 DataFrame (인덱스 = datetime)
            y: 라벨 Series (사용되지 않으나 sklearn 호환)
            t1: 각 관측의 라벨 종료 시점 Series
                (예: Triple-Barrier의 수평 장벽 도달 시점)
                None이면 퍼지 없이 일반 시계열 분할

        Yields:
            (train_indices, test_indices) 튜플
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # embargo 크기 (샘플 수)
        embargo_size = int(n_samples * self.embargo_pct)

        # fold 크기 결정
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        fold_starts = np.cumsum(np.r_[0, fold_sizes])

        for i in range(self.n_splits):
            test_start = fold_starts[i]
            test_end = fold_starts[i + 1]
            test_idx = indices[test_start:test_end]

            # 기본 학습 인덱스 (검증 세트 제외)
            train_idx = np.concatenate([
                indices[:test_start],
                indices[test_end:],
            ])

            # 퍼지: t1이 주어진 경우 라벨 겹침 제거
            if t1 is not None:
                train_idx = self._purge(
                    X, train_idx, test_idx, t1
                )

            # 엠바고: 검증 세트 직후 학습 샘플 제거
            if embargo_size > 0:
                embargo_start = test_end
                embargo_end = min(test_end + embargo_size, n_samples)
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
        """학습 세트에서 검증 세트와 라벨이 겹치는 관측을 제거합니다.

        겹침 조건: 학습 관측의 라벨 기간이 검증 기간과 교차
        - 학습 관측 시작(t_i) < 검증 끝(test_end) AND
        - 학습 관측 라벨 끝(t1_i) > 검증 시작(test_start)
        """
        test_times = X.index[test_idx]
        test_start = test_times.min()
        test_end = test_times.max()

        # 퍼지 대상: 학습 관측의 라벨 종료가 검증 시작 이후
        purge_mask = np.ones(len(train_idx), dtype=bool)

        for j, idx in enumerate(train_idx):
            label_end = t1.iloc[idx]
            obs_start = X.index[idx]

            # 관측 라벨이 검증 기간과 겹치면 제거
            if obs_start < test_end and label_end > test_start:
                purge_mask[j] = False

        return train_idx[purge_mask]

    def get_n_splits(self) -> int:
        """sklearn 호환: 분할 수 반환."""
        return self.n_splits


def purged_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    test_size: float = 0.2,
    purge_pct: float = 0.01,
) -> tuple:
    """단순 학습/테스트 분할 + 퍼지.

    시계열 끝부분을 테스트 세트로 사용하고,
    라벨 겹침 구간을 학습 세트에서 제거합니다.

    Args:
        X: 특성 DataFrame
        y: 라벨 Series
        t1: 라벨 종료 시점 Series
        test_size: 테스트 세트 비율 (기본 20%)
        purge_pct: 퍼지 비율

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    n = len(X)
    split_idx = int(n * (1 - test_size))

    # 퍼지 크기 계산
    purge_size = int(n * purge_pct)

    # 테스트: 뒷부분
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    # 학습: 앞부분에서 퍼지 적용
    train_end = max(split_idx - purge_size, 1)
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    return X_train, X_test, y_train, y_test
