"""ML 기반 Meta-Labeling (López de Prado Ch3.6~3.7)

Quantamental 접근법의 핵심: 2단계 레이블링

  1차 모델 (Primary): 방향 결정 (매수/매도)
     → 기존 book_integrator 규칙 기반 시그널 활용

  2차 모델 (Secondary/Meta): 실행 여부 + 포지션 크기
     → ML(Random Forest 등)로 정밀도(precision) 최적화

왜 Meta-Labeling인가?
  - 1차 모델의 재현율(recall)은 높으나 정밀도(precision)가 낮을 때
  - ML이 "이번 신호를 실행할까?"를 학습 → F1 점수 향상
  - 포지션 크기도 확률에 비례하여 조절 가능
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class MetaLabelResult:
    """Meta-Labeling 결과"""
    signal_date: str           # 원본 시그널 날짜
    primary_side: int          # 1차 모델 방향 (1=매수, -1=매도)
    meta_prob: float           # 2차 모델 확률 (0~1)
    meta_decision: bool        # 실행 여부
    position_size: float       # 포지션 크기 (0~1)
    features_used: list = None # 사용된 피처 목록


class MetaLabeler:
    """ML 기반 Meta-Labeling 모델.

    1차 모델(규칙 기반)의 신호를 받아,
    2차 모델(ML)이 실행 여부와 크기를 결정합니다.

    Args:
        min_prob: 실행 임계 확률 (기본 0.5)
        max_position: 최대 포지션 비율 (기본 1.0)
        n_estimators: Random Forest 트리 수 (기본 100)
        lookback: 학습 데이터 기간 (기본 120 bars)
    """

    def __init__(
        self,
        min_prob: float = 0.5,
        max_position: float = 1.0,
        n_estimators: int = 100,
        lookback: int = 120,
    ):
        self.min_prob = min_prob
        self.max_position = max_position
        self.n_estimators = n_estimators
        self.lookback = lookback
        self.model = None
        self.is_fitted = False
        self.feature_names: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "MetaLabeler":
        """2차 모델을 학습합니다.

        Args:
            X: 특성 DataFrame (기술적/기본적 지표)
            y: 메타 라벨 (1=성공, 0=실패)
            sample_weight: 샘플 가중치 (Ch4)

        Returns:
            self
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            logger.error("scikit-learn이 설치되지 않았습니다: pip install scikit-learn")
            return self

        # NaN 제거
        valid_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        if len(X_clean) < 30:
            logger.warning("학습 데이터 부족 ({} < 30)", len(X_clean))
            return self

        self.feature_names = list(X_clean.columns)

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_features="sqrt",
            min_samples_leaf=max(5, len(X_clean) // 50),
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )

        sw = None
        if sample_weight is not None:
            sw = sample_weight[valid_mask]

        self.model.fit(X_clean, y_clean, sample_weight=sw)
        self.is_fitted = True

        # 학습 성능 로그
        train_score = self.model.score(X_clean, y_clean)
        pos_ratio = float(y_clean.mean())
        logger.info(
            "MetaLabeler 학습 완료: samples={}, accuracy={:.3f}, positive_ratio={:.3f}",
            len(X_clean), train_score, pos_ratio
        )

        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """실행 확률을 예측합니다.

        Args:
            X: 특성 DataFrame

        Returns:
            확률 Series (0~1)
        """
        if not self.is_fitted:
            logger.warning("모델이 학습되지 않았습니다. 기본 확률 0.5를 반환합니다.")
            return pd.Series(0.5, index=X.index)

        # 피처 정렬
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        X_filled = X_aligned.fillna(0)

        proba = self.model.predict_proba(X_filled)
        # 양성 클래스(1) 확률
        pos_idx = list(self.model.classes_).index(1) if 1 in self.model.classes_ else -1
        if pos_idx >= 0:
            return pd.Series(proba[:, pos_idx], index=X.index)
        return pd.Series(0.5, index=X.index)

    def predict(
        self,
        X: pd.DataFrame,
        primary_side: pd.Series,
    ) -> pd.DataFrame:
        """Meta-Labeling 예측 (실행 여부 + 크기).

        Args:
            X: 특성 DataFrame
            primary_side: 1차 모델 방향 (1=매수, -1=매도)

        Returns:
            DataFrame with columns:
                side: 최종 방향 (0=미실행, ±1=실행)
                size: 포지션 크기 (0~max_position)
                prob: 실행 확률
        """
        proba = self.predict_proba(X)

        result = pd.DataFrame(index=X.index)
        result["prob"] = proba
        result["execute"] = proba >= self.min_prob
        result["side"] = primary_side * result["execute"].astype(int)
        result["size"] = self._prob_to_size(proba)
        result.loc[~result["execute"], "size"] = 0.0

        return result

    def _prob_to_size(self, proba: pd.Series) -> pd.Series:
        """확률을 포지션 크기로 변환합니다.

        평균 활성 베팅을 정규화하여 과도한 포지션을 방지합니다.
        """
        # 선형 매핑: [min_prob, 1.0] → [0, max_position]
        denom = max(1.0 - self.min_prob, 1e-10)
        size = (proba - self.min_prob) / denom
        size = size.clip(0, 1) * self.max_position
        return size

    def get_feature_importance(self) -> pd.Series:
        """피처 중요도를 반환합니다."""
        if not self.is_fitted:
            return pd.Series(dtype=float)

        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
        ).sort_values(ascending=False)


def create_meta_labels(
    close: pd.Series,
    events: pd.DataFrame,
    primary_side: pd.Series,
) -> pd.Series:
    """1차 모델 방향 + Triple-Barrier 결과로 메타 라벨을 생성합니다.

    메타 라벨:
      1 = 1차 모델이 올바른 방향을 예측 (수익)
      0 = 1차 모델이 잘못된 방향을 예측 (손실)

    Args:
        close: 종가 Series
        events: Triple-Barrier events (t1_final 포함)
        primary_side: 1차 모델 방향 (1=매수, -1=매도)

    Returns:
        메타 라벨 Series (0 또는 1)
    """
    meta_labels = pd.Series(0, index=events.index, dtype=int)

    for idx in events.index:
        try:
            t1_col = "t1_final" if "t1_final" in events.columns else "t1"
            t1 = events.at[idx, t1_col]

            if pd.isna(t1) or t1 not in close.index:
                continue

            ret = close.at[t1] / close.at[idx] - 1

            # 1차 모델 방향과 실제 수익 방향이 일치하면 1
            if idx in primary_side.index:
                side = primary_side.at[idx]
                if side * ret > 0:
                    meta_labels.at[idx] = 1

        except Exception as e:
            logger.debug("메타 라벨 생성 오류 at {}: {}", idx, e)
            continue

    return meta_labels


def build_meta_features(
    df: pd.DataFrame,
    events_index: pd.DatetimeIndex,
    feature_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """메타 모델용 특성 DataFrame을 구성합니다.

    Args:
        df: 원본 OHLCV + 기술지표 DataFrame
        events_index: 이벤트 시점 인덱스
        feature_cols: 사용할 피처 컬럼 (None이면 수치형 전체)

    Returns:
        특성 DataFrame (events_index 기준)
    """
    if feature_cols is None:
        # 수치형 컬럼 자동 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 가격 관련 컬럼 제외 (과적합 방지)
        exclude = {"open", "high", "low", "close", "volume", "adj_close"}
        feature_cols = [c for c in numeric_cols if c.lower() not in exclude]

    if not feature_cols:
        logger.warning("사용 가능한 피처가 없습니다.")
        return pd.DataFrame(index=events_index)

    # 이벤트 시점의 피처 추출
    common_idx = events_index[events_index.isin(df.index)]
    features = df.loc[common_idx, feature_cols].copy()

    return features
