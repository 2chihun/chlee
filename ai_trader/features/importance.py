"""Feature Importance (López de Prado Ch8)

과적합을 방지하고 모델을 해석하기 위한 3가지 피처 중요도 방법:

  1. MDI (Mean Decrease Impurity)
     - 트리 분할 시 불순도 감소의 평균
     - 빠르지만, 상관된 피처에 편향
     - IS(In-Sample) 방식

  2. MDA (Mean Decrease Accuracy)
     - 피처를 셔플 후 정확도 감소 측정
     - OOS(Out-of-Sample) 방식 → 과적합 탐지에 강함
     - Purged K-Fold CV 사용 권장

  3. SFI (Single Feature Importance)
     - 각 피처를 단독으로 사용한 모델 성능
     - 피처 간 상호작용은 무시하나, 개별 예측력 측정

사용 시나리오:
  - MDI로 빠른 스크리닝 → MDA로 OOS 검증 → SFI로 개별 확인
  - 중요도 낮은 피처 제거 → 과적합 감소, 학습 속도 향상
"""

import numpy as np
import pandas as pd
from typing import Optional
from loguru import logger


def feature_importance_mdi(
    model,
    feature_names: list[str],
) -> pd.Series:
    """MDI (Mean Decrease Impurity) 피처 중요도.

    Random Forest의 기본 feature_importances_를 활용합니다.
    개별 트리의 불순도 감소를 평균합니다.

    Args:
        model: 학습된 RandomForestClassifier/Regressor
        feature_names: 피처 이름 리스트

    Returns:
        정규화된 피처 중요도 Series (합 = 1)
    """
    if not hasattr(model, "feature_importances_"):
        logger.error("모델에 feature_importances_ 속성이 없습니다.")
        return pd.Series(dtype=float)

    importance = pd.Series(
        model.feature_importances_,
        index=feature_names,
        name="mdi",
    )

    # 개별 트리의 분산도 계산
    if hasattr(model, "estimators_"):
        imp_array = np.array([
            tree.feature_importances_ for tree in model.estimators_
        ])
        imp_std = pd.Series(
            imp_array.std(axis=0),
            index=feature_names,
            name="mdi_std",
        )
        importance = pd.DataFrame({"mdi": importance, "mdi_std": imp_std})

    return importance


def feature_importance_mda(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    purge_pct: float = 0.01,
    embargo_pct: float = 0.01,
    t1: Optional[pd.Series] = None,
    sample_weight: Optional[np.ndarray] = None,
    scoring: str = "accuracy",
    n_repeats: int = 1,
) -> pd.DataFrame:
    """MDA (Mean Decrease Accuracy) 피처 중요도.

    Purged K-Fold CV를 사용하여 OOS에서 피처를 셔플 후
    정확도 감소를 측정합니다.

    Args:
        model: sklearn 분류기/회귀기 (clone 가능)
        X: 특성 DataFrame
        y: 라벨 Series
        n_splits: CV 분할 수
        purge_pct: 퍼지 비율
        embargo_pct: 엠바고 비율
        t1: 라벨 종료 시점 (Purged CV용)
        sample_weight: 샘플 가중치
        scoring: 평가 지표 ("accuracy", "f1", "neg_log_loss")
        n_repeats: 셔플 반복 횟수

    Returns:
        DataFrame with columns: mda (평균 감소), mda_std (표준편차)
    """
    try:
        from sklearn.base import clone
        from sklearn.metrics import accuracy_score, f1_score, log_loss
    except ImportError:
        logger.error("scikit-learn이 필요합니다.")
        return pd.DataFrame()

    from backtest.purged_cv import PurgedKFoldCV

    cv = PurgedKFoldCV(n_splits=n_splits, purge_pct=purge_pct, embargo_pct=embargo_pct)

    # 평가 함수 선택
    scorers = {
        "accuracy": lambda y_t, y_p: accuracy_score(y_t, y_p),
        "f1": lambda y_t, y_p: f1_score(y_t, y_p, average="weighted"),
        "neg_log_loss": lambda y_t, y_p: -log_loss(y_t, y_p),
    }
    score_fn = scorers.get(scoring, scorers["accuracy"])

    feature_names = list(X.columns)
    n_features = len(feature_names)

    # 결과 저장: (fold, repeat, feature) → 감소량
    importance_scores = {f: [] for f in feature_names}

    for train_idx, test_idx in cv.split(X, y, t1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        sw_train = None
        if sample_weight is not None:
            sw_train = sample_weight[train_idx]

        # 모델 학습
        mdl = clone(model)
        mdl.fit(X_train, y_train, sample_weight=sw_train)

        # 기준 점수
        y_pred = mdl.predict(X_test)
        base_score = score_fn(y_test, y_pred)

        # 각 피처 셔플
        for feat_idx, feat_name in enumerate(feature_names):
            for _ in range(n_repeats):
                X_test_shuffled = X_test.copy()
                X_test_shuffled.iloc[:, feat_idx] = np.random.permutation(
                    X_test_shuffled.iloc[:, feat_idx].values
                )
                y_pred_shuffled = mdl.predict(X_test_shuffled)
                shuffled_score = score_fn(y_test, y_pred_shuffled)

                # 감소량 (양수 = 중요)
                importance_scores[feat_name].append(base_score - shuffled_score)

    # 집계
    result = pd.DataFrame({
        "mda": {f: np.mean(scores) for f, scores in importance_scores.items()},
        "mda_std": {f: np.std(scores) for f, scores in importance_scores.items()},
    })

    return result.sort_values("mda", ascending=False)


def feature_importance_sfi(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    purge_pct: float = 0.01,
    embargo_pct: float = 0.01,
    t1: Optional[pd.Series] = None,
    sample_weight: Optional[np.ndarray] = None,
    scoring: str = "accuracy",
) -> pd.DataFrame:
    """SFI (Single Feature Importance).

    각 피처를 단독으로 사용하여 모델을 학습하고
    OOS 성능을 측정합니다.

    Args:
        model: sklearn 분류기/회귀기
        X: 특성 DataFrame
        y: 라벨 Series
        n_splits: CV 분할 수
        기타 파라미터: MDA와 동일

    Returns:
        DataFrame with columns: sfi (평균), sfi_std (표준편차)
    """
    try:
        from sklearn.base import clone
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError:
        logger.error("scikit-learn이 필요합니다.")
        return pd.DataFrame()

    from backtest.purged_cv import PurgedKFoldCV

    cv = PurgedKFoldCV(n_splits=n_splits, purge_pct=purge_pct, embargo_pct=embargo_pct)

    scorers = {
        "accuracy": lambda y_t, y_p: accuracy_score(y_t, y_p),
        "f1": lambda y_t, y_p: f1_score(y_t, y_p, average="weighted"),
    }
    score_fn = scorers.get(scoring, scorers["accuracy"])

    feature_names = list(X.columns)
    results = {}

    for feat_name in feature_names:
        scores = []
        X_single = X[[feat_name]]

        for train_idx, test_idx in cv.split(X, y, t1):
            X_train = X_single.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X_single.iloc[test_idx]
            y_test = y.iloc[test_idx]

            sw_train = None
            if sample_weight is not None:
                sw_train = sample_weight[train_idx]

            try:
                mdl = clone(model)
                mdl.fit(X_train, y_train, sample_weight=sw_train)
                y_pred = mdl.predict(X_test)
                scores.append(score_fn(y_test, y_pred))
            except Exception as e:
                logger.debug("SFI 오류 ({}): {}", feat_name, e)
                continue

        if scores:
            results[feat_name] = {
                "sfi": np.mean(scores),
                "sfi_std": np.std(scores),
            }

    return pd.DataFrame(results).T.sort_values("sfi", ascending=False)


def generate_importance_report(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    t1: Optional[pd.Series] = None,
    sample_weight: Optional[np.ndarray] = None,
    top_n: int = 10,
) -> dict:
    """MDI + MDA + SFI 종합 피처 중요도 보고서를 생성합니다.

    Args:
        model: 학습된 트리 기반 모델
        X: 특성 DataFrame
        y: 라벨 Series
        t1: 라벨 종료 시점 (선택)
        sample_weight: 샘플 가중치 (선택)
        top_n: 상위 N개 피처 표시

    Returns:
        dict with keys: mdi, mda, sfi, summary
    """
    feature_names = list(X.columns)
    report = {}

    # MDI
    logger.info("MDI 피처 중요도 계산 중...")
    mdi = feature_importance_mdi(model, feature_names)
    report["mdi"] = mdi

    # MDA
    logger.info("MDA 피처 중요도 계산 중...")
    try:
        mda = feature_importance_mda(
            model, X, y, t1=t1, sample_weight=sample_weight
        )
        report["mda"] = mda
    except Exception as e:
        logger.warning("MDA 계산 실패: {}", e)
        report["mda"] = pd.DataFrame()

    # SFI
    logger.info("SFI 피처 중요도 계산 중...")
    try:
        sfi = feature_importance_sfi(
            model, X, y, t1=t1, sample_weight=sample_weight
        )
        report["sfi"] = sfi
    except Exception as e:
        logger.warning("SFI 계산 실패: {}", e)
        report["sfi"] = pd.DataFrame()

    # 요약: 세 방법 모두에서 중요한 피처
    summary = []
    if isinstance(mdi, pd.DataFrame):
        top_mdi = set(mdi.nlargest(top_n, "mdi").index)
    elif isinstance(mdi, pd.Series):
        top_mdi = set(mdi.nlargest(top_n).index)
    else:
        top_mdi = set()

    top_mda = set(report["mda"].nlargest(top_n, "mda").index) if len(report["mda"]) > 0 else set()
    top_sfi = set(report["sfi"].nlargest(top_n, "sfi").index) if len(report["sfi"]) > 0 else set()

    # 교집합: 3가지 방법 모두에서 상위인 피처
    consensus = top_mdi & top_mda & top_sfi
    report["consensus_features"] = sorted(consensus)
    report["top_mdi"] = sorted(top_mdi)
    report["top_mda"] = sorted(top_mda)
    report["top_sfi"] = sorted(top_sfi)

    logger.info(
        "피처 중요도 보고서 완료: consensus={}, mdi_top={}, mda_top={}, sfi_top={}",
        len(consensus), len(top_mdi), len(top_mda), len(top_sfi)
    )

    return report
