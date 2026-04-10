"""Bet Sizing (López de Prado Ch10)

예측 확률을 포지션 크기로 변환하는 모듈입니다.

기존 risk/manager.py의 고정 비율 포지션 크기 결정과 달리,
ML 모델의 예측 확률에 비례하여 동적으로 크기를 조절합니다.

핵심 개념:
  1. 확률 → 크기 변환: sigmoid/linear 매핑
  2. Average Active Bets 정규화: 동시 포지션 수 고려
  3. Size Discretization: 연속 크기를 이산 단계로 변환
  4. Dynamic Limit Price: 확률 기반 지정가 계산

워크플로우:
  ML 예측 확률 → bet_size() → 이산화 → risk manager 전달
"""

import numpy as np
import pandas as pd
from typing import Optional
from scipy import stats as scipy_stats
from loguru import logger


def bet_size_from_prob(
    prob: pd.Series,
    num_classes: int = 2,
    method: str = "sigmoid",
) -> pd.Series:
    """예측 확률을 연속 베팅 크기로 변환합니다.

    Args:
        prob: 양성 클래스 확률 (0~1)
        num_classes: 분류 클래스 수 (기본 2: 매수/매도)
        method: 변환 방법
            "sigmoid": 역 정규 CDF 기반 (López de Prado 권장)
            "linear": 선형 매핑

    Returns:
        베팅 크기 (-1 ~ +1)
        양수 = 매수, 음수 = 매도, 0 = 미실행
    """
    if method == "sigmoid":
        return _sigmoid_bet_size(prob, num_classes)
    elif method == "linear":
        return _linear_bet_size(prob, num_classes)
    else:
        logger.warning("알 수 없는 method '{}', sigmoid 사용", method)
        return _sigmoid_bet_size(prob, num_classes)


def _sigmoid_bet_size(prob: pd.Series, num_classes: int = 2) -> pd.Series:
    """역 정규 CDF 기반 베팅 크기 (Ch10 Eq 10.1).

    확률이 0.5 근처 → 크기 ≈ 0 (확신 없음)
    확률이 0 또는 1 근처 → 크기 ≈ ±1 (높은 확신)
    """
    # 2p - 1: [-1, +1] 범위로 변환
    signal = 2 * prob - 1

    # 역 정규 CDF로 비선형 변환
    # signal을 (0, 1) 범위로 클리핑 후 ppf 적용
    clipped = signal.clip(-0.999, 0.999)

    # z-score 변환
    z = clipped.apply(lambda x: scipy_stats.norm.ppf((x + 1) / 2))

    # 다시 CDF로 변환하여 [-1, 1] 범위 확보
    size = z.apply(lambda x: 2 * scipy_stats.norm.cdf(x) - 1)

    return size


def _linear_bet_size(prob: pd.Series, num_classes: int = 2) -> pd.Series:
    """선형 베팅 크기.

    단순하지만 극단 확률에서 과대 평가 위험.
    """
    return 2 * prob - 1


def discretize_bet_size(
    size: pd.Series,
    n_steps: int = 10,
) -> pd.Series:
    """연속 베팅 크기를 이산 단계로 변환합니다.

    실제 주문에서는 연속 크기보다 이산 단계가 실용적입니다.
    (예: 10단계 = 10%, 20%, ..., 100%)

    Args:
        size: 연속 베팅 크기 (-1 ~ +1)
        n_steps: 이산 단계 수 (기본 10)

    Returns:
        이산화된 크기 (-1 ~ +1, n_steps 단계)
    """
    step = 1.0 / n_steps
    discretized = (size / step).round() * step
    return discretized.clip(-1, 1)


def normalize_by_active_bets(
    size: pd.Series,
    concurrent_bets: pd.Series,
    max_leverage: float = 1.0,
) -> pd.Series:
    """평균 활성 베팅 수로 크기를 정규화합니다.

    동시에 많은 포지션이 활성일 때, 각 포지션 크기를 줄여
    총 노출을 제한합니다.

    Args:
        size: 원본 베팅 크기
        concurrent_bets: 각 시점의 동시 활성 베팅 수
        max_leverage: 최대 레버리지 (기본 1.0 = 100%)

    Returns:
        정규화된 베팅 크기
    """
    avg_bets = concurrent_bets.replace(0, 1)
    normalized = size / avg_bets * max_leverage
    return normalized.clip(-1, 1)


def dynamic_limit_price(
    prob: pd.Series,
    current_price: pd.Series,
    spread_pct: float = 0.1,
) -> pd.DataFrame:
    """확률 기반 동적 지정가를 계산합니다.

    높은 확률 → 시장가에 가까운 지정가 (체결 우선)
    낮은 확률 → 유리한 지정가 (가격 우선)

    Args:
        prob: 예측 확률 (0~1)
        current_price: 현재가
        spread_pct: 최대 스프레드 비율 (%)

    Returns:
        DataFrame with columns:
            buy_limit: 매수 지정가
            sell_limit: 매도 지정가
    """
    # 확률이 높을수록 스프레드 작게 (체결 우선)
    # 확률이 낮을수록 스프레드 크게 (유리한 가격)
    spread_factor = 1 - (2 * prob - 1).abs()  # 0.5 근처 → 최대 스프레드
    spread = spread_factor * spread_pct / 100

    return pd.DataFrame({
        "buy_limit": (current_price * (1 - spread)).round().astype(int),
        "sell_limit": (current_price * (1 + spread)).round().astype(int),
    }, index=prob.index)


def compute_bet_sizes(
    predictions: pd.DataFrame,
    concurrent_bets: Optional[pd.Series] = None,
    n_steps: int = 10,
    method: str = "sigmoid",
    max_leverage: float = 1.0,
) -> pd.DataFrame:
    """전체 Bet Sizing 파이프라인.

    ML 예측 결과를 받아 최종 포지션 크기까지 산출합니다.

    Args:
        predictions: DataFrame with columns:
            prob: 양성 확률
            side: 방향 (1=매수, -1=매도)
        concurrent_bets: 동시 활성 베팅 수 (선택)
        n_steps: 이산화 단계
        method: 크기 변환 방법
        max_leverage: 최대 레버리지

    Returns:
        DataFrame with columns:
            raw_size: 원본 크기
            normalized_size: 정규화 크기
            final_size: 이산화된 최종 크기
    """
    result = pd.DataFrame(index=predictions.index)

    # 1) 확률 → 연속 크기
    raw_size = bet_size_from_prob(predictions["prob"], method=method)
    # 방향 적용
    if "side" in predictions.columns:
        raw_size = raw_size.abs() * predictions["side"].apply(np.sign)
    result["raw_size"] = raw_size

    # 2) 활성 베팅 정규화
    if concurrent_bets is not None:
        common = result.index.intersection(concurrent_bets.index)
        result.loc[common, "normalized_size"] = normalize_by_active_bets(
            raw_size.loc[common],
            concurrent_bets.loc[common],
            max_leverage,
        )
    else:
        result["normalized_size"] = raw_size

    # 3) 이산화
    result["final_size"] = discretize_bet_size(
        result["normalized_size"], n_steps
    )

    return result
