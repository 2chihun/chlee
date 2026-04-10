"""Triple-Barrier Labeling (López de Prado Ch3)

전통적 고정기간 수익률 라벨 대신, 3개의 장벽을 사용하여
적응적으로 라벨을 생성합니다:

  1. 상단 장벽 (take-profit): 목표 수익 달성
  2. 하단 장벽 (stop-loss): 손절 수준 도달
  3. 수직 장벽 (max holding): 최대 보유 기간 만료

Meta-Labeling (2단계 레이블링):
  - 1차 모델: 방향(매수/매도) 결정
  - 2차 모델: 실행 여부 및 포지션 크기 결정
  → 전략 신호의 정밀도(precision)를 ML로 향상
"""

import numpy as np
import pandas as pd
from typing import Optional
from loguru import logger


def get_daily_volatility(
    close: pd.Series,
    span: int = 20,
) -> pd.Series:
    """일간 변동성 추정 (지수이동평균 기반).

    Triple-Barrier에서 장벽 폭을 동적으로 설정할 때 사용합니다.

    Args:
        close: 종가 Series (DatetimeIndex)
        span: EWM 스팬 (기본 20일)

    Returns:
        일간 수익률의 지수가중 표준편차
    """
    returns = close.pct_change().dropna()
    vol = returns.ewm(span=span, min_periods=max(span // 2, 1)).std()
    return vol


def apply_triple_barrier(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: tuple[float, float] = (1.0, 1.0),
    molecule: Optional[list] = None,
) -> pd.DataFrame:
    """Triple-Barrier 라벨링을 적용합니다.

    Args:
        close: 종가 Series (DatetimeIndex)
        events: DataFrame with columns:
            - t1: 수직 장벽 (최대 보유 기간 만료일)
            - trgt: 목표 변동성 (장벽 폭)
            - side: (선택) 1차 모델의 방향 (1=매수, -1=매도)
        pt_sl: (profit_taking 배수, stop_loss 배수)
               0이면 해당 장벽 비활성
        molecule: 처리할 인덱스 부분집합 (병렬처리용)

    Returns:
        DataFrame with columns:
            - t1: 실제 장벽 도달 시점
            - sl: 하단 장벽 도달 시점 (또는 NaT)
            - pt: 상단 장벽 도달 시점 (또는 NaT)
    """
    if molecule is None:
        molecule = events.index.tolist()

    # 출력 DataFrame
    out = events[["t1"]].copy()
    out = out.loc[molecule]

    # 이익실현/손절 가격 수준 설정
    if pt_sl[0] > 0:
        pt = pt_sl[0] * events.loc[molecule, "trgt"]
    else:
        pt = pd.Series(index=molecule, dtype=float)  # 비활성

    if pt_sl[1] > 0:
        sl = -pt_sl[1] * events.loc[molecule, "trgt"]
    else:
        sl = pd.Series(index=molecule, dtype=float)  # 비활성

    # 각 이벤트에 대해 장벽 도달 시점 탐색
    for loc in molecule:
        try:
            # 이벤트 시작 → 수직 장벽까지의 종가 경로
            closing_prices = close.loc[loc: events.at[loc, "t1"]]
            if len(closing_prices) < 2:
                continue

            # 수익률 경로
            path = closing_prices / closing_prices.iloc[0] - 1

            # 방향 고려 (meta-labeling)
            if "side" in events.columns:
                path = path * events.at[loc, "side"]

            # 상단 장벽 도달 시점
            pt_touch = pd.NaT
            if loc in pt.index and not pd.isna(pt.at[loc]):
                mask = path[path >= pt.at[loc]]
                if len(mask) > 0:
                    pt_touch = mask.index[0]

            # 하단 장벽 도달 시점
            sl_touch = pd.NaT
            if loc in sl.index and not pd.isna(sl.at[loc]):
                mask = path[path <= sl.at[loc]]
                if len(mask) > 0:
                    sl_touch = mask.index[0]

            out.at[loc, "pt"] = pt_touch
            out.at[loc, "sl"] = sl_touch

        except Exception as e:
            logger.debug("Triple-Barrier 라벨링 오류 at {}: {}", loc, e)
            continue

    return out


def get_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: tuple[float, float] = (1.0, 1.0),
    target: pd.Series = None,
    max_holding_bars: int = 20,
    min_ret: float = 0.0,
    side: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """이벤트를 생성하고 Triple-Barrier를 적용합니다.

    Args:
        close: 종가 Series
        t_events: 이벤트 발생 시점 (CUSUM 필터 등에서 얻음)
        pt_sl: (profit-taking 배수, stop-loss 배수)
        target: 목표 변동성 Series (None이면 자동 계산)
        max_holding_bars: 최대 보유 기간 (bars)
        min_ret: 최소 목표 수익률
        side: 1차 모델 방향 예측 (meta-labeling용)

    Returns:
        events DataFrame (t1, trgt, side 포함)
    """
    # 타겟 변동성
    if target is None:
        target = get_daily_volatility(close)

    # t_events를 close 인덱스와 교차
    t_events = t_events[t_events.isin(close.index)]
    t_events = t_events[t_events.isin(target.index)]

    if len(t_events) == 0:
        logger.warning("유효한 이벤트가 없습니다.")
        return pd.DataFrame(columns=["t1", "trgt", "side"])

    # 수직 장벽: 최대 보유 기간
    t1 = pd.Series(pd.NaT, index=t_events)
    for i, dt_val in enumerate(t_events):
        loc = close.index.get_loc(dt_val)
        end_loc = min(loc + max_holding_bars, len(close) - 1)
        t1.iloc[i] = close.index[end_loc]

    # 이벤트 DataFrame 구성
    events = pd.DataFrame({
        "t1": t1,
        "trgt": target.loc[t_events].values,
    }, index=t_events)

    # 최소 수익률 필터
    events = events[events["trgt"] >= min_ret]

    # 방향 (meta-labeling)
    if side is not None:
        events["side"] = side.loc[events.index]
    else:
        events["side"] = 1  # 기본: 매수

    # Triple-Barrier 적용
    barrier_result = apply_triple_barrier(close, events, pt_sl)

    # 실제 도달 장벽으로 t1 업데이트 (가장 먼저 도달한 장벽)
    for col in ["pt", "sl"]:
        if col in barrier_result.columns:
            events[col] = barrier_result[col]

    # 최종 t1: pt, sl, 수직장벽 중 가장 먼저 도달한 시점
    events["t1_final"] = events[["t1", "pt", "sl"]].min(axis=1)
    events.loc[events["t1_final"].isna(), "t1_final"] = events.loc[
        events["t1_final"].isna(), "t1"
    ]

    return events


def get_labels(
    events: pd.DataFrame,
    close: pd.Series,
) -> pd.Series:
    """장벽 도달 결과로부터 라벨을 생성합니다.

    Returns:
        라벨 Series:
          1 = 상단 장벽 도달 (수익)
         -1 = 하단 장벽 도달 (손실)
          0 = 수직 장벽 만료 (중립)
    """
    labels = pd.Series(0, index=events.index, dtype=int)

    for idx in events.index:
        try:
            t1_val = events.at[idx, "t1_final"] if "t1_final" in events.columns else events.at[idx, "t1"]

            if pd.isna(t1_val) or t1_val not in close.index:
                continue

            ret = close.at[t1_val] / close.at[idx] - 1

            # 방향 보정
            if "side" in events.columns:
                ret = ret * events.at[idx, "side"]

            # 라벨 부여
            if "pt" in events.columns and not pd.isna(events.at[idx, "pt"]):
                if events.at[idx, "pt"] <= t1_val:
                    labels.at[idx] = 1
                    continue
            if "sl" in events.columns and not pd.isna(events.at[idx, "sl"]):
                if events.at[idx, "sl"] <= t1_val:
                    labels.at[idx] = -1
                    continue

            # 수직 장벽 도달: 수익률 부호로 결정
            if ret > 0:
                labels.at[idx] = 1
            elif ret < 0:
                labels.at[idx] = -1

        except Exception as e:
            logger.debug("라벨 생성 오류 at {}: {}", idx, e)
            continue

    return labels


def cusum_filter(
    close: pd.Series,
    threshold: float,
) -> pd.DatetimeIndex:
    """CUSUM 필터로 이벤트 발생 시점을 감지합니다 (Ch2).

    가격 변동의 누적합이 임계값을 초과하면 이벤트를 발생시킵니다.
    정보 유입 시점을 포착하는 데 유용합니다.

    Args:
        close: 종가 Series
        threshold: CUSUM 임계값 (변동성의 배수 등)

    Returns:
        이벤트 발생 시점의 DatetimeIndex
    """
    events = []
    s_pos = 0.0
    s_neg = 0.0
    diff = close.pct_change().dropna()

    for i in range(len(diff)):
        s_pos = max(0, s_pos + diff.iloc[i])
        s_neg = min(0, s_neg + diff.iloc[i])

        if s_neg < -threshold:
            s_neg = 0.0
            events.append(diff.index[i])
        elif s_pos > threshold:
            s_pos = 0.0
            events.append(diff.index[i])

    return pd.DatetimeIndex(events)
