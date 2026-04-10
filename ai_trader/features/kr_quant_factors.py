# -*- coding: utf-8 -*-
"""강환국 - 할 수 있다! 퀀트 투자

한국 시장 특화 퀀트 팩터 모듈:
1. 밸류 팩터 (PER/PBR 프록시)
2. 모멘텀 팩터 (3·6·12개월 수익률)
3. 퀄리티 팩터 (수익 안정성/성장 일관성)
4. 소형주 프리미엄 (거래량 기반 프록시)
5. 듀얼 모멘텀 (절대+상대 모멘텀 결합)
6. 계절성 효과 (한국 시장 1월·연말 효과)
7. 복합 전략 (밸류+모멘텀 결합)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class KRQuantSignal:
    """한국 퀀트팩터 신호"""
    # 복합 팩터 점수 (0~1)
    composite_score: float
    # 개별 팩터 점수
    value_score: float = 0.5
    momentum_score: float = 0.5
    quality_score: float = 0.5
    small_cap_score: float = 0.5
    # 듀얼 모멘텀 상태
    dual_momentum_buy: bool = False
    absolute_momentum_positive: bool = False
    relative_momentum_rank: float = 0.5
    # 계절성
    seasonal_factor: float = 1.0
    # 복합 전략 플래그
    value_momentum_combo: bool = False
    # 포지션 배수 (0.3~1.5)
    position_multiplier: float = 1.0
    # 신뢰도 조정 (-0.3~+0.3)
    confidence_adjustment: float = 0.0
    # 설명
    note: str = ""


class ValueFactorK:
    """한국 시장 밸류 팩터

    OHLCV 기반 가치 프록시:
    - 52주 최고/최저 위치 (낮을수록 저평가 가능)
    - MDD 최고 회복점
    - 장기 평균 대비 괴리율
    """

    def score(self, df: pd.DataFrame, lookback: int = 252) -> float:
        """밸류 팩터 점수 (0~1, 높을수록 저평가)"""
        if len(df) < lookback:
            return 0.5

        recent = df.tail(lookback)
        current = recent["close"].iloc[-1]

        # 52주 범위 내 위치 (낮을수록 저평가)
        high_52 = recent["high"].max()
        low_52 = recent["low"].min()
        if high_52 == low_52:
            range_pos = 0.5
        else:
            range_pos = 1.0 - (current - low_52) / (high_52 - low_52)

        # 장기 이평 대비 괴리율
        ma_long = recent["close"].mean()
        if ma_long > 0:
            deviation = 1.0 - (current / ma_long)
            deviation_score = float(np.clip(deviation + 0.5, 0, 1))
        else:
            deviation_score = 0.5

        # MDD 회복 잠재력
        peak = recent["close"].cummax()
        drawdown = (recent["close"] - peak) / peak
        current_dd = drawdown.iloc[-1]
        dd_score = float(np.clip(-current_dd * 5, 0, 1))

        score = range_pos * 0.40 + deviation_score * 0.35 + dd_score * 0.25
        return float(np.clip(score, 0.0, 1.0))


class MomentumFactorK:
    """한국 시장 모멘텀 팩터

    강환국 방식: 3·6·12개월 수익률 가중 합산
    """

    def score(self, df: pd.DataFrame) -> float:
        """모멘텀 팩터 점수 (0~1)"""
        if len(df) < 252:
            if len(df) < 60:
                return 0.5
            # 데이터 부족 시 가용 기간만 사용
            ret_short = df["close"].iloc[-1] / df["close"].iloc[0] - 1
            return float(np.clip(ret_short * 5 + 0.5, 0, 1))

        current = df["close"].iloc[-1]

        # 3개월 수익률
        ret_3m = current / df["close"].iloc[-63] - 1 if len(df) >= 63 else 0
        # 6개월 수익률
        ret_6m = current / df["close"].iloc[-126] - 1 if len(df) >= 126 else 0
        # 12개월 수익률
        ret_12m = current / df["close"].iloc[-252] - 1 if len(df) >= 252 else 0

        # 강환국: 12개월 40% + 6개월 30% + 3개월 30%
        raw = ret_12m * 0.40 + ret_6m * 0.30 + ret_3m * 0.30

        # -50%~+50% 범위를 0~1로 매핑
        score = float(np.clip(raw + 0.5, 0.0, 1.0))
        return score

    def absolute_momentum(self, df: pd.DataFrame, period: int = 252) -> bool:
        """절대 모멘텀: 최근 N일 수익률이 양수인가?"""
        if len(df) < period:
            return False
        return df["close"].iloc[-1] > df["close"].iloc[-period]

    def dual_momentum_signal(self, df: pd.DataFrame) -> tuple:
        """듀얼 모멘텀: (절대모멘텀 통과, 상대모멘텀 순위백분위)"""
        abs_mom = self.absolute_momentum(df)

        # 상대 모멘텀 (현재 수익률 vs 과거 수익률 평균)
        if len(df) < 252:
            return abs_mom, 0.5

        ret_12m = df["close"].iloc[-1] / df["close"].iloc[-252] - 1

        # 과거 12개월 수익률의 분포에서 현재 위치 계산
        rolling_ret = df["close"].pct_change(252)
        valid_ret = rolling_ret.dropna()
        if len(valid_ret) < 10:
            return abs_mom, 0.5

        rank = (valid_ret < ret_12m).sum() / len(valid_ret)
        return abs_mom, float(rank)


class QualityFactorK:
    """한국 시장 퀄리티 팩터

    OHLCV 기반 퀄리티 프록시:
    - 수익 안정성 (수익률 표준편차 역수)
    - 성장 일관성 (양의 수익월 비율)
    - 거래량 안정성 (급변 없는 안정 거래)
    """

    def score(self, df: pd.DataFrame, lookback: int = 252) -> float:
        """퀄리티 팩터 점수 (0~1)"""
        if len(df) < lookback:
            return 0.5

        recent = df.tail(lookback)
        returns = recent["close"].pct_change().dropna()

        # 1. 수익 안정성 (변동성 역수)
        vol = returns.std()
        stability = 1.0 / (1.0 + vol * 15)  # 0~1

        # 2. 성장 일관성 (양의 수익 비율)
        monthly_returns = recent["close"].resample("ME").last().pct_change().dropna() if hasattr(recent.index, 'freq') else returns.rolling(20).sum().iloc[::20]
        if len(returns) > 0:
            positive_ratio = (returns > 0).sum() / len(returns)
        else:
            positive_ratio = 0.5

        # 3. 거래량 안정성
        vol_cv = recent["volume"].std() / recent["volume"].mean() if recent["volume"].mean() > 0 else 1.0
        vol_stability = 1.0 / (1.0 + vol_cv)  # 0~1

        score = stability * 0.40 + positive_ratio * 0.35 + vol_stability * 0.25
        return float(np.clip(score, 0.0, 1.0))


class SmallCapPremium:
    """소형주 프리미엄 프록시

    거래량 기준으로 소형주 여부를 추정.
    한국 시장에서 소형주 프리미엄은 장기적으로 유효.
    """

    def score(self, df: pd.DataFrame, lookback: int = 60) -> float:
        """소형주 프리미엄 점수 (0~1, 높을수록 소형주 특성)"""
        if len(df) < lookback:
            return 0.5

        recent = df.tail(lookback)
        avg_volume = recent["volume"].mean()
        avg_value = avg_volume * recent["close"].mean()

        # 일평균 거래대금이 작을수록 소형주 (프리미엄 기대)
        # 10억 이하 = 1.0, 100억 이상 = 0.0
        if avg_value <= 0:
            return 0.5
        score = 1.0 - float(np.clip(np.log10(avg_value + 1) / 11.0, 0, 1))
        return float(np.clip(score, 0.0, 1.0))


class KoreanSeasonality:
    """한국 시장 계절성

    강환국이 검증한 한국 시장 계절적 패턴:
    - 1월 효과 (새해 매수세)
    - 11~12월 윈도우 드레싱
    - 4~5월 약세 ("Sell in May")
    - 수요일 최고, 월요일 최저
    """

    # 월별 초과 수익률 기반 계수 (한국 시장 백테스트 결과 프록시)
    MONTHLY_FACTOR = {
        1: 1.10,   # 1월 효과
        2: 1.05,   # 설 이후 반등
        3: 1.00,   # 보합
        4: 0.95,   # Sell in May 전조
        5: 0.90,   # Sell in May
        6: 0.95,   # 약세 유지
        7: 1.00,   # 보합
        8: 0.95,   # 여름 조정
        9: 1.00,   # 보합
        10: 1.05,  # 4분기 시작 반등
        11: 1.05,  # 연말 랠리 전
        12: 1.10,  # 윈도우 드레싱
    }

    # 요일별 계수 (0=월 ~ 4=금)
    WEEKDAY_FACTOR = {
        0: 0.95,   # 월요일 약세
        1: 1.00,   # 화요일 보합
        2: 1.05,   # 수요일 최강
        3: 1.00,   # 목요일 보합
        4: 1.00,   # 금요일 보합
    }

    def get_factor(self, dt: Optional[datetime] = None) -> float:
        """현재 시점의 계절성 계수 (0.85~1.15)"""
        if dt is None:
            dt = datetime.now()

        month_factor = self.MONTHLY_FACTOR.get(dt.month, 1.0)
        weekday_factor = self.WEEKDAY_FACTOR.get(dt.weekday(), 1.0)

        return month_factor * weekday_factor


class KRQuantAnalyzer:
    """한국 시장 퀀트 팩터 통합 분석기"""

    def __init__(
        self,
        lookback: int = 252,
        weight_value: float = 0.30,
        weight_momentum: float = 0.30,
        weight_quality: float = 0.20,
        weight_small_cap: float = 0.20,
    ):
        self.lookback = lookback
        self.weights = {
            "value": weight_value,
            "momentum": weight_momentum,
            "quality": weight_quality,
            "small_cap": weight_small_cap,
        }
        self.value = ValueFactorK()
        self.momentum = MomentumFactorK()
        self.quality = QualityFactorK()
        self.small_cap = SmallCapPremium()
        self.seasonality = KoreanSeasonality()

    def analyze(self, df: pd.DataFrame) -> KRQuantSignal:
        """한국 퀀트팩터 종합 분석"""
        if len(df) < 60:
            return KRQuantSignal(
                composite_score=0.5,
                position_multiplier=1.0,
                confidence_adjustment=0.0,
                note="데이터 부족 (최소 60봉 필요)",
            )

        try:
            # 개별 팩터 점수
            v_score = self.value.score(df, self.lookback)
            m_score = self.momentum.score(df)
            q_score = self.quality.score(df, self.lookback)
            sc_score = self.small_cap.score(df)

            # 듀얼 모멘텀
            abs_mom, rel_rank = self.momentum.dual_momentum_signal(df)
            dual_buy = abs_mom and rel_rank >= 0.5

            # 복합 전략 (밸류+모멘텀)
            vm_combo = v_score >= 0.6 and m_score >= 0.5

            # 계절성
            seasonal = self.seasonality.get_factor()

            # 복합 점수
            composite = (
                v_score * self.weights["value"]
                + m_score * self.weights["momentum"]
                + q_score * self.weights["quality"]
                + sc_score * self.weights["small_cap"]
            )
            # 계절성 보정
            composite *= seasonal
            composite = float(np.clip(composite, 0.0, 1.0))

            # 포지션 배수
            if composite >= 0.7 and dual_buy:
                multiplier = 1.4
            elif composite >= 0.6:
                multiplier = 1.2
            elif composite >= 0.4:
                multiplier = 1.0
            elif composite >= 0.3:
                multiplier = 0.7
            else:
                multiplier = 0.5

            # 절대 모멘텀 음수면 보수적
            if not abs_mom:
                multiplier *= 0.8

            multiplier = float(np.clip(multiplier, 0.3, 1.5))

            # 신뢰도 조정
            conf_adj = (composite - 0.5) * 0.5
            if vm_combo:
                conf_adj += 0.05
            conf_adj = float(np.clip(conf_adj, -0.3, 0.3))

            # 설명
            notes = []
            if v_score >= 0.7:
                notes.append(f"밸류 우수({v_score:.2f})")
            if m_score >= 0.7:
                notes.append(f"모멘텀 강세({m_score:.2f})")
            if dual_buy:
                notes.append("듀얼모멘텀 매수")
            if vm_combo:
                notes.append("밸류+모멘텀 복합")
            if seasonal > 1.05:
                notes.append(f"계절성 유리({seasonal:.2f})")
            elif seasonal < 0.95:
                notes.append(f"계절성 불리({seasonal:.2f})")

            return KRQuantSignal(
                composite_score=round(composite, 4),
                value_score=round(v_score, 4),
                momentum_score=round(m_score, 4),
                quality_score=round(q_score, 4),
                small_cap_score=round(sc_score, 4),
                dual_momentum_buy=dual_buy,
                absolute_momentum_positive=abs_mom,
                relative_momentum_rank=round(rel_rank, 4),
                seasonal_factor=round(seasonal, 4),
                value_momentum_combo=vm_combo,
                position_multiplier=round(multiplier, 2),
                confidence_adjustment=round(conf_adj, 4),
                note=" | ".join(notes) if notes else "정상 범위",
            )

        except Exception as e:
            logger.warning(f"KR 퀀트팩터 분석 오류: {e}")
            return KRQuantSignal(
                composite_score=0.5,
                position_multiplier=1.0,
                confidence_adjustment=0.0,
                note=f"분석 오류: {e}",
            )
