"""홍용찬 "실전 퀀트투자" - 계량투자 팩터 분석 모듈

한국 시장 데이터(2000.7~2017.6) 기반 검증된 팩터들:
- 가치: PER/PBR/PSR/PCR 복합 분위수
- 수익성: ROE/ROA 품질, 영업이익률
- 성장성: 매출/영업이익 성장률 4유형
- 안전성: 부채비율, NCAV
- 배당: 배당+흑자 복합
- 모멘텀: 5-2 전략
- 캘린더: 월말월초, 수요일 효과
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Signal
# ──────────────────────────────────────────────

@dataclass
class QuantValueSignal:
    """퀀트 팩터 통합 시그널

    모든 점수는 0~1 범위 (1=최우수).
    """
    value_score: float = 0.5           # PER/PBR/PSR/PCR 복합 (0~1)
    profitability_score: float = 0.5   # ROE/ROA/이익률 (0~1)
    growth_score: float = 0.5          # 매출/영업이익 성장 (0~1)
    safety_score: float = 0.5          # 부채비율/NCAV (0~1)
    dividend_score: float = 0.5        # 배당+흑자 (0~1)
    momentum_score: float = 0.5        # 5-2 모멘텀 (0~1)
    calendar_adjustment: float = 1.0   # 캘린더 보정 계수 (0.9~1.1)
    composite_score: float = 0.5       # 전체 통합 (0~1)
    is_deep_value: bool = False        # 딥밸류 종목 여부
    position_multiplier: float = 1.0   # 포지션 배수 (0.5~1.5)
    confidence_delta: float = 0.0      # 신뢰도 보정
    note: str = ""


# ──────────────────────────────────────────────
# Helper: 가치 분위수 분석
# ──────────────────────────────────────────────

class ValuePercentileScorer:
    """PER/PBR/PSR/PCR 기반 저평가 분위수 점수

    가치지표가 낮을수록(저평가) 높은 점수 부여.
    OHLCV 기술적 프록시: 가격/거래량 비율로 밸류에이션 근사.
    """

    def __init__(self, lookback: int = 120):
        self.lookback = lookback

    def score(self, df: pd.DataFrame) -> float:
        if len(df) < self.lookback:
            return 0.5

        close = df["close"].astype(float).tail(self.lookback)
        volume = df["volume"].astype(float).tail(self.lookback)

        # 기술적 밸류 프록시: 가격 수준 역수 + 거래량 대비 가격
        price_level = close.iloc[-1]
        avg_price = close.mean()
        avg_volume = volume.mean()

        if avg_price <= 0 or avg_volume <= 0 or price_level <= 0:
            return 0.5

        # 저PER 프록시: 현재가 / 평균가 (낮을수록 저평가)
        price_ratio = price_level / avg_price

        # 저PBR 프록시: 가격 대비 거래량 밀도
        volume_density = (avg_volume * price_level) / (avg_price * avg_price)
        volume_density = min(volume_density, 5.0) / 5.0

        # 가격 변동 안정성 (저PSR/PCR 프록시)
        returns = close.pct_change().dropna()
        if len(returns) < 10:
            return 0.5
        downside = returns[returns < 0]
        downside_ratio = len(downside) / len(returns) if len(returns) > 0 else 0.5

        # 종합: 낮은 가격비율 + 높은 거래밀도 + 낮은 하방비율 = 저평가
        raw = (1.0 - min(price_ratio, 2.0) / 2.0) * 0.40 \
            + volume_density * 0.30 \
            + (1.0 - downside_ratio) * 0.30

        return float(np.clip(raw, 0, 1))


# ──────────────────────────────────────────────
# Helper: 수익성 품질
# ──────────────────────────────────────────────

class ProfitabilityScorer:
    """ROE/ROA/영업이익률 기술적 프록시

    수익 안정성과 추세 강도로 수익성 품질을 근사.
    """

    def __init__(self, lookback: int = 120):
        self.lookback = lookback

    def score(self, df: pd.DataFrame) -> float:
        if len(df) < self.lookback:
            return 0.5

        close = df["close"].astype(float).tail(self.lookback)
        returns = close.pct_change().dropna()

        if len(returns) < 20:
            return 0.5

        # ROE 프록시: 수익률 평균 (양의 수익률 지속 = 높은 ROE)
        mean_ret = returns.mean()
        ret_score = float(np.clip((mean_ret + 0.005) / 0.01, 0, 1))

        # ROA 프록시: 수익률 안정성 (낮은 변동성 = 안정적 ROA)
        volatility = returns.std()
        stability = float(np.clip(1.0 - volatility / 0.05, 0, 1))

        # 이익률 프록시: 양의 수익률 비율
        positive_ratio = (returns > 0).sum() / len(returns)

        score = ret_score * 0.40 + stability * 0.35 + positive_ratio * 0.25
        return float(np.clip(score, 0, 1))


# ──────────────────────────────────────────────
# Helper: 성장성 분석
# ──────────────────────────────────────────────

class GrowthAnalyzer:
    """매출/영업이익 성장률 4유형 기술적 프록시

    가격 추세 + 거래량 추세로 성장성을 근사.
    흑자전환 > 흑자지속 > 적자지속 > 적자전환
    """

    def __init__(self, lookback: int = 120):
        self.lookback = lookback

    def score(self, df: pd.DataFrame) -> float:
        if len(df) < self.lookback:
            return 0.5

        close = df["close"].astype(float).tail(self.lookback)
        volume = df["volume"].astype(float).tail(self.lookback)

        half = self.lookback // 2

        # 가격 추세 (매출 성장 프록시)
        first_half_avg = close.iloc[:half].mean()
        second_half_avg = close.iloc[half:].mean()
        if first_half_avg <= 0:
            return 0.5
        price_growth = (second_half_avg - first_half_avg) / first_half_avg

        # 거래량 추세 (영업이익 성장 프록시)
        vol_first = volume.iloc[:half].mean()
        vol_second = volume.iloc[half:].mean()
        if vol_first <= 0:
            vol_growth = 0.0
        else:
            vol_growth = (vol_second - vol_first) / vol_first

        # 4유형 분류
        price_up = price_growth > 0
        vol_up = vol_growth > 0

        if price_up and vol_up:
            base = 0.85      # 매출+영업이익 동시 증가
        elif price_up and not vol_up:
            base = 0.65      # 매출 증가, 영업이익 감소
        elif not price_up and vol_up:
            base = 0.45      # 매출 감소, 영업이익 증가
        else:
            base = 0.25      # 매출+영업이익 동시 감소

        # 성장 강도 보정
        growth_magnitude = float(np.clip(abs(price_growth) * 5, 0, 0.15))
        if price_up:
            base += growth_magnitude
        else:
            base -= growth_magnitude * 0.5

        return float(np.clip(base, 0, 1))


# ──────────────────────────────────────────────
# Helper: 안전성 분석
# ──────────────────────────────────────────────

class SafetyAnalyzer:
    """부채비율/NCAV 기술적 프록시

    가격 하방 리스크와 변동성 패턴으로 안전성을 근사.
    부채비율 100% 이하 → 안전, NCAV > 1 → 저평가 안전자산.
    """

    def __init__(self, lookback: int = 120, debt_penalty_threshold: float = 0.03):
        self.lookback = lookback
        self.debt_penalty_threshold = debt_penalty_threshold

    def score(self, df: pd.DataFrame) -> float:
        if len(df) < self.lookback:
            return 0.5

        close = df["close"].astype(float).tail(self.lookback)
        low = df["low"].astype(float).tail(self.lookback)
        high = df["high"].astype(float).tail(self.lookback)

        returns = close.pct_change().dropna()
        if len(returns) < 20:
            return 0.5

        # 부채비율 프록시: 최대 낙폭 (MDD) - 높은 MDD = 높은 부채
        cummax = close.cummax()
        drawdown = (close - cummax) / cummax
        max_dd = abs(drawdown.min())
        debt_proxy = float(np.clip(1.0 - max_dd / 0.30, 0, 1))

        # NCAV 프록시: 지지선 강도 (저점 지지 = 순자산 지지)
        support_ratio = low.iloc[-1] / close.iloc[-1] if close.iloc[-1] > 0 else 0.5
        ncav_proxy = float(np.clip(support_ratio, 0.5, 1.0)) * 2 - 1

        # 변동성 안정성
        vol = returns.std()
        vol_safety = float(np.clip(1.0 - vol / self.debt_penalty_threshold, 0, 1))

        score = debt_proxy * 0.40 + ncav_proxy * 0.30 + vol_safety * 0.30
        return float(np.clip(score, 0, 1))


# ──────────────────────────────────────────────
# Helper: 배당+흑자 복합
# ──────────────────────────────────────────────

class DividendProfitabilityScorer:
    """배당수익률 + 흑자 복합 점수

    흑자+배당 기업이 최고 수익률 (4유형 중 1위).
    기술적 프록시: 안정적 양의 수익률 + 낮은 변동성.
    """

    def __init__(self, lookback: int = 252):
        self.lookback = lookback

    def score(self, df: pd.DataFrame) -> float:
        if len(df) < min(self.lookback, 60):
            return 0.5

        actual_lookback = min(self.lookback, len(df))
        close = df["close"].astype(float).tail(actual_lookback)
        returns = close.pct_change().dropna()

        if len(returns) < 20:
            return 0.5

        # 흑자 프록시: 장기 수익률 양수 여부
        total_return = (close.iloc[-1] / close.iloc[0]) - 1
        is_profitable = total_return > 0

        # 배당 프록시: 수익률 안정성 (배당주는 안정적)
        vol = returns.std()
        stability = float(np.clip(1.0 - vol / 0.04, 0, 1))

        # 양의 수익일 비율
        pos_ratio = (returns > 0).sum() / len(returns)

        if is_profitable and stability > 0.5:
            base = 0.80   # 흑자 + 배당 유사
        elif is_profitable:
            base = 0.60   # 흑자 + 무배당 유사
        elif stability > 0.5:
            base = 0.40   # 적자 + 배당 유사
        else:
            base = 0.20   # 적자 + 무배당

        adjustment = (pos_ratio - 0.5) * 0.2
        return float(np.clip(base + adjustment, 0, 1))


# ──────────────────────────────────────────────
# Helper: 5-2 모멘텀
# ──────────────────────────────────────────────

class MomentumScorer:
    """5-2 모멘텀 전략 점수

    과거 5개월 수익률 상위 종목이 이후 2개월 우수 성과.
    OHLCV 기반: 중기 모멘텀 강도 측정.
    """

    def __init__(self, momentum_period: int = 100, evaluation_period: int = 40):
        self.momentum_period = momentum_period
        self.evaluation_period = evaluation_period

    def score(self, df: pd.DataFrame) -> float:
        required = self.momentum_period + self.evaluation_period
        if len(df) < required:
            return 0.5

        close = df["close"].astype(float)

        # 5개월(100거래일) 모멘텀
        momentum_start = close.iloc[-(required)]
        momentum_end = close.iloc[-(self.evaluation_period)]
        if momentum_start <= 0:
            return 0.5
        momentum_return = (momentum_end - momentum_start) / momentum_start

        # 최근 2개월 확인
        recent_return = (close.iloc[-1] - momentum_end) / momentum_end \
            if momentum_end > 0 else 0

        # 모멘텀 지속성 보너스
        continuation = 1.0 if (momentum_return > 0 and recent_return > 0) else 0.0

        raw = float(np.clip((momentum_return + 0.1) / 0.3, 0, 1)) * 0.60 \
            + float(np.clip((recent_return + 0.05) / 0.15, 0, 1)) * 0.25 \
            + continuation * 0.15

        return float(np.clip(raw, 0, 1))


# ──────────────────────────────────────────────
# Helper: 캘린더 효과 보정
# ──────────────────────────────────────────────

class CalendarEffectAdjuster:
    """캘린더 효과 보정 계수

    한국 시장 검증 결과:
    - 월말월초 효과: 1일 수익률 0.477% vs 전체 0.013% (36배)
    - 수요일 효과: 주중 최고 수익률
    - 1월 효과: 소형주 1월 우위
    """

    # 요일별 보정 (월~금, 한국시장 실증)
    WEEKDAY_FACTOR = {
        0: 1.01,   # 월
        1: 1.00,   # 화
        2: 1.03,   # 수 (최고)
        3: 1.00,   # 목
        4: 0.99,   # 금
    }

    # 월별 보정
    MONTH_FACTOR = {
        1: 1.03,   # 1월 효과
        2: 1.01,
        3: 1.00,
        4: 1.00,
        5: 0.99,
        6: 1.00,
        7: 1.01,   # 리밸런싱 시즌
        8: 0.99,
        9: 0.99,
        10: 1.00,
        11: 1.02,
        12: 1.01,
    }

    def adjust(self, df: pd.DataFrame) -> float:
        """현재 시점의 캘린더 보정 계수 반환 (0.9~1.1)"""
        if len(df) < 1:
            return 1.0

        last_dt = df["datetime"].iloc[-1] if "datetime" in df.columns else None
        if last_dt is None:
            return 1.0

        try:
            ts = pd.Timestamp(last_dt)
        except Exception:
            return 1.0

        weekday = ts.weekday()
        month = ts.month
        day = ts.day

        factor = 1.0
        factor *= self.WEEKDAY_FACTOR.get(weekday, 1.0)
        factor *= self.MONTH_FACTOR.get(month, 1.0)

        # 월말월초 효과 (25일 이후 또는 5일 이전)
        if day >= 25 or day <= 5:
            factor *= 1.02

        return float(np.clip(factor, 0.90, 1.10))


# ──────────────────────────────────────────────
# Integration: 퀀트밸류 통합 분석기
# ──────────────────────────────────────────────

class QuantValueAnalyzer:
    """홍용찬 실전 퀀트투자 통합 분석

    6개 팩터를 가중 결합하여 통합 점수 산출.
    가치 40% + 수익성 20% + 성장성 15% + 안전성 15% + 배당 10%.
    캘린더/모멘텀은 타이밍 보정으로 별도 적용.
    """

    def __init__(self, **kwargs):
        lookback = kwargs.get("lookback", 120)
        self._value_scorer = ValuePercentileScorer(lookback=lookback)
        self._profit_scorer = ProfitabilityScorer(lookback=lookback)
        self._growth_analyzer = GrowthAnalyzer(lookback=lookback)
        self._safety_analyzer = SafetyAnalyzer(
            lookback=lookback,
            debt_penalty_threshold=kwargs.get("debt_penalty_threshold", 0.03),
        )
        self._dividend_scorer = DividendProfitabilityScorer(
            lookback=kwargs.get("dividend_lookback", 252),
        )
        self._momentum_scorer = MomentumScorer(
            momentum_period=kwargs.get("momentum_period", 100),
            evaluation_period=kwargs.get("evaluation_period", 40),
        )
        self._calendar_adjuster = CalendarEffectAdjuster()

        # 가중치
        self._w_value = kwargs.get("weight_value", 0.40)
        self._w_profit = kwargs.get("weight_profitability", 0.20)
        self._w_growth = kwargs.get("weight_growth", 0.15)
        self._w_safety = kwargs.get("weight_safety", 0.15)
        self._w_dividend = kwargs.get("weight_dividend", 0.10)

        # 포지션 한계
        self._max_mult = kwargs.get("max_position_multiplier", 1.5)
        self._min_mult = kwargs.get("min_position_multiplier", 0.5)
        self._deep_value_threshold = kwargs.get("deep_value_threshold", 0.70)

    def analyze(self, df: pd.DataFrame) -> QuantValueSignal:
        """통합 퀀트팩터 분석"""
        if len(df) < 30:
            return QuantValueSignal(note="데이터 부족")

        # 1. 각 팩터 점수 산출
        value = self._value_scorer.score(df)
        profit = self._profit_scorer.score(df)
        growth = self._growth_analyzer.score(df)
        safety = self._safety_analyzer.score(df)
        dividend = self._dividend_scorer.score(df)
        momentum = self._momentum_scorer.score(df)
        calendar = self._calendar_adjuster.adjust(df)

        # 2. 가중 통합 점수
        composite = (
            value * self._w_value
            + profit * self._w_profit
            + growth * self._w_growth
            + safety * self._w_safety
            + dividend * self._w_dividend
        )
        composite = float(np.clip(composite, 0, 1))

        # 3. 딥밸류 판정
        is_deep = composite >= self._deep_value_threshold

        # 4. 포지션 배수 및 신뢰도
        multiplier = 1.0
        confidence_delta = 0.0
        notes = []

        if composite >= 0.75:
            multiplier *= 1.20
            confidence_delta += 0.10
            notes.append(f"고퀄리티({composite:.2f})")
        elif composite >= 0.60:
            multiplier *= 1.05
            confidence_delta += 0.03
        elif composite < 0.35:
            multiplier *= 0.70
            confidence_delta -= 0.10
            notes.append(f"저퀄리티({composite:.2f})")

        # 모멘텀 보정
        if momentum >= 0.70:
            multiplier *= 1.10
            confidence_delta += 0.05
            notes.append("강한 모멘텀")
        elif momentum < 0.30:
            multiplier *= 0.90
            confidence_delta -= 0.03

        # 캘린더 보정
        multiplier *= calendar

        # 딥밸류 보너스
        if is_deep:
            multiplier *= 1.10
            confidence_delta += 0.08
            notes.append("딥밸류")

        multiplier = float(np.clip(multiplier, self._min_mult, self._max_mult))

        return QuantValueSignal(
            value_score=round(value, 3),
            profitability_score=round(profit, 3),
            growth_score=round(growth, 3),
            safety_score=round(safety, 3),
            dividend_score=round(dividend, 3),
            momentum_score=round(momentum, 3),
            calendar_adjustment=round(calendar, 3),
            composite_score=round(composite, 3),
            is_deep_value=is_deep,
            position_multiplier=round(multiplier, 2),
            confidence_delta=round(confidence_delta, 2),
            note=", ".join(notes) if notes else "",
        )
