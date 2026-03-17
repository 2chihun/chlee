"""주식 품질 평가 모듈

이남우 「좋은 주식 나쁜 주식」 핵심 개념 기반.
- ROE 듀폰 분석 프록시 (수익률 × 안정성 × 레버리지)
- 패닉 매수 기회 감지 (52주 고점 대비 30%+ 급락)
- 자본집약도 분석 (ATR 변동성 기반)
- 산업 효과 프록시 (장기 EMA 기울기)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class StockQualitySignal:
    """주식 품질 통합 시그널"""
    # ROE 듀폰 프록시 (0~1, 높을수록 고품질)
    roe_quality: float = 0.5
    # 패닉 매수 기회 여부
    is_panic_buy: bool = False
    # 패닉 심도 (52주 고점 대비 하락률, 0~1)
    panic_depth: float = 0.0
    # 자본집약도 프록시 (0~1, 높을수록 자본집약적 = 위험)
    capital_intensity: float = 0.5
    # 산업 효과 프록시 (0~1, 높을수록 성장 산업)
    industry_momentum: float = 0.5
    # 종합 품질 점수 (0~1)
    quality_score: float = 0.5
    # 포지션 조정 배수
    position_multiplier: float = 1.0
    # 신뢰도 보정치
    confidence_delta: float = 0.0
    # 분석 메모
    note: str = ""


class ROEQualityScorer:
    """ROE 듀폰 모델 프록시 분석

    이남우: ROE = 순이익률 × 자산회전율 × 레버리지
    기술적 데이터로 프록시 구현:
    - 수익률 프록시: 장기 수익률 추세 (EMA60 기울기)
    - 안정성 프록시: 수익률 변동성의 역수 (안정할수록 좋음)
    - 레버리지 프록시: 급등/급락 빈도 (레버리지 높을수록 변동 심함)
    """

    def __init__(self, lookback: int = 120):
        self.lookback = lookback

    def score(self, df: pd.DataFrame) -> float:
        """ROE 품질 프록시 점수 산출 (0~1)"""
        if len(df) < self.lookback:
            return 0.5

        close = df["close"].astype(float).tail(self.lookback)
        returns = close.pct_change().dropna()

        if len(returns) < 20:
            return 0.5

        # 수익률 추세 (EMA60 기울기 정규화)
        ema60 = close.ewm(span=60, min_periods=30).mean()
        if len(ema60) >= 20:
            slope = (ema60.iloc[-1] - ema60.iloc[-20]) / ema60.iloc[-20]
            profitability = float(np.clip(slope * 5 + 0.5, 0, 1))
        else:
            profitability = 0.5

        # 안정성 (변동성의 역수, 낮을수록 안정적)
        volatility = float(returns.std())
        if volatility > 0:
            # 일별 변동성 2% 기준 정규화
            stability = float(np.clip(1.0 - volatility / 0.04, 0, 1))
        else:
            stability = 0.5

        # 레버리지 프록시 (급등/급락 빈도)
        extreme_count = int((returns.abs() > 0.05).sum())
        extreme_ratio = extreme_count / len(returns)
        leverage_risk = float(np.clip(1.0 - extreme_ratio * 10, 0, 1))

        # 듀폰 가중 합산: 수익률(40%) + 안정성(35%) + 레버리지(25%)
        score = profitability * 0.40 + stability * 0.35 + leverage_risk * 0.25
        return float(np.clip(score, 0, 1))


class PanicBuyDetector:
    """패닉 매수 기회 감지

    이남우: "우량주가 1~2년에 한 번 30% 이상 급락하면 매수 기회"
    - 52주 고점 대비 30%+ 하락 감지
    - 품질 점수가 높은 종목에서만 패닉 매수 유효
    """

    def __init__(
        self,
        panic_threshold: float = 0.30,
        lookback_days: int = 252,
        min_quality: float = 0.5,
    ):
        self.panic_threshold = panic_threshold
        self.lookback_days = lookback_days
        self.min_quality = min_quality

    def detect(
        self, df: pd.DataFrame, quality_score: float = 0.5
    ) -> tuple[bool, float]:
        """패닉 매수 기회 감지

        Returns:
            (is_panic_buy, panic_depth)
            panic_depth: 0~1 (52주 고점 대비 하락 비율)
        """
        if len(df) < 60:
            return False, 0.0

        close = df["close"].astype(float)
        lookback = min(self.lookback_days, len(close))
        high_52w = float(close.tail(lookback).max())

        if high_52w <= 0:
            return False, 0.0

        current = float(close.iloc[-1])
        decline = (high_52w - current) / high_52w

        if decline >= self.panic_threshold and quality_score >= self.min_quality:
            return True, float(np.clip(decline, 0, 1))

        return False, float(np.clip(decline, 0, 1))


class CapitalIntensityAnalyzer:
    """자본집약도 분석

    이남우: "자본집약적 기업은 이익 변동성이 높아 PER 할인"
    - ATR 변동성으로 자본집약도 프록시
    - 고변동 종목: 자본집약적 → 포지션 축소
    - 저변동 종목: 자산경량 → 포지션 유지/확대
    """

    def __init__(self, lookback: int = 60, high_atr_pct: float = 0.04):
        self.lookback = lookback
        self.high_atr_pct = high_atr_pct

    def analyze(self, df: pd.DataFrame) -> float:
        """자본집약도 프록시 (0~1, 높을수록 자본집약적 = 위험)"""
        if len(df) < self.lookback:
            return 0.5

        tail = df.tail(self.lookback)
        high = tail["high"].astype(float)
        low = tail["low"].astype(float)
        close = tail["close"].astype(float)

        # ATR 계산
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.mean())

        avg_price = float(close.mean())
        if avg_price <= 0:
            return 0.5

        # ATR을 가격 대비 비율로 정규화
        atr_pct = atr / avg_price
        # high_atr_pct(4%) 기준으로 0~1 정규화
        intensity = float(np.clip(atr_pct / self.high_atr_pct, 0, 1))

        return intensity


class IndustryMomentumScorer:
    """산업 효과 프록시

    이남우: "고성장 산업의 평범한 기업 > 사양 산업의 좋은 기업"
    - 장기 EMA 기울기로 산업(종목) 성장 모멘텀 측정
    - 상승 추세 종목 = 성장 산업 소속 가능성
    """

    def __init__(self, lookback: int = 120):
        self.lookback = lookback

    def score(self, df: pd.DataFrame) -> float:
        """산업 모멘텀 점수 (0~1)"""
        if len(df) < self.lookback:
            return 0.5

        close = df["close"].astype(float).tail(self.lookback)
        ema120 = close.ewm(span=self.lookback, min_periods=60).mean()

        if len(ema120) < 60:
            return 0.5

        # 장기 EMA 기울기 (60일 전 대비)
        slope = (ema120.iloc[-1] - ema120.iloc[-60]) / ema120.iloc[-60]
        # 20% 상승 = 1.0, -20% 하락 = 0.0
        score = float(np.clip(slope / 0.4 + 0.5, 0, 1))

        return score


class StockQualityAnalyzer:
    """주식 품질 통합 분석기

    이남우의 "좋은 주식 vs 나쁜 주식" 프레임워크:
    1. ROE 듀폰 프록시 → 기업 품질 평가
    2. 패닉 매수 감지 → 급락 시 매수 기회
    3. 자본집약도 분석 → 고변동 종목 포지션 축소
    4. 산업 효과 → 성장 모멘텀 반영
    """

    def __init__(
        self,
        roe_lookback: int = 120,
        panic_threshold: float = 0.30,
        panic_lookback: int = 252,
        intensity_lookback: int = 60,
        high_atr_pct: float = 0.04,
        min_quality_for_panic: float = 0.5,
        max_position_multiplier: float = 1.4,
        min_position_multiplier: float = 0.6,
    ):
        self._roe_scorer = ROEQualityScorer(lookback=roe_lookback)
        self._panic_detector = PanicBuyDetector(
            panic_threshold=panic_threshold,
            lookback_days=panic_lookback,
            min_quality=min_quality_for_panic,
        )
        self._intensity_analyzer = CapitalIntensityAnalyzer(
            lookback=intensity_lookback,
            high_atr_pct=high_atr_pct,
        )
        self._momentum_scorer = IndustryMomentumScorer(lookback=roe_lookback)
        self._max_mult = max_position_multiplier
        self._min_mult = min_position_multiplier

    def analyze(self, df: pd.DataFrame) -> StockQualitySignal:
        """주식 품질 통합 분석"""
        if len(df) < 30:
            return StockQualitySignal(note="데이터 부족")

        # 1. ROE 듀폰 프록시
        roe_quality = self._roe_scorer.score(df)

        # 2. 산업 모멘텀
        industry_momentum = self._momentum_scorer.score(df)

        # 3. 자본집약도
        capital_intensity = self._intensity_analyzer.analyze(df)

        # 4. 종합 품질 점수
        # ROE(40%) + 산업모멘텀(30%) + (1-자본집약도)(30%)
        quality_score = (
            roe_quality * 0.40
            + industry_momentum * 0.30
            + (1.0 - capital_intensity) * 0.30
        )
        quality_score = float(np.clip(quality_score, 0, 1))

        # 5. 패닉 매수 감지
        is_panic, panic_depth = self._panic_detector.detect(df, quality_score)

        # 6. 포지션 배수 계산
        multiplier = 1.0
        confidence_delta = 0.0
        notes = []

        # 고품질 종목: 배수 증가
        if quality_score >= 0.7:
            multiplier *= 1.15
            confidence_delta += 0.05
            notes.append(f"고품질({quality_score:.2f})")
        elif quality_score < 0.35:
            multiplier *= 0.7
            confidence_delta -= 0.1
            notes.append(f"저품질({quality_score:.2f})")

        # 패닉 매수 기회
        if is_panic:
            multiplier *= 1.2
            confidence_delta += 0.15
            notes.append(f"패닉매수(하락{panic_depth:.0%})")

        # 자본집약적 종목: 축소
        if capital_intensity > 0.7:
            multiplier *= 0.85
            confidence_delta -= 0.05
            notes.append(f"자본집약({capital_intensity:.2f})")

        multiplier = float(np.clip(multiplier, self._min_mult, self._max_mult))

        return StockQualitySignal(
            roe_quality=roe_quality,
            is_panic_buy=is_panic,
            panic_depth=panic_depth,
            capital_intensity=capital_intensity,
            industry_momentum=industry_momentum,
            quality_score=quality_score,
            position_multiplier=round(multiplier, 2),
            confidence_delta=round(confidence_delta, 2),
            note=", ".join(notes) if notes else "",
        )
