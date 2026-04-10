# -*- coding: utf-8 -*-
"""Ernest P. Chan - Quantitative Trading

백테스트 분석 도구 강화 모듈:
1. 켈리 기준선 개선 (Half-Kelly, 분수 켈리)
2. 트랜잭션 비용 모델 (슬리피지·시장충격 정밀화)
3. 전략 용량 추정 (시장충격 기반 최대 자본)
4. 최적 보유기간 추정 (자기상관 기반)
5. 수익률 분포 분석 (왜도·첨도·VaR)
6. 오버피팅 감지 (IS/OOS 성과 비교)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class BacktestAnalyticsSignal:
    """백테스트 분석 신호"""
    # 켈리 기준선 최적 비중 (0~1)
    kelly_fraction: float
    # Half-Kelly (보수적)
    half_kelly: float
    # 트랜잭션 비용 영향 (연 수익률 감소 추정, 예: -0.02 = 연 2% 손실)
    cost_impact: float
    # 전략 용량 추정 (원, 0이면 추정 불가)
    estimated_capacity: float
    # 최적 보유기간 (일)
    optimal_holding_days: int
    # VaR 95% (일일, 음수)
    var_95: float
    # 오버피팅 점수 (0~1, 높을수록 오버피팅 의심)
    overfit_score: float
    # 포지션 배수 (0.3~1.5)
    position_multiplier: float
    # 신뢰도 조정 (-0.3~+0.3)
    confidence_adjustment: float
    # 설명
    note: str = ""


class KellyCriterion:
    """개선된 켈리 기준선

    Kelly = (bp - q) / b
    - b: 승패비율 (avg_win / avg_loss)
    - p: 승률
    - q: 패률 (1-p)

    Half-Kelly를 안전 권장 (리스크 절반, 수익 75%).
    """

    def compute(self, returns: pd.Series) -> Tuple[float, float]:
        """켈리 비중 및 Half-Kelly 계산

        Returns:
            (full_kelly, half_kelly)
        """
        if len(returns) < 20:
            return 0.0, 0.0

        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            return 0.0, 0.0

        p = len(wins) / len(returns)
        q = 1 - p
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())

        if avg_loss == 0:
            return 0.0, 0.0

        b = avg_win / avg_loss
        kelly = (b * p - q) / b

        kelly = float(np.clip(kelly, 0.0, 1.0))
        return kelly, kelly / 2


class TransactionCostModel:
    """정밀 트랜잭션 비용 모델

    총 비용 = 수수료 + 슬리피지 + 시장충격

    - 수수료: 고정 비율 (편도 ~0.25%)
    - 슬리피지: 스프레드의 절반
    - 시장충격: 주문량 / 일평균거래량 × 변동성
    """

    def __init__(
        self,
        commission_rate: float = 0.00015,  # 편도 0.015%
        tax_rate: float = 0.0018,          # 매도세 0.18%
        spread_bps: float = 10.0,          # 평균 스프레드 10bps
    ):
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self.spread_bps = spread_bps

    def total_cost_per_trade(
        self, trade_value: float, avg_daily_volume: float, daily_volatility: float
    ) -> float:
        """1회 거래 총 비용 (비율)"""
        # 수수료 (왕복)
        commission = self.commission_rate * 2

        # 세금 (매도 시)
        tax = self.tax_rate

        # 슬리피지
        slippage = self.spread_bps / 10000 / 2

        # 시장충격 (Kyle's Lambda 프록시)
        if avg_daily_volume > 0 and daily_volatility > 0:
            participation_rate = trade_value / (avg_daily_volume + 1)
            impact = daily_volatility * np.sqrt(participation_rate)
        else:
            impact = 0.001

        return commission + tax + slippage + impact

    def annual_cost_impact(
        self,
        trades_per_year: int,
        avg_trade_value: float,
        avg_daily_volume: float,
        daily_volatility: float,
    ) -> float:
        """연간 총 비용 영향 (비율)"""
        cost_per_trade = self.total_cost_per_trade(
            avg_trade_value, avg_daily_volume, daily_volatility
        )
        return cost_per_trade * trades_per_year


class CapacityEstimator:
    """전략 용량 추정

    최대 투자 가능 자본 = 일 거래량 × 참여율 상한 × 평균 가격
    시장충격이 기대 수익의 50%를 초과하면 용량 초과.
    """

    def __init__(self, max_participation: float = 0.05):
        """
        Args:
            max_participation: 최대 일 거래량 참여율 (기본 5%)
        """
        self.max_participation = max_participation

    def estimate(self, df: pd.DataFrame) -> float:
        """전략 용량 추정 (원)"""
        if len(df) < 20:
            return 0.0

        recent = df.tail(20)
        avg_volume = recent["volume"].mean()
        avg_price = recent["close"].mean()

        if avg_volume <= 0 or avg_price <= 0:
            return 0.0

        capacity = avg_volume * self.max_participation * avg_price
        return float(capacity)


class HoldingPeriodOptimizer:
    """최적 보유기간 추정

    자기상관 분석: 수익률이 양의 자기상관을 보이는 최대 lag가 최적 보유기간.
    """

    def estimate(self, df: pd.DataFrame, max_lag: int = 30) -> int:
        """최적 보유기간 (일)"""
        if len(df) < max_lag * 2:
            return 5

        returns = df["close"].pct_change().dropna()
        if len(returns) < max_lag * 2:
            return 5

        best_lag = 1
        best_autocorr = 0.0

        for lag in range(1, max_lag + 1):
            autocorr = returns.autocorr(lag=lag)
            if autocorr is not None and autocorr > best_autocorr:
                best_autocorr = autocorr
                best_lag = lag

        return best_lag


class ReturnDistributionAnalyzer:
    """수익률 분포 분석

    정규분포 가정의 위험성을 검증.
    왜도(skewness), 첨도(kurtosis), VaR 계산.
    """

    def var_95(self, returns: pd.Series) -> float:
        """95% VaR (일일, 음수)"""
        if len(returns) < 20:
            return -0.02
        return float(np.percentile(returns.dropna(), 5))

    def var_99(self, returns: pd.Series) -> float:
        """99% VaR (일일, 음수)"""
        if len(returns) < 20:
            return -0.05
        return float(np.percentile(returns.dropna(), 1))

    def skewness(self, returns: pd.Series) -> float:
        """왜도 (음수면 좌측 꼬리 = 큰 하락 위험)"""
        if len(returns) < 20:
            return 0.0
        return float(returns.skew())

    def kurtosis(self, returns: pd.Series) -> float:
        """첨도 (3 초과 시 팻테일 = 극단 이벤트 빈번)"""
        if len(returns) < 20:
            return 3.0
        return float(returns.kurtosis() + 3)  # excess → raw

    def analyze(self, returns: pd.Series) -> Dict[str, float]:
        """수익률 분포 종합 분석"""
        return {
            "var_95": self.var_95(returns),
            "var_99": self.var_99(returns),
            "skewness": self.skewness(returns),
            "kurtosis": self.kurtosis(returns),
            "mean": float(returns.mean()) if len(returns) > 0 else 0.0,
            "std": float(returns.std()) if len(returns) > 0 else 0.0,
        }


class OverfitDetector:
    """오버피팅 감지

    In-Sample / Out-of-Sample 분할하여 성과 차이를 평가.
    차이가 클수록 오버피팅 의심.
    """

    def __init__(self, train_ratio: float = 0.7):
        self.train_ratio = train_ratio

    def detect(self, df: pd.DataFrame) -> float:
        """오버피팅 점수 (0~1, 높을수록 오버피팅 의심)

        간단한 프록시: 전반부(IS) vs 후반부(OOS) 성과 비교.
        전반부가 훨씬 좋으면 오버피팅 가능.
        """
        if len(df) < 60:
            return 0.0

        split = int(len(df) * self.train_ratio)
        returns = df["close"].pct_change().dropna()

        is_returns = returns.iloc[:split]
        oos_returns = returns.iloc[split:]

        if len(is_returns) < 20 or len(oos_returns) < 10:
            return 0.0

        is_sharpe = is_returns.mean() / (is_returns.std() + 1e-10) * np.sqrt(252)
        oos_sharpe = oos_returns.mean() / (oos_returns.std() + 1e-10) * np.sqrt(252)

        if is_sharpe <= 0:
            return 0.0

        # OOS Sharpe가 IS Sharpe의 절반 이하면 오버피팅 의심
        ratio = oos_sharpe / (is_sharpe + 1e-10)
        overfit = max(0, 1.0 - ratio)
        return float(np.clip(overfit, 0.0, 1.0))


class BacktestAnalyzer:
    """백테스트 분석 통합기"""

    def __init__(self, lookback: int = 252):
        self.lookback = lookback
        self.kelly = KellyCriterion()
        self.cost_model = TransactionCostModel()
        self.capacity = CapacityEstimator()
        self.holding = HoldingPeriodOptimizer()
        self.distribution = ReturnDistributionAnalyzer()
        self.overfit = OverfitDetector()

    def analyze(self, df: pd.DataFrame, trades_per_year: int = 50) -> BacktestAnalyticsSignal:
        """백테스트 분석 종합"""
        if len(df) < 60:
            return BacktestAnalyticsSignal(
                kelly_fraction=0.0,
                half_kelly=0.0,
                cost_impact=0.0,
                estimated_capacity=0.0,
                optimal_holding_days=5,
                var_95=-0.02,
                overfit_score=0.0,
                position_multiplier=1.0,
                confidence_adjustment=0.0,
                note="데이터 부족 (최소 60봉 필요)",
            )

        try:
            data = df.tail(self.lookback)
            returns = data["close"].pct_change().dropna()

            # 1. 켈리 기준선
            full_kelly, half_kelly = self.kelly.compute(returns)

            # 2. 트랜잭션 비용
            avg_vol = data["volume"].mean()
            avg_price = data["close"].mean()
            avg_trade = avg_price * 100  # 가정: 100주 단위
            daily_vol = returns.std()

            cost_impact = self.cost_model.annual_cost_impact(
                trades_per_year, avg_trade, avg_vol * avg_price, daily_vol
            )

            # 3. 전략 용량
            est_capacity = self.capacity.estimate(data)

            # 4. 최적 보유기간
            opt_holding = self.holding.estimate(data)

            # 5. VaR
            var95 = self.distribution.var_95(returns)

            # 6. 오버피팅 점수
            overfit_score = self.overfit.detect(data)

            # 포지션 배수
            if half_kelly >= 0.3 and overfit_score <= 0.3:
                multiplier = 1.3
            elif half_kelly >= 0.15:
                multiplier = 1.0
            elif half_kelly >= 0.05:
                multiplier = 0.7
            else:
                multiplier = 0.5

            if overfit_score >= 0.6:
                multiplier *= 0.7

            multiplier = float(np.clip(multiplier, 0.3, 1.5))

            # 신뢰도 조정
            conf_adj = half_kelly * 0.3 - overfit_score * 0.2 - cost_impact * 2
            conf_adj = float(np.clip(conf_adj, -0.3, 0.3))

            # 설명
            notes = []
            if half_kelly >= 0.2:
                notes.append(f"켈리 우수(HK={half_kelly:.2f})")
            if overfit_score >= 0.5:
                notes.append(f"오버피팅 주의({overfit_score:.2f})")
            if cost_impact > 0.05:
                notes.append(f"거래비용 과다(연{cost_impact:.1%})")
            if var95 < -0.03:
                notes.append(f"VaR95 주의({var95:.1%})")

            return BacktestAnalyticsSignal(
                kelly_fraction=round(full_kelly, 4),
                half_kelly=round(half_kelly, 4),
                cost_impact=round(cost_impact, 4),
                estimated_capacity=round(est_capacity, 0),
                optimal_holding_days=opt_holding,
                var_95=round(var95, 4),
                overfit_score=round(overfit_score, 4),
                position_multiplier=round(multiplier, 2),
                confidence_adjustment=round(conf_adj, 4),
                note=" | ".join(notes) if notes else "정상 범위",
            )

        except Exception as e:
            logger.warning(f"백테스트 분석 오류: {e}")
            return BacktestAnalyticsSignal(
                kelly_fraction=0.0,
                half_kelly=0.0,
                cost_impact=0.0,
                estimated_capacity=0.0,
                optimal_holding_days=5,
                var_95=-0.02,
                overfit_score=0.0,
                position_multiplier=1.0,
                confidence_adjustment=0.0,
                note=f"분석 오류: {e}",
            )
