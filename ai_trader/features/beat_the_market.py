"""에드워드 소프 시장 공략 분석 모듈

에드워드 O. 소프 "나는 어떻게 시장을 이겼나" 핵심 개념 기반.

켈리 기준(Kelly Criterion): 최적 배팅 비율 = edge / odds
이익수익률 비교: PER 역수 vs 채권금리 → 주식 매력도
통계적 차익: 가격 괴리 감지, 평균회귀 기회
복리 성장: 기하평균 극대화, 드로다운 관리

주요 기능:
- 켈리 기준 포지션 사이징
- 이익수익률 기반 매력도 점수
- 통계적 차익 기회 감지
- 복리 성장 품질 평가
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BeatTheMarketSignal:
    """에드워드 소프 시장 공략 통합 시그널"""
    kelly_fraction: float = 0.0         # 켈리 최적 배팅 비율 (0~1)
    earnings_yield_score: float = 0.5   # 이익수익률 매력도 (0~1)
    stat_arb_opportunity: float = 0.0   # 통계적 차익 기회 (0~1)
    compound_quality: float = 0.5       # 복리 성장 품질 (0~1)
    geometric_return: float = 0.0       # 기하평균 일간 수익률
    max_drawdown: float = 0.0           # 최대 드로다운 (음수)
    composite_score: float = 0.5        # 종합 점수 (0~1)
    position_multiplier: float = 1.0    # 포지션 조정 배수
    confidence_delta: float = 0.0       # 신뢰도 조정값
    note: str = ""


class KellyCriterionSizer:
    """켈리 기준 포지션 사이징

    소프: "최적 배팅 비율 = edge / odds"
    - 승률과 평균 이익/손실 비율에서 최적 포지션 크기 산출
    - Half-Kelly (절반 켈리) 사용 권장: 변동성 줄이면서 수익 유지
    """

    def __init__(self, lookback: int = 120):
        self.lookback = lookback

    def calculate(self, df: pd.DataFrame) -> float:
        """켈리 최적 배팅 비율 (0~1, Half-Kelly 적용)"""
        if len(df) < 30:
            return 0.0

        close = df["close"].astype(float).values
        n = min(len(close), self.lookback)
        returns = np.diff(close[-n:]) / close[-n:-1]

        if len(returns) == 0:
            return 0.0

        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            return 0.0

        win_rate = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        if avg_loss == 0:
            return 0.0

        # Kelly: f = (p * b - q) / b where p=win_rate, q=1-p, b=avg_win/avg_loss
        b = avg_win / avg_loss
        kelly = (win_rate * b - (1 - win_rate)) / b

        # Half-Kelly for safety
        half_kelly = kelly / 2.0

        return min(max(half_kelly, 0), 1.0)


class EarningsYieldComparator:
    """이익수익률 비교기

    소프: "PER의 역수(이익수익률)를 채권금리와 비교"
    - OHLCV에서 PER 직접 계산 불가 → 가격/장기평균 비율의 역수로 프록시
    - 이익수익률이 높을수록 (PER이 낮을수록) 주식 매력적
    """

    def __init__(self, lookback: int = 240):
        self.lookback = lookback

    def score(self, df: pd.DataFrame) -> float:
        """이익수익률 매력도 점수 (0~1, 높을수록 매력적)"""
        if len(df) < 60:
            return 0.5

        close = df["close"].astype(float).values
        n = min(len(close), self.lookback)
        recent = close[-n:]
        current = close[-1]

        # PER 프록시: 현재가 / 장기평균 (높으면 고PER)
        long_avg = np.mean(recent)
        if long_avg <= 0:
            return 0.5

        per_proxy = current / long_avg

        # 이익수익률 프록시 = 1 / PER 프록시
        ey_proxy = 1.0 / per_proxy if per_proxy > 0 else 0.5

        # 정규화: ey_proxy 0.7~1.5 → score 0~1
        score = min(max((ey_proxy - 0.7) / 0.8, 0), 1.0)

        return score


class StatArbDetector:
    """통계적 차익 기회 감지기

    소프: "가격 불일치 포착 → 차익거래"
    - 볼린저 밴드 이탈: 과매도 = 차익 기회
    - 이동평균 괴리율: 극단적 괴리 = 평균회귀 기대
    """

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def detect(self, df: pd.DataFrame) -> float:
        """통계적 차익 기회 점수 (0~1, 높을수록 기회 큼)"""
        if len(df) < 20:
            return 0.0

        close = df["close"].astype(float).values
        n = min(len(close), self.lookback)
        recent = close[-n:]
        current = close[-1]

        scores = []

        # 1. 볼린저 밴드 이탈 (하단 이탈 = 매수 기회)
        mean = np.mean(recent[-20:])
        std = np.std(recent[-20:])
        if std > 0:
            z_score = (current - mean) / std
            # z < -2 → 강한 매수 기회
            if z_score < -1.0:
                bb_score = min(abs(z_score) / 3.0, 1.0)
                scores.append(bb_score * 0.5)
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)

        # 2. 이동평균 괴리율
        if n >= 60:
            ma60 = np.mean(recent[-60:])
            if ma60 > 0:
                deviation = (current - ma60) / ma60
                if deviation < -0.05:
                    dev_score = min(abs(deviation) / 0.2, 1.0)
                    scores.append(dev_score * 0.5)
                else:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)

        return min(sum(scores), 1.0)


class CompoundGrowthTracker:
    """복리 성장 품질 추적기

    소프: "기하평균 수익률이 장기 부의 핵심"
    - 기하평균 일간 수익률 산출
    - 최대 드로다운 측정
    - 리스크 대비 수익 평가
    """

    def __init__(self, lookback: int = 240):
        self.lookback = lookback

    def evaluate(self, df: pd.DataFrame) -> tuple:
        """(복리품질점수, 기하평균수익률, 최대드로다운) 반환"""
        if len(df) < 30:
            return (0.5, 0.0, 0.0)

        close = df["close"].astype(float).values
        n = min(len(close), self.lookback)
        recent = close[-n:]

        # 기하평균 일간 수익률
        total_return = recent[-1] / recent[0] if recent[0] > 0 else 1.0
        geo_return = total_return ** (1.0 / n) - 1.0

        # 최대 드로다운
        peak = recent[0]
        max_dd = 0.0
        for price in recent:
            if price > peak:
                peak = price
            dd = (price - peak) / peak if peak > 0 else 0
            if dd < max_dd:
                max_dd = dd

        # 복리 품질: 양수 기하평균 + 낮은 드로다운 = 높은 품질
        geo_score = min(max(geo_return * 252 / 0.3 + 0.5, 0), 1.0)  # 연30% 기준
        dd_score = max(1.0 + max_dd / 0.3, 0)  # -30% DD까지 선형

        quality = geo_score * 0.6 + dd_score * 0.4

        return (min(max(quality, 0), 1.0), geo_return, max_dd)


class BeatTheMarketAnalyzer:
    """에드워드 소프 시장 공략 통합 분석기

    구성요소:
    - KellyCriterionSizer: 켈리 기준 포지션
    - EarningsYieldComparator: 이익수익률 매력도
    - StatArbDetector: 통계적 차익 기회
    - CompoundGrowthTracker: 복리 성장 품질

    composite_score:
    - 켈리 20% + 이익수익률 30% + 통계차익 25% + 복리품질 25%
    """

    WEIGHTS = {
        "kelly": 0.20,
        "earnings_yield": 0.30,
        "stat_arb": 0.25,
        "compound": 0.25,
    }

    def __init__(self, lookback: int = 240):
        self.lookback = lookback
        self._kelly = KellyCriterionSizer(min(lookback, 120))
        self._ey = EarningsYieldComparator(lookback)
        self._stat_arb = StatArbDetector(min(lookback, 60))
        self._compound = CompoundGrowthTracker(lookback)

    def analyze(self, df: pd.DataFrame) -> BeatTheMarketSignal:
        """OHLCV DataFrame → BeatTheMarketSignal"""
        if df is None or len(df) < 20:
            return BeatTheMarketSignal(note="데이터 부족")

        kelly = self._kelly.calculate(df)
        ey_score = self._ey.score(df)
        stat_arb = self._stat_arb.detect(df)
        compound_quality, geo_return, max_dd = self._compound.evaluate(df)

        # composite
        composite = (
            kelly * self.WEIGHTS["kelly"]
            + ey_score * self.WEIGHTS["earnings_yield"]
            + stat_arb * self.WEIGHTS["stat_arb"]
            + compound_quality * self.WEIGHTS["compound"]
        )
        composite = min(max(composite, 0), 1.0)

        # 포지션 배수: 켈리 기반 + 차익 기회
        pos_mult = 1.0
        if kelly > 0.3 and stat_arb > 0.3:
            pos_mult = min(1.0 + kelly * 0.3, 1.5)
        elif max_dd < -0.2:
            pos_mult = max(1.0 + max_dd, 0.5)

        conf_delta = (composite - 0.5) * 0.2

        notes = []
        if kelly > 0.3:
            notes.append(f"켈리({kelly:.0%})")
        if ey_score > 0.6:
            notes.append("이익수익률 매력적")
        if stat_arb > 0.3:
            notes.append(f"차익기회({stat_arb:.0%})")
        if compound_quality > 0.7:
            notes.append("복리성장 양호")
        if max_dd < -0.2:
            notes.append(f"드로다운 경고({max_dd:.0%})")

        return BeatTheMarketSignal(
            kelly_fraction=round(kelly, 4),
            earnings_yield_score=round(ey_score, 4),
            stat_arb_opportunity=round(stat_arb, 4),
            compound_quality=round(compound_quality, 4),
            geometric_return=round(geo_return, 6),
            max_drawdown=round(max_dd, 4),
            composite_score=round(composite, 4),
            position_multiplier=round(pos_mult, 4),
            confidence_delta=round(conf_delta, 4),
            note="; ".join(notes) if notes else "",
        )
