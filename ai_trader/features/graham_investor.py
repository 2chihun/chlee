"""벤저민 그레이엄 현명한 투자자 분석 모듈

벤저민 그레이엄 "현명한 투자자(개정4판)" 핵심 개념 기반.

안전마진(Margin of Safety): 내재가치 대비 충분히 낮은 가격에 매수
방어적 투자자 7대 기준: 규모, 재무건전성, 이익안정성, 배당, 성장, PER, PBR
미스터 마켓(Mr. Market): 시장의 과도한 공포/탐욕을 기회로 활용
저PER 전략: 장기적으로 저PER 종목군이 고PER 대비 초과수익

주요 기능:
- 안전마진 점수 (장기평균 대비 괴리율, 52주 저가 비율)
- 방어적 투자자 기준 프록시 (OHLCV 기반)
- 미스터 마켓 감지 (과도한 공포/탐욕 판단)
- 통합 그레이엄 점수
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class GrahamInvestorSignal:
    """그레이엄 가치투자 통합 시그널"""
    margin_of_safety: float = 0.0       # 안전마진 점수 (0~1, 높을수록 안전)
    defensive_score: float = 0.5        # 방어적 투자자 기준 점수 (0~1)
    mr_market_fear: float = 0.0         # 미스터 마켓 공포 수준 (0~1, 높을수록 공포)
    mr_market_greed: float = 0.0        # 미스터 마켓 탐욕 수준 (0~1, 높을수록 탐욕)
    low_per_proxy: float = 0.5          # 저PER 프록시 점수 (0~1, 높을수록 저평가)
    earnings_stability: float = 0.5     # 이익 안정성 프록시 (0~1)
    price_to_avg_ratio: float = 1.0     # 현재가/장기평균 비율
    # 자산배분
    equity_pct: float = 50.0           # 권고 주식 비중 (25~75%)
    cash_pct: float = 50.0            # 권고 현금/채권 비중 (25~75%)
    market_valuation: str = "fair"     # 시장 밸류에이션 (cheap/fair/expensive)
    rebalance_action: str = "hold"     # 리밸런싱 액션
    # 통합
    composite_score: float = 0.5        # 종합 그레이엄 점수 (0~1)
    position_multiplier: float = 1.0    # 포지션 조정 배수
    confidence_delta: float = 0.0       # 신뢰도 조정값
    note: str = ""


class MarginOfSafetyScorer:
    """안전마진 점수 산출기

    그레이엄: "투자의 비밀은 안전마진이라는 세 단어로 요약된다"
    - 현재가가 장기 평균 대비 충분히 낮으면 안전마진 존재
    - 52주 최저가 근처일수록 안전마진 높음
    - 거래량 밀도 대비 낮은 가격 = NCAV 프록시
    """

    def __init__(self, lookback: int = 240):
        self.lookback = lookback

    def score(self, df: pd.DataFrame) -> float:
        """안전마진 점수 (0~1, 높을수록 안전마진 큼)"""
        if len(df) < 60:
            return 0.0

        close = df["close"].astype(float).values
        n = min(len(close), self.lookback)
        recent = close[-n:]
        current = close[-1]

        scores = []

        # 1. 장기 평균 대비 괴리율 (평균 이하 = 안전마진)
        long_avg = np.mean(recent)
        if long_avg > 0:
            discount = 1.0 - current / long_avg
            # -20% 이하일수록 안전마진 높음 (최대 -40%에서 1.0)
            mos = min(max(discount / 0.4, 0), 1.0)
            scores.append(mos * 0.4)
        else:
            scores.append(0.0)

        # 2. 52주(240일) 최저가 대비 위치
        low_52w = np.min(recent[-min(240, n):])
        high_52w = np.max(recent[-min(240, n):])
        if high_52w > low_52w:
            position = (current - low_52w) / (high_52w - low_52w)
            # 저가 근처(position 작을수록) = 안전마진 높음
            low_score = max(1.0 - position, 0)
            scores.append(low_score * 0.35)
        else:
            scores.append(0.0)

        # 3. 거래량 가중 평균가 대비 할인 (VWAP 프록시)
        if "volume" in df.columns:
            vol = df["volume"].astype(float).values[-n:]
            total_vol = np.sum(vol)
            if total_vol > 0:
                vwap = np.sum(recent * vol[-n:]) / total_vol
                if vwap > 0:
                    vwap_discount = 1.0 - current / vwap
                    vwap_score = min(max(vwap_discount / 0.3, 0), 1.0)
                    scores.append(vwap_score * 0.25)
                else:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)

        return min(sum(scores), 1.0)


class DefensiveScreener:
    """방어적 투자자 기준 프록시 검사

    그레이엄 7대 기준 (OHLCV 프록시):
    1. 적정 규모 → 평균 거래대금 (시총 프록시)
    2. 재무건전성 → 가격 안정성 (유동비율 프록시)
    3. 이익 안정성 → 분기별 양수 수익률 비율
    4. 배당 지속성 → 장기 가격 우상향 안정성
    5. 이익 성장 → 장기 수익률 양수
    6. 적정 PER → 장기평균 대비 가격 합리성
    7. PER×PBR < 22.5 → 종합 밸류에이션 프록시
    """

    def __init__(self, lookback: int = 240):
        self.lookback = lookback

    def score(self, df: pd.DataFrame) -> float:
        """방어적 투자자 기준 프록시 점수 (0~1)"""
        if len(df) < 60:
            return 0.5

        close = df["close"].astype(float).values
        n = min(len(close), self.lookback)
        recent = close[-n:]

        criteria_met = 0
        total_criteria = 7

        # 1. 규모: 평균 거래대금 (높을수록 대형주)
        if "volume" in df.columns:
            vol = df["volume"].astype(float).values[-n:]
            avg_turnover = np.mean(recent * vol[-n:])
            if avg_turnover > 0:
                criteria_met += 1  # 거래가 존재하면 통과
        else:
            criteria_met += 0.5

        # 2. 재무건전성: 급락 빈도 낮음 (일 -5% 이상 하락 빈도)
        returns = np.diff(recent) / recent[:-1]
        crash_days = np.sum(returns < -0.05)
        if crash_days <= len(returns) * 0.03:  # 3% 미만
            criteria_met += 1

        # 3. 이익 안정성: 분기(60일) 단위 양수 수익률 비율
        quarters = n // 60
        if quarters >= 2:
            positive_q = 0
            for i in range(quarters):
                start = i * 60
                end = min(start + 60, n)
                q_return = (recent[end - 1] - recent[start]) / recent[start] if recent[start] > 0 else 0
                if q_return >= 0:
                    positive_q += 1
            if positive_q / quarters >= 0.7:
                criteria_met += 1
            else:
                criteria_met += positive_q / quarters * 0.7

        # 4. 배당 지속성: 장기 우상향 (배당주는 안정 우상향 경향)
        if n >= 120:
            half = n // 2
            first_half_avg = np.mean(recent[:half])
            second_half_avg = np.mean(recent[half:])
            if second_half_avg >= first_half_avg:
                criteria_met += 1

        # 5. 이익 성장: 장기 수익률 양수 (10년 EPS 33% 성장 프록시)
        long_return = (recent[-1] - recent[0]) / recent[0] if recent[0] > 0 else 0
        if long_return > 0.1:  # 10% 이상 상승
            criteria_met += 1
        elif long_return > 0:
            criteria_met += 0.5

        # 6. 적정 PER: 장기평균 대비 15% 이내 (PER 15배 프록시)
        long_avg = np.mean(recent)
        if long_avg > 0:
            premium = recent[-1] / long_avg - 1.0
            if premium <= 0.15:
                criteria_met += 1
            elif premium <= 0.3:
                criteria_met += 0.5

        # 7. PER×PBR < 22.5 프록시: 변동성 조정 밸류에이션
        vol = np.std(returns) if len(returns) > 0 else 0.03
        if long_avg > 0:
            premium_ratio = recent[-1] / long_avg
            composite_val = premium_ratio * (1.0 + vol * 10)
            if composite_val < 1.5:  # 22.5 프록시
                criteria_met += 1
            elif composite_val < 2.0:
                criteria_met += 0.5

        return min(criteria_met / total_criteria, 1.0)


class MrMarketDetector:
    """미스터 마켓 감지기

    그레이엄: "미스터 마켓은 매일 찾아와 주식을 사고팔자고 제안하는 조울증 동업자"
    - 과도한 공포: 급락 + 거래량 폭증 → 매수 기회
    - 과도한 탐욕: 급등 + 거래량 폭증 → 매수 자제
    """

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def detect(self, df: pd.DataFrame) -> tuple:
        """(공포 수준, 탐욕 수준) 각 0~1 반환"""
        if len(df) < 20:
            return (0.0, 0.0)

        close = df["close"].astype(float).values
        n = min(len(close), self.lookback)
        recent = close[-n:]
        returns = np.diff(recent) / recent[:-1]

        fear = 0.0
        greed = 0.0

        # 1. 최근 수익률 기반
        recent_5d = np.mean(returns[-5:]) if len(returns) >= 5 else 0
        recent_20d = np.mean(returns[-20:]) if len(returns) >= 20 else 0

        # 급락 = 공포
        if recent_5d < -0.02:
            fear += min(abs(recent_5d) / 0.05, 1.0) * 0.4
        if recent_20d < -0.01:
            fear += min(abs(recent_20d) / 0.03, 1.0) * 0.2

        # 급등 = 탐욕
        if recent_5d > 0.02:
            greed += min(recent_5d / 0.05, 1.0) * 0.4
        if recent_20d > 0.01:
            greed += min(recent_20d / 0.03, 1.0) * 0.2

        # 2. 거래량 이상 감지
        if "volume" in df.columns:
            vol = df["volume"].astype(float).values[-n:]
            if len(vol) >= 20:
                avg_vol = np.mean(vol[-20:])
                recent_vol = np.mean(vol[-5:]) if len(vol) >= 5 else avg_vol
                if avg_vol > 0:
                    vol_surge = recent_vol / avg_vol
                    if vol_surge > 2.0:
                        # 거래량 급증 + 하락 = 공포 강화
                        if recent_5d < 0:
                            fear += min((vol_surge - 1.0) / 3.0, 1.0) * 0.4
                        else:
                            greed += min((vol_surge - 1.0) / 3.0, 1.0) * 0.4

        return (min(fear, 1.0), min(greed, 1.0))


class AssetAllocationGuide:
    """그레이엄 자산배분 가이드

    그레이엄: "주식 비중은 25~75%, 채권은 75~25%. 기본은 50:50"
    - 시장 PER 수준이 역사적 고점 → 주식 비중 축소 (최소 25%)
    - 시장 PER 수준이 역사적 저점 → 주식 비중 확대 (최대 75%)
    - 비중 임계값 ±5% 초과 시 리밸런싱 실행

    OHLCV 기반 프록시:
    - 시장 PER → 현재가/장기평균가 비율 (고평가 = 주식 축소)
    - 변동성 수준 → 과열/공포 보조 지표
    - 52주 위치 → 시장 수준 판단
    """

    # 그레이엄 배분 범위
    MIN_EQUITY_PCT = 25.0   # 주식 최소 비중 (%)
    MAX_EQUITY_PCT = 75.0   # 주식 최대 비중 (%)
    DEFAULT_EQUITY_PCT = 50.0
    REBALANCE_THRESHOLD = 5.0  # ±5% 초과 시 리밸런싱

    def __init__(self, lookback: int = 240):
        self.lookback = lookback

    def recommend(self, df: pd.DataFrame) -> dict:
        """시장 상황 기반 주식 비중 권고

        Args:
            df: OHLCV DataFrame (시장 지수 또는 개별 종목)

        Returns:
            equity_pct: 권고 주식 비중 (25~75%)
            cash_pct: 권고 현금/채권 비중 (25~75%)
            market_valuation: 시장 밸류에이션 수준 (cheap/fair/expensive)
            rebalance_action: 리밸런싱 액션 (increase_equity/decrease_equity/hold)
            confidence: 판단 신뢰도 (0~1)
            note: 상세 설명
        """
        if df is None or len(df) < 60:
            return {
                "equity_pct": self.DEFAULT_EQUITY_PCT,
                "cash_pct": self.DEFAULT_EQUITY_PCT,
                "market_valuation": "unknown",
                "rebalance_action": "hold",
                "confidence": 0.0,
                "note": "데이터 부족 — 기본 50:50 유지",
            }

        close = df["close"].astype(float).values
        n = min(len(close), self.lookback)
        recent = close[-n:]
        current = close[-1]

        # ── 1. 시장 밸류에이션 판단 ──

        # 1a. 현재가/장기평균 비율 (PER 프록시)
        long_avg = np.mean(recent)
        price_ratio = current / long_avg if long_avg > 0 else 1.0

        # 1b. 52주 위치 (0=최저, 1=최고)
        low_52w = np.min(recent[-min(240, n):])
        high_52w = np.max(recent[-min(240, n):])
        if high_52w > low_52w:
            position_52w = (current - low_52w) / (high_52w - low_52w)
        else:
            position_52w = 0.5

        # 1c. 변동성 수준 (연환산)
        returns = np.diff(recent) / recent[:-1]
        ann_vol = np.std(returns) * np.sqrt(252) if len(returns) > 20 else 0.2

        # ── 2. 밸류에이션 종합 점수 (0=극저평가, 1=극고평가) ──
        valuation_score = (
            0.50 * np.clip((price_ratio - 0.7) / 0.6, 0, 1)  # 0.7~1.3 → 0~1
            + 0.30 * position_52w
            + 0.20 * np.clip(ann_vol / 0.4, 0, 1)  # 고변동성 = 과열 신호
        )
        valuation_score = np.clip(valuation_score, 0, 1)

        # ── 3. 밸류에이션 수준 분류 ──
        if valuation_score <= 0.30:
            market_val = "cheap"
        elif valuation_score <= 0.65:
            market_val = "fair"
        else:
            market_val = "expensive"

        # ── 4. 주식 비중 산출 ──
        # 저평가 → 75%, 적정 → 50%, 고평가 → 25%
        equity_pct = self.MAX_EQUITY_PCT - (
            valuation_score * (self.MAX_EQUITY_PCT - self.MIN_EQUITY_PCT)
        )
        equity_pct = np.clip(equity_pct, self.MIN_EQUITY_PCT, self.MAX_EQUITY_PCT)
        cash_pct = 100.0 - equity_pct

        # ── 5. 리밸런싱 판단 ──
        deviation = equity_pct - self.DEFAULT_EQUITY_PCT
        if deviation > self.REBALANCE_THRESHOLD:
            action = "increase_equity"
        elif deviation < -self.REBALANCE_THRESHOLD:
            action = "decrease_equity"
        else:
            action = "hold"

        # ── 6. 신뢰도 (데이터 길이 + 변동성 안정성) ──
        data_conf = min(n / self.lookback, 1.0)
        vol_stability = 1.0 - np.clip(abs(ann_vol - 0.2) / 0.3, 0, 1)
        confidence = 0.7 * data_conf + 0.3 * vol_stability

        # ── 노트 ──
        notes = []
        notes.append(f"가격/평균={price_ratio:.2f}")
        notes.append(f"52주위치={position_52w:.0%}")
        notes.append(f"변동성={ann_vol:.1%}")
        if market_val == "cheap":
            notes.append("저평가 구간 — 주식 비중 확대 권고")
        elif market_val == "expensive":
            notes.append("고평가 구간 — 주식 비중 축소 권고")

        return {
            "equity_pct": round(float(equity_pct), 1),
            "cash_pct": round(float(cash_pct), 1),
            "market_valuation": market_val,
            "rebalance_action": action,
            "valuation_score": round(float(valuation_score), 4),
            "confidence": round(float(confidence), 4),
            "note": "; ".join(notes),
        }


class GrahamInvestorAnalyzer:
    """그레이엄 현명한 투자자 통합 분석기

    구성요소:
    - MarginOfSafetyScorer: 안전마진 점수
    - DefensiveScreener: 방어적 투자자 기준
    - MrMarketDetector: 미스터 마켓 감지
    - 저PER 프록시: 장기평균 대비 저평가

    composite_score 산출:
    - 안전마진 30% + 방어적기준 25% + 미스터마켓(공포-탐욕) 20% + 저PER 25%
    """

    WEIGHTS = {
        "margin_of_safety": 0.30,
        "defensive": 0.25,
        "mr_market": 0.20,
        "low_per": 0.25,
    }

    def __init__(self, lookback: int = 240):
        self.lookback = lookback
        self._mos_scorer = MarginOfSafetyScorer(lookback)
        self._defensive = DefensiveScreener(lookback)
        self._mr_market = MrMarketDetector(min(lookback, 60))
        self._allocation = AssetAllocationGuide(lookback)

    def analyze(self, df: pd.DataFrame) -> GrahamInvestorSignal:
        """OHLCV DataFrame → GrahamInvestorSignal"""
        if df is None or len(df) < 20:
            return GrahamInvestorSignal(note="데이터 부족")

        # 안전마진
        mos = self._mos_scorer.score(df)

        # 방어적 투자자 기준
        defensive = self._defensive.score(df)

        # 미스터 마켓
        fear, greed = self._mr_market.detect(df)

        # 이익 안정성 프록시
        close = df["close"].astype(float).values
        n = min(len(close), self.lookback)
        recent = close[-n:]
        returns = np.diff(recent) / recent[:-1]
        positive_ratio = np.mean(returns > 0) if len(returns) > 0 else 0.5
        earnings_stab = min(max(positive_ratio * 1.2, 0), 1.0)

        # 저PER 프록시 (장기평균 대비 할인율)
        long_avg = np.mean(recent) if len(recent) > 0 else recent[-1]
        price_to_avg = recent[-1] / long_avg if long_avg > 0 else 1.0
        low_per = min(max(1.0 - (price_to_avg - 0.7) / 0.6, 0), 1.0)

        # 자산배분 가이드
        alloc = self._allocation.recommend(df)

        # Mr. Market 점수: 공포 = 매수 기회(+), 탐욕 = 경계(-)
        mr_market_score = min(max(0.5 + fear * 0.5 - greed * 0.5, 0), 1.0)

        # composite
        composite = (
            mos * self.WEIGHTS["margin_of_safety"]
            + defensive * self.WEIGHTS["defensive"]
            + mr_market_score * self.WEIGHTS["mr_market"]
            + low_per * self.WEIGHTS["low_per"]
        )
        composite = min(max(composite, 0), 1.0)

        # 포지션 배수: 높은 안전마진 + 공포 시 확대
        pos_mult = 1.0
        if mos > 0.5 and fear > 0.3:
            pos_mult = min(1.0 + mos * 0.3 + fear * 0.2, 1.5)
        elif greed > 0.5:
            pos_mult = max(1.0 - greed * 0.3, 0.5)

        # 신뢰도 조정
        conf_delta = (composite - 0.5) * 0.2

        # 노트
        notes = []
        if mos > 0.5:
            notes.append(f"안전마진 확보({mos:.0%})")
        if fear > 0.5:
            notes.append(f"미스터마켓 공포({fear:.0%})")
        if greed > 0.5:
            notes.append(f"미스터마켓 탐욕({greed:.0%})")
        if defensive > 0.7:
            notes.append("방어적 기준 충족")
        if low_per > 0.6:
            notes.append("저PER 프록시 양호")

        # 자산배분 노트 추가
        if alloc.get("market_valuation") == "cheap":
            notes.append(f"자산배분: 주식{alloc['equity_pct']:.0f}%↑")
        elif alloc.get("market_valuation") == "expensive":
            notes.append(f"자산배분: 주식{alloc['equity_pct']:.0f}%↓")

        return GrahamInvestorSignal(
            margin_of_safety=round(mos, 4),
            defensive_score=round(defensive, 4),
            mr_market_fear=round(fear, 4),
            mr_market_greed=round(greed, 4),
            low_per_proxy=round(low_per, 4),
            earnings_stability=round(earnings_stab, 4),
            price_to_avg_ratio=round(price_to_avg, 4),
            equity_pct=alloc.get("equity_pct", 50.0),
            cash_pct=alloc.get("cash_pct", 50.0),
            market_valuation=alloc.get("market_valuation", "fair"),
            rebalance_action=alloc.get("rebalance_action", "hold"),
            composite_score=round(composite, 4),
            position_multiplier=round(pos_mult, 4),
            confidence_delta=round(conf_delta, 4),
            note="; ".join(notes) if notes else "",
        )
