"""켄 피셔 "주식시장은 어떻게 반복되는가" 기반 시장 기억 분석 모듈

핵심 이론:
  - 시장은 기억하지만 투자자는 잊는다
  - "이번에는 다르다"는 생각은 언제나 틀렸다
  - 극단적 비관론 = 매수 기회 (불신의 비관론, Wall of Worry)
  - 변동성은 정상이다 - 높은 변동성 = 공포 = 기회
  - 실업률은 후행지표 - 고점 전 매수가 최적
  - 약세장 후 강한 반등은 반복되는 패턴

적용 모듈:
  - MarketMemoryAnalyzer: 시장 기억 상실 패턴 (공포/비관론 극단) 감지
  - WallOfWorryDetector: 불신의 비관론 감지 - 강세장 지속 신호
  - VolatilityNormalizer: 변동성 정상화 - 공포 = 기회 확인
  - BearMarketRecoveryDetector: 약세장→강세장 전환 조기 감지
  - FisherSignal: 통합 켄 피셔 시그널
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class FisherSignal:
    """켄 피셔 통합 시그널"""
    # 불신의 비관론 강도 (0~1, 높을수록 비관론 극단 → 매수 기회)
    wall_of_worry_score: float = 0.0
    # 변동성 정상화 점수 (0~1, 높을수록 공포/변동성 과잉)
    volatility_fear_score: float = 0.0
    # 약세장 회복 조기 신호 강도 (0~1)
    recovery_signal: float = 0.0
    # 켄 피셔 종합 신호 (-1~+1: -1=위험, 0=중립, +1=기회)
    fisher_composite: float = 0.0
    # 포지션 조정 배수 (0.5~1.5)
    position_multiplier: float = 1.0
    # 신뢰도 조정값 (+/-)
    confidence_delta: float = 0.0
    # 분석 노트
    note: str = ""


class VolatilityNormalizer:
    """변동성 정상화 분석기

    켄 피셔 핵심 원칙:
    - 변동성은 항상 존재하며 이것이 정상이다
    - 투자자들이 변동성을 비정상이라 착각할 때 = 공포 극단
    - 공포 극단 = 매수 기회 (단, 단기 추세와 결합 필요)
    - 안정적 성과가 지속되는 구간이 오히려 위험 (버블 전조)
    """

    def __init__(self, lookback: int = 60, atr_multiple_threshold: float = 2.0):
        self.lookback = lookback
        self.atr_multiple_threshold = atr_multiple_threshold

    def analyze(self, df: pd.DataFrame) -> dict:
        """변동성 공포 점수를 계산합니다.

        Returns:
            dict: {
                fear_score: 공포 점수 (0~1, 높을수록 공포 극단),
                volatility_regime: 'FEAR'/'NORMAL'/'CALM',
                is_opportunity: 공포 극단 매수 기회 여부,
                note: 분석 노트
            }
        """
        if len(df) < self.lookback:
            return {
                "fear_score": 0.0,
                "volatility_regime": "NORMAL",
                "is_opportunity": False,
                "note": "데이터 부족"
            }

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # ATR 계산 (변동성 측정)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        # 현재 ATR vs 장기 평균 ATR 비교
        current_atr = atr.iloc[-1]
        avg_atr = atr.rolling(self.lookback).mean().iloc[-1]

        if avg_atr <= 0:
            return {
                "fear_score": 0.0,
                "volatility_regime": "NORMAL",
                "is_opportunity": False,
                "note": "ATR 계산 불가"
            }

        atr_ratio = current_atr / avg_atr

        # 일별 수익률 변동성
        returns = close.pct_change().dropna()
        current_vol = returns.tail(10).std()
        avg_vol = returns.rolling(self.lookback).std().iloc[-1]

        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        # 공포 점수 = ATR 비율 + 수익률 변동성 비율 가중 평균
        fear_raw = (atr_ratio * 0.5 + vol_ratio * 0.5 - 1.0)
        fear_score = float(min(max(fear_raw / self.atr_multiple_threshold, 0.0), 1.0))

        # 변동성 레짐 판단
        if atr_ratio >= self.atr_multiple_threshold:
            regime = "FEAR"
            is_opportunity = True
            note = (
                f"변동성 공포 극단: ATR비율={atr_ratio:.2f}x "
                f"(켄피셔: 변동성=정상, 공포=기회)"
            )
        elif atr_ratio <= 0.6:
            regime = "CALM"
            is_opportunity = False
            note = (
                f"변동성 과도 안정: ATR비율={atr_ratio:.2f}x "
                f"(켄피셔: 안정적 수익 지속 구간이 위험)"
            )
        else:
            regime = "NORMAL"
            is_opportunity = False
            note = f"변동성 정상: ATR비율={atr_ratio:.2f}x"

        return {
            "fear_score": fear_score,
            "volatility_regime": regime,
            "is_opportunity": is_opportunity,
            "note": note
        }


class WallOfWorryDetector:
    """불신의 비관론(Wall of Worry) 감지기

    켄 피셔 원칙:
    - 강세장은 "불신의 비관론"을 타고 오른다
    - 호재가 나쁜 방향으로 해석되는 시기 = 강세장 지속 신호
    - 극단적 비관론(뉴 노멀, 더블딥 공포) 발생 직후 급등이 반복됨
    - 2009년 3월 바닥 후 3개월 44.1%, 12개월 74.3% 상승

    구현: 가격 데이터 기반 프록시 지표 사용
    - 가격 회복 시 거래량 부진 = "믿지 않는 랠리" = Wall of Worry
    - RSI 낮은데 가격 반등 = 기술적 불신 랠리
    - BB 하단 이탈 후 복귀 = 극단 공포 후 회복
    """

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def detect(self, df: pd.DataFrame) -> dict:
        """불신의 비관론 패턴을 감지합니다.

        Returns:
            dict: {
                wow_score: 불신 비관론 점수 (0~1),
                pattern: 'DISTRUST_RALLY'/'FEAR_EXTREME'/'NORMAL',
                confidence_boost: 신뢰도 부스트 (+값),
                note: 노트
            }
        """
        if len(df) < self.lookback + 10:
            return {
                "wow_score": 0.0,
                "pattern": "NORMAL",
                "confidence_boost": 0.0,
                "note": "데이터 부족"
            }

        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        # RSI 계산
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        # 볼린저 밴드
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_lower = bb_mid - 2 * bb_std
        bb_pctb = (close - bb_lower) / (4 * bb_std)  # 0=하단, 1=상단

        recent_bb = bb_pctb.iloc[-5:]
        bb_recovery = (recent_bb.iloc[-1] > 0.3) and (recent_bb.min() < 0.1)

        # 가격 회복 + 거래량 부진 = 불신 랠리
        price_change_5d = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]
        vol_change_5d = volume.tail(5).mean() / volume.tail(20).mean()
        distrust_rally = price_change_5d > 0.02 and vol_change_5d < 0.8

        # 극단 공포 후 반등 (RSI < 30에서 회복)
        rsi_extreme_recovery = (
            rsi.iloc[-10:-1].min() < 30 and current_rsi > 35
        )

        # 패턴 결정
        wow_score = 0.0
        pattern = "NORMAL"
        confidence_boost = 0.0
        notes = []

        if distrust_rally:
            wow_score += 0.4
            pattern = "DISTRUST_RALLY"
            confidence_boost += 0.1
            notes.append(f"불신 랠리(가격↑{price_change_5d:.1%}, 거래량{vol_change_5d:.1f}x)")

        if rsi_extreme_recovery:
            wow_score += 0.4
            pattern = "FEAR_EXTREME"
            confidence_boost += 0.15
            notes.append(f"공포 극단 회복(RSI저점→{current_rsi:.0f})")

        if bb_recovery:
            wow_score += 0.2
            confidence_boost += 0.05
            notes.append("BB하단 공포 후 회복")

        wow_score = min(wow_score, 1.0)

        note = " | ".join(notes) if notes else "불신 비관론 패턴 없음"
        if wow_score > 0:
            note += " → 켄피셔: 불신 랠리=강세장 지속 신호"

        return {
            "wow_score": wow_score,
            "pattern": pattern,
            "confidence_boost": confidence_boost,
            "note": note
        }


class BearMarketRecoveryDetector:
    """약세장→강세장 전환 조기 감지기

    켄 피셔 핵심 데이터:
    - 경기침체 종료 전 강세장 시작, 평균 27.5% 선반영
    - 약세장 바닥 후 3개월 44% 급등 패턴 반복
    - 실업률 고점 6개월 전 매수: 31.2% vs 고점 시 14.8%
    - "이번에는 달라" 비관론이 절정일 때 바닥 형성

    구현 전략:
    - 52주 저점 대비 반등률로 회복 강도 측정
    - 강한 반등 + 높은 거래량 = 강세장 초기 신호
    - 과도한 하락 후 기술적 반등 = 역발상 기회
    """

    def __init__(
        self,
        decline_threshold: float = 0.20,  # 약세장 기준 하락률 (20%)
        recovery_threshold: float = 0.10,  # 회복 신호 반등률 (10%)
        lookback_days: int = 252,
    ):
        self.decline_threshold = decline_threshold
        self.recovery_threshold = recovery_threshold
        self.lookback_days = lookback_days

    def detect(self, df: pd.DataFrame) -> dict:
        """약세장 회복 신호를 감지합니다.

        Returns:
            dict: {
                recovery_score: 회복 신호 강도 (0~1),
                bear_depth: 약세장 낙폭 (-값),
                recovery_pct: 바닥 대비 반등률,
                is_early_bull: 강세장 초기 여부,
                confidence_boost: 신뢰도 부스트,
                note: 노트
            }
        """
        lookback = min(self.lookback_days, len(df))
        if lookback < 60:
            return {
                "recovery_score": 0.0,
                "bear_depth": 0.0,
                "recovery_pct": 0.0,
                "is_early_bull": False,
                "confidence_boost": 0.0,
                "note": "데이터 부족"
            }

        close = df["close"].astype(float)
        recent = close.tail(lookback)
        volume = df["volume"].astype(float)

        # 기간 내 최고가/최저가
        peak = recent.max()
        trough = recent.min()
        current = close.iloc[-1]

        # 최고가 대비 낙폭
        bear_depth = (trough - peak) / peak  # 음수

        # 최저가 대비 반등률
        recovery_pct = (current - trough) / trough if trough > 0 else 0.0

        # 약세장 기준 충족 여부
        is_bear = bear_depth <= -self.decline_threshold

        # 회복 기준 충족 여부
        is_recovering = recovery_pct >= self.recovery_threshold

        # 거래량 확인 (회복 시 거래량 증가 = 신뢰도 상승)
        vol_ratio = volume.tail(10).mean() / volume.tail(60).mean()

        # 강세장 초기 신호: 약세장 + 회복 중 + 거래량 증가
        is_early_bull = is_bear and is_recovering and vol_ratio > 1.2

        # 회복 점수 계산
        recovery_score = 0.0
        confidence_boost = 0.0
        notes = []

        if is_bear:
            depth_score = min(abs(bear_depth) / 0.40, 1.0)  # 40% 하락 시 최대
            recovery_score += depth_score * 0.4
            notes.append(f"약세장낙폭{bear_depth:.1%}")

        if is_recovering:
            rec_score = min(recovery_pct / 0.30, 1.0)  # 30% 반등 시 최대
            recovery_score += rec_score * 0.4
            confidence_boost += 0.1
            notes.append(f"반등{recovery_pct:.1%}")

        if vol_ratio > 1.2:
            recovery_score += 0.2
            confidence_boost += 0.05
            notes.append(f"거래량확인({vol_ratio:.1f}x)")

        if is_early_bull:
            confidence_boost += 0.1
            notes.append("강세장초기신호")

        recovery_score = min(recovery_score, 1.0)

        note = " | ".join(notes) if notes else "회복 신호 없음"
        if is_early_bull:
            note += " → 켄피셔: 약세장후강반등패턴(역사적평균27.5%선반영)"

        return {
            "recovery_score": recovery_score,
            "bear_depth": bear_depth,
            "recovery_pct": recovery_pct,
            "is_early_bull": is_early_bull,
            "confidence_boost": confidence_boost,
            "note": note
        }


class MarketMemoryAnalyzer:
    """켄 피셔 시장 기억 분석기 (통합)

    "주식시장은 어떻게 반복되는가" 핵심 원칙 통합 적용:
    1. 극단적 비관론 = 매수 기회 (불신의 비관론)
    2. 변동성 정상화 = 공포가 공포가 아님
    3. 약세장 후 강한 반등 = 반복 패턴
    4. "이번에는 다르다" 착각 = 항상 틀림
    5. 강세장은 비관론자들이 잘못된 주장을 고수하는 동안 지속
    """

    def __init__(
        self,
        vol_lookback: int = 60,
        atr_threshold: float = 2.0,
        wow_lookback: int = 20,
        bear_decline: float = 0.20,
        bear_recovery: float = 0.10,
    ):
        self._vol = VolatilityNormalizer(vol_lookback, atr_threshold)
        self._wow = WallOfWorryDetector(wow_lookback)
        self._bear = BearMarketRecoveryDetector(bear_decline, bear_recovery)

    def analyze(self, df: pd.DataFrame) -> FisherSignal:
        """켄 피셔 통합 시그널을 생성합니다.

        Args:
            df: OHLCV DataFrame

        Returns:
            FisherSignal
        """
        # 1. 변동성 공포 분석
        vol_result = self._vol.analyze(df)

        # 2. 불신의 비관론 감지
        wow_result = self._wow.detect(df)

        # 3. 약세장 회복 감지
        bear_result = self._bear.detect(df)

        # 통합 점수 계산
        vol_fear = vol_result["fear_score"]         # 0~1 (공포 극단)
        wow_score = wow_result["wow_score"]         # 0~1 (불신 비관론)
        recovery = bear_result["recovery_score"]    # 0~1 (약세장 회복)

        # 켄 피셔 종합 신호 (-1~+1)
        # 공포+불신 비관론+회복 모두 높으면 → 강한 매수 신호
        opportunity_score = (
            vol_fear * 0.3 +
            wow_score * 0.4 +
            recovery * 0.3
        )

        # -1~+1 정규화: 0.5 이상 → 양수(기회), 미만 → 음수(위험)
        fisher_composite = (opportunity_score - 0.3) * 2.0  # 0.3 기준점
        fisher_composite = float(max(min(fisher_composite, 1.0), -1.0))

        # 포지션 배수 계산 (0.7~1.3)
        if fisher_composite > 0.5:
            position_multiplier = 1.3  # 강한 매수 기회
        elif fisher_composite > 0.2:
            position_multiplier = 1.1  # 기회
        elif fisher_composite < -0.5:
            position_multiplier = 0.7  # 위험 (안정적 성과 → 버블 전조)
        else:
            position_multiplier = 1.0  # 중립

        # 신뢰도 조정
        confidence_delta = (
            wow_result["confidence_boost"] +
            bear_result["confidence_boost"]
        )
        if vol_result["is_opportunity"]:
            confidence_delta += 0.1

        # 노트 구성
        notes = []
        if vol_result["volatility_regime"] != "NORMAL":
            notes.append(f"[변동성]{vol_result['note'][:50]}")
        if wow_result["wow_score"] > 0:
            notes.append(f"[불신비관론]{wow_result['note'][:50]}")
        if bear_result["recovery_score"] > 0:
            notes.append(f"[회복]{bear_result['note'][:50]}")

        note = " | ".join(notes) if notes else "켄피셔: 특이 패턴 없음"

        return FisherSignal(
            wall_of_worry_score=wow_score,
            volatility_fear_score=vol_fear,
            recovery_signal=recovery,
            fisher_composite=fisher_composite,
            position_multiplier=position_multiplier,
            confidence_delta=confidence_delta,
            note=note,
        )
