"""가치투자 분석 모듈

강방천 & 존리 "나의 첫 주식 교과서" 핵심 개념 기반.

강방천 4원칙: 좋은것을 사라, 쌀때 사라, 분산하라, 오래 함께 하라
존리: "We buy company, not paper" - 기업가치 기반 장기투자
역발상(Contrarian): 남들이 공포에 파는 1등 기업을 매수

주요 기능:
- 재무건전성 점수 (PER/PBR/ROE 기반)
- 저평가 판단 (가치 > 가격)
- 역발상 매수 기회 감지 (공포 극단 + 반등)
- 장기보유 확신도 (매도 4조건 검사)
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ValueInvestorSignal:
    """가치투자 통합 시그널"""
    fundamental_score: float = 0.5   # 재무건전성 (0~1)
    valuation_score: float = 0.5     # 저평가 점수 (0~1, 높을수록 저평가)
    contrarian_score: float = 0.0    # 역발상 기회 (0~1)
    hold_conviction: float = 0.5     # 보유 확신도 (0~1)
    sell_reason: Optional[str] = None  # 매도 4조건 해당 사유
    position_multiplier: float = 1.0  # 포지션 조정 배수 (0.5~1.5)
    confidence_delta: float = 0.0     # 신뢰도 조정값
    note: str = ""


class FundamentalScorer:
    """재무건전성 점수 산출기

    강방천: "좋은 것을 사라" = 재무적으로 건전한 기업
    존리: EPS/PER/PBR/ROE/EV/EBITDA 분석 능력이 투자의 기본

    기술적 지표(OHLCV)만으로는 재무 데이터를 직접 평가할 수 없으므로,
    가격 행동 패턴에서 재무 건전성의 프록시 지표를 추출합니다:
    - 장기 우상향 추세: 실적 성장 기업의 가격 특성
    - 낮은 변동성: 안정적 실적 기업
    - 꾸준한 거래량: 기관 관심 기업
    """

    def __init__(self, lookback: int = 120):
        self.lookback = lookback

    def score(self, df: pd.DataFrame) -> float:
        """재무건전성 프록시 점수 (0~1)"""
        if len(df) < self.lookback:
            return 0.5

        close = df["close"].astype(float).values
        recent = close[-self.lookback:]

        scores = []

        # 1. 장기 추세 점수 (우상향 = 좋은 기업)
        trend = (recent[-1] - recent[0]) / recent[0] if recent[0] > 0 else 0
        trend_score = min(max((trend + 0.5) / 1.0, 0), 1)  # -50%~+50% → 0~1
        scores.append(trend_score * 0.4)

        # 2. 변동성 안정성 (낮은 변동성 = 안정 실적)
        returns = np.diff(recent) / recent[:-1]
        vol = np.std(returns) if len(returns) > 0 else 0.03
        vol_score = max(1.0 - vol / 0.05, 0)  # 5% 이상 변동성은 0점
        scores.append(vol_score * 0.3)

        # 3. 거래량 안정성 (꾸준한 거래량 = 기관 관심)
        if "volume" in df.columns:
            vol_data = df["volume"].astype(float).values[-self.lookback:]
            if len(vol_data) > 20:
                avg_vol = np.mean(vol_data)
                vol_std = np.std(vol_data)
                vol_cv = vol_std / avg_vol if avg_vol > 0 else 1.0
                vol_stab = max(1.0 - vol_cv / 2.0, 0)
                scores.append(vol_stab * 0.3)
            else:
                scores.append(0.15)
        else:
            scores.append(0.15)

        return min(sum(scores), 1.0)


class ContrarianDetector:
    """역발상 매수 기회 감지기

    강방천: "쌀 때 사라" + "남들이 안 좋다고 할 때 1등 기업 매수"
    존리: 대중이 공포에 매도할 때가 매수 기회
    켄 피셔와 유사: 비관론 극단 = 매수 기회

    감지 조건:
    - RSI 극단 저점 (과매도 구간) 후 반등
    - 볼린저밴드 하단 이탈 후 복귀
    - 연속 하락 후 반등 캔들 (거래량 동반)
    """

    def __init__(self, rsi_threshold: float = 30.0,
                 consecutive_decline_days: int = 5):
        self.rsi_threshold = rsi_threshold
        self.consecutive_decline_days = consecutive_decline_days

    def detect(self, df: pd.DataFrame) -> float:
        """역발상 기회 점수 (0~1)"""
        if len(df) < 20:
            return 0.0

        scores = []

        # 1. RSI 과매도 후 반등 (존리: 공포 = 기회)
        if "rsi" in df.columns:
            rsi_vals = df["rsi"].astype(float).values
            current_rsi = rsi_vals[-1]
            recent_min_rsi = np.min(rsi_vals[-10:]) if len(rsi_vals) >= 10 else current_rsi
            if recent_min_rsi < self.rsi_threshold and current_rsi > recent_min_rsi + 5:
                # 과매도 후 반등 중
                recovery_strength = (current_rsi - recent_min_rsi) / 20.0
                scores.append(min(recovery_strength, 1.0) * 0.4)
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)

        # 2. 볼린저밴드 하단 이탈 후 복귀
        if "bb_lower" in df.columns and "bb_mid" in df.columns:
            close = df["close"].astype(float).values
            bb_lower = df["bb_lower"].astype(float).values
            bb_mid = df["bb_mid"].astype(float).values
            if len(close) >= 5:
                # 최근 5일 내 BB 하단 이탈 이력
                below_bb = close[-5:] < bb_lower[-5:]
                if np.any(below_bb) and close[-1] > bb_lower[-1]:
                    # 하단 이탈 후 복귀
                    recovery_pct = (close[-1] - bb_lower[-1]) / (bb_mid[-1] - bb_lower[-1]) if bb_mid[-1] > bb_lower[-1] else 0
                    scores.append(min(recovery_pct, 1.0) * 0.3)
                else:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)

        # 3. 연속 하락 후 반등 (거래량 동반)
        close = df["close"].astype(float).values
        if len(close) >= self.consecutive_decline_days + 1:
            changes = np.diff(close[-(self.consecutive_decline_days + 1):])
            decline_days = np.sum(changes[:-1] < 0)
            today_up = changes[-1] > 0

            if decline_days >= self.consecutive_decline_days - 1 and today_up:
                # 연속 하락 후 반등
                bounce_strength = 0.5
                if "volume" in df.columns:
                    vol = df["volume"].astype(float).values
                    if len(vol) >= 10:
                        avg_vol = np.mean(vol[-20:-1]) if len(vol) >= 20 else np.mean(vol[:-1])
                        if avg_vol > 0 and vol[-1] > avg_vol * 1.5:
                            bounce_strength = 1.0  # 거래량 동반 반등
                scores.append(bounce_strength * 0.3)
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)

        return min(sum(scores), 1.0)


class LongTermHoldEvaluator:
    """장기보유 확신도 평가기

    존리: "손절매 하지 마라" (단, 펀더멘털 변화 시에만 매도)
    강방천: "오래 함께 하라" (위대한 기업과 동행)

    매도 4조건 검사:
    1. 주가가 기업가치보다 비쌀 때 (고평가)
    2. 기업 펀더멘털이 변했을 때
    3. 세상이 변했을 때
    4. 더 좋은 주식을 발견했을 때 (기회비용)

    기술적 지표로 판단 가능한 1번, 2번을 중점 구현.
    """

    def __init__(self, overvalued_rsi: float = 80.0,
                 hold_min_days: int = 60):
        self.overvalued_rsi = overvalued_rsi
        self.hold_min_days = hold_min_days

    def evaluate(self, df: pd.DataFrame,
                 holding_days: int = 0) -> tuple:
        """장기보유 확신도와 매도 사유 평가

        Returns:
            (hold_conviction: float, sell_reason: Optional[str])
        """
        if len(df) < 20:
            return 0.5, None

        conviction = 0.7  # 기본 보유 확신도 (높음 = 존리 원칙)
        sell_reason = None

        close = df["close"].astype(float).values

        # 매도조건 1: 고평가 감지 (RSI 극단 과열 + BB 상단 돌파)
        overvalued = False
        if "rsi" in df.columns:
            current_rsi = float(df["rsi"].iloc[-1])
            if current_rsi > self.overvalued_rsi:
                overvalued = True
                conviction -= 0.2

        if "bb_upper" in df.columns:
            if close[-1] > float(df["bb_upper"].iloc[-1]):
                overvalued = True
                conviction -= 0.1

        if overvalued and "rsi" in df.columns and float(df["rsi"].iloc[-1]) > 85:
            sell_reason = "고평가경고(RSI극단+BB상단)"
            conviction -= 0.2

        # 매도조건 2: 펀더멘털 변화 프록시 (장기 추세 붕괴)
        if len(close) >= 60:
            ema60 = pd.Series(close).ewm(span=60, adjust=False).mean().values
            if close[-1] < ema60[-1] * 0.85:
                # 60일 EMA 대비 15% 이상 하락 = 추세 붕괴
                if sell_reason is None:
                    sell_reason = "펀더멘털변화의심(장기추세붕괴)"
                conviction -= 0.3

        # 보유 기간에 따른 확신도 보정 (존리: 시간이 수익을 결정)
        if holding_days > self.hold_min_days:
            conviction += 0.1  # 장기 보유 보너스
        elif holding_days < 5:
            conviction -= 0.05  # 초단기 패널티

        conviction = max(min(conviction, 1.0), 0.0)
        return conviction, sell_reason


class ValueInvestorAnalyzer:
    """가치투자 통합 분석기

    강방천 4원칙 + 존리 장기투자 + 역발상 투자를 통합합니다.

    Args:
        fundamental_lookback: 재무 프록시 분석 기간
        contrarian_rsi: 역발상 RSI 임계값
        hold_min_days: 최소 보유 권장일
        overvalued_rsi: 고평가 RSI 기준
    """

    def __init__(
        self,
        fundamental_lookback: int = 120,
        contrarian_rsi: float = 30.0,
        hold_min_days: int = 60,
        overvalued_rsi: float = 80.0,
    ):
        self._fundamental = FundamentalScorer(lookback=fundamental_lookback)
        self._contrarian = ContrarianDetector(rsi_threshold=contrarian_rsi)
        self._hold_eval = LongTermHoldEvaluator(
            overvalued_rsi=overvalued_rsi,
            hold_min_days=hold_min_days,
        )

    def analyze(
        self, df: pd.DataFrame, holding_days: int = 0
    ) -> ValueInvestorSignal:
        """가치투자 통합 분석

        Args:
            df: OHLCV DataFrame (최소 20봉 권장, 120봉 이상 최적)
            holding_days: 현재 보유 일수 (0이면 미보유)

        Returns:
            ValueInvestorSignal
        """
        # 재무건전성 프록시
        fund_score = self._fundamental.score(df)

        # 저평가 점수 (RSI 저점 + BB 하단 근처 = 저평가 프록시)
        val_score = self._calc_valuation_score(df)

        # 역발상 기회
        contrarian_score = self._contrarian.detect(df)

        # 장기보유 확신도 + 매도 사유
        hold_conv, sell_reason = self._hold_eval.evaluate(df, holding_days)

        # 포지션 배수 계산
        pos_mult = self._calc_position_multiplier(
            fund_score, val_score, contrarian_score
        )

        # 신뢰도 조정값
        conf_delta = self._calc_confidence_delta(
            fund_score, val_score, contrarian_score
        )

        # 노트
        notes = []
        if contrarian_score > 0.5:
            notes.append(f"역발상기회({contrarian_score:.2f})")
        if val_score > 0.7:
            notes.append(f"저평가({val_score:.2f})")
        if val_score < 0.3:
            notes.append(f"고평가경고({val_score:.2f})")
        if sell_reason:
            notes.append(sell_reason)

        return ValueInvestorSignal(
            fundamental_score=fund_score,
            valuation_score=val_score,
            contrarian_score=contrarian_score,
            hold_conviction=hold_conv,
            sell_reason=sell_reason,
            position_multiplier=pos_mult,
            confidence_delta=conf_delta,
            note=" | ".join(notes) if notes else "",
        )

    def _calc_valuation_score(self, df: pd.DataFrame) -> float:
        """저평가 점수 산출 (기술적 프록시)

        가치 > 가격일 때 높은 점수:
        - RSI 저점 = 과매도 = 저평가 프록시
        - BB %B 낮음 = 밴드 하단 = 저평가 프록시
        - 52주 저점 근처 = 저평가 프록시
        """
        scores = []

        # RSI 기반 저평가 (낮을수록 저평가)
        if "rsi" in df.columns:
            current_rsi = float(df["rsi"].iloc[-1])
            rsi_val = max(1.0 - current_rsi / 70.0, 0)
            scores.append(rsi_val * 0.35)
        else:
            scores.append(0.175)

        # BB %B 기반 저평가 (낮을수록 저평가)
        if "bb_pctb" in df.columns:
            pctb = float(df["bb_pctb"].iloc[-1])
            pctb_val = max(1.0 - pctb, 0)
            scores.append(pctb_val * 0.30)
        else:
            scores.append(0.15)

        # 52주 저점 대비 위치
        close = df["close"].astype(float).values
        if len(close) >= 60:
            lookback = min(252, len(close))
            high_52w = np.max(close[-lookback:])
            low_52w = np.min(close[-lookback:])
            if high_52w > low_52w:
                position = (close[-1] - low_52w) / (high_52w - low_52w)
                # 저점 근처일수록 높은 저평가 점수
                low_val = max(1.0 - position, 0)
                scores.append(low_val * 0.35)
            else:
                scores.append(0.175)
        else:
            scores.append(0.175)

        return min(sum(scores), 1.0)

    def _calc_position_multiplier(
        self, fund_score: float, val_score: float,
        contrarian_score: float,
    ) -> float:
        """포지션 조정 배수 (0.5~1.5)

        강방천: "좋은것을 쌀때 사라" → 재무건전+저평가 = 확대
        존리: 역발상 기회 = 적극 매수
        """
        mult = 1.0

        # 재무건전 + 저평가 = 포지션 확대
        if fund_score > 0.7 and val_score > 0.6:
            mult += 0.2  # 좋은 기업 + 저평가
        elif fund_score < 0.3 or val_score < 0.2:
            mult -= 0.3  # 나쁜 기업 또는 극단 고평가

        # 역발상 기회 = 추가 확대
        if contrarian_score > 0.6:
            mult += 0.2  # 공포 극단 매수 기회

        return round(max(min(mult, 1.5), 0.5), 2)

    def _calc_confidence_delta(
        self, fund_score: float, val_score: float,
        contrarian_score: float,
    ) -> float:
        """신뢰도 조정값 산출"""
        delta = 0.0

        # 역발상 기회 시 신뢰도 증가
        if contrarian_score > 0.6:
            delta += 0.10  # 공포 극단 = 매수 기회

        # 저평가 기업 신뢰도 증가
        if val_score > 0.7:
            delta += 0.05

        # 고평가 경고 시 신뢰도 감소
        if val_score < 0.3:
            delta -= 0.10  # 고평가 = 매수 자제

        # 재무건전성 낮으면 감소
        if fund_score < 0.3:
            delta -= 0.05

        return round(delta, 3)
