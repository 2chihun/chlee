"""환율 분석 모듈 - 백석현 "환율 모르면 주식투자 절대로 하지 마라" 기반

핵심 개념:
- 환율 = 분자(미국 경제) / 분모(미국 제외 세계 경제)
- 달러 사이클: 강세기(미국 상대적 우위) / 약세기(세계 상대적 우위)
- 달러 = 최후의 안전자산 (모든 자산 하락 시 유일하게 상승)
- AUD/JPY = 글로벌 위험선호도 바로미터
- KOSPI-환율 반비례 (공통 분모: 수출)

기술적 프록시 (가격 데이터만으로 환율 영향 감지):
- 달러 강세기: 수출주 불리, 내수주 상대적 유리
- 환율 급변: 시장 변동성 확대 신호
- 달러 방패: 포트폴리오의 15-25% 달러 자산 유지

임계값:
- 환율 일일 변동 ±1% (≈10원): 경보 발령
- 달러-원 상관계수 vs 세계증시: -0.59
- 한국주식-미국국채 상관: -0.55 (최상의 분산)
- 레버리지 상한: 1.5x (2x 절대 금지)
- 달러 자산 비중: 15-25% (항시 유지)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class DollarCyclePhase(Enum):
    """달러 사이클 단계"""
    STRONG_EARLY = "STRONG_EARLY"   # 달러 강세 초기 (미국 경제 우위 시작)
    STRONG_LATE = "STRONG_LATE"     # 달러 강세 후기 (과매수, 전환 임박)
    WEAK_EARLY = "WEAK_EARLY"       # 달러 약세 초기 (세계 경제 회복)
    WEAK_LATE = "WEAK_LATE"         # 달러 약세 후기 (과매도, 전환 임박)
    NEUTRAL = "NEUTRAL"             # 중립/판단 불가


@dataclass
class ExchangeRateSignal:
    """환율 분석 통합 시그널"""
    dollar_phase: DollarCyclePhase   # 달러 사이클 단계
    fx_alarm: bool                   # 환율 급변 경보 (±1%)
    fx_change_pct: float             # 환율 변동률 (%)
    dollar_strength: float           # 달러 강도 (0-100, 50=중립)
    risk_level: float                # 환율 리스크 수준 (0-100)
    kospi_impact: float              # KOSPI 영향도 (-1~+1)
    position_multiplier: float       # 포지션 조정 배수 (0.5-1.2)
    leverage_guard: float            # 레버리지 상한 (최대 1.5x)
    warning_message: str             # 경고 메시지


class DollarCycleAnalyzer:
    """달러 사이클 분석기

    백석현: 환율 = 분자(미국 경제) / 분모(미국 제외 세계 경제)
    - 분자 커지면 → 달러 강세 (환율 상승)
    - 분모 커지면 → 달러 약세 (환율 하락)

    기술적 프록시 (KOSPI 데이터 기반):
    - KOSPI-환율 반비례 → KOSPI 하락 = 환율 상승 (달러 강세)
    - 외국인 매도 = 원화→달러 전환 = 환율 상승
    - 달러 강세기에는 수출주 실적 개선이나 외국인 이탈로 주가 하락
    """

    def __init__(self, lookback: int = 120):
        self.lookback = lookback  # 약 6개월

    def analyze_dollar_strength(self, df: pd.DataFrame) -> float:
        """달러 강도 추정 (0-100, 50=중립)

        KOSPI 기반 역추정:
        - KOSPI 약세 + 거래량 감소 → 외국인 이탈 → 달러 강세 (점수 높음)
        - KOSPI 강세 + 거래량 증가 → 외국인 유입 → 달러 약세 (점수 낮음)
        """
        if df is None or len(df) < self.lookback:
            return 50.0

        close = df["close"].values
        volume = df["volume"].values if "volume" in df.columns else None

        # 장기 추세 (120일 이평 기울기)
        ma_long = np.mean(close[-self.lookback:])
        ma_short = np.mean(close[-20:])

        # KOSPI 강세 → 달러 약세 → 낮은 점수
        if ma_long > 0:
            trend = (ma_short - ma_long) / ma_long
            # KOSPI +10% → 달러 약세 (점수 20)
            # KOSPI -10% → 달러 강세 (점수 80)
            trend_score = 50 - trend * 300
        else:
            trend_score = 50.0

        # 거래량 추세 (외국인 참여도 프록시)
        if volume is not None and len(volume) >= self.lookback:
            vol_trend = np.mean(volume[-20:]) / np.mean(volume[-self.lookback:])
            # 거래량 증가 → 외국인 유입 → 달러 약세
            vol_adj = (1 - vol_trend) * 20
        else:
            vol_adj = 0.0

        strength = min(100.0, max(0.0, trend_score + vol_adj))
        return strength

    def determine_phase(self, dollar_strength: float,
                        strength_history: list = None) -> DollarCyclePhase:
        """달러 사이클 단계 판정

        Args:
            dollar_strength: 현재 달러 강도 (0-100)
            strength_history: 과거 강도 이력 (추세 판단용)
        """
        if strength_history and len(strength_history) >= 5:
            recent_trend = np.mean(strength_history[-5:]) - np.mean(
                strength_history[-20:] if len(strength_history) >= 20
                else strength_history
            )
        else:
            recent_trend = 0.0

        if dollar_strength >= 70:
            if recent_trend > 0:
                return DollarCyclePhase.STRONG_EARLY
            else:
                return DollarCyclePhase.STRONG_LATE
        elif dollar_strength <= 30:
            if recent_trend < 0:
                return DollarCyclePhase.WEAK_EARLY
            else:
                return DollarCyclePhase.WEAK_LATE
        else:
            return DollarCyclePhase.NEUTRAL


class FXAlarmSystem:
    """환율 급변 경보 시스템

    백석현: 환율 일일 ±1% (≈10원) 변동 시 경보
    - 경보 발생 빈도: 월 1회 수준
    - 환율 급변 = 시장 스트레스 신호

    기술적 프록시:
    - KOSPI 일일 변동률 2%+ = 환율 급변 동반 가능성 높음
    - 거래량 급증 + 급락 = 외국인 대규모 매도 (환율 상승)
    """

    def __init__(self, threshold_pct: float = 2.0):
        # KOSPI 2% 변동 ≈ 환율 1% 변동 (반비례 관계)
        self.threshold_pct = threshold_pct

    def check_alarm(self, df: pd.DataFrame) -> tuple:
        """환율 급변 경보 확인

        Returns:
            (alarm: bool, change_pct: float)
        """
        if df is None or len(df) < 2:
            return False, 0.0

        close = df["close"].values
        change_pct = abs(
            (close[-1] - close[-2]) / close[-2] * 100
        ) if close[-2] > 0 else 0.0

        alarm = change_pct >= self.threshold_pct
        return alarm, change_pct


class RiskBarometer:
    """글로벌 위험선호도 바로미터

    백석현: AUD/JPY = 세계 경제의 체온계
    - AUD 상승 (자원 수출국) + JPY 하락 (안전자산 회피) = 위험 선호
    - AUD 하락 + JPY 상승 = 위험 회피

    기술적 프록시 (KOSPI 기반):
    - RSI + 변동성 + 거래량 조합으로 위험선호도 추정
    - 고변동성 + 저거래량 = 위험 회피 (보수적 포지션)
    """

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def get_risk_appetite(self, df: pd.DataFrame) -> float:
        """위험선호도 점수 (0-100)

        높을수록 위험 선호 (공격적 포지션 가능)
        낮을수록 위험 회피 (방어적 포지션)
        """
        if df is None or len(df) < self.lookback + 14:
            return 50.0

        close = df["close"].values

        # RSI 기반 (상승 모멘텀 = 위험선호)
        rsi = self._calc_rsi(close, 14)
        rsi_score = rsi  # 0-100 그대로

        # 변동성 기반 (저변동성 = 위험선호)
        returns = np.diff(np.log(close[-self.lookback:]))
        vol = np.std(returns) * np.sqrt(252) * 100  # 연환산 %
        # 변동성 20% = 중립, 10% = 위험선호, 40% = 위험회피
        vol_score = max(0.0, min(100.0, 100 - (vol - 10) * 3.33))

        # 추세 기반 (상승 추세 = 위험선호)
        ma20 = np.mean(close[-20:])
        ma60 = np.mean(close[-min(60, len(close)):])
        if ma60 > 0:
            trend = (ma20 / ma60 - 1) * 100
            trend_score = min(100.0, max(0.0, 50 + trend * 10))
        else:
            trend_score = 50.0

        # 가중 합산
        appetite = rsi_score * 0.35 + vol_score * 0.35 + trend_score * 0.30
        return min(100.0, max(0.0, appetite))

    def _calc_rsi(self, close: np.ndarray, period: int) -> float:
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


class BiasCorrector:
    """심리 편향 보정기

    백석현의 3대 편향:
    1. 부정성 편향(Negativity Bias): 하락에 과잉 반응
    2. 최신 효과(Recency Effect): 최근 추세에 과잉 의존
    3. 레버리지 가드: 케인스 사례 - 방향 맞아도 레버리지로 파산

    기술적 구현:
    - 부정성 편향: 연속 하락 후 반등 신호 시 매수 신뢰도 보정
    - 최신 효과: 단기 vs 장기 추세 비교로 과잉 반응 필터링
    - 레버리지: 절대 상한 1.5x (2x 금지)
    """

    MAX_LEVERAGE = 1.5  # 절대 상한 (백석현 원칙)

    def correct_negativity_bias(self, df: pd.DataFrame,
                                 confidence: float) -> float:
        """부정성 편향 보정

        연속 하락 후 반등 구간에서 confidence를 과소평가하지 않도록 보정.
        5일 연속 하락 후 반등 시 +5~10% 신뢰도 보정.
        """
        if df is None or len(df) < 10:
            return confidence

        close = df["close"].values

        # 최근 5일 중 반등 확인
        if close[-1] > close[-2]:  # 오늘 상승
            # 그 전 5일 연속 하락 확인
            consec_down = 0
            for i in range(-2, -7, -1):
                if abs(i) < len(close) and close[i] < close[i - 1]:
                    consec_down += 1
                else:
                    break
            if consec_down >= 3:
                # 부정성 편향 보정: 과매도 반등에 대한 과소평가 교정
                bias_correction = min(0.10, consec_down * 0.02)
                corrected = min(1.0, confidence + bias_correction)
                logger.debug(
                    f"부정성 편향 보정: {confidence:.2f} → "
                    f"{corrected:.2f} (연속 {consec_down}일 하락 후 반등)"
                )
                return corrected

        return confidence

    def correct_recency_bias(self, df: pd.DataFrame,
                              confidence: float) -> float:
        """최신 효과 보정

        최근 5일 추세와 60일 추세가 반대일 때:
        - 단기 급등 but 장기 하락 → 신뢰도 과대 경고
        - 단기 급락 but 장기 상승 → 신뢰도 과소 보정
        """
        if df is None or len(df) < 60:
            return confidence

        close = df["close"].values
        ret_5d = (close[-1] - close[-6]) / close[-6] if close[-6] > 0 else 0
        ret_60d = (close[-1] - close[-61]) / close[-61] if close[-61] > 0 else 0

        # 단기 급등 but 장기 하락 → 과열 경고 (신뢰도 하향)
        if ret_5d > 0.05 and ret_60d < -0.05:
            correction = -0.05
        # 단기 급락 but 장기 상승 → 과잉 비관 보정 (신뢰도 상향)
        elif ret_5d < -0.05 and ret_60d > 0.05:
            correction = 0.05
        else:
            correction = 0.0

        corrected = min(1.0, max(0.0, confidence + correction))
        if correction != 0:
            logger.debug(
                f"최신 효과 보정: {confidence:.2f} → {corrected:.2f} "
                f"(5D={ret_5d:.1%}, 60D={ret_60d:.1%})"
            )
        return corrected

    def enforce_leverage_guard(self, leverage: float) -> float:
        """레버리지 가드 (절대 상한 1.5x)

        백석현 + 케인스 사례:
        "방향이 맞아도 레버리지 과다 사용 시 파산 가능"
        """
        if leverage > self.MAX_LEVERAGE:
            logger.warning(
                f"레버리지 가드 발동: {leverage:.1f}x → "
                f"{self.MAX_LEVERAGE}x (상한 강제 적용)"
            )
            return self.MAX_LEVERAGE
        return leverage


class ExchangeRateAnalyzer:
    """환율 분석 통합 모듈

    백석현 "환율 모르면 주식투자 절대로 하지 마라" 핵심 프레임워크:
    1. 달러 사이클 판단 (강세기/약세기)
    2. 환율 급변 경보 (±1%)
    3. 글로벌 위험선호도 (AUD/JPY 프록시)
    4. 심리 편향 보정 (부정성/최신/레버리지)
    5. KOSPI 영향도 (환율-주가 반비례)

    포지션 조정:
    - 달러 강세 초기: 0.7x (수출주 실적↑이나 외국인 이탈↓)
    - 달러 강세 후기: 0.8x (전환 임박, 주의)
    - 달러 약세 초기: 1.1x (외국인 유입, 한국 시장 유리)
    - 달러 약세 후기: 1.0x (중립, 전환 대비)
    - 환율 급변 시: 추가 20% 축소
    """

    def __init__(self, config=None):
        self.config = config
        self.cycle_analyzer = DollarCycleAnalyzer()
        self.fx_alarm = FXAlarmSystem()
        self.risk_barometer = RiskBarometer()
        self.bias_corrector = BiasCorrector()
        self._strength_history = []

    def analyze(self, df: pd.DataFrame) -> ExchangeRateSignal:
        """통합 환율 분석

        Args:
            df: OHLCV DataFrame

        Returns:
            ExchangeRateSignal: 환율 영향 분석 결과
        """
        if df is None or len(df) < 60:
            return ExchangeRateSignal(
                dollar_phase=DollarCyclePhase.NEUTRAL,
                fx_alarm=False,
                fx_change_pct=0.0,
                dollar_strength=50.0,
                risk_level=0.0,
                kospi_impact=0.0,
                position_multiplier=1.0,
                leverage_guard=1.5,
                warning_message="데이터 부족 - 분석 불가",
            )

        # 1. 달러 강도 분석
        dollar_strength = self.cycle_analyzer.analyze_dollar_strength(df)
        self._strength_history.append(dollar_strength)
        # 이력 최대 120개 유지
        if len(self._strength_history) > 120:
            self._strength_history = self._strength_history[-120:]

        # 2. 달러 사이클 판정
        dollar_phase = self.cycle_analyzer.determine_phase(
            dollar_strength, self._strength_history
        )

        # 3. 환율 급변 경보
        alarm, change_pct = self.fx_alarm.check_alarm(df)

        # 4. 위험선호도
        risk_appetite = self.risk_barometer.get_risk_appetite(df)

        # 5. 환율 리스크 수준
        risk_level = self._calc_risk_level(
            dollar_strength, alarm, risk_appetite
        )

        # 6. KOSPI 영향도 (-1~+1)
        kospi_impact = self._calc_kospi_impact(dollar_phase, alarm)

        # 7. 포지션 배수
        position_multiplier = self._calc_position_multiplier(
            dollar_phase, alarm, risk_appetite
        )

        # 경고 메시지
        warning = self._generate_warning(
            dollar_phase, alarm, change_pct, dollar_strength
        )

        signal = ExchangeRateSignal(
            dollar_phase=dollar_phase,
            fx_alarm=alarm,
            fx_change_pct=round(change_pct, 2),
            dollar_strength=round(dollar_strength, 1),
            risk_level=round(risk_level, 1),
            kospi_impact=round(kospi_impact, 2),
            position_multiplier=round(position_multiplier, 2),
            leverage_guard=BiasCorrector.MAX_LEVERAGE,
            warning_message=warning,
        )

        logger.debug(
            f"ExchangeRate: phase={dollar_phase.value}, "
            f"strength={dollar_strength:.1f}, "
            f"alarm={alarm}, multiplier={position_multiplier:.2f}"
        )
        return signal

    def correct_confidence(self, df: pd.DataFrame,
                           confidence: float) -> float:
        """심리 편향 보정된 신뢰도 반환"""
        conf = self.bias_corrector.correct_negativity_bias(df, confidence)
        conf = self.bias_corrector.correct_recency_bias(df, conf)
        return conf

    def _calc_risk_level(self, dollar_strength: float,
                         alarm: bool, risk_appetite: float) -> float:
        """환율 리스크 수준 (0-100)"""
        # 달러 극단(매우 강하거나 매우 약한) = 높은 리스크
        extreme = abs(dollar_strength - 50) * 2
        alarm_adj = 30.0 if alarm else 0.0
        # 위험회피(낮은 appetite) = 높은 리스크
        risk_avoid = max(0.0, (50 - risk_appetite))
        return min(100.0, extreme * 0.40 + alarm_adj + risk_avoid * 0.30)

    def _calc_kospi_impact(self, phase: DollarCyclePhase,
                           alarm: bool) -> float:
        """KOSPI 영향도 (-1~+1, 양수=긍정, 음수=부정)"""
        base_impact = {
            DollarCyclePhase.STRONG_EARLY: -0.3,  # 외국인 이탈
            DollarCyclePhase.STRONG_LATE: -0.1,    # 전환 기대
            DollarCyclePhase.WEAK_EARLY: 0.4,      # 외국인 유입
            DollarCyclePhase.WEAK_LATE: 0.2,       # 전환 주의
            DollarCyclePhase.NEUTRAL: 0.0,
        }
        impact = base_impact.get(phase, 0.0)
        if alarm:
            impact -= 0.2  # 급변 시 부정적 영향 강화
        return max(-1.0, min(1.0, impact))

    def _calc_position_multiplier(self, phase: DollarCyclePhase,
                                   alarm: bool,
                                   risk_appetite: float) -> float:
        """포지션 배수 결정"""
        base = {
            DollarCyclePhase.STRONG_EARLY: 0.7,
            DollarCyclePhase.STRONG_LATE: 0.8,
            DollarCyclePhase.WEAK_EARLY: 1.1,
            DollarCyclePhase.WEAK_LATE: 1.0,
            DollarCyclePhase.NEUTRAL: 1.0,
        }
        multiplier = base.get(phase, 1.0)

        # 환율 급변 시 20% 추가 축소
        if alarm:
            multiplier *= 0.8

        # 위험회피 심한 경우 추가 조정
        if risk_appetite < 30:
            multiplier *= 0.9

        return max(0.5, min(1.2, multiplier))

    def _generate_warning(self, phase: DollarCyclePhase,
                           alarm: bool, change_pct: float,
                           strength: float) -> str:
        """경고 메시지 생성"""
        msgs = []
        if alarm:
            msgs.append(
                f"[FX경보] 주가 변동 {change_pct:.1f}% - "
                f"환율 급변 동반 가능"
            )
        if phase in (DollarCyclePhase.STRONG_EARLY,
                     DollarCyclePhase.STRONG_LATE):
            msgs.append(
                f"[달러강세] 달러강도={strength:.0f}. "
                f"외국인 자금 이탈 주의"
            )
        elif phase in (DollarCyclePhase.WEAK_EARLY,
                       DollarCyclePhase.WEAK_LATE):
            msgs.append(
                f"[달러약세] 달러강도={strength:.0f}. "
                f"한국 시장 외국인 유입 기대"
            )
        if not msgs:
            msgs.append(f"[중립] 달러강도={strength:.0f}")
        return " | ".join(msgs)
