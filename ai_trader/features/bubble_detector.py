"""AI 버블 감지 모듈 - 조상철 "2026년 11월, 주식시장 대폭락" 기반

핵심 개념:
- CAPEX 대비 실적 GAP → 버블 초기 신호
- 연쇄 마진콜 피드백 루프 → 폭락 가속
- 만기 벽(Maturity Wall) → 시스템 리스크 타이밍
- 패시브 펀드 기계적 매도 → 폭락 확대

기술적 프록시 (가격 데이터만으로 버블 징후 감지):
- 과열도: 장기 이평 대비 괴리율 + RSI 극단 + 볼린저밴드 이탈
- 레버리지 리스크: 변동성 급증 + 거래량 급감 (유동성 고갈)
- 연쇄 매도 리스크: 갭다운 빈도 + 하한가 빈도 + 거래량 폭증
- 패시브 매도 리스크: 섹터 동조화 (베타 급증)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class BubblePhase(Enum):
    """버블 단계 (조상철 4단계 폭락 경로 기반)"""
    NORMAL = "NORMAL"       # 정상 시장
    EUPHORIA = "EUPHORIA"   # 과열/탐욕 단계 (레버리지 축적 중)
    PEAK = "PEAK"           # 정점 (만기벽 접근, 담보가치 하락 시작)
    BURST = "BURST"         # 붕괴 진행 (연쇄 마진콜, 킬 스위치)
    PANIC = "PANIC"         # 패닉 매도 (패시브 펀드 기계적 매도, 서킷브레이커)


@dataclass
class BubbleSignal:
    """버블 감지 통합 시그널"""
    phase: BubblePhase              # 현재 버블 단계
    bubble_score: float             # 버블 점수 (0-100, 높을수록 위험)
    overheat_score: float           # 과열도 점수 (0-100)
    leverage_risk: float            # 레버리지 리스크 (0-100)
    cascade_risk: float             # 연쇄 매도 리스크 (0-100)
    passive_risk: float             # 패시브 매도 리스크 (0-100)
    position_multiplier: float      # 포지션 조정 배수 (0.1-1.0)
    warning_message: str            # 경고 메시지


class OverheatDetector:
    """시장 과열도 감지 (버블 EUPHORIA 단계 탐지)

    장기 이평 대비 괴리율, RSI 극단, 볼린저밴드 이탈률 등으로
    CAPEX 과잉투자 → 주가 과열 구간을 기술적으로 감지.
    """

    def __init__(self, ma_period: int = 200, bb_period: int = 20,
                 bb_std: float = 2.0):
        self.ma_period = ma_period
        self.bb_period = bb_period
        self.bb_std = bb_std

    def analyze(self, df: pd.DataFrame) -> float:
        """과열도 점수 산출 (0-100)

        구성 요소 (가중합산):
        - 200일 이평 괴리율 (30%): 20%+ 이탈 = 100점
        - RSI 14일 (25%): 80+ = 100점
        - 볼린저밴드 상단 이탈 빈도 (20%): 최근 20일 중 이탈 비율
        - 연속 상승 일수 (15%): 10일+ 연속 = 100점
        - 52주 신고가 대비 위치 (10%): 98%+ = 100점
        """
        if df is None or len(df) < self.ma_period:
            return 0.0

        close = df["close"].values
        current = close[-1]

        # 1) 200일 이평 괴리율
        ma200 = np.mean(close[-self.ma_period:])
        deviation_pct = (current - ma200) / ma200 * 100 if ma200 > 0 else 0
        deviation_score = min(100.0, max(0.0, deviation_pct * 5))  # 20%→100

        # 2) RSI 14일
        rsi_val = self._calc_rsi(close, 14)
        rsi_score = max(0.0, (rsi_val - 50) * 2) if rsi_val > 50 else 0.0

        # 3) BB 상단 이탈 빈도 (최근 20일)
        bb_upper = self._calc_bb_upper(close, self.bb_period, self.bb_std)
        if bb_upper is not None:
            recent_closes = close[-self.bb_period:]
            bb_breach_count = sum(1 for c in recent_closes if c > bb_upper)
            bb_score = (bb_breach_count / self.bb_period) * 100
        else:
            bb_score = 0.0

        # 4) 연속 상승 일수
        consec_up = 0
        for i in range(len(close) - 1, 0, -1):
            if close[i] > close[i - 1]:
                consec_up += 1
            else:
                break
        consec_score = min(100.0, consec_up * 10)  # 10일→100

        # 5) 52주 신고가 대비 위치
        high_252 = max(close[-min(252, len(close)):])
        high_pct = (current / high_252 * 100) if high_252 > 0 else 0
        high_score = max(0.0, (high_pct - 90) * 10)  # 90%→0, 100%→100

        total = (deviation_score * 0.30
                 + rsi_score * 0.25
                 + bb_score * 0.20
                 + consec_score * 0.15
                 + high_score * 0.10)
        return min(100.0, max(0.0, total))

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

    def _calc_bb_upper(self, close: np.ndarray, period: int,
                       std_mult: float) -> Optional[float]:
        if len(close) < period:
            return None
        recent = close[-period:]
        return np.mean(recent) + std_mult * np.std(recent)


class CascadeRiskAnalyzer:
    """연쇄 매도 리스크 분석 (마진콜 피드백 루프 감지)

    조상철의 연쇄 마진콜 메커니즘:
    A기업 마진콜 → 매물 → 가격하락 → B기업 마진콜 → ... → 바닥

    기술적 프록시:
    - 갭다운 빈도 증가 (킬 스위치/강제 청산 징후)
    - 거래량 급증 (패닉 매도)
    - 하락 가속도 (가격 하락 속도 증가)
    - ATR 급등 (변동성 폭발)
    """

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def analyze(self, df: pd.DataFrame) -> float:
        """연쇄 매도 리스크 점수 (0-100)

        구성 요소:
        - 갭다운 빈도 (30%): 최근 20일 중 -2% 이상 갭다운 비율
        - 거래량 폭증 (25%): 평균 대비 3배 이상 거래량 빈도
        - 하락 가속도 (25%): 최근 5일 하락률 > 최근 20일 하락률
        - ATR 급등 (20%): ATR이 60일 평균 대비 2배 이상
        """
        if df is None or len(df) < max(self.lookback, 60):
            return 0.0

        close = df["close"].values
        open_ = df["open"].values if "open" in df.columns else close
        high = df["high"].values if "high" in df.columns else close
        low = df["low"].values if "low" in df.columns else close
        volume = df["volume"].values if "volume" in df.columns else None

        n = self.lookback

        # 1) 갭다운 빈도 (-2% 이상 갭)
        gap_downs = 0
        for i in range(-n, 0):
            if len(close) + i > 0:
                prev_close = close[i - 1]
                curr_open = open_[i]
                if prev_close > 0:
                    gap_pct = (curr_open - prev_close) / prev_close
                    if gap_pct <= -0.02:
                        gap_downs += 1
        gap_score = min(100.0, (gap_downs / n) * 500)  # 4번→100점

        # 2) 거래량 폭증
        if volume is not None and len(volume) >= 60:
            avg_vol_60 = np.mean(volume[-60:])
            if avg_vol_60 > 0:
                vol_spikes = sum(
                    1 for v in volume[-n:] if v > avg_vol_60 * 3
                )
                vol_score = min(100.0, (vol_spikes / n) * 500)
            else:
                vol_score = 0.0
        else:
            vol_score = 0.0

        # 3) 하락 가속도
        if len(close) >= n:
            ret_5d = (close[-1] - close[-6]) / close[-6] if close[-6] > 0 else 0
            ret_20d = (close[-1] - close[-n - 1]) / close[-n - 1] if close[-n - 1] > 0 else 0
            # 5일 하락률이 20일 하락률보다 크면 가속
            if ret_5d < 0 and ret_20d < 0 and abs(ret_20d) > 1e-10:
                accel = abs(ret_5d) / abs(ret_20d) * (n / 5)
                accel_score = min(100.0, max(0.0, (accel - 1) * 100))
            elif ret_5d < -0.05:
                accel_score = min(100.0, abs(ret_5d) * 1000)
            else:
                accel_score = 0.0
        else:
            accel_score = 0.0

        # 4) ATR 급등
        atr_14 = self._calc_atr(high, low, close, 14)
        atr_60 = self._calc_atr(high, low, close, 60)
        if atr_60 > 0:
            atr_ratio = atr_14 / atr_60
            atr_score = min(100.0, max(0.0, (atr_ratio - 1) * 100))
        else:
            atr_score = 0.0

        total = (gap_score * 0.30
                 + vol_score * 0.25
                 + accel_score * 0.25
                 + atr_score * 0.20)
        return min(100.0, max(0.0, total))

    def _calc_atr(self, high: np.ndarray, low: np.ndarray,
                  close: np.ndarray, period: int) -> float:
        if len(high) < period + 1:
            return 0.0
        tr_list = []
        for i in range(-period, 0):
            h = high[i]
            l = low[i]
            pc = close[i - 1]
            tr = max(h - l, abs(h - pc), abs(l - pc))
            tr_list.append(tr)
        return np.mean(tr_list)


class LeverageRiskMonitor:
    """레버리지 리스크 모니터 (시스템 리스크 프록시)

    조상철의 GPU 파이낸싱 구조를 일반화:
    - 변동성 급증 = 레버리지 해소(deleveraging) 징후
    - 거래량 급감 후 급증 = 유동성 고갈 → 강제 청산
    - 상관관계 급증 = 섹터 동조화 (패시브 매도)
    """

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def analyze(self, df: pd.DataFrame) -> float:
        """레버리지 리스크 점수 (0-100)

        구성 요소:
        - 변동성 비율 (40%): 최근 10일 변동성 / 60일 변동성
        - 유동성 고갈 (30%): 거래량 급감 후 급증 패턴
        - 일중 변동폭 확대 (30%): (고가-저가)/종가 비율 증가
        """
        if df is None or len(df) < self.lookback:
            return 0.0

        close = df["close"].values
        volume = df["volume"].values if "volume" in df.columns else None
        high = df["high"].values if "high" in df.columns else close
        low = df["low"].values if "low" in df.columns else close

        # 1) 변동성 비율 (10일 vol / 60일 vol)
        if len(close) >= self.lookback:
            returns = np.diff(np.log(close[-self.lookback:]))
            vol_10 = np.std(returns[-10:]) if len(returns) >= 10 else 0
            vol_60 = np.std(returns) if len(returns) > 0 else 0.001
            vol_ratio = vol_10 / vol_60 if vol_60 > 0 else 1.0
            vol_score = min(100.0, max(0.0, (vol_ratio - 1) * 100))
        else:
            vol_score = 0.0

        # 2) 유동성 고갈 패턴
        if volume is not None and len(volume) >= self.lookback:
            avg_vol = np.mean(volume[-self.lookback:])
            recent_vol = volume[-5:]
            if avg_vol > 0:
                # 최근 5일 중 급감(50% 이하) 또는 급증(300% 이상) 감지
                dry_days = sum(1 for v in recent_vol if v < avg_vol * 0.5)
                flood_days = sum(1 for v in recent_vol if v > avg_vol * 3.0)
                liquidity_score = min(100.0, (dry_days + flood_days) * 25)
            else:
                liquidity_score = 0.0
        else:
            liquidity_score = 0.0

        # 3) 일중 변동폭 확대
        if len(high) >= 20:
            intraday_ranges = []
            for i in range(-20, 0):
                if close[i] > 0:
                    r = (high[i] - low[i]) / close[i]
                    intraday_ranges.append(r)
            if intraday_ranges:
                avg_range = np.mean(intraday_ranges)
                recent_range = np.mean(intraday_ranges[-5:])
                if avg_range > 0:
                    range_ratio = recent_range / avg_range
                    range_score = min(100.0, max(0.0, (range_ratio - 1) * 100))
                else:
                    range_score = 0.0
            else:
                range_score = 0.0
        else:
            range_score = 0.0

        total = (vol_score * 0.40
                 + liquidity_score * 0.30
                 + range_score * 0.30)
        return min(100.0, max(0.0, total))


class PassiveFundCascadeEstimator:
    """패시브 펀드 연쇄 매도 리스크 추정

    조상철의 4단계 폭락 경로:
    엔비디아 급락 → AI 테마주 매도 → 패시브 펀드 기계적 매도 → 서킷브레이커

    기술적 프록시:
    - 대형주 동조 하락 (베타 수렴)
    - 거래량 동시 급증 (기계적 매도 특성)
    - 하락 종목 비율 (시장 폭: breadth)
    """

    def __init__(self, lookback: int = 10):
        self.lookback = lookback

    def analyze(self, df: pd.DataFrame) -> float:
        """패시브 매도 리스크 점수 (0-100)

        단일 종목 기준 프록시:
        - 시장 전체 급락 시 동조 하락 강도
        - 거래량 프로파일 이상 (기계적 매도 패턴)
        - 연속 하락 + 거래량 증가 조합
        """
        if df is None or len(df) < self.lookback + 5:
            return 0.0

        close = df["close"].values
        volume = df["volume"].values if "volume" in df.columns else None

        # 1) 연속 하락 + 거래량 증가 조합 (기계적 매도 특성)
        mech_score = 0.0
        down_with_vol_up = 0
        for i in range(-self.lookback, 0):
            if close[i] < close[i - 1]:
                if volume is not None and len(volume) + i > 0:
                    if volume[i] > volume[i - 1]:
                        down_with_vol_up += 1
        if self.lookback > 0:
            mech_score = min(100.0, (down_with_vol_up / self.lookback) * 200)

        # 2) 최근 하락률 크기
        if len(close) >= self.lookback + 1:
            ret = (close[-1] - close[-self.lookback - 1]) / close[-self.lookback - 1]
            if ret < 0:
                drop_score = min(100.0, abs(ret) * 500)  # -20%→100
            else:
                drop_score = 0.0
        else:
            drop_score = 0.0

        # 3) 거래량 가속도
        if volume is not None and len(volume) >= self.lookback:
            vol_first_half = np.mean(volume[-self.lookback:-self.lookback // 2])
            vol_second_half = np.mean(volume[-self.lookback // 2:])
            if vol_first_half > 0:
                vol_accel = vol_second_half / vol_first_half
                vol_accel_score = min(100.0, max(0.0, (vol_accel - 1) * 100))
            else:
                vol_accel_score = 0.0
        else:
            vol_accel_score = 0.0

        total = (mech_score * 0.40
                 + drop_score * 0.35
                 + vol_accel_score * 0.25)
        return min(100.0, max(0.0, total))


class BubbleDetector:
    """AI 버블 감지 통합 분석기

    조상철 "2026년 11월, 주식시장 대폭락"의 핵심 프레임워크:
    1. 과열 감지 (CAPEX 과잉 → 주가 괴리)
    2. 레버리지 리스크 (GPU 파이낸싱 → 변동성 급증)
    3. 연쇄 매도 (마진콜 → 킬 스위치 → 강제 청산)
    4. 패시브 매도 (ETF 기계적 매도 → 시장 전체 확산)

    임계값 (조상철):
    - ROI Gap < 0.5 → 위험
    - LTV 50% → 마진콜 기준선
    - 만기 집중 → 24개월 주기
    - 서킷브레이커: 7%/13%/20%
    """

    def __init__(self, config=None):
        self.config = config
        self.overheat = OverheatDetector()
        self.cascade = CascadeRiskAnalyzer()
        self.leverage = LeverageRiskMonitor()
        self.passive = PassiveFundCascadeEstimator()

    def analyze(self, df: pd.DataFrame) -> BubbleSignal:
        """통합 버블 분석

        Args:
            df: OHLCV DataFrame (columns: open, high, low, close, volume)

        Returns:
            BubbleSignal: 버블 단계, 점수, 포지션 배수
        """
        if df is None or len(df) < 60:
            return BubbleSignal(
                phase=BubblePhase.NORMAL,
                bubble_score=0.0,
                overheat_score=0.0,
                leverage_risk=0.0,
                cascade_risk=0.0,
                passive_risk=0.0,
                position_multiplier=1.0,
                warning_message="데이터 부족 - 분석 불가",
            )

        # 각 서브모듈 분석
        overheat = self.overheat.analyze(df)
        leverage = self.leverage.analyze(df)
        cascade = self.cascade.analyze(df)
        passive = self.passive.analyze(df)

        # 통합 버블 점수 (가중합산)
        # 과열(25%) + 레버리지(25%) + 연쇄매도(30%) + 패시브(20%)
        bubble_score = (overheat * 0.25
                        + leverage * 0.25
                        + cascade * 0.30
                        + passive * 0.20)

        # 버블 단계 판정
        phase = self._determine_phase(bubble_score, overheat, cascade)

        # 포지션 배수 결정
        position_multiplier = self._calc_position_multiplier(
            phase, bubble_score
        )

        # 경고 메시지
        warning = self._generate_warning(phase, bubble_score,
                                          overheat, cascade)

        signal = BubbleSignal(
            phase=phase,
            bubble_score=round(bubble_score, 1),
            overheat_score=round(overheat, 1),
            leverage_risk=round(leverage, 1),
            cascade_risk=round(cascade, 1),
            passive_risk=round(passive, 1),
            position_multiplier=round(position_multiplier, 2),
            warning_message=warning,
        )

        logger.debug(
            f"BubbleDetector: phase={phase.value}, "
            f"score={bubble_score:.1f}, "
            f"multiplier={position_multiplier:.2f}"
        )
        return signal

    def _determine_phase(self, score: float, overheat: float,
                         cascade: float) -> BubblePhase:
        """버블 단계 판정

        - PANIC: 연쇄매도 70+ AND 전체 70+
        - BURST: 연쇄매도 50+ OR 전체 60+
        - PEAK: 과열 70+ AND 전체 50+
        - EUPHORIA: 과열 50+ OR 전체 35+
        - NORMAL: 그 외
        """
        if cascade >= 70 and score >= 70:
            return BubblePhase.PANIC
        elif cascade >= 50 or score >= 60:
            return BubblePhase.BURST
        elif overheat >= 70 and score >= 50:
            return BubblePhase.PEAK
        elif overheat >= 50 or score >= 35:
            return BubblePhase.EUPHORIA
        else:
            return BubblePhase.NORMAL

    def _calc_position_multiplier(self, phase: BubblePhase,
                                   score: float) -> float:
        """버블 단계별 포지션 배수

        - NORMAL: 1.0 (정상)
        - EUPHORIA: 0.7~0.9 (주의)
        - PEAK: 0.4~0.6 (경계)
        - BURST: 0.2~0.3 (위험 - 매수 차단 권고)
        - PANIC: 0.1 (극위험 - 신규 매수 금지)
        """
        multipliers = {
            BubblePhase.NORMAL: 1.0,
            BubblePhase.EUPHORIA: max(0.7, 1.0 - score / 200),
            BubblePhase.PEAK: max(0.4, 0.7 - score / 200),
            BubblePhase.BURST: max(0.2, 0.4 - score / 300),
            BubblePhase.PANIC: 0.1,
        }
        return multipliers.get(phase, 1.0)

    def _generate_warning(self, phase: BubblePhase, score: float,
                           overheat: float, cascade: float) -> str:
        """경고 메시지 생성"""
        if phase == BubblePhase.PANIC:
            return (f"[PANIC] 패닉 매도 진행 중! 버블점수={score:.0f}, "
                    f"연쇄매도={cascade:.0f}. 신규 매수 금지!")
        elif phase == BubblePhase.BURST:
            return (f"[BURST] 버블 붕괴 진행! 버블점수={score:.0f}. "
                    f"포지션 대폭 축소 권고")
        elif phase == BubblePhase.PEAK:
            return (f"[PEAK] 시장 정점 근접! 과열={overheat:.0f}. "
                    f"신규 매수 자제, 이익실현 검토")
        elif phase == BubblePhase.EUPHORIA:
            return (f"[EUPHORIA] 과열 징후 감지. 과열={overheat:.0f}. "
                    f"포지션 축소 고려")
        else:
            return f"[NORMAL] 정상 시장. 버블점수={score:.0f}"
