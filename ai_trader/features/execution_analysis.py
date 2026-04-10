"""매매 집행 분석 보완 모듈 - 박병창 "현명한 당신의 주식투자 교과서" 기반

기존 candle_patterns.py, market_flow.py를 보완하는 추가 모듈:
- CANSLIM 스크리닝 (기술적 프록시)
- PEG 비율 계산
- 시간대별 최적 매매 전략
- 장대봉 분석 (50% 기준선)

핵심 임계값 (박병창):
- 체결강도 > 200: 극강 매수세
- 체결강도 > 100: 매수세 우세
- PEG < 1.0: 저평가 성장주
- 장대봉 조정 < 50%: 정상 조정
- 장대봉 조정 < 30%: 강한 황소
- 거래량비 (아침 1시간) > 50%: 강한 신호
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class CANSLIMScore:
    """CANSLIM 기술적 프록시 점수"""
    c_score: float  # C: Current earnings (최근 수익 모멘텀)
    a_score: float  # A: Annual earnings (연간 수익 추세)
    n_score: float  # N: New highs (신고가 근접도)
    s_score: float  # S: Supply/demand (수급)
    l_score: float  # L: Leader (주도주 여부)
    i_score: float  # I: Institutional (기관 참여도)
    m_score: float  # M: Market direction (시장 방향)
    total: float    # 총합 (0-100)


@dataclass
class ExecutionSignal:
    """매매 집행 분석 시그널"""
    canslim_score: float         # CANSLIM 점수 (0-100)
    peg_ratio: float             # PEG 비율 (< 1.0 저평가)
    optimal_time: str            # 최적 매매 시간대
    candle_force: float          # 장대봉 강도 (0-1)
    candle_force_intact: bool    # 장대봉 50% 기준선 유지 여부
    confidence_adjustment: float  # 신뢰도 조정값 (-0.2 ~ +0.2)


class CANSLIMScreener:
    """CANSLIM 기술적 프록시 스크리너

    박병창 + 윌리엄 오닐 CANSLIM:
    C - Current quarterly EPS → 최근 분기 수익 모멘텀 (가격 상승률)
    A - Annual EPS growth → 연간 수익 추세 (장기 가격 추세)
    N - New product/New high → 52주 신고가 근접도
    S - Supply and demand → 수급 (거래량 분석)
    L - Leader or Laggard → 주도주 판별 (상대 강도)
    I - Institutional sponsorship → 기관 참여도 (거래량 프로파일)
    M - Market direction → 시장 방향 (이평선 배열)

    모든 요소를 가격/거래량 데이터로 기술적 프록시 산출.
    """

    def __init__(self, lookback: int = 252):
        self.lookback = lookback

    def analyze(self, df: pd.DataFrame) -> CANSLIMScore:
        """CANSLIM 기술적 프록시 분석"""
        if df is None or len(df) < self.lookback:
            return CANSLIMScore(0, 0, 0, 0, 0, 0, 0, 0)

        close = df["close"].values
        volume = df["volume"].values if "volume" in df.columns else None
        high = df["high"].values if "high" in df.columns else close

        # C: 최근 분기(63일) 수익률 vs 이전 분기
        c = self._score_current_earnings(close)

        # A: 연간(252일) 추세 일관성
        a = self._score_annual_trend(close)

        # N: 52주 신고가 근접도
        n = self._score_new_high(close, high)

        # S: 수급 (거래량 증가 + 가격 상승 동반)
        s = self._score_supply_demand(close, volume)

        # L: 주도주 (상대 강도 - 자체 모멘텀)
        l = self._score_leadership(close)

        # I: 기관 참여도 (대량 거래 빈도)
        i = self._score_institutional(volume)

        # M: 시장 방향 (이평선 배열)
        m = self._score_market_direction(close)

        # 총합 (각 14.3%)
        total = (c + a + n + s + l + i + m) / 7.0

        return CANSLIMScore(
            c_score=round(c, 1),
            a_score=round(a, 1),
            n_score=round(n, 1),
            s_score=round(s, 1),
            l_score=round(l, 1),
            i_score=round(i, 1),
            m_score=round(m, 1),
            total=round(total, 1),
        )

    def _score_current_earnings(self, close: np.ndarray) -> float:
        """C: 최근 분기(63일) 수익률"""
        if len(close) < 127:
            return 50.0
        ret_q1 = (close[-1] - close[-64]) / close[-64] if close[-64] > 0 else 0
        ret_q2 = (close[-64] - close[-127]) / close[-127] if close[-127] > 0 else 0
        # 최근 분기 수익률이 이전 분기보다 높으면 가산
        if ret_q1 > ret_q2 and ret_q1 > 0:
            return min(100.0, 50 + ret_q1 * 500)
        elif ret_q1 > 0:
            return min(80.0, 40 + ret_q1 * 400)
        else:
            return max(0.0, 50 + ret_q1 * 500)

    def _score_annual_trend(self, close: np.ndarray) -> float:
        """A: 연간 추세 일관성 (4개 분기 모두 상승이면 고점)"""
        if len(close) < 252:
            return 50.0
        quarters_up = 0
        for i in range(4):
            start = -(i + 1) * 63
            end = -i * 63 if i > 0 else None
            q_start = close[start]
            q_end = close[end] if end else close[-1]
            if q_start > 0 and q_end > q_start:
                quarters_up += 1
        return quarters_up * 25.0

    def _score_new_high(self, close: np.ndarray,
                        high: np.ndarray) -> float:
        """N: 52주 신고가 근접도"""
        high_252 = max(high[-min(252, len(high)):])
        if high_252 > 0:
            proximity = close[-1] / high_252 * 100
            return min(100.0, max(0.0, proximity))
        return 50.0

    def _score_supply_demand(self, close: np.ndarray,
                              volume: Optional[np.ndarray]) -> float:
        """S: 수급 (가격 상승 + 거래량 증가 동반)"""
        if volume is None or len(volume) < 20:
            return 50.0
        avg_vol = np.mean(volume[-60:]) if len(volume) >= 60 else np.mean(volume)
        up_vol_days = 0
        total_days = min(20, len(close) - 1)
        for i in range(-total_days, 0):
            if close[i] > close[i - 1] and volume[i] > avg_vol:
                up_vol_days += 1
        return min(100.0, (up_vol_days / total_days) * 200) if total_days > 0 else 50.0

    def _score_leadership(self, close: np.ndarray) -> float:
        """L: 모멘텀 강도 (상대 강도 프록시)"""
        if len(close) < 60:
            return 50.0
        ret_20 = (close[-1] - close[-21]) / close[-21] if close[-21] > 0 else 0
        ret_60 = (close[-1] - close[-61]) / close[-61] if close[-61] > 0 else 0
        # 단기 > 장기 모멘텀이면 주도주 특성
        if ret_20 > ret_60 and ret_20 > 0:
            return min(100.0, 60 + ret_20 * 400)
        elif ret_20 > 0:
            return min(80.0, 40 + ret_20 * 400)
        return max(0.0, 50 + ret_20 * 500)

    def _score_institutional(self, volume: Optional[np.ndarray]) -> float:
        """I: 기관 참여도 (대량 거래 빈도 프록시)"""
        if volume is None or len(volume) < 60:
            return 50.0
        avg_vol = np.mean(volume[-60:])
        if avg_vol <= 0:
            return 50.0
        # 평균의 1.5배 이상 거래량 발생 빈도
        large_vol_days = sum(
            1 for v in volume[-20:] if v > avg_vol * 1.5
        )
        return min(100.0, large_vol_days * 12.5)  # 8일→100

    def _score_market_direction(self, close: np.ndarray) -> float:
        """M: 이평선 배열 (5 > 20 > 60 > 120 정배열)"""
        if len(close) < 120:
            return 50.0
        ma5 = np.mean(close[-5:])
        ma20 = np.mean(close[-20:])
        ma60 = np.mean(close[-60:])
        ma120 = np.mean(close[-120:])

        score = 0.0
        if ma5 > ma20:
            score += 25
        if ma20 > ma60:
            score += 25
        if ma60 > ma120:
            score += 25
        # 현재가 > 모든 이평선
        if close[-1] > max(ma5, ma20, ma60, ma120):
            score += 25
        return score


class PEGCalculator:
    """PEG 비율 계산기

    박병창: PEG = PER / 이익증가율
    - PEG < 1.0: 저평가 성장주
    - PEG = 1.0: 적정 평가
    - PEG > 2.0: 고평가

    기술적 프록시:
    PER 대신 가격/수익 모멘텀 비율 사용
    이익증가율 대신 가격 상승 가속도 사용
    """

    def calculate_peg_proxy(self, df: pd.DataFrame) -> float:
        """PEG 프록시 산출

        PEG proxy = (현재 가격/60일 이평) / (60일 수익률의 가속도)
        낮을수록 저평가 성장주 특성
        """
        if df is None or len(df) < 120:
            return 1.0  # 중립

        close = df["close"].values
        ma60 = np.mean(close[-60:])

        if ma60 <= 0:
            return 1.0

        # "PER" 프록시: 현재가/60일 이평 (1.0 이상 = 프리미엄)
        per_proxy = close[-1] / ma60

        # "이익증가율" 프록시: 최근 60일 수익률 / 이전 60일 수익률
        ret_recent = (close[-1] - close[-61]) / close[-61] if close[-61] > 0 else 0
        ret_prev = (close[-61] - close[-121]) / close[-121] if close[-121] > 0 else 0

        if ret_prev > 0.01:  # 이전 기간 수익 있을 때만
            growth_accel = ret_recent / ret_prev
        elif ret_recent > 0:
            growth_accel = 2.0  # 신규 성장
        else:
            growth_accel = 0.5  # 성장 둔화

        if growth_accel > 0.01:
            peg = per_proxy / growth_accel
        else:
            peg = per_proxy * 3.0  # 성장 없으면 고평가

        return max(0.1, min(5.0, peg))


class TimeBasedStrategy:
    """시간대별 매매 전략

    박병창:
    - 강세장: 아침 10시 전 매수 유리 (갭업 후 추가 상승)
    - 약세장: 오후 2시 후 매수 유리 (오전 투매 후 반등)
    - 장 초반 30분: 전일 수급 연장, 참고만
    - 장 마감 30분: 다음날 방향 결정

    거래량비 (아침 1시간):
    - > 50%: 매우 강한 방향성 (당일 추세 지속)
    - 30-50%: 보통
    - < 30%: 약한 방향성 (방향 변경 가능)
    """

    def get_optimal_time_window(self, df: pd.DataFrame) -> str:
        """현재 시장 상태에 따른 최적 매매 시간대 판단

        Returns:
            "MORNING": 오전 매수 유리 (강세장)
            "AFTERNOON": 오후 매수 유리 (약세장)
            "NEUTRAL": 시간 무관 (횡보장)
        """
        if df is None or len(df) < 20:
            return "NEUTRAL"

        close = df["close"].values

        # 최근 5일 추세로 강세/약세 판단
        ret_5d = (close[-1] - close[-6]) / close[-6] if close[-6] > 0 else 0
        ma20 = np.mean(close[-20:])
        above_ma = close[-1] > ma20

        if ret_5d > 0.02 and above_ma:
            return "MORNING"   # 강세장 → 아침 매수
        elif ret_5d < -0.02 and not above_ma:
            return "AFTERNOON"  # 약세장 → 오후 매수
        else:
            return "NEUTRAL"


class CandleForceAnalyzer:
    """장대봉 분석 (50% 기준선)

    박병창:
    - 장대 양봉 발생 후 조정 시, 장대봉 몸통의 50%까지 조정 = 정상
    - 50% 이하로 내려가면 = 약세 전환 신호
    - 30% 이하 조정 = 매우 강한 황소 (즉시 반등)

    장대봉 기준:
    - 일일 변동폭이 평균의 2배 이상
    - 양봉: 시가 < 종가 (몸통 상승)
    """

    def __init__(self, force_multiplier: float = 2.0):
        self.force_multiplier = force_multiplier

    def detect_candle_force(self, df: pd.DataFrame) -> tuple:
        """최근 장대봉 감지 및 50% 기준선 분석

        Returns:
            (force: float, intact: bool)
            force: 장대봉 강도 (0-1, 0=없음)
            intact: 50% 기준선 유지 여부
        """
        if df is None or len(df) < 30:
            return 0.0, True

        close = df["close"].values
        open_ = df["open"].values if "open" in df.columns else close
        high = df["high"].values if "high" in df.columns else close
        low = df["low"].values if "low" in df.columns else close

        # 평균 봉 크기
        avg_body = np.mean([abs(close[i] - open_[i])
                           for i in range(-20, 0)]) or 1.0

        # 최근 10일 내 장대봉 탐색
        for i in range(-10, 0):
            body = close[i] - open_[i]
            if body > avg_body * self.force_multiplier:
                # 장대 양봉 발견
                candle_mid = open_[i] + body * 0.5  # 50% 기준선
                candle_30 = open_[i] + body * 0.3   # 30% 기준선
                candle_top = close[i]

                # 그 이후 가격이 50% 기준선 유지하는지 확인
                subsequent = close[i + 1:] if i + 1 < 0 else [close[-1]]
                if len(subsequent) > 0:
                    min_after = min(subsequent)
                    force = body / avg_body / self.force_multiplier
                    force = min(1.0, force)

                    if min_after >= candle_mid:
                        # 50% 이상 유지 = 강한 황소
                        if min_after >= candle_top - body * 0.3:
                            # 30% 미만 조정 = 극강
                            return force, True
                        return force, True
                    else:
                        # 50% 이하 조정 = 약세 전환
                        return force, False

        return 0.0, True  # 장대봉 없음


class ExecutionAnalyzer:
    """매매 집행 분석 통합 모듈

    박병창 인사이트 보완 요소:
    1. CANSLIM 스크리닝 (기술적 프록시)
    2. PEG 비율 (가격 기반 프록시)
    3. 시간대별 전략
    4. 장대봉 50% 기준선
    """

    def __init__(self, config=None):
        self.config = config
        self.canslim = CANSLIMScreener()
        self.peg_calc = PEGCalculator()
        self.time_strategy = TimeBasedStrategy()
        self.candle_force = CandleForceAnalyzer()

    def analyze(self, df: pd.DataFrame) -> ExecutionSignal:
        """통합 집행 분석"""
        if df is None or len(df) < 60:
            return ExecutionSignal(
                canslim_score=0.0,
                peg_ratio=1.0,
                optimal_time="NEUTRAL",
                candle_force=0.0,
                candle_force_intact=True,
                confidence_adjustment=0.0,
            )

        # CANSLIM
        canslim = self.canslim.analyze(df)

        # PEG
        peg = self.peg_calc.calculate_peg_proxy(df)

        # 시간대
        optimal_time = self.time_strategy.get_optimal_time_window(df)

        # 장대봉
        force, intact = self.candle_force.detect_candle_force(df)

        # 신뢰도 조정
        adj = self._calc_confidence_adjustment(canslim.total, peg,
                                                force, intact)

        signal = ExecutionSignal(
            canslim_score=canslim.total,
            peg_ratio=round(peg, 2),
            optimal_time=optimal_time,
            candle_force=round(force, 2),
            candle_force_intact=intact,
            confidence_adjustment=round(adj, 2),
        )

        logger.debug(
            f"ExecutionAnalysis: CANSLIM={canslim.total:.1f}, "
            f"PEG={peg:.2f}, time={optimal_time}, "
            f"force={force:.2f}, intact={intact}"
        )
        return signal

    def _calc_confidence_adjustment(self, canslim: float,
                                     peg: float, force: float,
                                     intact: bool) -> float:
        """신뢰도 조정값 산출 (-0.2 ~ +0.2)"""
        adj = 0.0

        # CANSLIM 고점수: +0.05~0.10
        if canslim >= 70:
            adj += 0.10
        elif canslim >= 50:
            adj += 0.05

        # PEG 저평가: +0.05
        if peg < 1.0:
            adj += 0.05
        elif peg > 2.0:
            adj -= 0.05

        # 장대봉 강도
        if force > 0.5 and intact:
            adj += 0.05  # 강한 장대봉 + 기준선 유지
        elif force > 0.5 and not intact:
            adj -= 0.10  # 장대봉 붕괴 = 약세 전환

        return max(-0.20, min(0.20, adj))
