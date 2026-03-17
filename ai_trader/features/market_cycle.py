"""하워드 막스 마켓 사이클 분석 모듈

"투자와 마켓사이클의 법칙" (Howard Marks) 핵심 이론 구현:
- 마켓 사이클 포지셔닝: 현재 사이클 위치 측정 (0~100점)
- 심리 사이클: 탐욕/공포 지수
- 리스크 포지셔닝: 사이클 위치에 따른 공격/방어 비중
- 평균회귀: 극단적 위치에서 반전 신호
- 확률분포 이동: 사이클 위치에 따른 수익 확률 변화
- 극단 회피: "절대 never, 항상 always" 극단 포지션 차단
- 신용사이클 연동: 신용환경에 따른 포지션 조정
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from features.indicators import rsi, macd, bollinger_bands, atr, volume_ratio


# ---------------------------------------------------------------------------
# 상수 정의
# ---------------------------------------------------------------------------

# 사이클 구간 임계값
EARLY_PHASE_THRESHOLD = 30   # 0~30: 초기 (저점권)
LATE_PHASE_THRESHOLD = 70    # 70~100: 말기 (고점권)

# 심리 극단값 임계값
EXTREME_GREED_THRESHOLD = 75  # 75 이상: 극도의 탐욕
EXTREME_FEAR_THRESHOLD = 25   # 25 이하: 극도의 공포

# 사이클 구간 레이블
PHASE_EARLY = "EARLY"    # 초기: 저점 매집 구간
PHASE_MID = "MID"        # 중기: 추세 진행 구간
PHASE_LATE = "LATE"      # 말기: 고점 분산 구간

# 심리 레이블
SENTIMENT_GREED = "GREED"
SENTIMENT_NEUTRAL = "NEUTRAL"
SENTIMENT_FEAR = "FEAR"

# 리스크 포지셔닝 레이블
POSTURE_AGGRESSIVE = "AGGRESSIVE"   # 공격적 (사이클 저점)
POSTURE_NEUTRAL = "NEUTRAL"         # 중립적
POSTURE_DEFENSIVE = "DEFENSIVE"     # 방어적 (사이클 고점)


# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class CycleSignal:
    """마켓 사이클 분석 결과

    하워드 막스: 사이클 위치를 파악하고 그에 맞는 리스크를 취해야 한다.
    미래는 확률분포로 봐야 한다: 사이클 위치가 그 분포를 이동시킨다.
    """
    cycle_score: float       # 0~100 (0=최저점, 100=최고점)
    phase: str               # EARLY / MID / LATE
    sentiment: str           # GREED / NEUTRAL / FEAR
    risk_posture: str        # AGGRESSIVE / NEUTRAL / DEFENSIVE
    max_position_pct: float  # 최대 포지션 비율 (0.0~1.0)
    turning_point: bool      # 전환점 감지 여부
    confidence: float        # 신뢰도 (0.0~1.0)
    note: str                # 분석 메모
    # 확률분포 이동 관련 (하워드 막스 핵심 개념)
    profit_probability: float = 0.5   # 수익 확률 추정 (0.0~1.0)
    extreme_avoided: bool = False     # 극단 포지션 회피 여부


# ---------------------------------------------------------------------------
# 분석기 클래스
# ---------------------------------------------------------------------------

class MarketCycleAnalyzer:
    """하워드 막스 마켓 사이클 분석기

    "투자와 마켓사이클의 법칙" 핵심 개념:
    - 사이클은 반복되며, 극단에서는 반드시 되돌아온다 (평균회귀)
    - 사이클 고점 → 리스크 축소, 방어적 포지션
    - 사이클 저점 → 리스크 확대, 공격적 포지션
    - 심리(탐욕/공포)가 가격보다 더 과도하게 움직인다
    """

    def __init__(
        self,
        lookback_days: int = 252,
        early_phase_max_pos: float = 0.8,
        mid_phase_max_pos: float = 0.5,
        late_phase_max_pos: float = 0.2,
    ):
        """
        Args:
            lookback_days: 사이클 측정 기준 기간 (기본 252 거래일 = 1년)
            early_phase_max_pos: EARLY 구간 최대 포지션 비율
            mid_phase_max_pos: MID 구간 최대 포지션 비율
            late_phase_max_pos: LATE 구간 최대 포지션 비율
        """
        self.lookback_days = lookback_days
        self.early_phase_max_pos = early_phase_max_pos
        self.mid_phase_max_pos = mid_phase_max_pos
        self.late_phase_max_pos = late_phase_max_pos

    def analyze(self, df: pd.DataFrame) -> CycleSignal:
        """전체 사이클 분석을 수행하고 CycleSignal을 반환합니다.

        Args:
            df: OHLCV DataFrame ('open','high','low','close','volume' 필요)

        Returns:
            CycleSignal: 사이클 위치, 심리, 리스크 포지셔닝 종합 결과
        """
        try:
            # 최소 데이터 검증
            if len(df) < 20:
                return self._default_signal("데이터 부족 (최소 20봉 필요)")

            # 사이클 위치 계산
            cycle_info = self.get_cycle_position(df)
            cycle_score = cycle_info["score"]
            cycle_phase = cycle_info["phase"]
            cycle_confidence = cycle_info["confidence"]

            # 심리 지수 계산
            sentiment_info = self.get_sentiment_index(df)
            sentiment = sentiment_info["sentiment"]
            is_extreme = sentiment_info["extreme"]

            # 리스크 포지셔닝
            risk_info = self.get_risk_posture(cycle_score, sentiment)
            posture = risk_info["posture"]
            max_pos_pct = risk_info["max_position_pct"]

            # 전환점 감지
            turning_info = self.detect_cycle_turning_point(df)
            turning_point = turning_info["signal"]

            # 최종 신뢰도: 사이클 신뢰도 + 극단 심리 보정
            confidence = cycle_confidence
            if is_extreme:
                confidence = min(confidence + 0.1, 1.0)

            # 확률분포 이동 계산
            # 하워드 막스: 사이클 위치가 수익 확률분포를 이동시킨다.
            # EARLY(저점권) → 확률분포 우측 이동 (수익 확률 증가)
            # LATE(고점권) → 확률분포 좌측 이동 (손실 확률 증가)
            profit_probability = self._calc_profit_probability(
                cycle_score, sentiment
            )

            # 극단 회피 로직
            # 하워드 막스: "절대 never, 항상 always"가 아닌 확률적 접근
            # 극단적 고점/저점에서 포지션을 0 또는 100%로 만들지 않는다.
            max_pos_pct, extreme_avoided = self._avoid_extremes(
                max_pos_pct, cycle_score, sentiment
            )

            # 비고 메모 작성
            note = self._build_note(
                cycle_score, cycle_phase, sentiment, posture,
                turning_info, cycle_info["signals"]
            )

            return CycleSignal(
                cycle_score=cycle_score,
                phase=cycle_phase,
                sentiment=sentiment,
                risk_posture=posture,
                max_position_pct=max_pos_pct,
                turning_point=turning_point,
                confidence=confidence,
                note=note,
                profit_probability=profit_probability,
                extreme_avoided=extreme_avoided,
            )

        except Exception as exc:
            return self._default_signal(f"분석 오류: {exc}")

    def get_cycle_position(self, df: pd.DataFrame) -> dict:
        """사이클 위치 점수를 계산합니다 (0~100).

        하워드 막스: 사이클 위치를 파악하는 것이 최우선이다.
        여러 지표를 종합하여 현재 시장이 사이클의 어느 위치에 있는지 측정.

        구성 지표 (각 0~100점):
          1. 52주 고저 위치 (가격 위치) - 가중치 30%
          2. RSI 위치               - 가중치 25%
          3. 거래량 트렌드           - 가중치 15%
          4. 변동성(ATR) 수준        - 가중치 15%
          5. MACD 다이버전스         - 가중치 15%

        Returns:
            dict: {score, phase, confidence, signals}
        """
        signals = {}
        weights = {
            "price_position": 0.30,
            "rsi_position": 0.25,
            "volume_trend": 0.15,
            "volatility": 0.15,
            "macd_divergence": 0.15,
        }

        close = df["close"].astype(float)
        lookback = min(self.lookback_days, len(df))

        # ------------------------------------------------------------------
        # 1. 52주 고저 위치 (가격 포지셔닝)
        # 설명: 현재 가격이 지난 1년 고저 범위에서 어디에 있는지
        # ------------------------------------------------------------------
        try:
            period_high = df["high"].iloc[-lookback:].max()
            period_low = df["low"].iloc[-lookback:].min()
            current_price = float(close.iloc[-1])
            denom = period_high - period_low
            if denom > 0:
                price_score = (current_price - period_low) / denom * 100
            else:
                price_score = 50.0
            signals["price_position"] = round(price_score, 1)
        except Exception:
            signals["price_position"] = 50.0

        # ------------------------------------------------------------------
        # 2. RSI 위치
        # 설명: RSI 값 자체를 사이클 점수로 활용 (과매수=고점권, 과매도=저점권)
        # ------------------------------------------------------------------
        try:
            rsi_series = rsi(close, 14)
            rsi_val = float(rsi_series.iloc[-1])
            if np.isnan(rsi_val):
                rsi_val = 50.0
            signals["rsi_position"] = round(rsi_val, 1)
        except Exception:
            signals["rsi_position"] = 50.0

        # ------------------------------------------------------------------
        # 3. 거래량 트렌드
        # 설명: 거래량 증가 추세는 상승 사이클 후반을 시사
        # 단기 평균 거래량 > 장기 평균 거래량 → 고점 신호
        # ------------------------------------------------------------------
        try:
            vol = df["volume"].astype(float)
            short_vol = float(vol.iloc[-20:].mean())
            long_vol = float(vol.iloc[-lookback:].mean())
            if long_vol > 0:
                vol_ratio_val = short_vol / long_vol
                # 2배 이상 → 100점 (과열), 0.5배 이하 → 0점 (침체)
                vol_score = np.clip((vol_ratio_val - 0.5) / 1.5 * 100, 0, 100)
            else:
                vol_score = 50.0
            signals["volume_trend"] = round(float(vol_score), 1)
        except Exception:
            signals["volume_trend"] = 50.0

        # ------------------------------------------------------------------
        # 4. 변동성(ATR) 수준
        # 설명: 변동성 급증은 사이클 전환점(고점/저점)에서 발생
        # 단기 ATR > 장기 ATR → 극단 구간 (고점 또는 저점)
        # 여기서는 가격 위치와 조합하여 고점 과열 여부 판단
        # ------------------------------------------------------------------
        try:
            atr_series = atr(df, 14)
            atr_short = float(atr_series.iloc[-5:].mean())
            atr_long = float(atr_series.iloc[-lookback:].mean())
            if atr_long > 0:
                atr_ratio = atr_short / atr_long
                # 변동성 높고 가격도 고점 → 고점 과열 신호 (점수 높음)
                # 변동성 높고 가격 저점 → 공포 저점 신호 (점수 낮음)
                price_in_high = signals["price_position"] > 60
                if price_in_high:
                    # 고점 + 고변동성 = 사이클 말기
                    vol_atm_score = np.clip(atr_ratio * 50, 0, 100)
                else:
                    # 저점 + 고변동성 = 사이클 초기 (공포 저점)
                    vol_atm_score = np.clip(100 - atr_ratio * 50, 0, 100)
            else:
                vol_atm_score = 50.0
            signals["volatility"] = round(float(vol_atm_score), 1)
        except Exception:
            signals["volatility"] = 50.0

        # ------------------------------------------------------------------
        # 5. MACD 다이버전스
        # 설명: 가격 신고가인데 MACD 히스토그램이 감소 → 고점 신호 (점수 높음)
        #       가격 신저가인데 MACD 히스토그램이 증가 → 저점 신호 (점수 낮음)
        # ------------------------------------------------------------------
        try:
            _, _, hist = macd(close, 12, 26, 9)
            hist_recent = float(hist.iloc[-1])
            hist_prev = float(hist.iloc[-5:].mean())

            price_near_high = signals["price_position"] > 65
            price_near_low = signals["price_position"] < 35

            if price_near_high and hist_recent < hist_prev:
                # 가격 고점 + MACD 약화 = 사이클 고점 신호
                macd_score = 80.0
            elif price_near_low and hist_recent > hist_prev:
                # 가격 저점 + MACD 강화 = 사이클 저점 신호
                macd_score = 20.0
            else:
                # 중립: RSI 위치 기준으로 보정
                macd_score = signals["rsi_position"]

            signals["macd_divergence"] = round(macd_score, 1)
        except Exception:
            signals["macd_divergence"] = 50.0

        # ------------------------------------------------------------------
        # 최종 사이클 점수 (가중 평균)
        # ------------------------------------------------------------------
        score = sum(signals[k] * weights[k] for k in weights)
        score = round(np.clip(score, 0, 100), 1)

        # 사이클 구간 판단
        if score <= EARLY_PHASE_THRESHOLD:
            phase = PHASE_EARLY
        elif score >= LATE_PHASE_THRESHOLD:
            phase = PHASE_LATE
        else:
            phase = PHASE_MID

        # 신뢰도: 지표 간 일관성이 높을수록 높음
        score_values = list(signals.values())
        consistency = 1.0 - (float(np.std(score_values)) / 50.0)
        confidence = round(np.clip(consistency, 0.3, 1.0), 2)

        return {
            "score": score,
            "phase": phase,
            "confidence": confidence,
            "signals": signals,
        }

    def get_sentiment_index(self, df: pd.DataFrame) -> dict:
        """심리 지수를 계산합니다 (탐욕/공포).

        하워드 막스: "심리는 가격보다 더 극단으로 움직인다.
        탐욕이 극도에 달하면 팔고, 공포가 극도에 달하면 사라."

        구성 요소:
          - 볼린저밴드 위치 (과열/침체)
          - RSI 극단값 빈도 (과매수/과매도 발생 빈도)
          - 거래량 이상 급증 횟수
          - 고가/저가 돌파 빈도

        Returns:
            dict: {greed_score, fear_score, sentiment, extreme}
        """
        close = df["close"].astype(float)
        window = min(20, len(df))

        greed_components = []
        fear_components = []

        # ------------------------------------------------------------------
        # 1. 볼린저밴드 위치 (%B)
        # %B > 1.0: 상단 돌파 (탐욕), %B < 0.0: 하단 돌파 (공포)
        # ------------------------------------------------------------------
        try:
            _, _, _, _, pct_b = bollinger_bands(close, 20, 2.0)
            recent_pctb = pct_b.iloc[-window:]
            # 상단 초과 비율 → 탐욕 기여
            greed_from_bb = float((recent_pctb > 0.8).mean() * 100)
            # 하단 미달 비율 → 공포 기여
            fear_from_bb = float((recent_pctb < 0.2).mean() * 100)
            greed_components.append(greed_from_bb)
            fear_components.append(fear_from_bb)
        except Exception:
            greed_components.append(0.0)
            fear_components.append(0.0)

        # ------------------------------------------------------------------
        # 2. RSI 극단값 빈도
        # RSI > 70 빈도 → 탐욕, RSI < 30 빈도 → 공포
        # ------------------------------------------------------------------
        try:
            rsi_series = rsi(close, 14)
            recent_rsi = rsi_series.iloc[-window:]
            greed_from_rsi = float((recent_rsi > 70).mean() * 100)
            fear_from_rsi = float((recent_rsi < 30).mean() * 100)
            greed_components.append(greed_from_rsi)
            fear_components.append(fear_from_rsi)
        except Exception:
            greed_components.append(0.0)
            fear_components.append(0.0)

        # ------------------------------------------------------------------
        # 3. 거래량 이상 급증 횟수 (평균 2배 이상)
        # 거래량 급증은 과열 또는 공황을 반영
        # 가격 상승 중 급증 → 탐욕, 가격 하락 중 급증 → 공포
        # ------------------------------------------------------------------
        try:
            vol = df["volume"].astype(float)
            avg_vol = vol.rolling(window=20, min_periods=1).mean()
            vol_spike = vol > avg_vol * 2.0
            recent_spikes = vol_spike.iloc[-window:]
            price_up = close.diff() > 0

            spike_greed = float(
                (vol_spike & price_up).iloc[-window:].mean() * 100
            )
            spike_fear = float(
                (vol_spike & ~price_up).iloc[-window:].mean() * 100
            )
            greed_components.append(spike_greed)
            fear_components.append(spike_fear)
        except Exception:
            greed_components.append(0.0)
            fear_components.append(0.0)

        # ------------------------------------------------------------------
        # 4. 고가/저가 돌파 빈도 (52주 기준)
        # 신고가 돌파 빈도 → 탐욕, 신저가 돌파 빈도 → 공포
        # ------------------------------------------------------------------
        try:
            lookback = min(self.lookback_days, len(df))
            period_high = df["high"].iloc[-lookback:].expanding().max()
            period_low = df["low"].iloc[-lookback:].expanding().min()
            new_high = df["high"] >= period_high
            new_low = df["low"] <= period_low

            greed_from_breakout = float(new_high.iloc[-window:].mean() * 100)
            fear_from_breakout = float(new_low.iloc[-window:].mean() * 100)
            greed_components.append(greed_from_breakout)
            fear_components.append(fear_from_breakout)
        except Exception:
            greed_components.append(0.0)
            fear_components.append(0.0)

        # ------------------------------------------------------------------
        # 심리 점수 계산
        # ------------------------------------------------------------------
        greed_score = round(float(np.mean(greed_components)), 1)
        fear_score = round(float(np.mean(fear_components)), 1)

        # 순 심리 점수 (탐욕 - 공포, -100~+100 → 0~100 정규화)
        net_score = np.clip((greed_score - fear_score + 100) / 2, 0, 100)

        if net_score >= EXTREME_GREED_THRESHOLD:
            sentiment = SENTIMENT_GREED
        elif net_score <= EXTREME_FEAR_THRESHOLD:
            sentiment = SENTIMENT_FEAR
        else:
            sentiment = SENTIMENT_NEUTRAL

        # 극단적 심리 여부
        is_extreme = (
            net_score >= EXTREME_GREED_THRESHOLD
            or net_score <= EXTREME_FEAR_THRESHOLD
        )

        return {
            "greed_score": greed_score,
            "fear_score": fear_score,
            "net_score": round(float(net_score), 1),
            "sentiment": sentiment,
            "extreme": is_extreme,
        }

    def get_risk_posture(
        self, cycle_score: float, sentiment: str
    ) -> dict:
        """사이클 위치 기반 리스크 포지셔닝을 결정합니다.

        하워드 막스 핵심 원칙:
          - 사이클 고점(LATE): 방어적, 포지션 축소
          - 사이클 저점(EARLY): 공격적, 포지션 확대
          - 심리가 극단일 때 반대로 행동하라

        Args:
            cycle_score: 사이클 위치 (0~100)
            sentiment: 심리 레이블 (GREED/NEUTRAL/FEAR)

        Returns:
            dict: {posture, max_position_pct, aggressive, note}
        """
        # 기본 사이클 위치별 포지셔닝
        if cycle_score >= LATE_PHASE_THRESHOLD:
            posture = POSTURE_DEFENSIVE
            max_pos = self.late_phase_max_pos
            note = "사이클 말기: 방어적 포지션, 신규 매수 자제"
        elif cycle_score <= EARLY_PHASE_THRESHOLD:
            posture = POSTURE_AGGRESSIVE
            max_pos = self.early_phase_max_pos
            note = "사이클 초기: 공격적 포지션, 분할 매수 기회"
        else:
            posture = POSTURE_NEUTRAL
            max_pos = self.mid_phase_max_pos
            note = "사이클 중기: 중립적 포지션, 추세 추종"

        # 심리 극단값 보정 (탐욕 극단 → 더 방어적, 공포 극단 → 더 공격적)
        aggressive = False
        if sentiment == SENTIMENT_GREED:
            # 탐욕 극단: 포지션 한도 추가 축소 (-10%)
            max_pos = max(max_pos - 0.1, 0.1)
            note += " | 극단 탐욕 감지: 추가 방어"
        elif sentiment == SENTIMENT_FEAR:
            # 공포 극단: 포지션 한도 추가 확대 (+10%, 단 0.9 상한)
            max_pos = min(max_pos + 0.1, 0.9)
            aggressive = True
            note += " | 극단 공포 감지: 매수 기회 탐색"

        return {
            "posture": posture,
            "max_position_pct": round(max_pos, 2),
            "aggressive": aggressive,
            "note": note,
        }

    def detect_cycle_turning_point(self, df: pd.DataFrame) -> dict:
        """사이클 전환점을 감지합니다.

        하워드 막스: "극단에서 벗어나는 첫 신호를 잡아라."
        다이버전스와 극단 심리 반전을 조합하여 전환점을 탐지.

        감지 방법:
          1. 가격+거래량 다이버전스 (상승 다이버전스 / 하락 다이버전스)
          2. RSI 극단에서 반전 신호
          3. MACD 히스토그램 전환

        Returns:
            dict: {signal, strength, direction}
              signal: 전환점 감지 여부 (bool)
              strength: 신호 강도 (0.0~1.0)
              direction: 방향 ("UP"=저점 반전, "DOWN"=고점 반전, "NONE")
        """
        if len(df) < 15:
            return {"signal": False, "strength": 0.0, "direction": "NONE"}

        try:
            close = df["close"].astype(float)
            vol = df["volume"].astype(float)
            signals_detected = []

            # ----------------------------------------------------------
            # 신호 1: RSI 다이버전스 (최근 5봉 기준)
            # 가격 신저가 + RSI 저점 상승 → 상승 반전 (UP)
            # 가격 신고가 + RSI 고점 하락 → 하락 반전 (DOWN)
            # ----------------------------------------------------------
            rsi_direction = "NONE"
            try:
                rsi_series = rsi(close, 14)
                n = 5
                price_now = float(close.iloc[-1])
                price_prev = float(close.iloc[-(n + 1)])
                rsi_now = float(rsi_series.iloc[-1])
                rsi_prev = float(rsi_series.iloc[-(n + 1)])

                if (price_now < price_prev) and (rsi_now > rsi_prev):
                    # 가격 하락인데 RSI 상승 → 하락세 약화 = 상승 전환 신호
                    rsi_direction = "UP"
                    signals_detected.append(("rsi_divergence_up", 0.4))
                elif (price_now > price_prev) and (rsi_now < rsi_prev):
                    # 가격 상승인데 RSI 하락 → 상승세 약화 = 하락 전환 신호
                    rsi_direction = "DOWN"
                    signals_detected.append(("rsi_divergence_down", 0.4))
            except Exception:
                pass

            # ----------------------------------------------------------
            # 신호 2: 거래량 다이버전스
            # 가격 하락 + 거래량 감소 → 매도 세력 약화 → 상승 전환 가능
            # 가격 상승 + 거래량 감소 → 매수 세력 약화 → 하락 전환 가능
            # ----------------------------------------------------------
            vol_direction = "NONE"
            try:
                n = 5
                price_trend = float(close.iloc[-1]) - float(close.iloc[-(n + 1)])
                vol_trend = float(vol.iloc[-n:].mean()) - float(
                    vol.iloc[-(2 * n):-n].mean()
                )

                if price_trend < 0 and vol_trend < 0:
                    # 가격 하락 + 거래량 감소 → 하락 소진 = 상승 전환 가능
                    vol_direction = "UP"
                    signals_detected.append(("volume_divergence_up", 0.3))
                elif price_trend > 0 and vol_trend < 0:
                    # 가격 상승 + 거래량 감소 → 상승 소진 = 하락 전환 가능
                    vol_direction = "DOWN"
                    signals_detected.append(("volume_divergence_down", 0.3))
            except Exception:
                pass

            # ----------------------------------------------------------
            # 신호 3: MACD 히스토그램 전환 (0선 돌파)
            # 히스토그램이 음(-) → 양(+)으로 전환: 상승 전환
            # 히스토그램이 양(+) → 음(-)으로 전환: 하락 전환
            # ----------------------------------------------------------
            macd_direction = "NONE"
            try:
                _, _, hist = macd(close, 12, 26, 9)
                hist_now = float(hist.iloc[-1])
                hist_prev = float(hist.iloc[-2])

                if hist_prev < 0 and hist_now >= 0:
                    macd_direction = "UP"
                    signals_detected.append(("macd_cross_up", 0.3))
                elif hist_prev > 0 and hist_now <= 0:
                    macd_direction = "DOWN"
                    signals_detected.append(("macd_cross_down", 0.3))
            except Exception:
                pass

            # ----------------------------------------------------------
            # 결과 종합
            # ----------------------------------------------------------
            if not signals_detected:
                return {"signal": False, "strength": 0.0, "direction": "NONE"}

            # 방향 투표 (동일 방향 신호 합산)
            up_strength = sum(
                s for _, s in signals_detected if "up" in _
            )
            down_strength = sum(
                s for _, s in signals_detected if "down" in _
            )

            if up_strength >= 0.4 or down_strength >= 0.4:
                detected = True
                if up_strength >= down_strength:
                    direction = "UP"
                    strength = up_strength
                else:
                    direction = "DOWN"
                    strength = down_strength
                strength = round(min(strength, 1.0), 2)
            else:
                detected = False
                direction = "NONE"
                strength = 0.0

            return {
                "signal": detected,
                "strength": strength,
                "direction": direction,
            }

        except Exception as exc:
            return {"signal": False, "strength": 0.0, "direction": "NONE"}

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _calc_profit_probability(
        self, cycle_score: float, sentiment: str
    ) -> float:
        """수익 확률을 추정합니다.

        하워드 막스: 미래는 확률분포로 봐야 한다.
        사이클 위치가 그 분포(볼풀 속 공의 비율)를 바꾼다.

        - 사이클 저점(EARLY) + 극단 공포 → 수익 확률 최대 (~0.70)
        - 사이클 중간(MID) → 수익 확률 기본 (~0.50)
        - 사이클 고점(LATE) + 극단 탐욕 → 수익 확률 최소 (~0.30)

        Args:
            cycle_score: 사이클 점수 (0~100)
            sentiment: 심리 레이블

        Returns:
            float: 수익 확률 추정 (0.0~1.0)
        """
        try:
            # 기본 확률: 사이클 점수 역방향 매핑
            # score 0 → prob 0.70, score 50 → prob 0.50, score 100 → prob 0.30
            base_prob = 0.70 - (cycle_score / 100.0) * 0.40

            # 심리 보정
            if sentiment == SENTIMENT_FEAR:
                # 극단 공포 = 더 유리한 진입 → 확률 상향
                base_prob = min(base_prob + 0.05, 0.80)
            elif sentiment == SENTIMENT_GREED:
                # 극단 탐욕 = 더 불리한 진입 → 확률 하향
                base_prob = max(base_prob - 0.05, 0.20)

            return round(float(np.clip(base_prob, 0.0, 1.0)), 3)
        except Exception:
            return 0.5

    def _avoid_extremes(
        self,
        max_pos_pct: float,
        cycle_score: float,
        sentiment: str,
    ) -> tuple:
        """극단 포지션을 회피합니다.

        하워드 막스: 투자에서 "절대(never)" 또는 "항상(always)"은 없다.
        아무리 유리한 사이클 위치라도 100% 투자는 금물이다.
        아무리 불리한 사이클 위치라도 0% 포지션(완전 철수)도 극단이다.

        규칙:
          - 최대 포지션 상한: 0.9 (10% 현금 유보 최소 요건)
          - 최소 포지션 하한: 0.1 (완전 철수 방지)
          - 극단 탐욕(LATE+GREED): 0.15 상한 적용
          - 극단 공포(EARLY+FEAR): 0.75 상한 적용 (분할 매수)

        Args:
            max_pos_pct: 현재 최대 포지션 비율
            cycle_score: 사이클 점수
            sentiment: 심리 레이블

        Returns:
            tuple: (조정된 max_pos_pct, 극단회피 여부)
        """
        try:
            original = max_pos_pct
            extreme_avoided = False

            # 절대 상한: 아무리 좋은 상황도 90% 이상 투자 금지
            # (하워드 막스: 언제나 최악의 시나리오를 대비해야 한다)
            if max_pos_pct > 0.9:
                max_pos_pct = 0.9
                extreme_avoided = True

            # 절대 하한: 아무리 나쁜 상황도 10% 미만 철수 금지
            # (극단 공포 구간은 오히려 기회이므로 완전 철수는 실수)
            if max_pos_pct < 0.1:
                max_pos_pct = 0.1
                extreme_avoided = True

            # 극단 탐욕 + LATE 사이클: 추가 상한 (15%)
            # "모두가 낙관적일 때가 가장 위험하다"
            is_extreme_greed = (
                cycle_score >= 80 and sentiment == SENTIMENT_GREED
            )
            if is_extreme_greed and max_pos_pct > 0.15:
                max_pos_pct = 0.15
                extreme_avoided = True

            # 극단 공포 + EARLY 사이클: 상한 적용 (75%)
            # "모두가 비관적일 때 분할 매수하되, 한꺼번에 몰아넣지 않는다"
            is_extreme_fear = (
                cycle_score <= 20 and sentiment == SENTIMENT_FEAR
            )
            if is_extreme_fear and max_pos_pct > 0.75:
                max_pos_pct = 0.75
                extreme_avoided = True

            if not extreme_avoided and abs(max_pos_pct - original) > 0.001:
                extreme_avoided = True

            return round(max_pos_pct, 2), extreme_avoided

        except Exception:
            return max_pos_pct, False

    def _default_signal(self, note: str) -> CycleSignal:
        """기본(중립) CycleSignal을 반환합니다."""
        return CycleSignal(
            cycle_score=50.0,
            phase=PHASE_MID,
            sentiment=SENTIMENT_NEUTRAL,
            risk_posture=POSTURE_NEUTRAL,
            max_position_pct=self.mid_phase_max_pos,
            turning_point=False,
            confidence=0.3,
            note=note,
            profit_probability=0.5,
            extreme_avoided=False,
        )

    def _build_note(
        self,
        score: float,
        phase: str,
        sentiment: str,
        posture: str,
        turning_info: dict,
        signals: dict,
    ) -> str:
        """분석 결과 요약 메모를 생성합니다."""
        parts = [
            f"사이클점수={score:.1f}",
            f"구간={phase}",
            f"심리={sentiment}",
            f"포지셔닝={posture}",
        ]
        if turning_info["signal"]:
            parts.append(
                f"전환점감지({turning_info['direction']}"
                f" 강도={turning_info['strength']:.2f})"
            )
        # 주요 신호 값
        sig_str = " ".join(
            f"{k[:4]}={v:.0f}" for k, v in signals.items()
        )
        parts.append(f"[{sig_str}]")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# 모듈 수준 편의 함수
# ---------------------------------------------------------------------------

_default_analyzer = MarketCycleAnalyzer()


def analyze_cycle(df: pd.DataFrame) -> CycleSignal:
    """기본 설정으로 사이클 분석을 수행합니다.

    Args:
        df: OHLCV DataFrame

    Returns:
        CycleSignal
    """
    return _default_analyzer.analyze(df)


def get_cycle_position(df: pd.DataFrame) -> dict:
    """사이클 위치 점수를 반환합니다."""
    return _default_analyzer.get_cycle_position(df)


def get_sentiment_index(df: pd.DataFrame) -> dict:
    """심리 지수를 반환합니다."""
    return _default_analyzer.get_sentiment_index(df)


def get_risk_posture(cycle_score: float, sentiment: str) -> dict:
    """리스크 포지셔닝을 반환합니다."""
    return _default_analyzer.get_risk_posture(cycle_score, sentiment)


def detect_cycle_turning_point(df: pd.DataFrame) -> dict:
    """사이클 전환점을 반환합니다."""
    return _default_analyzer.detect_cycle_turning_point(df)
