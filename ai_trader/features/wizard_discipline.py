"""잭 슈웨거 '주식시장의 마법사들' 핵심 교훈 구현 모듈

64개 마법사 교훈에서 추출한 핵심 트레이딩 원칙:
- 트레이딩 일지 & 규율 추적 (교훈 7, 34)
- 뉴스/이벤트 반응 분석 (교훈 21, 49)
- 기회비용 기반 청산 판단 (교훈 22, 47)
- 확신도 기반 포지션 조절 (교훈 37)
- 지표 시너지 프레임워크 (교훈 54)
- 촉매 검증 진입 필터 (교훈 19)
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from features.indicators import (
    rsi, macd, bollinger_bands, atr, volume_ratio, obv,
)


# ────────────────────────────────────────────────
# dataclass
# ────────────────────────────────────────────────

@dataclass
class WizardSignal:
    """마법사 교훈 통합 시그널"""
    synergy_score: float = 0.0       # 지표 시너지 점수 (-1 ~ +1)
    synergy_strength: str = "NONE"   # STRONG / MODERATE / WEAK / NONE
    confidence: float = 0.5          # 확신도 (0 ~ 1)
    has_catalyst: bool = False       # 촉매 존재 여부
    catalyst_strength: float = 0.0   # 촉매 강도 (0 ~ 1)
    opportunity_cost_ok: bool = True  # 기회비용 적절 여부
    discipline_score: float = 100.0  # 규율 점수 (0 ~ 100)
    news_reaction: float = 0.0      # 뉴스 반응 점수 (-1 ~ +1)
    position_scale: float = 1.0     # 포지션 조절 배수


# ────────────────────────────────────────────────
# 1. TradingJournal (교훈 7, 34)
# ────────────────────────────────────────────────

class TradingJournal:
    """트레이딩 일지 & 규율 추적

    교훈 7: 시장 패턴을 관찰하고 기록하라
    교훈 34: 과거 매매 패턴을 분석하여 개선하라

    매매를 기록하고 패턴을 분석하여 규율 점수를 산출합니다.
    """

    def __init__(self):
        self._trades: list[dict] = []

    def record_trade(
        self,
        symbol: str,
        entry_date: str,
        exit_date: str,
        entry_price: float,
        exit_price: float,
        reason: str = "",
        confidence: float = 0.5,
        result: float = 0.0,
    ) -> None:
        """매매 기록 추가"""
        self._trades.append({
            "symbol": symbol,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "reason": reason,
            "confidence": confidence,
            "result": result,
            "pnl_pct": (
                (exit_price - entry_price) / entry_price * 100
                if entry_price > 0 else 0.0
            ),
        })

    def analyze_patterns(self) -> dict:
        """매매 패턴 분석

        Returns:
            dict: 승률, 평균손익, 연속손실, 고확신 승률 등
        """
        if not self._trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_pnl_pct": 0.0,
                "max_consecutive_loss": 0,
                "high_confidence_win_rate": 0.0,
                "avg_confidence": 0.0,
            }

        trades = self._trades
        total = len(trades)
        wins = sum(1 for t in trades if t["pnl_pct"] > 0)
        win_rate = wins / total * 100 if total > 0 else 0.0
        avg_pnl = sum(t["pnl_pct"] for t in trades) / total

        # 연속 손실 계산
        max_consec_loss = 0
        current_streak = 0
        for t in trades:
            if t["pnl_pct"] <= 0:
                current_streak += 1
                max_consec_loss = max(max_consec_loss, current_streak)
            else:
                current_streak = 0

        # 고확신(>0.7) 매매 승률
        high_conf = [t for t in trades if t["confidence"] >= 0.7]
        high_conf_wr = (
            sum(1 for t in high_conf if t["pnl_pct"] > 0)
            / len(high_conf) * 100
            if high_conf else 0.0
        )

        return {
            "total_trades": total,
            "win_rate": round(win_rate, 1),
            "avg_pnl_pct": round(avg_pnl, 2),
            "max_consecutive_loss": max_consec_loss,
            "high_confidence_win_rate": round(high_conf_wr, 1),
            "avg_confidence": round(
                sum(t["confidence"] for t in trades) / total, 2
            ),
        }

    def get_discipline_score(self) -> float:
        """규율 점수 산출 (0~100)

        기준:
        - 기본 100점에서 감점
        - 연속 3회 이상 손실: -20점
        - 평균 손실폭 > 평균 이익폭: -15점
        - 최근 10건 승률 < 40%: -15점
        - 고확신 매매 승률 < 50%: -10점 (확신과 결과 불일치)
        """
        if len(self._trades) < 3:
            return 100.0  # 데이터 부족 시 기본값

        score = 100.0
        patterns = self.analyze_patterns()

        # 연속 손실 감점
        if patterns["max_consecutive_loss"] >= 5:
            score -= 30.0
        elif patterns["max_consecutive_loss"] >= 3:
            score -= 20.0

        # 손실폭 vs 이익폭
        wins = [t["pnl_pct"] for t in self._trades if t["pnl_pct"] > 0]
        losses = [abs(t["pnl_pct"]) for t in self._trades if t["pnl_pct"] < 0]
        if wins and losses:
            avg_win = sum(wins) / len(wins)
            avg_loss = sum(losses) / len(losses)
            if avg_loss > avg_win:
                score -= 15.0

        # 최근 10건 승률
        recent = self._trades[-10:]
        recent_wr = (
            sum(1 for t in recent if t["pnl_pct"] > 0)
            / len(recent) * 100
        )
        if recent_wr < 40:
            score -= 15.0

        # 고확신 매매 승률
        if patterns["high_confidence_win_rate"] < 50 and len(
            [t for t in self._trades if t["confidence"] >= 0.7]
        ) >= 3:
            score -= 10.0

        return max(score, 0.0)


# ────────────────────────────────────────────────
# 2. NewsReactionAnalyzer (교훈 21, 49)
# ────────────────────────────────────────────────

class NewsReactionAnalyzer:
    """뉴스/이벤트 반응 분석

    교훈 21: 시장 반응이 기대와 다르면 즉시 청산하라
    교훈 49: 주가가 뉴스에 어떻게 반응하는지 주목하라

    호재에 하락 = 강한 매도 시그널
    악재에 상승 = 강한 매수 시그널
    """

    @staticmethod
    def score_news_reaction(
        df: pd.DataFrame,
        event_idx: int,
        lookback: int = 3,
        forward: int = 3,
    ) -> float:
        """이벤트 시점의 가격/거래량 반응 점수화

        Args:
            df: OHLCV DataFrame
            event_idx: 이벤트 발생 행 인덱스
            lookback: 이벤트 전 비교 기간
            forward: 이벤트 후 관찰 기간

        Returns:
            float: -1(악화) ~ +1(호전) 반응 점수
        """
        if event_idx < lookback or event_idx + forward >= len(df):
            return 0.0

        close = df["close"].values
        volume = df["volume"].values

        # 이벤트 전후 가격 변동
        pre_price = close[event_idx - lookback]
        event_price = close[event_idx]
        post_price = close[min(event_idx + forward, len(df) - 1)]

        if pre_price <= 0 or event_price <= 0:
            return 0.0

        # 이벤트 당일 방향 (갭 + 봉)
        event_return = (event_price - pre_price) / pre_price
        # 이벤트 후 방향
        post_return = (post_price - event_price) / event_price

        # 거래량 급증 여부
        avg_vol = np.mean(volume[max(0, event_idx - 20):event_idx])
        event_vol = volume[event_idx]
        vol_surge = (
            event_vol / avg_vol if avg_vol > 0 else 1.0
        )
        vol_weight = min(vol_surge / 2.0, 1.5)  # 최대 1.5배 가중

        # 반응 점수: 후속 움직임이 이벤트와 같은 방향이면 확인
        # 반대 방향이면 기대 갭 (expectation gap)
        if abs(event_return) < 0.001:
            return 0.0

        # 이벤트 방향과 후속 방향이 같으면 +, 반대면 -
        if event_return > 0:
            # 호재(상승) 후 계속 상승 = +, 하락 = -
            score = post_return / abs(event_return)
        else:
            # 악재(하락) 후 계속 하락 = -, 반등 = +
            score = -post_return / abs(event_return)

        # 거래량 가중 적용
        score *= vol_weight

        return float(np.clip(score, -1.0, 1.0))

    @staticmethod
    def detect_expectation_gap(df: pd.DataFrame, lookback: int = 5) -> bool:
        """시장 반응이 기대와 반대인 '기대 갭' 감지

        큰 거래량 이벤트 후 가격이 반대로 움직이면 True.
        ATR 대비 역방향 움직임 > 1.5배 시 감지.

        Args:
            df: OHLCV DataFrame (최소 lookback+3 행 필요)
            lookback: 이벤트 탐색 기간

        Returns:
            bool: 기대 갭 감지 여부
        """
        if len(df) < lookback + 3:
            return False

        close = df["close"].values
        volume = df["volume"].values
        atr_val = atr(df).values

        # 최근 lookback 기간에서 거래량 급증 이벤트 탐색
        recent_start = len(df) - lookback - 1
        avg_vol = np.mean(
            volume[max(0, recent_start - 20):recent_start]
        )

        for i in range(recent_start, len(df) - 2):
            if avg_vol <= 0:
                continue

            # 거래량 2배 이상 급증 이벤트
            if volume[i] < avg_vol * 2.0:
                continue

            # 이벤트 당일 방향
            event_direction = close[i] - close[max(0, i - 1)]
            # 이벤트 후 방향
            post_direction = close[min(i + 2, len(df) - 1)] - close[i]

            # ATR 기준 역방향 움직임
            current_atr = atr_val[i] if atr_val[i] > 0 else abs(event_direction)
            if current_atr <= 0:
                continue

            # 역방향이고 ATR의 1.5배 이상
            if (event_direction > 0 and post_direction < 0
                    and abs(post_direction) > current_atr * 1.5):
                return True
            if (event_direction < 0 and post_direction > 0
                    and abs(post_direction) > current_atr * 1.5):
                return True

        return False


# ────────────────────────────────────────────────
# 3. OpportunityCostEvaluator (교훈 22, 47)
# ────────────────────────────────────────────────

class OpportunityCostEvaluator:
    """기회비용 기반 청산 판단

    교훈 22: 포지션의 기회비용을 고려하라
    교훈 47: 인내심은 중요하지만, 정체된 포지션은 자원 낭비다

    보유 기간 대비 수익이 시장 수익률에 못 미치면 청산 권고.
    """

    def __init__(
        self,
        max_holding_days: int = 20,
        cost_threshold: float = 0.03,
    ):
        self.max_holding_days = max_holding_days
        self.cost_threshold = cost_threshold

    def evaluate(
        self,
        current_return: float,
        holding_days: int,
        market_return: float = 0.0,
    ) -> dict:
        """기회비용 평가

        Args:
            current_return: 현재 보유 수익률 (0.05 = 5%)
            holding_days: 보유 일수
            market_return: 같은 기간 시장(벤치마크) 수익률

        Returns:
            dict: action("HOLD"/"EXIT"), opportunity_cost, score
        """
        opportunity_cost = market_return - current_return

        # 판단 기준
        should_exit = (
            holding_days >= self.max_holding_days
            and opportunity_cost > self.cost_threshold
        )

        # 수익 정체 감지: 보유기간 길고 수익률 극히 낮음
        stagnant = (
            holding_days >= self.max_holding_days
            and abs(current_return) < 0.01  # 1% 미만
        )

        action = "EXIT" if (should_exit or stagnant) else "HOLD"

        return {
            "action": action,
            "opportunity_cost": round(opportunity_cost, 4),
            "holding_days": holding_days,
            "current_return": round(current_return, 4),
            "market_return": round(market_return, 4),
            "is_stagnant": stagnant,
            "score": round(1.0 - min(abs(opportunity_cost) * 10, 1.0), 2),
        }


# ────────────────────────────────────────────────
# 4. ConfidenceScaler (교훈 37)
# ────────────────────────────────────────────────

class ConfidenceScaler:
    """확신도 기반 포지션 조절

    교훈 37: 확신이 높을수록 큰 베팅을 하라
    "5%의 거래가 대부분의 수익을 만든다"

    확신도에 따라 포지션 크기를 0.3배~2.0배 조절합니다.
    """

    # 확신도 구간별 배수
    SCALE_MAP = {
        "VERY_HIGH": 2.0,  # 0.9+
        "HIGH": 1.5,       # 0.7~0.9
        "MEDIUM": 1.0,     # 0.5~0.7
        "LOW": 0.5,        # 0.3~0.5
        "VERY_LOW": 0.3,   # <0.3
    }

    def __init__(
        self,
        max_scale: float = 2.0,
        min_scale: float = 0.3,
    ):
        self.max_scale = max_scale
        self.min_scale = min_scale

    def scale_position(
        self,
        base_size: float,
        confidence_level: float,
    ) -> float:
        """확신도에 따라 포지션 크기를 조절

        Args:
            base_size: 기본 포지션 크기
            confidence_level: 확신도 (0~1)

        Returns:
            float: 조절된 포지션 크기
        """
        confidence_level = max(0.0, min(1.0, confidence_level))

        if confidence_level >= 0.9:
            scale = self.SCALE_MAP["VERY_HIGH"]
        elif confidence_level >= 0.7:
            scale = self.SCALE_MAP["HIGH"]
        elif confidence_level >= 0.5:
            scale = self.SCALE_MAP["MEDIUM"]
        elif confidence_level >= 0.3:
            scale = self.SCALE_MAP["LOW"]
        else:
            scale = self.SCALE_MAP["VERY_LOW"]

        scale = max(self.min_scale, min(self.max_scale, scale))
        return base_size * scale

    @staticmethod
    def calculate_confidence(signals: dict[str, float]) -> float:
        """복수 시그널의 합산 확신도

        Args:
            signals: {'indicator_name': score(-1~+1), ...}

        Returns:
            float: 합산 확신도 (0~1)
        """
        if not signals:
            return 0.5

        values = list(signals.values())
        # 양수(매수) 시그널의 비율과 강도
        positive = [v for v in values if v > 0]
        negative = [v for v in values if v < 0]

        if not positive and not negative:
            return 0.5

        # 방향 일치 비율 (매수 시그널 기준)
        alignment = len(positive) / len(values)
        # 평균 강도
        avg_strength = np.mean([abs(v) for v in values])

        confidence = alignment * 0.6 + avg_strength * 0.4
        return float(np.clip(confidence, 0.0, 1.0))


# ────────────────────────────────────────────────
# 5. IndicatorSynergy (교훈 54)
# ────────────────────────────────────────────────

@dataclass
class SynergySignal:
    """지표 시너지 결과"""
    score: float = 0.0          # 시너지 점수 (-1 ~ +1)
    strength: str = "NONE"      # STRONG / MODERATE / WEAK / NONE
    aligned_count: int = 0      # 방향 일치 지표 수
    total_count: int = 0        # 전체 지표 수
    dominant_direction: str = "NEUTRAL"  # BUY / SELL / NEUTRAL


class IndicatorSynergy:
    """지표 시너지 프레임워크

    교훈 54: 약한 시그널도 결합하면 강한 시그널이 된다

    개별 지표의 매수/매도 방향 일치도를 측정하여
    시그널 강도를 산출합니다.
    """

    # 지표별 가중치 (중요도)
    DEFAULT_WEIGHTS = {
        "rsi": 1.0,
        "macd": 1.2,
        "bb": 0.8,
        "volume": 1.0,
        "obv": 0.7,
        "ema_aligned": 1.3,
        "mfi": 0.8,
        "vwap": 0.9,
    }

    def __init__(self, weights: Optional[dict] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS

    def calculate_synergy(
        self,
        signals: dict[str, float],
    ) -> SynergySignal:
        """지표 시너지 계산

        Args:
            signals: {'indicator': score(-1~+1), ...}
                양수=매수, 음수=매도

        Returns:
            SynergySignal
        """
        if not signals:
            return SynergySignal()

        # 매수/매도 방향 분류
        buy_signals = {k: v for k, v in signals.items() if v > 0.05}
        sell_signals = {k: v for k, v in signals.items() if v < -0.05}
        neutral = {
            k: v for k, v in signals.items()
            if -0.05 <= v <= 0.05
        }

        total = len(signals)

        # 지배적 방향 결정
        if len(buy_signals) > len(sell_signals):
            dominant = "BUY"
            aligned = buy_signals
            opposed = sell_signals
        elif len(sell_signals) > len(buy_signals):
            dominant = "SELL"
            aligned = sell_signals
            opposed = buy_signals
        else:
            dominant = "NEUTRAL"
            aligned = {}
            opposed = {}

        aligned_count = len(aligned)

        # 가중 점수 계산
        weighted_sum = 0.0
        weight_total = 0.0
        for name, val in signals.items():
            w = self.weights.get(name, 1.0)
            weighted_sum += val * w
            weight_total += w

        score = weighted_sum / weight_total if weight_total > 0 else 0.0

        # 반대 방향 시그널 존재 시 감점
        if opposed:
            penalty = len(opposed) / total * 0.3
            if score > 0:
                score -= penalty
            else:
                score += penalty

        score = float(np.clip(score, -1.0, 1.0))

        # 강도 등급
        if aligned_count >= 4:
            strength = "STRONG"
        elif aligned_count >= 3:
            strength = "MODERATE"
        elif aligned_count >= 2:
            strength = "WEAK"
        else:
            strength = "NONE"

        return SynergySignal(
            score=round(score, 3),
            strength=strength,
            aligned_count=aligned_count,
            total_count=total,
            dominant_direction=dominant,
        )

    def get_wizard_signal(self, df: pd.DataFrame) -> dict:
        """DataFrame에서 전체 지표를 분석하여 시너지 판단

        Args:
            df: OHLCV + 지표가 추가된 DataFrame

        Returns:
            dict: synergy_signal, individual_scores
        """
        if len(df) < 20:
            return {
                "synergy": SynergySignal(),
                "scores": {},
            }

        latest = df.iloc[-1]
        scores = {}

        # RSI (30 이하 매수, 70 이상 매도)
        rsi_val = latest.get("rsi", 50)
        if rsi_val < 30:
            scores["rsi"] = 0.8
        elif rsi_val < 40:
            scores["rsi"] = 0.3
        elif rsi_val > 70:
            scores["rsi"] = -0.8
        elif rsi_val > 60:
            scores["rsi"] = -0.3
        else:
            scores["rsi"] = 0.0

        # MACD (히스토그램 방향)
        macd_hist = latest.get("macd_hist", 0)
        prev_hist = df.iloc[-2].get("macd_hist", 0) if len(df) > 1 else 0
        if macd_hist > 0 and macd_hist > prev_hist:
            scores["macd"] = 0.7
        elif macd_hist > 0:
            scores["macd"] = 0.3
        elif macd_hist < 0 and macd_hist < prev_hist:
            scores["macd"] = -0.7
        elif macd_hist < 0:
            scores["macd"] = -0.3
        else:
            scores["macd"] = 0.0

        # 볼린저밴드 (%B 기준)
        bb_pctb = latest.get("bb_pctb", 0.5)
        if bb_pctb < 0.0:
            scores["bb"] = 0.8  # 하단 돌파 = 과매도
        elif bb_pctb < 0.2:
            scores["bb"] = 0.4
        elif bb_pctb > 1.0:
            scores["bb"] = -0.8  # 상단 돌파 = 과매수
        elif bb_pctb > 0.8:
            scores["bb"] = -0.4
        else:
            scores["bb"] = 0.0

        # 거래량 비율
        vol_ratio_val = latest.get("vol_ratio", 1.0)
        if vol_ratio_val > 3.0:
            scores["volume"] = 0.6  # 강한 거래량
        elif vol_ratio_val > 2.0:
            scores["volume"] = 0.3
        elif vol_ratio_val < 0.5:
            scores["volume"] = -0.3
        else:
            scores["volume"] = 0.0

        # OBV 추세
        try:
            obv_val = latest.get("obv", 0)
            obv_prev = df.iloc[-5].get("obv", 0) if len(df) > 5 else obv_val
            if obv_val > obv_prev * 1.05:
                scores["obv"] = 0.5
            elif obv_val < obv_prev * 0.95:
                scores["obv"] = -0.5
            else:
                scores["obv"] = 0.0
        except Exception:
            scores["obv"] = 0.0

        # EMA 정배열
        ema_aligned = latest.get("ema_aligned", False)
        scores["ema_aligned"] = 0.7 if ema_aligned else -0.3

        # MFI
        mfi_val = latest.get("mfi", 50)
        if mfi_val < 20:
            scores["mfi"] = 0.7
        elif mfi_val < 30:
            scores["mfi"] = 0.3
        elif mfi_val > 80:
            scores["mfi"] = -0.7
        elif mfi_val > 70:
            scores["mfi"] = -0.3
        else:
            scores["mfi"] = 0.0

        # VWAP
        vwap_val = latest.get("vwap", 0)
        close_val = latest.get("close", 0)
        if close_val > 0 and vwap_val > 0:
            vwap_pct = (close_val - vwap_val) / vwap_val
            if vwap_pct > 0.01:
                scores["vwap"] = 0.4
            elif vwap_pct < -0.01:
                scores["vwap"] = -0.4
            else:
                scores["vwap"] = 0.0
        else:
            scores["vwap"] = 0.0

        synergy = self.calculate_synergy(scores)

        return {
            "synergy": synergy,
            "scores": scores,
        }


# ────────────────────────────────────────────────
# 6. CatalystVerifier (교훈 19)
# ────────────────────────────────────────────────

class CatalystVerifier:
    """촉매 검증 진입 필터

    교훈 19: 촉매가 있어야 타이밍이 맞다
    "좋은 종목이라도 촉매 없이 진입하면 기회비용이 크다"

    촉매 조건: 거래량 급증, 갭, 볼린저밴드 돌파 등
    """

    def __init__(self, volume_threshold: float = 2.0):
        self.volume_threshold = volume_threshold

    def has_catalyst(
        self,
        df: pd.DataFrame,
        lookback: int = 5,
    ) -> bool:
        """최근 기간 내 촉매 존재 여부 확인

        Args:
            df: OHLCV DataFrame
            lookback: 탐색 기간 (봉 수)

        Returns:
            bool: 촉매 존재 시 True
        """
        if len(df) < lookback + 20:
            return True  # 데이터 부족 시 통과 (안전)

        recent = df.iloc[-lookback:]
        volume = df["volume"].values
        close = df["close"].values

        # 평균 거래량 (최근 20일)
        start = max(0, len(df) - lookback - 20)
        end = len(df) - lookback
        avg_vol = np.mean(volume[start:end]) if end > start else 1.0

        for i in range(len(recent)):
            idx = len(df) - lookback + i

            # 조건 1: 거래량 급증 (평균 대비 threshold 배)
            if avg_vol > 0 and volume[idx] > avg_vol * self.volume_threshold:
                return True

            # 조건 2: 갭 (ATR 기준)
            if idx > 0:
                gap = abs(df["open"].iloc[idx] - close[idx - 1])
                atr_val = atr(df).iloc[idx] if idx < len(df) else 0
                if atr_val > 0 and gap > atr_val * 1.0:
                    return True

            # 조건 3: 볼린저밴드 돌파
            bb_pctb = df.iloc[idx].get("bb_pctb", 0.5) if "bb_pctb" in df.columns else 0.5
            if bb_pctb > 1.0 or bb_pctb < 0.0:
                return True

        return False

    def catalyst_strength(self, df: pd.DataFrame, lookback: int = 5) -> float:
        """촉매 강도 점수 (0~1)

        복수 촉매가 동시에 존재하면 점수 높음.

        Args:
            df: OHLCV DataFrame
            lookback: 탐색 기간

        Returns:
            float: 촉매 강도 (0~1)
        """
        if len(df) < lookback + 20:
            return 0.0

        score = 0.0
        volume = df["volume"].values
        close = df["close"].values

        start = max(0, len(df) - lookback - 20)
        end = len(df) - lookback
        avg_vol = np.mean(volume[start:end]) if end > start else 1.0

        for i in range(lookback):
            idx = len(df) - lookback + i

            # 거래량 급증
            if avg_vol > 0:
                vol_ratio = volume[idx] / avg_vol
                if vol_ratio > self.volume_threshold:
                    score += min(vol_ratio / 5.0, 0.4)

            # 갭
            if idx > 0:
                gap = abs(df["open"].iloc[idx] - close[idx - 1])
                atr_val = atr(df).iloc[idx] if idx < len(df) else 0
                if atr_val > 0:
                    gap_ratio = gap / atr_val
                    if gap_ratio > 1.0:
                        score += min(gap_ratio / 5.0, 0.3)

            # 볼린저밴드 돌파
            if "bb_pctb" in df.columns:
                pctb = df["bb_pctb"].iloc[idx]
                if pctb > 1.0 or pctb < 0.0:
                    score += 0.3

        return float(np.clip(score, 0.0, 1.0))


# ────────────────────────────────────────────────
# 통합 분석 함수
# ────────────────────────────────────────────────

def analyze_wizard_signals(
    df: pd.DataFrame,
    current_return: float = 0.0,
    holding_days: int = 0,
    market_return: float = 0.0,
    journal: Optional[TradingJournal] = None,
) -> WizardSignal:
    """마법사 교훈 통합 분석

    모든 분석기를 실행하여 WizardSignal을 반환합니다.

    Args:
        df: OHLCV + 지표 DataFrame
        current_return: 현재 포지션 수익률 (보유 중인 경우)
        holding_days: 보유 일수
        market_return: 시장 수익률
        journal: 트레이딩 일지 (선택)

    Returns:
        WizardSignal
    """
    result = WizardSignal()

    try:
        # 1. 지표 시너지
        synergy_analyzer = IndicatorSynergy()
        synergy_result = synergy_analyzer.get_wizard_signal(df)
        synergy = synergy_result["synergy"]
        result.synergy_score = synergy.score
        result.synergy_strength = synergy.strength

        # 2. 확신도 계산
        scaler = ConfidenceScaler()
        result.confidence = scaler.calculate_confidence(
            synergy_result["scores"]
        )

        # 3. 촉매 검증
        catalyst = CatalystVerifier()
        result.has_catalyst = catalyst.has_catalyst(df)
        result.catalyst_strength = catalyst.catalyst_strength(df)

        # 4. 기회비용 (보유 중인 경우)
        if holding_days > 0:
            oce = OpportunityCostEvaluator()
            oc_result = oce.evaluate(
                current_return, holding_days, market_return
            )
            result.opportunity_cost_ok = (oc_result["action"] != "EXIT")

        # 5. 규율 점수
        if journal is not None:
            result.discipline_score = journal.get_discipline_score()

        # 6. 뉴스 반응 (최근 이벤트)
        result.news_reaction = 0.0
        try:
            if not NewsReactionAnalyzer.detect_expectation_gap(df):
                result.news_reaction = 0.0  # 정상
            else:
                result.news_reaction = -0.5  # 기대 갭 감지
        except Exception:
            pass

        # 7. 포지션 스케일
        result.position_scale = scaler.scale_position(
            1.0, result.confidence
        )

    except Exception:
        pass  # 오류 시 기본값 유지

    return result
