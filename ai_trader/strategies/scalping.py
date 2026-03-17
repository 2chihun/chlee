"""분봉 단타 전략

5분봉 기준 RSI + 볼린저밴드 + MACD + 거래량 복합 시그널
"""

from typing import Optional

import pandas as pd
from loguru import logger

from strategies.base import BaseStrategy, Signal, SignalType
from features.indicators import (
    rsi, bollinger_bands, macd, vwap, volume_ratio, ema, atr,
    add_execution_strength, add_volume_spike,
)


class ScalpingStrategy(BaseStrategy):
    """분봉 단타 전략

    진입 조건 (매수):
      - RSI < 매수 기준 (기본 30) 또는 볼린저밴드 하단 이탈
      - MACD 히스토그램 양전환
      - 거래량 비율 > 최소 기준
      - VWAP 근처 또는 아래

    청산 조건 (매도):
      - RSI > 매도 기준 (기본 70) 또는 볼린저밴드 상단 근접
      - 손절/익절 도달
      - 보유 시간 초과
    """

    DEFAULT_PARAMS = {
        "rsi_period": 14,
        "rsi_buy": 30.0,
        "rsi_sell": 70.0,
        "bb_period": 20,
        "bb_std": 2.0,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "min_volume_ratio": 1.5,
        "stop_loss_pct": -1.0,
        "take_profit_pct": 2.0,
        "max_hold_bars": 60,  # 5분봉 기준 60개 = 5시간
    }

    def __init__(self, params: Optional[dict] = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(name="ScalpingStrategy", params=merged)

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """5분봉 데이터에 기술적 지표를 추가합니다."""
        result = df.copy()
        close = result["close"].astype(float)
        p = self.params

        # RSI
        result["rsi"] = rsi(close, p["rsi_period"])

        # 볼린저밴드
        result["bb_upper"], result["bb_mid"], result["bb_lower"], \
            result["bb_bw"], result["bb_pctb"] = bollinger_bands(
                close, p["bb_period"], p["bb_std"]
            )

        # MACD
        result["macd"], result["macd_signal"], result["macd_hist"] = macd(
            close, p["macd_fast"], p["macd_slow"], p["macd_signal"]
        )

        # VWAP
        result["vwap"] = vwap(result)

        # 거래량 비율
        result["vol_ratio"] = volume_ratio(result)

        # EMA
        result["ema_5"] = ema(close, 5)
        result["ema_20"] = ema(close, 20)

        # ATR (손절/익절 동적 산출용)
        result["atr"] = atr(result, 14)

        # 체결강도 (도서 p.88: 체결강도가 강해져야 한다)
        try:
            result = add_execution_strength(result)
        except Exception:
            result["execution_strength"] = 100.0

        # 거래량 급증 감지
        try:
            result = add_volume_spike(result)
        except Exception:
            result["volume_spike"] = 0

        # 시그널 컬럼
        result["signal"] = SignalType.HOLD.value
        result = self._generate_signals(result)

        return result

    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """전체 DataFrame에 대해 시그널을 계산합니다 (백테스트용)."""
        p = self.params

        # 매수 조건
        buy_rsi = df["rsi"] < p["rsi_buy"]
        buy_bb = df["bb_pctb"] < 0.1  # 볼린저 하단 근접
        buy_macd = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0)  # 양전환
        buy_volume = df["vol_ratio"] > p["min_volume_ratio"]
        buy_vwap = df["close"] <= df["vwap"] * 1.005  # VWAP 근처

        # 체결강도 > 100: 매수세 확인 (도서 p.88)
        buy_exec_strength = df["execution_strength"] > 100 if "execution_strength" in df.columns else True
        # 거래량 급증 감지
        buy_vol_spike = df["volume_spike"] == 1 if "volume_spike" in df.columns else False

        # 매수: (RSI 조건 AND 체결강도 OR BB 조건) AND MACD 양전환 AND (거래량 OR 급증)
        buy_signal = (
            ((buy_rsi & buy_exec_strength) | buy_bb)
            & buy_macd
            & (buy_volume | buy_vol_spike)
        )

        # 매도 조건
        sell_rsi = df["rsi"] > p["rsi_sell"]
        sell_bb = df["bb_pctb"] > 0.95  # 볼린저 상단 근접
        sell_macd = (df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0)  # 음전환

        # 체결강도 < 80 이고 RSI > 60이면 매도 신호 강화
        sell_exec_weak = pd.Series(False, index=df.index)
        if "execution_strength" in df.columns:
            sell_exec_weak = (df["execution_strength"] < 80) & (df["rsi"] > 60)

        sell_signal = sell_rsi | (sell_bb & sell_macd) | sell_exec_weak

        df.loc[buy_signal, "signal"] = SignalType.BUY.value
        df.loc[sell_signal, "signal"] = SignalType.SELL.value

        return df

    def generate_signal(
        self, df: pd.DataFrame, current_position: Optional[dict] = None
    ) -> Signal:
        """가장 최근 봉 기준으로 시그널을 생성합니다."""
        if len(df) < 2:
            return Signal(
                type=SignalType.HOLD, stock_code="", price=0,
                strategy_name=self.name,
            )

        analyzed = self.analyze(df)
        latest = analyzed.iloc[-1]
        prev = analyzed.iloc[-2]
        p = self.params
        price = int(latest["close"])
        stock_code = str(latest.get("stock_code", ""))

        # 포지션이 있는 경우: 청산 조건 검사
        if current_position:
            entry_price = current_position["avg_price"]
            pnl_pct = (price - entry_price) / entry_price * 100

            # 손절
            if pnl_pct <= p["stop_loss_pct"]:
                return Signal(
                    type=SignalType.SELL, stock_code=stock_code, price=price,
                    quantity=current_position["quantity"],
                    confidence=0.95, reason=f"손절: {pnl_pct:.2f}%",
                    strategy_name=self.name,
                )

            # 익절
            if pnl_pct >= p["take_profit_pct"]:
                return Signal(
                    type=SignalType.SELL, stock_code=stock_code, price=price,
                    quantity=current_position["quantity"],
                    confidence=0.9, reason=f"익절: {pnl_pct:.2f}%",
                    strategy_name=self.name,
                )

            # 기술적 매도 시그널
            if latest["signal"] == SignalType.SELL.value:
                return Signal(
                    type=SignalType.SELL, stock_code=stock_code, price=price,
                    quantity=current_position["quantity"],
                    confidence=0.7,
                    reason=f"기술적 매도 (RSI={latest['rsi']:.1f})",
                    strategy_name=self.name,
                )

            return Signal(
                type=SignalType.HOLD, stock_code=stock_code, price=price,
                strategy_name=self.name,
            )

        # 포지션이 없는 경우: 진입 조건 검사
        if latest["signal"] == SignalType.BUY.value:
            # 신뢰도 계산
            confidence = 0.5
            if latest["rsi"] < p["rsi_buy"]:
                confidence += 0.15
            if latest["bb_pctb"] < 0.1:
                confidence += 0.15
            if latest["vol_ratio"] > p["min_volume_ratio"] * 1.5:
                confidence += 0.1
            if latest["macd_hist"] > 0 and prev["macd_hist"] <= 0:
                confidence += 0.1
            # 체결강도 반영 (도서 p.88)
            try:
                if latest.get("execution_strength", 100) > 120:
                    confidence += 0.1
                if latest.get("volume_spike", 0) == 1:
                    confidence += 0.05
            except Exception:
                pass

            # 동적 손절/익절 (ATR 기반)
            atr_val = int(latest["atr"]) if latest["atr"] > 0 else int(price * 0.01)
            stop_loss = price - int(atr_val * 2)
            take_profit = price + int(atr_val * 3)

            return Signal(
                type=SignalType.BUY, stock_code=stock_code, price=price,
                stop_loss=stop_loss, take_profit=take_profit,
                confidence=min(confidence, 1.0),
                reason=(
                    f"매수: RSI={latest['rsi']:.1f}, BB%B={latest['bb_pctb']:.2f}, "
                    f"VR={latest['vol_ratio']:.1f}, "
                    f"체결강도={latest.get('execution_strength', 0):.0f}"
                ),
                strategy_name=self.name,
            )

        return Signal(
            type=SignalType.HOLD, stock_code=stock_code, price=price,
            strategy_name=self.name,
        )
