"""일중 스윙 전략

갭 분석 + 수급 + 이동평균선 배열 + 거래량 기반의 일중 스윙 매매
"""

from typing import Optional

import pandas as pd
from loguru import logger

from strategies.base import BaseStrategy, Signal, SignalType
from features.indicators import (
    sma, ema, rsi, macd, bollinger_bands, vwap,
    volume_ratio, atr, obv, mfi,
)


class SwingStrategy(BaseStrategy):
    """일중 스윙 전략

    진입 조건 (매수):
      - 전일 대비 갭 상승 후 눌림목 (또는 갭 하락 후 반등)
      - 이동평균선 정배열 (EMA5 > EMA20 > EMA60)
      - MFI 저점 반등
      - 거래량 폭증 (평균 대비 2배 이상)

    청산 조건 (매도):
      - 목표 수익률 도달
      - 손절 기준 도달
      - 장 마감 30분 전 강제 청산
      - 이동평균선 역배열 전환
    """

    DEFAULT_PARAMS = {
        "gap_threshold": 2.0,       # 갭 기준 (%)
        "min_volume_ratio": 2.0,    # 최소 거래량 비율
        "ema_short": 5,
        "ema_mid": 20,
        "ema_long": 60,
        "rsi_period": 14,
        "rsi_oversold": 35.0,
        "mfi_oversold": 30.0,
        "stop_loss_pct": -1.5,
        "take_profit_pct": 3.0,
        "trailing_stop_pct": 1.0,   # 트레일링 스톱
        "max_hold_minutes": 360,
    }

    def __init__(self, params: Optional[dict] = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(name="SwingStrategy", params=merged)

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터에 스윙 전략 지표를 추가합니다."""
        result = df.copy()
        close = result["close"].astype(float)
        p = self.params

        # 이동평균선
        result["ema_short"] = ema(close, p["ema_short"])
        result["ema_mid"] = ema(close, p["ema_mid"])
        result["ema_long"] = ema(close, p["ema_long"])

        # 정배열 여부
        result["ema_aligned"] = (
            (result["ema_short"] > result["ema_mid"]) &
            (result["ema_mid"] > result["ema_long"])
        )

        # RSI
        result["rsi"] = rsi(close, p["rsi_period"])

        # MACD
        result["macd_line"], result["macd_sig"], result["macd_hist"] = macd(close)

        # 볼린저밴드
        result["bb_upper"], result["bb_mid"], result["bb_lower"], \
            result["bb_bw"], result["bb_pctb"] = bollinger_bands(close)

        # VWAP
        result["vwap"] = vwap(result)

        # 거래량 분석
        result["vol_ratio"] = volume_ratio(result)
        result["obv"] = obv(result)
        result["mfi"] = mfi(result)

        # ATR
        result["atr"] = atr(result)

        # 갭 분석 (일봉 기준)
        if "prev_close" in result.columns:
            result["gap_pct"] = (result["open"] - result["prev_close"]) / result["prev_close"] * 100
        else:
            result["gap_pct"] = (result["open"] - close.shift(1)) / close.shift(1) * 100

        # 시그널 계산
        result["signal"] = SignalType.HOLD.value
        result = self._generate_signals(result)

        return result

    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """전체 시그널을 계산합니다."""
        p = self.params

        # 매수 조건 1: 눌림목 매수 (정배열 + RSI/MFI 저점 + 거래량)
        pullback_buy = (
            df["ema_aligned"] &
            (df["rsi"] < 50) &  # RSI 중립 이하
            (df["mfi"] < p["mfi_oversold"]) &  # MFI 과매도
            (df["vol_ratio"] > p["min_volume_ratio"]) &
            (df["close"] > df["ema_mid"])  # 중기 이동평균 위
        )

        # 매수 조건 2: 갭 상승 후 눌림 반등
        gap_buy = (
            (df["gap_pct"] > p["gap_threshold"]) &
            (df["rsi"] < 50) &
            (df["close"] > df["vwap"]) &
            (df["vol_ratio"] > p["min_volume_ratio"])
        )

        # 매수 조건 3: 갭 하락 후 VWAP 위 반등
        gap_reversal = (
            (df["gap_pct"] < -p["gap_threshold"]) &
            (df["close"] > df["vwap"]) &
            (df["macd_hist"] > 0) &
            (df["macd_hist"].shift(1) <= 0)
        )

        buy_signal = pullback_buy | gap_buy | gap_reversal

        # 매도 조건
        sell_ema = (
            (df["ema_short"] < df["ema_mid"]) &
            (df["ema_short"].shift(1) >= df["ema_mid"].shift(1))
        )  # 단기 이평 하향돌파
        sell_rsi = df["rsi"] > 75
        sell_vwap = (df["close"] < df["vwap"]) & (df["close"].shift(1) >= df["vwap"].shift(1))

        sell_signal = sell_ema | (sell_rsi & sell_vwap)

        df.loc[buy_signal, "signal"] = SignalType.BUY.value
        df.loc[sell_signal, "signal"] = SignalType.SELL.value

        return df

    def generate_signal(
        self, df: pd.DataFrame, current_position: Optional[dict] = None
    ) -> Signal:
        """가장 최근 데이터 기반으로 시그널을 생성합니다."""
        if len(df) < 2:
            return Signal(
                type=SignalType.HOLD, stock_code="", price=0,
                strategy_name=self.name,
            )

        analyzed = self.analyze(df)
        latest = analyzed.iloc[-1]
        p = self.params
        price = int(latest["close"])
        stock_code = str(latest.get("stock_code", ""))

        # 포지션 보유 중: 청산 검사
        if current_position:
            entry_price = current_position["avg_price"]
            pnl_pct = (price - entry_price) / entry_price * 100
            max_price = current_position.get("max_price", entry_price)

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

            # 트레일링 스톱
            if max_price > entry_price:
                trailing_pct = (price - max_price) / max_price * 100
                if trailing_pct <= -p["trailing_stop_pct"]:
                    return Signal(
                        type=SignalType.SELL, stock_code=stock_code, price=price,
                        quantity=current_position["quantity"],
                        confidence=0.85,
                        reason=f"트레일링스톱: 최고가 대비 {trailing_pct:.2f}%",
                        strategy_name=self.name,
                    )

            # 기술적 매도
            if latest["signal"] == SignalType.SELL.value:
                return Signal(
                    type=SignalType.SELL, stock_code=stock_code, price=price,
                    quantity=current_position["quantity"],
                    confidence=0.7,
                    reason=f"기술적 매도 (EMA 역전환 또는 RSI={latest['rsi']:.1f})",
                    strategy_name=self.name,
                )

            return Signal(
                type=SignalType.HOLD, stock_code=stock_code, price=price,
                strategy_name=self.name,
            )

        # 포지션 없음: 진입 검사
        if latest["signal"] == SignalType.BUY.value:
            confidence = 0.5
            reasons = []

            if latest.get("ema_aligned", False):
                confidence += 0.15
                reasons.append("정배열")
            if latest["vol_ratio"] > p["min_volume_ratio"] * 1.5:
                confidence += 0.1
                reasons.append(f"VR={latest['vol_ratio']:.1f}")
            if latest["mfi"] < p["mfi_oversold"]:
                confidence += 0.1
                reasons.append(f"MFI={latest['mfi']:.1f}")
            gap = latest.get("gap_pct", 0)
            if abs(gap) > p["gap_threshold"]:
                confidence += 0.1
                reasons.append(f"GAP={gap:.1f}%")
            if latest["close"] > latest["vwap"]:
                confidence += 0.05
                reasons.append("VWAP↑")

            atr_val = int(latest["atr"]) if latest["atr"] > 0 else int(price * 0.015)
            stop_loss = price - int(atr_val * 2)
            take_profit = price + int(atr_val * 4)

            return Signal(
                type=SignalType.BUY, stock_code=stock_code, price=price,
                stop_loss=stop_loss, take_profit=take_profit,
                confidence=min(confidence, 1.0),
                reason=f"스윙 매수: {', '.join(reasons)}",
                strategy_name=self.name,
            )

        return Signal(
            type=SignalType.HOLD, stock_code=stock_code, price=price,
            strategy_name=self.name,
        )
