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
    add_execution_strength, add_volume_spike,
)

# 캔들 패턴 모듈 (없으면 무시)
try:
    from features import candle_patterns as _cp
    _HAS_CANDLE_PATTERNS = True
except ImportError:
    _HAS_CANDLE_PATTERNS = False

# 마켓 사이클 모듈 (없으면 무시)
try:
    from features.market_cycle import (
        MarketCycleAnalyzer,
        CycleSignal,
        PHASE_EARLY,
        PHASE_LATE,
        POSTURE_DEFENSIVE,
        POSTURE_AGGRESSIVE,
    )
    _HAS_MARKET_CYCLE = True
except ImportError:
    _HAS_MARKET_CYCLE = False

# 캔들마스터 파동 분석 모듈 (없으면 무시)
try:
    from features.wave_position import WavePositionAnalyzer, WaveSignal
    _HAS_WAVE_POSITION = True
except ImportError:
    _HAS_WAVE_POSITION = False

# 잭 슈웨거 마법사 교훈 모듈 (없으면 무시)
try:
    from features.wizard_discipline import (
        IndicatorSynergy,
        CatalystVerifier,
        NewsReactionAnalyzer,
        OpportunityCostEvaluator,
    )
    _HAS_WIZARD = True
except ImportError:
    _HAS_WIZARD = False

# 켄 피셔 시장 기억 분석 모듈 (없으면 무시)
try:
    from features.market_memory import MarketMemoryAnalyzer, FisherSignal
    _HAS_MARKET_MEMORY = True
except ImportError:
    _HAS_MARKET_MEMORY = False

# 강방천&존리 가치투자 분석 모듈 (없으면 무시)
try:
    from features.value_investor import ValueInvestorAnalyzer, ValueInvestorSignal
    _HAS_VALUE_INVESTOR = True
except ImportError:
    _HAS_VALUE_INVESTOR = False

# 이남우 주식 품질 평가 모듈 (없으면 무시)
try:
    from features.stock_quality import StockQualityAnalyzer, StockQualitySignal
    _HAS_STOCK_QUALITY = True
except ImportError:
    _HAS_STOCK_QUALITY = False

# 서준식 딥밸류/채권형 주식 분석 모듈 (없으면 무시)
try:
    from features.deep_value import SeoJunsikAnalyzer, SeoJunsikSignal
    _HAS_DEEP_VALUE = True
except ImportError:
    _HAS_DEEP_VALUE = False


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
        # 마켓 사이클 분석기 초기화 (하워드 막스 사이클 이론)
        if _HAS_MARKET_CYCLE:
            self._cycle_analyzer = MarketCycleAnalyzer(
                lookback_days=merged.get("cycle_lookback_days", 252),
                early_phase_max_pos=merged.get("early_phase_max_pos", 0.8),
                mid_phase_max_pos=merged.get("mid_phase_max_pos", 0.5),
                late_phase_max_pos=merged.get("late_phase_max_pos", 0.2),
            )
        else:
            self._cycle_analyzer = None
        self._last_cycle: Optional["CycleSignal"] = None
        # 캔들마스터 파동 분석기 초기화
        if _HAS_WAVE_POSITION:
            self._wave_analyzer = WavePositionAnalyzer(
                lookback_weeks=merged.get("wave_lookback_weeks", 252),
                decline_threshold=merged.get("wave_decline_threshold", 0.50),
                horizontal_width_pct=merged.get("wave_horizontal_width_pct", 0.30),
                min_candle_count=merged.get("wave_min_candle_count", 50),
                max_price_multiple=merged.get("wave_max_price_multiple", 10.0),
            )
        else:
            self._wave_analyzer = None
        self._last_wave: Optional["WaveSignal"] = None
        # 켄 피셔 시장 기억 분석기 초기화
        if _HAS_MARKET_MEMORY:
            self._memory_analyzer = MarketMemoryAnalyzer(
                vol_lookback=merged.get("fisher_vol_lookback", 60),
                atr_threshold=merged.get("fisher_atr_threshold", 2.0),
                wow_lookback=merged.get("fisher_wow_lookback", 20),
                bear_decline=merged.get("fisher_bear_decline", 0.20),
                bear_recovery=merged.get("fisher_bear_recovery", 0.10),
            )
        else:
            self._memory_analyzer = None
        self._last_fisher: Optional["FisherSignal"] = None
        # 강방천&존리 가치투자 분석기 초기화
        if _HAS_VALUE_INVESTOR:
            self._value_analyzer = ValueInvestorAnalyzer(
                fundamental_lookback=merged.get("value_fundamental_lookback", 120),
                contrarian_rsi=merged.get("value_contrarian_rsi", 30.0),
                hold_min_days=merged.get("value_hold_min_days", 60),
                overvalued_rsi=merged.get("value_overvalued_rsi", 80.0),
            )
        else:
            self._value_analyzer = None
        self._last_value: Optional["ValueInvestorSignal"] = None
        # 이남우 주식 품질 분석기 초기화
        if _HAS_STOCK_QUALITY:
            self._quality_analyzer = StockQualityAnalyzer(
                roe_lookback=merged.get("quality_roe_lookback", 120),
                panic_threshold=merged.get("quality_panic_threshold", 0.30),
                panic_lookback=merged.get("quality_panic_lookback", 252),
                intensity_lookback=merged.get("quality_intensity_lookback", 60),
                high_atr_pct=merged.get("quality_high_atr_pct", 0.04),
            )
        else:
            self._quality_analyzer = None
        self._last_quality: Optional["StockQualitySignal"] = None

        # 서준식 딥밸류 분석기 초기화
        if _HAS_DEEP_VALUE:
            self._deep_value_analyzer = SeoJunsikAnalyzer(
                lookback=merged.get("deep_value_lookback", 252),
                target_return=merged.get("deep_value_target_return", 0.15),
                bond_type_threshold=merged.get("deep_value_bond_threshold", 0.5),
                safety_margin_threshold=merged.get("deep_value_safety_threshold", 0.5),
                deep_value_threshold=merged.get("deep_value_threshold", 0.60),
                overvalue_threshold=merged.get("deep_value_overvalue_threshold", 0.30),
            )
        else:
            self._deep_value_analyzer = None
        self._last_deep_value: Optional["SeoJunsikSignal"] = None

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

        # 체결강도 (도서 p.88)
        try:
            result = add_execution_strength(result)
        except Exception:
            result["execution_strength"] = 100.0

        # 거래량 급증 감지
        try:
            result = add_volume_spike(result)
        except Exception:
            result["volume_spike"] = 0

        # 캔들 패턴 분석 (도서 Part 3: 봉 해석)
        try:
            if _HAS_CANDLE_PATTERNS:
                result["bullish_engulfing"] = _cp.bullish_engulfing(result)
                result["morning_star"] = _cp.morning_star(result)
        except Exception:
            pass

        # 갭 분석 (일봉 기준)
        if "prev_close" in result.columns:
            result["gap_pct"] = (result["open"] - result["prev_close"]) / result["prev_close"] * 100
        else:
            result["gap_pct"] = (result["open"] - close.shift(1)) / close.shift(1) * 100

        # 마켓 사이클 분석 (하워드 막스 사이클 이론)
        try:
            if _HAS_MARKET_CYCLE and self._cycle_analyzer is not None:
                cycle_sig = self._cycle_analyzer.analyze(result)
                self._last_cycle = cycle_sig
                result["cycle_score"] = cycle_sig.cycle_score
                result["cycle_phase"] = cycle_sig.phase
                result["cycle_posture"] = cycle_sig.risk_posture
                result["cycle_max_pos"] = cycle_sig.max_position_pct
        except Exception:
            pass

        # 캔들마스터 파동 분석
        try:
            if _HAS_WAVE_POSITION and self._wave_analyzer is not None:
                wave_sig = self._wave_analyzer.analyze(result)
                self._last_wave = wave_sig
                result["wave_buy_score"] = wave_sig.buy_zone_score
                result["wave_type"] = wave_sig.wave_type
                result["wave_latter_half"] = int(wave_sig.is_latter_half)
        except Exception:
            pass

        # 캔들군 분석 (캔들마스터 방식)
        try:
            if _HAS_CANDLE_PATTERNS:
                result = _cp.detect_candle_groups(result)
                result = _cp.get_candle_group_signal(result)
        except Exception:
            pass

        # 켄 피셔 시장 기억 분석 (변동성 정상화, 불신의 비관론, 약세장 회복)
        try:
            if _HAS_MARKET_MEMORY and self._memory_analyzer is not None:
                fisher_sig = self._memory_analyzer.analyze(result)
                self._last_fisher = fisher_sig
                result["fisher_composite"] = fisher_sig.fisher_composite
                result["fisher_pos_mult"] = fisher_sig.position_multiplier
                result["fisher_vol_fear"] = fisher_sig.volatility_fear_score
                result["fisher_wow"] = fisher_sig.wall_of_worry_score
                result["fisher_v_shape"] = fisher_sig.v_shape_score
                result["fisher_early_bull"] = fisher_sig.early_bull_phase
                result["fisher_extreme_distrust"] = (
                    fisher_sig.extreme_distrust
                )
                result["fisher_exp_12m"] = fisher_sig.expected_12m_return
        except Exception:
            pass

        # 강방천&존리 가치투자 분석 (재무건전성, 저평가, 역발상)
        try:
            if _HAS_VALUE_INVESTOR and self._value_analyzer is not None:
                value_sig = self._value_analyzer.analyze(result)
                self._last_value = value_sig
                result["value_fundamental"] = value_sig.fundamental_score
                result["value_valuation"] = value_sig.valuation_score
                result["value_contrarian"] = value_sig.contrarian_score
        except Exception:
            pass

        # 이남우 주식 품질 분석 (ROE 듀폰, 패닉매수, 자본집약도)
        try:
            if _HAS_STOCK_QUALITY and self._quality_analyzer is not None:
                quality_sig = self._quality_analyzer.analyze(result)
                self._last_quality = quality_sig
                result["quality_score"] = quality_sig.quality_score
                result["quality_roe"] = quality_sig.roe_quality
                result["quality_panic"] = int(quality_sig.is_panic_buy)
                result["quality_intensity"] = quality_sig.capital_intensity
        except Exception:
            pass

        # 서준식 딥밸류 분석 (채권형주식, 기대수익률, 안전마진, 떨어지는칼날)
        try:
            if _HAS_DEEP_VALUE and self._deep_value_analyzer is not None:
                dv_sig = self._deep_value_analyzer.analyze(result)
                self._last_deep_value = dv_sig
                result["deep_value_bond_type"] = dv_sig.bond_type_score
                result["deep_value_expected_return"] = dv_sig.expected_return
                result["deep_value_safety_margin"] = dv_sig.safety_margin_score
                result["deep_value_falling_knife"] = dv_sig.falling_knife_score
                result["deep_value_buy_candidate"] = int(dv_sig.is_buy_candidate)
        except Exception:
            pass

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

            # --------------------------------------------------------
            # 마켓 사이클 필터 (하워드 막스 사이클 이론)
            # LATE phase(70점 이상) → 신규 매수 차단 (방어적)
            # EARLY phase(30점 이하) → 포지션 확대 (공격적)
            # 전환점 감지 → 신뢰도 +15%
            # --------------------------------------------------------
            cycle_block = False
            try:
                if _HAS_MARKET_CYCLE and self._last_cycle is not None:
                    cs = self._last_cycle
                    if cs.phase == PHASE_LATE:
                        # 사이클 말기: 신규 매수 자제
                        cycle_block = True
                        logger.info(
                            f"[사이클필터] LATE phase({cs.cycle_score:.1f}점) "
                            f"→ 신규 매수 차단. {cs.note}"
                        )
                    elif cs.phase == PHASE_EARLY:
                        # 사이클 초기: 공격적 매수 신호
                        confidence += 0.1
                        reasons.append(
                            f"사이클초기({cs.cycle_score:.0f}점)"
                        )
                    if cs.turning_point and not cycle_block:
                        # 전환점 감지: 신뢰도 +15%
                        confidence += 0.15
                        reasons.append("사이클전환점")
            except Exception:
                pass

            if cycle_block:
                return Signal(
                    type=SignalType.HOLD, stock_code=stock_code, price=price,
                    strategy_name=self.name,
                )

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

            # 캔들 패턴 신뢰도 (도서 Part 3: 봉 해석 기반)
            try:
                if latest.get("bullish_engulfing", False):
                    confidence += 0.1
                    reasons.append("양봉장악형")
                if latest.get("morning_star", False):
                    confidence += 0.15
                    reasons.append("샛별형")
            except Exception:
                pass

            # 주도주 추종: 거래량 급증 + EMA 정배열 (도서 Part 3)
            try:
                if (latest.get("volume_spike", 0) == 1
                        and latest.get("ema_aligned", False)):
                    confidence += 0.1
                    reasons.append("주도주후보")
            except Exception:
                pass

            # --------------------------------------------------------
            # 캔들마스터 파동 위치 분석 필터
            # 후반부 + 유효 → 신뢰도 +15%
            # 무효 조건 해당 → 매수 차단 (HOLD)
            # 캔들군 신호 결합
            # --------------------------------------------------------
            wave_block = False
            try:
                if _HAS_WAVE_POSITION and self._last_wave is not None:
                    ws = self._last_wave
                    if ws.invalidation is not None:
                        # 무효 조건 해당: 매수 차단
                        wave_block = True
                        logger.info(
                            "[파동필터] 무효 조건: {} → 매수 차단",
                            ws.invalidation
                        )
                    elif ws.is_latter_half and ws.invalidation is None:
                        # 후반부 + 유효: 신뢰도 +15%
                        confidence += 0.15
                        reasons.append(
                            f"파동후반부(점수{ws.buy_zone_score:.0f})"
                        )
                    if ws.buy_zone_score >= 70 and not wave_block:
                        confidence += 0.1
                        reasons.append(f"파동매수구간({ws.wave_type})")
            except Exception:
                pass

            if wave_block:
                return Signal(
                    type=SignalType.HOLD, stock_code=stock_code, price=price,
                    reason="파동 무효 조건 해당",
                    strategy_name=self.name,
                )

            # 캔들군 신호 결합 (캔들마스터)
            try:
                candle_group_sig = int(latest.get("candle_group_signal", 0))
                if candle_group_sig > 0:
                    confidence += 0.1
                    reasons.append("캔들군매수")
                elif candle_group_sig < 0:
                    confidence -= 0.1
                    reasons.append("캔들군매도압력")
            except Exception:
                pass

            # --------------------------------------------------------
            # 켄 피셔 시장 기억 필터
            # Wall of Worry(불신 비관론) → 신뢰도 부스트
            # 변동성 공포 극단 → 기회 신호
            # 약세장 회복 초기 → 신뢰도 부스트
            # V자 회복 대칭성 → 신뢰도 부스트 (신규)
            # 강세장 초기 → 신뢰도 대폭 부스트 (신규)
            # 극단 비관론 축적 → 역발상 매수 신호 (신규)
            # --------------------------------------------------------
            try:
                if _HAS_MARKET_MEMORY and self._last_fisher is not None:
                    fs = self._last_fisher
                    if fs.confidence_delta != 0:
                        confidence += fs.confidence_delta
                        reasons.append(f"켄피셔({fs.confidence_delta:+.2f})")
                    if fs.fisher_composite > 0.5:
                        reasons.append(f"시장기억기회({fs.fisher_composite:.2f})")
                    elif fs.fisher_composite < -0.5:
                        # 과도한 안정(버블 전조): 신뢰도 소폭 감소
                        confidence -= 0.05
                        reasons.append("시장기억경고")
                    # 강세장 초기: 역사적 최대 기회 (+23.1%/3개월, +46.6%/12개월)
                    if fs.early_bull_phase:
                        confidence += 0.10
                        reasons.append("강세장초기(켄피셔:12개월+46.6%역사평균)")
                    # 극단 비관론 축적: "이번에도 다르지 않다" 역발상
                    if fs.extreme_distrust:
                        confidence += 0.08
                        reasons.append("이번엔다르다착각(켄피셔:역발상매수)")
                    # V자 회복 대칭성 강함
                    if fs.v_shape_score > 0.7:
                        reasons.append(
                            f"V자회복대칭({fs.v_shape_score:.2f})"
                        )
            except Exception:
                pass

            # --------------------------------------------------------
            # 강방천&존리 가치투자 필터
            # 역발상 기회: contrarian_score > 0.6 → +10% 신뢰도
            # 고평가 경고: valuation_score < 0.3 → 매수 차단
            # 재무건전성 프록시 반영
            # --------------------------------------------------------
            value_block = False
            try:
                if _HAS_VALUE_INVESTOR and self._last_value is not None:
                    vs = self._last_value
                    if vs.confidence_delta != 0:
                        confidence += vs.confidence_delta
                        reasons.append(f"가치투자({vs.confidence_delta:+.2f})")
                    if vs.contrarian_score > 0.6:
                        reasons.append(
                            f"역발상기회({vs.contrarian_score:.2f})"
                        )
                    if vs.valuation_score < 0.3:
                        # 고평가 경고: 매수 차단
                        value_block = True
                        logger.info(
                            "[가치투자필터] 고평가 경고(val={:.2f}) → 매수 보류",
                            vs.valuation_score
                        )
            except Exception:
                pass

            if value_block:
                return Signal(
                    type=SignalType.HOLD, stock_code=stock_code, price=price,
                    reason="고평가 경고 (강방천&존리: 쌀때 사라)",
                    strategy_name=self.name,
                )

            # --------------------------------------------------------
            # 이남우 주식 품질 필터
            # 저품질 종목: 신뢰도 감소
            # 패닉 매수 기회: 신뢰도 +15%
            # 자본집약적 종목: 신뢰도 감소
            # --------------------------------------------------------
            quality_block = False
            try:
                if _HAS_STOCK_QUALITY and self._last_quality is not None:
                    qs = self._last_quality
                    if qs.confidence_delta != 0:
                        confidence += qs.confidence_delta
                    if qs.quality_score < 0.3:
                        quality_block = True
                        logger.info(
                            "[품질필터] 저품질 종목(score={:.2f}) → 매수 보류",
                            qs.quality_score
                        )
                    elif qs.is_panic_buy:
                        confidence += 0.15
                        reasons.append(
                            f"패닉매수기회(하락{qs.panic_depth:.0%})"
                        )
                    if qs.capital_intensity > 0.7 and not quality_block:
                        reasons.append(
                            f"자본집약({qs.capital_intensity:.2f})"
                        )
            except Exception:
                pass

            if quality_block:
                return Signal(
                    type=SignalType.HOLD, stock_code=stock_code, price=price,
                    reason="저품질 종목 (이남우: 나쁜 주식 회피)",
                    strategy_name=self.name,
                )

            # --------------------------------------------------------
            # 서준식 딥밸류/채권형 주식 필터
            # 고평가: 매수 차단
            # 매수 후보(기대수익률≥15%): 신뢰도 +12%
            # 떨어지는 칼날(우량주 급락): 신뢰도 +10%
            # --------------------------------------------------------
            deep_value_block = False
            try:
                if _HAS_DEEP_VALUE and self._last_deep_value is not None:
                    dv = self._last_deep_value
                    if dv.confidence_delta != 0:
                        confidence += dv.confidence_delta
                        reasons.append(f"딥밸류({dv.confidence_delta:+.2f})")
                    if dv.is_buy_candidate:
                        reasons.append(
                            f"매수후보(기대{dv.expected_return:.0%})"
                        )
                    if dv.falling_knife_score > 0.6:
                        reasons.append(
                            f"칼날잡기({dv.falling_knife_score:.2f})"
                        )
                    if dv.is_overvalued:
                        deep_value_block = True
                        logger.info(
                            "[딥밸류필터] 고평가(bond=%.2f,margin=%.2f) → 매수 보류",
                            dv.bond_type_score, dv.safety_margin_score
                        )
            except Exception:
                pass

            if deep_value_block:
                return Signal(
                    type=SignalType.HOLD, stock_code=stock_code, price=price,
                    reason="고평가 종목 (서준식: 안전마진 부족)",
                    strategy_name=self.name,
                )

            # --------------------------------------------------------
            # 잭 슈웨거 마법사 교훈 필터
            # 촉매 검증 (교훈 19), 지표 시너지 (교훈 54),
            # 뉴스 반응 (교훈 21, 49)
            # --------------------------------------------------------
            try:
                if _HAS_WIZARD:
                    # 촉매 검증: 촉매 없으면 진입 차단
                    catalyst_verifier = CatalystVerifier(
                        volume_threshold=self.params.get(
                            "catalyst_volume_threshold", 2.0
                        )
                    )
                    if not catalyst_verifier.has_catalyst(analyzed):
                        logger.info(
                            "[마법사필터] 촉매 미감지 → 매수 보류"
                        )
                        return Signal(
                            type=SignalType.HOLD,
                            stock_code=stock_code,
                            price=price,
                            reason="촉매 미감지 (슈웨거 교훈19)",
                            strategy_name=self.name,
                        )
                    # 촉매 강도 반영
                    cat_str = catalyst_verifier.catalyst_strength(
                        analyzed
                    )
                    if cat_str > 0.5:
                        confidence += 0.1
                        reasons.append(
                            f"촉매강도{cat_str:.1f}"
                        )

                    # 지표 시너지 프레임워크
                    synergy_analyzer = IndicatorSynergy()
                    syn_result = synergy_analyzer.get_wizard_signal(
                        analyzed
                    )
                    syn = syn_result["synergy"]
                    if syn.strength == "STRONG":
                        confidence += 0.15
                        reasons.append(
                            f"시너지STRONG({syn.aligned_count}개)"
                        )
                    elif syn.strength == "MODERATE":
                        confidence += 0.05
                        reasons.append("시너지MODERATE")
                    elif syn.strength == "WEAK":
                        confidence -= 0.1
                        reasons.append("시너지WEAK")

                    # 뉴스 반응 기대 갭 감지
                    if NewsReactionAnalyzer.detect_expectation_gap(
                        analyzed
                    ):
                        logger.info(
                            "[마법사필터] 기대갭 감지 → 매수 보류"
                        )
                        return Signal(
                            type=SignalType.HOLD,
                            stock_code=stock_code,
                            price=price,
                            reason="기대갭 감지 (슈웨거 교훈21)",
                            strategy_name=self.name,
                        )
            except Exception:
                pass

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
