"""서준식 "다시 쓰는 주식투자 교과서" - 딥밸류 분석 모듈

채권형 주식(Bond-Type Stock) 식별 및 가치투자 분석.
핵심 철학:
  - 채권형 주식: 안정적 ROE, 비경기순환, 저자본집약적 기업
  - 기대수익률 15%: BPS × (1+ROE)^10 복리 산출, 현재가 대비 15% 이상만 매수
  - 떨어지는 칼날 잡기: 가치 불변 우량주 급락 = 원칙대로 매수
  - 안전마진: 시장가 << 내재가치일 때만 투자

주요 클래스:
  - BondTypeStockScorer: 채권형 주식 적합도 (변동성+베타+자본집약도)
  - ExpectedReturnCalculator: 10년 복리 기대수익률 산출
  - SafetyMarginScorer: 안전마진 평가 (52주저점+MA비율+BB위치)
  - FallingKnifeDetector: 떨어지는 칼날 매수기회 감지
  - SeoJunsikAnalyzer: 통합 분석기
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SeoJunsikSignal:
    """서준식 딥밸류 분석 시그널"""

    bond_type_score: float = 0.5        # 채권형 주식 적합도 (0~1)
    expected_return: float = 0.0        # 기대수익률 (연환산, 예: 0.15=15%)
    safety_margin_score: float = 0.5    # 안전마진 점수 (0~1)
    falling_knife_score: float = 0.0    # 떨어지는 칼날 기회 (0~1)
    is_buy_candidate: bool = False      # 매수 후보 여부
    is_overvalued: bool = False         # 고평가 차단 여부
    position_multiplier: float = 1.0    # 포지션 배수 (0.7~1.3)
    confidence_delta: float = 0.0       # 신뢰도 조정 (-0.2~+0.2)
    note: str = ""


# ---------------------------------------------------------------------------
# 1. BondTypeStockScorer: 채권형 주식 적합도 평가
# ---------------------------------------------------------------------------
class BondTypeStockScorer:
    """채권형 주식 4-Point 체크리스트 기술적 프록시

    서준식 체크리스트:
      1) 비경기순환주 → 낮은 베타(시장 민감도)
      2) 저자본집약 → 낮은 ATR/가격 비율
      3) 이해 가능한 기업 → (OHLCV로 측정 불가, 가중치 재분배)
      4) 안정적 ROE → 낮은 수익률 변동성(가격 변동성 프록시)
    """

    def __init__(self, lookback: int = 252):
        self.lookback = lookback

    def score(self, df: pd.DataFrame) -> float:
        if len(df) < max(60, self.lookback):
            return 0.5

        close = df["close"].values[-self.lookback:]
        high = df["high"].values[-self.lookback:]
        low = df["low"].values[-self.lookback:]

        returns = np.diff(close) / close[:-1]
        returns = returns[np.isfinite(returns)]
        if len(returns) < 20:
            return 0.5

        # (1) 수익률 안정성 (낮은 변동성 = 안정적 ROE 프록시)
        daily_vol = float(np.std(returns))
        stability = float(np.clip(1.0 - daily_vol / 0.03, 0.0, 1.0))

        # (2) 베타 프록시 (수익률 절대값의 평균 대비 표준편차)
        abs_ret = np.abs(returns)
        mean_abs = float(np.mean(abs_ret)) if np.mean(abs_ret) > 0 else 0.01
        beta_proxy = float(np.std(abs_ret)) / mean_abs
        non_cyclicality = float(np.clip(1.0 - beta_proxy / 2.0, 0.0, 1.0))

        # (3) 자본집약도 프록시 (ATR / 가격 비율)
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
        atr_14 = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))
        current_price = float(close[-1])
        atr_pct = atr_14 / current_price if current_price > 0 else 0.04
        capital_light = float(np.clip(1.0 - atr_pct / 0.04, 0.0, 1.0))

        # 가중 합산
        composite = stability * 0.40 + non_cyclicality * 0.35 + capital_light * 0.25
        return round(float(np.clip(composite, 0.0, 1.0)), 4)


# ---------------------------------------------------------------------------
# 2. ExpectedReturnCalculator: 기대수익률 산출
# ---------------------------------------------------------------------------
class ExpectedReturnCalculator:
    """서준식 기대수익률 공식 (OHLCV 프록시 버전)

    원본 공식: BPS × (1+ROE)^10 / 현재가 → 연환산 복리 수익률
    프록시:
      - BPS → 52주 SMA (장기 평균가 = 장부가 프록시)
      - ROE → Sharpe ratio 정규화 (수익률/변동성 = 자본효율 프록시)
    """

    def __init__(
        self,
        lookback: int = 252,
        projection_years: int = 10,
        target_return: float = 0.15,
    ):
        self.lookback = lookback
        self.projection_years = projection_years
        self.target_return = target_return

    def calculate(self, df: pd.DataFrame) -> Tuple[float, bool]:
        """기대수익률 산출

        Returns:
            (연환산 기대수익률, 매수후보 여부)
        """
        if len(df) < max(60, self.lookback):
            return 0.0, False

        close = df["close"].values[-self.lookback:]
        returns = np.diff(close) / close[:-1]
        returns = returns[np.isfinite(returns)]
        if len(returns) < 20:
            return 0.0, False

        current_price = float(close[-1])
        if current_price <= 0:
            return 0.0, False

        # BPS 프록시: 52주 SMA
        bps_proxy = float(np.mean(close))

        # ROE 프록시: 연환산 Sharpe → 0~0.20 범위로 정규화
        ann_return = float(np.mean(returns)) * 252
        ann_vol = float(np.std(returns)) * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
        proxy_roe = float(np.clip(sharpe * 0.10, 0.0, 0.20))

        # 10년 복리 미래가치
        future_value = bps_proxy * (1.0 + proxy_roe) ** self.projection_years

        # 연환산 기대수익률
        if current_price > 0 and future_value > current_price:
            ratio = future_value / current_price
            expected = ratio ** (1.0 / self.projection_years) - 1.0
        elif current_price > 0:
            expected = (future_value / current_price) ** (
                1.0 / self.projection_years
            ) - 1.0
        else:
            expected = 0.0

        expected = float(np.clip(expected, -0.50, 0.50))
        is_buy = expected >= self.target_return
        return round(expected, 4), is_buy


# ---------------------------------------------------------------------------
# 3. SafetyMarginScorer: 안전마진 평가
# ---------------------------------------------------------------------------
class SafetyMarginScorer:
    """서준식 안전마진 원칙

    시장가가 내재가치 대비 충분히 낮을 때만 투자.
    기술적 프록시:
      - 52주 저점 대비 거리 (낮을수록 안전마진 큼)
      - 200일 이동평균 대비 비율 (아래일수록 안전마진 큼)
      - 볼린저밴드 위치 (하단 근접 = 저평가)
    """

    def __init__(self, lookback: int = 252):
        self.lookback = lookback

    def score(self, df: pd.DataFrame) -> float:
        if len(df) < max(20, self.lookback):
            return 0.5

        close = df["close"].values[-self.lookback:]
        current = float(close[-1])
        if current <= 0:
            return 0.5

        # (1) 52주 고저 대비 위치 (0=저점, 1=고점) → 반전
        high_52w = float(np.max(close))
        low_52w = float(np.min(close))
        price_range = high_52w - low_52w
        if price_range > 0:
            position = (current - low_52w) / price_range
            low_distance = float(np.clip(1.0 - position, 0.0, 1.0))
        else:
            low_distance = 0.5

        # (2) 200일 이동평균 대비 비율
        sma_len = min(200, len(close))
        sma_200 = float(np.mean(close[-sma_len:]))
        if sma_200 > 0:
            ma_ratio = current / sma_200
            ma_score = float(np.clip(1.0 - ma_ratio, -0.5, 0.5)) + 0.5
            ma_score = float(np.clip(ma_score, 0.0, 1.0))
        else:
            ma_score = 0.5

        # (3) 볼린저밴드 위치 (20일, 2σ)
        bb_len = min(20, len(close))
        bb_close = close[-bb_len:]
        bb_mean = float(np.mean(bb_close))
        bb_std = float(np.std(bb_close))
        if bb_std > 0:
            upper_bb = bb_mean + 2.0 * bb_std
            lower_bb = bb_mean - 2.0 * bb_std
            bb_range = upper_bb - lower_bb
            if bb_range > 0:
                bb_position = (current - lower_bb) / bb_range
                bb_score = float(np.clip(1.0 - bb_position, 0.0, 1.0))
            else:
                bb_score = 0.5
        else:
            bb_score = 0.5

        composite = low_distance * 0.35 + ma_score * 0.35 + bb_score * 0.30
        return round(float(np.clip(composite, 0.0, 1.0)), 4)


# ---------------------------------------------------------------------------
# 4. FallingKnifeDetector: 떨어지는 칼날 매수기회 감지
# ---------------------------------------------------------------------------
class FallingKnifeDetector:
    """서준식 '떨어지는 칼날을 잡아라' 원칙

    진정한 가치투자자는 원칙에 따라 하락하는 우량주를 매수한다.
    조건: 52주 고점 대비 20%+ 하락 + 펀더멘털 건전(OBV 프록시) + 채권형 주식
    """

    def __init__(self, decline_threshold: float = 0.20, lookback: int = 252):
        self.decline_threshold = decline_threshold
        self.lookback = lookback

    def detect(self, df: pd.DataFrame, bond_type_score: float = 0.5) -> float:
        if len(df) < max(60, self.lookback):
            return 0.0

        close = df["close"].values[-self.lookback:]
        volume = df["volume"].values[-self.lookback:]
        current = float(close[-1])
        high_52w = float(np.max(close))

        if high_52w <= 0:
            return 0.0

        # 고점 대비 하락률
        decline = (high_52w - current) / high_52w
        if decline < self.decline_threshold:
            return 0.0

        # 채권형 주식 최소 기준
        if bond_type_score < 0.4:
            return 0.0

        # OBV 추세 (20일) - 펀더멘털 건전성 프록시
        if len(close) >= 20 and len(volume) >= 20:
            price_changes = np.sign(np.diff(close[-20:]))
            obv_changes = price_changes * volume[-19:]
            obv_cum = np.cumsum(obv_changes)
            if len(obv_cum) >= 2:
                obv_slope = float(obv_cum[-1] - obv_cum[0])
                # OBV가 크게 하락하지 않으면 펀더멘털 건전
                obv_factor = 1.0 if obv_slope >= 0 else max(0.3, 1.0 + (obv_slope / (abs(obv_slope) + 1e-10)) * 0.7)
            else:
                obv_factor = 0.5
        else:
            obv_factor = 0.5

        # 거래량 유지 확인 (최근 10일 / 60일 평균)
        if len(volume) >= 60:
            vol_recent = float(np.mean(volume[-10:]))
            vol_long = float(np.mean(volume[-60:]))
            vol_ratio = vol_recent / vol_long if vol_long > 0 else 0.5
            vol_factor = float(np.clip(vol_ratio, 0.3, 1.0))
        else:
            vol_factor = 0.5

        # 하락 깊이 정규화 (20%=0.3, 40%=0.7, 50%+=1.0)
        depth_score = float(np.clip((decline - 0.10) / 0.40, 0.0, 1.0))

        score = depth_score * obv_factor * vol_factor * min(bond_type_score, 1.0)
        return round(float(np.clip(score, 0.0, 1.0)), 4)


# ---------------------------------------------------------------------------
# 5. SeoJunsikAnalyzer: 통합 분석기
# ---------------------------------------------------------------------------
class SeoJunsikAnalyzer:
    """서준식 딥밸류 통합 분석기

    4개 서브모듈을 통합하여 SeoJunsikSignal을 산출한다.
    composite = bond_type×0.30 + safety_margin×0.30
              + expected_return_norm×0.25 + falling_knife×0.15
    """

    def __init__(
        self,
        lookback: int = 252,
        target_return: float = 0.15,
        bond_type_threshold: float = 0.5,
        safety_margin_threshold: float = 0.5,
        deep_value_threshold: float = 0.60,
        overvalue_threshold: float = 0.30,
        max_position_multiplier: float = 1.3,
        min_position_multiplier: float = 0.7,
    ):
        self._bond_scorer = BondTypeStockScorer(lookback=lookback)
        self._return_calc = ExpectedReturnCalculator(
            lookback=lookback, target_return=target_return
        )
        self._margin_scorer = SafetyMarginScorer(lookback=lookback)
        self._knife_detector = FallingKnifeDetector(lookback=lookback)
        self._target_return = target_return
        self._deep_threshold = deep_value_threshold
        self._over_threshold = overvalue_threshold
        self._max_mult = max_position_multiplier
        self._min_mult = min_position_multiplier

    def analyze(self, df: pd.DataFrame) -> SeoJunsikSignal:
        """OHLCV DataFrame 분석 → SeoJunsikSignal 반환

        Args:
            df: OHLCV DataFrame (최소 60행, 252행 권장)
        """
        if len(df) < 20:
            return SeoJunsikSignal(note="데이터 부족")

        # 서브 스코어 산출
        bond_type = self._bond_scorer.score(df)
        expected_ret, is_buy_by_return = self._return_calc.calculate(df)
        safety_margin = self._margin_scorer.score(df)
        falling_knife = self._knife_detector.detect(df, bond_type_score=bond_type)

        # 기대수익률 정규화 (0~1): 30% 연간을 상한으로
        ret_norm = float(np.clip(expected_ret / 0.30, 0.0, 1.0))

        # 복합 점수
        composite = (
            bond_type * 0.30
            + safety_margin * 0.30
            + ret_norm * 0.25
            + falling_knife * 0.15
        )

        # 매수/매도 판단
        is_buy = composite > self._deep_threshold and is_buy_by_return
        is_over = bond_type < self._over_threshold and safety_margin < self._over_threshold

        # 포지션 배수
        if is_buy:
            pos_mult = self._max_mult
        elif is_over:
            pos_mult = self._min_mult
        else:
            pos_mult = 1.0

        # 신뢰도 조정
        conf_delta = 0.0
        notes = []

        if is_buy:
            conf_delta += 0.12
            notes.append(f"매수후보(기대{expected_ret:.0%})")

        if falling_knife > 0.6:
            conf_delta += 0.10
            notes.append(f"칼날잡기({falling_knife:.2f})")

        if is_over:
            conf_delta -= 0.10
            notes.append(f"고평가(채권형{bond_type:.2f},마진{safety_margin:.2f})")

        conf_delta = round(float(np.clip(conf_delta, -0.20, 0.20)), 2)

        return SeoJunsikSignal(
            bond_type_score=round(bond_type, 4),
            expected_return=round(expected_ret, 4),
            safety_margin_score=round(safety_margin, 4),
            falling_knife_score=round(falling_knife, 4),
            is_buy_candidate=is_buy,
            is_overvalued=is_over,
            position_multiplier=round(pos_mult, 2),
            confidence_delta=conf_delta,
            note=" | ".join(notes) if notes else "",
        )
