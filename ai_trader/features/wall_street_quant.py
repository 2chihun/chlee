"""영주 닐슨 "월스트리트 퀀트투자의 법칙" - 멀티팩터 퀀트 분석 모듈

월가 6조 원 운용 퀀트 전문가의 투자 프레임워크:
- 스마트베타 5팩터: 밸류/모멘텀/사이즈/퀄리티/저변동성
- 행동재무학 편향 감지: 과잉확신/손실회피/군집행동 등 7가지
- 팩터 리스크 분해: 체계적/비체계적 리스크 비율
- 다전략 결합: 상관관계 기반 최적 포트폴리오
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Signal
# ──────────────────────────────────────────────

@dataclass
class WallStreetQuantSignal:
    """월스트리트 퀀트 통합 시그널

    모든 점수는 0~1 범위 (1=최우수).
    """
    # 스마트베타 5팩터
    value_factor: float = 0.5          # 밸류 (저평가)
    momentum_factor: float = 0.5       # 모멘텀 (추세)
    size_factor: float = 0.5           # 사이즈 (소형주 프리미엄)
    quality_factor: float = 0.5        # 퀄리티 (수익 안정성)
    low_vol_factor: float = 0.5        # 저변동성 (안정성)
    # 행동편향
    behavioral_bias_score: float = 0.0  # 편향 강도 (0=없음, 1=극심)
    bias_types: str = ""               # 감지된 편향 유형
    # 팩터 리스크
    systematic_risk_ratio: float = 0.5  # 체계적 리스크 비율
    factor_concentration: float = 0.0   # 팩터 집중도 (높으면 위험)
    # 통합
    composite_score: float = 0.5       # 5팩터 통합 (0~1)
    strategy_alignment: float = 0.5    # 전략 정합성
    position_multiplier: float = 1.0   # 포지션 배수 (0.5~1.5)
    confidence_delta: float = 0.0      # 신뢰도 보정
    note: str = ""


# ──────────────────────────────────────────────
# Helper: 스마트베타 5팩터 스코어러
# ──────────────────────────────────────────────

class SmartBetaFactorScorer:
    """스마트베타 5대 팩터 점수 산출

    밸류, 모멘텀, 사이즈, 퀄리티, 저변동성을
    OHLCV 데이터 기반 기술적 프록시로 계산.
    """

    def __init__(self, lookback: int = 120):
        self.lookback = lookback

    def score_value(self, df: pd.DataFrame) -> float:
        """밸류 팩터: 가격 수준 대비 저평가도"""
        if len(df) < self.lookback:
            return 0.5
        close = df["close"].astype(float).tail(self.lookback)
        volume = df["volume"].astype(float).tail(self.lookback)
        price = close.iloc[-1]
        avg_price = close.mean()
        if avg_price <= 0 or price <= 0:
            return 0.5
        # 저평가 프록시: 현재가/평균가 역수
        ratio = avg_price / price
        # 거래량 밀도: 낮은 가격 + 높은 거래량 = 가치주 특성
        vol_density = volume.mean() / (price * 1000 + 1)
        val = 0.6 * min(ratio, 2.0) / 2.0 + 0.4 * min(vol_density, 1.0)
        return np.clip(val, 0.0, 1.0)

    def score_momentum(self, df: pd.DataFrame) -> float:
        """모멘텀 팩터: 다기간 수익률 기반"""
        if len(df) < self.lookback:
            return 0.5
        close = df["close"].astype(float)
        scores = []
        for period in [20, 60, 120]:
            if len(close) >= period:
                ret = (close.iloc[-1] / close.iloc[-period]) - 1
                # 수익률을 0~1로 변환 (-30%~+30% 범위)
                s = np.clip((ret + 0.3) / 0.6, 0.0, 1.0)
                scores.append(s)
        if not scores:
            return 0.5
        # 단기 > 장기 가중 (최근 모멘텀 중시)
        weights = [0.5, 0.3, 0.2][:len(scores)]
        w_sum = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / w_sum

    def score_size(self, df: pd.DataFrame) -> float:
        """사이즈 팩터: 소형주 프리미엄 (거래대금 프록시)"""
        if len(df) < 20:
            return 0.5
        close = df["close"].astype(float).tail(60)
        volume = df["volume"].astype(float).tail(60)
        # 평균 거래대금으로 시총 프록시
        avg_turnover = (close * volume).mean()
        if avg_turnover <= 0:
            return 0.5
        # 소형주일수록 높은 점수 (거래대금 낮을수록)
        # 로그 스케일로 정규화
        log_to = np.log10(avg_turnover + 1)
        # 대략 6(백만)~12(조) 범위에서 역수
        size_score = np.clip((12 - log_to) / 6, 0.0, 1.0)
        return size_score

    def score_quality(self, df: pd.DataFrame) -> float:
        """퀄리티 팩터: 수익 안정성 + 샤프비율 프록시"""
        if len(df) < self.lookback:
            return 0.5
        close = df["close"].astype(float).tail(self.lookback)
        returns = close.pct_change().dropna()
        if len(returns) < 20 or returns.std() == 0:
            return 0.5
        # 샤프비율 프록시
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        sharpe_score = np.clip((sharpe + 1) / 4, 0.0, 1.0)
        # 수익 안정성: 양의 수익 비율
        positive_ratio = (returns > 0).sum() / len(returns)
        return 0.6 * sharpe_score + 0.4 * positive_ratio

    def score_low_volatility(self, df: pd.DataFrame) -> float:
        """저변동성 팩터: 변동성 낮을수록 높은 점수"""
        if len(df) < self.lookback:
            return 0.5
        close = df["close"].astype(float).tail(self.lookback)
        returns = close.pct_change().dropna()
        if len(returns) < 20:
            return 0.5
        ann_vol = returns.std() * np.sqrt(252)
        # 연간변동성 0~60% 범위에서 역수
        vol_score = np.clip(1.0 - ann_vol / 0.6, 0.0, 1.0)
        return vol_score

    def score_all(self, df: pd.DataFrame) -> Dict[str, float]:
        return {
            "value": self.score_value(df),
            "momentum": self.score_momentum(df),
            "size": self.score_size(df),
            "quality": self.score_quality(df),
            "low_vol": self.score_low_volatility(df),
        }


# ──────────────────────────────────────────────
# Helper: 행동재무학 편향 감지
# ──────────────────────────────────────────────

class BehavioralBiasDetector:
    """행동재무학 7대 편향 감지

    OHLCV 데이터에서 시장 참가자의 비이성적 행동 패턴을 감지:
    - 과잉확신: 거래량 급증 + 변동성 확대
    - 손실회피: 하락 후 급격한 매도 증가
    - 군집행동: 거래량 이상 급등
    - 확증편향: 추세 과도 지속
    - 매몰비용: 하락에도 거래량 감소 (손절 회피)
    - 프레이밍: 정수 가격 근처 집중
    - 심리계좌: 특정 가격대 저항/지지
    """

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def detect(self, df: pd.DataFrame) -> Dict[str, float]:
        if len(df) < self.lookback:
            return {"total": 0.0, "types": ""}

        close = df["close"].astype(float).tail(self.lookback)
        volume = df["volume"].astype(float).tail(self.lookback)
        returns = close.pct_change().dropna()
        vol_ratio = volume / volume.rolling(20).mean()

        biases = {}

        # 1. 과잉확신: 높은 거래량 + 높은 변동성
        recent_vol_ratio = vol_ratio.tail(5).mean() if len(vol_ratio) >= 5 else 1.0
        recent_volatility = returns.tail(10).std() if len(returns) >= 10 else 0
        avg_volatility = returns.std() if len(returns) > 0 else 0.01
        overconf = 0.0
        if avg_volatility > 0:
            overconf = min((recent_vol_ratio - 1) * 0.3 + (recent_volatility / avg_volatility - 1) * 0.3, 1.0)
        biases["overconfidence"] = max(overconf, 0.0)

        # 2. 손실회피: 하락일 거래량 비율 vs 상승일
        neg_days = returns[returns < 0]
        pos_days = returns[returns >= 0]
        if len(pos_days) > 0 and len(neg_days) > 0:
            neg_vol = volume.iloc[-len(returns):][returns < 0].mean()
            pos_vol = volume.iloc[-len(returns):][returns >= 0].mean()
            if pos_vol > 0:
                loss_aversion = np.clip((neg_vol / pos_vol - 1) * 0.5, 0.0, 1.0)
            else:
                loss_aversion = 0.0
        else:
            loss_aversion = 0.0
        biases["loss_aversion"] = loss_aversion

        # 3. 군집행동: 최근 거래량이 평균의 2배 이상
        herd_score = np.clip((recent_vol_ratio - 2) * 0.5, 0.0, 1.0) if not np.isnan(recent_vol_ratio) else 0.0
        biases["herding"] = herd_score

        # 4. 확증편향: 추세 과도 지속 (연속 상승/하락일 비율)
        if len(returns) >= 10:
            recent = returns.tail(10)
            streak = (recent > 0).sum() / len(recent)
            confirm_bias = abs(streak - 0.5) * 2  # 0.5에서 멀수록 편향
            biases["confirmation_bias"] = min(confirm_bias, 1.0)
        else:
            biases["confirmation_bias"] = 0.0

        # 5. 매몰비용: 하락 추세 + 거래량 감소 (손절 회피)
        if len(returns) >= 20:
            price_trend = (close.iloc[-1] / close.iloc[-20]) - 1
            vol_trend = (volume.tail(5).mean() / volume.tail(20).mean()) - 1 if volume.tail(20).mean() > 0 else 0
            sunk_cost = max(-price_trend * 2, 0) * max(-vol_trend, 0)
            biases["sunk_cost"] = min(sunk_cost, 1.0)
        else:
            biases["sunk_cost"] = 0.0

        # 총합 (가중평균)
        weights = {
            "overconfidence": 0.25, "loss_aversion": 0.20,
            "herding": 0.20, "confirmation_bias": 0.15, "sunk_cost": 0.20,
        }
        total = sum(biases.get(k, 0) * w for k, w in weights.items())

        # 감지된 편향 유형
        detected = [k for k, v in biases.items() if v > 0.3]
        types_str = ",".join(detected) if detected else ""

        return {"total": min(total, 1.0), "types": types_str, **biases}


# ──────────────────────────────────────────────
# Helper: 팩터 리스크 분해
# ──────────────────────────────────────────────

class FactorRiskDecomposer:
    """팩터 리스크 분해: 체계적 vs 비체계적

    시장 수익률 프록시 대비 개별 종목의 베타를 추정하고,
    체계적/비체계적 리스크 비율을 산출.
    """

    def __init__(self, lookback: int = 120):
        self.lookback = lookback

    def decompose(self, df: pd.DataFrame) -> Dict[str, float]:
        if len(df) < self.lookback:
            return {"systematic_ratio": 0.5, "beta": 1.0, "concentration": 0.0}

        close = df["close"].astype(float).tail(self.lookback)
        returns = close.pct_change().dropna()
        if len(returns) < 30:
            return {"systematic_ratio": 0.5, "beta": 1.0, "concentration": 0.0}

        # 시장 프록시: 이동평균 수익률
        market_proxy = returns.rolling(20).mean().dropna()
        stock_returns = returns.tail(len(market_proxy))

        if len(market_proxy) < 20 or market_proxy.var() == 0:
            return {"systematic_ratio": 0.5, "beta": 1.0, "concentration": 0.0}

        # 베타 추정
        cov = np.cov(stock_returns.values, market_proxy.values)
        if cov.shape == (2, 2) and cov[1, 1] > 0:
            beta = cov[0, 1] / cov[1, 1]
        else:
            beta = 1.0

        # 체계적 리스크 비율
        total_var = returns.var()
        if total_var > 0:
            systematic_var = (beta ** 2) * market_proxy.var()
            systematic_ratio = np.clip(systematic_var / total_var, 0.0, 1.0)
        else:
            systematic_ratio = 0.5

        # 팩터 집중도: 한 방향으로의 쏠림 정도
        concentration = abs(beta - 1.0) / 2.0
        concentration = min(concentration, 1.0)

        return {
            "systematic_ratio": systematic_ratio,
            "beta": beta,
            "concentration": concentration,
        }


# ──────────────────────────────────────────────
# Helper: 다전략 결합기
# ──────────────────────────────────────────────

class StrategyCombiner:
    """다전략 상관관계 기반 결합기

    영주 닐슨: "달걀을 한 바구니에 담지 마라"
    - 최소 2~3개 전략 동시 운용
    - 상관관계가 낮은 전략끼리 결합 → 손실 상쇄 효과
    - 예측수익률 비례 또는 균등 배분

    사용:
        combiner = StrategyCombiner()
        weights = combiner.combine(strategy_scores)
    """

    def __init__(
        self,
        min_weight: float = 0.05,
        max_weight: float = 0.60,
        lookback: int = 120,
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.lookback = lookback

    def combine(
        self,
        strategy_scores: Dict[str, float],
        strategy_returns: Optional[Dict[str, pd.Series]] = None,
    ) -> Dict[str, float]:
        """전략별 최적 가중치를 산출합니다.

        Args:
            strategy_scores: 전략명 → 현재 신호 점수 (0~1)
            strategy_returns: 전략명 → 과거 수익률 시리즈 (상관관계 계산용)
                              None이면 점수 비례 균등 배분

        Returns:
            전략명 → 최적 가중치 (합계 1.0)
        """
        names = list(strategy_scores.keys())
        n = len(names)

        if n == 0:
            return {}
        if n == 1:
            return {names[0]: 1.0}

        # ── 1단계: 기본 가중치 (점수 비례) ──
        scores = np.array([max(strategy_scores[k], 0.01) for k in names])
        base_weights = scores / scores.sum()

        # ── 2단계: 상관관계 기반 분산 보정 ──
        if strategy_returns is not None and len(strategy_returns) >= 2:
            corr_adj = self._correlation_adjustment(names, strategy_returns)
            # 상관관계 보정과 점수 비례를 50:50 결합
            combined = 0.5 * base_weights + 0.5 * corr_adj
        else:
            combined = base_weights

        # ── 3단계: 가중치 제약 적용 ──
        combined = self._apply_constraints(combined)

        return {name: round(float(w), 4) for name, w in zip(names, combined)}

    def _correlation_adjustment(
        self,
        names: List[str],
        strategy_returns: Dict[str, pd.Series],
    ) -> np.ndarray:
        """상관관계 역수 기반 가중치 조정

        상관관계가 낮은 전략에 높은 가중치를 부여하여
        포트폴리오 분산 효과를 극대화합니다.
        """
        n = len(names)
        # 수익률 DataFrame 구성
        ret_dict = {}
        for name in names:
            if name in strategy_returns:
                s = strategy_returns[name].tail(self.lookback)
                ret_dict[name] = s
        if len(ret_dict) < 2:
            return np.ones(n) / n

        ret_df = pd.DataFrame(ret_dict).dropna()
        if len(ret_df) < 10:
            return np.ones(n) / n

        # 상관계수 행렬
        corr_matrix = ret_df.corr().values.copy()
        np.fill_diagonal(corr_matrix, 0.0)

        # 각 전략의 평균 상관관계 (다른 전략들과)
        avg_corr = np.abs(corr_matrix).mean(axis=1)

        # 상관관계 역수 → 낮은 상관 전략에 높은 가중치
        inv_corr = 1.0 / (avg_corr + 0.1)  # 0.1 = 안정화 상수

        # 누락된 전략은 균등 가중치
        weights = np.ones(n) / n
        col_names = list(ret_df.columns)
        for i, name in enumerate(names):
            if name in col_names:
                idx = col_names.index(name)
                weights[i] = inv_corr[idx]

        return weights / weights.sum()

    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """최소/최대 가중치 제약 적용"""
        n = len(weights)
        if n == 0:
            return weights

        # 최소 가중치 보장
        weights = np.maximum(weights, self.min_weight)
        # 최대 가중치 제한
        weights = np.minimum(weights, self.max_weight)
        # 재정규화
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            weights = np.ones(n) / n
        return weights

    def evaluate_diversification(
        self,
        strategy_returns: Dict[str, pd.Series],
    ) -> Dict[str, float]:
        """전략 간 분산 효과를 평가합니다.

        Returns:
            avg_correlation: 전략 간 평균 상관관계 (낮을수록 좋음)
            diversification_ratio: 분산 효과 비율 (높을수록 좋음)
            effective_strategies: 유효 전략 수 (HHI 역수)
        """
        if len(strategy_returns) < 2:
            return {
                "avg_correlation": 0.0,
                "diversification_ratio": 1.0,
                "effective_strategies": float(len(strategy_returns)),
            }

        ret_df = pd.DataFrame(strategy_returns).dropna()
        if len(ret_df) < 10:
            return {
                "avg_correlation": 0.0,
                "diversification_ratio": 1.0,
                "effective_strategies": float(len(strategy_returns)),
            }

        corr_matrix = ret_df.corr().values
        n = corr_matrix.shape[0]

        # 평균 상관관계 (대각선 제외)
        mask = ~np.eye(n, dtype=bool)
        avg_corr = float(np.abs(corr_matrix[mask]).mean())

        # 분산 효과 비율: 개별 변동성 합 / 포트폴리오 변동성
        individual_vols = ret_df.std().values
        # 균등 가중치 포트폴리오
        eq_weights = np.ones(n) / n
        cov_matrix = ret_df.cov().values
        port_var = eq_weights @ cov_matrix @ eq_weights
        port_vol = np.sqrt(port_var) if port_var > 0 else 1e-10
        weighted_vol_sum = (eq_weights * individual_vols).sum()
        div_ratio = weighted_vol_sum / port_vol if port_vol > 0 else 1.0

        # 유효 전략 수 (HHI 역수)
        hhi = float((eq_weights ** 2).sum())
        effective_n = 1.0 / hhi if hhi > 0 else float(n)

        return {
            "avg_correlation": round(avg_corr, 4),
            "diversification_ratio": round(float(div_ratio), 4),
            "effective_strategies": round(effective_n, 2),
        }


# ──────────────────────────────────────────────
# Main Analyzer
# ──────────────────────────────────────────────

class WallStreetQuantAnalyzer:
    """월스트리트 퀀트 통합 분석기

    영주 닐슨의 퀀트 프레임워크를 적용:
    1. 스마트베타 5팩터 스코어링
    2. 행동재무학 편향 감지 → 포지션 보정
    3. 팩터 리스크 분해 → 리스크 인식
    4. 전략 정합성 평가
    """

    # 팩터 가중치 (스마트베타 기본)
    FACTOR_WEIGHTS = {
        "value": 0.25,
        "momentum": 0.25,
        "quality": 0.20,
        "low_vol": 0.15,
        "size": 0.15,
    }

    def __init__(self, lookback: int = 120):
        self.factor_scorer = SmartBetaFactorScorer(lookback=lookback)
        self.bias_detector = BehavioralBiasDetector(lookback=min(lookback, 60))
        self.risk_decomposer = FactorRiskDecomposer(lookback=lookback)

    def analyze(self, df: pd.DataFrame) -> WallStreetQuantSignal:
        if df is None or len(df) < 30:
            return WallStreetQuantSignal(note="insufficient data")

        # 1. 스마트베타 5팩터 스코어링
        factors = self.factor_scorer.score_all(df)

        # 2. 행동편향 감지
        bias_result = self.bias_detector.detect(df)
        bias_total = bias_result.get("total", 0.0)
        bias_types = bias_result.get("types", "")

        # 3. 팩터 리스크 분해
        risk = self.risk_decomposer.decompose(df)

        # 4. 복합 점수 계산 (가중평균)
        composite = sum(
            factors.get(k, 0.5) * w
            for k, w in self.FACTOR_WEIGHTS.items()
        )

        # 5. 전략 정합성: 팩터 간 일관성 (표준편차 역수)
        factor_values = list(factors.values())
        factor_std = np.std(factor_values) if len(factor_values) > 1 else 0
        strategy_alignment = np.clip(1.0 - factor_std * 2, 0.0, 1.0)

        # 6. 포지션 배수: 편향이 높으면 축소
        bias_penalty = bias_total * 0.5  # 최대 50% 감소
        base_mult = 0.5 + composite  # 0.5 ~ 1.5
        position_mult = np.clip(base_mult - bias_penalty, 0.5, 1.5)

        # 7. 신뢰도 보정
        conf_delta = (composite - 0.5) * 0.1 * strategy_alignment

        # 노트 생성
        notes = []
        if composite >= 0.7:
            notes.append("strong multi-factor")
        elif composite <= 0.3:
            notes.append("weak factors")
        if bias_total > 0.5:
            notes.append(f"high bias({bias_types})")
        if risk.get("concentration", 0) > 0.5:
            notes.append("concentrated risk")

        return WallStreetQuantSignal(
            value_factor=factors.get("value", 0.5),
            momentum_factor=factors.get("momentum", 0.5),
            size_factor=factors.get("size", 0.5),
            quality_factor=factors.get("quality", 0.5),
            low_vol_factor=factors.get("low_vol", 0.5),
            behavioral_bias_score=bias_total,
            bias_types=bias_types,
            systematic_risk_ratio=risk.get("systematic_ratio", 0.5),
            factor_concentration=risk.get("concentration", 0.0),
            composite_score=composite,
            strategy_alignment=strategy_alignment,
            position_multiplier=round(position_mult, 3),
            confidence_delta=round(conf_delta, 4),
            note="; ".join(notes) if notes else "neutral",
        )
