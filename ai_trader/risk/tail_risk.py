"""나심 탈레브 'Skin in the Game' 핵심 리스크 관리 모듈

핵심 개념:
- Ergodicity: 앙상블 확률 != 시간 확률, 흡수 장벽(ruin) 존재 시 기대값 왜곡
- Ruin Probability: 반복 노출 시 파멸 확률 1로 수렴
- Fat-tail Detection: 정규분포 가정 탈피, 부분지수 분포 판별
- CVaR: 극단 손실의 조건부 기대값
- Bob Rubin Trade: 비대칭 보상 구조 탐지
- Precautionary Principle: 꼬리 위험 시 비용편익 분석 무효화
- Barbell Strategy: 90% 초보수 + 10% 고위험 배분
- Lindy Filter: 오래 검증된 전략/지표에 높은 가중치
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


# ─────────────────────────────────────────────
# Dataclass
# ─────────────────────────────────────────────

@dataclass
class TalebRiskSignal:
    """탈레브 리스크 통합 시그널"""
    # Ergodicity
    is_ergodic: bool = True
    ensemble_mean: float = 0.0
    time_mean: float = 0.0
    ergodicity_ratio: float = 1.0  # time/ensemble, <1 = non-ergodic

    # Ruin
    ruin_probability: float = 0.0  # 0~1
    expected_ruin_time: float = float('inf')  # periods
    ruin_safe: bool = True  # True if ruin prob < threshold

    # Fat tail
    is_fat_tailed: bool = False
    tail_index: float = 0.0  # Hill estimator alpha
    kurtosis_excess: float = 0.0

    # CVaR
    var_95: float = 0.0
    cvar_95: float = 0.0
    var_99: float = 0.0
    cvar_99: float = 0.0

    # Bob Rubin
    bob_rubin_score: float = 0.0  # 0~1, high = suspicious
    skewness: float = 0.0

    # Precautionary
    precautionary_block: bool = False  # True = block trading

    # Position adjustment
    position_scale: float = 1.0  # 0~1 multiplier

    # Barbell
    barbell_conservative_pct: float = 90.0
    barbell_aggressive_pct: float = 10.0


# ─────────────────────────────────────────────
# 1. ErgodicityChecker
# ─────────────────────────────────────────────

class ErgodicityChecker:
    """에르고딕성 검증기

    Ch.19 핵심: 100명이 카지노에 가는 것(앙상블) vs
    1명이 100일 가는 것(시간) -- 결과가 다르다.

    Theorem 1: E_n(X) >= E_T(X)
    시간 경로에서 흡수 장벽(ruin)이 존재하면
    앙상블 기대값이 시간 기대값보다 항상 크거나 같다.
    """

    def __init__(
        self,
        ruin_threshold: float = -0.20,
        n_simulations: int = 1000,
        n_periods: int = 252,
    ):
        self.ruin_threshold = ruin_threshold  # -20% = uncle point
        self.n_simulations = n_simulations
        self.n_periods = n_periods

    def check(self, returns: pd.Series) -> Dict[str, float]:
        """수익률 시계열의 에르고딕성 검증

        Monte Carlo 시뮬레이션으로 앙상블 평균과
        시간 평균(생존 경로만) 비교.

        Returns:
            dict with ensemble_mean, time_mean,
            ergodicity_ratio, survival_rate, is_ergodic
        """
        if len(returns) < 20:
            return {
                "ensemble_mean": 0.0,
                "time_mean": 0.0,
                "ergodicity_ratio": 1.0,
                "survival_rate": 1.0,
                "is_ergodic": True,
            }

        mu = returns.mean()
        sigma = returns.std()
        if sigma < 1e-10:
            return {
                "ensemble_mean": mu,
                "time_mean": mu,
                "ergodicity_ratio": 1.0,
                "survival_rate": 1.0,
                "is_ergodic": True,
            }

        # Bootstrap from actual returns
        n_ret = len(returns)
        rng = np.random.default_rng(42)

        survived = 0
        ensemble_finals: List[float] = []
        time_finals: List[float] = []

        n_periods = min(self.n_periods, n_ret * 2)

        for _ in range(self.n_simulations):
            idx = rng.integers(0, n_ret, size=n_periods)
            path_returns = returns.values[idx]

            # 누적 자산 (곱셈적 성장)
            wealth = np.cumprod(1 + path_returns)

            # 앙상블: 경로 무관 최종 자산
            ensemble_finals.append(wealth[-1])

            # 시간: 파멸(uncle point) 도달 여부 확인
            drawdown = (
                wealth / np.maximum.accumulate(wealth) - 1
            )
            hit_ruin = np.any(drawdown <= self.ruin_threshold)

            if not hit_ruin:
                time_finals.append(wealth[-1])
                survived += 1

        ensemble_mean = float(np.mean(ensemble_finals))

        if time_finals:
            time_mean = float(np.mean(time_finals))
        else:
            time_mean = 0.0

        survival_rate = survived / self.n_simulations

        denom = max(ensemble_mean, 1e-10)
        ratio = (
            time_mean / denom if ensemble_mean > 0 else 0.0
        )

        # 비에르고딕 판정: ratio<0.8 또는 생존율<70%
        is_ergodic = ratio >= 0.8 and survival_rate >= 0.7

        return {
            "ensemble_mean": round(ensemble_mean, 6),
            "time_mean": round(time_mean, 6),
            "ergodicity_ratio": round(ratio, 4),
            "survival_rate": round(survival_rate, 4),
            "is_ergodic": is_ergodic,
        }


# ─────────────────────────────────────────────
# 2. RuinProbEstimator
# ─────────────────────────────────────────────

class RuinProbEstimator:
    """파멸 확률 추정기

    Appendix C: 반복 노출 시 파멸 확률
    E(tau) = 1/(n*p) where n=노출빈도, p=기간별 파멸확률

    "아무리 작은 확률이라도 반복하면 1로 수렴"
    "파멸은 재생 불가능한 자원"
    """

    def __init__(
        self,
        max_ruin_prob: float = 0.01,
        horizon_periods: int = 252 * 10,
    ):
        """
        Args:
            max_ruin_prob: 허용 최대 파멸 확률 (기본 1%)
            horizon_periods: 투자 기간 (기본 10년, 일봉)
        """
        self.max_ruin_prob = max_ruin_prob
        self.horizon_periods = horizon_periods

    def estimate(
        self,
        returns: pd.Series,
        ruin_level: float = -0.30,
    ) -> Dict[str, float]:
        """파멸 확률 추정

        Args:
            returns: 일별 수익률
            ruin_level: 파멸 기준 (기본 -30% 최대 낙폭)

        Returns:
            dict with ruin_prob_per_period,
            cumulative_ruin_prob, expected_ruin_time,
            is_safe, max_safe_exposure
        """
        if len(returns) < 30:
            return self._safe_default()

        # 기간별 파멸 확률 추정 (historical)
        cum_ret = (1 + returns).cumprod()
        rolling_max = cum_ret.cummax()
        drawdowns = cum_ret / rolling_max - 1

        # 일별 파멸 확률
        n_ruin_events = (drawdowns <= ruin_level).sum()
        p_ruin_daily = n_ruin_events / len(returns)

        if p_ruin_daily < 1e-10:
            # 관측된 파멸 없음 -> EVT로 추정
            p_ruin_daily = self._evt_ruin_estimate(
                returns, ruin_level
            )

        # 누적 파멸 확률: P(ruin in T) = 1 - (1-p)^T
        cum_ruin = (
            1 - (1 - p_ruin_daily) ** self.horizon_periods
        )

        # 파멸까지 기대 시간
        if p_ruin_daily > 0:
            expected_time = 1.0 / p_ruin_daily
        else:
            expected_time = float('inf')

        is_safe = cum_ruin <= self.max_ruin_prob

        # 안전한 최대 노출 횟수
        if p_ruin_daily > 0:
            max_n = (
                np.log(1 - self.max_ruin_prob)
                / np.log(1 - p_ruin_daily)
            )
            max_safe = max(0, int(max_n))
        else:
            max_safe = self.horizon_periods

        return {
            "ruin_prob_per_period": round(p_ruin_daily, 8),
            "cumulative_ruin_prob": round(cum_ruin, 6),
            "expected_ruin_time": round(expected_time, 1),
            "is_safe": is_safe,
            "max_safe_exposure": max_safe,
        }

    def _evt_ruin_estimate(
        self,
        returns: pd.Series,
        ruin_level: float,
    ) -> float:
        """극단값 이론(EVT) 기반 파멸 확률 추정

        관측 데이터에 파멸 이벤트가 없을 때,
        꼬리 분포를 추정하여 확률 계산.
        """
        losses = -returns[returns < 0]
        if len(losses) < 10:
            return 0.0

        # GPD (Generalized Pareto Distribution) 적합
        threshold = losses.quantile(0.90)
        exceedances = losses[losses > threshold] - threshold

        if len(exceedances) < 5:
            return 0.0

        try:
            shape, _, scale = stats.genpareto.fit(
                exceedances, floc=0
            )
            target = abs(ruin_level)
            if target <= threshold:
                return float((losses >= target).mean())
            excess = target - threshold
            tail_prob = len(exceedances) / len(losses)
            surv = stats.genpareto.sf(
                excess, shape, 0, scale
            )
            return float(tail_prob * surv)
        except Exception:
            return 0.0

    def _safe_default(self) -> Dict[str, float]:
        return {
            "ruin_prob_per_period": 0.0,
            "cumulative_ruin_prob": 0.0,
            "expected_ruin_time": float('inf'),
            "is_safe": True,
            "max_safe_exposure": self.horizon_periods,
        }


# ─────────────────────────────────────────────
# 3. FatTailDetector
# ─────────────────────────────────────────────

class FatTailDetector:
    """Fat-tail 분포 검정기

    Appendix D: 부분지수(Subexponential) 분포 판별
    - 정규분포 가정 여부 검정
    - Hill estimator로 tail index 추정
    - 초과 첨도(excess kurtosis)로 fat-tail 판단

    분포 스펙트럼:
    Compact -> Subgaussian -> Gaussian
    -> Subexponential -> Power law
    Mediocristan <--경계--> Extremistan
    """

    def __init__(self, tail_pct: float = 0.05):
        """
        Args:
            tail_pct: Hill estimator용 꼬리 비율 (5%)
        """
        self.tail_pct = tail_pct

    def detect(
        self, returns: pd.Series
    ) -> Dict[str, float]:
        """Fat-tail 검정 실행

        Returns:
            dict with is_fat_tailed, tail_index,
            kurtosis_excess, jarque_bera_pvalue,
            normality_rejected, tail_ratio
        """
        if len(returns) < 50:
            return self._default()

        clean = returns.dropna()

        # 1. 초과 첨도 (정규=0, fat-tail>0)
        kurt = float(stats.kurtosis(clean, fisher=True))

        # 2. 왜도
        skew = float(stats.skew(clean))

        # 3. Jarque-Bera 정규성 검정
        jb_stat, jb_pval = stats.jarque_bera(clean)
        normality_rejected = jb_pval < 0.05

        # 4. Hill estimator (tail index)
        tail_idx = self._hill_estimator(clean)

        # 5. 꼬리 비율: 실제 극단값 vs 정규분포 예측
        tail_ratio = self._tail_ratio(clean)

        # Fat-tail 판정
        is_fat = (
            (kurt > 1.0)
            or (normality_rejected and tail_ratio > 1.5)
        )

        return {
            "is_fat_tailed": is_fat,
            "tail_index": round(tail_idx, 3),
            "kurtosis_excess": round(kurt, 3),
            "skewness": round(skew, 3),
            "jarque_bera_pvalue": round(float(jb_pval), 6),
            "normality_rejected": normality_rejected,
            "tail_ratio": round(tail_ratio, 3),
        }

    def _hill_estimator(self, data: pd.Series) -> float:
        """Hill estimator로 tail index(alpha) 추정

        Power law: P(X>x) ~ x^(-alpha)
        alpha가 작을수록 꼬리가 두꺼움.
        alpha <= 2: 무한 분산
        alpha <= 1: 무한 평균
        """
        abs_data = np.abs(data.values)
        abs_data = abs_data[abs_data > 0]
        n = len(abs_data)
        k = max(int(n * self.tail_pct), 5)

        if k >= n:
            return 0.0

        sorted_data = np.sort(abs_data)[::-1]  # 내림차순
        top_k = sorted_data[:k]
        threshold = sorted_data[k]

        if threshold <= 0:
            return 0.0

        log_ratios = np.log(top_k / threshold)
        mean_log = log_ratios.mean()

        if mean_log <= 0:
            return 0.0

        return 1.0 / mean_log  # alpha = 1/mean(log(Xi/Xk))

    def _tail_ratio(self, data: pd.Series) -> float:
        """실제 극단값 빈도 vs 정규분포 예측 비율

        정규분포에서 3sigma 이상 = 0.27%
        실제 데이터에서 3sigma 이상 비율 / 0.27%
        ratio > 1이면 fat-tail
        """
        mu = data.mean()
        sigma = data.std()
        if sigma < 1e-10:
            return 1.0

        threshold = 3.0 * sigma
        actual_pct = (np.abs(data - mu) > threshold).mean()
        expected_pct = 2 * stats.norm.sf(3.0)  # ~0.0027

        if expected_pct < 1e-10:
            return 1.0

        return float(actual_pct / expected_pct)

    def _default(self) -> Dict[str, float]:
        return {
            "is_fat_tailed": False,
            "tail_index": 0.0,
            "kurtosis_excess": 0.0,
            "skewness": 0.0,
            "jarque_bera_pvalue": 1.0,
            "normality_rejected": False,
            "tail_ratio": 1.0,
        }


# ─────────────────────────────────────────────
# 4. CVaRCalculator
# ─────────────────────────────────────────────

class CVaRCalculator:
    """Conditional VaR (Expected Shortfall) 계산기

    VaR가 "최대 손실"이라면,
    CVaR는 "VaR를 초과했을 때 평균 손실"
    -> 꼬리 위험의 실질적 크기 측정
    """

    @staticmethod
    def calculate(
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> Dict[str, float]:
        """VaR/CVaR 계산

        Args:
            returns: 수익률 시계열
            confidence: 신뢰 수준 (0.95 또는 0.99)

        Returns:
            dict with var, cvar (positive loss values)
        """
        if len(returns) < 20:
            return {"var": 0.0, "cvar": 0.0}

        clean = returns.dropna().values
        alpha = 1 - confidence

        # Historical VaR
        var_val = float(np.percentile(clean, alpha * 100))

        # CVaR = mean of losses beyond VaR
        tail_losses = clean[clean <= var_val]
        if len(tail_losses) > 0:
            cvar_val = float(tail_losses.mean())
        else:
            cvar_val = var_val

        return {
            "var": round(abs(var_val), 6),
            "cvar": round(abs(cvar_val), 6),
        }


# ─────────────────────────────────────────────
# 5. BobRubinDetector
# ─────────────────────────────────────────────

class BobRubinDetector:
    """Bob Rubin Trade 탐지기

    Appendix A: 안정적 수익 + 숨겨진 꼬리 위험 패턴
    - 높은 승률 + 음의 왜도 = 위험 신호
    - 보상 기간 < 꼬리 위험 사이클 = 구조적 문제
    - 에이전트 보수는 m(총 기대값)이 아닌
      음의 왜도에 의존
    """

    def __init__(self, lookback: int = 252):
        self.lookback = lookback

    def detect(
        self, returns: pd.Series
    ) -> Dict[str, float]:
        """Bob Rubin Trade 패턴 탐지

        Returns:
            dict with bob_rubin_score (0~1), win_rate,
            skewness, sharpe, max_loss_vs_avg_gain,
            suspicious
        """
        if len(returns) < 60:
            return self._default()

        recent = returns.tail(self.lookback)
        clean = recent.dropna()

        # 승률
        win_rate = (clean > 0).mean()

        # 왜도 (음의 왜도 = 위험)
        skew = float(stats.skew(clean))

        # Sharpe ratio
        mu = clean.mean()
        sigma = clean.std()
        sharpe = (
            (mu / sigma * np.sqrt(252))
            if sigma > 1e-10
            else 0
        )

        # 최대 손실 vs 평균 이익 비율
        gains = clean[clean > 0]
        losses = clean[clean < 0]

        if len(gains) > 0 and len(losses) > 0:
            avg_gain = gains.mean()
            max_loss = abs(losses.min())
            ratio = (
                max_loss / avg_gain
                if avg_gain > 1e-10
                else 0
            )
        else:
            ratio = 0.0

        # Bob Rubin 점수 계산 (0~1)
        score = 0.0

        # 높은 승률 (>70%) -> 의심
        if win_rate > 0.70:
            score += 0.25 * min(
                (win_rate - 0.70) / 0.20, 1.0
            )

        # 음의 왜도 -> 위험
        if skew < -0.5:
            score += 0.30 * min(abs(skew) / 2.0, 1.0)

        # 높은 Sharpe + 음의 왜도 -> 매우 의심
        if sharpe > 2.0 and skew < 0:
            score += 0.25 * min(
                (sharpe - 2.0) / 3.0, 1.0
            )

        # 최대 손실이 평균 이익의 10배 이상
        if ratio > 10:
            score += 0.20 * min((ratio - 10) / 20, 1.0)

        score = min(score, 1.0)

        return {
            "bob_rubin_score": round(score, 4),
            "win_rate": round(float(win_rate), 4),
            "skewness": round(skew, 4),
            "sharpe": round(float(sharpe), 4),
            "max_loss_vs_avg_gain": round(ratio, 2),
            "suspicious": score >= 0.5,
        }

    def _default(self) -> Dict[str, float]:
        return {
            "bob_rubin_score": 0.0,
            "win_rate": 0.5,
            "skewness": 0.0,
            "sharpe": 0.0,
            "max_loss_vs_avg_gain": 1.0,
            "suspicious": False,
        }


# ─────────────────────────────────────────────
# 6. PrecautionaryFilter
# ─────────────────────────────────────────────

class PrecautionaryFilter:
    """예방 원칙 필터

    Ch.19: "파멸이 존재하면 비용편익 분석 불가"
    "평균 4피트 깊이인 강을 건너지 마라"

    multiplicative/systemic/fat-tail 위험이 감지되면
    포지션을 차단하거나 대폭 축소.
    """

    def __init__(
        self,
        max_ruin_prob: float = 0.01,
        max_cvar_pct: float = 0.10,
        fat_tail_scale: float = 0.5,
    ):
        """
        Args:
            max_ruin_prob: 허용 최대 파멸 확률
            max_cvar_pct: 허용 최대 CVaR (자본 대비 %)
            fat_tail_scale: fat-tail 시 포지션 축소 배수
        """
        self.max_ruin_prob = max_ruin_prob
        self.max_cvar_pct = max_cvar_pct
        self.fat_tail_scale = fat_tail_scale

    def evaluate(
        self,
        ruin_result: Dict,
        fat_tail_result: Dict,
        cvar_result: Dict,
    ) -> Dict[str, float]:
        """예방 원칙 적용 판단

        Returns:
            dict with precautionary_block,
            position_scale, reasons
        """
        block = False
        scale = 1.0
        reasons: List[str] = []

        # 1. 파멸 확률 초과
        cum_ruin = ruin_result.get(
            "cumulative_ruin_prob", 0
        )
        if cum_ruin > self.max_ruin_prob:
            block = True
            reasons.append(
                f"ruin_prob {cum_ruin:.4f} "
                f"> {self.max_ruin_prob}"
            )
        elif cum_ruin > self.max_ruin_prob * 0.5:
            scale *= 0.5
            reasons.append(
                f"ruin_prob {cum_ruin:.4f} "
                f"approaching limit"
            )

        # 2. Fat-tail 감지
        if fat_tail_result.get("is_fat_tailed", False):
            scale *= self.fat_tail_scale
            reasons.append("fat_tail detected")

            # 극단적 fat-tail (첨도 > 5)
            kurt = fat_tail_result.get(
                "kurtosis_excess", 0
            )
            if kurt > 5.0:
                scale *= 0.5
                reasons.append(
                    f"extreme_kurtosis {kurt:.1f}"
                )

        # 3. CVaR 과다
        cvar_95 = cvar_result.get("cvar", 0)
        if cvar_95 > self.max_cvar_pct:
            scale *= max(
                self.max_cvar_pct / cvar_95, 0.3
            )
            reasons.append(
                f"cvar_95 {cvar_95:.4f} "
                f"> {self.max_cvar_pct}"
            )

        # 최소 스케일 제한
        scale = max(scale, 0.1)

        if block:
            scale = 0.0

        return {
            "precautionary_block": block,
            "position_scale": round(scale, 4),
            "reasons": reasons,
        }


# ─────────────────────────────────────────────
# 7. LindyFilter
# ─────────────────────────────────────────────

class LindyFilter:
    """린디 효과 기반 전략/지표 가중치

    Ch.15: 생존 시간이 길수록 앞으로도 오래 생존
    - 100년 된 전략(이동평균) > 5년 된 anomaly
    - 시간 = 최고의 필터
    """

    # 전략/지표별 알려진 역사 (년)
    DEFAULT_AGES: Dict[str, int] = {
        "sma": 80,              # 이동평균 (1940s~)
        "ema": 60,              # 지수이동평균
        "rsi": 46,              # Wilder 1978
        "macd": 46,             # Appel 1979
        "bollinger": 43,        # Bollinger 1983
        "momentum": 100,        # 모멘텀 (1920s~)
        "value": 90,            # 가치투자 (Graham 1934~)
        "mean_reversion": 100,  # 평균회귀
        "kelly": 70,            # Kelly 1956
        "trend_following": 150, # 추세추종 (1870s~)
        "pairs_trading": 35,    # 1990s~
        "ml_alpha": 15,         # ML 기반 알파
        "deep_learning": 10,    # 딥러닝
        "sentiment": 20,        # 센티먼트 분석
    }

    def __init__(
        self,
        custom_ages: Optional[Dict[str, int]] = None,
    ):
        self.ages = {**self.DEFAULT_AGES}
        if custom_ages:
            self.ages.update(custom_ages)

    def weight(
        self, strategies: List[str]
    ) -> Dict[str, float]:
        """전략 목록에 린디 가중치 부여

        가중치 = log(1 + age) / sum(log(1 + ages))
        오래된 전략이 로그적으로 더 높은 가중치

        Returns:
            dict of strategy_name -> weight (sum=1.0)
        """
        if not strategies:
            return {}

        raw: Dict[str, float] = {}
        for s in strategies:
            age = self.ages.get(s, 5)  # 기본 5년
            raw[s] = np.log1p(age)

        total = sum(raw.values())
        if total < 1e-10:
            n = len(strategies)
            return {s: 1.0 / n for s in strategies}

        return {
            s: round(v / total, 4)
            for s, v in raw.items()
        }


# ─────────────────────────────────────────────
# 8. BarbellAllocator
# ─────────────────────────────────────────────

class BarbellAllocator:
    """바벨 전략 배분기

    Ch.19: 90% 극보수(현금/국채) + 10% 극공격(고위험)
    중간 위험은 피하라 -- 숨겨진 꼬리 위험 가능성

    시장 상태에 따라 비율 동적 조정:
    - 고변동성/위기: 95/5 (더 보수적)
    - 저변동성/정상: 85/15 (약간 공격적)
    """

    def __init__(
        self,
        base_conservative: float = 0.90,
        base_aggressive: float = 0.10,
        vol_lookback: int = 60,
    ):
        self.base_conservative = base_conservative
        self.base_aggressive = base_aggressive
        self.vol_lookback = vol_lookback

    def allocate(
        self, returns: pd.Series
    ) -> Dict[str, float]:
        """바벨 배분 계산

        Returns:
            dict with conservative_pct, aggressive_pct,
            vol_regime, adjustment_reason
        """
        if len(returns) < 20:
            return {
                "conservative_pct": (
                    self.base_conservative * 100
                ),
                "aggressive_pct": (
                    self.base_aggressive * 100
                ),
                "vol_regime": "unknown",
                "adjustment_reason": "insufficient data",
            }

        recent = returns.tail(self.vol_lookback)
        vol = recent.std() * np.sqrt(252)

        # 장기 변동성 대비 현재 변동성
        long_vol = returns.std() * np.sqrt(252)

        if long_vol < 1e-10:
            vol_ratio = 1.0
        else:
            vol_ratio = vol / long_vol

        # 변동성 레짐 판단
        if vol_ratio > 1.5:
            regime = "crisis"
            cons = min(
                self.base_conservative + 0.05, 0.95
            )
        elif vol_ratio > 1.2:
            regime = "high_vol"
            cons = self.base_conservative + 0.03
        elif vol_ratio < 0.7:
            regime = "low_vol"
            cons = max(
                self.base_conservative - 0.05, 0.80
            )
        else:
            regime = "normal"
            cons = self.base_conservative

        aggr = 1.0 - cons

        return {
            "conservative_pct": round(cons * 100, 1),
            "aggressive_pct": round(aggr * 100, 1),
            "vol_regime": regime,
            "vol_ratio": round(vol_ratio, 3),
            "adjustment_reason": (
                f"vol_ratio={vol_ratio:.2f}"
            ),
        }


# ─────────────────────────────────────────────
# 9. TalebRiskAnalyzer (통합)
# ─────────────────────────────────────────────

class TalebRiskAnalyzer:
    """탈레브 리스크 통합 분석기

    모든 하위 모듈을 통합하여 단일 TalebRiskSignal 생성.
    risk/manager.py 파이프라인에서 호출.
    """

    def __init__(self, lookback: int = 252):
        self.lookback = lookback
        self._ergodicity = ErgodicityChecker(
            ruin_threshold=-0.20,
            n_simulations=500,
            n_periods=lookback,
        )
        self._ruin = RuinProbEstimator(
            max_ruin_prob=0.01,
            horizon_periods=252 * 10,
        )
        self._fat_tail = FatTailDetector(tail_pct=0.05)
        self._cvar = CVaRCalculator()
        self._bob_rubin = BobRubinDetector(lookback)
        self._precaution = PrecautionaryFilter()
        self._barbell = BarbellAllocator()

    def analyze(self, df: pd.DataFrame) -> TalebRiskSignal:
        """OHLCV DataFrame으로 통합 분석

        Args:
            df: OHLCV DataFrame (close 컬럼 필수)

        Returns:
            TalebRiskSignal
        """
        signal = TalebRiskSignal()

        if df is None or len(df) < 30:
            return signal

        close = df["close"].dropna()
        if len(close) < 30:
            return signal

        returns = close.pct_change().dropna()
        if len(returns) < 20:
            return signal

        # 개별 분석 실행 (독립적)
        try:
            ergo = self._ergodicity.check(returns)
            signal.is_ergodic = ergo["is_ergodic"]
            signal.ensemble_mean = ergo["ensemble_mean"]
            signal.time_mean = ergo["time_mean"]
            signal.ergodicity_ratio = (
                ergo["ergodicity_ratio"]
            )
        except Exception as e:
            logger.debug(f"ergodicity check failed: {e}")

        try:
            ruin = self._ruin.estimate(returns)
            signal.ruin_probability = (
                ruin["cumulative_ruin_prob"]
            )
            signal.expected_ruin_time = (
                ruin["expected_ruin_time"]
            )
            signal.ruin_safe = ruin["is_safe"]
        except Exception as e:
            logger.debug(f"ruin estimation failed: {e}")

        try:
            fat = self._fat_tail.detect(returns)
            signal.is_fat_tailed = fat["is_fat_tailed"]
            signal.tail_index = fat["tail_index"]
            signal.kurtosis_excess = (
                fat["kurtosis_excess"]
            )
        except Exception as e:
            logger.debug(
                f"fat tail detection failed: {e}"
            )

        try:
            cvar95 = self._cvar.calculate(returns, 0.95)
            cvar99 = self._cvar.calculate(returns, 0.99)
            signal.var_95 = cvar95["var"]
            signal.cvar_95 = cvar95["cvar"]
            signal.var_99 = cvar99["var"]
            signal.cvar_99 = cvar99["cvar"]
        except Exception as e:
            logger.debug(
                f"CVaR calculation failed: {e}"
            )

        try:
            bob = self._bob_rubin.detect(returns)
            signal.bob_rubin_score = (
                bob["bob_rubin_score"]
            )
            signal.skewness = bob["skewness"]
        except Exception as e:
            logger.debug(
                f"Bob Rubin detection failed: {e}"
            )

        # 예방 원칙 종합 판단
        try:
            ruin_r = self._ruin.estimate(returns)
            fat_r = self._fat_tail.detect(returns)
            cvar_r = self._cvar.calculate(returns, 0.95)

            prec = self._precaution.evaluate(
                ruin_r, fat_r, cvar_r
            )
            signal.precautionary_block = (
                prec["precautionary_block"]
            )
            signal.position_scale = prec["position_scale"]
        except Exception as e:
            logger.debug(
                f"precautionary filter failed: {e}"
            )

        # 바벨 배분
        try:
            barbell = self._barbell.allocate(returns)
            signal.barbell_conservative_pct = (
                barbell["conservative_pct"]
            )
            signal.barbell_aggressive_pct = (
                barbell["aggressive_pct"]
            )
        except Exception as e:
            logger.debug(
                f"barbell allocation failed: {e}"
            )

        logger.debug(
            f"TalebRisk: ergodic={signal.is_ergodic}, "
            f"ruin={signal.ruin_probability:.4f}, "
            f"fat_tail={signal.is_fat_tailed}, "
            f"cvar95={signal.cvar_95:.4f}, "
            f"bob_rubin={signal.bob_rubin_score:.2f}, "
            f"scale={signal.position_scale:.2f}"
        )

        return signal
