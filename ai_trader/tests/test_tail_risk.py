"""risk/tail_risk.py 테스트"""
import numpy as np
import pandas as pd
import pytest


# ── Helper: generate test data ──

def make_normal_returns(n=500, mu=0.0005, sigma=0.02, seed=42):
    """Normal distribution returns (thin-tailed)"""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n))


def make_fat_tail_returns(n=500, seed=42):
    """Fat-tailed returns (t-distribution, df=3)"""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_t(df=3, size=n) * 0.02)


def make_bob_rubin_returns(n=500, seed=42):
    """Bob Rubin pattern: many small gains, rare huge losses"""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.002, 0.005, n)  # steady small gains
    # Inject rare huge losses
    crash_idx = rng.choice(n, size=3, replace=False)
    rets[crash_idx] = rng.uniform(-0.15, -0.08, 3)
    return pd.Series(rets)


def make_ohlcv_df(n=500, seed=42):
    """Generate OHLCV dataframe from returns"""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.02, n)
    close = 10000 * np.cumprod(1 + rets)
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    opn = close * (1 + rng.uniform(-0.01, 0.01, n))
    vol = rng.integers(1000, 100000, n)
    return pd.DataFrame({
        "open": opn, "high": high, "low": low,
        "close": close, "volume": vol,
    })


# ── Tests ──

class TestErgodicityChecker:
    def test_normal_returns_are_ergodic(self):
        from risk.tail_risk import ErgodicityChecker
        # ruin_threshold -50%: 정상 수익률에서는 파멸 안 됨
        checker = ErgodicityChecker(
            ruin_threshold=-0.50, n_simulations=200
        )
        result = checker.check(make_normal_returns())
        assert result["is_ergodic"] is True
        assert result["survival_rate"] > 0.5
        assert result["ergodicity_ratio"] > 0

    def test_short_data_returns_default(self):
        from risk.tail_risk import ErgodicityChecker
        checker = ErgodicityChecker()
        result = checker.check(pd.Series([0.01, -0.01]))
        assert result["is_ergodic"] is True

    def test_high_vol_may_be_non_ergodic(self):
        from risk.tail_risk import ErgodicityChecker
        # Very high volatility with tight ruin threshold
        checker = ErgodicityChecker(
            ruin_threshold=-0.10, n_simulations=300
        )
        rets = make_fat_tail_returns(n=500)
        result = checker.check(rets)
        # Should at least run without error
        assert "ergodicity_ratio" in result


class TestRuinProbEstimator:
    def test_safe_returns(self):
        from risk.tail_risk import RuinProbEstimator
        est = RuinProbEstimator()
        result = est.estimate(make_normal_returns(n=500))
        assert "cumulative_ruin_prob" in result
        assert "is_safe" in result

    def test_short_data(self):
        from risk.tail_risk import RuinProbEstimator
        est = RuinProbEstimator()
        result = est.estimate(pd.Series([0.01]))
        assert result["is_safe"] is True

    def test_dangerous_returns(self):
        from risk.tail_risk import RuinProbEstimator
        est = RuinProbEstimator(max_ruin_prob=0.001)
        # High vol returns
        rets = make_fat_tail_returns(n=1000)
        result = est.estimate(rets, ruin_level=-0.15)
        assert "expected_ruin_time" in result


class TestFatTailDetector:
    def test_normal_not_fat_tailed(self):
        from risk.tail_risk import FatTailDetector
        det = FatTailDetector()
        result = det.detect(make_normal_returns(n=1000))
        # Normal dist should have low kurtosis
        assert result["kurtosis_excess"] < 3.0

    def test_t_dist_is_fat_tailed(self):
        from risk.tail_risk import FatTailDetector
        det = FatTailDetector()
        result = det.detect(make_fat_tail_returns(n=1000))
        assert result["is_fat_tailed"] is True
        assert result["kurtosis_excess"] > 1.0

    def test_short_data_default(self):
        from risk.tail_risk import FatTailDetector
        det = FatTailDetector()
        result = det.detect(pd.Series([0.01] * 10))
        assert result["is_fat_tailed"] is False

    def test_tail_ratio_above_one_for_fat(self):
        from risk.tail_risk import FatTailDetector
        det = FatTailDetector()
        result = det.detect(make_fat_tail_returns(n=2000))
        assert result["tail_ratio"] > 1.0


class TestCVaRCalculator:
    def test_basic_calculation(self):
        from risk.tail_risk import CVaRCalculator
        result = CVaRCalculator.calculate(
            make_normal_returns(), 0.95
        )
        assert result["var"] > 0
        assert result["cvar"] >= result["var"]

    def test_99_confidence(self):
        from risk.tail_risk import CVaRCalculator
        r95 = CVaRCalculator.calculate(
            make_normal_returns(), 0.95
        )
        r99 = CVaRCalculator.calculate(
            make_normal_returns(), 0.99
        )
        assert r99["var"] >= r95["var"]

    def test_short_data(self):
        from risk.tail_risk import CVaRCalculator
        result = CVaRCalculator.calculate(pd.Series([0.01]), 0.95)
        assert result["var"] == 0.0


class TestBobRubinDetector:
    def test_normal_not_suspicious(self):
        from risk.tail_risk import BobRubinDetector
        det = BobRubinDetector()
        result = det.detect(make_normal_returns())
        assert result["suspicious"] is False

    def test_bob_rubin_pattern_detected(self):
        from risk.tail_risk import BobRubinDetector
        det = BobRubinDetector(lookback=500)
        result = det.detect(make_bob_rubin_returns())
        assert result["bob_rubin_score"] > 0.1
        assert result["skewness"] < 0  # negative skew

    def test_short_data(self):
        from risk.tail_risk import BobRubinDetector
        det = BobRubinDetector()
        result = det.detect(pd.Series([0.01] * 10))
        assert result["bob_rubin_score"] == 0.0


class TestPrecautionaryFilter:
    def test_safe_conditions(self):
        from risk.tail_risk import PrecautionaryFilter
        f = PrecautionaryFilter()
        result = f.evaluate(
            {"cumulative_ruin_prob": 0.001},
            {"is_fat_tailed": False, "kurtosis_excess": 1.0},
            {"cvar": 0.03},
        )
        assert result["precautionary_block"] is False
        assert result["position_scale"] > 0.5

    def test_dangerous_conditions_block(self):
        from risk.tail_risk import PrecautionaryFilter
        f = PrecautionaryFilter(max_ruin_prob=0.01)
        result = f.evaluate(
            {"cumulative_ruin_prob": 0.05},
            {"is_fat_tailed": True, "kurtosis_excess": 8.0},
            {"cvar": 0.15},
        )
        assert result["precautionary_block"] is True
        assert result["position_scale"] == 0.0


class TestLindyFilter:
    def test_weight_distribution(self):
        from risk.tail_risk import LindyFilter
        f = LindyFilter()
        w = f.weight(["sma", "momentum", "ml_alpha"])
        assert abs(sum(w.values()) - 1.0) < 0.01
        # Older strategies should have higher weight
        assert w["momentum"] > w["ml_alpha"]
        assert w["sma"] > w["ml_alpha"]

    def test_empty_list(self):
        from risk.tail_risk import LindyFilter
        f = LindyFilter()
        assert f.weight([]) == {}


class TestBarbellAllocator:
    def test_normal_allocation(self):
        from risk.tail_risk import BarbellAllocator
        alloc = BarbellAllocator()
        result = alloc.allocate(make_normal_returns())
        assert result["conservative_pct"] > 80
        assert result["aggressive_pct"] < 20
        total = result["conservative_pct"] + result["aggressive_pct"]
        assert abs(total - 100.0) < 0.1

    def test_high_vol_more_conservative(self):
        from risk.tail_risk import BarbellAllocator
        alloc = BarbellAllocator()
        # High vol returns
        result = alloc.allocate(make_fat_tail_returns(n=500))
        assert result["conservative_pct"] >= 90


class TestTalebRiskAnalyzer:
    def test_full_analysis(self):
        from risk.tail_risk import TalebRiskAnalyzer
        analyzer = TalebRiskAnalyzer(lookback=252)
        df = make_ohlcv_df()
        signal = analyzer.analyze(df)
        assert hasattr(signal, "is_ergodic")
        assert hasattr(signal, "ruin_probability")
        assert hasattr(signal, "is_fat_tailed")
        assert hasattr(signal, "cvar_95")
        assert hasattr(signal, "position_scale")
        assert 0 <= signal.position_scale <= 1.0

    def test_short_data(self):
        from risk.tail_risk import TalebRiskAnalyzer
        analyzer = TalebRiskAnalyzer()
        df = pd.DataFrame({"close": [100, 101, 102]})
        signal = analyzer.analyze(df)
        assert signal.position_scale == 1.0

    def test_none_data(self):
        from risk.tail_risk import TalebRiskAnalyzer
        analyzer = TalebRiskAnalyzer()
        signal = analyzer.analyze(None)
        assert signal.is_ergodic is True
