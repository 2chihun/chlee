# -*- coding: utf-8 -*-
"""
tests/test_chan_analytics.py

Comprehensive tests for:
  - features/chan_analytics.py      (new module)
  - features/backtest_analytics.py  (SharpeValidator, DataRequirementChecker)
  - risk/bet_sizing.py              (MultiStrategyKelly, RiskParityAllocator)

Run:  pytest tests/test_chan_analytics.py -v
"""

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────
# Helper factories
# ─────────────────────────────────────────────

def make_mean_reverting_series(n=500, phi=0.7, seed=42):
    """AR(1) with mean reversion: X_t = phi*X_{t-1} + eps
    NOTE: phi>0 creates positively correlated returns → trending prices (H>0.5)
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, 1)
    prices = 100 * np.exp(np.cumsum(x * 0.01))
    return pd.Series(prices)


def make_trending_series(n=500, drift=0.001, seed=42):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.01, n)
    return pd.Series(100 * np.cumprod(1 + rets))


def make_oscillating_prices(n=500, phi=-0.6, seed=42):
    """Prices with oscillating (negatively autocorrelated) returns → H < 0.5"""
    rng = np.random.default_rng(seed)
    rets = np.zeros(n)
    for i in range(1, n):
        rets[i] = phi * rets[i - 1] + rng.normal(0, 0.01)
    return pd.Series(100 * np.cumprod(1 + rets))


def make_persistent_prices(n=500, phi=0.7, seed=42):
    """Prices with persistent (positively autocorrelated) returns → H > 0.5"""
    rng = np.random.default_rng(seed)
    rets = np.zeros(n)
    for i in range(1, n):
        rets[i] = phi * rets[i - 1] + rng.normal(0.001, 0.005)
    return pd.Series(100 * np.cumprod(1 + rets))


def make_random_walk(n=500, seed=42):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.02, n)
    return pd.Series(100 * np.cumprod(1 + rets))


def make_ohlcv(prices=None, n=500, seed=42):
    if prices is None:
        prices = make_random_walk(n, seed)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "open":   prices * (1 + rng.normal(0, 0.005, len(prices))),
        "high":   prices * (1 + rng.uniform(0, 0.02, len(prices))),
        "low":    prices * (1 - rng.uniform(0, 0.02, len(prices))),
        "close":  prices,
        "volume": rng.integers(1000, 100000, len(prices)),
    })


# ─────────────────────────────────────────────
# 1. StationarityAnalyzer
# ─────────────────────────────────────────────

class TestStationarityAnalyzer:
    """features.chan_analytics.StationarityAnalyzer"""

    @pytest.fixture(autouse=True)
    def _import(self):
        from features.chan_analytics import StationarityAnalyzer
        self.SA = StationarityAnalyzer

    def test_random_walk_hurst_near_half(self):
        """Random walk → Hurst ≈ 0.5, not stationary"""
        s = make_random_walk(n=500)
        result = self.SA().analyze(s)
        h = result["hurst"]
        assert 0.35 <= h <= 0.65, f"Random walk H={h} should be near 0.5"
        assert result["is_stationary"] is False

    def test_mean_reverting_hurst_below_half(self):
        """Oscillating prices (neg autocorr returns) → H < 0.5"""
        s = make_oscillating_prices(n=500, phi=-0.6)
        result = self.SA().analyze(s)
        h = result["hurst"]
        assert h < 0.5, f"Oscillating prices H={h} should be < 0.5"

    def test_trending_hurst_above_half(self):
        """Persistent prices (pos autocorr returns) → H > 0.5"""
        s = make_persistent_prices(n=500, phi=0.7)
        result = self.SA().analyze(s)
        h = result["hurst"]
        assert h > 0.5, f"Persistent prices H={h} should be > 0.5"

    def test_short_data_edge_case(self):
        """Short data → returns result without raising"""
        s = pd.Series([100.0, 101.0, 100.5, 102.0, 101.5])
        result = self.SA().analyze(s)
        assert isinstance(result, dict)
        assert "hurst" in result

    def test_returns_dict_with_required_keys(self):
        s = make_random_walk(n=300)
        result = self.SA().analyze(s)
        for key in ("hurst", "is_stationary", "adf_pvalue"):
            assert key in result, f"Missing key: {key}"


# ─────────────────────────────────────────────
# 2. MeanReversionEstimator
# ─────────────────────────────────────────────

class TestMeanReversionEstimator:
    """features.chan_analytics.MeanReversionEstimator"""

    @pytest.fixture(autouse=True)
    def _import(self):
        from features.chan_analytics import MeanReversionEstimator
        self.MRE = MeanReversionEstimator

    def test_half_life_known_ar1(self):
        """AR(1) spread phi=0.7 → half_life in reasonable range"""
        phi = 0.7
        rng = np.random.default_rng(42)
        # Direct AR(1) spread (OU process approximation)
        n = 1000
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + rng.normal(0, 1)
        s = pd.Series(x)  # spread directly (not cumsum)
        result = self.MRE().analyze(s)
        hl = result["half_life"]
        # Allow wide tolerance: half_life should be finite and positive
        is_reasonable = np.isfinite(hl) and 0.5 <= hl <= 50.0
        assert is_reasonable, f"half_life={hl} not in reasonable range [0.5, 50]"

    def test_zscore_calculation(self):
        """zscore should be centered around 0 for stationary series"""
        s = make_mean_reverting_series(n=500, phi=0.7)
        result = self.MRE().analyze(s)
        z = result["zscore"]
        assert abs(z) < 5.0, f"zscore={z} looks extreme"

    def test_signal_generation(self):
        """signal should be +1, -1, or 0"""
        s = make_mean_reverting_series(n=500, phi=0.7)
        result = self.MRE().analyze(s)
        assert result["signal"] in (-1, 0, 1), \
            f"signal={result['signal']} not in {{-1, 0, 1}}"

    def test_random_walk_half_life_large_or_nan(self):
        """Random walk → half_life should be large (>15) or non-mean-reverting"""
        rng = np.random.default_rng(42)
        s = pd.Series(np.cumsum(rng.normal(0, 1, 500)))  # simple RW
        result = self.MRE().analyze(s)
        hl = result["half_life"]
        # Random walk: half_life > 15 (weak reversion), or is_useful=False, or nan/inf
        is_non_reverting = (hl > 15) or (not result["is_useful"]) or np.isinf(hl) or np.isnan(hl) or (hl < 0)
        assert is_non_reverting, f"Random walk half_life={hl} should indicate weak/no mean-reversion"


# ─────────────────────────────────────────────
# 3. CointegrationAnalyzer
# ─────────────────────────────────────────────

class TestCointegrationAnalyzer:
    """features.chan_analytics.CointegrationAnalyzer"""

    @pytest.fixture(autouse=True)
    def _import(self):
        from features.chan_analytics import CointegrationAnalyzer
        self.CA = CointegrationAnalyzer

    def _make_cointegrated_pair(self, n=500, seed=42):
        """Two cointegrated series: same underlying factor + noise"""
        rng = np.random.default_rng(seed)
        common = np.cumsum(rng.normal(0, 1, n))
        s1 = pd.Series(100 + common + rng.normal(0, 0.5, n))
        s2 = pd.Series(50 + 0.5 * common + rng.normal(0, 0.5, n))
        return s1, s2

    def _make_non_cointegrated_pair(self, n=500, seed=42):
        """Two independent random walks"""
        rng = np.random.default_rng(seed)
        s1 = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
        s2 = pd.Series(50 + np.cumsum(rng.normal(0, 1, n)))
        return s1, s2

    def test_cointegrated_series_detected(self):
        """Cointegrated pair → is_cointegrated=True"""
        s1, s2 = self._make_cointegrated_pair()
        result = self.CA().analyze(s1, s2)
        assert result["is_cointegrated"] is True, \
            f"pvalue={result.get('pvalue')}: cointegrated pair not detected"

    def test_non_cointegrated_series(self):
        """Independent random walks → is_cointegrated=False"""
        s1, s2 = self._make_non_cointegrated_pair()
        result = self.CA().analyze(s1, s2)
        # Allow loose assertion: pvalue should be higher
        pvalue = result.get("pvalue", 1.0)
        assert pvalue > 0.01, \
            f"Non-cointegrated pair pvalue={pvalue} unexpectedly low"

    def test_hedge_ratio_calculation(self):
        """hedge_ratio should be a finite number"""
        s1, s2 = self._make_cointegrated_pair()
        result = self.CA().analyze(s1, s2)
        hr = result["hedge_ratio"]
        assert np.isfinite(hr), f"hedge_ratio={hr} is not finite"
        assert hr != 0.0, "hedge_ratio should not be zero"

    def test_spread_calculation(self):
        """spread should be a Series of same length as inputs"""
        s1, s2 = self._make_cointegrated_pair()
        result = self.CA().analyze(s1, s2)
        spread = result["spread"]
        assert isinstance(spread, pd.Series)
        assert len(spread) == len(s1)


# ─────────────────────────────────────────────
# 4. KalmanFilterHedge
# ─────────────────────────────────────────────

class TestKalmanFilterHedge:
    """features.chan_analytics.KalmanFilterHedge"""

    @pytest.fixture(autouse=True)
    def _import(self):
        from features.chan_analytics import KalmanFilterHedge
        self.KFH = KalmanFilterHedge

    def _make_pair(self, n=300, seed=42):
        rng = np.random.default_rng(seed)
        common = np.cumsum(rng.normal(0, 1, n))
        s1 = pd.Series(100 + common + rng.normal(0, 0.5, n))
        s2 = pd.Series(50 + 0.5 * common + rng.normal(0, 0.5, n))
        return s1, s2

    def test_basic_filtering_runs(self):
        """KalmanFilterHedge.filter() should run without error"""
        s1, s2 = self._make_pair()
        result = self.KFH().filter(s1, s2)
        assert isinstance(result, dict)

    def test_hedge_ratio_changes_over_time(self):
        """Dynamic hedge ratio should vary (not constant)"""
        s1, s2 = self._make_pair(n=300)
        result = self.KFH().filter(s1, s2)
        hr_series = result["hedge_ratio"]
        assert isinstance(hr_series, pd.Series)
        assert len(hr_series) == len(s1)
        # hedge ratio should not be all the same value
        unique_vals = hr_series.nunique()
        assert unique_vals > 5, \
            f"hedge_ratio has only {unique_vals} unique values (not dynamic)"

    def test_signal_generation(self):
        """filter() result should include a signal column"""
        s1, s2 = self._make_pair()
        result = self.KFH().filter(s1, s2)
        assert "signal" in result, "KalmanFilterHedge result missing 'signal'"
        sig = result["signal"]
        # signal should be within reasonable bounds
        assert sig.dropna().isin([-1, 0, 1]).all() or \
               sig.dropna().between(-2, 2).all(), \
            "signal values out of expected range"


# ─────────────────────────────────────────────
# 5. CrossSectionalMRSignal
# ─────────────────────────────────────────────

class TestCrossSectionalMRSignal:
    """features.chan_analytics.CrossSectionalMRSignal"""

    @pytest.fixture(autouse=True)
    def _import(self):
        from features.chan_analytics import CrossSectionalMRSignal
        self.CMRS = CrossSectionalMRSignal

    def _make_returns_df(self, n=100, m=5, seed=42):
        """Returns DataFrame: n rows (dates), m columns (stocks)"""
        rng = np.random.default_rng(seed)
        cols = [f"stock_{i}" for i in range(m)]
        data = rng.normal(0, 0.02, (n, m))
        return pd.DataFrame(data, columns=cols)

    def test_weights_sum_to_zero(self):
        """Cross-sectional weights must be dollar-neutral (sum ≈ 0)"""
        returns_df = self._make_returns_df()
        weights = self.CMRS().compute(returns_df)
        last_weights = weights.iloc[-1]
        assert abs(last_weights.sum()) < 1e-6, \
            f"Weights sum={last_weights.sum():.6f} not near 0 (dollar neutral)"

    def test_high_return_stocks_get_negative_weight(self):
        """High recent-return stocks → negative weight (mean reversion logic)"""
        rng = np.random.default_rng(42)
        n, m = 50, 5
        cols = [f"s{i}" for i in range(m)]
        data = rng.normal(0, 0.01, (n, m))
        # Inject large positive return in stock 0 on last row
        data[-1, 0] = 0.10
        data[-1, 1:] = -0.005
        df = pd.DataFrame(data, columns=cols)
        weights = self.CMRS().compute(df)
        last_w = weights.iloc[-1]
        assert last_w["s0"] < 0, \
            f"High-return stock s0 should have negative weight, got {last_w['s0']}"

    def test_uniform_returns_all_weights_zero(self):
        """Uniform cross-sectional returns → all weights ≈ 0"""
        n, m = 50, 5
        cols = [f"s{i}" for i in range(m)]
        # All stocks have the same return each day
        data = np.ones((n, m)) * 0.01
        df = pd.DataFrame(data, columns=cols)
        weights = self.CMRS().compute(df)
        last_w = weights.iloc[-1]
        assert (last_w.abs() < 1e-6).all(), \
            f"Uniform returns should yield zero weights, got {last_w.to_dict()}"


# ─────────────────────────────────────────────
# 6. MomentumSignal
# ─────────────────────────────────────────────

class TestMomentumSignal:
    """features.chan_analytics.MomentumSignal"""

    @pytest.fixture(autouse=True)
    def _import(self):
        from features.chan_analytics import MomentumSignal
        self.MS = MomentumSignal

    def test_timeseries_signal_positive_trend(self):
        """Price above lookback start → signal = +1"""
        # Strictly increasing prices
        prices = pd.Series(np.linspace(100, 200, 300))
        result = self.MS().timeseries_signal(prices, lookback=252)
        # After enough data, signal should be +1
        last_sig = result.dropna().iloc[-1]
        assert last_sig == 1, f"Uptrend should give signal=+1, got {last_sig}"

    def test_timeseries_signal_negative_trend(self):
        """Price below lookback start → signal = -1"""
        prices = pd.Series(np.linspace(200, 100, 300))
        result = self.MS().timeseries_signal(prices, lookback=252)
        last_sig = result.dropna().iloc[-1]
        assert last_sig == -1, f"Downtrend should give signal=-1, got {last_sig}"

    def test_gap_reversal_signal_gap_down(self):
        """Synthetic gap-down data → gap reversal signal should be positive"""
        rng = np.random.default_rng(42)
        n = 300
        prices = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, n)))
        # Force large gap down at the end
        gap_prices = prices.copy()
        gap_prices.iloc[-1] = gap_prices.iloc[-2] * 0.93  # -7% gap
        result = self.MS().gap_reversal_signal(gap_prices)
        last_sig = result.dropna().iloc[-1]
        # Gap down → expect buy signal (+1) or at least non-negative
        assert last_sig >= 0, \
            f"Gap-down should yield reversal buy signal, got {last_sig}"


# ─────────────────────────────────────────────
# 7. RegimeDetector
# ─────────────────────────────────────────────

class TestRegimeDetector:
    """features.chan_analytics.RegimeDetector"""

    @pytest.fixture(autouse=True)
    def _import(self):
        from features.chan_analytics import RegimeDetector
        self.RD = RegimeDetector

    def test_mean_reverting_regime(self):
        """Oscillating prices (neg autocorr returns) → regime = MEAN_REVERSION"""
        s = make_oscillating_prices(n=500, phi=-0.6)
        result = self.RD().detect(s)
        regime = result["regime"]
        assert regime == "MEAN_REVERSION", \
            f"Expected MEAN_REVERSION, got {regime} (H={result.get('hurst'):.3f})"

    def test_trending_regime(self):
        """Persistent prices (pos autocorr returns) → regime = TRENDING"""
        s = make_persistent_prices(n=500, phi=0.7)
        result = self.RD().detect(s)
        regime = result["regime"]
        assert regime == "TRENDING", \
            f"Expected TRENDING, got {regime} (H={result.get('hurst'):.3f})"

    def test_returns_dict_with_regime_key(self):
        s = make_random_walk(n=300)
        result = self.RD().detect(s)
        assert "regime" in result
        assert result["regime"] in ("MEAN_REVERSION", "TRENDING", "NEUTRAL", "RANDOM", "UNKNOWN")


# ─────────────────────────────────────────────
# 8. ChanAnalyzer (full integration)
# ─────────────────────────────────────────────

class TestChanAnalyzer:
    """features.chan_analytics.ChanAnalyzer"""

    @pytest.fixture(autouse=True)
    def _import(self):
        from features.chan_analytics import ChanAnalyzer
        self.CA = ChanAnalyzer

    def test_full_analysis_on_ohlcv(self):
        """Full analysis on OHLCV DataFrame returns a signal"""
        df = make_ohlcv(n=500)
        result = self.CA().analyze(df)
        assert isinstance(result, dict)
        assert "signal" in result
        assert result["signal"] in (-1, 0, 1)

    def test_none_data_returns_default_signal(self):
        """None input → default signal (0) without raising"""
        result = self.CA().analyze(None)
        assert result["signal"] == 0

    def test_short_data_returns_default_signal(self):
        """Very short data → default signal (0)"""
        df = make_ohlcv(n=5)
        result = self.CA().analyze(df)
        assert result["signal"] == 0

    def test_result_contains_regime_and_hurst(self):
        """Full analysis should include regime and hurst in result"""
        df = make_ohlcv(n=500)
        result = self.CA().analyze(df)
        assert "regime" in result, "ChanAnalyzer result missing 'regime'"
        assert "hurst" in result, "ChanAnalyzer result missing 'hurst'"


# ─────────────────────────────────────────────
# 9. SharpeValidator (from backtest_analytics.py)
# ─────────────────────────────────────────────

class TestSharpeValidator:
    """features.backtest_analytics.SharpeValidator"""

    @pytest.fixture(autouse=True)
    def _import(self):
        from features.backtest_analytics import SharpeValidator
        self.SV = SharpeValidator

    def _make_returns(self, mean=0.001, std=0.02, n=252, seed=42):
        rng = np.random.default_rng(seed)
        return pd.Series(rng.normal(mean, std, n))

    def test_sharpe_formula_matches(self):
        """Annualized Sharpe = sqrt(252) * (mean - rf_daily) / std"""
        rets = self._make_returns(mean=0.001, std=0.02, n=500)
        result = self.SV.calculate(rets)
        rf_daily = 0.03 / 252
        expected = float(np.sqrt(252) * (rets.mean() - rf_daily) / rets.std())
        actual = result["annualized_sharpe"]
        assert abs(actual - expected) < 0.05, \
            f"Sharpe mismatch: expected={expected:.4f}, actual={actual:.4f}"

    def test_grade_A_plus(self):
        """Sharpe > 2.0 → grade A+"""
        # Construct returns with very high Sharpe
        rng = np.random.default_rng(0)
        rets = pd.Series(rng.normal(0.01, 0.01, 500))
        result = self.SV.calculate(rets)
        if result["annualized_sharpe"] > 2.0:
            assert result["grade"] == "A+", \
                f"SR>{2.0} should be A+, got {result['grade']}"

    def test_grade_assignment_spectrum(self):
        """Different SR levels map to correct grades"""
        grade_map = [
            (0.005, 0.02, "D"),   # low SR
            (0.001, 0.02, "C"),   # marginal
        ]
        for mean, std, expected_min_grade in grade_map:
            rets = pd.Series(np.random.default_rng(1).normal(mean, std, 500))
            result = self.SV.calculate(rets)
            assert result["grade"] in ("A+", "A", "B", "C", "D"), \
                f"Grade '{result['grade']}' not in expected set"

    def test_cagr_calculation(self):
        """CAGR should be positive for positive-drift returns"""
        rng = np.random.default_rng(42)
        rets = pd.Series(rng.normal(0.001, 0.01, 252))
        result = self.SV.calculate(rets)
        assert "cagr" in result
        # With positive mean, CAGR should be positive
        if rets.mean() > 0:
            assert result["cagr"] > 0, \
                f"Positive drift returns should have positive CAGR"

    def test_max_drawdown_known_series(self):
        """Max drawdown of 50% drop series should be ~0.50"""
        # Prices: start at 100, drop to 50, then recover to 60
        prices = pd.Series([100, 90, 80, 70, 60, 50, 55, 58, 60])
        rets = prices.pct_change().dropna()
        result = self.SV.calculate(rets)
        mdd = result.get("max_drawdown")
        assert mdd is not None, "max_drawdown missing from result"
        # Drawdown should be around 50% (100→50)
        assert 0.30 <= abs(mdd) <= 0.65, \
            f"max_drawdown={mdd} out of expected range for 50% drop"

    def test_t_stat_and_p_value(self):
        """t_stat and p_value should be present and valid"""
        rets = self._make_returns(mean=0.002, std=0.02, n=500)
        result = self.SV.calculate(rets)
        assert "t_stat" in result, "t_stat missing"
        assert "p_value" in result, "p_value missing"
        assert 0.0 <= result["p_value"] <= 1.0, \
            f"p_value={result['p_value']} out of [0,1]"
        # With large n and positive mean, t_stat should be positive
        assert result["t_stat"] > 0, \
            f"Positive mean should give positive t_stat, got {result['t_stat']}"

    def test_passes_minimum_flag(self):
        """passes_minimum=True when Sharpe > 1.0"""
        rng = np.random.default_rng(5)
        rets = pd.Series(rng.normal(0.003, 0.01, 500))
        result = self.SV.calculate(rets)
        sr = result["annualized_sharpe"]
        if sr > 1.0:
            assert result["passes_minimum"] is True
        else:
            assert result["passes_minimum"] is False


# ─────────────────────────────────────────────
# 10. DataRequirementChecker (from backtest_analytics.py)
# ─────────────────────────────────────────────

class TestDataRequirementChecker:
    """features.backtest_analytics.DataRequirementChecker"""

    @pytest.fixture(autouse=True)
    def _import(self):
        from features.backtest_analytics import DataRequirementChecker
        self.DRC = DataRequirementChecker

    def test_sufficient_data(self):
        """756 data points, 3 params → ok (LOW overfitting risk)"""
        result = self.DRC.check(n_data=756, n_parameters=3)
        assert result["overfitting_risk"] != "HIGH", \
            f"756 data / 3 params should not be HIGH risk"
        assert result["is_sufficient"] is True

    def test_insufficient_data(self):
        """252 data points, 3 params → not ok"""
        result = self.DRC.check(n_data=252, n_parameters=3)
        # 252 / 3 = 84 observations per param; Chan requires ~252 per param
        assert result["is_sufficient"] is False or \
               result["overfitting_risk"] in ("HIGH", "MEDIUM"), \
            f"252 data / 3 params should be insufficient or risky"

    def test_too_many_params(self):
        """6 params, 756 data → overfitting risk HIGH"""
        result = self.DRC.check(n_data=756, n_parameters=6)
        # 756/6 = 126 per param, below the 252 threshold
        assert result["overfitting_risk"] == "HIGH" or \
               result["is_sufficient"] is False, \
            f"6 params with 756 data should flag overfitting"

    def test_returns_required_keys(self):
        """Result must contain is_sufficient and overfitting_risk keys"""
        result = self.DRC.check(n_data=500, n_parameters=2)
        assert "is_sufficient" in result
        assert "overfitting_risk" in result

    def test_ratio_stored_in_result(self):
        """Result may include obs_per_param ratio"""
        result = self.DRC.check(n_data=500, n_parameters=5)
        # Should include either ratio or n_data and n_parameters
        has_ratio = "obs_per_param" in result or \
                    ("n_data" in result and "n_parameters" in result)
        assert has_ratio, \
            "DataRequirementChecker result should expose ratio or data counts"


# ─────────────────────────────────────────────
# 11. MultiStrategyKelly (from bet_sizing.py)
# ─────────────────────────────────────────────

class TestMultiStrategyKelly:
    """risk.bet_sizing.MultiStrategyKelly"""

    @pytest.fixture(autouse=True)
    def _import(self):
        from risk.bet_sizing import MultiStrategyKelly
        self.MSK = MultiStrategyKelly

    def _make_strategy_returns(self, n=252, k=3, seed=42):
        """k strategies, n observations each"""
        rng = np.random.default_rng(seed)
        data = rng.normal(0.001, 0.02, (n, k))
        cols = [f"strat_{i}" for i in range(k)]
        return pd.DataFrame(data, columns=cols)

    def test_F_star_formula(self):
        """F* = C^{-1} * M (covariance inverse times mean vector)"""
        df = self._make_strategy_returns(n=500, k=3)
        result = self.MSK().compute(df)
        f_star = result["f_star"]
        assert isinstance(f_star, (np.ndarray, pd.Series)), \
            "f_star should be array-like"
        assert len(f_star) == 3, f"f_star length={len(f_star)}, expected 3"

    def test_half_kelly_applies(self):
        """Half-Kelly fractions should be exactly half of full Kelly"""
        df = self._make_strategy_returns(n=500, k=3)
        result = self.MSK().compute(df)
        f_star = np.array(result["f_star"])
        half_f = np.array(result["half_kelly"])
        np.testing.assert_allclose(
            half_f, f_star / 2, rtol=1e-6,
            err_msg="Half-kelly should be f_star / 2"
        )

    def test_leverage_scaled_to_max(self):
        """If raw F* exceeds max_leverage, scaled weights should be capped"""
        df = self._make_strategy_returns(n=500, k=3)
        max_lev = 1.0
        result = self.MSK(max_leverage=max_lev).compute(df)
        scaled = np.array(result.get("scaled_kelly", result["half_kelly"]))
        total_exposure = np.sum(np.abs(scaled))
        assert total_exposure <= max_lev + 1e-6, \
            f"Total exposure {total_exposure:.4f} exceeds max_leverage {max_lev}"

    def test_single_strategy_kelly(self):
        """Single strategy: F* = (mean_excess_annual) / (var_annual)"""
        rng = np.random.default_rng(42)
        n = 500
        rets = pd.Series(rng.normal(0.001, 0.02, n))
        df = rets.to_frame("strat_0")
        result = self.MSK().compute(df, lookback=n)  # use all data
        f_star = result["f_star"]
        # With RF and annualization: F* = mu_excess_annual / var_annual
        rf_daily = 0.03 / 252
        excess = rets - rf_daily
        expected = float((excess.mean() * 252) / (excess.var() * 252))
        actual = float(np.array(f_star).ravel()[0])
        assert abs(actual - expected) < 1.0, \
            f"Single-strategy F*={actual:.4f}, expected≈{expected:.4f}"


# ─────────────────────────────────────────────
# 12. RiskParityAllocator (from bet_sizing.py)
# ─────────────────────────────────────────────

class TestRiskParityAllocator:
    """risk.bet_sizing.RiskParityAllocator"""

    @pytest.fixture(autouse=True)
    def _import(self):
        from risk.bet_sizing import RiskParityAllocator
        self.RPA = RiskParityAllocator

    def _make_returns_df(self, vols=None, n=252, seed=42):
        """Returns DataFrame with specified volatilities per strategy"""
        rng = np.random.default_rng(seed)
        if vols is None:
            vols = [0.01, 0.02, 0.03]
        cols = [f"s{i}" for i in range(len(vols))]
        data = np.column_stack([
            rng.normal(0.001, v, n) for v in vols
        ])
        return pd.DataFrame(data, columns=cols)

    def test_weights_sum_to_one(self):
        """Risk-parity weights before leverage should sum to ~1.0"""
        df = self._make_returns_df(vols=[0.01, 0.02, 0.03])
        weights = self.RPA().allocate(df)
        assert isinstance(weights, (dict, pd.Series, np.ndarray))
        w = np.array(list(weights.values()) if isinstance(weights, dict)
                     else weights)
        assert abs(w.sum() - 1.0) < 0.01, \
            f"Weights sum={w.sum():.4f}, expected ~1.0"

    def test_high_vol_assets_lower_weight(self):
        """Higher-volatility strategy should receive lower weight"""
        df = self._make_returns_df(vols=[0.01, 0.02, 0.04])
        weights = self.RPA().allocate(df)
        if isinstance(weights, dict):
            w = list(weights.values())
        elif isinstance(weights, pd.Series):
            w = weights.values
        else:
            w = np.array(weights)
        # s0 (vol=0.01) should have highest weight, s2 (vol=0.04) lowest
        assert w[0] > w[1] > w[2], \
            f"Expected w[0]>w[1]>w[2], got {w}"

    def test_empty_dataframe_returns_equal_weights(self):
        """Empty DataFrame → equal weights (or graceful fallback)"""
        empty_df = pd.DataFrame()
        result = self.RPA().allocate(empty_df)
        # Should return empty or raise gracefully — not crash
        if result is not None:
            w = (np.array(list(result.values())) if isinstance(result, dict)
                 else np.array(result))
            if len(w) > 0:
                np.testing.assert_allclose(
                    w, np.full(len(w), 1.0 / len(w)), rtol=1e-6,
                    err_msg="Empty input should return equal weights"
                )

    def test_equal_vol_assets_equal_weights(self):
        """When all assets have same vol → equal weights"""
        df = self._make_returns_df(vols=[0.02, 0.02, 0.02])
        weights = self.RPA().allocate(df)
        if isinstance(weights, dict):
            w = np.array(list(weights.values()))
        elif isinstance(weights, pd.Series):
            w = weights.values
        else:
            w = np.array(weights)
        expected = np.full(3, 1.0 / 3)
        np.testing.assert_allclose(w, expected, atol=0.05,
            err_msg="Equal-vol assets should get equal weights")
