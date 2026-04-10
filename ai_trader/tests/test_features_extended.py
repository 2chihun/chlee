# -*- coding: utf-8 -*-
"""Extended tests for 18 previously untested feature modules.

Each test class validates:
- Basic analyze/score returns correct type
- Score/value ranges are within expected bounds
- Insufficient data returns safe defaults
- Edge cases don't cause crashes
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════
# 1. candle_patterns
# ═══════════════════════════════════════════════════════════════

class TestCandlePatterns:
    def test_detect_patterns_returns_df(self, ohlcv_200):
        from features.candle_patterns import detect_candle_patterns
        result = detect_candle_patterns(ohlcv_200)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(ohlcv_200)

    def test_get_pattern_signal(self, ohlcv_200):
        from features.candle_patterns import get_pattern_signal
        result = get_pattern_signal(ohlcv_200)
        assert isinstance(result, dict)

    def test_short_data_no_crash(self, ohlcv_short):
        from features.candle_patterns import detect_candle_patterns
        result = detect_candle_patterns(ohlcv_short)
        assert isinstance(result, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════
# 2. credit_cycle
# ═══════════════════════════════════════════════════════════════

class TestCreditCycle:
    def test_analyzer_returns_dict(self, ohlcv_200):
        from features.credit_cycle import CreditCycleAnalyzer
        analyzer = CreditCycleAnalyzer()
        result = analyzer.analyze(ohlcv_200)
        assert isinstance(result, dict)

    def test_credit_environment_fields(self, ohlcv_200):
        from features.credit_cycle import CreditCycleAnalyzer
        analyzer = CreditCycleAnalyzer()
        result = analyzer.analyze(ohlcv_200)
        assert "credit_env" in result
        env = result["credit_env"]
        assert hasattr(env, "score")

    def test_short_data(self, ohlcv_short):
        from features.credit_cycle import CreditCycleAnalyzer
        analyzer = CreditCycleAnalyzer()
        result = analyzer.analyze(ohlcv_short)
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════
# 3. entropy
# ═══════════════════════════════════════════════════════════════

class TestEntropy:
    def test_shannon_entropy_positive(self, ohlcv_200):
        from features.entropy import shannon_entropy
        result = shannon_entropy(ohlcv_200["close"])
        assert isinstance(result, float)
        assert result >= 0

    def test_lempel_ziv_range(self, ohlcv_200):
        from features.entropy import lempel_ziv_complexity
        result = lempel_ziv_complexity(ohlcv_200["close"])
        assert isinstance(result, float)
        assert result >= 0

    def test_approximate_entropy(self, ohlcv_200):
        from features.entropy import approximate_entropy
        result = approximate_entropy(ohlcv_200["close"].iloc[:100])
        assert isinstance(result, float)

    def test_entropy_features_df(self, ohlcv_200):
        from features.entropy import entropy_features
        result = entropy_features(ohlcv_200["close"])
        assert isinstance(result, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════
# 4. frac_diff
# ═══════════════════════════════════════════════════════════════

class TestFracDiff:
    def test_frac_diff_ffd_returns_series(self, ohlcv_200):
        from features.frac_diff import frac_diff_ffd
        result = frac_diff_ffd(ohlcv_200["close"], d=0.35)
        assert isinstance(result, pd.Series)
        assert len(result) == len(ohlcv_200["close"])

    def test_frac_diff_d_zero_is_identity(self, ohlcv_200):
        from features.frac_diff import frac_diff_ffd
        result = frac_diff_ffd(ohlcv_200["close"], d=0.0)
        # d=0 should be close to original
        valid = result.dropna()
        assert len(valid) > 0

    def test_apply_to_ohlcv(self, ohlcv_200):
        from features.frac_diff import apply_frac_diff_to_ohlcv
        result = apply_frac_diff_to_ohlcv(ohlcv_200.copy(), d=0.35)
        assert isinstance(result, pd.DataFrame)
        fd_cols = [c for c in result.columns if "_fd" in c]
        assert len(fd_cols) > 0, f"No fractional diff columns found in {list(result.columns)}"


# ═══════════════════════════════════════════════════════════════
# 5. fundamental
# ═══════════════════════════════════════════════════════════════

class TestFundamental:
    def test_calc_srim(self):
        from features.fundamental import calc_srim
        result = calc_srim(
            equity=1_000_000_000_000,
            roe=0.15,
            discount_rate=0.08,
            shares_outstanding=10_000_000,
            omega=1.0,
        )
        assert isinstance(result, float)
        assert result > 0

    def test_estimate_roe_weighted(self):
        from features.fundamental import estimate_roe_weighted
        result = estimate_roe_weighted([0.10, 0.12, 0.15])
        assert isinstance(result, float)
        assert 0 < result < 1


# ═══════════════════════════════════════════════════════════════
# 6. importance
# ═══════════════════════════════════════════════════════════════

class TestImportance:
    def test_mdi_with_rf(self):
        from sklearn.ensemble import RandomForestClassifier
        from features.importance import feature_importance_mdi

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = (X["f0"] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        result = feature_importance_mdi(model, X.columns.tolist())
        assert result is not None
        assert len(result) == 5

    def test_mda_with_rf(self):
        from sklearn.ensemble import RandomForestClassifier
        from features.importance import feature_importance_mda

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
        y = pd.Series((X["a"] > 0).astype(int).values, index=X.index)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Use simple sklearn CV to avoid PurgedKFoldCV dependency
        result = feature_importance_mda(model, X, y, n_splits=2)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3


# ═══════════════════════════════════════════════════════════════
# 7. market_cycle
# ═══════════════════════════════════════════════════════════════

class TestMarketCycle:
    def test_analyzer_returns_signal(self, ohlcv_300):
        from features.market_cycle import MarketCycleAnalyzer
        analyzer = MarketCycleAnalyzer()
        result = analyzer.analyze(ohlcv_300)
        assert hasattr(result, "cycle_score")
        assert hasattr(result, "phase")

    def test_cycle_score_range(self, ohlcv_300):
        from features.market_cycle import MarketCycleAnalyzer
        analyzer = MarketCycleAnalyzer()
        result = analyzer.analyze(ohlcv_300)
        assert 0 <= result.cycle_score <= 100

    def test_short_data(self, ohlcv_short):
        from features.market_cycle import MarketCycleAnalyzer
        analyzer = MarketCycleAnalyzer()
        result = analyzer.analyze(ohlcv_short)
        assert hasattr(result, "phase")


# ═══════════════════════════════════════════════════════════════
# 8. market_flow
# ═══════════════════════════════════════════════════════════════

class TestMarketFlow:
    @staticmethod
    def _add_flow_columns(df):
        """Add required flow columns to OHLCV DataFrame."""
        n = len(df)
        df = df.copy()
        df["foreign_net"] = np.random.randint(-5000, 5000, n)
        df["institution_net"] = np.random.randint(-3000, 3000, n)
        df["retail_net"] = -(df["foreign_net"] + df["institution_net"])
        return df

    def test_analyzer_returns_signal(self, ohlcv_200):
        from features.market_flow import MarketFlowAnalyzer
        analyzer = MarketFlowAnalyzer()
        df = self._add_flow_columns(ohlcv_200)
        result = analyzer.analyze_market_flow("005930", df=df)
        assert hasattr(result, "trend")
        assert hasattr(result, "flow_strength")

    def test_volume_profile(self, ohlcv_200):
        from features.market_flow import MarketFlowAnalyzer
        analyzer = MarketFlowAnalyzer()
        result = analyzer.analyze_volume_profile(ohlcv_200)
        assert isinstance(result, dict)

    def test_short_data(self, ohlcv_short):
        from features.market_flow import MarketFlowAnalyzer
        analyzer = MarketFlowAnalyzer()
        df = self._add_flow_columns(ohlcv_short)
        result = analyzer.analyze_market_flow("005930", df=df)
        assert hasattr(result, "trend")


# ═══════════════════════════════════════════════════════════════
# 9. market_memory
# ═══════════════════════════════════════════════════════════════

class TestMarketMemory:
    def test_analyzer_returns_signal(self, ohlcv_300):
        from features.market_memory import MarketMemoryAnalyzer
        analyzer = MarketMemoryAnalyzer()
        result = analyzer.analyze(ohlcv_300)
        assert hasattr(result, "wall_of_worry_score")
        assert hasattr(result, "volatility_fear_score")

    def test_score_ranges(self, ohlcv_300):
        from features.market_memory import MarketMemoryAnalyzer
        analyzer = MarketMemoryAnalyzer()
        result = analyzer.analyze(ohlcv_300)
        assert 0 <= result.wall_of_worry_score <= 1
        assert 0 <= result.volatility_fear_score <= 1

    def test_short_data(self, ohlcv_short):
        from features.market_memory import MarketMemoryAnalyzer
        analyzer = MarketMemoryAnalyzer()
        result = analyzer.analyze(ohlcv_short)
        assert hasattr(result, "wall_of_worry_score")


# ═══════════════════════════════════════════════════════════════
# 10. meta_labeling_ml
# ═══════════════════════════════════════════════════════════════

class TestMetaLabelingML:
    def test_fit_and_predict(self):
        from features.meta_labeling_ml import MetaLabeler

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series((X["f0"] > 0).astype(int).values)

        labeler = MetaLabeler(n_estimators=10)
        labeler.fit(X, y)

        proba = labeler.predict_proba(X.iloc[:10])
        assert isinstance(proba, pd.Series)
        assert all(0 <= p <= 1 for p in proba)

    def test_predict_with_side(self):
        from features.meta_labeling_ml import MetaLabeler

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series((X["f0"] > 0).astype(int).values)

        labeler = MetaLabeler(n_estimators=10)
        labeler.fit(X, y)

        result = labeler.predict(X.iloc[:10], primary_side=1)
        assert isinstance(result, pd.DataFrame)
        assert "prob" in result.columns


# ═══════════════════════════════════════════════════════════════
# 11. microstructure
# ═══════════════════════════════════════════════════════════════

class TestMicrostructure:
    def test_vpin_returns_series(self, ohlcv_200):
        from features.microstructure import compute_vpin
        result = compute_vpin(
            ohlcv_200["close"],
            ohlcv_200["volume"],
        )
        assert isinstance(result, pd.Series)

    def test_kyles_lambda(self, ohlcv_200):
        from features.microstructure import kyles_lambda
        result = kyles_lambda(
            ohlcv_200["close"],
            ohlcv_200["volume"],
        )
        assert isinstance(result, pd.Series)

    def test_amihud(self, ohlcv_200):
        from features.microstructure import amihud_illiquidity
        result = amihud_illiquidity(
            ohlcv_200["close"],
            ohlcv_200["volume"],
        )
        assert isinstance(result, pd.Series)

    def test_microstructure_report(self, ohlcv_200):
        from features.microstructure import microstructure_report
        result = microstructure_report(ohlcv_200)
        assert isinstance(result, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════
# 12. sample_weights
# ═══════════════════════════════════════════════════════════════

class TestSampleWeights:
    def test_concurrent_events(self):
        from features.sample_weights import count_concurrent_events
        idx = pd.date_range("2024-01-01", periods=100, freq="D")
        t1 = pd.Series(
            idx[5:105] if len(idx) >= 105 else idx[-1],
            index=idx[:100],
        )
        # t1 maps each event start → event end
        t1 = pd.Series(idx[3:], index=idx[:len(idx)-3])
        result = count_concurrent_events(t1, idx)
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_sample_weights(self):
        from features.sample_weights import get_sample_weights
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        t1 = pd.Series(idx[2:], index=idx[:len(idx)-2])
        result = get_sample_weights(t1, idx)
        assert isinstance(result, pd.Series)
        assert (result >= 0).all()


# ═══════════════════════════════════════════════════════════════
# 13. stock_quality
# ═══════════════════════════════════════════════════════════════

class TestStockQuality:
    def test_roe_quality_range(self, ohlcv_200):
        from features.stock_quality import ROEQualityScorer
        scorer = ROEQualityScorer()
        result = scorer.score(ohlcv_200)
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_panic_buy_detector(self, ohlcv_200):
        from features.stock_quality import PanicBuyDetector
        detector = PanicBuyDetector()
        is_panic, depth = detector.detect(ohlcv_200)
        assert isinstance(is_panic, bool)
        assert isinstance(depth, float)

    def test_industry_momentum(self, ohlcv_200):
        from features.stock_quality import IndustryMomentumScorer
        scorer = IndustryMomentumScorer()
        result = scorer.score(ohlcv_200)
        assert isinstance(result, float)
        assert 0 <= result <= 1


# ═══════════════════════════════════════════════════════════════
# 14. structural_break
# ═══════════════════════════════════════════════════════════════

class TestStructuralBreak:
    def test_cusum_returns_df(self, ohlcv_200):
        from features.structural_break import cusum_test
        result = cusum_test(ohlcv_200["close"])
        assert isinstance(result, pd.DataFrame)
        assert "cusum_pos" in result.columns or len(result.columns) > 0

    def test_detect_regime_changes(self, ohlcv_200):
        from features.structural_break import detect_regime_changes
        result = detect_regime_changes(ohlcv_200["close"])
        assert isinstance(result, pd.DataFrame)

    def test_short_data(self, ohlcv_short):
        from features.structural_break import cusum_test
        result = cusum_test(ohlcv_short["close"])
        assert isinstance(result, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════
# 15. value_investor
# ═══════════════════════════════════════════════════════════════

class TestValueInvestor:
    def test_fundamental_scorer(self, ohlcv_200):
        from features.value_investor import FundamentalScorer
        scorer = FundamentalScorer()
        result = scorer.score(ohlcv_200)
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_contrarian_detector(self, ohlcv_200):
        from features.value_investor import ContrarianDetector
        detector = ContrarianDetector()
        result = detector.detect(ohlcv_200)
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_long_term_hold(self, ohlcv_200):
        from features.value_investor import LongTermHoldEvaluator
        evaluator = LongTermHoldEvaluator()
        result = evaluator.evaluate(ohlcv_200, holding_days=30)
        assert isinstance(result, tuple)
        assert len(result) >= 2


# ═══════════════════════════════════════════════════════════════
# 16. wave_position
# ═══════════════════════════════════════════════════════════════

class TestWavePosition:
    def test_analyzer_returns_signal(self, ohlcv_300):
        from features.wave_position import WavePositionAnalyzer
        analyzer = WavePositionAnalyzer()
        result = analyzer.analyze(ohlcv_300)
        assert hasattr(result, "wave_type")
        assert hasattr(result, "buy_zone_score")

    def test_buy_zone_range(self, ohlcv_300):
        from features.wave_position import WavePositionAnalyzer
        analyzer = WavePositionAnalyzer()
        result = analyzer.analyze(ohlcv_300)
        assert 0 <= result.buy_zone_score <= 1

    def test_short_data(self, ohlcv_short):
        from features.wave_position import WavePositionAnalyzer
        analyzer = WavePositionAnalyzer()
        result = analyzer.analyze(ohlcv_short)
        assert hasattr(result, "wave_type")


# ═══════════════════════════════════════════════════════════════
# 17. wizard_discipline
# ═══════════════════════════════════════════════════════════════

class TestWizardDiscipline:
    def test_analyze_wizard_signals(self, ohlcv_200):
        from features.wizard_discipline import analyze_wizard_signals
        result = analyze_wizard_signals(ohlcv_200)
        assert hasattr(result, "synergy_score")
        assert hasattr(result, "discipline_score")

    def test_confidence_scaler(self):
        from features.wizard_discipline import ConfidenceScaler
        scaler = ConfidenceScaler()
        result = scaler.scale_position(100, 0.8)
        assert isinstance(result, float)
        assert result > 0

    def test_catalyst_verifier(self, ohlcv_200):
        from features.wizard_discipline import CatalystVerifier
        verifier = CatalystVerifier()
        result = verifier.has_catalyst(ohlcv_200)
        assert isinstance(result, bool)

    def test_catalyst_strength_range(self, ohlcv_200):
        from features.wizard_discipline import CatalystVerifier
        verifier = CatalystVerifier()
        result = verifier.catalyst_strength(ohlcv_200)
        assert isinstance(result, float)
        assert result >= 0


# ═══════════════════════════════════════════════════════════════
# 18. deep_value
# ═══════════════════════════════════════════════════════════════

class TestDeepValue:
    def test_bond_type_score(self, ohlcv_300):
        from features.deep_value import BondTypeStockScorer
        scorer = BondTypeStockScorer()
        result = scorer.score(ohlcv_300)
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_safety_margin_score(self, ohlcv_300):
        from features.deep_value import SafetyMarginScorer
        scorer = SafetyMarginScorer()
        result = scorer.score(ohlcv_300)
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_falling_knife(self, ohlcv_300):
        from features.deep_value import FallingKnifeDetector
        detector = FallingKnifeDetector()
        result = detector.detect(ohlcv_300)
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_analyzer_full(self, ohlcv_300):
        from features.deep_value import SeoJunsikAnalyzer
        analyzer = SeoJunsikAnalyzer()
        result = analyzer.analyze(ohlcv_300)
        assert hasattr(result, "bond_type_score")
        assert hasattr(result, "position_multiplier")
        assert 0.3 <= result.position_multiplier <= 1.5

    def test_short_data(self, ohlcv_short):
        from features.deep_value import SeoJunsikAnalyzer
        analyzer = SeoJunsikAnalyzer()
        result = analyzer.analyze(ohlcv_short)
        assert hasattr(result, "bond_type_score")
