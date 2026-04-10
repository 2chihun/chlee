"""사경인 분석 테스트"""
import numpy as np
import pandas as pd
import pytest


class TestSaKyungInAnalyzer:
    """features.sa_kyung_in_analyzer.SaKyungInAnalyzer"""

    @pytest.fixture(autouse=True)
    def _import(self):
        from features.sa_kyung_in_analyzer import SaKyungInAnalyzer, SaKyungInSignal
        self.SKA = SaKyungInAnalyzer
        self.Signal = SaKyungInSignal

    def _make_financials(self, ni=100, sales=1000, assets=2000, equity=1000, debt=500,
                        interest=50, ebit=200, ocf=80, ca=600, cl=400):
        """더미 재무제표"""
        return pd.DataFrame({
            'NI': [ni * 0.7, ni * 0.85, ni],
            'Sales': [sales * 0.8, sales * 0.9, sales],
            'Assets': [assets * 0.9, assets * 0.95, assets],
            'Equity': [equity * 0.9, equity * 0.95, equity],
            'Debt': [debt * 1.1, debt * 1.05, debt],
            'Interest': [interest] * 3,
            'EBIT': [ebit * 0.85, ebit * 0.9, ebit],
            'Operating_CF': [ocf * 0.75, ocf * 0.85, ocf],
            'Current_Assets': [ca] * 3,
            'Current_Liabilities': [cl] * 3,
        })

    def test_dupont_roe_calculation(self):
        """DuPont ROE = 순이익률 × 회전율 × 레버리지"""
        df = self._make_financials(ni=100, sales=1000, assets=2000, equity=1000)
        result = self.SKA().analyze(df)
        # NI/Sales = 0.1, Sales/Assets = 0.5, Assets/Equity = 2.0
        # ROE = 0.1 * 0.5 * 2.0 = 0.1
        assert 0.08 <= result.dupont_roe <= 0.12, f"ROE={result.dupont_roe}"

    def test_debt_to_equity_ratio(self):
        """부채비율 계산"""
        df = self._make_financials(equity=1000, debt=500)
        result = self.SKA().analyze(df)
        assert abs(result.debt_to_equity - 0.5) < 0.05

    def test_debt_risk_classification(self):
        """부채 위험도 분류"""
        safe_df = self._make_financials(equity=1000, debt=400)
        warn_df = self._make_financials(equity=1000, debt=800)
        danger_df = self._make_financials(equity=1000, debt=1200)

        assert self.SKA().analyze(safe_df).debt_risk == "safe"
        assert self.SKA().analyze(warn_df).debt_risk == "warn"
        assert self.SKA().analyze(danger_df).debt_risk == "danger"

    def test_cf_quality(self):
        """현금흐름 품질 (CF/NI)"""
        good_cf = self._make_financials(ni=100, ocf=85)
        poor_cf = self._make_financials(ni=100, ocf=50)

        assert self.SKA().analyze(good_cf).cf_quality > 0.8
        assert self.SKA().analyze(poor_cf).cf_quality < 0.8

    def test_pass_filters(self):
        """5가지 필터 통과"""
        healthy_df = self._make_financials(
            ni=100, sales=1000, assets=2000, equity=1000,
            debt=500, ocf=85, ebit=200
        )
        result = self.SKA().analyze(healthy_df)
        # 부채비율 < 1.0: pass
        assert result.pass_debt_filter is True
        # CF/NI > 0.8: pass
        assert result.pass_cf_filter is True
        # ROE > 15%: fail (약 10%)
        assert result.pass_roe_filter is False

    def test_recommendation(self):
        """권고 신호"""
        result = self.SKA().analyze(self._make_financials())
        assert result.recommendation in ("buy", "hold", "sell")

    def test_empty_data(self):
        """빈 데이터 처리"""
        result = self.SKA().analyze(pd.DataFrame())
        assert result.score == 0.0
        assert result.recommendation == "hold"
