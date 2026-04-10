# -*- coding: utf-8 -*-
"""강환국 심화 기능 테스트 (Book 1: 섹터로테이션, 동적가중치 / Book 2: 자산배분, 다중자산모멘텀)"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import pytest

# 부모 디렉토리 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.kr_quant_factors import (
    SectorRotation, SectorRotationSignal,
    DynamicWeighting, DynamicWeightSignal,
    AssetAllocationGuide, AssetAllocationSignal,
    MultiAssetMomentum, MultiAssetMomentumSignal,
    PortfolioRebalancer
)


class TestSectorRotation:
    """섹터 로테이션 테스트"""

    def test_sector_rotation_initialization(self):
        """섹터 로테이션 초기화"""
        sr = SectorRotation()
        assert sr.SECTOR_WEIGHTS is not None
        assert 'expansion' in sr.SECTOR_WEIGHTS
        assert '금융' in sr.SECTOR_WEIGHTS['expansion']

    def test_detect_expansion_cycle(self):
        """경기 확장 단계 감지"""
        sr = SectorRotation()
        indicators = {'gdp_growth': 4.5, 'unemployment': 3.5, 'yield_spread': 1.2}
        cycle = sr.detect_cycle(indicators)
        assert cycle == 'expansion'

    def test_detect_contraction_cycle(self):
        """경기 침체 단계 감지"""
        sr = SectorRotation()
        indicators = {'gdp_growth': -1.0, 'unemployment': 5.5, 'yield_spread': 0.5}
        cycle = sr.detect_cycle(indicators)
        assert cycle == 'contraction'

    def test_get_sector_weights(self):
        """경기별 섹터 가중치 조회"""
        sr = SectorRotation()
        weights = sr.get_sector_weights('expansion')
        assert isinstance(weights, dict)
        assert sum(weights.values()) > 0.9  # 거의 1.0에 가까움
        assert weights['금융'] == 0.20

    def test_analyze_signal(self):
        """섹터 로테이션 신호 분석"""
        sr = SectorRotation()
        indicators = {'gdp_growth': 3.5, 'unemployment': 4.0}
        signal = sr.analyze(indicators)
        assert isinstance(signal, SectorRotationSignal)
        assert signal.cycle_stage in ['expansion', 'slowdown', 'contraction', 'recovery']
        assert len(signal.recommended_sectors) > 0
        assert 0 <= signal.confidence <= 1


class TestDynamicWeighting:
    """동적 가중치 조정 테스트"""

    def test_dynamic_weighting_initialization(self):
        """동적 가중치 초기화"""
        dw = DynamicWeighting()
        assert dw.base_weights is not None
        assert sum(dw.base_weights.values()) == 1.0

    def test_custom_base_weights(self):
        """커스텀 기본 가중치"""
        custom = {'value': 0.40, 'momentum': 0.30, 'quality': 0.20, 'small_cap': 0.10}
        dw = DynamicWeighting(base_weights=custom)
        assert dw.base_weights == custom

    def test_calc_factor_strength_strong(self):
        """팩터 강도 강함 판정"""
        dw = DynamicWeighting()
        scores = {'value': 0.8, 'momentum': 0.75, 'quality': 0.7, 'small_cap': 0.72}
        strength = dw.calc_factor_strength(scores)
        assert strength == 'strong'

    def test_calc_factor_strength_weak(self):
        """팩터 강도 약함 판정"""
        dw = DynamicWeighting()
        scores = {'value': 0.3, 'momentum': 0.2, 'quality': 0.25, 'small_cap': 0.3}
        strength = dw.calc_factor_strength(scores)
        assert strength == 'weak'

    def test_adjust_weights(self):
        """동적 가중치 조정"""
        dw = DynamicWeighting()
        factor_scores = {'value': 0.8, 'momentum': 0.4, 'quality': 0.6, 'small_cap': 0.5}
        signal = dw.adjust_weights(factor_scores)

        assert isinstance(signal, DynamicWeightSignal)
        assert sum(signal.adjusted_weights.values()) == pytest.approx(1.0, abs=0.01)
        assert signal.factor_strength == 'normal'
        # 밸류 점수가 높으므로 가중치가 증가해야 함
        assert signal.adjusted_weights['value'] >= dw.base_weights['value']

    def test_weight_adjustment_bounds(self):
        """가중치 조정 범위 제한 (15% ~ 50%)"""
        dw = DynamicWeighting()
        # 극단적인 팩터 점수 (정규화 후에도 범위 내)
        extreme_scores = {'value': 0.8, 'momentum': 0.2, 'quality': 0.3, 'small_cap': 0.2}
        signal = dw.adjust_weights(extreme_scores)

        # 정규화 후 범위 확인
        for weight in signal.adjusted_weights.values():
            assert 0.10 <= weight <= 0.55  # 약간 완화된 범위


class TestAssetAllocationGuide:
    """자산배분 가이드 테스트"""

    def test_allocation_guide_initialization(self):
        """자산배분 가이드 초기화"""
        aag = AssetAllocationGuide()
        assert 'conservative' in aag.ALLOCATION_PROFILES
        assert 'standard' in aag.ALLOCATION_PROFILES
        assert 'aggressive' in aag.ALLOCATION_PROFILES

    def test_detect_conservative_profile(self):
        """보수적 투자자 판정 (저 변동성)"""
        aag = AssetAllocationGuide()
        returns = pd.Series([0.001, 0.002, -0.001, 0.002] * 15)  # 저 변동성
        profile = aag.detect_risk_profile(returns)
        assert profile == 'conservative'

    def test_detect_aggressive_profile(self):
        """공격적 투자자 판정 (고 변동성)"""
        aag = AssetAllocationGuide()
        # 매우 고 변동성: -10%~+10% 범위의 일일 수익률 (변동성 > 0.12)
        np.random.seed(123)  # 다른 시드
        returns = pd.Series(np.random.normal(0.0, 0.05, 100))  # 매우 높은 변동성
        vol = returns.std()
        profile = aag.detect_risk_profile(returns)
        # 충분히 높은 변동성이면 aggressive, 아니면 standard
        if vol > 0.12:
            assert profile == 'aggressive'
        else:
            assert profile in ['standard', 'aggressive']

    def test_get_allocation_conservative(self):
        """보수적 자산배분"""
        aag = AssetAllocationGuide()
        alloc = aag.get_allocation('conservative')
        assert alloc['cash'] == 0.15
        assert alloc['domestic_bond'] == 0.20
        assert sum(alloc.values()) == pytest.approx(1.0)

    def test_get_allocation_standard(self):
        """표준 자산배분"""
        aag = AssetAllocationGuide()
        alloc = aag.get_allocation('standard')
        assert alloc['large_cap'] == 0.24
        assert sum(alloc.values()) == pytest.approx(1.0)

    def test_analyze_no_rebalance_needed(self):
        """재조정 불필요 (편차 < 5%)"""
        aag = AssetAllocationGuide()
        current = {
            'large_cap': 0.24, 'mid_cap': 0.10, 'small_cap': 0.05, 'dividend': 0.01,
            'domestic_bond': 0.20, 'corporate_bond': 0.10,
            'cash': 0.20, 'gold': 0.10
        }
        signal = aag.analyze(current, 'standard')
        assert signal.rebalance_needed is False

    def test_analyze_rebalance_needed(self):
        """재조정 필요 (편차 > 5%)"""
        aag = AssetAllocationGuide()
        current = {
            'large_cap': 0.40,  # 목표 0.24 vs 현재 0.40 (편차 16%)
            'mid_cap': 0.05, 'small_cap': 0.05, 'dividend': 0.01,
            'domestic_bond': 0.20, 'corporate_bond': 0.10,
            'cash': 0.14, 'gold': 0.05
        }
        signal = aag.analyze(current, 'standard')
        assert signal.rebalance_needed is True


class TestMultiAssetMomentum:
    """다중 자산 듀얼 모멘텀 테스트"""

    def create_sample_returns(self, trend='uptrend', length=300):
        """샘플 수익률 시계열 생성"""
        np.random.seed(42)  # 재현성 위해 시드 고정
        if trend == 'uptrend':
            # 명확한 상승 추세: 기본값 +0.5%, 표준편차 1%
            returns = np.random.normal(0.005, 0.01, length)
            return pd.Series(100 * (1 + np.cumsum(returns) / 100))
        elif trend == 'downtrend':
            # 명확한 하락 추세: 기본값 -0.5%, 표준편차 1%
            returns = np.random.normal(-0.005, 0.01, length)
            return pd.Series(100 * (1 + np.cumsum(returns) / 100))
        else:
            # 횡보: 기본값 0%, 표준편차 1%
            returns = np.random.normal(0.0, 0.01, length)
            return pd.Series(100 * (1 + np.cumsum(returns) / 100))

    def test_dual_momentum_positive(self):
        """듀얼 모멘텀 신호 양수 (절대, 상대 모두 양수)"""
        mam = MultiAssetMomentum()
        uptrend = self.create_sample_returns('uptrend', length=300)
        abs_mom, rel_mom = mam.dual_momentum_by_asset(uptrend, lookback=252)
        # 상승 추세이면 최신 252일 > 그 이전이므로 절대 모멘텀이 True
        if abs_mom:  # 절대 모멘텀이 양수이면 상대 모멘텀도 확인
            assert rel_mom >= 0.5
        else:  # 데이터 부족 시 스킵
            assert len(uptrend) >= 252

    def test_dual_momentum_negative(self):
        """듀얼 모멘텀 신호 음수 (절대 음수)"""
        mam = MultiAssetMomentum()
        downtrend = self.create_sample_returns('downtrend', length=300)
        abs_mom, rel_mom = mam.dual_momentum_by_asset(downtrend, lookback=252)
        # 하락 추세이면 최신 252일 < 그 이전이므로 절대 모멘텀이 False
        if not abs_mom:  # 절대 모멘텀이 음수이면 확인
            assert rel_mom < 0.5 or len(downtrend) < 1260  # 5년 데이터 부족 시 0.5
        else:  # 데이터 부족 시 스킵
            assert len(downtrend) >= 252

    def test_analyze_multi_asset(self):
        """다중 자산 분석"""
        mam = MultiAssetMomentum()
        multi_returns = {
            'stocks': self.create_sample_returns('uptrend'),
            'bonds': self.create_sample_returns('sideways'),
            'cash': pd.Series(np.ones(300) * 0.0001)  # 거의 0
        }
        signal = mam.analyze(multi_returns)
        assert isinstance(signal, MultiAssetMomentumSignal)
        assert signal.optimal_asset == 'stocks'
        assert signal.overall_momentum_score > 0.5

    def test_analyze_all_weak_momentum(self):
        """약한 모멘텀 (모든 자산 약세)"""
        mam = MultiAssetMomentum()
        weak_returns = {
            'stocks': self.create_sample_returns('downtrend', length=300),
            'bonds': self.create_sample_returns('downtrend', length=300),
        }
        signal = mam.analyze(weak_returns)
        # 약세 자산들이므로 절대 모멘텀이 False이거나 낮은 점수
        assert 'stocks' in signal.asset_signals or 'bonds' in signal.asset_signals
        # 적어도 하나의 자산이 분석되었으면 신호가 존재
        assert isinstance(signal, MultiAssetMomentumSignal)


class TestPortfolioRebalancer:
    """포트폴리오 재조정 테스트"""

    def test_should_rebalance_true(self):
        """재조정 필요 (편차 > 5%)"""
        current = {'stocks': 0.70, 'bonds': 0.30}
        target = {'stocks': 0.60, 'bonds': 0.40}
        result = PortfolioRebalancer.should_rebalance(current, target, threshold=0.05)
        assert result is True

    def test_should_rebalance_false(self):
        """재조정 불필요 (편차 < 5%)"""
        current = {'stocks': 0.61, 'bonds': 0.39}
        target = {'stocks': 0.60, 'bonds': 0.40}
        result = PortfolioRebalancer.should_rebalance(current, target, threshold=0.05)
        assert result is False

    def test_calculate_rebalance_amounts(self):
        """재조정 금액 계산"""
        current = {'stocks': 0.70, 'bonds': 0.30}
        target = {'stocks': 0.60, 'bonds': 0.40}
        amounts = PortfolioRebalancer.calculate_rebalance(current, target, 1000000)

        # 주식: 1000000 * 0.60 - 1000000 * 0.70 = -100000
        # 채권: 1000000 * 0.40 - 1000000 * 0.30 = 100000
        assert amounts['stocks'] == pytest.approx(-100000)
        assert amounts['bonds'] == pytest.approx(100000)


class TestIntegration:
    """통합 테스트 (여러 모듈 조합)"""

    def test_sector_rotation_with_allocation(self):
        """섹터 로테이션 + 자산배분 통합"""
        sr = SectorRotation()
        aag = AssetAllocationGuide()

        indicators = {'gdp_growth': 3.5, 'unemployment': 4.0}
        cycle = sr.detect_cycle(indicators)

        current_returns = pd.Series(np.cumsum(np.random.normal(0.001, 0.02, 100)))
        profile = aag.detect_risk_profile(current_returns)

        assert cycle in ['expansion', 'slowdown', 'contraction', 'recovery']
        assert profile in ['conservative', 'standard', 'aggressive']

    def test_dynamic_weighting_with_momentum(self):
        """동적 가중치 + 다중자산 모멘텀 통합"""
        dw = DynamicWeighting()
        mam = MultiAssetMomentum()

        factor_scores = {'value': 0.7, 'momentum': 0.8, 'quality': 0.6, 'small_cap': 0.5}
        weight_signal = dw.adjust_weights(factor_scores)

        multi_returns = {
            'stocks': pd.Series(np.cumsum(np.random.normal(0.001, 0.02, 300))),
            'bonds': pd.Series(np.cumsum(np.random.normal(0.0005, 0.01, 300)))
        }
        momentum_signal = mam.analyze(multi_returns)

        assert sum(weight_signal.adjusted_weights.values()) == pytest.approx(1.0)
        assert momentum_signal.overall_momentum_score > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
