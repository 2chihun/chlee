"""재무제표 기반 펀더멘털 분석 모듈

사경인 S-RIM(잔여이익모델) 기반 적정주가 산정,
ROE 추정, PER/PBR/PEGR 계산 등 가치투자 핵심 지표
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger


# ── S-RIM 결과 ─────────────────────────────────────────────

@dataclass
class SRIMResult:
    """S-RIM 적정주가 산정 결과"""
    equity_per_share: float          # 주당 자기자본 (BPS)
    roe: float                       # 적용 ROE (%)
    discount_rate: float             # 할인율 (%)
    excess_profit: float             # 초과이익 (자기자본 × (ROE - 할인율))

    fair_value_optimistic: float     # 낙관적 적정가 (ω=1.0)
    fair_value_neutral: float        # 중립 적정가 (ω=0.9)
    fair_value_conservative: float   # 보수적 적정가 (ω=0.8) — 매수 기준

    current_price: float             # 현재가
    upside_pct: float                # 보수적 적정가 대비 상승여력 (%)

    @property
    def is_undervalued(self) -> bool:
        """보수적 적정가 대비 저평가 여부"""
        return self.current_price < self.fair_value_conservative

    @property
    def sell_target_1(self) -> float:
        """1차 매도 목표 (중립 적정가)"""
        return self.fair_value_neutral

    @property
    def sell_target_2(self) -> float:
        """2차 매도 목표 (낙관적 적정가)"""
        return self.fair_value_optimistic


# ── S-RIM 계산 ─────────────────────────────────────────────

def calc_srim(
    equity: float,
    roe: float,
    discount_rate: float,
    shares_outstanding: int,
    omega: float = 0.9,
) -> float:
    """S-RIM 모델로 주당 적정가를 계산합니다.

    Args:
        equity: 자기자본 (지배주주지분, 원)
        roe: ROE (소수, 예: 0.12 = 12%)
        discount_rate: 할인율 (소수, 예: 0.08 = 8%)
        shares_outstanding: 발행주식수
        omega: 지속계수 (0~1)

    Returns:
        주당 적정가 (원)
    """
    if shares_outstanding <= 0:
        return 0.0

    excess_profit = equity * (roe - discount_rate)

    if omega >= 1.0:
        # ω=1.0: 초과이익 영구 지속 → 기업가치 = 자기자본 + 초과이익/할인율
        if discount_rate <= 0:
            return equity / shares_outstanding
        enterprise_value = equity + excess_profit / discount_rate
    else:
        # ω<1.0: 기업가치 = 자기자본 + 초과이익 × ω / (1 + 할인율 - ω)
        denominator = 1 + discount_rate - omega
        if denominator <= 0:
            return equity / shares_outstanding
        enterprise_value = equity + excess_profit * omega / denominator

    return max(enterprise_value / shares_outstanding, 0)


def calc_srim_3scenarios(
    equity: float,
    roe: float,
    discount_rate: float,
    shares_outstanding: int,
    current_price: float,
) -> Optional[SRIMResult]:
    """S-RIM 3가지 시나리오(ω=1.0, 0.9, 0.8)로 적정주가를 산정합니다.

    Args:
        equity: 자기자본 (지배주주지분, 원)
        roe: ROE (소수, 예: 0.12)
        discount_rate: 할인율 (소수, 예: 0.08)
        shares_outstanding: 발행주식수
        current_price: 현재 주가

    Returns:
        SRIMResult 또는 None (데이터 부족 시)
    """
    if equity <= 0 or shares_outstanding <= 0 or current_price <= 0:
        logger.warning("S-RIM 계산 불가: 데이터 부족 (equity={}, shares={})",
                       equity, shares_outstanding)
        return None

    bps = equity / shares_outstanding
    excess_profit = equity * (roe - discount_rate)

    optimistic = calc_srim(equity, roe, discount_rate, shares_outstanding, omega=1.0)
    neutral = calc_srim(equity, roe, discount_rate, shares_outstanding, omega=0.9)
    conservative = calc_srim(equity, roe, discount_rate, shares_outstanding, omega=0.8)

    upside = (conservative - current_price) / current_price * 100 if current_price > 0 else 0

    return SRIMResult(
        equity_per_share=bps,
        roe=roe * 100,
        discount_rate=discount_rate * 100,
        excess_profit=excess_profit,
        fair_value_optimistic=optimistic,
        fair_value_neutral=neutral,
        fair_value_conservative=conservative,
        current_price=current_price,
        upside_pct=upside,
    )


# ── ROE 추정 ──────────────────────────────────────────────

def estimate_roe_weighted(roe_history: list[float]) -> float:
    """ROE 이력에서 가중평균 ROE를 추정합니다.

    가장 최근 값에 높은 비중을 부여합니다.
    - 1개: 그대로 반환
    - 2개: 최근 2 : 이전 1
    - 3개 이상: 최근 3 : 2 : 1 비중 (더 이전은 1)

    Args:
        roe_history: ROE 이력 (과거→최근 순서, 소수 예: [0.08, 0.10, 0.12])

    Returns:
        추정 ROE (소수)
    """
    if not roe_history:
        return 0.0

    n = len(roe_history)

    if n == 1:
        return roe_history[0]

    if n == 2:
        # 최근 2, 이전 1
        weights = [1, 2]
        total_w = sum(weights)
        return sum(r * w for r, w in zip(roe_history, weights)) / total_w

    # 3개 이상: 최근 3개에 3:2:1, 나머지 1
    weights = [1] * n
    weights[-1] = 3  # 가장 최근
    weights[-2] = 2  # 2번째 최근
    weights[-3] = 1  # 3번째 최근

    total_w = sum(weights)
    return sum(r * w for r, w in zip(roe_history, weights)) / total_w


def estimate_roe_trend(roe_history: list[float]) -> float:
    """ROE 추세를 분석하여 추정 ROE를 결정합니다.

    - 상승 추세: 가장 최근 값 사용
    - 하락 추세: 가장 최근 값 사용
    - 혼재: 가중평균 사용

    Args:
        roe_history: ROE 이력 (과거→최근 순서)

    Returns:
        추정 ROE (소수)
    """
    if len(roe_history) <= 1:
        return roe_history[0] if roe_history else 0.0

    # 추세 판단: 연속 증/감 확인
    diffs = [roe_history[i] - roe_history[i - 1] for i in range(1, len(roe_history))]

    all_up = all(d > 0 for d in diffs)
    all_down = all(d < 0 for d in diffs)

    if all_up or all_down:
        # 일관된 추세 → 최근값
        return roe_history[-1]

    # 혼재 → 가중평균
    return estimate_roe_weighted(roe_history)


# ── PER / PBR / PEGR ──────────────────────────────────────

def calc_per(price: float, eps: float) -> Optional[float]:
    """PER (주가수익비율) = 주가 / EPS"""
    if eps <= 0:
        return None  # 적자 시 의미 없음
    return price / eps


def calc_pbr(price: float, bps: float) -> Optional[float]:
    """PBR (주가순자산비율) = 주가 / BPS"""
    if bps <= 0:
        return None
    return price / bps


def calc_pegr(per: float, earnings_growth_rate: float) -> Optional[float]:
    """PEGR (PEG Ratio) = PER / 이익성장률

    PEGR < 1이면 저평가 가능성
    """
    if earnings_growth_rate <= 0 or per is None or per <= 0:
        return None
    return per / earnings_growth_rate


def calc_roe(net_income: float, equity: float) -> Optional[float]:
    """ROE (자기자본이익률) = 당기순이익 / 자기자본"""
    if equity <= 0:
        return None
    return net_income / equity


# ── 재무건전성 체크 ────────────────────────────────────────

@dataclass
class FinancialHealth:
    """재무건전성 평가 결과"""
    roe: Optional[float]              # ROE (%)
    per: Optional[float]              # PER
    pbr: Optional[float]              # PBR
    debt_ratio: Optional[float]       # 부채비율 (%)
    current_ratio: Optional[float]    # 유동비율 (%)
    op_margin: Optional[float]        # 영업이익률 (%)
    is_profitable: bool               # 영업이익 양수 여부
    roe_above_required: bool          # ROE > 요구수익률 여부

    @property
    def is_healthy(self) -> bool:
        """최소 재무건전성 기준 충족 여부"""
        return (
            self.is_profitable
            and self.roe_above_required
            and (self.debt_ratio is None or self.debt_ratio < 200)
        )


def evaluate_financial_health(
    net_income: float,
    equity: float,
    total_debt: float,
    total_assets: float,
    current_assets: float,
    current_liabilities: float,
    operating_income: float,
    revenue: float,
    price: float,
    eps: float,
    bps: float,
    required_return: float = 0.08,
) -> FinancialHealth:
    """재무건전성을 종합 평가합니다.

    Args:
        required_return: 요구수익률 (기본 8%, BBB- 5년 회사채 수익률 기준)
    """
    roe_val = calc_roe(net_income, equity)
    per_val = calc_per(price, eps)
    pbr_val = calc_pbr(price, bps)
    debt_ratio = (total_debt / equity * 100) if equity > 0 else None
    current_ratio = (current_assets / current_liabilities * 100) if current_liabilities > 0 else None
    op_margin = (operating_income / revenue * 100) if revenue > 0 else None

    return FinancialHealth(
        roe=roe_val * 100 if roe_val is not None else None,
        per=per_val,
        pbr=pbr_val,
        debt_ratio=debt_ratio,
        current_ratio=current_ratio,
        op_margin=op_margin,
        is_profitable=operating_income > 0,
        roe_above_required=roe_val is not None and roe_val > required_return,
    )


# ── BBB- 할인율 ───────────────────────────────────────────

# BBB- 5년 회사채 수익률 기본값 (한국신용평가 기준)
# 실제 운영 시에는 주기적으로 업데이트 필요
DEFAULT_BBB_MINUS_RATE = 0.08


def get_discount_rate(custom_rate: Optional[float] = None) -> float:
    """할인율(요구수익률)을 반환합니다.

    Args:
        custom_rate: 사용자 지정 할인율 (None이면 BBB- 기본값 사용)
    """
    if custom_rate is not None and custom_rate > 0:
        return custom_rate
    return DEFAULT_BBB_MINUS_RATE
