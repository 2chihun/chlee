"""사경인 재무제표 분석 (DuPont, 부채 건전성, 현금흐름)

사경인 저 '재무제표 모르면 주식투자 절대로 하지 마라' 기반:
- DuPont 분석 (ROE = 순이익률 × 회전율 × 레버리지)
- 부채 건전성 (부채비율, 이자보상배수)
- 현금흐름 품질 (영업CF / 순이익)
- 5가지 필터 종합 평가
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class SaKyungInSignal:
    """사경인 분석 신호"""
    dupont_roe: float = 0.0           # 현재 ROE
    dupont_roe_trend: float = 0.0     # 3년 ROE CAGR
    net_margin: float = 0.0            # 순이익률
    asset_turnover: float = 0.0        # 자산회전율
    financial_leverage: float = 0.0    # 재무레버리지

    # 부채 건전성
    debt_to_equity: float = 1.0        # 부채비율 (D/E)
    interest_coverage: float = 2.0     # 이자보상배수
    debt_risk: str = "neutral"         # "safe" / "warn" / "danger"

    # 현금흐름
    cf_quality: float = 0.0            # 영업CF / 순이익 비율

    # 종합
    pass_debt_filter: bool = False     # D/E < 1.0
    pass_cf_filter: bool = False       # CF/NI > 0.8
    pass_roe_filter: bool = False      # ROE > 15%
    pass_growth_filter: bool = False   # 매출액 CAGR > 10%
    pass_valuation_filter: bool = False  # PER < 시장평균 × 0.8

    recommendation: str = "hold"       # "buy" / "hold" / "sell"
    score: float = 0.0                 # 종합 점수 (0~100)


class SaKyungInAnalyzer:
    """사경인 재무제표 분석기"""

    def __init__(self):
        pass

    def analyze(self, financials: pd.DataFrame) -> SaKyungInSignal:
        """재무제표 분석

        Parameters
        ----------
        financials : pd.DataFrame
            컬럼: NI (순이익), Sales, Assets, Equity, Debt,
                 Interest, Operating_CF, Current_Assets, Current_Liabilities

        Returns
        -------
        SaKyungInSignal
        """
        signal = SaKyungInSignal()

        try:
            if financials.empty or len(financials) < 1:
                return signal

            # 최근 기간 데이터
            latest = financials.iloc[-1]
            ni = float(latest.get('NI', 0))
            sales = float(latest.get('Sales', 1))
            assets = float(latest.get('Assets', 1))
            equity = float(latest.get('Equity', 1))
            debt = float(latest.get('Debt', 0))

            # 1. DuPont 분석
            if sales > 0 and assets > 0 and equity > 0:
                signal.net_margin = ni / sales
                signal.asset_turnover = sales / assets
                signal.financial_leverage = assets / equity if equity > 0 else 1.0
                signal.dupont_roe = signal.net_margin * signal.asset_turnover * signal.financial_leverage

            # 3년 ROE CAGR
            if len(financials) >= 3:
                roe_3y = []
                for i in range(min(3, len(financials))):
                    y = financials.iloc[-(i+1)]
                    ni_y = float(y.get('NI', 0))
                    eq_y = float(y.get('Equity', 1))
                    roe_y = ni_y / eq_y if eq_y > 0 else 0
                    roe_3y.append(roe_y)

                if len(roe_3y) >= 2 and roe_3y[0] > 0:
                    signal.dupont_roe_trend = (roe_3y[-1] / roe_3y[0]) ** (1 / (len(roe_3y)-1)) - 1

            # 2. 부채 건전성
            signal.debt_to_equity = debt / equity if equity > 0 else 1.0

            interest = float(latest.get('Interest', 1))
            ebit = float(latest.get('EBIT', ni))
            signal.interest_coverage = ebit / interest if interest > 0 else 2.0

            if signal.debt_to_equity < 0.5:
                signal.debt_risk = "safe"
            elif signal.debt_to_equity < 1.0:
                signal.debt_risk = "warn"
            else:
                signal.debt_risk = "danger"

            # 3. 현금흐름 품질
            operating_cf = float(latest.get('Operating_CF', ni))
            signal.cf_quality = operating_cf / abs(ni) if ni != 0 else 0.8

            # 4. 필터 적용
            signal.pass_debt_filter = signal.debt_to_equity < 1.0
            signal.pass_cf_filter = signal.cf_quality > 0.8
            signal.pass_roe_filter = signal.dupont_roe > 0.15

            # 성장성, 밸류에이션은 외부 데이터 필요
            signal.pass_growth_filter = True  # 외부에서 설정
            signal.pass_valuation_filter = True  # 외부에서 설정

            # 종합 점수
            filters_passed = sum([
                signal.pass_debt_filter,
                signal.pass_cf_filter,
                signal.pass_roe_filter,
                signal.pass_growth_filter,
                signal.pass_valuation_filter
            ])

            signal.score = (filters_passed / 5.0) * 100

            # 권고
            if filters_passed >= 4:
                signal.recommendation = "buy"
            elif filters_passed >= 3:
                signal.recommendation = "hold"
            else:
                signal.recommendation = "sell"

        except Exception as e:
            logger.debug(f"SaKyungInAnalyzer.analyze 오류: {e}")

        return signal
