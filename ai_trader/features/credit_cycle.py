"""신용/자본시장 사이클 분석 모듈

하워드 막스 "투자와 마켓사이클의 법칙" 제9장 핵심 개념 구현:

신용사이클(Credit Cycle) 핵심 원리:
- 신용창구(Credit Window)는 열리고 닫힌다: 완화(EASY) → 긴축(TIGHT)
- 경기 번영 → 위험회피 감소 → 대출기준 완화 → 과잉대출 → 손실 → 긴축
- 신용사이클은 가장 변동이 심하고 경제/시장에 가장 큰 영향을 미친다
- 신용긴축 국면 = 저가매수 기회가 가장 잘 조성되는 구간

유동성 리스크 원리:
- "장기투자를 위해 단기로 빌리는 방식"은 신용시장이 닫히면 위기가 된다
- 자본시장이 경색되면 아무리 건실한 기업도 채무불이행 위험에 처한다
- 극단적 신용긴축 국면이 역투자자에게는 최고의 기회다

확률분포 이동 원리:
- 신용이 완화되면 수익 확률분포가 우측으로 이동 (유리한 방향)
- 신용이 긴축되면 수익 확률분포가 좌측으로 이동 (불리한 방향)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from features.indicators import atr, rsi, bollinger_bands


# ---------------------------------------------------------------------------
# 상수 정의
# ---------------------------------------------------------------------------

# 신용환경 레이블
CREDIT_EASY = "EASY"      # 신용 완화: 자금 풍부, 대출기준 완화
CREDIT_NORMAL = "NORMAL"  # 신용 정상: 균형 상태
CREDIT_TIGHT = "TIGHT"    # 신용 긴축: 자금 부족, 대출기준 강화

# 유동성 리스크 레이블
LIQUIDITY_LOW = "LOW"       # 유동성 충분
LIQUIDITY_MEDIUM = "MEDIUM" # 유동성 주의
LIQUIDITY_HIGH = "HIGH"     # 유동성 위기 신호

# 신용환경 점수 임계값 (0~100점, 높을수록 긴축)
CREDIT_TIGHT_THRESHOLD = 65   # 65점 이상: 긴축
CREDIT_EASY_THRESHOLD = 35    # 35점 이하: 완화

# 유동성 리스크 임계값
LIQUIDITY_HIGH_THRESHOLD = 2.5   # 평균 대비 2.5배 이상 거래량 급감
LIQUIDITY_MEDIUM_THRESHOLD = 1.5 # 평균 대비 1.5배 이상 거래량 감소


# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class CreditEnvironment:
    """신용환경 분석 결과

    하워드 막스: 신용사이클은 창문에 비유할 수 있다.
    창구가 열려 있을 때는 자금이 풍부하고 쉽게 구할 수 있다.
    창구가 닫혀 있을 때는 자금이 부족하고 구하기 어렵다.
    """
    status: str          # EASY / NORMAL / TIGHT
    score: float         # 0~100 (0=최완화, 100=최긴축)
    signals: dict        # 구성 신호 상세
    is_opportunity: bool # 역투자 기회 여부 (극단 긴축 시 True)
    note: str            # 분석 메모


@dataclass
class LiquidityRisk:
    """유동성 리스크 분석 결과

    하워드 막스: 자본시장이 경색되면 아무리 건실한 기업도
    채무불이행 위험에 처할 수 있다.
    """
    level: str           # LOW / MEDIUM / HIGH
    ratio: float         # 현재 거래량 / 평균 거래량 비율
    gap_down_count: int  # 최근 갭다운 발생 횟수 (유동성 위기 신호)
    divergence: bool     # 가격-거래량 다이버전스 여부
    note: str            # 분석 메모


# ---------------------------------------------------------------------------
# 신용사이클 분석기
# ---------------------------------------------------------------------------

class CreditCycleAnalyzer:
    """하워드 막스 신용/자본시장 사이클 분석기

    "투자와 마켓사이클의 법칙" 제9장 신용사이클 개념:
    - 신용환경 추정: ATR(변동성) + 거래량 + 갭다운 빈도 조합
    - 유동성 리스크: 거래량 급감 + 가격-거래량 다이버전스
    - 확률분포 이동: 신용긴축 = 좌측 이동, 신용완화 = 우측 이동

    주의: 실제 신용사이클은 기준금리, 회사채 스프레드 등 외부 데이터로
    측정하는 것이 이상적이나, 여기서는 가격/거래량 데이터만으로 추정한다.
    """

    def __init__(
        self,
        lookback_days: int = 60,
        atr_lookback: int = 60,
        volume_lookback: int = 20,
    ):
        """
        Args:
            lookback_days: 신용환경 측정 기준 기간 (기본 60 거래일)
            atr_lookback: ATR 기준 기간 (변동성 수준 측정)
            volume_lookback: 거래량 비교 기준 기간
        """
        self.lookback_days = lookback_days
        self.atr_lookback = atr_lookback
        self.volume_lookback = volume_lookback

    def get_credit_environment(self, df: pd.DataFrame) -> CreditEnvironment:
        """신용환경을 추정합니다.

        하워드 막스: 신용사이클이 부정적으로 바뀌어 만기채무를
        차환할 수 없게 되면 유동성 위기가 발생한다.

        추정 방법 (가격/거래량 데이터만 활용):
          1. ATR(변동성) 수준: 높으면 신용긴축 환경 시사
             - 불확실성 증가 → 대출기관 위험회피 상승
          2. 거래량 급감 여부: 자본시장 냉각 신호
             - 거래량 감소 = 자금 공급자 퇴장 시사
          3. 가격 갭다운 빈도: 유동성 위기 신호
             - 갭다운 = 급매도 = 자금 조달 압박

        Returns:
            CreditEnvironment: 신용환경 상태 및 점수
        """
        try:
            if len(df) < 20:
                return self._default_credit_env("데이터 부족 (최소 20봉 필요)")

            signals = {}
            lookback = min(self.lookback_days, len(df))

            # ----------------------------------------------------------
            # 신호 1: ATR 변동성 수준 (0~100점, 높을수록 긴축)
            # 단기 ATR이 장기 ATR보다 높으면 불확실성 증가 = 긴축 시사
            # ----------------------------------------------------------
            try:
                atr_series = atr(df, 14)
                atr_short = float(atr_series.iloc[-10:].mean())
                atr_long = float(atr_series.iloc[-lookback:].mean())
                if atr_long > 0:
                    atr_ratio = atr_short / atr_long
                    # ATR 비율 2.0이면 100점(극단 긴축), 0.5이면 0점(완화)
                    atr_score = float(
                        np.clip((atr_ratio - 0.5) / 1.5 * 100, 0, 100)
                    )
                else:
                    atr_score = 50.0
                signals["atr_tightness"] = round(atr_score, 1)
            except Exception:
                signals["atr_tightness"] = 50.0

            # ----------------------------------------------------------
            # 신호 2: 거래량 트렌드 (0~100점, 높을수록 긴축)
            # 거래량 급감 = 자본시장 냉각 = 신용긴축 환경 시사
            # ----------------------------------------------------------
            try:
                vol = df["volume"].astype(float)
                vol_recent = float(vol.iloc[-10:].mean())
                vol_base = float(vol.iloc[-lookback:].mean())
                if vol_base > 0:
                    vol_ratio = vol_recent / vol_base
                    # 거래량이 절반 이하 → 100점(긴축), 2배 이상 → 0점(완화)
                    vol_score = float(
                        np.clip((1.0 - vol_ratio + 0.5) / 1.0 * 100, 0, 100)
                    )
                else:
                    vol_score = 50.0
                signals["volume_decline"] = round(vol_score, 1)
            except Exception:
                signals["volume_decline"] = 50.0

            # ----------------------------------------------------------
            # 신호 3: 갭다운 빈도 (0~100점, 높을수록 긴축)
            # 갭다운 = 급매도 = 자금 조달 압박 = 유동성 위기 신호
            # ----------------------------------------------------------
            try:
                open_prices = df["open"].astype(float)
                prev_close = df["close"].astype(float).shift(1)
                gap_pct = (open_prices - prev_close) / prev_close * 100
                recent_gap = gap_pct.iloc[-20:]

                # 갭다운 (-1% 이상 하락 갭) 발생 비율
                gap_down_ratio = float((recent_gap < -1.0).mean())
                gap_score = float(np.clip(gap_down_ratio * 300, 0, 100))
                signals["gap_down_freq"] = round(gap_score, 1)
            except Exception:
                signals["gap_down_freq"] = 50.0

            # ----------------------------------------------------------
            # 신호 4: RSI 극단 하락 빈도 (0~100점, 높을수록 긴축)
            # RSI 30 이하 빈번 = 강한 매도 압력 = 신용긴축 환경 시사
            # ----------------------------------------------------------
            try:
                close = df["close"].astype(float)
                rsi_series = rsi(close, 14)
                recent_rsi = rsi_series.iloc[-20:]
                oversold_ratio = float((recent_rsi < 30).mean())
                rsi_score = float(np.clip(oversold_ratio * 300, 0, 100))
                signals["rsi_oversold_freq"] = round(rsi_score, 1)
            except Exception:
                signals["rsi_oversold_freq"] = 50.0

            # ----------------------------------------------------------
            # 종합 신용환경 점수 계산 (가중 평균)
            # 변동성과 거래량에 더 높은 가중치 부여 (직접적 시장 신호)
            # ----------------------------------------------------------
            weights = {
                "atr_tightness": 0.35,
                "volume_decline": 0.35,
                "gap_down_freq": 0.20,
                "rsi_oversold_freq": 0.10,
            }
            score = sum(signals[k] * weights[k] for k in weights)
            score = round(float(np.clip(score, 0, 100)), 1)

            # 신용환경 상태 판단
            if score >= CREDIT_TIGHT_THRESHOLD:
                status = CREDIT_TIGHT
            elif score <= CREDIT_EASY_THRESHOLD:
                status = CREDIT_EASY
            else:
                status = CREDIT_NORMAL

            # 역투자 기회 감지:
            # 극단 긴축(점수 75 이상) = 최고의 매수 기회 가능성
            # (하워드 막스: 신용사이클의 닫힌 국면은 저가매수 가능성 최고)
            is_opportunity = score >= 75.0

            note = self._build_credit_note(score, status, signals, is_opportunity)

            return CreditEnvironment(
                status=status,
                score=score,
                signals=signals,
                is_opportunity=is_opportunity,
                note=note,
            )

        except Exception as exc:
            return self._default_credit_env(f"분석 오류: {exc}")

    def get_liquidity_risk(self, df: pd.DataFrame) -> LiquidityRisk:
        """유동성 리스크를 분석합니다.

        하워드 막스: "장기투자를 위해 단기로 빌리는 방식"은
        신용시장이 닫히면 위기가 된다.

        측정 방법:
          - 거래량 평균 대비 현재 비율 (급감 = 위험)
          - 가격-거래량 다이버전스 (가격 상승 + 거래량 감소 = 위험)
          - 가격 갭다운 발생 횟수 (급매도 = 자금 조달 압박)

        Returns:
            LiquidityRisk: 유동성 리스크 수준
        """
        try:
            if len(df) < 10:
                return LiquidityRisk(
                    level=LIQUIDITY_LOW,
                    ratio=1.0,
                    gap_down_count=0,
                    divergence=False,
                    note="데이터 부족",
                )

            # ----------------------------------------------------------
            # 1. 거래량 비율 계산
            # ----------------------------------------------------------
            vol = df["volume"].astype(float)
            vol_window = min(self.volume_lookback, len(df))
            vol_base_window = min(60, len(df))

            vol_recent = float(vol.iloc[-5:].mean())
            vol_avg = float(vol.iloc[-vol_base_window:].mean())

            ratio = vol_recent / vol_avg if vol_avg > 0 else 1.0
            ratio = round(ratio, 3)

            # ----------------------------------------------------------
            # 2. 가격 갭다운 횟수 (최근 20봉)
            # ----------------------------------------------------------
            gap_down_count = 0
            try:
                open_prices = df["open"].astype(float)
                prev_close = df["close"].astype(float).shift(1)
                gap_pct = (open_prices - prev_close) / prev_close * 100
                recent_window = min(20, len(df))
                gap_down_count = int((gap_pct.iloc[-recent_window:] < -1.0).sum())
            except Exception:
                gap_down_count = 0

            # ----------------------------------------------------------
            # 3. 가격-거래량 다이버전스 감지
            # 가격 상승 + 거래량 감소 = 상승 소진 (유동성 약화)
            # 가격 하락 + 거래량 증가 = 공황 매도 (유동성 위기)
            # ----------------------------------------------------------
            divergence = False
            try:
                close = df["close"].astype(float)
                n = min(10, len(df) - 1)
                price_trend = float(close.iloc[-1]) - float(close.iloc[-(n + 1)])
                vol_trend = float(vol.iloc[-5:].mean()) - float(
                    vol.iloc[-(n + 1):-5].mean()
                ) if len(df) > n + 5 else 0.0

                # 가격 하락 + 거래량 증가 = 공황 매도 (가장 위험한 다이버전스)
                if price_trend < 0 and vol_trend > 0:
                    divergence = True
                # 가격 상승 + 거래량 급감 = 상승 소진 (유동성 약화)
                elif price_trend > 0 and ratio < 0.5:
                    divergence = True
            except Exception:
                divergence = False

            # ----------------------------------------------------------
            # 유동성 리스크 수준 판단
            # ----------------------------------------------------------
            if (ratio > LIQUIDITY_HIGH_THRESHOLD or
                    (gap_down_count >= 3 and divergence)):
                level = LIQUIDITY_HIGH
                note = (
                    f"유동성 위기 신호: 거래량비율={ratio:.2f}, "
                    f"갭다운={gap_down_count}회"
                )
            elif (ratio > LIQUIDITY_MEDIUM_THRESHOLD or
                  gap_down_count >= 2 or divergence):
                level = LIQUIDITY_MEDIUM
                note = (
                    f"유동성 주의: 거래량비율={ratio:.2f}, "
                    f"갭다운={gap_down_count}회, "
                    f"다이버전스={divergence}"
                )
            else:
                level = LIQUIDITY_LOW
                note = (
                    f"유동성 정상: 거래량비율={ratio:.2f}"
                )

            return LiquidityRisk(
                level=level,
                ratio=ratio,
                gap_down_count=gap_down_count,
                divergence=divergence,
                note=note,
            )

        except Exception as exc:
            return LiquidityRisk(
                level=LIQUIDITY_LOW,
                ratio=1.0,
                gap_down_count=0,
                divergence=False,
                note=f"분석 오류: {exc}",
            )

    def get_probability_shift(
        self,
        credit_env: CreditEnvironment,
        liquidity: LiquidityRisk,
    ) -> dict:
        """확률분포 이동 방향을 계산합니다.

        하워드 막스: 사이클 내의 포지션이 바뀌면 확률도 변한다.
        - 신용완화 → 확률분포 우측 이동 (수익 가능성 증가)
        - 신용긴축 → 확률분포 좌측 이동 (손실 가능성 증가)
        - 극단 긴축 + 공포 = 역투자 기회 (분포 반전 조짐)

        Returns:
            dict: {
                direction: "RIGHT"(유리) / "CENTER"(중립) / "LEFT"(불리),
                magnitude: 이동 강도 (0.0~1.0),
                profit_edge: 수익 우위 (양수=유리, 음수=불리),
                note: 분석 메모
            }
        """
        try:
            credit_score = credit_env.score
            liquidity_level = liquidity.level

            # 기본 확률 이동: 신용 점수 기반
            # credit_score 50 = 중립, 0 = 완화(+유리), 100 = 긴축(-불리)
            base_edge = (50.0 - credit_score) / 50.0  # -1.0 ~ +1.0

            # 유동성 리스크 보정
            liquidity_penalty = {
                LIQUIDITY_LOW: 0.0,
                LIQUIDITY_MEDIUM: -0.1,
                LIQUIDITY_HIGH: -0.2,
            }.get(liquidity_level, 0.0)

            # 극단 긴축 + 역투자 기회 보정
            # (하워드 막스: 신용사이클의 극단 닫힌 국면 = 저가매수 최고 기회)
            opportunity_bonus = 0.3 if credit_env.is_opportunity else 0.0

            profit_edge = float(
                np.clip(base_edge + liquidity_penalty + opportunity_bonus, -1.0, 1.0)
            )
            profit_edge = round(profit_edge, 3)

            magnitude = abs(profit_edge)

            if profit_edge > 0.1:
                direction = "RIGHT"  # 수익 확률 우위
                note = (
                    f"확률분포 우측 이동: 수익 우위 {profit_edge:+.2f} "
                    f"(신용={credit_env.status})"
                )
            elif profit_edge < -0.1:
                direction = "LEFT"   # 손실 확률 우위
                note = (
                    f"확률분포 좌측 이동: 손실 우위 {profit_edge:+.2f} "
                    f"(신용={credit_env.status})"
                )
            else:
                direction = "CENTER"
                note = f"확률분포 중립: edge={profit_edge:+.2f}"

            return {
                "direction": direction,
                "magnitude": round(magnitude, 3),
                "profit_edge": profit_edge,
                "note": note,
            }

        except Exception:
            return {
                "direction": "CENTER",
                "magnitude": 0.0,
                "profit_edge": 0.0,
                "note": "확률분포 계산 오류",
            }

    def analyze(self, df: pd.DataFrame) -> dict:
        """전체 신용사이클 분석을 수행합니다.

        Args:
            df: OHLCV DataFrame

        Returns:
            dict: {
                credit_env: CreditEnvironment,
                liquidity: LiquidityRisk,
                probability_shift: dict,
                position_multiplier: float (포지션 크기 조정 배수)
            }
        """
        try:
            credit_env = self.get_credit_environment(df)
            liquidity = self.get_liquidity_risk(df)
            prob_shift = self.get_probability_shift(credit_env, liquidity)

            # 포지션 크기 조정 배수 계산
            # TIGHT: 0.5~0.7배, NORMAL: 1.0배, EASY: 1.1~1.2배
            if credit_env.status == CREDIT_TIGHT:
                if credit_env.is_opportunity:
                    # 극단 긴축 = 역투자 매수 기회 → 소규모 분할 매수
                    pos_multiplier = 0.7
                else:
                    pos_multiplier = 0.5
            elif credit_env.status == CREDIT_EASY:
                pos_multiplier = 1.1
            else:
                pos_multiplier = 1.0

            # 유동성 위기 시 추가 축소
            if liquidity.level == LIQUIDITY_HIGH:
                pos_multiplier *= 0.7
            elif liquidity.level == LIQUIDITY_MEDIUM:
                pos_multiplier *= 0.85

            pos_multiplier = round(float(np.clip(pos_multiplier, 0.3, 1.2)), 2)

            return {
                "credit_env": credit_env,
                "liquidity": liquidity,
                "probability_shift": prob_shift,
                "position_multiplier": pos_multiplier,
            }

        except Exception as exc:
            default_env = self._default_credit_env(f"분석 오류: {exc}")
            default_liq = LiquidityRisk(
                level=LIQUIDITY_LOW,
                ratio=1.0,
                gap_down_count=0,
                divergence=False,
                note="기본값",
            )
            return {
                "credit_env": default_env,
                "liquidity": default_liq,
                "probability_shift": {
                    "direction": "CENTER",
                    "magnitude": 0.0,
                    "profit_edge": 0.0,
                    "note": "오류로 인한 기본값",
                },
                "position_multiplier": 1.0,
            }

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _default_credit_env(self, note: str) -> CreditEnvironment:
        """기본(중립) CreditEnvironment를 반환합니다."""
        return CreditEnvironment(
            status=CREDIT_NORMAL,
            score=50.0,
            signals={},
            is_opportunity=False,
            note=note,
        )

    def _build_credit_note(
        self,
        score: float,
        status: str,
        signals: dict,
        is_opportunity: bool,
    ) -> str:
        """신용환경 분석 요약 메모를 생성합니다."""
        parts = [
            f"신용환경={status}",
            f"긴축점수={score:.1f}",
        ]
        if is_opportunity:
            parts.append("역투자기회=True (극단긴축, 저가매수 가능성)")

        sig_str = " ".join(f"{k[:6]}={v:.0f}" for k, v in signals.items())
        parts.append(f"[{sig_str}]")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# 모듈 수준 편의 함수
# ---------------------------------------------------------------------------

_default_analyzer = CreditCycleAnalyzer()


def get_credit_environment(df: pd.DataFrame) -> CreditEnvironment:
    """신용환경을 분석합니다.

    Args:
        df: OHLCV DataFrame

    Returns:
        CreditEnvironment
    """
    return _default_analyzer.get_credit_environment(df)


def get_liquidity_risk(df: pd.DataFrame) -> LiquidityRisk:
    """유동성 리스크를 분석합니다.

    Args:
        df: OHLCV DataFrame

    Returns:
        LiquidityRisk
    """
    return _default_analyzer.get_liquidity_risk(df)


def analyze_credit_cycle(df: pd.DataFrame) -> dict:
    """전체 신용사이클 분석을 수행합니다.

    Args:
        df: OHLCV DataFrame

    Returns:
        dict: credit_env, liquidity, probability_shift, position_multiplier
    """
    return _default_analyzer.analyze(df)
