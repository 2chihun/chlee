"""켄 피셔 "주식시장은 어떻게 반복되는가" 기반 시장 기억 분석 모듈

핵심 이론 (150페이지 전체 학습 반영):
  - 시장은 기억하지만 투자자는 잊는다
  - "이번에는 다르다"는 생각은 언제나 틀렸다
  - 극단적 비관론 = 매수 기회 (불신의 비관론, Wall of Worry)
  - 변동성은 정상이다 - 높은 변동성 = 공포 = 기회
  - 실업률은 후행지표 - 고점 전 매수가 최적
  - 약세장 후 강한 반등은 반복되는 패턴
  - V자 회복 대칭성 - 약세장 손실의 2/3가 말기에 발생
  - 강세장 초기 3개월 평균 +23.1%, 12개월 평균 +46.6%
  - 극단 비관론 축적 = "이번엔 다르다" 착각 = 역발상 매수 신호
  - 돈과 시장은 절대 잊지 않지만, 사람들은 반드시 잊는다

적용 모듈:
  - MarketMemoryAnalyzer: 시장 기억 상실 패턴 (공포/비관론 극단) 감지
  - WallOfWorryDetector: 불신의 비관론 감지 - 강세장 지속 신호
  - VolatilityNormalizer: 변동성 정상화 - 공포 = 기회 확인
  - BearMarketRecoveryDetector: 약세장→강세장 전환 조기 감지
  - VShapeSymmetryBonus: V자 회복 대칭성 측정
  - EarlyBullPhaseDetector: 강세장 초기 구간 감지
  - ExtremeDistrustAccumulator: 극단 비관론 축적 감지
  - FisherSignal: 통합 켄 피셔 시그널
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class FisherSignal:
    """켄 피셔 통합 시그널"""
    # 불신의 비관론 강도 (0~1, 높을수록 비관론 극단 → 매수 기회)
    wall_of_worry_score: float = 0.0
    # 변동성 정상화 점수 (0~1, 높을수록 공포/변동성 과잉)
    volatility_fear_score: float = 0.0
    # 약세장 회복 조기 신호 강도 (0~1)
    recovery_signal: float = 0.0
    # 켄 피셔 종합 신호 (-1~+1: -1=위험, 0=중립, +1=기회)
    fisher_composite: float = 0.0
    # 포지션 조정 배수 (0.5~1.5)
    position_multiplier: float = 1.0
    # 신뢰도 조정값 (+/-)
    confidence_delta: float = 0.0
    # 분석 노트
    note: str = ""
    # V자 회복 대칭성 (0~1, 높을수록 V자 반등 강함)
    v_shape_score: float = 0.0
    # 강세장 초기 여부 (약세장 바닥 후 0~90거래일)
    early_bull_phase: bool = False
    # 극단 비관론 축적 여부 ("이번에는 다르다" 착각 극단)
    extreme_distrust: bool = False
    # 12개월 기대 수익률 (강세장 초기 역사 평균 기반)
    expected_12m_return: float = 0.0


class VolatilityNormalizer:
    """변동성 정상화 분석기

    켄 피셔 핵심 원칙:
    - 변동성은 항상 존재하며 이것이 정상이다
    - 투자자들이 변동성을 비정상이라 착각할 때 = 공포 극단
    - 공포 극단 = 매수 기회 (단, 단기 추세와 결합 필요)
    - 안정적 성과가 지속되는 구간이 오히려 위험 (버블 전조)
    """

    def __init__(self, lookback: int = 60, atr_multiple_threshold: float = 2.0):
        self.lookback = lookback
        self.atr_multiple_threshold = atr_multiple_threshold

    def analyze(self, df: pd.DataFrame) -> dict:
        """변동성 공포 점수를 계산합니다.

        Returns:
            dict: {
                fear_score: 공포 점수 (0~1, 높을수록 공포 극단),
                volatility_regime: 'FEAR'/'NORMAL'/'CALM',
                is_opportunity: 공포 극단 매수 기회 여부,
                note: 분석 노트
            }
        """
        if len(df) < self.lookback:
            return {
                "fear_score": 0.0,
                "volatility_regime": "NORMAL",
                "is_opportunity": False,
                "note": "데이터 부족"
            }

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # ATR 계산 (변동성 측정)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        # 현재 ATR vs 장기 평균 ATR 비교
        current_atr = atr.iloc[-1]
        avg_atr = atr.rolling(self.lookback).mean().iloc[-1]

        if avg_atr <= 0:
            return {
                "fear_score": 0.0,
                "volatility_regime": "NORMAL",
                "is_opportunity": False,
                "note": "ATR 계산 불가"
            }

        atr_ratio = current_atr / avg_atr

        # 일별 수익률 변동성
        returns = close.pct_change().dropna()
        current_vol = returns.tail(10).std()
        avg_vol = returns.rolling(self.lookback).std().iloc[-1]

        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        # 공포 점수 = ATR 비율 + 수익률 변동성 비율 가중 평균
        fear_raw = (atr_ratio * 0.5 + vol_ratio * 0.5 - 1.0)
        fear_score = float(min(max(fear_raw / self.atr_multiple_threshold, 0.0), 1.0))

        # 변동성 레짐 판단
        if atr_ratio >= self.atr_multiple_threshold:
            regime = "FEAR"
            is_opportunity = True
            note = (
                f"변동성 공포 극단: ATR비율={atr_ratio:.2f}x "
                f"(켄피셔: 변동성=정상, 공포=기회)"
            )
        elif atr_ratio <= 0.6:
            regime = "CALM"
            is_opportunity = False
            note = (
                f"변동성 과도 안정: ATR비율={atr_ratio:.2f}x "
                f"(켄피셔: 안정적 수익 지속 구간이 위험)"
            )
        else:
            regime = "NORMAL"
            is_opportunity = False
            note = f"변동성 정상: ATR비율={atr_ratio:.2f}x"

        return {
            "fear_score": fear_score,
            "volatility_regime": regime,
            "is_opportunity": is_opportunity,
            "note": note
        }


class WallOfWorryDetector:
    """불신의 비관론(Wall of Worry) 감지기

    켄 피셔 원칙:
    - 강세장은 "불신의 비관론"을 타고 오른다
    - 호재가 나쁜 방향으로 해석되는 시기 = 강세장 지속 신호
    - 극단적 비관론(뉴 노멀, 더블딥 공포) 발생 직후 급등이 반복됨
    - 2009년 3월 바닥 후 3개월 44.1%, 12개월 74.3% 상승

    구현: 가격 데이터 기반 프록시 지표 사용
    - 가격 회복 시 거래량 부진 = "믿지 않는 랠리" = Wall of Worry
    - RSI 낮은데 가격 반등 = 기술적 불신 랠리
    - BB 하단 이탈 후 복귀 = 극단 공포 후 회복
    """

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def detect(self, df: pd.DataFrame) -> dict:
        """불신의 비관론 패턴을 감지합니다.

        Returns:
            dict: {
                wow_score: 불신 비관론 점수 (0~1),
                pattern: 'DISTRUST_RALLY'/'FEAR_EXTREME'/'NORMAL',
                confidence_boost: 신뢰도 부스트 (+값),
                note: 노트
            }
        """
        if len(df) < self.lookback + 10:
            return {
                "wow_score": 0.0,
                "pattern": "NORMAL",
                "confidence_boost": 0.0,
                "note": "데이터 부족"
            }

        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        # RSI 계산
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        # 볼린저 밴드
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_lower = bb_mid - 2 * bb_std
        bb_pctb = (close - bb_lower) / (4 * bb_std)  # 0=하단, 1=상단

        recent_bb = bb_pctb.iloc[-5:]
        bb_recovery = (recent_bb.iloc[-1] > 0.3) and (recent_bb.min() < 0.1)

        # 가격 회복 + 거래량 부진 = 불신 랠리
        price_change_5d = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]
        vol_change_5d = volume.tail(5).mean() / volume.tail(20).mean()
        distrust_rally = price_change_5d > 0.02 and vol_change_5d < 0.8

        # 극단 공포 후 반등 (RSI < 30에서 회복)
        rsi_extreme_recovery = (
            rsi.iloc[-10:-1].min() < 30 and current_rsi > 35
        )

        # 패턴 결정
        wow_score = 0.0
        pattern = "NORMAL"
        confidence_boost = 0.0
        notes = []

        if distrust_rally:
            wow_score += 0.4
            pattern = "DISTRUST_RALLY"
            confidence_boost += 0.1
            notes.append(f"불신 랠리(가격↑{price_change_5d:.1%}, 거래량{vol_change_5d:.1f}x)")

        if rsi_extreme_recovery:
            wow_score += 0.4
            pattern = "FEAR_EXTREME"
            confidence_boost += 0.15
            notes.append(f"공포 극단 회복(RSI저점→{current_rsi:.0f})")

        if bb_recovery:
            wow_score += 0.2
            confidence_boost += 0.05
            notes.append("BB하단 공포 후 회복")

        wow_score = min(wow_score, 1.0)

        note = " | ".join(notes) if notes else "불신 비관론 패턴 없음"
        if wow_score > 0:
            note += " → 켄피셔: 불신 랠리=강세장 지속 신호"

        return {
            "wow_score": wow_score,
            "pattern": pattern,
            "confidence_boost": confidence_boost,
            "note": note
        }


class BearMarketRecoveryDetector:
    """약세장→강세장 전환 조기 감지기

    켄 피셔 핵심 데이터:
    - 경기침체 종료 전 강세장 시작, 평균 27.5% 선반영
    - 약세장 바닥 후 3개월 44% 급등 패턴 반복
    - 실업률 고점 6개월 전 매수: 31.2% vs 고점 시 14.8%
    - "이번에는 달라" 비관론이 절정일 때 바닥 형성

    구현 전략:
    - 52주 저점 대비 반등률로 회복 강도 측정
    - 강한 반등 + 높은 거래량 = 강세장 초기 신호
    - 과도한 하락 후 기술적 반등 = 역발상 기회
    """

    def __init__(
        self,
        decline_threshold: float = 0.20,  # 약세장 기준 하락률 (20%)
        recovery_threshold: float = 0.10,  # 회복 신호 반등률 (10%)
        lookback_days: int = 252,
    ):
        self.decline_threshold = decline_threshold
        self.recovery_threshold = recovery_threshold
        self.lookback_days = lookback_days

    def detect(self, df: pd.DataFrame) -> dict:
        """약세장 회복 신호를 감지합니다.

        Returns:
            dict: {
                recovery_score: 회복 신호 강도 (0~1),
                bear_depth: 약세장 낙폭 (-값),
                recovery_pct: 바닥 대비 반등률,
                is_early_bull: 강세장 초기 여부,
                confidence_boost: 신뢰도 부스트,
                note: 노트
            }
        """
        lookback = min(self.lookback_days, len(df))
        if lookback < 60:
            return {
                "recovery_score": 0.0,
                "bear_depth": 0.0,
                "recovery_pct": 0.0,
                "is_early_bull": False,
                "confidence_boost": 0.0,
                "note": "데이터 부족"
            }

        close = df["close"].astype(float)
        recent = close.tail(lookback)
        volume = df["volume"].astype(float)

        # 기간 내 최고가/최저가
        peak = recent.max()
        trough = recent.min()
        current = close.iloc[-1]

        # 최고가 대비 낙폭
        bear_depth = (trough - peak) / peak  # 음수

        # 최저가 대비 반등률
        recovery_pct = (current - trough) / trough if trough > 0 else 0.0

        # 약세장 기준 충족 여부
        is_bear = bear_depth <= -self.decline_threshold

        # 회복 기준 충족 여부
        is_recovering = recovery_pct >= self.recovery_threshold

        # 거래량 확인 (회복 시 거래량 증가 = 신뢰도 상승)
        vol_ratio = volume.tail(10).mean() / volume.tail(60).mean()

        # 강세장 초기 신호: 약세장 + 회복 중 + 거래량 증가
        is_early_bull = is_bear and is_recovering and vol_ratio > 1.2

        # 회복 점수 계산
        recovery_score = 0.0
        confidence_boost = 0.0
        notes = []

        if is_bear:
            depth_score = min(abs(bear_depth) / 0.40, 1.0)  # 40% 하락 시 최대
            recovery_score += depth_score * 0.4
            notes.append(f"약세장낙폭{bear_depth:.1%}")

        if is_recovering:
            rec_score = min(recovery_pct / 0.30, 1.0)  # 30% 반등 시 최대
            recovery_score += rec_score * 0.4
            confidence_boost += 0.1
            notes.append(f"반등{recovery_pct:.1%}")

        if vol_ratio > 1.2:
            recovery_score += 0.2
            confidence_boost += 0.05
            notes.append(f"거래량확인({vol_ratio:.1f}x)")

        if is_early_bull:
            confidence_boost += 0.1
            notes.append("강세장초기신호")

        recovery_score = min(recovery_score, 1.0)

        note = " | ".join(notes) if notes else "회복 신호 없음"
        if is_early_bull:
            note += " → 켄피셔: 약세장후강반등패턴(역사적평균27.5%선반영)"

        return {
            "recovery_score": recovery_score,
            "bear_depth": bear_depth,
            "recovery_pct": recovery_pct,
            "is_early_bull": is_early_bull,
            "confidence_boost": confidence_boost,
            "note": note
        }


class VShapeSymmetryBonus:
    """V자 회복 대칭성 분석기

    켄 피셔 핵심 원칙:
    - 약세장 손실의 약 2/3가 말기(마지막 구간)에 집중 발생
    - 극단적 낙폭 → 극단적 반등의 대칭성 (V자 패턴)
    - 낙폭이 클수록 반등도 강하다 (역사적 검증)
    - 강세장 초기 3개월 평균 +23.1% 수익률
    - 2009년 3월 후 3개월 +39.3%, 12개월 +68.6%

    구현:
    - 낙폭 기울기(fall_slope) vs 반등 기울기(recovery_slope) 비율
    - 대칭성이 높을수록 V자 신뢰도 증가
    """

    def __init__(
        self,
        slope_window: int = 20,     # 기울기 계산 윈도우 (거래일)
        min_fall_pct: float = 0.15, # 최소 낙폭 기준 (15%)
    ):
        self.slope_window = slope_window
        self.min_fall_pct = min_fall_pct

    def analyze(self, df: pd.DataFrame) -> dict:
        """V자 회복 대칭성을 분석합니다.

        Returns:
            dict: {
                v_shape_score: V자 대칭성 점수 (0~1),
                fall_depth: 최대 낙폭 (음수),
                recovery_slope_ratio: 반등/낙폭 기울기 비율,
                confidence_boost: 신뢰도 부스트,
                note: 노트
            }
        """
        lookback = min(252, len(df))
        if lookback < 60:
            return {
                "v_shape_score": 0.0,
                "fall_depth": 0.0,
                "recovery_slope_ratio": 0.0,
                "confidence_boost": 0.0,
                "note": "데이터 부족"
            }

        close = df["close"].astype(float).tail(lookback)

        # 최저점 위치 탐색
        trough_idx = close.idxmin()
        trough_pos = close.index.get_loc(trough_idx)

        # 최저점 이전 고점 탐색
        if trough_pos < self.slope_window:
            return {
                "v_shape_score": 0.0,
                "fall_depth": 0.0,
                "recovery_slope_ratio": 0.0,
                "confidence_boost": 0.0,
                "note": "최저점 이전 데이터 부족"
            }

        pre_trough = close.iloc[:trough_pos]
        peak_val = pre_trough.max()
        trough_val = close.iloc[trough_pos]
        current_val = close.iloc[-1]

        # 낙폭 계산
        fall_depth = (trough_val - peak_val) / peak_val  # 음수

        if abs(fall_depth) < self.min_fall_pct:
            return {
                "v_shape_score": 0.0,
                "fall_depth": fall_depth,
                "recovery_slope_ratio": 0.0,
                "confidence_boost": 0.0,
                "note": f"낙폭 불충분({fall_depth:.1%})"
            }

        # 낙폭 기울기: 최저점 직전 window 기간의 하락률
        fall_window_start = max(0, trough_pos - self.slope_window)
        fall_window_close = close.iloc[fall_window_start:trough_pos + 1]
        if len(fall_window_close) < 2:
            fall_slope = abs(fall_depth)
        else:
            fall_slope = abs(
                (fall_window_close.iloc[-1] - fall_window_close.iloc[0])
                / fall_window_close.iloc[0]
            )

        # 반등 기울기: 최저점 이후 현재까지의 상승률
        post_trough = close.iloc[trough_pos:]
        recovery_window = post_trough.iloc[:min(self.slope_window, len(post_trough))]
        if len(recovery_window) < 2:
            return {
                "v_shape_score": 0.0,
                "fall_depth": fall_depth,
                "recovery_slope_ratio": 0.0,
                "confidence_boost": 0.0,
                "note": "반등 구간 데이터 부족"
            }

        recovery_slope = abs(
            (recovery_window.iloc[-1] - recovery_window.iloc[0])
            / recovery_window.iloc[0]
        )

        # 대칭성 비율 = 반등 기울기 / 낙폭 기울기
        slope_ratio = (
            recovery_slope / fall_slope if fall_slope > 0 else 0.0
        )

        # V자 점수 (0~1): 대칭성 + 낙폭 크기 종합
        depth_bonus = min(abs(fall_depth) / 0.40, 1.0)  # 40% 낙폭 시 최대
        symmetry_score = min(slope_ratio, 2.0) / 2.0     # 2배 대칭 시 최대
        v_shape_score = float(
            depth_bonus * 0.5 + symmetry_score * 0.5
        )
        v_shape_score = min(v_shape_score, 1.0)

        # 신뢰도 부스트 결정
        confidence_boost = 0.0
        notes = [f"낙폭{fall_depth:.1%}"]
        notes.append(f"반등/낙폭기울기비={slope_ratio:.2f}x")

        if v_shape_score > 0.8 or (symmetry_score > 0.5 and abs(fall_depth) > 0.30):
            confidence_boost = 0.15
            notes.append("강한V자대칭")
        elif v_shape_score > 0.5:
            confidence_boost = 0.10
            notes.append("V자대칭감지")

        # 현재 최저점 대비 반등률
        recovery_from_trough = (current_val - trough_val) / trough_val
        notes.append(f"바닥대비+{recovery_from_trough:.1%}")

        note = " | ".join(notes)
        if confidence_boost > 0:
            note += " → 켄피셔: 낙폭클수록반등강함(약세장손실2/3말기집중)"

        return {
            "v_shape_score": v_shape_score,
            "fall_depth": fall_depth,
            "recovery_slope_ratio": slope_ratio,
            "confidence_boost": confidence_boost,
            "note": note
        }


class EarlyBullPhaseDetector:
    """강세장 초기 구간 감지기

    켄 피셔 핵심 데이터 (역사적 통계):
    - 강세장 초기 3개월 평균 수익률: +23.1%
    - 강세장 초기 12개월 평균 수익률: +46.6%
    - 2009년 3월 후 3개월: +39.3%, 12개월: +68.6%
    - 경기침체 종료 전 강세장 시작 (평균 27.5% 선반영)
    - 강세장 초기에 매수하지 못하면 전체 상승의 대부분을 놓침

    구현:
    - 52주 저점 대비 반등 + 거래량 확인 → 바닥 감지
    - 바닥 감지 후 경과 거래일로 초기/중기/후기 판단
    - 초기(0~90일): 최대 신뢰도 부스트
    - 중기(90~180일): 중간 부스트
    """

    def __init__(
        self,
        lookback_days: int = 252,
        recovery_threshold: float = 0.10,   # 바닥 대비 최소 반등률
        vol_confirm_ratio: float = 1.2,     # 거래량 확인 배수
        early_phase_days: int = 90,         # 초기 구간 거래일
        mid_phase_days: int = 180,          # 중기 구간 거래일
    ):
        self.lookback_days = lookback_days
        self.recovery_threshold = recovery_threshold
        self.vol_confirm_ratio = vol_confirm_ratio
        self.early_phase_days = early_phase_days
        self.mid_phase_days = mid_phase_days

    def detect(self, df: pd.DataFrame) -> dict:
        """강세장 초기 구간을 감지합니다.

        Returns:
            dict: {
                is_early_bull: 강세장 초기 여부,
                phase: 'EARLY'/'MID'/'LATE'/'NONE',
                days_since_bottom: 바닥 이후 경과 거래일,
                expected_12m_return: 12개월 기대 수익률,
                confidence_boost: 신뢰도 부스트,
                note: 노트
            }
        """
        lookback = min(self.lookback_days, len(df))
        if lookback < 90:
            return {
                "is_early_bull": False,
                "phase": "NONE",
                "days_since_bottom": 0,
                "expected_12m_return": 0.0,
                "confidence_boost": 0.0,
                "note": "데이터 부족"
            }

        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
        recent_close = close.tail(lookback)

        # 52주 최저점 탐색
        trough_idx = recent_close.idxmin()
        trough_pos = recent_close.index.get_loc(trough_idx)
        trough_val = recent_close.iloc[trough_pos]

        # 최저점 이후 데이터
        post_trough_close = recent_close.iloc[trough_pos:]
        days_since_bottom = len(post_trough_close) - 1  # 최저점 이후 거래일

        if days_since_bottom < 5:
            return {
                "is_early_bull": False,
                "phase": "NONE",
                "days_since_bottom": days_since_bottom,
                "expected_12m_return": 0.0,
                "confidence_boost": 0.0,
                "note": "바닥 이후 일수 부족"
            }

        current_val = close.iloc[-1]
        recovery_pct = (current_val - trough_val) / trough_val

        # 거래량 확인 (반등 시 거래량 증가)
        vol_ratio = volume.tail(10).mean() / volume.tail(60).mean()
        vol_confirmed = vol_ratio >= self.vol_confirm_ratio

        # 강세장 초기 조건: 충분한 반등 + (선택적) 거래량 확인
        is_recovering = recovery_pct >= self.recovery_threshold

        # 최저점 이전 약세장 여부 (20% 이상 하락)
        pre_trough_close = recent_close.iloc[:trough_pos + 1]
        if len(pre_trough_close) < 20:
            peak_before = pre_trough_close.max()
        else:
            peak_before = pre_trough_close.max()

        bear_depth = (trough_val - peak_before) / peak_before if peak_before > 0 else 0.0
        is_from_bear = bear_depth <= -0.20

        # 강세장 초기 = 약세장 이후 회복 + 회복 시작
        is_early_bull = is_from_bear and is_recovering

        # 구간 판단 및 기대 수익률 설정
        confidence_boost = 0.0
        expected_12m = 0.0
        notes = []

        if is_early_bull:
            notes.append(f"약세장낙폭{bear_depth:.1%}→반등{recovery_pct:.1%}")

            if days_since_bottom <= self.early_phase_days:
                phase = "EARLY"
                # 강세장 초기 3개월: 평균 +23.1%, 12개월: +46.6%
                expected_12m = 0.466
                confidence_boost = 0.15
                notes.append(
                    f"강세장초기({days_since_bottom}일) "
                    f"켄피셔: 12개월기대+46.6%"
                )
            elif days_since_bottom <= self.mid_phase_days:
                phase = "MID"
                expected_12m = 0.231  # 3개월 평균 기준 환산
                confidence_boost = 0.10
                notes.append(
                    f"강세장중기({days_since_bottom}일) "
                    f"켄피셔: 3개월기대+23.1%"
                )
            else:
                phase = "LATE"
                expected_12m = 0.0
                confidence_boost = 0.05
                notes.append(f"강세장후기({days_since_bottom}일)")

            if vol_confirmed:
                confidence_boost += 0.03
                notes.append(f"거래량확인({vol_ratio:.1f}x)")
        else:
            phase = "NONE"
            notes.append("강세장 초기 조건 미충족")

        note = " | ".join(notes)

        return {
            "is_early_bull": is_early_bull,
            "phase": phase,
            "days_since_bottom": days_since_bottom,
            "expected_12m_return": expected_12m,
            "confidence_boost": confidence_boost,
            "note": note
        }


class ExtremeDistrustAccumulator:
    """극단 비관론 축적 감지기 ("이번에는 다르다" 착각 극단)

    켄 피셔 핵심 원칙:
    - 모든 위기마다 투자자들은 "이번에는 다르다"고 믿는다
    - 역사적으로 이 믿음은 100% 틀렸다
    - 극단 비관론이 지속적으로 축적될 때 = 강세장 전환 직전
    - 불신(Wall of Worry) + 공포(Volatility Fear) 동시 극단화 = 최강 매수 신호
    - "사람들은 반드시 잊지만 시장은 잊지 않는다"

    구현:
    - WallOfWorry + VolatilityFear 복합 점수로 비관론 측정
    - 연속 N일 이상 비관론 고점 유지 → 축적 상태 감지
    - 극단 축적 = 포지션 배수 추가 부스트
    """

    def __init__(
        self,
        accumulation_days: int = 5,        # 축적 판정 연속 일수
        distrust_threshold: float = 0.55,  # 비관론 점수 임계값 (0~1)
        lookback: int = 30,                # 분석 윈도우
    ):
        self.accumulation_days = accumulation_days
        self.distrust_threshold = distrust_threshold
        self.lookback = lookback

    def detect(
        self,
        df: pd.DataFrame,
        wow_score: float,
        vol_fear_score: float,
    ) -> dict:
        """극단 비관론 축적 여부를 감지합니다.

        Args:
            df: OHLCV DataFrame
            wow_score: WallOfWorry 점수 (0~1)
            vol_fear_score: 변동성 공포 점수 (0~1)

        Returns:
            dict: {
                extreme_distrust: 극단 비관론 축적 여부,
                distrust_score: 현재 비관론 점수 (0~1),
                confidence_boost: 신뢰도 부스트,
                position_boost: 포지션 배수 추가 (+값),
                note: 노트
            }
        """
        if len(df) < self.lookback:
            return {
                "extreme_distrust": False,
                "distrust_score": 0.0,
                "confidence_boost": 0.0,
                "position_boost": 0.0,
                "note": "데이터 부족"
            }

        close = df["close"].astype(float)

        # 현재 복합 비관론 점수 (0~1 정규화)
        # wow_score(0~1) + vol_fear_score(0~1) → 합산 후 0~1로 정규화
        raw_distrust = (wow_score + vol_fear_score) / 2.0
        distrust_score = float(min(raw_distrust, 1.0))

        # 연속 하락 일수 (시장 지속 약세 = 비관론 누적 프록시)
        returns = close.pct_change().tail(self.lookback)
        negative_days = (returns < -0.005).sum()  # 0.5% 이상 하락일 수
        total_days = len(returns.dropna())
        negative_ratio = negative_days / total_days if total_days > 0 else 0.0

        # 극단 비관론 조건:
        # 1) 현재 비관론 점수가 임계값 초과
        # 2) 최근 구간에서 하락일 비율이 높음 (60% 이상)
        high_distrust_now = distrust_score >= self.distrust_threshold
        persistent_decline = negative_ratio >= 0.55

        # RSI 극단 저점 확인 (20 이하 극단)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        rsi_extreme = current_rsi < 25  # RSI 25 이하 = 극단 공포

        # 극단 비관론 축적 판정
        extreme_distrust = (
            high_distrust_now and
            (persistent_decline or rsi_extreme)
        )

        # 부스트 계산
        confidence_boost = 0.0
        position_boost = 0.0
        notes = [f"비관론점수={distrust_score:.2f}"]

        if extreme_distrust:
            confidence_boost = 0.08
            position_boost = 0.15  # 포지션 배수 +15%
            notes.append(
                f"극단비관론축적(하락일{negative_ratio:.0%}, RSI={current_rsi:.0f})"
            )
            notes.append("켄피셔: 이번에도다르지않다=역발상매수신호")
        else:
            if high_distrust_now:
                notes.append(f"비관론높음(하락일{negative_ratio:.0%})")
            if rsi_extreme:
                notes.append(f"RSI극단({current_rsi:.0f})")

        note = " | ".join(notes)

        return {
            "extreme_distrust": extreme_distrust,
            "distrust_score": distrust_score,
            "confidence_boost": confidence_boost,
            "position_boost": position_boost,
            "note": note
        }


class MarketMemoryAnalyzer:
    """켄 피셔 시장 기억 분석기 (통합)

    "주식시장은 어떻게 반복되는가" 핵심 원칙 통합 적용 (150페이지 전체):
    1. 극단적 비관론 = 매수 기회 (불신의 비관론)
    2. 변동성 정상화 = 공포가 공포가 아님
    3. 약세장 후 강한 반등 = 반복 패턴
    4. "이번에는 다르다" 착각 = 항상 틀림
    5. 강세장은 비관론자들이 잘못된 주장을 고수하는 동안 지속
    6. V자 회복 대칭성 = 낙폭이 클수록 반등도 강함
    7. 강세장 초기 진입 = 가장 큰 수익 기회 (12개월 평균 +46.6%)
    8. 극단 비관론 축적 = "이번에도 역시 다르지 않다" 신호
    """

    def __init__(
        self,
        vol_lookback: int = 60,
        atr_threshold: float = 2.0,
        wow_lookback: int = 20,
        bear_decline: float = 0.20,
        bear_recovery: float = 0.10,
        v_shape_window: int = 20,
        early_bull_lookback: int = 252,
        distrust_accumulation_days: int = 5,
    ):
        self._vol = VolatilityNormalizer(vol_lookback, atr_threshold)
        self._wow = WallOfWorryDetector(wow_lookback)
        self._bear = BearMarketRecoveryDetector(bear_decline, bear_recovery)
        self._vshape = VShapeSymmetryBonus(v_shape_window)
        self._early_bull = EarlyBullPhaseDetector(early_bull_lookback, bear_recovery)
        self._distrust = ExtremeDistrustAccumulator(distrust_accumulation_days)

    def analyze(self, df: pd.DataFrame) -> FisherSignal:
        """켄 피셔 통합 시그널을 생성합니다.

        Args:
            df: OHLCV DataFrame

        Returns:
            FisherSignal
        """
        # 1. 변동성 공포 분석
        vol_result = self._vol.analyze(df)

        # 2. 불신의 비관론 감지
        wow_result = self._wow.detect(df)

        # 3. 약세장 회복 감지
        bear_result = self._bear.detect(df)

        # 4. V자 회복 대칭성 분석
        vshape_result = self._vshape.analyze(df)

        # 5. 강세장 초기 구간 감지
        early_bull_result = self._early_bull.detect(df)

        # 6. 극단 비관론 축적 감지
        distrust_result = self._distrust.detect(
            df,
            wow_score=wow_result["wow_score"],
            vol_fear_score=vol_result["fear_score"],
        )

        # 서브모듈 점수 추출
        vol_fear = vol_result["fear_score"]          # 0~1
        wow_score = wow_result["wow_score"]          # 0~1
        recovery = bear_result["recovery_score"]     # 0~1
        v_shape = vshape_result["v_shape_score"]     # 0~1
        early_bull_val = (
            1.0 if early_bull_result["phase"] == "EARLY" else
            0.6 if early_bull_result["phase"] == "MID" else
            0.0
        )
        distrust_val = (
            distrust_result["distrust_score"]
            if distrust_result["extreme_distrust"] else 0.0
        )

        # 켄 피셔 종합 신호 계산 (6가지 서브모듈 가중 합산)
        # 가중치 합계 = 1.0
        opportunity_score = (
            vol_fear * 0.25 +    # 변동성 공포
            wow_score * 0.30 +   # 불신 비관론 (가장 중요)
            recovery * 0.20 +    # 약세장 회복
            v_shape * 0.10 +     # V자 대칭성
            early_bull_val * 0.10 +  # 강세장 초기
            distrust_val * 0.05  # 극단 비관론 축적
        )

        # -1~+1 정규화: 0.3 기준점 (낮은 값 = 위험/버블)
        fisher_composite = (opportunity_score - 0.3) * 2.0
        fisher_composite = float(max(min(fisher_composite, 1.0), -1.0))

        # 포지션 배수 계산 (0.7~1.5)
        base_multiplier = 1.0
        if fisher_composite > 0.5:
            base_multiplier = 1.3  # 강한 매수 기회
        elif fisher_composite > 0.2:
            base_multiplier = 1.1  # 기회
        elif fisher_composite < -0.5:
            base_multiplier = 0.7  # 위험 (과도 안정 → 버블 전조)
        else:
            base_multiplier = 1.0  # 중립

        # 극단 비관론 축적 시 추가 부스트
        position_multiplier = base_multiplier
        if distrust_result["extreme_distrust"]:
            position_multiplier = min(
                base_multiplier + distrust_result["position_boost"],
                1.5
            )

        # 신뢰도 조정 합산
        confidence_delta = (
            wow_result["confidence_boost"] +
            bear_result["confidence_boost"] +
            vshape_result["confidence_boost"] +
            early_bull_result["confidence_boost"] +
            distrust_result["confidence_boost"]
        )
        if vol_result["is_opportunity"]:
            confidence_delta += 0.1

        # 노트 구성
        notes = []
        if vol_result["volatility_regime"] != "NORMAL":
            notes.append(f"[변동성]{vol_result['note'][:50]}")
        if wow_result["wow_score"] > 0:
            notes.append(f"[불신비관론]{wow_result['note'][:50]}")
        if bear_result["recovery_score"] > 0:
            notes.append(f"[회복]{bear_result['note'][:50]}")
        if vshape_result["v_shape_score"] > 0.3:
            notes.append(f"[V자]{vshape_result['note'][:50]}")
        if early_bull_result["is_early_bull"]:
            notes.append(f"[강세장초기]{early_bull_result['note'][:50]}")
        if distrust_result["extreme_distrust"]:
            notes.append(f"[극단비관론]{distrust_result['note'][:50]}")

        # 역사적 기대 수익률 노트 추가
        exp_12m = early_bull_result["expected_12m_return"]
        if exp_12m > 0:
            notes.append(f"[기대수익률]12개월+{exp_12m:.1%}(역사평균)")

        note = " | ".join(notes) if notes else "켄피셔: 특이 패턴 없음"

        return FisherSignal(
            wall_of_worry_score=wow_score,
            volatility_fear_score=vol_fear,
            recovery_signal=recovery,
            fisher_composite=fisher_composite,
            position_multiplier=position_multiplier,
            confidence_delta=confidence_delta,
            note=note,
            v_shape_score=v_shape,
            early_bull_phase=early_bull_result["is_early_bull"],
            extreme_distrust=distrust_result["extreme_distrust"],
            expected_12m_return=exp_12m,
        )
