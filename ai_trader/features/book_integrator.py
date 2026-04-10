"""9개 도서 통합 시그널 모듈

모든 도서 기반 분석 모듈의 시그널을 가중 합산하여
최종 투자 판단을 도출하는 통합 분석기.

가중치 기본값:
- 하워드 막스 (사이클): 20%
- 백석현 (환율): 15%
- 조상철 (버블감지): 15%
- 잭 슈웨거 (규율): 12%
- 켄 피셔 (시장기억): 10%
- 서준식 (딥밸류): 8%
- 이남우 (품질): 8%
- 박병창 (집행): 7%
- 강방천&존리 (가치): 5%

상충 해소:
- 버블 BURST/PANIC → 다른 모든 매수 시그널 무시
- 사이클 LATE + 환율 경보 → 포지션 상한 30%
- 가치투자 매수 + 사이클 EARLY → 최적 매수 기회
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import pandas as pd
from loguru import logger


@dataclass
class IntegratedSignal:
    """10+1개 도서 통합 시그널 (ML 메타레이블링 포함)"""
    # 총합 점수 (0-100, 높을수록 매수 유리)
    composite_score: float

    # 행동 권고
    action: str  # "STRONG_BUY" / "BUY" / "HOLD" / "SELL" / "STRONG_SELL" / "BLOCK"

    # 포지션 배수 (0.0-2.0)
    position_multiplier: float

    # 레버리지 상한
    max_leverage: float

    # 각 모듈 개별 점수
    module_scores: Dict[str, float] = field(default_factory=dict)

    # 충돌 여부 및 해소 내역
    conflicts: list = field(default_factory=list)

    # 종합 이유
    reason: str = ""

    # ── ML Meta-Labeling (López de Prado) ──
    ml_meta_prob: float = 0.0      # 2차 모델 실행 확률
    ml_execute: bool = True        # ML 실행 허가 여부
    ml_bet_size: float = 1.0       # ML 베팅 크기 (0~1)
    ml_liquidity_risk: float = 0.0 # 유동성 위험 점수 (0~1)
    ml_entropy: float = 0.0        # 시장 엔트로피 (예측 난이도)


# 기본 가중치
DEFAULT_WEIGHTS = {
    "market_cycle": 0.20,      # 하워드 막스
    "exchange_rate": 0.15,     # 백석현
    "bubble_detector": 0.15,   # 조상철
    "wizard_discipline": 0.12, # 잭 슈웨거
    "market_memory": 0.10,     # 켄 피셔
    "deep_value": 0.08,        # 서준식
    "stock_quality": 0.08,     # 이남우
    "execution_analysis": 0.07, # 박병창
    "value_investor": 0.05,    # 강방천&존리
}


class ConflictResolver:
    """상충 시그널 해소

    우선순위 규칙:
    1. 안전 우선: 버블/패닉 경고는 항상 최우선
    2. 거시 > 미시: 사이클/환율이 개별 종목 시그널보다 우선
    3. 다수결: 동일 수준 시그널은 다수결
    """

    def resolve(self, module_signals: Dict[str, Any]) -> list:
        """상충 감지 및 해소

        Returns:
            충돌 해소 내역 리스트
        """
        conflicts = []

        # 1. 버블 감지 vs 가치투자 매수
        bubble = module_signals.get("bubble_detector")
        value = module_signals.get("value_investor")
        deep_value = module_signals.get("deep_value")

        if bubble and hasattr(bubble, 'phase'):
            phase_name = bubble.phase.value if hasattr(bubble.phase, 'value') else str(bubble.phase)
            if phase_name in ("BURST", "PANIC"):
                if value and hasattr(value, 'position_multiplier'):
                    if value.position_multiplier > 1.0:
                        conflicts.append(
                            f"[충돌해소] 버블 {phase_name} → "
                            f"가치투자 매수 시그널 무시 (안전 우선)"
                        )
                if deep_value and hasattr(deep_value, 'position_multiplier'):
                    if deep_value.position_multiplier > 1.0:
                        conflicts.append(
                            f"[충돌해소] 버블 {phase_name} → "
                            f"딥밸류 매수 시그널 무시 (안전 우선)"
                        )

        # 2. 사이클 LATE + 환율 경보
        cycle = module_signals.get("market_cycle")
        fx = module_signals.get("exchange_rate")

        if (cycle and hasattr(cycle, 'phase') and
                fx and hasattr(fx, 'fx_alarm')):
            cycle_phase = cycle.phase if isinstance(cycle.phase, str) else getattr(cycle.phase, 'value', str(cycle.phase))
            if "LATE" in str(cycle_phase) and fx.fx_alarm:
                conflicts.append(
                    "[충돌해소] LATE 사이클 + 환율 급변 → "
                    "포지션 상한 30%로 제한"
                )

        # 3. 가치투자 매수 + 사이클 EARLY
        if cycle and hasattr(cycle, 'phase'):
            cycle_phase = cycle.phase if isinstance(cycle.phase, str) else getattr(cycle.phase, 'value', str(cycle.phase))
            if "EARLY" in str(cycle_phase):
                if value and hasattr(value, 'position_multiplier'):
                    if value.position_multiplier > 1.0:
                        conflicts.append(
                            "[시너지] EARLY 사이클 + 가치투자 매수 → "
                            "최적 매수 기회 (포지션 확대 허용)"
                        )

        return conflicts


class SignalWeightManager:
    """시장 상황별 동적 가중치 조절

    시장 상태에 따라 가중치를 동적 조정:
    - 고변동성: 사이클/버블 가중치 ↑, 개별종목 ↓
    - 저변동성: 개별종목(가치/품질) ↑, 거시 ↓
    - 위기 상황: 환율/버블 ↑↑, 나머지 ↓
    """

    def __init__(self, base_weights: Dict[str, float] = None):
        self.base_weights = base_weights or DEFAULT_WEIGHTS.copy()

    def get_adjusted_weights(self, df: pd.DataFrame = None,
                              module_signals: Dict[str, Any] = None
                              ) -> Dict[str, float]:
        """동적 가중치 반환"""
        weights = self.base_weights.copy()

        if df is None or len(df) < 60:
            return weights

        import numpy as np
        close = df["close"].values
        returns = np.diff(np.log(close[-60:]))
        vol = np.std(returns) * np.sqrt(252)

        # 고변동성 (연환산 30%+): 거시 가중치 강화
        if vol > 0.30:
            weights["market_cycle"] *= 1.3
            weights["bubble_detector"] *= 1.3
            weights["exchange_rate"] *= 1.2
            weights["deep_value"] *= 0.7
            weights["stock_quality"] *= 0.7
            weights["value_investor"] *= 0.7

        # 저변동성 (연환산 10% 미만): 미시 가중치 강화
        elif vol < 0.10:
            weights["deep_value"] *= 1.3
            weights["stock_quality"] *= 1.3
            weights["value_investor"] *= 1.3
            weights["market_cycle"] *= 0.8
            weights["bubble_detector"] *= 0.7

        # 버블/위기 시: 안전 가중치 극대화
        if module_signals:
            bubble = module_signals.get("bubble_detector")
            if bubble and hasattr(bubble, 'phase'):
                phase_name = bubble.phase.value if hasattr(bubble.phase, 'value') else str(bubble.phase)
                if phase_name in ("BURST", "PANIC"):
                    weights["bubble_detector"] *= 2.0
                    weights["exchange_rate"] *= 1.5
                    weights["market_cycle"] *= 1.5
                    # 매수 성향 모듈 약화
                    for key in ("deep_value", "stock_quality",
                                "value_investor", "execution_analysis"):
                        weights[key] *= 0.3

        # 정규화 (합=1.0)
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights


class BookIntegrator:
    """9개 도서 통합 분석기

    모든 도서 기반 모듈의 시그널을 수집하고,
    가중 합산 + 상충 해소 + 동적 가중치 조절을 거쳐
    최종 투자 판단을 생성합니다.
    """

    def __init__(self, config=None):
        self.config = config
        self.weight_manager = SignalWeightManager()
        self.conflict_resolver = ConflictResolver()

    def integrate(self, df: pd.DataFrame,
                  module_signals: Dict[str, Any] = None
                  ) -> IntegratedSignal:
        """통합 분석 실행

        Args:
            df: OHLCV DataFrame
            module_signals: 각 모듈별 시그널 딕셔너리
                예: {"market_cycle": CycleSignal, "bubble_detector": BubbleSignal, ...}

        Returns:
            IntegratedSignal: 통합 시그널
        """
        if module_signals is None:
            module_signals = {}

        # 1. 동적 가중치 계산
        weights = self.weight_manager.get_adjusted_weights(
            df, module_signals
        )

        # 2. 각 모듈별 점수 추출 (0-100, 50=중립)
        module_scores = self._extract_scores(module_signals)

        # 3. 가중 합산
        composite = 0.0
        for key, score in module_scores.items():
            weight = weights.get(key, 0.0)
            composite += score * weight

        # 4. 상충 해소
        conflicts = self.conflict_resolver.resolve(module_signals)

        # 5. 포지션 배수 결정
        position_multiplier = self._calc_position_multiplier(
            composite, module_signals, conflicts
        )

        # 6. 레버리지 상한
        max_leverage = 1.5  # 백석현 절대 상한

        # 7. 행동 권고
        action = self._determine_action(
            composite, module_signals, conflicts
        )

        # 8. 종합 이유
        reason = self._generate_reason(
            composite, action, module_scores, conflicts
        )

        signal = IntegratedSignal(
            composite_score=round(composite, 1),
            action=action,
            position_multiplier=round(position_multiplier, 2),
            max_leverage=max_leverage,
            module_scores=module_scores,
            conflicts=conflicts,
            reason=reason,
        )

        # ── ML Meta-Labeling 보정 (López de Prado Ch3) ──
        signal = self._apply_ml_overlay(signal, df, module_signals)

        logger.info(
            f"BookIntegrator: score={composite:.1f}, "
            f"action={action}, multiplier={position_multiplier:.2f}, "
            f"ml_prob={signal.ml_meta_prob:.2f}, ml_size={signal.ml_bet_size:.2f}"
        )
        return signal

    def _extract_scores(self, signals: Dict[str, Any]) -> Dict[str, float]:
        """각 모듈의 시그널에서 0-100 점수 추출

        점수 기준: 50 = 중립, 100 = 강한 매수, 0 = 강한 매도
        """
        scores = {}

        for key, sig in signals.items():
            if sig is None:
                scores[key] = 50.0
                continue

            # 버블 감지: bubble_score 역전 (높을수록 위험 → 낮은 점수)
            if key == "bubble_detector" and hasattr(sig, 'bubble_score'):
                scores[key] = max(0.0, 100.0 - sig.bubble_score)

            # 환율: risk_level 역전
            elif key == "exchange_rate" and hasattr(sig, 'risk_level'):
                scores[key] = max(0.0, 100.0 - sig.risk_level)

            # 사이클: position_multiplier 기반
            elif key == "market_cycle" and hasattr(sig, 'cycle_score'):
                # cycle_score 높으면 LATE → 낮은 점수
                scores[key] = max(0.0, 100.0 - sig.cycle_score)

            # position_multiplier가 있는 모듈: 배수 기반
            elif hasattr(sig, 'position_multiplier'):
                mp = sig.position_multiplier
                scores[key] = min(100.0, max(0.0, mp * 50))

            # confidence_adjustment가 있는 모듈
            elif hasattr(sig, 'confidence_adjustment'):
                scores[key] = 50.0 + sig.confidence_adjustment * 100

            # 총점이 있는 모듈 (CANSLIM 등)
            elif hasattr(sig, 'total'):
                scores[key] = sig.total

            else:
                scores[key] = 50.0

        return scores

    def _calc_position_multiplier(self, composite: float,
                                   signals: Dict[str, Any],
                                   conflicts: list) -> float:
        """포지션 배수 결정

        기본: composite 점수 기반
        조정: 버블/사이클/환율 상태에 따른 상한 제한
        """
        # 기본 배수: 점수 40-60=1.0, 60+=1.2, <40=0.7
        if composite >= 70:
            base = 1.3
        elif composite >= 60:
            base = 1.1
        elif composite >= 40:
            base = 1.0
        elif composite >= 30:
            base = 0.7
        else:
            base = 0.5

        # 버블 위험 시 상한 제한
        bubble = signals.get("bubble_detector")
        if bubble and hasattr(bubble, 'position_multiplier'):
            base = min(base, bubble.position_multiplier)

        # 환율 위험 시 추가 제한
        fx = signals.get("exchange_rate")
        if fx and hasattr(fx, 'position_multiplier'):
            base = min(base, fx.position_multiplier * 1.1)

        # LATE + 환율 경보 충돌 시 30% 상한
        for conflict in conflicts:
            if "포지션 상한 30%" in conflict:
                base = min(base, 0.3)

        return max(0.1, min(2.0, base))

    def _determine_action(self, composite: float,
                           signals: Dict[str, Any],
                           conflicts: list) -> str:
        """행동 권고 결정"""
        # 버블 BURST/PANIC → 무조건 BLOCK
        bubble = signals.get("bubble_detector")
        if bubble and hasattr(bubble, 'phase'):
            phase_name = bubble.phase.value if hasattr(bubble.phase, 'value') else str(bubble.phase)
            if phase_name == "PANIC":
                return "BLOCK"
            elif phase_name == "BURST":
                return "STRONG_SELL"

        # 점수 기반
        if composite >= 75:
            return "STRONG_BUY"
        elif composite >= 60:
            return "BUY"
        elif composite >= 40:
            return "HOLD"
        elif composite >= 25:
            return "SELL"
        else:
            return "STRONG_SELL"

    def _generate_reason(self, composite: float, action: str,
                          scores: Dict[str, float],
                          conflicts: list) -> str:
        """종합 이유 생성"""
        parts = [f"통합점수={composite:.0f}, 권고={action}"]

        # 상위 3개 영향 모듈
        sorted_scores = sorted(scores.items(),
                               key=lambda x: abs(x[1] - 50),
                               reverse=True)[:3]
        for key, score in sorted_scores:
            direction = "긍정" if score > 50 else "부정"
            parts.append(f"{key}={score:.0f}({direction})")

        if conflicts:
            parts.append(f"충돌해소 {len(conflicts)}건")

        return " | ".join(parts)

    def _apply_ml_overlay(
        self,
        signal: IntegratedSignal,
        df: pd.DataFrame,
        module_signals: Dict[str, Any],
    ) -> IntegratedSignal:
        """ML Meta-Labeling으로 시그널을 보정합니다.

        기존 규칙 기반 시그널(1차 모델)을 유지하면서,
        ML 2차 모델이 실행 여부와 포지션 크기를 조절합니다.

        보정 로직:
          1) ML meta_prob < 0.4 → 실행 차단 (ml_execute=False)
          2) ML bet_size로 position_multiplier 보정
          3) 유동성 위험 높으면 크기 축소
          4) 엔트로피 높으면 크기 축소
        """
        try:
            # 엔트로피 계산 (예측 난이도)
            if "close" in df.columns and len(df) >= 50:
                from features.entropy import shannon_entropy
                returns = df["close"].pct_change().dropna().tail(50)
                signal.ml_entropy = shannon_entropy(returns, n_bins=10)

            # 유동성 위험 계산
            if "close" in df.columns and "volume" in df.columns and len(df) >= 50:
                from features.microstructure import compute_vpin
                vpin = compute_vpin(df["close"], df["volume"], window=30)
                last_vpin = vpin.dropna().iloc[-1] if vpin.notna().any() else 0
                signal.ml_liquidity_risk = float(last_vpin)

            # ML 베팅 크기 보정
            # composite_score를 확률로 변환 (0-100 → 0-1)
            prob = signal.composite_score / 100.0
            signal.ml_meta_prob = prob

            from risk.bet_sizing import bet_size_from_prob
            size_series = bet_size_from_prob(
                pd.Series([prob]), method="sigmoid"
            )
            ml_size = abs(float(size_series.iloc[0]))

            # 유동성 위험 패널티 (VPIN > 0.7이면 크기 50% 감소)
            if signal.ml_liquidity_risk > 0.7:
                ml_size *= 0.5

            # 엔트로피 패널티 (매우 불확실할 때 크기 감소)
            if signal.ml_entropy > 4.0:
                ml_size *= 0.7

            signal.ml_bet_size = round(ml_size, 3)

            # 실행 허가: 확률이 너무 낮으면 차단
            if prob < 0.4 or prob > 0.6:
                signal.ml_execute = True
            else:
                # 0.4~0.6 구간: 확신 부족, 실행 보류
                signal.ml_execute = False
                signal.ml_bet_size = 0.0

            # 최종 position_multiplier 보정
            if signal.ml_execute and signal.ml_bet_size > 0:
                signal.position_multiplier = round(
                    signal.position_multiplier * signal.ml_bet_size, 2
                )

        except Exception as e:
            logger.debug(f"ML overlay 적용 실패 (규칙 기반 유지): {e}")

        return signal
