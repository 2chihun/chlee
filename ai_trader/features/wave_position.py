"""캔들마스터 파동 위치 분석기

「캔들마스터의 주식 캔들 매매법」 핵심 개념을 구현합니다.

파동 위치 분석의 핵심:
1. 큰 하락 구간 감지 (고점 대비 50% 이상 하락)
2. 수평적 파동 구간 감지 (장기 횡보)
3. 후반부 판단 (파동 구간의 시간적/가격적 수렴)
4. 6가지 무효 조건 체크
5. 깔짝 파동 감지 (후반부 소파동)
6. 매수 구간 점수 산출
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 파동 유형 상수
# ---------------------------------------------------------------------------
WAVE_STANDARD = "STANDARD"          # 표준형: 큰 하락 → 수평 파동 → 후반부
WAVE_APPLICATION_1 = "APPLICATION_1"  # 응용1: 긴 수평 파동 (2년 이상)
WAVE_APPLICATION_2 = "APPLICATION_2"  # 응용2: 짧은 횡보 후 급등
WAVE_APPLICATION_3 = "APPLICATION_3"  # 응용3: 이중 바닥 후 수렴
WAVE_APPLICATION_4 = "APPLICATION_4"  # 응용4: 완만한 하락 후 장기 횡보
WAVE_NONE = "NONE"                    # 파동 구간 미감지


@dataclass
class WaveSignal:
    """파동 분석 결과"""
    has_big_decline: bool       # 큰 하락 구간 존재 여부
    decline_pct: float          # 고점 대비 하락률 (음수, 예: -0.55)
    is_horizontal: bool         # 수평적 파동 구간 여부
    is_latter_half: bool        # 후반부 여부
    wave_type: str              # STANDARD/APPLICATION_1~4/NONE
    invalidation: Optional[str]  # 6가지 무효 사유 (None이면 유효)
    buy_zone_score: float       # 매수 구간 점수 (0~100)
    note: str                   # 분석 메모


class WavePositionAnalyzer:
    """
    캔들마스터 파동 위치 분석기

    큰 하락 → 수평적 파동 → 후반부 수렴 → 깔짝 파동 출현 시
    매수 구간으로 판단합니다.

    6가지 무효 조건을 통해 잘못된 매수 구간을 필터링합니다.
    """

    def __init__(
        self,
        lookback_weeks: int = 252,
        decline_threshold: float = 0.50,
        horizontal_width_pct: float = 0.30,
        min_candle_count: int = 50,
        max_price_multiple: float = 10.0,
    ):
        """
        Args:
            lookback_weeks: 분석 대상 기간 (주간 봉 기준, 기본 252주 ≈ 약 5년)
            decline_threshold: 큰 하락 기준 (0.50 = 50%)
            horizontal_width_pct: 수평 파동 폭 기준 (전체 범위 대비 비율)
            min_candle_count: 최소 캔들 수 (이하면 무효)
            max_price_multiple: 최저점 대비 최대 가격 배수 (이상이면 무효)
        """
        self.lookback_weeks = lookback_weeks
        self.decline_threshold = decline_threshold
        self.horizontal_width_pct = horizontal_width_pct
        self.min_candle_count = min_candle_count
        self.max_price_multiple = max_price_multiple

    def detect_big_decline(self, df: pd.DataFrame) -> dict:
        """큰 하락 구간 감지

        고점 대비 decline_threshold(기본 50%) 이상 하락한 구간을 찾습니다.

        Args:
            df: OHLCV DataFrame (주간 데이터 권장)

        Returns:
            dict: {detected, peak_price, trough_price, decline_pct,
                   peak_idx, trough_idx}
        """
        try:
            highs = df["high"].values.astype(float)
            lows = df["low"].values.astype(float)
            closes = df["close"].values.astype(float)

            if len(highs) < 10:
                return {
                    "detected": False, "peak_price": 0, "trough_price": 0,
                    "decline_pct": 0.0, "peak_idx": 0, "trough_idx": 0,
                }

            # 전체 구간에서 최고점을 먼저 찾음
            peak_idx = int(np.argmax(highs))
            peak_price = float(highs[peak_idx])

            # 최고점 이후 최저점 찾기
            if peak_idx >= len(lows) - 1:
                # 최고점이 마지막이면 전체에서 찾기
                trough_idx = int(np.argmin(lows))
            else:
                after_peak_lows = lows[peak_idx:]
                trough_rel = int(np.argmin(after_peak_lows))
                trough_idx = peak_idx + trough_rel

            trough_price = float(lows[trough_idx])

            if peak_price <= 0:
                return {
                    "detected": False, "peak_price": 0, "trough_price": 0,
                    "decline_pct": 0.0, "peak_idx": 0, "trough_idx": 0,
                }

            decline_pct = (trough_price - peak_price) / peak_price
            detected = abs(decline_pct) >= self.decline_threshold

            return {
                "detected": detected,
                "peak_price": peak_price,
                "trough_price": trough_price,
                "decline_pct": round(decline_pct, 4),
                "peak_idx": peak_idx,
                "trough_idx": trough_idx,
            }
        except Exception:
            return {
                "detected": False, "peak_price": 0, "trough_price": 0,
                "decline_pct": 0.0, "peak_idx": 0, "trough_idx": 0,
            }

    def detect_horizontal_zone(self, df: pd.DataFrame, trough_idx: int) -> dict:
        """수평적 파동 구간 감지

        큰 하락 후 저점(trough_idx) 이후로 수평적 횡보 구간을 찾습니다.
        파동의 고저 폭이 전체 범위 대비 horizontal_width_pct 이내면 수평적 파동.

        Args:
            df: OHLCV DataFrame
            trough_idx: 큰 하락의 저점 인덱스

        Returns:
            dict: {detected, zone_start, zone_end, zone_width_pct, candle_count}
        """
        try:
            if trough_idx >= len(df) - 5:
                return {
                    "detected": False, "zone_start": 0, "zone_end": 0,
                    "zone_width_pct": 0.0, "candle_count": 0,
                }

            # 저점 이후 데이터
            zone_df = df.iloc[trough_idx:]
            highs = zone_df["high"].values.astype(float)
            lows = zone_df["low"].values.astype(float)

            zone_high = float(np.max(highs))
            zone_low = float(np.min(lows))

            # 전체 범위 (큰 하락 포함)
            total_high = float(df["high"].max())
            total_low = float(df["low"].min())
            total_range = total_high - total_low

            if total_range <= 0:
                return {
                    "detected": False, "zone_start": 0, "zone_end": 0,
                    "zone_width_pct": 0.0, "candle_count": 0,
                }

            zone_width = zone_high - zone_low
            zone_width_pct = zone_width / total_range

            # 수평적 파동: 전체 범위의 horizontal_width_pct 이내
            detected = (
                zone_width_pct <= self.horizontal_width_pct
                and len(zone_df) >= 10  # 최소 10봉 이상 횡보
            )

            return {
                "detected": detected,
                "zone_start": trough_idx,
                "zone_end": len(df) - 1,
                "zone_width_pct": round(zone_width_pct, 4),
                "candle_count": len(zone_df),
            }
        except Exception:
            return {
                "detected": False, "zone_start": 0, "zone_end": 0,
                "zone_width_pct": 0.0, "candle_count": 0,
            }

    def determine_latter_half(self, df: pd.DataFrame, zone_start: int) -> dict:
        """후반부 판단

        수평적 파동 구간의 시간적 후반부인지 판단합니다.
        또한 파동 높낮이가 줄어드는 수렴 형태인지도 확인합니다.

        Args:
            df: OHLCV DataFrame
            zone_start: 수평적 파동 구간 시작 인덱스

        Returns:
            dict: {is_latter_half, half_point, convergence_ratio}
        """
        try:
            zone_df = df.iloc[zone_start:]
            n = len(zone_df)

            if n < 10:
                return {
                    "is_latter_half": False,
                    "half_point": zone_start,
                    "convergence_ratio": 1.0,
                }

            half_point = zone_start + n // 2

            # 전반부와 후반부의 고저 범위 비교
            first_half = zone_df.iloc[:n // 2]
            second_half = zone_df.iloc[n // 2:]

            first_range = float(first_half["high"].max() - first_half["low"].min())
            second_range = float(second_half["high"].max() - second_half["low"].min())

            # 수렴 비율: 후반부 범위 / 전반부 범위 (1보다 작으면 수렴)
            convergence_ratio = (
                second_range / first_range if first_range > 0 else 1.0
            )

            # 현재 위치가 후반부인지 판단
            current_idx = len(df) - 1
            is_latter_half = current_idx >= half_point

            return {
                "is_latter_half": is_latter_half,
                "half_point": half_point,
                "convergence_ratio": round(convergence_ratio, 4),
            }
        except Exception:
            return {
                "is_latter_half": False,
                "half_point": zone_start,
                "convergence_ratio": 1.0,
            }

    def check_invalidation(self, df: pd.DataFrame, wave_info: dict) -> Optional[str]:
        """6가지 무효 조건 체크

        캔들마스터가 제시하는 매수 구간 무효 조건:
        1. 최저점에서 최고점 대비 50% 넘은 큰 파동 (수평적이지 않음)
        2. 수평적 움직임이 아닌 파동 구간 (상승/하락 추세 진행 중)
        3. 최저점에서 가격이 10배 이상 상승 (이미 많이 오른 종목)
        4. 상장 기간이 짧아 캔들 개수 적음 (데이터 부족)
        5. 완만한 파동 구간 (뚜렷한 큰 하락 없음)
        6. 저점으로부터 멀거나 여러 차례 돌파한 구간

        Args:
            df: OHLCV DataFrame
            wave_info: detect_big_decline 결과

        Returns:
            None이면 유효, 문자열이면 무효 사유
        """
        try:
            trough_idx = wave_info.get("trough_idx", 0)
            trough_price = wave_info.get("trough_price", 0)
            peak_price = wave_info.get("peak_price", 0)

            # 조건 4: 캔들 수 부족
            if len(df) < self.min_candle_count:
                return f"캔들 수 부족: {len(df)}개 < {self.min_candle_count}개 (상장 기간 짧음)"

            # 조건 5: 완만한 파동 (큰 하락이 감지되지 않음)
            if not wave_info.get("detected", False):
                decline_pct = abs(wave_info.get("decline_pct", 0))
                return f"뚜렷한 큰 하락 없음: 하락률 {decline_pct:.1%} < {self.decline_threshold:.1%}"

            # 저점 이후 데이터
            if trough_idx < len(df) - 1:
                after_trough = df.iloc[trough_idx:]
                after_high = float(after_trough["high"].max())
                after_low = float(after_trough["low"].min())

                # 조건 1: 저점 이후 파동이 고점의 50% 넘는 큰 파동
                if peak_price > 0:
                    recovery_pct = (after_high - trough_price) / peak_price
                    if recovery_pct > 0.50:
                        return (
                            f"저점 이후 큰 파동: 고점 대비 {recovery_pct:.1%} 회복 "
                            f"(수평적 파동 아님)"
                        )

                # 조건 2: 수평적이지 않은 파동 (강한 상승/하락 추세)
                if len(after_trough) >= 10:
                    closes = after_trough["close"].values.astype(float)
                    # 선형 회귀로 추세 판단
                    x = np.arange(len(closes))
                    if len(x) > 1:
                        slope = np.polyfit(x, closes, 1)[0]
                        avg_price = float(np.mean(closes))
                        if avg_price > 0:
                            slope_pct = slope / avg_price
                            # 기간당 0.5% 이상 기울기면 추세 진행 중
                            if abs(slope_pct) > 0.005:
                                direction = "상승" if slope_pct > 0 else "하락"
                                return (
                                    f"수평적이지 않은 파동: {direction} 추세 진행 중 "
                                    f"(기울기 {slope_pct:.4f})"
                                )

                # 조건 3: 저점 대비 10배 이상 상승
                if trough_price > 0:
                    current_price = float(df["close"].iloc[-1])
                    price_multiple = current_price / trough_price
                    if price_multiple >= self.max_price_multiple:
                        return (
                            f"저점 대비 {price_multiple:.1f}배 상승 "
                            f"(최대 {self.max_price_multiple}배 초과)"
                        )

                # 조건 6: 저점 대비 현재가가 너무 멀거나 여러 차례 돌파
                if trough_price > 0 and len(after_trough) >= 5:
                    current_price = float(df["close"].iloc[-1])
                    zone_range = after_high - after_low
                    if zone_range > 0:
                        # 현재가가 수평 구간 상단의 80% 이상이면 저점에서 멈
                        distance_from_low = (current_price - after_low) / zone_range
                        # 수평 구간 상단 돌파 횟수 계산
                        upper_line = after_low + zone_range * 0.8
                        crosses = 0
                        prev_above = False
                        for close_val in closes:
                            currently_above = close_val > upper_line
                            if currently_above and not prev_above:
                                crosses += 1
                            prev_above = currently_above

                        if crosses >= 3:
                            return (
                                f"상단선 {crosses}회 돌파 "
                                f"(여러 차례 돌파 후 실패한 구간)"
                            )

            return None  # 모든 무효 조건 통과 → 유효

        except Exception:
            return None  # 오류 시 유효로 간주 (안전)

    def detect_small_wave(self, df: pd.DataFrame, zone_start: int) -> dict:
        """깔짝 파동 감지

        후반부에서 나타나는 작은 상승-하락 파동을 감지합니다.
        수평 횡보 캔들군을 거느린 깔짝 파동이 매수 시점의 단서입니다.

        Args:
            df: OHLCV DataFrame
            zone_start: 수평적 파동 구간 시작 인덱스

        Returns:
            dict: {detected, wave_count, avg_amplitude, convergence}
        """
        try:
            zone_df = df.iloc[zone_start:]
            n = len(zone_df)

            if n < 10:
                return {
                    "detected": False, "wave_count": 0,
                    "avg_amplitude": 0.0, "convergence": 0.0,
                }

            # 후반부만 분석
            latter_df = zone_df.iloc[n // 2:]
            closes = latter_df["close"].values.astype(float)

            if len(closes) < 5:
                return {
                    "detected": False, "wave_count": 0,
                    "avg_amplitude": 0.0, "convergence": 0.0,
                }

            # 로컬 고점/저점을 이용하여 소파동 감지
            # 3봉 기준 로컬 극값 찾기
            peaks = []
            troughs = []
            for i in range(1, len(closes) - 1):
                if closes[i] > closes[i - 1] and closes[i] > closes[i + 1]:
                    peaks.append((i, closes[i]))
                elif closes[i] < closes[i - 1] and closes[i] < closes[i + 1]:
                    troughs.append((i, closes[i]))

            # 파동 수: 고점-저점 쌍의 수
            wave_count = min(len(peaks), len(troughs))

            if wave_count < 2:
                return {
                    "detected": False, "wave_count": wave_count,
                    "avg_amplitude": 0.0, "convergence": 0.0,
                }

            # 파동 진폭 계산
            amplitudes = []
            avg_price = float(np.mean(closes))
            for i in range(min(len(peaks), len(troughs))):
                amp = abs(peaks[i][1] - troughs[i][1])
                if avg_price > 0:
                    amplitudes.append(amp / avg_price)

            avg_amplitude = float(np.mean(amplitudes)) if amplitudes else 0.0

            # 수렴도: 마지막 파동 진폭 / 첫 파동 진폭
            convergence = 0.0
            if len(amplitudes) >= 2 and amplitudes[0] > 0:
                convergence = amplitudes[-1] / amplitudes[0]

            # 깔짝 파동 판단: 2개 이상 소파동 + 진폭이 작음 + 수렴
            detected = (
                wave_count >= 2
                and avg_amplitude < 0.10  # 평균가 대비 10% 미만 진폭
                and convergence < 1.0     # 수렴 (점점 작아짐)
            )

            return {
                "detected": detected,
                "wave_count": wave_count,
                "avg_amplitude": round(avg_amplitude, 4),
                "convergence": round(convergence, 4),
            }
        except Exception:
            return {
                "detected": False, "wave_count": 0,
                "avg_amplitude": 0.0, "convergence": 0.0,
            }

    def analyze(self, df: pd.DataFrame) -> WaveSignal:
        """통합 파동 분석

        1. 큰 하락 구간 감지
        2. 수평적 파동 구간 감지
        3. 후반부 판단
        4. 무효 조건 체크
        5. 깔짝 파동 감지
        6. 매수 구간 점수 산출

        Args:
            df: OHLCV DataFrame (주간 데이터 권장, 일봉도 가능)

        Returns:
            WaveSignal: 파동 분석 결과
        """
        try:
            # 분석 데이터 범위 제한
            analysis_df = df.tail(self.lookback_weeks).copy()

            if len(analysis_df) < 20:
                return WaveSignal(
                    has_big_decline=False, decline_pct=0.0,
                    is_horizontal=False, is_latter_half=False,
                    wave_type=WAVE_NONE, invalidation="데이터 부족",
                    buy_zone_score=0.0, note="분석 불가: 데이터 20봉 미만",
                )

            # 1단계: 큰 하락 구간 감지
            decline_info = self.detect_big_decline(analysis_df)

            # 2단계: 수평적 파동 구간 감지
            horizontal_info = {"detected": False, "zone_start": 0,
                              "zone_end": 0, "zone_width_pct": 0.0,
                              "candle_count": 0}
            if decline_info["detected"]:
                horizontal_info = self.detect_horizontal_zone(
                    analysis_df, decline_info["trough_idx"]
                )

            # 3단계: 후반부 판단
            latter_info = {"is_latter_half": False, "half_point": 0,
                          "convergence_ratio": 1.0}
            if horizontal_info["detected"]:
                latter_info = self.determine_latter_half(
                    analysis_df, horizontal_info["zone_start"]
                )

            # 4단계: 무효 조건 체크
            invalidation = self.check_invalidation(analysis_df, decline_info)

            # 5단계: 깔짝 파동 감지
            small_wave_info = {"detected": False, "wave_count": 0,
                              "avg_amplitude": 0.0, "convergence": 0.0}
            if horizontal_info["detected"]:
                small_wave_info = self.detect_small_wave(
                    analysis_df, horizontal_info["zone_start"]
                )

            # 6단계: 파동 유형 결정
            wave_type = self._determine_wave_type(
                decline_info, horizontal_info, latter_info, small_wave_info
            )

            # 7단계: 매수 구간 점수 산출
            buy_zone_score = self._calculate_buy_zone_score(
                decline_info, horizontal_info, latter_info,
                small_wave_info, invalidation
            )

            # 분석 메모 작성
            notes = []
            if decline_info["detected"]:
                notes.append(
                    f"큰하락({decline_info['decline_pct']:.1%})"
                )
            if horizontal_info["detected"]:
                notes.append(
                    f"수평파동(폭{horizontal_info['zone_width_pct']:.1%},"
                    f"{horizontal_info['candle_count']}봉)"
                )
            if latter_info["is_latter_half"]:
                notes.append(
                    f"후반부(수렴{latter_info['convergence_ratio']:.2f})"
                )
            if small_wave_info["detected"]:
                notes.append(
                    f"깔짝파동({small_wave_info['wave_count']}회,"
                    f"진폭{small_wave_info['avg_amplitude']:.2%})"
                )
            if invalidation:
                notes.append(f"무효:{invalidation}")

            return WaveSignal(
                has_big_decline=decline_info["detected"],
                decline_pct=decline_info["decline_pct"],
                is_horizontal=horizontal_info["detected"],
                is_latter_half=latter_info["is_latter_half"],
                wave_type=wave_type,
                invalidation=invalidation,
                buy_zone_score=buy_zone_score,
                note=" | ".join(notes) if notes else "파동 구간 미감지",
            )

        except Exception as exc:
            return WaveSignal(
                has_big_decline=False, decline_pct=0.0,
                is_horizontal=False, is_latter_half=False,
                wave_type=WAVE_NONE, invalidation=None,
                buy_zone_score=0.0, note=f"분석 오류: {exc}",
            )

    def _determine_wave_type(
        self,
        decline_info: dict,
        horizontal_info: dict,
        latter_info: dict,
        small_wave_info: dict,
    ) -> str:
        """파동 유형을 결정합니다."""
        if not decline_info["detected"]:
            return WAVE_NONE

        if not horizontal_info["detected"]:
            # 수평 파동은 없지만 큰 하락이 있는 경우
            return WAVE_APPLICATION_4  # 완만한 하락 후 장기 횡보 시작 전

        # 표준형: 큰 하락 + 수평 파동 + 후반부
        if latter_info["is_latter_half"] and small_wave_info["detected"]:
            return WAVE_STANDARD

        # 응용1: 긴 수평 파동 (캔들 수 100개 이상)
        if horizontal_info["candle_count"] >= 100:
            return WAVE_APPLICATION_1

        # 응용2: 짧은 횡보 (캔들 수 30개 미만) + 수렴
        if (horizontal_info["candle_count"] < 30
                and latter_info["convergence_ratio"] < 0.7):
            return WAVE_APPLICATION_2

        # 응용3: 이중 바닥 패턴 (소파동 2회 + 수렴)
        if (small_wave_info["wave_count"] == 2
                and small_wave_info["convergence"] < 0.8):
            return WAVE_APPLICATION_3

        if latter_info["is_latter_half"]:
            return WAVE_STANDARD

        return WAVE_NONE

    def _calculate_buy_zone_score(
        self,
        decline_info: dict,
        horizontal_info: dict,
        latter_info: dict,
        small_wave_info: dict,
        invalidation: Optional[str],
    ) -> float:
        """매수 구간 점수 산출 (0~100)

        각 조건별 가중치:
        - 큰 하락 존재: 20점
        - 수평적 파동: 25점
        - 후반부: 20점
        - 깔짝 파동: 20점
        - 수렴 형태: 15점
        - 무효 조건 해당 시: 0점
        """
        if invalidation is not None:
            return 0.0

        score = 0.0

        # 큰 하락 (20점)
        if decline_info["detected"]:
            # 하락률이 클수록 점수 높음 (50%→20점, 30%→10점)
            decline_abs = abs(decline_info["decline_pct"])
            score += min(20.0, (decline_abs / self.decline_threshold) * 20.0)

        # 수평적 파동 (25점)
        if horizontal_info["detected"]:
            # 폭이 좁을수록 점수 높음
            width_ratio = horizontal_info["zone_width_pct"]
            if width_ratio > 0:
                narrow_score = (1.0 - width_ratio / self.horizontal_width_pct)
                score += max(0, min(25.0, narrow_score * 25.0))
            else:
                score += 25.0

        # 후반부 (20점)
        if latter_info["is_latter_half"]:
            score += 20.0

        # 깔짝 파동 (20점)
        if small_wave_info["detected"]:
            # 파동 수가 많을수록 + 수렴할수록 점수 높음
            wave_score = min(10.0, small_wave_info["wave_count"] * 3.0)
            convergence_score = max(
                0, 10.0 * (1.0 - small_wave_info["convergence"])
            )
            score += wave_score + convergence_score

        # 수렴 형태 (15점)
        if latter_info.get("convergence_ratio", 1.0) < 1.0:
            conv = latter_info["convergence_ratio"]
            score += max(0, 15.0 * (1.0 - conv))

        return round(min(100.0, max(0.0, score)), 1)
