"""투자 주체별 수급 분석 모듈

박병창 저 「현명한 당신의 주식투자 교과서」 핵심 개념 구현:
- 외국인 매수/매도 → 대세 방향 결정
- 기관(투신/사모펀드) → 개별 종목 급등 주도
- 매수 주체가 이기는 쪽이 시장 방향 (황소 vs 곰)
- 주도 업종/주도 주식 추종 전략
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ── 수급 신호 데이터 ───────────────────────────────────────

@dataclass
class MarketFlowSignal:
    """수급 분석 신호"""
    trend: str                              # "bullish", "bearish", "neutral"
    dominant_player: str                    # "foreign", "institution", "retail"
    foreign_flow: float                     # 외국인 순매수 (양수=매수, 음수=매도)
    institution_flow: float                 # 기관 순매수
    retail_flow: float                      # 개인 순매수
    flow_strength: float                    # 수급 강도 (0~1)
    leading_sector: Optional[str] = None    # 주도 업종
    confidence: float = 0.0                 # 신뢰도 (0~1)


@dataclass
class LeadingStock:
    """주도주 정보"""
    stock_code: str
    stock_name: str
    foreign_net: float          # 외국인 순매수량
    institution_net: float      # 기관 순매수량
    change_pct: float           # 등락률 (%)
    volume_ratio: float         # 거래량 비율 (평균 대비)
    score: float                # 종합 점수


# ── KIS API TR-ID ──────────────────────────────────────────

TR_INVESTOR_TREND = "FHKST01010900"     # 종목별 투자자 매매동향
TR_SECTOR_INDEX = "FHKUP02110200"       # 업종별 지수
TR_EXECUTION = "FHKST01010300"          # 체결 정보


# ── 수급 분석기 ────────────────────────────────────────────

class MarketFlowAnalyzer:
    """투자 주체별 수급 분석기

    외국인/기관/개인의 매매 동향을 분석하여 시장 방향과
    주도주를 탐지합니다. KIS API collector가 없으면
    오프라인 모드로 DataFrame 기반 분석만 수행합니다.
    """

    # 기관 세부 분류
    INST_SHORT_TERM = {"투신", "사모"}          # 단기 트레이딩
    INST_LONG_TERM = {"연기금", "보험", "은행"}  # 장기 투자

    def __init__(self, collector=None):
        """
        Args:
            collector: KISDataCollector 인스턴스 (None이면 오프라인 모드)
        """
        self.collector = collector
        self._online = collector is not None

    # ── 종목별 수급 분석 ───────────────────────────────────

    def analyze_market_flow(
        self,
        stock_code: str,
        period: int = 20,
        df: Optional[pd.DataFrame] = None,
    ) -> MarketFlowSignal:
        """종목별 외국인/기관/개인 수급을 분석합니다.

        Args:
            stock_code: 종목코드 (6자리)
            period: 분석 기간 (일)
            df: 오프라인 모드용 수급 DataFrame
                필수 컬럼: date, foreign_net, institution_net, retail_net

        Returns:
            MarketFlowSignal 수급 분석 결과
        """
        if df is None:
            df = self._fetch_investor_trend(stock_code, period)

        if df.empty:
            logger.warning("수급 데이터 없음 [{}] — 기본값 반환", stock_code)
            return self._default_signal()

        # 기간 내 누적 순매수
        foreign_flow = float(df["foreign_net"].sum())
        institution_flow = float(df["institution_net"].sum())
        retail_flow = float(df["retail_net"].sum())

        # 연속 매수/매도 일수 (외국인 기준)
        foreign_streak = self._calc_streak(df["foreign_net"])

        # 지배적 주체 판별
        dominant_player = self._determine_dominant_player(
            foreign_flow, institution_flow, retail_flow,
        )

        # 추세 판단: 외국인이 대세 방향 결정
        trend = self._determine_trend(
            foreign_flow, institution_flow, foreign_streak,
        )

        # 수급 강도 계산 (0~1)
        flow_strength = self._calc_flow_strength(df)

        # 신뢰도: 연속일수 + 수급 일치도
        confidence = self._calc_confidence(
            foreign_flow, institution_flow, foreign_streak, period,
        )

        return MarketFlowSignal(
            trend=trend,
            dominant_player=dominant_player,
            foreign_flow=foreign_flow,
            institution_flow=institution_flow,
            retail_flow=retail_flow,
            flow_strength=flow_strength,
            confidence=confidence,
        )

    # ── 주도 업종 탐지 ─────────────────────────────────────

    def detect_leading_sector(
        self,
        df: Optional[pd.DataFrame] = None,
    ) -> list[dict]:
        """주도 업종을 탐지합니다.

        거래량 상위 업종 + 외국인/기관 순매수 업종을 기준으로
        주도 업종을 선별합니다.

        Args:
            df: 오프라인 모드용 업종별 DataFrame
                필수 컬럼: sector, volume, foreign_net, institution_net, change_pct

        Returns:
            주도 업종 리스트 (점수 내림차순)
            [{"sector": str, "score": float, "change_pct": float, ...}, ...]
        """
        if df is None:
            df = self._fetch_sector_data()

        if df.empty:
            logger.warning("업종 데이터 없음 — 빈 리스트 반환")
            return []

        results = []
        for _, row in df.iterrows():
            # 거래량 상대 비중
            vol_score = _safe_normalize(
                row.get("volume", 0), df["volume"].max(),
            )
            # 외국인/기관 순매수 점수
            flow_score = _safe_normalize(
                row.get("foreign_net", 0) + row.get("institution_net", 0),
                df[["foreign_net", "institution_net"]].sum(axis=1).abs().max(),
            )
            # 등락률 점수
            chg_score = _safe_normalize(
                row.get("change_pct", 0), df["change_pct"].abs().max(),
            )

            composite = vol_score * 0.3 + flow_score * 0.5 + chg_score * 0.2

            results.append({
                "sector": row.get("sector", ""),
                "score": round(composite, 4),
                "change_pct": row.get("change_pct", 0.0),
                "volume": row.get("volume", 0),
                "foreign_net": row.get("foreign_net", 0),
                "institution_net": row.get("institution_net", 0),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # ── 주도주 탐지 ────────────────────────────────────────

    def detect_leading_stocks(
        self,
        sector: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        top_n: int = 10,
    ) -> list[LeadingStock]:
        """주도주를 탐지합니다.

        외국인/기관 순매수 + 상승률 상위 종목을 선별합니다.

        Args:
            sector: 업종 필터 (None이면 전체)
            df: 오프라인 모드용 종목별 DataFrame
                필수 컬럼: stock_code, stock_name, foreign_net,
                          institution_net, change_pct, volume, avg_volume
            top_n: 상위 N개 반환

        Returns:
            LeadingStock 리스트 (점수 내림차순)
        """
        if df is None:
            df = self._fetch_stock_flow_data(sector)

        if df.empty:
            logger.warning("종목 수급 데이터 없음 — 빈 리스트 반환")
            return []

        # 업종 필터
        if sector and "sector" in df.columns:
            df = df[df["sector"] == sector].copy()

        if df.empty:
            return []

        # 거래량 비율 계산
        if "avg_volume" in df.columns:
            df["volume_ratio"] = df["volume"] / df["avg_volume"].replace(0, np.nan)
            df["volume_ratio"] = df["volume_ratio"].fillna(1.0)
        elif "volume_ratio" not in df.columns:
            df["volume_ratio"] = 1.0

        # 점수 계산: 외국인(40%) + 기관(30%) + 등락률(20%) + 거래량비율(10%)
        max_foreign = df["foreign_net"].abs().max() or 1.0
        max_inst = df["institution_net"].abs().max() or 1.0
        max_chg = df["change_pct"].abs().max() or 1.0
        max_vol_r = df["volume_ratio"].max() or 1.0

        results = []
        for _, row in df.iterrows():
            f_score = _safe_normalize(row["foreign_net"], max_foreign)
            i_score = _safe_normalize(row["institution_net"], max_inst)
            c_score = _safe_normalize(row["change_pct"], max_chg)
            v_score = _safe_normalize(row["volume_ratio"], max_vol_r)

            score = f_score * 0.4 + i_score * 0.3 + c_score * 0.2 + v_score * 0.1

            results.append(LeadingStock(
                stock_code=str(row.get("stock_code", "")),
                stock_name=str(row.get("stock_name", "")),
                foreign_net=float(row.get("foreign_net", 0)),
                institution_net=float(row.get("institution_net", 0)),
                change_pct=float(row.get("change_pct", 0)),
                volume_ratio=float(row.get("volume_ratio", 1.0)),
                score=round(score, 4),
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_n]

    # ── 체결강도 ───────────────────────────────────────────

    def get_execution_strength(
        self,
        stock_code: str,
        df: Optional[pd.DataFrame] = None,
    ) -> float:
        """체결강도를 계산합니다.

        체결강도 = (매수체결량 / 매도체결량) × 100
        - > 100: 매수 우세
        - < 100: 매도 우세
        - = 100: 균형

        Args:
            stock_code: 종목코드
            df: 오프라인 모드용 DataFrame
                필수 컬럼: buy_volume, sell_volume

        Returns:
            체결강도 (기본값 100.0)
        """
        if df is None:
            df = self._fetch_execution_data(stock_code)

        if df.empty:
            logger.debug("체결 데이터 없음 [{}] — 기본값 100.0", stock_code)
            return 100.0

        total_buy = df["buy_volume"].sum()
        total_sell = df["sell_volume"].sum()

        if total_sell == 0:
            return 200.0 if total_buy > 0 else 100.0

        return round(float(total_buy / total_sell * 100), 2)

    # ── 거래량 프로파일 분석 ────────────────────────────────

    def analyze_volume_profile(self, df: pd.DataFrame) -> dict:
        """거래량 프로파일을 분석합니다.

        대량거래 감지, 거래량 급증 패턴, 평균 대비 비율 등을
        산출합니다.

        Args:
            df: OHLCV DataFrame
                필수 컬럼: close, volume (선택: high, low)

        Returns:
            {
                "avg_volume": 평균 거래량,
                "current_ratio": 직전 거래량 / 평균 거래량,
                "volume_trend": "increasing" | "decreasing" | "stable",
                "spike_dates": 거래량 급증일 리스트,
                "price_volume_corr": 가격-거래량 상관계수,
                "large_trade_days": 대량거래일 수,
            }
        """
        if df.empty or "volume" not in df.columns:
            return self._default_volume_profile()

        vol = df["volume"].astype(float)
        close = df["close"].astype(float)

        avg_vol = vol.mean()
        std_vol = vol.std()

        # 직전 거래량 비율
        current_ratio = float(vol.iloc[-1] / avg_vol) if avg_vol > 0 else 1.0

        # 거래량 추세 (최근 5일 이동평균 vs 전체 평균)
        recent_avg = vol.tail(5).mean() if len(vol) >= 5 else vol.mean()
        if avg_vol > 0:
            trend_ratio = recent_avg / avg_vol
            if trend_ratio > 1.2:
                volume_trend = "increasing"
            elif trend_ratio < 0.8:
                volume_trend = "decreasing"
            else:
                volume_trend = "stable"
        else:
            volume_trend = "stable"

        # 거래량 급증일 (평균 + 2σ 초과)
        spike_threshold = avg_vol + 2 * std_vol if std_vol > 0 else avg_vol * 2
        spike_mask = vol > spike_threshold
        spike_dates = []
        if "date" in df.columns:
            spike_dates = df.loc[spike_mask, "date"].tolist()
        elif "datetime" in df.columns:
            spike_dates = df.loc[spike_mask, "datetime"].tolist()
        else:
            spike_dates = df.index[spike_mask].tolist()

        # 가격-거래량 상관계수
        if len(vol) >= 5:
            pv_corr = float(close.corr(vol))
            if np.isnan(pv_corr):
                pv_corr = 0.0
        else:
            pv_corr = 0.0

        # 대량거래일 (평균의 3배 초과)
        large_trade_days = int((vol > avg_vol * 3).sum())

        return {
            "avg_volume": round(avg_vol, 0),
            "current_ratio": round(current_ratio, 2),
            "volume_trend": volume_trend,
            "spike_dates": spike_dates,
            "price_volume_corr": round(pv_corr, 4),
            "large_trade_days": large_trade_days,
        }

    # ── 내부 헬퍼: 데이터 수집 ─────────────────────────────

    def _fetch_investor_trend(
        self, stock_code: str, period: int,
    ) -> pd.DataFrame:
        """KIS API에서 투자자 매매동향을 조회합니다."""
        if not self._online:
            return pd.DataFrame()

        try:
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": stock_code,
            }
            data = self.collector._get(
                "/uapi/domestic-stock/v1/quotations/inquire-investor",
                TR_INVESTOR_TREND,
                params,
            )
            records = []
            for item in data.get("output", [])[:period]:
                records.append({
                    "date": item.get("stck_bsop_date", ""),
                    "foreign_net": int(float(item.get("frgn_ntby_qty", 0) or 0)),
                    "institution_net": int(float(item.get("orgn_ntby_qty", 0) or 0)),
                    "retail_net": int(float(item.get("prsn_ntby_qty", 0) or 0)),
                })
            return pd.DataFrame(records)
        except Exception as e:
            logger.warning("투자자 매매동향 조회 실패 [{}]: {}", stock_code, e)
            return pd.DataFrame()

    def _fetch_sector_data(self) -> pd.DataFrame:
        """KIS API에서 업종별 데이터를 조회합니다."""
        if not self._online:
            return pd.DataFrame()

        # 실제 API 연동 시 구현 필요
        logger.debug("업종 데이터 API 미구현 — 빈 DataFrame 반환")
        return pd.DataFrame()

    def _fetch_stock_flow_data(
        self, sector: Optional[str] = None,
    ) -> pd.DataFrame:
        """KIS API에서 종목별 수급 데이터를 조회합니다."""
        if not self._online:
            return pd.DataFrame()

        # 실제 API 연동 시 구현 필요
        logger.debug("종목 수급 API 미구현 — 빈 DataFrame 반환")
        return pd.DataFrame()

    def _fetch_execution_data(self, stock_code: str) -> pd.DataFrame:
        """KIS API에서 체결 정보를 조회합니다."""
        if not self._online:
            return pd.DataFrame()

        try:
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": stock_code,
            }
            data = self.collector._get(
                "/uapi/domestic-stock/v1/quotations/inquire-ccnl",
                TR_EXECUTION,
                params,
            )
            records = []
            for item in data.get("output", []):
                records.append({
                    "buy_volume": int(float(item.get("total_askp_rsqn", 0) or 0)),
                    "sell_volume": int(float(item.get("total_bidp_rsqn", 0) or 0)),
                })
            return pd.DataFrame(records)
        except Exception as e:
            logger.warning("체결 데이터 조회 실패 [{}]: {}", stock_code, e)
            return pd.DataFrame()

    # ── 내부 헬퍼: 분석 로직 ───────────────────────────────

    @staticmethod
    def _calc_streak(series: pd.Series) -> int:
        """연속 매수/매도 일수를 계산합니다.

        양수이면 연속 매수일수(+), 음수이면 연속 매도일수(-)를 반환합니다.
        """
        if series.empty:
            return 0

        streak = 0
        last_sign = np.sign(series.iloc[-1])

        if last_sign == 0:
            return 0

        for val in reversed(series.values):
            if np.sign(val) == last_sign:
                streak += 1
            else:
                break

        return int(streak * last_sign)

    @staticmethod
    def _determine_dominant_player(
        foreign_flow: float,
        institution_flow: float,
        retail_flow: float,
    ) -> str:
        """지배적 매매 주체를 판별합니다."""
        flows = {
            "foreign": abs(foreign_flow),
            "institution": abs(institution_flow),
            "retail": abs(retail_flow),
        }
        return max(flows, key=flows.get)

    @staticmethod
    def _determine_trend(
        foreign_flow: float,
        institution_flow: float,
        foreign_streak: int,
    ) -> str:
        """시장 추세를 판단합니다.

        외국인이 대세 방향을 결정하며, 기관 동조 시 강화됩니다.
        """
        # 외국인 + 기관 동시 매수/매도 → 강한 신호
        if foreign_flow > 0 and institution_flow > 0:
            return "bullish"
        if foreign_flow < 0 and institution_flow < 0:
            return "bearish"

        # 외국인 단독 (연속 3일 이상)
        if foreign_streak >= 3:
            return "bullish"
        if foreign_streak <= -3:
            return "bearish"

        # 외국인 방향만으로 판단
        if foreign_flow > 0:
            return "bullish"
        if foreign_flow < 0:
            return "bearish"

        return "neutral"

    @staticmethod
    def _calc_flow_strength(df: pd.DataFrame) -> float:
        """수급 강도를 계산합니다 (0~1).

        외국인 + 기관의 순매수 일관성을 기반으로 산출합니다.
        """
        if df.empty:
            return 0.0

        total_days = len(df)
        if total_days == 0:
            return 0.0

        # 외국인과 기관이 같은 방향인 날의 비율
        combined = df["foreign_net"] + df["institution_net"]
        dominant_sign = np.sign(combined.sum())

        if dominant_sign == 0:
            return 0.0

        aligned_days = int((np.sign(combined) == dominant_sign).sum())
        consistency = aligned_days / total_days

        # 누적 순매수의 크기를 정규화
        abs_sum = combined.abs().sum()
        if abs_sum == 0:
            return 0.0

        directional_ratio = abs(combined.sum()) / abs_sum

        strength = (consistency * 0.6 + directional_ratio * 0.4)
        return round(min(max(strength, 0.0), 1.0), 4)

    @staticmethod
    def _calc_confidence(
        foreign_flow: float,
        institution_flow: float,
        foreign_streak: int,
        period: int,
    ) -> float:
        """신뢰도를 계산합니다 (0~1).

        외국인 연속일수 + 외국인/기관 방향 일치도를 기반으로 산출합니다.
        """
        # 연속일수 기여 (최대 0.5)
        streak_score = min(abs(foreign_streak) / period, 1.0) * 0.5

        # 방향 일치 기여 (최대 0.5)
        if foreign_flow == 0 and institution_flow == 0:
            alignment_score = 0.0
        else:
            same_direction = (
                (foreign_flow > 0 and institution_flow > 0)
                or (foreign_flow < 0 and institution_flow < 0)
            )
            alignment_score = 0.5 if same_direction else 0.15

        confidence = streak_score + alignment_score
        return round(min(max(confidence, 0.0), 1.0), 4)

    @staticmethod
    def _default_signal() -> MarketFlowSignal:
        """데이터 없을 때 기본 신호를 반환합니다."""
        return MarketFlowSignal(
            trend="neutral",
            dominant_player="retail",
            foreign_flow=0.0,
            institution_flow=0.0,
            retail_flow=0.0,
            flow_strength=0.0,
            confidence=0.0,
        )

    @staticmethod
    def _default_volume_profile() -> dict:
        """데이터 없을 때 기본 거래량 프로파일을 반환합니다."""
        return {
            "avg_volume": 0,
            "current_ratio": 1.0,
            "volume_trend": "stable",
            "spike_dates": [],
            "price_volume_corr": 0.0,
            "large_trade_days": 0,
        }


# ── 유틸리티 ───────────────────────────────────────────────

def _safe_normalize(value: float, max_value: float) -> float:
    """안전한 정규화 (0~1, max_value가 0이면 0 반환)"""
    if max_value == 0 or np.isnan(max_value):
        return 0.0
    return float(np.clip(value / max_value, -1.0, 1.0))
