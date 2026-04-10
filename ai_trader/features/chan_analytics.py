"""어니스트 찬 - 알고리즘 트레이딩 핵심 분석 모듈

Ernest P. Chan 'Algorithmic Trading' 기반:
1. 정상성 검정: ADF 테스트, Hurst 지수
2. 평균 회귀 분석: OU 반감기, Z-스코어 신호
3. 공적분 분석: Engle-Granger, 헤지 비율
4. 칼만 필터: 동적 헤지 비율 추정
5. 크로스섹션 평균 회귀: Khandani-Lo 신호
6. 모멘텀 신호: 시계열 모멘텀, 갭 모멘텀
7. 레짐 감지: 평균 회귀 vs 추세 추종
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

# statsmodels 조건부 임포트
try:
    from statsmodels.tsa.stattools import adfuller
    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False
    logger.warning("statsmodels 없음 - ADF 검정 비활성화")


# ---------------------------------------------------------------------------
# Class 1: StationarityAnalyzer
# ---------------------------------------------------------------------------

class StationarityAnalyzer:
    """정상성 검정기

    ADF 테스트와 Hurst 지수로 수익률 시계열의
    정상성(mean-reversion 여부)을 판단한다.

    ADF: H0=랜덤워크, λ<0이고 기각되면 평균 회귀
    Hurst: H<0.5=평균회귀, H=0.5=랜덤워크, H>0.5=추세
    """

    def test(self, prices: pd.Series) -> dict:
        """정상성 검정 수행

        Parameters
        ----------
        prices : pd.Series
            가격 시계열 (종가 등)

        Returns
        -------
        dict
            adf_stat, adf_pvalue, hurst, is_stationary, is_mean_reverting
        """
        result = {
            "adf_stat": np.nan,
            "adf_pvalue": 1.0,
            "hurst": 0.5,
            "is_stationary": False,
            "is_mean_reverting": False,
        }
        try:
            clean = prices.dropna()
            if len(clean) < 20:
                return result

            # ADF 검정 (log 가격에 적용)
            log_p = np.log(clean.values.astype(float))
            if _HAS_STATSMODELS:
                adf_out = adfuller(log_p, autolag="AIC")
                result["adf_stat"] = float(adf_out[0])
                result["adf_pvalue"] = float(adf_out[1])
                result["is_stationary"] = result["adf_pvalue"] < 0.05
            else:
                # fallback: 단순 회귀 ADF 근사
                y = log_p
                dy = np.diff(y)
                y_lag = y[:-1]
                if len(y_lag) > 1:
                    slope, _, _, _, _ = stats.linregress(y_lag, dy)
                    result["adf_stat"] = float(slope)
                    result["is_stationary"] = slope < -0.05

            # Hurst 지수
            result["hurst"] = self._hurst(clean)
            result["is_mean_reverting"] = result["hurst"] < 0.5

        except Exception as exc:
            logger.debug(f"StationarityAnalyzer.test 오류: {exc}")

        return result

    def analyze(self, prices: pd.Series) -> dict:
        """test() 메서드의 별칭 (API 호환)"""
        return self.test(prices)

    def _hurst(self, series: pd.Series) -> float:
        """Variance ratio Hurst exponent estimator"""
        try:
            lags = [2, 4, 8, 16, 32]
            log_prices = np.log(series.dropna().values.astype(float))
            tau: List[float] = []
            lagvec: List[int] = []
            for lag in lags:
                if lag >= len(log_prices):
                    continue
                pp = log_prices[lag:] - log_prices[:-lag]
                tau.append(np.std(pp, ddof=1))
                lagvec.append(lag)
            if len(tau) < 2:
                return 0.5
            # Fit: log(tau) = H * log(lag) + const
            log_lag = np.log(lagvec)
            log_tau = np.log(tau)
            H = np.polyfit(log_lag, log_tau, 1)[0]
            return float(np.clip(H, 0.0, 1.0))
        except Exception as exc:
            logger.debug(f"_hurst 오류: {exc}")
            return 0.5


# ---------------------------------------------------------------------------
# Class 2: MeanReversionEstimator
# ---------------------------------------------------------------------------

class MeanReversionEstimator:
    """평균 회귀 속도 추정기

    Ornstein-Uhlenbeck 공정 기반 반감기 계산:
    dy(t) = λy(t-1)dt + μdt + dε
    half_life = -log(2) / λ

    lookback = halflife 규칙 적용
    """

    def estimate(self, prices: pd.Series) -> dict:
        """OU 공정 파라미터 추정

        Parameters
        ----------
        prices : pd.Series
            가격 시계열

        Returns
        -------
        dict
            lambda_, half_life, lookback, is_useful
        """
        result = {
            "lambda_": np.nan,
            "half_life": np.nan,
            "lookback": 20,
            "is_useful": False,
        }
        try:
            clean = prices.dropna().astype(float)
            if len(clean) < 10:
                return result

            y = clean.values
            y_lag = y[:-1]
            delta_y = np.diff(y)

            # OLS: delta_y ~ y_lag
            slope, _, _, _, _ = stats.linregress(y_lag, delta_y)
            result["lambda_"] = float(slope)

            if slope < 0:
                hl = -np.log(2.0) / slope
                result["half_life"] = float(hl)
                lk = max(2, round(hl))
                result["lookback"] = int(lk)
                result["is_useful"] = 2.0 <= hl <= 250.0
            else:
                result["half_life"] = np.inf
                result["is_useful"] = False

        except Exception as exc:
            logger.debug(f"MeanReversionEstimator.estimate 오류: {exc}")

        return result

    def zscore(self, prices: pd.Series, lookback: int) -> pd.Series:
        """Z-스코어 계산

        Parameters
        ----------
        prices : pd.Series
            가격 시계열
        lookback : int
            롤링 윈도우

        Returns
        -------
        pd.Series
            Z-스코어 시계열
        """
        try:
            lookback = max(2, lookback)
            rolling_mean = prices.rolling(lookback).mean()
            rolling_std = prices.rolling(lookback).std(ddof=1)
            z = (prices - rolling_mean) / rolling_std.replace(0, np.nan)
            return z.fillna(0.0)
        except Exception as exc:
            logger.debug(f"MeanReversionEstimator.zscore 오류: {exc}")
            return pd.Series(0.0, index=prices.index)

    def signal(
        self,
        prices: pd.Series,
        lookback: int,
        entry_z: float = 1.0,
        exit_z: float = 0.0,
    ) -> pd.Series:
        """평균 회귀 매매 신호 생성

        Parameters
        ----------
        prices : pd.Series
            가격 시계열
        lookback : int
            롤링 윈도우
        entry_z : float
            진입 Z-스코어 임계값
        exit_z : float
            청산 Z-스코어 임계값

        Returns
        -------
        pd.Series
            +1(롱), -1(숏), 0(중립)
        """
        try:
            z = self.zscore(prices, lookback)
            sig = pd.Series(0, index=prices.index, dtype=int)
            position = 0
            for i in range(len(z)):
                zv = z.iloc[i]
                if np.isnan(zv):
                    sig.iloc[i] = 0
                    continue
                if position == 0:
                    if zv < -entry_z:
                        position = 1
                    elif zv > entry_z:
                        position = -1
                elif position == 1:
                    if abs(zv) < exit_z:
                        position = 0
                elif position == -1:
                    if abs(zv) < exit_z:
                        position = 0
                sig.iloc[i] = position
            return sig
        except Exception as exc:
            logger.debug(f"MeanReversionEstimator.signal 오류: {exc}")
            return pd.Series(0, index=prices.index, dtype=int)

    def analyze(self, prices: pd.Series) -> dict:
        """estimate + zscore + signal 통합 결과 반환"""
        result = self.estimate(prices)
        lk = result.get("lookback", 20)
        try:
            z_series = self.zscore(prices, lk)
            z = float(z_series.iloc[-1]) if not z_series.empty else 0.0
        except Exception:
            z = 0.0
        try:
            sig_series = self.signal(prices, lk)
            sig = int(sig_series.iloc[-1]) if not sig_series.empty else 0
        except Exception:
            sig = 0
        result["zscore"] = z
        result["signal"] = sig
        return result


# ---------------------------------------------------------------------------
# Class 3: CointegrationAnalyzer
# ---------------------------------------------------------------------------

class CointegrationAnalyzer:
    """공적분 분석기

    Engle-Granger(CADF) 방식으로 두 시리즈 간
    공적분 관계를 검정하고 헤지 비율을 산출한다.

    공적분 ≠ 상관관계:
    - 상관: 단기 동조화
    - 공적분: 장기 가격 격차의 정상성

    CADF 임계값: 1%=-3.819, 5%=-3.343, 10%=-3.042
    """

    # Engle-Granger CADF 임계값 (n=100 기준 근사)
    _CRITICAL = {
        "99%": -3.819,
        "95%": -3.343,
        "90%": -3.042,
    }

    def test(self, y1: pd.Series, y2: pd.Series) -> dict:
        """공적분 검정 수행

        Parameters
        ----------
        y1 : pd.Series
            종속 가격 시계열
        y2 : pd.Series
            독립 가격 시계열

        Returns
        -------
        dict
            hedge_ratio, spread, adf_stat, p_value,
            confidence, half_life
        """
        result = {
            "hedge_ratio": 1.0,
            "spread": pd.Series(dtype=float),
            "adf_stat": np.nan,
            "p_value": 1.0,
            "confidence": "none",
            "half_life": np.nan,
        }
        try:
            # 공통 인덱스 정렬
            common = y1.index.intersection(y2.index)
            if len(common) < 30:
                return result
            a = y1.loc[common].astype(float)
            b = y2.loc[common].astype(float)

            # OLS: y1 = hedge_ratio * y2 + intercept
            slope, intercept, _, _, _ = stats.linregress(b.values, a.values)
            result["hedge_ratio"] = float(slope)

            # 스프레드
            spr = a - slope * b
            result["spread"] = spr

            # ADF on spread
            if _HAS_STATSMODELS:
                adf_out = adfuller(spr.dropna().values, autolag="AIC")
                stat = float(adf_out[0])
                pval = float(adf_out[1])
                result["adf_stat"] = stat
                result["p_value"] = pval
            else:
                # fallback 근사
                y = spr.values
                dy = np.diff(y)
                y_lag = y[:-1]
                slope_r, _, _, _, _ = stats.linregress(y_lag, dy)
                stat = float(slope_r * len(y) ** 0.5)
                result["adf_stat"] = stat
                pval = 1.0

            # 신뢰도 판정
            conf = "none"
            for lvl, cv in self._CRITICAL.items():
                if result["adf_stat"] < cv:
                    conf = lvl
                    break
            result["confidence"] = conf

            # 반감기
            mr = MeanReversionEstimator()
            mr_res = mr.estimate(spr)
            result["half_life"] = mr_res["half_life"]

        except Exception as exc:
            logger.debug(f"CointegrationAnalyzer.test 오류: {exc}")

        return result

    def spread(
        self, y1: pd.Series, y2: pd.Series, hedge_ratio: float
    ) -> pd.Series:
        """스프레드 계산

        Parameters
        ----------
        y1 : pd.Series
        y2 : pd.Series
        hedge_ratio : float

        Returns
        -------
        pd.Series
            스프레드 = y1 - hedge_ratio * y2
        """
        return y1 - hedge_ratio * y2

    def analyze(self, y1: pd.Series, y2: pd.Series) -> dict:
        """test() 메서드의 별칭 + is_cointegrated/pvalue 키 추가"""
        result = self.test(y1, y2)
        result["is_cointegrated"] = result["confidence"] != "none"
        result["pvalue"] = result["p_value"]
        return result


# ---------------------------------------------------------------------------
# Class 4: KalmanFilterHedge
# ---------------------------------------------------------------------------

class KalmanFilterHedge:
    """칼만 필터 동적 헤지 비율 추정기

    Chan 최적 파라미터: delta=0.0001, Ve=0.001
    EWA-EWC 결과: APR 26.2%, Sharpe 2.4

    상태 방정식: β(t) = β(t-1) + ω(t-1)
    측정 방정식: y(t) = x(t)β(t) + ε(t)
    """

    def __init__(self, delta: float = 0.0001, Ve: float = 0.001):
        self.delta = delta
        self.Ve = Ve
        # Vw = delta/(1-delta) * I_2 (2차원: [hedge_ratio, intercept])
        self.Vw = (delta / (1 - delta)) * np.eye(2)

    def filter(self, y: pd.Series, x: pd.Series) -> dict:
        """칼만 필터 실행

        Parameters
        ----------
        y : pd.Series
            종속 시계열 (ETF1 등)
        x : pd.Series
            독립 시계열 (ETF2 등)

        Returns
        -------
        dict
            hedge_ratio, intercept, e, Q, signal 키 (각 pd.Series)
        """
        try:
            common = y.index.intersection(x.index)
            if len(common) < 10:
                return {}
            yv = y.loc[common].astype(float).values
            xv = x.loc[common].astype(float).values
            n = len(yv)

            # 초기화
            beta = np.zeros((n, 2))   # [hedge_ratio, intercept]
            P = np.zeros((n, 2, 2))   # 오차 공분산
            e_arr = np.zeros(n)
            Q_arr = np.zeros(n)

            beta[0] = np.array([1.0, 0.0])
            P[0] = self.Vw.copy()

            for t in range(1, n):
                # 예측
                beta_pred = beta[t - 1]
                P_pred = P[t - 1] + self.Vw

                # 측정 벡터
                Ft = np.array([xv[t], 1.0])

                # 예측 오차
                e = yv[t] - float(Ft @ beta_pred)
                Q = float(Ft @ P_pred @ Ft) + self.Ve

                # 칼만 이득
                K = (P_pred @ Ft) / Q

                # 상태 업데이트
                beta[t] = beta_pred + K * e
                P[t] = P_pred - np.outer(K, Ft) @ P_pred

                e_arr[t] = e
                Q_arr[t] = Q

            # 신호 생성: e < -sqrt(Q) → long y, short x (+1)
            #            e > +sqrt(Q) → short y, long x (-1)
            sqrt_Q = np.sqrt(np.abs(Q_arr))
            signal_arr = np.where(
                e_arr < -sqrt_Q, 1,
                np.where(e_arr > sqrt_Q, -1, 0)
            )

            idx = common
            return {
                "hedge_ratio": pd.Series(beta[:, 0], index=idx),
                "intercept": pd.Series(beta[:, 1], index=idx),
                "e": pd.Series(e_arr, index=idx),
                "Q": pd.Series(Q_arr, index=idx),
                "signal": pd.Series(signal_arr, index=idx),
            }
        except Exception as exc:
            logger.debug(f"KalmanFilterHedge.filter 오류: {exc}")
            return {}


# ---------------------------------------------------------------------------
# Class 5: CrossSectionalMRSignal
# ---------------------------------------------------------------------------

class CrossSectionalMRSignal:
    """크로스섹션 평균 회귀 신호 (Khandani-Lo)

    w_i = -(r_i - <r_j>) / Σ_k|r_k - <r_j>|

    어제 상대적 하락 → 오늘 매수
    어제 상대적 상승 → 오늘 공매도

    2008년 성과: APR 30% (금융위기 중 탁월)
    """

    def compute(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """각 날짜의 크로스섹션 비중 DataFrame 반환

        Parameters
        ----------
        returns_df : pd.DataFrame
            일별 수익률 (행=날짜, 열=종목)

        Returns
        -------
        pd.DataFrame
            날짜별 종목 비중 (달러 중립: 합계 ≈ 0)
        """
        try:
            if returns_df is None or len(returns_df) < 1:
                return pd.DataFrame()

            result_rows = []
            for t in range(len(returns_df)):
                r = returns_df.iloc[t]
                mean_r = r.mean()
                denom = (r - mean_r).abs().sum()
                if denom == 0:
                    w = pd.Series(0.0, index=returns_df.columns)
                else:
                    w = -(r - mean_r) / denom
                result_rows.append(w)
            return pd.DataFrame(result_rows, index=returns_df.index)
        except Exception as exc:
            logger.debug(f"CrossSectionalMRSignal.compute 오류: {exc}")
            return pd.DataFrame()

    def score(self, returns_df: pd.DataFrame) -> float:
        """전략 연환산 샤프 비율 계산

        Parameters
        ----------
        returns_df : pd.DataFrame
            일별 수익률 (행=날짜, 열=종목)

        Returns
        -------
        float
            연환산 샤프 비율
        """
        try:
            if returns_df is None or len(returns_df) < 10:
                return 0.0

            portfolio_rets = []
            for t in range(1, len(returns_df)):
                # t-1 수익률로 비중 산출
                r_prev = returns_df.iloc[t - 1]
                mean_r = r_prev.mean()
                denom = (r_prev - mean_r).abs().sum()
                if denom == 0:
                    portfolio_rets.append(0.0)
                    continue
                w = -(r_prev - mean_r) / denom
                # t일 수익률
                r_today = returns_df.iloc[t]
                pnl = float((w * r_today).sum())
                portfolio_rets.append(pnl)

            if len(portfolio_rets) < 2:
                return 0.0

            ret_arr = np.array(portfolio_rets)
            mean_ret = ret_arr.mean()
            std_ret = ret_arr.std(ddof=1)
            if std_ret == 0:
                return 0.0
            sharpe = (mean_ret / std_ret) * np.sqrt(252)
            return float(sharpe)
        except Exception as exc:
            logger.debug(f"CrossSectionalMRSignal.score 오류: {exc}")
            return 0.0


# ---------------------------------------------------------------------------
# Class 6: MomentumSignal
# ---------------------------------------------------------------------------

class MomentumSignal:
    """모멘텀 신호

    1. 시계열 모멘텀: 현재 가격 > N일 전 가격 → 매수
    2. 갭 모멘텀 (Buy-on-Gap 역전략):
       시가 갭다운 < -1SD AND 시가 > 20일 이평 → 매수
       (평균 회귀형 갭 전략)

    Chan 최적값: lookback=250, hold=25 (선물 모멘텀)
    Gap model: APR 8.7%, Sharpe 1.5 (S&P500, 2006-2012)
    """

    def timeseries_signal(
        self, prices: pd.Series, lookback: int = 252
    ) -> pd.Series:
        """시계열 모멘텀 신호

        Parameters
        ----------
        prices : pd.Series
            가격 시계열
        lookback : int
            모멘텀 기간 (기본 252 거래일)

        Returns
        -------
        pd.Series
            +1(상승 모멘텀), -1(하락 모멘텀)
        """
        try:
            past_price = prices.shift(lookback)
            signal = np.where(prices > past_price, 1, -1)
            return pd.Series(signal, index=prices.index, dtype=int)
        except Exception as exc:
            logger.debug(f"MomentumSignal.timeseries_signal 오류: {exc}")
            return pd.Series(0, index=prices.index, dtype=int)

    def gap_reversal_signal(
        self,
        df: pd.DataFrame,
        entry_z: float = 1.0,
        ma_period: int = 20,
        std_period: int = 90,
    ) -> pd.Series:
        """갭 역전 신호 (Buy-on-Gap 평균 회귀)

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV 데이터프레임 (open, close 필수)
        entry_z : float
            진입 Z-스코어 배수
        ma_period : int
            이동평균 기간
        std_period : int
            표준편차 계산 기간

        Returns
        -------
        pd.Series
            +1(갭다운 매수), -1(갭업 공매도), 0(대기)
        """
        try:
            required = {"open", "close"}
            if not required.issubset(df.columns):
                return pd.Series(0, index=df.index, dtype=int)

            prev_close = df["close"].shift(1)
            gap = (df["open"] - prev_close) / prev_close.replace(0, np.nan)

            # 갭의 롤링 표준편차
            gap_std = gap.rolling(std_period, min_periods=10).std(ddof=1)

            # 이동평균
            ma = df["close"].rolling(ma_period, min_periods=5).mean()

            # 신호
            gap_z = gap / gap_std.replace(0, np.nan)

            long_cond = (gap_z < -entry_z) & (df["open"] > ma)
            short_cond = (gap_z > entry_z) & (df["open"] < ma)

            signal = pd.Series(0, index=df.index, dtype=int)
            signal[long_cond] = 1
            signal[short_cond] = -1
            return signal
        except Exception as exc:
            logger.debug(f"MomentumSignal.gap_reversal_signal 오류: {exc}")
            return pd.Series(0, index=df.index, dtype=int)


# ---------------------------------------------------------------------------
# Class 7: RegimeDetector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """레짐 감지기 (평균 회귀 vs 추세 추종)

    Hurst 지수와 ADF p-value로 현재 레짐 판단:
    - H < 0.45 AND ADF p < 0.1 → MEAN_REVERSION
    - H > 0.55 → TRENDING
    - 그 외 → NEUTRAL

    rolling window로 최근 레짐 추적
    """

    MEAN_REVERSION = "MEAN_REVERSION"
    TRENDING = "TRENDING"
    NEUTRAL = "NEUTRAL"

    def __init__(self, lookback: int = 126):
        self.lookback = lookback
        self._stat = StationarityAnalyzer()

    def detect(self, prices: pd.Series) -> dict:
        """레짐 감지

        Parameters
        ----------
        prices : pd.Series
            가격 시계열 (전체, 최근 lookback 바 사용)

        Returns
        -------
        dict
            regime, hurst, adf_pvalue, confidence
        """
        result = {
            "regime": self.NEUTRAL,
            "hurst": 0.5,
            "adf_pvalue": 1.0,
            "confidence": 0.0,
        }
        try:
            recent = prices.dropna().iloc[-self.lookback:]
            if len(recent) < 30:
                return result

            stat = self._stat.test(recent)
            H = stat["hurst"]
            p = stat["adf_pvalue"]

            result["hurst"] = H
            result["adf_pvalue"] = p

            if H < 0.45 and p < 0.10:
                result["regime"] = self.MEAN_REVERSION
                # 신뢰도: Hurst 거리 + ADF 유의성 결합
                conf = (0.45 - H) / 0.45 * 0.5 + (0.10 - p) / 0.10 * 0.5
                result["confidence"] = float(np.clip(conf, 0.0, 1.0))
            elif H > 0.55:
                result["regime"] = self.TRENDING
                conf = (H - 0.55) / 0.45
                result["confidence"] = float(np.clip(conf, 0.0, 1.0))
            else:
                result["regime"] = self.NEUTRAL
                result["confidence"] = 0.0

        except Exception as exc:
            logger.debug(f"RegimeDetector.detect 오류: {exc}")

        return result


# ---------------------------------------------------------------------------
# Class 8: ChanAnalyticsSignal (dataclass)
# ---------------------------------------------------------------------------

@dataclass
class ChanAnalyticsSignal:
    """찬 분석 통합 신호"""

    # 정상성
    hurst: float = 0.5
    adf_pvalue: float = 0.5
    is_mean_reverting: bool = False

    # 평균 회귀
    half_life: float = 0.0
    lookback: int = 20
    zscore: float = 0.0
    mr_signal: int = 0  # +1/-1/0

    # 공적분 (페어 데이터 있는 경우)
    hedge_ratio: float = 1.0
    cointegration_confidence: str = "none"

    # 모멘텀
    ts_momentum: int = 0   # +1/-1
    gap_signal: int = 0    # +1/-1/0

    # 레짐
    regime: str = "NEUTRAL"  # MEAN_REVERSION/TRENDING/NEUTRAL

    # 포지션 스케일
    position_scale: float = 1.0
    note: str = ""


# ---------------------------------------------------------------------------
# Class 9: ChanAnalyzer (main integrator)
# ---------------------------------------------------------------------------

class ChanAnalyzer:
    """찬 통합 분석기

    단일 종목 OHLCV로 ChanAnalyticsSignal 생성.
    book_integrator.py에서 신호로 활용 가능.
    """

    def __init__(self, lookback: int = 126):
        self.lookback = lookback
        self._stat = StationarityAnalyzer()
        self._mr = MeanReversionEstimator()
        self._regime = RegimeDetector(lookback)
        self._momentum = MomentumSignal()

    def analyze_signal(self, df: pd.DataFrame) -> ChanAnalyticsSignal:
        """OHLCV 데이터프레임으로 통합 신호 생성

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV 데이터 (close 필수, open 있으면 갭 신호 추가)

        Returns
        -------
        ChanAnalyticsSignal
        """
        signal = ChanAnalyticsSignal()
        if df is None or len(df) < 60:
            signal.note = "데이터 부족 (최소 60봉 필요)"
            return signal

        try:
            prices = df["close"].dropna()

            # 1. 정상성 검정
            stat = self._stat.test(prices)
            signal.hurst = stat["hurst"]
            signal.adf_pvalue = stat["adf_pvalue"]
            signal.is_mean_reverting = stat["is_mean_reverting"]

            # 2. 평균 회귀 속도 추정
            mr = self._mr.estimate(prices)
            signal.half_life = mr["half_life"] if not np.isnan(
                mr["half_life"]
            ) else 0.0
            signal.lookback = mr["lookback"]

            # 3. Z-스코어 및 MR 신호
            if mr["is_useful"]:
                lk = mr["lookback"]
                z_series = self._mr.zscore(prices, lk)
                signal.zscore = float(z_series.iloc[-1]) if not z_series.empty else 0.0
                sig_series = self._mr.signal(prices, lk)
                signal.mr_signal = (
                    int(sig_series.iloc[-1]) if not sig_series.empty else 0
                )

            # 4. 레짐 감지
            reg = self._regime.detect(prices)
            signal.regime = reg["regime"]

            # 5. 시계열 모멘텀
            mom = self._momentum.timeseries_signal(prices)
            signal.ts_momentum = int(mom.iloc[-1]) if not mom.empty else 0

            # 6. 갭 역전 신호
            if all(c in df.columns for c in ["open", "close"]):
                gap = self._momentum.gap_reversal_signal(df)
                signal.gap_signal = int(gap.iloc[-1]) if not gap.empty else 0

            # 7. 포지션 스케일 (레짐 신뢰도 반영)
            if signal.regime == RegimeDetector.MEAN_REVERSION and signal.is_mean_reverting:
                signal.position_scale = 1.2
            elif signal.regime == RegimeDetector.TRENDING:
                signal.position_scale = 1.0
            else:
                signal.position_scale = 0.8

            signal.note = (
                f"regime={signal.regime}, H={signal.hurst:.3f}, "
                f"hl={signal.half_life:.1f}, z={signal.zscore:.2f}"
            )

        except Exception as exc:
            logger.debug(f"ChanAnalyzer.analyze_signal 오류: {exc}")
            signal.note = f"분석 오류: {exc}"

        return signal

    def analyze(self, df: pd.DataFrame) -> dict:
        """OHLCV 분석 결과를 dict로 반환 (테스트 호환)

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV 데이터

        Returns
        -------
        dict
            signal, regime, hurst, ... 키를 포함한 딕셔너리
        """
        _default = {
            "signal": 0,
            "regime": "NEUTRAL",
            "hurst": 0.5,
            "adf_pvalue": 1.0,
            "half_life": 0.0,
            "lookback": 20,
            "zscore": 0.0,
            "mr_signal": 0,
            "ts_momentum": 0,
            "gap_signal": 0,
            "position_scale": 1.0,
            "note": "데이터 부족",
        }
        if df is None or len(df) < 60:
            return _default
        try:
            sig = self.analyze_signal(df)
            return {
                "signal": sig.mr_signal,
                "regime": sig.regime,
                "hurst": sig.hurst,
                "adf_pvalue": sig.adf_pvalue,
                "half_life": sig.half_life,
                "lookback": sig.lookback,
                "zscore": sig.zscore,
                "mr_signal": sig.mr_signal,
                "ts_momentum": sig.ts_momentum,
                "gap_signal": sig.gap_signal,
                "position_scale": sig.position_scale,
                "note": sig.note,
            }
        except Exception as exc:
            logger.debug(f"ChanAnalyzer.analyze 오류: {exc}")
            return _default
