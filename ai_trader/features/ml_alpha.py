"""Stefan Jansen - Machine Learning for Algorithmic Trading

ML 湲곕컲 ?븣?뙆?뙥?꽣 異붿텧 諛? ?븰?긽釉? ?떊?샇 ?깮?꽦 紐⑤뱢.
?쇅遺? ML ?씪?씠釉뚮윭由? ?쓽議? ?뾾?씠 湲곗닠?쟻 ?봽濡앹떆濡? 援ы쁽:
1. ?븣?뙆 ?뙥?꽣 ?깮?꽦 (湲곗닠?쟻 吏??몴 湲곕컲)
2. ?뙥?꽣 以묒슂?룄 ?닚?쐞 (?젙蹂닿퀎?닔 IC 湲곕컲)
3. ?뵾泥? ?겢?윭?뒪?꽣留? (?긽愿? 湲곕컲 洹몃９?솕)
4. ?븰?긽釉? ?떊?샇 (?떎?닔 ?뙥?꽣?쓽 媛?以? ?빀?궛)
5. ?꽱?떚癒쇳듃 ?봽濡앹떆 (媛?寃?/嫄곕옒?웾 ?뙣?꽩 湲곕컲)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class MLAlphaSignal:
    """ML ?븣?뙆?뙥?꽣 ?떊?샇"""
    # ?븰?긽釉? ?젏?닔 (0~1, ?넂?쓣?닔濡? 留ㅼ닔 ?쑀由?)
    ensemble_score: float
    # ?긽?쐞 ?뙥?꽣 3媛쒖?? IC媛?
    top_factors: Dict[str, float] = field(default_factory=dict)
    # ?꽱?떚癒쇳듃 ?봽濡앹떆 (-1~+1)
    sentiment_proxy: float = 0.0
    # ?뙥?꽣 遺꾩궛?룄 (0~1, ?넂?쓣?닔濡? ?뙥?꽣 媛? ?쓽寃? ?씪移?)
    factor_agreement: float = 0.0
    # ?젙蹂? 媛먯뇿?쑉 (0~1, ?넂?쓣?닔濡? ?븣?뙆 鍮좊Ⅴ寃? ?냼硫?)
    alpha_decay: float = 0.0
    # ?룷吏??뀡 諛곗닔 (0.3~1.5)
    position_multiplier: float = 1.0
    # ?떊猶곕룄 議곗젙 (-0.3~+0.3)
    confidence_adjustment: float = 0.0
    # ?꽕紐?
    note: str = ""


class AlphaFactorGenerator:
    """湲곗닠?쟻 吏??몴 湲곕컲 ?븣?뙆?뙥?꽣 ?깮?꽦湲?

    媛? ?뙥?꽣?뒗 醫낅ぉ?쓽 誘몃옒 ?닔?씡瑜좉낵 ?긽愿??씠 ?엳?쓣 寃껋쑝濡? 湲곕???릺?뒗 吏??몴.
    """

    @staticmethod
    def momentum_factor(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """紐⑤찘??? ?뙥?꽣: N?씪 ?닔?씡瑜?"""
        return df["close"].pct_change(period)

    @staticmethod
    def mean_reversion_factor(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """?룊洹좏쉶洹? ?뙥?꽣: ?씠?룞?룊洹? ???鍮? 愿대━?쑉 (?뿭?쟾)"""
        ma = df["close"].rolling(period).mean()
        deviation = (df["close"] - ma) / ma
        return -deviation  # 愿대━媛? ?겢?닔濡? 諛섏쟾 湲곕??

    @staticmethod
    def volume_surprise_factor(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """嫄곕옒?웾 ?꽌?봽?씪?씠利? ?뙥?꽣: 鍮꾩젙?긽 嫄곕옒?웾"""
        avg_vol = df["volume"].rolling(period).mean()
        vol_std = df["volume"].rolling(period).std()
        return (df["volume"] - avg_vol) / vol_std.replace(0, np.nan)

    @staticmethod
    def volatility_factor(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """蹂??룞?꽦 ?뙥?꽣: ?떎?쁽 蹂??룞?꽦 (?뿭, ???蹂??룞 ?꽑?샇)"""
        returns = df["close"].pct_change()
        vol = returns.rolling(period).std()
        return -vol  # 蹂??룞?꽦 ?궙?쓣?닔濡? ?뼇?샇

    @staticmethod
    def price_acceleration_factor(df: pd.DataFrame) -> pd.Series:
        """媛?寃? 媛??냽?룄 ?뙥?꽣: ?닔?씡瑜좎쓽 蹂??솕?쑉"""
        ret = df["close"].pct_change()
        return ret.diff()

    @staticmethod
    def rsi_factor(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """RSI 湲곕컲 怨쇰ℓ?룄/怨쇰ℓ?닔 ?뙥?꽣 (0~1, ?젙洹쒗솕)"""
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        # 怨쇰ℓ?룄?씪?닔濡? ?넂??? ?젏?닔 (?뿭?쟾)
        return (100 - rsi) / 100

    @staticmethod
    def obv_trend_factor(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """OBV 異붿꽭 ?뙥?꽣: OBV?쓽 湲곗슱湲?"""
        direction = np.sign(df["close"].diff())
        obv = (direction * df["volume"]).cumsum()
        return obv.diff(period) / df["volume"].rolling(period).mean().replace(0, np.nan)

    @staticmethod
    def range_factor(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """?젅?씤吏? ?뙥?꽣: ?쁽?옱 媛?寃⑹쓽 N?씪 踰붿쐞 ?궡 ?쐞移? (0~1)"""
        high_max = df["high"].rolling(period).max()
        low_min = df["low"].rolling(period).min()
        range_size = high_max - low_min
        return (df["close"] - low_min) / range_size.replace(0, np.nan)

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """紐⑤뱺 ?븣?뙆?뙥?꽣 ?깮?꽦"""
        factors = pd.DataFrame(index=df.index)
        factors["f_momentum_5"] = self.momentum_factor(df, 5)
        factors["f_momentum_20"] = self.momentum_factor(df, 20)
        factors["f_momentum_60"] = self.momentum_factor(df, 60)
        factors["f_mean_rev_20"] = self.mean_reversion_factor(df, 20)
        factors["f_mean_rev_60"] = self.mean_reversion_factor(df, 60)
        factors["f_vol_surprise"] = self.volume_surprise_factor(df)
        factors["f_volatility"] = self.volatility_factor(df)
        factors["f_price_accel"] = self.price_acceleration_factor(df)
        factors["f_rsi"] = self.rsi_factor(df)
        factors["f_obv_trend"] = self.obv_trend_factor(df)
        factors["f_range"] = self.range_factor(df)
        return factors


class InformationCoefficientCalculator:
    """?젙蹂닿퀎?닔 (IC) 怨꾩궛湲?

    媛? ?뙥?꽣??? 誘몃옒 ?닔?씡瑜? 媛꾩쓽 ?닚?쐞 ?긽愿?(Spearman)?쓣 怨꾩궛.
    IC媛? ?넂??? ?뙥?꽣媛? ?삁痢〓젰?씠 媛뺥븳 ?뙥?꽣.
    """

    def __init__(self, forward_period: int = 5):
        self.forward_period = forward_period

    def compute_ic(self, factor: pd.Series, returns: pd.Series) -> float:
        """?떒?씪 ?뙥?꽣?쓽 IC 怨꾩궛 (?닚?쐞 ?긽愿?)"""
        valid = pd.DataFrame({"factor": factor, "returns": returns}).dropna()
        if len(valid) < 20:
            return 0.0

        # Spearman rank correlation ?봽濡앹떆 (scipy ?뾾?씠)
        rank_f = valid["factor"].rank()
        rank_r = valid["returns"].rank()
        n = len(valid)
        d_sq = ((rank_f - rank_r) ** 2).sum()
        rho = 1 - (6 * d_sq) / (n * (n**2 - 1))
        return float(np.clip(rho, -1.0, 1.0))

    def rank_factors(
        self, factors_df: pd.DataFrame, close: pd.Series
    ) -> Dict[str, float]:
        """紐⑤뱺 ?뙥?꽣?쓽 IC ?닚?쐞

        Returns:
            {factor_name: ic_value} IC ?젅???媛? ?궡由쇱감?닚
        """
        forward_ret = close.pct_change(self.forward_period).shift(-self.forward_period)

        ic_dict = {}
        for col in factors_df.columns:
            ic = self.compute_ic(factors_df[col], forward_ret)
            ic_dict[col] = ic

        return dict(sorted(ic_dict.items(), key=lambda x: abs(x[1]), reverse=True))


class FactorClusterer:
    """?뙥?꽣 ?겢?윭?뒪?꽣留? (?긽愿? 湲곕컲)

    ?긽愿??씠 ?넂??? ?뙥?꽣?겮由? 洹몃９?솕?븯?뿬 以묐났 ?뙥?꽣 ?젣嫄?.
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def cluster(self, factors_df: pd.DataFrame) -> List[List[str]]:
        """?긽愿? 湲곕컲 ?겢?윭?뒪?꽣留?"""
        corr = factors_df.corr().abs()
        used = set()
        clusters = []

        for col in corr.columns:
            if col in used:
                continue
            cluster = [col]
            used.add(col)
            for other in corr.columns:
                if other not in used and corr.loc[col, other] >= self.threshold:
                    cluster.append(other)
                    used.add(other)
            clusters.append(cluster)

        return clusters


class SentimentProxy:
    """?꽱?떚癒쇳듃 ?봽濡앹떆

    NLP ?뾾?씠 媛?寃?/嫄곕옒?웾 ?뙣?꽩?뿉?꽌 ?떆?옣 ?떖由щ?? 異붾줎.
    """

    def compute(self, df: pd.DataFrame, period: int = 20) -> float:
        """?꽱?떚癒쇳듃 ?봽濡앹떆 怨꾩궛 (-1~+1)"""
        if len(df) < period:
            return 0.0

        recent = df.tail(period)

        # 1. 媛?寃? 紐⑤찘??? (40%)
        ret = (recent["close"].iloc[-1] / recent["close"].iloc[0]) - 1
        momentum_score = float(np.clip(ret * 10, -1, 1))

        # 2. 嫄곕옒?웾 異붿꽭 (30%)
        vol_ma_early = recent["volume"].iloc[:period // 2].mean()
        vol_ma_late = recent["volume"].iloc[period // 2:].mean()
        if vol_ma_early > 0:
            vol_trend = (vol_ma_late / vol_ma_early) - 1
        else:
            vol_trend = 0.0
        vol_score = float(np.clip(vol_trend, -1, 1))

        # 3. ?뼇遊? 鍮꾩쑉 (30%)
        up_days = (recent["close"] > recent["open"]).sum()
        up_ratio = up_days / len(recent) if len(recent) > 0 else 0.5
        up_score = (up_ratio - 0.5) * 2  # -1 ~ +1

        sentiment = momentum_score * 0.4 + vol_score * 0.3 + up_score * 0.3
        return float(np.clip(sentiment, -1.0, 1.0))


class MLAlphaAnalyzer:
    """ML ?븣?뙆?뙥?꽣 ?넻?빀 遺꾩꽍湲?"""

    def __init__(self, lookback: int = 120, forward_period: int = 5):
        self.lookback = lookback
        self.factor_gen = AlphaFactorGenerator()
        self.ic_calc = InformationCoefficientCalculator(forward_period)
        self.clusterer = FactorClusterer(threshold=0.7)
        self.sentiment = SentimentProxy()

    def analyze(self, df: pd.DataFrame) -> MLAlphaSignal:
        """ML ?븣?뙆?뙥?꽣 醫낇빀 遺꾩꽍"""
        if len(df) < 60:
            return MLAlphaSignal(
                ensemble_score=0.5,
                position_multiplier=1.0,
                confidence_adjustment=0.0,
                note="?뜲?씠?꽣 遺?議? (理쒖냼 60遊? ?븘?슂)",
            )

        try:
            data = df.tail(self.lookback + 60)

            # 1. ?뙥?꽣 ?깮?꽦
            factors = self.factor_gen.generate_all(data)

            # 2. IC ?닚?쐞
            ic_ranks = self.ic_calc.rank_factors(factors.iloc[:-5], data["close"].iloc[:-5])
            top_factors = dict(list(ic_ranks.items())[:3])

            # 3. ?뙥?꽣 ?겢?윭?뒪?꽣留곸쑝濡? ????몴 ?뙥?꽣 ?꽑?젙
            clusters = self.clusterer.cluster(factors.dropna())
            representative_factors = [c[0] for c in clusters]

            # 4. ?븰?긽釉? ?젏?닔 (IC 媛?以? ?빀?궛)
            latest_factors = factors.iloc[-1]
            ensemble = 0.0
            total_weight = 0.0

            for fname in representative_factors:
                ic = ic_ranks.get(fname, 0.0)
                fval = latest_factors.get(fname, np.nan)
                if pd.notna(fval) and ic != 0:
                    # ?뙥?꽣媛믪쓣 0~1濡? ?젙洹쒗솕 (理쒓렐 遺꾪룷 湲곗??)
                    col = factors[fname].dropna()
                    if len(col) > 10:
                        pctile = (col < fval).sum() / len(col)
                    else:
                        pctile = 0.5

                    weight = abs(ic)
                    # IC 遺??샇?뿉 ?뵲?씪 諛⑺뼢 議곗젙
                    if ic > 0:
                        ensemble += pctile * weight
                    else:
                        ensemble += (1 - pctile) * weight
                    total_weight += weight

            if total_weight > 0:
                ensemble /= total_weight
            else:
                ensemble = 0.5

            ensemble = float(np.clip(ensemble, 0.0, 1.0))

            # 5. ?꽱?떚癒쇳듃 ?봽濡앹떆
            sent = self.sentiment.compute(data)

            # 6. ?뙥?꽣 ?씪移섎룄 (理쒓렐 ?뙥?꽣媛믩뱾?쓽 遺??샇 ?씪移? 鍮꾩쑉)
            latest_signs = np.sign(latest_factors.dropna())
            if len(latest_signs) > 0:
                agreement = abs(latest_signs.mean())
            else:
                agreement = 0.0

            # 7. ?븣?뙆 媛먯뇿?쑉 (?떒湲? IC vs ?옣湲? IC 李⑥씠)
            recent_factors = factors.tail(30)
            recent_ic = self.ic_calc.rank_factors(
                recent_factors.iloc[:-5], data["close"].tail(30).iloc[:-5]
            )
            avg_recent_ic = np.mean([abs(v) for v in recent_ic.values()]) if recent_ic else 0
            avg_full_ic = np.mean([abs(v) for v in ic_ranks.values()]) if ic_ranks else 0
            alpha_decay = max(0, avg_full_ic - avg_recent_ic) if avg_full_ic > 0 else 0

            # ?룷吏??뀡 諛곗닔
            if ensemble >= 0.65 and agreement >= 0.3:
                multiplier = 1.3
            elif ensemble >= 0.5:
                multiplier = 1.0
            elif ensemble >= 0.35:
                multiplier = 0.7
            else:
                multiplier = 0.5

            # ?븣?뙆 媛먯뇿媛? ?겕硫? 異뺤냼
            if alpha_decay > 0.1:
                multiplier *= 0.8

            multiplier = float(np.clip(multiplier, 0.3, 1.5))

            # ?떊猶곕룄 議곗젙
            conf_adj = (ensemble - 0.5) * 0.5 + agreement * 0.1
            conf_adj = float(np.clip(conf_adj, -0.3, 0.3))

            # ?꽕紐?
            notes = []
            if ensemble >= 0.65:
                notes.append(f"?븰?긽釉? 媛뺤꽭({ensemble:.2f})")
            elif ensemble <= 0.35:
                notes.append(f"?븰?긽釉? ?빟?꽭({ensemble:.2f})")
            if agreement >= 0.5:
                notes.append(f"?뙥?꽣 ?쓽寃? ?씪移?({agreement:.2f})")
            if alpha_decay > 0.1:
                notes.append(f"?븣?뙆 媛먯뇿 二쇱쓽({alpha_decay:.2f})")
            if top_factors:
                best = list(top_factors.keys())[0]
                notes.append(f"理쒓컯 ?뙥?꽣: {best}(IC={top_factors[best]:.3f})")

            return MLAlphaSignal(
                ensemble_score=round(ensemble, 4),
                top_factors=top_factors,
                sentiment_proxy=round(sent, 4),
                factor_agreement=round(agreement, 4),
                alpha_decay=round(alpha_decay, 4),
                position_multiplier=round(multiplier, 2),
                confidence_adjustment=round(conf_adj, 4),
                note=" | ".join(notes) if notes else "?젙?긽 踰붿쐞",
            )

        except Exception as e:
            logger.warning(f"ML Alpha 遺꾩꽍 ?삤瑜?: {e}")
            return MLAlphaSignal(
                ensemble_score=0.5,
                position_multiplier=1.0,
                confidence_adjustment=0.0,
                note=f"遺꾩꽍 ?삤瑜?: {e}",
            )
