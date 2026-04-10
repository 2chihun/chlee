"""Marcos L?pez de Prado - Advances in Financial Machine Learning (AFML)

AFML ?빑?떖 湲곕쾿?쓣 湲곗닠?쟻 ?봽濡앹떆濡? 援ы쁽:
1. ?듃由ы뵆 諛곕━?뼱 ?씪踰⑤쭅 (Triple Barrier Method)
2. CUSUM ?븘?꽣 (?씠踰ㅽ듃 ?깦?뵆留?)
3. 遺꾩닔李⑤텇 (Fractional Differentiation) - ?젙?긽?꽦 ?솗蹂?
4. 硫뷀???씪踰⑤쭅 (Meta-Labeling) - 湲곗〈 ?떊?샇 ?뭹吏? ?룊媛?
5. ?닚李⑥쟻 遺??듃?뒪?듃?옪 (Sequential Bootstrap) - 鍮꾩쨷蹂? ?깦?뵆

湲곗〈 ?떊?샇 ?넂 硫뷀???씪踰⑤쭅?쑝濡? ?뭹吏? 寃?利? ?넂 理쒖쥌 ?떊猶곕룄 議곗젙
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class BarrierType(Enum):
    """?듃由ы뵆 諛곕━?뼱 寃곌낵 ????엯"""
    UPPER = "upper"      # ?닔?씡 ?떎?쁽
    LOWER = "lower"      # ?넀?젅
    VERTICAL = "vertical"  # ?떆媛? 留뚮즺


@dataclass
class MetaLabelSignal:
    """AFML 硫뷀???씪踰⑤쭅 ?떊?샇"""
    # 硫뷀???씪踰? ?솗瑜? (0~1, 湲곗〈 ?떊?샇媛? ?닔?씡?쑝濡? ?씠?뼱吏? ?솗瑜?)
    meta_probability: float
    # ?듃由ы뵆 諛곕━?뼱 湲곕컲 湲곕?? ?닔?씡瑜?
    expected_return: float
    # CUSUM ?씠踰ㅽ듃 ?뿬遺? (True硫? 援ъ“?쟻 蹂??솕 媛먯??)
    cusum_event: bool
    # 遺꾩닔李⑤텇 d媛? (?젙?긽?꽦 理쒖냼 李⑤텇)
    frac_diff_d: float
    # ?닚李⑥쟻 遺??듃?뒪?듃?옪 ?쑀?땲?겕?땲?뒪 (0~1, ?젙蹂? ?룆由쎌꽦)
    uniqueness: float
    # ?룷吏??뀡 諛곗닔 (0.3~1.5)
    position_multiplier: float
    # ?떊猶곕룄 議곗젙媛? (-0.3~+0.3)
    confidence_adjustment: float
    # ?꽕紐?
    note: str = ""


class CUSUMFilter:
    """CUSUM ?씠踰ㅽ듃 ?븘?꽣

    ?늻?쟻?빀 ?젣?뼱 李⑦듃 湲곕쾿?쑝濡? 援ъ“?쟻 蹂??솕(?룊洹? ?씠?룞)瑜? 媛먯??.
    媛?寃? ?닔?씡瑜좎씠 ?엫怨꾧컪?쓣 珥덇낵?븯?뒗 ?씠踰ㅽ듃留? ?깦?뵆留?.
    """

    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold

    def detect_events(self, prices: pd.Series) -> pd.Series:
        """CUSUM ?씠踰ㅽ듃 媛먯??

        Args:
            prices: 醫낃?? ?떆由ъ쫰

        Returns:
            ?씠踰ㅽ듃 諛쒖깮 ?뿬遺? (bool Series)
        """
        if len(prices) < 10:
            return pd.Series(False, index=prices.index)

        log_returns = np.log(prices / prices.shift(1)).dropna()
        events = pd.Series(False, index=prices.index)

        s_pos = 0.0
        s_neg = 0.0

        for i in range(len(log_returns)):
            idx = log_returns.index[i]
            ret = log_returns.iloc[i]

            s_pos = max(0, s_pos + ret - self.threshold)
            s_neg = min(0, s_neg + ret + self.threshold)

            if s_pos > self.threshold:
                events.loc[idx] = True
                s_pos = 0.0
            elif s_neg < -self.threshold:
                events.loc[idx] = True
                s_neg = 0.0

        return events

    def is_current_event(self, prices: pd.Series) -> bool:
        """?쁽?옱 ?떆?젏?씠 CUSUM ?씠踰ㅽ듃?씤吏? ?솗?씤"""
        events = self.detect_events(prices)
        if events.empty:
            return False
        return bool(events.iloc[-1])


class TripleBarrierLabeler:
    """?듃由ы뵆 諛곕━?뼱 ?씪踰⑤쭅

    ?꽭 媛?吏? 諛곕━?뼱 以? 癒쇱?? ?룄?떖?븯?뒗 寃껋쑝濡? ?씪踰? 寃곗젙:
    - ?긽?떒 諛곕━?뼱: ?닔?씡 ?떎?쁽 (take_profit)
    - ?븯?떒 諛곕━?뼱: ?넀?젅 (stop_loss)
    - ?닔吏? 諛곕━?뼱: 蹂댁쑀 湲곌컙 留뚮즺 (max_holding_period)
    """

    def __init__(
        self,
        take_profit_pct: float = 0.02,
        stop_loss_pct: float = 0.01,
        max_holding_bars: int = 20,
    ):
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_bars = max_holding_bars

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """?쟾泥? ?뜲?씠?꽣?뿉 ?듃由ы뵆 諛곕━?뼱 ?씪踰? ?쟻?슜

        Args:
            df: 'close' 而щ읆 ?븘?슂

        Returns:
            'tb_label' (-1, 0, 1), 'tb_barrier' (?뼱?뒓 諛곕━?뼱), 'tb_return' 而щ읆 異붽??
        """
        result = df.copy()
        result["tb_label"] = 0
        result["tb_barrier"] = "vertical"
        result["tb_return"] = 0.0

        closes = result["close"].values

        for i in range(len(closes) - 1):
            entry_price = closes[i]
            if entry_price <= 0:
                continue

            upper = entry_price * (1 + self.take_profit_pct)
            lower = entry_price * (1 - self.stop_loss_pct)
            end_bar = min(i + self.max_holding_bars, len(closes) - 1)

            barrier_type = BarrierType.VERTICAL
            exit_price = closes[end_bar]

            for j in range(i + 1, end_bar + 1):
                if closes[j] >= upper:
                    barrier_type = BarrierType.UPPER
                    exit_price = closes[j]
                    break
                elif closes[j] <= lower:
                    barrier_type = BarrierType.LOWER
                    exit_price = closes[j]
                    break

            ret = (exit_price - entry_price) / entry_price

            if barrier_type == BarrierType.UPPER:
                result.iloc[i, result.columns.get_loc("tb_label")] = 1
            elif barrier_type == BarrierType.LOWER:
                result.iloc[i, result.columns.get_loc("tb_label")] = -1
            else:
                result.iloc[i, result.columns.get_loc("tb_label")] = 1 if ret > 0 else (-1 if ret < 0 else 0)

            result.iloc[i, result.columns.get_loc("tb_barrier")] = barrier_type.value
            result.iloc[i, result.columns.get_loc("tb_return")] = ret

        return result

    def get_current_expected_return(self, df: pd.DataFrame, lookback: int = 60) -> float:
        """理쒓렐 N嫄댁쓽 ?듃由ы뵆 諛곕━?뼱 寃곌낵?뿉?꽌 湲곕?? ?닔?씡瑜? ?궛異?"""
        labeled = self.apply(df.tail(lookback + self.max_holding_bars))
        recent = labeled.tail(lookback)
        if recent["tb_return"].std() == 0:
            return 0.0
        return float(recent["tb_return"].mean())


class FractionalDifferentiator:
    """遺꾩닔李⑤텇 (Fractional Differentiation)

    ?떆怨꾩뿴?쓽 硫붾え由?(?옣湲곗쓽議댁꽦)瑜? 理쒕???븳 蹂댁〈?븯硫댁꽌
    ?젙?긽?꽦?쓣 ?솗蹂댄븯?뒗 理쒖냼 李⑤텇 李⑥닔 d瑜? 李얜뒗?떎.
    d=0: ?썝蹂? (鍮꾩젙?긽), d=1: 1李? 李⑤텇 (?젙?긽?씠吏?留? 硫붾え由? ?넀?떎)
    理쒖쟻 d: ADF 寃??젙 ?넻怨쇳븯?뒗 理쒖냼 d
    """

    def __init__(self, max_d: float = 1.0, step: float = 0.1, threshold: float = 0.01):
        self.max_d = max_d
        self.step = step
        self.threshold = threshold

    @staticmethod
    def _get_weights(d: float, size: int) -> np.ndarray:
        """遺꾩닔李⑤텇 媛?以묒튂 怨꾩궛 (binomial series expansion)"""
        w = [1.0]
        for k in range(1, size):
            w.append(-w[-1] * (d - k + 1) / k)
        return np.array(w[::-1])

    def frac_diff(self, series: pd.Series, d: float, cutoff: float = 1e-5) -> pd.Series:
        """遺꾩닔李⑤텇 ?쟻?슜"""
        weights = self._get_weights(d, len(series))
        # 媛?以묒튂媛? cutoff ?씠?븯?씤 寃껋?? ?젣嫄? (硫붾え由? ?젅?빟)
        mask = np.abs(weights) > cutoff
        weights = weights[mask]
        width = len(weights)

        result = pd.Series(index=series.index, dtype=float)
        for i in range(width - 1, len(series)):
            result.iloc[i] = np.dot(weights, series.values[i - width + 1: i + 1])
        return result.dropna()

    def find_min_d(self, series: pd.Series) -> float:
        """ADF 寃??젙 湲곕컲 理쒖냼 ?젙?긽?꽦 d媛? ?깘?깋

        ?떎?젣 ADF 寃??젙 ????떊, ?옄湲곗긽愿? 媛먯냼瑜? ?봽濡앹떆濡? ?궗?슜.
        """
        if len(series) < 50:
            return 0.5

        for d in np.arange(self.step, self.max_d + self.step, self.step):
            diffed = self.frac_diff(series, d)
            if len(diffed) < 20:
                continue

            # ?옄湲곗긽愿? ?봽濡앹떆: lag-1 ?긽愿??씠 0.5 ?씠?븯硫? 異⑸텇?엳 ?젙?긽
            autocorr = diffed.autocorr(lag=1)
            if autocorr is not None and abs(autocorr) < 0.5:
                return round(d, 2)

        return round(self.max_d, 2)


class SequentialBootstrap:
    """?닚李⑥쟻 遺??듃?뒪?듃?옪

    ?몴蹂몄쓽 ?쑀?땲?겕?땲?뒪(?룆由쎌꽦)瑜? 怨꾩궛?븯?뿬
    ?젙蹂? 以묐났?씠 ?쟻??? 愿?痢≪튂?뿉 ?뜑 ?넂??? 媛?以묒튂瑜? 遺??뿬.
    """

    @staticmethod
    def get_indicator_matrix(labels: pd.Series, span: int = 5) -> pd.DataFrame:
        """吏??몴 ?뻾?젹 ?깮?꽦

        媛? ?씪踰⑥씠 ?쁺?뼢諛쏅뒗 湲곌컙(span)?쓣 ?씤?뵒耳??씠?꽣 ?뻾?젹濡? ?몴?쁽.
        """
        n = len(labels)
        ind_matrix = pd.DataFrame(0, index=labels.index, columns=range(n))

        for i in range(n):
            start = i
            end = min(i + span, n)
            for j in range(start, end):
                ind_matrix.iloc[j, i] = 1

        return ind_matrix

    @staticmethod
    def get_uniqueness(ind_matrix: pd.DataFrame) -> pd.Series:
        """媛? ?깦?뵆?쓽 ?룊洹? ?쑀?땲?겕?땲?뒪 怨꾩궛"""
        concurrency = ind_matrix.sum(axis=1)
        uniqueness = pd.Series(0.0, index=ind_matrix.columns)

        for i in ind_matrix.columns:
            active = ind_matrix[i] > 0
            if active.sum() == 0:
                uniqueness[i] = 0
            else:
                uniqueness[i] = (1.0 / concurrency[active]).mean()

        return uniqueness

    def compute_avg_uniqueness(self, df: pd.DataFrame, span: int = 5) -> float:
        """?뜲?씠?꽣?뀑?쓽 ?룊洹? ?쑀?땲?겕?땲?뒪 怨꾩궛"""
        if len(df) < span:
            return 0.5

        labels = pd.Series(range(len(df)), index=df.index)
        sample_size = min(len(df), 100)  # ?꽦?뒫?쓣 ?쐞?빐 ?젣?븳
        labels = labels.iloc[:sample_size]

        ind_matrix = self.get_indicator_matrix(labels, span=span)
        uniqueness = self.get_uniqueness(ind_matrix)

        return float(uniqueness.mean())


class MetaLabeler:
    """硫뷀???씪踰⑤쭅

    1李? 紐⑤뜽(湲곗〈 ?쟾?왂)?쓽 留ㅻℓ ?떊?샇(諛⑺뼢)?뒗 ?쑀吏??븯怨?,
    2李? 紐⑤뜽(硫뷀???씪踰?)?씠 ?빐?떦 ?떊?샇?쓽 ?꽦怨? ?솗瑜좎쓣 ?룊媛?.
    ?꽦怨? ?솗瑜좎씠 ?궙??? ?떊?샇?뒗 ?궗?씠利덈?? 異뺤냼?븯嫄곕굹 臾댁떆.
    """

    def __init__(self, lookback: int = 120, min_samples: int = 30):
        self.lookback = lookback
        self.min_samples = min_samples

    def evaluate_signal_quality(self, df: pd.DataFrame, signal_col: str = "signal") -> float:
        """湲곗〈 ?떊?샇?쓽 怨쇨굅 ?꽦怨듬쪧 湲곕컲 硫뷀???씪踰? ?솗瑜?

        Args:
            df: 'close', signal_col(-1,0,1) 而щ읆 ?븘?슂
            signal_col: 湲곗〈 ?떊?샇 而щ읆紐?

        Returns:
            0~1 硫뷀???씪踰? ?솗瑜? (?넂?쓣?닔濡? ?떊?샇 ?뭹吏? ?슦?닔)
        """
        if signal_col not in df.columns or len(df) < self.min_samples:
            return 0.5  # ?뜲?씠?꽣 遺?議? ?떆 以묐┰

        recent = df.tail(self.lookback).copy()

        # ?떊?샇 諛쒖깮 ?떆?젏留? 異붿텧
        signals = recent[recent[signal_col] != 0].copy()
        if len(signals) < 5:
            return 0.5

        # 媛? ?떊?샇 ?썑 forward return 怨꾩궛
        returns = recent["close"].pct_change().shift(-1)
        signal_returns = returns.loc[signals.index]

        # ?떊?샇 諛⑺뼢怨? ?닔?씡 諛⑺뼢 ?씪移? 鍮꾩쑉
        correct = (signals[signal_col] * signal_returns > 0).sum()
        total = signal_returns.notna().sum()

        if total == 0:
            return 0.5

        hit_rate = correct / total

        # ?닔?씡?쓽 ?겕湲곕룄 怨좊젮 (?젅??? ?닔?씡瑜? ?룊洹?)
        avg_win = signal_returns[signals[signal_col] * signal_returns > 0].abs().mean()
        avg_loss = signal_returns[signals[signal_col] * signal_returns <= 0].abs().mean()

        if pd.isna(avg_win):
            avg_win = 0.0
        if pd.isna(avg_loss) or avg_loss == 0:
            profit_factor = 1.0
        else:
            profit_factor = min(avg_win / avg_loss, 3.0) / 3.0  # 0~1 ?젙洹쒗솕

        # 硫뷀???씪踰? ?솗瑜? = ?쟻以묐쪧 60% + ?닔?씡?슂?씤 40%
        meta_prob = hit_rate * 0.6 + profit_factor * 0.4
        return float(np.clip(meta_prob, 0.0, 1.0))


class AFMLAnalyzer:
    """AFML ?넻?빀 遺꾩꽍湲?

    CUSUM ?븘?꽣 + ?듃由ы뵆 諛곕━?뼱 + 遺꾩닔李⑤텇 + 硫뷀???씪踰⑤쭅 + ?닚李⑥쟻 遺??듃?뒪?듃?옪
    紐⑤뱺 湲곕쾿?쓣 議고빀?븯?뿬 理쒖쥌 ?떊?샇 ?뭹吏? ?룊媛?
    """

    def __init__(
        self,
        cusum_threshold: float = 0.02,
        tp_pct: float = 0.02,
        sl_pct: float = 0.01,
        max_holding: int = 20,
        lookback: int = 120,
    ):
        self.cusum = CUSUMFilter(threshold=cusum_threshold)
        self.barrier = TripleBarrierLabeler(
            take_profit_pct=tp_pct,
            stop_loss_pct=sl_pct,
            max_holding_bars=max_holding,
        )
        self.frac_diff = FractionalDifferentiator()
        self.bootstrap = SequentialBootstrap()
        self.meta = MetaLabeler(lookback=lookback)
        self.lookback = lookback

    def analyze(self, df: pd.DataFrame) -> MetaLabelSignal:
        """AFML 醫낇빀 遺꾩꽍

        Args:
            df: OHLCV DataFrame ('open','high','low','close','volume')

        Returns:
            MetaLabelSignal
        """
        if len(df) < 50:
            return MetaLabelSignal(
                meta_probability=0.5,
                expected_return=0.0,
                cusum_event=False,
                frac_diff_d=0.5,
                uniqueness=0.5,
                position_multiplier=1.0,
                confidence_adjustment=0.0,
                note="?뜲?씠?꽣 遺?議? (理쒖냼 50遊? ?븘?슂)",
            )

        try:
            # 1. CUSUM ?씠踰ㅽ듃 媛먯??
            cusum_event = self.cusum.is_current_event(df["close"])

            # 2. ?듃由ы뵆 諛곕━?뼱 湲곕?? ?닔?씡瑜?
            expected_return = self.barrier.get_current_expected_return(df, self.lookback)

            # 3. 遺꾩닔李⑤텇 理쒖쟻 d
            frac_d = self.frac_diff.find_min_d(df["close"])

            # 4. ?닚李⑥쟻 遺??듃?뒪?듃?옪 ?쑀?땲?겕?땲?뒪
            uniqueness = self.bootstrap.compute_avg_uniqueness(df, span=5)

            # 5. 硫뷀???씪踰⑤쭅 (湲곗〈 ?떊?샇媛? ?엳?쑝硫? ?룊媛?, ?뾾?쑝硫? ?옄泥? RSI 湲곕컲)
            temp_df = df.copy()
            if "signal" not in temp_df.columns:
                # RSI 湲곕컲 媛꾩씠 ?떊?샇 ?깮?꽦
                delta = temp_df["close"].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss.replace(0, np.nan)
                rsi = 100 - (100 / (1 + rs))
                temp_df["signal"] = 0
                temp_df.loc[rsi < 30, "signal"] = 1
                temp_df.loc[rsi > 70, "signal"] = -1

            meta_prob = self.meta.evaluate_signal_quality(temp_df, "signal")

            # 醫낇빀 ?젏?닔 怨꾩궛
            # CUSUM ?씠踰ㅽ듃 + ?넂??? 硫뷀???솗瑜? + ?뼇?쓽 湲곕???닔?씡 = 媛뺥븳 ?떊?샇
            score = meta_prob * 0.40 + min(max(expected_return * 10 + 0.5, 0), 1) * 0.25
            score += uniqueness * 0.15 + (0.1 if cusum_event else 0.0)
            # 遺꾩닔李⑤텇 d媛? ?궙?쓣?닔濡? 硫붾え由? 蹂댁〈 (醫뗭?? ?떊?샇)
            memory_quality = max(0, 1.0 - frac_d) * 0.10
            score += memory_quality

            score = float(np.clip(score, 0.0, 1.0))

            # ?룷吏??뀡 諛곗닔 寃곗젙
            if score >= 0.7:
                multiplier = 1.3
            elif score >= 0.5:
                multiplier = 1.0
            elif score >= 0.3:
                multiplier = 0.7
            else:
                multiplier = 0.4

            # CUSUM ?씠踰ㅽ듃硫? 蹂댁닔 (援ъ“?쟻 蹂??솕 ?떆 ?떊以?)
            if cusum_event and expected_return < 0:
                multiplier *= 0.7

            multiplier = float(np.clip(multiplier, 0.3, 1.5))

            # ?떊猶곕룄 議곗젙
            confidence_adj = (score - 0.5) * 0.6  # -0.3 ~ +0.3
            confidence_adj = float(np.clip(confidence_adj, -0.3, 0.3))

            # ?꽕紐? ?깮?꽦
            notes = []
            if cusum_event:
                notes.append("CUSUM 援ъ“ 蹂??솕 媛먯??")
            if meta_prob >= 0.7:
                notes.append(f"硫뷀???씪踰? ?슦?닔({meta_prob:.2f})")
            elif meta_prob <= 0.3:
                notes.append(f"硫뷀???씪踰? 遺덈웾({meta_prob:.2f})")
            if expected_return > 0.01:
                notes.append(f"湲곕???닔?씡 ?뼇?샇({expected_return:.3f})")
            elif expected_return < -0.01:
                notes.append(f"湲곕???닔?씡 遺??젙({expected_return:.3f})")
            if frac_d <= 0.4:
                notes.append(f"硫붾え由? 蹂댁〈 ?슦?닔(d={frac_d:.1f})")

            return MetaLabelSignal(
                meta_probability=round(meta_prob, 4),
                expected_return=round(expected_return, 4),
                cusum_event=cusum_event,
                frac_diff_d=round(frac_d, 2),
                uniqueness=round(uniqueness, 4),
                position_multiplier=round(multiplier, 2),
                confidence_adjustment=round(confidence_adj, 4),
                note=" | ".join(notes) if notes else "?젙?긽 踰붿쐞",
            )

        except Exception as e:
            logger.warning(f"AFML 遺꾩꽍 ?삤瑜?: {e}")
            return MetaLabelSignal(
                meta_probability=0.5,
                expected_return=0.0,
                cusum_event=False,
                frac_diff_d=0.5,
                uniqueness=0.5,
                position_multiplier=1.0,
                confidence_adjustment=0.0,
                note=f"遺꾩꽍 ?삤瑜?: {e}",
            )
