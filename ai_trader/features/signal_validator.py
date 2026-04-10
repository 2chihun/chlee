"""David Aronson - Evidence-Based Technical Analysis

?떊?샇 ?넻怨꾧??利? 紐⑤뱢:
1. ?떊?샇 ?쟻以묐쪧 ?넻怨꾩쟻 ?쑀?쓽?꽦 寃??젙 (?씠?빆 寃??젙 ?봽濡앹떆)
2. ?떎以묎???젙 蹂댁젙 (Bonferroni/FDR)
3. ?뜲?씠?꽣留덉씠?떇 ?렪?뼢 ?젙?웾?솕
4. 遺??듃?뒪?듃?옪 ?떊猶곌뎄媛?
5. ?떊?샇 媛먯뇿 遺꾩꽍 (Decay Analysis)
6. 理쒖냼 ?몴蹂몄닔 ?궛異?

湲곗〈 42媛? ?떊?샇?쓽 ?넻怨꾩쟻 ?쑀?쓽?꽦?쓣 ?궗?썑 寃?利앺븯?뿬
怨쇱쟻?빀?맂 ?떊?샇瑜? ?븘?꽣留?.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ValidationResult:
    """媛쒕퀎 ?떊?샇 寃?利? 寃곌낵"""
    signal_name: str
    hit_rate: float           # ?쟻以묐쪧
    sample_size: int          # ?몴蹂? ?닔
    is_significant: bool      # ?넻怨꾩쟻 ?쑀?쓽 ?뿬遺?
    p_value: float            # p-媛?
    z_score: float            # z-?넻怨꾨웾
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    note: str = ""


@dataclass
class SignalValidationSignal:
    """?떊?샇寃?利? 醫낇빀 ?떊?샇"""
    # ?쑀?쓽?븳 ?떊?샇 鍮꾩쑉 (0~1)
    valid_signal_ratio: float
    # ?뜲?씠?꽣留덉씠?떇 ?렪?뼢 ?젏?닔 (0~1, ?넂?쓣?닔濡? ?렪?뼢 ?떖媛?)
    data_mining_bias: float
    # ?쟾泥? ?떊?샇 ?닔
    total_signals_tested: int
    # ?쑀?쓽?븳 ?떊?샇 ?닔
    significant_signals: int
    # 媛??옣 媛뺥븳 ?떊?샇??? ?빟?븳 ?떊?샇
    strongest_signal: str = ""
    weakest_signal: str = ""
    # ?룊洹? ?떊?샇 媛먯뇿?쑉 (0~1)
    avg_decay_rate: float = 0.0
    # 理쒖냼 沅뚯옣 ?몴蹂몄닔
    min_sample_required: int = 30
    # ?룷吏??뀡 諛곗닔 (0.3~1.5)
    position_multiplier: float = 1.0
    # ?떊猶곕룄 議곗젙 (-0.3~+0.3)
    confidence_adjustment: float = 0.0
    # 媛쒕퀎 寃?利? 寃곌낵
    validation_details: List[ValidationResult] = field(default_factory=list)
    # ?꽕紐?
    note: str = ""


class BinomialTest:
    """?씠?빆 寃??젙 ?봽濡앹떆

    媛??꽕: ?떊?샇 ?쟻以묐쪧?씠 50%(臾댁옉?쐞)蹂대떎 ?쑀?쓽?븯寃? ?넂???媛??
    H0: p = 0.5 (臾댁옉?쐞)
    H1: p > 0.5 (?삁痢〓젰 ?엳?쓬)
    """

    @staticmethod
    def test(successes: int, trials: int, null_prob: float = 0.5) -> Tuple[float, float, bool]:
        """?씠?빆 寃??젙

        Returns:
            (z_score, p_value, is_significant_at_5pct)
        """
        if trials < 10:
            return 0.0, 1.0, False

        observed_rate = successes / trials
        se = np.sqrt(null_prob * (1 - null_prob) / trials)

        if se == 0:
            return 0.0, 1.0, False

        z = (observed_rate - null_prob) / se

        # ?젙洹? 洹쇱궗 p-媛? (?떒痢?)
        # 過(-z) 洹쇱궗: Abramowitz & Stegun
        p_value = BinomialTest._norm_sf(z)

        is_significant = p_value < 0.05
        return float(z), float(p_value), is_significant

    @staticmethod
    def _norm_sf(z: float) -> float:
        """?몴以??젙洹쒕텇?룷 ?깮議댄븿?닔 (1 - CDF) 洹쇱궗"""
        if z < -8:
            return 1.0
        if z > 8:
            return 0.0

        # Horner form approximation
        t = 1.0 / (1.0 + 0.2316419 * abs(z))
        d = 0.3989422804014327  # 1/sqrt(2*pi)
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        pdf = d * np.exp(-0.5 * z * z)
        sf = pdf * poly

        if z > 0:
            return sf
        return 1.0 - sf


class MultipleTesting:
    """?떎以묎???젙 蹂댁젙

    留롮?? ?떊?샇瑜? ?룞?떆?뿉 寃??젙?븯硫? ?슦?뿰?엳 ?쑀?쓽?븳 寃곌낵媛? ?굹?삱 ?닔 ?엳?쓬.
    Bonferroni 蹂댁젙 (蹂댁닔?쟻) 諛? BH-FDR 蹂댁젙 (?쁽?떎?쟻).
    """

    @staticmethod
    def bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
        """Bonferroni 蹂댁젙: ?엫怨꾧컪 = 慣/m"""
        m = len(p_values)
        if m == 0:
            return []
        threshold = alpha / m
        return [p < threshold for p in p_values]

    @staticmethod
    def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> List[bool]:
        """BH-FDR 蹂댁젙: ?뜑 ?쁽?떎?쟻?씤 ?떎以묎???젙 蹂댁젙"""
        m = len(p_values)
        if m == 0:
            return []

        # p-媛? ?젙?젹 (?씤?뜳?뒪 蹂댁〈)
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        results = [False] * m

        max_k = 0
        for rank, (orig_idx, p) in enumerate(indexed, 1):
            threshold = rank * alpha / m
            if p <= threshold:
                max_k = rank

        # max_k ?씠?븯 ?닚?쐞?쓽 媛??꽕 紐⑤몢 湲곌컖
        for rank, (orig_idx, p) in enumerate(indexed, 1):
            if rank <= max_k:
                results[orig_idx] = True

        return results


class DataMiningBiasEstimator:
    """?뜲?씠?꽣留덉씠?떇 ?렪?뼢 ?젙?웾?솕

    留롮?? ?쟾?왂?쓣 ?떆?룄?븷?닔濡? ?슦?뿰?엳 醫뗭?? 寃곌낵瑜? 李얠쓣 ?솗瑜? 利앷??.
    deflated Sharpe Ratio 媛쒕뀗?쓽 媛꾨떒?븳 ?봽濡앹떆.
    """

    @staticmethod
    def estimate_bias(num_trials: int, best_sharpe: float, avg_sharpe: float = 0.0) -> float:
        """?뜲?씠?꽣留덉씠?떇 ?렪?뼢 ?젏?닔 (0~1)

        Args:
            num_trials: ?떆?룄?븳 ?쟾?왂 ?닔
            best_sharpe: 理쒓퀬 Sharpe ratio
            avg_sharpe: ?룊洹? Sharpe ratio (null hypothesis)

        Returns:
            0~1 ?렪?뼢 ?젏?닔 (?넂?쓣?닔濡? ?렪?뼢 ?떖?븿)
        """
        if num_trials <= 1:
            return 0.0

        # 湲곕?? 理쒕?? Sharpe (留롮씠 ?떆?룄?븷?닔濡? ?넂?쓬)
        # E[max(Z)] ? ?닖(2 * ln(N)) for N trials
        expected_max = np.sqrt(2 * np.log(num_trials))

        # 愿?痢〓맂 Sharpe媛? 湲곕?? 理쒕?? ?씠?븯硫? ?렪?뼢 ?쓽?떖
        if expected_max == 0:
            return 0.0

        deflation_ratio = best_sharpe / (expected_max + 1e-10)
        # 鍮꾩쑉?씠 1 ?씠?븯硫? ?슦?뿰 媛??뒫, 2 ?씠?긽?씠硫? 吏꾩쭨 ?븣?뙆 媛??뒫
        bias = 1.0 - float(np.clip(deflation_ratio / 2, 0, 1))
        return float(np.clip(bias, 0.0, 1.0))


class BootstrapConfidenceInterval:
    """遺??듃?뒪?듃?옪 ?떊猶곌뎄媛?

    遺꾪룷 媛??젙 ?뾾?씠 ?쟻以묐쪧?쓽 ?떊猶곌뎄媛꾩쓣 異붿젙.
    """

    @staticmethod
    def compute(
        successes: int, trials: int, n_bootstrap: int = 1000, alpha: float = 0.05
    ) -> Tuple[float, float]:
        """遺??듃?뒪?듃?옪 95% ?떊猶곌뎄媛?"""
        if trials < 10:
            return 0.0, 1.0

        rng = np.random.default_rng(42)
        hit_rate = successes / trials

        # 踰좊Ⅴ?늻?씠 遺??듃?뒪?듃?옪
        bootstrap_rates = []
        for _ in range(n_bootstrap):
            sample = rng.binomial(1, hit_rate, size=trials)
            bootstrap_rates.append(sample.mean())

        lower = float(np.percentile(bootstrap_rates, alpha / 2 * 100))
        upper = float(np.percentile(bootstrap_rates, (1 - alpha / 2) * 100))
        return lower, upper


class SignalDecayAnalyzer:
    """?떊?샇 媛먯뇿 遺꾩꽍

    ?떊?샇 諛쒖깮 ?썑 ?떆媛꾩씠 吏??궓?뿉 ?뵲瑜? ?삁痢〓젰 媛먯뇿瑜? 痢≪젙.
    """

    @staticmethod
    def measure_decay(
        returns_after_signal: pd.DataFrame, max_lag: int = 20
    ) -> float:
        """?떊?샇 媛먯뇿?쑉 (0~1, ?넂?쓣?닔濡? 鍮좊Ⅴ寃? 媛먯뇿)"""
        if len(returns_after_signal) < max_lag:
            return 0.5

        # 媛? lag?뿉?꽌?쓽 ?쟻以묐쪧 蹂??솕
        hit_rates = []
        for lag in range(1, min(max_lag + 1, len(returns_after_signal))):
            if f"ret_{lag}" in returns_after_signal.columns:
                hr = (returns_after_signal[f"ret_{lag}"] > 0).mean()
            else:
                # forward return 吏곸젒 怨꾩궛
                hr = 0.5
            hit_rates.append(hr)

        if len(hit_rates) < 2:
            return 0.5

        # ?쟻以묐쪧?쓽 媛먯냼 ?냽?룄
        hit_arr = np.array(hit_rates)
        diffs = np.diff(hit_arr)
        avg_decline = -np.mean(diffs)  # ?뼇?닔硫? 媛먯뇿

        return float(np.clip(avg_decline * 10, 0.0, 1.0))


class SignalValidator:
    """?떊?샇 ?넻怨꾧??利? ?넻?빀湲?"""

    def __init__(self, min_samples: int = 30, alpha: float = 0.05):
        self.min_samples = min_samples
        self.alpha = alpha
        self.binom = BinomialTest()
        self.multi = MultipleTesting()
        self.bias = DataMiningBiasEstimator()
        self.bootstrap = BootstrapConfidenceInterval()
        self.decay = SignalDecayAnalyzer()

    def validate_signal(
        self, signal_name: str, successes: int, trials: int
    ) -> ValidationResult:
        """媛쒕퀎 ?떊?샇 寃?利?"""
        if trials < self.min_samples:
            return ValidationResult(
                signal_name=signal_name,
                hit_rate=successes / trials if trials > 0 else 0.0,
                sample_size=trials,
                is_significant=False,
                p_value=1.0,
                z_score=0.0,
                note=f"?몴蹂? 遺?議? (理쒖냼 {self.min_samples}嫄? ?븘?슂, ?쁽?옱 {trials}嫄?)",
            )

        z, p, sig = self.binom.test(successes, trials)
        ci = self.bootstrap.compute(successes, trials)
        hit_rate = successes / trials

        return ValidationResult(
            signal_name=signal_name,
            hit_rate=round(hit_rate, 4),
            sample_size=trials,
            is_significant=sig,
            p_value=round(p, 6),
            z_score=round(z, 4),
            confidence_interval=ci,
        )

    def validate_all(
        self,
        signal_results: Dict[str, Tuple[int, int]],
        num_strategies_tried: int = 1,
    ) -> SignalValidationSignal:
        """蹂듭닔 ?떊?샇 醫낇빀 寃?利?

        Args:
            signal_results: {signal_name: (successes, trials)}
            num_strategies_tried: ?떆?룄?븳 ?쟾?왂 ?닔 (?렪?뼢 異붿젙?슜)
        """
        if not signal_results:
            return SignalValidationSignal(
                valid_signal_ratio=0.0,
                data_mining_bias=0.0,
                total_signals_tested=0,
                significant_signals=0,
                position_multiplier=1.0,
                confidence_adjustment=0.0,
                note="寃?利앺븷 ?떊?샇 ?뾾?쓬",
            )

        # 1. 媛쒕퀎 寃?利?
        results = []
        p_values = []
        for name, (succ, trials) in signal_results.items():
            res = self.validate_signal(name, succ, trials)
            results.append(res)
            p_values.append(res.p_value)

        # 2. ?떎以묎???젙 蹂댁젙 (BH-FDR)
        bh_significant = self.multi.benjamini_hochberg(p_values, self.alpha)
        for i, res in enumerate(results):
            res.is_significant = bh_significant[i]

        # 3. ?넻怨?
        sig_count = sum(r.is_significant for r in results)
        total = len(results)
        valid_ratio = sig_count / total if total > 0 else 0.0

        # 4. ?뜲?씠?꽣留덉씠?떇 ?렪?뼢
        hit_rates = [r.hit_rate for r in results if r.sample_size >= self.min_samples]
        if hit_rates:
            best = max(hit_rates)
            avg = np.mean(hit_rates)
            # Sharpe ?봽濡앹떆: (hit_rate - 0.5) * sqrt(N)
            best_sharpe = (best - 0.5) * np.sqrt(max(r.sample_size for r in results))
            dm_bias = self.bias.estimate_bias(num_strategies_tried, best_sharpe)
        else:
            dm_bias = 0.0

        # 5. 媛??옣 媛뺥븳/?빟?븳 ?떊?샇
        if results:
            sorted_results = sorted(results, key=lambda r: r.z_score, reverse=True)
            strongest = sorted_results[0].signal_name
            weakest = sorted_results[-1].signal_name
        else:
            strongest = ""
            weakest = ""

        # ?룷吏??뀡 諛곗닔
        if valid_ratio >= 0.7 and dm_bias <= 0.3:
            multiplier = 1.3
        elif valid_ratio >= 0.5:
            multiplier = 1.0
        elif valid_ratio >= 0.3:
            multiplier = 0.8
        else:
            multiplier = 0.5

        if dm_bias >= 0.6:
            multiplier *= 0.7

        multiplier = float(np.clip(multiplier, 0.3, 1.5))

        # ?떊猶곕룄 議곗젙
        conf_adj = (valid_ratio - 0.5) * 0.4 - dm_bias * 0.2
        conf_adj = float(np.clip(conf_adj, -0.3, 0.3))

        # ?꽕紐?
        notes = []
        notes.append(f"{sig_count}/{total} ?떊?샇 ?쑀?쓽")
        if dm_bias >= 0.5:
            notes.append(f"?뜲?씠?꽣留덉씠?떇 ?렪?뼢 二쇱쓽({dm_bias:.2f})")
        if valid_ratio >= 0.7:
            notes.append("?떊?샇 ?뭹吏? ?슦?닔")
        elif valid_ratio <= 0.3:
            notes.append("?떊?샇 ?뭹吏? 遺덈웾")

        return SignalValidationSignal(
            valid_signal_ratio=round(valid_ratio, 4),
            data_mining_bias=round(dm_bias, 4),
            total_signals_tested=total,
            significant_signals=sig_count,
            strongest_signal=strongest,
            weakest_signal=weakest,
            min_sample_required=self.min_samples,
            position_multiplier=round(multiplier, 2),
            confidence_adjustment=round(conf_adj, 4),
            validation_details=results,
            note=" | ".join(notes),
        )

    def validate_from_dataframe(
        self, df: pd.DataFrame, signal_columns: Optional[List[str]] = None
    ) -> SignalValidationSignal:
        """DataFrame?뿉?꽌 吏곸젒 ?떊?샇 寃?利?

        Args:
            df: 'close' + ?떊?샇 而щ읆(-1,0,1) ?룷?븿 DataFrame
            signal_columns: 寃?利앺븷 ?떊?샇 而щ읆 紐⑸줉 (?뾾?쑝硫? ?옄?룞 ?깘?깋)
        """
        if len(df) < self.min_samples:
            return SignalValidationSignal(
                valid_signal_ratio=0.0,
                data_mining_bias=0.0,
                total_signals_tested=0,
                significant_signals=0,
                position_multiplier=1.0,
                confidence_adjustment=0.0,
                note="?뜲?씠?꽣 遺?議?",
            )

        if signal_columns is None:
            # signal_ ?젒?몢?궗 而щ읆 ?옄?룞 ?깘?깋
            signal_columns = [c for c in df.columns if c.startswith("signal_") or c == "signal"]

        if not signal_columns:
            return SignalValidationSignal(
                valid_signal_ratio=0.5,
                data_mining_bias=0.0,
                total_signals_tested=0,
                significant_signals=0,
                position_multiplier=1.0,
                confidence_adjustment=0.0,
                note="?떊?샇 而щ읆 ?뾾?쓬",
            )

        forward_ret = df["close"].pct_change().shift(-1)
        signal_results = {}

        for col in signal_columns:
            signals = df[col]
            # ?떊?샇 諛쒖깮 ?떆?젏留?
            active = signals[signals != 0]
            if len(active) < 5:
                continue

            # ?쟻以?: ?떊?샇 諛⑺뼢怨? ?닔?씡瑜? 諛⑺뼢 ?씪移?
            correct = (active * forward_ret.loc[active.index] > 0).sum()
            total = forward_ret.loc[active.index].notna().sum()
            signal_results[col] = (int(correct), int(total))

        return self.validate_all(signal_results, num_strategies_tried=len(signal_columns))
