"""고급 백테스트 엔진

슬리피지 모델, 수수료/세금, 몬테카를로 시뮬레이션,
파라미터 최적화, 워크포워드 분석, 스트레스 테스트 지원
"""

import json
import itertools
import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from strategies.base import BaseStrategy, SignalType


# ── 비용 모델 ───────────────────────────────────────────────────

@dataclass
class CostModel:
    """매매 비용 모델"""
    commission_rate: float = 0.00015   # 수수료율 0.015% (매수+매도 각각)
    tax_rate: float = 0.0018           # 거래세 0.18% (매도 시에만)
    slippage_pct: float = 0.1          # 슬리피지 0.1%
    min_commission: int = 0            # 최소 수수료

    def buy_cost(self, price: int, quantity: int) -> int:
        amount = price * quantity
        commission = max(int(amount * self.commission_rate), self.min_commission)
        slippage = int(amount * self.slippage_pct / 100)
        return commission + slippage

    def sell_cost(self, price: int, quantity: int) -> int:
        amount = price * quantity
        commission = max(int(amount * self.commission_rate), self.min_commission)
        tax = int(amount * self.tax_rate)
        slippage = int(amount * self.slippage_pct / 100)
        return commission + tax + slippage

    def effective_buy_price(self, price: int) -> int:
        return int(price * (1 + self.slippage_pct / 100))

    def effective_sell_price(self, price: int) -> int:
        return int(price * (1 - self.slippage_pct / 100))


# ── 백테스트 결과 ───────────────────────────────────────────────

@dataclass
class BacktestMetrics:
    """백테스트 성과 지표"""
    initial_capital: int = 0
    final_capital: int = 0
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_profit_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_holding_bars: float = 0.0
    total_fees: int = 0
    total_taxes: int = 0
    start_date: str = ""
    end_date: str = ""
    strategy_name: str = ""
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TradeRecord:
    """개별 거래 기록"""
    entry_date: str
    exit_date: str
    stock_code: str
    side: str
    entry_price: int
    exit_price: int
    quantity: int
    pnl: int
    pnl_pct: float
    fee: int
    tax: int
    holding_bars: int
    reason: str = ""


# ── 백테스트 엔진 ───────────────────────────────────────────────

class BacktestEngine:
    """고급 백테스트 엔진"""

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: int = 10_000_000,
        cost_model: Optional[CostModel] = None,
        max_positions: int = 5,
        position_size_pct: float = 20.0,
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.cost_model = cost_model or CostModel()
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct

    def run(self, df: pd.DataFrame, stock_code: str = "TEST") -> BacktestMetrics:
        """백테스트를 실행합니다.

        Args:
            df: OHLCV DataFrame (date/datetime, open, high, low, close, volume)
            stock_code: 종목코드

        Returns:
            백테스트 성과 지표
        """
        if df.empty or len(df) < 20:
            logger.warning("데이터가 부족합니다 (최소 20봉 필요)")
            return BacktestMetrics(initial_capital=self.initial_capital)

        analyzed = self.strategy.analyze(df)
        capital = self.initial_capital
        position = None
        trades: list[TradeRecord] = []
        equity_curve = []
        max_equity = capital

        for i in range(1, len(analyzed)):
            row = analyzed.iloc[i]
            prev = analyzed.iloc[i - 1]
            price = int(row["close"])
            time_col = "datetime" if "datetime" in analyzed.columns else "date"
            current_time = str(row[time_col])

            current_equity = capital
            if position:
                current_equity += (price - position["avg_price"]) * position["quantity"]

            equity_curve.append({
                "time": current_time,
                "equity": current_equity,
                "capital": capital,
            })
            max_equity = max(max_equity, current_equity)

            signal_val = row.get("signal", SignalType.HOLD.value)

            # 포지션 보유 중
            if position:
                entry_price = position["avg_price"]
                pnl_pct = (price - entry_price) / entry_price * 100

                should_sell = False
                reason = ""

                # 손절
                sl_pct = self.strategy.params.get("stop_loss_pct", -1.0)
                if pnl_pct <= sl_pct:
                    should_sell = True
                    reason = f"손절 {pnl_pct:.2f}%"

                # 익절
                tp_pct = self.strategy.params.get("take_profit_pct", 2.0)
                if pnl_pct >= tp_pct:
                    should_sell = True
                    reason = f"익절 {pnl_pct:.2f}%"

                # 시그널 매도
                if signal_val == SignalType.SELL.value and not should_sell:
                    should_sell = True
                    reason = "시그널 매도"

                # 최대 보유 기간
                max_bars = self.strategy.params.get("max_hold_bars", 60)
                if position["holding_bars"] >= max_bars and not should_sell:
                    should_sell = True
                    reason = "보유기간 초과"

                if should_sell:
                    sell_price = self.cost_model.effective_sell_price(price)
                    sell_amount = sell_price * position["quantity"]
                    sell_cost = self.cost_model.sell_cost(price, position["quantity"])
                    pnl = sell_amount - position["amount"] - position["fee"] - sell_cost
                    pnl_pct_actual = pnl / position["amount"] * 100

                    capital += sell_amount - sell_cost
                    trades.append(TradeRecord(
                        entry_date=position["entry_date"],
                        exit_date=current_time,
                        stock_code=stock_code,
                        side="LONG",
                        entry_price=position["avg_price"],
                        exit_price=sell_price,
                        quantity=position["quantity"],
                        pnl=pnl,
                        pnl_pct=pnl_pct_actual,
                        fee=position["fee"] + int(sell_amount * self.cost_model.commission_rate),
                        tax=int(sell_amount * self.cost_model.tax_rate),
                        holding_bars=position["holding_bars"],
                        reason=reason,
                    ))
                    position = None
                else:
                    position["holding_bars"] += 1

            # 포지션 없음: 진입 검사
            elif signal_val == SignalType.BUY.value:
                buy_price = self.cost_model.effective_buy_price(price)
                position_amount = int(capital * self.position_size_pct / 100)
                quantity = position_amount // buy_price
                if quantity > 0:
                    amount = buy_price * quantity
                    fee = self.cost_model.buy_cost(price, quantity)
                    capital -= (amount + fee)
                    position = {
                        "stock_code": stock_code,
                        "avg_price": buy_price,
                        "quantity": quantity,
                        "amount": amount,
                        "fee": fee,
                        "entry_date": current_time,
                        "holding_bars": 0,
                    }

        # 잔여 포지션 강제 청산
        if position:
            last_price = int(analyzed.iloc[-1]["close"])
            sell_price = self.cost_model.effective_sell_price(last_price)
            sell_amount = sell_price * position["quantity"]
            sell_cost = self.cost_model.sell_cost(last_price, position["quantity"])
            pnl = sell_amount - position["amount"] - position["fee"] - sell_cost
            capital += sell_amount - sell_cost
            trades.append(TradeRecord(
                entry_date=position["entry_date"],
                exit_date=str(analyzed.iloc[-1][time_col]),
                stock_code=stock_code,
                side="LONG",
                entry_price=position["avg_price"],
                exit_price=sell_price,
                quantity=position["quantity"],
                pnl=pnl,
                pnl_pct=pnl / position["amount"] * 100,
                fee=position["fee"] + int(sell_amount * self.cost_model.commission_rate),
                tax=int(sell_amount * self.cost_model.tax_rate),
                holding_bars=position["holding_bars"],
                reason="강제청산",
            ))

        # 성과 지표 계산
        metrics = self._calc_metrics(trades, equity_curve, analyzed)
        metrics.params = self.strategy.get_params()
        metrics.strategy_name = self.strategy.name
        return metrics

    def _calc_metrics(
        self,
        trades: list[TradeRecord],
        equity_curve: list[dict],
        df: pd.DataFrame,
    ) -> BacktestMetrics:
        """성과 지표를 계산합니다."""
        m = BacktestMetrics(initial_capital=self.initial_capital)

        if not equity_curve:
            return m

        equities = [e["equity"] for e in equity_curve]
        m.final_capital = equities[-1]
        m.total_return_pct = (m.final_capital - m.initial_capital) / m.initial_capital * 100

        time_col = "datetime" if "datetime" in df.columns else "date"
        m.start_date = str(df.iloc[0][time_col])
        m.end_date = str(df.iloc[-1][time_col])

        # 연환산 수익률
        try:
            start = pd.Timestamp(m.start_date)
            end = pd.Timestamp(m.end_date)
            days = max((end - start).days, 1)
            m.annual_return_pct = m.total_return_pct * 365 / days
        except Exception:
            m.annual_return_pct = m.total_return_pct

        # MDD
        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            peak = max(peak, eq)
            if peak > 0:
                dd = (eq - peak) / peak * 100
                max_dd = min(max_dd, dd)
        m.max_drawdown_pct = max_dd

        # 샤프/소르티노
        if len(equities) > 1:
            eq_series = pd.Series(equities)
            returns = eq_series.pct_change().dropna()
            if len(returns) > 1 and returns.std() > 0:
                m.sharpe_ratio = float(returns.mean() / returns.std() * np.sqrt(252))
                neg_returns = returns[returns < 0]
                if len(neg_returns) > 0 and neg_returns.std() > 0:
                    m.sortino_ratio = float(returns.mean() / neg_returns.std() * np.sqrt(252))

        # 칼마 비율
        if m.max_drawdown_pct < 0:
            m.calmar_ratio = m.annual_return_pct / abs(m.max_drawdown_pct)

        # 거래 통계
        m.total_trades = len(trades)
        if trades:
            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl <= 0]
            m.winning_trades = len(wins)
            m.losing_trades = len(losses)
            m.win_rate = m.winning_trades / m.total_trades * 100
            m.avg_profit_pct = float(np.mean([t.pnl_pct for t in wins])) if wins else 0.0
            m.avg_loss_pct = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0
            total_profit = sum(t.pnl for t in wins)
            total_loss = abs(sum(t.pnl for t in losses))
            m.profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")
            m.avg_holding_bars = float(np.mean([t.holding_bars for t in trades]))
            m.total_fees = sum(t.fee for t in trades)
            m.total_taxes = sum(t.tax for t in trades)

            # 연속 승/패
            m.max_consecutive_wins = self._max_consecutive(trades, win=True)
            m.max_consecutive_losses = self._max_consecutive(trades, win=False)

        return m

    @staticmethod
    def _max_consecutive(trades: list[TradeRecord], win: bool) -> int:
        count = 0
        max_count = 0
        for t in trades:
            if (win and t.pnl > 0) or (not win and t.pnl <= 0):
                count += 1
                max_count = max(max_count, count)
            else:
                count = 0
        return max_count

    # ── 파라미터 최적화 ──────────────────────────────────────────

    def optimize(
        self,
        df: pd.DataFrame,
        param_grid: dict,
        metric: str = "sharpe_ratio",
        stock_code: str = "TEST",
    ) -> list[BacktestMetrics]:
        """그리드 서치로 파라미터를 최적화합니다.

        Args:
            df: OHLCV DataFrame
            param_grid: {"rsi_buy": [25, 30, 35], "rsi_sell": [65, 70, 75]} 형태
            metric: 최적화 기준 지표
            stock_code: 종목코드

        Returns:
            metric 기준 정렬된 결과 리스트
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        results = []

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            self.strategy.set_params(params)
            m = self.run(df, stock_code)
            results.append(m)
            logger.debug("최적화: {} → {}={:.4f}", params, metric, getattr(m, metric, 0))

        results.sort(key=lambda x: getattr(x, metric, 0), reverse=True)
        return results

    # ── 워크포워드 분석 ──────────────────────────────────────────

    def walk_forward(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        n_splits: int = 5,
        param_grid: Optional[dict] = None,
        stock_code: str = "TEST",
    ) -> list[BacktestMetrics]:
        """워크포워드 분석을 수행합니다.

        학습 기간에서 최적 파라미터를 찾고, 테스트 기간에서 검증합니다.

        Returns:
            각 분할의 테스트 결과 리스트
        """
        total_len = len(df)
        split_size = total_len // n_splits
        results = []

        for i in range(n_splits):
            start = i * split_size
            end = min(start + split_size, total_len)
            split_data = df.iloc[start:end].reset_index(drop=True)
            train_end = int(len(split_data) * train_ratio)
            train_data = split_data.iloc[:train_end]
            test_data = split_data.iloc[train_end:]

            if len(train_data) < 20 or len(test_data) < 10:
                continue

            # 학습 기간: 파라미터 최적화
            if param_grid:
                opt_results = self.optimize(train_data, param_grid, stock_code=stock_code)
                if opt_results:
                    best_params = opt_results[0].params
                    self.strategy.set_params(best_params)

            # 테스트 기간: 검증
            test_result = self.run(test_data, stock_code)
            results.append(test_result)
            logger.info(
                "워크포워드 {}/{}: 수익률={:.2f}%, 샤프={:.2f}",
                i + 1, n_splits, test_result.total_return_pct, test_result.sharpe_ratio,
            )

        return results

    # ── 몬테카를로 시뮬레이션 ────────────────────────────────────

    def monte_carlo(
        self,
        df: pd.DataFrame,
        n_simulations: int = 1000,
        stock_code: str = "TEST",
    ) -> dict:
        """몬테카를로 시뮬레이션으로 전략 견고성을 검증합니다.

        거래 순서를 랜덤으로 셔플하여 수익 분포를 분석합니다.

        Returns:
            시뮬레이션 결과 통계
        """
        # 먼저 원본 백테스트 실행
        base_result = self.run(df, stock_code)

        analyzed = self.strategy.analyze(df)
        # 시그널 위치 추출
        buy_indices = analyzed.index[analyzed["signal"] == SignalType.BUY.value].tolist()

        if not buy_indices:
            return {"error": "매수 시그널이 없습니다."}

        final_capitals = []
        max_drawdowns = []
        win_rates = []

        rng = np.random.default_rng(42)

        for sim in range(n_simulations):
            shuffled = rng.permutation(buy_indices).tolist()
            capital = self.initial_capital
            wins = 0
            total = 0
            peak = capital
            max_dd = 0.0

            for idx in shuffled:
                if idx + 1 >= len(analyzed):
                    continue
                entry = int(analyzed.iloc[idx]["close"])
                # 다음 매도 시그널 또는 max_hold_bars 이후 청산
                sell_idx = None
                max_bars = self.strategy.params.get("max_hold_bars", 60)
                for j in range(idx + 1, min(idx + max_bars + 1, len(analyzed))):
                    if analyzed.iloc[j]["signal"] == SignalType.SELL.value:
                        sell_idx = j
                        break
                if sell_idx is None:
                    sell_idx = min(idx + max_bars, len(analyzed) - 1)

                exit_price = int(analyzed.iloc[sell_idx]["close"])
                qty = int(capital * self.position_size_pct / 100) // entry
                if qty <= 0:
                    continue

                buy_cost = self.cost_model.buy_cost(entry, qty)
                sell_cost = self.cost_model.sell_cost(exit_price, qty)
                pnl = (exit_price - entry) * qty - buy_cost - sell_cost
                capital += pnl
                total += 1
                if pnl > 0:
                    wins += 1

                peak = max(peak, capital)
                if peak > 0:
                    dd = (capital - peak) / peak * 100
                    max_dd = min(max_dd, dd)

            final_capitals.append(capital)
            max_drawdowns.append(max_dd)
            win_rates.append(wins / total * 100 if total > 0 else 0)

        arr_cap = np.array(final_capitals)
        arr_dd = np.array(max_drawdowns)
        arr_wr = np.array(win_rates)

        return {
            "n_simulations": n_simulations,
            "base_return_pct": base_result.total_return_pct,
            "mean_final_capital": float(arr_cap.mean()),
            "median_final_capital": float(np.median(arr_cap)),
            "std_final_capital": float(arr_cap.std()),
            "percentile_5": float(np.percentile(arr_cap, 5)),
            "percentile_25": float(np.percentile(arr_cap, 25)),
            "percentile_75": float(np.percentile(arr_cap, 75)),
            "percentile_95": float(np.percentile(arr_cap, 95)),
            "mean_return_pct": float((arr_cap.mean() - self.initial_capital) / self.initial_capital * 100),
            "worst_return_pct": float((arr_cap.min() - self.initial_capital) / self.initial_capital * 100),
            "best_return_pct": float((arr_cap.max() - self.initial_capital) / self.initial_capital * 100),
            "mean_max_drawdown": float(arr_dd.mean()),
            "worst_max_drawdown": float(arr_dd.min()),
            "mean_win_rate": float(arr_wr.mean()),
            "prob_profit": float((arr_cap > self.initial_capital).sum() / n_simulations * 100),
        }

    # ── 스트레스 테스트 ──────────────────────────────────────────

    def stress_test(
        self,
        df: pd.DataFrame,
        stock_code: str = "TEST",
    ) -> dict:
        """극단적 시장 상황에서의 전략 성과를 테스트합니다."""
        results = {}

        # 원본
        base = self.run(df, stock_code)
        results["base"] = base.to_dict()

        # 시나리오 1: 급락 (-30%)
        crash_df = df.copy()
        mid = len(crash_df) // 2
        crash_range = slice(mid, min(mid + 5, len(crash_df)))
        for col in ["open", "high", "low", "close"]:
            crash_df.loc[crash_df.index[crash_range], col] = (
                crash_df.iloc[crash_range][col] * 0.7
            ).astype(int)
        crash_result = self.run(crash_df, stock_code)
        results["crash_30pct"] = crash_result.to_dict()

        # 시나리오 2: 급등 (+30%)
        surge_df = df.copy()
        for col in ["open", "high", "low", "close"]:
            surge_df.loc[surge_df.index[crash_range], col] = (
                surge_df.iloc[crash_range][col] * 1.3
            ).astype(int)
        surge_result = self.run(surge_df, stock_code)
        results["surge_30pct"] = surge_result.to_dict()

        # 시나리오 3: 고변동성 (가격 변동 2배)
        volatile_df = df.copy()
        mean_price = volatile_df["close"].mean()
        for col in ["open", "high", "low", "close"]:
            volatile_df[col] = (
                mean_price + (volatile_df[col] - mean_price) * 2
            ).astype(int).clip(lower=1)
        vol_result = self.run(volatile_df, stock_code)
        results["high_volatility"] = vol_result.to_dict()

        # 시나리오 4: 저변동성 횡보
        flat_df = df.copy()
        for col in ["open", "high", "low", "close"]:
            flat_df[col] = (
                mean_price + (flat_df[col] - mean_price) * 0.3
            ).astype(int).clip(lower=1)
        flat_result = self.run(flat_df, stock_code)
        results["low_volatility"] = flat_result.to_dict()

        # 시나리오 5: 거래량 급감
        low_vol_df = df.copy()
        low_vol_df["volume"] = (low_vol_df["volume"] * 0.1).astype(int).clip(lower=1)
        low_vol_result = self.run(low_vol_df, stock_code)
        results["low_volume"] = low_vol_result.to_dict()

        return results
