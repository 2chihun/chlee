"""AI Trader 핵심 모듈 테스트

API 키 없이 실행 가능한 단위 테스트
"""

import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── 샘플 데이터 생성 헬퍼 ────────────────────────────────────


def make_ohlcv(n: int = 200, base_price: float = 50000) -> pd.DataFrame:
    """테스트용 OHLCV 데이터 생성"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-02 09:00", periods=n, freq="5min")
    close = base_price + np.cumsum(np.random.randn(n) * 100)
    close = np.maximum(close, 1000)  # 음수 방지

    df = pd.DataFrame({
        "datetime": dates,
        "open": close + np.random.randn(n) * 50,
        "high": close + abs(np.random.randn(n) * 100),
        "low": close - abs(np.random.randn(n) * 100),
        "close": close,
        "volume": np.random.randint(1000, 100000, n),
    })
    df["open"] = df["open"].clip(lower=100)
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df


# ══════════════════════════════════════════════════════════════
# 1. 기술적 지표 테스트
# ══════════════════════════════════════════════════════════════


class TestIndicators(unittest.TestCase):
    """features/indicators.py 테스트"""

    def setUp(self):
        self.df = make_ohlcv(200)

    def test_add_all_indicators(self):
        from features.indicators import add_all_indicators
        result = add_all_indicators(self.df.copy())

        expected_cols = ["rsi_14", "macd", "macd_signal", "bb_upper",
                         "bb_lower", "bb_mid", "ema_5", "ema_20"]
        for col in expected_cols:
            self.assertIn(col, result.columns, f"Missing column: {col}")

        self.assertEqual(len(result), len(self.df))

    def test_rsi_range(self):
        from features.indicators import rsi
        result = rsi(self.df["close"])
        valid = result.dropna()
        self.assertTrue((valid >= 0).all(), "RSI < 0")
        self.assertTrue((valid <= 100).all(), "RSI > 100")

    def test_bollinger_bands(self):
        from features.indicators import bollinger_bands
        upper, mid, lower, bw, pctb = bollinger_bands(self.df["close"])
        mask = upper.notna() & mid.notna() & lower.notna()

        self.assertTrue((upper[mask] >= mid[mask]).all(),
                        "Upper band < Mid band")
        self.assertTrue((mid[mask] >= lower[mask]).all(),
                        "Mid band < Lower band")

    def test_macd(self):
        from features.indicators import macd
        line, sig, hist = macd(self.df["close"])
        self.assertEqual(len(line), len(self.df))
        valid_hist = hist.dropna()
        # MACD histogram ≈ line - signal
        expected = (line - sig).dropna()
        pd.testing.assert_series_equal(
            valid_hist, expected.loc[valid_hist.index],
            check_names=False, atol=1e-6
        )

    def test_vwap(self):
        from features.indicators import vwap
        result = vwap(self.df)
        self.assertEqual(len(result), len(self.df))
        valid = result.dropna()
        self.assertTrue((valid > 0).all(), "VWAP should be positive")

    def test_atr(self):
        from features.indicators import atr
        result = atr(self.df)
        valid = result.dropna()
        self.assertTrue((valid >= 0).all(), "ATR should be non-negative")


# ══════════════════════════════════════════════════════════════
# 2. 전략 테스트
# ══════════════════════════════════════════════════════════════


class TestStrategies(unittest.TestCase):
    """strategies/ 테스트"""

    def setUp(self):
        from features.indicators import add_all_indicators
        self.df = add_all_indicators(make_ohlcv(200))

    def test_scalping_signal_type(self):
        from strategies.scalping import ScalpingStrategy
        from strategies.base import SignalType

        strategy = ScalpingStrategy()
        signal = strategy.generate_signal(self.df)

        if signal is not None:
            self.assertIn(signal.type,
                          [SignalType.BUY, SignalType.SELL, SignalType.HOLD])
            self.assertGreaterEqual(signal.confidence, 0)
            self.assertLessEqual(signal.confidence, 1)

    def test_swing_signal_type(self):
        from strategies.swing import SwingStrategy
        from strategies.base import SignalType

        strategy = SwingStrategy()
        signal = strategy.generate_signal(self.df)

        if signal is not None:
            self.assertIn(signal.type,
                          [SignalType.BUY, SignalType.SELL, SignalType.HOLD])

    def test_signal_has_stop_loss(self):
        from strategies.scalping import ScalpingStrategy
        from strategies.base import SignalType

        strategy = ScalpingStrategy()
        signal = strategy.generate_signal(self.df)

        if signal and signal.type == SignalType.BUY:
            self.assertIsNotNone(signal.stop_loss,
                                 "BUY signal should have stop_loss")
            self.assertIsNotNone(signal.take_profit,
                                 "BUY signal should have take_profit")


# ══════════════════════════════════════════════════════════════
# 3. 리스크 매니저 테스트
# ══════════════════════════════════════════════════════════════


class TestRiskManager(unittest.TestCase):
    """risk/manager.py 테스트"""

    def setUp(self):
        self.risk_config = MagicMock()
        self.risk_config.max_position_size = 2_000_000
        self.risk_config.max_daily_loss = -100_000
        self.risk_config.max_positions = 5
        self.risk_config.min_confidence = 0.6
        self.risk_config.min_roe = 0.0
        self.risk_config.require_profitable = False
        self.risk_config.max_debt_ratio = 0.0

        self.db = MagicMock()
        self.session_mock = MagicMock()
        self.db.get_session.return_value = self.session_mock

    def test_daily_loss_limit(self):
        import datetime as dt_mod
        from risk.manager import RiskManager
        rm = RiskManager(self.risk_config, self.db)
        rm._daily_reset_date = dt_mod.date.today()  # 리셋 방지
        rm._daily_pnl = -200_000  # 이미 한도 초과

        from strategies.base import Signal, SignalType
        signal = Signal(
            type=SignalType.BUY,
            stock_code="005930",
            price=50000,
            confidence=0.8,
        )

        result = rm.check_signal(signal, 10_000_000)
        self.assertEqual(result.type, SignalType.HOLD,
                         "일일 손실 한도 초과 시 HOLD 반환해야 함")

    def test_position_sizing(self):
        from risk.manager import RiskManager
        from strategies.base import Signal, SignalType

        rm = RiskManager(self.risk_config, self.db)
        rm._daily_pnl = 0

        # 포지션 수 질의 모킹 (filter_by().count() / filter_by().first() 모두 처리)
        filter_mock = MagicMock()
        filter_mock.count.return_value = 0
        filter_mock.first.return_value = None
        query_mock = MagicMock()
        query_mock.filter_by.return_value = filter_mock
        self.session_mock.query.return_value = query_mock

        signal = Signal(
            type=SignalType.BUY,
            stock_code="005930",
            price=50000,
            confidence=0.8,
        )

        result = rm.check_signal(signal, 10_000_000)
        if result.type == SignalType.BUY:
            cost = result.quantity * result.price
            self.assertLessEqual(cost, self.risk_config.max_position_size,
                                 "포지션 금액이 한도를 초과하면 안 됨")
            self.assertGreater(result.quantity, 0, "포지션 크기는 양수여야 함")


# ══════════════════════════════════════════════════════════════
# 4. 백테스트 엔진 테스트
# ══════════════════════════════════════════════════════════════


class TestBacktestEngine(unittest.TestCase):
    """backtest/engine.py 테스트"""

    def setUp(self):
        from features.indicators import add_all_indicators
        self.df = add_all_indicators(make_ohlcv(500, base_price=50000))

    def test_backtest_run(self):
        from backtest.engine import BacktestEngine
        from strategies.scalping import ScalpingStrategy

        strategy = ScalpingStrategy()

        engine = BacktestEngine(strategy, initial_capital=10_000_000)
        metrics = engine.run(self.df)

        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.initial_capital, 0)
        self.assertIsNotNone(metrics.total_return_pct)
        self.assertIsNotNone(metrics.max_drawdown_pct)
        self.assertGreaterEqual(metrics.total_trades, 0)

    def test_cost_model(self):
        from backtest.engine import CostModel

        model = CostModel()

        buy_cost = model.buy_cost(50000, 10)
        self.assertGreater(buy_cost, 0, "매수 비용은 양수")

        sell_cost = model.sell_cost(50000, 10)
        self.assertGreater(sell_cost, 0, "매도 비용은 양수")

    def test_monte_carlo(self):
        from backtest.engine import BacktestEngine
        from strategies.scalping import ScalpingStrategy

        strategy = ScalpingStrategy()

        engine = BacktestEngine(strategy, initial_capital=10_000_000)
        _ = engine.run(self.df)

        if engine.strategy:
            mc = engine.monte_carlo(self.df, n_simulations=50)
            if "error" not in mc:
                self.assertIn("prob_profit", mc)
                self.assertIn("mean_return_pct", mc)


# ══════════════════════════════════════════════════════════════
# 5. 설정 테스트
# ══════════════════════════════════════════════════════════════


class TestConfig(unittest.TestCase):
    """config/settings.py 테스트"""

    def test_load_config_defaults(self):
        from config.settings import load_config
        config = load_config()

        self.assertIsNotNone(config.kis)
        self.assertIsNotNone(config.db)
        self.assertIsNotNone(config.risk)
        self.assertIsNotNone(config.strategy)
        self.assertIn(config.kis.trading_mode, ["paper", "live"])

    def test_risk_config_values(self):
        from config.settings import load_config
        config = load_config()

        self.assertGreater(config.risk.max_position_size, 0)
        self.assertLess(config.risk.max_daily_loss, 0)
        self.assertGreater(config.risk.max_positions, 0)


# ══════════════════════════════════════════════════════════════
# 6. 데이터 수집기 테스트 (Mock)
# ══════════════════════════════════════════════════════════════


class TestDataCollector(unittest.TestCase):
    """data/collector.py 모킹 테스트"""

    @patch("data.collector.requests.post")
    def test_auth_token(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "access_token": "test_token_123",
                "token_type": "Bearer",
                "expires_in": 86400,
            })
        )

        from data.collector import KISAuth
        config = MagicMock()
        config.app_key = "test_key"
        config.app_secret = "test_secret"
        config.base_url = "https://openapivts.koreainvestment.com:29443"

        auth = KISAuth(config)
        token = auth.access_token  # property 접근 시 _issue_token() 호출

        self.assertEqual(token, "test_token_123")

    @patch("data.collector.requests.get")
    def test_current_price(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "rt_cd": "0",
                "output": {
                    "stck_prpr": "72000",
                    "hts_kor_isnm": "삼성전자",
                    "prdy_vrss": "1000",
                    "prdy_ctrt": "1.41",
                    "acml_vol": "12345678",
                    "acml_tr_pbmn": "100000000000",
                    "stck_oprc": "71000",
                    "stck_hgpr": "72500",
                    "stck_lwpr": "70500",
                }
            })
        )

        from data.collector import KISDataCollector, KISAuth
        auth = MagicMock(spec=KISAuth)
        auth.access_token = "test_token"
        auth.config = MagicMock()
        auth.config.app_key = "key"
        auth.config.app_secret = "secret"
        auth.config.base_url = "https://openapivts.koreainvestment.com:29443"

        collector = KISDataCollector(auth)
        result = collector.get_current_price("005930")

        self.assertEqual(result["price"], 72000)


# ══════════════════════════════════════════════════════════════
# 7. 데이터베이스 테스트 (SQLite In-Memory)
# ══════════════════════════════════════════════════════════════


class TestDatabase(unittest.TestCase):
    """data/database.py SQLite 인메모리 테스트"""

    def test_create_tables(self):
        from data.database import Database, Trade
        from config.settings import DBConfig

        config = DBConfig(use_sqlite=True, sqlite_path=":memory:")

        db = Database(config)
        db.init_db()

        session = db.get_session()
        trade = Trade(
            stock_code="005930",
            stock_name="삼성전자",
            strategy="scalping",
            side="BUY",
            price=72000,
            quantity=10,
            fee=108,
            tax=0,
            executed_at=datetime.now(),
        )
        session.add(trade)
        session.commit()

        loaded = session.query(Trade).first()
        self.assertEqual(loaded.stock_code, "005930")
        self.assertEqual(loaded.price, 72000)
        session.close()


# ══════════════════════════════════════════════════════════════
# 8. WebSocket 파싱 테스트
# ══════════════════════════════════════════════════════════════


class TestWebSocketParsing(unittest.TestCase):
    """data/websocket_client.py 파싱 테스트"""

    def _make_ws(self):
        from data.websocket_client import KISWebSocket
        auth = MagicMock()
        auth.config = MagicMock()
        auth.config.base_url = "https://openapivts.koreainvestment.com:29443"
        auth.config.ws_url = "ws://ops.koreainvestment.com:31000"
        return KISWebSocket(auth)

    def test_parse_tick_field_indices(self):
        """체결 데이터의 ask/bid 가격 인덱스가 공식 API 스펙과 일치하는지 확인"""
        ws = self._make_ws()
        # H0STCNT0 columns: [0]종목코드 [1]시간 [2]현재가 [3]부호 [4]전일대비
        # [5]등락률 [6]가중평균가 [7]시가 [8]고가 [9]저가
        # [10]ASKP1 [11]BIDP1 [12]체결량 [13]누적거래량 ... [19]매도체결금액
        fields = ["005930", "100000", "72000", "2", "1000",
                  "1.41", "71500", "71000", "72500", "70500",
                  "72100", "71900", "500", "12345678", "100000000000",
                  "100", "200", "100", "55.5", "50000000", "60000000"]
        raw = f"0|H0STCNT0|001|{'%5E'.replace('%5E','^').join(fields)}"
        # Rebuild raw correctly
        raw = "0|H0STCNT0|001|" + "^".join(fields)

        tick = ws._parse_tick(raw)
        self.assertIsNotNone(tick)
        self.assertEqual(tick.ask_price, 72100)   # items[10] = ASKP1
        self.assertEqual(tick.bid_price, 71900)   # items[11] = BIDP1
        self.assertEqual(tick.price, 72000)        # items[2]
        self.assertEqual(tick.volume, 500)         # items[12]
        self.assertEqual(tick.acc_volume, 12345678) # items[13]

    def test_parse_orderbook_consecutive_layout(self):
        """호가 데이터가 연속 블록 배치 (공식 API 스펙)에 맞게 파싱되는지 확인"""
        ws = self._make_ws()
        # H0STASP0 columns: [0]종목코드 [1]시간 [2]시간구분
        # [3~12]ASKP1~10, [13~22]BIDP1~10, [23~32]ASKP_RSQN1~10,
        # [33~42]BIDP_RSQN1~10, [43]TOTAL_ASKP, [44]TOTAL_BIDP, ...
        items = ["005930", "100000", "0"]
        ask_p = [str(72000 + i * 100) for i in range(10)]    # [3~12]
        bid_p = [str(71900 - i * 100) for i in range(10)]    # [13~22]
        ask_v = [str(1000 + i * 10) for i in range(10)]      # [23~32]
        bid_v = [str(2000 + i * 10) for i in range(10)]      # [33~42]
        totals = ["15000", "25000"]                           # [43,44]
        # 남은 필드 (45~58) 패딩
        padding = ["0"] * 14

        items = items + ask_p + bid_p + ask_v + bid_v + totals + padding
        raw = "0|H0STASP0|001|" + "^".join(items)

        ob = ws._parse_orderbook(raw)
        self.assertIsNotNone(ob)
        self.assertEqual(ob.ask_prices[0], 72000)   # ASKP1
        self.assertEqual(ob.ask_prices[9], 72900)   # ASKP10
        self.assertEqual(ob.bid_prices[0], 71900)   # BIDP1
        self.assertEqual(ob.bid_prices[9], 71000)   # BIDP10
        self.assertEqual(ob.ask_volumes[0], 1000)   # ASKP_RSQN1
        self.assertEqual(ob.bid_volumes[0], 2000)   # BIDP_RSQN1
        self.assertEqual(ob.total_ask_volume, 15000)
        self.assertEqual(ob.total_bid_volume, 25000)


# ══════════════════════════════════════════════════════════════


if __name__ == "__main__":
    unittest.main(verbosity=2)
