"""AI Trader 메인 트레이딩 봇

전체 시스템을 조율하는 오케스트레이터
- 데이터 수집 → 지표 계산 → 전략 신호 → 리스크 체크 → 주문 실행
- 모의투자/실전투자 모드 지원
- 스케줄 기반
"""

import asyncio
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import schedule
from loguru import logger

from config.settings import load_config, AppConfig
from data.collector import KISAuth, KISDataCollector
from data.database import Database
from data.websocket_client import KISWebSocket
from data.backup import BackupManager, StatisticsEngine
from features.indicators import add_all_indicators
from strategies.scalping import ScalpingStrategy
from strategies.swing import SwingStrategy
from strategies.base import SignalType
from execution.order import KISOrderExecutor
from risk.manager import RiskManager


class AITrader:
    """AI 단타 트레이딩 봇 메인 클래스"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.running = False
        self._stop_event = threading.Event()

        # 로거 설정
        logger.remove()
        logger.add(sys.stderr, level="INFO",
                    format="<green>{time:HH:mm:ss}</green> | <level>{level:^8}</level> | {message}")
        logger.add("logs/trader_{time:YYYY-MM-DD}.log", rotation="1 day",
                    retention="30 days", level="DEBUG")

        # 컴포넌트 초기화
        logger.info("AI Trader 초기화 중...")

        self.db = Database(config.db)
        self.db.init_db()

        self.auth = KISAuth(config.kis)
        self.collector = KISDataCollector(self.auth)
        self.executor = KISOrderExecutor(self.auth)
        self.risk = RiskManager(config.risk, self.db)
        self.backup = BackupManager(config.backup, self.db)
        self.stats = StatisticsEngine(self.db)

        # 전략 초기화
        self.strategies = {
            "scalping": ScalpingStrategy(),
            "swing": SwingStrategy(),
        }

        # 감시 종목 (설정에서 로드)
        self.watchlist = config.strategy.watchlist

        # WebSocket (실시간 틱)
        self.ws: Optional[KISWebSocket] = None

        logger.info(f"초기화 완료 | 모드: {config.kis.trading_mode} | "
                    f"종목: {len(self.watchlist)}개 | "
                    f"전략: {list(self.strategies.keys())}")

    # ── 장 시간 체크 ──────────────────────────────────────────

    @staticmethod
    def is_market_hours() -> bool:
        """한국 주식시장 개장 시간 확인 (09:00~15:20)"""
        now = datetime.now()
        # 주말 체크
        if now.weekday() >= 5:
            return False
        market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=20, second=0, microsecond=0)
        return market_open <= now <= market_close

    @staticmethod
    def is_pre_market() -> bool:
        """장전 준비 시간 (08:30~09:00)"""
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        pre_open = now.replace(hour=8, minute=30, second=0, microsecond=0)
        market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
        return pre_open <= now < market_open

    # ── 데이터 수집 ──────────────────────────────────────────

    def collect_market_data(self, stock_code: str) -> Optional[pd.DataFrame]:
        """종목의 분봉 데이터 수집 및 지표 계산"""
        try:
            df = self.collector.get_minute_candles(stock_code, interval=5)
            if df is None or df.empty:
                return None

            df = add_all_indicators(df)
            return df

        except Exception as e:
            logger.error(f"데이터 수집 실패 [{stock_code}]: {e}")
            return None

    def save_candle_data(self, stock_code: str, df: pd.DataFrame):
        """수집한 캔들 데이터를 DB에 저장"""
        try:
            session = self.db.get_session()
            from data.database import MinuteCandle
            for _, row in df.tail(1).iterrows():
                candle = MinuteCandle(
                    stock_code=stock_code,
                    datetime=row.get("datetime", datetime.now()),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                )
                session.merge(candle)
            session.commit()
            session.close()
        except Exception as e:
            logger.debug(f"캔들 저장 오류: {e}")

    # ── 매매 사이클 ──────────────────────────────────────────

    def run_trading_cycle(self):
        """한 번의 매매 사이클 실행"""
        if not self.is_market_hours():
            return

        for stock_code in self.watchlist:
            if self._stop_event.is_set():
                break

            try:
                self._process_stock(stock_code)
            except Exception as e:
                logger.error(f"매매 사이클 오류 [{stock_code}]: {e}")

            time.sleep(0.2)  # API 호출 간격

    def _process_stock(self, stock_code: str):
        """개별 종목 처리"""
        df = self.collect_market_data(stock_code)
        if df is None or len(df) < 60:
            return

        self.save_candle_data(stock_code, df)

        # 현재가 조회
        current = self.collector.get_current_price(stock_code)
        if not current:
            return

        current_price = current.get("price", 0)
        stock_name = current.get("stock_code", stock_code)

        # 현재 보유 포지션 확인
        session = self.db.get_session()
        from data.database import Position
        position = session.query(Position).filter(
            Position.stock_code == stock_code,
            Position.is_active == True,
        ).first()

        current_pos = None
        if position:
            current_pos = {
                "stock_code": position.stock_code,
                "quantity": position.quantity,
                "avg_price": position.avg_price,
            }
        session.close()

        # 각 전략별 신호 생성
        for strategy_name, strategy in self.strategies.items():
            signal = strategy.generate_signal(df, current_pos)
            if signal is None or signal.type == SignalType.HOLD:
                continue

            # 매수 신호인 경우 리스크 체크
            if signal.type == SignalType.BUY:
                balance_info = self.executor.get_balance()
                available_cash = int(balance_info.get("available_cash", 10_000_000)) if balance_info else 10_000_000
                checked = self.risk.check_signal(signal, available_cash)
                if checked.type == SignalType.HOLD:
                    continue
                signal = checked

            # 주문 실행
            self._execute_signal(signal, stock_code, stock_name,
                                 int(current_price), strategy_name)

    def _execute_signal(self, signal, stock_code: str, stock_name: str,
                        price: int, strategy_name: str):
        """신호에 따른 주문 실행"""
        from strategies.base import SignalType
        from data.database import Trade

        if signal.type == SignalType.BUY:
            quantity = signal.quantity
            if quantity <= 0:
                return

            logger.info(
                f"📈 매수 신호 | {stock_name}({stock_code}) | "
                f"수량: {quantity} | 가격: {price:,}원 | "
                f"전략: {strategy_name} | 신뢰도: {signal.confidence:.2f}"
            )

            result = self.executor.buy(stock_code, quantity, price)
            if result and result.get("rt_cd") == "0":
                session = self.db.get_session()
                trade = Trade(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    strategy=strategy_name,
                    side="BUY",
                    price=price,
                    quantity=quantity,
                    fee=int(price * quantity * 0.00015),
                    tax=0,
                    executed_at=datetime.now(),
                )
                session.add(trade)
                session.commit()
                session.close()
                logger.info(f"✅ 매수 체결 | {stock_name} {quantity}주 @ {price:,}")

        elif signal.type == SignalType.SELL:
            # 보유 포지션 확인
            session = self.db.get_session()
            from data.database import Position
            position = session.query(Position).filter(
                Position.stock_code == stock_code,
                Position.is_active == True,
            ).first()

            if not position:
                session.close()
                return

            quantity = position.quantity
            buy_price = position.avg_price

            logger.info(
                f"📉 매도 신호 | {stock_name}({stock_code}) | "
                f"수량: {quantity} | 가격: {price:,}원 | "
                f"전략: {strategy_name}"
            )

            result = self.executor.sell(stock_code, quantity, price)
            if result and result.get("rt_cd") == "0":
                pnl = (price - buy_price) * quantity
                pnl_pct = ((price / buy_price) - 1) * 100 if buy_price else 0
                fee = int(price * quantity * 0.00015)
                tax = int(price * quantity * 0.0018)

                trade = Trade(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    strategy=strategy_name,
                    side="SELL",
                    price=price,
                    quantity=quantity,
                    pnl=pnl - fee - tax,
                    pnl_pct=round(pnl_pct, 2),
                    fee=fee,
                    tax=tax,
                    executed_at=datetime.now(),
                )
                session.add(trade)

                position.is_active = False
                session.commit()
                session.close()

                emoji = "🟢" if pnl > 0 else "🔴"
                logger.info(
                    f"{emoji} 매도 체결 | {stock_name} {quantity}주 @ {price:,} | "
                    f"손익: {pnl - fee - tax:,}원 ({pnl_pct:.2f}%)"
                )

                self.risk.update_daily_pnl(pnl - fee - tax)
            else:
                session.close()

    # ── 스케줄러 ─────────────────────────────────────────────

    def setup_schedule(self):
        """스케줄러 설정"""
        # 매매 사이클: 5분마다
        schedule.every(5).minutes.do(self.run_trading_cycle)

        # 장전 준비: 08:30
        schedule.every().day.at("08:30").do(self.pre_market_routine)

        # 장 마감 정리: 15:25
        schedule.every().day.at("15:25").do(self.post_market_routine)

        # 일일 백업: 00:30
        schedule.every().day.at("00:30").do(self.daily_backup)

        # 일일 통계: 15:35
        schedule.every().day.at("15:35").do(self.daily_stats)

        logger.info("스케줄러 설정 완료")

    def pre_market_routine(self):
        """장전 준비 루틴"""
        logger.info("=== 장전 준비 루틴 시작 ===")

        # 거래량 상위 종목 갱신
        try:
            vol_rank = self.collector.get_volume_rank()
            if vol_rank:
                top_codes = [item.get("mksc_shrn_iscd", "")
                             for item in vol_rank[:20]]
                existing = set(self.watchlist)
                new_stocks = [c for c in top_codes
                              if c and c not in existing][:5]
                if new_stocks:
                    self.watchlist.extend(new_stocks)
                    logger.info(f"감시 종목 추가: {new_stocks}")
        except Exception as e:
            logger.error(f"장전 루틴 오류: {e}")

        logger.info(f"장전 준비 완료 | 감시 종목: {len(self.watchlist)}개")

    def post_market_routine(self):
        """장 마감 정리 루틴"""
        logger.info("=== 장 마감 정리 루틴 시작 ===")

        # 일일 PnL 기록
        self.risk.record_daily_summary()

        # 보유 포지션 정리 안내
        session = self.db.get_session()
        from data.database import Position
        active = session.query(Position).filter(
            Position.is_active == True
        ).all()

        if active:
            logger.warning(f"미청산 포지션 {len(active)}개:")
            for p in active:
                logger.warning(f"  {p.stock_code} | {p.quantity}주 | "
                               f"매입가: {p.avg_price:,}")

        session.close()
        logger.info("장 마감 정리 완료")

    def daily_backup(self):
        """일일 백업"""
        logger.info("일일 백업 실행 중...")
        self.backup.run_full_backup()
        logger.info("일일 백업 완료")

    def daily_stats(self):
        """일일 통계 생성"""
        try:
            stats = self.stats.get_overall_stats()
            logger.info(
                f"=== 일일 통계 ===\n"
                f"총 거래: {stats.get('total_trades', 0)}회 | "
                f"승률: {stats.get('win_rate', 0):.1f}% | "
                f"총 손익: {stats.get('total_pnl', 0):,}원"
            )
        except Exception as e:
            logger.error(f"통계 생성 오류: {e}")

    # ── 메인 루프 ────────────────────────────────────────────

    def start(self):
        """봇 시작"""
        logger.info("=" * 50)
        logger.info("🚀 AI Trader 봇 시작")
        logger.info(f"   모드: {self.config.kis.trading_mode}")
        logger.info(f"   감시 종목: {self.watchlist}")
        logger.info("=" * 50)

        self.running = True
        self._stop_event.clear()

        self.setup_schedule()

        # 시그널 핸들러 (메인 스레드에서만 등록 가능)
        import threading
        if threading.current_thread() is threading.main_thread():
            def handle_signal(signum, frame):
                logger.info("종료 신호 수신, 안전하게 종료합니다...")
                self.stop()

            signal.signal(signal.SIGINT, handle_signal)
            signal.signal(signal.SIGTERM, handle_signal)

        # 메인 루프
        try:
            while self.running and not self._stop_event.is_set():
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("키보드 인터럽트 → 종료")
        finally:
            self.stop()

    def stop(self):
        """봇 정지"""
        self.running = False
        self._stop_event.set()

        if self.ws:
            asyncio.get_event_loop().run_until_complete(self.ws.disconnect())

        logger.info("🛑 AI Trader 봇 정지 완료")


# ── FastAPI + 봇 통합 실행을 위한 글로벌 인스턴스 ──────────────

_trader_instance: Optional[AITrader] = None
_trader_thread: Optional[threading.Thread] = None


def get_trader() -> Optional[AITrader]:
    return _trader_instance


def start_bot():
    global _trader_instance, _trader_thread
    if _trader_instance and _trader_instance.running:
        return {"status": "already_running"}

    config = load_config()
    _trader_instance = AITrader(config)

    _trader_thread = threading.Thread(target=_trader_instance.start, daemon=True)
    _trader_thread.start()
    return {"status": "started"}


def stop_bot():
    global _trader_instance
    if _trader_instance:
        _trader_instance.stop()
        return {"status": "stopped"}
    return {"status": "not_running"}


# ── 엔트리포인트 ─────────────────────────────────────────────

def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="AI Trader 단타 주식 투자 시스템")
    parser.add_argument("--mode", choices=["bot", "server", "both"],
                        default="both", help="실행 모드")
    parser.add_argument("--paper", action="store_true",
                        help="모의투자 모드 강제 설정")
    args = parser.parse_args()

    config = load_config()

    if args.paper:
        config.kis.trading_mode = "paper"

    if args.mode == "bot":
        trader = AITrader(config)
        trader.start()

    elif args.mode == "server":
        import uvicorn
        uvicorn.run("dashboard.api_server:app", host="0.0.0.0",
                    port=8000, reload=True)

    elif args.mode == "both":
        # 봇을 별도 스레드에서 실행
        global _trader_instance, _trader_thread
        _trader_instance = AITrader(config)
        _trader_thread = threading.Thread(
            target=_trader_instance.start, daemon=True
        )
        _trader_thread.start()

        # FastAPI 서버 실행 (메인 스레드)
        import uvicorn
        logger.info("FastAPI + 봇 통합 모드로 실행")
        uvicorn.run("dashboard.api_server:app", host="0.0.0.0",
                    port=8000, reload=False)


if __name__ == "__main__":
    main()
