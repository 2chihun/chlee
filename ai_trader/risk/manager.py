"""리스크 관리 모듈

포지션 사이징, 일일 손실 제한, 동시 보유 제한, 포트폴리오 리스크 관리,
재무건전성 기반 필터링 (S-RIM 책 기반)
"""

import datetime as dt
from typing import Optional

from loguru import logger
from sqlalchemy.orm import Session

from config.settings import RiskConfig
from data.database import Position, Trade, DailyPnL, FinancialData
from strategies.base import Signal, SignalType


class RiskManager:
    """리스크 관리자"""

    def __init__(self, config: RiskConfig, db):
        self.config = config
        self._db = db
        self._daily_pnl: int = 0
        self._daily_trades: int = 0
        self._daily_reset_date: Optional[dt.date] = None

    @property
    def db(self):
        """DB 세션을 반환합니다."""
        if hasattr(self._db, 'get_session'):
            return self._db.get_session()
        return self._db

    def _reset_if_new_day(self):
        """새로운 날이면 일일 통계를 리셋합니다."""
        today = dt.date.today()
        if self._daily_reset_date != today:
            self._daily_pnl = 0
            self._daily_trades = 0
            self._daily_reset_date = today

    def check_signal(self, signal: Signal, available_cash: int) -> Signal:
        """시그널이 리스크 기준을 만족하는지 검사합니다.

        통과하면 수량이 조정된 시그널을 반환하고,
        거부하면 HOLD 시그널을 반환합니다.
        """
        self._reset_if_new_day()

        if signal.type == SignalType.HOLD:
            return signal

        if signal.type == SignalType.SELL:
            return signal  # 매도는 항상 허용

        # ── 매수 시그널 검사 ──

        # 1. 일일 손실 한도 확인
        if self._daily_pnl <= self.config.max_daily_loss:
            logger.warning(
                "일일 손실 한도 도달: {}원 (한도: {}원)",
                self._daily_pnl, self.config.max_daily_loss,
            )
            return Signal(
                type=SignalType.HOLD, stock_code=signal.stock_code,
                price=signal.price,
                reason=f"일일 손실 한도 도달: {self._daily_pnl}원",
                strategy_name=signal.strategy_name,
            )

        # 2. 동시 보유 종목 수 확인
        current_positions = self.db.query(Position).count()
        if current_positions >= self.config.max_positions:
            logger.warning("최대 보유 종목 수 도달: {}", current_positions)
            return Signal(
                type=SignalType.HOLD, stock_code=signal.stock_code,
                price=signal.price,
                reason=f"최대 보유 종목 수 도달: {current_positions}",
                strategy_name=signal.strategy_name,
            )

        # 3. 동일 종목 중복 매수 방지
        existing = self.db.query(Position).filter_by(
            stock_code=signal.stock_code
        ).first()
        if existing:
            logger.warning("이미 보유 중인 종목: {}", signal.stock_code)
            return Signal(
                type=SignalType.HOLD, stock_code=signal.stock_code,
                price=signal.price,
                reason="이미 보유 중인 종목",
                strategy_name=signal.strategy_name,
            )

        # 3-1. 재무건전성 필터 (S-RIM 기반)
        reject_reason = self._check_financial_filter(signal.stock_code)
        if reject_reason:
            logger.info("재무 필터 거부 [{}]: {}", signal.stock_code, reject_reason)
            return Signal(
                type=SignalType.HOLD, stock_code=signal.stock_code,
                price=signal.price, reason=reject_reason,
                strategy_name=signal.strategy_name,
            )

        # 4. 포지션 사이징
        max_amount = min(self.config.max_position_size, available_cash)
        if signal.price <= 0:
            return Signal(
                type=SignalType.HOLD, stock_code=signal.stock_code,
                price=signal.price, reason="가격 정보 없음",
                strategy_name=signal.strategy_name,
            )

        quantity = max_amount // signal.price
        if quantity <= 0:
            logger.warning("매수 가능 수량 없음: 가용 현금={}원", available_cash)
            return Signal(
                type=SignalType.HOLD, stock_code=signal.stock_code,
                price=signal.price, reason="매수 가능 수량 부족",
                strategy_name=signal.strategy_name,
            )

        # 5. 최소 신뢰도 확인
        if signal.confidence < 0.5:
            logger.info(
                "신뢰도 부족: {} ({:.2f} < 0.5)",
                signal.stock_code, signal.confidence,
            )
            return Signal(
                type=SignalType.HOLD, stock_code=signal.stock_code,
                price=signal.price,
                reason=f"신뢰도 부족: {signal.confidence:.2f}",
                strategy_name=signal.strategy_name,
            )

        # 리스크 검사 통과 → 수량 설정
        adjusted = Signal(
            type=SignalType.BUY,
            stock_code=signal.stock_code,
            price=signal.price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            confidence=signal.confidence,
            reason=signal.reason,
            strategy_name=signal.strategy_name,
        )
        return adjusted

    def update_daily_pnl(self, pnl: int):
        """일일 손익을 업데이트합니다."""
        self._reset_if_new_day()
        self._daily_pnl += pnl
        self._daily_trades += 1

    def record_daily_summary(self):
        """일일 손익 요약을 DB에 기록합니다."""
        today = dt.date.today()
        trades = self.db.query(Trade).filter(
            Trade.executed_at >= dt.datetime.combine(today, dt.time.min),
            Trade.executed_at <= dt.datetime.combine(today, dt.time.max),
        ).all()

        if not trades:
            return

        sells = [t for t in trades if t.side == "SELL"]
        win_trades = [t for t in sells if t.pnl > 0]

        positions = self.db.query(Position).all()
        unrealized = sum(p.unrealized_pnl for p in positions)

        # MDD 계산 (간략)
        daily_records = self.db.query(DailyPnL).order_by(DailyPnL.date).all()
        portfolio_values = [r.portfolio_value for r in daily_records if r.portfolio_value > 0]
        max_dd = 0.0
        if portfolio_values:
            peak = portfolio_values[0]
            for pv in portfolio_values:
                peak = max(peak, pv)
                if peak > 0:
                    dd = (pv - peak) / peak * 100
                    max_dd = min(max_dd, dd)

        record = DailyPnL(
            date=today,
            total_pnl=self._daily_pnl,
            realized_pnl=sum(t.pnl for t in sells),
            unrealized_pnl=unrealized,
            total_fee=sum(t.fee for t in trades),
            total_tax=sum(t.tax for t in trades),
            trade_count=len(trades),
            win_count=len(win_trades),
            loss_count=len(sells) - len(win_trades),
            win_rate=len(win_trades) / len(sells) * 100 if sells else 0,
            max_drawdown=max_dd,
        )

        existing = self.db.query(DailyPnL).filter_by(date=today).first()
        if existing:
            for key, val in record.__dict__.items():
                if not key.startswith("_") and key != "id":
                    setattr(existing, key, val)
        else:
            self.db.add(record)

        self.db.commit()

    def _check_financial_filter(self, stock_code: str) -> Optional[str]:
        """재무건전성 필터를 검사합니다.

        RiskConfig의 min_roe, require_profitable, max_debt_ratio 설정에 따라
        해당 종목의 재무 데이터를 확인합니다.

        Returns:
            거부 사유 문자열 (통과 시 None)
        """
        # 필터 비활성화 시 통과
        if (self.config.min_roe <= 0
                and not self.config.require_profitable
                and self.config.max_debt_ratio <= 0):
            return None

        # 최근 재무 데이터 조회
        fin = (
            self.db.query(FinancialData)
            .filter_by(stock_code=stock_code)
            .order_by(FinancialData.fiscal_year.desc())
            .first()
        )

        if fin is None:
            # 재무 데이터 없으면 필터 통과 (데이터 미수집 상태 허용)
            return None

        # ROE 체크
        if self.config.min_roe > 0:
            roe_pct = fin.roe or 0.0
            if roe_pct < self.config.min_roe * 100:
                return f"ROE 부족: {roe_pct:.1f}% < {self.config.min_roe*100:.1f}%"

        # 영업이익 양수 체크
        if self.config.require_profitable:
            if (fin.operating_income or 0) <= 0:
                return f"영업적자 종목 (영업이익: {fin.operating_income:,}원)"

        # 부채비율 체크
        if self.config.max_debt_ratio > 0:
            debt_ratio = fin.debt_ratio or 0.0
            if debt_ratio > self.config.max_debt_ratio:
                return f"부채비율 초과: {debt_ratio:.1f}% > {self.config.max_debt_ratio:.1f}%"

        return None

    def get_portfolio_summary(self) -> dict:
        """포트폴리오 요약 정보를 반환합니다."""
        self._reset_if_new_day()
        positions = self.db.query(Position).all()

        total_investment = sum(p.avg_price * p.quantity for p in positions)
        total_current = sum(p.current_price * p.quantity for p in positions)
        total_unrealized = sum(p.unrealized_pnl for p in positions)

        return {
            "position_count": len(positions),
            "total_investment": total_investment,
            "total_current_value": total_current,
            "total_unrealized_pnl": total_unrealized,
            "daily_realized_pnl": self._daily_pnl,
            "daily_trade_count": self._daily_trades,
            "positions": [
                {
                    "stock_code": p.stock_code,
                    "stock_name": p.stock_name,
                    "quantity": p.quantity,
                    "avg_price": p.avg_price,
                    "current_price": p.current_price,
                    "pnl": p.unrealized_pnl,
                    "pnl_pct": p.unrealized_pnl_pct,
                    "strategy": p.strategy,
                }
                for p in positions
            ],
        }
