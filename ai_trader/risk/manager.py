"""리스크 관리 모듈

포지션 사이징, 일일 손실 제한, 동시 보유 제한, 포트폴리오 리스크 관리,
재무건전성 기반 필터링 (S-RIM 책 기반)

하워드 막스 마켓/신용 사이클 연동:
- LATE 사이클(70점+): max_position을 50% 축소
- 신용긴축(TIGHT): 추가 30% 축소
- 극단 공포 심리: 매수 기회 신호 (사이클 저점 시)
"""

import datetime as dt
from typing import Optional

import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session

from config.settings import RiskConfig, CandleMasterConfig, WizardConfig
from data.database import Position, Trade, DailyPnL, FinancialData
from strategies.base import Signal, SignalType


class RiskManager:
    """리스크 관리자

    하워드 막스 사이클 기반 포지션 조절:
    - LATE 사이클(70점+): max_position 50% 축소
    - 신용긴축(TIGHT): 추가 30% 축소
    - 극단 공포(EARLY+FEAR): 매수 기회 신호 (포지션 확대 허용)
    """

    def __init__(self, config: RiskConfig, db,
                 candle_master_config: Optional[CandleMasterConfig] = None,
                 wizard_config: Optional[WizardConfig] = None):
        self.config = config
        self._db = db
        self._daily_pnl: int = 0
        self._daily_trades: int = 0
        self._daily_reset_date: Optional[dt.date] = None
        # 캔들마스터 자금관리 설정
        self._cm_config = candle_master_config or CandleMasterConfig()
        # 잭 슈웨거 마법사 교훈 설정
        self._wizard_config = wizard_config or WizardConfig()
        self._trading_journal = None
        try:
            from features.wizard_discipline import (
                TradingJournal, ConfidenceScaler,
            )
            self._trading_journal = TradingJournal()
            self._confidence_scaler = ConfidenceScaler(
                max_scale=self._wizard_config.confidence_max_scale,
                min_scale=self._wizard_config.confidence_min_scale,
            )
        except ImportError:
            self._confidence_scaler = None

    @property
    def db(self):
        """DB 세션을 반환합니다.

        주의: 반환된 세션은 호출자가 close() 해야 합니다.
        check_signal / record_daily_summary 등 내부 메서드는
        _db_session() 컨텍스트 매니저를 사용하세요.
        """
        if hasattr(self._db, 'get_session'):
            return self._db.get_session()
        return self._db

    def _get_session(self):
        """세션을 반환합니다 (단일 진입점)."""
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

        # 2~3. DB 조회 (세션 명시적 관리)
        session = self._get_session()
        try:
            # 2. 동시 보유 종목 수 확인 (활성 포지션만)
            current_positions = session.query(Position).filter_by(
                is_active=True
            ).count()
            if current_positions >= self.config.max_positions:
                logger.warning("최대 보유 종목 수 도달: {}", current_positions)
                return Signal(
                    type=SignalType.HOLD, stock_code=signal.stock_code,
                    price=signal.price,
                    reason=f"최대 보유 종목 수 도달: {current_positions}",
                    strategy_name=signal.strategy_name,
                )

            # 3. 동일 종목 중복 매수 방지 (활성 포지션만)
            existing = session.query(Position).filter_by(
                stock_code=signal.stock_code, is_active=True
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
            reject_reason = self._check_financial_filter(signal.stock_code, session)
            if reject_reason:
                logger.info("재무 필터 거부 [{}]: {}", signal.stock_code, reject_reason)
                return Signal(
                    type=SignalType.HOLD, stock_code=signal.stock_code,
                    price=signal.price, reason=reject_reason,
                    strategy_name=signal.strategy_name,
                )
        finally:
            session.close()

        # 3-2. 물타기 방지 (캔들마스터)
        try:
            avg_down_reason = self._check_averaging_down(
                signal.stock_code, signal.price
            )
            if avg_down_reason:
                logger.info("물타기 방지 [{}]: {}", signal.stock_code, avg_down_reason)
                return Signal(
                    type=SignalType.HOLD, stock_code=signal.stock_code,
                    price=signal.price, reason=avg_down_reason,
                    strategy_name=signal.strategy_name,
                )
        except Exception:
            pass

        # 4. 포지션 사이징 (하워드 막스 사이클 기반 조정 포함)
        max_amount = self._apply_cycle_position_limit(
            self.config.max_position_size, available_cash, signal
        )

        # 4-1. 캔들마스터 포지션 한도 적용
        try:
            max_amount = self._apply_candle_master_position_limit(
                max_amount, available_cash, signal
            )
        except Exception:
            pass

        # 4-2. 슈웨거 마법사 교훈: 확신도 기반 포지션 스케일링 (교훈 37)
        # 주의: 확신도 스케일링은 기존 max_position_size 상한 내에서만 적용
        try:
            if (self._wizard_config.use_confidence_scaling
                    and self._confidence_scaler is not None):
                scaled = int(
                    self._confidence_scaler.scale_position(
                        float(max_amount), signal.confidence
                    )
                )
                # 원래 max_position_size와 available_cash 상한 유지
                max_amount = min(
                    scaled, self.config.max_position_size, available_cash
                )
                logger.debug(
                    "확신도 포지션 조정: conf={:.2f} → {}원",
                    signal.confidence, max_amount
                )
        except Exception:
            pass

        # 4-3. 슈웨거 마법사 교훈: 규율 점수 기반 제한 (교훈 7, 34)
        try:
            if (self._wizard_config.use_discipline_tracking
                    and self._trading_journal is not None):
                disc_score = self._trading_journal.get_discipline_score()
                if disc_score < self._wizard_config.discipline_min_score:
                    max_amount = int(max_amount * 0.5)
                    logger.info(
                        "규율 점수 부족({:.0f} < {:.0f}): 포지션 50% 제한",
                        disc_score,
                        self._wizard_config.discipline_min_score,
                    )
        except Exception:
            pass
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

    def _check_financial_filter(self, stock_code: str, session=None) -> Optional[str]:
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

        # 최근 재무 데이터 조회 (호출자 세션 재사용, 없으면 새 세션)
        own_session = session is None
        if own_session:
            session = self._get_session()
        try:
            fin = (
                session.query(FinancialData)
                .filter_by(stock_code=stock_code)
                .order_by(FinancialData.fiscal_year.desc())
                .first()
            )
        finally:
            if own_session:
                session.close()
                session = None

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

    def _apply_cycle_position_limit(
        self,
        max_position_size: int,
        available_cash: int,
        signal: Signal,
    ) -> int:
        """하워드 막스 사이클 기반 포지션 한도를 적용합니다.

        사이클 위치와 신용환경에 따라 max_position_size를 조정합니다.

        조정 규칙:
          1. LATE 사이클 (70점+): 50% 축소
             → 고점 구간에서는 방어적으로, 신규 매수 자제
          2. 신용긴축 (TIGHT): 추가 30% 축소
             → 신용창구가 닫힐 때는 레버리지 위험 증가
          3. 극단 공포 + EARLY 사이클: 정상 또는 소폭 확대
             → 모두가 두려워할 때가 매수 기회 (단, 분할 매수)

        CycleSignal이 없거나 오류 시 원래 max_position_size 반환.

        Args:
            max_position_size: 설정된 최대 포지션 크기 (원)
            available_cash: 가용 현금 (원)
            signal: 매수 시그널

        Returns:
            int: 조정된 최대 포지션 한도 (원)
        """
        try:
            # 사이클 필터가 비활성화된 경우 원래 값 반환
            if hasattr(self.config, 'use_cycle_filter'):
                if not self.config.use_cycle_filter:
                    return min(max_position_size, available_cash)

            # 시그널에 사이클 정보가 있으면 활용
            cycle_multiplier = 1.0
            credit_multiplier = 1.0
            fear_opportunity = False

            # 시그널 메타데이터에서 사이클 정보 추출 (있는 경우)
            if hasattr(signal, 'metadata') and signal.metadata:
                meta = signal.metadata

                # 사이클 점수 기반 조정
                cycle_score = meta.get('cycle_score', 50.0)
                cycle_phase = meta.get('cycle_phase', 'MID')
                sentiment = meta.get('sentiment', 'NEUTRAL')
                credit_status = meta.get('credit_status', 'NORMAL')

                # LATE 사이클: 50% 축소
                if cycle_score >= 70.0 or cycle_phase == 'LATE':
                    cycle_multiplier = 0.5
                    logger.info(
                        "사이클 LATE 구간: 포지션 50% 축소 "
                        "(score={:.1f})", cycle_score
                    )
                # EARLY + FEAR: 매수 기회 신호
                elif cycle_score <= 30.0 and sentiment == 'FEAR':
                    cycle_multiplier = 1.0  # 정상 유지 (분할 매수)
                    fear_opportunity = True
                    logger.info(
                        "극단 공포 + 사이클 EARLY: 매수 기회 신호 "
                        "(score={:.1f})", cycle_score
                    )

                # 신용긴축: 추가 30% 축소
                if credit_status == 'TIGHT':
                    credit_multiplier = 0.7
                    logger.info("신용환경 긴축(TIGHT): 포지션 추가 30% 축소")

            # 최종 조정 배수 (극단 회피: 0.3 하한)
            total_multiplier = max(
                cycle_multiplier * credit_multiplier, 0.3
            )

            adjusted_max = int(max_position_size * total_multiplier)
            result = min(adjusted_max, available_cash)

            if total_multiplier < 1.0:
                logger.debug(
                    "사이클 포지션 한도 조정: {}원 → {}원 (배수={:.2f})",
                    max_position_size, result, total_multiplier
                )

            return result

        except Exception as exc:
            logger.warning("사이클 포지션 조정 오류: {}", exc)
            return min(max_position_size, available_cash)

    def apply_cycle_adjustment(
        self,
        df: pd.DataFrame,
        base_max_position: Optional[int] = None,
    ) -> dict:
        """사이클 분석 결과로 포지션 한도를 동적으로 조정합니다.

        하워드 막스: "싸게 살 때는 공격적이어야 하지만
        비싸게 살 때는 후퇴해야 한다."

        이 메서드는 실시간 OHLCV 데이터를 받아 사이클/신용환경을
        직접 분석하고 포지션 한도를 계산합니다.

        Args:
            df: OHLCV DataFrame (최소 60봉 권장)
            base_max_position: 기준 포지션 한도 (None이면 config 값 사용)

        Returns:
            dict: {
                adjusted_max: 조정된 포지션 한도 (원),
                cycle_score: 사이클 점수,
                credit_status: 신용환경 상태,
                multiplier: 조정 배수,
                is_fear_opportunity: 극단 공포 매수 기회 여부,
                note: 분석 메모
            }
        """
        if base_max_position is None:
            base_max_position = self.config.max_position_size

        try:
            from features.market_cycle import MarketCycleAnalyzer
            from features.credit_cycle import CreditCycleAnalyzer

            # 사이클 분석
            cycle_analyzer = MarketCycleAnalyzer()
            cycle_signal = cycle_analyzer.analyze(df)

            # 신용사이클 분석
            credit_analyzer = CreditCycleAnalyzer()
            credit_result = credit_analyzer.analyze(df)
            credit_env = credit_result["credit_env"]

            # 포지션 조정 배수 계산
            multiplier = 1.0

            # 1. 사이클 기반 조정
            # LATE(70점+): 50% 축소
            if cycle_signal.cycle_score >= 70.0:
                multiplier *= 0.5
                cycle_note = f"LATE 사이클 (score={cycle_signal.cycle_score:.1f})"
            # EARLY(30점 이하): 정상 또는 확대
            elif cycle_signal.cycle_score <= 30.0:
                multiplier *= 1.0  # EARLY 구간은 정상 유지
                cycle_note = f"EARLY 사이클 (score={cycle_signal.cycle_score:.1f})"
            else:
                cycle_note = f"MID 사이클 (score={cycle_signal.cycle_score:.1f})"

            # 2. 신용환경 조정
            # 신용긴축(TIGHT): 추가 30% 축소
            if credit_env.status == 'TIGHT':
                multiplier *= 0.7
                credit_note = "신용긴축(TIGHT)"
            elif credit_env.status == 'EASY':
                multiplier *= 1.1  # 신용완화: 소폭 확대
                credit_note = "신용완화(EASY)"
            else:
                credit_note = "신용정상(NORMAL)"

            # 3. 극단 공포 매수 기회 확인
            is_fear_opportunity = (
                cycle_signal.cycle_score <= 30.0
                and cycle_signal.sentiment == 'FEAR'
                and credit_env.is_opportunity
            )

            # 3-1. 켄 피셔 시장 기억 분석 연동
            fisher_multiplier = 1.0
            fisher_note = ""
            try:
                from features.market_memory import MarketMemoryAnalyzer
                fisher_analyzer = MarketMemoryAnalyzer()
                fisher_sig = fisher_analyzer.analyze(df)
                fisher_multiplier = fisher_sig.position_multiplier
                fisher_note = f"켄피셔배수={fisher_multiplier:.2f}"
                if fisher_sig.note:
                    fisher_note += f"({fisher_sig.note[:40]})"
            except Exception:
                pass
            multiplier *= fisher_multiplier

            # 3-2. 강방천&존리 가치투자 분석 연동
            value_multiplier = 1.0
            value_note = ""
            try:
                from features.value_investor import ValueInvestorAnalyzer
                value_analyzer = ValueInvestorAnalyzer()
                value_sig = value_analyzer.analyze(df)
                value_multiplier = value_sig.position_multiplier
                value_note = f"가치투자배수={value_multiplier:.2f}"
                if value_sig.note:
                    value_note += f"({value_sig.note[:40]})"
            except Exception:
                pass
            multiplier *= value_multiplier

            # 3-3. 이남우 주식 품질 분석 연동
            quality_multiplier = 1.0
            quality_note = ""
            try:
                from features.stock_quality import StockQualityAnalyzer
                quality_analyzer = StockQualityAnalyzer()
                quality_sig = quality_analyzer.analyze(df)
                quality_multiplier = quality_sig.position_multiplier
                quality_note = f"품질배수={quality_multiplier:.2f}"
                if quality_sig.note:
                    quality_note += f"({quality_sig.note[:40]})"
            except Exception:
                pass
            multiplier *= quality_multiplier

            # 극단 회피: 배수 하한 0.3, 상한 1.3
            multiplier = round(float(max(min(multiplier, 1.3), 0.3)), 2)
            adjusted_max = int(base_max_position * multiplier)

            fisher_part = f" | {fisher_note}" if fisher_note else ""
            value_part = f" | {value_note}" if value_note else ""
            quality_part = f" | {quality_note}" if quality_note else ""
            note = (
                f"{cycle_note} | {credit_note}{fisher_part}"
                f"{value_part}{quality_part} | "
                f"배수={multiplier:.2f} | "
                f"공포기회={is_fear_opportunity}"
            )

            logger.info("사이클 포지션 조정: {}원 → {}원 | {}",
                       base_max_position, adjusted_max, note)

            return {
                "adjusted_max": adjusted_max,
                "cycle_score": cycle_signal.cycle_score,
                "credit_status": credit_env.status,
                "multiplier": multiplier,
                "is_fear_opportunity": is_fear_opportunity,
                "profit_probability": cycle_signal.profit_probability,
                "note": note,
            }

        except Exception as exc:
            logger.warning("사이클 조정 분석 오류: {}", exc)
            return {
                "adjusted_max": base_max_position,
                "cycle_score": 50.0,
                "credit_status": "NORMAL",
                "multiplier": 1.0,
                "is_fear_opportunity": False,
                "profit_probability": 0.5,
                "note": f"오류로 인한 기본값: {exc}",
            }

    def _apply_candle_master_position_limit(
        self,
        max_amount: int,
        available_cash: int,
        signal: Signal,
    ) -> int:
        """캔들마스터 자금관리 규칙을 적용합니다.

        캔들마스터 원칙:
        - 종목당 포지션 한도: 총 자산의 10% (소액 계좌는 20%)
        - 손절매 자동 설정: -10% 기본, 최대 -20%
        - 물타기(추가 매수로 평균단가 낮추기) 금지

        Args:
            max_amount: 기존 최대 포지션 금액
            available_cash: 가용 현금
            signal: 매수 시그널

        Returns:
            int: 캔들마스터 규칙이 적용된 최대 포지션 금액
        """
        try:
            cm = self._cm_config

            # 총 자산 추산 (가용 현금 + 현재 투자금)
            session = self._get_session()
            try:
                positions = session.query(Position).filter_by(
                    is_active=True
                ).all()
                total_invested = sum(
                    p.avg_price * p.quantity for p in positions
                )
            finally:
                session.close()

            total_assets = available_cash + total_invested

            # 소액/일반 계좌에 따른 종목당 비중 결정
            if total_assets <= cm.small_account_threshold:
                position_pct = cm.small_account_pct  # 소액: 20%
            else:
                position_pct = cm.max_position_pct   # 일반: 10%

            # 캔들마스터 포지션 한도
            cm_max = int(total_assets * position_pct)

            # 기존 한도와 캔들마스터 한도 중 작은 값 적용
            result = min(max_amount, cm_max, available_cash)

            if result < max_amount:
                logger.info(
                    "캔들마스터 포지션 한도 적용: {}원 → {}원 "
                    "(총자산 {}원, 비중 {:.0%})",
                    max_amount, result, total_assets, position_pct
                )

            return result

        except Exception as exc:
            logger.warning("캔들마스터 포지션 한도 적용 오류: {}", exc)
            return min(max_amount, available_cash)

    def _check_averaging_down(
        self,
        stock_code: str,
        current_price: int,
    ) -> Optional[str]:
        """물타기 방지 검사

        캔들마스터 원칙: 하락 중인 종목에 추가 매수(물타기) 금지.
        기존 포지션의 평균 매입가보다 현재가가 낮으면 추가 매수 차단.

        Args:
            stock_code: 종목 코드
            current_price: 현재 가격

        Returns:
            거부 사유 문자열 (물타기 아니면 None)
        """
        try:
            if not self._cm_config.no_averaging_down:
                return None

            session = self._get_session()
            try:
                existing = session.query(Position).filter_by(
                    stock_code=stock_code, is_active=True
                ).first()
            finally:
                session.close()

            if existing is None:
                return None  # 기존 포지션 없음 → 물타기 아님

            if current_price < existing.avg_price:
                return (
                    f"물타기 방지: 현재가 {current_price:,}원 < "
                    f"평균매입가 {existing.avg_price:,}원 "
                    f"(캔들마스터 원칙: 하락 중 추가 매수 금지)"
                )

            return None

        except Exception:
            return None  # 오류 시 통과 (안전)

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
