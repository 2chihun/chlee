"""?뵳?딅뮞?寃? ?꽴??뵳? 筌뤴뫀諭?







?猷뤄쭪????? ?沅???뵠筌??, ??뵬??뵬 ??????뼄 ??젫?釉?, ?猷???뻻 癰귣똻??? ??젫?釉?, ?猷???뱜?琉껆뵳?딆궎 ?뵳?딅뮞?寃? ?꽴??뵳?,



??삺?눧?떯援???읈?苑? 疫꿸퀡而? ?釉??苑ｏ쭕? (S-RIM 筌?? 疫꿸퀡而?)







?釉???뜖?諭? 筌띾맩?뮞 筌띾뜆?룇/??뻿??뒠 ?沅???뵠?寃? ?肉??猷?:



- LATE ?沅???뵠?寃?(70??젎+): max_position??뱽 50% ?빊類ㅻ꺖



- ??뻿??뒠疫뀀똻?뀧(TIGHT): ?빊遺??? 30% ?빊類ㅻ꺖



- 域밸갭?뼊 ??⑤벏猷? ??뼎?뵳?: 筌띲끉?땾 疫꿸퀬?돳 ??뻿??깈 (?沅???뵠?寃? ?????젎 ??뻻)



"""







import datetime as dt



from typing import Optional







import pandas as pd



from loguru import logger



from sqlalchemy.orm import Session







from config.settings import RiskConfig, CandleMasterConfig, WizardConfig



from data.database import Position, Trade, DailyPnL, FinancialData



from strategies.base import Signal, SignalType
from risk.tail_risk import TalebRiskAnalyzer











class RiskManager:



    """?뵳?딅뮞?寃? ?꽴??뵳?딆쁽







    ?釉???뜖?諭? 筌띾맩?뮞 ?沅???뵠?寃? 疫꿸퀡而? ?猷뤄쭪????? 鈺곌퀣?쟿:



    - LATE ?沅???뵠?寃?(70??젎+): max_position 50% ?빊類ㅻ꺖



    - ??뻿??뒠疫뀀똻?뀧(TIGHT): ?빊遺??? 30% ?빊類ㅻ꺖



    - 域밸갭?뼊 ??⑤벏猷?(EARLY+FEAR): 筌띲끉?땾 疫꿸퀬?돳 ??뻿??깈 (?猷뤄쭪????? ??넇??? ?肉???뒠)



    """







    def __init__(self, config: RiskConfig, db,



                 candle_master_config: Optional[CandleMasterConfig] = None,



                 wizard_config: Optional[WizardConfig] = None):



        self.config = config



        self._db = db



        self._daily_pnl: int = 0



        self._daily_trades: int = 0



        self._daily_reset_date: Optional[dt.date] = None



        # 筌?遺얜굶筌띾뜆?뮞?苑? ??쁽疫뀀뜃???뵳? ?苑???젟



        self._cm_config = candle_master_config or CandleMasterConfig()



        # ??삻 ??뭼??띃椰?? 筌띾뜄苡??沅? ?뤃癒곗뜒 ?苑???젟



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

        # 탈레브 꼬리 위험 분석기
        try:
            self._taleb = TalebRiskAnalyzer(lookback=252)
        except Exception:
            self._taleb = None







    @property



    def db(self):



        """DB ?苑???????뱽 獄쏆꼹?넎?鍮???빍??뼄.







        雅뚯눘?벥: 獄쏆꼹?넎?留? ?苑???????? ??깈?빊?뮇?쁽揶?? close() ?鍮??鍮? ?鍮???빍??뼄.



        check_signal / record_daily_summary ?踰? ?沅↓겫? 筌롫뗄苑??諭???뮉



        _db_session() ??뚢뫂?????뮞??뱜 筌띲끇?빍????몴? ?沅???뒠?釉??苑???뒄.



        """



        if hasattr(self._db, 'get_session'):



            return self._db.get_session()



        return self._db







    def _get_session(self):



        """?苑???????뱽 獄쏆꼹?넎?鍮???빍??뼄 (??뼊??뵬 筌욊쑴?뿯??젎)."""



        if hasattr(self._db, 'get_session'):



            return self._db.get_session()



        return self._db







    def _reset_if_new_day(self):



        """?源됪에?뮇?뒲 ?沅???뵠筌?? ??뵬??뵬 ??꽰??④쑬?? ?뵳?딅???鍮???빍??뼄."""



        today = dt.date.today()



        if self._daily_reset_date != today:



            self._daily_pnl = 0



            self._daily_trades = 0



            self._daily_reset_date = today







    def check_signal(self, signal: Signal, available_cash: int) -> Signal:



        """??뻻域밸챶瑗???뵠 ?뵳?딅뮞?寃? 疫꿸퀣????뱽 筌띾슣????釉???뮉筌?? 野???沅??鍮???빍??뼄.







        ??꽰??⑥눛釉?筌?? ??땾??쎗??뵠 鈺곌퀣?젟?留? ??뻻域밸챶瑗???뱽 獄쏆꼹?넎?釉?????,



        椰꾧퀡???釉?筌?? HOLD ??뻻域밸챶瑗???뱽 獄쏆꼹?넎?鍮???빍??뼄.



        """



        self._reset_if_new_day()







        if signal.type == SignalType.HOLD:



            return signal







        if signal.type == SignalType.SELL:



            return signal  # 筌띲끇猷???뮉 ?鍮??湲? ?肉???뒠







        # ?????? 筌띲끉?땾 ??뻻域밸챶瑗? 野???沅? ??????







        # 1. ??뵬??뵬 ??????뼄 ?釉??猷? ??넇??뵥



        if self._daily_pnl <= self.config.max_daily_loss:



            logger.warning(



                "??뵬??뵬 ??????뼄 ?釉??猷? ?猷???뼎: {}??뜚 (?釉??猷?: {}??뜚)",



                self._daily_pnl, self.config.max_daily_loss,



            )



            return Signal(



                type=SignalType.HOLD, stock_code=signal.stock_code,



                price=signal.price,



                reason=f"??뵬??뵬 ??????뼄 ?釉??猷? ?猷???뼎: {self._daily_pnl}??뜚",



                strategy_name=signal.strategy_name,



            )







        # 2~3. DB 鈺곌퀬?돳 (?苑????? 筌뤿굞?뻻??읅 ?꽴??뵳?)



        session = self._get_session()



        try:



            # 2. ?猷???뻻 癰귣똻??? ?넫?굝?걠 ??땾 ??넇??뵥 (??넞?苑? ?猷뤄쭪????∽쭕?)



            current_positions = session.query(Position).filter_by(



                is_active=True



            ).count()



            if current_positions >= self.config.max_positions:



                logger.warning("筌ㅼ뮆?? 癰귣똻??? ?넫?굝?걠 ??땾 ?猷???뼎: {}", current_positions)



                return Signal(



                    type=SignalType.HOLD, stock_code=signal.stock_code,



                    price=signal.price,



                    reason=f"筌ㅼ뮆?? 癰귣똻??? ?넫?굝?걠 ??땾 ?猷???뼎: {current_positions}",



                    strategy_name=signal.strategy_name,



                )







            # 3. ?猷???뵬 ?넫?굝?걠 餓λ쵎?궗 筌띲끉?땾 獄쎻뫗?? (??넞?苑? ?猷뤄쭪????∽쭕?)



            existing = session.query(Position).filter_by(



                stock_code=signal.stock_code, is_active=True



            ).first()



            if existing:



                logger.warning("??뵠沃?? 癰귣똻??? 餓λ쵐?뵥 ?넫?굝?걠: {}", signal.stock_code)



                return Signal(



                    type=SignalType.HOLD, stock_code=signal.stock_code,



                    price=signal.price,



                    reason="??뵠沃?? 癰귣똻??? 餓λ쵐?뵥 ?넫?굝?걠",



                    strategy_name=signal.strategy_name,



                )







            # 3-1. ??삺?눧?떯援???읈?苑? ?釉??苑? (S-RIM 疫꿸퀡而?)



            reject_reason = self._check_financial_filter(signal.stock_code, session)



            if reject_reason:



                logger.info("??삺?눧? ?釉??苑? 椰꾧퀡?? [{}]: {}", signal.stock_code, reject_reason)



                return Signal(



                    type=SignalType.HOLD, stock_code=signal.stock_code,



                    price=signal.price, reason=reject_reason,



                    strategy_name=signal.strategy_name,



                )



        finally:



            session.close()







        # 3-2. ?눧?눛??疫?? 獄쎻뫗?? (筌?遺얜굶筌띾뜆?뮞?苑?)



        try:



            avg_down_reason = self._check_averaging_down(



                signal.stock_code, signal.price



            )



            if avg_down_reason:



                logger.info("?눧?눛??疫?? 獄쎻뫗?? [{}]: {}", signal.stock_code, avg_down_reason)



                return Signal(



                    type=SignalType.HOLD, stock_code=signal.stock_code,



                    price=signal.price, reason=avg_down_reason,



                    strategy_name=signal.strategy_name,



                )



        except Exception:



            pass







        # 4. ?猷뤄쭪????? ?沅???뵠筌?? (?釉???뜖?諭? 筌띾맩?뮞 ?沅???뵠?寃? 疫꿸퀡而? 鈺곌퀣?젟 ?猷??釉?)



        max_amount = self._apply_cycle_position_limit(



            self.config.max_position_size, available_cash, signal



        )







        # 4-1. 筌?遺얜굶筌띾뜆?뮞?苑? ?猷뤄쭪????? ?釉??猷? ??읅??뒠



        try:



            max_amount = self._apply_candle_master_position_limit(



                max_amount, available_cash, signal



            )



        except Exception:



            pass







        # 4-2. ??뭼??띃椰?? 筌띾뜄苡??沅? ?뤃癒곗뜒: ??넇??뻿?猷? 疫꿸퀡而? ?猷뤄쭪????? ??뮞??????뵬筌?? (?뤃癒곗뜒 37)



        # 雅뚯눘?벥: ??넇??뻿?猷? ??뮞??????뵬筌띻낯?? 疫꿸퀣??? max_position_size ?湲??釉? ?沅??肉??苑뚳쭕? ??읅??뒠



        try:



            if (self._wizard_config.use_confidence_scaling



                    and self._confidence_scaler is not None):



                scaled = int(



                    self._confidence_scaler.scale_position(



                        float(max_amount), signal.confidence



                    )



                )



                # ??뜚??삋 max_position_size??? available_cash ?湲??釉? ????筌??



                max_amount = min(



                    scaled, self.config.max_position_size, available_cash



                )



                logger.debug(



                    "??넇??뻿?猷? ?猷뤄쭪????? 鈺곌퀣?젟: conf={:.2f} ??꼥 {}??뜚",



                    signal.confidence, max_amount



                )



        except Exception:



            pass







        # 4-3. ??뭼??띃椰?? 筌띾뜄苡??沅? ?뤃癒곗뜒: 域뱀뮇?몛 ??젎??땾 疫꿸퀡而? ??젫?釉? (?뤃癒곗뜒 7, 34)



        try:



            if (self._wizard_config.use_discipline_tracking



                    and self._trading_journal is not None):



                disc_score = self._trading_journal.get_discipline_score()



                if disc_score < self._wizard_config.discipline_min_score:



                    max_amount = int(max_amount * 0.5)



                    logger.info(



                        "域뱀뮇?몛 ??젎??땾 ?겫?鈺??({:.0f} < {:.0f}): ?猷뤄쭪????? 50% ??젫?釉?",



                        disc_score,



                        self._wizard_config.discipline_min_score,



                    )



        except Exception:



            pass







        # 4-4. ML Overlay: VPIN ?쑀?룞?꽦 由ъ뒪?겕 (AFML Ch19)



        try:



            ml_liquidity = None



            if hasattr(signal, 'metadata') and signal.metadata:



                ml_liquidity = signal.metadata.get('ml_liquidity_risk')



            if ml_liquidity is not None and ml_liquidity > 0.7:



                max_amount = int(max_amount * 0.5)



                logger.info(



                    "VPIN ?쑀?룞?꽦 由ъ뒪?겕 ?넂?쓬({:.2f} > 0.7): ?룷吏??뀡 50% 異뺤냼",



                    ml_liquidity,



                )



        except Exception:



            pass







        # 4-5. ML Overlay: ?뿏?듃濡쒗뵾 ?븘?꽣 (AFML Ch18)



        try:



            ml_entropy = None



            if hasattr(signal, 'metadata') and signal.metadata:



                ml_entropy = signal.metadata.get('ml_entropy')



            if ml_entropy is not None and ml_entropy > 4.0:



                max_amount = int(max_amount * 0.7)



                logger.info(



                    "?뿏?듃濡쒗뵾 ?넂?쓬({:.2f} > 4.0): ?룷吏??뀡 30% 異뺤냼",



                    ml_entropy,



                )



        except Exception:



            pass







        # 4-6. ML Overlay: Bet Sizing (AFML Ch10)



        try:



            ml_bet_size = None



            if hasattr(signal, 'metadata') and signal.metadata:



                ml_bet_size = signal.metadata.get('ml_bet_size')



            if ml_bet_size is not None and 0.0 < ml_bet_size < 1.0:



                max_amount = int(max_amount * ml_bet_size)



                logger.debug(



                    "ML bet sizing ?쟻?슜: bet_size={:.2f} ?넂 {}?썝",



                    ml_bet_size, max_amount,



                )



        except Exception:



            pass



        # 4-7. 그레이엄 자산배분 보정 (AssetAllocationGuide)
        try:
            equity_pct = None
            if hasattr(signal, 'metadata') and signal.metadata:
                equity_pct = signal.metadata.get('graham_equity_pct')
            if equity_pct is not None and equity_pct < 50.0:
                ratio = equity_pct / 50.0
                max_amount = int(max_amount * ratio)
                logger.info(
                    "그레이엄 자산배분: 주식 %.0f%% -> 포지션 %.0f%% 적용",
                    equity_pct, ratio * 100,
                )
        except Exception:
            pass

        # 4-8. 탈레브 꼬리 위험 보정 (TalebRiskAnalyzer)
        try:
            if self._taleb is not None and df is not None and len(df) > 0:
                taleb_signal = self._taleb.analyze(df)
                if taleb_signal.precautionary_block:
                    logger.warning(
                        "탈레브 꼬리 위험: 사전예방 차단 활성 -> max_amount=0"
                    )
                    max_amount = 0
                else:
                    prev = max_amount
                    max_amount = int(max_amount * taleb_signal.position_scale)
                    if taleb_signal.position_scale < 1.0:
                        logger.info(
                            "탈레브 꼬리 위험: scale=%.2f -> %d->%d",
                            taleb_signal.position_scale, prev, max_amount,
                        )
                # Store metadata for downstream
                if hasattr(signal, "metadata") and signal.metadata is not None:
                    signal.metadata["taleb_block"] = taleb_signal.precautionary_block
                    signal.metadata["taleb_scale"] = taleb_signal.position_scale
                elif hasattr(signal, "metadata"):
                    signal.metadata = {
                        "taleb_block": taleb_signal.precautionary_block,
                        "taleb_scale": taleb_signal.position_scale,
                    }
        except Exception:
            pass



        if signal.price <= 0:



            return Signal(



                type=SignalType.HOLD, stock_code=signal.stock_code,



                price=signal.price, reason="媛?寃? ?젙蹂? ?뾾?쓬",



                strategy_name=signal.strategy_name,



            )







        quantity = max_amount // signal.price



        if quantity <= 0:



            logger.warning("筌띲끉?땾 揶????뮟 ??땾??쎗 ?毓???벉: 揶????뒠 ??겱疫??={}??뜚", available_cash)



            return Signal(



                type=SignalType.HOLD, stock_code=signal.stock_code,



                price=signal.price, reason="筌띲끉?땾 揶????뮟 ??땾??쎗 ?겫?鈺??",



                strategy_name=signal.strategy_name,



            )







        # 5. 筌ㅼ뮇?꺖 ??뻿?뙳怨뺣즲 ??넇??뵥



        if signal.confidence < 0.5:



            logger.info(



                "??뻿?뙳怨뺣즲 ?겫?鈺??: {} ({:.2f} < 0.5)",



                signal.stock_code, signal.confidence,



            )



            return Signal(



                type=SignalType.HOLD, stock_code=signal.stock_code,



                price=signal.price,



                reason=f"??뻿?뙳怨뺣즲 ?겫?鈺??: {signal.confidence:.2f}",



                strategy_name=signal.strategy_name,



            )







        # ?뵳?딅뮞?寃? 野???沅? ??꽰???? ??꼥 ??땾??쎗 ?苑???젟



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



        """??뵬??뵬 ??????뵡??뱽 ?毓???쑓??뵠??뱜?鍮???빍??뼄."""



        self._reset_if_new_day()



        self._daily_pnl += pnl



        self._daily_trades += 1







    def record_daily_summary(self):



        """??뵬??뵬 ??????뵡 ??뒄?鍮???뱽 DB?肉? 疫꿸퀡以??鍮???빍??뼄."""



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







        # MDD ??④쑴沅? (揶쏄쑬?셽)



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



        """??삺?눧?떯援???읈?苑? ?釉??苑ｇ몴? 野???沅??鍮???빍??뼄.







        RiskConfig??벥 min_roe, require_profitable, max_debt_ratio ?苑???젟?肉? ?逾???뵬



        ?鍮???뼣 ?넫?굝?걠??벥 ??삺?눧? ??쑓??뵠?苑ｇ몴? ??넇??뵥?鍮???빍??뼄.







        Returns:



            椰꾧퀡?? ?沅????? ?눧紐꾩쁽?肉? (??꽰???? ??뻻 None)



        """



        # ?釉??苑? ?뜮袁れ넞?苑???넅 ??뻻 ??꽰????



        if (self.config.min_roe <= 0



                and not self.config.require_profitable



                and self.config.max_debt_ratio <= 0):



            return None







        # 筌ㅼ뮄?젏 ??삺?눧? ??쑓??뵠?苑? 鈺곌퀬?돳 (??깈?빊?뮇?쁽 ?苑????? ??삺?沅???뒠, ?毓???몵筌?? ?源? ?苑?????)



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



            # ??삺?눧? ??쑓??뵠?苑? ?毓???몵筌?? ?釉??苑? ??꽰???? (??쑓??뵠?苑? 沃섎챷?땾筌?? ?湲??源? ?肉???뒠)



            return None







        # ROE 筌ｋ똾寃?



        if self.config.min_roe > 0:



            roe_pct = fin.roe or 0.0



            if roe_pct < self.config.min_roe * 100:



                return f"ROE ?겫?鈺??: {roe_pct:.1f}% < {self.config.min_roe*100:.1f}%"







        # ??겫?毓???뵠??뵡 ?堉???땾 筌ｋ똾寃?



        if self.config.require_profitable:



            if (fin.operating_income or 0) <= 0:



                return f"??겫?毓???읅??쁽 ?넫?굝?걠 (??겫?毓???뵠??뵡: {fin.operating_income:,}??뜚)"







        # ?겫?筌?袁⑦돩??몛 筌ｋ똾寃?



        if self.config.max_debt_ratio > 0:



            debt_ratio = fin.debt_ratio or 0.0



            if debt_ratio > self.config.max_debt_ratio:



                return f"?겫?筌?袁⑦돩??몛 ?룯?뜃?궢: {debt_ratio:.1f}% > {self.config.max_debt_ratio:.1f}%"







        return None







    def _apply_cycle_position_limit(



        self,



        max_position_size: int,



        available_cash: int,



        signal: Signal,



    ) -> int:



        """?釉???뜖?諭? 筌띾맩?뮞 ?沅???뵠?寃? 疫꿸퀡而? ?猷뤄쭪????? ?釉??猷꾤몴? ??읅??뒠?鍮???빍??뼄.







        ?沅???뵠?寃? ??맄燁살꼷?? ??뻿??뒠??넎野껋럩肉? ?逾???뵬 max_position_size?몴? 鈺곌퀣?젟?鍮???빍??뼄.







        鈺곌퀣?젟 域뱀뮇?뒅:



          1. LATE ?沅???뵠?寃? (70??젎+): 50% ?빊類ㅻ꺖



             ??꼥 ??⑥쥙?젎 ?뤃?덉퍢?肉??苑???뮉 獄쎻뫗堉???읅??몵嚥??, ??뻿域?? 筌띲끉?땾 ??쁽??젫



          2. ??뻿??뒠疫뀀똻?뀧 (TIGHT): ?빊遺??? 30% ?빊類ㅻ꺖



             ??꼥 ??뻿??뒠筌≪럡?럡揶?? ??뼍??쁽 ?釉???뮉 ??쟿甕곌쑬?봺筌?? ??맄?肉? 筌앹빓??



          3. 域밸갭?뼊 ??⑤벏猷? + EARLY ?沅???뵠?寃?: ??젟?湲? ??굢??뮉 ??꺖?猷? ??넇???



             ??꼥 筌뤴뫀紐℡첎? ?紐???젻??뜖?釉? ?釉ｅ첎? 筌띲끉?땾 疫꿸퀬?돳 (??뼊, ?겫袁る막 筌띲끉?땾)







        CycleSignal??뵠 ?毓얍쳞怨뺢돌 ??궎?몴? ??뻻 ??뜚??삋 max_position_size 獄쏆꼹?넎.







        Args:



            max_position_size: ?苑???젟?留? 筌ㅼ뮆?? ?猷뤄쭪????? ?寃뺞묾? (??뜚)



            available_cash: 揶????뒠 ??겱疫?? (??뜚)



            signal: 筌띲끉?땾 ??뻻域밸챶瑗?







        Returns:



            int: 鈺곌퀣?젟?留? 筌ㅼ뮆?? ?猷뤄쭪????? ?釉??猷? (??뜚)



        """



        try:



            # ?沅???뵠?寃? ?釉??苑ｅ첎? ?뜮袁れ넞?苑???넅?留? 野껋럩?뒭 ??뜚??삋 揶?? 獄쏆꼹?넎



            if hasattr(self.config, 'use_cycle_filter'):



                if not self.config.use_cycle_filter:



                    return min(max_position_size, available_cash)







            # ??뻻域밸챶瑗??肉? ?沅???뵠?寃? ??젟癰귣떯?? ??뿳??몵筌?? ??넞??뒠



            cycle_multiplier = 1.0



            credit_multiplier = 1.0



            fear_opportunity = False







            # ??뻻域밸챶瑗? 筌롫??????쑓??뵠?苑??肉??苑? ?沅???뵠?寃? ??젟癰?? ?빊遺욱뀱 (??뿳??뮉 野껋럩?뒭)



            if hasattr(signal, 'metadata') and signal.metadata:



                meta = signal.metadata







                # ?沅???뵠?寃? ??젎??땾 疫꿸퀡而? 鈺곌퀣?젟



                cycle_score = meta.get('cycle_score', 50.0)



                cycle_phase = meta.get('cycle_phase', 'MID')



                sentiment = meta.get('sentiment', 'NEUTRAL')



                credit_status = meta.get('credit_status', 'NORMAL')







                # LATE ?沅???뵠?寃?: 50% ?빊類ㅻ꺖



                if cycle_score >= 70.0 or cycle_phase == 'LATE':



                    cycle_multiplier = 0.5



                    logger.info(



                        "?沅???뵠?寃? LATE ?뤃?덉퍢: ?猷뤄쭪????? 50% ?빊類ㅻ꺖 "



                        "(score={:.1f})", cycle_score



                    )



                # EARLY + FEAR: 筌띲끉?땾 疫꿸퀬?돳 ??뻿??깈



                elif cycle_score <= 30.0 and sentiment == 'FEAR':



                    cycle_multiplier = 1.0  # ??젟?湲? ????筌?? (?겫袁る막 筌띲끉?땾)



                    fear_opportunity = True



                    logger.info(



                        "域밸갭?뼊 ??⑤벏猷? + ?沅???뵠?寃? EARLY: 筌띲끉?땾 疫꿸퀬?돳 ??뻿??깈 "



                        "(score={:.1f})", cycle_score



                    )







                # ??뻿??뒠疫뀀똻?뀧: ?빊遺??? 30% ?빊類ㅻ꺖



                if credit_status == 'TIGHT':



                    credit_multiplier = 0.7



                    logger.info("??뻿??뒠??넎野?? 疫뀀똻?뀧(TIGHT): ?猷뤄쭪????? ?빊遺??? 30% ?빊類ㅻ꺖")







            # 筌ㅼ뮇伊? 鈺곌퀣?젟 獄쏄퀣?땾 (域밸갭?뼊 ??돳?逾?: 0.3 ?釉??釉?)



            total_multiplier = max(



                cycle_multiplier * credit_multiplier, 0.3



            )







            adjusted_max = int(max_position_size * total_multiplier)



            result = min(adjusted_max, available_cash)







            if total_multiplier < 1.0:



                logger.debug(



                    "?沅???뵠?寃? ?猷뤄쭪????? ?釉??猷? 鈺곌퀣?젟: {}??뜚 ??꼥 {}??뜚 (獄쏄퀣?땾={:.2f})",



                    max_position_size, result, total_multiplier



                )







            return result







        except Exception as exc:



            logger.warning("?沅???뵠?寃? ?猷뤄쭪????? 鈺곌퀣?젟 ??궎?몴?: {}", exc)



            return min(max_position_size, available_cash)







    def apply_cycle_adjustment(



        self,



        df: pd.DataFrame,



        base_max_position: Optional[int] = None,



    ) -> dict:



        """?沅???뵠?寃? ?겫袁⑷퐤 野껉퀗?궢嚥?? ?猷뤄쭪????? ?釉??猷꾤몴? ?猷???읅??몵嚥?? 鈺곌퀣?젟?鍮???빍??뼄.







        ?釉???뜖?諭? 筌띾맩?뮞: "??뼢野?? ?沅? ?釉???뮉 ??⑤벀爰???읅??뵠?堉??鍮? ?釉?筌??筌??



        ?뜮袁⑸뼢野?? ?沅? ?釉???뮉 ??뜎??닚?鍮??鍮? ?釉???뼄."







        ??뵠 筌롫뗄苑??諭???뮉 ??뼄??뻻揶?? OHLCV ??쑓??뵠?苑ｇ몴? 獄쏆룇釉? ?沅???뵠?寃?/??뻿??뒠??넎野껋럩?뱽



        筌욊낯?젔 ?겫袁⑷퐤?釉????? ?猷뤄쭪????? ?釉??猷꾤몴? ??④쑴沅??鍮???빍??뼄.







        Args:



            df: OHLCV DataFrame (筌ㅼ뮇?꺖 60?걡? 亦낅슣?삢)



            base_max_position: 疫꿸퀣?? ?猷뤄쭪????? ?釉??猷? (None??뵠筌?? config 揶?? ?沅???뒠)







        Returns:



            dict: {



                adjusted_max: 鈺곌퀣?젟?留? ?猷뤄쭪????? ?釉??猷? (??뜚),



                cycle_score: ?沅???뵠?寃? ??젎??땾,



                credit_status: ??뻿??뒠??넎野?? ?湲??源?,



                multiplier: 鈺곌퀣?젟 獄쏄퀣?땾,



                is_fear_opportunity: 域밸갭?뼊 ??⑤벏猷? 筌띲끉?땾 疫꿸퀬?돳 ?肉ч겫?,



                note: ?겫袁⑷퐤 筌롫뗀?걟



            }



        """



        if base_max_position is None:



            base_max_position = self.config.max_position_size







        try:



            from features.market_cycle import MarketCycleAnalyzer



            from features.credit_cycle import CreditCycleAnalyzer







            # ?沅???뵠?寃? ?겫袁⑷퐤



            cycle_analyzer = MarketCycleAnalyzer()



            cycle_signal = cycle_analyzer.analyze(df)







            # ??뻿??뒠?沅???뵠?寃? ?겫袁⑷퐤



            credit_analyzer = CreditCycleAnalyzer()



            credit_result = credit_analyzer.analyze(df)



            credit_env = credit_result["credit_env"]







            # ?猷뤄쭪????? 鈺곌퀣?젟 獄쏄퀣?땾 ??④쑴沅?



            multiplier = 1.0







            # 1. ?沅???뵠?寃? 疫꿸퀡而? 鈺곌퀣?젟



            # LATE(70??젎+): 50% ?빊類ㅻ꺖



            if cycle_signal.cycle_score >= 70.0:



                multiplier *= 0.5



                cycle_note = f"LATE ?沅???뵠?寃? (score={cycle_signal.cycle_score:.1f})"



            # EARLY(30??젎 ??뵠?釉?): ??젟?湲? ??굢??뮉 ??넇???



            elif cycle_signal.cycle_score <= 30.0:



                multiplier *= 1.0  # EARLY ?뤃?덉퍢??? ??젟?湲? ????筌??



                cycle_note = f"EARLY ?沅???뵠?寃? (score={cycle_signal.cycle_score:.1f})"



            else:



                cycle_note = f"MID ?沅???뵠?寃? (score={cycle_signal.cycle_score:.1f})"







            # 2. ??뻿??뒠??넎野?? 鈺곌퀣?젟



            # ??뻿??뒠疫뀀똻?뀧(TIGHT): ?빊遺??? 30% ?빊類ㅻ꺖



            if credit_env.status == 'TIGHT':



                multiplier *= 0.7



                credit_note = "??뻿??뒠疫뀀똻?뀧(TIGHT)"



            elif credit_env.status == 'EASY':



                multiplier *= 1.1  # ??뻿??뒠??끏??넅: ??꺖?猷? ??넇???



                credit_note = "??뻿??뒠??끏??넅(EASY)"



            else:



                credit_note = "??뻿??뒠??젟?湲?(NORMAL)"







            # 3. 域밸갭?뼊 ??⑤벏猷? 筌띲끉?땾 疫꿸퀬?돳 ??넇??뵥



            is_fear_opportunity = (



                cycle_signal.cycle_score <= 30.0



                and cycle_signal.sentiment == 'FEAR'



                and credit_env.is_opportunity



            )







            # 3-1. ???? ?逾????? ??뻻??삢 疫꿸퀣堉? ?겫袁⑷퐤 ?肉??猷?



            fisher_multiplier = 1.0



            fisher_note = ""



            try:



                from features.market_memory import MarketMemoryAnalyzer



                fisher_analyzer = MarketMemoryAnalyzer()



                fisher_sig = fisher_analyzer.analyze(df)



                fisher_multiplier = fisher_sig.position_multiplier



                fisher_note = f"??녠쑵逾????쏂쳸怨쀫땾={fisher_multiplier:.2f}"



                if fisher_sig.note:



                    fisher_note += f"({fisher_sig.note[:40]})"



            except Exception:



                pass



            multiplier *= fisher_multiplier







            # 3-2. 揶쏅베媛묕㎗?&鈺곕???봺 揶??燁살꼹?떮??쁽 ?겫袁⑷퐤 ?肉??猷?



            value_multiplier = 1.0



            value_note = ""



            try:



                from features.value_investor import ValueInvestorAnalyzer



                value_analyzer = ValueInvestorAnalyzer()



                value_sig = value_analyzer.analyze(df)



                value_multiplier = value_sig.position_multiplier



                value_note = f"揶??燁살꼹?떮??쁽獄쏄퀣?땾={value_multiplier:.2f}"



                if value_sig.note:



                    value_note += f"({value_sig.note[:40]})"



            except Exception:



                pass



            multiplier *= value_multiplier







            # 3-3. ??뵠?沅???뒭 雅뚯눘?뻼 ?萸뱄쭪? ?겫袁⑷퐤 ?肉??猷?



            quality_multiplier = 1.0



            quality_note = ""



            try:



                from features.stock_quality import StockQualityAnalyzer



                quality_analyzer = StockQualityAnalyzer()



                quality_sig = quality_analyzer.analyze(df)



                quality_multiplier = quality_sig.position_multiplier



                quality_note = f"?萸뱄쭪?뜄媛???땾={quality_multiplier:.2f}"



                if quality_sig.note:



                    quality_note += f"({quality_sig.note[:40]})"



            except Exception:



                pass



            multiplier *= quality_multiplier







            # 3-4. ?苑뚥빳???뻼 ?逾?獄쏅챶履? ?겫袁⑷퐤 ?肉??猷?



            deep_value_multiplier = 1.0



            deep_value_note = ""



            try:



                from features.deep_value import SeoJunsikAnalyzer



                dv_analyzer = SeoJunsikAnalyzer()



                dv_sig = dv_analyzer.analyze(df)



                deep_value_multiplier = dv_sig.position_multiplier



                deep_value_note = f"?逾?獄쏅챶履잒쳸怨쀫땾={deep_value_multiplier:.2f}"



                if dv_sig.note:



                    deep_value_note += f"({dv_sig.note[:40]})"



            except Exception:



                pass



            multiplier *= deep_value_multiplier







            # 3-5. 鈺곌퀣湲쏙㎗? 甕곌쑬?닜 揶쏅Ŋ?? ?肉??猷?



            bubble_multiplier = 1.0



            bubble_note = ""



            try:



                from features.bubble_detector import BubbleDetector



                bubble_det = BubbleDetector()



                bubble_sig = bubble_det.analyze(df)



                bubble_multiplier = bubble_sig.position_multiplier



                bubble_note = (



                    f"甕곌쑬?닜={bubble_sig.phase.value}"



                    f"(??젎??땾{bubble_sig.bubble_score:.0f},"



                    f"獄쏄퀣?땾{bubble_multiplier:.2f})"



                )



                # 甕곌쑬?닜 ??젎??땾 70+ ??꼥 max_position 40% ?빊類ㅻ꺖



                if bubble_sig.bubble_score >= 70:



                    bubble_multiplier = min(bubble_multiplier, 0.6)



                    bubble_note += "[??⑥쥙?맄?肉?]"



            except Exception:



                pass



            multiplier *= bubble_multiplier







            # 3-6. 獄쏄퉮苑???겱 ??넎??몛 ?겫袁⑷퐤 ?肉??猷?



            fx_multiplier = 1.0



            fx_note = ""



            try:



                from features.exchange_rate import ExchangeRateAnalyzer



                fx_analyzer = ExchangeRateAnalyzer()



                fx_sig = fx_analyzer.analyze(df)



                fx_multiplier = fx_sig.position_multiplier



                fx_note = (



                    f"??넎??몛={fx_sig.dollar_phase.value}"



                    f"(揶쏅베猷?{fx_sig.dollar_strength:.0f},"



                    f"獄쏄퀣?땾{fx_multiplier:.2f})"



                )



                if fx_sig.fx_alarm:



                    fx_multiplier *= 0.7  # ??넎??몛 疫뀀맧?? ??뻻 30% ?빊遺??? ?빊類ㅻ꺖



                    fx_note += "[野껋럥?궖]"



            except Exception:



                pass



            multiplier *= fx_multiplier







            # ??쟿甕곌쑬?봺筌?? ??쟿??? ?湲??釉? (獄쏄퉮苑???겱: 1.5x, 2x 疫뀀뜆??)



            try:



                from features.exchange_rate import BiasCorrector



                multiplier = BiasCorrector.enforce_leverage_guard(



                    BiasCorrector(), multiplier



                )



            except Exception:



                pass







            # 域밸갭?뼊 ??돳?逾?: 獄쏄퀣?땾 ?釉??釉? 0.1, ?湲??釉? 1.5 (??쟿甕곌쑬?봺筌?? 揶???諭?)



            multiplier = round(



                float(max(min(multiplier, 1.5), 0.1)), 2



            )



            adjusted_max = int(base_max_position * multiplier)







            fisher_part = f" | {fisher_note}" if fisher_note else ""



            value_part = f" | {value_note}" if value_note else ""



            quality_part = f" | {quality_note}" if quality_note else ""



            deep_value_part = f" | {deep_value_note}" if deep_value_note else ""



            bubble_part = f" | {bubble_note}" if bubble_note else ""



            fx_part = f" | {fx_note}" if fx_note else ""



            note = (



                f"{cycle_note} | {credit_note}{fisher_part}"



                f"{value_part}{quality_part}{deep_value_part}"



                f"{bubble_part}{fx_part} | "



                f"獄쏄퀣?땾={multiplier:.2f} | "



                f"??⑤벏猷룡묾怨좎돳={is_fear_opportunity}"



            )







            logger.info("?沅???뵠?寃? ?猷뤄쭪????? 鈺곌퀣?젟: {}??뜚 ??꼥 {}??뜚 | {}",



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



            logger.warning("?沅???뵠?寃? 鈺곌퀣?젟 ?겫袁⑷퐤 ??궎?몴?: {}", exc)



            return {



                "adjusted_max": base_max_position,



                "cycle_score": 50.0,



                "credit_status": "NORMAL",



                "multiplier": 1.0,



                "is_fear_opportunity": False,



                "profit_probability": 0.5,



                "note": f"??궎?몴?꼶以? ??뵥?釉? 疫꿸퀡?궚揶??: {exc}",



            }







    def _apply_candle_master_position_limit(



        self,



        max_amount: int,



        available_cash: int,



        signal: Signal,



    ) -> int:



        """筌?遺얜굶筌띾뜆?뮞?苑? ??쁽疫뀀뜃???뵳? 域뱀뮇?뒅??뱽 ??읅??뒠?鍮???빍??뼄.







        筌?遺얜굶筌띾뜆?뮞?苑? ??뜚燁??:



        - ?넫?굝?걠??뼣 ?猷뤄쭪????? ?釉??猷?: ?룯? ??쁽?沅???벥 10% (??꺖?釉? ??④쑴伊???뮉 20%)



        - ??????쟿筌?? ??쁽?猷? ?苑???젟: -10% 疫꿸퀡?궚, 筌ㅼ뮆?? -20%



        - ?눧?눛??疫??(?빊遺??? 筌띲끉?땾嚥?? ?猷딀뉩醫딅뼊揶?? ?沅숂빊遺쎈┛) 疫뀀뜆??







        Args:



            max_amount: 疫꿸퀣??? 筌ㅼ뮆?? ?猷뤄쭪????? 疫뀀뜆釉?



            available_cash: 揶????뒠 ??겱疫??



            signal: 筌띲끉?땾 ??뻻域밸챶瑗?







        Returns:



            int: 筌?遺얜굶筌띾뜆?뮞?苑? 域뱀뮇?뒅??뵠 ??읅??뒠?留? 筌ㅼ뮆?? ?猷뤄쭪????? 疫뀀뜆釉?



        """



        try:



            cm = self._cm_config







            # ?룯? ??쁽?沅? ?빊遺욧텦 (揶????뒠 ??겱疫?? + ??겱??삺 ??떮??쁽疫??)



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







            # ??꺖?釉?/??뵬獄?? ??④쑴伊??肉? ?逾꿰몴? ?넫?굝?걠??뼣 ?뜮袁⑹㉦ 野껉퀣?젟



            if total_assets <= cm.small_account_threshold:



                position_pct = cm.small_account_pct  # ??꺖?釉?: 20%



            else:



                position_pct = cm.max_position_pct   # ??뵬獄??: 10%







            # 筌?遺얜굶筌띾뜆?뮞?苑? ?猷뤄쭪????? ?釉??猷?



            cm_max = int(total_assets * position_pct)







            # 疫꿸퀣??? ?釉??猷???? 筌?遺얜굶筌띾뜆?뮞?苑? ?釉??猷? 餓?? ??삂??? 揶?? ??읅??뒠



            result = min(max_amount, cm_max, available_cash)







            if result < max_amount:



                logger.info(



                    "筌?遺얜굶筌띾뜆?뮞?苑? ?猷뤄쭪????? ?釉??猷? ??읅??뒠: {}??뜚 ??꼥 {}??뜚 "



                    "(?룯?빘?쁽?沅? {}??뜚, ?뜮袁⑹㉦ {:.0%})",



                    max_amount, result, total_assets, position_pct



                )







            return result







        except Exception as exc:



            logger.warning("筌?遺얜굶筌띾뜆?뮞?苑? ?猷뤄쭪????? ?釉??猷? ??읅??뒠 ??궎?몴?: {}", exc)



            return min(max_amount, available_cash)







    def _check_averaging_down(



        self,



        stock_code: str,



        current_price: int,



    ) -> Optional[str]:



        """?눧?눛??疫?? 獄쎻뫗?? 野???沅?







        筌?遺얜굶筌띾뜆?뮞?苑? ??뜚燁??: ?釉???뵭 餓λ쵐?뵥 ?넫?굝?걠?肉? ?빊遺??? 筌띲끉?땾(?눧?눛??疫??) 疫뀀뜆??.



        疫꿸퀣??? ?猷뤄쭪???????벥 ?猷딀뉩? 筌띲끉?뿯揶??癰귣???뼄 ??겱??삺揶??揶?? ?沅???몵筌?? ?빊遺??? 筌띲끉?땾 筌△뫀?뼊.







        Args:



            stock_code: ?넫?굝?걠 ?굜遺얜굡



            current_price: ??겱??삺 揶??野??







        Returns:



            椰꾧퀡?? ?沅????? ?눧紐꾩쁽?肉? (?눧?눛??疫?? ?釉???빍筌?? None)



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



                return None  # 疫꿸퀣??? ?猷뤄쭪????? ?毓???벉 ??꼥 ?눧?눛??疫?? ?釉???뻷







            if current_price < existing.avg_price:



                return (



                    f"?눧?눛??疫?? 獄쎻뫗??: ??겱??삺揶?? {current_price:,}??뜚 < "



                    f"?猷딀뉩醫듼꼻??뿯揶?? {existing.avg_price:,}??뜚 "



                    f"(筌?遺얜굶筌띾뜆?뮞?苑? ??뜚燁??: ?釉???뵭 餓?? ?빊遺??? 筌띲끉?땾 疫뀀뜆??)"



                )







            return None







        except Exception:



            return None  # ??궎?몴? ??뻻 ??꽰???? (?釉???읈)







    def get_portfolio_summary(self) -> dict:



        """?猷???뱜?琉껆뵳?딆궎 ??뒄?鍮? ??젟癰귣???? 獄쏆꼹?넎?鍮???빍??뼄."""



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

