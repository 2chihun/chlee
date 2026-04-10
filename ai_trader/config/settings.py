# -*- coding: cp949 -*-



"""AI Trader ??占????占쏙옙? ??占쏙옙????占쏙옙







KIS Open API ?占쏙옙轅몌옙?占쏙옙??? ???占쏙옙??占쏙옙? ??占?????占쏙옙??占??????? ??占????占?????? ??占쏙옙????占쏙옙.



PostgreSQL + TimescaleDB 占???????占쏙옙, Streamlit + FastAPI ?????六사솻占????占???.



"""







import os



from dataclasses import dataclass, field



from pathlib import Path



from dotenv import load_dotenv







load_dotenv()







BASE_DIR = Path(__file__).resolve().parent.parent











@dataclass



class KISConfig:



    """??占쏙옙?占쏙옙占??????占쏙옙???占쏙옙嶺뚯빘占?????? API ??占쏙옙????占쏙옙"""



    app_key: str = ""



    app_secret: str = ""



    account_no: str = ""



    is_paper: bool = True







    BASE_URL_LIVE: str = "https://openapi.koreainvestment.com:9443"



    BASE_URL_PAPER: str = "https://openapivts.koreainvestment.com:29443"



    WS_URL_LIVE: str = "ws://ops.koreainvestment.com:21000"



    WS_URL_PAPER: str = "ws://ops.koreainvestment.com:31000"

    @property
    def trading_mode(self) -> str:
        return "paper" if self.is_paper else "live"

    @property
    def base_url(self) -> str:
        return self.BASE_URL_PAPER if self.is_paper else self.BASE_URL_LIVE

    @property
    def ws_url(self) -> str:
        return self.WS_URL_PAPER if self.is_paper else self.WS_URL_LIVE











@dataclass



class DBConfig:



    """???占쏙옙??占????占쏙옙節덈┛?占쏙옙占??占????占?? ??占쏙옙????占쏙옙"""



    host: str = "localhost"



    port: int = 5432



    name: str = "ai_trader"



    user: str = "ai_trader"



    password: str = "ai_trader_pass"



    use_sqlite: bool = True



    sqlite_path: str = str(BASE_DIR / "data" / "ai_trader.db")

    @property
    def url(self) -> str:
        if self.use_sqlite:
            return f"sqlite:///{self.sqlite_path}"
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"











@dataclass



class RiskConfig:



    """?占????占쏙옙占???占??? ?占????占??? ??占쏙옙????占쏙옙"""



    max_position_size: int = 2_000_000



    max_daily_loss: int = -100_000



    max_positions: int = 5



    stop_loss_pct: float = -1.0



    take_profit_pct: float = 2.0



    min_confidence: float = 0.6



    max_loss_per_trade_pct: float = 3.0   # 椰꾬옙???占쏙옙 筌ㅼ뮆?? ??????占쏙옙 % (獄쏄퉮占?? 疫꿸퀣??)



    daily_loss_limit_pct: float = 5.0     # ??占쏙옙??占쏙옙 ??????占쏙옙 ?占???占?? % (獄쏄퉮占?? 疫꿸퀣??)



    min_roe: float = 0.0



    require_profitable: bool = False



    max_debt_ratio: float = 0.0











@dataclass



class FundamentalConfig:



    """??????占쏙옙嶺뚮‘瑗배땻? ?寃ヨ쥈?占쏙옙?占쏙옙 ??占쏙옙????占쏙옙"""



    discount_rate: float = 0.08



    srim_enabled: bool = True



    update_interval_days: int = 7











@dataclass



class StrategyConfig:



    """???占쏙옙???占쏙옙 ??占쏙옙????占쏙옙"""



    scalping_interval: str = "5min"



    scalping_rsi_buy: float = 30.0



    scalping_rsi_sell: float = 70.0



    scalping_bb_period: int = 20



    scalping_bb_std: float = 2.0



    swing_min_volume_ratio: float = 2.0



    swing_gap_threshold: float = 2.0



    swing_hold_limit_minutes: int = 360



    default_strategy: str = "swing"     # 疫꿸퀡?占쏙옙 ??占쏙옙??占쏙옙 (swing/scalping)



    buy_amount: int = 1_000_000          # 疫꿸퀡?占쏙옙 筌띲끉?占쏙옙 疫뀀뜆占?? (??占쏙옙)



    watchlist: list = field(default_factory=lambda: ["005930", "000660", "035720"])



    execution_strength_threshold: float = 100.0



    volume_spike_threshold: float = 2.5



    use_candle_patterns: bool = True



    use_market_flow: bool = True











@dataclass



class CycleConfig:



    """??占?????占쏙옙 ?占?????占???占??? ??占쏙옙????占쏙옙"""



    cycle_lookback_days: int = 252



    early_phase_max_pos: float = 0.8



    mid_phase_max_pos: float = 0.5



    late_phase_max_pos: float = 0.2



    use_cycle_filter: bool = True



    use_probability_shift: bool = True



    use_extreme_avoidance: bool = True











@dataclass



class CreditCycleConfig:



    """??占?????占쏙옙 ?占?????占???占??? ??占쏙옙????占쏙옙"""



    credit_lookback_days: int = 60



    tight_position_reduction: float = 0.3



    late_cycle_reduction: float = 0.5



    liquidity_crisis_threshold: float = 2.5



    use_credit_filter: bool = True



    use_contrarian_opportunity: bool = True











@dataclass



class CandleMasterConfig:



    """占????占쏙옙?占쏙옙占?? 嶺뚮씭?占쏙옙?占????占쏙옙? ??占쏙옙????占쏙옙"""



    decline_threshold: float = 0.50



    horizontal_width_pct: float = 0.30



    min_candle_count: int = 50



    max_price_multiple: float = 10.0



    lookback_weeks: int = 252



    max_position_pct: float = 0.10



    small_account_pct: float = 0.20



    small_account_threshold: float = 10_000_000



    stop_loss_default: float = -0.10



    stop_loss_max: float = -0.20



    target_profit_default: float = 2.0



    target_profit_min: float = 1.0



    use_wave_filter: bool = True



    use_candle_groups: bool = True



    min_buy_zone_score: float = 50.0



    no_averaging_down: bool = True











@dataclass



class WizardConfig:



    """嶺뚮씭?占쏙옙?占쏙옙??占??? ?占쏙옙占??占???占?? ??占쏙옙????占쏙옙"""



    use_catalyst_filter: bool = True



    catalyst_volume_threshold: float = 2.0



    use_opportunity_cost: bool = True



    opportunity_cost_days: int = 20



    opportunity_cost_threshold: float = 0.03



    use_confidence_scaling: bool = True



    confidence_max_scale: float = 2.0



    confidence_min_scale: float = 0.3



    use_synergy_filter: bool = True



    synergy_min_aligned: int = 3



    use_news_reaction: bool = True



    use_discipline_tracking: bool = True



    discipline_min_score: float = 50.0











@dataclass



class KenFisherConfig:



    """???? ????????? ???占쏙옙???占쏙옙 ??占쏙옙????占쏙옙"""



    volatility_lookback: int = 60



    atr_fear_threshold: float = 2.0



    wow_lookback: int = 20



    bear_decline_threshold: float = 0.20



    bear_recovery_threshold: float = 0.10



    bear_lookback_days: int = 252



    max_position_multiplier: float = 1.3



    min_position_multiplier: float = 0.7



    use_market_memory: bool = True



    use_volatility_fear: bool = True



    use_wall_of_worry: bool = True



    use_bear_recovery: bool = True











@dataclass



class ValueInvestorConfig:



    """?占쏙옙???占쏙옙?占쏙옙占????占쏙옙???占쏙옙 ??占쏙옙????占쏙옙"""



    fundamental_lookback: int = 120



    per_low_threshold: float = 10.0



    per_high_threshold: float = 25.0



    pbr_low_threshold: float = 0.8



    roe_min_threshold: float = 10.0



    contrarian_rsi_threshold: float = 30.0



    consecutive_decline_days: int = 5



    hold_min_days: int = 60



    overvalued_rsi: float = 80.0



    max_position_multiplier: float = 1.5



    min_position_multiplier: float = 0.5



    use_value_filter: bool = True



    use_contrarian: bool = True



    use_long_term_hold: bool = True











@dataclass



class StockQualityConfig:



    """?占쏙옙?占쏙옙?占쏙옙?占?? ??占쏙옙諭꾬옙??? ?寃ヨ쥈?占쏙옙?占쏙옙 ??占쏙옙????占쏙옙"""



    roe_lookback: int = 120



    panic_threshold: float = 0.30



    panic_lookback: int = 252



    min_quality_for_panic: float = 0.5



    intensity_lookback: int = 60



    high_atr_pct: float = 0.04



    max_position_multiplier: float = 1.4



    min_position_multiplier: float = 0.6



    use_quality_filter: bool = True



    use_panic_buy: bool = True



    use_capital_intensity: bool = True











@dataclass



class SeoJunsikConfig:



    """??占쏙옙?占쏙옙占?????占?? ???占쏙옙???占쏙옙 ??占쏙옙????占쏙옙"""



    lookback: int = 252



    target_return: float = 0.15



    bond_type_threshold: float = 0.5



    safety_margin_threshold: float = 0.5



    deep_value_threshold: float = 0.60



    overvalue_threshold: float = 0.30



    max_position_multiplier: float = 1.3



    min_position_multiplier: float = 0.7



    use_deep_value: bool = True











@dataclass



class BubbleDetectorConfig:



    """?占쏙옙怨뚯뫊??占쏙옙 ?占쏙옙?占쏙옙흮?? ??占쏙옙????占쏙옙"""



    ma_period: int = 200



    bb_period: int = 20



    bb_std: float = 2.0



    cascade_lookback: int = 20



    gap_down_threshold: float = -0.02



    leverage_lookback: int = 60



    vol_spike_threshold: float = 2.0



    euphoria_multiplier: float = 0.8



    peak_multiplier: float = 0.5



    burst_multiplier: float = 0.2



    panic_multiplier: float = 0.1



    use_bubble_detector: bool = True



    use_cascade_risk: bool = True



    use_passive_cascade: bool = True











@dataclass



class ExchangeRateConfig:



    """???占쏙옙??占?? ?寃ヨ쥈?占쏙옙?占쏙옙 ??占쏙옙????占쏙옙"""



    dollar_lookback: int = 120



    strong_threshold: float = 70.0



    weak_threshold: float = 30.0



    fx_alarm_threshold: float = 2.0



    risk_lookback: int = 20



    use_negativity_correction: bool = True



    use_recency_correction: bool = True



    max_leverage: float = 1.5



    strong_dollar_multiplier: float = 0.7



    weak_dollar_multiplier: float = 1.1



    fx_alarm_reduction: float = 0.8



    use_exchange_rate: bool = True



    use_bias_corrector: bool = True











@dataclass



class QuantValueConfig:



    """?????占?? ?占쏙옙???占쏙옙?? ??占쏙옙????占쏙옙"""



    lookback: int = 120



    dividend_lookback: int = 252



    momentum_period: int = 100



    evaluation_period: int = 40



    debt_penalty_threshold: float = 0.03



    deep_value_threshold: float = 0.70



    weight_value: float = 0.40



    weight_profitability: float = 0.20



    weight_growth: float = 0.15



    weight_safety: float = 0.15



    weight_dividend: float = 0.10



    max_position_multiplier: float = 1.5



    min_position_multiplier: float = 0.5



    use_quant_value: bool = True



    use_calendar_effect: bool = True



    use_momentum: bool = True











@dataclass



class WallStreetQuantConfig:



    """???占쏙옙??占????占???占????占쏙옙占?? ?????占?? ??占쏙옙????占쏙옙"""



    lookback: int = 120



    weight_value: float = 0.25



    weight_momentum: float = 0.25



    weight_quality: float = 0.20



    weight_low_vol: float = 0.15



    weight_size: float = 0.15



    bias_penalty_factor: float = 0.5



    bias_threshold: float = 0.3



    use_wall_street_quant: bool = True



    use_bias_detection: bool = True



    use_risk_decomposition: bool = True











@dataclass



class GrahamInvestorConfig:



    """?占쏙옙諛몄굡??占쏙옙??占???占??? ???占쏙옙???占쏙옙???占쏙옙 ??占쏙옙????占쏙옙"""



    lookback: int = 240



    weight_mos: float = 0.30



    weight_defensive: float = 0.25



    weight_mr_market: float = 0.20



    weight_low_per: float = 0.25



    use_graham_investor: bool = True



    use_margin_of_safety: bool = True



    use_mr_market: bool = True











@dataclass



class BeatTheMarketConfig:



    """??占?????占쏙옙 ??逾좂뼨轅몌옙?占썹뵳? ??占쏙옙????占쏙옙"""



    lookback: int = 240



    weight_kelly: float = 0.20



    weight_earnings_yield: float = 0.30



    weight_stat_arb: float = 0.25



    weight_compound: float = 0.25



    use_beat_the_market: bool = True



    use_kelly: bool = True



    use_stat_arb: bool = True











# ????????? ??六욕윜?? ??占쏙옙???占쏙옙? ?占쏙옙轅몌옙?占쏙옙??? Config (TOP 5) ?????????











@dataclass



class AFMLConfig:



    """Advances in Financial Machine Learning ??占쏙옙????占쏙옙"""



    cusum_threshold: float = 0.02



    take_profit_pct: float = 0.02



    stop_loss_pct: float = 0.01



    max_holding_bars: int = 20



    meta_lookback: int = 120



    use_meta_labeling: bool = True



    use_cusum_filter: bool = True



    use_frac_diff: bool = True















@dataclass



class MLConfig:



    """ML ???占쏙옙??占????占쏙옙???占????占?? ??占쏙옙????占쏙옙 (L筌ｌ겘ez de Prado ???占쏙옙???占쏙옙)"""



    # Triple-Barrier (Ch3)



    triple_barrier_pt: float = 1.0       # profit-taking ?占쏙옙?占쏙옙?????占쏙옙



    triple_barrier_sl: float = 1.0       # stop-loss ?占쏙옙?占쏙옙?????占쏙옙



    max_holding_bars: int = 20           # ???占쏙옙占???? ???占쏙옙?占쏙옙?? (bars)



    min_return: float = 0.001            # 嶺뚣끉占???占?? 嶺뚮ㅄ維싷쭗? ???占쏙옙??占???占???



    # Purged K-Fold CV (Ch7)



    cv_n_splits: int = 5                 # K-Fold ?寃ヨ쥈?占쏙옙占?? ???占쏙옙



    purge_pct: float = 0.01              # ???占쏙옙占???? ??占쏙옙熬곣뫗占??



    embargo_pct: float = 0.01            # ??占쏙옙?占쏙옙爾면겫????? ??占쏙옙熬곣뫗占??



    # Fractionally Differentiated Features (Ch5)



    frac_diff_d: float = 0.35            # ?寃ヨ쥈?占쏙옙?占쏙옙 嶺뚢뼰占?????? 嶺뚢뼰占????占쏙옙



    frac_diff_threshold: float = 1e-5    # ?占쏙옙??繞벿살탳??占쏙옙 ???占쏙옙??占?? ??占?????占쏙옙?占쏙옙????



    # Meta-Labeling (Ch3)



    meta_labeling: bool = True           # 嶺뚮∥???? ???占쏙옙??占????占쏙옙?占쏙옙?占쏙옙占?? ???占쏙옙??占쏙옙????占쏙옙



    meta_lookback: int = 120             # 嶺뚮∥???? 嶺뚮ㅄ占????占쏙옙 ??占쏙옙???占?? ?占쏙옙轅몌옙?占쏙옙???



    # Sample Weights (Ch4)



    use_sample_weights: bool = True      # ?占???????? ?占쏙옙??繞벿살탳??占쏙옙 ?占??????占쏙옙



    # CUSUM Filter (Ch2)



    cusum_threshold: float = 0.02        # CUSUM ??占?????占쏙옙?占쏙옙????



    # Feature Importance (Ch8)



    use_mdi: bool = True                 # Mean Decrease Impurity



    use_mda: bool = True                 # Mean Decrease Accuracy



    use_sfi: bool = True                 # Single Feature Importance



    # Structural Breaks (Ch17)



    use_structural_breaks: bool = True   # CUSUM/SADF ?占쏙옙?占쏙옙?占썼퉪???占쏙옙 揶쏅Ŋ??



    # Entropy Features (Ch18)



    use_entropy: bool = True             # Shannon/LZ/ApEn ?占????占쏙옙嚥≪뮉占??











@dataclass



class MLAlphaConfig:



    """ML Alpha ???占쏙옙??占쏙옙? ??占쏙옙????占쏙옙"""



    lookback: int = 120



    forward_period: int = 5



    cluster_threshold: float = 0.7



    use_ml_alpha: bool = True



    use_sentiment_proxy: bool = True



    use_factor_clustering: bool = True











@dataclass



class KRQuantConfig:



    """??占쏙옙?占쏙옙占??? ?????占?? ???占쏙옙??占쏙옙? ??占쏙옙????占쏙옙"""



    lookback: int = 252



    weight_value: float = 0.30



    weight_momentum: float = 0.30



    weight_quality: float = 0.20



    weight_small_cap: float = 0.20



    use_kr_quant: bool = True



    use_dual_momentum: bool = True



    use_seasonality: bool = True











@dataclass



class BacktestAnalyticsConfig:



    """?占쏙옙?占쏙옙?占쏙옙?????占????占?? ?寃ヨ쥈?占쏙옙?占쏙옙 ??占쏙옙????占쏙옙"""



    lookback: int = 252



    trades_per_year: int = 50



    max_participation: float = 0.05



    commission_rate: float = 0.00015



    tax_rate: float = 0.0018



    use_backtest_analytics: bool = True



    use_kelly: bool = True



    use_overfit_detection: bool = True











@dataclass



class SignalValidationConfig:



    """??六삣윜諛몄굡?占쏙옙? ?占쏙옙??占???? ??占쏙옙????占쏙옙"""



    min_samples: int = 30



    alpha: float = 0.05



    num_strategies_tried: int = 20



    use_signal_validation: bool = True



    use_multiple_testing: bool = True



    use_bias_estimation: bool = True











# ????????? ??占????占쏙옙? ??占쏙옙????占쏙옙 ?????????











@dataclass



class BookIntegratorConfig:



    """??占쏙옙???占쏙옙? ??占????占쏙옙? ?占쏙옙??繞벿살탳??占쏙옙 ??占쏙옙????占쏙옙"""



    weight_market_cycle: float = 0.20



    weight_exchange_rate: float = 0.15



    weight_bubble_detector: float = 0.15



    weight_wizard: float = 0.12



    weight_market_memory: float = 0.10



    weight_deep_value: float = 0.08



    weight_stock_quality: float = 0.08



    weight_execution: float = 0.07



    weight_value_investor: float = 0.05



    use_dynamic_weights: bool = True



    bubble_override: bool = True



    use_book_integrator: bool = True











@dataclass



class BackupConfig:



    """?占쏙옙?占쏙옙?占쏙옙占??? ??占쏙옙????占쏙옙"""



    backup_dir: str = str(BASE_DIR / "backups")



    parquet_dir: str = str(BASE_DIR / "backups" / "parquet")



    csv_dir: str = str(BASE_DIR / "backups" / "csv")



    keep_days: int = 30



    auto_backup_time: str = "00:30"











@dataclass



class AppConfig:



    """???占쏙옙占???? ??占쏙옙????占쎈끆占????????占?????? ??占쏙옙????占쏙옙"""



    kis: KISConfig = field(default_factory=KISConfig)



    db: DBConfig = field(default_factory=DBConfig)



    risk: RiskConfig = field(default_factory=RiskConfig)



    strategy: StrategyConfig = field(default_factory=StrategyConfig)



    fundamental: FundamentalConfig = field(default_factory=FundamentalConfig)



    cycle: CycleConfig = field(default_factory=CycleConfig)



    credit_cycle: CreditCycleConfig = field(default_factory=CreditCycleConfig)



    candle_master: CandleMasterConfig = field(default_factory=CandleMasterConfig)



    wizard: WizardConfig = field(default_factory=WizardConfig)



    ken_fisher: KenFisherConfig = field(default_factory=KenFisherConfig)



    value_investor: ValueInvestorConfig = field(default_factory=ValueInvestorConfig)



    stock_quality: StockQualityConfig = field(default_factory=StockQualityConfig)



    seo_junsik: SeoJunsikConfig = field(default_factory=SeoJunsikConfig)



    bubble_detector: BubbleDetectorConfig = field(default_factory=BubbleDetectorConfig)



    exchange_rate: ExchangeRateConfig = field(default_factory=ExchangeRateConfig)



    quant_value: QuantValueConfig = field(default_factory=QuantValueConfig)



    wall_street_quant: WallStreetQuantConfig = field(default_factory=WallStreetQuantConfig)



    graham_investor: GrahamInvestorConfig = field(default_factory=GrahamInvestorConfig)



    beat_the_market: BeatTheMarketConfig = field(default_factory=BeatTheMarketConfig)



    afml: AFMLConfig = field(default_factory=AFMLConfig)



    ml: MLConfig = field(default_factory=MLConfig)



    ml_alpha: MLAlphaConfig = field(default_factory=MLAlphaConfig)



    kr_quant: KRQuantConfig = field(default_factory=KRQuantConfig)



    backtest_analytics: BacktestAnalyticsConfig = field(default_factory=BacktestAnalyticsConfig)



    signal_validation: SignalValidationConfig = field(default_factory=SignalValidationConfig)



    book_integrator: BookIntegratorConfig = field(default_factory=BookIntegratorConfig)



    backup: BackupConfig = field(default_factory=BackupConfig)



    log_level: str = "INFO"



    api_host: str = "0.0.0.0"



    api_port: int = 8000











def load_config() -> AppConfig:



    """???占쏙옙?占쏙옙猿뗫윥?????占쏙옙??占쏙옙???占쏙옙? ??占쏙옙????占쏙옙??占?? ?占쏙옙?占쏙옙裕녻キ???占쏙옙???占????占??."""



    trading_mode = os.getenv("TRADING_MODE", "paper")







    kis = KISConfig(



        app_key=os.getenv("KIS_APP_KEY", ""),



        app_secret=os.getenv("KIS_APP_SECRET", ""),



        account_no=os.getenv("KIS_ACCOUNT_NO", ""),



        is_paper=(trading_mode == "paper"),



    )







    use_sqlite = os.getenv("DB_USE_SQLITE", "true").lower() == "true"



    db = DBConfig(



        host=os.getenv("DB_HOST", "localhost"),



        port=int(os.getenv("DB_PORT", "5432")),



        name=os.getenv("DB_NAME", "ai_trader"),



        user=os.getenv("DB_USER", "ai_trader"),



        password=os.getenv("DB_PASSWORD", "ai_trader_pass"),



        use_sqlite=use_sqlite,



        sqlite_path=os.getenv(



            "DB_SQLITE_PATH", str(BASE_DIR / "data" / "ai_trader.db")



        ),



    )







    risk = RiskConfig(



        max_position_size=int(os.getenv("MAX_POSITION_SIZE", "2000000")),



        max_daily_loss=int(os.getenv("MAX_DAILY_LOSS", "-100000")),



        max_positions=int(os.getenv("MAX_POSITIONS", "5")),



        stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "-1.0")),



        take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", "2.0")),



        min_roe=float(os.getenv("MIN_ROE", "0.0")),



        require_profitable=os.getenv(



            "REQUIRE_PROFITABLE", "false"



        ).lower() == "true",



        max_debt_ratio=float(os.getenv("MAX_DEBT_RATIO", "0.0")),



    )







    fundamental = FundamentalConfig(



        discount_rate=float(os.getenv("SRIM_DISCOUNT_RATE", "0.08")),



        srim_enabled=os.getenv("SRIM_ENABLED", "true").lower() == "true",



        update_interval_days=int(os.getenv("SRIM_UPDATE_DAYS", "7")),



    )







    watchlist_str = os.getenv("WATCHLIST", "005930,000660,035720")



    watchlist = [s.strip() for s in watchlist_str.split(",") if s.strip()]







    strategy = StrategyConfig(watchlist=watchlist)







    credit_cycle = CreditCycleConfig(



        credit_lookback_days=int(os.getenv("CREDIT_LOOKBACK_DAYS", "60")),



        tight_position_reduction=float(



            os.getenv("CREDIT_TIGHT_REDUCTION", "0.3")



        ),



        late_cycle_reduction=float(



            os.getenv("LATE_CYCLE_REDUCTION", "0.5")



        ),



        use_credit_filter=os.getenv(



            "USE_CREDIT_FILTER", "true"



        ).lower() == "true",



        use_contrarian_opportunity=os.getenv(



            "USE_CONTRARIAN_OPPORTUNITY", "true"



        ).lower() == "true",



    )







    return AppConfig(



        kis=kis,



        db=db,



        risk=risk,



        strategy=strategy,



        fundamental=fundamental,



        credit_cycle=credit_cycle,



        log_level=os.getenv("LOG_LEVEL", "INFO"),



        api_host=os.getenv("API_HOST", "0.0.0.0"),



        api_port=int(os.getenv("API_PORT", "8000")),



    )
