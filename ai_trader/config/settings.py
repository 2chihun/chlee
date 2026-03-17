"""AI Trader 전체 설정 관리

PostgreSQL + TimescaleDB 기반, Streamlit + FastAPI 아키텍처
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class KISConfig:
    """한국투자증권 API 설정"""
    app_key: str = ""
    app_secret: str = ""
    account_no: str = ""
    is_paper: bool = True

    BASE_URL_LIVE: str = "https://openapi.koreainvestment.com:9443"
    BASE_URL_PAPER: str = "https://openapivts.koreainvestment.com:29443"
    WS_URL_LIVE: str = "ws://ops.koreainvestment.com:21000"
    WS_URL_PAPER: str = "ws://ops.koreainvestment.com:31000"

    @property
    def base_url(self) -> str:
        return self.BASE_URL_PAPER if self.is_paper else self.BASE_URL_LIVE

    @property
    def ws_url(self) -> str:
        return self.WS_URL_PAPER if self.is_paper else self.WS_URL_LIVE

    @property
    def trading_mode(self) -> str:
        return "paper" if self.is_paper else "live"

    @trading_mode.setter
    def trading_mode(self, value: str):
        self.is_paper = (value == "paper")


@dataclass
class DBConfig:
    """데이터베이스 설정 (PostgreSQL + TimescaleDB)"""
    host: str = "localhost"
    port: int = 5432
    name: str = "ai_trader"
    user: str = "ai_trader"
    password: str = "ai_trader_pass"
    # SQLite 폴백 (PostgreSQL 미설치 시)
    use_sqlite: bool = True
    sqlite_path: str = str(BASE_DIR / "data" / "ai_trader.db")

    @property
    def url(self) -> str:
        if self.use_sqlite:
            return f"sqlite:///{self.sqlite_path}"
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class RiskConfig:
    """리스크 관리 설정"""
    max_position_size: int = 2_000_000
    max_daily_loss: int = -100_000
    max_positions: int = 5
    stop_loss_pct: float = -1.0
    take_profit_pct: float = 2.0
    min_confidence: float = 0.6
    # 재무 기반 필터 (S-RIM 책 기반)
    min_roe: float = 0.0  # 0이면 비활성화, 양수이면 ROE 필터 적용
    require_profitable: bool = False  # True이면 영업이익 양수 종목만 허용
    max_debt_ratio: float = 0.0  # 0이면 비활성화, 양수이면 부채비율 상한


@dataclass
class FundamentalConfig:
    """재무분석 (S-RIM) 설정"""
    # BBB- 5년 회사채 수익률 (할인율/요구수익률)
    discount_rate: float = 0.08
    # S-RIM 활성화 여부
    srim_enabled: bool = True
    # 재무 데이터 갱신 주기 (일)
    update_interval_days: int = 7


@dataclass
class StrategyConfig:
    """전략 설정"""
    # 분봉 단타
    scalping_interval: str = "5min"
    scalping_rsi_buy: float = 30.0
    scalping_rsi_sell: float = 70.0
    scalping_bb_period: int = 20
    scalping_bb_std: float = 2.0
    # 일중 스윙
    swing_min_volume_ratio: float = 2.0
    swing_gap_threshold: float = 2.0
    swing_hold_limit_minutes: int = 360
    # 감시 종목
    watchlist: list = field(default_factory=lambda: ["005930", "000660", "035720"])
    # 도서 기반 추가 설정 (현명한 당신의 주식투자 교과서)
    execution_strength_threshold: float = 100.0  # 체결강도 기준
    volume_spike_threshold: float = 2.5          # 거래량 급증 배수
    use_candle_patterns: bool = True             # 캔들 패턴 분석 사용
    use_market_flow: bool = True                 # 수급 분석 사용


@dataclass
class CycleConfig:
    """하워드 막스 마켓 사이클 설정"""
    # 사이클 측정 기준 기간 (거래일, 기본 252일 = 약 1년)
    cycle_lookback_days: int = 252
    # 사이클 구간별 최대 포지션 비율
    early_phase_max_pos: float = 0.8   # EARLY (0~30점): 공격적 매수
    mid_phase_max_pos: float = 0.5     # MID (30~70점): 중립
    late_phase_max_pos: float = 0.2    # LATE (70~100점): 방어적
    # 사이클 필터 활성화 여부
    use_cycle_filter: bool = True
    # 확률분포 이동 기반 포지션 조정 활성화
    use_probability_shift: bool = True
    # 극단 회피 로직 활성화 (절대 never/always 방지)
    use_extreme_avoidance: bool = True


@dataclass
class CreditCycleConfig:
    """하워드 막스 신용/자본시장 사이클 설정

    신용사이클은 가장 변동이 심하고 경제/시장에 가장 큰 영향을 미친다.
    신용창구가 열리고 닫히는 상황에 따라 포지션을 조절한다.
    """
    # 신용환경 측정 기준 기간 (거래일, 기본 60일 = 약 3개월)
    credit_lookback_days: int = 60
    # 신용긴축 시 포지션 축소 비율 (기본 30% 축소)
    tight_position_reduction: float = 0.3
    # LATE 사이클 시 포지션 축소 비율 (기본 50% 축소)
    late_cycle_reduction: float = 0.5
    # 유동성 위기 감지 임계값 (평균 거래량 대비 배수)
    liquidity_crisis_threshold: float = 2.5
    # 신용사이클 필터 활성화 여부
    use_credit_filter: bool = True
    # 역투자 기회 감지 활성화 (극단 긴축 시 매수 기회)
    use_contrarian_opportunity: bool = True


@dataclass
class CandleMasterConfig:
    """캔들마스터 파동 매매법 설정

    「캔들마스터의 주식 캔들 매매법」 핵심 개념 기반.
    파동 위치 분석, 캔들군 분석, 자금 관리 설정을 포함합니다.
    """
    # 파동 분석
    decline_threshold: float = 0.50      # 큰 하락 기준 (50%)
    horizontal_width_pct: float = 0.30   # 수평 파동 폭 기준 (30%)
    min_candle_count: int = 50           # 최소 캔들 수 (주간 기준 약 1년)
    max_price_multiple: float = 10.0     # 최대 가격 배수
    lookback_weeks: int = 252            # 분석 대상 기간 (주간 봉, 약 5년)

    # 자금 관리
    max_position_pct: float = 0.10       # 종목당 최대 비중 (10%)
    small_account_pct: float = 0.20      # 소액 종목당 비중 (20%)
    small_account_threshold: float = 10_000_000  # 소액 기준 (1000만원)

    # 손절/목표
    stop_loss_default: float = -0.10     # 기본 손절 (-10%)
    stop_loss_max: float = -0.20         # 최대 손절 (-20%)
    target_profit_default: float = 2.0   # 기본 목표 수익 (200%)
    target_profit_min: float = 1.0       # 최소 목표 수익 (100%)

    # 기능 활성화
    use_wave_filter: bool = True         # 파동 분석 필터 사용
    use_candle_groups: bool = True       # 캔들군 분석 사용

    # 매수 구간 점수 기준
    min_buy_zone_score: float = 50.0     # 최소 매수 구간 점수 (0~100)

    # 물타기 방지
    no_averaging_down: bool = True       # 물타기 방지 활성화


@dataclass
class WizardConfig:
    """잭 슈웨거 마법사 교훈 설정

    「주식시장의 마법사들」 64개 교훈 기반.
    촉매 검증, 기회비용, 확신도 스케일링, 지표 시너지, 규율 추적.
    """
    # 촉매 필터 (교훈 19)
    use_catalyst_filter: bool = True
    catalyst_volume_threshold: float = 2.0   # 평균 거래량 대비 배수
    # 기회비용 (교훈 22, 47)
    use_opportunity_cost: bool = True
    opportunity_cost_days: int = 20          # 기회비용 평가 기간 (거래일)
    opportunity_cost_threshold: float = 0.03  # 3% 이상 기회비용 시 청산
    # 확신도 스케일링 (교훈 37)
    use_confidence_scaling: bool = True
    confidence_max_scale: float = 2.0        # 최대 포지션 배수
    confidence_min_scale: float = 0.3        # 최소 포지션 배수
    # 지표 시너지 (교훈 54)
    use_synergy_filter: bool = True
    synergy_min_aligned: int = 3             # 최소 일치 지표 수
    # 뉴스 반응 (교훈 21, 49)
    use_news_reaction: bool = True
    # 규율 추적 (교훈 7, 34)
    use_discipline_tracking: bool = True
    discipline_min_score: float = 50.0       # 최소 규율 점수


@dataclass
class KenFisherConfig:
    """켄 피셔 "주식시장은 어떻게 반복되는가" 설정

    시장 기억 분석, 변동성 정상화, 불신의 비관론(Wall of Worry),
    약세장→강세장 전환 조기 감지 설정을 포함합니다.
    """
    # 변동성 공포 분석
    volatility_lookback: int = 60         # 변동성 분석 기간 (거래일)
    atr_fear_threshold: float = 2.0       # ATR 공포 배수 기준
    # 불신의 비관론 감지
    wow_lookback: int = 20                # Wall of Worry 분석 기간
    # 약세장 회복 감지
    bear_decline_threshold: float = 0.20  # 약세장 기준 하락률 (20%)
    bear_recovery_threshold: float = 0.10 # 회복 신호 반등률 (10%)
    bear_lookback_days: int = 252         # 약세장 분석 기간
    # 포지션 조정 범위
    max_position_multiplier: float = 1.3  # 최대 포지션 배수
    min_position_multiplier: float = 0.7  # 최소 포지션 배수
    # 기능 활성화
    use_market_memory: bool = True        # 시장 기억 분석 사용
    use_volatility_fear: bool = True      # 변동성 공포 분석 사용
    use_wall_of_worry: bool = True        # 불신의 비관론 감지 사용
    use_bear_recovery: bool = True        # 약세장 회복 감지 사용


@dataclass
class ValueInvestorConfig:
    """강방천&존리 가치투자 설정

    「나의 첫 주식 교과서」 핵심 개념 기반.
    재무건전성 프록시, 저평가 판단, 역발상 매수, 장기보유 확신도.
    """
    # 재무건전성 프록시
    fundamental_lookback: int = 120      # 분석 기간 (거래일)
    # 저평가 판단
    per_low_threshold: float = 10.0      # PER 저평가 기준
    per_high_threshold: float = 25.0     # PER 고평가 기준
    pbr_low_threshold: float = 0.8       # PBR 저평가 기준
    roe_min_threshold: float = 10.0      # ROE 최소 기준 (%)
    # 역발상 매수
    contrarian_rsi_threshold: float = 30.0   # 역발상 RSI 임계값
    consecutive_decline_days: int = 5        # 연속 하락 일수
    # 장기보유
    hold_min_days: int = 60              # 최소 보유 권장일
    overvalued_rsi: float = 80.0         # 고평가 RSI 기준
    # 포지션 조정 범위
    max_position_multiplier: float = 1.5  # 최대 포지션 배수
    min_position_multiplier: float = 0.5  # 최소 포지션 배수
    # 기능 활성화
    use_value_filter: bool = True        # 가치투자 필터 사용
    use_contrarian: bool = True          # 역발상 매수 감지 사용
    use_long_term_hold: bool = True      # 장기보유 확신도 사용


@dataclass
class StockQualityConfig:
    """이남우 "좋은 주식 나쁜 주식" 설정

    주식 품질 평가, 패닉 매수 감지, 자본집약도 분석 설정.
    """
    # ROE 듀폰 프록시
    roe_lookback: int = 120              # 분석 기간 (거래일)
    # 패닉 매수
    panic_threshold: float = 0.30        # 52주 고점 대비 하락률 기준 (30%)
    panic_lookback: int = 252            # 52주 고점 탐색 기간
    min_quality_for_panic: float = 0.5   # 패닉 매수 최소 품질 점수
    # 자본집약도
    intensity_lookback: int = 60         # 자본집약도 분석 기간
    high_atr_pct: float = 0.04           # 고변동 기준 (ATR/가격 4%)
    # 포지션 조정 범위
    max_position_multiplier: float = 1.4  # 최대 포지션 배수
    min_position_multiplier: float = 0.6  # 최소 포지션 배수
    # 기능 활성화
    use_quality_filter: bool = True      # 품질 필터 사용
    use_panic_buy: bool = True           # 패닉 매수 감지 사용
    use_capital_intensity: bool = True   # 자본집약도 분석 사용


@dataclass
class SeoJunsikConfig:
    """서준식 "다시 쓰는 주식투자 교과서" 설정

    채권형 주식 식별, 기대수익률 산출, 안전마진 평가, 떨어지는 칼날 감지.
    """
    lookback: int = 252                    # 분석 기간 (거래일, 약 1년)
    target_return: float = 0.15            # 목표 수익률 15% (버핏 기준)
    bond_type_threshold: float = 0.5       # 채권형 주식 임계값
    safety_margin_threshold: float = 0.5   # 안전마진 임계값
    deep_value_threshold: float = 0.60     # 딥밸류 복합 임계값
    overvalue_threshold: float = 0.30      # 고평가 차단 임계값
    max_position_multiplier: float = 1.3   # 최대 포지션 배수
    min_position_multiplier: float = 0.7   # 최소 포지션 배수
    use_deep_value: bool = True            # 딥밸류 필터 사용


@dataclass
class BackupConfig:
    """백업 설정"""
    backup_dir: str = str(BASE_DIR / "backups")
    parquet_dir: str = str(BASE_DIR / "backups" / "parquet")
    csv_dir: str = str(BASE_DIR / "backups" / "csv")
    keep_days: int = 30
    auto_backup_time: str = "00:30"


@dataclass
class AppConfig:
    """앱 전체 설정"""
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
    backup: BackupConfig = field(default_factory=BackupConfig)
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000


def load_config() -> AppConfig:
    """환경 변수에서 설정을 로드합니다."""
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
        sqlite_path=os.getenv("DB_SQLITE_PATH", str(BASE_DIR / "data" / "ai_trader.db")),
    )

    risk = RiskConfig(
        max_position_size=int(os.getenv("MAX_POSITION_SIZE", "2000000")),
        max_daily_loss=int(os.getenv("MAX_DAILY_LOSS", "-100000")),
        max_positions=int(os.getenv("MAX_POSITIONS", "5")),
        stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "-1.0")),
        take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", "2.0")),
        min_roe=float(os.getenv("MIN_ROE", "0.0")),
        require_profitable=os.getenv("REQUIRE_PROFITABLE", "false").lower() == "true",
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

    # 신용사이클 설정 로드
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
