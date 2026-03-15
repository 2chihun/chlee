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

    return AppConfig(
        kis=kis,
        db=db,
        risk=risk,
        strategy=strategy,
        fundamental=fundamental,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
    )
