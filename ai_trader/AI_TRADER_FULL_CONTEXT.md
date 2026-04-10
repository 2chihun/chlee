# AI Trader - 전체 프로젝트 컨텍스트 문서
> 다른 AI에게 전달하기 위한 통합 문서 (2026-03-18 생성 / **2026-03-25 최종 업데이트**)
>
> **업데이트 내역 (2026-03-25):**
> - 10번째 도서 추가: 홍용찬 "실전 퀀트투자" → `features/quant_value.py`
> - 신규 모듈: `quant_value.py` (PER/PBR/PSR/PCR 가치, ROE/ROA 수익성, 성장성, 안전성, 배당+흑자, 5-2 모멘텀, 캘린더 효과)
> - config 확장: `QuantValueConfig` 추가
> - 테스트 확장: 38개 → **42개** (전체 PASS)
> - 인사이트 파일: 10권 `C:\Copilot\captures\- 텍스트 변환\*_insight.md`
>
> **업데이트 내역 (2026-03-22):**
> - 9번째 도서 추가: 백석현 "환율 모르면 주식투자 절대로 하지 마라" → `features/exchange_rate.py`
> - 신규 모듈 3개: `bubble_detector.py` (조상철), `execution_analysis.py` (박병창 보완), `book_integrator.py` (9권 통합)
> - config 확장: `BubbleDetectorConfig`, `ExchangeRateConfig`, `BookIntegratorConfig` 추가
> - 테스트 확장: 21개 → **38개** (전체 PASS)
> - 리스크 관리: 버블 감지(40%축소) + 환율 급변(30%축소) + 레버리지 절대상한(1.5x) 추가
> - 인사이트 파일: 9권 전체 `C:\Copilot\captures\- 텍스트 변환\*_insight.md` 저장 완료

---

## 1. 프로젝트 개요

### 1.1 목적
한국투자증권(KIS) API 기반 **AI 자동매매 시스템**. 10권의 투자 서적에서 추출한 인사이트를 기술적 프록시로 변환하여 매매 신호에 통합.

### 1.2 핵심 스택
- **언어**: Python 3.12
- **API**: 한국투자증권 Open API (REST + WebSocket)
- **DB**: SQLite (기본) / PostgreSQL + TimescaleDB (옵션)
- **대시보드**: Streamlit + FastAPI
- **지표**: ta 라이브러리 + 자체 구현
- **스케줄링**: schedule 라이브러리
- **로깅**: loguru

### 1.3 실행 모드
```bash
# 봇만 실행
python main.py --mode bot --paper

# 서버만 실행 (FastAPI)
python main.py --mode server

# 봇 + 서버 통합 실행
python main.py --mode both
```

### 1.4 아키텍처 흐름
```
데이터 수집 (KIS API)
    ↓
기술적 지표 계산 (features/indicators.py)
    ↓
10개 도서 기반 분석 모듈 (features/*.py)
    ↓
전략 신호 생성 (strategies/swing.py, scalping.py)
    ↓
리스크 검사 (risk/manager.py)
    ↓
주문 실행 (execution/order.py)
    ↓
DB 기록 (data/database.py)
```

---

## 2. 파일 구조

```
C:\Copilot\ai_trader\
├── main.py                      # 오케스트레이터 (AITrader 클래스)
├── requirements.txt             # 의존성
├── .env                         # API 키 (KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT_NO)
│
├── config/
│   └── settings.py              # 13개 @dataclass 설정 (AppConfig)
│
├── data/
│   ├── collector.py             # KIS REST API 데이터 수집 (KISAuth, KISDataCollector)
│   ├── database.py              # SQLAlchemy ORM (Position, Trade, MinuteCandle, DailyPnL, FinancialData)
│   ├── websocket_client.py      # KIS WebSocket 실시간 틱
│   └── backup.py                # Parquet/CSV 백업 + StatisticsEngine
│
├── features/                    # ** 핵심: 9권 도서 기반 분석 모듈 (17개) **
│   ├── indicators.py            # 17개 기술적 지표 (SMA, EMA, RSI, MACD, BB, VWAP, ATR, OBV, MFI 등)
│   ├── fundamental.py           # S-RIM 잔여이익모델 + PER/PBR/PEGR
│   ├── candle_patterns.py       # 18개 캔들스틱 패턴 + 캔들군 분석 [박병창]
│   ├── market_flow.py           # 수급 분석 (외국인/기관/개인) [박병창]
│   ├── execution_analysis.py    # CANSLIM+PEG+시간대전략+장대봉 [박병창 보완] ★신규
│   ├── market_cycle.py          # 하워드 막스 마켓 사이클 (CycleSignal)
│   ├── credit_cycle.py          # 하워드 막스 신용 사이클 (CreditCycleSignal)
│   ├── wave_position.py         # 캔들마스터 파동 위치 분석 (WaveSignal)
│   ├── wizard_discipline.py     # 잭 슈웨거 마법사 교훈 (WizardSignal)
│   ├── market_memory.py         # 켄 피셔 시장 기억 (FisherSignal)
│   ├── value_investor.py        # 강방천&존리 가치투자 (ValueInvestorSignal)
│   ├── stock_quality.py         # 이남우 주식 품질 (StockQualitySignal)
│   ├── deep_value.py            # 서준식 딥밸류 (SeoJunsikSignal)
│   ├── bubble_detector.py       # 조상철 AI버블 감지 (BubbleSignal) ★신규
│   ├── exchange_rate.py         # 백석현 환율 분석 (ExchangeRateSignal)
│   ├── quant_value.py           # 홍용찬 퀀트밸류 (QuantValueSignal) ★신규
│   └── book_integrator.py       # 10권 통합 가중합산 (IntegratedSignal)
│
├── strategies/
│   ├── base.py                  # BaseStrategy ABC + Signal dataclass + SignalType enum
│   ├── scalping.py              # 5분봉 단타 (RSI + BB + MACD + 거래량)
│   └── swing.py                 # 일중 스윙 (갭 + 수급 + EMA배열 + 10개 도서 필터 통합)
│
├── risk/
│   └── manager.py               # 리스크 관리 (포지션 사이징 + 사이클 조정 + 물타기 방지)
│
├── execution/
│   └── order.py                 # KIS API 주문 실행 (매수/매도/잔고조회)
│
├── backtest/
│   └── engine.py                # 백테스트 엔진 (히스토리컬 시뮬레이션)
│
├── dashboard/
│   ├── api_server.py            # FastAPI REST API
│   └── streamlit_app.py         # Streamlit 대시보드
│
├── tests/
│   └── test_core.py             # 38개 테스트 케이스 (21 기존 + 17 신규)
│
└── tools/                       # 유틸리티 (OCR, 캡처 등)
```

---

## 3. 도서 학습 인사이트 요약 (9권)

### 3.1 「현명한 당신의 주식투자 교과서」 → candle_patterns.py, market_flow.py, indicators.py
- **18개 캔들스틱 패턴** 인식 (단일봉 9 + 이중봉 5 + 삼중봉 4)
- **수급 분석**: 외국인/기관/개인 매수매도, 주도섹터/주도주 탐지
- **체결강도**: 매수체결량/매도체결량 비율, 100 이상이면 매수 우위
- **거래량 급증**: 20일 평균의 2.5배 이상

### 3.2 「투자와 마켓사이클의 법칙」 하워드 막스 → market_cycle.py, credit_cycle.py
- **사이클 점수 (0~100)**: 52주고저(30%) + RSI(25%) + 거래량(15%) + ATR(15%) + MACD(15%)
- **3단계**: EARLY(0~30) 공격적 80%, MID(30~70) 중립 50%, LATE(70~100) 방어적 20%
- **확률분포 이동**: 저점→수익우위(0.70), 고점→손실우위(0.30)
- **신용사이클**: ATR(35%) + 거래량급감(35%) + 갭다운(20%) + RSI과매도(10%)
- **극단회피**: 상한 90%, 하한 10%, 극단탐욕+LATE=15% 상한 강제
- **전환점 감지**: RSI다이버전스 + 거래량다이버전스 + MACD전환

### 3.3 「주식 캔들 매매법」 캔들마스터 → wave_position.py, candle_patterns.py (보완)
- **파동 분석**: 큰 하락 → 수평 횡보(표준형/응용형) → 후반부 캔들군/캔들 신호
- **6가지 무효 조건**: 50%넘은 파동, 비수평, 10배초과, 짧은상장, 완만, 저점에서 먼 구간
- **캔들군**: 깔짝파동, 꼬리군(이중/다중/후퇴), 상기캔, 수렴캔
- **자금관리**: 10%/종목(1000만↑), 20%/종목(1000만↓), 손절 -10%, 물타기 금지

### 3.4 「주식시장의 마법사들」 잭 슈웨거 → wizard_discipline.py
- **촉매 검증 진입**: 거래량급증/갭/BB돌파 중 최소 1개 촉매 필요
- **8개 지표 시너지**: RSI+MACD+BB+거래량+OBV+EMA+MFI+VWAP 방향 일치도
- **확신도 포지션 조절**: 0.3x~2.0x 스케일링 (max_position_size 상한 유지)
- **기회비용 청산**: 보유 20일+ & 시장 대비 3%+ 기회비용 시 청산 권고
- **규율 점수**: 매매기록/패턴분석, 50점 미만 → 포지션 50% 제한

### 3.5 「주식시장은 어떻게 반복되는가」 켄 피셔 → market_memory.py
- **변동성 정상화**: ATR 공포 극단 → 역발상 기회
- **불신의 비관론(Wall of Worry)**: 반등 중 하락 횟수 → 불신 강할수록 상승 지속
- **V자 회복 대칭성**: 낙폭/반등 기울기 비율
- **강세장 초기 감지**: 12개월 기대수익률 +46.6%
- **극단 비관론 축적**: "이번엔 다르다" 착각 역발상

### 3.6 「나의 첫 주식 교과서」 강방천&존리 → value_investor.py
- **재무건전성 프록시**: 장기추세/변동성/거래량으로 산출
- **역발상 매수**: RSI과매도 반등 + BB하단 복귀 + 연속하락 반등
- **장기보유 4 매도조건**: 고평가/펀더멘털변화/세상변화/기회비용
- **저평가 vs 고평가**: valuation_score < 0.3이면 매수 차단

### 3.7 「좋은 주식 나쁜 주식」 이남우 → stock_quality.py
- **ROE 듀폰 프록시**: 수익률(40%) + 안정성(35%) + 레버리지(25%)
- **패닉 매수**: 52주 고점 대비 30%+ 하락 감지
- **자본집약도**: ATR/가격 비율
- **산업 모멘텀**: 섹터별 상대 강도
- **저품질 차단**: quality_score < 0.3이면 매수 보류

### 3.8 「다시 쓰는 주식투자 교과서」 서준식 → deep_value.py
- **채권형 주식**: 안정성(40%) + 비순환(35%) + 경자본(25%)
- **기대수익률 15%**: SMA252=BPS, Sharpe=ROE, 10년 복리
- **안전마진**: 52주저점거리(35%) + 이평거리(35%) + BB위치(30%)
- **떨어지는 칼날**: 20%+ 하락 + OBV 건전 + bond_type≥0.4
- **복합점수**: bond_type×0.30 + safety_margin×0.30 + return×0.25 + falling_knife×0.15

### 3.9 「2026년 11월, 주식시장 대폭락」 조상철 → bubble_detector.py ★신규
- **AI/GPU 버블 구조**: 950조 CAPEX → 가동률 40% → ROI Gap < 0.5 = 위험
- **24개월 만기 벽**: 대출 폭발 24개월 후 만기 집중 → 시스템 리스크
- **연쇄 마진콜**: LTV 50% 기준선 → 킬스위치 → 담보 투매 피드백 루프
- **4단계 폭락**: 섹터 리더 급락 → 테마주 매도 → 패시브 ETF 기계적 매도 → 서킷브레이커
- **버블 단계**: NORMAL / EUPHORIA / PEAK / BURST / PANIC
- **포지션 배수**: EUPHORIA=0.7~0.9 / PEAK=0.4~0.6 / BURST=0.2 / PANIC=0.1
- **서킷브레이커**: 7%→15분 / 13%→15분 / 20%→당일 거래 중단

### 3.10 「환율 모르면 주식투자 절대로 하지 마라」 백석현 → exchange_rate.py ★신규
- **환율 = 분자(미국경제) / 분모(세계경제)**: 미국 강세 → 달러 강세 → KOSPI 하락
- **달러 사이클**: STRONG_EARLY / STRONG_LATE / WEAK_EARLY / WEAK_LATE / NEUTRAL
- **달러-원 반비례**: KOSPI 2% 변동 ≈ 환율 1% 변동 (경보 기준)
- **글로벌 위험선호도**: RSI+변동성+추세 프록시 (AUD/JPY 대체)
- **레버리지 절대 금지**: 최대 1.5x (2x 금지 - 케인스 파산 사례)
- **심리 편향 보정**: 부정성 편향 / 최신 효과 / 레버리지 가드
- **달러 자산 비중**: 항시 15-25% 유지 권장
- **상관계수**: 한국주식-미국국채 -0.55 (최상의 분산)

### 3.11 「현명한 당신의 주식투자 교과서」 박병창 보완 → execution_analysis.py ★신규
- **CANSLIM 7요소**: C(분기수익) + A(연간추세) + N(신고가) + S(수급) + L(주도) + I(기관) + M(시장방향)
- **PEG 비율**: PER / 이익증가율 → < 1.0 저평가 성장주
- **시간대 전략**: 강세장=아침 10시 전 / 약세장=오후 2시 후
- **장대봉 50% 기준선**: 조정이 50% 이하 = 정상 / 초과 = 약세 전환

### 3.12 9권 통합 → book_integrator.py ★신규
- **가중합산**: 하워드막스(20%) + 백석현(15%) + 조상철(15%) + 잭슈웨거(12%) + 켄피셔(10%) + 서준식(8%) + 이남우(8%) + 박병창(7%) + 강방천&존리(5%)
- **동적 가중치**: 고변동성 → 거시 가중치 강화 / 저변동성 → 미시 가중치 강화
- **상충 해소**: 버블 BURST/PANIC → 매수 시그널 무시 (안전 우선)
- **행동 권고**: STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL / BLOCK

---

## 4. 핵심 통합 패턴

### 4.1 Signal Dataclass (strategies/base.py)
```python
class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class Signal:
    type: SignalType
    stock_code: str
    price: int
    quantity: int = 0
    stop_loss: int = 0
    take_profit: int = 0
    confidence: float = 0.0  # 0.0 ~ 1.0
    reason: str = ""
    strategy_name: str = ""
```

### 4.2 도서 모듈 통합 패턴 (try/except import guard)
```python
# swing.py에서의 패턴
try:
    from features.deep_value import SeoJunsikAnalyzer, SeoJunsikSignal
    _HAS_DEEP_VALUE = True
except ImportError:
    _HAS_DEEP_VALUE = False
```

### 4.3 분석기 → 전략 → 리스크 캐스케이드
```
analyze() 단계:
  각 도서 분석기.analyze(df) → *Signal 저장 (16개 모듈)
  결과 컬럼을 DataFrame에 추가

generate_signal() 단계 (완전한 순서):
  사이클필터 → 기본지표 → 캔들패턴 → 파동필터 → 켄피셔 → 가치투자
  → 품질 → 딥밸류 → 마법사 → [버블감지★] → [환율분석★] → [집행분석★]
  → [통합분석(BookIntegrator)★]
  각 필터가 HOLD/BLOCK 반환 가능 (차단)
  통과 시 confidence 누적 (0.5 시작, 최대 1.0)

risk/manager.py 단계:
  check_signal() → 일일손실/보유한도/중복/재무필터/물타기방지
  포지션 사이징: 사이클배수 × 캔들마스터한도 × 확신도스케일 × 규율제한
  apply_cycle_adjustment() → 사이클 × 신용 × 켄피셔 × 가치투자 × 품질 × 딥밸류
                            × 버블★ × 환율★
  최종 multiplier: max(min(multiplier, 1.5), 0.1) [★레버리지 가드 1.5x]
```

### 4.4 각 도서 모듈의 Signal Dataclass 구조

| 모듈 | Signal 클래스 | 주요 필드 |
|------|--------------|-----------|
| market_cycle | CycleSignal | cycle_score, phase, sentiment, risk_posture, max_position_pct, turning_point, profit_probability |
| credit_cycle | (dict) | credit_env.status (EASY/NORMAL/TIGHT), position_multiplier, is_opportunity |
| wave_position | WaveSignal | buy_zone_score, wave_type, is_latter_half, invalidation |
| wizard_discipline | WizardSignal | synergy, confidence, catalyst, opportunity_cost, discipline, news_reaction, position_multiplier |
| market_memory | FisherSignal | fisher_composite, volatility_fear_score, wall_of_worry_score, v_shape_score, early_bull_phase, extreme_distrust, expected_12m_return, position_multiplier, confidence_delta |
| value_investor | ValueInvestorSignal | fundamental_score, valuation_score, contrarian_score, hold_confidence, sell_reason, position_multiplier, confidence_delta |
| stock_quality | StockQualitySignal | quality_score, roe_quality, is_panic_buy, panic_depth, capital_intensity, industry_momentum, position_multiplier, confidence_delta |
| deep_value | SeoJunsikSignal | bond_type_score, expected_return, safety_margin_score, falling_knife_score, is_buy_candidate, is_overvalued, position_multiplier, confidence_delta |
| bubble_detector ★ | BubbleSignal | phase(NORMAL/EUPHORIA/PEAK/BURST/PANIC), bubble_score, overheat_score, leverage_risk, cascade_risk, passive_risk, position_multiplier |
| exchange_rate ★ | ExchangeRateSignal | dollar_phase, fx_alarm, fx_change_pct, dollar_strength, risk_level, kospi_impact, position_multiplier, leverage_guard |
| execution_analysis ★ | ExecutionSignal | canslim_score, peg_ratio, optimal_time, candle_force, candle_force_intact, confidence_adjustment |
| book_integrator ★ | IntegratedSignal | composite_score(0-100), action(STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL/BLOCK), position_multiplier, max_leverage, module_scores, conflicts |

---

## 5. 전체 설정 (config/settings.py)

### 5.1 AppConfig 구조
```python
@dataclass
class AppConfig:
    kis: KISConfig                    # KIS API 연결
    db: DBConfig                      # DB 연결
    risk: RiskConfig                  # 리스크 관리
    strategy: StrategyConfig          # 전략 파라미터
    fundamental: FundamentalConfig    # S-RIM 재무분석
    cycle: CycleConfig                # 하워드 막스 사이클
    credit_cycle: CreditCycleConfig   # 신용사이클
    candle_master: CandleMasterConfig # 캔들마스터
    wizard: WizardConfig              # 잭 슈웨거
    ken_fisher: KenFisherConfig       # 켄 피셔
    value_investor: ValueInvestorConfig # 강방천&존리
    stock_quality: StockQualityConfig # 이남우
    seo_junsik: SeoJunsikConfig       # 서준식
    bubble_detector: BubbleDetectorConfig  # 조상철 ★신규
    exchange_rate: ExchangeRateConfig      # 백석현 ★신규
    book_integrator: BookIntegratorConfig  # 통합 ★신규
    backup: BackupConfig              # 백업
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
```

### 5.2 주요 설정 기본값

| Config | 파라미터 | 기본값 | 설명 |
|--------|---------|--------|------|
| RiskConfig | max_position_size | 2,000,000원 | 종목당 최대 |
| RiskConfig | max_daily_loss | -100,000원 | 일일 손실 한도 |
| RiskConfig | max_positions | 5개 | 동시 보유 한도 |
| RiskConfig | stop_loss_pct | -1.0% | 손절 기준 |
| RiskConfig | take_profit_pct | 2.0% | 익절 기준 |
| CycleConfig | early/mid/late_max_pos | 0.8/0.5/0.2 | 사이클별 포지션 비율 |
| CandleMasterConfig | max_position_pct | 10% | 종목당 비중 |
| CandleMasterConfig | stop_loss_default | -10% | 캔들마스터 손절 |
| WizardConfig | confidence_max_scale | 2.0 | 최대 확신도 배수 |
| WizardConfig | discipline_min_score | 50 | 최소 규율 점수 |
| SeoJunsikConfig | target_return | 15% | 기대수익률 기준 |
| SeoJunsikConfig | deep_value_threshold | 0.60 | 딥밸류 매수 임계 |

---

## 6. 핵심 코드 전문

### 6.1 strategies/base.py
```python
"""매매 전략 기본 인터페이스"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """매매 시그널"""
    type: SignalType
    stock_code: str
    price: int
    quantity: int = 0
    stop_loss: int = 0
    take_profit: int = 0
    confidence: float = 0.0  # 0.0 ~ 1.0
    reason: str = ""
    strategy_name: str = ""


class BaseStrategy(ABC):
    def __init__(self, name: str, params: Optional[dict] = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터를 분석하여 지표와 시그널 컬럼을 추가합니다."""

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, current_position: Optional[dict] = None) -> Signal:
        """가장 최근 데이터 기반으로 매매 시그널을 생성합니다."""
```

### 6.2 DB 모델 (data/database.py 핵심)
```python
class MinuteCandle(Base):
    __tablename__ = "minute_candles"
    stock_code: str, datetime: datetime, open/high/low/close: float, volume: int

class Position(Base):
    __tablename__ = "positions"
    stock_code, stock_name, quantity, avg_price, current_price,
    unrealized_pnl, unrealized_pnl_pct, strategy, is_active, opened_at

class Trade(Base):
    __tablename__ = "trades"
    stock_code, stock_name, strategy, side(BUY/SELL),
    price, quantity, pnl, pnl_pct, fee, tax, executed_at

class DailyPnL(Base):
    __tablename__ = "daily_pnl"
    date, total_pnl, realized_pnl, unrealized_pnl,
    total_fee, total_tax, trade_count, win_count, loss_count,
    win_rate, max_drawdown, portfolio_value

class FinancialData(Base):
    __tablename__ = "financial_data"
    stock_code, fiscal_year, revenue, operating_income, net_income,
    total_assets, total_equity, bps, eps, roe, per, pbr, debt_ratio,
    srim_value, updated_at
```

---

## 7. Swing Strategy 전체 필터 순서 (generate_signal)

```
1. 포지션 보유 중? → 손절(-1.5%) / 익절(+3.0%) / 트레일링스톱(-1.0%) / 기술적매도 검사
2. 포지션 미보유 & signal==BUY:
   a. [사이클필터] LATE phase → 차단 / EARLY → +10% / 전환점 → +15%
   b. [기본지표] 정배열(+15%), 거래량(+10%), MFI(+10%), GAP(+10%), VWAP(+5%)
   c. [캔들패턴] 양봉장악형(+10%), 샛별형(+15%)
   d. [주도주] 거래량급증+정배열(+10%)
   e. [파동필터] 무효조건 → 차단 / 후반부(+15%) / 매수구간70+(+10%)
   f. [캔들군] 매수(+10%) / 매도(-10%)
   g. [켄피셔] confidence_delta 적용, 강세장초기(+10%), 극단비관론(+8%), V자회복
   h. [가치투자] confidence_delta, 역발상(+표시), 고평가 → 차단
   i. [품질필터] 저품질(<0.3) → 차단, 패닉매수(+15%), 자본집약도
   j. [딥밸류] confidence_delta, 매수후보(표시), 칼날잡기(표시), 고평가 → 차단
   k. [마법사] 촉매없음 → 차단 / 강도>0.5(+10%) / 시너지STRONG(+15%) / 기대갭 → 차단
   l. [버블★] BURST/PANIC → 차단 / PEAK → -15% / EUPHORIA → -5%
   m. [환율★] 급변경보 → -10% / 달러강세 → -5% / 달러약세 → +5% / 편향보정
   n. [집행★] CANSLIM 70+ → +10% / PEG<1.0 → +5% / 장대봉붕괴 → -10%
   o. [통합★] BLOCK → 차단 / composite 70+ → +10% / 30- → -10%
3. ATR 기반 손절/익절 설정, confidence(min 0.5~max 1.0), reasons 문자열 조합
```

---

## 8. Risk Manager 포지션 사이징 캐스케이드

```
check_signal() 흐름:
1. 일일 손실 한도 확인 (-100,000원)
2. 동시 보유 종목 수 확인 (5개)
3. 동일 종목 중복 매수 방지
4. 재무건전성 필터 (ROE, 영업이익, 부채비율)
5. 물타기 방지 (현재가 < 평균매입가 시 차단)
6. 포지션 사이징:
   a. 사이클 기반 조정 (LATE→50%, TIGHT→70%, EARLY+FEAR→100%)
   b. 캔들마스터 한도 (총자산의 10% 또는 20%)
   c. 확신도 스케일링 (0.3x~2.0x, max_position_size 상한)
   d. 규율 점수 (<50 → 50% 제한)
7. 최소 신뢰도 확인 (0.5 미만 → HOLD)

apply_cycle_adjustment() 배수 캐스케이드:
  사이클배수 × 신용배수 × 켄피셔배수 × 가치투자배수 × 품질배수 × 딥밸류배수
  × 버블배수★ × 환율배수★
  → clamp(0.1, 1.5) [레버리지 가드 1.5x ★]
```

---

## 9. 의존성 (requirements.txt)

```
requests>=2.31.0          # KIS API 호출
websockets>=12.0          # 실시간 틱
pydantic>=2.5.0           # 데이터 검증
python-dotenv>=1.0.0      # .env 로드
pandas>=2.1.0             # 데이터 처리
numpy>=1.26.0             # 수학 연산
ta>=0.11.0                # 기술적 지표
sqlalchemy>=2.0.0         # ORM
scikit-learn>=1.3.0       # ML (향후)
lightgbm>=4.1.0           # ML (향후)
streamlit>=1.29.0         # 대시보드
fastapi>=0.104.0          # REST API
uvicorn[standard]>=0.24.0 # ASGI 서버
schedule>=1.2.0           # 스케줄링
loguru>=0.7.0             # 로깅
```

---

## 10. 환경 변수 (.env)

```ini
# 한국투자증권 API
KIS_APP_KEY=your_app_key
KIS_APP_SECRET=your_app_secret
KIS_ACCOUNT_NO=12345678-01

# 거래 모드
TRADING_MODE=paper   # paper 또는 live

# 데이터베이스
DB_USE_SQLITE=true
DB_SQLITE_PATH=data/ai_trader.db

# 감시 종목
WATCHLIST=005930,000660,035720

# 로깅
LOG_LEVEL=INFO
```

---

## 11. 테스트

```bash
cd C:\Copilot\ai_trader
python -m pytest tests/test_core.py -v
# 38개 테스트 케이스 (2026-03-22 기준, 전체 PASS):
# - 설정 로드/저장
# - 지표 계산 (RSI, BB, MACD, VWAP)
# - 시그널 생성 (BUY/SELL/HOLD)
# - 리스크 검사 (손실한도, 포지션한도)
# - 포지션 사이징
# - DB 모델 CRUD
# ★ TestBubbleDetector (5개): 과열감지, 연쇄매도, 버블단계, 경고메시지
# ★ TestExchangeRate (5개): 달러강도, FX경보, 위험선호도, 레버리지가드
# ★ TestExecutionAnalysis (4개): CANSLIM, PEG, 시간대, 통합분석
# ★ TestBookIntegrator (3개): 가중치합산, 통합분석, 충돌해소
```

---

## 12. 확장 가이드

### 12.1 새 도서 모듈 추가 패턴
```python
# 1. features/new_module.py 생성
@dataclass
class NewSignal:
    score: float = 0.5
    position_multiplier: float = 1.0
    confidence_delta: float = 0.0
    note: str = ""

class NewAnalyzer:
    def __init__(self, lookback: int = 252):
        self.lookback = lookback

    def analyze(self, df: pd.DataFrame) -> NewSignal:
        # OHLCV 기반 기술적 프록시 계산
        # ...
        return NewSignal(score=score, position_multiplier=mult)

# 2. config/settings.py에 Config 추가
@dataclass
class NewConfig:
    lookback: int = 252
    use_new_filter: bool = True

# AppConfig에 추가:
# new_module: NewConfig = field(default_factory=NewConfig)

# 3. strategies/swing.py에 통합
try:
    from features.new_module import NewAnalyzer, NewSignal
    _HAS_NEW = True
except ImportError:
    _HAS_NEW = False

# __init__에 분석기 초기화
# analyze()에 분석 결과 컬럼 추가
# generate_signal()에 차단/부스트 로직 추가

# 4. risk/manager.py의 apply_cycle_adjustment()에 배수 추가
```

### 12.2 핵심 제약사항
- **OHLCV만 사용**: 재무제표/뉴스 API 미연동 → 기술적 프록시로 대체
- **단일 종목 분석**: 각 도서 모듈은 개별 종목 DataFrame 기반
- **실시간 성능**: 5분 간격 매매 사이클, WebSocket은 옵션
- **신뢰도 기반**: confidence 0.0~1.0, 0.5 미만 HOLD 강제

---

## 13. 학습된 도서 목록 (9권)

| # | 도서명 | 저자 | 페이지 | OCR 파일 | 적용 모듈 | 인사이트 파일 |
|---|--------|------|--------|----------|-----------|-------------|
| 1 | 현명한 당신의 주식투자 교과서 | 박병창 | 290p | book_text.txt | candle_patterns, market_flow, **execution_analysis★** | 박병창-현명한_당신의_주식투자 교과서_insight.md |
| 2 | 투자와 마켓사이클의 법칙 | 하워드 막스 | 137p | howard_marks_text.txt | market_cycle, credit_cycle | 하워드막스_투자와_마켓사이의_사이클의_법칙_insight.md |
| 3 | 주식 캔들 매매법 | 캔들마스터 | 283p | candle_master_text.txt | wave_position, candle_patterns | (미생성) |
| 4 | 주식시장의 마법사들 | 잭 슈웨거 | 180p | jack_schwager_text.txt | wizard_discipline | 잭_슈웨거-주식시작의_마법사들_insight.md |
| 5 | 주식시장은 어떻게 반복되는가 | 켄 피셔 | 150p | ken_fisher_text.txt | market_memory | 켄_피셔_주식시장은_어떻게_반복되는가_insight.md |
| 6 | 나의 첫 주식 교과서 | 강방천 & 존리 | 113p | kang_jon_text.txt | value_investor | 강방천&존리 - 나의 첫 주식 교과서_Insights.md |
| 7 | 좋은 주식 나쁜 주식 | 이남우 | 161p | lee_namwoo_text.txt | stock_quality | 이남우-좋은_주식_나쁜_주식_insight.md |
| 8 | 다시 쓰는 주식투자 교과서 | 서준식 | 100p | seo_junsik_text.txt | deep_value | 서준식-다시_쓰는_주식투자_교과서_insight.md |
| 9 | 2026년 11월, 주식시장 대폭락 ★ | 조상철 | 104p | (텍스트 변환) | **bubble_detector★** | 조상철-2026년_11월,_주식시장_대폭락_insight.md |
| 10 | 환율 모르면 주식투자 절대로 하지 마라 | 백석현 | 103p | (텍스트 변환) | exchange_rate | 백석현-환율_모르면_주식투자_절대로_하지_마라_insight.md |
| 11 | 실전 퀀트투자 ★ | 홍용찬 | 184p | (이미지 캡처) | **quant_value★** | 홍용찬-실전_퀀트투자_insight.md |

**총 약 1,804페이지** 학습, OCR 텍스트 `C:\Copilot\` 및 `C:\Copilot\captures\- 텍스트 변환\` 에 UTF-8 보관.

**인사이트 파일 위치**: `C:\Copilot\captures\- 텍스트 변환\*_insight.md` (10개)

---

## 14. 새 도서 추가 시 체크리스트

새 투자 도서가 추가될 때마다 아래 순서로 처리합니다 (자동 업데이트 원칙):

```
1. OCR 텍스트 파일 → C:\Copilot\captures\- 텍스트 변환\[도서명].txt
2. 인사이트 파일 생성 → [도서명]_insight.md (통일 템플릿)
   - 📖 도서 개요 / 🎯 핵심 인사이트 / 📊 프로그래밍 적용 포인트
   - ⚠️ 리스크 관리 원칙 / 💡 핵심 수치 & 임계값 / 🔗 연계점
3. 기능 모듈 생성 → C:\Copilot\ai_trader\features\[모듈명].py
4. Config 추가 → config/settings.py (BubbleDetectorConfig 패턴 참조)
5. AppConfig 필드 추가
6. strategies/swing.py 통합 (try/except import guard 패턴)
7. risk/manager.py apply_cycle_adjustment() 배수 추가
8. tests/test_core.py 테스트 클래스 추가 (최소 3개)
9. AI_TRADER_FULL_CONTEXT.md 업데이트 (이 문서)
10. UPGRADE_PLAN.md 업데이트
```

---

*끝. 이 문서와 함께 프로젝트 소스코드를 전달하면 다른 AI가 전체 맥락을 파악할 수 있습니다.*
*최종 업데이트: 2026-03-22 | 테스트: 38/38 PASS*
