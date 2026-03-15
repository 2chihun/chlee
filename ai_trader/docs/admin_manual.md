# AI Trader 관리자 매뉴얼

## 🛠️ 시스템 관리 & 운영 가이드

**버전:** 1.0.0  
**최종 업데이트:** 2025-01  
**대상 독자:** 시스템 관리자, DevOps

---

## 목차

1. [시스템 아키텍처](#1-시스템-아키텍처)
2. [설치 & 배포](#2-설치--배포)
3. [데이터베이스 관리](#3-데이터베이스-관리)
4. [백업 & 복구](#4-백업--복구)
5. [모니터링 & 로깅](#5-모니터링--로깅)
6. [API 레퍼런스](#6-api-레퍼런스)
7. [성능 튜닝](#7-성능-튜닝)
8. [보안 가이드](#8-보안-가이드)
9. [문제 해결](#9-문제-해결)
10. [부록: 코드 구조 상세](#10-부록-코드-구조-상세)

---

## 1. 시스템 아키텍처

### 1.1 전체 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                      AI Trader System                        │
│                                                              │
│  ┌─────────────┐     ┌──────────────┐    ┌───────────────┐  │
│  │  Streamlit   │────▶│   FastAPI     │───▶│   Trading     │  │
│  │  Dashboard   │◀────│   Server     │◀───│   Engine      │  │
│  │  (:8501)     │     │   (:8000)    │    │               │  │
│  └─────────────┘     └──────┬───────┘    └───────┬───────┘  │
│                             │                     │          │
│                      ┌──────▼───────┐    ┌───────▼───────┐  │
│                      │  SQLAlchemy   │    │   Strategies  │  │
│                      │  ORM Layer    │    │  ┌──────────┐ │  │
│                      └──────┬───────┘    │  │ Scalping  │ │  │
│                             │            │  │ Swing     │ │  │
│                      ┌──────▼───────┐    │  └──────────┘ │  │
│                      │  Database     │    └───────┬───────┘  │
│                      │  PostgreSQL   │            │          │
│                      │  + TimescaleDB│    ┌───────▼───────┐  │
│                      │  (or SQLite)  │    │ Risk Manager  │  │
│                      └──────────────┘    └───────┬───────┘  │
│                                                   │          │
│                                          ┌───────▼───────┐  │
│                                          │  KIS Order    │  │
│                                          │  Executor     │  │
│                                          └───────┬───────┘  │
│                                                   │          │
└───────────────────────────────────────────────────┼──────────┘
                                                    │
                                           ┌────────▼────────┐
                                           │ 한국투자증권 API │
                                           │  REST + WebSocket│
                                           └─────────────────┘
```

### 1.2 데이터 흐름

```
한국투자증권 API
       │
       ├── REST API (분봉, 일봉, 현재가)
       │       │
       │       ▼
       │   KISDataCollector
       │       │
       │       ├── 분봉 데이터 → add_all_indicators()
       │       │                      │
       │       │                      ▼
       │       │               Strategy.generate_signal()
       │       │                      │
       │       │                      ▼
       │       │               RiskManager.check_signal()
       │       │                      │
       │       │                      ▼
       │       │               KISOrderExecutor.buy()/sell()
       │       │                      │
       │       │                      ▼
       │       │               Trade → Database
       │       │
       │       └── 일봉 데이터 → DailyCandle → Database
       │
       └── WebSocket (실시간 틱)
               │
               ▼
           KISWebSocket
               │
               └── RealtimeTick, RealtimeOrderbook
```

### 1.3 모듈 의존성 맵

```
main.py
  ├── config.settings (AppConfig)
  ├── data.collector (KISAuth, KISDataCollector)
  ├── data.websocket_client (KISWebSocket)
  ├── data.database (Database)
  ├── data.backup (BackupManager, StatisticsEngine)
  ├── features.indicators (add_all_indicators)
  ├── strategies.scalping (ScalpingStrategy)
  ├── strategies.swing (SwingStrategy)
  ├── execution.order (KISOrderExecutor)
  └── risk.manager (RiskManager)

dashboard.api_server
  ├── config.settings
  ├── data.database
  ├── data.backup
  └── main (get_trader, start_bot, stop_bot)

dashboard.streamlit_app
  └── HTTP → FastAPI (:8000)
```

### 1.4 스레딩 모델

```
Main Thread (python main.py --mode both)
  │
  ├── Thread-1: AITrader.start()
  │     └── schedule loop (5분 간격 매매 사이클)
  │           ├── run_trading_cycle()
  │           ├── pre_market_routine()   @ 08:30
  │           ├── post_market_routine()  @ 15:25
  │           ├── daily_backup()         @ 00:30
  │           └── daily_stats()          @ 15:35
  │
  └── Main: uvicorn (FastAPI server :8000)
```

---

## 2. 설치 & 배포

### 2.1 시스템 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| CPU | 2코어 | 4코어+ |
| RAM | 4GB | 8GB+ |
| 스토리지 | 10GB | 50GB (시세 보관용) |
| Python | 3.10 | 3.11+ |
| OS | Windows 10 | Windows 11 / Ubuntu 22.04 |

### 2.2 Python 환경 설정

```bash
# 1. 가상환경 생성
python -m venv .venv

# 2. 활성화
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 추가 패키지 (PostgreSQL 사용 시)
pip install psycopg2-binary
# TimescaleDB 확장은 PostgreSQL 서버에 설치
```

### 2.3 PostgreSQL + TimescaleDB 설치

#### Windows

1. PostgreSQL 14+ 설치: https://www.postgresql.org/download/windows/
2. TimescaleDB 설치:
```powershell
# pgAdmin 또는 psql에서:
CREATE DATABASE ai_trader;
\c ai_trader
CREATE EXTENSION IF NOT EXISTS timescaledb;
```

#### Linux (Ubuntu)

```bash
# PostgreSQL
sudo apt install postgresql postgresql-contrib

# TimescaleDB
sudo add-apt-repository ppa:timescale/timescaledb-ppa
sudo apt update
sudo apt install timescaledb-2-postgresql-14

# 튜닝
sudo timescaledb-tune

# DB 생성
sudo -u postgres psql
CREATE DATABASE ai_trader;
\c ai_trader
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE USER ai_trader WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE ai_trader TO ai_trader;
```

### 2.4 서비스로 실행 (Linux systemd)

```ini
# /etc/systemd/system/ai-trader.service
[Unit]
Description=AI Trader Stock Trading Bot
After=network.target postgresql.service

[Service]
Type=simple
User=trader
WorkingDirectory=/opt/ai_trader
Environment=PATH=/opt/ai_trader/.venv/bin
ExecStart=/opt/ai_trader/.venv/bin/python main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable ai-trader
sudo systemctl start ai-trader
sudo systemctl status ai-trader
```

### 2.5 Windows 서비스 (NSSM)

```powershell
# NSSM 설치 후
nssm install AITrader "C:\Copilot\ai_trader\.venv\Scripts\python.exe" "main.py"
nssm set AITrader AppDirectory "C:\Copilot\ai_trader"
nssm start AITrader
```

---

## 3. 데이터베이스 관리

### 3.1 테이블 구조

```
┌──────────────────┐     ┌──────────────────┐
│   daily_candles   │     │  minute_candles   │
├──────────────────┤     ├──────────────────┤
│ id (PK)          │     │ id (PK)          │
│ stock_code       │     │ stock_code       │
│ date             │     │ datetime         │
│ open             │     │ open             │
│ high             │     │ high             │
│ low              │     │ low              │
│ close            │     │ close            │
│ volume           │     │ volume           │
│ created_at       │     │ created_at       │
└──────────────────┘     └──────────────────┘

┌──────────────────┐     ┌──────────────────┐
│     trades        │     │    positions      │
├──────────────────┤     ├──────────────────┤
│ id (PK)          │     │ id (PK)          │
│ stock_code       │     │ stock_code       │
│ stock_name       │     │ stock_name       │
│ strategy         │     │ strategy         │
│ side (BUY/SELL)  │     │ quantity         │
│ price            │     │ avg_price        │
│ quantity         │     │ is_active        │
│ pnl              │     │ entered_at       │
│ pnl_pct          │     │ created_at       │
│ fee              │     └──────────────────┘
│ tax              │
│ executed_at      │     ┌──────────────────┐
│ created_at       │     │    daily_pnl      │
└──────────────────┘     ├──────────────────┤
                         │ id (PK)          │
┌──────────────────┐     │ date             │
│  tick_data        │     │ total_pnl        │
├──────────────────┤     │ trade_count      │
│ id (PK)          │     │ win_count        │
│ stock_code       │     │ loss_count       │
│ datetime         │     │ win_rate         │
│ price            │     │ created_at       │
│ volume           │     └──────────────────┘
│ bid_price        │
│ ask_price        │     ┌──────────────────┐
│ created_at       │     │ backtest_results  │
└──────────────────┘     ├──────────────────┤
                         │ id (PK)          │
┌──────────────────┐     │ strategy         │
│   system_log      │     │ stock_code       │
├──────────────────┤     │ params (JSON)    │
│ id (PK)          │     │ metrics (JSON)   │
│ level            │     │ created_at       │
│ module           │     └──────────────────┘
│ message          │
│ created_at       │
└──────────────────┘
```

### 3.2 TimescaleDB 하이퍼테이블

시계열 데이터 테이블은 TimescaleDB 하이퍼테이블로 자동 변환됩니다:

```sql
-- 자동 설정 (database.py에서 수행)
SELECT create_hypertable('minute_candles', 'datetime',
                         if_not_exists => TRUE);
SELECT create_hypertable('tick_data', 'datetime',
                         if_not_exists => TRUE);
SELECT create_hypertable('daily_candles', 'date',
                         if_not_exists => TRUE);

-- 압축 정책 (30일 이상 데이터 자동 압축)
ALTER TABLE minute_candles SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'stock_code'
);
SELECT add_compression_policy('minute_candles',
                              INTERVAL '30 days');
```

### 3.3 인덱스

```sql
-- 주요 인덱스 (자동 생성)
CREATE INDEX ix_minute_candles_stock_datetime
    ON minute_candles(stock_code, datetime);

CREATE INDEX ix_trades_executed_at
    ON trades(executed_at);

CREATE INDEX ix_trades_strategy
    ON trades(strategy);

CREATE INDEX ix_positions_active
    ON positions(is_active) WHERE is_active = TRUE;
```

### 3.4 데이터 관리 쿼리

```sql
-- 데이터 크기 확인
SELECT hypertable_name,
       pg_size_pretty(hypertable_size(format('%I.%I',
           hypertable_schema, hypertable_name)::regclass))
FROM timescaledb_information.hypertables;

-- 오래된 데이터 삭제 (90일 이전)
DELETE FROM minute_candles
WHERE datetime < NOW() - INTERVAL '90 days';

-- 압축 상태 확인
SELECT * FROM timescaledb_information.compressed_hypertable_stats;
```

---

## 4. 백업 & 복구

### 4.1 자동 백업 정책

| 항목 | 주기 | 형식 | 보관기간 |
|------|------|------|---------|
| DB 전체 백업 | 매일 00:30 | SQLite copy / pg_dump | 30일 |
| 시세 데이터 | 매일 00:30 | Parquet (snappy) | 30일 |
| 매매 이력 | 매일 00:30 | CSV | 30일 |

### 4.2 백업 디렉토리 구조

```
backups/
├── db/
│   ├── ai_trader_20250115_003000.db        # SQLite 백업
│   └── ai_trader_20250115_003000.sql.gz    # PostgreSQL dump
├── parquet/
│   ├── minute_candles_20250115.parquet
│   └── daily_candles_20250115.parquet
└── csv/
    └── trades_202501.csv
```

### 4.3 수동 백업

```bash
# API 경유
curl -X POST http://localhost:8000/api/backup/run

# 대시보드에서: 💾 백업관리 → 수동 백업 실행
```

#### PostgreSQL 수동 백업

```bash
# 전체 백업
pg_dump -h localhost -U postgres ai_trader > backup.sql

# 압축 백업
pg_dump -h localhost -U postgres ai_trader | gzip > backup.sql.gz

# 특정 테이블만
pg_dump -h localhost -U postgres -t trades ai_trader > trades.sql
```

### 4.4 복구

#### SQLite 복구

```bash
# 백업 파일로 복구
copy backups\db\ai_trader_20250115_003000.db data\ai_trader.db
```

#### PostgreSQL 복구

```bash
# DB 재생성 후 복구
dropdb ai_trader
createdb ai_trader
psql ai_trader < backup.sql

# 압축 백업 복구
gunzip -c backup.sql.gz | psql ai_trader
```

### 4.5 Parquet 데이터 활용

```python
import pandas as pd

# Parquet 파일 읽기
df = pd.read_parquet("backups/parquet/minute_candles_20250115.parquet")

# 특정 종목 필터
samsung = df[df["stock_code"] == "005930"]
```

---

## 5. 모니터링 & 로깅

### 5.1 로그 구조

```
logs/
├── trader_2025-01-15.log    # 일별 로그 (30일 보관)
├── trader_2025-01-14.log
└── ...
```

### 5.2 로그 레벨

| 레벨 | 용도 |
|------|------|
| DEBUG | 상세 디버깅 (파일에만 기록) |
| INFO | 정상 운영 로그 |
| WARNING | 주의 필요 (미청산 포지션 등) |
| ERROR | 오류 발생 |
| CRITICAL | 시스템 중단급 오류 |

### 5.3 로그 예시

```
09:05:12 |   INFO   | 📈 매수 신호 | 삼성전자(005930) | 수량: 3 | 가격: 72,000원 | 전략: scalping | 신뢰도: 0.75
09:05:13 |   INFO   | ✅ 매수 체결 | 삼성전자 3주 @ 72,000
09:32:45 |   INFO   | 📉 매도 신호 | 삼성전자(005930) | 수량: 3 | 가격: 72,800원 | 전략: scalping
09:32:46 |   INFO   | 🟢 매도 체결 | 삼성전자 3주 @ 72,800 | 손익: 2,200원 (1.11%)
15:25:00 |   INFO   | === 장 마감 정리 루틴 시작 ===
15:35:00 |   INFO   | === 일일 통계 === 총 거래: 12회 | 승률: 66.7% | 총 손익: 15,800원
```

### 5.4 모니터링 포인트

| 항목 | 확인 방법 | 주기 |
|------|----------|------|
| 봇 실행 상태 | 대시보드 사이드바 or `GET /api/status` | 상시 |
| 일일 손익 | 대시보드 메인 | 매일 |
| API 토큰 만료 | 로그 확인 (자동 갱신) | 자동 |
| DB 용량 | 관리 쿼리 | 주간 |
| 백업 상태 | 백업 디렉토리 확인 | 매일 |

### 5.5 헬스체크

```bash
# API 서버 상태
curl http://localhost:8000/api/status

# 응답 예시
{
  "bot_running": true,
  "trading_mode": "paper",
  "watchlist_count": 8,
  "active_positions": 2,
  "daily_pnl": 15800,
  "uptime": "3h 25m"
}
```

---

## 6. API 레퍼런스

### 6.1 엔드포인트 목록

#### 시스템

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/status` | 시스템 상태 조회 |
| POST | `/api/bot/control` | 봇 제어 (start/stop) |

#### 포지션 & 거래

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/positions` | 보유 포지션 목록 |
| GET | `/api/trades?limit=50&strategy=scalping` | 거래 내역 조회 |

#### 통계

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/stats/overall` | 전체 통계 |
| GET | `/api/stats/strategy` | 전략별 통계 |
| GET | `/api/stats/monthly` | 월별 통계 |
| GET | `/api/stats/time` | 시간대별 통계 |
| GET | `/api/stats/drawdown` | 드로우다운 분석 |
| GET | `/api/stats/daily_pnl?days=30` | 일별 PnL |

#### 백테스트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/backtest/run` | 백테스트 실행 |
| POST | `/api/backtest/monte_carlo` | 몬테카를로 시뮬레이션 |
| POST | `/api/backtest/optimize` | 파라미터 최적화 |
| GET | `/api/backtest/stress_test` | 스트레스 테스트 |
| GET | `/api/backtest/results` | 백테스트 결과 조회 |

#### 백업

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/backup/run` | 수동 백업 실행 |

### 6.2 요청/응답 예시

#### 백테스트 실행

```bash
curl -X POST http://localhost:8000/api/backtest/run \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "scalping",
    "stock_code": "005930",
    "initial_capital": 10000000
  }'
```

```json
{
  "initial_capital": 10000000,
  "final_capital": 10450000,
  "total_return_pct": 4.5,
  "sharpe_ratio": 1.82,
  "max_drawdown_pct": -3.2,
  "win_rate": 62.5,
  "total_trades": 48,
  "profit_factor": 1.67,
  "avg_holding_bars": 8.3
}
```

#### 봇 제어

```bash
# 시작
curl -X POST http://localhost:8000/api/bot/control \
  -H "Content-Type: application/json" \
  -d '{"action": "start"}'

# 정지
curl -X POST http://localhost:8000/api/bot/control \
  -H "Content-Type: application/json" \
  -d '{"action": "stop"}'
```

---

## 7. 성능 튜닝

### 7.1 데이터베이스 튜닝

#### PostgreSQL 설정 (postgresql.conf)

```ini
# 메모리
shared_buffers = 1GB            # RAM의 25%
effective_cache_size = 3GB      # RAM의 75%
work_mem = 64MB

# WAL
wal_buffers = 16MB
max_wal_size = 2GB

# 쿼리 플래너
random_page_cost = 1.1          # SSD 사용 시
effective_io_concurrency = 200  # SSD 사용 시

# TimescaleDB
timescaledb.max_background_workers = 4
```

#### SQLite 튜닝 (자동 적용)

```python
# database.py에서 자동 설정
PRAGMA journal_mode = WAL;      # Write-Ahead Logging
PRAGMA synchronous = NORMAL;    # 성능 향상
PRAGMA cache_size = -64000;     # 64MB 캐시
```

### 7.2 API 호출 최적화

| 항목 | 설정 |
|------|------|
| REST API 호출 간격 | 50ms (초당 20회 이내) |
| WebSocket 구독 종목 | 최대 40개 |
| 대량 데이터 조회 | 분봉 최대 100개씩 페이징 |

### 7.3 메모리 최적화

```python
# Pandas DataFrame 메모리 최적화
df["volume"] = df["volume"].astype("int32")     # int64 → int32
df["close"] = df["close"].astype("float32")     # float64 → float32

# Parquet 압축 (snappy)
df.to_parquet("data.parquet", compression="snappy")
```

---

## 8. 보안 가이드

### 8.1 API 키 보안

```
⚠️ 보안 규칙
├── .env 파일은 .gitignore에 포함 (절대 커밋 금지)
├── API 키는 환경 변수로만 관리
├── 모의투자/실전투자 키 분리 관리
└── 정기적 키 갱신 (권장: 3개월)
```

### 8.2 네트워크 보안

- FastAPI 서버는 기본적으로 로컬호스트에서만 접근
- 외부 접근 필요 시 리버스 프록시 (nginx) + HTTPS 설정
- 방화벽에서 8000, 8501 포트 외부 차단

### 8.3 데이터베이스 보안

```sql
-- PostgreSQL: 전용 사용자 생성
CREATE USER ai_trader WITH PASSWORD 'strong_password_here';
GRANT CONNECT ON DATABASE ai_trader TO ai_trader;
GRANT USAGE ON SCHEMA public TO ai_trader;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO ai_trader;
```

### 8.4 체크리스트

- [ ] `.env` 파일 권한 제한 (chmod 600)
- [ ] `.gitignore`에 `.env` 포함 확인
- [ ] PostgreSQL 패스워드 복잡성 확인
- [ ] API 서버 로컬 접근만 허용
- [ ] 정기적 백업 확인
- [ ] 로그 파일 권한 확인

---

## 9. 문제 해결

### 9.1 일반 문제

#### API 토큰 만료 오류

```
증상: "접근토큰 발급 잔여횟수 부족"
원인: 토큰 갱신 실패
해결:
1. 로그에서 에러 확인
2. .env의 APP_KEY, APP_SECRET 확인
3. 한국투자증권 포털에서 앱키 상태 확인
4. 봇 재시작
```

#### 주문 실패

```
증상: rt_cd != "0"
원인: 잔고 부족, 주문 가격 오류, 시장 외 시간
해결:
1. 에러 코드 확인 (rt_cd, msg_cd)
2. 계좌 잔고 확인
3. 장 시간 확인 (09:00~15:20)
4. 호가 단위 확인
```

#### DB 연결 실패

```
증상: OperationalError
원인: PostgreSQL 미실행, 연결 정보 오류
해결:
1. PostgreSQL 서비스 상태 확인
   - Windows: services.msc → postgresql-x64-14
   - Linux: sudo systemctl status postgresql
2. .env DB 설정 확인
3. pg_hba.conf 인증 설정 확인
```

### 9.2 성능 문제

#### 높은 CPU 사용

```
원인: 과도한 API 호출, 대량 지표 계산
해결:
1. 감시 종목 수 줄이기
2. API 호출 간격 늘리기 (0.1초 → 0.2초)
3. 불필요한 지표 계산 제거
```

#### 메모리 부족

```
원인: 대량 데이터 메모리 적재
해결:
1. DataFrame 타입 최적화 (float64 → float32)
2. 오래된 데이터 자동 삭제 정책 설정
3. TimescaleDB 압축 정책 확인
```

### 9.3 로그 확인 방법

```bash
# 최근 로그 확인
type logs\trader_2025-01-15.log | findstr "ERROR"

# Linux
tail -f logs/trader_$(date +%Y-%m-%d).log
grep "ERROR\|WARNING" logs/trader_$(date +%Y-%m-%d).log
```

---

## 10. 부록: 코드 구조 상세

### 10.1 핵심 클래스 다이어그램

```
┌──────────────────────┐
│     AppConfig         │
├──────────────────────┤
│ + kis: KISConfig      │
│ + db: DBConfig        │
│ + risk: RiskConfig    │
│ + strategy: StratConf │
│ + backup: BackupConf  │
└───────────┬──────────┘
            │ uses
            ▼
┌──────────────────────┐     ┌──────────────────┐
│     AITrader          │────▶│   KISAuth         │
├──────────────────────┤     ├──────────────────┤
│ + config              │     │ + get_token()     │
│ + db                  │     └──────────────────┘
│ + collector           │
│ + executor            │     ┌──────────────────┐
│ + risk                │────▶│ KISDataCollector  │
│ + strategies          │     ├──────────────────┤
│ + watchlist           │     │ + get_current()   │
├──────────────────────┤     │ + get_candles()   │
│ + start()             │     └──────────────────┘
│ + stop()              │
│ + run_trading_cycle() │     ┌──────────────────┐
│ + pre_market()        │────▶│   BaseStrategy    │
│ + post_market()       │     ├──────────────────┤
└──────────────────────┘     │ + analyze()       │
                              │ + gen_signal()    │
┌──────────────────────┐     ├──────────────────┤
│   RiskManager         │     │ ScalpingStrategy  │
├──────────────────────┤     │ SwingStrategy     │
│ + check_signal()      │     └──────────────────┘
│ + position_sizing()   │
│ + daily_pnl           │     ┌──────────────────┐
└──────────────────────┘     │ BacktestEngine    │
                              ├──────────────────┤
┌──────────────────────┐     │ + run()           │
│  KISOrderExecutor     │     │ + optimize()      │
├──────────────────────┤     │ + monte_carlo()   │
│ + buy()               │     │ + walk_forward()  │
│ + sell()              │     │ + stress_test()   │
│ + get_balance()       │     └──────────────────┘
└──────────────────────┘
```

### 10.2 설정 파일 참조

| 파일 | 용도 |
|------|------|
| `.env` | 비밀 설정 (API 키, DB 비밀번호) |
| `config/settings.py` | 앱 설정 클래스 (환경 변수 로드) |
| `requirements.txt` | Python 패키지 의존성 |

### 10.3 수수료/세금 모델

```
매수 비용:
  주문금액 + 수수료(0.015%) + 슬리피지(0.1%)

매도 수익:
  주문금액 - 수수료(0.015%) - 세금(0.18%) - 슬리피지(0.1%)

세금:
  매도 시에만: 증권거래세 0.18% (코스피 기준)
  코스닥: 0.18%, 중소기업: 0.10%
```

### 10.4 주요 환경 변수 목록

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| KIS_APP_KEY | (필수) | 한국투자증권 앱키 |
| KIS_APP_SECRET | (필수) | 시크릿키 |
| KIS_ACCOUNT_NO | (필수) | 계좌번호 |
| TRADING_MODE | paper | paper/live |
| DB_USE_SQLITE | true | SQLite 사용 여부 |
| DB_HOST | localhost | PostgreSQL 호스트 |
| DB_PORT | 5432 | PostgreSQL 포트 |
| DB_NAME | ai_trader | DB 이름 |
| MAX_POSITION_SIZE | 2000000 | 종목당 최대 투자금 |
| MAX_DAILY_LOSS | -100000 | 일일 손실 한도 |
| MAX_POSITIONS | 5 | 최대 동시 포지션 |
| STOP_LOSS_PCT | -1.0 | 손절 기준 |
| TAKE_PROFIT_PCT | 2.0 | 익절 기준 |

---

> 📌 이 문서는 AI Trader 시스템의 관리 및 운영을 위한 참조 문서입니다.  
> 시스템 변경 시 반드시 이 문서를 업데이트하시기 바랍니다.
