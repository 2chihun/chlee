"""FastAPI 백엔드 서버

트레이딩 봇 제어, 데이터 조회, 백테스트 실행 API
"""

import datetime as dt
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from config.settings import load_config, AppConfig
from data.database import (
    Database, Trade, Position, DailyPnL,
    DailyCandle, MinuteCandle, BacktestResult,
)
from data.backup import BackupManager, StatisticsEngine
from backtest.engine import BacktestEngine, CostModel
from strategies.scalping import ScalpingStrategy
from strategies.swing import SwingStrategy


# ── 전역 상태 ───────────────────────────────────────────────────

_config: Optional[AppConfig] = None
_db: Optional[Database] = None
_stats: Optional[StatisticsEngine] = None
_bot_running: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _config, _db, _stats
    _config = load_config()
    _db = Database(_config.db)
    _db.create_tables()
    _stats = StatisticsEngine(_db)
    logger.info("FastAPI 서버 시작")
    yield
    if _db:
        _db.close()
    logger.info("FastAPI 서버 종료")


app = FastAPI(
    title="AI Trader API",
    description="AI 단타 주식 투자 시스템 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic 모델 ───────────────────────────────────────────────

class BacktestRequest(BaseModel):
    strategy: str = "scalping"
    stock_code: str = "005930"
    start_date: str = ""
    end_date: str = ""
    initial_capital: int = 10_000_000
    params: dict = {}


class MonteCarloRequest(BaseModel):
    strategy: str = "scalping"
    stock_code: str = "005930"
    n_simulations: int = 1000
    params: dict = {}


class OptimizeRequest(BaseModel):
    strategy: str = "scalping"
    stock_code: str = "005930"
    param_grid: dict = {}
    metric: str = "sharpe_ratio"


class BotControlRequest(BaseModel):
    action: str  # "start", "stop", "status"


# ── 상태 API ────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    return {
        "bot_running": _bot_running,
        "trading_mode": "모의투자" if _config.kis.is_paper else "실전투자",
        "db_type": "SQLite" if _config.db.use_sqlite else "PostgreSQL",
        "timestamp": dt.datetime.now().isoformat(),
    }


@app.post("/api/bot/control")
def control_bot(req: BotControlRequest):
    global _bot_running
    if req.action == "start":
        _bot_running = True
        return {"message": "트레이딩 봇 시작", "running": True}
    elif req.action == "stop":
        _bot_running = False
        return {"message": "트레이딩 봇 정지", "running": False}
    elif req.action == "status":
        return {"running": _bot_running}
    raise HTTPException(400, f"알 수 없는 명령: {req.action}")


# ── 포지션/거래 API ─────────────────────────────────────────────

@app.get("/api/positions")
def get_positions():
    session = _db.get_session()
    try:
        positions = session.query(Position).all()
        return [{
            "stock_code": p.stock_code, "stock_name": p.stock_name,
            "strategy": p.strategy, "quantity": p.quantity,
            "avg_price": p.avg_price, "current_price": p.current_price,
            "unrealized_pnl": p.unrealized_pnl,
            "unrealized_pnl_pct": p.unrealized_pnl_pct,
            "entered_at": p.entered_at.isoformat() if p.entered_at else "",
        } for p in positions]
    finally:
        session.close()


@app.get("/api/trades")
def get_trades(
    limit: int = Query(default=50, le=500),
    strategy: Optional[str] = None,
    start_date: Optional[str] = None,
):
    session = _db.get_session()
    try:
        q = session.query(Trade).order_by(Trade.executed_at.desc())
        if strategy:
            q = q.filter(Trade.strategy == strategy)
        if start_date:
            q = q.filter(Trade.executed_at >= dt.datetime.strptime(start_date, "%Y%m%d"))
        trades = q.limit(limit).all()
        return [{
            "id": t.id, "stock_code": t.stock_code,
            "stock_name": t.stock_name, "strategy": t.strategy,
            "side": t.side, "price": t.price, "quantity": t.quantity,
            "amount": t.amount, "pnl": t.pnl, "pnl_pct": t.pnl_pct,
            "fee": t.fee, "tax": t.tax,
            "executed_at": t.executed_at.isoformat() if t.executed_at else "",
        } for t in trades]
    finally:
        session.close()


# ── 통계 API ────────────────────────────────────────────────────

@app.get("/api/stats/overall")
def get_overall_stats():
    return _stats.get_overall_stats()


@app.get("/api/stats/strategy")
def get_strategy_stats():
    return _stats.get_strategy_stats()


@app.get("/api/stats/monthly")
def get_monthly_stats():
    df = _stats.get_monthly_stats()
    if df.empty:
        return []
    df["month"] = df["month"].astype(str)
    return df.to_dict(orient="records")


@app.get("/api/stats/time")
def get_time_analysis():
    return _stats.get_time_analysis()


@app.get("/api/stats/drawdown")
def get_drawdown():
    return _stats.get_drawdown_analysis()


@app.get("/api/stats/daily_pnl")
def get_daily_pnl(days: int = Query(default=30, le=365)):
    session = _db.get_session()
    try:
        cutoff = dt.date.today() - dt.timedelta(days=days)
        records = session.query(DailyPnL).filter(
            DailyPnL.date >= cutoff
        ).order_by(DailyPnL.date).all()
        return [{
            "date": str(r.date), "total_pnl": r.total_pnl,
            "realized_pnl": r.realized_pnl,
            "trade_count": r.trade_count,
            "win_rate": r.win_rate,
        } for r in records]
    finally:
        session.close()


# ── 백테스트 API ────────────────────────────────────────────────

def _get_strategy(name: str, params: dict):
    if name == "scalping":
        return ScalpingStrategy(params if params else None)
    elif name == "swing":
        return SwingStrategy(params if params else None)
    else:
        raise HTTPException(400, f"알 수 없는 전략: {name}")


def _get_sample_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """DB에서 일봉 데이터를 조회합니다. 없으면 샘플 데이터를 생성합니다."""
    session = _db.get_session()
    try:
        q = session.query(DailyCandle).filter(
            DailyCandle.stock_code == stock_code
        ).order_by(DailyCandle.date)
        if start_date:
            q = q.filter(DailyCandle.date >= dt.datetime.strptime(start_date, "%Y%m%d").date())
        if end_date:
            q = q.filter(DailyCandle.date <= dt.datetime.strptime(end_date, "%Y%m%d").date())

        records = q.all()
        if records:
            return pd.DataFrame([{
                "date": r.date, "open": r.open, "high": r.high,
                "low": r.low, "close": r.close, "volume": r.volume,
            } for r in records])
    finally:
        session.close()

    # DB에 데이터 없으면 랜덤 샘플 생성 (데모용)
    logger.warning("DB에 {} 데이터 없음 — 샘플 데이터 생성", stock_code)
    import numpy as np
    np.random.seed(42)
    n = 250
    dates = pd.bdate_range(end=dt.date.today(), periods=n)
    base = 70000
    changes = np.random.normal(0, 0.02, n)
    closes = [base]
    for c in changes[1:]:
        closes.append(int(closes[-1] * (1 + c)))

    df = pd.DataFrame({
        "date": dates,
        "open": [int(c * (1 + np.random.uniform(-0.01, 0.01))) for c in closes],
        "high": [int(c * (1 + abs(np.random.normal(0, 0.015)))) for c in closes],
        "low": [int(c * (1 - abs(np.random.normal(0, 0.015)))) for c in closes],
        "close": closes,
        "volume": np.random.randint(1000000, 50000000, n),
    })
    return df


@app.post("/api/backtest/run")
def run_backtest(req: BacktestRequest):
    strategy = _get_strategy(req.strategy, req.params)
    df = _get_sample_data(req.stock_code, req.start_date, req.end_date)
    if df.empty:
        raise HTTPException(400, "데이터가 없습니다.")

    engine = BacktestEngine(strategy, req.initial_capital)
    result = engine.run(df, req.stock_code)

    # 결과 DB 저장
    session = _db.get_session()
    try:
        import json
        record = BacktestResult(
            name=f"{req.strategy}_{req.stock_code}_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}",
            strategy=req.strategy,
            start_date=pd.Timestamp(result.start_date).date() if result.start_date else dt.date.today(),
            end_date=pd.Timestamp(result.end_date).date() if result.end_date else dt.date.today(),
            initial_capital=result.initial_capital,
            final_capital=result.final_capital,
            total_return_pct=result.total_return_pct,
            annual_return_pct=result.annual_return_pct,
            max_drawdown_pct=result.max_drawdown_pct,
            sharpe_ratio=result.sharpe_ratio,
            win_rate=result.win_rate,
            total_trades=result.total_trades,
            avg_profit_pct=result.avg_profit_pct,
            avg_loss_pct=result.avg_loss_pct,
            profit_factor=result.profit_factor,
            params_json=json.dumps(result.params, ensure_ascii=False),
        )
        session.add(record)
        session.commit()
    finally:
        session.close()

    return result.to_dict()


@app.post("/api/backtest/monte_carlo")
def run_monte_carlo(req: MonteCarloRequest):
    strategy = _get_strategy(req.strategy, req.params)
    df = _get_sample_data(req.stock_code, "", "")
    if df.empty:
        raise HTTPException(400, "데이터가 없습니다.")

    engine = BacktestEngine(strategy)
    result = engine.monte_carlo(df, req.n_simulations, req.stock_code)
    return result


@app.post("/api/backtest/optimize")
def run_optimize(req: OptimizeRequest):
    strategy = _get_strategy(req.strategy, {})
    df = _get_sample_data(req.stock_code, "", "")
    if df.empty:
        raise HTTPException(400, "데이터가 없습니다.")

    if not req.param_grid:
        req.param_grid = {
            "rsi_buy": [25, 30, 35],
            "rsi_sell": [65, 70, 75],
            "stop_loss_pct": [-0.5, -1.0, -1.5],
            "take_profit_pct": [1.5, 2.0, 3.0],
        }

    engine = BacktestEngine(strategy)
    results = engine.optimize(df, req.param_grid, req.metric, req.stock_code)
    return [r.to_dict() for r in results[:10]]


@app.get("/api/backtest/stress_test")
def run_stress_test(
    strategy: str = "scalping",
    stock_code: str = "005930",
):
    strat = _get_strategy(strategy, {})
    df = _get_sample_data(stock_code, "", "")
    if df.empty:
        raise HTTPException(400, "데이터가 없습니다.")

    engine = BacktestEngine(strat)
    return engine.stress_test(df, stock_code)


@app.get("/api/backtest/results")
def get_backtest_results(limit: int = 20):
    session = _db.get_session()
    try:
        results = session.query(BacktestResult).order_by(
            BacktestResult.created_at.desc()
        ).limit(limit).all()
        return [{
            "id": r.id, "name": r.name, "strategy": r.strategy,
            "start_date": str(r.start_date), "end_date": str(r.end_date),
            "total_return_pct": r.total_return_pct,
            "max_drawdown_pct": r.max_drawdown_pct,
            "sharpe_ratio": r.sharpe_ratio,
            "win_rate": r.win_rate, "total_trades": r.total_trades,
            "created_at": r.created_at.isoformat() if r.created_at else "",
        } for r in results]
    finally:
        session.close()


# ── 백업 API ────────────────────────────────────────────────────

@app.post("/api/backup/run")
def run_backup():
    backup = BackupManager(_config.backup, _db)
    backup.run_full_backup()
    return {"message": "백업 완료", "timestamp": dt.datetime.now().isoformat()}


# ── 서버 실행 ───────────────────────────────────────────────────

def start_api_server():
    import uvicorn
    config = load_config()
    uvicorn.run(
        "dashboard.api_server:app",
        host=config.api_host,
        port=config.api_port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    start_api_server()
