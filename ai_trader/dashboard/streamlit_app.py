"""AI Trader Streamlit 대시보드

실시간 포지션, 매매 이력, 백테스트, 통계, 시뮬레이션 UI
"""

import datetime as dt
import json

import requests
import streamlit as st
import pandas as pd
import numpy as np

# ── 설정 ────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="AI Trader 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def api_get(path: str, params: dict = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API 오류: {e}")
        return None


def api_post(path: str, data: dict = None):
    try:
        r = requests.post(f"{API_BASE}{path}", json=data, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API 오류: {e}")
        return None


# ── 사이드바 ────────────────────────────────────────────────────

st.sidebar.title("🤖 AI Trader")
page = st.sidebar.radio("메뉴", [
    "📊 대시보드",
    "📈 포지션",
    "📋 매매이력",
    "🔬 백테스트",
    "🎲 시뮬레이션",
    "📉 통계분석",
    "💾 백업관리",
    "⚙️ 설정",
])

# 봇 상태
status = api_get("/api/status")
if status:
    mode = status.get("trading_mode", "")
    running = status.get("bot_running", False)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**모드:** {mode}")
    st.sidebar.markdown(f"**상태:** {'🟢 실행 중' if running else '🔴 정지'}")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▶ 시작", disabled=running):
            api_post("/api/bot/control", {"action": "start"})
            st.rerun()
    with col2:
        if st.button("⏹ 정지", disabled=not running):
            api_post("/api/bot/control", {"action": "stop"})
            st.rerun()


# ── 페이지: 대시보드 ────────────────────────────────────────────

if page == "📊 대시보드":
    st.title("📊 AI Trader 대시보드")

    # 상단 카드
    col1, col2, col3, col4 = st.columns(4)

    stats = api_get("/api/stats/overall")
    if stats and "error" not in stats:
        col1.metric("총 수익", f"{stats.get('total_pnl', 0):,}원")
        col2.metric("승률", f"{stats.get('win_rate', 0):.1f}%")
        col3.metric("총 거래", f"{stats.get('total_trades', 0)}회")
        col4.metric("Profit Factor", f"{stats.get('profit_factor', 0):.2f}")
    else:
        col1.metric("총 수익", "0원")
        col2.metric("승률", "0%")
        col3.metric("총 거래", "0회")
        col4.metric("Profit Factor", "0.00")

    st.markdown("---")

    # 일별 수익 차트
    st.subheader("📈 일별 손익 추이")
    daily = api_get("/api/stats/daily_pnl", {"days": 30})
    if daily:
        df = pd.DataFrame(daily)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df["cumulative_pnl"] = df["total_pnl"].cumsum()
            st.line_chart(df.set_index("date")["cumulative_pnl"])
        else:
            st.info("아직 매매 데이터가 없습니다.")
    else:
        st.info("API 서버에 연결할 수 없습니다.")

    # 최근 거래
    st.subheader("📋 최근 거래")
    trades = api_get("/api/trades", {"limit": 10})
    if trades:
        df = pd.DataFrame(trades)
        if not df.empty:
            df = df[["executed_at", "stock_code", "stock_name", "strategy",
                      "side", "price", "quantity", "pnl", "pnl_pct"]]
            st.dataframe(df, use_container_width=True)
        else:
            st.info("거래 내역이 없습니다.")


# ── 페이지: 포지션 ──────────────────────────────────────────────

elif page == "📈 포지션":
    st.title("📈 현재 보유 포지션")

    positions = api_get("/api/positions")
    if positions:
        if len(positions) > 0:
            df = pd.DataFrame(positions)
            st.dataframe(df, use_container_width=True)

            total_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)
            st.metric("총 미실현 손익", f"{total_pnl:,}원")
        else:
            st.info("현재 보유 포지션이 없습니다.")
    else:
        st.warning("API 서버에 연결할 수 없습니다.")


# ── 페이지: 매매이력 ────────────────────────────────────────────

elif page == "📋 매매이력":
    st.title("📋 매매 체결 이력")

    col1, col2 = st.columns(2)
    with col1:
        limit = st.selectbox("조회 건수", [20, 50, 100, 200, 500], index=1)
    with col2:
        strategy_filter = st.selectbox("전략 필터", ["전체", "ScalpingStrategy", "SwingStrategy"])

    params = {"limit": limit}
    if strategy_filter != "전체":
        params["strategy"] = strategy_filter

    trades = api_get("/api/trades", params)
    if trades:
        df = pd.DataFrame(trades)
        if not df.empty:
            st.dataframe(df, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            sells = df[df["side"] == "SELL"]
            if not sells.empty:
                col1.metric("총 실현 손익", f"{sells['pnl'].sum():,}원")
                col2.metric("평균 수익률", f"{sells['pnl_pct'].mean():.2f}%")
                col3.metric("총 수수료+세금", f"{(sells['fee'].sum() + sells['tax'].sum()):,}원")
        else:
            st.info("매매 이력이 없습니다.")


# ── 페이지: 백테스트 ────────────────────────────────────────────

elif page == "🔬 백테스트":
    st.title("🔬 백테스트")

    tab1, tab2, tab3 = st.tabs(["📊 기본 백테스트", "🔧 파라미터 최적화", "🏋️ 스트레스 테스트"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            bt_strategy = st.selectbox("전략", ["scalping", "swing"], key="bt_strat")
            bt_stock = st.text_input("종목코드", "005930", key="bt_stock")
        with col2:
            bt_capital = st.number_input("초기 자본금", value=10_000_000, step=1_000_000, key="bt_cap")

        if st.button("백테스트 실행", type="primary"):
            with st.spinner("백테스트 실행 중..."):
                result = api_post("/api/backtest/run", {
                    "strategy": bt_strategy,
                    "stock_code": bt_stock,
                    "initial_capital": bt_capital,
                })
                if result:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("총 수익률", f"{result.get('total_return_pct', 0):.2f}%")
                    col2.metric("샤프 비율", f"{result.get('sharpe_ratio', 0):.2f}")
                    col3.metric("최대 낙폭", f"{result.get('max_drawdown_pct', 0):.2f}%")
                    col4.metric("승률", f"{result.get('win_rate', 0):.1f}%")

                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("총 거래 수", result.get("total_trades", 0))
                    col2.metric("Profit Factor", f"{result.get('profit_factor', 0):.2f}")
                    col3.metric("평균 보유 기간", f"{result.get('avg_holding_bars', 0):.1f}봉")

                    st.json(result)

    with tab2:
        opt_strategy = st.selectbox("전략", ["scalping", "swing"], key="opt_strat")
        opt_stock = st.text_input("종목코드", "005930", key="opt_stock")

        if st.button("최적화 실행", type="primary"):
            with st.spinner("파라미터 최적화 실행 중... (시간이 소요될 수 있습니다)"):
                result = api_post("/api/backtest/optimize", {
                    "strategy": opt_strategy,
                    "stock_code": opt_stock,
                })
                if result:
                    df = pd.DataFrame(result)
                    st.subheader("상위 10개 파라미터 조합")
                    display_cols = [
                        "total_return_pct", "sharpe_ratio", "max_drawdown_pct",
                        "win_rate", "total_trades", "profit_factor",
                    ]
                    available = [c for c in display_cols if c in df.columns]
                    st.dataframe(df[available], use_container_width=True)

    with tab3:
        stress_strategy = st.selectbox("전략", ["scalping", "swing"], key="stress_strat")
        stress_stock = st.text_input("종목코드", "005930", key="stress_stock")

        if st.button("스트레스 테스트 실행"):
            with st.spinner("스트레스 테스트 실행 중..."):
                result = api_get(
                    "/api/backtest/stress_test",
                    {"strategy": stress_strategy, "stock_code": stress_stock},
                )
                if result:
                    scenarios = list(result.keys())
                    data = []
                    for sc in scenarios:
                        r = result[sc]
                        data.append({
                            "시나리오": sc,
                            "수익률(%)": r.get("total_return_pct", 0),
                            "최대낙폭(%)": r.get("max_drawdown_pct", 0),
                            "샤프비율": r.get("sharpe_ratio", 0),
                            "승률(%)": r.get("win_rate", 0),
                        })
                    st.dataframe(pd.DataFrame(data), use_container_width=True)

    # 과거 백테스트 결과
    st.markdown("---")
    st.subheader("📜 백테스트 히스토리")
    history = api_get("/api/backtest/results")
    if history:
        df = pd.DataFrame(history)
        if not df.empty:
            st.dataframe(df, use_container_width=True)


# ── 페이지: 시뮬레이션 ──────────────────────────────────────────

elif page == "🎲 시뮬레이션":
    st.title("🎲 몬테카를로 시뮬레이션")

    col1, col2, col3 = st.columns(3)
    with col1:
        mc_strategy = st.selectbox("전략", ["scalping", "swing"])
    with col2:
        mc_stock = st.text_input("종목코드", "005930")
    with col3:
        mc_iterations = st.number_input("시뮬레이션 횟수", value=1000, step=100)

    if st.button("시뮬레이션 실행", type="primary"):
        with st.spinner(f"{mc_iterations}회 시뮬레이션 실행 중..."):
            result = api_post("/api/backtest/monte_carlo", {
                "strategy": mc_strategy,
                "stock_code": mc_stock,
                "n_simulations": mc_iterations,
            })
            if result and "error" not in result:
                col1, col2, col3 = st.columns(3)
                col1.metric("수익 확률", f"{result.get('prob_profit', 0):.1f}%")
                col2.metric("평균 수익률", f"{result.get('mean_return_pct', 0):.2f}%")
                col3.metric("최악 수익률", f"{result.get('worst_return_pct', 0):.2f}%")

                st.markdown("---")
                st.subheader("자본금 분포")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("5% 퍼센타일", f"{result.get('percentile_5', 0):,.0f}원")
                    st.metric("25% 퍼센타일", f"{result.get('percentile_25', 0):,.0f}원")
                with col2:
                    st.metric("75% 퍼센타일", f"{result.get('percentile_75', 0):,.0f}원")
                    st.metric("95% 퍼센타일", f"{result.get('percentile_95', 0):,.0f}원")

                st.markdown("---")
                st.metric("평균 MDD", f"{result.get('mean_max_drawdown', 0):.2f}%")
                st.metric("최악 MDD", f"{result.get('worst_max_drawdown', 0):.2f}%")


# ── 페이지: 통계분석 ────────────────────────────────────────────

elif page == "📉 통계분석":
    st.title("📉 통계 분석")

    tab1, tab2, tab3, tab4 = st.tabs([
        "전체 통계", "전략별", "월별 추이", "시간대 분석"
    ])

    with tab1:
        stats = api_get("/api/stats/overall")
        if stats and "error" not in stats:
            col1, col2, col3 = st.columns(3)
            col1.metric("총 수익", f"{stats.get('total_pnl', 0):,}원")
            col2.metric("승률", f"{stats.get('win_rate', 0):.1f}%")
            col3.metric("Profit Factor", f"{stats.get('profit_factor', 0):.2f}")

            col1, col2, col3 = st.columns(3)
            col1.metric("평균 이익률", f"{stats.get('avg_win_pct', 0):.2f}%")
            col2.metric("평균 손실률", f"{stats.get('avg_loss_pct', 0):.2f}%")
            col3.metric("최대 이익", f"{stats.get('max_win', 0):,}원")

            dd = api_get("/api/stats/drawdown")
            if dd:
                st.subheader("드로우다운 분석")
                col1, col2 = st.columns(2)
                col1.metric("최대 낙폭", f"{dd.get('max_drawdown_pct', 0):.2f}%")
                col2.metric("현재 낙폭", f"{dd.get('current_drawdown_pct', 0):.2f}%")
        else:
            st.info("매매 데이터가 없습니다.")

    with tab2:
        strategy_stats = api_get("/api/stats/strategy")
        if strategy_stats:
            for name, s in strategy_stats.items():
                st.subheader(f"📌 {name}")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("거래 수", s.get("trades", 0))
                col2.metric("승률", f"{s.get('win_rate', 0):.1f}%")
                col3.metric("총 수익", f"{s.get('total_pnl', 0):,}원")
                col4.metric("평균 수익률", f"{s.get('avg_pnl_pct', 0):.2f}%")
        else:
            st.info("전략별 데이터가 없습니다.")

    with tab3:
        monthly = api_get("/api/stats/monthly")
        if monthly:
            df = pd.DataFrame(monthly)
            if not df.empty:
                st.bar_chart(df.set_index("month")["total_pnl"])
                st.dataframe(df, use_container_width=True)
        else:
            st.info("월별 데이터가 없습니다.")

    with tab4:
        time_stats = api_get("/api/stats/time")
        if time_stats:
            df = pd.DataFrame([
                {"시간": f"{h}시", **v} for h, v in time_stats.items()
            ])
            if not df.empty:
                st.bar_chart(df.set_index("시간")["pnl"])
                st.dataframe(df, use_container_width=True)
        else:
            st.info("시간대별 데이터가 없습니다.")


# ── 페이지: 백업관리 ────────────────────────────────────────────

elif page == "💾 백업관리":
    st.title("💾 데이터 백업 관리")

    st.markdown("""
    ### 백업 정책
    - **DB 백업:** SQLite 파일 복사 / PostgreSQL pg_dump
    - **시세 데이터:** Parquet 형식으로 내보내기
    - **매매 이력:** CSV 형식으로 월별 내보내기
    - **자동 백업:** 매일 00:30 실행
    - **보관 기간:** 최근 30일
    """)

    if st.button("🔄 수동 백업 실행", type="primary"):
        with st.spinner("백업 실행 중..."):
            result = api_post("/api/backup/run")
            if result:
                st.success(f"백업 완료: {result.get('timestamp', '')}")


# ── 페이지: 설정 ────────────────────────────────────────────────

elif page == "⚙️ 설정":
    st.title("⚙️ 시스템 설정")

    if status:
        st.subheader("현재 설정")
        st.json(status)

    st.markdown("---")
    st.subheader("설정 가이드")
    st.markdown("""
    시스템 설정은 `.env` 파일을 통해 관리됩니다.

    **주요 설정 항목:**
    ```
    # 한국투자증권 API
    KIS_APP_KEY=your_app_key
    KIS_APP_SECRET=your_app_secret
    KIS_ACCOUNT_NO=12345678-01
    TRADING_MODE=paper  # paper=모의투자, live=실전

    # 데이터베이스
    DB_USE_SQLITE=true
    DB_HOST=localhost
    DB_PORT=5432

    # 리스크 관리
    MAX_POSITION_SIZE=2000000
    MAX_DAILY_LOSS=-100000
    MAX_POSITIONS=5
    STOP_LOSS_PCT=-1.0
    TAKE_PROFIT_PCT=2.0
    ```
    """)
