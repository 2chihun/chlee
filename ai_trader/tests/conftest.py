# -*- coding: utf-8 -*-
"""Shared pytest fixtures for AI Trader tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv_200():
    """200-bar OHLCV DataFrame with realistic price action."""
    np.random.seed(42)
    n = 200
    base_price = 50000
    dates = pd.date_range("2024-01-02 09:00", periods=n, freq="5min")
    close = base_price + np.cumsum(np.random.randn(n) * 100)
    close = np.maximum(close, 1000)

    df = pd.DataFrame({
        "datetime": dates,
        "open": close + np.random.randn(n) * 50,
        "high": close + abs(np.random.randn(n) * 100),
        "low": close - abs(np.random.randn(n) * 100),
        "close": close,
        "volume": np.random.randint(1000, 100000, n),
    })
    df["open"] = df["open"].clip(lower=100)
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df


@pytest.fixture
def ohlcv_300():
    """300-bar OHLCV DataFrame for modules requiring longer history."""
    np.random.seed(123)
    n = 300
    base_price = 50000
    dates = pd.date_range("2024-01-02 09:00", periods=n, freq="5min")
    close = base_price + np.cumsum(np.random.randn(n) * 100)
    close = np.maximum(close, 1000)

    df = pd.DataFrame({
        "datetime": dates,
        "open": close + np.random.randn(n) * 50,
        "high": close + abs(np.random.randn(n) * 100),
        "low": close - abs(np.random.randn(n) * 100),
        "close": close,
        "volume": np.random.randint(1000, 100000, n),
    })
    df["open"] = df["open"].clip(lower=100)
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df


@pytest.fixture
def ohlcv_short():
    """30-bar OHLCV DataFrame for insufficient-data edge cases."""
    np.random.seed(99)
    n = 30
    base_price = 50000
    dates = pd.date_range("2024-01-02 09:00", periods=n, freq="5min")
    close = base_price + np.cumsum(np.random.randn(n) * 100)
    close = np.maximum(close, 1000)

    df = pd.DataFrame({
        "datetime": dates,
        "open": close + np.random.randn(n) * 50,
        "high": close + abs(np.random.randn(n) * 100),
        "low": close - abs(np.random.randn(n) * 100),
        "close": close,
        "volume": np.random.randint(1000, 100000, n),
    })
    df["open"] = df["open"].clip(lower=100)
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df
