"""
Stock Market Data Loader Module
===============================
Fetches historical stock market prices using Yahoo Finance (yfinance).
Provides local caching and intelligent fallback generation to guarantee
uninterrupted demonstration during offline evaluations or network throttling.
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any


PERIOD_DAYS = {
    "1 month": ("1mo", 30),
    "3 months": ("3mo", 90),
    "6 months": ("6mo", 180),
    "1 year": ("1y", 365),
    "2 years": ("2y", 730),
    "5 years": ("5y", 1825),
}

DEFAULT_BASE_PRICES = {
    "AAPL": 225.0,
    "TSLA": 250.0,
    "MSFT": 420.0,
    "NVDA": 125.0,
    "AMZN": 185.0,
    "RELIANCE.NS": 1200.0,
    "TCS.NS": 4250.0,
    "INFY.NS": 1850.0,
    "HDFCBANK.NS": 1650.0,
    "DEFAULT": 150.0,
}


def generate_fallback_data(ticker: str, period_label: str) -> pd.DataFrame:
    """
    Generates realistic synthetic OHLCV time-series using Geometric Brownian Motion (GBM).
    Used as an emergency fallback if yfinance is unreachable or network is offline.
    """
    clean_ticker = ticker.upper().strip()
    base_price = DEFAULT_BASE_PRICES.get(clean_ticker, DEFAULT_BASE_PRICES["DEFAULT"])
    _, days = PERIOD_DAYS.get(period_label, ("1y", 365))
    
    # Generate business days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)
    
    if n_days < 10:
        dates = pd.bdate_range(end=end_date, periods=30)
        n_days = len(dates)

    np.random.seed(42 + sum(ord(c) for c in clean_ticker))
    
    # Geometric Brownian Motion
    daily_returns = np.random.normal(loc=0.0005, scale=0.018, size=n_days)
    price_series = base_price * np.exp(np.cumsum(daily_returns))
    
    opens = price_series * (1 + np.random.normal(0, 0.004, size=n_days))
    highs = np.maximum(opens, price_series) * (1 + np.abs(np.random.normal(0, 0.008, size=n_days)))
    lows = np.minimum(opens, price_series) * (1 - np.abs(np.random.normal(0, 0.008, size=n_days)))
    closes = price_series
    volumes = np.random.randint(15_000_000, 85_000_000, size=n_days)

    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes
    }, index=dates)
    df.index.name = "Date"
    return df


def fetch_stock_data(ticker: str, period_label: str = "1 year", force_fallback: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """
    Downloads historical stock market data from yfinance.
    Normalizes columns, sanitizes timezones, extracts latest snapshot metrics,
    and returns (data_df, snapshot_metrics, status_message).
    Supports force_fallback for offline viva demonstrations.
    """
    clean_ticker = ticker.upper().strip()
    period_code, _ = PERIOD_DAYS.get(period_label, ("1y", 365))
    
    df = None
    status_msg = "Live Market Data"
    currency = "₹" if clean_ticker.endswith(".NS") or clean_ticker.endswith(".BO") else "$"

    if force_fallback:
        status_msg = "Offline Evaluation Mode: Loaded academic synthetic market data."
        df = generate_fallback_data(clean_ticker, period_label)
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        latest_close = float(latest['Close'])
        prev_close = float(prev['Close'])
        daily_change = latest_close - prev_close
        pct_change = (daily_change / prev_close) * 100 if prev_close != 0 else 0.0

        snapshot = {
            "ticker": clean_ticker,
            "period": period_label,
            "currency": currency,
            "latest_price": latest_close,
            "daily_change": daily_change,
            "pct_change": pct_change,
            "volume": int(latest['Volume']),
            "open": float(latest['Open']),
            "high": float(latest['High']),
            "low": float(latest['Low']),
            "close": latest_close,
            "record_count": len(df),
            "start_date": df.index[0].strftime("%Y-%m-%d"),
            "end_date": df.index[-1].strftime("%Y-%m-%d"),
        }
        return df, snapshot, status_msg

    try:
        t = yf.Ticker(clean_ticker)
        # Fetch history
        hist = t.history(period=period_code)
        
        if hist is not None and not hist.empty and len(hist) >= 5:
            df = hist.copy()
            # Clean up columns and index
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Keep required standard columns
            req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in req_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing expected OHLCV column: {col}")
            df = df[req_cols].astype(float)
        else:
            status_msg = "Live data empty; loaded academic synthetic market data."
            df = generate_fallback_data(clean_ticker, period_label)

    except Exception as e:
        status_msg = f"Yahoo Finance connection issue ({str(e)[:35]}...); loaded fallback market data."
        df = generate_fallback_data(clean_ticker, period_label)

    # Calculate latest snapshot statistics
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    latest_close = float(latest['Close'])
    prev_close = float(prev['Close'])
    daily_change = latest_close - prev_close
    pct_change = (daily_change / prev_close) * 100 if prev_close != 0 else 0.0

    snapshot = {
        "ticker": clean_ticker,
        "period": period_label,
        "currency": currency,
        "latest_price": latest_close,
        "daily_change": daily_change,
        "pct_change": pct_change,
        "volume": int(latest['Volume']),
        "open": float(latest['Open']),
        "high": float(latest['High']),
        "low": float(latest['Low']),
        "close": latest_close,
        "record_count": len(df),
        "start_date": df.index[0].strftime("%Y-%m-%d"),
        "end_date": df.index[-1].strftime("%Y-%m-%d"),
    }

    return df, snapshot, status_msg
