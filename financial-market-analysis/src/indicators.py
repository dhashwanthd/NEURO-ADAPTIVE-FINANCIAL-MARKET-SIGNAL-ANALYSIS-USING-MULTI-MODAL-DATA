"""
Technical Indicators Module
===========================
Calculates key financial indicators from OHLCV market data:
- Simple Moving Averages (SMA 20, SMA 50)
- Exponential Moving Average (EMA 20)
- Relative Strength Index (RSI 14)
- Moving Average Convergence Divergence (MACD, Signal Line, Histogram)
- Bollinger Bands (Upper, Middle, Lower)
"""

import pandas as pd
import numpy as np


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes and appends technical indicators to the input stock price DataFrame.
    Expects standard OHLCV columns: ['Open', 'High', 'Low', 'Close', 'Volume'].
    
    Returns a copy of the DataFrame with added indicator columns.
    """
    if df is None or df.empty or len(df) < 5:
        return df

    # Create a copy to prevent SettingWithCopy warnings
    data = df.copy()

    # Ensure Close is float
    close = data['Close'].astype(float)

    # 1. Simple Moving Averages (SMA)
    data['SMA20'] = close.rolling(window=20, min_periods=1).mean()
    data['SMA50'] = close.rolling(window=50, min_periods=1).mean()

    # 2. Exponential Moving Average (EMA 20)
    data['EMA20'] = close.ewm(span=20, adjust=False).mean()

    # 3. Relative Strength Index (RSI 14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    # 14-period exponential moving average of gains and losses (Wilder's smoothing)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    # Avoid division by zero
    rs = avg_gain / (avg_loss + 1e-9)
    data['RSI'] = 100 - (100 / (1 + rs))
    # Fill initial NaN values gracefully with 50 (neutral)
    data['RSI'] = data['RSI'].fillna(50.0)

    # 4. Moving Average Convergence Divergence (MACD)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    # 5. Bollinger Bands (20 periods, 2 standard deviations)
    rolling_mean_20 = close.rolling(window=20, min_periods=1).mean()
    rolling_std_20 = close.rolling(window=20, min_periods=1).std().fillna(0)
    data['BB_Middle'] = rolling_mean_20
    data['BB_Upper'] = rolling_mean_20 + (2.0 * rolling_std_20)
    data['BB_Lower'] = rolling_mean_20 - (2.0 * rolling_std_20)

    # Ensure there are no lingering NaNs in the early indicator rows
    data = data.bfill().ffill()

    return data


def extract_signal_factors(latest_row: pd.Series, prev_row: pd.Series = None) -> list:
    """
    Evaluates rule-based technical conditions from the latest indicators
    to provide beginner-friendly, interpretable signal reasons for the final prediction.
    """
    factors = []
    
    close = float(latest_row.get('Close', 0))
    sma20 = float(latest_row.get('SMA20', 0))
    sma50 = float(latest_row.get('SMA50', 0))
    ema20 = float(latest_row.get('EMA20', 0))
    rsi = float(latest_row.get('RSI', 50))
    macd = float(latest_row.get('MACD', 0))
    macd_signal = float(latest_row.get('MACD_Signal', 0))
    bb_upper = float(latest_row.get('BB_Upper', 0))
    bb_lower = float(latest_row.get('BB_Lower', 0))
    sentiment = float(latest_row.get('Sentiment_Score', 0))

    # RSI condition
    if rsi < 30:
        factors.append({"factor": f"RSI is {rsi:.1f} (Oversold - potential rebound opportunity)", "type": "bullish"})
    elif rsi > 70:
        factors.append({"factor": f"RSI is {rsi:.1f} (Overbought - potential pullback risk)", "type": "bearish"})
    else:
        factors.append({"factor": f"RSI is {rsi:.1f} (Neutral momentum range: 30-70)", "type": "neutral"})

    # Moving Average Trend
    if close > sma20:
        factors.append({"factor": f"Current price (${close:.2f}) is trading above SMA20 (${sma20:.2f})", "type": "bullish"})
    else:
        factors.append({"factor": f"Current price (${close:.2f}) is trading below SMA20 (${sma20:.2f})", "type": "bearish"})

    if sma20 > sma50:
        factors.append({"factor": "Golden Trend: Short-term SMA20 is above medium-term SMA50", "type": "bullish"})
    else:
        factors.append({"factor": "Death Trend: Short-term SMA20 is below medium-term SMA50", "type": "bearish"})

    # MACD condition
    if macd > macd_signal:
        factors.append({"factor": f"MACD ({macd:.2f}) is above Signal line ({macd_signal:.2f}) - Bullish Momentum", "type": "bullish"})
    else:
        factors.append({"factor": f"MACD ({macd:.2f}) is below Signal line ({macd_signal:.2f}) - Bearish Momentum", "type": "bearish"})

    # Bollinger Bands condition
    if close >= bb_upper:
        factors.append({"factor": "Price is testing the Upper Bollinger Band (Volatility high)", "type": "bearish"})
    elif close <= bb_lower:
        factors.append({"factor": "Price is testing the Lower Bollinger Band (Potential support)", "type": "bullish"})
    else:
        factors.append({"factor": "Price is within the normal Bollinger Band envelope", "type": "neutral"})

    # News sentiment factor
    if sentiment > 0.15:
        factors.append({"factor": f"Financial News Sentiment is Positive (+{sentiment:.2f})", "type": "bullish"})
    elif sentiment < -0.15:
        factors.append({"factor": f"Financial News Sentiment is Negative ({sentiment:.2f})", "type": "bearish"})
    else:
        factors.append({"factor": f"Financial News Sentiment is Neutral ({sentiment:.2f})", "type": "neutral"})

    return factors
