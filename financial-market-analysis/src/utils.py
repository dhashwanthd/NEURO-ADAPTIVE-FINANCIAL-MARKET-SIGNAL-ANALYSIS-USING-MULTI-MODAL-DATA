"""
Utility Helpers and Visualization Module
========================================
Interactive Plotly chart builders, formatting helpers, and pipeline visualizers
for the Neuro-Adaptive Financial Market Signal Analysis dashboard.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any


def format_large_number(num: float) -> str:
    """Formats large integer or float values into human-readable compact notation (e.g., 34.2M)."""
    if num is None:
        return "N/A"
    abs_num = abs(num)
    if abs_num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif abs_num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif abs_num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return f"{num:,.0f}"


def format_currency(value: float, currency_sym: str = "$") -> str:
    """Formats price values with appropriate currency symbol and precision."""
    if value is None:
        return "N/A"
    return f"{currency_sym}{value:,.2f}"


def build_price_chart(df: pd.DataFrame, ticker: str, overlays: Dict[str, bool]) -> go.Figure:
    """
    Builds an interactive high-resolution financial chart with Candlesticks,
    Volume bars, and toggleable Technical Indicator overlays (SMA, EMA, Bollinger Bands).
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{ticker} Price & Overlays", "Trading Volume")
    )

    # 1. Candlestick Trace
    candlestick = go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price (OHLC)",
        increasing_line_color='#00C087',
        decreasing_line_color='#FF4D4F'
    )
    fig.add_trace(candlestick, row=1, col=1)

    # 2. Bollinger Bands Overlays
    if overlays.get('bollinger', True) and 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Upper'],
            line=dict(color='rgba(147, 112, 219, 0.4)', width=1, dash='dash'),
            name='BB Upper (20, 2σ)',
            hoverinfo='skip'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Lower'],
            line=dict(color='rgba(147, 112, 219, 0.4)', width=1, dash='dash'),
            name='BB Lower (20, 2σ)',
            fill='tonexty',
            fillcolor='rgba(147, 112, 219, 0.07)',
            hoverinfo='skip'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Middle'],
            line=dict(color='rgba(147, 112, 219, 0.7)', width=1.2),
            name='BB Middle (SMA 20)'
        ), row=1, col=1)

    # 3. SMA 20 Overlay
    if overlays.get('sma20', True) and 'SMA20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA20'],
            line=dict(color='#38bdf8', width=1.5),
            name='SMA 20'
        ), row=1, col=1)

    # 4. SMA 50 Overlay
    if overlays.get('sma50', True) and 'SMA50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA50'],
            line=dict(color='#f59e0b', width=1.5),
            name='SMA 50'
        ), row=1, col=1)

    # 5. EMA 20 Overlay
    if overlays.get('ema20', False) and 'EMA20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA20'],
            line=dict(color='#ec4899', width=1.5),
            name='EMA 20'
        ), row=1, col=1)

    # 6. Volume Bar Subplot
    vol_colors = ['#00C087' if c >= o else '#FF4D4F' for o, c in zip(df['Open'], df['Close'])]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        marker_color=vol_colors,
        name='Volume',
        opacity=0.75
    ), row=2, col=1)

    # Styling and Layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#161b22',
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        margin=dict(l=40, r=40, t=35, b=25),
        height=520,
        hovermode="x unified"
    )
    fig.update_xaxes(showgrid=True, gridcolor='#262c36', linecolor='#30363d')
    fig.update_yaxes(showgrid=True, gridcolor='#262c36', linecolor='#30363d')

    return fig


def build_indicator_charts(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """
    Constructs dedicated interactive visualizers for:
    1. RSI (Relative Strength Index with 70/30 thresholds)
    2. MACD (MACD line, Signal line, and colored momentum histogram)
    """
    # --- RSI Chart ---
    rsi_fig = go.Figure()
    
    rsi_fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI'],
        line=dict(color='#a855f7', width=2),
        name='RSI 14'
    ))

    # Overbought reference line (70)
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="Overbought (70)", annotation_position="top right")
    # Oversold reference line (30)
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="#10b981", annotation_text="Oversold (30)", annotation_position="bottom right")
    # Center reference line (50)
    rsi_fig.add_hline(y=50, line_dash="dot", line_color="#64748b")

    rsi_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#161b22',
        title=dict(text="Relative Strength Index (RSI 14)", font=dict(size=14)),
        margin=dict(l=35, r=35, t=35, b=20),
        height=260,
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor='#262c36'),
        xaxis=dict(showgrid=True, gridcolor='#262c36'),
        showlegend=False
    )

    # --- MACD Chart ---
    macd_fig = go.Figure()

    # MACD Line
    macd_fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD'],
        line=dict(color='#38bdf8', width=1.8),
        name='MACD (12, 26)'
    ))

    # Signal Line
    macd_fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD_Signal'],
        line=dict(color='#f97316', width=1.5, dash='dash'),
        name='Signal Line (9)'
    ))

    # Histogram
    hist_colors = ['#10b981' if val >= 0 else '#ef4444' for val in df['MACD_Hist']]
    macd_fig.add_trace(go.Bar(
        x=df.index, y=df['MACD_Hist'],
        marker_color=hist_colors,
        name='MACD Hist',
        opacity=0.65
    ))

    macd_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#161b22',
        title=dict(text="MACD (Moving Average Convergence Divergence)", font=dict(size=14)),
        margin=dict(l=35, r=35, t=35, b=20),
        height=260,
        xaxis=dict(showgrid=True, gridcolor='#262c36'),
        yaxis=dict(showgrid=True, gridcolor='#262c36'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9))
    )

    return rsi_fig, macd_fig


def build_confusion_matrix_chart(cm_df: pd.DataFrame) -> go.Figure:
    """Builds a styled heatmap of the multi-class confusion matrix."""
    fig = go.Figure(data=go.Heatmap(
        z=cm_df.values,
        x=cm_df.columns.tolist(),
        y=cm_df.index.tolist(),
        colorscale='Viridis',
        text=cm_df.values,
        texttemplate="%{text}",
        textfont=dict(size=14, color="white"),
        hoverongaps=False
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#161b22',
        margin=dict(l=40, r=40, t=30, b=30),
        height=280,
        xaxis=dict(title="Predicted Class"),
        yaxis=dict(title="Actual Class")
    )
    return fig


def build_feature_importance_chart(feat_df: pd.DataFrame) -> go.Figure:
    """Builds an interactive horizontal bar chart of XGBoost feature importances."""
    top_feats = feat_df.head(10).sort_values(by="Importance", ascending=True)
    
    fig = go.Figure(go.Bar(
        x=top_feats['Importance'],
        y=top_feats['Feature'],
        orientation='h',
        marker=dict(
            color=top_feats['Importance'],
            colorscale='Tealgrn'
        )
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#161b22',
        margin=dict(l=40, r=30, t=30, b=30),
        height=280,
        xaxis=dict(title="Relative Importance Score", showgrid=True, gridcolor='#262c36'),
        yaxis=dict(showgrid=False)
    )
    return fig
