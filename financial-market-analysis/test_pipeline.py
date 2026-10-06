import sys, os
sys.path.append(r"C:\Users\dhana\.gemini\antigravity\scratch\financial-market-analysis")

from src.data_loader import fetch_stock_data
from src.indicators import add_technical_indicators, extract_signal_factors
from src.sentiment import analyze_news_sentiment
from src.model import prepare_ml_dataset, train_and_evaluate_xgboost, generate_live_signal
from src.utils import build_price_chart, build_indicator_charts, build_confusion_matrix_chart, build_feature_importance_chart

tickers_to_test = ['AAPL', 'TSLA', 'RELIANCE.NS', 'TCS.NS']

for t in tickers_to_test:
    print(f"=== Testing ticker: {t} ===")
    df, snap, status = fetch_stock_data(t, "1 year")
    print(f"Rows: {len(df)}, Status: {status}, Price: {snap['latest_price']}")
    assert len(df) >= 10, "Data too short"

    df_ind = add_technical_indicators(df)
    for col in ['SMA20', 'SMA50', 'EMA20', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower']:
        assert col in df_ind.columns, f"Missing {col}"
    print("Indicators OK")

    news = analyze_news_sentiment(t, limit=5)
    print(f"News articles: {len(news['articles'])}, Overall score: {news['overall_score']}, Label: {news['overall_label']}")
    assert len(news['articles']) > 0, "No news"

    X, y, latest = prepare_ml_dataset(df_ind, current_sentiment_score=news['overall_score'])
    print(f"X shape: {X.shape}, y shape: {y.shape}, latest shape: {latest.shape}")
    
    res = train_and_evaluate_xgboost(X, y, test_size=0.20)
    print(f"Accuracy: {res['accuracy']:.3f}, Precision: {res['precision']:.3f}, Recall: {res['recall']:.3f}, F1: {res['f1_score']:.3f}")
    
    sig = generate_live_signal(res['model'], latest)
    print(f"Signal: {sig['signal']}, Confidence: {sig['confidence']:.1f}%, Probs: {sig['probabilities']}")
    assert sig['signal'] in ['BUY', 'SELL', 'HOLD'], "Invalid signal"

    factors = extract_signal_factors(df_ind.iloc[-1])
    print(f"Factors count: {len(factors)}")

    # Test chart creation
    p_fig = build_price_chart(df_ind, t, {'sma20': True, 'sma50': True, 'ema20': True, 'bollinger': True})
    r_fig, m_fig = build_indicator_charts(df_ind)
    cm_fig = build_confusion_matrix_chart(res['confusion_matrix'])
    fi_fig = build_feature_importance_chart(res['feature_importance'])
    print(f"All charts constructed successfully for {t}!\n")

print(">>> ALL 4 TICKERS (AAPL, TSLA, RELIANCE.NS, TCS.NS) PASSED PIPELINE VERIFICATION! <<<")
