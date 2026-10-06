# Neuro-Adaptive Financial Market Signal Analysis Using Multi-Modal Data

A complete, runnable, end-to-end Data Science Project designed for academic Problem-Based Learning (PBL) evaluations and quantitative market research.

---

## 1. Project Title
**Neuro-Adaptive Financial Market Signal Analysis Using Multi-Modal Data**

---

## 2. Project Description
Financial market price action is inherently non-linear, volatile, and driven by a combination of historical numerical price momentum and external qualitative sentiment. Traditional forecasting models often rely solely on historical prices or isolated technical rules. 

This project implements a **multi-modal quantitative trading signal analysis system** that fuses:
1. **Historical Stock Market OHLCV Data** (Open, High, Low, Close, Volume)
2. **Mathematical Technical Indicators** (SMA 20, SMA 50, EMA 20, RSI 14, MACD, Bollinger Bands)
3. **Financial News Sentiment Analysis** via Natural Language Processing (VADER)
4. **Machine Learning Model** (XGBoost Classifier) trained on chronological partitions to avoid lookahead bias

The application delivers actionable **BUY**, **SELL**, or **HOLD** recommendations with quantified confidence percentages and interpretable rule-based explanations through an interactive **Streamlit** dashboard.

---

## 3. Key Features
- **Asset Universe**: Pre-configured support for global megacaps (`AAPL`, `TSLA`, `MSFT`, `NVDA`, `AMZN`) and Indian equities (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`), plus custom ticker input.
- **Dynamic Lookback Periods**: 1 month, 3 months, 6 months, 1 year, 2 years, and 5 years.
- **Interactive Candlestick & Volume Visualizers**: Built with Plotly with toggleable SMA 20, SMA 50, EMA 20, and Bollinger Bands envelopes.
- **Technical Oscillators**: Separate interactive visualizers for RSI (14) with overbought/oversold boundaries and MACD with Signal line and Histogram.
- **NLP News Sentiment Pipeline**: Real-time Google News RSS parsing coupled with NLTK VADER sentiment scoring, mapping headlines to a normalized `[-1.0, +1.0]` sentiment index.
- **Offline / Viva Resilience**: Built-in fallback demo datasets and realistic synthetic market generators so the system runs smoothly even if internet drops or Yahoo Finance rate-limits during viva examinations.
- **Chronological ML Validation**: Strict time-based train/test splitting (80% train, 20% test) to prevent future data leakage.
- **Model Evaluation Suite**: Displays test Accuracy, Precision, Recall, F1 Score, interactive 3x3 Confusion Matrix, and XGBoost Feature Importances.
- **Explainable AI (XAI)**: Breaks down the final signal into clear, human-understandable technical and sentiment factors.

---

## 4. Technologies Used
- **Language**: Python 3.10+ / Python 3.14+
- **User Interface**: Streamlit
- **Market Data Feed**: `yfinance`
- **Data Manipulation & Numerics**: `pandas`, `numpy`
- **Machine Learning**: `xgboost`, `scikit-learn`
- **Data Visualization**: `plotly`, `matplotlib`
- **Natural Language Processing**: `nltk` (VADER Lexicon), `requests`, `xml.etree.ElementTree`

*Note: In accordance with project specifications, deep learning frameworks (TensorFlow, PyTorch) were intentionally excluded in favor of lightweight, fast, and highly interpretable gradient-boosted decision trees (XGBoost).*

---

## 5. System Architecture & Pipeline

```text
Data Collection (yfinance OHLCV)
       ↓
Data Preprocessing & Cleaning
       ↓
Technical Indicator Extraction (SMA, EMA, RSI, MACD, BB)
       ↓
Financial News Sentiment Analysis (VADER NLP Scoring)
       ↓
Multi-Modal Feature Fusion Matrix
       ↓
Chronological Train/Test Split (No Data Leakage)
       ↓
XGBoost Multi-Class Classifier Training
       ↓
Evaluation Metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
       ↓
Live Snapshot Inference → BUY / SELL / HOLD Signal + Confidence %
       ↓
Interactive Streamlit Dashboard
```

---

## 6. Installation

1. Open PowerShell or Command Prompt.
2. Navigate to the project directory:
   ```powershell
   cd "C:\Users\dhana\.gemini\antigravity\scratch\financial-market-analysis"
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 7. How to Run

Launch the Streamlit web dashboard:
```powershell
streamlit run app.py
```
The application will open automatically in your default browser at `http://localhost:8501`.

---

## 8. Example Usage
1. **Select an Asset**: Choose `AAPL` (Apple), `TSLA` (Tesla), `RELIANCE.NS` (Reliance), or `TCS.NS` (Tata Consultancy Services) from the sidebar.
2. **Select Historical Period**: Select `1 year` for optimal machine-learning training samples.
3. **Customize Chart**: Toggle overlays like `SMA 20`, `SMA 50`, and `Bollinger Bands`.
4. **Click "🚀 Analyze Stock"**:
   - The app downloads recent market data.
   - Calculates momentum oscillators and Bollinger bands.
   - Scrapes recent financial headlines and computes sentiment polarity.
   - Retrains the XGBoost model chronologically.
   - Displays real-time BUY / SELL / HOLD prediction, confidence, and metric breakdowns.

---

## 9. Machine Learning Methodology
- **Multi-Modal Feature Space**:
  - Price: `Open`, `High`, `Low`, `Close`, `Volume`
  - Trend: `SMA20`, `SMA50`, `EMA20`
  - Momentum: `RSI` (14-day), `MACD`, `MACD_Signal`
  - Volatility: `BB_Upper`, `BB_Middle`, `BB_Lower`
  - Sentiment: `Sentiment_Score` (aggregate NLP score)
- **Target Labeling**:
  - Forward Return: $R_{t+1} = \frac{Close_{t+1} - Close_t}{Close_t}$
  - Target Signal:
    - $\text{BUY (Class 2)}$ if $R_{t+1} > +0.3\%$
    - $\text{SELL (Class 0)}$ if $R_{t+1} < -0.3\%$
    - $\text{HOLD (Class 1)}$ if $-0.3\% \le R_{t+1} \le +0.3\%$
- **No Lookahead Bias**:
  - The latest row ($t=\text{today}$) has an unknown future target. It is strictly excluded from training and reserved solely for inference.
  - The dataset is split chronologically (first 80% for training, subsequent 20% for testing), avoiding random shuffle leakage common in flawed academic pipelines.

---

## 10. Adaptive / Neuro-Adaptive Aspect
The term **"Neuro-Adaptive"** in this project refers to the **closed-loop continuous calibration** of the system:
- When any asset is selected or time elapses, the pipeline automatically ingests the latest price bars and news articles.
- Technical parameters and sentiment distributions are recalibrated on the fly.
- The XGBoost model dynamically retrains on the latest historical window to adapt its split thresholds to current market regime shifts (e.g., transition from bull run to high-volatility consolidation).

---

## 11. Limitations & Future Work
- Daily frequency: Does not capture intraday high-frequency tick data.
- Headline-only sentiment: Full earnings transcripts and 10-K regulatory filings could provide deeper long-term sentiment insights.
- Future work could incorporate macroeconomic indicators (interest rates, CPI inflation, bond yields) into the multi-modal matrix.

---

## 12. Disclaimer
**Disclaimer:** This software is developed purely for educational and academic demonstration purposes as part of a college Problem-Based Learning (PBL) curriculum. It does not constitute investment or financial advice.
