# 🌸 Crypto Trading Bot · 4-Component Linear Scoring

**Team 5**  
- Dinh Thi Hong Anh — tdin@edu.hse.ru  
- Panasenkova Polina Alexandrovna — papanasenkova@edu.hse.ru  

---

## 📌 Overview

Pure Python implementation using **OpenRouter API** with **Gemini 2.5 Flash Lite**.  
Single script handles: data loading → LLM analysis → 4‑component scoring → backtest → CSV export.

**Optimization:** Weights tuned via `scipy.optimize.differential_evolution` (`optimize.py`).

---

## 🤖 LLM Integration

- **Model:** `google/gemini-2.5-flash-lite` via OpenRouter  
- **Prompt:** Technical indicators + news, framed for a *conservative risk analyst*  
- **Output (JSON):**  
  `sentiment_score` (-1..1), `market_mood`, `trend_strength`, `reversal_probability`,  
  `risk_level`, `recommended_action`, `confidence`, `reasoning`

---

## 📐 Scoring Logic

4‑component linear score (each normalized to ≈ -1..+1):
score = W_SENTIMENT * sentiment
+ W_RSI * (50 - rsi) / 50
+ W_BB * (0.5 - bb_pos) / 0.5
± W_ACTION * llm_action_sign


### Optimized Weights (from `optimize.py`)

| Component          | Weight | Logic                          |
|--------------------|--------|--------------------------------|
| LLM Sentiment      | 17.94  | bullish → +                    |
| RSI                | 2.29   | low RSI → + (oversold)         |
| Bollinger %B       | 12.99  | low band → +                   |
| LLM Action         | 19.06  | buy: + / sell: –               |

### Decision Rules

IF risk_level = "extreme" → SELL
IF score ≥ 3.58 → BUY
IF score ≤ 4.75 → SELL
IF RSI > 75 → SELL (emergency)
ELSE → HOLD

### Risk Management
- **Extreme risk override** (LLM)
- **RSI > 75 emergency sell**
- **Position sizing:** 10% of remaining capital per buy

---

## 📊 Backtest Results

| Metric          | Value        |
|-----------------|--------------|
| Sharpe Ratio    | **3.3925**   |
| Total Return    | **+3.75%**   |
| Max Drawdown    | **-0.31%**   |
| Win Rate        | **100.0%**   |
| Total Trades    | 15 (8 buys, 7 sells) |
| Final Value     | $10,374.91   |

---

## 🗂️ File Structure

├── README.md 
├── features.csv # market data (513 rows, 10 coins)
├── news.csv # news headlines (50 articles)
├── main.py # full pipeline (LLM + scoring + backtest)
├── optimize.py # weight optimizer (differential evolution)
├── trades.csv # generated trade log (created on run)
├── workflow.html # pipeline visualization
