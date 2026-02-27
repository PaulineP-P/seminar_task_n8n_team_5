"""
🌸 CRYPTO TRADING BOT · 4-COMPONENT LINEAR SCORING + TELEGRAM
==============================================================
Uses cached LLM analysis + optimized weights (from optimize.py)
Features: sentiment, RSI, Bollinger Bands, LLM action, Telegram alerts
"""

import os
import json
import math
import time
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from datetime import datetime

# Загружаем переменные из .env файла
load_dotenv()

# 💗 CONFIGURATION (из .env файла)
# ============================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("❌ ОШИБКА: OPENROUTER_API_KEY не найден в .env файле!")
    print("📝 Добавьте строку: OPENROUTER_API_KEY=ваш_ключ_от_openrouter")
    exit(1)

# Telegram Configuration (из .env файла)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)  # Автоматически включаем, если есть токен и chat_id
SEND_ONLY_SIGNALS = True  # Отправлять только buy/sell сигналы

if TELEGRAM_ENABLED:
    print(f"🤖 Telegram бот настроен: Chat ID = {TELEGRAM_CHAT_ID}")
else:
    print("⚠️ Telegram уведомления отключены (не указан TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID)")

MODEL = "google/gemini-2.5-flash-lite"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# input/output files
FEATURES_CSV = "features.csv"
NEWS_CSV = "news.csv"
OUTPUT_CSV = "trades.csv"

# backtest settings
STARTING_CAPITAL = 10000
TRANSACTION_FEE = 0.001  # 0.1%

# OPTIMIZED WEIGHTS (from optimize.py)
W_SENTIMENT = 17.9413   # llm sentiment dominates
W_RSI = 2.2901          # rsi adds small signal
W_BB = 12.9918          # bollinger bands matter
W_ACTION = 19.0590      # llm action is crucial
BUY_THRESH = 3.5803     # buy if score ≥ this
SELL_THRESH = 4.7507    # sell if score ≤ this
ALLOC_PCT = 0.10        # 10% of capital per buy


# 🤖 TELEGRAM BOT CLASS
# ============================================================
class TradingTelegramBot:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    def send_message(self, text):
        """Send message to Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            }
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            print(f"   ✅ Telegram message sent")
            return True
        except Exception as e:
            print(f"   ⚠️ Telegram send error: {e}")
            return False
    
    def send_photo(self, photo_path, caption=""):
        """Send photo to Telegram"""
        try:
            url = f"{self.base_url}/sendPhoto"
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {'chat_id': self.chat_id, 'caption': caption}
                response = requests.post(url, files=files, data=data, timeout=30)
                response.raise_for_status()
            return True
        except Exception as e:
            print(f"⚠️ Telegram photo error: {e}")
            return False


# 🧠 LLM EXPLAINER MODULE
# ============================================================
def generate_market_explanation(row, analysis, decision, reason):
    """
    Генерирует понятное объяснение для отправки в Telegram
    """
    # Извлекаем данные
    ticker = row['ticker']
    price = float(row['close'])
    rsi = float(row.get('rsi', 50))
    sentiment = float(analysis.get('sentiment_score', 0) or 0)
    market_mood = analysis.get('market_mood', 'neutral')
    risk_level = analysis.get('risk_level', 'medium')
    
    # Определяем эмодзи в зависимости от решения
    if decision == 'buy':
        signal_emoji = "🟢 ПОКУПКА"
    elif decision == 'sell':
        signal_emoji = "🔴 ПРОДАЖА"
    else:
        signal_emoji = "⚪ ДЕРЖАТЬ"
    
    # Создаем промпт для LLM
    prompt = f"""You are a cautious trading assistant providing educational analysis only.
Never give direct buy/sell advice. Explain conditions and scenarios.

Analyze this market data:

Symbol: {ticker}
Date: {row['date']}
Price: ${price:.4f}
RSI: {rsi:.1f}
Sentiment: {market_mood} (score: {sentiment:.2f})
Risk Level: {risk_level}
System Decision: {decision.upper()}
Reason: {reason}

Provide:
1. Current market condition (2-3 sentences) - explain what's happening right now
2. Key technical signals (2 bullet points) - what the indicators show
3. Sentiment context from news - how news affects the market
4. Possible scenarios: bullish/bearish/neutral with brief reasoning
5. Risk factors to watch - what could go wrong

Format for mobile reading. Use emojis. Keep under 300 words.
Be educational, not advisory. Use simple language."""

    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "max_tokens": 500,
        }
        
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        explanation = data["choices"][0]["message"]["content"]
        
        # Форматируем итоговое сообщение
        final_message = f"""<b>{signal_emoji} {ticker}</b> @ ${price:.4f}
📅 {row['date']}

{explanation}

<code>RSI: {rsi:.1f} | Sentiment: {sentiment:.2f} | Риск: {risk_level}</code>
<i>ℹ️ Образовательный анализ, не инвестиционный совет</i>"""
        
        return final_message
        
    except Exception as e:
        print(f"   ⚠️ LLM explanation error: {e}")
        # Fallback сообщение
        rsi_desc = "перекуплен" if rsi > 70 else "перепродан" if rsi < 30 else "нейтрален"
        
        fallback = f"""<b>{signal_emoji} {ticker}</b> @ ${price:.4f}
📅 {row['date']}

📊 <b>Технический анализ:</b>
• RSI: {rsi:.1f} ({rsi_desc})
• Сентимент: {sentiment:.2f} ({market_mood})
• Уровень риска: {risk_level}

💡 <b>Решение системы:</b> {decision.upper()}
<i>{reason}</i>

<code>⚠️ Детальный анализ временно недоступен</code>"""
        return fallback


# 📦 MODULE 1: DATA LOADING
# ============================================================
def load_data():
    """load market features + news, merge by date"""
    print("🌸 loading market data...")
    features = pd.read_csv(FEATURES_CSV)
    print(f"   → {len(features)} rows, {len(features.columns)} columns")

    print("🌸 loading news data...")
    news = pd.read_csv(NEWS_CSV)
    print(f"   → {len(news)} articles")

    # group news by date for quick lookup
    news_by_date = {}
    for _, row in news.iterrows():
        date = str(row["date"]).strip()
        title = str(row.get("title", ""))
        body = str(row.get("body", ""))
        if date not in news_by_date:
            news_by_date[date] = []
        news_by_date[date].append(f"{title}: {body[:200]}")

    def get_news_text(date):
        """get up to 3 news snippets for a given date"""
        articles = news_by_date.get(str(date).strip(), [])
        return "\n\n".join(articles[:3]) if articles else "No significant news today"

    # attach news to market data
    features["news_text"] = features["date"].apply(get_news_text)
    features = features.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"   📅 date range: {features['date'].min()} → {features['date'].max()}")
    print(f"   💰 tickers: {sorted(features['ticker'].unique())}")
    print(f"   📰 rows with news: {(features['news_text'] != 'No significant news today').sum()} ({features['news_text'].ne('No significant news today').mean()*100:.1f}%)")

    return features


# 🤖 MODULE 2: LLM PROMPT BUILDER
# ============================================================
def build_prompt(row):
    """
    create structured prompt for gemini 2.5 flash
    includes technical indicators + news (if any)
    """
    close = float(row.get("close", 0))
    ma7 = float(row.get("ma7", 0))
    ma20 = float(row.get("ma20", 0)) if pd.notna(row.get("ma20")) else 0
    rsi = float(row.get("rsi", 50))
    macd_hist = float(row.get("macd_hist", 0))
    bb_pos = float(row.get("bb_position", 0.5))
    vol7d = float(row.get("volatility_7d", 0))
    returns = float(row.get("returns", 0))

    # human-readable zones
    rsi_zone = "OVERSOLD" if rsi < 30 else ("OVERBOUGHT" if rsi > 70 else "neutral")
    bb_zone = "NEAR LOWER BAND" if bb_pos < 0.2 else ("NEAR UPPER BAND" if bb_pos > 0.8 else "middle")
    has_news = row["news_text"] != "No significant news today"

    # prompt designed for structured JSON output
    prompt = f"""You are a conservative cryptocurrency risk analyst. Output ONLY valid JSON.

Date: {row['date']} | Coin: {row['ticker']}
Price: ${close} | Return: {returns*100:.2f}% | Volatility: {vol7d*100:.2f}%
RSI: {rsi:.1f} [{rsi_zone}] | MACD: {macd_hist:.6f} | BB: {bb_pos:.3f} [{bb_zone}]
MA7: ${ma7:.4f} | MA20: {f'${ma20:.4f}' if ma20 > 0 else 'N/A'}

News: {row['news_text'] if has_news else 'None - use technicals only.'}

Output JSON: {{"sentiment_score": <-1 to 1>, "market_mood": "<bearish|neutral|bullish>", "trend_strength": <0-1>, "reversal_probability": <0-1>, "risk_level": "<low|medium|high|extreme>", "recommended_action": "<buy|sell|hold>", "confidence": <0-1>, "reasoning": "<brief>"}}"""

    return prompt


# 🌐 MODULE 3: OPENROUTER API CLIENT
# ============================================================
def call_llm(prompt, max_retries=3):
    """
    call gemini 2.5 flash via openrouter
    includes retry logic with exponential backoff
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return json.loads(data["choices"][0]["message"]["content"])
        except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"   ⚠️ retry {attempt+1}/{max_retries}: {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                # fallback response
                return {
                    "sentiment_score": 0, 
                    "market_mood": "neutral", 
                    "trend_strength": 0,
                    "reversal_probability": 0, 
                    "risk_level": "medium",
                    "recommended_action": "hold", 
                    "confidence": 0, 
                    "reasoning": "LLM failed"
                }


# 📐 MODULE 4: TRADING LOGIC
# ============================================================
def trading_decision(analysis, indicators):
    """
    convert llm output + indicators into buy/sell/hold
    uses 4-component linear model with optimized weights
    """
    sentiment = float(analysis.get("sentiment_score", 0) or 0)
    risk_level = str(analysis.get("risk_level", "medium")).lower()
    llm_action = str(analysis.get("recommended_action", "hold")).lower()
    confidence = float(analysis.get("confidence", 0) or 0)

    rsi = float(indicators.get("rsi", 50))
    bb_pos = float(indicators.get("bb_position", 0.5))

    # RISK GATE #1: extreme risk → force sell
    if risk_level == "extreme":
        return "sell", "EXTREME RISK (forced sell)", confidence

    # 4-component linear score
    score = (W_SENTIMENT * sentiment                     # sentiment: -1..+1
             + W_RSI * (50 - rsi) / 50                   # rsi: 0..100 → -1..+1
             + W_BB * (0.5 - bb_pos) / 0.5)              # bb: 0..1 → -1..+1

    # llm action: buy → +weight, sell → -weight, hold → 0
    if llm_action == "buy": 
        score += W_ACTION
    elif llm_action == "sell": 
        score -= W_ACTION

    # apply thresholds
    decision = "hold"
    reason = f"score={score:.1f}"

    if score >= BUY_THRESH:
        decision = "buy"
        reason = f"BUY signal: score={score:.1f} (sent={sentiment:.2f}, rsi={rsi:.0f}, bb={bb_pos:.2f}, llm={llm_action})"

    if score <= SELL_THRESH:
        decision = "sell"
        reason = f"SELL signal: score={score:.1f} (sent={sentiment:.2f}, rsi={rsi:.0f}, bb={bb_pos:.2f}, llm={llm_action})"

    # EMERGENCY GATE #2: rsi > 75 overrides everything
    if rsi > 75:
        decision = "sell"
        reason = f"EMERGENCY SELL: RSI overbought ({rsi:.1f} > 75)"

    return decision, reason, confidence


# 📊 MODULE 5: BACKTESTING ENGINE
# ============================================================
def run_backtest(trades_df):
    """
    simulate trading with $10k starting capital
    tracks positions, calculates sharpe, drawdown, win rate
    """
    print("\n📈 === BACKTESTING ===")
    capital = STARTING_CAPITAL
    positions = {}          # ticker → {qty, avg_price}
    daily_portfolio_values = []
    trade_results = []      # track each trade for win rate

    dates = sorted(trades_df["date"].unique())
    for date in dates:
        day_trades = trades_df[trades_df["date"] == date]
        
        # execute all trades for this day
        for _, trade in day_trades.iterrows():
            ticker, price, decision = trade["ticker"], float(trade["price"]), trade["decision"]

            if decision == "buy" and capital > 0:
                # allocate fixed % of remaining capital
                alloc = capital * ALLOC_PCT
                fee = alloc * TRANSACTION_FEE
                invest = alloc - fee
                qty = invest / price
                
                if ticker not in positions:
                    positions[ticker] = {"qty": 0, "avg_price": 0}
                
                old = positions[ticker]
                new_qty = old["qty"] + qty
                # update average cost basis
                if new_qty > 0:
                    positions[ticker] = {
                        "qty": new_qty,
                        "avg_price": (old["qty"] * old["avg_price"] + invest) / new_qty
                    }
                capital -= alloc
                trade_results.append({"action": "buy", "pnl": 0})

            elif decision == "sell" and ticker in positions and positions[ticker]["qty"] > 0:
                qty = positions[ticker]["qty"]
                proceeds = qty * price
                fee = proceeds * TRANSACTION_FEE
                net = proceeds - fee
                pnl = net - (qty * positions[ticker]["avg_price"])
                
                capital += net
                trade_results.append({"action": "sell", "pnl": pnl})
                positions[ticker] = {"qty": 0, "avg_price": 0}

        # mark portfolio to market at day end
        portfolio_value = capital
        for ticker, pos in positions.items():
            if pos["qty"] > 0:
                ticker_row = day_trades[day_trades["ticker"] == ticker]
                if not ticker_row.empty:
                    portfolio_value += pos["qty"] * float(ticker_row.iloc[0]["price"])
        daily_portfolio_values.append({"date": date, "value": portfolio_value})

    # calculate performance metrics
    pv = pd.DataFrame(daily_portfolio_values)
    if len(pv) > 1:
        pv["daily_return"] = pv["value"].pct_change().dropna()
        pv = pv.dropna()
        
        avg_return = pv["daily_return"].mean()
        std_return = pv["daily_return"].std()
        sharpe = (avg_return / std_return) * math.sqrt(365) if std_return > 0 else 0
        
        total_return = (pv["value"].iloc[-1] / STARTING_CAPITAL - 1) * 100
        
        max_val = pv["value"].cummax()
        drawdown = ((pv["value"] - max_val) / max_val).min() * 100
        
        sell_trades = [t for t in trade_results if t["action"] == "sell"]
        wins = len([t for t in sell_trades if t.get("pnl", 0) > 0])
        win_rate = (wins / len(sell_trades) * 100) if sell_trades else 0
        buy_trades = [t for t in trade_results if t["action"] == "buy"]

        print(f"   ✦ Sharpe Ratio:  {sharpe:.4f}")
        print(f"   ✦ Total Return:  {total_return:+.2f}%")
        print(f"   ✦ Max Drawdown:  {drawdown:.2f}%")
        print(f"   ✦ Win Rate:      {win_rate:.1f}%")
        print(f"   ✦ Total Trades:  {len(trade_results)} ({len(buy_trades)} buys, {len(sell_trades)} sells)")
        print(f"   ✦ Final Value:   ${pv['value'].iloc[-1]:.2f}")
        
        # Send summary to Telegram
        if TELEGRAM_ENABLED:
            try:
                bot = TradingTelegramBot(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                summary = f"""📊 <b>Backtest Results</b>

Sharpe Ratio: {sharpe:.4f}
Total Return: {total_return:+.2f}%
Max Drawdown: {drawdown:.2f}%
Win Rate: {win_rate:.1f}%
Total Trades: {len(trade_results)}
Final Value: ${pv['value'].iloc[-1]:.2f}"""
                bot.send_message(summary)
            except Exception as e:
                print(f"   ⚠️ Could not send Telegram summary: {e}")
        
        return {
            "sharpe": sharpe, 
            "total_return": total_return, 
            "max_drawdown": drawdown,
            "win_rate": win_rate, 
            "final_value": pv["value"].iloc[-1]
        }
    return None


# 🚀 MODULE 6: PROCESS ROW WITH TELEGRAM
# ============================================================
def process_row_with_telegram(idx, row, total, telegram_bot=None):
    """process a single row with optional telegram notification"""
    
    prompt = build_prompt(row)
    analysis = call_llm(prompt)
    indicators = {"rsi": row.get("rsi", 50), "bb_position": row.get("bb_position", 0.5)}
    decision, reason, confidence = trading_decision(analysis, indicators)
    
    result = {
        "idx": idx,
        "date": row["date"],
        "ticker": row["ticker"],
        "price": float(row["close"]),
        "decision": decision,
        "reason": reason,
        "sentiment": f"{float(analysis.get('sentiment_score', 0)):.3f}",
        "rsi": f"{float(row.get('rsi', 0)):.2f}",
        "confidence": f"{confidence:.3f}",
        "analysis": analysis,
        "indicators": indicators,
    }
    
    # Отправка в Telegram (если включено и есть сигнал)
    if telegram_bot and TELEGRAM_ENABLED:
        should_send = (
            (SEND_ONLY_SIGNALS and decision in ['buy', 'sell']) or 
            not SEND_ONLY_SIGNALS
        )
        
        if should_send:
            try:
                # Генерируем объяснение
                explanation = generate_market_explanation(row, analysis, decision, reason)
                
                # Отправляем
                telegram_bot.send_message(explanation)
                print(f"   📱 Уведомление отправлено для {row['ticker']} ({decision})")
            except Exception as e:
                print(f"   ⚠️ Ошибка отправки в Telegram: {e}")
    
    return result


# 🚀 MODULE 7: MAIN PIPELINE
# ============================================================
def main():
    start_time = time.time()
    
    print("🌸" + "="*70)
    print("🌸 CRYPTO TRADING BOT · 4-COMPONENT LINEAR MODEL + TELEGRAM")
    print("🌸" + "="*70)
    
    # Инициализируем Telegram бота
    telegram_bot = None
    if TELEGRAM_ENABLED:
        try:
            telegram_bot = TradingTelegramBot(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
            print("🤖 Telegram бот инициализирован")
            # Отправляем тестовое сообщение
            telegram_bot.send_message(
                "🚀 <b>Торговый бот запущен</b>\n"
                f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                "Начинаю анализ рынка. Буду присылать сигналы по мере обнаружения."
            )
        except Exception as e:
            print(f"⚠️ Ошибка инициализации Telegram: {e}")
            telegram_bot = None

    # Check if cache exists
    if os.path.exists("llm_cache.json"):
        print("\n💾 found llm_cache.json - using cached results (no API calls)")
        with open("llm_cache.json") as f:
            cache = json.load(f)

        results = []
        for item in cache:
            analysis = item["analysis"]
            indicators = item["indicators"]
            indicators["rsi"] = float(indicators.get("rsi", 50))
            indicators["bb_position"] = float(indicators.get("bb_position", 0.5))
            decision, reason, confidence = trading_decision(analysis, indicators)
            
            result = {
                "date": item["date"],
                "ticker": item["ticker"],
                "price": item["price"],
                "decision": decision,
                "reason": reason,
                "sentiment": f"{float(analysis.get('sentiment_score', 0)):.3f}",
                "rsi": f"{indicators['rsi']:.2f}",
                "confidence": f"{confidence:.3f}",
            }
            results.append(result)
            
            # Для кэшированных данных тоже можем отправлять уведомления
            # Но для этого нужны полные данные строки, поэтому пропускаем
    else:
        # First run: call LLM API for every row
        print("\n🆕 no cache found - calling LLM API (this will take ~30min)...")
        features = load_data()
        total = len(features)
        print(f"\n🚀 processing {total} rows (20 concurrent workers)...")
        print("=" * 60)

        results = [None] * total
        completed = 0

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {}
            for idx, row in features.iterrows():
                future = executor.submit(process_row_with_telegram, idx, row, total, telegram_bot)
                futures[future] = idx

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results[result["idx"]] = result
                    completed += 1
                    if completed % 20 == 0 or completed == total:
                        print(f"   ⏳ progress: {completed}/{total} ({completed/total*100:.0f}%)")
                except Exception as e:
                    idx = futures[future]
                    row = features.iloc[idx]
                    # fallback on error
                    results[idx] = {
                        "idx": idx, "date": row["date"], "ticker": row["ticker"],
                        "price": float(row["close"]), "decision": "hold",
                        "reason": f"Error: {e}", "sentiment": "0.000",
                        "rsi": f"{float(row.get('rsi', 0)):.2f}", "confidence": "0.000",
                    }
                    completed += 1

        # Save cache for future runs
        cache_data = []
        for r in results:
            if r:
                cache_data.append({
                    "idx": r.get("idx", 0), "date": r["date"], "ticker": r["ticker"],
                    "price": r["price"], "analysis": r.get("analysis", {}),
                    "indicators": r.get("indicators", {}),
                })
                # remove internal fields before exporting
                r.pop("idx", None)
                r.pop("analysis", None)
                r.pop("indicators", None)

        with open("llm_cache.json", "w") as f:
            json.dump(cache_data, f)
        print(f"\n💾 saved cache to llm_cache.json (next runs will be instant)")

    # Export results
    trades_df = pd.DataFrame(results)
    trades_df.to_csv(OUTPUT_CSV, index=False)
    
    total = len(trades_df)
    buy_c = len(trades_df[trades_df["decision"] == "buy"])
    sell_c = len(trades_df[trades_df["decision"] == "sell"])
    hold_c = len(trades_df[trades_df["decision"] == "hold"])
    
    print(f"\n📊 exported {total} trades to {OUTPUT_CSV}")
    print(f"   💗 buys:  {buy_c} ({buy_c/total*100:.1f}%)")
    print(f"   💔 sells: {sell_c} ({sell_c/total*100:.1f}%)")
    print(f"   🤍 holds: {hold_c} ({hold_c/total*100:.1f}%)")

    # Run backtest
    metrics = run_backtest(trades_df)
    
    elapsed = time.time() - start_time
    print(f"\n✨ completed in {elapsed:.1f}s")
    
    # Final Telegram message
    if telegram_bot and metrics:
        telegram_bot.send_message(
            f"✅ <b>Analysis Complete</b>\n"
            f"⏱️ Time: {elapsed:.1f}s\n"
            f"📊 Signals: {buy_c} buys, {sell_c} sells\n"
            f"💰 Sharpe: {metrics['sharpe']:.4f}"
        )
    
    return metrics


if __name__ == "__main__":
    main()
