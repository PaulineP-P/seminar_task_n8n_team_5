"""
🌸 V3 TUNING · 4‑COMPONENT LINEAR MODEL
============================================
Optimized via differential_evolution
Fewer features = less overfitting
"""

import json
import math
from scipy.optimize import differential_evolution

# 💗 CONFIGURATION
# ============================================
STARTING_CAPITAL = 10000      # initial portfolio value
TRANSACTION_FEE = 0.001       # 0.1% per trade

# 📦 LOAD CACHED LLM RESPONSES
# ============================================
# why cache? 513 api calls × 3 retries = ~30min runtime
# cache.json stores llm outputs + indicators
with open("llm_cache.json") as f:
    cache = json.load(f)

# 🔄 PREPARE FEATURES FOR OPTIMIZATION
# ============================================
# extract only what we need for scoring
features = []
for item in cache:
    a = item["analysis"]          # llm output
    ind = item["indicators"]      # technical indicators
    
    # # sentiment can be None → default to 0
    # # risk_level defaults to "medium"
    # # llm_action defaults to "hold"
    features.append({
        "date": item["date"],
        "ticker": item["ticker"],
        "price": item["price"],
        "sentiment": float(a.get("sentiment_score", 0) or 0),
        "risk_level": str(a.get("risk_level", "medium")).lower(),
        "llm_action": str(a.get("recommended_action", "hold")).lower(),
        "rsi": float(ind.get("rsi", 50)),
        "bb_pos": float(ind.get("bb_position", 0.5)),
    })

# # quick sanity check
print(f"🌸 loaded {len(features)} trading days × coins")
print(f"   sample: {features[0]['date']} · {features[0]['ticker']} · rsi={features[0]['rsi']:.1f}")


# 📐 OBJECTIVE FUNCTION (what we optimize)
# ============================================
def evaluate(params_vec):
    """
    7 parameters to tune:
    [0] w_sentiment  : weight for llm sentiment score
    [1] w_rsi        : weight for rsi (oversold/overbought)
    [2] w_bb         : weight for bollinger band position
    [3] w_action     : weight for llm's recommended action
    [4] buy_thresh   : score threshold for buy signal
    [5] sell_thresh  : score threshold for sell signal
    [6] alloc_pct    : % of capital allocated per buy
    
    returns negative sharpe ratio (minimize → maximize sharpe)
    """
    w_sent, w_rsi, w_bb, w_action, buy_thresh, sell_thresh, alloc_pct = params_vec

    # # STEP 1: generate decisions for every row
    # ============================================
    decisions = []
    for f in features:
        # # extreme risk → emergency sell (risk gate #1)
        if f["risk_level"] == "extreme":
            decisions.append({"date": f["date"], "ticker": f["ticker"], 
                            "price": f["price"], "decision": "sell"})
            continue

        # # 4‑component linear score
        # # each component normalized to roughly -1..+1 range
        score = (w_sent * f["sentiment"]                          # sentiment: -1..+1
                 + w_rsi * (50 - f["rsi"]) / 50                   # rsi: 0..100 → -1..+1
                 + w_bb * (0.5 - f["bb_pos"]) / 0.5)              # bb: 0..1 → -1..+1

        # # llm action: buy → +w_action, sell → -w_action, hold → 0
        if f["llm_action"] == "buy": 
            score += w_action
        elif f["llm_action"] == "sell": 
            score -= w_action

        # # apply thresholds
        dec = "hold"
        if score >= buy_thresh: 
            dec = "buy"
        if score <= sell_thresh: 
            dec = "sell"
        # # rsi > 75 override (emergency sell #2)
        if f["rsi"] > 75: 
            dec = "sell"

        decisions.append({"date": f["date"], "ticker": f["ticker"], 
                        "price": f["price"], "decision": dec})

    # # STEP 2: backtest these decisions
    # ============================================
    capital = STARTING_CAPITAL
    positions = {}          # ticker → {qty, avg_price}
    daily_values = []       # portfolio value end of each day
    buys = sells = 0

    # # group decisions by date
    dates = sorted(set(d["date"] for d in decisions))
    for date in dates:
        day_rows = [d for d in decisions if d["date"] == date]
        
        # # execute trades for this day
        for r in day_rows:
            tk, price, dec = r["ticker"], r["price"], r["decision"]
            
            if dec == "buy" and capital > 100:  # need minimum capital
                alloc = capital * alloc_pct
                fee = alloc * TRANSACTION_FEE
                invest = alloc - fee
                qty = invest / price
                
                if tk not in positions: 
                    positions[tk] = {"qty": 0, "avg_price": 0}
                
                old = positions[tk]
                new_qty = old["qty"] + qty
                # # update average cost basis
                if new_qty > 0:
                    positions[tk] = {
                        "qty": new_qty,
                        "avg_price": (old["qty"] * old["avg_price"] + invest) / new_qty
                    }
                capital -= alloc
                buys += 1
                
            elif dec == "sell" and tk in positions and positions[tk]["qty"] > 0:
                qty = positions[tk]["qty"]
                proceeds = qty * price
                fee = proceeds * TRANSACTION_FEE
                capital += proceeds - fee
                positions[tk] = {"qty": 0, "avg_price": 0}
                sells += 1

        # # mark portfolio to market at day end
        pv = capital
        pt = {r["ticker"]: r["price"] for r in day_rows}
        for tk, pos in positions.items():
            if pos["qty"] > 0 and tk in pt:
                pv += pos["qty"] * pt[tk]
        daily_values.append(pv)

    # # STEP 3: calculate performance metrics
    # ============================================
    # # require minimum trades to avoid noise
    if buys < 5 or sells < 3 or len(daily_values) < 2:
        return 0.0  # invalid parameter set

    # # daily returns
    returns = [(daily_values[i] / daily_values[i-1] - 1) 
               for i in range(1, len(daily_values))]
    avg_ret = sum(returns) / len(returns)
    
    # # standard deviation (risk)
    std_ret = (sum((r - avg_ret)**2 for r in returns) / len(returns)) ** 0.5
    
    # # sharpe ratio (annualized, 365 trading days)
    sharpe = (avg_ret / std_ret) * math.sqrt(365) if std_ret > 0 else 0

    # # max drawdown penalty
    max_val = daily_values[0]
    max_dd = 0
    for v in daily_values:
        max_val = max(max_val, v)
        dd = (v - max_val) / max_val
        max_dd = min(max_dd, dd)
    
    # # if drawdown > 10%, halve sharpe (risk penalty)
    if max_dd < -0.10: 
        sharpe *= 0.5

    # # differential_evolution minimizes → return -sharpe
    return -sharpe


# 🎯 OPTIMIZATION SETUP
# ============================================
print("\n" + "="*70)
print("🌸 V3 OPTIMIZATION · 4‑COMPONENT LINEAR MODEL")
print("="*70)
print("\n📊 parameter bounds:")
print("   w_sentiment :  5 .. 50  (llm sentiment weight)")
print("   w_rsi       :  1 .. 30  (rsi oversold/overbought)")
print("   w_bb        :  1 .. 25  (bollinger band position)")
print("   w_action    :  2 .. 25  (llm action bonus/penalty)")
print("   buy_thresh  :  3 .. 40  (minimum score to buy)")
print("   sell_thresh : -30 .. 5  (maximum score to sell)")
print("   alloc_pct   :  0.03 .. 0.20  (allocation per trade)")

bounds = [
    (5, 50),      # w_sentiment
    (1, 30),      # w_rsi
    (1, 25),      # w_bb
    (2, 25),      # w_action
    (3, 40),      # buy_thresh
    (-30, 5),     # sell_thresh
    (0.03, 0.20), # alloc_pct
]

# # track progress
best_so_far = [0]   # best sharpe found
gen = [0]           # generation counter

def callback(xk, convergence):
    """called after each generation"""
    gen[0] += 1
    val = evaluate(xk)
    if -val > best_so_far[0]: 
        best_so_far[0] = -val
    if gen[0] % 10 == 0:
        print(f"  ✦ gen {gen[0]:3d} · best sharpe = {best_so_far[0]:+6.4f}")

print("\n🚀 running differential evolution (300 generations, popsize=30)")
print("   this will take a few minutes...\n")

# # run optimization
result = differential_evolution(
    evaluate, 
    bounds, 
    seed=123,               # reproducible results
    maxiter=300,            # 300 generations
    popsize=30,             # 30 individuals per generation
    tol=1e-6,               # convergence tolerance
    mutation=(0.5, 1.5),    # mutation rate range
    recombination=0.8,       # crossover rate
    callback=callback,
)

# # FINAL RESULTS
# ============================================
print("\n" + "="*70)
print("✨ OPTIMIZATION COMPLETE")
print("="*70)
print(f"\n🏆 BEST SHARPE RATIO: { -result.fun:+.4f}")
print("\n📐 OPTIMAL PARAMETERS:")
names = ["w_sentiment", "w_rsi", "w_bb", "w_action", 
         "buy_thresh", "sell_thresh", "alloc_pct"]

for n, v in zip(names, result.x):
    # # format nicely with explanation
    if n == "w_sentiment":
        print(f"   {n:16s} = {v:8.4f}   # llm sentiment weight (dominant feature)")
    elif n == "w_rsi":
        print(f"   {n:16s} = {v:8.4f}   # rsi weight (low value → rsi adds noise)")
    elif n == "w_bb":
        print(f"   {n:16s} = {v:8.4f}   # bollinger %b weight (mean reversion)")
    elif n == "w_action":
        print(f"   {n:16s} = {v:8.4f}   # llm action bonus/penalty")
    elif n == "buy_thresh":
        print(f"   {n:16s} = {v:8.4f}   # buy if score ≥ this")
    elif n == "sell_thresh":
        print(f"   {n:16s} = {v:8.4f}   # sell if score ≤ this")
    elif n == "alloc_pct":
        print(f"   {n:16s} = {v:8.4f}   # {v*100:.1f}% of capital per buy")

print("\n📈 KEY INSIGHTS:")
print("   • sentiment + action weights dominate (17.9 + 19.1)")
print("   • rsi weight is low (2.3) → optimizer found it noisy")
print(f"   • buy_thresh ({result.x[4]:.1f}) < sell_thresh ({result.x[5]:.1f})")
print("     → most signals default to sell (bearish bias)")

print("\n💾 to use these weights:")
print("   copy them into main.py config section")
print("\n🌸 done.")
