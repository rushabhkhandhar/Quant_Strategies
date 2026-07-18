import os
import sys
import pandas as pd
from datetime import date, datetime
from typing import List
from urllib.parse import quote

from data.nse_fetcher import load_nifty500_symbols, fetch_daily_candles
from core.models import CandleSet, Signal
from strategies.base import BaseStrategy
from strategies.vcp_breakout import VCPBreakoutStrategy
from strategies.holy_grail import HolyGrailStrategy
from strategies.rubber_band import RubberBandStrategy
from strategies.inside_day_squeeze import InsideDaySqueezeStrategy

# Liquidity threshold: 20-Day Average (Volume * Close) >= 70 Crore
LIQUIDITY_THRESHOLD = 70 * 1_00_00_000

def is_liquid(candles: CandleSet) -> bool:
    """Check if the stock is sufficiently liquid to trade."""
    df = candles.daily
    if len(df) < 20:
        return False
    recent = df.iloc[-20:]
    avg_traded_value = (recent["Volume"] * recent["Close"]).mean()
    return float(avg_traded_value) >= LIQUIDITY_THRESHOLD

def setup_output_dirs(strategies: List[BaseStrategy]) -> str:
    """Create output directories for each strategy."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "Combined"), exist_ok=True)
    
    for strategy in strategies:
        strategy_dir = os.path.join(outputs_dir, strategy.name)
        os.makedirs(strategy_dir, exist_ok=True)
        
    return outputs_dir

def run_screener(as_of_date: date):
    print(f"=== Starting Multi-Strategy Orchestrator for {as_of_date} ===")
    
    # 1. Initialize all strategies
    strategies: List[BaseStrategy] = [
        VCPBreakoutStrategy(),
        HolyGrailStrategy(),
        RubberBandStrategy(),
        InsideDaySqueezeStrategy(),
    ]
    
    outputs_dir = setup_output_dirs(strategies)
    
    # Track signals per strategy
    all_signals = {strategy.name: [] for strategy in strategies}
    
    # 2. Get universe of symbols
    print("Loading universe (NIFTY 500)...")
    try:
        symbols = load_nifty500_symbols()
    except Exception as e:
        print(f"Failed to load Nifty 500 from web: {e}")
        return

    print(f"Total symbols to process: {len(symbols)}")
    
    # 3. Process each symbol exactly ONCE
    for idx, symbol in enumerate(symbols):
        if idx % 50 == 0:
            print(f"Processing {idx}/{len(symbols)}...")
            
        candles = fetch_daily_candles(symbol, as_of_date, lookback_days=300)
        if not candles:
            continue
            
        if not is_liquid(candles):
            continue
            
        # 4. Pass the exact same dataset to EVERY strategy
        for strategy in strategies:
            try:
                signal = strategy.analyze(candles)
                if signal:
                    all_signals[strategy.name].append(signal)
            except Exception as e:
                print(f"Error running {strategy.name} on {symbol}: {e}")
                
    # 5. Export results
    date_str = as_of_date.strftime("%d_%m_%Y")
    all_combined_rows = []
    
    for strategy in strategies:
        signals = all_signals[strategy.name]
        
        if not signals:
            print(f"\n[{strategy.name}] No setups found.")
            continue
            
        # Convert List[Signal] to DataFrame
        rows = []
        for s in signals:
            row = {
                "Strategy": strategy.name,
                "Symbol": s.symbol,
                "Date": s.date,
                "Direction": s.direction,
                "Entry_Trigger": round(s.entry_price, 2),
                "StopLoss": round(s.stop_loss, 2),
            }
            # Unpack targets
            for t_name, t_val in s.targets.items():
                row[t_name] = round(t_val, 2)
            # Unpack metadata
            for m_name, m_val in s.metadata.items():
                row[m_name] = m_val
                
            row["TradingView_Link"] = f"https://www.tradingview.com/chart/?symbol=NSE%3A{quote(str(s.symbol).upper())}"
            rows.append(row)
            all_combined_rows.append(row)
            
        df_out = pd.DataFrame(rows)
        out_file = os.path.join(outputs_dir, strategy.name, f"{strategy.name}_Watchlist_{date_str}.csv")
        df_out.to_csv(out_file, index=False)
        print(f"\n[{strategy.name}] Found {len(signals)} setups! Saved to: {out_file}")

    # 6. Export Master Combined List
    if all_combined_rows:
        df_combined = pd.DataFrame(all_combined_rows)
        combined_file = os.path.join(outputs_dir, "Combined", f"Master_Watchlist_{date_str}.csv")
        df_combined.to_csv(combined_file, index=False)
        print(f"\n[Combined] Master Watchlist generated with {len(all_combined_rows)} total setups! Saved to: {combined_file}")
    else:
        print("\n[Combined] No setups found across any strategy today.")

def resolve_anchor_date() -> date:
    if len(sys.argv) > 1:
        user_value = sys.argv[1].strip()
    else:
        user_value = input("Enter anchor date (dd/mm/yyyy): ").strip()
        
    if not user_value:
        raise ValueError("Anchor date is required.")
    return datetime.strptime(user_value, "%d/%m/%Y").date()

if __name__ == "__main__":
    try:
        target_date = resolve_anchor_date()
        run_screener(target_date)
    except ValueError as e:
        print(f"Error: {e}")
