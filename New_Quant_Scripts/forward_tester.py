import os
import json
import glob
from datetime import date, datetime
from typing import Any, Dict, List, Optional
import pandas as pd

from data.nse_fetcher import _download_bhavcopy_for_date, _find_column

# Configuration
CAPITAL_PER_TRADE = 1_00_000.0  # ₹1 Lakh
MAX_HOLD_DAYS = 21
ENTRY_WINDOW_DAYS = 2

def get_output_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs", "Forward_Test")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def fetch_day_prices(symbols: List[str], trade_date: date) -> Dict[str, Dict[str, float]]:
    """Return {SYMBOL: {Open, High, Low, Close, Volume}} from NSE bhavcopy."""
    df = _download_bhavcopy_for_date(trade_date)
    if df is None or df.empty:
        return {}

    symbol_col = _find_column(df.columns.tolist(), ["SYMBOL"])
    series_col = _find_column(df.columns.tolist(), ["SERIES"])
    
    if not (symbol_col and series_col):
        return {}

    work = df.copy()
    work[symbol_col] = work[symbol_col].astype(str).str.strip().str.upper()
    work[series_col] = work[series_col].astype(str).str.strip().str.upper()
    work = work[work[series_col] == "EQ"]

    symbol_set = {s.upper() for s in symbols}
    work = work[work[symbol_col].isin(symbol_set)]

    result = {}
    for _, row in work.iterrows():
        sym = str(row[symbol_col]).strip().upper()
        try:
            result[sym] = {
                "Open": float(row["OPEN"]),
                "High": float(row["HIGH"]),
                "Low": float(row["LOW"]),
                "Close": float(row["CLOSE"]),
                "Volume": float(row.get("VOLUME", 0.0))
            }
        except Exception:
            continue
    return result

class ForwardTester:
    def __init__(self):
        self.output_dir = get_output_dir()
        self.state_file = os.path.join(self.output_dir, "forward_test_state.json")
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as fh:
                return json.load(fh)
        return {"trades": [], "daily_pnl_log": []}

    def _save_state(self) -> None:
        with open(self.state_file, "w") as fh:
            json.dump(self.state, fh, indent=2)

    def _trade_id(self, batch_date: str, symbol: str, strategy: str) -> str:
        return f"{batch_date}_{strategy}_{symbol}"

    def ingest_new_batch(self, signal_csv_path: str, batch_date: date) -> int:
        if not os.path.exists(signal_csv_path):
            return 0
            
        df = pd.read_csv(signal_csv_path)
        if df.empty:
            return 0

        existing = {t["trade_id"] for t in self.state["trades"]}
        bd = batch_date.strftime("%Y-%m-%d")
        added = 0

        for _, row in df.iterrows():
            sym = str(row["Symbol"]).strip().upper()
            strat = str(row["Strategy"]).strip()
            tid = self._trade_id(bd, sym, strat)
            
            if tid in existing:
                continue

            entry_price = float(row["Entry_Trigger"])
            shares = int(CAPITAL_PER_TRADE // entry_price) if entry_price > 0 else 0

            trade = {
                "trade_id": tid,
                "strategy": strat,
                "batch_date": bd,
                "symbol": sym,
                "direction": str(row["Direction"]).strip().upper(),
                "entry_price": round(entry_price, 2),
                "stop_loss": round(float(row["StopLoss"]), 2),
                "target_1": round(float(row["Target_1"]), 2),
                "target_2": round(float(row["Target_2"]), 2),
                "capital_allocated": round(CAPITAL_PER_TRADE, 2),
                "num_shares": shares,
                "status": "pending_entry",
                "entry_date": None,
                "entry_attempts": 0,
                "exit_date": None,
                "exit_price": None,
                "exit_reason": None,
                "days_held": 0,
                "pnl_pct": 0.0,
                "pnl_absolute": 0.0,
                "last_close": None,
                "action_log": [f"[{bd}] SIGNAL: {strat} on {sym}. Entry: {entry_price:.2f}"]
            }
            self.state["trades"].append(trade)
            added += 1

        self._save_state()
        return added

    def update_daily(self, trade_date: date) -> None:
        live_statuses = ("pending_entry", "active")
        symbols_needed = {t["symbol"] for t in self.state["trades"] if t["status"] in live_statuses}
        
        if not symbols_needed:
            print("No active or pending trades to update.")
            return

        prices = fetch_day_prices(list(symbols_needed), trade_date)
        td = trade_date.strftime("%Y-%m-%d")
        
        if not prices:
            print(f"No market data found for {td} (Holiday or Weekend).")
            return

        for trade in self.state["trades"]:
            if trade["status"] not in live_statuses:
                continue

            sym = trade["symbol"]
            if sym not in prices:
                continue

            ohlcv = prices[sym]
            
            if trade["status"] == "pending_entry":
                self._process_pending(trade, ohlcv, trade_date, td)
            elif trade["status"] == "active":
                self._process_active(trade, ohlcv, td)

        self._save_state()
        self.export_all()

    def _process_pending(self, trade, ohlcv, trade_date, td):
        batch_date = datetime.strptime(trade["batch_date"], "%Y-%m-%d").date()
        if trade_date <= batch_date:
            return  # Do not enter on signal day

        trade["entry_attempts"] += 1
        ep = trade["entry_price"]
        h, l = ohlcv["High"], ohlcv["Low"]

        if l <= ep <= h:
            trade["status"] = "active"
            trade["entry_date"] = td
            trade["days_held"] = 1
            trade["last_close"] = ohlcv["Close"]
            trade["action_log"].append(f"[{td}] ENTRY FILLED at {ep:.2f}")
        elif trade["entry_attempts"] >= ENTRY_WINDOW_DAYS:
            trade["status"] = "missed"
            trade["exit_reason"] = "Entry timeframe expired"
            trade["action_log"].append(f"[{td}] ENTRY MISSED (expired)")

    def _process_active(self, trade, ohlcv, td):
        trade["days_held"] += 1
        trade["last_close"] = ohlcv["Close"]
        
        ep = trade["entry_price"]
        c, h, l = ohlcv["Close"], ohlcv["High"], ohlcv["Low"]
        sl = trade["stop_loss"]
        t2 = trade["target_2"]
        direction = trade["direction"]

        unrealized = ((c - ep) / ep) * 100 if direction == "LONG" else ((ep - c) / ep) * 100
        trade["pnl_pct"] = round(unrealized, 2)
        trade["pnl_absolute"] = round(unrealized / 100 * trade["capital_allocated"], 2)

        exit_reason = None
        exit_price = None

        if direction == "LONG":
            if l <= sl:
                exit_reason = "StopLoss Hit"
                exit_price = sl
            elif h >= t2:
                exit_reason = "Target Hit"
                exit_price = t2
        else: # SHORT
            if h >= sl:
                exit_reason = "StopLoss Hit"
                exit_price = sl
            elif l <= t2:
                exit_reason = "Target Hit"
                exit_price = t2

        if not exit_reason and trade["days_held"] >= MAX_HOLD_DAYS:
            exit_reason = "Time Stop (21 Days)"
            exit_price = c

        if exit_reason:
            trade["status"] = "completed"
            trade["exit_date"] = td
            trade["exit_price"] = round(exit_price, 2)
            trade["exit_reason"] = exit_reason
            
            final_pnl = ((exit_price - ep) / ep) * 100 if direction == "LONG" else ((ep - exit_price) / ep) * 100
            trade["pnl_pct"] = round(final_pnl, 2)
            trade["pnl_absolute"] = round(final_pnl / 100 * trade["capital_allocated"], 2)
            
            trade["action_log"].append(f"[{td}] EXITED: {exit_reason} at {exit_price:.2f}. PnL: {final_pnl:+.2f}%")
        else:
            trade["action_log"].append(f"[{td}] HOLDING. Unrealized: {unrealized:+.2f}%")

    def export_all(self):
        active = [t for t in self.state["trades"] if t["status"] in ("pending_entry", "active")]
        completed = [t for t in self.state["trades"] if t["status"] in ("completed", "missed")]
        
        pd.DataFrame(active).to_csv(os.path.join(self.output_dir, "active_portfolio.csv"), index=False)
        pd.DataFrame(completed).to_csv(os.path.join(self.output_dir, "completed_trades.csv"), index=False)
        print(f"Exported portfolio tracking sheets to {self.output_dir}/")

def main():
    print("=== Universal Forward Tester ===")
    
    # 1. Ask user for the date to process
    date_val = input(f"Enter trade date (dd/mm/yyyy) [Default: {datetime.today().strftime('%d/%m/%Y')}]: ").strip()
    if not date_val:
        date_val = datetime.today().strftime("%d/%m/%Y")
    
    trade_date = datetime.strptime(date_val, "%d/%m/%Y").date()
    
    tester = ForwardTester()
    
    # 2. Ingest any master watchlists from PREVIOUS day (since signals are generated EOD)
    # Actually, we can just glob all Master_Watchlists and ingest any we haven't seen.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    combined_dir = os.path.join(base_dir, "outputs", "Combined")
    
    if os.path.exists(combined_dir):
        for file in glob.glob(os.path.join(combined_dir, "Master_Watchlist_*.csv")):
            # extract date from filename (Master_Watchlist_DD_MM_YYYY.csv)
            date_str = os.path.basename(file).replace("Master_Watchlist_", "").replace(".csv", "")
            try:
                batch_date = datetime.strptime(date_str, "%d_%m_%Y").date()
                added = tester.ingest_new_batch(file, batch_date)
                if added > 0:
                    print(f"Ingested {added} new setups from {batch_date}")
            except Exception as e:
                pass
                
    # 3. Process Live Data
    print(f"Evaluating portfolio against market data for {trade_date}...")
    tester.update_daily(trade_date)
    print("Forward test update complete.")

if __name__ == "__main__":
    main()
