import os
from datetime import date, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

from core.models import CandleSet, Signal
from strategies.base import BaseStrategy
from data.nse_fetcher import fetch_daily_candles, fetch_bulk_history

class BacktestEngine:
    def __init__(self, strategy: BaseStrategy, symbols: List[str], start_date: date, end_date: date):
        self.strategy = strategy
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        
        self.initial_capital = 1_00_000.0  # Targeted for 1 Lakh retail investor
        self.max_allocation_per_trade = 0.15 # Max 15% of portfolio per trade
        self.risk_per_trade_pct = 0.02       # Max 2% portfolio risk per trade
        
        self.trades = []
        self.equity_curve = []
        
    def pre_fetch_data(self):
        """Fetch data for all symbols to cover the backtest period + lookback."""
        print(f"Pre-fetching historical data for {len(self.symbols)} symbols...")
        if "NIFTYBEES" not in self.symbols:
            self.symbols.append("NIFTYBEES")
            
        lookback = (self.end_date - self.start_date).days + 300
        
        self.history = fetch_bulk_history(self.symbols, self.end_date, lookback)
        
        # Calculate NIFTYBEES 50 SMA for Market Regime Filter
        if "NIFTYBEES" in self.history:
            nifty_df = self.history["NIFTYBEES"].copy()
            nifty_df["SMA_50"] = nifty_df["Close"].rolling(50).mean()
            self.history["NIFTYBEES"] = nifty_df
            
        print("Data loaded and vectorized successfully!")
                
    def _run_signal_generation(self):
        print(f"Generating signals for {self.strategy.name}...")
        self.signals = {} # date -> List[Signal]
        
        nifty_df = self.history.get("NIFTYBEES", None)
        
        for symbol, df in self.history.items():
            if symbol == "NIFTYBEES":
                continue # NIFTYBEES is only used for regime filtering
                
            # Precalculate indicators for the entire history
            df = self.strategy.prepare_data(df)
            
            # Iterate using integer indexing for O(1) memory views instead of O(N) boolean masks
            for i in range(49, len(df)): # Start at 49 to ensure at least 50 days of history
                t_date_ts = df.index[i]
                if t_date_ts.date() < self.start_date:
                    continue
                if t_date_ts.date() > self.end_date:
                    break
                    
                # Market Regime Check
                market_is_bullish = True
                if nifty_df is not None and t_date_ts in nifty_df.index:
                    nifty_day = nifty_df.loc[t_date_ts]
                    if isinstance(nifty_day, pd.DataFrame):
                        nifty_day = nifty_day.iloc[0]
                    if float(nifty_day["SMA_50"]) > 0 and float(nifty_day["Close"]) < float(nifty_day["SMA_50"]):
                        market_is_bullish = False
                    
                # Instant O(1) slice
                window_df = df.iloc[:i+1]
                    
                temp_candles = CandleSet(symbol=symbol, daily=window_df)
                try:
                    sig = self.strategy.analyze(temp_candles)
                    if sig:
                        if sig.direction == "LONG" and not market_is_bullish:
                            continue # Block LONG entries in a Bearish Market Regime
                            
                        dt_str = t_date_ts.strftime("%Y-%m-%d")
                        if dt_str not in self.signals:
                            self.signals[dt_str] = []
                        self.signals[dt_str].append(sig)
                except Exception as e:
                    pass
                    
    def run(self):
        self.pre_fetch_data()
        self._run_signal_generation()
        
        print("Simulating Portfolio Trades...")
        current_cash = self.initial_capital
        open_positions = [] # Dicts of trade info
        
        # Build a sorted list of all unique dates in the simulation
        all_dates = set()
        for df in self.history.values():
            all_dates.update(df[(df.index.date >= self.start_date) & (df.index.date <= self.end_date)].index.date)
        timeline = sorted(list(all_dates))
        
        for t_date in timeline:
            dt_str = t_date.strftime("%Y-%m-%d")
            
            # 1. Update Open Positions (Check for Exits)
            surviving_positions = []
            for pos in open_positions:
                symbol = pos['symbol']
                if symbol not in self.history or pd.Timestamp(t_date) not in self.history[symbol].index:
                    surviving_positions.append(pos)
                    continue
                    
                day_data = self.history[symbol].loc[pd.Timestamp(t_date)]
                # Some days have duplicate entries, take first
                if isinstance(day_data, pd.DataFrame):
                    day_data = day_data.iloc[0]
                    
                open_price = float(day_data['Open'])
                high = float(day_data['High'])
                low = float(day_data['Low'])
                close = float(day_data['Close'])
                
                # Dynamic ATR Chandelier Trailing Stop
                if high > pos.get('highest_high', 0.0):
                    pos['highest_high'] = high
                    atr = pos.get('entry_atr', 0.0)
                    if atr > 0:
                        new_sl = pos['highest_high'] - (3.0 * atr)
                        if new_sl > pos['stop_loss']:
                            pos['stop_loss'] = new_sl
                            
                exit_triggered = False
                
                # Check Stop Loss First (Pessimistic execution)
                if low <= pos['stop_loss']:
                    # Exit all remaining quantity at Stop Loss
                    exit_price = pos['stop_loss'] if pos['stop_loss'] <= high else open_price
                    revenue = pos['qty'] * pos['stop_loss']
                    exit_fee = revenue * 0.0015
                    current_cash += (revenue - exit_fee)
                    
                    self.trades.append({
                        "Symbol": symbol,
                        "Entry_Date": pos['entry_date'],
                        "Exit_Date": dt_str,
                        "Type": "Stop Loss" if pos['stop_loss'] < pos['entry_price'] else "Trailing SL",
                        "Profit": (revenue - exit_fee) - (pos['qty'] * (pos['entry_price'] + pos['entry_fee_per_share']))
                    })
                    exit_triggered = True
                    
                # Check Trailing MA Exit
                elif pos.get('trailing_ma') and pos['trailing_ma'] in day_data and close < float(day_data[pos['trailing_ma']]):
                    revenue = pos['qty'] * close
                    exit_fee = revenue * 0.0015
                    current_cash += (revenue - exit_fee)
                    
                    self.trades.append({
                        "Symbol": symbol,
                        "Entry_Date": pos['entry_date'],
                        "Exit_Date": dt_str,
                        "Type": f"Trailing {pos['trailing_ma']}",
                        "Profit": (revenue - exit_fee) - (pos['qty'] * (pos['entry_price'] + pos['entry_fee_per_share']))
                    })
                    exit_triggered = True
                    
                # Check Target 1 (50% Sell, Trail SL to Entry)
                elif not pos['t1_hit'] and high >= pos['target_1']:
                    sell_qty = pos['qty'] // 2
                    revenue = sell_qty * pos['target_1']
                    exit_fee = revenue * 0.0015
                    current_cash += (revenue - exit_fee)
                    pos['qty'] -= sell_qty
                    pos['t1_hit'] = True
                    pos['stop_loss'] = pos['entry_price'] # Trail SL to Breakeven
                    
                    self.trades.append({
                        "Symbol": symbol,
                        "Entry_Date": pos['entry_date'],
                        "Exit_Date": dt_str,
                        "Type": "Target 1 (50%)",
                        "Profit": (revenue - exit_fee) - (sell_qty * (pos['entry_price'] + pos['entry_fee_per_share']))
                    })
                    
                    # If Target 2 is also hit on the same day
                    if high >= pos['target_2']:
                        revenue_2 = pos['qty'] * pos['target_2']
                        exit_fee_2 = revenue_2 * 0.0015
                        current_cash += (revenue_2 - exit_fee_2)
                        self.trades.append({
                            "Symbol": symbol,
                            "Entry_Date": pos['entry_date'],
                            "Exit_Date": dt_str,
                            "Type": "Target 2 (Final)",
                            "Profit": (revenue_2 - exit_fee_2) - (pos['qty'] * (pos['entry_price'] + pos['entry_fee_per_share']))
                        })
                        exit_triggered = True
                        
                # Check Target 2
                elif pos['t1_hit'] and high >= pos['target_2']:
                    revenue = pos['qty'] * pos['target_2']
                    exit_fee = revenue * 0.0015
                    current_cash += (revenue - exit_fee)
                    self.trades.append({
                        "Symbol": symbol,
                        "Entry_Date": pos['entry_date'],
                        "Exit_Date": dt_str,
                        "Type": "Target 2 (Final)",
                        "Profit": (revenue - exit_fee) - (pos['qty'] * (pos['entry_price'] + pos['entry_fee_per_share']))
                    })
                    exit_triggered = True
                    
                if not exit_triggered:
                    pos['current_value'] = pos['qty'] * close
                    surviving_positions.append(pos)
                    
            open_positions = surviving_positions
            
            # 2. Open New Positions based on signals from YESTERDAY
            # In a real backtest, a signal generated at close of day T is executed on open/high of day T+1
            # Here we assume entry on trigger price.
            # To keep it simple, we check if today's High >= Entry Trigger.
            # (In reality, we should track pending orders, but we'll approximate).
            
            # Re-evaluate portfolio value to allocate capital
            portfolio_value = current_cash + sum(p.get('current_value', 0) for p in open_positions)
            
            # Get signals from previous trading days that are still valid (simplification: just yesterday's)
            prev_date_str = None
            idx = timeline.index(t_date)
            if idx > 0:
                prev_date_str = timeline[idx-1].strftime("%Y-%m-%d")
                
            daily_signals = self.signals.get(prev_date_str, [])
            
            for sig in daily_signals:
                if current_cash <= 0:
                    break
                    
                symbol = sig.symbol
                if symbol not in self.history or pd.Timestamp(t_date) not in self.history[symbol].index:
                    continue
                    
                day_data = self.history[symbol].loc[pd.Timestamp(t_date)]
                if isinstance(day_data, pd.DataFrame):
                    day_data = day_data.iloc[0]
                    
                high = float(day_data['High'])
                open_price = float(day_data['Open'])
                
                # Check if entry trigger was hit
                if high >= sig.entry_price:
                    # Execute trade
                    exec_price = max(open_price, sig.entry_price) # If it gapped up, we buy at open
                    
                    # --- Advanced Volatility-Based Position Sizing ---
                    # 1. Cap by Risk (Max 2% portfolio loss if SL is hit)
                    max_risk_amount = portfolio_value * self.risk_per_trade_pct
                    risk_per_share = exec_price - sig.stop_loss
                    if risk_per_share <= 0:
                        risk_per_share = exec_price * 0.01  # Fallback: assume 1% risk if SL is invalid/inverted
                        
                    qty_risk = int(max_risk_amount // risk_per_share)
                    
                    # 2. Cap by Max Allocation (Max 10% of portfolio size per trade)
                    max_allocation_amount = portfolio_value * self.max_allocation_per_trade
                    qty_alloc = int(max_allocation_amount // exec_price)
                    
                    # 3. Cap by Available Cash
                    qty_cash = int(current_cash // exec_price)
                    
                    # Take the most conservative constraint
                    qty = min(qty_risk, qty_alloc, qty_cash)
                    
                    if qty > 0:
                        cost = qty * exec_price
                        entry_fee = cost * 0.0015
                        current_cash -= (cost + entry_fee)
                        
                        t1_val = sig.targets.get('Target_1', exec_price * 1.03)
                        t2_val = sig.targets.get('Target_2', exec_price * 1.10)
                        
                        open_positions.append({
                            "symbol": symbol,
                            "entry_date": dt_str,
                            "entry_price": exec_price,
                            "entry_fee_per_share": entry_fee / qty,
                            "qty": qty,
                            "stop_loss": sig.stop_loss,
                            "entry_atr": sig.metadata.get('ATR', 0.0),
                            "highest_high": exec_price,
                            "target_1": t1_val,
                            "target_2": t2_val,
                            "trailing_ma": sig.metadata.get("trailing_ma", None),
                            "t1_hit": False,
                            "current_value": qty * float(day_data['Close'])
                        })
            
            # Record Equity
            portfolio_value = current_cash + sum(p.get('current_value', 0) for p in open_positions)
            self.equity_curve.append({
                "Date": t_date,
                "Equity": portfolio_value,
                "Cash": current_cash
            })
            
        return self._generate_metrics()
        
    def _generate_metrics(self):
        if not self.equity_curve:
            return {}
            
        eq_df = pd.DataFrame(self.equity_curve).set_index("Date")
        eq_df["Return"] = eq_df["Equity"].pct_change()
        
        total_return = (eq_df["Equity"].iloc[-1] / self.initial_capital) - 1
        
        # Max Drawdown
        rolling_max = eq_df["Equity"].cummax()
        drawdown = (eq_df["Equity"] - rolling_max) / rolling_max
        mdd = drawdown.min()
        
        # Sharpe (Risk Free Rate approx 5%)
        daily_rf = 0.05 / 252
        excess_returns = eq_df["Return"].dropna() - daily_rf
        sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std()) if excess_returns.std() > 0 else 0
        
        # Sortino
        downside_returns = excess_returns[excess_returns < 0]
        sortino = np.sqrt(252) * (excess_returns.mean() / downside_returns.std()) if downside_returns.std() > 0 else 0
        
        # Annualized Return & Calmar Ratio
        days = (eq_df.index[-1] - eq_df.index[0]).days
        annualized_return = ((eq_df["Equity"].iloc[-1] / self.initial_capital) ** (365 / days)) - 1 if days > 0 else 0
        calmar = annualized_return / abs(mdd) if mdd < 0 else 0
        
        # Win Rate
        winning_trades = len([t for t in self.trades if t["Profit"] > 0])
        total_trades = len(self.trades)
        win_rate = (winning_trades / total_trades) if total_trades > 0 else 0
        
        return {
            "Total_Trades": total_trades,
            "Win_Rate_%": round(win_rate * 100, 2),
            "Total_Return_%": round(total_return * 100, 2),
            "Max_Drawdown_%": round(mdd * 100, 2),
            "Sharpe_Ratio": round(sharpe, 2),
            "Sortino_Ratio": round(sortino, 2),
            "Calmar_Ratio": round(calmar, 2),
            "Final_Equity": round(eq_df["Equity"].iloc[-1], 2),
            "Equity_Curve": eq_df,
            "Trade_Log": pd.DataFrame(self.trades)
        }
