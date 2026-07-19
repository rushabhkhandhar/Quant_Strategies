import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import date
from core.models import CandleSet, Signal
from strategies.base import BaseStrategy
from data.nse_fetcher import fetch_bulk_history

class EnsembleBacktestEngine:
    def __init__(self, strategies: List[BaseStrategy], symbols: List[str], start_date: date, end_date: date):
        self.strategies = strategies
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        
        self.initial_capital = 1_00_000.0  # Targeted for 1 Lakh retail investor
        self.max_allocation_per_trade = 0.15
        self.risk_per_trade_pct = 0.02
        self.max_open_positions = 6
        
        self.history: Dict[str, pd.DataFrame] = {}
        self.signals: Dict[str, Dict[str, List[Signal]]] = {s.name: {} for s in strategies}
        
        self.trades = []
        self.equity_curve = []
        
        self.hierarchy = {
            "Rubber_Band_Mean_Reversion": 1,
            "VCP_Breakout": 2,
            "Holy_Grail_Pullback": 3,
            "Inside_Day_Squeeze": 4
        }
        
    def pre_fetch_data(self):
        # Ensure NIFTYBEES is fetched for the macro regime filter
        if "NIFTYBEES" not in self.symbols:
            self.symbols.append("NIFTYBEES")
        print(f"Pre-fetching historical data for {len(self.symbols)} symbols...")
        lookback = (self.end_date - self.start_date).days + 300
        
        self.history = fetch_bulk_history(self.symbols, self.end_date, lookback)
        
        # Calculate NIFTYBEES 50 SMA for Market Regime Filter
        if "NIFTYBEES" in self.history:
            nifty_df = self.history["NIFTYBEES"].copy()
            nifty_df["SMA_50"] = nifty_df["Close"].rolling(50).mean()
            self.history["NIFTYBEES"] = nifty_df
            
        print("Data loaded and vectorized successfully!")

    def _run_signal_generation(self):
        print(f"Generating signals for {len(self.strategies)} strategies...")
        
        nifty_df = self.history.get("NIFTYBEES", None)
        
        for symbol, original_df in self.history.items():
            if symbol == "NIFTYBEES":
                continue # NIFTYBEES is only used for regime filtering
                
            for strategy in self.strategies:
                # Precalculate indicators for the entire history for this specific strategy
                df = strategy.prepare_data(original_df.copy())
                
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
                            
                    window_df = df.iloc[:i+1]
                    temp_candles = CandleSet(symbol=symbol, daily=window_df)
                    
                    try:
                        sig = strategy.analyze(temp_candles)
                        if sig:
                            if sig.direction == "LONG" and not market_is_bullish:
                                continue # Block LONG entries in a Bearish Market Regime
                                
                            dt_str = t_date_ts.strftime("%Y-%m-%d")
                            if dt_str not in self.signals[strategy.name]:
                                self.signals[strategy.name][dt_str] = []
                            sig.metadata["strategy_name"] = strategy.name
                            self.signals[strategy.name][dt_str].append(sig)
                    except Exception:
                        pass
                        
    def run(self):
        self.pre_fetch_data()
        self._run_signal_generation()
        
        print("Simulating Portfolio Trades via Master Orchestrator...")
        
        current_cash = self.initial_capital
        open_positions = []
        
        all_dates = set()
        for df in self.history.values():
            all_dates.update(df.index.date)
        timeline = sorted([d for d in all_dates if self.start_date <= d <= self.end_date])
        
        for t_date in timeline:
            dt_str = t_date.strftime("%Y-%m-%d")
            
            surviving_positions = []
            for pos in open_positions:
                symbol = pos['symbol']
                if symbol not in self.history or pd.Timestamp(t_date) not in self.history[symbol].index:
                    surviving_positions.append(pos)
                    continue
                    
                day_data = self.history[symbol].loc[pd.Timestamp(t_date)]
                if isinstance(day_data, pd.DataFrame): day_data = day_data.iloc[0]
                    
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
                
                if low <= pos['stop_loss']:
                    exit_price = pos['stop_loss'] if pos['stop_loss'] <= high else open_price
                    revenue = pos['qty'] * pos['stop_loss']
                    exit_fee = revenue * 0.0015
                    current_cash += (revenue - exit_fee)
                    self.trades.append({
                        "Strategy": pos['strategy_name'],
                        "Symbol": symbol,
                        "Entry_Date": pos['entry_date'],
                        "Exit_Date": dt_str,
                        "Type": "Stop Loss" if pos['stop_loss'] < pos['entry_price'] else "Trailing SL",
                        "Profit": (revenue - exit_fee) - (pos['qty'] * (pos['entry_price'] + pos.get('entry_fee_per_share', 0)))
                    })
                    exit_triggered = True
                    
                elif not pos['t1_hit'] and high >= pos['target_1']:
                    sell_qty = pos['qty'] // 2
                    revenue = sell_qty * pos['target_1']
                    exit_fee = revenue * 0.0015
                    current_cash += (revenue - exit_fee)
                    pos['qty'] -= sell_qty
                    pos['t1_hit'] = True
                    pos['stop_loss'] = pos['entry_price']
                    self.trades.append({
                        "Strategy": pos['strategy_name'],
                        "Symbol": symbol,
                        "Entry_Date": pos['entry_date'],
                        "Exit_Date": dt_str,
                        "Type": "Target 1 (50%)",
                        "Profit": (revenue - exit_fee) - (sell_qty * (pos['entry_price'] + pos.get('entry_fee_per_share', 0)))
                    })
                    
                if not exit_triggered and high >= pos['target_2']:
                    revenue = pos['qty'] * pos['target_2']
                    exit_fee = revenue * 0.0015
                    current_cash += (revenue - exit_fee)
                    self.trades.append({
                        "Strategy": pos['strategy_name'],
                        "Symbol": symbol,
                        "Entry_Date": pos['entry_date'],
                        "Exit_Date": dt_str,
                        "Type": "Target 2",
                        "Profit": (revenue - exit_fee) - (pos['qty'] * (pos['entry_price'] + pos.get('entry_fee_per_share', 0)))
                    })
                    exit_triggered = True
                    
                if not exit_triggered:
                    pos['current_value'] = pos['qty'] * close
                    surviving_positions.append(pos)
                    
            open_positions = surviving_positions
            portfolio_value = current_cash + sum(p.get('current_value', 0) for p in open_positions)
            
            prev_date_str = None
            idx = timeline.index(t_date)
            if idx > 0:
                prev_date_str = timeline[idx-1].strftime("%Y-%m-%d")
                
            todays_signals = []
            for strategy in self.strategies:
                todays_signals.extend(self.signals[strategy.name].get(prev_date_str, []))
                
            if todays_signals:
                todays_signals.sort(key=lambda s: (
                    self.hierarchy.get(s.metadata.get("strategy_name", ""), 99),
                    -s.metadata.get("rank_score", 0.0)
                ))
                
                executed_symbols = set([p['symbol'] for p in open_positions])
                
                for sig in todays_signals:
                    if len(open_positions) >= self.max_open_positions or current_cash <= 0:
                        break
                        
                    symbol = sig.symbol
                    if symbol in executed_symbols:
                        continue
                        
                    if symbol not in self.history or pd.Timestamp(t_date) not in self.history[symbol].index:
                        continue
                        
                    day_data = self.history[symbol].loc[pd.Timestamp(t_date)]
                    if isinstance(day_data, pd.DataFrame): day_data = day_data.iloc[0]
                        
                    high = float(day_data['High'])
                    open_price = float(day_data['Open'])
                    
                    if high >= sig.entry_price:
                        exec_price = max(open_price, sig.entry_price)
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
                                "strategy_name": sig.metadata["strategy_name"],
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
                                "t1_hit": False,
                                "current_value": qty * float(day_data['Close'])
                            })
                            executed_symbols.add(symbol)
                            
            self.equity_curve.append({
                "Date": pd.Timestamp(t_date),
                "Cash": current_cash,
                "Positions_Value": portfolio_value - current_cash,
                "Equity": portfolio_value
            })
            
        return self._generate_metrics()
        
    def _generate_metrics(self):
        if not self.equity_curve:
            return {}
            
        eq_df = pd.DataFrame(self.equity_curve).set_index("Date")
        eq_df["Return"] = eq_df["Equity"].pct_change()
        
        total_return = (eq_df["Equity"].iloc[-1] / self.initial_capital) - 1
        
        rolling_max = eq_df["Equity"].cummax()
        drawdown = (eq_df["Equity"] - rolling_max) / rolling_max
        mdd = drawdown.min()
        
        daily_rf = 0.05 / 252
        excess_returns = eq_df["Return"].dropna() - daily_rf
        sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std()) if excess_returns.std() > 0 else 0
        
        downside_returns = excess_returns[excess_returns < 0]
        sortino = np.sqrt(252) * (excess_returns.mean() / downside_returns.std()) if downside_returns.std() > 0 else 0
        
        days = (eq_df.index[-1] - eq_df.index[0]).days
        annualized_return = ((eq_df["Equity"].iloc[-1] / self.initial_capital) ** (365 / days)) - 1 if days > 0 else 0
        calmar = annualized_return / abs(mdd) if mdd < 0 else 0
        
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
