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
        
        self.initial_capital = 1_000_000.0
        self.max_allocation_per_trade = 0.10
        self.max_open_positions = 10
        
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
        symbols_to_fetch = list(set(self.symbols + ["NIFTYBEES"]))
        print(f"Pre-fetching historical data for {len(symbols_to_fetch)} symbols...")
        lookback = (self.end_date - self.start_date).days + 300
        
        raw_history = fetch_bulk_history(symbols_to_fetch, self.end_date, lookback)
        
        print("Vectorizing indicators across all strategies...")
        for symbol, df in raw_history.items():
            for strategy in self.strategies:
                df = strategy.prepare_data(df)
            self.history[symbol] = df
            
        print("Loading NIFTYBEES Macro Regime Data...")
        if "NIFTYBEES" in self.history:
            nifty = self.history["NIFTYBEES"].copy()
            nifty["EMA_50"] = nifty["Close"].ewm(span=50, adjust=False).mean()
            self.nifty_history = nifty
            print("Macro Regime Data loaded successfully!")
        else:
            print("Warning: NIFTYBEES not found in Bhavcopy data. Regime filter will be disabled.")
            self.nifty_history = pd.DataFrame()
            
        print("Data loaded and fully vectorized successfully!")

    def _run_signal_generation(self):
        print("Generating signals across all strategies...")
        
        for symbol, df in self.history.items():
            for i in range(49, len(df)):
                t_date_ts = df.index[i]
                if t_date_ts.date() < self.start_date:
                    continue
                if t_date_ts.date() > self.end_date:
                    break
                    
                window_df = df.iloc[:i+1]
                temp_candles = CandleSet(symbol=symbol, daily=window_df)
                
                for strategy in self.strategies:
                    try:
                        sig = strategy.analyze(temp_candles)
                        if sig:
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
                    
                high, low, close = float(day_data['High']), float(day_data['Low']), float(day_data['Close'])
                open_price = float(day_data['Open'])
                
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
                
                # Determine Market Regime
                is_bull_market = True
                if hasattr(self, 'nifty_history') and not self.nifty_history.empty:
                    # Find closest previous Nifty date
                    available_dates = self.nifty_history.index[self.nifty_history.index <= pd.Timestamp(t_date)]
                    if len(available_dates) > 0:
                        closest_date = available_dates[-1]
                        nifty_row = self.nifty_history.loc[closest_date]
                        if isinstance(nifty_row, pd.DataFrame): nifty_row = nifty_row.iloc[0]
                        nifty_close = float(nifty_row["Close"])
                        nifty_ema_50 = float(nifty_row["EMA_50"])
                        if nifty_close < nifty_ema_50:
                            is_bull_market = False
                
                for sig in todays_signals:
                    if len(open_positions) >= self.max_open_positions or current_cash <= 0:
                        break
                        
                    # Market Regime Filter: Block Trend/Breakout trades if market is bearish
                    if not is_bull_market and sig.metadata.get("strategy_name") in ["VCP_Breakout", "Inside_Day_Squeeze", "Holy_Grail_Pullback"]:
                        continue
                        
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
                        trade_alloc = min(portfolio_value * self.max_allocation_per_trade, current_cash)
                        qty = int(trade_alloc // exec_price)
                        
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
