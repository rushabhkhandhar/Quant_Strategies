import os
import math
import logging
from datetime import date, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

from core.models import CandleSet, Signal
from strategies.base import BaseStrategy
from data.nse_fetcher import fetch_bulk_history

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Directional Signal Simulator.

    This engine simulates portfolio-level backtesting for LONG and SHORT
    strategies using a symmetric cash-accounting model. It is NOT a
    brokerage-level margin simulator:

    - LONG: Cash is spent to buy shares; selling returns proceeds.
    - SHORT: Cash equal to the notional value is reserved as collateral;
      closing the short returns collateral adjusted for PnL.

    This means SHORT positions do not model borrowing fees, margin calls,
    or short-sale restrictions. The model is sufficient for evaluating
    directional signal quality and portfolio-level risk metrics.
    """
    def __init__(self, strategy: BaseStrategy, symbols: List[str], start_date: date, end_date: date, use_wilder_atr: bool = False):
        self.strategy = strategy
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.use_wilder_atr = use_wilder_atr

        # Portfolio Configuration
        self.initial_capital = 50_000.0       # Initial portfolio value
        self.max_allocation_per_trade = 0.10  # Max 10% of portfolio per trade
        self.risk_per_trade_pct = 0.01        # Max 1% portfolio risk per trade
        self.max_gap_pct = 0.03               # Max 3% gap allowed for entry execution

        # Logs
        self.trades = []            # Detailed exit-event log (one per partial/full exit)
        self.closed_positions = []  # Position-level summary (one per completed position)
        self.equity_curve = []

    def pre_fetch_data(self):
        """Fetch data for all symbols to cover the backtest period + lookback."""
        print(f"Pre-fetching historical data for {len(self.symbols)} symbols...")
        if "NIFTYBEES" not in self.symbols:
            self.symbols = self.symbols + ["NIFTYBEES"]
            
        lookback = (self.end_date - self.start_date).days + 300
        
        self.history = fetch_bulk_history(self.symbols, self.end_date, lookback)
        
        # Calculate NIFTYBEES 200-day SMA for Market Regime Filter
        if "NIFTYBEES" in self.history:
            nifty_df = self.history["NIFTYBEES"].copy()
            nifty_df["SMA_200"] = nifty_df["Close"].rolling(200).mean()
            self.history["NIFTYBEES"] = nifty_df
            
        print("Data loaded and vectorized successfully!")
                
    def _run_signal_generation(self):
        print(f"Generating signals for {self.strategy.name}...")
        self.signals = {} # date -> List[Signal]
        
        nifty_df = self.history.get("NIFTYBEES", None)
        
        for symbol, df in self.history.items():
            if symbol == "NIFTYBEES":
                continue # NIFTYBEES is only used for regime filtering
                
            # If the strategy has a meta-labeling model, train it on the symbol's full history first
            if hasattr(self.strategy, 'train_meta_model'):
                print(f"[{symbol}] Training ML Meta-Model...")
                self.strategy.train_meta_model(df)
                
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
                    if float(nifty_day["SMA_200"]) > 0 and float(nifty_day["Close"]) < float(nifty_day["SMA_200"]):
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
                    logger.debug(f"[{symbol}] analyze() failed at {t_date_ts}: {e}")

    def _close_position(self, pos, exit_date_str, exit_price, exit_type):
        """
        Finalise a position: record the last partial exit in the trade log,
        then create exactly one closed_positions record with the total PnL
        across all partial exits.
        """
        remaining_qty = pos['qty']
        revenue = remaining_qty * exit_price
        exit_fee = revenue * 0.0015
        entry_fee_total = remaining_qty * pos['entry_fee_per_share']

        # Direction-aware PnL: LONG profits when exit > entry, SHORT when exit < entry
        if pos['direction'] == "LONG":
            partial_pnl = remaining_qty * (exit_price - pos['entry_price']) - exit_fee - entry_fee_total
            cash_returned = (remaining_qty * exit_price) - exit_fee
        else:
            partial_pnl = remaining_qty * (pos['entry_price'] - exit_price) - exit_fee - entry_fee_total
            cash_returned = remaining_qty * (2.0 * pos['entry_price'] - exit_price) - exit_fee

        self.trades.append({
            "Symbol": pos['symbol'],
            "Entry_Date": pos['entry_date'],
            "Exit_Date": exit_date_str,
            "Type": exit_type,
            "Qty": remaining_qty,
            "Profit": partial_pnl
        })

        # Accumulate into position totals
        total_partial_pnl = pos.get('total_partial_pnl', 0.0) + partial_pnl
        original_qty = pos['original_qty']

        # One closed_positions record per position
        self.closed_positions.append({
            "symbol": pos['symbol'],
            "direction": pos['direction'],
            "entry_date": pos['entry_date'],
            "exit_date": exit_date_str,
            "entry_price": pos['entry_price'],
            "exit_price": exit_price,
            "original_qty": original_qty,
            "pnl": total_partial_pnl,
            "holding_days": pos.get('days_held', 0) + 1,
            "initial_risk_per_share": pos['initial_risk_per_share'],
            "r_multiple": total_partial_pnl / (pos['initial_risk_per_share'] * original_qty) if pos['initial_risk_per_share'] > 0 else 0.0
        })

        return cash_returned  # Net cash received from the final exit

    def _record_partial_exit(self, pos, exit_date_str, exit_qty, exit_price, exit_type):
        """
        Record a partial exit (e.g. Target 1 at 50%) in the trade log
        and accumulate PnL on the position dict.
        Does NOT create a closed_positions record.
        """
        revenue = exit_qty * exit_price
        exit_fee = revenue * 0.0015
        entry_fee_total = exit_qty * pos['entry_fee_per_share']

        # Direction-aware PnL
        if pos['direction'] == "LONG":
            partial_pnl = exit_qty * (exit_price - pos['entry_price']) - exit_fee - entry_fee_total
            cash_returned = (exit_qty * exit_price) - exit_fee
        else:
            partial_pnl = exit_qty * (pos['entry_price'] - exit_price) - exit_fee - entry_fee_total
            cash_returned = exit_qty * (2.0 * pos['entry_price'] - exit_price) - exit_fee

        self.trades.append({
            "Symbol": pos['symbol'],
            "Entry_Date": pos['entry_date'],
            "Exit_Date": exit_date_str,
            "Type": exit_type,
            "Qty": exit_qty,
            "Profit": partial_pnl
        })

        # Accumulate on position
        pos['total_partial_pnl'] = pos.get('total_partial_pnl', 0.0) + partial_pnl
        pos['qty'] -= exit_qty

        return cash_returned  # Net cash received

    def _validate_signal(self, sig, exec_price):
        """
        Validate a signal before opening a position.
        Returns True if the signal is valid, False otherwise.
        Logs warnings for soft issues and errors for hard failures.
        """
        symbol = sig.symbol

        # Hard validations — skip the trade entirely
        if not math.isfinite(exec_price) or exec_price <= 0:
            logger.warning(f"[{symbol}] Skipping: invalid exec_price={exec_price}")
            return False
        if not math.isfinite(sig.stop_loss) or sig.stop_loss <= 0:
            logger.warning(f"[{symbol}] Skipping: invalid stop_loss={sig.stop_loss}")
            return False
        if not math.isfinite(sig.entry_price) or sig.entry_price <= 0:
            logger.warning(f"[{symbol}] Skipping: invalid entry_price={sig.entry_price}")
            return False

        # Direction-specific validations
        if sig.direction == "LONG":
            if sig.stop_loss >= exec_price:
                logger.warning(f"[{symbol}] Skipping LONG: stop_loss={sig.stop_loss} >= exec_price={exec_price}")
                return False
            t1 = sig.targets.get('Target_1', None)
            t2 = sig.targets.get('Target_2', None)
            if t1 is not None and t1 <= exec_price:
                logger.warning(f"[{symbol}] Skipping LONG: Target_1={t1} <= exec_price={exec_price}")
                return False
            if t2 is not None and t2 <= exec_price:
                logger.warning(f"[{symbol}] Skipping LONG: Target_2={t2} <= exec_price={exec_price}")
                return False
        elif sig.direction == "SHORT":
            if sig.stop_loss <= exec_price:
                logger.warning(f"[{symbol}] Skipping SHORT: stop_loss={sig.stop_loss} <= exec_price={exec_price}")
                return False
            t1 = sig.targets.get('Target_1', None)
            t2 = sig.targets.get('Target_2', None)
            if t1 is not None and t1 >= exec_price:
                logger.warning(f"[{symbol}] Skipping SHORT: Target_1={t1} >= exec_price={exec_price}")
                return False
            if t2 is not None and t2 >= exec_price:
                logger.warning(f"[{symbol}] Skipping SHORT: Target_2={t2} >= exec_price={exec_price}")
                return False

        # NaN checks on targets
        for key in ('Target_1', 'Target_2'):
            val = sig.targets.get(key, None)
            if val is not None and (not math.isfinite(val)):
                logger.warning(f"[{symbol}] Skipping: {key}={val} is not finite")
                return False

        # Soft validation — warn but allow
        atr_val = sig.metadata.get('ATR', 0.0)
        if atr_val is None or not math.isfinite(atr_val) or atr_val <= 0:
            logger.warning(f"[{symbol}] ATR missing or invalid ({atr_val}); Chandelier trailing stop will be disabled for this position.")

        return True

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
            
            # --- 1. Update Open Positions (Check for Exits) ---
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
                atr = pos.get('entry_atr', 0.0)
                if atr > 0:
                    if pos['direction'] == "LONG" and high > pos.get('highest_high', 0.0):
                        pos['highest_high'] = high
                        new_sl = pos['highest_high'] - (3.0 * atr)
                        if new_sl > pos['stop_loss']:
                            pos['stop_loss'] = new_sl
                    elif pos['direction'] == "SHORT" and low < pos.get('lowest_low', float('inf')):
                        pos['lowest_low'] = low
                        new_sl = pos['lowest_low'] + (3.0 * atr)
                        if new_sl < pos['stop_loss']:
                            pos['stop_loss'] = new_sl
                            
                exit_triggered = False
                
                # Check Stop Loss First (Pessimistic execution)
                sl_hit_long = pos['direction'] == "LONG" and low <= pos['stop_loss']
                sl_hit_short = pos['direction'] == "SHORT" and high >= pos['stop_loss']
                if sl_hit_long or sl_hit_short:
                    exit_price = pos['stop_loss']
                    cash_received = self._close_position(pos, dt_str, exit_price, "Stop Loss")
                    current_cash += cash_received
                    exit_triggered = True
                    
                # Check Trailing MA Exit (direction-aware)
                elif pos.get('trailing_ma') and pos['trailing_ma'] in day_data:
                    ma_val = float(day_data[pos['trailing_ma']])
                    ma_exit_long = pos['direction'] == "LONG" and close < ma_val
                    ma_exit_short = pos['direction'] == "SHORT" and close > ma_val
                    if ma_exit_long or ma_exit_short:
                        cash_received = self._close_position(pos, dt_str, close, f"Trailing {pos['trailing_ma']}")
                        current_cash += cash_received
                        exit_triggered = True
                    
                # Check Target 1 (50% Sell, Trail SL to Entry)
                elif not pos['t1_hit']:
                    long_target_hit = pos['direction'] == "LONG" and high >= pos['target_1']
                    short_target_hit = pos['direction'] == "SHORT" and low <= pos['target_1']
                    if long_target_hit or short_target_hit:
                        sell_qty = pos['qty'] // 2
                        if sell_qty > 0:
                            exec_target = max(open_price, pos['target_1']) if pos['direction'] == "LONG" else min(open_price, pos['target_1'])
                            cash_received = self._record_partial_exit(pos, dt_str, sell_qty, exec_target, "Target 1 (50%)")
                            current_cash += cash_received
                            pos['t1_hit'] = True
                            pos['stop_loss'] = pos['entry_price']  # Trail SL to breakeven
                        
                            # Check if Target 2 also hit on the same day
                            long_t2_hit = pos['direction'] == "LONG" and high >= pos['target_2']
                            short_t2_hit = pos['direction'] == "SHORT" and low <= pos['target_2']
                            if long_t2_hit or short_t2_hit:
                                exec_target_2 = max(open_price, pos['target_2']) if pos['direction'] == "LONG" else min(open_price, pos['target_2'])
                                cash_received = self._close_position(pos, dt_str, exec_target_2, "Target 2 (Final)")
                                current_cash += cash_received
                                exit_triggered = True
                        
                # Check Target 2 (remaining qty after T1 was hit on a prior day)
                elif pos['t1_hit']:
                    long_cond = pos['direction'] == "LONG" and high >= pos['target_2']
                    short_cond = pos['direction'] == "SHORT" and low <= pos['target_2']
                    if long_cond or short_cond:
                        exec_target = max(open_price, pos['target_2']) if pos['direction'] == "LONG" else min(open_price, pos['target_2'])
                        cash_received = self._close_position(pos, dt_str, exec_target, "Target 2 (Final)")
                        current_cash += cash_received
                        exit_triggered = True
                    
                # Time-based Exit (Stale Trade — only if T1 hasn't been hit)
                if not exit_triggered and not pos['t1_hit'] and pos.get('days_held', 0) >= 19:
                    cash_received = self._close_position(pos, dt_str, close, "Time Exit (20 Days)")
                    current_cash += cash_received
                    exit_triggered = True
                    
                if not exit_triggered:
                    # Mark-to-market: LONG value rises with price, SHORT value rises when price falls
                    if pos['direction'] == "LONG":
                        pos['current_value'] = pos['qty'] * close
                    else:
                        # SHORT: collateral locked = qty × entry; unrealised PnL = qty × (entry - close)
                        pos['current_value'] = pos['qty'] * (2.0 * pos['entry_price'] - close)
                    pos['days_held'] = pos.get('days_held', 0) + 1
                    surviving_positions.append(pos)
                    
            open_positions = surviving_positions
            
            # --- 2. Process New Entries (signals from previous day, executed today) ---
            idx = timeline.index(t_date)
            prev_date_str = timeline[idx-1].strftime("%Y-%m-%d") if idx > 0 else None
                
            daily_signals = self.signals.get(prev_date_str, [])
            daily_signals.sort(key=lambda s: s.metadata.get("rank_score", 0), reverse=True)
            
            for sig in daily_signals:
                if current_cash <= 0: break
                if sig.direction not in ("LONG", "SHORT"): continue
                    
                symbol = sig.symbol
                if any(p['symbol'] == symbol for p in open_positions): continue
                if symbol not in self.history or pd.Timestamp(t_date) not in self.history[symbol].index: continue
                    
                day_data = self.history[symbol].loc[pd.Timestamp(t_date)]
                if isinstance(day_data, pd.DataFrame): day_data = day_data.iloc[0]
                    
                high, low, open_price = float(day_data['High']), float(day_data['Low']), float(day_data['Open'])
                
                # Pending stop-order trigger check
                trigger_hit = (sig.direction == "LONG" and high >= sig.entry_price) or \
                              (sig.direction == "SHORT" and low <= sig.entry_price)
                if not trigger_hit:
                    continue

                # Realistic gap-aware execution pricing
                if sig.direction == "LONG":
                    exec_price = max(open_price, sig.entry_price)
                    # Gap protection: skip if open gaps up too far beyond trigger
                    if open_price > sig.entry_price * (1.0 + self.max_gap_pct):
                        continue
                else:
                    exec_price = min(open_price, sig.entry_price)
                    # Gap protection: skip if open gaps down too far beyond trigger
                    if open_price < sig.entry_price * (1.0 - self.max_gap_pct):
                        continue

                # Defensive signal validation
                if not self._validate_signal(sig, exec_price):
                    continue

                # --- Risk-Based Position Sizing ---
                # Recompute portfolio value to reflect any cash spent on earlier trades today
                portfolio_value = current_cash + sum(p.get('current_value', 0) for p in open_positions)
                
                risk_per_share = abs(exec_price - sig.stop_loss)
                if risk_per_share <= 0:
                    risk_per_share = exec_price * 0.01  # Fallback: assume 1% risk

                # 1. Cap by Risk (Max 1% portfolio loss if SL is hit)
                max_risk_amount = portfolio_value * self.risk_per_trade_pct
                qty_risk = int(max_risk_amount // risk_per_share)

                # 2. Cap by Max Allocation (Max 10% of portfolio per trade)
                max_alloc_amount = portfolio_value * self.max_allocation_per_trade
                qty_alloc = int(max_alloc_amount // exec_price)

                # 3. Cap by Available Cash
                qty_cash = int(current_cash // (exec_price * 1.0015))

                # Take the most conservative constraint
                qty = min(qty_risk, qty_alloc, qty_cash)

                if qty > 0:
                    cost = qty * exec_price
                    entry_fee = cost * 0.0015
                    current_cash -= (cost + entry_fee)
                    t1_val = sig.targets.get('Target_1', exec_price * 1.03 if sig.direction == "LONG" else exec_price * 0.97)
                    t2_val = sig.targets.get('Target_2', exec_price * 1.10 if sig.direction == "LONG" else exec_price * 0.90)
                    open_positions.append({
                        "symbol": symbol,
                        "direction": sig.direction,
                        "entry_date": dt_str,
                        "entry_price": exec_price,
                        "entry_fee_per_share": entry_fee / qty,
                        "qty": qty,
                        "original_qty": qty,
                        "stop_loss": sig.stop_loss,
                        "initial_risk_per_share": risk_per_share,
                        "entry_atr": sig.metadata.get('ATR', 0.0) if math.isfinite(sig.metadata.get('ATR', 0.0) or 0.0) else 0.0,
                        "highest_high": exec_price,
                        "lowest_low": exec_price,
                        "target_1": t1_val,
                        "target_2": t2_val,
                        "trailing_ma": sig.metadata.get("trailing_ma", None),
                        "t1_hit": False,
                        "days_held": 0,
                        "total_partial_pnl": 0.0,
                        "current_value": qty * float(day_data['Close']) if sig.direction == "LONG" else qty * (2.0 * exec_price - float(day_data['Close']))
                    })
            
            self.equity_curve.append({
                "Date": t_date,
                "Equity": current_cash + sum(p.get('current_value', 0) for p in open_positions),
                "Cash": current_cash
            })
            
        return self._generate_metrics()
        
    def _generate_metrics(self):
        if not self.equity_curve: return {}
        eq_df = pd.DataFrame(self.equity_curve).set_index("Date")
        eq_df["Return"] = eq_df["Equity"].pct_change()
        
        # --- Position-Level Metrics ---
        positions_df = pd.DataFrame(self.closed_positions)
        if not positions_df.empty:
            pos_win_rate = (positions_df['pnl'] > 0).mean()
            avg_r = positions_df['r_multiple'].mean()
            avg_holding = positions_df['holding_days'].mean()
            total_profit = positions_df.loc[positions_df['pnl'] > 0, 'pnl'].sum()
            total_loss = -positions_df.loc[positions_df['pnl'] < 0, 'pnl'].sum()
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            expectancy = positions_df['pnl'].mean()
            num_positions = len(positions_df)
        else:
            pos_win_rate = avg_r = avg_holding = profit_factor = expectancy = 0
            num_positions = 0
            
        # --- Equity Curve Metrics ---
        rolling_max = eq_df["Equity"].cummax()
        mdd = ((eq_df["Equity"] - rolling_max) / rolling_max).min()
        excess_returns = eq_df["Return"].dropna() - (0.05 / 252)
        sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std()) if excess_returns.std() > 0 else 0
        
        # Sortino
        downside_dev = np.sqrt(np.mean(np.minimum(0, excess_returns)**2))
        sortino = np.sqrt(252) * (excess_returns.mean() / downside_dev) if downside_dev > 0 else (float('inf') if excess_returns.mean() > 0 else 0.0)
        
        # Annualized Return & Calmar Ratio
        days = (eq_df.index[-1] - eq_df.index[0]).days
        total_return = (eq_df["Equity"].iloc[-1] / self.initial_capital) - 1
        annualized_return = ((eq_df["Equity"].iloc[-1] / self.initial_capital) ** (365 / days)) - 1 if days > 0 else 0
        calmar = annualized_return / abs(mdd) if mdd < 0 else (float('inf') if annualized_return > 0 else 0.0)
        
        # --- Exit-Event Stats (kept separately) ---
        total_exit_events = len(self.trades)
        
        return {
            # Position-Level Metrics
            "Positions": num_positions,
            "Position_Win_Rate_%": round(pos_win_rate * 100, 2),
            "Profit_Factor": round(profit_factor, 2),
            "Avg_R_Multiple": round(avg_r, 2),
            "Avg_Holding_Days": round(avg_holding, 1),
            "Expectancy_Per_Position": round(expectancy, 2),
            # Equity Curve Metrics
            "Total_Return_%": round(total_return * 100, 2),
            "Annualized_Return_%": round(annualized_return * 100, 2),
            "Max_Drawdown_%": round(mdd * 100, 2),
            "Sharpe_Ratio": round(sharpe, 2),
            "Sortino_Ratio": round(sortino, 2),
            "Calmar_Ratio": round(calmar, 2),
            "Final_Equity": round(eq_df["Equity"].iloc[-1], 2),
            # Logs
            "Exit_Events": total_exit_events,
            "Equity_Curve": eq_df,
            "Trade_Log": pd.DataFrame(self.trades),
            "Position_Log": pd.DataFrame(self.closed_positions)
        }
