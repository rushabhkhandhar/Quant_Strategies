import os
import argparse
from datetime import datetime, date
import pandas as pd

from backtest.engine import BacktestEngine
from backtest.ensemble_engine import EnsembleBacktestEngine
from data.nse_fetcher import load_nifty500_symbols
from strategies.base import BaseStrategy
from strategies.rubber_band import RubberBandStrategy
from strategies.fibonacci_bounce import FibonacciBounceStrategy
from strategies.macd_bb import MacdBbStrategy

STRATEGY_MAP = {
    "Rubber_Band_Mean_Reversion": RubberBandStrategy,
    "Fibonacci_Retracement_Bounce": FibonacciBounceStrategy,
    "MACD_BB_Expansion": MacdBbStrategy
}

def main():
    print("=== Multi-Strategy Backtester ===")
    print("Available Strategies:")
    strategies = list(STRATEGY_MAP.keys())
    for idx, s_name in enumerate(strategies, 1):
        print(f"{idx}. {s_name}")
    print("4. All Strategies (Ensemble Portfolio)")
        
    strat_idx = input("\nSelect strategy number: ").strip()
    
    if strat_idx != "4":
        try:
            selected_strategy = strategies[int(strat_idx) - 1]
        except (ValueError, IndexError):
            print("Invalid selection. Exiting.")
            return
    else:
        selected_strategy = "Ensemble_Portfolio"
        
    start_val = input("Enter start date (dd/mm/yyyy) [Default: 01/01/2024]: ").strip()
    end_val = input(f"Enter end date (dd/mm/yyyy) [Default: {datetime.today().strftime('%d/%m/%Y')}]: ").strip()
    
    if not start_val:
        start_val = "01/01/2024"
    if not end_val:
        end_val = datetime.today().strftime("%d/%m/%Y")
        
    start_date = datetime.strptime(start_val, "%d/%m/%Y").date()
    end_date = datetime.strptime(end_val, "%d/%m/%Y").date()
    
    print(f"\n=== Backtest: {selected_strategy} ===")
    print(f"Period: {start_date} to {end_date}")
    
    print("Loading Universe...")
    symbols = load_nifty500_symbols()
    
    if selected_strategy == "Ensemble_Portfolio":
        strategy_instances = [cls() for cls in STRATEGY_MAP.values()]
        engine = EnsembleBacktestEngine(strategy_instances, symbols, start_date, end_date)
    else:
        strategy_cls = STRATEGY_MAP[selected_strategy]
        strategy_instance = strategy_cls()
        engine = BacktestEngine(strategy_instance, symbols, start_date, end_date)
        
    metrics = engine.run()
    
    if not metrics:
        print("Backtest failed or no trades were taken.")
        return
        
    print("\n" + "="*45)
    print("          BACKTEST RESULTS")
    print("="*45)
    print(f"  Positions Taken    : {metrics['Positions']}")
    print(f"  Position Win Rate  : {metrics['Position_Win_Rate_%']}%")
    print(f"  Profit Factor      : {metrics['Profit_Factor']}")
    print(f"  Avg R-Multiple     : {metrics['Avg_R_Multiple']}")
    print(f"  Expectancy/Position: ₹{metrics['Expectancy_Per_Position']:,.2f}")
    print(f"  Avg Holding Days   : {metrics['Avg_Holding_Days']}")
    print("-"*45)
    print(f"  Total Return       : {metrics['Total_Return_%']}%")
    print(f"  Annualized Return  : {metrics['Annualized_Return_%']}%")
    print(f"  Final Equity       : ₹{metrics['Final_Equity']:,.2f}")
    print(f"  Max Drawdown (MDD) : {metrics['Max_Drawdown_%']}%")
    print(f"  Sharpe Ratio       : {metrics['Sharpe_Ratio']}")
    print(f"  Sortino Ratio      : {metrics['Sortino_Ratio']}")
    print(f"  Calmar Ratio       : {metrics['Calmar_Ratio']}")
    print("-"*45)
    print(f"  Exit Events        : {metrics['Exit_Events']}")
    print("="*45)
    
    # Save Trade Log
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "outputs", "Backtest_Results")
    os.makedirs(out_dir, exist_ok=True)
    
    trade_log: pd.DataFrame = metrics["Trade_Log"]
    if not trade_log.empty:
        log_path = os.path.join(out_dir, f"{selected_strategy}_trades.csv")
        trade_log.to_csv(log_path, index=False)
        print(f"\nSaved detailed trade log to: {log_path}")
        
    # Save Equity Curve
    eq_curve: pd.DataFrame = metrics["Equity_Curve"]
    if not eq_curve.empty:
        eq_path = os.path.join(out_dir, f"{selected_strategy}_equity.csv")
        eq_curve.to_csv(eq_path)
        print(f"Saved equity curve to: {eq_path}")
        
    # Save Metrics Summary
    summary_path = os.path.join(out_dir, f"{selected_strategy}_metrics.csv")
    summary_data = {k: v for k, v in metrics.items() if k not in ["Trade_Log", "Equity_Curve", "Position_Log"]}
    pd.DataFrame([summary_data]).to_csv(summary_path, index=False)
    print(f"Saved metrics summary to: {summary_path}")

if __name__ == "__main__":
    main()
