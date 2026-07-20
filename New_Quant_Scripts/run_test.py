from datetime import datetime
from backtest.engine import BacktestEngine
from strategies.fibonacci_bounce import FibonacciBounceStrategy
from data.nse_fetcher import load_nifty500_symbols

start_date = datetime.strptime("01/01/2025", "%d/%m/%Y").date()
end_date = datetime.strptime("19/07/2026", "%d/%m/%Y").date()

strategy = FibonacciBounceStrategy()
symbols = load_nifty500_symbols()
engine = BacktestEngine(strategy, symbols, start_date, end_date)
metrics = engine.run()

if metrics:
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
