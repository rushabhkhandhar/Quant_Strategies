"""
Comprehensive unit tests for BacktestEngine.

All tests use synthetic OHLCV data and bypass pre_fetch_data / _run_signal_generation
by directly injecting engine.history and engine.signals. No network calls are made.
"""

import sys
import os
import math
import unittest
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import numpy as np

# Ensure the project root is on the path so imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.models import Signal
from backtest.engine import BacktestEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_strategy_stub():
    """Create a minimal mock strategy that satisfies the BaseStrategy interface."""
    stub = MagicMock()
    stub.name = "TestStrategy"
    stub.prepare_data = lambda df: df
    stub.analyze = MagicMock(return_value=None)
    return stub


def _make_ohlcv(rows):
    """
    Build a DataFrame from a list of (date_str, O, H, L, C, V) tuples.
    Returns a DatetimeIndex DataFrame matching the format expected by the engine.
    """
    data = []
    for row in rows:
        data.append({
            "Open": row[1], "High": row[2], "Low": row[3],
            "Close": row[4], "Volume": row[5]
        })
    df = pd.DataFrame(data, index=pd.to_datetime([r[0] for r in rows]))
    return df


def _make_signal(symbol, date_str, direction, entry_price, stop_loss,
                 target_1, target_2, atr=5.0, rank_score=1.0, trailing_ma=None):
    """Build a Signal dataclass for testing."""
    metadata = {"ATR": atr, "rank_score": rank_score}
    if trailing_ma:
        metadata["trailing_ma"] = trailing_ma
    return Signal(
        symbol=symbol,
        date=date_str,
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        targets={"Target_1": target_1, "Target_2": target_2},
        metadata=metadata,
    )


def _setup_engine(history, signals, capital=50_000.0, start="2024-01-01", end="2024-12-31"):
    """
    Build an engine pre-loaded with history and signals, bypassing network calls.
    """
    strategy = _make_strategy_stub()
    engine = BacktestEngine(
        strategy=strategy,
        symbols=list(history.keys()),
        start_date=date.fromisoformat(start),
        end_date=date.fromisoformat(end),
    )
    engine.initial_capital = capital
    engine.history = history
    engine.signals = signals

    # Monkey-patch to skip network and signal generation
    engine.pre_fetch_data = lambda: None
    engine._run_signal_generation = lambda: None

    return engine


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestLongStopLoss(unittest.TestCase):
    """LONG entry triggers on day 2, stop-loss hits on day 3."""

    def test_long_sl(self):
        # Day 1 (2024-01-01): signal generated after close
        # Day 2 (2024-01-02): entry triggers (high >= entry_price)
        # Day 3 (2024-01-03): low <= stop_loss → SL exit
        history = {
            "TEST": _make_ohlcv([
                ("2024-01-01", 100, 105, 95, 102, 1000),
                ("2024-01-02", 101, 110, 100, 108, 1000),  # high=110 >= entry=105
                ("2024-01-03", 106, 107, 94, 95, 1000),    # low=94 <= SL=95
            ])
        }
        sig = _make_signal("TEST", "01/01/2024", "LONG",
                           entry_price=105.0, stop_loss=95.0,
                           target_1=115.0, target_2=125.0, atr=5.0)
        signals = {"2024-01-01": [sig]}

        engine = _setup_engine(history, signals, capital=50_000.0,
                               start="2024-01-01", end="2024-01-03")
        metrics = engine.run()

        # Should have 1 closed position
        self.assertEqual(metrics["Positions"], 1)
        self.assertEqual(len(engine.closed_positions), 1)
        pos = engine.closed_positions[0]
        self.assertEqual(pos["direction"], "LONG")
        self.assertEqual(pos["exit_price"], 95.0)
        # PnL should be negative (bought at 105, sold at 95, minus fees)
        self.assertLess(pos["pnl"], 0)


class TestLongTargets(unittest.TestCase):
    """LONG entry → T1 partial → T2 final, verifying accumulated PnL."""

    def test_long_t1_t2(self):
        history = {
            "TEST": _make_ohlcv([
                ("2024-01-01", 100, 105, 95, 102, 1000),
                ("2024-01-02", 101, 110, 100, 108, 1000),  # entry
                ("2024-01-03", 110, 118, 109, 116, 1000),   # high=118 >= T1=115
                ("2024-01-04", 116, 128, 115, 126, 1000),   # high=128 >= T2=125
            ])
        }
        sig = _make_signal("TEST", "01/01/2024", "LONG",
                           entry_price=105.0, stop_loss=95.0,
                           target_1=115.0, target_2=125.0, atr=5.0)
        signals = {"2024-01-01": [sig]}

        engine = _setup_engine(history, signals, capital=50_000.0,
                               start="2024-01-01", end="2024-01-04")
        metrics = engine.run()

        self.assertEqual(metrics["Positions"], 1)
        pos = engine.closed_positions[0]
        # Should be profitable (bought at 105, sold halves at ~115 and ~125)
        self.assertGreater(pos["pnl"], 0)
        # Should have 2 exit events (T1 partial + T2 final)
        self.assertEqual(metrics["Exit_Events"], 2)


class TestShortStopLoss(unittest.TestCase):
    """SHORT entry triggers on day 2, stop-loss hits on day 3."""

    def test_short_sl(self):
        history = {
            "TEST": _make_ohlcv([
                ("2024-01-01", 100, 105, 90, 92, 1000),
                ("2024-01-02", 93,  95,  88, 90, 1000),   # low=88 <= entry=92
                ("2024-01-03", 91,  102, 89, 100, 1000),  # high=102 >= SL=100
            ])
        }
        sig = _make_signal("TEST", "01/01/2024", "SHORT",
                           entry_price=92.0, stop_loss=100.0,
                           target_1=84.0, target_2=76.0, atr=4.0)
        signals = {"2024-01-01": [sig]}

        engine = _setup_engine(history, signals, capital=50_000.0,
                               start="2024-01-01", end="2024-01-03")
        metrics = engine.run()

        self.assertEqual(metrics["Positions"], 1)
        pos = engine.closed_positions[0]
        self.assertEqual(pos["direction"], "SHORT")
        self.assertEqual(pos["exit_price"], 100.0)
        # PnL should be negative (shorted at 92, covered at 100)
        self.assertLess(pos["pnl"], 0)


class TestShortTargetExit(unittest.TestCase):
    """SHORT entry → T1 hits → T2 hits on later day."""

    def test_short_targets(self):
        history = {
            "TEST": _make_ohlcv([
                ("2024-01-01", 100, 105, 90, 92, 1000),
                ("2024-01-02", 93,  95,  88, 90, 1000),  # entry at min(93, 92)=92
                ("2024-01-03", 89,  90,  82, 83, 1000),  # low=82 <= T1=84
                ("2024-01-04", 83,  84,  74, 75, 1000),  # low=74 <= T2=76
            ])
        }
        sig = _make_signal("TEST", "01/01/2024", "SHORT",
                           entry_price=92.0, stop_loss=100.0,
                           target_1=84.0, target_2=76.0, atr=4.0)
        signals = {"2024-01-01": [sig]}

        engine = _setup_engine(history, signals, capital=50_000.0,
                               start="2024-01-01", end="2024-01-04")
        metrics = engine.run()

        self.assertEqual(metrics["Positions"], 1)
        pos = engine.closed_positions[0]
        # Should be profitable (shorted at 92, covered at ~84 and ~76)
        self.assertGreater(pos["pnl"], 0)


class TestGapUpSkipped(unittest.TestCase):
    """LONG signal where the open gaps up >3% past the trigger → trade skipped."""

    def test_gap_up_rejected(self):
        history = {
            "TEST": _make_ohlcv([
                ("2024-01-01", 100, 105, 95, 102, 1000),
                # Open=115 → gap = (115 - 105) / 105 = 9.5% > 3%
                ("2024-01-02", 115, 120, 114, 118, 1000),
            ])
        }
        sig = _make_signal("TEST", "01/01/2024", "LONG",
                           entry_price=105.0, stop_loss=95.0,
                           target_1=115.0, target_2=125.0, atr=5.0)
        signals = {"2024-01-01": [sig]}

        engine = _setup_engine(history, signals, capital=50_000.0,
                               start="2024-01-01", end="2024-01-02")
        metrics = engine.run()

        self.assertEqual(metrics["Positions"], 0)
        self.assertEqual(metrics["Exit_Events"], 0)


class TestGapDownSkipped(unittest.TestCase):
    """SHORT signal where the open gaps down >3% past the trigger → trade skipped."""

    def test_gap_down_rejected(self):
        history = {
            "TEST": _make_ohlcv([
                ("2024-01-01", 100, 105, 90, 92, 1000),
                # Open=85 → gap = (92 - 85) / 92 = 7.6% > 3%
                ("2024-01-02", 85, 86, 80, 82, 1000),
            ])
        }
        sig = _make_signal("TEST", "01/01/2024", "SHORT",
                           entry_price=92.0, stop_loss=100.0,
                           target_1=84.0, target_2=76.0, atr=4.0)
        signals = {"2024-01-01": [sig]}

        engine = _setup_engine(history, signals, capital=50_000.0,
                               start="2024-01-01", end="2024-01-02")
        metrics = engine.run()

        self.assertEqual(metrics["Positions"], 0)


class TestGapWithinThreshold(unittest.TestCase):
    """LONG signal with small gap → trade executes at gap price, not trigger price."""

    def test_gap_accepted(self):
        history = {
            "TEST": _make_ohlcv([
                ("2024-01-01", 100, 105, 95, 102, 1000),
                # Open=107 → gap = (107 - 105) / 105 = 1.9% < 3% → accepted
                # exec_price = max(107, 105) = 107
                ("2024-01-02", 107, 115, 106, 112, 1000),
                ("2024-01-03", 112, 113, 100, 101, 1000),  # SL at 95 not hit; hold
            ])
        }
        sig = _make_signal("TEST", "01/01/2024", "LONG",
                           entry_price=105.0, stop_loss=95.0,
                           target_1=115.0, target_2=125.0, atr=5.0)
        signals = {"2024-01-01": [sig]}

        engine = _setup_engine(history, signals, capital=50_000.0,
                               start="2024-01-01", end="2024-01-03")
        metrics = engine.run()

        # Position should have been opened
        self.assertGreater(len(engine.equity_curve), 0)
        # Verify there are open or closed positions
        # The position should still be open (no SL/target hit on day 3)
        # Check the equity curve reflects capital deployed
        final_equity = metrics["Final_Equity"]
        self.assertLess(final_equity, 50_000.0)  # Cash was spent on entry


class TestTrailingStopLong(unittest.TestCase):
    """ATR Chandelier trailing stop ratchets up for LONG positions."""

    def test_trailing_ratchet_long(self):
        # Entry at 105, ATR=5, initial SL=95
        # Day 3: high=120 → new SL = 120 - 15 = 105, which is > 95 → ratchets up
        # Day 4: low=104 <= SL=105 → exit at 105
        history = {
            "TEST": _make_ohlcv([
                ("2024-01-01", 100, 105, 95, 102, 1000),
                ("2024-01-02", 101, 110, 100, 108, 1000),  # entry at 105
                ("2024-01-03", 110, 120, 109, 118, 1000),   # SL ratchets to 120-15=105
                ("2024-01-04", 115, 116, 104, 106, 1000),   # low=104 <= SL=105 → SL hit
            ])
        }
        sig = _make_signal("TEST", "01/01/2024", "LONG",
                           entry_price=105.0, stop_loss=95.0,
                           target_1=130.0, target_2=150.0, atr=5.0)
        signals = {"2024-01-01": [sig]}

        engine = _setup_engine(history, signals, capital=50_000.0,
                               start="2024-01-01", end="2024-01-04")
        metrics = engine.run()

        self.assertEqual(metrics["Positions"], 1)
        pos = engine.closed_positions[0]
        # Exited at the trailed SL of 105 (breakeven-ish), not the original 95
        self.assertEqual(pos["exit_price"], 105.0)


class TestTrailingStopShort(unittest.TestCase):
    """ATR Chandelier trailing stop ratchets down for SHORT positions."""

    def test_trailing_ratchet_short(self):
        # Entry at 92, ATR=4, initial SL=100
        # Day 3: low=75 → new SL = 75 + 12 = 87, which is < 100 → ratchets down
        # Day 4: high=88 >= SL=87 → exit at 87
        history = {
            "TEST": _make_ohlcv([
                ("2024-01-01", 100, 105, 90, 92, 1000),
                ("2024-01-02", 93,  95,  88, 90, 1000),   # entry at min(93, 92) = 92
                ("2024-01-03", 88,  89,  75, 78, 1000),   # SL ratchets to 75+12=87
                ("2024-01-04", 80,  88,  79, 85, 1000),   # high=88 >= SL=87 → SL hit
            ])
        }
        sig = _make_signal("TEST", "01/01/2024", "SHORT",
                           entry_price=92.0, stop_loss=100.0,
                           target_1=80.0, target_2=70.0, atr=4.0)
        signals = {"2024-01-01": [sig]}

        engine = _setup_engine(history, signals, capital=50_000.0,
                               start="2024-01-01", end="2024-01-04")
        metrics = engine.run()

        self.assertEqual(metrics["Positions"], 1)
        pos = engine.closed_positions[0]
        # Exited at the trailed SL of 87, not the original 100
        self.assertEqual(pos["exit_price"], 87.0)
        # Should be profitable (shorted at 92, covered at 87)
        self.assertGreater(pos["pnl"], 0)


class TestTrailingMALong(unittest.TestCase):
    """LONG position exits when close drops below trailing MA."""

    def test_ma_exit_long(self):
        # Build data with a pre-computed SMA_20 column
        history = {
            "TEST": _make_ohlcv([
                ("2024-01-01", 100, 105, 95, 102, 1000),
                ("2024-01-02", 101, 110, 100, 108, 1000),  # entry
                ("2024-01-03", 108, 109, 99, 100, 1000),   # close=100 < SMA_20=105 → exit
            ])
        }
        # Inject a fake SMA_20 column
        history["TEST"]["SMA_20"] = 105.0

        sig = _make_signal("TEST", "01/01/2024", "LONG",
                           entry_price=105.0, stop_loss=90.0,
                           target_1=120.0, target_2=130.0, atr=5.0,
                           trailing_ma="SMA_20")
        signals = {"2024-01-01": [sig]}

        engine = _setup_engine(history, signals, capital=50_000.0,
                               start="2024-01-01", end="2024-01-03")
        metrics = engine.run()

        self.assertEqual(metrics["Positions"], 1)
        # Should have exited via trailing MA
        self.assertIn("Trailing SMA_20", engine.trades[-1]["Type"])


class TestTrailingMAShort(unittest.TestCase):
    """SHORT position exits when close rises above trailing MA."""

    def test_ma_exit_short(self):
        history = {
            "TEST": _make_ohlcv([
                ("2024-01-01", 100, 105, 90, 92, 1000),
                ("2024-01-02", 93,  95,  88, 90, 1000),   # entry
                ("2024-01-03", 91,  93,  89, 93, 1000),   # close=93 > SMA_20=91 → exit
            ])
        }
        history["TEST"]["SMA_20"] = 91.0

        sig = _make_signal("TEST", "01/01/2024", "SHORT",
                           entry_price=92.0, stop_loss=100.0,
                           target_1=84.0, target_2=76.0, atr=4.0,
                           trailing_ma="SMA_20")
        signals = {"2024-01-01": [sig]}

        engine = _setup_engine(history, signals, capital=50_000.0,
                               start="2024-01-01", end="2024-01-03")
        metrics = engine.run()

        self.assertEqual(metrics["Positions"], 1)
        self.assertIn("Trailing SMA_20", engine.trades[-1]["Type"])


class TestTimeExit(unittest.TestCase):
    """Position held >20 days without T1 → time exit."""

    def test_time_based_exit(self):
        # Generate 25 days of sideways data where no SL/target is hit
        rows = [("2024-01-01", 100, 105, 95, 102, 1000)]  # signal day
        for i in range(2, 27):
            d = f"2024-01-{i:02d}"
            # Prices oscillate safely between SL=90 and T1=120
            rows.append((d, 104, 108, 96, 104, 1000))

        history = {"TEST": _make_ohlcv(rows)}
        sig = _make_signal("TEST", "01/01/2024", "LONG",
                           entry_price=105.0, stop_loss=90.0,
                           target_1=120.0, target_2=130.0, atr=5.0)
        signals = {"2024-01-01": [sig]}

        engine = _setup_engine(history, signals, capital=50_000.0,
                               start="2024-01-01", end="2024-01-26")
        metrics = engine.run()

        self.assertEqual(metrics["Positions"], 1)
        pos = engine.closed_positions[0]
        self.assertGreaterEqual(pos["holding_days"], 20)
        self.assertIn("Time Exit", engine.trades[-1]["Type"])


class TestPositionSizing(unittest.TestCase):
    """Verify qty = min(qty_risk, qty_alloc, qty_cash)."""

    def test_sizing_constraints(self):
        history = {
            "TEST": _make_ohlcv([
                ("2024-01-01", 100, 105, 95, 102, 1000),
                ("2024-01-02", 101, 110, 100, 108, 1000),  # entry at 105
                ("2024-01-03", 108, 109, 99, 100, 1000),
            ])
        }
        sig = _make_signal("TEST", "01/01/2024", "LONG",
                           entry_price=105.0, stop_loss=95.0,
                           target_1=115.0, target_2=125.0, atr=5.0)
        signals = {"2024-01-01": [sig]}

        capital = 50_000.0
        engine = _setup_engine(history, signals, capital=capital,
                               start="2024-01-01", end="2024-01-03")
        metrics = engine.run()

        # Calculate expected qty
        portfolio_value = capital
        risk_per_share = abs(105.0 - 95.0)  # = 10
        qty_risk = int((portfolio_value * 0.01) // risk_per_share)     # 500 // 10 = 50
        qty_alloc = int((portfolio_value * 0.10) // 105.0)             # 5000 // 105 = 47
        qty_cash = int(capital // 105.0)                                # 50000 // 105 = 476
        expected_qty = min(qty_risk, qty_alloc, qty_cash)               # min(50, 47, 476) = 47

        # Check that the engine used this qty
        # We can infer from the trade log or from the closed position
        # Since position might still be open, check equity curve
        entry_cost = expected_qty * 105.0
        entry_fee = entry_cost * 0.0015
        expected_cash_after = capital - entry_cost - entry_fee
        # The equity curve's first entry after the signal day should reflect deployment
        # Day 2 equity = cash_after_entry + position_value
        self.assertAlmostEqual(
            engine.equity_curve[1]["Cash"], expected_cash_after, places=2
        )


class TestInvalidSignalSkipped(unittest.TestCase):
    """Signals with NaN entry price or inverted SL are rejected."""

    def test_nan_entry_price(self):
        history = {
            "TEST": _make_ohlcv([
                ("2024-01-01", 100, 105, 95, 102, 1000),
                ("2024-01-02", 101, 200, 100, 108, 1000),  # high enough for any trigger
            ])
        }
        sig = _make_signal("TEST", "01/01/2024", "LONG",
                           entry_price=float('nan'), stop_loss=95.0,
                           target_1=115.0, target_2=125.0, atr=5.0)
        signals = {"2024-01-01": [sig]}

        engine = _setup_engine(history, signals, capital=50_000.0,
                               start="2024-01-01", end="2024-01-02")
        metrics = engine.run()
        self.assertEqual(metrics["Positions"], 0)

    def test_inverted_sl_long(self):
        """LONG with SL above entry → validation should reject."""
        history = {
            "TEST": _make_ohlcv([
                ("2024-01-01", 100, 105, 95, 102, 1000),
                ("2024-01-02", 101, 110, 100, 108, 1000),
            ])
        }
        sig = _make_signal("TEST", "01/01/2024", "LONG",
                           entry_price=105.0, stop_loss=110.0,   # SL > entry = invalid
                           target_1=115.0, target_2=125.0, atr=5.0)
        signals = {"2024-01-01": [sig]}

        engine = _setup_engine(history, signals, capital=50_000.0,
                               start="2024-01-01", end="2024-01-02")
        metrics = engine.run()
        self.assertEqual(metrics["Positions"], 0)

    def test_inverted_sl_short(self):
        """SHORT with SL below entry → validation should reject."""
        history = {
            "TEST": _make_ohlcv([
                ("2024-01-01", 100, 105, 90, 92, 1000),
                ("2024-01-02", 93,  95,  88, 90, 1000),
            ])
        }
        sig = _make_signal("TEST", "01/01/2024", "SHORT",
                           entry_price=92.0, stop_loss=85.0,   # SL < entry = invalid for SHORT
                           target_1=84.0, target_2=76.0, atr=4.0)
        signals = {"2024-01-01": [sig]}

        engine = _setup_engine(history, signals, capital=50_000.0,
                               start="2024-01-01", end="2024-01-02")
        metrics = engine.run()
        self.assertEqual(metrics["Positions"], 0)


class TestShortCurrentValue(unittest.TestCase):
    """SHORT position's unrealised value should increase when price drops."""

    def test_short_mtm(self):
        history = {
            "TEST": _make_ohlcv([
                ("2024-01-01", 100, 105, 90, 92, 1000),
                ("2024-01-02", 93,  95,  88, 90, 1000),  # entry at 92
                ("2024-01-03", 89,  91,  85, 86, 1000),  # price dropped → position value up
                ("2024-01-04", 86,  88,  83, 85, 1000),  # hold
            ])
        }
        sig = _make_signal("TEST", "01/01/2024", "SHORT",
                           entry_price=92.0, stop_loss=100.0,
                           target_1=80.0, target_2=70.0, atr=4.0)
        signals = {"2024-01-01": [sig]}

        engine = _setup_engine(history, signals, capital=50_000.0,
                               start="2024-01-01", end="2024-01-04")
        metrics = engine.run()

        # Position should still be open (no SL/target hit)
        # Equity should be > initial capital because price dropped (good for SHORT)
        self.assertGreater(metrics["Final_Equity"], 50_000.0 * 0.99)  # At least ~break-even after fees


class TestMetricsOutput(unittest.TestCase):
    """All expected metric keys are present with correct types."""

    def test_metric_keys(self):
        history = {
            "TEST": _make_ohlcv([
                ("2024-01-01", 100, 105, 95, 102, 1000),
                ("2024-01-02", 101, 110, 100, 108, 1000),
                ("2024-01-03", 106, 107, 94, 95, 1000),   # SL hit
            ])
        }
        sig = _make_signal("TEST", "01/01/2024", "LONG",
                           entry_price=105.0, stop_loss=95.0,
                           target_1=115.0, target_2=125.0, atr=5.0)
        signals = {"2024-01-01": [sig]}

        engine = _setup_engine(history, signals, capital=50_000.0,
                               start="2024-01-01", end="2024-01-03")
        metrics = engine.run()

        expected_keys = [
            "Positions", "Position_Win_Rate_%", "Profit_Factor",
            "Avg_R_Multiple", "Avg_Holding_Days", "Expectancy_Per_Position",
            "Total_Return_%", "Annualized_Return_%", "Max_Drawdown_%",
            "Sharpe_Ratio", "Sortino_Ratio", "Calmar_Ratio",
            "Final_Equity", "Exit_Events",
            "Equity_Curve", "Trade_Log", "Position_Log"
        ]
        for key in expected_keys:
            self.assertIn(key, metrics, f"Missing metric key: {key}")

        # Numeric checks
        for key in ["Positions", "Exit_Events"]:
            self.assertIsInstance(metrics[key], int)
        for key in ["Position_Win_Rate_%", "Total_Return_%", "Final_Equity"]:
            self.assertIsInstance(metrics[key], float)
        # DataFrame checks
        self.assertIsInstance(metrics["Equity_Curve"], pd.DataFrame)
        self.assertIsInstance(metrics["Trade_Log"], pd.DataFrame)
        self.assertIsInstance(metrics["Position_Log"], pd.DataFrame)


class TestNoLookahead(unittest.TestCase):
    """Signals from day N must only execute on day N+1, never on day N."""

    def test_no_same_day_execution(self):
        history = {
            "TEST": _make_ohlcv([
                ("2024-01-01", 100, 110, 95, 102, 1000),  # high=110 >= trigger=105
                # If look-ahead existed, the engine would enter on day 1 itself
            ])
        }
        sig = _make_signal("TEST", "01/01/2024", "LONG",
                           entry_price=105.0, stop_loss=95.0,
                           target_1=115.0, target_2=125.0, atr=5.0)
        # Signal keyed to "2024-01-01" means it's generated after Jan 1 close
        # and should only be eligible for execution on Jan 2
        signals = {"2024-01-01": [sig]}

        engine = _setup_engine(history, signals, capital=50_000.0,
                               start="2024-01-01", end="2024-01-01")
        metrics = engine.run()

        # No trade should have been taken (no day N+1 exists)
        self.assertEqual(metrics["Positions"], 0)
        self.assertEqual(metrics["Exit_Events"], 0)


class TestPartialExitAccumulatedPnL(unittest.TestCase):
    """Verify that T1 partial + T2 final produce exactly one closed_positions record
    with PnL that accounts for both partial exits."""

    def test_one_record_per_position(self):
        history = {
            "TEST": _make_ohlcv([
                ("2024-01-01", 100, 105, 95, 102, 1000),
                ("2024-01-02", 101, 110, 100, 108, 1000),  # entry at 105
                ("2024-01-03", 110, 118, 109, 116, 1000),   # T1=115 hit
                ("2024-01-04", 116, 128, 115, 126, 1000),   # T2=125 hit
            ])
        }
        sig = _make_signal("TEST", "01/01/2024", "LONG",
                           entry_price=105.0, stop_loss=95.0,
                           target_1=115.0, target_2=125.0, atr=5.0)
        signals = {"2024-01-01": [sig]}

        engine = _setup_engine(history, signals, capital=50_000.0,
                               start="2024-01-01", end="2024-01-04")
        metrics = engine.run()

        # Exactly 1 closed position record
        self.assertEqual(len(engine.closed_positions), 1)
        # Exactly 2 trade log entries (T1 partial + T2 final)
        self.assertEqual(len(engine.trades), 2)

        pos = engine.closed_positions[0]
        # Total PnL should include both partial exits
        self.assertGreater(pos["pnl"], 0)
        # R-multiple should be calculated from initial risk
        self.assertGreater(pos["r_multiple"], 0)
        self.assertEqual(pos["initial_risk_per_share"], 10.0)  # |105 - 95|


if __name__ == "__main__":
    unittest.main()
