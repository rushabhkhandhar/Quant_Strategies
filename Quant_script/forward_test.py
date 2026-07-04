"""Forward-testing (paper-trading) engine for Fibonacci Retracement strategy.

Reads daily signal CSVs produced by Fibonacci_Retracement_Candlestick_Volume.py
and tracks each trade through its full lifecycle:
  Signal → Pending Entry (2-day window) → Active → Exit (SL/Target/Timeout)

Trade rules
-----------
- Entry must fill within 2 *trading* days (weekends excluded).  If not, the
  stock is dropped and its capital is kept as cash (excluded from P&L).
- SL1 hit  → exit immediately, keep cash.
- T1 hit   → trailing SL activated at T1 price.  Targets stay.
- T2 hit   → trailing SL moved up to T2 price.  Targets stay.
- T3 hit   → exit at T3. Trade complete.
- 21 trading-day hold  → exit at close of day 21.

Capital allocation
------------------
Equal allocation per stock within a batch.  Default ₹1,00,000 per batch.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from Fibonacci_Retracement_Candlestick_Volume import (
	_download_bhavcopy_for_date,
	_find_column,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_CAPITAL_PER_BATCH: float = 1_00_000.0  # ₹1 Lakh
MAX_HOLD_DAYS: int = 21
ENTRY_WINDOW_DAYS: int = 2


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_output_dir() -> str:
	"""Return (and create) the fronttest output directory."""
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	output_dir = os.path.join(base_dir, "Output", "fronttest")
	os.makedirs(output_dir, exist_ok=True)
	return output_dir


def _fmt(d: date) -> str:
	"""ISO format for internal keys: YYYY-MM-DD."""
	return d.strftime("%Y-%m-%d")


def _parse(d: str) -> date:
	"""Parse YYYY-MM-DD string to date."""
	return datetime.strptime(d, "%Y-%m-%d").date()


def _display(d: date) -> str:
	"""DD/MM/YYYY for user-facing display."""
	return d.strftime("%d/%m/%Y")


# ──────────────────────────────────────────────────────────────────────────────
# Market data helper
# ──────────────────────────────────────────────────────────────────────────────
def fetch_day_prices(
	symbols: List[str],
	trade_date: date,
) -> Dict[str, Dict[str, float]]:
	"""Return ``{SYMBOL: {Open, High, Low, Close, Volume}}`` from NSE bhavcopy.

	Returns an empty dict when no bhavcopy is available (weekend / holiday).
	"""
	df = _download_bhavcopy_for_date(trade_date)
	if df is None or df.empty:
		return {}

	symbol_col = _find_column(df.columns, ["SYMBOL"])
	series_col = _find_column(df.columns, ["SERIES"])
	open_col = _find_column(df.columns, ["OPEN_PRICE", "OPEN"])
	high_col = _find_column(df.columns, ["HIGH_PRICE", "HIGH"])
	low_col = _find_column(df.columns, ["LOW_PRICE", "LOW"])
	close_col = _find_column(df.columns, ["CLOSE_PRICE", "CLOSE", "CLOSE_PRICE_"])
	volume_col = _find_column(
		df.columns, ["TOTTRDQTY", "TTL_TRD_QNTY", "VOLUME", "TOTTRD_QTY"]
	)

	if not all([symbol_col, series_col, open_col, high_col, low_col, close_col]):
		return {}

	work = df.copy()
	work[symbol_col] = work[symbol_col].astype(str).str.strip().str.upper()
	work[series_col] = work[series_col].astype(str).str.strip().str.upper()
	work = work[work[series_col] == "EQ"]

	symbol_set = {s.upper() for s in symbols}
	work = work[work[symbol_col].isin(symbol_set)]

	result: Dict[str, Dict[str, float]] = {}
	for _, row in work.iterrows():
		sym = str(row[symbol_col]).strip().upper()
		try:
			result[sym] = {
				"Open": float(row[open_col]),
				"High": float(row[high_col]),
				"Low": float(row[low_col]),
				"Close": float(row[close_col]),
				"Volume": float(row[volume_col]) if volume_col else 0.0,
			}
		except (ValueError, TypeError):
			continue
	return result


# ──────────────────────────────────────────────────────────────────────────────
# ForwardTestTracker
# ──────────────────────────────────────────────────────────────────────────────
class ForwardTestTracker:
	"""Stateful tracker for all forward-test batches and trades."""

	def __init__(self, capital_per_batch: float = DEFAULT_CAPITAL_PER_BATCH) -> None:
		self.output_dir = get_output_dir()
		self.state_file = os.path.join(self.output_dir, "forward_test_state.json")
		self.capital_per_batch = capital_per_batch
		self.state: Dict[str, Any] = self._load_state()

	# ── persistence ──────────────────────────────────────────────────────
	def _load_state(self) -> Dict[str, Any]:
		if os.path.exists(self.state_file):
			with open(self.state_file, "r") as fh:
				return json.load(fh)
		return self._empty_state()

	def _empty_state(self) -> Dict[str, Any]:
		return {
			"config": {
				"capital_per_batch": self.capital_per_batch,
				"max_hold_days": MAX_HOLD_DAYS,
				"entry_window_days": ENTRY_WINDOW_DAYS,
			},
			"trades": [],
			"daily_pnl_log": [],
		}

	def _save_state(self) -> None:
		with open(self.state_file, "w") as fh:
			json.dump(self.state, fh, indent=2, default=str)

	def reset_state(self) -> None:
		"""Clear all trades (used before backfill)."""
		self.state = self._empty_state()
		self._save_state()

	# ── helpers ───────────────────────────────────────────────────────────
	@staticmethod
	def _trade_id(batch_date_str: str, symbol: str) -> str:
		return f"{batch_date_str}_{symbol}"

	def _existing_ids(self) -> set:
		return {t["trade_id"] for t in self.state["trades"]}

	# ── ingest new batch ──────────────────────────────────────────────────
	def ingest_new_batch(self, signal_csv_path: str, batch_date: date) -> int:
		"""Register new signals as pending trades.  Returns count added."""
		df = pd.read_csv(signal_csv_path)
		if df.empty:
			return 0

		existing = self._existing_ids()
		bd = _fmt(batch_date)
		n_stocks = len(df)
		alloc = self.capital_per_batch / max(n_stocks, 1)
		added = 0

		for _, row in df.iterrows():
			sym = str(row["symbol"]).strip().upper()
			tid = self._trade_id(bd, sym)
			if tid in existing:
				continue

			entry_price = float(row["entry_price"])
			shares = int(alloc // entry_price) if entry_price > 0 else 0

			trade: Dict[str, Any] = {
				# identifiers
				"trade_id": tid,
				"batch_date": bd,
				"symbol": sym,
				"signal_date": str(row.get("date", "")),
				"signal_close": round(float(row.get("close", 0)), 2),
				# levels from scanner
				"entry_price": round(entry_price, 2),
				"stop_loss_1": round(float(row["stop_loss_1"]), 2),
				"stop_loss_2": round(float(row.get("stop_loss_2", 0)), 2),
				"stop_loss_3": round(float(row.get("stop_loss_3", 0)), 2),
				"target_1": round(float(row["target_1"]), 2),
				"target_2": round(float(row["target_2"]), 2),
				"target_3": round(float(row["target_3"]), 2),
				"swing_low": round(float(row.get("swing_low", 0)), 2),
				"swing_high": round(float(row.get("swing_high", 0)), 2),
				"fib_382": round(float(row.get("fib_382", 0)), 2),
				"fib_50": round(float(row.get("fib_50", 0)), 2),
				"fib_618": round(float(row.get("fib_618", 0)), 2),
				"pattern": str(row.get("pattern", "")),
				"entry_level_name": str(row.get("entry_level_name", "")),
				# capital
				"capital_allocated": round(alloc, 2),
				"num_shares": shares,
				"batch_stock_count": n_stocks,
				# state
				"status": "pending_entry",
				"entry_date": None,
				"entry_attempts": 0,
				"exit_date": None,
				"exit_price": None,
				"exit_reason": None,
				"days_held": 0,
				"current_sl": round(float(row["stop_loss_1"]), 2),
				"trailing_sl_active": False,
				"t1_hit": False,
				"t1_hit_date": None,
				"t2_hit": False,
				"t2_hit_date": None,
				"t3_hit": False,
				"t3_hit_date": None,
				"pnl_pct": 0.0,
				"pnl_absolute": 0.0,
				"last_close": None,
				"action_log": [
					f"[{bd}] SIGNAL: {sym} added to batch "
					f"({n_stocks} stocks, ₹{alloc:,.0f} alloc, {shares} shares). "
					f"Entry ₹{entry_price:.2f} | SL1 ₹{float(row['stop_loss_1']):.2f} | "
					f"T1 ₹{float(row['target_1']):.2f} | T2 ₹{float(row['target_2']):.2f} | "
					f"T3 ₹{float(row['target_3']):.2f} | Pattern: {row.get('pattern', '')}."
				],
			}
			self.state["trades"].append(trade)
			added += 1

		self._save_state()
		return added

	# ── main daily update ─────────────────────────────────────────────────
	def update_daily(self, trade_date: date) -> Dict[str, Any]:
		"""Fetch today's prices and update every pending / active trade."""
		live_statuses = ("pending_entry", "active", "t1_hit", "t2_hit")
		symbols_needed = {
			t["symbol"]
			for t in self.state["trades"]
			if t["status"] in live_statuses
		}
		if not symbols_needed:
			return {"updated": 0, "entries": 0, "exits": 0, "events": []}

		prices = fetch_day_prices(list(symbols_needed), trade_date)
		if not prices:
			return {
				"updated": 0,
				"entries": 0,
				"exits": 0,
				"events": [],
				"note": f"No market data for {_display(trade_date)} (holiday/weekend?)",
			}

		td = _fmt(trade_date)
		events: List[str] = []
		entries = exits = updated = 0

		for trade in self.state["trades"]:
			if trade["status"] not in live_statuses:
				continue

			sym = trade["symbol"]
			if sym not in prices:
				if trade["status"] == "pending_entry":
					trade["action_log"].append(
						f"[{td}] NO_DATA: {sym} missing from bhavcopy "
						f"(possibly suspended). Entry attempt NOT counted."
					)
				continue

			ohlcv = prices[sym]
			updated += 1

			if trade["status"] == "pending_entry":
				e, x = self._process_pending(trade, ohlcv, trade_date, td)
				entries += e
				exits += x
				if e:
					events.append(f"{sym}: Entry filled at ₹{trade['entry_price']:.2f}")
				if x:
					events.append(
						f"{sym}: Exited on entry day — {trade['exit_reason']}"
					)
				if trade["status"] == "entry_missed":
					events.append(f"{sym}: Entry missed — dropped from batch")
			else:
				x = self._process_active(trade, ohlcv, trade_date, td)
				exits += x
				if x:
					events.append(
						f"{sym}: Exited — {trade['exit_reason']} "
						f"P&L {trade['pnl_pct']:+.2f}%"
					)

			self._log_daily_pnl(trade, td, ohlcv)

		self._save_state()
		return {
			"trade_date": td,
			"updated": updated,
			"entries": entries,
			"exits": exits,
			"events": events,
		}

	# ── pending entry processing ──────────────────────────────────────────
	def _process_pending(
		self,
		trade: Dict[str, Any],
		ohlcv: Dict[str, float],
		trade_date: date,
		td: str,
	) -> tuple[int, int]:
		"""Handle a pending-entry trade.  Returns (entries, exits)."""
		batch_date = _parse(trade["batch_date"])
		if trade_date <= batch_date:
			return 0, 0  # don't attempt entry on signal day

		trade["entry_attempts"] += 1
		ep = trade["entry_price"]
		o, h, l, c, v = (
			ohlcv["Open"],
			ohlcv["High"],
			ohlcv["Low"],
			ohlcv["Close"],
			ohlcv["Volume"],
		)

		if l <= ep <= h:
			# ── entry filled ─────────────────────────────────────────
			trade["status"] = "active"
			trade["entry_date"] = td
			trade["days_held"] = 1
			trade["last_close"] = c
			pnl = ((c - ep) / ep) * 100
			trade["pnl_pct"] = round(pnl, 2)
			trade["pnl_absolute"] = round((c - ep) * trade["num_shares"], 2)

			log = (
				f"[{td}] ENTRY_FILLED (attempt {trade['entry_attempts']}"
				f"/{ENTRY_WINDOW_DAYS}): "
				f"Entry ₹{ep:.2f} within range "
				f"[₹{l:.2f}–₹{h:.2f}]. "
				f"O ₹{o:.2f} H ₹{h:.2f} L ₹{l:.2f} C ₹{c:.2f} "
				f"Vol {v:,.0f}. Unrealized {pnl:+.2f}%."
			)

			# check SL / targets on entry day itself
			ev = self._check_sl_and_targets(trade, ohlcv, td)
			if ev:
				log += f" | {ev}"
			trade["action_log"].append(log)

			return 1, (1 if trade["status"] == "completed" else 0)

		# ── entry NOT filled ─────────────────────────────────────────
		log = (
			f"[{td}] ENTRY_PENDING (attempt {trade['entry_attempts']}"
			f"/{ENTRY_WINDOW_DAYS}): "
			f"Entry ₹{ep:.2f} NOT in range "
			f"[₹{l:.2f}–₹{h:.2f}]. "
			f"O ₹{o:.2f} H ₹{h:.2f} L ₹{l:.2f} C ₹{c:.2f} Vol {v:,.0f}."
		)
		if trade["entry_attempts"] >= ENTRY_WINDOW_DAYS:
			trade["status"] = "entry_missed"
			trade["exit_reason"] = "entry_not_available"
			log += (
				f" DROPPED — entry not available within "
				f"{ENTRY_WINDOW_DAYS} trading days. Capital kept as cash."
			)
		trade["action_log"].append(log)
		return 0, 0

	# ── active trade processing ───────────────────────────────────────────
	def _process_active(
		self,
		trade: Dict[str, Any],
		ohlcv: Dict[str, float],
		trade_date: date,
		td: str,
	) -> int:
		"""Handle an active/t1_hit/t2_hit trade.  Returns 1 if exited."""
		trade["days_held"] += 1
		prev_close = trade["last_close"] or trade["entry_price"]
		ep = trade["entry_price"]
		o, h, l, c, v = (
			ohlcv["Open"],
			ohlcv["High"],
			ohlcv["Low"],
			ohlcv["Close"],
			ohlcv["Volume"],
		)
		trade["last_close"] = c

		unrealized = ((c - ep) / ep) * 100
		daily_chg = ((c - prev_close) / prev_close) * 100 if prev_close else 0
		trade["pnl_pct"] = round(unrealized, 2)
		trade["pnl_absolute"] = round((c - ep) * trade["num_shares"], 2)

		parts = [
			f"[{td}] Day {trade['days_held']}/{MAX_HOLD_DAYS}: "
			f"O ₹{o:.2f} H ₹{h:.2f} L ₹{l:.2f} C ₹{c:.2f} Vol {v:,.0f}. "
			f"Unrealized {unrealized:+.2f}% (₹{trade['pnl_absolute']:+,.2f}). "
			f"Daily chg {daily_chg:+.2f}%."
		]

		# SL / targets
		ev = self._check_sl_and_targets(trade, ohlcv, td)
		if ev:
			parts.append(ev)

		# timeout
		if trade["status"] in ("active", "t1_hit", "t2_hit"):
			if trade["days_held"] >= MAX_HOLD_DAYS:
				trade["status"] = "completed"
				trade["exit_date"] = td
				trade["exit_price"] = round(c, 2)
				trade["exit_reason"] = "max_hold_exit"
				fpnl = ((c - ep) / ep) * 100
				trade["pnl_pct"] = round(fpnl, 2)
				trade["pnl_absolute"] = round((c - ep) * trade["num_shares"], 2)
				parts.append(
					f"MAX_HOLD_EXIT: 21 trading days reached. "
					f"Exited at close ₹{c:.2f}. "
					f"Final P&L {fpnl:+.2f}% (₹{trade['pnl_absolute']:+,.2f})."
				)
			else:
				remaining = MAX_HOLD_DAYS - trade["days_held"]
				next_tgt = (
					f"T1 ₹{trade['target_1']}"
					if not trade["t1_hit"]
					else f"T2 ₹{trade['target_2']}"
					if not trade["t2_hit"]
					else f"T3 ₹{trade['target_3']}"
				)
				parts.append(
					f"Status: {trade['status'].upper()}. "
					f"SL ₹{trade['current_sl']:.2f}. "
					f"Next target: {next_tgt}. "
					f"Days remaining: {remaining}."
				)

		trade["action_log"].append(" | ".join(parts))
		return 1 if trade["status"] == "completed" else 0

	# ── SL and target checker ─────────────────────────────────────────────
	def _check_sl_and_targets(
		self,
		trade: Dict[str, Any],
		ohlcv: Dict[str, float],
		td: str,
	) -> str:
		"""Check stop-loss and targets.  Mutates trade state.

		Returns a human-readable event string (may be empty).
		"""
		h = ohlcv["High"]
		l = ohlcv["Low"]
		c = ohlcv["Close"]
		ep = trade["entry_price"]
		sl = trade["current_sl"]
		msgs: List[str] = []

		# ── SL check first (conservative) ────────────────────────────
		if l <= sl:
			exit_px = sl
			if trade["trailing_sl_active"]:
				reason = "trailing_sl_hit"
				msgs.append(
					f"TRAILING_SL_HIT: Low ₹{l:.2f} breached trailing SL "
					f"₹{sl:.2f}. Exited at ₹{exit_px:.2f}."
				)
			else:
				reason = "stop_loss_hit"
				msgs.append(
					f"SL1_HIT: Low ₹{l:.2f} breached SL1 ₹{sl:.2f}. "
					f"Exited at ₹{exit_px:.2f}. Cash preserved."
				)
			trade["status"] = "completed"
			trade["exit_date"] = td
			trade["exit_price"] = round(exit_px, 2)
			trade["exit_reason"] = reason
			fpnl = ((exit_px - ep) / ep) * 100
			trade["pnl_pct"] = round(fpnl, 2)
			trade["pnl_absolute"] = round((exit_px - ep) * trade["num_shares"], 2)
			msgs.append(f"Final P&L {fpnl:+.2f}% (₹{trade['pnl_absolute']:+,.2f}).")
			return " | ".join(msgs)

		# ── T3 (exit trigger) ────────────────────────────────────────
		if h >= trade["target_3"]:
			if not trade["t1_hit"]:
				trade["t1_hit"] = True
				trade["t1_hit_date"] = td
			if not trade["t2_hit"]:
				trade["t2_hit"] = True
				trade["t2_hit_date"] = td
			trade["t3_hit"] = True
			trade["t3_hit_date"] = td
			exit_px = trade["target_3"]
			trade["status"] = "completed"
			trade["exit_date"] = td
			trade["exit_price"] = round(exit_px, 2)
			trade["exit_reason"] = "target_3_achieved"
			fpnl = ((exit_px - ep) / ep) * 100
			trade["pnl_pct"] = round(fpnl, 2)
			trade["pnl_absolute"] = round((exit_px - ep) * trade["num_shares"], 2)
			msgs.append(
				f"T3_HIT: High ₹{h:.2f} reached T3 ₹{trade['target_3']:.2f}. "
				f"All targets hit. Exited at T3 ₹{exit_px:.2f}. "
				f"Final P&L {fpnl:+.2f}% (₹{trade['pnl_absolute']:+,.2f})."
			)
			return " | ".join(msgs)

		# ── T2 (no exit — move trailing SL to T2) ────────────────────
		if h >= trade["target_2"] and not trade["t2_hit"]:
			trade["t2_hit"] = True
			trade["t2_hit_date"] = td
			if not trade["t1_hit"]:
				trade["t1_hit"] = True
				trade["t1_hit_date"] = td
				trade["trailing_sl_active"] = True
			old_sl = trade["current_sl"]
			trade["current_sl"] = round(trade["target_2"], 2)
			trade["status"] = "t2_hit"
			msgs.append(
				f"T2_HIT: High ₹{h:.2f} reached T2 ₹{trade['target_2']:.2f}. "
				f"Trailing SL ₹{old_sl:.2f} → ₹{trade['current_sl']:.2f} (T2). "
				f"Target remaining: T3 ₹{trade['target_3']:.2f}."
			)
			return " | ".join(msgs)

		# ── T1 (no exit — activate trailing SL at T1) ────────────────
		if h >= trade["target_1"] and not trade["t1_hit"]:
			trade["t1_hit"] = True
			trade["t1_hit_date"] = td
			trade["trailing_sl_active"] = True
			old_sl = trade["current_sl"]
			trade["current_sl"] = round(trade["target_1"], 2)
			trade["status"] = "t1_hit"
			msgs.append(
				f"T1_HIT: High ₹{h:.2f} reached T1 ₹{trade['target_1']:.2f}. "
				f"Trailing SL activated. SL ₹{old_sl:.2f} → ₹{trade['current_sl']:.2f} (T1). "
				f"Targets remaining: T2 ₹{trade['target_2']:.2f}, T3 ₹{trade['target_3']:.2f}."
			)
			return " | ".join(msgs)

		# ── nothing triggered ────────────────────────────────────────
		next_tgt = (
			f"T1 ₹{trade['target_1']:.2f}"
			if not trade["t1_hit"]
			else f"T2 ₹{trade['target_2']:.2f}"
			if not trade["t2_hit"]
			else f"T3 ₹{trade['target_3']:.2f}"
		)
		msgs.append(
			f"HOLDING: SL ₹{sl:.2f} safe (low ₹{l:.2f}). "
			f"Next target {next_tgt}."
		)
		return " | ".join(msgs)

	# ── daily P&L log ─────────────────────────────────────────────────────
	def _log_daily_pnl(
		self,
		trade: Dict[str, Any],
		td: str,
		ohlcv: Dict[str, float],
	) -> None:
		ep = trade["entry_price"]
		c = ohlcv["Close"]

		if trade["status"] in ("pending_entry", "entry_missed"):
			unrealized = 0.0
		else:
			unrealized = ((c - ep) / ep) * 100 if ep > 0 else 0.0

		# determine event label
		event = "HOLDING"
		if trade["status"] == "pending_entry":
			event = f"ENTRY_PENDING ({trade['entry_attempts']}/{ENTRY_WINDOW_DAYS})"
		elif trade["status"] == "entry_missed":
			event = "ENTRY_MISSED"
		elif trade["status"] == "completed":
			event = (trade.get("exit_reason") or "COMPLETED").upper()
		elif trade.get("t2_hit_date") == td:
			event = "T2_HIT"
		elif trade.get("t1_hit_date") == td:
			event = "T1_HIT"
		elif trade.get("entry_date") == td:
			event = "ENTRY_FILLED"

		self.state["daily_pnl_log"].append(
			{
				"date": td,
				"trade_id": trade["trade_id"],
				"batch_date": trade["batch_date"],
				"symbol": trade["symbol"],
				"status": trade["status"],
				"entry_price": trade["entry_price"],
				"day_open": round(ohlcv["Open"], 2),
				"day_high": round(ohlcv["High"], 2),
				"day_low": round(ohlcv["Low"], 2),
				"day_close": round(ohlcv["Close"], 2),
				"day_volume": int(ohlcv["Volume"]),
				"unrealized_pnl_pct": round(unrealized, 2),
				"pnl_absolute": round(trade.get("pnl_absolute", 0), 2),
				"current_sl": trade["current_sl"],
				"trailing_sl_active": trade["trailing_sl_active"],
				"days_held": trade["days_held"],
				"event": event,
			}
		)

	# ── accessors ─────────────────────────────────────────────────────────
	def get_active_trades(self) -> List[Dict[str, Any]]:
		return [
			t
			for t in self.state["trades"]
			if t["status"] in ("pending_entry", "active", "t1_hit", "t2_hit")
		]

	def get_completed_trades(self) -> List[Dict[str, Any]]:
		return [
			t
			for t in self.state["trades"]
			if t["status"] in ("completed", "entry_missed")
		]

	# ── CSV exports ───────────────────────────────────────────────────────
	def export_all(self) -> None:
		"""Write all output CSVs including per-batch files."""
		self._export_per_batch_csvs()
		self._export_active_trades()
		self._export_completed_trades()
		self._export_daily_pnl()
		self._export_batch_summary()
		self._export_overall_summary()

	def _export_active_trades(self) -> None:
		rows = []
		for t in self.get_active_trades():
			rows.append(
				{
					"batch_date": t["batch_date"],
					"symbol": t["symbol"],
					"signal_date": t["signal_date"],
					"pattern": t["pattern"],
					"entry_price": t["entry_price"],
					"entry_date": t.get("entry_date") or "",
					"entry_attempts": t["entry_attempts"],
					"status": t["status"],
					"current_sl": t["current_sl"],
					"trailing_sl_active": t["trailing_sl_active"],
					"stop_loss_1": t["stop_loss_1"],
					"target_1": t["target_1"],
					"target_2": t["target_2"],
					"target_3": t["target_3"],
					"t1_hit": t["t1_hit"],
					"t1_hit_date": t.get("t1_hit_date") or "",
					"t2_hit": t["t2_hit"],
					"t2_hit_date": t.get("t2_hit_date") or "",
					"last_close": t.get("last_close") or "",
					"unrealized_pnl_pct": t["pnl_pct"],
					"pnl_absolute": t.get("pnl_absolute", 0),
					"days_held": t["days_held"],
					"days_remaining": MAX_HOLD_DAYS - t["days_held"],
					"capital_allocated": t["capital_allocated"],
					"num_shares": t["num_shares"],
					"action_log": " || ".join(t.get("action_log", [])),
				}
			)
		pd.DataFrame(rows).to_csv(
			os.path.join(self.output_dir, "active_trades.csv"), index=False
		)

	def _export_completed_trades(self) -> None:
		rows = []
		for t in self.get_completed_trades():
			rows.append(
				{
					"batch_date": t["batch_date"],
					"symbol": t["symbol"],
					"signal_date": t["signal_date"],
					"pattern": t["pattern"],
					"entry_price": t["entry_price"],
					"entry_date": t.get("entry_date") or "",
					"exit_date": t.get("exit_date") or "",
					"exit_price": t.get("exit_price") or "",
					"exit_reason": t.get("exit_reason") or "",
					"status": t["status"],
					"stop_loss_1": t["stop_loss_1"],
					"target_1": t["target_1"],
					"target_2": t["target_2"],
					"target_3": t["target_3"],
					"t1_hit": t["t1_hit"],
					"t1_hit_date": t.get("t1_hit_date") or "",
					"t2_hit": t["t2_hit"],
					"t2_hit_date": t.get("t2_hit_date") or "",
					"t3_hit": t["t3_hit"],
					"t3_hit_date": t.get("t3_hit_date") or "",
					"pnl_pct": t["pnl_pct"],
					"pnl_absolute": t.get("pnl_absolute", 0),
					"days_held": t["days_held"],
					"capital_allocated": t["capital_allocated"],
					"num_shares": t["num_shares"],
					"trailing_sl_was_active": t["trailing_sl_active"],
					"action_log": " || ".join(t.get("action_log", [])),
				}
			)
		pd.DataFrame(rows).to_csv(
			os.path.join(self.output_dir, "completed_trades.csv"), index=False
		)

	def _export_daily_pnl(self) -> None:
		pd.DataFrame(self.state["daily_pnl_log"]).to_csv(
			os.path.join(self.output_dir, "daily_pnl_log.csv"), index=False
		)

	def _export_batch_summary(self) -> None:
		trades = self.state["trades"]
		if not trades:
			pd.DataFrame().to_csv(
				os.path.join(self.output_dir, "batch_summary.csv"), index=False
			)
			return

		buckets: Dict[str, List[Dict]] = {}
		for t in trades:
			buckets.setdefault(t["batch_date"], []).append(t)

		rows = []
		for bd, bt in sorted(buckets.items()):
			entered = [t for t in bt if t["status"] != "entry_missed"]
			completed = [t for t in bt if t["status"] == "completed"]
			active = [
				t for t in bt if t["status"] in ("active", "t1_hit", "t2_hit")
			]
			pending = [t for t in bt if t["status"] == "pending_entry"]
			missed = [t for t in bt if t["status"] == "entry_missed"]

			sl_cnt = len(
				[
					t
					for t in completed
					if t.get("exit_reason") in ("stop_loss_hit", "trailing_sl_hit")
				]
			)
			t3_cnt = len(
				[
					t
					for t in completed
					if t.get("exit_reason") == "target_3_achieved"
				]
			)
			mh_cnt = len(
				[
					t
					for t in completed
					if t.get("exit_reason") == "max_hold_exit"
				]
			)

			# P&L only from entered trades
			pnl_trades = entered
			avg_pnl = (
				sum(t["pnl_pct"] for t in pnl_trades) / len(pnl_trades)
				if pnl_trades
				else 0
			)
			total_abs = sum(t.get("pnl_absolute", 0) for t in pnl_trades)

			rows.append(
				{
					"batch_date": bd,
					"total_signals": len(bt),
					"entries_filled": len(entered),
					"entries_missed": len(missed),
					"active": len(active),
					"pending": len(pending),
					"completed": len(completed),
					"sl_hit": sl_cnt,
					"t3_achieved": t3_cnt,
					"max_hold_exit": mh_cnt,
					"avg_pnl_pct": round(avg_pnl, 2),
					"total_pnl_absolute": round(total_abs, 2),
					"capital_deployed": round(
						sum(t["capital_allocated"] for t in entered), 2
					),
				}
			)

		pd.DataFrame(rows).to_csv(
			os.path.join(self.output_dir, "batch_summary.csv"), index=False
		)

	def _export_per_batch_csvs(self) -> None:
		"""Write one CSV per batch_date with full trade details.

		Each file is overwritten on every run so previous-day batch CSVs
		reflect the latest prices, P&L, status, and action_log.
		"""
		trades = self.state["trades"]
		if not trades:
			return

		buckets: Dict[str, List[Dict]] = {}
		for t in trades:
			buckets.setdefault(t["batch_date"], []).append(t)

		for bd, batch_trades in buckets.items():
			# derive filename: batch_03_06_2026.csv
			bd_date = _parse(bd)
			filename = f"batch_{bd_date.strftime('%d_%m_%Y')}.csv"

			rows = []
			for t in batch_trades:
				rows.append(
					{
						"batch_date": t["batch_date"],
						"symbol": t["symbol"],
						"signal_date": t["signal_date"],
						"pattern": t["pattern"],
						"entry_level_name": t.get("entry_level_name", ""),
						"entry_price": t["entry_price"],
						"entry_date": t.get("entry_date") or "",
						"entry_attempts": t["entry_attempts"],
						"status": t["status"],
						# levels
						"stop_loss_1": t["stop_loss_1"],
						"stop_loss_2": t["stop_loss_2"],
						"stop_loss_3": t["stop_loss_3"],
						"current_sl": t["current_sl"],
						"trailing_sl_active": t["trailing_sl_active"],
						"target_1": t["target_1"],
						"target_2": t["target_2"],
						"target_3": t["target_3"],
						"swing_low": t["swing_low"],
						"swing_high": t["swing_high"],
						"fib_382": t["fib_382"],
						"fib_50": t["fib_50"],
						"fib_618": t["fib_618"],
						# target hit flags
						"t1_hit": t["t1_hit"],
						"t1_hit_date": t.get("t1_hit_date") or "",
						"t2_hit": t["t2_hit"],
						"t2_hit_date": t.get("t2_hit_date") or "",
						"t3_hit": t["t3_hit"],
						"t3_hit_date": t.get("t3_hit_date") or "",
						# exit info
						"exit_date": t.get("exit_date") or "",
						"exit_price": t.get("exit_price") or "",
						"exit_reason": t.get("exit_reason") or "",
						# latest price & P&L
						"last_close": t.get("last_close") or "",
						"pnl_pct": t["pnl_pct"],
						"pnl_absolute": t.get("pnl_absolute", 0),
						"days_held": t["days_held"],
						"days_remaining": (
							MAX_HOLD_DAYS - t["days_held"]
							if t["status"] not in ("completed", "entry_missed")
							else 0
						),
						# capital
						"capital_allocated": t["capital_allocated"],
						"num_shares": t["num_shares"],
						"batch_stock_count": t["batch_stock_count"],
						"signal_close": t["signal_close"],
						# full action log
						"action_log": " || ".join(
							t.get("action_log", [])
						),
					}
				)

			pd.DataFrame(rows).to_csv(
				os.path.join(self.output_dir, filename), index=False
			)

	def _export_overall_summary(self) -> None:
		trades = self.state["trades"]
		entered = [t for t in trades if t["status"] != "entry_missed"]
		completed = [t for t in trades if t["status"] == "completed"]

		winners = [t for t in completed if t["pnl_pct"] > 0]
		losers = [t for t in completed if t["pnl_pct"] < 0]
		breakeven = [t for t in completed if t["pnl_pct"] == 0]

		avg_win = (
			sum(t["pnl_pct"] for t in winners) / len(winners) if winners else 0
		)
		avg_loss = (
			sum(t["pnl_pct"] for t in losers) / len(losers) if losers else 0
		)
		total_abs = sum(t.get("pnl_absolute", 0) for t in completed)
		total_cap = sum(t["capital_allocated"] for t in entered)

		summary = {
			"total_signals": len(trades),
			"entries_filled": len(entered),
			"entries_missed": len(trades) - len(entered),
			"total_completed": len(completed),
			"total_active": len(
				[
					t
					for t in trades
					if t["status"]
					in ("active", "t1_hit", "t2_hit", "pending_entry")
				]
			),
			"winners": len(winners),
			"losers": len(losers),
			"breakeven": len(breakeven),
			"win_rate_pct": (
				round(len(winners) / len(completed) * 100, 2)
				if completed
				else 0
			),
			"avg_winner_pnl_pct": round(avg_win, 2),
			"avg_loser_pnl_pct": round(avg_loss, 2),
			"total_pnl_absolute": round(total_abs, 2),
			"total_capital_deployed": round(total_cap, 2),
			"return_on_capital_pct": (
				round(total_abs / total_cap * 100, 2) if total_cap else 0
			),
			"sl_exits": len(
				[t for t in completed if t.get("exit_reason") == "stop_loss_hit"]
			),
			"trailing_sl_exits": len(
				[
					t
					for t in completed
					if t.get("exit_reason") == "trailing_sl_hit"
				]
			),
			"t3_achieved_exits": len(
				[
					t
					for t in completed
					if t.get("exit_reason") == "target_3_achieved"
				]
			),
			"max_hold_exits": len(
				[
					t
					for t in completed
					if t.get("exit_reason") == "max_hold_exit"
				]
			),
			"avg_days_held": (
				round(sum(t["days_held"] for t in completed) / len(completed), 1)
				if completed
				else 0
			),
		}

		pd.DataFrame([summary]).to_csv(
			os.path.join(self.output_dir, "overall_summary.csv"), index=False
		)

	# ── dashboard ─────────────────────────────────────────────────────────
	def print_dashboard(self) -> None:
		trades = self.state["trades"]
		active = self.get_active_trades()
		completed = self.get_completed_trades()

		print("\n" + "=" * 80)
		print("  FORWARD TEST DASHBOARD")
		print("=" * 80)

		print(f"\n  Total Signals : {len(trades)}")
		print(
			f"  Active/Pending: "
			f"{len([t for t in active if t['status'] != 'pending_entry'])} active, "
			f"{len([t for t in active if t['status'] == 'pending_entry'])} pending"
		)
		cmp = [t for t in completed if t["status"] == "completed"]
		missed = [t for t in completed if t["status"] == "entry_missed"]
		print(f"  Completed     : {len(cmp)}")
		print(f"  Entry Missed  : {len(missed)}")

		# ── active positions ─────────────────────────────────────────
		if active:
			print(f"\n  {'─' * 76}")
			print("  ACTIVE POSITIONS:")
			print(f"  {'─' * 76}")
			emojis = {
				"pending_entry": "⏳",
				"active": "📊",
				"t1_hit": "🎯",
				"t2_hit": "🎯🎯",
			}
			for t in active:
				em = emojis.get(t["status"], "")
				pnl = (
					f"{t['pnl_pct']:+.2f}%"
					if t["status"] != "pending_entry"
					else "N/A"
				)
				print(
					f"  {em} {t['symbol']:12s} | Batch {t['batch_date']} | "
					f"Entry ₹{t['entry_price']:>8.2f} | "
					f"SL ₹{t['current_sl']:>8.2f} | "
					f"P&L {pnl:>8s} | "
					f"Day {t['days_held']}/{MAX_HOLD_DAYS} | "
					f"{t['status']}"
				)

		# ── recent completed ─────────────────────────────────────────
		if cmp:
			print(f"\n  {'─' * 76}")
			print("  COMPLETED TRADES (last 10):")
			print(f"  {'─' * 76}")
			reason_emoji = {
				"stop_loss_hit": "🔴",
				"trailing_sl_hit": "🟡",
				"target_3_achieved": "🟢",
				"max_hold_exit": "⏰",
			}
			for t in cmp[-10:]:
				em = reason_emoji.get(t.get("exit_reason", ""), "")
				print(
					f"  {em} {t['symbol']:12s} | Batch {t['batch_date']} | "
					f"Entry ₹{t['entry_price']:>8.2f} | "
					f"Exit ₹{t.get('exit_price', 0):>8.2f} | "
					f"P&L {t['pnl_pct']:+.2f}% "
					f"(₹{t.get('pnl_absolute', 0):+,.2f}) | "
					f"Days {t['days_held']} | "
					f"{t.get('exit_reason', '')}"
				)

		# ── overall P&L ──────────────────────────────────────────────
		if cmp:
			total_pnl = sum(t.get("pnl_absolute", 0) for t in cmp)
			avg_pnl = sum(t["pnl_pct"] for t in cmp) / len(cmp)
			w = len([t for t in cmp if t["pnl_pct"] > 0])
			wr = w / len(cmp) * 100
			print(f"\n  {'─' * 76}")
			print(
				f"  OVERALL: Total P&L ₹{total_pnl:+,.2f} | "
				f"Avg {avg_pnl:+.2f}% | "
				f"Win Rate {wr:.1f}% ({w}/{len(cmp)})"
			)

		print("=" * 80 + "\n")
