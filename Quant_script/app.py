"""Orchestrator for daily forward-testing workflow.

Typical daily usage
-------------------
    python app.py                      # auto-detect today, scan + update
    python app.py --date 23/06/2026    # specific date
    python app.py --update-only        # skip scanning, just update trades
    python app.py --backfill           # replay all existing signal CSVs
    python app.py --skip-scan          # use existing CSV, don't run scanner
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from typing import List, Tuple

import pandas as pd

# ── imports from project ──────────────────────────────────────────────────────
from Fibonacci_Retracement_Candlestick_Volume import (
	_download_bhavcopy_for_date,
	build_output_name,
	get_output_dir as get_scanner_output_dir,
	load_sec_list_symbols,
	run_screen,
)
from forward_test import ForwardTestTracker, _fmt


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description="Forward-Test Orchestrator for Fibonacci Retracement strategy"
	)
	p.add_argument(
		"--date",
		default="",
		help="Anchor date in dd/mm/yyyy (default: today)",
	)
	p.add_argument(
		"--update-only",
		action="store_true",
		help="Only update existing trades — don't scan for new signals",
	)
	p.add_argument(
		"--backfill",
		action="store_true",
		help="Process all existing signal CSVs chronologically (resets state)",
	)
	p.add_argument(
		"--capital",
		type=float,
		default=100_000,
		help="Capital per batch in ₹ (default: 1,00,000)",
	)
	p.add_argument(
		"--skip-scan",
		action="store_true",
		help="Don't run the scanner — use an existing signal CSV if available",
	)
	return p.parse_args()


def discover_signal_csvs(scanner_dir: str) -> List[Tuple[date, str]]:
	"""Find all existing signal CSVs and return sorted (date, path) pairs."""
	prefix = "fibonacci_retracement_candlestick_volume_tomorrow_"
	pairs: List[Tuple[date, str]] = []
	if not os.path.isdir(scanner_dir):
		return pairs
	for fname in os.listdir(scanner_dir):
		if fname.startswith(prefix) and fname.endswith(".csv"):
			stem = fname[len(prefix) : -4]
			try:
				d = datetime.strptime(stem, "%d_%m_%Y").date()
				pairs.append((d, os.path.join(scanner_dir, fname)))
			except ValueError:
				continue
	pairs.sort(key=lambda x: x[0])
	return pairs


def run_scanner(as_of_date: date) -> str:
	"""Execute the Fibonacci scanner and return the output CSV path."""
	symbols = load_sec_list_symbols()
	out_dir = get_scanner_output_dir()

	print(f"  Scanning {len(symbols)} symbols for {as_of_date} …")
	results = run_screen(
		symbols,
		as_of_date=as_of_date,
		swing_lookback=60,
		near_level_pct=1.2,
		volume_lookback=20,
		volume_multiplier=1.2,
		min_swing_pct=6.0,
		export_charts=False,
		charts_dir=os.path.join(out_dir, "charts"),
		chart_lookback_bars=120,
		max_workers=max((os.cpu_count() or 2) - 1, 1),
		verbose=False,
	)

	csv_path = os.path.join(out_dir, build_output_name(as_of_date))
	results.to_csv(csv_path, index=False)
	print(f"  Scanner found {len(results)} signals → {csv_path}")
	return csv_path


def collect_trading_days(start: date, end: date) -> List[date]:
	"""Return dates that have bhavcopy data (i.e. actual trading days)."""
	days: List[date] = []
	cur = start
	while cur <= end:
		if cur.weekday() < 5:  # skip Sat/Sun
			df = _download_bhavcopy_for_date(cur)
			if df is not None and not df.empty:
				days.append(cur)
		cur += timedelta(days=1)
	return days


# ──────────────────────────────────────────────────────────────────────────────
# Modes
# ──────────────────────────────────────────────────────────────────────────────
def mode_backfill(tracker: ForwardTestTracker, target_date: date) -> None:
	"""Replay every existing signal CSV in chronological order."""
	scanner_dir = get_scanner_output_dir()
	csvs = discover_signal_csvs(scanner_dir)
	if not csvs:
		print("No signal CSVs found for backfill.")
		return

	start = csvs[0][0]
	end = target_date
	signal_map = dict(csvs)

	print(f"\n{'═' * 70}")
	print(f"  BACKFILL: {start} → {end}")
	print(f"  Signal CSVs found: {len(csvs)}")
	print(f"{'═' * 70}")

	# Reset state for a clean replay
	tracker.reset_state()

	print("  Discovering trading days …")
	trading_days = collect_trading_days(start, end)
	print(f"  Trading days: {len(trading_days)}\n")

	for td in trading_days:
		label = _fmt(td)

		# ingest batch if a CSV exists for this date
		if td in signal_map:
			csv_path = signal_map[td]
			added = tracker.ingest_new_batch(csv_path, td)
			print(
				f"  [{label}] 📥 Ingested {added} signals "
				f"from {os.path.basename(csv_path)}"
			)

		# update all trades with today's market data
		result = tracker.update_daily(td)
		u = result.get("updated", 0)
		e = result.get("entries", 0)
		x = result.get("exits", 0)
		if u:
			print(
				f"  [{label}] 🔄 Updated {u} trades  "
				f"(entries +{e}, exits +{x})"
			)
			for ev in result.get("events", []):
				print(f"           → {ev}")

	tracker.export_all()
	tracker.print_dashboard()
	print(f"  Backfill complete.  Output → {tracker.output_dir}\n")


def mode_update_only(tracker: ForwardTestTracker, target_date: date) -> None:
	"""Just update existing trades — no new signal ingestion."""
	print(f"\n  UPDATE ONLY: {_fmt(target_date)}")
	result = tracker.update_daily(target_date)
	print(f"  Updated {result.get('updated', 0)} trades.")
	for ev in result.get("events", []):
		print(f"  → {ev}")
	if result.get("note"):
		print(f"  ⚠ {result['note']}")
	tracker.export_all()
	tracker.print_dashboard()


def mode_daily(
	tracker: ForwardTestTracker,
	target_date: date,
	skip_scan: bool,
) -> None:
	"""Normal daily run: scan (optional) → ingest → update → export."""
	scanner_dir = get_scanner_output_dir()
	csv_path = os.path.join(scanner_dir, build_output_name(target_date))

	print(f"\n  DAILY RUN: {_fmt(target_date)}")

	# ── scan / locate signal CSV ─────────────────────────────────────
	if not skip_scan and not os.path.exists(csv_path):
		csv_path = run_scanner(target_date)
	elif os.path.exists(csv_path):
		print(f"  Using existing signal CSV: {os.path.basename(csv_path)}")
	else:
		print(f"  No signal CSV found and --skip-scan is set.")
		csv_path = ""

	# ── ingest new batch ─────────────────────────────────────────────
	if csv_path and os.path.exists(csv_path):
		added = tracker.ingest_new_batch(csv_path, target_date)
		print(f"  📥 Ingested {added} new signals.")
	else:
		print("  ⚠ No signal CSV to ingest.")

	# ── update all trades ────────────────────────────────────────────
	result = tracker.update_daily(target_date)
	u = result.get("updated", 0)
	e = result.get("entries", 0)
	x = result.get("exits", 0)
	print(f"  🔄 Updated {u} trades  (entries +{e}, exits +{x})")
	for ev in result.get("events", []):
		print(f"  → {ev}")
	if result.get("note"):
		print(f"  ⚠ {result['note']}")

	tracker.export_all()
	tracker.print_dashboard()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
	args = parse_args()

	if args.date.strip():
		target_date = datetime.strptime(args.date.strip(), "%d/%m/%Y").date()
	else:
		user_input = input("Enter anchor date (dd/mm/yyyy): ").strip()
		if not user_input:
			raise ValueError("Anchor date is required.")
		target_date = datetime.strptime(user_input, "%d/%m/%Y").date()

	tracker = ForwardTestTracker(capital_per_batch=args.capital)

	if args.backfill:
		mode_backfill(tracker, target_date)
	elif args.update_only:
		mode_update_only(tracker, target_date)
	else:
		mode_daily(tracker, target_date, skip_scan=args.skip_scan)

	print(f"  Output directory: {tracker.output_dir}")


if __name__ == "__main__":
	main()
