from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from Fibonacci_Retracement_Candlestick_Volume import (
	_download_bhavcopy_for_date,
	_find_column,
	_parse_ddmmyyyy,
	fibonacci_bounce_tomorrow_signal,
)


def parse_symbols(value: str) -> List[str]:
	return [s.strip().upper() for s in value.split(",") if s.strip()]


def load_symbols_from_csv(csv_path: str) -> List[str]:
	df = pd.read_csv(csv_path)
	cols = {str(c).strip().lower(): c for c in df.columns}
	if "symbol" not in cols:
		raise ValueError("Input CSV must have a 'symbol' column.")
	series = df[cols["symbol"]].astype(str).str.strip().str.upper()
	series = series[series != ""]
	return sorted(set(series.tolist()))


def iter_dates(start_date: date, end_date: date):
	current = start_date
	while current <= end_date:
		yield current
		current += timedelta(days=1)


def fetch_daily_history_nse(symbols: List[str], start_date: date, end_date: date) -> Dict[str, pd.DataFrame]:
	needed = ["Open", "High", "Low", "Close", "Volume"]
	symbols = [s.strip().upper() for s in symbols if s.strip()]
	symbol_set = set(symbols)
	rows_by_symbol: Dict[str, List[Dict[str, object]]] = {sym: [] for sym in symbols}

	for day in iter_dates(start_date, end_date):
		df = _download_bhavcopy_for_date(day)
		if df is None or df.empty:
			continue

		symbol_col = _find_column(df.columns, ["SYMBOL"])
		series_col = _find_column(df.columns, ["SERIES"])
		open_col = _find_column(df.columns, ["OPEN_PRICE", "OPEN"])
		high_col = _find_column(df.columns, ["HIGH_PRICE", "HIGH"])
		low_col = _find_column(df.columns, ["LOW_PRICE", "LOW"])
		close_col = _find_column(df.columns, ["CLOSE_PRICE", "CLOSE", "CLOSE_PRICE_"])
		volume_col = _find_column(df.columns, ["TOTTRDQTY", "TTL_TRD_QNTY", "VOLUME", "TOTTRD_QTY"])
		date_col = _find_column(df.columns, ["DATE1", "DATE", "TIMESTAMP"])

		if not all([symbol_col, series_col, open_col, high_col, low_col, close_col, date_col]):
			continue

		base_cols = [symbol_col, series_col, open_col, high_col, low_col, close_col, date_col]
		if volume_col:
			base_cols.append(volume_col)

		work = df[base_cols].copy()
		rename_map = {
			symbol_col: "symbol",
			series_col: "series",
			open_col: "Open",
			high_col: "High",
			low_col: "Low",
			close_col: "Close",
			date_col: "Date",
		}
		if volume_col:
			rename_map[volume_col] = "Volume"
		work = work.rename(columns=rename_map)

		work["symbol"] = work["symbol"].astype(str).str.strip().str.upper()
		work["series"] = work["series"].astype(str).str.strip().str.upper()
		work = work[(work["series"] == "EQ") & (work["symbol"].isin(symbol_set))]
		if work.empty:
			continue

		work["Date"] = pd.to_datetime(work["Date"], errors="coerce", dayfirst=True).dt.normalize()
		if "Volume" not in work.columns:
			work["Volume"] = 0.0

		for col in needed:
			work[col] = pd.to_numeric(work[col], errors="coerce")

		work = work.dropna(subset=["Date", "Open", "High", "Low", "Close"])
		if work.empty:
			continue

		for sym, grp in work.groupby("symbol"):
			rows_by_symbol[sym].extend(
				grp[["Date", "Open", "High", "Low", "Close", "Volume"]].to_dict("records")
			)

	histories: Dict[str, pd.DataFrame] = {}
	for sym in symbols:
		rows = rows_by_symbol.get(sym, [])
		if not rows:
			histories[sym] = pd.DataFrame(columns=needed)
			continue

		histories[sym] = (
			pd.DataFrame(rows)
			.sort_values("Date")
			.drop_duplicates(subset=["Date"], keep="last")
			.set_index("Date")
		)

	return histories


def evaluate_trade(
	future_df: pd.DataFrame,
	entry_price: float,
	stop_loss_1: float,
	target_1: float,
	target_2: float,
	target_3: float,
	max_hold_days: int,
) -> Tuple[str, Optional[str], Optional[int], float]:
	"""Return outcome, outcome_date, hold_days, max_target_hit.

	Assumption: if both SL and target hit in same day, SL is assumed first (conservative).
	"""
	if future_df.empty:
		return "not_triggered", None, None, 0.0

	window = future_df.head(max_hold_days)
	trigger_idx = None

	for idx, row in window.iterrows():
		if float(row["Low"]) <= entry_price <= float(row["High"]):
			trigger_idx = idx
			break

	if trigger_idx is None:
		return "not_triggered", None, None, 0.0

	started = False
	max_target_hit = 0.0
	days = 0

	for idx, row in window.iterrows():
		if not started:
			if idx != trigger_idx:
				continue
			started = True

		days += 1
		day_low = float(row["Low"])
		day_high = float(row["High"])

		if day_low <= stop_loss_1:
			return "stop_loss_1_hit", idx.strftime("%d/%m/%Y"), days, max_target_hit

		if day_high >= target_1:
			max_target_hit = max(max_target_hit, 1.0)
		if day_high >= target_2:
			max_target_hit = max(max_target_hit, 2.0)
		if day_high >= target_3:
			max_target_hit = max(max_target_hit, 3.0)

		if max_target_hit >= 3.0:
			return "target_3_hit", idx.strftime("%d/%m/%Y"), days, max_target_hit
		if max_target_hit >= 2.0:
			return "target_2_hit", idx.strftime("%d/%m/%Y"), days, max_target_hit
		if max_target_hit >= 1.0:
			return "target_1_hit", idx.strftime("%d/%m/%Y"), days, max_target_hit

	return "timeout", window.index[-1].strftime("%d/%m/%Y"), days, max_target_hit


def backtest_symbol(
	symbol: str,
	daily: pd.DataFrame,
	analysis_start: pd.Timestamp,
	max_hold_days: int,
	swing_lookback: int,
	near_level_pct: float,
	volume_lookback: int,
	volume_multiplier: float,
	min_swing_pct: float,
) -> List[Dict[str, object]]:
	trades: List[Dict[str, object]] = []
	if daily.empty:
		return trades

	for i in range(len(daily)):
		signal_date = daily.index[i]
		if signal_date < analysis_start:
			continue

		history = daily.iloc[: i + 1]
		signal = fibonacci_bounce_tomorrow_signal(
			history,
			swing_lookback=swing_lookback,
			near_level_pct=near_level_pct,
			volume_lookback=volume_lookback,
			volume_multiplier=volume_multiplier,
			min_swing_pct=min_swing_pct,
		)
		if signal is None:
			continue

		future = daily.iloc[i + 1 :]
		outcome, outcome_date, hold_days, max_target_hit = evaluate_trade(
			future_df=future,
			entry_price=float(signal["entry_price"]),
			stop_loss_1=float(signal["stop_loss_1"]),
			target_1=float(signal["target_1"]),
			target_2=float(signal["target_2"]),
			target_3=float(signal["target_3"]),
			max_hold_days=max_hold_days,
		)

		trades.append(
			{
				"symbol": symbol,
				"signal_date": signal_date.strftime("%d/%m/%Y"),
				"entry_price": round(float(signal["entry_price"]), 2),
				"stop_loss_1": round(float(signal["stop_loss_1"]), 2),
				"target_1": round(float(signal["target_1"]), 2),
				"target_2": round(float(signal["target_2"]), 2),
				"target_3": round(float(signal["target_3"]), 2),
				"pattern": str(signal["pattern"]),
				"outcome": outcome,
				"outcome_date": outcome_date or "",
				"hold_days": hold_days if hold_days is not None else "",
				"max_target_hit": max_target_hit,
			}
		)

	return trades


def summarize_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
	if trades_df.empty:
		return pd.DataFrame(
			[
				{
					"scope": "overall",
					"signals": 0,
					"triggered": 0,
					"not_triggered": 0,
					"target_1_plus": 0,
					"target_2_plus": 0,
					"target_3_plus": 0,
					"stop_loss_1_hit": 0,
					"timeout": 0,
					"success_rate_t1_pct": 0.0,
				}
			]
		)

	def _one(group: pd.DataFrame, scope: str) -> Dict[str, object]:
		signals = len(group)
		triggered = int((group["outcome"] != "not_triggered").sum())
		not_triggered = int((group["outcome"] == "not_triggered").sum())
		t1 = int(group["outcome"].isin(["target_1_hit", "target_2_hit", "target_3_hit"]).sum())
		t2 = int(group["outcome"].isin(["target_2_hit", "target_3_hit"]).sum())
		t3 = int((group["outcome"] == "target_3_hit").sum())
		sl = int((group["outcome"] == "stop_loss_1_hit").sum())
		timeout = int((group["outcome"] == "timeout").sum())
		rate = round((t1 / triggered) * 100.0, 2) if triggered > 0 else 0.0

		return {
			"scope": scope,
			"signals": signals,
			"triggered": triggered,
			"not_triggered": not_triggered,
			"target_1_plus": t1,
			"target_2_plus": t2,
			"target_3_plus": t3,
			"stop_loss_1_hit": sl,
			"timeout": timeout,
			"success_rate_t1_pct": rate,
		}

	summary_rows = [_one(trades_df, "overall")]
	for sym, grp in trades_df.groupby("symbol"):
		summary_rows.append(_one(grp, sym))

	return pd.DataFrame(summary_rows)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Backtest Fibonacci Retracement + Candlestick + Volume strategy outcomes"
	)
	parser.add_argument(
		"--stocks",
		default="",
		help="Comma-separated stock symbols to backtest",
	)
	parser.add_argument(
		"--stocks-csv",
		default="",
		help="Optional CSV path with symbol column to take stock list",
	)
	parser.add_argument(
		"--as-of-date",
		default="",
		help="End date in dd/mm/yyyy format (default: today)",
	)
	parser.add_argument(
		"--years",
		type=int,
		default=2,
		help="Backtest window in years",
	)
	parser.add_argument(
		"--max-hold-days",
		type=int,
		default=20,
		help="Maximum holding days after trigger",
	)
	parser.add_argument("--swing-lookback", type=int, default=60)
	parser.add_argument("--near-level-pct", type=float, default=1.2)
	parser.add_argument("--volume-lookback", type=int, default=20)
	parser.add_argument("--volume-multiplier", type=float, default=1.2)
	parser.add_argument("--min-swing-pct", type=float, default=6.0)
	parser.add_argument(
		"--out-prefix",
		default="fibonacci_backtest",
		help="Output prefix for trades and summary CSVs",
	)
	return parser.parse_args()


def get_output_dir() -> str:
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	script_name = os.path.splitext(os.path.basename(__file__))[0]
	output_dir = os.path.join(base_dir, "Output", script_name)
	os.makedirs(output_dir, exist_ok=True)
	return output_dir


def main() -> None:
	args = parse_args()
	if args.as_of_date.strip():
		end_date = _parse_ddmmyyyy(args.as_of_date)
	else:
		end_date = date.today()

	symbols: List[str] = []
	if args.stocks_csv.strip():
		symbols.extend(load_symbols_from_csv(args.stocks_csv.strip()))
	if args.stocks.strip():
		symbols.extend(parse_symbols(args.stocks))
	symbols = sorted(set(symbols))

	if not symbols:
		raise ValueError("Provide symbols via --stocks or --stocks-csv")

	analysis_start = pd.Timestamp(end_date - timedelta(days=365 * args.years))
	history_start = end_date - timedelta(days=365 * args.years + 260)

	all_trades: List[Dict[str, object]] = []
	history_map = fetch_daily_history_nse(symbols, history_start, end_date)

	for sym in symbols:
		daily = history_map.get(sym, pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]))
		trades = backtest_symbol(
			symbol=sym,
			daily=daily,
			analysis_start=analysis_start,
			max_hold_days=args.max_hold_days,
			swing_lookback=args.swing_lookback,
			near_level_pct=args.near_level_pct,
			volume_lookback=args.volume_lookback,
			volume_multiplier=args.volume_multiplier,
			min_swing_pct=args.min_swing_pct,
		)
		all_trades.extend(trades)

	trades_df = pd.DataFrame(all_trades)
	summary_df = summarize_trades(trades_df)

	output_dir = get_output_dir()
	prefix = args.out_prefix.strip() or "fibonacci_backtest"
	prefix = os.path.basename(prefix)
	trades_out = os.path.join(output_dir, f"{prefix}_trades_{end_date.strftime('%d_%m_%Y')}.csv")
	summary_out = os.path.join(output_dir, f"{prefix}_summary_{end_date.strftime('%d_%m_%Y')}.csv")

	trades_df.to_csv(trades_out, index=False)
	summary_df.to_csv(summary_out, index=False)

	print(f"Backtest symbols: {', '.join(symbols)}")
	print(f"Window: {analysis_start.strftime('%d/%m/%Y')} to {pd.Timestamp(end_date).strftime('%d/%m/%Y')}")
	print(f"Trades file: {trades_out}")
	print(f"Summary file: {summary_out}")

	if summary_df.empty:
		print("No signals found.")
	else:
		overall = summary_df.iloc[0]
		print("\n=== Overall Summary ===")
		print(f"signals: {int(overall['signals'])}")
		print(f"triggered: {int(overall['triggered'])}")
		print(f"target_1_plus: {int(overall['target_1_plus'])}")
		print(f"stop_loss_1_hit: {int(overall['stop_loss_1_hit'])}")
		print(f"success_rate_t1_pct: {float(overall['success_rate_t1_pct']):.2f}")


if __name__ == "__main__":
	main()
