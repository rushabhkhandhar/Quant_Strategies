from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Dict, List, Optional, Sequence
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd
import time


@dataclass
class CandleSet:
	daily: pd.DataFrame


_BHAVCOPY_CACHE: Dict[str, Optional[pd.DataFrame]] = {}


def _load_symbols_from_csv_urls(
	urls: Sequence[str],
	column_name: str = "SYMBOL",
	request_timeout: int = 20,
	retry_attempts: int = 3,
	retry_backoff_sec: float = 1.5,
) -> List[str]:
	headers = {
		"User-Agent": "Mozilla/5.0",
		"Accept": "text/csv,text/plain,*/*",
	}

	last_error: Optional[Exception] = None
	for url in urls:
		for attempt in range(1, retry_attempts + 1):
			try:
				request = Request(url, headers=headers)
				with urlopen(request, timeout=request_timeout) as response:
					content = response.read().decode("utf-8", errors="ignore")

				raw = pd.read_csv(
					StringIO(content),
					skipinitialspace=True,
					engine="python",
					on_bad_lines="skip",
				)
				raw.columns = [str(c).strip().upper() for c in raw.columns]

				col = column_name.strip().upper()
				if col not in raw.columns:
					raise RuntimeError(f"Unable to find {col} column in CSV: {url}")

				symbols = raw[col].astype(str).str.strip().str.upper()
				symbols = symbols[symbols.str.match(r"^[A-Z0-9&\-]+$")]
				symbols = symbols[symbols != col]

				unique_sorted = sorted(set(symbols.tolist()))
				if unique_sorted:
					return unique_sorted
			except Exception as exc:
				last_error = exc
				if attempt < retry_attempts:
					time.sleep(retry_backoff_sec * attempt)
					continue

	raise RuntimeError(f"Unable to load symbols from all sources. Last error: {last_error}")


def load_nse_futures_stock_symbols() -> List[str]:
	"""Fetch all NSE futures-eligible stock symbols online only (no fallback)."""
	symbols = _load_symbols_from_csv_urls(
		["https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"],
		column_name="SYMBOL",
	)

	index_symbols = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}
	unique_sorted = sorted(set([s for s in symbols if s not in index_symbols]))
	if not unique_sorted:
		raise RuntimeError("Online NSE futures stock list returned no valid symbols.")

	return unique_sorted


def load_nifty500_stock_symbols() -> List[str]:
	"""Fetch NIFTY 500 constituent symbols online only (no fallback)."""
	urls = [
		"https://niftyindices.com/IndexConstituent/ind_nifty500list.csv",
		"https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
	]
	symbols = _load_symbols_from_csv_urls(urls, column_name="Symbol", request_timeout=30)
	if not symbols:
		raise RuntimeError("Online NIFTY 500 list returned no valid symbols.")
	return symbols


def parse_symbols(value: str) -> List[str]:
	return [s.strip().upper() for s in value.split(",") if s.strip()]


def _parse_ddmmyyyy(value: str) -> date:
	try:
		return datetime.strptime(value.strip(), "%d/%m/%Y").date()
	except ValueError as exc:
		raise ValueError("Date must be in dd/mm/yyyy format.") from exc


def resolve_anchor_date(value: str) -> date:
	if value.strip():
		return _parse_ddmmyyyy(value)

	user_value = input("Enter anchor date (dd/mm/yyyy): ").strip()
	if not user_value:
		raise ValueError("Anchor date is required.")
	return _parse_ddmmyyyy(user_value)


def _download_bhavcopy_for_date(trade_date: date) -> Optional[pd.DataFrame]:
	"""Download one NSE full bhavcopy day and cache it by date string."""
	key = trade_date.strftime("%Y-%m-%d")
	if key in _BHAVCOPY_CACHE:
		return _BHAVCOPY_CACHE[key]

	url = (
		"https://nsearchives.nseindia.com/products/content/"
		f"sec_bhavdata_full_{trade_date.strftime('%d%m%Y')}.csv"
	)
	headers = {
		"User-Agent": "Mozilla/5.0",
		"Accept": "text/csv,text/plain,*/*",
	}
	request = Request(url, headers=headers)

	try:
		with urlopen(request, timeout=10) as response:
			content = response.read().decode("utf-8", errors="ignore")
		df = pd.read_csv(StringIO(content), engine="python", on_bad_lines="skip")
		df.columns = [str(c).strip().upper() for c in df.columns]
		_BHAVCOPY_CACHE[key] = df
		return df
	except Exception:
		_BHAVCOPY_CACHE[key] = None
		return None


def _find_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
	lookup = {str(c).strip().upper(): c for c in columns}
	for candidate in candidates:
		if candidate in lookup:
			return lookup[candidate]
	return None


def fetch_daily_candles(symbol: str, as_of_date: date) -> Optional[CandleSet]:
	rows: List[Dict[str, float]] = []
	max_lookback_days = 260

	for offset in range(0, max_lookback_days + 1):
		day = as_of_date - timedelta(days=offset)
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

		day_df = df.copy()
		day_df[symbol_col] = day_df[symbol_col].astype(str).str.strip().str.upper()
		day_df[series_col] = day_df[series_col].astype(str).str.strip().str.upper()
		filtered = day_df[(day_df[symbol_col] == symbol) & (day_df[series_col] == "EQ")]
		if filtered.empty:
			continue

		r = filtered.iloc[0]
		trade_date = pd.to_datetime(str(r[date_col]).strip(), errors="coerce", dayfirst=True)
		if pd.isna(trade_date):
			continue

		ohlc = pd.to_numeric(r[[open_col, high_col, low_col, close_col]], errors="coerce")
		if ohlc.isna().any():
			continue

		volume = 0.0
		if volume_col:
			vol = pd.to_numeric(pd.Series([r[volume_col]]), errors="coerce").iloc[0]
			if not pd.isna(vol):
				volume = float(vol)

		rows.append(
			{
				"Date": trade_date,
				"Open": float(ohlc.iloc[0]),
				"High": float(ohlc.iloc[1]),
				"Low": float(ohlc.iloc[2]),
				"Close": float(ohlc.iloc[3]),
				"Volume": volume,
			}
		)

		if len(rows) >= 180:
			break

	if not rows:
		return None

	daily = (
		pd.DataFrame(rows)
		.dropna(subset=["Date", "Open", "High", "Low", "Close"])
		.sort_values("Date")
		.drop_duplicates(subset=["Date"], keep="last")
		.set_index("Date")
	)

	if len(daily) < 70:
		return None

	return CandleSet(daily=daily)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
	delta = close.diff()
	gain = delta.clip(lower=0)
	loss = -delta.clip(upper=0)

	avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
	avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

	rs = avg_gain / avg_loss.replace(0, pd.NA)
	rsi = 100 - (100 / (1 + rs))
	return rsi.fillna(100)


def build_tradingview_link(symbol: str) -> str:
	return f"https://www.tradingview.com/chart/?symbol=NSE%3A{quote(str(symbol).upper())}"


def trend_momentum_tomorrow_signal(
	daily: pd.DataFrame,
	pullback_lookback: int,
	breakout_lookback: int,
	volume_multiplier: float,
	near_breakout_pct: float,
	close_position_min: float,
	rsi_dip_threshold: float,
	rsi_recovery_min: float,
	rsi_recovery_max: float,
) -> Optional[Dict[str, float]]:
	"""Tomorrow breakout watchlist using Trend + Momentum + Confirmation.

	Rules:
	1) Trend filter: EMA20 > EMA50 and current close > EMA20.
	2) Momentum setup: RSI(14) dipped below threshold recently (buy zone behavior).
	3) RSI recovery now: current RSI in recovery band and rising vs yesterday.
	4) Breakout readiness: current close is near but below recent breakout high.
	5) Confirmation proxy: bullish close near day high and above-average volume.
	"""
	if len(daily) < max(55, pullback_lookback + 2, breakout_lookback + 1):
		return None

	df = daily.copy()
	df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
	df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
	df["RSI14"] = compute_rsi(df["Close"], period=14)

	curr = df.iloc[-1]
	prev = df.iloc[-2]

	uptrend = (
		float(curr["EMA20"]) > float(curr["EMA50"])
		and float(curr["Close"]) > float(curr["EMA20"])
	)
	if not uptrend:
		return None

	recent = df.iloc[-(pullback_lookback + 1):-1]
	recent_rsi_min = float(recent["RSI14"].min())
	rsi_dip_ok = recent_rsi_min < rsi_dip_threshold
	if not rsi_dip_ok:
		return None

	curr_rsi = float(curr["RSI14"])
	prev_rsi = float(prev["RSI14"])
	rsi_recovery_ok = (
		curr_rsi >= rsi_recovery_min
		and curr_rsi <= rsi_recovery_max
		and curr_rsi > prev_rsi
	)
	if not rsi_recovery_ok:
		return None

	base = df.iloc[-(breakout_lookback + 1):-1]
	resistance = float(base["High"].max())
	close_price = float(curr["Close"])
	distance_pct = ((resistance - close_price) / resistance) * 100.0 if resistance > 0 else 999.0
	near_breakout = close_price <= resistance and distance_pct <= near_breakout_pct
	if not near_breakout:
		return None

	open_price = float(curr["Open"])
	high_price = float(curr["High"])
	low_price = float(curr["Low"])
	day_range = max(high_price - low_price, 1e-9)
	close_position = (close_price - low_price) / day_range
	bullish_pressure = close_price > open_price and close_position >= close_position_min
	if not bullish_pressure:
		return None

	avg_vol = float(base["Volume"].mean())
	curr_vol = float(curr["Volume"])
	if avg_vol > 0:
		vol_ok = curr_vol >= (volume_multiplier * avg_vol)
	else:
		vol_ok = True
	if not vol_ok:
		return None

	return {
		"signal_close": close_price,
		"ema20": float(curr["EMA20"]),
		"ema50": float(curr["EMA50"]),
		"rsi14": curr_rsi,
		"resistance": resistance,
		"distance_to_breakout_pct": max(distance_pct, 0.0),
		"close_position": close_position,
		"avg_volume": avg_vol,
		"curr_volume": curr_vol,
	}


def run_screen(
	symbols: Sequence[str],
	as_of_date: date,
	pullback_lookback: int,
	breakout_lookback: int,
	volume_multiplier: float,
	near_breakout_pct: float,
	close_position_min: float,
	rsi_dip_threshold: float,
	rsi_recovery_min: float,
	rsi_recovery_max: float,
	verbose: bool = False,
) -> pd.DataFrame:
	rows: List[Dict[str, object]] = []

	for symbol in symbols:
		candles = fetch_daily_candles(symbol, as_of_date=as_of_date)
		if candles is None:
			if verbose:
				print(f"{symbol}: SKIPPED (no_data)")
			continue

		signal = trend_momentum_tomorrow_signal(
			candles.daily,
			pullback_lookback=pullback_lookback,
			breakout_lookback=breakout_lookback,
			volume_multiplier=volume_multiplier,
			near_breakout_pct=near_breakout_pct,
			close_position_min=close_position_min,
			rsi_dip_threshold=rsi_dip_threshold,
			rsi_recovery_min=rsi_recovery_min,
			rsi_recovery_max=rsi_recovery_max,
		)
		if signal is None:
			if verbose:
				print(f"{symbol}: no_tomorrow_setup")
			continue

		rows.append(
			{
				"symbol": symbol,
				"date": candles.daily.index[-1].date().strftime("%d/%m/%Y"),
				"close": round(float(signal["signal_close"]), 2),
				"ema20": round(float(signal["ema20"]), 2),
				"ema50": round(float(signal["ema50"]), 2),
				"rsi14": round(float(signal["rsi14"]), 2),
				"resistance": round(float(signal["resistance"]), 2),
				"distance_to_breakout_pct": round(float(signal["distance_to_breakout_pct"]), 2),
				"close_position": round(float(signal["close_position"]), 3),
				"avg_volume": int(round(float(signal["avg_volume"]))),
				"curr_volume": int(round(float(signal["curr_volume"]))),
				"tradingview_link": build_tradingview_link(symbol),
			}
		)

		if verbose:
			print(f"{symbol}: TOMORROW_BREAKOUT_WATCH")

	if not rows:
		return pd.DataFrame(
			columns=[
				"symbol",
				"date",
				"close",
				"ema20",
				"ema50",
				"rsi14",
				"resistance",
				"distance_to_breakout_pct",
				"close_position",
				"avg_volume",
				"curr_volume",
				"tradingview_link",
			]
		)

	return pd.DataFrame(rows).sort_values(["distance_to_breakout_pct", "rsi14"], ascending=[True, True]).reset_index(drop=True)


def build_output_name(as_of_date: date) -> str:
	return f"trend_momentum_tomorrow_{as_of_date.strftime('%d_%m_%Y')}.csv"


def get_output_dir() -> str:
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	script_name = os.path.splitext(os.path.basename(__file__))[0]
	output_dir = os.path.join(base_dir, "Output", script_name)
	os.makedirs(output_dir, exist_ok=True)
	return output_dir


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Trend + Momentum + Confirmation screener for next-day breakout watchlist"
	)
	parser.add_argument(
		"--symbols",
		default="",
		help="Comma-separated symbols to run; if empty, runs all symbols from selected universe",
	)
	parser.add_argument(
		"--all-symbols",
		action="store_true",
		help="Run on all symbols from selected universe",
	)
	parser.add_argument(
		"--universe",
		choices=["futures", "nifty500"],
		default="nifty500",
		help="Symbol universe when not using --symbols",
	)
	parser.add_argument(
		"--as-of-date",
		default="",
		help="Anchor date in dd/mm/yyyy format; scan uses data up to and including this date",
	)
	parser.add_argument(
		"--pullback-lookback",
		type=int,
		default=15,
		help="Lookback bars for checking RSI dip below threshold",
	)
	parser.add_argument(
		"--breakout-lookback",
		type=int,
		default=20,
		help="Lookback bars (excluding current) used to define resistance",
	)
	parser.add_argument(
		"--volume-multiplier",
		type=float,
		default=1.0,
		help="Current volume must be >= this multiplier times average breakout-lookback volume",
	)
	parser.add_argument(
		"--near-breakout-pct",
		type=float,
		default=2.0,
		help="Current close must be within this percent below resistance",
	)
	parser.add_argument(
		"--close-position-min",
		type=float,
		default=0.55,
		help="Minimum close position in today's candle range (0 to 1)",
	)
	parser.add_argument(
		"--rsi-dip-threshold",
		type=float,
		default=40.0,
		help="Recent RSI dip threshold in uptrend",
	)
	parser.add_argument(
		"--rsi-recovery-min",
		type=float,
		default=40.0,
		help="Current RSI lower bound for recovery zone",
	)
	parser.add_argument(
		"--rsi-recovery-max",
		type=float,
		default=60.0,
		help="Current RSI upper bound for recovery zone",
	)
	parser.add_argument(
		"--out",
		default="",
		help="Optional explicit output CSV path",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Print per-symbol evaluation status",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	as_of_date = resolve_anchor_date(args.as_of_date)

	if args.universe == "futures":
		universe_name = "NSE futures"
		universe_loader = load_nse_futures_stock_symbols
	else:
		universe_name = "NIFTY 500"
		universe_loader = load_nifty500_stock_symbols

	if args.all_symbols:
		symbols = universe_loader()
		print(f"Running on all {universe_name} symbols: {len(symbols)}")
	else:
		symbols = parse_symbols(args.symbols)
		if not symbols:
			symbols = universe_loader()
			print(f"Running on all {universe_name} symbols: {len(symbols)}")
		else:
			print(f"Running only on symbols: {', '.join(symbols)}")

	results = run_screen(
		symbols,
		as_of_date=as_of_date,
		pullback_lookback=args.pullback_lookback,
		breakout_lookback=args.breakout_lookback,
		volume_multiplier=args.volume_multiplier,
		near_breakout_pct=args.near_breakout_pct,
		close_position_min=args.close_position_min,
		rsi_dip_threshold=args.rsi_dip_threshold,
		rsi_recovery_min=args.rsi_recovery_min,
		rsi_recovery_max=args.rsi_recovery_max,
		verbose=args.verbose,
	)

	output_dir = get_output_dir()
	if args.out.strip():
		out_name = os.path.basename(args.out.strip())
		output_path = os.path.join(output_dir, out_name)
	else:
		output_path = os.path.join(output_dir, build_output_name(as_of_date))
	results.to_csv(output_path, index=False)

	print(f"Anchor date: {as_of_date.strftime('%d/%m/%Y')}")
	print(f"Output file: {output_path}")
	print("\n=== Trend_Momentum Tomorrow Watchlist ===")
	if results.empty:
		print("No candidates")
	else:
		print(", ".join(results["symbol"].tolist()))
	print(f"Count: {len(results)}")


if __name__ == "__main__":
	main()
