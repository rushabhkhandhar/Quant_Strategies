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


@dataclass
class CandleSet:
	daily: pd.DataFrame


_BHAVCOPY_CACHE: Dict[str, Optional[pd.DataFrame]] = {}


def _load_symbols_from_csv_urls(
	urls: Sequence[str],
	column_name: str = "SYMBOL",
	request_timeout: int = 20,
) -> List[str]:
	headers = {
		"User-Agent": "Mozilla/5.0",
		"Accept": "text/csv,text/plain,*/*",
	}

	last_error: Optional[Exception] = None
	for url in urls:
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
			return _extract_symbols_from_dataframe(raw, column_name, source=url)
		except Exception as exc:
			last_error = exc
			continue

	raise RuntimeError(f"Unable to load symbols from all sources. Last error: {last_error}")


def _extract_symbols_from_dataframe(
	raw: pd.DataFrame,
	column_name: str,
	source: str,
) -> List[str]:
	raw.columns = [str(c).strip().upper() for c in raw.columns]
	col = column_name.strip().upper()
	if col not in raw.columns:
		raise RuntimeError(f"Unable to find {col} column in CSV: {source}")

	symbols = raw[col].astype(str).str.strip().str.upper()
	symbols = symbols[symbols.str.match(r"^[A-Z0-9&\-]+$")]
	symbols = symbols[symbols != col]

	unique_sorted = sorted(set(symbols.tolist()))
	if not unique_sorted:
		raise RuntimeError(f"No valid symbols found in CSV: {source}")
	return unique_sorted


def _load_symbols_from_local_csv(file_path: str, column_name: str = "SYMBOL") -> List[str]:
	if not os.path.exists(file_path):
		return []

	raw = pd.read_csv(
		file_path,
		skipinitialspace=True,
		engine="python",
		on_bad_lines="skip",
	)
	return _extract_symbols_from_dataframe(raw, column_name, source=file_path)


def load_nse_midcap_stock_symbols() -> List[str]:
	"""Fetch NSE midcap stock symbols online only.

	Uses NIFTY Midcap 150 constituents as the midcap universe.
	"""
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	cache_path = os.path.join(base_dir, "nifty_midcap_150_symbols.csv")

	urls = [
		"https://niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
		"https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
	]
	try:
		symbols = _load_symbols_from_csv_urls(urls, column_name="Symbol", request_timeout=30)
		return symbols
	except Exception as exc:
		cached = _load_symbols_from_local_csv(cache_path, column_name="SYMBOL")
		if cached:
			print(f"Using cached midcap symbols from {cache_path}")
			return cached
		raise RuntimeError(
			"Unable to load midcap symbols online and no cache found. "
			"Provide --symbols or create a cached file at "
			f"{cache_path}."
		) from exc


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

	if len(daily) < 80:
		return None

	return CandleSet(daily=daily)


def compute_macd(close: pd.Series) -> pd.DataFrame:
	ema12 = close.ewm(span=12, adjust=False).mean()
	ema26 = close.ewm(span=26, adjust=False).mean()
	macd_line = ema12 - ema26
	signal_line = macd_line.ewm(span=9, adjust=False).mean()
	hist = macd_line - signal_line
	return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def compute_bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
	middle = close.rolling(period).mean()
	std = close.rolling(period).std(ddof=0)
	upper = middle + std_mult * std
	lower = middle - std_mult * std
	width_pct = ((upper - lower) / middle.replace(0, pd.NA)) * 100.0
	return pd.DataFrame({"bb_mid": middle, "bb_upper": upper, "bb_lower": lower, "bb_width_pct": width_pct})


def build_tradingview_link(symbol: str) -> str:
	return f"https://www.tradingview.com/chart/?symbol=NSE%3A{quote(str(symbol).upper())}"


def trend_shift_vol_expansion_signal(
	daily: pd.DataFrame,
	bb_narrow_lookback: int,
	bb_width_max_pct: float,
	volume_multiplier: float,
	near_breakout_pct: float,
) -> Optional[Dict[str, float]]:
	"""Tomorrow setup using MACD trend shift + Bollinger volatility expansion.

	Bullish:
	- MACD bullish crossover today
	- Previous BB widths indicate narrow range
	- Close is near but below upper band (potential upside expansion tomorrow)
	- Volume above average

	Bearish:
	- MACD bearish crossover today
	- Previous BB widths indicate narrow range
	- Close is near but above lower band (potential downside expansion tomorrow)
	- Volume above average
	"""
	if len(daily) < max(60, bb_narrow_lookback + 25):
		return None

	df = daily.copy()
	macd_df = compute_macd(df["Close"])
	bb_df = compute_bollinger(df["Close"], period=20, std_mult=2.0)
	df = pd.concat([df, macd_df, bb_df], axis=1).dropna()
	if len(df) < bb_narrow_lookback + 3:
		return None

	curr = df.iloc[-1]
	prev = df.iloc[-2]
	bb_prev = df.iloc[-(bb_narrow_lookback + 1):-1]

	avg_width = float(bb_prev["bb_width_pct"].mean())
	narrow_band = avg_width <= bb_width_max_pct
	if not narrow_band:
		return None

	avg_vol = float(bb_prev["Volume"].mean())
	curr_vol = float(curr["Volume"])
	if avg_vol > 0:
		vol_ok = curr_vol >= (volume_multiplier * avg_vol)
	else:
		vol_ok = True
	if not vol_ok:
		return None

	bull_cross = float(curr["macd"]) > float(curr["signal"]) and float(prev["macd"]) <= float(prev["signal"])
	bear_cross = float(curr["macd"]) < float(curr["signal"]) and float(prev["macd"]) >= float(prev["signal"])

	close_price = float(curr["Close"])
	upper = float(curr["bb_upper"])
	lower = float(curr["bb_lower"])

	up_dist = ((upper - close_price) / upper) * 100.0 if upper > 0 else 999.0
	down_dist = ((close_price - lower) / max(lower, 1e-9)) * 100.0 if lower > 0 else 999.0

	near_upper = close_price <= upper and up_dist <= near_breakout_pct
	near_lower = close_price >= lower and down_dist <= near_breakout_pct

	direction = ""
	distance = 0.0
	if bull_cross and near_upper:
		direction = "bullish"
		distance = max(up_dist, 0.0)
	elif bear_cross and near_lower:
		direction = "bearish"
		distance = max(down_dist, 0.0)
	else:
		return None

	return {
		"direction": direction,
		"signal_close": close_price,
		"macd": float(curr["macd"]),
		"signal": float(curr["signal"]),
		"hist": float(curr["hist"]),
		"bb_upper": upper,
		"bb_lower": lower,
		"bb_width_avg_pct": avg_width,
		"distance_to_band_pct": distance,
		"avg_volume": avg_vol,
		"curr_volume": curr_vol,
	}


def run_screen(
	symbols: Sequence[str],
	as_of_date: date,
	bb_narrow_lookback: int,
	bb_width_max_pct: float,
	volume_multiplier: float,
	near_breakout_pct: float,
	verbose: bool = False,
) -> pd.DataFrame:
	rows: List[Dict[str, object]] = []

	for symbol in symbols:
		candles = fetch_daily_candles(symbol, as_of_date=as_of_date)
		if candles is None:
			if verbose:
				print(f"{symbol}: SKIPPED (no_data)")
			continue

		signal = trend_shift_vol_expansion_signal(
			candles.daily,
			bb_narrow_lookback=bb_narrow_lookback,
			bb_width_max_pct=bb_width_max_pct,
			volume_multiplier=volume_multiplier,
			near_breakout_pct=near_breakout_pct,
		)
		if signal is None:
			if verbose:
				print(f"{symbol}: no_setup")
			continue

		rows.append(
			{
				"symbol": symbol,
				"date": candles.daily.index[-1].date().strftime("%d/%m/%Y"),
				"direction": signal["direction"],
				"close": round(float(signal["signal_close"]), 2),
				"macd": round(float(signal["macd"]), 4),
				"signal_line": round(float(signal["signal"]), 4),
				"histogram": round(float(signal["hist"]), 4),
				"bb_upper": round(float(signal["bb_upper"]), 2),
				"bb_lower": round(float(signal["bb_lower"]), 2),
				"bb_width_avg_pct": round(float(signal["bb_width_avg_pct"]), 2),
				"distance_to_band_pct": round(float(signal["distance_to_band_pct"]), 2),
				"avg_volume": int(round(float(signal["avg_volume"]))),
				"curr_volume": int(round(float(signal["curr_volume"]))),
				"tradingview_link": build_tradingview_link(symbol),
			}
		)

		if verbose:
			print(f"{symbol}: {signal['direction'].upper()}_WATCH")

	if not rows:
		return pd.DataFrame(
			columns=[
				"symbol",
				"date",
				"direction",
				"close",
				"macd",
				"signal_line",
				"histogram",
				"bb_upper",
				"bb_lower",
				"bb_width_avg_pct",
				"distance_to_band_pct",
				"avg_volume",
				"curr_volume",
				"tradingview_link",
			]
		)

	return pd.DataFrame(rows).sort_values(["distance_to_band_pct", "bb_width_avg_pct"], ascending=[True, True]).reset_index(drop=True)


def build_output_name(as_of_date: date) -> str:
	return f"trend_shift_volatility_expansion_tomorrow_{as_of_date.strftime('%d_%m_%Y')}.csv"


def get_output_dir() -> str:
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	script_name = os.path.splitext(os.path.basename(__file__))[0]
	output_dir = os.path.join(base_dir, "Output", script_name)
	os.makedirs(output_dir, exist_ok=True)
	return output_dir


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Trend Shift + Volatility Expansion screener for NSE Midcap watchlist"
	)
	parser.add_argument(
		"--symbols",
		default="",
		help="Comma-separated symbols to run; if empty, runs full NSE midcap universe",
	)
	parser.add_argument(
		"--all-symbols",
		action="store_true",
		help="Run on all NSE midcap symbols",
	)
	parser.add_argument(
		"--as-of-date",
		default="",
		help="Anchor date in dd/mm/yyyy format; scan uses data up to and including this date",
	)
	parser.add_argument(
		"--bb-narrow-lookback",
		type=int,
		default=15,
		help="Lookback bars (excluding current) for narrow Bollinger range check",
	)
	parser.add_argument(
		"--bb-width-max-pct",
		type=float,
		default=8.0,
		help="Maximum average Bollinger band width percent to qualify as consolidation",
	)
	parser.add_argument(
		"--volume-multiplier",
		type=float,
		default=1.0,
		help="Current volume must be >= this multiplier times average recent volume",
	)
	parser.add_argument(
		"--near-breakout-pct",
		type=float,
		default=2.0,
		help="Current close must be within this percent distance to breakout band",
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

	if args.all_symbols:
		symbols = load_nse_midcap_stock_symbols()
		print(f"Running on all NSE midcap symbols: {len(symbols)}")
	else:
		symbols = parse_symbols(args.symbols)
		if not symbols:
			symbols = load_nse_midcap_stock_symbols()
			print(f"Running on all NSE midcap symbols: {len(symbols)}")
		else:
			print(f"Running only on symbols: {', '.join(symbols)}")

	results = run_screen(
		symbols,
		as_of_date=as_of_date,
		bb_narrow_lookback=args.bb_narrow_lookback,
		bb_width_max_pct=args.bb_width_max_pct,
		volume_multiplier=args.volume_multiplier,
		near_breakout_pct=args.near_breakout_pct,
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
	print("\n=== Trend Shift + Volatility Expansion Watchlist ===")
	if results.empty:
		print("No candidates")
	else:
		print(", ".join(results["symbol"].tolist()))
	print(f"Count: {len(results)}")


if __name__ == "__main__":
	main()
