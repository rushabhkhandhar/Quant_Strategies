from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote
from urllib.request import Request, urlopen
import matplotlib.pyplot as plt
import mplfinance as mpf 
import pandas as pd


@dataclass
class CandleSet:
	daily: pd.DataFrame


_BHAVCOPY_CACHE: Dict[str, Optional[pd.DataFrame]] = {}

# ---------------------------------------------------------------------------
# Liquidity threshold: 20-Day Average (Volume × Close Price) >= 70 Crore
# ---------------------------------------------------------------------------
LIQUIDITY_THRESHOLD_CRORE = 70
LIQUIDITY_THRESHOLD = LIQUIDITY_THRESHOLD_CRORE * 1_00_00_000  # 70 Cr in INR


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
			continue

	raise RuntimeError(f"Unable to load symbols from all sources. Last error: {last_error}")


def load_sec_list_symbols() -> List[str]:
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	candidate_paths = [
		os.path.join(base_dir, "Data", "sec_list.csv"),
		os.path.join(base_dir, "sec_list.csv"),
	]
	csv_path = next((path for path in candidate_paths if os.path.exists(path)), None)
	if csv_path is None:
		raise RuntimeError(
			"Unable to find sec_list.csv. Looked in: " + ", ".join(candidate_paths)
		)

	raw = pd.read_csv(
		csv_path,
		skipinitialspace=True,
		engine="python",
		on_bad_lines="skip",
	)
	raw.columns = [str(c).strip().upper() for c in raw.columns]

	if "SERIES" in raw.columns:
		raw = raw[raw["SERIES"].astype(str).str.upper() == "EQ"]

	if "SYMBOL" not in raw.columns:
		raise RuntimeError("Unable to find SYMBOL column in sec_list.csv")

	symbols = raw["SYMBOL"].astype(str).str.strip().str.upper()
	symbols = symbols[symbols.str.match(r"^[A-Z0-9&\-]+$")]
	unique_sorted = sorted(set(symbols.tolist()))
	if not unique_sorted:
		raise RuntimeError("sec_list.csv returned no valid symbols.")
	return unique_sorted


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


def _find_column(columns: Any, candidates: Sequence[str]) -> Optional[str]:
	lookup = {str(c).strip().upper(): c for c in columns}
	for candidate in candidates:
		if candidate in lookup:
			return lookup[candidate]
	return None


def fetch_daily_candles(symbol: str, as_of_date: date) -> Optional[CandleSet]:
	rows: List[Dict[str, float]] = []
	max_lookback_days = 320

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

		if not (symbol_col and series_col and open_col and high_col and low_col and close_col and date_col):
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

		if len(rows) >= 240:
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

	if len(daily) < 90:
		return None

	return CandleSet(daily=daily)


def is_liquid(daily: pd.DataFrame) -> bool:
	"""Return True if the 20-day average Volume × Close >= LIQUIDITY_THRESHOLD."""
	if len(daily) < 20:
		return False
	
	recent = daily.iloc[-20:]
	avg_traded_value = (recent["Volume"] * recent["Close"]).mean()
	return float(avg_traded_value) >= LIQUIDITY_THRESHOLD


def build_tradingview_link(symbol: str) -> str:
	return f"https://www.tradingview.com/chart/?symbol=NSE%3A{quote(str(symbol).upper())}"


def sanitize_filename(value: str) -> str:
	allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
	cleaned = "".join(ch if ch in allowed else "_" for ch in value)
	return cleaned.strip("_") or "chart"


def export_annotated_chart(
	symbol: str,
	daily: pd.DataFrame,
	signal: Dict[str, float],
	chart_path: str,
	lookback_bars: int,
) -> None:
	"""Export a candlestick chart with collision-safe level annotations."""
	plot_df = daily.tail(lookback_bars).copy()
	if plot_df.empty:
		return

	plot_df.index = pd.to_datetime(plot_df.index)
	plot_df = plot_df[["Open", "High", "Low", "Close", "Volume"]]

	signal_marker = pd.Series(float("nan"), index=plot_df.index)
	signal_marker.iloc[-1] = float(signal["signal_close"])

	addplots = [
		mpf.make_addplot(
			signal_marker,
			type="scatter",
			marker="o",
			markersize=60,
			color="#f59e0b",
			panel=0,
		)
	]

	mpf_style = mpf.make_mpf_style(base_mpf_style="yahoo", y_on_right=False)

	fig, axes = mpf.plot(
		plot_df,
		type="candle",
		style=mpf_style,
		volume=True,
		addplot=addplots,
		figscale=1.2,
		figratio=(16, 9),
		title=f"{symbol} | Fibonacci + Candlestick + Volume Setup",
		ylabel="Price",
		ylabel_lower="Volume",
		returnfig=True,
	)

	price_ax = axes[0] if isinstance(axes, list) else axes

	raw_levels = [
		("Swing Low", float(signal["swing_low"]), "#6b7280", "--"),
		("Swing High", float(signal["swing_high"]), "#111827", "--"),
		("Fib 38.2", float(signal["fib_382"]), "#2563eb", "-"),
		("Fib 50", float(signal["fib_50"]), "#1d4ed8", "-"),
		("Fib 61.8", float(signal["fib_618"]), "#1e40af", "-"),
		("Entry", float(signal["entry_price"]), "#16a34a", "-"),
		("SL1", float(signal["stop_loss_1"]), "#dc2626", ":"),
		("SL2", float(signal["stop_loss_2"]), "#ef4444", ":"),
		("SL3", float(signal["stop_loss_3"]), "#f87171", ":"),
		("T1", float(signal["target_1"]), "#059669", "-"),
		("T2", float(signal["target_2"]), "#047857", "-"),
		("T3", float(signal["target_3"]), "#065f46", "-"),
	]

	# Merge labels that sit on the same practical level (2-decimal precision).
	priority = {
		"Entry": 5,
		"T1": 4,
		"T2": 4,
		"T3": 4,
		"Fib 38.2": 3,
		"Fib 50": 3,
		"Fib 61.8": 3,
		"Swing High": 2,
		"Swing Low": 2,
		"SL1": 1,
		"SL2": 1,
		"SL3": 1,
	}
	merged: Dict[float, Dict[str, Any]] = {}
	for name, level, color, style in raw_levels:
		k = round(level, 2)
		if k not in merged:
			merged[k] = {
				"level": level,
				"names": [name],
				"color": color,
				"style": style,
				"prio": priority.get(name, 0),
			}
		else:
			item = merged[k]
			item["names"].append(name)
			if priority.get(name, 0) > item["prio"]:
				item["color"] = color
				item["style"] = style
				item["prio"] = priority.get(name, 0)

	levels = [
		(" / ".join(item["names"]), float(item["level"]), str(item["color"]), str(item["style"]))
		for item in merged.values()
	]

	for _, level, color, style in levels:
		price_ax.axhline(level, color=color, linestyle=style, linewidth=1.0, alpha=0.9)

	y_min, y_max = price_ax.get_ylim()
	y_range = max(y_max - y_min, 1e-9)
	min_gap = y_range * 0.025

	ordered = sorted(levels, key=lambda x: x[1], reverse=True)
	placed: List[float] = []
	label_rows: List[tuple[str, float, float, str]] = []
	for name, raw_y, color, _ in ordered:
		y_label = raw_y
		if placed and (placed[-1] - y_label) < min_gap:
			y_label = placed[-1] - min_gap
		y_label = min(max(y_label, y_min + min_gap), y_max - min_gap)
		placed.append(y_label)
		label_rows.append((name, raw_y, y_label, color))

	x_min, x_max = price_ax.get_xlim()
	x_span = max(x_max - x_min, 1e-9)
	x_text = x_max - 0.09 * x_span
	x_anchor = x_max - 0.25 * x_span

	for name, raw_y, y_label, color in label_rows:
		if abs(y_label - raw_y) > 1e-9:
			price_ax.plot([x_anchor, x_text - 0.01 * x_span], [raw_y, y_label], color=color, linewidth=0.8, alpha=0.9)
		price_ax.text(
			x_text,
			y_label,
			f"{name}: {raw_y:.2f}",
			color=color,
			fontsize=8,
			va="center",
			ha="left",
			bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5),
		)

	signal_y = float(signal["signal_close"])
	price_ax.text(
		x_text,
		signal_y,
		"Signal",
		color="#92400e",
		fontsize=8,
		va="bottom",
		ha="left",
		bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=1.5),
	)

	chart_dir = os.path.dirname(chart_path)
	if chart_dir:
		os.makedirs(chart_dir, exist_ok=True)
	fig.savefig(chart_path, dpi=150, bbox_inches="tight")
	plt.close(fig)


def compute_trade_levels(
	entry_level: float,
	fib_382: float,
	fib_618: float,
	swing_low: float,
	swing_high: float,
	current_low: float,
) -> Dict[str, float]:
	"""Build practical target and stop-loss tiers for long setups."""
	range_size = max(swing_high - swing_low, 1e-9)

	# Added a 1% buffer below the 61.8% level for SL1 to avoid stop-hunts
	stop_loss_1 = min(current_low, fib_618 * 0.99)
	stop_loss_2 = swing_low
	stop_loss_3 = swing_low - 0.25 * range_size

	target_1 = fib_382
	target_2 = swing_high
	target_3 = swing_high + 0.272 * range_size

	return {
		"entry_price": entry_level,
		"stop_loss_1": stop_loss_1,
		"stop_loss_2": stop_loss_2,
		"stop_loss_3": stop_loss_3,
		"target_1": target_1,
		"target_2": target_2,
		"target_3": target_3,
	}


def bullish_hammer(candle: "pd.Series[Any]") -> bool:
	open_price = float(candle["Open"])
	close_price = float(candle["Close"])
	high_price = float(candle["High"])
	low_price = float(candle["Low"])

	body = abs(close_price - open_price)
	lower_shadow = min(open_price, close_price) - low_price
	upper_shadow = high_price - max(open_price, close_price)
	day_range = max(high_price - low_price, 1e-9)

	# Relaxed: Can be red or green. Lower shadow must be at least 2x the body.
	# Upper shadow must be small.
	return (
		lower_shadow >= 2.0 * max(body, 1e-9)
		and upper_shadow <= (0.15 * day_range)
	)


def bullish_engulfing(prev_candle: "pd.Series[Any]", curr_candle: "pd.Series[Any]") -> bool:
	prev_open = float(prev_candle["Open"])
	prev_close = float(prev_candle["Close"])
	curr_open = float(curr_candle["Open"])
	curr_close = float(curr_candle["Close"])

	# Relaxed: Previous is red, Current is green.
	# Current close strongly engulfs previous open (ignoring exact gap constraints).
	return (
		prev_close < prev_open
		and curr_close > curr_open
		and curr_close > prev_open
	)


def fibonacci_bounce_tomorrow_signal(
	daily: pd.DataFrame,
	swing_lookback: int,
	near_level_pct: float,
	volume_lookback: int,
	volume_multiplier: float,
	min_swing_pct: float,
) -> Optional[Dict[str, Any]]:
	"""Fibonacci retracement bounce setup for next day.

	Rules:
	1) Build Fib from recent swing low to swing high.
	2) Current candle is near 50% or 61.8% retracement zone.
	3) Bullish candlestick confirmation (Hammer or Bullish Engulfing).
	4) Current volume is above average volume (spike).
	"""
	if len(daily) < max(swing_lookback + 2, volume_lookback + 2):
		return None

	# Macro Trend Filter: Ensure stock is in a general uptrend (Close > 200 EMA)
	ema_200 = daily["Close"].ewm(span=200, adjust=False).mean()
	if float(daily.iloc[-1]["Close"]) < float(ema_200.iloc[-1]):
		return None

	window = daily.iloc[-swing_lookback:]
	high_idx = window["High"].idxmax()
	
	# Stale Swing Check: Ensure the swing high occurred within the last 20 trading days
	days_since_high = len(window.loc[high_idx:]) - 1  # type: ignore
	if days_since_high > 20:
		return None

	prefix = window.loc[:high_idx]  # type: ignore
	if len(prefix) < 2:
		return None

	low_idx = prefix["Low"].idxmin()
	swing_high = float(window.loc[high_idx, "High"])  # type: ignore
	swing_low = float(prefix.loc[low_idx, "Low"])  # type: ignore

	if swing_high <= swing_low:
		return None

	swing_pct = ((swing_high - swing_low) / swing_low) * 100.0 if swing_low > 0 else 0.0
	if swing_pct < min_swing_pct:
		return None

	fib_382 = swing_high - 0.382 * (swing_high - swing_low)
	fib_50 = swing_high - 0.500 * (swing_high - swing_low)
	fib_618 = swing_high - 0.618 * (swing_high - swing_low)

	prev = daily.iloc[-2]
	curr = daily.iloc[-1]
	close_price = float(curr["Close"])
	low_price = float(curr["Low"])
	high_price = float(curr["High"])

	dist_50 = abs(close_price - fib_50) / fib_50 * 100.0 if fib_50 > 0 else 999.0
	dist_618 = abs(close_price - fib_618) / fib_618 * 100.0 if fib_618 > 0 else 999.0

	touched_50 = low_price <= fib_50 <= high_price
	touched_618 = low_price <= fib_618 <= high_price

	# Ensure price didn't just crash through and close significantly below support.
	# The close should be above, or extremely close to the fib level (within 0.5% below).
	defends_50 = touched_50 and close_price >= (fib_50 * 0.995)
	defends_618 = touched_618 and close_price >= (fib_618 * 0.995)

	near_50 = defends_50 or dist_50 <= near_level_pct
	near_618 = defends_618 or dist_618 <= near_level_pct
	if not (near_50 or near_618):
		return None

	pattern_name = ""
	if bullish_engulfing(prev, curr):
		pattern_name = "bullish_engulfing"
	elif bullish_hammer(curr):
		pattern_name = "hammer"
	else:
		return None

	vol_base = daily.iloc[-(volume_lookback + 1):-1]
	avg_vol = float(vol_base["Volume"].mean())
	curr_vol = float(curr["Volume"])
	if avg_vol > 0:
		vol_ok = curr_vol >= (volume_multiplier * avg_vol)
	else:
		vol_ok = True
	if not vol_ok:
		return None

	entry_level_name = "fib_50"
	entry_level = fib_50
	entry_distance = dist_50
	if dist_618 < dist_50:
		entry_level_name = "fib_618"
		entry_level = fib_618
		entry_distance = dist_618

	trade_levels = compute_trade_levels(
		entry_level=entry_level,
		fib_382=fib_382,
		fib_618=fib_618,
		swing_low=swing_low,
		swing_high=swing_high,
		current_low=low_price,
	)

	return {
		"signal_close": close_price,
		"swing_low": swing_low,
		"swing_high": swing_high,
		"fib_382": fib_382,
		"fib_50": fib_50,
		"fib_618": fib_618,
		"entry_level": entry_level,
		"entry_level_name": entry_level_name,
		"entry_distance_pct": entry_distance,
		"pattern": pattern_name,
		"avg_volume": avg_vol,
		"curr_volume": curr_vol,
		"entry_price": trade_levels["entry_price"],
		"stop_loss_1": trade_levels["stop_loss_1"],
		"stop_loss_2": trade_levels["stop_loss_2"],
		"stop_loss_3": trade_levels["stop_loss_3"],
		"target_1": trade_levels["target_1"],
		"target_2": trade_levels["target_2"],
		"target_3": trade_levels["target_3"],
	}


def _process_symbol(
	symbol: str,
	as_of_date: date,
	swing_lookback: int,
	near_level_pct: float,
	volume_lookback: int,
	volume_multiplier: float,
	min_swing_pct: float,
	export_charts: bool,
	charts_dir: str,
	chart_lookback_bars: int,
	verbose: bool,
) -> tuple[Optional[Dict[str, object]], Optional[str]]:
	candles = fetch_daily_candles(symbol, as_of_date=as_of_date)
	if candles is None:
		return None, f"{symbol}: SKIPPED (no_data)" if verbose else None

	# ------------------------------------------------------------------
	# Liquidity filter: 20-Day Average Volume × Close must be >= 70 Crore
	# ------------------------------------------------------------------
	if not is_liquid(candles.daily):
		recent = candles.daily.iloc[-20:]
		avg_traded_cr = (recent["Volume"] * recent["Close"]).mean() / 1_00_00_000 if len(recent) > 0 else 0
		return None, (
			f"{symbol}: SKIPPED (low_liquidity {avg_traded_cr:.1f} Cr < {LIQUIDITY_THRESHOLD_CRORE} Cr avg)"
			if verbose else None
		)

	signal = fibonacci_bounce_tomorrow_signal(
		candles.daily,
		swing_lookback=swing_lookback,
		near_level_pct=near_level_pct,
		volume_lookback=volume_lookback,
		volume_multiplier=volume_multiplier,
		min_swing_pct=min_swing_pct,
	)
	if signal is None:
		return None, f"{symbol}: no_setup" if verbose else None

	chart_rel_path = ""
	if export_charts:
		chart_file = f"{sanitize_filename(symbol)}_{as_of_date.strftime('%d_%m_%Y')}.png"
		chart_rel_path = os.path.join(charts_dir, chart_file)
		export_annotated_chart(
			symbol=symbol,
			daily=candles.daily,
			signal=signal,
			chart_path=chart_rel_path,
			lookback_bars=chart_lookback_bars,
		)

	row = {
		"symbol": symbol,
		"date": candles.daily.index[-1].date().strftime("%d/%m/%Y"),
		"close": round(float(signal["signal_close"]), 2),
		"swing_low": round(float(signal["swing_low"]), 2),
		"swing_high": round(float(signal["swing_high"]), 2),
		"fib_382": round(float(signal["fib_382"]), 2),
		"fib_50": round(float(signal["fib_50"]), 2),
		"fib_618": round(float(signal["fib_618"]), 2),
		"entry_level_name": signal["entry_level_name"],
		"entry_level": round(float(signal["entry_level"]), 2),
		"entry_distance_pct": round(float(signal["entry_distance_pct"]), 2),
		"entry_price": round(float(signal["entry_price"]), 2),
		"pattern": signal["pattern"],
		"stop_loss_1": round(float(signal["stop_loss_1"]), 2),
		"stop_loss_2": round(float(signal["stop_loss_2"]), 2),
		"stop_loss_3": round(float(signal["stop_loss_3"]), 2),
		"target_1": round(float(signal["target_1"]), 2),
		"target_2": round(float(signal["target_2"]), 2),
		"target_3": round(float(signal["target_3"]), 2),
		"avg_volume": int(round(float(signal["avg_volume"]))),
		"curr_volume": int(round(float(signal["curr_volume"]))),
		"annotated_chart_path": chart_rel_path,
		"tradingview_link": build_tradingview_link(symbol),
	}

	return row, f"{symbol}: FIB_BOUNCE_SETUP" if verbose else None


def run_screen(
	symbols: Sequence[str],
	as_of_date: date,
	swing_lookback: int,
	near_level_pct: float,
	volume_lookback: int,
	volume_multiplier: float,
	min_swing_pct: float,
	export_charts: bool,
	charts_dir: str,
	chart_lookback_bars: int,
	max_workers: int,
	verbose: bool = False,
) -> pd.DataFrame:
	rows: List[Dict[str, object]] = []
	if export_charts:
		os.makedirs(charts_dir, exist_ok=True)

	if max_workers and max_workers > 1:
		with ProcessPoolExecutor(max_workers=max_workers) as executor:
			futures = [
				executor.submit(
					_process_symbol,
					symbol,
					as_of_date,
					swing_lookback,
					near_level_pct,
					volume_lookback,
					volume_multiplier,
					min_swing_pct,
					export_charts,
					charts_dir,
					chart_lookback_bars,
					verbose,
				)
				for symbol in symbols
			]
			for future in as_completed(futures):
				row, message = future.result()
				if message and verbose:
					print(message)
				if row:
					rows.append(row)
	else:
		for symbol in symbols:
			row, message = _process_symbol(
				symbol,
				as_of_date,
				swing_lookback,
				near_level_pct,
				volume_lookback,
				volume_multiplier,
				min_swing_pct,
				export_charts,
				charts_dir,
				chart_lookback_bars,
				verbose,
			)
			if message and verbose:
				print(message)
			if row:
				rows.append(row)

	if not rows:
		return pd.DataFrame(
			columns=[
				"symbol",
				"date",
				"close",
				"swing_low",
				"swing_high",
				"fib_382",
				"fib_50",
				"fib_618",
				"entry_level_name",
				"entry_level",
				"entry_distance_pct",
				"entry_price",
				"pattern",
				"stop_loss_1",
				"stop_loss_2",
				"stop_loss_3",
				"target_1",
				"target_2",
				"target_3",
				"avg_volume",
				"curr_volume",
				"annotated_chart_path",
				"tradingview_link",
			]
		)

	return (
		pd.DataFrame(rows)
		.sort_values(["entry_distance_pct", "pattern"], ascending=[True, True])
		.reset_index(drop=True)
	)


def build_output_name(as_of_date: date) -> str:
	return f"fibonacci_retracement_candlestick_volume_tomorrow_{as_of_date.strftime('%d_%m_%Y')}.csv"


def get_output_dir() -> str:
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	script_name = os.path.splitext(os.path.basename(__file__))[0]
	output_dir = os.path.join(base_dir, "Output", script_name)
	os.makedirs(output_dir, exist_ok=True)
	return output_dir


def main() -> None:
	as_of_date = resolve_anchor_date("")
	universe_name = "SEC list"
	universe_loader = load_sec_list_symbols
	symbols = universe_loader()
	print(f"Running on all {universe_name} symbols: {len(symbols)}")
	print(f"Liquidity filter: Volume × Close >= {LIQUIDITY_THRESHOLD_CRORE} Crore")

	output_dir = get_output_dir()
	swing_lookback = 60
	near_level_pct = 1.2
	volume_lookback = 20
	volume_multiplier = 1.2
	min_swing_pct = 6.0
	export_charts = False
	charts_dir = os.path.join(output_dir, "charts")
	chart_lookback_bars = 120
	max_workers = max((os.cpu_count() or 2) - 1, 1)
	verbose = False

	results = run_screen(
		symbols,
		as_of_date=as_of_date,
		swing_lookback=swing_lookback,
		near_level_pct=near_level_pct,
		volume_lookback=volume_lookback,
		volume_multiplier=volume_multiplier,
		min_swing_pct=min_swing_pct,
		export_charts=export_charts,
		charts_dir=charts_dir,
		chart_lookback_bars=chart_lookback_bars,
		max_workers=max_workers,
		verbose=verbose,
	)

	output_path = os.path.join(output_dir, build_output_name(as_of_date))
	results.to_csv(output_path, index=False)

	print(f"Anchor date: {as_of_date.strftime('%d/%m/%Y')}")
	print(f"Output file: {output_path}")
	print(f"\n=== Fibonacci Retracement + Candlestick + Volume Tomorrow Watchlist ({universe_name}) ===")
	if results.empty:
		print("No candidates")
	else:
		print(", ".join(results["symbol"].tolist()))
	print(f"Count: {len(results)}")


if __name__ == "__main__":
	main()