from __future__ import annotations

import http.cookiejar
import json
import os
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import BytesIO, StringIO
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

import pandas as pd


@dataclass
class CandleSet:
	daily: pd.DataFrame


_BHAVCOPY_CACHE: Dict[str, Optional[pd.DataFrame]] = {}
_FO_SYMBOL_CACHE: Optional[set] = None

PRICE_SPIKE_PCT = 5.0
VOLUME_SPIKE_MULT = 2.0
AVG_VOLUME_LOOKBACK = 20
MIN_AVG_VOLUME = 1_000_000
RSI_PERIOD = 14
RSI_MIN = 55.0
FIB_LOOKBACK = 60
ENTRY_MODE = "next_open"
EXIT_MODE = "target_stop_time"
HOLD_DAYS = 21
TARGET_SOURCE = "swing_high"
STOP_SOURCE = "fib_618"
EXIT_CONFLICT_RULE = "stop_first"
MAX_ENTRY_GAP_PCT = 2.0
MONTHLY_LOOKBACK_YEARS = 2
HISTORY_LOOKBACK_DAYS = MONTHLY_LOOKBACK_YEARS * 365
FORWARD_LOOKAHEAD_DAYS = 30
FO_BHAV_LOOKBACK_DAYS = 10

NSE_BASE_URL = "https://www.nseindia.com"
NSE_FO_INDEX_URL = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
NSE_FO_MKTLOTS_URLS = (
	"https://archives.nseindia.com/content/fo/fo_mktlots.csv",
	"https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv",
)
NSE_FO_BHAVCOPY_URLS = (
	"https://archives.nseindia.com/content/fo/fo{date}bhav.csv.zip",
	"https://nsearchives.nseindia.com/content/fo/fo{date}bhav.csv.zip",
)
NSE_HEADERS = {
	"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
	"(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
	"Accept": "application/json, text/plain, */*",
	"Accept-Language": "en-US,en;q=0.9",
	"Referer": "https://www.nseindia.com/",
	"Connection": "keep-alive",
}

def load_sec_list_symbols(anchor_date: date) -> List[str]:
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	csv_path = os.path.join(base_dir, "sec_list.csv")
	if not os.path.exists(csv_path):
		raise RuntimeError(f"Unable to find sec_list.csv at {csv_path}")

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

	fo_symbols = set(load_fo_symbols(anchor_date))
	symbols = symbols[symbols.isin(fo_symbols)]
	unique_sorted = sorted(set(symbols.tolist()))
	if not unique_sorted:
		raise RuntimeError("sec_list.csv returned no F&O EQ symbols.")
	return unique_sorted


def _parse_ddmmyyyy(value: str) -> date:
	try:
		return datetime.strptime(value.strip(), "%d/%m/%Y").date()
	except ValueError as exc:
		raise ValueError("Date must be in dd/mm/yyyy format.") from exc


def resolve_anchor_date() -> date:
	user_value = input("Enter as-of date (dd/mm/yyyy): ").strip()
	if not user_value:
		raise ValueError("As-of date is required.")
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


def _download_fo_bhavcopy_for_date(trade_date: date) -> Optional[pd.DataFrame]:
	date_token = trade_date.strftime("%d%b%Y").upper()
	for base_url in NSE_FO_BHAVCOPY_URLS:
		url = base_url.format(date=date_token)
		headers = dict(NSE_HEADERS)
		headers["Accept"] = "application/zip,application/octet-stream,*/*"
		request = Request(url, headers=headers)
		try:
			with urlopen(request, timeout=15) as response:
				content = response.read()
			with zipfile.ZipFile(BytesIO(content)) as zf:
				csv_name = zf.namelist()[0]
				with zf.open(csv_name) as csv_file:
					df = pd.read_csv(csv_file)
			df.columns = [str(c).strip().upper() for c in df.columns]
			return df
		except Exception:
			continue
	return None


def _load_fo_symbols_from_mktlots() -> List[str]:
	last_error: Optional[Exception] = None
	for url in NSE_FO_MKTLOTS_URLS:
		headers = dict(NSE_HEADERS)
		headers["Accept"] = "text/csv,text/plain,*/*"
		request = Request(url, headers=headers)
		try:
			with urlopen(request, timeout=15) as response:
				content = response.read().decode("utf-8", errors="ignore")
			data = pd.read_csv(StringIO(content), engine="python", on_bad_lines="skip")
			data.columns = [str(c).strip().upper() for c in data.columns]
			symbol_col = _find_column(data.columns, ["SYMBOL", "UNDERLYING"])
			if not symbol_col:
				raise RuntimeError("FO market lots file missing SYMBOL column.")
			symbols = (
				data[symbol_col]
				.astype(str)
				.str.strip()
				.str.upper()
				.loc[lambda s: s.str.match(r"^[A-Z0-9&\-]+$")]
			)
			unique = sorted(set(symbols.tolist()))
			if not unique:
				raise RuntimeError("FO market lots file returned no symbols.")
			print("Using FO market lots file for symbol list.")
			return unique
		except Exception as exc:
			last_error = exc
			continue

	raise RuntimeError("FO market lots download failed.") from last_error


def _fetch_nse_json(url: str) -> dict:
	cookie_jar = http.cookiejar.CookieJar()
	opener = build_opener(HTTPCookieProcessor(cookie_jar))
	opener.addheaders = list(NSE_HEADERS.items())
	try:
		opener.open(NSE_BASE_URL, timeout=10)
		with opener.open(url, timeout=10) as response:
			payload = response.read().decode("utf-8", errors="ignore")
			return json.loads(payload)
	except Exception as exc:
		raise RuntimeError("Unable to fetch NSE F&O symbol list.") from exc


def _load_fo_symbols_from_bhavcopy(as_of_date: date) -> List[str]:
	for offset in range(0, FO_BHAV_LOOKBACK_DAYS + 1):
		check_date = as_of_date - timedelta(days=offset)
		df = _download_fo_bhavcopy_for_date(check_date)
		if df is None or df.empty:
			continue

		instrument_col = _find_column(df.columns, ["INSTRUMENT"])
		symbol_col = _find_column(df.columns, ["SYMBOL"])
		if not instrument_col or not symbol_col:
			continue

		instruments = df[instrument_col].astype(str).str.strip().str.upper()
		filtered = df[instruments == "FUTSTK"]
		if filtered.empty:
			continue

		symbols = (
			filtered[symbol_col]
			.astype(str)
			.str.strip()
			.str.upper()
			.loc[lambda s: s.str.match(r"^[A-Z0-9&\-]+$")]
		)
		unique = sorted(set(symbols.tolist()))
		if unique:
			print(f"Using F&O bhavcopy from {check_date.strftime('%d/%m/%Y')} for symbol list.")
			return unique

	raise RuntimeError("Unable to build F&O symbol list from bhavcopy.")


def load_fo_symbols(as_of_date: Optional[date] = None) -> List[str]:
	global _FO_SYMBOL_CACHE
	if _FO_SYMBOL_CACHE is not None:
		return sorted(_FO_SYMBOL_CACHE)

	try:
		data = _fetch_nse_json(NSE_FO_INDEX_URL)
		items = data.get("data", []) if isinstance(data, dict) else []
		fo_symbols: List[str] = []
		for item in items:
			symbol = str(item.get("symbol", "")).strip().upper()
			if symbol:
				fo_symbols.append(symbol)

		unique = sorted(set(fo_symbols))
		if not unique:
			raise RuntimeError("NSE F&O symbol list returned no symbols.")
		_FO_SYMBOL_CACHE = set(unique)
		return unique
	except Exception as api_exc:
		if as_of_date is None:
			raise
		try:
			unique = _load_fo_symbols_from_mktlots()
			_FO_SYMBOL_CACHE = set(unique)
			return unique
		except Exception as mktlots_exc:
			try:
				unique = _load_fo_symbols_from_bhavcopy(as_of_date)
				_FO_SYMBOL_CACHE = set(unique)
				return unique
			except Exception as bhav_exc:
				raise RuntimeError(
					"Unable to fetch F&O symbol list from API, market lots, or bhavcopy."
				) from bhav_exc


def _find_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
	lookup = {str(c).strip().upper(): c for c in columns}
	for candidate in candidates:
		if candidate in lookup:
			return lookup[candidate]
	return None


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
	delta = close.diff()
	gain = delta.clip(lower=0)
	loss = -delta.clip(upper=0)

	avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
	avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

	rs = avg_gain / avg_loss.replace(0, pd.NA)
	rsi = 100 - (100 / (1 + rs))
	return rsi


def _safe_float(value: object) -> Optional[float]:
	if value is None or pd.isna(value):
		return None
	return float(value)


def _format_date(value: Optional[date]) -> str:
	return value.strftime("%d/%m/%Y") if value else ""


def _resolve_target_stop(row: object) -> Tuple[Optional[float], Optional[float]]:
	if TARGET_SOURCE == "swing_high":
		target = _safe_float(getattr(row, "SwingHigh", None))
	elif TARGET_SOURCE == "fib_618":
		target = _safe_float(getattr(row, "Fib618", None))
	else:
		target = _safe_float(getattr(row, "Fib50", None))

	if STOP_SOURCE == "fib_618":
		stop = _safe_float(getattr(row, "Fib618", None))
	elif STOP_SOURCE == "fib_50":
		stop = _safe_float(getattr(row, "Fib50", None))
	else:
		stop = _safe_float(getattr(row, "SwingLow", None))

	return target, stop


def _compute_exit(
	symbol_data: pd.DataFrame,
	entry_idx: int,
	hold_days: int,
	target_price: Optional[float],
	stop_price: Optional[float],
) -> Tuple[Optional[date], Optional[float], Optional[str]]:
	if entry_idx is None or entry_idx >= len(symbol_data):
		return None, None, None

	if hold_days <= 0:
		hold_days = 1

	time_stop_idx = min(entry_idx + hold_days - 1, len(symbol_data) - 1)
	time_stop_date = symbol_data.loc[time_stop_idx, "Date"]

	for idx in range(entry_idx, time_stop_idx + 1):
		row = symbol_data.loc[idx]
		open_price = _safe_float(row.get("Open"))
		if stop_price is not None and open_price is not None and open_price < stop_price:
			return row["Date"], open_price, "gap-stop"
		if target_price is not None and open_price is not None and open_price > target_price:
			return row["Date"], open_price, "target"

		hit_stop = stop_price is not None and row["Low"] <= stop_price
		hit_target = target_price is not None and row["High"] >= target_price
		if hit_stop and hit_target:
			# Conservative assumption: stop triggers before target on the same day.
			if EXIT_CONFLICT_RULE == "target_first":
				return row["Date"], target_price, "target"
			return row["Date"], stop_price, "stop"
		if hit_stop:
			return row["Date"], stop_price, "stop"
		if hit_target:
			return row["Date"], target_price, "target"

	return time_stop_date, _safe_float(symbol_data.loc[time_stop_idx, "Close"]), "time-stop"


def build_tradingview_link(symbol: str) -> str:
	return f"https://www.tradingview.com/chart/?symbol=NSE%3A{quote(str(symbol).upper())}"


def collect_history(
	anchor_date: date,
	symbols: Sequence[str],
	end_date: Optional[date] = None,
) -> pd.DataFrame:
	frames: List[pd.DataFrame] = []
	symbol_set = set(symbols)
	start_date = anchor_date - timedelta(days=HISTORY_LOOKBACK_DAYS)
	end_date = end_date or anchor_date

	day = start_date
	processed = 0
	while day <= end_date:
		df = _download_bhavcopy_for_date(day)
		if df is None or df.empty:
			day += timedelta(days=1)
			continue

		symbol_col = _find_column(df.columns, ["SYMBOL"])
		series_col = _find_column(df.columns, ["SERIES"])
		open_col = _find_column(df.columns, ["OPEN_PRICE", "OPEN", "OPEN_PRICE_"])
		high_col = _find_column(df.columns, ["HIGH_PRICE", "HIGH"])
		low_col = _find_column(df.columns, ["LOW_PRICE", "LOW"])
		close_col = _find_column(df.columns, ["CLOSE_PRICE", "CLOSE", "CLOSE_PRICE_"])
		volume_col = _find_column(df.columns, ["TOTTRDQTY", "TTL_TRD_QNTY", "VOLUME", "TOTTRD_QTY"])
		date_col = _find_column(df.columns, ["DATE1", "DATE", "TIMESTAMP"])

		if not all([symbol_col, series_col, high_col, low_col, close_col, date_col]):
			day += timedelta(days=1)
			continue

		day_df = df.copy()
		day_df[symbol_col] = day_df[symbol_col].astype(str).str.strip().str.upper()
		day_df[series_col] = day_df[series_col].astype(str).str.strip().str.upper()
		filtered = day_df[(day_df[symbol_col].isin(symbol_set)) & (day_df[series_col] == "EQ")]
		if filtered.empty:
			day += timedelta(days=1)
			continue

		trade_dates = pd.to_datetime(filtered[date_col].astype(str).str.strip(), errors="coerce", dayfirst=True)
		open_vals = pd.to_numeric(filtered[open_col], errors="coerce") if open_col else pd.Series(pd.NA, index=filtered.index)
		high_vals = pd.to_numeric(filtered[high_col], errors="coerce")
		low_vals = pd.to_numeric(filtered[low_col], errors="coerce")
		close_vals = pd.to_numeric(filtered[close_col], errors="coerce")
		vol_vals = pd.to_numeric(filtered[volume_col], errors="coerce") if volume_col else 0.0

		frame = pd.DataFrame(
			{
				"Symbol": filtered[symbol_col].astype(str).str.strip().str.upper(),
				"Date": trade_dates.dt.date,
				"Open": open_vals,
				"High": high_vals,
				"Low": low_vals,
				"Close": close_vals,
				"Volume": vol_vals,
			}
		).dropna(subset=["Date", "High", "Low", "Close"])

		if not frame.empty:
			frames.append(frame)

		processed += 1
		if processed % 60 == 0:
			print(f"Processed {processed} trading days...")

		day += timedelta(days=1)

	if not frames:
		return pd.DataFrame(columns=["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"])

	return pd.concat(frames, ignore_index=True)


def detect_spikes(
	history: pd.DataFrame,
	anchor_date: date,
) -> List[Dict[str, object]]:
	rows: List[Dict[str, object]] = []
	if history.empty:
		return rows

	data = history.sort_values(["Symbol", "Date"]).copy()
	data["PrevClose"] = data.groupby("Symbol")["Close"].shift(1)
	data["AvgVolume"] = data.groupby("Symbol")["Volume"].transform(
		lambda s: s.rolling(AVG_VOLUME_LOOKBACK).mean()
	)
	data["RSI14"] = data.groupby("Symbol")["Close"].transform(
		lambda s: compute_rsi(s, RSI_PERIOD)
	)
	data["SwingHigh"] = data.groupby("Symbol")["High"].transform(
		lambda s: s.rolling(FIB_LOOKBACK).max()
	)
	data["SwingLow"] = data.groupby("Symbol")["Low"].transform(
		lambda s: s.rolling(FIB_LOOKBACK).min()
	)
	fib_range = (data["SwingHigh"] - data["SwingLow"]).replace(0, pd.NA)
	data["Fib50"] = data["SwingHigh"] - 0.5 * fib_range
	data["Fib618"] = data["SwingHigh"] - 0.618 * fib_range
	data["SpikePct"] = (data["Close"] - data["PrevClose"]) / data["PrevClose"] * 100.0
	data["VolumeMultiple"] = data["Volume"] / data["AvgVolume"]

	mask = data["Date"] == anchor_date
	mask &= data["PrevClose"].notna() & (data["PrevClose"] > 0)
	mask &= data["Open"].notna()
	mask &= data["AvgVolume"].notna() & (data["AvgVolume"] >= MIN_AVG_VOLUME)
	mask &= data["SpikePct"] >= PRICE_SPIKE_PCT
	mask &= data["VolumeMultiple"] >= VOLUME_SPIKE_MULT
	mask &= data["Close"] > data["Open"]
	mask &= data["RSI14"].notna() & (data["RSI14"] >= RSI_MIN)
	mask &= data["SwingHigh"].notna() & data["SwingLow"].notna()
	mask &= data["Fib50"].notna() & data["Fib618"].notna()
	mask &= data["Close"] > data["Fib50"]

	filtered = data[mask].copy()
	if filtered.empty:
		return rows

	for row in filtered.itertuples(index=False):
		symbol_data = data[data["Symbol"] == row.Symbol].reset_index(drop=True)
		t0_date = row.Date
		t0_close = _safe_float(row.Close)
		if t0_close is None:
			continue
		max_entry_price = t0_close * (1 + MAX_ENTRY_GAP_PCT / 100.0)

		entry_date = None
		entry_price = None
		entry_status = "pending"
		exit_date = None
		exit_price = None
		exit_reason = ""

		if ENTRY_MODE == "next_open":
			future_rows = symbol_data[symbol_data["Date"] > t0_date]
			entry_row = future_rows.iloc[0] if not future_rows.empty else None
		else:
			entry_row = symbol_data.loc[symbol_data["Date"] == t0_date].iloc[0]

		if entry_row is not None:
			entry_date = entry_row["Date"]
			entry_open = _safe_float(entry_row.get("Open"))
			if entry_open is not None and entry_open <= max_entry_price:
				entry_status = "entered"
				entry_price = entry_open
				target_price, stop_price = _resolve_target_stop(row)
				exit_date, exit_price, exit_reason = _compute_exit(
					symbol_data,
					int(entry_row.name),
					HOLD_DAYS if EXIT_MODE == "target_stop_time" else 1,
					target_price,
					stop_price,
				)
			elif entry_open is not None:
				entry_status = "gap-skip"

		rows.append(
			{
				"symbol": row.Symbol,
				"t0_date": row.Date.strftime("%d/%m/%Y"),
				"entry_date": _format_date(entry_date),
				"entry_price": round(entry_price, 2) if entry_price is not None else None,
				"max_entry_price": round(max_entry_price, 2),
				"entry_status": entry_status,
				"spike_pct": round(float(row.SpikePct), 2),
				"volume_multiple": round(float(row.VolumeMultiple), 2),
				"rsi14": round(float(row.RSI14), 2),
				"swing_high": round(float(row.SwingHigh), 2),
				"swing_low": round(float(row.SwingLow), 2),
				"fib_50": round(float(row.Fib50), 2),
				"fib_618": round(float(row.Fib618), 2),
				"exit_date": _format_date(exit_date),
				"exit_price": round(exit_price, 2) if exit_price is not None else None,
				"exit_reason": exit_reason or "",
				"tradingview_link": build_tradingview_link(row.Symbol),
			}
		)

	return rows


def build_output_name(as_of_date: date) -> str:
	return f"calendar_spike_strategy_{as_of_date.strftime('%d_%m_%Y')}.csv"


def get_output_dir() -> str:
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	script_name = os.path.splitext(os.path.basename(__file__))[0]
	output_dir = os.path.join(base_dir, "Output", script_name)
	os.makedirs(output_dir, exist_ok=True)
	return output_dir


def main() -> None:
	anchor_date = resolve_anchor_date()
	symbols = load_sec_list_symbols(anchor_date)

	print(f"Running on NSE F&O EQ symbols: {len(symbols)}")
	print("Building daily history. This can take a while...")

	history_end_date = anchor_date + timedelta(days=FORWARD_LOOKAHEAD_DAYS)
	history = collect_history(
		anchor_date=anchor_date,
		symbols=symbols,
		end_date=history_end_date,
	)

	results = detect_spikes(history, anchor_date)
	if results:
		df = pd.DataFrame(results)
		df = df.sort_values(["t0_date", "symbol"]).reset_index(drop=True)
	else:
		df = pd.DataFrame(
			columns=[
				"symbol",
				"t0_date",
				"entry_date",
				"entry_price",
				"max_entry_price",
				"entry_status",
				"spike_pct",
				"volume_multiple",
				"rsi14",
				"swing_high",
				"swing_low",
				"fib_50",
				"fib_618",
				"exit_date",
				"exit_price",
				"exit_reason",
				"tradingview_link",
			]
		)

	output_dir = get_output_dir()
	output_path = os.path.join(output_dir, build_output_name(anchor_date))
	df.to_csv(output_path, index=False)

	print(f"As-of date: {anchor_date.strftime('%d/%m/%Y')}")
	print(f"Output file: {output_path}")
	print("\n=== Calendar Spike Strategy (Price + Volume) ===")
	if df.empty:
		print("No candidates")
	else:
		print(", ".join(df["symbol"].tolist()))
	print(f"Count: {len(df)}")


if __name__ == "__main__":
	main()
