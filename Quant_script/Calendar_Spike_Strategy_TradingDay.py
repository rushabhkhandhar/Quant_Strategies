from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd


@dataclass
class CandleSet:
	daily: pd.DataFrame


_BHAVCOPY_CACHE: Dict[str, Optional[pd.DataFrame]] = {}

PRICE_SPIKE_PCT = 5.0
VOLUME_SPIKE_MULT = 2.0
AVG_VOLUME_LOOKBACK = 20
MIN_AVG_VOLUME = 1_000_000
DIRECTION = "bullish"

MONTHLY_LOOKBACK_YEARS = 2
def _resolve_monthly_days(anchor_date: date) -> List[int]:
	days = [anchor_date.day - 2, anchor_date.day - 1, anchor_date.day]
	return [d for d in days if 1 <= d <= 31]


def load_sec_list_symbols() -> List[str]:
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
	unique_sorted = sorted(set(symbols.tolist()))
	if not unique_sorted:
		raise RuntimeError("sec_list.csv returned no valid symbols.")
	return unique_sorted


def _parse_ddmmyyyy(value: str) -> date:
	try:
		return datetime.strptime(value.strip(), "%d/%m/%Y").date()
	except ValueError as exc:
		raise ValueError("Date must be in dd/mm/yyyy format.") from exc


def resolve_anchor_date() -> date:
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


def _find_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
	lookup = {str(c).strip().upper(): c for c in columns}
	for candidate in candidates:
		if candidate in lookup:
			return lookup[candidate]
	return None


def build_tradingview_link(symbol: str) -> str:
	return f"https://www.tradingview.com/chart/?symbol=NSE%3A{quote(str(symbol).upper())}"


def _month_iter(end_date: date, months_back: int) -> Iterable[Tuple[int, int]]:
	year = end_date.year
	month = end_date.month
	for _ in range(months_back + 1):
		yield year, month
		month -= 1
		if month == 0:
			month = 12
			year -= 1


def _make_date_safe(year: int, month: int, day: int) -> Optional[date]:
	try:
		return date(year, month, day)
	except ValueError:
		return None


def build_target_dates(anchor_date: date) -> Dict[date, List[str]]:
	date_types: Dict[date, List[str]] = {}
	monthly_days = _resolve_monthly_days(anchor_date)

	monthly_start = anchor_date - timedelta(days=MONTHLY_LOOKBACK_YEARS * 365)
	for year, month in _month_iter(anchor_date, MONTHLY_LOOKBACK_YEARS * 12):
		for day in monthly_days:
			dt = _make_date_safe(year, month, day)
			if dt is None:
				continue
			if dt < monthly_start or dt > anchor_date:
				continue
			date_types.setdefault(dt, []).append("monthly")

	return date_types


def adjust_to_previous_trading_day(
	date_types: Dict[date, List[str]],
	trading_days: List[date],
) -> Dict[date, List[str]]:
	if not trading_days:
		return {}

	trading_days_sorted = sorted(trading_days)
	adjusted: Dict[date, List[str]] = {}

	for target_date, labels in date_types.items():
		if target_date < trading_days_sorted[0]:
			continue

		candidate = None
		for day in reversed(trading_days_sorted):
			if day <= target_date:
				candidate = day
				break

		if candidate is None:
			continue

		existing = adjusted.setdefault(candidate, [])
		for label in labels:
			if label not in existing:
				existing.append(label)

	return adjusted


def collect_series_data(
	anchor_date: date,
	symbols: Sequence[str],
) -> Tuple[Dict[str, List[Tuple[date, float, float]]], List[date]]:
	series_data: Dict[str, List[Tuple[date, float, float]]] = {symbol: [] for symbol in symbols}
	trading_days: List[date] = []
	symbol_set = set(symbols)
	start_date = anchor_date - timedelta(days=MONTHLY_LOOKBACK_YEARS * 365)

	day = start_date
	processed = 0
	while day <= anchor_date:
		df = _download_bhavcopy_for_date(day)
		if df is None or df.empty:
			day += timedelta(days=1)
			continue

		trading_days.append(day)

		symbol_col = _find_column(df.columns, ["SYMBOL"])
		series_col = _find_column(df.columns, ["SERIES"])
		close_col = _find_column(df.columns, ["CLOSE_PRICE", "CLOSE", "CLOSE_PRICE_"])
		volume_col = _find_column(df.columns, ["TOTTRDQTY", "TTL_TRD_QNTY", "VOLUME", "TOTTRD_QTY"])
		date_col = _find_column(df.columns, ["DATE1", "DATE", "TIMESTAMP"])

		if not all([symbol_col, series_col, close_col, date_col]):
			day += timedelta(days=1)
			continue

		day_df = df.copy()
		day_df[symbol_col] = day_df[symbol_col].astype(str).str.strip().str.upper()
		day_df[series_col] = day_df[series_col].astype(str).str.strip().str.upper()
		filtered = day_df[(day_df[symbol_col].isin(symbol_set)) & (day_df[series_col] == "EQ")]
		if filtered.empty:
			day += timedelta(days=1)
			continue

		for row in filtered.itertuples(index=False):
			row_dict = row._asdict() if hasattr(row, "_asdict") else None
			if row_dict is not None:
				symbol = str(row_dict[symbol_col]).strip().upper()
				trade_date = pd.to_datetime(str(row_dict[date_col]).strip(), errors="coerce", dayfirst=True)
				close_val = pd.to_numeric(row_dict[close_col], errors="coerce")
				vol_val = 0.0
				if volume_col:
					vol_series = pd.to_numeric(pd.Series([row_dict[volume_col]]), errors="coerce").iloc[0]
					if not pd.isna(vol_series):
						vol_val = float(vol_series)
			else:
				row_list = list(row)
				idx = {col: i for i, col in enumerate(df.columns)}
				symbol = str(row_list[idx[symbol_col]]).strip().upper()
				trade_date = pd.to_datetime(str(row_list[idx[date_col]]).strip(), errors="coerce", dayfirst=True)
				close_val = pd.to_numeric(row_list[idx[close_col]], errors="coerce")
				vol_val = 0.0
				if volume_col:
					vol_series = pd.to_numeric(pd.Series([row_list[idx[volume_col]]]), errors="coerce").iloc[0]
					if not pd.isna(vol_series):
						vol_val = float(vol_series)

			if pd.isna(trade_date) or pd.isna(close_val):
				continue

			trade_day = trade_date.date()
			series_data[symbol].append((trade_day, float(close_val), float(vol_val)))

		processed += 1
		if processed % 60 == 0:
			print(f"Processed {processed} trading days...")

		day += timedelta(days=1)

	return series_data, trading_days


def detect_spikes(
	series_data: Dict[str, List[Tuple[date, float, float]]],
	date_types: Dict[date, List[str]],
) -> List[Dict[str, object]]:
	rows: List[Dict[str, object]] = []

	for symbol, entries in series_data.items():
		if not entries:
			continue

		df = pd.DataFrame(entries, columns=["Date", "Close", "Volume"]).sort_values("Date")
		df["PrevClose"] = df["Close"].shift(1)
		df["AvgVolume"] = df["Volume"].rolling(AVG_VOLUME_LOOKBACK).mean()

		for _, row in df.iterrows():
			trade_date = row["Date"]
			if trade_date not in date_types:
				continue
			prev_close = row["PrevClose"]
			avg_vol = row["AvgVolume"]
			if pd.isna(prev_close) or pd.isna(avg_vol) or prev_close == 0:
				continue

			close_val = float(row["Close"])
			vol_val = float(row["Volume"])
			if avg_vol < MIN_AVG_VOLUME:
				continue

			if DIRECTION == "bullish" and close_val <= prev_close:
				continue
			if DIRECTION == "bearish" and close_val >= prev_close:
				continue
			price_change_pct = abs((close_val - prev_close) / prev_close) * 100.0
			vol_multiple = vol_val / avg_vol if avg_vol > 0 else 0.0

			if price_change_pct < PRICE_SPIKE_PCT:
				continue
			if vol_multiple < VOLUME_SPIKE_MULT:
				continue

			for interval_type in date_types[trade_date]:
				rows.append(
					{
						"symbol": symbol,
						"date": trade_date.strftime("%d/%m/%Y"),
						"interval_type": interval_type,
						"close": round(close_val, 2),
						"prev_close": round(float(prev_close), 2),
						"price_change_pct": round(price_change_pct, 2),
						"volume": int(round(vol_val)),
						"avg20_volume": int(round(avg_vol)),
						"volume_multiple": round(vol_multiple, 2),
						"tradingview_link": build_tradingview_link(symbol),
						"year": trade_date.year,
					}
				)

	if not rows:
		return rows

	symbol_years: Dict[str, set[int]] = {}
	for row in rows:
		symbol_years.setdefault(row["symbol"], set()).add(int(row["year"]))

	filtered = [row for row in rows if len(symbol_years.get(row["symbol"], set())) >= 2]
	for row in filtered:
		row.pop("year", None)

	return filtered


def build_output_name(as_of_date: date) -> str:
	return f"calendar_spike_strategy_tradingday_{as_of_date.strftime('%d_%m_%Y')}.csv"


def build_comparison_name(as_of_date: date) -> str:
	return f"calendar_spike_strategy_comparison_{as_of_date.strftime('%d_%m_%Y')}.csv"


def build_base_output_name(as_of_date: date) -> str:
	return f"calendar_spike_strategy_{as_of_date.strftime('%d_%m_%Y')}.csv"


def get_output_dir() -> str:
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	script_name = os.path.splitext(os.path.basename(__file__))[0]
	output_dir = os.path.join(base_dir, "Output", script_name)
	os.makedirs(output_dir, exist_ok=True)
	return output_dir


def compare_outputs(anchor_date: date, new_rows: List[Dict[str, object]]) -> None:
	output_dir = get_output_dir()
	base_path = os.path.join(output_dir, build_base_output_name(anchor_date))
	comparison_path = os.path.join(output_dir, build_comparison_name(anchor_date))

	new_df = pd.DataFrame(new_rows)
	if os.path.exists(base_path):
		old_df = pd.read_csv(base_path)
	else:
		old_df = pd.DataFrame(columns=["symbol", "date", "interval_type", "price_change_pct", "volume_multiple"])

	old_keys = set(zip(old_df.get("symbol", []), old_df.get("date", []), old_df.get("interval_type", [])))
	new_keys = set(zip(new_df.get("symbol", []), new_df.get("date", []), new_df.get("interval_type", [])))
	all_keys = sorted(old_keys | new_keys)

	rows: List[Dict[str, object]] = []
	old_lookup = {
		(symbol, dt, itype): row
		for _, row in old_df.iterrows()
		for symbol, dt, itype in [(row.get("symbol"), row.get("date"), row.get("interval_type"))]
	}
	new_lookup = {
		(symbol, dt, itype): row
		for _, row in new_df.iterrows()
		for symbol, dt, itype in [(row.get("symbol"), row.get("date"), row.get("interval_type"))]
	}

	for key in all_keys:
		symbol, dt, itype = key
		old_row = old_lookup.get(key)
		new_row = new_lookup.get(key)
		row = {
			"symbol": symbol,
			"date": dt,
			"interval_type": itype,
			"in_calendar": key in old_keys,
			"in_tradingday": key in new_keys,
		}
		row["price_change_pct"] = (
			new_row.get("price_change_pct") if new_row is not None else old_row.get("price_change_pct")
		)
		row["volume_multiple"] = (
			new_row.get("volume_multiple") if new_row is not None else old_row.get("volume_multiple")
		)
		rows.append(row)

	pd.DataFrame(rows).to_csv(comparison_path, index=False)


def main() -> None:
	anchor_date = resolve_anchor_date()
	calendar_dates = build_target_dates(anchor_date)
	symbols = load_sec_list_symbols()

	print(f"Running on all SEC list symbols: {len(symbols)}")
	print("Building daily history. This can take a while...")

	series_data, trading_days = collect_series_data(
		anchor_date=anchor_date,
		symbols=symbols,
	)
	adjusted_dates = adjust_to_previous_trading_day(calendar_dates, trading_days)

	results = detect_spikes(series_data, adjusted_dates)
	if results:
		df = pd.DataFrame(results)
		df = df.sort_values(["date", "interval_type", "symbol"]).reset_index(drop=True)
	else:
		df = pd.DataFrame(
			columns=[
				"symbol",
				"date",
				"interval_type",
				"close",
				"prev_close",
				"price_change_pct",
				"volume",
				"avg20_volume",
				"volume_multiple",
				"tradingview_link",
			]
		)

	output_dir = get_output_dir()
	output_path = os.path.join(output_dir, build_output_name(anchor_date))
	df.to_csv(output_path, index=False)

	compare_outputs(anchor_date, results)

	print(f"Anchor date: {anchor_date.strftime('%d/%m/%Y')}")
	print(f"Output file: {output_path}")
	print("\n=== Calendar Spike Strategy (Trading-Day Adjusted) ===")
	if df.empty:
		print("No candidates")
	else:
		print(", ".join(df["symbol"].tolist()))
	print(f"Count: {len(df)}")


if __name__ == "__main__":
	main()
