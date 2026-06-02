from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Dict, List, Optional, Sequence

import pandas as pd
import requests
from urllib.parse import quote


@dataclass
class EventRecord:
	symbol: str
	event_type: str
	event_date: date
	subject: str
	source: str


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
				resp = requests.get(url, headers=headers, timeout=request_timeout)
				resp.raise_for_status()
				content = resp.text

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


def load_nifty500_stock_symbols() -> List[str]:
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


def resolve_anchor_date() -> date:
	user_value = input("Enter anchor date (dd/mm/yyyy): ").strip()
	if not user_value:
		raise ValueError("Anchor date is required.")
	return _parse_ddmmyyyy(user_value)


def resolve_days_ahead(default_days: int = 10) -> int:
	user_value = input(f"Enter days ahead (default {default_days}): ").strip()
	if not user_value:
		return default_days
	try:
		return max(int(user_value), 1)
	except ValueError as exc:
		raise ValueError("Days ahead must be a positive integer.") from exc


def _parse_nse_date(value: str) -> Optional[date]:
	if not value:
		return None

	text = str(value).strip()
	if not text:
		return None

	if " " in text:
		text = text.split(" ")[0].strip()

	for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%d/%m/%Y", "%d%b%Y", "%Y-%m-%d"):
		try:
			return datetime.strptime(text, fmt).date()
		except ValueError:
			continue
	return None


def _parse_nse_datetime(value: str) -> Optional[date]:
	if not value:
		return None

	text = str(value).strip()
	if not text:
		return None

	for fmt in (
		"%d-%b-%Y %H:%M:%S",
		"%d-%m-%Y %H:%M:%S",
		"%d/%m/%Y %H:%M:%S",
		"%Y-%m-%d %H:%M:%S",
		"%Y-%m-%d",
	):
		try:
			return datetime.strptime(text, fmt).date()
		except ValueError:
			continue
	return None


def build_tradingview_link(symbol: str) -> str:
	return f"https://www.tradingview.com/chart/?symbol=NSE%3A{quote(str(symbol).upper())}"


BASE_URL = "https://www.nseindia.com"
API_BASE = "https://www.nseindia.com/api"

HEADERS = {
	"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
	"AppleWebKit/537.36 (KHTML, like Gecko) "
	"Chrome/124.0.0.0 Safari/537.36",
	"Accept": "application/json, text/plain, */*",
	"Accept-Language": "en-US,en;q=0.9",
	"Referer": "https://www.nseindia.com/",
	"Origin": "https://www.nseindia.com",
	"Connection": "keep-alive",
}


def make_session() -> requests.Session:
	session = requests.Session()
	session.headers.update(HEADERS)
	session.get(BASE_URL, timeout=15)
	return session


def get_json(session: requests.Session, url: str, retries: int = 3) -> Optional[object]:
	for attempt in range(1, retries + 1):
		try:
			resp = session.get(url, timeout=20)
			resp.raise_for_status()
			try:
				return resp.json()
			except ValueError:
				return None
		except requests.RequestException:
			if attempt < retries:
				time.sleep(2 ** attempt)
				continue
	return None


def fetch_corporate_announcements(
	session: requests.Session,
	from_date: date,
	to_date: date,
) -> List[Dict[str, object]]:
	from_q = from_date.strftime("%d-%m-%Y")
	to_q = to_date.strftime("%d-%m-%Y")
	url = f"{API_BASE}/corporate-announcements?index=equities&from_date={from_q}&to_date={to_q}"
	data = get_json(session, url)
	if not isinstance(data, list):
		return []
	return data


def fetch_corporate_actions(session: requests.Session) -> List[Dict[str, object]]:
	url = f"{API_BASE}/corporates-corporateActions?index=equities"
	data = get_json(session, url)
	if not isinstance(data, list):
		return []
	return data


def fetch_corporate_board_meetings(session: requests.Session) -> List[Dict[str, object]]:
	url = f"{API_BASE}/corporate-board-meetings?index=equities"
	data = get_json(session, url)
	if not isinstance(data, list):
		return []
	return data


def classify_event(subject: str) -> Optional[str]:
	text = subject.lower()
	if "dividend" in text:
		return "dividend"
	if (
		"financial results" in text
		or "results" in text
		or "earnings" in text
		or "quarter" in text
		or "quarterly" in text
		or "unaudited" in text
		or "audited" in text
		or "outcome of board meeting" in text
		or "board meeting" in text
		or "press release" in text
		or "general updates" in text
	):
		return "earnings"
	return None


def _parse_nse_any_date(value: object) -> Optional[date]:
	text = "" if value is None else str(value)
	return _parse_nse_datetime(text) or _parse_nse_date(text)


def _extract_announcement_event_date(row: Dict[str, object]) -> Optional[date]:
	for key in (
		"bm_date",
		"board_meeting_date",
		"boardMeetingDate",
		"meeting_date",
		"meetingDate",
		"date_of_board_meeting",
		"event_date",
		"eventDate",
		"result_date",
		"resultDate",
		"period_end_date",
		"periodEndDate",
		"period_end",
		"periodEnd",
	):
		value = row.get(key, "")
		parsed = _parse_nse_any_date(value)
		if parsed is not None:
			return parsed

	return (
		_parse_nse_datetime(row.get("sort_date", ""))
		or _parse_nse_datetime(row.get("an_dt", ""))
		or _parse_nse_date(str(row.get("anouncement_date", "") or row.get("announcementDate", "")))
	)


def extract_events(
	announcements: List[Dict[str, object]],
	actions: List[Dict[str, object]],
	board_meetings: List[Dict[str, object]],
	universe: set[str],
	window_start: date,
	window_end: date,
	debug: bool = False,
) -> List[EventRecord]:
	events: List[EventRecord] = []
	ann_counts = {
		"total": 0,
		"symbol": 0,
		"subject": 0,
		"etype": 0,
		"date": 0,
	}
	act_counts = {
		"total": 0,
		"symbol": 0,
		"subject": 0,
		"etype": 0,
		"date": 0,
	}
	bm_counts = {
		"total": 0,
		"symbol": 0,
		"subject": 0,
		"etype": 0,
		"date": 0,
	}

	for row in announcements:
		ann_counts["total"] += 1
		symbol = str(row.get("symbol", "")).strip().upper()
		if symbol not in universe:
			continue
		ann_counts["symbol"] += 1
		subject = str(row.get("subject", "") or row.get("desc", "")).strip()
		if not subject:
			continue
		ann_counts["subject"] += 1
		etype = classify_event(subject)
		if etype is None:
			continue
		ann_counts["etype"] += 1
		adate = _extract_announcement_event_date(row)
		if adate is None:
			continue
		if not (window_start <= adate <= window_end):
			continue
		ann_counts["date"] += 1
		events.append(EventRecord(symbol=symbol, event_type=etype, event_date=adate, subject=subject, source="announcement"))

	for row in actions:
		act_counts["total"] += 1
		symbol = str(row.get("symbol", "")).strip().upper()
		if symbol not in universe:
			continue
		act_counts["symbol"] += 1
		subject = str(row.get("purpose", "") or row.get("subject", "")).strip()
		if not subject:
			continue
		act_counts["subject"] += 1
		etype = classify_event(subject)
		if etype is None:
			continue
		act_counts["etype"] += 1
		date_text = str(row.get("exDate", "") or row.get("recordDate", "") or row.get("announcementDate", ""))
		event_date = _parse_nse_date(date_text)
		if event_date is None:
			continue
		if not (window_start <= event_date <= window_end):
			continue
		act_counts["date"] += 1
		events.append(EventRecord(symbol=symbol, event_type=etype, event_date=event_date, subject=subject, source="corporate_action"))

	for row in board_meetings:
		bm_counts["total"] += 1
		symbol = str(row.get("bm_symbol", "") or row.get("symbol", "")).strip().upper()
		if symbol not in universe:
			continue
		bm_counts["symbol"] += 1
		subject = str(row.get("bm_purpose", "") or row.get("bm_desc", "") or row.get("purpose", "")).strip()
		if not subject:
			continue
		bm_counts["subject"] += 1
		etype = classify_event(subject)
		if etype is None:
			continue
		bm_counts["etype"] += 1
		event_date = _parse_nse_date(str(row.get("bm_date", "") or row.get("meetingDate", "")))
		if event_date is None:
			continue
		if not (window_start <= event_date <= window_end):
			continue
		bm_counts["date"] += 1
		events.append(EventRecord(symbol=symbol, event_type=etype, event_date=event_date, subject=subject, source="board_meeting"))

	if debug:
		print("\n[DEBUG] Announcement filter counts:")
		print(ann_counts)
		print("[DEBUG] Corporate action filter counts:")
		print(act_counts)
		print("[DEBUG] Board meeting filter counts:")
		print(bm_counts)
		print(f"[DEBUG] Total matchedx events: {len(events)}")

	return events


def run_screen(
	symbols: Sequence[str],
	as_of_date: date,
	days_ahead: int,
) -> pd.DataFrame:
	window_start = as_of_date
	window_end = as_of_date + timedelta(days=days_ahead)

	session = make_session()
	announcements = fetch_corporate_announcements(session, window_start, window_end)
	actions = fetch_corporate_actions(session)
	board_meetings = fetch_corporate_board_meetings(session)
	events = extract_events(
		announcements=announcements,
		actions=actions,
		board_meetings=board_meetings,
		universe=set(symbols),
		window_start=window_start,
		window_end=window_end,
		debug=False,
	)

	if not events:
		return pd.DataFrame(columns=[
			"symbol",
			"event_type",
			"event_date",
			"days_to_event",
			"subject",
			"source",
			"tradingview_link",
		])

	rows = []
	for e in events:
		rows.append({
			"symbol": e.symbol,
			"event_type": e.event_type,
			"event_date": e.event_date.strftime("%d/%m/%Y"),
			"days_to_event": (e.event_date - as_of_date).days,
			"subject": e.subject,
			"source": e.source,
			"tradingview_link": build_tradingview_link(e.symbol),
		})

	return (
		pd.DataFrame(rows)
		.sort_values(["event_date", "event_type", "symbol"], ascending=[True, True, True])
		.reset_index(drop=True)
	)


def build_output_name(as_of_date: date) -> str:
	return f"earnings_dividends_upcoming_{as_of_date.strftime('%d_%m_%Y')}.csv"


def get_output_dir() -> str:
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	script_name = os.path.splitext(os.path.basename(__file__))[0]
	output_dir = os.path.join(base_dir, "Output", script_name)
	os.makedirs(output_dir, exist_ok=True)
	return output_dir


def main() -> None:
	as_of_date = resolve_anchor_date()
	days_ahead = resolve_days_ahead()
	symbols = load_nifty500_stock_symbols()
	print(f"Running on all NIFTY 500 symbols: {len(symbols)}")

	results = run_screen(
		symbols=symbols,
		as_of_date=as_of_date,
		days_ahead=days_ahead,
	)

	output_dir = get_output_dir()
	output_path = os.path.join(output_dir, build_output_name(as_of_date))
	results.to_csv(output_path, index=False)

	print(f"Anchor date: {as_of_date.strftime('%d/%m/%Y')}")
	print(f"Output file: {output_path}")
	print("\n=== Upcoming Earnings/Dividends (<= N days) ===")
	if results.empty:
		print("No candidates")
	else:
		print(", ".join(results["symbol"].tolist()))
	print(f"Count: {len(results)}")


if __name__ == "__main__":
	main()
