import os
import time
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Any, Dict, List, Optional, Sequence
from urllib.request import Request, urlopen
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from core.models import CandleSet

# Global in-memory cache to avoid repeated network requests in the same run
_BHAVCOPY_CACHE: Dict[str, Optional[pd.DataFrame]] = {}

def _download_bhavcopy_for_date(trade_date: date) -> Optional[pd.DataFrame]:
    """Download one NSE full bhavcopy day (or load from in-memory cache)."""
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
        
        # PRE-PROCESS ONCE: Extract required columns and set index for O(1) lookups
        symbol_col = _find_column(df.columns.tolist(), ["SYMBOL"])
        series_col = _find_column(df.columns.tolist(), ["SERIES"])
        open_col = _find_column(df.columns.tolist(), ["OPEN_PRICE", "OPEN"])
        high_col = _find_column(df.columns.tolist(), ["HIGH_PRICE", "HIGH"])
        low_col = _find_column(df.columns.tolist(), ["LOW_PRICE", "LOW"])
        close_col = _find_column(df.columns.tolist(), ["CLOSE_PRICE", "CLOSE", "CLOSE_PRICE_"])
        volume_col = _find_column(df.columns.tolist(), ["TOTTRDQTY", "TTL_TRD_QNTY", "VOLUME", "TOTTRD_QTY"])
        
        if not (symbol_col and series_col and open_col and high_col and low_col and close_col):
            _BHAVCOPY_CACHE[key] = None
            return None
            
        df[symbol_col] = df[symbol_col].astype(str).str.strip().str.upper()
        df[series_col] = df[series_col].astype(str).str.strip().str.upper()
        
        # Filter EQ only
        df = df[df[series_col] == "EQ"]
        
        # Rename and keep only necessary columns
        rename_dict = {
            open_col: "OPEN", high_col: "HIGH", low_col: "LOW", close_col: "CLOSE"
        }
        if volume_col:
            rename_dict[volume_col] = "VOLUME"
            
        df = df.rename(columns=rename_dict)
        if "VOLUME" not in df.columns:
            df["VOLUME"] = 0.0
            
        # Set symbol as index for instant lookups
        df = df.set_index(symbol_col)[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
        df = df.apply(pd.to_numeric, errors="coerce")
        
        _BHAVCOPY_CACHE[key] = df
        return df
    except Exception:
        # Save a marker for missing data to avoid re-requesting empty weekends/holidays
        _BHAVCOPY_CACHE[key] = None
        return None

def _find_column(columns: Any, candidates: Sequence[str]) -> Optional[str]:
    lookup = {str(c).strip().upper(): c for c in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None

def fetch_daily_candles(symbol: str, as_of_date: date, lookback_days: int = 320) -> Optional[CandleSet]:
    """Fetch a history of daily candles for a given symbol up to as_of_date."""
    rows: List[Dict[str, Any]] = []

    # 1. Pre-fetch all missing days in parallel to drastically speed up network I/O
    days_to_fetch = [as_of_date - timedelta(days=offset) for offset in range(lookback_days + 1)]
    missing_days = [d for d in days_to_fetch if d.strftime("%Y-%m-%d") not in _BHAVCOPY_CACHE]
    
    if missing_days:
        with ThreadPoolExecutor(max_workers=20) as executor:
            list(executor.map(_download_bhavcopy_for_date, missing_days))

    # 2. Extract data sequentially (all requests will now instantly hit the RAM cache)
    for day in days_to_fetch:
        df = _download_bhavcopy_for_date(day)
        if df is None or df.empty:
            continue

        if symbol in df.index:
            r = df.loc[symbol]
            # Some Bhavcopies have duplicate EQ entries accidentally, take the first one
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
                
            if pd.isna(r["OPEN"]) or pd.isna(r["CLOSE"]):
                continue
                
            rows.append({
                "Date": pd.to_datetime(day),
                "Open": float(r["OPEN"]),
                "High": float(r["HIGH"]),
                "Low": float(r["LOW"]),
                "Close": float(r["CLOSE"]),
                "Volume": float(r["VOLUME"]) if not pd.isna(r["VOLUME"]) else 0.0
            })

    if not rows:
        return None

    daily = (
        pd.DataFrame(rows)
        .dropna(subset=["Date", "Open", "High", "Low", "Close"])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .set_index("Date")
    )

    if len(daily) < 10:  # Minimum valid days to form a CandleSet
        return None

    return CandleSet(symbol=symbol, daily=daily)

# --- Symbol Loaders ---

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

def load_nifty500_symbols() -> List[str]:
    """Fetch NIFTY 500 constituent symbols."""
    urls = [
        "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv",
        "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    ]
    return _load_symbols_from_csv_urls(urls, column_name="Symbol")
