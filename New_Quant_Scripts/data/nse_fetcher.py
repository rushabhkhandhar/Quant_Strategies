import os
import time
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Any, Dict, List, Optional, Sequence
from urllib.request import Request, urlopen
import pandas as pd

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

    for offset in range(0, lookback_days + 1):
        day = as_of_date - timedelta(days=offset)
        df = _download_bhavcopy_for_date(day)
        if df is None or df.empty:
            continue

        symbol_col = _find_column(df.columns.tolist(), ["SYMBOL"])
        series_col = _find_column(df.columns.tolist(), ["SERIES"])
        open_col = _find_column(df.columns.tolist(), ["OPEN_PRICE", "OPEN"])
        high_col = _find_column(df.columns.tolist(), ["HIGH_PRICE", "HIGH"])
        low_col = _find_column(df.columns.tolist(), ["LOW_PRICE", "LOW"])
        close_col = _find_column(df.columns.tolist(), ["CLOSE_PRICE", "CLOSE", "CLOSE_PRICE_"])
        volume_col = _find_column(df.columns.tolist(), ["TOTTRDQTY", "TTL_TRD_QNTY", "VOLUME", "TOTTRD_QTY"])
        date_col = _find_column(df.columns.tolist(), ["DATE1", "DATE", "TIMESTAMP"])

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
