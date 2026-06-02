from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Sequence
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd


@dataclass
class CandleSet:
    weekly: pd.DataFrame
    daily: pd.DataFrame


_BHAVCOPY_CACHE: Dict[str, Optional[pd.DataFrame]] = {}
_SYMBOL_DAILY_CACHE: Dict[tuple[str, int, tuple[str, ...]], Dict[str, pd.DataFrame]] = {}


@dataclass
class StrategyExecution:
    name: str
    results: pd.DataFrame
    bullish: pd.DataFrame
    bearish: pd.DataFrame


@dataclass
class StrategySpec:
    name: str
    runner: Callable[..., StrategyExecution]


def _find_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lookup = {str(c).strip().upper(): c for c in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


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


def _iter_dates(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def _build_daily_map_for_symbols(
    symbols: Sequence[str],
    as_of_date: date,
    max_lookback_days: int,
) -> Dict[str, pd.DataFrame]:
    symbols_upper = [str(s).strip().upper() for s in symbols if str(s).strip()]
    sorted_key = tuple(sorted(set(symbols_upper)))
    cache_key = (as_of_date.strftime("%Y-%m-%d"), int(max_lookback_days), sorted_key)
    if cache_key in _SYMBOL_DAILY_CACHE:
        return _SYMBOL_DAILY_CACHE[cache_key]

    symbol_set = set(symbols_upper)
    rows_by_symbol: Dict[str, List[Dict[str, object]]] = {sym: [] for sym in symbols_upper}

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
        date_col = _find_column(df.columns, ["DATE1", "DATE", "TIMESTAMP"])

        if not all([symbol_col, series_col, open_col, high_col, low_col, close_col, date_col]):
            continue

        work = df[[symbol_col, series_col, open_col, high_col, low_col, close_col, date_col]].copy()
        work = work.rename(
            columns={
                symbol_col: "symbol",
                series_col: "series",
                open_col: "Open",
                high_col: "High",
                low_col: "Low",
                close_col: "Close",
                date_col: "Date",
            }
        )

        work["symbol"] = work["symbol"].astype(str).str.strip().str.upper()
        work["series"] = work["series"].astype(str).str.strip().str.upper()
        work = work[(work["series"] == "EQ") & (work["symbol"].isin(symbol_set))]
        if work.empty:
            continue

        work["Date"] = pd.to_datetime(work["Date"], errors="coerce", dayfirst=True).dt.normalize()
        for col in ["Open", "High", "Low", "Close"]:
            work[col] = pd.to_numeric(work[col], errors="coerce")

        work = work.dropna(subset=["Date", "Open", "High", "Low", "Close"])
        if work.empty:
            continue

        for sym, grp in work.groupby("symbol"):
            rows_by_symbol[sym].extend(grp[["Date", "Open", "High", "Low", "Close"]].to_dict("records"))

    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols_upper:
        rows = rows_by_symbol.get(sym, [])
        if not rows:
            out[sym] = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
            continue

        out[sym] = (
            pd.DataFrame(rows)
            .sort_values("Date")
            .drop_duplicates(subset=["Date"], keep="last")
            .set_index("Date")
        )

    _SYMBOL_DAILY_CACHE[cache_key] = out
    return out


def _fetch_daily_from_bhavcopy(symbol: str, as_of_date: date, max_lookback_days: int) -> pd.DataFrame:
    prebuilt = _build_daily_map_for_symbols([symbol], as_of_date, max_lookback_days)
    return prebuilt.get(symbol.strip().upper(), pd.DataFrame(columns=["Open", "High", "Low", "Close"]))


def _build_candles_from_daily(daily: pd.DataFrame) -> Optional[CandleSet]:
    if daily.empty:
        return None

    weekly = (
        daily.resample("W-FRI")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna(subset=["Open", "High", "Low", "Close"])
    )

    if len(weekly) < 3 or len(daily) < 3:
        return None

    return CandleSet(weekly=weekly, daily=daily)


def _fetch_daily_from_bhavcopy_legacy(symbol: str, as_of_date: date, max_lookback_days: int) -> pd.DataFrame:
    symbol = symbol.strip().upper()
    rows: List[Dict[str, object]] = []

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
        date_col = _find_column(df.columns, ["DATE1", "DATE", "TIMESTAMP"])

        if not all([symbol_col, series_col, open_col, high_col, low_col, close_col, date_col]):
            continue

        work = df[[symbol_col, series_col, open_col, high_col, low_col, close_col, date_col]].copy()
        work = work.rename(
            columns={
                symbol_col: "symbol",
                series_col: "series",
                open_col: "Open",
                high_col: "High",
                low_col: "Low",
                close_col: "Close",
                date_col: "Date",
            }
        )

        work["symbol"] = work["symbol"].astype(str).str.strip().str.upper()
        work["series"] = work["series"].astype(str).str.strip().str.upper()
        work = work[(work["symbol"] == symbol) & (work["series"] == "EQ")]
        if work.empty:
            continue

        work["Date"] = pd.to_datetime(work["Date"], errors="coerce", dayfirst=True).dt.normalize()
        for col in ["Open", "High", "Low", "Close"]:
            work[col] = pd.to_numeric(work[col], errors="coerce")

        work = work.dropna(subset=["Date", "Open", "High", "Low", "Close"])
        if work.empty:
            continue

        rows.extend(work[["Date", "Open", "High", "Low", "Close"]].to_dict("records"))

    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

    return (
        pd.DataFrame(rows)
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .set_index("Date")
    )


def _fetch_candles_weekly_daily(symbol: str, as_of_date: date, max_lookback_days: int = 420) -> Optional[CandleSet]:
    daily = _fetch_daily_from_bhavcopy(symbol=symbol, as_of_date=as_of_date, max_lookback_days=max_lookback_days)
    return _build_candles_from_daily(daily)


def _extract_weekly_daily_points(candles: CandleSet) -> Dict[str, float]:
    w = candles.weekly
    d = candles.daily
    w_curr = w.iloc[-1]
    w1 = w.iloc[-2]
    w2 = w.iloc[-3]
    d1 = d.iloc[-1]
    d2 = d.iloc[-2]

    return {
        "w1_high": float(w1["High"]),
        "w1_low": float(w1["Low"]),
        "w1_close": float(w1["Close"]),
        "w2_high": float(w2["High"]),
        "w2_low": float(w2["Low"]),
        "weekly_high": float(w_curr["High"]),
        "weekly_low": float(w_curr["Low"]),
        "d1_high": float(d1["High"]),
        "d1_low": float(d1["Low"]),
        "d1_close": float(d1["Close"]),
        "d2_high": float(d2["High"]),
        "d2_low": float(d2["Low"]),
        "d2_close": float(d2["Close"]),
    }


def _weekly_pattern_1(values: Dict[str, float]) -> bool:
    return (
        values["w1_high"] > values["w2_high"]
        and values["w1_close"] < values["w2_high"]
        and values["w1_low"] > values["w2_low"]
        and values["weekly_low"] > values["w1_low"]
        and values["d1_low"] < values["d2_low"]
        and values["d1_close"] > values["d2_low"]
    )


def _weekly_pattern_2(values: Dict[str, float]) -> bool:
    return (
        values["w1_low"] < values["w2_low"]
        and values["w1_close"] > values["w2_low"]
        and values["w1_high"] < values["w2_high"]
        and values["weekly_high"] < values["w1_high"]
        and values["d1_high"] > values["d2_high"]
        and values["d1_close"] < values["d2_close"]
    )


def _inside_bar_points(daily: pd.DataFrame) -> Dict[str, float]:
    curr = daily.iloc[-1]
    d1 = daily.iloc[-2]
    d2 = daily.iloc[-3]

    return {
        "curr_high": float(curr["High"]),
        "curr_low": float(curr["Low"]),
        "curr_close": float(curr["Close"]),
        "d1_open": float(d1["Open"]),
        "d1_high": float(d1["High"]),
        "d1_low": float(d1["Low"]),
        "d1_close": float(d1["Close"]),
        "d2_open": float(d2["Open"]),
        "d2_high": float(d2["High"]),
        "d2_low": float(d2["Low"]),
        "d2_close": float(d2["Close"]),
    }


def _inside_bar_bullish(v: Dict[str, float]) -> bool:
    return (
        v["d2_open"] < v["d2_close"]
        and abs(v["d2_close"] - v["d2_open"]) > abs(v["d2_high"] - v["d2_low"]) * 0.6
        and v["d1_high"] <= v["d2_high"]
        and v["d1_low"] >= v["d2_low"]
        and v["curr_low"] < v["d1_low"]
        and v["curr_high"] < v["d1_high"]
        and v["curr_close"] > v["d1_low"]
    )


def _inside_bar_bearish(v: Dict[str, float]) -> bool:
    return (
        v["d2_open"] > v["d2_close"]
        and abs(v["d2_open"] - v["d2_close"]) > abs(v["d2_high"] - v["d2_low"]) * 0.6
        and v["d1_high"] < v["d2_high"]
        and v["d1_low"] > v["d2_low"]
        and v["curr_high"] > v["d1_high"]
        and v["curr_low"] > v["d1_low"]
        and v["curr_close"] < v["d1_high"]
    )


def _double_sweep_points(daily: pd.DataFrame) -> Dict[str, float]:
    d1 = daily.iloc[-1]
    d2 = daily.iloc[-2]
    d3 = daily.iloc[-3]

    return {
        "d1_open": float(d1["Open"]),
        "d1_high": float(d1["High"]),
        "d1_low": float(d1["Low"]),
        "d1_close": float(d1["Close"]),
        "d2_open": float(d2["Open"]),
        "d2_high": float(d2["High"]),
        "d2_low": float(d2["Low"]),
        "d2_close": float(d2["Close"]),
        "d3_open": float(d3["Open"]),
        "d3_high": float(d3["High"]),
        "d3_low": float(d3["Low"]),
        "d3_close": float(d3["Close"]),
    }


def _double_sweep_bullish(v: Dict[str, float]) -> bool:
    return (
        v["d1_high"] > v["d2_high"]
        and v["d1_low"] > v["d2_low"]
        and v["d1_close"] < v["d2_high"]
        and v["d1_open"] < v["d2_high"]
        and v["d2_high"] > v["d3_high"]
        and v["d2_low"] > v["d3_low"]
        and v["d2_close"] < v["d3_high"]
        and v["d2_open"] < v["d3_high"]
    )


def _double_sweep_bearish(v: Dict[str, float]) -> bool:
    return (
        v["d1_low"] < v["d2_low"]
        and v["d1_high"] < v["d2_high"]
        and v["d1_close"] > v["d2_low"]
        and v["d1_open"] > v["d2_low"]
        and v["d2_low"] < v["d3_low"]
        and v["d2_high"] < v["d3_high"]
        and v["d2_close"] > v["d3_low"]
        and v["d2_open"] > v["d3_low"]
    )


def _daily_fvg_sweep_points(daily: pd.DataFrame) -> Dict[str, float]:
    curr = daily.iloc[-1]
    d1 = daily.iloc[-2]
    d2 = daily.iloc[-3]
    d3 = daily.iloc[-4]

    return {
        "curr_open": float(curr["Open"]),
        "curr_high": float(curr["High"]),
        "curr_low": float(curr["Low"]),
        "curr_close": float(curr["Close"]),
        "d1_open": float(d1["Open"]),
        "d1_high": float(d1["High"]),
        "d1_low": float(d1["Low"]),
        "d1_close": float(d1["Close"]),
        "d2_open": float(d2["Open"]),
        "d2_high": float(d2["High"]),
        "d2_low": float(d2["Low"]),
        "d2_close": float(d2["Close"]),
        "d3_open": float(d3["Open"]),
        "d3_high": float(d3["High"]),
        "d3_low": float(d3["Low"]),
        "d3_close": float(d3["Close"]),
    }


def _daily_fvg_sweep_bullish(v: Dict[str, float]) -> bool:
    return (
        (v["d3_low"] - v["d1_high"]) > (v["curr_close"] * 0.01)
        and v["d3_low"] > v["d2_low"]
        and v["d2_low"] > v["d1_low"]
        and v["d2_high"] < v["d3_high"]
        and v["d1_high"] < v["d2_high"]
        and v["curr_high"] > v["d1_high"]
        and v["curr_close"] < v["d1_high"]
        and v["curr_low"] > v["d1_low"]
    )


def _daily_fvg_sweep_bearish(v: Dict[str, float]) -> bool:
    return (
        (v["d3_high"] - v["d1_low"]) < (v["curr_close"] * 0.01)
        and v["d3_high"] < v["d2_high"]
        and v["d3_high"] < v["d1_low"]
        and v["d2_high"] < v["d1_high"]
        and v["d2_low"] > v["d3_low"]
        and v["d1_low"] > v["d2_low"]
        and v["curr_low"] < v["d1_low"]
        and v["curr_close"] > v["d1_low"]
        and v["curr_high"] < v["d1_high"]
    )


def _ema5_sweep_points(daily: pd.DataFrame) -> Dict[str, float]:
    work = daily.copy()
    work["ema5"] = work["Close"].ewm(span=5, adjust=False, min_periods=5).mean()
    work = work.dropna(subset=["ema5"])

    curr = work.iloc[-1]
    prev = work.iloc[-2]

    return {
        "curr_open": float(curr["Open"]),
        "curr_high": float(curr["High"]),
        "curr_low": float(curr["Low"]),
        "curr_close": float(curr["Close"]),
        "curr_ema5": float(curr["ema5"]),
        "prev_open": float(prev["Open"]),
        "prev_high": float(prev["High"]),
        "prev_low": float(prev["Low"]),
        "prev_close": float(prev["Close"]),
        "prev_ema5": float(prev["ema5"]),
    }


def _ema5_sweep_bullish(v: Dict[str, float]) -> bool:
    return (
        v["curr_low"] > v["curr_ema5"]
        and v["prev_low"] > v["prev_ema5"]
        and v["curr_high"] > v["prev_high"]
        and v["curr_close"] < v["prev_high"]
    )


def _ema5_sweep_bearish(v: Dict[str, float]) -> bool:
    return (
        v["curr_high"] < v["curr_ema5"]
        and v["prev_high"] < v["prev_ema5"]
        and v["curr_low"] < v["prev_low"]
        and v["curr_close"] > v["prev_low"]
    )


def _build_tradingview_link(symbol: str) -> str:
    return f"https://www.tradingview.com/chart/?symbol=NSE%3A{quote(str(symbol).upper())}"


def _extract_signal_frames(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bullish = (
        results.loc[results["bullish_match"] == True, ["symbol"]]
        .sort_values("symbol")
        .reset_index(drop=True)
    )
    bearish = (
        results.loc[results["bearish_match"] == True, ["symbol"]]
        .sort_values("symbol")
        .reset_index(drop=True)
    )

    for frame in (bullish, bearish):
        frame["tradingview_link"] = frame["symbol"].apply(_build_tradingview_link)

    return bullish, bearish


def run_weekly_vs_daily(
    symbols: Sequence[str],
    as_of_date: date,
    verbose: bool = False,
    print_values: bool = False,
    daily_map: Optional[Dict[str, pd.DataFrame]] = None,
) -> StrategyExecution:
    results = pd.DataFrame(
        {
            "symbol": list(symbols),
            "bearish_match": False,
            "bullish_match": pd.NA,
            "final_signal": pd.NA,
            "status": "pending",
        }
    )

    values_cache: Dict[str, Dict[str, float]] = {}

    for idx, symbol in enumerate(symbols):
        if daily_map is None:
            daily = _fetch_daily_from_bhavcopy(symbol=symbol, as_of_date=as_of_date, max_lookback_days=420)
        else:
            daily = daily_map.get(str(symbol).upper(), pd.DataFrame(columns=["Open", "High", "Low", "Close"]))

        candles = _build_candles_from_daily(daily)
        if candles is None:
            results.at[idx, "status"] = "no_data"
            if verbose:
                print(f"{symbol}: bearish=SKIPPED (no_data)")
            continue

        values = _extract_weekly_daily_points(candles)
        values_cache[str(symbol)] = values

        if print_values:
            print(f"\n{symbol} extracted values:")
            for key in sorted(values.keys()):
                print(f"  {key}: {values[key]:.2f}")

        bearish = _weekly_pattern_2(values)
        results.at[idx, "bearish_match"] = bearish
        results.at[idx, "status"] = "bearish_done"

        if verbose:
            print(f"{symbol}: bearish={bearish}")

    for idx, symbol in enumerate(symbols):
        if str(symbol) not in values_cache:
            continue

        values = values_cache[str(symbol)]
        bullish = _weekly_pattern_1(values)
        bearish = bool(results.at[idx, "bearish_match"])

        results.at[idx, "bullish_match"] = bullish
        results.at[idx, "final_signal"] = bullish or bearish
        results.at[idx, "status"] = "complete"

        if verbose:
            print(f"{symbol}: bullish={bullish}, final_signal={bullish or bearish}")

    bullish, bearish = _extract_signal_frames(results)
    return StrategyExecution(
        name="weekly_vs_daily_sweep",
        results=results,
        bullish=bullish,
        bearish=bearish,
    )


def run_inside_bar_daily_sweep(
    symbols: Sequence[str],
    as_of_date: date,
    verbose: bool = False,
    print_values: bool = False,
    daily_map: Optional[Dict[str, pd.DataFrame]] = None,
) -> StrategyExecution:
    _ = print_values
    results = pd.DataFrame(
        {
            "symbol": list(symbols),
            "bullish_match": False,
            "bearish_match": False,
            "final_signal": False,
            "status": "pending",
        }
    )

    for idx, symbol in enumerate(symbols):
        if daily_map is None:
            daily = _fetch_daily_from_bhavcopy(symbol=symbol, as_of_date=as_of_date, max_lookback_days=160)
        else:
            daily = daily_map.get(str(symbol).upper(), pd.DataFrame(columns=["Open", "High", "Low", "Close"]))

        if len(daily) < 4:
            results.at[idx, "status"] = "no_data"
            if verbose:
                print(f"{symbol}: SKIPPED (no_data)")
            continue

        values = _inside_bar_points(daily)
        bullish = _inside_bar_bullish(values)
        bearish = _inside_bar_bearish(values)

        results.at[idx, "bullish_match"] = bullish
        results.at[idx, "bearish_match"] = bearish
        results.at[idx, "final_signal"] = bullish or bearish
        results.at[idx, "status"] = "complete"

        if verbose:
            print(f"{symbol}: bullish={bullish}, bearish={bearish}")

    bullish, bearish = _extract_signal_frames(results)
    return StrategyExecution(
        name="inside_bar_pattern_daily_sweep",
        results=results,
        bullish=bullish,
        bearish=bearish,
    )


def run_double_sweep(
    symbols: Sequence[str],
    as_of_date: date,
    verbose: bool = False,
    print_values: bool = False,
    daily_map: Optional[Dict[str, pd.DataFrame]] = None,
) -> StrategyExecution:
    _ = print_values
    results = pd.DataFrame(
        {
            "symbol": list(symbols),
            "bullish_match": False,
            "bearish_match": False,
            "final_signal": False,
            "status": "pending",
        }
    )

    for idx, symbol in enumerate(symbols):
        if daily_map is None:
            daily = _fetch_daily_from_bhavcopy(symbol=symbol, as_of_date=as_of_date, max_lookback_days=60)
        else:
            daily = daily_map.get(str(symbol).upper(), pd.DataFrame(columns=["Open", "High", "Low", "Close"]))

        if len(daily) < 3:
            results.at[idx, "status"] = "no_data"
            if verbose:
                print(f"{symbol}: SKIPPED (no_data)")
            continue

        values = _double_sweep_points(daily)
        bullish = _double_sweep_bullish(values)
        bearish = _double_sweep_bearish(values)

        results.at[idx, "bullish_match"] = bullish
        results.at[idx, "bearish_match"] = bearish
        results.at[idx, "final_signal"] = bullish or bearish
        results.at[idx, "status"] = "complete"

        if verbose:
            print(f"{symbol}: bullish={bullish}, bearish={bearish}")

    bullish, bearish = _extract_signal_frames(results)
    return StrategyExecution(
        name="double_sweep",
        results=results,
        bullish=bullish,
        bearish=bearish,
    )


def run_daily_fvg_sweep(
    symbols: Sequence[str],
    as_of_date: date,
    verbose: bool = False,
    print_values: bool = False,
    daily_map: Optional[Dict[str, pd.DataFrame]] = None,
) -> StrategyExecution:
    _ = print_values
    results = pd.DataFrame(
        {
            "symbol": list(symbols),
            "bullish_match": False,
            "bearish_match": False,
            "final_signal": False,
            "status": "pending",
        }
    )

    for idx, symbol in enumerate(symbols):
        if daily_map is None:
            daily = _fetch_daily_from_bhavcopy(symbol=symbol, as_of_date=as_of_date, max_lookback_days=80)
        else:
            daily = daily_map.get(str(symbol).upper(), pd.DataFrame(columns=["Open", "High", "Low", "Close"]))

        if len(daily) < 4:
            results.at[idx, "status"] = "no_data"
            if verbose:
                print(f"{symbol}: SKIPPED (no_data)")
            continue

        values = _daily_fvg_sweep_points(daily)
        bullish = _daily_fvg_sweep_bullish(values)
        bearish = _daily_fvg_sweep_bearish(values)

        results.at[idx, "bullish_match"] = bullish
        results.at[idx, "bearish_match"] = bearish
        results.at[idx, "final_signal"] = bullish or bearish
        results.at[idx, "status"] = "complete"

        if verbose:
            print(f"{symbol}: bullish={bullish}, bearish={bearish}")

    bullish, bearish = _extract_signal_frames(results)
    return StrategyExecution(
        name="daily_fvg_sweep",
        results=results,
        bullish=bullish,
        bearish=bearish,
    )


def run_ema5_sweep(
    symbols: Sequence[str],
    as_of_date: date,
    verbose: bool = False,
    print_values: bool = False,
    daily_map: Optional[Dict[str, pd.DataFrame]] = None,
) -> StrategyExecution:
    _ = print_values
    results = pd.DataFrame(
        {
            "symbol": list(symbols),
            "bullish_match": False,
            "bearish_match": False,
            "final_signal": False,
            "status": "pending",
        }
    )

    for idx, symbol in enumerate(symbols):
        if daily_map is None:
            daily = _fetch_daily_from_bhavcopy(symbol=symbol, as_of_date=as_of_date, max_lookback_days=40)
        else:
            daily = daily_map.get(str(symbol).upper(), pd.DataFrame(columns=["Open", "High", "Low", "Close"]))

        if len(daily) < 6:
            results.at[idx, "status"] = "no_data"
            if verbose:
                print(f"{symbol}: SKIPPED (no_data)")
            continue

        values = _ema5_sweep_points(daily)
        bullish = _ema5_sweep_bullish(values)
        bearish = _ema5_sweep_bearish(values)

        results.at[idx, "bullish_match"] = bullish
        results.at[idx, "bearish_match"] = bearish
        results.at[idx, "final_signal"] = bullish or bearish
        results.at[idx, "status"] = "complete"

        if verbose:
            print(f"{symbol}: bullish={bullish}, bearish={bearish}")

    bullish, bearish = _extract_signal_frames(results)
    return StrategyExecution(
        name="ema5_sweep",
        results=results,
        bullish=bullish,
        bearish=bearish,
    )


def strategy_registry() -> Dict[str, StrategySpec]:
    return {
        "weekly_vs_daily_sweep": StrategySpec(
            name="weekly_vs_daily_sweep",
            runner=run_weekly_vs_daily,
        ),
        "inside_bar_pattern_daily_sweep": StrategySpec(
            name="inside_bar_pattern_daily_sweep",
            runner=run_inside_bar_daily_sweep,
        ),
        "double_sweep": StrategySpec(
            name="double_sweep",
            runner=run_double_sweep,
        ),
        "daily_fvg_sweep": StrategySpec(
            name="daily_fvg_sweep",
            runner=run_daily_fvg_sweep,
        ),
        "ema5_sweep": StrategySpec(
            name="ema5_sweep",
            runner=run_ema5_sweep,
        ),
    }


def load_default_symbols() -> List[str]:
    url = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,text/plain,*/*",
    }
    request = Request(url, headers=headers)
    with urlopen(request, timeout=20) as response:
        content = response.read().decode("utf-8", errors="ignore")

    raw = pd.read_csv(
        StringIO(content),
        skipinitialspace=True,
        engine="python",
        on_bad_lines="skip",
    )
    raw.columns = [str(c).strip().upper() for c in raw.columns]

    if "SYMBOL" not in raw.columns:
        raise RuntimeError("Unable to find SYMBOL column in NSE F&O market lot CSV.")

    symbols = raw["SYMBOL"].astype(str).str.strip().str.upper()
    symbols = symbols[symbols.str.match(r"^[A-Z0-9&\-]+$")]
    symbols = symbols[symbols != "SYMBOL"]

    index_symbols = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}
    symbols = symbols[~symbols.isin(index_symbols)]

    unique_sorted = sorted(set(symbols.tolist()))
    if not unique_sorted:
        raise RuntimeError("Online NSE futures stock list returned no valid symbols.")

    return unique_sorted


def run_strategies(
    strategy_names: Sequence[str],
    symbols: Sequence[str],
    as_of_date: date,
    verbose: bool = False,
    print_values: bool = False,
    parallel: bool = True,
) -> List[StrategyExecution]:
    registry = strategy_registry()
    for name in strategy_names:
        if name not in registry:
            valid = ", ".join(sorted(registry.keys()))
            raise ValueError(f"Unknown strategy '{name}'. Valid: {valid}")

    lookback_by_strategy = {
        "weekly_vs_daily_sweep": 420,
        "inside_bar_pattern_daily_sweep": 160,
        "double_sweep": 60,
        "daily_fvg_sweep": 80,
        "ema5_sweep": 40,
    }
    max_lookback = max(lookback_by_strategy.get(name, 60) for name in strategy_names) if strategy_names else 60
    daily_map = _build_daily_map_for_symbols(symbols=symbols, as_of_date=as_of_date, max_lookback_days=max_lookback)

    if parallel and len(strategy_names) > 1:
        results_by_name: Dict[str, StrategyExecution] = {}
        with ThreadPoolExecutor(max_workers=min(4, len(strategy_names))) as executor:
            future_to_name = {
                executor.submit(
                    registry[name].runner,
                    symbols=symbols,
                    as_of_date=as_of_date,
                    verbose=verbose,
                    print_values=print_values,
                    daily_map=daily_map,
                ): name
                for name in strategy_names
            }

            for future in as_completed(future_to_name):
                name = future_to_name[future]
                results_by_name[name] = future.result()

        return [results_by_name[name] for name in strategy_names]

    executions: List[StrategyExecution] = []
    for name in strategy_names:
        execution = registry[name].runner(
            symbols=symbols,
            as_of_date=as_of_date,
            verbose=verbose,
            print_values=print_values,
            daily_map=daily_map,
        )
        executions.append(execution)

    return executions
