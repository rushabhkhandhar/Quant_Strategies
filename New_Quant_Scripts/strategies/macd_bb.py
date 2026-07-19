from typing import Optional
import pandas as pd

from core.models import CandleSet, Signal
from strategies.base import BaseStrategy

class MacdBbStrategy(BaseStrategy):
    """
    Trend Shift + Volatility Expansion Strategy.
    
    Rules:
    1) Bollinger Bands (20,2) width has been narrow (<8%) over last 15 days.
    2) MACD (12, 26, 9) crosses over today.
    3) Price is near the outer band (<2% distance).
    4) Volume is above the 15-day average.
    """
    
    def __init__(self):
        super().__init__()
        self.bb_narrow_lookback = 15
        self.bb_width_max_pct = 8.0
        self.volume_multiplier = 1.0
        self.near_breakout_pct = 2.0

    @property
    def name(self) -> str:
        return "MACD_BB_Expansion"

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # MACD (12, 26, 9)
        if "macd" not in df.columns:
            ema12 = df["Close"].ewm(span=12, adjust=False).mean()
            ema26 = df["Close"].ewm(span=26, adjust=False).mean()
            df["macd"] = ema12 - ema26
            df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["hist"] = df["macd"] - df["signal"]

        # Bollinger Bands (20, 2)
        if "bb_mid" not in df.columns:
            middle = df["Close"].rolling(20).mean()
            std = df["Close"].rolling(20).std(ddof=0)
            df["bb_mid"] = middle
            df["bb_upper"] = middle + (2.0 * std)
            df["bb_lower"] = middle - (2.0 * std)
            df["bb_width_pct"] = ((df["bb_upper"] - df["bb_lower"]) / middle.replace(0, pd.NA)) * 100.0

        # ATR (14) for Targets
        if "ATR" not in df.columns:
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df['Close'].shift()).abs()
            low_close = (df['Low'] - df['Close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(window=14).mean()

        return df

    def analyze(self, candles: CandleSet) -> Optional[Signal]:
        df = candles.daily
        if len(df) < max(60, self.bb_narrow_lookback + 25):
            return None

        # Ensure indicators are calculated
        df = self.prepare_data(df)
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # bb_prev represents the last 15 days, EXCLUDING today
        bb_prev = df.iloc[-(self.bb_narrow_lookback + 1):-1]
        
        # Volatility Contraction Check
        avg_width = float(bb_prev["bb_width_pct"].mean())
        if avg_width > self.bb_width_max_pct:
            return None

        # Volume Expansion Check
        avg_vol = float(bb_prev["Volume"].mean())
        curr_vol = float(curr["Volume"])
        if avg_vol > 0 and curr_vol < (self.volume_multiplier * avg_vol):
            return None

        # Liquidity Filter (Approx 70 Cr based on recent volume)
        if (avg_vol * float(curr["Close"])) < 70_000_000:
            return None

        # MACD Crossover
        bull_cross = float(curr["macd"]) > float(curr["signal"]) and float(prev["macd"]) <= float(prev["signal"])
        bear_cross = float(curr["macd"]) < float(curr["signal"]) and float(prev["macd"]) >= float(prev["signal"])

        close_price = float(curr["Close"])
        upper = float(curr["bb_upper"])
        lower = float(curr["bb_lower"])

        up_dist = ((upper - close_price) / upper) * 100.0 if upper > 0 else 999.0
        down_dist = ((close_price - lower) / max(lower, 1e-9)) * 100.0 if lower > 0 else 999.0

        near_upper = close_price <= upper and up_dist <= self.near_breakout_pct
        near_lower = close_price >= lower and down_dist <= self.near_breakout_pct

        direction = ""
        distance = 0.0
        if bull_cross and near_upper:
            direction = "LONG"
            distance = max(up_dist, 0.0)
        elif bear_cross and near_lower:
            direction = "SHORT"
            distance = max(down_dist, 0.0)
        else:
            return None
            
        curr_atr = float(curr["ATR"])
        bb_mid = float(curr["bb_mid"])

        if direction == "LONG":
            entry_price = float(curr["High"]) * 1.002
            stop_loss = bb_mid
            target_1 = entry_price + (3.0 * curr_atr)
            target_2 = entry_price + (3.0 * curr_atr)
        else:
            entry_price = float(curr["Low"]) * 0.998
            stop_loss = bb_mid
            target_1 = entry_price - (3.0 * curr_atr)
            target_2 = entry_price - (3.0 * curr_atr)
            
        risk = abs(entry_price - stop_loss)
        if risk <= 0:
            return None

        return Signal(
            symbol=candles.symbol,
            date=candles.latest_date.strftime("%d/%m/%Y"),
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            targets={"Target_1": target_1, "Target_2": target_2},
            metadata={
                "Close": round(close_price, 2),
                "macd": round(float(curr["macd"]), 4),
                "signal_line": round(float(curr["signal"]), 4),
                "histogram": round(float(curr["hist"]), 4),
                "bb_upper": round(upper, 2),
                "bb_lower": round(lower, 2),
                "bb_width_avg_pct": round(avg_width, 2),
                "distance_to_band_pct": round(distance, 2),
                "ATR": round(curr_atr, 2),
                "Avg_Volume": int(avg_vol),
                "Curr_Volume": int(curr_vol),
                "rank_score": -distance # Prioritize closest to the band
            }
        )
