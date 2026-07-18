from typing import Optional
import pandas as pd

from core.models import CandleSet, Signal
from strategies.base import BaseStrategy

def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI) using Wilder's Smoothing."""
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    # Wilder's Smoothing (alpha = 1 / period, which translates to center of mass = period - 1)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

class RubberBandStrategy(BaseStrategy):
    """
    Stretched Rubber Band (Mean Reversion) Strategy.
    
    Finds stocks in a macro uptrend that have suffered a severe short-term 
    panic sell-off. It looks for extreme deviation from the 20-day mean 
    along with deeply oversold RSI, then triggers when selling exhaustion appears.
    """

    @property
    def name(self) -> str:
        return "Rubber_Band_Mean_Reversion"

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if "EMA_20" not in df.columns:
            df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
        if "EMA_200" not in df.columns:
            df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
        if "RSI_14" not in df.columns:
            df["RSI_14"] = calc_rsi(df, period=14)
        return df

    def analyze(self, candles: CandleSet) -> Optional[Signal]:
        df = candles.daily
        if len(df) < 200:
            return None

        # Ensure indicators are calculated
        df = self.prepare_data(df)
        
        curr_row = df.iloc[-1]
        
        curr_close = float(curr_row["Close"])
        curr_ema_20 = float(df['EMA_20'].iloc[-1])
        curr_ema_200 = float(df['EMA_200'].iloc[-1])
        curr_rsi = float(df['RSI_14'].iloc[-1])
        
        # 1. Macro Trend Filter (No catching falling knives in bear markets)
        if curr_close < curr_ema_200:
            return None
            
        # 2. Extreme Deviation Filter
        # The stock must be drastically stretched below its short-term mean (e.g., >8% below 20 EMA)
        deviation_pct = ((curr_close - curr_ema_20) / curr_ema_20) * 100
        if deviation_pct > -8.0:
            return None
            
        # 3. Oversold RSI Filter
        if pd.isna(curr_rsi) or curr_rsi > 30.0:
            return None
            
        # 4. Exhaustion / Reversal Candle Check
        # We don't want to buy while it's actively crashing down. We want signs the selling stopped.
        # Condition: The close must be in the upper 50% of today's range (a hammer or strong bounce).
        daily_range = float(curr_row["High"]) - float(curr_row["Low"])
        if daily_range <= 0:
            return None
            
        close_pct_of_range = (curr_close - float(curr_row["Low"])) / daily_range
        if close_pct_of_range < 0.50:
            return None # Still closing near the lows, too dangerous

        # Entry and Stop Loss
        entry_price = float(curr_row["High"]) * 1.002 # Buy slightly above today's high (confirmation)
        stop_loss = float(curr_row["Low"]) * 0.99     # Stop below the panic low
        
        risk = entry_price - stop_loss
        if risk <= 0:
            return None
            
        # Mean reversion targets are usually the mean itself!
        target_1 = curr_ema_20  # Reversion back to the 20 EMA
        target_2 = entry_price + (risk * 3.0) # Or 3R if it turns into a new leg up

        return Signal(
            symbol=candles.symbol,
            date=candles.latest_date.strftime("%d/%m/%Y"),
            direction="LONG",
            entry_price=entry_price,
            stop_loss=stop_loss,
            targets={"T1 (Mean Rev)": target_1, "T2 (3R)": target_2},
            metadata={
                "Close": round(curr_close, 2),
                "Deviation_%": round(deviation_pct, 2),
                "RSI_14": round(curr_rsi, 2)
            }
        )
