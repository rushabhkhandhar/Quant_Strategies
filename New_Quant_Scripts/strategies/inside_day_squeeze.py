from typing import Optional
import pandas as pd
import numpy as np

from core.models import CandleSet, Signal
from strategies.base import BaseStrategy

class InsideDaySqueezeStrategy(BaseStrategy):
    """
    Inside-Day Volatility Squeeze Strategy.
    
    Identifies stocks in an uptrend that are experiencing extreme volatility 
    contraction (Bollinger Band squeeze) combined with an 'Inside Day' candlestick.
    This signifies explosive energy coiling up for a breakout.
    """

    @property
    def name(self) -> str:
        return "Inside_Day_Squeeze"

    def analyze(self, candles: CandleSet) -> Optional[Signal]:
        df = candles.daily
        if len(df) < 50:
            return None

        # Calculate Indicators
        ema_50 = df["Close"].ewm(span=50, adjust=False).mean()
        
        # Bollinger Bands (20-period, 2 std dev)
        rolling_mean = df["Close"].rolling(window=20).mean()
        rolling_std = df["Close"].rolling(window=20).std()
        upper_bb = rolling_mean + (rolling_std * 2)
        lower_bb = rolling_mean - (rolling_std * 2)
        
        vol_20 = df["Volume"].rolling(window=20).mean()
        
        curr_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        curr_close = float(curr_row["Close"])
        curr_ema_50 = float(ema_50.iloc[-1])
        
        # 1. Trend Filter
        if curr_close < curr_ema_50:
            return None
            
        # 2. Bollinger Band Squeeze Check (Volatility Contraction)
        # The width of the bands must be very tight, e.g., less than 6% of the price.
        bb_width_pct = ((float(upper_bb.iloc[-1]) - float(lower_bb.iloc[-1])) / float(rolling_mean.iloc[-1])) * 100
        if pd.isna(bb_width_pct) or bb_width_pct > 6.0:
            return None
            
        # 3. Inside Day Check (Price Contraction)
        # Today's high must be strictly lower than yesterday's high, AND today's low strictly higher than yesterday's low.
        is_inside_day = (float(curr_row["High"]) < float(prev_row["High"])) and \
                        (float(curr_row["Low"]) > float(prev_row["Low"]))
                        
        if not is_inside_day:
            return None
            
        # 4. Volume Check
        # Volume on the inside day should be below average, indicating equilibrium before the storm.
        curr_vol = float(curr_row["Volume"])
        if curr_vol >= float(vol_20.iloc[-1]):
            return None

        # Entry and Stop Loss
        entry_price = float(curr_row["High"]) * 1.002 # Trigger slightly above the inside day high
        stop_loss = float(curr_row["Low"]) * 0.99     # Stop just below the inside day low
        
        risk = entry_price - stop_loss
        if risk <= 0:
            return None
            
        target_1 = entry_price + (risk * 2.0)
        target_2 = entry_price + (risk * 3.0)

        return Signal(
            symbol=candles.symbol,
            date=candles.latest_date.strftime("%d/%m/%Y"),
            direction="LONG",
            entry_price=entry_price,
            stop_loss=stop_loss,
            targets={"T1 (2R)": target_1, "T2 (3R)": target_2},
            metadata={
                "Close": round(curr_close, 2),
                "BB_Width_%": round(bb_width_pct, 2),
                "Inside_Day": "Yes"
            }
        )
