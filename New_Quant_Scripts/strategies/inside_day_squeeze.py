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

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if "EMA_50" not in df.columns:
            df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
        if "BB_Mean" not in df.columns:
            df["BB_Mean"] = df["Close"].rolling(window=20).mean()
            df["BB_Std"] = df["Close"].rolling(window=20).std()
            df["Upper_BB"] = df["BB_Mean"] + (df["BB_Std"] * 2)
            df["Lower_BB"] = df["BB_Mean"] - (df["BB_Std"] * 2)
        if "VOL_20" not in df.columns:
            df["VOL_20"] = df["Volume"].rolling(window=20).mean()
        return df

    def analyze(self, candles: CandleSet) -> Optional[Signal]:
        df = candles.daily
        if len(df) < 50:
            return None

        # Ensure indicators are calculated
        df = self.prepare_data(df)
        
        curr_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        curr_close = float(curr_row["Close"])
        curr_ema_50 = float(df['EMA_50'].iloc[-1])
        
        upper_bb = float(df['Upper_BB'].iloc[-1])
        lower_bb = float(df['Lower_BB'].iloc[-1])
        rolling_mean = float(df['BB_Mean'].iloc[-1])
        vol_20_curr = float(df['VOL_20'].iloc[-1])
        
        # 1. Trend Filter
        if curr_close < curr_ema_50:
            return None
            
        # 2. Bollinger Band Squeeze Check (Volatility Contraction)
        # The width of the bands must be very tight, e.g., less than 6% of the price.
        bb_width_pct = ((upper_bb - lower_bb) / rolling_mean) * 100
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
        if curr_vol >= vol_20_curr:
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
                "Inside_Day": "Yes",
                "rank_score": -bb_width_pct
            }
        )
