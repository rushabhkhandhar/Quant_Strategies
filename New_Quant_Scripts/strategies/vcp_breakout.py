from typing import Optional
import pandas as pd

from core.models import CandleSet, Signal
from strategies.base import BaseStrategy

class VCPBreakoutStrategy(BaseStrategy):
    """
    Volatility Contraction Pattern (VCP) Strategy
    
    Identifies stocks in a macro uptrend that have formed a tight price
    consolidation (low volatility) with drying volume. This setup prepares
    for a high-volume breakout.
    """

    @property
    def name(self) -> str:
        return "VCP_Breakout"

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if "EMA_200" not in df.columns:
            df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
        if "EMA_50" not in df.columns:
            df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
        if "VOL_50" not in df.columns:
            df["VOL_50"] = df["Volume"].rolling(window=50).mean()
        if "High_200" not in df.columns:
            df["High_200"] = df["High"].rolling(window=200).max()
        if "Low_200" not in df.columns:
            df["Low_200"] = df["Low"].rolling(window=200).min()
        return df

    def analyze(self, candles: CandleSet) -> Optional[Signal]:
        df = candles.daily
        if len(df) < 200:
            return None
            
        # Ensure indicators are calculated
        df = self.prepare_data(df)

        curr_row = df.iloc[-1]
        curr_close = float(curr_row["Close"])
        curr_vol = float(curr_row["Volume"])

        # Indicators for current day
        ema_200 = float(df["EMA_200"].iloc[-1])
        ema_50 = float(df["EMA_50"].iloc[-1])
        vol_50 = float(df["VOL_50"].iloc[-1])
        
        high_200 = float(df["High_200"].iloc[-1])
        low_200 = float(df["Low_200"].iloc[-1])

        # Ensure it's a true market leader: Close is at least 20% above the 200 EMA
        if not (curr_close > ema_50 and ema_50 > ema_200 and curr_close >= (ema_200 * 1.20)):
            return None

        # 2. Contraction (Tight Base) Check
        # The high-to-low range over the last 15 days should be relatively tight
        consolidation_window = df.iloc[-15:]
        max_high = float(consolidation_window["High"].max())
        min_low = float(consolidation_window["Low"].min())
        
        if max_high <= 0:
            return None
            
        base_depth_pct = ((max_high - min_low) / max_high) * 100
        
        # A tight base is usually < 6% deep in the last 2-3 weeks
        if base_depth_pct > 6.0:
            return None
            
        # Proximity to Breakout: The current close must be within 1.5% of the resistance high
        if curr_close < (max_high * 0.985):
            return None

        # 3. Volume Contraction Check
        # Average volume recently should be lower than the long-term average,
        # proving that supply (selling pressure) has dried up.
        vol_50 = float(df["Volume"].iloc[-50:].mean())
        vol_10 = float(df["Volume"].iloc[-10:].mean())
        
        # Extreme volume dry-up: 10-day volume is less than 60% of the 50-day average
        if vol_50 <= 0 or vol_10 >= (vol_50 * 0.60):
            return None

        # Calculate practical stops and targets
        entry_price = max_high * 1.002  # Breakout trigger just above the base high
        stop_loss = min_low * 0.99      # SL just below the tight base
        
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
                "Base_Depth_Pct": round(base_depth_pct, 2),
                "Vol_10_vs_50": round(vol_10 / vol_50, 2),
                "Resistance": round(max_high, 2)
            }
        )
