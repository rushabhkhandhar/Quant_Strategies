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

    def analyze(self, candles: CandleSet) -> Optional[Signal]:
        df = candles.daily
        if len(df) < 200:
            return None

        # 1. Macro Trend Filter
        ema_50 = df["Close"].ewm(span=50, adjust=False).mean()
        ema_200 = df["Close"].ewm(span=200, adjust=False).mean()
        
        curr_close = float(df.iloc[-1]["Close"])
        curr_ema_50 = float(ema_50.iloc[-1])
        curr_ema_200 = float(ema_200.iloc[-1])

        # Ensure it's a true market leader: Close is at least 20% above the 200 EMA
        if not (curr_close > curr_ema_50 and curr_ema_50 > curr_ema_200 and curr_close >= (curr_ema_200 * 1.20)):
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
