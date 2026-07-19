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
    3) Price is near the outer band (<1.0% distance).
    4) Volume is at least 2.0x the 20-day average.
    """
    
    def __init__(self):
        super().__init__()
        self.bb_narrow_lookback = 15
        self.bb_width_max_pct = 8.0
        self.volume_multiplier = 2.0  # volume expansion multiplier (e.g., 2× avg volume)
        self.near_breakout_pct = 1.0  # % distance from band
        # Removed unused ATR parameters (min_atr, max_atr_lookback)
        self.min_vol_value = 70_000_000  # legacy liquidity filter (kept for backward compatibility)
        self.trend_sma_window = 200  # market regime filter (SMA)

    @property
    def name(self) -> str:
        return "MACD_BB_Expansion"

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
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
        df.loc[:, "sma_200"] = df["Close"].rolling(self.trend_sma_window).mean()

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
        if len(df) < max(200, self.bb_narrow_lookback + 25):
            return None

        # Ensure indicators are calculated
        df = self.prepare_data(df)
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Volatility Contraction: ensure *all* recent widths are below the threshold for sustained narrowness
        recent_widths = df["bb_width_pct"].iloc[-(self.bb_narrow_lookback + 1):-1]
        if recent_widths.isna().any() or (recent_widths > self.bb_width_max_pct).any():
            return None
        
        # Volume Filter: compute 20‑day rolling average volume excluding the current candle
        avg_vol_20 = df["Volume"].shift(1).rolling(window=20).mean()
        if pd.isna(avg_vol_20.iloc[-1]):
            return None
        # Liquidity filter: ensure the average traded value over the past 20 days meets a minimum threshold
        avg_traded_value_20 = (df["Volume"] * df["Close"]).shift(1).rolling(window=20).mean()
        if avg_traded_value_20.iloc[-1] < self.min_vol_value:
            return None
        # Volume expansion filter: ensure today's volume is at least volume_multiplier × avg volume
        if float(curr["Volume"]) < (self.volume_multiplier * avg_vol_20.iloc[-1]):
            return None
        
        # Trend filter (only trade in direction of 200 SMA)
        if not pd.isna(curr["sma_200"]):
            is_uptrend = curr["Close"] >= curr["sma_200"]
            is_downtrend = curr["Close"] <= curr["sma_200"]
        else:
            is_uptrend = is_downtrend = False

        # ATR must be a valid number
        if pd.isna(curr["ATR"]):
            return None

        # MACD Crossover detection
        bull_cross = float(curr["macd"]) > float(curr["signal"]) and float(prev["macd"]) <= float(prev["signal"])
        bear_cross = float(curr["macd"]) < float(curr["signal"]) and float(prev["macd"]) >= float(prev["signal"])
        
        close_price = float(curr["Close"])
        upper = float(curr["bb_upper"])
        lower = float(curr["bb_lower"])
        
        up_dist = abs(upper - close_price) / upper * 100.0 if upper > 0 else 999.0
        down_dist = abs(close_price - lower) / max(lower, 1e-9) * 100.0 if lower > 0 else 999.0
        
        direction = ""
        distance = 0.0
        if bull_cross and is_uptrend and up_dist < self.near_breakout_pct:
            direction = "LONG"
            distance = up_dist
        elif bear_cross and is_downtrend and down_dist < self.near_breakout_pct:
            direction = "SHORT"
            distance = down_dist
        else:
            return None
        
        # Composite ranking score (higher is better)
        hist_strength = abs(float(curr["hist"]))
        vol_exp = float(curr["Volume"]) / avg_vol_20.iloc[-1] if avg_vol_20.iloc[-1] != 0 else 0
        # Robust ATR expansion handling (avoid NaN or zero division)
        curr_atr_val = float(curr["ATR"]) if not pd.isna(curr["ATR"]) else 0.0
        prev_atr_val = float(prev["ATR"]) if not pd.isna(prev["ATR"]) else 0.0
        atr_exp = (curr_atr_val / prev_atr_val) if prev_atr_val > 0 else 1.0
        # We penalize larger distance from Bollinger band
        rank_score = (0.4 * hist_strength) + (0.3 * vol_exp) + (0.2 * atr_exp) - (0.1 * distance)
        
        curr_atr = float(curr["ATR"])

        # Realistic entry: signal generated after today's close; entry is placed slightly above today’s high (LONG) or below today's low (SHORT) to ensure execution on the next trading day when price moves in the breakout direction
        if direction == "LONG":
            entry_price = float(curr["High"]) * 1.001
            stop_loss = entry_price - (2.0 * curr_atr)
            target_1 = entry_price + (2.0 * curr_atr)
            target_2 = entry_price + (4.0 * curr_atr)
        else:
            entry_price = float(curr["Low"]) * 0.999
            stop_loss = entry_price + (2.0 * curr_atr)
            target_1 = entry_price - (2.0 * curr_atr)
            target_2 = entry_price - (4.0 * curr_atr)
            
        return Signal(
            symbol=candles.symbol,
            date=candles.latest_date.strftime("%d/%m/%Y"),
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            targets={"Target_1": target_1, "Target_2": target_2},
            metadata={
                "macd_cross": True,
                "bb_width_pct": round(float(curr["bb_width_pct"]), 2),
                "vol_vs_avg": round(float(curr["Volume"]) / avg_vol_20.iloc[-1], 2),
                "rank_score": round(rank_score, 4),
                "ATR": float(curr["ATR"]) if not pd.isna(curr["ATR"]) else 0.0
            }
        )
