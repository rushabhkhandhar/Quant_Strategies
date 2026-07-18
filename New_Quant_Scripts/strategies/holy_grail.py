from typing import Optional
import pandas as pd
import numpy as np

from core.models import CandleSet, Signal
from strategies.base import BaseStrategy

def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate ADX, +DI, and -DI using Wilder's Smoothing."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    pos_dm = pd.Series(pos_dm, index=df.index)
    neg_dm = pd.Series(neg_dm, index=df.index)
    
    # Wilder's Smoothing (alpha = 1/period)
    def wilder_smooth(s: pd.Series, n: int) -> pd.Series:
        res = np.full_like(s, np.nan, dtype=float)
        if len(s) < n:
            return pd.Series(res, index=s.index)
            
        s_vals = s.to_numpy(dtype=float)
        res[n] = np.nansum(s_vals[1:n+1])
        for i in range(n+1, len(s)):
            res[i] = res[i-1] - (res[i-1] / n) + s_vals[i]
        return pd.Series(res, index=s.index)
        
    atr = wilder_smooth(tr, period)
    plus_di = 100 * (wilder_smooth(pos_dm, period) / atr)
    minus_di = 100 * (wilder_smooth(neg_dm, period) / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = wilder_smooth(dx, period)
    
    result = pd.DataFrame({'ADX': adx, '+DI': plus_di, '-DI': minus_di}, index=df.index)
    return result

class HolyGrailStrategy(BaseStrategy):
    """
    Linda Raschke's 'Holy Grail' Strategy.
    
    Finds stocks in extremely powerful trends (ADX > 30) that have pulled back 
    to touch their 20-period EMA. Buy trigger is placed above the high of the 
    pullback candle.
    """

    @property
    def name(self) -> str:
        return "Holy_Grail_Pullback"

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if "EMA_20" not in df.columns:
            df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
        if "EMA_50" not in df.columns:
            df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
        if "EMA_200" not in df.columns:
            df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
        if "VOL_20" not in df.columns:
            df["VOL_20"] = df["Volume"].rolling(window=20).mean()
        if "ADX" not in df.columns:
            adx_data = calc_adx(df, period=14)
            df["ADX"] = adx_data["ADX"]
            df["+DI"] = adx_data["+DI"]
            df["-DI"] = adx_data["-DI"]
        return df

    def analyze(self, candles: CandleSet) -> Optional[Signal]:
        df = candles.daily
        if len(df) < 50:
            return None

        # Ensure indicators are calculated
        df = self.prepare_data(df)
        
        curr_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        curr_adx = float(df['ADX'].iloc[-1])
        prev3_adx = float(df['ADX'].iloc[-4])  # ADX 3 days ago
        curr_plus_di = float(df['+DI'].iloc[-1])
        curr_minus_di = float(df['-DI'].iloc[-1])
        
        curr_ema_20 = float(df['EMA_20'].iloc[-1])
        curr_ema_50 = float(df['EMA_50'].iloc[-1])
        curr_ema_200 = float(df['EMA_200'].iloc[-1])
        
        curr_vol = float(curr_row["Volume"])
        curr_vol_20 = float(df['VOL_20'].iloc[-1])
        
        # 1. Extreme Trend Filter (ADX > 35 and Rising)
        if pd.isna(curr_adx) or curr_adx < 35:
            return None
        if curr_adx <= prev3_adx:  # ADX must be rising
            return None
            
        # 2. Macro Trend Alignment (20 EMA > 50 EMA > 200 EMA) and Direction
        if not (curr_ema_20 > curr_ema_50 and curr_ema_50 > curr_ema_200):
            return None
        if curr_plus_di <= curr_minus_di:
            return None
            
        # 3. Pullback Filter (The low of today or yesterday must have touched/pierced the 20 EMA)
        # We also want the close to be reasonably close to the EMA, not a huge bounce already.
        touched_ema = (curr_row["Low"] <= curr_ema_20 <= curr_row["High"]) or \
                      (prev_row["Low"] <= curr_ema_20 <= prev_row["High"])
                      
        if not touched_ema:
            return None
            
        # 3b. Volume Dry-up on Pullback
        if curr_vol >= curr_vol_20:
            return None
            
        # 4. Momentum Check: Ensure the stock is still generally closing above the EMA
        if curr_row["Close"] < (curr_ema_20 * 0.98):
            return None # Dropped too far below EMA, trend might be breaking

        # Entry and Stop Loss
        entry_price = float(curr_row["High"]) * 1.002 # Buy slightly above today's high
        stop_loss = float(curr_row["Low"]) * 0.99     # Stop below today's low
        
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
                "ADX": round(curr_adx, 2),
                "EMA_20": round(curr_ema_20, 2),
                "Pullback_Low": round(float(prev_row["Low"]), 2),
                "rank_score": curr_adx
            }
        )
