from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import warnings

# Suppress hmmlearn warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning, module="hmmlearn")

from core.models import CandleSet, Signal
from strategies.base import BaseStrategy

class FibonacciBounceStrategy(BaseStrategy):
    """
    Fibonacci Retracement + Candlestick + Volume Strategy + HMM Regime Filter.
    
    Rules:
    1) Build Fib from recent swing low to swing high (60-day window).
    2) Current candle is near 50% or 61.8% retracement zone.
    3) Bullish candlestick confirmation (Hammer or Bullish Engulfing).
    4) Current volume is above average volume.
    5) Macro Trend: HMM identifies current market regime as Bull/Calm.
    """
    
    def __init__(self):
        super().__init__()
        self.swing_lookback = 60
        self.near_level_pct = 1.2
        self.volume_lookback = 20
        self.volume_multiplier = 1.0  
        self.min_swing_pct = 6.0
        self.hmm_window = 252

    @property
    def name(self) -> str:
        return "Fibonacci_Bounce"

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if "VOL_20" not in df.columns:
            df["VOL_20"] = df["Volume"].rolling(window=20).mean()
        
        if "Log_Return" not in df.columns:
            # Calculate daily logarithmic returns
            df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
            
        return df

    def _get_current_regime(self, df: pd.DataFrame) -> str:
        returns = df["Log_Return"].dropna()
        if len(returns) < 20: 
            return "Unknown"
            
        window_returns = returns.iloc[-self.hmm_window:]
        X = window_returns.values.reshape(-1, 1)
        
        # Fit a 2-state GaussianHMM
        model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
        try:
            model.fit(X)
            hidden_states = model.predict(X)
        except Exception:
            return "Unknown"
            
        # Dynamically identify high-volatility vs low-volatility states
        variances = np.array([np.diag(model.covars_[i]) for i in range(2)])
        high_vol_state = np.argmax(variances)
        
        current_state = hidden_states[-1]
        
        if current_state == high_vol_state:
            return "Bear/High_Vol"
        else:
            return "Bull/Calm"

    def _bullish_hammer(self, candle: pd.Series) -> bool:
        open_price = float(candle["Open"])
        close_price = float(candle["Close"])
        high_price = float(candle["High"])
        low_price = float(candle["Low"])

        body = abs(close_price - open_price)
        lower_shadow = min(open_price, close_price) - low_price
        upper_shadow = high_price - max(open_price, close_price)
        day_range = max(high_price - low_price, 1e-9)

        return (
            lower_shadow >= 2.0 * max(body, 1e-9)
            and upper_shadow <= (0.15 * day_range)
        )

    def _bullish_engulfing(self, prev_candle: pd.Series, curr_candle: pd.Series) -> bool:
        prev_open = float(prev_candle["Open"])
        prev_close = float(prev_candle["Close"])
        curr_open = float(curr_candle["Open"])
        curr_close = float(curr_candle["Close"])

        return (
            prev_close < prev_open
            and curr_close > curr_open
            and curr_close > prev_open
            and curr_open <= prev_close
        )

    def analyze(self, candles: CandleSet) -> Optional[Signal]:
        df = candles.daily
        if len(df) < max(50, self.swing_lookback + 25):
            return None

        # Ensure indicators are calculated
        df = self.prepare_data(df)
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        close_price = float(curr["Close"])
        low_price = float(curr["Low"])
        high_price = float(curr["High"])
        curr_vol = float(curr["Volume"])
        
        # 1. HMM Regime Filter
        current_regime = self._get_current_regime(df)
        if current_regime == "Bear/High_Vol" or current_regime == "Unknown":
            return None

        # 2. Liquidity Filter (Approx 70 Cr)
        avg_vol = float(df["VOL_20"].iloc[-1])
        if (avg_vol * close_price) < 70_000_000:
            return None

        # 3. Identify Swing
        window = df.iloc[-self.swing_lookback:]
        high_idx = window[::-1]["High"].idxmax()
        
        # Stale Swing Check: Ensure the swing high occurred within the last 20 trading days
        days_since_high = len(window.loc[high_idx:]) - 1
        if days_since_high > 20:
            return None

        prefix = window.loc[:high_idx]
        if len(prefix) < 2:
            return None

        low_idx = prefix["Low"].idxmin()
        swing_high = float(window.loc[high_idx, "High"])
        swing_low = float(prefix.loc[low_idx, "Low"])

        if swing_high <= swing_low:
            return None

        swing_pct = ((swing_high - swing_low) / swing_low) * 100.0 if swing_low > 0 else 0.0
        if swing_pct < self.min_swing_pct:
            return None

        # 4. Fibonacci Levels
        fib_382 = swing_high - 0.382 * (swing_high - swing_low)
        fib_50 = swing_high - 0.500 * (swing_high - swing_low)
        fib_618 = swing_high - 0.618 * (swing_high - swing_low)

        dist_50 = abs(close_price - fib_50) / fib_50 * 100.0 if fib_50 > 0 else 999.0
        dist_618 = abs(close_price - fib_618) / fib_618 * 100.0 if fib_618 > 0 else 999.0

        touched_50 = low_price <= fib_50 <= high_price
        touched_618 = low_price <= fib_618 <= high_price

        defends_50 = touched_50 and close_price >= (fib_50 * 0.995)
        defends_618 = touched_618 and close_price >= (fib_618 * 0.995)

        # Require the close to be >= the support level minus a tiny buffer
        near_50 = (defends_50 or dist_50 <= self.near_level_pct) and close_price >= (fib_50 * 0.995)
        near_618 = (defends_618 or dist_618 <= self.near_level_pct) and close_price >= (fib_618 * 0.995)
        
        if not (near_50 or near_618):
            return None

        # 5. Candlestick Pattern
        pattern_name = ""
        if self._bullish_engulfing(prev, curr):
            pattern_name = "bullish_engulfing"
        elif self._bullish_hammer(curr):
            pattern_name = "hammer"
        else:
            return None

        # 6. Volume Spike
        prev_avg_vol = float(df["VOL_20"].iloc[-2])
        if prev_avg_vol > 0:
            if curr_vol < (self.volume_multiplier * prev_avg_vol):
                return None
        
        # 7. Calculate Risk and Levels
        entry_price = fib_50 if near_50 else fib_618
        
        # Base the stop loss on the absolute low of the candlestick pattern
        pattern_low = min(low_price, float(prev["Low"])) if pattern_name == "bullish_engulfing" else low_price
        # Stop loss with a 1% buffer below the 61.8% level for SL1 to avoid stop-hunts
        stop_loss = min(pattern_low, fib_618 * 0.99)
        
        risk = entry_price - stop_loss
        if risk <= 0:
            return None

        target_1 = fib_382
        target_2 = swing_high

        range_size = max(swing_high - swing_low, 1e-9)
        stop_loss_2 = swing_low
        stop_loss_3 = swing_low - 0.25 * range_size
        target_3 = swing_high + 0.272 * range_size

        return Signal(
            symbol=candles.symbol,
            date=candles.latest_date.strftime("%d/%m/%Y"),
            direction="LONG",
            entry_price=entry_price,
            stop_loss=stop_loss,
            targets={"Target_1": target_1, "Target_2": target_2},
            metadata={
                "Regime": current_regime,
                "Close": round(close_price, 2),
                "Swing_Low": round(swing_low, 2),
                "Swing_High": round(swing_high, 2),
                "Fib_382": round(fib_382, 2),
                "Fib_50": round(fib_50, 2),
                "Fib_618": round(fib_618, 2),
                "Entry_Level_Name": "fib_50" if near_50 else "fib_618",
                "Entry_Level": round(fib_50 if near_50 else fib_618, 2),
                "Distance_Pct": round(dist_50 if near_50 else dist_618, 2),
                "Pattern": pattern_name,
                "StopLoss_2": round(stop_loss_2, 2),
                "StopLoss_3": round(stop_loss_3, 2),
                "Target_3": round(target_3, 2),
                "Avg_Volume": int(prev_avg_vol),
                "Curr_Volume": int(curr_vol),
                "rank_score": -dist_50 if near_50 else -dist_618
            }
        )
