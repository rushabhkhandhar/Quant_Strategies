from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import warnings
from xgboost import XGBClassifier

# Suppress all warnings from hmmlearn (including RuntimeWarning for convergence issues)
import warnings
import logging
warnings.filterwarnings("ignore", module="hmmlearn")
warnings.filterwarnings("ignore", message=".*attribute is set.*")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

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
        self.meta_model = None
        self.chandelier_multiplier = 2.5
        self.target_annual_volatility = 0.15
        self.max_leverage = 2.0

    @property
    def name(self) -> str:
        return "Fibonacci_Bounce"

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if "VOL_20" not in df.columns:
            df["VOL_20"] = df["Volume"].rolling(window=20).mean()
        
        if "Log_Return" not in df.columns:
            # Calculate daily logarithmic returns
            df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
            
        if "EMA_50" not in df.columns:
            df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
            
        if "Volat_20" not in df.columns:
            df["Volat_20"] = df["Log_Return"].rolling(window=20).std()
            
        if "Annualized_Vol_20" not in df.columns:
            df["Annualized_Vol_20"] = df["Volat_20"] * np.sqrt(252)
            
        if "Parkinson_Vol_20" not in df.columns:
            low_safe = df["Low"].replace(0, 1e-9)
            daily_var = (np.log(df["High"] / low_safe))**2
            rolling_sum = daily_var.rolling(window=20).sum()
            variance_20 = rolling_sum * (1.0 / (4.0 * 20 * np.log(2)))
            df["Parkinson_Vol_20"] = np.sqrt(variance_20.clip(lower=0.0))
            
        if "HMM_Regime" not in df.columns:
            df["HMM_Regime"] = "Unknown"
            returns_array = df["Log_Return"].fillna(0).values
            hmm_regimes = np.array(["Unknown"] * len(df), dtype=object)
            
            is_fitted = False
            high_vol_state = 0
            model = None
            
            for i in range(self.hmm_window, len(df)):
                # Fit the model every 63 trading days (approx 1 quarter)
                if i == self.hmm_window or i % 63 == 0:
                    window_returns = returns_array[i - self.hmm_window : i].reshape(-1, 1)
                    try:
                        # Instantiate a fresh model to prevent initialization overwrite warnings
                        model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
                        model.fit(window_returns)
                        variances = np.array([np.diag(model.covars_[j]) for j in range(2)])
                        high_vol_state = np.argmax(variances)
                        is_fitted = True
                    except Exception:
                        is_fitted = False
                
                if is_fitted:
                    # Predict the regime for the *current* day using the last hmm_window days
                    current_window = returns_array[i - self.hmm_window + 1 : i + 1].reshape(-1, 1)
                    try:
                        hidden_states = model.predict(current_window)
                        current_state = hidden_states[-1]
                        if current_state == high_vol_state:
                            hmm_regimes[i] = "Bear/High_Vol"
                        else:
                            hmm_regimes[i] = "Bull/Calm"
                    except Exception:
                        pass
                        
            df["HMM_Regime"] = hmm_regimes

        return df

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

    def _apply_triple_barrier(self, df: pd.DataFrame, entry_idx: Any, entry_price: float, target_price: float, stop_loss: float, max_holding_days: int = 10) -> int:
        if entry_idx not in df.index:
            return 0
        int_idx = df.index.get_loc(entry_idx)
        if int_idx + 1 >= len(df):
            return 0
        
        future_df = df.iloc[int_idx + 1 : int_idx + 1 + max_holding_days]
        for idx, row in future_df.iterrows():
            if row['High'] >= target_price:
                return 1
            if row['Low'] <= stop_loss:
                return 0
        return 0

    def _extract_features(self, df: pd.DataFrame, idx: Any) -> dict:
        if idx not in df.index:
            sub_df = df
            row = df.iloc[-1]
        else:
            int_idx = df.index.get_loc(idx)
            sub_df = df.iloc[:int_idx + 1]
            row = sub_df.iloc[-1]
            
        volatility_20 = row.get("Volat_20", 0)
        vol_20 = row.get("VOL_20", 1e-9)
        volume_ratio = row["Volume"] / vol_20 if vol_20 > 0 else 1.0
        
        ema_50 = row.get("EMA_50", row["Close"])
        dist_to_ema50 = ((row["Close"] - ema_50) / ema_50) * 100 if ema_50 > 0 else 0
        
        window = sub_df.iloc[-self.swing_lookback:]
        swing_size_pct = 0.0
        if len(window) > 0:
            high_idx = window[::-1]["High"].idxmax()
            prefix = window.loc[:high_idx]
            if len(prefix) >= 2:
                low_idx = prefix["Low"].idxmin()
                swing_high = float(window.loc[high_idx, "High"])
                swing_low = float(prefix.loc[low_idx, "Low"])
                swing_size_pct = ((swing_high - swing_low) / swing_low) * 100.0 if swing_low > 0 else 0.0
                
        return {
            "volatility_20": float(volatility_20),
            "volume_ratio": float(volume_ratio),
            "dist_to_ema50": float(dist_to_ema50),
            "swing_size_pct": float(swing_size_pct)
        }

    def train_meta_model(self, historical_data: pd.DataFrame):
        df = self.prepare_data(historical_data.copy())
        
        X = []
        y = []
        
        min_days = max(50, self.swing_lookback + 25)
        for i in range(min_days, len(df)):
            sub_df = df.iloc[:i]
            current_date = sub_df.index[-1]
            
            candle_set = CandleSet(symbol="TRAIN", daily=sub_df)
            signal = self.analyze(candle_set)
            
            if signal is not None:
                features = self._extract_features(df, current_date)
                label = self._apply_triple_barrier(
                    df=df,
                    entry_idx=current_date,
                    entry_price=signal.entry_price,
                    target_price=signal.targets.get("Target_1", signal.entry_price * 1.1),
                    stop_loss=signal.stop_loss,
                    max_holding_days=10
                )
                
                X.append([
                    features["volatility_20"],
                    features["volume_ratio"],
                    features["dist_to_ema50"],
                    features["swing_size_pct"]
                ])
                y.append(label)
                
        if len(X) > 10 and len(set(y)) > 1:
            X_arr = np.array(X)
            y_arr = np.array(y)
            self.meta_model = XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
            self.meta_model.fit(X_arr, y_arr)
        else:
            self.meta_model = None

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
        current_regime = str(df["HMM_Regime"].iloc[-1])
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
        
        # Dynamic Initial Stop-Loss using Chandelier Exit logic
        current_parkinson_vol = float(df["Parkinson_Vol_20"].iloc[-1])
        if pd.isna(current_parkinson_vol):
            current_parkinson_vol = 0.0
            
        chandelier_distance = self.chandelier_multiplier * current_parkinson_vol
        stop_loss = entry_price - chandelier_distance
        
        risk = entry_price - stop_loss
        if risk <= 0:
            return None

        # Calculate Volatility Target Position Sizing
        current_annualized_vol = float(df["Annualized_Vol_20"].iloc[-1])
        if pd.isna(current_annualized_vol) or current_annualized_vol <= 0:
            current_annualized_vol = 1e-9
            
        vol_target_multiplier = self.target_annual_volatility / current_annualized_vol
        vol_target_multiplier = max(0.2, min(vol_target_multiplier, self.max_leverage))

        target_1 = fib_382
        target_2 = swing_high

        range_size = max(swing_high - swing_low, 1e-9)
        stop_loss_2 = swing_low
        stop_loss_3 = swing_low - 0.25 * range_size
        target_3 = swing_high + 0.272 * range_size

        sig = Signal(
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
                "rank_score": -dist_50 if near_50 else -dist_618,
                "Parkinson_Vol_20": round(current_parkinson_vol, 6),
                "Chandelier_Distance": round(chandelier_distance, 4),
                "Annualized_Vol_20": round(current_annualized_vol, 4),
                "Position_Multiplier": round(vol_target_multiplier, 4)
            }
        )
        
        # Note for Backtester/Execution Engine:
        # 1. The execution engine should trail this stop-loss by tracking the Highest High 
        #    during the trade and subtracting (self.chandelier_multiplier * Parkinson_Vol_20).
        # 2. The execution engine should multiply the standard baseline risk allocation 
        #    by Position_Multiplier when executing the trade.

        if getattr(self, 'meta_model', None) is not None:
            features = self._extract_features(df, df.index[-1])
            X_pred = np.array([[
                features["volatility_20"],
                features["volume_ratio"],
                features["dist_to_ema50"],
                features["swing_size_pct"]
            ]])
            prob = self.meta_model.predict_proba(X_pred)[0]
            prob_1 = prob[1] if len(prob) > 1 else 0.0
            
            if prob_1 < 0.65:
                return None
                
            sig.metadata["ML_Probability"] = round(float(prob_1), 4)

        return sig
