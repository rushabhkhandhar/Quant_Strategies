from abc import ABC, abstractmethod
from typing import Optional

from core.models import CandleSet, Signal

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the strategy (used for output folders)."""
        pass

    @abstractmethod
    def analyze(self, candles: CandleSet) -> Optional[Signal]:
        """
        Analyze a set of candles and return a Signal if a setup is found,
        otherwise return None.
        """
        pass
        
    def prepare_data(self, df):
        """
        Optional hook to pre-calculate indicators on the entire dataframe 
        for vectorized backtesting performance. By default, returns df untouched.
        """
        return df
