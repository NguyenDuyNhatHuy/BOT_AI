# --- Do NOT remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np

class StochRSIConfirmStrategy(IStrategy):
    """
    Strategy for detecting market tops/bottoms using Stochastic Oscillator and RSI.
    Combines Stoch (13/3/3) and RSI (14) to confirm market reversals and detect divergence.
    """

    # Minimal ROI and stoploss
    minimal_roi = {
        "0": 0.1
    }

    stoploss = -0.1

    timeframe = '30m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add Stochastic, RSI, and helper columns for detecting confirmed tops/bottoms.
        """
        # Handle NaN values (replace with the mean of the previous 14 values)
        dataframe['high'] = dataframe['high'].fillna(dataframe['high'].rolling(14).mean())
        dataframe['low'] = dataframe['low'].fillna(dataframe['low'].rolling(14).mean())
        dataframe['close'] = dataframe['close'].fillna(dataframe['close'].rolling(14).mean())

        # Stochastic Oscillator (13/3/3)
        stoch = ta.STOCH(dataframe, fastk_period=13, slowk_period=3, slowd_period=3)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']

        # Relative Strength Index (RSI, 14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Detect Stochastic peaks and valleys
        dataframe = self.detect_stoch_peaks_and_valleys(dataframe)

        # Detect RSI peaks and valleys
        dataframe = self.detect_rsi_peaks_and_valleys(dataframe)

        # Detect market tops and bottoms based on Stochastic peaks/valleys and price close
        dataframe = self.detect_market_tops_and_bottoms(dataframe)

        # (Optional) Add other divergence check columns if desired
        dataframe['bullish_divergence'] = dataframe.apply(
            lambda row: self.is_bullish_divergence_price_rsi(dataframe, row.name), axis=1
        )
        dataframe['bearish_divergence'] = dataframe.apply(
            lambda row: self.is_bearish_divergence_price_rsi(dataframe, row.name), axis=1
        )
        
        
        dataframe.to_csv('stoch_test1.csv', index=False)

        return dataframe

    def detect_stoch_peaks_and_valleys(self, dataframe: DataFrame) -> DataFrame:
        """
        Identify Stochastic tops (peaks) and bottoms (valleys) based on the corrected understanding.
        """
        dataframe['stoch_top'] = 0
        dataframe['stoch_bottom'] = 0

        i = 1
        while i < len(dataframe):
            # Potential Peak (top)
            if dataframe['stoch_k'].iloc[i - 1] >= 80:
                j = i - 1
                while j >= 0 and dataframe['stoch_k'].iloc[j] >= 80:
                    j -= 1

                if j < 0:
                    i += 1
                    continue
                
                highest_idx = j + 1
                for k in range(j + 2, i):
                    if dataframe['stoch_k'].iloc[k] > dataframe['stoch_k'].iloc[highest_idx]:
                        highest_idx = k

                # Check if stoch_k crosses below 20 to confirm the peak
                
                while i < len(dataframe) and dataframe['stoch_k'].iloc[i] >= 20 :
                    if dataframe['stoch_k'].iloc[i] > dataframe['stoch_k'].iloc[highest_idx]:
                        highest_idx = i
                    i += 1

                if i < len(dataframe):
                  dataframe.loc[highest_idx, 'stoch_top'] = 1

            # Potential Valley (bottom)
            elif dataframe['stoch_k'].iloc[i - 1] <= 20:
                j = i - 1
                while j >= 0 and dataframe['stoch_k'].iloc[j] <= 20:
                    j -= 1

                if j < 0:
                    i += 1
                    continue
                
                lowest_idx = j + 1
                for k in range(j + 2, i):
                    if dataframe['stoch_k'].iloc[k] < dataframe['stoch_k'].iloc[lowest_idx]:
                        lowest_idx = k
                
                # Check if stoch_k crosses above 80 to confirm the valley
                while i < len(dataframe) and dataframe['stoch_k'].iloc[i] <= 80:
                    if dataframe['stoch_k'].iloc[i] < dataframe['stoch_k'].iloc[lowest_idx]:
                        lowest_idx = i
                    i += 1

                if i < len(dataframe):
                  dataframe.loc[lowest_idx, 'stoch_bottom'] = 1
            
            else:
                i += 1

        return dataframe

    def detect_rsi_peaks_and_valleys(self, dataframe: DataFrame) -> DataFrame:
        """
        Identify RSI tops (peaks) and bottoms (valleys) using the same logic as Stoch peaks/valleys.
        """
        dataframe['rsi_top'] = 0
        dataframe['rsi_bottom'] = 0

        i = 1
        while i < len(dataframe):
            # Potential Peak (top)
            if dataframe['rsi'].iloc[i - 1] >= 70:
                j = i - 1
                while j >= 0 and dataframe['rsi'].iloc[j] >= 70:
                    j -= 1

                if j < 0:
                    i += 1
                    continue

                highest_idx = j + 1
                for k in range(j + 2, i):
                    if dataframe['rsi'].iloc[k] > dataframe['rsi'].iloc[highest_idx]:
                        highest_idx = k

                # Check if rsi crosses below 30 to confirm the peak
                while i < len(dataframe) and dataframe['rsi'].iloc[i] >= 30:
                    if dataframe['rsi'].iloc[i] > dataframe['rsi'].iloc[highest_idx]:
                        highest_idx = i
                    i += 1

                if i < len(dataframe):
                    dataframe.loc[highest_idx, 'rsi_top'] = 1

            # Potential Valley (bottom)
            elif dataframe['rsi'].iloc[i - 1] <= 30:
                j = i - 1
                while j >= 0 and dataframe['rsi'].iloc[j] <= 30:
                    j -= 1

                if j < 0:
                    i += 1
                    continue

                lowest_idx = j + 1
                for k in range(j + 2, i):
                    if dataframe['rsi'].iloc[k] < dataframe['rsi'].iloc[lowest_idx]:
                        lowest_idx = k

                # Check if rsi crosses above 70 to confirm the valley
                while i < len(dataframe) and dataframe['rsi'].iloc[i] <= 70:
                    if dataframe['rsi'].iloc[i] < dataframe['rsi'].iloc[lowest_idx]:
                        lowest_idx = i
                    i += 1

                if i < len(dataframe):
                    dataframe.loc[lowest_idx, 'rsi_bottom'] = 1

            else:
                i += 1

        return dataframe

    def detect_market_tops_and_bottoms(self, dataframe: DataFrame) -> DataFrame:
        """
        Identify market tops and bottoms based on Stochastic tops/bottoms and price close.
        """
        dataframe['market_top'] = 0
        dataframe['market_bottom'] = 0

        for i in range(len(dataframe)):
            if dataframe['stoch_top'].iloc[i] == 1:
                dataframe.loc[i, 'market_top'] = 1

            if dataframe['stoch_bottom'].iloc[i] == 1:
                dataframe.loc[i, 'market_bottom'] = 1

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate buy signals based on confirmed bullish divergence (price vs RSI) or other logic.
        """
        dataframe['buy'] = 0

        for i in range(1, len(dataframe)):
            # Example: detect rsi_bottom and bullish divergence
            if (
                dataframe['rsi_bottom'].iloc[i] == 1 and
                self.is_bullish_divergence_price_rsi(dataframe, i)
            ):
                dataframe.loc[i, 'buy'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate sell signals based on confirmed bearish divergence (price vs RSI) or other logic.
        """
        dataframe['sell'] = 0

        for i in range(1, len(dataframe)):
            # Example: detect rsi_top and bearish divergence
            if (
                dataframe['rsi_top'].iloc[i] == 1 and
                self.is_bearish_divergence_price_rsi(dataframe, i)
            ):
                dataframe.loc[i, 'sell'] = 1

        return dataframe

    def is_bullish_divergence_price_rsi(self, dataframe: DataFrame, idx: int) -> bool:
        """
        Check for bullish divergence using previously detected peaks and valleys.
        """
        if idx < 2:  # Need at least 2 points to compare
            return False

        # 1. Find the most recent previous market bottom
        prev_market_bottom_idx = None
        for i in range(idx - 1, -1, -1):
            if dataframe['market_bottom'].iloc[i] > 0:
                prev_market_bottom_idx = i
                break

        if prev_market_bottom_idx is None:
            return False  # No previous market bottom found

        # 2. Find the most recent previous RSI bottom corresponding to the market bottom
        prev_rsi_bottom_idx = None
        for i in range(prev_market_bottom_idx, -1, -1):
            if dataframe['rsi_bottom'].iloc[i] == 1:
                prev_rsi_bottom_idx = i
                break
                
        if prev_rsi_bottom_idx is None:
            return False
        
        # Check for valid divergence
        if prev_rsi_bottom_idx > prev_market_bottom_idx:
            return False

        # 3. Compare
        prev_price_bottom = dataframe['close'].iloc[prev_market_bottom_idx]
        curr_price_bottom = dataframe['close'].iloc[idx]
        prev_rsi_bottom = dataframe['rsi'].iloc[prev_rsi_bottom_idx]
        curr_rsi_bottom = dataframe['rsi'].iloc[idx]

        is_price_higher_low = (curr_price_bottom > prev_price_bottom)
        is_rsi_lower_low = (curr_rsi_bottom < prev_rsi_bottom)

        return is_price_higher_low and is_rsi_lower_low

    def is_bearish_divergence_price_rsi(self, dataframe: DataFrame, idx: int) -> bool:
        """
        Check for bearish divergence using previously detected peaks and valleys.
        """
        if idx < 2:
            return False

        # 1. Find the most recent previous market top
        prev_market_top_idx = None
        for i in range(idx - 1, -1, -1):
            if dataframe['market_top'].iloc[i] > 0:
                prev_market_top_idx = i
                break

        if prev_market_top_idx is None:
            return False

        # 2. Find the most recent previous RSI top corresponding to the market top
        prev_rsi_top_idx = None
        for i in range(prev_market_top_idx, -1, -1):
            if dataframe['rsi_top'].iloc[i] == 1:
                prev_rsi_top_idx = i
                break
        
        if prev_rsi_top_idx is None:
            return False
            
        # Check for valid divergence
        if prev_rsi_top_idx > prev_market_top_idx:
            return False

        # 3. Compare
        prev_price_top = dataframe['close'].iloc[prev_market_top_idx]
        curr_price_top = dataframe['close'].iloc[idx]
        prev_rsi_top = dataframe['rsi'].iloc[prev_rsi_top_idx]
        curr_rsi_top = dataframe['rsi'].iloc[idx]

        is_price_higher_high = (curr_price_top > prev_price_top)
        is_rsi_lower_high = (curr_rsi_top < prev_rsi_top)

        return is_price_higher_high and is_rsi_lower_high
