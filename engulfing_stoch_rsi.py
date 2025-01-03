from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import pandas as pd
import numpy as np
import talib.abstract as ta

class engulfing(IStrategy):
    """
    Ví dụ Strategy sử dụng Engulfing theo 2 dạng:
      1) Engulfing đảo chiều không sideway (2 nến).
      2) Engulfing đảo chiều có sideway (nến gốc + breakout).

    Kết hợp thêm kiểm tra (close - open) so sánh với nến trước, 
    và phân kỳ RSI, Stoch để tính signal_score.
    
    BỔ SUNG: 
      - Chỉ Long khi xu hướng tăng (close > EMA200).
      - Chỉ Short khi xu hướng giảm (close < EMA200).
    """

    can_short: bool = True
    timeframe = '30m'

    minimal_roi = {"0": 0.20}
    stoploss = -0.10

    # ------------------------------------------------------------------------
    # Bộ lọc so sánh "thay đổi lớn hơn 20%" giữa nến hiện tại và nến trước
    # ------------------------------------------------------------------------
    def filter_candles_10_percent(self, df: pd.DataFrame) -> pd.Series:
        df['ratio_n'] = (df['close'] - df['open']) / df['open']
        df['ratio_n_minus_1'] = (
            (df['close'].shift(1) - df['open'].shift(1)) 
            / df['open'].shift(1)
        )
        df['ratio_n_minus_1'] = df['ratio_n_minus_1'].replace(0, np.nan)
        df['ratio_change'] = abs(
            (df['ratio_n'] - df['ratio_n_minus_1']) 
            / df['ratio_n_minus_1']
        )
        return df['ratio_change'] > 0.2

    # ------------------------------------------------------------------------
    # Các hàm phát hiện phân kỳ (divergence) - tối ưu bằng vector hoá
    # ------------------------------------------------------------------------
    def detect_regular_bullish_divergence_stoch(self, df: DataFrame) -> pd.Series:
        df['regular_bullish_divergence_stoch'] = 0
        cond = (
            (df['close'] < df['close'].shift(1)) &
            (df['stoch_k'] > df['stoch_k'].shift(1))
        )
        df.loc[cond, 'regular_bullish_divergence_stoch'] = 1
        return df['regular_bullish_divergence_stoch']

    def detect_regular_bullish_divergence_rsi(self, df: DataFrame) -> pd.Series:
        df['regular_bullish_divergence_rsi'] = 0
        cond = (
            (df['close'] < df['close'].shift(1)) &
            (df['rsi'] > df['rsi'].shift(1))
        )
        df.loc[cond, 'regular_bullish_divergence_rsi'] = 1
        return df['regular_bullish_divergence_rsi']

    def detect_regular_bearish_divergence_stoch(self, df: DataFrame) -> pd.Series:
        df['regular_bearish_divergence_stoch'] = 0
        cond = (
            (df['close'] > df['close'].shift(1)) &
            (df['stoch_k'] < df['stoch_k'].shift(1))
        )
        df.loc[cond, 'regular_bearish_divergence_stoch'] = 1
        return df['regular_bearish_divergence_stoch']

    def detect_regular_bearish_divergence_rsi(self, df: DataFrame) -> pd.Series:
        df['regular_bearish_divergence_rsi'] = 0
        cond = (
            (df['close'] > df['close'].shift(1)) &
            (df['rsi'] < df['rsi'].shift(1))
        )
        df.loc[cond, 'regular_bearish_divergence_rsi'] = 1
        return df['regular_bearish_divergence_rsi']

    def detect_hidden_bullish_divergence_stoch(self, df: DataFrame) -> pd.Series:
        df['hidden_bullish_divergence_stoch'] = 0
        cond = (
            (df['close'] > df['close'].shift(1)) &
            (df['stoch_k'] < df['stoch_k'].shift(1))
        )
        df.loc[cond, 'hidden_bullish_divergence_stoch'] = 1
        return df['hidden_bullish_divergence_stoch']

    def detect_hidden_bullish_divergence_rsi(self, df: DataFrame) -> pd.Series:
        df['hidden_bullish_divergence_rsi'] = 0
        cond = (
            (df['close'] > df['close'].shift(1)) &
            (df['rsi'] < df['rsi'].shift(1))
        )
        df.loc[cond, 'hidden_bullish_divergence_rsi'] = 1
        return df['hidden_bullish_divergence_rsi']

    def detect_hidden_bearish_divergence_stoch(self, df: DataFrame) -> pd.Series:
        df['hidden_bearish_divergence_stoch'] = 0
        cond = (
            (df['close'] < df['close'].shift(1)) &
            (df['stoch_k'] > df['stoch_k'].shift(1))
        )
        df.loc[cond, 'hidden_bearish_divergence_stoch'] = 1
        return df['hidden_bearish_divergence_stoch']

    def detect_hidden_bearish_divergence_rsi(self, df: DataFrame) -> pd.Series:
        df['hidden_bearish_divergence_rsi'] = 0
        cond = (
            (df['close'] < df['close'].shift(1)) &
            (df['rsi'] > df['rsi'].shift(1))
        )
        df.loc[cond, 'hidden_bearish_divergence_rsi'] = 1
        return df['hidden_bearish_divergence_rsi']

    # ------------------------------------------------------------------------
    # Engulfing 2-nến
    # ------------------------------------------------------------------------
    def detect_engulfing_no_sideway(self, df: DataFrame) -> DataFrame:
        # Nến (n-1) đỏ
        cond_prev_red = (df['close'].shift(1) < df['open'].shift(1))
        # Nến (n) xanh
        cond_curr_green = (df['close'] > df['open'])

        cond_bullish = (
            cond_prev_red &
            cond_curr_green &
            (df['open'] <= df['close'].shift(1)) &
            (df['close'] >= df['open'].shift(1))
        )

        # Nến (n-1) xanh
        cond_prev_green = (df['close'].shift(1) > df['open'].shift(1))
        # Nến (n) đỏ
        cond_curr_red = (df['close'] < df['open'])

        cond_bearish = (
            cond_prev_green &
            cond_curr_red &
            (df['open'] >= df['close'].shift(1)) &
            (df['close'] <= df['open'].shift(1))
        )

        df.loc[cond_bullish, 'my_engulfing'] = 1
        df.loc[cond_bearish, 'my_engulfing'] = -1

        return df

    # ------------------------------------------------------------------------
    # Engulfing sideway: *KHÔNG GIỚI HẠN* số nến sideway
    # ------------------------------------------------------------------------
    def detect_engulfing_sideway(self, df: DataFrame) -> DataFrame:
        nrows = len(df)
        i = 0
        while i < nrows - 1:
            open_i = df.at[i, 'open']
            close_i = df.at[i, 'close']
            sideway_low = min(open_i, close_i)
            sideway_high = max(open_i, close_i)

            # Xác định màu nến gốc
            is_red = (close_i < open_i)
            is_green = (close_i > open_i)

            j = i + 1

            # Quét các nến kế tiếp đến khi breakout
            while j < nrows:
                c_j = df.at[j, 'close']
                if sideway_low <= c_j <= sideway_high:
                    j += 1
                else:
                    break

            if j < nrows:
                breakout_close = df.at[j, 'close']
                # nến gốc đỏ => breakout lên => +1
                if is_red and breakout_close > sideway_high:
                    df.at[j, 'my_engulfing'] = 1
                # nến gốc xanh => breakout xuống => -1
                if is_green and breakout_close < sideway_low:
                    df.at[j, 'my_engulfing'] = -1

            i += 1

        return df

    # ------------------------------------------------------------------------
    # populate_indicators: khai báo chỉ báo, gọi hàm Engulfing & Divergence
    # ------------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # (1) Gọi hàm filter
        dataframe['ratio'] = self.filter_candles_10_percent(dataframe)

        # (2) RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # (3) Stochastic Oscillator
        stoch_d, stoch_k = ta.STOCH(
            dataframe['high'], 
            dataframe['low'], 
            dataframe['close'], 
            fastk_period=13, 
            slowk_period=3, 
            slowd_period=3
        )
        dataframe['stoch_k'] = stoch_k
        dataframe['stoch_d'] = stoch_d

        # (4) Đường EMA 200 để xác định trend
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        # (5) Phát hiện phân kỳ
        dataframe['regular_bullish_divergence_stoch']  = self.detect_regular_bullish_divergence_stoch(dataframe)
        dataframe['regular_bearish_divergence_stoch']  = self.detect_regular_bearish_divergence_stoch(dataframe)
        dataframe['regular_bullish_divergence_rsi']    = self.detect_regular_bullish_divergence_rsi(dataframe)
        dataframe['regular_bearish_divergence_rsi']    = self.detect_regular_bearish_divergence_rsi(dataframe)
        dataframe['hidden_bullish_divergence_stoch']   = self.detect_hidden_bullish_divergence_stoch(dataframe)
        dataframe['hidden_bearish_divergence_stoch']   = self.detect_hidden_bearish_divergence_stoch(dataframe)
        dataframe['hidden_bullish_divergence_rsi']     = self.detect_hidden_bullish_divergence_rsi(dataframe)
        dataframe['hidden_bearish_divergence_rsi']     = self.detect_hidden_bearish_divergence_rsi(dataframe)

        # (6) Engulfing
        dataframe['my_engulfing'] = 0
        dataframe = self.detect_engulfing_no_sideway(dataframe)
        dataframe = self.detect_engulfing_sideway(dataframe)

        # Debug tùy chọn
        # dataframe.to_csv("debug_indicators.csv")
        return dataframe

    # ------------------------------------------------------------------------
    # populate_entry_trend: Tín hiệu vào lệnh
    # ------------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Khởi tạo cột vào lệnh
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        # A) Tín hiệu BULLISH
        dataframe['signal_score_bull'] = 0
        dataframe['signal_score_bull'] += (dataframe['my_engulfing'] > 0) 
        dataframe['signal_score_bull'] += dataframe['hidden_bullish_divergence_stoch']
        dataframe['signal_score_bull'] += dataframe['hidden_bullish_divergence_rsi']
        dataframe['signal_score_bull'] += dataframe['regular_bullish_divergence_stoch']
        dataframe['signal_score_bull'] += dataframe['regular_bullish_divergence_rsi']
        dataframe['signal_score_bull'] += (dataframe['rsi'] < 30).astype(int)
        dataframe['signal_score_bull'] += (dataframe['stoch_k'] < 20).astype(int)

        # Chỉ LONG khi:
        # 1) Xu hướng TĂNG (close > ema200)
        # 2) signal_score_bull >= 3
        # 3) Nến xanh (close > open)
        dataframe.loc[
            (dataframe['close'] > dataframe['ema200']) &
            (dataframe['signal_score_bull'] >= 3) &
            (dataframe['close'] > dataframe['open']),
            'enter_long'
        ] = 1

        # B) Tín hiệu BEARISH
        dataframe['signal_score_bear'] = 0
        dataframe['signal_score_bear'] += (dataframe['my_engulfing'] < 0) 
        dataframe['signal_score_bear'] += dataframe['hidden_bearish_divergence_stoch']
        dataframe['signal_score_bear'] += dataframe['hidden_bearish_divergence_rsi']
        dataframe['signal_score_bear'] += dataframe['regular_bearish_divergence_stoch']
        dataframe['signal_score_bear'] += dataframe['regular_bearish_divergence_rsi']
        dataframe['signal_score_bear'] += (dataframe['rsi'] > 70).astype(int)
        dataframe['signal_score_bear'] += (dataframe['stoch_k'] > 80).astype(int)

        # Chỉ SHORT khi:
        # 1) Xu hướng GIẢM (close < ema200)
        # 2) signal_score_bear >= 3
        # 3) Nến đỏ (close < open)
        dataframe.loc[
            (dataframe['close'] < dataframe['ema200']) &
            (dataframe['signal_score_bear'] >= 3) &
            (dataframe['close'] < dataframe['open']),
            'enter_short'
        ] = 1

        # Debug tùy chọn
        dataframe.to_csv('entry_data.csv')

        return dataframe

    # ------------------------------------------------------------------------
    # populate_exit_trend: Tín hiệu thoát lệnh
    # ------------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        return dataframe
