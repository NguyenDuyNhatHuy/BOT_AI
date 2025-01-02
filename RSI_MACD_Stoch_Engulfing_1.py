from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import pandas as pd
import numpy as np
import talib.abstract as ta

class RSI_MACD_Stoch_Engulfing_1(IStrategy):
    """
    Ví dụ Strategy sử dụng Engulfing + phát hiện phân kỳ RSI, Stoch (phiên bản pivot).
    """

    can_short: bool = True
    timeframe = '30m'

    minimal_roi = {"0": 0.2}
    stoploss = -0.10

    # --------------------------------------------------------
    # 1) HÀM TÌM PIVOT (ĐÁY / ĐỈNH)
    # --------------------------------------------------------
    def find_pivot_lows(self, df: pd.DataFrame) -> pd.Series:
        """
        Trả về Series True/False đánh dấu các pivot low (đáy cục bộ).
        pivot low nếu close[i] < close[i-1] và close[i] < close[i+1].
        """
        lows = [False] * len(df)
        for i in range(1, len(df) - 1):
            if (df['close'][i] < df['close'][i - 1]) and (df['close'][i] < df['close'][i + 1]):
                lows[i] = True
        return pd.Series(lows, index=df.index)

    def find_pivot_highs(self, df: pd.DataFrame) -> pd.Series:
        """
        Trả về Series True/False đánh dấu các pivot high (đỉnh cục bộ).
        pivot high nếu close[i] > close[i-1] và close[i] > close[i+1].
        """
        highs = [False] * len(df)
        for i in range(1, len(df) - 1):
            if (df['close'][i] > df['close'][i - 1]) and (df['close'][i] > df['close'][i + 1]):
                highs[i] = True
        return pd.Series(highs, index=df.index)

    # --------------------------------------------------------
    # 2) HÀM PHÁT HIỆN PHÂN KỲ REGULAR / HIDDEN, BULLISH / BEARISH
    #    (STOCH & RSI) DỰA TRÊN PIVOT
    # --------------------------------------------------------
    def detect_regular_bullish_divergence_stoch(self, df: pd.DataFrame) -> pd.Series:
        """
        Regular Bullish Divergence (Stoch): 
          - Giá tạo đáy thấp hơn (pivot low mới < pivot low cũ).
          - Stoch tạo đáy cao hơn (pivot low mới > pivot low cũ).
        """
        # Tìm pivot low
        df['pivot_low'] = self.find_pivot_lows(df)
        
        # Chuẩn bị mảng 0/1 để đánh dấu phân kỳ
        divergence = [0] * len(df)
        
        # Lấy index của các pivot low
        pivot_indices = df.index[df['pivot_low'] == True].tolist()
        
        # So sánh từng cặp pivot liên tiếp
        for i in range(1, len(pivot_indices)):
            idx_old = pivot_indices[i - 1]
            idx_new = pivot_indices[i]
            # lower low về giá?
            price_lower_low = df['close'][idx_new] < df['close'][idx_old]
            # higher low về Stoch?
            stoch_higher_low = df['stoch_k'][idx_new] > df['stoch_k'][idx_old]
            if price_lower_low and stoch_higher_low:
                divergence[idx_new] = 1
        
        return pd.Series(divergence, index=df.index)

    def detect_regular_bullish_divergence_rsi(self, df: pd.DataFrame) -> pd.Series:
        """
        Regular Bullish Divergence (RSI):
          - Giá tạo đáy thấp hơn.
          - RSI tạo đáy cao hơn.
        """
        df['pivot_low'] = self.find_pivot_lows(df)
        divergence = [0] * len(df)
        pivot_indices = df.index[df['pivot_low'] == True].tolist()
        
        for i in range(1, len(pivot_indices)):
            idx_old = pivot_indices[i - 1]
            idx_new = pivot_indices[i]
            price_lower_low = df['close'][idx_new] < df['close'][idx_old]
            rsi_higher_low = df['rsi'][idx_new] > df['rsi'][idx_old]
            if price_lower_low and rsi_higher_low:
                divergence[idx_new] = 1
        
        return pd.Series(divergence, index=df.index)

    def detect_regular_bearish_divergence_stoch(self, df: pd.DataFrame) -> pd.Series:
        """
        Regular Bearish Divergence (Stoch):
          - Giá tạo đỉnh cao hơn.
          - Stoch tạo đỉnh thấp hơn.
        """
        df['pivot_high'] = self.find_pivot_highs(df)
        divergence = [0] * len(df)
        pivot_indices = df.index[df['pivot_high'] == True].tolist()

        for i in range(1, len(pivot_indices)):
            idx_old = pivot_indices[i - 1]
            idx_new = pivot_indices[i]
            price_higher_high = df['close'][idx_new] > df['close'][idx_old]
            stoch_lower_high = df['stoch_k'][idx_new] < df['stoch_k'][idx_old]
            if price_higher_high and stoch_lower_high:
                divergence[idx_new] = 1

        return pd.Series(divergence, index=df.index)

    def detect_regular_bearish_divergence_rsi(self, df: pd.DataFrame) -> pd.Series:
        """
        Regular Bearish Divergence (RSI):
          - Giá tạo đỉnh cao hơn.
          - RSI tạo đỉnh thấp hơn.
        """
        df['pivot_high'] = self.find_pivot_highs(df)
        divergence = [0] * len(df)
        pivot_indices = df.index[df['pivot_high'] == True].tolist()

        for i in range(1, len(pivot_indices)):
            idx_old = pivot_indices[i - 1]
            idx_new = pivot_indices[i]
            price_higher_high = df['close'][idx_new] > df['close'][idx_old]
            rsi_lower_high = df['rsi'][idx_new] < df['rsi'][idx_old]
            if price_higher_high and rsi_lower_high:
                divergence[idx_new] = 1

        return pd.Series(divergence, index=df.index)

    def detect_hidden_bullish_divergence_stoch(self, df: pd.DataFrame) -> pd.Series:
        """
        Hidden Bullish Divergence (Stoch):
          - Giá tạo đáy cao hơn (pivot low mới > pivot low cũ).
          - Stoch tạo đáy thấp hơn (pivot low mới < pivot low cũ).
        """
        df['pivot_low'] = self.find_pivot_lows(df)
        divergence = [0] * len(df)
        pivot_indices = df.index[df['pivot_low'] == True].tolist()

        for i in range(1, len(pivot_indices)):
            idx_old = pivot_indices[i - 1]
            idx_new = pivot_indices[i]
            price_higher_low = df['close'][idx_new] > df['close'][idx_old]
            stoch_lower_low = df['stoch_k'][idx_new] < df['stoch_k'][idx_old]
            if price_higher_low and stoch_lower_low:
                divergence[idx_new] = 1

        return pd.Series(divergence, index=df.index)

    def detect_hidden_bullish_divergence_rsi(self, df: pd.DataFrame) -> pd.Series:
        """
        Hidden Bullish Divergence (RSI):
          - Giá tạo đáy cao hơn.
          - RSI tạo đáy thấp hơn.
        """
        df['pivot_low'] = self.find_pivot_lows(df)
        divergence = [0] * len(df)
        pivot_indices = df.index[df['pivot_low'] == True].tolist()

        for i in range(1, len(pivot_indices)):
            idx_old = pivot_indices[i - 1]
            idx_new = pivot_indices[i]
            price_higher_low = df['close'][idx_new] > df['close'][idx_old]
            rsi_lower_low = df['rsi'][idx_new] < df['rsi'][idx_old]
            if price_higher_low and rsi_lower_low:
                divergence[idx_new] = 1

        return pd.Series(divergence, index=df.index)

    def detect_hidden_bearish_divergence_stoch(self, df: pd.DataFrame) -> pd.Series:
        """
        Hidden Bearish Divergence (Stoch):
          - Giá tạo đỉnh thấp hơn.
          - Stoch tạo đỉnh cao hơn.
        """
        df['pivot_high'] = self.find_pivot_highs(df)
        divergence = [0] * len(df)
        pivot_indices = df.index[df['pivot_high'] == True].tolist()

        for i in range(1, len(pivot_indices)):
            idx_old = pivot_indices[i - 1]
            idx_new = pivot_indices[i]
            price_lower_high = df['close'][idx_new] < df['close'][idx_old]
            stoch_higher_high = df['stoch_k'][idx_new] > df['stoch_k'][idx_old]
            if price_lower_high and stoch_higher_high:
                divergence[idx_new] = 1

        return pd.Series(divergence, index=df.index)

    def detect_hidden_bearish_divergence_rsi(self, df: pd.DataFrame) -> pd.Series:
        """
        Hidden Bearish Divergence (RSI):
          - Giá tạo đỉnh thấp hơn.
          - RSI tạo đỉnh cao hơn.
        """
        df['pivot_high'] = self.find_pivot_highs(df)
        divergence = [0] * len(df)
        pivot_indices = df.index[df['pivot_high'] == True].tolist()

        for i in range(1, len(pivot_indices)):
            idx_old = pivot_indices[i - 1]
            idx_new = pivot_indices[i]
            price_lower_high = df['close'][idx_new] < df['close'][idx_old]
            rsi_higher_high = df['rsi'][idx_new] > df['rsi'][idx_old]
            if price_lower_high and rsi_higher_high:
                divergence[idx_new] = 1

        return pd.Series(divergence, index=df.index)

    # --------------------------------------------------------
    # (Giữ nguyên các hàm Engulfing & filter cũ)
    # --------------------------------------------------------
    def filter_candles_10_percent(self, df: pd.DataFrame) -> pd.Series:
        """
        So sánh thân nến hiện tại (n) với thân nến trước (n-1):
          - ratio_n        = (close_n - open_n) / open_n
          - ratio_n_minus1 = (close_(n-1) - open_(n-1)) / open_(n-1)
          - Tính thay đổi %: abs( (ratio_n - ratio_n_minus1) / ratio_n_minus1 )
          => True nếu thay đổi > 0.2 (20%).
        """
        # ratio n
        df['ratio_n'] = (df['close'] - df['open']) / df['open']

        # ratio (n-1)
        df['ratio_n_minus_1'] = (
            (df['close'].shift(1) - df['open'].shift(1)) 
            / df['open'].shift(1)
        )

        # Tránh chia 0
        df['ratio_n_minus_1'] = df['ratio_n_minus_1'].replace(0, np.nan)

        # Tính thay đổi %
        df['ratio_change'] = abs(
            (df['ratio_n'] - df['ratio_n_minus_1']) 
            / df['ratio_n_minus_1']
        )

        # Ví dụ: > 0.2 => thay đổi lớn hơn 20%
        condition = df['ratio_change'] > 0.2
        return condition

    def detect_engulfing_no_sideway(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engulfing đảo chiều không sideway (2 nến):
         - Bullish Engulfing: 
            + Nến (n-1) đỏ
            + Nến (n) xanh
            + open(n) <= close(n-1) & close(n) >= open(n-1)
         - Bearish Engulfing:
            + Nến (n-1) xanh
            + Nến (n) đỏ
            + open(n) >= close(n-1) & close(n) <= open(n-1)
        => df['my_engulfing'] = +1 / -1 / 0
        """
        cond_prev_red    = (df['close'].shift(1) < df['open'].shift(1))
        cond_curr_green  = (df['close'] > df['open'])
        cond_bullish = (
            cond_prev_red &
            cond_curr_green &
            (df['open'] <= df['close'].shift(1)) &
            (df['close'] >= df['open'].shift(1))
        )

        cond_prev_green = (df['close'].shift(1) > df['open'].shift(1))
        cond_curr_red   = (df['close'] < df['open'])
        cond_bearish = (
            cond_prev_green &
            cond_curr_red &
            (df['open'] >= df['close'].shift(1)) &
            (df['close'] <= df['open'].shift(1))
        )

        df.loc[cond_bullish, 'my_engulfing'] = 1
        df.loc[cond_bearish, 'my_engulfing'] = -1

        return df

    def detect_engulfing_sideway(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engulfing đảo chiều CÓ sideway:
        - Nến gốc (i) => xác định sideway_low/high
        - Tìm nến j breakout (đóng cửa thoát khỏi khung sideway_low, sideway_high).
        - Nến (j) breakout => 
            nếu nến gốc đỏ   => breakout lên => bullish
            nếu nến gốc xanh => breakout xuống => bearish
        => Ghi +1 / -1 vào df['my_engulfing'] tại nến j (breakout).
        """
        nrows = len(df)
        i = 0
        while i < nrows - 1:
            open_i = df.at[i, 'open']
            close_i = df.at[i, 'close']
            sideway_low = min(open_i, close_i)
            sideway_high = max(open_i, close_i)

            is_red   = (close_i < open_i)
            is_green = (close_i > open_i)

            j = i + 1
            # Duyệt liên tiếp các nến trong [sideway_low, sideway_high]
            while j < nrows:
                c_j = df.at[j, 'close']
                if sideway_low <= c_j <= sideway_high:
                    j += 1
                else:
                    break

            if j < nrows:
                breakout_close = df.at[j, 'close']
                if is_red and breakout_close > sideway_high:
                    df.at[j, 'my_engulfing'] = 1
                if is_green and breakout_close < sideway_low:
                    df.at[j, 'my_engulfing'] = -1

            i += 1
        return df

    # --------------------------------------------------------
    # 3) POPULATE INDICATORS - gọi hàm phân kỳ pivot mới
    # --------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Khai báo các chỉ báo cần thiết, và gọi hàm Engulfing (cả 2-nến & sideway).
        """
        # (1) Gọi hàm filter_candles_10_percent
        dataframe['ratio'] = self.filter_candles_10_percent(dataframe)

        # (2) Tính RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # (3) Stochastic
        stoch_d, stoch_k  = ta.STOCH(
            dataframe['high'], dataframe['low'], dataframe['close'], 
            fastk_period=13, slowk_period=3, slowd_period=3
        )
        dataframe['stoch_k'] = stoch_k
        dataframe['stoch_d'] = stoch_d

        # (4) Phát hiện phân kỳ (pivot-based)
        dataframe['regular_bullish_divergence_stoch'] = self.detect_regular_bullish_divergence_stoch(dataframe)
        dataframe['regular_bearish_divergence_stoch'] = self.detect_regular_bearish_divergence_stoch(dataframe)
        dataframe['regular_bullish_divergence_rsi']   = self.detect_regular_bullish_divergence_rsi(dataframe)
        dataframe['regular_bearish_divergence_rsi']   = self.detect_regular_bearish_divergence_rsi(dataframe)
        dataframe['hidden_bullish_divergence_stoch']  = self.detect_hidden_bullish_divergence_stoch(dataframe)
        dataframe['hidden_bearish_divergence_stoch']  = self.detect_hidden_bearish_divergence_stoch(dataframe)
        dataframe['hidden_bullish_divergence_rsi']    = self.detect_hidden_bullish_divergence_rsi(dataframe)
        dataframe['hidden_bearish_divergence_rsi']    = self.detect_hidden_bearish_divergence_rsi(dataframe)

        # (5) Đường EMA 200
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        # (6) Xác định Engulfing
        dataframe['my_engulfing'] = 0
        dataframe = self.detect_engulfing_no_sideway(dataframe)
        dataframe = self.detect_engulfing_sideway(dataframe)

        return dataframe

    # --------------------------------------------------------
    # 4) TÍN HIỆU VÀO LỆNH
    # --------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        - Cộng điểm (signal_score_bull, signal_score_bear).
        - Xác định điều kiện Long/Short.
        """

        # Khởi tạo cột enter_long / enter_short
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        # (A) Tín hiệu BULLISH
        dataframe['signal_score_bull'] = 0
        # Engulfing dương => +2
        dataframe['signal_score_bull'] += (dataframe['my_engulfing'] > 0) * 2

        # Phân kỳ ẩn Bullish => +1
        dataframe['signal_score_bull'] += dataframe['hidden_bullish_divergence_stoch'].astype(int)
        dataframe['signal_score_bull'] += dataframe['hidden_bullish_divergence_rsi'].astype(int)
        # Phân kỳ thường Bullish => +1
        dataframe['signal_score_bull'] += dataframe['regular_bullish_divergence_stoch'].astype(int)
        dataframe['signal_score_bull'] += dataframe['regular_bullish_divergence_rsi'].astype(int)

        # RSI < 30 => +1
        dataframe['signal_score_bull'] += (dataframe['rsi'] < 30).astype(int)
        # Stoch < 20 => +1
        dataframe['signal_score_bull'] += (dataframe['stoch_k'] < 20).astype(int)

        dataframe.loc[
            (dataframe['close'] > dataframe['ema200']) &
            (dataframe['signal_score_bull'] >= 3) &
            (dataframe['close'] > dataframe['open']),
            'enter_long'
        ] = 1

        # (B) Tín hiệu BEARISH
        dataframe['signal_score_bear'] = 0
        # Engulfing âm => +2
        dataframe['signal_score_bear'] += (dataframe['my_engulfing'] < 0) * 2

        # Phân kỳ ẩn Bullish => +1
        dataframe['signal_score_bear'] += dataframe['hidden_bullish_divergence_stoch'].astype(int)
        dataframe['signal_score_bear'] += dataframe['hidden_bullish_divergence_rsi'].astype(int)
        # Phân kỳ thường Bullish => +1
        dataframe['signal_score_bear'] += dataframe['regular_bullish_divergence_stoch'].astype(int)
        dataframe['signal_score_bear'] += dataframe['regular_bullish_divergence_rsi'].astype(int)

        # Phân kỳ ẩn Bearish => +1
        dataframe['signal_score_bear'] += dataframe['hidden_bearish_divergence_stoch'].astype(int)
        dataframe['hidden_bearish_divergence_rsi'].astype(int)
        # Phân kỳ thường Bearish => +1
        dataframe['signal_score_bear'] += dataframe['regular_bearish_divergence_stoch'].astype(int)
        dataframe['signal_score_bear'] += dataframe['regular_bearish_divergence_rsi'].astype(int)

        # RSI > 70 => +1
        dataframe['signal_score_bear'] += (dataframe['rsi'] > 70).astype(int)
        # Stoch > 80 => +1
        dataframe['signal_score_bear'] += (dataframe['stoch_k'] > 80).astype(int)

        dataframe.loc[
            (dataframe['close'] < dataframe['ema200']) &
            (dataframe['signal_score_bear'] >= 3) &
            (dataframe['close'] < dataframe['open']),
            'enter_short'
        ] = 1

        dataframe.to_csv('entry_data.csv')
        return dataframe

    # --------------------------------------------------------
    # 5) TÍN HIỆU THOÁT LỆNH
    # --------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Thoát lệnh đơn giản (tùy chỉnh theo ý muốn).
        """
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        # Ví dụ: nếu muốn exit khi RSI > 50
        # dataframe.loc[
        #     (dataframe['rsi'] > 50),
        #     ['exit_long', 'exit_short']
        # ] = 1

        return dataframe
