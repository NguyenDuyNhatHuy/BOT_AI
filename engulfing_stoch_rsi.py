from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import pandas as pd
import numpy as np
import talib.abstract as ta

class engulfing_stoch_rsi(IStrategy):
    """
    Ví dụ Strategy sử dụng Engulfing theo 2 dạng:
      1) Engulfing đảo chiều không sideway (2 nến).
      2) Engulfing đảo chiều có sideway (nến gốc + breakout).

    Kết hợp thêm kiểm tra (close - open) so sánh với nến trước, 
    và phân kỳ RSI, Stoch để tính signal_score.
    """

    can_short: bool = True
    timeframe = '1h'

    minimal_roi = {"0": 0.1}
    stoploss = -0.2

    # Số nến sideway tối đa
    max_sideway_bars = 3

    def filter_candles_10_percent(self, df: pd.DataFrame) -> pd.Series:
        """
        So sánh thân nến hiện tại (n) với thân nến trước (n-1):
          - ratio_n        = (close_n - open_n) / open_n
          - ratio_n_minus1 = (close_(n-1) - open_(n-1)) / open_(n-1)
          - Tính thay đổi %: abs( (ratio_n - ratio_n_minus1) / ratio_n_minus1 )
          => True nếu thay đổi > 0.2 (20%).

        Trả về 1 Series True/False.
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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Khai báo các chỉ báo cần thiết, và gọi hàm Engulfing (cả 2-nến & sideway).
        """

        # Gọi hàm filter_candles_10_percent -> trả về True/False => gán cột 'ratio'
        # Bạn có thể dùng cột này trong logic mua/bán nếu muốn.
        dataframe['ratio'] = self.filter_candles_10_percent(dataframe)

        # Thêm RSI (chỉ để minh hoạ)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Stochastic Oscillator
        stoch_d, stoch_k = ta.STOCH(dataframe['high'], dataframe['low'], dataframe['close'], fastk_period=13, slowk_period=3, slowd_period=3)
        dataframe['stoch_k'] = stoch_k
        dataframe['stoch_d'] = stoch_d

        # Phân kỳ thường của Stochastic
        dataframe['stoch_regular_bullish'] = self.detect_stoch_regular_bullish_divergence(dataframe)
        dataframe['stoch_regular_bearish'] = self.detect_stoch_regular_bearish_divergence(dataframe)

        # Phân kỳ ẩn của Stochastic
        dataframe['stoch_hidden_bullish'] = self.detect_stoch_hidden_bullish_divergence(dataframe)
        dataframe['stoch_hidden_bearish'] = self.detect_stoch_hidden_bearish_divergence(dataframe)

        # Phân kỳ thường của RSI
        dataframe['rsi_regular_bullish'] = self.detect_rsi_regular_bullish_divergence(dataframe)
        dataframe['rsi_regular_bearish'] = self.detect_rsi_regular_bearish_divergence(dataframe)

        # Phân kỳ ẩn của RSI
        dataframe['rsi_hidden_bullish'] = self.detect_rsi_hidden_bullish_divergence(dataframe)
        dataframe['rsi_hidden_bearish'] = self.detect_rsi_hidden_bearish_divergence(dataframe)

        # Tạo cột my_engulfing = 0 mặc định
        dataframe['my_engulfing'] = 0

        # Xác định Engulfing 2 nến => ghi vào my_engulfing
        dataframe = self.detect_engulfing_no_sideway(dataframe)

        # Xác định Engulfing sideway => ghi vào my_engulfing
        dataframe = self.detect_engulfing_sideway(dataframe)

        return dataframe

    def detect_stoch_regular_bullish_divergence(self, dataframe: DataFrame) -> DataFrame:
        """
        Phát hiện phân kỳ tăng thường (Stochastic): Giá tạo đáy thấp hơn, Stoch tạo đáy cao hơn.
        """
        dataframe['stoch_regular_bullish'] = 0
        for i in range(2, len(dataframe)):
            price_lower_low = dataframe['close'][i] < dataframe['close'][i-1]
            stoch_higher_low = dataframe['stoch_k'][i] > dataframe['stoch_k'][i-1] and dataframe['stoch_k'][i] < 20

            if price_lower_low and stoch_higher_low:
                dataframe['stoch_regular_bullish'][i] = 1
        return dataframe['stoch_regular_bullish']

    def detect_stoch_regular_bearish_divergence(self, dataframe: DataFrame) -> DataFrame:
        """
        Phát hiện phân kỳ giảm thường (Stochastic): Giá tạo đỉnh cao hơn, Stoch tạo đỉnh thấp hơn.
        """
        dataframe['stoch_regular_bearish'] = 0
        for i in range(2, len(dataframe)):
            price_higher_high = dataframe['close'][i] > dataframe['close'][i-1]
            stoch_lower_high = dataframe['stoch_k'][i] < dataframe['stoch_k'][i-1] and dataframe['stoch_k'][i] > 80

            if price_higher_high and stoch_lower_high:
                dataframe['stoch_regular_bearish'][i] = 1
        return dataframe['stoch_regular_bearish']

    def detect_stoch_hidden_bullish_divergence(self, dataframe: DataFrame) -> DataFrame:
        """
        Phát hiện phân kỳ ẩn tăng (Stochastic): Giá tạo đáy cao hơn, Stoch tạo đáy thấp hơn trong vùng giá nhỏ của phân kỳ thường.
        """
        dataframe['stoch_hidden_bullish'] = 0
        for i in range(2, len(dataframe)):
            # Giá tạo đáy cao hơn nhưng trong vùng giá nhỏ của phân kỳ thường
            price_higher_low = dataframe['close'][i] > dataframe['close'][i-1] and dataframe['close'][i-1] > dataframe['close'][i-2]
            
            # Stochastic tạo đáy thấp hơn và nằm dưới 20
            stoch_lower_low = dataframe['stoch_k'][i] < dataframe['stoch_k'][i-1] and dataframe['stoch_k'][i] < 20

            if price_higher_low and stoch_lower_low:
                dataframe['stoch_hidden_bullish'][i] = 1
        return dataframe['stoch_hidden_bullish']

    def detect_stoch_hidden_bearish_divergence(self, dataframe: DataFrame) -> DataFrame:
        """
        Phát hiện phân kỳ ẩn giảm (Stochastic): Giá tạo đỉnh thấp hơn, Stoch tạo đỉnh cao hơn trong vùng giá nhỏ của phân kỳ thường.
        """
        dataframe['stoch_hidden_bearish'] = 0
        for i in range(2, len(dataframe)):
            # Giá tạo đỉnh thấp hơn nhưng trong vùng giá nhỏ của phân kỳ thường
            price_lower_high = dataframe['close'][i] < dataframe['close'][i-1] and dataframe['close'][i-1] < dataframe['close'][i-2]
            
            # Stochastic tạo đỉnh cao hơn và nằm trên 80
            stoch_higher_high = dataframe['stoch_k'][i] > dataframe['stoch_k'][i-1] and dataframe['stoch_k'][i] > 80

            if price_lower_high and stoch_higher_high:
                dataframe['stoch_hidden_bearish'][i] = 1
        return dataframe['stoch_hidden_bearish']
    

    def detect_rsi_regular_bullish_divergence(self, dataframe: DataFrame) -> DataFrame:
        """
        Phát hiện phân kỳ tăng thường (RSI): Giá tạo đáy thấp hơn, RSI tạo đáy cao hơn.
        """
        dataframe['rsi_regular_bullish'] = 0
        for i in range(2, len(dataframe)):
            # Giá tạo đáy thấp hơn
            price_lower_low = dataframe['close'][i] < dataframe['close'][i-1]
            
            # RSI tạo đáy cao hơn
            rsi_higher_low = dataframe['rsi'][i] > dataframe['rsi'][i-1]

            if price_lower_low and rsi_higher_low:
                dataframe['rsi_regular_bullish'][i] = 1
        return dataframe['rsi_regular_bullish']

    def detect_rsi_regular_bearish_divergence(self, dataframe: DataFrame) -> DataFrame:
        """
        Phát hiện phân kỳ giảm thường (RSI): Giá tạo đỉnh cao hơn, RSI tạo đỉnh thấp hơn.
        """
        dataframe['rsi_regular_bearish'] = 0
        for i in range(2, len(dataframe)):
            # Giá tạo đỉnh cao hơn
            price_higher_high = dataframe['close'][i] > dataframe['close'][i-1]
            
            # RSI tạo đỉnh thấp hơn
            rsi_lower_high = dataframe['rsi'][i] < dataframe['rsi'][i-1]

            if price_higher_high and rsi_lower_high:
                dataframe['rsi_regular_bearish'][i] = 1
        return dataframe['rsi_regular_bearish']

    def detect_rsi_hidden_bullish_divergence(self, dataframe: DataFrame) -> DataFrame:
        """
        Phát hiện phân kỳ ẩn tăng (RSI): Giá tạo đáy cao hơn, RSI tạo đáy thấp hơn trong vùng giá nhỏ của phân kỳ thường.
        """
        dataframe['rsi_hidden_bullish'] = 0
        for i in range(2, len(dataframe)):
            # Giá tạo đáy cao hơn nhưng trong vùng giá nhỏ của phân kỳ thường
            price_higher_low = dataframe['close'][i] > dataframe['close'][i-1] and dataframe['close'][i-1] > dataframe['close'][i-2]
            
            # RSI tạo đáy thấp hơn
            rsi_lower_low = dataframe['rsi'][i] < dataframe['rsi'][i-1]

            if price_higher_low and rsi_lower_low:
                dataframe['rsi_hidden_bullish'][i] = 1
        return dataframe['rsi_hidden_bullish']

    def detect_rsi_hidden_bearish_divergence(self, dataframe: DataFrame) -> DataFrame:
        """
        Phát hiện phân kỳ ẩn giảm (RSI): Giá tạo đỉnh thấp hơn, RSI tạo đỉnh cao hơn trong vùng giá nhỏ của phân kỳ thường.
        """
        dataframe['rsi_hidden_bearish'] = 0
        for i in range(2, len(dataframe)):
            # Giá tạo đỉnh thấp hơn nhưng trong vùng giá nhỏ của phân kỳ thường
            price_lower_high = dataframe['close'][i] < dataframe['close'][i-1] and dataframe['close'][i-1] < dataframe['close'][i-2]
            
            # RSI tạo đỉnh cao hơn
            rsi_higher_high = dataframe['rsi'][i] > dataframe['rsi'][i-1]

            if price_lower_high and rsi_higher_high:
                dataframe['rsi_hidden_bearish'][i] = 1
        return dataframe['rsi_hidden_bearish']

    def detect_engulfing_no_sideway(self, df: DataFrame) -> DataFrame:
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

        # Nến (n-1) đỏ
        cond_prev_red = (df['close'].shift(1) < df['open'].shift(1))
        # Nến (n) xanh
        cond_curr_green = (df['close'] > df['open'])

        # Bullish Engulfing (2-nến)
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

        # Bearish Engulfing (2-nến)
        cond_bearish = (
            cond_prev_green &
            cond_curr_red &
            (df['open'] >= df['close'].shift(1)) &
            (df['close'] <= df['open'].shift(1))
        )

        df.loc[cond_bullish, 'my_engulfing'] = 1
        df.loc[cond_bearish, 'my_engulfing'] = -1

        return df

    def detect_engulfing_sideway(self, df: DataFrame) -> DataFrame:
        """
        Engulfing đảo chiều CÓ sideway:
          - Nến gốc (i) => xác định sideway_low/high
          - Tối đa self.max_sideway_bars nến tiếp theo 
            (n+1,...,n+max_sideway_bars) đóng cửa trong [low, high].
          - Nến (j) breakout => nếu nến gốc đỏ => breakout lên => bullish
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

            # Nến gốc đỏ/xanh
            is_red   = (close_i < open_i)
            is_green = (close_i > open_i)

            j = i + 1
            sideway_count = 0

            # Duyệt các nến liên tiếp trong [sideway_low, sideway_high]
            while j < nrows:
                c_j = df.at[j, 'close']
                # Nến j vẫn trong khung sideway?
                if sideway_low <= c_j <= sideway_high:
                    sideway_count += 1
                    # Nếu đã vượt quá max_sideway_bars => dừng
                    if sideway_count >= self.max_sideway_bars:
                        break
                    j += 1
                else:
                    # Nến j = breakout
                    break

            if j < nrows:
                # nến j = breakout
                breakout_close = df.at[j, 'close']
                # nến gốc đỏ => breakout lên => +1
                if is_red and breakout_close > sideway_high:
                    df.at[j, 'my_engulfing'] = 1

                # nến gốc xanh => breakout xuống => -1
                if is_green and breakout_close < sideway_low:
                    df.at[j, 'my_engulfing'] = -1

            i += 1

        return df

    # ------------------------------
    # TÍN HIỆU VÀO LỆNH
    # ------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        - Tính phân kỳ RSI, Stoch.
        - Cộng điểm (signal_score_bull, signal_score_bear).
        - Xác định điều kiện Long/Short.
        """
        # Khởi tạo cột signal scores
        dataframe['signal_score_bull'] = 0
        dataframe['signal_score_bear'] = 0

        # (A) Tín hiệu BULLISH
        # Engulfing dương => +2
        dataframe['signal_score_bull'] += (dataframe['my_engulfing'] > 0).astype(int)
        # Phân kỳ thường Bullish => +1 (hoặc)
        dataframe['signal_score_bull'] += (
            (dataframe['stoch_regular_bullish'] | dataframe['rsi_regular_bullish']).astype(int)
        )
        # Phân kỳ ẩn Bullish => +1 (hoặc)
        dataframe['signal_score_bull'] += (
            (dataframe['stoch_hidden_bullish'] | dataframe['rsi_hidden_bullish']).astype(int)
        )

        

        # RSI < 30 => +1
        dataframe['signal_score_bull'] += (dataframe['rsi'] < 30).astype(int)

        # Stoch < 20 => +1
        dataframe['signal_score_bull'] += (dataframe['stoch_k'] < 20).astype(int)

        # => Long khi tổng điểm >= 3 và nến xanh
        dataframe['enter_long'] = (
            (dataframe['signal_score_bull'] >= 3) &
            (dataframe['close'] > dataframe['open'])
        ).astype(int)

        # (B) Tín hiệu BEARISH
        # Engulfing âm => +2
        dataframe['signal_score_bear'] += (dataframe['my_engulfing'] < 0).astype(int)
        # Phân kỳ thường Bearish => +1 (hoặc)
        dataframe['signal_score_bear'] += (
            (dataframe['stoch_regular_bearish'] | dataframe['rsi_regular_bearish']).astype(int)
        )

        # Phân kỳ ẩn Bearish => +1 (hoặc)
        dataframe['signal_score_bear'] += (
            (dataframe['stoch_hidden_bearish'] | dataframe['rsi_hidden_bearish']).astype(int)
        )


        # RSI > 70 => +1
        dataframe['signal_score_bear'] += (dataframe['rsi'] > 70).astype(int)

        # Stoch > 80 => +1
        dataframe['signal_score_bear'] += (dataframe['stoch_k'] > 80).astype(int)

        # => Short khi tổng điểm >= 3 và nến đỏ
        dataframe['enter_short'] = (
            (dataframe['signal_score_bear'] >= 3) &
            (dataframe['close'] < dataframe['open'])
        ).astype(int)

        return dataframe

    # ------------------------------
    # TÍN HIỆU THOÁT LỆNH
    # ------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Thoát lệnh đơn giản:
        - RSI > 50 => exit (hoặc có thể dùng ROI/Stoploss).
        """
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        dataframe.loc[
            (dataframe['rsi'] > 50),
            ['exit_long', 'exit_short']
        ] = 1

        return dataframe
