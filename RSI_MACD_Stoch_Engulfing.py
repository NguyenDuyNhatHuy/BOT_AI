from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class RSI_MACD_Stoch_Engulfing(IStrategy):
    timeframe = '1h'  # Khung thời gian 1 giờ
    minimal_roi = {  # Thiết lập lợi nhuận tối thiểu
        "0": 0.1,
    }
    stoploss = -0.2  # Dừng lỗ tại -20%

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI(14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # MACD và Signal Line
        macd, macdsignal, macdhist = ta.MACD(dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd
        dataframe['macdsignal'] = macdsignal

        # Stochastic Oscillator
        stoch_k, stoch_d = ta.STOCH(dataframe['high'], dataframe['low'], dataframe['close'], 
                                    fastk_period=13, slowk_period=3, slowd_period=3)
        dataframe['stoch_k'] = stoch_k
        dataframe['stoch_d'] = stoch_d

        # Mô hình nến Engulfing
        dataframe['engulfing'] = ta.CDLENGULFING(dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close'])

        # Xác định sideway (theo logic mới)
        dataframe['is_sideway'] = False
        for i in range(1, len(dataframe)):
            prev_close = dataframe['close'].iloc[i - 1]
            curr_close = dataframe['close'].iloc[i]
            prev_open = dataframe['open'].iloc[i - 1]

            # Kiểm tra điểm đảo chiều tăng hoặc giảm
            if prev_close < prev_open and curr_close > prev_open:
                dataframe['is_sideway'].iloc[i] = True  # Đảo chiều tăng
            elif prev_close > prev_open and curr_close < prev_open:
                dataframe['is_sideway'].iloc[i] = True  # Đảo chiều giảm

        return dataframe

    def identify_peaks_and_troughs_structure(self, dataframe: DataFrame) -> DataFrame:
        # Đỉnh và đáy dựa trên cấu trúc thị trường (phục vụ phân kỳ thường)
        dataframe['structure_peak'] = (dataframe['stoch_k'] > dataframe['stoch_k'].shift(1)) & (dataframe['stoch_k'] > dataframe['stoch_k'].shift(-1))
        dataframe['structure_trough'] = (dataframe['stoch_k'] < dataframe['stoch_k'].shift(1)) & (dataframe['stoch_k'] < dataframe['stoch_k'].shift(-1))

        # Loại bỏ đỉnh giả: chỉ giữ đỉnh cao nhất trong khoảng thời gian
        dataframe['valid_peak'] = dataframe['structure_peak'] & (
            dataframe['stoch_k'] >= dataframe['stoch_k'].rolling(window=5, center=True).max()
        )

        # Loại bỏ đáy giả: chỉ giữ đáy thấp nhất trong khoảng thời gian
        dataframe['valid_trough'] = dataframe['structure_trough'] & (
            dataframe['stoch_k'] <= dataframe['stoch_k'].rolling(window=5, center=True).min()
        )

        return dataframe

    def identify_all_peaks_and_troughs(self, dataframe: DataFrame) -> DataFrame:
        # Đỉnh và đáy bất kỳ (phục vụ phân kỳ ẩn)
        dataframe['any_peak'] = (dataframe['stoch_k'] > dataframe['stoch_k'].shift(1)) & (dataframe['stoch_k'] > dataframe['stoch_k'].shift(-1))
        dataframe['any_trough'] = (dataframe['stoch_k'] < dataframe['stoch_k'].shift(1)) & (dataframe['stoch_k'] < dataframe['stoch_k'].shift(-1))
        return dataframe

    def divergence(self, dataframe: DataFrame, col1: str, col2: str, divergence_type: str) -> DataFrame:
        if divergence_type == 'regular':
            dataframe = self.identify_peaks_and_troughs_structure(dataframe)

            # Phân kỳ thường (Regular Divergence)
            dataframe['bullish_divergence_regular'] = (
                (dataframe[col1] > dataframe[col1].shift(1)) &  # Giá tạo đáy cao hơn
                (dataframe['valid_trough']) &  # Stochastic tạo đáy thấp hơn
                (dataframe['stoch_k'] < dataframe['stoch_k'].shift(1))
            )
            dataframe['bearish_divergence_regular'] = (
                (dataframe[col1] < dataframe[col1].shift(1)) &  # Giá tạo đỉnh thấp hơn
                (dataframe['valid_peak']) &  # Stochastic tạo đỉnh cao hơn
                (dataframe['stoch_k'] > dataframe['stoch_k'].shift(1))
            )

        elif divergence_type == 'hidden':
            dataframe = self.identify_all_peaks_and_troughs(dataframe)

            # Phân kỳ ẩn (Hidden Divergence)
            dataframe['bullish_divergence_hidden'] = (
                (dataframe[col1] < dataframe[col1].shift(1)) &  # Giá tạo đáy thấp hơn
                (dataframe['any_trough']) &  # Stochastic tạo đáy cao hơn hoặc bất kỳ
                (dataframe['stoch_k'] > dataframe['stoch_k'].shift(1))
            )
            dataframe['bearish_divergence_hidden'] = (
                (dataframe[col1] > dataframe[col1].shift(1)) &  # Giá tạo đỉnh cao hơn
                (dataframe['any_peak']) &  # Stochastic tạo đỉnh thấp hơn hoặc bất kỳ
                (dataframe['stoch_k'] < dataframe['stoch_k'].shift(1))
            )

        return dataframe

    # Tín hiệu Long Buy
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Phân kỳ RSI và MACD
        dataframe = self.divergence(dataframe, 'close', 'rsi', 'regular')
        dataframe = self.divergence(dataframe, 'close', 'rsi', 'hidden')
        dataframe = self.divergence(dataframe, 'close', 'macd', 'regular')
        dataframe = self.divergence(dataframe, 'close', 'macd', 'hidden')

        # Tính điểm tín hiệu
        dataframe['signal_score'] = 0
        dataframe['signal_score'] += (dataframe['engulfing'] > 0) *2 #& (~dataframe['is_sideway'])  # Engulfing hợp lệ được tính 2 điểm
        dataframe['signal_score'] += dataframe['bullish_divergence_hidden']  # Phân kỳ ẩn Bullish 1 điểm
        dataframe['signal_score'] += dataframe['bullish_divergence_regular']  # Phân kỳ thường Bullish 1 điểm
        dataframe['signal_score'] += (dataframe['rsi'] < 30).astype(int)      # RSI < 30 1 điểm
        dataframe['signal_score'] += (dataframe['macd'] == dataframe['macdsignal']).astype(int)  # MACD cắt lên Signal Line 1 điểm
        dataframe['signal_score'] += (dataframe['stoch_k'] < 20).astype(int)  # Stochastic < 20 1 điểm

        # Điều kiện Mua nếu tổng điểm >= 3
        dataframe.loc[
            (dataframe['signal_score'] >= 4),
            'buy'] = 1

        return dataframe

    # Tín hiệu Short Sell
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Phân kỳ RSI và MACD
        dataframe = self.divergence(dataframe, 'close', 'rsi', 'regular')
        dataframe = self.divergence(dataframe, 'close', 'rsi', 'hidden')
        dataframe = self.divergence(dataframe, 'close', 'macd', 'regular')
        dataframe = self.divergence(dataframe, 'close', 'macd', 'hidden')

        # Tính điểm tín hiệu
        dataframe['signal_score'] = 0
        dataframe['signal_score'] += (dataframe['engulfing'] < 0)*2 #& (~dataframe['is_sideway'])  # Engulfing hợp lệ được tính 2 điểm
        dataframe['signal_score'] += dataframe['bearish_divergence_hidden'] 
        dataframe['signal_score'] += dataframe['bearish_divergence_regular']  # Phân kỳ thường Bearish 1 điểm

        dataframe['signal_score'] += (dataframe['rsi'] > 70).astype(int)      # RSI > 70 1 điểm
        dataframe['signal_score'] += (dataframe['macd'] == dataframe['macdsignal']).astype(int)  # MACD cắt xuống Signal Line 1 điểm
        dataframe['signal_score'] += (dataframe['stoch_k'] > 80).astype(int)  # Stochastic > 80 1 điểm
        
        # Điều kiện Bán nếu tổng điểm >= 3
        dataframe.loc[
            (dataframe['signal_score'] >= 4),
            'sell'] = 1

        return dataframe
