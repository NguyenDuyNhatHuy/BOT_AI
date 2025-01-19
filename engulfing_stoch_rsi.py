import numpy as np
import pandas as pd
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from scipy.signal import find_peaks


class MyStrategy(IStrategy):
    timeframe = '30m'
    stoploss = -0.1
    minimal_roi = {"0": 0.12}
    max_open_trades = 4
    allow_entry_with_open_trades = True
    can_short: bool = True

    # ===============================
    # Các hàm hỗ trợ (Helpers)
    # ===============================
    
    # Hàm để tính toán Stochastic
    def calculate_stoch(dataframe: DataFrame, k_period: int = 13, d_period: int = 3) -> DataFrame:
        """
        Thêm các cột Stochastic %K và %D vào dataframe.
        """
        stoch_d, stoch_k = ta.STOCH(
            dataframe['high'], 
            dataframe['low'], 
            dataframe['close'],
            fastk_period=k_period, 
            slowk_period=d_period, 
            slowd_period=d_period
        )
        dataframe['stoch_k'] = stoch_k
        dataframe['stoch_d'] = stoch_d
        return dataframe

    # Hàm để xác định đỉnh/đáy của giá close dựa trên Stochastic
    def identify_price_peaks_and_valleys(dataframe: DataFrame) -> DataFrame:
        """
        Xác định đỉnh/đáy của giá close dựa trên đỉnh/đáy của Stochastic %K.
        """
        # Xác định đỉnh/đáy của Stochastic
        dataframe['stoch_peak'] = (
            (dataframe['stoch_k'] > 80) & 
            (dataframe['stoch_k'].shift(-1) < 80)
        ).astype('int')  # Đỉnh của Stochastic

        dataframe['stoch_valley'] = (
            (dataframe['stoch_k'] < 20) & 
            (dataframe['stoch_k'].shift(-1) > 20)
        ).astype('int')  # Đáy của Stochastic

        # Xác định đỉnh/đáy của giá close
        dataframe['price_peak'] = np.where(dataframe['stoch_peak'] == 1, dataframe['close'], np.nan)
        dataframe['price_valley'] = np.where(dataframe['stoch_valley'] == 1, dataframe['close'], np.nan)

        return dataframe

    def find_rsi_peaks_troughs_scipy(self, df: pd.DataFrame, rsi_col: str = 'rsi', 
                                     price_col: str = 'close', **kwargs) -> pd.DataFrame:
        """
        Xác định đỉnh và đáy của RSI sử dụng scipy.signal.find_peaks.
        
        Thêm các cột:
          - 'rsi_peak': True nếu điểm là đỉnh của RSI.
          - 'rsi_trough': True nếu điểm là đáy của RSI.
          - 'price_at_rsi_peak': Giá đóng cửa tại điểm đỉnh RSI.
          - 'price_at_rsi_trough': Giá đóng cửa tại điểm đáy RSI.
        """
        # Lấy chuỗi RSI
        rsi_series = df[rsi_col]
        
        # Tìm các chỉ số đỉnh và đáy của RSI
        peaks, _ = find_peaks(rsi_series, **kwargs)
        troughs, _ = find_peaks(-rsi_series, **kwargs)
        
        # Khởi tạo cột cho đỉnh và đáy RSI
        df['rsi_peak'] = False
        df['rsi_trough'] = False
        df.loc[df.index[peaks], 'rsi_peak'] = True
        df.loc[df.index[troughs], 'rsi_trough'] = True
        
        # Gán giá đóng cửa tại các điểm đỉnh và đáy RSI
        df['price_at_rsi_peak'] = np.nan
        df['price_at_rsi_trough'] = np.nan
        df.loc[df.index[peaks], 'price_at_rsi_peak'] = df.loc[df.index[peaks], price_col]
        df.loc[df.index[troughs], 'price_at_rsi_trough'] = df.loc[df.index[troughs], price_col]
        
        return df

    def check_rsi_in_stoch_regions(self, df: pd.DataFrame, tolerance: float = 0.0) -> pd.DataFrame:
        """
        Kiểm tra xem giá đóng cửa tại các đỉnh/đáy RSI có thuộc tập hợp giá đóng cửa của 
        các đỉnh/đáy Stoch hay không.

        Các cột cần có trong df:
          - 'price_peak_region': Giá đóng cửa của vùng đỉnh theo Stoch (None nếu không thuộc vùng).
          - 'price_trough_region': Giá đóng cửa của vùng đáy theo Stoch (None nếu không thuộc vùng).
          - 'price_at_rsi_peak': Giá đóng cửa tại đỉnh RSI (NaN nếu không phải đỉnh).
          - 'price_at_rsi_trough': Giá đóng cửa tại đáy RSI (NaN nếu không phải đáy).

        Tham số:
          - tolerance: Khoảng sai số cho so sánh giá (mặc định = 0 => so sánh chính xác).

        Kết quả:
          Thêm 2 cột:
            - 'rsi_peak_in_stoch': True nếu giá tại đỉnh RSI nằm trong tập hợp giá đỉnh Stoch.
            - 'rsi_trough_in_stoch': True nếu giá tại đáy RSI nằm trong tập hợp giá đáy Stoch.
        """
        # Lấy tập hợp các giá đóng cửa từ Stoch (loại bỏ các giá None)
        stoch_peak_prices = df['price_peak_region'].dropna().unique()
        stoch_trough_prices = df['price_trough_region'].dropna().unique()
        
        def is_in_set(price, price_set, tol):
            """
            Kiểm tra xem một giá có thuộc tập hợp price_set không, với khoảng sai số tol.
            """
            if tol == 0:
                return price in price_set
            else:
                return any(abs(price - sp) <= tol for sp in price_set)
        
        # Danh sách kết quả cho mỗi dòng
        rsi_peak_in_stoch = []
        rsi_trough_in_stoch = []
        
        for idx, row in df.iterrows():
            # Kiểm tra giá tại đỉnh RSI nếu không NaN
            if not np.isnan(row['price_at_rsi_peak']):
                price_peak_rsi = row['price_at_rsi_peak']
                in_peak = is_in_set(price_peak_rsi, stoch_peak_prices, tolerance)
                rsi_peak_in_stoch.append(in_peak)
            else:
                rsi_peak_in_stoch.append(False)
            
            # Kiểm tra giá tại đáy RSI nếu không NaN
            if not np.isnan(row['price_at_rsi_trough']):
                price_trough_rsi = row['price_at_rsi_trough']
                in_trough = is_in_set(price_trough_rsi, stoch_trough_prices, tolerance)
                rsi_trough_in_stoch.append(in_trough)
            else:
                rsi_trough_in_stoch.append(False)
        
        df['rsi_peak_in_stoch'] = rsi_peak_in_stoch
        df['rsi_trough_in_stoch'] = rsi_trough_in_stoch
        
        return df

    def check_bullish_divergence_overlapping(self, df: pd.DataFrame, price_col: str = 'close', rsi_col: str = 'rsi') -> pd.DataFrame:
        """
        Kiểm tra phân kỳ tăng (bullish divergence) tại các điểm đáy giao nhau.
        
        Điều kiện:
        - Chỉ xét các điểm đáy giao nhau có được đánh dấu qua cột 'rsi_trough_in_stoch'.
        - Giả sử ta có các điểm đáy giao nhau được xác định bởi chỉ số từ DataFrame.
        - Phân kỳ tăng được xác định khi:
                Giá hiện tại (ở điểm đáy hiện tại) < Giá tại điểm đáy trước (lower low)
            nhưng
                RSI hiện tại > RSI tại điểm đáy trước (higher low).
                
        Nếu điều kiện thỏa, hàm gán True cho điểm hiện tại ở cột 'bullish_divergence'.
        
        Returns:
        DataFrame có thêm cột 'bullish_divergence'.
        """
        df['bullish_divergence'] = False
        
        # Lấy các chỉ số (index) của các điểm đáy giao nhau (trong đó cột rsi_trough_in_stoch là True)
        trough_indices = df.index[df['rsi_trough_in_stoch'] == True].tolist()
        
        # Kiểm tra phân kỳ tăng ở các điểm đáy sau điểm đầu tiên trong danh sách
        for i in range(1, len(trough_indices)):
            idx_prev = trough_indices[i - 1]
            idx_curr = trough_indices[i]
            # Xét giá đóng cửa: cần có lower low (giá hiện tại nhỏ hơn giá trước)
            if df.loc[idx_curr, price_col] < df.loc[idx_prev, price_col]:
                # Và RSI: cần có higher low (RSI hiện tại lớn hơn RSI trước)
                if df.loc[idx_curr, rsi_col] > df.loc[idx_prev, rsi_col]:
                    df.loc[idx_curr, 'bullish_divergence'] = True
                    
        return df

    def check_bearish_divergence_overlapping(self, df: pd.DataFrame, price_col: str = 'close', rsi_col: str = 'rsi') -> pd.DataFrame:
        """
        Kiểm tra phân kỳ giảm (bearish divergence) tại các điểm đỉnh giao nhau.
        
        Điều kiện:
        - Chỉ xét các điểm đỉnh giao nhau có được đánh dấu qua cột 'rsi_peak_in_stoch'.
        - Phân kỳ giảm được xác định khi:
                Giá hiện tại (ở điểm đỉnh hiện tại) > Giá tại điểm đỉnh trước (higher high)
            nhưng
                RSI hiện tại < RSI tại điểm đỉnh trước (lower high).
                
        Nếu điều kiện thỏa, hàm gán True cho điểm hiện tại ở cột 'bearish_divergence'.
        
        Returns:
        DataFrame có thêm cột 'bearish_divergence'.
        """
        df['bearish_divergence'] = False

        # Lấy các chỉ số (index) của các điểm đỉnh giao nhau (trong đó cột rsi_peak_in_stoch là True)
        peak_indices = df.index[df['rsi_peak_in_stoch'] == True].tolist()
        
        # Kiểm tra phân kỳ giảm ở các điểm đỉnh sau điểm đầu tiên trong danh sách
        for i in range(1, len(peak_indices)):
            idx_prev = peak_indices[i - 1]
            idx_curr = peak_indices[i]
            # Xét giá đóng cửa: cần có higher high (giá hiện tại cao hơn giá trước)
            if df.loc[idx_curr, price_col] > df.loc[idx_prev, price_col]:
                # Và RSI: cần có lower high (RSI hiện tại thấp hơn RSI trước)
                if df.loc[idx_curr, rsi_col] < df.loc[idx_prev, rsi_col]:
                    df.loc[idx_curr, 'bearish_divergence'] = True
                    
        return df

    # ===============================
    # Các hàm phát hiện mô hình nến (Candlestick Patterns)
    # ===============================
    
    def _detect_bullish_engulfing(self, df: DataFrame) -> pd.Series:
        """Phát hiện mô hình nến bullish engulfing."""
        cond_n1_red = df['close'].shift(1) < df['open'].shift(1)
        cond_n_green = df['close'] > df['open']
        cond_close_up = df['close'] > df['close'].shift(1)
        return (cond_n1_red & cond_n_green & cond_close_up).astype(int)

    def _detect_bearish_engulfing(self, df: DataFrame) -> pd.Series:
        """Phát hiện mô hình nến bearish engulfing."""
        cond_n1_green = df['close'].shift(1) > df['open'].shift(1)
        cond_n_red = df['close'] < df['open']
        cond_close_down = df['close'] < df['close'].shift(1)
        return (cond_n1_green & cond_n_red & cond_close_down).astype(int)

    # ===============================
    # Các hàm phát hiện cắt nhau của Stoch
    # ===============================
    
    def _detect_stoch_cross_up(self, df: DataFrame) -> pd.Series:
        """Phát hiện tín hiệu cắt lên: stoch_k cắt lên stoch_d."""
        cond_now = df['stoch_k'] > df['stoch_d']
        cond_prev = df['stoch_k'].shift(1) <= df['stoch_d'].shift(1)
        return (cond_now & cond_prev).astype(int)

    def _detect_stoch_cross_down(self, df: DataFrame) -> pd.Series:
        """Phát hiện tín hiệu cắt xuống: stoch_k cắt xuống stoch_d."""
        cond_now = df['stoch_k'] < df['stoch_d']
        cond_prev = df['stoch_k'].shift(1) >= df['stoch_d'].shift(1)
        return (cond_now & cond_prev).astype(int)
    
    
    # ===============================
    # Hàm populate_indicators: Tính toán các chỉ báo kỹ thuật
    # ===============================
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict,
                            upper_threshold_stoch: int = 80, lower_threshold_stoch: int = 20,
                            rsi_period: int = 14, upper_threshold_rsi: int = 70, lower_threshold_rsi: int = 30,
                            stoch_fastk_period: int = 14, stoch_slowk_period: int = 3, stoch_slowd_period: int = 3,
                            price_col: str = 'close', open_col: str = 'open', high_col: str = 'high', 
                            low_col: str = 'low', rsi_buffer: int = 5) -> pd.DataFrame:
        df = dataframe.copy()

        # 1. Tính RSI và MACD
        df['rsi'] = ta.RSI(df, timeperiod=rsi_period, price=price_col)
        df['macd'], df['macd_signal'], _ = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['rsi_70'] = 70
        df['rsi_30'] = 30

        # 2. Tính Stoch
        df = self.calculate_stoch(df, stoch_fastk_period, stoch_slowk_period, stoch_slowd_period)

        df = self.annotate_price_regions(df, price_col='close', stoch_col='stoch_k', upper_threshold=80, lower_threshold=20)
    
        # 3. Xác định đỉnh/đáy RSI kèm giá đóng cửa tại những điểm đó
        # (Giả sử cột 'rsi' đã tồn tại trong DataFrame)
        df = self.find_rsi_peaks_troughs_scipy(df, rsi_col='rsi', price_col='close', distance=5)
        
        # 4. Kiểm tra xem giá đóng cửa của đỉnh/đáy RSI có thuộc tập hợp giá đóng cửa của
        #    đỉnh/đáy Stoch hay không (với tolerance = 0 nếu so sánh chính xác)
        df = self.check_rsi_in_stoch_regions(df, tolerance=0.0)

        # Kiểm tra phân kỳ tăng tại các điểm đáy giao nhau
        df = self.check_bullish_divergence_overlapping(df, price_col='close', rsi_col='rsi')

        # Kiểm tra phân kỳ giảm tại các điểm đỉnh giao nhau
        df = self.check_bearish_divergence_overlapping(df, price_col='close', rsi_col='rsi')

        # 5. Phát hiện mô hình engulfing
        df['engulfing_bull'] = self._detect_bullish_engulfing(df)
        df['engulfing_bear'] = self._detect_bearish_engulfing(df)
        
        return df

    # ===============================
    # Hàm populate_entry_trend: Điều kiện vào lệnh
    # ===============================
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> pd.DataFrame:
        df = dataframe.copy()
        df['enter_long'] = 0
        df['enter_short'] = 0
        df['cond_bull'] = 0
        df['cond_bear'] = 0

        # ----- Điều kiện Long -----
        df['cond_bull'] += (df['engulfing_bull'] == 1)
        df['cond_bull'] += (df['rsi'] < 30)
        df['cond_bull'] += (df['bullish_divergence'] == True)
        df['cond_bull'] += (self._detect_stoch_cross_up(df) == 1)

        
        df.loc[df['cond_bull'] >= 3, 'enter_long'] = 1

        # ----- Điều kiện Short -----
        df['cond_bear'] += (df['engulfing_bear'] == 1)
        df['cond_bear'] += (df['rsi'] > 70)
        df['cond_bear'] += (df['bearish_divergence'] == True)
        df['cond_bear'] += (self._detect_stoch_cross_down(df) == 1)

        
        df.loc[df['cond_bear'] >= 3, 'enter_short'] = 1
        df.to_csv('data5.csv')
        return df

    # ===============================
    # Hàm populate_exit_trend: Điều kiện thoát lệnh
    # ===============================
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe
