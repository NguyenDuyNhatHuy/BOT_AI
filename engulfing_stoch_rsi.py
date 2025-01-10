# =============================================================================
# File: divergence_engulfing_strategy.py
# Mục đích: Ví dụ Strategy Freqtrade (điều chỉnh logic vào lệnh)
#           - Engulfing dựa trên giá đóng cửa
#           - RSI (quá mua/quá bán)
#           - Phân kỳ RSI (thường + ẩn) bằng pivot cục bộ
#           - Stoch K/D cắt nhau
#           - Vào lệnh nếu >= 3/4 tiêu chí
# =============================================================================

import numpy as np
import pandas as pd
import talib.abstract as ta

from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame


class engulfing_stoch_rsi(IStrategy):
    """
    Strategy Freqtrade:
      - Dựa trên 4 tiêu chí:
          1) Engulfing (đảo chiều) dựa vào giá đóng cửa
          2) RSI (quá mua/quá bán)
          3) Phân kỳ RSI (thường + ẩn) qua pivot cục bộ
          4) Stoch_K cắt Stoch_D (lên / xuống)
      - Vào lệnh Long nếu >= 3/4 điều kiện Bullish
      - Vào lệnh Short nếu >= 3/4 điều kiện Bearish
    """

    timeframe = '30m'
    stoploss = -0.1          # cắt lỗ 10% (ví dụ)
    minimal_roi = {"0": 0.12}  # chốt lời 12% (ví dụ)

    max_open_trades = 4
    allow_entry_with_open_trades = True
    
    # Cho phép mở vị thế short (dành cho futures/margin)
    can_short: bool = True
    can_long: bool = True

    # ----------------------------------------------------------------------
    #  (1) Hàm phát hiện Engulfing (Bullish / Bearish)
    # ----------------------------------------------------------------------
    def _detect_bullish_engulfing(self, dataframe: DataFrame) -> pd.Series:
        """
        Engulfing Tăng (Bullish) dựa trên giá close:
          - Nến (n-1) đỏ: close(n-1) < open(n-1)
          - Nến (n)   xanh: close(n) > open(n)
          - close(n)  > close(n-1)
        """
        cond_n1_red   = (dataframe['close'].shift(1) < dataframe['open'].shift(1))
        cond_n_green  = (dataframe['close'] > dataframe['open'])
        cond_close_up = (dataframe['close'] > dataframe['close'].shift(1))

        return (cond_n1_red & cond_n_green & cond_close_up).astype(int)

    def _detect_bearish_engulfing(self, dataframe: DataFrame) -> pd.Series:
        """
        Engulfing Giảm (Bearish) dựa trên giá close:
          - Nến (n-1) xanh: close(n-1) > open(n-1)
          - Nến (n)   đỏ:   close(n) < open(n)
          - close(n)  < close(n-1)
        """
        cond_n1_green   = (dataframe['close'].shift(1) > dataframe['open'].shift(1))
        cond_n_red      = (dataframe['close'] < dataframe['open'])
        cond_close_down = (dataframe['close'] < dataframe['close'].shift(1))

        return (cond_n1_green & cond_n_red & cond_close_down).astype(int)

    # ----------------------------------------------------------------------
    #  (2) Hàm phát hiện phân kỳ RSI (thường + ẩn) dựa trên pivot cục bộ
    # ----------------------------------------------------------------------
    def _detect_rsi_divergence(self, dataframe: DataFrame, rsi_col: str = 'rsi') -> pd.Series:
        """
        Trả về Series 1 (có phân kỳ RSI) hoặc 0 (không).
        Phân kỳ gồm:
          - Bullish Regular  : Giá LL, RSI HL
          - Bearish Regular  : Giá HH, RSI LH
          - Bullish Hidden   : Giá HL, RSI LL
          - Bearish Hidden   : Giá LH, RSI HH
        Dựa trên pivot (đỉnh/đáy) cục bộ của close & RSI.
        """
        length = len(dataframe)
        rsi_div = np.zeros(length, dtype=int)

        closep = dataframe['close'].values
        rsip   = dataframe[rsi_col].values

        last_pivot_low_idx  = None
        last_pivot_high_idx = None

        for i in range(1, length - 1):
            # Kiểm tra pivot low
            is_pivot_low = (closep[i] < closep[i - 1]) and (closep[i] < closep[i + 1])
            # Kiểm tra pivot high
            is_pivot_high = (closep[i] > closep[i - 1]) and (closep[i] > closep[i + 1])

            # -------------- Pivot LOW --------------
            if is_pivot_low:
                if last_pivot_low_idx is not None:
                    old_idx   = last_pivot_low_idx
                    old_price = closep[old_idx]
                    new_price = closep[i]
                    old_rsi   = rsip[old_idx]
                    new_rsi   = rsip[i]

                    # Bullish Regular:   (new_price < old_price) & (new_rsi > old_rsi)
                    bullish_reg = (new_price < old_price) and (new_rsi > old_rsi)
                    # Bullish Hidden:    (new_price > old_price) & (new_rsi < old_rsi)
                    bullish_hid = (new_price > old_price) and (new_rsi < old_rsi)

                    if bullish_reg or bullish_hid:
                        rsi_div[i] = 1

                last_pivot_low_idx = i

            # -------------- Pivot HIGH --------------
            if is_pivot_high:
                if last_pivot_high_idx is not None:
                    old_idx   = last_pivot_high_idx
                    old_price = closep[old_idx]
                    new_price = closep[i]
                    old_rsi   = rsip[old_idx]
                    new_rsi   = rsip[i]

                    # Bearish Regular:   (new_price > old_price) & (new_rsi < old_rsi)
                    bearish_reg = (new_price > old_price) and (new_rsi < old_rsi)
                    # Bearish Hidden:    (new_price < old_price) & (new_rsi > old_rsi)
                    bearish_hid = (new_price < old_price) and (new_rsi > old_rsi)

                    if bearish_reg or bearish_hid:
                        rsi_div[i] = 1

                last_pivot_high_idx = i

        return pd.Series(rsi_div, index=dataframe.index)

    # ----------------------------------------------------------------------
    #  (3) Hàm phát hiện Stoch_K cắt Stoch_D (lên/xuống)
    # ----------------------------------------------------------------------
    def _detect_stoch_cross_up(self, dataframe: DataFrame) -> pd.Series:
        """
        %K cắt lên %D:
         - (K > D) ở nến hiện tại
         - (K <= D) ở nến trước
        """
        cond_now  = dataframe['stoch_k'] > dataframe['stoch_d']
        cond_prev = dataframe['stoch_k'].shift(1) <= dataframe['stoch_d'].shift(1)
        return (cond_now & cond_prev).astype(int)

    def _detect_stoch_cross_down(self, dataframe: DataFrame) -> pd.Series:
        """
        %K cắt xuống %D:
         - (K < D) ở nến hiện tại
         - (K >= D) ở nến trước
        """
        cond_now  = dataframe['stoch_k'] < dataframe['stoch_d']
        cond_prev = dataframe['stoch_k'].shift(1) >= dataframe['stoch_d'].shift(1)
        return (cond_now & cond_prev).astype(int)

    # ----------------------------------------------------------------------
    #  (4) populate_indicators: Tính các chỉ báo
    # ----------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Stochastic (14,3,3)
        stoch_data = ta.STOCH(dataframe,
                              fastk_period=14,
                              slowk_period=3,
                              slowd_period=3)
        dataframe['stoch_k'] = stoch_data['slowk']
        dataframe['stoch_d'] = stoch_data['slowd']

        # Engulfing
        dataframe['engulfing_bull'] = self._detect_bullish_engulfing(dataframe)
        dataframe['engulfing_bear'] = self._detect_bearish_engulfing(dataframe)

        # Phân kỳ RSI (thường + ẩn)
        dataframe['rsi_divergence'] = self._detect_rsi_divergence(dataframe, rsi_col='rsi')

        return dataframe

    # ----------------------------------------------------------------------
    #  (5) populate_entry_trend: Tín hiệu VÀO lệnh (Long/Short)
    # ----------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Gắn cột 'enter_long' và 'enter_short' cho Freqtrade:
          - Long if >= 3/4 điều kiện Bullish
          - Short if >=3/4 điều kiện Bearish
        """
        # Mặc định
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        # ----- Điều kiện Long -----
        cond_engulf_bull  = (dataframe['engulfing_bull'] == 1)
        cond_rsi_oversold = (dataframe['rsi'] < 30)
        cond_rsi_div      = (dataframe['rsi_divergence'] == 1)
        cond_stoch_up     = (self._detect_stoch_cross_up(dataframe) == 1)

        dataframe['long_score'] = (
            cond_engulf_bull.astype(int) +
            cond_rsi_oversold.astype(int) +
            cond_rsi_div.astype(int) +
            cond_stoch_up.astype(int)
        )

        # Nếu >= 3 => enter_long = 1
        dataframe.loc[
            (dataframe['long_score'] >= 3),
            'enter_long'
        ] = 1

        # ----- Điều kiện Short -----
        cond_engulf_bear    = (dataframe['engulfing_bear'] == 1)
        cond_rsi_overbought = (dataframe['rsi'] > 70)
        cond_rsi_div2       = (dataframe['rsi_divergence'] == 1)
        cond_stoch_down     = (self._detect_stoch_cross_down(dataframe) == 1)

        dataframe['short_score'] = (
            cond_engulf_bear.astype(int) +
            cond_rsi_overbought.astype(int) +
            cond_rsi_div2.astype(int) +
            cond_stoch_down.astype(int)
        )

        # Nếu >= 3 => enter_short = 1
        dataframe.loc[
            (dataframe['short_score'] >= 3),
            'enter_short'
        ] = 1
        dataframe.to_csv('data.csv')
        return dataframe

    # ----------------------------------------------------------------------
    #  (6) populate_exit_trend: Tín hiệu THOÁT lệnh (Long/Short)
    # ----------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Nơi bạn có thể đặt điều kiện thoát lệnh (exit_long / exit_short).
        Tạm thời để trống => thoát lệnh bằng stoploss hoặc ROI.
        """
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe
