import datetime

import numpy as np
from freqtrade.persistence.trade_model import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import pandas as pd

class BullishDivergenceStrategy_2(IStrategy):
    # Cài đặt timeframe giao dịch
    timeframe = '1h'

    # Các chỉ số mặc định sẽ được tính toán
    minimal_roi = {
        "0": 0.1,  # Lợi nhuận tối thiểu 10%
        "240": 0.05,  # Lợi nhuận tối thiểu 5% sau 4 giờ
        "480": 0  # Lợi nhuận tối thiểu 0% sau 8 giờ
    }
    stoploss = -0.20  # Stop loss 20%

    # Cấu hình trailing stop loss (tùy chọn)
    trailing_stop = True
    trailing_stop_positive = 0.02  # Kích hoạt trailing stop loss khi lợi nhuận đạt 2%
    trailing_stop_positive_offset = 0.05  # Bắt đầu trailing ở mức 5% lợi nhuận
    trailing_only_offset_is_reached = True  # Chỉ kích hoạt trailing khi đạt mức lợi nhuận offset

    # Không sử dụng short selling (bán khống)
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    def calculate_smma(self, data, period):
        """Calculates the Smoothed Moving Average (SMMA)."""
        smma = []
        for i in range(len(data)):
            if i < period:
                smma.append(float('nan'))
            elif i == period:
                smma.append(sum(data[:period]) / period)
            else:
                smma.append((smma[-1] * (period - 1) + data[i]) / period)
        return pd.Series(smma, index=data.index)

    def is_bullish_engulfing(self, df, index):
        """Kiểm tra mô hình nến Bullish Engulfing tại index."""
        if index == 0:
            return False
        current = df.iloc[index]
        previous = df.iloc[index - 1]
        return (current['close'] > current['open'] and
                previous['close'] < previous['open'] and
                current['close'] > previous['open'] and
                current['open'] < previous['close'])

    def is_bearish_engulfing(self, df, index):
        """Kiểm tra mô hình nến Bearish Engulfing tại index."""
        if index == 0:
            return False
        current = df.iloc[index]
        previous = df.iloc[index - 1]
        return (current['close'] < current['open'] and
                previous['close'] > previous['open'] and
                current['close'] < previous['open'] and
                current['open'] > previous['close'])

    def calculate_divergence(self, indicator, prices, order=5):
        """
        Tính toán phân kỳ giữa chỉ báo và giá.

        Args:
        indicator: Series chứa giá trị chỉ báo.
        prices: Series chứa giá.
        order: Số lượng nến để kiểm tra đỉnh/đáy.

        Returns:
        Series chứa giá trị phân kỳ (1: phân kỳ tăng, -1: phân kỳ giảm, 0: không phân kỳ).
        """
        divergence = pd.Series(0, index=indicator.index)

        for i in range(order, len(indicator)):
            # Tìm đỉnh/đáy cho chỉ báo
            indicator_highs = indicator.iloc[i - order:i].nlargest(1)
            indicator_lows = indicator.iloc[i - order:i].nsmallest(1)

            # Tìm đỉnh/đáy cho giá
            price_highs = prices.iloc[i - order:i].nlargest(1)
            price_lows = prices.iloc[i - order:i].nsmallest(1)

            # Kiểm tra phân kỳ tăng
            if not indicator_highs.empty and not price_highs.empty:
                if indicator.iloc[i] > indicator_highs.iloc[0] and prices.iloc[i] < price_highs.iloc[0]:
                    divergence.iloc[i] = 1  # Phân kỳ tăng
            if not indicator_lows.empty and not price_lows.empty:
                if indicator.iloc[i] < indicator_lows.iloc[0] and prices.iloc[i] > price_lows.iloc[0]:
                    divergence.iloc[i] = 1  # Phân kỳ tăng

            # Kiểm tra phân kỳ giảm
            if not indicator_highs.empty and not price_highs.empty:
                if indicator.iloc[i] < indicator_highs.iloc[0] and prices.iloc[i] > price_highs.iloc[0]:
                    divergence.iloc[i] = -1  # Phân kỳ giảm
            if not indicator_lows.empty and not price_lows.empty:
                if indicator.iloc[i] > indicator_lows.iloc[0] and prices.iloc[i] < price_lows.iloc[0]:
                    divergence.iloc[i] = -1

        return divergence

    def find_top_bottom(self, data, order=5):
        """
        Tìm đỉnh và đáy của chuỗi dữ liệu.

        Args:
        data: Series chứa dữ liệu.
        order: Số lượng nến để kiểm tra đỉnh/đáy.

        Returns:
        Series chứa giá trị 1 tại các đỉnh, -1 tại các đáy, 0 ở các vị trí khác.
        """
        tops_bottoms = pd.Series(0, index=data.index)

        for i in range(order, len(data) - order):
            # Tìm đỉnh
            is_top = True
            for j in range(1, order + 1):
                if data.iloc[i] <= data.iloc[i - j] or data.iloc[i] <= data.iloc[i + j]:
                    is_top = False
                    break
            if is_top:
                tops_bottoms.iloc[i] = 1

            # Tìm đáy
            is_bottom = True
            for j in range(1, order + 1):
                if data.iloc[i] >= data.iloc[i - j] or data.iloc[i] >= data.iloc[i + j]:
                    is_bottom = False
                    break
            if is_bottom:
                tops_bottoms.iloc[i] = -1

        return tops_bottoms

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Tính toán các chỉ báo kỹ thuật dựa trên file "Logic của chiến lược.txt".

        Args:
        dataframe: Pandas DataFrame chứa dữ liệu giá (open, high, low, close).
        metadata: dict chứa thông tin về cặp giao dịch.

        Returns:
        Pandas DataFrame với các cột chỉ báo được thêm vào.
        """

        # Stochastic Oscillator
        stoch_k, stoch_d = ta.STOCH(dataframe['high'], dataframe['low'], dataframe['close'], fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe['stoch_k'] = stoch_k
        dataframe['stoch_d'] = stoch_d
        dataframe['stoch_top_bottom'] = self.find_top_bottom(dataframe['stoch_k'])

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        dataframe['rsi_divergence'] = self.calculate_divergence(dataframe['rsi'], dataframe['close'])

        # MACD
        macd, macdsignal, macdhist = ta.MACD(dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd
        dataframe['macd_signal'] = macdsignal
        dataframe['macd_hist'] = macdhist
        dataframe['macd_divergence'] = self.calculate_divergence(dataframe['macd'], dataframe['close'])
        # SMMA
        dataframe['smma_5'] = self.calculate_smma(dataframe['close'], 5)
        dataframe['smma_13'] = self.calculate_smma(dataframe['close'], 13)

        # Engulfing
        dataframe['bullish_engulfing'] = [self.is_bullish_engulfing(dataframe, i) for i in range(len(dataframe))]
        dataframe['bearish_engulfing'] = [self.is_bearish_engulfing(dataframe, i) for i in range(len(dataframe))]

        # Điểm cho tín hiệu mua
        dataframe['buy_points'] = 0
        dataframe.loc[dataframe['bullish_engulfing'], 'buy_points'] += 2
        dataframe.loc[dataframe['rsi'] < 30, 'buy_points'] += 1
        dataframe.loc[dataframe['rsi_divergence'] == 1, 'buy_points'] += 1
        dataframe.loc[dataframe['macd'] > dataframe['macd_signal'], 'buy_points'] += 1
        dataframe.loc[dataframe['macd_divergence'] == 1, 'buy_points'] += 1

        # Điểm cho tín hiệu bán
        dataframe['sell_points'] = 0
        dataframe.loc[dataframe['bearish_engulfing'], 'sell_points'] += 2
        dataframe.loc[dataframe['rsi'] > 70, 'sell_points'] += 1
        dataframe.loc[dataframe['rsi_divergence'] == -1, 'sell_points'] += 1
        dataframe.loc[dataframe['macd'] < dataframe['macd_signal'], 'sell_points'] += 1
        dataframe.loc[dataframe['macd_divergence'] == -1, 'sell_points'] += 1

        return dataframe

    def determine_trend_from_stoch_top_bottom(self, df, order=5, smoothing=3):
        """
        Determines the trend based on Stochastic tops and bottoms.

        Args:
        df: Pandas DataFrame containing the data, including the 'stoch_top_bottom' column.
        order: Number of candles to check for top/bottom.
        smoothing: Smoothing period to determine the trend.

        Returns:
        Pandas DataFrame with the 'trend' column added (-1: downtrend, 1: uptrend, 0: unclear).
        """
        df['trend'] = 0

        # Get the indices of tops and bottoms
        top_indices = df.index[df['stoch_top_bottom'] == 1]
        bottom_indices = df.index[df['stoch_top_bottom'] == -1]

        # Smooth the top and bottom indices
        smoothed_tops = pd.Series(index=top_indices).rolling(window=smoothing, min_periods=1).mean()
        smoothed_bottoms = pd.Series(index=bottom_indices).rolling(window=smoothing, min_periods=1).mean()

        for i in range(order, len(df)):
            # Check for uptrend
            if i in bottom_indices:
                last_bottoms = smoothed_bottoms.loc[:i].dropna().tail(2)
                if len(last_bottoms) == 2 and last_bottoms.iloc[1] > last_bottoms.iloc[0]:
                    df.loc[i, 'trend'] = 1  # Uptrend

            # Check for downtrend
            if i in top_indices:
                last_tops = smoothed_tops.loc[:i].dropna().tail(2)
                if len(last_tops) == 2 and last_tops.iloc[1] < last_tops.iloc[0]:
                    df.loc[i, 'trend'] = -1  # Downtrend

        return df

    def determine_trading_range(self, df, order=5):
        """
        Determines the trading range based on Stochastic tops and bottoms.

        Args:
        df: Pandas DataFrame containing the data, including the 'stoch_top_bottom' column.
        order: Number of candles to check for top/bottom.

        Returns:
        Pandas DataFrame with 'range_high' and 'range_low' columns added.
        """

        df['range_high'] = float('nan')
        df['range_low'] = float('nan')

        top_indices = df.index[df['stoch_top_bottom'] == 1].tolist()
        bottom_indices = df.index[df['stoch_top_bottom'] == -1].tolist()

        # Combine and sort all indices
        all_indices = sorted(top_indices + bottom_indices)

        for i in range(1, len(all_indices)):
            current_index = all_indices[i]
            previous_index = all_indices[i - 1]

            # Top after bottom
            if current_index in top_indices and previous_index in bottom_indices:
                df.loc[previous_index:current_index, 'range_high'] = df.loc[current_index, 'high']
                df.loc[previous_index:current_index, 'range_low'] = df.loc[previous_index, 'low']

            # Bottom after top
            elif current_index in bottom_indices and previous_index in top_indices:
                df.loc[previous_index:current_index, 'range_high'] = df.loc[previous_index, 'high']
                df.loc[previous_index:current_index, 'range_low'] = df.loc[current_index, 'low']

        return df
    
    def identify_dmz_zones(self, df, order=5):
        """
        Identifies Demand and Supply Zones (DMZ).

        Args:
            df: Pandas DataFrame containing the data, including 'high' and 'low' columns.
            order: Number of candles to consider for identifying a significant swing.

        Returns:
            Pandas DataFrame with 'dmz_type' column added ('demand', 'supply', or np.nan).
        """

        df['dmz_type'] = np.nan

        # Identify significant swing highs and lows
        df['swing_high'] = df['high'].rolling(window=order * 2 + 1, center=True).apply(lambda x: x.iloc[order] == x.max(), raw=True)
        df['swing_low'] = df['low'].rolling(window=order * 2 + 1, center=True).apply(lambda x: x.iloc[order] == x.min(), raw=True)

        for i in range(order, len(df) - order):
            if df['swing_high'].iloc[i] == 1:
                # Check for a supply zone (previous significant low higher than current low)
                is_supply = False
                for j in range(1, order + 1):
                    if df['low'].iloc[i] < df['low'].iloc[i - j]:
                        is_supply = True
                        break
                if is_supply:
                    df.loc[i, 'dmz_type'] = 'supply'

            elif df['swing_low'].iloc[i] == 1:
                # Check for a demand zone (previous significant high lower than current high)
                is_demand = False
                for j in range(1, order + 1):
                    if df['high'].iloc[i] > df['high'].iloc[i - j]:
                        is_demand = True
                        break
                if is_demand:
                    df.loc[i, 'dmz_type'] = 'demand'

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Determine trading range
        dataframe = self.determine_trading_range(dataframe)
        # Determine trend based on Stochastic tops and bottoms
        dataframe = self.determine_trend_from_stoch_top_bottom(dataframe)

        dataframe.loc[
            (
                (dataframe['buy_points'] >= 3) &
                #(dataframe['trend'] == 1) &
                (dataframe['range_high'].notna())
            ),
            'buy'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Determine trading range
        dataframe = self.determine_trading_range(dataframe)
        # Determine trend based on Stochastic tops and bottoms
        dataframe = self.determine_trend_from_stoch_top_bottom(dataframe)
        dataframe.loc[
            (
                (dataframe['sell_points'] >= 3) &
                #(dataframe['trend'] == -1) &
                (dataframe['range_high'].notna())
            ),
            'sell'] = 1
        return dataframe

    # Các hàm điều kiện thoát lệnh mẫu (`exit_long_on_smma_cross`, `exit_short_on_smma_cross`, `exit_long_on_rsi_above_70`, `exit_short_on_rsi_below_30`, `default_exit_after_n_candles`)
    def exit_long_on_smma_cross(self, df, i):
        return df['smma_5'].iloc[i] < df['smma_13'].iloc[i] and df['smma_5'].iloc[i - 1] >= df['smma_13'].iloc[i - 1]

    def exit_short_on_smma_cross(self, df, i):
        return df['smma_5'].iloc[i] > df['smma_13'].iloc[i] and df['smma_5'].iloc[i - 1] <= df['smma_13'].iloc[i - 1]

    def exit_long_on_rsi_above_70(self, df, i):
        return df['rsi'].iloc[i] > 70

    def exit_short_on_rsi_below_30(self, df, i):
        return df['rsi'].iloc[i] < 30
    
    def exit_long_on_dmz_zones(self, df, i, trade: Trade):
        # Lấy thông tin của lệnh trade
        entry_price = trade.open_rate

        # Tìm vùng Supply Zone gần nhất phía trên giá hiện tại
        supply_zones = df[(df['dmz_type'] == 'supply') & (df.index > i)]
        
        if not supply_zones.empty:
            nearest_supply_zone_index = supply_zones.index[0]
            nearest_supply_zone_high = df.loc[nearest_supply_zone_index, 'high']
            nearest_supply_zone_low = df.loc[nearest_supply_zone_index, 'low']

            # Chia Supply Zone thành 3 phần
            supply_zone_part_1 = nearest_supply_zone_low + (nearest_supply_zone_high - nearest_supply_zone_low) / 3
            supply_zone_part_2 = nearest_supply_zone_low + 2 * (nearest_supply_zone_high - nearest_supply_zone_low) / 3

            # Kiểm tra các điều kiện thoát lệnh
            if df['smma_5'].iloc[i] >= supply_zone_part_1 and df['smma_5'].iloc[i-1] < supply_zone_part_1:
                return True  # Thoát 50%
            elif df['smma_5'].iloc[i] < df['smma_13'].iloc[i] and df['smma_5'].iloc[i-1] >= df['smma_13'].iloc[i-1]:
                return True # Thoát lệnh nếu SMMA 5 cắt xuống SMMA 13
            elif df['high'].iloc[i] >= supply_zone_part_2:
                return True  # Thoát 100% nếu giá vượt qua phần thứ 2
            elif df['high'].iloc[i] >= nearest_supply_zone_high:
                return True  # Thoát lệnh nếu giá chạm đỉnh của Supply Zone

        return False

    def exit_short_on_dmz_zones(self, df, i, trade: Trade):
        # Lấy thông tin của lệnh trade
        entry_price = trade.open_rate

        # Tìm vùng Demand Zone gần nhất phía dưới giá hiện tại
        demand_zones = df[(df['dmz_type'] == 'demand') & (df.index > i)]
        
        if not demand_zones.empty:
            nearest_demand_zone_index = demand_zones.index[0]
            nearest_demand_zone_high = df.loc[nearest_demand_zone_index, 'high']
            nearest_demand_zone_low = df.loc[nearest_demand_zone_index, 'low']

            # Chia Demand Zone thành 3 phần
            demand_zone_part_1 = nearest_demand_zone_high - (nearest_demand_zone_high - nearest_demand_zone_low) / 3
            demand_zone_part_2 = nearest_demand_zone_high - 2 * (nearest_demand_zone_high - nearest_demand_zone_low) / 3

            # Kiểm tra các điều kiện thoát lệnh
            if df['smma_5'].iloc[i] <= demand_zone_part_1 and df['smma_5'].iloc[i-1] > demand_zone_part_1:
                return True  # Thoát 50%
            elif df['smma_5'].iloc[i] > df['smma_13'].iloc[i] and df['smma_5'].iloc[i-1] <= df['smma_13'].iloc[i-1]:
                return True # Thoát lệnh nếu SMMA 5 cắt lên SMMA 13
            elif df['low'].iloc[i] <= demand_zone_part_2:
                return True  # Thoát 100% nếu giá xuống dưới phần thứ 2
            elif df['low'].iloc[i] <= nearest_demand_zone_low:
                return True  # Thoát lệnh nếu giá chạm đáy của Demand Zone

        return False

    def default_exit_after_n_candles(self, df, i, trade: Trade, n=24):
        # Kiểm tra xem lệnh đã mở bao lâu (số nến)
        open_time = trade.open_date_utc
        current_time = df['date'].iloc[i]
        time_since_open = (current_time - open_time).total_seconds() / 60 / int(self.timeframe[:-1])  # Số phút / số phút mỗi nến

        # Thoát lệnh nếu đã mở quá n nến
        if time_since_open >= n:
            return True

        return False

    def populate_exit(self, df: DataFrame, metadata: dict) -> DataFrame:
        df = self.identify_dmz_zones(df)

        long_exit_conditions = [
            lambda df, i, trade: self.exit_long_on_smma_cross(df, i),
            lambda df, i, trade: self.exit_long_on_rsi_above_70(df, i),
            lambda df, i, trade: self.exit_long_on_dmz_zones(df, i, trade),
        ]
        short_exit_conditions = [
            lambda df, i, trade: self.exit_short_on_smma_cross(df, i),
            lambda df, i, trade: self.exit_short_on_rsi_below_30(df, i),
            lambda df, i, trade: self.exit_short_on_dmz_zones(df, i, trade),
        ]

        # Sử dụng danh sách các điều kiện thoát lệnh cho mỗi trade
        trades = Trade.get_trades_proxy()
        for trade in trades:
            if trade.is_open:
                if trade.is_short:
                    # Gán điều kiện thoát lệnh cho lệnh Short
                    trade.exit_conditions = short_exit_conditions
                else:
                    # Gán điều kiện thoát lệnh cho lệnh Long
                    trade.exit_conditions = long_exit_conditions

        # Kiểm tra các điều kiện thoát lệnh cho từng trade
        for index, row in df.iterrows():
            for trade in trades:
                if trade.is_open and trade.pair == metadata['pair']:
                    if trade.is_short:
                        for condition in trade.exit_conditions:
                            if condition(df, index, trade):
                                df.loc[index, 'sell'] = 1
                                break
                    else:
                        for condition in trade.exit_conditions:
                            if condition(df, index, trade):
                                df.loc[index, 'sell'] = 1
                                break

        return df