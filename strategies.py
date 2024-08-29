import backtrader as bt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import backtrader as bt
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class KAMAHurstStrategy(bt.Strategy):
    params = (
        ("kama_period", 10),  # KAMA period
        ("hurst_period", 50),  # Hurst Exponent period
        ("hurst_threshold", 0.5),  # Hurst Exponent threshold for persistent trend
    )

    def __init__(self):
        self.kama = bt.indicators.KAMA(self.data.close, period=self.params.kama_period)
        self.hurst = bt.indicators.HurstExponent(self.data.close, period=self.params.hurst_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            # Entry condition with KAMA and Hurst Exponent
            if self.kama > self.kama[-1] and self.hurst > self.params.hurst_threshold:
                self.order = self.buy(size=position_size)

        else:
            # Exit condition with falling KAMA or mean-reverting behavior indicated by Hurst Exponent
            if self.kama < self.kama[-1] or self.hurst < self.params.hurst_threshold:
                self.log('Position Closed, KAMA falling or mean-reverting behavior indicated by Hurst Exponent')
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            elif order.issell():
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class KAMAHurstBollingerADXStrategy(bt.Strategy):
    params = (
        ("kama_period", 10),  # KAMA period
        ("hurst_period", 50),  # Hurst exponent period
        ("hurst_threshold", 0.5),  # Hurst Exponent threshold for persistent trend
        ("aDX_threshold", 25),  # ADX threshold for trend strength
        ("bollinger_band_multiplier", 2.0),  # Bollinger Bands multiplier
    )

    def __init__(self):
        self.kama = bt.indicators.KAMA(self.data.close, period=self.params.kama_period)
        self.hurst = bt.indicators.HurstExponent(self.data.close, period=self.params.hurst_period)
        self.aDX = bt.indicators.ADX(self.data, period=self.params.hurst_period)
        self.bollingerBands = bt.indicators.BollingerBands(self.data.close, period=20, devfactor=self.params.bollinger_band_multiplier)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            # Entry condition with KAMA, Hurst Exponent, and ADX
            if self.kama > self.kama[-1] and self.hurst > self.params.hurst_threshold and self.aDX > self.params.aDX_threshold:
                self.order = self.buy(size=position_size)

        else:
            # Exit condition with falling KAMA, mean-reverting behavior indicated by Hurst Exponent, or ADX below threshold
            if self.kama < self.kama[-1] and self.hurst < self.params.hurst_threshold and self.aDX < self.params.aDX_threshold:
                self.log('Position Closed, KAMA falling or mean-reverting behavior indicated by Hurst Exponent or ADX dropping below threshold')
                self.order = self.close()
    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            elif order.issell():
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None



class EMAADXTrendFollowingStrategy(bt.Strategy):
    params = (
        ("ema_short_period", 50),
        ("ema_long_period", 200),
        ("adx_period", 14),
        ("adx_level", 25),
    )

    def __init__(self):
        self.ema_short = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.ema_short_period)
        self.ema_long = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.ema_long_period)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(period=self.params.adx_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price*.99
            
            # if self.ema_short[0] > self.ema_long[0] and self.ema_short[-1] <= self.ema_long[-1] and self.adx > self.params.adx_level:
            if self.ema_short[0] > self.ema_long[0] and self.adx > self.params.adx_level:
                self.log('Buy Create, EMA(50) Crosses Above EMA(200) and ADX: %.2f' % self.adx[0])
                self.order = self.buy(size=position_size)

        else:
            if self.ema_short[0] < self.ema_long[0] or self.adx <= self.params.adx_level:
                self.log('Position Closed, EMA(50) Crosses Below EMA(200) or ADX: %.2f' % self.adx[0])
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class EnhancedTrendFollowingStrategy(bt.Strategy):
    params = (
        ("ema_short_period", 50),
        ("ema_long_period", 200),
        ("adx_period", 14),
        ("adx_level", 25),
        ("risk_factor", 0.02),  # Risk 2% of capital per trade
    )

    def __init__(self):
        self.ema_short = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.ema_short_period)
        self.ema_long = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.ema_long_period)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(period=self.params.adx_period)
        self.atr = bt.indicators.AverageTrueRange(self.data, period=14)

        self.order = None

    def next(self):
        # Skip if there is an open order
        if self.order:
            return

        # Check for entry conditions
        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = (cash * self.params.risk_factor) / (self.atr[0] * asset_price)
            
            # Check for a strong bullish trend condition
            if (
                self.ema_short[0] > self.ema_long[0] and
                self.ema_short[0] > self.ema_long[0] * 1.02 and  # Confirming trend strength
                self.adx > self.params.adx_level
            ):
                self.log(
                    'Buy Create, EMA(50) Crosses Above EMA(200), ADX: %.2f, Position Size: %.2f' % (self.adx[0], position_size)
                )
                self.order = self.buy(size=position_size)

        # Check for exit conditions
        else:
            # Check for a bearish trend or weakening trend condition
            if self.ema_short[0] < self.ema_long[0] or self.adx <= self.params.adx_level:
                self.log('Position Closed, EMA(50) Crosses Below EMA(200) or ADX: %.2f' % self.adx[0])
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            # Log executed trade details and performance metrics
            if order.isbuy():
                self.log(
                    "Executed BUY (Price: %.2f, Size: %.2f, Portfolio Value: %.2f)" %
                    (order.executed.price, order.executed.size, self.broker.get_value())
                )
            else:
                self.log(
                    "Executed SELL (Price: %.2f, Size: %.2f, Portfolio Value: %.2f)" %
                    (order.executed.price, order.executed.size, self.broker.get_value())
                )
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        
        # Reset order flag
        self.order = None

class MeanReversionStrategy(bt.Strategy):
    params = (
        ("bollinger_period", 20),
        ("bollinger_dev", 2),
    )

    def __init__(self):
        self.bollinger = bt.indicators.BollingerBands(self.data.close, period=self.params.bollinger_period, devfactor=self.params.bollinger_dev)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price*.99
            if self.data.close < self.bollinger.lines.bot:
                self.log('Buy Create, Price touches lower Bollinger Band')
                self.order = self.buy(size=position_size)

        else:
            if self.data.close > self.bollinger.lines.top:
                self.log('Position Closed, Price touches upper Bollinger Band')
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class SequentialStrategy(bt.Strategy):
    params = (
        ("bollinger_period", 20),
        ("bollinger_dev", 2),
        ("momentum_period", 14),
        ("momentum_threshold", 0),
        ("macd_short_period", 12),  # Adjust this parameter
        ("macd_long_period", 26),   # Adjust this parameter
        ("macd_signal_period", 9),  # Adjust this parameter
        ("macd_threshold", 0),      # Adjust this parameter
    )

    def __init__(self):
        self.bollinger = bt.indicators.BollingerBands(self.data.close, period=self.params.bollinger_period, devfactor=self.params.bollinger_dev)
        self.momentum = bt.indicators.Momentum(self.data.close, period=self.params.momentum_period)
        self.macd_trend_filter = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.macd_short_period,
            period_me2=self.params.macd_long_period,
            period_signal=self.params.macd_signal_period
        )
        self.order = None

    def next(self):
        if self.order:
            return

        # Check if MACD indicates a strong trend
        if self.macd_trend_filter.macd - self.macd_trend_filter.signal > self.params.macd_threshold:
            return

        # Mean Reversion Strategy
        if not self.position:
            if self.data.close < self.bollinger.lines.bot:
                self.log('Buy Create - Mean Reversion, Price below lower Bollinger Band')
                self.order = self.buy()

        # Momentum Strategy
        else:
            if self.momentum > self.params.momentum_threshold:
                self.log('Sell Create - Momentum, Price has strong positive momentum')
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            elif order.issell():
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None




class MACDStrategy(bt.Strategy):
    def __init__(self):
        self.macd = bt.indicators.MACDHisto(period_me1=12, period_me2=26, period_signal=9)
        self.ema = bt.indicators.ExponentialMovingAverage(self.data, period=30)
        self.order = None
        
    def next(self):
        if self.order:
            return
        
        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price*.99

            if self.macd[0] > 0 and self.macd[-1] <= 0:
                self.log('Buy Create, MACD: %.2f' % self.macd[0])
                self.order = self.buy(size=position_size)
                
        else:
            if self.macd[0] < 0 and self.macd[-1] >= 0:
                self.log('Position Closed, MACD: %.2f' % self.macd[0])
                self.order = self.close()
              
    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
    
    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f, Value: %.2f, Commission %.2f)" %
                          (order.executed.price, order.executed.value, order.executed.comm))
            else:
                self.log("Executed SELL (Price: %.2f, Value: %.2f, Commission %.2f)" %
                          (order.executed.price, order.executed.value, order.executed.comm))
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class RSIOverboughtOversoldStrategy(bt.Strategy):
    params = (
        ("rsi_period", 14),
        ("rsi_oversold", 30),
        ("rsi_overbought", 70),
    )

    def __init__(self):
        self.rsi = bt.indicators.RelativeStrengthIndex(period=self.params.rsi_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            if self.rsi < self.params.rsi_oversold:
                self.log('Buy Create, RSI crosses below 30 (oversold)')
                self.order = self.buy(size=position_size)

        else:
            if self.rsi > self.params.rsi_overbought:
                self.log('Position Closed, RSI crosses above 70 (overbought)')
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class DualMovingAverageCrossoverStrategy(bt.Strategy):
    params = (
        ("short_ema_period", 10),
        ("long_ema_period", 50),
    )

    def __init__(self):
        self.short_ema = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.short_ema_period)
        self.long_ema = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.long_ema_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            if self.short_ema > self.long_ema and self.short_ema[-1] <= self.long_ema[-1]:
                self.log('Buy Create, Short EMA(10) crosses above Long EMA(50)')
                self.order = self.buy(size=position_size)

        else:
            if self.short_ema < self.long_ema and self.short_ema[-1] >= self.long_ema[-1]:
                self.log('Position Closed, Short EMA(10) crosses below Long EMA(50)')
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class StochasticRSIConfluenceStrategy(bt.Strategy):
    params = (
        ("stochastic_period", 14),
        ("stochastic_dfast", 3),
        ("stochastic_dslow", 3),
        ("rsi_period", 14),
        ("rsi_oversold", 30),
    )

    def __init__(self):
        self.stochastic = bt.indicators.Stochastic(self.data, period=self.params.stochastic_period,
                                                    period_dfast=self.params.stochastic_dfast,
                                                    period_dslow=self.params.stochastic_dslow)
        self.rsi = bt.indicators.RelativeStrengthIndex(period=self.params.rsi_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            if self.stochastic.lines.percK < 20 and self.rsi < self.params.rsi_oversold:
                self.log('Buy Create, Stochastic and RSI signal oversold conditions')
                self.order = self.buy(size=position_size)

        else:
            # Add sell conditions if needed
            pass

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class ATRBreakoutStrategy(bt.Strategy):
    params = (
        ("atr_period", 14),
        ("atr_multiplier", 1.0),
    )

    def __init__(self):
        self.atr = bt.indicators.AverageTrueRange(period=self.params.atr_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            if self.data.high[0] > self.data.high[-1] + self.atr[0] * self.params.atr_multiplier:
                self.log('Buy Create, ATR Breakout condition')
                self.order = self.buy(size=position_size)

        else:
            if self.data.low[0] < self.data.high[-1] - self.atr[0] * self.params.atr_multiplier:
                self.log('Position Closed, ATR Breakout condition')
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class IchimokuCloudBreakoutStrategy(bt.Strategy):
    def __init__(self):
        self.ichimoku = bt.indicators.Ichimoku(self.data)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            if self.data.close[0] > self.ichimoku.lines.senkou_span_a[0] and \
                    self.data.close[0] > self.ichimoku.lines.senkou_span_b[0] and \
                    self.data.close[-1] <= self.ichimoku.lines.senkou_span_a[-1] and \
                    self.data.close[-1] <= self.ichimoku.lines.senkou_span_b[-1]:
                self.log('Buy Create, Price crosses above Ichimoku Cloud')
                self.order = self.buy(size=position_size)

        else:
            if self.data.close[0] < self.ichimoku.lines.senkou_span_a[0] and \
                    self.data.close[0] < self.ichimoku.lines.senkou_span_b[0] and \
                    self.data.close[-1] >= self.ichimoku.lines.senkou_span_a[-1] and \
                    self.data.close[-1] >= self.ichimoku.lines.senkou_span_b[-1]:
                self.log('Position Closed, Price crosses below Ichimoku Cloud')
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class PSARReversalStrategy(bt.Strategy):
    params = (
        ("psar_af", 0.02),
        ("psar_afmax", 0.2),
    )

    def __init__(self):
        self.psar = bt.indicators.ParabolicSAR(af=self.params.psar_af, afmax=self.params.psar_afmax)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            if self.data.close[0] < self.psar[0] and self.data.close[-1] > self.psar[-1]:
                self.log('Buy Create, Price is below PSAR and flips to above')
                self.order = self.buy(size=position_size)

        else:
            if self.data.close[0] > self.psar[0] and self.data.close[-1] < self.psar[-1]:
                self.log('Position Closed, Price is above PSAR and flips to below')
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None



class MFI(bt.Indicator):
    lines = ('mfi',)
    params = dict(period=14)

    alias = ('MoneyFlowIndicator',)

    def __init__(self):
        tprice = (self.data.close + self.data.low + self.data.high) / 3.0
        mfraw = tprice * self.data.volume

        flowpos = bt.ind.SumN(mfraw * (tprice > tprice(-1)), period=self.p.period)
        flowneg = bt.ind.SumN(mfraw * (tprice < tprice(-1)), period=self.p.period)

        mfiratio = bt.ind.DivByZero(flowpos, flowneg, zero=100.0)
        self.l.mfi = 100.0 - 100.0 / (1.0 + mfiratio)

class MFIDivergenceStrategy(bt.Strategy):
    params = (
        ("mfi_period", 14),
    )

    def __init__(self):
        self.mfi = MFI(self.data, period=self.params.mfi_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            # Look for divergence - MFI making new highs and price isn't
            if self.mfi > self.mfi[-1] and self.data.close < self.data.close[-1]:
                self.log('Buy Create, MFI is making new highs and price isn\'t')
                self.order = self.buy(size=position_size)

        else:
            # Add sell conditions if needed
            pass

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class TEMACrossoverStrategy(bt.Strategy):
    params = (
        ("tema_short_period", 10),
        ("tema_long_period", 50),
    )

    def __init__(self):
        self.tema_short = bt.indicators.TripleExponentialMovingAverage(self.data.close, period=self.params.tema_short_period)
        self.tema_long = bt.indicators.TripleExponentialMovingAverage(self.data.close, period=self.params.tema_long_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            # Buy on TEMA(10) crossing above TEMA(50)
            if self.tema_short[0] > self.tema_long[0] and self.tema_short[-1] <= self.tema_long[-1]:
                self.log('Buy Create, TEMA(10) crossing above TEMA(50)')
                self.order = self.buy(size=position_size)

        else:
            # Sell on the opposite crossover
            if self.tema_short[0] < self.tema_long[0] and self.tema_short[-1] >= self.tema_long[-1]:
                self.log('Position Closed, TEMA(10) crossing below TEMA(50)')
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class HMATrendContinuationStrategy(bt.Strategy):
    params = (
        ("hma_period", 20),
    )

    def __init__(self):
        self.hma = bt.indicators.HullMovingAverage(self.data.close, period=self.params.hma_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            # Buy when the price is above HMA and the HMA is rising
            if self.data.close[0] > self.hma[0] and self.hma[0] > self.hma[-1]:
                self.log('Buy Create, Price above HMA and HMA rising')
                self.order = self.buy(size=position_size)

        else:
            # Sell when the opposite occurs
            if self.data.close[0] < self.hma[0] and self.hma[0] < self.hma[-1]:
                self.log('Position Closed, Price below HMA and HMA falling')
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class KAMAAdaptiveStrategy(bt.Strategy):
    params = (
        ("kama_period", 20),
    )

    def __init__(self):
        self.kama = bt.indicators.KAMA(self.data.close, period=self.params.kama_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            # Buy when the price is above KAMA and KAMA is rising
            if self.data.close[0] > self.kama[0] and self.kama[0] > self.kama[-1]:
                self.log('Buy Create, Price above KAMA and KAMA rising')
                self.order = self.buy(size=position_size)

        else:
            # Sell when the opposite occurs
            if self.data.close[0] < self.kama[0] and self.kama[0] < self.kama[-1]:
                self.log('Position Closed, Price below KAMA and KAMA falling')
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

from backtrader.indicators import SumN, TrueLow, TrueRange

class UltimateOscillator(bt.Indicator):
    
    lines = ('uo',)

    params = (('p1', 7),
              ('p2', 14),
              ('p3', 28),
              ('upperband', 70.0),
              ('lowerband', 30.0),
    )

    def _plotinit(self):
        baseticks = [10.0, 50.0, 90.0]
        hlines = [self.params.upperband, self.params.lowerband]

        self.plotinfo.plotyhlines = hlines
        self.plotinfo.plotyticks = baseticks + hlines

    def __init__(self):
        bp = self.data.close - TrueLow(self.data)
        tr = TrueRange(self.data)

        av7 = SumN(bp, period=self.params.p1) / SumN(tr, period=self.params.p1)
        av14 = SumN(bp, period=self.params.p2) / SumN(tr, period=self.params.p2)
        av28 = SumN(bp, period=self.params.p3) / SumN(tr, period=self.params.p3)

        uo = 100.0 * (4.0 * av7 + 2.0 * av14 + av28) / (4.0 + 2.0 + 1.0)
        self.lines.uo = uo
class UltimateOscillatorDivergenceStrategy(bt.Strategy):
    params = (('p1', 7),
                  ('p2', 14),
                  ('p3', 28),
                  ('upperband', 70.0),
                  ('lowerband', 30.0),
        )
    def __init__(self):
        self.uo = UltimateOscillator(self.data, 
                                     p1=self.params.p1,
                                     p2=self.params.p2, 
                                     p3=self.params.p3)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            # Look for bullish divergence - UO making new highs and price isn't
            if self.uo > self.uo[-1] and self.data.close < self.data.close[-1]:
                self.log('Buy Create, Bullish divergence on Ultimate Oscillator')
                self.order = self.buy(size=position_size)

        else:
            # Add sell conditions if needed
            pass

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class WilliamsRReversalStrategy(bt.Strategy):
    params = (
        ("williamsr_period", 14),
        ("oversold_threshold", -80),
        ("overbought_threshold", -20),
    )

    def __init__(self):
        self.williamsr = bt.indicators.WilliamsR(period=self.params.williamsr_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            # Buy when Williams %R crosses below oversold threshold
            if self.williamsr < self.params.oversold_threshold and self.williamsr[-1] >= self.params.oversold_threshold:
                self.log('Buy Create, Williams %R crossed below -80')
                self.order = self.buy(size=position_size)

        else:
            # Sell when Williams %R crosses above overbought threshold
            if self.williamsr > self.params.overbought_threshold and self.williamsr[-1] <= self.params.overbought_threshold:
                self.log('Position Closed, Williams %R crossed above -20')
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class PivotPointBreakoutStrategy(bt.Strategy):
  
    def __init__(self):
        self.pivot_point = bt.indicators.PivotPoint(self.data)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            # Buy when the price crosses above the pivot point
            if self.data.close[0] > self.pivot_point.lines.p[0] and self.data.close[-1] <= self.pivot_point.lines.p[-1]:
                self.log('Buy Create, Price crossed above Pivot Point')
                self.order = self.buy(size=position_size)

        else:
            # Sell when the price crosses below the pivot point
            if self.data.close[0] < self.pivot_point.lines.p[0] and self.data.close[-1] >= self.pivot_point.lines.p[-1]:
                self.log('Position Closed, Price crossed below Pivot Point')
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class DemaTrendFollowingStrategy(bt.Strategy):
    params = (
        ("dema_period", 20),
    )

    def __init__(self):
        self.dema = bt.indicators.DoubleExponentialMovingAverage(self.data.close, period=self.params.dema_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            # Buy when the price is above DEMA and DEMA is rising
            if self.data.close[0] > self.dema[0] and self.dema[0] > self.dema[-1]:
                self.log('Buy Create, Price is above DEMA and DEMA is rising')
                self.order = self.buy(size=position_size)

        else:
            # Sell when the price is below DEMA or DEMA is falling
            if self.data.close[0] < self.dema[0] or self.dema[0] < self.dema[-1]:
                self.log('Position Closed, Price is below DEMA or DEMA is falling')
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class TsiMomentumStrategy(bt.Strategy):
    params = (
        ("tsi_period", 25),
        ("tsi_signal_period", 13),
    )

    def __init__(self):
        self.tsi = bt.indicators.TrueStrengthIndicator(self.data.close, period1=self.params.tsi_period, period2=self.params.tsi_signal_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            # Buy when TSI crosses above its signal line
            if self.tsi > 0 and self.tsi[-1] <= 0:
                self.log('Buy Create, TSI crossed above its signal line')
                self.order = self.buy(size=position_size)

        else:
            # Sell on the opposite TSI crossover
            if self.tsi < 0 and self.tsi[-1] >= 0:
                self.log('Position Closed, TSI crossed below its signal line')
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class AdxStrengthConfirmationStrategy(bt.Strategy):
    params = (
        ("adx_period", 14),
        ("adx_strength_threshold", 25),
    )

    def __init__(self):
        self.adx = bt.indicators.AverageDirectionalMovementIndex(period=self.params.adx_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            # Buy when ADX is above the threshold and rising
            if self.adx > self.params.adx_strength_threshold and self.adx > self.adx[-1]:
                self.log('Buy Create, ADX is above 25 and rising')
                self.order = self.buy(size=position_size)

        else:
            # No selling condition in this example
            pass

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class MomentumBreakoutStrategy(bt.Strategy):
    params = (
        ("momentum_period", 14),
    )

    def __init__(self):
        self.momentum = bt.indicators.Momentum(self.data.close, period=self.params.momentum_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            # Buy when 14-period Momentum crosses above zero
            if self.momentum > 0 and self.momentum[-1] <= 0:
                self.log('Buy Create, Momentum crossed above zero')
                self.order = self.buy(size=position_size)

        else:
            # Sell when 14-period Momentum crosses below zero
            if self.momentum < 0 and self.momentum[-1] >= 0:
                self.log('Position Closed, Momentum crossed below zero')
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class AOEmaTrendConfirmationStrategy(bt.Strategy):
    params = (
        
        ("ema_short_period", 9),
        ("ema_long_period", 21),
    )

    def __init__(self):
        self.ao = bt.indicators.AwesomeOscillator()
        self.ema_short = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.ema_short_period)
        self.ema_long = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.ema_long_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            # Buy when AO is above zero, and short-term EMA is above long-term EMA
            if self.ao > 0 and self.ema_short > self.ema_long and self.ao[-1] <= 0:
                self.log('Buy Create, AO is above zero and EMA crossover')
                self.order = self.buy(size=position_size)

        else:
            # Sell when AO is below zero or short-term EMA is below long-term EMA
            if self.ao < 0 or self.ema_short < self.ema_long:
                self.log('Position Closed, AO is below zero or EMA crossover')
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class CCITrendReversalStrategy(bt.Strategy):
    params = (
        ("cci_period", 14),
        ("overbought_level", 100),
        ("oversold_level", -100),
    )

    def __init__(self):
        self.cci = bt.indicators.CommodityChannelIndex(period=self.params.cci_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            # Buy when CCI crosses above 100
            if self.cci > self.params.overbought_level and self.cci[-1] <= self.params.overbought_level:
                self.log('Buy Create, CCI crossed above 100')
                self.order = self.buy(size=position_size)

        else:
            # Sell when CCI crosses below -100
            if self.cci < self.params.oversold_level and self.cci[-1] >= self.params.oversold_level:
                self.log('Position Closed, CCI crossed below -100')
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class RMIStochasticStrategy(bt.Strategy):
    params = (
        ("rmi_period", 20),
        ("rmi_movav", bt.indicators.SmoothedMovingAverage),
        ("rmi_upperband", 70.0),
        ("rmi_lowerband", 30.0),
        ("rmi_safediv", True),
        ("rmi_safehigh", 100.0),
        ("rmi_safelow", 50.0),
        ("rmi_lookback", 5),
    )

    def __init__(self):
        self.rmi = bt.indicators.RelativeMomentumIndex(
            period=self.params.rmi_period,
            movav=self.params.rmi_movav,
            upperband=self.params.rmi_upperband,
            lowerband=self.params.rmi_lowerband,
            safediv=self.params.rmi_safediv,
            safehigh=self.params.rmi_safehigh,
            safelow=self.params.rmi_safelow,
            lookback=self.params.rmi_lookback
        )
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            # Buy when RMI crosses above upperband
            if self.rmi > self.params.rmi_upperband and self.rmi[-1] <= self.params.rmi_upperband and cash > 0:
                self.log('Buy Create, RMI crossed above upperband')
                self.order = self.buy(size=position_size)

        else:
            # Sell when RMI crosses below lowerband
            if self.rmi < self.params.rmi_lowerband and self.rmi[-1] >= self.params.rmi_lowerband:
                self.log('Position Closed, RMI crossed below lowerband')
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class PercentRankPullbackStrategy(bt.Strategy):
    params = (
        ("percentrank_period", 14),
        ("percentrank_oversold", .20),
        ("percentrank_overbought", .80),
    )

    def __init__(self):
        self.percentrank = bt.indicators.PercentRank(self.data.close, period=self.params.percentrank_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            # Buy when PercentRank drops below 20
            if self.percentrank < self.params.percentrank_oversold and self.percentrank[-1] >= self.params.percentrank_oversold:
                self.log('Buy Create, PercentRank dropped below 20')
                self.order = self.buy(size=position_size)

        else:
            # Sell when PercentRank goes above 80
            if self.percentrank > self.params.percentrank_overbought and self.percentrank[-1] <= self.params.percentrank_overbought:
                self.log('Position Closed, PercentRank went above 80')
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            else:
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None


class ATR_EMA_Strategy(bt.Strategy):
    params = (
        ("ema_short_period", 10),
        ("ema_long_period", 50),
        ("atr_period", 14),
        ("atr_multiplier", 1.5),
    )

    def __init__(self):
        self.ema_short = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.ema_short_period)
        self.ema_long = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.ema_long_period)
        self.atr = bt.indicators.AverageTrueRange(self.data, period=self.params.atr_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            atr_value = self.atr[0]

            # Adjust position size based on ATR
            position_size = (cash * 0.01) / atr_value  # Adjust the multiplier (0.01) based on your risk preference

            # Enter trades in the direction of the EMA trend
            if self.ema_short > self.ema_long and self.ema_short[-1] <= self.ema_long[-1]:
                self.log('Buy Create, EMA Trend: Up')
                self.order = self.buy(size=position_size)
        else:
            # Close position when EMA trend reverses
            if self.ema_short < self.ema_long and self.ema_short[-1] >= self.ema_long[-1]:
                self.log('Position Closed, EMA Trend: Down')
                self.order = self.close()

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            elif order.issell():
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class ATRTrailingStopStrategy(bt.Strategy):
    params = (
        ("atr_period", 14),
        ("atr_multiplier", 2.0),
        ("trail_percent", 0.02),
    )

    def __init__(self):
        self.atr = bt.indicators.AverageTrueRange(period=self.params.atr_period)
        self.trail_stop = None
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            # Buy Signal: No position and close above the ATR trailing stop
            if self.trail_stop is None or self.data.close[0] > self.trail_stop:
                self.log('Buy Create, Close: %.2f' % self.data.close[0])
                self.order = self.buy(size=position_size)
                self.trail_stop = self.data.close[0] - self.atr[0] * self.params.atr_multiplier

        else:
            # Sell Signal: Close below the ATR trailing stop
            if self.data.close[0] < self.trail_stop:
                self.log('Position Closed, Close: %.2f' % self.data.close[0])
                self.order = self.close()

        # Update ATR trailing stop value
        self.trail_stop = max(self.trail_stop, self.data.close[0] - self.atr[0] * self.params.atr_multiplier * (1 - self.params.trail_percent))

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            elif order.issell():
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None

class DynamicRSIBollingerStrategy(bt.Strategy):
    params = (
        ("rsi_period", 14),  # Period for RSI
        ("bollinger_period", 20),  # Period for Bollinger Bands
        ("bollinger_dev", 2),  # Standard deviation multiplier for Bollinger Bands
        ("rsi_lower_threshold", 30),  # Lower RSI threshold for buying
        ("rsi_upper_threshold", 70),  # Upper RSI threshold for selling
    )

    def __init__(self):
        self.rsi = bt.indicators.RelativeStrengthIndex(self.data.close, period=self.params.rsi_period)
        self.bollinger = bt.indicators.BollingerBands(self.data.close, period=self.params.bollinger_period, devfactor=self.params.bollinger_dev)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            cash = self.broker.get_cash()
            asset_price = self.data.close[0]
            position_size = cash / asset_price * 0.99

            # Buy Signal: RSI below dynamic threshold based on Bollinger Bands width
            if self.rsi < self.calculate_dynamic_rsi_threshold():
                self.order = self.buy(size=position_size)

        else:
            # Sell Signal: RSI above dynamic threshold based on Bollinger Bands width
            if self.rsi > self.calculate_dynamic_rsi_threshold():
                self.log('Position Closed, RSI above dynamic threshold')
                self.order = self.close()

    def calculate_dynamic_rsi_threshold(self):
        bollinger_width = self.bollinger.lines.bot - self.bollinger.lines.top
        dynamic_threshold = self.params.rsi_upper_threshold - (bollinger_width * 2)
        return dynamic_threshold

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.log("Executed BUY (Price: %.2f)" % order.executed.price)
            elif order.issell():
                self.log("Executed SELL (Price: %.2f)" % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order was canceled/margin/rejected")
        self.order = None
import logging
class StochasticSRStrategy(bt.Strategy):
    '''Trading strategy that utilizes the Stochastic Oscillator indicator for oversold/overbought entry points, 
    and previous support/resistance via Donchian Channels as well as a max loss in pips for risk levels.'''
    # parameters for Stochastic Oscillator and max loss in pips
    # Donchian Channels to determine previous support/resistance levels will use the given period as well
    # http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/Stochastic.php5 for Stochastic Oscillator formula and description
    params = (('period', 14), ('pfast', 3), ('pslow', 3), ('upperLimit', 80), ('lowerLimit', 20), ('stop_pips', .002))

    def __init__(self):
        '''Initializes logger and variables required for the strategy implementation.'''
        # initialize logger for log function (set to critical to prevent any unwanted autologs, not using log objects because only care about logging one thing)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(format='%(message)s', level=logging.CRITICAL, handlers=[
            logging.FileHandler("LOG.log"),
            logging.StreamHandler()
            ])

        self.order = None
        self.donchian_stop_price = None
        self.price = None
        self.stop_price = None
        self.stop_donchian = None

        self.stochastic = bt.indicators.Stochastic(self.data, period=self.params.period, period_dfast=self.params.pfast, period_dslow=self.params.pslow, 
        upperband=self.params.upperLimit, lowerband=self.params.lowerLimit)

    def log(self, txt, doprint=True):
        '''logs the pricing, orders, pnl, time/date, etc for each trade made in this strategy to a LOG.log file as well as to the terminal.'''
        date = self.data.datetime.date(0)
        time = self.data.datetime.time(0)
        if (doprint):
            logging.critical(str(date) + ' ' + str(time) + ' -- ' + txt)


    def notify_trade(self, trade):
        '''Run on every next iteration, logs the P/L with and without commission whenever a trade is closed.'''
        if trade.isclosed:
            self.log('CLOSE -- P/L gross: {}  net: {}'.format(trade.pnl, trade.pnlcomm))


    def notify_order(self, order):
        '''Run on every next iteration, logs the order execution status whenever an order is filled or rejected, 
        setting the order parameter back to None if the order is filled or cancelled to denote that there are no more pending orders.'''
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            if order.isbuy():
                self.log('BUY -- units: 10000  price: {}  value: {}  comm: {}'.format(order.executed.price, order.executed.value, order.executed.comm))
                self.price = order.executed.price
            elif order.issell():
                self.log('SELL -- units: 10000  price: {}  value: {}  comm: {}'.format(order.executed.price, order.executed.value, order.executed.comm))
                self.price = order.executed.price
        elif order.status in [order.Rejected, order.Margin]:
            self.log('Order rejected/margin')
        
        self.order = None


    def stop(self):
        '''At the end of the strategy backtest, logs the ending value of the portfolio as well as one or multiple parameter values for strategy optimization purposes.'''
        self.log('(period {}) Ending Value: {}'.format(self.params.period, self.broker.getvalue()), doprint=True)


    def next(self):
        '''Checks to see if Stochastic Oscillator, position, and order conditions meet the entry or exit conditions for the execution of buy and sell orders.'''
        if self.order:
            # if there is a pending order, don't do anything
            return
        if self.position.size == 0:
            # When stochastic crosses back below 80, enter short position.
            if self.stochastic.lines.percD[-1] >= 80 and self.stochastic.lines.percD[0] <= 80:
                # stop price at last support level in self.params.period periods
                self.donchian_stop_price = max(self.data.high.get(size=self.params.period))
                self.order = self.sell()
                # stop loss order for max loss of self.params.stop_pips pips
                self.stop_price = self.buy(exectype=bt.Order.Stop, price=self.data.close[0]+self.params.stop_pips, oco=self.stop_donchian)
                # stop loss order for donchian SR price level
                self.stop_donchian = self.buy(exectype=bt.Order.Stop, price=self.donchian_stop_price, oco=self.stop_price)
            # when stochastic crosses back above 20, enter long position.
            elif self.stochastic.lines.percD[-1] <= 20 and self.stochastic.lines.percD[0] >= 20:
                # stop price at last resistance level in self.params.period periods
                self.donchian_stop_price = min(self.data.low.get(size=self.params.period))
                self.order = self.buy()
                # stop loss order for max loss of self.params.stop_pips pips
                self.stop_price = self.sell(exectype=bt.Order.Stop, price=self.data.close[0]-self.params.stop_pips, oco=self.stop_donchian)
                # stop loss order for donchian SR price level
                self.stop_donchian = self.sell(exectype=bt.Order.Stop, price=self.donchian_stop_price, oco=self.stop_price) 
  
        if self.position.size > 0:
            # When stochastic is above 70, close out of long position
            if (self.stochastic.lines.percD[0] >= 70):
                self.close(oco=self.stop_price)
        if self.position.size < 0:
            # When stochastic is below 30, close out of short position
            if (self.stochastic.lines.percD[0] <= 30):
                self.close(oco=self.stop_price)