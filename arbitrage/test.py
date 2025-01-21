import backtrader as bt
import pandas as pd

output_file = 'D:\\FutureData\\ricequant\\1d_2017to2024_noadjust.h5'
df_I = pd.read_hdf(output_file, key='/I').reset_index()
df_RB = pd.read_hdf(output_file, key='/RB').reset_index()

# 在策略中计算spread
class MyStrategy(bt.Strategy):
    def __init__(self):

        self.spread = bt.indicators.Spread(self.data0, self.data1)
        
        # 添加其他指标
        self.sma = bt.indicators.SMA(self.spread, period=20)
        self.rsi = bt.indicators.RSI(self.spread)

# class St(bt.Strategy):
#     def __init__(self):
#         self.sma = bt.indicators.SimpleMovingAverage(self.spread,subplot =True)


data0 =bt.feeds.PandasData(dataname=df_I,

                           datetime='date',
                           nocase=True,
                           )
data1 =bt.feeds.PandasData(dataname=df_RB,

                           datetime='date',
                           nocase=True,
                           )
cerebro = bt.Cerebro()
cerebro.adddata(data0, name='STOCK1')
cerebro.adddata(data1, name='STOCK2')
cerebro.addstrategy(MyStrategy)
cerebro.run(oldsync = True)
cerebro.plot(numfigs=5)

