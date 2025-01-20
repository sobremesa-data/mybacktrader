import backtrader as bt
import pandas as pd

class SpreadBollingerStrategy(bt.Strategy):
    '''基于布林带的价差套利策略'''
    
    params = (
        ('period', 20),       # 布林带周期
        ('devfactor', 2),     # 布林带标准差倍数
        ('size', 1),         # 交易规模
    )
    
    def __init__(self):
        # 计算价差
        self.spread = bt.indicators.Spread(self.data0, self.data1,plot=True,subplot=True)
        
        # 布林带指标
        self.boll = bt.indicators.BollingerBands(
            self.spread,
            period=self.p.period,
            devfactor=self.p.devfactor
        )
        
        # # 关闭原始数据的绘图
        # self.data0.plotinfo.plot = False
        # self.data1.plotinfo.plot = False
        
        # # 设置spread和布林带在同一个子图中显示
        # self.spread.plotinfo.plotmaster = self.boll
        
    def next(self):
        # 如果没有持仓
        if not self.position:
            # 当价差低于下轨，做多价差（买入data0，卖出data1）
            if self.spread[0] < self.boll.lines.bot[0]:
                print(f'做多价差: {self.datetime.date()}')
                print(f'Spread: {self.spread[0]:.2f}, 下轨: {self.boll.lines.bot[0]:.2f}')
                self.buy(data=self.data0, size=self.p.size)
                self.sell(data=self.data1, size=self.p.size)
                
            # 当价差高于上轨，做空价差（卖出data0，买入data1）
            elif self.spread[0] > self.boll.lines.top[0]:
                print(f'做空价差: {self.datetime.date()}')
                print(f'Spread: {self.spread[0]:.2f}, 上轨: {self.boll.lines.top[0]:.2f}')
                self.sell(data=self.data0, size=self.p.size)
                self.buy(data=self.data1, size=self.p.size)
        
        # 如果持有多头价差仓位
        elif self.getposition(self.data0).size > 0:
            # 当价差回归到中轨，平仓
            if self.spread[0] >= self.boll.lines.mid[0]:
                print(f'平多价差: {self.datetime.date()}')
                print(f'Spread: {self.spread[0]:.2f}, 中轨: {self.boll.lines.mid[0]:.2f}')
                self.close(data=self.data0)
                self.close(data=self.data1)
        
        # 如果持有空头价差仓位
        elif self.getposition(self.data0).size < 0:
            # 当价差回归到中轨，平仓
            if self.spread[0] <= self.boll.lines.mid[0]:
                print(f'平空价差: {self.datetime.date()}')
                print(f'Spread: {self.spread[0]:.2f}, 中轨: {self.boll.lines.mid[0]:.2f}')
                self.close(data=self.data0)
                self.close(data=self.data1)

# 创建回测引擎
cerebro = bt.Cerebro()

# 读取数据
output_file = 'D:\\FutureData\\ricequant\\1d_2017to2024_noadjust.h5'
df_I = pd.read_hdf(output_file, key='/I').reset_index()
df_RB = pd.read_hdf(output_file, key='/RB').reset_index()

# 添加数据
data0 = bt.feeds.PandasData(
    dataname=df_I,
    datetime='date',
    nocase=True,
)
data1 = bt.feeds.PandasData(
    dataname=df_RB,
    datetime='date',
    nocase=True,
)

cerebro.adddata(data0, name='I')  # 铁矿石期货
cerebro.adddata(data1, name='RB')  # 螺纹钢期货

# 设置初始资金
cerebro.broker.setcash(1000000.0)

# 添加策略
cerebro.addstrategy(SpreadBollingerStrategy)

# 运行回测
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
results = cerebro.run()
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

# 绘制结果
cerebro.plot(
)