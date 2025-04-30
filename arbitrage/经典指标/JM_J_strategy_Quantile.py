import backtrader as bt
import pandas as pd
import numpy as np
import sys
import os
import datetime

from arbitrage.myutil import calculate_spread, check_and_align_data, cointegration_ratio
# https://mp.weixin.qq.com/s/na-5duJiRM1fTJF0WrcptA


def calculate_rolling_spread(
        df0: pd.DataFrame,          # 必含 'date' 与价格列
        df1: pd.DataFrame,
        window: int = 90,
        fields=('open', 'high', 'low', 'close')
    ) -> pd.DataFrame:
    """
    计算滚动 β，并为指定价格字段生成价差 (spread)：
        spread_x = price0_x - β_{t-1} * price1_x
    """
    # 1) 用收盘价对齐合并（β 仍用 close 估计）
    df = (df0.set_index('date')[['close']]
              .rename(columns={'close': 'close0'})
              .join(df1.set_index('date')[['close']]
                        .rename(columns={'close': 'close1'}),
                    how='inner'))

    # 2) 估计 β_t ，再向前挪一天
    beta_raw   = df['close0'].rolling(window).cov(df['close1']) / \
                 df['close1'].rolling(window).var()
    beta_shift = beta_raw.shift(1).round(1)        # 防未来 + 保留 1 位小数

    # 3) 把 β 拼回主表（便于后面 vectorized 计算）
    df = df.assign(beta=beta_shift)

    # 4) 对每个字段算 spread
    out_cols = {'date': df.index, 'beta': beta_shift}
    for f in fields:
        if f not in ('open','high','low','close'):
            raise ValueError(f'未知字段 {f}')
        p0 = df0.set_index('date')[f]
        p1 = df1.set_index('date')[f]
        aligned = p0.to_frame(name=f'price0_{f}').join(
                  p1.to_frame(name=f'price1_{f}'), how='inner')
        spread_f = aligned[f'price0_{f}'] - beta_shift * aligned[f'price1_{f}']
        out_cols[f'{f}'] = spread_f

    # 5) 整理输出
    out = (pd.DataFrame(out_cols)
             .dropna()
             .reset_index(drop=True))
    out['date'] = pd.to_datetime(out['date'])
    return out
# 读取数据
output_file = 'D:\\FutureData\\ricequant\\1d_2017to2024_noadjust.h5'
df0 = pd.read_hdf(output_file, key='/J').reset_index()
df1 = pd.read_hdf(output_file, key='/JM').reset_index()

# 确保日期列格式正确
df0['date'] = pd.to_datetime(df0['date'])
df1['date'] = pd.to_datetime(df1['date'])

# 计算滚动价差
df_spread = calculate_rolling_spread(df0, df1, window=90)
print("滚动价差计算完成，系数示例：")
print(df_spread.head())

fromdate = datetime.datetime(2018, 1, 1)
todate = datetime.datetime(2025, 1, 1)

# 创建自定义数据类以支持beta列
class SpreadData(bt.feeds.PandasData):
    lines = ('beta',)  # 添加beta线
    
    params = (
        ('datetime', 'date'),  # 日期列
        ('close', 'close'),    # 价差列作为close
        ('beta', 'beta'),      # beta列
        ('nocase', True),      # 列名不区分大小写
    )

# 添加数据
data0 = bt.feeds.PandasData(dataname=df0, datetime='date', nocase=True, fromdate=fromdate, todate=todate)
data1 = bt.feeds.PandasData(dataname=df1, datetime='date', nocase=True, fromdate=fromdate, todate=todate)
data2 = SpreadData(dataname=df_spread, fromdate=fromdate, todate=todate)

# 创建分位数指标（自定义）
class QuantileIndicator(bt.Indicator):
    lines = ('upper', 'lower', 'mid')
    params = (
        ('period', 30),
        ('upper_quantile', 0.85),  # 上轨分位数
        ('lower_quantile', 0.15),  # 下轨分位数
    )

    def __init__(self):
        self.addminperiod(self.p.period)
        self.spread_data = []

    def next(self):
        self.spread_data.append(self.data[0])
        if len(self.spread_data) > self.p.period:
            self.spread_data.pop(0)  # 保持固定长度

        if len(self.spread_data) >= self.p.period:
            spread_array = np.array(self.spread_data)
            self.lines.upper[0] = np.quantile(spread_array, self.p.upper_quantile)
            self.lines.lower[0] = np.quantile(spread_array, self.p.lower_quantile)
            self.lines.mid[0] = np.median(spread_array)
        else:
            self.lines.upper[0] = self.data[0]
            self.lines.lower[0] = self.data[0]
            self.lines.mid[0] = self.data[0]


class DynamicSpreadQuantileStrategy(bt.Strategy):
    params = (
        ('lookback_period', 60),    # 回看周期
        ('upper_quantile', 0.9),   # 上轨分位数（90%）
        ('lower_quantile', 0.1),   # 下轨分位数（10%）
    )

    def __init__(self):
        # 计算价差的分位数指标
        self.quantile = QuantileIndicator(
            self.data2.close, 
            period=self.p.lookback_period,
            upper_quantile=self.p.upper_quantile,
            lower_quantile=self.p.lower_quantile,
            subplot=True
        )

        # 交易状态
        self.order = None
        self.entry_price = 0

    def next(self):
        if self.order:
            return

        # 获取当前beta值
        current_beta = self.data2.beta[0]
        
        # 处理缺失beta情况
        if pd.isna(current_beta) or current_beta <= 0:
            return
            
        # 动态设置交易规模
        self.size0 = 10  # 固定J的规模
        self.size1 = round(current_beta * 10)  # 根据beta调整JM的规模
        
        # 打印调试信息
        if len(self) % 20 == 0:  # 每20个bar打印一次，减少输出
            print(f'{self.datetime.date()}: beta={current_beta}, J:{self.size0}手, JM:{self.size1}手')

        # 使用分位数指标进行交易决策
        spread = self.data2.close[0]
        upper_band = self.quantile.upper[0]
        lower_band = self.quantile.lower[0]
        mid_band = self.quantile.mid[0]
        pos = self.getposition(self.data0).size

        # 开平仓逻辑
        if pos == 0:  # 没有持仓
            if spread > upper_band:
                # 价差高于上轨（90%分位数），做空价差（做空J，做多JM）
                self._open_position(short=True)
            elif spread < lower_band:
                # 价差低于下轨（10%分位数），做多价差（做多J，做空JM）
                self._open_position(short=False)
        else:  # 已有持仓
            if pos > 0 and spread >= mid_band:  # 持有多头且价差回归到中位数
                self._close_positions()
            elif pos < 0 and spread <= mid_band:  # 持有空头且价差回归到中位数
                self._close_positions()

    def _open_position(self, short):
        '''动态配比下单'''
        # 确认交易规模有效
        if not hasattr(self, 'size0') or not hasattr(self, 'size1'):
            self.size0 = 10  # 默认值
            self.size1 = round(self.data2.beta[0] * 10) if not pd.isna(self.data2.beta[0]) else 14
        
        if short:
            print(f'做空J {self.size0}手, 做多JM {self.size1}手')
            self.buy(data=self.data0, size=self.size0)
            self.sell(data=self.data1, size=self.size1)
        else:
            print(f'做多J {self.size0}手, 做空JM {self.size1}手')
            self.sell(data=self.data0, size=self.size0)
            self.buy(data=self.data1, size=self.size1)
        self.entry_price = self.data2.close[0]

    def _close_positions(self):
        self.close(data=self.data0)
        self.close(data=self.data1)

    def notify_trade(self, trade):
        if trade.isclosed:
            print('TRADE %s CLOSED %s, PROFIT: GROSS %.2f, NET %.2f, PRICE %d' %
                  (trade.ref, bt.num2date(trade.dtclose), trade.pnl, trade.pnlcomm, trade.value))
        elif trade.justopened:
            print('TRADE %s OPENED %s  , SIZE %2d, PRICE %d ' % (
            trade.ref, bt.num2date(trade.dtopen), trade.size, trade.value))

# 创建回测引擎
cerebro = bt.Cerebro(stdstats=False)
cerebro.adddata(data0, name='data0')
cerebro.adddata(data1, name='data1')
cerebro.adddata(data2, name='spread')

# 添加策略
cerebro.addstrategy(DynamicSpreadQuantileStrategy)

# 设置初始资金
cerebro.broker.setcash(100000)
cerebro.broker.set_shortcash(False)
cerebro.addanalyzer(bt.analyzers.DrawDown)  # 回撤分析器
cerebro.addanalyzer(bt.analyzers.ROIAnalyzer, period=bt.TimeFrame.Days)
cerebro.addanalyzer(bt.analyzers.SharpeRatio,
                    timeframe=bt.TimeFrame.Days,  # 按日数据计算
                    riskfreerate=0,            # 默认年化1%的风险无风险利率
                    annualize=True,           # 不进行年化
                    )
cerebro.addanalyzer(bt.analyzers.Returns,
                    tann=bt.TimeFrame.Days,  # 年化因子，252 个交易日
                    )
# cerebro.addanalyzer(bt.analyzers.CAGRAnalyzer, period=bt.TimeFrame.Days)  # 这里的period可以是daily, weekly, monthly等
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)

cerebro.addobserver(bt.observers.Trades)
cerebro.addobserver(bt.observers.CumValue)

# 运行回测
results = cerebro.run()

# 获取分析结果
drawdown = results[0].analyzers.drawdown.get_analysis()
sharpe = results[0].analyzers.sharperatio.get_analysis()
roi = results[0].analyzers.roianalyzer.get_analysis()
total_returns = results[0].analyzers.returns.get_analysis()  # 获取总回报率
# cagr = results[0].analyzers.cagranalyzer.get_analysis()

# 打印分析结果
print("=============回测结果================")
print(f"\nSharpe Ratio: {sharpe['sharperatio']:.2f}")
print(f"Drawdown: {drawdown['max']['drawdown']:.2f} %")
print(f"Annualized/Normalized return: {total_returns['rnorm100']:.2f}%")  #
print(f"Total compound return: {roi['roi100']:.2f}%")
# print(f"年化收益: {cagr['cagr']:.2f} ")
# print(f"夏普比率: {cagr['sharpe']:.2f}")
# 绘制结果
cerebro.plot(volume=False, spread=True) 