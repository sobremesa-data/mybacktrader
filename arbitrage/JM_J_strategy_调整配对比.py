import backtrader as bt
import pandas as pd
import numpy as np
import sys
import os
import datetime

from arbitrage.myutil import calculate_spread, check_and_align_data,cointegration_ratio
# https://mp.weixin.qq.com/s/na-5duJiRM1fTJF0WrcptA



class DynamicSpreadStrategy(bt.Strategy):
    params = (
        ('period', 15),
        ('devfactor', 1.5),
        ('lookback_months', 6),  # 回溯月数
    )

    def __init__(self):
        # 动态交易规模
        self.size0 = 10  # 默认初始值
        self.size1 = 14

        # 布林带指标
        self.boll = bt.indicators.BollingerBands(
            self.data2.close,
            period=self.p.period,
            devfactor=self.p.devfactor,
            subplot=False
        )

        # 添加每月定时器
        self.add_timer(
            when=bt.timer.SESSION_START,
            monthdays=[1],  # 每月首日触发
            monthcarry=True
        )

        # 交易状态
        self.order = None
        self.entry_price = 0

    def notify_timer(self, timer, when, *args,  ** kwargs):
        '''每月重新计算配比'''
        # 获取过去6个月数据
        lookback_days = self.p.lookback_months * 21  # 按每月21个交易日估算
        if len(self.data0) < lookback_days or len(self.data1) < lookback_days:
            return

        # 获取价格序列
        prices0 = np.array(self.data0.close.get(size=lookback_days))
        prices1 = np.array(self.data1.close.get(size=lookback_days))

        # 计算最新配比
        beta_ratio, _ = cointegration_ratio(prices0, prices1)
        self.size0, self.size1 = beta_ratio

        print(f'{self.datetime.date()}: 更新配比 铁矿石:{self.size0}手 螺纹钢:{self.size1}手')

    def next(self):
        if self.order:
            return

        spread = self.data2.close[0]
        mid = self.boll.lines.mid[0]
        pos = self.getposition(self.data0).size

        # 开平仓逻辑
        if pos == 0:
            if spread > self.boll.lines.top[0]:
                self._open_position(short=True)
            elif spread < self.boll.lines.bot[0]:
                self._open_position(short=False)
        else:
            if (spread <= mid and pos < 0) or (spread >= mid and pos > 0):
                self._close_positions()

    def _open_position(self, short):
        '''动态配比下单'''
        if short:
            self.sell(data=self.data0, size=self.size0)
            self.buy(data=self.data1, size=self.size1)
        else:
            self.buy(data=self.data0, size=self.size0)
            self.sell(data=self.data1, size=self.size1)
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

    # def notify_order(self, order):
    #     if order.status in [order.Submitted, order.Accepted]:
    #         # 订单状态 submitted/accepted，处于未决订单状态。
    #         return
    #
    #     # 订单已决，执行如下语句
    #     if order.status in [order.Completed]:
    #         if order.isbuy():
    #             print(f'executed date {bt.num2date(order.executed.dt)},executed price {order.executed.price}, created date {bt.num2date(order.created.dt)}')


# 读取数据
output_file = 'D:\\FutureData\\ricequant\\1d_2017to2024_noadjust.h5'
df0 = pd.read_hdf(output_file, key='/J').reset_index()
df1 = pd.read_hdf(output_file, key='/JM').reset_index()

## 炼焦利润=焦炭coal期货价格-1.4*焦煤期货价格-其他成本
df_spread = calculate_spread(df0, df1, 1, 1.4)
print(df0.head())

fromdate = datetime.datetime(2020, 1, 1)
todate = datetime.datetime(2025, 1, 1)

# 添加数据
data0 = bt.feeds.PandasData(dataname=df0, datetime='date', nocase=True, fromdate=fromdate, todate=todate)
data1 = bt.feeds.PandasData(dataname=df1, datetime='date', nocase=True, fromdate=fromdate, todate=todate)
data2 = bt.feeds.PandasData(dataname=df_spread, datetime='date', nocase=True, fromdate=fromdate, todate=todate)

# 创建回测引擎
cerebro = bt.Cerebro(stdstats=False)
cerebro.adddata(data0, name='data0')
cerebro.adddata(data1, name='data1')
cerebro.adddata(data2, name='spread')

# cerebro.broker.setcommission(
#     commission=0.001,  # 0.1% 费率
#     margin=False,       # 非保证金交易
#     mult=1,            # 价格乘数
# )
# # # 百分比滑点
# cerebro.broker.set_slippage_perc(
#     perc=0.0005,        # 0.5% 滑点
#     slip_open=True,    # 影响开盘价
#     slip_limit=True,   # 影响限价单
#     slip_match=True,   # 调整成交价
#     slip_out=True      # 允许滑出价格范围
# )
# 添加策略
cerebro.addstrategy(DynamicSpreadStrategy)
##########################################################################################
# 设置初始资金
cerebro.broker.setcash(80000)
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
cerebro.addanalyzer(bt.analyzers.CAGRAnalyzer, period=bt.TimeFrame.Days)  # 这里的period可以是daily, weekly, monthly等
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)

# cerebro.addobserver(bt.observers.CashValue)
# cerebro.addobserver(bt.observers.Value)

cerebro.addobserver(bt.observers.Trades)
# cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.CumValue)

# 运行回测
results = cerebro.run()
#
# 获取分析结果
drawdown = results[0].analyzers.drawdown.get_analysis()
sharpe = results[0].analyzers.sharperatio.get_analysis()
roi = results[0].analyzers.roianalyzer.get_analysis()
total_returns = results[0].analyzers.returns.get_analysis()  # 获取总回报率
cagr = results[0].analyzers.cagranalyzer.get_analysis()
trade_analysis = results[0].analyzers.tradeanalyzer.get_analysis()  # 通过名称获取分析结果

# # 打印分析结果
print("=============回测结果================")
print(f"\nSharpe Ratio: {sharpe['sharperatio']:.2f}")
print(f"Drawdown: {drawdown['max']['drawdown']:.2f} %")
print(f"Annualized/Normalized return: {total_returns['rnorm100']:.2f}%")  #
print(f"Total compound return: {roi['roi100']:.2f}%")
print(f"年化收益: {cagr['cagr']:.2f} ")
print(f"夏普比率: {cagr['sharpe']:.2f}")
print(trade_analysis)
# 绘制结果
cerebro.plot(volume=False, spread=True)

