import backtrader as bt
import pandas as pd
import sys
import os

from arbitrage.myutil import calculate_spread, check_and_align_data


# 焦炭利润策略
class SpreadBollingerStrategy(bt.Strategy):
    params = (
        ('period', 15),  # 布林带周期
        ('devfactor', 1.5),  # 布林带标准差倍数
        ('size_j', 1),  # 焦炭交易规模
        ('size_jm', 1.4),  # 焦煤交易规模
    )

    def __init__(self):
        # 布林带指标
        self.boll = bt.indicators.BollingerBands(
            self.data2.close,  # 使用外部计算的价差
            period=self.p.period,
            devfactor=self.p.devfactor,
            subplot=False
        )

        # 交易状态
        self.order = None

        # 记录交易信息
        self.trades = []
        self.current_trade = None

        # 记录每年的净值
        self.year_values = {}

    def next(self):
        # 如果有未完成订单，跳过
        if self.order:
            return

        # 获取当前价差
        spread = self.data2.close[0]
        upper = self.boll.lines.top[0]
        lower = self.boll.lines.bot[0]
        if not self.position:  # 如果没有持仓，则买入
            self.order = self.buy(data=self.data0, size=self.p.size_jm, price=self.data0.close[0])
            self.order = self.sell(data=self.data1, size=self.p.size_j, price=self.data0.close[0])

        # # 交易逻辑
        # if not self.position:
        #     # 开仓条件
        #     if spread > upper:
        #         # 做空价差：卖焦炭买焦煤
        #         self.sell(data=self.data0, size=self.p.size_coke)  # 卖5手焦炭
        #         self.buy(data=self.data1, size=self.p.size_coal)  # 买1手焦煤
        #         self.current_trade = {
        #             'entry_date': self.data.datetime.date(0),
        #             'entry_price': spread,
        #             'type': 'short'
        #         }
        #
        #     elif spread < lower:
        #         # 做多价差：买焦炭卖焦煤
        #         self.buy(data=self.data0, size=self.p.size_coke)  # 买5手焦炭
        #         self.sell(data=self.data1, size=self.p.size_coal)  # 卖1手焦煤
        #         self.current_trade = {
        #             'entry_date': self.data.datetime.date(0),
        #             'entry_price': spread,
        #             'type': 'long'
        #         }
        #
        # else:
        #     # 平仓条件 如果价差回归到中轨以下，说明均值回归可能已经完成，平掉多头仓位。
        #     if (spread <= self.boll.lines.mid[0] and self.position.size > 0) or \
        #             (spread >= self.boll.lines.mid[0] and self.position.size < 0):
        #         self.close(data=self.data0)
        #         self.close(data=self.data1)
        #         if self.current_trade:
        #             self.current_trade['exit_date'] = self.data.datetime.date(0)
        #             self.current_trade['exit_price'] = spread
        #             self.current_trade['pnl'] = (spread - self.current_trade['entry_price']) * (
        #                 -1 if self.current_trade['type'] == 'short' else 1)
        #             self.trades.append(self.current_trade)
        #             self.current_trade = None

    def notify_order(self, order):
        # 订单状态通知
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            self.order = None


# 读取数据
output_file = 'D:\\FutureData\\ricequant\\1d_2017to2024_noadjust.h5'
df_JM = pd.read_hdf(output_file, key='/JM').reset_index()
df_J = pd.read_hdf(output_file, key='/J').reset_index()

# 计算价差
# df_coke, df_coal = check_and_align_data(df_coke, df_coal)
## 炼焦利润 = 焦炭期货价格 - 1.4 焦煤期货价格
df_spread = calculate_spread(df_JM, df_J, 1, 1.4)

print(f"价差数据形状: {df_spread.shape}")

# 添加数据
data0 = bt.feeds.PandasData(dataname=df_JM, datetime='date', nocase=True)
data1 = bt.feeds.PandasData(dataname=df_J, datetime='date', nocase=True)
data2 = bt.feeds.PandasData(dataname=df_spread, datetime='date', nocase=True)

# 创建回测引擎
cerebro = bt.Cerebro()
cerebro.adddata(data0, name='coke')
cerebro.adddata(data1, name='coal')
cerebro.adddata(data2, name='spread')

# 添加策略
cerebro.addstrategy(SpreadBollingerStrategy)

##########################################################################################
# 设置初始资金
cerebro.broker.setcash(1000000.0)

# 添加分析器：SharpeRatio、DrawDown、AnnualReturn 和 Returns
cerebro.addanalyzer(bt.analyzers.SharpeRatio,
                    timeframe=bt.TimeFrame.Days,  # 按日数据计算
                    riskfreerate=0,  # 默认年化1%的风险无风险利率
                    annualize=True,  # 不进行年化
                    fund = True

                    )
cerebro.addanalyzer(bt.analyzers.AnnualReturn, fund = True)
cerebro.addanalyzer(bt.analyzers.DrawDown, fund = True)  # 回撤分析器
# cerebro.addanalyzer(bt.analyzers.Returns,
#                     tann=bt.TimeFrame.Days,  # 年化因子，252 个交易日
#                     )  # 自定义名称

cerebro.addanalyzer(bt.analyzers.CAGRAnalyzer, period=bt.TimeFrame.Days)  # 这里的period可以是daily, weekly, monthly等
# 运行回测
results = cerebro.run()

# 获取分析结果
sharpe = results[0].analyzers.sharperatio.get_analysis()
drawdown = results[0].analyzers.drawdown.get_analysis()
annual_returns = results[0].analyzers.annualreturn.get_analysis()
# total_returns = results[0].analyzers.returns.get_analysis()  # 获取总回报率
cagr = results[0].analyzers.cagranalyzer.get_analysis()

# 打印分析结果
print("=============回测结果================")
print(f"\n夏普比率: {sharpe['sharperatio']:.2f}")
print(f"最大回撤: {drawdown['max']['drawdown']:.2f} %")
# print(f"总回报率: {total_returns['rnorm100']:.2f}%")  # 打印总回报率
print(f"年化收益: {cagr['cagr100']:.2f} %")

# # 打印年度回报率
# print("\n年度回报率:")
# print("=" * 80)
# print("{:<8} {:<12}".format("年份", "回报率"))
# for year, return_rate in annual_returns.items():
#     print("{:<8} {:<12.2%}".format(year, return_rate))
# 绘制结果
cerebro.plot(volume=False)