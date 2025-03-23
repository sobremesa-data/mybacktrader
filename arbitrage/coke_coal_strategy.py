 class CokeCoalArbitrageStrategy(bt.Strategy):
    params = (
        ('window', 20),  # 移动窗口大小
        ('std_multiplier', 1),  # 标准差倍数
        ('coke_coal_ratio', 1.4),  # 焦炭焦煤比例
        ('other_cost', 0),  # 其他成本
    )

    def __init__(self):
        # 计算焦炭利润
        self.coke_profit = self.data0.close * self.p.coke_coal_ratio * self.data1.close - self.p.other_cost

        # 计算移动窗口均值和标准差
        self.meanval = bt.indicators.SimpleMovingAverage(self.coke_profit, period=self.p.window)
        self.stdval = bt.indicators.StdDev(self.coke_profit, period=self.p.window)

        # 上下穿越信号
        self.con_cross_1 = bt.indicators.CrossOver(self.coke_profit, self.meanval + self.p.std_multiplier * self.stdval)
        self.con_cross_2 = bt.indicators.CrossUnder(self.coke_profit, self.meanval - self.p.std_multiplier * self.stdval)

        # 交易状态
        self.order = None
        self.dont_reentry_long = False
        self.dont_reentry_short = False

    def next(self):
        # 如果有未完成订单，跳过
        if self.order:
            return

        # 做多条件
        if not self.position and self.con_cross_1 and not self.dont_reentry_long:
            self.buy(data=self.data0, price=self.data0.open[0])  # 做多焦炭
            self.dont_reentry_long = True
            self.dont_reentry_short = False

        # 做空条件
        elif not self.position and self.con_cross_2 and not self.dont_reentry_short:
            self.sell(data=self.data0, price=self.data0.open[0])  # 做空焦炭
            self.dont_reentry_short = True
            self.dont_reentry_long = False

    def notify_order(self, order):
        # 订单状态通知
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            self.order = None
            if order.isbuy():
                self.dont_reentry_long = False
            elif order.issell():
                self.dont_reentry_short = False

# ... existing code ...