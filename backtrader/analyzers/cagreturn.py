import math
import pandas as pd
import numpy as np
import backtrader as bt
from collections import OrderedDict
from backtrader import TimeFrameAnalyzerBase

class CAGRAnalyzer(TimeFrameAnalyzerBase):
    '''Calculates the Compound Annual Growth Rate (CAGR) for a strategy.
    
    Params:
    
      - ``period`` (default: ``daily``):
        The time period to consider for annualization, options: `daily`, `weekly`, `monthly`, `yearly`
        
      - ``fund`` (default: ``None``):
        If ``None`` the actual mode of the broker (fundmode - True/False) will be autodetected to decide if the returns are based on the total net asset value or on the fund value.
    '''
    
    params = (
        ('period', None),  # Default is daily
        ('fund', None),       # Optionally set fund mode
    )
    
    _TANN = {
        bt.TimeFrame.Days: 252.0,    # Daily data
        bt.TimeFrame.Weeks: 52.0,    # Weekly data
        bt.TimeFrame.Months: 12.0,   # Monthly data
        bt.TimeFrame.Years: 1.0,     # Yearly data
    }
  
    def start(self):
        '''Initialize the start value and fundmode'''
        super(CAGRAnalyzer, self).start()


        # 获取初始值（可以是策略的资产值或者基金值）
        if not self._fundmode:
            # self._value_start = self.strategy.broker.getvalue()
            self._value_start = self.strategy.broker._valuemkt

        else:
            self._value_start = self.strategy.broker.fundvalue
        
        # 初始化累计收益率的初始值
        self._cum_return = 1.0  # 用1.0来初始化，以便于累乘

        # 用于存储收益率的时间步
        self._returns = []

    def stop(self):
        '''Calculate CAGR and store results'''
        super(CAGRAnalyzer, self).stop()

        # 根据传入的周期（如daily）选择年化因子
        annual_factor = self._TANN.get(self.p.period, 252.0)
        
        # 计算总年数
        num_years = len(self._returns) / annual_factor  # 数据长度除以年化因子
        
        # 计算年化复合增长率（CAGR）
        if num_years > 0:
            cagr = (self._cum_return) ** (1 / num_years) - 1
        else:
            cagr = 0.0  # 如果没有数据，设置为0
        #计算sharp self.p.riskfreerate
        mean_return = np.mean(self._returns) * annual_factor
        var_return = np.std(self._returns) * np.sqrt(annual_factor)
        sharpe = mean_return/var_return
        # 存储结果
        self.rets['cagr'] = cagr
        self.rets['cagr100'] = cagr * 100  # 转换为百分比形式
        self.rets['sharpe'] = sharpe

    def next(self):
        '''Calculate returns on each time step'''
        # 计算每个时间步骤的收益率

        # if self._value_start != 0.0:


        # current_value = self.strategy.broker.getvalue() if not self._fundmode else self.strategy.broker.fundvalue
        current_value = self.strategy.broker._valuemkt if not self._fundmode else self.strategy.broker.fundvalue

        #分子分母为0
        daily_return =0 if self._value_start == 0 else  (current_value / self._value_start) - 1
        daily_return = 0 if daily_return == -1 else daily_return

        self._returns.append(daily_return)  # 将当前时间的收益率存储起来

        # 累乘每日收益率
        self._cum_return *= (1 + daily_return)
        # print(self._cum_return,daily_return,self.strategy.broker._valuemkt,self.strategy.broker.getvalue(),self.strategy.broker.getcash())


        # 更新初始值（当新的时间段（天、周、月等）开始时，使用当前值作为新的初始值）
        # self._value_start = self.strategy.broker.getvalue() if not self._fundmode else self.strategy.broker.fundvalue
        self._value_start = self.strategy.broker._valuemkt if not self._fundmode else self.strategy.broker.fundvalue

    def get_analysis(self):
        '''Returns the CAGR value'''
        return self.rets