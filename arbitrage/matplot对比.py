import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 读取数据
output_file = 'D:\\FutureData\\ricequant\\1d_2017to2024_noadjust.h5'
df_I = pd.read_hdf(output_file, key='/I').reset_index()
df_RB = pd.read_hdf(output_file, key='/RB').reset_index()

# 计算价差
spread = df_I['close'] - df_RB['close']

# 计算移动平均和标准差
period = 20
spread_ma = spread.rolling(window=period).mean()
spread_std = spread.rolling(window=period).std()

# 计算上下轨
spread_open = 2.0
spread_close = 0.5
upper = spread_ma + spread_open * spread_std
lower = spread_ma - spread_open * spread_std

# 创建图表
plt.figure(figsize=(16, 8))

# 绘制spread价差
plt.plot(df_I['date'], spread, label='Spread (I-RB)', color='gray', alpha=0.6)
plt.plot(df_I['date'], spread_ma, label='MA20', color='blue', linewidth=1)
plt.plot(df_I['date'], upper, label='Upper Band', color='red', linestyle='--', alpha=0.7)
plt.plot(df_I['date'], lower, label='Lower Band', color='green', linestyle='--', alpha=0.7)

# 设置图表格式
plt.title('I-RB Spread with Bands', fontsize=12)
plt.xlabel('Date')
plt.ylabel('Price Spread')
plt.grid(True, alpha=0.3)
plt.legend()

# 旋转x轴日期标签
plt.xticks(rotation=45)

# 自动调整布局
plt.tight_layout()

# 显示图表
plt.show()

# 打印一些基本统计信息
print(f"Spread统计信息:")
print(f"平均值: {spread.mean():.2f}")
print(f"标准差: {spread.std():.2f}")
print(f"最大值: {spread.max():.2f}")
print(f"最小值: {spread.min():.2f}")