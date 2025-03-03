import pandas as pd
import numpy as np

# 1. 首先确认两个DataFrame的index是否相同
def check_and_align_data(df1, df2, date_column='date'):
    """
    检查并对齐两个DataFrame的数据
    """
    # 确保date列作为index
    if date_column in df1.columns:
        df1 = df1.set_index(date_column)
    if date_column in df2.columns:
        df2 = df2.set_index(date_column)
    
    # 找出共同的日期
    common_dates = df1.index.intersection(df2.index)
    
    # 检查是否有缺失的日期
    missing_in_df1 = df2.index.difference(df1.index)
    missing_in_df2 = df1.index.difference(df2.index)
    
    if len(missing_in_df1) > 0:
        print(f"在df_I中缺失的日期数: {len(missing_in_df1)}")
    if len(missing_in_df2) > 0:
        print(f"在df_RB中缺失的日期数: {len(missing_in_df2)}")
    
    # 对齐数据
    df1_aligned = df1.loc[common_dates]
    df2_aligned = df2.loc[common_dates]
    
    return df1_aligned, df2_aligned

# 2. 计算价差
def calculate_spread(df1, df2, factor1=5,factor2=1, columns=['open', 'high', 'low', 'close', 'volume']):
    """
    计算两个DataFrame之间的价差
    :param df1: 第一个DataFrame
    :param df2: 第二个DataFrame
    :param columns: 需要计算价差的列
    :param factor: 价差计算时的乘数因子
    :return: 包含价差的DataFrame
    """
    # 对齐数据
    df1_aligned, df2_aligned = check_and_align_data(df1, df2)

    # 创建价差DataFrame
    df_spread = pd.DataFrame(index=df1_aligned.index)

    # 对每个列进行相减
    for col in columns:
        if col in df1_aligned.columns and col in df2_aligned.columns:
            df_spread[f'{col}'] = factor1 * df1_aligned[col] - factor2*df2_aligned[col]

    return df_spread.reset_index()


def calculate_regression_ratio(price_a, price_b, ma, mb):
    """
    回归匹配持仓比例（整数版）
    :param price_a: 品种A价格序列（pd.Series）
    :param price_b: 品种B价格序列（pd.Series）
    :param ma: 品种A合约乘数
    :param mb: 品种B合约乘数
    :return: 整数配比 (Na, Nb)
    """
    # 对齐数据
    merged = pd.concat([price_a, price_b], axis=1).dropna()

    # 线性回归获取μ系数（自变量price_a，因变量price_b）
    beta = np.polyfit(merged.iloc[:, 0], merged.iloc[:, 1], 1)[0]

    # 获取最新价格
    pa = merged.iloc[-1, 0]
    pb = merged.iloc[-1, 1]

    # 计算理论配比
    ratio = beta * (ma * pa) / (mb * pb)

    # 返回最简整数比
    return simplify_ratio(ratio)


def calculate_volatility_ratio(price_c, price_d, mc, md):
    """
    波动率匹配持仓比例（整数版）
    :param price_c: 品种C价格序列（pd.Series）
    :param price_d: 品种D价格序列（pd.Series）
    :param mc: 品种C合约乘数
    :param md: 品种D合约乘数
    :return: 整数配比 (Nc, Nd)
    """
    # 对齐数据
    merged = pd.concat([price_c, price_d], axis=1).dropna()

    # 计算年化波动率（假设日频数据）
    vol_c = np.log(merged.iloc[:, 0] / merged.iloc[:, 0].shift(1)).std() * np.sqrt(252)
    vol_d = np.log(merged.iloc[:, 1] / merged.iloc[:, 1].shift(1)).std() * np.sqrt(252)

    # 获取最新价格
    pc = merged.iloc[-1, 0]
    pd = merged.iloc[-1, 1]

    # 计算理论配比
    ratio = (vol_c * mc * pc) / (vol_d * md * pd)

    # 返回最简整数比
    return simplify_ratio(ratio)


def calculate_contract_value_ratio(price_e, price_f, me, mf):
    """
    合约价值匹配持仓比例（整数版）
    :param price_e: 品种E价格序列（pd.Series）
    :param price_f: 品种F价格序列（pd.Series）
    :param me: 品种E合约乘数
    :param mf: 品种F合约乘数
    :return: 整数配比 (Ne, Nf)
    """
    # 对齐数据
    merged = pd.concat([price_e, price_f], axis=1).dropna()

    # 获取最新价格
    pe = merged.iloc[-1, 0]
    pf = merged.iloc[-1, 1]

    # 计算理论配比
    ratio = (me * pe) / (mf * pf)

    # 返回最简整数比
    return simplify_ratio(ratio)


def simplify_ratio(ratio, max_denominator=100):
    """
    将浮点比例转换为最简整数比
    :param ratio: 浮点比例值
    :param max_denominator: 最大允许的分母值
    :return: (分子, 分母) 的元组
    """
    from fractions import Fraction
    frac = Fraction(ratio).limit_denominator(max_denominator)
    return (frac.numerator, frac.denominator)