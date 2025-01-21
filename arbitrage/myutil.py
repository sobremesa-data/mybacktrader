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
def calculate_spread(df_I, df_RB, columns=['open', 'high', 'low', 'close', 'volume']):
    """
    计算两个DataFrame之间的价差
    """
    # 对齐数据
    df_I_aligned, df_RB_aligned = check_and_align_data(df_I, df_RB)
    
    # 创建价差DataFrame
    df_spread = pd.DataFrame(index=df_I_aligned.index)
    
    # 对每个列进行相减
    for col in columns:
        if col in df_I_aligned.columns and col in df_RB_aligned.columns:
            df_spread[f'{col}'] = 5*df_I_aligned[col] - df_RB_aligned[col]
    
    return df_spread.reset_index()