# -*- coding: utf-8 -*-
"""
因子预处理模块
================
原始因子值不能直接用于模型，需要进行三步标准化处理：

1. 去极值 (De-extreme): 剔除异常值，避免极端值主导模型
   方法: MAD (Median Absolute Deviation) 法
       - 计算 |x - median|
       - 取其中位数 MAD = median(|x - median|)
       - 设定边界 [median - n*1.4826*MAD, median + n*1.4826*MAD]
       - 超出边界的截断到边界 (winsorize)
       其中 1.4826 = 1/Φ^(-1)(0.75)，使 MAD 在正态下与 std 等价

2. 标准化 (Standardization): Z-score 标准化
       z = (x - mean) / std
   使每个因子均值为0、标准差为1，消除量纲差异

3. 中性化 (Neutralization): 控制其他变量的影响
   方法: 线性回归取残差
       y = Xβ + ε  →  ε = y - Xβ
   对每个截面，用市值和行业对因子做回归，取残差作为中性化后的因子
   目的：剥离行业和市场规模的混杂效应，得到"纯因子"
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def winsorize_mad(series, n=5):
    """
    MAD 法去极值（缩尾处理）

    Parameters
    ----------
    series : pd.Series
        原始因子值（一个截面上）
    n : float
        MAD 的倍数，常用3或5

    Returns
    -------
    pd.Series
        去极值后的因子值
    """
    median = series.median()
    mad = (series - median).abs().median()
    # 1.4826 让 MAD 在正态分布下与标准差等价
    sigma = 1.4826 * mad
    if sigma == 0 or np.isnan(sigma):
        return series
    lower = median - n * sigma
    upper = median + n * sigma
    return series.clip(lower, upper)


def standardize(series):
    """
    Z-score 标准化
    z = (x - mean) / std
    """
    std = series.std()
    if std == 0 or np.isnan(std):
        return series - series.mean()
    return (series - series.mean()) / std


def neutralize(factor_df, neut_vars, date_col="date"):
    """
    对因子值进行中性化处理（横截面回归取残差）

    步骤:
        对每个日期的截面：
            factor = β0 + β1*industry_dummy + β2*ln(mktcap) + ε
            取 ε 作为中性化后的因子值

    Parameters
    ----------
    factor_df : pd.DataFrame
        含因子值及中性化变量（行业dummy、市值等）
    neut_vars : list
        用于回归的中性化变量列名

    Returns
    -------
    pd.Series
        中性化后的因子残差
    """
    # 拼接自变量矩阵
    X = factor_df[neut_vars].values
    # 处理 NaN
    mask = ~np.isnan(X).any(axis=1)
    X_clean = X[mask]
    y_clean = factor_df.loc[factor_df.index[mask]].values

    if len(X_clean) == 0 or X_clean.shape[1] == 0:
        return pd.Series(factor_df.values, index=factor_df.index)

    # OLS 回归
    reg = LinearRegression().fit(X_clean, y_clean)
    residual = factor_df.values - reg.predict(X)
    return pd.Series(residual, index=factor_df.index)


def preprocess_factors(factors, stock_pool, neutralize_on=["SIZE", "INDUSTRY"]):
    """
    对因子表执行三步预处理：
        去极值 → 标准化 → 中性化（市值+行业）

    Parameters
    ----------
    factors : pd.DataFrame
        原始因子表
    stock_pool : pd.DataFrame
        股票池信息（含 industry 列，用于构造行业哑变量）
    neutralize_on : list
        中性化变量

    Returns
    -------
    pd.DataFrame
        预处理后的因子表
    """
    # 因子列
    factor_cols = ["EP", "BP", "SIZE", "MOM20", "MOM60",
                   "REV20", "VOL20", "TURN20", "ROE"]

    # 合并行业信息（来自股票池）
    industry_map = stock_pool[["stock_code", "industry"]].drop_duplicates()
    factors = factors.merge(industry_map, on="stock_code", how="left")
    # 行业哑变量
    industry_dummies = pd.get_dummies(factors["industry"], prefix="IND", drop_first=True)

    # 合并市值（用SIZE因子作为市值代理）
    neut_vars = ["SIZE"] + list(industry_dummies.columns)

    # 拼接
    factors_full = pd.concat([factors, industry_dummies], axis=1)

    processed_list = []
    # 按日期分组做截面处理
    for date, group in factors_full.groupby("date"):
        g = group.copy()
        for col in factor_cols:
            # 1. 去极值
            g[col] = winsorize_mad(g[col], n=5)
            # 2. 标准化
            g[col] = standardize(g[col])
            # 3. 中性化（除SIZE外，SIZE不再对自身中性化）
            if col != "SIZE":
                try:
                    X = g[neut_vars].astype(float).values
                    y = g[col].values
                    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
                    if mask.sum() > len(neut_vars) + 5:
                        reg = LinearRegression().fit(X[mask], y[mask])
                        g[col] = y - reg.predict(X)
                except Exception:
                    pass  # 出错则跳过中性化
        processed_list.append(g)

    processed = pd.concat(processed_list, ignore_index=True)
    # 删除中间列（行业哑变量），保留因子表
    keep_cols = ["date", "stock_code"] + factor_cols
    return processed[keep_cols]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_generator import generate_data
    from factor_engineering import build_all_factors

    price_df, fund_df, stock_pool = generate_data()
    factors = build_all_factors(price_df, fund_df)
    print(f"原始因子表维度: {factors.shape}")
    print(factors.head())

    processed = preprocess_factors(factors, stock_pool)
    print(f"\n预处理后因子表维度: {processed.shape}")
    print(processed.head())
    print("\n各因子均值（应接近0）:")
    print(processed.iloc[:, 2:].mean().round(4))
    print("\n各因子标准差（应接近1）:")
    print(processed.iloc[:, 2:].std().round(4))
