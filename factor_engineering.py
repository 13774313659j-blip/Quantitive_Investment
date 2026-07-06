# -*- coding: utf-8 -*-
"""
因子构建模块
================
本模块从原始行情和基本面数据计算各类选股因子。

因子分类（参考Barra/CNE5体系简化版）：
    1. 价值类:    EP（市盈率倒数）、BP（市净率倒数）
    2. 规模类:    LNCAP（对数市值，体现小市值效应）
    3. 动量类:    MOM20、MOM60（过去20/60日收益）
    4. 反转类:    REV20（短期反转，过去20日收益的负值）
    5. 波动率类:  VOL20（20日收益率标准差）
    6. 流动性类:  TURN20（20日平均换手率）
    7. 质量类:    ROE（净资产收益率）

注意：因子值在截面上越大，理论上未来收益越高（除规模、波动率、反转外）。
"""
import numpy as np
import pandas as pd


def calc_value_factors(fund_df):
    """
    价值因子: EP, BP
    EP = 1 / PE，BP = 1 / PB
    高 EP/BP 表示估值便宜，理论上有价值溢价
    """
    df = fund_df.copy()
    df["EP"] = 1.0 / df["pe"]
    df["BP"] = 1.0 / df["pb"]
    return df[["date", "stock_code", "EP", "BP"]]


def calc_size_factor(fund_df):
    """
    规模因子: LNCAP = ln(流通市值)
    小市值溢价效应：小盘股收益通常高于大盘股
    为使方向一致（值越大预测收益越高），取负: SIZE = -ln(mktcap)
    """
    df = fund_df.copy()
    df["SIZE"] = -np.log(df["mktcap"])
    return df[["date", "stock_code", "SIZE"]]


def calc_momentum_reversal(price_df, windows=(20, 60)):
    """
    动量因子与反转因子
    MOM_N  = 过去N日累计收益率
    REV20  = -MOM20  （短期反转，过去涨得多的未来收益低）

    Parameters
    ----------
    price_df : pd.DataFrame
        行情数据
    windows : tuple
        动量计算窗口
    """
    # 转为宽表：行=日期, 列=股票, 值=收盘价
    close = price_df.pivot(index="date", columns="stock_code", values="close")

    records = []
    for w in windows:
        # 过去 w 日收益率 = close_t / close_{t-w} - 1
        mom = close.pct_change(w)
        mom = mom.stack().reset_index()
        mom.columns = ["date", "stock_code", f"MOM{w}"]
        records.append(mom)

    # 反转因子
    rev = (-close.pct_change(20)).stack().reset_index()
    rev.columns = ["date", "stock_code", "REV20"]
    records.append(rev)

    result = records[0]
    for r in records[1:]:
        result = result.merge(r, on=["date", "stock_code"], how="outer")
    return result


def calc_volatility_factor(price_df, window=20):
    """
    波动率因子: VOL_N = 过去N日日收益率的标准差
    低波动率异常（Low Volatility Anomaly）：低波动率股票反而收益更好
    为使方向一致，取负: VOL = -std
    """
    close = price_df.pivot(index="date", columns="stock_code", values="close")
    daily_ret = close.pct_change()
    vol = -daily_ret.rolling(window).std()
    result = vol.stack().reset_index()
    result.columns = ["date", "stock_code", f"VOL{window}"]
    return result


def calc_liquidity_factor(fund_df, price_df, window=20):
    """
    流动性因子: TURN_N = 过去N日平均换手率
    高换手率通常预示未来收益较低（流动性折价）
    取负使方向一致: TURN = -avg_turnover
    """
    # 用最新的换手率作为代理（实际应使用滚动均值）
    turn = fund_df[["date", "stock_code", "turnover"]].copy()
    turn = turn.rename(columns={"turnover": "TURN20"})
    turn["TURN20"] = -turn["TURN20"]
    return turn


def calc_quality_factor(fund_df):
    """
    质量因子: ROE
    高 ROE 公司盈利能力强，长期有质量溢价
    """
    return fund_df[["date", "stock_code", "roe"]].rename(columns={"roe": "ROE"})


def build_all_factors(price_df, fund_df):
    """
    构建全部因子并合并为一张因子表

    Returns
    -------
    pd.DataFrame
        列: ['date', 'stock_code', 'EP', 'BP', 'SIZE', 'MOM20',
             'MOM60', 'REV20', 'VOL20', 'TURN20', 'ROE']
    """
    # 各类因子
    val = calc_value_factors(fund_df)
    size = calc_size_factor(fund_df)
    mom_rev = calc_momentum_reversal(price_df)
    vol = calc_volatility_factor(price_df)
    liq = calc_liquidity_factor(fund_df, price_df)
    qual = calc_quality_factor(fund_df)

    # 依次合并
    factors = val.merge(size, on=["date", "stock_code"], how="outer")
    factors = factors.merge(mom_rev, on=["date", "stock_code"], how="outer")
    factors = factors.merge(vol, on=["date", "stock_code"], how="outer")
    factors = factors.merge(liq, on=["date", "stock_code"], how="outer")
    factors = factors.merge(qual, on=["date", "stock_code"], how="outer")

    # 前向填充基本面类因子（季度披露，期间保持不变）
    factor_cols = ["EP", "BP", "SIZE", "TURN20", "ROE"]
    for col in factor_cols:
        factors = factors.sort_values(["stock_code", "date"])
        factors[col] = factors.groupby("stock_code")[col].ffill()

    # 删除前期因窗口不足产生的 NaN
    factors = factors.dropna()
    return factors


def calc_forward_returns(price_df, periods=(1, 5, 10, 20)):
    """
    计算未来N日收益率（用于IC检验和训练标签）

    Parameters
    ----------
    periods : tuple
        预测周期列表（日）

    Returns
    -------
    pd.DataFrame
        列: ['date','stock_code','FWD_RET_1','FWD_RET_5',...]
    """
    close = price_df.pivot(index="date", columns="stock_code", values="close")
    fwd_dfs = []
    for p in periods:
        fwd = close.shift(-p) / close - 1
        fwd = fwd.stack().reset_index()
        fwd.columns = ["date", "stock_code", f"FWD_RET_{p}"]
        fwd_dfs.append(fwd)

    result = fwd_dfs[0]
    for d in fwd_dfs[1:]:
        result = result.merge(d, on=["date", "stock_code"], how="outer")
    return result


if __name__ == "__main__":
    from data_generator import generate_data
    price_df, fund_df, _ = generate_data()
    factors = build_all_factors(price_df, fund_df)
    print("=== 因子表 ===")
    print(factors.head(10))
    print(f"\n因子表维度: {factors.shape}")
    print(f"\n各因子描述统计:")
    print(factors.iloc[:, 2:].describe().round(3))
