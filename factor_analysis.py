# -*- coding: utf-8 -*-
"""
因子有效性检验模块
====================
检验因子对未来收益的预测能力，常用两种方法：

1. IC 分析 (Information Coefficient)
   IC = corr(factor_t, forward_return_{t+1})
   即因子值与下一期收益的 Spearman 秩相关系数
   指标：
       - IC 均值：|IC| > 0.03 通常视为有效因子，>0.05 较强
       - ICIR = IC均值 / IC标准差：衡量因子稳定性，>0.5 较好
       - IC 胜率：IC > 0 的比例

2. 分层回测 (Layered / Stratified Backtest)
   按因子值排序，分5组（quantile），比较各组未来收益：
       - 多空收益 = 第5组 - 第1组
       - 单调性：因子值越大收益应越高（强因子）
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def calc_ic_series(factors, fwd_returns, factor_name, fwd_col="FWD_RET_5"):
    """
    计算因子的 IC 时间序列

    Parameters
    ----------
    factors : pd.DataFrame
        因子表
    fwd_returns : pd.DataFrame
        未来收益表
    factor_name : str
        因子列名
    fwd_col : str
        未来收益列名

    Returns
    -------
    pd.Series
        IC 时间序列（按日期排序）
    """
    merged = factors.merge(fwd_returns, on=["date", "stock_code"], how="inner")
    merged = merged.dropna(subset=[factor_name, fwd_col])

    ic_list = []
    for date, group in merged.groupby("date"):
        if len(group) < 10:
            continue
        # Spearman 秩相关系数
        ic, _ = spearmanr(group[factor_name], group[fwd_col])
        ic_list.append({"date": date, "IC": ic})

    return pd.DataFrame(ic_list).set_index("date")["IC"]


def summarize_ic(ic_series):
    """
    汇总 IC 统计指标
    """
    return {
        "IC_mean": ic_series.mean(),
        "IC_std": ic_series.std(),
        "ICIR": ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
        "IC_win_rate": (ic_series > 0).mean(),
        "IC_abs_gt_0.03": (ic_series.abs() > 0.03).mean(),
    }


def layered_backtest(factors, fwd_returns, factor_name, fwd_col="FWD_RET_5",
                     n_groups=5):
    """
    分层回测：按因子值分5组，统计各组未来收益

    Parameters
    ----------
    n_groups : int
        分组数量

    Returns
    -------
    pd.DataFrame
        每组在每个截面的平均收益，列: ['date','group','ret']
    """
    merged = factors.merge(fwd_returns, on=["date", "stock_code"], how="inner")
    merged = merged.dropna(subset=[factor_name, fwd_col])

    records = []
    for date, group in merged.groupby("date"):
        if len(group) < n_groups * 5:
            continue
        # qcut 分组
        try:
            group = group.copy()
            group["group"] = pd.qcut(group[factor_name], n_groups,
                                     labels=False, duplicates="drop")
        except Exception:
            continue
        for g, sub in group.groupby("group"):
            records.append({
                "date": date,
                "group": int(g) + 1,
                "ret": sub[fwd_col].mean(),
            })

    return pd.DataFrame(records)


def calc_long_short_return(layered_df):
    """
    计算多空收益（最高组 - 最低组）
    """
    pivot = layered_df.pivot(index="date", columns="group", values="ret")
    if pivot.shape[1] < 2:
        return pd.Series(dtype=float)
    long_short = pivot.iloc[:, -1] - pivot.iloc[:, 0]
    return long_short


def analyze_all_factors(factors, fwd_returns, factor_names, fwd_col="FWD_RET_5",
                        n_groups=5):
    """
    对所有因子进行IC分析和分层回测，汇总结果

    Returns
    -------
    pd.DataFrame
        每行一个因子，列: ['factor','IC_mean','IC_std','ICIR','IC_win_rate',
                      'long_short_ret','long_short_sharpe']
    pd.DataFrame
        分层回测结果
    """
    summary_list = []
    layered_all = []

    for fname in factor_names:
        # IC 分析
        ic_series = calc_ic_series(factors, fwd_returns, fname, fwd_col)
        if len(ic_series) == 0:
            continue
        ic_stats = summarize_ic(ic_series)

        # 分层回测
        layered = layered_backtest(factors, fwd_returns, fname, fwd_col, n_groups)
        layered["factor"] = fname
        layered_all.append(layered)

        ls_ret = calc_long_short_return(layered)
        ls_mean = ls_ret.mean() if len(ls_ret) > 0 else np.nan
        ls_std = ls_ret.std() if len(ls_ret) > 0 else np.nan
        # 年化夏普（假设 fwd_col 是5日收益）
        annual_factor = 252 / 5
        ls_sharpe = (ls_mean / ls_std * np.sqrt(annual_factor)
                     if ls_std > 0 else 0)

        summary_list.append({
            "factor": fname,
            "IC_mean": ic_stats["IC_mean"],
            "IC_std": ic_stats["IC_std"],
            "ICIR": ic_stats["ICIR"],
            "IC_win_rate": ic_stats["IC_win_rate"],
            "long_short_ret": ls_mean,
            "long_short_sharpe": ls_sharpe,
        })

    summary_df = pd.DataFrame(summary_list)
    layered_df = pd.concat(layered_all, ignore_index=True) if layered_all else pd.DataFrame()
    return summary_df, layered_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_generator import generate_data
    from factor_engineering import build_all_factors, calc_forward_returns

    price_df, fund_df, _ = generate_data()
    factors = build_all_factors(price_df, fund_df)
    fwd = calc_forward_returns(price_df, periods=(5,))

    summary, layered = analyze_all_factors(
        factors, fwd,
        ["EP", "BP", "SIZE", "MOM20", "MOM60", "REV20", "VOL20", "TURN20", "ROE"],
        fwd_col="FWD_RET_5"
    )
    print("=== 因子IC检验汇总 ===")
    print(summary.round(4))
    print(f"\n=== 分层回测结果（前20行）===")
    print(layered.head(20))
