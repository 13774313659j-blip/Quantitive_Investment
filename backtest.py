# -*- coding: utf-8 -*-
"""
回测模块
================
根据因子得分构建选股组合，并回测业绩表现。

回测流程：
    1. 在每个调仓日，根据模型得分对股票排序
    2. 选取得分最高的前 N 只股票等权构建组合
    3. 持有至下次调仓日，计算组合收益
    4. 与基准（全市场等权）对比

业绩评价指标：
    - 年化收益率
    - 年化波动率
    - 夏普比率 (Sharpe Ratio) = (年化收益 - 无风险利率) / 年化波动率
    - 最大回撤 (Max Drawdown)
    - 胜率相对基准
    - 超额收益
"""
import numpy as np
import pandas as pd


RISK_FREE_RATE = 0.025  # 年化无风险利率
TRADING_DAYS = 252      # 一年交易日数


def build_portfolio(result_df, score_col, top_n=20, rebalance_freq="M"):
    """
    根据得分选股构建组合

    Parameters
    ----------
    result_df : pd.DataFrame
        含日期、股票、得分、未来收益
    score_col : str
        得分列名
    top_n : int
        每次选股数量
    rebalance_freq : str
        调仓频率，'M'=月

    Returns
    -------
    pd.DataFrame
        每期组合收益：['rebalance_date','portfolio_ret','benchmark_ret','excess_ret']
    """
    df = result_df.copy()
    df["rebalance_date"] = df["date"].dt.to_period(rebalance_freq).dt.to_timestamp()

    records = []
    for rdate, group in df.groupby("rebalance_date"):
        if len(group) < top_n * 5:
            continue
        # 截面选股：得分最高的 top_n
        top = group.nlargest(top_n, score_col)
        # 等权组合收益
        port_ret = top["FWD_RET_5"].mean()
        # 基准：全部股票等权
        bench_ret = group["FWD_RET_5"].mean()
        records.append({
            "rebalance_date": rdate,
            "portfolio_ret": port_ret,
            "benchmark_ret": bench_ret,
            "excess_ret": port_ret - bench_ret,
            "n_stocks": len(top),
        })

    return pd.DataFrame(records)


def calc_performance(returns, freq=5, rf=RISK_FREE_RATE):
    """
    计算业绩评价指标

    Parameters
    ----------
    returns : pd.Series
        收益率序列
    freq : int
        收益率周期（5=周收益率，1=日收益率）
    rf : float
        年化无风险利率

    Returns
    -------
    dict
        各业绩指标
    """
    annual_factor = TRADING_DAYS / freq
    cum = (1 + returns).cumprod()

    # 年化收益率
    total_ret = cum.iloc[-1] - 1
    n_periods = len(returns)
    annual_ret = (1 + total_ret) ** (annual_factor / n_periods) - 1

    # 年化波动率
    annual_vol = returns.std() * np.sqrt(annual_factor)

    # 夏普比率
    sharpe = (annual_ret - rf) / annual_vol if annual_vol > 0 else 0

    # 最大回撤
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    return {
        "annual_return": annual_ret,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "total_return": total_ret,
        "win_rate": (returns > 0).mean(),
        "n_periods": n_periods,
    }


def backtest_strategy(result_df, score_col, top_n=20):
    """
    完整回测流程：构建组合 → 计算业绩

    Returns
    -------
    pd.DataFrame
        组合收益序列
    dict
        业绩指标
    """
    portfolio = build_portfolio(result_df, score_col, top_n=top_n)

    if len(portfolio) == 0:
        return pd.DataFrame(), {}

    perf_port = calc_performance(portfolio["portfolio_ret"], freq=5)
    perf_bench = calc_performance(portfolio["benchmark_ret"], freq=5)
    perf_excess = calc_performance(portfolio["excess_ret"], freq=5)

    return portfolio, {
        "portfolio": perf_port,
        "benchmark": perf_bench,
        "excess": perf_excess,
    }


def compare_strategies(result_df, score_cols, top_n=20):
    """
    对比多个策略的回测结果

    Returns
    -------
    pd.DataFrame
        每行一个策略
    """
    rows = []
    for col in score_cols:
        portfolio, perf = backtest_strategy(result_df, col, top_n=top_n)
        if not perf:
            continue
        p = perf["portfolio"]
        b = perf["benchmark"]
        rows.append({
            "strategy": col,
            "annual_return": p["annual_return"],
            "annual_volatility": p["annual_volatility"],
            "sharpe_ratio": p["sharpe_ratio"],
            "max_drawdown": p["max_drawdown"],
            "win_rate": p["win_rate"],
            "excess_annual_return": p["annual_return"] - b["annual_return"],
            "excess_sharpe": perf["excess"]["sharpe_ratio"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_generator import generate_data
    from factor_engineering import build_all_factors, calc_forward_returns
    from factor_preprocessing import preprocess_factors
    from ml_model import make_classification_label, walk_forward_train_predict

    price_df, fund_df, stock_pool = generate_data()
    factors = build_all_factors(price_df, fund_df)
    factors_proc = preprocess_factors(factors, stock_pool)
    fwd = calc_forward_returns(price_df, periods=(5,))

    merged = make_classification_label(factors_proc, fwd)
    result = walk_forward_train_predict(merged, train_months=12, test_months=1)

    portfolio, perf = backtest_strategy(result, "score_rf", top_n=20)
    print("=== 组合收益序列（前10期）===")
    print(portfolio.head(10))

    print("\n=== 业绩对比 ===")
    for k, v in perf.items():
        print(f"\n[{k}]")
        for m, n in v.items():
            print(f"  {m}: {n:.4f}")

    print("\n=== 多策略对比 ===")
    comparison = compare_strategies(result, ("score_eq", "score_lr", "score_rf"))
    print(comparison.round(4))
