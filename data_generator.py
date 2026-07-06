# -*- coding: utf-8 -*-
"""
数据生成模块
================
本模块用于生成模拟的A股市场数据，包括：
1. 股票日线行情（OHLCV）
2. 基本面数据（ROE、市值、PE、PB、换手率）
3. 行业分类

说明：
    本项目出于"可复现、无需联网"的考虑，使用几何布朗运动(GBM)合成股票价格数据，
    并在收益率生成过程中刻意注入了小市值溢价、价值溢价、动量效应、质量溢价等
    真实市场中常见的因子效应，使得后续多因子模型能演示出有效的因子选股逻辑。

    实际生产环境中，可替换为 tushare / baostock / akshare 等数据源，
    接口签名保持一致即可。
"""
import numpy as np
import pandas as pd


# A股行业分类（简化版，覆盖常见的申万一级行业）
INDUSTRIES = [
    "银行", "非银金融", "食品饮料", "医药生物", "电子",
    "计算机", "通信", "传媒", "电力设备", "机械设备",
    "基础化工", "钢铁", "有色金属", "建筑材料", "房地产",
    "交通运输", "公用事业", "商贸零售", "纺织服饰", "汽车",
]

# 股票池：200只模拟股票
N_STOCKS = 200
# 回测时间区间
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"
# 无风险利率（年化），用于计算夏普比率
RISK_FREE_RATE = 0.025
# 调仓频率（月）
REBALANCE_MONTHS = 1


def generate_stock_pool(n_stocks=N_STOCKS, seed=42):
    """
    生成股票池：股票代码、行业、初始市值

    股票代码采用 A 股规则：
        - 沪市主板: 600xxx.SH
        - 深市主板: 000xxx.SZ
        - 创业板:   300xxx.SZ

    Parameters
    ----------
    n_stocks : int
        股票数量
    seed : int
        随机种子，保证可复现

    Returns
    -------
    pd.DataFrame
        列: ['stock_code', 'industry', 'list_date', 'initial_mktcap']
    """
    rng = np.random.default_rng(seed)

    codes = []
    # 沪市 600000-600999
    sh_codes = [f"600{i:03d}.SH" for i in range(0, 1000)]
    # 深市 000001-002999
    sz_codes = [f"{i:06d}.SZ" for i in range(1, 3000)]
    # 创业板 300000-300999
    cy_codes = [f"300{i:03d}.SZ" for i in range(0, 1000)]

    all_codes = sh_codes + sz_codes + cy_codes
    rng.shuffle(all_codes)
    codes = all_codes[:n_stocks]

    # 行业随机分配
    industries = rng.choice(INDUSTRIES, size=n_stocks)

    # 初始市值（亿元）：对数正态分布，使市值分布更接近真实
    # 中位市值约 100 亿，大票可达千亿
    initial_mktcap = rng.lognormal(mean=np.log(100), sigma=0.9, size=n_stocks)

    # 上市日期：在回测开始前 1-3 年上市
    list_dates = pd.date_range("2017-01-01", "2018-12-31", periods=n_stocks).to_pydatetime()
    rng.shuffle(list_dates)

    df = pd.DataFrame({
        "stock_code": codes,
        "industry": industries,
        "list_date": list_dates,
        "initial_mktcap": initial_mktcap,
    })
    return df


def generate_price_data(stock_pool, start_date=START_DATE, end_date=END_DATE,
                         seed=42):
    """
    生成日线行情数据

    收益率模型（注入因子效应）：
        r_t = r_f + β_size * (-log(mktcap))    # 小市值溢价
            + β_value * (EP)                    # 价值溢价
            + β_mom  * lag(r)                   # 动量效应
            + β_qual * ROE                      # 质量溢价
            + ε                                   # 随机扰动

    价格通过几何布朗运动(GBM)生成：
        S_{t} = S_{t-1} * exp(r_t)

    Parameters
    ----------
    stock_pool : pd.DataFrame
        generate_stock_pool 的输出
    start_date, end_date : str
        日期区间

    Returns
    -------
    pd.DataFrame
        列: ['date','stock_code','open','high','low','close','volume']
    """
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(start_date, end_date)  # 工作日
    n_days = len(dates)
    n_stocks = len(stock_pool)

    # ---------- 每只股票的基本面属性 ----------
    # ROE：正常分布在 0~25% 之间，部分公司为负
    roe = rng.normal(loc=8.0, scale=6.0, size=n_stocks)  # 单位 %
    roe = np.clip(roe, -15, 35)

    # 市净率PB：1~8 之间，部分金融股较低
    pb = rng.lognormal(mean=np.log(2.5), sigma=0.5, size=n_stocks)
    pb = np.clip(pb, 0.5, 12)

    # 换手率：日换手 0.2%~5%
    turnover = rng.lognormal(mean=np.log(1.2), sigma=0.6, size=n_stocks)

    # 初始市值（亿元）
    mktcap0 = stock_pool["initial_mktcap"].values

    # ---------- 生成收益率序列 ----------
    # 因子载荷（参数大小反映该因子的预测强度）
    BETA_SIZE = -0.015   # 小市值股票收益更高
    BETA_VALUE = 0.004   # EP 越大(便宜) 收益越高
    BETA_MOM = 0.25      # 动量效应：本月收益影响下月
    BETA_QUAL = 0.0015   # 高 ROE 收益更高
    VOL_DAILY = 0.02     # 日波动率

    # 标准化的小市值因子值（log 市值的负向 z-score）
    log_mktcap = np.log(mktcap0)
    size_factor = (log_mktcap - log_mktcap.mean()) / log_mktcap.std()
    size_factor = -size_factor  # 取负，使小市值因子值为正

    # 标准化 EP（市盈率倒数），先给定 PE 再倒推
    pe = rng.lognormal(mean=np.log(25), sigma=0.5, size=n_stocks)
    pe = np.clip(pe, 5, 120)
    ep = 1.0 / pe  # earnings-to-price
    ep_factor = (ep - ep.mean()) / ep.std()

    # 标准化 ROE
    roe_factor = (roe - roe.mean()) / roe.std()

    # 静态因子暴露（每只股票的横截面特征）
    static_alpha = (
        BETA_SIZE * size_factor
        + BETA_VALUE * ep_factor
        + BETA_QUAL * roe_factor
    ) / 252  # 转换为日收益

    # 横截面波动（每只股票独有的风险溢价）
    cross_sectional_vol = rng.normal(0, 0.0005, size=n_stocks)

    # 生成日收益率矩阵 (n_days, n_stocks)
    returns = np.zeros((n_days, n_stocks))
    prev_ret = np.zeros(n_stocks)
    for t in range(n_days):
        # 动量项（昨日收益 * 系数）
        momentum = BETA_MOM * prev_ret / 252
        # 基础收益（年化约 5%）+ 因子收益 + 动量 + 随机扰动
        base = 0.05 / 252
        noise = rng.normal(0, VOL_DAILY, size=n_stocks)
        r_t = base + static_alpha + momentum + cross_sectional_vol + noise
        returns[t] = r_t
        prev_ret = r_t

    # ---------- 累积成价格 ----------
    # 初始价格 5~50 元
    init_prices = rng.uniform(5, 50, size=n_stocks)
    price_paths = np.zeros_like(returns)
    price_paths[0] = init_prices
    for t in range(1, n_days):
        price_paths[t] = price_paths[t - 1] * np.exp(returns[t])

    # 市值随价格变动
    mktcap_paths = price_paths / price_paths[0] * mktcap0[:, None].T  # (n_days, n_stocks)

    # ---------- 组装成 DataFrame ----------
    records = []
    for i in range(n_stocks):
        for t in range(n_days):
            close = price_paths[t, i]
            # OHLC: 在收盘价基础上随机波动
            open_ = close * (1 + rng.normal(0, 0.005))
            high = max(open_, close) * (1 + abs(rng.normal(0, 0.005)))
            low = min(open_, close) * (1 - abs(rng.normal(0, 0.005)))
            # 成交量（手）：与市值、换手率正相关
            volume = mktcap_paths[t, i] * 1e8 * turnover[i] / 100 / 100  # 简化
            volume = int(volume)
            records.append({
                "date": dates[t],
                "stock_code": stock_pool["stock_code"].iloc[i],
                "open": round(open_, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume,
            })

    price_df = pd.DataFrame(records)

    # ---------- 基本面数据 ----------
    fundamentals = pd.DataFrame({
        "date": dates[0],  # 静态属性（实际应按财报披露更新）
        "stock_code": stock_pool["stock_code"],
        "roe": roe,
        "pe": pe,
        "pb": pb,
        "mktcap": mktcap0,         # 流通市值（亿元）
        "turnover": turnover,       # 日换手率 %
        "ep": ep,
        "bp": 1.0 / pb,
    })

    # 把基本面扩展到每个交易日（简化：假设季度披露，期间不变）
    fund_list = []
    quarterly_dates = pd.date_range(start_date, end_date, freq="QE")
    for qd in quarterly_dates:
        f = fundamentals.copy()
        f["date"] = qd
        fund_list.append(f)
    fund_df = pd.concat(fund_list, ignore_index=True)

    return price_df, fund_df


def generate_data():
    """
    生成完整模拟数据集

    Returns
    -------
    price_df : pd.DataFrame
        日线行情
    fund_df : pd.DataFrame
        基本面数据（按季度）
    stock_pool : pd.DataFrame
        股票池信息
    """
    stock_pool = generate_stock_pool()
    price_df, fund_df = generate_price_data(stock_pool)
    return price_df, fund_df, stock_pool


if __name__ == "__main__":
    price_df, fund_df, stock_pool = generate_data()
    print("=== 行情数据 ===")
    print(price_df.head())
    print(f"\n行情数据维度: {price_df.shape}")
    print(f"股票数量: {price_df['stock_code'].nunique()}")
    print(f"日期区间: {price_df['date'].min()} ~ {price_df['date'].max()}")

    print("\n=== 基本面数据 ===")
    print(fund_df.head())
    print(f"\n基本面数据维度: {fund_df.shape}")

    print("\n=== 股票池 ===")
    print(stock_pool.head())
    print(f"\n行业分布:")
    print(stock_pool["industry"].value_counts())
