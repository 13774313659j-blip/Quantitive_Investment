# -*- coding: utf-8 -*-
"""
多因子选股模型 - 主流程
=========================
完整Pipeline:
    1. 数据生成       → data_generator
    2. 因子构建       → factor_engineering
    3. 因子预处理     → factor_preprocessing
    4. 因子有效性检验 → factor_analysis (IC分析)
    5. 机器学习因子合成→ ml_model (逻辑回归/随机森林)
    6. 组合回测       → backtest
    7. 业绩评价 + 输出 → 结果保存到 results/

运行方式：
    python src/main.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# 把 src 目录加入 path（便于 import）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import generate_data
from factor_engineering import build_all_factors, calc_forward_returns
from factor_preprocessing import preprocess_factors
from factor_analysis import analyze_all_factors
from ml_model import (make_classification_label, walk_forward_train_predict,
                      evaluate_predictions, get_factor_importance)
from backtest import compare_strategies, backtest_strategy

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

FACTOR_COLS = ["EP", "BP", "SIZE", "MOM20", "MOM60",
               "REV20", "VOL20", "TURN20", "ROE"]


def save_csv(df, filename):
    """保存到 results/ 目录"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  ✓ 已保存: {path}")


def step1_data():
    """步骤1: 生成模拟数据"""
    print("\n" + "=" * 60)
    print("步骤 1/6: 生成模拟 A 股数据")
    print("=" * 60)
    price_df, fund_df, stock_pool = generate_data()

    os.makedirs(DATA_DIR, exist_ok=True)
    save_csv(price_df, os.path.join("..", "data", "price_data.csv"))
    save_csv(fund_df, os.path.join("..", "data", "fundamental_data.csv"))
    save_csv(stock_pool, os.path.join("..", "data", "stock_pool.csv"))

    print(f"\n  股票数量: {price_df['stock_code'].nunique()}")
    print(f"  日期范围: {price_df['date'].min()} ~ {price_df['date'].max()}")
    print(f"  行情数据: {price_df.shape}")
    print(f"  基本面数据: {fund_df.shape}")
    return price_df, fund_df, stock_pool


def step2_factors(price_df, fund_df):
    """步骤2: 构建因子"""
    print("\n" + "=" * 60)
    print("步骤 2/6: 构建选股因子")
    print("=" * 60)
    factors = build_all_factors(price_df, fund_df)
    fwd = calc_forward_returns(price_df, periods=(5,))
    print(f"  因子表维度: {factors.shape}")
    print(f"  因子列表: {FACTOR_COLS}")
    print(f"\n  各因子描述统计:")
    desc = factors[FACTOR_COLS].describe().round(4)
    print(desc)
    save_csv(factors, "factors_raw.csv")
    return factors, fwd


def step3_preprocess(factors, stock_pool):
    """步骤3: 因子预处理"""
    print("\n" + "=" * 60)
    print("步骤 3/6: 因子预处理（去极值 + 标准化 + 中性化）")
    print("=" * 60)
    factors_proc = preprocess_factors(factors, stock_pool)
    print(f"  预处理后维度: {factors_proc.shape}")
    print(f"\n  均值（应接近0）:")
    print(factors_proc[FACTOR_COLS].mean().round(4))
    print(f"\n  标准差（应接近1）:")
    print(factors_proc[FACTOR_COLS].std().round(4))
    save_csv(factors_proc, "factors_processed.csv")
    return factors_proc


def step4_factor_analysis(factors, fwd):
    """步骤4: 因子有效性检验"""
    print("\n" + "=" * 60)
    print("步骤 4/6: 因子有效性检验（IC分析 + 分层回测）")
    print("=" * 60)
    summary, layered = analyze_all_factors(
        factors, fwd, FACTOR_COLS, fwd_col="FWD_RET_5", n_groups=5
    )
    print("\n  因子IC检验汇总:")
    print(summary.round(4))
    save_csv(summary, "factor_ic_summary.csv")
    if len(layered) > 0:
        save_csv(layered, "factor_layered_backtest.csv")
    return summary, layered


def step5_ml(factors_proc, fwd):
    """步骤5: 机器学习因子合成"""
    print("\n" + "=" * 60)
    print("步骤 5/6: 机器学习因子合成（Walk-Forward 滚动训练）")
    print("=" * 60)

    merged = make_classification_label(factors_proc, fwd)
    print(f"  训练样本量: {len(merged)}")
    print(f"  正样本比例: {merged['label'].mean():.3f}")

    # Walk-Forward 滚动训练
    result = walk_forward_train_predict(merged, train_months=12, test_months=1)
    print(f"  样本外预测样本量: {len(result)}")

    # 模型评估
    eval_df = evaluate_predictions(result)
    print("\n  模型评估结果:")
    print(eval_df.round(4))
    save_csv(eval_df, "model_evaluation.csv")

    # 因子重要性
    importance = get_factor_importance(merged)
    print("\n  因子重要性排序:")
    print(importance)
    save_csv(importance, "factor_importance.csv")

    save_csv(result[["date", "stock_code", "score_eq", "score_lr",
                      "score_rf", "label", "FWD_RET_5"]], "model_predictions.csv")
    return result, eval_df, importance


def step6_backtest(result):
    """步骤6: 组合回测"""
    print("\n" + "=" * 60)
    print("步骤 6/6: 组合回测与业绩评价")
    print("=" * 60)

    strategies = ["score_eq", "score_lr", "score_rf"]
    comparison = compare_strategies(result, strategies, top_n=20)
    print("\n  策略业绩对比:")
    print(comparison.round(4))
    save_csv(comparison, "strategy_comparison.csv")

    # 详细输出随机森林策略的组合收益序列
    portfolio, perf = backtest_strategy(result, "score_rf", top_n=20)
    print(f"\n  随机森林策略详细业绩:")
    for k, v in perf.items():
        print(f"\n  [{k}]")
        for m, n in v.items():
            print(f"    {m}: {n:.4f}")
    save_csv(portfolio, "portfolio_returns.csv")

    return comparison, portfolio, perf


def main():
    print("=" * 60)
    print("  A股多因子选股模型 - 项目主流程")
    print("  Multi-Factor Stock Selection Model for A-Shares")
    print("=" * 60)

    # 1. 数据
    price_df, fund_df, stock_pool = step1_data()

    # 2. 因子构建
    factors, fwd = step2_factors(price_df, fund_df)

    # 3. 因子预处理
    factors_proc = step3_preprocess(factors, stock_pool)

    # 4. 因子有效性检验
    step4_factor_analysis(factors_proc, fwd)

    # 5. 机器学习因子合成
    result, eval_df, importance = step5_ml(factors_proc, fwd)

    # 6. 回测
    comparison, portfolio, perf = step6_backtest(result)

    print("\n" + "=" * 60)
    print("  全部流程已完成！")
    print(f"  结果文件保存在: {RESULTS_DIR}")
    print("=" * 60)

    # 输出文件列表
    print("\n生成的文件:")
    for fname in sorted(os.listdir(RESULTS_DIR)):
        path = os.path.join(RESULTS_DIR, fname)
        size = os.path.getsize(path) / 1024
        print(f"  - {fname}  ({size:.1f} KB)")


if __name__ == "__main__":
    main()
