# -*- coding: utf-8 -*-
"""
机器学习因子合成模块
========================
单一因子预测能力有限，需要将多个因子合成为一个综合得分。
本模块使用两种机器学习方法对比：

1. 等权合成 (Equal Weight, 基线)
   简单对所有因子取平均
   优点：稳健，无需训练；缺点：忽略因子间相对重要性

2. 逻辑回归 (Logistic Regression)
   把"未来是否跑赢中位数"作为二分类标签 y∈{0,1}：
       P(y=1) = σ(w·x + b),  σ(z) = 1/(1+e^{-z})
   通过最大化对数似然求解权重 w：
       L = Σ[y·log(σ) + (1-y)·log(1-σ)]
   优点：可解释性强、可给出因子权重；缺点：假设线性关系

3. 随机森林 (Random Forest)
   集成多棵决策树，每棵树在随机子样本+随机特征上训练，最终投票
   优点：自动捕捉非线性关系和因子交互；缺点：可解释性较弱

标签构造：
    label = 1 if forward_return > cross_section_median else 0
    （二分类，避免回归对极端值敏感）

样本外验证 (Walk-Forward)：
    训练集用历史数据，测试集用未来数据，避免数据泄露
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


FACTOR_COLS = ["EP", "BP", "SIZE", "MOM20", "MOM60",
               "REV20", "VOL20", "TURN20", "ROE"]


def make_classification_label(factors, fwd_returns, fwd_col="FWD_RET_5"):
    """
    构造二分类标签：未来收益是否高于截面中位数

    Returns
    -------
    pd.DataFrame
        含 ['date','stock_code', 因子列, 'label', fwd_col]
    """
    merged = factors.merge(
        fwd_returns[["date", "stock_code", fwd_col]],
        on=["date", "stock_code"], how="inner"
    )
    merged = merged.dropna(subset=FACTOR_COLS + [fwd_col])

    # 截面中位数标签
    merged["label"] = merged.groupby("date")[fwd_col].transform(
        lambda x: (x > x.median()).astype(int)
    )
    return merged


def equal_weight_score(factors):
    """
    等权合成：对所有因子取平均
    """
    df = factors.copy()
    df["score_eq"] = df[FACTOR_COLS].mean(axis=1)
    return df


def train_logistic(train_df, factor_cols=FACTOR_COLS):
    """
    训练逻辑回归模型

    使用 L2 正则化（C=1.0）防止过拟合
    class_weight='balanced' 处理样本不平衡
    """
    X_train = train_df[factor_cols].values
    y_train = train_df["label"].values
    model = LogisticRegression(
        C=1.0, max_iter=1000,
        class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(train_df, factor_cols=FACTOR_COLS):
    """
    训练随机森林分类器

    n_estimators=200: 200棵树
    max_depth=6: 限制深度防过拟合
    min_samples_leaf=20: 叶子节点最小样本数
    """
    X_train = train_df[factor_cols].values
    y_train = train_df["label"].values
    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def walk_forward_train_predict(merged_df, train_months=12, test_months=1):
    """
    Walk-Forward 滚动训练：
        训练集 = 过去 train_months 个月
        测试集 = 下 test_months 个月
        每次窗口向前滚动

    这种方法模拟真实场景下"用历史数据训练，预测未来"，
    避免使用未来数据（数据泄露）。

    Returns
    -------
    pd.DataFrame
        含预测得分和真实标签的测试集
    """
    # 按月重采样得到训练/测试边界
    merged_df = merged_df.sort_values("date").copy()
    merged_df["month"] = merged_df["date"].dt.to_period("M")
    months = sorted(merged_df["month"].unique())

    results = []
    # 至少要训练集 + 测试集
    for i in range(train_months, len(months) - test_months + 1, test_months):
        train_months_list = months[i - train_months:i]
        test_months_list = months[i:i + test_months]

        train_df = merged_df[merged_df["month"].isin(train_months_list)]
        test_df = merged_df[merged_df["month"].isin(test_months_list)]

        if len(train_df) < 500 or len(test_df) < 50:
            continue

        # 训练两种模型
        lr_model = train_logistic(train_df)
        rf_model = train_random_forest(train_df)

        # 预测
        X_test = test_df[FACTOR_COLS].values

        # 逻辑回归: P(y=1)
        test_df = test_df.copy()
        test_df["score_lr"] = lr_model.predict_proba(X_test)[:, 1]
        # 随机森林: P(y=1)
        test_df["score_rf"] = rf_model.predict_proba(X_test)[:, 1]
        # 等权
        test_df["score_eq"] = test_df[FACTOR_COLS].mean(axis=1)

        results.append(test_df)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def evaluate_predictions(result_df, score_cols=("score_eq", "score_lr", "score_rf")):
    """
    评估各模型预测效果

    指标：
        1. AUC: 预测概率与真实标签的排序能力，0.5为随机
        2. IC: 得分与未来收益的 Spearman 相关
        3. Accuracy: 分类准确率
    """
    summary = []
    for col in score_cols:
        row = {"model": col}

        # AUC
        try:
            row["AUC"] = roc_auc_score(result_df["label"], result_df[col])
        except Exception:
            row["AUC"] = np.nan

        # IC（得分 vs 未来收益）
        from scipy.stats import spearmanr
        ic_list = []
        for date, g in result_df.groupby("date"):
            if len(g) > 10:
                ic, _ = spearmanr(g[col], g["FWD_RET_5"])
                if not np.isnan(ic):
                    ic_list.append(ic)
        if ic_list:
            ic_series = pd.Series(ic_list)
            row["IC_mean"] = ic_series.mean()
            row["ICIR"] = ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0
        else:
            row["IC_mean"] = np.nan
            row["ICIR"] = np.nan

        # 分类准确率
        pred = (result_df[col] > 0.5).astype(int)
        row["accuracy"] = accuracy_score(result_df["label"], pred)

        summary.append(row)

    return pd.DataFrame(summary)


def get_factor_importance(merged_df, factor_cols=FACTOR_COLS):
    """
    使用全样本训练随机森林，提取因子重要性
    （feature_importances_ 基于不纯度下降量计算）

    Returns
    -------
    pd.DataFrame
        列: ['factor','importance']
    """
    X = merged_df[factor_cols].values
    y = merged_df["label"].values
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X, y)
    return pd.DataFrame({
        "factor": factor_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_generator import generate_data
    from factor_engineering import build_all_factors, calc_forward_returns
    from factor_preprocessing import preprocess_factors

    price_df, fund_df, stock_pool = generate_data()
    factors = build_all_factors(price_df, fund_df)
    factors_proc = preprocess_factors(factors, stock_pool)
    fwd = calc_forward_returns(price_df, periods=(5,))

    merged = make_classification_label(factors_proc, fwd)
    print(f"合并后样本量: {len(merged)}")
    print(f"正样本比例: {merged['label'].mean():.3f}")

    result = walk_forward_train_predict(merged, train_months=12, test_months=1)
    print(f"\n样本外预测结果维度: {result.shape}")
    print(result[["date", "stock_code", "score_eq", "score_lr", "score_rf",
                   "label", "FWD_RET_5"]].head())

    print("\n=== 模型评估 ===")
    eval_df = evaluate_predictions(result)
    print(eval_df.round(4))

    print("\n=== 因子重要性 ===")
    imp = get_factor_importance(merged)
    print(imp)
