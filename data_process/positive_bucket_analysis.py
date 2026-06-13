import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "results.csv"

OUTPUT_BUCKET_PATH = BASE_DIR / "positive_bucket_analysis.csv"
OUTPUT_RECOMMEND_PATH = BASE_DIR / "positive_bucket_recommendation.txt"
PLOT_PATH = BASE_DIR / "positive_bucket_plot.png"


def build_bins():
    """
    只分析被判为正类的样本（默认 prob >= 0.5）
    并将这些正类预测样本按概率分桶。
    """
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0000001]
    labels = [
        "0.50-0.60",
        "0.60-0.70",
        "0.70-0.80",
        "0.80-0.90",
        "0.90-0.95",
        "0.95-1.00",
    ]
    return bins, labels


def analyze_positive_buckets(df):
    required_cols = {"Label", "prob"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV 必须包含列: {required_cols}，当前列为: {list(df.columns)}"
        )

    df = df.copy()
    df["Label"] = df["Label"].astype(int)
    df["prob"] = df["prob"].astype(float)

    # 只分析小模型预测为正类的样本
    positive_df = df[df["prob"] >= 0.5].copy()

    if positive_df.empty:
        raise ValueError("没有 prob >= 0.5 的样本，无法进行正类预测内部分析。")

    bins, labels = build_bins()

    positive_df["bucket"] = pd.cut(
        positive_df["prob"],
        bins=bins,
        labels=labels,
        right=False,          # 左闭右开 [a, b)
        include_lowest=True
    )

    rows = []
    total_pred_pos = len(positive_df)
    total_tp = int((positive_df["Label"] == 1).sum())
    total_fp = int((positive_df["Label"] == 0).sum())

    for label in labels:
        bucket_df = positive_df[positive_df["bucket"] == label]

        total_count = len(bucket_df)
        tp_count = int((bucket_df["Label"] == 1).sum())
        fp_count = int((bucket_df["Label"] == 0).sum())

        precision = tp_count / total_count if total_count > 0 else 0.0
        fp_ratio_in_all_fp = fp_count / total_fp if total_fp > 0 else 0.0
        tp_ratio_in_all_tp = tp_count / total_tp if total_tp > 0 else 0.0
        sample_ratio_in_pred_pos = total_count / total_pred_pos if total_pred_pos > 0 else 0.0

        rows.append({
            "bucket": label,
            "count": total_count,
            "tp_count": tp_count,
            "fp_count": fp_count,
            "precision": precision,
            "fp_ratio_in_all_fp": fp_ratio_in_all_fp,
            "tp_ratio_in_all_tp": tp_ratio_in_all_tp,
            "sample_ratio_in_pred_pos": sample_ratio_in_pred_pos,
        })

    result_df = pd.DataFrame(rows)

    # 一个简单的“送LLM收益”指标：
    # 假正例越多、precision越低，越值得送LLM
    result_df["llm_gain_score"] = result_df["fp_count"] * (1 - result_df["precision"])

    return result_df, total_pred_pos, total_tp, total_fp


def generate_recommendation(result_df, total_pred_pos, total_tp, total_fp):
    non_empty_df = result_df[result_df["count"] > 0].copy()

    if non_empty_df.empty:
        return "所有分桶都为空，无法生成建议。"

    max_fp_row = non_empty_df.loc[non_empty_df["fp_count"].idxmax()]
    min_precision_row = non_empty_df.loc[non_empty_df["precision"].idxmin()]
    best_gain_row = non_empty_df.loc[non_empty_df["llm_gain_score"].idxmax()]

    lines = []
    lines.append("=" * 70)
    lines.append("正类预测内部区间分析总结")
    lines.append("=" * 70)
    lines.append(f"预测为正类的样本总数: {total_pred_pos}")
    lines.append(f"其中真正例总数 TP: {total_tp}")
    lines.append(f"其中假正例总数 FP: {total_fp}")
    lines.append("")

    lines.append("1. 假正例最多的区间")
    lines.append(
        f"   bucket={max_fp_row['bucket']}, "
        f"count={int(max_fp_row['count'])}, "
        f"fp_count={int(max_fp_row['fp_count'])}, "
        f"tp_count={int(max_fp_row['tp_count'])}, "
        f"precision={max_fp_row['precision']:.4f}"
    )
    lines.append("")

    lines.append("2. Precision 最低的区间")
    lines.append(
        f"   bucket={min_precision_row['bucket']}, "
        f"count={int(min_precision_row['count'])}, "
        f"fp_count={int(min_precision_row['fp_count'])}, "
        f"tp_count={int(min_precision_row['tp_count'])}, "
        f"precision={min_precision_row['precision']:.4f}"
    )
    lines.append("")

    lines.append("3. 综合来看最值得送 LLM 的区间")
    lines.append(
        f"   bucket={best_gain_row['bucket']}, "
        f"count={int(best_gain_row['count'])}, "
        f"fp_count={int(best_gain_row['fp_count'])}, "
        f"tp_count={int(best_gain_row['tp_count'])}, "
        f"precision={best_gain_row['precision']:.4f}, "
        f"llm_gain_score={best_gain_row['llm_gain_score']:.4f}"
    )
    lines.append("")
    lines.append("解释：")
    lines.append("优先考虑 fp_count 高、precision 低、同时 tp_count 不为 0 的桶。")
    lines.append("这样的区间最适合作为 LLM 的二次复核区间，用于降低误报。")

    return "\n".join(lines)


def paint(result_df):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("未检测到 matplotlib，跳过绘图。")
        return

    plot_df = result_df.copy()

    x = np.arange(len(plot_df))
    width = 0.24

    plt.figure(figsize=(12, 6))

    # 左轴：样本数
    ax1 = plt.gca()
    bars1 = ax1.bar(x - width, plot_df["count"], width=width, label="Count")
    bars2 = ax1.bar(x, plot_df["tp_count"], width=width, label="TP Count")
    bars3 = ax1.bar(x + width, plot_df["fp_count"], width=width, label="FP Count")

    ax1.set_xlabel("Probability Bucket")
    ax1.set_ylabel("Sample Count")
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_df["bucket"], rotation=20)
    ax1.grid(True, axis="y", alpha=0.3)

    # 右轴：precision
    ax2 = ax1.twinx()
    line = ax2.plot(x, plot_df["precision"], marker="o", label="Precision")
    ax2.set_ylabel("Precision")
    ax2.set_ylim(0, 1.05)

    # 高亮最值得送LLM的桶
    if (plot_df["count"] > 0).any():
        best_idx = plot_df.loc[plot_df["llm_gain_score"].idxmax()].name
        ax1.axvspan(best_idx - 0.5, best_idx + 0.5, alpha=0.15)
        ax1.text(
            best_idx,
            max(plot_df["count"].max(), 1) * 0.95,
            "Best for LLM",
            ha="center",
            va="top"
        )

    # 合并图例
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.title("Positive Prediction Bucket Analysis")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    print(f"图像已保存到: {PLOT_PATH}")


def main():
    df = pd.read_csv(CSV_PATH)

    result_df, total_pred_pos, total_tp, total_fp = analyze_positive_buckets(df)
    result_df.to_csv(OUTPUT_BUCKET_PATH, index=False)

    recommendation = generate_recommendation(result_df, total_pred_pos, total_tp, total_fp)
    with open(OUTPUT_RECOMMEND_PATH, "w", encoding="utf-8") as f:
        f.write(recommendation)

    print("=" * 70)
    print("正类预测内部分析完成")
    print("=" * 70)
    print(result_df.to_string(index=False))
    print("=" * 70)
    print(recommendation)
    print("=" * 70)
    print(f"分桶统计结果已保存到: {OUTPUT_BUCKET_PATH}")
    print(f"文字建议已保存到: {OUTPUT_RECOMMEND_PATH}")

    paint(result_df)


if __name__ == "__main__":
    main()