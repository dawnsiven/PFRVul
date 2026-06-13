import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "results.csv"
OUTPUT_PATH = BASE_DIR / "threshold_analysis.csv"
PLOT_PATH = BASE_DIR / "threshold_metrics.png"


def fpr_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if fp + tn == 0:
        return 0.0
    return fp / (fp + tn)


def evaluate_at_threshold(y_true, prob, threshold):
    y_pred = (prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fpr_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "accuracy": acc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "fpr": fpr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "pred_pos_rate": y_pred.mean()
    }


def main():
    df = pd.read_csv(CSV_PATH)

    required_cols = {"Label", "prob"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV 必须包含列: {required_cols}，当前列为: {list(df.columns)}")

    y_true = df["Label"].astype(int).values
    prob = df["prob"].astype(float).values

    thresholds = np.arange(0.0, 1.0001, 0.01)

    results = []
    for th in thresholds:
        results.append(evaluate_at_threshold(y_true, prob, th))

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_PATH, index=False)

    best_f1_row = result_df.loc[result_df["f1"].idxmax()]
    best_recall_under_fpr = result_df[result_df["fpr"] <= 0.1]
    best_recall_row = None
    if not best_recall_under_fpr.empty:
        best_recall_row = best_recall_under_fpr.loc[best_recall_under_fpr["recall"].idxmax()]

    default_row = result_df[np.isclose(result_df["threshold"], 0.5)].iloc[0]

    print("=" * 60)
    print("默认阈值 0.5 的结果")
    print(default_row.to_string())
    print("=" * 60)

    print("F1 最优阈值")
    print(best_f1_row.to_string())
    print("=" * 60)

    if best_recall_row is not None:
        print("在 FPR <= 0.1 条件下 Recall 最优阈值")
        print(best_recall_row.to_string())
        print("=" * 60)
    else:
        print("没有找到满足 FPR <= 0.1 的阈值")
        print("=" * 60)

    print(f"完整阈值分析已保存到: {OUTPUT_PATH}")
def paint():    
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("未检测到 matplotlib，跳过绘图。")
        return

    df = pd.read_csv(OUTPUT_PATH)

    plt.figure(figsize=(8, 5))
    plt.plot(df["threshold"], df["recall"], label="Recall")
    plt.plot(df["threshold"], df["precision"], label="Precision")
    plt.plot(df["threshold"], df["f1"], label="F1")
    plt.plot(df["threshold"], df["fpr"], label="FPR")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Metrics under Different Thresholds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    print(f"指标曲线图已保存到: {PLOT_PATH}")


if __name__ == "__main__":
    # main()
    paint()
