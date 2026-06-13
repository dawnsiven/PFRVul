# improve_recall

这个目录用于合并 `CodeBERT` 和 `UniXcoder` 在同一数据集上的预测结果，并统计合并后的指标，重点是验证“并集策略”是否能提升召回率。

## 功能

- 读取两个模型各自输出的 `results.csv`
- 按 `Index` 对齐同一条样本的预测结果
- 生成合并后的 `result.csv`
- 统计两个单模型和合并结果的分类指标
- 统计两个模型预测的重合度和互补情况

## 当前默认逻辑

默认使用 `union` 策略，也就是：

- 只要 `CodeBERT` 或 `UniXcoder` 有一个预测为阳性 `1`
- 那么最终结果 `Final_Prediction` 就记为阳性 `1`

对应关系如下：

- `1, 1 -> 1`
- `1, 0 -> 1`
- `0, 1 -> 1`
- `0, 0 -> 0`

这个策略适合做 recall 提升分析。

## 输入目录

默认读取下面两个目录中的结果文件：

- `outputs/CodeBERT_imbalance/<dataset>/results.csv`
- `outputs/UniXcoder_imbalance/<dataset>/results.csv`

例如数据集名是 `bigvul_cwe20_1_1`，那么读取的是：

- `outputs/CodeBERT_imbalance/bigvul_cwe20_1_1/results.csv`
- `outputs/UniXcoder_imbalance/bigvul_cwe20_1_1/results.csv`

## 输出目录

脚本会在下面目录生成结果：

- `improve_recall/output/<dataset>/result.csv`
- `improve_recall/output/<dataset>/summary.json`
- `improve_recall/output/<dataset>/summary.csv`

## 使用方法

下面命令都需要在项目根目录执行：

```bash
cd /home/zjr123/LLM4CVD-main
```

### 批量处理多个 BigVul CWE 数据集

如果你想一次性处理下面这几组数据：

- `bigvul_cwe119_1_1`
- `bigvul_cwe125_1_1`
- `bigvul_cwe200_1_1`
- `bigvul_cwe264_1_1`
- `bigvul_cwe399_1_1`

可以直接运行：

```bash
bash improve_recall/run_batch_bigvul_cwes.sh
```

脚本位置：

- `improve_recall/run_batch_bigvul_cwes.sh`

### 1. 合并两个模型结果

```bash
python3 improve_recall/merge_results.py --dataset bigvul_cwe20_1_1
```

可选参数：

- `--dataset`：数据集目录名，必填
- `--strategy`：合并策略，可选 `union`、`intersection`、`codebert`、`unixcoder`
- `--codebert-root`：CodeBERT 结果根目录
- `--unixcoder-root`：UniXcoder 结果根目录
- `--output-root`：输出根目录

示例：

```bash
python3 improve_recall/merge_results.py \
  --dataset bigvul_cwe20_1_1 \
  --strategy union
```

### 2. 统计指标

```bash
python3 improve_recall/summarize_results.py --dataset bigvul_cwe20_1_1
```

可选参数：

- `--dataset`：数据集目录名，必填
- `--output-root`：合并结果根目录

### 3. 汇总 output 目录下各类型的数值变化

如果你想统一统计 `improve_recall/output` 下所有数据集的指标变化，可以运行：

```bash
python3 improve_recall/summarize_output_changes.py
```

默认会扫描：

- `improve_recall/output/*/summary.json`

并生成：

- `improve_recall/output_summary/dataset_changes.csv`
- `improve_recall/output_summary/dataset_changes.json`
- `improve_recall/output_summary/overall_changes.json`

其中：

- `dataset_changes.csv`：每个数据集一行，包含单模型、合并结果和变化量
- `dataset_changes.json`：和上面同内容的 JSON 版本
- `overall_changes.json`：所有数据集的平均变化情况

## result.csv 字段说明

`result.csv` 中包含以下字段：

- `Index`：样本索引
- `CWE`：样本 CWE 信息
- `Label`：真实标签
- `CodeBERT_Prediction`：CodeBERT 预测标签
- `CodeBERT_Prob`：CodeBERT 预测为阳性的概率
- `UniXcoder_Prediction`：UniXcoder 预测标签
- `UniXcoder_Prob`：UniXcoder 预测为阳性的概率
- `Final_Prediction`：合并后的最终标签
- `Final_Prob`：合并后的概率值
- `Strategy`：当前使用的合并策略

## summary.json 内容说明

`summary.json` 主要包含四部分：

- `codebert`：CodeBERT 单模型指标
- `unixcoder`：UniXcoder 单模型指标
- `final`：合并后结果指标
- `overlap`：两个模型预测重合情况

### 基础分类指标

每个模型和最终结果都会统计：

- `count`
- `tp`
- `tn`
- `fp`
- `fn`
- `accuracy`
- `precision`
- `recall`
- `f1`
- `fpr`

### overlap 字段说明

- `same_prediction_count`：两个模型预测完全一致的样本数
- `same_prediction_ratio`：两个模型预测一致的比例
- `different_prediction_count`：两个模型预测不一致的样本数
- `different_prediction_ratio`：两个模型预测不一致的比例
- `both_positive_count`：两个模型都预测为阳性的样本数
- `both_negative_count`：两个模型都预测为阴性的样本数
- `codebert_only_positive_count`：只有 CodeBERT 预测为阳性的样本数
- `unixcoder_only_positive_count`：只有 UniXcoder 预测为阳性的样本数
- `positive_intersection_count`：两个模型阳性预测集合的交集大小
- `positive_union_count`：两个模型阳性预测集合的并集大小
- `positive_jaccard`：两个模型阳性预测集合的 Jaccard 重合度

另外还有两个辅助字段：

- `disagreement_count`：预测不一致的样本数
- `rescued_positives`：并集策略额外补回来的正样本数

## summary.csv 内容说明

`summary.csv` 是表格版摘要，适合后续做批量汇总、统计或画图。

目前包含：

- `count / tp / tn / fp / fn`
- `accuracy / precision / recall / f1 / fpr`
- `same_prediction_ratio`
- `different_prediction_ratio`
- `positive_jaccard`

## output_summary 内容说明

`summarize_output_changes.py` 会额外生成一个总汇总目录：

- `improve_recall/output_summary`

其中主要统计：

- 每个数据集的 `accuracy / precision / recall / f1 / fpr`
- `Final` 相比 `CodeBERT` 的变化量
- `Final` 相比 `UniXcoder` 的变化量
- `Final` 相比单模型最优值的变化量
- 各数据集的重合率、阳性重合度、补回的正样本数

## 一个完整示例

```bash
python3 improve_recall/merge_results.py --dataset bigvul_cwe20_1_1
python3 improve_recall/summarize_results.py --dataset bigvul_cwe20_1_1
```

运行完成后可查看：

- `improve_recall/output/bigvul_cwe20_1_1/result.csv`
- `improve_recall/output/bigvul_cwe20_1_1/summary.json`
- `improve_recall/output/bigvul_cwe20_1_1/summary.csv`

## 注意事项

- 两个模型的 `results.csv` 必须来自同一个数据集
- 两个文件中的 `Index` 必须能够一一对齐
- 同一个 `Index` 的 `Label` 必须一致，否则脚本会报错
- 如果你只想重新做统计，不需要再次合并，直接运行 `summarize_results.py` 即可
