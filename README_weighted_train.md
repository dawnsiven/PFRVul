# Weighted Train README

## 目标

这份文档说明当前仓库中新增的“基于误判样本加权训练”流程，主要覆盖以下内容：

- 为 `data/*_subsampled/alpaca/*.json` 数据增加 `weight` 字段
- 让 `CodeBERT_imp` 和 `UniXcoder_imp` 在训练、验证、测试时加载 `weight`
- 在不改变原始 loss 公式的前提下，对逐样本 loss 按 `weight` 做加权平均
- 根据 `outputs/CodeBERT_imbalance_test/.../results.csv` 中的误判样本，生成新的 weighted train 文件
- 让 `scripts/train_imbalance_test.sh` 默认优先使用 weighted train

## 当前已支持的模块

- `CodeBERT_imp/run.py`
- `CodeBERT_imp/model.py`
- `UniXcoder_imp/run.py`
- `UniXcoder_imp/model.py`
- `scripts/build_weighted_dataset.py`
- `scripts/train_imbalance_test.sh`

## 数据格式

训练、验证、测试数据现在支持可选字段：

```json
{
  "index": 104511,
  "input": "source code ...",
  "instruction": "",
  "output": "0",
  "weight": 1.0
}
```

说明：

- `weight` 缺失时，程序默认按 `1.0` 处理
- 当所有样本 `weight=1.0` 时，训练行为与原来一致

## 已生成的数据

当前已经补过默认权重的目录：

- `data/bigvul_cwe20_subsampled/alpaca`

并已生成一个示例 weighted train 文件：

- `data/bigvul_cwe20_subsampled/alpaca/bigvul_cwe20_1_1_train_weighted.json`

该文件是根据：

- `outputs/CodeBERT_imbalance_test/bigvul_cwe20_1_1/results.csv`

中的误判样本生成的。当前结果中共有 `298` 个误判样本，这些样本在 weighted train 文件中的 `weight` 被设为 `2.0`。

## loss 加权方式

### UniXcoder_imp

`UniXcoder_imp/model.py` 中原先使用的是分类 loss。现在改为：

1. 先计算逐样本 loss
2. 再按样本 `weight` 做归一化加权平均

等价形式：

```text
weighted_loss = sum(loss_i * weight_i) / sum(weight_i)
```

### CodeBERT_imp

`CodeBERT_imp/model.py` 中原先是手写二分类对数损失。现在保留原始公式，只改为：

1. 先计算逐样本 loss
2. 再按样本 `weight` 做归一化加权平均

所以：

- 原始 loss 公式没变
- 默认权重全 1 时结果不变

## 生成 weighted train 文件

使用脚本：

- `scripts/build_weighted_dataset.py`

默认命令：

```bash
python3 scripts/build_weighted_dataset.py
```

默认会做两件事：

1. 给 `data/bigvul_cwe20_subsampled/alpaca/bigvul_cwe20_*.json` 补 `weight=1.0`
2. 读取 `outputs/CodeBERT_imbalance_test/bigvul_cwe20_1_1/results.csv`，生成：
   `data/bigvul_cwe20_subsampled/alpaca/bigvul_cwe20_1_1_train_weighted.json`

常用参数：

```bash
python3 scripts/build_weighted_dataset.py \
  --dataset-dir data/bigvul_cwe20_subsampled/alpaca \
  --dataset-pattern "bigvul_cwe20_*.json" \
  --train-file data/bigvul_cwe20_subsampled/alpaca/bigvul_cwe20_1_1_train.json \
  --result-csv outputs/CodeBERT_imbalance_test/bigvul_cwe20_1_1/results.csv \
  --output-train-file data/bigvul_cwe20_subsampled/alpaca/bigvul_cwe20_1_1_train_weighted.json \
  --error-weight 2.0 \
  --update-mode set
```

参数说明：

- `--error-weight`
  误判样本的新权重，默认 `2.0`
- `--update-mode`
  支持 `set`、`add`、`multiply`
- `--weight-key`
  默认为 `weight`
- `--index-key`
  默认为 `index`

## 训练脚本默认行为

脚本：

- `scripts/train_imbalance_test.sh`

现在默认启用：

```bash
USE_WEIGHTED_TRAIN=1
```

逻辑如下：

1. 优先使用 `..._train_weighted.json`
2. 如果 weighted 文件不存在，但 `outputs/CodeBERT_imbalance_test/.../results.csv` 存在，则自动生成 weighted train 文件
3. 如果两者都不存在，则回退到原始 `..._train.json`

## 使用方式

### 训练 CodeBERT_imp

```bash
./scripts/train_imbalance_test.sh bigvul_cwe20 CodeBERT_imp 1 1 0
```

### 训练 UniXcoder_imp

```bash
./scripts/train_imbalance_test.sh bigvul_cwe20 UniXcoder_imp 1 1 0
```

### 显式关闭 weighted train

```bash
USE_WEIGHTED_TRAIN=0 ./scripts/train_imbalance_test.sh bigvul_cwe20 CodeBERT_imp 1 1 0
```

## 直接指定 weighted train 文件

如果你不想通过脚本自动选择，也可以直接在训练命令里手动指定：

```bash
--train_data_file data/bigvul_cwe20_subsampled/alpaca/bigvul_cwe20_1_1_train_weighted.json
```

## 注意事项

- 当前 weighted train 的映射依赖 `index` 字段一一对应
- `results.csv` 至少需要包含 `Index`、`Label`、`Prediction`
- `scripts/build_weighted_dataset.py` 对列名做了大小写兼容处理
- 当前 `train_imbalance_test.sh` 的 weighted 自动生成逻辑依赖 `CodeBERT_imbalance_test` 的结果文件
- 如果你换了数据集名、长度区间或正负样本比例，需要保证对应的 `results.csv` 路径存在

## 推荐流程

1. 先跑一轮 `CodeBERT_imbalance_test`，得到 `results.csv`
2. 用 `scripts/build_weighted_dataset.py` 生成 weighted train 文件
3. 用 `CodeBERT_imp` 或 `UniXcoder_imp` 训练 weighted 版本
4. 如有需要，通过调节 `--error-weight` 和 `--update-mode` 继续做对比实验

## 相关文件

- `scripts/build_weighted_dataset.py`
- `scripts/train_imbalance_test.sh`
- `CodeBERT_imp/run.py`
- `CodeBERT_imp/model.py`
- `UniXcoder_imp/run.py`
- `UniXcoder_imp/model.py`
