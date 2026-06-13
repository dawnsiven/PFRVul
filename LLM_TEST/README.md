# LLM_TEST 使用说明

如果你要通过 FastAPI 调用这一部分的能力，而不是直接跑脚本，可以看单独的接口文档：[API_README.md](/home/zjr123/LLM4CVD-main/LLM_TEST/API_README.md)。

`LLM_TEST/` 是一个围绕“小模型预测结果复审”的独立工作区，核心流程是：

`extract_positive_samples.py` -> `llm_api_judge.py` / `llm_local_judge.py` -> `recompute_metrics.py`

如果你要做“整份 test 集直接用 LLM 检测”，现在也有一套独立补充流程：

`prepare_full_test_samples.py` -> `llm_api_judge.py` -> `evaluate_llm_full_dataset.py`

其中：

- `prepare_full_test_samples.py` 只负责把原始 `*_test.json` 转成 `full_test_samples.jsonl`
- `llm_api_judge.py` 负责实际调用 OpenAI 兼容接口
- `evaluate_llm_full_dataset.py` 负责对整份 test 集的 LLM 检测结果计算指标

目前这个目录既包含可执行脚本，也包含提示词、实验输出和临时结果。为了不影响现有脚本路径，我这次先按“职责分层”的方式整理使用规范，而不直接大规模移动脚本文件。

## 推荐组织方式

建议把 `LLM_TEST/` 按下面的职责来理解和维护：

```text
LLM_TEST/
├── README.md
├── .env.example                  # 环境变量模板
├── .gitignore                    # 忽略中间结果和输出
├── exp.yaml                      # 配置模板，读取环境变量
├── config_utils.py               # 配置加载与变量展开
├── extract_positive_samples.py   # 第 1 步：提取小模型判为正样本的记录
├── llm_api_judge.py              # 第 2 步-A：调用 API 模型复审
├── llm_local_judge.py            # 第 2 步-B：调用本地 HF/Ollama 兼容模型复审
├── recompute_metrics.py          # 第 3 步：把 LLM 结果合回去并重算指标
├── recompute_metrics_with_confidence_gate.py
├── ensemble_vote.py              # 多个 LLM 结果做投票
├── run_review.sh                 # API 复审的一键入口
├── run_ensemble_vote.sh          # 投票实验的一键入口
├── plot_reviewed_subset_metrics.py
├── test_baidu_api.py             # 本地接口连通性测试脚本
├── Prompt/                       # 提示词版本
├── intermediate/                 # 第 1 步输出的中间数据
└── output/                       # 第 2/3 步及投票实验输出
```

如果后续你准备继续长期维护这个目录，建议演进到下面这种结构：

```text
LLM_TEST/
├── configs/
├── prompts/
├── runners/
├── analysis/
├── intermediate/
└── output/
```

但当前脚本里已经写死了不少 `LLM_TEST/Prompt/...`、`LLM_TEST/output/...` 路径，所以现阶段更稳妥的做法是：

- 保留现有文件位置，避免一次性改动太多脚本路径
- 用 `README + .env.example + .gitignore` 先把“怎么放、怎么跑、哪些是产物”讲清楚
- 等流程稳定后，再做真正的目录迁移

## 目录职责

### 1. 配置层

- `exp.yaml`
  只保留参数名和 `${ENV_VAR}` 占位符，不直接写死实验值。
- `.env`
  保存当前实验的真实参数，例如数据集、结果文件、提示词、模型名、API 地址。
- `.env.example`
  作为模板，复制一份再改成本次实验的 `.env`。

推荐做法：

```bash
cp LLM_TEST/.env.example LLM_TEST/.env
```

### 2. 提取层

- `extract_positive_samples.py`
  从原始 `results.csv` 中抽出 `Prediction == 1` 的记录，并和原始测试集样本对齐。

输出目录：

- `LLM_TEST/intermediate/<dataset_tag>/summary.json`
- `LLM_TEST/intermediate/<dataset_tag>/positive_indices.json`
- `LLM_TEST/intermediate/<dataset_tag>/positive_samples.jsonl`
- `LLM_TEST/intermediate/<dataset_tag>/unmatched_indices.json`

### 3. 复审层

- `llm_api_judge.py`
  调用 OpenAI 兼容接口复审。
- `llm_api_judge_with_examples.py`
  调用 OpenAI 兼容接口复审，并给每个待判断样本注入一个“相似历史误判案例”。
- `llm_local_judge.py`
  用本地 Hugging Face 模型复审。
- `Prompt/`
  存放提示词版本，建议一个版本一个文件。

输出目录：

- `LLM_TEST/output/<实验目录>/llm_judgments.jsonl`
- `LLM_TEST/output/<实验目录>/llm_predictions.csv`
- `LLM_TEST/output/<实验目录>/llm_summary.json`

### 4. 合并评估层

- `recompute_metrics.py`
  把 `llm_predictions.csv` 合回原始 `results.csv`，生成最终指标。
- `recompute_metrics_with_confidence_gate.py`
  做“高置信度保留小模型，低置信度交给 LLM”的策略实验。
- `ensemble_vote.py`
  对多个 `llm_predictions.csv` 做投票。
- `plot_reviewed_subset_metrics.py`
  把 `metrics.json` 画成 SVG 图。

### 5. 入口脚本

- `run_review.sh`
  适合单模型 API 复审。
- `run_ensemble_vote.sh`
  适合复用已有多个模型结果做投票。
- `test_baidu_api.py`
  实际上更像本地模型接口探活脚本，不只是 Baidu。

## 推荐命名方式

当前目录里同时存在这些命名风格：

- `bigvul_cwe119_1_1`
- `bigvul_1_llama3.2_positive_0.5_0.9`
- `bigvul_cwe119_1_1_CWE-119_0.5`

这会导致后面很难看出“数据集 / 小模型 / 提示词 / 复审模型”分别是什么。建议统一成：

```text
<dataset>_<small_model>_<review_model>_<prompt_version>
```

例如：

```text
bigvul_llama3.2_deepseek-r1_CWE-119_0.6
```

如果还要体现提取条件，可以补在中间目录名里：

```text
bigvul_llama3.2_positive_prob_0.5_0.9
```

## 推荐使用方式

### 一键流程

如果你已经准备好 `.env` 和 `exp.yaml`，最简单的做法是直接跑：

```bash
bash LLM_TEST/run_review.sh
```

或者手动指定：

```bash
bash LLM_TEST/run_review.sh LLM_TEST/exp.yaml LLM_TEST/.env 100
```

这里的 `100` 表示只复审前 `100` 条样本。

如果你不想再手动修改一堆 `.env` 字段，现在也可以直接用自动模式，只给这 3 个核心输入：

```bash
bash LLM_TEST/run_review.sh \
  --result-model llama3.2_lora_imbalance \
  --dataset bigvul_1 \
  --prob-range 0.5-0.9
```

自动模式会根据你的输入自动推导：

- `RESULTS_CSV=outputs/<result-model>/<dataset>/results.csv`
- `DATA_JSON`：自动从 `data/` 下推断对应的 `*_test.json`
- `INPUT_JSON=LLM_TEST/intermediate/<dataset>_<result-model>_positive_<min>_<max>/positive_samples.jsonl`
- `LLM 输出目录=LLM_TEST/output/<review_root>/<dataset>_<result-model>_<review_model>_<prompt_version>/`

这时 `.env` 主要只需要保留“复审模型配置”，例如：

- `LLM_MODEL`
- `API_BASE`
- `API_KEY_ENV` / `OPENAI_API_KEY`
- `PROMPT_FILE`
- `OUTPUT_ROOT`

### 分步流程

### 第 1 步：提取小模型判正的样本

```bash
python3 LLM_TEST/extract_positive_samples.py \
  --config LLM_TEST/exp.yaml \
  --env_file LLM_TEST/.env
```

如果你要做概率筛选，也可以直接加：

```bash
python3 LLM_TEST/extract_positive_samples.py \
  --config LLM_TEST/exp.yaml \
  --env_file LLM_TEST/.env \
  --min_prob 0.5 \
  --max_prob 0.9
```

### 第 2 步：用 API 模型复审

```bash
python3 LLM_TEST/llm_api_judge.py \
  --config LLM_TEST/exp.yaml \
  --env_file LLM_TEST/.env \
  --limit 100 \
  --output_by_prompt_version
```

如果你想用 `LLM_TEST/Prompt/CWE-20_1.txt` 这种“目标代码 + 相似历史案例”提示词，可以运行：

```bash
python3 LLM_TEST/llm_api_judge_with_examples.py \
  --config LLM_TEST/exp.yaml \
  --env_file LLM_TEST/.env \
  --input_json LLM_TEST/intermediate/bigvul_cwe20_1_1_UniXcoder_imbalance_positive_0_1/positive_samples.jsonl \
  --prompt_file LLM_TEST/Prompt/CWE-20_1.txt \
  --example_records_json error_similarity_analysis/cwe20_bundle/bigvul_cwe20_1_1/codebert_errors.json \
  --example_similarity_csv error_similarity_analysis/cwe20_bundle/bigvul_cwe20_1_1/similarity/codebert_errors_mean_top10.csv \
  --output_by_prompt_version
```

适用场景：

- DeepSeek/OpenAI 兼容接口
- 本地 `Ollama` 暴露的 `v1/chat/completions`

### 第 2 步：用本地模型复审

```bash
python3 LLM_TEST/llm_local_judge.py \
  --input_json LLM_TEST/intermediate/<dataset_tag>/positive_samples.jsonl \
  --prompt_file LLM_TEST/Prompt/CWE-119.txt \
  --model llama3.2 \
  --output_by_prompt_version
```

`--model` 支持：

- 别名：`llama2`、`codellama`、`llama3`、`llama3.1`、`llama3.2`
- 本地目录
- Hugging Face repo id

### 第 3 步：重算指标

```bash
python3 LLM_TEST/recompute_metrics.py \
  --config LLM_TEST/exp.yaml \
  --env_file LLM_TEST/.env
```

如果你要手动指定复审结果：

```bash
python3 LLM_TEST/recompute_metrics.py \
  --results_csv outputs/<run>/results.csv \
  --llm_predictions_csv LLM_TEST/output/<run>/llm_predictions.csv \
  --output_dir LLM_TEST/output/<run>
```

### 多模型投票

### 直接运行预设

```bash
bash LLM_TEST/run_ensemble_vote.sh
```

或者：

```bash
bash LLM_TEST/run_ensemble_vote.sh bigvul_cwe119_1_1_CWE-119_0.5 majority3
```

### 手动指定输入

```bash
python3 LLM_TEST/ensemble_vote.py \
  --inputs \
  deepseek=LLM_TEST/output/deepseek/<tag>/llm_predictions.csv \
  baidu=LLM_TEST/output/baidu/<tag>/llm_predictions.csv \
  llama=LLM_TEST/output/llama3.2:3b/<tag>/llm_predictions.csv \
  --strategy majority \
  --intersection_only \
  --output_dir LLM_TEST/output/ensemble_<tag>
```

## 常见输出物

### `intermediate/`

- `summary.json`：提取统计信息
- `positive_indices.json`：被抽中的样本索引
- `positive_samples.jsonl`：后续给 LLM 的输入
- `unmatched_indices.json`：没对齐上的索引

### `output/`

- `llm_predictions.csv`：复审后的标签
- `merged_results.csv`：合并后的最终结果
- `metrics.json`：最终指标
- `*_predictions.csv` / `*_metrics.json`：投票实验结果

## 当前目录里建议保留和建议弱化的文件

建议长期保留：

- `config_utils.py`
- `extract_positive_samples.py`
- `llm_api_judge.py`
- `llm_local_judge.py`
- `recompute_metrics.py`
- `ensemble_vote.py`
- `run_review.sh`
- `run_ensemble_vote.sh`
- `Prompt/`
- `README.md`
- `.env.example`

建议归为辅助工具：

- `recompute_metrics_with_confidence_gate.py`
- `plot_reviewed_subset_metrics.py`
- `test_baidu_api.py`

建议视情况清理或不提交版本库：

- `__pycache__/`
- `intermediate/`
- `output/`
- 各种临时 `.env.*`

## 一个稳妥的日常工作流

每次新实验只做下面几件事：

1. 复制一个新的 `.env` 模板，改成本次实验参数。
2. 先跑 `extract_positive_samples.py`，确认 `positive_samples.jsonl` 正常。
3. 再跑 `llm_api_judge.py` 或 `llm_local_judge.py`。
4. 跑 `recompute_metrics.py` 生成最终指标。
5. 如果是对比实验，再跑 `ensemble_vote.py`。

这样目录会始终保持成：

- `Prompt/` 放提示词版本
- `intermediate/` 放抽样结果
- `output/` 放复审和指标结果
- 根目录只放脚本、配置和说明文档

这就是目前最适合这个仓库状态的整理方式：不破坏已有脚本路径，但把文件角色、命名规则和使用流程固定下来。
