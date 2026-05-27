# FastAPI Backend

这个目录提供了一个面向前端的 FastAPI 后端，用来把仓库原本的 `scripts/*.sh` 能力暴露成 HTTP 接口。  
后端本身不直接重写训练逻辑，而是继续调用现有脚本，例如 `train.sh`、`test.sh`、`finetune.sh`、`inference.sh` 等。

除了训练和推理接口，现在也包含了 `LLM_TEST/` 复审工作流的接口封装。  
`LLM_TEST` 的专门接口说明见：[LLM_TEST/API_README.md](/home/zjr123/LLM4CVD-main/LLM_TEST/API_README.md)。

## 目录说明

- `app.py`: FastAPI 入口，定义所有路由
- `schemas.py`: 请求体和响应体的数据结构
- `job_runner.py`: 后台任务启动、状态维护、终止逻辑

## 启动方式

在仓库根目录执行：

```bash
pip install -r requirements.txt
./scripts/run_fastapi.sh 0.0.0.0 8000
```

启动后可访问：

- `http://127.0.0.1:8000/docs`：Swagger 文档页面
- `http://127.0.0.1:8000/health`：健康检查

如果前端本地开发端口需要跨域访问，可以配置：

```bash
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:3000 ./scripts/run_fastapi.sh
```

## 设计思路

所有训练、测试、推理任务都按“异步作业”处理：

1. 前端调用某个 `POST` 接口创建任务
2. 后端立即返回一个 `job_id`
3. 前端轮询 `GET /api/jobs/{job_id}` 查看状态
4. 前端按需调用 `GET /api/jobs/{job_id}/log` 获取日志尾部内容

任务状态一般有这些值：

- `queued`: 已创建，尚未启动
- `running`: 正在运行
- `completed`: 成功完成
- `failed`: 执行失败
- `stopped`: 被手动停止

任务元数据会持久化到本地 SQLite：

- 数据库位置：`fastapi_backend/state/jobs.db`
- FastAPI 重启后，历史任务仍可通过 `job_id` 查询
- 日志、模型输出、`results.csv` 仍然保存在原有文件路径，不写入数据库
- 重启恢复时，如果旧任务进程还活着，会继续显示为 `running`
- 如果旧任务进程已经不存在，后端会结合 `results.csv` 是否存在来把状态修正为 `completed` 或 `failed`

## 通用返回结构

大部分创建任务接口和任务详情接口返回结构一致，核心字段如下：

```json
{
  "job_id": "0d9d7b8d0f9b4b2fa6f9470f4aaf1234",
  "job_type": "classical",
  "script_name": "train.sh",
  "status": "running",
  "command": [
    "bash",
    "/home/zjr123/LLM4CVD-main/scripts/train.sh",
    "reveal",
    "ReGVD",
    "0-512",
    "0"
  ],
  "pid": 12345,
  "created_at": "2026-04-05T08:00:00.000000",
  "started_at": "2026-04-05T08:00:00.100000",
  "finished_at": null,
  "return_code": null,
  "output_dir": "/home/zjr123/LLM4CVD-main/outputs/ReGVD/reveal_0-512",
  "log_file": "/home/zjr123/LLM4CVD-main/outputs/ReGVD/reveal_0-512/train_ReGVD_reveal_0-512.log",
  "result_csv": "/home/zjr123/LLM4CVD-main/outputs/ReGVD/reveal_0-512/results.csv",
  "params": {
    "action": "train",
    "dataset_name": "reveal",
    "model_name": "ReGVD",
    "length": "0-512",
    "cuda": "0"
  },
  "error_message": null
}
```

字段说明：

- `job_id`: 任务唯一 ID
- `job_type`: 任务分类
- `script_name`: 实际调用的脚本
- `status`: 当前状态
- `command`: 最终执行的命令
- `pid`: 任务进程号
- `output_dir`: 输出目录
- `log_file`: 日志文件路径
- `result_csv`: 结果 CSV 路径
- `params`: 创建任务时提交的参数
- `error_message`: 启动失败时的错误信息

## 接口总览

### 1. `GET /health`

用途：检查服务是否存活。

响应示例：

```json
{
  "status": "ok"
}
```

### 2. `GET /api/meta/options`

用途：给前端表单提供候选项，例如模型名、数据集目录、长度区间、类别不平衡比例。

返回内容包括：

- `graph_models`
- `llm_models`
- `datasets.regular`
- `datasets.imbalance`
- `regular_lengths`
- `imbalance_lengths`
- `imbalance_ratios`

请求示例：

```bash
curl http://127.0.0.1:8000/api/meta/options
```

### 2.1 `GET /api/outputs`

用途：浏览 `outputs/` 目录下的文件和子目录，适合前端做文件树或结果面板。

查询参数：

- `relative_path`: 相对于 `outputs/` 的路径，默认空字符串，表示根目录

调用示例：

```bash
curl "http://127.0.0.1:8000/api/outputs"
curl "http://127.0.0.1:8000/api/outputs?relative_path=UniXcoder_imbalance"
```

返回示例：

```json
{
  "base_dir": "/home/zjr123/LLM4CVD-main/outputs",
  "relative_path": "UniXcoder_imbalance",
  "entries": [
    {
      "name": "bigvul_cwe119_1_1",
      "relative_path": "UniXcoder_imbalance/bigvul_cwe119_1_1",
      "entry_type": "directory",
      "size": null
    }
  ]
}
```

### 2.2 `GET /api/outputs/text`

用途：读取 `outputs/` 中的文本文件内容，适合前端显示 `.log`、`.csv`、`.py` 等文本。

查询参数：

- `relative_path`: 相对于 `outputs/` 的文件路径
- `max_chars`: 最多返回多少字符，默认 `200000`

调用示例：

```bash
curl "http://127.0.0.1:8000/api/outputs/text?relative_path=UniXcoder_imbalance/bigvul_cwe119_1_1/results.csv"
```

### 2.3 `GET /api/outputs/file`

用途：直接返回 `outputs/` 中的文件，适合前端加载图片、下载 CSV、打开日志原文件。

查询参数：

- `relative_path`: 相对于 `outputs/` 的文件路径

调用示例：

```bash
curl -O "http://127.0.0.1:8000/api/outputs/file?relative_path=UniXcoder_imbalance/bigvul_cwe119_1_1/results.csv"
```

图片场景下，前端可以直接把这个接口当作 `img` 的 `src`：

```html
<img src="http://127.0.0.1:8000/api/outputs/file?relative_path=UniXcoder_imbalance/bigvul_cwe119_1_1/true_distribution.png" />
```

### 2.4 `POST /api/frontend/code-inference`

用途：给前端直接提交一段待检测代码，调用已经训练好的小模型做单样本推理。  
这个接口不会要求前端提供 `output` 真值标签，因为真实用户提交的代码本来就未知是否有漏洞。

当前支持模型：

- `CodeBERT`
- `UniXcoder`

接口行为：

- 后端会在 `data/temp_inference/<uuid>/` 下创建临时目录
- 将本次输入保存为 `input.json`
- 写入 `metadata.json`
- 将预测结果保存为 `prediction.json`
- 同时把预测结果直接返回给前端

请求体：

```json
{
  "model_name": "CodeBERT",
  "checkpoint_dir": "outputs/CodeBERT/cvefixes_cwe20_0-512",
  "code": "int main() { char buf[8]; gets(buf); return 0; }",
  "instruction": "Detect whether the following code contains vulnerabilities.",
  "block_size": 512,
  "sample_index": 0,
  "device": "auto"
}
```

字段说明：

- `model_name`: 当前支持 `CodeBERT` 或 `UniXcoder`
- `checkpoint_dir`: 已训练模型的输出目录，不是 `model.bin` 文件本身，例如 `outputs/CodeBERT/cvefixes_cwe20_0-512`
- `code`: 前端提交的待检测代码
- `instruction`: 可选，自定义任务提示词
- `block_size`: 最大 token 长度，默认 `512`
- `sample_index`: 可选，给单条样本一个索引，默认 `0`
- `device`: `auto`、`cpu`、`cuda`

成功响应示例：

```json
{
  "model_name": "CodeBERT",
  "checkpoint_dir": "/home/zjr123/LLM4CVD-main/outputs/CodeBERT/cvefixes_cwe20_0-512",
  "checkpoint_file": "/home/zjr123/LLM4CVD-main/outputs/CodeBERT/cvefixes_cwe20_0-512/checkpoint-best-f1/model.bin",
  "device": "cpu",
  "prediction": 1,
  "is_vulnerable": true,
  "vulnerability_probability": 0.6056362390518188,
  "non_vulnerable_probability": 0.39436376094818115,
  "temp_dir": "/home/zjr123/LLM4CVD-main/data/temp_inference/4d8961084c034464bdb2d36613831a29",
  "input_json": "/home/zjr123/LLM4CVD-main/data/temp_inference/4d8961084c034464bdb2d36613831a29/input.json"
}
```

返回字段说明：

- `prediction`: `0` 表示判定为无漏洞，`1` 表示判定为有漏洞
- `is_vulnerable`: `prediction == 1` 的布尔形式
- `vulnerability_probability`: 判定为漏洞的概率分数
- `non_vulnerable_probability`: 判定为无漏洞的概率分数
- `temp_dir`: 本次请求生成的临时目录
- `input_json`: 本次请求保存下来的输入文件路径

调用示例：

```bash
curl -X POST http://127.0.0.1:8000/api/frontend/code-inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "CodeBERT",
    "checkpoint_dir": "outputs/CodeBERT/cvefixes_cwe20_0-512",
    "code": "int main() { char buf[8]; gets(buf); return 0; }",
    "block_size": 512,
    "device": "cpu"
  }'
```

前端 `fetch` 示例：

```js
const response = await fetch("http://127.0.0.1:8000/api/frontend/code-inference", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    model_name: "CodeBERT",
    checkpoint_dir: "outputs/CodeBERT/cvefixes_cwe20_0-512",
    code: codeInput,
    block_size: 512,
    device: "auto",
  }),
});

if (!response.ok) {
  const error = await response.json();
  throw new Error(error.detail || "Inference request failed");
}

const result = await response.json();
console.log(result.prediction, result.vulnerability_probability);
```

适合的前端展示方式：

- 用 `is_vulnerable` 显示“检测到漏洞 / 未检测到漏洞”
- 用 `vulnerability_probability` 画进度条或置信度标签
- 用 `temp_dir` 做问题排查或回溯

注意：

- 这个接口是同步返回，不走 `job_id` 轮询模式
- 它适合单样本或少量即时检测，不适合大批量离线任务
- 如果 `checkpoint_dir/checkpoint-best-f1/model.bin` 不存在，请求会返回 `404`
- 如果请求 `cuda` 但当前机器没有可用 GPU，请求会返回 `400`

### 2.5 `POST /api/frontend/code-inference-file`

用途：直接读取仓库内一个 JSON 文件中的代码样本或切分结果，批量调用小模型做检测。  
这个接口适合你现在这种场景，例如直接检测：
`data/user_demo_chunks/user_demo_chunks_0-512_test.json`

接口行为：

- 支持读取 `/api/frontend/code-chunking` 保存下来的结果文件
- 也兼容读取简单的 JSON 数组，只要每项里有 `input`、`code` 或 `text`
- 检测结果会保存到 `outputs/<输入文件所在目录名>/` 下
- 同步返回汇总信息和前若干条预览结果

请求体示例：

```json
{
  "model_name": "CodeBERT",
  "checkpoint_dir": "outputs/CodeBERT/cvefixes_cwe20_0-512",
  "input_json": "data/user_demo_chunks/user_demo_chunks_0-512_test.json",
  "block_size": 512,
  "device": "auto",
  "preview_limit": 20
}
```

成功响应示例：

```json
{
  "model_name": "CodeBERT",
  "checkpoint_dir": "/home/zjr123/LLM4CVD-main/outputs/CodeBERT/cvefixes_cwe20_0-512",
  "checkpoint_file": "/home/zjr123/LLM4CVD-main/outputs/CodeBERT/cvefixes_cwe20_0-512/checkpoint-best-f1/model.bin",
  "device": "cpu",
  "input_json": "/home/zjr123/LLM4CVD-main/data/user_demo_chunks/user_demo_chunks_0-512_test.json",
  "result_json": "/home/zjr123/LLM4CVD-main/outputs/user_demo_chunks/user_demo_chunks_0-512_test_CodeBERT_frontend_inference.json",
  "result_csv": "/home/zjr123/LLM4CVD-main/outputs/user_demo_chunks/user_demo_chunks_0-512_test_CodeBERT_frontend_inference.csv",
  "total_samples": 128,
  "vulnerable_samples": 17,
  "preview": [
    {
      "index": 0,
      "sample_id": "demo-1",
      "filename": "main.c",
      "chunk_index": 0,
      "vulnerability_probability": 0.82,
      "code": "char buf[8]; gets(buf);"
    }
  ]
}
```

落盘文件说明：

- `outputs/<输入文件所在目录名>/<input_stem>_<model_name>_frontend_inference.json`: 完整推理结果
- `outputs/<输入文件所在目录名>/<input_stem>_<model_name>_frontend_inference.csv`: 便于和现有分析脚本衔接的表格结果

调用示例：

```bash
curl -X POST http://127.0.0.1:8000/api/frontend/code-inference-file \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "CodeBERT",
    "checkpoint_dir": "outputs/CodeBERT/cvefixes_cwe20_0-512",
    "input_json": "data/user_demo_chunks/user_demo_chunks_0-512_test.json",
    "block_size": 512,
    "device": "cpu"
  }'
```

### 2.6 `GET /api/meta/frontend-input-json-options`

用途：给前端提供 `input_json` 下拉候选，只扫描：

- `data/` 下一级目录名以 `user_` 开头的目录
- 这些目录里的 `.json` 文件

返回的 `path` 是仓库相对路径，前端可直接回填到
`POST /api/frontend/code-inference-file` 的 `input_json` 字段。

成功响应示例：

```json
{
  "items": [
    {
      "label": "user_demo_chunks_0-512_test.json",
      "path": "data/user_demo_chunks/user_demo_chunks_0-512_test.json"
    },
    {
      "label": "user_xxx_0-512_test.json",
      "path": "data/user_xxx/user_xxx_0-512_test.json"
    }
  ]
}
```

调用示例：

```bash
curl http://127.0.0.1:8000/api/meta/frontend-input-json-options
```

### 2.7 `GET /api/frontend/user-data`

用途：只浏览 `data/` 下 `user_` 前缀目录及其内容。  
适合前端拿切分结果、原始请求文件、示例文件，或者为 `chunk-review` 回取 code 内容。

接口行为：

- 不传 `relative_path` 时：返回 `data/` 下所有 `user_` 前缀目录
- 传 `relative_path=user_xxx` 时：返回该目录下的文件列表
- 只允许访问 `data/user_*` 作用域，不会越界到其他目录

根目录示例：

```bash
curl "http://127.0.0.1:8000/api/frontend/user-data"
```

查看某个用户目录示例：

```bash
curl "http://127.0.0.1:8000/api/frontend/user-data?relative_path=user_demo_chunks"
```

### 2.8 `GET /api/frontend/user-data/text`

用途：读取 `data/user_*` 目录下某个文本文件内容。

调用示例：

```bash
curl "http://127.0.0.1:8000/api/frontend/user-data/text?relative_path=user_demo_chunks/chunking_request.json"
```

### 2.9 `GET /api/frontend/user-outputs`

用途：只浏览 `outputs/` 下 `user_` 前缀目录及其内容。  
适合前端查看文件批量推理结果、后续汇总文件等。

接口行为：

- 不传 `relative_path` 时：返回 `outputs/` 下所有 `user_` 前缀目录
- 传 `relative_path=user_xxx` 时：返回该目录下的文件列表
- 只允许访问 `outputs/user_*` 作用域

调用示例：

```bash
curl "http://127.0.0.1:8000/api/frontend/user-outputs"
curl "http://127.0.0.1:8000/api/frontend/user-outputs?relative_path=user_demo_chunks"
```

### 2.10 `GET /api/frontend/user-outputs/text`

用途：读取 `outputs/user_*` 目录下某个文本文件内容。

调用示例：

```bash
curl "http://127.0.0.1:8000/api/frontend/user-outputs/text?relative_path=user_demo_chunks/user_demo_chunks_0-512_test_CodeBERT_frontend_inference.json"
```

### 2.11 `POST /api/frontend/code-chunking`

用途：给前端直接提交单段代码或多段代码，返回纯分割结果。  
这个接口只做代码分块，不做前后版本对比，也不会筛选“漏洞位置”相关片段。

接口行为：

- 对支持 ASTChunk 的语言优先走 AST 分割
- AST 分割失败或语言不支持时，退回固定行数分割
- 如果某个 chunk 超过 `max_tokens`，会继续向下拆分
- 前端需要额外传入一个 `folder_name`
- 后端会在 `data/user_<folder_name>/` 下保存切分请求、完整切分结果、前 5 个示例
- 同步返回全部 chunk 结果，不走 `job_id`

单代码请求体示例：

```json
{
  "code": "int main() {\n  char buf[8];\n  gets(buf);\n  return 0;\n}",
  "folder_name": "demo_chunks",
  "language": "c",
  "filename": "main.c",
  "sample_id": "demo-1",
  "max_chars": 1800,
  "max_tokens": 512,
  "fallback_lines": 80
}
```

批量请求体示例：

```json
{
  "folder_name": "demo_chunks",
  "items": [
    {
      "sample_id": "sample-1",
      "filename": "a.c",
      "language": "c",
      "code": "int main() { return 0; }"
    },
    {
      "sample_id": "sample-2",
      "filename": "b.py",
      "language": "python",
      "code": "def run():\n    print('hello')"
    }
  ],
  "max_chars": 1800,
  "max_tokens": 512,
  "fallback_lines": 80
}
```

成功响应示例：

```json
{
  "folder_name": "user_demo_chunks",
  "storage_dir": "/home/zjr123/LLM4CVD-main/data/user_demo_chunks",
  "total_inputs": 1,
  "total_chunks": 1,
  "chunks": [
    {
      "sample_id": "demo-1",
      "filename": "main.c",
      "chunk_index": 0,
      "text": "int main() {\n  char buf[8];\n  gets(buf);\n  return 0;\n}",
      "start_line": 1,
      "end_line": 5,
      "token_count": 22
    }
  ],
  "examples": [
    {
      "sample_id": "demo-1",
      "filename": "main.c",
      "chunk_index": 0,
      "text": "int main() {\n  char buf[8];\n  gets(buf);\n  return 0;\n}",
      "start_line": 1,
      "end_line": 5,
      "token_count": 22
    }
  ],
  "results": [
    {
      "sample_id": "demo-1",
      "filename": "main.c",
      "language": "c",
      "normalized_language": "cpp",
      "chunk_source": "fallback:cpp",
      "max_chars": 1800,
      "max_tokens": 512,
      "fallback_lines": 80,
      "tokenizer_model_path": "/home/zjr123/LLM4CVD-main/model/Llama-3.2-1B",
      "chunk_count": 1,
      "chunks": [
        {
          "index": 0,
          "text": "int main() {\n  char buf[8];\n  gets(buf);\n  return 0;\n}",
          "start_line": 1,
          "end_line": 5,
          "token_count": 22
        }
      ]
    }
  ]
}
```

返回字段说明：

- `folder_name`: 实际保存使用的目录名，格式固定为 `user_<folder_name>`
- `storage_dir`: 切分结果保存目录
- `total_inputs`: 本次处理的代码样本数
- `total_chunks`: 全部样本切出来的 chunk 总数
- `chunks`: 顶层平铺后的 chunk 列表，方便前端直接渲染单次结果或批量总览
- `examples`: 返回前 5 个 chunk 示例，方便前端做切分结果预览
- `chunk_source`: 实际使用的分割方式，例如 `ast:python`、`fallback:cpp`
- `normalized_language`: 规范化后的语言名
- `chunks[].start_line` / `chunks[].end_line`: 该 chunk 对应的原始代码行号范围
- `chunks[].token_count`: 该 chunk 的 token 数

落盘文件说明：

- `data/user_<folder_name>/chunking_request.json`: 前端原始请求
- `data/user_<folder_name>/user_<folder_name>_0-<max_tokens>_test.json`: 完整切分结果
- `data/user_<folder_name>/chunking_examples.json`: 前 5 个示例

适合的前端使用方式：

- 单代码输入框调用这个接口，直接展示“分割预览”
- 多文件上传后批量调用这个接口，按 `sample_id` 或 `filename` 展示结果
- 用户点某个 chunk 时，根据 `start_line` / `end_line` 高亮原代码对应范围

调用示例：

```bash
curl -X POST http://127.0.0.1:8000/api/frontend/code-chunking \
  -H "Content-Type: application/json" \
  -d '{
    "code": "int main() {\n  char buf[8];\n  gets(buf);\n  return 0;\n}",
    "folder_name": "demo_chunks",
    "language": "c",
    "filename": "main.c"
  }'
```

### 2.12 `POST /api/frontend/chunk-review`

用途：给前端直接提交“完整代码 + 小模型判为可疑的 chunk”，让大模型在完整语境下复核这些 chunk 是否真的存在漏洞。  
这是一个全新的同步接口，不会影响现有 `LLM_TEST/llm_api_judge.py` 和 `/api/jobs/llm-test/judge` 的旧流程。

接口行为：

- 前端传入完整代码 `code`
- 前端传入一个或多个小模型判为可疑的 `chunks`
- 后端会对每个 chunk 单独调用一次大模型
- 每次提示词里只包含两段代码：`1. 当前 chunk`，`2. 该 chunk 所属的完整原代码`
- 最终返回整体漏洞判断，以及逐 chunk 的复核结果

请求体示例：

```json
{
  "code": "int main() {\n  char buf[8];\n  gets(buf);\n  return 0;\n}",
  "language": "c",
  "prompt_file": "LLM_TEST/Prompt/CWE-119_0.5.txt",
  "model": "deepseek-chat",
  "chunks": [
    {
      "chunk_id": "chunk-1",
      "text": "char buf[8];\ngets(buf);",
      "start_line": 2,
      "end_line": 3,
      "prediction": 1,
      "vulnerability_probability": 0.91
    }
  ],
  "temperature": 0.0,
  "max_tokens": 512,
  "timeout": 120
}
```

成功响应示例：

```json
{
  "model": "deepseek-chat",
  "prompt_file": "/home/zjr123/LLM4CVD-main/LLM_TEST/Prompt/CWE-119_0.5.txt",
  "prediction": 1,
  "is_vulnerable": true,
  "parse_status": "json.vulnerable",
  "parsed_response": {
    "vulnerable": "yes",
    "cwe": "CWE-242",
    "reason": "gets reads unbounded input into a fixed-size buffer."
  },
  "chunk_verdicts": [
    {
      "chunk_id": "chunk-1",
      "prediction": 1,
      "vulnerable": "yes",
      "cwe": "CWE-242",
      "reason": "gets reads unbounded input into a fixed-size buffer.",
      "parse_status": "json.vulnerable",
      "start_line": 2,
      "end_line": 3,
      "raw_response": "{\"vulnerable\":\"yes\",\"cwe\":\"CWE-242\",\"reason\":\"...\"}"
    }
  ],
  "raw_response": "[chunk-1] {\"vulnerable\":\"yes\",\"cwe\":\"CWE-242\",\"reason\":\"...\"}",
  "chunk_count": 1
}
```

返回字段说明：

- `prediction`: 整体判定，`1` 表示存在漏洞，`0` 表示不存在漏洞
- `is_vulnerable`: `prediction` 的布尔形式
- `chunk_verdicts`: 逐个可疑 chunk 的复核结果
- `parsed_response`: 单 chunk 请求时，对该 chunk 的解析结果；多 chunk 时为空
- `chunk_verdicts[].raw_response`: 该 chunk 单独调用大模型后的原始返回
- `raw_response`: 全部 chunk 原始返回的汇总文本
- `chunk_count`: 本次送审的可疑 chunk 数量

调用示例：

```bash
curl -X POST http://127.0.0.1:8000/api/frontend/chunk-review \
  -H "Content-Type: application/json" \
  -d '{
    "code": "int main() {\n  char buf[8];\n  gets(buf);\n  return 0;\n}",
    "language": "c",
    "prompt_file": "LLM_TEST/Prompt/CWE-119_0.5.txt",
    "chunks": [
      {
        "chunk_id": "chunk-1",
        "text": "char buf[8];\ngets(buf);",
        "start_line": 2,
        "end_line": 3,
        "prediction": 1,
        "vulnerability_probability": 0.91
      }
    ]
  }'
```

### 3. `POST /api/jobs/classical`

用途：创建传统模型训练或测试任务，对应：

- `scripts/train.sh`
- `scripts/test.sh`

请求体：

```json
{
  "action": "train",
  "dataset_name": "reveal",
  "model_name": "ReGVD",
  "length": "0-512",
  "cuda": "0"
}
```

字段说明：

- `action`: `train` 或 `test`
- `dataset_name`: 数据集名
- `model_name`: `Devign`、`ReGVD`、`GraphCodeBERT`、`CodeBERT`、`UniXcoder`
- `length`: 数据长度区间，例如 `0-512`
- `cuda`: 使用的 GPU 编号，默认 `0`

调用示例：

```bash
curl -X POST http://127.0.0.1:8000/api/jobs/classical \
  -H "Content-Type: application/json" \
  -d '{
    "action": "train",
    "dataset_name": "reveal",
    "model_name": "ReGVD",
    "length": "0-512",
    "cuda": "0"
  }'
```

### 4. `POST /api/jobs/classical-imbalance`

用途：创建传统模型的不平衡数据训练或测试任务，对应：

- `scripts/train_imbalance.sh`
- `scripts/test_imbalance.sh`

请求体：

```json
{
  "action": "train",
  "dataset_name": "draper",
  "model_name": "CodeBERT",
  "length": "1",
  "pos_ratio": "0.2",
  "cuda": "0"
}
```

字段说明：

- `action`: `train` 或 `test`
- `dataset_name`: 数据集名
- `model_name`: 传统模型名
- `length`: 长度区间
- `pos_ratio`: 正样本比例
- `cuda`: GPU 编号

### 5. `POST /api/jobs/llm`

用途：创建大模型常规微调或推理任务，对应：

- `scripts/finetune.sh`
- `scripts/inference.sh`

请求体：

```json
{
  "action": "finetune",
  "dataset_name": "reveal",
  "model_name": "llama3.1",
  "length": "0-512",
  "batch_size": 16,
  "cuda": "0"
}
```

字段说明：

- `action`: `finetune` 或 `inference`
- `dataset_name`: 数据集名
- `model_name`: `llama2`、`llama3`、`llama3.1`、`codellama`
- `length`: 长度区间
- `batch_size`: 仅 `finetune` 时必填
- `cuda`: GPU 编号

如果 `action=inference`，可以不传 `batch_size`。

### 6. `POST /api/jobs/llm-imbalance`

用途：创建大模型在不平衡数据上的微调或推理任务，对应：

- `scripts/finetune_imbalance.sh`
- `scripts/inference_imbalance.sh`

请求体：

```json
{
  "action": "finetune",
  "dataset_name": "draper",
  "model_name": "llama3.1",
  "pos_ratio": "0.2",
  "cuda": "0"
}
```

### 7. `POST /api/jobs/ablation`

用途：创建 LoRA 消融实验任务，对应：

- `scripts/finetune_ablation.sh`
- `scripts/inference_ablation.sh`

请求体：

```json
{
  "action": "finetune",
  "dataset_name": "reveal",
  "model_name": "llama3.1",
  "r": 8,
  "alpha": 16,
  "cuda": "0"
}
```

字段说明：

- `r`: LoRA rank
- `alpha`: LoRA alpha

### 8. `POST /api/jobs/to-graph`

用途：执行图转换任务，对应：

- `scripts/to_graph.sh`

请求体：

```json
{
  "dataset_name": "reveal",
  "length": "0-512"
}
```

调用示例：

```bash
curl -X POST http://127.0.0.1:8000/api/jobs/to-graph \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "reveal",
    "length": "0-512"
  }'
```

### 9. `GET /api/jobs`

用途：获取当前后端已持久化的全部任务列表。

说明：

- 返回值是数组
- 按任务创建时间倒序排列
- 包含服务重启前已写入 SQLite 的历史任务

调用示例：

```bash
curl http://127.0.0.1:8000/api/jobs
```

### 10. `GET /api/jobs/{job_id}`

用途：查询某个任务的详细状态。

调用示例：

```bash
curl http://127.0.0.1:8000/api/jobs/0d9d7b8d0f9b4b2fa6f9470f4aaf1234
```

前端通常用这个接口做轮询。

### 11. `GET /api/jobs/{job_id}/log`

用途：读取某个任务日志文件的最后若干行，适合前端实时展示日志面板。

查询参数：

- `lines`: 返回最后多少行，默认 `200`，范围 `1` 到 `2000`

调用示例：

```bash
curl "http://127.0.0.1:8000/api/jobs/0d9d7b8d0f9b4b2fa6f9470f4aaf1234/log?lines=100"
```

返回示例：

```json
{
  "job_id": "0d9d7b8d0f9b4b2fa6f9470f4aaf1234",
  "log_file": "/home/zjr123/LLM4CVD-main/outputs/ReGVD/reveal_0-512/train_ReGVD_reveal_0-512.log",
  "content": "epoch 1 ...\nepoch 2 ..."
}
```

### 12. `POST /api/jobs/{job_id}/stop`

用途：终止正在运行的任务。

说明：

- 后端会尝试结束对应进程组
- 成功后任务状态会变成 `stopped`

调用示例：

```bash
curl -X POST http://127.0.0.1:8000/api/jobs/0d9d7b8d0f9b4b2fa6f9470f4aaf1234/stop
```

## 前端对接建议

推荐的基本流程：

1. 页面初始化时调用 `GET /api/meta/options`
2. 页面初始化时也可以调用 `GET /api/outputs` 加载已有输出目录
3. 用户选择参数后，调用对应的 `POST /api/jobs/...`
4. 拿到 `job_id` 后，每隔 2 到 5 秒轮询 `GET /api/jobs/{job_id}`
5. 如果需要实时日志，同时轮询 `GET /api/jobs/{job_id}/log?lines=100`
6. 任务完成后，用 `output_dir` 对应的相对路径继续调用 `GET /api/outputs`
7. 文本文件用 `GET /api/outputs/text`，图片和下载文件用 `GET /api/outputs/file`
8. 当任务状态变为 `completed`、`failed` 或 `stopped` 时停止轮询

## 一个完整示例

下面以传统模型训练为例：

### 第一步：提交训练任务

```bash
curl -X POST http://127.0.0.1:8000/api/jobs/classical \
  -H "Content-Type: application/json" \
  -d '{
    "action": "train",
    "dataset_name": "reveal",
    "model_name": "ReGVD",
    "length": "0-512",
    "cuda": "0"
  }'
```

### 第二步：查询任务状态

```bash
curl http://127.0.0.1:8000/api/jobs/<job_id>
```

### 第三步：查看日志

```bash
curl "http://127.0.0.1:8000/api/jobs/<job_id>/log?lines=100"
```

### 第四步：任务结束后读取输出

返回的任务信息里会提供：

- `output_dir`
- `log_file`
- `result_csv`

前端可以把这些路径显示出来，或者交给后续文件下载接口继续扩展。

## 当前限制

- 运行中任务的实时控制仍依赖当前进程持有的子进程信息；服务重启后虽然可以恢复任务记录，但无法恢复原始 Python `Popen` 对象
- 重启恢复时，对旧任务是否完成的判断主要依赖进程是否仍存在，以及 `results.csv` 是否存在，不能完全替代更严格的任务队列系统
- 日志接口只返回日志尾部文本
- 输出文件下载通过 `GET /api/outputs/file` 提供，数据库只保存文件路径元数据

## 后续可扩展方向

- 增加按任务类型筛选列表
- 增加 WebSocket 日志推送
- 增加鉴权和用户隔离
