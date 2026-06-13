# Chunk Review CLI

这个命令行工具复用 `fastapi_backend/frontend_chunk_review.py` 中现有的“可疑 chunk + 完整代码”复核逻辑，但不需要启动 FastAPI。

适合后端直接批量跑一组或多组代码复核。

## 运行方式

在仓库根目录执行：

```bash
python3 -m fastapi_backend.chunk_review_cli \
  --input_json /path/to/review_groups.json \
  --prompt_file LLM_TEST/Prompt/CWE-119_0.5.txt
```

常用可选参数：

- `--output_json`: 指定输出文件
- `--config`: 默认 `LLM_TEST/exp.yaml`
- `--env_file`: 默认 `LLM_TEST/.env`
- `--model`
- `--api_base`
- `--api_key`
- `--group_workers`: 多组并发数，默认 `1`
- `--fail_fast_on_error`

如果不传 `--output_json`，默认输出到：

```text
<input_json 同目录>/<input_stem>.reviewed.json
```

## 输入格式

输入 JSON 支持三种形式：

1. 单组对象
2. 多组数组
3. 带 `groups` 字段的对象

每组支持这些字段：

```json
{
  "group_id": "sample-1",
  "language": "c",
  "code": "int main() { return 0; }",
  "chunks": [
    {
      "chunk_id": "chunk-1",
      "text": "return 0;",
      "start_line": 1,
      "end_line": 1
    }
  ]
}
```

也可以把代码和 chunk 单独放文件：

```json
{
  "group_id": "sample-2",
  "language": "c",
  "code_file": "inputs/sample2.c",
  "chunks_file": "inputs/sample2_chunks.json"
}
```

相对路径会按 `input_json` 所在目录解析。

## 多组示例

```json
{
  "groups": [
    {
      "group_id": "case-1",
      "language": "c",
      "code_file": "cases/case1.c",
      "chunks": [
        {
          "chunk_id": "case-1-chunk-1",
          "text": "strcpy(dst, src);",
          "start_line": 12,
          "end_line": 12
        }
      ]
    },
    {
      "group_id": "case-2",
      "language": "java",
      "code": "class Demo { void run(String x) { Runtime.getRuntime().exec(x); } }",
      "chunks": [
        {
          "chunk_id": "case-2-chunk-1",
          "text": "Runtime.getRuntime().exec(x);",
          "start_line": 1,
          "end_line": 1
        }
      ]
    }
  ]
}
```

## 输出格式

输出 JSON 会汇总每组结果，核心结构如下：

```json
{
  "group_count": 2,
  "completed_groups": 2,
  "failed_groups": 0,
  "results": [
    {
      "group_id": "case-1",
      "chunk_count": 1,
      "result": {
        "prediction": 1,
        "is_vulnerable": true,
        "chunk_verdicts": []
      }
    }
  ]
}
```

其中每个 `result` 的内部字段与 `/api/frontend/chunk-review` 接口返回结构保持一致。

## 推荐批量链路

如果你手上已经有前端后端落盘的切分结果和小模型结果，推荐先运行：

```bash
python3 -m fastapi_backend.prepare_chunk_review_groups \
  --chunking_json data/user_demo_chunks/user_demo_chunks_0-512_test.json \
  --inference_json outputs/user_demo_chunks/user_demo_chunks_0-512_test_CodeBERT_frontend_inference.json
```

它会自动：

- 从 `chunking_request.json` 取回完整代码
- 从小模型推理结果里筛出 `prediction=1` 的 chunk
- 按样本聚合成 `groups`
- 输出默认文件 `chunk_review_groups.json`

然后再把这个文件喂给本工具：

```bash
python3 -m fastapi_backend.chunk_review_cli \
  --input_json data/user_demo_chunks/chunk_review_groups.json \
  --prompt_file LLM_TEST/Prompt/CWE-119_0.5.txt
```
