# PFRVul：基于多智能体协同推理的代码漏洞分析平台

本项目为本科毕业设计《PFRVul：基于多智能体协同推理的代码漏洞分析平台》的开源实现。

近年来，大语言模型（LLM）在代码理解、程序分析和软件安全领域展现出较强能力，但直接应用于漏洞检测任务时仍面临误报率高、长代码上下文处理困难以及推理结果不稳定等问题。

为解决上述问题，本项目提出了一种面向代码漏洞检测场景的多智能体协同分析框架（Multi-Agent Workflow），构建了由 **Screening Agent、Reviewer Agent 和 Consensus Agent** 组成的分层推理体系，实现漏洞初筛、语义复审与决策融合。

同时，为支持实验管理与系统落地，项目开发了基于 FastAPI + Vue 的可视化平台，实现模型调用、Prompt 管理、任务调度、实验评测与结果展示等功能。

项目主要研究内容包括：

* 基于 AST 的长代码结构化检索与切分
* 多智能体协同漏洞分析框架
* Prompt Engineering 与结构化推理
* 多模型投票机制
* LoRA Reviewer 微调框架
* 漏洞检测系统平台开发

---

# 研究模型

| 类型    | 模型                       |
| ----- | ------------------------ |
| 小模型   | CodeBERT                 |
| 小模型   | UniXcoder                |
| 大语言模型 | DeepSeek                 |
| 大语言模型 | Moonshot                 |
| 大语言模型 | OpenAI Compatible Models |
| Agent | Screening Agent          |
| Agent | Reviewer Agent           |
| Agent | Consensus Agent          |

---

# 实验数据集

项目在多个公开漏洞数据集上开展实验：

| 数据集        | 简介                  |
| ---------- | ------------------- |
| Devign     | 函数级漏洞检测数据集          |
| DiverseVul | 大规模多类型漏洞数据集         |
| CVEfixes   | 基于真实 CVE 修复记录构建的数据集 |

---

# 研究内容

## 1. 小模型漏洞检测

研究并复现：

* CodeBERT
* UniXcoder

等代码预训练模型在漏洞检测任务中的表现。

---

## 2. 大模型直接漏洞分析

研究大语言模型在漏洞检测场景下的能力，包括：

* Prompt设计
* Chain-of-Thought推理
* 结构化输出
* 漏洞解释生成

---

## 3. 多模型投票机制

实现：

* 任意模型判定为漏洞
* 多数投票
* 全票通过

等融合策略，对比不同决策方式对检测性能的影响。

---

## 4. Reviewer框架

构建：

```text
代码输入
    ↓
小模型初筛
    ↓
Reviewer Agent复审
    ↓
结果修正
```

利用大模型对小模型预测结果进行二次验证，降低误报率。

---

## 5. 多智能体协同分析框架

整体流程如下：

```text
代码输入
      ↓
Screening Agent
      ↓
AST检索与上下文构建
      ↓
Reviewer Agent
      ↓
Consensus Agent
      ↓
最终漏洞报告
```

---

# 系统平台

项目提供完整 Web 平台。

主要功能包括：

* 漏洞分析任务管理
* Prompt管理
* 模型路由
* 实验评测
* 结果可视化
* Agent推理过程展示

---

# FastAPI 后端

启动后端：

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Swagger：

```text
http://127.0.0.1:8000/docs
```

---

# 项目结构

```text
PFRVul
├── backend/                # FastAPI后端
├── frontend/               # Vue前端
├── agents/                 # Agent实现
├── prompts/                # Prompt模板
├── models/                 # 模型接口
├── evaluation/             # 实验评测
├── datasets/               # 数据处理
├── outputs/                # 实验结果
└── docs/                   # 项目文档
```

---

# 技术栈

后端：

* Python
* FastAPI
* SQLAlchemy

前端：

* Vue3
* Element Plus

AI组件：

* CodeBERT
* UniXcoder
* DeepSeek
* Moonshot
* LoRA
* Agent Workflow

