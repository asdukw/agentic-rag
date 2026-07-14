# Ragas 评测报告模板

> 状态：模板。除非明确填入一次实际执行的 JSON 报告，本文件不声称任何质量结果。

项目唯一的评测入口是：

```bash
uv run hybrid-rag evaluate --testset <testset.json>
```

该命令必须设置 `DEEPSEEK_API_KEY`。它会对每个测试样本运行当前 RAG 的真实 `ask` 流程，随后由
Ragas 使用 DeepSeek 评审实际回答和召回上下文。

## 1. 执行范围

| 项目 | 本次记录 |
| --- | --- |
| 日期与代码提交 | `NOT RUN` |
| 测试集路径与文件 hash | `NOT RUN` |
| 测试集 `schema_version` | `1` |
| 测试集 `corpus_content_hash` | `NOT RUN` |
| 数据库 / index profile | `NOT RUN` |
| 检索模式 | `mix`（默认） |
| DeepSeek answer / judge 模型与 endpoint | `NOT RUN` |
| 检索参数（Top-K、token budget、graph hops、权重、reranker） | `NOT RUN` |

推荐将命令及其输出 JSON 一同记录。例如：

```bash
uv run hybrid-rag evaluate \
  --testset data/processed/my-ragas-testset.json \
  --modes naive,mix \
  --output artifacts/evaluations/ragas-demo.json \
  --json
```

默认只评测 `mix`。只有在同一测试集、同一 profile 与同一参数下显式传入多个 `--modes` 时，模式间结果
才适合比较。

## 2. 测试集契约与语料绑定

测试集必须是 JSON envelope：

```json
{
  "schema_version": "1",
  "corpus_content_hash": "<64位小写十六进制hash>",
  "cases": [
    {
      "user_input": "问题",
      "reference": "参考答案",
      "reference_contexts": ["参考上下文"]
    }
  ]
}
```

`corpus_content_hash` 来自待评测 index profile 的 document/chunk 语料指纹，而非原始 PDF 的 SHA-256、
图谱快照 hash 或任意字符串。先在该语料上运行：

```bash
uv run hybrid-rag ingest <corpus-dir>
uv run hybrid-rag build-index --json
```

从第二条命令输出中复制 `corpus_content_hash`，再传给生成脚本：

```bash
uv run scripts/ragas_testset_demo.py \
  --source-dir <corpus-dir> \
  --corpus-content-hash <build-index输出的corpus_content_hash>
```

脚本不会自行从 PDF 计算该值，因为项目的语料指纹还依赖导入后的 document/chunk 身份与内容。语料、导入
配置、分块配置或目标 profile 改变后，必须重新 `ingest`、`build-index`，并生成带新 hash 的测试集。
评测入口会拒绝 hash 不匹配或裸数组格式的测试集。

## 3. 指标解释

| 指标 | Ragas 评估对象 | 解读边界 |
| --- | --- | --- |
| Faithfulness | 回答是否被本次实际召回的上下文支持 | 不代表回答覆盖了所有正确知识。 |
| Factual correctness | 回答相对 `reference` 的事实正确性 | 受参考答案完整度与评审模型影响。 |
| Context precision | 召回上下文的相关性与排序质量 | 不直接衡量最终回答措辞。 |
| Context recall | 召回上下文对 `reference` 所需信息的覆盖 | 依赖 `reference_contexts` 和参考答案质量。 |

每个模式的 JSON 包含逐 case 分数与均值。出现缺失、异常或低分时，应先检查原始 question、reference、
reference contexts、实际 retrieved contexts、回答与 `rtr_` trace，而不是仅依据均值下结论。

## 4. 结果摘要

| 模式 | Cases | Faithfulness | Factual correctness | Context precision | Context recall | 输出 JSON |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| mix | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` |

如果同一执行传入多个模式，为每个模式增加一行；不要将不同语料 hash、profile、模型或 Ragas 测试集的
均值混为一组比较。

## 5. 成本与复现信息

输出的 `provenance` 会记录测试集文件 SHA-256、锁定 profile、模式、完整检索参数，以及回答与评审模型的
非敏感运行配置。记录实际 JSON 输出中的 `cost` 字段。`cost.retrieval` 为每条 `rtr_` trace 保留可观测的 DeepSeek
成本快照，并仅在完整 usage 可用时给出估算；共享 query client 时采用最后一个累计快照，避免重复相加。
当前 Ragas API 不暴露评审模型 usage，因此 `cost.judge` 与 `cost.total` 会明确显示为 `unknown`，不会
猜测金额。任何估算取决于响应中的缓存命中输入、缓存未命中输入和输出 token，以及 `.env` 中的价格配置；
它不是 DeepSeek 账单。还应保留：

- 测试集文件及其 `corpus_content_hash`；
- index profile ID、embedding 配置与图谱快照；
- `evaluate` 的完整命令行、模式和检索参数；
- 输出 JSON 与相关 `rtr_` trace；
- 运行日期、代码提交和 DeepSeek 模型 / endpoint。

Ragas 分数只适用于上述冻结条件。扩展语料、重分块、更新索引或更换回答/评审模型后，应重新生成测试集
或至少重新验证语料 hash，再单独报告新的执行结果。
