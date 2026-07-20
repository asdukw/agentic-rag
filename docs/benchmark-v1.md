# Benchmark v1（2026-07-20）

> 历史基线，已由 [Benchmark v2](benchmark-v2.md) 取代。本文保留当时的失败现象和诊断过程；其中 round-robin、未人工审校题集和无 reranker 等条件不代表当前实现。

## 结论

当前应把 `naive` 作为默认检索基线，不应把 `mix` 或 `agentic` 宣传为已验证的升级方案。

- 60 题全量 retrieval-only 基准中，`naive` 与 `mix` 的 Raw Hit@8 同为 0.620。
- `mix` 仅在宏平均 Raw Recall@8 上高 0.013；`naive` 的 MRR、NDCG、micro recall、交付上下文质量和延迟均更好。
- `mix` 的主要短板是多上下文题：Raw Hit@8 为 0.300，低于 `naive` 的 0.500。
- 6 题端到端 smoke 中，`agentic` 的 Faithfulness 只有 0.662，平均每题 3.5 次工具调用、14.96 秒，没有证明其额外复杂度有收益。
- 三种模式都没有正确拒答 smoke 中的不可回答题。

## 数据与运行配置

- 语料：10 篇 RAG 论文，467 个正常 chunk。
- 图谱：1,806 个节点、1,198 条边；图构建运行 `gbr_6c9bbdb3738c43feb7974a356fd429a6`。
- 索引：`BAAI/bge-m3`，1,024 维；profile `idx_5e96c37b0d9f624d2d09`。
- 全量题集：60 题，包含 30 单跳、10 总结推理、10 多上下文、10 不可回答；10 篇论文各覆盖 7 次。
- Top K：8；候选倍数：4；上下文预算：2,400 tokens；图最大跳数：2；无 reranker。
- 全量检索使用本地 BGE-M3 和 `deterministic_keywords`，无外部 LLM 调用。
- 60 条标注均为 `unreviewed`；结果属于自动化基准，尚不是人工验收后的黄金集。

题集 SHA-256：`51DAFA1E56AF0D6F785053D469CA4D1CB62274215E0B6980093F8527412A6A92`。

## 60 题 retrieval-only 结果

相关性指标仅聚合 50 道可回答题；10 道不可回答题按定义排除。Raw Top-K 衡量检索排序，Delivered Context 衡量 token budget 裁剪后实际交付给回答器的证据。

| 模式 | Raw Hit@8 | Raw Recall@8 | Raw MRR | Raw NDCG@8 | Raw micro recall | Context Hit@8 | Context Recall@8 | Context MRR | Context NDCG@8 | 平均延迟 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | 0.620 | 0.577 | 0.330 | 0.386 | 0.508 | 0.580 | 0.540 | 0.322 | 0.371 | 3.580s | 4.290s |
| mix | 0.620 | 0.590 | 0.322 | 0.385 | 0.492 | 0.540 | 0.520 | 0.311 | 0.361 | 4.158s | 4.817s |

`mix - naive` 的配对结果：

- Raw Hit@8：3 胜、44 平、3 负，均值差 0.000。
- Raw Recall@8：3 胜、43 平、4 负，均值差 +0.013。
- Raw MRR / NDCG@8：均为 7 胜、31 平、12 负。
- Context Hit@8：2 胜、44 平、4 负，均值差 -0.040。
- `mix` 平均慢 0.578 秒，约增加 16.1% 延迟。

### 按题型

| 题型 | 模式 | N | Raw Hit@8 | Raw Recall@8 | Raw MRR | Context Hit@8 |
|---|---|---:|---:|---:|---:|---:|
| single_hop | naive | 30 | 0.633 | 0.633 | 0.370 | 0.600 |
| single_hop | mix | 30 | 0.667 | 0.667 | 0.370 | 0.567 |
| summary_reasoning | naive | 10 | 0.700 | 0.700 | 0.410 | 0.700 |
| summary_reasoning | mix | 10 | 0.800 | 0.800 | 0.460 | 0.800 |
| multi_context | naive | 10 | 0.500 | 0.280 | 0.130 | 0.400 |
| multi_context | mix | 10 | 0.300 | 0.150 | 0.060 | 0.200 |

## 6 题端到端 smoke

该 smoke 是独立生成的 6 题集，只覆盖 2 篇论文。查询与回答模型为 `deepseek-v4-flash`，评审模型为 `deepseek-v4-pro`。RAGAS 指标只评估 5 道可回答题；不可回答题单独计算拒答行为。

| 模式 | Faithfulness | Factual Correctness | Context Precision | Context Recall | Hit@8 | MRR | 不可回答拒答率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive | 0.888 | 0.200 | 0.401 | 0.600 | 0.600 | 0.450 | 0.000 |
| mix | 0.897 | 0.214 | 0.373 | 0.467 | 0.600 | 0.500 | 0.000 |
| agentic | 0.662 | 0.266 | 0.367 | 0.550 | 0.600 | 0.367 | 0.000 |

`agentic` 还出现了 0.800 的可回答题响应率，说明 5 道可回答题中有 1 道错误拒答；其引用合法率为 1.000，但参考证据精度只有 0.217。

## 成本与中断说明

- 完整图构建的估算成本为约 ¥1.183。
- `naive + mix` smoke 的查询/回答可观测成本为约 ¥0.0187；当前 RAGAS API 不暴露 judge usage，因此 judge 与总成本未知。
- 本地 60 题 retrieval-only 运行的外部 API 成本为 ¥0。
- 60 题端到端运行已发起，但第一笔请求收到 DeepSeek `402 Insufficient Balance`，因此没有端到端全量分数。不得把 retrieval-only 结果描述成完整答案质量基准。

## 局限

- 题集由 LLM 生成且尚未人工复核，可能存在参考答案或不可回答标签错误。
- retrieval-only 使用确定性关键词；它不能与使用 DeepSeek 关键词的 smoke 做严格横向归因。
- Evidence ID 以页/section 为粒度，不是 chunk ID；跨页 chunk 可能一次命中多个 gold ID，NDCG 需结合该口径解释。
- Delivered Context 的下降同时包含检索排序和上下文装配损失，不能全部归因于检索器。
- `mix` 当前的来源编排语义需要结合 trace 分析，不能只看最终融合分数。

## 下一步

1. 人工复核 60 题，优先检查 10 道不可回答题和 10 道多上下文题。
2. 对 `mix` 的多上下文失败做 trace 级误差分析，检查图路由、round-robin 编排和 context budget。
3. 修复拒答策略，再用相同 smoke 回归。
4. 充值后补跑 60 题端到端 `naive + mix`；保留 judge 8,192-token 上限并增加评分检查点。
5. 在稳定基线后再评估本地 reranker，不先扩大 `agentic`。

## 复现

本地 retrieval-only：

```powershell
$env:HF_HOME = Join-Path (Resolve-Path '.').Path '.tmp\hf-benchmark'
$env:HF_HUB_DISABLE_XET = '1'
uv run scripts/evaluate_retrieval.py `
  --db storage/app.db `
  --profile idx_5e96c37b0d9f624d2d09 `
  --testset artifacts/ragas/rag-papers-benchmark-v1.json `
  --modes naive,mix `
  --output artifacts/evaluations/rag-papers-retrieval-only-naive-mix-v1.json
```

余额恢复后的端到端全量命令：

```powershell
$env:DEEPSEEK_JUDGE_MAX_OUTPUT_TOKENS = '8192'
uv run scripts/evaluate_no_rerank.py `
  --full `
  --db storage/app.db `
  --profile idx_5e96c37b0d9f624d2d09 `
  --testset artifacts/ragas/rag-papers-benchmark-v1.json `
  --modes naive,mix `
  --output artifacts/evaluations/rag-papers-benchmark-naive-mix-no-rerank-v1.json
```

全量 retrieval-only 报告 SHA-256：`69CCFE3D88B85358F7E6815A10170F3D05C2CA23A6251C5461E5C404987F29B9`。
