# Benchmark v3 预注册方案（待执行）

> 状态：指标与实验设计已冻结，尚未运行 benchmark，也没有 v3 结果。

## 版本边界

Benchmark v3 是新的**评测协议与报告版本**，不是新题集版本。

- 继续使用人工审校后的 v2 题集，不重新生成问题。
- 题集 schema 仍为 `2`，共 60 题。
- 只有发现客观标注错误时才修题；不得根据模式得分改题或调参。
- v2 历史报告保留原名称与原始数值；v3 直接使用当前策略名称。

冻结输入：

| 项目 | 值 |
|---|---|
| 题集 | `artifacts/ragas/rag-papers-benchmark-v2.json` |
| 题集 SHA-256 | `27C53590105D002D1E6FDB2B529242954B5C8423EE1791E005A6BA1EFD4663B2` |
| schema version | `2` |
| corpus content hash | `a809ebc5c4ad960466663cee899ece9c4e016703f94fa5c5d4838ea9998077f4` |
| 题量 | 30 single-hop、10 summary/reasoning、10 multi-context、10 unanswerable |
| index profile | `idx_5e96c37b0d9f624d2d09` |

运行时仍必须在报告中记录实际 commit、dirty state、依赖版本、模型、CUDA、GPU、profile 和配置 hash。正式结果只接受 clean commit 上的运行。

## 研究问题

v3 只回答三个问题：

1. BM25 相比纯 Dense 提供了什么增量？
2. 图检索与多跳补证加入 Hybrid 后，Mix 是否改善跨文档和多证据问题？
3. 这些收益在启用同一个 cross-encoder reranker 后是否仍然存在，代价是多少？

不把“某个模式所有指标全面第一”设为目标，也不根据这 60 题搜索最佳权重。

## 配对实验矩阵

每次命令只比较两个模式，以获得同题配对差值、胜平负和 bootstrap 区间。

| ID | Baseline | Candidate | Reranker | 解释 |
|---|---|---|---|---|
| E1 | `bm25` | `dense` | 关闭 | 稀疏与稠密基础能力 |
| E2 | `dense` | `hybrid` | 关闭 | BM25 对 Dense 的直接增量 |
| E3 | `dense` | `hybrid` | 开启 | 强 reranker 下 BM25 的边际价值 |
| E4 | `hybrid` | `mix` | 关闭 | 图路由、多跳和多上下文补证的原始增量 |
| E5 | `hybrid` | `mix` | 开启 | 主展示实验，完整本地检索栈 |

E5 是主结果；其他实验用于解释收益来自哪里，不进行跨实验挑选最佳配置。

## 冻结运行配置

- Top K：8。
- 每路候选池：`top_k × candidate_multiplier`，当前为 32。
- rerank 候选池：`top_k × rerank_candidate_multiplier`，当前为 32。
- context budget：2,400 tokens。
- 图最大跳数：2。
- embedding：`BAAI/bge-m3`，1,024 维，FP32。
- reranker：`BAAI/bge-reranker-v2-m3`，启用实验使用 CUDA FP16。
- 语义证据阈值：cosine similarity >= 0.75。
- 查询扩展：`deterministic_keywords`，retrieval-only 阶段不调用外部 LLM。
- 模式执行顺序逐题交替；第一次调用只作 warmup，不计入指标。

如果运行前配置与以上不同，必须先修改本方案并提交，不能在看到结果后补写。

## Retrieval-only 指标

### 1. Exact-page

以人工标注的 document/page/section evidence ID 为相关性标准，同时报告 Raw Top-K 与 Delivered Context：

- Hit@8
- Recall@8
- micro Recall@8
- MRR
- nDCG@8

它判断精确证据是否命中以及排序位置，但可能受到分页和 chunk 边界影响。

### 2. Document-level

以 gold document ID 为相关性标准：

- Hit@8
- Recall@8
- MRR
- nDCG@8

它判断目标来源是否到达候选和最终上下文，是解释 Mix 跨来源价值的主口径之一。

### 3. Semantic evidence

对每个 reference context，取其与检索证据的最大 embedding cosine similarity：

- Mean Max Similarity
- Threshold Recall：超过 0.75 的 reference context 比例
- All References Covered：是否所有 reference context 都超过阈值

该指标同时报告 Raw Top-K 与 Delivered Context，不替代 exact-page 指标。

### 4. Multi-evidence

- Complete-chain Rate：所有 gold exact-page evidence 同时命中的题目比例。
- Document Coverage：命中的 gold documents / 全部 gold documents。
- All-documents-covered Rate：所有 gold documents 是否同时到齐。
- Distinct Source Count：结果中的不同来源文档数。

以上指标同样区分 Raw Top-K 与 Delivered Context，并按题型拆分。标题中的 multi-context 结论只使用 `by_question_type.multi_context` 子集。Distinct Source Count 是描述性指标，来源更多不自动代表质量更高，因此不计入胜负判断。

### 5. 性能

- Mean latency
- P50 latency
- P95 latency
- 逐题 latency delta

延迟的配对胜负方向为越低越好。60 次查询不足以稳定估计 P99，因此 v3 不报告 P99。

## 配对统计

对双方均有有效值的同一道题，计算 `candidate - baseline`：

- eligible pairs
- candidate wins / ties / baseline wins
- win / tie / loss rate
- mean delta
- paired bootstrap 95% CI

Bootstrap 固定为：

- 20,000 次重采样。
- seed `20260721`。
- 重采样单位为逐题配对差值，不分别重采样两个模式。
- percentile interval，2.5%–97.5%。
- 同时报告 overall 与每个 question type。

区间用于表达 60 题下的不确定性，不把“区间未跨 0”包装成超出本题集范围的普遍结论。

## 端到端与 Agentic 指标

端到端评测与 retrieval-only 分开报告。先运行固定的 6 题 stratified smoke；只有 API 预算允许时才运行完整 60 题。Smoke 只验证链路，不用于比较模式优劣。

### 生成与 Claim

- Faithfulness
- Claim Precision
- Claim Recall
- Claim F1：precision 与 recall 的调和平均
- Factual Correctness：当前报告中等于 Claim F1
- Context Precision
- Context Recall

Claim precision 与 recall 使用相同 judge 配置分别评分；不能把独立 judge 的小幅差异解释为确定性事实。

### 引用

当前回答协议是“整份答案 + 一组 citation IDs”，不是逐 claim 行内引用。因此 v3 只声称答案级 attribution：

- Citation Claim Support：回答 claims 被所有已引用上下文的并集支持的比例。
- Citation Correctness：Citation Claim Support 的报告别名。
- Citation Completeness：引用上下文覆盖 gold reference evidence 的比例。
- Unsupported Claim Rate：`1 - Citation Claim Support`。
- Citation ID Validity：仅验证 ID 是否来自允许的已读证据，作为结构指标，不等价于语义正确性。

v3 不声称 inline attribution accuracy。若以后需要逐句引用正确率，应先升级回答协议为 claim → citations 映射。

### 拒答

把 `answerable` 设为正类，报告：

- TP：可回答且作答。
- FP：不可回答但强行作答。
- FN：可回答但拒答。
- TN：不可回答且正确拒答。
- Answerability Precision / Recall / F1
- False-answer Rate：`FP / (FP + TN)`
- False-refusal Rate：`FN / (TP + FN)`
- Refusal Accuracy

### Agentic 诊断

如果运行 Agentic 模式，额外报告：

- tool call count、successful tool calls、tool calls by name
- read/cited evidence count
- evidence utilization
- citation validity 与 reference-evidence precision/recall
- duration 与 refusal accuracy

这些是轨迹诊断指标。v3 不构造没有人工 gold route 的“路由正确率”。

## 主指标与解释规则

对外摘要只展示以下主指标，其他指标保留在机器可读报告中：

1. Exact-page Delivered Recall@8
2. Document Delivered Recall@8 / nDCG@8
3. Semantic Delivered Threshold Recall
4. Multi-context Complete-chain Rate
5. Multi-context All-documents-covered Rate
6. P50 / P95 latency
7. paired mean delta、win/tie/loss 和 95% CI

解释规则预先固定：

- Hybrid 的价值主要由 E2/E3 判断。
- Mix 的价值主要由 E4/E5 的 multi-context、document 和 semantic delivered-context 指标判断。
- Mix 不需要在 single-hop exact-page 上全面领先才算有价值。
- 如果质量差值很小或区间跨 0，应报告“未观察到稳定优势”。
- 所有质量收益必须同时给出延迟代价。
- 不计算单一加权总分，避免通过权重制造“冠军”。

## 暂不引入

作为学习项目，v3 不扩展以下生产指标：

- P99、吞吐和并发压测
- TTFT 流式指标
- 索引 freshness lag
- 多租户泄漏、PII、数据投毒和完整 prompt-injection 红队矩阵
- 在线满意度、转人工率和业务转化
- judge cost per correct answer（当前 judge usage 不可完整观测）

这些属于生产验证，不影响本项目展示检索、证据、回答和评测方法论。

## 预期产物

正式运行后生成：

- `docs/benchmark-v3.md`：人类可读结论、限制和复现命令。
- `docs/benchmark-v3-summary.json`：公开、精简、机器可读结果。
- `artifacts/evaluations/benchmark-v3-*.json`：本地完整逐题 trace 报告，不提交仓库。

在本方案通过审核前，不运行或发布 Benchmark v3 结果。
