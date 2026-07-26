# Benchmark v3（2026-07-26）

> 状态：预注册的 E1–E5 retrieval-only 矩阵已全部执行。端到端回答与
> Agentic 评测不包含在本报告中。

## 结论

本轮在 clean commit、同一语料、同一索引 profile 和同一 GPU 上完成了
BM25、Dense、Hybrid 与 Mix 的五组配对实验。结果不支持“增加检索路由一定
提升效果”这一简单结论：

- 纯 Dense 明显优于纯 BM25；当前确定性关键词和论文语料下，BM25 不能独立
  承担主检索器。
- 无 reranker 时，等权加入 BM25 会削弱 Dense：Hybrid 的 Exact-page
  Delivered Recall@8 从 0.56 降至 0.49。
- 启用同一个 cross-encoder 后，Dense 与 Hybrid 的 Exact-page Delivered
  Recall@8 同为 0.70；BM25 的边际价值缩小到少量文档和语义覆盖增量。
- Mix 的稳定价值集中在交付上下文的跨文档覆盖和排序，而不是精确页命中：
  主实验 E5 中 Document Delivered Recall@8 从 0.88 提升至 0.95，
  Document Delivered nDCG@8 从 0.846 提升至 0.922。
- E5 的 Exact-page Delivered Recall@8 仅从 0.70 提升至 0.71，
  Semantic Delivered Threshold Recall 从 0.76 提升至 0.82；两者的
  bootstrap 区间均跨 0，不能表述为稳定的全面优势。
- E5 中 Mix 平均增加 0.384 秒延迟，P95 从 3.052 秒增至 3.519 秒。

因此，当前推荐把 `dense + reranker` 作为精简基线，把
`mix + reranker` 作为更重视跨来源覆盖时的可选策略。`hybrid` 不应在关闭
reranker 时被默认视为 Dense 的升级。

## 实验输入与环境

| 项目 | 值 |
|---|---|
| Commit | `716e6a11845012ec4676ec7cec24be3608febdf5` |
| Git 状态 | clean |
| 语料 | 10 篇 RAG 论文，467 个 normal chunk |
| 题集 | 60 题：30 single-hop、10 summary/reasoning、10 multi-context、10 unanswerable |
| 可回答题 | 50 |
| 题集 SHA-256 | `27c53590105d002d1e6fdb2b529242954b5c8423ee1791e005a6ba1efd4663b2` |
| Corpus content hash | `a809ebc5c4ad960466663cee899ece9c4e016703f94fa5c5d4838ea9998077f4` |
| Index profile | `idx_5e96c37b0d9f624d2d09` |
| Embedding | `BAAI/bge-m3`，1,024 维，FP32 |
| Reranker | `BAAI/bge-reranker-v2-m3`，CUDA FP16 |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| Torch | `2.13.0+cu130` |
| Top K / candidate pool | 8 / 32 |
| Context budget | 2,400 tokens |
| 图最大跳数 | 2 |
| 语义证据阈值 | cosine similarity >= 0.75 |
| 外部 LLM 调用 | 0 |

所有实验逐题交替执行两个模式，并在计时前执行一次不计入指标的 warmup。
配对区间使用固定 seed `20260721`、20,000 次 case-level paired percentile
bootstrap。

## 五组实验

下表只展示预注册主指标的总体均值。相关性指标聚合 50 道可回答题，延迟聚合
全部 60 题。

| 实验 | 模式 | Exact Context Recall@8 | Document Context Recall@8 | Document Context nDCG@8 | Semantic Context Recall | All Documents Covered | 平均延迟 | P95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E1 无 rerank | BM25 | 0.00 | 0.41 | 0.289 | 0.11 | 0.38 | 1.662s | 1.918s |
|  | Dense | **0.56** | **0.92** | **0.843** | **0.74** | **0.86** | 1.802s | 1.843s |
| E2 无 rerank | Dense | **0.56** | **0.92** | **0.843** | **0.74** | **0.86** | 1.238s | 1.370s |
|  | Hybrid | 0.49 | 0.88 | 0.765 | 0.65 | 0.80 | 1.333s | 1.464s |
| E3 FP16 rerank | Dense | 0.70 | 0.87 | 0.846 | 0.74 | 0.80 | **2.500s** | 2.908s |
|  | Hybrid | 0.70 | **0.88** | **0.846** | **0.76** | **0.82** | 2.610s | **2.834s** |
| E4 无 rerank | Hybrid | 0.49 | 0.88 | 0.765 | 0.65 | 0.80 | **1.389s** | **1.540s** |
|  | Mix | **0.52** | **0.96** | **0.877** | **0.70** | **0.92** | 1.618s | 1.926s |
| E5 FP16 rerank | Hybrid | 0.70 | 0.88 | 0.846 | 0.76 | 0.82 | **2.600s** | **3.052s** |
|  | Mix | **0.71** | **0.95** | **0.922** | **0.82** | **0.92** | 2.984s | 3.519s |

不同实验中的绝对延迟不能直接配对比较；模型加载状态、系统负载和执行时段会
影响均值。模式增量以每个实验内部的逐题配对差值为准。

## 配对结果

### E2：BM25 加入 Dense 的原始增量

无 reranker 时，Hybrid 相比 Dense：

| 指标 | 平均差值 | 95% CI | 胜 / 平 / 负 |
|---|---:|---:|---:|
| Exact Context Recall@8 | -0.070 | [-0.140, -0.020] | 0 / 45 / 5 |
| Document Context Recall@8 | -0.040 | [-0.080, -0.010] | 0 / 46 / 4 |
| Document Context nDCG@8 | -0.077 | [-0.131, -0.026] | 2 / 31 / 17 |
| Semantic Context Recall | -0.090 | [-0.170, -0.030] | 0 / 44 / 6 |
| 平均延迟 | +0.094s | [+0.065, +0.123] | 14 / 0 / 46 |

等权 BM25 融合在当前配置下带来可观察的退化。该结果说明 Hybrid 的工程价值
不能只靠模式名称成立；BM25 查询词质量、分数校准与融合权重都需要独立验证。

### E3：强 reranker 下 BM25 的边际价值

启用同一个 FP16 reranker 后，Hybrid 相比 Dense：

| 指标 | 平均差值 | 95% CI | 胜 / 平 / 负 |
|---|---:|---:|---:|
| Exact Context Recall@8 | 0.000 | [-0.060, +0.060] | 1 / 48 / 1 |
| Document Context Recall@8 | +0.010 | [0.000, +0.030] | 1 / 49 / 0 |
| Semantic Context Recall | +0.020 | [0.000, +0.060] | 1 / 49 / 0 |
| 平均延迟 | +0.111s | [+0.052, +0.167] | 14 / 0 / 46 |

reranker 将两种候选池重排为几乎相同的 Top-8。Hybrid 的剩余增量只集中在一
道题，不足以说明 BM25 在强 Dense 基线之上具有普遍收益。

### E4：图路由与补证的原始增量

无 reranker 时，Mix 相比 Hybrid：

| 指标 | 平均差值 | 95% CI | 胜 / 平 / 负 |
|---|---:|---:|---:|
| Exact Context Recall@8 | +0.030 | [-0.060, +0.120] | 5 / 43 / 2 |
| Document Context Recall@8 | +0.080 | [+0.030, +0.140] | 7 / 43 / 0 |
| Document Context nDCG@8 | +0.112 | [+0.043, +0.182] | 17 / 28 / 5 |
| Semantic Context Recall | +0.050 | [-0.040, +0.140] | 6 / 42 / 2 |
| All Documents Covered | +0.120 | [+0.040, +0.220] | 6 / 44 / 0 |
| 平均延迟 | +0.229s | [+0.193, +0.268] | 0 / 0 / 60 |

Mix 在跨文档覆盖和文档排序上表现出更稳定的增量，但 Exact-page 与语义覆盖
区间仍跨 0。

### E5：完整本地检索栈

启用同一个 FP16 reranker 后，Mix 相比 Hybrid：

| 指标 | 平均差值 | 95% CI | 胜 / 平 / 负 |
|---|---:|---:|---:|
| Exact Context Recall@8 | +0.010 | [-0.090, +0.110] | 5 / 41 / 4 |
| Document Context Recall@8 | +0.070 | [0.000, +0.150] | 7 / 41 / 2 |
| Document Context nDCG@8 | +0.076 | [+0.022, +0.138] | 10 / 38 / 2 |
| Semantic Context Recall | +0.060 | [-0.020, +0.150] | 6 / 42 / 2 |
| Complete-chain Rate | +0.020 | [-0.080, +0.120] | 4 / 43 / 3 |
| All Documents Covered | +0.100 | [0.000, +0.200] | 6 / 43 / 1 |
| 平均延迟 | +0.384s | [+0.309, +0.465] | 6 / 0 / 54 |

Document Context nDCG@8 是 E5 中区间明确高于 0 的主要质量指标。其他指标
多数平局或区间触及、跨过 0，因此结论限定为“改善部分跨来源问题的交付上下文
覆盖与排序”，不表述为全面提升。

## Multi-context 子集

10 道 multi-context 题上：

| 实验 | 模式 | Exact Context Recall@8 | Document Context Recall@8 | Document Context nDCG@8 | Semantic Context Recall | Complete-chain | All Documents Covered |
|---|---|---:|---:|---:|---:|---:|---:|
| E4 无 rerank | Hybrid | 0.05 | 0.50 | 0.493 | 0.25 | 0.00 | 0.10 |
|  | Mix | **0.20** | **0.80** | **0.762** | **0.40** | 0.00 | **0.60** |
| E5 FP16 rerank | Hybrid | 0.05 | 0.60 | 0.552 | 0.30 | 0.00 | 0.30 |
|  | Mix | **0.15** | **0.75** | **0.739** | **0.40** | **0.10** | **0.60** |

E5 的 multi-context Document Context nDCG@8 配对差值为 +0.186，95% CI
[+0.010, +0.334]。但该子集只有 10 题，单题会让 rate 指标变化 0.10；
Document Recall、Semantic Recall 和 All Documents Covered 的区间仍跨 0。

## Reranker 的收益与代价

虽然 E2/E3、E4/E5 是不同时间运行，不能作为严格的逐题延迟配对，但相关性
结果是确定性的，可用于解释 reranker 的影响：

- Dense Exact Context Recall@8：0.56 → 0.70。
- Hybrid Exact Context Recall@8：0.49 → 0.70。
- Mix Exact Context Recall@8：0.52 → 0.71。
- Mix Semantic Context Recall：0.70 → 0.82。

Cross-encoder 是精确页排序提升的主要来源，也会使 Dense、Hybrid 和 Mix 的
最终 Top-8 更相似。FP16 合规运行中，E5 的 Hybrid 与 Mix 平均延迟分别为
2.600 秒和 2.984 秒。

## 限制

- 60 题适合作为开发集和工程回归，不足以支持跨语料的普遍结论。
- multi-context 只有 10 题，区间较宽。
- 题集与被比较策略共享同一开发语料；尚无独立 held-out 题集。
- Hugging Face 模型 revision 未由项目配置显式锁定。
- 本报告只评估检索与交付上下文，不等同于最终答案质量。
- 端到端回答、引用语义正确性、拒答和 Agentic 轨迹需单独运行，并会产生外部
  LLM 成本。

## 复现

五个本地完整报告位于 `artifacts/evaluations/`，默认不提交。正式运行均使用：

```powershell
$env:HYBRID_RAG_RETRIEVAL_RERANKER_USE_FP16 = "true"

uv run python scripts/evaluate_retrieval.py `
  --testset artifacts/ragas/rag-papers-benchmark-v2.json `
  --db storage/app.db `
  --profile idx_5e96c37b0d9f624d2d09 `
  --modes hybrid,mix `
  --top 8 `
  --rerank `
  --semantic-threshold 0.75 `
  --output artifacts/evaluations/benchmark-v3-e5-hybrid-vs-mix-rerank.json
```

各实验产物 SHA-256：

| 实验 | SHA-256 |
|---|---|
| E1 | `73adbc262f467f2884cbe5744f273ed3dfaf4a1e15587fb7ef98a5647faee447` |
| E2 | `3fc8f531a682f708c2bbbc94ddc22f92ff56e92132817820e89d3d9600a131b1` |
| E3 | `68f4dad6061a9a548159ddc257ddb7059ac8f8345c69d6a31115c358d8620184` |
| E4 | `56b6a482290765aa34218d238e32caf9358c2eb725469be2f92f03fbfcd148de` |
| E5 | `00907dd06419a0566b9a0fa210fabafc9a8547c12550084c2eb3b101c2c6c141` |

