# Benchmark v2（2026-07-21）

> 这是命名修正前的历史报告。报告中的 `naive` 等价于当前 `hybrid`
>（Dense + BM25）；当时的 `hybrid` 已更名为 `graph_hybrid`。指标与原始 JSON
> 标签保持不变，下一轮 benchmark 将直接使用新命名重新生成。

## 结论

本轮已经完成融合排序、扩大候选池、multi-hop 补证、multi-context 覆盖约束、题集人工修订、reranker 与多口径指标评测。

- 在同一 CUDA 环境、同一 reranker 下，`mix` 不是所有指标都高于 `naive`：Exact-page Raw Recall@8 为 0.720 vs 0.730。
- `mix` 的优势主要体现在实际交付证据的跨文档覆盖与语义覆盖：Document Context Recall@8 为 0.950 vs 0.880，Semantic Context Recall 为 0.820 vs 0.760。
- 在 10 道 multi-context 题上，`mix` 的 Exact-page Raw Recall@8 为 0.150 vs 0.050，Document Context Recall@8 为 0.750 vs 0.600，Semantic Context Recall 为 0.400 vs 0.300。
- reranker 明显提高 exact-page 排序质量，但会压缩 `mix` 与 `naive` 在 exact-page 指标上的差距；`mix` 的差异化价值应表述为跨来源覆盖和交付上下文质量，而不是单跳页面命中全面领先。
- 60 题适合作为开发集、回归基准和求职项目证据，不足以支持强统计结论。50 道可回答题中多数配对结果相同，multi-context 只有 10 题。

该报告是 retrieval-only 基准，不等同于最终答案质量评测。完整端到端 RAGAS 运行仍受 DeepSeek 账户余额不足影响。

## 实现范围

1. `mix` 不再 round-robin 编排，改为三路分数归一化后的加权融合排序。
2. `naive`、`local`、`global` 各保留最多 32 个候选，融合后把 32 个候选交给 `BAAI/bge-reranker-v2-m3`，最终输出 Top-8。
3. 仅把至少两跳的图路径映射回来源 chunk，以 0.25 权重、有限候选数补入融合池。
4. 对显式比较型 multi-context 查询做确定性拆分；每个子查询单独检索，并建立跨文档 coverage anchors。rerank 后保留 anchors，上下文装配采用 coverage-first。
5. 将 v1 的 60 题逐题人工审计为 v2：21 题接受、39 题重写、0 题删除，60 题均标记为 `reviewed`。
6. 同时报告 exact-page、document-level 和 semantic-evidence 指标，并区分 Raw Top-K 与实际 Delivered Context。

## CUDA 验证

- GPU：NVIDIA GeForce RTX 5070 Laptop GPU，8,151 MiB，计算能力 12.0。
- 驱动：610.47；驱动支持 CUDA 13.3。
- Torch：`2.13.0+cu130`；`torch.version.cuda == 13.0`；`torch.cuda.is_available() == true`。
- 项目依赖已固定为 Windows/Linux 默认从 PyTorch 官方 `cu130` index 安装，避免后续 `uv sync` 恢复成 CPU wheel。
- CUDA 矩阵乘法烟测在 `cuda:0` 完成。
- BGE-M3 embedding 和 bge reranker 的参数设备均验证为 `cuda:0`。为保持既有索引口径，embedding 使用 FP32；reranker 使用 FP16。
- 双模型烟测峰值分配显存约 3.29 GiB；完整评测期间 `nvidia-smi` 观测约 6.7–6.9 GiB，GPU 峰值利用率 99–100%。
- rerank 报告内的 accelerator provenance 记录了 Torch CUDA build、CUDA 可用性和设备名称。

## 数据与口径

- 语料：10 篇 RAG 论文，467 个正常 chunk。
- 索引：`BAAI/bge-m3`，1,024 维；profile `idx_5e96c37b0d9f624d2d09`。
- 题集：60 题，其中 single-hop 30、summary-reasoning 10、multi-context 10、unanswerable 10。
- 相关性指标仅聚合 50 道可回答题；10 道不可回答题不进入 Hit/Recall/MRR/NDCG 聚合。
- Top K：8；每路候选上限与 rerank 候选上限：32；context budget：2,400 tokens；图最大跳数：2。
- 语义证据阈值：BGE-M3 cosine similarity >= 0.75。
- 查询关键词：`deterministic_keywords`；外部 LLM 调用为 0；本地评测 API 成本为 ¥0。

题集 SHA-256：`27C53590105D002D1E6FDB2B529242954B5C8423EE1791E005A6BA1EFD4663B2`。

## 同 GPU、启用 reranker 的配对结果

| 指标 | naive | mix | mix - naive |
|---|---:|---:|---:|
| Exact Raw Hit@8 | 0.760 | 0.740 | -0.020 |
| Exact Raw Recall@8 | 0.730 | 0.720 | -0.010 |
| Exact Raw MRR | 0.623 | 0.633 | +0.010 |
| Exact Raw NDCG@8 | 0.646 | 0.653 | +0.007 |
| Exact Context Recall@8 | 0.700 | 0.710 | +0.010 |
| Exact Context NDCG@8 | 0.635 | 0.655 | +0.020 |
| Document Raw Recall@8 | 0.980 | 0.980 | 0.000 |
| Document Context Recall@8 | 0.880 | 0.950 | +0.070 |
| Document Context NDCG@8 | 0.846 | 0.922 | +0.076 |
| Semantic Raw Recall | 0.800 | 0.850 | +0.050 |
| Semantic Context Recall | 0.760 | 0.820 | +0.060 |
| 平均延迟 | 2.951s | 3.344s | +0.393s |
| P95 延迟 | 3.357s | 3.914s | +0.558s |

逐题配对结果进一步说明了差异来源：

- Exact Raw Recall：`mix` 4 胜、42 平、4 负；平均差 -0.010。
- Document Context Recall：`mix` 7 胜、41 平、2 负；平均差 +0.070。
- Document Context NDCG：`mix` 10 胜、38 平、2 负；平均差 +0.076。
- Semantic Context Recall：`mix` 6 胜、42 平、2 负；平均差 +0.060。

因此，reranker 后两种模式对大部分题返回相同或等价证据。`mix` 的新增价值集中在少量需要补充第二文档或第二语义侧面的题，而不是每道单跳题都改变 exact page。

## multi-context 结果

| 指标 | naive | mix | mix - naive |
|---|---:|---:|---:|
| Exact Raw Recall@8 | 0.050 | 0.150 | +0.100 |
| Exact Context Recall@8 | 0.050 | 0.150 | +0.100 |
| Document Raw Recall@8 | 0.900 | 0.900 | 0.000 |
| Document Context Recall@8 | 0.600 | 0.750 | +0.150 |
| Document Context NDCG@8 | 0.552 | 0.739 | +0.186 |
| Semantic Raw Recall | 0.500 | 0.550 | +0.050 |
| Semantic Context Recall | 0.300 | 0.400 | +0.100 |

multi-context 的 exact-page 绝对值仍偏低。Trace 显示 cross-encoder 使用完整复合问题评分时，会把部分第一阶段已命中的第二证据页挤出 Top-8；coverage anchors 能保住目标文档，但未必命中人工指定的精确页。后续改进应在独立 held-out 题集上评估 subquery-aware reranking 或每个 aspect 的证据配额，不能根据当前 10 题直接调参。

## reranker 的收益与代价

下表比较同一 GPU 环境的无 rerank 与 rerank 结果。

| 模式 | 指标 | 无 rerank | rerank | 差值 |
|---|---|---:|---:|---:|
| naive | Exact Raw Recall@8 | 0.570 | 0.730 | +0.160 |
| naive | Exact Context Recall@8 | 0.490 | 0.700 | +0.210 |
| naive | Semantic Context Recall | 0.650 | 0.760 | +0.110 |
| naive | 平均延迟 | 1.744s | 2.951s | +69.2% |
| mix | Exact Raw Recall@8 | 0.620 | 0.720 | +0.100 |
| mix | Exact Context Recall@8 | 0.520 | 0.710 | +0.190 |
| mix | Semantic Context Recall | 0.700 | 0.820 | +0.120 |
| mix | 平均延迟 | 2.088s | 3.344s | +60.1% |

reranker 是本轮 exact-page 提升的主要来源。它也解释了为什么启用后 `mix` 的 exact-page 优势不明显：naive 的 32 个候选中通常已经包含目标文档，强 cross-encoder 会把两种第一阶段候选池重新排成相近的 Top-8。

## 60 题是否足够

60 题对求职项目是合适的第一版规模：题型结构明确、可以逐题人工复核、一次全量回归在消费级 GPU 上约 4–7 分钟，能展示可复现的工程闭环。

但它不适合宣称统计意义上的全面领先：

- 只有 50 题进入相关性指标。
- multi-context 只有 10 题，单题就会让 Hit 类指标变化 0.10。
- rerank 配对中 Exact Raw Recall 有 42/50 个平局。
- 20,000 次 paired bootstrap（固定种子 20260721）给出的 95% 区间为：Exact Raw Recall 差值 -0.010 `[-0.110, 0.080]`；Document Context Recall +0.070 `[0.000, 0.150]`；Semantic Context Recall +0.060 `[-0.020, 0.150]`；Document Context NDCG +0.076 `[0.022, 0.138]`。
- multi-context 的 Exact Raw Recall 差值 +0.100，95% bootstrap 区间仍为 `[0.000, 0.250]`。

建议把当前 60 题固定为 development/regression set，另增 100–140 道不参与调参的 held-out 题；其中至少 30–50 道 multi-context。对外陈述使用置信区间和逐题胜/平/负，而不是只报一个均值。

## 复现

以下命令用于作者本机的冻结语料与 profile。论文 PDF、v2 题集、SQLite 和完整 trace 报告未提交到仓库，因此 fresh clone 只能在自有语料上复现流程，不能仅靠公开文件重建本次逐题结果。原始报告还记录了 dirty working tree，且未锁定 Hugging Face revision；公开汇总不会将其表述为字节级可复现。

安装并验证 CUDA：

```powershell
uv sync
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

无 rerank、同 GPU 配对评测：

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
uv run python scripts/evaluate_retrieval.py `
  --testset artifacts/ragas/rag-papers-benchmark-v2.json `
  --db storage/app.db `
  --profile idx_5e96c37b0d9f624d2d09 `
  --modes hybrid,mix `
  --top 8 `
  --semantic-threshold 0.75 `
  --output artifacts/evaluations/rag-papers-retrieval-v2-no-rerank-gpu.json
```

启用 reranker 的同 GPU 配对评测：

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HYBRID_RAG_RETRIEVAL_RERANKER_USE_FP16 = 'true'
uv run python scripts/evaluate_retrieval.py `
  --testset artifacts/ragas/rag-papers-benchmark-v2.json `
  --db storage/app.db `
  --profile idx_5e96c37b0d9f624d2d09 `
  --modes hybrid,mix `
  --top 8 `
  --rerank `
  --semantic-threshold 0.75 `
  --output artifacts/evaluations/rag-papers-retrieval-v2-rerank.json
```

## 产物

- 可公开复核的机器可读汇总：[`benchmark-v2-summary.json`](benchmark-v2-summary.json)。
- 人工修订题集：`artifacts/ragas/rag-papers-benchmark-v2.json`，SHA-256 `27C53590105D002D1E6FDB2B529242954B5C8423EE1791E005A6BA1EFD4663B2`。
- 同 GPU 无 rerank 报告：`artifacts/evaluations/rag-papers-retrieval-v2-no-rerank-gpu.json`，SHA-256 `824931F69D4C7378668FB29F4F7D8FE0E19A58C27FE1A899915D4A70D7100316`。
- 同 GPU rerank 报告：`artifacts/evaluations/rag-papers-retrieval-v2-rerank.json`，SHA-256 `A3D8959BB6B21F555646E1FCC074F5297BC45B7A2BA10EE96C1D71DF0F3712AD`。
