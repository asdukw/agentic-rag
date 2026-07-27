# Agentic RAG Lab

一个以“有界规划、证据约束、过程可审计”为核心的 Agentic RAG 学习项目。

项目借鉴 GraphRAG 的图谱建模与多跳取证思想，以及 LightRAG 的实体/关系双层索引和轻量化检索设计，从文档导入、知识图谱抽取和多路检索出发，将 Chunk 检索、实体检索、关系检索、图扩展、证据读取和回答封装为 Agent 工具，最终实现一个可分解问题、动态规划检索路径、受预算约束并输出可验证引用的 RAG Agent。

Hybrid 与 Graph 检索不是项目终点，而是 Agentic RAG 的工具底座。项目重点不是复刻某个框架或追求单一最高分，而是展示如何设计 Planner/Tool 边界、证据契约、预算控制、运行轨迹和可复现评测。

> 定位：Agentic RAG 方法论与工程实践作品，不是生产级通用知识库。

[Benchmark v3](docs/benchmark-v3.md) · [机器可读摘要](docs/benchmark-v3-summary.json) · [完整架构](docs/architecture.md) · [设计边界](docs/adr/001-build-vs-reuse.md)

## 检索底座结果

Benchmark v3 验证的是 Agent 可调用的检索工具底座，而不是 Agentic 模式的端到端收益。实验已在 clean commit 上完成预注册的 E1–E5 retrieval-only 矩阵；在 10 篇 RAG 论文、60 道人工审校题、Top-8 和同一 CUDA FP16 reranker 配置下，主实验 E5 的结果如下，相关性指标只聚合其中 50 道可回答题。

| 指标 | hybrid | mix | mix - hybrid |
|---|---:|---:|---:|
| Exact-page Raw Recall@8 | 0.730 | 0.720 | -0.010 |
| Exact-page Context Recall@8 | 0.700 | 0.710 | +0.010 |
| Document Context Recall@8 | 0.880 | 0.950 | **+0.070** |
| Semantic Context Recall | 0.760 | 0.820 | **+0.060** |
| Context NDCG@8 | 0.635 | 0.655 | +0.020 |
| 平均延迟 | 2.600s | 2.984s | +0.384s |

在 10 道 multi-context 题上，`mix` 的优势更集中：

| 指标 | hybrid | mix |
|---|---:|---:|
| Exact-page Raw Recall@8 | 0.050 | **0.150** |
| Document Context Recall@8 | 0.600 | **0.750** |
| Semantic Context Recall | 0.300 | **0.400** |

这个结果没有被包装成“mix 全面胜出”：强 reranker 会让高度重叠的候选池趋同，因此 `mix` 的价值主要体现在跨文档覆盖和最终交付上下文，而不是每道单跳题都提高精确页命中。E1–E3 还表明，纯 Dense 明显优于纯 BM25；无 reranker 时，当前等权 Hybrid 弱于 Dense，而启用 cross-encoder 后两者的 Exact-page Context Recall@8 同为 0.700。完整配置、逐题胜负、paired bootstrap 区间和局限见 [Benchmark v3](docs/benchmark-v3.md)。

运行环境：RTX 5070 Laptop GPU，`torch 2.13.0+cu130`，BGE-M3 embedding 使用 FP32，bge reranker 使用 FP16。本表对应的 retrieval-only 评测不调用外部 LLM，API 成本为 ¥0。

## Agentic RAG 核心能力

- **受限 Planner/Tool 循环**：Planner 只能选择项目定义的只读工具，包括 Chunk、Entity、Relation 检索、图扩展、证据读取和证据回答，不能绕过工具边界直接访问数据。
- **动态取证而非固定流水线**：Agent 根据问题、已发现证据和图前沿决定下一步；复杂问题可拆分为多个子任务，并交给 2～3 个隔离 worker 并行检索。
- **显式预算与停止条件**：step、search、read、graph expansion、graph hop、并发数和 evidence token 均有上限，重复工具调用会被拦截，预算耗尽时安全停止。
- **证据是一等状态**：检索到的 Chunk 必须先被 Agent 显式读取才能进入回答上下文，最终 citation 只能来自本次运行实际读取的证据白名单。
- **轻量图谱工具底座**：借鉴 GraphRAG 与 LightRAG，为 Chunk、Entity、Relation 建立三类索引；实体用于局部定位，关系用于高层语义入口，图扩展用于有界多跳补证，不依赖社区报告。
- **可恢复、可审计运行**：构图使用 LangGraph checkpoint 支持中断恢复；Agent 通过 SSE 输出 planner action、tool result、answer 和 termination reason，并将完整报告落盘。
- **检索与回答可回放**：稳定 evidence ID、索引 profile、语料 hash、候选与重排分数、模型用量和 retrieval trace 支持审计与 replay。
- **以评测约束设计**：区分 raw candidates 与 delivered context，并报告 exact-page、document-level、semantic-evidence、延迟、成本和运行环境 provenance。

项目没有训练或微调 embedding、reranker、LLM，也没有执行网格搜索等系统化超参数优化；检索配置是固定的工程默认值。

## 架构

```mermaid
flowchart LR
    subgraph Entry["交互入口"]
        CLI["Typer CLI"]
        UI["React Workbench"] --> API["FastAPI + SSE"]
    end

    subgraph Offline["离线建库"]
        Docs["PDF / Markdown / TXT"] --> Ingest["解析 · 清洗 · 章节感知分块"]
        Ingest --> Corpus[("Documents + Chunks<br/>Stable IDs · Content Hash")]
        Corpus --> Extract["可恢复图谱抽取<br/>校验 · 补漏/修复 · 审核"]
        Extract --> Graph[("Entities · Relations · Evidence")]
        Corpus --> Index["BGE-M3 三路索引"]
        Graph --> Index
        Index --> Profile[("Pinned Index Profile")]
    end

    subgraph Online["Agentic 在线取证"]
        Question["Question"] --> Planner["Bounded Planner"]
        Planner --> Tools["Read-only Tools<br/>Chunk · Entity · Relation · Expand · Read"]
        Profile --> Tools
        Tools --> State["Evidence + Graph Frontier<br/>Budget State"]
        State --> Planner
        State --> Answer["Evidence-only Answer<br/>Citation Allowlist"]
        Planner --> Trace[("Agent Events + Replayable Trace")]
        Tools --> Trace
        Answer --> Trace
    end

    CLI --> Docs
    CLI --> Question
    API --> Docs
    API --> Question
```

业务数据、索引 profile 与 trace 位于 workspace 级 SQLite；LangGraph checkpoint 使用独立 SQLite，只保存可恢复工作流状态。第三方库承担解析、持久化、编排和模型适配，稳定 ID、规范化、融合、预算、引用与评测契约由项目自身控制。

## 检索模式

| 模式 | 召回入口 | 适合问题 |
|---|---|---|
| `dense` | chunk dense vector | 纯语义检索基线 |
| `bm25` | deterministic BM25 | 纯关键词检索基线 |
| `hybrid` | dense + BM25 加权融合 | 行业标准 Hybrid Search |
| `graph_local` | entity vector → graph neighbors → source chunks | 实体、方法、数据集或局部关系 |
| `graph_global` | relation vector → graph paths → source chunks | 主题、关系、总结和全局问题 |
| `graph_hybrid` | graph_local + graph_global 加权融合 | 只使用图谱侧的组合检索 |
| `mix` | hybrid + graph_local + graph_global 加权融合 | 固定链路下的通用默认组合 |
| `agentic` | 有界 Planner 动态选择上述只读工具 | 需要分解、跨文档或迭代取证的问题 |

实体、关系和图路径只作为检索线索，最终 citation 必须回到原始 chunk。

## 5 分钟本地体验

需要 Python 3.11–3.13 和 [uv](https://docs.astral.sh/uv/)。以下流程使用仓库内两个小型 fixture，不需要 API key：

```bash
uv sync

uv run hrag ingest tests/fixtures/corpus --db storage/portfolio-demo.db

uv run hrag build-index --db storage/portfolio-demo.db

uv run hrag retrieve "How does dense retrieval differ from hybrid search?" --db storage/portfolio-demo.db --mode hybrid --json
```

首次建立索引会下载本地 `BAAI/bge-m3` 权重，不需要 embedding API key。Windows/Linux 默认安装 PyTorch CUDA 13.0 build；检测到兼容 GPU 时使用 CUDA，否则回退 CPU。macOS 使用普通 PyTorch build。

### 启用图谱与回答

复制 [.env.example](.env.example) 为 `.env`，配置 `DEEPSEEK_API_KEY` 后执行：

```bash
uv run hrag build-graph --db storage/portfolio-demo.db --limit 10

uv run hrag build-index --db storage/portfolio-demo.db

uv run hrag ask "How do graph routes complement chunk retrieval?" --db storage/portfolio-demo.db --mode agentic
```

图谱构建可中断恢复；命令会输出 run ID，之后可用 `hrag build-graph --resume <run-id>` 继续。默认每个 chunk 最多调用模型两次：有效初始抽取进入一次独立、可审计的补漏，无效初始抽取则把第二次调用用于修复，两者互斥。构图会报告孤立实体但不自动删除，因为它们仍可作为 `graph_local` 的 entity → chunk 召回入口。

### 启动 Web 工作台

需要 [Bun](https://bun.sh/)：

```bash
# 终端 1：Python API
uv run hrag serve

# 终端 2：React/Vite UI
cd web
bun install --frozen-lockfile
bun run dev
```

在浏览器中创建 workspace，上传 PDF、Markdown 或 TXT，再按“导入文档 → 构建知识图谱 → 构建索引”执行。每个 workspace 拥有独立 uploads、业务数据库和 LangGraph checkpoint；DeepSeek key 只存在于 Python 进程，不会传给浏览器。

## 评测方法

v2 题集由 v1 的 60 道自动生成题逐题审计得到：21 道接受、39 道重写、0 道删除，最终包含：

- 30 道 single-hop
- 10 道 summary/reasoning
- 10 道 multi-context
- 10 道 unanswerable

评测固定 index profile 和 corpus hash，交替执行待比较模式，避免运行顺序偏差。指标分为五层：

1. **Exact-page**：是否命中人工指定的文档页/section evidence ID。
2. **Document-level**：是否覆盖目标文档，减少 chunk 边界对结果的干扰。
3. **Semantic-evidence**：检索上下文是否在语义上覆盖参考证据。
4. **Multi-evidence**：所有必要证据是否同时到齐、目标文档覆盖率和来源数。
5. **Answer behavior**：claim precision/recall/F1、faithfulness、答案级引用支持与覆盖，以及拒答混淆矩阵。

前四个检索层又区分 Raw Top-K 与 token budget 后真正交给回答器的 Delivered Context。60 题适合工程回归和学习项目展示，但不足以支持“全面显著领先”的强结论。

当且仅当比较两个模式时，retrieval-only 报告会自动给出逐题胜/平/负、按题型拆分的 paired delta，以及固定 seed、20,000 次 paired percentile bootstrap 的 95% 区间。Claim precision/recall 使用相同 judge 配置分别评分，F1 是二者的调和平均。引用评测遵循当前“整份答案共享一组 citation IDs”的协议：correctness 判断回答 claims 是否被引用上下文集合支持，completeness 衡量引用集合对 gold exact-page evidence 的覆盖；不宣称逐句 inline attribution accuracy。

模式名称本身就是可配对的实验条件：`--modes dense,hybrid` 直接测量加入 BM25 的增量，`--modes hybrid,mix` 测量继续加入图谱路由后的增量；两组都可统一选择是否启用 reranker。

评测入口和复现配置见：

- [Benchmark v3：E1–E5 结果、配对区间与局限](docs/benchmark-v3.md)
- [Benchmark v3 预注册方案：实验矩阵、指标口径与判定规则](docs/benchmark-v3-plan.md)
- [Benchmark v3 机器可读结果摘要](docs/benchmark-v3-summary.json)
- [Benchmark v2：历史结果与改造过程](docs/benchmark-v2.md)
- [`scripts/evaluate_retrieval.py`](scripts/evaluate_retrieval.py)：无需外部 LLM 的 retrieval-only 配对评测
- [`scripts/curate_rag_benchmark.py`](scripts/curate_rag_benchmark.py)：v1 → v2 的确定性人工审计决策

论文原文、生成题集、SQLite 和大型 JSON trace 属于本地实验产物，由 Git 忽略；仓库保留评测代码、方法、机器可读结果摘要和产物 SHA-256。因此 fresh clone 可以在 fixture 或自有语料上复现评测流程，但不能仅凭公开仓库重建这次论文 benchmark 的完整逐题结果。

## 工程边界

- 支持带文本层的 PDF、Markdown 和 TXT；不包含扫描件 OCR。
- DeepSeek 只用于可选的图谱抽取、Planner、证据约束回答和 Ragas judge；embedding 与 reranking 均可本地运行。
- 默认关闭 cross-encoder reranker，方便 CPU 环境学习和调试；可在 `.env` 中启用 `flagembedding` provider。
- Web workspace 面向本地单用户演示，不以多租户生产部署为目标。
- Benchmark v3 是 retrieval-only；完整端到端答案质量仍需在明确模型、预算和语料版本下单独评测。

## 项目结构

```text
src/hybrid_rag/
├── ingest/       # loader、清洗、章节感知分块与稳定 ID
├── extraction/   # 图谱抽取、校验、修复与规范化
├── retrieval/    # Dense、BM25、图谱召回、融合、rerank 与 trace
├── agentic/      # 有界 Planner、worker、工具与审计事件
├── evaluation/   # evidence contract、retrieval metrics 与 Ragas
├── storage/      # SQLite repository 与 migrations
├── runtime.py    # FastAPI 工作台共享的模型与服务工厂
└── web_api.py    # React/TypeScript 工作台的 HTTP + SSE 边界

web/              # React + TypeScript + Vite 工作台
scripts/          # 语料准备、题集审计和 benchmark 入口
docs/              # 架构、ADR 与实验报告
```

## 延伸阅读

- [完整架构与持久化边界](docs/architecture.md)
- [Benchmark v1：问题是怎样暴露出来的](docs/benchmark-v1.md)
- [Benchmark v3：完整检索消融、收益与代价](docs/benchmark-v3.md)
- [Benchmark v2：改造过程与历史结果](docs/benchmark-v2.md)
- [ADR 001：自研与复用边界](docs/adr/001-build-vs-reuse.md)
- [ADR 002：图谱抽取与恢复语义](docs/adr/002-graph-extraction-checkpoints.md)
- [ADR 003：索引、失效与 trace](docs/adr/003-retrieval-index-and-trace.md)
