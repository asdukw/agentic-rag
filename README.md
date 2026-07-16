# Hybrid RAG Lab

一个以学习和演示为主的 Hybrid/Graph RAG 项目。Web UI 由用户上传自己的文档；仓库内的小型 fixture
语料仅用于 CLI 开发和演示。项目展示从文档导入、索引与可选知识图谱构建，到检索、回答和 citation
追溯的完整效果。
不以生产部署或通用知识库能力为主要目标。

```text
论文 PDF → 文本与 chunk → 可选知识图谱 → 三路索引 → 检索/回答 → citation 与 replay trace
```

项目自身实现实体归一、关系合并、图谱检索、融合评分、上下文裁剪和引用追踪；第三方库只
承担解析、持久化、编排或模型 API 适配。

## Quick Start

需要 Python 3.11--3.13、[uv](https://docs.astral.sh/uv/) 和 [Bun](https://bun.sh/)。下面的 CLI 演示使用
仓库内的 `tests/fixtures/corpus`；实际使用请通过 Web UI 上传自己的语料。首次建立索引时会下载本地 BGE-M3
模型权重，因此需要网络，但不需要 embedding API key。默认的 `ask` 使用 Agentic + DeepSeek，需先配置
`DEEPSEEK_API_KEY`；CLI 默认数据库为 `storage/app.db`。

```bash
uv sync --dev

# 导入内置 fixture 语料并切成可追溯的 chunks。
uv run hrag ingest tests/fixtures/corpus

# 可选：抽取实体和关系（需要 DEEPSEEK_API_KEY）。
# 只使用 naive，或只需 mix 的 chunk 路径时可跳过这一步。
uv run hrag build-graph --limit 10

# 用默认的本地 BGE-M3 embedding 为 chunk 建立语义索引。
uv run hrag build-index

# 默认由 DeepSeek planner 调用受限工具、选择证据并生成引用受约束的回答。
uv run hrag ask "How does mix retrieval combine evidence?"
```

也可用准备脚本串行执行 ingest、构图与建索引，并打印 Ragas 所需的 `corpus_content_hash`。语料未变化且
数据库中已有 graph-backed active index 时，脚本会跳过构图和 embedding，直接读取已有 profile：

```bash
uv run scripts/prepare_corpus.py
```

完成后可在输出中查看回答所依据的 citation，用于观察 RAG 的检索与证据归因效果。fixture 语料较小，
适合快速验证命令流程，不代表论文语料上的演示效果。

- `ingest`、`build-index` 等命令会自动升级 SQLite schema。
- 默认 embedding 是 FlagEmbedding 的 `BAAI/bge-m3` dense vector（1024 维、最长 8192 tokens）；
  它在本机运行，首次使用会下载模型，后续优先使用本地 Hugging Face snapshot，不再为已缓存模型执行
  远端元数据检查。CPU 环境保持 `EMBEDDING_USE_FP16=false`，有兼容 CUDA GPU 时可设为 `true` 加速。
  修改 embedding 的模型、维度、最大长度或精度后，必须重新运行 `build-index`。
- `hash-token-v1` 仅保留给已有 profile 兼容和快速离线测试，不再作为 CLI 或演示默认值。
- `naive` 同时使用 chunk 的 dense 向量分数和本地 BM25 词法分数，并分别归一化后融合；BM25
  直接读取已索引的 chunk 文本，不需要额外模型、服务或重建索引。
- 融合后的候选默认保留一阶段顺序，不运行二阶段精排。需要更高候选质量时，可显式设置
  `HYBRID_RAG_RETRIEVAL_RERANKER_PROVIDER=flagembedding`，使用本地 cross-encoder
  `BAAI/bge-reranker-v2-m3` 批量计算 `[query, passage]` 对的相关性，再决定最终 Top-K。

### Agentic RAG Web 工作台

Web 工作台使用 Bun + React/TypeScript 展示 Agent 的规划、工具调用、证据集和最终引用；Python 是唯一的
Agent 主循环与安全边界。模型只能在受限工具中选择下一步，不能访问 SQLite、文件、embedding 向量、API key，
也不能触发 ingest、构图或建索引等副作用操作。

```bash
# 终端 1：Python Agent API。DeepSeek 密钥只由此进程从 .env 读取。
uv run hrag serve

# 终端 2：Bun + React 界面。
cd web
bun install
bun run dev
```

浏览器打开 Bun 输出的本地地址即可。Web 默认启用 DeepSeek planner，以严格 JSON action 在每轮选择一个工具，
或为比较题、跨文档题拆分 2～3 个并行检索 Worker；
需要离线调试时可取消页面中的 DeepSeek 选项。当前工具为：chunk dense/BM25 检索、实体检索、关系检索、最多两跳的
图扩展、受 token 预算限制的证据读取，以及只基于已读 chunk 的回答。每次 run 会将完整事件记录写入
`artifacts/agent-runs/`；Agent 中间搜索保持只读，不再为每个 Worker 单独写 `rtr_` trace。

Web UI 不再使用共享的 `storage/app.db` 或项目级语料目录。先在页面创建本地 workspace，再上传 PDF、
Markdown 或 TXT；每个 workspace 都拥有自己的 `uploads/`、`workspace.db` 和 LangGraph checkpoint，位于
`storage/workspaces/<workspace-id>/`。在页面按“导入文档 → 构建知识图谱 → 构建 / 复用索引”顺序操作；构图
会调用 DeepSeek，因此始终由用户手动触发。不同 workspace 的语料、图谱和索引完全隔离。

实体、关系与图路径只能作为检索线索，不能直接作为事实引用。最终回答的 citation 只能指向本次 Agent run
实际读取的原始 chunk；没有足够证据时系统会返回 `insufficient_evidence`。

## 项目功能

### FlagEmbedding cross-encoder 精排

默认关闭二阶段精排，以免 CPU 环境下的 cross-encoder 显著拖慢检索。需要时先启用
`BAAI/bge-reranker-v2-m3`；`uv sync` 会安装 FlagEmbedding，首次检索会下载相应权重：

```bash
export HYBRID_RAG_RETRIEVAL_RERANKER_PROVIDER=flagembedding
uv run hrag retrieve "How does LightRAG use entities?" --mode mix --json
```

首次检索会下载模型权重。实现会批量计算 `[query, passage]` 对的交叉编码器分数，并在 trace 中保留
原始 logit 与 sigmoid 归一化后的 0--1 分数。仅在兼容 CUDA 的 GPU 环境中将
`HYBRID_RAG_RETRIEVAL_RERANKER_USE_FP16=true`；CPU 环境保持 `false`。将 provider 恢复为 `none`
即可跳过所有二阶段精排。

### 构建知识图谱

配置 `DEEPSEEK_API_KEY` 后，抽取实体与关系；完成后重新建索引以纳入 entity/relation 向量：

```bash
uv run hrag build-graph --limit 10
uv run hrag build-index
```

构图默认会重试历史上失败的抽取；仅在明确需要跳过它们时使用 `--no-retry-failed`。

没有图谱时，`naive` 可正常使用，默认的 `mix` 仍会保留 chunk 路径；`hybrid` 仅组合图谱的
`local` 与 `global` 路径，因此可能没有结果。构图后，`local`、`global`、`hybrid` 与 `mix`
都能提供实体、关系和 NetworkX 路径证据。

### DeepSeek 人民币成本估算

DeepSeek 的上游响应提供 token 用量而非账单金额。项目会用响应中的实际模型、缓存命中输入、
缓存未命中输入和输出 token，结合 `.env` 中的六项单价估算人民币成本；`build-graph`、
`retrieve --deepseek`、默认启用 DeepSeek 的 `ask` 与 `evaluate` 的 JSON/trace/报告都会保留可观测的
用量和估算结果。Ragas 当前不暴露评审模型 usage，因此 `evaluate` 的 judge 与 total 成本会明确标为
`unknown`，不会猜测为缓存未命中或零成本。

模型仅按实际响应中的 `deepseek-v4-flash` 或 `deepseek-v4-pro` 精确匹配价格表。若上游没有返回两类
缓存 token，或两者之和与输入 token 不一致，成本状态会显示为 `unknown`，而不会猜测为缓存未命中；
因此金额是基于 response usage 的估算，不是 DeepSeek 账单。

### 选择检索模式与查看证据

```bash
uv run hrag retrieve "How does LightRAG use entities?" --mode mix --json
uv run hrag retrieval replay rtr_<id> --json
```

`retrieve` 可选 `--mode naive|local|global|hybrid|mix`，默认 `mix`：`naive` 是项目扩展的 chunk dense +
BM25 召回；`local` 从实体命中扩展图邻居；`global` 从关系命中汇聚证据；`hybrid` 只组合
`local + global`；`mix` 以 LightRAG 的语义组合 `naive + local + global`。复合模式按来源轮询
候选并按 chunk ID 去重，之后统一精排和裁剪。每次检索都会产生可 replay 的 `rtr_` trace；其中保留
naive 的 dense/BM25 分项、路由贡献、reranker 候选池和最终名次。

`ask` 默认使用 `--mode agentic --deepseek --no-rerank`：Planner 跨查询和工具选择最终 evidence，并可将独立
检索任务 fork 给隔离 Worker 并行执行；工具返回结果默认保留一阶段顺序，需要提高单次工具候选质量时可加
`--rerank`。固定链路仍可用
`--mode naive|local|global|hybrid|mix`，离线调试可用 `--no-deepseek`。
Agentic 的 `search_chunks` 子策略使用 `dense`、`bm25`、`dense_bm25` 三个名称；其中
`dense_bm25` 是 chunk 内部的语义与词法融合，不等同于顶层图谱模式 `hybrid`。
Planner 每轮还会收到当前 profile 的 chunk/entity/relation 数量：关系、组件拓扑、依赖和多跳路径问题在
图谱可用时可以选择实体/关系检索与图扩展。工具描述提供适用场景，但不强制路由，最终由 Main Agent 根据
问题、既有结果与当前索引能力自主选择。

### 使用 Ragas 评测

`evaluate` 是统一评测入口。测试集由项目自己的轻量生成器创建；Ragas 只负责 faithfulness、factual
correctness、context precision 和 context recall。项目另外计算 Hit@k、Recall@k、MRR、nDCG；
Agentic 模式还记录工具调用、证据利用、引用有效性、延迟和拒答准确率。回答、测试集生成和 Ragas
指标评审都会调用 DeepSeek，因此需要 `DEEPSEEK_API_KEY`。

先对待评测语料完成 `ingest` 与 `build-index`。`build-index --json` 会输出当前 index profile 的
`corpus_content_hash`；将该 64 位小写十六进制值原样传给生成脚本。该值由实际导入的 document/chunk
身份和内容生成，不能用 PDF 文件哈希、图谱 hash 或自行编造的值替代。

```bash
# 对指定 workspace 建立或刷新待评测的索引，并记录输出中的 corpus_content_hash。
uv run hrag build-index \
  --db storage/workspaces/<workspace-id>/workspace.db \
  --json

# 默认按 hash 从配置的 SQLite ready profile 读取实际索引的 normal chunk，生成 60 条测试，
# 并写入 artifacts/ragas/ragas-testset.json。
# 分布为 30 single-hop、10 summary/reasoning、10 multi-context、10 unanswerable；
# 每篇论文至少覆盖 5 条。生成过程不调用 Ragas TestsetGenerator。
# DEEPSEEK_API_KEY 是必需的。
uv run scripts/ragas_testset.py \
  --corpus-content-hash <build-index输出的corpus_content_hash>

# Workspace 索引需同时指定数据库；source-dir 只提供 SOURCES.json 出处声明。
uv run scripts/ragas_testset.py \
  --db storage/workspaces/<workspace-id>/workspace.db \
  --source-dir storage/workspaces/<workspace-id>/uploads \
  --testset-size 60 \
  --corpus-content-hash <build-index输出的corpus_content_hash> \
  --output artifacts/ragas/workspace-ragas-testset.json

# 低成本冒烟生成：默认只读索引中的 2 篇、每篇前 6 个 normal chunk，并生成 6 条 case。
uv run scripts/ragas_smoke.py \
  --corpus-content-hash <build-index输出的corpus_content_hash>

# 默认在相同测试集与索引上比较 agentic、mix 和 naive。
uv run hrag evaluate \
  --db storage/workspaces/<workspace-id>/workspace.db \
  --testset artifacts/ragas/my-ragas-testset.json

# 全量评测前先做六模式冒烟：确定性抽取 3 条 single-hop，以及各 1 条
# summary/reasoning、multi-context、unanswerable；报告保留原测试集 case index。
uv run hrag evaluate \
  --testset artifacts/ragas/ragas-testset-1.json \
  --modes naive,local,global,hybrid,mix,agentic \
  --smoke \
  --output artifacts/evaluations/six-modes-smoke.json

# 推荐用下面两个包装脚本做成对实验。两者默认均为六模式、六条 smoke case；
# 前者强制关闭所有 rerank，后者为固定模式和 Agentic 工具同时启用 FlagEmbedding。
uv run scripts/evaluate_no_rerank.py \
  --testset artifacts/ragas/ragas-testset-1.json

uv run scripts/evaluate_rerank.py \
  --testset artifacts/ragas/ragas-testset-1.json

# 确认 smoke 正常后，显式添加 --full 才会评测完整测试集。
uv run scripts/evaluate_no_rerank.py \
  --testset artifacts/ragas/ragas-testset-1.json \
  --full

# 可选：在同一测试集和同一索引上比较多个模式。
uv run hrag evaluate \
  --db storage/workspaces/<workspace-id>/workspace.db \
  --testset artifacts/ragas/my-ragas-testset.json \
  --modes agentic,mix,naive

# 无需调用模型，检查每篇论文是否至少覆盖 5 条。
uv run scripts/ragas_coverage.py --strict --min-questions-per-document 5

```

测试集必须是以下 envelope；裸 JSON 数组会被拒绝：

```json
{
  "schema_version": "2",
  "corpus_content_hash": "<64位小写十六进制hash>",
  "sources": [
    {
      "source_uri": "file:corpus/example.pdf",
      "filename": "example.pdf",
      "title": "论文标题",
      "authors": ["作者"],
      "venue": "CVPR",
      "year": 2026,
      "pages": "1-10",
      "official_url": "https://openaccess.thecvf.com/...",
      "rights_notice": "Copyright remains with the applicable rights holders."
    }
  ],
  "cases": [
    {
      "user_input": "问题",
      "reference": "参考答案",
      "reference_contexts": ["生成该题时使用的参考上下文"],
      "evidence_ids": ["evd_稳定证据ID"],
      "context_evidence_ids": ["evd_稳定证据ID"],
      "document_ids": ["doc_文档ID"],
      "question_type": "single_hop",
      "answerable": true,
      "evidence_quotes": ["来自原文的逐字证据"],
      "generator_model": "deepseek-v4-flash",
      "prompt_version": "1",
      "review_status": "unreviewed"
    }
  ]
}
```

生成脚本复用 ingest 的 loader、清洗和 chunk 质量分类，只从 `normal` segment 选择证据，递归支持
`.pdf`、`.md`、`.markdown` 与 `.txt`。默认生成 60 条，并在调用模型前确定题型分布和每篇论文覆盖率；
可用 `--testset-size`、`--min-cases-per-document`、`--max-documents` 或
`--max-segments-per-document` 调整范围。模型输出经过严格 JSON、逐字 evidence quote、证据归属和重复问题
校验；失败原因会写入修复 prompt，默认最多重试两次。`ragas_smoke.py` 复用相同实现，但使用低成本默认值和
`artifacts/ragas/ragas-smoke-testset.json` 输出。若目标文件已存在，两个入口都会自动追加 `-1`、`-2`
等序号选择新文件，不覆盖已有评测集。`data/corpus` 中的论文文件仅供本地使用并由 Git 忽略；仓库只提交
`SOURCES.json`。生成器要求每个实际载入文档都存在对应来源记录，并将这些记录写入公开测试集及评测报告的
provenance。公共 helper 位于
`hybrid_rag.evaluation.testset`：`load_evaluation_documents`、`plan_golden_cases`、
`generate_golden_cases`、`build_evaluation_testset_envelope` 与 `write_evaluation_testset`。无论选取多少文件，
`corpus_content_hash` 都必须对应实际用于 `evaluate` 的完整 index profile。语料内容、导入/分块配置或
待评测 profile 改变后，重新执行 `ingest`、`build-index`，取新的 hash 并重新生成测试集。默认结果写入
`artifacts/evaluations/ragas-<测试集文件名>.json`；默认比较 `agentic,mix,naive`。测试集文件是本地生成物，
不是仓库提供的通用测试集；必须评测由自己当前 workspace profile 语料 hash 绑定的文件。评测报告还会
记录 Agentic 工具时间线、证据与引用指标、测试集文件 SHA-256、锁定的 profile、检索参数，以及模型的
非敏感运行配置，便于区分同一语料上的不同题集或执行条件。

### 上传用户语料

产品语料由用户在 Web UI 中上传，不提供内置论文下载命令。启动服务和页面后，创建 workspace，上传
PDF、Markdown 或 TXT，再依次点击“导入文档 → 构建知识图谱 → 构建 / 复用索引”。每个 workspace 使用
独立的数据库和上传目录，互不影响。

```bash
uv run hrag serve
cd web
bun install
bun run dev
```

命令行的 `ingest <目录或文件> --db <SQLite文件>` 仍保留，供开发、自动化或批处理使用；它不会下载任何语料。

### 配置 DeepSeek

本项目的 embedding 固定使用本地 FlagEmbedding 模型；查看 [.env.example](.env.example) 可配置
DeepSeek 模型、API 地址和 Ragas 评审模型。

## 工程保证与边界

- document/chunk ID、处理配置、图谱来源和 index profile 均可追溯。
- SQLite 外键和事务保证文档更新不会留下孤儿 chunk。
- 模型抽取需经过 Pydantic、局部引用和原文证据校验；每条实体/关系保留来源 chunk。
- profile 绑定 embedding 配置、语料和图谱快照；历史 `rtr_` trace 可离线 replay。
- Ragas 评测会锁定 profile 并校验测试集的语料指纹；结果只适用于声明的语料、索引和模型配置。

当前限制：默认 PDF adapter 面向有文本层的论文，不做扫描件 OCR；复杂版面、表格和公式结构可在
既有 loader 接口后接入 Docling。在线模型的质量、延迟和费用需要在冻结语料和明确配置下单独评测。

## 更多资料

- [架构图](docs/architecture.md)
- [ADR 001：自研与复用边界](docs/adr/001-build-vs-reuse.md)
- [ADR 002：抽取与恢复语义](docs/adr/002-graph-extraction-checkpoints.md)
- [ADR 003：索引与 trace](docs/adr/003-retrieval-index-and-trace.md)
