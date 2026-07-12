# Hybrid RAG

一个面向求职展示的轻量 Graph-RAG 项目：复刻 LightRAG 的核心工程链路，同时保留
每一步检索和证据来源的可解释性。

当前主链路覆盖文档导入、图谱构建、多路检索，以及可复现的评测和演示：

```text
PDF / Markdown / TXT
  -> loader
  -> conservative cleaner
  -> section-aware, token-bounded chunker
  -> SQLAlchemy transaction
  -> SQLite documents + chunks + ingest_runs
  -> LangGraph: extract -> validate -> repair/review
  -> deterministic entity normalization + relation merge
  -> SQLite canonical graph + NetworkX MultiDiGraph
  -> chunk/entity/relation embedding texts + independent SQLite vector indexes
  -> naive/local/global recall -> hybrid fusion -> cited context/answer + replayable trace
  -> fixed benchmark -> blinded naive/hybrid comparison -> JSON/Markdown report
```

## Quick start

需要 Python 3.11-3.13 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync --dev
uv run hybrid-rag db upgrade
uv run hybrid-rag ingest tests/fixtures/corpus --db .tmp/demo.db
uv run hybrid-rag stats --db .tmp/demo.db
uv run pytest -q
uv run ruff check .
```

再次运行相同的 `ingest` 命令会报告 `skipped`，不会生成重复记录。修改一个源文件后，
只会在单个事务中替换该文档对应的 chunks。

下载版本固定的 10 篇公开 arXiv 论文并做真实 PDF smoke test（PDF 已被 Git 忽略）：

```bash
uv run hybrid-rag corpus download
uv run hybrid-rag ingest data/raw --db storage/app.db
uv run hybrid-rag stats --db storage/app.db
```

`data/corpus.json` 同时固定 arXiv revision 和 PDF SHA-256。2026-07-10 的实际验收结果：

| documents | chunks | contextualized tokens | max chunk tokens | failures |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 476 | 202,802 | 512 | 0 |

未修改数据的第二次导入为 `10 skipped`，预检耗时约 0.04 秒；实际数值可能随 parser、
cleaner 或 chunker 版本升级而变化，配置哈希会触发重建。

查看某篇文档的 chunk、章节、页码和字符偏移：

```bash
uv run hybrid-rag inspect <document-id> --db .tmp/demo.db
```

## 阶段二：抽取与构图

首次或存在未缓存 chunk 时，在 `.env` 中配置 DeepSeek key；不要提交真实密钥。全缓存
构建、只读检查以及无需再次调用模型的 resume 不要求 key：

```dotenv
DEEPSEEK_API_KEY=
HYBRID_RAG_GRAPH_CHECKPOINT_PATH=storage/langgraph.db
```

先导入文档，再从数据库中的 chunks 构图。建议首次用 `--limit` 做小批量检查：

```bash
uv run hybrid-rag build-graph --db .tmp/demo.db --limit 10
uv run hybrid-rag build-graph --db .tmp/demo.db --json
uv run hybrid-rag graph stats --db .tmp/demo.db
uv run hybrid-rag graph stats --db .tmp/demo.db --json
```

DeepSeek 请求使用 OpenAI-compatible API，显式关闭 Thinking，并启用 JSON Output。JSON
语法合法后仍需经过 Pydantic、局部实体引用和原文证据校验；项目代码再执行确定性的实体
归一与关系合并。每个规范实体和关系都保留来源 chunk ID、证据文本及可用的字符 span。

抽取结果按 chunk 持久化，每次实际请求作为 append-only attempt 留档。LangGraph 的运行
checkpoint 单独写入 `HYBRID_RAG_GRAPH_CHECKPOINT_PATH` 指向的 SQLite 文件。进程中断后，
使用报告中的 `gbr_` run ID 恢复；成功 chunk 不会再次请求模型：

```bash
uv run hybrid-rag build-graph --db .tmp/demo.db --resume gbr_<id>
uv run hybrid-rag build-graph --db .tmp/demo.db --retry-failed
```

单个 chunk 的终止失败不会回滚其他成功结果，但构建命令会以状态
`completed_with_failures` 和退出码 `1` 结束；正常完成或有意暂停等待复核时退出码为 `0`，
中断后退出码为 `130`。`--retry-failed` 才会重新入队已达到尝试上限的失败样本。

需要人工复核时，以 `--review` 构建；本次新抽取且验证通过的 build item 先进入
`needs_review`，
不会立即写入当前规范图。相同抽取配置下已验证的缓存默认继续受信任；如需重新审核全部
chunk，应变更抽取配置（例如使用新的 prompt/schema 版本）后创建新 run：

```bash
uv run hybrid-rag build-graph --db .tmp/demo.db --review
uv run hybrid-rag graph review xtr_<id> --decision approve
uv run hybrid-rag graph review xtr_<id> --decision reject --note "unsupported evidence"
uv run hybrid-rag build-graph --db .tmp/demo.db --resume gbr_<id>
```

同一 extraction 若同时等待多个 run 的独立复核，`graph review` 会要求额外传入
`--run gbr_<id>`，不会把一个 run 的决定传播到另一个 run。

`graph stats` 输出节点、边、weakly connected components 和 Top-K 实体。以下 ID 均可追踪：
run (`gbr_`)、chunk extraction (`xtr_`)、attempt (`xat_`)、entity (`ent_`) 和 relation
(`rel_`)：

```bash
uv run hybrid-rag graph inspect <object-id> --db .tmp/demo.db
uv run hybrid-rag graph inspect <object-id> --db .tmp/demo.db --raw
```

普通 inspect 会显示经过验证的 evidence quotes 以支持溯源；`--raw` 额外用于失败诊断，
可能显示完整 prompt、原始 chunk 和模型响应。认证信息不会持久化或输出。

## 阶段三：多路索引与检索

先在当前 chunks 和当前规范图上构建独立的 chunk、entity、relation 索引；索引 profile 会记录
embedding 配置、维度、图谱无关的 corpus-content hash、图谱绑定 snapshot hash 和图谱 run。profile
identity 同时绑定图谱 run，因此重建不同图谱不会覆写旧向量或旧 trace。默认 `hash-token-v1` 是确定性、无需
下载模型或联网的开发/CI baseline；它保证链路可复现，不应替代基准后选定的语义 embedding 模型。
可通过 `openai-compatible` adapter 接入经基准验证的外部 embedding endpoint。

```bash
uv run hybrid-rag build-index --db .tmp/demo.db
uv run hybrid-rag retrieval stats --db .tmp/demo.db

uv run hybrid-rag retrieve "How does LightRAG use entities?" --mode naive --db .tmp/demo.db
uv run hybrid-rag retrieve "How does LightRAG use entities?" --mode local --db .tmp/demo.db
uv run hybrid-rag retrieve "How does LightRAG use entities?" --mode global --db .tmp/demo.db
uv run hybrid-rag retrieve "How does LightRAG use entities?" --mode hybrid --db .tmp/demo.db --json

# 默认离线回答只复述选中的证据；加 --deepseek 才会请求模型做关键词和受证据约束的回答。
uv run hybrid-rag ask "How does LightRAG use entities?" --mode hybrid --db .tmp/demo.db --json
uv run hybrid-rag ask "How does LightRAG use entities?" --mode hybrid --deepseek --db .tmp/demo.db

# 每次 retrieve/ask 产生 rtr_ trace；重放不重新 embedding 或调用模型。
uv run hybrid-rag retrieval replay rtr_<id> --db .tmp/demo.db --json
```

`hybrid` 用线程池并行运行三条召回路径：chunk 向量的 naive、以实体命中展开相邻关系的
local、以关系命中汇聚证据的 global。随后项目代码按路归一化、加权融合、按 chunk 去重、补充
有界 NetworkX 路径，并在 token budget 内选择最终上下文。每一项结果都返回分数、路由贡献、
图路径、source chunk 和稳定 citation ID；模型输出的 citation 必须是这些选中 chunk ID 的
精确子集。

默认配置可在 `.env.example` 查看。选择外部 adapter 时设置：

```dotenv
HYBRID_RAG_RETRIEVAL_EMBEDDING_PROVIDER=openai-compatible
HYBRID_RAG_RETRIEVAL_EMBEDDING_BASE_URL=https://your-compatible-endpoint/v1
HYBRID_RAG_RETRIEVAL_EMBEDDING_API_KEY=
HYBRID_RAG_RETRIEVAL_EMBEDDING_MODEL=your-embedding-model
HYBRID_RAG_RETRIEVAL_EMBEDDING_DIMENSIONS=1024
```

## 阶段四：评测与 Demo

仓库提交了一个版本化的 24 题离线 benchmark：事实、比较、关系和跨文档综合题各 6
题。题集绑定图谱无关的 document/chunk corpus hash；`evaluate` 在开始时解析一次 active
profile（或显式的 `--profile`），固定该 profile 和图谱 snapshot，并拒绝错误语料库。每题每种
模式都持久化可 replay 的 `rtr_` trace；报告包含 citation 检查、evidence-hit/faithfulness
proxy、延迟、盲评映射和锁定的 index provenance。

默认使用确定性的离线盲评 fallback，因此不会为了得到评测结果而伪造 DeepSeek 调用或成本。
稳定的 `evr_` 表示可复现配置，唯一的 `evx_` 表示一次实际执行，产物文件名为
`evr_<...>-evx_<...>.json/.md`，不会静默覆盖前一次延迟或成本测量。

从阶段三升级的既有数据库先运行一次 `build-index`。它会在可复用 profile 上补充
corpus-content provenance，不会重新调用 embedding provider；随后才可执行 `evaluate`。

```bash
# 先完成 ingest、build-graph 和 build-index；默认结果写入已忽略的 artifacts/ 目录。
uv run hybrid-rag evaluate --db .tmp/demo.db --json
uv run hybrid-rag evaluate --db .tmp/demo.db --output-dir artifacts/evaluations

# 锁定一个已有 profile，适合冻结真实论文评测时使用。
uv run hybrid-rag evaluate --db .tmp/demo.db --profile idx_<id> --json

# 只有在设置 DEEPSEEK_API_KEY 后才会请求 DeepSeek judge。
# A/B 标签会逐题确定性打乱，judge 只能看到题目、候选答案和允许的 citation ID。
uv run hybrid-rag evaluate --db .tmp/demo.db --deepseek-judge --json

# 浏览式演示：选择数据库、构建索引、比较四种模式，并查看 citation、路径和 trace。
uv run streamlit run src/hybrid_rag/demo.py
```

外部 embedding 的调用和费用不会被伪装成离线零成本；当前 CLI 会明确标记为 `unknown`，直到
有完整、经核实的 usage 与价格披露。hash embedding 下，外部 judge 的费用只有在显式提供每百万
input/output token 的价格假设、且没有 judge fallback 时才会计算为 `estimated`；否则同样标记为
`unknown`。报告会记录 judge 的模型、endpoint、JSON/Thinking 设置，但不包含密钥。对真实论文
质量、在线延迟和实际成本的结论必须在冻结语料与凭据可用的环境中单独运行，不能由 fixture 结果替代。

## 已实现的工程保证

- document/chunk ID 由来源、内容和处理配置确定性生成。
- parser、cleaner、chunker、tokenizer 版本与结果一起留档。
- SQLite 开启外键，文档更新与 chunk 替换在同一事务完成。
- 单个损坏文件不会阻断同批其他文档，错误写入结构化导入报告。
- 每个 chunk 保留 document ID、section path、页码或字符偏移和 token count。
- 模型 JSON 必须通过 schema、局部引用和来源证据校验，合法空抽取不会进入无效重试。
- 抽取配置与图配置分别哈希；修改归一规则可复用成功抽取，不必重新调用模型。
- 当前规范图可从数据库证据确定性重建为 NetworkX `MultiDiGraph`。
- 文档内容变化会在替换旧 chunks 前使关联 graph snapshot 与 active embedding profile 失效，
  防止使用陈旧向量；历史 `rtr_` trace 仍可离线重放。
- embedding profile 的身份绑定语义配置、切块语料和 graph run；Alembic `0004` 会把旧 profile、
  vector 与 trace 关联迁移到这个图谱快照身份。
- 自动化测试使用本地 fixture，不依赖网络或 DeepSeek API。

## Agent 的边界

文档导入仍是普通的确定性 ETL。LangGraph 只编排阶段二的并行、有限重试、checkpoint 和
人工复核；实体归一、关系合并、图谱、local/global/hybrid 检索和融合评分仍由本项目
实现。

完整阶段划分见 [实施计划](docs/implementation-plan.md)，框架复用边界见
[ADR 001](docs/adr/001-build-vs-reuse.md)，抽取与恢复语义见
[ADR 002](docs/adr/002-graph-extraction-checkpoints.md)，索引和 trace 语义见
[ADR 003](docs/adr/003-retrieval-index-and-trace.md)。系统图见
[架构图](docs/architecture.md)，真实 benchmark 的记录规则见
[评测报告模板](docs/evaluation-report.md)，可直接用于录屏的流程见
[90 秒演示脚本](docs/demo-script.md)。

## 当前限制

- 默认 PDF adapter 面向有文本层的论文；扫描件不做 OCR。
- 复杂多栏阅读顺序、公式和表格结构不是本阶段承诺，可在既有 loader 接口后替换为
  Docling。
- 当前 token 计数使用可配置的 tiktoken encoding；context budget 使用同一计数器。更换
  embedding provider、模型、维度或 embedding-text schema 会生成新的 profile，可确定性重建。
- 默认实体归一采用保守规则，无法可靠判定为同一对象的候选会暂时保持分离。
- DeepSeek 在线调用存在延迟和输出波动；自动化测试全部使用本地 scripted client，不把
  外部 API 可用性作为测试前提。
- 固定 fixture benchmark 只验证端到端契约和回归；它不是关于真实论文检索质量的结论。
