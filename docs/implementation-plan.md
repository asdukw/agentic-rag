# Hybrid RAG 实施计划

## 1. 目标与原则

本项目复刻 LightRAG 的核心工程思想，但不直接封装官方 LightRAG，也不让通用
RAG 框架代替核心算法。

工程边界如下：

- 复用成熟库完成 PDF 解析、数据校验、数据库事务、工作流编排和日志。
- 自己实现实体/关系 schema、实体归一、关系合并、知识图谱、三路索引、
  local/global/hybrid 检索、融合评分、上下文裁剪和引用追踪。
- 每个阶段都交付一条可运行、可测试、可演示的纵向链路，不提前创建空模块。

## 2. 技术决策

| 能力 | 选择 | 说明 |
| --- | --- | --- |
| Python | 3.13（兼容 3.11-3.13） | 避免 Python 3.14 对后续 ML 依赖的兼容风险 |
| 数据契约 | Pydantic | 隔离第三方框架类型，保证模块边界稳定 |
| PDF 解析 | 轻量 PDF adapter；预留 Docling adapter | 默认安装快速、无模型下载；复杂版面可切到 Docling |
| Chunk 计数 | tiktoken adapter | 可替换 tokenizer，切块配置与结果一同留档 |
| 持久化 | SQLAlchemy 2 + Alembic + SQLite | 提供事务、约束、迁移和后续 schema 演进能力 |
| CLI | Typer + Rich | 统一验收入口，脚本只保留薄包装（如需要） |
| Agent 编排 | LangGraph（从阶段二引入） | 只负责状态、并行、重试、checkpoint 和人工复核 |
| DeepSeek | 官方 OpenAI-compatible API + 自有 client | 不依赖可能滞后的框架 provider adapter |
| 图存储 | NetworkX | 第一版便于解释、遍历和序列化 |
| 向量检索 | 后续基准后确定 | 保留 vector store adapter，避免早期锁定实现 |

LangGraph 是低层编排运行时，不会替项目决定 prompt 或检索架构，适合作为薄编排层：
[LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)。论文 PDF
若需要更强的版面、表格和阅读顺序恢复，可在同一 loader 接口下接入
[Docling](https://docling-project.github.io/docling/)。

## 3. 阶段一：Ingestion Foundation（已完成）

完成日期：2026-07-10。

实测结果：版本固定且带 SHA-256 的 10 篇 arXiv PDF 全部导入成功，生成 476 个
chunks、202,802 个 contextualized tokens；最大 chunk 为 512 tokens。重复导入 10 篇
全部在解析前跳过，约 0.04 秒完成。13 项自动化测试与 Ruff 检查通过，总覆盖率 89%。

### 3.1 定义

“第一步”是建立可重复、可追溯、幂等的论文导入流水线，而不是只下载论文或只建
目录：

```text
PDF / Markdown / TXT
  -> LoaderRegistry
  -> ParsedDocument
  -> conservative cleaner
  -> section-aware/token-bounded chunker
  -> SQLite transaction
  -> ingest report
```

### 3.2 范围

包含：

- `src/hybrid_rag` 包结构、环境配置、日志和统一 CLI。
- PDF、Markdown、TXT loader 与可替换 adapter 接口。
- 保守清洗；不激进删除 References、公式或表格文本。
- 按章节/段落优先、token 上限兜底的确定性切块。
- 稳定 document/chunk ID、内容哈希、parser/chunker 版本。
- SQLite 外键、唯一约束、单文档事务和 schema migration。
- 未变化跳过、内容变化后原子替换旧 chunks、单文件错误隔离。
- `stats`、`inspect` 和结构化导入报告。
- 无网络单元/集成测试与小型 fixture corpus。

不包含：DeepSeek、实体关系抽取、NetworkX、embedding、向量库、问答、FastAPI、
Streamlit 和评测。

### 3.3 验收

```bash
uv sync --dev
uv run alembic upgrade head
uv run pytest -q
uv run ruff check .
uv run hybrid-rag ingest tests/fixtures/corpus --db .tmp/demo.db
uv run hybrid-rag ingest tests/fixtures/corpus --db .tmp/demo.db
uv run hybrid-rag stats --db .tmp/demo.db
```

Definition of Done：

- 三种格式都能映射到统一 `ParsedDocument`；损坏文件不阻断同批其他文件。
- 所有 chunk 非空、不超过配置上限，并保留 document、section、页码/字符偏移。
- 同一输入和配置产生相同 ID；第二次导入数量不增长且报告为 skipped。
- 单个文件内容变化时，仅该文档的 chunks 在同一事务内被替换。
- SQLite 不存在孤儿 chunk，外键和唯一约束实际生效。
- 自动化测试不访问网络；真实论文 smoke test 与 CI fixture 分离。

## 4. 阶段二：DeepSeek 抽取与图谱（已完成）

完成日期：2026-07-11。

离线验收结果：52 项自动化测试与 Ruff 检查通过，总覆盖率 89%。Alembic 已验证空库、
带阶段一数据的升级以及 `0002 -> 0001 -> 0002` 往返；scripted client 端到端覆盖非法
JSON 修复、永久失败隔离、成功缓存、人工复核暂停/恢复和 worker 中断后的 lease 回收。
fixture smoke run 处理 5 个 chunks，生成 2 个规范实体、1 条关系和 1 个弱连通分量；第二次
同配置运行 5 个 chunks 全部命中缓存，没有新增模型调用。

当前环境未提供 `DEEPSEEK_API_KEY`，因此没有伪造在线模型的延迟、token 或成本数据；
真实 `deepseek-v4-flash` 小批量 smoke test 仍需在有凭据的环境执行，但不阻断离线
Definition of Done。

引入 LangGraph，状态图保持显式：

```text
load pending chunks
  -> parallel extract
  -> Pydantic validate
  -> repair/retry (bounded)
  -> optional human review
  -> normalize entities
  -> merge relations
  -> persist + build NetworkX graph
```

实际实现采用父图和 per-chunk 子图：父图只在 checkpoint 中保存 run/任务 ID 与小型指标，
chunk 正文、raw response、attempt、validated result 和 provenance 以业务数据库为事实来源。
每次真实 API 请求都执行“短事务 claim -> 事务外调用 -> append-only attempt ->
complete/requeue/fail”，attempt 显式归属 run；即使 checkpoint 丢失，也能从业务库恢复
同一调用上限和成本统计。人工复核状态归属 build item，不污染跨 run 共享的成功抽取缓存。
LangGraph checkpoint 使用独立 SQLite 文件，run ID 同时作为 `thread_id`。

关键实现要求：

- `deepseek-v4-flash` 用于批量抽取，模型名必须配置化。
- 抽取显式关闭默认 Thinking：`thinking.type = disabled`。
- JSON Output 只保证 JSON 语法，仍要 Pydantic schema 校验、空内容检查和有限重试。
- 每个实体/关系必须保留 `source_chunk_ids` 和证据文本。
- LangGraph 只编排节点；归一规则、候选召回和合并决策由项目代码实现。

官方依据：[DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode/)、
[Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/)、
[更新日志](https://api-docs.deepseek.com/updates/)。

验收：`build-graph` 可中断恢复，失败样本可追踪，图谱能输出节点/边/连通分量和
Top-K 实体。

验收入口：

```bash
uv run hybrid-rag build-graph --db .tmp/demo.db --limit 10
uv run hybrid-rag graph stats --db .tmp/demo.db
uv run hybrid-rag graph inspect <gbr_|xtr_|xat_|ent_|rel_ id> --db .tmp/demo.db
uv run pytest --cov=hybrid_rag --cov-report=term-missing -q
uv run ruff check .
```

## 5. 阶段三：多路索引与四种检索（已完成）

完成日期：2026-07-12。

离线验收结果：68 项自动化测试与 Ruff 检查通过，总覆盖率 87%。fixture 端到端链路覆盖导入、scripted
图谱构建、chunk/entity/relation 三路索引、naive/local/global/hybrid 四种模式、并行融合、
NetworkX 路径、token budget、citation、离线 answer、trace 序列化和 replay；CLI 端到端覆盖
`build-index`、`retrieve`、`ask` 与 `retrieval replay`。当前环境未提供 `DEEPSEEK_API_KEY`
或外部 embedding endpoint 凭据，因此没有伪造在线关键词、回答或向量模型的延迟/成本数据；
默认确定性 hash embedding 使离线 Definition of Done 可重复验证。

实现采用 SQLite JSON vector baseline 与明确的 adapter 边界：Alembic `0003` 建立
`embedding_profiles`、`embedding_vectors` 和 `retrieval_traces`，`0004` 将 profile identity、
唯一约束和历史 trace 关联扩展为 graph-run-aware。profile 同时绑定 provider/model/dimension/text
schema、图谱无关的 corpus-content hash 与当前 graph snapshot；完整 profile 才会在同一事务中
激活。默认 `hash-token-v1` 仅用于离线开发和 CI，`OpenAI-compatible` embedding adapter
保留给阶段四基准选定的真实模型，不改变核心检索算法或持久化契约。

- 为 chunk/entity/relation 建独立 embedding 文本与向量索引。
- 实现 naive、local、global、hybrid 四种 retriever，共享统一结果 schema。
- hybrid 并行运行三路召回，执行归一化、加权融合、去重、图扩展、融合后的可替换 rerank
  和 token budget 裁剪。
- 返回命中分数、图路径、来源 chunk 和最终上下文，保证可解释。
- DeepSeek 只负责关键词抽取和基于证据生成，不让 Agent 自由选择不可控工具链。

验收：同一问题可切换四种模式，答案有引用，retrieval trace 可序列化并重放。

验收入口：

```bash
uv run alembic upgrade head
uv run hybrid-rag build-index --db .tmp/demo.db --json
uv run hybrid-rag retrieve "How does LightRAG use entities?" --mode naive --db .tmp/demo.db --json
uv run hybrid-rag retrieve "How does LightRAG use entities?" --mode local --db .tmp/demo.db --json
uv run hybrid-rag retrieve "How does LightRAG use entities?" --mode global --db .tmp/demo.db --json
uv run hybrid-rag retrieve "How does LightRAG use entities?" --mode hybrid --db .tmp/demo.db --json
uv run hybrid-rag ask "How does LightRAG use entities?" --mode hybrid --db .tmp/demo.db --json
uv run hybrid-rag retrieval replay rtr_<id> --db .tmp/demo.db --json
uv run pytest -q
uv run ruff check .
```

## 6. 阶段四：评测与 Demo（已完成）

完成日期：2026-07-12。

离线验收结果：97 项自动化测试与 Ruff 检查通过，总覆盖率 86%。仓库提交版本化 24 题 benchmark，四类
题型（事实、比较、关系、跨文档综合）各 6 题；fixture 端到端覆盖相同图谱和索引快照上的
naive/hybrid 成对评测、证据/citation 检查、延迟记录、持久化 `rtr_` trace、盲评映射、
JSON/Markdown 报告和 DeepSeek judge adapter 的失败回退。`evaluate` 会在开始时锁定 profile，
校验 benchmark 的图谱无关 corpus-content hash，并记录 graph snapshot；因此不会在运行过程中
混用 active profile，也不会在错误语料库上静默给出分数。默认确定性 blind fallback 不发起外部
模型调用，因而不把离线执行时间误报为模型延迟或成本。

`--deepseek-judge` 为显式 opt-in：候选答案逐题映射为匿名 A/B 标签，judge 只能基于给定
问题、候选答案、citation ID 和 rubric 返回结果；它不能检索额外资料。该调用复用自有
OpenAI-compatible client，并保持 JSON Output、Thinking disabled、schema 校验和 usage 留档。
报告同时记录 judge model、endpoint 与采样/输出设置。每次执行生成唯一 `evx_`，使同一
`evr_` 可复现配置的 JSON/Markdown artifact 不会彼此覆写。外部 embedding 始终是 `unknown`
成本，除非有完整经核实的 usage/price disclosure；任一外部 judge fallback 也会降级为
`unknown`。当前环境没有 `DEEPSEEK_API_KEY` 或真实 embedding endpoint 凭据，因此没有把
fixture 分数、离线延迟或确定性 hash embedding 成本包装成真实论文或在线模型结论。

从阶段三升级的数据库应先执行一次 `build-index`：现有可复用 profile 会被原地补充
corpus-content provenance，不触发外部 embedding 请求；随后 `evaluate` 才会接受它。

交付物包括 Streamlit 演示（四模式结果、citation、entity/relation/chunk 命中、NetworkX
路径和 naive/hybrid 对比）、[架构图](architecture.md)、[评测报告模板](evaluation-report.md)
与 [90 秒演示脚本](demo-script.md)。

- 固定 20-30 个事实型、对比型、关系型和跨文档综合问题。
- 比较 naive 与 hybrid 的 evidence hit rate、faithfulness、answer win rate、延迟和成本。
- Judge 使用 `deepseek-v4-pro` 只是待验证假设；保留盲评、顺序随机化和人工抽查，
  披露同厂模型自评偏差。
- Streamlit 展示答案、引用、实体/关系/chunk 命中、图路径和两种模式对比。
- README、架构图、评测报告和 1-2 分钟演示素材形成求职交付物。

验收入口：

```bash
uv run alembic upgrade head
uv run hybrid-rag build-index --db .tmp/demo.db --json
uv run hybrid-rag evaluate --db .tmp/demo.db --json
uv run hybrid-rag evaluate --db .tmp/demo.db --profile idx_<id> --json
uv run hybrid-rag evaluate --db .tmp/demo.db --deepseek-judge --json  # 需要 DEEPSEEK_API_KEY
uv run streamlit run src/hybrid_rag/demo.py
uv run pytest -q
uv run ruff check .
```

## 7. 数据策略

- 仓库提交版本化 manifest 和少量自制/允许分发的测试 fixture，不提交大批 PDF、
  SQLite、向量索引或模型缓存。
- 第一阶段真实 smoke test 固定 10 篇 RAG/Agent/GraphRAG 论文；下载与导入分离。
- “200 chunks”是规模观测值，不是质量替代指标；报告实际数量、token 分布、失败率
  和溯源完整性。

## 8. 风险与控制

- **PDF 复杂版面**：轻量 parser 先覆盖文本型论文；扫描件、复杂公式/表格记录为
  限制，必要时替换为 Docling adapter。
- **Tokenizer 不一致**：tokenizer 名称、chunk 参数和 config hash 一同保存；更换
  embedding 模型后可确定性重建 chunks。
- **框架锁定**：第三方对象只能出现在 adapter 内，领域模型不继承 LangChain、
  LlamaIndex 或 Docling 类型。
- **抽取成本与失败**：缓存、并发上限、指数退避、最大重试和按 chunk 恢复。
- **项目辨识度**：不使用官方 LightRAG、预制 PropertyGraphIndex 或 GraphRAG
  retriever 作为主链路。
