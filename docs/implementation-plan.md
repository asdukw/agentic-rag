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

## 4. 阶段二：DeepSeek 抽取与图谱

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

## 5. 阶段三：多路索引与四种检索

- 为 chunk/entity/relation 建独立 embedding 文本与向量索引。
- 实现 naive、local、global、hybrid 四种 retriever，共享统一结果 schema。
- hybrid 并行运行三路召回，执行归一化、加权融合、去重、图扩展和 token budget
  裁剪。
- 返回命中分数、图路径、来源 chunk 和最终上下文，保证可解释。
- DeepSeek 只负责关键词抽取和基于证据生成，不让 Agent 自由选择不可控工具链。

验收：同一问题可切换四种模式，答案有引用，retrieval trace 可序列化并重放。

## 6. 阶段四：评测与 Demo

- 固定 20-30 个事实型、对比型、关系型和跨文档综合问题。
- 比较 naive 与 hybrid 的 evidence hit rate、faithfulness、answer win rate、延迟和成本。
- Judge 使用 `deepseek-v4-pro` 只是待验证假设；保留盲评、顺序随机化和人工抽查，
  披露同厂模型自评偏差。
- Streamlit 展示答案、引用、实体/关系/chunk 命中、图路径和两种模式对比。
- README、架构图、评测报告和 1-2 分钟演示素材形成求职交付物。

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
