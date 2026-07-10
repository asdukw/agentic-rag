# Hybrid RAG

一个面向求职展示的轻量 Graph-RAG 项目：复刻 LightRAG 的核心工程链路，同时保留
每一步检索和证据来源的可解释性。

当前已完成阶段一 **Ingestion Foundation**：

```text
PDF / Markdown / TXT
  -> loader
  -> conservative cleaner
  -> section-aware, token-bounded chunker
  -> SQLAlchemy transaction
  -> SQLite documents + chunks + ingest_runs
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

## 已实现的工程保证

- document/chunk ID 由来源、内容和处理配置确定性生成。
- parser、cleaner、chunker、tokenizer 版本与结果一起留档。
- SQLite 开启外键，文档更新与 chunk 替换在同一事务完成。
- 单个损坏文件不会阻断同批其他文档，错误写入结构化导入报告。
- 每个 chunk 保留 document ID、section path、页码或字符偏移和 token count。
- 自动化测试使用本地 fixture，不依赖网络或 DeepSeek API。

## 为什么现在没有 Agent

文档导入是确定性 ETL，Agent 不会提升正确性。阶段二开始会用 LangGraph 薄编排
“抽取 → Pydantic 校验 → 修复/重试 → 人工复核 → 持久化”，但实体归一、关系合并、
图谱、local/global/hybrid 检索和融合评分仍由本项目实现。

完整阶段划分见 [实施计划](docs/implementation-plan.md)，框架复用边界见
[ADR 001](docs/adr/001-build-vs-reuse.md)。

## 当前限制

- 默认 PDF adapter 面向有文本层的论文；扫描件不做 OCR。
- 复杂多栏阅读顺序、公式和表格结构不是本阶段承诺，可在既有 loader 接口后替换为
  Docling。
- 当前 token 计数使用可配置的 tiktoken encoding；阶段三确定 embedding 模型后会做
  tokenizer 对齐并通过 config hash 触发可重复重建。
