# Hybrid RAG

一个轻量、可解释的 Graph-RAG 项目：从 PDF、Markdown 或 TXT 导入文档，构建可追溯的
chunk、实体和关系，再以 `naive`、`local`、`global` 或 `hybrid` 模式返回带 citation 的证据。

```text
文件 → 文本与 chunk → 可选知识图谱 → 三路索引 → 检索/回答 → citation 与 replay trace
```

项目自身实现实体归一、关系合并、图谱检索、融合评分、上下文裁剪和引用追踪；第三方库只
承担解析、持久化、编排或模型 API 适配。

## 基础工作流

需要 Python 3.11--3.13 和 [uv](https://docs.astral.sh/uv/)。下面使用仓库内置的 fixture，
无需网络、模型或 API key；把目录替换成自己的论文目录即可。

```bash
uv sync --dev

# 导入文件并切成可追溯的 chunks。
uv run hybrid-rag ingest tests/fixtures/corpus --db .tmp/demo.db

# 为 chunk 建立默认的本地确定性索引。
uv run hybrid-rag build-index --db .tmp/demo.db

# 检索并生成基于所选证据的离线回答。
uv run hybrid-rag ask "What is retrieval-augmented generation?" --db .tmp/demo.db
```

`ingest`、`build-index` 等命令会自动升级 SQLite schema。默认的 `hash-token-v1` 是一个
确定性本地特征哈希 embedding，适合开发、测试和演示；它不是语义模型。

## 可选功能

### 使用自己的文档

```bash
uv run hybrid-rag ingest data/raw --db storage/app.db
uv run hybrid-rag build-index --db storage/app.db
```

支持 PDF、Markdown 和 TXT。重复导入未变化文件会跳过；文档变化会使旧索引失效，需要再次
运行 `build-index`。

### 构建知识图谱

配置 `DEEPSEEK_API_KEY` 后，抽取实体与关系；完成后重新建索引以纳入 entity/relation 向量：

```bash
uv run hybrid-rag build-graph --db .tmp/demo.db --limit 10
uv run hybrid-rag build-index --db .tmp/demo.db
```

没有图谱时，`naive` 可正常使用，`hybrid` 会保留 chunk 路径；构图后 `local` 和 `global`
也能提供实体、关系和 NetworkX 路径证据。

### 选择检索模式与查看证据

```bash
uv run hybrid-rag retrieve "How does LightRAG use entities?" --mode hybrid --db .tmp/demo.db --json
uv run hybrid-rag retrieval replay rtr_<id> --db .tmp/demo.db --json
```

可选 `--mode naive|local|global|hybrid`。每次检索都会产生可 replay 的 `rtr_` trace；`--deepseek`
才会启用模型做关键词提取或受证据约束的回答。

### 运行评测或演示界面

```bash
uv run hybrid-rag evaluate --db .tmp/demo.db
uv run streamlit run src/hybrid_rag/demo.py
```

`evaluate` 使用固定题集对比 `naive` 与 `hybrid`，并写出带 profile、图谱快照、citation、trace、
延迟和成本状态的 JSON/Markdown artifact。Streamlit 演示可查看四种模式、证据、图路径和对比结果。

### 接入外部 embedding 或真实论文语料

查看 [.env.example](.env.example) 配置 OpenAI-compatible embedding endpoint、DeepSeek 模型和
评测选项。可选的公开语料下载命令是：

```bash
uv run hybrid-rag corpus download
```

下载受本机网络与证书配置影响；也可以手动将允许使用的 PDF 放入 `data/raw` 后直接执行
`ingest data/raw`。

## 工程保证与边界

- document/chunk ID、处理配置、图谱来源和 index profile 均可追溯。
- SQLite 外键和事务保证文档更新不会留下孤儿 chunk。
- 模型抽取需经过 Pydantic、局部引用和原文证据校验；每条实体/关系保留来源 chunk。
- profile 绑定 embedding 配置、语料和图谱快照；历史 `rtr_` trace 可离线 replay。
- 评测会锁定 profile 并校验语料指纹；fixture 结果只用于回归验证，不能代表真实论文质量。

当前限制：默认 PDF adapter 面向有文本层的论文，不做扫描件 OCR；复杂版面、表格和公式结构可在
既有 loader 接口后接入 Docling。在线模型的质量、延迟和费用需要在冻结语料和明确配置下单独评测。

## 更多资料

- [架构图](docs/architecture.md)
- [评测报告模板](docs/evaluation-report.md)
- [演示脚本](docs/demo-script.md)
- [实施计划](docs/implementation-plan.md)
- [ADR 001：自研与复用边界](docs/adr/001-build-vs-reuse.md)
- [ADR 002：抽取与恢复语义](docs/adr/002-graph-extraction-checkpoints.md)
- [ADR 003：索引与 trace](docs/adr/003-retrieval-index-and-trace.md)
