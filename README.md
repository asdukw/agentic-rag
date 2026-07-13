# Hybrid RAG

一个以学习和演示为主的 Hybrid/Graph RAG 项目。项目使用一组固定版本的 RAG 相关论文作为
语料，展示从论文 PDF 导入、索引与可选知识图谱构建，到检索、回答和 citation 追溯的完整效果；
不以生产部署或通用知识库能力为主要目标。

```text
论文 PDF → 文本与 chunk → 可选知识图谱 → 三路索引 → 检索/回答 → citation 与 replay trace
```

项目自身实现实体归一、关系合并、图谱检索、融合评分、上下文裁剪和引用追踪；第三方库只
承担解析、持久化、编排或模型 API 适配。

## Quick Start

需要 Python 3.11--3.13 和 [uv](https://docs.astral.sh/uv/)。默认流程会下载
`data/corpus.json` 中定义、版本固定且经过 SHA-256 校验的 RAG 论文到 `data/raw`；下载需要网络，
其余步骤不需要模型或 API key。默认数据库为 `storage/app.db`。

```bash
uv sync --dev

# 下载内置论文语料（需要网络）。
uv run hybrid-rag corpus download

# 导入论文并切成可追溯的 chunks。
uv run hybrid-rag ingest data/raw

# 为 chunk 建立默认的本地确定性索引。
uv run hybrid-rag build-index

# 检索并生成基于论文证据的离线回答。
uv run hybrid-rag ask "How does retrieval-augmented generation improve knowledge-intensive NLP tasks?"
```

完成后可在输出中查看回答所依据的论文 citation，用于观察 RAG 的检索与证据归因效果。若只想
离线验证命令流程，可将 `data/raw` 替换为 `tests/fixtures/corpus`；fixture 仅用于测试，不代表论文
语料的演示效果。

- `ingest`、`build-index` 等命令会自动升级 SQLite schema。
- 默认的 `hash-token-v1` 是确定性本地特征哈希 embedding，适合开发、测试和演示；它不是语义模型。
- `naive` 同时使用 chunk 的 dense 向量分数和本地 BM25 词法分数，并分别归一化后融合；BM25
  直接读取已索引的 chunk 文本，不需要额外模型、服务或重建索引。
- 融合后的候选默认还会经过本地确定性的 lexical reranker：它比较候选集内 BM25、查询词覆盖率、
  有序词邻近度和少量首阶段分数，再决定最终 Top-K。可在 `.env` 中将
  `HYBRID_RAG_RETRIEVAL_RERANKER_PROVIDER=none` 关闭。

## 可选功能

### 使用 FlagEmbedding cross-encoder 精排

默认 lexical reranker 不依赖模型，适合离线学习和演示。启用真正的 query-passage
cross-encoder 精排需要完成以下三步。

1. 安装可选依赖：

```bash
uv sync --extra reranker
```

2. 在项目根目录的 `.env` 中写入以下配置（若尚未创建 `.env`，可参考 `.env.example`）：

```dotenv
HYBRID_RAG_RETRIEVAL_RERANKER_PROVIDER=flagembedding
HYBRID_RAG_RETRIEVAL_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
HYBRID_RAG_RETRIEVAL_RERANKER_USE_FP16=false
```

3. 正常执行检索或问答命令即可：

```bash
uv run hybrid-rag retrieve "How does LightRAG use entities?" --mode hybrid --json
```

首次检索会下载模型权重。实现会批量计算 `[query, passage]` 对的交叉编码器分数，并在 trace 中保留
原始 logit 与 sigmoid 归一化后的 0--1 分数。仅在兼容 CUDA 的 GPU 环境中将
`HYBRID_RAG_RETRIEVAL_RERANKER_USE_FP16=true`；CPU 环境保持 `false`。仍可将 provider 设为 `none`，
跳过所有二阶段精排。

### 使用自己的文档

```bash
uv run hybrid-rag ingest data/raw
uv run hybrid-rag build-index
```

支持 PDF、Markdown 和 TXT。重复导入未变化文件会跳过；文档变化会使旧索引失效，需要再次
运行 `build-index`。

### 构建知识图谱

配置 `DEEPSEEK_API_KEY` 后，抽取实体与关系；完成后重新建索引以纳入 entity/relation 向量：

```bash
uv run hybrid-rag build-graph --limit 10
uv run hybrid-rag build-index
```

没有图谱时，`naive` 可正常使用，`hybrid` 会保留 chunk 路径；构图后 `local` 和 `global`
也能提供实体、关系和 NetworkX 路径证据。

### 选择检索模式与查看证据

```bash
uv run hybrid-rag retrieve "How does LightRAG use entities?" --mode hybrid --json
uv run hybrid-rag retrieval replay rtr_<id> --json
```

可选 `--mode naive|local|global|hybrid`：`naive` 是 chunk dense + BM25 召回，`local` 从实体
命中扩展图邻居，`global` 从关系命中汇聚证据，`hybrid` 固定融合三条路径。每次检索都会产生
可 replay 的 `rtr_` trace；其中保留 naive 的 dense/BM25 分项，以及融合后 reranker 的候选池、
原始/归一化分数和最终名次。`--deepseek` 才会启用模型做关键词提取或受证据约束的回答。

### 运行评测或演示界面

```bash
uv run hybrid-rag evaluate
uv run streamlit run src/hybrid_rag/demo.py
```

`evaluate` 使用固定题集对比 `naive` 与 `hybrid`，并写出带 profile、图谱快照、citation、trace、
延迟和成本状态的 JSON/Markdown artifact。Streamlit 演示可查看四种模式、证据、图路径和对比结果。

### 接入外部 embedding 或自定义论文语料

查看 [.env.example](.env.example) 配置 OpenAI-compatible embedding endpoint、DeepSeek 模型和
评测选项。内置论文语料由 `data/corpus.json` 定义；也可以指定自己的论文清单和输出目录：

```bash
uv run hybrid-rag corpus download --manifest path/to/corpus.json --output data/raw
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
