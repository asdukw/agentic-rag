# Hybrid RAG

一个以学习和演示为主的 Hybrid/Graph RAG 项目。默认使用仓库内的小型 fixture 语料，展示从文档导入、
索引与可选知识图谱构建，到检索、回答和 citation 追溯的完整效果；也可选用一组固定版本的 RAG 相关论文
作为更完整的演示语料。
不以生产部署或通用知识库能力为主要目标。

```text
论文 PDF → 文本与 chunk → 可选知识图谱 → 三路索引 → 检索/回答 → citation 与 replay trace
```

项目自身实现实体归一、关系合并、图谱检索、融合评分、上下文裁剪和引用追踪；第三方库只
承担解析、持久化、编排或模型 API 适配。

## Quick Start

需要 Python 3.11--3.13 和 [uv](https://docs.astral.sh/uv/)。默认流程使用仓库内的
`tests/fixtures/corpus`，不需要下载论文；首次建立索引时会下载本地 BGE-M3 模型权重，因此需要网络，
但不需要 embedding API key。默认数据库为 `storage/app.db`。

```bash
uv sync --dev

# 导入内置 fixture 语料并切成可追溯的 chunks。
uv run hybrid-rag ingest tests/fixtures/corpus

# 可选：抽取实体和关系，为 local、global、hybrid 及 mix 提供图谱证据（需要 DEEPSEEK_API_KEY）。
# 只使用 naive，或只需 mix 的 chunk 路径时可跳过这一步。
uv run hybrid-rag build-graph --limit 10

# 用默认的本地 BGE-M3 embedding 为 chunk 建立语义索引。
uv run hybrid-rag build-index

# 检索并生成基于语料证据的离线回答。
uv run hybrid-rag ask "How does mix retrieval combine evidence?"
```

完成后可在输出中查看回答所依据的 citation，用于观察 RAG 的检索与证据归因效果。fixture 语料较小，
适合快速验证命令流程，不代表论文语料上的演示效果。

- `ingest`、`build-index` 等命令会自动升级 SQLite schema。
- 默认 embedding 是 FlagEmbedding 的 `BAAI/bge-m3` dense vector（1024 维、最长 8192 tokens）；
  它在本机运行，首次使用会下载模型。CPU 环境保持 `EMBEDDING_USE_FP16=false`，有兼容 CUDA GPU 时可
  设为 `true` 加速。修改 embedding 的模型、维度、最大长度或精度后，必须重新运行 `build-index`。
- `hash-token-v1` 仅保留给已有 profile 兼容和快速离线测试，不再作为 CLI 或演示默认值。
- `naive` 同时使用 chunk 的 dense 向量分数和本地 BM25 词法分数，并分别归一化后融合；BM25
  直接读取已索引的 chunk 文本，不需要额外模型、服务或重建索引。
- 融合后的候选默认使用本地 FlagEmbedding cross-encoder
  `BAAI/bge-reranker-v2-m3` 精排：它批量计算 `[query, passage]` 对的相关性，再决定最终 Top-K。
  首次检索会下载精排模型；将 `HYBRID_RAG_RETRIEVAL_RERANKER_PROVIDER=none` 可跳过二阶段精排。

## 项目功能

### FlagEmbedding cross-encoder 精排

默认已启用 `BAAI/bge-reranker-v2-m3`。`uv sync` 会安装 FlagEmbedding，首次检索会下载相应权重。
正常执行检索或问答命令即可：

```bash
uv run hybrid-rag retrieve "How does LightRAG use entities?" --mode mix --json
```

首次检索会下载模型权重。实现会批量计算 `[query, passage]` 对的交叉编码器分数，并在 trace 中保留
原始 logit 与 sigmoid 归一化后的 0--1 分数。仅在兼容 CUDA 的 GPU 环境中将
`HYBRID_RAG_RETRIEVAL_RERANKER_USE_FP16=true`；CPU 环境保持 `false`。仍可将 provider 设为 `none`，
跳过所有二阶段精排。

### 构建知识图谱

配置 `DEEPSEEK_API_KEY` 后，抽取实体与关系；完成后重新建索引以纳入 entity/relation 向量：

```bash
uv run hybrid-rag build-graph --limit 10
uv run hybrid-rag build-index
```

没有图谱时，`naive` 可正常使用，默认的 `mix` 仍会保留 chunk 路径；`hybrid` 仅组合图谱的
`local` 与 `global` 路径，因此可能没有结果。构图后，`local`、`global`、`hybrid` 与 `mix`
都能提供实体、关系和 NetworkX 路径证据。

### DeepSeek 人民币成本估算

DeepSeek 的上游响应提供 token 用量而非账单金额。项目会用响应中的实际模型、缓存命中输入、
缓存未命中输入和输出 token，结合 `.env` 中的六项单价估算人民币成本；`build-graph`、
`retrieve --deepseek`、`ask --deepseek` 与 `evaluate --deepseek-judge` 的 JSON/trace/报告都会保留
对应的用量和估算结果。


模型仅按实际响应中的 `deepseek-v4-flash` 或 `deepseek-v4-pro` 精确匹配价格表。若上游没有返回两类
缓存 token，或两者之和与输入 token 不一致，成本状态会显示为 `unknown`，而不会猜测为缓存未命中；
因此金额是基于 response usage 的估算，不是 DeepSeek 账单。

### 选择检索模式与查看证据

```bash
uv run hybrid-rag retrieve "How does LightRAG use entities?" --mode mix --json
uv run hybrid-rag retrieval replay rtr_<id> --json
```

可选 `--mode naive|local|global|hybrid|mix`，默认 `mix`：`naive` 是项目扩展的 chunk dense +
BM25 召回；`local` 从实体命中扩展图邻居；`global` 从关系命中汇聚证据；`hybrid` 只组合
`local + global`；`mix` 以 LightRAG 的语义组合 `naive + local + global`。复合模式按来源轮询
候选并按 chunk ID 去重，之后统一精排和裁剪。每次检索都会产生可 replay 的 `rtr_` trace；其中保留
naive 的 dense/BM25 分项、路由贡献、reranker 候选池和最终名次。`--deepseek` 才会启用模型做关键词
提取或受证据约束的回答。

### 运行评测或演示界面

```bash
# 默认使用可复现的离线指标和盲评，不调用 DeepSeek。
uv run hybrid-rag evaluate

# 可选：让 DeepSeek 对匿名的 naive / hybrid A/B 结果进行盲评。
# 需要设置 DEEPSEEK_API_KEY，且会产生 API 调用费用。
uv run hybrid-rag evaluate --deepseek-judge

# 使用 Ragas 生成的测试集评测实际回答和召回上下文。
# 同样需要 DEEPSEEK_API_KEY，且会调用回答模型与 Ragas 评审模型。
uv run hybrid-rag ragas-evaluate --testset data/processed/ragas-testset-demo.json

uv run streamlit run src/hybrid_rag/demo.py
```

`evaluate` 使用固定题集对比 `naive` 与 `hybrid`，并写出带 profile、图谱快照、citation、trace、
延迟和成本状态的 JSON/Markdown artifact。默认评测使用离线、确定性的盲评规则；传入
`--deepseek-judge` 后，DeepSeek 仅根据匿名的答案、引用和评测指标进行 A/B 裁判，不能检索额外资料。
Streamlit 演示可查看五种模式、证据、图路径和对比结果。
默认 `evaluate` 的结果仅供参考，适合作为快速 smoke 检查，不应视为回答或检索质量的正式结论。
`ragas-evaluate` 会保留现有离线 benchmark 不变：它读取 Ragas 生成的 JSON，调用当前 RAG 的
`ask` 流程取得实际 `response` 与 `retrieved_contexts`，再分别计算 faithfulness、factual correctness、
context precision 和 context recall。默认评测 `mix`；可用 `--modes naive,mix` 比较多个模式，结果写入
`artifacts/evaluations/ragas-<测试集文件名>.json`。

### 下载论文语料（可选）

若需要使用更接近真实研究资料的演示语料，可下载 `data/corpus.json` 中定义、版本固定且经过
SHA-256 校验的 RAG 论文到 `data/raw`。这一步需要网络：

```bash
uv run hybrid-rag corpus download
uv run hybrid-rag ingest data/raw
```

也可以手动将允许使用的 PDF 放入 `data/raw` 后直接执行 `ingest data/raw`。

### 配置 DeepSeek

本项目的 embedding 固定使用本地 FlagEmbedding 模型；查看 [.env.example](.env.example) 可配置
DeepSeek 模型、API 地址和评测选项。

### 自定义论文语料

内置论文语料由 `data/corpus.json` 定义；也可以指定自己的论文清单和输出目录：

```bash
uv run hybrid-rag corpus download --manifest path/to/corpus.json --output data/raw
```

下载受本机网络与证书配置影响。

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
