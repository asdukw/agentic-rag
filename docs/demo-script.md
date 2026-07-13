# Hybrid RAG 90 秒演示脚本

> 目标：在约 90 秒内展示系统不是一个黑盒聊天界面，而是一条可追溯、可比较、可复现的
> Graph-RAG 工程链路。

## 0--15 秒：问题与边界

打开 Streamlit 演示页，说明：这个项目复刻 LightRAG 的核心思路，但实体归一、关系合并、
图谱检索、融合评分和 citation 追踪都由项目自身实现；LangGraph 只负责阶段二的编排。

## 15--30 秒：从数据到索引

选择已导入、已构图的 SQLite 数据库，点击 **Build / refresh index**。展示 index profile 的
embedding provider、模型、维度、corpus-content hash 和 graph snapshot，说明 chunk、entity、relation
是三套独立索引；默认 hash embedding 仅用于可重复的离线开发。

## 30--50 秒：四种检索模式

输入一个论文问题，选择 `hybrid` 并提交。依次指出：

- `naive` 将 chunk 的 dense 向量与本地 BM25 词法分数分别归一化后融合；
- `local` 从实体命中扩展图邻居；
- `global` 从关系命中汇聚证据；
- `hybrid` 并行执行前三条路径，按路归一化、融合、去重；融合后的 Top-M 再经过 `.env` 配置的
  reranker（默认本地 lexical，也可为 FlagEmbedding cross-encoder），最后才在 token budget 内裁剪上下文。

## 50--70 秒：证据而不是黑盒答案

展示最终答案下方的 citation、chunk 原文、实体/关系命中和 NetworkX 路径。展开 rerank 表格，
指出候选的重排序前排名、重排序分项与最终排名。回答模型只能接收已经选中的证据，citation
必须是允许 chunk ID 的子集；点击 trace/replay 可查看本次分数、路径和最终上下文，无需重新调用
embedding 或模型。

## 70--85 秒：比较与评测

切换到 **Compare naive vs hybrid**。展示相同问题在两个模式下的答案和证据，随后运行：

```bash
uv run hybrid-rag evaluate --json
```

说明固定 24 题 benchmark 会先锁定 profile、校验 corpus-content hash，再输出
`evr_...-evx_...` JSON/Markdown artifact。它记录 evidence hit、citation grounding proxy、
延迟、可 replay 的 `rtr_` trace、匿名 A/B 盲评和成本状态。默认离线 fallback 不伪造外部模型
成本；外部 embedding 的成本在无完整核实披露时为 `unknown`，有凭据时才使用 `--deepseek-judge`。

## 85--90 秒：收束

总结：每个结论都能回到 document/chunk、实体关系、图路径和 trace；真实论文质量、在线
延迟与费用则必须在冻结语料和明确模型配置下单独报告，不能由 fixture 结果冒充。
