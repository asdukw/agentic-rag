# Hybrid RAG 90 秒演示脚本

> 目标：在约 90 秒内展示系统不是一个黑盒聊天界面，而是一条可追溯、可比较、可复现的
> Graph-RAG 工程链路。

## 0--15 秒：问题与边界

打开 Streamlit 演示页，说明：这个项目复刻 LightRAG 的核心思路，但实体归一、关系合并、
图谱检索、融合评分和 citation 追踪都由项目自身实现；LangGraph 只负责阶段二的编排。

## 15--30 秒：从数据到索引

选择已导入、已构图的 SQLite 数据库，点击 **Build / refresh index**。展示 index profile 的
embedding provider、模型、维度、编码参数、corpus-content hash 和 graph snapshot，说明 chunk、entity、relation
是三套独立索引；默认使用本地 BGE-M3，hash embedding 仅用于可重复的离线开发和测试。

## 30--50 秒：五种检索模式

输入一个论文问题，保持默认的 `mix` 并提交。依次指出：

- `naive` 将 chunk 的 dense 向量与本地 BM25 词法分数分别归一化后融合；
- `local` 从实体命中扩展图邻居；
- `global` 从关系命中汇聚证据；
- `hybrid` 并行组合 `local + global` 两条图谱路径；
- `mix` 是默认模式，在 `hybrid` 的基础上加入 `naive` chunk 路径。复合模式先按来源轮询候选、
  按 chunk ID 去重，再统一精排和 token budget 裁剪。关闭精排时保留该首阶段顺序。

## 50--70 秒：证据而不是黑盒答案

展示最终答案下方的 citation、chunk 原文、实体/关系命中和 NetworkX 路径。展开 rerank 表格，
指出候选的重排序前排名、重排序分项与最终排名。回答模型只能接收已经选中的证据，citation
必须是允许 chunk ID 的子集；点击 trace/replay 可查看本次分数、路径和最终上下文，无需重新调用
embedding 或模型。

## 70--85 秒：Ragas 评测

先展示 `build-index --json` 输出中的 `corpus_content_hash`，再运行：

```bash
uv run hybrid-rag evaluate --testset data/processed/my-ragas-testset.json --json
```

说明测试集是带 `schema_version`、`corpus_content_hash` 和 `cases` 的 Ragas envelope；评测开始时会
校验它与选定 profile 的 document/chunk 语料是否一致。每个 case 都调用真实的 `ask` 流程取得回答和
召回上下文，再计算 faithfulness、factual correctness、context precision 与 context recall。默认评测
`mix`，可显式传 `--modes naive,mix` 比较模式。`DEEPSEEK_API_KEY` 为必需项，因为回答和 Ragas
评审都会调用 DeepSeek，并在 JSON 报告中保留相应结果与成本信息。

## 85--90 秒：收束

总结：每个结论都能回到 document/chunk、实体关系、图路径和 trace；真实论文质量、在线
延迟与费用则必须在冻结语料和明确模型配置下单独报告，不能由 fixture 结果冒充。
