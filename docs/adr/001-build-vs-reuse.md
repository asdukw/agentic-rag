# ADR 001：框架复用边界

- 状态：Accepted
- 日期：2026-07-10

## 背景

项目需要同时体现工程效率和对 LightRAG 的真实理解。全部手写会把时间消耗在 PDF
布局、事务、重试等通用问题上；使用一体化 GraphRAG/QueryEngine 又会掩盖核心实现。

## 决策

1. 第一阶段使用普通 Python 服务编排确定性 ETL，不引入 Agent。
2. PDF 解析、Pydantic 校验、SQLAlchemy/Alembic、CLI 和 tokenizer 采用成熟库。
3. 第二阶段起使用 LangGraph 作为薄编排层；DeepSeek 通过项目自己的 client 调用。
4. 实体归一、关系合并、图谱、索引文本构造、`dense`/`bm25`/`hybrid` 基础检索，以及
   `graph_local`/`graph_global`/`graph_hybrid` 图谱检索和默认
   `mix`（hybrid + graph_local + graph_global）复合检索、融合评分、
   上下文预算和引用追踪必须由项目实现。
5. 所有第三方实现放在 adapter 后，领域 schema 不暴露框架类型。

## 后果

优点：核心算法可解释、测试容易、框架可替换，也能复用可靠的通用基础设施。

代价：需要维护少量 adapter；第一版轻量 PDF parser 对复杂版面的还原弱于 Docling，
但可以在不改动下游契约的情况下替换。
