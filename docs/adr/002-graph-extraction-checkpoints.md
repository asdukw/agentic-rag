# ADR 002：图抽取、检查点与规范化物化

- 状态：Accepted
- 日期：2026-07-11

## 背景

阶段二需要把不可预测的模型调用接入可恢复的工程流水线，同时保证实体归一、关系合并
和知识图谱仍由本项目控制。模型可能返回空内容、截断内容、语法合法但不满足领域约束的
JSON，也可能在部分 chunk 已成功后遇到限流、网络错误或进程中断。

因此，仅保存最终 NetworkX 文件或仅依赖 LangGraph checkpoint 都不足以满足以下要求：

- 成功 chunk 不因进程中断而重复付费，失败 chunk 能定位到具体请求和校验阶段。
- 更换抽取 prompt 与更换归一规则是两类不同的失效事件，不应一概重新调用模型。
- 每个规范实体和关系都能追溯到来源 chunk 与原始证据，而不是只保留模型摘要。
- 同一输入和同一配置产生稳定的实体、关系与图指标，不受并发完成顺序影响。

## 决策

### 1. DeepSeek 只出现在自有 adapter 后

项目通过 DeepSeek 官方 OpenAI-compatible API 调用模型，但使用自有 client/wrapper，领域
代码不依赖框架 provider 类型。抽取模型名、base URL、超时和并发上限均配置化，API key
只从环境读取且不得写入日志、checkpoint 或失败记录。

每次抽取请求必须显式设置：

```json
{
  "response_format": {"type": "json_object"},
  "thinking": {"type": "disabled"}
}
```

JSON Output 只作为语法层约束，不能替代领域校验。provider/SDK 的隐式重试关闭或归零，
重试由项目代码统一计数和留档，避免实际请求次数不可审计。

### 2. 模型输出使用局部引用，并接受严格领域校验

一次 chunk 抽取中的实体使用响应内局部 ID（例如 `e1`、`e2`），关系端点只能引用同一
响应中已声明的局部实体。模型不生成全局实体 ID，也不生成 `source_chunk_ids`；这些值由
宿主代码根据当前 chunk 和确定性 ID 规则补充。

JSON 解析后必须经过 Pydantic 校验和跨字段校验，包括：

- 实体局部 ID 唯一，关系的起点和终点引用存在。
- 名称、类型、关系类型和证据等必填字段满足长度与空白约束。
- 证据文本或证据 span 能映射回当前 chunk；不能用模型摘要冒充原文证据。
- 当前 chunk 之外的来源 ID 不被接受。
- `{"entities": [], "relations": []}` 是无相关事实 chunk 的合法结果，不触发修复。

持久化的实体和关系领域对象必须公开 `source_chunk_ids`，并通过独立 evidence 记录保留
chunk ID、证据文本及可用的字符 span。模型输出 schema、规范化后的领域 schema 和
NetworkX 属性 schema 相互隔离。

### 3. 抽取配置和图配置使用不同哈希

`extraction_config_hash` 描述会改变模型语义输出的配置，至少包含模型名、抽取/修复 prompt
版本、`glean` prompt 版本与补漏预算、输出 schema 版本以及 JSON/Thinking 模式。chunk
内容或上述任一配置变化时，生成新的 chunk extraction；并发数、批次上限等纯执行参数
不使成功缓存失效。

`graph_config_hash` 描述规范化、候选选择、别名合并、关系方向/类型归一和图物化规则的
版本。仅图配置变化时，从已有的成功抽取重新物化图，不重新调用 DeepSeek。

两个哈希都随 run 和结果留档，禁止把不同配置的中间结果静默混入同一个当前图。

### 4. 领域进度逐 chunk 持久化，attempt 只追加

SQLite 中保存项目拥有的运行与审计状态：

- graph build run 记录本次构建的配置、状态、计数和起止时间。
- chunk extraction 以 chunk 内容和 `extraction_config_hash` 唯一标识，保存当前状态和
  已验证的结构化结果。
- extraction attempt 是 append-only 记录，保存序号、`extract`/`repair`/`glean` 阶段、结果类型、
  provider 请求 ID、token usage、错误摘要以及必要的原始响应；每条 attempt 显式归属
  发起调用的 run，重叠 run 不重复计算同一次调用成本。补漏是独立 attempt，不隐藏在首次
  抽取的响应或成本中。

成功抽取按 chunk 独立提交；单个 chunk 失败不回滚其他 chunk，也不阻断同批后续任务。
原始响应可能包含语料文本，默认报告不输出它，仅在显式诊断时读取。任何保存的请求信息
都必须剔除认证头和 API key。

### 5. LangGraph checkpoint 和领域记录职责分离

LangGraph 负责显式状态图、并行调度、有界修复、interrupt/resume 和节点间状态传递，并
使用持久化 SQLite checkpointer。graph build run ID 同时作为 LangGraph `thread_id`，从而
能够恢复同一执行线程。

checkpoint 是编排恢复数据，不是抽取和图谱的事实来源；`chunk_extractions`、attempt、
entity/relation/evidence 记录才是可查询、可迁移的领域状态。恢复时先用领域记录校准
checkpoint：已成功的 chunk 不再调用模型，遗留的 `running` 状态按可恢复任务重新入队，
不会凭 checkpoint 重复写入实体或证据。

### 6. 人工复核是可选的显式暂停点

默认情况下，严格校验通过的抽取可直接进入规范化。启用人工复核时，对应 build item 进入
`needs_review`，run 进入 `awaiting_review`，LangGraph 在物化之前 interrupt。复核操作必须
记录决定和可选说明；批准后沿同一 `thread_id` 恢复，要求重试的样本生成新的 append-only
attempt，而不是覆写旧响应。

复核状态属于 run 的 build item，而不是跨 run 共享的 extraction 缓存；因此并发的普通
run 和复核 run 不会互相污染。第一版复核决策只提供 approve/reject：拒绝只使对应 run
失败，不修改其他 run 已验证的共享缓存；需要重新抽取时必须创建新的 extraction 配置。
未经批准的数据不得进入当前规范图，人工复核也不会把任意手工文本绕过 Pydantic 和证据
校验直接写入图谱。

### 7. 规范化、关系合并和 ID 由项目代码确定

LangGraph 和 DeepSeek 不负责最终实体消歧或关系合并。项目代码执行版本化、确定性的规则：

- 对实体名称、类型和别名做保守归一，产生候选后按明确规则决定是否合并。
- 关系以规范起点、规范终点、归一关系类型和方向作为合并依据。
- 合并只去除重复事实，不丢弃来源；所有 chunk ID、证据文本和 span 去重后稳定排序。
- canonical entity/relation ID 由规范字段和图配置确定性生成，与并发完成顺序无关。
- 无法可靠归一的候选保持分离，优先保留可解释性而不是激进合并。
- 有证据但暂时没有关系的孤立实体继续保留并计入图统计；它们仍能支持实体向量命中后
  直接回到来源 chunk，因此不把“没有边”当作自动删除条件。

prompt 可以提供候选描述，但不能覆盖这些规则。

### 8. 数据库保存当前规范物化及完整 provenance

关系数据库中的 entity、relation 和 evidence 表表示当前选定图配置的规范物化，而不是
把每次运行的聚合结果无限追加到同一图中。重新物化从符合条件的成功/已批准 extraction
确定性构建当前状态，并清除不再被任何证据支持的规范对象，避免源文档更新后留下幽灵
节点或边。

每个 entity/relation 的 `source_chunk_ids` 从 evidence 记录派生或同步生成；证据记录保留
对 chunk 的外键和原文。文档/chunk 被替换或删除后，相应 extraction 与 evidence 必须通过
外键级联或同一事务清理。run 与 append-only attempt 继续保留审计历史，即使当前图已由
新配置重新物化。

### 9. NetworkX MultiDiGraph 是可重建的运行时视图

当前规范物化可确定性重建为 `networkx.MultiDiGraph`：节点键为 canonical entity ID，边键
为 canonical relation ID。使用有向多重图是为了保留相同实体对之间不同类型或语义的
关系，而不是在加载时覆盖其中一条。

节点和边携带类型、描述、来源计数和 evidence ID 等可解释属性。连通分量统计对有向图
使用 weakly connected components，并采用稳定的并列排序规则输出 Top-K 实体。NetworkX
不是唯一持久化来源；图对象丢失后必须能从数据库完整重建。

### 10. 失败、修复、重试和恢复语义明确

项目区分空内容、截断、非法 JSON、schema 错误、证据错误、provider 可重试错误和 provider
终止错误。语义修复和传输重试各自有界，每一次实际模型请求都生成 attempt。

默认 `max_attempts=2` 时，初始 `extract` 通过校验后，第二次调用用于一次独立 `glean`，要求
模型返回包含基线并补充遗漏事实的完整结果；若初始 `extract` 无效，第二次调用改用于
`repair`，所以同一 chunk 的修复与补漏互斥。`glean` 候选必须再次通过领域校验，并完整
保留基线中的实体、关系及证据；调用失败、结果无效或缺失任一基线事实时，系统采用已经
验证的初始结果，且保留补漏 attempt 供成本与失败审计。

新 run 会复用相同 extraction 配置下的成功缓存。恢复已有 run 时只处理 `pending`、可回收
的 `running` 和未完成任务；已达到上限的终止失败保持失败，只有显式 `retry-failed` 或人工
复核决定才会重新入队。重新入队新增 attempt，不删除失败历史。

中断或部分失败后的 run 可处于 `awaiting_review`、`completed_with_failures` 或 `failed`；
重跑/恢复不得重复生成规范对象或 evidence。run 报告至少保留 chunk 状态计数、attempt
计数、失败分类、token usage、图节点/边/连通分量指标和 Top-K 实体，但默认不包含原始
模型响应。

## 后果

优点：模型调用成本可缓存，任意失败可追踪到 chunk 和 attempt，编排可中断恢复；抽取
prompt 和图算法能够独立演进；规范图与每条证据均可解释并可确定性重建。

代价：需要同时维护 LangGraph checkpoint 与项目领域状态，并在恢复时做一致性校准；
当前图重新物化需要额外事务和孤儿清理；保守归一会暂时保留部分重复实体。

明确不采用的替代方案包括：把 JSON mode 当作完整校验、让模型直接生成全局 ID、只保存
NetworkX pickle、让 LangGraph checkpoint 充当业务数据库、以及用通用 GraphRAG 框架的
预制实体合并或 retriever 取代项目核心逻辑。
