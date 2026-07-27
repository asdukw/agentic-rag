import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ApiError,
  buildWorkspaceGraph,
  buildWorkspaceIndex,
  createWorkspace,
  health,
  ingestWorkspace,
  listWorkspaces,
  runtimeDefaults,
  streamAgentRun,
  uploadWorkspaceFile,
} from "./api";
import type {
  AgentBudget,
  AgentEvent,
  AgentRunRequest,
  EvidenceItem,
  GroundedAnswer,
  Workspace,
} from "./types";

const DEFAULT_BUDGET: AgentBudget = {
  max_steps: 8,
  max_searches: 3,
  max_graph_expansions: 4,
  max_reads: 2,
  max_evidence_chunks: 8,
  max_graph_hops: 2,
  evidence_token_budget: 2400,
};

const TOOL_LABELS: Record<string, { title: string; copy: string }> = {
  search_chunks: { title: "Chunk search", copy: "Dense, BM25, or both — chosen per question." },
  search_entities: { title: "Entity search", copy: "Use entities as graph retrieval anchors." },
  search_relations: {
    title: "Relation search",
    copy: "Search relation semantics and source passages.",
  },
  expand_graph: {
    title: "Graph expansion",
    copy: "Incrementally reveals one-hop neighbors from the current graph frontier.",
  },
  read_evidence: { title: "Read evidence", copy: "Only discovered chunks enter the evidence set." },
  answer_from_evidence: {
    title: "Grounded answer",
    copy: "Citations are verified against read chunks.",
  },
};

export default function App() {
  const [apiBase, setApiBase] = useState(
    () =>
      localStorage.getItem("agentic-rag-lab-api") ??
      localStorage.getItem("hybrid-rag-lab-api") ??
      localStorage.getItem("hybrid-rag-api") ??
      "",
  );
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [workspaceId, setWorkspaceId] = useState("");
  const [workspaceName, setWorkspaceName] = useState("");
  const [profileId, setProfileId] = useState("");
  const [question, setQuestion] = useState("");
  const [useDeepSeek, setUseDeepSeek] = useState(true);
  const [useReranker, setUseReranker] = useState(false);
  const [budget, setBudget] = useState(DEFAULT_BUDGET);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [answer, setAnswer] = useState<GroundedAnswer | null>(null);
  const [evidence, setEvidence] = useState<EvidenceItem[]>([]);
  const [status, setStatus] = useState<"checking" | "online" | "offline">("checking");
  const [running, setRunning] = useState(false);
  const [workspaceBusy, setWorkspaceBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const base = apiBase.trim();
  const runId = events[0]?.run_id;
  const plannerActions = useMemo(
    () => events.filter((event) => event.event === "planner_action"),
    [events],
  );

  useEffect(() => {
    localStorage.setItem("agentic-rag-lab-api", apiBase);
  }, [apiBase]);

  const refreshRuntime = useCallback(async () => {
    setStatus("checking");
    try {
      const [current, defaults, workspaceResponse] = await Promise.all([
        health(base),
        runtimeDefaults(base),
        listWorkspaces(base),
      ]);
      setStatus(current.status === "ok" ? "online" : "offline");
      setBudget((value) => ({ ...value, ...defaults.agent_budget }));
      setWorkspaces(workspaceResponse.workspaces);
      setWorkspaceId((selected) =>
        workspaceResponse.workspaces.some((workspace) => workspace.id === selected)
          ? selected
          : (workspaceResponse.workspaces[0]?.id ?? ""),
      );
    } catch {
      setStatus("offline");
    }
  }, [base]);

  useEffect(() => {
    void refreshRuntime();
  }, [refreshRuntime]);

  async function runAgent() {
    if (!workspaceId) {
      setError("请先创建并选择一个工作区。");
      return;
    }
    if (!question.trim()) {
      setError("请输入问题后再运行 Agent。");
      return;
    }
    setRunning(true);
    setError(null);
    setNotice(null);
    setEvents([]);
    setAnswer(null);
    setEvidence([]);
    const payload: AgentRunRequest = {
      question: question.trim(),
      workspace_id: workspaceId,
      ...(profileId.trim() ? { profile_id: profileId.trim() } : {}),
      use_deepseek: useDeepSeek,
      reranker_enabled: useReranker,
      budget,
    };
    try {
      await streamAgentRun(base, payload, (event) => {
        setEvents((current) => [...current, event]);
        if (event.event === "answer") {
          const data = event.data as { answer?: GroundedAnswer; evidence?: EvidenceItem[] };
          setAnswer(data.answer ?? null);
          setEvidence(data.evidence ?? []);
        }
        if (event.event === "failed") {
          setError(String(event.data.error ?? "Agent 运行失败。"));
        }
        if (event.event === "completed") {
          setNotice(
            `Run ${event.run_id} 已完成：${String(event.data.termination_reason ?? "done")}`,
          );
        }
      });
      void refreshRuntime();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "无法连接到 Python Agent 服务。");
    } finally {
      setRunning(false);
    }
  }

  async function createNewWorkspace() {
    if (!workspaceName.trim()) {
      setError("请输入工作区名称。");
      return;
    }
    setWorkspaceBusy(true);
    setError(null);
    try {
      const workspace = await createWorkspace(base, workspaceName.trim());
      setWorkspaceName("");
      setWorkspaceId(workspace.id);
      setNotice(`已创建工作区：${workspace.name}`);
      await refreshRuntime();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "创建工作区失败。");
    } finally {
      setWorkspaceBusy(false);
    }
  }

  async function uploadDocument(file: File | undefined) {
    if (!workspaceId || !file) return;
    setWorkspaceBusy(true);
    setError(null);
    try {
      const workspace = await uploadWorkspaceFile(base, workspaceId, file);
      setWorkspaces((current) =>
        current.map((item) => (item.id === workspace.id ? workspace : item)),
      );
      setNotice(`已上传 ${file.name}；下一步请点击“导入文档”。`);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "上传失败。");
    } finally {
      setWorkspaceBusy(false);
    }
  }

  async function ingestDocuments() {
    if (!workspaceId) return;
    setWorkspaceBusy(true);
    setError(null);
    setNotice("正在导入并分块上传文档…");
    try {
      const report = await ingestWorkspace(base, workspaceId);
      setNotice(`导入完成：${report.discovered} 文件，写入 ${report.chunks_written} chunks。`);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "导入失败。");
    } finally {
      setWorkspaceBusy(false);
    }
  }

  async function buildGraph() {
    if (!workspaceId) return;
    setWorkspaceBusy(true);
    setError(null);
    setNotice("正在构建知识图谱；这会调用 DeepSeek，耗时取决于文档 chunk 数。");
    try {
      const report = await buildWorkspaceGraph(base, workspaceId);
      setNotice(
        `图谱完成：${report.graph.nodes} nodes / ${report.graph.edges} edges / ${report.chunks.succeeded} chunks。`,
      );
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "构图失败。");
    } finally {
      setWorkspaceBusy(false);
    }
  }

  async function requestIndexBuild() {
    if (!workspaceId) return;
    setWorkspaceBusy(true);
    setError(null);
    setNotice("正在构建索引…");
    try {
      const report = await buildWorkspaceIndex(base, workspaceId);
      setNotice(
        `索引就绪：${report.chunks} chunks / ${report.entities} entities / ${report.relations} relations`,
      );
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "构建索引失败。");
    } finally {
      setWorkspaceBusy(false);
    }
  }

  return (
    <main className="shell">
      <header className="masthead">
        <div>
          <p className="eyebrow">Agentic RAG Lab · Python Agent Runtime</p>
          <h1>Agentic evidence workbench</h1>
          <p className="lede">模型决定下一步检索路径；Python 强制执行证据、图谱和引用边界。</p>
        </div>
        <button className={`status ${status}`} type="button" onClick={() => void refreshRuntime()}>
          <span />
          {status === "online"
            ? "Agent API 已连接"
            : status === "checking"
              ? "检查连接"
              : "API 未连接"}
        </button>
      </header>

      <section className="hero-grid">
        <section className="question-card">
          <div className="section-kicker">01 · Ask</div>
          <label htmlFor="question">研究问题</label>
          <textarea
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="例如：LightRAG 如何将实体和关系用于不同范围的检索？"
            rows={4}
          />
          <div className="run-row">
            <button
              className="run-button"
              type="button"
              onClick={() => void runAgent()}
              disabled={running || workspaceBusy || !workspaceId}
            >
              {running ? "Agent 正在执行…" : "启动 Agent Run"}
            </button>
            <label className="toggle">
              <input
                type="checkbox"
                checked={useDeepSeek}
                onChange={(event) => setUseDeepSeek(event.target.checked)}
              />{" "}
              使用 DeepSeek 规划与回答
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={useReranker}
                onChange={(event) => setUseReranker(event.target.checked)}
              />{" "}
              工具内候选重排
            </label>
          </div>
          <p className="hint">不开启模型时，系统会以确定性策略执行相同的受限工具链。</p>
        </section>

        <aside className="runtime-card">
          <div className="section-kicker">Runtime</div>
          <label>
            API Base URL
            <input
              value={apiBase}
              onChange={(event) => setApiBase(event.target.value)}
              placeholder="同域 /api 留空"
            />
          </label>
          <label>
            当前工作区
            <select value={workspaceId} onChange={(event) => setWorkspaceId(event.target.value)}>
              <option value="">选择工作区</option>
              {workspaces.map((workspace) => (
                <option key={workspace.id} value={workspace.id}>
                  {workspace.name} · {workspace.uploads.length} 文件
                </option>
              ))}
            </select>
          </label>
          <label>
            新建工作区
            <input
              value={workspaceName}
              onChange={(event) => setWorkspaceName(event.target.value)}
              placeholder="例如：LightRAG 论文"
            />
          </label>
          <button
            className="subtle-button"
            type="button"
            disabled={workspaceBusy}
            onClick={() => void createNewWorkspace()}
          >
            创建本地工作区
          </button>
          <label>
            上传语料（PDF / MD / TXT）
            <input
              type="file"
              accept=".pdf,.md,.markdown,.txt"
              disabled={!workspaceId || workspaceBusy}
              onChange={(event) => void uploadDocument(event.target.files?.[0])}
            />
          </label>
          <p className="hint">
            {workspaceId
              ? `已选工作区包含 ${workspaces.find((item) => item.id === workspaceId)?.uploads.length ?? 0} 个上传文件。`
              : "每个工作区使用独立的文档、图谱、索引和 SQLite 数据库。"}
          </p>
          <label>
            Index Profile（可选）
            <input
              value={profileId}
              onChange={(event) => setProfileId(event.target.value)}
              placeholder="idx_…"
            />
          </label>
          <div className="budget-line">
            <span>最多 planner steps</span>
            <input
              type="number"
              min="1"
              max="12"
              value={budget.max_steps}
              onChange={(event) =>
                setBudget({
                  ...budget,
                  max_steps: bounded(event.target.value, 1, 12, budget.max_steps),
                })
              }
            />
          </div>
          <div className="budget-line">
            <span>证据 token 预算</span>
            <input
              type="number"
              min="128"
              max="8000"
              value={budget.evidence_token_budget}
              onChange={(event) =>
                setBudget({
                  ...budget,
                  evidence_token_budget: bounded(
                    event.target.value,
                    128,
                    8000,
                    budget.evidence_token_budget,
                  ),
                })
              }
            />
          </div>
          <button
            className="subtle-button"
            type="button"
            disabled={!workspaceId || workspaceBusy}
            onClick={() => void ingestDocuments()}
          >
            1. 导入上传文档
          </button>
          <button
            className="subtle-button"
            type="button"
            disabled={!workspaceId || workspaceBusy}
            onClick={() => void buildGraph()}
          >
            2. 构建知识图谱
          </button>
          <button
            className="subtle-button"
            type="button"
            disabled={!workspaceId || workspaceBusy}
            onClick={() => void requestIndexBuild()}
          >
            3. 构建 / 复用索引
          </button>
        </aside>
      </section>

      {(error || notice) && (
        <div className={error ? "banner error" : "banner"}>{error ?? notice}</div>
      )}

      <section className="tool-strip" aria-label="Agent 工具边界">
        {Object.entries(TOOL_LABELS).map(([key, value]) => (
          <article key={key}>
            <span>{String(Object.keys(TOOL_LABELS).indexOf(key) + 1).padStart(2, "0")}</span>
            <h2>{value.title}</h2>
            <p>{value.copy}</p>
          </article>
        ))}
      </section>

      <section className="run-layout">
        <section className="timeline-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">02 · Observe</p>
              <h2>Agent timeline</h2>
            </div>
            <span>{runId ?? "等待运行"}</span>
          </div>
          {!events.length ? (
            <EmptyTimeline />
          ) : (
            <ol className="timeline">
              {events.map((event, index) => (
                <TimelineEvent
                  event={event}
                  index={index}
                  key={`${event.run_id}-${event.step}-${event.event}`}
                />
              ))}
            </ol>
          )}
        </section>

        <section className="answer-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">03 · Verify</p>
              <h2>Grounded answer</h2>
            </div>
            <span>{answer?.citations.length ?? 0} citations</span>
          </div>
          {answer ? (
            <>
              <p className={answer.insufficient_evidence ? "answer insufficient" : "answer"}>
                {answer.answer}
              </p>
              <div className="citation-row">
                {answer.citations.map((citation) => (
                  <span key={citation}>{citation}</span>
                ))}
              </div>
            </>
          ) : (
            <p className="empty-copy">
              回答只会在 Agent 读取证据后出现。引用仅能来自本次会话已经读取的 chunk。
            </p>
          )}
        </section>
      </section>

      <section className="evidence-panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Evidence set</p>
            <h2>已读取的原始文档片段</h2>
          </div>
          <span>{evidence.length} chunks</span>
        </div>
        {evidence.length ? (
          <div className="evidence-grid">
            {evidence.map((item) => (
              <EvidenceCard item={item} key={item.chunk_id} />
            ))}
          </div>
        ) : (
          <p className="empty-copy">
            Agent 的实体、关系与路径只是检索线索；只有这里的原文 chunk 可以支持最终回答。
          </p>
        )}
      </section>

      {plannerActions.length > 0 && (
        <details className="raw-log">
          <summary>查看结构化规划记录</summary>
          <pre>{JSON.stringify(plannerActions, null, 2)}</pre>
        </details>
      )}
    </main>
  );
}

function TimelineEvent({ event, index }: { event: AgentEvent; index: number }) {
  const detail =
    event.event === "planner_action"
      ? String(event.data.rationale ?? event.data.action ?? "")
      : event.event === "tool_result"
        ? String(event.data.summary ?? "")
        : event.event === "answer"
          ? "答案与可引用证据已准备。"
          : event.event === "run_started"
            ? `固定 profile: ${String(event.data.profile_id ?? "—")}`
            : String(event.data.termination_reason ?? event.data.error ?? "");
  return (
    <li className={`event ${event.event}`}>
      <span className="event-index">{String(index + 1).padStart(2, "0")}</span>
      <div>
        <strong>{labelFor(event)}</strong>
        <p>{detail}</p>
        {event.event === "tool_result" && (
          <details>
            <summary>工具结果</summary>
            <pre>{JSON.stringify(event.data.data ?? {}, null, 2)}</pre>
          </details>
        )}
      </div>
    </li>
  );
}

function EvidenceCard({ item }: { item: EvidenceItem }) {
  const section = item.section_path?.join(" / ") || "未分段";
  return (
    <article className="evidence-card">
      <div className="evidence-meta">
        <span>{item.citation_id}</span>
        <span>{item.token_count} tokens</span>
      </div>
      <h3>{item.document_title}</h3>
      <p className="section">{section}</p>
      <p>{item.text}</p>
    </article>
  );
}

function EmptyTimeline() {
  return (
    <div className="empty-timeline">
      <span>◌</span>
      <p>每一步会显示 planner 选择的工具、工具摘要、证据读取与终止原因。</p>
    </div>
  );
}

function bounded(value: string, min: number, max: number, fallback: number) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? Math.max(min, Math.min(max, parsed)) : fallback;
}

function labelFor(event: AgentEvent) {
  if (event.event === "tool_result") {
    return `工具已执行 · ${toolLabel(event.data.tool)}`;
  }
  return {
    run_started: "会话已固定",
    planner_action: "Planner 决策",
    answer: "证据约束回答",
    completed: "运行完成",
    failed: "运行失败",
  }[event.event];
}

function toolLabel(value: unknown) {
  const tool = typeof value === "string" ? value : "unknown";
  return (
    {
      search_chunks: "Chunk 检索（search_chunks）",
      search_entities: "实体检索（search_entities）",
      search_relations: "关系检索（search_relations）",
      expand_graph: "图谱扩展（expand_graph）",
      read_evidence: "读取证据（read_evidence）",
      answer_from_evidence: "基于证据回答（answer_from_evidence）",
    }[tool] ?? tool
  );
}
