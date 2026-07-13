"""Streamlit evidence-first demo for the Hybrid RAG retrieval pipeline.

Run from the repository root after ingesting documents, building a graph, and
building an index::

    uv run streamlit run src/hybrid_rag/demo.py

The UI deliberately invokes only fixed retrieval modes.  A model is optional
and can perform only the two bounded operations exposed by ``QueryClient``:
keyword extraction and an answer over already selected evidence.
"""

# ruff: noqa: RUF001

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypeVar

from hybrid_rag.config import DeepSeekSettings, RetrievalSettings, Settings, sqlite_url
from hybrid_rag.ingest.tokenizer import TiktokenCounter
from hybrid_rag.retrieval.embedding import (
    BGEM3EmbeddingProvider,
    HashEmbeddingProvider,
)
from hybrid_rag.retrieval.models import (
    CandidateHit,
    GraphPath,
    RerankTrace,
    RetrievalMode,
    RetrievalResult,
)
from hybrid_rag.retrieval.query import (
    DeepSeekQueryClient,
    DeterministicQueryClient,
    QueryClient,
)
from hybrid_rag.retrieval.reranker import create_reranker
from hybrid_rag.retrieval.service import AnswerResult, RetrievalOptions, RetrievalService
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.migrations import upgrade_database

_T = TypeVar("_T")
Language: TypeAlias = Literal["en", "zh"]

_LANGUAGE_OPTIONS: dict[str, Language] = {"中文": "zh", "English": "en"}
_MODE_LABELS: dict[Language, dict[RetrievalMode, str]] = {
    "en": {
        RetrievalMode.NAIVE: "Naive — chunk dense + BM25 recall",
        RetrievalMode.LOCAL: "Local — entity-led graph recall",
        RetrievalMode.GLOBAL: "Global — relation-led graph recall",
        RetrievalMode.HYBRID: "Hybrid — fixed three-route fusion",
    },
    "zh": {
        RetrievalMode.NAIVE: "Naive — Chunk 向量 + BM25 词法召回",
        RetrievalMode.LOCAL: "Local — 以实体为起点的图谱召回",
        RetrievalMode.GLOBAL: "Global — 以关系为起点的图谱召回",
        RetrievalMode.HYBRID: "Hybrid — 三路固定融合",
    },
}
_TEXT: dict[Language, dict[str, str]] = {
    "en": {
        "title": "Hybrid RAG · evidence-first retrieval demo",
        "subtitle": (
            "Each request uses one explicitly selected retriever. Hybrid is a deterministic fusion "
            "of naive, local, and global routes; no agent may select tools or hidden sources."
        ),
        "execution_settings": "Execution settings",
        "database_url": "SQLite database path or URL",
        "database_help": (
            "A path is converted to an absolute SQLite URL. Existing CLI defaults also work."
        ),
        "embedding_provider": "Embedding provider",
        "embedding_model": "Embedding model",
        "embedding_dimensions": "Embedding dimensions",
        "embedding_help": (
            "`flagembedding` uses local BGE-M3 by default. `hash` is retained only for "
            "compatibility and tests."
        ),
        "retrieval_budget": "Retrieval budget",
        "top_k": "Top K",
        "context_token_budget": "Context token budget",
        "maximum_graph_hops": "Maximum graph hops",
        "reranking": "Reranking",
        "enable_reranker": "Enable configured reranker",
        "reranker_help": (
            "Uses the configured local FlagEmbedding cross-encoder. Turn this off to keep "
            "first-stage order; model weights load only when reranking runs."
        ),
        "rerank_candidate_multiplier": "Rerank candidate multiplier",
        "rerank_candidate_help": "Rerank Top K × this many first-stage candidates.",
        "use_deepseek": "Use DeepSeek for keywords + answer",
        "deepseek_help": (
            "When off, the demo uses deterministic offline keyword and answer functions."
        ),
        "deepseek_warning": (
            "DeepSeek is limited to keyword extraction and an answer over the displayed citation "
            "allowlist; it cannot choose retrieval tools or sources."
        ),
        "build_index": "Build index",
        "building_index": "Embedding chunk, entity, and relation index texts...",
        "index_ready": "Ready: {chunks} chunks, {entities} entities, {relations} relations",
        "index_build_failed": "Index build failed",
        "replay_trace": "Replay trace",
        "replay": "Replay",
        "replay_missing": "Enter an rtr_ trace ID to replay.",
        "replay_failed": "Replay failed",
        "replayed_trace": "Replayed trace",
        "stored_answer": "Stored answer",
        "retriever_mode": "Retriever mode",
        "question": "Question",
        "question_placeholder": "How does LightRAG use entities?",
        "retrieve_and_answer": "Retrieve and answer",
        "compare_naive_hybrid": "Compare naive vs hybrid",
        "compare_help": (
            "Runs exactly the naive and hybrid modes, then shows their evidence and answers "
            "side by side."
        ),
        "question_required": "Enter a question first.",
        "retrieving": "Retrieving only from the selected index and graph snapshot...",
        "request_failed": "Request failed",
        "naive_vs_hybrid": "Naive vs hybrid",
        "comparison_caption": (
            "Both columns were run as named, fixed modes. The answer client received only each "
            "column's displayed citation context."
        ),
        "answer": "Answer",
        "citations": "Citations",
        "mode": "Mode",
        "profile": "profile",
        "trace": "trace",
        "not_persisted": "not persisted",
        "context_tokens": "Context tokens",
        "context_chunks": "Context chunks",
        "fused_hits": "Final hits",
        "graph_paths": "Graph paths",
        "citation_context": "Citation context",
        "no_context": "No chunk fit the selected context budget.",
        "unsectioned": "Unsectioned",
        "page_unknown": "page unknown",
        "page": "page {page}",
        "pages": "pages {start}-{end}",
        "document": "document",
        "tokens": "tokens",
        "score": "score",
        "graph_support": "graph support",
        "entities": "entities",
        "relations": "relations",
        "route_scores_candidates": "Route scores and fused candidates",
        "rerank_trace": "Rerank decisions",
        "no_rerank": "Reranking is disabled for this request.",
        "no_route_candidates": "This trace did not retain route candidates.",
        "final_context_trace": "Final context and replayable trace",
        "no_graph_path": "No graph path was needed for this result.",
        "route": "route",
        "candidates": "candidates",
        "top_object": "top object",
        "top_score": "top score",
        "object": "object",
        "kind": "kind",
        "rank": "rank",
        "pre_rerank_rank": "pre-rerank rank",
        "pre_rerank_score": "pre-rerank score",
        "rerank_score": "rerank score",
        "retrieval_score": "retrieval score",
        "route_scores": "route scores",
        "score_components": "score components",
        "source_chunks": "source chunks",
        "nodes": "nodes",
        "supporting_chunks": "supporting chunks",
        "kind_chunk": "chunk",
        "kind_entity": "entity",
        "kind_relation": "relation",
    },
    "zh": {
        "title": "Hybrid RAG · 证据优先检索演示",
        "subtitle": (
            "每次请求只使用一个明确选择的检索器。Hybrid 是 naive、local 与 global 路径的"
            "确定性融合；Agent 不能自行选择工具或隐藏信息源。"
        ),
        "execution_settings": "执行设置",
        "database_url": "SQLite 数据库路径或 URL",
        "database_help": "文件路径会转换为绝对 SQLite URL；也可直接使用 CLI 的默认值。",
        "embedding_provider": "Embedding 提供方",
        "embedding_model": "Embedding 模型",
        "embedding_dimensions": "Embedding 维度",
        "embedding_help": "默认 `flagembedding` 使用本地 BGE-M3；`hash` 仅保留用于兼容和测试。",
        "retrieval_budget": "检索预算",
        "top_k": "Top K",
        "context_token_budget": "上下文 Token 预算",
        "maximum_graph_hops": "最大图跳数",
        "reranking": "重排序",
        "enable_reranker": "启用已配置的重排序器",
        "reranker_help": (
            "使用已配置的本地 FlagEmbedding cross-encoder；关闭后保留首阶段排序，"
            "仅在实际精排时下载并加载模型权重。"
        ),
        "rerank_candidate_multiplier": "重排序候选倍率",
        "rerank_candidate_help": "对 Top K × 此倍率的首阶段候选进行重排序。",
        "use_deepseek": "使用 DeepSeek 提取关键词并生成回答",
        "deepseek_help": "关闭时，演示会使用确定性的离线关键词和回答函数。",
        "deepseek_warning": (
            "DeepSeek 仅能提取关键词，并基于页面展示的 citation 白名单生成回答；"
            "它不能选择检索工具或信息源。"
        ),
        "build_index": "构建索引",
        "building_index": "正在为 Chunk、实体和关系构建 embedding 索引…",
        "index_ready": "索引就绪：{chunks} 个 Chunk、{entities} 个实体、{relations} 条关系",
        "index_build_failed": "构建索引失败",
        "replay_trace": "重放 Trace",
        "replay": "重放",
        "replay_missing": "请输入要重放的 rtr_ trace ID。",
        "replay_failed": "重放失败",
        "replayed_trace": "已重放的 Trace",
        "stored_answer": "已存储的回答",
        "retriever_mode": "检索模式",
        "question": "问题",
        "question_placeholder": "LightRAG 如何使用实体？",
        "retrieve_and_answer": "检索并回答",
        "compare_naive_hybrid": "对比 naive 与 hybrid",
        "compare_help": "只运行 naive 和 hybrid 两种固定模式，并并排展示它们的证据和回答。",
        "question_required": "请先输入问题。",
        "retrieving": "仅从选定的索引与图谱快照中检索…",
        "request_failed": "请求失败",
        "naive_vs_hybrid": "Naive 与 Hybrid 对比",
        "comparison_caption": (
            "两列均按指定的固定模式运行；回答客户端只能看到各列展示的 citation 上下文。"
        ),
        "answer": "回答",
        "citations": "引用",
        "mode": "模式",
        "profile": "Profile",
        "trace": "Trace",
        "not_persisted": "未持久化",
        "context_tokens": "上下文 Token",
        "context_chunks": "上下文 Chunk",
        "fused_hits": "最终命中",
        "graph_paths": "图路径",
        "citation_context": "引用上下文",
        "no_context": "没有 Chunk 能放入当前上下文预算。",
        "unsectioned": "未分章节",
        "page_unknown": "页码未知",
        "page": "第 {page} 页",
        "pages": "第 {start}-{end} 页",
        "document": "文档",
        "tokens": "Token",
        "score": "分数",
        "graph_support": "图谱支持",
        "entities": "实体",
        "relations": "关系",
        "route_scores_candidates": "路由分数与融合候选",
        "rerank_trace": "重排序决策",
        "no_rerank": "本次请求未启用重排序。",
        "no_route_candidates": "此 trace 未保留路由候选。",
        "final_context_trace": "最终上下文与可重放 Trace",
        "no_graph_path": "此结果不需要图路径。",
        "route": "路由",
        "candidates": "候选数",
        "top_object": "最高对象",
        "top_score": "最高分数",
        "object": "对象",
        "kind": "类型",
        "rank": "排名",
        "pre_rerank_rank": "重排序前排名",
        "pre_rerank_score": "重排序前分数",
        "rerank_score": "重排序分数",
        "retrieval_score": "召回分数",
        "route_scores": "路由分数",
        "score_components": "分数分量",
        "source_chunks": "来源 Chunk",
        "nodes": "节点",
        "supporting_chunks": "支持 Chunk",
        "kind_chunk": "Chunk",
        "kind_entity": "实体",
        "kind_relation": "关系",
    },
}


def ui_text(language: Language, key: str, /, **values: object) -> str:
    """Return one UI string while keeping language choices outside domain code."""

    return _TEXT[language][key].format(**values)


@dataclass(frozen=True, slots=True)
class DemoRuntime:
    """All execution choices exposed by the demo sidebar."""

    database_url: str
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    options: RetrievalOptions
    use_deepseek: bool


def database_url_from_input(value: str, settings: Settings) -> str:
    """Accept a SQLite URL or a friendly filesystem path from the UI."""

    normalized = value.strip()
    if not normalized:
        return settings.database_url
    if "://" in normalized:
        return normalized
    return sqlite_url(Path(normalized))


def create_embedding_provider(runtime: DemoRuntime, settings: RetrievalSettings):
    """Construct the configured adapter without importing CLI-only helpers."""

    if runtime.embedding_provider == "flagembedding":
        return BGEM3EmbeddingProvider(
            model=runtime.embedding_model,
            dimensions=runtime.embedding_dimensions,
            batch_size=settings.embedding_batch_size,
            max_length=settings.embedding_max_length,
            use_fp16=settings.embedding_use_fp16,
        )
    if runtime.embedding_provider == "hash":
        return HashEmbeddingProvider(
            dimensions=runtime.embedding_dimensions,
            model=runtime.embedding_model,
        )
    raise ValueError("embedding provider must be 'flagembedding' or 'hash'")


def create_query_client(use_deepseek: bool, settings: DeepSeekSettings) -> QueryClient:
    """Select an explicit offline client or the constrained DeepSeek adapter."""

    if not use_deepseek:
        return DeterministicQueryClient()
    api_key = settings.api_key.get_secret_value().strip() if settings.api_key else ""
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is required when DeepSeek is enabled")
    return DeepSeekQueryClient(
        api_key=api_key,
        model=settings.query_model,
        answer_model=settings.answer_model,
        base_url=settings.base_url,
        max_output_tokens=max(512, settings.answer_max_output_tokens),
        timeout_seconds=settings.timeout_seconds,
    )


def create_service(
    runtime: DemoRuntime,
    settings: Settings,
    retrieval: RetrievalSettings,
) -> tuple[Database, RetrievalService]:
    """Upgrade the selected database and return a short-lived retrieval service."""

    upgrade_database(runtime.database_url)
    database = Database(runtime.database_url)
    service = RetrievalService(
        database,
        create_embedding_provider(runtime, retrieval),
        TiktokenCounter(settings.tokenizer_name),
        reranker=create_reranker(
            runtime.options.reranker_provider,
            runtime.options.reranker_model,
            use_fp16=runtime.options.reranker_use_fp16,
        ),
    )
    return database, service


async def ask_question(
    service: RetrievalService,
    question: str,
    *,
    mode: RetrievalMode,
    options: RetrievalOptions,
    use_deepseek: bool,
    deepseek_settings: DeepSeekSettings,
) -> AnswerResult:
    """Run the fixed retrieve-then-answer flow and close an online adapter."""

    client = create_query_client(use_deepseek, deepseek_settings)
    try:
        return await service.ask(question, query_client=client, mode=mode, options=options)
    finally:
        await _close_client(client)


async def compare_naive_and_hybrid(
    service: RetrievalService,
    question: str,
    *,
    options: RetrievalOptions,
    use_deepseek: bool,
    deepseek_settings: DeepSeekSettings,
) -> tuple[AnswerResult, AnswerResult]:
    """Execute two named modes only; no model controls the retrieval route."""

    client = create_query_client(use_deepseek, deepseek_settings)
    try:
        naive = await service.ask(
            question,
            query_client=client,
            mode=RetrievalMode.NAIVE,
            options=options,
        )
        hybrid = await service.ask(
            question,
            query_client=client,
            mode=RetrievalMode.HYBRID,
            options=options,
        )
        return naive, hybrid
    finally:
        await _close_client(client)


async def _close_client(client: object) -> None:
    close = getattr(client, "close", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def run_async(coroutine: Coroutine[Any, Any, _T]) -> _T:
    """Run a request in Streamlit's normal synchronous script execution model."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    raise RuntimeError("the demo cannot start an async request inside an active event loop")


def route_rows(result: RetrievalResult, *, language: Language = "en") -> list[dict[str, Any]]:
    """Compact, JSON-safe route summaries for a Streamlit table."""

    rows: list[dict[str, Any]] = []
    for route_name, route in result.trace.routes.items():
        top = route.hits[0] if route.hits else None
        rows.append(
            {
                ui_text(language, "route"): route_name,
                ui_text(language, "candidates"): route.candidate_count,
                ui_text(language, "top_object"): top.object_id if top else "—",
                ui_text(language, "top_score"): round(top.score, 4) if top else None,
                ui_text(language, "score_components"): (
                    _format_score_components(top.score_components) if top else "—"
                ),
            }
        )
    return rows


def hit_rows(
    hits: tuple[CandidateHit, ...],
    *,
    language: Language = "en",
) -> list[dict[str, Any]]:
    """Flatten fused score provenance without hiding per-route contributions."""

    return [
        {
            ui_text(language, "object"): hit.object_id,
            ui_text(language, "kind"): _kind_label(hit.kind, language),
            ui_text(language, "score"): round(hit.score, 4),
            ui_text(language, "rank"): hit.rank,
            ui_text(language, "retrieval_score"): (
                round(hit.retrieval_score, 4) if hit.retrieval_score is not None else "—"
            ),
            ui_text(language, "rerank_score"): (
                round(hit.rerank_score, 4) if hit.rerank_score is not None else "—"
            ),
            ui_text(language, "route_scores"): _format_scores(hit.route_scores),
            ui_text(language, "score_components"): _format_score_components(hit.score_components),
            ui_text(language, "source_chunks"): ", ".join(hit.source_chunk_ids),
        }
        for hit in hits
    ]


def rerank_rows(rerank: RerankTrace, *, language: Language = "en") -> list[dict[str, Any]]:
    """Flatten the second-stage decision without hiding recall provenance."""

    return [
        {
            ui_text(language, "object"): hit.object_id,
            ui_text(language, "pre_rerank_rank"): hit.pre_rerank_rank,
            ui_text(language, "pre_rerank_score"): round(hit.pre_rerank_score, 4),
            ui_text(language, "rerank_score"): round(hit.score, 4),
            ui_text(language, "rank"): hit.final_rank,
            ui_text(language, "score_components"): _format_score_components(hit.components),
        }
        for hit in rerank.hits
    ]


def graph_path_rows(
    paths: tuple[GraphPath, ...],
    *,
    language: Language = "en",
) -> list[dict[str, Any]]:
    """Present retained NetworkX paths exactly as the trace recorded them."""

    return [
        {
            ui_text(language, "nodes"): " → ".join(path.node_ids),
            ui_text(language, "relations"): " → ".join(path.relation_ids),
            ui_text(language, "supporting_chunks"): ", ".join(path.source_chunk_ids),
            ui_text(language, "score"): round(path.score, 4),
        }
        for path in paths
    ]


def _format_scores(scores: dict[str, float]) -> str:
    return ", ".join(f"{route}={score:.4f}" for route, score in sorted(scores.items()))


def _format_score_components(components: dict[str, Any]) -> str:
    if not components:
        return "—"
    return "; ".join(
        (
            f"{name}: raw={component.raw_score:.4f}, "
            f"norm={component.normalized_score:.4f}, "
            f"weighted={component.weighted_score:.4f}"
        )
        for name, component in sorted(components.items())
    )


def _kind_label(kind: str, language: Language) -> str:
    return _TEXT[language].get(f"kind_{kind}", kind)


def _render_answer(
    st: Any,
    result: AnswerResult,
    *,
    language: Language,
    title: str | None = None,
) -> None:
    st.subheader(title or ui_text(language, "answer"))
    if result.answer.insufficient_evidence:
        st.warning(result.answer.answer)
    else:
        st.write(result.answer.answer)
        st.caption(ui_text(language, "citations") + ": " + ", ".join(result.answer.citations))
    _render_retrieval(st, result.retrieval, language=language)


def _render_retrieval(st: Any, result: RetrievalResult, *, language: Language) -> None:
    """Expose citations, vector routes, graph support, and replayable decisions."""

    st.caption(
        f"{ui_text(language, 'mode')}: `{result.mode.value}` · "
        f"{ui_text(language, 'profile')}: `{result.profile_id}` · "
        f"{ui_text(language, 'trace')}: `{result.trace_id or ui_text(language, 'not_persisted')}`"
    )
    metrics = st.columns(4)
    metrics[0].metric(ui_text(language, "context_tokens"), result.context_tokens)
    metrics[1].metric(ui_text(language, "context_chunks"), len(result.context_items))
    metrics[2].metric(ui_text(language, "fused_hits"), len(result.hits))
    metrics[3].metric(ui_text(language, "graph_paths"), len(result.graph_paths))

    st.markdown("#### " + ui_text(language, "citation_context"))
    if not result.context_items:
        st.info(ui_text(language, "no_context"))
    for item in result.context_items:
        section = (
            " / ".join(item.section_path) if item.section_path else ui_text(language, "unsectioned")
        )
        pages = _page_label(item.page_start, item.page_end, language=language)
        with st.expander(f"[{item.citation_id}] {item.document_title} — {section}"):
            st.caption(
                f"{ui_text(language, 'document')} `{item.document_id}` · {pages} · "
                f"{item.token_count} {ui_text(language, 'tokens')} · "
                f"{ui_text(language, 'score')} {item.score:.4f} · "
                f"{_format_scores(item.route_scores)}"
            )
            if item.source_entity_ids or item.source_relation_ids:
                st.caption(
                    f"{ui_text(language, 'graph_support')}: "
                    f"{ui_text(language, 'entities')}={', '.join(item.source_entity_ids) or '—'}; "
                    f"{ui_text(language, 'relations')}={', '.join(item.source_relation_ids) or '—'}"
                )
            st.write(item.text)

    with st.expander(ui_text(language, "route_scores_candidates"), expanded=False):
        rows = route_rows(result, language=language)
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.info(ui_text(language, "no_route_candidates"))
        fused = hit_rows(result.hits, language=language)
        if fused:
            st.dataframe(fused, width="stretch", hide_index=True)
        if result.trace.rerank is None:
            st.caption(ui_text(language, "no_rerank"))
        else:
            rerank = result.trace.rerank
            st.markdown("#### " + ui_text(language, "rerank_trace"))
            st.caption(
                f"{rerank.provider} / {rerank.model} / {rerank.version} · "
                f"{ui_text(language, 'candidates')} ≤ {rerank.candidate_limit}"
            )
            st.dataframe(rerank_rows(rerank, language=language), width="stretch", hide_index=True)

    with st.expander(ui_text(language, "graph_paths"), expanded=False):
        paths = graph_path_rows(result.graph_paths, language=language)
        if paths:
            st.dataframe(paths, width="stretch", hide_index=True)
        else:
            st.info(ui_text(language, "no_graph_path"))

    with st.expander(ui_text(language, "final_context_trace"), expanded=False):
        st.code(result.context, language="text")
        st.json(result.trace.model_dump(mode="json"))


def _page_label(page_start: int | None, page_end: int | None, *, language: Language = "en") -> str:
    if page_start is None:
        return ui_text(language, "page_unknown")
    if page_end is None or page_end == page_start:
        return ui_text(language, "page", page=page_start)
    return ui_text(language, "pages", start=page_start, end=page_end)


def _language_from_sidebar(st: Any) -> Language:
    """Return the persisted UI language before rendering localized controls."""

    selected = st.sidebar.selectbox(
        "语言 / Language",
        options=tuple(_LANGUAGE_OPTIONS),
        index=0,
        key="hybrid_rag_language",
    )
    return _LANGUAGE_OPTIONS[selected]


def _runtime_from_sidebar(
    st: Any,
    *,
    language: Language,
) -> tuple[DemoRuntime, Settings, RetrievalSettings, DeepSeekSettings]:
    settings = Settings()
    retrieval = RetrievalSettings()
    deepseek = DeepSeekSettings()

    st.sidebar.header(ui_text(language, "execution_settings"))
    database_input = st.sidebar.text_input(
        ui_text(language, "database_url"),
        value=settings.database_url,
        help=ui_text(language, "database_help"),
        key="hybrid_rag_database_input",
    )
    available_providers = ("flagembedding", "hash")
    default_provider = (
        retrieval.embedding_provider
        if retrieval.embedding_provider in available_providers
        else "flagembedding"
    )
    provider = st.sidebar.selectbox(
        ui_text(language, "embedding_provider"),
        options=available_providers,
        index=available_providers.index(default_provider),
        key="hybrid_rag_embedding_provider",
    )
    model = st.sidebar.text_input(
        ui_text(language, "embedding_model"),
        value=retrieval.embedding_model,
        key="hybrid_rag_embedding_model",
    )
    dimensions = int(
        st.sidebar.number_input(
            ui_text(language, "embedding_dimensions"),
            min_value=32,
            max_value=4096,
            value=retrieval.embedding_dimensions,
            step=1,
            key="hybrid_rag_embedding_dimensions",
        )
    )
    st.sidebar.caption(ui_text(language, "embedding_help"))

    st.sidebar.subheader(ui_text(language, "retrieval_budget"))
    top_k = int(
        st.sidebar.number_input(
            ui_text(language, "top_k"),
            min_value=1,
            max_value=100,
            value=retrieval.top_k,
            key="hybrid_rag_top_k",
        )
    )
    context_budget = int(
        st.sidebar.number_input(
            ui_text(language, "context_token_budget"),
            min_value=128,
            max_value=20_000,
            value=retrieval.context_token_budget,
            step=128,
            key="hybrid_rag_context_budget",
        )
    )
    graph_hops = int(
        st.sidebar.slider(
            ui_text(language, "maximum_graph_hops"),
            min_value=1,
            max_value=4,
            value=retrieval.graph_max_hops,
            key="hybrid_rag_graph_hops",
        )
    )
    st.sidebar.subheader(ui_text(language, "reranking"))
    rerank_enabled = st.sidebar.toggle(
        ui_text(language, "enable_reranker"),
        value=retrieval.reranker_provider != "none",
        help=ui_text(language, "reranker_help"),
        key="hybrid_rag_rerank_enabled",
    )
    rerank_candidate_multiplier = int(
        st.sidebar.number_input(
            ui_text(language, "rerank_candidate_multiplier"),
            min_value=1,
            max_value=32,
            value=retrieval.rerank_candidate_multiplier,
            help=ui_text(language, "rerank_candidate_help"),
            key="hybrid_rag_rerank_candidate_multiplier",
        )
    )
    use_deepseek = st.sidebar.checkbox(
        ui_text(language, "use_deepseek"),
        value=False,
        help=ui_text(language, "deepseek_help"),
        key="hybrid_rag_use_deepseek",
    )
    if use_deepseek:
        st.sidebar.warning(ui_text(language, "deepseek_warning"))

    options = RetrievalOptions(
        top_k=top_k,
        candidate_multiplier=retrieval.candidate_multiplier,
        context_token_budget=context_budget,
        graph_max_hops=graph_hops,
        naive_weight=retrieval.naive_weight,
        local_weight=retrieval.local_weight,
        global_weight=retrieval.global_weight,
        naive_dense_weight=retrieval.naive_dense_weight,
        naive_bm25_weight=retrieval.naive_bm25_weight,
        bm25_k1=retrieval.bm25_k1,
        bm25_b=retrieval.bm25_b,
        reranker_provider=retrieval.reranker_provider if rerank_enabled else "none",
        reranker_model=retrieval.reranker_model,
        reranker_use_fp16=retrieval.reranker_use_fp16,
        rerank_candidate_multiplier=rerank_candidate_multiplier,
    )
    runtime = DemoRuntime(
        database_url=database_url_from_input(database_input, settings),
        embedding_provider=provider,
        embedding_model=model.strip(),
        embedding_dimensions=dimensions,
        options=options,
        use_deepseek=use_deepseek,
    )
    return runtime, settings, retrieval, deepseek


def _with_service(
    runtime: DemoRuntime,
    settings: Settings,
    retrieval: RetrievalSettings,
    action: Callable[[RetrievalService], _T],
) -> _T:
    database, service = create_service(runtime, settings, retrieval)
    try:
        return action(service)
    finally:
        database.dispose()


def main() -> None:
    """Render the Streamlit application; import Streamlit only at UI startup."""

    try:
        import streamlit as st
    except ImportError as error:  # pragma: no cover - exercised only without the optional UI extra.
        raise RuntimeError(
            "Streamlit is required for the demo. Install project dependencies, then run "
            "`uv run streamlit run src/hybrid_rag/demo.py`."
        ) from error

    st.set_page_config(page_title="Hybrid RAG", page_icon="◈", layout="wide")
    language = _language_from_sidebar(st)
    st.title(ui_text(language, "title"))
    st.caption(ui_text(language, "subtitle"))

    runtime, settings, retrieval_settings, deepseek_settings = _runtime_from_sidebar(
        st,
        language=language,
    )
    build_col, replay_col = st.sidebar.columns(2)
    if build_col.button(
        ui_text(language, "build_index"),
        width="stretch",
        key="hybrid_rag_build_index",
    ):
        try:
            with st.spinner(ui_text(language, "building_index")):
                report = _with_service(
                    runtime,
                    settings,
                    retrieval_settings,
                    lambda service: service.build_index(),
                )
            st.sidebar.success(
                ui_text(
                    language,
                    "index_ready",
                    chunks=report.chunks,
                    entities=report.entities,
                    relations=report.relations,
                )
            )
            st.sidebar.caption(f"{ui_text(language, 'profile')} `{report.profile_id}`")
        except Exception as error:  # UI boundary: preserve detailed domain error text.
            st.sidebar.error(
                f"{ui_text(language, 'index_build_failed')}: {type(error).__name__}: {error}"
            )

    replay_trace_id = replay_col.text_input(
        ui_text(language, "replay_trace"),
        placeholder="rtr_…",
        label_visibility="collapsed",
        key="hybrid_rag_replay_trace_id",
    )
    if replay_col.button(
        ui_text(language, "replay"),
        width="stretch",
        key="hybrid_rag_replay",
    ):
        if not replay_trace_id.strip():
            st.sidebar.warning(ui_text(language, "replay_missing"))
        else:
            try:
                replayed = _with_service(
                    runtime,
                    settings,
                    retrieval_settings,
                    lambda service: (
                        service.replay_answer(replay_trace_id.strip())
                        or service.replay(replay_trace_id.strip())
                    ),
                )
                st.session_state["hybrid_rag_replay"] = replayed
            except Exception as error:  # UI boundary: preserve detailed domain error text.
                st.sidebar.error(
                    f"{ui_text(language, 'replay_failed')}: {type(error).__name__}: {error}"
                )

    if replayed := st.session_state.get("hybrid_rag_replay"):
        st.subheader(ui_text(language, "replayed_trace"))
        if isinstance(replayed, AnswerResult):
            _render_answer(
                st,
                replayed,
                language=language,
                title=ui_text(language, "stored_answer"),
            )
        else:
            _render_retrieval(st, replayed, language=language)

    mode = st.radio(
        ui_text(language, "retriever_mode"),
        options=tuple(RetrievalMode),
        format_func=lambda value: _MODE_LABELS[language][value],
        horizontal=True,
        index=tuple(RetrievalMode).index(RetrievalMode.HYBRID),
        key="hybrid_rag_mode",
    )
    question = st.text_area(
        ui_text(language, "question"),
        placeholder=ui_text(language, "question_placeholder"),
        height=96,
        key="hybrid_rag_question",
    )
    query_col, comparison_col = st.columns(2)
    query_clicked = query_col.button(
        ui_text(language, "retrieve_and_answer"),
        type="primary",
        width="stretch",
        key="hybrid_rag_retrieve_answer",
    )
    compare_clicked = comparison_col.button(
        ui_text(language, "compare_naive_hybrid"),
        width="stretch",
        help=ui_text(language, "compare_help"),
        key="hybrid_rag_compare",
    )

    if query_clicked or compare_clicked:
        if not question.strip():
            st.warning(ui_text(language, "question_required"))
            return
        try:
            with st.spinner(ui_text(language, "retrieving")):
                if query_clicked:
                    answer_result = _with_service(
                        runtime,
                        settings,
                        retrieval_settings,
                        lambda service: run_async(
                            ask_question(
                                service,
                                question,
                                mode=mode,
                                options=runtime.options,
                                use_deepseek=runtime.use_deepseek,
                                deepseek_settings=deepseek_settings,
                            )
                        ),
                    )
                    st.session_state["hybrid_rag_answer"] = answer_result
                    st.session_state.pop("hybrid_rag_comparison", None)
                else:
                    comparison = _with_service(
                        runtime,
                        settings,
                        retrieval_settings,
                        lambda service: run_async(
                            compare_naive_and_hybrid(
                                service,
                                question,
                                options=runtime.options,
                                use_deepseek=runtime.use_deepseek,
                                deepseek_settings=deepseek_settings,
                            )
                        ),
                    )
                    st.session_state["hybrid_rag_comparison"] = comparison
                    st.session_state.pop("hybrid_rag_answer", None)
        except Exception as error:  # UI boundary: errors are actionable in the app, not silent.
            st.error(f"{ui_text(language, 'request_failed')}: {type(error).__name__}: {error}")

    if answer_result := st.session_state.get("hybrid_rag_answer"):
        _render_answer(st, answer_result, language=language)

    if comparison := st.session_state.get("hybrid_rag_comparison"):
        st.subheader(ui_text(language, "naive_vs_hybrid"))
        st.caption(ui_text(language, "comparison_caption"))
        naive, hybrid = comparison
        naive_column, hybrid_column = st.columns(2)
        with naive_column:
            _render_answer(
                st,
                naive,
                language=language,
                title=_MODE_LABELS[language][RetrievalMode.NAIVE],
            )
        with hybrid_column:
            _render_answer(
                st,
                hybrid,
                language=language,
                title=_MODE_LABELS[language][RetrievalMode.HYBRID],
            )


if __name__ == "__main__":
    main()
