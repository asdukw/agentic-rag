"""Streamlit evidence-first demo for the Hybrid RAG retrieval pipeline.

Run from the repository root after ingesting documents, building a graph, and
building an index::

    uv run streamlit run src/hybrid_rag/demo.py

The UI deliberately invokes only fixed retrieval modes.  A model is optional
and can perform only the two bounded operations exposed by ``QueryClient``:
keyword extraction and an answer over already selected evidence.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from hybrid_rag.config import DeepSeekSettings, RetrievalSettings, Settings, sqlite_url
from hybrid_rag.ingest.tokenizer import TiktokenCounter
from hybrid_rag.retrieval.embedding import (
    HashEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from hybrid_rag.retrieval.models import CandidateHit, GraphPath, RetrievalMode, RetrievalResult
from hybrid_rag.retrieval.query import (
    DeepSeekQueryClient,
    DeterministicQueryClient,
    QueryClient,
)
from hybrid_rag.retrieval.service import AnswerResult, RetrievalOptions, RetrievalService
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.migrations import upgrade_database

_T = TypeVar("_T")
_MODE_LABELS = {
    RetrievalMode.NAIVE: "Naive — chunk vector recall",
    RetrievalMode.LOCAL: "Local — entity-led graph recall",
    RetrievalMode.GLOBAL: "Global — relation-led graph recall",
    RetrievalMode.HYBRID: "Hybrid — fixed three-route fusion",
}


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

    if runtime.embedding_provider == "hash":
        return HashEmbeddingProvider(
            dimensions=runtime.embedding_dimensions,
            model=runtime.embedding_model,
        )
    if runtime.embedding_provider == "openai-compatible":
        if not settings.embedding_base_url:
            raise ValueError(
                "HYBRID_RAG_RETRIEVAL_EMBEDDING_BASE_URL is required for "
                "openai-compatible embeddings"
            )
        api_key = (
            settings.embedding_api_key.get_secret_value().strip()
            if settings.embedding_api_key
            else ""
        )
        return OpenAICompatibleEmbeddingProvider(
            api_key=api_key or None,
            base_url=settings.embedding_base_url,
            model=runtime.embedding_model,
            dimensions=runtime.embedding_dimensions,
        )
    raise ValueError("embedding provider must be 'hash' or 'openai-compatible'")


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


def route_rows(result: RetrievalResult) -> list[dict[str, Any]]:
    """Compact, JSON-safe route summaries for a Streamlit table."""

    rows: list[dict[str, Any]] = []
    for route_name, route in result.trace.routes.items():
        top = route.hits[0] if route.hits else None
        rows.append(
            {
                "route": route_name,
                "candidates": route.candidate_count,
                "top object": top.object_id if top else "—",
                "top score": round(top.score, 4) if top else None,
            }
        )
    return rows


def hit_rows(hits: tuple[CandidateHit, ...]) -> list[dict[str, Any]]:
    """Flatten fused score provenance without hiding per-route contributions."""

    return [
        {
            "object": hit.object_id,
            "kind": hit.kind,
            "score": round(hit.score, 4),
            "rank": hit.rank,
            "route scores": _format_scores(hit.route_scores),
            "source chunks": ", ".join(hit.source_chunk_ids),
        }
        for hit in hits
    ]


def graph_path_rows(paths: tuple[GraphPath, ...]) -> list[dict[str, Any]]:
    """Present retained NetworkX paths exactly as the trace recorded them."""

    return [
        {
            "nodes": " → ".join(path.node_ids),
            "relations": " → ".join(path.relation_ids),
            "supporting chunks": ", ".join(path.source_chunk_ids),
            "score": round(path.score, 4),
        }
        for path in paths
    ]


def _format_scores(scores: dict[str, float]) -> str:
    return ", ".join(f"{route}={score:.4f}" for route, score in sorted(scores.items()))


def _render_answer(st: Any, result: AnswerResult, *, title: str = "Answer") -> None:
    st.subheader(title)
    if result.answer.insufficient_evidence:
        st.warning(result.answer.answer)
    else:
        st.write(result.answer.answer)
        st.caption("Citations: " + ", ".join(result.answer.citations))
    _render_retrieval(st, result.retrieval)


def _render_retrieval(st: Any, result: RetrievalResult) -> None:
    """Expose citations, vector routes, graph support, and replayable decisions."""

    st.caption(
        f"Mode: `{result.mode.value}` · profile: `{result.profile_id}` · "
        f"trace: `{result.trace_id or 'not persisted'}`"
    )
    metrics = st.columns(4)
    metrics[0].metric("Context tokens", result.context_tokens)
    metrics[1].metric("Context chunks", len(result.context_items))
    metrics[2].metric("Fused hits", len(result.hits))
    metrics[3].metric("Graph paths", len(result.graph_paths))

    st.markdown("#### Citation context")
    if not result.context_items:
        st.info("No chunk fit the selected context budget.")
    for item in result.context_items:
        section = " / ".join(item.section_path) if item.section_path else "Unsectioned"
        pages = _page_label(item.page_start, item.page_end)
        with st.expander(f"[{item.citation_id}] {item.document_title} — {section}"):
            st.caption(
                f"document `{item.document_id}` · {pages} · {item.token_count} tokens · "
                f"score {item.score:.4f} · {_format_scores(item.route_scores)}"
            )
            if item.source_entity_ids or item.source_relation_ids:
                st.caption(
                    "graph support: "
                    f"entities={', '.join(item.source_entity_ids) or '—'}; "
                    f"relations={', '.join(item.source_relation_ids) or '—'}"
                )
            st.write(item.text)

    with st.expander("Route scores and fused candidates", expanded=False):
        rows = route_rows(result)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("This trace did not retain route candidates.")
        fused = hit_rows(result.hits)
        if fused:
            st.dataframe(fused, use_container_width=True, hide_index=True)

    with st.expander("Graph paths", expanded=False):
        paths = graph_path_rows(result.graph_paths)
        if paths:
            st.dataframe(paths, use_container_width=True, hide_index=True)
        else:
            st.info("No graph path was needed for this result.")

    with st.expander("Final context and replayable trace", expanded=False):
        st.code(result.context, language="text")
        st.json(result.trace.model_dump(mode="json"))


def _page_label(page_start: int | None, page_end: int | None) -> str:
    if page_start is None:
        return "page unknown"
    if page_end is None or page_end == page_start:
        return f"page {page_start}"
    return f"pages {page_start}-{page_end}"


def _runtime_from_sidebar(
    st: Any,
) -> tuple[DemoRuntime, Settings, RetrievalSettings, DeepSeekSettings]:
    settings = Settings()
    retrieval = RetrievalSettings()
    deepseek = DeepSeekSettings()

    st.sidebar.header("Execution settings")
    database_input = st.sidebar.text_input(
        "SQLite database path or URL",
        value=settings.database_url,
        help="A path is converted to an absolute SQLite URL. Existing CLI defaults also work.",
    )
    available_providers = ("hash", "openai-compatible")
    default_provider = (
        retrieval.embedding_provider
        if retrieval.embedding_provider in available_providers
        else "hash"
    )
    provider = st.sidebar.selectbox(
        "Embedding provider",
        options=available_providers,
        index=available_providers.index(default_provider),
    )
    model = st.sidebar.text_input("Embedding model", value=retrieval.embedding_model)
    dimensions = int(
        st.sidebar.number_input(
            "Embedding dimensions",
            min_value=32,
            max_value=4096,
            value=retrieval.embedding_dimensions,
            step=1,
        )
    )
    st.sidebar.caption(
        "`hash` is deterministic and offline. `openai-compatible` reads its base URL and key "
        "from HYBRID_RAG_RETRIEVAL_EMBEDDING_* settings."
    )

    st.sidebar.subheader("Retrieval budget")
    top_k = int(
        st.sidebar.number_input("Top K", min_value=1, max_value=100, value=retrieval.top_k)
    )
    context_budget = int(
        st.sidebar.number_input(
            "Context token budget",
            min_value=128,
            max_value=20_000,
            value=retrieval.context_token_budget,
            step=128,
        )
    )
    graph_hops = int(
        st.sidebar.slider(
            "Maximum graph hops",
            min_value=1,
            max_value=4,
            value=retrieval.graph_max_hops,
        )
    )
    use_deepseek = st.sidebar.checkbox(
        "Use DeepSeek for keywords + answer",
        value=False,
        help="When off, the demo uses deterministic offline keyword and answer functions.",
    )
    if use_deepseek:
        st.sidebar.warning(
            "DeepSeek is limited to keyword extraction and an answer over the displayed citation "
            "allowlist; it cannot choose retrieval tools or sources."
        )

    options = RetrievalOptions(
        top_k=top_k,
        candidate_multiplier=retrieval.candidate_multiplier,
        context_token_budget=context_budget,
        graph_max_hops=graph_hops,
        naive_weight=retrieval.naive_weight,
        local_weight=retrieval.local_weight,
        global_weight=retrieval.global_weight,
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
    st.title("Hybrid RAG · evidence-first retrieval demo")
    st.caption(
        "Each request uses one explicitly selected retriever. Hybrid is a deterministic fusion of "
        "naive, local, and global routes; no agent may select tools or hidden sources."
    )

    runtime, settings, retrieval_settings, deepseek_settings = _runtime_from_sidebar(st)
    build_col, replay_col = st.sidebar.columns(2)
    if build_col.button("Build index", use_container_width=True):
        try:
            with st.spinner("Embedding chunk, entity, and relation index texts..."):
                report = _with_service(
                    runtime,
                    settings,
                    retrieval_settings,
                    lambda service: service.build_index(),
                )
            st.sidebar.success(
                f"Ready: {report.chunks} chunks, {report.entities} entities, "
                f"{report.relations} relations"
            )
            st.sidebar.caption(f"profile `{report.profile_id}`")
        except Exception as error:  # UI boundary: preserve detailed domain error text.
            st.sidebar.error(f"Index build failed: {type(error).__name__}: {error}")

    replay_trace_id = replay_col.text_input(
        "Replay trace",
        placeholder="rtr_…",
        label_visibility="collapsed",
    )
    if replay_col.button("Replay", use_container_width=True):
        if not replay_trace_id.strip():
            st.sidebar.warning("Enter an rtr_ trace ID to replay.")
        else:
            try:
                replayed = _with_service(
                    runtime,
                    settings,
                    retrieval_settings,
                    lambda service: service.replay_answer(replay_trace_id.strip())
                    or service.replay(replay_trace_id.strip()),
                )
                st.session_state["hybrid_rag_replay"] = replayed
            except Exception as error:  # UI boundary: preserve detailed domain error text.
                st.sidebar.error(f"Replay failed: {type(error).__name__}: {error}")

    if replayed := st.session_state.get("hybrid_rag_replay"):
        st.subheader("Replayed trace")
        if isinstance(replayed, AnswerResult):
            _render_answer(st, replayed, title="Stored answer")
        else:
            _render_retrieval(st, replayed)

    mode = st.radio(
        "Retriever mode",
        options=tuple(RetrievalMode),
        format_func=lambda value: _MODE_LABELS[value],
        horizontal=True,
        index=tuple(RetrievalMode).index(RetrievalMode.HYBRID),
    )
    question = st.text_area(
        "Question",
        placeholder="How does LightRAG use entities?",
        height=96,
    )
    query_col, comparison_col = st.columns(2)
    query_clicked = query_col.button(
        "Retrieve and answer",
        type="primary",
        use_container_width=True,
    )
    compare_clicked = comparison_col.button(
        "Compare naive vs hybrid",
        use_container_width=True,
        help=(
            "Runs exactly the naive and hybrid modes, then shows their evidence and answers "
            "side by side."
        ),
    )

    if query_clicked or compare_clicked:
        if not question.strip():
            st.warning("Enter a question first.")
            return
        try:
            with st.spinner("Retrieving only from the selected index and graph snapshot..."):
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
            st.error(f"Request failed: {type(error).__name__}: {error}")

    if answer_result := st.session_state.get("hybrid_rag_answer"):
        _render_answer(st, answer_result)

    if comparison := st.session_state.get("hybrid_rag_comparison"):
        st.subheader("Naive vs hybrid")
        st.caption(
            "Both columns were run as named, fixed modes. The answer client received only each "
            "column's displayed citation context."
        )
        naive, hybrid = comparison
        naive_column, hybrid_column = st.columns(2)
        with naive_column:
            _render_answer(st, naive, title="Naive")
        with hybrid_column:
            _render_answer(st, hybrid, title="Hybrid")


if __name__ == "__main__":
    main()
