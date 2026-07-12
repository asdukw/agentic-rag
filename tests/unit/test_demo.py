from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

import hybrid_rag.demo as demo
from hybrid_rag.config import DeepSeekSettings, RetrievalSettings, Settings, sqlite_url
from hybrid_rag.retrieval.embedding import HashEmbeddingProvider, OpenAICompatibleEmbeddingProvider
from hybrid_rag.retrieval.models import (
    RerankComponentTrace,
    RerankTrace,
    RerankTraceHit,
    RetrievalMode,
)
from hybrid_rag.retrieval.query import DeepSeekQueryClient, DeterministicQueryClient
from hybrid_rag.retrieval.service import RetrievalOptions


@dataclass
class RecordingClient:
    closed: bool = False

    async def close(self) -> None:
        self.closed = True


@dataclass
class RecordingService:
    responses: dict[RetrievalMode, object]
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    async def ask(self, question: str, **kwargs: Any) -> object:
        self.calls.append((question, kwargs))
        return self.responses[kwargs["mode"]]


@dataclass
class FailingService:
    async def ask(self, question: str, **kwargs: Any) -> object:
        raise LookupError("no matching index")


def test_database_url_from_input_preserves_url_and_normalizes_paths(tmp_path: Path) -> None:
    configured = "sqlite:///configured.db"
    settings = Settings(database_url=configured)

    assert demo.database_url_from_input("   ", settings) == configured
    assert demo.database_url_from_input("postgresql://example.test/rag", settings) == (
        "postgresql://example.test/rag"
    )

    path = tmp_path / "nested" / "demo.db"
    assert demo.database_url_from_input(str(path), settings) == sqlite_url(path)
    assert path.parent.is_dir()


def test_demo_translation_catalog_is_complete_and_formats_chinese_values() -> None:
    assert set(demo._TEXT["en"]) == set(demo._TEXT["zh"])
    assert demo.ui_text("zh", "title") == "Hybrid RAG · 证据优先检索演示"
    assert demo.ui_text("zh", "index_ready", chunks=3, entities=2, relations=1).startswith(
        "索引就绪"
    )
    assert demo._MODE_LABELS["zh"][RetrievalMode.HYBRID] == "Hybrid — 三路固定融合"
    assert demo._page_label(None, None, language="zh") == "页码未知"
    assert demo._page_label(2, 4, language="zh") == "第 2-4 页"


def test_demo_defaults_to_chinese_and_preserves_widget_state_on_language_switch() -> None:
    assert demo.__file__ is not None
    app = AppTest.from_file(Path(demo.__file__)).run(timeout=15)

    assert not app.exception
    assert [item.value for item in app.title] == ["Hybrid RAG · 证据优先检索演示"]
    assert [item.label for item in app.button] == [
        "检索并回答",
        "对比 naive 与 hybrid",
        "构建索引",
        "重放",
    ]
    assert [item.label for item in app.toggle] == ["启用词法重排序器"]

    app.text_area[0].set_value("保留的问题")
    app.run(timeout=15)
    app.sidebar.selectbox[0].set_value("English")
    app.run(timeout=15)

    assert not app.exception
    assert [item.label for item in app.button] == [
        "Retrieve and answer",
        "Compare naive vs hybrid",
        "Build index",
        "Replay",
    ]
    assert app.text_area[0].value == "保留的问题"


def test_demo_rerank_rows_keep_pre_and_post_ranking_visible() -> None:
    rerank = RerankTrace(
        provider="lexical",
        model="lexical-coverage-v1",
        version="lexical-reranker-v1",
        candidate_limit=4,
        hits=(
            RerankTraceHit(
                object_id="chk-1",
                pre_rerank_rank=2,
                pre_rerank_score=0.2,
                score=0.9,
                final_rank=1,
                components={
                    "coverage": RerankComponentTrace(
                        raw_score=1.0,
                        normalized_score=1.0,
                        weight=1.0,
                        weighted_score=1.0,
                    )
                },
            ),
        ),
    )

    rows = demo.rerank_rows(rerank, language="zh")

    assert rows == [
        {
            "对象": "chk-1",
            "重排序前排名": 2,
            "重排序前分数": 0.2,
            "重排序分数": 0.9,
            "排名": 1,
            "分数分量": "coverage: raw=1.0000, norm=1.0000, weighted=1.0000",
        }
    ]


def test_demo_embedding_provider_selection_is_explicit_and_offline() -> None:
    runtime = _runtime(embedding_provider="hash", embedding_model="demo-hash", dimensions=96)

    provider = demo.create_embedding_provider(runtime, RetrievalSettings())

    assert isinstance(provider, HashEmbeddingProvider)
    assert provider.model == "demo-hash"
    assert provider.dimensions == 96

    with pytest.raises(ValueError, match="EMBEDDING_BASE_URL"):
        demo.create_embedding_provider(
            _runtime(embedding_provider="openai-compatible"),
            RetrievalSettings(),
        )

    external = demo.create_embedding_provider(
        _runtime(
            embedding_provider="openai-compatible",
            embedding_model="embedding-test",
            dimensions=64,
        ),
        RetrievalSettings(
            embedding_base_url="https://embeddings.example.test/v1",
            embedding_api_key="test-key",
        ),
    )
    assert isinstance(external, OpenAICompatibleEmbeddingProvider)
    assert external.base_url == "https://embeddings.example.test/v1"
    assert external.model == "embedding-test"
    assert external.dimensions == 64

    with pytest.raises(ValueError, match="embedding provider"):
        demo.create_embedding_provider(
            _runtime(embedding_provider="unsupported"),
            RetrievalSettings(),
        )


def test_demo_query_client_requires_credentials_only_for_explicit_deepseek_use() -> None:
    offline = demo.create_query_client(False, DeepSeekSettings())
    assert isinstance(offline, DeterministicQueryClient)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        demo.create_query_client(True, DeepSeekSettings(api_key=None))

    online = demo.create_query_client(
        True,
        DeepSeekSettings(
            api_key="test-key",
            query_model="keyword-test",
            answer_model="answer-test",
            answer_max_output_tokens=256,
        ),
    )
    assert isinstance(online, DeepSeekQueryClient)
    assert online.keyword_model == "keyword-test"
    assert online.answer_model == "answer-test"
    assert online.max_output_tokens == 512


def test_create_service_upgrades_the_selected_sqlite_database(tmp_path: Path) -> None:
    database_url = sqlite_url(tmp_path / "demo.db")
    runtime = _runtime(database_url=database_url, dimensions=64)

    database, service = demo.create_service(
        runtime,
        Settings(database_url=database_url),
        RetrievalSettings(),
    )
    try:
        assert service.database is database
        assert isinstance(service.embedding_provider, HashEmbeddingProvider)
        assert service.embedding_provider.dimensions == 64
        assert service.token_counter.name == "cl100k_base"
        assert (tmp_path / "demo.db").is_file()
    finally:
        database.dispose()


def test_ask_question_uses_the_selected_mode_and_always_closes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = RecordingClient()
    service = RecordingService(responses={RetrievalMode.LOCAL: "local-answer"})
    options = RetrievalOptions(top_k=3, context_token_budget=512)
    monkeypatch.setattr(demo, "create_query_client", lambda *_: client)

    result = asyncio.run(
        demo.ask_question(
            service,  # type: ignore[arg-type]
            "Which entity matters?",
            mode=RetrievalMode.LOCAL,
            options=options,
            use_deepseek=False,
            deepseek_settings=DeepSeekSettings(),
        )
    )

    assert result == "local-answer"
    assert service.calls == [
        (
            "Which entity matters?",
            {
                "query_client": client,
                "mode": RetrievalMode.LOCAL,
                "options": options,
            },
        )
    ]
    assert client.closed


def test_compare_naive_and_hybrid_cannot_delegate_mode_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = RecordingClient()
    service = RecordingService(
        responses={RetrievalMode.NAIVE: "naive-answer", RetrievalMode.HYBRID: "hybrid-answer"}
    )
    options = RetrievalOptions(top_k=2, context_token_budget=512)
    monkeypatch.setattr(demo, "create_query_client", lambda *_: client)

    compared = asyncio.run(
        demo.compare_naive_and_hybrid(
            service,  # type: ignore[arg-type]
            "Compare retrieval evidence.",
            options=options,
            use_deepseek=True,
            deepseek_settings=DeepSeekSettings(api_key="unused-by-test"),
        )
    )

    assert compared == ("naive-answer", "hybrid-answer")
    assert [call[1]["mode"] for call in service.calls] == [
        RetrievalMode.NAIVE,
        RetrievalMode.HYBRID,
    ]
    assert all(call[1]["query_client"] is client for call in service.calls)
    assert all(call[1]["options"] is options for call in service.calls)
    assert client.closed


def test_ask_question_closes_client_when_retrieval_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RecordingClient()
    monkeypatch.setattr(demo, "create_query_client", lambda *_: client)

    with pytest.raises(LookupError, match="no matching index"):
        asyncio.run(
            demo.ask_question(
                FailingService(),  # type: ignore[arg-type]
                "Question",
                mode=RetrievalMode.HYBRID,
                options=RetrievalOptions(context_token_budget=512),
                use_deepseek=False,
                deepseek_settings=DeepSeekSettings(),
            )
        )

    assert client.closed


def test_run_async_returns_values_and_rejects_nested_event_loops() -> None:
    async def value() -> str:
        return "done"

    assert demo.run_async(value()) == "done"

    async def nested() -> None:
        coroutine = value()
        with pytest.raises(RuntimeError, match="active event loop"):
            demo.run_async(coroutine)
        coroutine.close()

    asyncio.run(nested())


def _runtime(
    *,
    database_url: str = "sqlite:///demo.db",
    embedding_provider: str = "hash",
    embedding_model: str = "hash-token-v1",
    dimensions: int = 384,
) -> demo.DemoRuntime:
    return demo.DemoRuntime(
        database_url=database_url,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimensions=dimensions,
        options=RetrievalOptions(context_token_budget=512),
        use_deepseek=False,
    )
