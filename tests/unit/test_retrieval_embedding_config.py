from __future__ import annotations

from hybrid_rag.config import RetrievalSettings
from hybrid_rag.ids import canonical_json_hash
from hybrid_rag.retrieval.models import IndexSemanticConfig


def test_retrieval_settings_default_to_local_bge_m3() -> None:
    settings = RetrievalSettings(_env_file=None)

    assert settings.embedding_provider == "flagembedding"
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_dimensions == 1024
    assert settings.embedding_batch_size == 12
    assert settings.embedding_max_length == 8192
    assert not settings.embedding_use_fp16


def test_embedding_options_participate_in_index_identity_without_changing_legacy_hashes() -> None:
    legacy = IndexSemanticConfig(
        provider="hash",
        model="hash-token-v1",
        dimensions=384,
    )
    bge = IndexSemanticConfig(
        provider="flagembedding",
        model="BAAI/bge-m3",
        dimensions=1024,
        provider_options={
            "max_length": 8192,
            "normalize_embeddings": True,
            "use_fp16": False,
        },
    )
    changed_precision = bge.model_copy(
        update={
            "provider_options": {
                "max_length": 8192,
                "normalize_embeddings": True,
                "use_fp16": True,
            }
        }
    )

    assert legacy.config_hash == canonical_json_hash(
        {
            "provider": "hash",
            "model": "hash-token-v1",
            "dimensions": 384,
            "text_schema_version": "1",
        }
    )
    assert bge.config_hash != changed_precision.config_hash
