"""Provider-independent extraction and knowledge-graph domain primitives."""

from hybrid_rag.extraction.client import (
    CompletionResult,
    DeepSeekClient,
    ExtractionClient,
    ProviderError,
    RetryableProviderError,
    TerminalProviderError,
)
from hybrid_rag.extraction.graph import (
    GraphStats,
    TopEntity,
    build_networkx_graph,
    node_link_json,
    node_link_payload,
    summarize_graph,
)
from hybrid_rag.extraction.normalization import (
    merge_relations,
    normalize_entities,
    normalize_entity_alias,
    normalize_predicate,
)
from hybrid_rag.extraction.prompts import (
    build_extraction_messages,
    build_gleaning_messages,
    build_repair_messages,
)
from hybrid_rag.extraction.schemas import (
    CanonicalEntity,
    CanonicalRelation,
    EntityCandidate,
    EntityMention,
    EntityNormalizationResult,
    EntityType,
    EvidenceSpan,
    ExtractionConfig,
    GraphConfig,
    RelationCandidate,
    RelationMention,
    RelationMergeResult,
    ValidatedChunkExtraction,
)
from hybrid_rag.extraction.validation import (
    ExtractionValidationError,
    ValidationFailureKind,
    ValidationIssue,
    validate_completion,
)

__all__ = [
    "CanonicalEntity",
    "CanonicalRelation",
    "CompletionResult",
    "DeepSeekClient",
    "EntityCandidate",
    "EntityMention",
    "EntityNormalizationResult",
    "EntityType",
    "EvidenceSpan",
    "ExtractionClient",
    "ExtractionConfig",
    "ExtractionValidationError",
    "GraphConfig",
    "GraphStats",
    "ProviderError",
    "RelationCandidate",
    "RelationMention",
    "RelationMergeResult",
    "RetryableProviderError",
    "TerminalProviderError",
    "TopEntity",
    "ValidatedChunkExtraction",
    "ValidationFailureKind",
    "ValidationIssue",
    "build_extraction_messages",
    "build_gleaning_messages",
    "build_networkx_graph",
    "build_repair_messages",
    "merge_relations",
    "node_link_json",
    "node_link_payload",
    "normalize_entities",
    "normalize_entity_alias",
    "normalize_predicate",
    "summarize_graph",
    "validate_completion",
]
