# ADR 003: Three-way retrieval indexes and replayable evidence traces

## Status

Accepted — 2026-07-12.

## Context

Stage three needs vector recall for chunks, entities, and relations without making a
vector database, an agent framework, or a provider SDK responsible for retrieval
semantics. The project also needs an offline-testable default while the production
embedding model remains a benchmark decision.

## Decision

- Persist an `embedding_profiles` record and one JSON vector row per `chunk`,
  `entity`, or `relation` in SQLite. The profile identity includes the embedding
  provider/model/dimensions/text-schema, graph-independent corpus-content hash, and
  graph build run/snapshot. An atomic replacement activates only a complete index;
  a distinct graph run has a distinct profile identity rather than overwriting vectors.
- Build distinct embedding texts: contextualized source text for chunks, normalized
  name/type/aliases/description for entities, and named endpoint/predicate/
  description for relations.
- Use a project-owned `EmbeddingProvider` protocol. The default
  `hash-token-v1` adapter is deterministic and local, intended for CI and demo
  reproducibility; an OpenAI-compatible embedding adapter can be configured after
  a benchmark selects a real model.
- Implement cosine ranking, local/global graph expansion, per-route min-max
  normalization, weighted fusion, de-duplication, NetworkX path expansion, and
  token-budget context clipping in project code.
- Persist every retrieval as an `rtr_` trace containing input, index identity,
  route candidates, fusion components, graph paths, final context and optional
  answer. Replay reads that stored result without re-embedding or re-ranking.
- When ingestion replaces a document's chunks, deactivate every profile that
  contains those source chunks before deletion. The stale vectors and historical
  traces remain auditable, but only a rebuilt `ready` profile can be queried.
- Query-time DeepSeek use is constrained to JSON keyword extraction and a JSON
  answer whose citations are validated against the retrieval-selected chunk IDs.

## Consequences

- A real embedding provider can replace the default adapter without changing the
  index schema or retrieval algorithms.
- JSON vector storage is deliberately simple and explainable, not a large-corpus
  ANN solution. A future vector-store adapter can retain the same profile/vector
  contract after benchmark evidence justifies it.
- Retrieval remains reproducible at the trace level even when an external answer
  model is unavailable or changes over time.
