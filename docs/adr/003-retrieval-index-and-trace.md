# ADR 003: Three-way retrieval indexes and replayable evidence traces

## Status

Accepted — 2026-07-12; amended — 2026-07-13.

## Context

Stage three needs vector recall for chunks, entities, and relations without making a
vector database, an agent framework, or a provider SDK responsible for retrieval
semantics. The project also needs a locally runnable semantic default and a fast,
offline-testable adapter for CI.

## Decision

- Persist an `embedding_profiles` record and one JSON vector row per `chunk`,
  `entity`, or `relation` in SQLite. The profile identity includes the embedding
  provider/model/dimensions, vector-affecting encoding options, and text-schema,
  graph-independent corpus-content hash, and graph build run/snapshot. An atomic replacement activates only a complete index;
  a distinct graph run has a distinct profile identity rather than overwriting vectors.
- Build distinct embedding texts: contextualized source text for chunks, normalized
  name/type/aliases/description for entities, and named endpoint/predicate/
  description for relations.
- Use a project-owned `EmbeddingProvider` protocol. The default is the local
  FlagEmbedding `BAAI/bge-m3` adapter. `hash-token-v1` remains an explicit
  deterministic adapter for CI and historical profile compatibility.
- Implement chunk dense + local BM25 lexical ranking for `naive`, local/global
  graph expansion, per-route min-max normalization, weighted fusion,
  de-duplication, NetworkX path expansion, a post-fusion local FlagEmbedding
  cross-encoder rerank, and token-budget context clipping in project code. The naive trace
  retains raw, normalized, and weighted dense/BM25 contributions; the rerank
  trace retains its candidate pool, component scores, and final rank. Setting
  the reranker provider to `none` leaves the first-stage order unchanged.
- Persist every retrieval as an `rtr_` trace containing input, index identity,
  route candidates, fusion components, graph paths, final context and optional
  answer. Replay reads that stored result without re-embedding or re-ranking.
- When ingestion replaces a document's chunks, deactivate every profile that
  contains those source chunks before deletion. The stale vectors and historical
  traces remain auditable, but only a rebuilt `ready` profile can be queried.
- Query-time DeepSeek use is constrained to JSON keyword extraction and a JSON
  answer whose citations are validated against the retrieval-selected chunk IDs.

## Consequences

- The learning project supports only local FlagEmbedding and compatibility hash
  embedding adapters; adding another provider requires an explicit adapter and ADR.
- JSON vector storage is deliberately simple and explainable, not a large-corpus
  ANN solution. A future vector-store adapter can retain the same profile/vector
  contract after benchmark evidence justifies it.
- Retrieval remains reproducible at the trace level even when an external answer
  model is unavailable or changes over time.
