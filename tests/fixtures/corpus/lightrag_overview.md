# LightRAG Notes

## Motivation

Naive retrieval-augmented generation usually retrieves isolated text chunks. This is useful for
local facts, but it can lose relationships that span documents or sections.

LightRAG-style systems extract entities and relations before retrieval. The graph provides an
additional route to evidence while the original chunks remain the source of truth.

## Retrieval modes

Local retrieval starts from concrete entities and expands their nearby relations. It is suitable
for questions about a named method, dataset, metric, or author.

Global retrieval starts from higher-level relation or topic signals. It is suitable for summary
and comparison questions that require evidence from several parts of a corpus.

Hybrid retrieval combines the local entity-led and global relation-led graph routes. Mix retrieval
adds the chunk route to those graph routes. The final context must retain source identifiers so
that an answer can cite its evidence.
