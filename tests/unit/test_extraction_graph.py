from __future__ import annotations

import json

from hybrid_rag.extraction.graph import (build_networkx_graph, node_link_json,
                                         summarize_graph)
from hybrid_rag.extraction.normalization import (merge_relations,
                                                 normalize_entities)
from hybrid_rag.extraction.schemas import (EntityMention, EntityType,
                                           EvidenceSpan, RelationMention)


def _evidence(chunk_id: str, quote: str) -> tuple[EvidenceSpan, ...]:
    return (
        EvidenceSpan(
            source_chunk_id=chunk_id,
            quote=quote,
            char_start=0,
            char_end=len(quote),
        ),
    )


def _entity(
    mention_id: str,
    chunk_id: str,
    name: str,
    entity_type: EntityType,
    *,
    aliases: tuple[str, ...] = (),
) -> EntityMention:
    return EntityMention(
        id=mention_id,
        name=name,
        entity_type=entity_type,
        description=f"Description of {name}.",
        aliases=aliases,
        source_chunk_ids=(chunk_id,),
        evidence=_evidence(chunk_id, name),
    )


def _relation(
    relation_id: str,
    chunk_id: str,
    source_id: str,
    target_id: str,
    predicate: str,
) -> RelationMention:
    quote = f"{source_id} {predicate} {target_id}"
    return RelationMention(
        id=relation_id,
        source_mention_id=source_id,
        target_mention_id=target_id,
        predicate=predicate,
        description=quote,
        source_chunk_ids=(chunk_id,),
        evidence=_evidence(chunk_id, quote),
    )


def test_alias_union_is_deterministic_and_type_conservative() -> None:
    mentions = [
        _entity(
            "emn_1",
            "chk_1",
            "Retrieval-Augmented Generation",
            EntityType.METHOD,
            aliases=("RAG",),
        ),
        _entity("emn_2", "chk_2", "rag", EntityType.METHOD),
        _entity("emn_3", "chk_3", "RAG", EntityType.DATASET),
    ]

    first = normalize_entities(mentions)
    second = normalize_entities(reversed(mentions))

    assert first == second
    assert len(first.entities) == 2
    assert first.mention_to_entity["emn_1"] == first.mention_to_entity["emn_2"]
    assert first.mention_to_entity["emn_1"] != first.mention_to_entity["emn_3"]
    merged = next(
        entity for entity in first.entities if entity.id == first.mention_to_entity["emn_1"]
    )
    assert merged.source_chunk_ids == ("chk_1", "chk_2")
    assert {item.quote for item in merged.evidence} == {
        "Retrieval-Augmented Generation",
        "rag",
    }


def test_exact_directed_relation_merge_and_graph_export() -> None:
    entities = [
        _entity("emn_a", "chk_1", "Method A", EntityType.METHOD),
        _entity("emn_b", "chk_1", "Dataset B", EntityType.DATASET),
        _entity("emn_c", "chk_2", "Isolated C", EntityType.CONCEPT),
    ]
    normalized = normalize_entities(entities)
    mentions = [
        _relation("rmn_1", "chk_1", "emn_a", "emn_b", "USES"),
        _relation("rmn_2", "chk_2", "emn_a", "emn_b", "USES"),
        _relation("rmn_3", "chk_2", "emn_b", "emn_a", "USES"),
    ]

    merged = merge_relations(mentions, normalized.mention_to_entity)
    graph = build_networkx_graph(normalized.entities, merged.relations)
    stats = summarize_graph(graph, top_k=2)
    payload = json.loads(node_link_json(graph))

    assert len(merged.relations) == 2
    forward = next(
        relation
        for relation in merged.relations
        if relation.source_entity_id == normalized.mention_to_entity["emn_a"]
    )
    assert forward.source_chunk_ids == ("chk_1", "chk_2")
    assert stats.nodes == 3
    assert stats.edges == 2
    assert stats.weak_components == 2
    assert stats.weak_component_sizes == (2, 1)
    assert stats.isolate_ids == (normalized.mention_to_entity["emn_c"],)
    assert len(stats.top_entities) == 2
    assert payload["directed"] is True
    assert payload["multigraph"] is True
    assert [node["id"] for node in payload["nodes"]] == sorted(graph.nodes)
    assert len(payload["edges"]) == 2
