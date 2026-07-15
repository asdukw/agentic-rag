from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict

from hybrid_rag.extraction.schemas import CanonicalEntity, CanonicalRelation


class TopEntity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    entity_id: str
    canonical_name: str
    degree: int
    in_degree: int
    out_degree: int


class GraphStats(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    nodes: int
    edges: int
    weak_components: int
    weak_component_sizes: tuple[int, ...]
    isolate_ids: tuple[str, ...]
    top_entities: tuple[TopEntity, ...]


def build_networkx_graph(
    entities: Iterable[CanonicalEntity],
    relations: Iterable[CanonicalRelation],
) -> nx.MultiDiGraph:
    """Project persisted domain records into a deterministic directed multigraph."""

    ordered_entities = sorted(entities, key=lambda item: item.id)
    ordered_relations = sorted(relations, key=lambda item: item.id)
    _require_unique((item.id for item in ordered_entities), "entity")
    _require_unique((item.id for item in ordered_relations), "relation")

    graph = nx.MultiDiGraph(schema_version="1")
    entity_ids = {entity.id for entity in ordered_entities}
    for entity in ordered_entities:
        graph.add_node(
            entity.id,
            canonical_name=entity.canonical_name,
            entity_type=entity.entity_type,
            description=entity.description,
            aliases=list(entity.aliases),
            source_chunk_ids=list(entity.source_chunk_ids),
            evidence=[item.model_dump(mode="json") for item in entity.evidence],
        )

    for relation in ordered_relations:
        missing = {
            value
            for value in (relation.source_entity_id, relation.target_entity_id)
            if value not in entity_ids
        }
        if missing:
            raise ValueError(
                f"relation {relation.id} references missing canonical entities: {sorted(missing)}"
            )
        graph.add_edge(
            relation.source_entity_id,
            relation.target_entity_id,
            key=relation.id,
            id=relation.id,
            predicate=relation.predicate,
            description=relation.description,
            source_chunk_ids=list(relation.source_chunk_ids),
            evidence=[item.model_dump(mode="json") for item in relation.evidence],
        )
    return graph


def summarize_graph(graph: nx.MultiDiGraph, *, top_k: int = 10) -> GraphStats:
    if top_k < 0:
        raise ValueError("top_k must not be negative")
    if not graph.is_directed() or not graph.is_multigraph():
        raise TypeError("knowledge graph must be a directed multigraph")

    components = [tuple(sorted(component)) for component in nx.weakly_connected_components(graph)]
    components.sort(key=lambda component: (-len(component), component))
    isolates = tuple(sorted(nx.isolates(graph)))
    ranked = []
    for entity_id in sorted(graph.nodes):
        in_degree = int(graph.in_degree(entity_id))
        out_degree = int(graph.out_degree(entity_id))
        ranked.append(
            TopEntity(
                entity_id=entity_id,
                canonical_name=str(graph.nodes[entity_id].get("canonical_name", entity_id)),
                degree=in_degree + out_degree,
                in_degree=in_degree,
                out_degree=out_degree,
            )
        )
    ranked.sort(
        key=lambda item: (
            -item.degree,
            item.canonical_name.casefold(),
            item.entity_id,
        )
    )
    return GraphStats(
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        weak_components=len(components),
        weak_component_sizes=tuple(len(component) for component in components),
        isolate_ids=isolates,
        top_entities=tuple(ranked[:top_k]),
    )


def node_link_payload(graph: nx.MultiDiGraph) -> dict[str, Any]:
    """Return stable NetworkX node-link data using the explicit ``edges`` key."""

    if not graph.is_directed() or not graph.is_multigraph():
        raise TypeError("knowledge graph must be a directed multigraph")
    nodes = [
        {"id": node_id, **_sorted_attributes(attributes)}
        for node_id, attributes in sorted(graph.nodes(data=True), key=lambda item: item[0])
    ]
    edges = [
        {
            "source": source,
            "target": target,
            "key": key,
            **_sorted_attributes(attributes),
        }
        for source, target, key, attributes in sorted(
            graph.edges(keys=True, data=True),
            key=lambda item: (item[0], item[1], item[2]),
        )
    ]
    return {
        "directed": True,
        "multigraph": True,
        "graph": _sorted_attributes(graph.graph),
        "nodes": nodes,
        "edges": edges,
    }


def node_link_json(graph: nx.MultiDiGraph, *, indent: int | None = None) -> str:
    return json.dumps(
        node_link_payload(graph),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
    )


def _sorted_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    return {key: attributes[key] for key in sorted(attributes)}


def _require_unique(values: Iterable[str], label: str) -> None:
    ordered = list(values)
    if len(ordered) != len(set(ordered)):
        raise ValueError(f"duplicate {label} IDs are not allowed")
