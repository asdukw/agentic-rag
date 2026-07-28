"""Immutable, profile-scoped graph snapshots for repeated agent expansion."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Literal

from hybrid_rag.storage.retrieval_repository import IndexItem, LoadedIndex


@dataclass(frozen=True, slots=True)
class GraphNeighborRef:
    """One directed view of a relation incident to an entity."""

    neighbor_entity_id: str
    relation: IndexItem
    direction: Literal["incoming", "outgoing"]


@dataclass(frozen=True, slots=True)
class ProfileGraphSnapshot:
    """Read-only lookup structures derived from one immutable index profile."""

    profile_id: str
    chunks_by_id: Mapping[str, IndexItem]
    entities_by_id: Mapping[str, IndexItem]
    relations_by_id: Mapping[str, IndexItem]
    adjacency: Mapping[str, tuple[GraphNeighborRef, ...]]
    incident_relation_ids: Mapping[str, frozenset[str]]

    @classmethod
    def from_index(cls, index: LoadedIndex) -> ProfileGraphSnapshot:
        adjacency: dict[str, list[GraphNeighborRef]] = defaultdict(list)
        incident_relation_ids: dict[str, set[str]] = defaultdict(set)
        for relation in index.relations:
            source = str(relation.metadata["source_entity_id"])
            target = str(relation.metadata["target_entity_id"])
            adjacency[source].append(
                GraphNeighborRef(
                    neighbor_entity_id=target,
                    relation=relation,
                    direction="outgoing",
                )
            )
            adjacency[target].append(
                GraphNeighborRef(
                    neighbor_entity_id=source,
                    relation=relation,
                    direction="incoming",
                )
            )
            incident_relation_ids[source].add(relation.object_id)
            incident_relation_ids[target].add(relation.object_id)

        stable_adjacency = {
            entity_id: tuple(
                sorted(
                    neighbors,
                    key=lambda neighbor: (
                        neighbor.relation.object_id,
                        neighbor.neighbor_entity_id,
                        neighbor.direction,
                    ),
                )
            )
            for entity_id, neighbors in adjacency.items()
        }
        return cls(
            profile_id=index.profile.id,
            chunks_by_id=MappingProxyType({item.object_id: item for item in index.chunks}),
            entities_by_id=MappingProxyType({item.object_id: item for item in index.entities}),
            relations_by_id=MappingProxyType({item.object_id: item for item in index.relations}),
            adjacency=MappingProxyType(stable_adjacency),
            incident_relation_ids=MappingProxyType(
                {
                    entity_id: frozenset(relation_ids)
                    for entity_id, relation_ids in incident_relation_ids.items()
                }
            ),
        )


class ProfileGraphSnapshotCache:
    """Small thread-safe LRU keyed by database identity and immutable profile ID."""

    def __init__(self, *, max_profiles: int = 8) -> None:
        if max_profiles < 1:
            raise ValueError("max_profiles must be positive")
        self.max_profiles = max_profiles
        self._lock = RLock()
        self._snapshots: OrderedDict[tuple[str, str], ProfileGraphSnapshot] = OrderedDict()

    def get_or_load(
        self,
        *,
        database_identity: str,
        profile_id: str,
        loader: Callable[[], LoadedIndex],
    ) -> ProfileGraphSnapshot:
        key = (database_identity, profile_id)
        with self._lock:
            cached = self._snapshots.get(key)
            if cached is not None:
                self._snapshots.move_to_end(key)
                return cached

        loaded = ProfileGraphSnapshot.from_index(loader())
        if loaded.profile_id != profile_id:
            raise ValueError(
                "loaded graph snapshot profile does not match the requested profile "
                f"({loaded.profile_id} != {profile_id})"
            )

        with self._lock:
            cached = self._snapshots.get(key)
            if cached is not None:
                self._snapshots.move_to_end(key)
                return cached
            self._snapshots[key] = loaded
            self._snapshots.move_to_end(key)
            while len(self._snapshots) > self.max_profiles:
                self._snapshots.popitem(last=False)
            return loaded


DEFAULT_PROFILE_GRAPH_CACHE = ProfileGraphSnapshotCache()


__all__ = [
    "DEFAULT_PROFILE_GRAPH_CACHE",
    "GraphNeighborRef",
    "ProfileGraphSnapshot",
    "ProfileGraphSnapshotCache",
]
