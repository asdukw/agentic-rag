from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import case, delete, func, or_, select, update
from sqlalchemy.orm import Session

from hybrid_rag.deepseek_costs import DeepSeekUsage, aggregate_deepseek_usage
from hybrid_rag.deepseek_costs import deepseek_usage as make_deepseek_usage
from hybrid_rag.ids import canonical_json_hash, stable_id
from hybrid_rag.storage.models import (
    ChunkExtractionRecord,
    ChunkRecord,
    EntityEvidenceRecord,
    EntityRecord,
    ExtractionAttemptRecord,
    GraphBuildItemRecord,
    GraphBuildRunRecord,
    RelationEvidenceRecord,
    RelationRecord,
)

RUN_STATUSES = {
    "running",
    "awaiting_review",
    "completed",
    "completed_with_failures",
    "failed",
}
EXTRACTION_STATUSES = {"pending", "running", "succeeded", "needs_review", "failed"}
ATTEMPT_STAGES = {"extract", "repair"}


def _nonnegative_optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


class GraphRepositoryError(RuntimeError):
    """Base error for invalid graph persistence operations."""


class StaleExtractionLeaseError(GraphRepositoryError):
    """Raised when a reclaimed worker tries to finalize an old lease."""


@dataclass(frozen=True, slots=True)
class BuildRunState:
    id: str
    extraction_config_hash: str
    graph_config_hash: str
    corpus_hash: str
    model: str
    prompt_version: str
    schema_version: str
    workflow_version: str
    status: str
    review_required: bool
    started_at: datetime
    updated_at: datetime
    finished_at: datetime | None
    total_chunks: int
    cached_chunks: int
    scheduled_chunks: int
    succeeded_chunks: int
    needs_review_chunks: int
    failed_chunks: int
    attempt_count: int
    extract_attempt_count: int
    repair_attempt_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    entity_count: int
    relation_count: int
    component_count: int
    largest_component_nodes: int
    isolated_entity_count: int
    report: dict[str, Any]
    error: str | None


@dataclass(frozen=True, slots=True)
class JobPreparation:
    run_id: str
    total: int
    cached: int
    scheduled: int
    pending: int
    succeeded: int
    needs_review: int
    failed: int


@dataclass(frozen=True, slots=True)
class ExtractionClaim:
    extraction_id: str
    attempt_id: str
    attempt_ordinal: int
    run_id: str
    run_attempt_number: int
    chunk_id: str
    stage: str
    lease_token: str
    lease_expires_at: datetime


class GraphRepository:
    """SQL persistence for extraction jobs and the current canonical graph snapshot.

    Methods deliberately never commit. Callers own short database transactions and
    must perform provider calls after the transaction containing ``claim_extraction``
    has committed.
    """

    def compute_corpus_hash(
        self,
        session: Session,
        chunk_ids: Sequence[str] | None = None,
        *,
        limit: int | None = None,
    ) -> str:
        chunks = self._select_chunks(session, chunk_ids, limit=limit)
        return self._hash_chunks(chunks)

    def begin_run(
        self,
        session: Session,
        *,
        extraction_config_hash: str,
        graph_config_hash: str,
        model: str,
        prompt_version: str,
        schema_version: str,
        workflow_version: str,
        review_required: bool = False,
        chunk_ids: Sequence[str] | None = None,
        limit: int | None = None,
        corpus_hash: str | None = None,
        run_id: str | None = None,
    ) -> BuildRunState:
        identifier = run_id or f"gbr_{uuid4().hex}"
        existing = session.get(GraphBuildRunRecord, identifier)
        if existing is not None:
            expected = (
                extraction_config_hash,
                graph_config_hash,
                model,
                prompt_version,
                schema_version,
                workflow_version,
            )
            actual = (
                existing.extraction_config_hash,
                existing.graph_config_hash,
                existing.model,
                existing.prompt_version,
                existing.schema_version,
                existing.workflow_version,
            )
            if actual != expected:
                raise GraphRepositoryError(
                    f"run {identifier} cannot resume with different semantic configuration"
                )
            return self._run_state(existing)
        if not identifier.startswith("gbr_"):
            raise ValueError("graph build run IDs must use the gbr_ prefix")

        digest = corpus_hash or self.compute_corpus_hash(session, chunk_ids, limit=limit)
        now = datetime.now(UTC)
        record = GraphBuildRunRecord(
            id=identifier,
            extraction_config_hash=extraction_config_hash,
            graph_config_hash=graph_config_hash,
            corpus_hash=digest,
            model=model,
            prompt_version=prompt_version,
            schema_version=schema_version,
            workflow_version=workflow_version,
            status="running",
            review_required=review_required,
            started_at=now,
            updated_at=now,
            total_chunks=0,
            cached_chunks=0,
            scheduled_chunks=0,
            succeeded_chunks=0,
            needs_review_chunks=0,
            failed_chunks=0,
            attempt_count=0,
            extract_attempt_count=0,
            repair_attempt_count=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            entity_count=0,
            relation_count=0,
            component_count=0,
            largest_component_nodes=0,
            isolated_entity_count=0,
            report_json={},
        )
        session.add(record)
        session.flush()
        return self._run_state(record)

    def get_run(self, session: Session, run_id: str) -> BuildRunState | None:
        record = session.get(GraphBuildRunRecord, run_id)
        return self._run_state(record) if record is not None else None

    def prepare_jobs(
        self,
        session: Session,
        run_id: str,
        *,
        chunk_ids: Sequence[str] | None = None,
        limit: int | None = None,
        retry_failed: bool = True,
    ) -> JobPreparation:
        run = self._require_run(session, run_id)
        if run.status == "failed":
            raise GraphRepositoryError(f"failed graph build run cannot resume: {run_id}")
        existing_items = (
            session.scalar(
                select(func.count())
                .select_from(GraphBuildItemRecord)
                .where(GraphBuildItemRecord.run_id == run_id)
            )
            or 0
        )
        if existing_items:
            if retry_failed:
                self._requeue_failed(session, run_id)
            self._refresh_run_counters(session, run)
            return self._preparation(run)

        chunks = self._select_chunks(session, chunk_ids, limit=limit)
        actual_corpus_hash = self._hash_chunks(chunks)
        if actual_corpus_hash != run.corpus_hash:
            raise GraphRepositoryError(
                "selected chunks no longer match the run corpus hash; start a new run"
            )

        for ordinal, chunk in enumerate(chunks):
            extraction_id = stable_id(
                "xtr", chunk.id, chunk.content_hash, run.extraction_config_hash
            )
            extraction = session.get(ChunkExtractionRecord, extraction_id)
            if extraction is None:
                extraction = ChunkExtractionRecord(
                    id=extraction_id,
                    chunk_id=chunk.id,
                    extraction_config_hash=run.extraction_config_hash,
                    model=run.model,
                    prompt_version=run.prompt_version,
                    schema_version=run.schema_version,
                    status="pending",
                    attempt_count=0,
                )
                session.add(extraction)
                session.flush()
            elif (
                extraction.chunk_id != chunk.id
                or extraction.extraction_config_hash != run.extraction_config_hash
                or extraction.model != run.model
                or extraction.prompt_version != run.prompt_version
                or extraction.schema_version != run.schema_version
            ):
                raise GraphRepositoryError(
                    f"deterministic extraction ID collision for {extraction_id}"
                )
            if retry_failed and extraction.status == "failed":
                self._reset_for_retry(extraction)
            disposition = "cached" if extraction.status == "succeeded" else "scheduled"
            review_status = (
                "pending" if run.review_required and disposition == "scheduled" else "not_required"
            )
            session.add(
                GraphBuildItemRecord(
                    run_id=run_id,
                    extraction_id=extraction.id,
                    ordinal=ordinal,
                    disposition=disposition,
                    review_status=review_status,
                )
            )

        session.flush()
        self._refresh_run_counters(session, run)
        return self._preparation(run)

    def list_pending_jobs(
        self,
        session: Session,
        run_id: str,
        *,
        now: datetime | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self._require_run(session, run_id)
        run_attempt_counts = {
            extraction_id: count
            for extraction_id, count in session.execute(
                select(ExtractionAttemptRecord.extraction_id, func.count())
                .where(ExtractionAttemptRecord.run_id == run_id)
                .group_by(ExtractionAttemptRecord.extraction_id)
            ).tuples()
        }
        current = now or datetime.now(UTC)
        eligible = or_(
            (ChunkExtractionRecord.status == "pending")
            & or_(
                ChunkExtractionRecord.next_attempt_at.is_(None),
                ChunkExtractionRecord.next_attempt_at <= current,
            ),
            (ChunkExtractionRecord.status == "running")
            & (ChunkExtractionRecord.lease_expires_at <= current),
        )
        statement = (
            select(ChunkExtractionRecord, ChunkRecord)
            .join(
                GraphBuildItemRecord,
                GraphBuildItemRecord.extraction_id == ChunkExtractionRecord.id,
            )
            .join(ChunkRecord, ChunkRecord.id == ChunkExtractionRecord.chunk_id)
            .where(GraphBuildItemRecord.run_id == run_id, eligible)
            .order_by(ChunkRecord.id)
        )
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            statement = statement.limit(limit)
        return [
            {
                "id": extraction.id,
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "status": extraction.status,
                "attempt_count": extraction.attempt_count,
                "run_attempt_count": run_attempt_counts.get(extraction.id, 0),
                "text": chunk.text,
                "contextualized_text": chunk.contextualized_text,
            }
            for extraction, chunk in session.execute(statement).all()
        ]

    def claim_extraction(
        self,
        session: Session,
        extraction_id: str,
        *,
        run_id: str | None = None,
        stage: str,
        messages: Sequence[Mapping[str, Any]],
        lease_seconds: float = 300,
        max_attempts: int | None = None,
        now: datetime | None = None,
    ) -> ExtractionClaim | None:
        if stage not in ATTEMPT_STAGES:
            raise ValueError(f"unsupported attempt stage: {stage}")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        owner_run_id = self._attempt_run_id(session, extraction_id, run_id)
        owned_attempts = (
            session.scalar(
                select(func.count())
                .select_from(ExtractionAttemptRecord)
                .where(
                    ExtractionAttemptRecord.extraction_id == extraction_id,
                    ExtractionAttemptRecord.run_id == owner_run_id,
                )
            )
            or 0
        )
        if max_attempts is not None:
            if max_attempts <= 0:
                raise ValueError("max_attempts must be positive")
            if owned_attempts >= max_attempts:
                return None
        current = now or datetime.now(UTC)
        lease_expires_at = current + timedelta(seconds=lease_seconds)
        token = uuid4().hex
        eligible = or_(
            (ChunkExtractionRecord.status == "pending")
            & or_(
                ChunkExtractionRecord.next_attempt_at.is_(None),
                ChunkExtractionRecord.next_attempt_at <= current,
            ),
            (ChunkExtractionRecord.status == "running")
            & (ChunkExtractionRecord.lease_expires_at <= current),
        )
        conditions = [ChunkExtractionRecord.id == extraction_id, eligible]
        claimed = session.execute(
            update(ChunkExtractionRecord)
            .where(*conditions)
            .values(
                status="running",
                attempt_count=ChunkExtractionRecord.attempt_count + 1,
                lease_token=token,
                lease_expires_at=lease_expires_at,
                next_attempt_at=None,
                updated_at=current,
            )
            .returning(
                ChunkExtractionRecord.attempt_count,
                ChunkExtractionRecord.chunk_id,
            )
        ).one_or_none()
        if claimed is None:
            return None

        ordinal, chunk_id = claimed
        session.execute(
            update(ExtractionAttemptRecord)
            .where(
                ExtractionAttemptRecord.extraction_id == extraction_id,
                ExtractionAttemptRecord.outcome == "running",
            )
            .values(outcome="interrupted", finished_at=current)
        )
        attempt_id = stable_id("xat", extraction_id, str(ordinal))
        session.add(
            ExtractionAttemptRecord(
                id=attempt_id,
                extraction_id=extraction_id,
                run_id=owner_run_id,
                ordinal=ordinal,
                stage=stage,
                outcome="running",
                messages_json=[dict(message) for message in messages],
                response_metadata_json={},
                started_at=current,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )
        )
        session.flush()
        return ExtractionClaim(
            extraction_id=extraction_id,
            attempt_id=attempt_id,
            attempt_ordinal=ordinal,
            run_id=owner_run_id,
            run_attempt_number=owned_attempts + 1,
            chunk_id=chunk_id,
            stage=stage,
            lease_token=token,
            lease_expires_at=lease_expires_at,
        )

    def record_attempt(
        self,
        session: Session,
        claim: ExtractionClaim,
        *,
        outcome: str,
        raw_response: str | None = None,
        response_metadata: Mapping[str, Any] | None = None,
        error: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
        latency_seconds: float | None = None,
        finished_at: datetime | None = None,
    ) -> dict[str, Any]:
        attempt = session.get(ExtractionAttemptRecord, claim.attempt_id)
        if attempt is None or attempt.extraction_id != claim.extraction_id:
            raise GraphRepositoryError(f"attempt not found: {claim.attempt_id}")
        if attempt.outcome != "running":
            raise GraphRepositoryError(
                f"attempt {claim.attempt_id} is already finalized as {attempt.outcome}"
            )
        if outcome == "running":
            raise ValueError("record_attempt requires a final outcome")
        if min(prompt_tokens, completion_tokens, total_tokens or 0) < 0:
            raise ValueError("token usage cannot be negative")
        finished = finished_at or datetime.now(UTC)
        attempt.outcome = outcome
        attempt.raw_response = raw_response
        attempt.response_metadata_json = dict(response_metadata or {})
        attempt.error = error
        attempt.finished_at = finished
        attempt.latency_seconds = (
            latency_seconds
            if latency_seconds is not None
            else self._seconds_between(attempt.started_at, finished)
        )
        attempt.prompt_tokens = prompt_tokens
        attempt.completion_tokens = completion_tokens
        attempt.total_tokens = (
            total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
        )
        session.flush()
        return self._attempt_dict(attempt)

    def complete_extraction(
        self,
        session: Session,
        claim: ExtractionClaim,
        result: Mapping[str, Any] | Any,
        *,
        needs_review: bool = False,
        completed_at: datetime | None = None,
    ) -> dict[str, Any]:
        completed = completed_at or datetime.now(UTC)
        payload = self._as_mapping(result)
        owner_item = session.get(GraphBuildItemRecord, (claim.run_id, claim.extraction_id))
        if needs_review and (owner_item is None or owner_item.review_status != "pending"):
            raise GraphRepositoryError(
                "review-required completion is not linked to a pending review item"
            )
        changed = session.execute(
            update(ChunkExtractionRecord)
            .where(
                ChunkExtractionRecord.id == claim.extraction_id,
                ChunkExtractionRecord.status == "running",
                ChunkExtractionRecord.lease_token == claim.lease_token,
            )
            .values(
                status="succeeded",
                result_json=payload,
                error=None,
                lease_token=None,
                lease_expires_at=None,
                next_attempt_at=None,
                updated_at=completed,
                completed_at=completed,
            )
        ).rowcount
        if changed != 1:
            raise StaleExtractionLeaseError(f"lease is no longer current for {claim.extraction_id}")
        session.execute(
            update(ExtractionAttemptRecord)
            .where(
                ExtractionAttemptRecord.id == claim.attempt_id,
                ExtractionAttemptRecord.outcome == "running",
            )
            .values(
                outcome="succeeded",
                finished_at=completed,
            )
        )
        self._refresh_linked_runs(session, claim.extraction_id)
        return self._extraction_dict(self._require_extraction(session, claim.extraction_id))

    def requeue_extraction(
        self,
        session: Session,
        claim: ExtractionClaim,
        *,
        error: str,
        outcome: str,
        retry_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Persist a failed call and release its lease for another claim/attempt."""
        return self.fail_extraction(
            session,
            claim,
            error=error,
            outcome=outcome,
            status="pending",
            retry_at=retry_at,
            finished_at=finished_at,
        )

    def fail_exhausted_extraction(
        self,
        session: Session,
        extraction_id: str,
        *,
        run_id: str,
        max_attempts: int,
        error: str = "attempt budget exhausted before an interrupted call completed",
        now: datetime | None = None,
    ) -> bool:
        """Close an eligible orphan that cannot claim another billed attempt."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        owned_attempts = (
            session.scalar(
                select(func.count())
                .select_from(ExtractionAttemptRecord)
                .where(
                    ExtractionAttemptRecord.extraction_id == extraction_id,
                    ExtractionAttemptRecord.run_id == run_id,
                )
            )
            or 0
        )
        if owned_attempts < max_attempts:
            return False
        current = now or datetime.now(UTC)
        eligible = or_(
            ChunkExtractionRecord.status == "pending",
            (ChunkExtractionRecord.status == "running")
            & (ChunkExtractionRecord.lease_expires_at <= current),
        )
        changed = session.execute(
            update(ChunkExtractionRecord)
            .where(
                ChunkExtractionRecord.id == extraction_id,
                eligible,
            )
            .values(
                status="failed",
                error=error,
                lease_token=None,
                lease_expires_at=None,
                next_attempt_at=None,
                updated_at=current,
                completed_at=current,
            )
        ).rowcount
        if changed != 1:
            return False
        session.execute(
            update(ExtractionAttemptRecord)
            .where(
                ExtractionAttemptRecord.extraction_id == extraction_id,
                ExtractionAttemptRecord.run_id == run_id,
                ExtractionAttemptRecord.outcome == "running",
            )
            .values(outcome="interrupted", error=error, finished_at=current)
        )
        self._refresh_linked_runs(session, extraction_id)
        return True

    def fail_extraction(
        self,
        session: Session,
        claim: ExtractionClaim,
        *,
        error: str,
        outcome: str,
        status: str = "failed",
        retry_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> dict[str, Any]:
        if status not in {"pending", "failed"}:
            raise ValueError("failed extraction status must be pending or failed")
        finished = finished_at or datetime.now(UTC)
        changed = session.execute(
            update(ChunkExtractionRecord)
            .where(
                ChunkExtractionRecord.id == claim.extraction_id,
                ChunkExtractionRecord.status == "running",
                ChunkExtractionRecord.lease_token == claim.lease_token,
            )
            .values(
                status=status,
                error=error,
                lease_token=None,
                lease_expires_at=None,
                next_attempt_at=retry_at if status == "pending" else None,
                updated_at=finished,
                completed_at=finished if status != "pending" else None,
            )
        ).rowcount
        if changed != 1:
            raise StaleExtractionLeaseError(f"lease is no longer current for {claim.extraction_id}")
        session.execute(
            update(ExtractionAttemptRecord)
            .where(
                ExtractionAttemptRecord.id == claim.attempt_id,
                ExtractionAttemptRecord.outcome == "running",
            )
            .values(outcome=outcome, error=error, finished_at=finished)
        )
        self._refresh_linked_runs(session, claim.extraction_id)
        return self._extraction_dict(self._require_extraction(session, claim.extraction_id))

    def review_extraction(
        self,
        session: Session,
        extraction_id: str,
        *,
        decision: str,
        run_id: str | None = None,
        result: Mapping[str, Any] | Any | None = None,
        notes: str | None = None,
        reviewed_at: datetime | None = None,
    ) -> dict[str, Any]:
        extraction = self._require_extraction(session, extraction_id)
        if extraction.status != "succeeded" or extraction.result_json is None:
            raise GraphRepositoryError(
                f"extraction has no validated result to review: {extraction_id}"
            )
        if result is not None:
            raise GraphRepositoryError(
                "edited review payloads must be revalidated before repository approval"
            )
        pending_statement = select(GraphBuildItemRecord).where(
            GraphBuildItemRecord.extraction_id == extraction_id,
            GraphBuildItemRecord.review_status == "pending",
        )
        if run_id is not None:
            pending_statement = pending_statement.where(GraphBuildItemRecord.run_id == run_id)
        pending_items = list(session.scalars(pending_statement))
        if not pending_items:
            raise GraphRepositoryError(
                f"extraction is not awaiting review in any run: {extraction_id}"
            )
        if run_id is None and len(pending_items) > 1:
            raise GraphRepositoryError("extraction awaits review in multiple runs; specify run_id")
        reviewed = reviewed_at or datetime.now(UTC)
        if decision == "approve":
            review_status = "approved"
        elif decision == "reject":
            review_status = "rejected"
        else:
            raise ValueError("decision must be approve or reject")
        for item in pending_items:
            item.review_status = review_status
            item.review_notes = notes
            item.reviewed_at = reviewed
        self._refresh_linked_runs(session, extraction_id)
        for run_id in self._linked_run_ids(session, extraction_id):
            run = session.get(GraphBuildRunRecord, run_id)
            if run is not None and run.status == "awaiting_review" and run.needs_review_chunks == 0:
                run.status = "running"
                run.finished_at = None
        session.flush()
        payload = self._extraction_dict(extraction)
        payload["review_status"] = review_status
        payload["reviewed_run_ids"] = sorted(item.run_id for item in pending_items)
        return payload

    def accepted_results(self, session: Session, run_id: str) -> list[dict[str, Any]]:
        self._require_run(session, run_id)
        statement = (
            select(ChunkExtractionRecord)
            .join(
                GraphBuildItemRecord,
                GraphBuildItemRecord.extraction_id == ChunkExtractionRecord.id,
            )
            .where(
                GraphBuildItemRecord.run_id == run_id,
                ChunkExtractionRecord.status == "succeeded",
                GraphBuildItemRecord.review_status.in_({"not_required", "approved"}),
            )
            .order_by(ChunkExtractionRecord.chunk_id, ChunkExtractionRecord.id)
        )
        return [self._extraction_dict(record) for record in session.scalars(statement)]

    def replace_snapshot(
        self,
        session: Session,
        run_id: str,
        entities: Sequence[Mapping[str, Any] | Any],
        relations: Sequence[Mapping[str, Any] | Any],
        *,
        component_count: int = 0,
        largest_component_nodes: int = 0,
        isolated_entity_count: int = 0,
    ) -> dict[str, Any]:
        """Replace all current graph rows; caller transaction makes this atomic."""
        run = self._require_run(session, run_id)
        if run.status == "failed":
            raise GraphRepositoryError("failed graph build cannot replace the current snapshot")
        if min(component_count, largest_component_nodes, isolated_entity_count) < 0:
            raise ValueError("graph metrics cannot be negative")
        extraction_by_chunk = {
            chunk_id: extraction_id
            for chunk_id, extraction_id in session.execute(
                select(ChunkExtractionRecord.chunk_id, ChunkExtractionRecord.id)
                .join(
                    GraphBuildItemRecord,
                    GraphBuildItemRecord.extraction_id == ChunkExtractionRecord.id,
                )
                .where(GraphBuildItemRecord.run_id == run_id)
            ).tuples()
        }

        session.execute(delete(RelationRecord))
        session.execute(delete(EntityRecord))
        session.flush()

        entity_payloads = [self._as_mapping(value) for value in entities]
        relation_payloads = [self._as_mapping(value) for value in relations]
        entity_ids: set[str] = set()
        for payload in entity_payloads:
            entity_id = str(payload["id"])
            if entity_id in entity_ids:
                raise GraphRepositoryError(f"duplicate canonical entity ID: {entity_id}")
            entity_ids.add(entity_id)
            evidence = [self._as_mapping(value) for value in payload.get("evidence", [])]
            source_chunk_ids = self._source_chunk_ids(payload, evidence)
            record = EntityRecord(
                id=entity_id,
                build_run_id=run_id,
                graph_config_hash=run.graph_config_hash,
                canonical_name=str(payload["canonical_name"]),
                normalized_name=str(
                    payload.get("normalized_name", str(payload["canonical_name"]).casefold())
                ),
                entity_type=str(payload["entity_type"]),
                description=str(payload["description"]),
                aliases_json=self._sorted_strings(payload.get("aliases", [])),
                source_chunk_ids_json=source_chunk_ids,
            )
            session.add(record)
            self._add_entity_evidence(
                session,
                record,
                evidence,
                extraction_by_chunk,
            )
        session.flush()

        relation_ids: set[str] = set()
        for payload in relation_payloads:
            relation_id = str(payload["id"])
            if relation_id in relation_ids:
                raise GraphRepositoryError(f"duplicate canonical relation ID: {relation_id}")
            source_entity_id = str(payload["source_entity_id"])
            target_entity_id = str(payload["target_entity_id"])
            if source_entity_id not in entity_ids or target_entity_id not in entity_ids:
                raise GraphRepositoryError(
                    f"relation {relation_id} references an entity outside the snapshot"
                )
            relation_ids.add(relation_id)
            evidence = [self._as_mapping(value) for value in payload.get("evidence", [])]
            record = RelationRecord(
                id=relation_id,
                build_run_id=run_id,
                graph_config_hash=run.graph_config_hash,
                source_entity_id=source_entity_id,
                target_entity_id=target_entity_id,
                predicate=str(payload["predicate"]),
                description=str(payload["description"]),
                source_chunk_ids_json=self._source_chunk_ids(payload, evidence),
            )
            session.add(record)
            self._add_relation_evidence(
                session,
                record,
                evidence,
                extraction_by_chunk,
            )

        session.flush()
        run.entity_count = len(entity_payloads)
        run.relation_count = len(relation_payloads)
        run.component_count = component_count
        run.largest_component_nodes = largest_component_nodes
        run.isolated_entity_count = isolated_entity_count
        run.updated_at = datetime.now(UTC)
        return self.stats(session, run_id=run_id)

    def finalize_run(
        self,
        session: Session,
        run_id: str,
        *,
        status: str | None = None,
        report: Mapping[str, Any] | Any | None = None,
        error: str | None = None,
        finished_at: datetime | None = None,
    ) -> BuildRunState:
        run = self._require_run(session, run_id)
        attaching_terminal_report = status is None and report is not None and run.error is None
        if run.status == "failed" and status != "failed" and not attaching_terminal_report:
            raise GraphRepositoryError(f"failed graph build run cannot be reopened: {run_id}")
        self._refresh_run_counters(session, run)
        final_status = status or self._derived_run_status(run)
        if final_status not in RUN_STATUSES:
            raise ValueError(f"unsupported graph build status: {final_status}")
        now = finished_at or datetime.now(UTC)
        run.status = final_status
        run.updated_at = now
        run.finished_at = None if final_status in {"running", "awaiting_review"} else now
        if report is not None:
            run.report_json = self._as_mapping(report)
        run.error = error
        session.flush()
        return self._run_state(run)

    def stats(
        self,
        session: Session,
        *,
        run_id: str | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        run = self._current_snapshot_run(session, run_id)
        if run is None:
            return {
                "run_id": run_id,
                "status": None,
                "chunks": {},
                "nodes": 0,
                "edges": 0,
                "weakly_connected_components": 0,
                "largest_component_nodes": 0,
                "isolated_nodes": 0,
                "top_entities": [],
            }
        entities = list(
            session.scalars(
                select(EntityRecord)
                .where(EntityRecord.build_run_id == run.id)
                .order_by(EntityRecord.id)
            )
        )
        persisted_report = run.report_json.get("final_report")
        if (
            not entities
            and run.entity_count
            and isinstance(persisted_report, Mapping)
            and persisted_report.get("run_id") == run.id
        ):
            persisted = persisted_report
            graph = persisted.get("graph", {})
            return {
                "run_id": run.id,
                "status": run.status,
                "chunks": persisted.get("chunks", {}),
                "attempts": persisted.get("attempts", {}),
                "usage": persisted.get("usage", {}),
                "nodes": graph.get("nodes", run.entity_count),
                "edges": graph.get("edges", run.relation_count),
                "weakly_connected_components": graph.get(
                    "weakly_connected_components", run.component_count
                ),
                "largest_component_nodes": graph.get(
                    "largest_component_nodes", run.largest_component_nodes
                ),
                "isolated_nodes": graph.get("isolated_nodes", run.isolated_entity_count),
                "top_entities": list(graph.get("top_entities", []))[:top_k],
            }
        endpoints = session.execute(
            select(RelationRecord.source_entity_id, RelationRecord.target_entity_id).where(
                RelationRecord.build_run_id == run.id
            )
        ).all()
        degree: Counter[str] = Counter()
        for source_id, target_id in endpoints:
            degree[source_id] += 1
            degree[target_id] += 1
        ordered = sorted(
            entities,
            key=lambda entity: (
                -degree[entity.id],
                entity.canonical_name.casefold(),
                entity.id,
            ),
        )[:top_k]
        return {
            "run_id": run.id,
            "status": run.status,
            "chunks": {
                "total": run.total_chunks,
                "cached": run.cached_chunks,
                "scheduled": run.scheduled_chunks,
                "succeeded": run.succeeded_chunks,
                "needs_review": run.needs_review_chunks,
                "failed": run.failed_chunks,
            },
            "attempts": {
                "total": run.attempt_count,
                "extract": run.extract_attempt_count,
                "repair": run.repair_attempt_count,
            },
            "usage": {
                "prompt_tokens": run.prompt_tokens,
                "completion_tokens": run.completion_tokens,
                "total_tokens": run.total_tokens,
            },
            "nodes": len(entities),
            "edges": len(endpoints),
            "weakly_connected_components": run.component_count,
            "largest_component_nodes": run.largest_component_nodes,
            "isolated_nodes": sum(degree[entity.id] == 0 for entity in entities),
            "top_entities": [
                {
                    "id": entity.id,
                    "name": entity.canonical_name,
                    "type": entity.entity_type,
                    "degree": degree[entity.id],
                    "source_chunks": len(entity.source_chunk_ids_json),
                }
                for entity in ordered
            ],
        }

    def deepseek_usage(self, session: Session, run_id: str) -> tuple[DeepSeekUsage, ...]:
        """Return response usage grouped by graph stage and actual response model.

        Cache token fields have been persisted in attempt response metadata since
        the graph workflow was introduced.  Keeping this projection at the
        repository boundary lets old runs remain readable: a run without a
        complete cache split is reported as unpriced instead of guessed.
        """

        run = session.get(GraphBuildRunRecord, run_id)
        if run is None:
            raise GraphRepositoryError(f"graph build run not found: {run_id}")
        rows = session.execute(
            select(
                ExtractionAttemptRecord.stage,
                ExtractionAttemptRecord.response_metadata_json,
                ExtractionAttemptRecord.prompt_tokens,
                ExtractionAttemptRecord.completion_tokens,
            )
            .where(ExtractionAttemptRecord.run_id == run_id)
            .order_by(ExtractionAttemptRecord.id)
        ).all()
        records: list[DeepSeekUsage] = []
        for stage, metadata, prompt_tokens, completion_tokens in rows:
            response = metadata if isinstance(metadata, Mapping) else {}
            model = response.get("model")
            if not isinstance(model, str) or not model.strip():
                if not prompt_tokens and not completion_tokens:
                    continue
                model = "<response-model-unavailable>"
            records.append(
                make_deepseek_usage(
                    operation=str(stage),
                    model=model,
                    prompt_tokens=int(prompt_tokens),
                    cache_hit_tokens=_nonnegative_optional_int(response.get("cache_hit_tokens")),
                    cache_miss_tokens=_nonnegative_optional_int(response.get("cache_miss_tokens")),
                    completion_tokens=int(completion_tokens),
                )
            )
        return aggregate_deepseek_usage(records)

    def inspect(self, session: Session, object_id: str) -> dict[str, Any] | None:
        if object_id.startswith("gbr_"):
            run = session.get(GraphBuildRunRecord, object_id)
            if run is None:
                return None
            payload = asdict(self._run_state(run))
            payload["kind"] = "graph_build_run"
            payload["items"] = self._run_items(session, object_id)
            return payload
        if object_id.startswith("xtr_"):
            extraction = session.get(ChunkExtractionRecord, object_id)
            if extraction is None:
                return None
            payload = self._extraction_dict(extraction)
            payload["kind"] = "chunk_extraction"
            payload["attempts"] = [
                self._attempt_dict(attempt)
                for attempt in session.scalars(
                    select(ExtractionAttemptRecord)
                    .where(ExtractionAttemptRecord.extraction_id == object_id)
                    .order_by(ExtractionAttemptRecord.ordinal)
                )
            ]
            payload["reviews"] = [
                {
                    "run_id": item.run_id,
                    "status": item.review_status,
                    "notes": item.review_notes,
                    "reviewed_at": item.reviewed_at,
                }
                for item in session.scalars(
                    select(GraphBuildItemRecord)
                    .where(GraphBuildItemRecord.extraction_id == object_id)
                    .order_by(GraphBuildItemRecord.run_id)
                )
                if item.review_status != "not_required"
            ]
            return payload
        if object_id.startswith("xat_"):
            attempt = session.get(ExtractionAttemptRecord, object_id)
            if attempt is None:
                return None
            return {"kind": "extraction_attempt", **self._attempt_dict(attempt)}
        if object_id.startswith("ent_"):
            entity = session.get(EntityRecord, object_id)
            if entity is None:
                return None
            return self._entity_inspect(session, entity)
        if object_id.startswith("rel_"):
            relation = session.get(RelationRecord, object_id)
            if relation is None:
                return None
            return self._relation_inspect(session, relation)
        return None

    def invalidate_runs_for_document(
        self,
        session: Session,
        document_id: str,
        *,
        reason: str = "source document changed after graph extraction",
    ) -> int:
        run_ids = list(
            session.scalars(
                select(GraphBuildItemRecord.run_id)
                .join(
                    ChunkExtractionRecord,
                    ChunkExtractionRecord.id == GraphBuildItemRecord.extraction_id,
                )
                .join(ChunkRecord, ChunkRecord.id == ChunkExtractionRecord.chunk_id)
                .where(ChunkRecord.document_id == document_id)
                .distinct()
            )
        )
        if not run_ids:
            return 0
        session.execute(delete(RelationRecord).where(RelationRecord.build_run_id.in_(run_ids)))
        session.execute(delete(EntityRecord).where(EntityRecord.build_run_id.in_(run_ids)))
        now = datetime.now(UTC)
        session.execute(
            update(GraphBuildRunRecord)
            .where(GraphBuildRunRecord.id.in_(run_ids))
            .values(
                status="failed",
                error=reason,
                entity_count=0,
                relation_count=0,
                component_count=0,
                largest_component_nodes=0,
                isolated_entity_count=0,
                updated_at=now,
                finished_at=now,
            )
        )
        return len(run_ids)

    @staticmethod
    def _hash_chunks(chunks: Sequence[ChunkRecord]) -> str:
        parts = sorted((chunk.id, chunk.content_hash) for chunk in chunks)
        return canonical_json_hash({"chunks": parts})

    @staticmethod
    def _select_chunks(
        session: Session,
        chunk_ids: Sequence[str] | None,
        *,
        limit: int | None,
    ) -> list[ChunkRecord]:
        statement = (
            select(ChunkRecord)
            .where(ChunkRecord.quality_class == "normal")
            .order_by(
                ChunkRecord.document_id,
                ChunkRecord.ordinal,
                ChunkRecord.id,
            )
        )
        if chunk_ids is not None:
            statement = statement.where(ChunkRecord.id.in_(list(chunk_ids)))
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        chunks = list(session.scalars(statement))
        return chunks[:limit] if limit is not None else chunks

    @staticmethod
    def _run_state(record: GraphBuildRunRecord) -> BuildRunState:
        return BuildRunState(
            id=record.id,
            extraction_config_hash=record.extraction_config_hash,
            graph_config_hash=record.graph_config_hash,
            corpus_hash=record.corpus_hash,
            model=record.model,
            prompt_version=record.prompt_version,
            schema_version=record.schema_version,
            workflow_version=record.workflow_version,
            status=record.status,
            review_required=record.review_required,
            started_at=record.started_at,
            updated_at=record.updated_at,
            finished_at=record.finished_at,
            total_chunks=record.total_chunks,
            cached_chunks=record.cached_chunks,
            scheduled_chunks=record.scheduled_chunks,
            succeeded_chunks=record.succeeded_chunks,
            needs_review_chunks=record.needs_review_chunks,
            failed_chunks=record.failed_chunks,
            attempt_count=record.attempt_count,
            extract_attempt_count=record.extract_attempt_count,
            repair_attempt_count=record.repair_attempt_count,
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            total_tokens=record.total_tokens,
            entity_count=record.entity_count,
            relation_count=record.relation_count,
            component_count=record.component_count,
            largest_component_nodes=record.largest_component_nodes,
            isolated_entity_count=record.isolated_entity_count,
            report=dict(record.report_json),
            error=record.error,
        )

    @staticmethod
    def _preparation(run: GraphBuildRunRecord) -> JobPreparation:
        pending = run.total_chunks - (
            run.succeeded_chunks + run.needs_review_chunks + run.failed_chunks
        )
        return JobPreparation(
            run_id=run.id,
            total=run.total_chunks,
            cached=run.cached_chunks,
            scheduled=run.scheduled_chunks,
            pending=max(pending, 0),
            succeeded=run.succeeded_chunks,
            needs_review=run.needs_review_chunks,
            failed=run.failed_chunks,
        )

    @staticmethod
    def _require_run(session: Session, run_id: str) -> GraphBuildRunRecord:
        record = session.get(GraphBuildRunRecord, run_id)
        if record is None:
            raise GraphRepositoryError(f"graph build run not found: {run_id}")
        return record

    @staticmethod
    def _require_extraction(session: Session, extraction_id: str) -> ChunkExtractionRecord:
        record = session.get(ChunkExtractionRecord, extraction_id)
        if record is None:
            raise GraphRepositoryError(f"chunk extraction not found: {extraction_id}")
        return record

    def _refresh_run_counters(self, session: Session, run: GraphBuildRunRecord) -> None:
        statuses = session.execute(
            select(
                ChunkExtractionRecord.status,
                GraphBuildItemRecord.review_status,
            )
            .join(
                GraphBuildItemRecord,
                GraphBuildItemRecord.extraction_id == ChunkExtractionRecord.id,
            )
            .where(GraphBuildItemRecord.run_id == run.id)
        ).all()
        succeeded = sum(
            extraction_status == "succeeded" and review_status in {"not_required", "approved"}
            for extraction_status, review_status in statuses
        )
        needs_review = sum(
            extraction_status in {"succeeded", "needs_review"} and review_status == "pending"
            for extraction_status, review_status in statuses
        )
        failed = sum(
            extraction_status == "failed" or review_status == "rejected"
            for extraction_status, review_status in statuses
        )
        disposition_counts = {
            disposition: count
            for disposition, count in session.execute(
                select(GraphBuildItemRecord.disposition, func.count())
                .where(GraphBuildItemRecord.run_id == run.id)
                .group_by(GraphBuildItemRecord.disposition)
            ).tuples()
        }
        attempt_row = session.execute(
            select(
                func.count(ExtractionAttemptRecord.id),
                func.coalesce(
                    func.sum(case((ExtractionAttemptRecord.stage == "extract", 1), else_=0)),
                    0,
                ),
                func.coalesce(
                    func.sum(case((ExtractionAttemptRecord.stage == "repair", 1), else_=0)),
                    0,
                ),
                func.coalesce(func.sum(ExtractionAttemptRecord.prompt_tokens), 0),
                func.coalesce(func.sum(ExtractionAttemptRecord.completion_tokens), 0),
                func.coalesce(func.sum(ExtractionAttemptRecord.total_tokens), 0),
            )
            .join(
                GraphBuildItemRecord,
                GraphBuildItemRecord.extraction_id == ExtractionAttemptRecord.extraction_id,
            )
            .where(
                GraphBuildItemRecord.run_id == run.id,
                ExtractionAttemptRecord.run_id == run.id,
            )
        ).one()
        run.total_chunks = len(statuses)
        run.cached_chunks = disposition_counts.get("cached", 0)
        run.scheduled_chunks = disposition_counts.get("scheduled", 0)
        run.succeeded_chunks = succeeded
        run.needs_review_chunks = needs_review
        run.failed_chunks = failed
        run.attempt_count = attempt_row[0] or 0
        run.extract_attempt_count = attempt_row[1] or 0
        run.repair_attempt_count = attempt_row[2] or 0
        run.prompt_tokens = attempt_row[3] or 0
        run.completion_tokens = attempt_row[4] or 0
        run.total_tokens = attempt_row[5] or 0
        run.updated_at = datetime.now(UTC)

    def _refresh_linked_runs(self, session: Session, extraction_id: str) -> None:
        for run_id in self._linked_run_ids(session, extraction_id):
            run = session.get(GraphBuildRunRecord, run_id)
            if run is not None:
                self._refresh_run_counters(session, run)
                if run.needs_review_chunks and run.status == "running":
                    run.status = "awaiting_review"
                    run.updated_at = datetime.now(UTC)

    @staticmethod
    def _linked_run_ids(session: Session, extraction_id: str) -> list[str]:
        return list(
            session.scalars(
                select(GraphBuildItemRecord.run_id).where(
                    GraphBuildItemRecord.extraction_id == extraction_id
                )
            )
        )

    @staticmethod
    def _attempt_run_id(
        session: Session,
        extraction_id: str,
        requested_run_id: str | None,
    ) -> str:
        statement = (
            select(GraphBuildRunRecord.id)
            .join(
                GraphBuildItemRecord,
                GraphBuildItemRecord.run_id == GraphBuildRunRecord.id,
            )
            .where(
                GraphBuildItemRecord.extraction_id == extraction_id,
                GraphBuildRunRecord.status.in_({"running", "awaiting_review"}),
            )
            .order_by(GraphBuildRunRecord.started_at, GraphBuildRunRecord.id)
        )
        if requested_run_id is not None:
            statement = statement.where(GraphBuildRunRecord.id == requested_run_id)
        candidates = list(session.scalars(statement))
        if len(candidates) != 1:
            qualifier = requested_run_id or "an unambiguous active run"
            raise GraphRepositoryError(f"extraction {extraction_id} is not linked to {qualifier}")
        return candidates[0]

    def _requeue_failed(self, session: Session, run_id: str) -> None:
        records = session.scalars(
            select(ChunkExtractionRecord)
            .join(
                GraphBuildItemRecord,
                GraphBuildItemRecord.extraction_id == ChunkExtractionRecord.id,
            )
            .where(
                GraphBuildItemRecord.run_id == run_id,
                ChunkExtractionRecord.status == "failed",
            )
        )
        for record in records:
            self._reset_for_retry(record)

    @staticmethod
    def _reset_for_retry(record: ChunkExtractionRecord) -> None:
        record.status = "pending"
        record.error = None
        record.lease_token = None
        record.lease_expires_at = None
        record.next_attempt_at = None
        record.completed_at = None

    @staticmethod
    def _derived_run_status(run: GraphBuildRunRecord) -> str:
        unfinished = run.total_chunks - (
            run.succeeded_chunks + run.needs_review_chunks + run.failed_chunks
        )
        if unfinished > 0:
            return "running"
        if run.needs_review_chunks:
            return "awaiting_review"
        if run.failed_chunks and run.succeeded_chunks:
            return "completed_with_failures"
        if run.failed_chunks:
            return "failed"
        return "completed"

    @staticmethod
    def _as_mapping(value: Mapping[str, Any] | Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return dict(model_dump(mode="json"))
        raise TypeError(f"expected a mapping or Pydantic model, received {type(value).__name__}")

    @staticmethod
    def _sorted_strings(values: Sequence[Any]) -> list[str]:
        return sorted({str(value) for value in values}, key=lambda value: (value.casefold(), value))

    def _source_chunk_ids(
        self,
        payload: Mapping[str, Any],
        evidence: Sequence[Mapping[str, Any]],
    ) -> list[str]:
        values = list(payload.get("source_chunk_ids", []))
        values.extend(item.get("source_chunk_id", item.get("chunk_id")) for item in evidence)
        return self._sorted_strings([value for value in values if value is not None])

    def _add_entity_evidence(
        self,
        session: Session,
        entity: EntityRecord,
        evidence: Sequence[Mapping[str, Any]],
        extraction_by_chunk: Mapping[str, str],
    ) -> None:
        seen: set[tuple[str, str, int, int]] = set()
        for item in evidence:
            chunk_id, extraction_id, quote, start, end = self._evidence_values(
                item, extraction_by_chunk
            )
            key = (extraction_id, quote, start, end)
            if key in seen:
                continue
            seen.add(key)
            session.add(
                EntityEvidenceRecord(
                    id=str(
                        item.get(
                            "id",
                            stable_id("eev", entity.id, extraction_id, quote, str(start), str(end)),
                        )
                    ),
                    entity_id=entity.id,
                    chunk_id=chunk_id,
                    extraction_id=extraction_id,
                    mention_id=self._optional_string(item.get("mention_id")),
                    quote=quote,
                    char_start=start,
                    char_end=end,
                )
            )

    def _add_relation_evidence(
        self,
        session: Session,
        relation: RelationRecord,
        evidence: Sequence[Mapping[str, Any]],
        extraction_by_chunk: Mapping[str, str],
    ) -> None:
        seen: set[tuple[str, str, int, int]] = set()
        for item in evidence:
            chunk_id, extraction_id, quote, start, end = self._evidence_values(
                item, extraction_by_chunk
            )
            key = (extraction_id, quote, start, end)
            if key in seen:
                continue
            seen.add(key)
            session.add(
                RelationEvidenceRecord(
                    id=str(
                        item.get(
                            "id",
                            stable_id(
                                "rev",
                                relation.id,
                                extraction_id,
                                quote,
                                str(start),
                                str(end),
                            ),
                        )
                    ),
                    relation_id=relation.id,
                    chunk_id=chunk_id,
                    extraction_id=extraction_id,
                    mention_id=self._optional_string(item.get("mention_id")),
                    quote=quote,
                    char_start=start,
                    char_end=end,
                )
            )

    @staticmethod
    def _evidence_values(
        item: Mapping[str, Any],
        extraction_by_chunk: Mapping[str, str],
    ) -> tuple[str, str, str, int, int]:
        chunk_id = str(item.get("source_chunk_id", item.get("chunk_id", "")))
        if not chunk_id:
            raise GraphRepositoryError("evidence is missing source_chunk_id")
        expected_extraction = extraction_by_chunk.get(chunk_id)
        if expected_extraction is None:
            raise GraphRepositoryError(f"evidence chunk {chunk_id} is not part of the graph build")
        extraction_id = str(item.get("extraction_id", expected_extraction))
        if extraction_id != expected_extraction:
            raise GraphRepositoryError(
                f"evidence extraction {extraction_id} does not own chunk {chunk_id}"
            )
        quote = str(item.get("quote", ""))
        if not quote:
            raise GraphRepositoryError("evidence quote cannot be empty")
        start = int(item.get("char_start", 0))
        end = int(item.get("char_end", start + len(quote)))
        if start < 0 or end < start:
            raise GraphRepositoryError("evidence character span is invalid")
        return chunk_id, extraction_id, quote, start, end

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        return None if value is None else str(value)

    @staticmethod
    def _seconds_between(start: datetime, end: datetime) -> float:
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)
        return max((end - start).total_seconds(), 0.0)

    @staticmethod
    def _extraction_dict(record: ChunkExtractionRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "chunk_id": record.chunk_id,
            "extraction_config_hash": record.extraction_config_hash,
            "model": record.model,
            "prompt_version": record.prompt_version,
            "schema_version": record.schema_version,
            "status": record.status,
            "result": record.result_json,
            "error": record.error,
            "attempt_count": record.attempt_count,
            "lease_expires_at": record.lease_expires_at,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "completed_at": record.completed_at,
            "review_notes": record.review_notes,
            "reviewed_at": record.reviewed_at,
        }

    @staticmethod
    def _attempt_dict(record: ExtractionAttemptRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "extraction_id": record.extraction_id,
            "run_id": record.run_id,
            "ordinal": record.ordinal,
            "stage": record.stage,
            "outcome": record.outcome,
            "messages": record.messages_json,
            "raw_response": record.raw_response,
            "response_metadata": record.response_metadata_json,
            "error": record.error,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
            "latency_seconds": record.latency_seconds,
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": record.completion_tokens,
            "total_tokens": record.total_tokens,
        }

    @staticmethod
    def _run_items(session: Session, run_id: str) -> list[dict[str, Any]]:
        rows = session.execute(
            select(GraphBuildItemRecord, ChunkExtractionRecord)
            .join(
                ChunkExtractionRecord,
                ChunkExtractionRecord.id == GraphBuildItemRecord.extraction_id,
            )
            .where(GraphBuildItemRecord.run_id == run_id)
            .order_by(GraphBuildItemRecord.ordinal)
        ).all()
        values = []
        for item, extraction in rows:
            if item.review_status == "rejected":
                status = "failed"
            elif item.review_status == "pending" and extraction.status == "succeeded":
                status = "needs_review"
            else:
                status = extraction.status
            values.append(
                {
                    "ordinal": item.ordinal,
                    "disposition": item.disposition,
                    "extraction_id": extraction.id,
                    "chunk_id": extraction.chunk_id,
                    "status": status,
                    "extraction_status": extraction.status,
                    "review_status": item.review_status,
                    "review_notes": item.review_notes,
                }
            )
        return values

    @staticmethod
    def _entity_inspect(session: Session, entity: EntityRecord) -> dict[str, Any]:
        evidence = list(
            session.scalars(
                select(EntityEvidenceRecord)
                .where(EntityEvidenceRecord.entity_id == entity.id)
                .order_by(
                    EntityEvidenceRecord.chunk_id,
                    EntityEvidenceRecord.char_start,
                    EntityEvidenceRecord.id,
                )
            )
        )
        neighbors = session.execute(
            select(
                RelationRecord.id,
                RelationRecord.source_entity_id,
                RelationRecord.target_entity_id,
                RelationRecord.predicate,
            )
            .where(
                or_(
                    RelationRecord.source_entity_id == entity.id,
                    RelationRecord.target_entity_id == entity.id,
                )
            )
            .order_by(RelationRecord.id)
        ).all()
        return {
            "kind": "entity",
            "id": entity.id,
            "build_run_id": entity.build_run_id,
            "graph_config_hash": entity.graph_config_hash,
            "canonical_name": entity.canonical_name,
            "normalized_name": entity.normalized_name,
            "entity_type": entity.entity_type,
            "description": entity.description,
            "aliases": entity.aliases_json,
            "source_chunk_ids": entity.source_chunk_ids_json,
            "evidence": [
                {
                    "id": item.id,
                    "chunk_id": item.chunk_id,
                    "extraction_id": item.extraction_id,
                    "mention_id": item.mention_id,
                    "quote": item.quote,
                    "char_start": item.char_start,
                    "char_end": item.char_end,
                }
                for item in evidence
            ],
            "neighbors": [
                {
                    "relation_id": row.id,
                    "source_entity_id": row.source_entity_id,
                    "target_entity_id": row.target_entity_id,
                    "predicate": row.predicate,
                }
                for row in neighbors
            ],
        }

    @staticmethod
    def _relation_inspect(session: Session, relation: RelationRecord) -> dict[str, Any]:
        evidence = list(
            session.scalars(
                select(RelationEvidenceRecord)
                .where(RelationEvidenceRecord.relation_id == relation.id)
                .order_by(
                    RelationEvidenceRecord.chunk_id,
                    RelationEvidenceRecord.char_start,
                    RelationEvidenceRecord.id,
                )
            )
        )
        return {
            "kind": "relation",
            "id": relation.id,
            "build_run_id": relation.build_run_id,
            "graph_config_hash": relation.graph_config_hash,
            "source_entity_id": relation.source_entity_id,
            "target_entity_id": relation.target_entity_id,
            "predicate": relation.predicate,
            "description": relation.description,
            "source_chunk_ids": relation.source_chunk_ids_json,
            "evidence": [
                {
                    "id": item.id,
                    "chunk_id": item.chunk_id,
                    "extraction_id": item.extraction_id,
                    "mention_id": item.mention_id,
                    "quote": item.quote,
                    "char_start": item.char_start,
                    "char_end": item.char_end,
                }
                for item in evidence
            ],
        }

    @staticmethod
    def _current_snapshot_run(session: Session, run_id: str | None) -> GraphBuildRunRecord | None:
        if run_id is not None:
            return session.get(GraphBuildRunRecord, run_id)
        current_run_id = session.scalar(select(EntityRecord.build_run_id).limit(1))
        if current_run_id is not None:
            return session.get(GraphBuildRunRecord, current_run_id)
        return session.scalar(
            select(GraphBuildRunRecord)
            .order_by(GraphBuildRunRecord.started_at.desc(), GraphBuildRunRecord.id.desc())
            .limit(1)
        )
