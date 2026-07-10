from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from hybrid_rag.ids import canonical_json_hash, file_source_uri, sha256_file, stable_id
from hybrid_rag.ingest.chunker import SectionTokenChunker
from hybrid_rag.ingest.cleaner import CLEANER_NAME, CLEANER_VERSION, clean_document
from hybrid_rag.ingest.loaders import LoaderRegistry
from hybrid_rag.schemas import FileFailure, IngestReport
from hybrid_rag.storage.database import Database
from hybrid_rag.storage.repository import IngestRepository


class IngestionService:
    def __init__(
        self,
        database: Database,
        chunker: SectionTokenChunker,
        loaders: LoaderRegistry | None = None,
        repository: IngestRepository | None = None,
    ) -> None:
        self.database = database
        self.chunker = chunker
        self.loaders = loaders or LoaderRegistry()
        self.repository = repository or IngestRepository()

    @property
    def config_hash(self) -> str:
        return canonical_json_hash(
            {
                "cleaner": {"name": CLEANER_NAME, "version": CLEANER_VERSION},
                "chunker": self.chunker.config,
            }
        )

    def ingest(self, source: Path) -> IngestReport:
        source = source.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(source)

        root, files = self._discover(source)
        started_at = datetime.now(UTC)
        counters = {"inserted": 0, "updated": 0, "skipped": 0}
        chunks_written = 0
        failures: list[FileFailure] = []

        for path in files:
            try:
                source_uri = file_source_uri(path, root)
                loader = self.loaders.for_path(path)
                document_id = stable_id("doc", source_uri)
                content_hash = sha256_file(path)
                with self.database.session_factory() as session:
                    if self.repository.is_current(
                        session,
                        document_id,
                        content_hash,
                        loader.parser_name,
                        loader.parser_version,
                        self.config_hash,
                    ):
                        counters["skipped"] += 1
                        continue

                document = clean_document(loader.load(path, source_uri))
                if not document.text:
                    raise ValueError("document contains no extractable text")
                chunks = self.chunker.split(document)
                if not chunks:
                    raise ValueError("document produced no chunks")
                with self.database.session_factory.begin() as session:
                    result = self.repository.upsert_document(
                        session, document, chunks, self.config_hash
                    )
                counters[result.status] += 1
                chunks_written += result.chunks_written
            except Exception as error:
                failures.append(
                    FileFailure(
                        path=str(path),
                        error_type=type(error).__name__,
                        message=str(error),
                    )
                )

        finished_at = datetime.now(UTC)
        report = IngestReport(
            run_id=f"run_{uuid4().hex}",
            source_path=str(source),
            config_hash=self.config_hash,
            started_at=started_at,
            finished_at=finished_at,
            discovered=len(files),
            inserted=counters["inserted"],
            updated=counters["updated"],
            skipped=counters["skipped"],
            failed=len(failures),
            chunks_written=chunks_written,
            failures=failures,
        )
        with self.database.session_factory.begin() as session:
            self.repository.record_run(session, report)
        return report

    def _discover(self, source: Path) -> tuple[Path, list[Path]]:
        if source.is_file():
            self.loaders.for_path(source)
            return source.parent, [source]
        files = sorted(
            (
                path
                for path in source.rglob("*")
                if path.is_file() and path.suffix.lower() in self.loaders.supported_suffixes
            ),
            key=lambda path: path.as_posix().casefold(),
        )
        return source, files
