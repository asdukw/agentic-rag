from __future__ import annotations

import time
from pathlib import Path
from shutil import copyfileobj
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field, HttpUrl, field_validator

from hybrid_rag.ids import sha256_file


class CorpusPaper(BaseModel):
    arxiv_id: str
    title: str
    abs_url: HttpUrl
    pdf_url: HttpUrl
    sha256: str

    @field_validator("arxiv_id")
    @classmethod
    def validate_arxiv_id(cls, value: str) -> str:
        allowed = set("0123456789.v")
        if not value or any(character not in allowed for character in value):
            raise ValueError("arxiv_id must be a versioned numeric identifier")
        if "v" not in value:
            raise ValueError("arxiv_id must pin an arXiv version")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.casefold()
        invalid_character = any(character not in "0123456789abcdef" for character in normalized)
        if len(normalized) != 64 or invalid_character:
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
        return normalized


class CorpusManifest(BaseModel):
    name: str
    papers: list[CorpusPaper] = Field(min_length=1)

    @field_validator("papers")
    @classmethod
    def unique_papers(cls, papers: list[CorpusPaper]) -> list[CorpusPaper]:
        ids = [paper.arxiv_id for paper in papers]
        if len(ids) != len(set(ids)):
            raise ValueError("manifest contains duplicate arXiv IDs")
        return papers


class DownloadResult(BaseModel):
    arxiv_id: str
    path: str
    status: str
    size_bytes: int = 0
    sha256: str | None = None
    error: str | None = None


class DownloadReport(BaseModel):
    manifest: str
    destination: str
    results: list[DownloadResult]

    @property
    def downloaded(self) -> int:
        return sum(result.status == "downloaded" for result in self.results)

    @property
    def skipped(self) -> int:
        return sum(result.status == "skipped" for result in self.results)

    @property
    def failed(self) -> int:
        return sum(result.status == "failed" for result in self.results)


def load_manifest(path: Path) -> CorpusManifest:
    return CorpusManifest.model_validate_json(path.read_text(encoding="utf-8"))


def download_manifest(
    manifest_path: Path,
    destination: Path,
    delay_seconds: float = 3.0,
    timeout_seconds: float = 90.0,
) -> DownloadReport:
    manifest_path = manifest_path.expanduser().resolve()
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    results: list[DownloadResult] = []

    for index, paper in enumerate(manifest.papers):
        target = destination / f"{paper.arxiv_id}.pdf"
        existing_digest = sha256_file(target) if _is_pdf(target) else None
        if existing_digest == paper.sha256:
            results.append(
                DownloadResult(
                    arxiv_id=paper.arxiv_id,
                    path=str(target),
                    status="skipped",
                    size_bytes=target.stat().st_size,
                    sha256=existing_digest,
                )
            )
            continue

        temporary = target.with_suffix(".pdf.part")
        try:
            request = Request(
                str(paper.pdf_url),
                headers={
                    "User-Agent": "hybrid-rag/0.1 (portfolio research project)",
                    "Accept": "application/pdf",
                },
            )
            with (
                urlopen(request, timeout=timeout_seconds) as response,
                temporary.open("wb") as output,
            ):
                copyfileobj(response, output)
            if not _is_pdf(temporary):
                raise ValueError("downloaded response is not a PDF")
            digest = sha256_file(temporary)
            if digest != paper.sha256:
                raise ValueError(f"SHA-256 mismatch: expected {paper.sha256}, downloaded {digest}")
            temporary.replace(target)
            results.append(
                DownloadResult(
                    arxiv_id=paper.arxiv_id,
                    path=str(target),
                    status="downloaded",
                    size_bytes=target.stat().st_size,
                    sha256=digest,
                )
            )
        except Exception as error:
            temporary.unlink(missing_ok=True)
            results.append(
                DownloadResult(
                    arxiv_id=paper.arxiv_id,
                    path=str(target),
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                )
            )
        if index < len(manifest.papers) - 1 and delay_seconds > 0:
            time.sleep(delay_seconds)

    return DownloadReport(
        manifest=str(manifest_path),
        destination=str(destination),
        results=results,
    )


def _is_pdf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 5:
        return False
    with path.open("rb") as stream:
        return stream.read(5) == b"%PDF-"
