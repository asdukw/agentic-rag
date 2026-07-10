from pathlib import Path

from hybrid_rag.corpus import load_manifest


def test_public_corpus_manifest_is_versioned_and_unique() -> None:
    manifest_path = Path(__file__).parents[2] / "data" / "corpus.json"

    manifest = load_manifest(manifest_path)

    assert len(manifest.papers) == 10
    assert len({paper.arxiv_id for paper in manifest.papers}) == 10
    assert all("v" in paper.arxiv_id for paper in manifest.papers)
    assert all(len(paper.sha256) == 64 for paper in manifest.papers)
