"""Generate a small, cost-bounded Ragas test set for pipeline smoke checks."""

from ragas_testset import ROOT, ScriptDefaults, main

SMOKE_DEFAULTS = ScriptDefaults(
    output=ROOT / "artifacts" / "ragas" / "ragas-smoke-testset.json",
    testset_size=6,
    max_documents=2,
    max_segments_per_document=6,
    description=__doc__,
)


if __name__ == "__main__":
    main(SMOKE_DEFAULTS)
