# Phase 4 evaluation report template

> Status: **template; no benchmark results are asserted in this document.**
> Fill a row only after saving the corpus manifest, question set, index profile,
> retrieval traces, model metadata, and run logs needed to reproduce it.

## 1. Evaluation objective

The primary comparison is whether `hybrid` improves evidence retrieval and
answer quality over `naive` on a fixed research-paper corpus, while disclosing
the latency and cost trade-off.  `local` and `global` are diagnostic routes:
they help explain where a hybrid result came from, but should not be tuned on
the held-out scoring set after the comparison begins.

This report separates two forms of evidence:

| Evidence level | What it can establish | What it cannot establish |
| --- | --- | --- |
| Offline fixture run | Deterministic data flow, schema compatibility, trace/replay behavior, citation plumbing, and regressions in the test corpus. | Retrieval quality on real papers, real API latency/cost, generalization, or judge agreement. |
| Fixed real-paper benchmark | Comparative quality and operational measurements for the stated corpus, model versions, configurations, and date. | A universal ranking across corpora, future model versions, or unmeasured production traffic. |

The current project has offline fixture coverage.  It must not be reported as a
quality benchmark, and an empty table below means “not run”, not zero or a
negative result.

## 2. Pre-run record

Complete this section before looking at comparative scores.

| Field | Value |
| --- | --- |
| Evaluation date and timezone | `NOT RUN` |
| Git commit | `NOT RUN` |
| Python / dependency lock identity | `NOT RUN` |
| Corpus manifest path and SHA-256 | `NOT RUN` |
| Graph-independent document/chunk corpus-content hash | `NOT RUN` |
| Graph-bound index source snapshot hash | `NOT RUN` |
| Documents, chunks, entities, relations | `NOT RUN` |
| Chunker/tokenizer configuration hash | `NOT RUN` |
| Graph build run ID and extraction model/settings | `NOT RUN` |
| Embedding profile ID, provider, model, dimension, text-schema hash | `NOT RUN` |
| Naive dense/BM25 weights, BM25 tokenizer version, k1 and b | `NOT RUN` |
| Retrieval options per mode | `NOT RUN` |
| Question-set version and SHA-256 | `NOT RUN` |
| Evaluation random seed | `NOT RUN` |
| Code/config changes allowed after freeze | `NOT RUN` |

Keep raw manifests, CLI JSON output, and `rtr_` trace IDs outside the report
table.  The report should reference their immutable paths or artifact hashes.

## 3. Corpus and question protocol

### Corpus

Use a versioned manifest of allowed-to-distribute fixture documents or a
separately downloadable real-paper set.  Record title, stable identifier/URL,
download date, file SHA-256, parser version, and any exclusions (for example,
scanned PDFs or unreadable tables).  Do not commit large PDFs, SQLite files,
vector indexes, or model caches to the repository.

The benchmark must declare the graph-independent corpus-content hash derived
from the frozen documents/chunks.  Fail the run when it differs from the pinned
profile.  Record the graph-bound source snapshot hash and graph build run beside
it: a graph rebuild may be a legitimate comparison condition, but it must never
silently change which vectors are evaluated.

For the real benchmark, freeze the source documents before authoring the final
questions.  Rebuilding a graph or index is allowed only when its new profile
and configuration are recorded; do not mix results from different snapshots in
one aggregate.

### Question set

Prepare 20--30 questions across these predefined strata:

| Stratum | Target count | Required annotation |
| --- | ---: | --- |
| Fact | `NOT SET` | Short expected fact and at least one gold evidence span/chunk. |
| Comparison | `NOT SET` | Entities or methods compared, expected distinction, and gold evidence. |
| Relation | `NOT SET` | Expected source--predicate--target claim and evidence. |
| Cross-document synthesis | `NOT SET` | Required documents, synthesis criterion, and evidence for each component. |

Give each item a stable ID, a question, a type, gold source chunk IDs or
passage anchors, and a concise answer rubric.  A second annotator should review
the gold evidence for a stratified subset; record disagreements and resolution
rules.  Keep development questions separate from held-out evaluation questions
when tuning fusion weights, graph hops, context budget, prompts, or embedding
provider.

## 4. Run matrix

Use the same frozen corpus, pinned profile ID, top-k, token budget, and question
wording for all rows unless the row explicitly studies a parameter change. Do
not resolve the active profile anew for each question.

| Run ID | Mode | Keyword source | Answer source | Profile | Context budget | Cache condition | Status |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| `NOT RUN` | naive | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | cold / warm | pending |
| `NOT RUN` | local | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | cold / warm | pending |
| `NOT RUN` | global | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | cold / warm | pending |
| `NOT RUN` | hybrid | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | cold / warm | pending |

Run every question in every selected mode.  Save the JSON result and replayable trace ID
for each invocation.  Report failures, retries, missing indexes, and
out-of-budget answers rather than dropping them from denominators.

## 5. Metrics and calculation rules

### Retrieval and citation metrics

| Metric | Unit / formula | Reporting rule |
| --- | --- | --- |
| Evidence hit rate@k | Questions with at least one gold evidence chunk in the final cited context divided by eligible questions. | State `k`, whether the final post-budget context or pre-budget candidates were used, and the denominator. |
| Evidence recall@k | Gold evidence chunks present in final context divided by annotated gold chunks. | Use when items can have multiple gold chunks. |
| Citation precision | Returned citation IDs judged to support the answer divided by returned citation IDs. | Count unsupported or nonexistent citations as errors. |
| Citation validity | Citations that are in the retrieval evidence whitelist divided by answer citations. | The constrained client should make this 100%; report any violation as a defect, not a quality win. |
| Graph-path usefulness | Per-question reviewer label for whether a returned path materially supports the answer. | Report sample size and rubric; do not infer it merely from path existence. |

### Answer metrics

| Metric | Unit / formula | Reporting rule |
| --- | --- | --- |
| Faithfulness | Answers fully supported by cited context divided by answered questions. | Judge only supplied evidence, not outside knowledge.  Include abstentions in a separate rate. |
| Task correctness | Answers satisfying the frozen question rubric divided by questions. | Report by question stratum as well as overall. |
| Pairwise answer win rate | Wins / (wins + losses), with ties reported separately, for hybrid versus naive. | Do not fold ties into wins without stating the rule. |
| Citation-complete answer rate | Answers that cover every required claim with appropriate evidence divided by eligible questions. | Particularly important for cross-document synthesis. |

### Operational metrics

| Metric | Unit / formula | Reporting rule |
| --- | --- | --- |
| Retrieval latency | Wall-clock milliseconds from request start through trace persistence. | Report p50, p95, min/max, number of samples, hardware, and cache condition. |
| Answer latency | Wall-clock milliseconds after evidence selection through answer completion. | Separate from retrieval latency and identify provider calls. |
| Index build time | Total and per partition (chunk/entity/relation). | State corpus size, vector count, and whether an index was rebuilt or reused. |
| Provider cost | Actual input/output token counts and billed cost, by provider/model. | State currency, pricing-source date, retries, failures, and whether values are billed or estimates. |

Do not calculate a single blended “quality score” unless the weighting is fixed
before results are inspected.  Present confidence intervals or paired
per-question differences where the sample size supports it; with 20--30 items,
prefer transparent counts and uncertainty over strong significance claims.

## 6. Blind comparison and judge protocol

1. Produce paired answers for each question from the frozen `naive` and
   `hybrid` configurations.
2. Render them with identical formatting.  Hide route name, score, trace ID,
   and ordering-dependent metadata; keep only the answer and the evidence
   needed for the stated rubric.
3. Randomly assign the two outputs to labels `A` and `B` independently for each
   question.  Save the mapping and the random seed before judging; do not reveal
   it to judges.
4. Randomize question order independently of A/B assignment.  If the same
   judge evaluates both orders, include a reverse-order subset to detect
   position bias.
5. Ask for `A win`, `B win`, `tie`, or `both unsupported`, plus a short rubric
   reason.  Resolve only predeclared disagreement cases, and retain original
   labels and annotations.
6. Unblind only after annotations are frozen.  Report ties, abstentions,
   disagreements, and excluded cases.

`deepseek-v4-pro` is a proposed judge, not an established ground truth.  If it
is used, record endpoint, exact model identifier, date, system prompt, sampling
parameters, thinking setting, retry policy, and full judge rubric.  The model
must evaluate only the supplied answer/evidence/rubric; it must not retrieve
additional material.  Because the extraction or answer model may be from the
same provider, disclose same-provider self-evaluation bias and do not use one
model judge as the only decision source.  Include blinded human review of a
predeclared stratified sample and, where feasible, an independent judge or
adjudicator.

## 7. Latency and cost disclosure

For each mode, run and report cold-cache and warm-cache conditions separately.
Use a monotonic clock around these stages:

```text
keyword extraction -> query embedding -> route retrieval/fusion -> context crop
                    -> optional answer generation -> trace persistence
```

Record request count, retry count, errors, input/output tokens, and provider
response metadata without storing secrets.  If an endpoint does not return
usage or a rate is unavailable, write `unavailable`; never invent a cost from a
latency measurement.  If cost is estimated from public pricing, label it
`estimated`, cite the pricing date/source in the run artifact, and state which
tokens and retries were included.

Fixture-only tests use deterministic hash embeddings and deterministic/offline
query behavior.  They may report local execution time for regression tracking,
but they must not be presented as external embedding or DeepSeek latency/cost.

## 8. Results table (leave unfilled until a run completes)

| Mode | Questions | Evidence hit rate@k | Faithfulness | Correctness | Hybrid-vs-naive wins/losses/ties | Retrieval p50/p95 | Answer p50/p95 | Actual provider cost | Notes |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| naive | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | n/a | `NOT RUN` | `NOT RUN` | `NOT RUN` |  |
| local | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | n/a | `NOT RUN` | `NOT RUN` | `NOT RUN` |  |
| global | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | n/a | `NOT RUN` | `NOT RUN` | `NOT RUN` |  |
| hybrid | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` |  |

### Interpretation (complete after unblinding)

- **Retrieval finding:** `NOT RUN`
- **Answer-quality finding:** `NOT RUN`
- **Latency/cost trade-off:** `NOT RUN`
- **Known failures and exclusions:** `NOT RUN`
- **Decision for the next iteration:** `NOT RUN`

## 9. Publication checklist

- [ ] Corpus and question manifests are versioned and hash-identified.
- [ ] Pinned index profile, corpus-content hash, graph-bound snapshot, source graph run, tokenizer, and retrieval configuration are disclosed.
- [ ] All modes use the same frozen question set and stated cache condition.
- [ ] Raw JSON outputs and retrieval traces are archived with their run IDs.
- [ ] Blind A/B mapping, seed, rubric, and unblinded annotations are retained.
- [ ] Provider model/version/date/settings and cost source are disclosed.
- [ ] Fixture verification is labeled separately from real-corpus results.
- [ ] Same-provider judge bias, judge disagreements, and human audit limits are disclosed.
