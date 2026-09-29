# Default CorkScrewDB Embedder Plan

This plan is scoped to `eos-embed-v1`, the small sealed local default embedder candidate for CorkScrewDB, with reproducible local training, retrieval scoring, and TurboQuant-first serving gates. The shipping alias is `corkscrewdb-default-embedder` once the promotion gates pass. This plan does not claim state-of-the-art quality or superiority over hosted/open embedding models until scored rows exist in the baseline matrix.

Imported BGE (`BAAI/bge-small-en-v1.5`) is a distinct non-default candidate,
not a replacement for the current shipping alias. It can become first-class
only after the exact selected-package full gate passes with package SHA256
`841b0d851c06290daeeab4bf4d25cb1dd7bb87920316dac950e1b556a3bae763`,
identity SHA256
`a356a4b7dc29a8d0f0a7b7bd45e7a9d2afbfa651c1a5bfaa05008c7157ba9637`,
the preserved BGE query/document role contract, `quality_claim=false`, and
`default_alias_changed=false`. Its notice/provenance packet lives in the
top-level `NOTICE`; do not make it the `corkscrewdb-default-embedder` alias
without a separate default-swap migration and release decision.

## Current Promoted Dense Default

- Artifact: `runs/eos-s40-nfcorpus-compact-mined-narrow-candidate-v1-20260623T032556Z/candidate/eos-embed-v1.sealed.mll`
- Durable asset: `assets/corkscrewdb-default-embedder/corkscrewdb-default-embedder.mll`
- SHA256: `e0eca16ff34ebb88ca96862d58c3ac7f02dbf4b124599fdf96f25344ac02e408`
- Training data: `runs/eos-s40-nfcorpus-compact-mined-narrow-candidate-v1-20260623T032556Z/data/train-anchor-plus-nfcorpus-compact-mined-64.jsonl`; 102 protected anchor rows plus 64 NFCorpus train-mined compact hard-negative rows; no NFCorpus test rows used for training or selection; LR `0.000000025`; HN `5`; compact objective `256:4=0.05`; prefix seed `5581486560434873699`; dev-mined hard eval only.
- Release package SHA256: `188265db16992ab24be15e678c5f7e175bebad769e8d844e8b0f50ffc23bd5bf`; tokenizer SHA256: `64cf63223cb3f97125040677a573e6ab6c625cff1f6f338f4e680a4c9f7a42f5`.

Dense short-set rows:

| Dataset | nDCG@10 | recall@100 | Delta vs s40 nDCG@10 | Delta vs s40 recall@100 |
| --- | ---: | ---: | ---: | ---: |
| SciFact | 0.5645379155 | 0.7964444444 | +0.0000000000 | +0.0000000000 |
| NFCorpus | 0.205745967860765 | 0.242066067459883 | +0.000174802593872 vs s40 | +0.000007371369600 vs s40 |
| FiQA | 0.121260940614285 | 0.351678208622653 | +0.000000000000000 vs s40 | +0.000000000000000 vs s40 |

Treat the s40 NFCorpus compact-mined narrow artifact as the current promoted default for `eos-embed-v1` on the measured short retrieval set and q4/fp16/o200 compact policy, not as broad robustness or hosted-model parity. The durable in-repo asset is `assets/corkscrewdb-default-embedder/corkscrewdb-default-embedder.mll`, with tokenizer compatibility sidecar `assets/corkscrewdb-default-embedder/corkscrewdb-default-embedder.tokenizer.mll`; the ignored `runs/eos-s40-nfcorpus-compact-mined-narrow-candidate-v1-20260623T032556Z/` directory remains provenance. The dense gate passed all 6 checks against s40, with macro nDCG@10 `+0.000058267531291` and macro recall@100 `+0.000002457123200`. Package and sealed inspection report `package verify: OK`; the previous s40, nf005, targeted-v3, and June 10 deephard-full artifacts remain predecessor comparison provenance.

Release validation for the promoted narrow default is repeatable with `ferrous-wheel run scripts/smoke_eos_default_embedder_release.fw`, which now expects durable asset SHA256 `e0eca16ff34ebb88ca96862d58c3ac7f02dbf4b124599fdf96f25344ac02e408`, release package SHA256 `188265db16992ab24be15e678c5f7e175bebad769e8d844e8b0f50ffc23bd5bf`, and tokenizer SHA256 `64cf63223cb3f97125040677a573e6ab6c625cff1f6f338f4e680a4c9f7a42f5`. This is release packaging evidence only unless the optional CorkScrewDB consuming-startup root is configured.

Current CorkScrewDB local-flat packed-parent API evidence is refreshed for the promoted narrow default in `.tiller/scratch/codex/eos-narrow-default-corkscrewdb-api-refresh-v1-report.md`. Fresh SciFact child vectors live at `runs/eos-vector-caches/eos-narrow-default-scifact-child-w128-o32-128d` with `5,183` documents, `12,468` child vectors, `300` queries, `128d`, and CUDA backend. Packed-parent run `runs/eos-narrow-default-corkscrewdb-budget-quality-packed-q4q8-main-20260623T050216Z` used layout `packed_parent_multivectors`, metadata `none`, ordinal child IDs, `quantized_only` storage, and flat exact parent search against CorkScrewDB repo `$HOME/work/corkscrewdb`. The q4 release row records nDCG@10 `0.452380`, recall@100 `0.758556`, DB directory multiple `0.041676x`, vector payload multiple `0.013312x`, p95 `16.076127ms`, planner fit `180`, target fit `true`, and target storage multiple `0.554545x`. q8 remains diagnostic: nDCG@10 `0.472132`, recall@100 `0.773556`, DB directory multiple `0.066734x`, vector payload multiple `0.025841x`, p95 `36.113081ms`, planner fit `93`, target fit `false`, and target storage multiple `1.074026x`. This is local flat packed-parent evidence only; HNSW, time-series, remote mode, federation, hosted parity, and service SLO claims remain out of scope.

## Scoreboard Promotion Gate

Default embedder candidates must pass the scoreboard gate against the accepted dense candidate before promotion. The gate compares every selected dataset and metric independently; macro gains are reported for context but do not hide a per-dataset miss. The historical targeted-v3 acceptance used both the June 10 strict sealed anchor and the v2 candidate as zero-tolerance comparisons. The nf005 default was accepted by passing the selected current-release gate against targeted-v3 with all 6 checks passing; s40 was accepted by passing the dense gate against nf005; the current narrow default was accepted by passing the dense gate against s40. Future dense candidates should compare against the narrow default scoreboard.

```bash
go run ./cmd/eos gate-scoreboard \
  --category short_retrieval \
  --baseline eos \
  --datasets scifact,nfcorpus,fiqa \
  --metrics ndcg_at_10,recall_at_100 \
  runs/<candidate-scoreboard>/scoreboard.json \
  runs/<accepted-dense-scoreboard>/scoreboard.json
```

Use `--tolerance` only for an explicitly accepted numeric rounding margin. For TurboQuant rows, add the matching `--baseline`, `--method`, and `--bits` filters so the command compares one unambiguous row per dataset in both scoreboards. The current promoted compact policy is `q4/fp16/rerank-overfetch=200`, method `turboquant_ip_b4_overfetch200_fp16_rerank`, bits `4`, fp16 rerank storage, total compression `1.5900621118x`. The narrow default passed strict seeded compact non-regression against the s40 q4/fp16/o200 anchor: SciFact unchanged; NFCorpus nDCG@10 `+0.000174802593872`, recall@100 `+0.000006604583581`; FiQA nDCG@10 `+0.000180234154897`, recall@100 `+0.000771604938272`; macro nDCG@10 `+0.000118345582923`, recall@100 `+0.000259403173951`. The s40 serving smoke remains predecessor proxy evidence until refreshed for this default.
`--baseline eos` falls back to legacy `manta` rows when exact `eos` rows are absent, so the gate can compare new scoreboards with the current legacy-named anchor without rewriting provenance.

Hybrid retrieval rows are eligible only as calibrated retrieval-surface evidence. Run `scripts/calibrate_eos_embed_hybrid_retrieval.fw` first, select method/alpha/RRF settings on the configured dev split, and apply the selected setting unchanged to test. The calibration summary must include dense and BM25 sanity rows plus the protection gate deltas against dense `ndcg_at_10` and `recall_at_100`; use that selected setting in any later `eos-hybrid` scoreboard row. Hybrid per-query rows expose optional dense/BM25 component ranks plus raw and normalized component scores for fused candidates, so routing experiments should prefer those diagnostics over query-ID allowlists. `--dense-protect-top-k N` is the narrow product guard for preserving dense winners without query-ID special cases; report it separately from the selected fusion setting. Do not treat a passing hybrid calibration as a dense model promotion.

BM25-dot CorkScrewDB API evidence now exists for the promoted narrow default local flat hybrid retrieval-policy lane via `scripts/smoke_eos_corkscrewdb_hybrid_policy.fw`, with `EOS_CORKSCREW_HYBRID_SMOKE_SPARSE_METHOD=bm25_dot`. Full uncapped q4 BM25-compatible sparse-dot rows passed on SciFact, NFCorpus, and FiQA through CorkScrewDB `WithSparse`, `SearchMulti`, and public `WeightedFusion{Dense:0.5,Sparse:0.5}`. Hybrid nDCG@10 / recall@100: SciFact `0.723139732393 / 0.932888888889`, NFCorpus `0.316761336578 / 0.293902897482`, FiQA `0.224427168561 / 0.502074751380`. Dense-only rows for the same runs were SciFact `0.559571129519 / 0.796444444444`, NFCorpus `0.205279642844 / 0.242243795394`, and FiQA `0.119552386600 / 0.352919418544`.

s40-era hybrid policy candidate status: the previous durable default asset SHA256 `f494915a0d78b24205d5018bb701bf40cabbedee4bc8b96b6a1920b19131da5a` has command-level and full uncapped CorkScrewDB `SearchMulti` evidence in `runs/eos-s40-current-default-hybrid-policy-v1-20260621T103500Z/`. Command-level `eval-retrieval-hybrid --method minmax_blend --alpha 0.5 --top-k 100` passed SciFact `0.717644867485 / 0.932888888889`, NFCorpus `0.311170830256 / 0.290149896585`, and FiQA `0.219415915378 / 0.500671683426` for hybrid nDCG@10 / recall@100, with positive deltas over s40 dense on all six checks. Full uncapped BM25-dot `SearchMulti` smokes passed with SciFact `0.724378172602 / 0.932888888889`, NFCorpus `0.315090347196 / 0.293990495284`, and FiQA `0.223742712175 / 0.504002444743`, all `overall_pass=true`. Promotion boundary: this remains predecessor local lexical+dense retrieval-policy evidence, not dense model promotion.

For FiQA and mixed-objective iteration, use the reusable guarded candidate runner as the standard loop:

```bash
EOS_REPO_ROOT=$PWD \
EOS_GUARD_ANCHOR_SCOREBOARD=runs/manta-embed-v1-deephard-full-ft-20260610T0000Z-sealed-scoreboard/scoreboard.json \
ferrous-wheel run scripts/run_manta_embed_v1_guarded_candidate.fw
```

The runner composes the existing candidate training script, scoreboard script, and `eos gate-scoreboard`. It writes a guard manifest and summary under `runs/eos-embed-v1-guarded-<timestamp>/`, and rejects the candidate unless every selected dataset metric clears the anchor. Use `EOS_GUARD_FAIL_ON_GATE=0` only for smoke runs where the rejected manifest is the expected output.

## Baseline Matrix

Maintain a scoreboard with explicit rows for:

- Current sealed Eos anchor (`baseline=eos`; legacy anchor rows may still be `baseline=manta`).
- BM25 lexical baseline.
- BGE, Qwen, Jina, Voyage, Cohere, and OpenAI external vector caches.
- Any later local Eos candidate, including `eos-hybrid`, `eos-turboquant`, and `eos-turboquant-rerank` rows when those product surfaces are evaluated.

Rows without evaluated vectors stay marked `not_scored`. Do not replace missing rows with claims from model cards or unrelated public benchmarks. For external models, the provider boundary is a BEIR-aligned pair of caches:

- `doc-vectors.jsonl`
- `query-vectors.jsonl`

Each row needs `id` or `_id` plus one of `vector`, `embedding`, or `values`.

The first practical leading-family local baseline is Qwen3 0.6B:

```bash
python3 scripts/export_retrieval_vectors.py \
  --preset qwen3-0.6b \
  --dataset-dir datasets/eos-embed-v1/raw/scifact/scifact \
  --output-root runs/external-vector-caches/qwen3-0.6b \
  --dataset-name scifact \
  --batch-size 16
```

Mixedbread/mxbai can use the same provider-boundary lane:

```bash
python3 scripts/export_retrieval_vectors.py \
  --preset mxbai-large \
  --dataset-dir datasets/eos-embed-v1/raw/scifact/scifact \
  --output-root runs/external-vector-caches/mxbai-large \
  --dataset-name scifact \
  --batch-size 16
```

This script is a provider-boundary exporter. Its optional Python dependencies stay outside Eos, and its default output is `runs/external-vector-caches/<provider>/<dataset>/doc-vectors.jsonl` plus `query-vectors.jsonl`. The older `scripts/export_qwen3_retrieval_vectors.py` entry point remains available for compatibility and accepts `--model-name` for non-Qwen SentenceTransformers models.
The mxbai preset applies Mixedbread's retrieval query prompt and leaves document text unprefixed.

For parent-child multi-vector evidence, add document chunking flags. This keeps the query cache unchanged but writes `child-doc-vectors.jsonl` rows with `parent_id`, deterministic `child_id`, and `embedding`:

```bash
python3 scripts/export_retrieval_vectors.py \
  --preset qwen3-0.6b \
  --dataset-dir datasets/manta-embed-v1/raw/scifact/scifact \
  --output-root runs/external-vector-caches/qwen3-0.6b-scifact-child-w128-o32 \
  --dataset-name scifact \
  --batch-size 16 \
  --document-chunk-words 128 \
  --document-chunk-overlap 32 \
  --document-chunk-min-words 16
```

Current external comparison state: Qwen3 is locally consolidated for SciFact, NFCorpus, and full exportable-text FiQA. Its useful compact external row is direct q8 at about `3.98x` vector compression, with SciFact q8 nDCG@10 `0.704128`, NFCorpus q8 nDCG@10 `0.368763`, and FiQA q8 nDCG@10 `0.449614`. mxbai remains stronger than Qwen3 in the existing local short-set evidence. Qwen3 FiQA is not raw-row-complete or judged-coverage complete because one judged test document had empty unexportable text and was skipped.

## TurboQuant Gate

Every promotion candidate needs a dense reference and TurboQuant IP document-vector rows over the same vectors:

```bash
go run ./cmd/eos eval-retrieval-vectors \
  --dataset scifact \
  --backend qwen3 \
  --artifact Qwen/Qwen3-Embedding-0.6B \
  --doc-vectors runs/external-vector-caches/qwen3-0.6b/scifact/doc-vectors.jsonl \
  --query-vectors runs/external-vector-caches/qwen3-0.6b/scifact/query-vectors.jsonl \
  --metrics-json runs/scifact.qwen3.retrieval.metrics.json \
  datasets/eos-embed-v1/raw/scifact/scifact

go run ./cmd/eos eval-retrieval-vectors-turboquant \
  --dataset scifact \
  --backend qwen3 \
  --artifact Qwen/Qwen3-Embedding-0.6B \
  --doc-vectors runs/external-vector-caches/qwen3-0.6b/scifact/doc-vectors.jsonl \
  --query-vectors runs/external-vector-caches/qwen3-0.6b/scifact/query-vectors.jsonl \
  --bits 2,4,8 \
  --metrics-json runs/scifact.qwen3.turboquant.metrics.json \
  --metrics-tsv runs/scifact.qwen3.turboquant.metrics.tsv \
  datasets/eos-embed-v1/raw/scifact/scifact
```

The scoreboard harness appends the dense row and, when enabled, q2/q4/q8 rows with method, bit width, quality deltas, vector bytes, dense vector bytes, compression, docs/s, and scores/s:

```bash
EOS_SCOREBOARD_EXTERNAL_VECTOR_ROOT=runs/external-vector-caches/qwen3-0.6b \
EOS_SCOREBOARD_EXTERNAL_VECTOR_DATASETS=scifact \
EOS_SCOREBOARD_EXTERNAL_VECTOR_BASELINE=qwen3 \
EOS_SCOREBOARD_EXTERNAL_VECTOR_BACKEND=qwen3 \
EOS_SCOREBOARD_EXTERNAL_VECTOR_ARTIFACT=Qwen/Qwen3-Embedding-0.6B \
EOS_SCOREBOARD_EXTERNAL_VECTOR_TURBOQUANT=1 \
EOS_SCOREBOARD_EXTERNAL_VECTOR_TURBOQUANT_BITS=2,4,8 \
ferrous-wheel run scripts/score_manta_embed_v1_baselines.fw
```

Local sealed Eos candidates can emit the same product-surface rows without exporting an external vector cache:

```bash
EOS_SCOREBOARD_ARTIFACT=runs/<candidate>/eos-embed-v1.sealed.mll \
EOS_SCOREBOARD_RETRIEVAL_ROOT=datasets/eos-embed-v1 \
EOS_SCOREBOARD_RETRIEVAL_DATASETS=scifact,nfcorpus,fiqa \
EOS_SCOREBOARD_TURBOQUANT=1 \
EOS_SCOREBOARD_TURBOQUANT_BITS=4 \
EOS_SCOREBOARD_TURBOQUANT_RERANK_OVERFETCH=200 \
EOS_SCOREBOARD_TURBOQUANT_RERANK_STORAGE=fp16 \
EOS_SCOREBOARD_TURBOQUANT_BASELINE=eos-turboquant \
EOS_SCOREBOARD_TURBOQUANT_RERANK_BASELINE=eos-turboquant-rerank \
ferrous-wheel run scripts/score_manta_embed_v1_baselines.fw
```

This produces direct `eos-turboquant` rows and fp16 sidecar rerank `eos-turboquant-rerank` rows from one `eval-retrieval-turboquant` metrics file. Use `--baseline eos-turboquant --method turboquant_ip_b4 --bits 4` to inspect direct q4, but do not treat direct q4 or direct q8 as default-promotion candidates. Gate the promoted compact profile with `--baseline eos-turboquant-rerank --method turboquant_ip_b4_overfetch200_fp16_rerank --bits 4 --metrics ndcg_at_10,recall_at_100,total_compression_ratio` against the seeded compact anchor path above, not the legacy recovery-sweep scoreboard.

When the default embedder needs to feed the vector-cache evaluators instead of live `eval-retrieval`, use the Go-native exporter. It loads packaged or sealed Eos embedding artifacts through the runtime tokenizer/batch path and writes the same JSONL shape as external caches:

```bash
go run ./cmd/eos export-retrieval-vectors \
  --dataset scifact \
  --batch-size 64 \
  runs/default-embedder/eos-embed-v1.sealed.mll \
  data/beir/scifact \
  runs/eos-vector-caches/eos-embed-v1/scifact
```

For Eos-owned parent-child evidence, enable deterministic word chunking. This writes `child-doc-vectors.jsonl` rows with `parent_id`, `child_id`, and `embedding`, plus the unchanged query cache for `eval-retrieval-multivector-turboquant`:

```bash
go run ./cmd/eos export-retrieval-vectors \
  --dataset scifact \
  --batch-size 64 \
  --output-dim 128 \
  --document-chunk-words 128 \
  --document-chunk-overlap 32 \
  --document-chunk-min-words 16 \
  --manifest-json runs/eos-vector-caches/eos-embed-v1-scifact-child-w128-o32-128d/manifest.json \
  runs/default-embedder/eos-embed-v1.sealed.mll \
  data/beir/scifact \
  runs/eos-vector-caches/eos-embed-v1-scifact-child-w128-o32-128d/scifact
```

`--output-dim 128` writes a prefix-truncated compact child cache and L2-renormalizes the truncated vectors before writing. The manifest keeps `dimension` as the written vector dimension and records `model_dimension`/`output_dimension` so the truncation is auditable. Treat this as a measurement bridge for the compact-child thesis, not as a trained Matryoshka embedding or native 128d Eos head; the trained compact head remains the stronger future path.

Record, per dataset and candidate:

- Dense nDCG@10/nDCG@100, MRR@10, precision@1/5/10, hit@1/5/10, MAP@10/MAP@100, and recall@10/100.
- q2/q4/q8 quality deltas, especially nDCG@10 and recall@100 deltas against the dense row.
- Vector bytes, rerank storage, rerank sidecar bytes, total vector bytes, compression ratio, and total compression ratio.
- Quantization docs/s.
- Direct IP scores/s, per-query p50/p95/p99/max scoring latency, and rerank overfetch/rerank score counts when rerank rows are enabled.

Use the quality columns for different failure modes: nDCG and MAP judge ranked relevance, precision/hit@k judge first-screen success, and recall@100 judges candidate-pool coverage for reranking or multi-stage retrieval. Use vector bytes and compression ratio for index-footprint decisions, and throughput columns for the path they actually measure. External vector-cache dense rows measure cache load plus scoring, not live encoder throughput for Qwen, BGE, hosted APIs, or other providers.

The default bit width should be selected from measured q4/q8 rows. q2 is useful pressure testing but should not become default unless quality loss is explicitly acceptable for the target workload.

Use the local serving proxy before default-alias promotion:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_eos_default_embedder_serving.fw
```

The smoke selects `turboquant_ip_b4_overfetch200_fp16_rerank` as the promoted compact profile, writes a summary TSV and manifest under `runs/eos-default-embedder-serving-smoke-<timestamp>/`, and can gate total compression plus optional p95 latency. This is CorkScrewDB-relevant TurboQuant serving proxy evidence only, not a CorkScrewDB API load/index/search smoke. The s40 serving smoke in `runs/eos-default-embedder-serving-smoke-20260620T161633Z/` remains predecessor proxy evidence. For formal compact `gate-scoreboard` comparisons, compare future candidates against the current promoted compact scoreboard; the s40 compact scoreboard remains predecessor provenance. Use the local flat CorkScrewDB API smoke when the actual `PutVector`/`SearchVector` integration path needs to pass:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_corkscrewdb_child_vectors.fw
```

That smoke defaults to a tiny synthetic time-series child-vector cache, can consume externally prepared child/query/qrels paths with `EOS_CORKSCREW_SMOKE_CHILD_VECTORS`, `EOS_CORKSCREW_SMOKE_QUERY_VECTORS`, and `EOS_CORKSCREW_SMOKE_QRELS`, and requires a local CorkScrewDB checkout rather than silently pulling a network dependency. `EOS_CORKSCREW_SMOKE_OVERFETCH` accepts comma-separated values such as `100,12468`; the harness loads one DB per bit width and reuses it across overfetch values so serving recall/latency sweeps avoid repeated insert cost. Use exhaustive/full child overfetch when comparing against offline cache-evaluator parity, with the expectation that latency rises. Treat it as local flat CorkScrewDB load/index/search/storage accounting evidence, not remote/federation/HNSW evidence or a model-quality benchmark.

Current release-artifact local Eos TurboQuant result: q4/fp16 sidecar rerank at overfetch200 is the promoted compact policy for the narrow default. It passed strict seeded compact non-regression against the s40 q4/fp16/o200 anchor with macro nDCG@10 `+0.000118345582923` and macro recall@100 `+0.000259403173951`; the older nf005 q4/175 and q4/150 misses remain predecessor policy history.

## Multi-Vector Storage Planning

For the concise product thesis, exact claim boundaries, current evidence numbers, and next promotion gates for many compact child vectors per CorkScrewDB parent object, see [TurboQuant Multi-Vector Frontier](turboquant-multivector-frontier.md). This section remains the command-level operating reference.

The direct multi-vector lane is a storage/accounting thesis, not a retrieval-quality claim: one parent CorkScrewDB object can keep many quantized child vectors for windows, events, spans, or time-series observations while staying near the byte budget of one dense fp32 parent vector. The strict high-child-count local flat DB-directory claim now means `packed_child_id_mode=ordinal` with `packed_metadata_mode=none` or `minimal`; source child IDs and full metadata preserve richer product identity but require a different measured DB budget envelope. Measure the planner budget with:

```bash
go run ./cmd/eos plan-multivector-storage \
  --dim 128 \
  --baseline-dim 3072 \
  --bits 2,4,8 \
  --vectors-per-object 64,128,256,384 \
  --vector-overhead-bytes 32 \
  --packed-object-overhead-bytes 32 \
  --objects 1000
```

For time-series/window-vector planning, derive the child-vector count from series lengths:

```bash
go run ./cmd/eos plan-multivector-storage \
  --dim 128 \
  --baseline-dim 3072 \
  --bits 2,4,8 \
  --series-lengths 256,1024 \
  --window-size 64 \
  --window-stride 16 \
  --vector-overhead-bytes 32 \
  --packed-object-overhead-bytes 32 \
  --objects 1000
```

The planner uses one vector per covering window, including a tail window when needed. This is a storage gate for the TurboQuant time-series vector hypothesis, not a numeric time-series quality benchmark; run a retrieval or forecasting quality harness separately before making quality claims. Do not pass `--vectors-per-object` with `--series-lengths`; explicit conflicts fail so manual and derived modes remain separate.

The TSV/JSON rows report `baseline_dim`, `dense_parent_bytes`, `dense_baseline_bytes`, raw `quantized_vector_bytes`, `vector_overhead_bytes`, `dense_vector_storage_bytes`, `quantized_vector_storage_bytes`, `total_quantized_bytes`, packed-parent fields such as `packed_object_overhead_bytes`, `packed_quantized_storage_bytes`, `packed_total_quantized_bytes`, `packed_vectors_that_fit_in_one_dense_vector`, compression ratios, and optional time-series fields `series_length`, `window_size`, `window_stride`, and `derived_window_count`. When `--baseline-dim` is omitted or `0`, the dense budget is the same dimension as the child vector, preserving the same-dim interpretation: 128-dimensional q2 stores a child vector payload in 36 bytes, so 14 payload-only children fit inside one 512-byte fp32 vector budget; with 32 bytes of packed parent-object overhead, 14 q2, 7 q4, or 3 q8 child payloads fit. When modeling compact children against a larger baseline, pass `--dim 128 --baseline-dim 3072`; one dense baseline vector is 12,288 payload bytes, so 341 q2 payload-only children fit in that one-vector budget and 128 children cost about `0.375x` of it before object/index metadata. With the overhead-aware packed-parent accounting used by the smoke (`--packed-object-overhead-bytes 32`), one 3072d dense parent-vector storage budget fits 341 q2, 180 q4, or 93 q8 128d children. Current CorkScrewDB-style per-child-entry accounting uses `--vector-overhead-bytes`; packed-parent target accounting uses `--packed-object-overhead-bytes` to pay object overhead once per parent while each child remains a compact TurboQuant payload. Treat planner fit counts as byte-budget accounting and read them beside measured local flat packed-parent API rows. With CorkScrewDB compact snapshot v5 ordinal encoding, the q2-341 strict ordinal quantized-only packed-child shape persists at approximately one dense parent-vector budget; without that compact snapshot path, or for richer child records, keep DB directory cost separate.

The packed-parent sensitivity matrix in `runs/eos-packed-parent-storage-sensitivity-matrix-20260616T000000Z/` is the current release-boundary evidence for high-child-count DB directory size, measured against CorkScrewDB commit `c208f9b50d29f9fdf19771c4b093332c7c8fd0b4`. On q2/341 (`100` parents, `34,100` children), only `none`/`ordinal` (`1.007334x`, p95 `1.937486ms`) and `minimal`/`ordinal` (`1.009744x`, p95 `6.228571ms`) passed; source child IDs were about `2.74x` DB multiple and full metadata was `5.17x` to `5.92x`. On q4/100 (`100` parents, `10,000` children), only `none`/`ordinal` (`0.561698x`, p95 `6.849447ms`) and `minimal`/`ordinal` (`0.564106x`, p95 `9.298850ms`) passed; source child IDs were about `1.07x` and full metadata was `1.78x` to `2.01x`. The compact SciFact q4 rows stayed far below one DB budget across modes, but average only `2.405557` children per parent, so they are not high-child-count proof.

For an executable overhead-aware budget-frontier artifact, run:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_multivector_budget_frontier.fw
```

The smoke uses the planner as the source of truth and writes `summary.tsv`, `manifest.json`, planner JSON, and command logs under `runs/eos-multivector-budget-frontier-smoke-<timestamp>/`. Defaults model `128d` compact children against a `3072d` dense baseline, q2/q4/q8, child counts `1,16,64,100,128,181,256,341`, `32` bytes per current stored vector, `32` bytes per packed parent object, no sidecar, and `1000` objects. Set `EOS_MV_BUDGET_SMOKE_BASELINE_DIMS=128,384,768,1024,1536,3072` to measure the same compact-child shape against multiple dense parent dimensions in one run; this comma-list takes precedence over the backward-compatible `EOS_MV_BUDGET_SMOKE_BASELINE_DIM`. The default gates are current q2 >= `181`, current q4 >= `100`, current q8 >= `64`, packed q2 >= `341`, packed q4 >= `180`, and packed q8 >= `93` children fitting in one dense-vector budget. A deterministic parent-budget frontier run should use `EOS_MV_BUDGET_SMOKE_VECTORS_PER_OBJECT=64,100,128,180,256,341` with the baseline list above to document the precise claim: same-dimension children fit only single-digit to low-tens counts, while a 3072d dense-parent budget fits hundreds of packed q2/q4 child vectors and tens of q8 child vectors. Read this beside the time-series smoke and local CorkScrewDB API smoke: it proves byte accounting only, not retrieval quality, remote serving, HNSW, federation, API latency, or measured DB-directory cost.

Use the use-case frontier smoke when the claim needs named product shapes rather than a dimension grid:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_multivector_usecase_frontier.fw
```

This smoke still uses `plan-multivector-storage` as the source of truth, but emits planner-backed rows for representative usages: a same-dim `128d` baseline control with `100` children, document spans with `100` children against a `3072d` dense parent-vector budget, time-series windows derived from `series_length=1648`, `window_size=64`, `window_stride=16` into exactly `100` windows, an event/trace timeline with `180` children, and a q2 frontier at `341` children. It writes `summary.tsv`, `manifest.json`, planner JSON, and logs under `runs/eos-multivector-usecase-frontier-smoke-<timestamp>/`. The default gates assert the narrow thesis: same-dimension 100-child packed q2/q4 does not fit, packed q2/q4 100-child document and time-series profiles do fit under a 3072d dense baseline, packed q4 fits `180` children at the edge, and packed q2 fits `341` children. q8 100-child rows are reported but intentionally not gated to fit.

For the executable bridge from byte accounting to cache-only quality, run:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_multivector_budget_quality.fw
```

The default run uses `runs/eos-128d-child-cache-quality-20260615T000000Z/scifact/child-doc-vectors.jsonl`, matching query vectors, and `datasets/manta-embed-v1/raw/scifact/scifact`. It evaluates dense/q2/q4/q8 parent-child retrieval, infers the `128d` child dimension from dense child-vector bytes, then plans q2/q4/q8 for the actual parent count with child counts including `1,64,100,128,181,256,341`, `--baseline-dim 3072`, `--vector-overhead-bytes 32`, and `--sidecar-storage none`. S40-era predecessor interpretation: q4 is near dense on the 128d SciFact child cache (`ndcg@10` drop about `0.002630`, `recall@100` drop about `0.001667`) while the overhead-aware planner fits `123` q4 children in one 3072d dense-vector budget, so the `100` child-vector target fits. Treat this as Eos cache evidence only: the 128d cache is a prefix-truncated bridge, not a native Matryoshka artifact, and the planner overhead is an explicit input rather than a measured CorkScrewDB directory or API result. Refresh before using it as new-default evidence.

For the actual local flat CorkScrewDB API budget-quality check on the same compact cache, run:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_corkscrewdb_budget_quality.fw
```

The wrapper runs `scripts/smoke_corkscrewdb_child_vectors.fw` against the Eos 128d SciFact child/query cache and test qrels with q4 overfetch `100,12468`, then joins the resulting local `PutVector`/`SearchVector` metrics to the overhead-aware planner row for `--dim 128 --baseline-dim 3072 --vector-overhead-bytes 32 --sidecar-storage none`. It writes `summary.tsv`, `manifest.json`, command logs, nested `corkscrewdb/` artifacts, and `planner.json` under `runs/eos-corkscrewdb-budget-quality-smoke-<timestamp>/`. Historical local-flat result: q4 overfetch100 records `ndcg@10=0.407586`, `recall@100=0.724111`, DB directory multiple `0.048935x`, and p95 search latency about `11.8ms`; q4 full overfetch records `recall@100=0.741889`; planner fit remains `123` q4 children, so the `100` child target fits. The packed-parent path now forwards and records the compact storage modes from the child-vector smoke. Promoted narrow-default packed-parent API evidence is `runs/eos-narrow-default-corkscrewdb-budget-quality-packed-q4q8-main-20260623T050216Z`, exported from `runs/eos-vector-caches/eos-narrow-default-scifact-child-w128-o32-128d` and run against CorkScrewDB repo `$HOME/work/corkscrewdb`. It measured `5,183` parents, `12,468` children, and `128d` vectors with local flat `packed_parent_multivectors`, `packed_metadata_mode=none`, `packed_child_id_mode=ordinal`, `quantized_only` storage, and flat index: q4 release-gate evidence recorded nDCG@10 `0.452380`, recall@100 `0.758556`, DB directory multiple `0.041676x`, vector payload multiple `0.013312x`, p95 `16.076127ms`, planner fit `180`, `target_child_fit=true`, and target storage multiple `0.554545x`; q8 diagnostic evidence recorded nDCG@10 `0.472132`, recall@100 `0.773556`, DB directory multiple `0.066734x`, vector payload multiple `0.025841x`, p95 `36.113081ms`, planner fit `93`, `target_child_fit=false`, and target storage multiple `1.074026x`, so q8 remains diagnostic under the 100-child planner target and DB directory gate. The s40, nf005, and older targeted-v3 packed-parent local flat runs remain predecessor comparison/provenance. Read these rows beside the cache-only smoke and pure planner smoke as local flat CorkScrewDB API evidence for compact child-vector storage, separate from q4/fp16 sidecar rerank evidence; do not generalize them to HNSW, time-series, remote mode, federation, hosted parity, or a service SLO.

For HNSW index-search validation on the same compact cache, run:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_corkscrewdb_hnsw_quality.fw
```

This wrapper runs the child-vector smoke with `index_type=hnsw`, `vector_storage=raw`, and a post-load HNSW rebuild, then joins the same overhead-aware q4 planner row under `runs/eos-corkscrewdb-hnsw-quality-smoke-<timestamp>/`. Current promoted narrow-default HNSW evidence is `runs/eos-current-default-corkscrewdb-hnsw-refresh-v1-20260623T055809Z`, using explicit current cache paths from `runs/eos-vector-caches/eos-narrow-default-scifact-child-w128-o32-128d`. q4 overfetch100 records nDCG@10 `0.445689`, recall@100 `0.735222`, raw HNSW DB directory multiple `0.243494x`, vector payload multiple `0.013312x`, p95 `2.324070ms`, planner fit `123`, `target_child_fit=true`, and target storage multiple `0.811688x`; q4 overfetch12468 records nDCG@10 `0.452380`, recall@100 `0.758556`, and p95 `49.635238ms`. The q4 rebuild took about `399.435824s`; q8 was not run. CorkScrewDB rejects HNSW with `quantized_only`, so this smoke deliberately trades compact quantized-only persistence for raw-vector HNSW validation. Keep the raw HNSW DB-directory multiple separate from the flat q4 quantized-only storage multiple; the q4 vector payload/planner columns are planner comparisons, not HNSW persistence claims. This is local API evidence only, not remote mode, federation, hosted parity, or a service SLO.

For a first time-series/window quality seam, export text-rendered numeric windows and run the existing parent-child evaluator against parent-series qrels:

```bash
go run ./cmd/eos export-timeseries-vectors \
  --dataset sensor-window-retrieval \
  --batch-size 64 \
  --output-dim 128 \
  --window-size 64 \
  --window-stride 16 \
  --manifest-json runs/timeseries-window-cache-128d/manifest.json \
  runs/default-embedder/eos-embed-v1.sealed.mll \
  data/timeseries/sensor-series.jsonl \
  data/timeseries/queries.jsonl \
  runs/timeseries-window-cache-128d
```

```bash
go run ./cmd/eos eval-retrieval-multivector-turboquant \
  --dataset sensor-window-retrieval \
  --backend text-rendered-timeseries-windows \
  --artifact eos-embed-v1-prefix128 \
  --doc-vectors runs/timeseries-window-cache-128d/child-doc-vectors.jsonl \
  --query-vectors runs/timeseries-window-cache-128d/query-vectors.jsonl \
  --qrels data/timeseries/qrels/test.tsv \
  --bits 2,4,8 \
  --baseline-dim 3072 \
  runs/timeseries-window-cache-128d
```

`export-timeseries-vectors` expects one JSONL object per parent series with `id` or `_id` and numeric `values`. It writes deterministic child IDs like `series-id#window-0000`, renders each numeric window as stable text with values and simple stats, embeds those windows with the same Eos text embedder path as retrieval export, and also writes BEIR helper files `corpus.jsonl` and `queries.jsonl` into the output directory. Use that output directory as the evaluator dataset directory and pass parent-series qrels with `--qrels`; the qrels corpus IDs must be parent series IDs. This is a quality harness for text-rendered numeric windows, not a final trained numeric time-series encoder.

Reusable smoke:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_timeseries_window_vectors.fw
```

This script creates a tiny synthetic five-series dataset, regenerates the time-series child-vector cache with the sealed Eos artifact, runs dense/q2/q4/q8 multivector TurboQuant evaluation, and then runs the storage planner with overhead-aware baseline accounting. Its `summary.tsv` and `manifest.json` demonstrate the narrow economics claim: many 128d quantized window vectors can fit under one 3072d dense-vector budget while q8 stays at dense quality on this controlled smoke. Override `EOS_TS_WINDOW_SMOKE_ARTIFACT`, `EOS_TS_WINDOW_SMOKE_OUTPUT_DIM`, `EOS_TS_WINDOW_SMOKE_WINDOW_SIZE`, `EOS_TS_WINDOW_SMOKE_WINDOW_STRIDE`, `EOS_TS_WINDOW_SMOKE_BITS`, `EOS_TS_WINDOW_SMOKE_BASELINE_DIM`, and `EOS_TS_WINDOW_SMOKE_VECTOR_OVERHEAD_BYTES` to reuse the harness on another artifact or shape.

To prove the same synthetic window child vectors through local CorkScrewDB `PutVector`/`SearchVector`, run:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_corkscrewdb_timeseries_windows.fw
```

The wrapper nests the time-series smoke under `timeseries/`, feeds the generated `child-doc-vectors.jsonl`, `query-vectors.jsonl`, and `qrels.tsv` into `scripts/smoke_corkscrewdb_child_vectors.fw`, and defaults CorkScrewDB to flat `quantized_only` storage. Its joined `summary.tsv` reports parent count, child-window count, derived windows per parent, q4/q8 quality, planner fit, measured DB directory multiple, vector payload multiple, and p95 latency. S40-era predecessor result: `5` parents, `25` child windows, `5` derived windows per parent; q4 flat/quantized_only records `ndcg@10=1.000000`, `recall@100=1.000000`, planner fit `123`, vector payload multiple `0.034180x`, and DB directory multiple `0.117643x`; q8 records `ndcg@10=0.926186`, `recall@100=1.000000`, planner fit `75`, vector payload multiple `0.060221x`, and DB directory multiple `0.143685x`. Observed p95 latency for the default synthetic smoke has been sub-ms, but it varies by run. This is local flat API evidence for synthetic text-rendered numeric windows only; it is not remote mode, federation, HNSW, a trained numeric time-series encoder, or a claim that DB directory bytes and planner/vector payload bytes are the same accounting layer. Refresh before using it as new-default evidence.

Use `EOS_TS_WINDOW_SMOKE_SCENARIO=contrastive_needle` for the standalone cache smoke, or `EOS_CORKSCREW_TS_WINDOW_SCENARIO=contrastive_needle` for the CorkScrewDB wrapper, when the question is whether packed child multivectors preserve local window/facet structure better than one mean-pooled parent vector. The scenario keeps parent series mostly similar and inserts one query-specific short regime. Verified local flat CorkScrewDB q2 runs on the same `24` parent, `264` child-window inputs showed packed parent multivectors ahead of `single_parent_vectors`: packed q2 `none`/`ordinal` run `runs/eos-corkscrewdb-timeseries-contrastive-needle-q2-packed-20260616T000000Z/` recorded `ndcg@10=0.769694`, `recall@100=1.000000`, planner packed fit `341`, packed planner multiple `0.034740x`, vector payload multiple `0.032227x`, DB directory multiple `0.042772x`, and p95 `0.045773ms`; single-parent q2 run `runs/eos-corkscrewdb-timeseries-contrastive-needle-q2-single-parent-20260616T000000Z/` recorded `ndcg@10=0.677304`, `recall@100=1.000000`, vector payload multiple `0.002930x`, DB directory multiple `0.019687x`, and p95 `0.032332ms`. q4 also favored packed, `0.831824` nDCG@10 versus `0.818766`, with recall@100 `1.000000` for both. This is a synthetic local-window contrast, separate from the q2-341 packed storage-capacity proof.

Scale the same path to 100 q4 child windows per parent and enough parent variants to separate fixed DB-directory overhead from per-vector TurboQuant payload accounting with:

```bash
EOS_REPO_ROOT=$PWD \
EOS_CORKSCREW_TS_WINDOW_BITS=4 \
EOS_CORKSCREW_TS_WINDOW_SERIES_LENGTH=1648 \
EOS_CORKSCREW_TS_WINDOW_SERIES_VARIANTS=20 \
EOS_CORKSCREW_TS_WINDOW_WINDOW_SIZE=64 \
EOS_CORKSCREW_TS_WINDOW_WINDOW_STRIDE=16 \
EOS_CORKSCREW_TS_WINDOW_TARGET_CHILDREN=100 \
EOS_CORKSCREW_TS_WINDOW_TOP_PARENTS=100 \
EOS_CORKSCREW_TS_WINDOW_OVERFETCH=500 \
EOS_CORKSCREW_TS_WINDOW_MAX_VECTOR_PAYLOAD_MULTIPLE=0.70 \
EOS_CORKSCREW_TS_WINDOW_MAX_DB_DIR_MULTIPLE=0 \
ferrous-wheel run scripts/smoke_eos_corkscrewdb_timeseries_windows.fw
```

The 1648/64/16 windowing derives exactly 100 child windows per parent and keeps q8 out of the run. With `EOS_CORKSCREW_TS_WINDOW_SERIES_VARIANTS=20`, the five query patterns expand to `100` parent series and `10,000` child windows while keeping the original five pattern queries; each query has `20` relevant parent variants, for `100` relevant query-parent pairs. Verified separate-child run `runs/eos-corkscrewdb-timeseries-window-scale-q4-100-variants20-20260616T000000Z/` recorded q4 flat/`quantized_only` with planner fit `123`, planner storage multiple `0.811688x`, vector payload multiple `0.683594x`, DB directory multiple `2.293748x`, DB bytes `2,818,558`, `ndcg@10=0.352927`, `recall@100=0.560000`, p95 `11.948319ms`, and overfetch `500`.

Packed parent-object storage has now been measured on the same q4 scaled shape with local flat exact parent search. Verified `EOS_CORKSCREW_SMOKE_PACKED_METADATA_MODE=none` and `EOS_CORKSCREW_SMOKE_PACKED_CHILD_ID_MODE=ordinal` run `runs/eos-corkscrewdb-timeseries-window-scale-q4-100-variants20-packed-minimal-20260616T000000Z/` recorded `layout=packed_parent_multivectors`, `parent_insert_count=100`, `parent_search_exact=true`, `overfetch_children=0`, DB directory multiple `0.844660x`, DB bytes `1,037,918`, `ndcg@10=0.352927`, `recall@100=1.000000`, and p95 `5.916649ms`. DB bytes were `0.421237x` of the prior full-metadata packed run and `0.368244x` of the separate-child run. This is a storage/API evidence win for packed parent objects; the recall delta reflects exact parent rollup versus bounded separate-child overfetch. It remains synthetic text-rendered local flat evidence, not remote/federation/HNSW, production quality, or a trained numeric time-series encoder claim.

The corrected q2 packed-parent frontier has also been measured through the same local flat API path after TurboQuant IP byte-accounting, packed overhead fixes, and CorkScrewDB compact snapshot v5 commit `c208f9b50d29f9fdf19771c4b093332c7c8fd0b4`. Unified wrapper run `runs/eos-corkscrewdb-timeseries-window-q2-341-compact-v5-unified-20260616T000000Z/` generated the child/query/qrels inputs, packed planner evidence, and persisted DB directory evidence for q2 `128d`, `packed_parent_multivectors`, `packed_metadata_mode=none`, and `packed_child_id_mode=ordinal` with `100` parents, `34,100` child windows, and `341` windows per parent. It measured `quantized_vector_bytes=36`, `quantized_child_bytes=1,227,600`, vector payload multiple `0.9990234375x`, packed planner bytes `12,308`, packed planner multiple `0.999025974025974x`, measured DB directory bytes `1,237,818`, DB directory multiple `1.0073388671875x`, `ndcg@10=0.4493940305106442`, `recall@100=1.000000`, and p95 `1.418733ms`, passing planner-fit, vector-payload, DB-directory, and p95 gates. Cite this as local flat packed-parent API evidence that compact snapshot v5 ordinal encoding brings the strict q2-341 quantized-only persisted DB directory to approximately one dense parent-vector budget; without that compact snapshot path, or for richer child records, keep DB directory cost separate.

The first quality harness for that lane is cache-only and still outside the CorkScrewDB API:

```bash
go run ./cmd/eos eval-retrieval-multivector-turboquant \
  --dataset scifact \
  --backend eos-child-cache-128d \
  --artifact eos-embed-v1-prefix128 \
  --doc-vectors runs/eos-vector-caches/eos-embed-v1-scifact-child-w128-o32-128d/scifact/child-doc-vectors.jsonl \
  --query-vectors runs/eos-vector-caches/eos-embed-v1-scifact-child-w128-o32-128d/scifact/query-vectors.jsonl \
  --bits 2,4,8 \
  --baseline-dim 3072 \
  --metrics-json runs/scifact.128d-child.multivector-turboquant.metrics.json \
  --metrics-tsv runs/scifact.128d-child.multivector-turboquant.metrics.tsv \
  datasets/manta-embed-v1/raw/scifact/scifact
```

Document child-vector JSONL accepts `parent_id`, `child_id`, and one vector field among `vector`, `embedding`, or `values`. When `parent_id` is absent, `id` or `_id` is used as both parent and child, so one-vector caches remain a valid degenerate multi-vector input. The evaluator scores every child vector, aggregates by max child score per parent, and evaluates parent IDs against BEIR qrels. Strict coverage is the default: if any qrels-relevant parent is missing from the child-vector cache, the run fails instead of filtering that parent out and inflating metrics. Use `--allow-missing-relevant` only for diagnostic smoke runs where incomplete qrel coverage is intentional. Its dense row uses the same max-child aggregation over fp32 child vectors; q2/q4/q8 rows quantize children with a deterministic TurboQuant IP seed, configurable with `--quantizer-seed`, and use direct TurboQuant IP scoring without fp16 rerank sidecars. JSON/TSV metrics include `allow_missing_relevant`, `quantizer_seed`, `baseline_dim`, parent count, child-vector count, average and max children per parent, dense baseline bytes, dense child bytes, per-vector quantized bytes, total quantized child bytes, dense-child compression, vectors that fit in one dense baseline, storage multiple versus one dense baseline vector per parent, scored child pairs, quality deltas, scores/s, and query latency summaries. Leave `--baseline-dim` at `0` only for same-dimension accounting; use `1024` or `3072` when the claim is compact children versus a large dense embedder.

First measured SciFact parent-child evidence: Qwen3 0.6B child cache, `128` word chunks, `32` word overlap, and `16` word minimum trailing chunk produced `5,183` parents and `12,468` children, averaging `2.41` children per parent. The run evaluated `300` qrels queries with strict qrels coverage and recorded the deterministic quantizer seed in metrics.

| row | child nDCG@10 | child recall@100 | compression | parent-budget multiple |
| --- | ---: | ---: | ---: | ---: |
| dense-child | 0.717467 | 0.953333 | n/a | n/a |
| q2 | 0.701697 | 0.950000 | 15.75x | 0.15x |
| q4 | 0.714895 | 0.956667 | 7.94x | 0.30x |
| q8 | 0.716310 | 0.953333 | 3.98x | 0.60x |

Compared with the existing one-vector Qwen3 SciFact evidence, dense child-max improves over dense `0.702026` nDCG@10 / `0.946667` recall@100, and direct q8 child-max improves over q8 `0.702657` nDCG@10 / `0.946667` recall@100 while storing q8 children below one dense-parent-vector budget.

Mixedbread `mixedbread-ai/mxbai-embed-large-v1` is the current stronger external SciFact child-cache baseline on the same parent-child lane. The requested `datasets/eos-embed-v1/raw/scifact/scifact` path was absent, so the run used `datasets/manta-embed-v1/raw/scifact/scifact`, matching the Qwen3 child evidence. It used `128` word chunks, `32` overlap, and `16` minimum trailing words, producing `5,183` parents, `12,468` child vectors, `2.405557` average children per parent, and `300` evaluated qrels queries with strict coverage (`allow_missing_relevant=false`).

| row | child nDCG@10 | child recall@100 | compression | parent-budget multiple | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense-child | 0.747175 | 0.970000 | n/a | n/a | 12.497 ms |
| q2 | 0.712790 | 0.956667 | 15.75x | 0.15x | 4.754 ms |
| q4 | 0.739489 | 0.965000 | 7.94x | 0.30x | 77.250 ms |
| q8 | 0.747799 | 0.966667 | 3.98x | 0.60x | 157.876 ms |

mxbai is higher than Qwen3 child-max on dense, q2, q4, and q8 nDCG@10 and recall@100. The q8 mxbai row beats Qwen3 q8 by `+0.031489` nDCG@10 and `+0.013334` recall@100. Treat mxbai as the stronger external SciFact child-cache baseline, while Qwen3 remains useful as a compact leading-family baseline.

The sealed Eos/default path is now measured end-to-end for this lane: `runs/manta-embed-v1-deephard-full-ft-20260610T0000Z/manta-embed-v1.sealed.mll` exported a full Go-native SciFact child cache from `datasets/manta-embed-v1/raw/scifact/scifact` with `128` word chunks, `32` overlap, and `16` minimum trailing words. Export counts were `5,183` docs, `300` queries, `12,468` children, dim `256`, CUDA backend, and `57.771s` elapsed. The strict eval used `allow_missing_relevant=false`, `339` relevant pairs, `3,740,400` scored child pairs, and quantizer seed `5581486560434873699`.

| row | Eos child nDCG@10 | Eos child recall@100 | compression | parent-budget multiple | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense-child | 0.462489 | 0.778111 | n/a | n/a | 3.129 ms |
| q2 | 0.383295 | 0.719667 | 15.06x | 0.16x | 1.159 ms |
| q4 | 0.449435 | 0.773111 | 7.76x | 0.31x | 17.819 ms |
| q8 | 0.461862 | 0.774778 | 3.94x | 0.61x | 39.192 ms |

This proves the sealed `.mll` -> Go-native child vector cache -> strict TurboQuant multivector eval path, but it does not promote the current sealed Eos anchor on SciFact child-cache quality. Eos dense-child is materially below full mxbai `0.747175` nDCG@10 / `0.970000` recall@100 and Qwen3 `0.717467` / `0.953333`; q8 preserves Eos dense-child quality closely, and q4 is near but drops more. The main deficit is model quality, not TurboQuant storage or scoring.

The real local CorkScrewDB layout comparison for the same full Eos SciFact child cache now covers packed child spans versus mean-pooled single-parent vectors. Wrapper run `runs/eos-real-scifact-full-packed-none-ordinal-q4q8-diagnostic2-20260616T000000Z/` used `packed_parent_multivectors`, `packed_metadata_mode=none`, and `packed_child_id_mode=ordinal`; run `runs/eos-real-scifact-full-single-parent-q4q8-diagnostic-20260616T000000Z/` used `single_parent_vectors`. Both used `5,183` parents, `12,468` child spans, `300` queries, and `256d` vectors from `runs/eos-vector-caches/eos-embed-v1-scifact-child-w128-o32-full/scifact/`. Packed q4/q8 recorded nDCG@10 `0.449435`/`0.461862`, recall@100 `0.773111`/`0.774778`, DB directory multiples `0.032900x`/`0.057958x`, and p95 `15.934540ms`/`30.366188ms`. Single-parent q4/q8 recorded nDCG@10 `0.406498`/`0.422597`, recall@100 `0.743111`/`0.745889`, DB directory multiples `0.022411x`/`0.032827x`, and p95 `7.075771ms`/`13.242884ms`. This advances the default CorkScrewDB embedder path from synthetic layout contrast to real document-span evidence: packed preserves child-span evidence better, while single-parent is the cheaper/faster mean-pooled baseline. The run disabled old q4 quality/storage/planner floors for diagnostic comparison and does not claim remote, HNSW, federation, or native Matryoshka behavior.

TurboQuant's strategic win for CorkScrewDB is not only q4/q8 compression of one vector. Direct compact child vectors let a parent object carry many cheap vectors for windows, spans, time-series slices, events, and other multi-vector schemas. Be precise in product planning: same-dimension child vectors do not fit hundreds of children inside the budget of one same-dimension fp32 parent vector. Tens to hundreds become plausible when compact child dimensions are planned against a 1024 to 3072 dimensional fp32 baseline with `--baseline-dim` or when the product explicitly budgets multiple dense-parent equivalents. That needs a measured planner and eval lane, not hand-wavy storage copy.

Keep this separate from q4/fp16 rerank. `--sidecar-storage fp16` intentionally adds a per-child fp16 sidecar and shows why sidecars destroy the high-child-count storage argument: the sidecar is useful for quality-preserving two-stage rerank, but it is not the direct hundred-child storage mode.

## Data And Teacher Growth

Grow the training/evaluation signal before increasing model size:

- Add scored teacher batches from BGE, Qwen, Jina, Voyage, Cohere, and OpenAI.
- Keep provider outputs as vector caches or scorer batches, not as provider SDK dependencies inside Eos.
- For Qwen3/mxbai hard-negative distillation, run `eos audit-teacher-scores`, then `eos filter-teacher-scores` before training. The next Qwen3 guarded shape should use agreement-filtered `teacher_scores` so teacher loss applies only where the teacher ranks the labeled positive above the negatives; static source-weight-only shaping already failed to balance SciFact gains against NFCorpus/FiQA regressions.
- Current prepared Qwen3/mxbai short-set input: `runs/eos-shortset-agreement-teacher-prep-v1-20260621T000000Z/shortset.qwen3-mxbai.agreement-filtered.train-hard-negatives.jsonl`. It preserves all `4919` hard-negative examples and keeps averaged agreement `teacher_scores` for `2154` rows: SciFact `667/919`, NFCorpus `373/2000`, FiQA `1114/2000`. Combined audit reports positive top-1 rate `1.000000`, mean positive margin `0.130634`, and `2765` unscored preserved examples. FiQA coverage is exportable-text coverage with explicit empty raw corpus text skipping in the bridge manifests, not raw-row-complete or judged-coverage-complete evidence. This short-set lane is superseded as the immediate next training action by the research-only MS MARCO hard-candidate smoke result below.
- MS MARCO hard-candidate dual-teacher work is research-only: `release_train_allowed=false`, `commercial_use_allowed=false`, and `train_allowed_for_research=true`. The clean agreement artifact is mechanically valid, but the first bounded smoke did not show matched pairwise/hard lift and should not be promoted or full-scoreboarded as-is. The LR-control follow-up selected `1.25e-8` after the prior `2.5e-8` smoke moved and regressed AUC/margin; eval salvage at batch `8` exactly matched matched-bootstrap final/hard metrics, showed no top1/MRR lift, and movement diagnostics found zero changed vectors at 64d, 128d, and full dimensions. Close strict clean-agreement LR tuning here: no retrieval scoreboard or compact scoring for this branch.
- MS MARCO conflict adjudication is scored with the official `Qwen/Qwen3-Reranker-0.6B` revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`; the local Hugging Face cache was about `1207488344` bytes, and the known-bad seq-cls Qwen3 reranker path was not used. The scorer processed the staged `2,048` request rows in `29.666s` (`69.036` req/s) on CUDA bfloat16 batch `8` using yes/no final-token logits. Over `512` conflict rows, outcomes were `alternate_relevant=253`, `ambiguous=141`, and `confirmed_ordering=118`. All `5` zero-margin fixtures were present: `2` distinct-text ties were confirmed, and `3` normalized duplicate-text ties stayed ambiguous. The completed score-spectrum artifact has `2,891` rich rows (`2,379` clean + `512` conflict) and `2,497` current-trainer preview rows, with legal gates false/false/true preserved. Alternate-relevant rows remain multi-positive soft-target rows, not current hard-negative trainer rows.
- Use hard negatives and retrieval-error mining to target the weakest datasets in the matrix.
- Try Matryoshka-style dimension slicing before moving to larger embedding dimensions, so CorkScrewDB can trade quality for memory and scoring cost.

## Default Promotion Gate

Do not promote a default CorkScrewDB embedder until the relevant gate lanes are separated and the required lane for the claim has passed:

Dense asset gate:

- The sealed `.mll` package verifies by SHA256 and package metadata.
- The dense baseline matrix row compares against the accepted dense predecessor on the selected datasets and metrics.
- The release smoke verifies the checked-in asset, embedded package metadata, local `embed-text` load, and CorkScrewDB embedded default startup when claiming the packaged default asset.
- The docs name the measured default and avoid unsupported standing claims.

Compact TurboQuant gate:

- The baseline matrix has explicit direct TurboQuant and TurboQuant rerank rows, with missing external rows still visible as `not_scored`.
- q4 and q8 default choices are measured on the same datasets, with quality deltas, vector bytes, compression, docs/s, scores/s, rerank overfetch where applicable, and serving p50/p95/p99/max attached.
- A CorkScrewDB load/index/search smoke has passed with the candidate vectors; use `scripts/smoke_corkscrewdb_child_vectors.fw` for the local flat API path and override its input-cache env vars for candidate-specific child vectors. Keep `EOS_CORKSCREW_SMOKE_LAYOUT=separate_child_vectors` for the current child-overfetch baseline, or set `EOS_CORKSCREW_SMOKE_LAYOUT=packed_parent_multivectors` to measure local flat `PutMultiVector` plus exact `SearchParentsVector` packed-parent persistence. Predecessor nf005 packed-parent evidence uses cache `runs/eos-vector-caches/eos-nf005-current-default-scifact-child-w128-o32-128d/` and run `runs/eos-nf005-current-default-corkscrewdb-budget-quality-packed-q4q8-20260617T142823Z/`. The local flat q4 row records nDCG@10 `0.451122`, recall@100 `0.755222`, DB multiple `0.020372x`, vector payload multiple `0.013312x`, p95 `142.971679ms`, planner fit `180`, target fit `true`, and target storage multiple `0.554545x`; q8 is diagnostic at nDCG@10 `0.472867`, recall@100 `0.780222`, DB multiple `0.032901x`, p95 `22.988553ms`, planner fit `93`, and target fit `false`. Predecessor targeted-v3 packed-parent evidence remains historical comparison/provenance only.

Optional hybrid retrieval-policy gate:

- Hybrid retrieval policy rows are calibrated or otherwise explicitly selected on the documented dev split, then applied unchanged to test.
- CorkScrewDB API evidence uses the intended dense+sparse policy path, records dense-only and hybrid metrics, and states the sparse method. The s40-era BM25-dot rows in `runs/eos-s40-current-default-hybrid-policy-v1-20260621T103500Z/` are local flat API evidence for `WithSparse` plus `SearchMulti`; they do not alter the dense default asset and do not prove remote, HNSW, federation, or hosted surfaces. Refresh them before claiming hybrid/API coverage for the new default.
- Do not use hybrid retrieval-policy evidence to bypass dense asset or compact TurboQuant promotion gates.

## Next Actions

1. Optionally extend evidence to remote CorkScrewDB mode, HNSW, or federation before making any claims about those surfaces; the current release evidence is local packaging evidence, and BM25-dot hybrid rows remain s40-era local flat API evidence until refreshed.
2. Keep the full short-set external matrix current, with Qwen3 FiQA labeled as full exportable-text rather than raw-row-complete or judged-coverage complete.
3. Continue model-improvement work only with a changed signal or objective: for MS MARCO hard-candidate research, do not train from the score-spectrum preview as the main quality path because it excludes the `253` alternate-relevant rows; the next engineering move is research-only trainer support for multi-positive/soft-target score-spectrum rows. Do not spend more on strict clean-agreement LR tuning; otherwise continue larger-model bootstrap work and Matryoshka-style dimension slicing rather than treating BM25-dot hybrid retrieval rows as dense model improvement.
