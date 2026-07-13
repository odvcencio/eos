# Production Embedding Candidate

Use `scripts/train_manta_embed_v1_candidate.fw` to create a release-grade `eos-embed-v1` candidate. The workflow wraps the current Eos CLI primitives with production guardrails:

- refuses temporary input paths unless explicitly overridden
- refuses dirty repositories unless explicitly overridden
- records repo commit, Go version, selected environment, dataset SHA256, artifact SHA256, and run config
- trains from either a raw corpus or prepared JSONL
- runs eval-only verification on a copied package and requires `optimizer_updates=0`
- runs a separate hard holdout eval by default
- exports and verifies a sealed MLL
- compares dense retrieval against TurboQuant IP-preserving quantized document vectors before default CorkScrewDB embedder promotion
- supports metric gates through environment variables

## Imported BGE Candidate Boundary

The imported BGE package for `BAAI/bge-small-en-v1.5` is a first-class
non-default candidate only after the exact selected-package full gate passes.
It remains separate from the `corkscrewdb-default-embedder` alias: the package
uses an imported BERT/WordPiece runtime, 384d vectors, CLS pooling, L2
normalization, and the BGE query prefix role contract. Current imported BGE
vector export is `pretrained_bert_host_reference`; CUDA matmul or layer-level
acceleration evidence is not a full device encoder contract. Use
`eos export-pretrained-bert-retrieval-vectors --require-device-encoder` for
energy/readiness gates that must fail unless manifest `encoder_execution`
provenance validates full device-side encoder execution. The candidate package
SHA256 is `841b0d851c06290daeeab4bf4d25cb1dd7bb87920316dac950e1b556a3bae763`,
the identity SHA256 is
`a356a4b7dc29a8d0f0a7b7bd45e7a9d2afbfa651c1a5bfaa05008c7157ba9637`,
`quality_claim=false`, and `default_alias_changed=false`.

Do not treat BGE package availability as a default-embedder promotion. Keep the
current default release smoke and 256d CorkScrewDB manifest checks attached to
`corkscrewdb-default-embedder`; use the top-level `NOTICE` for the shipped BGE
source, snapshot, package identity, role contract, quality boundary, and MIT
notice packet.

## Preflight

Run the local preflight before spending trainer time on a candidate:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/verify_manta_production.fw
```

This uses generated tiny fixtures to verify the public Eos surface, build `cmd/eos`, initialize and train a `.mll` package, run eval-only checks with `optimizer_updates=0`, export a sealed `.mll`, and inspect both packages. It is a production path check, not a model quality result.

## Acquire Datasets

The default acquisition workflow downloads public BEIR retrieval datasets (`scifact`, `nfcorpus`, and `fiqa`), converts qrels to Eos text-pair JSONL, writes a tokenizer corpus, and emits initial threshold gates:

```bash
EOS_REPO_ROOT=$PWD \
EOS_DATASET_ROOT=/data/manta/datasets/eos-embed-v1 \
ferrous-wheel run scripts/acquire_manta_embed_v1_datasets.fw
```

The processed outputs are:

```text
/data/manta/datasets/eos-embed-v1/processed/train.jsonl
/data/manta/datasets/eos-embed-v1/processed/eval.jsonl
/data/manta/datasets/eos-embed-v1/processed/hard-eval.jsonl
/data/manta/datasets/eos-embed-v1/processed/tokenizer-corpus.txt
/data/manta/datasets/eos-embed-v1/processed/thresholds.env
```

Review dataset licenses before commercial use. The default avoids MS MARCO because it is commonly distributed with non-commercial research terms.

## Process Corpus Pretraining

Use the process-doc lane when a candidate should include local Tiller/Codex operating language before BEIR alignment. The builder reads local process files such as `AGENTS.md`, `.codex/agents/*.toml`, and `.codex/skills/**/SKILL.md`, chunks them, and emits text hard-negative JSONL with:

```json
{"query":"process guidance in AGENTS.md","positive":"...","negatives":["..."],"source":"process:agents","group_id":"process-agents-md-0"}
```

For a small local smoke:

```bash
EOS_REPO_ROOT=$PWD \
EOS_PRETRAIN_BEIR=0 \
EOS_PROCESS_PRETRAIN=1 \
EOS_PROCESS_PRETRAIN_MAX_DOCS=8 \
EOS_PROCESS_PRETRAIN_MAX_ROWS=16 \
EOS_PRETRAIN_OUT=$PWD/datasets/eos-embed-v1/processed/process-pretrain-smoke.jsonl \
ferrous-wheel run scripts/build_pretrain_pairs.fw
```

For the normal Stage A pretraining file, add process rows alongside the existing BEIR synthetic rows:

```bash
EOS_REPO_ROOT=$PWD \
EOS_DATASET_ROOT=/data/manta/datasets/eos-embed-v1/raw \
EOS_PROCESS_PRETRAIN=1 \
EOS_PROCESS_PRETRAIN_INCLUDE_DOCS=1 \
EOS_PRETRAIN_OUT=/data/manta/datasets/eos-embed-v1/processed/pretrain-pairs.jsonl \
ferrous-wheel run scripts/build_pretrain_pairs.fw
```

`EOS_PROCESS_PRETRAIN_INCLUDE_DOCS=1` adds `docs/**/*.md`. Use `EOS_PROCESS_PRETRAIN_PATHS=path1,path2` for extra files or directories, `EOS_PROCESS_PRETRAIN_MAX_DOCS` and `EOS_PROCESS_PRETRAIN_MAX_ROWS` for bounded probes, and `EOS_PROCESS_PRETRAIN_CHUNK_WORDS` when the default `220`-word chunks are too large or small.

Set `EOS_PROCESS_PRETRAIN_SYNTHETIC_QUERIES=1` with `EOS_PROCESS_PRETRAIN=1` to replace the generic `process guidance in <path>` queries with deterministic path-, heading-, and term-derived query-style rows such as `Where is <topic> described in the Eos process?`. These rows keep the same hard-negative JSONL shape and use `process-query:*` sources. Treat this as generated-data plumbing/proof until a downstream retrieval or training gate shows model-quality movement.

A bounded end-to-end process-corpus smoke generated `12` process rows, trained through the hard-negative path with `optimizer_updates=42`, and completed a separate eval-only pass with `optimizer_updates=0`. Treat this as plumbing proof only, not a quality result.

The shipping pipeline reads `/data/manta/datasets/eos-embed-v1/processed/pretrain-pairs.jsonl` when it builds `shipping-mixed-pretrain-plus-beir.jsonl`. If that file already exists, rebuild it with the command above before running:

```bash
EOS_REPO_ROOT=$PWD \
EOS_DATASET_ROOT=/data/manta/datasets/eos-embed-v1 \
EOS_SHIP_RUN_ROOT=/data/manta/runs/eos-embed-v1-process-pretrain \
ferrous-wheel run scripts/train_manta_embed_v1_shipping_pipeline.fw
```

For a direct candidate experiment, feed the process JSONL through the existing hard-negative path:

```bash
EOS_REPO_ROOT=$PWD \
EOS_RUN_ROOT=/data/manta/runs \
EOS_RUN_ID=eos-embed-v1-process-pretrain-a \
EOS_TRAIN_JSONL=/data/manta/datasets/eos-embed-v1/processed/pretrain-pairs.jsonl \
EOS_EVAL_JSONL=/data/manta/datasets/eos-embed-v1/processed/eval.jsonl \
EOS_HARD_EVAL_JSONL=/data/manta/datasets/eos-embed-v1/processed/hard-eval.jsonl \
EOS_HARD_NEGATIVE_TRAIN=1 \
EOS_HARD_NEGATIVES_PER_QUERY=1 \
EOS_EPOCHS=1 \
ferrous-wheel run scripts/train_manta_embed_v1_candidate.fw
```

## Prepared JSONL Path

Prepared JSONL is the preferred path for production because train/eval splits are fixed before training starts.

```bash
EOS_RUN_ROOT=/data/manta/runs \
EOS_REPO_ROOT=$PWD \
EOS_RUN_ID=eos-embed-v1-20260412-a \
EOS_TRAIN_JSONL=/data/manta/datasets/eos-embed-v1/processed/train.jsonl \
EOS_EVAL_JSONL=/data/manta/datasets/eos-embed-v1/processed/eval.jsonl \
EOS_HARD_EVAL_JSONL=/data/manta/datasets/eos-embed-v1/processed/hard-eval.jsonl \
EOS_TOKENIZER_CORPUS=/data/manta/datasets/eos-embed-v1/processed/tokenizer-corpus.txt \
EOS_THRESHOLDS_ENV=/data/manta/datasets/eos-embed-v1/processed/thresholds.env \
EOS_EPOCHS=3 \
EOS_BATCH_SIZE=1024 \
EOS_LR=0.005 \
EOS_TEMPERATURE=0.05 \
EOS_SELECT_METRIC=score_margin \
EOS_EVAL_EVERY_STEPS=0 \
EOS_MIN_AUC=0.70 \
EOS_MIN_THRESHOLD_ACCURACY=0.65 \
EOS_MIN_SCORE_MARGIN=0.05 \
EOS_MAX_LOSS=0.35 \
ferrous-wheel run scripts/train_manta_embed_v1_candidate.fw
```

`EOS_THRESHOLDS_ENV` loads the acquisition workflow's current gate file and records its SHA256 in the run manifest. Explicitly exported `EOS_*` values still override values from that file. Set `EOS_TOKENIZER=/path/to/tokenizer.mll` when you want to reuse an existing tokenizer instead of training one from `EOS_TOKENIZER_CORPUS`.

When prepared JSONL is text and a tokenizer is available, the production workflow tokenizes train, validation, and hard-eval JSONL into run-local token files before training. Training and eval then read token JSONL directly, which front-loads BPE cost, makes the optimizer profile reflect model work, and records the generated token files in `datasets.sha256`.
Use `eos train-embed --no-tokenizer` when directly training token JSONL beside a sibling tokenizer; otherwise the CLI intentionally auto-discovers that tokenizer and treats the JSONL as text.
Use `eos diagnose-train-metrics /path/to/train.metrics.json` to explain backend use, transfer pressure, and suspicious training/eval counters from any run. Use `eos gate-train-metrics --thresholds /path/to/thresholds.env --scope quality /path/to/final-eval.metrics.json` to apply the same quality gate outside the production workflow. Use `--scope efficiency` for training throughput and backend counters, and `--scope eval-only` to enforce zero optimizer updates on validation runs.

## TurboQuant Retrieval Gate

Default CorkScrewDB embedder promotion requires a vector-index cost check in addition to dense retrieval quality. The repo-local serving proxy is `scripts/smoke_eos_default_embedder_serving.fw`; for the promoted narrow compact policy, it selects the q4/fp16/rerank-overfetch=200 profile and records p50/p95/p99/max per-query scoring latency:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_eos_default_embedder_serving.fw
```

This smoke uses the in-repo TurboQuant evaluator as CorkScrewDB-relevant vector-index and serving proxy evidence. It is not an actual CorkScrewDB API load/index/search smoke. The promoted narrow release artifact at `runs/eos-s40-nfcorpus-compact-mined-narrow-candidate-v1-20260623T032556Z/candidate/eos-embed-v1.sealed.mll`, sealed SHA256 `e0eca16ff34ebb88ca96862d58c3ac7f02dbf4b124599fdf96f25344ac02e408`, passed strict seeded q4/fp16/o200 compact non-regression against the s40 compact anchor: macro nDCG@10 `+0.000118345582923`, recall@100 `+0.000259403173951`, total compression `1.5900621118x`. The capped serving smoke in `runs/eos-default-embedder-serving-smoke-20260620T161633Z/` selected q4/fp16/o200 with SciFact nDCG@10 `0.7846268033`, recall@100 `0.95`, total compression `1.5900621118x`, and p95 `0.984950ms`; keep it as s40-era predecessor proxy evidence until refreshed for the new default. The formal reproducible predecessor q4/fp16/o200 compact gate anchor is `runs/eos-nf005-compact-anchor-provenance-repair-v1-20260619T091223Z/anchor-q4-fp16-overfetch200-scoreboard/scoreboard.json`; its selected rows carry `quantizer_seed=5581486560434873699`. For the local flat CorkScrewDB API path, run `scripts/smoke_corkscrewdb_child_vectors.fw`; by default it creates or consumes child-vector caches, loads child vectors with CorkScrewDB `PutVector`, queries with `SearchVector`, rolls child hits up to parent IDs by max score, and records storage, latency, nDCG@10, and recall@100:

Use `scripts/bench_eos_default_embedder_serving_energy.fw` beside the serving smoke when a promotion packet needs measured power/energy:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/bench_eos_default_embedder_serving_energy.fw
```

The gate writes `summary.tsv`, `manifest.json`, `power-samples.jsonl`, and command logs under `runs/eos-default-embedder-serving-energy-<timestamp>/`. It builds a run-local `eos` binary once before power sampling, records the build command, binary path, and binary SHA256 in `manifest.json`, then excludes that build from phase wall/power/energy. It distinguishes encoder-only work (`eos export-retrieval-vectors`) from index/scoring work (`eos eval-retrieval-turboquant`). Encoder-only export encodes documents + queries and reports `workload_item_count` plus `energy_joules_per_workload_item`; index/scoring reports `energy_joules_per_query`. If `nvidia-smi` power telemetry is missing or too sparse, the artifact records `unsupported` or `indeterminate` and leaves denominator-specific energy costs blank; set `EOS_ENERGY_BENCH_REQUIRE_POWER=1` to make missing telemetry fail on known-good measurement hosts.

To measure the imported `BAAI/bge-small-en-v1.5` package path instead of the native Eos artifact path, pass the package as the artifact and select imported-BERT workload mode:

```bash
EOS_REPO_ROOT=$PWD \
EOS_ENERGY_BENCH_WORKLOAD_MODE=imported_bert \
EOS_ENERGY_BENCH_ARTIFACT=/path/to/imported-bge-small-en-v1.5.mll \
ferrous-wheel run scripts/bench_eos_default_embedder_serving_energy.fw
```

Imported mode exports caches with `eos export-pretrained-bert-retrieval-vectors --package <artifact> --use-package-role-contract`, then scores those exact `doc-vectors.jsonl` and `query-vectors.jsonl` caches with `eos eval-retrieval-vectors-turboquant`. The vector export manifest records `encoder_execution`, including selected backend, configured `EOS_BERT_DENSE_ACCEL` policy, observed CUDA dense matmul counters, whether a full device encoder contract was validated, and the explicit host-reference/device boundary. CUDA dense counters are opportunistic partial acceleration evidence, not a full-device encoder claim. Set `EOS_ENERGY_REQUIRE_DEVICE_ENCODER=1` on the energy wrapper, or add `--require-device-encoder` directly, only for readiness or energy gates that must fail closed when imported BGE is still host-reference. Energy denominators stay the same: encoder-only uses documents + queries, and index/scoring uses queries.

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_corkscrewdb_child_vectors.fw
```

Set `EOS_CORKSCREW_SMOKE_OVERFETCH` to a comma-separated list, for example `100,12468`, to sweep separate-child serving recall/latency without rebuilding the per-bit DB for each overfetch value. Exhaustive/full child overfetch can compare against offline cache-evaluator parity, but it is a higher-latency diagnostic rather than the default serving setting. Set `EOS_CORKSCREW_SMOKE_LAYOUT=packed_parent_multivectors` to measure the local flat packed-parent API instead: the smoke writes one parent with `PutMultiVector`, searches with exact `SearchParentsVector`, records `parent_search_exact=true`, `parent_insert_count`, and `overfetch_children=0`, and requires `quantized_only` vector storage. Set `EOS_CORKSCREW_SMOKE_LAYOUT=single_parent_vectors` for the cheapest local flat baseline: the smoke groups child-vector inputs by `parent_id`, mean-pools and L2-normalizes one vector per parent, stores parent IDs with `PutVector`, and searches those parent IDs directly with `search_vector_parent_single`. Use these layouts only for local flat DB-size/latency evidence, not remote mode, federation, or HNSW.

S40-era packed-parent local flat API evidence is attached in `.tiller/scratch/codex/eos-corkscrewdb-main-rawstore-api-compat-v1-report.md`, with run `runs/eos-s40-current-default-corkscrewdb-budget-quality-packed-q4q8-main-20260620T165050Z/`, vector cache `runs/eos-vector-caches/eos-s40-current-default-scifact-child-w128-o32-128d/`, and CorkScrewDB main commit `511f5d24408d9aeba21941954d29cca3569875da`. The q4 release-gate row used `packed_parent_multivectors`, `packed_metadata_mode=none`, ordinal child IDs, `quantized_only` storage, and flat index across `5,183` parents, `12,468` children, and `128d` vectors, recording nDCG@10 `0.452971`, recall@100 `0.755222`, DB directory multiple `0.041675x`, vector payload multiple `0.013312x`, p95 `13.434893ms`, planner fit `180`, `target_child_fit=true`, and target storage multiple `0.554545x`. The q8 row is diagnostic only: nDCG@10 `0.472424`, recall@100 `0.776889`, DB directory multiple `0.066733x`, vector payload multiple `0.025841x`, p95 `21.874919ms`, planner fit `93`, `target_child_fit=false`, and target storage multiple `1.074026x`. This direct quantized packed-parent API evidence stays separate from the q4/fp16/rerank-overfetch=200 serving proxy evidence and covers only local flat API load/index/search, not remote mode, federation, HNSW, hosted parity, or a service SLO. Refresh it before claiming API-level quality for the new default. The q8 row misses the 100-child target fit and DB directory gate.

For the separate parent/single-vector compact-cache lane, use `scripts/eval_eos_prefix_dim_curve.fw`. It exports prefix-dimension document/query caches with `export-retrieval-vectors --output-dim`, L2-renormalizes those prefix vectors, and scores dense plus direct TurboQuant q-bit vector indexes. The full-query s40-era `128d` q4/q8 run `runs/eos-prefix-128-q4q8-full-query-current-default-v1-20260620T021328Z/` records `128d q8` as the near-dense compact cache reference at `0.257812` storage multiple, or about `3.878788x` vector-payload compression: SciFact `0.492185`/`0.760778`, NFCorpus `0.184394`/`0.215245`, and FiQA `0.097320`/`0.301407` nDCG@10/recall@100. `128d q4` is the aggressive compact candidate at `0.132812` storage multiple, or about `7.529412x` vector-payload compression: SciFact `0.478427`/`0.758556`, NFCorpus `0.178007`/`0.216910`, and FiQA `0.092377`/`0.297274`. These predecessor metrics are vector-payload accounting, not full CorkScrewDB DB-byte evidence, not q4/fp16 rerank evidence, and not proof of a trained Matryoshka/native 128d head. Refresh before using them as new-default evidence. q2 remains out of the default-quality lane without a trained compact head.

The CorkScrewDB smoke defaults to tiny synthetic time-series vectors generated by `scripts/smoke_eos_timeseries_window_vectors.fw` and uses a local checkout from `EOS_CORKSCREW_SMOKE_CORKSCREWDB_REPO` or `/home/draco/work/corkscrewdb`. It proves the selected local flat load/index/search/accounting path only; it is not remote mode, federation, HNSW, or a model-quality benchmark. For lower-level release checks, run `eos eval-retrieval-turboquant` on at least one capped BEIR-style dataset, and on the full selected retrieval set before updating the `corkscrewdb-default-embedder` alias:

```bash
go run ./cmd/eos eval-retrieval-turboquant \
  --dataset scifact \
  --max-docs 200 \
  --max-queries 20 \
  --bits 2,4,8 \
  --rerank-overfetch 200 \
  --rerank-storage fp16 \
  --metrics-json /data/manta/runs/<run-id>/turboquant-scifact.json \
  --metrics-tsv /data/manta/runs/<run-id>/turboquant-scifact.tsv \
  /data/manta/runs/<run-id>/eos-embed-v1.sealed.mll \
  /data/manta/datasets/eos-embed-v1/raw/scifact
```

The CorkScrewDB embedded default-provider promotion has been checkpointed in CorkScrewDB commit `d19780622b3e879bedfdc202e84966d6437441d5`. For repeatable release validation from a clean Eos checkout, run:

```bash
ferrous-wheel run scripts/smoke_eos_default_embedder_release.fw
```

To include the consuming CorkScrewDB startup check without editing that checkout, pass its root:

```bash
EOS_DEFAULT_EMBEDDER_RELEASE_CORKSCREWDB_ROOT=/home/draco/work/corkscrewdb \
ferrous-wheel run scripts/smoke_eos_default_embedder_release.fw
```

The release smoke verifies the checked-in artifact SHA256 `e0eca16ff34ebb88ca96862d58c3ac7f02dbf4b124599fdf96f25344ac02e408`, tokenizer SHA256 `64cf63223cb3f97125040677a573e6ab6c625cff1f6f338f4e680a4c9f7a42f5`, package metadata, local `embed-text` load as `f16[256]`, and, when CorkScrewDB is supplied, the temp DB manifest embedding id `corkscrewdb-default-embedder` dim `256`.

The JSON/TSV rows include the dense float32 reference, TurboQuant bit width, nDCG@10/nDCG@100, MRR@10, precision@1/5/10, hit@1/5/10, MAP@10/MAP@100, recall@10/100, quality deltas, vector bytes, rerank storage, rerank sidecar bytes, total vector bytes, compression ratio, total compression ratio, quantization docs/s, direct IP scoring throughput, per-query scoring latency summaries, and optional rerank overfetch/score counts. CorkScrewDB integration is a separate local smoke for this gate; these are the CorkScrewDB-relevant storage and scoring metrics that must stay attached to a promoted default embedder.

Interpret the metric groups separately for promotion. Quality metrics decide whether the candidate ranks relevant documents well enough: nDCG and MAP capture ordering, precision/hit@k capture first-screen success, and recall@100 captures candidate-pool coverage. Compression metrics decide whether q4/q8 are worth the footprint reduction. Throughput metrics are path-specific: sealed Eos evals can include local encoder time, while cached external-vector rows only measure cache load/scoring and do not represent live provider or external model encoding throughput.

For the productized CorkScrewDB default-embedder thesis around many compact child vectors per parent object, including exact fit boundaries and current evidence numbers, see [TurboQuant Multi-Vector Frontier](turboquant-multivector-frontier.md). Keep that thesis separate from the q4/fp16 sidecar rerank gate in this section.

For CorkScrewDB multi-vector and time-series designs, use the storage planner before running quality experiments:

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

For time-series/window-vector planning, derive the child-vector count from point counts, window size, and stride instead of hand-entering `--vectors-per-object`:

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

This uses one vector per covering window: a series no longer than the window gets one vector, and longer series include a tail window when the stride does not land exactly on the end. Do not pass `--vectors-per-object` with `--series-lengths`; the CLI fails explicit conflicts so manual and derived modes stay distinct.

This planner answers a different question from `eval-retrieval-turboquant`: how many direct quantized child vectors per parent object fit in the storage cost of one dense fp32 baseline vector. It is a storage gate, not a numeric time-series quality benchmark. The output is TSV by default and optional JSON with fields including `dim`, `baseline_dim`, `bits`, `objects`, `vectors_per_object`, optional `series_length`/`window_size`/`window_stride`/`derived_window_count`, `dense_parent_bytes`, `dense_baseline_bytes`, raw `quantized_vector_bytes`, `vector_overhead_bytes`, `dense_vector_storage_bytes`, `quantized_vector_storage_bytes`, `total_quantized_bytes`, packed-parent fields such as `packed_object_overhead_bytes`, `packed_quantized_storage_bytes`, `packed_total_quantized_bytes`, `packed_vectors_that_fit_in_one_dense_vector`, and compression ratios. Omitting `--baseline-dim` preserves same-dim accounting (`baseline_dim=dim`); with `32` bytes of packed parent-object overhead, same-dim packed 128d children fit only 14 q2, 7 q4, or 3 q8 children. Omitting `--vector-overhead-bytes` preserves ideal payload-only accounting for the current per-child-entry model; omitting `--packed-object-overhead-bytes` makes the packed-parent design target payload-only. Use `--dim 128 --baseline-dim 3072` to test the large-baseline interpretation where one 3072d fp32 vector has a 12,288-byte payload, q2 128d children have a 36-byte raw payload, and 128 payload-only children use about `0.375x` of the baseline budget. With `--packed-object-overhead-bytes 32`, a 3072d dense parent-vector storage budget fits 341 q2, 180 q4, or 93 q8 128d packed children; current per-child-entry q2/q4/q8 fit counts are lower because the packed layout pays object overhead once per parent and keeps child vectors as compact TurboQuant payloads. Pair this planner with `scripts/smoke_corkscrewdb_child_vectors.fw` in `packed_parent_multivectors` layout when measured CorkScrewDB packed parent-object persistence is required. The intended lane is direct child vectors for windows, spans, event histories, or per-object time-series slices. Do not enable `--sidecar-storage fp16` for that lane unless the product explicitly accepts the extra storage; fp16 sidecars are for quality-preserving rerank profiles and erase most of the hundred-child storage advantage when attached to every child vector.

Use the executable budget-frontier smoke when the claim needs a repeatable artifact instead of an inline planner command:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_multivector_budget_frontier.fw
```

The default smoke runs the planner for `128d` compact children against a `3072d` dense baseline with q2/q4/q8, child counts `1,16,64,100,128,181,256,341`, `32` bytes of current per-vector overhead, `32` bytes of packed parent-object overhead, no sidecar, and `1000` objects. Set `EOS_MV_BUDGET_SMOKE_BASELINE_DIMS=128,384,768,1024,1536,3072` to measure the same compact-child shape against multiple dense parent dimensions in one run; this comma-list takes precedence over the backward-compatible singular `EOS_MV_BUDGET_SMOKE_BASELINE_DIM`. It writes `summary.tsv`, `manifest.json`, planner JSON, and command logs under `runs/eos-multivector-budget-frontier-smoke-<timestamp>/`; the default gates assert current per-child-entry fit counts q2 >= `181`, q4 >= `100`, q8 >= `64`, and packed-parent target fit counts q2 >= `341`, q4 >= `180`, q8 >= `93`. This is storage/accounting only, not retrieval quality, not CorkScrewDB API latency, and not measured DB-directory cost.

Use the use-case frontier smoke when product examples need exact planner rows:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_multivector_usecase_frontier.fw
```

The default scenarios cover the same-dim 100-child control, 100-child document spans, 100 time-series windows derived from `1648/64/16`, a 180-child event/trace timeline, and a 341-child q2 frontier. The smoke writes `summary.tsv`, `manifest.json`, planner JSON, and logs under `runs/eos-multivector-usecase-frontier-smoke-<timestamp>/`; gates require same-dim 100-child q2/q4 packed rows to fail, 3072d-baseline 100-child q2/q4 packed rows to fit, 180-child packed q4 to fit at the edge, and 341-child packed q2 to fit. q8 rows are present for contrast but are not required to fit.

For a synthetic event/trace cache-only proof lane, use `eos export-event-trace-vectors` or run `scripts/smoke_eos_event_trace_vectors.fw`. The exporter accepts one parent trace per JSONL row with `id` or `_id` and an `events` array, renders each event into a deterministic child vector text row, writes BEIR helper files, and feeds `eval-retrieval-multivector-turboquant`. This is text-rendered synthetic event evidence only; real incident/session retrieval needs a real workload and separate CorkScrewDB API measurement. Local flat packed-parent API evidence is recorded in `runs/eos-corkscrewdb-event-trace-packed-q4-180-20260616T000000Z/` against the separate-child comparison `runs/eos-corkscrewdb-event-trace-separate-q4-180-20260616T000000Z/`, with packed/separate DB bytes `0.359266x`. The single-parent-vector baseline on the same inputs is `runs/eos-corkscrewdb-event-trace-single-parent-q4-180-20260616T000000Z/`: it stores `5` mean-pooled parent vectors, records DB bytes `1,717`, DB directory multiple `0.027946x`, p95 `0.050206ms`, and `ndcg@10=1.000000` / `recall@100=1.000000`; treat it as a storage/quality baseline that compresses away child-level event facets.

Use the budget-quality smoke when the claim must include both cache-only retrieval quality and overhead-aware planner capacity:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_multivector_budget_quality.fw
```

The default run uses the local Eos 128d SciFact child cache in `runs/eos-128d-child-cache-quality-20260615T000000Z/`, evaluates dense/q2/q4/q8 max-child parent retrieval, then plans the inferred 128d child vectors against one 3072d dense baseline vector with `32` overhead bytes and no sidecar. S40-era predecessor interpretation: q4 stays near dense on this cache (`ndcg@10` drop about `0.002630`, `recall@100` drop about `0.001667`) and the planner fits `123` overhead-aware q4 children per dense-vector budget, so the `100` child-vector target fits. This remains cache-only evidence; it does not measure CorkScrewDB API latency, HNSW/index behavior, remote mode, or DB-directory size, and should be refreshed before using it as new-default evidence.

Use the CorkScrewDB budget-quality smoke when the same compact-cache claim must pass through the actual local flat CorkScrewDB API:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_corkscrewdb_budget_quality.fw
```

The default run feeds the Eos 128d SciFact child cache and qrels into `scripts/smoke_corkscrewdb_child_vectors.fw` with q4 overfetch `100,12468`, then joins those `PutVector`/`SearchVector` metrics to `plan-multivector-storage` with `--dim 128 --baseline-dim 3072 --vector-overhead-bytes 32 --sidecar-storage none`. It writes a combined `summary.tsv`, `manifest.json`, command logs, nested `corkscrewdb/` run artifacts, and `planner.json` under `runs/eos-corkscrewdb-budget-quality-smoke-<timestamp>/`. Historical API evidence: q4 overfetch100 has `ndcg@10=0.407586`, `recall@100=0.724111`, DB directory multiple `0.048935x`, and p95 search latency about `11.8ms`; q4 full overfetch has `recall@100=0.741889`; the planner fit is `123` q4 children, so the 100-child target fits. The packed parent variant can now pass through `EOS_CORKSCREW_BUDGET_QUALITY_PACKED_METADATA_MODE` and `EOS_CORKSCREW_BUDGET_QUALITY_PACKED_CHILD_ID_MODE`. S40-era packed-parent API run `runs/eos-s40-current-default-corkscrewdb-budget-quality-packed-q4q8-main-20260620T165050Z/` used fresh then-checked-in-default vectors in `runs/eos-vector-caches/eos-s40-current-default-scifact-child-w128-o32-128d/` and CorkScrewDB main commit `511f5d24408d9aeba21941954d29cca3569875da`. It measured `5,183` parents, `12,468` children, and `128d` vectors through local flat `PutMultiVector`/exact `SearchParentsVector` with `packed_parent_multivectors`, `packed_metadata_mode=none`, `packed_child_id_mode=ordinal`, `quantized_only` storage, and flat index: q4 release-gate evidence recorded nDCG@10 `0.452971`, recall@100 `0.755222`, DB directory multiple `0.041675x`, vector payload multiple `0.013312x`, p95 `13.434893ms`, planner fit `180`, `target_child_fit=true`, and target storage multiple `0.554545x`; q8 diagnostic evidence recorded nDCG@10 `0.472424`, recall@100 `0.776889`, DB directory multiple `0.066733x`, vector payload multiple `0.025841x`, p95 `21.874919ms`, planner fit `93`, `target_child_fit=false`, and target storage multiple `1.074026x`. The nf005 and targeted-v3 packed-parent local flat runs remain historical comparison/provenance only. Treat this as s40-era local flat CorkScrewDB API evidence, separate from q4/fp16 sidecar rerank evidence; refresh before claiming API-level quality for the new default. It is not remote mode, federation, HNSW, hosted parity, q4/fp16 alias promotion, or native Matryoshka evidence. q8 misses the 100-child target fit and DB directory gate.

Use `scripts/smoke_eos_corkscrewdb_hybrid_policy.fw` when the selected Eos hybrid policy must be demonstrated through CorkScrewDB's public dense+sparse API. The smoke writes a temporary Go module with a local `replace` to `/home/draco/work/corkscrewdb`, exports or reuses Eos BEIR document/query vectors, loads CorkScrewDB with `WithSparse`, and queries with `SearchMulti` plus `WeightedFusion{Dense:0.5,Sparse:0.5}`. `EOS_CORKSCREW_HYBRID_SMOKE_SPARSE_METHOD=tfidf_l2` preserves the historical lexical TF-IDF sparse channel, while `bm25_dot` encodes doc sparse values as BM25 `idf * tfWeight` (`k1=0.9`, `b=0.4`, avgdl over loaded docs) and query values as query term frequency.

Historical TF-IDF sparse translation rows:

| Run | Bits | Docs | Queries | Relevant pairs | Dense-only nDCG@10 / recall@100 | Hybrid nDCG@10 / recall@100 | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `runs/eos-corkscrewdb-hybrid-policy-smoke-20260620T232130Z/` | 8 | 600 | 20 | 22 | 0.729755 / 0.950000 | 0.928080 / 1.000000 | pass |
| `runs/eos-corkscrewdb-hybrid-policy-full-scifact-v1-20260620T232832Z/` | 8 | 5,183 | 300 | 339 | 0.565375 / 0.796444 | 0.703644 / 0.934556 | pass |
| `runs/eos-corkscrewdb-hybrid-policy-full-scifact-q4-v1-20260620T000000Z/` | 4 | 5,183 | 300 | 339 | 0.550000 / 0.799111 | 0.695287 / 0.936222 | pass |
| `runs/eos-corkscrewdb-hybrid-policy-full-nfcorpus-q4-v1-20260620T000000Z/` | 4 | 3,633 | 323 | 12,334 | 0.203265 / 0.241053 | 0.301358 / 0.293417 | pass |
| `runs/eos-corkscrewdb-hybrid-policy-full-fiqa-q4-v1-20260620T000000Z/` | 4 | 57,600 | 648 | 1,705 | 0.119637 / 0.345019 | 0.189401 / 0.482542 | pass |
| `runs/eos-corkscrewdb-hybrid-policy-full-scifact-q2-v1-20260620T000000Z/` | 2 | 5,183 | 300 | 339 | 0.472358 / 0.760889 | 0.688250 / 0.924556 | pass |

These are API-level lexical sparse translation rows with TF-IDF sparse features. They remain useful compatibility evidence, but they are not exact `eval-retrieval-hybrid` BM25 parity. The q2 row passes the hybrid-vs-dense API gate, but its direct dense-only score is materially lower than q8, so treat q2 as compact diagnostic evidence rather than a default-quality direct-vector profile.

BM25-dot sparse q4 API rows:

| Run | Bits | Docs | Queries | Relevant pairs | Dense-only nDCG@10 / recall@100 | Hybrid nDCG@10 / recall@100 | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `runs/eos-corkscrewdb-hybrid-policy-full-scifact-q4-bm25dot-v1-20260620T000000Z/` | 4 | 5,183 | 300 | 339 | 0.548851 / 0.793111 | 0.726845 / 0.936222 | pass |
| `runs/eos-corkscrewdb-hybrid-policy-full-nfcorpus-q4-bm25dot-v1-20260620T000000Z/` | 4 | 3,633 | 323 | 12,334 | 0.201400 / 0.233861 | 0.311750 / 0.291122 | pass |
| `runs/eos-corkscrewdb-hybrid-policy-full-fiqa-q4-bm25dot-v1-20260620T000000Z/` | 4 | 57,600 | 648 | 1,705 | 0.118427 / 0.341251 | 0.223099 / 0.499790 | pass |

S40-era BM25-dot SearchMulti rows for `assets/corkscrewdb-default-embedder/corkscrewdb-default-embedder.mll`, old SHA256 `f494915a0d78b24205d5018bb701bf40cabbedee4bc8b96b6a1920b19131da5a`, run root `runs/eos-s40-current-default-hybrid-policy-v1-20260621T103500Z/`:

| Dataset | Docs | Queries | Dense-only nDCG@10 / recall@100 | Hybrid nDCG@10 / recall@100 | overall_pass |
| --- | ---: | ---: | ---: | ---: | --- |
| SciFact | 5,183 | 300 | 0.560000301382 / 0.799777777778 | 0.724378172602 / 0.932888888889 | true |
| NFCorpus | 3,633 | 323 | 0.205106071529 / 0.242729344406 | 0.315090347196 / 0.293990495284 | true |
| FiQA | 57,600 | 648 | 0.121205765861 / 0.352411233314 | 0.223742712175 / 0.504002444743 | true |

The BM25-dot rows are BM25-compatible sparse-dot API evidence: the sparse dot product matches the in-repo BM25 scoring formula for matching terms modulo float32 rounding, then CorkScrewDB applies its public `WeightedFusion` min-max normalization. This predecessor run is local lexical+dense ranking-policy evidence over the old s40 dense asset. These rows are not a dense default asset change or CorkScrewDB default alias change, and they do not prove remote mode, HNSW, federation, or hosted-model parity. Refresh them before claiming hybrid/API coverage for the new default.

For a real SciFact document-span layout comparison, set `EOS_CORKSCREW_BUDGET_QUALITY_LAYOUT=packed_parent_multivectors` or `single_parent_vectors` and point the wrapper at `runs/eos-vector-caches/eos-embed-v1-scifact-child-w128-o32-full/scifact/`. Verified q4/q8 diagnostic runs on `5,183` parents, `12,468` child spans, `300` queries, and `256d` vectors show the expected tradeoff. Packed `none`/`ordinal` run `runs/eos-real-scifact-full-packed-none-ordinal-q4q8-diagnostic2-20260616T000000Z/` recorded q4/q8 nDCG@10 `0.449435`/`0.461862`, recall@100 `0.773111`/`0.774778`, DB directory multiples `0.032900x`/`0.057958x`, and p95 `15.934540ms`/`30.366188ms`. Single-parent run `runs/eos-real-scifact-full-single-parent-q4q8-diagnostic-20260616T000000Z/` recorded q4/q8 nDCG@10 `0.406498`/`0.422597`, recall@100 `0.743111`/`0.745889`, DB directory multiples `0.022411x`/`0.032827x`, and p95 `7.075771ms`/`13.242884ms`. Use this as local flat document-span evidence that packed child multivectors preserve span evidence better than mean-pooled parent vectors, at higher DB size and latency. The run disabled old q4 floors for comparison; it is not remote, HNSW, federation, or native Matryoshka evidence.

Use the HNSW/raw CorkScrewDB budget-quality smoke when the same compact cache must validate HNSW index search rather than compact persistence:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_eos_corkscrewdb_hnsw_quality.fw
```

The HNSW wrapper feeds the same Eos 128d SciFact child cache into `scripts/smoke_corkscrewdb_child_vectors.fw` with `index_type=hnsw`, `vector_storage=raw`, and a post-load HNSW rebuild, then joins the same q4 planner row. Current promoted narrow-default HNSW evidence is `runs/eos-current-default-corkscrewdb-hnsw-refresh-v1-20260623T055809Z`, using explicit current cache paths from `runs/eos-vector-caches/eos-narrow-default-scifact-child-w128-o32-128d`. q4 overfetch100 has nDCG@10 `0.445689`, recall@100 `0.735222`, raw HNSW DB directory multiple `0.243494x`, vector payload multiple `0.013312x`, p95 `2.324070ms`, planner fit `123`, `target_child_fit=true`, and target storage multiple `0.811688x`; q4 overfetch12468 has nDCG@10 `0.452380`, recall@100 `0.758556`, and p95 `49.635238ms`. The q4 rebuild took about `399.435824s`; q8 was not run. CorkScrewDB rejects HNSW with `quantized_only`, so this is raw-vector index-search validation. It intentionally does not claim the flat quantized-only DB-size multiple; use the vector payload/planner columns only for compact q4 planning comparisons. This is local API evidence only, not remote mode, federation, hosted parity, or a service SLO.

Use the time-series/window CorkScrewDB smoke when the child vectors should be generated from synthetic parent series and then proven through the actual local flat API:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_eos_corkscrewdb_timeseries_windows.fw
```

The wrapper runs `scripts/smoke_eos_timeseries_window_vectors.fw` into a nested `timeseries/` directory, then loads the generated window child vectors into CorkScrewDB with flat `quantized_only` persistence. Its joined summary shows parent count, child-window count, derived windows per parent, q4/q8 quality, planner fit, DB directory multiple, vector payload multiple, and p95 latency. S40-era result: `5` parents, `25` child windows, `5` derived windows per parent; q4 flat/quantized_only records `ndcg@10=1.000000`, `recall@100=1.000000`, planner fit `123`, vector payload multiple `0.034180x`, and DB directory multiple `0.117643x`; q8 records `ndcg@10=0.926186`, `recall@100=1.000000`, planner fit `75`, vector payload multiple `0.060221x`, and DB directory multiple `0.143685x`. Observed p95 latency for the default synthetic smoke has been sub-ms, but it varies by run. This is local `PutVector`/`SearchVector` evidence for synthetic text-rendered numeric windows only, not remote/federation/HNSW and not a trained numeric time-series encoder. Keep measured DB directory size separate from planner/vector payload accounting, and refresh before using it as new-default evidence.

For the q4-only parent-variant stress shape that makes the direct child-vector budget concrete while scaling parent objects enough to separate fixed DB-directory overhead from per-vector TurboQuant payload accounting, run:

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

That 1648-point series length with 64-point windows and stride 16 derives exactly 100 child windows per parent. With `EOS_CORKSCREW_TS_WINDOW_SERIES_VARIANTS=20`, the five query patterns expand to `100` parent series and `10,000` child windows while keeping the original five pattern queries; each query has `20` relevant parent variants, for `100` relevant query-parent pairs. Verified separate-child run `runs/eos-corkscrewdb-timeseries-window-scale-q4-100-variants20-20260616T000000Z/` recorded q4 flat/`quantized_only` with planner fit `123`, planner storage multiple `0.811688x`, vector payload multiple `0.683594x`, measured DB directory multiple `2.293748x`, DB bytes `2,818,558`, `ndcg@10=0.352927`, `recall@100=0.560000`, p95 `11.948319ms`, and overfetch `500`.

The same scaled q4 shape now has packed parent-object evidence. Run `EOS_CORKSCREW_SMOKE_LAYOUT=packed_parent_multivectors` with flat `quantized_only` storage to write one parent via `PutMultiVector` and query exact local flat parents with `SearchParentsVector`. The strict storage-frontier mode uses `EOS_CORKSCREW_SMOKE_PACKED_METADATA_MODE=none` or `minimal` and `EOS_CORKSCREW_SMOKE_PACKED_CHILD_ID_MODE=ordinal`; source child IDs and full metadata are richer product modes with materially larger measured DB envelopes. Verified `none`/`ordinal` run `runs/eos-corkscrewdb-timeseries-window-scale-q4-100-variants20-packed-minimal-20260616T000000Z/` recorded `parent_insert_count=100`, `parent_search_exact=true`, `overfetch_children=0`, the same `0.683594x` vector payload multiple, DB directory multiple `0.844660x`, DB bytes `1,037,918`, `ndcg@10=0.352927`, `recall@100=1.000000`, and p95 `5.916649ms`. DB bytes were `0.421237x` of the prior full-metadata packed run and `0.368244x` of the separate-child run. The recall improvement is an exact parent-rollup result versus the separate-child run's bounded overfetch, not a trained encoder quality gain. Corrected q2-341 packed-parent frontier evidence is now unified in wrapper run `runs/eos-corkscrewdb-timeseries-window-q2-341-compact-v5-unified-20260616T000000Z/`, which generated child/query/qrels inputs, packed planner evidence, and measured DB directory evidence against CorkScrewDB commit `c208f9b50d29f9fdf19771c4b093332c7c8fd0b4`. The shape is q2 `128d`, `100` parents, `34,100` child windows, `341` windows per parent, `packed_parent_multivectors`, `packed_metadata_mode=none`, `packed_child_id_mode=ordinal`, `quantized_vector_bytes=36`, `quantized_child_bytes=1,227,600`, vector payload multiple `0.9990234375x`, packed planner bytes `12,308`, packed planner multiple `0.999025974025974x`, measured DB directory bytes `1,237,818`, DB directory multiple `1.0073388671875x`, `ndcg@10=0.4493940305106442`, `recall@100=1.000000`, and p95 `1.418733ms`; it passed planner-fit, vector-payload, DB-directory, and p95 gates. Use the q2 row for near-one vector payload and packed planner accounting; with CorkScrewDB compact snapshot v5 ordinal encoding, the persisted DB directory is approximately one dense parent-vector budget for this strict shape. Without that compact snapshot path, or for richer child records, keep DB directory cost separate. The single-parent-vector baseline on the same scaled inputs is `runs/eos-corkscrewdb-timeseries-window-scale-q4-100-variants20-single-parent-20260616T000000Z/`: it stores `100` mean-pooled parent vectors, records DB bytes `23,578`, DB directory multiple `0.019188x`, vector payload multiple `0.006836x`, `ndcg@10=0.616983`, `recall@100=1.000000`, and p95 `0.140022ms`; it is cheaper because it no longer preserves the `10,000` child-window vectors.

The release-readiness sensitivity matrix is `runs/eos-packed-parent-storage-sensitivity-matrix-20260616T000000Z/`, measured against CorkScrewDB commit `c208f9b50d29f9fdf19771c4b093332c7c8fd0b4`. It keeps planner bytes, vector payload bytes, and measured DB directory bytes separate. For q2/341 time-series (`100` parents, `34,100` children), `none`/`ordinal` (`1.007334x`, p95 `1.937486ms`) and `minimal`/`ordinal` (`1.009744x`, p95 `6.228571ms`) pass; `none`/`source` and `minimal`/`source` are about `2.74x`, while full metadata reaches `5.17x` to `5.92x`. For q4/100 time-series (`100` parents, `10,000` children), `none`/`ordinal` (`0.561698x`, p95 `6.849447ms`) and `minimal`/`ordinal` (`0.564106x`, p95 `9.298850ms`) pass; source child IDs cross the one-budget line at about `1.07x`, and full metadata is `1.78x` to `2.01x`. The SciFact compact q4 rows remain far under one DB budget across metadata and child-ID modes, but they average only `2.405557` children per parent and should not be cited as high-child-count proof.

These are synthetic text-rendered local flat CorkScrewDB smokes: they improve DB overhead measurement by scaling parent objects and make the strict ordinal/none-or-minimal packed parent-object storage win durable, but they are not remote/federation/HNSW evidence, production quality evidence, or a final numeric time-series encoder claim. The q4-only profile keeps the DB directory threshold disabled by default and gates the vector-payload accounting instead.

To move from storage math to a cache-only quality harness for time-series windows, export text-rendered numeric windows and reuse the existing multivector TurboQuant evaluator. The series JSONL has one parent series per row with `id` or `_id` and numeric `values`; qrels must use the parent series IDs as corpus IDs:

```bash
go run ./cmd/eos export-timeseries-vectors \
  --dataset sensor-window-retrieval \
  --batch-size 64 \
  --output-dim 128 \
  --window-size 64 \
  --window-stride 16 \
  --manifest-json /data/manta/runs/<run-id>/sensor-window-export-128d.json \
  /data/manta/runs/<run-id>/eos-embed-v1.sealed.mll \
  /data/manta/timeseries/sensor-series.jsonl \
  /data/manta/timeseries/queries.jsonl \
  /data/manta/runs/<run-id>/sensor-window-cache-128d
```

```bash
go run ./cmd/eos eval-retrieval-multivector-turboquant \
  --dataset sensor-window-retrieval \
  --backend text-rendered-timeseries-windows \
  --artifact eos-embed-v1-prefix128 \
  --doc-vectors /data/manta/runs/<run-id>/sensor-window-cache-128d/child-doc-vectors.jsonl \
  --query-vectors /data/manta/runs/<run-id>/sensor-window-cache-128d/query-vectors.jsonl \
  --qrels /data/manta/timeseries/qrels/test.tsv \
  --bits 2,4,8 \
  --baseline-dim 3072 \
  /data/manta/runs/<run-id>/sensor-window-cache-128d
```

`export-timeseries-vectors` renders each window as stable text containing the series ID, window bounds, values, and simple stats before embedding with the normal Eos text embedder path. It also writes BEIR helper `corpus.jsonl` and `queries.jsonl` files into the output directory, so use the export directory as the evaluator dataset directory and pass the parent-series qrels with `--qrels`. This is a bridge quality harness for numeric windows represented as text, not a final trained numeric time-series encoder or a CorkScrewDB load/index/search benchmark.

For a reproducible local smoke of that bridge, run:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/smoke_eos_timeseries_window_vectors.fw
```

The smoke generates five synthetic 12-point parent series, exports 128d text-rendered windows with `--window-size 4 --window-stride 2`, evaluates dense/q2/q4/q8 parent-child retrieval, and runs the storage planner with `--baseline-dim 3072 --vector-overhead-bytes 32`. It writes `summary.tsv` and `manifest.json` under `runs/eos-timeseries-window-vector-smoke-<timestamp>` so the claim is concrete: on this synthetic cache, q8 should match dense quality within the configured tolerance while using a small fraction of a large dense baseline-vector budget. Set `EOS_TS_WINDOW_SMOKE_ARTIFACT` if the default sealed artifact is not present.

After storage planning, run the cache-only parent-child quality harness before making product claims:

For Eos-owned/default embedders, create the cache with the Go-native exporter so the sealed `.mll` runtime path, tokenizer, batching, and normalization match `eval-retrieval` without Python or SentenceTransformers:

```bash
go run ./cmd/eos export-retrieval-vectors \
  --dataset scifact \
  --batch-size 64 \
  --output-dim 128 \
  --document-chunk-words 128 \
  --document-chunk-overlap 32 \
  --document-chunk-min-words 16 \
  --manifest-json /data/manta/runs/<run-id>/scifact-child-vector-export-128d.json \
  /data/manta/runs/<run-id>/eos-embed-v1.sealed.mll \
  /data/manta/datasets/eos-embed-v1/raw/scifact \
  /data/manta/runs/<run-id>/scifact-child-cache-128d
```

Without `--document-chunk-words`, the exporter writes `doc-vectors.jsonl` and `query-vectors.jsonl` for `eval-retrieval-vectors` or `eval-retrieval-vectors-turboquant`. With chunking enabled, it writes `child-doc-vectors.jsonl` and `query-vectors.jsonl`; child IDs are deterministic `parent#chunk-0000` word windows and the manifest records document count, query count, child-vector count, written dimension, model dimension, backend, artifact, and output paths. `--output-dim 128` is a prefix-truncated compact cache followed by L2 renormalization. It is a measurement bridge for storage/quality evidence, not a trained Matryoshka or native 128d projection head; a trained compact head is the stronger future path before promotion claims.

```bash
go run ./cmd/eos eval-retrieval-multivector-turboquant \
  --dataset scifact \
  --backend child-cache \
  --artifact <embedder-label> \
  --doc-vectors /data/manta/runs/<run-id>/scifact-child-cache-128d/child-doc-vectors.jsonl \
  --query-vectors /data/manta/runs/<run-id>/scifact-child-cache-128d/query-vectors.jsonl \
  --bits 2,4,8 \
  --baseline-dim 3072 \
  --metrics-json /data/manta/runs/<run-id>/multivector-turboquant-128d.json \
  --metrics-tsv /data/manta/runs/<run-id>/multivector-turboquant-128d.tsv \
  /data/manta/datasets/eos-embed-v1/raw/scifact
```

This evaluator treats qrels document IDs as parent IDs. Each document-vector JSONL row may include `parent_id`, `child_id`, and `vector`/`embedding`/`values`; absent `parent_id` falls back to `id` or `_id` for one-vector compatibility. The dense reference scores all child vectors and rolls them up by max score per parent, then direct TurboQuant rows quantize child vectors and perform the same max-child parent aggregation. Runs fail by default when any qrels-relevant parent is absent from the child-vector cache; `--allow-missing-relevant` preserves the old filtered behavior only for incomplete-cache diagnostics. TurboQuant rows use a deterministic IP quantizer seed, configurable with `--quantizer-seed`, and metrics record both `allow_missing_relevant` and `quantizer_seed`. Pass `--baseline-dim` when the storage claim is compact child vectors versus a larger dense embedder budget; leaving it at `0` uses the child dimension and is only same-dimension accounting. The metrics report `baseline_dim`, parent/child counts, average and max children per parent, dense baseline bytes, dense child bytes, per-vector quantized bytes, total quantized child bytes, dense-child compression, vectors that fit in one dense baseline, storage multiple relative to one dense baseline vector per parent, scored child pairs, quality metrics/deltas, scores/s, and per-query latency. Follow it with `scripts/smoke_corkscrewdb_child_vectors.fw` when the CorkScrewDB `PutVector`/`SearchVector` path itself must be proven.

For hosted or open external embedders, export BEIR-aligned `doc-vectors.jsonl` and `query-vectors.jsonl` caches and run both `eos eval-retrieval-vectors` and `eos eval-retrieval-vectors-turboquant`. `scripts/export_qwen3_retrieval_vectors.py` is the first provider-boundary exporter for the leading Qwen3 family baseline:

```bash
python3 scripts/export_qwen3_retrieval_vectors.py \
  --dataset-dir datasets/eos-embed-v1/raw/scifact/scifact \
  --output-root runs/external-vector-caches/qwen3-0.6b \
  --dataset-name scifact \
  --model-name Qwen/Qwen3-Embedding-0.6B
```

For parent-child multi-vector runs, keep the same provider boundary and add chunking flags. The exporter writes `child-doc-vectors.jsonl` with `parent_id`, deterministic `child_id`, and `embedding`, while still writing the normal `query-vectors.jsonl`:

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

Current SciFact child-cache evidence uses `datasets/manta-embed-v1/raw/scifact/scifact`, because `datasets/eos-embed-v1/raw/scifact/scifact` was absent for the local runs. With `128` word chunks, `32` overlap, `16` minimum trailing words, `5,183` parents, `12,468` child vectors, and strict qrels coverage (`allow_missing_relevant=false`), `mixedbread-ai/mxbai-embed-large-v1` is stronger than Qwen3 0.6B at dense, q2, q4, and q8 child-max retrieval. Its q8 row scored `0.747799` nDCG@10 / `0.966667` recall@100, beating Qwen3 q8 by `+0.031489` nDCG@10 and `+0.013334` recall@100. The sealed Eos artifact `runs/manta-embed-v1-deephard-full-ft-20260610T0000Z/manta-embed-v1.sealed.mll` now has full Go-native child-cache evidence on the same strict lane: export counts were `5,183` docs, `300` queries, `12,468` children, dim `256`, CUDA backend, and `57.771s`; strict eval used `339` relevant pairs, `3,740,400` scored child pairs, and seed `5581486560434873699`. Eos q8 scored `0.461862` nDCG@10 / `0.774778` recall@100 at `3.94x` compression and `0.61x` parent budget, closely preserving Eos dense-child `0.462489` / `0.778111` but remaining materially below mxbai and Qwen3. Treat this as pipeline proof, not default-quality promotion evidence.

Use `--backend` and `--artifact` labels to keep BGE, Qwen, Jina, Voyage, Cohere, OpenAI, and sealed Eos rows comparable. In the scoreboard harness, set `EOS_SCOREBOARD_EXTERNAL_VECTOR_TURBOQUANT=1` and `EOS_SCOREBOARD_EXTERNAL_VECTOR_TURBOQUANT_BITS=2,4,8` to append q2/q4/q8 rows for the same cache; add `EOS_SCOREBOARD_EXTERNAL_VECTOR_TURBOQUANT_RERANK_OVERFETCH=200` and `EOS_SCOREBOARD_EXTERNAL_VECTOR_TURBOQUANT_RERANK_STORAGE=fp16` when fp16 sidecar rerank rows should be emitted for an external cache. External baseline rows remain `not_scored` until the dense and TurboQuant cache metrics are present.

Current external comparison state: Qwen3 is locally consolidated for SciFact, NFCorpus, and full exportable-text FiQA. Its useful compact external row is direct q8 at about `3.98x` vector compression, with SciFact q8 nDCG@10 `0.704128`, NFCorpus q8 nDCG@10 `0.368763`, and FiQA q8 nDCG@10 `0.449614`. mxbai remains stronger than Qwen3 in the existing local short-set evidence. Qwen3 FiQA is not raw-row-complete or judged-coverage complete because one judged test document had empty unexportable text and was skipped.

For targeted quality-frontier mining, first emit per-query diagnostics with `--per-query-jsonl` for selected Eos and an external vector cache. Then run `scripts/mine_retrieval_quality_frontier.py` to produce machine-readable Eos-underperformance rows and optional text hard-negative JSONL. Use `--method`/`--bits` for symmetric multimethod filtering, or `--eos-method`/`--teacher-method` plus `--eos-bits`/`--teacher-bits` when comparing asymmetric surfaces such as direct or diagnostic-fusion Eos rows against q4 external teacher rows. Treat the output as protected candidate training data; it still needs the normal selected-vs-anchor gates on SciFact, NFCorpus, and FiQA before any model promotion.

The current dense-quality reranker-teacher frontier is request-only, not training input. `runs/eos-narrow-default-reranker-teacher-frontier-audit-v1-20260623T065159Z/` produced the older NFCorpus train/dev hard-negative frontier and proved the scorer path: `cmd/teacher-bridge --mode tei-rerank` can score TEI `/rerank`, and RTX50/Blackwell hosts should use `ghcr.io/huggingface/text-embeddings-inference:120-1.9`. The broader selected-diverse BGE audit at `runs/eos-nfcorpus-broader-cleaned-mining-quality-signal-v1-20260623T101157Z/` remains historical audit-only evidence: `BAAI/bge-reranker-large` was strongest on that slice at `65/128` (`0.507812`), but agreement and margins were too weak for training. The MiniLM control did not change that decision.

The repaired NFCorpus subset64 BGE resmoke at `runs/eos-nfcorpus-repaired-subset64-bge-resmoke-v1-20260623T132032Z` improved the bounded diagnostic but did not change the production boundary. BGE base selected/any-positive top1 is `35/64 = 0.546875` with margin `0.216738` and norm entropy `0.980823`; BGE large is `33/64 = 0.515625` with margin `0.215547` and norm entropy `0.983283`; BGE v2-m3 is `33/64 = 0.515625` with margin `0.194146` and norm entropy `0.979424`. Duplicate/qrels-positive negative leaks are zero and selected-positive equals any-positive top1; the repaired rows improve over the prior cleaned subset64 from base `29->35`, large `22->33`, and v2-m3 `10->33`. Keep these generated rows `request_only=true`, `train_allowed=false`, `no_train`, and audit-only; do not promote a source or train from them.

The cleaned full-frontier production decision record is `runs/eos-nfcorpus-cleaned-full-reranker-frontier-jina-audit-v1-20260623T113125Z/` plus the mxbai-large HF audit at `runs/eos-mxbai-rerank-large-hf-subset-audit-v1-20260623T140353Z`. Jina v2 rebuilt the NFCorpus frontier with doc-id provenance, selected `1188` examples (`932` train + `256` dev), scored `7128/7128` request rows, imported `1188/1188` examples with missing `0`, and failed teacher quality at `160/1188 = 0.134680` selected/any-positive top1. `BAAI/bge-reranker-large` was then scored on the same cleaned/provenance-safe full frontier, reusing the existing TEI cache with no download. It wrote `debug-bge-large-full-clean-v1/` and `data/nfcorpus.train-dev.cleaned-full-frontier.bge-large.*`, scored `7128/7128` request rows in `1:03.02` with `duplicates_skipped=0`, imported `1188` examples with missing `0`, and audited `1188` scored / `0` missing / `0` duplicate-positive negatives. BGE-large selected/any-positive top1 was lower than Jina at `142/1188 = 0.119529`, mean normalized entropy was `0.993488`, and the filter kept `142` while clearing `1046`. `mixedbread-ai/mxbai-rerank-base-v2` still cannot be served by TEI `120-1.9` on this host, but HF/SentenceTransformers scoring was mechanically unblocked in `runs/eos-mxbai-rerank-base-hf-unblock-smoke-v1-20260623T124344Z`. The tiny raw-logit smoke passed; the deterministic cleaned-frontier subset64 then scored/imported/audited/filtered with positive top1 `1/64 = 0.015625`, mean margin `-7.836426`, mean normalized entropy `0.476268`, filter kept `1`, and cleared `63`. Because that subset failed the `0.35` gate, full mxbai-base scoring was skipped and no full-frontier mxbai-base metric exists. `mixedbread-ai/mxbai-rerank-large-v2` used the repaired source subset from `runs/eos-nfcorpus-repaired-frontier-subset64-v1-20260623T131418Z/`; its raw-logit smoke passed at relevant `12.25` and irrelevant `-0.5625`, repaired subset64 passed at `37/64 = 0.578125`, mean margin `1.179688`, norm entropy `0.442453`, kept `37/64`, but full cleaned-frontier scoring covered `7128` request rows / `1188` examples with missing `0` and duplicate-positive negative candidates `0` and failed at `172/1188 = 0.144781`, mean margin `-2.897175`, norm entropy `0.612920`, kept `172/1188`. Mark the mxbai-base evidence `subset_quality_gate_failed` / `teacher_quality_blocked` / `no_train`; mark mxbai-large full-frontier evidence `teacher_quality_blocked` / `no_train`. Decision: `teacher_quality_blocked` / `no_train` for Jina, BGE-large, mxbai-base subset evidence, and mxbai-large full-frontier evidence on this cleaned frontier; all generated rows remain `request_only=true` and `train_allowed=false`, and dense training must not continue from them. Do not train from the old dirty full-frontier artifacts either.

Hybrid lexical+dense retrieval is a separate retrieval/rerank product surface, not a model-promotion shortcut. Before using hybrid rows as CorkScrewDB default-embedder evidence, select fusion parameters on a dev split and apply the selected setting unchanged to test:

```bash
EOS_REPO_ROOT=$PWD \
EOS_HYBRID_CAL_MODE=dense \
EOS_HYBRID_CAL_ARTIFACT=/path/to/eos-embed-v1.sealed.mll \
EOS_HYBRID_CAL_DATASET_ROOT=/data/manta/datasets/eos-embed-v1 \
EOS_HYBRID_CAL_DATASETS=fiqa \
EOS_HYBRID_CAL_DEV_SPLIT=dev \
EOS_HYBRID_CAL_TEST_SPLIT=test \
ferrous-wheel run scripts/calibrate_eos_embed_hybrid_retrieval.fw
```

For external vector caches, set `EOS_HYBRID_CAL_MODE=vectors`, `EOS_HYBRID_CAL_VECTOR_ROOT`, `EOS_HYBRID_CAL_VECTOR_BACKEND`, and optionally `EOS_HYBRID_CAL_VECTOR_ARTIFACT`. The calibration harness writes JSON, TSV, Markdown, per-setting metrics, per-query JSONL, and command logs under `runs/<run-id>/`. It selects by dev `ndcg_at_10`, tie-breaks by `mrr_at_10` then `recall_at_100`, reports dense and BM25 sanity checks, and records a protection gate showing whether selected test hybrid regresses against dense by more than `EOS_HYBRID_CAL_MAX_NDCG10_REGRESSION` or `EOS_HYBRID_CAL_MAX_RECALL100_REGRESSION`. Hybrid per-query `top_k` rows include optional component diagnostics for each fused candidate: source ranks, raw dense/BM25 scores, and minmax/zscore normalized component scores when available.

The sparse lexical hash-head sidecar has a separate experimental calibration harness: `scripts/calibrate_eos_sparse_lexical_head_hybrid.fw`. It uses `export-sparse-lexical-labels`, `fit-sparse-lexical-head`, `export-retrieval-vectors`, and `eval-sparse-lexical-head-vectors-hybrid` to calibrate split-specific hashed sparse-label fusion. This is distinct from BM25 `eval-retrieval-hybrid` calibration and remains non-default sidecar evidence only: it is not a trained neural sparse head, not dense model promotion, and not an asset or alias change.

The learned sparse lexical projection head is a newer experimental scaffold, exposed as `fit-sparse-lexical-projection-head` and `eval-sparse-lexical-projection-head-vectors-hybrid`. It uses sparse lexical labels only during fitting to learn hashed-bin dense-vector prototypes, then predicts sparse bins from dense document/query vector caches at evaluation time without loading eval-time sparse labels. The head records the fit split, but held-out evaluation can use a different qrels/corpus split as long as the dataset and vector dimensions are compatible. This separates it from the current split-specific hash-head evaluator, which scores exported sparse query labels directly. Treat projection-head rows as prototype-sidecar evidence only; they are not default retrieval behavior, not a shipped sparse head, and not promotion evidence without held-out quality gates. Prototype retention ranking is configurable with `--prototype-rank support|total_weight|avg_weight`; `support` is the default and preserves the original support-first policy, while weight-aware policies remain experimental probes. For projection-head sweeps, `--dense-candidates-only` enables an experimental recall-preserving rerank mode: the projected sparse score can reorder dense top-k candidates, but sparse-only candidates cannot displace dense tail candidates. This is non-default diagnostic evidence for recall-loss investigations.

The sparse lexical linear head is the first trainable sidecar scaffold in this lane, exposed as `fit-sparse-lexical-linear-head` and `eval-sparse-lexical-linear-head-vectors-hybrid`. It retains hashed sparse bins by support/weight policy, trains deterministic per-bin linear weights from L2-normalized dense document/query vector caches to sparse-label weights, and predicts sparse bins from dense vectors at evaluation time without loading eval-time sparse labels. It remains explicitly experimental and non-default: it is not a default embedder change, not a CorkScrewDB asset or alias change, and not promotion evidence without later held-out sweeps.

Updated projection-head capacity evidence is recorded under `runs/eos-projection-head-capacity-term-sweep-v1-20260622T094939Z/`, with weight-aware policy validation under `runs/eos-projection-head-weight-aware-ranking-v1-20260622T104100Z/` and full-FiQA validation under `runs/eos-projection-head-fiqa-full-validation-v1-20260622T105155Z/`. The strongest full rows remained under the default `support` prototype policy: SciFact full `p8192_t32 support zscore alpha0.25 protect5`, nDCG@10 `0.568415`, R@100 `0.796444`, dense `0.564538`; NFCorpus full `p8192_t32 support zscore alpha0.25 protect0`, nDCG@10 `0.210277`, R@100 `0.242059`, dense `0.205571`. FiQA lift was capped exploratory evidence only: `max_queries_100` `p512_t128 support alpha0.30`, nDCG@10 `0.104322`, R@100 `0.291298`, dense maxq100 `0.098636`. Full FiQA did not beat dense: the best full row was `p512_t32 support alpha0.25`, nDCG@10 `0.120537` versus dense `0.121261`, with recall@100 matching `0.351678`. Keep `support` as the default; `avg_weight` underperformed, and `total_weight` had only a tiny NFCorpus lift. These rows support recall-preserving projection reranking research only; they are not a default policy, promotion decision, or evidence that projected sparse candidates should enter the candidate set.

The full short-set sparse-head calibration run at `runs/eos-sparse-head-hybrid-calibration-shortset-full-v1-20260622T080053Z/` passed protection gates for SciFact, NFCorpus, and FiQA with fixed `hybrid_zscore_alpha0.25` positive versus dense everywhere and dev-selected `hybrid_minmax_alpha0.75` positive on NFCorpus and FiQA. Keep it experimental/non-default; it is split-specific hashed sparse-label sidecar evidence, not a shipped neural sparse head or promotion decision.

Use `eos eval-retrieval-hybrid` for sealed Eos artifacts and `eos eval-retrieval-vectors-hybrid` for external vector caches when applying a calibrated setting; both fuse dense top-k and BM25 top-k over their union. Add `--dense-protect-top-k N` only when the product policy intentionally preserves the original dense top-N prefix before appending the fused hybrid tail; `0` keeps the unguarded fusion order. Add `--dense-candidates-only` only for experimental rerank evidence where dense top-k candidate coverage must be preserved while BM25 or projected sparse scores reorder those dense candidates; it is not the default hybrid policy. The prior FiQA-selected setting was `--method minmax --alpha 0.75`, where `alpha` is the BM25 weight, but new default claims should cite the current calibration summary rather than reusing that setting blindly. RRF is available with `--method rrf --rrf-k 60 --rrf-lambda 1.0`. In scoreboards, set the selected values explicitly:

```bash
EOS_SCOREBOARD_HYBRID_RETRIEVAL=1
EOS_SCOREBOARD_HYBRID_METHOD=minmax
EOS_SCOREBOARD_HYBRID_ALPHA=0.75
EOS_SCOREBOARD_HYBRID_RRF_K=60
EOS_SCOREBOARD_HYBRID_RRF_LAMBDA=1.0
```

This appends `eos-hybrid` rows by default and, when external vector datasets are configured, `<external>-hybrid` rows. Scoreboard `method` is explicit, such as `hybrid_minmax_alpha0.75` or `hybrid_rrf_k60_lambda1.0`, so gates can select the hybrid mode independently from dense Eos, BM25, and TurboQuant rows. Set `EOS_SCOREBOARD_HYBRID_BASELINE_LABEL=manta-hybrid` only when intentionally producing a legacy-labeled scoreboard. The FiQA dev-selected `minmax_alpha0.75` result is evidence for guarded hybrid retrieval because it improved held-out FiQA test nDCG@10 over dense v2, strict anchor, and BM25-only; it is not evidence that the dense model itself should be promoted.

Promoted narrow-default full uncapped CorkScrewDB BM25-dot `SearchMulti` smokes now cover the local lexical+dense retrieval-policy lane for `assets/corkscrewdb-default-embedder/corkscrewdb-default-embedder.mll`, SHA256 `e0eca16ff34ebb88ca96862d58c3ac7f02dbf4b124599fdf96f25344ac02e408`. With public `WeightedFusion{Dense:0.5,Sparse:0.5}` and q4 vectors, SciFact run `runs/eos-narrow-default-hybrid-policy-full-scifact-q4-bm25dot-20260623T050356Z` measured dense `0.559571129519`/`0.796444444444` and hybrid `0.723139732393`/`0.932888888889`; NFCorpus run `runs/eos-narrow-default-hybrid-policy-full-nfcorpus-q4-bm25dot-20260623T050540Z` measured dense `0.205279642844`/`0.242243795394` and hybrid `0.316761336578`/`0.293902897482`; FiQA continuation run `runs/eos-narrow-default-hybrid-policy-full-fiqa-q4-bm25dot-continuation-20260623T052918Z` measured dense `0.119552386600`/`0.352919418544` and hybrid `0.224427168561`/`0.502074751380`. All three passed the hybrid-vs-dense gates. Treat these as local API retrieval-policy evidence, not dense model promotion; they do not prove remote mode, federation, HNSW, hosted parity, or a service SLO. The s40 run `runs/eos-s40-current-default-hybrid-policy-v1-20260621T103500Z/` remains predecessor provenance with SciFact `0.724378172602`/`0.932888888889`, NFCorpus `0.315090347196`/`0.293990495284`, and FiQA `0.223742712175`/`0.504002444743`.

Before updating the default-shipping anchor, run the scoreboard-level promotion gate against the accepted dense candidate scoreboard. Targeted-v3 was accepted by passing zero-tolerance dense gates against both the June 10 strict sealed anchor and the v2 candidate; future dense candidates should use targeted-v3 as the comparison point once its scoreboard is the accepted anchor.

```bash
go run ./cmd/eos gate-scoreboard \
  --category short_retrieval \
  --baseline eos \
  --datasets scifact,nfcorpus,fiqa \
  --metrics ndcg_at_10,recall_at_100 \
  /data/manta/runs/<candidate-scoreboard>/scoreboard.json \
  /data/manta/runs/<accepted-dense-scoreboard>/scoreboard.json
```

The command fails on any missing selected row or any dataset metric below the anchor minus `--tolerance`. Macro deltas are printed only as summary evidence; they are not a substitute for every selected dataset metric passing.

Compatibility note: older scoreboards and run directories use the legacy `manta` / `manta-hybrid` row labels and `manta-embed-v1` artifact names for the same embedder lineage. `eos gate-scoreboard --baseline eos` falls back to legacy `manta` rows when exact `eos` rows are absent, and `--baseline eos-hybrid` similarly falls back to `manta-hybrid`.

Canonical active row labels are `eos` for dense `eos-embed-v1`, `eos-hybrid` for lexical+dense hybrid retrieval, `eos-turboquant` for local direct TurboQuant rows, and `eos-turboquant-rerank` for local TurboQuant candidate overfetch plus rerank rows. Enable the promoted local compact lane in the scoreboard with `EOS_SCOREBOARD_TURBOQUANT=1`, `EOS_SCOREBOARD_TURBOQUANT_BITS=4`, `EOS_SCOREBOARD_TURBOQUANT_RERANK_OVERFETCH=200`, `EOS_SCOREBOARD_TURBOQUANT_RERANK_STORAGE=fp16`, `EOS_SCOREBOARD_TURBOQUANT_BASELINE=eos-turboquant`, and `EOS_SCOREBOARD_TURBOQUANT_RERANK_BASELINE=eos-turboquant-rerank`. Gate the compact profile with `--baseline eos-turboquant-rerank --method turboquant_ip_b4_overfetch200_fp16_rerank --bits 4 --metrics ndcg_at_10,recall_at_100,total_compression_ratio` against `runs/eos-nf005-compact-anchor-provenance-repair-v1-20260619T091223Z/anchor-q4-fp16-overfetch200-scoreboard/scoreboard.json`, whose selected q4/fp16/o200 rows include `quantizer_seed=5581486560434873699`. The older q4/fp16/o200 recovery sweep row is historical selection evidence and a stricter FiQA recall reference (`0.3516782086`), not the formal compact `gate-scoreboard` anchor because it lacks `quantizer_seed`. The CorkScrewDB shipping alias should point at the promoted `corkscrewdb-default-embedder` artifact only after the dense, hybrid where relevant, and TurboQuant gates have passed under their documented policies.

Previous measured TurboQuant promotion surface: q4/fp16 sidecar rerank at overfetch250 was the previously selected compact retrieval profile before targeted-v3. It passed the selected-vs-anchor scoreboard gate on SciFact, NFCorpus, and FiQA for `ndcg_at_10,recall_at_100,total_compression_ratio` as `eos-turboquant-rerank` / `turboquant_ip_b4_overfetch250_fp16_rerank` / bits `4`, with total compression `1.590062x`, in `runs/eos-q4-fp16-overfetch250-gate-20260615T000000Z/`. This is a two-stage compact retrieval profile, not q4-only retrieval: direct q4 and direct q8 are not default-promotion candidates. The q8/fp16 sidecar rerank at overfetch125 was the lower-rerank-cost fallback for that earlier evidence set: `turboquant_ip_b8_overfetch125_fp16_rerank`, total compression `1.326425x`, evidence in `runs/eos-fp16-overfetch125-gate-20260614T000000Z/`.

For predecessor EOS-named release packages, keep dense promotion evidence separate from current quantized deployment readiness. The targeted-v3 q4/fp16 overfetch250 compact gate required an explicit `recall@100` tolerance `0.00025` and is now predecessor evidence only. The nf005 q4/175 and q4/150 misses remain compact-policy history; the current s40 compact policy is q4/fp16 overfetch200 with total compression `1.5900621118x`, selected because it passes strict seeded non-regression against the nf005 q4/fp16/o200 anchor. Do not generalize this to direct q4, remote CorkScrewDB, HNSW, federation, or hosted-model parity.

For candidate iteration, prefer the guarded runner so training, scoring, and the scoreboard gate produce one acceptance manifest:

```bash
EOS_REPO_ROOT=$PWD \
EOS_GUARD_ANCHOR_SCOREBOARD=runs/manta-embed-v1-deephard-full-ft-20260610T0000Z-sealed-scoreboard/scoreboard.json \
ferrous-wheel run scripts/run_manta_embed_v1_guarded_candidate.fw
```

The runner inherits the normal `EOS_TRAIN_JSONL`, `EOS_EVAL_JSONL`, `EOS_HARD_EVAL_JSONL`, tokenizer, training, and scoreboard knobs. It defaults to `scifact,nfcorpus,fiqa`, `ndcg_at_10,recall_at_100`, `short_retrieval`, and `eos`, writes `runs/eos-embed-v1-guarded-<timestamp>/manifest.json`, and exits nonzero when the candidate is rejected. Set `EOS_GUARD_BASELINE=manta` only for legacy-labeled candidate scoreboards. Set `EOS_GUARD_FAIL_ON_GATE=0` only when you need a non-failing smoke that still records `"gate_status": "rejected"`. Dirty working trees remain blocked by the training script unless the caller sets `EOS_ALLOW_DIRTY=1` or explicitly sets `EOS_GUARD_ALLOW_DIRTY=1`.

For hard-negative teacher runs, use `EOS_TEACHER_SOURCE_WEIGHTS` when teacher audits agree with labels on some sources but hurt others. Source keys use exact, family, then `*` fallback, and weight `0` disables teacher contribution for that source. The next guarded Qwen3 shape should pair global `EOS_TEACHER_LOSS_WEIGHT=0.10` with `EOS_TEACHER_SOURCE_WEIGHTS=scifact=1,nfcorpus=0,fiqa=0.25`.

For a dry smoke with existing artifacts:

```bash
EOS_REPO_ROOT=$PWD \
EOS_GUARD_CANDIDATE_DIR=runs/<candidate-run> \
EOS_GUARD_SCOREBOARD_JSON=runs/<candidate-scoreboard>/scoreboard.json \
EOS_GUARD_ANCHOR_SCOREBOARD=runs/<anchor-scoreboard>/scoreboard.json \
EOS_GUARD_FAIL_ON_GATE=0 \
ferrous-wheel run scripts/run_manta_embed_v1_guarded_candidate.fw
```

Contrastive training uses pair-length-aware bucketing by default so batches reach larger exact-length matmul groups. `EOS_TRAIN_LENGTH_BUCKET_WINDOW` controls the shuffled sort window; larger values can improve grouping but must be profiled because they reduce local length randomness and can increase per-batch working-set pressure.

The production workflow defaults `EOS_EVAL_EVERY_STEPS=0`. Keep within-epoch eval disabled for full candidate runs unless you are debugging convergence; epoch eval, final validation eval, and hard holdout eval still run and are enough for release gating. On the acquired full split, step-level eval every 4 batches adds many full eval passes and dominates transfer without improving the optimizer update itself.

Eval-only candidate gates batch pairwise forward encodes by exact token length. `EOS_TRAIN_PAIR_EVAL_BATCH_SIZE` defaults to `256`; set `EOS_TRAIN_DISABLE_BATCHED_PAIR_EVAL=1` only for scalar A/B checks.

`EOS_TRAIN_ENABLE_FAST_GELU=1` is available for throughput experiments. It changes GELU math from precise tanh to a bounded rational tanh approximation, so use it only when the exact-GELU candidate and fast-GELU candidate are both evaluated against validation and hard holdout gates.

## Metric Thresholds

The default acquired eval files are pairwise positive/negative judgments with one deterministic sampled negative per positive. The initial release gates are:

```text
EOS_SELECT_METRIC=score_margin
EOS_MIN_AUC=0.70
EOS_MIN_THRESHOLD_ACCURACY=0.65
EOS_MIN_SCORE_MARGIN=0.05
EOS_MAX_LOSS=0.35
```

These gates are intentionally concrete rather than advisory. AUC must clear 0.70 as a threshold-free separability check, calibrated threshold accuracy must clear 0.65 against a 0.50 random baseline, positive scores must beat negative scores by at least 0.05 on average, and pairwise loss must stay at or below 0.35. Tighten them after the first stable full-size candidate establishes the project baseline.

## Training Efficiency Gates

Production candidate runs can also enforce hardware-specific training efficiency gates from `train.metrics.json`:

```text
EOS_MIN_TRAIN_PAIRS_PER_SEC=85000
EOS_MIN_OPTIMIZER_STEPS_PER_SEC=0.08
EOS_MAX_MATMUL_RUNS=400000
EOS_MAX_MATMUL_RUN_UPLOAD_MB=950000
EOS_MAX_MATMUL_RUN_DOWNLOAD_MB=485000
```

Set these from a known-good run on the target trainer host. Keep quality gates hardware-independent, and keep efficiency gates host-specific so a CPU fallback or data-transfer regression does not silently produce a slow candidate.

## Corpus Path

Use corpus mode only when you intentionally want this run to mine the train/eval pairs:

```bash
EOS_RUN_ROOT=/data/manta/runs \
EOS_REPO_ROOT=$PWD \
EOS_RUN_ID=eos-embed-v1-20260412-corpus-a \
EOS_CORPUS=/data/manta/corpus/prod-corpus.txt \
EOS_HARD_EVAL_JSONL=/data/manta/datasets/eos-embed-v1/hard-eval.jsonl \
EOS_EPOCHS=3 \
EOS_BATCH_SIZE=1024 \
EOS_MAX_PAIRS=0 \
EOS_EVAL_PAIRS=512 \
ferrous-wheel run scripts/train_manta_embed_v1_candidate.fw
```

Corpus mode writes mined pairs and the tokenizer into the run directory, then records their SHA256 values in `datasets.sha256`.

## JSONL Formats

Token contrastive JSONL:

```json
{"query_tokens":[1,2,3],"positive_tokens":[1,2,3],"query_mask":[1,1,1],"positive_mask":[1,1,1]}
```

Text-pair JSONL:

```json
{"query":"how to reset password","document":"reset your password from account settings","label":1}
{"left":"how to reset password","right":"billing invoice export","label":0}
```

Positive-only text pairs can train contrastively. Mixed positive/negative text pairs are valid for eval-only gates.

Token-pair eval JSONL:

```json
{"left_tokens":[1,2,3],"right_tokens":[1,2,3],"left_mask":[1,1,1],"right_mask":[1,1,1],"target":1}
{"left_tokens":[1,2,3],"right_tokens":[9,8],"left_mask":[1,1,1],"right_mask":[1,1],"target":0}
```

## Release Gate

Current dense promotion candidate: `assets/corkscrewdb-default-embedder/corkscrewdb-default-embedder.mll`, SHA256 `e0eca16ff34ebb88ca96862d58c3ac7f02dbf4b124599fdf96f25344ac02e408`. It is the durable CorkScrewDB external-provider asset for the narrow compact-mined release; the ignored run path `runs/eos-s40-nfcorpus-compact-mined-narrow-candidate-v1-20260623T032556Z/candidate/eos-embed-v1.sealed.mll` remains provenance. It was selected because it passed dense short-set non-regression against s40 with macro nDCG@10 `+0.000058267531291` and macro recall@100 `+0.000002457123200`, and passed strict seeded q4/fp16/o200 compact non-regression against s40 with macro nDCG@10 `+0.000118345582923`, macro recall@100 `+0.000259403173951`, and total compression `1.5900621118x`. The release package SHA256 remains `188265db16992ab24be15e678c5f7e175bebad769e8d844e8b0f50ffc23bd5bf`; tokenizer SHA256 remains `64cf63223cb3f97125040677a573e6ab6c625cff1f6f338f4e680a4c9f7a42f5`. Package and sealed inspection report `package verify: OK`, sealed inspection reports `package: embedded sealed MLL`, and final plus hard eval logs record `optimizer_updates=0`.

| Dataset | s40 nDCG@10 | s40 recall@100 | Delta vs nf005 nDCG@10 | Delta vs nf005 recall@100 |
| --- | ---: | ---: | ---: | ---: |
| SciFact | 0.5645379155 | 0.7964444444 | +0.0000000000 | +0.0000000000 |
| NFCorpus | 0.205571 | 0.242059 | +0.000213 | +0.000011 |
| FiQA | 0.121261 | 0.351678 | +0.000151 | +0.000000 |

The s40 predecessor package at `runs/eos-frontier-teacher-sentinel-balance-sweep-v1-s40-20260620T154736Z/`, old SHA256 `f494915a0d78b24205d5018bb701bf40cabbedee4bc8b96b6a1920b19131da5a`, the nf005 predecessor package at `runs/current-release-qwen3-nf005-continuation-20260616T224102Z/candidate/`, the targeted-v3 package at `runs/eos-embed-v1-targeted-v3-release-package-20260616T000000Z/`, and the legacy source artifact `runs/eos-embed-v1-targeted-neargate-v3-low-lr-restorebest-20260614T000000Z/targeted-v3-lr000002-restorebest-manta/manta-embed-v1.sealed.mll`, SHA256 `ea776e2fca7fdade7ee05396b2ee8980e220899e2515853c83a4bca34cf87242`, remain comparison provenance. The previous strict sealed anchor is `runs/manta-embed-v1-deephard-full-ft-20260610T0000Z/manta-embed-v1.sealed.mll`, SHA256 `a7461b47784ea7434cf6048f33f6c281ef19887cfa9d0c699b6f2fba079f2b67`, with sealed scoreboard under `runs/manta-embed-v1-deephard-full-ft-20260610T0000Z-sealed-scoreboard/`. Its sealed-vs-train-package comparison recorded zero nonzero quality or count deltas against `runs/manta-embed-v1-deephard-full-ft-20260610T0000Z-scoreboard/`. Treat s40 as predecessor measured dense short-set evidence, not a broad release claim. Promoted narrow-default CorkScrewDB local flat packed-parent API evidence is `runs/eos-narrow-default-corkscrewdb-budget-quality-packed-q4q8-main-20260623T050216Z`, using fresh SciFact child vectors from `runs/eos-vector-caches/eos-narrow-default-scifact-child-w128-o32-128d` (`5,183` documents, `12,468` child vectors, `300` queries, `128d`, CUDA) and CorkScrewDB repo `/home/draco/work/corkscrewdb`: q4 release row records nDCG@10 `0.452380`, recall@100 `0.758556`, DB directory multiple `0.041676x`, vector payload multiple `0.013312x`, p95 `16.076127ms`, planner fit `180`, target fit `true`, and target storage multiple `0.554545x`; q8 is diagnostic with nDCG@10 `0.472132`, recall@100 `0.773556`, DB directory multiple `0.066734x`, vector payload multiple `0.025841x`, p95 `36.113081ms`, planner fit `93`, target fit `false`, and target storage multiple `1.074026x`. Current HNSW/raw evidence for the same cache is `runs/eos-current-default-corkscrewdb-hnsw-refresh-v1-20260623T055809Z`: q4 overfetch100 records nDCG@10 `0.445689`, recall@100 `0.735222`, raw HNSW DB directory multiple `0.243494x`, vector payload multiple `0.013312x`, p95 `2.324070ms`, planner fit `123`, and target fit `true`; q4 overfetch12468 records nDCG@10 `0.452380`, recall@100 `0.758556`, and p95 `49.635238ms`. q8 HNSW was not run. These cover local flat packed-parent and local HNSW/raw API only, not remote mode, federation, hosted parity, or a service SLO.

A candidate is releasable only when:

- `manifest.json` status is `success`
- `logs/final-eval.log` and `logs/hard-eval.log` report `optimizer_updates=0`
- configured metric gates pass on hard eval
- `logs/inspect-package.log` reports `package verify: OK`
- `logs/inspect-sealed.log` reports `package verify: OK`
- `artifacts.sha256` contains the sealed MLL hash
- `eos gate-scoreboard --datasets scifact,nfcorpus,fiqa --metrics ndcg_at_10,recall_at_100 <candidate-scoreboard>/scoreboard.json <sealed-anchor-scoreboard>/scoreboard.json` passes

The release artifact is:

```text
runs/eos-frontier-teacher-sentinel-balance-sweep-v1-s40-20260620T154736Z/eos-embed-v1.sealed.mll
```

Keep the full run directory with the released artifact. It is the audit trail for reproducing or rejecting the candidate later.
