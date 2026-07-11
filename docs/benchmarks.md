# Benchmarks

Eos has three benchmark layers:

- Go microbenchmarks for isolated runtime kernels and trainer math.
- End-to-end training smokes for the native default embedding-model path.
- Embedder scoreboards for retrieval quality and serving efficiency claims.

For the compact child-vector storage frontier that sits beside the embedding benchmarks, see [TurboQuant Multi-Vector Frontier](turboquant-multivector-frontier.md). It centralizes the CorkScrewDB/Eos many-child-vectors-per-parent thesis, evidence boundaries, and promotion gates. The current high-child-count measured DB-directory boundary comes from `runs/eos-packed-parent-storage-sensitivity-matrix-20260616T000000Z/`: strict storage-frontier mode uses ordinal child IDs plus none/minimal packed metadata, while source child IDs and full metadata are richer product modes with larger DB envelopes.

Run the default microbenchmarks with Ferrous Wheel:

```bash
EOS_BENCH_ROOT=$PWD ferrous-wheel run scripts/bench.fw
```

Run CUDA microbenchmarks:

```bash
EOS_BENCH_ROOT=$PWD EOS_BENCH_CUDA=1 ferrous-wheel run scripts/bench.fw
```

Run the sparse-attention x TurboQuant measurement layer:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/bench_sparse_attention.fw
```

The sparse-attention harness first writes a routed preflight plan as `sparse-attention-plan.tsv` and `sparse-attention-plan.json`, then records CUDA benchmark output as `sparse-attention-bench.jsonl`, `sparse-attention-bench.txt`, `sparse-attention-bench-summary.tsv`, and `sparse-attention-scaling.tsv` under `runs/<run-id>/`. The Go benchmark lines and summary table include selected-key count, candidate-key budget, estimated scores per query, score fraction, subquadratic-plan flag, TurboQuant K/V MiB, and logical K/V compression ratio. The scaling table fits log-log time alpha for exact f16 and routed TurboQuant rows, and the harness fails routed TurboQuant when alpha exceeds `EOS_SPARSE_BENCH_MAX_ROUTED_TIME_ALPHA` (`0.95` by default; set `0` to disable during exploratory runs). Keep exact dense costs bounded with `EOS_SPARSE_BENCH_EXACT_KEY_LENS` (`1024,4096` by default) while extending routed scaling with `EOS_SPARSE_BENCH_ROUTED_KEY_LENS` (`1024,4096,16384` by default).

Bridge the sparse encoder-shaped smoke into the long-context evidence flow with:

```bash
go run ./cmd/eos smoke-sparse-embedding-encoder \
  --run-dir runs/eos-sparse-embedding-encoder-smoke-local \
  --backend host \
  --seq-len 64 \
  --query-len 2 \
  --dim 8 \
  --top-k 2 \
  --route-top-blocks 2 \
  --preflight-key-lens 4096,32768
```

The smoke writes `manifest.json`, `summary.tsv`, `scorecard.json`, and `scorecard.tsv`. The scorecard row records runtime/backend metadata, 32k preflight status, the 32768-key score fraction, TurboQuant bit width and seed, parity status, `evidence_level=smoke_synthetic_kernel_evidence`, and `quality_claim=false`. It is a reproducible sparse-enabled encoder smoke artifact, not retrieval-quality evidence and not a LongEmbed claim.

Export the first retrieval-compatible sparse-token pooling prototype from an existing trainable embedding package with:

```bash
go run ./cmd/eos export-sparse-token-pool-vectors \
  --dataset repo-docs-sparse-prototype \
  --split test \
  --manifest-json runs/eos-sparse-token-pool-retrieval-prototype/manifest.json \
  --document-chunk-words 256 \
  --document-chunk-overlap 32 \
  --document-chunk-min-words 64 \
  --max-tokens 4096 \
  --top-k 64 \
  --route-block-size 64 \
  --route-top-blocks 8 \
  --bits 4 \
  --seed 5581486560434873699 \
  /path/to/eos-embed-trainable.mll \
  datasets/longembed/repo-docs \
  runs/eos-sparse-token-pool-retrieval-prototype/repo-docs
```

This writes `child-doc-vectors.jsonl` or `doc-vectors.jsonl`, `query-vectors.jsonl`, and a manifest with `method=experimental_sparse_token_pool` and `quality_claim=false`. It loads the tokenizer and sibling weights, gathers actual token_embedding rows, applies Q/K/V/O/projection weights only when manifest names and tensor shapes line up, and runs host-reference routed `TurboSparseAttentionReference` over TurboQuant-encoded K/V. Score the cache with `eos eval-retrieval-vectors` or `eos eval-retrieval-vectors-turboquant` before comparing it to dense or external baselines. This is a bridge row for retrieval evaluation only, not production embedder output, not a trained sparse encoder, and not LongEmbed proof.

Export the distinct sparse encoder host-reference parent-vector row with:

```bash
go run ./cmd/eos export-sparse-encoder-vectors \
  --dataset repo-docs-sparse-encoder-host \
  --split test \
  --manifest-json runs/eos-sparse-encoder-entrypoint/repo-docs/manifest.json \
  --max-tokens 4096 \
  --top-k 224 \
  --route-block-size 128 \
  --route-top-blocks 8 \
  --bits 4 \
  --seed 5581486560434873699 \
  /path/to/eos-embed-trainable.mll \
  datasets/longembed/repo-docs \
  runs/eos-sparse-encoder-entrypoint/repo-docs/vectors
```

This command writes parent `doc-vectors.jsonl` plus `query-vectors.jsonl` and requires full manifest encoder weights. Its manifest uses `method=experimental_sparse_encoder_host_reference`, `evidence_level=retrieval_cache_host_reference_sparse_encoder`, `quality_claim=false`, `require_full_encoder=true`, `full_encoder_applied=true`, and host-reference sparse-plan audit fields for the maximum observed document token length. It also records that this TurboQuant sparse path materializes dense K/V and decodes K/V on host. Treat this row as retrieval-cache prototype evidence only, not trained LongEmbed evidence, sealed runtime sparse inference, or production quality. Score q-bit rows with `eos eval-retrieval-vectors-turboquant --quantizer-seed <seed>` so cache provenance matches the export seed.

Calibrate sparse routing policy quality separately from kernel timing:

```bash
go run ./cmd/eos calibrate-sparse-routing \
  --run-dir runs/eos-sparse-routing-calibration-local \
  --seq-len 4096 \
  --query-len 8 \
  --dim 64 \
  --top-k 64 \
  --route-block-size 10 \
  --route-top-blocks 35,36,37,38,39,40 \
  --route-modes anchor,multiprobe,summary_mean,summary_mean_radius,summary_maxnorm,summary_blend_radius,summary_multirep,summary_hier_multirep,learned_block_linear,oracle_block_max \
  --route-probes 1,2,4,8 \
  --route-summary-counts 1,2,4,8 \
  --route-hier-group-blocks 4,8 \
  --route-hier-top-groups 8,10,12,16,20 \
  --route-summary-alphas 0,0.25,0.5,0.75,1 \
  --route-radius-weights -0.25,0,0.25,0.5 \
  --learned-router-train-seeds 5581486560434873699 \
  --learned-router-eval-seeds 5581486560434873699,5581486560434873700 \
  --max-score-fraction 0.2 \
  --min-exact-topk-recall 0.95 \
  --min-exact-topk-recall-min 0.95 \
  --min-output-cosine 0.98
```

The command writes `calibration.json` and `calibration.tsv` under the run directory. Use `--route-modes` to compare deployable heuristics with teacher-only `oracle_block_max`; `summary_mean_radius` uses a deployable block mean plus scalar radius upper bound, `summary_maxnorm` scores the largest-L2-norm key representative in each block, and `summary_blend_radius` scores `mean + alpha*(maxnorm_rep-mean)` plus `beta*||query||_2*radius` using `--route-summary-alphas` and `--route-radius-weights`. `summary_multirep` is a calibration-only multi-summary probe: it precomputes deterministic farthest-point representatives per block and routes by the max query dot across `--route-summary-counts`; score accounting charges `block_count * route_summary_count` route scores plus candidate-key work. `summary_hier_multirep` is also calibration-only: it groups fine blocks by `--route-hier-group-blocks`, scores one coarse max-norm representative per group, scores `route_summary_count` fine representatives only inside the selected `--route-hier-top-groups`, then selects final fine blocks from that subset. Its score accounting charges coarse group scores plus selected fine-block summary scores plus candidate-key work. `learned_block_linear` is a calibration-only learned-summary candidate: it fits a tiny deterministic logistic linear scorer from `oracle_block_max` top-block labels on synthetic training seeds, then evaluates from learned weights and cheap per-block features only. Its route score accounting uses a fused learned representative dot per block plus scalar features and records teacher score work separately from evaluation score work. Treat the output as routing calibration evidence only: it does not prove downstream retrieval quality or runtime selector feasibility.

Run the default-model training smoke from a local asset package:

```bash
EOS_BENCH_ROOT=$PWD EOS_BENCH_MODEL_ASSETS=/path/to/assets/eos-embed-v1 ferrous-wheel run scripts/bench.fw
```

The model smoke copies the package into a temporary directory before training. It does not mutate the source asset directory.

Evaluate an existing candidate package against a token JSONL or text-pair JSONL eval file without running optimizer steps:

```bash
go run ./cmd/eos train-embed --eval-only /path/to/eos-embed-v1.mll /path/to/eval-mini.jsonl
```

When the package has a sibling `.tokenizer.mll`, text eval JSONL is tokenized automatically. Pass `--tokenizer /path/to/tokenizer.mll` to use an explicit tokenizer.

For a production candidate run with acquired BEIR-format datasets, provenance, metric gates, sealed export, and SHA256 manifests, use `scripts/acquire_manta_embed_v1_datasets.fw` followed by `scripts/train_manta_embed_v1_candidate.fw`. See `docs/production-embedding.md`.

Build the long-context embedder wedge scoreboard from an existing sealed candidate:

```bash
EOS_REPO_ROOT=$PWD \
EOS_SCOREBOARD_ARTIFACT=/path/to/eos-embed-v1.sealed.mll \
EOS_SCOREBOARD_PAIRWISE_JSONL=/data/manta/datasets/eos-embed-v1/processed/eval.jsonl \
EOS_SCOREBOARD_HARD_JSONL=/data/manta/datasets/eos-embed-v1/processed/hard-eval.jsonl \
EOS_SCOREBOARD_RETRIEVAL_ROOT=/data/manta/datasets/eos-embed-v1 \
EOS_SCOREBOARD_RETRIEVAL_DATASETS=scifact,nfcorpus,fiqa \
ferrous-wheel run scripts/score_manta_embed_v1_baselines.fw
```

The scoreboard run writes `scoreboard.tsv`, `scoreboard.json`, per-task metrics JSON, command logs, and a run-local `eos` binary under `runs/<run-id>/`. Dense local rows default to `baseline=eos`; hybrid rows default to `baseline=eos-hybrid`; local TurboQuant rows use `eos-turboquant` for direct quantized scoring and `eos-turboquant-rerank` for quantized candidate overfetch plus rerank storage modes such as fp16; external TurboQuant rows use `<external>-turboquant`; external child-vector rows use `<external>-dense-child` for dense child max and `<external>-turboquant-child` for q-bit child max rows. Set `EOS_SCOREBOARD_BASELINE_LABEL=manta` and `EOS_SCOREBOARD_HYBRID_BASELINE_LABEL=manta-hybrid` only when intentionally producing a legacy-labeled scoreboard. Pairwise rows use `EOS_SCOREBOARD_PAIRWISE_ARTIFACT` when set, or infer the sibling trainable package from a sealed artifact path. Add `EOS_SCOREBOARD_LONG_ROOT` and `EOS_SCOREBOARD_LONG_DATASETS` when long-document retrieval datasets are prepared.

Promoted narrow-default q4 BM25-dot hybrid short-retrieval evidence used `assets/corkscrewdb-default-embedder/corkscrewdb-default-embedder.mll` at SHA256 `e0eca16ff34ebb88ca96862d58c3ac7f02dbf4b124599fdf96f25344ac02e408` with full uncapped CorkScrewDB `SearchMulti` smokes. It is ranking-policy evidence for the local `eos-hybrid` API surface, not dense `eos` promotion, and does not prove remote mode, HNSW, federation, or hosted parity.

| Dataset | Run | Dense nDCG@10 / recall@100 | Hybrid nDCG@10 / recall@100 | Gate |
| --- | --- | ---: | ---: | --- |
| SciFact | `runs/eos-narrow-default-hybrid-policy-full-scifact-q4-bm25dot-20260623T050356Z` | 0.559571129519 / 0.796444444444 | 0.723139732393 / 0.932888888889 | pass |
| NFCorpus | `runs/eos-narrow-default-hybrid-policy-full-nfcorpus-q4-bm25dot-20260623T050540Z` | 0.205279642844 / 0.242243795394 | 0.316761336578 / 0.293902897482 | pass |
| FiQA | `runs/eos-narrow-default-hybrid-policy-full-fiqa-q4-bm25dot-continuation-20260623T052918Z` | 0.119552386600 / 0.352919418544 | 0.224427168561 / 0.502074751380 | pass |

The s40 command-level and full uncapped BM25-dot `SearchMulti` rows in `runs/eos-s40-current-default-hybrid-policy-v1-20260621T103500Z/` remain predecessor provenance. They passed with SciFact `0.724378172602`/`0.932888888889`, NFCorpus `0.315090347196`/`0.293990495284`, and FiQA `0.223742712175`/`0.504002444743` hybrid nDCG@10/recall@100, all `overall_pass=true`.

Run the parent/single-vector prefix-dimension compact-quality harness when the question is "how much quality does the current default artifact keep after prefix truncation plus direct q-bit vector-index compression?":

```bash
EOS_REPO_ROOT=$PWD \
EOS_PREFIX_CURVE_ARTIFACT=$PWD/assets/corkscrewdb-default-embedder/corkscrewdb-default-embedder.mll \
EOS_PREFIX_CURVE_DATASET_ROOT=$PWD/datasets/manta-embed-v1 \
EOS_PREFIX_CURVE_DATASETS=scifact,nfcorpus,fiqa \
EOS_PREFIX_CURVE_DIMS=32,64,96,128 \
EOS_PREFIX_CURVE_BITS=2,4,8 \
EOS_PREFIX_CURVE_MAX_QUERIES=300 \
EOS_PREFIX_CURVE_TOP_K=100 \
EOS_PREFIX_CURVE_BATCH_SIZE=64 \
ferrous-wheel run scripts/eval_eos_prefix_dim_curve.fw
```

Use `EOS_PREFIX_CURVE_MAX_QUERIES=0`, `EOS_PREFIX_CURVE_DIMS=128`, and `EOS_PREFIX_CURVE_BITS=4,8` for the current full-query short-set confirmation. The harness writes `prefix-dim-curve.summary.json`, `prefix-dim-curve.summary.tsv`, export manifests, metrics JSON, and command logs under `runs/<run-id>/`. Its prefix rows use `export-retrieval-vectors --output-dim`, which prefix-truncates and L2-renormalizes existing embeddings. They are not trained Matryoshka/native 128d heads unless the artifact was trained with a prefix/Matryoshka objective. Compression fields are vector-payload accounting from the vector-cache/TurboQuant metrics, not full CorkScrewDB DB-byte evidence and not q4/fp16 rerank evidence.

s40-era full-query `128d` parent/single-vector evidence is in `runs/eos-prefix-128-q4q8-full-query-current-default-v1-20260620T021328Z/` and `.tiller/scratch/codex/eos-prefix-128-q4q8-full-query-current-default-v1-report.md`; refresh before using it as new-default evidence for the narrow promotion:

| dataset | method | nDCG@10 | recall@100 | storage multiple | compression |
| --- | --- | ---: | ---: | ---: | ---: |
| scifact | dense | 0.494851 | 0.761889 | 1.000000 | 1.000000x |
| scifact | q4 | 0.478427 | 0.758556 | 0.132812 | 7.529412x |
| scifact | q8 | 0.492185 | 0.760778 | 0.257812 | 3.878788x |
| nfcorpus | dense | 0.184242 | 0.214583 | 1.000000 | 1.000000x |
| nfcorpus | q4 | 0.178007 | 0.216910 | 0.132812 | 7.529412x |
| nfcorpus | q8 | 0.184394 | 0.215245 | 0.257812 | 3.878788x |
| fiqa | dense | 0.098073 | 0.299695 | 1.000000 | 1.000000x |
| fiqa | q4 | 0.092377 | 0.297274 | 0.132812 | 7.529412x |
| fiqa | q8 | 0.097320 | 0.301407 | 0.257812 | 3.878788x |

Interpretation for product lanes: these `128d q8` and `128d q4` rows are s40-era/predecessor prefix evidence, not promoted narrow-default evidence until refreshed. `128d q8` is the near-dense compact cache reference in that predecessor lane, at about `3.878788x` vector-payload compression. `128d q4` is the aggressive compact candidate in that same lane, at about `7.529412x` vector-payload compression, with full-query nDCG deltas versus dense of `-0.016424` SciFact, `-0.006235` NFCorpus, and `-0.005696` FiQA. q2 is not default-quality-ready without a trained compact head: on the bounded prefix curve it remained materially below dense at `128d`, and at `32d` it was the clear collapse setting on every dataset. Keep this parent/single-vector prefix evidence distinct from the packed-parent child-vector evidence below, which measures many vectors per parent object and separate CorkScrewDB layout/storage questions.

For the current synthetic late-needle long-context smoke, run:

```bash
EOS_REPO_ROOT=$PWD \
EOS_LONG_CONTEXT_SMOKE_RUN_DIR=runs/eos-long-context-chunked-baseline-smoke-v1-repro \
ferrous-wheel run scripts/smoke_eos_long_context_chunked_baseline.fw
```

This harness generates a BEIR-shaped synthetic long-document dataset with relevant evidence near the end of each document, then compares live Eos single-vector retrieval, BM25, exported Eos single-vector cache retrieval, dense chunked child-vector max rollup, and q2/q4/q8 TurboQuant child-vector max rollup. The current promoted-default rerun at `runs/eos-current-default-long-context-chunked-baseline-v1-20260623T055729Z/`, using `assets/corkscrewdb-default-embedder/corkscrewdb-default-embedder.mll`, recorded direct Eos nDCG@10 `0.119502020`, single-vector cache nDCG@10 `0.157617787`, and perfect nDCG@10/recall@100 for BM25 plus chunked dense/q2/q4/q8 rows. Treat this as synthetic late-needle smoke evidence for the harness shape, not LongEmbed proof or a real long-document benchmark. The predecessor checkpoint is `runs/eos-long-context-chunked-baseline-smoke-v1-20260618T054701Z/`; see `.tiller/scratch/codex/eos-current-default-long-context-wedge-rerun-v1-report.md` and `.tiller/scratch/codex/eos-long-context-chunked-baseline-smoke-v1-report.md` for detailed artifact summaries.

Set `EOS_LONG_CONTEXT_SMOKE_SCENARIO=semantic_late_needle` for a harder synthetic semantic/paraphrase long-document smoke; it defaults to `144` parents and `36` queries. The verified run at `runs/eos-semantic-long-context-smoke-v1-20260618T061747Z/` recorded nDCG@10 of Eos direct `0.067554`, BM25 `0.037130`, Eos q4 chunked `0.020641`, Qwen3 0.6B 128d q4 chunked `0.322934`, and mxbai-large 128d q4 chunked `0.282152`. Recall@100 is less diagnostic in this run because `top_k=100` over `144` parents is generous and the strongest external rows saturate it, so use nDCG@10 as the compact-vector comparison signal. This is still not LongEmbed proof, and the hand-authored synthetic topics may be idiosyncratic, but it is now a useful non-saturated compact-vector comparison smoke. See `.tiller/scratch/codex/eos-semantic-long-context-smoke-v1-report.md` and `runs/eos-semantic-long-context-smoke-v1-20260618T061747Z/comparison.tsv` for details.

The external preset exporter also supports `--output-dim` for prefix truncation plus L2 renormalization, giving compact vector-cache parity through `scripts/export_retrieval_vectors.py` wrapper commands. Offline cached Qwen3 0.6B and mxbai-large runs on `late-needle-w96-o16` produced native 1024d and compact 128d child caches at `runs/eos-late-needle-external-baselines-v1-20260618T060510Z/`: native dense/q2/q4/q8 rows were perfect for both embedders; compact 128d dense/q4/q8 rows were also perfect; compact q2 kept recall@100 at `1.0` but dropped to Qwen3 nDCG@10 `0.948904` and mxbai nDCG@10 `0.958333`. Because this is still synthetic and BM25 also scores perfectly, treat it as exporter and comparison-harness evidence, not LongEmbed proof.

Build a small non-synthetic local-doc long-retrieval lane with:

```bash
EOS_REPO_ROOT=$PWD ferrous-wheel run scripts/build_repo_docs_longembed_dataset.fw
```

This writes a BEIR-shaped `datasets/longembed/repo-docs` dataset from local repository Markdown and skill docs. The verified lane currently has `11` documents, `80` heuristic qrels, and `6` documents at `>=2048` words. In the checkpoint run summarized at `.tiller/scratch/codex/eos-repo-docs-longembed-lane-v1-report.md`, BM25 reached nDCG@10 `0.717080`, the Eos default single-vector row reached `0.644872`, Qwen3 0.6B 128d dense child-max reached `0.771170`, and Qwen3 q4 child-max reached `0.739269`. Treat this as local harness coverage over real repo text, not LongEmbed proof: qrels are deterministic path/heading heuristics, the dataset is small and repo-specific, and BM25 benefits from path and heading lexical overlap.

Convert official LongEmbed Hugging Face configs into the same BEIR-shaped layout with:

```bash
python3 scripts/convert_longembed_to_beir.py \
  --dataset needle,passkey \
  --output-root datasets/longembed-official \
  --max-docs 200 \
  --max-queries 50
```

The adapter supports the official `dwzhu/LongEmbed` configs `narrativeqa`, `qmsum`, `2wikimqa`, `summ_screen_fd`, `passkey`, and `needle`, reading the `corpus`, `queries`, and `qrels` splits and writing `<output-root>/<dataset>/{corpus.jsonl,queries.jsonl,qrels/test.tsv,dataset-manifest.json}`. It lazily imports Hugging Face `datasets` only for online acquisition; no-network fixture conversion uses `--input-root <root>` with `<root>/<dataset>/{corpus,queries,qrels}.jsonl`. Use `--context-length 4096` for synthetic `needle`/`passkey` slices when source rows expose a recognized length field or an ID marker; otherwise the manifest records that no length filter was applied. These files are acquisition/conversion artifacts only. LongEmbed proof still requires actual wedge or scoreboard eval rows over the converted datasets.

To append those repo-docs child-vector cache rows directly into `scoreboard.json`, point the scoreboard at a cache root containing `<root>/repo-docs/child-doc-vectors.jsonl` and `query-vectors.jsonl`:

```bash
EOS_REPO_ROOT=$PWD \
EOS_SCOREBOARD_ARTIFACT=/path/to/eos-embed-v1.sealed.mll \
EOS_SCOREBOARD_LONG_ROOT=$PWD/datasets/longembed \
EOS_SCOREBOARD_EXTERNAL_MULTIVECTOR_ROOT=runs/external-vector-caches/qwen3-0.6b-repo-docs-128d \
EOS_SCOREBOARD_EXTERNAL_MULTIVECTOR_DATASETS=repo-docs \
EOS_SCOREBOARD_EXTERNAL_MULTIVECTOR_BASELINE=qwen3-0.6b-128d-child \
EOS_SCOREBOARD_EXTERNAL_MULTIVECTOR_BACKEND=qwen3-0.6b-128d-child \
EOS_SCOREBOARD_EXTERNAL_MULTIVECTOR_ARTIFACT=Qwen/Qwen3-Embedding-0.6B \
EOS_SCOREBOARD_EXTERNAL_MULTIVECTOR_BITS=4 \
EOS_SCOREBOARD_EXTERNAL_MULTIVECTOR_BASELINE_DIM=1024 \
ferrous-wheel run scripts/score_manta_embed_v1_baselines.fw
```

The external multivector group is opt-in via `EOS_SCOREBOARD_EXTERNAL_MULTIVECTOR_DATASETS`. It reuses `eos eval-retrieval-multivector-turboquant`, writes one dense child-max row plus one row per requested bit width, and preserves `quantizer_seed` on q-bit rows for compact-scoreboard provenance gates. The current repo-docs child-vector scoreboard evidence is:

| Row | Method | nDCG@10 | recall@100 |
| --- | --- | ---: | ---: |
| BM25 | bm25 | 0.717080 | 1.000000 |
| Eos default | cuda dense | 0.644872 | 1.000000 |
| Eos 128d child | dense child-max | 0.597491 | 1.000000 |
| Eos 128d child | q2 child-max | 0.580548 | 1.000000 |
| Eos 128d child | q4 child-max | 0.603952 | 1.000000 |
| Eos 128d child | q8 child-max | 0.600218 | 1.000000 |
| Qwen3 0.6B 128d child | dense child-max | 0.771170 | 1.000000 |
| Qwen3 0.6B 128d child | q2 child-max | 0.694290 | 1.000000 |
| Qwen3 0.6B 128d child | q4 child-max | 0.739269 | 1.000000 |
| Qwen3 0.6B 128d child | q8 child-max | 0.769390 | 1.000000 |
| mxbai-large 128d child | dense child-max | 0.713075 | 1.000000 |
| mxbai-large 128d child | q2 child-max | 0.627584 | 1.000000 |
| mxbai-large 128d child | q4 child-max | 0.691586 | 1.000000 |
| mxbai-large 128d child | q8 child-max | 0.708765 | 1.000000 |

The q-bit rows use quantizer seed `5581486560434873699`. Treat the Eos 128d child rows as prefix truncation plus L2 renormalization from the default 256d artifact, not as evidence for a trained Matryoshka head. Treat every repo-docs row as local harness evidence over deterministic repo-specific qrels, not LongEmbed proof.

The repo-docs token-span sweeps add diagnostic sparse-token-pool child-vector evidence for the Eos lineage, using run-local maxseq1024 retargets rather than trained long-context artifacts. The best 256d row was token span `64/16`: dense `0.647864`, q4 `0.650561`, q8 `0.652997`, `211` child vectors, and q4 storage `2.472656x` versus a 256d dense parent-vector budget. The best compact 128d row was `48/12`: dense `0.660476`, q4 `0.659998`, q8 `0.655515`, `290` child vectors, q4 storage `0.438x` versus a 1024d parent budget and `1.751x` versus a 256d parent budget.

| Row | Best span/overlap | Dense nDCG@10 | q4 nDCG@10 | q8 nDCG@10 | Child vectors | q4 storage |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Eos 256d token-span | 64/16 | 0.647864 | 0.650561 | 0.652997 | 211 | 2.472656x vs 256d parent |
| Eos 128d compact token-span | 48/12 | 0.660476 | 0.659998 | 0.655515 | 290 | 0.438x vs 1024d parent; 1.751x vs 256d parent |

These rows narrowly beat the Eos default repo-docs dense row (`0.644872`) on this lane, but they still trail the external compact child-vector references, including Qwen3 0.6B 128d child q4 `0.739269` and mxbai-large 128d child q4 `0.691586`. Provenance lives in `runs/eos-lc-token-span-sweep-v1-20260619T004023Z`, `runs/eos-lc-compact-token-span-sweep-v1-20260619T005356Z`, `.tiller/scratch/codex/eos-lc-token-span-sweep-v1-report.md`, and `.tiller/scratch/codex/eos-lc-compact-token-span-sweep-v1-report.md`. The current promoted-default capped cache rerun at `runs/eos-current-default-long-context-wedge-cache-v1-20260623T055955Z/` used the local repo-docs dataset plus matching local Qwen3/mxbai child-vector caches only. All rows have `quality_claim=false`: Eos direct nDCG@10 `0.634327008`, Eos sparse token-span dense/q4 `0.656691634`/`0.639676979`, best Eos fusion `0.638514924`, Qwen3 dense/q4 `0.807649216`/`0.735872282`, and mxbai dense/q4 `0.783370960`/`0.750464207`; all recall@100 values saturated at `1.0`. Keep `quality_claim=false`: this is tiny heuristic-qrels repo-docs cache/harness evidence, not LongEmbed proof, and it does not prove long-token consumption beyond the capped observed tokenizer-output length.

Current s40 official capped LongEmbed comparisons remain diagnostic and non-promotable. The official doc20/query20 external-cache blocker is cleared in `runs/eos-official-longembed-external-qwen3-mxbai-compare-v2-{qmsum,2wikimqa}/`: on `qmsum`, Eos direct is `0.517955842`, best Eos fusion is `0.536813696`, Qwen3 q4 is `0.876293470`, and mxbai q4 is `0.806946535`; on `2wikimqa`, Eos direct is `0.739153782`, best Eos fusion is `0.744085103`, and both Qwen3 q4 and mxbai q4 are `1.000000000`. The promoted-package sidecar retarget run `runs/eos-current-default-retarget-official-longembed-v1-qmsum-20260623T063632Z/` confirms retargeted sidecars and token-span export over QMSum doc20/query20 with max observed document tokens `4096`: Eos direct `0.517955842`, Eos token-span dense `0.547133297`, Eos token-span q4 `0.567516406`, best Eos fusion `0.540547745`, Qwen3 q4 `0.848162307`, and mxbai q4 `0.828480363`; all recall@100 values are `1.0`, all rows are `quality_claim=false`, and 2WikiMultihopQA was not rerun in this slice. Sparse encoder rows are available in `runs/eos-official-longembed-sparse-encoder-rows-v1-{qmsum,2wikimqa}/`, but they are host-reference retrieval-cache evidence and quality-negative versus direct: `qmsum` sparse dense/q4 `0.458166919`/`0.420963072`, `2wikimqa` sparse dense/q4 `0.504805535`/`0.505901435`. Preserve `quality_claim=false`, the capped doc20/query20 caveat, the external-cache caveat, the host-reference sparse-encoder caveat, and the external child-count/storage caveat when citing these rows.

`eos eval-retrieval-multivector-turboquant` also has diagnostic parent rollup controls for these child-vector caches: `--aggregation max|top2-mean|top3-mean|top5-mean` and `--child-count-penalty FLOAT`. The default remains max-child with no penalty. On the compact 128d `48/12` repo-docs token-span fixture, anchor q4 nDCG@10 was `0.659998`; max-child with penalty `0.01` reached q4 `0.665813`, closing only a small part of the external gap, while top-N mean hurt this fixture. Treat these flags as an eval/profile surface only, not a model-quality claim or default-promotion signal.

Calibrate hybrid retrieval before using `eos-hybrid` rows as product evidence:

```bash
EOS_REPO_ROOT=$PWD \
EOS_HYBRID_CAL_MODE=vectors \
EOS_HYBRID_CAL_VECTOR_ROOT=runs/external-vector-caches/qwen3-0.6b \
EOS_HYBRID_CAL_VECTOR_BACKEND=qwen3 \
EOS_HYBRID_CAL_DATASETS=nfcorpus \
EOS_HYBRID_CAL_DEV_SPLIT=dev \
EOS_HYBRID_CAL_TEST_SPLIT=test \
ferrous-wheel run scripts/calibrate_eos_embed_hybrid_retrieval.fw
```

The calibration run measures dense, BM25, and candidate hybrid settings on dev, chooses by nDCG@10 with MRR@10 and recall@100 tie-breakers, then evaluates only the selected setting on test. Its summary JSON/TSV/Markdown includes dense/BM25 comparisons, selected test metrics, protection gate deltas against dense, command logs, and optional sentinel query rows from `EOS_HYBRID_CAL_SENTINEL_QUERY_IDS`. Hybrid per-query `top_k` rows include optional component evidence for each fused candidate: dense/BM25 source ranks, raw scores, and minmax/zscore normalized scores when available. When applying a selected setting with `eos eval-retrieval-hybrid` or `eos eval-retrieval-vectors-hybrid`, `--dense-protect-top-k N` preserves the original dense top-N prefix before appending the fused hybrid tail; leave it at `0` for the unguarded fusion order.

Retrieval metric glossary:

- `ndcg_at_10` and `ndcg_at_100`: rank-discounted relevance quality at shallow and broad retrieval depths.
- `mrr_at_10`: reciprocal rank of the first relevant result within the first 10 results.
- `precision_at_1`, `precision_at_5`, and `precision_at_10`: relevant results divided by the fixed cutoff.
- `hit_at_1`, `hit_at_5`, and `hit_at_10`: whether at least one relevant result appears by the cutoff, averaged across queries.
- `map_at_10` and `map_at_100`: mean average precision over relevant positives up to the cutoff.
- `recall_at_10` and `recall_at_100`: fraction of qrels positives recovered by the cutoff.

For default-shipping decisions, read these together. nDCG and MAP catch ranking quality, precision/hit@k reflect first-screen usefulness, and recall@100 protects downstream reranking or broader candidate generation. Compression rows (`vector_bytes`, `dense_vector_bytes`, `compression_ratio`) describe index footprint; throughput rows (`documents_per_second`, `queries_per_second`, `scores_per_second`, `docs_per_second`) describe encoder, cache-load, quantization, and scoring speed depending on the path. Cached external-vector rows do not measure live provider or model encoder throughput; they only measure loading/scoring the already-exported vectors.

Score external embedders by exporting document and query vector JSONL caches, then run the same BEIR metric code:

```bash
python3 scripts/export_qwen3_retrieval_vectors.py \
  --dataset-dir datasets/eos-embed-v1/raw/scifact/scifact \
  --output-root runs/external-vector-caches/qwen3-0.6b \
  --dataset-name scifact \
  --model-name Qwen/Qwen3-Embedding-0.6B \
  --batch-size 16

go run ./cmd/eos eval-retrieval-vectors \
  --dataset scifact \
  --backend qwen3 \
  --artifact Qwen/Qwen3-Embedding-0.6B \
  --doc-vectors runs/external-vector-caches/qwen3-0.6b/scifact/doc-vectors.jsonl \
  --query-vectors runs/external-vector-caches/qwen3-0.6b/scifact/query-vectors.jsonl \
  --metrics-json runs/scifact.qwen3.retrieval.metrics.json \
  datasets/eos-embed-v1/raw/scifact/scifact
```

`scripts/export_qwen3_retrieval_vectors.py` keeps Qwen3/SentenceTransformers dependencies outside the Go runtime. It writes `embedding` arrays and requests normalized embeddings by default; the Eos evaluator normalizes vectors again before scoring. Use caps such as `--max-docs 200 --max-queries 20` for smoke runs.

Each vector cache row must contain `id` or `_id` plus one of `vector`, `embedding`, or `values`, for example `{"id":"doc-1","vector":[0.1,0.2]}`. The evaluator normalizes vectors, requires matching dimensions, scores cosine/dot-product ranking, and emits the same `manta.embedding_retrieval_metrics.v1` JSON as `eval-retrieval`. To add external rows to the Ferrous Wheel scoreboard, place caches at `<EOS_SCOREBOARD_EXTERNAL_VECTOR_ROOT>/<dataset>/doc-vectors.jsonl` and `query-vectors.jsonl`, then set `EOS_SCOREBOARD_EXTERNAL_VECTOR_DATASETS`, `EOS_SCOREBOARD_EXTERNAL_VECTOR_BASELINE`, `EOS_SCOREBOARD_EXTERNAL_VECTOR_BACKEND`, and optionally `EOS_SCOREBOARD_EXTERNAL_VECTOR_ARTIFACT`.

Compare the same external caches after TurboQuant IP document-vector compression:

```bash
go run ./cmd/eos eval-retrieval-vectors-turboquant \
  --dataset scifact \
  --backend qwen3 \
  --artifact Qwen/Qwen3-Embedding-0.6B \
  --doc-vectors /data/external-vector-caches/scifact/doc-vectors.jsonl \
  --query-vectors /data/external-vector-caches/scifact/query-vectors.jsonl \
  --bits 2,4,8 \
  --metrics-json runs/scifact.qwen3.turboquant.metrics.json \
  --metrics-tsv runs/scifact.qwen3.turboquant.metrics.tsv \
  /data/manta/datasets/eos-embed-v1/raw/scifact/scifact
```

This external-cache TurboQuant path is the main apples-to-apples comparison surface for CorkScrewDB default promotion: BGE, Qwen, Jina, Voyage, Cohere, OpenAI, and sealed Eos rows can all be scored as dense vectors and as q2/q4/q8 TurboQuant document indexes without adding provider SDKs to Eos.

The scoreboard can append TurboQuant rows for the same external caches:

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

Append local sealed Eos TurboQuant rows in the same scoreboard:

```bash
EOS_SCOREBOARD_TURBOQUANT=1 \
EOS_SCOREBOARD_TURBOQUANT_BITS=4 \
EOS_SCOREBOARD_TURBOQUANT_RERANK_OVERFETCH=200 \
EOS_SCOREBOARD_TURBOQUANT_RERANK_STORAGE=fp16 \
EOS_SCOREBOARD_TURBOQUANT_BASELINE=eos-turboquant \
EOS_SCOREBOARD_TURBOQUANT_RERANK_BASELINE=eos-turboquant-rerank \
ferrous-wheel run scripts/score_manta_embed_v1_baselines.fw
```

The direct and rerank rows share the same metrics JSON but use separate scoreboard baselines. Gate the promoted compact reranked profile with `--baseline eos-turboquant-rerank --method turboquant_ip_b4_overfetch200_fp16_rerank --bits 4 --metrics ndcg_at_10,recall_at_100,total_compression_ratio`. Direct q4 and direct q8 are useful diagnostics, but they are not default-promotion candidates; the promoted q4 profile is two-stage q4 retrieval plus fp16 sidecar rerank.

Qwen3 is now locally consolidated for SciFact and NFCorpus only. Its useful compact external row is direct q8 at about `3.98x` vector compression, with SciFact q8 nDCG@10 `0.704128` and NFCorpus q8 nDCG@10 `0.368763`. mxbai remains stronger than Qwen3 in the existing local SciFact/NFCorpus evidence. Qwen3 FiQA is still subset-only, so do not use it for full short-set claims until the cache is repaired.

Run the TurboQuant retrieval storage/scoring gate against the same sealed candidate and a capped BEIR-style corpus:

```bash
go run ./cmd/eos eval-retrieval-turboquant \
  --dataset scifact \
  --max-docs 200 \
  --max-queries 20 \
  --bits 2,4,8 \
  --rerank-overfetch 200 \
  --rerank-storage fp16 \
  --metrics-json runs/turboquant-retrieval-scifact.json \
  --metrics-tsv runs/turboquant-retrieval-scifact.tsv \
  /path/to/eos-embed-v1.sealed.mll \
  /data/manta/datasets/eos-embed-v1/raw/scifact
```

The TurboQuant gate embeds the corpus once, records the dense float32 reference nDCG@10/recall@100, then quantizes document vectors with `m31labs.dev/turboquant` IP-preserving quantizers and reports per-bit quality deltas, vector bytes, rerank storage, rerank sidecar bytes, total vector bytes, compression ratio, total compression ratio, docs/s, scores/s, per-query p50/p95/p99/max scoring latency, and optional rerank overfetch/score counts. Use the capped command for smoke/release checks and remove the caps for a full CorkScrewDB-relevant vector-index promotion run.

For the promoted compact policy, prefer the repo-local serving proxy:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_eos_default_embedder_serving.fw
```

It selects q4/fp16/rerank-overfetch=200 and writes `summary.tsv` plus `manifest.json` under `runs/eos-default-embedder-serving-smoke-<timestamp>/`. This is not an actual CorkScrewDB API smoke; it is the in-repo TurboQuant serving proxy. For the promoted narrow release artifact, strict seeded compact gating against the s40 q4/fp16/o200 anchor passed: SciFact unchanged; NFCorpus nDCG delta `+0.000174802593872`, recall delta `+0.000006604583581`; FiQA nDCG delta `+0.000180234154897`, recall delta `+0.000771604938272`; macro nDCG delta `+0.000118345582923`, recall delta `+0.000259403173951`; total compression `1.5900621118x`. The s40 capped serving smoke in `runs/eos-default-embedder-serving-smoke-20260620T161633Z/` remains predecessor proxy evidence. Treat this as compact promotion evidence tied to release sealed SHA256 `e0eca16ff34ebb88ca96862d58c3ac7f02dbf4b124599fdf96f25344ac02e408`, not CorkScrewDB API smoke or hosted-model parity. The formal seeded s40 q4/fp16/o200 anchor remains predecessor provenance.

Attach energy/cost evidence with the bounded power gate:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/bench_eos_default_embedder_serving_energy.fw
```

The energy gate runs two local phases under the same caps: encoder-only vector export with `eos export-retrieval-vectors`, then the TurboQuant serving proxy with `eos eval-retrieval-turboquant`. It samples GPU power through `nvidia-smi` and writes `manifest.json`, `summary.tsv`, `power-samples.jsonl`, command logs, encoder vector-export manifest, and TurboQuant metrics under `runs/eos-default-embedder-serving-energy-<timestamp>/`. Encoder-only export measures both documents and queries, so its cost denominator is `workload_item_count = documents + queries` and `energy_joules_per_workload_item`; it does not report `energy_joules_per_query`. The index/scoring phase reports query-denominated `energy_joules_per_query`. When power telemetry is unavailable or insufficient, the manifest records `overall_status=unsupported` or `indeterminate` and omits denominator-specific energy costs rather than inventing them. Set `EOS_ENERGY_BENCH_REQUIRE_POWER=1` only for machines where GPU power telemetry is expected and missing telemetry should fail the gate.

For the local flat CorkScrewDB child-vector API path, run:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_corkscrewdb_child_vectors.fw
```

The default run generates a tiny synthetic time-series child-vector cache, loads children into CorkScrewDB with `PutVector`, searches query vectors with `SearchVector`, rolls child hits up to parent IDs by max score, and writes per-bit metrics, `summary.tsv`, and `manifest.json` under `runs/eos-corkscrewdb-child-vector-smoke-<timestamp>/`. Set `EOS_CORKSCREW_SMOKE_CHILD_VECTORS`, `EOS_CORKSCREW_SMOKE_QUERY_VECTORS`, and `EOS_CORKSCREW_SMOKE_QRELS` to reuse a full child-vector cache. `EOS_CORKSCREW_SMOKE_OVERFETCH` also accepts comma-separated values such as `100,12468`; the separate-child smoke loads one DB per bit width and reuses it across overfetch values so serving recall/latency sweeps do not pay duplicate insert cost. Exhaustive/full child overfetch is useful when comparing against offline cache-evaluator parity, but it is expected to cost more latency. This is local flat API/storage/latency/quality smoke evidence only; it is not remote mode, federation, HNSW, or a model-quality benchmark.

To measure CorkScrewDB packed parent-object persistence instead of separate child entries, run the same smoke with `EOS_CORKSCREW_SMOKE_LAYOUT=packed_parent_multivectors`, `EOS_CORKSCREW_SMOKE_INDEX_TYPE=flat`, and `EOS_CORKSCREW_SMOKE_VECTOR_STORAGE=quantized_only`. Packed mode groups children by parent, calls `PutMultiVector` once per parent, and queries parents with exact local flat `SearchParentsVector`; it records `layout`, `search_mode`, `parent_search_exact=true`, `parent_insert_count`, and `overfetch_children=0`. For strict storage-frontier measurements, set `EOS_CORKSCREW_SMOKE_PACKED_CHILD_ID_MODE=ordinal` and `EOS_CORKSCREW_SMOKE_PACKED_METADATA_MODE=none` or `minimal`; source child IDs and full metadata should be benchmarked as separate richer-record envelopes. Compare measured DB directory multiple against the default `separate_child_vectors` row at the same bit width, and keep vector payload accounting separate from persisted DB directory size.

For the pure multi-vector storage/accounting frontier, run:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_eos_multivector_budget_frontier.fw
```

This smoke calls `go run ./cmd/eos plan-multivector-storage` for the default `128d` compact-child versus `3072d` dense-baseline shape, records planner JSON, command logs, `summary.tsv`, and `manifest.json` under `runs/eos-multivector-budget-frontier-smoke-<timestamp>/`, and gates current per-child-entry fit counts at q2 >= 181, q4 >= 100, q8 >= 64 plus packed-parent target fit counts at q2 >= 341, q4 >= 180, q8 >= 93 children per dense-vector budget. Interpret it beside the time-series and CorkScrewDB API smokes as byte accounting only: it does not measure retrieval quality, CorkScrewDB API latency, or measured DB-directory cost.

For the product/use-case frontier, run:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_eos_multivector_usecase_frontier.fw
```

This smoke also calls the planner, but records named scenarios in `summary.tsv` and `manifest.json`: same-dim 100-child control, 100-child document spans, 100 time-series windows derived from `series_length=1648`, `window_size=64`, `window_stride=16`, a 180-child event/trace timeline, and a 341-child q2 frontier. Its gates keep the claim narrow: same-dim 100-child packed q2/q4 fails, 3072d-baseline packed q2/q4 100-child scenarios fit, packed q4 fits 180 children at the edge, and packed q2 fits 341 children. q8 rows are reported as contrast rather than required to fit.

For the combined cache-only quality plus overhead-aware budget gate, run:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_eos_multivector_budget_quality.fw
```

The default smoke reuses the existing Eos 128d SciFact child cache under `runs/eos-128d-child-cache-quality-20260615T000000Z/`, calls `eval-retrieval-multivector-turboquant`, infers the 128d child shape from the dense child-vector bytes, then calls `plan-multivector-storage` with `--baseline-dim 3072 --vector-overhead-bytes 32` and a `100` child-vector fit gate. S40-era predecessor interpretation: direct q4 is near dense on this cache (`ndcg@10` drop about `0.002630`, `recall@100` drop about `0.001667`) while the overhead-aware planner fits `123` q4 child vectors in one 3072d dense-vector budget, so 100 child vectors fit with overhead. It remains cache-only evidence, not CorkScrewDB API latency or DB-directory measurement. Refresh before using it as new-default evidence.

For the actual local flat CorkScrewDB API budget-quality smoke on the compact Eos child cache, run:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_eos_corkscrewdb_budget_quality.fw
```

This wrapper feeds `runs/eos-128d-child-cache-quality-20260615T000000Z/scifact/child-doc-vectors.jsonl`, matching query vectors, and SciFact test qrels into `scripts/smoke_corkscrewdb_child_vectors.fw` with q4 overfetch `100,12468`, then joins the API metrics to the overhead-aware planner row for `--dim 128 --baseline-dim 3072 --vector-overhead-bytes 32 --sidecar-storage none`. The default run writes `summary.tsv`, `manifest.json`, command logs, nested CorkScrewDB artifacts under `corkscrewdb/`, and `planner.json` under `runs/eos-corkscrewdb-budget-quality-smoke-<timestamp>/`. Historical API result: q4 overfetch100 records `ndcg@10=0.407586`, `recall@100=0.724111`, DB directory multiple `0.048935x`, and p95 search latency about `11.8ms`; q4 full overfetch records `recall@100=0.741889`; the planner fits `123` q4 children in one 3072d dense-vector budget and the 100-child target fits. The packed-parent budget wrapper now forwards and records compact packed knobs. Promoted narrow-default packed-parent API run `runs/eos-narrow-default-corkscrewdb-budget-quality-packed-q4q8-main-20260623T050216Z` used fresh vectors in `runs/eos-vector-caches/eos-narrow-default-scifact-child-w128-o32-128d` (`documents=5183`, `child_vectors=12468`, `queries=300`, `dimension=128`, `backend=cuda`) and CorkScrewDB repo `/home/draco/work/corkscrewdb`. It measured `5,183` parents, `12,468` children, and `128d` vectors through local flat `PutMultiVector`/exact `SearchParentsVector` with `packed_parent_multivectors`, `packed_metadata_mode=none`, `packed_child_id_mode=ordinal`, `quantized_only` storage, and flat index: q4 release-gate evidence recorded nDCG@10 `0.452380`, recall@100 `0.758556`, DB directory multiple `0.041676x`, vector payload multiple `0.013312x`, p95 `16.076127ms`, planner fit `180`, target fit `true`, and target storage multiple `0.554545x`; q8 diagnostic evidence recorded nDCG@10 `0.472132`, recall@100 `0.773556`, DB directory multiple `0.066734x`, vector payload multiple `0.025841x`, p95 `36.113081ms`, planner fit `93`, target fit `false`, and target storage multiple `1.074026x`. The s40, nf005, and older targeted-v3 diagnostic runs remain predecessor comparison/provenance only. This is local flat CorkScrewDB evidence for compact packed child vectors beside the cache-only quality smoke and pure planner smoke. It is separate from q4/fp16 sidecar rerank evidence and is not remote mode, federation, HNSW, hosted parity, q4/fp16 alias promotion, a native Matryoshka embedding result, or a service SLO. q8 misses target fit and the DB directory gate.

For HNSW index-search validation over the current promoted narrow-default cache, `runs/eos-current-default-corkscrewdb-hnsw-refresh-v1-20260623T055809Z/` refreshed the same SciFact child-vector lane with `index_type=hnsw` and `vector_storage=raw`. q4 overfetch100 recorded nDCG@10 `0.445689`, recall@100 `0.735222`, vector payload multiple `0.013312x`, raw HNSW DB directory multiple `0.243494x`, p95 `2.324070ms`, planner fit `123`, and target fit `true`; q4 overfetch12468 recorded nDCG@10 `0.452380`, recall@100 `0.758556`, and p95 `49.635238ms`. The rebuild took about `399.435824s`. q8 was not run. CorkScrewDB rejects HNSW with `quantized_only`, so the DB directory multiple is raw HNSW persistence cost; the vector payload multiple and planner fields are compact q4 accounting only. This is local CorkScrewDB API evidence, not remote mode, federation, hosted parity, or a service SLO.

For API-level dense+sparse policy translation evidence, run `scripts/smoke_eos_corkscrewdb_hybrid_policy.fw`. It exports or reuses Eos BEIR document/query vector caches, loads a CorkScrewDB `WithSparse` collection, and compares dense-only `SearchMulti` to `SearchMulti` with `WeightedFusion{Dense:0.5,Sparse:0.5}` as the public-API equivalent of the selected `minmax_blend alpha=0.5` policy. `EOS_CORKSCREW_HYBRID_SMOKE_SPARSE_METHOD=tfidf_l2` preserves the historical lexical TF-IDF sparse channel, while `bm25_dot` encodes doc sparse values as BM25 `idf * tfWeight` (`k1=0.9`, `b=0.4`, avgdl over loaded docs) and query values as query term frequency.

Historical TF-IDF sparse translation rows:

| Run | Bits | Docs | Queries | Relevant pairs | Dense-only nDCG@10 / recall@100 | Hybrid nDCG@10 / recall@100 | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `runs/eos-corkscrewdb-hybrid-policy-smoke-20260620T232130Z/` | 8 | 600 | 20 | 22 | 0.729755 / 0.950000 | 0.928080 / 1.000000 | pass |
| `runs/eos-corkscrewdb-hybrid-policy-full-scifact-v1-20260620T232832Z/` | 8 | 5,183 | 300 | 339 | 0.565375 / 0.796444 | 0.703644 / 0.934556 | pass |
| `runs/eos-corkscrewdb-hybrid-policy-full-scifact-q4-v1-20260620T000000Z/` | 4 | 5,183 | 300 | 339 | 0.550000 / 0.799111 | 0.695287 / 0.936222 | pass |
| `runs/eos-corkscrewdb-hybrid-policy-full-nfcorpus-q4-v1-20260620T000000Z/` | 4 | 3,633 | 323 | 12,334 | 0.203265 / 0.241053 | 0.301358 / 0.293417 | pass |
| `runs/eos-corkscrewdb-hybrid-policy-full-fiqa-q4-v1-20260620T000000Z/` | 4 | 57,600 | 648 | 1,705 | 0.119637 / 0.345019 | 0.189401 / 0.482542 | pass |
| `runs/eos-corkscrewdb-hybrid-policy-full-scifact-q2-v1-20260620T000000Z/` | 2 | 5,183 | 300 | 339 | 0.472358 / 0.760889 | 0.688250 / 0.924556 | pass |

Treat these rows as CorkScrewDB API translation evidence with TF-IDF lexical sparse features. They remain useful compatibility evidence, but they are not exact `eval-retrieval-hybrid` BM25 parity. q2 still passes the hybrid-vs-dense API gate, but direct dense-only quality drops versus q8, so keep q2 framed as compact diagnostic evidence rather than a default-quality direct-vector profile.

BM25-dot sparse q4 API rows:

| Run | Bits | Docs | Queries | Relevant pairs | Dense-only nDCG@10 / recall@100 | Hybrid nDCG@10 / recall@100 | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `runs/eos-corkscrewdb-hybrid-policy-full-scifact-q4-bm25dot-v1-20260620T000000Z/` | 4 | 5,183 | 300 | 339 | 0.548851 / 0.793111 | 0.726845 / 0.936222 | pass |
| `runs/eos-corkscrewdb-hybrid-policy-full-nfcorpus-q4-bm25dot-v1-20260620T000000Z/` | 4 | 3,633 | 323 | 12,334 | 0.201400 / 0.233861 | 0.311750 / 0.291122 | pass |
| `runs/eos-corkscrewdb-hybrid-policy-full-fiqa-q4-bm25dot-v1-20260620T000000Z/` | 4 | 57,600 | 648 | 1,705 | 0.118427 / 0.341251 | 0.223099 / 0.499790 | pass |

narrow-default BM25-dot SearchMulti rows for `assets/corkscrewdb-default-embedder/corkscrewdb-default-embedder.mll` at SHA256 `e0eca16ff34ebb88ca96862d58c3ac7f02dbf4b124599fdf96f25344ac02e408`:

| Dataset | Docs | Queries | Dense-only nDCG@10 / recall@100 | Hybrid nDCG@10 / recall@100 | overall_pass |
| --- | ---: | ---: | ---: | ---: | --- |
| SciFact | 5,183 | 300 | 0.559571129519 / 0.796444444444 | 0.723139732393 / 0.932888888889 | true |
| NFCorpus | 3,633 | 323 | 0.205279642844 / 0.242243795394 | 0.316761336578 / 0.293902897482 | true |
| FiQA | 57,600 | 648 | 0.119552386600 / 0.352919418544 | 0.224427168561 / 0.502074751380 | true |

The BM25-dot rows are BM25-compatible sparse-dot API evidence: the sparse dot product matches the in-repo BM25 scoring formula for matching terms modulo float32 rounding, then CorkScrewDB applies its public `WeightedFusion` min-max normalization. The s40-era run remains predecessor local lexical+dense ranking-policy evidence over the previous default dense asset. These rows do not change default assets or aliases and do not prove remote mode, HNSW, federation, or hosted-model parity.

The same wrapper also supports `EOS_CORKSCREW_BUDGET_QUALITY_LAYOUT=single_parent_vectors` for a real document-span layout baseline. On the existing full Eos SciFact child cache at `runs/eos-vector-caches/eos-embed-v1-scifact-child-w128-o32-full/scifact/`, the wrapper compared q4/q8 packed `none`/`ordinal` parent multivectors with mean-pooled single-parent vectors over `5,183` parents, `12,468` child spans, `300` queries, and `256d` vectors. Packed run `runs/eos-real-scifact-full-packed-none-ordinal-q4q8-diagnostic2-20260616T000000Z/` recorded q4 `ndcg@10=0.449435`, `recall@100=0.773111`, DB directory multiple `0.032900x`, and p95 `15.934540ms`; q8 recorded `0.461862`, `0.774778`, `0.057958x`, and `30.366188ms`. Single-parent run `runs/eos-real-scifact-full-single-parent-q4q8-diagnostic-20260616T000000Z/` recorded q4 `ndcg@10=0.406498`, `recall@100=0.743111`, DB directory multiple `0.022411x`, and p95 `7.075771ms`; q8 recorded `0.422597`, `0.745889`, `0.032827x`, and `13.242884ms`. Treat this as real local flat document-span layout evidence: packed preserves child-span evidence better at higher storage/latency, while single-parent is smaller and faster but mean-pools span structure away. The diagnostic run disabled old q4 quality/storage/planner floors; it does not claim remote mode, HNSW, federation, or native Matryoshka behavior.

For the same compact cache through CorkScrewDB HNSW/raw-vector indexing, run:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_eos_corkscrewdb_hnsw_quality.fw
```

This wrapper uses `scripts/smoke_corkscrewdb_child_vectors.fw` with `EOS_CORKSCREW_SMOKE_INDEX_TYPE=hnsw`, `EOS_CORKSCREW_SMOKE_VECTOR_STORAGE=raw`, and rebuilds HNSW after loading before joining the same planner row. It writes `summary.tsv`, `manifest.json`, command logs, nested CorkScrewDB artifacts under `corkscrewdb/`, and `planner.json` under `runs/eos-corkscrewdb-hnsw-quality-smoke-<timestamp>/`. Current promoted narrow-default HNSW result `runs/eos-current-default-corkscrewdb-hnsw-refresh-v1-20260623T055809Z` records q4 overfetch100 nDCG@10 `0.445689`, recall@100 `0.735222`, raw HNSW DB directory multiple `0.243494x`, vector payload multiple `0.013312x`, rebuild time about `399.435824s`, p95 search latency `2.324070ms`, planner fit `123`, and target fit `true`; q4 overfetch12468 records nDCG@10 `0.452380`, recall@100 `0.758556`, and p95 `49.635238ms`. q8 HNSW was not run. CorkScrewDB rejects `quantized_only` with HNSW, so this smoke trades compact quantized-only persistence for raw-vector HNSW index-search validation. Do not use its raw DB directory multiple as the q4 quantized-only storage claim; keep the q4 vector payload/planner fit columns as separate planner accounting. This is local API evidence only, not remote mode, federation, hosted parity, or a service SLO.

For synthetic time-series window child vectors through the actual local flat CorkScrewDB API, run:

```bash
EOS_REPO_ROOT=$PWD \
ferrous-wheel run scripts/smoke_eos_corkscrewdb_timeseries_windows.fw
```

This wrapper first runs `scripts/smoke_eos_timeseries_window_vectors.fw` into a nested `timeseries/` directory, then feeds the generated `child-doc-vectors.jsonl`, `query-vectors.jsonl`, and `qrels.tsv` into `scripts/smoke_corkscrewdb_child_vectors.fw` using flat `quantized_only` storage by default. It writes a joined `summary.tsv` and `manifest.json` under `runs/eos-corkscrewdb-timeseries-window-smoke-<timestamp>/`, plus nested `timeseries/` and `corkscrewdb/` artifacts. S40-era predecessor result: `5` parents, `25` child windows, `5` derived windows per parent; q4 flat/quantized_only records `ndcg@10=1.000000`, `recall@100=1.000000`, planner fit `123`, vector payload multiple `0.034180x`, and DB directory multiple `0.117643x`; q8 records `ndcg@10=0.926186`, `recall@100=1.000000`, planner fit `75`, vector payload multiple `0.060221x`, and DB directory multiple `0.143685x`. Observed p95 latency for the default synthetic smoke has been sub-ms, but it varies by run. Treat this as proof that cheap child/window vectors under parent series are usable through local CorkScrewDB `PutVector`/`SearchVector`; the inputs are synthetic text-rendered numeric windows, not a trained numeric time-series encoder, and DB directory size is measured separately from vector-payload/planner accounting. Refresh before using it as new-default evidence.

For a contrastive parent-layout smoke, set `EOS_CORKSCREW_TS_WINDOW_SCENARIO=contrastive_needle` and compare `EOS_CORKSCREW_TS_WINDOW_LAYOUT=packed_parent_multivectors` against `single_parent_vectors` on the same generated local-window inputs. The scenario keeps each parent mostly on a shared baseline and inserts one short distinctive regime, so mean-pooling can dilute the query-relevant local window while packed child multivectors keep the window/facet available. Verified q2 runs recorded `24` parents, `264` child windows, and `11` windows per parent: packed q2 `none`/`ordinal` run `runs/eos-corkscrewdb-timeseries-contrastive-needle-q2-packed-20260616T000000Z/` recorded `ndcg@10=0.769694`, `recall@100=1.000000`, planner packed fit `341`, packed planner multiple `0.034740x`, vector payload multiple `0.032227x`, DB directory multiple `0.042772x`, and p95 `0.045773ms`; single-parent q2 run `runs/eos-corkscrewdb-timeseries-contrastive-needle-q2-single-parent-20260616T000000Z/` recorded `ndcg@10=0.677304`, `recall@100=1.000000`, vector payload multiple `0.002930x`, DB directory multiple `0.019687x`, and p95 `0.032332ms`. q4 showed the same direction but smaller split: packed `0.831824` nDCG@10 versus single-parent `0.818766`, both with recall@100 `1.000000`. Treat this as a local-window/facet preservation smoke, not the q2-341 storage-capacity proof.

For a q4-only parent-variant scale/stress profile with 100 child windows per parent and enough parent objects to separate fixed DB-directory overhead from per-vector TurboQuant payload accounting, run:

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

The 1648/64/16 shape derives exactly 100 covering windows per parent. With `EOS_CORKSCREW_TS_WINDOW_SERIES_VARIANTS=20`, the five query patterns expand to `100` parent series and `10,000` child windows while keeping the original five pattern queries; each query has `20` relevant parent variants, for `100` relevant query-parent pairs. Verified separate-child run `runs/eos-corkscrewdb-timeseries-window-scale-q4-100-variants20-20260616T000000Z/` recorded q4 flat/`quantized_only` with planner fit `123`, planner storage multiple `0.811688x`, vector payload multiple `0.683594x`, DB directory multiple `2.293748x`, DB bytes `2,818,558`, `ndcg@10=0.352927`, `recall@100=0.560000`, p95 `11.948319ms`, and overfetch `500`.

For packed parent-object persistence at the same q4 scaled shape, rerun with `EOS_CORKSCREW_SMOKE_LAYOUT=packed_parent_multivectors`, flat `quantized_only`, and exact parent search. The packed smoke also supports `EOS_CORKSCREW_SMOKE_PACKED_METADATA_MODE=full|minimal|none` and `EOS_CORKSCREW_SMOKE_PACKED_CHILD_ID_MODE=source|ordinal`; defaults preserve the full/source behavior. Verified packed `none`/`ordinal` run `runs/eos-corkscrewdb-timeseries-window-scale-q4-100-variants20-packed-minimal-20260616T000000Z/` recorded `layout=packed_parent_multivectors`, `parent_insert_count=100`, `parent_search_exact=true`, `overfetch_children=0`, the same planner/vector payload multiples and `ndcg@10=0.352927`, plus `recall@100=1.000000`, DB directory multiple `0.844660x`, DB bytes `1,037,918`, and p95 `5.916649ms`. Against the prior full-metadata packed run, DB bytes were `0.421237x` as large, about `57.88%` lower; against the separate-child run, DB bytes were `0.368244x` as large, about `63.18%` lower. Treat the recall gain as exact parent rollup versus bounded separate-child overfetch, not as model-quality improvement.

Corrected q2-341 packed-parent frontier evidence is now a unified API run rather than planner math alone. Wrapper run `runs/eos-corkscrewdb-timeseries-window-q2-341-compact-v5-unified-20260616T000000Z/`, against CorkScrewDB commit `c208f9b50d29f9fdf19771c4b093332c7c8fd0b4` (`update(snapshot): Add compact format for quantized ordinal children`), generated the child/query/qrels inputs, packed planner evidence, and measured persisted DB directory evidence for q2 `128d`, `packed_parent_multivectors`, `packed_metadata_mode=none`, and `packed_child_id_mode=ordinal` with `100` parents, `34,100` child windows, and `341` windows per parent. It recorded `quantized_vector_bytes=36`, `quantized_child_bytes=1,227,600`, vector payload multiple `0.9990234375x`, packed planner bytes `12,308`, packed planner multiple `0.999025974025974x`, measured DB directory bytes `1,237,818`, DB directory multiple `1.0073388671875x`, `ndcg@10=0.4493940305106442`, `recall@100=1.000000`, and p95 `1.418733ms`, passing planner-fit, vector-payload, DB-directory, and p95 gates. With CorkScrewDB compact snapshot v5 ordinal encoding, the persisted DB directory is approximately one dense parent-vector budget for this strict shape; without that compact snapshot path, or for richer child records, keep DB directory cost separate.

These scaled time-series rows are synthetic text-rendered smoke evidence: they improve DB overhead measurement by scaling parent objects and show a local flat packed-parent storage win, but they are not production quality, remote/federation/HNSW evidence, or a trained numeric time-series encoder claim. Keep the DB directory max disabled unless a fresh run establishes a stable machine-local threshold.

Current measured release-artifact compact-profile result: q4/fp16 sidecar rerank at overfetch200 is the promoted narrow compact policy. It passes strict seeded non-regression against the s40 q4/fp16/o200 anchor with macro nDCG@10 `+0.000118345582923`, macro recall@100 `+0.000259403173951`, and total compression `1.5900621118x`. Do not generalize this result to direct q4, remote CorkScrewDB, HNSW, federation, or hosted-model parity.

Current dense local candidate:

```text
assets/corkscrewdb-default-embedder/corkscrewdb-default-embedder.mll
```

The s40 NFCorpus compact-mined narrow package is the measured dense and compact promotion candidate for the `eos-embed-v1` lineage. The durable CorkScrewDB external-provider asset lives at `assets/corkscrewdb-default-embedder/corkscrewdb-default-embedder.mll`; the ignored release run `runs/eos-s40-nfcorpus-compact-mined-narrow-candidate-v1-20260623T032556Z/` remains provenance. It was trained from `runs/eos-s40-nfcorpus-compact-mined-narrow-candidate-v1-20260623T032556Z/data/train-anchor-plus-nfcorpus-compact-mined-64.jsonl` with 102 protected anchor rows plus 64 NFCorpus train-mined compact hard-negative rows, LR `0.000000025`, HN `5`, compact objective `256:4=0.05`, and prefix seed `5581486560434873699`; release sealed SHA256 is `e0eca16ff34ebb88ca96862d58c3ac7f02dbf4b124599fdf96f25344ac02e408`, release package SHA256 is `188265db16992ab24be15e678c5f7e175bebad769e8d844e8b0f50ffc23bd5bf`, tokenizer SHA256 is `64cf63223cb3f97125040677a573e6ab6c625cff1f6f338f4e680a4c9f7a42f5`, and training JSONL SHA256 is `72830906ac4e9e57a6c108002187f059654a818ca61265ef2f14a445d88418b0`. Package and sealed inspection report `package verify: OK`. The s40, nf005, targeted-v3, and June 10 deephard-full packages remain predecessor evidence.

Repeatable release validation is `ferrous-wheel run scripts/smoke_eos_default_embedder_release.fw`; with `EOS_DEFAULT_EMBEDDER_RELEASE_CORKSCREWDB_ROOT=/home/draco/work/corkscrewdb`, it also starts CorkScrewDB without `-eos-artifact` and checks the temp manifest records embedding id `corkscrewdb-default-embedder` dim `256`. Current CorkScrewDB API evidence for the narrow default is `.tiller/scratch/codex/eos-narrow-default-corkscrewdb-api-refresh-v1-report.md`: local-flat packed-parent q4 records nDCG@10 `0.452380`, recall@100 `0.758556`, DB directory multiple `0.041676x`, vector payload multiple `0.013312x`, p95 `16.076127ms`, planner fit `180`, and target fit `true`; q8 remains diagnostic because target fit is `false` and DB directory multiple `0.066734x` misses the q4 gate. Full uncapped BM25-dot hybrid SearchMulti rows passed on SciFact, NFCorpus, and FiQA; see `.tiller/scratch/codex/eos-narrow-default-fiqa-hybrid-continuation-v1-report.md` for the FiQA continuation. HNSW and time-series rows remain predecessor or pending evidence unless explicitly rerun.

| Dataset | narrow default nDCG@10 | narrow default recall@100 | Delta vs s40 nDCG@10 | Delta vs s40 recall@100 |
| --- | ---: | ---: | ---: | ---: |
| SciFact | 0.5645379155 | 0.7964444444 | +0.0000000000 | +0.0000000000 |
| NFCorpus | 0.205745967860765 | 0.242066067459883 | +0.000174802593872 | +0.000007371369600 |
| FiQA | 0.121260940614285 | 0.351678208622653 | +0.000000000000000 | +0.000000000000000 |

For the June 10 strict anchor, pairwise eval-only rows completed with `optimizer_updates=0`, CUDA forward backend, eval AUC `0.825890`, and hard AUC `0.812725`.

Run a full retrieval-alignment round from an existing candidate when retrieval is behind the BM25 or open-model baselines:

```bash
EOS_REPO_ROOT=$PWD \
EOS_ALIGN_INITIAL_ARTIFACT=/path/to/eos-embed-v1.sealed.mll \
EOS_ALIGN_DATASETS=scifact,nfcorpus,fiqa \
ferrous-wheel run scripts/run_manta_embed_v1_retrieval_alignment_round.fw
```

The alignment harness writes a baseline scoreboard, run-local model-hard negatives, a retrained candidate package, a candidate scoreboard, and `retrieval-alignment-summary.tsv/json` with per-dataset nDCG@10 and recall@100 deltas.
Candidate training defaults to batch `64` and one explicit hard negative per query. This keeps full mined rounds bounded; larger batches or more explicit negatives should be treated as throughput experiments because pair work grows quickly.
Set `EOS_ALIGN_MODEL_HARD_DATASET_WEIGHTS=dataset=weight,...` to allocate more mined examples to weak datasets in the next mixed round.
Set `EOS_ALIGN_CANDIDATE_SOURCE_WEIGHTS=source=weight,...` to source-balance hard-negative batches when the train JSONL has `source` fields. Family keys such as `fiqa` also apply to mined sources such as `fiqa:model` unless an exact key is present.
Set `EOS_ALIGN_GATE_CANDIDATE=1` for promotion-style rounds. The gate records the summary, then fails when macro nDCG@10 is below `EOS_ALIGN_MIN_MACRO_NDCG_DELTA` or any dataset regresses beyond `EOS_ALIGN_MAX_DATASET_NDCG_REGRESSION`; `EOS_ALIGN_MIN_DATASET_NDCG_RATIO` can enforce an additional per-dataset nDCG ratio floor. Use `EOS_ALIGN_MAX_DATASET_RECALL_AT_100_REGRESSION` and `EOS_ALIGN_MIN_DATASET_RECALL_AT_100_RATIO` to also guard recall@100.
Set `EOS_ALIGN_CANDIDATE_CONTRASTIVE_LOSS=grouped_infonce` to test the query-grouped hard-negative objective. This counts only each query's own positive/negative candidate set in the training loss, which is useful as a retrieval-ranking ablation when corpus ranking regresses. The first grouped-only gated run rejected the candidate, so keep this flag as an experiment rather than a promotion path.
Set `EOS_ALIGN_CANDIDATE_CONTRASTIVE_LOSS=hybrid_infonce` to keep the global hard-negative InfoNCE matrix and add a weighted grouped term. The first weight-`0.25` gated run improved FiQA but regressed SciFact/NFCorpus. The first strict pass used `EOS_ALIGN_CANDIDATE_GROUPED_LOSS_WEIGHT=0.10`, `EOS_ALIGN_CANDIDATE_LR=0.000025`, and `EOS_ALIGN_CANDIDATE_SOURCE_WEIGHTS=scifact=1,nfcorpus=2,fiqa=1`.
Model-hard and BM25 mining can now emit `teacher_scores` for each query's positive plus explicit negatives. Use `eos export-teacher-score-requests <hard-negatives.jsonl> <requests.jsonl>` to hand the same candidate groups to an external scorer, then import scored rows with `eos import-teacher-scores`. Set `EOS_ALIGN_CANDIDATE_TEACHER_LOSS_WEIGHT` above zero to blend a soft teacher-distribution cross-entropy into hard-negative training, and tune `EOS_ALIGN_CANDIDATE_TEACHER_TEMPERATURE` when the teacher ranking is too sharp or too flat.
The latest reranker-teacher frontier audits generated request rows and working true-reranker plumbing, but still do not unblock training. `runs/eos-narrow-default-reranker-teacher-frontier-audit-v1-20260623T065159Z/` contains the older request-only NFCorpus train/dev frontier; it proved TEI `/rerank` plumbing with `cmd/teacher-bridge --mode tei-rerank`, and on RTX50/Blackwell the viable local image is `ghcr.io/huggingface/text-embeddings-inference:120-1.9`. The broader selected-diverse BGE audit in `runs/eos-nfcorpus-broader-cleaned-mining-quality-signal-v1-20260623T101157Z/` is historical audit evidence, not a training gate: `BAAI/bge-reranker-large` was strongest on that slice at `65/128` (`0.507812`), but agreement and margins were too weak.

The repaired subset64 BGE resmoke in `runs/eos-nfcorpus-repaired-subset64-bge-resmoke-v1-20260623T132032Z` improves the bounded NFCorpus diagnostic but remains audit-only / `no_train`. BGE base scored `35/64 = 0.546875`, margin `0.216738`, norm entropy `0.980823`; BGE large scored `33/64 = 0.515625`, margin `0.215547`, norm entropy `0.983283`; BGE v2-m3 scored `33/64 = 0.515625`, margin `0.194146`, norm entropy `0.979424`. Duplicate/qrels-positive negative leaks are zero, selected-positive equals any-positive top1, and the repair improves prior cleaned subset64 BGE rows from base `29->35`, large `22->33`, and v2-m3 `10->33`; generated rows remain `request_only=true` and `train_allowed=false`, with no source promotion or training allowed.

The decisive full-frontier evidence is now the cleaned doc-id-provenance run root `runs/eos-nfcorpus-cleaned-full-reranker-frontier-jina-audit-v1-20260623T113125Z/` plus the mxbai-large HF audit at `runs/eos-mxbai-rerank-large-hf-subset-audit-v1-20260623T140353Z`. Jina v2 scored `7128/7128` request rows and failed teacher quality at `160/1188 = 0.134680` selected/any-positive top1. Reusing the same cleaned frontier and TEI cache, `BAAI/bge-reranker-large` scored all `7128/7128` rows in `1:03.02` with `duplicates_skipped=0`, imported `1188` examples with missing `0`, audited `1188` scored / `0` missing / `0` duplicate-positive negatives, and produced artifacts under `debug-bge-large-full-clean-v1/` plus `data/nfcorpus.train-dev.cleaned-full-frontier.bge-large.*`. BGE-large selected/any-positive top1 was worse than Jina at `142/1188 = 0.119529`, mean normalized entropy was `0.993488`, and the filter kept `142` while clearing `1046`. `mixedbread-ai/mxbai-rerank-base-v2` remains blocked for TEI `120-1.9` serving, but HF/SentenceTransformers scoring was mechanically unblocked in `runs/eos-mxbai-rerank-base-hf-unblock-smoke-v1-20260623T124344Z`: the tiny raw-logit smoke passed, then the deterministic cleaned-frontier subset64 scored/imported/audited/filtered with positive top1 `1/64 = 0.015625`, mean margin `-7.836426`, mean normalized entropy `0.476268`, filter kept `1`, and cleared `63`. That subset failed the `0.35` quality gate, so no full `1188`-example mxbai-base metric exists; record it as `subset_quality_gate_failed` / `no_train` rather than comparing it to the full Jina/BGE rows. `mixedbread-ai/mxbai-rerank-large-v2` used the repaired source subset from `runs/eos-nfcorpus-repaired-frontier-subset64-v1-20260623T131418Z/`; its raw-logit smoke passed at relevant `12.25` and irrelevant `-0.5625`, repaired subset64 passed at `37/64 = 0.578125`, mean margin `1.179688`, norm entropy `0.442453`, kept `37/64`, but full cleaned-frontier scoring covered `7128` request rows / `1188` examples with missing `0` and duplicate-positive negative candidates `0` and failed at `172/1188 = 0.144781`, mean margin `-2.897175`, norm entropy `0.612920`, kept `172/1188`. Decision: Jina, BGE-large, the mxbai-base subset, and mxbai-large full-frontier evidence are `teacher_quality_blocked` / `no_train` for this cleaned frontier; all generated rows remain `request_only=true` and `train_allowed=false`, and no training should use them. The next benchmark action should pivot to capacity/bootstrap, a compact/Matryoshka objective, or a different data strategy rather than treating this cleaned full frontier as trainable teacher data.
A lower-LR ratchet from that strict-pass checkpoint with `EOS_ALIGN_CANDIDATE_GROUPED_LOSS_WEIGHT=0.05`, `EOS_ALIGN_CANDIDATE_LR=0.0000125`, and `EOS_ALIGN_CANDIDATE_SOURCE_WEIGHTS=scifact=1,nfcorpus=3,fiqa=1` improved pairwise AUC but failed the retrieval gate. Do not treat repeated ratchets on the same mined blend as the default next step; remine model-hard negatives from the promoted artifact or add teacher-score distillation.
A fresh-mining round from the strict-pass checkpoint with `EOS_ALIGN_MODEL_HARD_DATASET_WEIGHTS=scifact=1,nfcorpus=3,fiqa=1`, `EOS_ALIGN_CANDIDATE_GROUPED_LOSS_WEIGHT=0.05`, `EOS_ALIGN_CANDIDATE_LR=0.0000125`, and `EOS_ALIGN_CANDIDATE_SOURCE_WEIGHTS=scifact=1,nfcorpus=2,fiqa=1` passed the retrieval gate: macro nDCG@10 improved from `0.138397` to `0.145568`. Because recall@100 dipped, future promotion-style rounds should set an explicit recall floor when using this recipe.
A teacher-distilled follow-up from that checkpoint used fresh model-hard examples with `teacher_scores`, `EOS_ALIGN_CANDIDATE_TEACHER_LOSS_WEIGHT=0.20`, `EOS_ALIGN_CANDIDATE_LR=0.000010`, and recall floors. The first full gated run with `EOS_ALIGN_CANDIDATE_SOURCE_WEIGHTS=scifact=1,nfcorpus=2,fiqa=1` improved macro nDCG@10 to `0.146301` but failed the NFCorpus nDCG floor. Reusing the same teacher-scored JSONL with `scifact=1,nfcorpus=3,fiqa=1` fixed that failure and raised macro nDCG@10 to `0.147862` while recall@100 stayed flat or improved on all three retrieval sets. Holding that recipe and softening `EOS_ALIGN_CANDIDATE_TEACHER_TEMPERATURE` to `1.5` raised macro nDCG@10 again to `0.148143`; the retrieval comparison gate passed with SciFact `0.331139`, NFCorpus `0.084325`, and FiQA `0.028967`. Temperature `1.25` and `2.0` also passed the gate but landed lower at macro `0.147645` and `0.148029`, respectively. LR `0.000008` at temperature `1.5` passed at macro `0.147625` and improved NFCorpus to `0.084927`, but lost enough SciFact to keep LR `0.000010` as the current best. Source weights `scifact=1,nfcorpus=4,fiqa=1` raised SciFact to `0.331362` but failed the NFCorpus nDCG floor and regressed FiQA. Source weights `scifact=2,nfcorpus=3,fiqa=1` passed the older fresh-baseline comparison at macro `0.146288`, but pairwise guardrails fell to validation AUC `0.815529` / hard AUC `0.808770` and macro stayed below the current best, so the local source-weight sweep is closed. Full BM25 teacher-score coverage plus `example_zscore` normalization is now a near-anchor branch: SciFact `0.329417`, NFCorpus `0.085742`, FiQA `0.029123`, macro `0.148094`, validation AUC `0.823282`, and hard AUC `0.815203`. It passed the stale-baseline gate by `+0.002526` macro but missed the current best by `0.000049`. The follow-up SciFact-recovery sampler (`scifact=2,nfcorpus=3,fiqa=1`) failed early with validation/hard AUC `0.817674` / `0.810527` and SciFact `0.326679`, so local normalization/source reshuffling is closed; move to external/synthetic teacher signal.
A deeper Lane B mining round from the temperature-`1.5` best requested `9000` model-hard examples with `EOS_ALIGN_MODEL_HARD_NEGATIVES=5`, `EOS_ALIGN_MODEL_HARD_CANDIDATE_TOP_K=400`, and `EOS_ALIGN_CANDIDATE_HARD_NEGATIVES=2`. The run trained `13,038` blended hard-negative examples at `4262` train pairs/s with CUDA-backed forward/optimizer/activation/contrastive, but failed the promotion gate: macro nDCG@10 fell from `0.148144` to `0.143866`, SciFact fell to `0.320576`, and FiQA fell to `0.026453`. NFCorpus rose slightly to `0.084568`, so the next isolation run should reuse the same mined JSONL with one candidate hard negative.
The one-hard-negative reuse trained the same deep-mined JSONL at `5056` train pairs/s and improved NFCorpus to a new high-water mark of `0.087300`, but it still failed against the current best: SciFact `0.324630`, FiQA `0.025679`, macro `0.145870`, validation AUC `0.802976`, and hard AUC `0.800840`. Since the deep-mined JSONL already contains `5400` NFCorpus model-hard examples, the next reuse should drop the NF3 training source schedule and try balanced source weights.
Balanced source weights on that same deep-mined HN1 JSONL regressed further to macro `0.144915`: SciFact `0.322932`, NFCorpus `0.086364`, FiQA `0.025450`, validation AUC `0.794627`, and hard AUC `0.794320`. Do not continue source-sampler rescue on this file; the next local isolation should keep the NF3 HN1 shape but lower LR to test a smaller update.
Lowering the HN1 NF3 reuse to LR `0.000005` improved FiQA to `0.027508` and kept NFCorpus high at `0.086784`, but SciFact remained low at `0.323136`, macro landed at `0.145809`, validation AUC was `0.803438`, and hard AUC was `0.800508`. The remaining local check is reduced grouped pressure; otherwise move this lane to external-teacher work.
Reducing grouped pressure to `EOS_ALIGN_CANDIDATE_GROUPED_LOSS_WEIGHT=0.025` at LR `0.000005` gave the best Lane B deep-mined balance but still failed promotion: SciFact `0.325439`, NFCorpus `0.086645`, FiQA `0.027204`, macro `0.146429`, validation AUC `0.804674`, and hard AUC `0.801615`. Close this deep-mined file for balanced promotion and move the next improvement path to external teacher import or a larger `embed-m` run.
The first `embed-m` capacity probes separated mechanical viability from quality. The full target shape (`32768` vocab, max sequence `512`, dim `192`, hidden `384`, `3` repeats) initialized, but full-corpus tokenizer training stayed CPU-bound for more than fifteen minutes before any optimizer step, so true `32768`-vocab iteration needs a cached tokenizer artifact or faster trainer first. A cached-tokenizer probe (`16384` vocab, max sequence `512`) trained and sealed at batch `64`, processing `1.393M` actual train pairs in `19m20s` with CUDA forward/optimizer/activation/contrastive and `1460.78` train pairs/s, but rejected hard: validation AUC `0.595854`, hard AUC `0.598887`, SciFact `0.160753`, NFCorpus `0.060778`, FiQA `0.012688`, macro `0.078073`. A scratch-style cached-tokenizer run with pure `infonce`, LR `0.002`, and one epoch rejected even earlier with validation AUC `0.495137` and hard AUC `0.498731`. Do not continue blind random-start `embed-m`; the next larger-model proof needs dimension-compatible bootstrapping, staged pretraining, or imported external teacher scores.

Compare a retrieval-only candidate scoreboard against a prior alignment summary without rerunning the full alignment harness:

```bash
EOS_COMPARE_BASELINE_SUMMARY_JSON=/path/to/retrieval-alignment-summary.json \
EOS_COMPARE_CANDIDATE_SCOREBOARD_JSON=/path/to/candidate-scoreboard/scoreboard.json \
ferrous-wheel run scripts/compare_manta_embed_v1_retrieval_candidate.fw
```

The comparison writes `retrieval-comparison-summary.tsv/json` beside the candidate scoreboard and applies the same macro nDCG@10, per-dataset nDCG, and recall@100 floors by default.

If you want a binary runner instead of `run` mode:

```bash
ferrous-wheel build scripts/bench.fw -o bin/manta-bench
bin/manta-bench
```

Capture CPU or heap profiles for any `eos` command with:

```bash
EOS_CPU_PROFILE=/tmp/manta.cpu.pprof go run ./cmd/eos train-embed ...
EOS_MEM_PROFILE=/tmp/manta.mem.pprof go run ./cmd/eos train-embed ...
```

Then inspect with `go tool pprof -top /tmp/manta.cpu.pprof`.

For repeatable GPU A/B profiles, use the Ferrous Wheel harness:

```bash
EOS_REPO_ROOT=$PWD \
EOS_GPU_PROFILE_ASSETS=/path/to/assets/eos-embed-v1 \
EOS_GPU_PROFILE_TRAIN=/path/to/train-mini.jsonl \
EOS_GPU_PROFILE_EVAL=/path/to/eval-mini.jsonl \
EOS_GPU_PROFILE_VARIANTS=default,disable-batched-forward \
EOS_GPU_PROFILE_ENV_DISABLE_BATCHED_FORWARD=EOS_TRAIN_DISABLE_BATCHED_FORWARD=1 \
ferrous-wheel run scripts/profile_manta_gpu_efficiency.fw
```

The profile harness copies `.mll` package assets per variant, runs `train-embed`, writes each variant's `run.log`, `time.txt`, `cpu.pprof`, and `pprof-top.txt`, then writes a root `summary.tsv` with throughput and accelerator counters.
Set `EOS_GPU_PROFILE_NO_TOKENIZER=1` when profiling already-tokenized JSONL so the copied sibling tokenizer does not force text mode.

## Current Default Model Smoke

The current reference smoke uses:

- model package: `eos-embed-v1`
- encoder repeats: `2`
- tokenizer vocab: `2454`
- max sequence length: `256`
- train set: `4096` contrastive examples
- eval set: `512` contrastive examples
- batch size: `1024`
- loss: InfoNCE
- temperature: `0.05`
- learning rate: `0.005`
- eval cadence: every `4` steps

Latest local CUDA result:

```text
throughput: elapsed=5.139s examples/s=896.74 pairs/s=867250.60 train_examples/s=845.15 train_pairs/s=865437.87 eval_examples/s=1752.73 eval_pairs/s=897395.54 optimizer_steps/s=0.83
accelerators: forward=cuda optimizer=cuda activation=host contrastive=cuda
profile delta: matmul_bind_calls=30 matmul_runs=13644 matmul_run_upload_mb=3727.56 matmul_run_download_mb=2000.00 optimizer_updates=28 activation_calls=0 contrastive_calls=4
```

This is the promoted default benchmark path. It includes CUDA matmul scratch-buffer reuse, grouped batched backward, exact-length grouped contrastive forward for variable-length text, pair-length-aware bucketing across shuffled windows, strided-batched cuBLAS for grouped attention matmuls, rank-3 transpose support for grouped attention backward, batch-1024 contrastive training, sequence matmul bindings disabled by default, Q/K/V forward projection coalescing through a multi-bound-right CUDA path, Q/K/V attention-gradient accumulation through one concatenated shared-left GEMM, combined V/K attention backward gradients in one doubled-batch GEMM, Q/K/V input-gradient accumulation into one resident-right output download with one CUDA sync per accumulated group, and guarded grouped activation-backward helpers kept behind the activation accelerator flag. The larger batch keeps the full in-batch negative set intact, improves contrastive signal, cuts optimizer/contrastive calls on this smoke, and reduces per-pair orchestration overhead. Length bucketing is on by default for CLI contrastive training; set `--length-bucket-batches=false` to disable it or `EOS_TRAIN_LENGTH_BUCKET_WINDOW` to tune the shuffled sort window. Disabling per-sequence matmul bindings trades a small upload increase for a large reduction in backend binding churn. Q/K/V coalescing preserves per-weight residency and quantization while uploading each shared left-hand activation once across query/key/value projections. Concatenated shared-left attention-gradient coalescing computes `input^T*[dQ|dK|dV]` as one standard GEMM, then splits the result back into the Q/K/V weight gradients. V/K gradient coalescing computes `scores^T*dMixed` and `dPreSoftmax^T*Q` in a single strided-batched dispatch because both use the same transpose shape. Input-gradient coalescing computes `dQ*Wq^T + dK*Wk^T + dV*Wv^T` against resident Q/K/V weights, synchronizes once after the accumulated cuBLAS calls, and downloads the accumulated result once.

Pairwise eval-only gates also use exact-length batched forward chunks by default. On the acquired hard eval set, the current default `EOS_TRAIN_PAIR_EVAL_BATCH_SIZE=256` measured `9.85s`, `6704` matmul runs, and `4261.18 MB` uploaded versus `16.16s`, `53504` matmul runs, and `5135.84 MB` uploaded with `EOS_TRAIN_DISABLE_BATCHED_PAIR_EVAL=1`. Eval metrics matched within float tolerance.

Read the throughput line with both lenses:

- `train_examples/s` measures actual encoder training throughput.
- `train_pairs/s` measures in-batch contrastive work and grows roughly with `batch^2`.
- `optimizer_steps/s` protects the benchmark from hiding that very large batches reduce update count on small datasets.

## Recent Perf Delta

The training hot path moved as follows on the same mini smoke:

| Path | Train pairs/s | Matmul runs |
| --- | ---: | ---: |
| Instrumented baseline | `21975.83` | `409600` |
| CUDA scratch reuse | `22933.16` | `409600` |
| Grouped batched backward default | `33328.94` | `237568` |
| Exact-length grouped forward default | `35856.58` | `140056` |
| Strided-batched grouped attention | `42594.29` | `107552` |
| Rank-3 transpose batched attention backward | `82262.50` | `50208` |
| Batch-512 benchmark smoke | `177192.16` | `28848` |
| Batch-1024 default benchmark smoke | `407407.98` | `16464` |
| Disable sequence matmul bindings by default | `707265.87` | `16464` |
| Return grouped bound-right outputs as views | `742477.76` | `16464` |
| Concatenate shared-left Q/K/V gradient matmul | `762372.72` | `15336` |
| Combine attention V/K gradient matmuls | `803746.93` | `14772` |
| Accumulate Q/K/V input gradients with resident weights | `825766.14` | `13644` |
| Single-sync accumulated Q/K/V input gradients | `865437.87` | `13644` |

The main wins came from grouping real text batches by sequence length during backward, coalescing parameter-gradient matmuls into taller `X^T*dY` operations, grouping contrastive forward sequences by exact token length inside each original batch, promoting rank-3 x rank-3 CUDA matmul to `cublasSgemmStridedBatched`, allowing strided-batched matmul to handle transpose flags directly, and increasing the effective contrastive batch. The forward grouping keeps the full in-batch negative set intact and avoids padding, so attention math does not change.

The Q/K/V multi-bound-right path is transfer progress: it reduces matmul run uploads from `4173.15 MB` to `3727.56 MB` while preserving each weight's resident quantized form. The concatenated shared-left gradient path and combined V/K gradient path are dispatch-count wins on top of that. Accumulated input gradients keep the Q/K/V weights resident and reduce matmul run downloads from `2208.41 MB` to `2000.00 MB`. Single-sync accumulation removes the redundant per-term CUDA syncs inside that primitive while keeping the same transfer profile. Batch-1024 A/B measured `845.15 train_examples/s`, `865437.87 train_pairs/s`, and `13644` matmul runs by default versus `748.52 train_examples/s`, `766485.96 train_pairs/s`, and `13644` matmul runs with `EOS_CUDA_DISABLE_ACCUMULATED_MATMUL_SINGLE_SYNC=1`.

Batched forward materialization now keeps downloaded backend outputs as per-sequence views and passes layer outputs forward by view instead of copying them through temporary buffers. On a real acquired-text 512 train / 128 eval A/B at batch 256, trainer elapsed moved from `9.557s` to `8.235s` and train examples/s moved from `57.91` to `70.67`; matmul upload/download counters stayed at `5838.20 MB` / `2990.88 MB`, as expected, because this cuts host copies rather than device transfer.

Attention backprop now also keeps batched Q/K/V gradient outputs as views instead of allocating zeroed per-sequence buffers and copying accelerator outputs into them. On the acquired-text 512 train / 128 eval CPU profile, `backpropAttentionSequences` moved from `4.69s` cumulative to `2.21s`, and `runtime.memclrNoHeapPointers` moved from `3.27s` to `1.26s`. Matmul counters stay unchanged; this is a host allocation and copy reduction inside the existing CUDA dispatch pattern.

The production candidate workflow keeps `EOS_EVAL_EVERY_STEPS=0` by default. Step-level eval is still available for convergence debugging, but it inserts full eval passes into `train.log`; on the acquired full split, `--eval-every-steps 4` adds `21` extra contrastive eval passes across 3 epochs at batch 1024. Keep release gates in the final validation and hard holdout evals so training transfer reflects optimizer work first.

Ranked BPE tokenization removed a startup/data-ingest bottleneck before longer training runs. A direct batch-2048 `train-embed` CPU profile moved tokenizer encode time from `2.13s` cumulative (`BPETokenizer.Encode` -> `bpeMerge` -> `applyMerge`) to `0.25s` cumulative (`bpeMergeRanked`). End-to-end throughput remains dominated by training transfer/orchestration after tokenization, but corpus ingestion no longer burns a large fraction of host CPU.

Ranked BPE now compacts each selected merge in place instead of allocating a fresh token slice for every merge pass. On the acquired-text 512 train / 128 eval CPU profile, `applyRankedMerge` moved from `0.86s` cumulative to `0.45s`; total tokenizer encode time moved from `3.39s` to `3.10s`. The full run remains dominated by backend orchestration, memory clearing, and host-device traffic, so this is a data-ingest allocation cut rather than a matmul-counter change.

Prepared-text ingestion now keeps a trainer-local tokenization cache across train and eval loading. On the same acquired-text 512 train / 128 eval profile, with `1.38x` repeated text fields, `BPETokenizer.Encode` moved from `3.57s` cumulative to `1.98s`, train text tokenization moved from `2.82s` to `1.60s`, pair eval text tokenization moved from `0.75s` to `0.38s`, and trainer elapsed moved from `8.414s` to `7.577s`. Matmul counters stayed unchanged at `2788` runs, `5838.20 MB` uploaded, and `2990.88 MB` downloaded; this is an in-memory ingest optimization, and candidate packages, tokenizers, train profiles, checkpoints, and sealed outputs remain `.mll` artifacts.

Prepared JSONL production runs can now pretokenize once with `eos tokenize-embed` and train with `eos train-embed --no-tokenizer`. On the 512 train / 128 eval mini smoke, a same-code text JSONL profile still spent `3.61s` cumulative in `BPETokenizer.Encode` (`22.59%` of CPU samples), while the token JSONL profile removed tokenizer encode from the hot path. The GPU counters were identical for the two runs at `2712` matmul runs, `5838.20 MB` uploaded, and `2988.88 MB` downloaded; text measured `24948.46` train pairs/s and token JSONL measured `22414.32` train pairs/s in one noisy pair of local runs. Treat this as a training-profile cleanup and reproducibility win, not a claimed device-throughput gain.

Unbound activation acceleration now has a default shape ceiling so `EOS_TRAIN_ENABLE_ACTIVATION_ACCEL=1` does not route long-document activation groups through standalone upload/download kernels. On the tokenized acquired 4096 train / 512 eval split, fully unbounded CUDA activation regressed to `1m42.293s` and `42420.11` train pairs/s with `744` activation calls. Host activation measured `1m0.212s` and `73554.96` train pairs/s. Shape-limited opt-in measured `1m0.321s` and `73329.90` train pairs/s with `694` activation calls and identical train/eval metrics. On the smaller 512 / 128 tokenized smoke, the same shape-limited opt-in path measured `25239.78` train pairs/s versus `23997.21` for the host-activation default in the A/B run. Keep activation acceleration as an experiment until activation residency removes the extra transfers.

Fast GELU is available as an opt-in training math approximation for candidate-throughput experiments. `EOS_TRAIN_ENABLE_FAST_GELU=1` replaces the precise tanh call in host GELU forward/backward with a bounded rational tanh approximation and keeps GELU backward on the host so forward/backward use matching math. On the tokenized 512 train / 128 eval smoke, precise GELU measured `24191.69` train pairs/s and fast GELU measured `32525.01` train pairs/s; eval AUC moved from `0.580322` to `0.581543`. On the tokenized 4096 train / 512 eval split, fast GELU measured `86779.56` train pairs/s versus the prior clean precise-GELU baseline of `73554.96`; eval AUC moved from `0.548126` to `0.547104`. Treat this as a speed/quality knob for candidate runs and keep the exact GELU default until larger validation clears the tradeoff.

Batched matmul upload staging now detects already-contiguous split views and uploads the backing span directly instead of copying those views into scratch first. On the same acquired-text 512 train / 128 eval profile, trainer elapsed moved from `7.577s` to `6.396s`, `runtime.memmove` moved from `1.07s` to `0.58s`, `runtime.memclrNoHeapPointers` moved from `0.78s` to `0.60s`, and `flattenFixedFloat32MatricesScratch` was down to `0.34s` cumulative. Matmul counters again stayed unchanged at `2788` runs, `5838.20 MB` uploaded, and `2990.88 MB` downloaded; the win is less host staging around the same CUDA work.

Trainer-owned scratch buffers now reuse transient float32 flattening inputs for batched matmul dispatches. On the same direct batch-2048 CPU profile, `flattenFixedFloat32Matrices` moved from roughly `0.72s` cumulative to `0.09s` cumulative. The remaining host-side allocation/copy profile is mostly broader activation/state materialization and unavoidable host-device transfer until full device residency lands.

## Batch Sweep

Batch size is now the largest exposed training knob. On the same 4096-example mini smoke:

| Batch | Run steps | Train examples/s | Train pairs/s | Matmul runs | Max RSS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `512` | `8` | `558.19` | `285791.12` | `28848` | `1.02 GB` |
| `1024` | `4` | `845.15` | `865437.87` | `13644` | `1.52 GB` |
| `2048` | `2` | `827.20` | `1694107.56` | `9408` | `2.50 GB` |
| `4096` | `1` | `741.49` | `3037152.08` | `5616` | `4.47 GB` |

Batch 2048 is a useful ceiling or large-corpus setting, but it halves optimizer steps on this mini smoke. Batch 4096 is one update and regresses example throughput despite very high pair throughput. Batch 1024 is the benchmark default because it keeps multiple updates in the smoke and captures most of the real throughput win.

## How Much Faster Can It Get?

The current profile still shows the next bottleneck is backend transfer/orchestration, not raw math:

```text
MatMulRuns ~= 13644 per 4096-example mini smoke at batch 1024
RunUploadedBytes ~= 3.73 GiB
RunDownloadedBytes ~= 2.00 GiB
```

Reasonable next targets:

- `850-1000 train_examples/s`: reduce host materialization and launch overhead inside same-length groups.
- `1000+ train_examples/s`: keep forward/backward intermediates device-resident across full layer groups and move optimizer/activation state through the same device-resident path.
- `1200+ train_examples/s`: persistent device-resident training steps with fused attention/FFN backward and fewer per-layer dispatches.

The practical ceiling for this tiny model is dominated by orchestration and host-device transfer, not raw GEMM throughput. Larger models will shift more time into actual math, but the same device-residency work is still required to get high GPU utilization.

## Relevant Flags

```bash
EOS_TRAIN_DISABLE_BATCHED_BACKWARD=1
```

Disables the promoted grouped batched-backward path and returns to per-sequence backward.

```bash
EOS_TRAIN_DISABLE_BATCHED_FORWARD=1
```

Disables the promoted batched forward path and returns to per-sequence forward encoding. Batched forward is enabled by default because the larger default-model run underfeeds the GPU unless forward work is coalesced aggressively.

```bash
EOS_TRAIN_DISABLE_BATCHED_PAIR_EVAL=1
```

Disables exact-length batched forward chunks for pairwise `train-embed --eval-only` runs. This is useful for A/B checks against the scalar pair encoder.

```bash
EOS_TRAIN_PAIR_EVAL_BATCH_SIZE=256
```

Controls how many pair examples each pairwise eval chunk may contain before grouping by exact token length. The default is `256`; larger chunks reduce dispatch count further but can increase materialization pressure and wall time.

```bash
EOS_TRAIN_ENABLE_SEQUENCE_MATMUL_BINDINGS=1
```

Re-enables per-sequence matmul bindings. These are disabled by default because batch-1024 grouped training spends more time binding and unbinding short-lived sequence tensors than it saves in uploads. Keep this for small-batch experiments and regression checks.

```bash
EOS_TRAIN_DISABLE_QKV_MULTI_BOUND=1
```

Disables Q/K/V forward projection coalescing. By default CUDA uses one uploaded left-hand activation with three resident right-hand Q/K/V matrices for same-length forward groups. This cuts transfer bytes while preserving each right-hand weight's own quantization state.

```bash
EOS_TRAIN_DISABLE_SHARED_LEFT_MATMUL=1
```

Disables shared-left matmul coalescing. By default Eos first tries the concatenated shared-left gradient path for attention backward Q/K/V weight-gradient matmuls, then falls back to the backend shared-left interface when concatenation is disabled or unavailable.

```bash
EOS_TRAIN_DISABLE_CONCAT_SHARED_LEFT_MATMUL=1
```

Disables only the concatenated shared-left path. This keeps the older backend shared-left fallback enabled for A/B testing.

```bash
EOS_TRAIN_DISABLE_COMBINED_ATTENTION_VK_GRAD=1
```

Disables only the combined V/K attention backward gradient path. This returns to separate strided-batched GEMMs for `scores^T*dMixed` and `dPreSoftmax^T*Q`.

```bash
EOS_TRAIN_DISABLE_ACCUMULATED_ATTENTION_INPUT_GRAD=1
```

Disables only the accumulated Q/K/V attention input-gradient path. This returns to three resident right-hand matmuls for `dQ*Wq^T`, `dK*Wk^T`, and `dV*Wv^T`, followed by host accumulation.

```bash
EOS_CUDA_DISABLE_ACCUMULATED_MATMUL_SINGLE_SYNC=1
```

Disables CUDA single-sync accumulation inside the backend accumulated resident-right matmul primitive. This keeps the same logical training path but synchronizes after each accumulated cuBLAS term, which is useful for A/B testing backend launch overhead.

```bash
EOS_TRAIN_ENABLE_FAST_GELU=1
```

Uses a bounded rational tanh approximation for host GELU forward/backward. This is opt-in because it changes training math. It can materially improve CPU-bound trainer throughput while the model still uses host activation math, but production candidates should compare validation and hard-holdout metrics against exact GELU before promotion.

```bash
EOS_TRAIN_ENABLE_ACTIVATION_ACCEL=1
```

Enables CUDA/Metal activation backward acceleration. Activation backward batches GELU, softmax, and layernorm across same-length groups before dispatching. It is still opt-in because standalone activation kernels are upload/download-bound without broader activation residency. Large unbound groups fall back to host math by default through `EOS_TRAIN_ACTIVATION_ACCEL_MAX_ELEMENTS`.

```bash
EOS_TRAIN_ACTIVATION_ACCEL_MAX_ELEMENTS=1048576
```

Caps unbound activation accelerator calls by `rows * cols`. The default is `1048576`. Set it lower to force more host fallback for memory-transfer-heavy profiles; set it to `0` to remove the shape limit when profiling a fully unbounded activation path.

```bash
EOS_TRAIN_ENABLE_SOFTMAX_BACKWARD_ACCEL=1
```

Enables only the attention softmax backward activation path while keeping GELU and layernorm backward on the host. This is a profiling seam, not a promoted default.

```bash
EOS_TRAIN_DISABLE_SOFTMAX_BACKWARD_ACCEL=1
```

Disables the softmax backward activation path even when the broader activation accelerator is enabled. Use it to isolate GELU/layernorm activation experiments from attention softmax experiments.
