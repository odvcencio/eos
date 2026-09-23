# AOQT V7 Dev Split Harness

`scripts/build_aoqt_v7_dev_split.py` creates deterministic query-level
train/dev manifests from an existing AOQT Stage 2 calibration plan.  It is a
splitter and validator only: it does not train, score, evaluate, read official
metrics, or consume vector/score payloads.

For V7, this validator and `scripts/aoqt_v7_movement_canary_gate.py` are a
mandatory composition. Build and validate the split manifest first, then run
the movement canary only against a dev proxy report that binds to the exact
validated fold. Neither artifact alone is sufficient as a V7 canary decision.

## Inputs

- AOQT Stage 2 plan-only calibration manifest, defaulting to
  `runs/eos-d384-aoqt-trainonly-v1-20260903T092452Z/aoqt/dataset-scoped/plan.pairable.raw.v4.json`.
- Pinned qid-only official exclusion manifest, defaulting to
  `runs/eos-d384-aoqt-trainonly-v1-20260903T092452Z/exclusions-scoped-v2/official-test.json`.
- Closed official heldout registry constants from `scripts/aoqt_heldout_gate.py`.

The official registry check is intentionally qid-only.  The harness verifies
the official manifest schema/name, per-domain qid counts, qid-set SHA-256s,
source qrels SHA-256s, rollup SHA-256, and the binding already recorded inside
the source Stage 2 plan.  It never reads official metric files.

## Split Unit

The split unit is `(dataset, qid)`.  All AOQT rows that share a query stay in
the same fold, including q3/q5 top-10 rows and the NFCorpus q3 boundary row.
This avoids row sibling leakage from repeated query evidence.

Each group records:

- `group_id`, `dataset`, `qid`, and qid hash.
- source provenance hash from the Stage 2 plan.
- sibling `row_ids`, `row_ids_sha256`, row count, and pair count.
- observed bits, buckets, guard classes, and stratum counts.

## Fold Policy

The default is 3-fold cross-validation over all non-official calibration query
groups.  The assignment is deterministic greedy balancing over row-count
strata:

- dataset/domain;
- dataset source hash;
- quantization bit;
- guard class;
- boundary bucket.

An optional reserve can be requested with `--reserve-fraction`, but the default
is no reserve so all non-official calibration groups participate in CV.

## Outputs

`build` writes an `eos.aoqt.v7_dev_split_manifest.v1` JSON manifest containing
the population, groups, folds, optional reserve, source-plan binding, official
qid registry binding, and leakage proof.

`validate` rebuilds the manifest from the source plan and official qid registry
and fails if any manifest content changed.  Its compact receipt uses schema
`eos.aoqt.v7_dev_split_validation.v1`.

Example:

```bash
python3 scripts/build_aoqt_v7_dev_split.py build \
  --output .tiller/scratch/codex/aoqt-v7-dev-split/aoqt-v7-dev-split.raw-v4.json

python3 scripts/build_aoqt_v7_dev_split.py validate \
  --manifest .tiller/scratch/codex/aoqt-v7-dev-split/aoqt-v7-dev-split.raw-v4.json \
  --output .tiller/scratch/codex/aoqt-v7-dev-split/aoqt-v7-dev-split.raw-v4.validation.json
```

## Real Raw-v4 Build

The initial real-data build used the default plan and official qid registry.
It produced 8,003 query groups, 16,414 AOQT rows, and 370,201 pair references.
Dataset group counts were FIQA 5,422, NFCorpus 1,781, and SciFact 800.

The leakage proof passed with:

- 0 official heldout qid overlaps;
- 0 duplicate dev group memberships;
- 0 train/dev group overlaps in every fold;
- 0 row sibling leaks;
- 0 unknown groups.

This is dev-split evidence only, not heldout quality evidence.

## Dev-Only Runner Binding

`eos train-aoqt-sidecar` keeps the empty optimizer mode as the production-safe
default.  AOQT V7 trust-region telemetry is available only when the run supplies
all of:

- `--optimizer-mode aoqt-v7-dev-trust-region-v1`
- `--allow-research-only-aoqt`
- `--allow-dev-aoqt-v7`
- `--dev-output-dir <dir>`

The V7 dev path writes train metrics plus raw transform/evidence JSON under the
dev output directory. It rejects `--output`, never writes a candidate package,
and skips official candidate eligibility artifacts. When a split manifest is
provided, bind it with `--split-manifest`, `--expected-split-manifest-sha256`,
and `--fold-id`; partial split/fold bindings fail closed.

### V7-r2 forward-consistency prerequisite

V7-r2 is a stricter, still-dev-only mode. A preregistered V7-r2 run must set

- `--optimizer-mode aoqt-v7-r2-dev-trust-region-v1`;
- `--require-forward-consistency-probe`;
- `--allow-research-only-aoqt`, `--allow-dev-aoqt-v7`, and `--dev-output-dir`;
- `--lr 0.01` exactly; and
- the complete split binding (`--split-manifest`,
  `--expected-split-manifest-sha256`, and `--fold-id`).

The runner selects exactly eight train rows by a deterministic SHA-256 ordering
bound to the workplan seed, fold id, split-manifest hash, materialized manifest
hash, and materialized row-id hash. It probes the smallest V7 magnitude
`0.0003125` for exactly D384's `1536` coordinates, the deterministic top eight
coordinates, and configured blocks `[2, 4, 8]`. The preregistered sign policy is
`exact-sign-zero-neutral-v1` with an explicit near-zero epsilon of `0`: finite
analytic and actual values are mapped only by `<0`, `==0`, and `>0`, with no
silent tolerance. Each receipt records the analytic STE q3/protected directional
derivatives, actual prepared-IP component deltas, signs, indices, directions,
and pass/fail result. Any sign disagreement fails closed before optimization.
The runner re-reads and hashes the manifest, rows, preflight, and split inputs
before writing a receipt; selected row IDs must be members of the bound materialized
row-id set.

The artifact schema is
`eos.aoqt.v7_dev_failclosed_diagnostics.v1`, written beside metrics as
`<metrics>.dev.failclosed.json`. It includes optimizer diagnostics and every
proposal receipt, exact IO/preflight/split bindings, and all quality, release,
official, heldout, and commercial flags set to `false`. No candidate package is
written. The default empty optimizer mode and production package path do not
enable or require this probe. This implementation task does not run the probe
on the real fold materialization or execute any proxy/heldout evaluation.

### V7-r2 probe-only authorization

To produce a standalone probe receipt without optimizer, metrics, dev-evidence,
or candidate-package artifacts, add `--forward-consistency-probe-only` and
provide all of the following:

- `--allow-research-only-aoqt` and `--allow-dev-aoqt-v7`;
- `--optimizer-mode aoqt-v7-r2-dev-trust-region-v1` and
  `--require-forward-consistency-probe`;
- `--lr 0.01` exactly;
- the complete split binding (`--split-manifest`,
  `--expected-split-manifest-sha256`, and `--fold-id`);
- all normal manifest, rows, preflight, anchor, compatibility, source-hash,
  and vector-cache input pins; and
- `--forward-consistency-probe-receipt-json <receipt.json>`.

`--metrics-json`, `--dev-output-dir`, `--output`, and `--max-steps` are not
needed for probe-only mode. If a metrics path is supplied it remains an input
reservation only; no metrics or optimizer artifact is written. A passed or
failed probe writes only the exclusive, bound receipt. A load/preflight/objective
failure before a receipt exists writes no substitute failure artifact.

### V7-r3 actual-direction probe-only authorization

V7-r3 is an explicit dev-only actual-direction search. It uses
`--optimizer-mode aoqt-v7-r3-dev-actual-direction-v1`,
`--require-actual-direction-probe`, `--allow-research-only-aoqt`, and
`--allow-dev-aoqt-v7`. The learning rate must be exactly `0.01`, and the
complete split binding is required. The probe selects eight train rows by its
own deterministic seed, ranks the analytic q3/protected gradients only as a
direction prior, and evaluates actual projected endpoints for the top eight
coordinates and blocks `[2,4,8]`. It evaluates six magnitudes
`[0.01,0.005,0.0025,0.00125,0.000625,0.0003125]` under both global signs,
for exactly 132 endpoints. Actual q3/protected eligibility and the existing
loss/component budgets decide the at-most-two full transaction candidates;
the analytic protected-cone sign is not an authority in r3.

The receipt schema is `eos.aoqt.v7_r3_actual_direction_probe.v1`. It includes
the endpoint hash chain, subset/selection seed, objective-contract,
eligibility-policy, schedule, ranking, IO/preflight, and split bindings. All
quality, release, official, heldout, and commercial claims are false. A
probe-only run writes exactly one receipt and no metrics, transform, dev
evidence, package, or fail-closed sidecar. The exact command shape is:

```text
eos train-aoqt-sidecar \
  --allow-research-only-aoqt --allow-dev-aoqt-v7 \
  --optimizer-mode aoqt-v7-r3-dev-actual-direction-v1 \
  --actual-direction-probe-only --require-actual-direction-probe \
  --lr 0.01 \
  --manifest <manifest.json> --rows <rows.jsonl> --preflight <preflight.json> \
  --expected-manifest-sha256 <manifest-sha256> --expected-rows-sha256 <rows-sha256> \
  --expected-preflight-sha256 <preflight-sha256> \
  --split-manifest <split-manifest.json> \
  --expected-split-manifest-sha256 <split-sha256> --fold-id <fold-id> \
  --expected-anchor-artifact-sha256 <anchor-sha256> \
  --expected-anchor-package-manifest-sha256 <package-manifest-sha256> \
  --anchor-embedding-space-id <embedding-space-id> \
  --compatibility-digest <compatibility-sha256> \
  --source-artifact-sha256 <source-sha256> --vector-cache-sha256 <vector-sha256> \
  --actual-direction-probe-receipt-json <receipt.json>
```

`--metrics-json`, `--dev-output-dir`, `--output`, and `--max-steps` are not
needed. This implementation does not run the r3 probe against real data or
invoke any quality, proxy, heldout, official, or package gate.

### V7-r4 actual-coordinate probe-only authorization

V7-r4 is the scalar, projected-endpoint successor to r3. It selects exactly
twelve train rows, ranks the analytic q3/protected gradients only to choose the
top 32 coordinates, and evaluates six magnitudes under both global signs for
exactly 384 prepared-IP endpoints. It is probe-only and therefore performs no
optimizer step, training, package publication, proxy retrieval, or heldout
evaluation.

The receipt schema is
`eos.aoqt.v7_r4_actual_coordinate_probe.v1`. Its coverage proof has six
canonical components: `q3_gain`, `q3_order_guard`, `q3_score_distill`,
`q5_order_guard`, `q5_score_distill`, and `nf_boundary_guard`. The
`score_distill_activation_semantics` field is
`structural_candidate_count_v1`: a row counts as score-distill coverage when
its score-distill weight is positive and it has at least two prepared
candidates. Coverage does not require a non-zero numeric score-distillation
loss; a zero loss can be a valid, fully covered baseline. `baseline_activation`
still records the independently replayed objective activation counts.

`ValidateBound` re-reads the manifest, rows, preflight, split binding, and
objective configuration, then independently replays one baseline plus all 384
prepared-IP endpoints. It compares projected direction vectors, candidate
losses and component sums, activation counts, candidate errors, eligibility
booleans/reasons, and endpoint hash chain/tail. The receipt records this cost
as `validation_replay_endpoint_count=384`,
`validation_replay_objective_evaluation_count=385`, and
`validation_replay_cost=baseline_plus_384_prepared_ip_endpoint_evaluations`.

## Fold-Train Materializer

`scripts/materialize_aoqt_v7_fold_train.py` filters an existing Stage 2
materialized AOQT calibration set to exactly one validated V7 fold's train
`row_ids`. It writes a runner-compatible `rows.jsonl`, `manifest.json`, and
`preflight.json`, plus a separate `row-ids.json` receipt with the split/source
bindings. The tool fails closed on stale expected hashes, failed or stale split
validation receipts, official-overlap leakage proof failures, missing or
duplicate row IDs, source row-id hash drift, existing output files, and
manifest/preflight count mismatches. It does not train, evaluate, export
vectors, invoke the official gate, or mutate the source materialization.

The real fold-0 materialization from the raw-v4 split is:

```text
.tiller/scratch/codex/aoqt-v7-fold-train/fold-0-r4/
```

It is bound to split manifest SHA-256
`77a0ccf9fa9dd77dff88c8494d98a091a2e3b216cbea6d974e22f585aca484ae`,
validation receipt SHA-256
`b5adfc621548d84b151f298b9f4bb056eb1f4fe496610bb28b7fb71f578e248a`,
source plan SHA-256
`6607a5907e9a4c0ffab63d53756b7e4555893a95a1b19193d83b554fc0a20b0a`,
source materialized manifest SHA-256
`558f4751f19891af821134a96e06f49e93d71ab0e51db234094a4253a2c0908e`,
source materialized rows SHA-256
`495dbdc3f67c49493a732c0cbe99f4554b305dc43607464573141ee7b6d0d82e`,
and source materialized preflight SHA-256
`c6796e9f4dbba85a2ce4c0c1503f40d3dbae6f7843ebbd8d71cd50d6fb74fb24`.

Final fold-0-r4 outputs:

- `row_count=10942`
- `candidate_count=134003`
- `pair_count=499228`
- Go-canonical manifest SHA-256
  `ea4132590ccd275521b5fb0a57753021e4ac91422cbed86d0dfd59402ff16c16`
- rows JSONL SHA-256
  `27841ae195ba84cc1e8113583153a3b28b66ad3d3fe562ce48b034bfe3cf3020`
- preflight JSON SHA-256
  `043ca68ccb6a7e1f0a2db1335c5a6ef2bd4f2fdc200b391622f662076d7551e6`
- row-id receipt JSON SHA-256
  `f3191490831d5160831212ea12ee90b8199c308a40aee93e06d6a7bfe13543a5`
- fold train JSON row-id set SHA-256
  `f72d13604ec764b16f6337632ccd3e132b2a4124902340e4a02089e7a66fcd6c`
- materialized AOQT row-id SHA-256
  `9ff23aa83c621bd7be583ccdedfb5b334f0c53674b0888b8ecb10a3a0fb2ca4b`

A strict out-of-tree Go loader smoke passed on these r4 artifacts with the same
row, candidate, pair, manifest, rows, and preflight hashes.

## Non-Official Full-Retrieval Dev Proxy

`eos transform-aoqt-vectors` applies a validated AOQT Givens sidecar to flat
document/query vector-cache JSONL files. The command preserves IDs, child/parent
IDs, roles, and all non-vector JSON fields while replacing the first present
`vector`, `embedding`, or `values` vector field. Real V7 use keeps the default
canonical D384 topology checks: `dim=384`, `stages=8`, `pairs_per_stage=192`,
and `angle_count=1536`. The emitted
`eos.aoqt.vector_cache_transform_binding.v1` sidecar binds the input caches,
output caches, transform file, pairings hash, angles hash, and orthogonality
audit.

`scripts/build_aoqt_v7_non_official_dev_proxy.py` builds the canary-facing
`eos.aoqt.v7_non_official_dev_proxy.v1` report. It is not an official heldout
gate and writes `official_heldout_gate=false` plus `quality_claim=false`. The
producer validates the split manifest/fold binding, optional split validation
receipt, D384 transform sidecar audit, official-qid exclusion manifest, and
optional train metrics/dev evidence/preflight hashes. It materializes dev-only
queries and qrels from non-official source dataset directories, uses full corpus
vector caches, runs or plans native dense/q3/q5 anchor and candidate cache
evals, recomputes candidate-anchor deltas, and computes direct q3 top-10
sequence churn from per-query JSONL.

Known FIQA empty-relevance rows are handled explicitly: dev qids with no
positive qrels are audited as empty-relevance qids, while missing vectors for
any non-empty relevant document or any dev query fail closed. Official qids and
official qrels/metrics/caches are rejected for this proxy.

Real fold-0 command shape, to run only when a full dev proxy is authorized:

```bash
python3 scripts/build_aoqt_v7_non_official_dev_proxy.py \
  --split-manifest .tiller/scratch/codex/aoqt-v7-dev-split/aoqt-v7-dev-split.raw-v4.json \
  --split-validation .tiller/scratch/codex/aoqt-v7-dev-split/aoqt-v7-dev-split.raw-v4.validation.json \
  --fold-id fold-0 \
  --official-qids runs/eos-d384-aoqt-trainonly-v1-20260903T092452Z/exclusions-scoped-v2/official-test.json \
  --sidecar <candidate-dev-output>/aoqt-sidecar-transform.json \
  --train-metrics <candidate-dev-output>/metrics.json \
  --dev-evidence <candidate-dev-output>/aoqt-sidecar-dev-evidence.json \
  --preflight <candidate-dev-output>/preflight.json \
  --eos-bin ./eos \
  --dataset-dir fiqa=<non-official-fiqa-source-dir> \
  --dataset-dir nfcorpus=<non-official-nfcorpus-source-dir> \
  --dataset-dir scifact=<non-official-scifact-source-dir> \
  --anchor-doc-vectors fiqa=<full-fiqa-doc-vectors.jsonl> \
  --anchor-doc-vectors nfcorpus=<full-nfcorpus-doc-vectors.jsonl> \
  --anchor-doc-vectors scifact=<full-scifact-doc-vectors.jsonl> \
  --anchor-query-vectors fiqa=<full-fiqa-query-vectors.jsonl> \
  --anchor-query-vectors nfcorpus=<full-nfcorpus-query-vectors.jsonl> \
  --anchor-query-vectors scifact=<full-scifact-query-vectors.jsonl> \
  --output .tiller/scratch/codex/aoqt-v7-dev-proxy/fold-0/report.json \
  --work-dir .tiller/scratch/codex/aoqt-v7-dev-proxy/fold-0/work \
  --execute
```

## V7-r6 progressive train-only screen

V7-r6 is a required, research-only, probe-only screen. It has no optimizer
step and does not write metrics, transform, dev-evidence, package, retrieval,
proxy, heldout, or official artifacts. It binds the canonical V7-r4 endpoint
receipt and the V7-r5 receipt to the same materialized fold/preflight inputs,
then deterministically partitions the train rows into seven disjoint semantic
strata: `fiqa.q3.top10.*`, `fiqa.q5.top10.*`, `nfcorpus.q3.top10.*`,
`nfcorpus.q5.top10.*`, `scifact.q3.top10.*`, `scifact.q5.top10.*`, and
`nfcorpus.q3.nf80_120.*`. Dataset, row-id prefix, guard class, and candidate
source must agree; all other rows fail closed. The fixed seed/hash order binds
the exact populations `[3575,3600,959,950,533,532,793]` and quotas
`[146,148,39,39,22,22,96]` for `S512` and `[293,295,79,78,44,43,192]` for
`S1024`; `S512` is a strict subset of `S1024`. Stratum estimates use the exact
without-replacement Horvitz--Thompson factor `population/sample_count`.

Both stages start from an identity transform and evaluate the same exact 31
eligible r4 ordinals on the actual prepared-IP objective. Ranking first places
transaction-feasible endpoints ahead of infeasible diagnostics. The fallback
tail is ordered by hard-gate violation count, aggregate excess over strict or
declared allowance thresholds, actual loss delta, and ordinal; no scalar budget
tradeoff is applied.
The fixed `K=6` selection is the S1024 ranking prefix. The full-objective
top-K replay adds one fresh identity baseline and six selected-endpoint
evaluations, for 71 total stage/full evaluations. Dense preservation remains an
explicit deferred gate for later full-training verification. Every stage and
endpoint must have finite loss, all six objective components, and positive activation;
row/hash/official overlap fails closed. A successful receipt uses schema
`eos.aoqt.v7_r6_progressive_screen.v1` and leaves every quality, release,
official, heldout, and commercial claim false.

The exact command shape is:

```text
eos screen-aoqt-v7-r6 \
  --allow-research-only-aoqt --allow-dev-aoqt-v7 \
  --optimizer-mode aoqt-v7-r6-progressive-screen-v1 \
  --progressive-screen-only --probe-only \
  --manifest <manifest.json> --rows <rows.jsonl> --preflight <preflight.json> \
  --expected-manifest-sha256 <manifest-sha256> --expected-rows-sha256 <rows-sha256> \
  --expected-preflight-sha256 <preflight-sha256> \
  --split-manifest <split-manifest.json> \
  --expected-split-manifest-sha256 <split-sha256> --fold-id <fold-id> \
  --expected-anchor-artifact-sha256 <anchor-sha256> \
  --expected-anchor-package-manifest-sha256 <package-manifest-sha256> \
  --anchor-embedding-space-id <embedding-space-id> \
  --compatibility-digest <compatibility-sha256> \
  --source-artifact-sha256 <source-sha256> --vector-cache-sha256 <vector-sha256> \
  --canonical-r4-receipt-json <r4-receipt.json> \
  --canonical-r4-receipt-sha256 ae0950589be98ba35cadb8ec9097963019a7eeeaf3951db400b4c08986203844 \
  --canonical-r5-receipt-json <r5-receipt.json> \
  --canonical-r5-receipt-sha256 30e1fe48788ba8c23172ad01e60c336e0e2df10fc5efd0e61c4d4e7255ba36d7 \
  --progressive-screen-receipt-json <r6-screen-receipt.json>
```

The file-backed CLI's preregistered accounting is always 71 prepared-IP objective evaluations:
two identity baselines plus 31 actual projected endpoints for each of S512 and
S1024, followed by one identity baseline and six selected-endpoint full-
objective replays. The in-memory runtime API retains a receipt-only test mode
at 64 evaluations, but the file-backed CLI rejects `--stop-after-receipts` so a
persisted screen cannot omit full verification. This implementation task did
not run the command on real fold data; no training, metrics/package generation,
or official evaluation was performed.
