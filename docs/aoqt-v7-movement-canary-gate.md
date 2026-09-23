# AOQT V7 Movement Canary Gate

`scripts/aoqt_v7_movement_canary_gate.py` is a train/dev canary for AOQT V7
research candidates. It is deliberately separate from the official heldout
gate: it reads training metrics, dev evidence, the emitted AOQT sidecar
transform, and a non-official dev proxy report, then writes deterministic
schema-stamped JSON. It never runs training and never evaluates official
heldout data.

All inputs fail closed unless they carry the explicit canary contracts:

- train metrics schema `eos.q3_aoqt_sidecar_metrics.v2`;
- train metrics plan, summary, and diagnostics optimizer mode
  `aoqt-v7-dev-trust-region-v1`;
- train metrics diagnostics `dev_only: true`;
- train metrics diagnostics coordinate search strategy
  `q3_gain_protected_cone_trust_region_dev_v1`;
- train metrics/dev evidence split binding schema
  `eos.aoqt.v7_train_split_binding.v1`;
- train metrics accepted coordinate proposal receipts with actual
  strictly sorted unique in-bounds `moved_angle_indices` and exact
  `moved_angle_count`;
- dev evidence schema `eos.q3_aoqt_sidecar_dev_evidence.v1`;
- dev evidence hashes bound to the actual `--train-metrics`, `--sidecar`, and
  required `--preflight` files;
- sidecar schema `eos.aoqt.givens_transform_sidecar.v1`;
- dev proxy schema `eos.aoqt.v7_non_official_dev_proxy.v1`;
- dev proxy evidence label `non_official_dev_proxy`;
- split manifest schema `eos.aoqt.v7_dev_split_manifest.v1`;
- split validation schema `eos.aoqt.v7_dev_split_validation.v1` with
  `passed: true`;
- dev proxy split binding schema `eos.aoqt.v7_dev_proxy_split_binding.v1`;
- train sidecar binding schema `eos.aoqt.v7_train_sidecar_binding.v1`.

Policy `aoqt-v7-movement-canary-conservative-v1` preregisters one fixed
interpretation of the movement thresholds:

- reject identity, single-angle, tiny-norm, or non-finite transforms;
- require at least 16 nonzero angles and at least 1% nonzero angles;
- require `angle_l2 >= 2e-4` and at least 4 unique nonzero angle values;
- require at least 8 accepted proposals and at least 15% accepted proposals;
- require positive coordinate and trust-region proposal attempts, a positive
  trust-region radius, coordinate proposal accounting, trust-region proposal
  accounting, and matching coordinate/trust-region attempt counts;
- require an accepted coordinate proposal receipt with `moved_angle_count >= 2`,
  or an accepted single-coordinate trust-region movement explicitly marked with
  `trust_region_accepted_single_coordinate_rule:
  "aoqt-v7-dev-preregistered-accepted-single-coordinate-v1"`;
- require q3 top10 churn of at least 25 total and at least 1 per domain;
- require preferred dev q3 macro nDCG@10 lift of at least `0.0006`;
- require per-domain q3 nDCG@10 and recall@100 deltas to be nonnegative;
- permit dense nDCG@10 delta down to `-0.0005` and q5 nDCG@10 delta down to
  `-0.001`.

The hard minimum q3 macro lift of `0.0004` is reachable only by selecting
`--macro-lift-policy hard-min-explicit`, which changes the emitted policy id to
`aoqt-v7-movement-canary-conservative-v1-explicit-hard-min`.

The split validator and movement canary are a mandatory composition for V7:
build and validate the dev split manifest first, then run this canary with
`--split-validation` against train metrics, dev evidence, and a dev proxy report
that all bind to that exact validated fold. The canary requires the validation
receipt schema, `passed: true`, split manifest file SHA-256, split payload
SHA-256, source-plan SHA-256, official qid registry SHA-256, and the validation
receipt's own file and payload SHA-256. A canary result without the
split-validator evidence is not a V7 decision artifact.

The dev proxy report is expected to identify itself as
`"evidence_label": "non_official_dev_proxy"` and bind itself to the exact
`--split-manifest` fold used to build the dev report. The binding must include
the split manifest file SHA-256, split payload SHA-256, source-plan SHA-256,
official qid registry SHA-256, official registry source SHA-256, fold id,
split-validation file and payload SHA-256s, the supplied official-qids file
SHA-256, per-domain dev qid counts, per-domain dev qid-set SHA-256s, and the
folded `qids_by_dataset` SHA-256. The canary recomputes each independently
verifiable value from the split manifest and validation receipt and rejects an
unbound, stale, or different split. The dev proxy producer also compares the
supplied official-qids file SHA-256 and registry source hash to the split's
official registry exactly, then emits the official-qids file, payload, source,
and qids-by-dataset hashes in the report.

The train metrics and dev evidence must carry an identical `split_binding`
computed from the same fold's `train` side. The canary recomputes the train row
count, train row-id set SHA-256, per-domain train qid counts, per-domain train
qid-set SHA-256s, and train `qids_by_dataset` SHA-256 from the supplied split
manifest. The native runner also records the materialized calibration manifest,
rows JSONL, preflight, and materialized row-id SHA-256s in that binding.

The dev evidence artifact hashes are checked against live files, not trusted as
self-reported metadata. `dev_evidence.metrics_sha256` must equal the SHA-256 of
the supplied `--train-metrics` file; `dev_evidence.transform_sha256` must equal
the SHA-256 of the supplied raw `--sidecar` file; `dev_evidence.pairings_sha256`
and `dev_evidence.angles_sha256` must equal the recomputed raw sidecar stage
digests; and `dev_evidence.preflight_sha256` must equal the SHA-256 of the
required `--preflight` file. Stale, copied, count-only, or mismatched evidence
fails closed before the canary decision is accepted.

The gate recomputes q3 macro nDCG@10 lift as the equal-domain average of the
three supplied per-domain deltas. If a report also supplies
`macro.q3_ndcg_at_10_delta`, that supplied value must match the recomputed
average within `1e-12`; callers cannot inflate the macro independently from
domain evidence. The accepted domain shape is:

```json
{
  "schema": "eos.aoqt.v7_non_official_dev_proxy.v1",
  "evidence_label": "non_official_dev_proxy",
  "dev_only": true,
  "official_heldout_gate": false,
  "quality_claim": false,
  "official_claim": false,
  "release_claim": false,
  "commercial_claim": false,
  "split_manifest": {
    "schema": "eos.aoqt.v7_dev_proxy_split_binding.v1",
    "split_manifest_schema": "eos.aoqt.v7_dev_split_manifest.v1",
    "split_manifest_file_sha256": "64 lowercase hex",
    "split_manifest_payload_sha256": "64 lowercase hex",
    "split_validation_schema": "eos.aoqt.v7_dev_split_validation.v1",
    "split_validation_file_sha256": "64 lowercase hex",
    "split_validation_payload_sha256": "64 lowercase hex",
    "source_plan_sha256": "64 lowercase hex",
    "official_qid_registry_sha256": "64 lowercase hex",
    "official_registry_source_sha256": "64 lowercase hex",
    "official_qids_file_sha256": "64 lowercase hex",
    "fold_id": "fold-0",
    "dev_qid_count_by_dataset": {"fiqa": 1, "nfcorpus": 1, "scifact": 1},
    "dev_qid_set_sha256_by_dataset": {
      "fiqa": "64 lowercase hex",
      "nfcorpus": "64 lowercase hex",
      "scifact": "64 lowercase hex"
    },
    "dev_qids_by_dataset_sha256": "64 lowercase hex"
  },
  "domains": {
    "fiqa": {
      "delta": {
        "q3": {"ndcg_at_10": 0.0001, "recall_at_100": 0.0},
        "dense": {"ndcg_at_10": 0.0},
        "q5": {"ndcg_at_10": 0.0}
      },
      "q3_top10_churn": 9
    }
  },
  "macro": {"q3_ndcg_at_10_delta": 0.0007}
}
```

The script also accepts equivalent heldout-gate-style nested delta keys such as
`domains.<domain>.delta.q3.ndcg_at_10`. Missing, non-finite, or ambiguous
evidence fails closed. Dev proxy reports also fail closed if any official,
quality, release, or commercial claim flag is true.

The sidecar must be a native AOQT Givens v1 transform bound to the canonical V7
topology: `kind=aoqt_givens_v1`, `dim=384`, `seed=191`, 8 stages, 192 disjoint
D384 coordinate pairs per stage, and 1536 total angles. The pairings digest must
be the canonical D384 seed191 digest
`dcac29b5b7ab30291bb6791b1e88178e1c755d5d3f995a06efe29a4f430bc6f8`;
sequential or otherwise noncanonical disjoint pairings are rejected. Its audit
`pairings_sha256` and `angles_sha256` must hash the actual sidecar stages, and
`orthogonality_frobenius_per_dim` must be present, finite, and nonnegative.

The gate accepts either an explicit canary-side `sidecar_binding` in the train
metrics or current native EOS output. Native sidecars do not need to carry the
extra canary `schema` or `topology` envelope, but the native train metrics must
bind the same values through `topology.pairings_sha256` and
`summary.angles_sha256`. The normalizer derives the internal canary envelope
only from observed native fields and rejects forged or stale bindings. When an
explicit `sidecar_binding` is present, any native `topology.pairings_sha256` or
`summary.angles_sha256` fields that are also present must still match the
supplied sidecar; contradictory native hashes fail closed instead of being
ignored.

The train metrics optimizer provenance is also bound across the emitted plan,
summary, and diagnostics. The canary records the observed optimizer modes,
`dev_only` flag, coordinate strategy, coordinate proposal counters,
trust-region proposal counters, trust-region radius, accepted receipt movement,
and accepted trust-region shape under `movement.optimizer_provenance`.
Production micro-tail, generic mislabeled, missing-coordinate-path, and
inconsistent-counter artifacts are allowed to reach the deterministic canary
result, but fail with explicit provenance checks rather than passing through
generic accepted-proposal counters.

Accepted proposal receipts must be authentic coordinate receipts. Every accepted
receipt must have `accepted: true`, `kind: "coordinate"`, coordinate strategy
`q3_gain_protected_cone_trust_region_dev_v1`, a positive `moved_angle_count`,
and `moved_angle_indices` that are strictly sorted, unique, within
`[0, 1536)`, and exactly the same length as `moved_angle_count`. The aggregate
accepted proposal count must match the accepted coordinate proposal count, so a
report cannot pass with only a forged count or a receipt that omits the moved
indices.

Example:

```bash
python3 scripts/aoqt_v7_movement_canary_gate.py \
  --train-metrics path/to/train.metrics.json \
  --sidecar path/to/aoqt-sidecar.json \
  --dev-evidence path/to/aoqt-sidecar-dev-evidence.json \
  --dev-proxy-report path/to/dev-proxy.fold-0.json \
  --split-manifest path/to/aoqt-v7-dev-split.json \
  --split-validation path/to/aoqt-v7-dev-split.validation.json \
  --preflight path/to/train-preflight.json \
  --output-json path/to/canary.fold-0.json
```
