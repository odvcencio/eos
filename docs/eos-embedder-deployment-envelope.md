# EOS Embedder Deployment Envelope

`scripts/summarize_eos_embedder_deployment_envelope.py` builds a compact,
auditable report for comparing an EOS embedder candidate against the deployment
questions that matter before release:

- deployable artifact bytes measured from the selected artifact, plus separate
  supplied support/evidence file totals;
- dense, fp16, and TurboQuant IP vector payload projections for chosen corpus
  sizes;
- retrieval leaderboard rows and macro quality when a TSV is supplied;
- optional TurboQuant retrieval metrics from
  `manta.embedding_turboquant_retrieval_metrics.v1`;
- optional serving-energy and startup/load/encode evidence;
- legal/provenance gates for commercial or free/open release.

The tool does not run training, embedding, indexing, or model benchmarks. It
summarizes existing evidence and clearly labels each section as `measured`,
`projected`, `missing`, `unknown`, or `blocked`.

## Current Stage B Native EOS Candidate

Stage B in the repair pipeline is the native EOS 256d lineage, not the separate
384d imported BGE candidate. Use `--dimension 256` for that artifact:

```bash
python3 scripts/summarize_eos_embedder_deployment_envelope.py \
  --artifact runs/eos-embed-v1-repair-pipeline-20260831T235942Z/stage-b-balanced-beir-recovery-20260901T055440Z/eos-embed-v1.sealed.mll \
  --tokenizer runs/eos-embed-v1-repair-pipeline-20260831T235942Z/stage-b-balanced-beir-recovery-20260901T055440Z/eos-embed-v1.tokenizer.mll \
  --weights runs/eos-embed-v1-repair-pipeline-20260831T235942Z/stage-b-balanced-beir-recovery-20260901T055440Z/eos-embed-v1.weights.mll \
  --dimension 256 \
  --model-family native-eos-stage-b \
  --leaderboard-tsv runs/eos-embed-v1-repair-pipeline-20260831T235942Z/stage-b-balanced-beir-recovery-20260901T055440Z/retrieval/final-retrieval-eval/leaderboard.tsv \
  --legal-manifest-json runs/eos-embed-v1-repair-pipeline-20260831T235942Z/score-spectrum/stage-b.manifest.json \
  --corpus-size 1000000,10000000 \
  --bits 1,2,3,4,5,8 \
  --output-json runs/eos-embed-v1-repair-pipeline-20260831T235942Z/stage-b-balanced-beir-recovery-20260901T055440Z/deployment-envelope.json \
  --output-tsv runs/eos-embed-v1-repair-pipeline-20260831T235942Z/stage-b-balanced-beir-recovery-20260901T055440Z/deployment-envelope.tsv
```

When demonstrating training checkpoint evidence rather than the deployable
artifact envelope, pass `--checkpoint` with
`runs/eos-embed-v1-repair-pipeline-20260831T235942Z/stage-b-balanced-beir-recovery-20260901T055440Z/eos-embed-v1.embed-train.mll`.
Do not treat checkpoint or support-file totals as deployment footprint.

## Accounting Rules

Vector projections are payload-only. They exclude ids, metadata, ANN graphs,
database page layout, mmap overhead, network framing, and application-level
indexes.

Dense payload bytes are `corpus_vectors * dimension * 4`. fp16 payload bytes are
`corpus_vectors * dimension * 2`.

TurboQuant IP payload bytes are projected as:

```text
ceil(dimension * max(bits - 1, 0) / 8) + ceil(dimension / 8) + 8
```

The first term is the MSE buffer byte ceiling at `bits-1`. The second term is
the sign buffer byte ceiling. The final 8 bytes are Norm/ResNorm. This handles
`b1` as sign-only plus Norm/ResNorm and keeps non-byte-aligned dimensions honest.

Retrieval leaderboard throughput is vector scoring throughput, not
end-to-end service throughput. Use the serving energy benchmark and
startup/load/encode gate evidence to fill those sections.

Legal gates are separate from quality. `commercial_use_allowed=false` blocks a
commercial release, and `release_train_allowed=false` blocks open/free release
of the trained artifact even if retrieval quality looks promising.
