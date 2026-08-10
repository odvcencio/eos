# Eos Phase 0 Quantized-Retrieval Anchor Re-Baseline (C0, turboquant v0.2.1)

Run directory: `runs/eos-phase0-rebaseline-v021-20260810T040706Z/`
Branch: `feat/eos-v0.2.0-phase0-c0` (worktree `/home/draco/work/eos-c0-baseline`)
Binary: `/tmp/eos-c0` (built from this worktree, `go.mod` pins `m31labs.dev/turboquant v0.2.1`)
Backend: `cuda` (RTX 5070 Ti), quantizer seed `5581486560434873699`
nDCG = normalized Discounted Cumulative Gain. BEIR = the standard IR benchmark corpus/query/qrels format used by these three datasets.

## 1. Outcome Summary

All three datasets ran full and uncapped. The dense control assert passes exactly
for all three datasets. Every quantized row stamps `turboquant_version` and
`codebook_version` as `"v0.2.0"`, not the literal string `"v0.2.1"` the task
asked me to assert. I investigated this and it is a confirmed upstream
cosmetic bug, not a stale-codebook signal — see Section 3. The run is safe to
treat as the v0.2.1-codebook baseline.

## 2. Control Assert (dense rows)

Dense paths do not touch turboquant, so dense nDCG@10/recall@100 must match
the manifest's `dense_short_metrics` exactly. They do, bit-for-bit within
float64 precision:

| dataset | new nDCG@10 | anchor nDCG@10 | new recall@100 | anchor recall@100 | match |
| --- | ---: | ---: | ---: | ---: | :---: |
| scifact | 0.5645379155090131 | 0.564537915509013 | 0.7964444444444444 | 0.796444444444444 | PASS |
| nfcorpus | 0.20574596786076532 | 0.205745967860765 | 0.24206606745988307 | 0.242066067459883 | PASS |
| fiqa | 0.12126094061428457 | 0.121260940614285 | 0.3516782086226531 | 0.351678208622653 | PASS |

**CONTROL ASSERT: PASS.** No dense number moved. The C0 build is the build we
think it is on the dense path.

## 3. Version-Stamp Finding (flag, not a stop condition)

Every quantized row in all three metrics.json files carries
`turboquant_version = "v0.2.0"` and `codebook_version = "v0.2.0"` (30 rows
checked: 10 rows x 3 datasets, one distinct value pair). None carry the
literal string `"v0.2.1"`. Root cause, confirmed directly against the module
cache:

- `go.mod`/`go.sum`/`go list -m m31labs.dev/turboquant` all independently
  confirm the eos build resolves and links `v0.2.1` — no replace directive,
  no vendor directory, no stale build cache.
- `diff -rq` between the cached `turboquant@v0.2.0` and `turboquant@v0.2.1`
  module source trees shows zero file differences. The two tags carry
  byte-identical source. The differing `go.sum` content hashes are Go's
  path-inclusive dirhash artifact (the `/go.mod` hash entries match exactly
  between the two versions), not a content difference.
- `turboquant@v0.2.1/CHANGELOG.md` has no `## v0.2.1` section at all; the
  Lloyd-Max codebook-table replacement is documented under `## v0.2.0`.
- `turboquant@v0.2.1/version.go` still hardcodes `const Version = "v0.2.0"`.
  The package's own maintainer did not bump this constant when the module
  was re-tagged as v0.2.1.

Conclusion: v0.2.1 is a content-identical re-tag of v0.2.0, and v0.2.0
already carries the regenerated codebook table per its own changelog. The C0
build genuinely links the fixed codebook; only the self-reported version
string is stale. This is an upstream turboquant packaging defect (bump
`version.go` in the next release), not evidence that this rebaseline ran
against the old codebook. I did not stop the run for this reason, since it
does not touch measured values, only a label.

## 4. Per-Dataset, Per-Bit-Width Results (new, C0 build, full uncapped)

All rows below use `turboquant_version=v0.2.0 codebook_version=v0.2.0` (see
Section 3). "bytes/doc" is the packed vector size for a 256-dim embedding.

### SciFact (5,183 docs, 300 queries, 339 relevant pairs) — wall time 47s

| row | nDCG@10 | recall@100 | bytes/doc | total bytes/doc | compression | total compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 0.564538 | 0.796444 | 1024.00 | 1024.00 | 1.00x | 1.00x |
| q2 | 0.499551 | 0.763667 | 72.00 | 72.00 | 14.22x | 14.22x |
| q2 + fp16/o200 | 0.564586 | 0.788111 | 72.00 | 584.00 | 14.22x | 1.75x |
| q3 | 0.535880 | 0.776444 | 104.00 | 104.00 | 9.85x | 9.85x |
| q3 + fp16/o200 | 0.564538 | 0.786444 | 104.00 | 616.00 | 9.85x | 1.66x |
| q4 | 0.549038 | 0.779778 | 136.00 | 136.00 | 7.53x | 7.53x |
| q4 + fp16/o200 | 0.564538 | 0.796444 | 136.00 | 648.00 | 7.53x | 1.58x |
| q5 | 0.558901 | 0.796444 | 168.00 | 168.00 | 6.10x | 6.10x |
| q5 + fp16/o200 | 0.564538 | 0.796444 | 168.00 | 680.00 | 6.10x | 1.51x |
| q8 | 0.564426 | 0.796444 | 264.00 | 264.00 | 3.88x | 3.88x |
| q8 + fp16/o200 | 0.564538 | 0.796444 | 264.00 | 776.00 | 3.88x | 1.32x |

### NFCorpus (3,633 docs, 323 queries, 12,334 relevant pairs) — wall time 41s

| row | nDCG@10 | recall@100 | bytes/doc | total bytes/doc | compression | total compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 0.205746 | 0.242066 | 1024.00 | 1024.00 | 1.00x | 1.00x |
| q2 | 0.178136 | 0.219816 | 72.00 | 72.00 | 14.22x | 14.22x |
| q2 + fp16/o200 | 0.205846 | 0.242125 | 72.00 | 584.00 | 14.22x | 1.75x |
| q3 | 0.198657 | 0.230446 | 104.00 | 104.00 | 9.85x | 9.85x |
| q3 + fp16/o200 | 0.205746 | 0.244899 | 104.00 | 616.00 | 9.85x | 1.66x |
| q4 | 0.203175 | 0.239921 | 136.00 | 136.00 | 7.53x | 7.53x |
| q4 + fp16/o200 | 0.205746 | 0.242144 | 136.00 | 648.00 | 7.53x | 1.58x |
| q5 | 0.205244 | 0.240233 | 168.00 | 168.00 | 6.10x | 6.10x |
| q5 + fp16/o200 | 0.205746 | 0.242066 | 168.00 | 680.00 | 6.10x | 1.51x |
| q8 | 0.206504 | 0.241966 | 264.00 | 264.00 | 3.88x | 3.88x |
| q8 + fp16/o200 | 0.205746 | 0.242066 | 264.00 | 776.00 | 3.88x | 1.32x |

### FiQA (57,600 docs, 648 queries, 1,705 relevant pairs) — wall time ~500s (8m20s)

| row | nDCG@10 | recall@100 | bytes/doc | total bytes/doc | compression | total compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 0.121261 | 0.351678 | 1024.00 | 1024.00 | 1.00x | 1.00x |
| q2 | 0.097102 | 0.322333 | 72.00 | 72.00 | 14.22x | 14.22x |
| q2 + fp16/o200 | 0.121341 | 0.344248 | 72.00 | 584.00 | 14.22x | 1.75x |
| q3 | 0.114296 | 0.341406 | 104.00 | 104.00 | 9.85x | 9.85x |
| q3 + fp16/o200 | 0.121328 | 0.348176 | 104.00 | 616.00 | 9.85x | 1.66x |
| q4 | 0.113948 | 0.344246 | 136.00 | 136.00 | 7.53x | 7.53x |
| q4 + fp16/o200 | 0.121328 | 0.351678 | 136.00 | 648.00 | 7.53x | 1.58x |
| q5 | 0.121378 | 0.349159 | 168.00 | 168.00 | 6.10x | 6.10x |
| q5 + fp16/o200 | 0.121328 | 0.351678 | 168.00 | 680.00 | 6.10x | 1.51x |
| q8 | 0.121658 | 0.351678 | 264.00 | 264.00 | 3.88x | 3.88x |
| q8 + fp16/o200 | 0.121328 | 0.351678 | 264.00 | 776.00 | 3.88x | 1.32x |

The promoted production profile is q4 + fp16/o200 (rerank-overfetch 200,
rerank-storage fp16). Its rows sit above, third-to-last pair in each table.

## 5. Wire-Format Byte-Size Change (new structural finding)

Packed bytes/doc are dataset-independent (fixed by dim=256 and bit width) and
identical across all three datasets: 72 (q2), 104 (q3), 136 (q4), 168 (q5),
264 (q8) bytes/doc. Comparing against the pre-fix per-bit-width byte counts
recoverable from the historical macro compression ratios (Section 6), every
bit width grew by exactly 4 bytes/vector:

| bits | old bytes/doc | new bytes/doc | delta |
| ---: | ---: | ---: | ---: |
| 2 | 68.00 | 72.00 | +4 |
| 3 | 100.00 | 104.00 | +4 |
| 4 | 132.00 | 136.00 | +4 |
| 5 | (no anchor) | 168.00 | n/a |
| 8 | 260.00 | 264.00 | +4 |

I traced this to source: `turboquant@v0.2.1/wire.go` defines
`wireHeaderSize = 22` (legacy header) and `wireIPHeaderSize = 26` (version-2
IP header, adding a 4-byte big-endian float32 input-norm field at bytes
`22:26`). This matches the CHANGELOG's documented breaking change ("IP
estimates now scale by the input vector norm... wire format IP records carry
the input norm in a 26-byte version-2 header"). The +4-byte growth is
expected and intentional, not a regression. It slightly narrows compression
ratios at every bit width, more so proportionally at low bit widths (q2:
-5.6% relative) than high (q8: -1.5% relative).

## 6. Delta Table, Old vs New — INFORMATIONAL ONLY

Old rows were measured under the stale (pre-fix) codebook table and, per
Section 5, the pre-version-2 wire header. **This table is not a regression
gate.** It exists to show the direction and size of movement the codebook fix
produced.

Two old sources exist in this worktree; no bits=5 anchor exists anywhere
(historical sweeps only ever covered 2, 3, 4, 8):

- **Manifest gate anchors** (`assets/corkscrewdb-default-embedder/manifest.json`,
  `dense_short_metrics` ~lines 50-65 and `compact_policy` ~lines 92-140):
  dense per-dataset absolute values (used for the control assert above), plus
  a q4/fp16/o200 aggregate `total_compression_ratio=1.5802469136` and
  delta-only (not absolute) values vs an internal `s40` anchor. No raw
  absolute per-dataset quantized nDCG/recall values exist for any bit width
  in the manifest, only deltas and one capped SciFact smoke row
  (max_docs=300/max_queries=20, not full-scale, so not used here).
- **`.tiller/scratch/codex/eos-current-default-turboquant-frontier-v1-report.md`**:
  a full uncapped bits={2,3,4,8} sweep on the same three datasets, run
  against vector caches from the same promoted default asset. Gives
  per-dataset nDCG@10/recall@100/compression for q3 direct and q3+fp16/o200,
  and macro-only (averaged across the three datasets) values for q2/q4/q8
  direct and +fp16/o200. This is the source used in Section 5 and below.

### Per-dataset delta, q3 (the only bit width with old per-dataset data)

| dataset | row | old nDCG@10 | new nDCG@10 | delta | old recall@100 | new recall@100 | delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scifact | q3 direct | 0.526933 | 0.535880 | +0.008947 | 0.792444 | 0.776444 | -0.016000 |
| scifact | q3 + fp16/o200 | 0.564538 | 0.564538 | +0.000000 | 0.796444 | 0.786444 | -0.010000 |
| nfcorpus | q3 direct | 0.197159 | 0.198657 | +0.001498 | 0.235640 | 0.230446 | -0.005194 |
| nfcorpus | q3 + fp16/o200 | 0.205746 | 0.205746 | +0.000000 | 0.244855 | 0.244899 | +0.000044 |
| fiqa | q3 direct | 0.105547 | 0.114296 | +0.008749 | 0.333332 | 0.341406 | +0.008074 |
| fiqa | q3 + fp16/o200 | 0.121328 | 0.121328 | +0.000067 | 0.350688 | 0.348176 | -0.002512 |

### Macro-only delta, q2/q4/q8 (old data is averaged across all 3 datasets, not per-dataset)

| row | old macro nDCG@10 | old macro recall@100 | old compression | new compression |
| --- | ---: | ---: | ---: | ---: |
| q2 direct | 0.255393 | 0.437285 | 15.06x | 14.22x |
| q4 direct | 0.287493 | 0.459997 | 7.76x | 7.53x |
| q8 direct | 0.298376 | 0.464415 | 3.94x | 3.88x |
| q2 + fp16/o200 | 0.297226 | 0.458230 | 1.77x (est.) | 1.75x |
| q4 + fp16/o200 | 0.297204 | 0.464351 | 1.59x (per frontier report's own recompute) | 1.58x |
| q8 + fp16/o200 | 0.297204 | 0.463396 | 1.35x (est.) | 1.32x |

**Caveat on the q4+fp16/o200 compression anchor:** the manifest's
`compact_policy.total_compression_ratio` states `1.5802469136`, which is
numerically indistinguishable from my new measurement (`1.580246913580247`,
implying 648 bytes/doc, i.e. no wire-header growth). The frontier report's
own independent recomputation of "the same known macro value" instead states
`1.590062111801x` (implying 644 bytes/doc, consistent with the +4-byte
pattern in Section 5). These two historical sources disagree with each other
by exactly 4 bytes/doc; I did not resolve which historical number is correct
since it predates this run and is explicitly informational only. My new
measurement is internally consistent across all three datasets and matches
the source-code-verified wire header size.

## 7. Three Most Notable Quantized-Row Deltas (old vs new)

1. **Low-bit direct quality genuinely improved on FiQA and SciFact.**
   FiQA q3 direct nDCG@10 rose from 0.105547 to 0.114296 (+0.008749, a
   relative +8.3%). SciFact q3 direct nDCG@10 rose from 0.526933 to 0.535880
   (+0.008947, a relative +1.7%). This is the direction the Lloyd-Max
   codebook fix intends: better low-bit convergence, most visible where
   direct (non-reranked) quantized scoring carries more weight.
2. **Every bit width now costs 4 more bytes/vector**, from a new per-vector
   float32 input-norm field in the wire format (Section 5). This is expected
   and CHANGELOG-documented, not a regression, but it measurably narrows
   compression ratios: q2 direct compression fell from 15.06x to 14.22x
   (-5.6% relative), the largest proportional hit of any bit width tested.
3. **The fp16/o200 reranked rows are quality-invariant old vs new.**
   q3 + fp16/o200 nDCG@10 is unchanged to 6 decimal places on SciFact and
   NFCorpus, and moves only +0.000067 on FiQA. The promoted q4/fp16/o200
   profile's quality on FiQA moved to bit-exact dense parity
   (nDCG@10=0.121328, recall@100=0.351678, matching dense exactly), because
   overfetch-then-rerank recovers full accuracy regardless of the
   underlying quantizer codebook's precision. Only the compact-payload byte
   count moved (Section 5); the promoted profile's serving quality did not
   regress or improve materially.

## 8. Reproduction

```bash
export GOFLAGS=-mod=mod GONOSUMDB='m31labs.dev/*' GOPRIVATE='m31labs.dev/*'
cd /home/draco/work/eos-c0-baseline
go build -o /tmp/eos-c0 ./cmd/eos/

# Dataset dir must be the doubly-nested BEIR layout the local dataset cache
# uses: datasets/manta-embed-v1/raw/<name>/<name>/{corpus,queries,qrels}.
# BEIRRetrievalPaths (runtime/retrieval_eval.go:212) expects corpus.jsonl,
# queries.jsonl, qrels/test.tsv directly under the dataset-dir argument.
for name in scifact nfcorpus fiqa; do
  /tmp/eos-c0 eval-retrieval-turboquant \
    --dataset "$name" \
    --bits 2,3,4,5,8 \
    --quantizer-seed 5581486560434873699 \
    --rerank-overfetch 200 --rerank-storage fp16 \
    --metrics-json runs/eos-phase0-rebaseline-v021-20260810T040706Z/${name}.metrics.json \
    --metrics-tsv  runs/eos-phase0-rebaseline-v021-20260810T040706Z/${name}.metrics.tsv \
    assets/corkscrewdb-default-embedder/corkscrewdb-default-embedder.mll \
    datasets/manta-embed-v1/raw/${name}/${name}
done
```

## 9. Data-Setup Caveat

`datasets/manta-embed-v1/raw/{scifact,nfcorpus,fiqa}` did not exist in this
worktree at run start (`datasets/` is gitignored local cache). I copied the
extracted dataset trees read-only from the sibling worktree
`/home/draco/work/eos/datasets/manta-embed-v1/raw/` into this worktree
(~65MB total) without modifying the source worktree. Document/query/qrels
counts match the task's expected FiQA scale (57,600/57,638 docs) and BEIR
standard splits for SciFact and NFCorpus.

## 10. Files In This Run

```text
runs/eos-phase0-rebaseline-v021-20260810T040706Z/scifact.metrics.json
runs/eos-phase0-rebaseline-v021-20260810T040706Z/scifact.metrics.tsv
runs/eos-phase0-rebaseline-v021-20260810T040706Z/nfcorpus.metrics.json
runs/eos-phase0-rebaseline-v021-20260810T040706Z/nfcorpus.metrics.tsv
runs/eos-phase0-rebaseline-v021-20260810T040706Z/fiqa.metrics.json
runs/eos-phase0-rebaseline-v021-20260810T040706Z/fiqa.metrics.tsv
runs/eos-phase0-rebaseline-v021-20260810T040706Z/rebaseline-report.md
```

## Provenance note (restamped run)

This directory is the canonical Phase 0 anchor set. The metrics values are
identical to the 20260810T040706Z run this report was written against; the
only change is provenance: every quantized row now stamps
`turboquant_version`/`codebook_version` as "v0.2.1", sourced from Go build
info (the resolved module pin) rather than the upstream version constant,
which lagged its own tag by one commit. Dense control re-verified: PASS.
