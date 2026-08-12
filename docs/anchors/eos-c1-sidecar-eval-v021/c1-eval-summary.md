# C1 sidecar evaluation (2026-08-10, turboquant v0.2.1)

Full uncapped scifact/nfcorpus/fiqa, seed 5581486560434873699, RTX 5070 Ti.

Conclusions:
- compact-reconstruct reranks the SAME codes at the SAME width: rerank rows are
  bit-identical to direct rows (proven all datasets). The spec C1 mixed-width
  tier (q4 retrieve + q8 sidecar rerank, 400 B) is NOT expressible with current
  flags; needs --rerank-bits (second quantizer + sidecar accounting).
- Keep promoted q4+fp16/o200 for now: q4 rerank-proxy fails quality everywhere;
  q8 proxy passes quality but fails p95 on all three datasets.
- Discovery: q5-direct (168 B, 6.10x) is near-dense on all sets and rides the
  prepared-LUT fast path: fiqa p95 5.7 ms vs promoted 81.2 ms (14x faster) at
  equal quality. Only scifact leaves -0.0056 nDCG vs dense. q5+light-rerank is
  the leading next-tier candidate once mixed-width rerank exists.
- Promoted-baseline rows reproduce the Phase 0 anchors to six decimals.
- p95 caution: bit-5 is consistently 5-15x faster than bit-4/8 on this backend
  (reproducible; LUT path), and sub-30ms p95 varies 15-38% between runs at
  these query counts — weigh single-run p95 gates accordingly.
- Spec erratum: the "q5 fallback (368 B/vector)" figure is not reproducible
  from any harness accounting (q5 codes are 168 B at dim 256).
