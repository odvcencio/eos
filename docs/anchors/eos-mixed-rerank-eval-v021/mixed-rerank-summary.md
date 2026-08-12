# Mixed-width rerank evaluation (2026-08-12, turboquant v0.2.1)

Full uncapped scifact/nfcorpus/fiqa, seed 5581486560434873699, --rerank-bits harness.

- q4+q8 (400 B/vector, 2.56x) and q5+q8 (432 B, 2.37x): quality gate PASS 6/6
  (within 0.002 nDCG@10 of dense; several rows above dense).
- q4+q8 and q5+q8 nDCG are bit-identical on all sets: at overfetch 200 the q8
  sidecar's ceiling governs; the primary width affects only latency.
- q5+q8 p95 beats q4+q8 everywhere (LUT fast path), 5.3x at fiqa scale
  (24.3 ms vs 127.9 ms); fp16 baseline still fastest on small corpora.
- Recommendation: stage q5+q8 as candidate replacement for the promoted
  q4+fp16 profile (equal quality, 648 -> 432 B/vector), gated on a repeat-run
  p95 check at fiqa-or-larger scale. Keep q4+q8 for hard 400 B budgets.
- Known gap: --per-query-jsonl diagnostics do not yet reflect mixed-width rerank.
