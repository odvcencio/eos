---
mdpp: "0.1"
title: "EOS Training And Inference Performance Plan"
date: 2026-08-15
status: draft
scope: "10-14 week implementation specification for EOS training quality, training efficiency, inference performance, Apple acceleration, and Go SIMD experiments"
---

# EOS Training And Inference Performance Plan

This is an implementation specification, not a result announcement. It reconciles current EOS repository evidence, Hyphae memory, and current official Go and Apple documentation available on 2026-08-15. Measurements are labeled as measured. Everything else is proposed work, a dependency, or a gate.

## Executive Decision Summary

1. EOS keeps Go-native compiler/runtime ownership. Core Go owns orchestration, ABI and artifact schemas, and generated kernels. The `.mll` artifact remains backend-neutral and EOS-owned. External frameworks are optional adapters, parity oracles, import/export bridges, or product deployment surfaces. They fail closed and do not define default runtime semantics.
2. CUDA remains the performance reference. K4 closes the contiguous resident-gradient slab; K5 reduces the resident optimizer Go/C boundary behind an optional typed batch contract. The current CUDA driver boundary is still a narrow cgo wrapper, not zero FFI; the roadmap is machine-generated typed wrappers and direct driver ownership. Optimize call count and synchronization count before byte count.
3. Apple core is Direct Metal first. MLX/MLX-C is not a core backend. The Apple path is persistent `MTLBuffer` residency and command batching, followed by selective cached MPSGraph dense islands. MLX-C comes last as an optional parity/import/teacher oracle outside default builds and default runtime.
4. Backend truth precedes kernel breadth. Add a versioned backend feature/requirement contract and per-run accounting for host/device/fallback steps, fallback reasons, upload/download bytes, syncs, residency hits/misses, and `full_device_execution`. Artifact device-residency requirements fail closed.
5. Inference work moves data structures onto the device. Build persistent/fused embedding inference, device-resident TurboQuant prepared index plus top-k and q5+q8 rerank, compressed KV/TurboSparse decode, stronger Metal paths, and WebGPU only after persistent buffers and delayed readback are real.
6. Keep `go.mod` on Go 1.26. As of 2026-08-15, Go 1.27 release notes are still draft and explicitly say Go 1.27 is not released. `GOEXPERIMENT=simd` adds portable, vector-size-agnostic `simd`, revises amd64 `simd/archsimd`, and adds arm64 NEON and WebAssembly 128-bit SIMD; the API remains unstable. Go 1.26 scalar/default is the release baseline. SIMD is limited to build-tagged microbench lanes with scalar parity/fallback and no production dependency until release/API stability and representative wins.
7. Teacher and data signal come before bigger models. Complete provenance-safe Qwen3/mxbai teacher-cache scoring, agreement, margin, and leak filtering before one bounded dense pilot. Scale only after macro nDCG moves by at least `+0.001` with floors intact; promotion target is `+0.010`.
8. Compact serving promotion is parallel, not dense-quality evidence. q5+q8 at `432 B/vector` passed 6/6 quality gates as a candidate replacement for q4+fp16 at `648 B/vector`, but needs repeated p95 timing. That does not prove dense model quality.

## Current Execution Checkpoint: Kernel-First K0 (2026-08-13)

The first implementation slice of the native toolchain is now in the branch,
behind the existing source-backed behavior:

- The compiler emits a versioned `KernelABI` for generated CUDA and Metal
  variants. Pure-Go GoTreeSitter parses the generated C++-compatible signature;
  vendor qualifiers and Metal attributes are masked in a byte-preserving view,
  while address spaces, access mode, and binding locations are recovered from
  the original source. The ABI carries the parser mode and a SHA-256 fingerprint
  of the exact source.
- `KernelVariant.Binary` is an optional, hash-checked offline image descriptor
  (`ptx`, `cubin`, `air`, or `metallib`). `compiler.CompileKernelVariants` and
  `eos compile --offline-backend ...` are the explicit build actions that attach
  images; each image carries both its own hash and the source fingerprint it was
  compiled from. Ordinary compilation still produces portable source-backed
  artifacts.
- CUDA's native loader accepts a sealed offline image and calls the driver
  module loader directly; Darwin's Metal path does the same for `metallib`.
  NVRTC/Metal source compilation remains the compatibility fallback when a
  variant has no compatible image. This removes runtime source compilation
  without pretending that driver module/pipeline load and launch FFI can
  disappear.
- MLL sealing now places image bytes in the skippable `XKBI` custom section and
  leaves the backend-neutral artifact metadata source-backed. This avoids
  base64 expansion in `XMTA`; readers that do not understand `XKBI` still retain
  the source/host-fallback contract.

The next correctness/performance gate is the K1 dispatch contract slice:

- Generated CUDA/Metal ABI arguments now carry an explicit `kind` (`pointer` or
  `value`) in addition to type, access, address space, and location. This is
  the minimum information a future generic Go-to-driver argument bridge needs;
  runtime code does not guess from C spellings.
- The backend derives a launch contract for the currently promoted row-wise,
  RoPE, elementwise, and score families, validates it against the typed ABI
  before native compilation, and records a stable contract fingerprint. Metal
  built-in thread identifiers are excluded from the runtime buffer vector.
- Every symbolic run now exposes structured `ExecutionAccounting` plus stable
  metadata for total/kernel/device/host/fallback steps, kernel launches,
  upload/download bytes, synchronizations, graph captures/replays, residency
  hits/misses, fallback reasons, and the strict `full_device_execution` claim.
  Missing runtime telemetry remains zero rather than being inferred from a
  backend name.
- Legacy source-backed variants without an ABI remain loadable and are marked
  `launch_abi_status=legacy_unverified`; newly emitted variants are validated.
  A mismatch in a present ABI fails before driver compilation, preserving the
  fail-closed behavior instead of silently launching a wrong argument vector.

Measured in this checkpoint: compiler, artifact, backend, CUDA, Metal, and CLI
tests pass; no CUDA compiler is installed in the current Linux environment, so
no PTX throughput or device-parity claim is made here.

## Current Preparation Checkpoint: K2 (2026-08-14)

The next pre-training hardening slice removes avoidable allocation and launch
plumbing overhead without changing the backend-neutral artifact contract:

- Compact resident training now reuses activation and backward arenas by exact
  `CompactForwardShape`. Consumed handles remain precisely recognizable as
  already released, while their device buffers move to a per-shape pool
  bounded by consumed arena concurrency and flushed on reconfiguration or
  close.
- The trainer memoizes the layer/name/shape/position configuration around its
  per-batch preparation hook, so unchanged preparation does not call the
  explicit reconfiguration API (which still flushes safely when invoked by a
  caller) or flush the warm pool/resident gradients.
- Compact train int32 token/mask/role/status staging buffers are copied into
  the reusable arena instead of being freed and reallocated for every
  microbatch. `ArenaReuseHits` and `ArenaAllocations` are explicit counters for
  the warm-up/performance gate.
- Generated CUDA row-wise, RoPE, elementwise, and row-score families use a
  typed argument bridge (`typed_args_v1`) backed by the validated K1 launch
  contract. The bridge keeps pointer/value storage stable for the driver and
  provides one common path for the later batched/asynchronous launch bridge;
  existing auxiliary wrappers remain until their contracts are promoted.
- Metal RoPE emission now carries `seq_len` and resets batched positions by
  sequence, keeping the cross-backend K1 contract truthful instead of falling
  through to a four-argument stale variant.

The K2 verification gate is green for `go test ./...`, `CGO_ENABLED=0 go test
./...`, `go vet ./...`, and the exact-shape arena-pool unit tests. These are
allocation/ABI correctness gates only; no CUDA device throughput or embedder
quality claim is made until a quiet-host warm benchmark records launch,
synchronization, byte, residency, and loss-parity deltas.

The host-only pool microbenchmark (`-benchtime=100x -benchmem`) measured
`72.71 ns/op`, `0 B/op`, and `0 allocs/op` on the current Intel Core Ultra 9
285 environment. This validates the reuse bookkeeping, not device allocation
latency; the CUDA warm-run gate remains outstanding.

A live two-step CUDA parity profile at the small `B=1,T=2,D=4,H=6,L=2`
fixture then recorded `ArenaReuseHits=1`, `ArenaAllocations=1`, `136` kernel
launches, `4` synchronizations, and `FallbackOrUnhandled=0`; gradients and
weights remained within the existing hard parity tolerance. This is a
correctness and reuse signal, not a throughput claim or a release-shape
benchmark.

## Current Preparation Checkpoint: K3 (2026-08-15)

The next compact-train preparation slice removes another warm-step transfer and
allocation source:

- Resident gradient buffers now warm in an exact-element-count pool. A normal
  step still gets fresh generation/token metadata, while a later step reuses
  the device allocation after the prior step's resident-gradient consumers have
  finished. Reconfiguration and close flush the pool; abort recycles it safely.
- Gradient initialization uses CUDA `cuMemsetD32` instead of allocating a host
  zero slice and copying it over the driver boundary for every resident
  parameter. The old host-side path is no longer on the compact resident-train
  step boundary.
- Compact-forward and compact-train token and mask flattening reuse host
  staging capacity under their accelerator mutexes, so repeated buckets do not
  allocate short-lived flattened slices before the device copy.
- `GradientReuseHits` and `GradientAllocations` make the warm-up behavior
  observable alongside the K2 arena counters.

K3 verification is limited to lifecycle/allocation correctness and native-only
build safety. It does not claim a device throughput win until the target-shape
warm benchmark measures the reduced allocation and transfer path.

The live two-step `B=1,T=2,D=4,H=6,L=2` CUDA parity fixture now records
`GradientReuseHits=15`, `GradientAllocations=15`, and `GradientZeroCalls=2`
after the trainer releases sealed gradients; hard parity, `136` launches, and
`4` synchronizations remain unchanged. This is a warm-path readiness signal,
not a release-shape throughput result.

The `15/15` K3 values are retained as historical K3 evidence, but they are
not the current physical-allocation model. K3 counted one event per logical
resident gradient view. K4 changes the accounting and implementation to one
layout-keyed contiguous slab containing those logical views: a cold exact-shape
step has one physical slab allocation, a warm step has one slab reuse, and one
`cuMemsetD32` zeroes the acquired slab per step. `ResidentGradBytes` remains
the logical sum of the per-ref byte lengths. The K4 warm benchmark reports the
measured-step delta after its excluded warm-up as `GradientAllocations=0` and
`GradientReuseHits=1`; it does not rewrite the K3 history.

## Current Preparation Checkpoint: K4 (2026-08-15)

K4 closes the resident-gradient preparation and warm-gate slice:

- One layout-keyed contiguous CUDA gradient slab owns the logical resident
  gradient refs for an exact shape. The layout key is deterministic and
  independent of ref ordering; lifecycle transitions invalidate stale tokens,
  recycle the active slab safely, and flush active and pooled slabs on close or
  reconfiguration.
- `BeginCompactTrainStep` acquires and zeroes exactly one slab with one
  `cuMemsetD32` call per step, then publishes gradient refs only after fallible
  device work succeeds. `GradientAllocations`, `GradientReuseHits`, and
  `GradientZeroCalls` are physical slab-level counters; they no longer imply
  one device allocation or memset per logical gradient ref.
- The approved K4 lifecycle review found no P0/P1 issue in contiguous layout,
  pooling, stale-token invalidation, Begin/Release/Abort/Configure/Close
  transitions, lock ownership, or gradient-pointer consumers. This is a
  lifecycle/counter checkpoint, not a throughput claim.

The repaired CUDA warm gate is opt-in and exact-profile only. It accepts the
canonical synthetic fixture or the `next`/`descriptor` profile; arbitrary
shape overrides and the incomplete checkpoint-only estimator are not accepted.
Each run is exactly one excluded warm-up plus one measured `b.N=1` step and
must use `-benchtime=1x`. The gate fails closed before CUDA setup when
`b.N != 1`; an unset opt-in gate intentionally skips. Factory/setup errors,
nil or non-CUDA backends, and any nonzero compact-train fallback/unhandled
counter are fatal. The measured allocation/reuse delta is `0/1` for both
accepted profiles because the warm-up owns the cold allocation.

The latest qualified K4 target-shape diagnostic is
`B=1,T=256,D=128,H=256,L=2,O=128,heads=4`: `629.733 ms` host wall time,
`65` compact kernel launches, `2` compact synchronizations, `0` compact
fallback/unhandled, `0` graph captures/replays, and gradient allocation/reuse
`0/1`. The canonical profile measured `3.917 ms`, `68` launches, `2`
synchronizations, and the same clean fallback/graph and `0/1` allocation/reuse
delta. These are local, synthetic, one-step, shared-host diagnostics only;
neither value is a release-throughput, production latency, or graph-support
claim.

## Current Preparation Checkpoint: K5 (2026-08-15)

K5 adds the optional typed `ResidentGradientOptimizerBatchAccelerator` batch
contract while retaining the scalar path for unsupported backends and callers.
Complete preflight still rejects
duplicate, stale, mixed-owner, metadata-mismatched, or already-used gradient
refs before launch. The CUDA batch wrapper submits the existing optimizer
kernel once per descriptor and performs one mandatory completion barrier;
`DeferSync` suppresses only immediate host parameter copies. Partial enqueue or
barrier failures drain and poison the step rather than publishing it.

For the canonical `N=15` resident optimizer updates, source and counter
evidence is deterministic: Go/C entries change `15 -> 1`, context sets change
`15 -> 1`, and completion barriers change `15 -> 1`, while CUDA kernel launches
remain `15`. This is a call and barrier reduction, not a kernel-count or
throughput claim. Live telemetry reports batch calls / resident kernel
launches / batch kernel synchronizations as follows:

| profile | batch calls | resident launches | batch syncs | result |
| --- | ---: | ---: | ---: | --- |
| canonical one-step | 1 | 15 | 1 | CUDA batch path; compact fallback `0` |
| next one-step | 1 | 14 | 1 | CUDA batch path; compact fallback `0` |
| two-step parity | 2 | 30 | 2 | immediate and deferred parity pass |
| host/scalar baseline | 0 | 0 | 0 | scalar fallback remains intact |

The resident parity gates also report clean handles and no compact fallback or
unhandled work. The latest single-sample K5 timings vary with shared-host load,
approximately `4.5-5.5 ms` for canonical and `651 ms` for next, so K5 makes no
speedup claim. The durable win is the measured deterministic reduction in
Go/C entries, context sets, and barriers; repeat quiet-host timing is required
before any wall-clock claim.

K5 reduces one narrow cgo boundary; it does not make CUDA FFI zero. Core Go
continues to own orchestration, typed ABI contracts, artifact schemas, and
generated kernels. The current CUDA driver boundary remains a narrow cgo
wrapper. The longer-term architecture is machine-generated typed wrappers and
direct driver ownership, with every reduction validated against scalar parity,
fallback accounting, and failure poisoning.

## Current Preparation Checkpoint: K6 (2026-08-15)

K6 completes the compact-forward readback slice with a typed
status-to-pooled-to-active bridge. The wrapper validates its typed destinations
and sources, sets the CUDA context once, performs synchronous copies in that
order, and retains no Go pointer after return. The successful readback ledger
is:

| readback metric | before | after |
| --- | ---: | ---: |
| Go/C readback entries | 3 | 1 |
| CUDA context sets | 3 | 1 |
| device copies | 3 | 3 |
| readback bytes | unchanged | unchanged |
| extra stream synchronizations | 0 | 0 |

No kernel, optimizer, graph, H2D, compact-byte, compact-sync, or fallback
behavior changed. If the device status is nonzero, the bridge copies status
only, stops before pooled/active, returns the exact existing status error,
publishes no handle, and does not count a successful readback batch. Successful
stage progress remains available for partial-byte accounting and cleanup.

Live compact-forward telemetry is batch entries / context sets / device copies:

| profile | readback entries | context sets | device copies | result |
| --- | ---: | ---: | ---: | --- |
| canonical one-step | 1 | 1 | 3 | K5 batch `1/15/1`, compact `68/2`, fallback `0` |
| two-step parity | 2 | 2 | 6 | K5 batch `2/30/2`, parity and handles clean |
| varying `T=2,3,4` | 3 | 3 | 9 | K5 batch one call/barrier, compact `204/6`, fallback `0` |
| next one-step | 1 | 1 | 3 | K5 batch `1/14/1`, compact `65/2`, fallback `0` |
| host/scalar baseline | 0 | 0 | 0 | host batch telemetry remains zero |

The latest warm samples are `4.537 ms` canonical and `678.554 ms` next,
measured on a shared host with one warm-up excluded. They are deterministic
counter-gate diagnostics only, not throughput or production-latency claims.
The K6 code review approved the slice with no P0/P1 finding. A deterministic
CUDA D2H fault injector is still a residual coverage gap for pooled- and
active-copy failures; status-first early-stop, validation, stage-progress
cleanup, and zero-successful-batch behavior are covered by tests and review.

## Current Preparation Checkpoint: K7 (2026-08-15)

K7 completes the `CUDA-GRAPH` investigation as a bounded correctness and
counter gate for the default-off `EOS_CUDA_COMPACT_TRAIN_FORWARD_GRAPH` flag:

- The graph covers forward compute only. H2D uploads, K6 status/pooled/active
  readback, backward, and optimizer work remain outside it. The first exact-key
  call runs direct and then records a non-executing capture; later exact-key
  calls replay.
- A cache hit requires the exact shape/configuration plus stable bound pointers,
  allocator/arena/workspace/resident generations, and device, CUDA-context, and
  stream identities. Pointer equality is mandatory; a near-shape or same-size
  replacement is not a hit. Compact graph owners now close before
  optimizer-owned resident weights, including constructor rollback and
  alias-safe teardown.

| warm profile | direct forward submissions | whole-step direct submissions | graph executed nodes | total device work |
| --- | ---: | ---: | ---: | ---: |
| canonical | `24 -> 0` | `68 -> 44` | `24` | `68` |
| next | `22 -> 0` | `65 -> 43` | `22` | `65` |

Total compact synchronizations remain `2`, graph-local normal synchronizations
remain `0`, and K5 resident-optimizer plus K6 packed-readback counters are
unchanged. With the flag off, the direct baseline remains `24/68` (canonical)
and `22/65` (next), with graph counters at zero.

Failure staging is truthful: failed capture never publishes a partial cache;
post-enqueue replay/boundary failure drains the queued work, invalidates the
cache, records failure/fallback, resets gather state, and runs the direct
fallback. Teardown uses a context-safe graph owner. K7 reduces a narrow CUDA
cgo boundary but does not make CUDA FFI zero. These synthetic exact-profile
checks make no speedup claim. Reset-generation/runtime-health invalidation,
cache eviction for many exact buckets, and expanded graph-on/off activation and
backward parity remain risks. Representative repeated end-to-end performance,
including H2D and D2H, must still show at least `10%` improvement for
promotion; the selected-package BGE/FiQA training remains blocked and must not
start.

## Reconciled Progress And Evidence Ledger

Historical pre-S3 documented CUDA baseline:

- The documented pre-S3 benchmark path is `845.15` train examples/s and `865437.87` train pairs/s at batch 1024.
- Boundary: historical baseline only. These numbers are explicitly stale for current S3 throughput and must not be described as the current default after S3 work.
- Evidence: `docs/benchmarks.md`; `README.md`.

Measured or reported training-efficiency progress:

- S2 compact/Matryoshka accelerator un-gating reports `1.08x`.
- S1a GEMM contrastive loss reports `1.72x` on the compact lane and about `33%` less device memory.
- S7 tokenizer training dropped from about 40 minutes to `5.07s`.
- Boundary: Hyphae spec progress notes; remeasure S2 and S1a at release shape before promotion. S7 closes tokenizer training gate G5, not model quality.
- Evidence: `/home/draco/.hyphae/spaces/m31labs-eos/specs/eos-embedder-v1-upgrade.md`; `runtime/tokenizer_train_incremental_test.go` references S7.

Measured or reported residency progress:

- S3a/b resident attention behind `EOS_EMBED_ATTN_RESIDENT` cut upload `33.8%`, download `7.8%`, matmul runs `25.3%`, and wall-clock improved `4.1%`.
- S3c reports deterministic deltas of matmul runs `-31.8%`, upload `-52.75%`, and download `-7.8%`; wall-clock is withheld under shared-host load.
- S3d has pre-merge validation evidence from a bounded reconstructed 2,560-example / 20-step / 1,310,720-pair A/B: OFF/ON loss `5.3669305` / `5.3669443`, delta `+0.0000138`; optimizer updates and contrastive calls exact; matmul runs `336288 -> 287596` (`-48692`); upload `44045.75 -> 36089.97 MB` (`-7955.78 MB`); download `59287.37 -> 50404.74 MB` (`-8882.63 MB`). No wall-clock claim is made under rising load. The exact prior corpus is absent, and explicit skip/flush counters were not surfaced.
- Evidence: local pre-merge scratch reports `.tiller/scratch/codex/eos-s3d-integration-repair-v1-report.md` and `.tiller/scratch/codex/eos-s3d-deterministic-gate-v1-report.md`; durable facts are captured in this plan and the Hyphae follow-up.

Measured compact retrieval anchors:

- Phase 0 TurboQuant v0.2.1 rebaseline exists. Dense controls match; quantized rows stamp stale package strings due an upstream cosmetic bug.
- Correct q4+fp16/o200 accounting is `648 B/vector` and `1.5802469136x`; this supersedes stale `1.5900621118x` claims in older docs and scratch reports.
- q5+q8 at `432 B/vector` and q4+q8 at `400 B/vector` pass 6/6 mixed-rerank quality gates; candidate only, repeat p95 at FiQA-or-larger scale before promotion.
- Newly verified TurboQuant contract: v0.2.1 default Hadamard rotation uses `rounds=3`; the current CUDA native encode/decode/TurboSparse path supports only `rounds=1`. The planner now parses rounds and fails closed on absent/default or explicit multi-round specs instead of silently downgrading.
- Evidence: `docs/anchors/eos-phase0-rebaseline-v021/rebaseline-report.md`; `docs/anchors/eos-mixed-rerank-eval-v021/mixed-rerank-summary.md`; `docs/eos-distillation-compact-default-spec.md` is stale on the q4+fp16 ratio.

Durable implementation lessons:

- cgo call count dominates current GPU-offload cost on the measured host. This is a durable design lesson, not a universal hardware law.
- K4 and K5 make that lesson measurable without overstating ownership: Go
  still owns orchestration, ABI/artifacts, and generated kernels; the current
  CUDA driver boundary is a narrow cgo wrapper. K5 removes 14 of 15 resident
  optimizer Go/C entries and barriers in the canonical batch contract, but FFI
  is not yet zero. Machine-generated typed wrappers and direct driver ownership
  remain the roadmap.
- Duplicate resident refs are about `18-20%`; only `refcount==1` can skip downloads until upstream duplicate gradients are coalesced.
- Wall-clock A/B under shared load is noisy; deterministic counters are the first gate.
- K4's `15/15`-to-slab correction is a counter-definition change, not a
  historical rewrite: K3 per-ref counters remain in the ledger, while K4
  physical slab counters report one cold allocation and warm reuse (the
  measured warm-gate delta is `0/1`).
- K6 makes packed readback equally explicit: three Go/C/context boundaries
  become one, but three device copies and all readback bytes remain. Status is
  copied first and a nonzero status publishes no handle; deterministic D2H
  fault injection for pooled/active copies remains a coverage gap.
- Evidence: `/home/draco/.hyphae/spaces/m31labs-eos/inbox/agents/2026-08-12-sequoia-s3-residency-lessons.md`; `.tiller/scratch/codex/eos-k4-gradient-slab-review.md`; `.tiller/scratch/codex/eos-k4-ffi-hotspot-analysis.md`; `.tiller/scratch/codex/eos-k5-live-gate.md`; `.tiller/scratch/codex/eos-k6-forward-readback-report.md`; `.tiller/scratch/codex/eos-k6-live-gate.md`.

External current-docs facts:

- Go status as of 2026-08-15: Go 1.27 release notes remain draft and
  explicitly say Go 1.27 is not released. Under `GOEXPERIMENT=simd`, the
  draft describes portable size-agnostic `simd`, a revised amd64
  `simd/archsimd`, arm64 NEON, and WebAssembly 128-bit SIMD; the API is
  unstable. Go 1.26 scalar/default remains the release baseline. Verified from
  the [Go 1.27 draft release notes](https://go.dev/doc/go1.27), [architecture-specific SIMD proposal #73787](https://github.com/golang/go/issues/73787), and [portable SIMD proposal #78902](https://github.com/golang/go/issues/78902).
- Apple Accelerate is documented as CPU vector/BLAS/vDSP/BNNS computation, and Core ML is documented as app integration that can let the OS select CPU/GPU/Neural Engine. Verified from [Apple Accelerate](https://developer.apple.com/documentation/accelerate), [Apple Core ML](https://developer.apple.com/documentation/coreml), and [MLComputeUnits](https://developer.apple.com/documentation/coreml/mlcomputeunits).

### External Primary References

- Go 1.27 / SIMD (status checked 2026-08-15): [Go 1.27 draft release notes](https://go.dev/doc/go1.27), [generic SIMD proposal #78902](https://github.com/golang/go/issues/78902), and [arch-specific SIMD proposal #73787](https://github.com/golang/go/issues/73787).
- MLX and MLX-C: [MLX v0.32.0 release](https://github.com/ml-explore/mlx/releases/tag/v0.32.0), [MLX-C overview](https://ml-explore.github.io/mlx-c/build/html/overview.html), and [MLX-C install](https://ml-explore.github.io/mlx-c/build/html/install.html).
- Apple graph/runtime references: [MPSGraph](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph), [MPSGraphExecutable](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphexecutable), [Accelerate](https://developer.apple.com/documentation/accelerate), [BNNS](https://developer.apple.com/documentation/accelerate/bnns-library/), [Core ML](https://developer.apple.com/documentation/coreml), and [MLComputeUnits](https://developer.apple.com/documentation/coreml/mlcomputeunits).
- NVIDIA references: [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) and [cuBLAS/cuBLASLt documentation](https://docs.nvidia.com/cuda/cublas/index.html).

## Target Architecture And Contracts

EOS should make backend execution truth a load-bearing artifact and run property, not an inference from which package happened to be selected.

``` mermaid
flowchart LR
  MLL["backend-neutral .mll\nrequirements + feature versions"] --> Loader["runtime loader\nfail-closed capability check"]
  Loader --> CUDA["CUDA resident path\nperformance reference"]
  Loader --> Metal["Direct Metal resident path\nApple core"]
  Loader --> CPU["CPU fallback\ntruthfully accounted"]
  Loader --> Portable["Vulkan/DirectML/WebGPU\nportable surfaces"]
  CUDA --> RunAcct["per-run accounting\nsteps, bytes, syncs, residency, fallback reasons"]
  Metal --> RunAcct
  CPU --> RunAcct
  Portable --> RunAcct
```

Backend feature contract:

- Artifact requirements: `.mll` metadata declares required backend features, minimum feature versions, residency requirements, and whether host fallback is allowed for that artifact or run.
- Backend declarations: each backend reports feature versions for device execution, matmul residency, prepared TurboQuant scoring, fused embedding, sparse attention, compressed KV decode, train-resident attention/FFN, graph replay, and whether it has `CapabilityHostFallback`.
- Fallback policy: `host_fallback_allowed` is an artifact/run policy distinct from backend `CapabilityHostFallback`. A backend can be capable of host fallback and still must fail closed when the artifact or run policy forbids fallback.
- Loader behavior: `EOS_REQUIRE_BACKEND` and programmatic backend requirements fail closed when a required feature is missing. Warnings are not enough for promotion gates.
- Run accounting: every run records `backend_selected`, `full_device_execution`, `host_step_count`, `device_step_count`, `fallback_step_count`, `fallback_reasons`, upload/download bytes, sync count, graph captures/replays, residency hits/misses, and per-feature coverage.
- Promotion use: readiness, energy, serving, and training promotion gates inspect the contract fields directly. They cannot rely on backend name alone.

Public/internal interface changes:

- Backend contract: public `.mll` requirement metadata and CLI summaries; internal backend feature registry with versioned ids and strict requirement matcher.
- Run accounting: public train/eval/export manifests and scoreboards; internal shared counters, fallback reason enums, device/host step accounting, and residency hit/miss counters.
- Inference: stable public `embed`, retrieval, sparse attention, and decode entrypoints; internal persistent embedding and prepared-index handles with explicit lifecycle and device ownership.
- Apple: public backend remains `metal`; internal path is Direct Metal resident buffers and command batching first, cached MPSGraph islands second, MLX/Core ML outside default path.
- Go SIMD: no public SIMD types; private build-tagged helpers with scalar fallback and drift tests.

Non-goals:

- No default runtime dependency on MLX, Core ML, BNNSGraph, Python, C++, or hosted providers.
- No `.mll` semantics defined by `mlmodelc`, MLX arrays, Core ML packages, ONNX, or framework-specific graph formats.
- No public Go SIMD API until Go SIMD is stable and proven on representative EOS paths.
- No dense model promotion from hybrid retrieval, q5+q8 serving, imported BGE, or internal pair metrics.
- No broad kernel expansion before truthful fallback and residency accounting lands.

## Accelerant Decision Matrix

- Direct CUDA: P0 performance reference for resident training, prepared retrieval, compressed KV decode, and graph replay. Kill on parity loss, quality-floor loss, or deterministic counter regression beyond gate.
- cuBLASLt: P1 for fixed dense islands after call/sync reductions. Use heuristics, epilogues, and matmul+bias/GELU/LN opportunities. Kill if no measured win over current cuBLAS or if heuristic cache is unstable.
- CUDA Graphs: P1 for fixed-bucket replay after resident-step coordinator. Kill if capture overhead or shape churn erases benefit.
- Direct Metal: P0 Apple core for persistent `MTLBuffer`, command batching, direct kernels, and explicit counters. Kill any Apple core claim that cannot produce truthful device-resident accounting or parity.
- MPSGraph: P2 Apple selective cached dense islands after Direct Metal residency exists. Kill if graph compile/cache cost exceeds dispatch savings or opaque fallback cannot be accounted.
- MLX/MLX-C: P4 optional parity/import/teacher oracle only. Current MLX v0.32 exposes Apple, Linux CUDA, and CPU install/package surfaces; that broadens oracle coverage but does not make MLX an EOS backend or artifact semantic authority. Kill if it pressures default build/runtime, `.mll` semantics, or the core backend path.
- Accelerate/vDSP/BLAS or BNNS: P4 Darwin CPU fallback or coarse-island POC only after accounting. cgo-call overhead makes fine-grained use unattractive. BNNSGraph/`mlmodelc` must not become EOS artifact semantics. Kill if fine-grained calls add overhead, BNNSGraph shapes `.mll`, or accounting cannot distinguish host/backend work.
- Core ML / ANE: P5 export-only deployment adapter and non-goal for core runtime. It lets the OS select CPU/GPU/ANE, which cedes backend/artifact control. Kill any default EOS runtime dependency, promotion gate dependency, or loss of `.mll` backend-neutral ownership.
- WebGPU: P3 portable browser/device surface after persistent buffers and delayed readback. Kill if readback-per-step or host fallback dominates.
- Vulkan / DirectML: P4 portability surfaces until measured device paths exist. Kill performance/readiness claims without device counters.
- Go `simd` / `simd/archsimd`: P3 private CPU POC only with build tags and
  scalar fallback. The Go 1.27 APIs remain draft/unstable as of 2026-08-15,
  so Go 1.26 scalar/default is the release baseline and production must not
  depend on SIMD until release/API stability and representative wins. First
  target TurboQuant prepared scoring, rotations, quant-dequant, and dots; then
  private EOS row dot/norm/layernorm helpers. Kill on less than `1.2x`
  representative EOS CPU path, less than `1.15x` over existing TurboQuant
  assembly where relevant, allocations, ranking drift, or default-build
  regression.

## Phased Roadmap

Phase 0, weeks 1-2, close safety and observability:

- Critical path: close S3d safely, land the K1 launch-contract gate, add
  execution accounting, and add backend feature contract design and first
  manifest fields.
- Exit gate: S3d default-off gates pass, no host fallback after resident errors, and run manifests expose required counters.
- Kernel-first K0 slice: emit typed GoTreeSitter-derived ABI metadata and make
  offline CUDA/Metal images an explicit, hash-checked artifact option. Keep
  NVRTC/source fallback until device parity is demonstrated.
- Kernel-first K1 slice: validate the typed launch contract before native
  compilation and expose run-level device/host/fallback accounting. Keep the
  generic argument bridge and device byte/sync telemetry as the next measured
  sub-gates rather than claiming they already exist.
- K2 preparation slice: warm exact-shape compact-train arenas and staging
  buffers, use the typed CUDA launch bridge for promoted generated families,
  and repair Metal RoPE ABI parity before the next embedder training run.
- K3 preparation slice: device-zero resident gradients, warm gradient buffers,
  and reuse compact-train host flattening capacity before the target-shape warm
  benchmark.
- K4 preparation slice: replace per-ref resident-gradient allocation/zeroing
  with one layout-keyed contiguous slab, expose physical slab allocation and
  reuse counters, and repair the exact-profile one-step CUDA warm gate.
- K5 preparation slice: add the optional typed resident-optimizer batch, prove
  the canonical `15 -> 1` Go/C/context/barrier reduction while retaining `15`
  kernel launches, and keep scalar host fallback/parity clean.
- K6 preparation slice: complete the typed compact-forward status/pooled/active
  readback bridge, reducing Go/C and context boundaries `3 -> 1` while keeping
  device copies, bytes, synchronizations, and fallback semantics unchanged.

Phase 1, weeks 2-4, resident CUDA step skeleton:

- Critical path: investigate fixed-bucket compact CUDA Graph or typed launch-
  array execution after K6 readback completion, first confirming exact device-
  pointer stability and excluding H2D/readback from the candidate slice;
  resident-step coordination, duplicate-ref-safe download elision,
  host/device/fallback accounting, and S3e full-step gradients remain in scope.
- Exit gate: deterministic counters prove fewer calls/syncs, quiet-host mini smoke is non-regressing, and fallback reasons are exhaustive.

Phase 2, weeks 4-7, CUDA performance reference:

- Critical path: S3e full-step gradients, cuBLASLt fixed islands, CUDA Graph replay for fixed buckets.
- Exit gate: batch-1024 mini smoke `>=1000` train examples/s, real-text batch-256 `>=140`, quality parity, and no stale-weight eval.

Phase 3, weeks 5-9, inference device path:

- Critical path: persistent/fused embedding path, device-resident TurboQuant prepared index/top-k, q5+q8 rerank repeat p95.
- Exit gate: q5+q8 repeats quality and p95, and prepared index avoids host readback in measured path.

Phase 4, weeks 7-11, Apple core:

- Critical path: Metal persistent buffers, command batching, accounting parity, then MPSGraph POC.
- Exit gate: Apple run reports truthful device/fallback counters and beats CPU fallback on representative embedding/retrieval paths.

Phase 5, weeks 8-14, decode and CPU SIMD:

- Critical path: compressed KV/TurboSparse decode integration; Go SIMD POC upstream in TurboQuant, then private EOS helpers.
- Exit gate: decode metadata proves compressed KV path, and SIMD gates pass without default-build or ranking drift.

Phase 6, weeks 1-14 parallel, training quality pilot:

- Critical path: teacher/data audit, bounded dense pilot, dense promotion decision.
- Exit gate: pilot scales only if macro nDCG `>= +0.001`, macro recall `>= -0.001`, no dataset nDCG is `< -0.002`, and no dataset recall is `< -0.003`; promotion target remains `+0.010` with accepted release floors.

Parallel lanes:

- Observability depends on nothing: `OBS-EXEC`, `CAP-V2`, benchmark matrix. Stop performance claims if counters are ambiguous.
- CUDA training depends on S3d close: `RES-COORD`, `S3E-STEP`, `CUDA-GRAPH`. Default-off flags and host path remain; resident errors abort the step.
- Inference retrieval depends on accounting fields: `RETRIEVE-GPU`, `Q5Q8-PROMOTE`. Keep q4+fp16 promoted until repeated p95 and quality pass.
- Decode depends on feature contract: `KV-DECODE`. Decode falls back only when artifact allows it and logs it.
- Apple depends on CAP-V2 draft: `METAL-RES`, `MPSGRAPH-POC`, `MLX-ORACLE`. Direct Metal remains core; adapters cannot become default.
- CPU SIMD depends on stable default toolchain: `SIMD-TQ`. Build tags and scalar fallback keep Go 1.26 default green.
- Dense quality depends on teacher audit: `TEACHER-AUDIT`, `DENSE-PILOT`. No scale-up if exploration movement is absent.

## Benchmark And Promotion Matrix

- Training mini smoke: Linux CUDA and CPU fallback; 10 microbench reps and 3-5 end-to-end reps; track train examples/s, train pairs/s, optimizer steps/s, calls, syncs, bytes, and loadavg. Warn on more than `5%` perf regression and fail on more than `10%`.
- Real-text train: Linux CUDA and CPU fallback; same repeat counts; track train examples/s, loss parity, and backend counters. Same perf gates; quality parity required.
- Resident train safety: Linux CUDA; focused tests plus one/two-step parity; track resident refs, duplicate skip counts, fallback reasons, and poisoned-step behavior. Fail on any silent host fallback or stale host/device state.
- Apple Metal: Apple Metal and Darwin CPU fallback; cold/warm split; track device steps, host fallback, buffer residency hits/misses, and command count. Fail if claimed path lacks device counters.
- Portable backends: Vulkan, DirectML, WebGPU; cold/warm split; track host/device/fallback step counts. No performance claim until measured device execution.
- Retrieval serving: CUDA first, Metal second; 10 query microbench reps and 3-5 full dataset reps; track nDCG@10, recall@100, p50/p95/p99, bytes/vector, upload/readback bytes. Quality floors are nDCG regression <=`0.002` and recall regression <=`0.01`; p95 repeat is required for q5+q8.
- Decode: CUDA first; 10 token-step microbench reps and 3-5 E2E prompts; track compressed KV applied, dense KV materialized false, tokens/s, and memory. Fail if dense K/V is materialized in claimed compressed path.
- Energy: CUDA and Apple when telemetry is available; 3-5 E2E reps; track J/query, J/token, peak power, and unsupported telemetry status. Missing telemetry is a caveat or fail when explicitly required.
- Artifact: all promoted paths; every package records artifact bytes, vector bytes, requirement metadata, toolchain, dataset, and diff fingerprint. Fail on missing fingerprint or requirement mismatch.

Every benchmark packet records hardware, driver/toolchain, OS, Go version, artifact hash, dataset hash, git diff fingerprint, command line, environment flags, cold/warm split, loadavg, median, MAD or IQR, and raw per-rep rows.

## Descriptor-Backed Task List

### S3D-CLOSE / Close S3d Resident-Train Safety

- role/profile: `tiller-debugger` or `tiller-worker`.
- objective: finish S3d repair so resident attention/FFN download elision is safe, default-off, and incapable of host fallback after resident errors.
- context paths: `runtime/embedding_trainer.go`; `runtime/backend/backend.go`; CUDA resident-train files under `runtime/backends/cuda/`; `.tiller/scratch/codex/eos-s3d-integration-repair-v1-report.md`; Hyphae S3 lessons.
- constraints: do not enable by default; preserve duplicate-ref safety; fail closed on resident error; do not hide broader CUDA failures.
- expected outputs: focused code fixes, tests, and report with exact changed files and flags.
- verification target: focused resident train tests pass; skip counts, flush counts, and resident-gradient attribution are surfaced and verified; `git diff --check`; broader runtime CUDA failures enumerated if still present.
- budget tier/model ceiling: high, `gpt-5.5 high` if debugging; medium for bounded fixes.
- sandbox/permission needs: full local build/test; no commit.
- dependencies/blockers: CUDA device availability; explicit skip/flush counter surfacing; resident-gradient attribution.
- checkpoint criteria: focused safety slice verified and default-off.
- report contract: Outcome; files; commands/results; residual failures; checkpoint candidate; Arbiter next action.

### KERNEL-ABI-OFFLINE / Build-Time Kernel Contract And Image Loader

- role/profile: `tiller-worker` with compiler/runtime review.
- objective: derive typed CUDA/Metal launch ABIs with pure-Go GoTreeSitter,
  compile selected variants offline, and load PTX/cubin/AIR/metallib directly
  through the minimal driver adapter.
- context paths: `compiler/kernel_abi.go`; `compiler/kernel_toolchain.go`;
  `artifact/eos/module.go`; `runtime/backend/backend.go`;
  `runtime/backends/cuda/native_linux.go`; `cmd/eos/main.go`.
- constraints: preserve source-backed artifacts and NVRTC fallback; never make
  cgo/MLX the compiler or artifact authority; fail closed on ABI/hash mismatch;
  keep image format/backend semantics explicit.
- expected outputs: versioned ABI schema, offline compiler command, binary
  validation, direct CUDA module loading, and a future native MLL binary section.
- verification target: GoTreeSitter signature tests; JSON/MLL round trips;
  fake-toolchain sealing test; CUDA/Metal package tests; real PTX parity and
  p95 comparison when `nvcc` and a GPU are available.
- budget tier/model ceiling: medium-high, `gpt-5.5 medium` plus review.
- sandbox/permission needs: local compiler invocation only when explicitly
  requested; no default network or toolchain download.
- dependencies/blockers: CUDA/Metal compiler availability; driver parity;
  MLL section support for large images.
- checkpoint criteria: source-backed behavior unchanged, offline image path is
  opt-in, binary hashes and ABI fingerprints validate, and fallback is truthful.
- report contract: Outcome; exact files and commands; source/image mode;
  parity/performance evidence; caveats; checkpoint candidate; next action.

### KERNEL-DISPATCH-K1 / Typed Launch Contract And Run Accounting

- role/profile: `tiller-worker` with compiler/runtime review.
- objective: make K0's typed ABI operational at the native dispatch boundary
  and expose truthful per-run counters before reducing FFI or adding graph
  capture.
- context paths: `artifact/eos/module.go`; `compiler/kernel_abi.go`;
  `runtime/backend/native_kernel.go`; `runtime/backend/symbolic.go`;
  `runtime/backend/backend.go`.
- constraints: validate only generated launch families with a known contract;
  preserve legacy source-backed artifacts as explicitly unverified; do not
  infer device execution from backend identity; keep fallback behavior intact.
- expected outputs: ABI pointer/value kind, launch contract fingerprint,
  pre-driver validation, structured execution accounting, and focused tests.
- verification target: compiler/artifact/backend tests; mismatch rejection;
  deterministic accounting tests for device, host, and fallback paths; full
  default test/vet gates; no CUDA throughput claim without a device.
- budget tier/model ceiling: medium, `gpt-5.5 medium`.
- sandbox/permission needs: local Go tests; no network or external toolchain.
- dependencies/blockers: K0 artifact schema; CUDA/Metal hardware only for
  later launch parity and telemetry sub-gates.
- checkpoint criteria: current slice is merged locally with green default
  verification and `tmp/` excluded; generic launch bridge remains a separate
  measured gate.
- report contract: Outcome; changed paths; contract/accounting fields;
  verification; caveats; checkpoint candidate; Arbiter next action.

### OBS-EXEC / Truthful Execution Accounting

- role/profile: `tiller-worker`.
- objective: add run-level accounting for host/device/fallback steps, fallback reasons, bytes, syncs, residency hits/misses, graph captures/replays, and `full_device_execution`.
- context paths: `runtime/backend/backend.go`; `runtime/runtime.go`; `runtime/pretrained_bert_retrieval_vector_export.go`; train/eval metric writers.
- constraints: counters distinguish unavailable from zero; no optimistic device claim; keep existing metric consumers compatible.
- expected outputs: shared accounting structs, manifest/metrics fields, focused tests.
- verification target: unit tests prove fallback and device paths report distinct counters; existing metric tests updated.
- budget tier/model ceiling: medium, `gpt-5.5 medium`.
- sandbox/permission needs: local tests; no network; no commit.
- dependencies/blockers: none; should precede broad kernel expansion.
- checkpoint criteria: accounting fields available and verified without behavior change.
- report contract: Outcome; field list; tests; caveats; checkpoint candidate; next action.

### CAP-V2 / Versioned Backend Feature And Requirement Contract

- role/profile: `tiller-worker` with architect review.
- objective: add `.mll` backend requirement metadata and backend feature-version declarations with fail-closed loader checks.
- context paths: `artifact/eos`; `runtime/runtime.go`; `runtime/backend`; backend implementations under `runtime/backends/`; `docs/inspection.md`.
- constraints: backend-neutral names; no CUDA concepts in artifact semantics; host fallback allowed only when artifact/run policy permits it, even when the selected backend has `CapabilityHostFallback`.
- expected outputs: feature ids, version matcher, inspect output, tests for missing feature rejection.
- verification target: required backend/feature tests fail closed and emit actionable errors.
- budget tier/model ceiling: medium-high, `gpt-5.5 medium` plus review.
- sandbox/permission needs: local tests; no commit.
- dependencies/blockers: OBS-EXEC field vocabulary.
- checkpoint criteria: contract is inspectable and backwards-compatible for artifacts without strict requirements.
- report contract: Outcome; feature schema; changed files; tests; caveats; next action.

### RES-COORD / Resident-Step Coordinator

- role/profile: `tiller-worker`.
- objective: coordinate full training step residency across attention, FFN, activations, optimizer refs, duplicate refs, and flush/abort lifecycle.
- context paths: `runtime/embedding_trainer.go`; resident CUDA accelerators; Hyphae S3 lessons.
- constraints: `refcount==1` only for download elision; abort on resident error; default-off; deterministic counters first.
- expected outputs: step coordinator, lifecycle tests, A/B counters.
- verification target: one-step and two-step parity; duplicate-ref tests; no stale refs after abort.
- budget tier/model ceiling: medium-high.
- sandbox/permission needs: CUDA tests.
- dependencies/blockers: S3D-CLOSE; OBS-EXEC.
- checkpoint criteria: coordinator isolated behind flag and verified.
- report contract: Outcome; counters; tests; caveats; checkpoint candidate; next action.

### S3E-STEP / Full-Step Resident Gradients

- role/profile: `tiller-worker`.
- objective: keep pooled outputs and layer gradients resident through the full training step, then bridge resident gradients into optimizer updates.
- context paths: `runtime/embedding_trainer.go`; CUDA resident-train files under `runtime/backends/cuda/`; optimizer accelerator files.
- constraints: no stale host eval; no partial resident gradients after failure; host path remains complete.
- expected outputs: full-step gradient refs, optimizer bridge, parity tests, profile report.
- verification target: mini smoke and real-text A/B counters; train/eval parity; resident optimizer tests.
- budget tier/model ceiling: high.
- sandbox/permission needs: CUDA build/test and benchmark.
- dependencies/blockers: RES-COORD.
- checkpoint criteria: default-off full-step path passes correctness and counters.
- report contract: Outcome; measured vs proposed deltas; commands; caveats; next action.

### CUDA-GRAPH / Fixed-Bucket CUDA Graph Replay

- status: complete only as a bounded correctness/counter gate; no speedup claim.
- role/profile: `tiller-worker`.
- objective: capture and replay fixed-shape resident training/inference buckets after allocations and stream work are stable. **Completed for forward-only compact-train replay.**
- context paths: `runtime/backends/cuda/matmul_accel.go`; resident train stats; backend graph counters.
- constraints: forward only; exclude H2D, compact readback, backward, and
  optimizer work; require exact pointer/generation/device/context/stream keys;
  keep capture/replay default-off and preserve drained direct fallback. A
  typed launch-array option remains a separate next efficiency slice.
- expected outputs: **met** — capture cache, replay counters, fixed-bucket
  tests, exact-profile report, and lifecycle/failure repairs.
- verification target: **met for bounded canonical/next correctness and
  counters**; representative repeated end-to-end improvement of >=`10%`
  without parity drift remains the promotion gate (`5%` is diagnostic only).
- budget tier/model ceiling: medium.
- sandbox/permission needs: CUDA.
- dependencies/blockers: S3E-STEP; OBS-EXEC; K6 readback completion; stable
  device pointers. BGE/FiQA training remains blocked.
- checkpoint criteria: **met only as a bounded correctness/counter gate**;
  promotion and any speedup claim remain withheld.
- report contract: Outcome; capture shapes; counters; tests; caveats; next action.

### PACKED-FORWARD-READBACK / Narrow cgo Readback Fan-Out

- status: complete/historical (K6, 2026-08-15).
- role/profile: `tiller-worker` with CUDA runtime review.
- objective: combine compact-forward status, pooled, and active readbacks under
  one context/entry while preserving ordering and the three underlying device
  copies. **Completed by K6.**
- context paths: `runtime/backends/cuda/compact_train_accel_linux.go`;
  `runtime/backends/cuda/native_linux.go`; compact resident profile counters.
- constraints: reduce three D2H cgo entries to one context/entry; do not claim
  fewer device copies or fewer bytes; status must be checked before pooled
  results are returned; keep scalar/fallback behavior unchanged.
- expected outputs: typed narrow wrapper, counter evidence, and parity/error
  tests. Launch-array and graph capture remain separate follow-up slices.
- verification target: **met** — exact compact launch/sync/byte/fallback parity,
  one packed readback context/entry, and no stale active/pooled bucket state.
- budget tier/model ceiling: medium, `gpt-5.5 medium`.
- sandbox/permission needs: CUDA build/test; no commit.
- dependencies/blockers: K4 slab lifecycle; K5 optimizer batch; stable exact
  shape and pointer ownership.
- checkpoint criteria: **met** — deterministic cgo/context reduction is
  verified without changing device-copy counts, and graph/launch-array work
  remains unstarted.
- report contract: Outcome; before/after entries and copies; parity/tests;
  caveats; checkpoint candidate; Arbiter next action.

### METAL-RES / Direct Metal Persistent Buffers

- role/profile: `tiller-worker` on Apple host.
- objective: add persistent `MTLBuffer` residency and command batching for core Apple inference/training primitives.
- context paths: `runtime/backends/metal`; `runtime/backend`; CAP-V2 output.
- constraints: Direct Metal is Apple core; no MLX/Core ML default dependency; report host fallback honestly.
- expected outputs: buffer cache, command batching, counters, Apple benchmark packet.
- verification target: Metal beats CPU fallback on representative path and reports device/fallback counts truthfully.
- budget tier/model ceiling: medium-high.
- sandbox/permission needs: Apple Metal host.
- dependencies/blockers: CAP-V2; OBS-EXEC.
- checkpoint criteria: device-resident Metal slice verified on Apple hardware.
- report contract: Outcome; hardware; counters; tests; caveats; next action.

### MPSGRAPH-POC / Cached MPSGraph Dense Islands

- role/profile: `tiller-worker`.
- objective: test cached MPSGraph execution for dense islands after Direct Metal residency exists.
- context paths: `runtime/backends/metal`; Apple MPS/MPSGraph docs; Metal benchmark outputs.
- constraints: selective islands only; no opaque fallback claims; cache compile overhead measured.
- expected outputs: POC behind flag, A/B timing, compile/cache accounting.
- verification target: win over Direct Metal for selected island or explicit rejection report.
- budget tier/model ceiling: medium.
- sandbox/permission needs: Apple host.
- dependencies/blockers: METAL-RES.
- checkpoint criteria: only if measured win and accounting is clear.
- report contract: Outcome; island shapes; measured deltas; caveats; next action.

### MLX-ORACLE / Optional MLX-C Parity Or Import Oracle

- role/profile: `tiller-worker`.
- objective: build an optional non-default MLX-C oracle for Apple parity, import, or teacher-cache comparison.
- context paths: Apple/MLX adapter area if created; `.mll` contract docs.
- constraints: not a core backend; not in default build/runtime; no `.mll` semantics from MLX.
- expected outputs: optional adapter or rejection memo with build tags and fail-closed behavior.
- verification target: default build unaffected; oracle output compared against EOS path on fixed fixtures.
- budget tier/model ceiling: low-medium.
- sandbox/permission needs: Apple host; optional external dependency.
- dependencies/blockers: METAL-RES; MPSGRAPH-POC decision.
- checkpoint criteria: only if isolated and useful as oracle.
- report contract: Outcome; isolation proof; parity data; caveats; next action.

### SIMD-TQ / Go SIMD TurboQuant POC

- role/profile: `tiller-worker`.
- objective: prototype Go `GOEXPERIMENT=simd` helpers first in TurboQuant for prepared scoring, rotations, quant-dequant, and dots; then consider private EOS helpers.
- context paths: `go.mod`; TurboQuant repo; EOS retrieval CPU helpers; official Go 1.27 docs.
- constraints: as of 2026-08-15 Go 1.27 and its release notes remain draft/
  unreleased and the SIMD API is unstable; keep EOS `go.mod` on Go 1.26 as the
  scalar/default release baseline; use only experimental build-tagged
  microbench lanes with scalar parity/fallback; no public SIMD types, no
  production dependency before release/API stability and representative wins,
  and no default-build regression.
- expected outputs: benchmarks, build-tagged POC, ranking drift tests, recommendation.
- verification target: >=`1.2x` representative EOS CPU path or >=`1.15x` over existing TurboQuant assembly; zero allocations and no ranking drift.
- budget tier/model ceiling: medium.
- sandbox/permission needs: unreleased Go 1.27 experimental toolchain optional;
  default Go 1.26 tests.
- dependencies/blockers: official Go SIMD instability.
- checkpoint criteria: POC only if gated and isolated.
- report contract: Outcome; toolchains; perf rows; drift tests; caveats; next action.

### RETRIEVE-GPU / Device-Resident TurboQuant Retrieval

- role/profile: `tiller-worker`.
- objective: implement prepared device-resident TurboQuant index, top-k, and rerank path with delayed readback.
- context paths: retrieval files under `runtime/`; TurboQuant prepared scoring; CUDA/Metal backends.
- constraints: exact top-k parity; no host readback before final ids/scores; account upload/readback bytes.
- expected outputs: prepared index handles, device top-k tests, benchmark report.
- verification target: exact top-k parity against the same host TurboQuant prepared-scoring row; v0.2.1 three-round CUDA parity before any current-default device claim; p95 latency; bytes and readback counters. Dense retrieval quality floors are separate promotion gates and do not substitute for the parity oracle.
- budget tier/model ceiling: high.
- sandbox/permission needs: CUDA first, Apple later.
- dependencies/blockers: OBS-EXEC; CAP-V2; q5+q8 anchor.
- checkpoint criteria: candidate path verified behind flag.
- report contract: Outcome; quality table; latency table; counters; caveats; next action.

### KV-DECODE / Compressed KV And TurboSparse Decode

- role/profile: `tiller-worker`.
- objective: integrate compressed KV/TurboSparse into decode so claimed paths do not materialize dense K/V or decode on host.
- context paths: `runtime/sparse_token_pool_vector_export.go`; `runtime/retrieval_vector_export_test.go`; README KV cache surfaces; `docs/consumer-subquadratic-gpu-spec.md`.
- constraints: metadata proves `dense_kv_materialized=false` for claimed path; host reference remains diagnostic.
- expected outputs: decode step integration, metadata, parity and scaling tests.
- verification target: attention parity, memory gate, tokens/s, compressed-KV metadata.
- budget tier/model ceiling: high.
- sandbox/permission needs: CUDA.
- dependencies/blockers: CAP-V2; RETRIEVE-GPU prepared structures where shared.
- checkpoint criteria: compressed decode path verified and fail-closed.
- report contract: Outcome; memory/timing; metadata; tests; caveats; next action.

### BENCH-MATRIX / Promotion Benchmark Matrix

- role/profile: `tiller-worker`.
- objective: build reproducible benchmark matrix for Linux CUDA, Apple Metal, CPU fallback, and portable backend surfaces with cold/warm split and fingerprints.
- context paths: `docs/benchmarks.md`; benchmark scripts under `scripts/`; training/eval scripts; run manifest writers.
- constraints: 10 microbench reps; 3-5 E2E reps; median plus MAD/IQR; loadavg; hardware/toolchain/artifact/dataset/diff fingerprints.
- expected outputs: harness updates, template report, sample run.
- verification target: matrix emits machine-readable JSON/TSV and fails on missing required fingerprints.
- budget tier/model ceiling: medium.
- sandbox/permission needs: local benchmark runs; Apple/CUDA optional by lane.
- dependencies/blockers: OBS-EXEC.
- checkpoint criteria: harness and docs verified on one local backend.
- report contract: Outcome; files; sample output paths; caveats; next action.

### CI-GATE / Trunk Fold Safety

- role/profile: `tiller-worker`.
- objective: protect the current mixed S3/runtime/CUDA/documentation slice before any trunk fold or merge.
- context paths: `.github/workflows/ci.yml`; `README.md`; `docs/eos-training-inference-performance-plan.md`; runtime test files; CUDA resident-train files; runtime backend files; current mixed diff.
- constraints: no merge with red checks; `tmp/` excluded; stage only explicit intended paths in the eventual merge owner; do not include unrelated dirty work; no commit from this descriptor unless explicitly asked.
- expected outputs: CI-ready validation report, exact path list, skipped path list, and failure triage if any command fails.
- verification target: `CGO_ENABLED=0 go test ./artifact/eos ./cmd/eos ./compiler ./models ./runtime ./runtime/backend ./runtime/backends/cuda ./runtime/backends/metal ./syntax`; focused live CUDA resident tests where a CUDA device is available; GitHub Actions green before merge; `git diff --check` on explicit paths; `tmp/` excluded.
- budget tier/model ceiling: medium-high, `gpt-5.5 medium` with debugger escalation if failures are non-local.
- sandbox/permission needs: local Go test execution; optional CUDA device; GitHub Actions visibility if remote CI is involved; no staging/commit by default.
- dependencies/blockers: S3D-CLOSE status; OBS-EXEC field changes if included; availability of CUDA host for live checks.
- checkpoint criteria: all required local checks and GitHub Actions are green, or failures are explicitly attributed and merge is blocked.
- report contract: Outcome; exact commands/results; path list; excluded paths; CI status; caveats; checkpoint candidate; next action.

### DOC-HYGIENE / Benchmark Wording Hygiene

- role/profile: `tiller-worker`.
- objective: keep benchmark wording synchronized with current evidence so historical pre-S3 numbers are not described as current throughput.
- context paths: `README.md`; `docs/eos-training-inference-performance-plan.md`; `docs/benchmarks.md`; Hyphae S3 evidence.
- constraints: documentation-only; no benchmark invention; label stale baselines explicitly; do not edit source or tests.
- expected outputs: narrow wording patches and lint/format results.
- verification target: `rg` finds no remaining claim that `845.15` / `865437.87` is current post-S3 throughput; mdpp lint/fmt pass for edited Markdown; `git diff --check` on edited docs.
- budget tier/model ceiling: low-medium.
- sandbox/permission needs: read/write docs only; no staging/commit.
- dependencies/blockers: current S3 benchmark packet availability for any replacement numbers.
- checkpoint criteria: stale wording corrected without changing measured values.
- report contract: Outcome; edited lines; verification commands/results; caveats; next action.

### TEACHER-AUDIT / Provenance-Safe Teacher-Cache Audit

- role/profile: `tiller-worker`.
- objective: complete Qwen3/mxbai teacher-cache scoring, agreement, margin, and leak filtering before dense training.
- context paths: `docs/eos-distillation-compact-default-spec.md`; `docs/production-embedding.md`; teacher cache scratch reports.
- constraints: no hosted live calls in training; no MS MARCO release claim; train/dev/test boundaries explicit; leak rows untrainable.
- expected outputs: manifests, agreement/margin reports, rejected-row report, dataset SHA256s.
- verification target: manifests parse; qrels and exact-text leak checks pass; teacher agreement stats emitted.
- budget tier/model ceiling: medium.
- sandbox/permission needs: local data/cache access; network only if explicitly acquiring public data.
- dependencies/blockers: dataset license/provenance review.
- checkpoint criteria: audit artifacts complete; no training yet.
- report contract: Outcome; counts; paths; caveats; next action.

### DENSE-PILOT / Bounded Dense Teacher/Data Pilot

- role/profile: `tiller-worker`.
- objective: run one bounded dense pilot after teacher audit to test retrieval movement before scaling.
- context paths: `docs/eos-distillation-compact-default-spec.md`; training scripts; Phase 0 anchors.
- constraints: teacher/data signal before bigger models; no compact rescue; MS MARCO research-only unless license policy changes; no scale-up without movement.
- expected outputs: candidate package, train/eval metrics, dense scoreboards, error analysis.
- verification target: exploration macro nDCG `>= +0.001`, macro recall `>= -0.001`, no dataset nDCG `< -0.002`, and no dataset recall `< -0.003`; promotion target `+0.010` with accepted release floors.
- budget tier/model ceiling: high for run/debug loop.
- sandbox/permission needs: training hardware and datasets.
- dependencies/blockers: TEACHER-AUDIT; BENCH-MATRIX quality harness.
- checkpoint criteria: pilot report complete, regardless pass/fail.
- report contract: Outcome; scoreboards; gates; caveats; next action.

### Q5Q8-PROMOTE / q5+q8 Serving Promotion Candidate

- role/profile: `tiller-worker`.
- objective: repeat and gate q5+q8 as replacement candidate for q4+fp16/o200.
- context paths: `docs/anchors/eos-mixed-rerank-eval-v021/mixed-rerank-summary.md`; retrieval files under `runtime/`; `docs/production-embedding.md`.
- constraints: candidate only; q4+fp16 remains promoted until repeat p95 and quality pass; does not count as dense quality evidence.
- expected outputs: repeat full uncapped scoreboards, p95 latency, per-query diagnostics fixed or caveated.
- verification target: 6/6 quality pass; p95 repeat at FiQA-or-larger scale; `432 B/vector`, `2.37x` accounting.
- budget tier/model ceiling: medium.
- sandbox/permission needs: CUDA preferred.
- dependencies/blockers: RETRIEVE-GPU optional but not required for host rerank repeat.
- checkpoint criteria: promotion packet or explicit rejection.
- report contract: Outcome; quality/latency table; caveats; next action.

## Risks And Open Decisions

- Wall-clock noise on shared host: use deterministic counters first; require quiet-host timing and loadavg for 5-10% claims.
- K6 D2H fault injection: status-first early stop and stage-progress cleanup are
  covered, but deterministic injected failures at pooled/active copy points
  remain residual test-hardening work.
- S3d residual validation risks: explicit skip counts, flush counts, resident-gradient attribution, and exact corpus provenance must be surfaced before the next resident-step coordinator claim.
- Backend name overclaims device execution: CAP-V2 and OBS-EXEC make device/fallback truth inspectable.
- Apple framework temptation: Direct Metal remains core; MPSGraph selective; MLX/Core ML/BNNS adapters do not own `.mll`.
- Go SIMD instability: as of 2026-08-15 Go 1.27 is unreleased and its SIMD
  API is unstable; keep Go 1.26 scalar/default, and allow only build-tagged
  microbench lanes with scalar parity/fallback until release/API stability and
  representative wins.
- Dense-quality false positives: retrieval scoreboards decide; internal AUC/margin/top1 are diagnostics only.
- Compact serving confusion: q5+q8 is serving evidence, not dense model evidence.
- MS MARCO licensing: research-only unless acquisition manifest and policy explicitly allow release use.
- Portable backend claims: Vulkan/DirectML/WebGPU remain surfaces until measured device paths exist.

## Immediate Next Action

The old K1-first action and packed-forward readback are historical; K4, K5,
K6, and the bounded K7 CUDA-Graph correctness/counter gate are now verified.
The next bounded efficiency slice is an investigation, not an implementation
claim: prefer a measured typed launch-array, kernel-fusion/coalescing, or
resident-transfer-reduction path after selecting a stable bottleneck. Keep new
work default-off, expose truthful launch/transfer telemetry, require hard
parity, and require a representative repeated end-to-end win of at least `10%`
before promotion (`5%` is diagnostic only). Do not start training for this
slice.

The next controlled encoder-v2.1 training run remains blocked by the selected-
package BGE FiQA export and related pretrained/export jobs. Hold it and do not
start it until `.tiller/scratch/codex/encoder-v21-controlled-training-ready-descriptor-v1.md`
launch conditions are satisfied. Go SIMD work remains
experimental build-tagged microbench coverage only; the Go 1.26 scalar/default
baseline stays release-facing.
