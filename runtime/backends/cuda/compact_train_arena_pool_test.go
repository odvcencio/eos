//go:build linux && cgo

package cuda

import (
	"testing"

	"m31labs.dev/eos/runtime/backend"
)

func TestCompactTrainArenaPoolReusesExactShape(t *testing.T) {
	shape := backend.CompactForwardShape{
		Batch: 1, Tokens: 3, ModelDim: 4, FFNDim: 8,
		Heads: 2, HeadDim: 2, Layers: 1, OutputDim: 4,
	}
	arena := &compactTrainArena{shape: shape, bytes: 128, workspaceBytes: 64}
	accel := &CompactTrainAccelerator{
		arenaPool: map[backend.CompactForwardShape][]*compactTrainArena{},
	}

	accel.recycleArenaLocked(arena)
	got, err := accel.prepareArenaLocked(shape)
	if err != nil {
		t.Fatalf("take pooled arena: %v", err)
	}
	if got != arena {
		t.Fatalf("pooled arena pointer changed: got %p want %p", got, arena)
	}
	if accel.stats.ArenaReuseHits != 1 || accel.stats.ArenaAllocations != 0 {
		t.Fatalf("unexpected reuse stats: %+v", accel.stats)
	}
	if len(accel.arenaPool[shape]) != 0 {
		t.Fatalf("pooled arena was not removed after take")
	}
}

func TestCompactTrainArenaPoolResetOnRecycle(t *testing.T) {
	shape := backend.CompactForwardShape{Batch: 1, Tokens: 1, ModelDim: 2, FFNDim: 2, Heads: 1, HeadDim: 2, Layers: 1, OutputDim: 2}
	token := &compactTrainHandleToken{}
	arena := &compactTrainArena{
		shape:      shape,
		id:         9,
		generation: 11,
		token:      token,
		live:       true,
		geluFast:   true,
	}
	accel := &CompactTrainAccelerator{}

	accel.recycleArenaLocked(arena)
	if arena.id != 0 || arena.generation != 0 || arena.token != nil || arena.live || arena.geluFast {
		t.Fatalf("recycle did not clear transient arena state: %+v", arena)
	}
	if len(accel.arenaPool[shape]) != 1 {
		t.Fatalf("recycled arena missing from exact-shape pool")
	}
}

func TestCompactTrainGradientBuffersWarmAfterTwoSteps(t *testing.T) {
	accel, cleanup := newBoundCompactTrainTestAccelerator(t, false, false)
	defer cleanup()
	shape := backend.CompactForwardShape{Batch: 1, Tokens: 2, ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 2, OutputDim: 4}
	refs := compactTrainResidentRefsForTest(t, accel, shape)
	for step := uint64(1); step <= 3; step++ {
		if err := accel.BeginCompactTrainStep(step, refs); err != nil {
			t.Fatalf("begin step %d: %v", step, err)
		}
		if err := accel.EndCompactTrainStep(step); err != nil {
			t.Fatalf("end step %d: %v", step, err)
		}
		if err := accel.ReleaseCompactTrainGradients(step); err != nil {
			t.Fatalf("release gradients step %d: %v", step, err)
		}
	}
	stats := accel.CompactTrainStats()
	wantRefs := int64(len(refs))
	if stats.GradientAllocations != wantRefs || stats.GradientReuseHits != 2*wantRefs {
		t.Fatalf("gradient allocations/reuse = %d/%d, want %d/%d: %+v", stats.GradientAllocations, stats.GradientReuseHits, wantRefs, 2*wantRefs, stats)
	}
	if stats.GradientZeroCalls != 3 {
		t.Fatalf("gradient zero calls = %d, want 3", stats.GradientZeroCalls)
	}
	if err := accel.AbortCompactTrainStep(3); err != nil {
		t.Fatalf("abort warm gradient step: %v", err)
	}
}

func TestFlattenInt32IntoReusesCapacity(t *testing.T) {
	first := flattenInt32Into(make([]int32, 0, 8), [][]int32{{1, 2}, {3, 4}})
	if len(first) != 4 || first[0] != 1 || first[3] != 4 {
		t.Fatalf("first flatten = %v, want [1 2 3 4]", first)
	}
	addr := &first[0]
	second := flattenInt32Into(first, [][]int32{{5}, {6, 7}})
	if &second[0] != addr {
		t.Fatal("flattened staging did not reuse backing storage")
	}
	if len(second) != 3 || second[0] != 5 || second[2] != 7 {
		t.Fatalf("second flatten = %v, want [5 6 7]", second)
	}
}

func BenchmarkCompactTrainArenaPoolHit(b *testing.B) {
	shape := backend.CompactForwardShape{Batch: 1, Tokens: 3, ModelDim: 4, FFNDim: 8, Heads: 2, HeadDim: 2, Layers: 1, OutputDim: 4}
	accel := &CompactTrainAccelerator{
		arenaPool: map[backend.CompactForwardShape][]*compactTrainArena{},
	}
	arena := &compactTrainArena{shape: shape}
	accel.recycleArenaLocked(arena)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		accel.recycleArenaLocked(arena)
		if _, err := accel.prepareArenaLocked(shape); err != nil {
			b.Fatal(err)
		}
	}
}
