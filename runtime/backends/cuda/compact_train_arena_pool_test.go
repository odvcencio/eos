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
