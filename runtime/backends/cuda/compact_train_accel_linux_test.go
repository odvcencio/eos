//go:build linux && cgo

package cuda

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"m31labs.dev/eos/runtime/backend"
)

func TestCompactTrainForwardPooledParityAndAccounting(t *testing.T) {
	for _, tc := range []struct {
		name       string
		rope       bool
		projection bool
		gelu       string
		tokens     [][]int32
		masks      [][]int32
		roles      []int32
	}{
		{
			name:   "exact_rope_no_projection",
			rope:   true,
			gelu:   backend.CompactForwardGELUExact,
			tokens: [][]int32{{2, 1, 3}, {1, 4, 2}},
			masks:  [][]int32{{1, 1, 1}, {1, 0, 1}},
			roles:  []int32{0, 2},
		},
		{
			name:       "fast_projection",
			projection: true,
			gelu:       backend.CompactForwardGELUFast,
			tokens:     [][]int32{{2, 1, 3}},
			masks:      [][]int32{{1, 1, 0}},
			roles:      []int32{1},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			accel, cleanup := newBoundCompactTrainTestAccelerator(t, tc.rope, tc.projection)
			defer cleanup()
			shape := backend.CompactForwardShape{Batch: len(tc.tokens), Tokens: len(tc.tokens[0]), ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 2, OutputDim: 4}
			if tc.projection {
				shape.OutputDim = 3
				shape.HasOutputProjection = true
			}
			refs := compactTrainResidentRefsForTest(t, accel, shape)
			if err := accel.BeginCompactTrainStep(7, refs); err != nil {
				t.Fatalf("begin step: %v", err)
			}
			defer func() {
				if err := accel.EndCompactTrainStep(7); err != nil {
					t.Fatalf("end step: %v", err)
				}
			}()
			req := backend.CompactTrainForwardRequest{
				Shape:        shape,
				Tokens:       tc.tokens,
				Masks:        tc.masks,
				Roles:        tc.roles,
				ResidentRefs: refs,
				GELUMode:     tc.gelu,
				StepID:       7,
			}
			got, err := accel.RunCompactTrainForward(req)
			if err != nil {
				t.Fatalf("compact train forward: %v", err)
			}
			defer func() {
				if err := accel.ReleaseCompactTrainHandle(got.Handle); err != nil {
					t.Fatalf("release handle: %v", err)
				}
			}()
			packedReq := backend.CompactForwardRequest{
				Shape:        req.Shape,
				Tokens:       req.Tokens,
				Masks:        req.Masks,
				Roles:        req.Roles,
				ResidentRefs: req.ResidentRefs,
				GELUMode:     req.GELUMode,
			}
			wantPacked := hostCompactForwardForCUDATest(packedReq, tc.rope, tc.projection)
			wantPooled := make([]float32, 0, shape.Batch*shape.OutputDim)
			for b := 0; b < shape.Batch; b++ {
				span := compactForwardSpanByName(wantPacked.Layout, compactForwardSequenceSpanName(b, "final.pooled"))
				wantPooled = append(wantPooled, wantPacked.Data[span.Offset:span.Offset+span.Len]...)
			}
			if got.Pooled == nil || len(got.Pooled.Shape) != 2 || got.Pooled.Shape[0] != shape.Batch || got.Pooled.Shape[1] != shape.OutputDim {
				t.Fatalf("pooled shape = %+v", got.Pooled)
			}
			assertFloatSlicesClose(t, got.Pooled.F32, wantPooled, 1e-6)
			for i, mask := range tc.masks {
				if got.ActiveCounts[i] != int32(activeHost(mask)) {
					t.Fatalf("active[%d] = %d, want %d", i, got.ActiveCounts[i], activeHost(mask))
				}
			}
			stats := accel.CompactTrainStats()
			_, packedFloats, _ := compactForwardPackedLayout(shape)
			if stats.ForwardCalls != 1 || stats.HandlesCreated != 1 || stats.LiveHandles != 1 {
				t.Fatalf("forward/handle stats = %+v", stats)
			}
			if stats.PooledDownloadedBytes != int64(shape.Batch*shape.OutputDim*4) {
				t.Fatalf("pooled downloaded bytes = %d, want %d", stats.PooledDownloadedBytes, shape.Batch*shape.OutputDim*4)
			}
			if stats.StatusDownloadedBytes != int64(4+shape.Batch*4) {
				t.Fatalf("status/active bytes = %d, want %d", stats.StatusDownloadedBytes, 4+shape.Batch*4)
			}
			if accel.arena == nil {
				t.Fatal("compact train arena missing after forward")
			}
			if tc.projection {
				if accel.arena.preProjectionPooled == 0 {
					t.Fatal("projected compact train arena missing pre-projection pooled scratch")
				}
				if accel.arena.preProjectionPooled == accel.arena.finalPooled {
					t.Fatal("projected compact train arena aliased pre-projection pooled scratch with final pooled")
				}
			} else if accel.arena.preProjectionPooled != 0 {
				t.Fatalf("no-projection compact train arena pre-projection pooled scratch = %#x, want 0", uintptr(accel.arena.preProjectionPooled))
			}
			if stats.ActivationArenaBytes != compactTrainExpectedArenaBytes(shape) {
				t.Fatalf("activation arena bytes = %d, want %d", stats.ActivationArenaBytes, compactTrainExpectedArenaBytes(shape))
			}
			if stats.PackedBytesAvoided != int64(packedFloats*4) {
				t.Fatalf("packed bytes avoided = %d, want %d", stats.PackedBytesAvoided, packedFloats*4)
			}
			if base := accel.CompactForwardAccelerator.Stats(); base.PackedDownloads != 0 || base.PackedBytes != 0 {
				t.Fatalf("packed stats changed on compact train path: %+v", base)
			}
		})
	}
}

func TestCompactTrainForwardActualShapePooledOnlySmoke(t *testing.T) {
	shape := backend.CompactForwardShape{Batch: 1, Tokens: 75, ModelDim: 128, FFNDim: 512, Heads: 4, HeadDim: 32, Layers: 2, OutputDim: 128}
	weights := compactTrainShapeTestWeights(shape, false)
	accel, cleanup := newBoundCompactTrainShapeTestAccelerator(t, shape, false, false, weights)
	defer cleanup()
	tokens := make([]int32, shape.Tokens)
	masks := make([]int32, shape.Tokens)
	for i := range tokens {
		tokens[i] = int32(i%5 + 1)
		masks[i] = 1
	}
	req := backend.CompactTrainForwardRequest{
		Shape:        shape,
		Tokens:       [][]int32{tokens},
		Masks:        [][]int32{masks},
		Roles:        []int32{0},
		ResidentRefs: nil,
		GELUMode:     backend.CompactForwardGELUFast,
		StepID:       11,
	}
	refs := compactTrainResidentRefsForTest(t, accel, shape)
	req.ResidentRefs = refs
	if err := accel.BeginCompactTrainStep(11, refs); err != nil {
		t.Fatalf("begin step: %v", err)
	}
	got, err := accel.RunCompactTrainForward(req)
	if err != nil {
		t.Fatalf("compact train actual-shape forward: %v", err)
	}
	wantPooled := hostCompactTrainPooledForCUDATest(req, weights)
	maxAbs := compactTrainMaxAbs(got.Pooled.F32, wantPooled)
	t.Logf("actual-shape compact train pooled parity max_abs=%.8g", maxAbs)
	assertFloatSlicesClose(t, got.Pooled.F32, wantPooled, 1e-6)
	if len(got.ActiveCounts) != 1 || got.ActiveCounts[0] != int32(shape.Tokens) {
		t.Fatalf("active counts = %+v, want [%d]", got.ActiveCounts, shape.Tokens)
	}
	stats := accel.CompactTrainStats()
	_, packedFloats, _ := compactForwardPackedLayout(shape)
	if stats.PooledDownloadedBytes != int64(shape.Batch*shape.OutputDim*4) {
		t.Fatalf("pooled downloaded bytes = %d, want %d", stats.PooledDownloadedBytes, shape.Batch*shape.OutputDim*4)
	}
	if stats.StatusDownloadedBytes != int64(4+shape.Batch*4) {
		t.Fatalf("status/active bytes = %d, want %d", stats.StatusDownloadedBytes, 4+shape.Batch*4)
	}
	if stats.ActivationArenaBytes != compactTrainExpectedArenaBytes(shape) {
		t.Fatalf("activation arena bytes = %d, want %d", stats.ActivationArenaBytes, compactTrainExpectedArenaBytes(shape))
	}
	if stats.PackedBytesAvoided != int64(packedFloats*4) {
		t.Fatalf("packed bytes avoided = %d, want %d", stats.PackedBytesAvoided, packedFloats*4)
	}
	if base := accel.CompactForwardAccelerator.Stats(); base.PackedDownloads != 0 || base.PackedBytes != 0 {
		t.Fatalf("packed stats changed on compact train path: %+v", base)
	}
	if err := accel.ReleaseCompactTrainHandle(got.Handle); err != nil {
		t.Fatalf("release actual-shape handle: %v", err)
	}
	if err := accel.EndCompactTrainStep(11); err != nil {
		t.Fatalf("end step: %v", err)
	}
	stats = accel.CompactTrainStats()
	if stats.LiveHandles != 0 || stats.HandlesCreated != 1 || stats.HandlesReleased != 1 {
		t.Fatalf("post-release handle stats = %+v", stats)
	}
}

func TestCompactTrainForwardProjectedWideOutputDimParityAndAccounting(t *testing.T) {
	shape := backend.CompactForwardShape{Batch: 1, Tokens: 3, ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 2, OutputDim: 6, HasOutputProjection: true}
	weights := compactTrainShapeTestWeights(shape, true)
	accel, cleanup := newBoundCompactTrainShapeTestAccelerator(t, shape, false, true, weights)
	defer cleanup()
	refs := compactTrainResidentRefsForTest(t, accel, shape)
	if err := accel.BeginCompactTrainStep(13, refs); err != nil {
		t.Fatalf("begin step: %v", err)
	}
	req := backend.CompactTrainForwardRequest{
		Shape:        shape,
		Tokens:       [][]int32{{2, 1, 3}},
		Masks:        [][]int32{{1, 0, 1}},
		Roles:        []int32{2},
		ResidentRefs: refs,
		GELUMode:     backend.CompactForwardGELUFast,
		StepID:       13,
	}
	got, err := accel.RunCompactTrainForward(req)
	if err != nil {
		t.Fatalf("compact train projected wide forward: %v", err)
	}
	wantPooled := hostCompactTrainPooledForCUDATest(req, weights)
	if got.Pooled == nil || len(got.Pooled.Shape) != 2 || got.Pooled.Shape[0] != shape.Batch || got.Pooled.Shape[1] != shape.OutputDim {
		t.Fatalf("pooled shape = %+v", got.Pooled)
	}
	if len(got.Pooled.F32) != shape.Batch*shape.OutputDim {
		t.Fatalf("pooled data len = %d, want %d", len(got.Pooled.F32), shape.Batch*shape.OutputDim)
	}
	maxAbs := compactTrainMaxAbs(got.Pooled.F32, wantPooled)
	t.Logf("projected O>D compact train pooled parity max_abs=%.8g", maxAbs)
	assertFloatSlicesClose(t, got.Pooled.F32, wantPooled, 1e-6)
	if len(got.ActiveCounts) != 1 || got.ActiveCounts[0] != 2 {
		t.Fatalf("active counts = %+v, want [2]", got.ActiveCounts)
	}
	stats := accel.CompactTrainStats()
	_, packedFloats, _ := compactForwardPackedLayout(shape)
	if stats.ForwardCalls != 1 || stats.HandlesCreated != 1 || stats.LiveHandles != 1 {
		t.Fatalf("forward/handle stats = %+v", stats)
	}
	if stats.PooledDownloadedBytes != int64(shape.Batch*shape.OutputDim*4) {
		t.Fatalf("pooled downloaded bytes = %d, want %d", stats.PooledDownloadedBytes, shape.Batch*shape.OutputDim*4)
	}
	if stats.StatusDownloadedBytes != int64(4+shape.Batch*4) {
		t.Fatalf("status/active bytes = %d, want %d", stats.StatusDownloadedBytes, 4+shape.Batch*4)
	}
	if accel.arena == nil {
		t.Fatal("compact train arena missing after projected wide forward")
	}
	if accel.arena.preProjectionPooled == 0 {
		t.Fatal("projected wide compact train arena missing pre-projection pooled scratch")
	}
	if accel.arena.preProjectionPooled == accel.arena.finalPooled {
		t.Fatal("projected wide compact train arena aliased pre-projection pooled scratch with final pooled")
	}
	if stats.ActivationArenaBytes != compactTrainExpectedArenaBytes(shape) {
		t.Fatalf("activation arena bytes = %d, want %d", stats.ActivationArenaBytes, compactTrainExpectedArenaBytes(shape))
	}
	if stats.PackedBytesAvoided != int64(packedFloats*4) {
		t.Fatalf("packed bytes avoided = %d, want %d", stats.PackedBytesAvoided, packedFloats*4)
	}
	if base := accel.CompactForwardAccelerator.Stats(); base.PackedDownloads != 0 || base.PackedBytes != 0 {
		t.Fatalf("packed stats changed on compact train path: %+v", base)
	}
	if err := accel.ReleaseCompactTrainHandle(got.Handle); err != nil {
		t.Fatalf("release projected wide handle: %v", err)
	}
	if err := accel.EndCompactTrainStep(13); err != nil {
		t.Fatalf("end step: %v", err)
	}
	stats = accel.CompactTrainStats()
	if stats.LiveHandles != 0 || stats.HandlesCreated != 1 || stats.HandlesReleased != 1 {
		t.Fatalf("post-release handle stats = %+v", stats)
	}
}

func TestCompactTrainForwardHandleLifecycle(t *testing.T) {
	accel, cleanup := newBoundCompactTrainTestAccelerator(t, true, false)
	defer cleanup()
	shape := backend.CompactForwardShape{Batch: 1, Tokens: 3, ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 2, OutputDim: 4}
	refs := compactTrainResidentRefsForTest(t, accel, shape)
	if err := accel.BeginCompactTrainStep(17, refs); err != nil {
		t.Fatalf("begin step: %v", err)
	}
	req := backend.CompactTrainForwardRequest{
		Shape:        shape,
		Tokens:       [][]int32{{2, 1, 3}},
		Masks:        [][]int32{{1, 1, 1}},
		Roles:        []int32{0},
		ResidentRefs: refs,
		GELUMode:     backend.CompactForwardGELUExact,
		StepID:       17,
	}
	first, err := accel.RunCompactTrainForward(req)
	if err != nil {
		t.Fatalf("first forward: %v", err)
	}
	if _, err := accel.RunCompactTrainForward(req); err == nil || !strings.Contains(err.Error(), "live handle") {
		t.Fatalf("second forward with live handle err = %v, want live handle", err)
	}
	if !first.Handle.Token.Alive() {
		t.Fatal("handle token not alive after forward")
	}
	if err := accel.ReleaseCompactTrainHandle(first.Handle); err != nil {
		t.Fatalf("first release: %v", err)
	}
	if first.Handle.Token.Alive() {
		t.Fatal("handle token alive after release")
	}
	if err := accel.ReleaseCompactTrainHandle(first.Handle); err == nil || !strings.Contains(err.Error(), "already released") {
		t.Fatalf("double release err = %v, want already released", err)
	}
	second, err := accel.RunCompactTrainForward(req)
	if err != nil {
		t.Fatalf("second forward after release: %v", err)
	}
	if second.Handle.Generation == first.Handle.Generation {
		t.Fatalf("generation reused: %d", second.Handle.Generation)
	}
	if err := accel.ReleaseCompactTrainHandle(first.Handle); err == nil || !strings.Contains(err.Error(), "stale") {
		t.Fatalf("stale release err = %v, want stale", err)
	}
	if err := accel.ReleaseCompactTrainHandle(second.Handle); err != nil {
		t.Fatalf("second release: %v", err)
	}
	if err := accel.EndCompactTrainStep(17); err != nil {
		t.Fatalf("end step: %v", err)
	}
	if stats := accel.CompactTrainStats(); stats.LiveHandles != 0 || stats.HandlesCreated != 2 || stats.HandlesReleased != 2 {
		t.Fatalf("lifecycle stats = %+v", stats)
	}
}

func TestCompactTrainStepLifecycleValidation(t *testing.T) {
	accel, cleanup := newBoundCompactTrainTestAccelerator(t, true, false)
	defer cleanup()
	shape := backend.CompactForwardShape{Batch: 1, Tokens: 3, ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 2, OutputDim: 4}
	refs := compactTrainResidentRefsForTest(t, accel, shape)
	req := backend.CompactTrainForwardRequest{
		Shape:        shape,
		Tokens:       [][]int32{{2, 1, 3}},
		Masks:        [][]int32{{1, 1, 1}},
		Roles:        []int32{0},
		ResidentRefs: refs,
		GELUMode:     backend.CompactForwardGELUExact,
		StepID:       31,
	}
	if err := accel.PreflightCompactTrainForward(req); err != nil {
		t.Fatalf("preflight before begin: %v", err)
	}
	if _, err := accel.RunCompactTrainForward(req); err == nil || !strings.Contains(err.Error(), "step is not active") {
		t.Fatalf("forward before begin err = %v, want inactive step", err)
	}
	if stats := accel.CompactTrainStats(); stats != (backend.CompactTrainAcceleratorStats{}) {
		t.Fatalf("stats after rejected pre-begin calls = %+v, want zero", stats)
	}
	if accel.arena != nil || len(accel.grads) != 0 {
		t.Fatalf("state mutated before begin: arena=%v grads=%d", accel.arena != nil, len(accel.grads))
	}
	zeroStats := accel.CompactTrainStats()
	incompleteRefs := append([]backend.CompactForwardResidentRef(nil), refs[1:]...)
	if err := accel.BeginCompactTrainStep(30, incompleteRefs); err == nil || !strings.Contains(err.Error(), "missing") {
		t.Fatalf("incomplete begin err = %v, want missing", err)
	}
	if accel.stepActive || accel.stepID != 0 || accel.gradGen != 0 || len(accel.grads) != 0 {
		t.Fatalf("incomplete begin mutated step state: active=%v step=%d gradGen=%d grads=%d", accel.stepActive, accel.stepID, accel.gradGen, len(accel.grads))
	}
	assertCompactTrainStatsUnchanged(t, zeroStats, accel.CompactTrainStats(), "incomplete begin")

	if err := accel.BeginCompactTrainStep(31, refs); err != nil {
		t.Fatalf("begin step: %v", err)
	}
	beginStats := accel.CompactTrainStats()
	if err := accel.BeginCompactTrainStep(32, nil); err == nil || !strings.Contains(err.Error(), "already active") {
		t.Fatalf("nested begin with zero refs err = %v, want already active", err)
	}
	assertCompactTrainStatsUnchanged(t, beginStats, accel.CompactTrainStats(), "nested begin before handle")
	wrong := req
	wrong.StepID = 30
	if err := accel.PreflightCompactTrainForward(wrong); err != nil {
		t.Fatalf("preflight wrong step: %v", err)
	}
	assertCompactTrainStatsUnchanged(t, beginStats, accel.CompactTrainStats(), "wrong-step preflight")
	if _, err := accel.RunCompactTrainForward(wrong); err == nil || !strings.Contains(err.Error(), "stale") {
		t.Fatalf("forward wrong step err = %v, want stale", err)
	}
	assertCompactTrainStatsUnchanged(t, beginStats, accel.CompactTrainStats(), "wrong-step forward")
	if accel.arena != nil {
		t.Fatal("wrong-step forward allocated arena")
	}

	forward, err := accel.RunCompactTrainForward(req)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if forward.Handle.StepID != 31 || forward.Handle.Token.StepID() != 31 {
		t.Fatalf("handle step = %d token step = %d, want 31", forward.Handle.StepID, forward.Handle.Token.StepID())
	}
	liveStats := accel.CompactTrainStats()
	wrongHandle := forward.Handle
	wrongHandle.StepID = 30
	if _, err := accel.RunCompactTrainBackward(backend.CompactTrainBackwardRequest{Handle: wrongHandle, GradPooled: backend.NewTensorF32([]int{1, 4}, make([]float32, 4))}); err == nil || !strings.Contains(err.Error(), "stale") {
		t.Fatalf("backward wrong-step handle err = %v, want stale", err)
	}
	assertCompactTrainStatsUnchanged(t, liveStats, accel.CompactTrainStats(), "wrong-step backward")
	if !forward.Handle.Token.Alive() {
		t.Fatal("wrong-step backward consumed handle")
	}
	if err := accel.ReleaseCompactTrainHandle(wrongHandle); err == nil || !strings.Contains(err.Error(), "stale") {
		t.Fatalf("release wrong-step handle err = %v, want stale", err)
	}
	assertCompactTrainStatsUnchanged(t, liveStats, accel.CompactTrainStats(), "wrong-step release")
	if !forward.Handle.Token.Alive() {
		t.Fatal("wrong-step release consumed handle")
	}
	if err := accel.ReleaseCompactTrainHandle(forward.Handle); err != nil {
		t.Fatalf("release: %v", err)
	}
	releasedStats := accel.CompactTrainStats()
	if err := accel.BeginCompactTrainStep(32, refs); err == nil || !strings.Contains(err.Error(), "already active") {
		t.Fatalf("nested begin after release err = %v, want already active", err)
	}
	assertCompactTrainStatsUnchanged(t, releasedStats, accel.CompactTrainStats(), "nested begin after release")
	if err := accel.EndCompactTrainStep(30); err == nil || !strings.Contains(err.Error(), "stale") {
		t.Fatalf("wrong end err = %v, want stale", err)
	}
	assertCompactTrainStatsUnchanged(t, releasedStats, accel.CompactTrainStats(), "wrong-step end")
	if err := accel.BeginCompactTrainStep(32, refs); err == nil || !strings.Contains(err.Error(), "already active") {
		t.Fatalf("begin after wrong end err = %v, want already active", err)
	}
	if err := accel.EndCompactTrainStep(31); err != nil {
		t.Fatalf("exact end: %v", err)
	}
	if err := accel.EndCompactTrainStep(31); err == nil || !strings.Contains(err.Error(), "not active") {
		t.Fatalf("repeat end err = %v, want not active", err)
	}
	endedGradGen, endedStepID := accel.gradGen, accel.stepID
	endedStats := accel.CompactTrainStats()
	if err := accel.BeginCompactTrainStep(99, nil); err == nil || !strings.Contains(err.Error(), "resident refs are required") {
		t.Fatalf("invalid inactive begin err = %v, want resident refs required", err)
	}
	if accel.stepActive || accel.stepID != endedStepID || accel.gradGen != endedGradGen {
		t.Fatalf("invalid inactive begin mutated step state: active=%v step=%d/%d gradGen=%d/%d", accel.stepActive, accel.stepID, endedStepID, accel.gradGen, endedGradGen)
	}
	assertCompactTrainStatsUnchanged(t, endedStats, accel.CompactTrainStats(), "invalid inactive begin")
	if err := accel.BeginCompactTrainStep(32, refs); err != nil {
		t.Fatalf("begin next step: %v", err)
	}
	if err := accel.ReleaseCompactTrainHandle(forward.Handle); err == nil || !strings.Contains(err.Error(), "stale") {
		t.Fatalf("old-step handle release err = %v, want stale", err)
	}
	if err := accel.EndCompactTrainStep(32); err != nil {
		t.Fatalf("end next step: %v", err)
	}
}

func TestCompactTrainForwardExactCurrentGradSetPreMutation(t *testing.T) {
	projectionShape := backend.CompactForwardShape{Batch: 1, Tokens: 2, ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 2, OutputDim: 3, HasOutputProjection: true}
	accel, cleanup := newBoundCompactTrainShapeTestAccelerator(t, projectionShape, false, true, compactForwardTestWeights(true))
	defer cleanup()
	projectionRefs := compactTrainResidentRefsForTest(t, accel, projectionShape)
	if err := accel.BeginCompactTrainStep(33, projectionRefs); err != nil {
		t.Fatalf("begin projection step: %v", err)
	}
	noProjectionShape := backend.CompactForwardShape{Batch: 1, Tokens: 2, ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 2, OutputDim: 4}
	noProjectionRefs := compactTrainResidentRefsForTest(t, accel, noProjectionShape)
	req := backend.CompactTrainForwardRequest{
		Shape:        noProjectionShape,
		Tokens:       [][]int32{{2, 1}},
		Masks:        [][]int32{{1, 1}},
		Roles:        []int32{0},
		ResidentRefs: noProjectionRefs,
		GELUMode:     backend.CompactForwardGELUExact,
		StepID:       33,
	}
	beginStats := accel.CompactTrainStats()
	if _, err := accel.RunCompactTrainForward(req); err == nil || !strings.Contains(err.Error(), "unexpected") {
		t.Fatalf("forward with mismatched current grad set err = %v, want unexpected", err)
	}
	assertCompactTrainStatsUnchanged(t, beginStats, accel.CompactTrainStats(), "mismatched current grad set forward")
	if accel.arena != nil {
		t.Fatal("mismatched current grad set allocated arena")
	}
	if err := accel.EndCompactTrainStep(33); err != nil {
		t.Fatalf("end projection step: %v", err)
	}
}

func TestCompactTrainPublicBackwardSuccessConsumesHandleAndSealsRefs(t *testing.T) {
	accel, cleanup := newBoundCompactTrainTestAccelerator(t, false, false)
	defer cleanup()
	shape := backend.CompactForwardShape{Batch: 1, Tokens: 2, ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 2, OutputDim: 4}
	refs := compactTrainResidentRefsForTest(t, accel, shape)
	if err := accel.BeginCompactTrainStep(11, refs); err != nil {
		t.Fatalf("begin step: %v", err)
	}
	req := backend.CompactTrainForwardRequest{
		Shape:        shape,
		Tokens:       [][]int32{{2, 1}},
		Masks:        [][]int32{{1, 1}},
		Roles:        []int32{0},
		ResidentRefs: refs,
		StepID:       11,
	}
	result, err := accel.RunCompactTrainForward(req)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	got, err := accel.RunCompactTrainBackward(backend.CompactTrainBackwardRequest{
		Handle:     result.Handle,
		GradPooled: backend.NewTensorF32([]int{1, 4}, seqData(4, 0.031, -0.047)),
	})
	if err != nil {
		t.Fatalf("public backward: %v", err)
	}
	if result.Handle.Token.Alive() {
		t.Fatal("public backward did not consume handle")
	}
	if _, err := accel.RunCompactTrainBackward(backend.CompactTrainBackwardRequest{
		Handle:     result.Handle,
		GradPooled: backend.NewTensorF32([]int{1, 4}, make([]float32, 4)),
	}); err == nil || !strings.Contains(err.Error(), "live handle") {
		t.Fatalf("double backward err = %v, want live handle", err)
	}
	if err := accel.ReleaseCompactTrainHandle(result.Handle); err == nil || !strings.Contains(err.Error(), "already released") {
		t.Fatalf("release after public backward err = %v, want already released", err)
	}
	if err := accel.EndCompactTrainStep(11); err != nil {
		t.Fatalf("end step: %v", err)
	}
	if _, err := accel.copyResidentGradientForDebug(residentGradRefByName(t, got.ResidentGradRefs, "token_embedding")); err != nil {
		t.Fatalf("sealed token grad copy after end: %v", err)
	}
	if err := accel.BeginCompactTrainStep(12, refs); err != nil {
		t.Fatalf("begin next step: %v", err)
	}
	for _, ref := range got.ResidentGradRefs {
		if ref.Token != nil && ref.Token.Alive() {
			t.Fatalf("resident grad ref %q alive after new begin", ref.Name)
		}
		if _, err := accel.copyResidentGradientForDebug(ref); err == nil || !strings.Contains(err.Error(), "stale") {
			t.Fatalf("old resident grad copy err = %v, want stale", err)
		}
	}
	if err := accel.EndCompactTrainStep(12); err != nil {
		t.Fatalf("end next step: %v", err)
	}
}

func TestCompactTrainPublicBackwardPostMutationFailurePoisonsStep(t *testing.T) {
	accel, cleanup := newBoundCompactTrainTestAccelerator(t, true, false)
	defer cleanup()
	shape := backend.CompactForwardShape{Batch: 1, Tokens: 2, ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 2, OutputDim: 4}
	refs := compactTrainResidentRefsForTest(t, accel, shape)
	if err := accel.BeginCompactTrainStep(41, refs); err != nil {
		t.Fatalf("begin step: %v", err)
	}
	req := backend.CompactTrainForwardRequest{
		Shape:        shape,
		Tokens:       [][]int32{{2, 2}},
		Masks:        [][]int32{{1, 1}},
		Roles:        []int32{1},
		ResidentRefs: refs,
		GELUMode:     backend.CompactForwardGELUExact,
		StepID:       41,
	}
	forward, err := accel.RunCompactTrainForward(req)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	poisonedRef := residentGradRefByName(t, accel.residentGradientRefsLocked(), "layer1_ffn_down")
	accel.debugForceBackwardFailureAfterGradMutation = true
	before := accel.CompactTrainStats()
	_, err = accel.RunCompactTrainBackward(backend.CompactTrainBackwardRequest{
		Handle:     forward.Handle,
		GradPooled: backend.NewTensorF32([]int{1, 4}, seqData(4, 0.031, -0.047)),
	})
	if err == nil || !strings.Contains(err.Error(), "forced failure") {
		t.Fatalf("forced backward err = %v, want forced failure", err)
	}
	after := accel.CompactTrainStats()
	if !accel.stepPoisoned {
		t.Fatal("forced post-mutation failure did not poison step")
	}
	if !forward.Handle.Token.Alive() {
		t.Fatal("poisoned backward consumed handle")
	}
	if after.FallbackOrUnhandled != before.FallbackOrUnhandled+1 || after.KernelLaunches <= before.KernelLaunches || after.KernelSynchronizations <= before.KernelSynchronizations || after.LastBackwardLaunches <= 0 || after.LastBackwardSyncs <= 0 {
		t.Fatalf("poisoned failure stats before=%+v after=%+v", before, after)
	}
	if poisonedRef.Token.Alive() {
		t.Fatal("poisoned resident gradient token is still alive")
	}
	if _, err := accel.copyResidentGradientForDebug(poisonedRef); err == nil || !strings.Contains(err.Error(), "poisoned") {
		t.Fatalf("poisoned resident grad copy err = %v, want poisoned", err)
	}
	if _, err := accel.RunCompactTrainBackward(backend.CompactTrainBackwardRequest{
		Handle:     forward.Handle,
		GradPooled: backend.NewTensorF32([]int{1, 4}, make([]float32, 4)),
	}); err == nil || !strings.Contains(err.Error(), "poisoned") {
		t.Fatalf("second backward on poisoned step err = %v, want poisoned", err)
	}
	if err := accel.BeginCompactTrainStep(41, refs); err == nil || !strings.Contains(err.Error(), "already active") {
		t.Fatalf("poison recovery begin with live handle err = %v, want already active", err)
	}
	if err := accel.ReleaseCompactTrainHandle(forward.Handle); err != nil {
		t.Fatalf("release poisoned handle: %v", err)
	}
	if err := accel.EndCompactTrainStep(41); err == nil || !strings.Contains(err.Error(), "poisoned") {
		t.Fatalf("end poisoned step err = %v, want poisoned", err)
	}
	if _, err := accel.copyResidentGradientForDebug(poisonedRef); err == nil || !strings.Contains(err.Error(), "poisoned") {
		t.Fatalf("poisoned resident grad after release/end err = %v, want poisoned", err)
	}
	accel.debugForceBackwardFailureAfterGradMutation = false
	if err := accel.BeginCompactTrainStep(41, refs); err != nil {
		t.Fatalf("begin recovery step with reused step id: %v", err)
	}
	if poisonedRef.Token.Alive() {
		t.Fatal("old poisoned token became alive after recovery begin")
	}
	if _, err := accel.copyResidentGradientForDebug(poisonedRef); err == nil || !strings.Contains(err.Error(), "stale") {
		t.Fatalf("old poisoned resident grad after recovery err = %v, want stale", err)
	}
	recovered, err := accel.RunCompactTrainForward(req)
	if err != nil {
		t.Fatalf("recovery forward: %v", err)
	}
	got, err := accel.RunCompactTrainBackward(backend.CompactTrainBackwardRequest{
		Handle:     recovered.Handle,
		GradPooled: backend.NewTensorF32([]int{1, 4}, seqData(4, 0.019, -0.023)),
	})
	if err != nil {
		t.Fatalf("recovery backward: %v", err)
	}
	assertResidentGradientRefsUniqueAndComplete(t, got.ResidentGradRefs, accel.requiredResidentNames(shape))
	if err := accel.EndCompactTrainStep(41); err != nil {
		t.Fatalf("end recovery step: %v", err)
	}
	if base := accel.CompactForwardAccelerator.Stats(); base.PackedDownloads != 0 || base.PackedBytes != 0 {
		t.Fatalf("packed stats changed on poisoned compact train path: %+v", base)
	}
}

func TestCompactTrainFinalOutputBackwardDebugParityAndResidentRefs(t *testing.T) {
	for _, tc := range []struct {
		name       string
		projection bool
		rope       bool
		tokens     [][]int32
		masks      [][]int32
		roles      []int32
	}{
		{
			name:   "no_projection_b2_masks",
			rope:   true,
			tokens: [][]int32{{2, 1, 3}, {1, 4, 2}},
			masks:  [][]int32{{1, 1, 1}, {1, 0, 1}},
			roles:  []int32{0, 2},
		},
		{
			name:       "projection_inactive_row",
			projection: true,
			tokens:     [][]int32{{2, 1, 3}},
			masks:      [][]int32{{1, 1, 0}},
			roles:      []int32{1},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			accel, cleanup := newBoundCompactTrainTestAccelerator(t, tc.rope, tc.projection)
			defer cleanup()
			shape := backend.CompactForwardShape{Batch: len(tc.tokens), Tokens: len(tc.tokens[0]), ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 2, OutputDim: 4}
			if tc.projection {
				shape.OutputDim = 3
				shape.HasOutputProjection = true
			}
			refs := compactTrainResidentRefsForTest(t, accel, shape)
			if err := accel.BeginCompactTrainStep(21, refs); err != nil {
				t.Fatalf("begin step: %v", err)
			}
			req := backend.CompactTrainForwardRequest{
				Shape:        shape,
				Tokens:       tc.tokens,
				Masks:        tc.masks,
				Roles:        tc.roles,
				ResidentRefs: refs,
				GELUMode:     backend.CompactForwardGELUExact,
				StepID:       21,
			}
			forward, err := accel.RunCompactTrainForward(req)
			if err != nil {
				t.Fatalf("forward: %v", err)
			}
			gradPooled := backend.NewTensorF32([]int{shape.Batch, shape.OutputDim}, seqData(shape.Batch*shape.OutputDim, 0.031, -0.047))
			got, err := accel.runCompactTrainFinalOutputBackwardForDebug(backend.CompactTrainBackwardRequest{Handle: forward.Handle, GradPooled: gradPooled})
			if err != nil {
				t.Fatalf("debug final backward: %v", err)
			}
			if forward.Handle.Token.Alive() {
				t.Fatal("debug final backward did not consume handle")
			}
			wantHidden, wantProjection := hostCompactTrainFinalOutputBackwardForCUDATest(req, tc.rope, tc.projection, gradPooled.F32)
			t.Logf("final-output grad_hidden max_abs=%g", compactTrainMaxAbs(got.GradHidden.F32, wantHidden))
			assertFloatSlicesClose(t, got.GradHidden.F32, wantHidden, 3e-8)
			if tc.projection {
				ref := residentGradRefByName(t, got.ResidentGradRefs, "output_projection")
				grad, err := accel.copyResidentGradientForDebug(ref)
				if err != nil {
					t.Fatalf("copy resident output projection grad: %v", err)
				}
				t.Logf("final-output output_projection grad max_abs=%g", compactTrainMaxAbs(grad.F32, wantProjection))
				assertFloatSlicesClose(t, grad.F32, wantProjection, 3e-8)
				stale := ref
				stale.Generation++
				if _, err := accel.copyResidentGradientForDebug(stale); err == nil || !strings.Contains(err.Error(), "metadata mismatch") {
					t.Fatalf("stale generation copy err = %v, want metadata mismatch", err)
				}
				wrongName := ref
				wrongName.Name = "other"
				if _, err := accel.copyResidentGradientForDebug(wrongName); err == nil || !strings.Contains(err.Error(), "owner/name mismatch") {
					t.Fatalf("wrong-name copy err = %v, want owner/name mismatch", err)
				}
			}
			if err := accel.EndCompactTrainStep(21); err != nil {
				t.Fatalf("end step: %v", err)
			}
			for _, ref := range got.ResidentGradRefs {
				if _, err := accel.copyResidentGradientForDebug(ref); err != nil {
					t.Fatalf("sealed resident grad copy after end for %q: %v", ref.Name, err)
				}
			}
			if err := accel.BeginCompactTrainStep(22, refs); err != nil {
				t.Fatalf("begin second step: %v", err)
			}
			for _, ref := range got.ResidentGradRefs {
				if ref.Token != nil && ref.Token.Alive() {
					t.Fatalf("resident grad ref %q alive after new begin", ref.Name)
				}
				if _, err := accel.copyResidentGradientForDebug(ref); err == nil || !strings.Contains(err.Error(), "stale") {
					t.Fatalf("old-step resident grad copy err = %v, want stale", err)
				}
			}
			if err := accel.EndCompactTrainStep(22); err != nil {
				t.Fatalf("end second step: %v", err)
			}
			stats := accel.CompactTrainStats()
			if stats.LiveHandles != 0 || stats.BackwardCalls != 1 || stats.GradPooledUploadedBytes != int64(shape.Batch*shape.OutputDim*4) {
				t.Fatalf("final backward stats = %+v", stats)
			}
		})
	}
}

func TestCompactTrainResidualLayerNormBackwardKernelParity(t *testing.T) {
	accel, cleanup := newBoundCompactTrainTestAccelerator(t, false, false)
	defer cleanup()
	rows, cols := 3, 4
	gradOut := seqData(rows*cols, 0.019, -0.041)
	pre := seqData(rows*cols, -0.027, 0.083)
	normalized := layerNormHost(pre, rows, cols)
	want := make([]float32, rows*cols)
	for row := 0; row < rows; row++ {
		base := row * cols
		compactTrainBackwardLayerNormRow(want[base:base+cols], gradOut[base:base+cols], normalized[base:base+cols], pre[base:base+cols])
	}
	gradPtr, err := accel.device.uploadFloat32(gradOut)
	if err != nil {
		t.Fatalf("upload grad: %v", err)
	}
	defer accel.device.freeBuffer(gradPtr)
	normPtr, err := accel.device.uploadFloat32(normalized)
	if err != nil {
		t.Fatalf("upload normalized: %v", err)
	}
	defer accel.device.freeBuffer(normPtr)
	prePtr, err := accel.device.uploadFloat32(pre)
	if err != nil {
		t.Fatalf("upload pre: %v", err)
	}
	defer accel.device.freeBuffer(prePtr)
	outPtr, err := accel.device.allocFloat32(rows * cols)
	if err != nil {
		t.Fatalf("alloc out: %v", err)
	}
	defer accel.device.freeBuffer(outPtr)
	if err := accel.launchLayerNormBackward(gradPtr, normPtr, prePtr, outPtr, rows, cols); err != nil {
		t.Fatalf("launch layernorm backward: %v", err)
	}
	got := make([]float32, rows*cols)
	if err := accel.device.downloadFloat32(got, outPtr); err != nil {
		t.Fatalf("download layernorm backward: %v", err)
	}
	t.Logf("resident layernorm backward max_abs=%g rmse=%g", compactTrainMaxAbs(got, want), compactTrainRMSE(got, want))
	assertFloatSlicesClose(t, got, want, 3e-8)
}

func TestCompactTrainGELUBackwardKernelExactAndFastParity(t *testing.T) {
	accel, cleanup := newBoundCompactTrainTestAccelerator(t, false, false)
	defer cleanup()
	gradOut := seqData(17, -0.013, 0.037)
	preAct := []float32{-3.5, -3, -2.25, -1.5, -0.75, -0.25, 0, 0.25, 0.75, 1.5, 2.25, 3, 3.5, -0.11, 0.43, -1.17, 2.01}
	for _, fast := range []bool{false, true} {
		want := make([]float32, len(preAct))
		compactTrainFillGELUBackwardMul(want, gradOut, preAct, fast)
		gradPtr, err := accel.device.uploadFloat32(gradOut)
		if err != nil {
			t.Fatalf("upload grad fast=%v: %v", fast, err)
		}
		prePtr, err := accel.device.uploadFloat32(preAct)
		if err != nil {
			accel.device.freeBuffer(gradPtr)
			t.Fatalf("upload pre fast=%v: %v", fast, err)
		}
		outPtr, err := accel.device.allocFloat32(len(preAct))
		if err != nil {
			accel.device.freeBuffer(gradPtr)
			accel.device.freeBuffer(prePtr)
			t.Fatalf("alloc out fast=%v: %v", fast, err)
		}
		if err := accel.launchGELUBackward(gradPtr, prePtr, outPtr, len(preAct), fast); err != nil {
			accel.device.freeBuffer(gradPtr)
			accel.device.freeBuffer(prePtr)
			accel.device.freeBuffer(outPtr)
			t.Fatalf("launch gelu backward fast=%v: %v", fast, err)
		}
		got := make([]float32, len(preAct))
		if err := accel.device.downloadFloat32(got, outPtr); err != nil {
			accel.device.freeBuffer(gradPtr)
			accel.device.freeBuffer(prePtr)
			accel.device.freeBuffer(outPtr)
			t.Fatalf("download gelu backward fast=%v: %v", fast, err)
		}
		accel.device.freeBuffer(gradPtr)
		accel.device.freeBuffer(prePtr)
		accel.device.freeBuffer(outPtr)
		t.Logf("resident gelu backward fast=%v max_abs=%g rmse=%g", fast, compactTrainMaxAbs(got, want), compactTrainRMSE(got, want))
		assertFloatSlicesClose(t, got, want, 3e-8)
	}
}

func TestCompactTrainAttentionBackwardKernelParity(t *testing.T) {
	for _, tc := range []struct {
		name  string
		shape backend.CompactForwardShape
		mask  []int32
	}{
		{
			name:  "heads1_t1",
			shape: backend.CompactForwardShape{Batch: 1, Tokens: 1, ModelDim: 4, FFNDim: 5, Heads: 1, HeadDim: 4, Layers: 1, OutputDim: 4},
			mask:  []int32{1},
		},
		{
			name:  "heads2_t3_inactive_key",
			shape: backend.CompactForwardShape{Batch: 1, Tokens: 3, ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 1, OutputDim: 4},
			mask:  []int32{1, 0, 1},
		},
		{
			name:  "heads2_b2_t2",
			shape: backend.CompactForwardShape{Batch: 2, Tokens: 2, ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 1, OutputDim: 4},
			mask:  []int32{1, 1, 1, 0},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			accel, cleanup := newBoundCompactTrainShapeTestAccelerator(t, tc.shape, false, false, compactTrainShapeTestWeights(tc.shape, false))
			defer cleanup()
			rows := tc.shape.Batch * tc.shape.Tokens
			q := seqData(rows*tc.shape.ModelDim, 0.011, -0.017)
			k := seqData(rows*tc.shape.ModelDim, -0.007, 0.019)
			v := seqData(rows*tc.shape.ModelDim, 0.013, -0.023)
			gradMixed := seqData(rows*tc.shape.ModelDim, -0.005, 0.031)
			probs := make([]float32, tc.shape.Batch*tc.shape.Heads*tc.shape.Tokens*tc.shape.Tokens)
			for b := 0; b < tc.shape.Batch; b++ {
				mask := tc.mask[b*tc.shape.Tokens : (b+1)*tc.shape.Tokens]
				qb := q[b*tc.shape.Tokens*tc.shape.ModelDim : (b+1)*tc.shape.Tokens*tc.shape.ModelDim]
				kb := k[b*tc.shape.Tokens*tc.shape.ModelDim : (b+1)*tc.shape.Tokens*tc.shape.ModelDim]
				vb := v[b*tc.shape.Tokens*tc.shape.ModelDim : (b+1)*tc.shape.Tokens*tc.shape.ModelDim]
				score, _ := attentionHost(qb, kb, vb, mask, tc.shape)
				copy(probs[b*tc.shape.Heads*tc.shape.Tokens*tc.shape.Tokens:], score)
			}
			wantQ, wantK, wantV := compactTrainAttentionBackwardHost(gradMixed, q, k, v, probs, tc.shape)
			gradMixedPtr, err := accel.device.uploadFloat32(gradMixed)
			if err != nil {
				t.Fatalf("upload gradMixed: %v", err)
			}
			defer accel.device.freeBuffer(gradMixedPtr)
			qPtr, err := accel.device.uploadFloat32(q)
			if err != nil {
				t.Fatalf("upload q: %v", err)
			}
			defer accel.device.freeBuffer(qPtr)
			kPtr, err := accel.device.uploadFloat32(k)
			if err != nil {
				t.Fatalf("upload k: %v", err)
			}
			defer accel.device.freeBuffer(kPtr)
			vPtr, err := accel.device.uploadFloat32(v)
			if err != nil {
				t.Fatalf("upload v: %v", err)
			}
			defer accel.device.freeBuffer(vPtr)
			probsPtr, err := accel.device.uploadFloat32(probs)
			if err != nil {
				t.Fatalf("upload probs: %v", err)
			}
			defer accel.device.freeBuffer(probsPtr)
			gradQPtr, err := accel.device.allocFloat32(len(wantQ))
			if err != nil {
				t.Fatalf("alloc gradQ: %v", err)
			}
			defer accel.device.freeBuffer(gradQPtr)
			gradKPtr, err := accel.device.allocFloat32(len(wantK))
			if err != nil {
				t.Fatalf("alloc gradK: %v", err)
			}
			defer accel.device.freeBuffer(gradKPtr)
			gradVPtr, err := accel.device.allocFloat32(len(wantV))
			if err != nil {
				t.Fatalf("alloc gradV: %v", err)
			}
			defer accel.device.freeBuffer(gradVPtr)
			if err := accel.launchAttentionBackward(gradMixedPtr, qPtr, kPtr, vPtr, probsPtr, gradQPtr, gradKPtr, gradVPtr, tc.shape); err != nil {
				t.Fatalf("launch attention backward: %v", err)
			}
			gotQ := make([]float32, len(wantQ))
			if err := accel.device.downloadFloat32(gotQ, gradQPtr); err != nil {
				t.Fatalf("download gradQ: %v", err)
			}
			gotK := make([]float32, len(wantK))
			if err := accel.device.downloadFloat32(gotK, gradKPtr); err != nil {
				t.Fatalf("download gradK: %v", err)
			}
			gotV := make([]float32, len(wantV))
			if err := accel.device.downloadFloat32(gotV, gradVPtr); err != nil {
				t.Fatalf("download gradV: %v", err)
			}
			t.Logf("attention backward q max_abs=%g rmse=%g", compactTrainMaxAbs(gotQ, wantQ), compactTrainRMSE(gotQ, wantQ))
			t.Logf("attention backward k max_abs=%g rmse=%g", compactTrainMaxAbs(gotK, wantK), compactTrainRMSE(gotK, wantK))
			t.Logf("attention backward v max_abs=%g rmse=%g", compactTrainMaxAbs(gotV, wantV), compactTrainRMSE(gotV, wantV))
			assertFloatSlicesClose(t, gotQ, wantQ, 3e-8)
			assertFloatSlicesClose(t, gotK, wantK, 3e-8)
			assertFloatSlicesClose(t, gotV, wantV, 3e-8)
		})
	}
}

func TestCompactTrainFFNBackwardDebugParityAndResidentAccumulation(t *testing.T) {
	for _, tc := range []struct {
		name       string
		layers     int
		projection bool
		gelu       string
		tokens     [][]int32
		masks      [][]int32
		roles      []int32
	}{
		{
			name:   "b1_t2_exact",
			layers: 1,
			gelu:   backend.CompactForwardGELUExact,
			tokens: [][]int32{{2, 1}},
			masks:  [][]int32{{1, 1}},
			roles:  []int32{0},
		},
		{
			name:   "b1_t2_exact_2layers",
			layers: 2,
			gelu:   backend.CompactForwardGELUExact,
			tokens: [][]int32{{2, 1}},
			masks:  [][]int32{{1, 1}},
			roles:  []int32{0},
		},
		{
			name:   "b2_t3_masks_fast",
			layers: 2,
			gelu:   backend.CompactForwardGELUFast,
			tokens: [][]int32{{2, 1, 3}, {1, 4, 2}},
			masks:  [][]int32{{1, 1, 0}, {1, 0, 1}},
			roles:  []int32{0, 2},
		},
		{
			name:       "projection_boundary",
			layers:     2,
			projection: true,
			gelu:       backend.CompactForwardGELUExact,
			tokens:     [][]int32{{2, 1, 3}},
			masks:      [][]int32{{1, 1, 0}},
			roles:      []int32{1},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			shape := backend.CompactForwardShape{Batch: len(tc.tokens), Tokens: len(tc.tokens[0]), ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: tc.layers, OutputDim: 4}
			if tc.projection {
				shape.OutputDim = 3
				shape.HasOutputProjection = true
			}
			accel, cleanup := newBoundCompactTrainShapeTestAccelerator(t, shape, true, tc.projection, compactForwardTestWeights(tc.projection))
			defer cleanup()
			refs := compactTrainResidentRefsForTest(t, accel, shape)
			if err := accel.BeginCompactTrainStep(41, refs); err != nil {
				t.Fatalf("begin step: %v", err)
			}
			req := backend.CompactTrainForwardRequest{
				Shape:        shape,
				Tokens:       tc.tokens,
				Masks:        tc.masks,
				Roles:        tc.roles,
				ResidentRefs: refs,
				GELUMode:     tc.gelu,
				StepID:       41,
			}
			forward, err := accel.RunCompactTrainForward(req)
			if err != nil {
				t.Fatalf("forward: %v", err)
			}
			gradPooled := backend.NewTensorF32([]int{shape.Batch, shape.OutputDim}, seqData(shape.Batch*shape.OutputDim, 0.031, -0.047))
			got, err := accel.runCompactTrainFFNBackwardForDebug(backend.CompactTrainBackwardRequest{Handle: forward.Handle, GradPooled: gradPooled})
			if err != nil {
				t.Fatalf("ffn debug backward: %v", err)
			}
			if got.Layer != shape.Layers-1 {
				t.Fatalf("debug layer = %d, want %d", got.Layer, shape.Layers-1)
			}
			want := hostCompactTrainTopFFNBackwardForCUDATest(req, true, tc.projection, gradPooled.F32)
			t.Logf("ffn debug grad_hidden max_abs=%g rmse=%g", compactTrainMaxAbs(got.GradHidden.F32, want.gradHidden), compactTrainRMSE(got.GradHidden.F32, want.gradHidden))
			t.Logf("ffn debug attention_boundary max_abs=%g rmse=%g", compactTrainMaxAbs(got.GradAttentionBoundary.F32, want.gradAttention), compactTrainRMSE(got.GradAttentionBoundary.F32, want.gradAttention))
			assertFloatSlicesClose(t, got.GradHidden.F32, want.gradHidden, 3e-8)
			assertFloatSlicesClose(t, got.GradAttentionBoundary.F32, want.gradAttention, 3e-8)
			layerPrefix := fmt.Sprintf("layer%d_", shape.Layers-1)
			upRef := residentGradRefByName(t, got.ResidentGradRefs, layerPrefix+"ffn_up")
			downRef := residentGradRefByName(t, got.ResidentGradRefs, layerPrefix+"ffn_down")
			upGrad, err := accel.copyResidentGradientForDebug(upRef)
			if err != nil {
				t.Fatalf("copy ffn_up grad: %v", err)
			}
			downGrad, err := accel.copyResidentGradientForDebug(downRef)
			if err != nil {
				t.Fatalf("copy ffn_down grad: %v", err)
			}
			t.Logf("ffn_up grad max_abs=%g rmse=%g", compactTrainMaxAbs(upGrad.F32, want.ffnUp), compactTrainRMSE(upGrad.F32, want.ffnUp))
			t.Logf("ffn_down grad max_abs=%g rmse=%g", compactTrainMaxAbs(downGrad.F32, want.ffnDown), compactTrainRMSE(downGrad.F32, want.ffnDown))
			assertFloatSlicesClose(t, upGrad.F32, want.ffnUp, 3e-8)
			assertFloatSlicesClose(t, downGrad.F32, want.ffnDown, 3e-8)
			assertCompactTrainBackwardWorkspaceDistinct(t, accel, shape)
			if err := accel.EndCompactTrainStep(41); err != nil {
				t.Fatalf("end step: %v", err)
			}
		})
	}
}

func TestCompactTrainFFNBackwardResidentGradAccumulatesAcrossBuckets(t *testing.T) {
	accel, cleanup := newBoundCompactTrainTestAccelerator(t, true, false)
	defer cleanup()
	shape := backend.CompactForwardShape{Batch: 1, Tokens: 2, ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 2, OutputDim: 4}
	refs := compactTrainResidentRefsForTest(t, accel, shape)
	if err := accel.BeginCompactTrainStep(51, refs); err != nil {
		t.Fatalf("begin step: %v", err)
	}
	defer func() {
		if err := accel.EndCompactTrainStep(51); err != nil {
			t.Fatalf("end step: %v", err)
		}
	}()
	var wantUp, wantDown []float32
	var lastRefs []backend.ResidentGradientRef
	for bucket, tokens := range [][][]int32{{{2, 1}}, {{3, 4}}} {
		req := backend.CompactTrainForwardRequest{
			Shape:        shape,
			Tokens:       tokens,
			Masks:        [][]int32{{1, 1}},
			Roles:        []int32{int32(bucket)},
			ResidentRefs: refs,
			GELUMode:     backend.CompactForwardGELUExact,
			StepID:       51,
		}
		forward, err := accel.RunCompactTrainForward(req)
		if err != nil {
			t.Fatalf("forward bucket %d: %v", bucket, err)
		}
		gradPooled := backend.NewTensorF32([]int{1, shape.OutputDim}, seqData(shape.OutputDim, 0.017+float64(bucket)*0.003, -0.029))
		got, err := accel.runCompactTrainFFNBackwardForDebug(backend.CompactTrainBackwardRequest{Handle: forward.Handle, GradPooled: gradPooled})
		if err != nil {
			t.Fatalf("ffn backward bucket %d: %v", bucket, err)
		}
		lastRefs = got.ResidentGradRefs
		want := hostCompactTrainTopFFNBackwardForCUDATest(req, true, false, gradPooled.F32)
		if wantUp == nil {
			wantUp = make([]float32, len(want.ffnUp))
			wantDown = make([]float32, len(want.ffnDown))
		}
		addFloat32SlicesForCUDATest(wantUp, want.ffnUp)
		addFloat32SlicesForCUDATest(wantDown, want.ffnDown)
	}
	upGrad, err := accel.copyResidentGradientForDebug(residentGradRefByName(t, lastRefs, "layer1_ffn_up"))
	if err != nil {
		t.Fatalf("copy accumulated ffn_up: %v", err)
	}
	downGrad, err := accel.copyResidentGradientForDebug(residentGradRefByName(t, lastRefs, "layer1_ffn_down"))
	if err != nil {
		t.Fatalf("copy accumulated ffn_down: %v", err)
	}
	t.Logf("accumulated ffn_up max_abs=%g rmse=%g", compactTrainMaxAbs(upGrad.F32, wantUp), compactTrainRMSE(upGrad.F32, wantUp))
	t.Logf("accumulated ffn_down max_abs=%g rmse=%g", compactTrainMaxAbs(downGrad.F32, wantDown), compactTrainRMSE(downGrad.F32, wantDown))
	assertFloatSlicesClose(t, upGrad.F32, wantUp, 3e-8)
	assertFloatSlicesClose(t, downGrad.F32, wantDown, 3e-8)
	stats := accel.CompactTrainStats()
	if stats.BackwardCalls != 2 || stats.GradPooledUploadedBytes != int64(2*shape.OutputDim*4) || stats.LiveHandles != 0 {
		t.Fatalf("accumulation stats = %+v", stats)
	}
}

func TestCompactTrainLayerBackwardDebugParityAndResidentAccumulation(t *testing.T) {
	for _, tc := range []struct {
		name       string
		layers     int
		projection bool
		rope       bool
		gelu       string
		tokens     [][]int32
		masks      [][]int32
		roles      []int32
	}{
		{
			name:   "one_layer_rope_t2",
			layers: 1,
			rope:   true,
			gelu:   backend.CompactForwardGELUExact,
			tokens: [][]int32{{2, 1}},
			masks:  [][]int32{{1, 1}},
			roles:  []int32{0},
		},
		{
			name:   "one_layer_no_rope_heads2_t3_inactive",
			layers: 1,
			gelu:   backend.CompactForwardGELUExact,
			tokens: [][]int32{{2, 1, 3}},
			masks:  [][]int32{{1, 0, 1}},
			roles:  []int32{1},
		},
		{
			name:   "one_layer_rope_b2_fast",
			layers: 1,
			rope:   true,
			gelu:   backend.CompactForwardGELUFast,
			tokens: [][]int32{{2, 1, 3}, {1, 4, 2}},
			masks:  [][]int32{{1, 1, 0}, {1, 0, 1}},
			roles:  []int32{0, 2},
		},
		{
			name:       "one_layer_projection",
			layers:     1,
			projection: true,
			rope:       true,
			gelu:       backend.CompactForwardGELUExact,
			tokens:     [][]int32{{2, 1, 3}},
			masks:      [][]int32{{1, 1, 0}},
			roles:      []int32{1},
		},
		{
			name:   "two_layer_top_to_lower",
			layers: 2,
			rope:   true,
			gelu:   backend.CompactForwardGELUExact,
			tokens: [][]int32{{2, 1}},
			masks:  [][]int32{{1, 1}},
			roles:  []int32{0},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			shape := backend.CompactForwardShape{Batch: len(tc.tokens), Tokens: len(tc.tokens[0]), ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: tc.layers, OutputDim: 4}
			if tc.projection {
				shape.OutputDim = 3
				shape.HasOutputProjection = true
			}
			accel, cleanup := newBoundCompactTrainShapeTestAccelerator(t, shape, tc.rope, tc.projection, compactForwardTestWeights(tc.projection))
			defer cleanup()
			refs := compactTrainResidentRefsForTest(t, accel, shape)
			if err := accel.BeginCompactTrainStep(61, refs); err != nil {
				t.Fatalf("begin step: %v", err)
			}
			req := backend.CompactTrainForwardRequest{
				Shape:        shape,
				Tokens:       tc.tokens,
				Masks:        tc.masks,
				Roles:        tc.roles,
				ResidentRefs: refs,
				GELUMode:     tc.gelu,
				StepID:       61,
			}
			forward, err := accel.RunCompactTrainForward(req)
			if err != nil {
				t.Fatalf("forward: %v", err)
			}
			gradPooled := backend.NewTensorF32([]int{shape.Batch, shape.OutputDim}, seqData(shape.Batch*shape.OutputDim, 0.031, -0.047))
			got, err := accel.runCompactTrainLayerBackwardForDebug(backend.CompactTrainBackwardRequest{Handle: forward.Handle, GradPooled: gradPooled})
			if err != nil {
				t.Fatalf("layer debug backward: %v", err)
			}
			if got.Layer != shape.Layers-1 {
				t.Fatalf("debug layer = %d, want %d", got.Layer, shape.Layers-1)
			}
			want := hostCompactTrainTopLayerBackwardForCUDATest(req, tc.rope, tc.projection, gradPooled.F32)
			t.Logf("layer input max_abs=%g rmse=%g", compactTrainMaxAbs(got.GradLayerInput.F32, want.gradLayerInput), compactTrainRMSE(got.GradLayerInput.F32, want.gradLayerInput))
			assertFloatSlicesClose(t, got.GradLayerInput.F32, want.gradLayerInput, 3e-8)
			if want.gradRoPEInput != nil {
				if got.GradRoPEInput == nil {
					t.Fatal("missing RoPE-transposed input grad")
				}
				t.Logf("rope input max_abs=%g rmse=%g", compactTrainMaxAbs(got.GradRoPEInput.F32, want.gradRoPEInput), compactTrainRMSE(got.GradRoPEInput.F32, want.gradRoPEInput))
				assertFloatSlicesClose(t, got.GradRoPEInput.F32, want.gradRoPEInput, 3e-8)
			} else if got.GradRoPEInput != nil {
				t.Fatalf("unexpected RoPE-transposed input grad: %+v", got.GradRoPEInput.Shape)
			}
			layerPrefix := fmt.Sprintf("layer%d_", shape.Layers-1)
			for _, item := range []struct {
				name string
				want []float32
			}{
				{layerPrefix + "attn_q", want.attnQ},
				{layerPrefix + "attn_k", want.attnK},
				{layerPrefix + "attn_v", want.attnV},
				{layerPrefix + "attn_o", want.attnO},
				{layerPrefix + "ffn_up", want.ffnUp},
				{layerPrefix + "ffn_down", want.ffnDown},
			} {
				grad, err := accel.copyResidentGradientForDebug(residentGradRefByName(t, got.ResidentGradRefs, item.name))
				if err != nil {
					t.Fatalf("copy %s grad: %v", item.name, err)
				}
				t.Logf("%s grad max_abs=%g rmse=%g", item.name, compactTrainMaxAbs(grad.F32, item.want), compactTrainRMSE(grad.F32, item.want))
				assertFloatSlicesClose(t, grad.F32, item.want, 3e-8)
			}
			assertCompactTrainBackwardWorkspaceDistinct(t, accel, shape)
			if err := accel.EndCompactTrainStep(61); err != nil {
				t.Fatalf("end step: %v", err)
			}
		})
	}
}

func TestCompactTrainLayerBackwardResidentGradAccumulatesAcrossBuckets(t *testing.T) {
	shape := backend.CompactForwardShape{Batch: 1, Tokens: 2, ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 1, OutputDim: 4}
	accel, cleanup := newBoundCompactTrainShapeTestAccelerator(t, shape, true, false, compactForwardTestWeights(false))
	defer cleanup()
	refs := compactTrainResidentRefsForTest(t, accel, shape)
	if err := accel.BeginCompactTrainStep(71, refs); err != nil {
		t.Fatalf("begin step: %v", err)
	}
	defer func() {
		if err := accel.EndCompactTrainStep(71); err != nil {
			t.Fatalf("end step: %v", err)
		}
	}()
	wantByName := map[string][]float32{}
	var lastRefs []backend.ResidentGradientRef
	for bucket, tokens := range [][][]int32{{{2, 1}}, {{3, 4}}} {
		req := backend.CompactTrainForwardRequest{
			Shape:        shape,
			Tokens:       tokens,
			Masks:        [][]int32{{1, 1}},
			Roles:        []int32{int32(bucket)},
			ResidentRefs: refs,
			GELUMode:     backend.CompactForwardGELUExact,
			StepID:       71,
		}
		forward, err := accel.RunCompactTrainForward(req)
		if err != nil {
			t.Fatalf("forward bucket %d: %v", bucket, err)
		}
		gradPooled := backend.NewTensorF32([]int{1, shape.OutputDim}, seqData(shape.OutputDim, 0.017+float64(bucket)*0.003, -0.029))
		got, err := accel.runCompactTrainLayerBackwardForDebug(backend.CompactTrainBackwardRequest{Handle: forward.Handle, GradPooled: gradPooled})
		if err != nil {
			t.Fatalf("layer backward bucket %d: %v", bucket, err)
		}
		lastRefs = got.ResidentGradRefs
		want := hostCompactTrainTopLayerBackwardForCUDATest(req, true, false, gradPooled.F32)
		for _, item := range []struct {
			name string
			data []float32
		}{
			{"layer0_attn_q", want.attnQ},
			{"layer0_attn_k", want.attnK},
			{"layer0_attn_v", want.attnV},
			{"layer0_attn_o", want.attnO},
			{"layer0_ffn_up", want.ffnUp},
			{"layer0_ffn_down", want.ffnDown},
		} {
			if wantByName[item.name] == nil {
				wantByName[item.name] = make([]float32, len(item.data))
			}
			addFloat32SlicesForCUDATest(wantByName[item.name], item.data)
		}
	}
	for name, want := range wantByName {
		grad, err := accel.copyResidentGradientForDebug(residentGradRefByName(t, lastRefs, name))
		if err != nil {
			t.Fatalf("copy accumulated %s: %v", name, err)
		}
		t.Logf("accumulated %s max_abs=%g rmse=%g", name, compactTrainMaxAbs(grad.F32, want), compactTrainRMSE(grad.F32, want))
		assertFloatSlicesClose(t, grad.F32, want, 3e-8)
	}
	stats := accel.CompactTrainStats()
	if stats.BackwardCalls != 2 || stats.GradPooledUploadedBytes != int64(2*shape.OutputDim*4) || stats.LiveHandles != 0 {
		t.Fatalf("accumulation stats = %+v", stats)
	}
}

func TestCompactTrainPublicBackwardFullParityAndResidentRefs(t *testing.T) {
	for _, tc := range []struct {
		name       string
		layers     int
		projection bool
		rope       bool
		gelu       string
		tokens     [][]int32
		masks      [][]int32
		roles      []int32
	}{
		{
			name:   "b1_t2_l1_rope_exact_duplicate_tokens",
			layers: 1,
			rope:   true,
			gelu:   backend.CompactForwardGELUExact,
			tokens: [][]int32{{2, 2}},
			masks:  [][]int32{{1, 1}},
			roles:  []int32{1},
		},
		{
			name:       "b2_t3_l2_no_rope_fast_mask_repeated_role_projection",
			layers:     2,
			projection: true,
			gelu:       backend.CompactForwardGELUFast,
			tokens:     [][]int32{{2, 1, 3}, {1, 4, 2}},
			masks:      [][]int32{{1, 0, 1}, {1, 1, 0}},
			roles:      []int32{2, 2},
		},
		{
			name:   "b1_t2_l2_rope_exact",
			layers: 2,
			rope:   true,
			gelu:   backend.CompactForwardGELUExact,
			tokens: [][]int32{{2, 1}},
			masks:  [][]int32{{1, 1}},
			roles:  []int32{0},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			shape := backend.CompactForwardShape{Batch: len(tc.tokens), Tokens: len(tc.tokens[0]), ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: tc.layers, OutputDim: 4}
			if tc.projection {
				shape.OutputDim = 3
				shape.HasOutputProjection = true
			}
			accel, cleanup := newBoundCompactTrainShapeTestAccelerator(t, shape, tc.rope, tc.projection, compactForwardTestWeights(tc.projection))
			defer cleanup()
			refs := compactTrainResidentRefsForTest(t, accel, shape)
			if err := accel.BeginCompactTrainStep(81, refs); err != nil {
				t.Fatalf("begin step: %v", err)
			}
			req := backend.CompactTrainForwardRequest{
				Shape:        shape,
				Tokens:       tc.tokens,
				Masks:        tc.masks,
				Roles:        tc.roles,
				ResidentRefs: refs,
				GELUMode:     tc.gelu,
				StepID:       81,
			}
			forward, err := accel.RunCompactTrainForward(req)
			if err != nil {
				t.Fatalf("forward: %v", err)
			}
			badStats := accel.CompactTrainStats()
			if _, err := accel.RunCompactTrainBackward(backend.CompactTrainBackwardRequest{
				Handle:     forward.Handle,
				GradPooled: backend.NewTensorF32([]int{shape.Batch, shape.OutputDim + 1}, make([]float32, shape.Batch*(shape.OutputDim+1))),
			}); err == nil || !strings.Contains(err.Error(), "pooled gradient shape") {
				t.Fatalf("bad grad shape err = %v, want shape", err)
			}
			assertCompactTrainStatsUnchanged(t, badStats, accel.CompactTrainStats(), "bad public backward shape")
			if !forward.Handle.Token.Alive() {
				t.Fatal("bad public backward consumed handle")
			}
			gradPooled := backend.NewTensorF32([]int{shape.Batch, shape.OutputDim}, seqData(shape.Batch*shape.OutputDim, 0.031, -0.047))
			got, err := accel.RunCompactTrainBackward(backend.CompactTrainBackwardRequest{Handle: forward.Handle, GradPooled: gradPooled})
			if err != nil {
				t.Fatalf("public backward: %v", err)
			}
			want := hostCompactTrainFullBackwardForCUDATest(req, tc.rope, tc.projection, gradPooled.F32)
			assertResidentGradientRefsUniqueAndComplete(t, got.ResidentGradRefs, accel.requiredResidentNames(shape))
			assertCompactTrainResidentGradientsClose(t, accel, got.ResidentGradRefs, want, 3e-8)
			if err := accel.EndCompactTrainStep(81); err != nil {
				t.Fatalf("end step: %v", err)
			}
			assertCompactTrainResidentGradientsClose(t, accel, got.ResidentGradRefs, want, 3e-8)
			stats := accel.CompactTrainStats()
			if stats.BackwardCalls != 1 || stats.GradPooledUploadedBytes != int64(shape.Batch*shape.OutputDim*4) || stats.LiveHandles != 0 || stats.FallbackOrUnhandled != 0 {
				t.Fatalf("public backward stats = %+v", stats)
			}
			if base := accel.CompactForwardAccelerator.Stats(); base.PackedDownloads != 0 || base.PackedBytes != 0 {
				t.Fatalf("packed stats changed on compact train path: %+v", base)
			}
		})
	}
}

func TestCompactTrainPublicBackwardAccumulatesVaryingBuckets(t *testing.T) {
	shape2 := backend.CompactForwardShape{Batch: 1, Tokens: 2, ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 2, OutputDim: 4}
	accel, cleanup := newBoundCompactTrainShapeTestAccelerator(t, shape2, true, false, compactForwardTestWeights(false))
	defer cleanup()
	refs := compactTrainResidentRefsForTest(t, accel, shape2)
	if err := accel.BeginCompactTrainStep(91, refs); err != nil {
		t.Fatalf("begin step: %v", err)
	}
	wantByName := map[string][]float32{}
	var lastRefs []backend.ResidentGradientRef
	for bucket, req := range []backend.CompactTrainForwardRequest{
		{
			Shape:        shape2,
			Tokens:       [][]int32{{2, 2}},
			Masks:        [][]int32{{1, 1}},
			Roles:        []int32{1},
			ResidentRefs: refs,
			GELUMode:     backend.CompactForwardGELUExact,
			StepID:       91,
		},
		{
			Shape:        backend.CompactForwardShape{Batch: 1, Tokens: 3, ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 2, OutputDim: 4},
			Tokens:       [][]int32{{3, 4, 3}},
			Masks:        [][]int32{{1, 0, 1}},
			Roles:        []int32{1},
			ResidentRefs: refs,
			GELUMode:     backend.CompactForwardGELUExact,
			StepID:       91,
		},
	} {
		forward, err := accel.RunCompactTrainForward(req)
		if err != nil {
			t.Fatalf("forward bucket %d: %v", bucket, err)
		}
		gradPooled := backend.NewTensorF32([]int{req.Shape.Batch, req.Shape.OutputDim}, seqData(req.Shape.Batch*req.Shape.OutputDim, 0.017+float64(bucket)*0.003, -0.029))
		got, err := accel.RunCompactTrainBackward(backend.CompactTrainBackwardRequest{Handle: forward.Handle, GradPooled: gradPooled})
		if err != nil {
			t.Fatalf("public backward bucket %d: %v", bucket, err)
		}
		lastRefs = got.ResidentGradRefs
		want := hostCompactTrainFullBackwardForCUDATest(req, true, false, gradPooled.F32)
		addCompactTrainWantByName(wantByName, want)
	}
	assertCompactTrainResidentGradientsClose(t, accel, lastRefs, wantByName, 3e-8)
	if err := accel.EndCompactTrainStep(91); err != nil {
		t.Fatalf("end step: %v", err)
	}
	stats := accel.CompactTrainStats()
	if stats.BackwardCalls != 2 || stats.GradPooledUploadedBytes != int64(2*shape2.OutputDim*4) || stats.LiveHandles != 0 {
		t.Fatalf("varying bucket stats = %+v", stats)
	}
}

func TestCompactTrainPublicBackwardActualShapeFullParity(t *testing.T) {
	shape := backend.CompactForwardShape{Batch: 1, Tokens: 75, ModelDim: 128, FFNDim: 512, Heads: 2, HeadDim: 64, Layers: 2, OutputDim: 128}
	weights := compactTrainShapeTestWeights(shape, false)
	accel, cleanup := newBoundCompactTrainShapeTestAccelerator(t, shape, true, false, weights)
	defer cleanup()
	refs := compactTrainResidentRefsForTest(t, accel, shape)
	if err := accel.BeginCompactTrainStep(101, refs); err != nil {
		t.Fatalf("begin step: %v", err)
	}
	tokens := make([]int32, shape.Tokens)
	masks := make([]int32, shape.Tokens)
	for i := range tokens {
		tokens[i] = int32(i%5 + 1)
		masks[i] = 1
		if i%11 == 7 {
			masks[i] = 0
		}
	}
	req := backend.CompactTrainForwardRequest{
		Shape:        shape,
		Tokens:       [][]int32{tokens},
		Masks:        [][]int32{masks},
		Roles:        []int32{0},
		ResidentRefs: refs,
		GELUMode:     backend.CompactForwardGELUFast,
		StepID:       101,
	}
	forward, err := accel.RunCompactTrainForward(req)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	gradPooled := backend.NewTensorF32([]int{shape.Batch, shape.OutputDim}, seqData(shape.Batch*shape.OutputDim, 0.00031, -0.00047))
	got, err := accel.RunCompactTrainBackward(backend.CompactTrainBackwardRequest{Handle: forward.Handle, GradPooled: gradPooled})
	if err != nil {
		t.Fatalf("public backward actual shape: %v", err)
	}
	assertResidentGradientRefsUniqueAndComplete(t, got.ResidentGradRefs, accel.requiredResidentNames(shape))
	if len(got.ResidentGradRefs) != 14 {
		t.Fatalf("resident grad ref count = %d, want 14", len(got.ResidentGradRefs))
	}
	want := hostCompactTrainFullBackwardWithWeightsForCUDATest(req, true, false, gradPooled.F32, weights)
	assertCompactTrainResidentGradientsClose(t, accel, got.ResidentGradRefs, want, 3e-8)
	assertCompactTrainBackwardWorkspaceDistinct(t, accel, shape)
	if err := accel.EndCompactTrainStep(101); err != nil {
		t.Fatalf("end step: %v", err)
	}
	assertCompactTrainResidentGradientsClose(t, accel, got.ResidentGradRefs, want, 3e-8)
	stats := accel.CompactTrainStats()
	wantBytes := int64(0)
	for _, ref := range refs {
		wantBytes += int64(ref.Elements * 4)
	}
	wantWorkspaceBytes := int64((shape.Batch*shape.OutputDim + shape.Batch*shape.Tokens*shape.OutputDim + 10*shape.Batch*shape.Tokens*shape.ModelDim + 2*shape.Batch*shape.Tokens*shape.FFNDim) * 4)
	wantUploadedBytes := int64((shape.Tokens + shape.Tokens + shape.Batch + 1 + shape.OutputDim) * 4)
	if stats.ResidentGradBytes != wantBytes || stats.WorkspaceArenaBytes != wantWorkspaceBytes || stats.UploadedBytes != wantUploadedBytes || stats.GradPooledUploadedBytes != int64(shape.OutputDim*4) || stats.LiveHandles != 0 || stats.BackwardCalls != 1 || stats.FallbackOrUnhandled != 0 {
		t.Fatalf("actual-shape stats = %+v want resident bytes %d", stats, wantBytes)
	}
	if base := accel.CompactForwardAccelerator.Stats(); base.PackedDownloads != 0 || base.PackedBytes != 0 {
		t.Fatalf("packed stats changed on compact train path: %+v", base)
	}
}

func TestCompactTrainHandleReleaseAfterCloseFailsClosed(t *testing.T) {
	accel, cleanup := newBoundCompactTrainTestAccelerator(t, false, false)
	defer cleanup()
	shape := backend.CompactForwardShape{Batch: 1, Tokens: 2, ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 2, OutputDim: 4}
	refs := compactTrainResidentRefsForTest(t, accel, shape)
	if err := accel.BeginCompactTrainStep(19, refs); err != nil {
		t.Fatalf("begin step: %v", err)
	}
	req := backend.CompactTrainForwardRequest{
		Shape:        shape,
		Tokens:       [][]int32{{2, 1}},
		Masks:        [][]int32{{1, 1}},
		Roles:        []int32{0},
		ResidentRefs: refs,
		StepID:       19,
	}
	result, err := accel.RunCompactTrainForward(req)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	accel.Close()
	if result.Handle.Token.Alive() {
		t.Fatal("handle token alive after accelerator close")
	}
	if err := accel.ReleaseCompactTrainHandle(result.Handle); err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("release after close err = %v, want closed", err)
	}
}

func newBoundCompactTrainTestAccelerator(t *testing.T, rope, projection bool) (*CompactTrainAccelerator, func()) {
	t.Helper()
	shape := backend.CompactForwardShape{ModelDim: 4, FFNDim: 5, Layers: 2, OutputDim: 4}
	if projection {
		shape.OutputDim = 3
		shape.HasOutputProjection = true
	}
	return newBoundCompactTrainShapeTestAccelerator(t, shape, rope, projection, compactForwardTestWeights(projection))
}

func newBoundCompactTrainShapeTestAccelerator(t *testing.T, shape backend.CompactForwardShape, rope, projection bool, weights map[string]*backend.Tensor) (*CompactTrainAccelerator, func()) {
	t.Helper()
	optAny, err := NewOptimizerAccelerator()
	if err != nil {
		t.Fatalf("new optimizer accelerator: %v", err)
	}
	opt := optAny.(*optimizerAccelerator)
	accel, err := NewCompactTrainAccelerator()
	if err != nil {
		opt.Close()
		t.Fatalf("new compact train accelerator: %v", err)
	}
	layers := make([]CompactForwardLayerNames, shape.Layers)
	for layer := range layers {
		prefix := fmt.Sprintf("layer%d_", layer)
		layers[layer] = CompactForwardLayerNames{
			AttentionQ: prefix + "attn_q",
			AttentionK: prefix + "attn_k",
			AttentionV: prefix + "attn_v",
			AttentionO: prefix + "attn_o",
			FFNUp:      prefix + "ffn_up",
			FFNDown:    prefix + "ffn_down",
		}
	}
	outName := ""
	if projection {
		outName = "output_projection"
	}
	accel.Configure(layers, "token_embedding", "role_embedding", outName, rope)
	for name, tensor := range weights {
		zero := backend.NewTensorF32(tensor.Shape, make([]float32, len(tensor.F32)))
		cfg := backend.OptimizerUpdateConfig{Optimizer: "sgd", LearningRate: 0, Scale: 1, DeferSync: true}
		if err := opt.ApplyUpdate(name, cfg, tensor.Clone(), nil, nil, zero); err != nil {
			accel.Close()
			opt.Close()
			t.Fatalf("make resident %s: %v", name, err)
		}
		ref, ok := opt.ResidentParameter(name)
		if !ok {
			accel.Close()
			opt.Close()
			t.Fatalf("resident %s missing", name)
		}
		if err := accel.BindCompactTrainResident(name, tensor, ref); err != nil {
			accel.Close()
			opt.Close()
			t.Fatalf("bind resident %s: %v", name, err)
		}
	}
	return accel, func() {
		accel.Close()
		opt.Close()
	}
}

func compactTrainResidentRefsForTest(t *testing.T, accel *CompactTrainAccelerator, shape backend.CompactForwardShape) []backend.CompactForwardResidentRef {
	t.Helper()
	names := accel.requiredResidentNames(shape)
	refs := make([]backend.CompactForwardResidentRef, 0, len(names))
	for _, name := range names {
		token := accel.bridged[name]
		if token == nil {
			t.Fatalf("missing test resident token %q", name)
		}
		binding, ok := accel.bindingForName(name)
		if !ok {
			t.Fatalf("missing test resident binding %q", name)
		}
		refs = append(refs, backend.CompactForwardResidentRef{
			Name:     name,
			Backend:  accel.Backend(),
			Token:    token,
			Elements: binding.elements,
		})
	}
	return refs
}

func compactTrainExpectedArenaBytes(shape backend.CompactForwardShape) int64 {
	B, T, D, H := shape.Batch, shape.Tokens, shape.ModelDim, shape.FFNDim
	rows := B * T
	modelElems := rows * D
	ffnElems := rows * H
	scoreElems := B * shape.Heads * T * T
	total := modelElems + B + modelElems + B*shape.OutputDim + 2*modelElems
	if shape.HasOutputProjection {
		total += B*D + rows*shape.OutputDim
	}
	total += shape.Layers * (8*modelElems + scoreElems + 2*ffnElems)
	return int64(total * 4)
}

func compactTrainShapeTestWeights(shape backend.CompactForwardShape, projection bool) map[string]*backend.Tensor {
	const vocab = 6
	const roles = 3
	weights := map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{vocab, shape.ModelDim}, seqData(vocab*shape.ModelDim, 0.0007, -0.03)),
		"role_embedding":  backend.NewTensorF32([]int{roles, shape.ModelDim}, seqData(roles*shape.ModelDim, 0.0003, -0.01)),
	}
	for layer := 0; layer < shape.Layers; layer++ {
		prefix := fmt.Sprintf("layer%d_", layer)
		weights[prefix+"attn_q"] = backend.NewTensorF32([]int{shape.ModelDim, shape.ModelDim}, seqData(shape.ModelDim*shape.ModelDim, 0.00021+float64(layer)*0.00003, -0.012))
		weights[prefix+"attn_k"] = backend.NewTensorF32([]int{shape.ModelDim, shape.ModelDim}, seqData(shape.ModelDim*shape.ModelDim, -0.00017-float64(layer)*0.00002, 0.011))
		weights[prefix+"attn_v"] = backend.NewTensorF32([]int{shape.ModelDim, shape.ModelDim}, seqData(shape.ModelDim*shape.ModelDim, 0.00013+float64(layer)*0.00002, -0.008))
		weights[prefix+"attn_o"] = backend.NewTensorF32([]int{shape.ModelDim, shape.ModelDim}, seqData(shape.ModelDim*shape.ModelDim, -0.00019, 0.009+float64(layer)*0.001))
		weights[prefix+"ffn_up"] = backend.NewTensorF32([]int{shape.ModelDim, shape.FFNDim}, seqData(shape.ModelDim*shape.FFNDim, 0.00015+float64(layer)*0.00001, -0.005))
		weights[prefix+"ffn_down"] = backend.NewTensorF32([]int{shape.FFNDim, shape.ModelDim}, seqData(shape.FFNDim*shape.ModelDim, -0.00014, 0.006+float64(layer)*0.0005))
	}
	if projection {
		weights["output_projection"] = backend.NewTensorF32([]int{shape.ModelDim, shape.OutputDim}, seqData(shape.ModelDim*shape.OutputDim, 0.00025, -0.01))
	}
	return weights
}

func hostCompactTrainPooledForCUDATest(req backend.CompactTrainForwardRequest, weights map[string]*backend.Tensor) []float32 {
	shape := req.Shape
	out := make([]float32, 0, shape.Batch*shape.OutputDim)
	for b := 0; b < shape.Batch; b++ {
		current := gatherHost(weights["token_embedding"], weights["role_embedding"], req.Tokens[b], req.Roles[b], false)
		for layer := 0; layer < shape.Layers; layer++ {
			prefix := fmt.Sprintf("layer%d_", layer)
			q := matmulHost(current, shape.Tokens, shape.ModelDim, weights[prefix+"attn_q"].F32, shape.ModelDim)
			k := matmulHost(current, shape.Tokens, shape.ModelDim, weights[prefix+"attn_k"].F32, shape.ModelDim)
			v := matmulHost(current, shape.Tokens, shape.ModelDim, weights[prefix+"attn_v"].F32, shape.ModelDim)
			_, mixed := attentionHost(q, k, v, req.Masks[b], shape)
			attnOut := matmulHost(mixed, shape.Tokens, shape.ModelDim, weights[prefix+"attn_o"].F32, shape.ModelDim)
			hidden := layerNormHost(addHost(attnOut, current), shape.Tokens, shape.ModelDim)
			ffnHidden := matmulHost(hidden, shape.Tokens, shape.ModelDim, weights[prefix+"ffn_up"].F32, shape.FFNDim)
			activated := geluHost(ffnHidden, req.GELUMode)
			ffnOut := matmulHost(activated, shape.Tokens, shape.FFNDim, weights[prefix+"ffn_down"].F32, shape.ModelDim)
			current = layerNormHost(addHost(ffnOut, hidden), shape.Tokens, shape.ModelDim)
		}
		outputRows := normalizeRowsHost(current, shape.Tokens, shape.ModelDim)
		if shape.HasOutputProjection {
			outputRows = matmulHost(outputRows, shape.Tokens, shape.ModelDim, weights["output_projection"].F32, shape.OutputDim)
		}
		out = append(out, poolHost(outputRows, req.Masks[b], shape.Tokens, shape.OutputDim)...)
	}
	return out
}

func hostCompactForwardWithWeightsForCUDATest(req backend.CompactForwardRequest, rope, projection bool, weights map[string]*backend.Tensor) backend.CompactForwardResult {
	shape := req.Shape
	layout, total, _ := compactForwardPackedLayout(shape)
	data := make([]float32, total)
	put := func(name string, values []float32) {
		span := compactForwardSpanByName(layout, name)
		copy(data[span.Offset:span.Offset+span.Len], values)
	}
	for b := 0; b < shape.Batch; b++ {
		current := gatherHost(weights["token_embedding"], weights["role_embedding"], req.Tokens[b], req.Roles[b], rope)
		for layer := 0; layer < shape.Layers; layer++ {
			prefix := fmt.Sprintf("layer%d_", layer)
			input := append([]float32(nil), current...)
			q := matmulHost(input, shape.Tokens, shape.ModelDim, weights[prefix+"attn_q"].F32, shape.ModelDim)
			k := matmulHost(input, shape.Tokens, shape.ModelDim, weights[prefix+"attn_k"].F32, shape.ModelDim)
			v := matmulHost(input, shape.Tokens, shape.ModelDim, weights[prefix+"attn_v"].F32, shape.ModelDim)
			scores, mixed := attentionHost(q, k, v, req.Masks[b], shape)
			attnOut := matmulHost(mixed, shape.Tokens, shape.ModelDim, weights[prefix+"attn_o"].F32, shape.ModelDim)
			attnResidual := addHost(attnOut, input)
			hidden := layerNormHost(attnResidual, shape.Tokens, shape.ModelDim)
			ffnHidden := matmulHost(hidden, shape.Tokens, shape.ModelDim, weights[prefix+"ffn_up"].F32, shape.FFNDim)
			activated := geluHost(ffnHidden, req.GELUMode)
			ffnOut := matmulHost(activated, shape.Tokens, shape.FFNDim, weights[prefix+"ffn_down"].F32, shape.ModelDim)
			ffnResidual := addHost(ffnOut, hidden)
			projected := layerNormHost(ffnResidual, shape.Tokens, shape.ModelDim)
			normalized := normalizeRowsHost(projected, shape.Tokens, shape.ModelDim)
			pooled := poolHost(normalized, req.Masks[b], shape.Tokens, shape.ModelDim)
			put(compactForwardLayerSpanName(b, layer, "activeCount"), []float32{float32(activeHost(req.Masks[b]))})
			for _, item := range []struct {
				name string
				vals []float32
			}{{"input", input}, {"hidden", hidden}, {"attnQ", q}, {"attnK", k}, {"attnV", v}, {"attnScores", scores}, {"attnMixed", mixed}, {"attnOutput", attnOut}, {"attnResidual", attnResidual}, {"ffnHidden", ffnHidden}, {"activated", activated}, {"ffnOutput", ffnOut}, {"ffnResidual", ffnResidual}, {"projected", projected}, {"normalized", normalized}, {"pooled", pooled}} {
				if item.name == "pooled" && layer == shape.Layers-1 && shape.OutputDim != shape.ModelDim {
					continue
				}
				put(compactForwardLayerSpanName(b, layer, item.name), item.vals)
			}
			current = projected
		}
		finalNorm := normalizeRowsHost(current, shape.Tokens, shape.ModelDim)
		outputRows := finalNorm
		if projection {
			outputRows = matmulHost(finalNorm, shape.Tokens, shape.ModelDim, weights["output_projection"].F32, shape.OutputDim)
		}
		finalPooled := poolHost(outputRows, req.Masks[b], shape.Tokens, shape.OutputDim)
		if shape.OutputDim != shape.ModelDim {
			put(compactForwardLayerSpanName(b, shape.Layers-1, "pooled"), finalPooled)
		}
		put(compactForwardSequenceSpanName(b, "final.normalized"), finalNorm)
		put(compactForwardSequenceSpanName(b, "final.outputRows"), outputRows)
		put(compactForwardSequenceSpanName(b, "final.pooled"), finalPooled)
	}
	return backend.CompactForwardResult{Layout: layout, Data: data}
}

func hostCompactTrainFinalOutputBackwardForCUDATest(req backend.CompactTrainForwardRequest, rope, projection bool, gradPooled []float32) ([]float32, []float32) {
	return hostCompactTrainFinalOutputBackwardWithWeightsForCUDATest(req, rope, projection, gradPooled, compactForwardTestWeights(projection))
}

func hostCompactTrainFinalOutputBackwardWithWeightsForCUDATest(req backend.CompactTrainForwardRequest, rope, projection bool, gradPooled []float32, weights map[string]*backend.Tensor) ([]float32, []float32) {
	packedReq := backend.CompactForwardRequest{
		Shape:        req.Shape,
		Tokens:       req.Tokens,
		Masks:        req.Masks,
		Roles:        req.Roles,
		ResidentRefs: req.ResidentRefs,
		GELUMode:     req.GELUMode,
	}
	packed := hostCompactForwardWithWeightsForCUDATest(packedReq, rope, projection, weights)
	shape := req.Shape
	hidden := make([]float32, shape.Batch*shape.Tokens*shape.ModelDim)
	var gradProjection []float32
	if shape.HasOutputProjection {
		gradProjection = make([]float32, shape.ModelDim*shape.OutputDim)
	}
	for b := 0; b < shape.Batch; b++ {
		projectedSpan := compactForwardSpanByName(packed.Layout, compactForwardLayerSpanName(b, shape.Layers-1, "projected"))
		normalizedSpan := compactForwardSpanByName(packed.Layout, compactForwardSequenceSpanName(b, "final.normalized"))
		projected := packed.Data[projectedSpan.Offset : projectedSpan.Offset+projectedSpan.Len]
		normalized := packed.Data[normalizedSpan.Offset : normalizedSpan.Offset+normalizedSpan.Len]
		active := 0
		for _, m := range req.Masks[b] {
			if m != 0 {
				active++
			}
		}
		invActive := float32(1) / float32(active)
		gradRows := make([]float32, shape.Tokens*shape.OutputDim)
		for row := 0; row < shape.Tokens; row++ {
			if req.Masks[b][row] == 0 {
				continue
			}
			for col := 0; col < shape.OutputDim; col++ {
				gradRows[row*shape.OutputDim+col] = gradPooled[b*shape.OutputDim+col] * invActive
			}
		}
		gradNorm := gradRows
		if shape.HasOutputProjection {
			gradNorm = make([]float32, shape.Tokens*shape.ModelDim)
			proj := weights["output_projection"].F32
			for d := 0; d < shape.ModelDim; d++ {
				for o := 0; o < shape.OutputDim; o++ {
					sum := float32(0)
					for row := 0; row < shape.Tokens; row++ {
						sum += normalized[row*shape.ModelDim+d] * gradRows[row*shape.OutputDim+o]
						gradNorm[row*shape.ModelDim+d] += gradRows[row*shape.OutputDim+o] * proj[d*shape.OutputDim+o]
					}
					gradProjection[d*shape.OutputDim+o] += sum
				}
			}
		}
		for row := 0; row < shape.Tokens; row++ {
			base := row * shape.ModelDim
			normSq := float32(0)
			for col := 0; col < shape.ModelDim; col++ {
				normSq += projected[base+col] * projected[base+col]
			}
			norm := float32(math.Sqrt(float64(normSq)))
			if norm == 0 {
				continue
			}
			dotNG := float32(0)
			for col := 0; col < shape.ModelDim; col++ {
				dotNG += normalized[base+col] * gradNorm[base+col]
			}
			for col := 0; col < shape.ModelDim; col++ {
				hidden[(b*shape.Tokens+row)*shape.ModelDim+col] = (gradNorm[base+col] - normalized[base+col]*dotNG) / norm
			}
		}
	}
	return hidden, gradProjection
}

type compactTrainTopFFNBackwardWant struct {
	gradHidden    []float32
	gradAttention []float32
	ffnUp         []float32
	ffnDown       []float32
}

type compactTrainTopLayerBackwardWant struct {
	gradLayerInput []float32
	gradRoPEInput  []float32
	attnQ          []float32
	attnK          []float32
	attnV          []float32
	attnO          []float32
	ffnUp          []float32
	ffnDown        []float32
}

func hostCompactTrainFullBackwardForCUDATest(req backend.CompactTrainForwardRequest, rope, projection bool, gradPooled []float32) map[string][]float32 {
	return hostCompactTrainFullBackwardWithWeightsForCUDATest(req, rope, projection, gradPooled, compactForwardTestWeights(projection))
}

func hostCompactTrainFullBackwardWithWeightsForCUDATest(req backend.CompactTrainForwardRequest, rope, projection bool, gradPooled []float32, weights map[string]*backend.Tensor) map[string][]float32 {
	packedReq := backend.CompactForwardRequest{
		Shape:        req.Shape,
		Tokens:       req.Tokens,
		Masks:        req.Masks,
		Roles:        req.Roles,
		ResidentRefs: req.ResidentRefs,
		GELUMode:     req.GELUMode,
	}
	packed := hostCompactForwardWithWeightsForCUDATest(packedReq, rope, projection, weights)
	shape := req.Shape
	want := map[string][]float32{
		"token_embedding": make([]float32, len(weights["token_embedding"].F32)),
		"role_embedding":  make([]float32, len(weights["role_embedding"].F32)),
	}
	for layer := 0; layer < shape.Layers; layer++ {
		prefix := fmt.Sprintf("layer%d_", layer)
		want[prefix+"attn_q"] = make([]float32, shape.ModelDim*shape.ModelDim)
		want[prefix+"attn_k"] = make([]float32, shape.ModelDim*shape.ModelDim)
		want[prefix+"attn_v"] = make([]float32, shape.ModelDim*shape.ModelDim)
		want[prefix+"attn_o"] = make([]float32, shape.ModelDim*shape.ModelDim)
		want[prefix+"ffn_up"] = make([]float32, shape.ModelDim*shape.FFNDim)
		want[prefix+"ffn_down"] = make([]float32, shape.FFNDim*shape.ModelDim)
	}
	gradCurrent, gradProjection := hostCompactTrainFinalOutputBackwardWithWeightsForCUDATest(req, rope, projection, gradPooled, weights)
	if shape.HasOutputProjection {
		want["output_projection"] = gradProjection
	}
	for layerIdx := shape.Layers - 1; layerIdx >= 0; layerIdx-- {
		next := make([]float32, shape.Batch*shape.Tokens*shape.ModelDim)
		prefix := fmt.Sprintf("layer%d_", layerIdx)
		attnQW := weights[prefix+"attn_q"].F32
		attnKW := weights[prefix+"attn_k"].F32
		attnVW := weights[prefix+"attn_v"].F32
		attnOW := weights[prefix+"attn_o"].F32
		up := weights[prefix+"ffn_up"].F32
		down := weights[prefix+"ffn_down"].F32
		for b := 0; b < shape.Batch; b++ {
			span := func(name string) []float32 {
				s := compactForwardSpanByName(packed.Layout, compactForwardLayerSpanName(b, layerIdx, name))
				return packed.Data[s.Offset : s.Offset+s.Len]
			}
			input := span("input")
			hidden := span("hidden")
			attnQ := span("attnQ")
			attnK := span("attnK")
			attnV := span("attnV")
			attnMixed := span("attnMixed")
			attnScores := span("attnScores")
			attnResidual := span("attnResidual")
			ffnHidden := span("ffnHidden")
			activated := span("activated")
			ffnResidual := span("ffnResidual")
			projected := span("projected")
			base := b * shape.Tokens * shape.ModelDim
			gradProjected := gradCurrent[base : base+shape.Tokens*shape.ModelDim]
			gradFFNResidual := make([]float32, shape.Tokens*shape.ModelDim)
			for row := 0; row < shape.Tokens; row++ {
				rowBase := row * shape.ModelDim
				compactTrainBackwardLayerNormRow(
					gradFFNResidual[rowBase:rowBase+shape.ModelDim],
					gradProjected[rowBase:rowBase+shape.ModelDim],
					projected[rowBase:rowBase+shape.ModelDim],
					ffnResidual[rowBase:rowBase+shape.ModelDim],
				)
			}
			compactTrainMatMulLeftTransposeAccum(activated, gradFFNResidual, want[prefix+"ffn_down"], shape.Tokens, shape.FFNDim, shape.ModelDim)
			gradActivatedPre := compactTrainMatMulRightTranspose(gradFFNResidual, down, shape.Tokens, shape.FFNDim, shape.ModelDim)
			gradActivated := make([]float32, shape.Tokens*shape.FFNDim)
			compactTrainFillGELUBackwardMul(gradActivated, gradActivatedPre, ffnHidden, req.GELUMode == backend.CompactForwardGELUFast)
			compactTrainMatMulLeftTransposeAccum(hidden, gradActivated, want[prefix+"ffn_up"], shape.Tokens, shape.ModelDim, shape.FFNDim)
			gradHiddenFromFFN := compactTrainMatMulRightTranspose(gradActivated, up, shape.Tokens, shape.ModelDim, shape.FFNDim)
			gradHidden := make([]float32, shape.Tokens*shape.ModelDim)
			for i := range gradHidden {
				gradHidden[i] = gradFFNResidual[i] + gradHiddenFromFFN[i]
			}
			gradAttention := make([]float32, shape.Tokens*shape.ModelDim)
			for row := 0; row < shape.Tokens; row++ {
				rowBase := row * shape.ModelDim
				compactTrainBackwardLayerNormRow(
					gradAttention[rowBase:rowBase+shape.ModelDim],
					gradHidden[rowBase:rowBase+shape.ModelDim],
					hidden[rowBase:rowBase+shape.ModelDim],
					attnResidual[rowBase:rowBase+shape.ModelDim],
				)
			}
			compactTrainMatMulLeftTransposeAccum(attnMixed, gradAttention, want[prefix+"attn_o"], shape.Tokens, shape.ModelDim, shape.ModelDim)
			gradMixed := compactTrainMatMulRightTranspose(gradAttention, attnOW, shape.Tokens, shape.ModelDim, shape.ModelDim)
			gradQ, gradK, gradV := compactTrainAttentionBackwardHost(gradMixed, attnQ, attnK, attnV, attnScores, backend.CompactForwardShape{
				Batch:    1,
				Tokens:   shape.Tokens,
				ModelDim: shape.ModelDim,
				Heads:    shape.Heads,
				HeadDim:  shape.HeadDim,
			})
			compactTrainMatMulLeftTransposeAccum(input, gradQ, want[prefix+"attn_q"], shape.Tokens, shape.ModelDim, shape.ModelDim)
			compactTrainMatMulLeftTransposeAccum(input, gradK, want[prefix+"attn_k"], shape.Tokens, shape.ModelDim, shape.ModelDim)
			compactTrainMatMulLeftTransposeAccum(input, gradV, want[prefix+"attn_v"], shape.Tokens, shape.ModelDim, shape.ModelDim)
			gradInputQ := compactTrainMatMulRightTranspose(gradQ, attnQW, shape.Tokens, shape.ModelDim, shape.ModelDim)
			gradInputK := compactTrainMatMulRightTranspose(gradK, attnKW, shape.Tokens, shape.ModelDim, shape.ModelDim)
			gradInputV := compactTrainMatMulRightTranspose(gradV, attnVW, shape.Tokens, shape.ModelDim, shape.ModelDim)
			for i := range gradAttention {
				next[base+i] = ((gradAttention[i] + gradInputQ[i]) + gradInputK[i]) + gradInputV[i]
			}
		}
		gradCurrent = next
	}
	if rope {
		for b := 0; b < shape.Batch; b++ {
			base := b * shape.Tokens * shape.ModelDim
			applyProductionRoPETransposeHost(gradCurrent[base:base+shape.Tokens*shape.ModelDim], shape.Tokens, shape.ModelDim)
		}
	}
	vocab := weights["token_embedding"].Shape[0]
	roleRows := weights["role_embedding"].Shape[0]
	for b := 0; b < shape.Batch; b++ {
		for row, tok := range req.Tokens[b] {
			if tok >= 0 && int(tok) < vocab {
				srcBase := (b*shape.Tokens + row) * shape.ModelDim
				dstBase := int(tok) * shape.ModelDim
				for col := 0; col < shape.ModelDim; col++ {
					want["token_embedding"][dstBase+col] += gradCurrent[srcBase+col]
				}
			}
		}
		role := req.Roles[b]
		if role >= 0 && int(role) < roleRows {
			dstBase := int(role) * shape.ModelDim
			for row := 0; row < shape.Tokens; row++ {
				srcBase := (b*shape.Tokens + row) * shape.ModelDim
				for col := 0; col < shape.ModelDim; col++ {
					want["role_embedding"][dstBase+col] += gradCurrent[srcBase+col]
				}
			}
		}
	}
	return want
}

func hostCompactTrainTopFFNBackwardForCUDATest(req backend.CompactTrainForwardRequest, rope, projection bool, gradPooled []float32) compactTrainTopFFNBackwardWant {
	packedReq := backend.CompactForwardRequest{
		Shape:        req.Shape,
		Tokens:       req.Tokens,
		Masks:        req.Masks,
		Roles:        req.Roles,
		ResidentRefs: req.ResidentRefs,
		GELUMode:     req.GELUMode,
	}
	packed := hostCompactForwardForCUDATest(packedReq, rope, projection)
	shape := req.Shape
	gradProjected, _ := hostCompactTrainFinalOutputBackwardForCUDATest(req, rope, projection, gradPooled)
	layerIdx := shape.Layers - 1
	out := compactTrainTopFFNBackwardWant{
		gradHidden:    make([]float32, shape.Batch*shape.Tokens*shape.ModelDim),
		gradAttention: make([]float32, shape.Batch*shape.Tokens*shape.ModelDim),
		ffnUp:         make([]float32, shape.ModelDim*shape.FFNDim),
		ffnDown:       make([]float32, shape.FFNDim*shape.ModelDim),
	}
	weights := compactForwardTestWeights(projection)
	up := weights[fmt.Sprintf("layer%d_ffn_up", layerIdx)].F32
	down := weights[fmt.Sprintf("layer%d_ffn_down", layerIdx)].F32
	for b := 0; b < shape.Batch; b++ {
		span := func(name string) []float32 {
			s := compactForwardSpanByName(packed.Layout, compactForwardLayerSpanName(b, layerIdx, name))
			return packed.Data[s.Offset : s.Offset+s.Len]
		}
		projected := span("projected")
		ffnResidual := span("ffnResidual")
		activated := span("activated")
		ffnHidden := span("ffnHidden")
		hidden := span("hidden")
		attnResidual := span("attnResidual")
		gradBase := b * shape.Tokens * shape.ModelDim
		gradFFNResidual := make([]float32, shape.Tokens*shape.ModelDim)
		for row := 0; row < shape.Tokens; row++ {
			base := row * shape.ModelDim
			compactTrainBackwardLayerNormRow(
				gradFFNResidual[base:base+shape.ModelDim],
				gradProjected[gradBase+base:gradBase+base+shape.ModelDim],
				projected[base:base+shape.ModelDim],
				ffnResidual[base:base+shape.ModelDim],
			)
		}
		compactTrainMatMulLeftTransposeAccum(activated, gradFFNResidual, out.ffnDown, shape.Tokens, shape.FFNDim, shape.ModelDim)
		gradActivatedPre := compactTrainMatMulRightTranspose(gradFFNResidual, down, shape.Tokens, shape.FFNDim, shape.ModelDim)
		gradActivated := make([]float32, shape.Tokens*shape.FFNDim)
		compactTrainFillGELUBackwardMul(gradActivated, gradActivatedPre, ffnHidden, req.GELUMode == backend.CompactForwardGELUFast)
		compactTrainMatMulLeftTransposeAccum(hidden, gradActivated, out.ffnUp, shape.Tokens, shape.ModelDim, shape.FFNDim)
		gradHiddenFromFFN := compactTrainMatMulRightTranspose(gradActivated, up, shape.Tokens, shape.ModelDim, shape.FFNDim)
		gradHidden := make([]float32, shape.Tokens*shape.ModelDim)
		for i := range gradHidden {
			gradHidden[i] = gradFFNResidual[i] + gradHiddenFromFFN[i]
			out.gradHidden[gradBase+i] = gradHidden[i]
		}
		for row := 0; row < shape.Tokens; row++ {
			base := row * shape.ModelDim
			compactTrainBackwardLayerNormRow(
				out.gradAttention[gradBase+base:gradBase+base+shape.ModelDim],
				gradHidden[base:base+shape.ModelDim],
				hidden[base:base+shape.ModelDim],
				attnResidual[base:base+shape.ModelDim],
			)
		}
	}
	return out
}

func hostCompactTrainTopLayerBackwardForCUDATest(req backend.CompactTrainForwardRequest, rope, projection bool, gradPooled []float32) compactTrainTopLayerBackwardWant {
	packedReq := backend.CompactForwardRequest{
		Shape:        req.Shape,
		Tokens:       req.Tokens,
		Masks:        req.Masks,
		Roles:        req.Roles,
		ResidentRefs: req.ResidentRefs,
		GELUMode:     req.GELUMode,
	}
	packed := hostCompactForwardForCUDATest(packedReq, rope, projection)
	shape := req.Shape
	layerIdx := shape.Layers - 1
	ffn := hostCompactTrainTopFFNBackwardForCUDATest(req, rope, projection, gradPooled)
	out := compactTrainTopLayerBackwardWant{
		gradLayerInput: make([]float32, shape.Batch*shape.Tokens*shape.ModelDim),
		attnQ:          make([]float32, shape.ModelDim*shape.ModelDim),
		attnK:          make([]float32, shape.ModelDim*shape.ModelDim),
		attnV:          make([]float32, shape.ModelDim*shape.ModelDim),
		attnO:          make([]float32, shape.ModelDim*shape.ModelDim),
		ffnUp:          ffn.ffnUp,
		ffnDown:        ffn.ffnDown,
	}
	weights := compactForwardTestWeights(projection)
	prefix := fmt.Sprintf("layer%d_", layerIdx)
	attnQW := weights[prefix+"attn_q"].F32
	attnKW := weights[prefix+"attn_k"].F32
	attnVW := weights[prefix+"attn_v"].F32
	attnOW := weights[prefix+"attn_o"].F32
	for b := 0; b < shape.Batch; b++ {
		span := func(name string) []float32 {
			s := compactForwardSpanByName(packed.Layout, compactForwardLayerSpanName(b, layerIdx, name))
			return packed.Data[s.Offset : s.Offset+s.Len]
		}
		input := span("input")
		attnQ := span("attnQ")
		attnK := span("attnK")
		attnV := span("attnV")
		attnMixed := span("attnMixed")
		scores := span("attnScores")
		gradBase := b * shape.Tokens * shape.ModelDim
		gradAttention := ffn.gradAttention[gradBase : gradBase+shape.Tokens*shape.ModelDim]
		compactTrainMatMulLeftTransposeAccum(attnMixed, gradAttention, out.attnO, shape.Tokens, shape.ModelDim, shape.ModelDim)
		gradMixed := compactTrainMatMulRightTranspose(gradAttention, attnOW, shape.Tokens, shape.ModelDim, shape.ModelDim)
		gradQ, gradK, gradV := compactTrainAttentionBackwardHost(gradMixed, attnQ, attnK, attnV, scores, backend.CompactForwardShape{
			Batch:    1,
			Tokens:   shape.Tokens,
			ModelDim: shape.ModelDim,
			Heads:    shape.Heads,
			HeadDim:  shape.HeadDim,
		})
		compactTrainMatMulLeftTransposeAccum(input, gradQ, out.attnQ, shape.Tokens, shape.ModelDim, shape.ModelDim)
		compactTrainMatMulLeftTransposeAccum(input, gradK, out.attnK, shape.Tokens, shape.ModelDim, shape.ModelDim)
		compactTrainMatMulLeftTransposeAccum(input, gradV, out.attnV, shape.Tokens, shape.ModelDim, shape.ModelDim)
		gradInputQ := compactTrainMatMulRightTranspose(gradQ, attnQW, shape.Tokens, shape.ModelDim, shape.ModelDim)
		gradInputK := compactTrainMatMulRightTranspose(gradK, attnKW, shape.Tokens, shape.ModelDim, shape.ModelDim)
		gradInputV := compactTrainMatMulRightTranspose(gradV, attnVW, shape.Tokens, shape.ModelDim, shape.ModelDim)
		for i := range gradAttention {
			out.gradLayerInput[gradBase+i] = ((gradAttention[i] + gradInputQ[i]) + gradInputK[i]) + gradInputV[i]
		}
	}
	if layerIdx == 0 && rope {
		out.gradRoPEInput = append([]float32(nil), out.gradLayerInput...)
		for b := 0; b < shape.Batch; b++ {
			base := b * shape.Tokens * shape.ModelDim
			applyProductionRoPETransposeHost(out.gradRoPEInput[base:base+shape.Tokens*shape.ModelDim], shape.Tokens, shape.ModelDim)
		}
	}
	return out
}

func compactTrainAttentionBackwardHost(gradMixed, q, k, v, probs []float32, shape backend.CompactForwardShape) ([]float32, []float32, []float32) {
	gradQ := make([]float32, len(q))
	gradK := make([]float32, len(k))
	gradV := make([]float32, len(v))
	scale := float32(1 / math.Sqrt(float64(shape.HeadDim)))
	for b := 0; b < shape.Batch; b++ {
		baseRows := b * shape.Tokens
		for head := 0; head < shape.Heads; head++ {
			headOffset := head * shape.HeadDim
			scoreHeadBase := (b*shape.Heads + head) * shape.Tokens * shape.Tokens
			for query := 0; query < shape.Tokens; query++ {
				queryBase := (baseRows+query)*shape.ModelDim + headOffset
				scoreRowBase := scoreHeadBase + query*shape.Tokens
				dot := float32(0)
				for key := 0; key < shape.Tokens; key++ {
					keyBase := (baseRows+key)*shape.ModelDim + headOffset
					sum := float32(0)
					for col := 0; col < shape.HeadDim; col++ {
						sum += gradMixed[queryBase+col] * v[keyBase+col]
					}
					dot += sum * probs[scoreRowBase+key]
				}
				for key := 0; key < shape.Tokens; key++ {
					keyBase := (baseRows+key)*shape.ModelDim + headOffset
					sum := float32(0)
					for col := 0; col < shape.HeadDim; col++ {
						sum += gradMixed[queryBase+col] * v[keyBase+col]
					}
					prob := probs[scoreRowBase+key]
					preGrad := prob * (sum - dot) * scale
					for col := 0; col < shape.HeadDim; col++ {
						gradV[keyBase+col] += prob * gradMixed[queryBase+col]
						gradQ[queryBase+col] += preGrad * k[keyBase+col]
						gradK[keyBase+col] += preGrad * q[queryBase+col]
					}
				}
			}
		}
	}
	return gradQ, gradK, gradV
}

func applyProductionRoPETransposeHost(data []float32, rows, cols int) {
	for row := 0; row < rows; row++ {
		base := row * cols
		for col := 0; col+1 < cols; col += 2 {
			theta := float64(row) / math.Pow(10000, float64(col)/float64(cols))
			c, s := float32(math.Cos(theta)), float32(math.Sin(theta))
			x0, x1 := data[base+col], data[base+col+1]
			data[base+col] = x0*c + x1*s
			data[base+col+1] = -x0*s + x1*c
		}
	}
}

func compactTrainBackwardLayerNormRow(dst, gradOut, normalized, pre []float32) {
	mean := float32(0)
	for _, value := range pre {
		mean += value
	}
	mean /= float32(len(pre))
	variance := float32(0)
	for _, value := range pre {
		centered := value - mean
		variance += centered * centered
	}
	variance /= float32(len(pre))
	invStd := float32(1.0 / math.Sqrt(float64(variance)+1e-5))
	sumGrad := float32(0)
	sumGradNorm := float32(0)
	for i := range gradOut {
		sumGrad += gradOut[i]
		sumGradNorm += gradOut[i] * normalized[i]
	}
	n := float32(len(pre))
	for i := range gradOut {
		dst[i] = (invStd / n) * (n*gradOut[i] - sumGrad - normalized[i]*sumGradNorm)
	}
}

func compactTrainFillGELUBackwardMul(dst, gradOut, preAct []float32, fast bool) {
	for i, value := range preAct {
		inner := float32(0.7978845608) * (value + float32(0.044715)*value*value*value)
		t := float32(math.Tanh(float64(inner)))
		tanhGrad := float32(1) - t*t
		if fast {
			t = compactTrainFastTanh(inner)
			tanhGrad = compactTrainFastTanhDerivative(inner)
		}
		innerGrad := float32(0.7978845608) * (1 + float32(3*0.044715)*value*value)
		dst[i] = gradOut[i] * (0.5*(1+t) + 0.5*value*tanhGrad*innerGrad)
	}
}

func compactTrainFastTanh(inner float32) float32 {
	if inner >= 3 {
		return 1
	}
	if inner <= -3 {
		return -1
	}
	x2 := inner * inner
	return inner * (27 + x2) / (27 + 9*x2)
}

func compactTrainFastTanhDerivative(x float32) float32 {
	if x >= 3 || x <= -3 {
		return 0
	}
	x2 := x * x
	diff := x2 - 9
	den := 3 + x2
	return (diff * diff) / (9 * den * den)
}

func compactTrainMatMulLeftTransposeAccum(lhs, gradOut, gradWeight []float32, rows, inDim, outDim int) {
	for i := 0; i < inDim; i++ {
		for o := 0; o < outDim; o++ {
			sum := float32(0)
			for r := 0; r < rows; r++ {
				sum += lhs[r*inDim+i] * gradOut[r*outDim+o]
			}
			gradWeight[i*outDim+o] += sum
		}
	}
}

func compactTrainMatMulRightTranspose(gradOut, weight []float32, rows, inDim, outDim int) []float32 {
	gradIn := make([]float32, rows*inDim)
	for r := 0; r < rows; r++ {
		for i := 0; i < inDim; i++ {
			sum := float32(0)
			for o := 0; o < outDim; o++ {
				sum += gradOut[r*outDim+o] * weight[i*outDim+o]
			}
			gradIn[r*inDim+i] = sum
		}
	}
	return gradIn
}

func addFloat32SlicesForCUDATest(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

func addCompactTrainWantByName(dst, src map[string][]float32) {
	for name, values := range src {
		if dst[name] == nil {
			dst[name] = make([]float32, len(values))
		}
		addFloat32SlicesForCUDATest(dst[name], values)
	}
}

func assertResidentGradientRefsUniqueAndComplete(t *testing.T, refs []backend.ResidentGradientRef, wantNames []string) {
	t.Helper()
	seen := map[string]bool{}
	for _, ref := range refs {
		if ref.Name == "" {
			t.Fatal("resident grad ref with empty name")
		}
		if seen[ref.Name] {
			t.Fatalf("duplicate resident grad ref %q in %+v", ref.Name, refs)
		}
		seen[ref.Name] = true
	}
	if len(refs) != len(wantNames) {
		t.Fatalf("resident grad refs = %d, want %d: %+v", len(refs), len(wantNames), refs)
	}
	for _, name := range wantNames {
		if !seen[name] {
			t.Fatalf("resident grad ref %q missing from %+v", name, refs)
		}
	}
}

func assertCompactTrainResidentGradientsClose(t *testing.T, accel *CompactTrainAccelerator, refs []backend.ResidentGradientRef, want map[string][]float32, tol float32) {
	t.Helper()
	for name, wantValues := range want {
		ref := residentGradRefByName(t, refs, name)
		grad, err := accel.copyResidentGradientForDebug(ref)
		if err != nil {
			t.Fatalf("copy resident grad %s: %v", name, err)
		}
		t.Logf("%s grad max_abs=%g rmse=%g", name, compactTrainMaxAbs(grad.F32, wantValues), compactTrainRMSE(grad.F32, wantValues))
		assertFloatSlicesClose(t, grad.F32, wantValues, tol)
	}
}

func assertCompactTrainBackwardWorkspaceDistinct(t *testing.T, accel *CompactTrainAccelerator, shape backend.CompactForwardShape) {
	t.Helper()
	if accel.arena == nil {
		t.Fatal("missing compact train arena")
	}
	arena := accel.arena
	ptrs := map[uintptr]string{}
	for name, ptr := range map[string]uintptr{
		"gradPooled":        uintptr(arena.gradPooled),
		"gradOutputRows":    uintptr(arena.gradOutputRows),
		"gradNormalized":    uintptr(arena.gradNormalized),
		"gradHidden":        uintptr(arena.gradHidden),
		"gradFFNResidual":   uintptr(arena.gradFFNResidual),
		"gradActivatedPre":  uintptr(arena.gradActivatedPre),
		"gradActivated":     uintptr(arena.gradActivated),
		"gradHiddenFromFFN": uintptr(arena.gradHiddenFromFFN),
		"gradAttention":     uintptr(arena.gradAttention),
		"gradMixed":         uintptr(arena.gradMixed),
		"gradQ":             uintptr(arena.gradQ),
		"gradK":             uintptr(arena.gradK),
		"gradV":             uintptr(arena.gradV),
		"gradRoPE":          uintptr(arena.gradRoPE),
	} {
		if ptr == 0 {
			t.Fatalf("workspace %s was not allocated", name)
		}
		if prev, ok := ptrs[ptr]; ok {
			t.Fatalf("workspace %s aliases %s", name, prev)
		}
		ptrs[ptr] = name
	}
	rows := shape.Batch * shape.Tokens
	wantBytes := int64((shape.Batch*shape.OutputDim + rows*shape.OutputDim + 10*rows*shape.ModelDim + 2*rows*shape.FFNDim) * 4)
	if got := accel.CompactTrainStats().WorkspaceArenaBytes; got != wantBytes {
		t.Fatalf("workspace bytes = %d, want %d", got, wantBytes)
	}
}

func residentGradRefByName(t *testing.T, refs []backend.ResidentGradientRef, name string) backend.ResidentGradientRef {
	t.Helper()
	for _, ref := range refs {
		if ref.Name == name {
			return ref
		}
	}
	t.Fatalf("resident grad ref %q missing from %+v", name, refs)
	return backend.ResidentGradientRef{}
}

func compactTrainMaxAbs(got, want []float32) float32 {
	maxAbs := float32(0)
	for i := range got {
		d := abs32(got[i] - want[i])
		if d > maxAbs {
			maxAbs = d
		}
	}
	return maxAbs
}

func compactTrainRMSE(got, want []float32) float32 {
	if len(got) == 0 || len(got) != len(want) {
		return 0
	}
	sum := float64(0)
	for i := range got {
		d := float64(got[i] - want[i])
		sum += d * d
	}
	return float32(math.Sqrt(sum / float64(len(got))))
}

func assertFloatSlicesClose(t *testing.T, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len got=%d want=%d", len(got), len(want))
	}
	maxAbs := float32(0)
	maxIndex := 0
	for i := range want {
		if !finite32(got[i]) {
			t.Fatalf("nonfinite got[%d]=%g", i, got[i])
		}
		d := abs32(got[i] - want[i])
		if d > maxAbs {
			maxAbs = d
			maxIndex = i
		}
	}
	if maxAbs > tol {
		t.Fatalf("max abs = %g at %d got=%g want=%g tol=%g", maxAbs, maxIndex, got[maxIndex], want[maxIndex], tol)
	}
}

func assertCompactTrainStatsUnchanged(t *testing.T, want, got backend.CompactTrainAcceleratorStats, label string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s stats changed: got %+v want %+v", label, got, want)
	}
}
