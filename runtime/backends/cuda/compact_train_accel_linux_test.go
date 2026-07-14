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

func TestCompactTrainBackwardUnsupportedDoesNotReleaseHandle(t *testing.T) {
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
	_, err = accel.RunCompactTrainBackward(backend.CompactTrainBackwardRequest{
		Handle:     result.Handle,
		GradPooled: backend.NewTensorF32([]int{1, 4}, make([]float32, 4)),
	})
	if err == nil || !strings.Contains(err.Error(), "unsupported") {
		t.Fatalf("backward err = %v, want unsupported", err)
	}
	if !result.Handle.Token.Alive() {
		t.Fatal("unsupported backward released handle")
	}
	if err := accel.EndCompactTrainStep(11); err == nil || !strings.Contains(err.Error(), "live handle") {
		t.Fatalf("end with live handle err = %v, want live handle", err)
	}
	if err := accel.ReleaseCompactTrainHandle(result.Handle); err != nil {
		t.Fatalf("release after unsupported backward: %v", err)
	}
	if err := accel.EndCompactTrainStep(11); err != nil {
		t.Fatalf("end step: %v", err)
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
				if _, err := accel.copyResidentGradientForDebug(ref); err == nil || !strings.Contains(err.Error(), "not active") {
					t.Fatalf("ended-step resident grad copy err = %v, want not active", err)
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

func hostCompactTrainFinalOutputBackwardForCUDATest(req backend.CompactTrainForwardRequest, rope, projection bool, gradPooled []float32) ([]float32, []float32) {
	packedReq := backend.CompactForwardRequest{
		Shape:        req.Shape,
		Tokens:       req.Tokens,
		Masks:        req.Masks,
		Roles:        req.Roles,
		ResidentRefs: req.ResidentRefs,
		GELUMode:     req.GELUMode,
	}
	weights := compactForwardTestWeights(projection)
	packed := hostCompactForwardForCUDATest(packedReq, rope, projection)
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
