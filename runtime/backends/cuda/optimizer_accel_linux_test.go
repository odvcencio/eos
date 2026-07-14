//go:build linux && cgo

package cuda

import (
	"math"
	"testing"
	"time"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

func TestCUDAOptimizerAcceleratorKeepsResidentStateAcrossUpdates(t *testing.T) {
	accelAny, err := NewOptimizerAccelerator()
	if err != nil {
		t.Fatalf("new optimizer accelerator: %v", err)
	}
	if accelAny == nil {
		t.Skip("no cuda optimizer accelerator available")
	}
	accel, ok := accelAny.(*optimizerAccelerator)
	if !ok {
		t.Fatalf("optimizer accelerator type = %T, want *optimizerAccelerator", accelAny)
	}
	defer accel.Close()

	param := backend.NewTensorF32([]int{2, 2}, []float32{
		0.5, -0.25,
		1.0, -0.75,
	})
	mom1 := backend.NewTensorF32([]int{2, 2}, []float32{
		0.1, -0.05,
		0.2, -0.1,
	})
	mom2 := backend.NewTensorF32([]int{2, 2}, []float32{
		0.01, 0.02,
		0.03, 0.04,
	})
	cfg := backend.OptimizerUpdateConfig{
		Optimizer:    "adamw",
		Step:         1,
		LearningRate: 0.01,
		WeightDecay:  0.001,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		Scale:        1,
	}
	if err := accel.ApplyUpdate("projection", cfg, param, mom1, mom2, backend.NewTensorF32([]int{2, 2}, []float32{
		0.2, -0.1,
		0.05, -0.15,
	})); err != nil {
		t.Fatalf("first update: %v", err)
	}
	stateA, ok := accel.resident["projection"]
	if !ok {
		t.Fatal("expected resident optimizer state after first update")
	}
	if len(accel.resident) != 1 {
		t.Fatalf("resident state count = %d, want 1", len(accel.resident))
	}

	cfg.Step = 2
	if err := accel.ApplyUpdate("projection", cfg, param, mom1, mom2, backend.NewTensorF32([]int{2, 2}, []float32{
		-0.05, 0.15,
		0.1, -0.2,
	})); err != nil {
		t.Fatalf("second update: %v", err)
	}
	stateB, ok := accel.resident["projection"]
	if !ok {
		t.Fatal("expected resident optimizer state after second update")
	}
	if len(accel.resident) != 1 {
		t.Fatalf("resident state count = %d, want 1 after reuse", len(accel.resident))
	}
	if stateA.param != stateB.param || stateA.mom1 != stateB.mom1 || stateA.mom2 != stateB.mom2 {
		t.Fatalf("expected resident buffers to be reused, before=%+v after=%+v", stateA, stateB)
	}

	for i := range mom1.F32 {
		mom1.F32[i] = 0
	}
	for i := range mom2.F32 {
		mom2.F32[i] = 0
	}
	if err := accel.SyncState("projection", param, mom1, mom2, true); err != nil {
		t.Fatalf("sync resident state: %v", err)
	}
	allZero := true
	for i := range mom1.F32 {
		if mom1.F32[i] != 0 || mom2.F32[i] != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Fatal("expected sync to materialize resident moment state")
	}
	stateC, ok := accel.resident["projection"]
	if !ok {
		t.Fatal("expected resident optimizer state after sync")
	}
	if stateB.param != stateC.param || stateB.mom1 != stateC.mom1 || stateB.mom2 != stateC.mom2 {
		t.Fatalf("expected sync to preserve resident buffers, before=%+v after=%+v", stateB, stateC)
	}
	stats := accel.Stats()
	if stats.UpdateCalls != 2 {
		t.Fatalf("update calls = %d, want 2", stats.UpdateCalls)
	}
	if stats.SyncCalls != 1 {
		t.Fatalf("sync calls = %d, want 1", stats.SyncCalls)
	}
	if stats.UploadedBytes <= 0 {
		t.Fatalf("uploaded bytes = %d, want positive", stats.UploadedBytes)
	}
	if stats.DownloadedBytes <= 0 {
		t.Fatalf("downloaded bytes = %d, want positive", stats.DownloadedBytes)
	}
	if stats.ResidentParams != 1 {
		t.Fatalf("resident params = %d, want 1", stats.ResidentParams)
	}
}

func TestCUDAOptimizerDeferredSyncParityAndExplicitDownloads(t *testing.T) {
	deferredAny, err := NewOptimizerAccelerator()
	if err != nil {
		t.Fatalf("new deferred optimizer accelerator: %v", err)
	}
	if deferredAny == nil {
		t.Skip("no cuda optimizer accelerator available")
	}
	immediateAny, err := NewOptimizerAccelerator()
	if err != nil {
		t.Fatalf("new immediate optimizer accelerator: %v", err)
	}
	if immediateAny == nil {
		t.Skip("no cuda optimizer accelerator available")
	}
	deferred := deferredAny.(*optimizerAccelerator)
	immediate := immediateAny.(*optimizerAccelerator)
	defer deferred.Close()
	defer immediate.Close()

	paramA := backend.NewTensorF32([]int{2, 2}, []float32{0.5, -0.25, 1.0, -0.75})
	mom1A := backend.NewTensorF32([]int{2, 2}, []float32{0.1, -0.05, 0.2, -0.1})
	mom2A := backend.NewTensorF32([]int{2, 2}, []float32{0.01, 0.02, 0.03, 0.04})
	paramB := paramA.Clone()
	mom1B := mom1A.Clone()
	mom2B := mom2A.Clone()
	grads := [][]float32{
		{0.2, -0.1, 0.05, -0.15},
		{-0.05, 0.15, 0.1, -0.2},
		{0.03, 0.04, -0.02, 0.08},
	}
	base := backend.OptimizerUpdateConfig{
		Optimizer:    "adamw",
		LearningRate: 0.01,
		WeightDecay:  0.001,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		Scale:        1,
	}
	for i, grad := range grads {
		cfg := base
		cfg.Step = i + 1
		cfg.DeferSync = true
		if err := deferred.ApplyUpdate("projection", cfg, paramA, mom1A, mom2A, backend.NewTensorF32([]int{2, 2}, grad)); err != nil {
			t.Fatalf("deferred update %d: %v", i+1, err)
		}
		cfg.DeferSync = false
		if err := immediate.ApplyUpdate("projection", cfg, paramB, mom1B, mom2B, backend.NewTensorF32([]int{2, 2}, grad)); err != nil {
			t.Fatalf("immediate update %d: %v", i+1, err)
		}
	}
	beforeSync := deferred.Stats()
	if beforeSync.DownloadedBytes != 0 {
		t.Fatalf("deferred downloaded bytes before sync = %d, want 0", beforeSync.DownloadedBytes)
	}
	if beforeSync.DeferredSyncUpdates != int64(len(grads)) {
		t.Fatalf("deferred sync updates = %d, want %d", beforeSync.DeferredSyncUpdates, len(grads))
	}
	if err := deferred.SyncStateWithReason("projection", paramA, mom1A, mom2A, true, "checkpoint"); err != nil {
		t.Fatalf("deferred sync: %v", err)
	}
	if err := immediate.SyncState("projection", paramB, mom1B, mom2B, true); err != nil {
		t.Fatalf("immediate sync: %v", err)
	}
	assertCloseF32Slice(t, "param", paramA.F32, paramB.F32, 5e-6)
	assertCloseF32Slice(t, "mom1", mom1A.F32, mom1B.F32, 5e-6)
	assertCloseF32Slice(t, "mom2", mom2A.F32, mom2B.F32, 5e-6)
	afterSync := deferred.Stats()
	if afterSync.ForcedSyncCalls != 1 || afterSync.LastForcedSyncReason != "checkpoint" {
		t.Fatalf("forced sync stats = %d/%q, want 1/checkpoint", afterSync.ForcedSyncCalls, afterSync.LastForcedSyncReason)
	}
	if afterSync.DownloadedBytes == 0 {
		t.Fatal("deferred sync did not download host state at explicit boundary")
	}
}

func TestCUDAOptimizerResidentParameterFeedsMatMulBeforeHostSync(t *testing.T) {
	optAny, err := NewOptimizerAccelerator()
	if err != nil {
		t.Fatalf("new optimizer accelerator: %v", err)
	}
	if optAny == nil {
		t.Skip("no cuda optimizer accelerator available")
	}
	mmAny, err := NewMatMulAccelerator()
	if err != nil {
		t.Fatalf("new matmul accelerator: %v", err)
	}
	if mmAny == nil {
		t.Skip("no cuda matmul accelerator available")
	}
	opt := optAny.(*optimizerAccelerator)
	mm := mmAny.(*matMulAccelerator)
	defer opt.Close()
	defer mm.Close()

	hostBefore := []float32{0.5, -0.25, 1.0, -0.75}
	param := backend.NewTensorF32([]int{2, 2}, hostBefore)
	mom1 := backend.NewTensorF32([]int{2, 2}, []float32{0, 0, 0, 0})
	mom2 := backend.NewTensorF32([]int{2, 2}, []float32{0, 0, 0, 0})
	cfg := backend.OptimizerUpdateConfig{
		Optimizer:    "adamw",
		Step:         1,
		LearningRate: 0.01,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		Scale:        1,
		DeferSync:    true,
	}
	if err := opt.ApplyUpdate("projection", cfg, param, mom1, mom2, backend.NewTensorF32([]int{2, 2}, []float32{0.2, -0.1, 0.05, -0.15})); err != nil {
		t.Fatalf("deferred update: %v", err)
	}
	assertCloseF32Slice(t, "host param before explicit sync", param.F32, hostBefore, 0)
	ref, ok := opt.ResidentParameter("projection")
	if !ok {
		t.Fatal("missing resident optimizer parameter")
	}
	if err := mm.BindMatrixFromResident("projection", param, ref); err != nil {
		t.Fatalf("bind resident matrix: %v", err)
	}
	lhs := backend.NewTensorF32([]int{1, 2}, []float32{2, -1})
	result, err := mm.RunMatMulWithBoundRight(lhs, "projection", eosartifact.ValueType{Kind: eosartifact.ValueTensor, Tensor: &eosartifact.TensorType{DType: "f32", Shape: []string{"1", "2"}}}, false, false)
	if err != nil {
		t.Fatalf("run bound matmul: %v", err)
	}
	if err := opt.SyncState("projection", param, mom1, mom2, true); err != nil {
		t.Fatalf("sync updated param: %v", err)
	}
	want := []float32{
		lhs.F32[0]*param.F32[0] + lhs.F32[1]*param.F32[2],
		lhs.F32[0]*param.F32[1] + lhs.F32[1]*param.F32[3],
	}
	got := result.Outputs[0].F32
	assertCloseF32Slice(t, "resident matmul", got, want, 5e-6)
	if mm.Stats().UploadedBytes != 0 {
		t.Fatalf("matmul bind uploaded bytes = %d, want 0 for resident bridge", mm.Stats().UploadedBytes)
	}
}

func TestCUDAOptimizerResidentParameterRejectsStaleReplacement(t *testing.T) {
	optAny, err := NewOptimizerAccelerator()
	if err != nil {
		t.Fatalf("new optimizer accelerator: %v", err)
	}
	if optAny == nil {
		t.Skip("no cuda optimizer accelerator available")
	}
	mmAny, err := NewMatMulAccelerator()
	if err != nil {
		t.Fatalf("new matmul accelerator: %v", err)
	}
	if mmAny == nil {
		t.Skip("no cuda matmul accelerator available")
	}
	opt := optAny.(*optimizerAccelerator)
	mm := mmAny.(*matMulAccelerator)
	defer opt.Close()
	defer mm.Close()

	param := backend.NewTensorF32([]int{2, 2}, []float32{0.5, -0.25, 1.0, -0.75})
	mom1 := backend.NewTensorF32([]int{2, 2}, []float32{0, 0, 0, 0})
	mom2 := backend.NewTensorF32([]int{2, 2}, []float32{0, 0, 0, 0})
	cfg := backend.OptimizerUpdateConfig{
		Optimizer:    "adamw",
		Step:         1,
		LearningRate: 0.01,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		Scale:        1,
		DeferSync:    true,
	}
	if err := opt.ApplyUpdate("projection", cfg, param, mom1, mom2, backend.NewTensorF32([]int{2, 2}, []float32{0.2, -0.1, 0.05, -0.15})); err != nil {
		t.Fatalf("initial deferred update: %v", err)
	}
	ref, ok := opt.ResidentParameter("projection")
	if !ok {
		t.Fatal("missing resident optimizer parameter")
	}
	if err := mm.BindMatrixFromResident("projection", param, ref); err != nil {
		t.Fatalf("bind resident matrix: %v", err)
	}
	cfg.Step = 2
	replacement := backend.NewTensorF32([]int{1, 3}, []float32{0.1, 0.2, 0.3})
	replMom1 := backend.NewTensorF32([]int{1, 3}, []float32{0, 0, 0})
	replMom2 := backend.NewTensorF32([]int{1, 3}, []float32{0, 0, 0})
	if err := opt.ApplyUpdate("projection", cfg, replacement, replMom1, replMom2, backend.NewTensorF32([]int{1, 3}, []float32{0.01, 0.02, 0.03})); err != nil {
		t.Fatalf("replacement deferred update: %v", err)
	}
	if ref.Token.Alive() {
		t.Fatal("old resident token is alive after replacement")
	}
	_, err = mm.RunMatMulWithBoundRight(
		backend.NewTensorF32([]int{1, 2}, []float32{1, 1}),
		"projection",
		eosartifact.ValueType{Kind: eosartifact.ValueTensor, Tensor: &eosartifact.TensorType{DType: "f32"}},
		false,
		false,
	)
	if err == nil {
		t.Fatal("stale resident binding run succeeded after replacement")
	}
}

func TestCUDAOptimizerResidentParameterRejectsAfterClose(t *testing.T) {
	optAny, err := NewOptimizerAccelerator()
	if err != nil {
		t.Fatalf("new optimizer accelerator: %v", err)
	}
	if optAny == nil {
		t.Skip("no cuda optimizer accelerator available")
	}
	mmAny, err := NewMatMulAccelerator()
	if err != nil {
		t.Fatalf("new matmul accelerator: %v", err)
	}
	if mmAny == nil {
		t.Skip("no cuda matmul accelerator available")
	}
	opt := optAny.(*optimizerAccelerator)
	mm := mmAny.(*matMulAccelerator)
	defer mm.Close()

	param := backend.NewTensorF32([]int{2, 2}, []float32{0.5, -0.25, 1.0, -0.75})
	mom1 := backend.NewTensorF32([]int{2, 2}, []float32{0, 0, 0, 0})
	mom2 := backend.NewTensorF32([]int{2, 2}, []float32{0, 0, 0, 0})
	cfg := backend.OptimizerUpdateConfig{
		Optimizer:    "adamw",
		Step:         1,
		LearningRate: 0.01,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		Scale:        1,
		DeferSync:    true,
	}
	if err := opt.ApplyUpdate("projection", cfg, param, mom1, mom2, backend.NewTensorF32([]int{2, 2}, []float32{0.2, -0.1, 0.05, -0.15})); err != nil {
		t.Fatalf("deferred update: %v", err)
	}
	ref, ok := opt.ResidentParameter("projection")
	if !ok {
		t.Fatal("missing resident optimizer parameter")
	}
	if err := mm.BindMatrixFromResident("projection", param, ref); err != nil {
		t.Fatalf("bind resident matrix: %v", err)
	}
	opt.Close()
	if ref.Token.Alive() {
		t.Fatal("resident token is alive after optimizer close")
	}
	_, err = mm.RunMatMulWithBoundRight(
		backend.NewTensorF32([]int{1, 2}, []float32{1, 1}),
		"projection",
		eosartifact.ValueType{Kind: eosartifact.ValueTensor, Tensor: &eosartifact.TensorType{DType: "f32"}},
		false,
		false,
	)
	if err == nil {
		t.Fatal("resident binding run succeeded after optimizer close")
	}
}

func assertCloseF32Slice(t *testing.T, label string, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", label, len(got), len(want))
	}
	for i := range got {
		if float32(math.Abs(float64(got[i]-want[i]))) > tol {
			t.Fatalf("%s[%d] = %.8f, want %.8f (tol %.8f)", label, i, got[i], want[i], tol)
		}
	}
}

func BenchmarkCUDAOptimizerDeferredSyncTransfers(b *testing.B) {
	for _, mode := range []string{"immediate", "forced_sync_each_update", "deferred"} {
		b.Run(mode, func(b *testing.B) {
			accelAny, err := NewOptimizerAccelerator()
			if err != nil {
				b.Fatalf("new optimizer accelerator: %v", err)
			}
			if accelAny == nil {
				b.Skip("no cuda optimizer accelerator available")
			}
			accel := accelAny.(*optimizerAccelerator)
			defer accel.Close()
			elements := 16 * 1024
			param := backend.NewTensorF32([]int{128, 128}, filledCUDAOptimizerBench(elements, 0.01))
			mom1 := backend.NewTensorF32([]int{128, 128}, make([]float32, elements))
			mom2 := backend.NewTensorF32([]int{128, 128}, make([]float32, elements))
			grad := backend.NewTensorF32([]int{128, 128}, filledCUDAOptimizerBench(elements, 0.001))
			cfg := backend.OptimizerUpdateConfig{
				Optimizer:    "adamw",
				LearningRate: 0.001,
				Beta1:        0.9,
				Beta2:        0.999,
				Epsilon:      1e-8,
				Scale:        1,
				DeferSync:    mode == "deferred",
			}
			b.ResetTimer()
			updateStart := time.Now()
			for i := 0; i < b.N; i++ {
				cfg.Step = i + 1
				if err := accel.ApplyUpdate("bench_projection", cfg, param, mom1, mom2, grad); err != nil {
					b.Fatalf("update: %v", err)
				}
				if mode == "forced_sync_each_update" {
					if err := accel.SyncStateWithReason("bench_projection", param, mom1, mom2, true, "benchmark_step"); err != nil {
						b.Fatalf("step sync: %v", err)
					}
				}
			}
			updateElapsed := time.Since(updateStart)
			b.StopTimer()
			updateStats := accel.Stats()
			boundaryElapsed := time.Duration(0)
			if mode == "deferred" {
				boundaryStart := time.Now()
				if err := accel.SyncStateWithReason("bench_projection", param, mom1, mom2, true, "benchmark"); err != nil {
					b.Fatalf("sync: %v", err)
				}
				boundaryElapsed = time.Since(boundaryStart)
			}
			totalStats := accel.Stats()
			steps := float64(b.N)
			b.ReportMetric(float64(updateElapsed.Nanoseconds())/steps, "update_only_ns/op")
			b.ReportMetric(float64((updateElapsed+boundaryElapsed).Nanoseconds())/steps, "total_with_boundary_ns/op")
			b.ReportMetric(float64(updateStats.UploadedBytes)/steps, "update_upload_B/update")
			b.ReportMetric(float64(updateStats.DownloadedBytes)/steps, "update_download_B/update")
			b.ReportMetric(float64(totalStats.UploadedBytes)/steps, "total_upload_B/update")
			b.ReportMetric(float64(totalStats.DownloadedBytes)/steps, "total_download_B/update")
			if mode == "deferred" {
				b.ReportMetric(float64(boundaryElapsed.Nanoseconds()), "boundary_sync_ns/sync")
				b.ReportMetric(float64(totalStats.UploadedBytes-updateStats.UploadedBytes), "boundary_upload_B/sync")
				b.ReportMetric(float64(totalStats.DownloadedBytes-updateStats.DownloadedBytes), "boundary_download_B/sync")
			}
			b.ReportMetric(float64(totalStats.DeferredSyncUpdates)/steps, "deferred_updates/update")
		})
	}
}

func filledCUDAOptimizerBench(n int, scale float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i%17)-8) * scale
	}
	return out
}
