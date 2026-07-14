//go:build linux && cgo

package cuda

import (
	"math"
	"strings"
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
	beforePreflight := accel.Stats()
	preflightParam := param.Clone()
	preflightMom1 := mom1.Clone()
	preflightMom2 := mom2.Clone()
	preflightGrad := backend.NewTensorF32([]int{2, 2}, []float32{
		0.03, -0.04,
		0.07, -0.08,
	})
	if err := accel.PreflightApplyUpdate("projection", cfg, preflightParam, preflightMom1, preflightMom2, preflightGrad); err != nil {
		t.Fatalf("valid optimizer preflight: %v", err)
	}
	if err := accel.PreflightApplyUpdate("projection", cfg, preflightParam, preflightMom1, preflightMom2, preflightGrad); err != nil {
		t.Fatalf("repeated optimizer preflight: %v", err)
	}
	if after := accel.Stats(); after != beforePreflight {
		t.Fatalf("optimizer preflight mutated stats: before=%+v after=%+v", beforePreflight, after)
	}
	assertCloseF32Slice(t, "preflight param", preflightParam.F32, param.F32, 0)
	assertCloseF32Slice(t, "preflight mom1", preflightMom1.F32, mom1.F32, 0)
	assertCloseF32Slice(t, "preflight mom2", preflightMom2.F32, mom2.F32, 0)
	if err := accel.PreflightApplyUpdate("projection", cfg, preflightParam, preflightMom1, preflightMom2, backend.NewTensorF32([]int{1, 3}, []float32{1, 2, 3})); err == nil || !strings.Contains(err.Error(), "grad size") {
		t.Fatalf("bad grad preflight err = %v, want grad size", err)
	}
	badCfg := cfg
	badCfg.Optimizer = "rmsprop"
	if err := accel.PreflightApplyUpdate("projection", badCfg, preflightParam, preflightMom1, preflightMom2, preflightGrad); err == nil || !strings.Contains(err.Error(), "does not support") {
		t.Fatalf("bad optimizer preflight err = %v, want unsupported optimizer", err)
	}
	badCfg = cfg
	badCfg.DeferSync = true
	if err := accel.PreflightApplyUpdate("projection", badCfg, backend.NewTensorF32([]int{1, 3}, []float32{1, 2, 3}), backend.NewTensorF32([]int{1, 3}, make([]float32, 3)), backend.NewTensorF32([]int{1, 3}, make([]float32, 3)), backend.NewTensorF32([]int{1, 3}, []float32{0, 0, 0})); err == nil || !strings.Contains(err.Error(), "metadata mismatch") {
		t.Fatalf("resident binding preflight err = %v, want metadata mismatch", err)
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

func TestCUDAOptimizerApplyUpdateWithResidentGradParityAndFailures(t *testing.T) {
	shape := backend.CompactForwardShape{Batch: 1, Tokens: 2, ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 2, OutputDim: 4}
	weights := compactForwardTestWeights(false)
	trainOptAny, err := NewOptimizerAccelerator()
	if err != nil {
		t.Fatalf("new train optimizer accelerator: %v", err)
	}
	if trainOptAny == nil {
		t.Skip("no cuda optimizer accelerator available")
	}
	trainOpt := trainOptAny.(*optimizerAccelerator)
	trainAccel, err := NewCompactTrainAccelerator()
	if err != nil {
		trainOpt.Close()
		t.Fatalf("new compact train accelerator: %v", err)
	}
	defer trainAccel.Close()
	defer trainOpt.Close()
	layers := make([]CompactForwardLayerNames, shape.Layers)
	for layer := range layers {
		prefix := "layer" + string(rune('0'+layer)) + "_"
		layers[layer] = CompactForwardLayerNames{
			AttentionQ: prefix + "attn_q",
			AttentionK: prefix + "attn_k",
			AttentionV: prefix + "attn_v",
			AttentionO: prefix + "attn_o",
			FFNUp:      prefix + "ffn_up",
			FFNDown:    prefix + "ffn_down",
		}
	}
	trainAccel.Configure(layers, "token_embedding", "role_embedding", "", true)
	paramA := weights["layer1_ffn_up"].Clone()
	mom1A := backend.NewTensorF32(paramA.Shape, seqData(len(paramA.F32), 0.0002, -0.003))
	mom2A := backend.NewTensorF32(paramA.Shape, seqData(len(paramA.F32), 0.0001, 0.004))
	paramB := paramA.Clone()
	mom1B := mom1A.Clone()
	mom2B := mom2A.Clone()
	for name, tensor := range weights {
		seedTensor := tensor.Clone()
		seedMom1 := backend.NewTensorF32(tensor.Shape, make([]float32, len(tensor.F32)))
		seedMom2 := backend.NewTensorF32(tensor.Shape, make([]float32, len(tensor.F32)))
		if name == "layer1_ffn_up" {
			seedTensor = paramB
			seedMom1 = mom1B
			seedMom2 = mom2B
		}
		if err := trainOpt.EnsureResidentParameter(name, seedTensor, seedMom1, seedMom2); err != nil {
			t.Fatalf("seed resident %s: %v", name, err)
		}
		ref, ok := trainOpt.ResidentParameter(name)
		if !ok {
			t.Fatalf("resident %s missing", name)
		}
		if err := trainAccel.BindCompactTrainResident(name, tensor, ref); err != nil {
			t.Fatalf("bind resident %s: %v", name, err)
		}
	}
	refs := compactTrainResidentRefsForTest(t, trainAccel, shape)
	if err := trainAccel.BeginCompactTrainStep(201, refs); err != nil {
		t.Fatalf("begin compact train step: %v", err)
	}
	req := backend.CompactTrainForwardRequest{
		Shape:        shape,
		Tokens:       [][]int32{{2, 1}},
		Masks:        [][]int32{{1, 1}},
		Roles:        []int32{0},
		ResidentRefs: refs,
		GELUMode:     backend.CompactForwardGELUExact,
		StepID:       201,
	}
	forward, err := trainAccel.RunCompactTrainForward(req)
	if err != nil {
		t.Fatalf("compact train forward: %v", err)
	}
	backward, err := trainAccel.RunCompactTrainBackward(backend.CompactTrainBackwardRequest{
		Handle:     forward.Handle,
		GradPooled: backend.NewTensorF32([]int{shape.Batch, shape.OutputDim}, seqData(shape.Batch*shape.OutputDim, 0.031, -0.047)),
	})
	if err != nil {
		t.Fatalf("compact train backward: %v", err)
	}
	gradRef := residentGradRefByName(t, backward.ResidentGradRefs, "layer1_ffn_up")
	reallocatedRef := residentGradRefByName(t, backward.ResidentGradRefs, "layer1_ffn_down")
	hostGrad, err := trainAccel.copyResidentGradientForDebug(gradRef)
	if err != nil {
		t.Fatalf("copy resident gradient: %v", err)
	}
	if err := trainAccel.EndCompactTrainStep(201); err != nil {
		t.Fatalf("end compact train step: %v", err)
	}

	hostOptAny, err := NewOptimizerAccelerator()
	if err != nil {
		t.Fatalf("new host optimizer accelerator: %v", err)
	}
	if hostOptAny == nil {
		t.Skip("no cuda optimizer accelerator available")
	}
	otherOptAny, err := NewOptimizerAccelerator()
	if err != nil {
		hostOptAny.(*optimizerAccelerator).Close()
		t.Fatalf("new other optimizer accelerator: %v", err)
	}
	if otherOptAny == nil {
		hostOptAny.(*optimizerAccelerator).Close()
		t.Skip("no cuda optimizer accelerator available")
	}
	hostOpt := hostOptAny.(*optimizerAccelerator)
	otherOpt := otherOptAny.(*optimizerAccelerator)
	defer hostOpt.Close()
	defer otherOpt.Close()
	cfg := backend.OptimizerUpdateConfig{
		Optimizer:    "adamw",
		Step:         1,
		LearningRate: 0.01,
		WeightDecay:  0.001,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		Scale:        0.25,
	}
	if err := hostOpt.ApplyUpdate("layer1_ffn_up", cfg, paramA, mom1A, mom2A, hostGrad); err != nil {
		t.Fatalf("host grad update: %v", err)
	}
	if err := otherOpt.EnsureResidentParameter("layer1_ffn_up", paramB.Clone(), mom1B.Clone(), mom2B.Clone()); err != nil {
		t.Fatalf("seed other optimizer: %v", err)
	}
	if err := otherOpt.ApplyUpdateWithResidentGrad("layer1_ffn_up", cfg, paramB.Clone(), mom1B.Clone(), mom2B.Clone(), gradRef); err == nil || !strings.Contains(err.Error(), "different optimizer") {
		t.Fatalf("cross-optimizer resident grad update err = %v, want different optimizer", err)
	}
	downParam := weights["layer1_ffn_down"].Clone()
	downMom1 := backend.NewTensorF32(downParam.Shape, make([]float32, len(downParam.F32)))
	downMom2 := backend.NewTensorF32(downParam.Shape, make([]float32, len(downParam.F32)))
	replacement := backend.NewTensorF32([]int{1, 1}, []float32{0})
	replaceCfg := backend.OptimizerUpdateConfig{Optimizer: "sgd", Step: 101, LearningRate: 0, Scale: 1, DeferSync: true}
	if err := trainOpt.ApplyUpdate("layer1_ffn_down", replaceCfg, replacement, nil, nil, backend.NewTensorF32([]int{1, 1}, []float32{0})); err != nil {
		t.Fatalf("replace resident parameter with different allocation: %v", err)
	}
	restoreCfg := cfg
	restoreCfg.Step = 102
	restoreCfg.LearningRate = 0
	if err := trainOpt.ApplyUpdate("layer1_ffn_down", restoreCfg, downParam, downMom1, downMom2, backend.NewTensorF32(downParam.Shape, make([]float32, len(downParam.F32)))); err != nil {
		t.Fatalf("restore resident parameter with new token: %v", err)
	}
	if err := trainOpt.ApplyUpdateWithResidentGrad("layer1_ffn_down", cfg, downParam, downMom1, downMom2, reallocatedRef); err == nil || !strings.Contains(err.Error(), "token is stale") {
		t.Fatalf("reallocated resident grad update err = %v, want stale token", err)
	}
	beforePreflight := trainOpt.Stats()
	beforeTrainPreflight := trainAccel.CompactTrainStats()
	if err := trainOpt.PreflightApplyUpdateWithResidentGrad("layer1_ffn_up", cfg, paramB, mom1B, mom2B, gradRef); err != nil {
		t.Fatalf("resident grad preflight: %v", err)
	}
	if err := trainOpt.PreflightApplyUpdateWithResidentGrad("layer1_ffn_up", cfg, paramB, mom1B, mom2B, gradRef); err != nil {
		t.Fatalf("repeated resident grad preflight: %v", err)
	}
	if after := trainOpt.Stats(); after != beforePreflight {
		t.Fatalf("resident grad preflight mutated optimizer stats: before=%+v after=%+v", beforePreflight, after)
	}
	if after := trainAccel.CompactTrainStats(); after != beforeTrainPreflight {
		t.Fatalf("resident grad preflight mutated compact train stats: before=%+v after=%+v", beforeTrainPreflight, after)
	}
	beforeResident := trainOpt.Stats()
	if err := trainOpt.ApplyUpdateWithResidentGrad("layer1_ffn_up", cfg, paramB, mom1B, mom2B, gradRef); err != nil {
		t.Fatalf("resident grad update: %v", err)
	}
	assertCloseF32Slice(t, "param", paramB.F32, paramA.F32, 3e-8)
	if err := trainOpt.SyncState("layer1_ffn_up", paramB, mom1B, mom2B, true); err != nil {
		t.Fatalf("sync resident updated state: %v", err)
	}
	if err := hostOpt.SyncState("layer1_ffn_up", paramA, mom1A, mom2A, true); err != nil {
		t.Fatalf("sync host updated state: %v", err)
	}
	assertCloseF32Slice(t, "synced param", paramB.F32, paramA.F32, 3e-8)
	assertCloseF32Slice(t, "synced mom1", mom1B.F32, mom1A.F32, 3e-8)
	assertCloseF32Slice(t, "synced mom2", mom2B.F32, mom2A.F32, 3e-8)
	afterResident := trainOpt.Stats()
	if afterResident.UploadedBytes != beforeResident.UploadedBytes {
		t.Fatalf("resident grad update uploaded bytes delta = %d, want 0", afterResident.UploadedBytes-beforeResident.UploadedBytes)
	}
	if afterResident.ResidentGradUpdateCalls != 1 || afterResident.ResidentGradUploadBytesAvoided != int64(len(hostGrad.F32)*4) {
		t.Fatalf("resident grad stats = %+v, want one call and %d bytes avoided", afterResident, len(hostGrad.F32)*4)
	}
	trainStats := trainAccel.CompactTrainStats()
	if trainStats.HostGradUploadBytesAvoided != int64(len(hostGrad.F32)*4) || trainStats.OptimizerResidentGradNanos <= 0 {
		t.Fatalf("compact train optimizer bridge stats = %+v", trainStats)
	}
	if err := trainOpt.ApplyUpdateWithResidentGrad("layer1_ffn_up", cfg, paramB, mom1B, mom2B, gradRef); err == nil || !strings.Contains(err.Error(), "already used") {
		t.Fatalf("duplicate resident grad update err = %v, want already used", err)
	}
	stale := gradRef
	stale.Generation++
	if err := trainOpt.ApplyUpdateWithResidentGrad("layer1_ffn_up", cfg, paramB, mom1B, mom2B, stale); err == nil || !strings.Contains(err.Error(), "metadata mismatch") {
		t.Fatalf("stale resident grad update err = %v, want metadata mismatch", err)
	}
	wrongName := gradRef
	wrongName.Name = "layer0_ffn_up"
	if err := trainOpt.ApplyUpdateWithResidentGrad("layer1_ffn_up", cfg, paramB, mom1B, mom2B, wrongName); err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("wrong name resident grad update err = %v, want name mismatch", err)
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
