package eosruntime

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/compiler"
	"m31labs.dev/eos/runtime/backend"
	"m31labs.dev/eos/runtime/backends/cuda"
	"m31labs.dev/eos/runtime/backends/metal"
)

func TestFlattenFixedFloat32MatricesScratchReusesContiguousViews(t *testing.T) {
	trainer := &EmbeddingTrainer{}
	base := []float32{1, 2, 3, 4, 5, 6}
	matrices := [][]float32{
		base[0:2],
		base[2:4],
		base[4:6],
	}
	out, ok := trainer.flattenFixedFloat32MatricesScratch(0, matrices, 2)
	if !ok {
		t.Fatal("flatten contiguous matrices failed")
	}
	if len(out) != len(base) {
		t.Fatalf("flattened len = %d, want %d", len(out), len(base))
	}
	if &out[0] != &base[0] {
		t.Fatal("contiguous flatten copied instead of reusing the backing array")
	}
	if len(trainer.scratchF32) != 0 {
		t.Fatalf("scratch buffer count = %d, want 0", len(trainer.scratchF32))
	}
}

func TestFlattenFixedFloat32MatricesScratchCopiesNonContiguousViews(t *testing.T) {
	trainer := &EmbeddingTrainer{}
	left := []float32{1, 2}
	right := []float32{3, 4}
	out, ok := trainer.flattenFixedFloat32MatricesScratch(0, [][]float32{left, right}, 2)
	if !ok {
		t.Fatal("flatten non-contiguous matrices failed")
	}
	if len(out) != 4 {
		t.Fatalf("flattened len = %d, want 4", len(out))
	}
	if &out[0] == &left[0] {
		t.Fatal("non-contiguous flatten reused left backing array")
	}
	for i, want := range []float32{1, 2, 3, 4} {
		if out[i] != want {
			t.Fatalf("out[%d] = %v, want %v", i, out[i], want)
		}
	}
	if len(trainer.scratchF32) != 1 {
		t.Fatalf("scratch buffer count = %d, want 1", len(trainer.scratchF32))
	}
}

func assertCloseF32Slice(t *testing.T, label string, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", label, len(got), len(want))
	}
	for i := range got {
		if abs32(got[i]-want[i]) > tol {
			t.Fatalf("%s[%d] = %.8f, want %.8f (tol %.8f)", label, i, got[i], want[i], tol)
		}
	}
}

type countingMatMulAccelerator struct {
	bindCalls         int
	runCalls          int
	maxRunBatches     int
	boundRightRuns    int
	multiBoundRuns    int
	sharedLeftRuns    int
	accumulatedRuns   int
	maxBoundRightRows int
	maxSharedLeftRHS  int
	maxAccumTerms     int
	maxRunOutputCols  int
	uploadedBytes     int64
	downloadedBytes   int64
	bindUploadedBytes int64
	bound             map[string]*backend.Tensor
	closed            bool
}

type fakeResidentOptimizerToken struct {
	tensor     *backend.Tensor
	generation uint64
	alive      bool
}

func (t *fakeResidentOptimizerToken) OptimizerResidentParameterToken() {}

func (t *fakeResidentOptimizerToken) CompactForwardResidentToken() {}

func (t *fakeResidentOptimizerToken) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (t *fakeResidentOptimizerToken) Generation() uint64 {
	if t == nil {
		return 0
	}
	return t.generation
}

func (t *fakeResidentOptimizerToken) Alive() bool {
	return t != nil && t.alive
}

type fakeResidentOptimizerAccelerator struct {
	resident            map[string]*fakeResidentOptimizerToken
	applyErr            error
	applyErrNames       map[string]error
	applyCalls          int64
	preflightErr        error
	preflightErrNames   map[string]error
	preflightCalls      int64
	preflightApplyErr   error
	preflightApplyCalls int64
	residentApplyCalls  int64
	syncErr             error
	closed              bool
	stats               backend.OptimizerAcceleratorStats
}

func (a *fakeResidentOptimizerAccelerator) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (a *fakeResidentOptimizerAccelerator) ApplyUpdate(name string, cfg backend.OptimizerUpdateConfig, tensor, mom1, mom2, grad *backend.Tensor) error {
	a.applyCalls++
	if a.applyErr != nil {
		return a.applyErr
	}
	if err := a.applyErrNames[name]; err != nil {
		return err
	}
	if tensor == nil || grad == nil {
		return fmt.Errorf("fake resident optimizer requires tensor and grad")
	}
	if a.resident == nil {
		a.resident = map[string]*fakeResidentOptimizerToken{}
	}
	target := tensor
	if cfg.DeferSync {
		if existing := a.resident[name]; existing != nil && existing.tensor != nil {
			target = existing.tensor
		} else {
			target = tensor.Clone()
			a.resident[name] = &fakeResidentOptimizerToken{tensor: target, generation: 1, alive: true}
		}
	}
	applyOptimizerUpdate(EmbeddingTrainConfig{
		Optimizer:    cfg.Optimizer,
		LearningRate: cfg.LearningRate,
		WeightDecay:  cfg.WeightDecay,
		Beta1:        cfg.Beta1,
		Beta2:        cfg.Beta2,
		Epsilon:      cfg.Epsilon,
	}, cfg.Step, target, mom1, mom2, grad.F32, cfg.Scale)
	a.stats.UpdateCalls++
	a.stats.TensorUpdateCalls++
	if cfg.DeferSync {
		a.stats.DeferredSyncUpdates++
	}
	return nil
}

func (a *fakeResidentOptimizerAccelerator) PreflightApplyUpdate(name string, _ backend.OptimizerUpdateConfig, tensor, mom1, mom2, grad *backend.Tensor) error {
	a.preflightApplyCalls++
	if a.preflightApplyErr != nil {
		return a.preflightApplyErr
	}
	if err := a.preflightErrNames[name]; err != nil {
		return err
	}
	if tensor == nil || grad == nil {
		return fmt.Errorf("fake optimizer preflight %q requires tensor and grad", name)
	}
	if len(grad.F32) != len(tensor.F32) {
		return fmt.Errorf("fake optimizer preflight %q grad mismatch", name)
	}
	if mom1 == nil || mom2 == nil {
		return fmt.Errorf("fake optimizer preflight %q requires moments", name)
	}
	if len(mom1.F32) != len(tensor.F32) || len(mom2.F32) != len(tensor.F32) {
		return fmt.Errorf("fake optimizer preflight %q moment mismatch", name)
	}
	return nil
}

func (a *fakeResidentOptimizerAccelerator) SyncState(name string, tensor, mom1, mom2 *backend.Tensor, includeMoments bool) error {
	a.stats.SyncCalls++
	if a.syncErr != nil {
		return a.syncErr
	}
	if token := a.resident[name]; token != nil && token.tensor != nil && tensor != nil {
		copy(tensor.F32, token.tensor.F32)
	}
	return nil
}

func (a *fakeResidentOptimizerAccelerator) Stats() backend.OptimizerAcceleratorStats {
	return a.stats
}

func (a *fakeResidentOptimizerAccelerator) Close() {
	a.closed = true
	for _, token := range a.resident {
		token.alive = false
	}
}

func (a *fakeResidentOptimizerAccelerator) ResidentParameter(name string) (backend.OptimizerResidentParameter, bool) {
	token := a.resident[name]
	if token == nil || token.tensor == nil || !token.alive {
		return backend.OptimizerResidentParameter{}, false
	}
	return backend.OptimizerResidentParameter{Backend: eosartifact.BackendCUDA, Token: token, Elements: len(token.tensor.F32)}, true
}

func (a *fakeResidentOptimizerAccelerator) EnsureResidentParameter(name string, tensor, mom1, mom2 *backend.Tensor) error {
	if tensor == nil {
		return fmt.Errorf("fake resident seed %q requires tensor", name)
	}
	if a.resident == nil {
		a.resident = map[string]*fakeResidentOptimizerToken{}
	}
	if existing := a.resident[name]; existing != nil && existing.tensor != nil && existing.alive {
		return nil
	}
	a.resident[name] = &fakeResidentOptimizerToken{tensor: tensor.Clone(), generation: uint64(len(a.resident) + 1), alive: true}
	return nil
}

func (a *fakeResidentOptimizerAccelerator) ApplyUpdateWithResidentGrad(name string, cfg backend.OptimizerUpdateConfig, tensor, mom1, mom2 *backend.Tensor, grad backend.ResidentGradientRef) error {
	a.residentApplyCalls++
	if a.applyErr != nil {
		return a.applyErr
	}
	if tensor == nil {
		return fmt.Errorf("fake resident optimizer requires tensor")
	}
	zeros := backend.NewTensorF32(tensor.Shape, make([]float32, len(tensor.F32)))
	if grad.Elements != 0 && grad.Elements != len(tensor.F32) {
		return fmt.Errorf("fake resident gradient %q elements %d, want %d", grad.Name, grad.Elements, len(tensor.F32))
	}
	return a.ApplyUpdate(name, cfg, tensor, mom1, mom2, zeros)
}

func (a *fakeResidentOptimizerAccelerator) PreflightApplyUpdateWithResidentGrad(name string, _ backend.OptimizerUpdateConfig, tensor, mom1, mom2 *backend.Tensor, grad backend.ResidentGradientRef) error {
	a.preflightCalls++
	if a.preflightErr != nil {
		return a.preflightErr
	}
	if err := a.preflightErrNames[name]; err != nil {
		return err
	}
	if tensor == nil || mom1 == nil || mom2 == nil {
		return fmt.Errorf("fake resident optimizer preflight %q requires tensor and moments", name)
	}
	if grad.Name != name || grad.Elements != len(tensor.F32) {
		return fmt.Errorf("fake resident optimizer preflight %q gradient mismatch", name)
	}
	if len(mom1.F32) != len(tensor.F32) || len(mom2.F32) != len(tensor.F32) {
		return fmt.Errorf("fake resident optimizer preflight %q moment mismatch", name)
	}
	return nil
}

type residentAwareCountingMatMulAccelerator struct {
	countingMatMulAccelerator
	residentBindCalls int
}

type fakeCompactForwardAccelerator struct {
	trainer        *EmbeddingTrainer
	forward        *compactEmbeddingForwardWeights
	configured     bool
	bound          map[string]backend.OptimizerResidentParameter
	preflightErr   error
	runErr         error
	corruptResult  bool
	preflightCalls int64
	stats          backend.CompactForwardAcceleratorStats
}

type fakeCompactTrainHandleToken struct {
	alive      bool
	generation uint64
	stepID     uint64
}

func (t *fakeCompactTrainHandleToken) CompactTrainHandleToken() {}
func (t *fakeCompactTrainHandleToken) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}
func (t *fakeCompactTrainHandleToken) Generation() uint64 { return t.generation }
func (t *fakeCompactTrainHandleToken) StepID() uint64     { return t.stepID }
func (t *fakeCompactTrainHandleToken) Alive() bool        { return t != nil && t.alive }

type fakeResidentGradientToken struct {
	alive      bool
	generation uint64
}

func (t *fakeResidentGradientToken) ResidentGradientToken() {}
func (t *fakeResidentGradientToken) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}
func (t *fakeResidentGradientToken) Generation() uint64 { return t.generation }
func (t *fakeResidentGradientToken) Alive() bool        { return t != nil && t.alive }

type fakeCompactTrainAccelerator struct {
	configured      bool
	bound           map[string]backend.OptimizerResidentParameter
	preflightCalls  int64
	beginCalls      int64
	forwardCalls    int64
	backwardCalls   int64
	endCalls        int64
	abortCalls      int64
	releaseCalls    int64
	backwardGrads   []*backend.Tensor
	forwardRequests []backend.CompactTrainForwardRequest
	nextGeneration  uint64
	live            map[uint64]*fakeCompactTrainHandleToken
	supportBackward bool
	activeStep      uint64
	sealedStep      uint64
	gradientTokens  []*fakeResidentGradientToken
	preflightErr    error
	forwardErr      error
	backwardErr     error
	endErr          error
	releaseErr      error
	closed          bool
	stats           backend.CompactTrainAcceleratorStats
}

func (a *fakeCompactTrainAccelerator) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (a *fakeCompactTrainAccelerator) Close() {
	a.closed = true
	for _, token := range a.live {
		token.alive = false
	}
	for _, token := range a.gradientTokens {
		token.alive = false
	}
	a.live = nil
	a.gradientTokens = nil
	a.activeStep = 0
	a.sealedStep = 0
	a.stats.LiveHandles = 0
}

func (a *fakeCompactTrainAccelerator) ConfigureCompactForward([]backend.CompactForwardLayerConfig, string, string, string, bool) {
	a.configured = true
}

func (a *fakeCompactTrainAccelerator) BindCompactTrainResident(name string, tensor *backend.Tensor, ref backend.OptimizerResidentParameter) error {
	if a.bound == nil {
		a.bound = map[string]backend.OptimizerResidentParameter{}
	}
	a.bound[name] = ref
	return nil
}

func (a *fakeCompactTrainAccelerator) PreflightCompactTrainForward(req backend.CompactTrainForwardRequest) error {
	a.preflightCalls++
	if a.preflightErr != nil {
		return a.preflightErr
	}
	if !a.configured {
		return fmt.Errorf("fake compact train was not configured")
	}
	if len(a.bound) == 0 {
		return fmt.Errorf("fake compact train has no resident bindings")
	}
	if len(req.ResidentRefs) == 0 {
		return fmt.Errorf("fake compact train request has no resident refs")
	}
	return validateCompactForwardPackedInputs(req.Shape, req.Tokens, req.Masks, req.Roles)
}

func (a *fakeCompactTrainAccelerator) BeginCompactTrainStep(stepID uint64, refs []backend.CompactForwardResidentRef) error {
	if !a.supportBackward {
		return fmt.Errorf("fake compact train backward unsupported")
	}
	a.beginCalls++
	if len(refs) == 0 {
		return fmt.Errorf("fake compact train begin requires resident refs")
	}
	a.activeStep = stepID
	a.sealedStep = 0
	return nil
}

func (a *fakeCompactTrainAccelerator) EndCompactTrainStep(stepID uint64) error {
	if !a.supportBackward {
		return fmt.Errorf("fake compact train backward unsupported")
	}
	a.endCalls++
	if a.endErr != nil {
		return a.endErr
	}
	if a.activeStep != stepID {
		return fmt.Errorf("fake compact train stale end step")
	}
	a.activeStep = 0
	a.sealedStep = stepID
	return nil
}

func (a *fakeCompactTrainAccelerator) AbortCompactTrainStep(stepID uint64) error {
	if !a.supportBackward {
		return fmt.Errorf("fake compact train backward unsupported")
	}
	a.abortCalls++
	if a.activeStep != 0 && a.activeStep != stepID || a.sealedStep != 0 && a.sealedStep != stepID {
		return fmt.Errorf("fake compact train stale abort step")
	}
	for gen, token := range a.live {
		if token.alive {
			token.alive = false
			a.stats.HandlesReleased++
		}
		delete(a.live, gen)
	}
	a.stats.LiveHandles = 0
	for _, token := range a.gradientTokens {
		token.alive = false
	}
	a.gradientTokens = nil
	a.activeStep = 0
	a.sealedStep = 0
	return nil
}

func (a *fakeCompactTrainAccelerator) RunCompactTrainBackward(req backend.CompactTrainBackwardRequest) (backend.CompactTrainBackwardResult, error) {
	if !a.supportBackward {
		return backend.CompactTrainBackwardResult{}, fmt.Errorf("fake compact train backward unsupported")
	}
	a.backwardCalls++
	if a.backwardErr != nil {
		return backend.CompactTrainBackwardResult{}, a.backwardErr
	}
	token, ok := req.Handle.Token.(*fakeCompactTrainHandleToken)
	if !ok || token == nil || !token.alive {
		return backend.CompactTrainBackwardResult{}, fmt.Errorf("fake compact train backward requires live handle")
	}
	token.alive = false
	delete(a.live, token.generation)
	a.stats.BackwardCalls++
	a.stats.HandlesReleased++
	a.stats.LiveHandles--
	a.backwardGrads = append(a.backwardGrads, req.GradPooled.Clone())
	refs := make([]backend.ResidentGradientRef, 0, len(a.bound))
	for name, ref := range a.bound {
		a.nextGeneration++
		gradToken := &fakeResidentGradientToken{alive: true, generation: a.nextGeneration}
		a.gradientTokens = append(a.gradientTokens, gradToken)
		refs = append(refs, backend.ResidentGradientRef{
			Name:       name,
			Backend:    eosartifact.BackendCUDA,
			Token:      gradToken,
			Elements:   ref.Elements,
			Generation: a.nextGeneration,
			StepID:     req.Handle.StepID,
		})
	}
	return backend.CompactTrainBackwardResult{ResidentGradRefs: refs}, nil
}

func (a *fakeCompactTrainAccelerator) RunCompactTrainForward(req backend.CompactTrainForwardRequest) (backend.CompactTrainForwardResult, error) {
	a.forwardCalls++
	if a.forwardErr != nil {
		return backend.CompactTrainForwardResult{}, a.forwardErr
	}
	a.forwardRequests = append(a.forwardRequests, req)
	a.nextGeneration++
	token := &fakeCompactTrainHandleToken{alive: true, generation: a.nextGeneration, stepID: req.StepID}
	if a.live == nil {
		a.live = map[uint64]*fakeCompactTrainHandleToken{}
	}
	a.live[token.generation] = token
	a.stats.ForwardCalls++
	a.stats.HandlesCreated++
	a.stats.LiveHandles++
	a.stats.PooledDownloadedBytes += int64(req.Shape.Batch * req.Shape.OutputDim * 4)
	return backend.CompactTrainForwardResult{
		Handle: backend.CompactTrainHandle{Backend: eosartifact.BackendCUDA, Token: token, Shape: req.Shape, Generation: token.generation, StepID: token.stepID},
		Pooled: backend.NewTensorF32([]int{req.Shape.Batch, req.Shape.OutputDim}, make([]float32, req.Shape.Batch*req.Shape.OutputDim)),
	}, nil
}

func (a *fakeCompactTrainAccelerator) ReleaseCompactTrainHandle(handle backend.CompactTrainHandle) error {
	a.releaseCalls++
	if a.releaseErr != nil {
		return a.releaseErr
	}
	token, ok := handle.Token.(*fakeCompactTrainHandleToken)
	if !ok || token == nil || !token.alive {
		return fmt.Errorf("fake compact train stale handle")
	}
	token.alive = false
	delete(a.live, token.generation)
	a.stats.HandlesReleased++
	a.stats.LiveHandles--
	return nil
}

func (a *fakeCompactTrainAccelerator) CompactTrainStats() backend.CompactTrainAcceleratorStats {
	return a.stats
}

func (a *fakeCompactForwardAccelerator) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (a *fakeCompactForwardAccelerator) ConfigureCompactForward(layers []backend.CompactForwardLayerConfig, tokenName, roleName, outputProjectionName string, useRoPE bool) {
	a.configured = true
}

func (a *fakeCompactForwardAccelerator) BindCompactForwardResident(name string, tensor *backend.Tensor, ref backend.OptimizerResidentParameter) error {
	if a.bound == nil {
		a.bound = map[string]backend.OptimizerResidentParameter{}
	}
	a.bound[name] = ref
	return nil
}

func (a *fakeCompactForwardAccelerator) PreflightCompactForward(req backend.CompactForwardRequest) error {
	a.preflightCalls++
	if a.preflightErr != nil {
		return a.preflightErr
	}
	if !a.configured {
		return fmt.Errorf("fake compact forward was not configured")
	}
	if len(a.bound) == 0 {
		return fmt.Errorf("fake compact forward has no resident bindings")
	}
	if len(req.ResidentRefs) == 0 {
		return fmt.Errorf("fake compact forward request has no resident refs")
	}
	return validateCompactForwardPackedInputs(req.Shape, req.Tokens, req.Masks, req.Roles)
}

func (a *fakeCompactForwardAccelerator) RunCompactForward(req backend.CompactForwardRequest) (backend.CompactForwardResult, error) {
	if a.runErr != nil {
		a.stats.UnhandledCalls++
		return backend.CompactForwardResult{}, a.runErr
	}
	if a.trainer == nil || a.forward == nil {
		a.stats.UnhandledCalls++
		return backend.CompactForwardResult{}, fmt.Errorf("fake compact forward is missing host encoder")
	}
	seqs := make([]*embeddingEncodedSequence, len(req.Tokens))
	var finalRows [][]float32
	if req.Shape.HasOutputProjection {
		finalRows = make([][]float32, len(req.Tokens))
	}
	for i := range req.Tokens {
		seq, err := a.trainer.encodeCompactSequence(req.Tokens[i], req.Masks[i], req.Roles[i], a.forward)
		if err != nil {
			a.stats.UnhandledCalls++
			return backend.CompactForwardResult{}, err
		}
		seqs[i] = seq
		if req.Shape.HasOutputProjection {
			d := req.Shape.ModelDim
			current := seq.finalLayer().projected
			rows, err := a.trainer.compactFinalOutputRows(current, req.Shape.Tokens, d, a.forward.outputProjection)
			if err != nil {
				a.stats.UnhandledCalls++
				return backend.CompactForwardResult{}, err
			}
			finalRows[i] = rows
		}
	}
	result, err := packCompactForwardEncodedSequences(req.Shape, seqs, finalRows)
	if err != nil {
		a.stats.UnhandledCalls++
		return backend.CompactForwardResult{}, err
	}
	if a.corruptResult && len(result.Data) > 0 {
		result.Data = result.Data[:len(result.Data)-1]
	}
	a.stats.RunCalls++
	a.stats.PackedDownloads++
	a.stats.PackedBytes += int64(len(result.Data) * 4)
	a.stats.DownloadedBytes += int64(len(result.Data) * 4)
	a.stats.StatusDownloadedBytes += 4
	return result, nil
}

func (a *fakeCompactForwardAccelerator) Stats() backend.CompactForwardAcceleratorStats {
	return a.stats
}

func (a *residentAwareCountingMatMulAccelerator) BindMatrixFromResident(name string, tensor *backend.Tensor, ref backend.OptimizerResidentParameter) error {
	token, ok := ref.Token.(*fakeResidentOptimizerToken)
	if !ok || token == nil || !token.Alive() || token.tensor == nil {
		return fmt.Errorf("invalid fake resident token for %q", name)
	}
	if ref.Elements != len(token.tensor.F32) {
		return fmt.Errorf("resident token %q elements = %d, want %d", name, ref.Elements, len(token.tensor.F32))
	}
	if a.bound == nil {
		a.bound = map[string]*backend.Tensor{}
	}
	a.residentBindCalls++
	a.bound[name] = token.tensor
	return nil
}

type failingResidentBindMatMulAccelerator struct {
	countingMatMulAccelerator
	bindErr error
}

func (a *failingResidentBindMatMulAccelerator) BindMatrixFromResident(name string, tensor *backend.Tensor, ref backend.OptimizerResidentParameter) error {
	if a.bindErr != nil {
		return a.bindErr
	}
	return fmt.Errorf("forced resident bind failure for %q", name)
}

func (a *countingMatMulAccelerator) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}
func (a *countingMatMulAccelerator) RunMatMul(inputs []*backend.Tensor, outputType eosartifact.ValueType) (backend.StepDispatchResult, error) {
	a.runCalls++
	if len(inputs) == 2 && len(inputs[0].Shape) == 3 && len(inputs[1].Shape) == 3 {
		lhs := inputs[0]
		rhs := inputs[1]
		if lhs.Shape[0] == rhs.Shape[0] && lhs.Shape[2] == rhs.Shape[1] {
			if lhs.Shape[0] > a.maxRunBatches {
				a.maxRunBatches = lhs.Shape[0]
			}
			if rhs.Shape[2] > a.maxRunOutputCols {
				a.maxRunOutputCols = rhs.Shape[2]
			}
			out := make([]float32, lhs.Shape[0]*lhs.Shape[1]*rhs.Shape[2])
			a.uploadedBytes += int64((len(lhs.F32) + len(rhs.F32)) * 4)
			a.downloadedBytes += int64(len(out) * 4)
			for batch := 0; batch < lhs.Shape[0]; batch++ {
				lhsBase := batch * lhs.Shape[1] * lhs.Shape[2]
				rhsBase := batch * rhs.Shape[1] * rhs.Shape[2]
				outBase := batch * lhs.Shape[1] * rhs.Shape[2]
				fillHostMatMul(
					lhs.F32[lhsBase:lhsBase+lhs.Shape[1]*lhs.Shape[2]],
					lhs.Shape[1],
					lhs.Shape[2],
					rhs.F32[rhsBase:rhsBase+rhs.Shape[1]*rhs.Shape[2]],
					rhs.Shape[2],
					out[outBase:outBase+lhs.Shape[1]*rhs.Shape[2]],
				)
			}
			return backend.StepDispatchResult{Outputs: []*backend.Tensor{
				backend.NewTensorF32([]int{lhs.Shape[0], lhs.Shape[1], rhs.Shape[2]}, out),
			}}, nil
		}
	}
	return backend.StepDispatchResult{}, nil
}
func (a *countingMatMulAccelerator) RunMatMulWithTranspose(inputs []*backend.Tensor, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	a.runCalls++
	if len(inputs) == 2 && len(inputs[0].Shape) == 3 && len(inputs[1].Shape) == 3 {
		lhs := inputs[0]
		rhs := inputs[1]
		lhsRows, lhsCols := lhs.Shape[1], lhs.Shape[2]
		rhsRows, rhsCols := rhs.Shape[1], rhs.Shape[2]
		outRows, outCols, ok := trainerMatMulShape(lhsRows, lhsCols, rhsRows, rhsCols, transposeLeft, transposeRight)
		if lhs.Shape[0] == rhs.Shape[0] && ok {
			if lhs.Shape[0] > a.maxRunBatches {
				a.maxRunBatches = lhs.Shape[0]
			}
			if outCols > a.maxRunOutputCols {
				a.maxRunOutputCols = outCols
			}
			out := make([]float32, lhs.Shape[0]*outRows*outCols)
			a.uploadedBytes += int64((len(lhs.F32) + len(rhs.F32)) * 4)
			a.downloadedBytes += int64(len(out) * 4)
			for batch := 0; batch < lhs.Shape[0]; batch++ {
				lhsBase := batch * lhsRows * lhsCols
				rhsBase := batch * rhsRows * rhsCols
				outBase := batch * outRows * outCols
				fillHostMatMulTranspose(
					lhs.F32[lhsBase:lhsBase+lhsRows*lhsCols],
					lhsRows,
					lhsCols,
					rhs.F32[rhsBase:rhsBase+rhsRows*rhsCols],
					rhsRows,
					rhsCols,
					transposeLeft,
					transposeRight,
					out[outBase:outBase+outRows*outCols],
				)
			}
			return backend.StepDispatchResult{Outputs: []*backend.Tensor{
				backend.NewTensorF32([]int{lhs.Shape[0], outRows, outCols}, out),
			}}, nil
		}
	}
	if len(inputs) == 2 && len(inputs[0].Shape) == 2 && len(inputs[1].Shape) == 2 {
		lhs := inputs[0]
		rhs := inputs[1]
		lhsRows, lhsCols := lhs.Shape[0], lhs.Shape[1]
		rhsRows, rhsCols := rhs.Shape[0], rhs.Shape[1]
		outRows, outCols, ok := trainerMatMulShape(lhsRows, lhsCols, rhsRows, rhsCols, transposeLeft, transposeRight)
		if ok {
			if outCols > a.maxRunOutputCols {
				a.maxRunOutputCols = outCols
			}
			out := make([]float32, outRows*outCols)
			a.uploadedBytes += int64((len(lhs.F32) + len(rhs.F32)) * 4)
			a.downloadedBytes += int64(len(out) * 4)
			fillHostMatMulTranspose(lhs.F32, lhsRows, lhsCols, rhs.F32, rhsRows, rhsCols, transposeLeft, transposeRight, out)
			return backend.StepDispatchResult{Outputs: []*backend.Tensor{
				backend.NewTensorF32([]int{outRows, outCols}, out),
			}}, nil
		}
	}
	return backend.StepDispatchResult{}, nil
}
func (a *countingMatMulAccelerator) BindMatrix(name string, tensor *backend.Tensor) error {
	if a.bound == nil {
		a.bound = map[string]*backend.Tensor{}
	}
	a.bindCalls++
	a.bound[name] = tensor
	if tensor != nil {
		a.bindUploadedBytes += int64(len(tensor.F32) * 4)
	}
	return nil
}
func (a *countingMatMulAccelerator) UnbindMatrix(name string) error {
	delete(a.bound, name)
	return nil
}
func (a *countingMatMulAccelerator) RunMatMulWithBoundLeft(leftName string, rhs *backend.Tensor, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	a.runCalls++
	return backend.StepDispatchResult{}, nil
}
func (a *countingMatMulAccelerator) RunMatMulWithBoundRight(lhs *backend.Tensor, rightName string, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	a.runCalls++
	a.boundRightRuns++
	if lhs != nil && len(lhs.Shape) > 0 && lhs.Shape[0] > a.maxBoundRightRows {
		a.maxBoundRightRows = lhs.Shape[0]
	}
	if lhs == nil || len(lhs.Shape) != 2 || a.bound == nil {
		return backend.StepDispatchResult{}, nil
	}
	rhs := a.bound[rightName]
	if rhs == nil || len(rhs.Shape) != 2 {
		return backend.StepDispatchResult{}, nil
	}
	outRows, outCols, ok := trainerMatMulShape(lhs.Shape[0], lhs.Shape[1], rhs.Shape[0], rhs.Shape[1], transposeLeft, transposeRight)
	if !ok {
		return backend.StepDispatchResult{}, nil
	}
	out := make([]float32, outRows*outCols)
	fillHostMatMulTranspose(lhs.F32, lhs.Shape[0], lhs.Shape[1], rhs.F32, rhs.Shape[0], rhs.Shape[1], transposeLeft, transposeRight, out)
	a.uploadedBytes += int64(len(lhs.F32) * 4)
	a.downloadedBytes += int64(len(out) * 4)
	return backend.StepDispatchResult{Outputs: []*backend.Tensor{
		backend.NewTensorF32([]int{outRows, outCols}, out),
	}}, nil
}
func (a *countingMatMulAccelerator) RunMatMulWithBoundRights(lhs *backend.Tensor, rightNames []string, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) ([]backend.StepDispatchResult, error) {
	a.runCalls += len(rightNames)
	a.multiBoundRuns++
	a.boundRightRuns += len(rightNames)
	if lhs != nil && len(lhs.Shape) > 0 && lhs.Shape[0] > a.maxBoundRightRows {
		a.maxBoundRightRows = lhs.Shape[0]
	}
	if lhs != nil {
		a.uploadedBytes += int64(len(lhs.F32) * 4)
	}
	results := make([]backend.StepDispatchResult, len(rightNames))
	rows := 0
	inner := 0
	if lhs != nil && len(lhs.Shape) == 2 {
		rows = lhs.Shape[0]
		inner = lhs.Shape[1]
	}
	for i, name := range rightNames {
		cols := inner
		resultRows := rows
		var out []float32
		if a.bound != nil {
			if rhs := a.bound[name]; rhs != nil && len(rhs.Shape) == 2 {
				cols = rhs.Shape[1]
				outRows, outCols, ok := trainerMatMulShape(rows, inner, rhs.Shape[0], rhs.Shape[1], transposeLeft, transposeRight)
				if ok {
					resultRows = outRows
					cols = outCols
					out = make([]float32, outRows*outCols)
					fillHostMatMulTranspose(lhs.F32, lhs.Shape[0], lhs.Shape[1], rhs.F32, rhs.Shape[0], rhs.Shape[1], transposeLeft, transposeRight, out)
					a.downloadedBytes += int64(len(out) * 4)
				}
			}
		}
		if out == nil {
			out = make([]float32, resultRows*cols)
		}
		results[i] = backend.StepDispatchResult{Outputs: []*backend.Tensor{
			backend.NewTensorF32([]int{resultRows, cols}, out),
		}, Metadata: map[string]any{"rhs_binding": name}}
	}
	return results, nil
}
func (a *countingMatMulAccelerator) RunAccumulatedMatMulsWithBoundRights(lhsInputs []*backend.Tensor, rightNames []string, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	a.runCalls++
	a.accumulatedRuns++
	a.boundRightRuns += len(rightNames)
	if len(rightNames) > a.maxAccumTerms {
		a.maxAccumTerms = len(rightNames)
	}
	if len(lhsInputs) != len(rightNames) || len(lhsInputs) == 0 {
		return backend.StepDispatchResult{}, nil
	}
	var out []float32
	var outShape []int
	for i, lhs := range lhsInputs {
		if lhs == nil || len(lhs.Shape) != 2 {
			return backend.StepDispatchResult{}, nil
		}
		if lhs.Shape[0] > a.maxBoundRightRows {
			a.maxBoundRightRows = lhs.Shape[0]
		}
		rhs := (*backend.Tensor)(nil)
		if a.bound != nil {
			rhs = a.bound[rightNames[i]]
		}
		if rhs == nil || len(rhs.Shape) != 2 {
			return backend.StepDispatchResult{}, nil
		}
		outRows, outCols, ok := trainerMatMulShape(lhs.Shape[0], lhs.Shape[1], rhs.Shape[0], rhs.Shape[1], transposeLeft, transposeRight)
		if !ok {
			return backend.StepDispatchResult{}, nil
		}
		if i == 0 {
			outShape = []int{outRows, outCols}
			out = make([]float32, outRows*outCols)
		} else if len(outShape) != 2 || outShape[0] != outRows || outShape[1] != outCols {
			return backend.StepDispatchResult{}, nil
		}
		step := make([]float32, outRows*outCols)
		fillHostMatMulTranspose(lhs.F32, lhs.Shape[0], lhs.Shape[1], rhs.F32, rhs.Shape[0], rhs.Shape[1], transposeLeft, transposeRight, step)
		a.uploadedBytes += int64(len(lhs.F32) * 4)
		a.downloadedBytes += int64(len(step) * 4)
		addFloat32Slice(out, step)
	}
	return backend.StepDispatchResult{Outputs: []*backend.Tensor{
		backend.NewTensorF32(outShape, out),
	}}, nil
}
func (a *countingMatMulAccelerator) RunMatMulsWithSharedLeft(lhs *backend.Tensor, rhs []*backend.Tensor, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) ([]backend.StepDispatchResult, error) {
	a.runCalls++
	a.sharedLeftRuns++
	if len(rhs) > a.maxSharedLeftRHS {
		a.maxSharedLeftRHS = len(rhs)
	}
	results := make([]backend.StepDispatchResult, len(rhs))
	lhsRows := 0
	lhsCols := 0
	if lhs != nil && len(lhs.Shape) == 2 {
		lhsRows = lhs.Shape[0]
		lhsCols = lhs.Shape[1]
	}
	for i, right := range rhs {
		rhsRows := 0
		rhsCols := 0
		if right != nil && len(right.Shape) == 2 {
			rhsRows = right.Shape[0]
			rhsCols = right.Shape[1]
		}
		rows, cols, ok := trainerMatMulShape(lhsRows, lhsCols, rhsRows, rhsCols, transposeLeft, transposeRight)
		if !ok {
			rows, cols = lhsCols, rhsCols
		}
		results[i] = backend.StepDispatchResult{Outputs: []*backend.Tensor{
			backend.NewTensorF32([]int{rows, cols}, make([]float32, rows*cols)),
		}}
	}
	return results, nil
}
func (a *countingMatMulAccelerator) Stats() backend.MatMulAcceleratorStats {
	return backend.MatMulAcceleratorStats{
		BindCalls:          int64(a.bindCalls),
		UploadedBytes:      a.bindUploadedBytes,
		BoundMatrices:      int64(len(a.bound)),
		RunCalls:           int64(a.runCalls),
		BoundRightCalls:    int64(a.boundRightRuns),
		RunUploadedBytes:   a.uploadedBytes,
		RunDownloadedBytes: a.downloadedBytes,
	}
}
func (a *countingMatMulAccelerator) Close() {
	a.closed = true
}

type fakeContrastiveAccelerator struct {
	closed bool
}

func (a *fakeContrastiveAccelerator) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (a *fakeContrastiveAccelerator) RunInfoNCE(query, positive *backend.Tensor, cfg backend.ContrastiveLossConfig) (backend.ContrastiveGradResult, error) {
	return backend.ContrastiveGradResult{}, fmt.Errorf("fake contrastive accelerator is not implemented")
}

func (a *fakeContrastiveAccelerator) RunInfoNCEWithTargets(query, candidates *backend.Tensor, targetIndexes []int, cfg backend.ContrastiveLossConfig) (backend.ContrastiveGradResult, error) {
	return backend.ContrastiveGradResult{}, fmt.Errorf("fake contrastive accelerator is not implemented")
}

func (a *fakeContrastiveAccelerator) Stats() backend.ContrastiveAcceleratorStats {
	return backend.ContrastiveAcceleratorStats{}
}

func (a *fakeContrastiveAccelerator) Close() {
	a.closed = true
}

type countingActivationAccelerator struct {
	bindCalls              int
	unbindCalls            int
	geluBackwardCalls      int
	softmaxBackwardCalls   int
	layerNormBackwardCalls int
	bound                  map[string]*backend.Tensor
	closed                 bool
}

func (a *countingActivationAccelerator) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (a *countingActivationAccelerator) BindTensor(name string, tensor *backend.Tensor) error {
	if a.bound == nil {
		a.bound = map[string]*backend.Tensor{}
	}
	a.bindCalls++
	a.bound[name] = tensor
	return nil
}

func (a *countingActivationAccelerator) UnbindTensor(name string) error {
	a.unbindCalls++
	delete(a.bound, name)
	return nil
}

func (a *countingActivationAccelerator) RunGELUBackwardMul(gradOut, preAct *backend.Tensor) (*backend.Tensor, error) {
	a.geluBackwardCalls++
	if gradOut == nil || preAct == nil {
		return backend.NewTensorF32(nil, nil), nil
	}
	out := make([]float32, len(gradOut.F32))
	for i := range out {
		out[i] = gradOut.F32[i] * geluBackward(preAct.F32[i])
	}
	return backend.NewTensorF32(append([]int(nil), gradOut.Shape...), out), nil
}

func (a *countingActivationAccelerator) RunGELUBackwardMulWithBoundPreAct(gradOut *backend.Tensor, preActName string) (*backend.Tensor, error) {
	preAct := a.bound[preActName]
	return a.RunGELUBackwardMul(gradOut, preAct)
}

func (a *countingActivationAccelerator) RunSoftmaxBackwardRows(gradOut, probs *backend.Tensor) (*backend.Tensor, error) {
	a.softmaxBackwardCalls++
	if gradOut == nil || probs == nil || len(gradOut.Shape) != 2 {
		return backend.NewTensorF32(nil, nil), nil
	}
	out := make([]float32, len(gradOut.F32))
	rows, cols := gradOut.Shape[0], gradOut.Shape[1]
	for row := 0; row < rows; row++ {
		backwardSoftmaxRow(out[row*cols:(row+1)*cols], gradOut.F32[row*cols:(row+1)*cols], probs.F32[row*cols:(row+1)*cols])
	}
	return backend.NewTensorF32(append([]int(nil), gradOut.Shape...), out), nil
}

func (a *countingActivationAccelerator) RunSoftmaxBackwardRowsWithBoundProbs(gradOut *backend.Tensor, probsName string) (*backend.Tensor, error) {
	probs := a.bound[probsName]
	return a.RunSoftmaxBackwardRows(gradOut, probs)
}

func (a *countingActivationAccelerator) RunLayerNormBackwardRows(gradOut, normalized, pre *backend.Tensor) (*backend.Tensor, error) {
	a.layerNormBackwardCalls++
	if gradOut == nil || normalized == nil || pre == nil || len(gradOut.Shape) != 2 {
		return backend.NewTensorF32(nil, nil), nil
	}
	out := make([]float32, len(gradOut.F32))
	rows, cols := gradOut.Shape[0], gradOut.Shape[1]
	for row := 0; row < rows; row++ {
		backwardLayerNormRow(
			out[row*cols:(row+1)*cols],
			gradOut.F32[row*cols:(row+1)*cols],
			normalized.F32[row*cols:(row+1)*cols],
			pre.F32[row*cols:(row+1)*cols],
		)
	}
	return backend.NewTensorF32(append([]int(nil), gradOut.Shape...), out), nil
}

func (a *countingActivationAccelerator) RunLayerNormBackwardRowsWithBoundInputs(gradOut *backend.Tensor, normalizedName, preName string) (*backend.Tensor, error) {
	return a.RunLayerNormBackwardRows(gradOut, a.bound[normalizedName], a.bound[preName])
}

func (a *countingActivationAccelerator) Stats() backend.ActivationAcceleratorStats {
	return backend.ActivationAcceleratorStats{
		BindCalls:              int64(a.bindCalls),
		GELUBackwardCalls:      int64(a.geluBackwardCalls),
		SoftmaxBackwardCalls:   int64(a.softmaxBackwardCalls),
		LayerNormBackwardCalls: int64(a.layerNormBackwardCalls),
		BoundTensors:           int64(len(a.bound)),
	}
}

func (a *countingActivationAccelerator) Close() {
	a.closed = true
}

func TestEmbeddingTrainerRejectsNonTrainableParams(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding")
param projection: q8[D, E] @weight("weights/projection")

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "frozen_train_embed_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	_, err = NewEmbeddingTrainer(bundle.Artifact, tinyMaskedEmbeddingManifest(), map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			0.9, 0.1,
			0.3, 0.7,
			0.6, 0.4,
		}),
		"projection": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	}, EmbeddingTrainConfig{})
	if err == nil {
		t.Fatal("expected non-trainable param error")
	}
	if !strings.Contains(err.Error(), `not marked @trainable`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEmbeddingTrainerTrainStepReducesLossAndExportsQuantizedWeights(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_q8"
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"projection": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	}, EmbeddingTrainConfig{LearningRate: 0.2})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}

	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{1}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{0}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
		{LeftTokens: []int32{1}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
	}

	before, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	for i := 0; i < 32; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}
	after, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval after: %v", err)
	}
	if after.Loss >= before.Loss {
		t.Fatalf("loss did not decrease: before=%f after=%f", before.Loss, after.Loss)
	}

	exported, err := trainer.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export weights: %v", err)
	}
	if got := exported["token_embedding"].DType; got != "q8" {
		t.Fatalf("token_embedding export dtype = %q, want q8", got)
	}
	if got := exported["projection"].DType; got != "q8" {
		t.Fatalf("projection export dtype = %q, want q8", got)
	}

	loadOpts, err := trainer.ExportLoadOptions()
	if err != nil {
		t.Fatalf("export load options: %v", err)
	}
	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, manifest, loadOpts...)
	if err != nil {
		t.Fatalf("load trained model: %v", err)
	}
	result, err := model.Embed(context.Background(), []int32{0})
	if err != nil {
		t.Fatalf("embed trained model: %v", err)
	}
	if result.Embeddings == nil {
		t.Fatal("expected embedding output")
	}
	if got := result.Embeddings.DType; got != "f16" {
		t.Fatalf("embedding dtype = %q, want f16", got)
	}
}

func TestEmbeddingTrainerRejectsUnsupportedCompactArchitecture(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)
	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_q8"
	manifest.ArchitectureVersion = EmbeddingArchitectureCompactTransformerV1
	manifest.ModelDim = 2
	manifest.OutputDim = 2
	manifest.AttentionHeads = 1
	manifest.HeadDim = 2
	manifest.ParameterTying = EmbeddingParameterTyingUntied
	_, err = NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"projection": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	}, EmbeddingTrainConfig{LearningRate: 0.2})
	if err == nil || !strings.Contains(err.Error(), "compact_transformer_v1 is not supported by trainable package initialization yet") {
		t.Fatalf("NewEmbeddingTrainer error = %v, want unsupported compact error", err)
	}
}

func TestEmbeddingTrainerTrainStepSupportsFFNGELUAndExportsQuantizedWeights(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param ffn_up: q8[D, H] @weight("weights/ffn_up") @trainable
param projection: q8[H, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let ffn_up_f = dequant(ffn_up)
    let projection_f = dequant(projection)
    let ffn_hidden = @matmul(hidden, ffn_up_f)
    let activated = gelu(ffn_hidden)
    let projected = @matmul(activated, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let ffn_up_f = dequant(ffn_up)
    let projection_f = dequant(projection)
    let ffn_hidden = @matmul(hidden, ffn_up_f)
    let activated = gelu(ffn_hidden)
    let projected = @matmul(activated, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_ffn_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_ffn_q8"
	manifest.HiddenProjectionParam = "ffn_up"
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"ffn_up": backend.NewTensorF32([]int{2, 3}, []float32{
			1, 0, 1,
			0, 1, 1,
		}),
		"projection": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			0.5, 0.5,
		}),
	}, EmbeddingTrainConfig{LearningRate: 0.05})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}

	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{1}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{0}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
		{LeftTokens: []int32{1}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
	}

	before, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	for i := 0; i < 32; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}
	after, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval after: %v", err)
	}
	if after.Loss >= before.Loss {
		t.Fatalf("ffn loss did not decrease: before=%f after=%f", before.Loss, after.Loss)
	}

	exported, err := trainer.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export weights: %v", err)
	}
	if got := exported["token_embedding"].DType; got != "q8" {
		t.Fatalf("token_embedding export dtype = %q, want q8", got)
	}
	if got := exported["ffn_up"].DType; got != "q8" {
		t.Fatalf("ffn_up export dtype = %q, want q8", got)
	}
	if got := exported["projection"].DType; got != "q8" {
		t.Fatalf("projection export dtype = %q, want q8", got)
	}

	loadOpts, err := trainer.ExportLoadOptions()
	if err != nil {
		t.Fatalf("export load options: %v", err)
	}
	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, manifest, loadOpts...)
	if err != nil {
		t.Fatalf("load trained ffn model: %v", err)
	}
	result, err := model.Embed(context.Background(), []int32{0})
	if err != nil {
		t.Fatalf("embed trained ffn model: %v", err)
	}
	if result.Embeddings == nil {
		t.Fatal("expected embedding output")
	}
	if got := result.Embeddings.DType; got != "f16" {
		t.Fatalf("embedding dtype = %q, want f16", got)
	}
}

func TestEmbeddingTrainerTrainStepSupportsAttentionAndExportsQuantizedWeights(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)

	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{1}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{0}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
		{LeftTokens: []int32{1}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
	}

	before, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	for i := 0; i < 32; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}
	after, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval after: %v", err)
	}
	if after.Loss >= before.Loss {
		t.Fatalf("attention loss did not decrease: before=%f after=%f", before.Loss, after.Loss)
	}

	exported, err := trainer.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export weights: %v", err)
	}
	for _, name := range []string{"token_embedding", "attn_q", "attn_k", "attn_v", "attn_o", "projection"} {
		if got := exported[name].DType; got != "q8" {
			t.Fatalf("%s export dtype = %q, want q8", name, got)
		}
	}
}

func TestEmbeddingTrainerEvaluatePairsImprovesSeparation(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_q8"
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"projection": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	}, EmbeddingTrainConfig{LearningRate: 0.05})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}

	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{1}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{0}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
		{LeftTokens: []int32{1}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
	}

	before, err := trainer.EvaluatePairs(batch)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	for i := 0; i < 24; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}
	after, err := trainer.EvaluatePairs(batch)
	if err != nil {
		t.Fatalf("eval after: %v", err)
	}
	if after.ScoreMargin <= before.ScoreMargin {
		t.Fatalf("score margin did not improve: before=%f after=%f", before.ScoreMargin, after.ScoreMargin)
	}
	if after.PairAccuracy < before.PairAccuracy {
		t.Fatalf("pair accuracy regressed: before=%f after=%f", before.PairAccuracy, after.PairAccuracy)
	}
}

func TestEvalScoreMetricsCalibratePositiveScoreShift(t *testing.T) {
	metrics := EmbeddingEvalMetrics{
		PairAccuracy:  0.5,
		PositiveCount: 2,
		NegativeCount: 2,
	}
	finalizeEvalScoreMetrics(&metrics, []embeddingEvalScore{
		{Score: 0.9, Positive: true},
		{Score: 0.8, Positive: true},
		{Score: 0.7, Positive: false},
		{Score: 0.6, Positive: false},
	})

	assertClose(t, metrics.PairAccuracy, 0.5, 0.000001)
	assertClose(t, metrics.ThresholdAccuracy, 1, 0.000001)
	assertClose(t, metrics.ScoreThreshold, 0.8, 0.000001)
	assertClose(t, metrics.ROCAUC, 1, 0.000001)
}

func TestEvalRankMetricsTrackGroupedPairwiseRetrieval(t *testing.T) {
	metrics := EmbeddingEvalMetrics{}
	finalizeEvalRankMetrics(&metrics, []embeddingEvalRankScore{
		{QueryKey: "q1", Score: 0.8, Positive: true},
		{QueryKey: "q1", Score: 0.2, Positive: false},
		{QueryKey: "q1", Score: 0.1, Positive: false},
		{QueryKey: "q2", Score: 0.3, Positive: true},
		{QueryKey: "q2", Score: 0.9, Positive: false},
		{QueryKey: "q2", Score: 0.4, Positive: false},
		{QueryKey: "q3", Score: 0.9, Positive: false},
	})

	assertClose(t, metrics.Top1Accuracy, 0.5, 0.000001)
	assertClose(t, metrics.Top5Accuracy, 1, 0.000001)
	assertClose(t, metrics.Top10Accuracy, 1, 0.000001)
	assertClose(t, metrics.MeanReciprocalRank, 2.0/3.0, 0.000001)
	assertClose(t, metrics.MeanPositiveRank, 2, 0.000001)
}

func TestEmbeddingTrainerEvaluatePairsTracksGroupedRankingMetrics(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_ranked_eval_embed_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_ranked_eval_embed_q8"
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 0,
		}),
		"projection": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	}, EmbeddingTrainConfig{LearningRate: 0.05})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}

	got, err := trainer.EvaluatePairs([]EmbeddingPairExample{
		{LeftTokens: []int32{0}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{0}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
		{LeftTokens: []int32{1}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
		{LeftTokens: []int32{1}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
	})
	if err != nil {
		t.Fatalf("evaluate pairs: %v", err)
	}

	assertClose(t, got.Top1Accuracy, 1, 0.000001)
	assertClose(t, got.Top5Accuracy, 1, 0.000001)
	assertClose(t, got.Top10Accuracy, 1, 0.000001)
	assertClose(t, got.MeanReciprocalRank, 1, 0.000001)
	assertClose(t, got.MeanPositiveRank, 1, 0.000001)
}

func TestDefaultEmbeddingCheckpointPath(t *testing.T) {
	got := DefaultEmbeddingCheckpointPath("/tmp/tiny_train_embed_q8.mll")
	if want := "/tmp/tiny_train_embed_q8.embed-train.mll"; got != want {
		t.Fatalf("checkpoint path = %q, want %q", got, want)
	}
}

func TestEmbeddingTrainerCheckpointRoundTrip(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_q8"
	cfg := EmbeddingTrainConfig{LearningRate: 0.05}
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"projection": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	}, cfg)
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}

	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{1}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{0}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
		{LeftTokens: []int32{1}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
	}

	for i := 0; i < 4; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}
	checkpoint, err := trainer.Checkpoint()
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if checkpoint.Step != 4 {
		t.Fatalf("checkpoint step = %d, want 4", checkpoint.Step)
	}
	if checkpoint.TokenMoment1 == nil || checkpoint.TokenMoment2 == nil || checkpoint.ProjMoment1 == nil || checkpoint.ProjMoment2 == nil {
		t.Fatal("expected optimizer moments in checkpoint")
	}

	path := filepath.Join(t.TempDir(), "tiny_train_embed_q8.embed-train.mll")
	if err := checkpoint.WriteFile(path); err != nil {
		t.Fatalf("write checkpoint: %v", err)
	}
	loaded, err := ReadEmbeddingTrainCheckpointFile(path)
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	restored, err := NewEmbeddingTrainerFromCheckpoint(bundle.Artifact, loaded)
	if err != nil {
		t.Fatalf("restore trainer: %v", err)
	}

	beforeA, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval original: %v", err)
	}
	beforeB, err := restored.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval restored: %v", err)
	}
	assertClose(t, beforeA.Loss, beforeB.Loss, 0.000001)
	assertClose(t, beforeA.AverageScore, beforeB.AverageScore, 0.000001)

	if _, err := trainer.TrainStep(batch); err != nil {
		t.Fatalf("train original after restore: %v", err)
	}
	if _, err := restored.TrainStep(batch); err != nil {
		t.Fatalf("train restored after restore: %v", err)
	}
	exportA, err := trainer.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export original: %v", err)
	}
	exportB, err := restored.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export restored: %v", err)
	}
	assertTensorClose(t, exportA["token_embedding"], exportB["token_embedding"].Shape, exportB["token_embedding"].F32)
	assertTensorClose(t, exportA["projection"], exportB["projection"].Shape, exportB["projection"].F32)
}

func TestEmbeddingTrainCheckpointRetainsUnknownTensorsAndMoments(t *testing.T) {
	path := filepath.Join(t.TempDir(), "generic.embed-train.mll")
	checkpoint := EmbeddingTrainCheckpoint{
		Version:        EmbeddingTrainCheckpointVersion,
		Manifest:       tinyEmbeddingManifest(),
		Config:         EmbeddingTrainConfig{Optimizer: "adamw"},
		Step:           12,
		TokenEmbedding: backend.NewTensorF32([]int{3, 2}, []float32{1, 0, 0, 1, 1, 1}),
		Projection:     backend.NewTensorF32([]int{2, 2}, []float32{1, 0, 0, 1}),
		TokenMoment1:   backend.NewTensorF32([]int{3, 2}, []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
		TokenMoment2:   backend.NewTensorF32([]int{3, 2}, []float32{0.6, 0.5, 0.4, 0.3, 0.2, 0.1}),
		ProjMoment1:    backend.NewTensorF32([]int{2, 2}, []float32{0.1, 0.2, 0.3, 0.4}),
		ProjMoment2:    backend.NewTensorF32([]int{2, 2}, []float32{0.4, 0.3, 0.2, 0.1}),
		Tensors: map[string]*backend.Tensor{
			"layers.0.attn_q": backend.NewTensorF32([]int{2, 2}, []float32{0.7, 0.8, 0.9, 1.0}),
		},
		MomentTensors: map[string]*backend.Tensor{
			"layers.0.attn_q_moment_1": backend.NewTensorF32([]int{2, 2}, []float32{0.01, 0.02, 0.03, 0.04}),
		},
	}
	if err := checkpoint.WriteFile(path); err != nil {
		t.Fatalf("write checkpoint: %v", err)
	}
	loaded, err := ReadEmbeddingTrainCheckpointFile(path)
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if loaded.Step != checkpoint.Step {
		t.Fatalf("step = %d, want %d", loaded.Step, checkpoint.Step)
	}
	assertTensorClose(t, loaded.TokenEmbedding, checkpoint.TokenEmbedding.Shape, checkpoint.TokenEmbedding.F32)
	assertTensorClose(t, loaded.Projection, checkpoint.Projection.Shape, checkpoint.Projection.F32)
	assertTensorClose(t, loaded.TokenMoment1, checkpoint.TokenMoment1.Shape, checkpoint.TokenMoment1.F32)
	assertTensorClose(t, loaded.ProjMoment2, checkpoint.ProjMoment2.Shape, checkpoint.ProjMoment2.F32)
	gotTensor := loaded.Tensors["layers.0.attn_q"]
	if gotTensor == nil {
		t.Fatalf("missing retained generic tensor: %+v", loaded.Tensors)
	}
	assertTensorClose(t, gotTensor, []int{2, 2}, []float32{0.7, 0.8, 0.9, 1.0})
	gotMoment := loaded.MomentTensors["layers.0.attn_q_moment_1"]
	if gotMoment == nil {
		t.Fatalf("missing retained generic moment tensor: %+v", loaded.MomentTensors)
	}
	assertTensorClose(t, gotMoment, []int{2, 2}, []float32{0.01, 0.02, 0.03, 0.04})
}

func TestEmbeddingTrainCheckpointRetainsCompactGenericTensorsAndMoments(t *testing.T) {
	path := filepath.Join(t.TempDir(), "compact.embed-train.mll")
	checkpoint := EmbeddingTrainCheckpoint{
		Version: EmbeddingTrainCheckpointVersion,
		Manifest: EmbeddingManifest{
			Name:                  "compact",
			ArchitectureVersion:   EmbeddingArchitectureCompactTransformerV1,
			ModelDim:              4,
			OutputDim:             3,
			AttentionHeads:        2,
			HeadDim:               2,
			FFNDim:                8,
			ParameterTying:        EmbeddingParameterTyingUntied,
			TokenEmbeddingParam:   "token_embedding",
			RoleConditioning:      EmbeddingRoleConditioningAdditiveV1,
			RoleEmbeddingParam:    "role_embedding",
			AttentionMaskMode:     EmbeddingAttentionMaskModeKey,
			AttentionScoreScale:   EmbeddingAttentionScoreScaleKeyDimRSQ,
			PositionEncoding:      EmbeddingPositionEncodingRoPE,
			ProjectionParam:       "layer0_ffn_down",
			OutputProjectionParam: "output_projection",
			Tokenizer:             TokenizerManifest{VocabSize: 3, MaxSequence: 2},
		},
		Config: EmbeddingTrainConfig{Optimizer: "adamw"},
		Tensors: map[string]*backend.Tensor{
			"token_embedding":   backend.NewTensorF32([]int{3, 4}, []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}),
			"layer0_attn_q":     backend.NewTensorF32([]int{4, 4}, make([]float32, 16)),
			"output_projection": backend.NewTensorF32([]int{4, 3}, make([]float32, 12)),
		},
		MomentTensors: map[string]*backend.Tensor{
			"token_embedding_moment_1":   backend.NewTensorF32([]int{3, 4}, make([]float32, 12)),
			"layer0_attn_q_moment_1":     backend.NewTensorF32([]int{4, 4}, make([]float32, 16)),
			"output_projection_moment_2": backend.NewTensorF32([]int{4, 3}, make([]float32, 12)),
		},
	}
	if err := checkpoint.WriteFile(path); err != nil {
		t.Fatalf("write checkpoint: %v", err)
	}
	loaded, err := ReadEmbeddingTrainCheckpointFile(path)
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if loaded.TokenEmbedding != nil || loaded.Projection != nil {
		t.Fatalf("compact checkpoint populated legacy fields: token=%v projection=%v", loaded.TokenEmbedding, loaded.Projection)
	}
	assertTensorClose(t, loaded.Tensors["token_embedding"], []int{3, 4}, checkpoint.Tensors["token_embedding"].F32)
	assertTensorClose(t, loaded.Tensors["output_projection"], []int{4, 3}, checkpoint.Tensors["output_projection"].F32)
	if loaded.MomentTensors["token_embedding_moment_1"] == nil || loaded.MomentTensors["output_projection_moment_2"] == nil {
		t.Fatalf("missing compact generic moments: %+v", loaded.MomentTensors)
	}
}

func TestLoadCompactEmbeddingTrainStateFromCheckpointValidatesStructuredState(t *testing.T) {
	checkpoint := compactTrainStateTestCheckpoint(3)
	state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err != nil {
		t.Fatalf("load compact train state: %v", err)
	}
	if len(state.Layers) != 2 {
		t.Fatalf("layer count = %d, want 2", len(state.Layers))
	}
	assertTensorClose(t, state.TokenEmbedding.Tensor, []int{5, 4}, checkpoint.Tensors["token_embedding"].F32)
	assertTensorClose(t, state.RoleEmbedding.Tensor, []int{3, 4}, checkpoint.Tensors["role_embedding"].F32)
	assertTensorClose(t, state.Layers[1].FFNDown.Tensor, []int{6, 4}, checkpoint.Tensors["layer1_ffn_down"].F32)
	assertTensorClose(t, state.OutputProjection.Tensor, []int{4, 3}, checkpoint.Tensors["output_projection"].F32)
	if state.TokenEmbedding.Moment1 == nil || state.Layers[0].AttentionQuery.Moment2 == nil {
		t.Fatalf("expected typed moments: token=%+v layer0q=%+v", state.TokenEmbedding, state.Layers[0].AttentionQuery)
	}
}

func TestLoadCompactEmbeddingTrainStateFromCheckpointRejectsMissingTensor(t *testing.T) {
	checkpoint := compactTrainStateTestCheckpoint(3)
	delete(checkpoint.Tensors, "layer1_attn_v")
	delete(checkpoint.MomentTensors, "layer1_attn_v_moment_1")
	delete(checkpoint.MomentTensors, "layer1_attn_v_moment_2")
	_, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err == nil || !strings.Contains(err.Error(), `tensor "layer1_attn_v" is required`) {
		t.Fatalf("LoadCompactEmbeddingTrainStateFromCheckpoint error = %v, want missing layer1_attn_v", err)
	}
}

func TestLoadCompactEmbeddingTrainStateFromCheckpointRejectsWrongShape(t *testing.T) {
	checkpoint := compactTrainStateTestCheckpoint(3)
	checkpoint.Tensors["layer1_ffn_down"] = backend.NewTensorF32([]int{4, 6}, make([]float32, 24))
	delete(checkpoint.MomentTensors, "layer1_ffn_down_moment_1")
	delete(checkpoint.MomentTensors, "layer1_ffn_down_moment_2")
	_, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err == nil || !strings.Contains(err.Error(), `tensor "layer1_ffn_down" shape [4 6], want [6 4]`) {
		t.Fatalf("LoadCompactEmbeddingTrainStateFromCheckpoint error = %v, want layer1_ffn_down shape error", err)
	}
}

func TestLoadCompactEmbeddingTrainStateFromCheckpointRejectsLegacyCheckpoint(t *testing.T) {
	checkpoint := EmbeddingTrainCheckpoint{
		Version: EmbeddingTrainCheckpointVersion,
		Manifest: EmbeddingManifest{
			Name:                "legacy",
			TokenEmbeddingParam: "token_embedding",
			ProjectionParam:     "projection",
			Tokenizer:           TokenizerManifest{VocabSize: 5},
		},
		TokenEmbedding: backend.NewTensorF32([]int{5, 4}, make([]float32, 20)),
		Projection:     backend.NewTensorF32([]int{4, 4}, make([]float32, 16)),
	}
	_, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err == nil || !strings.Contains(err.Error(), `compact train state requires architecture_version="compact_transformer_v1"`) {
		t.Fatalf("LoadCompactEmbeddingTrainStateFromCheckpoint error = %v, want compact architecture rejection", err)
	}
}

func TestLoadCompactEmbeddingTrainStateFromCheckpointOutputProjectionContract(t *testing.T) {
	withProjection := compactTrainStateTestCheckpoint(3)
	delete(withProjection.Tensors, "output_projection")
	delete(withProjection.MomentTensors, "output_projection_moment_1")
	delete(withProjection.MomentTensors, "output_projection_moment_2")
	_, err := LoadCompactEmbeddingTrainStateFromCheckpoint(withProjection, withProjection.Manifest)
	if err == nil || !strings.Contains(err.Error(), `tensor "output_projection" is required`) {
		t.Fatalf("LoadCompactEmbeddingTrainStateFromCheckpoint error = %v, want required output_projection", err)
	}

	withoutProjection := compactTrainStateTestCheckpoint(4)
	state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(withoutProjection, withoutProjection.Manifest)
	if err != nil {
		t.Fatalf("load compact train state without projection: %v", err)
	}
	if state.OutputProjection != nil {
		t.Fatalf("output projection = %+v, want nil when output_dim == model_dim", state.OutputProjection)
	}
	withoutProjection.Tensors["output_projection"] = backend.NewTensorF32([]int{4, 4}, make([]float32, 16))
	_, err = LoadCompactEmbeddingTrainStateFromCheckpoint(withoutProjection, withoutProjection.Manifest)
	if err == nil || !strings.Contains(err.Error(), `tensor "output_projection" is present but output_dim (4) equals model_dim (4)`) {
		t.Fatalf("LoadCompactEmbeddingTrainStateFromCheckpoint error = %v, want unexpected output_projection error", err)
	}
}

func TestLoadCompactEmbeddingTrainStateFromCheckpointRejectsDuplicateTensorNames(t *testing.T) {
	checkpoint := compactTrainStateTestCheckpoint(6)
	checkpoint.Manifest.OutputProjectionParam = "layer0_ffn_up"
	checkpoint.Tensors["layer0_ffn_up"] = backend.NewTensorF32([]int{4, 6}, make([]float32, 24))
	checkpoint.MomentTensors["layer0_ffn_up_moment_1"] = backend.NewTensorF32([]int{4, 6}, make([]float32, 24))
	checkpoint.MomentTensors["layer0_ffn_up_moment_2"] = backend.NewTensorF32([]int{4, 6}, make([]float32, 24))
	delete(checkpoint.Tensors, "output_projection")
	delete(checkpoint.MomentTensors, "output_projection_moment_1")
	delete(checkpoint.MomentTensors, "output_projection_moment_2")

	_, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err == nil || !strings.Contains(err.Error(), `tensor name "layer0_ffn_up" is used by both layer0_ffn_up and output_projection`) {
		t.Fatalf("LoadCompactEmbeddingTrainStateFromCheckpoint error = %v, want duplicate output projection name rejection", err)
	}
}

func compactTrainStateTestCheckpoint(outputDim int) EmbeddingTrainCheckpoint {
	manifest := EmbeddingManifest{
		Name:                  "compact",
		ArchitectureVersion:   EmbeddingArchitectureCompactTransformerV1,
		EncoderRepeats:        2,
		ModelDim:              4,
		OutputDim:             outputDim,
		AttentionHeads:        2,
		HeadDim:               2,
		FFNDim:                6,
		ParameterTying:        EmbeddingParameterTyingUntied,
		TokenEmbeddingParam:   "token_embedding",
		RoleConditioning:      EmbeddingRoleConditioningAdditiveV1,
		RoleEmbeddingParam:    "role_embedding",
		AttentionMaskMode:     EmbeddingAttentionMaskModeKey,
		AttentionScoreScale:   EmbeddingAttentionScoreScaleKeyDimRSQ,
		PositionEncoding:      EmbeddingPositionEncodingRoPE,
		AttentionQueryParam:   "layer0_attn_q",
		AttentionKeyParam:     "layer0_attn_k",
		AttentionValueParam:   "layer0_attn_v",
		AttentionOutputParam:  "layer0_attn_o",
		HiddenProjectionParam: "layer0_ffn_up",
		ProjectionParam:       "layer0_ffn_down",
		Tokenizer:             TokenizerManifest{VocabSize: 5, MaxSequence: 8},
	}
	if outputDim != manifest.ModelDim {
		manifest.OutputProjectionParam = "output_projection"
	}
	tensors := map[string]*backend.Tensor{
		"token_embedding": compactTrainStateTestTensor([]int{5, 4}, 0.03),
		"role_embedding":  compactTrainStateTestTensor([]int{3, 4}, 0.01),
	}
	for i := 0; i < manifest.EncoderRepeats; i++ {
		offset := float32(i+1) * 0.02
		tensors[compactLayerTensorName(i, "attn_q")] = compactTrainStateTestTensor([]int{4, 4}, 0.04+offset)
		tensors[compactLayerTensorName(i, "attn_k")] = compactTrainStateTestTensor([]int{4, 4}, 0.05+offset)
		tensors[compactLayerTensorName(i, "attn_v")] = compactTrainStateTestTensor([]int{4, 4}, 0.06+offset)
		tensors[compactLayerTensorName(i, "attn_o")] = compactTrainStateTestTensor([]int{4, 4}, 0.07+offset)
		tensors[compactLayerTensorName(i, "ffn_up")] = compactTrainStateTestTensor([]int{4, 6}, 0.08+offset)
		tensors[compactLayerTensorName(i, "ffn_down")] = compactTrainStateTestTensor([]int{6, 4}, 0.09+offset)
	}
	if manifest.OutputProjectionParam != "" {
		tensors[manifest.OutputProjectionParam] = compactTrainStateTestTensor([]int{4, outputDim}, 0.11)
	}
	moments := make(map[string]*backend.Tensor, len(tensors)*2)
	for name, tensor := range tensors {
		moments[name+"_moment_1"] = zeroLikeMaster(tensor)
		moments[name+"_moment_2"] = zeroLikeMaster(tensor)
	}
	return EmbeddingTrainCheckpoint{
		Version:       EmbeddingTrainCheckpointVersion,
		Manifest:      manifest,
		Config:        EmbeddingTrainConfig{Optimizer: "adamw"},
		Tensors:       tensors,
		MomentTensors: moments,
	}
}

func compactTrainStateTestTensor(shape []int, scale float32) *backend.Tensor {
	n := 1
	for _, dim := range shape {
		n *= dim
	}
	data := make([]float32, n)
	for i := range data {
		sign := float32(1)
		if i%2 == 1 {
			sign = -1
		}
		data[i] = sign * scale * float32((i%7)+1)
	}
	return backend.NewTensorF32(shape, data)
}

func newCompactEmbeddingTrainerForTest(t testing.TB, outputDim int) *EmbeddingTrainer {
	t.Helper()
	checkpoint := compactTrainStateTestCheckpoint(outputDim)
	state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err != nil {
		t.Fatalf("load compact state: %v", err)
	}
	trainer, err := newCompactEmbeddingTrainerFromTrainState(&eosartifact.Module{Name: "compact"}, state)
	if err != nil {
		t.Fatalf("new compact trainer: %v", err)
	}
	return trainer
}

func compactTrainStateTestModule(checkpoint EmbeddingTrainCheckpoint) *eosartifact.Module {
	params := make([]eosartifact.Param, 0, len(checkpoint.Tensors))
	for name, tensor := range checkpoint.Tensors {
		dtype := "f32"
		if name == checkpoint.Manifest.OutputProjectionParam {
			dtype = "f16"
		}
		shape := make([]string, len(tensor.Shape))
		for i, dim := range tensor.Shape {
			shape[i] = fmt.Sprintf("%d", dim)
		}
		params = append(params, eosartifact.Param{
			Name:      name,
			Type:      eosartifact.ValueType{Kind: eosartifact.ValueTensor, Tensor: &eosartifact.TensorType{DType: dtype, Shape: shape}},
			Binding:   "weights/" + name,
			Trainable: true,
		})
	}
	return &eosartifact.Module{Name: checkpoint.Manifest.Name, Params: params}
}

func TestCompactEmbeddingTrainerServingPackageVectorParity(t *testing.T) {
	checkpoint := compactTrainStateTestCheckpoint(3)
	checkpoint.Manifest.MaskInput = "attention_mask"
	bundle, err := compiler.Build(compactTrainStateServingSourceForTest(checkpoint.Manifest), compiler.Options{ModuleName: checkpoint.Manifest.Name})
	if err != nil {
		t.Fatalf("build compact serving module: %v", err)
	}
	state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err != nil {
		t.Fatalf("load compact state: %v", err)
	}
	trainer, err := newCompactEmbeddingTrainerFromTrainState(bundle.Artifact, state)
	if err != nil {
		t.Fatalf("new compact trainer: %v", err)
	}
	t.Cleanup(trainer.Close)

	packagePath := filepath.Join(t.TempDir(), "compact_parity.mll")
	if _, err := trainer.WriteEmbeddingPackage(packagePath); err != nil {
		t.Fatalf("write compact embedding package: %v", err)
	}
	model, err := New(cuda.New(), metal.New()).LoadEmbeddingPackage(context.Background(), packagePath)
	if err != nil {
		t.Fatalf("load compact embedding package: %v", err)
	}

	tests := []struct {
		name   string
		tokens []int32
		mask   []int32
		role   string
	}{
		{
			name:   "raw padded",
			tokens: []int32{1, 2, 0},
			mask:   []int32{1, 1, 0},
			role:   EmbeddingRoleRaw,
		},
		{
			name:   "query unpadded",
			tokens: []int32{1, 2, 3},
			mask:   []int32{1, 1, 1},
			role:   EmbeddingRoleQuery,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			roleIndex, err := checkpoint.Manifest.normalized().roleIndex(tt.role)
			if err != nil {
				t.Fatalf("role index: %v", err)
			}
			trainerSeq, err := trainer.encodeCompactSequence(tt.tokens, tt.mask, roleIndex, trainer.prepareCompactForwardWeights())
			if err != nil {
				t.Fatalf("trainer encode compact sequence: %v", err)
			}
			result, err := model.EmbedWithRole(context.Background(), tt.tokens, tt.role)
			if err != nil {
				t.Fatalf("serving embed: %v", err)
			}
			// The package path exports checkpoint master weights through f16 tensors,
			// so serving observes one f16 round-trip that the trainer eval path does not.
			assertFloat32SlicesClose(t, result.Embeddings.F32, trainerSeq.pooled, 3e-4)
		})
	}
}

func compactTrainStateServingSourceForTest(manifest EmbeddingManifest) []byte {
	var b strings.Builder
	b.WriteString("param token_embedding: f16[V, D] @weight(\"weights/token_embedding\") @trainable\n")
	b.WriteString("param role_embedding: f16[3, D] @weight(\"weights/role_embedding\") @trainable\n")
	for i := 0; i < manifest.EncoderRepeats; i++ {
		prefix := fmt.Sprintf("layer%d", i)
		fmt.Fprintf(&b, "param %s_attn_q: f16[D, D] @weight(\"weights/%s_attn_q\") @trainable\n", prefix, prefix)
		fmt.Fprintf(&b, "param %s_attn_k: f16[D, D] @weight(\"weights/%s_attn_k\") @trainable\n", prefix, prefix)
		fmt.Fprintf(&b, "param %s_attn_v: f16[D, D] @weight(\"weights/%s_attn_v\") @trainable\n", prefix, prefix)
		fmt.Fprintf(&b, "param %s_attn_o: f16[D, D] @weight(\"weights/%s_attn_o\") @trainable\n", prefix, prefix)
		fmt.Fprintf(&b, "param %s_ffn_up: f16[D, H] @weight(\"weights/%s_ffn_up\") @trainable\n", prefix, prefix)
		fmt.Fprintf(&b, "param %s_ffn_down: f16[H, D] @weight(\"weights/%s_ffn_down\") @trainable\n", prefix, prefix)
	}
	if manifest.OutputDim != manifest.ModelDim {
		b.WriteString("param output_projection: f16[D, O] @weight(\"weights/output_projection\") @trainable\n")
	}
	b.WriteString("\n")
	b.WriteString("pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T], role_ids: i32[T]) -> f16[O] {\n")
	b.WriteString(compactTrainStateServingPipelineBodyForTest(manifest))
	b.WriteString("}\n\n")
	b.WriteString("pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T], role_ids: i32[B, T]) -> f16[B, O] {\n")
	b.WriteString(compactTrainStateServingPipelineBodyForTest(manifest))
	b.WriteString("}\n")
	return []byte(b.String())
}

func compactTrainStateServingPipelineBodyForTest(manifest EmbeddingManifest) string {
	var b strings.Builder
	b.WriteString("    let hidden_q = gather(token_embedding, tokens)\n")
	b.WriteString("    let role_hidden_q = gather(role_embedding, role_ids)\n")
	b.WriteString("    let hidden_f = dequant(hidden_q)\n")
	b.WriteString("    let role_hidden_f = dequant(role_hidden_q)\n")
	b.WriteString("    let hidden = rope(hidden_f + role_hidden_f)\n")
	prev := "hidden"
	for i := 0; i < manifest.EncoderRepeats; i++ {
		prefix := fmt.Sprintf("layer%d", i)
		fmt.Fprintf(&b, "    let %s_wq = dequant(%s_attn_q)\n", prefix, prefix)
		fmt.Fprintf(&b, "    let %s_wk = dequant(%s_attn_k)\n", prefix, prefix)
		fmt.Fprintf(&b, "    let %s_wv = dequant(%s_attn_v)\n", prefix, prefix)
		fmt.Fprintf(&b, "    let %s_wo = dequant(%s_attn_o)\n", prefix, prefix)
		fmt.Fprintf(&b, "    let %s_ffn_up_f = dequant(%s_ffn_up)\n", prefix, prefix)
		fmt.Fprintf(&b, "    let %s_ffn_down_f = dequant(%s_ffn_down)\n", prefix, prefix)
		fmt.Fprintf(&b, "    let %s_q = @matmul(%s, %s_wq)\n", prefix, prev, prefix)
		fmt.Fprintf(&b, "    let %s_k = @matmul(%s, %s_wk)\n", prefix, prev, prefix)
		fmt.Fprintf(&b, "    let %s_v = @matmul(%s, %s_wv)\n", prefix, prev, prefix)
		fmt.Fprintf(&b, "    let %s_mixed = compact_multihead_attention_h%d(%s_q, %s_k, %s_v, attention_mask)\n", prefix, manifest.AttentionHeads, prefix, prefix, prefix)
		fmt.Fprintf(&b, "    let %s_attended = @matmul(%s_mixed, %s_wo)\n", prefix, prefix, prefix)
		fmt.Fprintf(&b, "    let %s_attn_hidden = layernorm(%s_attended + %s)\n", prefix, prefix, prev)
		fmt.Fprintf(&b, "    let %s_ffn_hidden = @matmul(%s_attn_hidden, %s_ffn_up_f)\n", prefix, prefix, prefix)
		fmt.Fprintf(&b, "    let %s_activated = gelu(%s_ffn_hidden)\n", prefix, prefix)
		fmt.Fprintf(&b, "    let %s_projected = @matmul(%s_activated, %s_ffn_down_f)\n", prefix, prefix, prefix)
		fmt.Fprintf(&b, "    let %s_encoded = layernorm(%s_projected + %s_attn_hidden)\n", prefix, prefix, prefix)
		prev = prefix + "_encoded"
	}
	fmt.Fprintf(&b, "    let normalized = normalize(%s)\n", prev)
	if manifest.OutputDim != manifest.ModelDim {
		b.WriteString("    let output_projection_f = dequant(output_projection)\n")
		b.WriteString("    let output_projected = @matmul(normalized, output_projection_f)\n")
		b.WriteString("    return mean_pool(output_projected, attention_mask)\n")
		return b.String()
	}
	b.WriteString("    return mean_pool(normalized, attention_mask)\n")
	return b.String()
}

func TestCompactEmbeddingTrainerTrainStepMovesState(t *testing.T) {
	checkpoint := compactTrainStateTestCheckpoint(3)
	state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err != nil {
		t.Fatalf("load compact state: %v", err)
	}
	trainer, err := newCompactEmbeddingTrainerFromTrainState(&eosartifact.Module{Name: "compact"}, state)
	if err != nil {
		t.Fatalf("new compact trainer: %v", err)
	}
	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{1, 2, 3}, RightTokens: []int32{1, 2, 3}, Target: 1},
		{LeftTokens: []int32{1, 2, 3}, RightTokens: []int32{3, 2, 1}, Target: -1},
	}
	beforeStep := trainer.TrainProfile().Step
	metrics, err := trainer.EvaluatePairs(batch)
	if err != nil {
		t.Fatalf("evaluate compact pairs: %v", err)
	}
	if !compactTestFinite(metrics.Loss) || !compactTestFinite(metrics.AverageScore) || metrics.PairCount != len(batch) {
		t.Fatalf("compact eval metrics = %+v, want finite pair metrics", metrics)
	}
	if got := trainer.TrainProfile().Step; got != beforeStep {
		t.Fatalf("compact eval step = %d, want unchanged %d", got, beforeStep)
	}
	snapshots := snapshotCompactTrainStateTensors(trainer.compactState)
	trainMetrics, err := trainer.TrainStep(batch)
	if err != nil {
		t.Fatalf("compact TrainStep: %v", err)
	}
	if !compactTestFinite(trainMetrics.Loss) || !compactTestFinite(trainMetrics.AverageScore) || trainMetrics.BatchSize != len(batch) {
		t.Fatalf("compact train metrics = %+v, want finite pair metrics", trainMetrics)
	}
	if got := trainer.TrainProfile().Step; got != beforeStep+1 {
		t.Fatalf("compact train step = %d, want %d", got, beforeStep+1)
	}
	delta := aggregateParameterDeltaStats(snapshots)
	if delta.NonzeroCount == 0 || delta.L2Norm == 0 {
		t.Fatalf("compact TrainStep did not move train tensors: %+v", delta)
	}
	after, err := trainer.EvaluatePairs(batch)
	if err != nil {
		t.Fatalf("evaluate compact pairs after train: %v", err)
	}
	if !compactTestFinite(after.Loss) || !compactTestFinite(after.AverageScore) {
		t.Fatalf("compact post-train eval metrics = %+v, want finite metrics", after)
	}
	if got := trainer.TrainProfile().Step; got != beforeStep+1 {
		t.Fatalf("compact eval after train step = %d, want %d", got, beforeStep+1)
	}
}

func TestCompactEmbeddingTrainerTrainContrastiveStepMovesState(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	batch := compactContrastiveBatchForTest()
	beforeStep := trainer.TrainProfile().Step
	snapshots := snapshotCompactTrainStateTensors(trainer.compactState)
	metrics, err := trainer.TrainContrastiveStep(batch)
	if err != nil {
		t.Fatalf("compact TrainContrastiveStep: %v", err)
	}
	if !compactTestFinite(metrics.Loss) || !compactTestFinite(metrics.AverageScore) || metrics.BatchSize != len(batch)*len(batch) {
		t.Fatalf("compact contrastive metrics = %+v, want finite contrastive metrics", metrics)
	}
	if got := trainer.TrainProfile().Step; got != beforeStep+1 {
		t.Fatalf("compact contrastive step = %d, want %d", got, beforeStep+1)
	}
	delta := aggregateParameterDeltaStats(snapshots)
	if delta.NonzeroCount == 0 || delta.L2Norm == 0 {
		t.Fatalf("compact TrainContrastiveStep did not move train tensors: %+v", delta)
	}
	eval, err := trainer.EvaluateContrastive(batch)
	if err != nil {
		t.Fatalf("evaluate compact contrastive after train: %v", err)
	}
	if !compactTestFinite(eval.Loss) || !compactTestFinite(eval.PositiveMeanScore) || eval.PairCount != len(batch)*len(batch) {
		t.Fatalf("compact contrastive eval = %+v, want finite eval metrics", eval)
	}
}

func TestCompactEmbeddingTrainerTrainContrastiveStepSupportsInfoNCE(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05
	batch := compactContrastiveBatchForTest()
	beforeStep := trainer.TrainProfile().Step
	snapshots := snapshotCompactTrainStateTensors(trainer.compactState)
	metrics, err := trainer.TrainContrastiveStep(batch)
	if err != nil {
		t.Fatalf("compact TrainContrastiveStep InfoNCE: %v", err)
	}
	if !compactTestFinite(metrics.Loss) || !compactTestFinite(metrics.AverageScore) || metrics.BatchSize != len(batch)*len(batch) {
		t.Fatalf("compact InfoNCE metrics = %+v, want finite contrastive metrics", metrics)
	}
	if got := trainer.TrainProfile().Step; got != beforeStep+1 {
		t.Fatalf("compact InfoNCE step = %d, want %d", got, beforeStep+1)
	}
	delta := aggregateParameterDeltaStats(snapshots)
	if delta.NonzeroCount == 0 || delta.L2Norm == 0 {
		t.Fatalf("compact InfoNCE TrainContrastiveStep did not move train tensors: %+v", delta)
	}
}

func TestCompactEmbeddingTrainerTrainHardNegativeContrastiveStepMovesState(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	trainer.config.ContrastiveLoss = "grouped_infonce"
	trainer.config.Temperature = 0.05
	batch := compactHardNegativeBatchForTest()
	beforeProfile := trainer.TrainProfile()
	beforeStep := beforeProfile.Step
	snapshots := snapshotCompactTrainStateTensors(trainer.compactState)
	metrics, err := trainer.TrainHardNegativeContrastiveStep(batch)
	if err != nil {
		t.Fatalf("compact TrainHardNegativeContrastiveStep: %v", err)
	}
	if !compactTestFinite(metrics.Loss) || !compactTestFinite(metrics.AverageScore) || metrics.BatchSize != 4 {
		t.Fatalf("compact hard-negative metrics = %+v, want finite grouped metrics", metrics)
	}
	if got := trainer.TrainProfile().Step; got != beforeStep+1 {
		t.Fatalf("compact hard-negative step = %d, want %d", got, beforeStep+1)
	}
	if got := trainer.TrainProfile().Optimizer.UpdateCalls - beforeProfile.Optimizer.UpdateCalls; got != 1 {
		t.Fatalf("compact hard-negative optimizer updates delta = %d, want 1", got)
	}
	delta := aggregateParameterDeltaStats(snapshots)
	if delta.NonzeroCount == 0 || delta.L2Norm == 0 {
		t.Fatalf("compact TrainHardNegativeContrastiveStep did not move train tensors: %+v", delta)
	}
}

func TestCompactEmbeddingTrainerTrainHardNegativeContrastiveStepSupportsInfoNCE(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05
	batch := compactHardNegativeBatchForTest()
	metrics, err := trainer.TrainHardNegativeContrastiveStep(batch)
	if err != nil {
		t.Fatalf("compact TrainHardNegativeContrastiveStep InfoNCE: %v", err)
	}
	if !compactTestFinite(metrics.Loss) || !compactTestFinite(metrics.AverageScore) || metrics.BatchSize != 8 {
		t.Fatalf("compact hard-negative InfoNCE metrics = %+v, want finite rectangular metrics", metrics)
	}
	if got := trainer.TrainProfile().Step; got != 1 {
		t.Fatalf("compact hard-negative InfoNCE step = %d, want 1", got)
	}
}

func TestCompactEmbeddingTrainerTrainHardNegativeContrastiveStepSupportsDefaultLoss(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	trainer.config.Temperature = 0.05
	metrics, err := trainer.TrainHardNegativeContrastiveStep(compactHardNegativeBatchForTest())
	if err != nil {
		t.Fatalf("compact TrainHardNegativeContrastiveStep default loss: %v", err)
	}
	if !compactTestFinite(metrics.Loss) || !compactTestFinite(metrics.AverageScore) || metrics.BatchSize != 8 {
		t.Fatalf("compact hard-negative default metrics = %+v, want finite rectangular metrics", metrics)
	}
	if got := trainer.TrainProfile().Step; got != 1 {
		t.Fatalf("compact hard-negative default step = %d, want 1", got)
	}
}

func TestCompactEmbeddingTrainerHardNegativeSupportsTeacherDistribution(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	trainer.config.ContrastiveLoss = "grouped_infonce"
	trainer.config.Temperature = 0.05
	trainer.config.TeacherTemperature = 1
	trainer.config.TeacherLossWeight = 0.5
	batch := compactHardNegativeBatchForTest()
	for i := range batch {
		batch[i].TeacherScores = []float32{1.2, 0.1}
	}
	beforeProfile := trainer.TrainProfile()
	metrics, err := trainer.TrainHardNegativeContrastiveStep(batch)
	if err != nil {
		t.Fatalf("compact hard-negative teacher distribution: %v", err)
	}
	if !compactTestFinite(metrics.Loss) || !compactTestFinite(metrics.AverageScore) || metrics.BatchSize != 8 {
		t.Fatalf("compact hard-negative teacher metrics = %+v, want finite metrics with teacher pairs", metrics)
	}
	if got := trainer.TrainProfile().Step; got != beforeProfile.Step+1 {
		t.Fatalf("compact hard-negative teacher step = %d, want %d", got, beforeProfile.Step+1)
	}
	if got := trainer.TrainProfile().Optimizer.UpdateCalls - beforeProfile.Optimizer.UpdateCalls; got != 1 {
		t.Fatalf("compact hard-negative teacher optimizer updates delta = %d, want 1", got)
	}
}

func TestCompactEmbeddingTrainerHardNegativeRejectsMalformedTeacherScoresBeforeMutation(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	trainer.config.ContrastiveLoss = "grouped_infonce"
	trainer.config.Temperature = 0.05
	trainer.config.TeacherLossWeight = 0.5
	batch := compactHardNegativeBatchForTest()
	batch[0].TeacherScores = []float32{1.2}
	beforeProfile := trainer.TrainProfile()
	_, err := trainer.TrainHardNegativeContrastiveStep(batch)
	if err == nil || !strings.Contains(err.Error(), "hard-negative teacher_scores length 1 does not match candidate count 2 for batch 0") {
		t.Fatalf("compact malformed teacher_scores error = %v", err)
	}
	if got := trainer.TrainProfile().Step; got != beforeProfile.Step {
		t.Fatalf("compact malformed teacher_scores mutated step = %d, want %d", got, beforeProfile.Step)
	}
	if got := trainer.TrainProfile().Optimizer.UpdateCalls - beforeProfile.Optimizer.UpdateCalls; got != 0 {
		t.Fatalf("compact malformed teacher_scores optimizer updates delta = %d, want 0", got)
	}
}

func TestCompactEmbeddingTrainerHardNegativeTeacherSourceWeightZeroSuppressesTeacherPairs(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	trainer.config.ContrastiveLoss = "grouped_infonce"
	trainer.config.Temperature = 0.05
	trainer.config.TeacherLossWeight = 0.5
	trainer.config.TeacherSourceWeights = map[string]float32{"*": 0}
	batch := compactHardNegativeBatchForTest()
	for i := range batch {
		batch[i].Source = "fiqa:model"
		batch[i].TeacherScores = []float32{1.2, 0.1}
	}
	metrics, err := trainer.TrainHardNegativeContrastiveStep(batch)
	if err != nil {
		t.Fatalf("compact hard-negative teacher source weight zero: %v", err)
	}
	if !compactTestFinite(metrics.Loss) || !compactTestFinite(metrics.AverageScore) || metrics.BatchSize != 4 {
		t.Fatalf("compact hard-negative source weight zero metrics = %+v, want base grouped pair count", metrics)
	}
	if got := trainer.TrainProfile().Step; got != 1 {
		t.Fatalf("compact hard-negative source weight zero step = %d, want 1", got)
	}
}

func TestCompactEmbeddingTrainerHardNegativeCheckpointRestoreParity(t *testing.T) {
	checkpoint := compactTrainStateTestCheckpoint(3)
	checkpoint.Step = 13
	mod := compactTrainStateTestModule(checkpoint)
	state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err != nil {
		t.Fatalf("load compact state: %v", err)
	}
	trainer, err := newCompactEmbeddingTrainerFromTrainState(mod, state)
	if err != nil {
		t.Fatalf("new compact trainer: %v", err)
	}
	trainer.config.ContrastiveLoss = "grouped_infonce"
	trainer.config.Temperature = 0.05
	batch := compactHardNegativeBatchForTest()
	if _, err := trainer.TrainHardNegativeContrastiveStep(batch); err != nil {
		t.Fatalf("train compact hard-negative before checkpoint: %v", err)
	}
	evalPairs, err := BuildEmbeddingHardNegativeEvalPairs(batch, 1)
	if err != nil {
		t.Fatalf("build hard-negative eval pairs: %v", err)
	}
	before, err := trainer.EvaluatePairs(evalPairs)
	if err != nil {
		t.Fatalf("evaluate compact hard-negative pairs before checkpoint: %v", err)
	}
	saved, err := trainer.Checkpoint()
	if err != nil {
		t.Fatalf("checkpoint compact hard-negative trainer: %v", err)
	}
	if saved.Step != checkpoint.Step+1 {
		t.Fatalf("checkpoint step = %d, want %d", saved.Step, checkpoint.Step+1)
	}
	restored, err := NewEmbeddingTrainerFromCheckpoint(mod, saved)
	if err != nil {
		t.Fatalf("restore compact hard-negative trainer: %v", err)
	}
	after, err := restored.EvaluatePairs(evalPairs)
	if err != nil {
		t.Fatalf("evaluate restored compact hard-negative pairs: %v", err)
	}
	assertClose(t, before.Loss, after.Loss, 0.000001)
	assertClose(t, before.AverageScore, after.AverageScore, 0.000001)
	if got := restored.TrainProfile().Step; got != checkpoint.Step+1 {
		t.Fatalf("restored compact hard-negative step = %d, want %d", got, checkpoint.Step+1)
	}
}

func TestCompactEmbeddingTrainerTrainScoreSpectrumStepMovesState(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	trainer.config.Temperature = 0.05
	batch := tinyEmbeddingScoreSpectrumDataset()
	beforeStep := trainer.TrainProfile().Step
	snapshots := snapshotCompactTrainStateTensors(trainer.compactState)
	metrics, err := trainer.TrainScoreSpectrumStep(batch)
	if err != nil {
		t.Fatalf("compact TrainScoreSpectrumStep: %v", err)
	}
	if !compactTestFinite(metrics.Loss) || !compactTestFinite(metrics.AverageScore) || metrics.BatchSize != 4 {
		t.Fatalf("compact score-spectrum metrics = %+v, want finite row-local metrics", metrics)
	}
	if got := trainer.TrainProfile().Step; got != beforeStep+1 {
		t.Fatalf("compact score-spectrum step = %d, want %d", got, beforeStep+1)
	}
	delta := aggregateParameterDeltaStats(snapshots)
	if delta.NonzeroCount == 0 || delta.L2Norm == 0 {
		t.Fatalf("compact TrainScoreSpectrumStep did not move train tensors: %+v", delta)
	}
	eval, err := trainer.EvaluateScoreSpectrum(batch)
	if err != nil {
		t.Fatalf("evaluate compact score-spectrum after train: %v", err)
	}
	if !compactTestFinite(eval.Loss) || !compactTestFinite(eval.AverageScore) || eval.CandidateCount != 4 {
		t.Fatalf("compact score-spectrum eval = %+v, want finite eval metrics", eval)
	}
}

func TestCompactEmbeddingTrainerScoreSpectrumCheckpointRestoreParity(t *testing.T) {
	checkpoint := compactTrainStateTestCheckpoint(3)
	checkpoint.Step = 17
	mod := compactTrainStateTestModule(checkpoint)
	state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err != nil {
		t.Fatalf("load compact state: %v", err)
	}
	trainer, err := newCompactEmbeddingTrainerFromTrainState(mod, state)
	if err != nil {
		t.Fatalf("new compact trainer: %v", err)
	}
	trainer.config.Temperature = 0.05
	batch := tinyEmbeddingScoreSpectrumDataset()
	if _, err := trainer.TrainScoreSpectrumStep(batch); err != nil {
		t.Fatalf("train compact score-spectrum before checkpoint: %v", err)
	}
	before, err := trainer.EvaluateScoreSpectrum(batch)
	if err != nil {
		t.Fatalf("evaluate compact score-spectrum before checkpoint: %v", err)
	}
	saved, err := trainer.Checkpoint()
	if err != nil {
		t.Fatalf("checkpoint compact score-spectrum trainer: %v", err)
	}
	if saved.Step != checkpoint.Step+1 {
		t.Fatalf("checkpoint step = %d, want %d", saved.Step, checkpoint.Step+1)
	}
	restored, err := NewEmbeddingTrainerFromCheckpoint(mod, saved)
	if err != nil {
		t.Fatalf("restore compact score-spectrum trainer: %v", err)
	}
	after, err := restored.EvaluateScoreSpectrum(batch)
	if err != nil {
		t.Fatalf("evaluate restored compact score-spectrum: %v", err)
	}
	assertScoreSpectrumEvalMetricsClose(t, after, before)
	if got := restored.TrainProfile().Step; got != checkpoint.Step+1 {
		t.Fatalf("restored compact score-spectrum step = %d, want %d", got, checkpoint.Step+1)
	}
}

func TestCompactEmbeddingTrainerTrainListwiseGeometryStepMovesState(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	trainer.config.Temperature = 0.05
	batch := tinyTokenizedListwiseGeometryBatches(false)
	beforeStep := trainer.TrainProfile().Step
	snapshots := snapshotCompactTrainStateTensors(trainer.compactState)
	metrics, err := trainer.TrainListwiseGeometryStepWithDiagnostics(batch, true)
	if err != nil {
		t.Fatalf("compact TrainListwiseGeometryStepWithDiagnostics: %v", err)
	}
	if !compactTestFinite(metrics.Loss) || !compactTestFinite(metrics.AverageScore) || metrics.BatchSize != 4 {
		t.Fatalf("compact listwise metrics = %+v, want finite row-local metrics", metrics)
	}
	if metrics.Movement == nil {
		t.Fatalf("compact listwise diagnostics missing")
	}
	if !compactTestFinite(metrics.Movement.Gradient.L2Norm) || metrics.Movement.Gradient.L2Norm <= 0 || metrics.Movement.Gradient.NonzeroCount <= 0 {
		t.Fatalf("compact listwise gradient movement = %+v, want finite nonzero aggregate", metrics.Movement.Gradient)
	}
	if !compactTestFinite(metrics.Movement.ParameterDelta.L2Norm) || metrics.Movement.ParameterDelta.L2Norm <= 0 || metrics.Movement.ParameterDelta.NonzeroCount <= 0 {
		t.Fatalf("compact listwise parameter delta = %+v, want finite nonzero aggregate", metrics.Movement.ParameterDelta)
	}
	if got := trainer.TrainProfile().Step; got != beforeStep+1 {
		t.Fatalf("compact listwise step = %d, want %d", got, beforeStep+1)
	}
	delta := aggregateParameterDeltaStats(snapshots)
	if delta.NonzeroCount == 0 || delta.L2Norm == 0 {
		t.Fatalf("compact TrainListwiseGeometryStepWithDiagnostics did not move train tensors: %+v", delta)
	}
	eval, err := trainer.EvaluateListwiseGeometry(batch)
	if err != nil {
		t.Fatalf("evaluate compact listwise after train: %v", err)
	}
	if !compactTestFinite(eval.Loss) || !compactTestFinite(eval.AverageScore) || eval.QueryCount != 2 || eval.DocumentCellCount != 4 {
		t.Fatalf("compact listwise eval = %+v, want finite eval metrics", eval)
	}
}

func TestCompactEmbeddingTrainerListwiseGeometryCheckpointRestoreParity(t *testing.T) {
	checkpoint := compactTrainStateTestCheckpoint(3)
	checkpoint.Step = 19
	mod := compactTrainStateTestModule(checkpoint)
	state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err != nil {
		t.Fatalf("load compact state: %v", err)
	}
	trainer, err := newCompactEmbeddingTrainerFromTrainState(mod, state)
	if err != nil {
		t.Fatalf("new compact trainer: %v", err)
	}
	trainer.config.Temperature = 0.05
	batch := tinyTokenizedListwiseGeometryBatches(false)
	if _, err := trainer.TrainListwiseGeometryStep(batch); err != nil {
		t.Fatalf("train compact listwise before checkpoint: %v", err)
	}
	before, err := trainer.EvaluateListwiseGeometry(batch)
	if err != nil {
		t.Fatalf("evaluate compact listwise before checkpoint: %v", err)
	}
	saved, err := trainer.Checkpoint()
	if err != nil {
		t.Fatalf("checkpoint compact listwise trainer: %v", err)
	}
	if saved.Step != checkpoint.Step+1 {
		t.Fatalf("checkpoint step = %d, want %d", saved.Step, checkpoint.Step+1)
	}
	restored, err := NewEmbeddingTrainerFromCheckpoint(mod, saved)
	if err != nil {
		t.Fatalf("restore compact listwise trainer: %v", err)
	}
	after, err := restored.EvaluateListwiseGeometry(batch)
	if err != nil {
		t.Fatalf("evaluate restored compact listwise: %v", err)
	}
	assertListwiseGeometryEvalMetricsClose(t, after, before)
	if got := restored.TrainProfile().Step; got != checkpoint.Step+1 {
		t.Fatalf("restored compact listwise step = %d, want %d", got, checkpoint.Step+1)
	}
}

func TestCompactEmbeddingTrainerListwiseGeometryUnsupportedModes(t *testing.T) {
	for name, tc := range map[string]struct {
		configure func(*EmbeddingTrainer)
		want      string
	}{
		"matryoshka": {
			configure: func(trainer *EmbeddingTrainer) {
				trainer.config.MatryoshkaDims = []int{2}
				trainer.config.MatryoshkaWeights = []float32{0.5}
			},
			want: "does not support matryoshka objectives",
		},
		"turboquant_prefix_bits": {
			configure: func(trainer *EmbeddingTrainer) {
				trainer.config.TurboQuantPrefixBits = []int{2}
			},
			want: "does not support turboquant prefix objectives",
		},
		"turboquant_prefix_objectives": {
			configure: func(trainer *EmbeddingTrainer) {
				trainer.config.TurboQuantPrefixObjectives = []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.25}}
			},
			want: "does not support turboquant prefix objectives",
		},
		"turboquant_compact": {
			configure: func(trainer *EmbeddingTrainer) {
				trainer.config.TurboQuantCompactObjectives = []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.25}}
			},
			want: "does not support turboquant compact objectives",
		},
		"turboquant_rank_margin": {
			configure: func(trainer *EmbeddingTrainer) {
				trainer.config.TurboQuantRankMarginObjectives = []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.25}}
			},
			want: "does not support turboquant rank-margin objectives",
		},
	} {
		trainer := newCompactEmbeddingTrainerForTest(t, 3)
		trainer.config.Temperature = 0.05
		tc.configure(trainer)
		beforeStep := trainer.TrainProfile().Step
		_, err := trainer.TrainListwiseGeometryStepWithDiagnostics(tinyTokenizedListwiseGeometryBatches(false), true)
		if err == nil || !strings.Contains(err.Error(), "compact_transformer_v1 listwise geometry training") || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("%s compact listwise error = %v, want %q", name, err, tc.want)
		}
		if got := trainer.TrainProfile().Step; got != beforeStep {
			t.Fatalf("%s compact unsupported listwise mutated step = %d, want %d", name, got, beforeStep)
		}
	}
}

func TestCompactEmbeddingTrainerHardNegativeUnsupportedModes(t *testing.T) {
	for name, tc := range map[string]struct {
		configure func(*EmbeddingTrainer, []EmbeddingHardNegativeExample) []EmbeddingHardNegativeExample
		want      string
	}{
		"hybrid_loss": {
			configure: func(trainer *EmbeddingTrainer, batch []EmbeddingHardNegativeExample) []EmbeddingHardNegativeExample {
				trainer.config.ContrastiveLoss = "hybrid_infonce"
				return batch
			},
			want: `does not support contrastive_loss "hybrid_infonce"`,
		},
		"matryoshka": {
			configure: func(trainer *EmbeddingTrainer, batch []EmbeddingHardNegativeExample) []EmbeddingHardNegativeExample {
				trainer.config.MatryoshkaDims = []int{2}
				trainer.config.MatryoshkaWeights = []float32{0.5}
				return batch
			},
			want: "supports InfoNCE and grouped InfoNCE only",
		},
		"turboquant_prefix_bits": {
			configure: func(trainer *EmbeddingTrainer, batch []EmbeddingHardNegativeExample) []EmbeddingHardNegativeExample {
				trainer.config.TurboQuantPrefixBits = []int{2}
				return batch
			},
			want: "supports InfoNCE and grouped InfoNCE only",
		},
		"turboquant_prefix_objectives": {
			configure: func(trainer *EmbeddingTrainer, batch []EmbeddingHardNegativeExample) []EmbeddingHardNegativeExample {
				trainer.config.TurboQuantPrefixObjectives = []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.25}}
				return batch
			},
			want: "supports InfoNCE and grouped InfoNCE only",
		},
		"turboquant_compact": {
			configure: func(trainer *EmbeddingTrainer, batch []EmbeddingHardNegativeExample) []EmbeddingHardNegativeExample {
				trainer.config.TurboQuantCompactObjectives = []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.25}}
				return batch
			},
			want: "supports InfoNCE and grouped InfoNCE only",
		},
		"turboquant_rank_margin": {
			configure: func(trainer *EmbeddingTrainer, batch []EmbeddingHardNegativeExample) []EmbeddingHardNegativeExample {
				trainer.config.TurboQuantRankMarginObjectives = []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.25}}
				return batch
			},
			want: "supports InfoNCE and grouped InfoNCE only",
		},
	} {
		trainer := newCompactEmbeddingTrainerForTest(t, 3)
		beforeStep := trainer.TrainProfile().Step
		batch := tc.configure(trainer, compactHardNegativeBatchForTest())
		_, err := trainer.TrainHardNegativeContrastiveStep(batch)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("%s compact hard-negative error = %v, want %q", name, err, tc.want)
		}
		if got := trainer.TrainProfile().Step; got != beforeStep {
			t.Fatalf("%s compact unsupported hard-negative mutated step = %d, want %d", name, got, beforeStep)
		}
	}
}

func TestCompactEmbeddingTrainerScoreSpectrumUnsupportedModes(t *testing.T) {
	for name, tc := range map[string]struct {
		configure func(*EmbeddingTrainer)
		want      string
	}{
		"matryoshka": {
			configure: func(trainer *EmbeddingTrainer) {
				trainer.config.MatryoshkaDims = []int{2}
				trainer.config.MatryoshkaWeights = []float32{0.5}
			},
			want: "does not support matryoshka objectives",
		},
		"turboquant_prefix_bits": {
			configure: func(trainer *EmbeddingTrainer) {
				trainer.config.TurboQuantPrefixBits = []int{2}
			},
			want: "does not support turboquant prefix objectives",
		},
		"turboquant_prefix_objectives": {
			configure: func(trainer *EmbeddingTrainer) {
				trainer.config.TurboQuantPrefixObjectives = []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.25}}
			},
			want: "does not support turboquant prefix objectives",
		},
		"turboquant_compact": {
			configure: func(trainer *EmbeddingTrainer) {
				trainer.config.TurboQuantCompactObjectives = []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.25}}
			},
			want: "does not support turboquant compact objectives",
		},
		"turboquant_rank_margin": {
			configure: func(trainer *EmbeddingTrainer) {
				trainer.config.TurboQuantRankMarginObjectives = []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.25}}
			},
			want: "does not support turboquant rank-margin objectives",
		},
	} {
		trainer := newCompactEmbeddingTrainerForTest(t, 3)
		trainer.config.Temperature = 0.05
		tc.configure(trainer)
		beforeStep := trainer.TrainProfile().Step
		_, err := trainer.TrainScoreSpectrumStep(tinyEmbeddingScoreSpectrumDataset())
		if err == nil || !strings.Contains(err.Error(), "compact_transformer_v1 score-spectrum training") || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("%s compact score-spectrum error = %v, want %q", name, err, tc.want)
		}
		if got := trainer.TrainProfile().Step; got != beforeStep {
			t.Fatalf("%s compact unsupported score-spectrum mutated step = %d, want %d", name, got, beforeStep)
		}
	}
}

func TestCompactEmbeddingTrainerContrastiveCheckpointRestoreParity(t *testing.T) {
	checkpoint := compactTrainStateTestCheckpoint(3)
	checkpoint.Step = 11
	mod := compactTrainStateTestModule(checkpoint)
	state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err != nil {
		t.Fatalf("load compact state: %v", err)
	}
	trainer, err := newCompactEmbeddingTrainerFromTrainState(mod, state)
	if err != nil {
		t.Fatalf("new compact trainer: %v", err)
	}
	batch := compactContrastiveBatchForTest()
	if _, err := trainer.TrainContrastiveStep(batch); err != nil {
		t.Fatalf("train compact contrastive before checkpoint: %v", err)
	}
	before, err := trainer.EvaluateContrastive(batch)
	if err != nil {
		t.Fatalf("evaluate compact contrastive before checkpoint: %v", err)
	}
	saved, err := trainer.Checkpoint()
	if err != nil {
		t.Fatalf("checkpoint compact contrastive trainer: %v", err)
	}
	if saved.Step != checkpoint.Step+1 {
		t.Fatalf("checkpoint step = %d, want %d", saved.Step, checkpoint.Step+1)
	}
	restored, err := NewEmbeddingTrainerFromCheckpoint(mod, saved)
	if err != nil {
		t.Fatalf("restore compact contrastive trainer: %v", err)
	}
	after, err := restored.EvaluateContrastive(batch)
	if err != nil {
		t.Fatalf("evaluate restored compact contrastive: %v", err)
	}
	assertClose(t, before.Loss, after.Loss, 0.000001)
	assertClose(t, before.PositiveMeanScore, after.PositiveMeanScore, 0.000001)
	assertClose(t, before.ScoreMargin, after.ScoreMargin, 0.000001)
	if got := restored.TrainProfile().Step; got != checkpoint.Step+1 {
		t.Fatalf("restored compact step = %d, want %d", got, checkpoint.Step+1)
	}
}

func TestCompactEmbeddingTrainerContrastivePrefixObjectivesRemainUnsupported(t *testing.T) {
	for name, configure := range map[string]func(*EmbeddingTrainer){
		"matryoshka": func(trainer *EmbeddingTrainer) {
			trainer.config.MatryoshkaDims = []int{2}
			trainer.config.MatryoshkaWeights = []float32{0.5}
		},
		"turboquant_prefix_bits": func(trainer *EmbeddingTrainer) {
			trainer.config.TurboQuantPrefixBits = []int{2}
		},
		"turboquant_prefix_objectives": func(trainer *EmbeddingTrainer) {
			trainer.config.TurboQuantPrefixObjectives = []TurboQuantPrefixObjective{{Dim: 2, BitWidth: 2, Weight: 0.25}}
		},
	} {
		trainer := newCompactEmbeddingTrainerForTest(t, 3)
		configure(trainer)
		beforeStep := trainer.TrainProfile().Step
		_, err := trainer.TrainContrastiveStep(compactContrastiveBatchForTest())
		if err == nil || !strings.Contains(err.Error(), "compact_transformer_v1 contrastive training supports pair-MSE and InfoNCE only") {
			t.Fatalf("%s compact contrastive error = %v, want explicit unsupported", name, err)
		}
		if got := trainer.TrainProfile().Step; got != beforeStep {
			t.Fatalf("%s compact unsupported contrastive mutated step = %d, want %d", name, got, beforeStep)
		}
	}
}

func compactContrastiveBatchForTest() []EmbeddingContrastiveExample {
	return []EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2, 3}, PositiveTokens: []int32{1, 2, 3}, QueryMask: []int32{1, 1, 1}, PositiveMask: []int32{1, 1, 1}},
		{QueryTokens: []int32{3, 2, 1}, PositiveTokens: []int32{3, 1, 2}, QueryMask: []int32{1, 1, 1}, PositiveMask: []int32{1, 1, 1}},
	}
}

func compactHardNegativeBatchForTest() []EmbeddingHardNegativeExample {
	return []EmbeddingHardNegativeExample{
		{QueryTokens: []int32{1, 2, 3}, PositiveTokens: []int32{1, 2, 3}, NegativeTokens: [][]int32{{3, 2, 1}}, QueryMask: []int32{1, 1, 1}, PositiveMask: []int32{1, 1, 1}, NegativeMasks: [][]int32{{1, 1, 1}}},
		{QueryTokens: []int32{3, 2, 1}, PositiveTokens: []int32{3, 1, 2}, NegativeTokens: [][]int32{{1, 3, 2}}, QueryMask: []int32{1, 1, 1}, PositiveMask: []int32{1, 1, 1}, NegativeMasks: [][]int32{{1, 1, 1}}},
	}
}

func TestCompactEmbeddingTrainerCheckpointRestoreAndExportRoundTrip(t *testing.T) {
	checkpoint := compactTrainStateTestCheckpoint(3)
	checkpoint.Step = 7
	checkpoint.Config = EmbeddingTrainConfig{Optimizer: "adamw", LearningRate: 0.003, Temperature: 0.07}
	mod := compactTrainStateTestModule(checkpoint)
	state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err != nil {
		t.Fatalf("load compact state: %v", err)
	}
	trainer, err := newCompactEmbeddingTrainerFromTrainState(mod, state)
	if err != nil {
		t.Fatalf("new compact trainer: %v", err)
	}
	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{1, 2, 3}, RightTokens: []int32{1, 2, 3}, Target: 1},
		{LeftTokens: []int32{1, 2, 3}, RightTokens: []int32{3, 2, 1}, Target: -1},
	}
	snapshots := snapshotCompactTrainStateTensors(trainer.compactState)
	if _, err := trainer.TrainStep(batch); err != nil {
		t.Fatalf("train compact before checkpoint: %v", err)
	}
	delta := aggregateParameterDeltaStats(snapshots)
	if delta.NonzeroCount == 0 || delta.L2Norm == 0 {
		t.Fatalf("compact TrainStep before checkpoint did not move tensors: %+v", delta)
	}
	before, err := trainer.EvaluatePairs(batch)
	if err != nil {
		t.Fatalf("evaluate compact before checkpoint: %v", err)
	}
	saved, err := trainer.Checkpoint()
	if err != nil {
		t.Fatalf("checkpoint compact trainer: %v", err)
	}
	if saved.TokenEmbedding != nil || saved.Projection != nil {
		t.Fatalf("compact checkpoint populated legacy fields: token=%v projection=%v", saved.TokenEmbedding, saved.Projection)
	}
	if saved.Step != checkpoint.Step+1 {
		t.Fatalf("checkpoint step = %d, want %d", saved.Step, checkpoint.Step+1)
	}
	if saved.Config.LearningRate != checkpoint.Config.LearningRate || saved.Config.Temperature != checkpoint.Config.Temperature {
		t.Fatalf("checkpoint config = %+v, want %+v", saved.Config, checkpoint.Config)
	}
	if saved.Manifest.ArchitectureVersion != EmbeddingArchitectureCompactTransformerV1 ||
		saved.Manifest.EncoderRepeats != 2 ||
		saved.Manifest.OutputProjectionParam != "output_projection" {
		t.Fatalf("checkpoint compact manifest not preserved: %+v", saved.Manifest)
	}
	for _, name := range []string{"token_embedding", "role_embedding", "layer0_attn_q", "layer1_ffn_down", "output_projection"} {
		if saved.Tensors[name] == nil || !tensorShapeEquals(saved.Tensors[name], checkpoint.Tensors[name].Shape) {
			t.Fatalf("checkpoint tensor %q shape = %v, want %v", name, tensorShapeForError(saved.Tensors[name]), checkpoint.Tensors[name].Shape)
		}
		if saved.MomentTensors[name+"_moment_1"] == nil || saved.MomentTensors[name+"_moment_2"] == nil {
			t.Fatalf("checkpoint missing compact moments for %q: %+v", name, saved.MomentTensors)
		}
	}

	path := filepath.Join(t.TempDir(), "compact.embed-train.mll")
	if err := saved.WriteFile(path); err != nil {
		t.Fatalf("write compact checkpoint: %v", err)
	}
	loaded, err := ReadEmbeddingTrainCheckpointFile(path)
	if err != nil {
		t.Fatalf("read compact checkpoint: %v", err)
	}
	restored, err := NewEmbeddingTrainerFromCheckpoint(mod, loaded)
	if err != nil {
		t.Fatalf("restore compact trainer: %v", err)
	}
	after, err := restored.EvaluatePairs(batch)
	if err != nil {
		t.Fatalf("evaluate compact restored: %v", err)
	}
	assertClose(t, before.Loss, after.Loss, 0.000001)
	assertClose(t, before.AverageScore, after.AverageScore, 0.000001)
	if got := restored.TrainProfile().Step; got != checkpoint.Step+1 {
		t.Fatalf("restored compact step = %d, want %d", got, checkpoint.Step+1)
	}

	exportA, err := trainer.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export compact trainer: %v", err)
	}
	exportB, err := restored.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export restored compact trainer: %v", err)
	}
	if exportA["output_projection"].DType != "f16" {
		t.Fatalf("output_projection export dtype = %q, want f16", exportA["output_projection"].DType)
	}
	for _, name := range []string{"token_embedding", "layer0_attn_q", "layer1_ffn_down", "output_projection"} {
		assertTensorClose(t, exportA[name], exportB[name].Shape, exportB[name].F32)
	}
}

func TestCompactEmbeddingTrainerEvaluatePairsSkipsLegacyBatchedForwardWithAccelerator(t *testing.T) {
	t.Setenv("EOS_TRAIN_BATCHED_FORWARD", "1")
	t.Setenv("EOS_TRAIN_BATCHED_PAIR_EVAL", "1")
	t.Setenv("EOS_TRAIN_PAIR_EVAL_BATCH_SIZE", "8")
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	trainer.forwardBackend = eosartifact.BackendCUDA
	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{1, 2, 3}, RightTokens: []int32{1, 2, 3}, Target: 1},
		{LeftTokens: []int32{1, 2, 3}, RightTokens: []int32{3, 2, 1}, Target: -1},
	}
	metrics, err := trainer.EvaluatePairs(batch)
	if err != nil {
		t.Fatalf("evaluate compact pairs with fake accelerator: %v", err)
	}
	if !compactTestFinite(metrics.Loss) || !compactTestFinite(metrics.AverageScore) || metrics.PairCount != len(batch) {
		t.Fatalf("compact eval metrics = %+v, want finite pair metrics", metrics)
	}
	if fake.bindCalls == 0 {
		t.Fatalf("compact eval did not bind compact forward weights: %+v", fake)
	}
	if fake.boundRightRuns == 0 {
		t.Fatalf("compact eval did not use bound-right compact matmuls: %+v", fake)
	}
	if fake.multiBoundRuns == 0 {
		t.Fatalf("compact eval did not coalesce compact q/k/v matmuls: %+v", fake)
	}
	if fake.sharedLeftRuns != 0 || fake.accumulatedRuns != 0 {
		t.Fatalf("compact eval used non-forward compact accelerator path: %+v", fake)
	}
}

func TestCompactEmbeddingTrainerEncodeSequenceInputsSkipsLegacyBatchedForwardWithAccelerator(t *testing.T) {
	t.Setenv("EOS_TRAIN_BATCHED_FORWARD", "1")
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	trainer.forwardBackend = eosartifact.BackendCUDA
	forward := trainer.prepareForwardWeights()
	seqs, err := trainer.encodeSequenceInputs([]embeddingSequenceInput{
		{tokens: []int32{1, 2, 3}, role: trainer.queryRoleIndex(), label: "query"},
		{tokens: []int32{3, 2, 1}, role: trainer.documentRoleIndex(), label: "doc"},
	}, forward, false)
	if err != nil {
		t.Fatalf("encode compact sequence inputs with fake accelerator: %v", err)
	}
	defer trainer.releaseEncodedSequences(seqs)
	if len(seqs) != 2 || len(seqs[0].pooled) != 3 || len(seqs[1].pooled) != 3 {
		t.Fatalf("encoded compact sequences = %+v, want two pooled dim-3 sequences", seqs)
	}
	if fake.bindCalls == 0 {
		t.Fatalf("compact encodeSequenceInputs did not bind compact forward weights: %+v", fake)
	}
	if fake.boundRightRuns == 0 {
		t.Fatalf("compact encodeSequenceInputs did not use bound-right compact matmuls: %+v", fake)
	}
	if fake.multiBoundRuns == 0 {
		t.Fatalf("compact encodeSequenceInputs did not coalesce compact q/k/v matmuls: %+v", fake)
	}
	if fake.sharedLeftRuns != 0 || fake.accumulatedRuns != 0 {
		t.Fatalf("compact encodeSequenceInputs used non-forward compact accelerator path: %+v", fake)
	}
}

func TestCompactEmbeddingTrainerPackedForwardGateOffUsesHostPath(t *testing.T) {
	t.Setenv(compactPackedForwardEnv, "")
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	if trainer.compactForwardAccel != nil {
		t.Fatalf("compact packed accelerator instantiated with gate off")
	}
	start := trainer.TrainProfile()
	seqs, err := trainer.encodeSequenceInputs([]embeddingSequenceInput{
		{tokens: []int32{1, 2, 3}, role: trainer.rawRoleIndex()},
		{tokens: []int32{1, 2, 3}, role: trainer.rawRoleIndex()},
	}, trainer.prepareForwardWeights(), true)
	if err != nil {
		t.Fatalf("host compact encode inputs: %v", err)
	}
	defer trainer.releaseEncodedSequences(seqs)
	delta := diffTrainProfile(start, trainer.TrainProfile())
	if delta.CompactForward != nil {
		t.Fatalf("gate-off compact forward stats = %+v, want unavailable nil", delta.CompactForward)
	}
	if delta.CompactForwardTrainer.AttemptedCalls != 0 || delta.CompactForwardTrainer.BucketCount != 0 {
		t.Fatalf("gate-off trainer compact stats = %+v, want no selected packed attempts", delta.CompactForwardTrainer)
	}
}

func overrideTrainerConstructorsForTest(
	t *testing.T,
	matmul func() (backend.MatMulAccelerator, eosartifact.BackendKind, error),
	optimizer func() (backend.OptimizerAccelerator, eosartifact.BackendKind, error),
	activation func() (backend.ActivationAccelerator, eosartifact.BackendKind, trainerActivationAccelMode, error),
	compactForward func() (backend.CompactForwardAccelerator, eosartifact.BackendKind, error),
	compactTrain ...func() (backend.CompactTrainAccelerator, eosartifact.BackendKind, error),
) {
	t.Helper()
	prevMatMul := newTrainerMatMulAccelerator
	prevOptimizer := newTrainerOptimizerAccelerator
	prevActivation := newTrainerActivationAccelerator
	prevCompactForward := newTrainerCompactForwardAccelerator
	prevCompactTrain := newTrainerCompactTrainAccelerator
	if matmul != nil {
		newTrainerMatMulAccelerator = matmul
	}
	if optimizer != nil {
		newTrainerOptimizerAccelerator = optimizer
	}
	if activation != nil {
		newTrainerActivationAccelerator = activation
	}
	if compactForward != nil {
		newTrainerCompactForwardAccelerator = compactForward
	}
	if len(compactTrain) != 0 && compactTrain[0] != nil {
		newTrainerCompactTrainAccelerator = compactTrain[0]
	}
	t.Cleanup(func() {
		newTrainerMatMulAccelerator = prevMatMul
		newTrainerOptimizerAccelerator = prevOptimizer
		newTrainerActivationAccelerator = prevActivation
		newTrainerCompactForwardAccelerator = prevCompactForward
		newTrainerCompactTrainAccelerator = prevCompactTrain
	})
}

func TestEmbeddingTrainerCompactPackedForwardGateOnDoesNotInstantiateForNoncompact(t *testing.T) {
	t.Setenv(compactPackedForwardEnv, "1")
	compactConstructorCalls := 0
	overrideTrainerConstructorsForTest(t, nil, nil, nil,
		func() (backend.CompactForwardAccelerator, eosartifact.BackendKind, error) {
			compactConstructorCalls++
			return &fakeCompactForwardAccelerator{}, eosartifact.BackendCUDA, nil
		},
	)

	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	t.Cleanup(trainer.Close)
	if compactConstructorCalls != 0 {
		t.Fatalf("noncompact constructor called compact constructor %d times", compactConstructorCalls)
	}
	if trainer.compactForwardAccel != nil || trainer.compactForwardBackend != "" {
		t.Fatalf("noncompact compact forward accel/backend = %T/%q, want nil/empty", trainer.compactForwardAccel, trainer.compactForwardBackend)
	}
	profile := trainer.TrainProfile()
	if profile.CompactForwardBackend != "" || profile.CompactForward != nil {
		t.Fatalf("noncompact compact profile backend/stats = %q/%+v, want empty/nil", profile.CompactForwardBackend, profile.CompactForward)
	}
	if profile.CompactForwardTrainer != (embeddingCompactForwardTrainerStats{}) {
		t.Fatalf("noncompact compact trainer stats = %+v, want zero", profile.CompactForwardTrainer)
	}
}

func TestCompactEmbeddingTrainerPackedForwardGateOnNoRegisteredImplementationIsNoop(t *testing.T) {
	t.Setenv(compactPackedForwardEnv, "1")
	compactConstructorCalls := 0
	overrideTrainerConstructorsForTest(t, nil, nil, nil,
		func() (backend.CompactForwardAccelerator, eosartifact.BackendKind, error) {
			compactConstructorCalls++
			return nil, "", nil
		},
	)

	checkpoint := compactTrainStateTestCheckpoint(3)
	state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err != nil {
		t.Fatalf("load compact state: %v", err)
	}
	trainer, err := newCompactEmbeddingTrainerFromTrainState(&eosartifact.Module{Name: "compact"}, state)
	if err != nil {
		t.Fatalf("new compact trainer with no compact factory: %v", err)
	}
	t.Cleanup(trainer.Close)
	if compactConstructorCalls != 1 {
		t.Fatalf("compact constructor calls = %d, want 1", compactConstructorCalls)
	}
	if trainer.compactForwardAccel != nil || trainer.compactForwardBackend != "" {
		t.Fatalf("no registered compact implementation accel/backend = %T/%q, want nil/empty", trainer.compactForwardAccel, trainer.compactForwardBackend)
	}
	profile := trainer.TrainProfile()
	if profile.CompactForwardBackend != "" || profile.CompactForward != nil {
		t.Fatalf("no registered compact profile backend/stats = %q/%+v, want empty/nil", profile.CompactForwardBackend, profile.CompactForward)
	}
}

func TestCompactEmbeddingTrainerResidentTrainGateOffNoResourceOrStats(t *testing.T) {
	t.Setenv(compactResidentTrainEnv, "")

	checkpoint := compactTrainStateTestCheckpoint(3)
	state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err != nil {
		t.Fatalf("load compact state: %v", err)
	}
	trainer, err := newCompactEmbeddingTrainerFromTrainState(&eosartifact.Module{Name: "compact"}, state)
	if err != nil {
		t.Fatalf("new compact trainer: %v", err)
	}
	t.Cleanup(trainer.Close)
	if trainer.compactTrainAccel != nil || trainer.compactTrainBackend != "" {
		t.Fatalf("gate-off compact train accel/backend = %T/%q, want nil/empty", trainer.compactTrainAccel, trainer.compactTrainBackend)
	}
	profile := trainer.TrainProfile()
	if profile.CompactTrainBackend != "" || profile.CompactTrain != nil {
		t.Fatalf("gate-off compact train profile backend/stats = %q/%+v, want empty/nil", profile.CompactTrainBackend, profile.CompactTrain)
	}
}

func TestCompactEmbeddingTrainerResidentTrainGateOnNoRegisteredImplementationIsNoop(t *testing.T) {
	t.Setenv(compactResidentTrainEnv, "1")
	compactTrainConstructorCalls := 0
	overrideTrainerConstructorsForTest(t, nil, nil, nil, nil,
		func() (backend.CompactTrainAccelerator, eosartifact.BackendKind, error) {
			compactTrainConstructorCalls++
			return nil, "", nil
		},
	)

	checkpoint := compactTrainStateTestCheckpoint(3)
	state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err != nil {
		t.Fatalf("load compact state: %v", err)
	}
	trainer, err := newCompactEmbeddingTrainerFromTrainState(&eosartifact.Module{Name: "compact"}, state)
	if err != nil {
		t.Fatalf("new compact trainer: %v", err)
	}
	t.Cleanup(trainer.Close)
	if compactTrainConstructorCalls != 1 {
		t.Fatalf("compact train constructor calls = %d, want 1", compactTrainConstructorCalls)
	}
	if trainer.compactTrainAccel != nil || trainer.compactTrainBackend != "" {
		t.Fatalf("no compact train implementation accel/backend = %T/%q, want nil/empty", trainer.compactTrainAccel, trainer.compactTrainBackend)
	}
}

func TestCompactEmbeddingTrainerResidentTrainFactoryFailureFailsClosedAndCleansUp(t *testing.T) {
	t.Setenv(compactResidentTrainEnv, "1")
	matmul := &countingMatMulAccelerator{}
	optimizer := &fakeResidentOptimizerAccelerator{}
	activation := &countingActivationAccelerator{}
	compactForward := &fakeCompactForwardAccelerator{}
	overrideTrainerConstructorsForTest(t,
		func() (backend.MatMulAccelerator, eosartifact.BackendKind, error) {
			return matmul, eosartifact.BackendCUDA, nil
		},
		func() (backend.OptimizerAccelerator, eosartifact.BackendKind, error) {
			return optimizer, eosartifact.BackendCUDA, nil
		},
		func() (backend.ActivationAccelerator, eosartifact.BackendKind, trainerActivationAccelMode, error) {
			return activation, eosartifact.BackendCUDA, trainerActivationAccelMode{fullBackward: true}, nil
		},
		func() (backend.CompactForwardAccelerator, eosartifact.BackendKind, error) {
			return compactForward, eosartifact.BackendCUDA, nil
		},
		func() (backend.CompactTrainAccelerator, eosartifact.BackendKind, error) {
			return nil, "", fmt.Errorf("forced compact train factory failure")
		},
	)

	checkpoint := compactTrainStateTestCheckpoint(3)
	state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err != nil {
		t.Fatalf("load compact state: %v", err)
	}
	trainer, err := newCompactEmbeddingTrainerFromTrainState(&eosartifact.Module{Name: "compact"}, state)
	if err == nil {
		t.Cleanup(trainer.Close)
		t.Fatal("new compact trainer succeeded, want compact train factory failure")
	}
	if !strings.Contains(err.Error(), "forced compact train factory failure") {
		t.Fatalf("constructor error = %v, want forced compact train factory failure", err)
	}
	if !matmul.closed || !optimizer.closed || !activation.closed {
		t.Fatalf("constructor cleanup matmul/optimizer/activation closed = %t/%t/%t, want all true", matmul.closed, optimizer.closed, activation.closed)
	}
}

func TestCompactEmbeddingTrainerResidentTrainForwardValidationReleasesHandles(t *testing.T) {
	t.Setenv(compactResidentTrainEnv, "1")
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	host := newCompactEmbeddingTrainerForTest(t, 3)
	trainer.optimizerAccel = &fakeResidentOptimizerAccelerator{}
	trainer.compactForwardAccel = nil
	fake := &fakeCompactTrainAccelerator{}
	trainer.compactTrainAccel = fake
	trainer.compactTrainBackend = eosartifact.BackendCUDA
	forward := host.prepareForwardWeights()
	inputs := []embeddingSequenceInput{
		{tokens: []int32{1, 2, 3}, mask: []int32{1, 1, 1}, role: trainer.rawRoleIndex()},
		{tokens: []int32{3, 4}, mask: []int32{1, 0}, role: trainer.queryRoleIndex()},
		{tokens: []int32{2, 4}, mask: []int32{1, 1}, role: trainer.documentRoleIndex()},
	}
	if err := trainer.validateCompactResidentTrainForwardHandles(inputs, forward); err != nil {
		t.Fatalf("resident train forward validation: %v", err)
	}
	if fake.forwardCalls != 2 || fake.releaseCalls != 2 || fake.stats.LiveHandles != 0 {
		t.Fatalf("forward/release/live = %d/%d/%d, want 2/2/0", fake.forwardCalls, fake.releaseCalls, fake.stats.LiveHandles)
	}
	profile := trainer.TrainProfile()
	if profile.CompactTrain == nil || profile.CompactTrain.ForwardCalls != 2 || profile.CompactTrain.LiveHandles != 0 {
		t.Fatalf("compact train profile stats = %+v", profile.CompactTrain)
	}
	if trainer.compactForwardSelected {
		t.Fatal("forward-only validation marked normal compact forward selected")
	}
}

func TestCompactEmbeddingTrainerPackedForwardFactoryFailureFailsClosedAndCleansUp(t *testing.T) {
	t.Setenv(compactPackedForwardEnv, "1")
	t.Setenv("EOS_TRAIN_ENABLE_ACTIVATION_ACCEL", "1")
	t.Setenv("EOS_TRAIN_ENABLE_SOFTMAX_BACKWARD_ACCEL", "")
	matmul := &countingMatMulAccelerator{}
	optimizer := &fakeResidentOptimizerAccelerator{}
	activation := &countingActivationAccelerator{}
	overrideTrainerConstructorsForTest(t,
		func() (backend.MatMulAccelerator, eosartifact.BackendKind, error) {
			return matmul, eosartifact.BackendCUDA, nil
		},
		func() (backend.OptimizerAccelerator, eosartifact.BackendKind, error) {
			return optimizer, eosartifact.BackendCUDA, nil
		},
		func() (backend.ActivationAccelerator, eosartifact.BackendKind, trainerActivationAccelMode, error) {
			return activation, eosartifact.BackendCUDA, trainerActivationAccelMode{fullBackward: true}, nil
		},
		func() (backend.CompactForwardAccelerator, eosartifact.BackendKind, error) {
			return nil, "", fmt.Errorf("forced compact factory failure")
		},
	)

	checkpoint := compactTrainStateTestCheckpoint(3)
	state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err != nil {
		t.Fatalf("load compact state: %v", err)
	}
	trainer, err := newCompactEmbeddingTrainerFromTrainState(&eosartifact.Module{Name: "compact"}, state)
	if err == nil {
		t.Cleanup(trainer.Close)
		t.Fatal("new compact trainer succeeded, want compact factory failure")
	}
	if !strings.Contains(err.Error(), "forced compact factory failure") {
		t.Fatalf("constructor error = %v, want forced compact factory failure", err)
	}
	if !matmul.closed || !optimizer.closed || !activation.closed {
		t.Fatalf("constructor cleanup matmul/optimizer/activation closed = %t/%t/%t, want all true", matmul.closed, optimizer.closed, activation.closed)
	}
}

func TestCompactEmbeddingTrainerPackedForwardGroupsRestoresOrderAndStats(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	host := newCompactEmbeddingTrainerForTest(t, 3)
	opt := &fakeResidentOptimizerAccelerator{}
	fake := &fakeCompactForwardAccelerator{trainer: host, forward: host.prepareCompactForwardWeights()}
	trainer.optimizerAccel = opt
	trainer.compactForwardAccel = fake
	trainer.compactForwardBackend = eosartifact.BackendCUDA

	inputs := []embeddingSequenceInput{
		{tokens: []int32{1, 2, 3}, mask: []int32{1, 1, 1}, role: trainer.rawRoleIndex()},
		{tokens: []int32{3, 4}, mask: []int32{1, 0}, role: trainer.queryRoleIndex()},
		{tokens: []int32{1, 2, 3}, mask: []int32{2, 7, 1}, role: trainer.rawRoleIndex()},
		{tokens: []int32{2, 4}, mask: []int32{1, 1}, role: trainer.documentRoleIndex()},
	}
	seqs, err := trainer.encodeSequenceInputs(inputs, trainer.prepareForwardWeights(), true)
	if err != nil {
		t.Fatalf("packed compact encode inputs: %v", err)
	}
	defer trainer.releaseEncodedSequences(seqs)
	if len(seqs) != len(inputs) || seqs[0] != seqs[2] {
		t.Fatalf("dedupe/order restore failed: len=%d shared=%t", len(seqs), len(seqs) == len(inputs) && seqs[0] == seqs[2])
	}
	if seqs[1] == seqs[3] || seqs[1].role != trainer.queryRoleIndex() || seqs[3].role != trainer.documentRoleIndex() {
		t.Fatalf("role/order restore failed")
	}
	profile := trainer.TrainProfile()
	if profile.CompactForward == nil {
		t.Fatal("compact forward stats unavailable after fake packed path")
	}
	if profile.CompactForward.RunCalls != 2 || fake.preflightCalls != 2 {
		t.Fatalf("run/preflight calls = %d/%d, want two exact-length buckets", profile.CompactForward.RunCalls, fake.preflightCalls)
	}
	if profile.CompactForwardTrainer.AttemptedCalls != 1 || profile.CompactForwardTrainer.BucketCount != 2 || profile.CompactForwardTrainer.FallbackOrUnhandled != 0 || profile.CompactForwardTrainer.PreflightFailures != 0 {
		t.Fatalf("trainer compact stats = %+v", profile.CompactForwardTrainer)
	}
	if profile.CompactForwardTrainer.ResidentRefCount == 0 {
		t.Fatalf("resident refs = %d, want non-zero", profile.CompactForwardTrainer.ResidentRefCount)
	}
	if !trainer.compactForwardSelected {
		t.Fatal("packed compact forward did not mark selected path active")
	}
}

func TestCompactEmbeddingTrainerPackedForwardPreflightFailsClosed(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	host := newCompactEmbeddingTrainerForTest(t, 3)
	trainer.optimizerAccel = &fakeResidentOptimizerAccelerator{}
	trainer.compactForwardAccel = &fakeCompactForwardAccelerator{trainer: host, forward: host.prepareCompactForwardWeights(), preflightErr: fmt.Errorf("forced preflight")}
	_, err := trainer.encodeSequenceInputs([]embeddingSequenceInput{{tokens: []int32{1, 2, 3}, role: trainer.rawRoleIndex()}}, trainer.prepareForwardWeights(), true)
	if err == nil || !strings.Contains(err.Error(), "forced preflight") {
		t.Fatalf("packed preflight err = %v, want forced preflight", err)
	}
	if trainer.compactForwardTrainer.PreflightFailures != 1 || trainer.compactForwardTrainer.FallbackOrUnhandled != 0 {
		t.Fatalf("preflight stats = %+v", trainer.compactForwardTrainer)
	}
}

func TestCompactEmbeddingTrainerPackedForwardRunFailureKeepsSelectedFalse(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	host := newCompactEmbeddingTrainerForTest(t, 3)
	trainer.optimizerAccel = &fakeResidentOptimizerAccelerator{}
	trainer.compactForwardAccel = &fakeCompactForwardAccelerator{trainer: host, forward: host.prepareCompactForwardWeights(), runErr: fmt.Errorf("forced run failure")}
	trainer.compactForwardSelected = true
	_, err := trainer.encodeSequenceInputs([]embeddingSequenceInput{{tokens: []int32{1, 2, 3}, role: trainer.rawRoleIndex()}}, trainer.prepareForwardWeights(), true)
	if err == nil || !strings.Contains(err.Error(), "forced run failure") {
		t.Fatalf("packed run err = %v, want forced run failure", err)
	}
	if trainer.compactForwardSelected {
		t.Fatal("packed run failure left compactForwardSelected=true")
	}
	if !trainer.optimizerParamRequiresHostForwardRead(trainer.compactState.TokenEmbedding.Name) {
		t.Fatal("packed run failure allowed token resident deferral")
	}
	if !trainer.optimizerParamRequiresHostForwardRead(trainer.compactState.RoleEmbedding.Name) {
		t.Fatal("packed run failure allowed role resident deferral")
	}
	if trainer.compactForwardTrainer.FallbackOrUnhandled != 1 {
		t.Fatalf("fallback/unhandled = %d, want 1", trainer.compactForwardTrainer.FallbackOrUnhandled)
	}
}

func TestCompactEmbeddingTrainerPackedForwardReconstructFailureKeepsSelectedFalse(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	host := newCompactEmbeddingTrainerForTest(t, 3)
	trainer.optimizerAccel = &fakeResidentOptimizerAccelerator{}
	trainer.compactForwardAccel = &fakeCompactForwardAccelerator{trainer: host, forward: host.prepareCompactForwardWeights(), corruptResult: true}
	trainer.compactForwardSelected = true
	_, err := trainer.encodeSequenceInputs([]embeddingSequenceInput{{tokens: []int32{1, 2, 3}, role: trainer.rawRoleIndex()}}, trainer.prepareForwardWeights(), true)
	if err == nil || !strings.Contains(err.Error(), "reconstruct") {
		t.Fatalf("packed reconstruct err = %v, want reconstruct failure", err)
	}
	if trainer.compactForwardSelected {
		t.Fatal("packed reconstruct failure left compactForwardSelected=true")
	}
	if !trainer.optimizerParamRequiresHostForwardRead(trainer.compactState.TokenEmbedding.Name) {
		t.Fatal("packed reconstruct failure allowed token resident deferral")
	}
	if trainer.compactForwardTrainer.FallbackOrUnhandled != 1 {
		t.Fatalf("fallback/unhandled = %d, want 1", trainer.compactForwardTrainer.FallbackOrUnhandled)
	}
}

func TestCompactEmbeddingTrainerConstructorReportsHostOrDeviceAccelerators(t *testing.T) {
	checkpoint := compactTrainStateTestCheckpoint(3)
	state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err != nil {
		t.Fatalf("load compact state: %v", err)
	}
	trainer, err := newCompactEmbeddingTrainerFromTrainState(&eosartifact.Module{Name: "compact"}, state)
	if err != nil {
		t.Fatalf("new compact trainer: %v", err)
	}
	t.Cleanup(trainer.Close)
	profile := trainer.TrainProfile()
	if trainer.forwardMatMul == nil {
		if profile.ForwardBackend != eosartifact.BackendKind("host") {
			t.Fatalf("host-only compact forward backend = %q, want host", profile.ForwardBackend)
		}
	} else if profile.ForwardBackend != trainer.forwardMatMul.Backend() {
		t.Fatalf("compact forward backend = %q, want accelerator backend %q", profile.ForwardBackend, trainer.forwardMatMul.Backend())
	}
	if trainer.optimizerAccel == nil {
		if profile.OptimizerBackend != eosartifact.BackendKind("host") {
			t.Fatalf("host-only compact optimizer backend = %q, want host", profile.OptimizerBackend)
		}
	} else if profile.OptimizerBackend != trainer.optimizerAccel.Backend() {
		t.Fatalf("compact optimizer backend = %q, want accelerator backend %q", profile.OptimizerBackend, trainer.optimizerAccel.Backend())
	}
	if trainer.activationAccel == nil {
		if profile.ActivationBackend != eosartifact.BackendKind("host") {
			t.Fatalf("host-only compact activation backend = %q, want host", profile.ActivationBackend)
		}
	} else if profile.ActivationBackend != trainer.activationAccel.Backend() {
		t.Fatalf("compact activation backend = %q, want accelerator backend %q", profile.ActivationBackend, trainer.activationAccel.Backend())
	}
}

func TestCompactEmbeddingTrainerUsesMatMulAcceleratorForForwardAndBackward(t *testing.T) {
	host := newCompactEmbeddingTrainerForTest(t, 3)
	host.forwardMatMul = nil
	host.forwardBackend = eosartifact.BackendKind("host")
	accelerated := newCompactEmbeddingTrainerForTest(t, 3)
	fake := &countingMatMulAccelerator{}
	accelerated.forwardMatMul = fake
	accelerated.forwardBackend = eosartifact.BackendCUDA

	tokens := []int32{1, 2, 3}
	mask, err := host.prepareMask(tokens, nil)
	if err != nil {
		t.Fatalf("prepare host mask: %v", err)
	}
	hostSeq, err := host.encodeCompactSequence(tokens, mask, host.rawRoleIndex(), host.prepareCompactForwardWeights())
	if err != nil {
		t.Fatalf("host compact encode: %v", err)
	}
	accelMask, err := accelerated.prepareMask(tokens, nil)
	if err != nil {
		t.Fatalf("prepare accelerated mask: %v", err)
	}
	accelSeq, err := accelerated.encodeCompactSequence(tokens, accelMask, accelerated.rawRoleIndex(), accelerated.prepareCompactForwardWeights())
	if err != nil {
		t.Fatalf("accelerated compact encode: %v", err)
	}
	assertTensorClose(t, backend.NewTensorF32([]int{len(hostSeq.pooled)}, accelSeq.pooled), []int{len(hostSeq.pooled)}, hostSeq.pooled)
	forwardCalls := fake.runCalls
	if forwardCalls == 0 {
		t.Fatal("compact forward did not use trainer matmul accelerator helper")
	}

	grads := newCompactEmbeddingGradState(accelerated.compactState)
	gradPooled := make([]float32, len(accelSeq.pooled))
	for i := range gradPooled {
		gradPooled[i] = 0.01 * float32(i+1)
	}
	if err := accelerated.backpropCompactEncodedSequence(accelSeq, gradPooled, accelerated.prepareCompactForwardWeights(), grads); err != nil {
		t.Fatalf("accelerated compact backprop: %v", err)
	}
	if fake.runCalls <= forwardCalls {
		t.Fatalf("compact backward did not use trainer matmul accelerator helper: before=%d after=%d", forwardCalls, fake.runCalls)
	}
	if len(grads.outputProjection) != len(accelerated.compactState.OutputProjection.Tensor.F32) {
		t.Fatalf("output projection grad len = %d, want %d", len(grads.outputProjection), len(accelerated.compactState.OutputProjection.Tensor.F32))
	}
	for _, grad := range compactEmbeddingGradSlices(grads) {
		for _, value := range grad {
			if !compactTestFinite(value) {
				t.Fatalf("compact accelerated gradient contains non-finite value: %v", value)
			}
		}
	}
}

func TestCompactBackwardWeightTransposeReadsResidentWeights(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	name := "layer0_ffn_down"
	hostWeight := backend.NewTensorF32([]int{3, 4}, []float32{
		100, 100, 100, 100,
		100, 100, 100, 100,
		100, 100, 100, 100,
	})
	residentWeight := backend.NewTensorF32([]int{3, 4}, []float32{
		1, 2, 3, 4,
		-1, -2, -3, -4,
		0.5, 1.5, 2.5, 3.5,
	})
	opt := &fakeResidentOptimizerAccelerator{resident: map[string]*fakeResidentOptimizerToken{
		name: {tensor: residentWeight, generation: 1, alive: true},
	}}
	mm := &residentAwareCountingMatMulAccelerator{}
	trainer.optimizerAccel = opt
	trainer.forwardMatMul = mm
	trainer.deferOptimizerSync = true
	trainer.momentsDirty = true

	if err := trainer.bindForwardMatrix(name, hostWeight); err != nil {
		t.Fatalf("bind resident forward matrix: %v", err)
	}
	lhs := []float32{
		1, 0, 1, 0,
		0, 1, 0, 1,
	}
	out := make([]float32, 2*3)
	if err := trainer.fillWeightTransposeMatMul(lhs, 2, 4, name, hostWeight, "compact_test_host_fallback", out); err != nil {
		t.Fatalf("resident transpose matmul: %v", err)
	}
	want := make([]float32, len(out))
	fillHostMatMulTranspose(lhs, 2, 4, residentWeight.F32, 3, 4, false, true, want)
	assertCloseF32Slice(t, "resident compact transpose", out, want, 1e-6)
	if mm.residentBindCalls != 1 {
		t.Fatalf("resident bind calls = %d, want 1", mm.residentBindCalls)
	}
	if opt.stats.SyncCalls != 0 {
		t.Fatalf("host sync calls = %d, want 0 while resident binding is live", opt.stats.SyncCalls)
	}
}

func TestCompactDeferredOptimizerKeepsHostReadEmbeddingsCurrent(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	opt := &fakeResidentOptimizerAccelerator{}
	trainer.optimizerAccel = opt
	trainer.forwardMatMul = &residentAwareCountingMatMulAccelerator{}
	trainer.deferOptimizerSync = true

	if trainer.compactState.RoleEmbedding == nil {
		t.Fatal("compact fixture is missing role embedding")
	}
	grads := newCompactEmbeddingGradState(trainer.compactState)
	grads.token[0] = 1
	grads.role[0] = -1
	grads.layers[0].ffnDown[0] = 1
	tokenBefore := append([]float32(nil), trainer.compactState.TokenEmbedding.Tensor.F32...)
	roleBefore := append([]float32(nil), trainer.compactState.RoleEmbedding.Tensor.F32...)
	ffnDown := trainer.compactState.Layers[0].FFNDown
	ffnDownBefore := append([]float32(nil), ffnDown.Tensor.F32...)

	if err := trainer.applyCompactOptimizerUpdates(grads, 1); err != nil {
		t.Fatalf("compact optimizer update: %v", err)
	}

	if _, ok := opt.ResidentParameter(trainer.compactState.TokenEmbedding.Name); ok {
		t.Fatalf("token embedding was left resident-deferred; compact forward reads it from host")
	}
	if _, ok := opt.ResidentParameter(trainer.compactState.RoleEmbedding.Name); ok {
		t.Fatalf("role embedding was left resident-deferred; compact forward reads it from host")
	}
	if float32SlicesClose(trainer.compactState.TokenEmbedding.Tensor.F32, tokenBefore, 0) {
		t.Fatalf("token embedding host tensor did not receive immediate update")
	}
	if float32SlicesClose(trainer.compactState.RoleEmbedding.Tensor.F32, roleBefore, 0) {
		t.Fatalf("role embedding host tensor did not receive immediate update")
	}
	ref, ok := opt.ResidentParameter(ffnDown.Name)
	if !ok {
		t.Fatalf("matrix parameter %q was not kept resident-deferred", ffnDown.Name)
	}
	token, ok := ref.Token.(*fakeResidentOptimizerToken)
	if !ok || token.tensor == nil {
		t.Fatalf("matrix resident token = %#v, want fake token with tensor", ref.Token)
	}
	if !float32SlicesClose(ffnDown.Tensor.F32, ffnDownBefore, 0) {
		t.Fatalf("deferred matrix host tensor changed before explicit sync")
	}
	if float32SlicesClose(token.tensor.F32, ffnDownBefore, 0) {
		t.Fatalf("deferred matrix resident tensor did not receive update")
	}
}

func TestEmbeddingTrainerResidentOptimizerUpdateErrorFailsClosed(t *testing.T) {
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	param := backend.NewTensorF32([]int{2, 2}, []float32{1, 2, 3, 4})
	mom1 := backend.NewTensorF32([]int{2, 2}, []float32{0, 0, 0, 0})
	mom2 := backend.NewTensorF32([]int{2, 2}, []float32{0, 0, 0, 0})
	before := append([]float32(nil), param.F32...)
	name := "projection"
	trainer.optimizerAccel = &fakeResidentOptimizerAccelerator{
		applyErr: fmt.Errorf("forced resident update failure"),
		resident: map[string]*fakeResidentOptimizerToken{
			name: {tensor: param.Clone(), generation: 1, alive: true},
		},
	}
	trainer.forwardMatMul = &residentAwareCountingMatMulAccelerator{}
	trainer.deferOptimizerSync = true
	err := trainer.applyOptimizerUpdate(name, param, mom1, mom2, []float32{0.1, 0.2, 0.3, 0.4}, 1)
	if err == nil || !strings.Contains(err.Error(), "forced resident update failure") {
		t.Fatalf("apply optimizer update error = %v, want forced resident failure", err)
	}
	assertCloseF32Slice(t, "host tensor after failed resident update", param.F32, before, 0)
}

func TestCompactResidentBindAndHostSyncFailureFailsClosed(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	name := "layer0_ffn_down"
	hostWeight := backend.NewTensorF32([]int{3, 4}, []float32{
		100, 100, 100, 100,
		100, 100, 100, 100,
		100, 100, 100, 100,
	})
	residentWeight := backend.NewTensorF32([]int{3, 4}, []float32{
		1, 2, 3, 4,
		-1, -2, -3, -4,
		0.5, 1.5, 2.5, 3.5,
	})
	syncErr := fmt.Errorf("forced fallback sync failure")
	opt := &fakeResidentOptimizerAccelerator{
		syncErr: syncErr,
		resident: map[string]*fakeResidentOptimizerToken{
			name: {tensor: residentWeight, generation: 1, alive: true},
		},
	}
	mm := &failingResidentBindMatMulAccelerator{bindErr: fmt.Errorf("forced resident bind failure")}
	trainer.optimizerAccel = opt
	trainer.forwardMatMul = mm
	trainer.deferOptimizerSync = true
	trainer.momentsDirty = true

	err := trainer.bindForwardMatrix(name, hostWeight)
	if err == nil || !strings.Contains(err.Error(), "forced fallback sync failure") {
		t.Fatalf("bindForwardMatrix error = %v, want fallback sync failure", err)
	}
	if mm.bindCalls != 0 {
		t.Fatalf("host BindMatrix calls = %d, want 0 after failed resident bind plus failed sync", mm.bindCalls)
	}

	out := make([]float32, 2*3)
	err = trainer.fillWeightTransposeMatMul([]float32{1, 0, 1, 0, 0, 1, 0, 1}, 2, 4, name, hostWeight, "compact_test_host_fallback", out)
	if err == nil || !strings.Contains(err.Error(), "forced fallback sync failure") {
		t.Fatalf("fallback matmul error = %v, want sync failure", err)
	}
}

func TestEmbeddingTrainerCloseWithErrorPreservesResidentStateOnSyncFailure(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	name := trainer.compactState.TokenEmbedding.Name
	token := &fakeResidentOptimizerToken{
		tensor:     trainer.compactState.TokenEmbedding.Tensor.Clone(),
		generation: 1,
		alive:      true,
	}
	opt := &fakeResidentOptimizerAccelerator{
		syncErr: fmt.Errorf("forced close sync failure"),
		resident: map[string]*fakeResidentOptimizerToken{
			name: token,
		},
	}
	trainer.optimizerAccel = opt
	trainer.momentsDirty = true

	err := trainer.CloseWithError()
	if err == nil || !strings.Contains(err.Error(), "forced close sync failure") {
		t.Fatalf("CloseWithError error = %v, want close sync failure", err)
	}
	if trainer.CloseError() != err {
		t.Fatalf("CloseError = %v, want same error %v", trainer.CloseError(), err)
	}
	if trainer.optimizerAccel == nil {
		t.Fatal("optimizer accelerator was cleared after failed close sync")
	}
	if opt.closed {
		t.Fatal("optimizer accelerator was closed after failed close sync")
	}
	if !token.Alive() {
		t.Fatal("resident token was invalidated after failed close sync")
	}

	opt.syncErr = nil
	if err := trainer.CloseWithError(); err != nil {
		t.Fatalf("second CloseWithError after clearing sync error: %v", err)
	}
	if trainer.CloseError() != nil {
		t.Fatalf("CloseError after successful close = %v, want nil", trainer.CloseError())
	}
	if !opt.closed {
		t.Fatal("optimizer accelerator was not closed after successful retry")
	}
	if token.Alive() {
		t.Fatal("resident token is still alive after successful close")
	}
}

func TestCompactEmbeddingTrainerUsesActivationAcceleratorForBackward(t *testing.T) {
	host := newCompactEmbeddingTrainerForTest(t, 3)
	host.forwardMatMul = nil
	host.forwardBackend = eosartifact.BackendKind("host")
	accelerated := newCompactEmbeddingTrainerForTest(t, 3)
	accelerated.forwardMatMul = nil
	accelerated.forwardBackend = eosartifact.BackendKind("host")
	activation := &countingActivationAccelerator{}
	accelerated.activationAccel = activation
	accelerated.activationBackend = eosartifact.BackendCUDA
	accelerated.activationAccelFull = true
	accelerated.softmaxBackwardAccel = true

	tokens := []int32{1, 2, 3}
	hostSeq := compactEncodeForBackwardTest(t, host, tokens)
	accelSeq := compactEncodeForBackwardTest(t, accelerated, tokens)
	gradPooled := compactPooledGradForTest(accelSeq)
	hostGrads := newCompactEmbeddingGradState(host.compactState)
	if err := host.backpropCompactEncodedSequence(hostSeq, gradPooled, host.prepareCompactForwardWeights(), hostGrads); err != nil {
		t.Fatalf("host compact backprop: %v", err)
	}
	accelGrads := newCompactEmbeddingGradState(accelerated.compactState)
	if err := accelerated.backpropCompactEncodedSequence(accelSeq, gradPooled, accelerated.prepareCompactForwardWeights(), accelGrads); err != nil {
		t.Fatalf("accelerated compact backprop: %v", err)
	}

	compactAssertGradSlicesClose(t, accelGrads, hostGrads)
	wantLayerNorm, wantGELU, wantSoftmax := compactActivationBackwardCallCounts(accelSeq)
	if activation.layerNormBackwardCalls != wantLayerNorm {
		t.Fatalf("compact layernorm backward calls = %d, want %d", activation.layerNormBackwardCalls, wantLayerNorm)
	}
	if activation.geluBackwardCalls != wantGELU {
		t.Fatalf("compact gelu backward calls = %d, want %d", activation.geluBackwardCalls, wantGELU)
	}
	if activation.softmaxBackwardCalls != wantSoftmax {
		t.Fatalf("compact softmax backward calls = %d, want %d", activation.softmaxBackwardCalls, wantSoftmax)
	}
	profile := accelerated.TrainProfile()
	if profile.Activation.LayerNormBackwardCalls != int64(wantLayerNorm) ||
		profile.Activation.GELUBackwardCalls != int64(wantGELU) ||
		profile.Activation.SoftmaxBackwardCalls != int64(wantSoftmax) {
		t.Fatalf("compact activation profile = %+v, want layernorm=%d gelu=%d softmax=%d", profile.Activation, wantLayerNorm, wantGELU, wantSoftmax)
	}
}

func TestCompactEmbeddingTrainerSkipsGELUActivationAcceleratorForFastGELU(t *testing.T) {
	t.Setenv("EOS_TRAIN_ENABLE_FAST_GELU", "1")
	host := newCompactEmbeddingTrainerForTest(t, 3)
	host.forwardMatMul = nil
	host.forwardBackend = eosartifact.BackendKind("host")
	accelerated := newCompactEmbeddingTrainerForTest(t, 3)
	accelerated.forwardMatMul = nil
	accelerated.forwardBackend = eosartifact.BackendKind("host")
	activation := &countingActivationAccelerator{}
	accelerated.activationAccel = activation
	accelerated.activationBackend = eosartifact.BackendCUDA
	accelerated.activationAccelFull = true
	accelerated.softmaxBackwardAccel = true

	tokens := []int32{1, 2, 3}
	hostSeq := compactEncodeForBackwardTest(t, host, tokens)
	accelSeq := compactEncodeForBackwardTest(t, accelerated, tokens)
	gradPooled := compactPooledGradForTest(accelSeq)
	hostGrads := newCompactEmbeddingGradState(host.compactState)
	if err := host.backpropCompactEncodedSequence(hostSeq, gradPooled, host.prepareCompactForwardWeights(), hostGrads); err != nil {
		t.Fatalf("host compact fast-gelu backprop: %v", err)
	}
	accelGrads := newCompactEmbeddingGradState(accelerated.compactState)
	if err := accelerated.backpropCompactEncodedSequence(accelSeq, gradPooled, accelerated.prepareCompactForwardWeights(), accelGrads); err != nil {
		t.Fatalf("accelerated compact fast-gelu backprop: %v", err)
	}

	compactAssertGradSlicesClose(t, accelGrads, hostGrads)
	wantLayerNorm, _, wantSoftmax := compactActivationBackwardCallCounts(accelSeq)
	if activation.geluBackwardCalls != 0 {
		t.Fatalf("compact fast-gelu gelu backward calls = %d, want 0", activation.geluBackwardCalls)
	}
	if activation.layerNormBackwardCalls != wantLayerNorm {
		t.Fatalf("compact fast-gelu layernorm backward calls = %d, want %d", activation.layerNormBackwardCalls, wantLayerNorm)
	}
	if activation.softmaxBackwardCalls != wantSoftmax {
		t.Fatalf("compact fast-gelu softmax backward calls = %d, want %d", activation.softmaxBackwardCalls, wantSoftmax)
	}
}

func TestCompactEmbeddingTrainerActivationAcceleratorHonorsElementLimit(t *testing.T) {
	t.Setenv("EOS_TRAIN_ACTIVATION_ACCEL_MAX_ELEMENTS", "1")
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	trainer.forwardMatMul = nil
	trainer.forwardBackend = eosartifact.BackendKind("host")
	activation := &countingActivationAccelerator{}
	trainer.activationAccel = activation
	trainer.activationBackend = eosartifact.BackendCUDA
	trainer.activationAccelFull = true
	trainer.softmaxBackwardAccel = true

	seq := compactEncodeForBackwardTest(t, trainer, []int32{1, 2, 3})
	grads := newCompactEmbeddingGradState(trainer.compactState)
	if err := trainer.backpropCompactEncodedSequence(seq, compactPooledGradForTest(seq), trainer.prepareCompactForwardWeights(), grads); err != nil {
		t.Fatalf("compact limited activation backprop: %v", err)
	}
	if activation.layerNormBackwardCalls != 0 || activation.geluBackwardCalls != 0 || activation.softmaxBackwardCalls != 0 {
		t.Fatalf("compact activation calls with element limit = layernorm %d gelu %d softmax %d, want all zero",
			activation.layerNormBackwardCalls, activation.geluBackwardCalls, activation.softmaxBackwardCalls)
	}
}

func compactEncodeForBackwardTest(t *testing.T, trainer *EmbeddingTrainer, tokens []int32) *embeddingEncodedSequence {
	t.Helper()
	mask, err := trainer.prepareMask(tokens, nil)
	if err != nil {
		t.Fatalf("prepare compact mask: %v", err)
	}
	seq, err := trainer.encodeCompactSequence(tokens, mask, trainer.rawRoleIndex(), trainer.prepareCompactForwardWeights())
	if err != nil {
		t.Fatalf("encode compact sequence: %v", err)
	}
	return seq
}

func compactPooledGradForTest(seq *embeddingEncodedSequence) []float32 {
	gradPooled := make([]float32, len(seq.pooled))
	for i := range gradPooled {
		gradPooled[i] = 0.01 * float32(i+1)
	}
	return gradPooled
}

func compactActivationBackwardCallCounts(seq *embeddingEncodedSequence) (layerNorm, gelu, softmax int) {
	for _, layer := range seq.layers {
		if len(layer.tokens) == 0 {
			continue
		}
		layerNorm += 2
		gelu++
		softmax += len(layer.attnScores) / (len(layer.tokens) * len(layer.tokens))
	}
	return layerNorm, gelu, softmax
}

func compactAssertGradSlicesClose(t *testing.T, got, want *compactEmbeddingGradState) {
	t.Helper()
	gotSlices := compactEmbeddingGradSlices(got)
	wantSlices := compactEmbeddingGradSlices(want)
	if len(gotSlices) != len(wantSlices) {
		t.Fatalf("compact grad slice count = %d, want %d", len(gotSlices), len(wantSlices))
	}
	for i := range gotSlices {
		if len(gotSlices[i]) != len(wantSlices[i]) {
			t.Fatalf("compact grad slice %d len = %d, want %d", i, len(gotSlices[i]), len(wantSlices[i]))
		}
		assertTensorClose(t, backend.NewTensorF32([]int{len(gotSlices[i])}, gotSlices[i]), []int{len(wantSlices[i])}, wantSlices[i])
	}
}

func TestCompactEmbeddingTrainerFitEvalOnlyDoesNotTrain(t *testing.T) {
	checkpoint := compactTrainStateTestCheckpoint(3)
	state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err != nil {
		t.Fatalf("load compact state: %v", err)
	}
	trainer, err := newCompactEmbeddingTrainerFromTrainState(&eosartifact.Module{Name: "compact"}, state)
	if err != nil {
		t.Fatalf("new compact trainer: %v", err)
	}
	evalSet := []EmbeddingPairExample{
		{LeftTokens: []int32{1, 2}, RightTokens: []int32{1, 2}, Target: 1},
		{LeftTokens: []int32{1, 2}, RightTokens: []int32{2, 1}, Target: -1},
	}
	summary, err := trainer.Fit(nil, evalSet, EmbeddingTrainRunConfig{EvalOnly: true})
	if err != nil {
		t.Fatalf("compact eval-only fit: %v", err)
	}
	if summary.StepsRun != 0 || summary.StepsCompleted != checkpoint.Step || trainer.TrainProfile().Step != checkpoint.Step {
		t.Fatalf("compact eval-only steps summary=%d/%d trainer=%d want unchanged %d", summary.StepsRun, summary.StepsCompleted, trainer.TrainProfile().Step, checkpoint.Step)
	}
	if summary.DeltaProfile.Optimizer.UpdateCalls != 0 || trainer.TrainProfile().Optimizer.UpdateCalls != 0 {
		t.Fatalf("compact eval-only optimizer updates summary=%d trainer=%d, want zero", summary.DeltaProfile.Optimizer.UpdateCalls, trainer.TrainProfile().Optimizer.UpdateCalls)
	}
	if summary.FinalEval == nil || !compactTestFinite(summary.FinalEval.Loss) {
		t.Fatalf("compact eval-only final eval = %+v, want finite metrics", summary.FinalEval)
	}
	beforeTrainStep := trainer.TrainProfile().Step
	beforeTrainUpdates := trainer.TrainProfile().Optimizer.UpdateCalls
	snapshots := snapshotCompactTrainStateTensors(trainer.compactState)
	trainSummary, err := trainer.Fit(evalSet, nil, EmbeddingTrainRunConfig{Epochs: 1, BatchSize: len(evalSet), Shuffle: false})
	if err != nil {
		t.Fatalf("compact pairwise Fit: %v", err)
	}
	if trainSummary.StepsRun != 1 || trainer.TrainProfile().Step != beforeTrainStep+1 {
		t.Fatalf("compact Fit steps summary=%d trainer=%d want one update from %d", trainSummary.StepsRun, trainer.TrainProfile().Step, beforeTrainStep)
	}
	if trainSummary.DeltaProfile.Optimizer.UpdateCalls != 1 || trainer.TrainProfile().Optimizer.UpdateCalls != beforeTrainUpdates+1 {
		t.Fatalf("compact Fit optimizer updates summary=%d trainer=%d want one update from %d", trainSummary.DeltaProfile.Optimizer.UpdateCalls, trainer.TrainProfile().Optimizer.UpdateCalls, beforeTrainUpdates)
	}
	if !compactTestFinite(trainSummary.FinalTrain.Loss) {
		t.Fatalf("compact Fit final train = %+v, want finite metrics", trainSummary.FinalTrain)
	}
	delta := aggregateParameterDeltaStats(snapshots)
	if delta.NonzeroCount == 0 || delta.L2Norm == 0 {
		t.Fatalf("compact Fit did not move train tensors: %+v", delta)
	}
}

func TestCompactEmbeddingTrainerOutputProjectionDimensions(t *testing.T) {
	for _, tt := range []struct {
		name      string
		outputDim int
	}{
		{name: "with_projection", outputDim: 3},
		{name: "without_projection", outputDim: 4},
	} {
		t.Run(tt.name, func(t *testing.T) {
			checkpoint := compactTrainStateTestCheckpoint(tt.outputDim)
			state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
			if err != nil {
				t.Fatalf("load compact state: %v", err)
			}
			trainer, err := newCompactEmbeddingTrainerFromTrainState(&eosartifact.Module{Name: "compact"}, state)
			if err != nil {
				t.Fatalf("new compact trainer: %v", err)
			}
			mask, err := trainer.prepareMask([]int32{1, 2, 3}, nil)
			if err != nil {
				t.Fatalf("prepare mask: %v", err)
			}
			encoded, err := trainer.encodeSequence([]int32{1, 2, 3}, mask, trainer.rawRoleIndex(), nil, nil, nil, nil, nil, nil, nil, nil, false)
			if err != nil {
				t.Fatalf("encode compact sequence: %v", err)
			}
			if len(encoded.pooled) != tt.outputDim {
				t.Fatalf("pooled dim = %d, want %d", len(encoded.pooled), tt.outputDim)
			}
		})
	}
}

func TestCompactEmbeddingTrainerMaskAndTokenOrderSensitivity(t *testing.T) {
	checkpoint := compactTrainStateTestCheckpoint(4)
	state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
	if err != nil {
		t.Fatalf("load compact state: %v", err)
	}
	trainer, err := newCompactEmbeddingTrainerFromTrainState(&eosartifact.Module{Name: "compact"}, state)
	if err != nil {
		t.Fatalf("new compact trainer: %v", err)
	}
	pooled := func(tokens, mask []int32) []float32 {
		prepared, err := trainer.prepareMask(tokens, mask)
		if err != nil {
			t.Fatalf("prepare mask: %v", err)
		}
		encoded, err := trainer.encodeSequence(tokens, prepared, trainer.rawRoleIndex(), nil, nil, nil, nil, nil, nil, nil, nil, false)
		if err != nil {
			t.Fatalf("encode compact sequence: %v", err)
		}
		return encoded.pooled
	}
	maskedA := pooled([]int32{1, 2, 3}, []int32{1, 1, 0})
	maskedB := pooled([]int32{1, 2, 4}, []int32{1, 1, 0})
	if !float32SlicesClose(maskedA, maskedB, 1e-6) {
		t.Fatalf("masked padding changed compact pooled embedding: %v vs %v", maskedA, maskedB)
	}
	ordered := pooled([]int32{1, 2, 3}, []int32{1, 1, 1})
	reordered := pooled([]int32{2, 1, 3}, []int32{1, 1, 1})
	if float32SlicesClose(ordered, reordered, 1e-6) {
		t.Fatalf("compact pooled embedding is insensitive to token order: %v vs %v", ordered, reordered)
	}
}

func TestCompactEmbeddingTrainerMultiHeadAttentionDiffersFromFusedHeadAndGradientsFinite(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 4)
	if trainer.manifest.AttentionHeads != 2 || trainer.manifest.HeadDim != 2 {
		t.Fatalf("compact fixture attention layout = heads %d head_dim %d, want 2/2", trainer.manifest.AttentionHeads, trainer.manifest.HeadDim)
	}
	forward := trainer.prepareCompactForwardWeights()
	if forward == nil || len(forward.layers) == 0 {
		t.Fatal("missing compact forward layer")
	}
	tokens := []int32{1, 2, 3}
	mask, err := trainer.prepareMask(tokens, nil)
	if err != nil {
		t.Fatalf("prepare mask: %v", err)
	}
	input, err := embeddingInputForTokens(forward.token, tokens)
	if err != nil {
		t.Fatalf("embedding input: %v", err)
	}
	if err := addRoleEmbeddingToInput(input, forward.role, trainer.rawRoleIndex(), len(tokens)); err != nil {
		t.Fatalf("role input: %v", err)
	}
	if err := applyEmbeddingPositionEncoding(input, len(tokens), forward.token.Shape[1], trainer.manifest.PositionEncoding); err != nil {
		t.Fatalf("position encoding: %v", err)
	}
	multi, err := trainer.encodeCompactLayer(tokens, mask, append([]float32(nil), input...), forward.layers[0])
	if err != nil {
		t.Fatalf("encode multi-head compact layer: %v", err)
	}
	fusedLayer := forward.layers[0]
	fusedLayer.attentionHeads = 1
	fusedLayer.headDim = trainer.manifest.ModelDim
	fused, err := trainer.encodeCompactLayer(tokens, mask, append([]float32(nil), input...), fusedLayer)
	if err != nil {
		t.Fatalf("encode fused-head compact layer: %v", err)
	}
	if len(multi.attnScores) != trainer.manifest.AttentionHeads*len(tokens)*len(tokens) {
		t.Fatalf("multi-head score len = %d, want heads*T*T", len(multi.attnScores))
	}
	if float32SlicesClose(multi.attnMixed, fused.attnMixed, 1e-6) {
		t.Fatalf("two-head attention mixed output matched fused-head output: %v", multi.attnMixed)
	}

	batch := compactFiniteDiffPairBatch()
	metrics, grads := compactPairBatchAnalyticGradForTest(t, trainer, batch)
	if !compactTestFinite(metrics.Loss) || metrics.BatchSize != len(batch) {
		t.Fatalf("compact multi-head analytic metrics = %+v, want finite batch metrics", metrics)
	}
	gradStats := aggregateScaledGradientStats(1, compactEmbeddingGradSlices(grads)...)
	if !compactTestFinite(gradStats.L2Norm) || gradStats.L2Norm <= 0 || gradStats.NonzeroCount <= 0 {
		t.Fatalf("compact multi-head gradient stats = %+v, want finite non-zero gradients", gradStats)
	}
	for sliceIndex, grad := range compactEmbeddingGradSlices(grads) {
		for valueIndex, value := range grad {
			if !compactTestFinite(value) {
				t.Fatalf("compact multi-head grad[%d][%d] = %v, want finite", sliceIndex, valueIndex, value)
			}
		}
	}
	snapshots := snapshotCompactTrainStateTensors(trainer.compactState)
	trainMetrics, err := trainer.TrainStep(batch)
	if err != nil {
		t.Fatalf("compact multi-head TrainStep: %v", err)
	}
	if !compactTestFinite(trainMetrics.Loss) || !compactTestFinite(trainMetrics.AverageScore) {
		t.Fatalf("compact multi-head train metrics = %+v, want finite", trainMetrics)
	}
	delta := aggregateParameterDeltaStats(snapshots)
	if !compactTestFinite(delta.L2Norm) || delta.L2Norm <= 0 || delta.NonzeroCount <= 0 {
		t.Fatalf("compact multi-head parameter delta = %+v, want finite non-zero update", delta)
	}
}

func TestCompactEmbeddingTrainerFiniteDifferenceOutputProjectionAndAttentionGradients(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	batch := compactFiniteDiffPairBatch()
	metrics, grads := compactPairBatchAnalyticGradForTest(t, trainer, batch)
	if !compactTestFinite(metrics.Loss) || metrics.BatchSize != len(batch) {
		t.Fatalf("compact analytic metrics = %+v, want finite batch metrics", metrics)
	}
	if grads.outputProjection == nil {
		t.Fatal("missing output_projection analytic gradients")
	}
	outputIndex := compactFiniteDiffMaxAbsIndex(grads.outputProjection, 1e-6)
	if outputIndex < 0 {
		t.Fatalf("output_projection gradients are all near zero: %v", grads.outputProjection)
	}
	compactAssertFiniteDiffGrad(t, "output_projection", grads.outputProjection[outputIndex], func(delta float32) {
		trainer.compactState.OutputProjection.Tensor.F32[outputIndex] += delta
	}, func() float32 {
		return compactPairBatchLossForTest(t, trainer, batch)
	})

	attentionGrad := grads.layers[0].attnK
	attentionIndex := compactFiniteDiffMaxAbsIndex(attentionGrad, 1e-6)
	if attentionIndex < 0 {
		t.Fatalf("layer0_attn_k gradients are all near zero: %v", attentionGrad)
	}
	compactAssertFiniteDiffGrad(t, "layer0_attn_k", attentionGrad[attentionIndex], func(delta float32) {
		trainer.compactState.Layers[0].AttentionKey.Tensor.F32[attentionIndex] += delta
	}, func() float32 {
		return compactPairBatchLossForTest(t, trainer, batch)
	})
}

func TestCompactEmbeddingTrainerFiniteDifferenceTokenEmbeddingRoPEAndMaskGradients(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	if trainer.manifest.PositionEncoding != EmbeddingPositionEncodingRoPE {
		t.Fatalf("compact test fixture position encoding = %q, want RoPE", trainer.manifest.PositionEncoding)
	}
	batch := compactFiniteDiffMaskedPairBatch()
	_, grads := compactPairBatchAnalyticGradForTest(t, trainer, batch)
	d := trainer.compactState.TokenEmbedding.Tensor.Shape[1]
	activeBase := 1 * d
	activeIndex := activeBase + compactFiniteDiffMaxAbsIndex(grads.token[activeBase:activeBase+d], 1e-6)
	if activeIndex < activeBase {
		t.Fatalf("active token row gradients are all near zero: %v", grads.token[activeBase:activeBase+d])
	}
	compactAssertFiniteDiffGrad(t, "token_embedding_rope_active", grads.token[activeIndex], func(delta float32) {
		trainer.compactState.TokenEmbedding.Tensor.F32[activeIndex] += delta
	}, func() float32 {
		return compactPairBatchLossForTest(t, trainer, batch)
	})

	maskedBase := 4 * d
	for col, got := range grads.token[maskedBase : maskedBase+d] {
		if abs := float32(math.Abs(float64(got))); abs > 1e-6 {
			t.Fatalf("masked padding token grad[%d] = %.9g, want zero", col, got)
		}
	}
	maskedIndex := maskedBase + 1
	compactAssertFiniteDiffGrad(t, "token_embedding_masked_padding", grads.token[maskedIndex], func(delta float32) {
		trainer.compactState.TokenEmbedding.Tensor.F32[maskedIndex] += delta
	}, func() float32 {
		return compactPairBatchLossForTest(t, trainer, batch)
	})
}

func compactFiniteDiffPairBatch() []EmbeddingPairExample {
	return []EmbeddingPairExample{
		{LeftTokens: []int32{1, 2, 3}, RightTokens: []int32{3, 1, 2}, Target: 0.35},
	}
}

func compactFiniteDiffMaskedPairBatch() []EmbeddingPairExample {
	return []EmbeddingPairExample{
		{
			LeftTokens:  []int32{1, 2, 4},
			LeftMask:    []int32{1, 1, 0},
			RightTokens: []int32{2, 3, 4},
			RightMask:   []int32{1, 1, 0},
			Target:      0.2,
		},
	}
}

func compactPairBatchAnalyticGradForTest(t *testing.T, trainer *EmbeddingTrainer, batch []EmbeddingPairExample) (EmbeddingTrainMetrics, *compactEmbeddingGradState) {
	t.Helper()
	if trainer == nil || trainer.compactState == nil {
		t.Fatal("compact trainer is not initialized")
	}
	trainer.invalidateForwardWeights()
	forward := trainer.prepareForwardWeights()
	if forward == nil || forward.compact == nil {
		t.Fatal("missing compact forward weights")
	}
	grads := newCompactEmbeddingGradState(trainer.compactState)
	totalLoss := float32(0)
	totalScore := float32(0)
	for i, example := range batch {
		left, right, err := trainer.encodeExamplePair(example, nil, nil, nil, nil, nil, nil, nil, nil, false)
		if err != nil {
			t.Fatalf("encode compact finite-diff pair %d: %v", i, err)
		}
		score, gradLeft, gradRight := cosineGrad(left.pooled, right.pooled)
		scale := score - example.Target
		totalLoss += 0.5 * scale * scale
		totalScore += score
		for j := range gradLeft {
			gradLeft[j] *= scale
			gradRight[j] *= scale
		}
		if err := trainer.backpropCompactEncodedSequence(left, gradLeft, forward.compact, grads); err != nil {
			t.Fatalf("backprop compact finite-diff pair %d left: %v", i, err)
		}
		if err := trainer.backpropCompactEncodedSequence(right, gradRight, forward.compact, grads); err != nil {
			t.Fatalf("backprop compact finite-diff pair %d right: %v", i, err)
		}
	}
	batchScale := float32(1) / float32(len(batch))
	compactScaleGradState(grads, batchScale)
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * batchScale,
		AverageScore: totalScore * batchScale,
		BatchSize:    len(batch),
	}, grads
}

func compactPairBatchLossForTest(t *testing.T, trainer *EmbeddingTrainer, batch []EmbeddingPairExample) float32 {
	t.Helper()
	trainer.invalidateForwardWeights()
	metrics, err := trainer.runBatch(batch, false)
	if err != nil {
		t.Fatalf("compact finite-diff loss eval: %v", err)
	}
	if !compactTestFinite(metrics.Loss) {
		t.Fatalf("compact finite-diff loss = %v, want finite", metrics.Loss)
	}
	return metrics.Loss
}

func compactAssertFiniteDiffGrad(t *testing.T, name string, analytic float32, mutate func(float32), loss func() float32) {
	t.Helper()
	const eps = float32(1e-3)
	base := loss()
	mutate(eps)
	plus := loss()
	mutate(-2 * eps)
	minus := loss()
	mutate(eps)
	restored := loss()
	if diff := float32(math.Abs(float64(restored - base))); diff > 2e-6 {
		t.Fatalf("%s parameter restore changed loss by %.9g (base %.9g restored %.9g)", name, diff, base, restored)
	}
	numeric := (plus - minus) / (2 * eps)
	if !compactGradClose(analytic, numeric, 2e-3, 0.12) {
		t.Fatalf("%s analytic grad %.9g finite-diff %.9g (plus %.9g minus %.9g base %.9g)", name, analytic, numeric, plus, minus, base)
	}
}

func compactGradClose(a, b, absTol, relTol float32) bool {
	diff := float32(math.Abs(float64(a - b)))
	if diff <= absTol {
		return true
	}
	scale := float32(math.Max(math.Abs(float64(a)), math.Abs(float64(b))))
	if scale == 0 {
		return diff <= absTol
	}
	return diff/scale <= relTol
}

func compactFiniteDiffMaxAbsIndex(values []float32, minAbs float32) int {
	bestIndex := -1
	bestAbs := minAbs
	for i, value := range values {
		abs := float32(math.Abs(float64(value)))
		if abs > bestAbs {
			bestAbs = abs
			bestIndex = i
		}
	}
	return bestIndex
}

func compactScaleGradState(grads *compactEmbeddingGradState, scale float32) {
	if grads == nil {
		return
	}
	scaleFloat32Slice(grads.token, scale)
	scaleFloat32Slice(grads.role, scale)
	for i := range grads.layers {
		scaleFloat32Slice(grads.layers[i].attnQ, scale)
		scaleFloat32Slice(grads.layers[i].attnK, scale)
		scaleFloat32Slice(grads.layers[i].attnV, scale)
		scaleFloat32Slice(grads.layers[i].attnO, scale)
		scaleFloat32Slice(grads.layers[i].ffnUp, scale)
		scaleFloat32Slice(grads.layers[i].ffnDown, scale)
	}
	scaleFloat32Slice(grads.outputProjection, scale)
}

func compactTestFinite(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

func assertListwiseGeometryEvalMetricsClose(t *testing.T, got, want EmbeddingListwiseGeometryEvalMetrics) {
	t.Helper()
	assertClose(t, got.Loss, want.Loss, 0.000001)
	assertClose(t, got.AverageScore, want.AverageScore, 0.000001)
	assertClose(t, got.TeacherCrossEntropy, want.TeacherCrossEntropy, 0.000001)
	assertClose(t, got.TeacherKL, want.TeacherKL, 0.000001)
	assertClose(t, got.TeacherTop1Agreement, want.TeacherTop1Agreement, 0.000001)
	assertClose(t, got.AnyPositiveTop1, want.AnyPositiveTop1, 0.000001)
	if got.QueryCount != want.QueryCount ||
		got.DocumentCellCount != want.DocumentCellCount ||
		got.BatchCount != want.BatchCount ||
		got.AnyPositiveQueryCount != want.AnyPositiveQueryCount {
		t.Fatalf("listwise eval counts = %+v, want %+v", got, want)
	}
}

func snapshotCompactTrainStateTensors(state *CompactEmbeddingTrainState) []embeddingTrainTensorSnapshot {
	if state == nil {
		return nil
	}
	tensors := []*backend.Tensor{state.TokenEmbedding.Tensor}
	if state.RoleEmbedding != nil {
		tensors = append(tensors, state.RoleEmbedding.Tensor)
	}
	for i := range state.Layers {
		layer := state.Layers[i]
		tensors = append(
			tensors,
			layer.AttentionQuery.Tensor,
			layer.AttentionKey.Tensor,
			layer.AttentionValue.Tensor,
			layer.AttentionOutput.Tensor,
			layer.FFNUp.Tensor,
			layer.FFNDown.Tensor,
		)
	}
	if state.OutputProjection != nil {
		tensors = append(tensors, state.OutputProjection.Tensor)
	}
	return snapshotEmbeddingTrainTensors(tensors...)
}

func float32SlicesClose(a, b []float32, tol float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		diff := a[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			return false
		}
	}
	return true
}

func TestEmbeddingTrainerFFNCheckpointRoundTrip(t *testing.T) {
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	batch := tinyEmbeddingPairDataset()

	for i := 0; i < 4; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}

	checkpoint, err := trainer.Checkpoint()
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	restored, err := NewEmbeddingTrainerFromCheckpoint(trainer.module, checkpoint)
	if err != nil {
		t.Fatalf("restore checkpoint: %v", err)
	}

	if _, err := trainer.TrainStep(batch); err != nil {
		t.Fatalf("continue original trainer: %v", err)
	}
	if _, err := restored.TrainStep(batch); err != nil {
		t.Fatalf("continue restored trainer: %v", err)
	}

	exportA, err := trainer.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export original: %v", err)
	}
	exportB, err := restored.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export restored: %v", err)
	}
	assertTensorClose(t, exportA["token_embedding"], exportB["token_embedding"].Shape, exportB["token_embedding"].F32)
	assertTensorClose(t, exportA["ffn_up"], exportB["ffn_up"].Shape, exportB["ffn_up"].F32)
	assertTensorClose(t, exportA["projection"], exportB["projection"].Shape, exportB["projection"].F32)
}

func TestEmbeddingTrainerAttentionCheckpointRoundTrip(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	batch := tinyEmbeddingPairDataset()

	for i := 0; i < 4; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}

	checkpoint, err := trainer.Checkpoint()
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	restored, err := NewEmbeddingTrainerFromCheckpoint(trainer.module, checkpoint)
	if err != nil {
		t.Fatalf("restore checkpoint: %v", err)
	}
	t.Cleanup(restored.Close)

	if _, err := trainer.TrainStep(batch); err != nil {
		t.Fatalf("continue original trainer: %v", err)
	}
	if _, err := restored.TrainStep(batch); err != nil {
		t.Fatalf("continue restored trainer: %v", err)
	}

	exportA, err := trainer.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export original: %v", err)
	}
	exportB, err := restored.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export restored: %v", err)
	}
	for _, name := range []string{"token_embedding", "attn_q", "attn_k", "attn_v", "attn_o", "projection"} {
		assertTensorClose(t, exportA[name], exportB[name].Shape, exportB[name].F32)
	}
}

func TestEmbeddingTrainerTrainStepSupportsEncoderResidualLayerNormAndExportsQuantizedWeights(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.02)

	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0, 2}, RightTokens: []int32{1, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0},
		{LeftTokens: []int32{0, 0}, RightTokens: []int32{0, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0.5},
		{LeftTokens: []int32{1, 1}, RightTokens: []int32{1, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0.5},
		{LeftTokens: []int32{0, 1}, RightTokens: []int32{1, 0}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0},
	}
	beforeMaster := map[string][]float32{
		"token_embedding": append([]float32(nil), trainer.tokenEmbed.F32...),
		"attn_q":          append([]float32(nil), trainer.attentionQuery.F32...),
		"attn_k":          append([]float32(nil), trainer.attentionKey.F32...),
		"attn_v":          append([]float32(nil), trainer.attentionValue.F32...),
		"attn_o":          append([]float32(nil), trainer.attentionOutput.F32...),
		"ffn_up":          append([]float32(nil), trainer.hiddenProjection.F32...),
		"projection":      append([]float32(nil), trainer.projection.F32...),
	}

	before, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	for i := 0; i < 32; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}
	after, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval after: %v", err)
	}
	if after.Loss > before.Loss+0.000001 {
		t.Fatalf("encoder loss regressed: before=%f after=%f", before.Loss, after.Loss)
	}

	exported, err := trainer.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export weights: %v", err)
	}
	for _, name := range []string{"token_embedding", "attn_q", "attn_k", "attn_v", "attn_o", "ffn_up", "projection"} {
		if got := exported[name].DType; got != "q8" {
			t.Fatalf("%s export dtype = %q, want q8", name, got)
		}
	}
	changed := false
	for _, name := range []string{"token_embedding", "attn_q", "attn_k", "attn_v", "attn_o", "ffn_up", "projection"} {
		master := trainerMasterTensorByName(trainer, name)
		for i, value := range master.F32 {
			if abs32(value-beforeMaster[name][i]) > 1e-6 {
				changed = true
				break
			}
		}
		if changed {
			break
		}
	}
	if !changed {
		t.Fatal("expected encoder train step to update at least one exported weight tensor")
	}
}

func TestEmbeddingTrainerTrainStepSupportsRepeatedEncoderAndExportsQuantizedWeights(t *testing.T) {
	trainer := newTinyTrainableRepeatedEncoderEmbeddingTrainer(t, 0.02)
	if got := trainer.encoderRepeats(); got != 2 {
		t.Fatalf("encoder repeats = %d, want 2", got)
	}

	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0, 2}, RightTokens: []int32{1, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0},
		{LeftTokens: []int32{0, 0}, RightTokens: []int32{0, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0.5},
		{LeftTokens: []int32{1, 1}, RightTokens: []int32{1, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0.5},
		{LeftTokens: []int32{0, 1}, RightTokens: []int32{1, 0}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0},
	}
	before, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval before: %v", err)
	}
	beforeProjection := append([]float32(nil), trainer.projection.F32...)
	for i := 0; i < 24; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}
	after, err := trainer.EvalBatch(batch)
	if err != nil {
		t.Fatalf("eval after: %v", err)
	}
	if after.Loss > before.Loss+0.000001 {
		t.Fatalf("repeated encoder loss regressed: before=%f after=%f", before.Loss, after.Loss)
	}
	exported, err := trainer.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export weights: %v", err)
	}
	for _, name := range []string{"token_embedding", "attn_q", "attn_k", "attn_v", "attn_o", "ffn_up", "projection"} {
		if got := exported[name].DType; got != "q8" {
			t.Fatalf("%s export dtype = %q, want q8", name, got)
		}
	}
	changed := false
	for i, value := range trainer.projection.F32 {
		if abs32(value-beforeProjection[i]) > 1e-6 {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("expected repeated encoder train step to update projection weights")
	}
}

func TestEmbeddingTrainerEncoderV2PaddingInvariantAndOrderSensitive(t *testing.T) {
	trainer := newTinyTrainableRepeatedEncoderEmbeddingTrainer(t, 0.02)
	trainer.manifest.Tokenizer.MaxSequence = 8
	trainer.manifest.AttentionMaskMode = EmbeddingAttentionMaskModeKey
	trainer.manifest.PositionEncoding = EmbeddingPositionEncodingRoPE

	base := embedTrainerTokensForTest(t, trainer, []int32{0, 1}, []int32{1, 1})
	padded := embedTrainerTokensForTest(t, trainer, []int32{0, 1, 2}, []int32{1, 1, 0})
	padMaxAbs, padL2 := embeddingVectorDiffStats(base, padded)
	if padMaxAbs > 1e-6 || padL2 > 1e-6 {
		t.Fatalf("v2 padding changed embedding: max_abs=%.9g l2=%.9g base=%v padded=%v", padMaxAbs, padL2, base, padded)
	}

	ordered := embedTrainerTokensForTest(t, trainer, []int32{0, 1, 2}, []int32{1, 1, 1})
	reordered := embedTrainerTokensForTest(t, trainer, []int32{2, 1, 0}, []int32{1, 1, 1})
	orderMaxAbs, orderL2 := embeddingVectorDiffStats(ordered, reordered)
	if orderMaxAbs <= 1e-5 && orderL2 <= 1e-5 {
		t.Fatalf("v2 positional path is not order-sensitive: max_abs=%.9g l2=%.9g ordered=%v reordered=%v", orderMaxAbs, orderL2, ordered, reordered)
	}
}

func TestEmbeddingTrainerAttentionScoreScaleChangesForward(t *testing.T) {
	rawTrainer := newTinyTrainableRepeatedEncoderEmbeddingTrainer(t, 0.02)
	rawTrainer.manifest.Tokenizer.MaxSequence = 8
	rawTrainer.manifest.AttentionMaskMode = EmbeddingAttentionMaskModeKey
	rawTrainer.manifest.PositionEncoding = EmbeddingPositionEncodingRoPE
	rawTrainer.manifest.AttentionScoreScale = EmbeddingAttentionScoreScaleNone

	scaledTrainer := newTinyTrainableRepeatedEncoderEmbeddingTrainer(t, 0.02)
	scaledTrainer.manifest.Tokenizer.MaxSequence = 8
	scaledTrainer.manifest.AttentionMaskMode = EmbeddingAttentionMaskModeKey
	scaledTrainer.manifest.PositionEncoding = EmbeddingPositionEncodingRoPE
	scaledTrainer.manifest.AttentionScoreScale = EmbeddingAttentionScoreScaleKeyDimRSQ

	raw := embedTrainerTokensForTest(t, rawTrainer, []int32{0, 1, 2}, []int32{1, 1, 1})
	scaled := embedTrainerTokensForTest(t, scaledTrainer, []int32{0, 1, 2}, []int32{1, 1, 1})
	maxAbs, l2 := embeddingVectorDiffStats(raw, scaled)
	if maxAbs <= 1e-6 && l2 <= 1e-6 {
		t.Fatalf("scaled attention forward matched raw scores: max_abs=%.9g l2=%.9g raw=%v scaled=%v", maxAbs, l2, raw, scaled)
	}
}

func TestEmbeddingTrainerAttentionScoreScaleBackpropScalesQKOnly(t *testing.T) {
	state := func() *embeddingSequenceState {
		return &embeddingSequenceState{
			tokens:       []int32{0, 1},
			input:        []float32{1, 0, 0, 0, 0, 1, 0, 0},
			attnQ:        []float32{1, 0, 0, 1, -1, 2, 0.5, 0},
			attnK:        []float32{0.5, -1, 1, 0, 2, 0, -0.5, 1},
			attnV:        []float32{1, 2, 3, 4, -1, 0.5, 2, -2},
			attnScores:   []float32{0.8, 0.2, 0.3, 0.7},
			attnMixed:    []float32{0.6, 1.7, 2.8, 2.8, -0.4, 0.95, 2.3, -0.2},
			attnResidual: make([]float32, 8),
			hidden:       make([]float32, 8),
		}
	}
	identity := backend.NewTensorF32([]int{4, 4}, []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	})
	gradHidden := []float32{
		0.2, -0.4, 0.6, -0.8,
		1.1, 0.7, -0.3, 0.9,
	}
	run := func(scaleMode string) (gradQ, gradK, gradV []float32) {
		trainer := &EmbeddingTrainer{manifest: EmbeddingManifest{AttentionScoreScale: scaleMode}}
		gradQ = make([]float32, 16)
		gradK = make([]float32, 16)
		gradV = make([]float32, 16)
		gradO := make([]float32, 16)
		trainer.backpropAttentionSequence(state(), gradHidden, identity, identity, identity, identity, gradQ, gradK, gradV, gradO, 4)
		return gradQ, gradK, gradV
	}
	rawQ, rawK, rawV := run(EmbeddingAttentionScoreScaleNone)
	scaledQ, scaledK, scaledV := run(EmbeddingAttentionScoreScaleKeyDimRSQ)

	for i := range rawQ {
		assertClose(t, scaledQ[i], rawQ[i]*0.5, 1e-6)
		assertClose(t, scaledK[i], rawK[i]*0.5, 1e-6)
		assertClose(t, scaledV[i], rawV[i], 1e-6)
	}
}

func TestEmbeddingTrainerRoPEBackwardRotatesTokenEmbeddingGradients(t *testing.T) {
	trainer := &EmbeddingTrainer{
		manifest:   EmbeddingManifest{PositionEncoding: EmbeddingPositionEncodingRoPE},
		tokenEmbed: backend.NewTensorF32([]int{4, 4}, make([]float32, 16)),
	}
	tokens := []int32{1, 2, 1}
	gradInput := []float32{
		0.2, -0.4, 0.6, -0.8,
		1.1, 0.7, -0.3, 0.9,
		-0.5, 0.25, 1.5, -1.25,
	}
	originalGradInput := append([]float32(nil), gradInput...)
	got := make([]float32, len(trainer.tokenEmbed.F32))
	trainer.accumulateInputTokenGrad(tokens, gradInput, got)

	rotatedRows := append([]float32(nil), gradInput...)
	applyRoPETransposeToRowsInPlace(rotatedRows, len(tokens), trainer.tokenEmbed.Shape[1])
	want := make([]float32, len(got))
	accumulateTokenGrad(tokens, rotatedRows, want, trainer.tokenEmbed.Shape[1], trainer.tokenEmbed.Shape[0])

	raw := make([]float32, len(got))
	accumulateTokenGrad(tokens, gradInput, raw, trainer.tokenEmbed.Shape[1], trainer.tokenEmbed.Shape[0])

	maxAbs, _ := embeddingVectorDiffStats(got, want)
	if maxAbs > 1e-6 {
		t.Fatalf("RoPE token grad mismatch: max_abs=%.9g got=%v want=%v", maxAbs, got, want)
	}
	rawMaxAbs, _ := embeddingVectorDiffStats(raw, want)
	if rawMaxAbs <= 1e-4 {
		t.Fatalf("test did not distinguish raw accumulation from inverse-RoPE accumulation: raw=%v want=%v", raw, want)
	}
	mutatedMaxAbs, _ := embeddingVectorDiffStats(gradInput, originalGradInput)
	if mutatedMaxAbs != 0 {
		t.Fatalf("accumulateInputTokenGrad mutated input gradients: before=%v after=%v", originalGradInput, gradInput)
	}
}

func TestEmbeddingTrainerEncoderV1RemainsPermutationInvariantWithoutPositionEncoding(t *testing.T) {
	trainer := newTinyTrainableRepeatedEncoderEmbeddingTrainer(t, 0.02)
	trainer.manifest.Tokenizer.MaxSequence = 8

	ordered := embedTrainerTokensForTest(t, trainer, []int32{0, 1, 2}, []int32{1, 1, 1})
	reordered := embedTrainerTokensForTest(t, trainer, []int32{2, 1, 0}, []int32{1, 1, 1})
	maxAbs, l2 := embeddingVectorDiffStats(ordered, reordered)
	if maxAbs > 1e-6 || l2 > 1e-6 {
		t.Fatalf("v1 encoder unexpectedly became order-sensitive: max_abs=%.9g l2=%.9g ordered=%v reordered=%v", maxAbs, l2, ordered, reordered)
	}
}

func TestEmbeddingTrainerMaskedAttentionZerosPaddedKeyColumns(t *testing.T) {
	scores := []float32{
		1, 2, 8,
		3, 4, 9,
		5, 6, 10,
	}
	softmaxRowsMaskedColumnsInPlace(scores, 3, 3, []int32{1, 1, 0})
	for row := 0; row < 3; row++ {
		base := row * 3
		if scores[base+2] != 0 {
			t.Fatalf("masked key probability row %d = %.9g, want 0", row, scores[base+2])
		}
		assertClose(t, scores[base]+scores[base+1], 1, 1e-6)
	}
}

func embedTrainerTokensForTest(t *testing.T, trainer *EmbeddingTrainer, tokens, mask []int32) []float32 {
	t.Helper()
	preparedMask, err := trainer.prepareMask(tokens, mask)
	if err != nil {
		t.Fatalf("prepare mask tokens=%v mask=%v: %v", tokens, mask, err)
	}
	forward := trainer.prepareForwardWeights()
	seq, err := trainer.encodeSequence(tokens, preparedMask, 0, forward.token, forward.role, forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, false)
	if err != nil {
		t.Fatalf("encode tokens=%v mask=%v: %v", tokens, mask, err)
	}
	return append([]float32(nil), seq.pooled...)
}

func embeddingVectorDiffStats(a, b []float32) (float32, float32) {
	maxAbs := float32(0)
	sumSquares := float64(0)
	for i := range a {
		diff := abs32(a[i] - b[i])
		if diff > maxAbs {
			maxAbs = diff
		}
		sumSquares += float64(diff * diff)
	}
	return maxAbs, float32(math.Sqrt(sumSquares))
}

func TestEmbeddingTrainerEncoderCheckpointRoundTrip(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.02)
	batch := tinyEncoderPairDataset()

	for i := 0; i < 4; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}

	checkpoint, err := trainer.Checkpoint()
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	restored, err := NewEmbeddingTrainerFromCheckpoint(trainer.module, checkpoint)
	if err != nil {
		t.Fatalf("restore checkpoint: %v", err)
	}
	t.Cleanup(restored.Close)

	if _, err := trainer.TrainStep(batch); err != nil {
		t.Fatalf("continue original trainer: %v", err)
	}
	if _, err := restored.TrainStep(batch); err != nil {
		t.Fatalf("continue restored trainer: %v", err)
	}

	exportA, err := trainer.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export original: %v", err)
	}
	exportB, err := restored.ExportInferenceWeights()
	if err != nil {
		t.Fatalf("export restored: %v", err)
	}
	for _, name := range []string{"token_embedding", "attn_q", "attn_k", "attn_v", "attn_o", "ffn_up", "projection"} {
		assertTensorClose(t, exportA[name], exportB[name].Shape, exportB[name].F32)
	}
}

func TestEmbeddingTrainerForwardMatMulAcceleratorMatchesHost(t *testing.T) {
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul == nil {
		t.Skip("no trainer matmul accelerator available")
	}
	if trainer.forwardBackend != eosartifact.BackendCUDA && trainer.forwardBackend != eosartifact.BackendMetal {
		t.Fatalf("forward backend = %q, want cuda or metal", trainer.forwardBackend)
	}
	rhs := backend.NewTensorF32([]int{2, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	got, ok := trainer.tryForwardMatMul([]float32{
		1, 2,
		3, 4,
	}, 2, 2, rhs, 3)
	if !ok {
		t.Fatal("accelerated matmul was not used")
	}
	want := make([]float32, 6)
	fillHostMatMul([]float32{
		1, 2,
		3, 4,
	}, 2, 2, rhs.F32, 3, want)
	assertTensorClose(t, backend.NewTensorF32([]int{2, 3}, got), []int{2, 3}, want)
}

func TestEmbeddingTrainerTransposedMatMulAcceleratorMatchesHost(t *testing.T) {
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul == nil {
		t.Skip("no trainer matmul accelerator available")
	}
	lhs := []float32{
		1, 2, 3,
		4, 5, 6,
	}
	rhs := []float32{
		1, 0, 2, 1,
		0, 1, 3, 2,
	}
	got, ok := trainer.tryTrainerMatMul(lhs, 2, 3, rhs, 2, 4, true, false)
	if !ok {
		t.Fatal("accelerated transposed matmul was not used")
	}
	want := make([]float32, 12)
	fillHostMatMulTranspose(lhs, 2, 3, rhs, 2, 4, true, false, want)
	assertTensorClose(t, backend.NewTensorF32([]int{3, 4}, got), []int{3, 4}, want)
}

func TestEmbeddingTrainerBoundRightMatMulMatchesHostAndRefreshes(t *testing.T) {
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul == nil {
		t.Skip("no trainer matmul accelerator available")
	}
	lhs := backend.NewTensorF32([]int{2, 2}, []float32{
		1, 2,
		3, 4,
	})
	rhsA := backend.NewTensorF32([]int{2, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	if err := trainer.forwardMatMul.BindMatrix(trainer.projParam.Name, rhsA); err != nil {
		t.Fatalf("bind rhsA: %v", err)
	}
	resultA, err := trainer.forwardMatMul.RunMatMulWithBoundRight(lhs, trainer.projParam.Name, eosartifact.ValueType{
		Kind: eosartifact.ValueTensor,
		Tensor: &eosartifact.TensorType{
			DType: "f32",
		},
	}, false, false)
	if err != nil {
		t.Fatalf("run bound rhsA: %v", err)
	}
	wantA := make([]float32, 6)
	fillHostMatMul([]float32{
		1, 2,
		3, 4,
	}, 2, 2, rhsA.F32, 3, wantA)
	assertTensorClose(t, resultA.Outputs[0], []int{2, 3}, wantA)
	if got := resultA.Metadata["rhs_binding"]; got != trainer.projParam.Name {
		t.Fatalf("rhs_binding = %v, want %q", got, trainer.projParam.Name)
	}

	rhsB := backend.NewTensorF32([]int{2, 3}, []float32{
		2, 0, 1,
		1, 3, 2,
	})
	if err := trainer.forwardMatMul.BindMatrix(trainer.projParam.Name, rhsB); err != nil {
		t.Fatalf("bind rhsB: %v", err)
	}
	resultB, err := trainer.forwardMatMul.RunMatMulWithBoundRight(lhs, trainer.projParam.Name, eosartifact.ValueType{
		Kind: eosartifact.ValueTensor,
		Tensor: &eosartifact.TensorType{
			DType: "f32",
		},
	}, false, false)
	if err != nil {
		t.Fatalf("run bound rhsB: %v", err)
	}
	wantB := make([]float32, 6)
	fillHostMatMul([]float32{
		1, 2,
		3, 4,
	}, 2, 2, rhsB.F32, 3, wantB)
	assertTensorClose(t, resultB.Outputs[0], []int{2, 3}, wantB)
}

func TestEmbeddingTrainerBoundLeftMatMulMatchesHostAndRefreshes(t *testing.T) {
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul == nil {
		t.Skip("no trainer matmul accelerator available")
	}
	lhsA := backend.NewTensorF32([]int{2, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	rhs := backend.NewTensorF32([]int{2, 2}, []float32{
		1, 0,
		2, 1,
	})
	if err := trainer.forwardMatMul.BindMatrix(trainer.hiddenParam.Name, lhsA); err != nil {
		t.Fatalf("bind lhsA: %v", err)
	}
	resultA, err := trainer.forwardMatMul.RunMatMulWithBoundLeft(trainer.hiddenParam.Name, rhs, eosartifact.ValueType{
		Kind: eosartifact.ValueTensor,
		Tensor: &eosartifact.TensorType{
			DType: "f32",
		},
	}, true, false)
	if err != nil {
		t.Fatalf("run bound lhsA: %v", err)
	}
	wantA := make([]float32, 6)
	fillHostMatMulTranspose(lhsA.F32, 2, 3, rhs.F32, 2, 2, true, false, wantA)
	assertTensorClose(t, resultA.Outputs[0], []int{3, 2}, wantA)
	if got := resultA.Metadata["lhs_binding"]; got != trainer.hiddenParam.Name {
		t.Fatalf("lhs_binding = %v, want %q", got, trainer.hiddenParam.Name)
	}

	lhsB := backend.NewTensorF32([]int{2, 3}, []float32{
		2, 1, 0,
		3, 2, 1,
	})
	if err := trainer.forwardMatMul.BindMatrix(trainer.hiddenParam.Name, lhsB); err != nil {
		t.Fatalf("bind lhsB: %v", err)
	}
	resultB, err := trainer.forwardMatMul.RunMatMulWithBoundLeft(trainer.hiddenParam.Name, rhs, eosartifact.ValueType{
		Kind: eosartifact.ValueTensor,
		Tensor: &eosartifact.TensorType{
			DType: "f32",
		},
	}, true, false)
	if err != nil {
		t.Fatalf("run bound lhsB: %v", err)
	}
	wantB := make([]float32, 6)
	fillHostMatMulTranspose(lhsB.F32, 2, 3, rhs.F32, 2, 2, true, false, wantB)
	assertTensorClose(t, resultB.Outputs[0], []int{3, 2}, wantB)
}

func TestEmbeddingTrainerOptimizerAcceleratorMatchesHostAdamW(t *testing.T) {
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	if trainer.optimizerAccel == nil {
		t.Skip("no trainer optimizer accelerator available")
	}
	paramA := backend.NewTensorF32([]int{2, 2}, []float32{
		0.5, -0.25,
		1.0, -0.75,
	})
	mom1A := backend.NewTensorF32([]int{2, 2}, []float32{
		0.1, -0.05,
		0.2, -0.1,
	})
	mom2A := backend.NewTensorF32([]int{2, 2}, []float32{
		0.01, 0.02,
		0.03, 0.04,
	})
	grad := []float32{
		0.2, -0.1,
		0.05, -0.15,
	}
	paramB := paramA.Clone()
	mom1B := mom1A.Clone()
	mom2B := mom2A.Clone()

	cfg := trainer.optimizerUpdateConfig(0.5)
	cfg.Step = 3
	if err := trainer.optimizerAccel.ApplyUpdate(trainer.projParam.Name, cfg, paramA, mom1A, mom2A, backend.NewTensorF32([]int{2, 2}, grad)); err != nil {
		t.Fatalf("accelerated optimizer update: %v", err)
	}
	if err := trainer.optimizerAccel.SyncState(trainer.projParam.Name, paramA, mom1A, mom2A, true); err != nil {
		t.Fatalf("sync accelerated optimizer state: %v", err)
	}
	applyOptimizerUpdate(trainer.config, cfg.Step, paramB, mom1B, mom2B, grad, cfg.Scale)

	assertTensorClose(t, paramA, paramB.Shape, paramB.F32)
	assertTensorClose(t, mom1A, mom1B.Shape, mom1B.F32)
	assertTensorClose(t, mom2A, mom2B.Shape, mom2B.F32)
}

func TestEmbeddingTrainerCheckpointSyncsResidentOptimizerMoments(t *testing.T) {
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	if trainer.optimizerAccel == nil {
		t.Skip("no trainer optimizer accelerator available")
	}
	batch := tinyEmbeddingPairDataset()

	if _, err := trainer.TrainStep(batch); err != nil {
		t.Fatalf("train step: %v", err)
	}
	allZero := true
	for _, v := range trainer.projMom1.F32 {
		if abs32(v) > 1e-9 {
			allZero = false
			break
		}
	}
	if !allZero {
		t.Fatal("expected resident optimizer path to defer host moment sync until checkpoint")
	}

	checkpoint, err := trainer.Checkpoint()
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if checkpoint.ProjMoment1 == nil || checkpoint.ProjMoment2 == nil {
		t.Fatal("expected checkpoint to include projection moments")
	}
	nonZero := false
	for i := range checkpoint.ProjMoment1.F32 {
		if abs32(checkpoint.ProjMoment1.F32[i]) > 1e-9 || abs32(checkpoint.ProjMoment2.F32[i]) > 1e-9 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Fatal("expected checkpoint to sync resident optimizer moments")
	}
}

func TestTrainerActivationAccelModeFromEnv(t *testing.T) {
	for _, tc := range []struct {
		name    string
		env     map[string]string
		full    bool
		softmax bool
	}{
		{
			name: "default disabled",
		},
		{
			name: "full enables all activation backward",
			env: map[string]string{
				"EOS_TRAIN_ENABLE_ACTIVATION_ACCEL": "1",
			},
			full:    true,
			softmax: true,
		},
		{
			name: "softmax only",
			env: map[string]string{
				"EOS_TRAIN_ENABLE_SOFTMAX_BACKWARD_ACCEL": "1",
			},
			softmax: true,
		},
		{
			name: "global disable wins",
			env: map[string]string{
				"EOS_TRAIN_DISABLE_ACTIVATION_ACCEL":      "1",
				"EOS_TRAIN_ENABLE_ACTIVATION_ACCEL":       "1",
				"EOS_TRAIN_ENABLE_SOFTMAX_BACKWARD_ACCEL": "1",
			},
		},
		{
			name: "softmax disable can narrow full mode",
			env: map[string]string{
				"EOS_TRAIN_ENABLE_ACTIVATION_ACCEL":        "1",
				"EOS_TRAIN_DISABLE_SOFTMAX_BACKWARD_ACCEL": "1",
			},
			full: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("EOS_TRAIN_DISABLE_ACTIVATION_ACCEL", "")
			t.Setenv("EOS_TRAIN_ENABLE_ACTIVATION_ACCEL", "")
			t.Setenv("EOS_TRAIN_ENABLE_SOFTMAX_BACKWARD_ACCEL", "")
			t.Setenv("EOS_TRAIN_DISABLE_SOFTMAX_BACKWARD_ACCEL", "")
			for name, value := range tc.env {
				t.Setenv(name, value)
			}
			got := trainerActivationAccelModeFromEnv()
			if got.fullBackward != tc.full {
				t.Fatalf("full backward = %v, want %v", got.fullBackward, tc.full)
			}
			if got.softmaxBackward != tc.softmax {
				t.Fatalf("softmax backward = %v, want %v", got.softmaxBackward, tc.softmax)
			}
		})
	}
}

func TestFastGELUApproximationIsOptIn(t *testing.T) {
	t.Setenv("EOS_TRAIN_ENABLE_FAST_GELU", "")
	if fastGELUEnabled() {
		t.Fatal("fast GELU enabled by default")
	}
	t.Setenv("EOS_TRAIN_ENABLE_FAST_GELU", "1")
	if !fastGELUEnabled() {
		t.Fatal("fast GELU env did not enable approximation")
	}
	x := float32(1.25)
	if geluForwardMode(x, true) == geluForwardMode(x, false) {
		t.Fatal("fast GELU forward unexpectedly matched precise path exactly")
	}
	if geluBackwardMode(x, true) == geluBackwardMode(x, false) {
		t.Fatal("fast GELU backward unexpectedly matched precise path exactly")
	}
	if fastTanh(4) != 1 || fastTanh(-4) != -1 {
		t.Fatal("fast tanh did not clamp outside approximation range")
	}
}

func TestEmbeddingTrainerGELUBackwardAcceleratorMatchesHost(t *testing.T) {
	t.Setenv("EOS_TRAIN_ENABLE_ACTIVATION_ACCEL", "1")
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	if trainer.activationAccel == nil {
		t.Skip("no trainer activation accelerator available")
	}
	gradOut := []float32{
		0.2, -0.1, 0.05,
		-0.25, 0.4, -0.3,
	}
	preAct := []float32{
		-1.0, -0.5, 0.0,
		0.5, 1.0, 1.5,
	}
	got, ok := trainer.tryGELUBackwardMul(gradOut, preAct, 2, 3, "")
	if !ok {
		t.Fatal("accelerated gelu backward was not used")
	}
	want := make([]float32, len(gradOut))
	for i := range want {
		want[i] = gradOut[i] * geluBackward(preAct[i])
	}
	assertTensorClose(t, backend.NewTensorF32([]int{2, 3}, got), []int{2, 3}, want)
}

func TestEmbeddingTrainerActivationAccelShapeLimitSkipsLargeUnboundCalls(t *testing.T) {
	t.Setenv("EOS_TRAIN_ACTIVATION_ACCEL_MAX_ELEMENTS", "4")
	activation := &countingActivationAccelerator{}
	trainer := &EmbeddingTrainer{
		activationAccel:      activation,
		activationAccelFull:  true,
		softmaxBackwardAccel: true,
	}
	grad := [][]float32{
		{0.2, -0.1, 0.05},
		{-0.25, 0.4, -0.3},
	}
	pre := [][]float32{
		{-1.0, -0.5, 0.0},
		{0.5, 1.0, 1.5},
	}
	if _, ok := trainer.tryBatchedGELUBackwardMul(grad, pre, 1, 3); ok {
		t.Fatal("expected large unbound activation call to fall back to host")
	}
	if activation.geluBackwardCalls != 0 {
		t.Fatalf("gelu backward calls = %d, want 0", activation.geluBackwardCalls)
	}

	t.Setenv("EOS_TRAIN_ACTIVATION_ACCEL_MAX_ELEMENTS", "0")
	if _, ok := trainer.tryBatchedGELUBackwardMul(grad, pre, 1, 3); !ok {
		t.Fatal("expected unlimited activation shape limit to allow accelerator")
	}
	if activation.geluBackwardCalls != 1 {
		t.Fatalf("gelu backward calls = %d, want 1", activation.geluBackwardCalls)
	}
}

func TestEmbeddingTrainerActivationAccelShapeLimitAllowsBoundInputs(t *testing.T) {
	t.Setenv("EOS_TRAIN_ACTIVATION_ACCEL_MAX_ELEMENTS", "4")
	activation := &countingActivationAccelerator{}
	trainer := &EmbeddingTrainer{
		activationAccel:     activation,
		activationAccelFull: true,
	}
	pre := []float32{-1.0, -0.5, 0.0, 0.5, 1.0, 1.5}
	if err := activation.BindTensor("pre", backend.NewTensorF32([]int{2, 3}, pre)); err != nil {
		t.Fatalf("bind pre: %v", err)
	}
	grad := []float32{0.2, -0.1, 0.05, -0.25, 0.4, -0.3}
	if _, ok := trainer.tryGELUBackwardMul(grad, nil, 2, 3, "pre"); !ok {
		t.Fatal("expected bound activation input to bypass unbound shape limit")
	}
	if activation.geluBackwardCalls != 1 {
		t.Fatalf("gelu backward calls = %d, want 1", activation.geluBackwardCalls)
	}
}

func TestEmbeddingTrainerSoftmaxBackwardAcceleratorMatchesHost(t *testing.T) {
	t.Setenv("EOS_TRAIN_ENABLE_ACTIVATION_ACCEL", "")
	t.Setenv("EOS_TRAIN_ENABLE_SOFTMAX_BACKWARD_ACCEL", "1")
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.activationAccel == nil {
		t.Skip("no trainer activation accelerator available")
	}
	if trainer.activationAccelFull {
		t.Fatal("softmax-only env unexpectedly enabled full activation backward")
	}
	if !trainer.softmaxBackwardAccel {
		t.Fatal("softmax-only env did not enable softmax backward acceleration")
	}
	gradOut := []float32{
		0.3, -0.1,
		-0.2, 0.4,
	}
	probs := []float32{
		0.7, 0.3,
		0.25, 0.75,
	}
	got, ok := trainer.trySoftmaxBackwardRows(gradOut, probs, 2, 2, "")
	if !ok {
		t.Fatal("accelerated softmax backward was not used")
	}
	want := make([]float32, len(gradOut))
	for row := 0; row < 2; row++ {
		backwardSoftmaxRow(want[row*2:(row+1)*2], gradOut[row*2:(row+1)*2], probs[row*2:(row+1)*2])
	}
	assertTensorClose(t, backend.NewTensorF32([]int{2, 2}, got), []int{2, 2}, want)
	if _, ok := trainer.tryGELUBackwardMul(gradOut, probs, 2, 2, ""); ok {
		t.Fatal("softmax-only env unexpectedly enabled gelu backward acceleration")
	}
}

func TestEmbeddingTrainerLayerNormBackwardAcceleratorMatchesHost(t *testing.T) {
	t.Setenv("EOS_TRAIN_ENABLE_ACTIVATION_ACCEL", "1")
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.05)
	if trainer.activationAccel == nil {
		t.Skip("no trainer activation accelerator available")
	}
	gradOut := []float32{
		0.2, -0.1, 0.3,
		-0.4, 0.25, 0.15,
	}
	pre := []float32{
		1.2, -0.4, 0.1,
		0.5, 1.0, -0.5,
	}
	normalized := make([]float32, len(pre))
	for row := 0; row < 2; row++ {
		layerNormRow(normalized[row*3:(row+1)*3], pre[row*3:(row+1)*3])
	}
	got, ok := trainer.tryLayerNormBackwardRows(gradOut, normalized, pre, 2, 3, "", "")
	if !ok {
		t.Fatal("accelerated layernorm backward was not used")
	}
	want := make([]float32, len(gradOut))
	for row := 0; row < 2; row++ {
		backwardLayerNormRow(
			want[row*3:(row+1)*3],
			gradOut[row*3:(row+1)*3],
			normalized[row*3:(row+1)*3],
			pre[row*3:(row+1)*3],
		)
	}
	assertTensorClose(t, backend.NewTensorF32([]int{2, 3}, got), []int{2, 3}, want)
}

func TestEmbeddingTrainerBatchedGELUBackwardAcceleratorMatchesHost(t *testing.T) {
	t.Setenv("EOS_TRAIN_ENABLE_ACTIVATION_ACCEL", "1")
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	if trainer.activationAccel == nil {
		t.Skip("no trainer activation accelerator available")
	}
	gradOut := [][]float32{
		{
			0.2, -0.1, 0.05,
			-0.25, 0.4, -0.3,
		},
		{
			-0.15, 0.35, -0.05,
			0.45, -0.2, 0.1,
		},
	}
	preAct := [][]float32{
		{
			-1.0, -0.5, 0.0,
			0.5, 1.0, 1.5,
		},
		{
			1.25, -1.25, 0.75,
			-0.75, 0.25, -0.25,
		},
	}
	got, ok := trainer.tryBatchedGELUBackwardMul(gradOut, preAct, 2, 3)
	if !ok {
		t.Fatal("accelerated batched gelu backward was not used")
	}
	if len(got) != len(gradOut) {
		t.Fatalf("batched gelu outputs = %d, want %d", len(got), len(gradOut))
	}
	for batch := range got {
		want := make([]float32, len(gradOut[batch]))
		for i := range want {
			want[i] = gradOut[batch][i] * geluBackward(preAct[batch][i])
		}
		assertTensorClose(t, backend.NewTensorF32([]int{2, 3}, got[batch]), []int{2, 3}, want)
	}
}

func TestEmbeddingTrainerBatchedSoftmaxBackwardAcceleratorMatchesHost(t *testing.T) {
	t.Setenv("EOS_TRAIN_ENABLE_ACTIVATION_ACCEL", "1")
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.activationAccel == nil {
		t.Skip("no trainer activation accelerator available")
	}
	gradOut := [][]float32{
		{
			0.3, -0.1,
			-0.2, 0.4,
		},
		{
			0.1, -0.25,
			0.35, -0.15,
		},
	}
	probs := [][]float32{
		{
			0.7, 0.3,
			0.25, 0.75,
		},
		{
			0.6, 0.4,
			0.1, 0.9,
		},
	}
	got, ok := trainer.tryBatchedSoftmaxBackwardRows(gradOut, probs, 2, 2)
	if !ok {
		t.Fatal("accelerated batched softmax backward was not used")
	}
	if len(got) != len(gradOut) {
		t.Fatalf("batched softmax outputs = %d, want %d", len(got), len(gradOut))
	}
	for batch := range got {
		want := make([]float32, len(gradOut[batch]))
		for row := 0; row < 2; row++ {
			backwardSoftmaxRow(want[row*2:(row+1)*2], gradOut[batch][row*2:(row+1)*2], probs[batch][row*2:(row+1)*2])
		}
		assertTensorClose(t, backend.NewTensorF32([]int{2, 2}, got[batch]), []int{2, 2}, want)
	}
}

func TestEmbeddingTrainerBatchedLayerNormBackwardAcceleratorMatchesHost(t *testing.T) {
	t.Setenv("EOS_TRAIN_ENABLE_ACTIVATION_ACCEL", "1")
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.05)
	if trainer.activationAccel == nil {
		t.Skip("no trainer activation accelerator available")
	}
	gradOut := [][]float32{
		{
			0.2, -0.1, 0.3,
			-0.4, 0.25, 0.15,
		},
		{
			-0.35, 0.05, 0.45,
			0.1, -0.3, 0.2,
		},
	}
	pre := [][]float32{
		{
			1.2, -0.4, 0.1,
			0.5, 1.0, -0.5,
		},
		{
			-1.0, 0.75, 0.25,
			1.5, -0.25, 0.0,
		},
	}
	normalized := make([][]float32, len(pre))
	for batch := range pre {
		normalized[batch] = make([]float32, len(pre[batch]))
		for row := 0; row < 2; row++ {
			layerNormRow(normalized[batch][row*3:(row+1)*3], pre[batch][row*3:(row+1)*3])
		}
	}
	got, ok := trainer.tryBatchedLayerNormBackwardRows(gradOut, normalized, pre, 2, 3)
	if !ok {
		t.Fatal("accelerated batched layernorm backward was not used")
	}
	if len(got) != len(gradOut) {
		t.Fatalf("batched layernorm outputs = %d, want %d", len(got), len(gradOut))
	}
	for batch := range got {
		want := make([]float32, len(gradOut[batch]))
		for row := 0; row < 2; row++ {
			backwardLayerNormRow(
				want[row*3:(row+1)*3],
				gradOut[batch][row*3:(row+1)*3],
				normalized[batch][row*3:(row+1)*3],
				pre[batch][row*3:(row+1)*3],
			)
		}
		assertTensorClose(t, backend.NewTensorF32([]int{2, 3}, got[batch]), []int{2, 3}, want)
	}
}

func TestEmbeddingTrainerBatchedForwardSkipsUnusedActivationBindings(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	matmul := &countingMatMulAccelerator{}
	activation := &countingActivationAccelerator{}
	trainer.forwardMatMul = matmul
	trainer.activationAccel = activation
	trainer.activationAccelFull = true
	trainer.softmaxBackwardAccel = true

	forward := trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	batch := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{0, 0}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 1}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
	}
	queries, positives, err := trainer.encodeContrastiveBatch(batch, forward, true)
	if err != nil {
		t.Fatalf("encode contrastive batch: %v", err)
	}
	defer trainer.releaseEncodedSequences(queries)
	defer trainer.releaseEncodedSequences(positives)
	if activation.bindCalls != 0 {
		t.Fatalf("activation bind calls = %d, want 0 unused per-sequence activation binds in batched forward", activation.bindCalls)
	}
	if matmul.boundRightRuns == 0 {
		t.Fatal("expected batched forward to keep using matmul acceleration")
	}
}

func TestEmbeddingTrainerBatchedForwardSkipsSingletonActivationBindings(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	trainer.forwardMatMul = &countingMatMulAccelerator{}
	activation := &countingActivationAccelerator{}
	trainer.activationAccel = activation
	trainer.activationAccelFull = true
	trainer.softmaxBackwardAccel = true

	forward := trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	batch := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0}, PositiveTokens: []int32{0, 0}, QueryMask: []int32{1}, PositiveMask: []int32{1, 1}},
	}
	queries, positives, err := trainer.encodeContrastiveBatch(batch, forward, true)
	if err != nil {
		t.Fatalf("encode contrastive batch: %v", err)
	}
	defer trainer.releaseEncodedSequences(queries)
	defer trainer.releaseEncodedSequences(positives)
	if activation.bindCalls != 0 {
		t.Fatalf("activation bind calls = %d, want 0 unused singleton activation binds in batched forward", activation.bindCalls)
	}
}

func TestEmbeddingTrainerBatchedForwardKeepsActivationBindingsWhenBatchedBackwardDisabled(t *testing.T) {
	t.Setenv("EOS_TRAIN_DISABLE_BATCHED_BACKWARD", "1")
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	trainer.forwardMatMul = &countingMatMulAccelerator{}
	activation := &countingActivationAccelerator{}
	trainer.activationAccel = activation
	trainer.activationAccelFull = true
	trainer.softmaxBackwardAccel = true

	forward := trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	batch := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{0, 0}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 1}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
	}
	queries, positives, err := trainer.encodeContrastiveBatch(batch, forward, true)
	if err != nil {
		t.Fatalf("encode contrastive batch: %v", err)
	}
	defer trainer.releaseEncodedSequences(queries)
	defer trainer.releaseEncodedSequences(positives)
	if activation.bindCalls == 0 {
		t.Fatal("expected activation bindings when batched backward is disabled")
	}
}

func TestEmbeddingTrainerBatchedBackwardSkipsSingletonUnboundActivationKernels(t *testing.T) {
	t.Setenv("EOS_TRAIN_ENABLE_FAST_GELU", "")
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	trainer.forwardMatMul = &countingMatMulAccelerator{}
	activation := &countingActivationAccelerator{}
	trainer.activationAccel = activation
	trainer.activationAccelFull = true
	trainer.softmaxBackwardAccel = true

	batch := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0}, PositiveTokens: []int32{0}, QueryMask: []int32{1}, PositiveMask: []int32{1}},
		{QueryTokens: []int32{1}, PositiveTokens: []int32{1, 2}, QueryMask: []int32{1}, PositiveMask: []int32{1, 1}},
	}
	if _, err := trainer.TrainContrastiveStep(batch); err != nil {
		t.Fatalf("train contrastive step: %v", err)
	}
	if activation.geluBackwardCalls != 1 {
		t.Fatalf("gelu backward calls = %d, want 1 grouped batched call", activation.geluBackwardCalls)
	}
	if activation.softmaxBackwardCalls != 1 {
		t.Fatalf("softmax backward calls = %d, want 1 grouped batched call", activation.softmaxBackwardCalls)
	}
	if activation.layerNormBackwardCalls != 2 {
		t.Fatalf("layernorm backward calls = %d, want 2 grouped batched calls", activation.layerNormBackwardCalls)
	}
}

func TestEmbeddingTrainerSingleForwardKeepsActivationBindings(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.05)
	activation := &countingActivationAccelerator{}
	trainer.activationAccel = activation
	trainer.activationAccelFull = true
	trainer.softmaxBackwardAccel = true

	forward := trainer.prepareForwardWeights()
	mask, err := trainer.prepareMask([]int32{0, 2}, nil)
	if err != nil {
		t.Fatalf("prepare mask: %v", err)
	}
	seq, err := trainer.encodeSequence([]int32{0, 2}, mask, 0, forward.token, forward.role, forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, true)
	if err != nil {
		t.Fatalf("encode sequence: %v", err)
	}
	defer trainer.releaseEncodedSequenceBindings(seq)
	if activation.bindCalls == 0 {
		t.Fatal("expected single-sequence forward to keep activation bindings for bound backward paths")
	}
}

func TestEmbeddingTrainerAttentionActivationsBindAndRelease(t *testing.T) {
	t.Setenv("EOS_TRAIN_ENABLE_SEQUENCE_MATMUL_BINDINGS", "1")
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul == nil {
		t.Skip("no trainer matmul accelerator available")
	}
	tokenForward := forwardTensorForParam(trainer.tokenParam, trainer.tokenEmbed, trainer.config.WeightBits)
	attnQForward := forwardTensorForParam(trainer.attnQParam, trainer.attentionQuery, trainer.config.WeightBits)
	attnKForward := forwardTensorForParam(trainer.attnKParam, trainer.attentionKey, trainer.config.WeightBits)
	attnVForward := forwardTensorForParam(trainer.attnVParam, trainer.attentionValue, trainer.config.WeightBits)
	attnOForward := forwardTensorForParam(trainer.attnOParam, trainer.attentionOutput, trainer.config.WeightBits)
	projForward := forwardTensorForParam(trainer.projParam, trainer.projection, trainer.config.WeightBits)
	trainer.primeForwardWeightResidency(attnQForward, attnKForward, attnVForward, attnOForward, nil, projForward)

	mask, err := trainer.prepareMask([]int32{0, 2}, nil)
	if err != nil {
		t.Fatalf("prepare mask: %v", err)
	}
	state, err := trainer.encodeSequence([]int32{0, 2}, mask, 0, tokenForward, nil, attnQForward, attnKForward, attnVForward, attnOForward, nil, projForward, true)
	if err != nil {
		t.Fatalf("encode sequence: %v", err)
	}
	layer := state.finalLayer()
	if layer == nil {
		t.Fatal("expected final attention layer")
	}
	if layer.inputBinding == "" || layer.hiddenBinding == "" || layer.attnQBinding == "" || layer.attnKBinding == "" || layer.attnVBinding == "" || layer.attnScoresBinding == "" || layer.attnMixedBinding == "" {
		t.Fatalf("expected attention activation bindings, got input=%q hidden=%q q=%q k=%q v=%q scores=%q mixed=%q", layer.inputBinding, layer.hiddenBinding, layer.attnQBinding, layer.attnKBinding, layer.attnVBinding, layer.attnScoresBinding, layer.attnMixedBinding)
	}
	result, err := trainer.forwardMatMul.RunMatMulWithBoundRight(
		backend.NewTensorF32([]int{2, 2}, layer.attnQ),
		layer.attnKBinding,
		eosartifact.ValueType{Kind: eosartifact.ValueTensor, Tensor: &eosartifact.TensorType{DType: "f32"}},
		false,
		true,
	)
	if err != nil {
		t.Fatalf("run with bound attention key: %v", err)
	}
	wantScores := make([]float32, 4)
	fillHostMatMulTranspose(layer.attnQ, 2, 2, layer.attnK, 2, 2, false, true, wantScores)
	assertTensorClose(t, result.Outputs[0], []int{2, 2}, wantScores)
	leftResult, err := trainer.forwardMatMul.RunMatMulWithBoundLeft(
		layer.inputBinding,
		backend.NewTensorF32([]int{2, 2}, layer.attnQ),
		eosartifact.ValueType{Kind: eosartifact.ValueTensor, Tensor: &eosartifact.TensorType{DType: "f32"}},
		true,
		false,
	)
	if err != nil {
		t.Fatalf("run with bound input activation: %v", err)
	}
	wantGrad := make([]float32, 4)
	fillHostMatMulTranspose(layer.input, 2, 2, layer.attnQ, 2, 2, true, false, wantGrad)
	assertTensorClose(t, leftResult.Outputs[0], []int{2, 2}, wantGrad)
	trainer.releaseEncodedSequenceBindings(state)
	if layer.inputBinding != "" || layer.hiddenBinding != "" || layer.attnQBinding != "" || layer.attnKBinding != "" || layer.attnVBinding != "" || layer.attnScoresBinding != "" || layer.attnMixedBinding != "" {
		t.Fatalf("expected bindings released, got input=%q hidden=%q q=%q k=%q v=%q scores=%q mixed=%q", layer.inputBinding, layer.hiddenBinding, layer.attnQBinding, layer.attnKBinding, layer.attnVBinding, layer.attnScoresBinding, layer.attnMixedBinding)
	}
	if _, err := trainer.forwardMatMul.RunMatMulWithBoundRight(
		backend.NewTensorF32([]int{2, 2}, layer.attnQ),
		"seq_missing_k",
		eosartifact.ValueType{Kind: eosartifact.ValueTensor, Tensor: &eosartifact.TensorType{DType: "f32"}},
		false,
		true,
	); err == nil {
		t.Fatal("expected missing bound-right activation error")
	}
	if _, err := trainer.forwardMatMul.RunMatMulWithBoundLeft(
		"seq_missing_input",
		backend.NewTensorF32([]int{2, 2}, layer.attnQ),
		eosartifact.ValueType{Kind: eosartifact.ValueTensor, Tensor: &eosartifact.TensorType{DType: "f32"}},
		true,
		false,
	); err == nil {
		t.Fatal("expected missing bound-left activation error")
	}
}

func TestEmbeddingTrainerFFNActivationsBindAndRelease(t *testing.T) {
	t.Setenv("EOS_TRAIN_ENABLE_SEQUENCE_MATMUL_BINDINGS", "1")
	trainer := newTinyTrainableFFNEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul == nil {
		t.Skip("no trainer matmul accelerator available")
	}
	tokenForward := forwardTensorForParam(trainer.tokenParam, trainer.tokenEmbed, trainer.config.WeightBits)
	hiddenForward := forwardTensorForParam(trainer.hiddenParam, trainer.hiddenProjection, trainer.config.WeightBits)
	projForward := forwardTensorForParam(trainer.projParam, trainer.projection, trainer.config.WeightBits)
	trainer.primeForwardWeightResidency(nil, nil, nil, nil, hiddenForward, projForward)

	mask, err := trainer.prepareMask([]int32{0, 2}, nil)
	if err != nil {
		t.Fatalf("prepare mask: %v", err)
	}
	state, err := trainer.encodeSequence([]int32{0, 2}, mask, 0, tokenForward, nil, nil, nil, nil, nil, hiddenForward, projForward, true)
	if err != nil {
		t.Fatalf("encode sequence: %v", err)
	}
	layer := state.finalLayer()
	if layer == nil {
		t.Fatal("expected final ffn layer")
	}
	if layer.inputBinding == "" || layer.hiddenBinding == "" || layer.activatedBinding == "" {
		t.Fatalf("expected ffn activation bindings, got input=%q hidden=%q activated=%q", layer.inputBinding, layer.hiddenBinding, layer.activatedBinding)
	}
	result, err := trainer.forwardMatMul.RunMatMulWithBoundLeft(
		layer.activatedBinding,
		backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
		eosartifact.ValueType{Kind: eosartifact.ValueTensor, Tensor: &eosartifact.TensorType{DType: "f32"}},
		true,
		false,
	)
	if err != nil {
		t.Fatalf("run with bound activated state: %v", err)
	}
	want := make([]float32, 6)
	fillHostMatMulTranspose(layer.activated, 2, 3, []float32{
		1, 0,
		0, 1,
	}, 2, 2, true, false, want)
	assertTensorClose(t, result.Outputs[0], []int{3, 2}, want)
	trainer.releaseEncodedSequenceBindings(state)
	if layer.inputBinding != "" || layer.hiddenBinding != "" || layer.activatedBinding != "" {
		t.Fatalf("expected ffn bindings released, got input=%q hidden=%q activated=%q", layer.inputBinding, layer.hiddenBinding, layer.activatedBinding)
	}
}

func TestEmbeddingTrainerForwardWeightCacheReusesTensorsUntilUpdate(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	first := trainer.prepareForwardWeights()
	second := trainer.prepareForwardWeights()
	if first == nil || second == nil {
		t.Fatal("expected cached forward weights")
	}
	if first != second {
		t.Fatal("expected prepareForwardWeights to reuse cached forward weight bundle")
	}
	if first.token != second.token || first.attnQ != second.attnQ || first.attnK != second.attnK || first.attnV != second.attnV || first.attnO != second.attnO || first.proj != second.proj {
		t.Fatal("expected cached forward tensors to be reused")
	}

	if err := trainer.applyOptimizerUpdate(trainer.projParam.Name, trainer.projection, trainer.projMom1, trainer.projMom2, make([]float32, len(trainer.projection.F32)), 1); err != nil {
		t.Fatalf("apply optimizer update: %v", err)
	}

	third := trainer.prepareForwardWeights()
	if third == nil {
		t.Fatal("expected forward weights after update")
	}
	if third != first {
		t.Fatal("expected optimizer update to refresh cached forward weight bundle in place")
	}
	if third.proj != first.proj {
		t.Fatal("expected projection forward tensor to be refreshed in place")
	}
}

func TestEmbeddingTrainerPrimeForwardWeightResidencySkipsRedundantBinds(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake

	forward := trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	firstCalls := fake.bindCalls
	if firstCalls != 5 {
		t.Fatalf("initial bind calls = %d, want 5", firstCalls)
	}

	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	if fake.bindCalls != firstCalls {
		t.Fatalf("redundant bind calls = %d, want %d", fake.bindCalls, firstCalls)
	}
	stats := trainer.ForwardResidencyStats()
	if stats.BindSkips != 1 {
		t.Fatalf("bind skips = %d, want 1", stats.BindSkips)
	}
	if stats.MatMul.BindCalls != int64(firstCalls) {
		t.Fatalf("backend bind calls = %d, want %d", stats.MatMul.BindCalls, firstCalls)
	}

	if err := trainer.applyOptimizerUpdate(trainer.projParam.Name, trainer.projection, trainer.projMom1, trainer.projMom2, make([]float32, len(trainer.projection.F32)), 1); err != nil {
		t.Fatalf("apply optimizer update: %v", err)
	}
	forward = trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	if fake.bindCalls != firstCalls+5 {
		t.Fatalf("rebind calls after invalidation = %d, want %d", fake.bindCalls, firstCalls+5)
	}
	stats = trainer.ForwardResidencyStats()
	if stats.BindSkips != 1 {
		t.Fatalf("bind skips after invalidation = %d, want 1", stats.BindSkips)
	}
	if stats.MatMul.BindCalls != int64(firstCalls+5) {
		t.Fatalf("backend bind calls after invalidation = %d, want %d", stats.MatMul.BindCalls, firstCalls+5)
	}
}

func TestEmbeddingTrainerEvaluatePairsSkipsSequenceBindingChurn(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake

	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0, 2}, RightTokens: []int32{0, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 1},
	}
	if _, err := trainer.EvaluatePairs(batch); err != nil {
		t.Fatalf("evaluate pairs: %v", err)
	}
	if fake.bindCalls != 5 {
		t.Fatalf("bind calls = %d, want only 5 forward-weight binds", fake.bindCalls)
	}
	stats := trainer.ForwardResidencyStats()
	if stats.MatMul.BindCalls != 5 {
		t.Fatalf("forward residency bind calls = %d, want 5", stats.MatMul.BindCalls)
	}
}

func TestEmbeddingTrainerEvaluatePairsUsesBatchedForwardChunks(t *testing.T) {
	t.Setenv("EOS_TRAIN_PAIR_EVAL_BATCH_SIZE", "2")
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake

	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0, 2}, RightTokens: []int32{1, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 1},
		{LeftTokens: []int32{2, 0}, RightTokens: []int32{2, 1}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 1},
		{LeftTokens: []int32{0, 2}, RightTokens: []int32{1, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: -1},
	}
	metrics, err := trainer.EvaluatePairs(batch)
	if err != nil {
		t.Fatalf("evaluate pairs: %v", err)
	}
	if metrics.PairCount != len(batch) {
		t.Fatalf("pair count = %d, want %d", metrics.PairCount, len(batch))
	}
	if trainer.sequenceBindingID != 0 {
		t.Fatalf("sequence binding count = %d, want default sequence matmul bindings disabled", trainer.sequenceBindingID)
	}
	if fake.bindCalls != 5 {
		t.Fatalf("bind calls = %d, want only 5 forward-weight binds", fake.bindCalls)
	}
	if fake.multiBoundRuns == 0 {
		t.Fatal("expected pairwise eval to coalesce q/k/v bound-right matmuls")
	}
	if fake.maxBoundRightRows < 8 {
		t.Fatalf("max bound-right lhs rows = %d, want chunked pairwise eval rows", fake.maxBoundRightRows)
	}
}

func TestPairwiseEvalBatchSizeDefaultAndEnv(t *testing.T) {
	t.Setenv("EOS_TRAIN_PAIR_EVAL_BATCH_SIZE", "")
	if got := pairwiseEvalBatchSize(1024); got != 512 {
		t.Fatalf("default pairwise eval batch size = %d, want 512", got)
	}
	if got := pairwiseEvalBatchSize(128); got != 128 {
		t.Fatalf("capped pairwise eval batch size = %d, want total size", got)
	}
	t.Setenv("EOS_TRAIN_PAIR_EVAL_BATCH_SIZE", "64")
	if got := pairwiseEvalBatchSize(1024); got != 64 {
		t.Fatalf("env pairwise eval batch size = %d, want 64", got)
	}
}

func TestEmbeddingTrainerTrainContrastiveStepEncodesEachSequenceOncePerBatch(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	batch := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{0, 0}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 1}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{2, 0}, PositiveTokens: []int32{2, 2}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
	}

	forward := trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	queries, positives, err := trainer.encodeContrastiveBatch(batch, forward, true)
	if err != nil {
		t.Fatalf("encode contrastive batch: %v", err)
	}
	defer trainer.releaseEncodedSequences(queries)
	defer trainer.releaseEncodedSequences(positives)
	if trainer.sequenceBindingID != 0 {
		t.Fatalf("sequence binding count = %d, want default sequence matmul bindings disabled", trainer.sequenceBindingID)
	}
	if fake.boundRightRuns == 0 {
		t.Fatalf("batched forward path did not attempt bound-right matmul")
	}
	if fake.multiBoundRuns == 0 {
		t.Fatalf("batched forward path did not coalesce q/k/v bound-right matmuls")
	}
	if fake.maxBoundRightRows < 6 {
		t.Fatalf("max bound-right lhs rows = %d, want batched rows", fake.maxBoundRightRows)
	}
}

func TestEmbeddingTrainerTrainStepUsesBatchedPairwiseForward(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0, 2}, RightTokens: []int32{0, 0}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 1},
		{LeftTokens: []int32{1, 2}, RightTokens: []int32{1, 1}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 1},
		{LeftTokens: []int32{2, 0}, RightTokens: []int32{2, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0},
	}

	if _, err := trainer.TrainStep(batch); err != nil {
		t.Fatalf("train pairwise step: %v", err)
	}
	if trainer.step != 1 {
		t.Fatalf("step = %d, want 1", trainer.step)
	}
	if fake.boundRightRuns == 0 {
		t.Fatalf("pairwise train path did not attempt bound-right matmul")
	}
	if fake.multiBoundRuns == 0 {
		t.Fatalf("pairwise train path did not coalesce q/k/v bound-right matmuls")
	}
	if fake.maxBoundRightRows < 6 {
		t.Fatalf("max bound-right lhs rows = %d, want batched pair rows", fake.maxBoundRightRows)
	}
}

func TestEmbeddingTrainerEvalBatchUsesPairwiseEvalGate(t *testing.T) {
	t.Setenv("EOS_TRAIN_DISABLE_BATCHED_PAIR_TRAIN", "1")
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0, 2}, RightTokens: []int32{0, 0}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 1},
		{LeftTokens: []int32{1, 2}, RightTokens: []int32{1, 1}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 1},
		{LeftTokens: []int32{2, 0}, RightTokens: []int32{2, 2}, LeftMask: []int32{1, 1}, RightMask: []int32{1, 1}, Target: 0},
	}

	if _, err := trainer.EvalBatch(batch); err != nil {
		t.Fatalf("eval pairwise batch: %v", err)
	}
	if trainer.step != 0 {
		t.Fatalf("step = %d, want eval-only batch to leave step unchanged", trainer.step)
	}
	if fake.multiBoundRuns == 0 {
		t.Fatalf("pairwise eval path did not coalesce q/k/v bound-right matmuls")
	}
	if fake.maxBoundRightRows < 6 {
		t.Fatalf("max bound-right lhs rows = %d, want batched eval pair rows", fake.maxBoundRightRows)
	}
}

func TestEmbeddingTrainerBatchedForwardReusesDuplicateSequences(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	batch := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{1, 2}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{1, 2}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{2, 2}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
	}

	forward := trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	queries, positives, err := trainer.encodeContrastiveBatch(batch, forward, true)
	if err != nil {
		t.Fatalf("encode contrastive batch: %v", err)
	}
	defer trainer.releaseEncodedSequences(queries)
	defer trainer.releaseEncodedSequences(positives)
	if queries[0] != queries[1] || queries[1] != queries[2] {
		t.Fatalf("duplicate query token sequences were not reused")
	}
	if positives[0] != positives[1] {
		t.Fatalf("duplicate positive token sequences were not reused")
	}
	if positives[2] == positives[0] {
		t.Fatalf("distinct positive token sequence reused duplicate encoding")
	}
	if fake.maxBoundRightRows != 6 {
		t.Fatalf("max bound-right lhs rows = %d, want only 3 unique sequences of length 2", fake.maxBoundRightRows)
	}
}

func TestEmbeddingTrainerBatchedForwardGroupsVariableSequenceLengths(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	batch := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0}, PositiveTokens: []int32{0, 2}, QueryMask: []int32{1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{1}, PositiveTokens: []int32{1, 2}, QueryMask: []int32{1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{2, 0}, PositiveTokens: []int32{2}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1}},
	}

	forward := trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	queries, positives, err := trainer.encodeContrastiveBatch(batch, forward, true)
	if err != nil {
		t.Fatalf("encode contrastive batch: %v", err)
	}
	defer trainer.releaseEncodedSequences(queries)
	defer trainer.releaseEncodedSequences(positives)
	if trainer.sequenceBindingID != 0 {
		t.Fatalf("sequence binding count = %d, want default sequence matmul bindings disabled", trainer.sequenceBindingID)
	}
	if fake.boundRightRuns == 0 {
		t.Fatalf("batched forward path did not attempt bound-right matmul")
	}
	if fake.multiBoundRuns == 0 {
		t.Fatalf("batched forward path did not coalesce q/k/v bound-right matmuls")
	}
	if fake.maxBoundRightRows <= 2 {
		t.Fatalf("max bound-right lhs rows = %d, want length-grouped rows above any single sequence", fake.maxBoundRightRows)
	}
	if fake.runCalls == 0 || fake.maxRunBatches < 2 {
		t.Fatalf("attention matmul run calls=%d max batches=%d, want batched attention dispatch", fake.runCalls, fake.maxRunBatches)
	}
}

func TestEmbeddingTrainerQKVMultiBoundCanBeDisabled(t *testing.T) {
	t.Setenv("EOS_TRAIN_DISABLE_QKV_MULTI_BOUND", "1")
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	batch := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{0, 0}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 1}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
	}

	forward := trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	queries, positives, err := trainer.encodeContrastiveBatch(batch, forward, true)
	if err != nil {
		t.Fatalf("encode contrastive batch: %v", err)
	}
	defer trainer.releaseEncodedSequences(queries)
	defer trainer.releaseEncodedSequences(positives)
	if fake.multiBoundRuns != 0 {
		t.Fatalf("multi-bound q/k/v runs = %d, want disabled", fake.multiBoundRuns)
	}
	if fake.boundRightRuns == 0 {
		t.Fatal("expected fallback bound-right matmuls when q/k/v coalescing is disabled")
	}
}

func TestEmbeddingTrainerAttentionBackwardUsesConcatenatedSharedLeftQKVGradMatMul(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.005)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05

	if _, err := trainer.TrainContrastiveStep(tinyEncoderContrastiveDataset()); err != nil {
		t.Fatalf("train contrastive step: %v", err)
	}
	if fake.sharedLeftRuns != 0 {
		t.Fatalf("shared-left backend runs = %d, want concatenated standard matmul path", fake.sharedLeftRuns)
	}
	if fake.maxRunOutputCols < 6 {
		t.Fatalf("max standard matmul output cols = %d, want concatenated q/k/v width", fake.maxRunOutputCols)
	}
}

func TestEmbeddingTrainerConcatenatedSharedLeftQKVGradMatchesSeparateMatMuls(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.005)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	trainer.forwardMatMul = &countingMatMulAccelerator{}

	lhsMatrices := [][]float32{
		{
			1, 2,
			3, 4,
		},
		{
			-1, 2,
			0.5, -3,
		},
	}
	rhsA := [][]float32{
		{
			0.5, -1,
			2, 3,
		},
		{
			1, 4,
			-2, 0.25,
		},
	}
	rhsB := [][]float32{
		{
			1.5, 0.25,
			-0.5, 2,
		},
		{
			3, -1,
			0.75, 2,
		},
	}
	rhsC := [][]float32{
		{
			-1, 2,
			4, 0.5,
		},
		{
			2, 1,
			-1.5, 3,
		},
	}

	got, ok := trainer.tryConcatenatedSharedLeftAccumulatedTransposeMatMuls(lhsMatrices, [][][]float32{rhsA, rhsB, rhsC}, 2, 2, 2)
	if !ok {
		t.Fatal("expected concatenated shared-left matmul to run")
	}
	if len(got) != 3 {
		t.Fatalf("result count = %d, want 3", len(got))
	}
	for setIndex, rhsSet := range [][][]float32{rhsA, rhsB, rhsC} {
		want := make([]float32, 4)
		for i := range lhsMatrices {
			step := make([]float32, 4)
			fillHostMatMulTranspose(lhsMatrices[i], 2, 2, rhsSet[i], 2, 2, true, false, step)
			addFloat32Slice(want, step)
		}
		assertTensorClose(t, backend.NewTensorF32([]int{2, 2}, got[setIndex]), []int{2, 2}, want)
	}
}

func TestEmbeddingTrainerConcatenatedSharedLeftQKVGradCanBeDisabled(t *testing.T) {
	t.Setenv("EOS_TRAIN_DISABLE_CONCAT_SHARED_LEFT_MATMUL", "1")
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.005)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05

	if _, err := trainer.TrainContrastiveStep(tinyEncoderContrastiveDataset()); err != nil {
		t.Fatalf("train contrastive step: %v", err)
	}
	if fake.sharedLeftRuns == 0 {
		t.Fatal("expected shared-left backend fallback when concatenated path is disabled")
	}
	if fake.maxSharedLeftRHS < 3 {
		t.Fatalf("max shared-left rhs count = %d, want at least 3", fake.maxSharedLeftRHS)
	}
}

func TestEmbeddingTrainerCombinedAttentionVKGradMatchesSeparateMatMuls(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.005)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake

	attnScores := [][]float32{
		{
			0.2, 0.8,
			0.6, 0.4,
		},
		{
			0.9, 0.1,
			0.3, 0.7,
		},
	}
	gradMixed := [][]float32{
		{
			1, -2,
			3, 4,
		},
		{
			-1, 0.5,
			2, -3,
		},
	}
	gradPreSoftmax := [][]float32{
		{
			0.5, -1,
			2, 0.25,
		},
		{
			1.5, 0.75,
			-0.5, 3,
		},
	}
	attnQ := [][]float32{
		{
			2, 1,
			-1, 0.5,
		},
		{
			0.25, -2,
			1, 4,
		},
	}

	gotV, gotK, ok := trainer.tryCombinedAttentionValueKeyGradMatMul(attnScores, gradMixed, gradPreSoftmax, attnQ, 2, 2)
	if !ok {
		t.Fatal("expected combined V/K gradient matmul to run")
	}
	if fake.maxRunBatches != 4 {
		t.Fatalf("max run batches = %d, want 4 combined batches", fake.maxRunBatches)
	}
	for i := range attnScores {
		wantV := make([]float32, 4)
		fillHostMatMulTranspose(attnScores[i], 2, 2, gradMixed[i], 2, 2, true, false, wantV)
		assertTensorClose(t, backend.NewTensorF32([]int{2, 2}, gotV[i]), []int{2, 2}, wantV)

		wantK := make([]float32, 4)
		fillHostMatMulTranspose(gradPreSoftmax[i], 2, 2, attnQ[i], 2, 2, true, false, wantK)
		assertTensorClose(t, backend.NewTensorF32([]int{2, 2}, gotK[i]), []int{2, 2}, wantK)
	}
}

func TestEmbeddingTrainerCombinedAttentionVKGradCanBeDisabled(t *testing.T) {
	t.Setenv("EOS_TRAIN_DISABLE_COMBINED_ATTENTION_VK_GRAD", "1")
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.005)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	trainer.forwardMatMul = &countingMatMulAccelerator{}

	_, _, ok := trainer.tryCombinedAttentionValueKeyGradMatMul(
		[][]float32{{1, 0, 0, 1}},
		[][]float32{{1, 2, 3, 4}},
		[][]float32{{0.5, 0, 0, 0.5}},
		[][]float32{{2, 0, 0, 2}},
		2,
		2,
	)
	if ok {
		t.Fatal("expected combined V/K gradient matmul to be disabled")
	}
}

func TestEmbeddingTrainerAttentionBackwardUsesAccumulatedInputGradMatMul(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.005)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05

	if _, err := trainer.TrainContrastiveStep(tinyEncoderContrastiveDataset()); err != nil {
		t.Fatalf("train contrastive step: %v", err)
	}
	if fake.accumulatedRuns == 0 {
		t.Fatal("expected accumulated input-gradient bound-right matmul")
	}
	if fake.maxAccumTerms < 3 {
		t.Fatalf("max accumulated terms = %d, want q/k/v terms", fake.maxAccumTerms)
	}
}

func TestEmbeddingTrainerAccumulatedAttentionInputGradMatchesSeparateMatMuls(t *testing.T) {
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.005)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake

	attentionQuery := backend.NewTensorF32([]int{2, 2}, []float32{
		1, 2,
		3, 4,
	})
	attentionKey := backend.NewTensorF32([]int{2, 2}, []float32{
		-1, 0.5,
		2, -0.25,
	})
	attentionValue := backend.NewTensorF32([]int{2, 2}, []float32{
		0.25, 1.5,
		-2, 3,
	})
	if err := fake.BindMatrix(trainer.attnQParam.Name, attentionQuery); err != nil {
		t.Fatalf("bind q: %v", err)
	}
	if err := fake.BindMatrix(trainer.attnKParam.Name, attentionKey); err != nil {
		t.Fatalf("bind k: %v", err)
	}
	if err := fake.BindMatrix(trainer.attnVParam.Name, attentionValue); err != nil {
		t.Fatalf("bind v: %v", err)
	}

	gradQ := [][]float32{
		{
			1, -1,
			0.5, 2,
		},
		{
			-0.25, 3,
			1.5, -2,
		},
	}
	gradK := [][]float32{
		{
			2, 0.25,
			-1.5, 1,
		},
		{
			0.5, -0.5,
			3, 1,
		},
	}
	gradV := [][]float32{
		{
			-1, 4,
			2, -0.75,
		},
		{
			1, 2,
			-3, 0.25,
		},
	}

	got, ok := trainer.tryAccumulatedAttentionInputGradMatMul(gradQ, gradK, gradV, 2, 2, attentionQuery, attentionKey, attentionValue)
	if !ok {
		t.Fatal("expected accumulated attention input-gradient matmul to run")
	}
	if fake.accumulatedRuns != 1 {
		t.Fatalf("accumulated runs = %d, want 1", fake.accumulatedRuns)
	}
	for i := range gradQ {
		want := make([]float32, 4)
		step := make([]float32, 4)
		fillHostMatMulTranspose(gradQ[i], 2, 2, attentionQuery.F32, 2, 2, false, true, step)
		addFloat32Slice(want, step)
		for j := range step {
			step[j] = 0
		}
		fillHostMatMulTranspose(gradK[i], 2, 2, attentionKey.F32, 2, 2, false, true, step)
		addFloat32Slice(want, step)
		for j := range step {
			step[j] = 0
		}
		fillHostMatMulTranspose(gradV[i], 2, 2, attentionValue.F32, 2, 2, false, true, step)
		addFloat32Slice(want, step)
		assertTensorClose(t, backend.NewTensorF32([]int{2, 2}, got[i]), []int{2, 2}, want)
	}
}

func TestEmbeddingTrainerAccumulatedAttentionInputGradCanBeDisabled(t *testing.T) {
	t.Setenv("EOS_TRAIN_DISABLE_ACCUMULATED_ATTENTION_INPUT_GRAD", "1")
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.005)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake

	attentionQuery := backend.NewTensorF32([]int{2, 2}, []float32{1, 0, 0, 1})
	if err := fake.BindMatrix(trainer.attnQParam.Name, attentionQuery); err != nil {
		t.Fatalf("bind q: %v", err)
	}
	if err := fake.BindMatrix(trainer.attnKParam.Name, attentionQuery); err != nil {
		t.Fatalf("bind k: %v", err)
	}
	if err := fake.BindMatrix(trainer.attnVParam.Name, attentionQuery); err != nil {
		t.Fatalf("bind v: %v", err)
	}
	if _, ok := trainer.tryAccumulatedAttentionInputGradMatMul(
		[][]float32{{1, 2, 3, 4}},
		[][]float32{{1, 2, 3, 4}},
		[][]float32{{1, 2, 3, 4}},
		2,
		2,
		attentionQuery,
		attentionQuery,
		attentionQuery,
	); ok {
		t.Fatal("expected accumulated attention input-gradient matmul to be disabled")
	}
	if fake.accumulatedRuns != 0 {
		t.Fatalf("accumulated runs = %d, want disabled", fake.accumulatedRuns)
	}
}

func TestEmbeddingTrainerSharedLeftQKVGradMatMulCanBeDisabled(t *testing.T) {
	t.Setenv("EOS_TRAIN_DISABLE_SHARED_LEFT_MATMUL", "1")
	trainer := newTinyTrainableEncoderEmbeddingTrainer(t, 0.005)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	trainer.config.ContrastiveLoss = "infonce"
	trainer.config.Temperature = 0.05

	if _, err := trainer.TrainContrastiveStep(tinyEncoderContrastiveDataset()); err != nil {
		t.Fatalf("train contrastive step: %v", err)
	}
	if fake.sharedLeftRuns != 0 {
		t.Fatalf("shared-left matmul runs = %d, want disabled", fake.sharedLeftRuns)
	}
}

func TestEmbeddingTrainerBatchedForwardCanBeDisabled(t *testing.T) {
	t.Setenv("EOS_TRAIN_DISABLE_BATCHED_FORWARD", "1")
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	batch := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{0, 0}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 1}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{2, 0}, PositiveTokens: []int32{2, 2}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
	}

	forward := trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	queries, positives, err := trainer.encodeContrastiveBatch(batch, forward, true)
	if err != nil {
		t.Fatalf("encode contrastive batch: %v", err)
	}
	defer trainer.releaseEncodedSequences(queries)
	defer trainer.releaseEncodedSequences(positives)
	if fake.boundRightRuns == 0 {
		t.Fatal("expected regular per-sequence bound-right matmul calls")
	}
	if fake.maxBoundRightRows > 2 {
		t.Fatalf("max bound-right lhs rows = %d, want per-sequence rows", fake.maxBoundRightRows)
	}
}

func TestEmbeddingTrainerSequenceMatMulBindingsCanBeEnabled(t *testing.T) {
	t.Setenv("EOS_TRAIN_ENABLE_SEQUENCE_MATMUL_BINDINGS", "1")
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	batch := []EmbeddingContrastiveExample{
		{QueryTokens: []int32{0, 2}, PositiveTokens: []int32{0, 0}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 1}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{2, 0}, PositiveTokens: []int32{2, 2}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
	}

	forward := trainer.prepareForwardWeights()
	trainer.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	queries, positives, err := trainer.encodeContrastiveBatch(batch, forward, true)
	if err != nil {
		t.Fatalf("encode contrastive batch: %v", err)
	}
	defer trainer.releaseEncodedSequences(queries)
	defer trainer.releaseEncodedSequences(positives)
	if trainer.sequenceBindingID != 6 {
		t.Fatalf("sequence binding count = %d, want 6 encoded sequences", trainer.sequenceBindingID)
	}
	if fake.bindCalls <= 5 {
		t.Fatalf("bind calls = %d, want sequence matmul bindings beyond forward weights", fake.bindCalls)
	}
}

func TestEmbeddingTrainerWriteEmbeddingPackage(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_q8"
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"projection": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	}, EmbeddingTrainConfig{LearningRate: 0.05})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}
	for i := 0; i < 8; i++ {
		if _, err := trainer.TrainStep([]EmbeddingPairExample{
			{LeftTokens: []int32{0}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
			{LeftTokens: []int32{0}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
		}); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}

	packagePath := filepath.Join(t.TempDir(), "tiny_train_embed_q8.mll")
	paths, err := trainer.WriteEmbeddingPackage(packagePath)
	if err != nil {
		t.Fatalf("write embedding package: %v", err)
	}
	if paths.ArtifactPath != packagePath {
		t.Fatalf("artifact path = %q, want %q", paths.ArtifactPath, packagePath)
	}
	if paths.MemoryPlanPath != DefaultMemoryPlanPath(packagePath) {
		t.Fatalf("memory plan path = %q, want %q", paths.MemoryPlanPath, DefaultMemoryPlanPath(packagePath))
	}
	if paths.PackageManifestPath != DefaultPackageManifestPath(packagePath) {
		t.Fatalf("package manifest path = %q, want %q", paths.PackageManifestPath, DefaultPackageManifestPath(packagePath))
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), packagePath)
	if err != nil {
		t.Fatalf("load embedding package: %v", err)
	}
	result, err := model.Embed(context.Background(), []int32{0})
	if err != nil {
		t.Fatalf("embed from written package: %v", err)
	}
	if result.Embeddings == nil {
		t.Fatal("expected embeddings from written package")
	}
	if got := result.Embeddings.DType; got != "f16" {
		t.Fatalf("embedding dtype = %q, want f16", got)
	}
	if model.MemoryPlan() == nil {
		t.Fatal("expected memory plan on loaded model")
	}
}

func TestEmbeddingTrainerWriteTrainingPackageAndReload(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_q8"
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"projection": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	}, EmbeddingTrainConfig{LearningRate: 0.05})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}

	batch := []EmbeddingPairExample{
		{LeftTokens: []int32{0}, RightTokens: []int32{0}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: 1},
		{LeftTokens: []int32{0}, RightTokens: []int32{1}, LeftMask: []int32{1}, RightMask: []int32{1}, Target: -1},
	}
	for i := 0; i < 6; i++ {
		if _, err := trainer.TrainStep(batch); err != nil {
			t.Fatalf("train step %d: %v", i, err)
		}
	}

	path := filepath.Join(t.TempDir(), "tiny_train_embed_q8.mll")
	paths, err := trainer.WriteTrainingPackage(path)
	if err != nil {
		t.Fatalf("write training package: %v", err)
	}
	restored, err := LoadEmbeddingTrainerPackage(path)
	if err != nil {
		t.Fatalf("load training package: %v", err)
	}
	beforeA, err := trainer.EvaluatePairs(batch)
	if err != nil {
		t.Fatalf("eval original: %v", err)
	}
	beforeB, err := restored.EvaluatePairs(batch)
	if err != nil {
		t.Fatalf("eval restored: %v", err)
	}
	assertClose(t, beforeA.Loss, beforeB.Loss, 0.000001)
	assertClose(t, beforeA.ScoreMargin, beforeB.ScoreMargin, 0.000001)
	if paths.TrainManifestPath != DefaultEmbeddingTrainManifestPath(path) {
		t.Fatalf("train manifest path = %q, want %q", paths.TrainManifestPath, DefaultEmbeddingTrainManifestPath(path))
	}
	if paths.CheckpointPath != DefaultEmbeddingCheckpointPath(path) {
		t.Fatalf("checkpoint path = %q, want %q", paths.CheckpointPath, DefaultEmbeddingCheckpointPath(path))
	}
	if paths.MemoryPlanPath != DefaultMemoryPlanPath(path) {
		t.Fatalf("memory plan path = %q, want %q", paths.MemoryPlanPath, DefaultMemoryPlanPath(path))
	}
	if paths.TrainProfilePath != DefaultEmbeddingTrainProfilePath(path) {
		t.Fatalf("training profile path = %q, want %q", paths.TrainProfilePath, DefaultEmbeddingTrainProfilePath(path))
	}
	if paths.PackageManifestPath != DefaultPackageManifestPath(path) {
		t.Fatalf("package manifest path = %q, want %q", paths.PackageManifestPath, DefaultPackageManifestPath(path))
	}
	profile, err := ReadEmbeddingTrainProfileFile(paths.TrainProfilePath)
	if err != nil {
		t.Fatalf("read training profile: %v", err)
	}
	if profile.Step != trainer.step {
		t.Fatalf("training profile step = %d, want %d", profile.Step, trainer.step)
	}
	if restored.MemoryPlan() == nil {
		t.Fatal("expected restored trainer memory plan")
	}
}

func newTinyTrainableFFNEmbeddingTrainer(t *testing.T, learningRate float32) *EmbeddingTrainer {
	t.Helper()
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param ffn_up: q8[D, H] @weight("weights/ffn_up") @trainable
param projection: q8[H, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let ffn_up_f = dequant(ffn_up)
    let projection_f = dequant(projection)
    let ffn_hidden = @matmul(hidden, ffn_up_f)
    let activated = gelu(ffn_hidden)
    let projected = @matmul(activated, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let ffn_up_f = dequant(ffn_up)
    let projection_f = dequant(projection)
    let ffn_hidden = @matmul(hidden, ffn_up_f)
    let activated = gelu(ffn_hidden)
    let projected = @matmul(activated, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_ffn_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_ffn_q8"
	manifest.HiddenProjectionParam = "ffn_up"
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"ffn_up": backend.NewTensorF32([]int{2, 3}, []float32{
			1, 0, 1,
			0, 1, 1,
		}),
		"projection": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			0.5, 0.5,
		}),
	}, EmbeddingTrainConfig{LearningRate: learningRate})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}
	t.Cleanup(trainer.Close)
	return trainer
}

func newTinyTrainableAttentionEmbeddingTrainer(t *testing.T, learningRate float32) *EmbeddingTrainer {
	t.Helper()
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param attn_q: q8[D, D] @weight("weights/attn_q") @trainable
param attn_k: q8[D, D] @weight("weights/attn_k") @trainable
param attn_v: q8[D, D] @weight("weights/attn_v") @trainable
param attn_o: q8[D, D] @weight("weights/attn_o") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let wq_f = dequant(attn_q)
    let wk_f = dequant(attn_k)
    let wv_f = dequant(attn_v)
    let wo_f = dequant(attn_o)
    let projection_f = dequant(projection)
    let q = @matmul(hidden, wq_f)
    let k = @matmul(hidden, wk_f)
    let v = @matmul(hidden, wv_f)
    let kt = transpose(k)
    let scores = @matmul(q, kt)
    let probs = softmax(scores)
    let mixed = @matmul(probs, v)
    let attended = @matmul(mixed, wo_f)
    let projected = @matmul(attended, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let wq_f = dequant(attn_q)
    let wk_f = dequant(attn_k)
    let wv_f = dequant(attn_v)
    let wo_f = dequant(attn_o)
    let projection_f = dequant(projection)
    let q = @matmul(hidden, wq_f)
    let k = @matmul(hidden, wk_f)
    let v = @matmul(hidden, wv_f)
    let kt = transpose(k)
    let scores = @matmul(q, kt)
    let probs = softmax(scores)
    let mixed = @matmul(probs, v)
    let attended = @matmul(mixed, wo_f)
    let projected = @matmul(attended, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_attn_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_attn_q8"
	manifest.AttentionQueryParam = "attn_q"
	manifest.AttentionKeyParam = "attn_k"
	manifest.AttentionValueParam = "attn_v"
	manifest.AttentionOutputParam = "attn_o"
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"attn_q": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
		"attn_k": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
		"attn_v": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
		"attn_o": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
		"projection": backend.NewTensorF32([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	}, EmbeddingTrainConfig{LearningRate: learningRate})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}
	t.Cleanup(trainer.Close)
	return trainer
}

func newTinyTrainableEncoderEmbeddingTrainer(t *testing.T, learningRate float32) *EmbeddingTrainer {
	t.Helper()
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param attn_q: q8[D, D] @weight("weights/attn_q") @trainable
param attn_k: q8[D, D] @weight("weights/attn_k") @trainable
param attn_v: q8[D, D] @weight("weights/attn_v") @trainable
param attn_o: q8[D, D] @weight("weights/attn_o") @trainable
param ffn_up: q8[D, H] @weight("weights/ffn_up") @trainable
param projection: q8[H, D] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[D] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let wq_f = dequant(attn_q)
    let wk_f = dequant(attn_k)
    let wv_f = dequant(attn_v)
    let wo_f = dequant(attn_o)
    let ffn_up_f = dequant(ffn_up)
    let projection_f = dequant(projection)
    let q = @matmul(hidden, wq_f)
    let k = @matmul(hidden, wk_f)
    let v = @matmul(hidden, wv_f)
    let kt = transpose(k)
    let scores = @matmul(q, kt)
    let probs = softmax(scores)
    let mixed = @matmul(probs, v)
    let attended = @matmul(mixed, wo_f)
    let attn_hidden = layernorm(attended + hidden)
    let ffn_hidden = @matmul(attn_hidden, ffn_up_f)
    let activated = gelu(ffn_hidden)
    let projected = @matmul(activated, projection_f)
    let encoded = layernorm(projected + attn_hidden)
    let normalized = normalize(encoded)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, D] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let wq_f = dequant(attn_q)
    let wk_f = dequant(attn_k)
    let wv_f = dequant(attn_v)
    let wo_f = dequant(attn_o)
    let ffn_up_f = dequant(ffn_up)
    let projection_f = dequant(projection)
    let q = @matmul(hidden, wq_f)
    let k = @matmul(hidden, wk_f)
    let v = @matmul(hidden, wv_f)
    let kt = transpose(k)
    let scores = @matmul(q, kt)
    let probs = softmax(scores)
    let mixed = @matmul(probs, v)
    let attended = @matmul(mixed, wo_f)
    let attn_hidden = layernorm(attended + hidden)
    let ffn_hidden = @matmul(attn_hidden, ffn_up_f)
    let activated = gelu(ffn_hidden)
    let projected = @matmul(activated, projection_f)
    let encoded = layernorm(projected + attn_hidden)
    let normalized = normalize(encoded)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_encoder_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_encoder_q8"
	manifest.AttentionQueryParam = "attn_q"
	manifest.AttentionKeyParam = "attn_k"
	manifest.AttentionValueParam = "attn_v"
	manifest.AttentionOutputParam = "attn_o"
	manifest.AttentionResidual = true
	manifest.AttentionLayerNorm = true
	manifest.HiddenProjectionParam = "ffn_up"
	manifest.FFNResidual = true
	manifest.FFNLayerNorm = true
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 3}, []float32{
			0.9, 0.1, 0.2,
			0.2, 0.8, 0.3,
			0.6, 0.4, 0.7,
		}),
		"attn_q": backend.NewTensorF32([]int{3, 3}, []float32{
			0.9, 0.1, 0.0,
			0.1, 0.8, 0.1,
			0.0, 0.2, 0.9,
		}),
		"attn_k": backend.NewTensorF32([]int{3, 3}, []float32{
			0.8, 0.2, 0.1,
			0.0, 0.9, 0.1,
			0.1, 0.1, 0.8,
		}),
		"attn_v": backend.NewTensorF32([]int{3, 3}, []float32{
			0.7, 0.2, 0.1,
			0.2, 0.7, 0.2,
			0.1, 0.3, 0.8,
		}),
		"attn_o": backend.NewTensorF32([]int{3, 3}, []float32{
			0.6, 0.2, 0.2,
			0.2, 0.7, 0.1,
			0.1, 0.2, 0.7,
		}),
		"ffn_up": backend.NewTensorF32([]int{3, 4}, []float32{
			0.7, 0.1, 0.4, 0.2,
			0.2, 0.8, 0.5, 0.3,
			0.4, 0.3, 0.7, 0.6,
		}),
		"projection": backend.NewTensorF32([]int{4, 3}, []float32{
			0.6, 0.2, 0.2,
			0.3, 0.5, 0.2,
			0.2, 0.3, 0.5,
			0.4, 0.1, 0.5,
		}),
	}, EmbeddingTrainConfig{LearningRate: learningRate})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}
	t.Cleanup(trainer.Close)
	return trainer
}

func newTinyTrainableRepeatedEncoderEmbeddingTrainer(t *testing.T, learningRate float32) *EmbeddingTrainer {
	t.Helper()
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param attn_q: q8[D, D] @weight("weights/attn_q") @trainable
param attn_k: q8[D, D] @weight("weights/attn_k") @trainable
param attn_v: q8[D, D] @weight("weights/attn_v") @trainable
param attn_o: q8[D, D] @weight("weights/attn_o") @trainable
param ffn_up: q8[D, H] @weight("weights/ffn_up") @trainable
param projection: q8[H, D] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[D] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let wq_f = dequant(attn_q)
    let wk_f = dequant(attn_k)
    let wv_f = dequant(attn_v)
    let wo_f = dequant(attn_o)
    let ffn_up_f = dequant(ffn_up)
    let projection_f = dequant(projection)

    let q1 = @matmul(hidden, wq_f)
    let k1 = @matmul(hidden, wk_f)
    let v1 = @matmul(hidden, wv_f)
    let kt1 = transpose(k1)
    let scores1 = @matmul(q1, kt1)
    let probs1 = softmax(scores1)
    let mixed1 = @matmul(probs1, v1)
    let attended1 = @matmul(mixed1, wo_f)
    let attn_hidden1 = layernorm(attended1 + hidden)
    let ffn_hidden1 = @matmul(attn_hidden1, ffn_up_f)
    let activated1 = gelu(ffn_hidden1)
    let projected1 = @matmul(activated1, projection_f)
    let encoded1 = layernorm(projected1 + attn_hidden1)

    let q2 = @matmul(encoded1, wq_f)
    let k2 = @matmul(encoded1, wk_f)
    let v2 = @matmul(encoded1, wv_f)
    let kt2 = transpose(k2)
    let scores2 = @matmul(q2, kt2)
    let probs2 = softmax(scores2)
    let mixed2 = @matmul(probs2, v2)
    let attended2 = @matmul(mixed2, wo_f)
    let attn_hidden2 = layernorm(attended2 + encoded1)
    let ffn_hidden2 = @matmul(attn_hidden2, ffn_up_f)
    let activated2 = gelu(ffn_hidden2)
    let projected2 = @matmul(activated2, projection_f)
    let encoded2 = layernorm(projected2 + attn_hidden2)
    let normalized = normalize(encoded2)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, D] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let wq_f = dequant(attn_q)
    let wk_f = dequant(attn_k)
    let wv_f = dequant(attn_v)
    let wo_f = dequant(attn_o)
    let ffn_up_f = dequant(ffn_up)
    let projection_f = dequant(projection)

    let q1 = @matmul(hidden, wq_f)
    let k1 = @matmul(hidden, wk_f)
    let v1 = @matmul(hidden, wv_f)
    let kt1 = transpose(k1)
    let scores1 = @matmul(q1, kt1)
    let probs1 = softmax(scores1)
    let mixed1 = @matmul(probs1, v1)
    let attended1 = @matmul(mixed1, wo_f)
    let attn_hidden1 = layernorm(attended1 + hidden)
    let ffn_hidden1 = @matmul(attn_hidden1, ffn_up_f)
    let activated1 = gelu(ffn_hidden1)
    let projected1 = @matmul(activated1, projection_f)
    let encoded1 = layernorm(projected1 + attn_hidden1)

    let q2 = @matmul(encoded1, wq_f)
    let k2 = @matmul(encoded1, wk_f)
    let v2 = @matmul(encoded1, wv_f)
    let kt2 = transpose(k2)
    let scores2 = @matmul(q2, kt2)
    let probs2 = softmax(scores2)
    let mixed2 = @matmul(probs2, v2)
    let attended2 = @matmul(mixed2, wo_f)
    let attn_hidden2 = layernorm(attended2 + encoded1)
    let ffn_hidden2 = @matmul(attn_hidden2, ffn_up_f)
    let activated2 = gelu(ffn_hidden2)
    let projected2 = @matmul(activated2, projection_f)
    let encoded2 = layernorm(projected2 + attn_hidden2)
    let normalized = normalize(encoded2)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_encoder_q8x2"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_train_embed_encoder_q8x2"
	manifest.AttentionQueryParam = "attn_q"
	manifest.AttentionKeyParam = "attn_k"
	manifest.AttentionValueParam = "attn_v"
	manifest.AttentionOutputParam = "attn_o"
	manifest.AttentionResidual = true
	manifest.AttentionLayerNorm = true
	manifest.HiddenProjectionParam = "ffn_up"
	manifest.FFNResidual = true
	manifest.FFNLayerNorm = true
	manifest.EncoderRepeats = 2
	trainer, err := NewEmbeddingTrainer(bundle.Artifact, manifest, map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{3, 3}, []float32{
			0.9, 0.1, 0.2,
			0.2, 0.8, 0.3,
			0.6, 0.4, 0.7,
		}),
		"attn_q": backend.NewTensorF32([]int{3, 3}, []float32{
			0.9, 0.1, 0.0,
			0.1, 0.8, 0.1,
			0.0, 0.2, 0.9,
		}),
		"attn_k": backend.NewTensorF32([]int{3, 3}, []float32{
			0.8, 0.2, 0.1,
			0.0, 0.9, 0.1,
			0.1, 0.2, 0.8,
		}),
		"attn_v": backend.NewTensorF32([]int{3, 3}, []float32{
			0.7, 0.1, 0.2,
			0.2, 0.7, 0.1,
			0.1, 0.3, 0.8,
		}),
		"attn_o": backend.NewTensorF32([]int{3, 3}, []float32{
			0.8, 0.0, 0.2,
			0.1, 0.9, 0.0,
			0.2, 0.1, 0.8,
		}),
		"ffn_up": backend.NewTensorF32([]int{3, 4}, []float32{
			0.6, 0.2, 0.1, 0.3,
			0.1, 0.7, 0.2, 0.2,
			0.2, 0.1, 0.8, 0.4,
		}),
		"projection": backend.NewTensorF32([]int{4, 3}, []float32{
			0.7, 0.1, 0.2,
			0.2, 0.6, 0.1,
			0.1, 0.2, 0.8,
			0.3, 0.1, 0.5,
		}),
	}, EmbeddingTrainConfig{LearningRate: learningRate})
	if err != nil {
		t.Fatalf("new repeated encoder trainer: %v", err)
	}
	t.Cleanup(trainer.Close)
	return trainer
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

func trainerMasterTensorByName(trainer *EmbeddingTrainer, name string) *backend.Tensor {
	switch name {
	case "token_embedding":
		return trainer.tokenEmbed
	case "attn_q":
		return trainer.attentionQuery
	case "attn_k":
		return trainer.attentionKey
	case "attn_v":
		return trainer.attentionValue
	case "attn_o":
		return trainer.attentionOutput
	case "ffn_up":
		return trainer.hiddenProjection
	case "projection":
		return trainer.projection
	default:
		return nil
	}
}
