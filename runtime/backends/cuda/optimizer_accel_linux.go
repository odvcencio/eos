//go:build linux && cgo

package cuda

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#include <cuda.h>
*/
import "C"

import (
	"fmt"
	"math"
	"sync"
	"time"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

const optimizerKernelSource = `
extern "C" __global__ void manta_optimizer_update(
    float* param,
    float* mom1,
    float* mom2,
    const float* grad,
    int elements,
    int mode,
    float learning_rate,
    float weight_decay,
    float beta1,
    float beta2,
    float corr1,
    float corr2,
    float epsilon,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    float g = grad[idx] * scale;
    if (weight_decay != 0.0f) {
        g += weight_decay * param[idx];
    }
    if (mode == 0) {
        param[idx] -= learning_rate * g;
        return;
    }
    float m = beta1 * mom1[idx] + (1.0f - beta1) * g;
    float v = beta2 * mom2[idx] + (1.0f - beta2) * g * g;
    mom1[idx] = m;
    mom2[idx] = v;
    float m_hat = corr1 != 0.0f ? (m / corr1) : m;
    float v_hat = corr2 != 0.0f ? (v / corr2) : v;
    param[idx] -= learning_rate * (m_hat / (sqrtf(v_hat) + epsilon));
}
`

type optimizerAccelerator struct {
	mu              sync.RWMutex
	device          *deviceRuntime
	kernel          *auxKernel
	resident        map[string]residentOptimizerState
	stats           backend.OptimizerAcceleratorStats
	nextGen         uint64
	lastLogicalStep int
}

type residentOptimizerState struct {
	param      C.CUdeviceptr
	mom1       C.CUdeviceptr
	mom2       C.CUdeviceptr
	elements   int
	hasMoments bool
	generation uint64
	token      *optimizerResidentParameterToken
}

type optimizerResidentParameterToken struct {
	owner      *optimizerAccelerator
	name       string
	generation uint64
}

func (t *optimizerResidentParameterToken) OptimizerResidentParameterToken() {}

func (t *optimizerResidentParameterToken) CompactForwardResidentToken() {}

func (t *optimizerResidentParameterToken) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (t *optimizerResidentParameterToken) Generation() uint64 {
	if t == nil {
		return 0
	}
	return t.generation
}

func (t *optimizerResidentParameterToken) Alive() bool {
	if t == nil {
		return false
	}
	_, unlock, err := t.lockCurrent()
	if unlock != nil {
		unlock()
	}
	return err == nil
}

func (t *optimizerResidentParameterToken) lockCurrent() (residentOptimizerState, func(), error) {
	if t == nil || t.owner == nil || t.name == "" {
		return residentOptimizerState{}, nil, fmt.Errorf("cuda optimizer resident token is invalid")
	}
	t.owner.mu.RLock()
	unlock := func() { t.owner.mu.RUnlock() }
	if t.owner.device == nil {
		unlock()
		return residentOptimizerState{}, nil, fmt.Errorf("cuda optimizer resident token %q owner is closed", t.name)
	}
	state, ok := t.owner.resident[t.name]
	if !ok {
		unlock()
		return residentOptimizerState{}, nil, fmt.Errorf("cuda optimizer resident token %q is no longer resident", t.name)
	}
	if state.generation != t.generation {
		unlock()
		return residentOptimizerState{}, nil, fmt.Errorf("cuda optimizer resident token %q generation %d is stale, current %d", t.name, t.generation, state.generation)
	}
	if state.token != t {
		unlock()
		return residentOptimizerState{}, nil, fmt.Errorf("cuda optimizer resident token %q is stale", t.name)
	}
	return state, unlock, nil
}

func init() {
	backend.RegisterOptimizerAccelerator(eosartifact.BackendCUDA, NewOptimizerAccelerator)
}

func NewOptimizerAccelerator() (backend.OptimizerAccelerator, error) {
	device, err := newDeviceRuntime()
	if err != nil {
		return nil, err
	}
	if device == nil {
		return nil, nil
	}
	kernel, err := device.compileAuxKernel(optimizerKernelSource, "manta_optimizer_update")
	if err != nil {
		device.close()
		return nil, err
	}
	return &optimizerAccelerator{device: device, kernel: kernel, resident: map[string]residentOptimizerState{}}, nil
}

func (a *optimizerAccelerator) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (a *optimizerAccelerator) Stats() backend.OptimizerAcceleratorStats {
	if a == nil {
		return backend.OptimizerAcceleratorStats{}
	}
	a.mu.RLock()
	defer a.mu.RUnlock()
	stats := a.stats
	if stats.TensorUpdateCalls == 0 {
		stats.TensorUpdateCalls = stats.UpdateCalls
	}
	stats.ResidentParams = int64(len(a.resident))
	stats.CompactTrainTelemetry = backend.CloneCompactTrainTelemetry(a.stats.CompactTrainTelemetry)
	return stats
}

func (a *optimizerAccelerator) addCompactTrainTelemetry(phase backend.CompactTrainTelemetryPhase, delta backend.CompactTrainPhaseTelemetry) {
	if a == nil || !eosCudaCompactProfileEventsEnabled {
		return
	}
	if a.stats.CompactTrainTelemetry == nil {
		a.stats.CompactTrainTelemetry = &backend.CompactTrainTelemetry{}
	}
	a.stats.CompactTrainTelemetry.Enabled = true
	if a.stats.CompactTrainTelemetry.View == "" {
		a.stats.CompactTrainTelemetry.View = backend.CompactTrainTelemetryViewOptimizer
	}
	backend.AddCompactTrainTelemetryPhase(a.stats.CompactTrainTelemetry, phase, delta)
}

// compactTrainK5WorkloadTelemetry is the workload-only accounting boundary
// for the existing native optimizer batch bridge. It intentionally does not
// include event-control calls; callers add those diagnostics separately.
func compactTrainK5WorkloadTelemetry(progress optimizerUpdateBatchProgress) backend.CompactTrainPhaseTelemetry {
	phase := backend.CompactTrainPhaseTelemetry{
		HostDescriptorAllocs: boolInt64(progress.DescriptorAllocated),
		Attempted:            int64(progress.Attempted),
		Enqueued:             int64(progress.Enqueued),
		KernelLaunches:       int64(progress.Enqueued),
	}
	if !progress.Called {
		if progress.Err != nil {
			phase.Failures = 1
			phase.FailureStage = progress.FailureStage
		}
		if phase.FailureStage == "" {
			phase.FailureStage = progress.FailureStage
		}
		return phase
	}
	phase.GoCalls = 1
	// ContextSets is the attempted cuCtxSetCurrent count. A context failure is
	// carried by FailureStage; successful completion is the success signal.
	phase.ContextSets = 1
	phase.DriverCalls = int64(progress.Attempted)
	if progress.FailureStage != "context" {
		// The bridge's existing completion barrier is one driver-side
		// synchronization attempt after the launch prefix.
		phase.DriverCalls++
		phase.StreamSynchronizes = 1
	}
	if progress.Err == nil {
		phase.Completed = int64(progress.Enqueued)
	} else {
		phase.Failures = 1
		phase.FailureStage = progress.FailureStage
	}
	if progress.FailIndex >= 0 {
		phase.FailIndex = int64(progress.FailIndex)
		phase.HasFailIndex = true
	}
	return phase
}

func (a *optimizerAccelerator) ApplyUpdate(name string, cfg backend.OptimizerUpdateConfig, tensor, mom1, mom2, grad *backend.Tensor) error {
	if a == nil {
		return fmt.Errorf("cuda optimizer accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	validated, err := a.validateOptimizerUpdateLocked(name, cfg, tensor, mom1, mom2, grad, false)
	if err != nil {
		return err
	}
	elements := validated.elements
	if elements == 0 {
		return nil
	}
	start := time.Now()
	gradBuf, err := a.device.uploadFloat32(grad.F32)
	if err != nil {
		return err
	}
	defer a.device.freeBuffer(gradBuf)
	a.stats.UploadedBytes += int64(len(grad.F32) * 4)
	state, transient, err := a.ensureResidentState(name, tensor, mom1, mom2, elements, validated.mode == 1)
	if err != nil {
		return err
	}
	if transient {
		defer a.releaseResidentState(state)
	}

	corr1 := float32(1)
	corr2 := float32(1)
	if validated.mode == 1 {
		corr1 -= float32(math.Pow(float64(cfg.Beta1), float64(cfg.Step)))
		corr2 -= float32(math.Pow(float64(cfg.Beta2), float64(cfg.Step)))
	}
	block := 128
	grid := (elements + block - 1) / block
	if err := a.device.launchOptimizerUpdate(
		a.kernel,
		uint(grid),
		uint(block),
		state.param,
		state.mom1,
		state.mom2,
		gradBuf,
		elements,
		validated.mode,
		cfg.LearningRate,
		cfg.WeightDecay,
		cfg.Beta1,
		cfg.Beta2,
		corr1,
		corr2,
		cfg.Epsilon,
		cfg.Scale,
	); err != nil {
		return err
	}
	a.stats.UpdateCalls++
	a.stats.TensorUpdateCalls++
	if cfg.Step != 0 && cfg.Step != a.lastLogicalStep {
		a.stats.LogicalSteps++
		a.lastLogicalStep = cfg.Step
	}
	if cfg.DeferSync && name != "" {
		a.stats.DeferredSyncUpdates++
	} else {
		if err := a.device.downloadFloat32(tensor.F32, state.param); err != nil {
			return err
		}
		a.stats.DownloadedBytes += int64(len(tensor.F32) * 4)
	}
	a.stats.UpdateNanos += time.Since(start).Nanoseconds()
	a.stats.ResidentParams = int64(len(a.resident))
	a.updatePerStepStats()
	return nil
}

type optimizerUpdateValidation struct {
	elements int
	mode     int
}

func (a *optimizerAccelerator) PreflightApplyUpdate(name string, cfg backend.OptimizerUpdateConfig, tensor, mom1, mom2, grad *backend.Tensor) error {
	if a == nil {
		return fmt.Errorf("cuda optimizer accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	_, err := a.validateOptimizerUpdateLocked(name, cfg, tensor, mom1, mom2, grad, true)
	return err
}

func (a *optimizerAccelerator) validateOptimizerUpdateLocked(name string, cfg backend.OptimizerUpdateConfig, tensor, mom1, mom2, grad *backend.Tensor, preflight bool) (optimizerUpdateValidation, error) {
	if a.device == nil || a.kernel == nil {
		return optimizerUpdateValidation{}, fmt.Errorf("cuda optimizer accelerator is not initialized")
	}
	if tensor == nil || grad == nil {
		return optimizerUpdateValidation{}, fmt.Errorf("cuda optimizer update requires tensor and grad")
	}
	elements := len(tensor.F32)
	if elements == 0 {
		return optimizerUpdateValidation{}, nil
	}
	if len(grad.F32) != elements {
		return optimizerUpdateValidation{}, fmt.Errorf("cuda optimizer grad size %d does not match tensor size %d", len(grad.F32), elements)
	}
	mode := 1
	switch cfg.Optimizer {
	case "", "adamw":
		if mom1 == nil || mom2 == nil {
			return optimizerUpdateValidation{}, fmt.Errorf("cuda adamw update requires first and second moment tensors")
		}
	case "sgd":
		mode = 0
	default:
		return optimizerUpdateValidation{}, fmt.Errorf("cuda optimizer update does not support %q", cfg.Optimizer)
	}
	if mom1 != nil && len(mom1.F32) != elements {
		return optimizerUpdateValidation{}, fmt.Errorf("cuda optimizer first moment size %d does not match tensor size %d", len(mom1.F32), elements)
	}
	if mom2 != nil && len(mom2.F32) != elements {
		return optimizerUpdateValidation{}, fmt.Errorf("cuda optimizer second moment size %d does not match tensor size %d", len(mom2.F32), elements)
	}
	if preflight && cfg.DeferSync && name != "" {
		if state, ok := a.resident[name]; ok && (state.elements != elements || state.hasMoments != (mode == 1)) {
			return optimizerUpdateValidation{}, fmt.Errorf("cuda optimizer state %q metadata mismatch", name)
		}
	}
	return optimizerUpdateValidation{elements: elements, mode: mode}, nil
}

type residentGradientUpdateValidation struct {
	elements     int
	mode         int
	state        residentOptimizerState
	residentGrad *compactTrainGradient
	gradOwner    *CompactTrainAccelerator
}

// residentGradientTokenOwner resolves the compact owner without taking its
// mutex. The token's owner pointer is immutable; all mutable validation is
// performed after the caller acquires the owner lock in optimizer-mutex order.
func residentGradientTokenOwner(grad backend.ResidentGradientRef) (*compactTrainGradientToken, *CompactTrainAccelerator, error) {
	gradToken, ok := grad.Token.(*compactTrainGradientToken)
	if !ok || gradToken == nil {
		return nil, nil, fmt.Errorf("cuda optimizer resident gradient %q has invalid cuda token", grad.Name)
	}
	if gradToken.Backend() != eosartifact.BackendCUDA {
		return nil, nil, fmt.Errorf("cuda optimizer resident gradient %q token backend %q, want cuda", grad.Name, gradToken.Backend())
	}
	gradOwner := gradToken.owner
	if gradOwner == nil {
		return nil, nil, fmt.Errorf("cuda optimizer resident gradient %q owner is nil", grad.Name)
	}
	return gradToken, gradOwner, nil
}

// validateResidentGradientUpdateLocked is the single validation path for
// scalar preflight and apply. a.mu must be held. On success it also returns
// with the gradient owner's mutex held so the validated allocation cannot
// change before apply launches (or preflight returns).
func (a *optimizerAccelerator) validateResidentGradientUpdateLocked(name string, cfg backend.OptimizerUpdateConfig, tensor, mom1, mom2 *backend.Tensor, grad backend.ResidentGradientRef) (*residentGradientUpdateValidation, error) {
	if a.device == nil || a.kernel == nil {
		return nil, fmt.Errorf("cuda optimizer accelerator is not initialized")
	}
	if name == "" {
		return nil, fmt.Errorf("cuda optimizer resident-gradient update requires a parameter name")
	}
	if tensor == nil {
		return nil, fmt.Errorf("cuda optimizer resident-gradient update %q requires tensor", name)
	}
	elements := len(tensor.F32)
	if elements == 0 {
		return &residentGradientUpdateValidation{}, nil
	}
	gradToken, gradOwner, err := residentGradientTokenOwner(grad)
	if err != nil {
		return nil, err
	}
	gradOwner.mu.Lock()
	validated, err := a.validateResidentGradientUpdateWithOwnerLocked(name, cfg, tensor, mom1, mom2, grad, elements, gradToken, gradOwner)
	if err != nil {
		gradOwner.mu.Unlock()
		return nil, err
	}
	return validated, nil
}

// validateResidentGradientUpdateWithOwnerLocked performs the mutable portion
// of resident-gradient validation. a.mu and gradOwner.mu must both be held;
// the owner mutex remains held on success.
func (a *optimizerAccelerator) validateResidentGradientUpdateWithOwnerLocked(name string, cfg backend.OptimizerUpdateConfig, tensor, mom1, mom2 *backend.Tensor, grad backend.ResidentGradientRef, elements int, gradToken *compactTrainGradientToken, gradOwner *CompactTrainAccelerator) (*residentGradientUpdateValidation, error) {
	if grad.Name != name {
		return nil, fmt.Errorf("cuda optimizer resident gradient name %q does not match parameter %q", grad.Name, name)
	}
	if grad.Backend != eosartifact.BackendCUDA {
		return nil, fmt.Errorf("cuda optimizer resident gradient %q backend %q, want cuda", grad.Name, grad.Backend)
	}
	if grad.Elements != elements {
		return nil, fmt.Errorf("cuda optimizer resident gradient %q elements %d does not match tensor size %d", grad.Name, grad.Elements, elements)
	}
	mode := 1
	switch cfg.Optimizer {
	case "", "adamw":
		if mom1 == nil || mom2 == nil {
			return nil, fmt.Errorf("cuda adamw resident-gradient update requires first and second moment tensors")
		}
	case "sgd":
		mode = 0
	default:
		return nil, fmt.Errorf("cuda optimizer resident-gradient update does not support %q", cfg.Optimizer)
	}
	if mom1 != nil && len(mom1.F32) != elements {
		return nil, fmt.Errorf("cuda optimizer first moment size %d does not match tensor size %d", len(mom1.F32), elements)
	}
	if mom2 != nil && len(mom2.F32) != elements {
		return nil, fmt.Errorf("cuda optimizer second moment size %d does not match tensor size %d", len(mom2.F32), elements)
	}
	state, ok := a.resident[name]
	if !ok {
		return nil, fmt.Errorf("cuda optimizer state %q is not resident", name)
	}
	if state.elements != elements || state.hasMoments != (mode == 1) {
		return nil, fmt.Errorf("cuda optimizer state %q metadata mismatch", name)
	}
	if gradOwner.closed || gradOwner.device == nil {
		return nil, fmt.Errorf("cuda optimizer resident gradient %q owner is closed", grad.Name)
	}
	if !gradOwner.stepSealed || gradOwner.stepActive {
		return nil, fmt.Errorf("cuda optimizer resident gradient %q step %d is not sealed", grad.Name, grad.StepID)
	}
	if gradOwner.stepPoisoned {
		return nil, fmt.Errorf("cuda optimizer resident gradient %q step %d is poisoned", grad.Name, grad.StepID)
	}
	if gradToken.owner != gradOwner || gradToken.name != grad.Name {
		return nil, fmt.Errorf("cuda optimizer resident gradient %q owner/name mismatch", grad.Name)
	}
	if gradToken.generation != grad.Generation || gradToken.stepID != grad.StepID || gradToken.elements != grad.Elements {
		return nil, fmt.Errorf("cuda optimizer resident gradient %q token metadata mismatch", grad.Name)
	}
	residentGrad := gradOwner.grads[grad.Name]
	if residentGrad == nil || residentGrad.token != gradToken || residentGrad.ptr == 0 {
		return nil, fmt.Errorf("cuda optimizer resident gradient %q is stale", grad.Name)
	}
	if residentGrad.generation != grad.Generation || residentGrad.stepID != grad.StepID || residentGrad.elements != grad.Elements {
		return nil, fmt.Errorf("cuda optimizer resident gradient %q generation/elements mismatch", grad.Name)
	}
	if residentGrad.optimizerOwner != a {
		return nil, fmt.Errorf("cuda optimizer resident gradient %q belongs to a different optimizer", grad.Name)
	}
	if residentGrad.optimizerToken == nil || residentGrad.optimizerToken != state.token {
		return nil, fmt.Errorf("cuda optimizer resident gradient %q optimizer token is stale", grad.Name)
	}
	if residentGrad.optimizerGeneration != state.generation {
		return nil, fmt.Errorf("cuda optimizer resident gradient %q optimizer generation %d is stale, current %d", grad.Name, residentGrad.optimizerGeneration, state.generation)
	}
	if residentGrad.optimizerParam != state.param {
		return nil, fmt.Errorf("cuda optimizer resident gradient %q optimizer parameter pointer is stale", grad.Name)
	}
	if residentGrad.optimizerUsed {
		return nil, fmt.Errorf("cuda optimizer resident gradient %q was already used for step %d", grad.Name, grad.StepID)
	}
	return &residentGradientUpdateValidation{elements: elements, mode: mode, state: state, residentGrad: residentGrad, gradOwner: gradOwner}, nil
}

func (a *optimizerAccelerator) PreflightApplyUpdateWithResidentGrad(name string, cfg backend.OptimizerUpdateConfig, tensor, mom1, mom2 *backend.Tensor, grad backend.ResidentGradientRef) error {
	if a == nil {
		return fmt.Errorf("cuda optimizer accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	validated, err := a.validateResidentGradientUpdateLocked(name, cfg, tensor, mom1, mom2, grad)
	if err != nil {
		return err
	}
	if validated.gradOwner != nil {
		validated.gradOwner.mu.Unlock()
	}
	return nil
}

// validateResidentGradientBatchLocked validates the complete batch while
// holding a.mu. It resolves and compares owners before taking any owner lock,
// then takes exactly one owner lock and leaves it held on success.
func (a *optimizerAccelerator) validateResidentGradientBatchLocked(updates []backend.ResidentGradientOptimizerBatchUpdate) ([]*residentGradientUpdateValidation, *CompactTrainAccelerator, error) {
	if a.device == nil || a.kernel == nil {
		return nil, nil, fmt.Errorf("cuda optimizer accelerator is not initialized")
	}
	if len(updates) == 0 {
		return nil, nil, nil
	}
	validated := make([]*residentGradientUpdateValidation, len(updates))
	tokens := make([]*compactTrainGradientToken, len(updates))
	owner := (*CompactTrainAccelerator)(nil)
	seen := make(map[string]struct{}, len(updates))
	for i, update := range updates {
		if update.Name == "" {
			return nil, nil, fmt.Errorf("cuda optimizer resident-gradient batch update %d requires a parameter name", i)
		}
		if _, exists := seen[update.Name]; exists {
			return nil, nil, fmt.Errorf("cuda optimizer resident-gradient batch has duplicate parameter %q", update.Name)
		}
		seen[update.Name] = struct{}{}
		if update.Tensor == nil {
			return nil, nil, fmt.Errorf("cuda optimizer resident-gradient update %q requires tensor", update.Name)
		}
		if len(update.Tensor.F32) == 0 {
			continue
		}
		token, candidate, err := residentGradientTokenOwner(update.Grad)
		if err != nil {
			return nil, nil, err
		}
		if owner == nil {
			owner = candidate
		} else if owner != candidate {
			return nil, nil, fmt.Errorf("cuda optimizer resident-gradient batch mixes gradient owners")
		}
		tokens[i] = token
	}
	if owner == nil {
		for i := range validated {
			validated[i] = &residentGradientUpdateValidation{}
		}
		return validated, nil, nil
	}
	owner.mu.Lock()
	for i, update := range updates {
		if len(update.Tensor.F32) == 0 {
			validated[i] = &residentGradientUpdateValidation{}
			continue
		}
		item, err := a.validateResidentGradientUpdateWithOwnerLocked(update.Name, update.Config, update.Tensor, update.Mom1, update.Mom2, update.Grad, len(update.Tensor.F32), tokens[i], owner)
		if err != nil {
			owner.mu.Unlock()
			return nil, nil, err
		}
		validated[i] = item
	}
	return validated, owner, nil
}

func (a *optimizerAccelerator) PreflightApplyUpdateWithResidentGradBatch(updates []backend.ResidentGradientOptimizerBatchUpdate) error {
	if a == nil {
		return fmt.Errorf("cuda optimizer accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	_, owner, err := a.validateResidentGradientBatchLocked(updates)
	if owner != nil {
		owner.mu.Unlock()
	}
	return err
}

func (a *optimizerAccelerator) ApplyUpdateWithResidentGrad(name string, cfg backend.OptimizerUpdateConfig, tensor, mom1, mom2 *backend.Tensor, grad backend.ResidentGradientRef) error {
	if a == nil {
		return fmt.Errorf("cuda optimizer accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	validated, err := a.validateResidentGradientUpdateLocked(name, cfg, tensor, mom1, mom2, grad)
	if err != nil {
		return err
	}
	if validated.gradOwner != nil {
		defer validated.gradOwner.mu.Unlock()
	}
	if validated.elements == 0 {
		return nil
	}
	elements := validated.elements
	mode := validated.mode
	state := validated.state
	residentGrad := validated.residentGrad
	gradOwner := validated.gradOwner
	start := time.Now()
	corr1 := float32(1)
	corr2 := float32(1)
	if mode == 1 {
		corr1 -= float32(math.Pow(float64(cfg.Beta1), float64(cfg.Step)))
		corr2 -= float32(math.Pow(float64(cfg.Beta2), float64(cfg.Step)))
	}
	block := 128
	grid := (elements + block - 1) / block
	if err := a.device.launchOptimizerUpdate(
		a.kernel,
		uint(grid),
		uint(block),
		state.param,
		state.mom1,
		state.mom2,
		residentGrad.ptr,
		elements,
		mode,
		cfg.LearningRate,
		cfg.WeightDecay,
		cfg.Beta1,
		cfg.Beta2,
		corr1,
		corr2,
		cfg.Epsilon,
		cfg.Scale,
	); err != nil {
		return err
	}
	residentGrad.optimizerUsed = true
	a.stats.UpdateCalls++
	a.stats.TensorUpdateCalls++
	a.stats.ResidentGradUpdateCalls++
	avoided := int64(elements * 4)
	a.stats.ResidentGradUploadBytesAvoided += avoided
	gradOwner.stats.HostGradUploadBytesAvoided += avoided
	if cfg.Step != 0 && cfg.Step != a.lastLogicalStep {
		a.stats.LogicalSteps++
		a.lastLogicalStep = cfg.Step
	}
	if cfg.DeferSync && name != "" {
		a.stats.DeferredSyncUpdates++
	} else {
		if err := a.device.downloadFloat32(tensor.F32, state.param); err != nil {
			return err
		}
		a.stats.DownloadedBytes += int64(len(tensor.F32) * 4)
	}
	elapsed := time.Since(start).Nanoseconds()
	a.stats.UpdateNanos += elapsed
	a.stats.ResidentGradUpdateNanos += elapsed
	gradOwner.stats.OptimizerResidentGradNanos += elapsed
	a.stats.ResidentParams = int64(len(a.resident))
	a.updatePerStepStats()
	return nil
}

// poisonResidentGradientBatchLocked drains the optimizer stream best-effort
// and makes the compact owner unusable until its step is aborted. It is called
// with a.mu and owner.mu held after batch preflight, so no caller can recycle
// the gradient slab while the drain is in progress.
func (a *optimizerAccelerator) poisonResidentGradientBatchLocked(owner *CompactTrainAccelerator, err error) error {
	if a != nil && a.device != nil {
		if drainErr := a.device.synchronize(); drainErr != nil && err != nil {
			err = fmt.Errorf("%w; cuda optimizer batch drain: %v", err, drainErr)
		}
	}
	if owner != nil {
		owner.stepPoisoned = true
		// Keep the sealed lifecycle bit set so the trainer's deferred abort can
		// recycle the slab and invalidate every old token. Release still rejects
		// the poisoned step, while AbortCompactTrainStep accepts it for cleanup.
		owner.stepSealed = true
	}
	return err
}

// commitResidentGradientBatchDeviceWork records all work that is known to have
// completed once the native batch enqueue and barrier succeed. It must run
// before any immediate host readback: a later D2H failure cannot undo device
// updates, optimizer-used marks, or their timing/accounting.
func (a *optimizerAccelerator) commitResidentGradientBatchDeviceWork(updates []backend.ResidentGradientOptimizerBatchUpdate, validated []*residentGradientUpdateValidation, owner *CompactTrainAccelerator, kernelLaunches int, elapsed int64) {
	for i, update := range updates {
		item := validated[i]
		if item == nil || item.elements == 0 {
			continue
		}
		item.residentGrad.optimizerUsed = true
		a.stats.UpdateCalls++
		a.stats.TensorUpdateCalls++
		a.stats.ResidentGradUpdateCalls++
		avoided := int64(item.elements * 4)
		a.stats.ResidentGradUploadBytesAvoided += avoided
		if owner != nil {
			owner.stats.HostGradUploadBytesAvoided += avoided
		}
		if update.Config.Step != 0 && update.Config.Step != a.lastLogicalStep {
			a.stats.LogicalSteps++
			a.lastLogicalStep = update.Config.Step
		}
		if update.Config.DeferSync && update.Name != "" {
			a.stats.DeferredSyncUpdates++
		}
	}
	a.stats.ResidentGradBatchCalls++
	a.stats.ResidentGradBatchKernelLaunches += int64(kernelLaunches)
	a.stats.ResidentGradBatchKernelSyncs++
	a.stats.UpdateNanos += elapsed
	a.stats.ResidentGradUpdateNanos += elapsed
	if owner != nil {
		owner.stats.OptimizerResidentGradNanos += elapsed
	}
	a.stats.ResidentParams = int64(len(a.resident))
	a.updatePerStepStats()
}

func (a *optimizerAccelerator) readbackResidentGradientBatch(updates []backend.ResidentGradientOptimizerBatchUpdate, validated []*residentGradientUpdateValidation, download func([]float32, uint64) error) error {
	if download == nil {
		return fmt.Errorf("cuda optimizer resident-gradient batch readback is unavailable")
	}
	for i, update := range updates {
		item := validated[i]
		if item == nil || item.elements == 0 || (update.Config.DeferSync && update.Name != "") {
			continue
		}
		if err := download(update.Tensor.F32, uint64(item.state.param)); err != nil {
			return fmt.Errorf("cuda optimizer resident-gradient batch readback %q: %w", update.Name, err)
		}
		a.stats.DownloadedBytes += int64(len(update.Tensor.F32) * 4)
		a.updatePerStepStats()
	}
	return nil
}

func (a *optimizerAccelerator) ApplyUpdateWithResidentGradBatch(updates []backend.ResidentGradientOptimizerBatchUpdate) error {
	if a == nil {
		return fmt.Errorf("cuda optimizer accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	validated, owner, err := a.validateResidentGradientBatchLocked(updates)
	if err != nil {
		return err
	}
	if owner != nil {
		defer owner.mu.Unlock()
	}
	if len(updates) == 0 {
		return nil
	}
	launches := make([]optimizerUpdateLaunch, 0, len(updates))
	for i, update := range updates {
		item := validated[i]
		if item == nil || item.elements == 0 {
			continue
		}
		corr1 := float32(1)
		corr2 := float32(1)
		if item.mode == 1 {
			corr1 -= float32(math.Pow(float64(update.Config.Beta1), float64(update.Config.Step)))
			corr2 -= float32(math.Pow(float64(update.Config.Beta2), float64(update.Config.Step)))
		}
		block := uint(128)
		grid := uint((item.elements-1)/int(block) + 1)
		launches = append(launches, optimizerUpdateLaunch{
			grid:         grid,
			block:        block,
			param:        uint64(item.state.param),
			Mom1:         uint64(item.state.mom1),
			Mom2:         uint64(item.state.mom2),
			grad:         uint64(item.residentGrad.ptr),
			elements:     item.elements,
			mode:         item.mode,
			learningRate: update.Config.LearningRate,
			weightDecay:  update.Config.WeightDecay,
			beta1:        update.Config.Beta1,
			beta2:        update.Config.Beta2,
			corr1:        corr1,
			corr2:        corr2,
			epsilon:      update.Config.Epsilon,
			scale:        update.Config.Scale,
		})
	}
	if len(launches) == 0 {
		return nil
	}
	start := time.Now()
	if !eosCudaCompactProfileEventsEnabled {
		enqueued, attempted, launchErr := a.device.launchOptimizerUpdateBatch(a.kernel, launches)
		if launchErr != nil {
			return a.poisonResidentGradientBatchLocked(owner, fmt.Errorf("cuda optimizer resident-gradient batch launch (attempted=%t, %d/%d enqueued): %w", attempted, enqueued, len(launches), launchErr))
		}
		if enqueued != len(launches) {
			return a.poisonResidentGradientBatchLocked(owner, fmt.Errorf("cuda optimizer resident-gradient batch reported %d/%d enqueued without error", enqueued, len(launches)))
		}
		// The native batch call has completed its single stream barrier at this
		// point. Commit all device-work accounting before any immediate host
		// readbacks so a later D2H failure still reports the completed batch.
		elapsed := time.Since(start).Nanoseconds()
		a.commitResidentGradientBatchDeviceWork(updates, validated, owner, len(launches), elapsed)
		if err := a.readbackResidentGradientBatch(updates, validated, func(dst []float32, ptr uint64) error {
			return a.device.downloadFloat32(dst, C.CUdeviceptr(ptr))
		}); err != nil {
			return a.poisonResidentGradientBatchLocked(owner, err)
		}
		return nil
	}
	k5Event := false
	k5EndRecorded := false
	eventRecords := int64(0)
	k5TelemetryGoCalls := int64(0)
	k5EventControlFailures := int64(0)
	k5EventControlStage := ""
	k5EventProgress := profileEventProgress{}
	if eosCudaCompactProfileEventsEnabled {
		startResult, startErr := a.device.profileEventRecordStats(string(backend.CompactTrainPhaseK5Batch), false)
		if startResult.PairCreateCalled {
			k5TelemetryGoCalls++
		}
		if startResult.RecordCalled {
			k5TelemetryGoCalls++
		}
		k5EventProgress = addProfileEventProgress(k5EventProgress, startResult.Progress)
		if startErr == nil {
			eventRecords++
			// Binding is a cgo control call; it does not issue a CUDA
			// context or driver operation.
			k5TelemetryGoCalls++
			if a.device.profileEventBindK5End(string(backend.CompactTrainPhaseK5Batch), true) == nil {
				k5Event = true
			} else {
				k5EventControlFailures++
				k5EventControlStage = "event_bind"
				k5TelemetryGoCalls++
				if err := a.device.profileEventBindK5End(string(backend.CompactTrainPhaseK5Batch), false); err != nil {
					k5EventControlFailures++
					k5EventControlStage = "event_bind"
				}
			}
		}
	}
	batchProgress := a.device.launchOptimizerUpdateBatchWithProgress(a.kernel, launches)
	enqueued := batchProgress.Enqueued
	attempted := batchProgress.Attempted != 0
	err = batchProgress.Err
	if eosCudaCompactProfileEventsEnabled && k5Event {
		// The native batch wrapper records the end marker immediately before
		// its existing barrier. Never add a Go-side end record after the
		// wrapper has returned: that would not bracket the queued work.
		var endProgress profileEventProgress
		k5EndRecorded, endProgress = a.device.profileEventK5EndProgress()
		k5EventProgress = addProfileEventProgress(k5EventProgress, endProgress)
		k5TelemetryGoCalls++ // cgo getter; no context/driver operation.
		if k5EndRecorded {
			eventRecords++
		}
		k5TelemetryGoCalls++ // cgo unbind; no context/driver operation.
		if unbindErr := a.device.profileEventBindK5End(string(backend.CompactTrainPhaseK5Batch), false); unbindErr != nil {
			k5EventControlFailures++
			k5EventControlStage = "event_bind"
		}
	}
	k5Phase := compactTrainK5WorkloadTelemetry(batchProgress)
	// Event-control progress is added separately from workload progress. The K5
	// end marker is recorded by the existing batch bridge immediately before
	// its barrier, and its typed attempted/success result is merged below.
	k5Phase.GoCalls += k5TelemetryGoCalls
	k5Phase.HostNanos = time.Since(start).Nanoseconds()
	k5Phase.EventRecords = eventRecords
	k5Phase.EventControlFailures += k5EventControlFailures
	if k5EventControlStage != "" {
		k5Phase.EventControlStage = k5EventControlStage
	}
	addCompactTrainEventProgress(&k5Phase, k5EventProgress, false, "")
	a.addCompactTrainTelemetry(backend.CompactTrainPhaseK5Batch, k5Phase)
	if owner != nil {
		owner.telemetryAdd(backend.CompactTrainPhaseK5Batch, k5Phase)
	}
	if k5Event && k5EndRecorded && err == nil {
		nanos, queryCalled, queryProgress, elapsedErr := a.device.profileEventElapsedStats(string(backend.CompactTrainPhaseK5Batch))
		if queryCalled {
			query := backend.CompactTrainPhaseTelemetry{GoCalls: 1}
			addCompactTrainEventProgress(&query, queryProgress, elapsedErr != nil, "event_query")
			if elapsedErr == nil {
				query.EventQueries = 1
				query.DeviceElapsedNanos = nanos
				if a.stats.CompactTrainTelemetry != nil {
					a.stats.CompactTrainTelemetry.EventTiming = true
				}
				if owner != nil {
					if owner.stats.Telemetry != nil {
						owner.stats.Telemetry.EventTiming = true
					}
				}
			}
			a.addCompactTrainTelemetry(backend.CompactTrainPhaseK5Batch, query)
			if owner != nil {
				owner.telemetryAdd(backend.CompactTrainPhaseK5Batch, query)
			}
		}
	}
	if err != nil {
		return a.poisonResidentGradientBatchLocked(owner, fmt.Errorf("cuda optimizer resident-gradient batch launch (attempted=%t, %d/%d enqueued): %w", attempted, enqueued, len(launches), err))
	}
	if enqueued != len(launches) {
		return a.poisonResidentGradientBatchLocked(owner, fmt.Errorf("cuda optimizer resident-gradient batch reported %d/%d enqueued without error", enqueued, len(launches)))
	}
	// The native batch call has completed its single stream barrier at this
	// point. Commit all device-work accounting before any immediate host
	// readbacks so a later D2H failure still reports the completed batch.
	elapsed := time.Since(start).Nanoseconds()
	a.commitResidentGradientBatchDeviceWork(updates, validated, owner, len(launches), elapsed)

	// The one batch barrier above is mandatory even when every item defers its
	// host copy. Immediate items read back only after that barrier, preserving
	// the scalar path's host-visible state without adding stream barriers.
	if err := a.readbackResidentGradientBatch(updates, validated, func(dst []float32, ptr uint64) error {
		return a.device.downloadFloat32(dst, C.CUdeviceptr(ptr))
	}); err != nil {
		return a.poisonResidentGradientBatchLocked(owner, err)
	}
	return nil
}

func (a *optimizerAccelerator) SyncState(name string, tensor, mom1, mom2 *backend.Tensor, includeMoments bool) error {
	return a.syncState(name, tensor, mom1, mom2, includeMoments, "")
}

func (a *optimizerAccelerator) SyncStateWithReason(name string, tensor, mom1, mom2 *backend.Tensor, includeMoments bool, reason string) error {
	return a.syncState(name, tensor, mom1, mom2, includeMoments, reason)
}

func (a *optimizerAccelerator) syncState(name string, tensor, mom1, mom2 *backend.Tensor, includeMoments bool, reason string) error {
	if a == nil {
		return fmt.Errorf("cuda optimizer accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.device == nil {
		return fmt.Errorf("cuda optimizer accelerator is not initialized")
	}
	if name == "" {
		return fmt.Errorf("cuda optimizer sync requires a resident parameter name")
	}
	start := time.Now()
	state, ok := a.resident[name]
	if !ok {
		return fmt.Errorf("cuda optimizer state %q is not resident", name)
	}
	if tensor != nil {
		if len(tensor.F32) != state.elements {
			return fmt.Errorf("cuda optimizer tensor size %d does not match resident size %d", len(tensor.F32), state.elements)
		}
		if err := a.device.downloadFloat32(tensor.F32, state.param); err != nil {
			return err
		}
		a.stats.DownloadedBytes += int64(len(tensor.F32) * 4)
	}
	if !includeMoments {
		a.stats.SyncCalls++
		if reason != "" {
			a.stats.ForcedSyncCalls++
			a.stats.LastForcedSyncReason = reason
		}
		a.stats.SyncNanos += time.Since(start).Nanoseconds()
		a.stats.ResidentParams = int64(len(a.resident))
		a.updatePerStepStats()
		return nil
	}
	if state.hasMoments {
		if mom1 != nil {
			if len(mom1.F32) != state.elements {
				return fmt.Errorf("cuda optimizer first moment size %d does not match resident size %d", len(mom1.F32), state.elements)
			}
			if err := a.device.downloadFloat32(mom1.F32, state.mom1); err != nil {
				return err
			}
			a.stats.DownloadedBytes += int64(len(mom1.F32) * 4)
		}
		if mom2 != nil {
			if len(mom2.F32) != state.elements {
				return fmt.Errorf("cuda optimizer second moment size %d does not match resident size %d", len(mom2.F32), state.elements)
			}
			if err := a.device.downloadFloat32(mom2.F32, state.mom2); err != nil {
				return err
			}
			a.stats.DownloadedBytes += int64(len(mom2.F32) * 4)
		}
	}
	a.stats.SyncCalls++
	if reason != "" {
		a.stats.ForcedSyncCalls++
		a.stats.LastForcedSyncReason = reason
	}
	a.stats.SyncNanos += time.Since(start).Nanoseconds()
	a.stats.ResidentParams = int64(len(a.resident))
	a.updatePerStepStats()
	return nil
}

func (a *optimizerAccelerator) ResidentParameter(name string) (backend.OptimizerResidentParameter, bool) {
	if a == nil || name == "" {
		return backend.OptimizerResidentParameter{}, false
	}
	a.mu.RLock()
	defer a.mu.RUnlock()
	state, ok := a.resident[name]
	if !ok {
		return backend.OptimizerResidentParameter{}, false
	}
	return backend.OptimizerResidentParameter{
		Backend:  eosartifact.BackendCUDA,
		Token:    state.token,
		Elements: state.elements,
	}, true
}

func (a *optimizerAccelerator) EnsureResidentParameter(name string, tensor, mom1, mom2 *backend.Tensor) error {
	if a == nil {
		return fmt.Errorf("cuda optimizer accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.device == nil {
		return fmt.Errorf("cuda optimizer accelerator is not initialized")
	}
	if name == "" {
		return fmt.Errorf("cuda optimizer resident seed requires a parameter name")
	}
	if tensor == nil {
		return fmt.Errorf("cuda optimizer resident seed %q requires tensor", name)
	}
	elements := len(tensor.F32)
	if elements == 0 {
		return nil
	}
	if mom1 == nil || mom2 == nil {
		return fmt.Errorf("cuda optimizer resident seed %q requires moments", name)
	}
	if len(mom1.F32) != elements || len(mom2.F32) != elements {
		return fmt.Errorf("cuda optimizer resident seed %q moment sizes do not match tensor size %d", name, elements)
	}
	_, transient, err := a.ensureResidentState(name, tensor, mom1, mom2, elements, true)
	if err != nil {
		return err
	}
	if transient {
		return fmt.Errorf("cuda optimizer resident seed %q unexpectedly created transient state", name)
	}
	a.updatePerStepStats()
	return nil
}

func (a *optimizerAccelerator) updatePerStepStats() {
	if a == nil || a.stats.UpdateCalls == 0 {
		return
	}
	steps := float64(a.stats.LogicalSteps)
	if steps == 0 {
		steps = float64(a.stats.UpdateCalls)
	}
	a.stats.UploadedBytesPerStep = float64(a.stats.UploadedBytes) / steps
	a.stats.DownloadedBytesPerStep = float64(a.stats.DownloadedBytes) / steps
}

func (a *optimizerAccelerator) ensureResidentState(name string, tensor, mom1, mom2 *backend.Tensor, elements int, requireMoments bool) (residentOptimizerState, bool, error) {
	if name == "" {
		state, err := a.freshResidentState(tensor, mom1, mom2, elements, requireMoments)
		return state, true, err
	}
	if a.resident == nil {
		a.resident = map[string]residentOptimizerState{}
	}
	if state, ok := a.resident[name]; ok {
		if state.elements == elements && state.hasMoments == requireMoments {
			return state, false, nil
		}
		a.releaseResidentState(state)
		delete(a.resident, name)
	}
	state, err := a.freshResidentState(tensor, mom1, mom2, elements, requireMoments)
	if err != nil {
		return residentOptimizerState{}, false, err
	}
	a.nextGen++
	state.generation = a.nextGen
	state.token = &optimizerResidentParameterToken{owner: a, name: name, generation: state.generation}
	a.resident[name] = state
	a.stats.ResidentParams = int64(len(a.resident))
	return state, false, nil
}

func (a *optimizerAccelerator) freshResidentState(tensor, mom1, mom2 *backend.Tensor, elements int, requireMoments bool) (residentOptimizerState, error) {
	paramBuf, err := a.device.uploadFloat32(tensor.F32)
	if err != nil {
		return residentOptimizerState{}, err
	}
	state := residentOptimizerState{
		param:      paramBuf,
		elements:   elements,
		hasMoments: requireMoments,
	}
	a.stats.UploadedBytes += int64(len(tensor.F32) * 4)
	if requireMoments {
		mom1Buf, err := a.device.uploadFloat32(mom1.F32)
		if err != nil {
			a.releaseResidentState(state)
			return residentOptimizerState{}, err
		}
		state.mom1 = mom1Buf
		a.stats.UploadedBytes += int64(len(mom1.F32) * 4)
		mom2Buf, err := a.device.uploadFloat32(mom2.F32)
		if err != nil {
			a.releaseResidentState(state)
			return residentOptimizerState{}, err
		}
		state.mom2 = mom2Buf
		a.stats.UploadedBytes += int64(len(mom2.F32) * 4)
	}
	return state, nil
}

func (a *optimizerAccelerator) releaseResidentState(state residentOptimizerState) {
	if a == nil || a.device == nil {
		return
	}
	_ = a.device.freeBuffer(state.param)
	_ = a.device.freeBuffer(state.mom1)
	_ = a.device.freeBuffer(state.mom2)
}

func (a *optimizerAccelerator) Close() {
	if a == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.device == nil {
		return
	}
	for name, state := range a.resident {
		a.releaseResidentState(state)
		delete(a.resident, name)
	}
	a.stats.ResidentParams = 0
	a.device.destroyAuxKernel(a.kernel)
	a.kernel = nil
	a.resident = nil
	a.device.close()
	a.device = nil
}
