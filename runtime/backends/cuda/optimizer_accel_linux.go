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
	return stats
}

func (a *optimizerAccelerator) ApplyUpdate(name string, cfg backend.OptimizerUpdateConfig, tensor, mom1, mom2, grad *backend.Tensor) error {
	if a == nil {
		return fmt.Errorf("cuda optimizer accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.device == nil || a.kernel == nil {
		return fmt.Errorf("cuda optimizer accelerator is not initialized")
	}
	if tensor == nil || grad == nil {
		return fmt.Errorf("cuda optimizer update requires tensor and grad")
	}
	elements := len(tensor.F32)
	if elements == 0 {
		return nil
	}
	if len(grad.F32) != elements {
		return fmt.Errorf("cuda optimizer grad size %d does not match tensor size %d", len(grad.F32), elements)
	}
	start := time.Now()
	mode := 1
	switch cfg.Optimizer {
	case "", "adamw":
		if mom1 == nil || mom2 == nil {
			return fmt.Errorf("cuda adamw update requires first and second moment tensors")
		}
	case "sgd":
		mode = 0
	default:
		return fmt.Errorf("cuda optimizer update does not support %q", cfg.Optimizer)
	}
	if mom1 != nil && len(mom1.F32) != elements {
		return fmt.Errorf("cuda optimizer first moment size %d does not match tensor size %d", len(mom1.F32), elements)
	}
	if mom2 != nil && len(mom2.F32) != elements {
		return fmt.Errorf("cuda optimizer second moment size %d does not match tensor size %d", len(mom2.F32), elements)
	}
	gradBuf, err := a.device.uploadFloat32(grad.F32)
	if err != nil {
		return err
	}
	defer a.device.freeBuffer(gradBuf)
	a.stats.UploadedBytes += int64(len(grad.F32) * 4)
	state, transient, err := a.ensureResidentState(name, tensor, mom1, mom2, elements, mode == 1)
	if err != nil {
		return err
	}
	if transient {
		defer a.releaseResidentState(state)
	}

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
		gradBuf,
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
		Token:    &optimizerResidentParameterToken{owner: a, name: name, generation: state.generation},
		Elements: state.elements,
	}, true
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
