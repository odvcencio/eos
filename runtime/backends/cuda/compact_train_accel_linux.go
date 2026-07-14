//go:build linux && cgo

package cuda

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#include <cuda.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct {
	CUcontext ctx;
	CUdevice device;
	int major;
	int minor;
	int primary_ctx;
	void* blas;
	CUstream stream;
} EosCudaRuntime;

typedef struct {
	CUmodule module;
	CUfunction function;
} EosCudaKernel;

static int eosCudaLaunchCompactTrainFinalProjectionGrad(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr normalized, CUdeviceptr gradRows, CUdeviceptr gradOutProjection, int rows, int modelDim, int outDim, char** err);
static int eosCudaLaunchCompactTrainFinalHiddenGrad(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr projected, CUdeviceptr normalized, CUdeviceptr outputProjection, CUdeviceptr gradPooled, CUdeviceptr masks, CUdeviceptr active, CUdeviceptr gradRows, CUdeviceptr gradNormalized, CUdeviceptr gradHidden, int batch, int seq, int modelDim, int outDim, int hasProjection, char** err);

static char* eos_compact_train_dup_cu_error(const char* prefix, CUresult res) {
	const char* name = 0;
	const char* detail = 0;
	cuGetErrorName(res, &name);
	cuGetErrorString(res, &detail);
	if (name == 0) name = "CU_UNKNOWN";
	if (detail == 0) detail = "unknown";
	size_t n = 0;
	while (prefix[n] != 0) n++;
	size_t a = 0;
	while (name[a] != 0) a++;
	size_t b = 0;
	while (detail[b] != 0) b++;
	char* out = (char*)malloc(n + a + b + 6);
	if (out == 0) return 0;
	snprintf(out, n + a + b + 6, "%s: %s (%s)", prefix, name, detail);
	return out;
}

static int eos_compact_train_launch(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, void** args, char** err) {
	CUresult res = cuCtxSetCurrent(rt->ctx);
	if (res != CUDA_SUCCESS) {
		*err = eos_compact_train_dup_cu_error("cuCtxSetCurrent", res);
		return 1;
	}
	res = cuLaunchKernel(kernel->function, grid, 1, 1, block, 1, 1, 0, rt->stream, args, 0);
	if (res != CUDA_SUCCESS) {
		*err = eos_compact_train_dup_cu_error("cuLaunchKernel", res);
		return 1;
	}
	res = cuStreamSynchronize(rt->stream);
	if (res != CUDA_SUCCESS) {
		*err = eos_compact_train_dup_cu_error("cuStreamSynchronize", res);
		return 1;
	}
	return 0;
}

static int eosCudaLaunchCompactTrainFinalProjectionGrad(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr normalized, CUdeviceptr gradRows, CUdeviceptr gradOutProjection, int rows, int modelDim, int outDim, char** err) {
	void* args[] = {&normalized, &gradRows, &gradOutProjection, &rows, &modelDim, &outDim};
	return eos_compact_train_launch(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchCompactTrainFinalHiddenGrad(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr projected, CUdeviceptr normalized, CUdeviceptr outputProjection, CUdeviceptr gradPooled, CUdeviceptr masks, CUdeviceptr active, CUdeviceptr gradRows, CUdeviceptr gradNormalized, CUdeviceptr gradHidden, int batch, int seq, int modelDim, int outDim, int hasProjection, char** err) {
	void* args[] = {&projected, &normalized, &outputProjection, &gradPooled, &masks, &active, &gradRows, &gradNormalized, &gradHidden, &batch, &seq, &modelDim, &outDim, &hasProjection};
	return eos_compact_train_launch(rt, kernel, grid, block, args, err);
}
*/
import "C"

import (
	"fmt"
	"sync/atomic"
	"time"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

type CompactTrainAccelerator struct {
	*CompactForwardAccelerator
	stats        backend.CompactTrainAcceleratorStats
	arena        *compactTrainArena
	trainKernels compactTrainKernels
	grads        map[string]*compactTrainGradient
	gradGen      uint64
	stepID       uint64
	stepActive   bool
	nextHandleID uint64
	closed       bool
}

type compactTrainKernels struct {
	finalProjectionGrad *auxKernel
	finalHiddenGrad     *auxKernel
}

type compactTrainGradient struct {
	ptr        C.CUdeviceptr
	elements   int
	generation uint64
	stepID     uint64
	token      *compactTrainGradientToken
}

type compactTrainGradientToken struct {
	owner      *CompactTrainAccelerator
	name       string
	generation uint64
	stepID     uint64
	elements   int
}

type compactTrainArena struct {
	shape      backend.CompactForwardShape
	generation uint64
	live       bool
	token      *compactTrainHandleToken

	tokens C.CUdeviceptr
	masks  C.CUdeviceptr
	roles  C.CUdeviceptr
	status C.CUdeviceptr

	input               C.CUdeviceptr
	active              C.CUdeviceptr
	finalNorm           C.CUdeviceptr
	outputRows          C.CUdeviceptr
	preProjectionPooled C.CUdeviceptr
	finalPooled         C.CUdeviceptr
	gradPooled          C.CUdeviceptr
	gradOutputRows      C.CUdeviceptr
	gradNormalized      C.CUdeviceptr
	gradHidden          C.CUdeviceptr
	modelScratch        []C.CUdeviceptr
	ffnScratch          []C.CUdeviceptr
	layers              []compactTrainLayerArena
	bytes               int64
	workspaceBytes      int64
}

type compactTrainLayerArena struct {
	input        C.CUdeviceptr
	hidden       C.CUdeviceptr
	attnQ        C.CUdeviceptr
	attnK        C.CUdeviceptr
	attnV        C.CUdeviceptr
	attnScores   C.CUdeviceptr
	attnMixed    C.CUdeviceptr
	attnResidual C.CUdeviceptr
	ffnHidden    C.CUdeviceptr
	activated    C.CUdeviceptr
	ffnResidual  C.CUdeviceptr
	projected    C.CUdeviceptr
}

type compactTrainHandleToken struct {
	backend    eosartifact.BackendKind
	generation uint64
	stepID     uint64
	id         uint64
	alive      atomic.Bool
}

func (t *compactTrainHandleToken) CompactTrainHandleToken() {}
func (t *compactTrainHandleToken) Backend() eosartifact.BackendKind {
	if t == nil {
		return ""
	}
	return t.backend
}
func (t *compactTrainHandleToken) Generation() uint64 {
	if t == nil {
		return 0
	}
	return t.generation
}
func (t *compactTrainHandleToken) StepID() uint64 {
	if t == nil {
		return 0
	}
	return t.stepID
}
func (t *compactTrainHandleToken) Alive() bool {
	return t != nil && t.alive.Load()
}

func (t *compactTrainGradientToken) ResidentGradientToken() {}
func (t *compactTrainGradientToken) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}
func (t *compactTrainGradientToken) Generation() uint64 {
	if t == nil {
		return 0
	}
	return t.generation
}
func (t *compactTrainGradientToken) Alive() bool {
	if t == nil || t.owner == nil || t.name == "" {
		return false
	}
	t.owner.mu.Lock()
	defer t.owner.mu.Unlock()
	if t.owner.closed || t.owner.device == nil {
		return false
	}
	grad := t.owner.grads[t.name]
	return grad != nil && grad.token == t && grad.generation == t.generation && grad.stepID == t.stepID && grad.elements == t.elements && grad.ptr != 0
}

const compactTrainBackwardKernelSource = `
extern "C" __global__ void eos_compact_train_final_projection_grad(
    const float* normalized, const float* gradRows, float* gradOutProjection,
    int rows, int modelDim, int outDim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = modelDim * outDim;
    if (idx >= total) return;
    int o = idx % outDim;
    int d = idx / outDim;
    float sum = 0.0f;
    for (int r = 0; r < rows; ++r) {
        sum += normalized[r * modelDim + d] * gradRows[r * outDim + o];
    }
    gradOutProjection[idx] += sum;
}

extern "C" __global__ void eos_compact_train_final_hidden_grad(
    const float* projected, const float* normalized, const float* outputProjection,
    const float* gradPooled, const int* masks, const int* active,
    float* gradRows, float* gradNormalized, float* gradHidden,
    int batch, int seq, int modelDim, int outDim, int hasProjection
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * modelDim;
    if (idx >= total) return;
    int d = idx % modelDim;
    int row = idx / modelDim;
    int b = row / seq;
    int pos = row - b * seq;
    int activeCount = active[b];
    float invActive = activeCount > 0 ? 1.0f / (float)activeCount : 0.0f;
    float gnorm = 0.0f;
    if (masks[b * seq + pos] != 0 && activeCount > 0) {
        if (hasProjection != 0) {
            for (int o = 0; o < outDim; ++o) {
                float grow = gradPooled[b * outDim + o] * invActive;
                if (d == 0) gradRows[row * outDim + o] = grow;
                gnorm += grow * outputProjection[d * outDim + o];
            }
        } else {
            gnorm = gradPooled[b * modelDim + d] * invActive;
            gradRows[row * modelDim + d] = gnorm;
        }
    } else if (hasProjection != 0) {
        if (d == 0) {
            for (int o = 0; o < outDim; ++o) gradRows[row * outDim + o] = 0.0f;
        }
    } else {
        gradRows[row * modelDim + d] = 0.0f;
    }
    gradNormalized[idx] = gnorm;
    float normSq = 0.0f;
    float dotNG = 0.0f;
    int base = row * modelDim;
    for (int c = 0; c < modelDim; ++c) {
        float p = projected[base + c];
        normSq += p * p;
        float gc = 0.0f;
        if (masks[b * seq + pos] != 0 && activeCount > 0) {
            if (hasProjection != 0) {
                for (int o = 0; o < outDim; ++o) {
                    gc += gradPooled[b * outDim + o] * invActive * outputProjection[c * outDim + o];
                }
            } else {
                gc = gradPooled[b * modelDim + c] * invActive;
            }
        }
        dotNG += normalized[base + c] * gc;
    }
    float norm = sqrtf(normSq);
    if (norm == 0.0f) {
        gradHidden[idx] = 0.0f;
    } else {
        gradHidden[idx] = (gnorm - normalized[idx] * dotNG) / norm;
    }
}
`

func init() {
	backend.RegisterCompactTrainAccelerator(eosartifact.BackendCUDA, func() (backend.CompactTrainAccelerator, error) {
		return NewCompactTrainAccelerator()
	})
}

func NewCompactTrainAccelerator() (*CompactTrainAccelerator, error) {
	base, err := NewCompactForwardAccelerator()
	if err != nil {
		return nil, err
	}
	compile := func(entry string) (*auxKernel, error) {
		k, err := base.device.compileAuxKernel(compactTrainBackwardKernelSource, entry)
		if err != nil {
			base.Close()
			return nil, err
		}
		return k, nil
	}
	proj, err := compile("eos_compact_train_final_projection_grad")
	if err != nil {
		return nil, err
	}
	hidden, err := compile("eos_compact_train_final_hidden_grad")
	if err != nil {
		base.device.destroyAuxKernel(proj)
		base.Close()
		return nil, err
	}
	return &CompactTrainAccelerator{
		CompactForwardAccelerator: base,
		trainKernels:              compactTrainKernels{finalProjectionGrad: proj, finalHiddenGrad: hidden},
		grads:                     map[string]*compactTrainGradient{},
	}, nil
}

func (a *CompactTrainAccelerator) Backend() eosartifact.BackendKind { return eosartifact.BackendCUDA }

func (a *CompactTrainAccelerator) Close() {
	if a == nil || a.CompactForwardAccelerator == nil {
		return
	}
	a.mu.Lock()
	a.closed = true
	a.releaseGradientsLocked()
	a.releaseArenaLocked()
	if a.device != nil {
		a.device.destroyAuxKernel(a.trainKernels.finalProjectionGrad)
		a.device.destroyAuxKernel(a.trainKernels.finalHiddenGrad)
	}
	a.mu.Unlock()
	a.CompactForwardAccelerator.Close()
}

func (a *CompactTrainAccelerator) CompactTrainStats() backend.CompactTrainAcceleratorStats {
	if a == nil || a.CompactForwardAccelerator == nil {
		return backend.CompactTrainAcceleratorStats{}
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.stats
}

func (a *CompactTrainAccelerator) BindCompactTrainResident(name string, tensor *backend.Tensor, ref backend.OptimizerResidentParameter) error {
	if a == nil || a.CompactForwardAccelerator == nil {
		return fmt.Errorf("cuda compact train accelerator is not initialized")
	}
	return a.BindResident(name, tensor, ref)
}

func (a *CompactTrainAccelerator) PreflightCompactTrainForward(req backend.CompactTrainForwardRequest) error {
	if a == nil || a.CompactForwardAccelerator == nil {
		return fmt.Errorf("cuda compact train accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.preflightCompactTrainForwardLocked(req)
}

func (a *CompactTrainAccelerator) preflightCompactTrainForwardLocked(req backend.CompactTrainForwardRequest) error {
	if a.device == nil || a.closed {
		return fmt.Errorf("cuda compact train accelerator is closed")
	}
	shape := req.Shape
	if err := validateCompactForwardShape(shape); err != nil {
		return err
	}
	if err := validateCompactForwardInputs(shape, req.Tokens, req.Masks, req.Roles); err != nil {
		return err
	}
	if _, err := compactForwardGELUFast(req.GELUMode); err != nil {
		return err
	}
	if len(a.layers) != shape.Layers || len(a.bindings.layer) != shape.Layers {
		return fmt.Errorf("cuda compact train layer bindings %d, want %d", len(a.layers), shape.Layers)
	}
	if err := a.validateBindings(shape); err != nil {
		return err
	}
	if err := a.preflightResidentRefs(req.ResidentRefs, shape); err != nil {
		return err
	}
	return nil
}

func (a *CompactTrainAccelerator) BeginCompactTrainStep(stepID uint64, refs []backend.CompactForwardResidentRef) error {
	if a == nil || a.CompactForwardAccelerator == nil {
		return fmt.Errorf("cuda compact train accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.device == nil || a.closed {
		return fmt.Errorf("cuda compact train accelerator is closed")
	}
	if a.stepActive {
		return fmt.Errorf("cuda compact train step %d is already active", a.stepID)
	}
	if len(refs) == 0 {
		return fmt.Errorf("cuda compact train resident refs are required")
	}
	byName := make(map[string]backend.CompactForwardResidentRef, len(refs))
	for _, ref := range refs {
		if ref.Name == "" {
			return fmt.Errorf("cuda compact train resident ref has empty name")
		}
		if _, exists := byName[ref.Name]; exists {
			return fmt.Errorf("cuda compact train resident ref %q is duplicated", ref.Name)
		}
		if err := a.validateStepResidentRefLocked(ref); err != nil {
			return err
		}
		byName[ref.Name] = ref
	}
	nextGradGen := a.gradGen + 1
	pending := make(map[string]*compactTrainGradient, len(refs))
	freePending := func() {
		for _, grad := range pending {
			if grad != nil && grad.ptr != 0 {
				_ = a.device.freeBuffer(grad.ptr)
				grad.ptr = 0
			}
		}
	}
	var bytes int64
	for _, ref := range refs {
		ptr, err := a.device.allocFloat32(ref.Elements)
		if err != nil {
			freePending()
			return err
		}
		if err := a.device.copyFloat32ToBuffer(ptr, make([]float32, ref.Elements)); err != nil {
			_ = a.device.freeBuffer(ptr)
			freePending()
			return err
		}
		token := &compactTrainGradientToken{owner: a, name: ref.Name, generation: nextGradGen, stepID: stepID, elements: ref.Elements}
		pending[ref.Name] = &compactTrainGradient{ptr: ptr, elements: ref.Elements, generation: nextGradGen, stepID: stepID, token: token}
		bytes += int64(ref.Elements * 4)
	}
	a.releaseGradientsLocked()
	a.gradGen = nextGradGen
	a.stepID = stepID
	a.stepActive = true
	a.grads = pending
	a.stats.GradientZeroCalls++
	a.stats.ResidentGradBytes = bytes
	return nil
}

func (a *CompactTrainAccelerator) EndCompactTrainStep(stepID uint64) error {
	if a == nil || a.CompactForwardAccelerator == nil {
		return fmt.Errorf("cuda compact train accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.device == nil || a.closed {
		return fmt.Errorf("cuda compact train accelerator is closed")
	}
	if !a.stepActive {
		return fmt.Errorf("cuda compact train step is not active")
	}
	if a.stepID != stepID {
		return fmt.Errorf("cuda compact train step %d is stale, current %d", stepID, a.stepID)
	}
	if a.arena != nil && a.arena.live {
		return fmt.Errorf("cuda compact train end step with live handle")
	}
	a.stepActive = false
	return nil
}

func (a *CompactTrainAccelerator) RunCompactTrainBackward(req backend.CompactTrainBackwardRequest) (backend.CompactTrainBackwardResult, error) {
	if a == nil || a.CompactForwardAccelerator == nil {
		return backend.CompactTrainBackwardResult{}, fmt.Errorf("cuda compact train accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.validateBackwardRequestLocked(req); err != nil {
		return backend.CompactTrainBackwardResult{}, err
	}
	a.stats.FallbackOrUnhandled++
	return backend.CompactTrainBackwardResult{}, fmt.Errorf("cuda compact train full layer backward is unsupported in this bounded final-output slice")
}

type compactTrainFinalOutputDebugResult struct {
	ResidentGradRefs []backend.ResidentGradientRef
	GradHidden       *backend.Tensor
}

func (a *CompactTrainAccelerator) runCompactTrainFinalOutputBackwardForDebug(req backend.CompactTrainBackwardRequest) (compactTrainFinalOutputDebugResult, error) {
	if a == nil || a.CompactForwardAccelerator == nil {
		return compactTrainFinalOutputDebugResult{}, fmt.Errorf("cuda compact train accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	start := time.Now()
	if err := a.validateBackwardRequestLocked(req); err != nil {
		return compactTrainFinalOutputDebugResult{}, err
	}
	arena := a.arena
	shape := arena.shape
	launchesBefore := a.CompactForwardAccelerator.stats.KernelLaunches
	syncsBefore := a.CompactForwardAccelerator.stats.KernelSynchronizations
	if err := a.ensureBackwardWorkspaceLocked(arena); err != nil {
		return compactTrainFinalOutputDebugResult{}, err
	}
	if err := a.device.copyFloat32ToBuffer(arena.gradPooled, req.GradPooled.F32); err != nil {
		return compactTrainFinalOutputDebugResult{}, err
	}
	a.stats.UploadedBytes += int64(len(req.GradPooled.F32) * 4)
	a.stats.GradPooledUploadedBytes += int64(len(req.GradPooled.F32) * 4)
	lastProjected := arena.input
	if shape.Layers > 0 {
		lastProjected = arena.layers[shape.Layers-1].projected
	}
	outProjectionPtr := C.CUdeviceptr(0)
	if shape.HasOutputProjection {
		outProjectionPtr = a.bindings.out.ptr
	}
	if err := a.launchFinalHiddenGrad(lastProjected, arena.finalNorm, outProjectionPtr, arena.gradPooled, arena.masks, arena.active, arena.gradOutputRows, arena.gradNormalized, arena.gradHidden, shape); err != nil {
		return compactTrainFinalOutputDebugResult{}, err
	}
	if shape.HasOutputProjection {
		grad, ok := a.grads[a.outName]
		if !ok || grad == nil || grad.ptr == 0 {
			return compactTrainFinalOutputDebugResult{}, fmt.Errorf("cuda compact train output projection gradient %q is not resident", a.outName)
		}
		if err := a.launchFinalProjectionGrad(arena.finalNorm, arena.gradOutputRows, grad.ptr, shape.Batch*shape.Tokens, shape.ModelDim, shape.OutputDim); err != nil {
			return compactTrainFinalOutputDebugResult{}, err
		}
	}
	hidden := make([]float32, shape.Batch*shape.Tokens*shape.ModelDim)
	if err := a.device.downloadFloat32(hidden, arena.gradHidden); err != nil {
		return compactTrainFinalOutputDebugResult{}, err
	}
	if err := a.consumeHandleLocked(req.Handle); err != nil {
		return compactTrainFinalOutputDebugResult{}, err
	}
	backwardLaunches := a.CompactForwardAccelerator.stats.KernelLaunches - launchesBefore
	backwardSyncs := a.CompactForwardAccelerator.stats.KernelSynchronizations - syncsBefore
	a.stats.BackwardCalls++
	a.stats.BackwardNanos += time.Since(start).Nanoseconds()
	a.stats.DownloadedBytes += int64(len(hidden) * 4)
	a.stats.KernelLaunches += backwardLaunches
	a.stats.KernelSynchronizations += backwardSyncs
	a.stats.LastBackwardLaunches = backwardLaunches
	a.stats.LastBackwardSyncs = backwardSyncs
	return compactTrainFinalOutputDebugResult{
		ResidentGradRefs: a.residentGradientRefsLocked(),
		GradHidden:       backend.NewTensorF32([]int{shape.Batch, shape.Tokens, shape.ModelDim}, hidden),
	}, nil
}

func (a *CompactTrainAccelerator) copyResidentGradientForDebug(ref backend.ResidentGradientRef) (*backend.Tensor, error) {
	if a == nil || a.CompactForwardAccelerator == nil {
		return nil, fmt.Errorf("cuda compact train accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	grad, err := a.validateResidentGradientRefLocked(ref)
	if err != nil {
		return nil, err
	}
	host := make([]float32, grad.elements)
	if err := a.device.downloadFloat32(host, grad.ptr); err != nil {
		return nil, err
	}
	a.stats.DownloadedBytes += int64(len(host) * 4)
	return backend.NewTensorF32([]int{grad.elements}, host), nil
}

func (a *CompactTrainAccelerator) validateStepResidentRefLocked(ref backend.CompactForwardResidentRef) error {
	if ref.Backend != eosartifact.BackendCUDA {
		return fmt.Errorf("cuda compact train resident ref %q backend %q, want cuda", ref.Name, ref.Backend)
	}
	if ref.Token == nil {
		return fmt.Errorf("cuda compact train resident ref %q token is nil", ref.Name)
	}
	if ref.Token.Backend() != eosartifact.BackendCUDA {
		return fmt.Errorf("cuda compact train resident ref %q token backend %q, want cuda", ref.Name, ref.Token.Backend())
	}
	token, ok := ref.Token.(*optimizerResidentParameterToken)
	if !ok || token == nil {
		return fmt.Errorf("cuda compact train resident ref %q has invalid cuda token", ref.Name)
	}
	binding, ok := a.bindingForName(ref.Name)
	if !ok {
		return fmt.Errorf("cuda compact train resident binding %q is missing", ref.Name)
	}
	state, unlock, err := token.lockCurrent()
	if err != nil {
		return err
	}
	defer unlock()
	if ref.Elements != state.elements {
		return fmt.Errorf("cuda compact train resident ref %q elements %d, want %d", ref.Name, ref.Elements, state.elements)
	}
	if binding.ptr != state.param || binding.elements != state.elements {
		return fmt.Errorf("cuda compact train resident ref %q no longer matches bound device state", ref.Name)
	}
	bridged := a.bridged[ref.Name]
	if bridged == nil || bridged.generation != token.generation {
		return fmt.Errorf("cuda compact train resident ref %q generation %d does not match bound generation", ref.Name, token.generation)
	}
	return nil
}

func (a *CompactTrainAccelerator) validateBackwardRequestLocked(req backend.CompactTrainBackwardRequest) error {
	if a.device == nil || a.closed {
		return fmt.Errorf("cuda compact train accelerator is closed")
	}
	if !a.stepActive {
		return fmt.Errorf("cuda compact train step is not active")
	}
	if a.arena == nil || !a.arena.live {
		return fmt.Errorf("cuda compact train backward requires a live handle")
	}
	if err := a.validateHandleLocked(req.Handle); err != nil {
		return err
	}
	shape := a.arena.shape
	if req.GradPooled == nil || req.GradPooled.DType != "f32" || len(req.GradPooled.Shape) != 2 || req.GradPooled.Shape[0] != shape.Batch || req.GradPooled.Shape[1] != shape.OutputDim || len(req.GradPooled.F32) != shape.Batch*shape.OutputDim {
		return fmt.Errorf("cuda compact train pooled gradient shape %v, want [%d %d]", tensorShapeForError(req.GradPooled), shape.Batch, shape.OutputDim)
	}
	return nil
}

func (a *CompactTrainAccelerator) validateResidentGradientRefLocked(ref backend.ResidentGradientRef) (*compactTrainGradient, error) {
	if a.device == nil || a.closed {
		return nil, fmt.Errorf("cuda compact train accelerator is closed")
	}
	if !a.stepActive {
		return nil, fmt.Errorf("cuda compact train step is not active")
	}
	if ref.StepID != a.stepID {
		return nil, fmt.Errorf("cuda compact train resident gradient %q step %d is stale, current %d", ref.Name, ref.StepID, a.stepID)
	}
	if ref.Backend != eosartifact.BackendCUDA {
		return nil, fmt.Errorf("cuda compact train resident gradient %q backend %q, want cuda", ref.Name, ref.Backend)
	}
	token, ok := ref.Token.(*compactTrainGradientToken)
	if !ok || token == nil {
		return nil, fmt.Errorf("cuda compact train resident gradient %q has invalid token", ref.Name)
	}
	if token.owner != a || token.name != ref.Name {
		return nil, fmt.Errorf("cuda compact train resident gradient %q owner/name mismatch", ref.Name)
	}
	if token.generation != ref.Generation || token.stepID != ref.StepID || token.elements != ref.Elements {
		return nil, fmt.Errorf("cuda compact train resident gradient %q token metadata mismatch", ref.Name)
	}
	grad := a.grads[ref.Name]
	if grad == nil || grad.token != token || grad.ptr == 0 {
		return nil, fmt.Errorf("cuda compact train resident gradient %q is stale", ref.Name)
	}
	if grad.generation != ref.Generation || grad.stepID != ref.StepID || grad.elements != ref.Elements {
		return nil, fmt.Errorf("cuda compact train resident gradient %q generation/elements mismatch", ref.Name)
	}
	return grad, nil
}

func (a *CompactTrainAccelerator) validateHandleLocked(handle backend.CompactTrainHandle) error {
	if !a.stepActive {
		return fmt.Errorf("cuda compact train step is not active")
	}
	token, ok := handle.Token.(*compactTrainHandleToken)
	if !ok || token == nil {
		return fmt.Errorf("cuda compact train handle has invalid token")
	}
	if handle.Backend != eosartifact.BackendCUDA || token.Backend() != eosartifact.BackendCUDA {
		return fmt.Errorf("cuda compact train handle backend mismatch")
	}
	if handle.StepID != a.stepID || token.StepID() != a.stepID || token.StepID() != handle.StepID {
		return fmt.Errorf("cuda compact train handle step %d is stale, current %d", handle.StepID, a.stepID)
	}
	if a.arena == nil || a.arena.token != token || a.arena.generation != handle.Generation || token.Generation() != handle.Generation {
		return fmt.Errorf("cuda compact train handle is stale")
	}
	if handle.Shape != a.arena.shape {
		return fmt.Errorf("cuda compact train handle shape mismatch")
	}
	if !token.Alive() {
		return fmt.Errorf("cuda compact train handle already released")
	}
	return nil
}

func (a *CompactTrainAccelerator) consumeHandleLocked(handle backend.CompactTrainHandle) error {
	if err := a.validateHandleLocked(handle); err != nil {
		return err
	}
	token := handle.Token.(*compactTrainHandleToken)
	if !token.alive.CompareAndSwap(true, false) {
		return fmt.Errorf("cuda compact train handle already released")
	}
	a.arena.live = false
	a.stats.HandlesReleased++
	a.stats.LiveHandles--
	return nil
}

func (a *CompactTrainAccelerator) ensureBackwardWorkspaceLocked(arena *compactTrainArena) error {
	shape := arena.shape
	rows := shape.Batch * shape.Tokens
	alloc := func(dst *C.CUdeviceptr, elems int) error {
		if elems == 0 || *dst != 0 {
			return nil
		}
		ptr, err := a.device.allocFloat32(elems)
		if err != nil {
			return err
		}
		*dst = ptr
		arena.workspaceBytes += int64(elems * 4)
		return nil
	}
	if err := alloc(&arena.gradPooled, shape.Batch*shape.OutputDim); err != nil {
		return err
	}
	if err := alloc(&arena.gradOutputRows, rows*shape.OutputDim); err != nil {
		return err
	}
	if err := alloc(&arena.gradNormalized, rows*shape.ModelDim); err != nil {
		return err
	}
	if err := alloc(&arena.gradHidden, rows*shape.ModelDim); err != nil {
		return err
	}
	a.stats.WorkspaceArenaBytes = arena.workspaceBytes
	return nil
}

func (a *CompactTrainAccelerator) residentGradientRefsLocked() []backend.ResidentGradientRef {
	refs := make([]backend.ResidentGradientRef, 0, len(a.grads))
	names := a.requiredResidentNamesForCurrentStepLocked()
	for _, name := range names {
		grad := a.grads[name]
		if grad == nil {
			continue
		}
		refs = append(refs, backend.ResidentGradientRef{
			Name:       name,
			Backend:    eosartifact.BackendCUDA,
			Token:      grad.token,
			Elements:   grad.elements,
			Generation: grad.generation,
			StepID:     grad.stepID,
		})
	}
	return refs
}

func (a *CompactTrainAccelerator) requiredResidentNamesForCurrentStepLocked() []string {
	if a.arena != nil {
		return a.requiredResidentNames(a.arena.shape)
	}
	names := make([]string, 0, len(a.grads))
	for name := range a.grads {
		names = append(names, name)
	}
	return names
}

func tensorShapeForError(t *backend.Tensor) []int {
	if t == nil {
		return nil
	}
	return t.Shape
}

func (a *CompactTrainAccelerator) launchFinalProjectionGrad(normalized, gradRows, gradOutProjection C.CUdeviceptr, rows, modelDim, outDim int) error {
	grid, block, err := checkedLaunch1D("cuda compact train final projection grad", modelDim*outDim, 128)
	if err != nil {
		return err
	}
	var errStr *C.char
	if C.eosCudaLaunchCompactTrainFinalProjectionGrad(a.device.ptr, a.trainKernels.finalProjectionGrad.ptr, grid, block, normalized, gradRows, gradOutProjection, C.int(rows), C.int(modelDim), C.int(outDim), &errStr) != 0 {
		return cStringError(errStr)
	}
	a.recordKernelLaunch()
	return nil
}

func (a *CompactTrainAccelerator) launchFinalHiddenGrad(projected, normalized, outputProjection, gradPooled, masks, active, gradRows, gradNormalized, gradHidden C.CUdeviceptr, shape backend.CompactForwardShape) error {
	grid, block, err := checkedLaunch1D("cuda compact train final hidden grad", shape.Batch*shape.Tokens*shape.ModelDim, 128)
	if err != nil {
		return err
	}
	hasProjection := 0
	if shape.HasOutputProjection {
		hasProjection = 1
	}
	var errStr *C.char
	if C.eosCudaLaunchCompactTrainFinalHiddenGrad(a.device.ptr, a.trainKernels.finalHiddenGrad.ptr, grid, block, projected, normalized, outputProjection, gradPooled, masks, active, gradRows, gradNormalized, gradHidden, C.int(shape.Batch), C.int(shape.Tokens), C.int(shape.ModelDim), C.int(shape.OutputDim), C.int(hasProjection), &errStr) != 0 {
		return cStringError(errStr)
	}
	a.recordKernelLaunch()
	return nil
}

func (a *CompactTrainAccelerator) RunCompactTrainForward(req backend.CompactTrainForwardRequest) (backend.CompactTrainForwardResult, error) {
	if a == nil || a.CompactForwardAccelerator == nil {
		return backend.CompactTrainForwardResult{}, fmt.Errorf("cuda compact train accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	start := time.Now()
	result, err := a.runCompactTrainForwardLocked(req)
	if err != nil {
		return backend.CompactTrainForwardResult{}, err
	}
	a.stats.ForwardCalls++
	a.stats.ForwardNanos += time.Since(start).Nanoseconds()
	return result, nil
}

func (a *CompactTrainAccelerator) runCompactTrainForwardLocked(req backend.CompactTrainForwardRequest) (backend.CompactTrainForwardResult, error) {
	if a.device == nil || a.closed {
		return backend.CompactTrainForwardResult{}, fmt.Errorf("cuda compact train accelerator is closed")
	}
	if !a.stepActive {
		return backend.CompactTrainForwardResult{}, fmt.Errorf("cuda compact train step is not active")
	}
	if req.StepID != a.stepID {
		return backend.CompactTrainForwardResult{}, fmt.Errorf("cuda compact train forward step %d is stale, current %d", req.StepID, a.stepID)
	}
	if err := a.preflightCompactTrainForwardLocked(req); err != nil {
		return backend.CompactTrainForwardResult{}, err
	}
	if a.arena != nil && a.arena.live {
		return backend.CompactTrainForwardResult{}, fmt.Errorf("cuda compact train arena already has a live handle")
	}
	unlocks, err := a.lockBridgeTokens()
	if err != nil {
		return backend.CompactTrainForwardResult{}, err
	}
	defer unlockResidentBridgeTokens(unlocks)
	arena, err := a.prepareArenaLocked(req.Shape)
	if err != nil {
		return backend.CompactTrainForwardResult{}, err
	}
	arena.generation++
	shape := req.Shape
	geluFast, _ := compactForwardGELUFast(req.GELUMode)
	launchesBefore := a.CompactForwardAccelerator.stats.KernelLaunches
	syncsBefore := a.CompactForwardAccelerator.stats.KernelSynchronizations
	B, T, D, H, L, O := shape.Batch, shape.Tokens, shape.ModelDim, shape.FFNDim, shape.Layers, shape.OutputDim
	rows := B * T
	tokensFlat := flattenInt32(req.Tokens)
	masksFlat := flattenInt32(req.Masks)
	if err := a.replaceUploadedInt32(&arena.tokens, tokensFlat); err != nil {
		return backend.CompactTrainForwardResult{}, err
	}
	if err := a.replaceUploadedInt32(&arena.masks, masksFlat); err != nil {
		return backend.CompactTrainForwardResult{}, err
	}
	if err := a.replaceUploadedInt32(&arena.roles, req.Roles); err != nil {
		return backend.CompactTrainForwardResult{}, err
	}
	if err := a.replaceUploadedInt32(&arena.status, []int32{0}); err != nil {
		return backend.CompactTrainForwardResult{}, err
	}
	uploaded := int64((len(tokensFlat) + len(masksFlat) + len(req.Roles) + 1) * 4)
	if err := a.launchGather(a.bindings.token, a.bindings.role, arena.tokens, arena.roles, arena.input, arena.status, shape); err != nil {
		return backend.CompactTrainForwardResult{}, err
	}
	current := arena.input
	for layerIdx := 0; layerIdx < L; layerIdx++ {
		layer := a.bindings.layer[layerIdx]
		saved := &arena.layers[layerIdx]
		saved.input = current
		attnOut := arena.modelScratch[0]
		ffnOut := arena.modelScratch[1]
		if err := a.launchMM(current, layer.q, saved.attnQ, rows, D, D); err != nil {
			return backend.CompactTrainForwardResult{}, err
		}
		if err := a.launchMM(current, layer.k, saved.attnK, rows, D, D); err != nil {
			return backend.CompactTrainForwardResult{}, err
		}
		if err := a.launchMM(current, layer.v, saved.attnV, rows, D, D); err != nil {
			return backend.CompactTrainForwardResult{}, err
		}
		if err := a.launchAttention(saved.attnQ, saved.attnK, saved.attnV, arena.masks, saved.attnScores, saved.attnMixed, shape); err != nil {
			return backend.CompactTrainForwardResult{}, err
		}
		if err := a.launchMM(saved.attnMixed, layer.o, attnOut, rows, D, D); err != nil {
			return backend.CompactTrainForwardResult{}, err
		}
		if err := a.launchResidualLayerNorm(attnOut, current, saved.hidden, saved.attnResidual, rows, D); err != nil {
			return backend.CompactTrainForwardResult{}, err
		}
		if err := a.launchMM(saved.hidden, layer.up, saved.ffnHidden, rows, D, H); err != nil {
			return backend.CompactTrainForwardResult{}, err
		}
		if err := a.launchGELU(saved.ffnHidden, saved.activated, rows*H, geluFast); err != nil {
			return backend.CompactTrainForwardResult{}, err
		}
		if err := a.launchMM(saved.activated, layer.down, ffnOut, rows, H, D); err != nil {
			return backend.CompactTrainForwardResult{}, err
		}
		if err := a.launchResidualLayerNorm(ffnOut, saved.hidden, saved.projected, saved.ffnResidual, rows, D); err != nil {
			return backend.CompactTrainForwardResult{}, err
		}
		current = saved.projected
	}
	firstPooled := arena.finalPooled
	if shape.HasOutputProjection {
		firstPooled = arena.preProjectionPooled
	}
	if err := a.launchFinalize(current, arena.masks, arena.finalNorm, firstPooled, arena.active, B, T, D, true); err != nil {
		return backend.CompactTrainForwardResult{}, err
	}
	outputRows := arena.finalNorm
	if shape.HasOutputProjection {
		outputRows = arena.outputRows
		if err := a.launchMM(arena.finalNorm, a.bindings.out, outputRows, rows, D, O); err != nil {
			return backend.CompactTrainForwardResult{}, err
		}
		if err := a.launchFinalize(outputRows, arena.masks, outputRows, arena.finalPooled, arena.active, B, T, O, false); err != nil {
			return backend.CompactTrainForwardResult{}, err
		}
	}
	_ = outputRows
	status := []int32{0}
	if err := a.device.downloadInt32(status, arena.status); err != nil {
		return backend.CompactTrainForwardResult{}, err
	}
	if status[0] != 0 {
		return backend.CompactTrainForwardResult{}, fmt.Errorf("cuda compact train gather status %d", status[0])
	}
	pooled := make([]float32, B*O)
	if err := a.device.downloadFloat32(pooled, arena.finalPooled); err != nil {
		return backend.CompactTrainForwardResult{}, err
	}
	activeCounts := make([]int32, B)
	if err := a.device.downloadInt32(activeCounts, arena.active); err != nil {
		return backend.CompactTrainForwardResult{}, err
	}
	token := &compactTrainHandleToken{backend: eosartifact.BackendCUDA, generation: arena.generation, stepID: a.stepID, id: a.nextHandleID + 1}
	token.alive.Store(true)
	a.nextHandleID = token.id
	arena.token = token
	arena.live = true
	pooledBytes := int64(len(pooled) * 4)
	statusBytes := int64(4)
	activeBytes := int64(len(activeCounts) * 4)
	_, packedFloats, _ := compactForwardPackedLayout(shape)
	a.stats.HandlesCreated++
	a.stats.LiveHandles++
	a.stats.UploadedBytes += uploaded
	a.stats.DownloadedBytes += pooledBytes + statusBytes + activeBytes
	a.stats.PooledDownloadedBytes += pooledBytes
	a.stats.StatusDownloadedBytes += statusBytes + activeBytes
	a.stats.PackedBytesAvoided += int64(packedFloats * 4)
	forwardLaunches := a.CompactForwardAccelerator.stats.KernelLaunches - launchesBefore
	forwardSyncs := a.CompactForwardAccelerator.stats.KernelSynchronizations - syncsBefore
	a.stats.KernelLaunches += forwardLaunches
	a.stats.KernelSynchronizations += forwardSyncs
	a.stats.LastShape = shape
	a.stats.LastForwardLaunches = forwardLaunches
	a.stats.LastForwardSyncs = forwardSyncs
	a.stats.ActivationArenaBytes = arena.bytes
	a.stats.WorkspaceArenaBytes = 0
	return backend.CompactTrainForwardResult{
		Handle: backend.CompactTrainHandle{
			Backend:    eosartifact.BackendCUDA,
			Token:      token,
			Shape:      shape,
			Generation: token.generation,
			StepID:     token.stepID,
		},
		Pooled:       backend.NewTensorF32([]int{B, O}, pooled),
		ActiveCounts: activeCounts,
	}, nil
}

func (a *CompactTrainAccelerator) ReleaseCompactTrainHandle(handle backend.CompactTrainHandle) error {
	if a == nil || a.CompactForwardAccelerator == nil {
		return fmt.Errorf("cuda compact train accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.device == nil || a.closed {
		return fmt.Errorf("cuda compact train accelerator is closed")
	}
	return a.consumeHandleLocked(handle)
}

func (a *CompactTrainAccelerator) prepareArenaLocked(shape backend.CompactForwardShape) (*compactTrainArena, error) {
	if a.arena != nil && a.arena.shape == shape {
		return a.arena, nil
	}
	a.releaseArenaLocked()
	arena := &compactTrainArena{shape: shape}
	B, T, D, H := shape.Batch, shape.Tokens, shape.ModelDim, shape.FFNDim
	rows := B * T
	modelElems := rows * D
	ffnElems := rows * H
	scoreElems := B * shape.Heads * T * T
	allocF := func(dst *C.CUdeviceptr, elems int) error {
		ptr, err := a.device.allocFloat32(elems)
		if err != nil {
			return err
		}
		*dst = ptr
		arena.bytes += int64(elems * 4)
		return nil
	}
	allocI := func(dst *C.CUdeviceptr, elems int) error {
		ptr, err := a.device.allocInt32(elems)
		if err != nil {
			return err
		}
		*dst = ptr
		arena.bytes += int64(elems * 4)
		return nil
	}
	if err := allocF(&arena.input, modelElems); err != nil {
		a.freeArena(arena)
		return nil, err
	}
	if err := allocI(&arena.active, B); err != nil {
		a.freeArena(arena)
		return nil, err
	}
	if err := allocF(&arena.finalNorm, modelElems); err != nil {
		a.freeArena(arena)
		return nil, err
	}
	if shape.HasOutputProjection {
		if err := allocF(&arena.preProjectionPooled, B*D); err != nil {
			a.freeArena(arena)
			return nil, err
		}
		if err := allocF(&arena.outputRows, rows*shape.OutputDim); err != nil {
			a.freeArena(arena)
			return nil, err
		}
	}
	if err := allocF(&arena.finalPooled, B*shape.OutputDim); err != nil {
		a.freeArena(arena)
		return nil, err
	}
	for i := 0; i < 2; i++ {
		var ptr C.CUdeviceptr
		if err := allocF(&ptr, modelElems); err != nil {
			a.freeArena(arena)
			return nil, err
		}
		arena.modelScratch = append(arena.modelScratch, ptr)
	}
	arena.layers = make([]compactTrainLayerArena, shape.Layers)
	for i := range arena.layers {
		layer := &arena.layers[i]
		for _, dst := range []*C.CUdeviceptr{&layer.hidden, &layer.attnQ, &layer.attnK, &layer.attnV, &layer.attnMixed, &layer.attnResidual, &layer.ffnResidual, &layer.projected} {
			if err := allocF(dst, modelElems); err != nil {
				a.freeArena(arena)
				return nil, err
			}
		}
		if err := allocF(&layer.attnScores, scoreElems); err != nil {
			a.freeArena(arena)
			return nil, err
		}
		if err := allocF(&layer.ffnHidden, ffnElems); err != nil {
			a.freeArena(arena)
			return nil, err
		}
		if err := allocF(&layer.activated, ffnElems); err != nil {
			a.freeArena(arena)
			return nil, err
		}
	}
	a.arena = arena
	a.stats.ActivationArenaBytes = arena.bytes
	return arena, nil
}

func (a *CompactTrainAccelerator) replaceUploadedInt32(dst *C.CUdeviceptr, data []int32) error {
	if *dst != 0 {
		_ = a.device.freeBuffer(*dst)
		*dst = 0
	}
	ptr, err := a.device.uploadInt32(data)
	if err != nil {
		return err
	}
	*dst = ptr
	return nil
}

func (a *CompactTrainAccelerator) releaseArenaLocked() {
	if a.arena != nil {
		a.freeArena(a.arena)
		a.arena = nil
	}
	a.stats.ActivationArenaBytes = 0
	a.stats.WorkspaceArenaBytes = 0
	a.stats.LiveHandles = 0
}

func (a *CompactTrainAccelerator) releaseGradientsLocked() {
	if a == nil || a.device == nil {
		return
	}
	for _, grad := range a.grads {
		if grad == nil {
			continue
		}
		if grad.ptr != 0 {
			_ = a.device.freeBuffer(grad.ptr)
			grad.ptr = 0
		}
	}
	a.grads = map[string]*compactTrainGradient{}
	a.stats.ResidentGradBytes = 0
}

func (a *CompactTrainAccelerator) freeArena(arena *compactTrainArena) {
	if a == nil || a.device == nil || arena == nil {
		return
	}
	for _, ptr := range []C.CUdeviceptr{arena.tokens, arena.masks, arena.roles, arena.status, arena.input, arena.active, arena.finalNorm, arena.outputRows, arena.preProjectionPooled, arena.finalPooled, arena.gradPooled, arena.gradOutputRows, arena.gradNormalized, arena.gradHidden} {
		if ptr != 0 {
			_ = a.device.freeBuffer(ptr)
		}
	}
	for _, ptr := range arena.modelScratch {
		if ptr != 0 {
			_ = a.device.freeBuffer(ptr)
		}
	}
	for _, ptr := range arena.ffnScratch {
		if ptr != 0 {
			_ = a.device.freeBuffer(ptr)
		}
	}
	for i := range arena.layers {
		layer := &arena.layers[i]
		for _, ptr := range []C.CUdeviceptr{layer.hidden, layer.attnQ, layer.attnK, layer.attnV, layer.attnScores, layer.attnMixed, layer.attnResidual, layer.ffnHidden, layer.activated, layer.ffnResidual, layer.projected} {
			if ptr != 0 {
				_ = a.device.freeBuffer(ptr)
			}
		}
	}
	if arena.token != nil {
		arena.token.alive.Store(false)
	}
}
