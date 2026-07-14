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

static int eosCudaLaunchCompactGather(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr token, CUdeviceptr role, CUdeviceptr tokens, CUdeviceptr roles, CUdeviceptr out0, CUdeviceptr status, int batch, int seq, int modelDim, int vocab, int roleRows, int useRole, int useRoPE, char** err);
static int eosCudaLaunchCompactMatmul(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr lhs, CUdeviceptr rhs, CUdeviceptr out0, int rows, int inner, int cols, char** err);
static int eosCudaLaunchCompactAttention(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr q, CUdeviceptr k, CUdeviceptr v, CUdeviceptr masks, CUdeviceptr scores, CUdeviceptr mixed, int batch, int seq, int modelDim, int heads, int headDim, char** err);
static int eosCudaLaunchCompactResidualLayerNorm(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr src, CUdeviceptr residual, CUdeviceptr out0, CUdeviceptr residualOut, int rows, int cols, char** err);
static int eosCudaLaunchCompactGELU(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr src, CUdeviceptr dst, int elements, int fast, char** err);
static int eosCudaLaunchCompactFinalize(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr projected, CUdeviceptr masks, CUdeviceptr normalized, CUdeviceptr pooled, CUdeviceptr active, int batch, int seq, int width, int normalize, char** err);
static char* eos_compact_dup_cu_error(const char* prefix, CUresult res) {
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

static int eos_compact_launch(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, void** args, char** err) {
	CUresult res = cuCtxSetCurrent(rt->ctx);
	if (res != CUDA_SUCCESS) {
		*err = eos_compact_dup_cu_error("cuCtxSetCurrent", res);
		return 1;
	}
	res = cuLaunchKernel(kernel->function, grid, 1, 1, block, 1, 1, 0, rt->stream, args, 0);
	if (res != CUDA_SUCCESS) {
		*err = eos_compact_dup_cu_error("cuLaunchKernel", res);
		return 1;
	}
	return 0;
}

static int eosCudaLaunchCompactGather(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr token, CUdeviceptr role, CUdeviceptr tokens, CUdeviceptr roles, CUdeviceptr out0, CUdeviceptr status, int batch, int seq, int modelDim, int vocab, int roleRows, int useRole, int useRoPE, char** err) {
	void* args[] = {&token, &role, &tokens, &roles, &out0, &status, &batch, &seq, &modelDim, &vocab, &roleRows, &useRole, &useRoPE};
	return eos_compact_launch(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchCompactMatmul(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr lhs, CUdeviceptr rhs, CUdeviceptr out0, int rows, int inner, int cols, char** err) {
	void* args[] = {&lhs, &rhs, &out0, &rows, &inner, &cols};
	return eos_compact_launch(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchCompactAttention(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr q, CUdeviceptr k, CUdeviceptr v, CUdeviceptr masks, CUdeviceptr scores, CUdeviceptr mixed, int batch, int seq, int modelDim, int heads, int headDim, char** err) {
	void* args[] = {&q, &k, &v, &masks, &scores, &mixed, &batch, &seq, &modelDim, &heads, &headDim};
	return eos_compact_launch(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchCompactResidualLayerNorm(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr src, CUdeviceptr residual, CUdeviceptr out0, CUdeviceptr residualOut, int rows, int cols, char** err) {
	void* args[] = {&src, &residual, &out0, &residualOut, &rows, &cols};
	return eos_compact_launch(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchCompactGELU(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr src, CUdeviceptr dst, int elements, int fast, char** err) {
	void* args[] = {&src, &dst, &elements, &fast};
	return eos_compact_launch(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchCompactFinalize(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr projected, CUdeviceptr masks, CUdeviceptr normalized, CUdeviceptr pooled, CUdeviceptr active, int batch, int seq, int width, int normalize, char** err) {
	void* args[] = {&projected, &masks, &normalized, &pooled, &active, &batch, &seq, &width, &normalize};
	return eos_compact_launch(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchCompactPack(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr src, CUdeviceptr dst, int srcOffset, int dstOffset, int elements, char** err) {
	void* args[] = {&src, &dst, &srcOffset, &dstOffset, &elements};
	return eos_compact_launch(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchCompactPackInt(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr src, CUdeviceptr dst, int srcOffset, int dstOffset, int elements, char** err) {
	void* args[] = {&src, &dst, &srcOffset, &dstOffset, &elements};
	return eos_compact_launch(rt, kernel, grid, block, args, err);
}
*/
import "C"

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

const compactForwardKernelSource = `
extern "C" __global__ void eos_compact_gather(
    const float* token, const float* role, const int* tokens, const int* roles,
    float* out, int* status,
    int batch, int seq, int modelDim, int vocab, int roleRows, int useRole, int useRoPE
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * modelDim;
    if (idx >= total) return;
    int col = idx % modelDim;
    int row = idx / modelDim;
    int tok = tokens[row];
    if (tok < 0 || tok >= vocab) {
        status[0] = 1;
        out[idx] = 0.0f;
        return;
    }
    float v = token[tok * modelDim + col];
    if (useRole != 0) {
        int b = row / seq;
        int r = roles[b];
        if (r < 0 || r >= roleRows) {
            status[0] = 2;
            out[idx] = 0.0f;
            return;
        }
        v += role[r * modelDim + col];
    }
    out[idx] = v;
    if (useRoPE != 0) {
        int pairCol = col - (col & 1);
        if (pairCol + 1 < modelDim) {
            int pos = row % seq;
            float theta = (float)pos / powf(10000.0f, (float)pairCol / (float)modelDim);
            float c = cosf(theta);
            float s = sinf(theta);
            float x0 = token[tok * modelDim + pairCol];
            float x1 = token[tok * modelDim + pairCol + 1];
            if (useRole != 0) {
                int b = row / seq;
                int r = roles[b];
                x0 += role[r * modelDim + pairCol];
                x1 += role[r * modelDim + pairCol + 1];
            }
            if ((col & 1) == 0) {
                out[idx] = x0 * c - x1 * s;
            } else {
                out[idx] = x0 * s + x1 * c;
            }
        }
    }
}

extern "C" __global__ void eos_compact_matmul(
    const float* lhs, const float* rhs, float* out,
    int rows, int inner, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    int c = idx % cols;
    int r = idx / cols;
    float sum = 0.0f;
    for (int k = 0; k < inner; ++k) {
        sum += lhs[r * inner + k] * rhs[k * cols + c];
    }
    out[idx] = sum;
}

extern "C" __global__ void eos_compact_attention(
    const float* q, const float* k, const float* v, const int* masks,
    float* scores, float* mixed,
    int batch, int seq, int modelDim, int heads, int headDim
) {
    int job = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * seq;
    if (job >= total) return;
    int query = job % seq;
    int tmp = job / seq;
    int head = tmp % heads;
    int b = tmp / heads;
    int baseRows = b * seq;
    int headOffset = head * headDim;
    int scoreBase = ((b * heads + head) * seq + query) * seq;
    float scale = rsqrtf((float)headDim);
    float maxVal = -3.4028234663852886e38f;
    int active = 0;
    for (int key = 0; key < seq; ++key) {
        float sum = 0.0f;
        int qBase = (baseRows + query) * modelDim + headOffset;
        int kBase = (baseRows + key) * modelDim + headOffset;
        for (int c = 0; c < headDim; ++c) {
            sum += q[qBase + c] * k[kBase + c];
        }
        float val = sum * scale;
        if (masks[baseRows + key] == 0) {
            scores[scoreBase + key] = 0.0f;
            continue;
        }
        scores[scoreBase + key] = val;
        if (active == 0 || val > maxVal) maxVal = val;
        active++;
    }
    if (active == 0) {
        for (int c = 0; c < headDim; ++c) {
            mixed[(baseRows + query) * modelDim + headOffset + c] = 0.0f;
        }
        return;
    }
    float sumExp = 0.0f;
    for (int key = 0; key < seq; ++key) {
        if (masks[baseRows + key] == 0) continue;
        float e = expf(scores[scoreBase + key] - maxVal);
        scores[scoreBase + key] = e;
        sumExp += e;
    }
    float inv = sumExp == 0.0f ? 0.0f : 1.0f / sumExp;
    for (int key = 0; key < seq; ++key) {
        if (masks[baseRows + key] != 0) scores[scoreBase + key] *= inv;
    }
    for (int c = 0; c < headDim; ++c) {
        float acc = 0.0f;
        for (int key = 0; key < seq; ++key) {
            acc += scores[scoreBase + key] * v[(baseRows + key) * modelDim + headOffset + c];
        }
        mixed[(baseRows + query) * modelDim + headOffset + c] = acc;
    }
}

extern "C" __global__ void eos_compact_residual_layernorm(
    const float* src, const float* residual, float* out, float* residualOut,
    int rows, int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    int base = row * cols;
    float mean = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float v = src[base + c] + residual[base + c];
        residualOut[base + c] = v;
        mean += v;
    }
    mean /= (float)cols;
    float variance = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float d = residualOut[base + c] - mean;
        variance += d * d;
    }
    variance /= (float)cols;
    float invStd = rsqrtf(variance + 1.0e-5f);
    for (int c = 0; c < cols; ++c) {
        out[base + c] = (residualOut[base + c] - mean) * invStd;
    }
}

static __device__ float eos_compact_fast_tanh(float x) {
    if (x >= 3.0f) return 1.0f;
    if (x <= -3.0f) return -1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

extern "C" __global__ void eos_compact_gelu(const float* src, float* dst, int elements, int fast) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) return;
    float x = src[idx];
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    float t = fast != 0 ? eos_compact_fast_tanh(inner) : tanhf(inner);
    dst[idx] = 0.5f * x * (1.0f + t);
}

extern "C" __global__ void eos_compact_finalize(
    const float* projected, const int* masks, float* normalized, float* pooled, int* active,
    int batch, int seq, int width, int normalizeRows
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    int count = 0;
    for (int c = 0; c < width; ++c) pooled[b * width + c] = 0.0f;
    for (int row = 0; row < seq; ++row) {
        int base = (b * seq + row) * width;
        float normSq = 0.0f;
        for (int c = 0; c < width; ++c) {
            float v = projected[base + c];
            normSq += v * v;
        }
        float norm = sqrtf(normSq);
        for (int c = 0; c < width; ++c) {
            float v = projected[base + c];
            if (normalizeRows != 0 && norm != 0.0f) v /= norm;
            normalized[base + c] = v;
            if (masks[b * seq + row] != 0) pooled[b * width + c] += v;
        }
        if (masks[b * seq + row] != 0) count++;
    }
    active[b] = count;
    if (count != 0) {
        float inv = 1.0f / (float)count;
        for (int c = 0; c < width; ++c) pooled[b * width + c] *= inv;
    }
}

extern "C" __global__ void eos_compact_pack(const float* src, float* dst, int srcOffset, int dstOffset, int elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) return;
    dst[dstOffset + idx] = src[srcOffset + idx];
}

extern "C" __global__ void eos_compact_pack_int(const int* src, float* dst, int srcOffset, int dstOffset, int elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) return;
    dst[dstOffset + idx] = (float)src[srcOffset + idx];
}
`

type CompactForwardAccelerator struct {
	mu                                      sync.Mutex
	device                                  *deviceRuntime
	kernels                                 compactForwardKernels
	bindings                                compactForwardBindings
	stats                                   backend.CompactForwardAcceleratorStats
	useRoPE                                 bool
	syncEachLaunch                          bool
	launchesSinceBoundary                   int64
	debugForceForwardFailureAfterFirstLayer bool
	tokenName                               string
	roleName                                string
	layers                                  []CompactForwardLayerNames
	outName                                 string
	bridged                                 map[string]*optimizerResidentParameterToken
}

type CompactForwardLayerNames struct {
	AttentionQ string
	AttentionK string
	AttentionV string
	AttentionO string
	FFNUp      string
	FFNDown    string
}

type compactForwardKernels struct {
	gather *auxKernel
	mm     *auxKernel
	attn   *auxKernel
	ln     *auxKernel
	gelu   *auxKernel
	final  *auxKernel
	pack   *auxKernel
	packI  *auxKernel
}

type compactForwardBindings struct {
	token residentMatrix
	role  residentMatrix
	layer []compactForwardLayerBinding
	out   residentMatrix
}

type compactForwardLayerBinding struct {
	q    residentMatrix
	k    residentMatrix
	v    residentMatrix
	o    residentMatrix
	up   residentMatrix
	down residentMatrix
}

func init() {
	backend.RegisterCompactForwardAccelerator(eosartifact.BackendCUDA, func() (backend.CompactForwardAccelerator, error) {
		return NewCompactForwardAccelerator()
	})
}

func NewCompactForwardAccelerator() (*CompactForwardAccelerator, error) {
	device, err := newDeviceRuntime()
	if err != nil || device == nil {
		return nil, err
	}
	compile := func(entry string) (*auxKernel, error) {
		k, err := device.compileAuxKernel(compactForwardKernelSource, entry)
		if err != nil {
			device.close()
			return nil, err
		}
		return k, nil
	}
	gather, err := compile("eos_compact_gather")
	if err != nil {
		return nil, err
	}
	mm, err := compile("eos_compact_matmul")
	if err != nil {
		return nil, err
	}
	attn, err := compile("eos_compact_attention")
	if err != nil {
		return nil, err
	}
	ln, err := compile("eos_compact_residual_layernorm")
	if err != nil {
		return nil, err
	}
	gelu, err := compile("eos_compact_gelu")
	if err != nil {
		return nil, err
	}
	final, err := compile("eos_compact_finalize")
	if err != nil {
		return nil, err
	}
	pack, err := compile("eos_compact_pack")
	if err != nil {
		return nil, err
	}
	packI, err := compile("eos_compact_pack_int")
	if err != nil {
		return nil, err
	}
	return &CompactForwardAccelerator{
		device:         device,
		kernels:        compactForwardKernels{gather: gather, mm: mm, attn: attn, ln: ln, gelu: gelu, final: final, pack: pack, packI: packI},
		syncEachLaunch: cudaEnvFlagEnabled("EOS_CUDA_COMPACT_SYNC_EACH_LAUNCH"),
		bridged:        map[string]*optimizerResidentParameterToken{},
	}, nil
}

func (a *CompactForwardAccelerator) Backend() eosartifact.BackendKind { return eosartifact.BackendCUDA }

func (a *CompactForwardAccelerator) Close() {
	if a == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.device == nil {
		return
	}
	a.device.destroyAuxKernel(a.kernels.gather)
	a.device.destroyAuxKernel(a.kernels.mm)
	a.device.destroyAuxKernel(a.kernels.attn)
	a.device.destroyAuxKernel(a.kernels.ln)
	a.device.destroyAuxKernel(a.kernels.gelu)
	a.device.destroyAuxKernel(a.kernels.final)
	a.device.destroyAuxKernel(a.kernels.pack)
	a.device.destroyAuxKernel(a.kernels.packI)
	a.device.close()
	a.device = nil
}

func (a *CompactForwardAccelerator) Stats() backend.CompactForwardAcceleratorStats {
	if a == nil {
		return backend.CompactForwardAcceleratorStats{}
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.stats
}

func (a *CompactForwardAccelerator) ConfigureCompactForward(names []backend.CompactForwardLayerConfig, tokenName, roleName, outputProjectionName string, useRoPE bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.layers = append(a.layers[:0], compactForwardLayerNamesFromBackend(names)...)
	a.tokenName = tokenName
	a.roleName = roleName
	a.outName = outputProjectionName
	a.useRoPE = useRoPE
}

func (a *CompactForwardAccelerator) Configure(names []CompactForwardLayerNames, tokenName, roleName, outputProjectionName string, useRoPE bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.layers = append(a.layers[:0], names...)
	a.tokenName = tokenName
	a.roleName = roleName
	a.outName = outputProjectionName
	a.useRoPE = useRoPE
}

func compactForwardLayerNamesFromBackend(names []backend.CompactForwardLayerConfig) []CompactForwardLayerNames {
	out := make([]CompactForwardLayerNames, len(names))
	for i, name := range names {
		out[i] = CompactForwardLayerNames{
			AttentionQ: name.AttentionQ,
			AttentionK: name.AttentionK,
			AttentionV: name.AttentionV,
			AttentionO: name.AttentionO,
			FFNUp:      name.FFNUp,
			FFNDown:    name.FFNDown,
		}
	}
	return out
}

func (a *CompactForwardAccelerator) BindCompactForwardResident(name string, tensor *backend.Tensor, ref backend.OptimizerResidentParameter) error {
	return a.BindResident(name, tensor, ref)
}

func (a *CompactForwardAccelerator) BindResident(name string, tensor *backend.Tensor, ref backend.OptimizerResidentParameter) error {
	if a == nil {
		return fmt.Errorf("cuda compact forward accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.bindResidentLocked(name, tensor, ref)
}

func (a *CompactForwardAccelerator) PreflightCompactForward(req backend.CompactForwardRequest) error {
	if a == nil {
		return fmt.Errorf("cuda compact forward accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.device == nil {
		return fmt.Errorf("cuda compact forward accelerator is closed")
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
		return fmt.Errorf("cuda compact forward layer bindings %d, want %d", len(a.layers), shape.Layers)
	}
	if err := a.validateBindings(shape); err != nil {
		return err
	}
	if err := a.preflightResidentRefs(req.ResidentRefs, shape); err != nil {
		return err
	}
	_, _, err := compactForwardPackedLayout(shape)
	return err
}

func (a *CompactForwardAccelerator) bindResidentLocked(name string, tensor *backend.Tensor, ref backend.OptimizerResidentParameter) error {
	if a.device == nil {
		return fmt.Errorf("cuda compact forward accelerator is closed")
	}
	if name == "" {
		return fmt.Errorf("cuda compact forward resident binding requires a name")
	}
	if ref.Backend != eosartifact.BackendCUDA {
		return fmt.Errorf("cuda compact forward binding %q has backend %q", name, ref.Backend)
	}
	token, ok := ref.Token.(*optimizerResidentParameterToken)
	if !ok || token == nil {
		return fmt.Errorf("cuda compact forward binding %q has invalid optimizer token", name)
	}
	state, unlock, err := token.lockCurrent()
	if err != nil {
		return err
	}
	defer unlock()
	if tensor == nil || len(tensor.Shape) != 2 {
		return fmt.Errorf("cuda compact forward binding %q must be rank-2", name)
	}
	if ref.Elements != len(tensor.F32) || state.elements != len(tensor.F32) {
		return fmt.Errorf("cuda compact forward binding %q elements do not match tensor", name)
	}
	a.setBinding(name, residentMatrix{ptr: state.param, rows: tensor.Shape[0], cols: tensor.Shape[1], elements: len(tensor.F32)})
	a.bridged[name] = token
	return nil
}

func (a *CompactForwardAccelerator) setBinding(name string, m residentMatrix) {
	if name == a.tokenName {
		a.bindings.token = m
	}
	if name == a.roleName {
		a.bindings.role = m
	}
	if name == a.outName {
		a.bindings.out = m
	}
	if len(a.bindings.layer) != len(a.layers) {
		a.bindings.layer = make([]compactForwardLayerBinding, len(a.layers))
	}
	for i, layer := range a.layers {
		switch name {
		case layer.AttentionQ:
			a.bindings.layer[i].q = m
		case layer.AttentionK:
			a.bindings.layer[i].k = m
		case layer.AttentionV:
			a.bindings.layer[i].v = m
		case layer.AttentionO:
			a.bindings.layer[i].o = m
		case layer.FFNUp:
			a.bindings.layer[i].up = m
		case layer.FFNDown:
			a.bindings.layer[i].down = m
		}
	}
}

func (a *CompactForwardAccelerator) RunCompactForward(req backend.CompactForwardRequest) (backend.CompactForwardResult, error) {
	if a == nil {
		return backend.CompactForwardResult{}, fmt.Errorf("cuda compact forward accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	start := time.Now()
	result, err := a.runCompactForwardLocked(req)
	if err != nil {
		err = a.drainKernelError(err)
		a.stats.UnhandledCalls++
		return backend.CompactForwardResult{}, err
	}
	a.stats.RunCalls++
	a.stats.RunNanos += time.Since(start).Nanoseconds()
	return result, nil
}

func (a *CompactForwardAccelerator) runCompactForwardLocked(req backend.CompactForwardRequest) (result backend.CompactForwardResult, err error) {
	if a.device == nil {
		return backend.CompactForwardResult{}, fmt.Errorf("cuda compact forward accelerator is closed")
	}
	shape := req.Shape
	if err := validateCompactForwardShape(shape); err != nil {
		return backend.CompactForwardResult{}, err
	}
	if err := validateCompactForwardInputs(shape, req.Tokens, req.Masks, req.Roles); err != nil {
		return backend.CompactForwardResult{}, err
	}
	geluFast, err := compactForwardGELUFast(req.GELUMode)
	if err != nil {
		return backend.CompactForwardResult{}, err
	}
	if len(a.layers) != shape.Layers || len(a.bindings.layer) != shape.Layers {
		return backend.CompactForwardResult{}, fmt.Errorf("cuda compact forward layer bindings %d, want %d", len(a.layers), shape.Layers)
	}
	if err := a.validateBindings(shape); err != nil {
		return backend.CompactForwardResult{}, err
	}
	if err := a.preflightResidentRefs(req.ResidentRefs, shape); err != nil {
		return backend.CompactForwardResult{}, err
	}
	unlocks, err := a.lockBridgeTokens()
	if err != nil {
		return backend.CompactForwardResult{}, err
	}
	defer unlockResidentBridgeTokens(unlocks)

	layout, total, err := compactForwardPackedLayout(shape)
	if err != nil {
		return backend.CompactForwardResult{}, err
	}
	launchCountBefore := a.stats.KernelLaunches
	syncCountBefore := a.stats.KernelSynchronizations
	B, T, D, H, L, O := shape.Batch, shape.Tokens, shape.ModelDim, shape.FFNDim, shape.Layers, shape.OutputDim
	if _, err := compactForwardMemoryEstimateBytes(shape, a.bindings.token.rows, a.bindings.role.rows); err != nil {
		return backend.CompactForwardResult{}, err
	}
	rows := B * T
	modelElems := rows * D
	ffnElems := rows * H
	scoreElems := B * shape.Heads * T * T
	outputElems := rows * O
	tokensFlat := flattenInt32(req.Tokens)
	masksFlat := flattenInt32(req.Masks)
	tokBuf, err := a.device.uploadInt32(tokensFlat)
	if err != nil {
		return backend.CompactForwardResult{}, err
	}
	defer a.device.freeBuffer(tokBuf)
	maskBuf, err := a.device.uploadInt32(masksFlat)
	if err != nil {
		return backend.CompactForwardResult{}, err
	}
	defer a.device.freeBuffer(maskBuf)
	roleBuf, err := a.device.uploadInt32(req.Roles)
	if err != nil {
		return backend.CompactForwardResult{}, err
	}
	defer a.device.freeBuffer(roleBuf)
	statusBuf, err := a.device.uploadInt32([]int32{0})
	if err != nil {
		return backend.CompactForwardResult{}, err
	}
	defer a.device.freeBuffer(statusBuf)
	uploaded := int64((len(tokensFlat) + len(masksFlat) + len(req.Roles) + 1) * 4)

	input, err := a.device.allocFloat32(modelElems)
	if err != nil {
		return backend.CompactForwardResult{}, err
	}
	defer a.device.freeBuffer(input)
	current := input
	packed, err := a.device.allocFloat32(total)
	if err != nil {
		return backend.CompactForwardResult{}, err
	}
	defer a.device.freeBuffer(packed)
	downloadPacked, err := a.device.allocFloat32(total + 1)
	if err != nil {
		return backend.CompactForwardResult{}, err
	}
	defer a.device.freeBuffer(downloadPacked)

	modelBufs := make([]C.CUdeviceptr, 0, 11)
	ffnBufs := make([]C.CUdeviceptr, 0, 2)
	for i := 0; i < 13; i++ {
		buf, err := a.device.allocFloat32(modelElems)
		if err != nil {
			return backend.CompactForwardResult{}, err
		}
		defer a.device.freeBuffer(buf)
		modelBufs = append(modelBufs, buf)
	}
	for i := 0; i < 2; i++ {
		buf, err := a.device.allocFloat32(ffnElems)
		if err != nil {
			return backend.CompactForwardResult{}, err
		}
		defer a.device.freeBuffer(buf)
		ffnBufs = append(ffnBufs, buf)
	}
	scores, err := a.device.allocFloat32(scoreElems)
	if err != nil {
		return backend.CompactForwardResult{}, err
	}
	defer a.device.freeBuffer(scores)
	finalNorm, err := a.device.allocFloat32(modelElems)
	if err != nil {
		return backend.CompactForwardResult{}, err
	}
	defer a.device.freeBuffer(finalNorm)
	outputRows := finalNorm
	if shape.HasOutputProjection {
		outputRows, err = a.device.allocFloat32(outputElems)
		if err != nil {
			return backend.CompactForwardResult{}, err
		}
		defer a.device.freeBuffer(outputRows)
	}
	finalPooled, err := a.device.allocFloat32(B * O)
	if err != nil {
		return backend.CompactForwardResult{}, err
	}
	defer a.device.freeBuffer(finalPooled)
	activeBuf, err := a.device.allocInt32(B)
	if err != nil {
		return backend.CompactForwardResult{}, err
	}
	defer a.device.freeBuffer(activeBuf)
	defer func() {
		if err != nil {
			err = a.drainKernelError(err)
		}
	}()

	if err := a.launchGather(a.bindings.token, a.bindings.role, tokBuf, roleBuf, input, statusBuf, shape); err != nil {
		return backend.CompactForwardResult{}, err
	}
	for layerIdx := 0; layerIdx < L; layerIdx++ {
		layer := a.bindings.layer[layerIdx]
		q, k, v := modelBufs[0], modelBufs[1], modelBufs[2]
		mixed, attnOut, attnResidual, hidden := modelBufs[3], modelBufs[4], modelBufs[5], modelBufs[6]
		ffnHidden, activated := ffnBufs[0], ffnBufs[1]
		ffnOut, ffnResidual := modelBufs[7], modelBufs[8]
		projected := modelBufs[9]
		if current == projected {
			projected = modelBufs[10]
		}
		layerNormalized, layerPooled := modelBufs[11], modelBufs[12]
		if err := a.launchMM(current, layer.q, q, rows, D, D); err != nil {
			return backend.CompactForwardResult{}, err
		}
		if err := a.launchMM(current, layer.k, k, rows, D, D); err != nil {
			return backend.CompactForwardResult{}, err
		}
		if err := a.launchMM(current, layer.v, v, rows, D, D); err != nil {
			return backend.CompactForwardResult{}, err
		}
		if err := a.launchAttention(q, k, v, maskBuf, scores, mixed, shape); err != nil {
			return backend.CompactForwardResult{}, err
		}
		if err := a.launchMM(mixed, layer.o, attnOut, rows, D, D); err != nil {
			return backend.CompactForwardResult{}, err
		}
		if err := a.launchResidualLayerNorm(attnOut, current, hidden, attnResidual, rows, D); err != nil {
			return backend.CompactForwardResult{}, err
		}
		if err := a.launchMM(hidden, layer.up, ffnHidden, rows, D, H); err != nil {
			return backend.CompactForwardResult{}, err
		}
		if err := a.launchGELU(ffnHidden, activated, ffnElems, geluFast); err != nil {
			return backend.CompactForwardResult{}, err
		}
		if err := a.launchMM(activated, layer.down, ffnOut, rows, H, D); err != nil {
			return backend.CompactForwardResult{}, err
		}
		if err := a.launchResidualLayerNorm(ffnOut, hidden, projected, ffnResidual, rows, D); err != nil {
			return backend.CompactForwardResult{}, err
		}
		if err := a.launchFinalize(projected, maskBuf, layerNormalized, layerPooled, activeBuf, B, T, D, true); err != nil {
			return backend.CompactForwardResult{}, err
		}
		srcs := map[string]C.CUdeviceptr{
			"input":        current,
			"hidden":       hidden,
			"attnQ":        q,
			"attnK":        k,
			"attnV":        v,
			"attnScores":   scores,
			"attnMixed":    mixed,
			"attnOutput":   attnOut,
			"attnResidual": attnResidual,
			"ffnHidden":    ffnHidden,
			"activated":    activated,
			"ffnOutput":    ffnOut,
			"ffnResidual":  ffnResidual,
			"projected":    projected,
			"normalized":   layerNormalized,
			"pooled":       layerPooled,
		}
		if err := a.packLayer(layout, packed, srcs, activeBuf, shape, layerIdx); err != nil {
			return backend.CompactForwardResult{}, err
		}
		if layerIdx == 0 && a.debugForceForwardFailureAfterFirstLayer {
			return backend.CompactForwardResult{}, fmt.Errorf("cuda compact forward forced failure after first layer")
		}
		current = projected
	}
	if err := a.launchFinalize(current, maskBuf, finalNorm, finalPooled, activeBuf, B, T, D, true); err != nil {
		return backend.CompactForwardResult{}, err
	}
	if shape.HasOutputProjection {
		if err := a.launchMM(finalNorm, a.bindings.out, outputRows, rows, D, O); err != nil {
			return backend.CompactForwardResult{}, err
		}
		if err := a.launchFinalize(outputRows, maskBuf, outputRows, finalPooled, activeBuf, B, T, O, false); err != nil {
			return backend.CompactForwardResult{}, err
		}
	}
	if err := a.packFinal(layout, packed, finalNorm, outputRows, finalPooled, shape); err != nil {
		return backend.CompactForwardResult{}, err
	}
	if err := a.packIntCopy(statusBuf, downloadPacked, 0, 0, 1); err != nil {
		return backend.CompactForwardResult{}, err
	}
	if err := a.packCopy(packed, downloadPacked, 0, 1, total); err != nil {
		return backend.CompactForwardResult{}, err
	}
	if err := a.synchronizeKernelBoundary(); err != nil {
		return backend.CompactForwardResult{}, err
	}
	download := make([]float32, total+1)
	if err := a.device.downloadFloat32(download, downloadPacked); err != nil {
		return backend.CompactForwardResult{}, err
	}
	status := int32(download[0])
	data := append([]float32(nil), download[1:]...)
	downloaded := int64((total + 1) * 4)
	packedBytes := int64(total * 4)
	statusBytes := int64(4)
	a.stats.UploadedBytes += uploaded
	a.stats.DownloadedBytes += downloaded
	a.stats.StatusDownloadedBytes += statusBytes
	a.stats.PackedDownloads++
	a.stats.PackedBytes += packedBytes
	a.stats.LastPackedFloats = total
	a.stats.LastPackedBytes = packedBytes
	a.stats.LastUploadBytes = uploaded
	a.stats.LastDownloadBytes = downloaded
	a.stats.LastStatusDownloadedBytes = statusBytes
	a.stats.LastKernelLaunches = a.stats.KernelLaunches - launchCountBefore
	a.stats.LastKernelSynchronizations = a.stats.KernelSynchronizations - syncCountBefore
	if status != 0 {
		return backend.CompactForwardResult{}, fmt.Errorf("cuda compact forward gather status %d", status)
	}
	return backend.CompactForwardResult{Layout: layout, Data: data}, nil
}

func (a *CompactForwardAccelerator) validateBindings(shape backend.CompactForwardShape) error {
	D, H := shape.ModelDim, shape.FFNDim
	if a.bindings.token.rows <= 0 || a.bindings.token.cols != D {
		return fmt.Errorf("cuda compact forward token embedding shape [%d %d], want [V %d]", a.bindings.token.rows, a.bindings.token.cols, D)
	}
	if a.roleName != "" && (a.bindings.role.rows <= 0 || a.bindings.role.cols != D) {
		return fmt.Errorf("cuda compact forward role embedding shape [%d %d], want [R %d]", a.bindings.role.rows, a.bindings.role.cols, D)
	}
	for i, l := range a.bindings.layer {
		for name, m := range map[string]residentMatrix{"attn_q": l.q, "attn_k": l.k, "attn_v": l.v, "attn_o": l.o} {
			if m.rows != D || m.cols != D {
				return fmt.Errorf("cuda compact forward layer %d %s shape [%d %d], want [%d %d]", i, name, m.rows, m.cols, D, D)
			}
		}
		if l.up.rows != D || l.up.cols != H {
			return fmt.Errorf("cuda compact forward layer %d ffn_up shape [%d %d], want [%d %d]", i, l.up.rows, l.up.cols, D, H)
		}
		if l.down.rows != H || l.down.cols != D {
			return fmt.Errorf("cuda compact forward layer %d ffn_down shape [%d %d], want [%d %d]", i, l.down.rows, l.down.cols, H, D)
		}
	}
	if shape.HasOutputProjection && (a.bindings.out.rows != D || a.bindings.out.cols != shape.OutputDim) {
		return fmt.Errorf("cuda compact forward output projection shape [%d %d], want [%d %d]", a.bindings.out.rows, a.bindings.out.cols, D, shape.OutputDim)
	}
	return nil
}

func (a *CompactForwardAccelerator) preflightResidentRefs(refs []backend.CompactForwardResidentRef, shape backend.CompactForwardShape) error {
	required := a.requiredResidentNames(shape)
	if len(refs) == 0 {
		return fmt.Errorf("cuda compact forward resident refs are required")
	}
	byName := make(map[string]backend.CompactForwardResidentRef, len(refs))
	for _, ref := range refs {
		if ref.Name == "" {
			return fmt.Errorf("cuda compact forward resident ref has empty name")
		}
		if _, exists := byName[ref.Name]; exists {
			return fmt.Errorf("cuda compact forward resident ref %q is duplicated", ref.Name)
		}
		if ref.Backend != eosartifact.BackendCUDA {
			return fmt.Errorf("cuda compact forward resident ref %q backend %q, want cuda", ref.Name, ref.Backend)
		}
		if ref.Token == nil {
			return fmt.Errorf("cuda compact forward resident ref %q token is nil", ref.Name)
		}
		if ref.Token.Backend() != eosartifact.BackendCUDA {
			return fmt.Errorf("cuda compact forward resident ref %q token backend %q, want cuda", ref.Name, ref.Token.Backend())
		}
		token, ok := ref.Token.(*optimizerResidentParameterToken)
		if !ok || token == nil {
			return fmt.Errorf("cuda compact forward resident ref %q has invalid cuda token", ref.Name)
		}
		state, unlock, err := token.lockCurrent()
		if err != nil {
			return err
		}
		if ref.Elements != state.elements {
			unlock()
			return fmt.Errorf("cuda compact forward resident ref %q elements %d, want %d", ref.Name, ref.Elements, state.elements)
		}
		unlock()
		byName[ref.Name] = ref
	}
	for _, name := range required {
		ref, ok := byName[name]
		if !ok {
			return fmt.Errorf("cuda compact forward resident ref %q is missing", name)
		}
		binding, ok := a.bindingForName(name)
		if !ok {
			return fmt.Errorf("cuda compact forward resident binding %q is missing", name)
		}
		token, ok := ref.Token.(*optimizerResidentParameterToken)
		if !ok || token == nil {
			return fmt.Errorf("cuda compact forward resident ref %q has invalid cuda token", name)
		}
		if bridged := a.bridged[name]; bridged == nil || bridged.generation != token.generation {
			return fmt.Errorf("cuda compact forward resident ref %q generation %d does not match bound generation", name, token.generation)
		}
		state, unlock, err := token.lockCurrent()
		if err != nil {
			return err
		}
		if binding.ptr != state.param || binding.elements != state.elements || ref.Elements != state.elements {
			unlock()
			return fmt.Errorf("cuda compact forward resident ref %q no longer matches bound device state", name)
		}
		unlock()
	}
	return nil
}

func (a *CompactForwardAccelerator) requiredResidentNames(shape backend.CompactForwardShape) []string {
	names := make([]string, 0, 2+len(a.layers)*6)
	if a.tokenName != "" {
		names = append(names, a.tokenName)
	}
	if a.roleName != "" {
		names = append(names, a.roleName)
	}
	for _, layer := range a.layers {
		names = append(names, layer.AttentionQ, layer.AttentionK, layer.AttentionV, layer.AttentionO, layer.FFNUp, layer.FFNDown)
	}
	if shape.HasOutputProjection && a.outName != "" {
		names = append(names, a.outName)
	}
	out := names[:0]
	for _, name := range names {
		if name != "" {
			out = append(out, name)
		}
	}
	return out
}

func (a *CompactForwardAccelerator) lockBridgeTokens() ([]func(), error) {
	names := make([]string, 0, 2+6*len(a.layers))
	names = append(names, a.tokenName, a.roleName, a.outName)
	for _, l := range a.layers {
		names = append(names, l.AttentionQ, l.AttentionK, l.AttentionV, l.AttentionO, l.FFNUp, l.FFNDown)
	}
	unlocks := make([]func(), 0, len(names))
	seen := map[string]bool{}
	for _, name := range names {
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		tok := a.bridged[name]
		if tok == nil {
			unlockResidentBridgeTokens(unlocks)
			return nil, fmt.Errorf("cuda compact forward resident binding %q is missing", name)
		}
		state, unlock, err := tok.lockCurrent()
		if err != nil {
			unlockResidentBridgeTokens(unlocks)
			return nil, err
		}
		binding, ok := a.bindingForName(name)
		if !ok || binding.ptr != state.param || binding.elements != state.elements {
			unlock()
			unlockResidentBridgeTokens(unlocks)
			return nil, fmt.Errorf("cuda compact forward resident binding %q no longer matches optimizer token", name)
		}
		unlocks = append(unlocks, unlock)
	}
	return unlocks, nil
}

func (a *CompactForwardAccelerator) bindingForName(name string) (residentMatrix, bool) {
	if name == a.tokenName {
		return a.bindings.token, a.bindings.token.ptr != 0
	}
	if name == a.roleName {
		return a.bindings.role, a.bindings.role.ptr != 0
	}
	if name == a.outName {
		return a.bindings.out, a.bindings.out.ptr != 0
	}
	for i, layer := range a.layers {
		if i >= len(a.bindings.layer) {
			break
		}
		bound := a.bindings.layer[i]
		switch name {
		case layer.AttentionQ:
			return bound.q, bound.q.ptr != 0
		case layer.AttentionK:
			return bound.k, bound.k.ptr != 0
		case layer.AttentionV:
			return bound.v, bound.v.ptr != 0
		case layer.AttentionO:
			return bound.o, bound.o.ptr != 0
		case layer.FFNUp:
			return bound.up, bound.up.ptr != 0
		case layer.FFNDown:
			return bound.down, bound.down.ptr != 0
		}
	}
	return residentMatrix{}, false
}

func (a *CompactForwardAccelerator) launchGather(token, role residentMatrix, tokens, roles, out, status C.CUdeviceptr, shape backend.CompactForwardShape) error {
	grid, block, err := checkedLaunch1D("cuda compact gather", shape.Batch*shape.Tokens*shape.ModelDim, 128)
	if err != nil {
		return err
	}
	var errStr *C.char
	useRole, useRoPE := 0, 0
	if a.roleName != "" {
		useRole = 1
	}
	if a.useRoPE {
		useRoPE = 1
	}
	if C.eosCudaLaunchCompactGather(a.device.ptr, a.kernels.gather.ptr, grid, block, token.ptr, role.ptr, tokens, roles, out, status, C.int(shape.Batch), C.int(shape.Tokens), C.int(shape.ModelDim), C.int(token.rows), C.int(role.rows), C.int(useRole), C.int(useRoPE), &errStr) != 0 {
		return cStringError(errStr)
	}
	return a.recordKernelLaunch()
}

func (a *CompactForwardAccelerator) launchMM(lhs C.CUdeviceptr, rhs residentMatrix, out C.CUdeviceptr, rows, inner, cols int) error {
	grid, block, err := checkedLaunch1D("cuda compact matmul", rows*cols, 128)
	if err != nil {
		return err
	}
	var errStr *C.char
	if C.eosCudaLaunchCompactMatmul(a.device.ptr, a.kernels.mm.ptr, grid, block, lhs, rhs.ptr, out, C.int(rows), C.int(inner), C.int(cols), &errStr) != 0 {
		return cStringError(errStr)
	}
	return a.recordKernelLaunch()
}

func (a *CompactForwardAccelerator) launchAttention(q, k, v, masks, scores, mixed C.CUdeviceptr, shape backend.CompactForwardShape) error {
	grid, block, err := checkedLaunch1D("cuda compact attention", shape.Batch*shape.Heads*shape.Tokens, 128)
	if err != nil {
		return err
	}
	var errStr *C.char
	if C.eosCudaLaunchCompactAttention(a.device.ptr, a.kernels.attn.ptr, grid, block, q, k, v, masks, scores, mixed, C.int(shape.Batch), C.int(shape.Tokens), C.int(shape.ModelDim), C.int(shape.Heads), C.int(shape.HeadDim), &errStr) != 0 {
		return cStringError(errStr)
	}
	return a.recordKernelLaunch()
}

func (a *CompactForwardAccelerator) launchResidualLayerNorm(src, residual, out, residualOut C.CUdeviceptr, rows, cols int) error {
	grid, block, err := checkedLaunch1D("cuda compact residual layernorm", rows, 128)
	if err != nil {
		return err
	}
	var errStr *C.char
	if C.eosCudaLaunchCompactResidualLayerNorm(a.device.ptr, a.kernels.ln.ptr, grid, block, src, residual, out, residualOut, C.int(rows), C.int(cols), &errStr) != 0 {
		return cStringError(errStr)
	}
	return a.recordKernelLaunch()
}

func (a *CompactForwardAccelerator) launchGELU(src, dst C.CUdeviceptr, elements int, fast bool) error {
	grid, block, err := checkedLaunch1D("cuda compact gelu", elements, 128)
	if err != nil {
		return err
	}
	var errStr *C.char
	fastFlag := 0
	if fast {
		fastFlag = 1
	}
	if C.eosCudaLaunchCompactGELU(a.device.ptr, a.kernels.gelu.ptr, grid, block, src, dst, C.int(elements), C.int(fastFlag), &errStr) != 0 {
		return cStringError(errStr)
	}
	return a.recordKernelLaunch()
}

func (a *CompactForwardAccelerator) launchFinalize(projected, masks, normalized, pooled, active C.CUdeviceptr, batch, seq, width int, normalize bool) error {
	grid, block, err := checkedLaunch1D("cuda compact finalize", batch, 128)
	if err != nil {
		return err
	}
	var errStr *C.char
	normalizeFlag := 0
	if normalize {
		normalizeFlag = 1
	}
	if C.eosCudaLaunchCompactFinalize(a.device.ptr, a.kernels.final.ptr, grid, block, projected, masks, normalized, pooled, active, C.int(batch), C.int(seq), C.int(width), C.int(normalizeFlag), &errStr) != 0 {
		return cStringError(errStr)
	}
	return a.recordKernelLaunch()
}

func (a *CompactForwardAccelerator) packCopy(src, dst C.CUdeviceptr, srcOffset, dstOffset, elements int) error {
	if elements == 0 {
		return nil
	}
	grid, block, err := checkedLaunch1D("cuda compact pack", elements, 128)
	if err != nil {
		return err
	}
	var errStr *C.char
	if C.eosCudaLaunchCompactPack(a.device.ptr, a.kernels.pack.ptr, grid, block, src, dst, C.int(srcOffset), C.int(dstOffset), C.int(elements), &errStr) != 0 {
		return cStringError(errStr)
	}
	return a.recordKernelLaunch()
}

func (a *CompactForwardAccelerator) packIntCopy(src, dst C.CUdeviceptr, srcOffset, dstOffset, elements int) error {
	if elements == 0 {
		return nil
	}
	grid, block, err := checkedLaunch1D("cuda compact pack int", elements, 128)
	if err != nil {
		return err
	}
	var errStr *C.char
	if C.eosCudaLaunchCompactPackInt(a.device.ptr, a.kernels.packI.ptr, grid, block, src, dst, C.int(srcOffset), C.int(dstOffset), C.int(elements), &errStr) != 0 {
		return cStringError(errStr)
	}
	return a.recordKernelLaunch()
}

func (a *CompactForwardAccelerator) recordKernelLaunch() error {
	a.stats.KernelLaunches++
	a.launchesSinceBoundary++
	if a.syncEachLaunch {
		if err := a.device.synchronize(); err != nil {
			return err
		}
		a.recordKernelSynchronization()
	}
	return nil
}

func (a *CompactForwardAccelerator) recordKernelSynchronization() {
	a.stats.KernelSynchronizations++
	a.launchesSinceBoundary = 0
}

func (a *CompactForwardAccelerator) synchronizeKernelBoundary() error {
	if a.syncEachLaunch {
		return nil
	}
	if a.launchesSinceBoundary == 0 {
		return nil
	}
	if err := a.device.synchronize(); err != nil {
		return err
	}
	a.recordKernelSynchronization()
	return nil
}

func (a *CompactForwardAccelerator) drainKernelError(err error) error {
	if err == nil || a == nil || a.device == nil || a.launchesSinceBoundary == 0 {
		return err
	}
	drainErr := a.device.synchronize()
	if drainErr != nil {
		return errors.Join(err, drainErr)
	}
	a.recordKernelSynchronization()
	return err
}

func (a *CompactForwardAccelerator) packLayer(layout backend.CompactForwardPackedStateLayout, packed C.CUdeviceptr, srcs map[string]C.CUdeviceptr, active C.CUdeviceptr, shape backend.CompactForwardShape, layer int) error {
	for seq := 0; seq < shape.Batch; seq++ {
		baseModel := seq * shape.Tokens * shape.ModelDim
		baseFFN := seq * shape.Tokens * shape.FFNDim
		baseScores := seq * shape.Heads * shape.Tokens * shape.Tokens
		for _, field := range []string{"input", "hidden", "attnQ", "attnK", "attnV", "attnScores", "attnMixed", "attnOutput", "attnResidual", "ffnHidden", "activated", "ffnOutput", "ffnResidual", "projected", "normalized", "pooled"} {
			span := compactForwardSpanByName(layout, compactForwardLayerSpanName(seq, layer, field))
			srcOffset := baseModel
			switch field {
			case "ffnHidden", "activated":
				srcOffset = baseFFN
			case "attnScores":
				srcOffset = baseScores
			case "pooled":
				srcOffset = seq * shape.ModelDim
			}
			if layer == shape.Layers-1 && field == "pooled" && shape.OutputDim != shape.ModelDim {
				continue
			}
			if err := a.packCopy(srcs[field], packed, srcOffset, span.Offset, span.Len); err != nil {
				return err
			}
		}
		activeSpan := compactForwardSpanByName(layout, compactForwardLayerSpanName(seq, layer, "activeCount"))
		if err := a.packIntCopy(active, packed, seq, activeSpan.Offset, 1); err != nil {
			return err
		}
	}
	return nil
}

func (a *CompactForwardAccelerator) packFinal(layout backend.CompactForwardPackedStateLayout, packed, finalNorm, outputRows, finalPooled C.CUdeviceptr, shape backend.CompactForwardShape) error {
	for seq := 0; seq < shape.Batch; seq++ {
		modelBase := seq * shape.Tokens * shape.ModelDim
		outBase := seq * shape.Tokens * shape.OutputDim
		pooledBase := seq * shape.OutputDim
		if err := a.packCopy(finalNorm, packed, modelBase, compactForwardSpanByName(layout, compactForwardSequenceSpanName(seq, "final.normalized")).Offset, shape.Tokens*shape.ModelDim); err != nil {
			return err
		}
		if err := a.packCopy(outputRows, packed, outBase, compactForwardSpanByName(layout, compactForwardSequenceSpanName(seq, "final.outputRows")).Offset, shape.Tokens*shape.OutputDim); err != nil {
			return err
		}
		if err := a.packCopy(finalPooled, packed, pooledBase, compactForwardSpanByName(layout, compactForwardSequenceSpanName(seq, "final.pooled")).Offset, shape.OutputDim); err != nil {
			return err
		}
		if shape.OutputDim != shape.ModelDim {
			span := compactForwardSpanByName(layout, compactForwardLayerSpanName(seq, shape.Layers-1, "pooled"))
			if err := a.packCopy(finalPooled, packed, pooledBase, span.Offset, shape.OutputDim); err != nil {
				return err
			}
		}
	}
	return nil
}

func compactForwardPackedLayout(shape backend.CompactForwardShape) (backend.CompactForwardPackedStateLayout, int, error) {
	if err := validateCompactForwardShape(shape); err != nil {
		return backend.CompactForwardPackedStateLayout{}, 0, err
	}
	layout := backend.CompactForwardPackedStateLayout{Version: backend.CompactForwardPackedStateVersion, Shape: shape}
	offset := 0
	add := func(name string, n int) {
		layout.Spans = append(layout.Spans, backend.CompactForwardStateSpan{Name: name, Offset: offset, Len: n})
		offset += n
	}
	model := shape.Tokens * shape.ModelDim
	ffn := shape.Tokens * shape.FFNDim
	scores := shape.Heads * shape.Tokens * shape.Tokens
	for seq := 0; seq < shape.Batch; seq++ {
		for layer := 0; layer < shape.Layers; layer++ {
			add(compactForwardLayerSpanName(seq, layer, "activeCount"), 1)
			for _, name := range []string{"input", "hidden", "attnQ", "attnK", "attnV"} {
				add(compactForwardLayerSpanName(seq, layer, name), model)
			}
			add(compactForwardLayerSpanName(seq, layer, "attnScores"), scores)
			for _, name := range []string{"attnMixed", "attnOutput", "attnResidual"} {
				add(compactForwardLayerSpanName(seq, layer, name), model)
			}
			for _, name := range []string{"ffnHidden", "activated"} {
				add(compactForwardLayerSpanName(seq, layer, name), ffn)
			}
			for _, name := range []string{"ffnOutput", "ffnResidual", "projected", "normalized"} {
				add(compactForwardLayerSpanName(seq, layer, name), model)
			}
			if layer == shape.Layers-1 {
				add(compactForwardLayerSpanName(seq, layer, "pooled"), shape.OutputDim)
			} else {
				add(compactForwardLayerSpanName(seq, layer, "pooled"), shape.ModelDim)
			}
		}
		add(compactForwardSequenceSpanName(seq, "final.normalized"), model)
		add(compactForwardSequenceSpanName(seq, "final.outputRows"), shape.Tokens*shape.OutputDim)
		add(compactForwardSequenceSpanName(seq, "final.pooled"), shape.OutputDim)
	}
	return layout, offset, nil
}

func validateCompactForwardShape(shape backend.CompactForwardShape) error {
	if shape.Batch <= 0 || shape.Tokens <= 0 || shape.ModelDim <= 0 || shape.FFNDim <= 0 || shape.Heads <= 0 || shape.HeadDim <= 0 || shape.Layers <= 0 || shape.OutputDim <= 0 {
		return fmt.Errorf("cuda compact forward shape must be positive: %+v", shape)
	}
	if !shape.HasOutputProjection && shape.OutputDim != shape.ModelDim {
		return fmt.Errorf("cuda compact forward output_dim=%d must equal model_dim=%d without output projection", shape.OutputDim, shape.ModelDim)
	}
	if shape.Heads*shape.HeadDim != shape.ModelDim {
		return fmt.Errorf("cuda compact forward heads=%d head_dim=%d do not match model_dim=%d", shape.Heads, shape.HeadDim, shape.ModelDim)
	}
	for label, v := range map[string]int{"model": shape.Batch * shape.Tokens * shape.ModelDim, "ffn": shape.Batch * shape.Tokens * shape.FFNDim, "scores": shape.Batch * shape.Heads * shape.Tokens * shape.Tokens} {
		if v <= 0 || int64(v) > int64(1<<31-1) {
			return fmt.Errorf("cuda compact forward %s elements %d exceed supported shape", label, v)
		}
	}
	return nil
}

func validateCompactForwardInputs(shape backend.CompactForwardShape, tokens, masks [][]int32, roles []int32) error {
	if len(tokens) != shape.Batch || len(masks) != shape.Batch || len(roles) != shape.Batch {
		return fmt.Errorf("cuda compact forward batch input mismatch")
	}
	for i := 0; i < shape.Batch; i++ {
		if len(tokens[i]) != shape.Tokens || len(masks[i]) != shape.Tokens {
			return fmt.Errorf("cuda compact forward sequence %d length mismatch", i)
		}
		active := 0
		for _, v := range masks[i] {
			if v != 0 {
				active++
			}
		}
		if active == 0 {
			return fmt.Errorf("cuda compact forward sequence %d mask selects zero tokens", i)
		}
	}
	return nil
}

func compactForwardGELUFast(mode string) (bool, error) {
	switch mode {
	case "", backend.CompactForwardGELUExact:
		return false, nil
	case backend.CompactForwardGELUFast:
		return true, nil
	default:
		return false, fmt.Errorf("cuda compact forward unsupported GELU mode %q", mode)
	}
}

func compactForwardMemoryEstimateBytes(shape backend.CompactForwardShape, vocabRows, roleRows int) (int64, error) {
	if err := validateCompactForwardShape(shape); err != nil {
		return 0, err
	}
	if vocabRows <= 0 {
		return 0, fmt.Errorf("cuda compact forward vocab rows must be positive")
	}
	if roleRows < 0 {
		return 0, fmt.Errorf("cuda compact forward role rows must be non-negative")
	}
	layout, packedFloats, err := compactForwardPackedLayout(shape)
	if err != nil {
		return 0, err
	}
	_ = layout
	add := func(total *int64, label string, elems int64) error {
		maxInt64 := int64(^uint64(0) >> 1)
		if elems < 0 || elems > maxInt64/4 {
			return fmt.Errorf("cuda compact forward %s byte estimate overflows", label)
		}
		bytes := elems * 4
		if *total > maxInt64-bytes {
			return fmt.Errorf("cuda compact forward memory estimate overflows")
		}
		*total += bytes
		return nil
	}
	mul := func(label string, values ...int) (int64, error) {
		maxInt64 := int64(^uint64(0) >> 1)
		out := int64(1)
		for _, value := range values {
			if value < 0 {
				return 0, fmt.Errorf("cuda compact forward %s dimension %d is negative", label, value)
			}
			if value != 0 && out > maxInt64/int64(value) {
				return 0, fmt.Errorf("cuda compact forward %s element estimate overflows", label)
			}
			out *= int64(value)
		}
		return out, nil
	}
	B, T, D, H, O := shape.Batch, shape.Tokens, shape.ModelDim, shape.FFNDim, shape.OutputDim
	rows, err := mul("rows", B, T)
	if err != nil {
		return 0, err
	}
	modelElems, err := mul("model", B, T, D)
	if err != nil {
		return 0, err
	}
	ffnElems, err := mul("ffn", B, T, H)
	if err != nil {
		return 0, err
	}
	scoreElems, err := mul("scores", B, shape.Heads, T, T)
	if err != nil {
		return 0, err
	}
	outputElems, err := mul("output", B, T, O)
	if err != nil {
		return 0, err
	}
	residentToken, err := mul("resident token", vocabRows, D)
	if err != nil {
		return 0, err
	}
	residentRole, err := mul("resident role", roleRows, D)
	if err != nil {
		return 0, err
	}
	layerD2, err := mul("resident layer d2", D, D)
	if err != nil {
		return 0, err
	}
	layerDH, err := mul("resident layer dh", D, H)
	if err != nil {
		return 0, err
	}
	maxInt64 := int64(^uint64(0) >> 1)
	if layerD2 > maxInt64/4 || layerDH > maxInt64/2 {
		return 0, fmt.Errorf("cuda compact forward resident layer element estimate overflows")
	}
	perLayerD2 := 4 * layerD2
	perLayerDH := 2 * layerDH
	if perLayerD2 > maxInt64-perLayerDH {
		return 0, fmt.Errorf("cuda compact forward resident layer element estimate overflows")
	}
	perLayer := perLayerD2 + perLayerDH
	if int64(shape.Layers) > maxInt64/perLayer {
		return 0, fmt.Errorf("cuda compact forward resident layer element estimate overflows")
	}
	residentLayer := int64(shape.Layers) * perLayer
	residentOut := int64(0)
	if shape.HasOutputProjection {
		residentOut, err = mul("resident output projection", D, O)
		if err != nil {
			return 0, err
		}
	}
	total := int64(0)
	for _, item := range []struct {
		label string
		elems int64
	}{
		{"resident_token", residentToken},
		{"resident_role", residentRole},
		{"resident_layers", residentLayer},
		{"resident_output_projection", residentOut},
		{"input_current", modelElems},
		{"packed", int64(packedFloats)},
		{"download_packed_with_status", int64(packedFloats) + 1},
		{"model_scratch", 13 * modelElems},
		{"ffn_scratch", 2 * ffnElems},
		{"scores", scoreElems},
		{"final_norm", modelElems},
		{"output_rows", outputElems},
		{"final_pooled", int64(B) * int64(O)},
		{"active_i32", int64(B)},
		{"tokens_i32", rows},
		{"masks_i32", rows},
		{"roles_i32", int64(B)},
		{"status_i32", 1},
	} {
		if err := add(&total, item.label, item.elems); err != nil {
			return 0, err
		}
	}
	return total, nil
}

func flattenInt32(in [][]int32) []int32 {
	n := 0
	for _, row := range in {
		n += len(row)
	}
	out := make([]int32, 0, n)
	for _, row := range in {
		out = append(out, row...)
	}
	return out
}

func compactForwardLayerSpanName(seq, layer int, field string) string {
	return fmt.Sprintf("seq%d.layer%d.%s", seq, layer, field)
}

func compactForwardSequenceSpanName(seq int, field string) string {
	return fmt.Sprintf("seq%d.%s", seq, field)
}

func compactForwardSpanByName(layout backend.CompactForwardPackedStateLayout, name string) backend.CompactForwardStateSpan {
	for _, span := range layout.Spans {
		if span.Name == name {
			return span
		}
	}
	return backend.CompactForwardStateSpan{}
}

func compactForwardUnsupported(err error) bool {
	return err != nil && strings.Contains(err.Error(), "unsupported")
}
