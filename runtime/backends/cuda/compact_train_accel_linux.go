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
static int eosCudaLaunchCompactTrainLayerNormBackward(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr gradOut, CUdeviceptr normalized, CUdeviceptr pre, CUdeviceptr out0, int rows, int cols, char** err);
static int eosCudaLaunchCompactTrainMatMulLeftTransposeAccum(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr lhs, CUdeviceptr gradOut, CUdeviceptr gradWeight, int rows, int inDim, int outDim, char** err);
static int eosCudaLaunchCompactTrainMatMulRightTranspose(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr gradOut, CUdeviceptr weight, CUdeviceptr gradIn, int rows, int inDim, int outDim, char** err);
static int eosCudaLaunchCompactTrainGELUBackward(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr gradOut, CUdeviceptr preAct, CUdeviceptr out0, int elements, int fast, char** err);
static int eosCudaLaunchCompactTrainAdd(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr left, CUdeviceptr right, CUdeviceptr out0, int elements, char** err);
static int eosCudaLaunchCompactTrainAttentionBackward(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr gradMixed, CUdeviceptr q, CUdeviceptr k, CUdeviceptr v, CUdeviceptr probs, CUdeviceptr gradQ, CUdeviceptr gradK, CUdeviceptr gradV, int batch, int seq, int modelDim, int heads, int headDim, char** err);
static int eosCudaLaunchCompactTrainRoPETranspose(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr src, CUdeviceptr out0, int rows, int cols, int seq, char** err);
static int eosCudaLaunchCompactTrainInputScatter(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr gradInput, CUdeviceptr tokens, CUdeviceptr roles, CUdeviceptr gradToken, CUdeviceptr gradRole, int batch, int seq, int modelDim, int vocab, int roleRows, int useRole, char** err);

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

static int eosCudaLaunchCompactTrainLayerNormBackward(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr gradOut, CUdeviceptr normalized, CUdeviceptr pre, CUdeviceptr out0, int rows, int cols, char** err) {
	void* args[] = {&gradOut, &normalized, &pre, &out0, &rows, &cols};
	return eos_compact_train_launch(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchCompactTrainMatMulLeftTransposeAccum(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr lhs, CUdeviceptr gradOut, CUdeviceptr gradWeight, int rows, int inDim, int outDim, char** err) {
	void* args[] = {&lhs, &gradOut, &gradWeight, &rows, &inDim, &outDim};
	return eos_compact_train_launch(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchCompactTrainMatMulRightTranspose(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr gradOut, CUdeviceptr weight, CUdeviceptr gradIn, int rows, int inDim, int outDim, char** err) {
	void* args[] = {&gradOut, &weight, &gradIn, &rows, &inDim, &outDim};
	return eos_compact_train_launch(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchCompactTrainGELUBackward(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr gradOut, CUdeviceptr preAct, CUdeviceptr out0, int elements, int fast, char** err) {
	void* args[] = {&gradOut, &preAct, &out0, &elements, &fast};
	return eos_compact_train_launch(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchCompactTrainAdd(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr left, CUdeviceptr right, CUdeviceptr out0, int elements, char** err) {
	void* args[] = {&left, &right, &out0, &elements};
	return eos_compact_train_launch(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchCompactTrainAttentionBackward(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr gradMixed, CUdeviceptr q, CUdeviceptr k, CUdeviceptr v, CUdeviceptr probs, CUdeviceptr gradQ, CUdeviceptr gradK, CUdeviceptr gradV, int batch, int seq, int modelDim, int heads, int headDim, char** err) {
	void* args[] = {&gradMixed, &q, &k, &v, &probs, &gradQ, &gradK, &gradV, &batch, &seq, &modelDim, &heads, &headDim};
	return eos_compact_train_launch(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchCompactTrainRoPETranspose(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr src, CUdeviceptr out0, int rows, int cols, int seq, char** err) {
	void* args[] = {&src, &out0, &rows, &cols, &seq};
	return eos_compact_train_launch(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchCompactTrainInputScatter(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr gradInput, CUdeviceptr tokens, CUdeviceptr roles, CUdeviceptr gradToken, CUdeviceptr gradRole, int batch, int seq, int modelDim, int vocab, int roleRows, int useRole, char** err) {
	void* args[] = {&gradInput, &tokens, &roles, &gradToken, &gradRole, &batch, &seq, &modelDim, &vocab, &roleRows, &useRole};
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
	arenas       map[*compactTrainHandleToken]*compactTrainArena
	trainKernels compactTrainKernels
	grads        map[string]*compactTrainGradient
	gradGen      uint64
	stepID       uint64
	stepActive   bool
	stepSealed   bool
	stepPoisoned bool
	nextHandleID uint64
	closed       bool

	debugForceBackwardFailureAfterGradMutation bool
}

type compactTrainKernels struct {
	finalProjectionGrad *auxKernel
	finalHiddenGrad     *auxKernel
	layerNormBackward   *auxKernel
	matMulLeftTAccum    *auxKernel
	matMulRightT        *auxKernel
	geluBackward        *auxKernel
	add                 *auxKernel
	attentionBackward   *auxKernel
	ropeTranspose       *auxKernel
	inputScatter        *auxKernel
}

type compactTrainGradient struct {
	ptr                 C.CUdeviceptr
	elements            int
	generation          uint64
	stepID              uint64
	token               *compactTrainGradientToken
	optimizerOwner      *optimizerAccelerator
	optimizerToken      *optimizerResidentParameterToken
	optimizerGeneration uint64
	optimizerParam      C.CUdeviceptr
	optimizerUsed       bool
}

type compactTrainResidentRefSnapshot struct {
	ref        backend.CompactForwardResidentRef
	token      *optimizerResidentParameterToken
	owner      *optimizerAccelerator
	generation uint64
	param      C.CUdeviceptr
	elements   int
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
	id         uint64
	generation uint64
	live       bool
	token      *compactTrainHandleToken
	geluFast   bool

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
	gradFFNResidual     C.CUdeviceptr
	gradActivatedPre    C.CUdeviceptr
	gradActivated       C.CUdeviceptr
	gradHiddenFromFFN   C.CUdeviceptr
	gradAttention       C.CUdeviceptr
	gradMixed           C.CUdeviceptr
	gradQ               C.CUdeviceptr
	gradK               C.CUdeviceptr
	gradV               C.CUdeviceptr
	gradRoPE            C.CUdeviceptr
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
	owner      *CompactTrainAccelerator
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
	return !t.owner.stepPoisoned && grad != nil && grad.token == t && grad.generation == t.generation && grad.stepID == t.stepID && grad.elements == t.elements && grad.ptr != 0
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

extern "C" __global__ void eos_compact_train_layernorm_backward(
    const float* gradOut, const float* normalized, const float* pre, float* out0,
    int rows, int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    int base = row * cols;
    float mean = 0.0f;
    for (int c = 0; c < cols; ++c) {
        mean += pre[base + c];
    }
    mean /= (float)cols;
    float variance = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float centered = pre[base + c] - mean;
        variance += centered * centered;
    }
    variance /= (float)cols;
    float invStd = rsqrtf(variance + 1.0e-5f);
    float sumGrad = 0.0f;
    float sumGradNorm = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float g = gradOut[base + c];
        sumGrad += g;
        sumGradNorm += g * normalized[base + c];
    }
    float n = (float)cols;
    for (int c = 0; c < cols; ++c) {
        out0[base + c] = (invStd / n) * (n * gradOut[base + c] - sumGrad - normalized[base + c] * sumGradNorm);
    }
}

extern "C" __global__ void eos_compact_train_matmul_left_t_accum(
    const float* lhs, const float* gradOut, float* gradWeight,
    int rows, int inDim, int outDim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = inDim * outDim;
    if (idx >= total) return;
    int o = idx % outDim;
    int i = idx / outDim;
    float sum = 0.0f;
    for (int r = 0; r < rows; ++r) {
        sum += lhs[r * inDim + i] * gradOut[r * outDim + o];
    }
    gradWeight[idx] += sum;
}

extern "C" __global__ void eos_compact_train_matmul_right_t(
    const float* gradOut, const float* weight, float* gradIn,
    int rows, int inDim, int outDim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * inDim;
    if (idx >= total) return;
    int i = idx % inDim;
    int r = idx / inDim;
    float sum = 0.0f;
    for (int o = 0; o < outDim; ++o) {
        sum += gradOut[r * outDim + o] * weight[i * outDim + o];
    }
    gradIn[idx] = sum;
}

static __device__ float eos_compact_train_fast_tanh(float x) {
    if (x >= 3.0f) return 1.0f;
    if (x <= -3.0f) return -1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

static __device__ float eos_compact_train_fast_tanh_derivative(float x) {
    if (x >= 3.0f || x <= -3.0f) return 0.0f;
    float x2 = x * x;
    float diff = x2 - 9.0f;
    float den = 3.0f + x2;
    return (diff * diff) / (9.0f * den * den);
}

extern "C" __global__ void eos_compact_train_gelu_backward(
    const float* gradOut, const float* preAct, float* out0, int elements, int fast
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) return;
    float x = preAct[idx];
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    float t = fast != 0 ? eos_compact_train_fast_tanh(inner) : tanhf(inner);
    float innerGrad = 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
    float tanhGrad = fast != 0 ? eos_compact_train_fast_tanh_derivative(inner) : (1.0f - t * t);
    float geluGrad = 0.5f * (1.0f + t) + 0.5f * x * tanhGrad * innerGrad;
    out0[idx] = gradOut[idx] * geluGrad;
}

extern "C" __global__ void eos_compact_train_add(
    const float* left, const float* right, float* out0, int elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) return;
    out0[idx] = left[idx] + right[idx];
}

extern "C" __global__ void eos_compact_train_attention_backward(
    const float* gradMixed, const float* q, const float* k, const float* v, const float* probs,
    float* gradQ, float* gradK, float* gradV,
    int batch, int seq, int modelDim, int heads, int headDim
) {
    int job = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads;
    if (job >= total) return;
    int head = job % heads;
    int b = job / heads;
    int baseRows = b * seq;
    int headOffset = head * headDim;
    float scale = rsqrtf((float)headDim);
    for (int row = 0; row < seq; ++row) {
        int base = (baseRows + row) * modelDim + headOffset;
        for (int c = 0; c < headDim; ++c) {
            gradQ[base + c] = 0.0f;
            gradK[base + c] = 0.0f;
            gradV[base + c] = 0.0f;
        }
    }
    int scoreHeadBase = (b * heads + head) * seq * seq;
    for (int query = 0; query < seq; ++query) {
        int queryBase = (baseRows + query) * modelDim + headOffset;
        int scoreRowBase = scoreHeadBase + query * seq;
        float dot = 0.0f;
        for (int key = 0; key < seq; ++key) {
            int keyBase = (baseRows + key) * modelDim + headOffset;
            float sum = 0.0f;
            for (int c = 0; c < headDim; ++c) {
                sum += gradMixed[queryBase + c] * v[keyBase + c];
            }
            dot += sum * probs[scoreRowBase + key];
        }
        for (int key = 0; key < seq; ++key) {
            int keyBase = (baseRows + key) * modelDim + headOffset;
            float sum = 0.0f;
            for (int c = 0; c < headDim; ++c) {
                sum += gradMixed[queryBase + c] * v[keyBase + c];
            }
            float prob = probs[scoreRowBase + key];
            float preGrad = prob * (sum - dot) * scale;
            for (int c = 0; c < headDim; ++c) {
                gradV[keyBase + c] += prob * gradMixed[queryBase + c];
                gradQ[queryBase + c] += preGrad * k[keyBase + c];
                gradK[keyBase + c] += preGrad * q[queryBase + c];
            }
        }
    }
}

extern "C" __global__ void eos_compact_train_rope_transpose(
    const float* src, float* out0, int rows, int cols, int seq
) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int pairsPerRow = (cols + 1) / 2;
    int total = rows * pairsPerRow;
    if (pair >= total) return;
    int row = pair / pairsPerRow;
    int col = (pair - row * pairsPerRow) * 2;
    int base = row * cols + col;
    if (col + 1 >= cols) {
        out0[base] = src[base];
        return;
    }
    int pos = seq > 0 ? row % seq : row;
    double theta = (double)pos / pow(10000.0, (double)col / (double)cols);
    float c = (float)cos(theta);
    float s = (float)sin(theta);
    float x0 = src[base];
    float x1 = src[base + 1];
    out0[base] = x0 * c + x1 * s;
    out0[base + 1] = -x0 * s + x1 * c;
}

extern "C" __global__ void eos_compact_train_input_scatter(
    const float* gradInput, const int* tokens, const int* roles,
    float* gradToken, float* gradRole,
    int batch, int seq, int modelDim, int vocab, int roleRows, int useRole
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tokenTotal = vocab * modelDim;
    int roleTotal = useRole != 0 ? roleRows * modelDim : 0;
    int total = tokenTotal + roleTotal;
    if (idx >= total) return;
    if (idx < tokenTotal) {
        int col = idx % modelDim;
        int tok = idx / modelDim;
        float sum = 0.0f;
        for (int row = 0; row < batch * seq; ++row) {
            if (tokens[row] == tok) {
                sum += gradInput[row * modelDim + col];
            }
        }
        gradToken[idx] += sum;
        return;
    }
    int ridx = idx - tokenTotal;
    int col = ridx % modelDim;
    int role = ridx / modelDim;
    float sum = 0.0f;
    for (int b = 0; b < batch; ++b) {
        if (roles[b] == role) {
            for (int pos = 0; pos < seq; ++pos) {
                sum += gradInput[(b * seq + pos) * modelDim + col];
            }
        }
    }
    gradRole[ridx] += sum;
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
	ln, err := compile("eos_compact_train_layernorm_backward")
	if err != nil {
		base.device.destroyAuxKernel(proj)
		base.device.destroyAuxKernel(hidden)
		base.Close()
		return nil, err
	}
	leftT, err := compile("eos_compact_train_matmul_left_t_accum")
	if err != nil {
		base.device.destroyAuxKernel(proj)
		base.device.destroyAuxKernel(hidden)
		base.device.destroyAuxKernel(ln)
		base.Close()
		return nil, err
	}
	rightT, err := compile("eos_compact_train_matmul_right_t")
	if err != nil {
		base.device.destroyAuxKernel(proj)
		base.device.destroyAuxKernel(hidden)
		base.device.destroyAuxKernel(ln)
		base.device.destroyAuxKernel(leftT)
		base.Close()
		return nil, err
	}
	gelu, err := compile("eos_compact_train_gelu_backward")
	if err != nil {
		base.device.destroyAuxKernel(proj)
		base.device.destroyAuxKernel(hidden)
		base.device.destroyAuxKernel(ln)
		base.device.destroyAuxKernel(leftT)
		base.device.destroyAuxKernel(rightT)
		base.Close()
		return nil, err
	}
	add, err := compile("eos_compact_train_add")
	if err != nil {
		base.device.destroyAuxKernel(proj)
		base.device.destroyAuxKernel(hidden)
		base.device.destroyAuxKernel(ln)
		base.device.destroyAuxKernel(leftT)
		base.device.destroyAuxKernel(rightT)
		base.device.destroyAuxKernel(gelu)
		base.Close()
		return nil, err
	}
	attnBackward, err := compile("eos_compact_train_attention_backward")
	if err != nil {
		base.device.destroyAuxKernel(proj)
		base.device.destroyAuxKernel(hidden)
		base.device.destroyAuxKernel(ln)
		base.device.destroyAuxKernel(leftT)
		base.device.destroyAuxKernel(rightT)
		base.device.destroyAuxKernel(gelu)
		base.device.destroyAuxKernel(add)
		base.Close()
		return nil, err
	}
	ropeTranspose, err := compile("eos_compact_train_rope_transpose")
	if err != nil {
		base.device.destroyAuxKernel(proj)
		base.device.destroyAuxKernel(hidden)
		base.device.destroyAuxKernel(ln)
		base.device.destroyAuxKernel(leftT)
		base.device.destroyAuxKernel(rightT)
		base.device.destroyAuxKernel(gelu)
		base.device.destroyAuxKernel(add)
		base.device.destroyAuxKernel(attnBackward)
		base.Close()
		return nil, err
	}
	inputScatter, err := compile("eos_compact_train_input_scatter")
	if err != nil {
		base.device.destroyAuxKernel(proj)
		base.device.destroyAuxKernel(hidden)
		base.device.destroyAuxKernel(ln)
		base.device.destroyAuxKernel(leftT)
		base.device.destroyAuxKernel(rightT)
		base.device.destroyAuxKernel(gelu)
		base.device.destroyAuxKernel(add)
		base.device.destroyAuxKernel(attnBackward)
		base.device.destroyAuxKernel(ropeTranspose)
		base.Close()
		return nil, err
	}
	return &CompactTrainAccelerator{
		CompactForwardAccelerator: base,
		trainKernels: compactTrainKernels{
			finalProjectionGrad: proj,
			finalHiddenGrad:     hidden,
			layerNormBackward:   ln,
			matMulLeftTAccum:    leftT,
			matMulRightT:        rightT,
			geluBackward:        gelu,
			add:                 add,
			attentionBackward:   attnBackward,
			ropeTranspose:       ropeTranspose,
			inputScatter:        inputScatter,
		},
		grads:  map[string]*compactTrainGradient{},
		arenas: map[*compactTrainHandleToken]*compactTrainArena{},
	}, nil
}

func (a *CompactTrainAccelerator) Backend() eosartifact.BackendKind { return eosartifact.BackendCUDA }

func (a *CompactTrainAccelerator) ConfigureCompactForward(names []backend.CompactForwardLayerConfig, tokenName, roleName, outputProjectionName string, useRoPE bool) {
	a.configureCompactTrain(compactForwardLayerNamesFromBackend(names), tokenName, roleName, outputProjectionName, useRoPE)
}

func (a *CompactTrainAccelerator) Configure(names []CompactForwardLayerNames, tokenName, roleName, outputProjectionName string, useRoPE bool) {
	a.configureCompactTrain(names, tokenName, roleName, outputProjectionName, useRoPE)
}

func (a *CompactTrainAccelerator) configureCompactTrain(names []CompactForwardLayerNames, tokenName, roleName, outputProjectionName string, useRoPE bool) {
	if a == nil || a.CompactForwardAccelerator == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.releaseArenasLocked()
	a.releaseGradientsLocked()
	a.stepActive = false
	a.stepSealed = false
	a.stepPoisoned = false
	a.layers = append(a.layers[:0], names...)
	a.tokenName = tokenName
	a.roleName = roleName
	a.outName = outputProjectionName
	a.useRoPE = useRoPE
}

func (a *CompactTrainAccelerator) Close() {
	if a == nil || a.CompactForwardAccelerator == nil {
		return
	}
	a.mu.Lock()
	a.closed = true
	a.releaseGradientsLocked()
	a.releaseArenasLocked()
	if a.device != nil {
		a.device.destroyAuxKernel(a.trainKernels.finalProjectionGrad)
		a.device.destroyAuxKernel(a.trainKernels.finalHiddenGrad)
		a.device.destroyAuxKernel(a.trainKernels.layerNormBackward)
		a.device.destroyAuxKernel(a.trainKernels.matMulLeftTAccum)
		a.device.destroyAuxKernel(a.trainKernels.matMulRightT)
		a.device.destroyAuxKernel(a.trainKernels.geluBackward)
		a.device.destroyAuxKernel(a.trainKernels.add)
		a.device.destroyAuxKernel(a.trainKernels.attentionBackward)
		a.device.destroyAuxKernel(a.trainKernels.ropeTranspose)
		a.device.destroyAuxKernel(a.trainKernels.inputScatter)
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
	if err := a.validateTrainForwardResidentRefsExactLocked(req.ResidentRefs, shape); err != nil {
		return err
	}
	return nil
}

func (a *CompactTrainAccelerator) BeginCompactTrainStep(stepID uint64, refs []backend.CompactForwardResidentRef) error {
	if a == nil || a.CompactForwardAccelerator == nil {
		return fmt.Errorf("cuda compact train accelerator is not initialized")
	}
	snapshots, unlockOptimizer, err := lockCompactTrainResidentRefs(refs)
	if err != nil {
		return err
	}
	defer unlockOptimizer()
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.device == nil || a.closed {
		return fmt.Errorf("cuda compact train accelerator is closed")
	}
	if a.stepActive && (!a.stepPoisoned || a.stats.LiveHandles > 0) {
		return fmt.Errorf("cuda compact train step %d is already active", a.stepID)
	}
	if len(snapshots) == 0 {
		return fmt.Errorf("cuda compact train resident refs are required")
	}
	if err := a.validateBeginResidentRefSetLocked(snapshots); err != nil {
		return err
	}
	nextGradGen := a.gradGen + 1
	pending := make(map[string]*compactTrainGradient, len(snapshots))
	freePending := func() {
		for _, grad := range pending {
			if grad != nil && grad.ptr != 0 {
				_ = a.device.freeBuffer(grad.ptr)
				grad.ptr = 0
			}
		}
	}
	var bytes int64
	for _, snapshot := range snapshots {
		ref := snapshot.ref
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
		gradToken := &compactTrainGradientToken{owner: a, name: ref.Name, generation: nextGradGen, stepID: stepID, elements: ref.Elements}
		pending[ref.Name] = &compactTrainGradient{
			ptr:                 ptr,
			elements:            ref.Elements,
			generation:          nextGradGen,
			stepID:              stepID,
			token:               gradToken,
			optimizerOwner:      snapshot.owner,
			optimizerToken:      snapshot.token,
			optimizerGeneration: snapshot.generation,
			optimizerParam:      snapshot.param,
		}
		bytes += int64(ref.Elements * 4)
	}
	a.releaseGradientsLocked()
	a.releaseArenasLocked()
	a.gradGen = nextGradGen
	a.stepID = stepID
	a.stepActive = true
	a.stepSealed = false
	a.stepPoisoned = false
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
	if a.stats.LiveHandles > 0 {
		return fmt.Errorf("cuda compact train end step with %d live handles", a.stats.LiveHandles)
	}
	if a.stepPoisoned {
		return fmt.Errorf("cuda compact train step %d is poisoned", a.stepID)
	}
	a.stepActive = false
	a.stepSealed = true
	return nil
}

func (a *CompactTrainAccelerator) AbortCompactTrainStep(stepID uint64) error {
	if a == nil || a.CompactForwardAccelerator == nil {
		return fmt.Errorf("cuda compact train accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.device == nil || a.closed {
		return fmt.Errorf("cuda compact train accelerator is closed")
	}
	if !a.stepActive && !a.stepSealed {
		return fmt.Errorf("cuda compact train step is not active")
	}
	if a.stepID != stepID {
		return fmt.Errorf("cuda compact train abort step %d is stale, current %d", stepID, a.stepID)
	}
	a.releaseArenasLocked()
	a.releaseGradientsLocked()
	a.stepActive = false
	a.stepSealed = false
	a.stepPoisoned = false
	return nil
}

func (a *CompactTrainAccelerator) RunCompactTrainBackward(req backend.CompactTrainBackwardRequest) (backend.CompactTrainBackwardResult, error) {
	if a == nil || a.CompactForwardAccelerator == nil {
		return backend.CompactTrainBackwardResult{}, fmt.Errorf("cuda compact train accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	start := time.Now()
	arena, err := a.validateBackwardRequestLocked(req)
	if err != nil {
		return backend.CompactTrainBackwardResult{}, err
	}
	shape := arena.shape
	launchesBefore := a.CompactForwardAccelerator.stats.KernelLaunches
	syncsBefore := a.CompactForwardAccelerator.stats.KernelSynchronizations
	if err := a.runFinalOutputBackwardLocked(req, arena, shape); err != nil {
		return backend.CompactTrainBackwardResult{}, a.poisonBackwardErrorLocked(err, launchesBefore, syncsBefore, start)
	}
	for layerIdx := shape.Layers - 1; layerIdx >= 0; layerIdx-- {
		if err := a.runLayerFFNBackwardLocked(layerIdx, arena, shape); err != nil {
			return backend.CompactTrainBackwardResult{}, a.poisonBackwardErrorLocked(err, launchesBefore, syncsBefore, start)
		}
		if a.debugForceBackwardFailureAfterGradMutation {
			return backend.CompactTrainBackwardResult{}, a.poisonBackwardErrorLocked(fmt.Errorf("cuda compact train forced failure after gradient mutation"), launchesBefore, syncsBefore, start)
		}
		if err := a.runLayerAttentionBackwardLocked(layerIdx, arena, shape); err != nil {
			return backend.CompactTrainBackwardResult{}, a.poisonBackwardErrorLocked(err, launchesBefore, syncsBefore, start)
		}
	}
	scatterInput := arena.gradHidden
	if a.useRoPE {
		if err := a.launchRoPETranspose(arena.gradHidden, arena.gradRoPE, shape.Batch*shape.Tokens, shape.ModelDim, shape.Tokens); err != nil {
			return backend.CompactTrainBackwardResult{}, a.poisonBackwardErrorLocked(err, launchesBefore, syncsBefore, start)
		}
		scatterInput = arena.gradRoPE
	}
	if err := a.runInputScatterBackwardLocked(scatterInput, arena, shape); err != nil {
		return backend.CompactTrainBackwardResult{}, a.poisonBackwardErrorLocked(err, launchesBefore, syncsBefore, start)
	}
	if err := a.synchronizeKernelBoundary(); err != nil {
		return backend.CompactTrainBackwardResult{}, a.poisonBackwardErrorLocked(err, launchesBefore, syncsBefore, start)
	}
	if err := a.consumeHandleLocked(req.Handle); err != nil {
		return backend.CompactTrainBackwardResult{}, a.poisonBackwardErrorLocked(err, launchesBefore, syncsBefore, start)
	}
	backwardLaunches := a.CompactForwardAccelerator.stats.KernelLaunches - launchesBefore
	backwardSyncs := a.CompactForwardAccelerator.stats.KernelSynchronizations - syncsBefore
	a.stats.BackwardCalls++
	a.stats.BackwardNanos += time.Since(start).Nanoseconds()
	a.stats.KernelLaunches += backwardLaunches
	a.stats.KernelSynchronizations += backwardSyncs
	a.stats.LastBackwardLaunches = backwardLaunches
	a.stats.LastBackwardSyncs = backwardSyncs
	return backend.CompactTrainBackwardResult{ResidentGradRefs: a.residentGradientRefsLocked()}, nil
}

func (a *CompactTrainAccelerator) poisonBackwardErrorLocked(err error, launchesBefore, syncsBefore int64, start time.Time) error {
	if err == nil {
		return nil
	}
	err = a.drainKernelError(err)
	backwardLaunches := a.CompactForwardAccelerator.stats.KernelLaunches - launchesBefore
	backwardSyncs := a.CompactForwardAccelerator.stats.KernelSynchronizations - syncsBefore
	a.stepPoisoned = true
	a.stepSealed = false
	a.stats.FallbackOrUnhandled++
	a.stats.KernelLaunches += backwardLaunches
	a.stats.KernelSynchronizations += backwardSyncs
	a.stats.LastBackwardLaunches = backwardLaunches
	a.stats.LastBackwardSyncs = backwardSyncs
	a.stats.BackwardNanos += time.Since(start).Nanoseconds()
	return err
}

type compactTrainFinalOutputDebugResult struct {
	ResidentGradRefs []backend.ResidentGradientRef
	GradHidden       *backend.Tensor
}

type compactTrainFFNBackwardDebugResult struct {
	ResidentGradRefs      []backend.ResidentGradientRef
	GradHidden            *backend.Tensor
	GradAttentionBoundary *backend.Tensor
	Layer                 int
}

type compactTrainLayerBackwardDebugResult struct {
	ResidentGradRefs []backend.ResidentGradientRef
	GradLayerInput   *backend.Tensor
	GradRoPEInput    *backend.Tensor
	Layer            int
}

func (a *CompactTrainAccelerator) runCompactTrainFinalOutputBackwardForDebug(req backend.CompactTrainBackwardRequest) (compactTrainFinalOutputDebugResult, error) {
	if a == nil || a.CompactForwardAccelerator == nil {
		return compactTrainFinalOutputDebugResult{}, fmt.Errorf("cuda compact train accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	start := time.Now()
	arena, err := a.validateBackwardRequestLocked(req)
	if err != nil {
		return compactTrainFinalOutputDebugResult{}, err
	}
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
	if err := a.synchronizeKernelBoundary(); err != nil {
		return compactTrainFinalOutputDebugResult{}, err
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

func (a *CompactTrainAccelerator) runCompactTrainFFNBackwardForDebug(req backend.CompactTrainBackwardRequest) (compactTrainFFNBackwardDebugResult, error) {
	if a == nil || a.CompactForwardAccelerator == nil {
		return compactTrainFFNBackwardDebugResult{}, fmt.Errorf("cuda compact train accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	start := time.Now()
	arena, err := a.validateBackwardRequestLocked(req)
	if err != nil {
		return compactTrainFFNBackwardDebugResult{}, err
	}
	shape := arena.shape
	if shape.Layers <= 0 {
		return compactTrainFFNBackwardDebugResult{}, fmt.Errorf("cuda compact train ffn debug backward requires at least one layer")
	}
	launchesBefore := a.CompactForwardAccelerator.stats.KernelLaunches
	syncsBefore := a.CompactForwardAccelerator.stats.KernelSynchronizations
	if err := a.runFinalOutputBackwardLocked(req, arena, shape); err != nil {
		return compactTrainFFNBackwardDebugResult{}, err
	}
	layerIdx := shape.Layers - 1
	if err := a.runLayerFFNBackwardLocked(layerIdx, arena, shape); err != nil {
		return compactTrainFFNBackwardDebugResult{}, err
	}
	if err := a.synchronizeKernelBoundary(); err != nil {
		return compactTrainFFNBackwardDebugResult{}, err
	}
	hidden := make([]float32, shape.Batch*shape.Tokens*shape.ModelDim)
	if err := a.device.downloadFloat32(hidden, arena.gradHidden); err != nil {
		return compactTrainFFNBackwardDebugResult{}, err
	}
	boundary := make([]float32, shape.Batch*shape.Tokens*shape.ModelDim)
	if err := a.device.downloadFloat32(boundary, arena.gradAttention); err != nil {
		return compactTrainFFNBackwardDebugResult{}, err
	}
	if err := a.consumeHandleLocked(req.Handle); err != nil {
		return compactTrainFFNBackwardDebugResult{}, err
	}
	backwardLaunches := a.CompactForwardAccelerator.stats.KernelLaunches - launchesBefore
	backwardSyncs := a.CompactForwardAccelerator.stats.KernelSynchronizations - syncsBefore
	a.stats.BackwardCalls++
	a.stats.BackwardNanos += time.Since(start).Nanoseconds()
	a.stats.DownloadedBytes += int64((len(hidden) + len(boundary)) * 4)
	a.stats.KernelLaunches += backwardLaunches
	a.stats.KernelSynchronizations += backwardSyncs
	a.stats.LastBackwardLaunches = backwardLaunches
	a.stats.LastBackwardSyncs = backwardSyncs
	return compactTrainFFNBackwardDebugResult{
		ResidentGradRefs:      a.residentGradientRefsLocked(),
		GradHidden:            backend.NewTensorF32([]int{shape.Batch, shape.Tokens, shape.ModelDim}, hidden),
		GradAttentionBoundary: backend.NewTensorF32([]int{shape.Batch, shape.Tokens, shape.ModelDim}, boundary),
		Layer:                 layerIdx,
	}, nil
}

func (a *CompactTrainAccelerator) runCompactTrainLayerBackwardForDebug(req backend.CompactTrainBackwardRequest) (compactTrainLayerBackwardDebugResult, error) {
	if a == nil || a.CompactForwardAccelerator == nil {
		return compactTrainLayerBackwardDebugResult{}, fmt.Errorf("cuda compact train accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	start := time.Now()
	arena, err := a.validateBackwardRequestLocked(req)
	if err != nil {
		return compactTrainLayerBackwardDebugResult{}, err
	}
	shape := arena.shape
	if shape.Layers <= 0 {
		return compactTrainLayerBackwardDebugResult{}, fmt.Errorf("cuda compact train layer debug backward requires at least one layer")
	}
	launchesBefore := a.CompactForwardAccelerator.stats.KernelLaunches
	syncsBefore := a.CompactForwardAccelerator.stats.KernelSynchronizations
	if err := a.runFinalOutputBackwardLocked(req, arena, shape); err != nil {
		return compactTrainLayerBackwardDebugResult{}, err
	}
	layerIdx := shape.Layers - 1
	if err := a.runLayerFFNBackwardLocked(layerIdx, arena, shape); err != nil {
		return compactTrainLayerBackwardDebugResult{}, err
	}
	if err := a.runLayerAttentionBackwardLocked(layerIdx, arena, shape); err != nil {
		return compactTrainLayerBackwardDebugResult{}, err
	}
	if err := a.synchronizeKernelBoundary(); err != nil {
		return compactTrainLayerBackwardDebugResult{}, err
	}
	layerInput := make([]float32, shape.Batch*shape.Tokens*shape.ModelDim)
	if err := a.device.downloadFloat32(layerInput, arena.gradHidden); err != nil {
		return compactTrainLayerBackwardDebugResult{}, err
	}
	var ropeInput []float32
	if layerIdx == 0 && a.useRoPE {
		if err := a.launchRoPETranspose(arena.gradHidden, arena.gradRoPE, shape.Batch*shape.Tokens, shape.ModelDim, shape.Tokens); err != nil {
			return compactTrainLayerBackwardDebugResult{}, err
		}
		if err := a.synchronizeKernelBoundary(); err != nil {
			return compactTrainLayerBackwardDebugResult{}, err
		}
		ropeInput = make([]float32, shape.Batch*shape.Tokens*shape.ModelDim)
		if err := a.device.downloadFloat32(ropeInput, arena.gradRoPE); err != nil {
			return compactTrainLayerBackwardDebugResult{}, err
		}
	}
	if err := a.consumeHandleLocked(req.Handle); err != nil {
		return compactTrainLayerBackwardDebugResult{}, err
	}
	backwardLaunches := a.CompactForwardAccelerator.stats.KernelLaunches - launchesBefore
	backwardSyncs := a.CompactForwardAccelerator.stats.KernelSynchronizations - syncsBefore
	a.stats.BackwardCalls++
	a.stats.BackwardNanos += time.Since(start).Nanoseconds()
	a.stats.DownloadedBytes += int64(len(layerInput) * 4)
	if ropeInput != nil {
		a.stats.DownloadedBytes += int64(len(ropeInput) * 4)
	}
	a.stats.KernelLaunches += backwardLaunches
	a.stats.KernelSynchronizations += backwardSyncs
	a.stats.LastBackwardLaunches = backwardLaunches
	a.stats.LastBackwardSyncs = backwardSyncs
	result := compactTrainLayerBackwardDebugResult{
		ResidentGradRefs: a.residentGradientRefsLocked(),
		GradLayerInput:   backend.NewTensorF32([]int{shape.Batch, shape.Tokens, shape.ModelDim}, layerInput),
		Layer:            layerIdx,
	}
	if ropeInput != nil {
		result.GradRoPEInput = backend.NewTensorF32([]int{shape.Batch, shape.Tokens, shape.ModelDim}, ropeInput)
	}
	return result, nil
}

func (a *CompactTrainAccelerator) runFinalOutputBackwardLocked(req backend.CompactTrainBackwardRequest, arena *compactTrainArena, shape backend.CompactForwardShape) error {
	if err := a.ensureBackwardWorkspaceLocked(arena); err != nil {
		return err
	}
	if err := a.device.copyFloat32ToBuffer(arena.gradPooled, req.GradPooled.F32); err != nil {
		return err
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
		return err
	}
	if shape.HasOutputProjection {
		grad, ok := a.grads[a.outName]
		if !ok || grad == nil || grad.ptr == 0 {
			return fmt.Errorf("cuda compact train output projection gradient %q is not resident", a.outName)
		}
		if err := a.launchFinalProjectionGrad(arena.finalNorm, arena.gradOutputRows, grad.ptr, shape.Batch*shape.Tokens, shape.ModelDim, shape.OutputDim); err != nil {
			return err
		}
	}
	return nil
}

func (a *CompactTrainAccelerator) runLayerAttentionBackwardLocked(layerIdx int, arena *compactTrainArena, shape backend.CompactForwardShape) error {
	if layerIdx < 0 || layerIdx >= len(arena.layers) || layerIdx >= len(a.bindings.layer) || layerIdx >= len(a.layers) {
		return fmt.Errorf("cuda compact train layer %d is out of range", layerIdx)
	}
	if err := a.ensureBackwardWorkspaceLocked(arena); err != nil {
		return err
	}
	rows, d := shape.Batch*shape.Tokens, shape.ModelDim
	layer := arena.layers[layerIdx]
	binding := a.bindings.layer[layerIdx]
	names := a.layers[layerIdx]
	attnOGrad, ok := a.grads[names.AttentionO]
	if !ok || attnOGrad == nil || attnOGrad.ptr == 0 {
		return fmt.Errorf("cuda compact train attn_o gradient %q is not resident", names.AttentionO)
	}
	if err := a.launchMatMulLeftTransposeAccum(layer.attnMixed, arena.gradAttention, attnOGrad.ptr, rows, d, d); err != nil {
		return err
	}
	if err := a.launchMatMulRightTranspose(arena.gradAttention, binding.o.ptr, arena.gradMixed, rows, d, d); err != nil {
		return err
	}
	if err := a.launchAttentionBackward(arena.gradMixed, layer.attnQ, layer.attnK, layer.attnV, layer.attnScores, arena.gradQ, arena.gradK, arena.gradV, shape); err != nil {
		return err
	}
	for _, item := range []struct {
		name string
		grad C.CUdeviceptr
	}{
		{names.AttentionQ, arena.gradQ},
		{names.AttentionK, arena.gradK},
		{names.AttentionV, arena.gradV},
	} {
		resident, ok := a.grads[item.name]
		if !ok || resident == nil || resident.ptr == 0 {
			return fmt.Errorf("cuda compact train attention gradient %q is not resident", item.name)
		}
		if err := a.launchMatMulLeftTransposeAccum(layer.input, item.grad, resident.ptr, rows, d, d); err != nil {
			return err
		}
	}
	if err := a.launchMatMulRightTranspose(arena.gradQ, binding.q.ptr, arena.gradHiddenFromFFN, rows, d, d); err != nil {
		return err
	}
	if err := a.launchAdd(arena.gradAttention, arena.gradHiddenFromFFN, arena.gradFFNResidual, rows*d); err != nil {
		return err
	}
	if err := a.launchMatMulRightTranspose(arena.gradK, binding.k.ptr, arena.gradHiddenFromFFN, rows, d, d); err != nil {
		return err
	}
	if err := a.launchAdd(arena.gradFFNResidual, arena.gradHiddenFromFFN, arena.gradFFNResidual, rows*d); err != nil {
		return err
	}
	if err := a.launchMatMulRightTranspose(arena.gradV, binding.v.ptr, arena.gradHiddenFromFFN, rows, d, d); err != nil {
		return err
	}
	if err := a.launchAdd(arena.gradFFNResidual, arena.gradHiddenFromFFN, arena.gradHidden, rows*d); err != nil {
		return err
	}
	return nil
}

func (a *CompactTrainAccelerator) runLayerFFNBackwardLocked(layerIdx int, arena *compactTrainArena, shape backend.CompactForwardShape) error {
	if layerIdx < 0 || layerIdx >= len(arena.layers) || layerIdx >= len(a.bindings.layer) || layerIdx >= len(a.layers) {
		return fmt.Errorf("cuda compact train layer %d is out of range", layerIdx)
	}
	if err := a.ensureBackwardWorkspaceLocked(arena); err != nil {
		return err
	}
	rows, d, h := shape.Batch*shape.Tokens, shape.ModelDim, shape.FFNDim
	layer := arena.layers[layerIdx]
	binding := a.bindings.layer[layerIdx]
	names := a.layers[layerIdx]
	if err := a.launchLayerNormBackward(arena.gradHidden, layer.projected, layer.ffnResidual, arena.gradFFNResidual, rows, d); err != nil {
		return err
	}
	ffnDownGrad, ok := a.grads[names.FFNDown]
	if !ok || ffnDownGrad == nil || ffnDownGrad.ptr == 0 {
		return fmt.Errorf("cuda compact train ffn_down gradient %q is not resident", names.FFNDown)
	}
	if err := a.launchMatMulLeftTransposeAccum(layer.activated, arena.gradFFNResidual, ffnDownGrad.ptr, rows, h, d); err != nil {
		return err
	}
	if err := a.launchMatMulRightTranspose(arena.gradFFNResidual, binding.down.ptr, arena.gradActivatedPre, rows, h, d); err != nil {
		return err
	}
	if err := a.launchGELUBackward(arena.gradActivatedPre, layer.ffnHidden, arena.gradActivated, rows*h, arena.geluFast); err != nil {
		return err
	}
	ffnUpGrad, ok := a.grads[names.FFNUp]
	if !ok || ffnUpGrad == nil || ffnUpGrad.ptr == 0 {
		return fmt.Errorf("cuda compact train ffn_up gradient %q is not resident", names.FFNUp)
	}
	if err := a.launchMatMulLeftTransposeAccum(layer.hidden, arena.gradActivated, ffnUpGrad.ptr, rows, d, h); err != nil {
		return err
	}
	if err := a.launchMatMulRightTranspose(arena.gradActivated, binding.up.ptr, arena.gradHiddenFromFFN, rows, d, h); err != nil {
		return err
	}
	if err := a.launchAdd(arena.gradFFNResidual, arena.gradHiddenFromFFN, arena.gradHidden, rows*d); err != nil {
		return err
	}
	if err := a.launchLayerNormBackward(arena.gradHidden, layer.hidden, layer.attnResidual, arena.gradAttention, rows, d); err != nil {
		return err
	}
	return nil
}

func (a *CompactTrainAccelerator) runInputScatterBackwardLocked(gradInput C.CUdeviceptr, arena *compactTrainArena, shape backend.CompactForwardShape) error {
	tokenGrad, ok := a.grads[a.tokenName]
	if !ok || tokenGrad == nil || tokenGrad.ptr == 0 {
		return fmt.Errorf("cuda compact train token gradient %q is not resident", a.tokenName)
	}
	rolePtr := C.CUdeviceptr(0)
	roleRows := 0
	if a.roleName != "" && a.bindings.role.ptr != 0 {
		roleGrad, ok := a.grads[a.roleName]
		if !ok || roleGrad == nil || roleGrad.ptr == 0 {
			return fmt.Errorf("cuda compact train role gradient %q is not resident", a.roleName)
		}
		rolePtr = roleGrad.ptr
		roleRows = a.bindings.role.rows
	}
	return a.launchInputScatter(gradInput, arena.tokens, arena.roles, tokenGrad.ptr, rolePtr, shape.Batch, shape.Tokens, shape.ModelDim, a.bindings.token.rows, roleRows)
}

func lockCompactTrainResidentRefs(refs []backend.CompactForwardResidentRef) (map[string]compactTrainResidentRefSnapshot, func(), error) {
	if len(refs) == 0 {
		return nil, func() {}, nil
	}
	tokens := make(map[string]*optimizerResidentParameterToken, len(refs))
	var owner *optimizerAccelerator
	for _, ref := range refs {
		if ref.Name == "" {
			return nil, nil, fmt.Errorf("cuda compact train resident ref has empty name")
		}
		if _, exists := tokens[ref.Name]; exists {
			return nil, nil, fmt.Errorf("cuda compact train resident ref %q is duplicated", ref.Name)
		}
		if ref.Backend != eosartifact.BackendCUDA {
			return nil, nil, fmt.Errorf("cuda compact train resident ref %q backend %q, want cuda", ref.Name, ref.Backend)
		}
		if ref.Token == nil {
			return nil, nil, fmt.Errorf("cuda compact train resident ref %q token is nil", ref.Name)
		}
		if ref.Token.Backend() != eosartifact.BackendCUDA {
			return nil, nil, fmt.Errorf("cuda compact train resident ref %q token backend %q, want cuda", ref.Name, ref.Token.Backend())
		}
		token, ok := ref.Token.(*optimizerResidentParameterToken)
		if !ok || token == nil {
			return nil, nil, fmt.Errorf("cuda compact train resident ref %q has invalid cuda token", ref.Name)
		}
		if token.owner == nil {
			return nil, nil, fmt.Errorf("cuda compact train resident ref %q owner is nil", ref.Name)
		}
		if token.name != ref.Name {
			return nil, nil, fmt.Errorf("cuda compact train resident ref %q token name %q mismatch", ref.Name, token.name)
		}
		if owner == nil {
			owner = token.owner
		} else if owner != token.owner {
			return nil, nil, fmt.Errorf("cuda compact train resident refs span multiple optimizer owners")
		}
		tokens[ref.Name] = token
	}
	owner.mu.RLock()
	unlock := func() { owner.mu.RUnlock() }
	if owner.device == nil {
		unlock()
		return nil, nil, fmt.Errorf("cuda compact train resident optimizer is closed")
	}
	out := make(map[string]compactTrainResidentRefSnapshot, len(refs))
	for _, ref := range refs {
		token := tokens[ref.Name]
		state, ok := owner.resident[ref.Name]
		if !ok {
			unlock()
			return nil, nil, fmt.Errorf("cuda compact train resident token %q is no longer resident", ref.Name)
		}
		if state.token != token {
			unlock()
			return nil, nil, fmt.Errorf("cuda compact train resident token %q is stale", ref.Name)
		}
		if state.generation != token.generation || state.generation != ref.Token.Generation() {
			unlock()
			return nil, nil, fmt.Errorf("cuda compact train resident token %q generation %d is stale, current %d", ref.Name, token.generation, state.generation)
		}
		if ref.Elements != state.elements {
			unlock()
			return nil, nil, fmt.Errorf("cuda compact train resident ref %q elements %d, want %d", ref.Name, ref.Elements, state.elements)
		}
		out[ref.Name] = compactTrainResidentRefSnapshot{
			ref:        ref,
			token:      token,
			owner:      owner,
			generation: state.generation,
			param:      state.param,
			elements:   state.elements,
		}
	}
	return out, unlock, nil
}

func (a *CompactTrainAccelerator) validateBeginResidentRefSetLocked(byName map[string]compactTrainResidentRefSnapshot) error {
	required := a.requiredBeginResidentNamesLocked()
	requiredSet := make(map[string]bool, len(required))
	for _, name := range required {
		requiredSet[name] = true
		snapshot, ok := byName[name]
		if !ok {
			return fmt.Errorf("cuda compact train resident ref %q is missing", name)
		}
		binding, ok := a.bindingForName(name)
		if !ok {
			return fmt.Errorf("cuda compact train resident binding %q is missing", name)
		}
		if snapshot.ref.Elements != binding.elements || snapshot.elements != binding.elements {
			return fmt.Errorf("cuda compact train resident ref %q elements %d, want %d", name, snapshot.ref.Elements, binding.elements)
		}
		if binding.ptr != snapshot.param {
			return fmt.Errorf("cuda compact train resident ref %q no longer matches bound device state", name)
		}
		bridged := a.bridged[name]
		if bridged == nil || bridged != snapshot.token || bridged.generation != snapshot.generation {
			return fmt.Errorf("cuda compact train resident ref %q generation %d does not match bound generation", name, snapshot.generation)
		}
	}
	for name := range byName {
		if !requiredSet[name] {
			return fmt.Errorf("cuda compact train resident ref %q is unexpected", name)
		}
	}
	return nil
}

func (a *CompactTrainAccelerator) validateTrainForwardResidentRefsExactLocked(refs []backend.CompactForwardResidentRef, shape backend.CompactForwardShape) error {
	required := a.requiredResidentNames(shape)
	requiredSet := make(map[string]bool, len(required))
	for _, name := range required {
		requiredSet[name] = true
	}
	for _, ref := range refs {
		if !requiredSet[ref.Name] {
			return fmt.Errorf("cuda compact train resident ref %q is unexpected", ref.Name)
		}
	}
	return nil
}

func (a *CompactTrainAccelerator) requiredBeginResidentNamesLocked() []string {
	return a.requiredResidentNames(backend.CompactForwardShape{HasOutputProjection: a.outName != ""})
}

func (a *CompactTrainAccelerator) validateBackwardRequestLocked(req backend.CompactTrainBackwardRequest) (*compactTrainArena, error) {
	if a.device == nil || a.closed {
		return nil, fmt.Errorf("cuda compact train accelerator is closed")
	}
	if !a.stepActive {
		return nil, fmt.Errorf("cuda compact train step is not active")
	}
	if a.stepPoisoned {
		return nil, fmt.Errorf("cuda compact train step %d is poisoned", a.stepID)
	}
	arena, err := a.validateHandleLocked(req.Handle)
	if err != nil {
		return nil, err
	}
	shape := arena.shape
	if req.GradPooled == nil || req.GradPooled.DType != "f32" || len(req.GradPooled.Shape) != 2 || req.GradPooled.Shape[0] != shape.Batch || req.GradPooled.Shape[1] != shape.OutputDim || len(req.GradPooled.F32) != shape.Batch*shape.OutputDim {
		return nil, fmt.Errorf("cuda compact train pooled gradient shape %v, want [%d %d]", tensorShapeForError(req.GradPooled), shape.Batch, shape.OutputDim)
	}
	if err := a.validateCurrentStepGradSetLocked(shape); err != nil {
		return nil, err
	}
	return arena, nil
}

func (a *CompactTrainAccelerator) validateResidentGradientRefLocked(ref backend.ResidentGradientRef) (*compactTrainGradient, error) {
	if a.device == nil || a.closed {
		return nil, fmt.Errorf("cuda compact train accelerator is closed")
	}
	if !a.stepActive && !a.stepSealed {
		return nil, fmt.Errorf("cuda compact train step is not active")
	}
	if a.stepPoisoned {
		return nil, fmt.Errorf("cuda compact train resident gradient %q step %d is poisoned", ref.Name, ref.StepID)
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

func (a *CompactTrainAccelerator) validateCurrentStepGradSetLocked(shape backend.CompactForwardShape) error {
	if !a.stepActive {
		return fmt.Errorf("cuda compact train step is not active")
	}
	if a.stepPoisoned {
		return fmt.Errorf("cuda compact train step %d is poisoned", a.stepID)
	}
	required := a.requiredResidentNames(shape)
	requiredSet := make(map[string]bool, len(required))
	for _, name := range required {
		requiredSet[name] = true
		grad := a.grads[name]
		if grad == nil {
			return fmt.Errorf("cuda compact train resident gradient %q is missing", name)
		}
		binding, ok := a.bindingForName(name)
		if !ok {
			return fmt.Errorf("cuda compact train resident binding %q is missing", name)
		}
		if grad.ptr == 0 || grad.token == nil || grad.token.owner != a || grad.token.name != name {
			return fmt.Errorf("cuda compact train resident gradient %q is not live", name)
		}
		if grad.stepID != a.stepID || grad.token.stepID != a.stepID {
			return fmt.Errorf("cuda compact train resident gradient %q step is stale", name)
		}
		if grad.generation != a.gradGen || grad.token.generation != a.gradGen {
			return fmt.Errorf("cuda compact train resident gradient %q generation is stale", name)
		}
		if grad.elements != binding.elements || grad.token.elements != binding.elements {
			return fmt.Errorf("cuda compact train resident gradient %q elements %d, want %d", name, grad.elements, binding.elements)
		}
	}
	for name := range a.grads {
		if !requiredSet[name] {
			return fmt.Errorf("cuda compact train resident gradient %q is unexpected", name)
		}
	}
	if len(a.grads) != len(required) {
		return fmt.Errorf("cuda compact train resident gradient count %d, want %d", len(a.grads), len(required))
	}
	return nil
}

func (a *CompactTrainAccelerator) validateHandleLocked(handle backend.CompactTrainHandle) (*compactTrainArena, error) {
	if !a.stepActive {
		return nil, fmt.Errorf("cuda compact train step is not active")
	}
	token, ok := handle.Token.(*compactTrainHandleToken)
	if !ok || token == nil {
		return nil, fmt.Errorf("cuda compact train handle has invalid token")
	}
	if token.owner != a {
		return nil, fmt.Errorf("cuda compact train handle owner mismatch")
	}
	if handle.Backend != eosartifact.BackendCUDA || token.Backend() != eosartifact.BackendCUDA {
		return nil, fmt.Errorf("cuda compact train handle backend mismatch")
	}
	if handle.StepID != a.stepID || token.StepID() != a.stepID || token.StepID() != handle.StepID {
		return nil, fmt.Errorf("cuda compact train handle step %d is stale, current %d", handle.StepID, a.stepID)
	}
	arena := a.arenas[token]
	if arena == nil || arena.token != token || arena.id != token.id || arena.generation != handle.Generation || token.Generation() != handle.Generation {
		return nil, fmt.Errorf("cuda compact train handle is stale")
	}
	if handle.Shape != arena.shape {
		return nil, fmt.Errorf("cuda compact train handle shape mismatch")
	}
	if !arena.live || !token.Alive() {
		return nil, fmt.Errorf("cuda compact train handle already released")
	}
	return arena, nil
}

func (a *CompactTrainAccelerator) consumeHandleLocked(handle backend.CompactTrainHandle) error {
	arena, err := a.validateHandleLocked(handle)
	if err != nil {
		return err
	}
	token := handle.Token.(*compactTrainHandleToken)
	if !token.alive.CompareAndSwap(true, false) {
		return fmt.Errorf("cuda compact train handle already released")
	}
	arena.live = false
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
	if err := alloc(&arena.gradFFNResidual, rows*shape.ModelDim); err != nil {
		return err
	}
	if err := alloc(&arena.gradActivatedPre, rows*shape.FFNDim); err != nil {
		return err
	}
	if err := alloc(&arena.gradActivated, rows*shape.FFNDim); err != nil {
		return err
	}
	if err := alloc(&arena.gradHiddenFromFFN, rows*shape.ModelDim); err != nil {
		return err
	}
	if err := alloc(&arena.gradAttention, rows*shape.ModelDim); err != nil {
		return err
	}
	if err := alloc(&arena.gradMixed, rows*shape.ModelDim); err != nil {
		return err
	}
	if err := alloc(&arena.gradQ, rows*shape.ModelDim); err != nil {
		return err
	}
	if err := alloc(&arena.gradK, rows*shape.ModelDim); err != nil {
		return err
	}
	if err := alloc(&arena.gradV, rows*shape.ModelDim); err != nil {
		return err
	}
	if err := alloc(&arena.gradRoPE, rows*shape.ModelDim); err != nil {
		return err
	}
	a.refreshArenaStatsLocked()
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
	return a.recordKernelLaunch()
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
	return a.recordKernelLaunch()
}

func (a *CompactTrainAccelerator) launchLayerNormBackward(gradOut, normalized, pre, out C.CUdeviceptr, rows, cols int) error {
	grid, block, err := checkedLaunch1D("cuda compact train layernorm backward", rows, 128)
	if err != nil {
		return err
	}
	var errStr *C.char
	if C.eosCudaLaunchCompactTrainLayerNormBackward(a.device.ptr, a.trainKernels.layerNormBackward.ptr, grid, block, gradOut, normalized, pre, out, C.int(rows), C.int(cols), &errStr) != 0 {
		return cStringError(errStr)
	}
	return a.recordKernelLaunch()
}

func (a *CompactTrainAccelerator) launchMatMulLeftTransposeAccum(lhs, gradOut, gradWeight C.CUdeviceptr, rows, inDim, outDim int) error {
	grid, block, err := checkedLaunch1D("cuda compact train matmul left transpose accum", inDim*outDim, 128)
	if err != nil {
		return err
	}
	var errStr *C.char
	if C.eosCudaLaunchCompactTrainMatMulLeftTransposeAccum(a.device.ptr, a.trainKernels.matMulLeftTAccum.ptr, grid, block, lhs, gradOut, gradWeight, C.int(rows), C.int(inDim), C.int(outDim), &errStr) != 0 {
		return cStringError(errStr)
	}
	return a.recordKernelLaunch()
}

func (a *CompactTrainAccelerator) launchMatMulRightTranspose(gradOut, weight, gradIn C.CUdeviceptr, rows, inDim, outDim int) error {
	grid, block, err := checkedLaunch1D("cuda compact train matmul right transpose", rows*inDim, 128)
	if err != nil {
		return err
	}
	var errStr *C.char
	if C.eosCudaLaunchCompactTrainMatMulRightTranspose(a.device.ptr, a.trainKernels.matMulRightT.ptr, grid, block, gradOut, weight, gradIn, C.int(rows), C.int(inDim), C.int(outDim), &errStr) != 0 {
		return cStringError(errStr)
	}
	return a.recordKernelLaunch()
}

func (a *CompactTrainAccelerator) launchGELUBackward(gradOut, preAct, out C.CUdeviceptr, elements int, fast bool) error {
	grid, block, err := checkedLaunch1D("cuda compact train gelu backward", elements, 128)
	if err != nil {
		return err
	}
	fastFlag := 0
	if fast {
		fastFlag = 1
	}
	var errStr *C.char
	if C.eosCudaLaunchCompactTrainGELUBackward(a.device.ptr, a.trainKernels.geluBackward.ptr, grid, block, gradOut, preAct, out, C.int(elements), C.int(fastFlag), &errStr) != 0 {
		return cStringError(errStr)
	}
	return a.recordKernelLaunch()
}

func (a *CompactTrainAccelerator) launchAttentionBackward(gradMixed, q, k, v, probs, gradQ, gradK, gradV C.CUdeviceptr, shape backend.CompactForwardShape) error {
	grid, block, err := checkedLaunch1D("cuda compact train attention backward", shape.Batch*shape.Heads, 128)
	if err != nil {
		return err
	}
	var errStr *C.char
	if C.eosCudaLaunchCompactTrainAttentionBackward(a.device.ptr, a.trainKernels.attentionBackward.ptr, grid, block, gradMixed, q, k, v, probs, gradQ, gradK, gradV, C.int(shape.Batch), C.int(shape.Tokens), C.int(shape.ModelDim), C.int(shape.Heads), C.int(shape.HeadDim), &errStr) != 0 {
		return cStringError(errStr)
	}
	return a.recordKernelLaunch()
}

func (a *CompactTrainAccelerator) launchRoPETranspose(src, out C.CUdeviceptr, rows, cols, seq int) error {
	pairs := rows * ((cols + 1) / 2)
	grid, block, err := checkedLaunch1D("cuda compact train rope transpose", pairs, 128)
	if err != nil {
		return err
	}
	var errStr *C.char
	if C.eosCudaLaunchCompactTrainRoPETranspose(a.device.ptr, a.trainKernels.ropeTranspose.ptr, grid, block, src, out, C.int(rows), C.int(cols), C.int(seq), &errStr) != 0 {
		return cStringError(errStr)
	}
	return a.recordKernelLaunch()
}

func (a *CompactTrainAccelerator) launchInputScatter(gradInput, tokens, roles, gradToken, gradRole C.CUdeviceptr, batch, seq, modelDim, vocab, roleRows int) error {
	useRole := 0
	if gradRole != 0 && roleRows > 0 {
		useRole = 1
	}
	total := vocab*modelDim + useRole*roleRows*modelDim
	grid, block, err := checkedLaunch1D("cuda compact train input scatter", total, 128)
	if err != nil {
		return err
	}
	var errStr *C.char
	if C.eosCudaLaunchCompactTrainInputScatter(a.device.ptr, a.trainKernels.inputScatter.ptr, grid, block, gradInput, tokens, roles, gradToken, gradRole, C.int(batch), C.int(seq), C.int(modelDim), C.int(vocab), C.int(roleRows), C.int(useRole), &errStr) != 0 {
		return cStringError(errStr)
	}
	return a.recordKernelLaunch()
}

func (a *CompactTrainAccelerator) launchAdd(left, right, out C.CUdeviceptr, elements int) error {
	grid, block, err := checkedLaunch1D("cuda compact train add", elements, 128)
	if err != nil {
		return err
	}
	var errStr *C.char
	if C.eosCudaLaunchCompactTrainAdd(a.device.ptr, a.trainKernels.add.ptr, grid, block, left, right, out, C.int(elements), &errStr) != 0 {
		return cStringError(errStr)
	}
	return a.recordKernelLaunch()
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

func (a *CompactTrainAccelerator) runCompactTrainForwardLocked(req backend.CompactTrainForwardRequest) (result backend.CompactTrainForwardResult, err error) {
	if a.device == nil || a.closed {
		return backend.CompactTrainForwardResult{}, fmt.Errorf("cuda compact train accelerator is closed")
	}
	shape := req.Shape
	launchesBefore := a.CompactForwardAccelerator.stats.KernelLaunches
	syncsBefore := a.CompactForwardAccelerator.stats.KernelSynchronizations
	var uploaded, pooledBytes, statusBytes, activeBytes int64
	statsPublished := false
	publishForwardStats := func(failed bool) {
		if statsPublished {
			return
		}
		statsPublished = true
		forwardLaunches := a.CompactForwardAccelerator.stats.KernelLaunches - launchesBefore
		forwardSyncs := a.CompactForwardAccelerator.stats.KernelSynchronizations - syncsBefore
		if failed && forwardLaunches == 0 && forwardSyncs == 0 && uploaded == 0 && pooledBytes == 0 && statusBytes == 0 && activeBytes == 0 {
			return
		}
		a.stats.UploadedBytes += uploaded
		a.stats.DownloadedBytes += pooledBytes + statusBytes + activeBytes
		a.stats.PooledDownloadedBytes += pooledBytes
		a.stats.StatusDownloadedBytes += statusBytes + activeBytes
		a.stats.KernelLaunches += forwardLaunches
		a.stats.KernelSynchronizations += forwardSyncs
		a.stats.LastShape = shape
		a.stats.LastForwardLaunches = forwardLaunches
		a.stats.LastForwardSyncs = forwardSyncs
		if failed {
			a.stats.FallbackOrUnhandled++
		}
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
	if err := a.validateCurrentStepGradSetLocked(req.Shape); err != nil {
		return backend.CompactTrainForwardResult{}, err
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
	keepArena := false
	defer func() {
		if !keepArena {
			a.freeArena(arena)
		}
	}()
	defer func() {
		if err != nil {
			err = a.drainKernelError(err)
			publishForwardStats(true)
		}
	}()
	a.nextHandleID++
	arena.id = a.nextHandleID
	arena.generation = a.nextHandleID
	geluFast, _ := compactForwardGELUFast(req.GELUMode)
	arena.geluFast = geluFast
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
	uploaded = int64((len(tokensFlat) + len(masksFlat) + len(req.Roles) + 1) * 4)
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
		if layerIdx == 0 && a.debugForceForwardFailureAfterFirstLayer {
			return backend.CompactTrainForwardResult{}, fmt.Errorf("cuda compact train forced forward failure after first layer")
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
	if err := a.synchronizeKernelBoundary(); err != nil {
		return backend.CompactTrainForwardResult{}, err
	}
	status := []int32{0}
	if err := a.device.downloadInt32(status, arena.status); err != nil {
		return backend.CompactTrainForwardResult{}, err
	}
	statusBytes = int64(4)
	if status[0] != 0 {
		return backend.CompactTrainForwardResult{}, fmt.Errorf("cuda compact train gather status %d", status[0])
	}
	pooled := make([]float32, B*O)
	if err := a.device.downloadFloat32(pooled, arena.finalPooled); err != nil {
		return backend.CompactTrainForwardResult{}, err
	}
	pooledBytes = int64(len(pooled) * 4)
	activeCounts := make([]int32, B)
	if err := a.device.downloadInt32(activeCounts, arena.active); err != nil {
		return backend.CompactTrainForwardResult{}, err
	}
	activeBytes = int64(len(activeCounts) * 4)
	token := &compactTrainHandleToken{owner: a, backend: eosartifact.BackendCUDA, generation: arena.generation, stepID: a.stepID, id: arena.id}
	token.alive.Store(true)
	arena.token = token
	arena.live = true
	a.arenas[token] = arena
	keepArena = true
	_, packedFloats, _ := compactForwardPackedLayout(shape)
	a.stats.HandlesCreated++
	a.stats.LiveHandles++
	a.stats.PackedBytesAvoided += int64(packedFloats * 4)
	publishForwardStats(false)
	a.refreshArenaStatsLocked()
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

func (a *CompactTrainAccelerator) releaseArenasLocked() {
	for token, arena := range a.arenas {
		if arena != nil && arena.live {
			if token != nil && token.alive.CompareAndSwap(true, false) {
				a.stats.HandlesReleased++
			}
			arena.live = false
		}
		a.freeArena(arena)
		delete(a.arenas, token)
	}
	a.stats.ActivationArenaBytes = 0
	a.stats.WorkspaceArenaBytes = 0
	a.stats.LiveHandles = 0
}

func (a *CompactTrainAccelerator) refreshArenaStatsLocked() {
	var activationBytes, workspaceBytes int64
	for _, arena := range a.arenas {
		if arena != nil {
			activationBytes += arena.bytes
			workspaceBytes += arena.workspaceBytes
		}
	}
	a.stats.ActivationArenaBytes = activationBytes
	a.stats.WorkspaceArenaBytes = workspaceBytes
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
	for _, ptr := range []C.CUdeviceptr{arena.tokens, arena.masks, arena.roles, arena.status, arena.input, arena.active, arena.finalNorm, arena.outputRows, arena.preProjectionPooled, arena.finalPooled, arena.gradPooled, arena.gradOutputRows, arena.gradNormalized, arena.gradHidden, arena.gradFFNResidual, arena.gradActivatedPre, arena.gradActivated, arena.gradHiddenFromFFN, arena.gradAttention, arena.gradMixed, arena.gradQ, arena.gradK, arena.gradV, arena.gradRoPE} {
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
