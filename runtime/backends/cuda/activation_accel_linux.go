//go:build linux && cgo

package cuda

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#include <cuda.h>
*/
import "C"

import (
	"fmt"
	"time"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

const geluBackwardMulKernelSource = `
extern "C" __global__ void manta_gelu_backward_mul(
    const float* grad_out,
    const float* pre_act,
    float* out0,
    int elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    float x = pre_act[idx];
    float cubic = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * cubic);
    float tanh_inner = tanhf(inner);
    float sech2 = 1.0f - tanh_inner * tanh_inner;
    float inner_grad = 0.7978845608f * (1.0f + (3.0f * 0.044715f) * x * x);
    float grad = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * inner_grad;
    out0[idx] = grad_out[idx] * grad;
}
`

const softmaxBackwardRowsKernelSource = `
extern "C" __global__ void manta_softmax_backward_rows(
    const float* grad_out,
    const float* probs,
    float* out0,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    int base = row * cols;
    float dot = 0.0f;
    for (int col = 0; col < cols; ++col) {
        dot += grad_out[base + col] * probs[base + col];
    }
    for (int col = 0; col < cols; ++col) {
        out0[base + col] = probs[base + col] * (grad_out[base + col] - dot);
    }
}
`

// forwardSoftmaxRowsKernelSource is the on-device forward row softmax,
// numerically identical to the host softmaxRowsInPlace (max-subtract, exp,
// normalize), in-place, one thread per row. It is the keystone of the
// device-resident forward dataflow: keeping attention scores on the device
// between the scores GEMM and the mixed GEMM removes a download/host-compute/
// upload round-trip and is a prerequisite for whole-forward CUDA-graph capture.
const forwardSoftmaxRowsKernelSource = `
extern "C" __global__ void manta_softmax_forward_rows(
    float* data,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    int base = row * cols;
    float maxVal = data[base];
    for (int col = 1; col < cols; ++col) {
        float v = data[base + col];
        if (v > maxVal) {
            maxVal = v;
        }
    }
    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
        float e = expf(data[base + col] - maxVal);
        data[base + col] = e;
        sum += e;
    }
    if (sum == 0.0f) {
        return;
    }
    float inv = 1.0f / sum;
    for (int col = 0; col < cols; ++col) {
        data[base + col] *= inv;
    }
}
`

// forwardSoftmaxRowsMaskedKernelSource is forwardSoftmaxRowsKernelSource
// plus an inline pre-softmax scale and a per-sequence key mask, numerically
// identical to the host softmaxRowsMaskedColumnsInPlace fed
// scale-premultiplied scores: masked columns are excluded from the row max,
// from the exp/normalize accumulation, and are zeroed in the output --
// matching state.mask's "key" semantics (a masked column excludes that key
// position from every query row's softmax; it does not skip query rows).
// rows is Batch*SeqLen; mask is Batch*SeqLen int32 flags flattened in the
// same batch order as data, one flag per key position, reused across every
// query row of its sequence (row's sequence = row/seqLen). scale=1
// reproduces the stage (a) forward-only caller's already-scaled-Q
// convention unchanged; the attention-resident TRAIN accelerator instead
// passes the real attention score scale and leaves Q/K unscaled, so this
// one kernel serves both callers.
const forwardSoftmaxRowsMaskedKernelSource = `
extern "C" __global__ void manta_softmax_forward_rows_masked(
    float* data,
    const int* mask,
    int rows,
    int cols,
    int seqLen,
    float scale
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    int base = row * cols;
    int maskBase = (row / seqLen) * seqLen;
    float maxVal = -3.4028234663852886e38f;
    int active = 0;
    for (int col = 0; col < cols; ++col) {
        if (mask[maskBase + col] == 0) {
            data[base + col] = 0.0f;
            continue;
        }
        float v = data[base + col] * scale;
        data[base + col] = v;
        if (active == 0 || v > maxVal) {
            maxVal = v;
        }
        active = 1;
    }
    if (active == 0) {
        for (int col = 0; col < cols; ++col) {
            data[base + col] = 0.0f;
        }
        return;
    }
    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
        if (mask[maskBase + col] == 0) {
            continue;
        }
        float e = expf(data[base + col] - maxVal);
        data[base + col] = e;
        sum += e;
    }
    if (sum == 0.0f) {
        return;
    }
    float inv = 1.0f / sum;
    for (int col = 0; col < cols; ++col) {
        if (mask[maskBase + col] != 0) {
            data[base + col] *= inv;
        }
    }
}
`

// attnSoftmaxBackwardScaledRowsKernelSource is softmaxBackwardRowsKernelSource
// plus a trailing multiply by scale, fused into one kernel/one launch. It
// reproduces the host backpropAttentionSequences pair of steps --
// softmaxBackward(gradScores, probs) then scaleFloat32Slice(_, scoreScale)
// -- exactly: probs already carries zeros at masked columns (from
// forwardSoftmaxRowsMaskedKernelSource or its host equivalent), so the
// standard softmax-Jacobian dot product already excludes masked columns and
// this kernel needs no mask input of its own.
const attnSoftmaxBackwardScaledRowsKernelSource = `
extern "C" __global__ void manta_attn_softmax_backward_scaled_rows(
    const float* gradOut,
    const float* probs,
    float* out0,
    int rows,
    int cols,
    float scale
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    int base = row * cols;
    float dot = 0.0f;
    for (int col = 0; col < cols; ++col) {
        dot += gradOut[base + col] * probs[base + col];
    }
    for (int col = 0; col < cols; ++col) {
        out0[base + col] = scale * (probs[base + col] * (gradOut[base + col] - dot));
    }
}
`

// forwardGeluKernelSource is the on-device forward GELU (tanh approximation),
// numerically matching the host geluForward.
const forwardGeluKernelSource = `
extern "C" __global__ void manta_gelu_forward(
    const float* src,
    float* dst,
    int elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    float x = src[idx];
    float cubic = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * cubic);
    dst[idx] = 0.5f * x * (1.0f + tanhf(inner));
}
`

// forwardLayerNormRowsKernelSource is the on-device forward row layernorm
// (no learnable scale/shift), matching the host layerNormRow.
const forwardLayerNormRowsKernelSource = `
extern "C" __global__ void manta_layernorm_forward_rows(
    const float* src,
    float* dst,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    int base = row * cols;
    float mean = 0.0f;
    for (int col = 0; col < cols; ++col) {
        mean += src[base + col];
    }
    mean /= (float)cols;
    float variance = 0.0f;
    for (int col = 0; col < cols; ++col) {
        float centered = src[base + col] - mean;
        variance += centered * centered;
    }
    variance /= (float)cols;
    float inv_std = 1.0f / sqrtf(variance + 1e-5f);
    for (int col = 0; col < cols; ++col) {
        dst[base + col] = (src[base + col] - mean) * inv_std;
    }
}
`

// forwardResidualAddKernelSource is the on-device element-wise residual add
// (out = a + b), keeping residual streams on the device between GEMMs.
const forwardResidualAddKernelSource = `
extern "C" __global__ void manta_residual_add(
    const float* a,
    const float* b,
    float* out0,
    int elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    out0[idx] = a[idx] + b[idx];
}
`

const layerNormBackwardRowsKernelSource = `
extern "C" __global__ void manta_layernorm_backward_rows(
    const float* grad_out,
    const float* normalized,
    const float* pre,
    float* out0,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    int base = row * cols;
    float mean = 0.0f;
    for (int col = 0; col < cols; ++col) {
        mean += pre[base + col];
    }
    mean /= (float)cols;
    float variance = 0.0f;
    for (int col = 0; col < cols; ++col) {
        float centered = pre[base + col] - mean;
        variance += centered * centered;
    }
    variance /= (float)cols;
    float inv_std = rsqrtf(variance + 1e-5f);
    float sum_grad = 0.0f;
    float sum_grad_norm = 0.0f;
    for (int col = 0; col < cols; ++col) {
        float g = grad_out[base + col];
        sum_grad += g;
        sum_grad_norm += g * normalized[base + col];
    }
    float n = (float)cols;
    for (int col = 0; col < cols; ++col) {
        out0[base + col] = (inv_std / n) * (n * grad_out[base + col] - sum_grad - normalized[base + col] * sum_grad_norm);
    }
}
`

type activationAccelerator struct {
	device          *deviceRuntime
	geluKernel      *auxKernel
	softmaxKernel   *auxKernel
	layerNormKernel *auxKernel
	stats           backend.ActivationAcceleratorStats
	bound           map[string]residentActivationTensor
}

type residentActivationTensor struct {
	ptr      C.CUdeviceptr
	rows     int
	cols     int
	elements int
}

func init() {
	backend.RegisterActivationAccelerator(eosartifact.BackendCUDA, NewActivationAccelerator)
}

func NewActivationAccelerator() (backend.ActivationAccelerator, error) {
	device, err := newDeviceRuntime()
	if err != nil {
		return nil, err
	}
	if device == nil {
		return nil, nil
	}
	kernel, err := device.compileAuxKernel(geluBackwardMulKernelSource, "manta_gelu_backward_mul")
	if err != nil {
		device.close()
		return nil, err
	}
	softmaxKernel, err := device.compileAuxKernel(softmaxBackwardRowsKernelSource, "manta_softmax_backward_rows")
	if err != nil {
		device.destroyAuxKernel(kernel)
		device.close()
		return nil, err
	}
	layerNormKernel, err := device.compileAuxKernel(layerNormBackwardRowsKernelSource, "manta_layernorm_backward_rows")
	if err != nil {
		device.destroyAuxKernel(kernel)
		device.destroyAuxKernel(softmaxKernel)
		device.close()
		return nil, err
	}
	return &activationAccelerator{device: device, geluKernel: kernel, softmaxKernel: softmaxKernel, layerNormKernel: layerNormKernel, bound: map[string]residentActivationTensor{}}, nil
}

func (a *activationAccelerator) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (a *activationAccelerator) Stats() backend.ActivationAcceleratorStats {
	if a == nil {
		return backend.ActivationAcceleratorStats{}
	}
	stats := a.stats
	stats.BoundTensors = int64(len(a.bound))
	return stats
}

func (a *activationAccelerator) BindTensor(name string, tensor *backend.Tensor) error {
	if a == nil || a.device == nil {
		return fmt.Errorf("cuda activation accelerator is not initialized")
	}
	if name == "" {
		return fmt.Errorf("cuda activation binding name is required")
	}
	rows, cols, err := rowWiseShape(tensor)
	if err != nil {
		return err
	}
	elements := len(tensor.F32)
	if resident, ok := a.bound[name]; ok {
		if resident.elements == elements {
			if err := a.device.copyFloat32ToBuffer(resident.ptr, tensor.F32); err != nil {
				return err
			}
			resident.rows = rows
			resident.cols = cols
			a.bound[name] = resident
			a.stats.BindCalls++
			a.stats.UploadedBytes += int64(elements * 4)
			a.stats.BoundTensors = int64(len(a.bound))
			return nil
		}
		_ = a.device.freeBuffer(resident.ptr)
		delete(a.bound, name)
	}
	ptr, err := a.device.uploadFloat32(tensor.F32)
	if err != nil {
		return err
	}
	a.bound[name] = residentActivationTensor{
		ptr:      ptr,
		rows:     rows,
		cols:     cols,
		elements: elements,
	}
	a.stats.BindCalls++
	a.stats.UploadedBytes += int64(elements * 4)
	a.stats.BoundTensors = int64(len(a.bound))
	return nil
}

func (a *activationAccelerator) UnbindTensor(name string) error {
	if a == nil || a.device == nil {
		return fmt.Errorf("cuda activation accelerator is not initialized")
	}
	resident, ok := a.bound[name]
	if !ok {
		return nil
	}
	if err := a.device.freeBuffer(resident.ptr); err != nil {
		return err
	}
	delete(a.bound, name)
	a.stats.BoundTensors = int64(len(a.bound))
	return nil
}

func (a *activationAccelerator) RunGELUBackwardMul(gradOut, preAct *backend.Tensor) (*backend.Tensor, error) {
	if a == nil || a.device == nil || a.geluKernel == nil {
		return nil, fmt.Errorf("cuda activation accelerator is not initialized")
	}
	if gradOut == nil || preAct == nil || !gradOut.EqualShape(preAct) {
		return nil, fmt.Errorf("cuda gelu backward requires matching grad and pre-activation tensors")
	}
	rows, cols, err := rowWiseShape(gradOut, preAct)
	if err != nil {
		return nil, err
	}
	return a.runGELUBackwardMul(gradOut, rows, cols, nil, preAct)
}

func (a *activationAccelerator) RunGELUBackwardMulWithBoundPreAct(gradOut *backend.Tensor, preActName string) (*backend.Tensor, error) {
	if a == nil || a.device == nil || a.geluKernel == nil {
		return nil, fmt.Errorf("cuda activation accelerator is not initialized")
	}
	resident, ok := a.bound[preActName]
	if !ok {
		return nil, fmt.Errorf("cuda activation binding %q is not resident", preActName)
	}
	if gradOut == nil || gradOut.Rank() != 2 || gradOut.Shape[0] != resident.rows || gradOut.Shape[1] != resident.cols {
		return nil, fmt.Errorf("cuda gelu backward requires grad shape [%d %d] to match resident binding %q", resident.rows, resident.cols, preActName)
	}
	return a.runGELUBackwardMul(gradOut, resident.rows, resident.cols, &resident, nil)
}

func (a *activationAccelerator) RunSoftmaxBackwardRows(gradOut, probs *backend.Tensor) (*backend.Tensor, error) {
	if a == nil || a.device == nil || a.softmaxKernel == nil {
		return nil, fmt.Errorf("cuda activation accelerator is not initialized")
	}
	rows, cols, err := rowWiseShape(gradOut, probs)
	if err != nil {
		return nil, err
	}
	return a.runSoftmaxBackwardRows(gradOut, rows, cols, nil, probs)
}

func (a *activationAccelerator) RunSoftmaxBackwardRowsWithBoundProbs(gradOut *backend.Tensor, probsName string) (*backend.Tensor, error) {
	if a == nil || a.device == nil || a.softmaxKernel == nil {
		return nil, fmt.Errorf("cuda activation accelerator is not initialized")
	}
	resident, ok := a.bound[probsName]
	if !ok {
		return nil, fmt.Errorf("cuda activation binding %q is not resident", probsName)
	}
	if gradOut == nil || gradOut.Rank() != 2 || gradOut.Shape[0] != resident.rows || gradOut.Shape[1] != resident.cols {
		return nil, fmt.Errorf("cuda softmax backward requires grad shape [%d %d] to match resident binding %q", resident.rows, resident.cols, probsName)
	}
	return a.runSoftmaxBackwardRows(gradOut, resident.rows, resident.cols, &resident, nil)
}

func (a *activationAccelerator) RunLayerNormBackwardRows(gradOut, normalized, pre *backend.Tensor) (*backend.Tensor, error) {
	if a == nil || a.device == nil || a.layerNormKernel == nil {
		return nil, fmt.Errorf("cuda activation accelerator is not initialized")
	}
	rows, cols, err := rowWiseShape(gradOut, normalized, pre)
	if err != nil {
		return nil, err
	}
	return a.runLayerNormBackwardRows(gradOut, rows, cols, nil, nil, normalized, pre)
}

func (a *activationAccelerator) RunLayerNormBackwardRowsWithBoundInputs(gradOut *backend.Tensor, normalizedName, preName string) (*backend.Tensor, error) {
	if a == nil || a.device == nil || a.layerNormKernel == nil {
		return nil, fmt.Errorf("cuda activation accelerator is not initialized")
	}
	normResident, ok := a.bound[normalizedName]
	if !ok {
		return nil, fmt.Errorf("cuda activation binding %q is not resident", normalizedName)
	}
	preResident, ok := a.bound[preName]
	if !ok {
		return nil, fmt.Errorf("cuda activation binding %q is not resident", preName)
	}
	if normResident.rows != preResident.rows || normResident.cols != preResident.cols {
		return nil, fmt.Errorf("cuda layernorm backward resident bindings %q and %q shape mismatch", normalizedName, preName)
	}
	if gradOut == nil || gradOut.Rank() != 2 || gradOut.Shape[0] != normResident.rows || gradOut.Shape[1] != normResident.cols {
		return nil, fmt.Errorf("cuda layernorm backward requires grad shape [%d %d] to match resident bindings", normResident.rows, normResident.cols)
	}
	return a.runLayerNormBackwardRows(gradOut, normResident.rows, normResident.cols, &normResident, &preResident, nil, nil)
}

func (a *activationAccelerator) runGELUBackwardMul(gradOut *backend.Tensor, rows, cols int, preResident *residentActivationTensor, preAct *backend.Tensor) (*backend.Tensor, error) {
	start := time.Now()
	gradBuf, err := a.device.uploadFloat32(gradOut.F32)
	if err != nil {
		return nil, err
	}
	defer a.device.freeBuffer(gradBuf)
	a.stats.UploadedBytes += int64(len(gradOut.F32) * 4)
	preBuf := C.CUdeviceptr(0)
	if preResident != nil {
		preBuf = preResident.ptr
	} else {
		preBuf, err = a.device.uploadFloat32(preAct.F32)
		if err != nil {
			return nil, err
		}
		defer a.device.freeBuffer(preBuf)
		a.stats.UploadedBytes += int64(len(preAct.F32) * 4)
	}
	outHost := make([]float32, len(gradOut.F32))
	outBuf, err := a.device.allocFloat32(len(outHost))
	if err != nil {
		return nil, err
	}
	defer a.device.freeBuffer(outBuf)
	block := uint(128)
	grid := uint((rows*cols + int(block) - 1) / int(block))
	if err := a.device.launchAuxElementWise(a.geluKernel, grid, block, gradBuf, preBuf, outBuf, rows*cols); err != nil {
		return nil, err
	}
	if err := a.device.downloadFloat32(outHost, outBuf); err != nil {
		return nil, err
	}
	a.stats.GELUBackwardCalls++
	a.stats.DownloadedBytes += int64(len(outHost) * 4)
	a.stats.RunNanos += time.Since(start).Nanoseconds()
	return backend.NewTensorF32(append([]int(nil), gradOut.Shape...), outHost), nil
}

func (a *activationAccelerator) runSoftmaxBackwardRows(gradOut *backend.Tensor, rows, cols int, probsResident *residentActivationTensor, probs *backend.Tensor) (*backend.Tensor, error) {
	start := time.Now()
	gradBuf, err := a.device.uploadFloat32(gradOut.F32)
	if err != nil {
		return nil, err
	}
	defer a.device.freeBuffer(gradBuf)
	a.stats.UploadedBytes += int64(len(gradOut.F32) * 4)
	probsBuf := C.CUdeviceptr(0)
	if probsResident != nil {
		probsBuf = probsResident.ptr
	} else {
		probsBuf, err = a.device.uploadFloat32(probs.F32)
		if err != nil {
			return nil, err
		}
		defer a.device.freeBuffer(probsBuf)
		a.stats.UploadedBytes += int64(len(probs.F32) * 4)
	}
	outHost := make([]float32, len(gradOut.F32))
	outBuf, err := a.device.allocFloat32(len(outHost))
	if err != nil {
		return nil, err
	}
	defer a.device.freeBuffer(outBuf)
	block := uint(128)
	grid := uint((rows + int(block) - 1) / int(block))
	if err := a.device.launchAuxSoftmaxBackwardRows(a.softmaxKernel, grid, block, gradBuf, probsBuf, outBuf, rows, cols); err != nil {
		return nil, err
	}
	if err := a.device.downloadFloat32(outHost, outBuf); err != nil {
		return nil, err
	}
	a.stats.SoftmaxBackwardCalls++
	a.stats.DownloadedBytes += int64(len(outHost) * 4)
	a.stats.RunNanos += time.Since(start).Nanoseconds()
	return backend.NewTensorF32(append([]int(nil), gradOut.Shape...), outHost), nil
}

func (a *activationAccelerator) runLayerNormBackwardRows(gradOut *backend.Tensor, rows, cols int, normalizedResident, preResident *residentActivationTensor, normalized, pre *backend.Tensor) (*backend.Tensor, error) {
	start := time.Now()
	gradBuf, err := a.device.uploadFloat32(gradOut.F32)
	if err != nil {
		return nil, err
	}
	defer a.device.freeBuffer(gradBuf)
	a.stats.UploadedBytes += int64(len(gradOut.F32) * 4)
	normBuf := C.CUdeviceptr(0)
	if normalizedResident != nil {
		normBuf = normalizedResident.ptr
	} else {
		normBuf, err = a.device.uploadFloat32(normalized.F32)
		if err != nil {
			return nil, err
		}
		defer a.device.freeBuffer(normBuf)
		a.stats.UploadedBytes += int64(len(normalized.F32) * 4)
	}
	preBuf := C.CUdeviceptr(0)
	if preResident != nil {
		preBuf = preResident.ptr
	} else {
		preBuf, err = a.device.uploadFloat32(pre.F32)
		if err != nil {
			return nil, err
		}
		defer a.device.freeBuffer(preBuf)
		a.stats.UploadedBytes += int64(len(pre.F32) * 4)
	}
	outHost := make([]float32, len(gradOut.F32))
	outBuf, err := a.device.allocFloat32(len(outHost))
	if err != nil {
		return nil, err
	}
	defer a.device.freeBuffer(outBuf)
	block := uint(128)
	grid := uint((rows + int(block) - 1) / int(block))
	if err := a.device.launchAuxLayerNormBackwardRows(a.layerNormKernel, grid, block, gradBuf, normBuf, preBuf, outBuf, rows, cols); err != nil {
		return nil, err
	}
	if err := a.device.downloadFloat32(outHost, outBuf); err != nil {
		return nil, err
	}
	a.stats.LayerNormBackwardCalls++
	a.stats.DownloadedBytes += int64(len(outHost) * 4)
	a.stats.RunNanos += time.Since(start).Nanoseconds()
	return backend.NewTensorF32(append([]int(nil), gradOut.Shape...), outHost), nil
}

func rowWiseShape(tensors ...*backend.Tensor) (rows, cols int, err error) {
	if len(tensors) == 0 {
		return 0, 0, fmt.Errorf("row-wise activation requires at least one tensor")
	}
	ref := tensors[0]
	if ref == nil || ref.Rank() != 2 {
		return 0, 0, fmt.Errorf("row-wise activation requires rank-2 tensors")
	}
	rows, cols = ref.Shape[0], ref.Shape[1]
	for _, tensor := range tensors[1:] {
		if tensor == nil || !tensor.EqualShape(ref) {
			return 0, 0, fmt.Errorf("row-wise activation requires matching shapes")
		}
	}
	return rows, cols, nil
}

func (a *activationAccelerator) Close() {
	if a == nil || a.device == nil {
		return
	}
	for name, resident := range a.bound {
		_ = a.device.freeBuffer(resident.ptr)
		delete(a.bound, name)
	}
	a.stats.BoundTensors = 0
	a.device.destroyAuxKernel(a.geluKernel)
	a.device.destroyAuxKernel(a.softmaxKernel)
	a.device.destroyAuxKernel(a.layerNormKernel)
	a.geluKernel = nil
	a.softmaxKernel = nil
	a.layerNormKernel = nil
	a.bound = nil
	a.device.close()
	a.device = nil
}
