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
	"time"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

const contrastiveScoresKernelSource = `
extern "C" __global__ void manta_contrastive_scores(
    const float* query,
    const float* candidates,
    const float* query_norms,
    const float* candidate_norms,
    float* scores,
    int query_rows,
    int candidate_rows,
    int width
) {
    long long idx = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    long long total = (long long)query_rows * (long long)candidate_rows;
    if (idx >= total) {
        return;
    }
    int row = (int)(idx / candidate_rows);
    int col = (int)(idx - (long long)row * candidate_rows);
    float qn = query_norms[row];
    float pn = candidate_norms[col];
    if (qn == 0.0f || pn == 0.0f) {
        scores[idx] = 0.0f;
        return;
    }
    const float* q = query + row * width;
    const float* p = candidates + col * width;
    float dot = 0.0f;
    for (int k = 0; k < width; ++k) {
        dot += q[k] * p[k];
    }
    scores[idx] = dot / (qn * pn);
}
`

const infoNCEScalesKernelSource = `
extern "C" __global__ void manta_infonce_scales(
    const float* scores,
    const float* target_indexes,
    float* scales,
    float* row_loss,
    float* row_score,
    int query_rows,
    int candidate_rows,
    float temperature
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= query_rows) {
        return;
    }
    int base = row * candidate_rows;
    int target_index = (int)target_indexes[row];
    if (target_index < 0 || target_index >= candidate_rows) {
        target_index = 0;
    }
    float temp = temperature > 0.0f ? temperature : 0.05f;
    float max_logit = scores[base] / temp;
    float score_sum = 0.0f;
    for (int col = 0; col < candidate_rows; ++col) {
        float score = scores[base + col];
        score_sum += score;
        float logit = score / temp;
        if (logit > max_logit) {
            max_logit = logit;
        }
    }
    float denom = 0.0f;
    for (int col = 0; col < candidate_rows; ++col) {
        denom += expf(scores[base + col] / temp - max_logit);
    }
    float target_prob = 0.0f;
    for (int col = 0; col < candidate_rows; ++col) {
        float prob = 0.0f;
        if (denom != 0.0f) {
            prob = expf(scores[base + col] / temp - max_logit) / denom;
        }
        if (col == target_index) {
            target_prob = prob;
            scales[base + col] = (prob - 1.0f) / temp;
        } else {
            scales[base + col] = prob / temp;
        }
    }
    if (target_prob < 1e-12f) {
        target_prob = 1e-12f;
    }
    row_loss[row] = -logf(target_prob);
    row_score[row] = score_sum;
}
`

const contrastiveGradKernelSource = `
extern "C" __global__ void manta_contrastive_grad(
    const float* query,
    const float* candidates,
    const float* query_norms,
    const float* candidate_norms,
    const float* scores,
    const float* scales,
    float* query_grads,
    float* candidate_grads,
    int query_rows,
    int candidate_rows,
    int width
) {
    long long idx = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    long long total = (long long)query_rows * (long long)candidate_rows * (long long)width;
    if (idx >= total) {
        return;
    }
    int k = (int)(idx % width);
    long long pair = idx / width;
    int row = (int)(pair / candidate_rows);
    int col = (int)(pair - (long long)row * candidate_rows);
    float qn = query_norms[row];
    float pn = candidate_norms[col];
    if (qn == 0.0f || pn == 0.0f) {
        return;
    }
    int qBase = row * width;
    int pBase = col * width;
    float score = scores[pair];
    float scale = scales[pair];
    float denom = qn * pn;
    float qScale = score / (qn * qn);
    float pScale = score / (pn * pn);
    float qGrad = scale * (candidates[pBase + k] / denom - query[qBase + k] * qScale);
    float pGrad = scale * (query[qBase + k] / denom - candidates[pBase + k] * pScale);
    atomicAdd(&query_grads[qBase + k], qGrad);
    atomicAdd(&candidate_grads[pBase + k], pGrad);
}
`

// contrastiveNormalizeRowsKernelSource, contrastiveAxisReduceKernelSource,
// and contrastiveGradCorrectionKernelSource (S1a) back runInfoNCEGEMM, the
// cuBLAS GEMM-based InfoNCE path. They replace contrastiveScoresKernelSource
// and contrastiveGradKernelSource's O(B*C) scalar-dot-product scoring and
// O(B*C*d) atomicAdd gradient accumulation with cuBLAS matmuls plus small
// row-wise helper kernels; see the algebra comment above runInfoNCEGEMM.

const contrastiveNormalizeRowsKernelSource = `
extern "C" __global__ void manta_contrastive_normalize_rows(
    const float* data,
    float* normalized,
    float* norms,
    int rows,
    int width
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    int base = row * width;
    float sumSq = 0.0f;
    for (int k = 0; k < width; ++k) {
        float value = data[base + k];
        sumSq += value * value;
    }
    float norm = sqrtf(sumSq);
    norms[row] = norm;
    if (norm == 0.0f) {
        // Matches the zero-norm guard in contrastiveScoresKernelSource /
        // contrastiveGradKernelSource: a zeroed row makes every score and
        // gradient touching it come out to 0 without a branch downstream.
        for (int k = 0; k < width; ++k) {
            normalized[base + k] = 0.0f;
        }
        return;
    }
    float inv = 1.0f / norm;
    for (int k = 0; k < width; ++k) {
        normalized[base + k] = data[base + k] * inv;
    }
}
`

const contrastiveAxisReduceKernelSource = `
extern "C" __global__ void manta_contrastive_axis_reduce(
    const float* scores,
    const float* scales,
    float* out0,
    int outerCount,
    int innerCount,
    int outerStride,
    int innerStride
) {
    int outer = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer >= outerCount) {
        return;
    }
    long long base = (long long)outer * (long long)outerStride;
    float sum = 0.0f;
    for (int inner = 0; inner < innerCount; ++inner) {
        long long idx = base + (long long)inner * (long long)innerStride;
        sum += scores[idx] * scales[idx];
    }
    out0[outer] = sum;
}
`

const contrastiveGradCorrectionKernelSource = `
extern "C" __global__ void manta_contrastive_grad_correction(
    float* grad,
    const float* hat,
    const float* axisSum,
    const float* norms,
    int rows,
    int width
) {
    long long idx = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    long long total = (long long)rows * (long long)width;
    if (idx >= total) {
        return;
    }
    int row = (int)(idx / width);
    float norm = norms[row];
    if (norm == 0.0f) {
        grad[idx] = 0.0f;
        return;
    }
    grad[idx] = (grad[idx] - axisSum[row] * hat[idx]) / norm;
}
`

type contrastiveAccelerator struct {
	device *deviceRuntime
	// scoreKernel, gradKernel back runInfoNCEAtomic, the retired
	// one-thread-per-pair scoring kernel and atomicAdd gradient kernel. They
	// stay compiled and callable so EOS_CUDA_CONTRASTIVE_GEMM=0 is a real
	// rollback path (S1a), not dead code.
	scoreKernel *auxKernel
	gradKernel  *auxKernel
	// scaleKernel backs manta_infonce_scales, the O(B*C) row-wise
	// softmax/scales kernel. It is shared by both runInfoNCEAtomic and
	// runInfoNCEGEMM unchanged (S1a: "keep the row-wise softmax/scales
	// kernel -- it is O(B*C) and fine").
	scaleKernel *auxKernel
	// normalizeKernel, axisReduceKernel, gradCorrectionKernel back
	// runInfoNCEGEMM (S1a).
	normalizeKernel      *auxKernel
	axisReduceKernel     *auxKernel
	gradCorrectionKernel *auxKernel
	// useGEMM selects runInfoNCEGEMM over runInfoNCEAtomic; see
	// cudaContrastiveGEMMEnabled. Cached at construction so a single
	// accelerator instance is consistent for its whole lifetime.
	useGEMM bool
	stats   backend.ContrastiveAcceleratorStats
}

// cudaContrastiveGEMMEnabled reports whether a newly constructed accelerator
// routes InfoNCE through runInfoNCEGEMM (S1a's cuBLAS GEMM path) instead of
// runInfoNCEAtomic (the retired one-thread-per-pair scoring kernel plus
// atomicAdd gradient kernel). It defaults to enabled:
// TestCUDAContrastiveGEMMMatchesHostAcrossShapes holds the GEMM path to the
// host reference within tolerance across the tested shapes. Set
// EOS_CUDA_CONTRASTIVE_GEMM=0 (or false/no/off) to force the retired path,
// for example to bisect a regression.
func cudaContrastiveGEMMEnabled() bool {
	switch cudaEnv("EOS_CUDA_CONTRASTIVE_GEMM") {
	case "0", "false", "FALSE", "no", "NO", "off", "OFF":
		return false
	default:
		return true
	}
}

func init() {
	backend.RegisterContrastiveAccelerator(eosartifact.BackendCUDA, NewContrastiveAccelerator)
}

func NewContrastiveAccelerator() (backend.ContrastiveAccelerator, error) {
	device, err := newDeviceRuntime()
	if err != nil {
		return nil, err
	}
	if device == nil {
		return nil, nil
	}
	scoreKernel, err := device.compileAuxKernel(contrastiveScoresKernelSource, "manta_contrastive_scores")
	if err != nil {
		device.close()
		return nil, err
	}
	scaleKernel, err := device.compileAuxKernel(infoNCEScalesKernelSource, "manta_infonce_scales")
	if err != nil {
		device.destroyAuxKernel(scoreKernel)
		device.close()
		return nil, err
	}
	gradKernel, err := device.compileAuxKernel(contrastiveGradKernelSource, "manta_contrastive_grad")
	if err != nil {
		device.destroyAuxKernel(scoreKernel)
		device.destroyAuxKernel(scaleKernel)
		device.close()
		return nil, err
	}
	normalizeKernel, err := device.compileAuxKernel(contrastiveNormalizeRowsKernelSource, "manta_contrastive_normalize_rows")
	if err != nil {
		device.destroyAuxKernel(scoreKernel)
		device.destroyAuxKernel(scaleKernel)
		device.destroyAuxKernel(gradKernel)
		device.close()
		return nil, err
	}
	axisReduceKernel, err := device.compileAuxKernel(contrastiveAxisReduceKernelSource, "manta_contrastive_axis_reduce")
	if err != nil {
		device.destroyAuxKernel(scoreKernel)
		device.destroyAuxKernel(scaleKernel)
		device.destroyAuxKernel(gradKernel)
		device.destroyAuxKernel(normalizeKernel)
		device.close()
		return nil, err
	}
	gradCorrectionKernel, err := device.compileAuxKernel(contrastiveGradCorrectionKernelSource, "manta_contrastive_grad_correction")
	if err != nil {
		device.destroyAuxKernel(scoreKernel)
		device.destroyAuxKernel(scaleKernel)
		device.destroyAuxKernel(gradKernel)
		device.destroyAuxKernel(normalizeKernel)
		device.destroyAuxKernel(axisReduceKernel)
		device.close()
		return nil, err
	}
	return &contrastiveAccelerator{
		device:               device,
		scoreKernel:          scoreKernel,
		scaleKernel:          scaleKernel,
		gradKernel:           gradKernel,
		normalizeKernel:      normalizeKernel,
		axisReduceKernel:     axisReduceKernel,
		gradCorrectionKernel: gradCorrectionKernel,
		useGEMM:              cudaContrastiveGEMMEnabled(),
	}, nil
}

func (a *contrastiveAccelerator) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (a *contrastiveAccelerator) RunInfoNCE(query, positive *backend.Tensor, cfg backend.ContrastiveLossConfig) (backend.ContrastiveGradResult, error) {
	if a == nil || a.device == nil || a.scoreKernel == nil || a.scaleKernel == nil || a.gradKernel == nil {
		return backend.ContrastiveGradResult{}, fmt.Errorf("cuda contrastive accelerator is not initialized")
	}
	rows, width, err := contrastiveShape(query, positive)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	if rows < 2 {
		return backend.ContrastiveGradResult{}, fmt.Errorf("cuda infonce requires at least two rows")
	}
	targetIndexes := make([]int, rows)
	for i := range targetIndexes {
		targetIndexes[i] = i
	}
	return a.runInfoNCE(query, positive, targetIndexes, cfg, rows, rows, width)
}

func (a *contrastiveAccelerator) RunInfoNCEWithTargets(query, candidates *backend.Tensor, targetIndexes []int, cfg backend.ContrastiveLossConfig) (backend.ContrastiveGradResult, error) {
	if a == nil || a.device == nil || a.scoreKernel == nil || a.scaleKernel == nil || a.gradKernel == nil {
		return backend.ContrastiveGradResult{}, fmt.Errorf("cuda contrastive accelerator is not initialized")
	}
	queryRows, candidateRows, width, err := contrastiveRectShape(query, candidates)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	if queryRows < 1 {
		return backend.ContrastiveGradResult{}, fmt.Errorf("cuda infonce requires at least one query row")
	}
	if candidateRows < 2 {
		return backend.ContrastiveGradResult{}, fmt.Errorf("cuda infonce requires at least two candidate rows")
	}
	if len(targetIndexes) != queryRows {
		return backend.ContrastiveGradResult{}, fmt.Errorf("cuda infonce target indexes length %d does not match query rows %d", len(targetIndexes), queryRows)
	}
	for row, target := range targetIndexes {
		if target < 0 || target >= candidateRows {
			return backend.ContrastiveGradResult{}, fmt.Errorf("cuda infonce target index %d for row %d is outside %d candidates", target, row, candidateRows)
		}
	}
	return a.runInfoNCE(query, candidates, targetIndexes, cfg, queryRows, candidateRows, width)
}

// runInfoNCE dispatches to the GEMM-based path (S1a) or the retired
// one-thread-per-pair/atomicAdd path, per a.useGEMM (cudaContrastiveGEMMEnabled).
func (a *contrastiveAccelerator) runInfoNCE(query, candidates *backend.Tensor, targetIndexes []int, cfg backend.ContrastiveLossConfig, queryRows, candidateRows, width int) (backend.ContrastiveGradResult, error) {
	if a.useGEMM {
		return a.runInfoNCEGEMM(query, candidates, targetIndexes, cfg, queryRows, candidateRows, width)
	}
	return a.runInfoNCEAtomic(query, candidates, targetIndexes, cfg, queryRows, candidateRows, width)
}

// runInfoNCEAtomic is the retired InfoNCE path: a one-thread-per-pair scoring
// kernel (contrastiveScoresKernelSource) and an atomicAdd gradient kernel
// (contrastiveGradKernelSource). Kept byte-for-byte as the S1a rollback path;
// see cudaContrastiveGEMMEnabled / EOS_CUDA_CONTRASTIVE_GEMM.
func (a *contrastiveAccelerator) runInfoNCEAtomic(query, candidates *backend.Tensor, targetIndexes []int, cfg backend.ContrastiveLossConfig, queryRows, candidateRows, width int) (backend.ContrastiveGradResult, error) {
	temperature := cfg.Temperature
	if temperature <= 0 {
		temperature = 0.05
	}
	start := time.Now()
	queryNorms := tensorRowNorms(query.F32, queryRows, width)
	candidateNorms := tensorRowNorms(candidates.F32, candidateRows, width)
	queryBuf, err := a.device.uploadFloat32(query.F32)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(queryBuf)
	candidateBuf, err := a.device.uploadFloat32(candidates.F32)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(candidateBuf)
	queryNormBuf, err := a.device.uploadFloat32(queryNorms)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(queryNormBuf)
	candidateNormBuf, err := a.device.uploadFloat32(candidateNorms)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(candidateNormBuf)
	targetData := make([]float32, len(targetIndexes))
	for i, target := range targetIndexes {
		targetData[i] = float32(target)
	}
	targetBuf, err := a.device.uploadFloat32(targetData)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(targetBuf)
	a.stats.UploadedBytes += int64((len(query.F32) + len(candidates.F32) + len(queryNorms) + len(candidateNorms) + len(targetData)) * 4)

	pairCount := queryRows * candidateRows
	scoresBuf, err := a.device.allocFloat32(pairCount)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(scoresBuf)
	scalesBuf, err := a.device.allocFloat32(pairCount)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(scalesBuf)
	rowLossBuf, err := a.device.allocFloat32(queryRows)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(rowLossBuf)
	rowScoreBuf, err := a.device.allocFloat32(queryRows)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(rowScoreBuf)

	queryZeroGrads := make([]float32, queryRows*width)
	queryGradBuf, err := a.device.uploadFloat32(queryZeroGrads)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(queryGradBuf)
	candidateZeroGrads := make([]float32, candidateRows*width)
	candidateGradBuf, err := a.device.uploadFloat32(candidateZeroGrads)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(candidateGradBuf)
	a.stats.UploadedBytes += int64((len(queryZeroGrads) + len(candidateZeroGrads)) * 4)

	block := uint(128)
	scoreGrid := uint((pairCount + int(block) - 1) / int(block))
	if err := a.device.launchAuxContrastiveScores(a.scoreKernel, scoreGrid, block, queryBuf, candidateBuf, queryNormBuf, candidateNormBuf, scoresBuf, queryRows, candidateRows, width); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	rowGrid := uint((queryRows + int(block) - 1) / int(block))
	if err := a.device.launchAuxInfoNCEScales(a.scaleKernel, rowGrid, block, scoresBuf, targetBuf, scalesBuf, rowLossBuf, rowScoreBuf, queryRows, candidateRows, temperature); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	gradElements := pairCount * width
	gradGrid := uint((gradElements + int(block) - 1) / int(block))
	if err := a.device.launchAuxContrastiveGrad(a.gradKernel, gradGrid, block, queryBuf, candidateBuf, queryNormBuf, candidateNormBuf, scoresBuf, scalesBuf, queryGradBuf, candidateGradBuf, queryRows, candidateRows, width); err != nil {
		return backend.ContrastiveGradResult{}, err
	}

	queryGrads := make([]float32, queryRows*width)
	candidateGrads := make([]float32, candidateRows*width)
	rowLoss := make([]float32, queryRows)
	rowScore := make([]float32, queryRows)
	if err := a.device.downloadFloat32(queryGrads, queryGradBuf); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	if err := a.device.downloadFloat32(candidateGrads, candidateGradBuf); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	if err := a.device.downloadFloat32(rowLoss, rowLossBuf); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	if err := a.device.downloadFloat32(rowScore, rowScoreBuf); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	a.stats.DownloadedBytes += int64((len(queryGrads) + len(candidateGrads) + len(rowLoss) + len(rowScore)) * 4)
	a.stats.RunCalls++
	a.stats.RunNanos += time.Since(start).Nanoseconds()
	return backend.ContrastiveGradResult{
		QueryGrads:    backend.NewTensorF32([]int{queryRows, width}, queryGrads),
		PositiveGrads: backend.NewTensorF32([]int{candidateRows, width}, candidateGrads),
		LossSum:       sumFloat32(rowLoss),
		ScoreSum:      sumFloat32(rowScore),
	}, nil
}

// runInfoNCEGEMM is the S1a InfoNCE path: cuBLAS GEMMs plus small row-wise
// helper kernels instead of the O(B*C) scalar-dot scoring kernel and the
// O(B*C*d) atomicAdd gradient kernel that runInfoNCEAtomic still uses.
//
// Notation (row-major, one row per example; Cn = candidateRows):
//
//	Q, C        raw query/candidate embeddings, shape [B,d] / [Cn,d]
//	qn, cn      per-row L2 norms of Q, C -- computed on device by
//	            manta_contrastive_normalize_rows, replacing the host
//	            tensorRowNorms precompute runInfoNCEAtomic still does
//	qhat, chat  row-normalized Q, C: qhat[i] = Q[i]/qn[i], chat[j] = C[j]/cn[j]
//	S           score matrix, S[i,j] = dot(qhat[i], chat[j])
//	scale       the InfoNCE softmax gradient w.r.t. the logit,
//	            scale[i,j] = (prob[i,j] - onehot(j==target_i)) / temperature
//	            -- unchanged, still produced by manta_infonce_scales
//
// Score: dot(qhat[i], chat[j]) = dot(Q[i],C[j])/(qn[i]*cn[j]) is exactly the
// cosine similarity contrastiveScoresKernelSource computes, so normalizing
// first and taking a plain dot product (S = Qhat @ Chat^T, one cuBLAS call)
// reproduces the same score without ever forming a Q[i]/cn[j]-style cross
// term. A zero-norm row becomes an all-zero qhat/chat row
// (manta_contrastive_normalize_rows), which zeroes every score touching it
// through the dot product alone -- the same outcome as
// contrastiveScoresKernelSource's explicit qn==0||pn==0 guard, with no
// per-pair branch needed.
//
// Gradient: contrastiveGradKernelSource computes, per element k of pair (i,j):
//
//	qGrad[i,k] += scale[i,j] * ( C[j,k]/(qn[i]*cn[j]) - Q[i,k]*S[i,j]/qn[i]^2 )
//	cGrad[j,k] += scale[i,j] * ( Q[i,k]/(qn[i]*cn[j]) - C[j,k]*S[i,j]/cn[j]^2 )
//
// Substituting C[j,k] = chat[j,k]*cn[j] and Q[i,k] = qhat[i,k]*qn[i] cancels
// the (qn,cn) cross terms:
//
//	C[j,k]/(qn[i]*cn[j])   = chat[j,k]/qn[i]
//	Q[i,k]*S[i,j]/qn[i]^2  = qhat[i,k]*S[i,j]/qn[i]
//	=> qGrad[i,k] += (scale[i,j]/qn[i]) * ( chat[j,k] - S[i,j]*qhat[i,k] )
//
// and symmetrically:
//
//	=> cGrad[j,k] += (scale[i,j]/cn[j]) * ( qhat[i,k] - S[i,j]*chat[j,k] )
//
// Summing over the pair index that is not k's own row turns each sum into a
// GEMM plus a rank-1 correction that only depends on the row itself:
//
//	qGrad[i,:] = (1/qn[i]) * ( sum_j scale[i,j]*chat[j,:] - qhat[i,:] * sum_j scale[i,j]*S[i,j] )
//	           = (1/qn[i]) * ( (Scale  @ Chat)[i,:]       - qhat[i,:] * rowSum[i] )
//	cGrad[j,:] = (1/cn[j]) * ( sum_i scale[i,j]*qhat[i,:] - chat[j,:] * sum_i scale[i,j]*S[i,j] )
//	           = (1/cn[j]) * ( (Scale^T@ Qhat)[j,:]       - chat[j,:] * colSum[j] )
//
// where rowSum[i] = sum_j scale[i,j]*S[i,j] and colSum[j] = sum_i scale[i,j]*S[i,j]
// are the same elementwise product (Scale .* S) reduced along each axis
// (manta_contrastive_axis_reduce, called once per axis: row axis for
// rowSum, candidate axis for colSum). qGrad[i,:]/cGrad[j,:] are zero when
// qn[i]/cn[j] is zero, matching contrastiveGradKernelSource's zero-norm
// guard; manta_contrastive_grad_correction folds that guard in alongside the
// division.
//
// So: two GEMMs (Scale@Chat for dQhat, Scale^T@Qhat for dChat) produce the
// sum term in cuBLAS instead of one thread per scalar output with two
// atomicAdd calls, and manta_contrastive_grad_correction applies the rank-1
// correction and the final division by qn/cn in one O(B*d + Cn*d) pass. This
// replaces the retired O(B*Cn*d) atomicAdd kernel with two O(B*Cn*d) GEMMs
// (cuBLAS tiles and parallelizes this far better than a scalar kernel) plus
// O(B*d + Cn*d + B*Cn) helper-kernel work, and it never needs 64-bit pair*d
// indexing (contrastiveGradKernelSource's overflow guard) because no kernel
// here ever iterates the full B*Cn*d space.
func (a *contrastiveAccelerator) runInfoNCEGEMM(query, candidates *backend.Tensor, targetIndexes []int, cfg backend.ContrastiveLossConfig, queryRows, candidateRows, width int) (backend.ContrastiveGradResult, error) {
	temperature := cfg.Temperature
	if temperature <= 0 {
		temperature = 0.05
	}
	start := time.Now()

	queryBuf, err := a.device.uploadFloat32(query.F32)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(queryBuf)
	candidateBuf, err := a.device.uploadFloat32(candidates.F32)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(candidateBuf)
	targetData := make([]float32, len(targetIndexes))
	for i, target := range targetIndexes {
		targetData[i] = float32(target)
	}
	targetBuf, err := a.device.uploadFloat32(targetData)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(targetBuf)
	a.stats.UploadedBytes += int64((len(query.F32) + len(candidates.F32) + len(targetData)) * 4)

	// qhat/chat/qn/cn are computed on device; no host tensorRowNorms call.
	qHatBuf, err := a.device.allocFloat32(queryRows * width)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(qHatBuf)
	qNormBuf, err := a.device.allocFloat32(queryRows)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(qNormBuf)
	cHatBuf, err := a.device.allocFloat32(candidateRows * width)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(cHatBuf)
	cNormBuf, err := a.device.allocFloat32(candidateRows)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(cNormBuf)

	block := uint(128)
	qRowGrid := uint((queryRows + int(block) - 1) / int(block))
	if err := a.device.launchAuxContrastiveNormalizeRows(a.normalizeKernel, qRowGrid, block, queryBuf, qHatBuf, qNormBuf, queryRows, width); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	cRowGrid := uint((candidateRows + int(block) - 1) / int(block))
	if err := a.device.launchAuxContrastiveNormalizeRows(a.normalizeKernel, cRowGrid, block, candidateBuf, cHatBuf, cNormBuf, candidateRows, width); err != nil {
		return backend.ContrastiveGradResult{}, err
	}

	// S = Qhat @ Chat^T via the shared cuBLAS matmul path (matmul_accel.go /
	// native_linux.go), CUBLAS_PEDANTIC_MATH + CUBLAS_ATOMICS_NOT_ALLOWED
	// deterministic (set once at runtime creation; see compact_train_accel_linux.go).
	pairCount := queryRows * candidateRows
	scoresBuf, err := a.device.allocFloat32(pairCount)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(scoresBuf)
	if err := a.device.matMulCublasWithBetaNoSyncInts(qHatBuf, cHatBuf, scoresBuf, queryRows, width, candidateRows, width, false, true, 0); err != nil {
		return backend.ContrastiveGradResult{}, err
	}

	// manta_infonce_scales is unchanged (S1a keeps this kernel as-is).
	scalesBuf, err := a.device.allocFloat32(pairCount)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(scalesBuf)
	rowLossBuf, err := a.device.allocFloat32(queryRows)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(rowLossBuf)
	rowScoreBuf, err := a.device.allocFloat32(queryRows)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(rowScoreBuf)
	rowGrid := uint((queryRows + int(block) - 1) / int(block))
	if err := a.device.launchAuxInfoNCEScales(a.scaleKernel, rowGrid, block, scoresBuf, targetBuf, scalesBuf, rowLossBuf, rowScoreBuf, queryRows, candidateRows, temperature); err != nil {
		return backend.ContrastiveGradResult{}, err
	}

	// rowSum/colSum: the same (Scale .* S) elementwise product reduced along
	// each axis -- see the algebra above.
	rowSumBuf, err := a.device.allocFloat32(queryRows)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(rowSumBuf)
	colSumBuf, err := a.device.allocFloat32(candidateRows)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(colSumBuf)
	if err := a.device.launchAuxContrastiveAxisReduce(a.axisReduceKernel, rowGrid, block, scoresBuf, scalesBuf, rowSumBuf, queryRows, candidateRows, candidateRows, 1); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	colGrid := uint((candidateRows + int(block) - 1) / int(block))
	if err := a.device.launchAuxContrastiveAxisReduce(a.axisReduceKernel, colGrid, block, scoresBuf, scalesBuf, colSumBuf, candidateRows, queryRows, 1, candidateRows); err != nil {
		return backend.ContrastiveGradResult{}, err
	}

	// dQhat = Scale @ Chat, dChat = Scale^T @ Qhat. Both GEMMs run with
	// beta=0, so cuBLAS never reads the destination buffer's prior content;
	// the cuMemsetD32 zero-fill below is defense in depth (S1a item 4 --
	// replaces the retired path's host zero-array-and-upload), not a
	// correctness dependency of the GEMM itself.
	queryGradBuf, err := a.device.allocFloat32(queryRows * width)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(queryGradBuf)
	if err := a.device.memsetFloat32Zero(queryGradBuf, queryRows*width); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	candidateGradBuf, err := a.device.allocFloat32(candidateRows * width)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(candidateGradBuf)
	if err := a.device.memsetFloat32Zero(candidateGradBuf, candidateRows*width); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	if err := a.device.matMulCublasWithBetaNoSyncInts(scalesBuf, cHatBuf, queryGradBuf, queryRows, candidateRows, candidateRows, width, false, false, 0); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	if err := a.device.matMulCublasWithBetaNoSyncInts(scalesBuf, qHatBuf, candidateGradBuf, queryRows, candidateRows, queryRows, width, true, false, 0); err != nil {
		return backend.ContrastiveGradResult{}, err
	}

	// Apply the rank-1 correction and the final division by qn/cn, in place.
	qGradGrid := uint((queryRows*width + int(block) - 1) / int(block))
	if err := a.device.launchAuxContrastiveGradCorrection(a.gradCorrectionKernel, qGradGrid, block, queryGradBuf, qHatBuf, rowSumBuf, qNormBuf, queryRows, width); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	cGradGrid := uint((candidateRows*width + int(block) - 1) / int(block))
	if err := a.device.launchAuxContrastiveGradCorrection(a.gradCorrectionKernel, cGradGrid, block, candidateGradBuf, cHatBuf, colSumBuf, cNormBuf, candidateRows, width); err != nil {
		return backend.ContrastiveGradResult{}, err
	}

	queryGrads := make([]float32, queryRows*width)
	candidateGrads := make([]float32, candidateRows*width)
	rowLoss := make([]float32, queryRows)
	rowScore := make([]float32, queryRows)
	if err := a.device.downloadFloat32(queryGrads, queryGradBuf); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	if err := a.device.downloadFloat32(candidateGrads, candidateGradBuf); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	if err := a.device.downloadFloat32(rowLoss, rowLossBuf); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	if err := a.device.downloadFloat32(rowScore, rowScoreBuf); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	a.stats.DownloadedBytes += int64((len(queryGrads) + len(candidateGrads) + len(rowLoss) + len(rowScore)) * 4)
	a.stats.RunCalls++
	a.stats.RunNanos += time.Since(start).Nanoseconds()
	return backend.ContrastiveGradResult{
		QueryGrads:    backend.NewTensorF32([]int{queryRows, width}, queryGrads),
		PositiveGrads: backend.NewTensorF32([]int{candidateRows, width}, candidateGrads),
		LossSum:       sumFloat32(rowLoss),
		ScoreSum:      sumFloat32(rowScore),
	}, nil
}

func (a *contrastiveAccelerator) Stats() backend.ContrastiveAcceleratorStats {
	if a == nil {
		return backend.ContrastiveAcceleratorStats{}
	}
	return a.stats
}

func (a *contrastiveAccelerator) Close() {
	if a == nil || a.device == nil {
		return
	}
	a.device.destroyAuxKernel(a.scoreKernel)
	a.device.destroyAuxKernel(a.scaleKernel)
	a.device.destroyAuxKernel(a.gradKernel)
	a.device.destroyAuxKernel(a.normalizeKernel)
	a.device.destroyAuxKernel(a.axisReduceKernel)
	a.device.destroyAuxKernel(a.gradCorrectionKernel)
	a.scoreKernel = nil
	a.scaleKernel = nil
	a.gradKernel = nil
	a.normalizeKernel = nil
	a.axisReduceKernel = nil
	a.gradCorrectionKernel = nil
	a.device.close()
	a.device = nil
}

func contrastiveShape(query, positive *backend.Tensor) (rows, width int, err error) {
	if query == nil || positive == nil {
		return 0, 0, fmt.Errorf("cuda contrastive tensors are required")
	}
	if query.Rank() != 2 || positive.Rank() != 2 {
		return 0, 0, fmt.Errorf("cuda contrastive tensors must be rank-2")
	}
	if query.Shape[0] != positive.Shape[0] || query.Shape[1] != positive.Shape[1] {
		return 0, 0, fmt.Errorf("cuda contrastive tensor shape mismatch %v vs %v", query.Shape, positive.Shape)
	}
	if len(query.F32) != len(positive.F32) || len(query.F32) != query.Shape[0]*query.Shape[1] {
		return 0, 0, fmt.Errorf("cuda contrastive tensor data does not match shape")
	}
	return query.Shape[0], query.Shape[1], nil
}

func contrastiveRectShape(query, candidates *backend.Tensor) (queryRows, candidateRows, width int, err error) {
	if query == nil || candidates == nil {
		return 0, 0, 0, fmt.Errorf("cuda contrastive tensors are required")
	}
	if query.Rank() != 2 || candidates.Rank() != 2 {
		return 0, 0, 0, fmt.Errorf("cuda contrastive tensors must be rank-2")
	}
	if query.Shape[1] != candidates.Shape[1] {
		return 0, 0, 0, fmt.Errorf("cuda contrastive tensor width mismatch %v vs %v", query.Shape, candidates.Shape)
	}
	if len(query.F32) != query.Shape[0]*query.Shape[1] || len(candidates.F32) != candidates.Shape[0]*candidates.Shape[1] {
		return 0, 0, 0, fmt.Errorf("cuda contrastive tensor data does not match shape")
	}
	return query.Shape[0], candidates.Shape[0], query.Shape[1], nil
}

func tensorRowNorms(data []float32, rows, width int) []float32 {
	out := make([]float32, rows)
	for row := 0; row < rows; row++ {
		base := row * width
		sum := float32(0)
		for col := 0; col < width; col++ {
			value := data[base+col]
			sum += value * value
		}
		out[row] = float32(math.Sqrt(float64(sum)))
	}
	return out
}

func sumFloat32(values []float32) float32 {
	sum := float32(0)
	for _, value := range values {
		sum += value
	}
	return sum
}
