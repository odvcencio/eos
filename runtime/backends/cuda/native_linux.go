//go:build linux && cgo

package cuda

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -L/usr/lib/x86_64-linux-gnu -lnvrtc -lcuda -lcublas
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <cuda.h>
#include <nvrtc.h>
#include <cublas_v2.h>

typedef struct {
	CUcontext ctx;
	CUdevice device;
	int major;
	int minor;
	int primary_ctx;
	cublasHandle_t blas;
	CUstream stream;
} EosCudaRuntime;

typedef struct {
	CUmodule module;
	CUfunction function;
} EosCudaKernel;

static char* manta_dup_cstr(const char* s) {
	if (s == NULL) {
		return NULL;
	}
	size_t n = strlen(s) + 1;
	char* out = (char*)malloc(n);
	if (out == NULL) {
		return NULL;
	}
	memcpy(out, s, n);
	return out;
}

static char* manta_dup_format(const char* prefix, const char* value) {
	if (prefix == NULL) prefix = "error";
	if (value == NULL) value = "unknown";
	size_t n = strlen(prefix) + strlen(value) + 3;
	char* out = (char*)malloc(n);
	if (out == NULL) {
		return NULL;
	}
	snprintf(out, n, "%s: %s", prefix, value);
	return out;
}

static char* manta_dup_cu_error(const char* prefix, CUresult res) {
	const char* name = NULL;
	const char* detail = NULL;
	cuGetErrorName(res, &name);
	cuGetErrorString(res, &detail);
	if (name == NULL) name = "CU_UNKNOWN";
	if (detail == NULL) detail = "unknown";
	size_t n = strlen(prefix) + strlen(name) + strlen(detail) + 6;
	char* out = (char*)malloc(n);
	if (out == NULL) {
		return NULL;
	}
	snprintf(out, n, "%s: %s (%s)", prefix, name, detail);
	return out;
}

static char* manta_dup_nvrtc_error(const char* prefix, nvrtcResult res) {
	const char* detail = nvrtcGetErrorString(res);
	if (detail == NULL) detail = "unknown";
	return manta_dup_format(prefix, detail);
}

static const char* manta_cublas_status_name(cublasStatus_t status) {
	switch (status) {
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
	case CUBLAS_STATUS_NOT_SUPPORTED:
		return "CUBLAS_STATUS_NOT_SUPPORTED";
	case CUBLAS_STATUS_LICENSE_ERROR:
		return "CUBLAS_STATUS_LICENSE_ERROR";
	default:
		return "CUBLAS_STATUS_UNKNOWN";
	}
}

static char* manta_dup_cublas_error(const char* prefix, cublasStatus_t status) {
	return manta_dup_format(prefix, manta_cublas_status_name(status));
}

static int eosCudaRuntimeCreate(EosCudaRuntime** out, char** err) {
	EosCudaRuntime* rt = NULL;
	CUdevice device = 0;
	CUcontext ctx = NULL;
	int major = 0;
	int minor = 0;
	cublasHandle_t blas = NULL;
	CUresult cuRes = cuInit(0);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuInit", cuRes);
		return 1;
	}
	cuRes = cuDeviceGet(&device, 0);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuDeviceGet", cuRes);
		return 1;
	}
	cuRes = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR)", cuRes);
		return 1;
	}
	cuRes = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuDeviceGetAttribute(COMPUTE_CAPABILITY_MINOR)", cuRes);
		return 1;
	}
	// Repeated runtime loads must not allocate independent CUDA contexts.
	// The primary context is shared by the process and avoids suite-wide VRAM exhaustion.
	cuRes = cuDevicePrimaryCtxRetain(&ctx, device);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuDevicePrimaryCtxRetain", cuRes);
		return 1;
	}
	cuRes = cuCtxSetCurrent(ctx);
	if (cuRes != CUDA_SUCCESS) {
		cuDevicePrimaryCtxRelease(device);
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cublasStatus_t blasRes = cublasCreate(&blas);
	if (blasRes != CUBLAS_STATUS_SUCCESS) {
		cuDevicePrimaryCtxRelease(device);
		*err = manta_dup_cublas_error("cublasCreate", blasRes);
		return 1;
	}
	rt = (EosCudaRuntime*)malloc(sizeof(EosCudaRuntime));
	if (rt == NULL) {
		cublasDestroy(blas);
		cuDevicePrimaryCtxRelease(device);
		*err = manta_dup_format("malloc", "failed to allocate runtime");
		return 1;
	}
	rt->ctx = ctx;
	rt->device = device;
	rt->major = major;
	rt->minor = minor;
	rt->primary_ctx = 1;
	rt->blas = blas;
	rt->stream = NULL;
	// A created (non-default) stream is required so kernel launches and cuBLAS
	// work can be captured into a CUDA graph; the legacy default stream (stream
	// 0) cannot be captured. It must be a BLOCKING stream (CU_STREAM_DEFAULT):
	// our device<->host transfers use the synchronous cuMemcpy{H2D,D2H}, which
	// run on the legacy default stream. A CU_STREAM_NON_BLOCKING work stream
	// opts out of ordering with the default stream, so those copies raced
	// in-flight kernels/GEMMs — producing non-deterministic, size-dependent
	// garbage (orthogonal embeddings for D>=128). A blocking created stream is
	// still capturable and orders correctly with the synchronous copies.
	// cublasSetStream binds GEMMs to the same stream so one sync covers both.
	cuRes = cuStreamCreate(&rt->stream, CU_STREAM_DEFAULT);
	if (cuRes != CUDA_SUCCESS) {
		cublasDestroy(blas);
		cuDevicePrimaryCtxRelease(device);
		free(rt);
		*err = manta_dup_cu_error("cuStreamCreate", cuRes);
		return 1;
	}
	blasRes = cublasSetStream(blas, rt->stream);
	if (blasRes != CUBLAS_STATUS_SUCCESS) {
		cuStreamDestroy(rt->stream);
		cublasDestroy(blas);
		cuDevicePrimaryCtxRelease(device);
		free(rt);
		*err = manta_dup_cublas_error("cublasSetStream", blasRes);
		return 1;
	}
	*out = rt;
	return 0;
}

static void eosCudaRuntimeDestroy(EosCudaRuntime* rt) {
	if (rt == NULL) {
		return;
	}
	if (rt->ctx != NULL) {
		cuCtxSetCurrent(rt->ctx);
	}
	if (rt->stream != NULL) {
		cuStreamDestroy(rt->stream);
	}
	if (rt->blas != NULL) {
		cublasDestroy(rt->blas);
	}
	if (rt->ctx != NULL) {
		if (rt->primary_ctx) {
			cuDevicePrimaryCtxRelease(rt->device);
		} else {
			cuCtxDestroy(rt->ctx);
		}
	}
	free(rt);
}

static int eosCudaCompileKernel(EosCudaRuntime* rt, const char* src, const char* entry, EosCudaKernel** out, char** log, char** err) {
	nvrtcProgram program;
	nvrtcResult nvRes;
	size_t logSize = 0;
	size_t ptxSize = 0;
	char arch[64];
	char* ptx = NULL;
	EosCudaKernel* kernel = NULL;
	CUmodule module = NULL;
	CUfunction function = NULL;

	nvRes = nvrtcCreateProgram(&program, src, entry, 0, NULL, NULL);
	if (nvRes != NVRTC_SUCCESS) {
		*err = manta_dup_nvrtc_error("nvrtcCreateProgram", nvRes);
		return 1;
	}

	snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", rt->major, rt->minor);
	const char* opts[] = {"--std=c++14", arch};
	nvRes = nvrtcCompileProgram(program, 2, opts);
	nvrtcGetProgramLogSize(program, &logSize);
	if (logSize > 1) {
		*log = (char*)malloc(logSize);
		if (*log != NULL) {
			nvrtcGetProgramLog(program, *log);
		}
	}
	if (nvRes != NVRTC_SUCCESS) {
		*err = manta_dup_nvrtc_error("nvrtcCompileProgram", nvRes);
		nvrtcDestroyProgram(&program);
		return 1;
	}

	nvRes = nvrtcGetPTXSize(program, &ptxSize);
	if (nvRes != NVRTC_SUCCESS) {
		*err = manta_dup_nvrtc_error("nvrtcGetPTXSize", nvRes);
		nvrtcDestroyProgram(&program);
		return 1;
	}
	ptx = (char*)malloc(ptxSize);
	if (ptx == NULL) {
		*err = manta_dup_format("malloc", "failed to allocate PTX buffer");
		nvrtcDestroyProgram(&program);
		return 1;
	}
	nvRes = nvrtcGetPTX(program, ptx);
	nvrtcDestroyProgram(&program);
	if (nvRes != NVRTC_SUCCESS) {
		free(ptx);
		*err = manta_dup_nvrtc_error("nvrtcGetPTX", nvRes);
		return 1;
	}

	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		free(ptx);
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuModuleLoadDataEx(&module, ptx, 0, NULL, NULL);
	free(ptx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuModuleLoadDataEx", cuRes);
		return 1;
	}
	cuRes = cuModuleGetFunction(&function, module, entry);
	if (cuRes != CUDA_SUCCESS) {
		cuModuleUnload(module);
		*err = manta_dup_cu_error("cuModuleGetFunction", cuRes);
		return 1;
	}

	kernel = (EosCudaKernel*)malloc(sizeof(EosCudaKernel));
	if (kernel == NULL) {
		cuModuleUnload(module);
		*err = manta_dup_format("malloc", "failed to allocate kernel");
		return 1;
	}
	kernel->module = module;
	kernel->function = function;
	*out = kernel;
	return 0;
}

static void eosCudaKernelDestroy(EosCudaKernel* kernel) {
	if (kernel == NULL) {
		return;
	}
	if (kernel->module != NULL) {
		cuModuleUnload(kernel->module);
	}
	free(kernel);
}

static int eosCudaMemAlloc(EosCudaRuntime* rt, CUdeviceptr* out, size_t bytes, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuMemAlloc(out, bytes);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuMemAlloc", cuRes);
		return 1;
	}
	return 0;
}

static int eosCudaMemFree(EosCudaRuntime* rt, CUdeviceptr ptr, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuMemFree(ptr);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuMemFree", cuRes);
		return 1;
	}
	return 0;
}

static int eosCudaMemcpyHtoD(EosCudaRuntime* rt, CUdeviceptr dst, const void* src, size_t bytes, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuMemcpyHtoD(dst, src, bytes);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuMemcpyHtoD", cuRes);
		return 1;
	}
	return 0;
}

static int eosCudaMemcpyDtoH(EosCudaRuntime* rt, void* dst, CUdeviceptr src, size_t bytes, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuMemcpyDtoH(dst, src, bytes);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuMemcpyDtoH", cuRes);
		return 1;
	}
	return 0;
}

static int eosCudaSynchronize(EosCudaRuntime* rt, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuStreamSynchronize(rt->stream);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuStreamSynchronize", cuRes);
		return 1;
	}
	return 0;
}

static int eosCudaReadFloat32(EosCudaRuntime* rt, CUdeviceptr src, int elementIndex, float* out, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	CUdeviceptr addr = src + ((CUdeviceptr)elementIndex * sizeof(float));
	cuRes = cuMemcpyDtoH(out, addr, sizeof(float));
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuMemcpyDtoH", cuRes);
		return 1;
	}
	return 0;
}

static int eosCudaBlasIsamax(EosCudaRuntime* rt, CUdeviceptr src, int elements, int* outIndex, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	const float* ptr = (const float*)(uintptr_t)src;
	cublasStatus_t blasRes = cublasIsamax(rt->blas, elements, ptr, 1, outIndex);
	if (blasRes != CUBLAS_STATUS_SUCCESS) {
		*err = manta_dup_cublas_error("cublasIsamax", blasRes);
		return 1;
	}
	return 0;
}

static int eosCudaLaunch1D(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, void** args, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuLaunchKernel(kernel->function, grid, 1, 1, block, 1, 1, 0, rt->stream, args, NULL);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuLaunchKernel", cuRes);
		return 1;
	}
	cuRes = cuStreamSynchronize(rt->stream);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuStreamSynchronize", cuRes);
		return 1;
	}
	return 0;
}

static int eosCudaLaunchRowWise(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr in0, CUdeviceptr out0, int rows, int cols, char** err) {
	void* args[] = {&in0, &out0, &rows, &cols};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchRoPE(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr in0, CUdeviceptr out0, int rows, int cols, int seqLen, char** err) {
	void* args[] = {&in0, &out0, &rows, &cols, &seqLen};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchElementWise(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr lhs, CUdeviceptr rhs, CUdeviceptr out0, int elements, char** err) {
	void* args[] = {&lhs, &rhs, &out0, &elements};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchUnary(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr in0, CUdeviceptr out0, int elements, char** err) {
	void* args[] = {&in0, &out0, &elements};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchScore(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr query, CUdeviceptr docs, CUdeviceptr out0, int rows, int cols, char** err) {
	void* args[] = {&query, &docs, &out0, &rows, &cols};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchOptimizerUpdate(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr param, CUdeviceptr mom1, CUdeviceptr mom2, CUdeviceptr grad, int elements, int mode, float learningRate, float weightDecay, float beta1, float beta2, float corr1, float corr2, float epsilon, float scale, char** err) {
	void* args[] = {&param, &mom1, &mom2, &grad, &elements, &mode, &learningRate, &weightDecay, &beta1, &beta2, &corr1, &corr2, &epsilon, &scale};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchSoftmaxBackwardRows(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr gradOut, CUdeviceptr probs, CUdeviceptr out0, int rows, int cols, char** err) {
	void* args[] = {&gradOut, &probs, &out0, &rows, &cols};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchSoftmaxForwardRows(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr data, int rows, int cols, char** err) {
	void* args[] = {&data, &rows, &cols};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchGeluForward(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr src, CUdeviceptr dst, int elements, char** err) {
	void* args[] = {&src, &dst, &elements};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchLayerNormForwardRows(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr src, CUdeviceptr dst, int rows, int cols, char** err) {
	void* args[] = {&src, &dst, &rows, &cols};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchResidualAdd(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr a, CUdeviceptr b, CUdeviceptr out0, int elements, char** err) {
	void* args[] = {&a, &b, &out0, &elements};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchLayerNormBackwardRows(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr gradOut, CUdeviceptr normalized, CUdeviceptr pre, CUdeviceptr out0, int rows, int cols, char** err) {
	void* args[] = {&gradOut, &normalized, &pre, &out0, &rows, &cols};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchBERTEmbeddingAffineLayerNorm(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr tokenEmb, CUdeviceptr positionEmb, CUdeviceptr tokenTypeEmb, CUdeviceptr gamma, CUdeviceptr beta, CUdeviceptr inputIDs, CUdeviceptr tokenTypeIDs, CUdeviceptr out0, CUdeviceptr status, int rows, int tokens, int hidden, int vocab, int maxPositions, int typeVocab, double epsilon, char** err) {
	void* args[] = {&tokenEmb, &positionEmb, &tokenTypeEmb, &gamma, &beta, &inputIDs, &tokenTypeIDs, &out0, &status, &rows, &tokens, &hidden, &vocab, &maxPositions, &typeVocab, &epsilon};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchBERTExactGELU(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr src, CUdeviceptr dst, int elements, char** err) {
	void* args[] = {&src, &dst, &elements};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchBERTResidualAffineLayerNorm(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr src, CUdeviceptr residual, CUdeviceptr gamma, CUdeviceptr beta, CUdeviceptr out0, int rows, int hidden, double epsilon, char** err) {
	void* args[] = {&src, &residual, &gamma, &beta, &out0, &rows, &hidden, &epsilon};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchBERTCLSL2(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr hiddenStates, CUdeviceptr out0, int batch, int tokens, int hidden, char** err) {
	void* args[] = {&hiddenStates, &out0, &batch, &tokens, &hidden};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchBERTBiasAdd(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr data, CUdeviceptr bias, int rows, int cols, char** err) {
	void* args[] = {&data, &bias, &rows, &cols};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchBERTAttentionContext(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr query, CUdeviceptr key, CUdeviceptr value, CUdeviceptr attentionMask, CUdeviceptr out0, int batch, int tokens, int hidden, int heads, int headDim, char** err) {
	void* args[] = {&query, &key, &value, &attentionMask, &out0, &batch, &tokens, &hidden, &heads, &headDim};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchQuantizeInPlace(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr data, int elements, float levels, float scale, char** err) {
	void* args[] = {&data, &elements, &levels, &scale};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchGDN(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr input, CUdeviceptr beta, CUdeviceptr gamma, CUdeviceptr out0, int elements, int channels, int height, int width, int inverse, char** err) {
	void* args[] = {&input, &beta, &gamma, &out0, &elements, &channels, &height, &width, &inverse};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchConv2D(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr input, CUdeviceptr weight, CUdeviceptr bias, CUdeviceptr out0, int elements, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth, int inPerGroup, int outPerGroup, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW, int hasBias, char** err) {
	void* args[] = {&input, &weight, &bias, &out0, &elements, &inChannels, &inHeight, &inWidth, &outChannels, &outHeight, &outWidth, &inPerGroup, &outPerGroup, &kernelH, &kernelW, &strideH, &strideW, &padH, &padW, &dilationH, &dilationW, &hasBias};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchConv2DTranspose(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr input, CUdeviceptr weight, CUdeviceptr bias, CUdeviceptr out0, int elements, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth, int inPerGroup, int outPerGroup, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW, int hasBias, char** err) {
	void* args[] = {&input, &weight, &bias, &out0, &elements, &inChannels, &inHeight, &inWidth, &outChannels, &outHeight, &outWidth, &inPerGroup, &outPerGroup, &kernelH, &kernelW, &strideH, &strideW, &padH, &padW, &dilationH, &dilationW, &hasBias};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchTurboQEncode(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr input, CUdeviceptr coords, CUdeviceptr norms, CUdeviceptr scratchWork, CUdeviceptr scratchRotated, CUdeviceptr perm, CUdeviceptr signs1, CUdeviceptr signs2, CUdeviceptr centroids, CUdeviceptr boundaries, int vectors, int channels, int height, int width, int levels, char** err) {
	void* args[] = {&input, &coords, &norms, &scratchWork, &scratchRotated, &perm, &signs1, &signs2, &centroids, &boundaries, &vectors, &channels, &height, &width, &levels};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchTurboQDecode(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr coords, CUdeviceptr norms, CUdeviceptr out0, CUdeviceptr scratchWork, CUdeviceptr scratchRotated, CUdeviceptr perm, CUdeviceptr signs1, CUdeviceptr signs2, CUdeviceptr centroids, int vectors, int channels, int height, int width, int levels, char** err) {
	void* args[] = {&coords, &norms, &out0, &scratchWork, &scratchRotated, &perm, &signs1, &signs2, &centroids, &vectors, &channels, &height, &width, &levels};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchSparseAttention(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr query, CUdeviceptr key, CUdeviceptr value, CUdeviceptr out0, int rank, int kvLayout, int batches, int queryLen, int keyLen, int queryDim, int valueDim, int topK, char** err) {
	void* args[] = {&query, &key, &value, &out0, &rank, &kvLayout, &batches, &queryLen, &keyLen, &queryDim, &valueDim, &topK};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchTurboSparseAttention(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr query, CUdeviceptr keyCoords, CUdeviceptr keyNorms, CUdeviceptr valueCoords, CUdeviceptr valueNorms, CUdeviceptr out0, CUdeviceptr keyScratchWork, CUdeviceptr keyScratchDecoded, CUdeviceptr valueScratchWork, CUdeviceptr valueScratchDecoded, CUdeviceptr keyPerm, CUdeviceptr keySigns1, CUdeviceptr keySigns2, CUdeviceptr keyCentroids, CUdeviceptr valuePerm, CUdeviceptr valueSigns1, CUdeviceptr valueSigns2, CUdeviceptr valueCentroids, int rank, int batches, int queryLen, int keyLen, int queryDim, int valueDim, int topK, int keyLevels, int valueLevels, int routeBlockSize, int routeTopBlocks, char** err) {
	void* args[] = {&query, &keyCoords, &keyNorms, &valueCoords, &valueNorms, &out0, &keyScratchWork, &keyScratchDecoded, &valueScratchWork, &valueScratchDecoded, &keyPerm, &keySigns1, &keySigns2, &keyCentroids, &valuePerm, &valueSigns1, &valueSigns2, &valueCentroids, &rank, &batches, &queryLen, &keyLen, &queryDim, &valueDim, &topK, &keyLevels, &valueLevels, &routeBlockSize, &routeTopBlocks};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchMSEPartials(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr lhs, CUdeviceptr rhs, CUdeviceptr partials, int elements, char** err) {
	void* args[] = {&lhs, &rhs, &partials, &elements};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchMSSSIMPartials(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr lhs, CUdeviceptr rhs, CUdeviceptr partials, int elements, char** err) {
	void* args[] = {&lhs, &rhs, &partials, &elements};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchScalarSum(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr values, CUdeviceptr out0, int count, char** err) {
	void* args[] = {&values, &out0, &count};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchRDLoss(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr distortion, CUdeviceptr rate, CUdeviceptr out0, float lambda, char** err) {
	void* args[] = {&distortion, &rate, &out0, &lambda};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchCrossEntropyPartials(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr codes, CUdeviceptr logits, CUdeviceptr partials, int elements, int mode, int layout, int levels, int bits, int codeRank, int codeN, int codeC, int codeH, int codeW, int logitsLen, int logitsN, int logitsC, int logitsH, int logitsW, int sigmaMode, char** err) {
	void* args[] = {&codes, &logits, &partials, &elements, &mode, &layout, &levels, &bits, &codeRank, &codeN, &codeC, &codeH, &codeW, &logitsLen, &logitsN, &logitsC, &logitsH, &logitsW, &sigmaMode};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchContrastiveScores(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr query, CUdeviceptr positive, CUdeviceptr queryNorms, CUdeviceptr positiveNorms, CUdeviceptr scores, int queryRows, int candidateRows, int width, char** err) {
	void* args[] = {&query, &positive, &queryNorms, &positiveNorms, &scores, &queryRows, &candidateRows, &width};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchInfoNCEScales(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr scores, CUdeviceptr targetIndexes, CUdeviceptr scales, CUdeviceptr rowLoss, CUdeviceptr rowScore, int queryRows, int candidateRows, float temperature, char** err) {
	void* args[] = {&scores, &targetIndexes, &scales, &rowLoss, &rowScore, &queryRows, &candidateRows, &temperature};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaLaunchContrastiveGrad(EosCudaRuntime* rt, EosCudaKernel* kernel, unsigned int grid, unsigned int block, CUdeviceptr query, CUdeviceptr positive, CUdeviceptr queryNorms, CUdeviceptr positiveNorms, CUdeviceptr scores, CUdeviceptr scales, CUdeviceptr queryGrads, CUdeviceptr positiveGrads, int queryRows, int candidateRows, int width, char** err) {
	void* args[] = {&query, &positive, &queryNorms, &positiveNorms, &scores, &scales, &queryGrads, &positiveGrads, &queryRows, &candidateRows, &width};
	return eosCudaLaunch1D(rt, kernel, grid, block, args, err);
}

static int eosCudaMatMulCublasWithBetaNoSync(EosCudaRuntime* rt, CUdeviceptr lhs, CUdeviceptr rhs, CUdeviceptr out0, int lhsRows, int lhsCols, int rhsRows, int rhsCols, int transposeLeft, int transposeRight, float betaValue, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	int rows = transposeLeft ? lhsCols : lhsRows;
	int inner = transposeLeft ? lhsRows : lhsCols;
	int rhsInner = transposeRight ? rhsCols : rhsRows;
	int cols = transposeRight ? rhsRows : rhsCols;
	if (inner != rhsInner) {
		*err = manta_dup_format("cublasSgemm", "shape mismatch");
		return 1;
	}
	const float alpha = 1.0f;
	const float beta = betaValue;
	const float* lhsPtr = (const float*)(uintptr_t)lhs;
	const float* rhsPtr = (const float*)(uintptr_t)rhs;
	float* outPtr = (float*)(uintptr_t)out0;
	cublasStatus_t blasRes = cublasSgemm(
		rt->blas,
		transposeRight ? CUBLAS_OP_T : CUBLAS_OP_N,
		transposeLeft ? CUBLAS_OP_T : CUBLAS_OP_N,
		cols,
		rows,
		inner,
		&alpha,
		rhsPtr,
		rhsCols,
		lhsPtr,
		lhsCols,
		&beta,
		outPtr,
		cols
	);
	if (blasRes != CUBLAS_STATUS_SUCCESS) {
		*err = manta_dup_cublas_error("cublasSgemm", blasRes);
		return 1;
	}
	return 0;
}

static int eosCudaMatMulCublasWithBeta(EosCudaRuntime* rt, CUdeviceptr lhs, CUdeviceptr rhs, CUdeviceptr out0, int lhsRows, int lhsCols, int rhsRows, int rhsCols, int transposeLeft, int transposeRight, float betaValue, char** err) {
	if (eosCudaMatMulCublasWithBetaNoSync(rt, lhs, rhs, out0, lhsRows, lhsCols, rhsRows, rhsCols, transposeLeft, transposeRight, betaValue, err) != 0) {
		return 1;
	}
	return eosCudaSynchronize(rt, err);
}

static int eosCudaMatMulCublas(EosCudaRuntime* rt, CUdeviceptr lhs, CUdeviceptr rhs, CUdeviceptr out0, int lhsRows, int lhsCols, int rhsRows, int rhsCols, int transposeLeft, int transposeRight, char** err) {
	return eosCudaMatMulCublasWithBeta(rt, lhs, rhs, out0, lhsRows, lhsCols, rhsRows, rhsCols, transposeLeft, transposeRight, 0.0f, err);
}

static int eosCudaMatMulCublasStridedBatched(EosCudaRuntime* rt, CUdeviceptr lhs, CUdeviceptr rhs, CUdeviceptr out0, int batches, int lhsRows, int lhsCols, int rhsRows, int rhsCols, int transposeLeft, int transposeRight, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	if (batches <= 0 || lhsRows <= 0 || lhsCols <= 0 || rhsRows <= 0 || rhsCols <= 0) {
		*err = manta_dup_format("cublasSgemmStridedBatched", "invalid shape");
		return 1;
	}
	int rows = transposeLeft ? lhsCols : lhsRows;
	int inner = transposeLeft ? lhsRows : lhsCols;
	int rhsInner = transposeRight ? rhsCols : rhsRows;
	int cols = transposeRight ? rhsRows : rhsCols;
	if (inner != rhsInner) {
		*err = manta_dup_format("cublasSgemmStridedBatched", "shape mismatch");
		return 1;
	}
	const float alpha = 1.0f;
	const float beta = 0.0f;
	const float* lhsPtr = (const float*)(uintptr_t)lhs;
	const float* rhsPtr = (const float*)(uintptr_t)rhs;
	float* outPtr = (float*)(uintptr_t)out0;
	long long int lhsStride = (long long int)lhsRows * (long long int)lhsCols;
	long long int rhsStride = (long long int)rhsRows * (long long int)rhsCols;
	long long int outStride = (long long int)rows * (long long int)cols;
	cublasStatus_t blasRes = cublasSgemmStridedBatched(
		rt->blas,
		transposeRight ? CUBLAS_OP_T : CUBLAS_OP_N,
		transposeLeft ? CUBLAS_OP_T : CUBLAS_OP_N,
		cols,
		rows,
		inner,
		&alpha,
		rhsPtr,
		rhsCols,
		rhsStride,
		lhsPtr,
		lhsCols,
		lhsStride,
		&beta,
		outPtr,
		cols,
		outStride,
		batches
	);
	if (blasRes != CUBLAS_STATUS_SUCCESS) {
		*err = manta_dup_cublas_error("cublasSgemmStridedBatched", blasRes);
		return 1;
	}
	cuRes = cuStreamSynchronize(rt->stream);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuStreamSynchronize", cuRes);
		return 1;
	}
	return 0;
}

static void eosCudaFreeCString(char* s) {
	if (s != NULL) {
		free(s);
	}
}

typedef struct {
	CUgraph graph;
	CUgraphExec exec;
} EosCudaGraph;

// eosCudaBeginCapture puts the runtime's (non-default) stream into
// thread-local capture mode. Subsequent stream work — kernel launches and
// cuBLAS GEMMs, both already bound to rt->stream — is recorded into a graph
// instead of executing. cuBLAS workspace for every GEMM shape MUST be warmed
// up before capture begins; allocations are illegal during capture.
static int eosCudaBeginCapture(EosCudaRuntime* rt, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuStreamBeginCapture(rt->stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuStreamBeginCapture", cuRes);
		return 1;
	}
	return 0;
}

// eosCudaEndCapture ends capture, instantiates the recorded graph into an
// executable graph, and returns both (the graph is retained for teardown).
static int eosCudaEndCapture(EosCudaRuntime* rt, EosCudaGraph** out, char** err) {
	CUgraph graph = NULL;
	CUresult cuRes = cuStreamEndCapture(rt->stream, &graph);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuStreamEndCapture", cuRes);
		return 1;
	}
	CUgraphExec exec = NULL;
	cuRes = cuGraphInstantiate(&exec, graph, 0);
	if (cuRes != CUDA_SUCCESS) {
		cuGraphDestroy(graph);
		*err = manta_dup_cu_error("cuGraphInstantiate", cuRes);
		return 1;
	}
	EosCudaGraph* g = (EosCudaGraph*)malloc(sizeof(EosCudaGraph));
	if (g == NULL) {
		cuGraphExecDestroy(exec);
		cuGraphDestroy(graph);
		*err = manta_dup_format("malloc", "failed to allocate graph");
		return 1;
	}
	g->graph = graph;
	g->exec = exec;
	*out = g;
	return 0;
}

// eosCudaGraphLaunch replays a captured graph and synchronizes once. The
// graph references the device pointers recorded at capture time, so replay
// recomputes against whatever those (stable) buffers currently hold.
static int eosCudaGraphLaunch(EosCudaRuntime* rt, EosCudaGraph* g, char** err) {
	CUresult cuRes = cuCtxSetCurrent(rt->ctx);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuCtxSetCurrent", cuRes);
		return 1;
	}
	cuRes = cuGraphLaunch(g->exec, rt->stream);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuGraphLaunch", cuRes);
		return 1;
	}
	cuRes = cuStreamSynchronize(rt->stream);
	if (cuRes != CUDA_SUCCESS) {
		*err = manta_dup_cu_error("cuStreamSynchronize", cuRes);
		return 1;
	}
	return 0;
}

static void eosCudaGraphDestroy(EosCudaGraph* g) {
	if (g == NULL) {
		return;
	}
	if (g->exec != NULL) {
		cuGraphExecDestroy(g->exec);
	}
	if (g->graph != NULL) {
		cuGraphDestroy(g->graph);
	}
	free(g);
}
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"
	"unsafe"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
	turboquant "m31labs.dev/turboquant"
)

const forwardQuantizeKernelSource = `
extern "C" __global__ void manta_forward_quantize_in_place(
    float* data,
    int elements,
    float levels,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements || scale == 0.0f || levels <= 0.0f) {
        return;
    }
    float q = roundf(data[idx] / scale);
    if (q > levels) {
        q = levels;
    }
    if (q < -levels) {
        q = -levels;
    }
    data[idx] = q * scale;
}
`

const gdnKernelSource = `
extern "C" __global__ void manta_gdn_forward(
    const float* input,
    const float* beta,
    const float* gamma,
    float* out,
    int elements,
    int channels,
    int height,
    int width,
    int inverse
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    int spatial = height * width;
    int c = (idx / spatial) % channels;
    int n = idx / (channels * spatial);
    int hw = idx % spatial;
    float sum = beta[c];
    for (int j = 0; j < channels; ++j) {
        int input_idx = ((n * channels + j) * spatial) + hw;
        float v = input[input_idx];
        sum += gamma[c * channels + j] * v * v;
    }
    if (sum < 1.0e-12f) {
        sum = 1.0e-12f;
    }
    float scale = sqrtf(sum);
    if (inverse != 0) {
        out[idx] = input[idx] * scale;
    } else {
        out[idx] = input[idx] / scale;
    }
}
`

const convKernelSource = `
extern "C" __global__ void manta_conv2d_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* out,
    int elements,
    int inChannels,
    int inHeight,
    int inWidth,
    int outChannels,
    int outHeight,
    int outWidth,
    int inPerGroup,
    int outPerGroup,
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int hasBias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    int ox = idx % outWidth;
    int rem = idx / outWidth;
    int oy = rem % outHeight;
    rem /= outHeight;
    int oc = rem % outChannels;
    int n = rem / outChannels;
    int group = oc / outPerGroup;
    float sum = hasBias ? bias[oc] : 0.0f;
    for (int icg = 0; icg < inPerGroup; ++icg) {
        int ic = group * inPerGroup + icg;
        for (int ky = 0; ky < kernelH; ++ky) {
            int iy = oy * strideH + ky * dilationH - padH;
            if (iy < 0 || iy >= inHeight) {
                continue;
            }
            for (int kx = 0; kx < kernelW; ++kx) {
                int ix = ox * strideW + kx * dilationW - padW;
                if (ix < 0 || ix >= inWidth) {
                    continue;
                }
                int inputIdx = ((n * inChannels + ic) * inHeight + iy) * inWidth + ix;
                int weightIdx = ((oc * inPerGroup + icg) * kernelH + ky) * kernelW + kx;
                sum += input[inputIdx] * weight[weightIdx];
            }
        }
    }
    out[idx] = sum;
}

extern "C" __global__ void manta_conv2d_transpose_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* out,
    int elements,
    int inChannels,
    int inHeight,
    int inWidth,
    int outChannels,
    int outHeight,
    int outWidth,
    int inPerGroup,
    int outPerGroup,
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int hasBias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    int ox = idx % outWidth;
    int rem = idx / outWidth;
    int oy = rem % outHeight;
    rem /= outHeight;
    int oc = rem % outChannels;
    int n = rem / outChannels;
    int group = oc / outPerGroup;
    int ocg = oc - group * outPerGroup;
    float sum = hasBias ? bias[oc] : 0.0f;
    for (int icg = 0; icg < inPerGroup; ++icg) {
        int ic = group * inPerGroup + icg;
        for (int ky = 0; ky < kernelH; ++ky) {
            int yNumerator = oy + padH - ky * dilationH;
            if (yNumerator % strideH != 0) {
                continue;
            }
            int iy = yNumerator / strideH;
            if (iy < 0 || iy >= inHeight) {
                continue;
            }
            for (int kx = 0; kx < kernelW; ++kx) {
                int xNumerator = ox + padW - kx * dilationW;
                if (xNumerator % strideW != 0) {
                    continue;
                }
                int ix = xNumerator / strideW;
                if (ix < 0 || ix >= inWidth) {
                    continue;
                }
                int inputIdx = ((n * inChannels + ic) * inHeight + iy) * inWidth + ix;
                int weightIdx = ((ic * outPerGroup + ocg) * kernelH + ky) * kernelW + kx;
                sum += input[inputIdx] * weight[weightIdx];
            }
        }
    }
    out[idx] = sum;
}
`

const turboQKernelSource = `
static __device__ int manta_tq_highest_power_of_two_le(int value) {
    int power = 1;
    while ((power << 1) <= value) {
        power <<= 1;
    }
    return power;
}

static __device__ void manta_tq_fwht_normalized(float* values, int size) {
    if (size <= 1) {
        return;
    }
    for (int step = 1; step < size; step <<= 1) {
        int jump = step << 1;
        for (int i = 0; i < size; i += jump) {
            for (int j = i; j < i + step; ++j) {
                float a = values[j];
                float b = values[j + step];
                values[j] = a + b;
                values[j + step] = a - b;
            }
        }
    }
    float scale = rsqrtf((float)size);
    for (int i = 0; i < size; ++i) {
        values[i] *= scale;
    }
}

static __device__ void manta_tq_apply_blocks(float* values, int channels) {
    int offset = 0;
    int remaining = channels;
    while (remaining > 0) {
        int size = manta_tq_highest_power_of_two_le(remaining);
        manta_tq_fwht_normalized(values + offset, size);
        offset += size;
        remaining -= size;
    }
}

static __device__ int manta_tq_vector_offset_to_nyx(int vector, int height, int width, int* n, int* y, int* x) {
    *x = vector % width;
    int rem = vector / width;
    *y = rem % height;
    *n = rem / height;
    return (*n * height + *y) * width + *x;
}

static __device__ int manta_tq_nchw_offset(int n, int c, int y, int x, int channels, int height, int width) {
    return ((n * channels + c) * height + y) * width + x;
}

static __device__ int manta_tq_nearest_centroid(float value, const float* boundaries, int levels) {
    int lo = 0;
    int hi = levels - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (boundaries[mid] > value) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

static __device__ int manta_tq_quantize_norm(float norm) {
    if (norm <= 0.0f || isnan(norm)) {
        return 0;
    }
    if (isinf(norm) && norm > 0.0f) {
        return 255;
    }
    float t = (logf(norm) + 16.0f) / 32.0f;
    if (t <= 0.0f) {
        return 0;
    }
    if (t >= 1.0f) {
        return 255;
    }
    return (int)roundf(t * 255.0f);
}

static __device__ float manta_tq_dequantize_norm(int encoded) {
    if (encoded < 0) {
        encoded = 0;
    }
    if (encoded > 255) {
        encoded = 255;
    }
    return expf(-16.0f + ((float)encoded / 255.0f) * 32.0f);
}

extern "C" __global__ void manta_turboq_encode(
    const float* input,
    float* coords,
    float* norms,
    float* scratchWork,
    float* scratchRotated,
    const float* perm,
    const float* signs1,
    const float* signs2,
    const float* centroids,
    const float* boundaries,
    int vectors,
    int channels,
    int height,
    int width,
    int levels
) {
    int vector = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector >= vectors) {
        return;
    }
    int n, y, x;
    int normOffset = manta_tq_vector_offset_to_nyx(vector, height, width, &n, &y, &x);
    float* work = scratchWork + vector * channels;
    float* rotated = scratchRotated + vector * channels;
    float normSq = 0.0f;
    for (int c = 0; c < channels; ++c) {
        float value = input[manta_tq_nchw_offset(n, c, y, x, channels, height, width)];
        normSq += value * value;
    }
    float norm = sqrtf(normSq);
    float scale = 1.0f;
    if (norm > 1.0e-12f) {
        scale = 1.0f / norm;
    }
    for (int i = 0; i < channels; ++i) {
        int p = (int)(perm[i] + 0.5f);
        float value = input[manta_tq_nchw_offset(n, p, y, x, channels, height, width)];
        work[i] = value * scale * signs1[i];
    }
    manta_tq_apply_blocks(work, channels);
    for (int i = 0; i < channels; ++i) {
        int p = (int)(perm[i] + 0.5f);
        rotated[p] = work[i] * signs2[i];
    }
    for (int c = 0; c < channels; ++c) {
        int idx = manta_tq_nearest_centroid(rotated[c], boundaries, levels);
        coords[manta_tq_nchw_offset(n, c, y, x, channels, height, width)] = (float)idx;
    }
    norms[normOffset] = (float)manta_tq_quantize_norm(norm);
}

extern "C" __global__ void manta_turboq_decode(
    const float* coords,
    const float* norms,
    float* out,
    float* scratchWork,
    float* scratchRotated,
    const float* perm,
    const float* signs1,
    const float* signs2,
    const float* centroids,
    int vectors,
    int channels,
    int height,
    int width,
    int levels
) {
    int vector = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector >= vectors) {
        return;
    }
    int n, y, x;
    int normOffset = manta_tq_vector_offset_to_nyx(vector, height, width, &n, &y, &x);
    float* work = scratchWork + vector * channels;
    float* rotated = scratchRotated + vector * channels;
    for (int c = 0; c < channels; ++c) {
        int idx = (int)roundf(coords[manta_tq_nchw_offset(n, c, y, x, channels, height, width)]);
        if (idx < 0) {
            idx = 0;
        }
        if (idx >= levels) {
            idx = levels - 1;
        }
        rotated[c] = centroids[idx];
    }
    for (int i = 0; i < channels; ++i) {
        int p = (int)(perm[i] + 0.5f);
        work[i] = rotated[p] * signs2[i];
    }
    manta_tq_apply_blocks(work, channels);
    int normCode = (int)roundf(norms[normOffset]);
    float norm = manta_tq_dequantize_norm(normCode);
    for (int i = 0; i < channels; ++i) {
        int p = (int)(perm[i] + 0.5f);
        out[manta_tq_nchw_offset(n, p, y, x, channels, height, width)] = work[i] * signs1[i] * norm;
    }
}
`

const sparseAttentionKernelSource = `
#define EOS_SPARSE_ATTENTION_MAX_TOPK 128

static __device__ float manta_sparse_attention_score(
    const float* query,
    const float* key,
    int rank,
    int kvLayout,
    int queryRow,
    int batch,
    int keyIndex,
    int queryLen,
    int keyLen,
    int queryDim
) {
    float sum = 0.0f;
    int queryBase = rank == 3 ? ((batch * queryLen + queryRow) * queryDim) : (queryRow * queryDim);
    for (int d = 0; d < queryDim; ++d) {
        int keyIndexFlat = 0;
        if (kvLayout == 1) {
            keyIndexFlat = ((batch * queryDim + d) * keyLen) + keyIndex;
        } else {
            keyIndexFlat = rank == 3 ? ((batch * keyLen + keyIndex) * queryDim + d) : (keyIndex * queryDim + d);
        }
        sum += query[queryBase + d] * key[keyIndexFlat];
    }
    return sum;
}

static __device__ bool manta_sparse_attention_selected(const int* selected, int selectedCount, int candidate) {
    for (int i = 0; i < selectedCount; ++i) {
        if (selected[i] == candidate) {
            return true;
        }
    }
    return false;
}

extern "C" __global__ void manta_sparse_attention_forward(
    const float* query,
    const float* key,
    const float* value,
    float* out,
    int rank,
    int kvLayout,
    int batches,
    int queryLen,
    int keyLen,
    int queryDim,
    int valueDim,
    int topK
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int totalRows = batches * queryLen;
    if (row >= totalRows) {
        return;
    }
    if (topK > EOS_SPARSE_ATTENTION_MAX_TOPK) {
        return;
    }
    int batch = rank == 3 ? row / queryLen : 0;
    int queryRow = rank == 3 ? row - batch * queryLen : row;
    int selected[EOS_SPARSE_ATTENTION_MAX_TOPK];
    float selectedScores[EOS_SPARSE_ATTENTION_MAX_TOPK];
    int selectedCount = 0;
    for (int pick = 0; pick < topK; ++pick) {
        int bestIndex = -1;
        float bestScore = -3.4028234663852886e38f;
        for (int k = 0; k < keyLen; ++k) {
            if (manta_sparse_attention_selected(selected, selectedCount, k)) {
                continue;
            }
            float score = manta_sparse_attention_score(query, key, rank, kvLayout, queryRow, batch, k, queryLen, keyLen, queryDim);
            if (score > bestScore || (score == bestScore && (bestIndex < 0 || k < bestIndex))) {
                bestScore = score;
                bestIndex = k;
            }
        }
        if (bestIndex < 0) {
            break;
        }
        selected[selectedCount] = bestIndex;
        selectedScores[selectedCount] = bestScore;
        ++selectedCount;
    }
    if (selectedCount == 0) {
        return;
    }
    float maxScore = selectedScores[0];
    for (int i = 1; i < selectedCount; ++i) {
        if (selectedScores[i] > maxScore) {
            maxScore = selectedScores[i];
        }
    }
    float denom = 0.0f;
    for (int i = 0; i < selectedCount; ++i) {
        denom += expf(selectedScores[i] - maxScore);
    }
    if (denom == 0.0f || isnan(denom)) {
        return;
    }
    int outBase = rank == 3 ? ((batch * queryLen + queryRow) * valueDim) : (queryRow * valueDim);
    for (int vd = 0; vd < valueDim; ++vd) {
        float sum = 0.0f;
        for (int i = 0; i < selectedCount; ++i) {
            float weight = expf(selectedScores[i] - maxScore) / denom;
            int valueIndex = 0;
            if (kvLayout == 1) {
                valueIndex = ((batch * valueDim + vd) * keyLen) + selected[i];
            } else {
                valueIndex = rank == 3 ? ((batch * keyLen + selected[i]) * valueDim + vd) : (selected[i] * valueDim + vd);
            }
            sum += weight * value[valueIndex];
        }
        out[outBase + vd] = sum;
    }
}
`

const turboSparseAttentionKernelSource = turboQKernelSource + `
#define EOS_TURBO_SPARSE_ATTENTION_MAX_TOPK 128
#define EOS_TURBO_SPARSE_ATTENTION_MAX_ROUTE_BLOCKS 128

static __device__ bool manta_turbo_sparse_attention_better(float lhsScore, int lhsIndex, float rhsScore, int rhsIndex) {
    if (lhsScore > rhsScore) {
        return true;
    }
    if (lhsScore < rhsScore) {
        return false;
    }
    return lhsIndex < rhsIndex;
}

static __device__ void manta_turbo_sparse_attention_insert(
    int* selected,
    float* selectedScores,
    int* selectedCount,
    int topK,
    int keyIndex,
    float score
) {
    int pos = *selectedCount;
    if (*selectedCount < topK) {
        *selectedCount = *selectedCount + 1;
    } else {
        pos = topK - 1;
        if (!manta_turbo_sparse_attention_better(score, keyIndex, selectedScores[pos], selected[pos])) {
            return;
        }
    }
    while (pos > 0 && manta_turbo_sparse_attention_better(score, keyIndex, selectedScores[pos - 1], selected[pos - 1])) {
        selected[pos] = selected[pos - 1];
        selectedScores[pos] = selectedScores[pos - 1];
        --pos;
    }
    selected[pos] = keyIndex;
    selectedScores[pos] = score;
}

static __device__ void manta_turbo_sparse_decode_vector(
    const float* coords,
    const float* norms,
    float* work,
    float* decoded,
    const float* perm,
    const float* signs1,
    const float* signs2,
    const float* centroids,
    int batch,
    int vectorIndex,
    int channels,
    int vectors,
    int levels
) {
    int normOffset = batch * vectors + vectorIndex;
    for (int c = 0; c < channels; ++c) {
        int idx = (int)roundf(coords[manta_tq_nchw_offset(batch, c, vectorIndex, 0, channels, vectors, 1)]);
        if (idx < 0) {
            idx = 0;
        }
        if (idx >= levels) {
            idx = levels - 1;
        }
        decoded[c] = centroids[idx];
    }
    for (int i = 0; i < channels; ++i) {
        int p = (int)(perm[i] + 0.5f);
        work[i] = decoded[p] * signs2[i];
    }
    manta_tq_apply_blocks(work, channels);
    int normCode = (int)roundf(norms[normOffset]);
    float norm = manta_tq_dequantize_norm(normCode);
    for (int i = 0; i < channels; ++i) {
        int p = (int)(perm[i] + 0.5f);
        decoded[p] = work[i] * signs1[i] * norm;
    }
}

extern "C" __global__ void manta_turbo_sparse_attention_forward(
    const float* query,
    const float* keyCoords,
    const float* keyNorms,
    const float* valueCoords,
    const float* valueNorms,
    float* out,
    float* keyScratchWork,
    float* keyScratchDecoded,
    float* valueScratchWork,
    float* valueScratchDecoded,
    const float* keyPerm,
    const float* keySigns1,
    const float* keySigns2,
    const float* keyCentroids,
    const float* valuePerm,
    const float* valueSigns1,
    const float* valueSigns2,
    const float* valueCentroids,
    int rank,
    int batches,
    int queryLen,
    int keyLen,
    int queryDim,
    int valueDim,
    int topK,
    int keyLevels,
    int valueLevels,
    int routeBlockSize,
    int routeTopBlocks
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int totalRows = batches * queryLen;
    if (row >= totalRows) {
        return;
    }
    if (topK <= 0 || topK > EOS_TURBO_SPARSE_ATTENTION_MAX_TOPK) {
        return;
    }
    int batch = rank == 3 ? row / queryLen : 0;
    int queryRow = rank == 3 ? row - batch * queryLen : row;
    int queryBase = rank == 3 ? ((batch * queryLen + queryRow) * queryDim) : (queryRow * queryDim);
    int routed = routeBlockSize > 0 && routeTopBlocks > 0;
    int blockCount = 0;
    if (routed) {
        if (routeBlockSize > keyLen) {
            routeBlockSize = keyLen;
        }
        blockCount = (keyLen + routeBlockSize - 1) / routeBlockSize;
        if (routeTopBlocks > blockCount) {
            routeTopBlocks = blockCount;
        }
        if (routeTopBlocks <= 0 || routeTopBlocks > EOS_TURBO_SPARSE_ATTENTION_MAX_ROUTE_BLOCKS) {
            return;
        }
    }
    float* keyWork = keyScratchWork + row * queryDim;
    float* decodedKey = keyScratchDecoded + row * queryDim;

    int selected[EOS_TURBO_SPARSE_ATTENTION_MAX_TOPK];
    float selectedScores[EOS_TURBO_SPARSE_ATTENTION_MAX_TOPK];
    int selectedCount = 0;
    if (routed) {
        int selectedBlocks[EOS_TURBO_SPARSE_ATTENTION_MAX_ROUTE_BLOCKS];
        float selectedBlockScores[EOS_TURBO_SPARSE_ATTENTION_MAX_ROUTE_BLOCKS];
        int selectedBlockCount = 0;
        for (int block = 0; block < blockCount; ++block) {
            int start = block * routeBlockSize;
            int end = start + routeBlockSize;
            if (end > keyLen) {
                end = keyLen;
            }
            int anchor = start + ((end - start) >> 1);
            manta_turbo_sparse_decode_vector(keyCoords, keyNorms, keyWork, decodedKey, keyPerm, keySigns1, keySigns2, keyCentroids, batch, anchor, queryDim, keyLen, keyLevels);
            float score = 0.0f;
            for (int d = 0; d < queryDim; ++d) {
                score += query[queryBase + d] * decodedKey[d];
            }
            manta_turbo_sparse_attention_insert(selectedBlocks, selectedBlockScores, &selectedBlockCount, routeTopBlocks, block, score);
        }
        for (int bi = 0; bi < selectedBlockCount; ++bi) {
            int start = selectedBlocks[bi] * routeBlockSize;
            int end = start + routeBlockSize;
            if (end > keyLen) {
                end = keyLen;
            }
            for (int k = start; k < end; ++k) {
                manta_turbo_sparse_decode_vector(keyCoords, keyNorms, keyWork, decodedKey, keyPerm, keySigns1, keySigns2, keyCentroids, batch, k, queryDim, keyLen, keyLevels);
                float score = 0.0f;
                for (int d = 0; d < queryDim; ++d) {
                    score += query[queryBase + d] * decodedKey[d];
                }
                manta_turbo_sparse_attention_insert(selected, selectedScores, &selectedCount, topK, k, score);
            }
        }
    } else {
        for (int k = 0; k < keyLen; ++k) {
            manta_turbo_sparse_decode_vector(keyCoords, keyNorms, keyWork, decodedKey, keyPerm, keySigns1, keySigns2, keyCentroids, batch, k, queryDim, keyLen, keyLevels);
            float score = 0.0f;
            for (int d = 0; d < queryDim; ++d) {
                score += query[queryBase + d] * decodedKey[d];
            }
            manta_turbo_sparse_attention_insert(selected, selectedScores, &selectedCount, topK, k, score);
        }
    }
    if (selectedCount == 0) {
        return;
    }
    float maxScore = selectedScores[0];
    for (int i = 1; i < selectedCount; ++i) {
        if (selectedScores[i] > maxScore) {
            maxScore = selectedScores[i];
        }
    }
    float denom = 0.0f;
    for (int i = 0; i < selectedCount; ++i) {
        denom += expf(selectedScores[i] - maxScore);
    }
    if (denom == 0.0f || isnan(denom)) {
        return;
    }

    int outBase = rank == 3 ? ((batch * queryLen + queryRow) * valueDim) : (queryRow * valueDim);
    for (int vd = 0; vd < valueDim; ++vd) {
        out[outBase + vd] = 0.0f;
    }
    float* valueWork = valueScratchWork + row * valueDim;
    float* decodedValue = valueScratchDecoded + row * valueDim;
    for (int i = 0; i < selectedCount; ++i) {
        float weight = expf(selectedScores[i] - maxScore) / denom;
        manta_turbo_sparse_decode_vector(valueCoords, valueNorms, valueWork, decodedValue, valuePerm, valueSigns1, valueSigns2, valueCentroids, batch, selected[i], valueDim, keyLen, valueLevels);
        for (int vd = 0; vd < valueDim; ++vd) {
            out[outBase + vd] += weight * decodedValue[vd];
        }
    }
}
`

const mseLossKernelSource = `
extern "C" __global__ void manta_mse_partials(
    const float* lhs,
    const float* rhs,
    float* partials,
    int elements
) {
    __shared__ float shared[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (int i = idx; i < elements; i += stride) {
        float diff = lhs[i] - rhs[i];
        sum += diff * diff;
    }
    shared[tid] = sum;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partials[blockIdx.x] = shared[0];
    }
}
`

const msssimLossKernelSource = `
extern "C" __global__ void manta_msssim_partials(
    const float* lhs,
    const float* rhs,
    float* partials,
    int elements
) {
    __shared__ float sumA[256];
    __shared__ float sumB[256];
    __shared__ float sumAA[256];
    __shared__ float sumBB[256];
    __shared__ float sumAB[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    float a = 0.0f;
    float b = 0.0f;
    float aa = 0.0f;
    float bb = 0.0f;
    float ab = 0.0f;
    for (int i = idx; i < elements; i += stride) {
        float x = lhs[i];
        float y = rhs[i];
        a += x;
        b += y;
        aa += x * x;
        bb += y * y;
        ab += x * y;
    }
    sumA[tid] = a;
    sumB[tid] = b;
    sumAA[tid] = aa;
    sumBB[tid] = bb;
    sumAB[tid] = ab;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sumA[tid] += sumA[tid + offset];
            sumB[tid] += sumB[tid + offset];
            sumAA[tid] += sumAA[tid + offset];
            sumBB[tid] += sumBB[tid + offset];
            sumAB[tid] += sumAB[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        int base = blockIdx.x * 5;
        partials[base + 0] = sumA[0];
        partials[base + 1] = sumB[0];
        partials[base + 2] = sumAA[0];
        partials[base + 3] = sumBB[0];
        partials[base + 4] = sumAB[0];
    }
}
`

const scalarLossKernelSource = `
extern "C" __global__ void manta_scalar_sum(
    const float* values,
    float* out,
    int count
) {
    __shared__ float shared[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < count; i += blockDim.x) {
        sum += values[i];
    }
    shared[tid] = sum;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[0] = shared[0];
    }
}

extern "C" __global__ void manta_rd_loss(
    const float* distortion,
    const float* rate,
    float* out,
    float lambda
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = distortion[0] + lambda * rate[0];
    }
}
`

const crossEntropyKernelSource = `
static __device__ int manta_clamp_int(int value, int lo, int hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

static __device__ float manta_neg_log2(float p) {
    if (!(p >= 1.0e-12f)) {
        p = 1.0e-12f;
    }
    return -logf(p) * 1.4426950408889634f;
}

static __device__ float manta_softmax_probability_strided(const float* values, int base, int step, int count, int idx) {
    float maxv = values[base];
    for (int i = 1; i < count; ++i) {
        float v = values[base + i * step];
        if (v > maxv) {
            maxv = v;
        }
    }
    float sum = 0.0f;
    float target = 0.0f;
    for (int i = 0; i < count; ++i) {
        float v = expf(values[base + i * step] - maxv);
        if (i == idx) {
            target = v;
        }
        sum += v;
    }
    if (sum == 0.0f) {
        return 1.0f / (float)count;
    }
    return target / sum;
}

static __device__ void manta_unpack_offset4(int offset, int channels, int height, int width, int* n, int* c, int* h, int* w) {
    int spatial = height * width;
    *w = offset % width;
    int rem = offset / width;
    *h = rem % height;
    rem /= height;
    *c = rem % channels;
    *n = rem / channels;
}

static __device__ float manta_categorical_probability(
    const float* logits,
    int layout,
    int offset,
    int idx,
    int levels,
    int codeC,
    int codeH,
    int codeW,
    int logitsLen,
    int logitsC,
    int logitsH,
    int logitsW
) {
    if (layout == 0 || logits == 0 || logitsLen == 0) {
        return 1.0f / (float)levels;
    }
    if (layout == 1) {
        return manta_softmax_probability_strided(logits, 0, 1, levels, idx);
    }
    if (layout == 2) {
        return manta_softmax_probability_strided(logits, offset * levels, 1, levels, idx);
    }
    if (layout == 3) {
        int n, c, h, w;
        manta_unpack_offset4(offset, codeC, codeH, codeW, &n, &c, &h, &w);
        int baseChannel = c * levels;
        int base = ((n * logitsC + baseChannel) * logitsH + h) * logitsW + w;
        return manta_softmax_probability_strided(logits, base, logitsH * logitsW, levels, idx);
    }
    float p = 1.0f / (1.0f + expf(-logits[offset % logitsLen]));
    if (idx == 0) {
        return 1.0f - p;
    }
    int denom = levels - 1;
    if (denom < 1) {
        denom = 1;
    }
    return p / (float)denom;
}

static __device__ float manta_bit_probability(
    const float* logits,
    int layout,
    int offset,
    int bit,
    int bitValue,
    int bits,
    int codeC,
    int codeH,
    int codeW,
    int logitsLen,
    int logitsC,
    int logitsH,
    int logitsW
) {
    if (layout == 0 || logits == 0 || logitsLen == 0) {
        return 0.5f;
    }
    if (layout == 1) {
        return manta_softmax_probability_strided(logits, bit * 2, 1, 2, bitValue);
    }
    if (layout == 2) {
        return manta_softmax_probability_strided(logits, (offset * bits + bit) * 2, 1, 2, bitValue);
    }
    if (layout == 3) {
        int n, c, h, w;
        manta_unpack_offset4(offset, codeC, codeH, codeW, &n, &c, &h, &w);
        int ch = c * bits * 2 + bit * 2;
        int base = ((n * logitsC + ch) * logitsH + h) * logitsW + w;
        return manta_softmax_probability_strided(logits, base, logitsH * logitsW, 2, bitValue);
    }
    float p = 1.0f / (1.0f + expf(-logits[(offset * bits + bit) % logitsLen]));
    if (bitValue == 1) {
        return p;
    }
    return 1.0f - p;
}

static __device__ float manta_normal_cdf(float x) {
    return 0.5f * (1.0f + erff(x * 0.7071067811865475f));
}

static __device__ float manta_norm_sigma(float raw, int sigmaMode) {
    if (sigmaMode == 1) {
        if (raw > 32.0f) {
            return raw + 1.0e-6f;
        }
        return log1pf(expf(raw)) + 1.0e-6f;
    }
    if (sigmaMode == 2) {
        return expf(raw);
    }
    return raw;
}

static __device__ float manta_log_normal_loss(
    const float* codes,
    const float* params,
    int offset,
    int codeH,
    int codeW,
    int paramsC,
    int paramsH,
    int paramsW,
    int sigmaMode
) {
    int x = offset % codeW;
    int rem = offset / codeW;
    int y = rem % codeH;
    int n = rem / codeH;
    float mu = params[((n * paramsC + 0) * paramsH + y) * paramsW + x];
    float sigma = manta_norm_sigma(params[((n * paramsC + 1) * paramsH + y) * paramsW + x], sigmaMode);
    int sym = manta_clamp_int((int)roundf(codes[offset]), 0, 255);
    float span = 32.0f;
    float p;
    if (sym <= 0) {
        float hi = -16.0f + (0.5f / 255.0f) * span;
        p = manta_normal_cdf((hi - mu) / sigma);
    } else if (sym >= 255) {
        float lo = -16.0f + (254.5f / 255.0f) * span;
        p = 1.0f - manta_normal_cdf((lo - mu) / sigma);
    } else {
        float lo = -16.0f + (((float)sym - 0.5f) / 255.0f) * span;
        float hi = -16.0f + (((float)sym + 0.5f) / 255.0f) * span;
        p = manta_normal_cdf((hi - mu) / sigma) - manta_normal_cdf((lo - mu) / sigma);
    }
    return manta_neg_log2(p);
}

extern "C" __global__ void manta_cross_entropy_partials(
    const float* codes,
    const float* logits,
    float* partials,
    int elements,
    int mode,
    int layout,
    int levels,
    int bits,
    int codeRank,
    int codeN,
    int codeC,
    int codeH,
    int codeW,
    int logitsLen,
    int logitsN,
    int logitsC,
    int logitsH,
    int logitsW,
    int sigmaMode
) {
    __shared__ float shared[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (int i = idx; i < elements; i += stride) {
        if (mode == 2) {
            sum += manta_log_normal_loss(codes, logits, i, codeH, codeW, logitsC, logitsH, logitsW, sigmaMode);
        } else if (mode == 1) {
            int maxSymbol = (1 << bits) - 1;
            int symbol = manta_clamp_int((int)roundf(codes[i]), 0, maxSymbol);
            for (int bit = 0; bit < bits; ++bit) {
                int shift = bits - 1 - bit;
                int bitValue = (symbol >> shift) & 1;
                float p = manta_bit_probability(logits, layout, i, bit, bitValue, bits, codeC, codeH, codeW, logitsLen, logitsC, logitsH, logitsW);
                sum += manta_neg_log2(p);
            }
        } else {
            int symbol = manta_clamp_int((int)roundf(codes[i]), 0, levels - 1);
            float p = manta_categorical_probability(logits, layout, i, symbol, levels, codeC, codeH, codeW, logitsLen, logitsC, logitsH, logitsW);
            sum += manta_neg_log2(p);
        }
    }
    shared[tid] = sum;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partials[blockIdx.x] = shared[0];
    }
}
`

const bertEmbeddingAffineLayerNormKernelSource = `
extern "C" __global__ void manta_bert_embedding_affine_layernorm(
    const float* token_embeddings,
    const float* position_embeddings,
    const float* token_type_embeddings,
    const float* gamma,
    const float* beta,
    const int* input_ids,
    const int* token_type_ids,
    float* out0,
    int* status,
    int rows,
    int tokens,
    int hidden,
    int vocab,
    int max_positions,
    int type_vocab,
    double epsilon
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    int token_id = input_ids[row];
    int position_id = row % tokens;
    int token_type_id = token_type_ids[row];
    if (token_id < 0 || token_id >= vocab || position_id < 0 || position_id >= max_positions || token_type_id < 0 || token_type_id >= type_vocab) {
        atomicExch(status, 1);
        return;
    }
    int out_base = row * hidden;
    int token_base = token_id * hidden;
    int position_base = position_id * hidden;
    int type_base = token_type_id * hidden;
    double mean = 0.0;
    for (int d = 0; d < hidden; ++d) {
        float value = token_embeddings[token_base + d] + position_embeddings[position_base + d] + token_type_embeddings[type_base + d];
        out0[out_base + d] = value;
        mean += (double)value;
    }
    mean /= (double)hidden;
    double variance = 0.0;
    for (int d = 0; d < hidden; ++d) {
        double centered = (double)out0[out_base + d] - mean;
        variance += centered * centered;
    }
    variance /= (double)hidden;
    double inv_std = rsqrt(variance + epsilon);
    for (int d = 0; d < hidden; ++d) {
        double normalized = ((double)out0[out_base + d] - mean) * inv_std;
        out0[out_base + d] = (float)(normalized * (double)gamma[d] + (double)beta[d]);
    }
}
`

const bertExactGELUKernelSource = `
extern "C" __global__ void manta_bert_exact_gelu(
    const float* src,
    float* dst,
    int elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    double x = (double)src[idx];
    dst[idx] = (float)(0.5 * x * (1.0 + erf(x * 0.70710678118654752440)));
}
`

const bertResidualAffineLayerNormKernelSource = `
extern "C" __global__ void manta_bert_residual_affine_layernorm(
    const float* src,
    const float* residual,
    const float* gamma,
    const float* beta,
    float* out0,
    int rows,
    int hidden,
    double epsilon
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    int base = row * hidden;
    double mean = 0.0;
    for (int d = 0; d < hidden; ++d) {
        float value = src[base + d] + residual[base + d];
        out0[base + d] = value;
        mean += (double)value;
    }
    mean /= (double)hidden;
    double variance = 0.0;
    for (int d = 0; d < hidden; ++d) {
        double centered = (double)out0[base + d] - mean;
        variance += centered * centered;
    }
    variance /= (double)hidden;
    double inv_std = rsqrt(variance + epsilon);
    for (int d = 0; d < hidden; ++d) {
        double normalized = ((double)out0[base + d] - mean) * inv_std;
        out0[base + d] = (float)(normalized * (double)gamma[d] + (double)beta[d]);
    }
}
`

const bertCLSL2KernelSource = `
extern "C" __global__ void manta_bert_cls_l2(
    const float* hidden_states,
    float* out0,
    int batch,
    int tokens,
    int hidden
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) {
        return;
    }
    int src_base = b * tokens * hidden;
    int out_base = b * hidden;
    double norm2 = 0.0;
    for (int d = 0; d < hidden; ++d) {
        float value = hidden_states[src_base + d];
        out0[out_base + d] = value;
        norm2 += (double)value * (double)value;
    }
    if (norm2 == 0.0) {
        for (int d = 0; d < hidden; ++d) {
            out0[out_base + d] = 0.0f;
        }
        return;
    }
    double inv_norm = rsqrt(norm2);
    for (int d = 0; d < hidden; ++d) {
        out0[out_base + d] = (float)((double)out0[out_base + d] * inv_norm);
    }
}
`

const bertBiasAddKernelSource = `
extern "C" __global__ void manta_bert_bias_add(
    float* data,
    const float* bias,
    int rows,
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elements = rows * cols;
    if (idx >= elements) {
        return;
    }
    int col = idx % cols;
    data[idx] += bias[col];
}
`

const bertAttentionContextKernelSource = `
extern "C" __global__ void manta_bert_attention_context(
    const float* query,
    const float* key,
    const float* value,
    const int* attention_mask,
    float* out0,
    int batch,
    int tokens,
    int hidden,
    int heads,
    int head_dim
) {
    __shared__ double logits[512];
    __shared__ float qvec[32];
    __shared__ double reduce[256];
    __shared__ int active_reduce[256];

    int job = blockIdx.x;
    int jobs = batch * tokens * heads;
    if (job >= jobs) {
        return;
    }
    if (batch <= 0 || tokens <= 0 || tokens > 512 || heads <= 0 || head_dim <= 0 || head_dim > 32 || hidden != heads * head_dim) {
        return;
    }
    int head = job % heads;
    int query_index = job / heads;
    int b = query_index / tokens;
    int query_token = query_index % tokens;
    int query_row = b * tokens + query_token;
    int head_base = head * head_dim;
    double scale = rsqrt((double)head_dim);

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        qvec[d] = query[query_row * hidden + head_base + d];
    }
    __syncthreads();

    double local_max = -1.0 / 0.0;
    int local_active = 0;
    for (int key_token = threadIdx.x; key_token < tokens; key_token += blockDim.x) {
        double logit = -1.0 / 0.0;
        if (attention_mask[b * tokens + key_token] != 0) {
            int key_row = b * tokens + key_token;
            double dot = 0.0;
            for (int d = 0; d < head_dim; ++d) {
                dot += (double)qvec[d] * (double)key[key_row * hidden + head_base + d];
            }
            logit = dot * scale;
            if (logit > local_max) {
                local_max = logit;
            }
            local_active += 1;
        }
        logits[key_token] = logit;
    }
    reduce[threadIdx.x] = local_max;
    active_reduce[threadIdx.x] = local_active;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            double other = reduce[threadIdx.x + stride];
            if (other > reduce[threadIdx.x]) {
                reduce[threadIdx.x] = other;
            }
            active_reduce[threadIdx.x] += active_reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    double max_logit = reduce[0];
    int active_keys = active_reduce[0];

    int out_base = query_row * hidden + head_base;
    if (active_keys == 0) {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            out0[out_base + d] = 0.0f;
        }
        return;
    }

    double local_sum = 0.0;
    for (int key_token = threadIdx.x; key_token < tokens; key_token += blockDim.x) {
        if (attention_mask[b * tokens + key_token] != 0) {
            local_sum += exp(logits[key_token] - max_logit);
        }
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    double sum_exp = reduce[0];

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        double acc = 0.0;
        for (int key_token = 0; key_token < tokens; ++key_token) {
            if (attention_mask[b * tokens + key_token] == 0) {
                continue;
            }
            int key_row = b * tokens + key_token;
            double prob = exp(logits[key_token] - max_logit) / sum_exp;
            acc += prob * (double)value[key_row * hidden + head_base + d];
        }
        out0[out_base + d] = (float)acc;
    }
}
`

type deviceRuntime struct {
	ptr                         *C.EosCudaRuntime
	bgeFullEncoderMu            sync.Mutex
	residentMatrices            map[string]residentMatrix
	bertResidentTensors         map[string]residentTensor
	bertResidentCache           map[string]bertCUDAResidentBindingCache
	bertSelectedContractCache   map[string]bertCUDASelectedContract
	matMulScratch               map[string]deviceScratchBuffer
	quantizeKernel              *auxKernel
	gdnKernel                   *auxKernel
	conv2DKernel                *auxKernel
	conv2DTransKernel           *auxKernel
	turboQEncodeKernel          *auxKernel
	turboQDecodeKernel          *auxKernel
	sparseAttnKernel            *auxKernel
	turboSparseKernel           *auxKernel
	mseLossKernel               *auxKernel
	msssimLossKernel            *auxKernel
	scalarAddKernel             *auxKernel
	rdLossKernel                *auxKernel
	crossEntropyKernel          *auxKernel
	bertEmbeddingKernel         *auxKernel
	bertExactGELUKernel         *auxKernel
	bertResidualLayerNormKernel *auxKernel
	bertCLSL2Kernel             *auxKernel
	bertBiasAddKernel           *auxKernel
	bertAttentionContextKernel  *auxKernel
	matMulStats                 backend.MatMulAcceleratorStats
	graphCache                  map[string]*cudaGraph
}

type residentMatrix struct {
	ptr      C.CUdeviceptr
	rows     int
	cols     int
	elements int
}

type residentTensor struct {
	ptr      C.CUdeviceptr
	shape    []int
	elements int
}

type deviceScratchBuffer struct {
	ptr      C.CUdeviceptr
	elements int
}

type deviceKernel struct {
	ptr       *C.EosCudaKernel
	shapeKind cudaShapeKind
}

type auxKernel struct {
	ptr *C.EosCudaKernel
}

type cudaShapeKind int

const (
	cudaShapeUnsupported cudaShapeKind = iota
	cudaShapeRowWise
	cudaShapeElementWiseBinary
	cudaShapeElementWiseUnary
	cudaShapeRowScore
	// cudaShapeRoPE is like cudaShapeRowWise but the launched kernel also
	// receives the original sequence length so it can recover each token's
	// intra-sequence position (row % seq_len) instead of using the flattened
	// [B*T] row index as the rotary position directly.
	cudaShapeRoPE
)

func newDeviceRuntime() (*deviceRuntime, error) {
	var rt *C.EosCudaRuntime
	var errStr *C.char
	if C.eosCudaRuntimeCreate(&rt, &errStr) != 0 {
		return nil, cStringError(errStr)
	}
	return &deviceRuntime{ptr: rt, residentMatrices: map[string]residentMatrix{}, bertResidentTensors: map[string]residentTensor{}, bertResidentCache: map[string]bertCUDAResidentBindingCache{}, bertSelectedContractCache: map[string]bertCUDASelectedContract{}, matMulScratch: map[string]deviceScratchBuffer{}, graphCache: map[string]*cudaGraph{}}, nil
}

func (rt *deviceRuntime) close() {
	if rt == nil || rt.ptr == nil {
		return
	}
	for name, resident := range rt.residentMatrices {
		_ = rt.freeBuffer(resident.ptr)
		delete(rt.residentMatrices, name)
	}
	for name, resident := range rt.bertResidentTensors {
		_ = rt.freeBuffer(resident.ptr)
		delete(rt.bertResidentTensors, name)
	}
	for name := range rt.bertResidentCache {
		delete(rt.bertResidentCache, name)
	}
	for name := range rt.bertSelectedContractCache {
		delete(rt.bertSelectedContractCache, name)
	}
	for name, scratch := range rt.matMulScratch {
		_ = rt.freeBuffer(scratch.ptr)
		delete(rt.matMulScratch, name)
	}
	rt.destroyAuxKernel(rt.quantizeKernel)
	rt.quantizeKernel = nil
	rt.destroyAuxKernel(rt.gdnKernel)
	rt.gdnKernel = nil
	rt.destroyAuxKernel(rt.conv2DKernel)
	rt.conv2DKernel = nil
	rt.destroyAuxKernel(rt.conv2DTransKernel)
	rt.conv2DTransKernel = nil
	rt.destroyAuxKernel(rt.turboQEncodeKernel)
	rt.turboQEncodeKernel = nil
	rt.destroyAuxKernel(rt.turboQDecodeKernel)
	rt.turboQDecodeKernel = nil
	rt.destroyAuxKernel(rt.sparseAttnKernel)
	rt.sparseAttnKernel = nil
	rt.destroyAuxKernel(rt.turboSparseKernel)
	rt.turboSparseKernel = nil
	rt.destroyAuxKernel(rt.mseLossKernel)
	rt.mseLossKernel = nil
	rt.destroyAuxKernel(rt.msssimLossKernel)
	rt.msssimLossKernel = nil
	rt.destroyAuxKernel(rt.scalarAddKernel)
	rt.scalarAddKernel = nil
	rt.destroyAuxKernel(rt.rdLossKernel)
	rt.rdLossKernel = nil
	rt.destroyAuxKernel(rt.crossEntropyKernel)
	rt.crossEntropyKernel = nil
	rt.destroyAuxKernel(rt.bertEmbeddingKernel)
	rt.bertEmbeddingKernel = nil
	rt.destroyAuxKernel(rt.bertExactGELUKernel)
	rt.bertExactGELUKernel = nil
	rt.destroyAuxKernel(rt.bertResidualLayerNormKernel)
	rt.bertResidualLayerNormKernel = nil
	rt.destroyAuxKernel(rt.bertCLSL2Kernel)
	rt.bertCLSL2Kernel = nil
	rt.destroyAuxKernel(rt.bertBiasAddKernel)
	rt.bertBiasAddKernel = nil
	rt.destroyAuxKernel(rt.bertAttentionContextKernel)
	rt.bertAttentionContextKernel = nil
	for sig, g := range rt.graphCache {
		g.destroy()
		delete(rt.graphCache, sig)
	}
	C.eosCudaRuntimeDestroy(rt.ptr)
	rt.ptr = nil
}

func (rt *deviceRuntime) matMulStatsSnapshot() backend.MatMulAcceleratorStats {
	if rt == nil {
		return backend.MatMulAcceleratorStats{}
	}
	stats := rt.matMulStats
	stats.BoundMatrices = int64(len(rt.residentMatrices))
	return stats
}

func (rt *deviceRuntime) recordMatMulRun(start time.Time, uploadedBytes, downloadedBytes int64, boundLeft, boundRight bool) {
	if rt == nil {
		return
	}
	rt.matMulStats.RunCalls++
	if boundLeft {
		rt.matMulStats.BoundLeftCalls++
	}
	if boundRight {
		rt.matMulStats.BoundRightCalls++
	}
	rt.matMulStats.RunUploadedBytes += uploadedBytes
	rt.matMulStats.RunDownloadedBytes += downloadedBytes
	rt.matMulStats.RunNanos += time.Since(start).Nanoseconds()
}

func (rt *deviceRuntime) attachDeviceExecution(prog *backend.NativeKernelProgram, kernel eosartifact.Kernel) error {
	shapeKind := classifyCUDAKernel(kernel)
	if shapeKind == cudaShapeUnsupported {
		prog.LaunchConfig["device_execution"] = false
		prog.LaunchConfig["execution_mode"] = "host_fallback"
		return nil
	}
	if shapeKind == cudaShapeRoPE {
		// classifyCUDAKernel routed here purely on the backend-neutral
		// kernel.Body[0].Op == "rope". The GPU code that will actually run,
		// though, is whatever source was JIT-compiled from
		// prog.Compiled.Source -- persisted verbatim inside the sealed .mll
		// artifact at compile time. Artifacts sealed before the seq_len fix
		// carry the OLD 4-param `rope_cuda(in0, out0, rows, cols)` source.
		// runRoPEKernel/launchRoPE unconditionally pass 5 args; per the CUDA
		// driver ABI (cuLaunchKernel with a raw void* args[] array) a
		// 4-param kernel silently ignores the extra 5th argument rather than
		// erroring, so the old batch-position-leaking bug would keep
		// executing with no error at all. Fail loudly here instead.
		if err := validateRoPEKernelABI(kernel.Name, prog.Compiled); err != nil {
			return err
		}
	}
	deviceKernel, err := rt.compileKernel(prog.Compiled, shapeKind)
	if err != nil {
		return err
	}
	prog.LaunchConfig["device_execution"] = true
	prog.LaunchConfig["execution_mode"] = "cuda_device"
	prog.LaunchConfig["launch_compiler"] = "nvrtc"
	prog.Run = func(inputs []*backend.Tensor) ([]*backend.Tensor, error) {
		return rt.runKernel(deviceKernel, kernel, prog, inputs)
	}
	return nil
}

func (rt *deviceRuntime) compileKernel(compiled backend.CompiledKernel, shapeKind cudaShapeKind) (*deviceKernel, error) {
	src := C.CString(compiled.Source)
	entry := C.CString(compiled.Entry)
	defer C.free(unsafe.Pointer(src))
	defer C.free(unsafe.Pointer(entry))

	var kernel *C.EosCudaKernel
	var logStr *C.char
	var errStr *C.char
	rc := C.eosCudaCompileKernel(rt.ptr, src, entry, &kernel, &logStr, &errStr)
	logText := cStringValue(logStr)
	if rc != 0 {
		err := cStringError(errStr)
		if logText != "" {
			return nil, fmt.Errorf("%w\n%s", err, logText)
		}
		return nil, err
	}
	return &deviceKernel{ptr: kernel, shapeKind: shapeKind}, nil
}

func (rt *deviceRuntime) compileAuxKernel(source, entry string) (*auxKernel, error) {
	src := C.CString(source)
	cEntry := C.CString(entry)
	defer C.free(unsafe.Pointer(src))
	defer C.free(unsafe.Pointer(cEntry))

	var kernel *C.EosCudaKernel
	var logStr *C.char
	var errStr *C.char
	rc := C.eosCudaCompileKernel(rt.ptr, src, cEntry, &kernel, &logStr, &errStr)
	logText := cStringValue(logStr)
	if rc != 0 {
		err := cStringError(errStr)
		if logText != "" {
			return nil, fmt.Errorf("%w\n%s", err, logText)
		}
		return nil, err
	}
	return &auxKernel{ptr: kernel}, nil
}

func (rt *deviceRuntime) destroyAuxKernel(kernel *auxKernel) {
	if kernel == nil || kernel.ptr == nil {
		return
	}
	C.eosCudaKernelDestroy(kernel.ptr)
}

func (rt *deviceRuntime) runKernel(deviceKernel *deviceKernel, kernel eosartifact.Kernel, prog *backend.NativeKernelProgram, inputs []*backend.Tensor) ([]*backend.Tensor, error) {
	switch deviceKernel.shapeKind {
	case cudaShapeRowWise:
		return rt.runRowWiseKernel(deviceKernel, kernel, prog, inputs)
	case cudaShapeRoPE:
		return rt.runRoPEKernel(deviceKernel, kernel, prog, inputs)
	case cudaShapeElementWiseBinary:
		return rt.runElementWiseKernel(deviceKernel, kernel, prog, inputs)
	case cudaShapeElementWiseUnary:
		return rt.runUnaryKernel(deviceKernel, kernel, prog, inputs)
	case cudaShapeRowScore:
		return rt.runScoreKernel(deviceKernel, kernel, prog, inputs)
	default:
		return prog.Run(inputs)
	}
}

func (rt *deviceRuntime) runRowWiseKernel(deviceKernel *deviceKernel, kernel eosartifact.Kernel, prog *backend.NativeKernelProgram, inputs []*backend.Tensor) ([]*backend.Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("kernel %q expected 1 input for row-wise launch, got %d", kernel.Name, len(inputs))
	}
	in := inputs[0]
	if in == nil || (len(in.Shape) != 2 && len(in.Shape) != 3) {
		return nil, fmt.Errorf("kernel %q expected rank-2 or rank-3 input", kernel.Name)
	}
	outShape := append([]int(nil), in.Shape...)
	rows := in.Shape[0]
	cols := in.Shape[len(in.Shape)-1]
	if len(in.Shape) == 3 {
		rows = in.Shape[0] * in.Shape[1]
	}
	if rows == 0 || cols == 0 {
		return []*backend.Tensor{newOutputTensor(kernel, outShape, make([]float32, rows*cols))}, nil
	}
	inBuf, err := rt.uploadFloat32(in.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(inBuf)
	outHost := make([]float32, rows*cols)
	outBuf, err := rt.allocFloat32(len(outHost))
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(outBuf)
	rowsArg := C.int(rows)
	colsArg := C.int(cols)
	block := C.uint(firstIntValue(prog.LaunchConfig["launch_block_size"], 128))
	grid := C.uint((rows + int(block) - 1) / int(block))
	if err := rt.launchRowWise(deviceKernel.ptr, grid, block, inBuf, outBuf, rowsArg, colsArg); err != nil {
		return nil, err
	}
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return nil, err
	}
	return []*backend.Tensor{newOutputTensor(kernel, outShape, outHost)}, nil
}

// runRoPEKernel is runRowWiseKernel's counterpart for the "rope" op. Unlike
// the other row-wise ops (normalize/layernorm/softmax), RoPE's result for a
// given row depends on that token's position *within its own sequence*. For
// a rank-3 [B, T, D] input the row-wise launch flattens B and T into a single
// `rows = B*T` grid, so the kernel additionally needs `seq_len = T` to
// recover the true position via `row % seq_len` — otherwise identical tokens
// rotate differently depending on which batch row they land in (see
// compiler.ropeBody / cudaNativeEmitter). For rank-2 [T, D] input there is no
// batching, so seq_len == rows and the modulo is a no-op.
func (rt *deviceRuntime) runRoPEKernel(deviceKernel *deviceKernel, kernel eosartifact.Kernel, prog *backend.NativeKernelProgram, inputs []*backend.Tensor) ([]*backend.Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("kernel %q expected 1 input for rope launch, got %d", kernel.Name, len(inputs))
	}
	in := inputs[0]
	if in == nil || (len(in.Shape) != 2 && len(in.Shape) != 3) {
		return nil, fmt.Errorf("kernel %q expected rank-2 or rank-3 input", kernel.Name)
	}
	outShape := append([]int(nil), in.Shape...)
	rows := in.Shape[0]
	cols := in.Shape[len(in.Shape)-1]
	seqLen := rows
	if len(in.Shape) == 3 {
		seqLen = in.Shape[1]
		rows = in.Shape[0] * in.Shape[1]
	}
	if rows == 0 || cols == 0 {
		return []*backend.Tensor{newOutputTensor(kernel, outShape, make([]float32, rows*cols))}, nil
	}
	inBuf, err := rt.uploadFloat32(in.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(inBuf)
	outHost := make([]float32, rows*cols)
	outBuf, err := rt.allocFloat32(len(outHost))
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(outBuf)
	rowsArg := C.int(rows)
	colsArg := C.int(cols)
	seqLenArg := C.int(seqLen)
	block := C.uint(firstIntValue(prog.LaunchConfig["launch_block_size"], 128))
	grid := C.uint((rows + int(block) - 1) / int(block))
	if err := rt.launchRoPE(deviceKernel.ptr, grid, block, inBuf, outBuf, rowsArg, colsArg, seqLenArg); err != nil {
		return nil, err
	}
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return nil, err
	}
	return []*backend.Tensor{newOutputTensor(kernel, outShape, outHost)}, nil
}

func (rt *deviceRuntime) runElementWiseKernel(deviceKernel *deviceKernel, kernel eosartifact.Kernel, prog *backend.NativeKernelProgram, inputs []*backend.Tensor) ([]*backend.Tensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("kernel %q expected 2 inputs for element-wise launch, got %d", kernel.Name, len(inputs))
	}
	lhs := inputs[0]
	rhs := inputs[1]
	if lhs == nil || rhs == nil || !lhs.EqualShape(rhs) {
		return nil, fmt.Errorf("kernel %q expected matching input shapes", kernel.Name)
	}
	elements := lhs.Elements()
	lhsBuf, err := rt.uploadFloat32(lhs.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(lhsBuf)
	rhsBuf, err := rt.uploadFloat32(rhs.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(rhsBuf)
	outHost := make([]float32, elements)
	outBuf, err := rt.allocFloat32(elements)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(outBuf)
	elementsArg := C.int(elements)
	block := C.uint(firstIntValue(prog.LaunchConfig["launch_block_size"], 128))
	grid := C.uint((elements + int(block) - 1) / int(block))
	if err := rt.launchElementWise(deviceKernel.ptr, grid, block, lhsBuf, rhsBuf, outBuf, elementsArg); err != nil {
		return nil, err
	}
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return nil, err
	}
	return []*backend.Tensor{newOutputTensor(kernel, append([]int(nil), lhs.Shape...), outHost)}, nil
}

func (rt *deviceRuntime) runUnaryKernel(deviceKernel *deviceKernel, kernel eosartifact.Kernel, prog *backend.NativeKernelProgram, inputs []*backend.Tensor) ([]*backend.Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("kernel %q expected 1 input for unary launch, got %d", kernel.Name, len(inputs))
	}
	in := inputs[0]
	if in == nil {
		return nil, fmt.Errorf("kernel %q expected non-nil input", kernel.Name)
	}
	elements := in.Elements()
	inBuf, err := rt.uploadFloat32(in.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(inBuf)
	outHost := make([]float32, elements)
	outBuf, err := rt.allocFloat32(elements)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(outBuf)
	elementsArg := C.int(elements)
	block := C.uint(firstIntValue(prog.LaunchConfig["launch_block_size"], 128))
	grid := C.uint((elements + int(block) - 1) / int(block))
	if err := rt.launchUnary(deviceKernel.ptr, grid, block, inBuf, outBuf, elementsArg); err != nil {
		return nil, err
	}
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return nil, err
	}
	return []*backend.Tensor{newOutputTensor(kernel, append([]int(nil), in.Shape...), outHost)}, nil
}

func (rt *deviceRuntime) runScoreKernel(deviceKernel *deviceKernel, kernel eosartifact.Kernel, prog *backend.NativeKernelProgram, inputs []*backend.Tensor) ([]*backend.Tensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("kernel %q expected 2 inputs for score launch, got %d", kernel.Name, len(inputs))
	}
	query := inputs[0]
	docs := inputs[1]
	if query == nil || docs == nil || len(query.Shape) != 1 || len(docs.Shape) != 2 || query.Shape[0] != docs.Shape[1] {
		return nil, fmt.Errorf("kernel %q expected query rank-1 and docs rank-2 inputs", kernel.Name)
	}
	rows := docs.Shape[0]
	cols := docs.Shape[1]
	queryBuf, err := rt.uploadFloat32(query.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(queryBuf)
	docsBuf, err := rt.uploadFloat32(docs.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(docsBuf)
	outHost := make([]float32, rows)
	outBuf, err := rt.allocFloat32(rows)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(outBuf)
	rowsArg := C.int(rows)
	colsArg := C.int(cols)
	block := C.uint(firstIntValue(prog.LaunchConfig["launch_block_size"], 128))
	grid := C.uint((rows + int(block) - 1) / int(block))
	if err := rt.launchScore(deviceKernel.ptr, grid, block, queryBuf, docsBuf, outBuf, rowsArg, colsArg); err != nil {
		return nil, err
	}
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return nil, err
	}
	return []*backend.Tensor{newOutputTensor(kernel, []int{rows}, outHost)}, nil
}

func classifyCUDAKernel(kernel eosartifact.Kernel) cudaShapeKind {
	if len(kernel.Body) < 2 {
		return cudaShapeUnsupported
	}
	switch kernel.Body[0].Op {
	case "normalize", "rmsnorm", "layernorm", "softmax":
		return cudaShapeRowWise
	case "rope":
		return cudaShapeRoPE
	case "binary_add", "binary_sub", "binary_mul", "binary_div":
		return cudaShapeElementWiseBinary
	case "dequant", "gelu":
		return cudaShapeElementWiseUnary
	case "dot", "cosine", "l2_distance":
		return cudaShapeRowScore
	default:
		return cudaShapeUnsupported
	}
}

// ropeEntrySignaturePattern locates a CUDA kernel entry point's declared
// parameter list: `void <entry>(...)`. It is scoped to the named entry point
// (rather than a bare `strings.Contains(source, "seq_len")` on the whole
// source blob) so that an unrelated "seq_len" token elsewhere in the file
// can't produce a false pass.
var ropeEntrySignaturePattern = regexp.MustCompile(`\bvoid\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)`)

// ropeEntryParams returns the parenthesized parameter-list text for the
// named CUDA kernel entry point declared in source, e.g. for
// `extern "C" __global__ void rope_cuda(const float* in0, float* out0, int
// rows, int cols, int seq_len)` and entry "rope_cuda" it returns
// "const float* in0, float* out0, int rows, int cols, int seq_len".
func ropeEntryParams(source, entry string) (string, bool) {
	for _, m := range ropeEntrySignaturePattern.FindAllStringSubmatch(source, -1) {
		if m[1] == entry {
			return m[2], true
		}
	}
	return "", false
}

// validateRoPEKernelABI guards against stale-artifact rope kernels: `.mll`
// modules sealed before the seq_len fix (see compiler.go's
// emitCUDARoPEKernel / cudaNativeEmitter.ropeBody) persist a 4-parameter
// `rope_cuda(in0, out0, rows, cols)` kernel source with the old, batch-
// row-leaking `theta = row / powf(...)` math baked in. The current
// runRoPEKernel/launchRoPE always launch with 5 arguments (in0, out0, rows,
// cols, seq_len); dispatching those 5 args at a 4-param kernel doesn't
// error -- the CUDA driver's raw `void* args[]` launch ABI has no arity
// checking, so the extra seq_len pointer is simply never read and the old
// bug keeps executing silently.
//
// This inspects prog.Compiled.Source/Entry -- the actual bytes nvrtc is
// about to JIT-compile and the GPU is about to execute -- rather than only
// trusting the KernelVariant.Meta["rope_abi"] provenance tag set by
// emitKernelVariants, since a source/metadata desync (stale tag, hand-built
// artifact, future refactor) must not be able to mask a stale kernel body.
func validateRoPEKernelABI(kernelName string, compiled backend.CompiledKernel) error {
	params, ok := ropeEntryParams(compiled.Source, compiled.Entry)
	if !ok || !strings.Contains(params, "seq_len") {
		return fmt.Errorf("stale rope kernel ABI for kernel %q (compiled before seq_len fix): recompile/reseal the module with a current eos binary (init-model --bootstrap-from <old.mll>)", kernelName)
	}
	return nil
}

func newOutputTensor(kernel eosartifact.Kernel, shape []int, data []float32) *backend.Tensor {
	dtype := "f16"
	if len(kernel.Outputs) > 0 && kernel.Outputs[0].Type.Tensor != nil && kernel.Outputs[0].Type.Tensor.DType != "" {
		dtype = kernel.Outputs[0].Type.Tensor.DType
	}
	switch dtype {
	case "f32":
		return &backend.Tensor{DType: "f32", Shape: append([]int(nil), shape...), F32: data}
	default:
		return &backend.Tensor{DType: "f16", Shape: append([]int(nil), shape...), F32: data}
	}
}

func checkedCUDABytes(label string, elements, bytesPerElement int) (C.size_t, error) {
	if elements < 0 {
		return 0, fmt.Errorf("%s element count %d is negative", label, elements)
	}
	if bytesPerElement <= 0 {
		return 0, fmt.Errorf("%s byte width %d is invalid", label, bytesPerElement)
	}
	maxInt := int(^uint(0) >> 1)
	if elements > maxInt/bytesPerElement {
		return 0, fmt.Errorf("%s byte size overflows int: elements=%d bytes_per_element=%d", label, elements, bytesPerElement)
	}
	return C.size_t(elements * bytesPerElement), nil
}

func checkedCInt(label string, value int) (C.int, error) {
	const minCInt = -1 << 31
	const maxCInt = 1<<31 - 1
	if value < minCInt || value > maxCInt {
		return 0, fmt.Errorf("%s=%d overflows C.int", label, value)
	}
	return C.int(value), nil
}

func checkedCUint(label string, value uint) (C.uint, error) {
	if uint64(value) > uint64(^uint32(0)) {
		return 0, fmt.Errorf("%s=%d overflows C.uint", label, value)
	}
	return C.uint(value), nil
}

func checkedLaunch1D(label string, elements int, block uint) (C.uint, C.uint, error) {
	if elements <= 0 {
		return 0, 0, fmt.Errorf("%s launch requires positive element count, got %d", label, elements)
	}
	if block == 0 {
		return 0, 0, fmt.Errorf("%s launch block size must be positive", label)
	}
	grid64 := (uint64(elements) + uint64(block) - 1) / uint64(block)
	if grid64 == 0 || grid64 > uint64(^uint32(0)) {
		return 0, 0, fmt.Errorf("%s launch grid %d overflows C.uint", label, grid64)
	}
	blockArg, err := checkedCUint(label+" block", block)
	if err != nil {
		return 0, 0, err
	}
	return C.uint(grid64), blockArg, nil
}

func checkedProduct(label string, dims ...int) (int, error) {
	product := 1
	maxInt := int(^uint(0) >> 1)
	for _, dim := range dims {
		if dim < 0 {
			return 0, fmt.Errorf("%s dimension %d is negative", label, dim)
		}
		if dim != 0 && product > maxInt/dim {
			return 0, fmt.Errorf("%s product overflows int for dimensions %v", label, dims)
		}
		product *= dim
	}
	return product, nil
}

func checkedInt32Product(label string, dims ...int) error {
	const maxInt32 = int64(1<<31 - 1)
	product := int64(1)
	for _, dim := range dims {
		if dim < 0 {
			return fmt.Errorf("%s dimension %d is negative", label, dim)
		}
		if dim == 0 {
			product = 0
			continue
		}
		if product > maxInt32/int64(dim) {
			return fmt.Errorf("%s product overflows int32 index range for dimensions %v", label, dims)
		}
		product *= int64(dim)
	}
	return nil
}

func validateCUDAF32CompatibleDType(t *backend.Tensor, label string) error {
	if t == nil {
		return fmt.Errorf("%s tensor is nil", label)
	}
	if t.DType != "f32" && t.DType != "f16" {
		return fmt.Errorf("%s dtype %q is not f32-compatible", label, t.DType)
	}
	return nil
}

func validateCUDAF32CompatibleTensor(t *backend.Tensor, label string) error {
	if err := validateCUDAF32CompatibleDType(t, label); err != nil {
		return err
	}
	if t.Elements() != len(t.F32) {
		return fmt.Errorf("%s tensor backing length mismatch", label)
	}
	return nil
}

func (rt *deviceRuntime) allocFloat32(elements int) (C.CUdeviceptr, error) {
	var ptr C.CUdeviceptr
	var errStr *C.char
	bytes, err := checkedCUDABytes("cuda float32 allocation", elements, 4)
	if err != nil {
		return 0, err
	}
	if C.eosCudaMemAlloc(rt.ptr, &ptr, bytes, &errStr) != 0 {
		return 0, cStringError(errStr)
	}
	return ptr, nil
}

func (rt *deviceRuntime) allocInt32(elements int) (C.CUdeviceptr, error) {
	var ptr C.CUdeviceptr
	var errStr *C.char
	bytes, err := checkedCUDABytes("cuda int32 allocation", elements, 4)
	if err != nil {
		return 0, err
	}
	if C.eosCudaMemAlloc(rt.ptr, &ptr, bytes, &errStr) != 0 {
		return 0, cStringError(errStr)
	}
	return ptr, nil
}

func (rt *deviceRuntime) uploadFloat32(data []float32) (C.CUdeviceptr, error) {
	ptr, err := rt.allocFloat32(len(data))
	if err != nil {
		return 0, err
	}
	if err := rt.copyFloat32ToBuffer(ptr, data); err != nil {
		_ = rt.freeBuffer(ptr)
		return 0, err
	}
	return ptr, nil
}

func (rt *deviceRuntime) uploadInt32(data []int32) (C.CUdeviceptr, error) {
	ptr, err := rt.allocInt32(len(data))
	if err != nil {
		return 0, err
	}
	if err := rt.copyInt32ToBuffer(ptr, data); err != nil {
		_ = rt.freeBuffer(ptr)
		return 0, err
	}
	return ptr, nil
}

func (rt *deviceRuntime) matMulScratchFloat32(name string, elements int) (C.CUdeviceptr, error) {
	if elements == 0 {
		return 0, nil
	}
	if rt.matMulScratch == nil {
		rt.matMulScratch = map[string]deviceScratchBuffer{}
	}
	if scratch, ok := rt.matMulScratch[name]; ok && scratch.elements >= elements {
		return scratch.ptr, nil
	}
	if scratch, ok := rt.matMulScratch[name]; ok {
		_ = rt.freeBuffer(scratch.ptr)
		delete(rt.matMulScratch, name)
	}
	ptr, err := rt.allocFloat32(elements)
	if err != nil {
		return 0, err
	}
	rt.matMulScratch[name] = deviceScratchBuffer{ptr: ptr, elements: elements}
	return ptr, nil
}

func (rt *deviceRuntime) uploadMatMulScratchFloat32(name string, data []float32) (C.CUdeviceptr, error) {
	ptr, err := rt.matMulScratchFloat32(name, len(data))
	if err != nil {
		return 0, err
	}
	if len(data) == 0 {
		return ptr, nil
	}
	if err := rt.copyFloat32ToBuffer(ptr, data); err != nil {
		return 0, err
	}
	return ptr, nil
}

func (rt *deviceRuntime) copyFloat32ToBuffer(ptr C.CUdeviceptr, data []float32) error {
	var src unsafe.Pointer
	if len(data) > 0 {
		src = unsafe.Pointer(&data[0])
	}
	bytes, err := checkedCUDABytes("cuda float32 upload", len(data), 4)
	if err != nil {
		return err
	}
	var errStr *C.char
	if C.eosCudaMemcpyHtoD(rt.ptr, ptr, src, bytes, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) copyInt32ToBuffer(ptr C.CUdeviceptr, data []int32) error {
	var src unsafe.Pointer
	if len(data) > 0 {
		src = unsafe.Pointer(&data[0])
	}
	bytes, err := checkedCUDABytes("cuda int32 upload", len(data), 4)
	if err != nil {
		return err
	}
	var errStr *C.char
	if C.eosCudaMemcpyHtoD(rt.ptr, ptr, src, bytes, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) downloadFloat32(dst []float32, src C.CUdeviceptr) error {
	if len(dst) == 0 {
		return nil
	}
	bytes, err := checkedCUDABytes("cuda float32 download", len(dst), 4)
	if err != nil {
		return err
	}
	var errStr *C.char
	if C.eosCudaMemcpyDtoH(rt.ptr, unsafe.Pointer(&dst[0]), src, bytes, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) downloadInt32(dst []int32, src C.CUdeviceptr) error {
	if len(dst) == 0 {
		return nil
	}
	bytes, err := checkedCUDABytes("cuda int32 download", len(dst), 4)
	if err != nil {
		return err
	}
	var errStr *C.char
	if C.eosCudaMemcpyDtoH(rt.ptr, unsafe.Pointer(&dst[0]), src, bytes, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) readFloat32At(src C.CUdeviceptr, elementIndex int) (float32, error) {
	var out C.float
	var errStr *C.char
	if C.eosCudaReadFloat32(rt.ptr, src, C.int(elementIndex), &out, &errStr) != 0 {
		return 0, cStringError(errStr)
	}
	return float32(out), nil
}

func (rt *deviceRuntime) maxAbsFloat32(src C.CUdeviceptr, elements int) (float32, error) {
	if elements <= 0 {
		return 0, nil
	}
	var index C.int
	var errStr *C.char
	if C.eosCudaBlasIsamax(rt.ptr, src, C.int(elements), &index, &errStr) != 0 {
		return 0, cStringError(errStr)
	}
	if index <= 0 {
		return 0, nil
	}
	value, err := rt.readFloat32At(src, int(index)-1)
	if err != nil {
		return 0, err
	}
	return float32(math.Abs(float64(value))), nil
}

func (rt *deviceRuntime) ensureQuantizeKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.quantizeKernel != nil {
		return rt.quantizeKernel, nil
	}
	kernel, err := rt.compileAuxKernel(forwardQuantizeKernelSource, "manta_forward_quantize_in_place")
	if err != nil {
		return nil, err
	}
	rt.quantizeKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) ensureGDNKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.gdnKernel != nil {
		return rt.gdnKernel, nil
	}
	kernel, err := rt.compileAuxKernel(gdnKernelSource, "manta_gdn_forward")
	if err != nil {
		return nil, err
	}
	rt.gdnKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) ensureConv2DKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.conv2DKernel != nil {
		return rt.conv2DKernel, nil
	}
	kernel, err := rt.compileAuxKernel(convKernelSource, "manta_conv2d_forward")
	if err != nil {
		return nil, err
	}
	rt.conv2DKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) ensureConv2DTransposeKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.conv2DTransKernel != nil {
		return rt.conv2DTransKernel, nil
	}
	kernel, err := rt.compileAuxKernel(convKernelSource, "manta_conv2d_transpose_forward")
	if err != nil {
		return nil, err
	}
	rt.conv2DTransKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) ensureTurboQEncodeKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.turboQEncodeKernel != nil {
		return rt.turboQEncodeKernel, nil
	}
	kernel, err := rt.compileAuxKernel(turboQKernelSource, "manta_turboq_encode")
	if err != nil {
		return nil, err
	}
	rt.turboQEncodeKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) ensureTurboQDecodeKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.turboQDecodeKernel != nil {
		return rt.turboQDecodeKernel, nil
	}
	kernel, err := rt.compileAuxKernel(turboQKernelSource, "manta_turboq_decode")
	if err != nil {
		return nil, err
	}
	rt.turboQDecodeKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) ensureSparseAttentionKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.sparseAttnKernel != nil {
		return rt.sparseAttnKernel, nil
	}
	kernel, err := rt.compileAuxKernel(sparseAttentionKernelSource, "manta_sparse_attention_forward")
	if err != nil {
		return nil, err
	}
	rt.sparseAttnKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) ensureTurboSparseAttentionKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.turboSparseKernel != nil {
		return rt.turboSparseKernel, nil
	}
	kernel, err := rt.compileAuxKernel(turboSparseAttentionKernelSource, "manta_turbo_sparse_attention_forward")
	if err != nil {
		return nil, err
	}
	rt.turboSparseKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) ensureMSELossKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.mseLossKernel != nil {
		return rt.mseLossKernel, nil
	}
	kernel, err := rt.compileAuxKernel(mseLossKernelSource, "manta_mse_partials")
	if err != nil {
		return nil, err
	}
	rt.mseLossKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) ensureMSSSIMLossKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.msssimLossKernel != nil {
		return rt.msssimLossKernel, nil
	}
	kernel, err := rt.compileAuxKernel(msssimLossKernelSource, "manta_msssim_partials")
	if err != nil {
		return nil, err
	}
	rt.msssimLossKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) ensureScalarAddKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.scalarAddKernel != nil {
		return rt.scalarAddKernel, nil
	}
	kernel, err := rt.compileAuxKernel(scalarLossKernelSource, "manta_scalar_sum")
	if err != nil {
		return nil, err
	}
	rt.scalarAddKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) ensureRDLossKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.rdLossKernel != nil {
		return rt.rdLossKernel, nil
	}
	kernel, err := rt.compileAuxKernel(scalarLossKernelSource, "manta_rd_loss")
	if err != nil {
		return nil, err
	}
	rt.rdLossKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) ensureCrossEntropyKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.crossEntropyKernel != nil {
		return rt.crossEntropyKernel, nil
	}
	kernel, err := rt.compileAuxKernel(crossEntropyKernelSource, "manta_cross_entropy_partials")
	if err != nil {
		return nil, err
	}
	rt.crossEntropyKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) runGDNStep(inputs []*backend.Tensor, outputType eosartifact.ValueType, inverse bool) (backend.StepDispatchResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if len(inputs) < 3 || inputs[0] == nil || inputs[1] == nil || inputs[2] == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda gdn expects input, beta, and gamma")
	}
	input, beta, gamma := inputs[0], inputs[1], inputs[2]
	if len(input.Shape) != 4 {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda gdn expects NCHW input")
	}
	elements := input.Elements()
	if elements == 0 {
		return backend.StepDispatchResult{Outputs: []*backend.Tensor{newStepOutputTensor(outputType, input.Shape, nil)}}, nil
	}
	channels, height, width := input.Shape[1], input.Shape[2], input.Shape[3]
	if len(beta.F32) < channels || len(gamma.F32) < channels*channels {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda gdn beta/gamma shape mismatch")
	}
	kernel, err := rt.ensureGDNKernel()
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	inputBuf, err := rt.uploadFloat32(input.F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(inputBuf)
	betaBuf, err := rt.uploadFloat32(beta.F32[:channels])
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(betaBuf)
	gammaBuf, err := rt.uploadFloat32(gamma.F32[:channels*channels])
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(gammaBuf)
	outBuf, err := rt.allocFloat32(elements)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(outBuf)
	block := uint(128)
	grid := uint((elements + int(block) - 1) / int(block))
	inverseFlag := 0
	entry := "__builtin_cuda_gdn"
	op := "gdn"
	if inverse {
		inverseFlag = 1
		entry = "__builtin_cuda_igdn"
		op = "igdn"
	}
	if err := rt.launchGDN(kernel, grid, block, inputBuf, betaBuf, gammaBuf, outBuf, elements, channels, height, width, inverseFlag); err != nil {
		return backend.StepDispatchResult{}, err
	}
	outHost := make([]float32, elements)
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	return backend.StepDispatchResult{
		Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, input.Shape, outHost)},
		VariantEntry: entry,
		Metadata: map[string]any{
			"dispatch_mode":    "backend_step",
			"device_execution": true,
			"execution_mode":   "cuda_device",
			"launch_api":       "cuda_driver",
			"op":               op,
		},
	}, nil
}

func (rt *deviceRuntime) runConv2DStep(inputs []*backend.Tensor, outputType eosartifact.ValueType, cfg cudaConv2DConfig) (backend.StepDispatchResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if len(inputs) < 2 || inputs[0] == nil || inputs[1] == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda conv2d expects input and weight")
	}
	kernel, err := rt.ensureConv2DKernel()
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	outShape := []int{cfg.batches, cfg.outChannels, cfg.outHeight, cfg.outWidth}
	outputElements := cfg.batches * cfg.outChannels * cfg.outHeight * cfg.outWidth
	if outputElements == 0 {
		return cudaConvStepResult(outputType, "__builtin_cuda_conv2d", "conv2d", outShape, nil, cfgMetadata(cfg)), nil
	}
	inputBuf, err := rt.uploadFloat32(inputs[0].F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(inputBuf)
	weightBuf, err := rt.uploadFloat32(inputs[1].F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(weightBuf)
	var biasBuf C.CUdeviceptr
	if cfg.hasBias {
		biasBuf, err = rt.uploadFloat32(inputs[2].F32[:cfg.outChannels])
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		defer rt.freeBuffer(biasBuf)
	}
	outBuf, err := rt.allocFloat32(outputElements)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(outBuf)
	block := uint(128)
	grid := uint((outputElements + int(block) - 1) / int(block))
	if err := rt.launchConv2D(kernel, grid, block, inputBuf, weightBuf, biasBuf, outBuf, outputElements, cfg); err != nil {
		return backend.StepDispatchResult{}, err
	}
	outHost := make([]float32, outputElements)
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	return cudaConvStepResult(outputType, "__builtin_cuda_conv2d", "conv2d", outShape, outHost, cfgMetadata(cfg)), nil
}

func (rt *deviceRuntime) runConv2DTransposeStep(inputs []*backend.Tensor, outputType eosartifact.ValueType, cfg cudaConv2DTransposeConfig) (backend.StepDispatchResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if len(inputs) < 2 || inputs[0] == nil || inputs[1] == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda conv2d_transpose expects input and weight")
	}
	kernel, err := rt.ensureConv2DTransposeKernel()
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	outShape := []int{cfg.batches, cfg.outChannels, cfg.outHeight, cfg.outWidth}
	outputElements := cfg.batches * cfg.outChannels * cfg.outHeight * cfg.outWidth
	if outputElements == 0 {
		return cudaConvStepResult(outputType, "__builtin_cuda_conv2d_transpose", "conv2d_transpose", outShape, nil, cfgTransposeMetadata(cfg)), nil
	}
	inputBuf, err := rt.uploadFloat32(inputs[0].F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(inputBuf)
	weightBuf, err := rt.uploadFloat32(inputs[1].F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(weightBuf)
	var biasBuf C.CUdeviceptr
	if cfg.hasBias {
		biasBuf, err = rt.uploadFloat32(inputs[2].F32[:cfg.outChannels])
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		defer rt.freeBuffer(biasBuf)
	}
	outBuf, err := rt.allocFloat32(outputElements)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(outBuf)
	block := uint(128)
	grid := uint((outputElements + int(block) - 1) / int(block))
	if err := rt.launchConv2DTranspose(kernel, grid, block, inputBuf, weightBuf, biasBuf, outBuf, outputElements, cfg); err != nil {
		return backend.StepDispatchResult{}, err
	}
	outHost := make([]float32, outputElements)
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	return cudaConvStepResult(outputType, "__builtin_cuda_conv2d_transpose", "conv2d_transpose", outShape, outHost, cfgTransposeMetadata(cfg)), nil
}

func (rt *deviceRuntime) runTurboQEncodeStep(inputs []*backend.Tensor, _ eosartifact.ValueType, cfg cudaTurboQConfig) (backend.StepDispatchResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if len(inputs) != 1 || inputs[0] == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda turboquant_encode expects input")
	}
	input := inputs[0]
	kernel, err := rt.ensureTurboQEncodeKernel()
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	spec, err := rt.uploadTurboQSpec(cfg)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer spec.free(rt)
	coordsShape := append([]int(nil), input.Shape...)
	normShape := []int{cfg.batches, cfg.height, cfg.width}
	outputElements := cfg.batches * cfg.channels * cfg.height * cfg.width
	vectors := cfg.batches * cfg.height * cfg.width
	if outputElements == 0 || vectors == 0 {
		return turboQEncodeStepResult(cfg, coordsShape, nil, normShape, nil), nil
	}
	inputBuf, err := rt.uploadFloat32(input.F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(inputBuf)
	coordsBuf, err := rt.allocFloat32(outputElements)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(coordsBuf)
	normsBuf, err := rt.allocFloat32(vectors)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(normsBuf)
	scratchWork, err := rt.allocFloat32(outputElements)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(scratchWork)
	scratchRotated, err := rt.allocFloat32(outputElements)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(scratchRotated)
	block := uint(128)
	grid := uint((vectors + int(block) - 1) / int(block))
	if err := rt.launchTurboQEncode(kernel, grid, block, inputBuf, coordsBuf, normsBuf, scratchWork, scratchRotated, spec, vectors, cfg); err != nil {
		return backend.StepDispatchResult{}, err
	}
	coordsHost := make([]float32, outputElements)
	if err := rt.downloadFloat32(coordsHost, coordsBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	normsHost := make([]float32, vectors)
	if err := rt.downloadFloat32(normsHost, normsBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	return turboQEncodeStepResult(cfg, coordsShape, coordsHost, normShape, normsHost), nil
}

func (rt *deviceRuntime) runTurboQDecodeStep(inputs []*backend.Tensor, outputType eosartifact.ValueType, cfg cudaTurboQConfig) (backend.StepDispatchResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if len(inputs) != 2 || inputs[0] == nil || inputs[1] == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda turboquant_decode expects coords and norms")
	}
	kernel, err := rt.ensureTurboQDecodeKernel()
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	spec, err := rt.uploadTurboQSpec(cfg)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer spec.free(rt)
	outShape := append([]int(nil), inputs[0].Shape...)
	outputElements := cfg.batches * cfg.channels * cfg.height * cfg.width
	vectors := cfg.batches * cfg.height * cfg.width
	if outputElements == 0 || vectors == 0 {
		return turboQDecodeStepResult(outputType, cfg, outShape, nil), nil
	}
	coordsBuf, err := rt.uploadFloat32(inputs[0].F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(coordsBuf)
	normsBuf, err := rt.uploadFloat32(inputs[1].F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(normsBuf)
	outBuf, err := rt.allocFloat32(outputElements)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(outBuf)
	scratchWork, err := rt.allocFloat32(outputElements)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(scratchWork)
	scratchRotated, err := rt.allocFloat32(outputElements)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(scratchRotated)
	block := uint(128)
	grid := uint((vectors + int(block) - 1) / int(block))
	if err := rt.launchTurboQDecode(kernel, grid, block, coordsBuf, normsBuf, outBuf, scratchWork, scratchRotated, spec, vectors, cfg); err != nil {
		return backend.StepDispatchResult{}, err
	}
	outHost := make([]float32, outputElements)
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	return turboQDecodeStepResult(outputType, cfg, outShape, outHost), nil
}

func (rt *deviceRuntime) runSparseAttentionStep(inputs []*backend.Tensor, outputType eosartifact.ValueType, cfg cudaSparseAttentionConfig) (backend.StepDispatchResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if len(inputs) != 3 || inputs[0] == nil || inputs[1] == nil || inputs[2] == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda sparse_attention expects query, key, and value")
	}
	kernel, err := rt.ensureSparseAttentionKernel()
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	outShape := sparseAttentionOutputShape(cfg)
	if cfg.outputLen == 0 {
		return sparseAttentionStepResult(outputType, cfg, outShape, nil), nil
	}
	queryBuf, err := rt.uploadFloat32(inputs[0].F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(queryBuf)
	keyBuf, err := rt.uploadFloat32(inputs[1].F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(keyBuf)
	valueBuf, err := rt.uploadFloat32(inputs[2].F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(valueBuf)
	outBuf, err := rt.allocFloat32(cfg.outputLen)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(outBuf)
	block := uint(128)
	rows := cfg.batches * cfg.queryLen
	grid := uint((rows + int(block) - 1) / int(block))
	if err := rt.launchSparseAttention(kernel, grid, block, queryBuf, keyBuf, valueBuf, outBuf, cfg); err != nil {
		return backend.StepDispatchResult{}, err
	}
	outHost := make([]float32, cfg.outputLen)
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	return sparseAttentionStepResult(outputType, cfg, outShape, outHost), nil
}

func (rt *deviceRuntime) runTurboSparseAttentionStep(inputs []*backend.Tensor, outputType eosartifact.ValueType, cfg cudaTurboSparseAttentionConfig) (backend.StepDispatchResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if len(inputs) != 5 || inputs[0] == nil || inputs[1] == nil || inputs[2] == nil || inputs[3] == nil || inputs[4] == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda turbo_sparse_attention expects query, key coords/norms, and value coords/norms")
	}
	kernel, err := rt.ensureTurboSparseAttentionKernel()
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	keySpec, err := rt.uploadTurboQSpec(cfg.key)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer keySpec.free(rt)
	valueSpec, err := rt.uploadTurboQSpec(cfg.value)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer valueSpec.free(rt)
	outShape := sparseAttentionOutputShape(cfg.sparse)
	if cfg.sparse.outputLen == 0 {
		return turboSparseAttentionStepResult(outputType, cfg, outShape, nil), nil
	}
	queryBuf, err := rt.uploadFloat32(inputs[0].F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(queryBuf)

	keyCoordsBuf, err := rt.uploadFloat32(inputs[1].F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(keyCoordsBuf)
	keyNormsBuf, err := rt.uploadFloat32(inputs[2].F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(keyNormsBuf)
	valueCoordsBuf, err := rt.uploadFloat32(inputs[3].F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(valueCoordsBuf)
	valueNormsBuf, err := rt.uploadFloat32(inputs[4].F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(valueNormsBuf)

	rows := cfg.sparse.batches * cfg.sparse.queryLen
	keyScratchElements := rows * cfg.sparse.queryDim
	valueScratchElements := rows * cfg.sparse.valueDim
	keyScratchWork, err := rt.allocFloat32(keyScratchElements)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(keyScratchWork)
	keyScratchDecoded, err := rt.allocFloat32(keyScratchElements)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(keyScratchDecoded)
	valueScratchWork, err := rt.allocFloat32(valueScratchElements)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(valueScratchWork)
	valueScratchDecoded, err := rt.allocFloat32(valueScratchElements)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(valueScratchDecoded)

	block := uint(128)
	outBuf, err := rt.allocFloat32(cfg.sparse.outputLen)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(outBuf)
	grid := uint((rows + int(block) - 1) / int(block))
	if err := rt.launchTurboSparseAttention(kernel, grid, block, queryBuf, keyCoordsBuf, keyNormsBuf, valueCoordsBuf, valueNormsBuf, outBuf, keyScratchWork, keyScratchDecoded, valueScratchWork, valueScratchDecoded, keySpec, valueSpec, cfg); err != nil {
		return backend.StepDispatchResult{}, err
	}
	outHost := make([]float32, cfg.sparse.outputLen)
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	return turboSparseAttentionStepResult(outputType, cfg, outShape, outHost), nil
}

func (rt *deviceRuntime) runMSELossStep(inputs []*backend.Tensor, outputType eosartifact.ValueType) (backend.StepDispatchResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if len(inputs) != 2 || inputs[0] == nil || inputs[1] == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda mse_loss expects two inputs")
	}
	lhs, rhs := inputs[0], inputs[1]
	if !lhs.EqualShape(rhs) || len(lhs.F32) != len(rhs.F32) {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda mse_loss shape mismatch")
	}
	elements := len(lhs.F32)
	if elements == 0 {
		return cudaScalarStepResult(outputType, "__builtin_cuda_mse_loss", "mse_loss", "cuda_reduction", 0), nil
	}
	kernel, err := rt.ensureMSELossKernel()
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	lhsBuf, err := rt.uploadFloat32(lhs.F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(lhsBuf)
	rhsBuf, err := rt.uploadFloat32(rhs.F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(rhsBuf)
	grid, block := reductionLaunchConfig(elements)
	partialsBuf, err := rt.allocFloat32(int(grid))
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(partialsBuf)
	if err := rt.launchMSEPartials(kernel, grid, block, lhsBuf, rhsBuf, partialsBuf, elements); err != nil {
		return backend.StepDispatchResult{}, err
	}
	partials := make([]float32, int(grid))
	if err := rt.downloadFloat32(partials, partialsBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	value := sumFloat32(partials) / float32(elements)
	return cudaScalarStepResult(outputType, "__builtin_cuda_mse_loss", "mse_loss", "cuda_reduction", value), nil
}

func (rt *deviceRuntime) runMSSSIMLossStep(inputs []*backend.Tensor, outputType eosartifact.ValueType) (backend.StepDispatchResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if len(inputs) != 2 || inputs[0] == nil || inputs[1] == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda ms_ssim_loss expects two inputs")
	}
	lhs, rhs := inputs[0], inputs[1]
	if !lhs.EqualShape(rhs) || len(lhs.F32) != len(rhs.F32) {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda ms_ssim_loss shape mismatch")
	}
	elements := len(lhs.F32)
	if elements == 0 {
		return cudaScalarStepResult(outputType, "__builtin_cuda_ms_ssim_loss", "ms_ssim_loss", "cuda_reduction", 0), nil
	}
	kernel, err := rt.ensureMSSSIMLossKernel()
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	lhsBuf, err := rt.uploadFloat32(lhs.F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(lhsBuf)
	rhsBuf, err := rt.uploadFloat32(rhs.F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(rhsBuf)
	grid, block := reductionLaunchConfig(elements)
	partialsBuf, err := rt.allocFloat32(int(grid) * 5)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(partialsBuf)
	if err := rt.launchMSSSIMPartials(kernel, grid, block, lhsBuf, rhsBuf, partialsBuf, elements); err != nil {
		return backend.StepDispatchResult{}, err
	}
	partials := make([]float32, int(grid)*5)
	if err := rt.downloadFloat32(partials, partialsBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	sumA, sumB, sumAA, sumBB, sumAB := 0.0, 0.0, 0.0, 0.0, 0.0
	for i := 0; i < len(partials); i += 5 {
		sumA += float64(partials[i])
		sumB += float64(partials[i+1])
		sumAA += float64(partials[i+2])
		sumBB += float64(partials[i+3])
		sumAB += float64(partials[i+4])
	}
	value := float32(msSSIMLossFromMoments(sumA, sumB, sumAA, sumBB, sumAB, elements))
	return cudaScalarStepResult(outputType, "__builtin_cuda_ms_ssim_loss", "ms_ssim_loss", "cuda_reduction", value), nil
}

func (rt *deviceRuntime) runScalarAddStep(inputs []*backend.Tensor, outputType eosartifact.ValueType) (backend.StepDispatchResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if len(inputs) == 0 {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda scalar_add expects at least one input")
	}
	values := make([]float32, len(inputs))
	for i, input := range inputs {
		if input == nil || len(input.F32) != 1 {
			return backend.StepDispatchResult{}, fmt.Errorf("cuda scalar_add input %d must be scalar", i)
		}
		values[i] = input.F32[0]
	}
	kernel, err := rt.ensureScalarAddKernel()
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	valuesBuf, err := rt.uploadFloat32(values)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(valuesBuf)
	outBuf, err := rt.allocFloat32(1)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(outBuf)
	if err := rt.launchScalarSum(kernel, 1, cudaReductionBlockSize, valuesBuf, outBuf, len(values)); err != nil {
		return backend.StepDispatchResult{}, err
	}
	out := []float32{0}
	if err := rt.downloadFloat32(out, outBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	return cudaScalarStepResult(outputType, "__builtin_cuda_scalar_add", "scalar_add", "cuda_scalar", out[0]), nil
}

func (rt *deviceRuntime) runRDLossStep(inputs []*backend.Tensor, outputType eosartifact.ValueType, lambda float32) (backend.StepDispatchResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if len(inputs) != 2 || inputs[0] == nil || inputs[1] == nil || len(inputs[0].F32) != 1 || len(inputs[1].F32) != 1 {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda rate_distortion_loss expects scalar distortion and rate")
	}
	if math.IsNaN(float64(lambda)) || math.IsInf(float64(lambda), 0) || lambda < 0 {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda rate_distortion_loss lambda must be finite and non-negative")
	}
	kernel, err := rt.ensureRDLossKernel()
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	distortionBuf, err := rt.uploadFloat32(inputs[0].F32[:1])
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(distortionBuf)
	rateBuf, err := rt.uploadFloat32(inputs[1].F32[:1])
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(rateBuf)
	outBuf, err := rt.allocFloat32(1)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(outBuf)
	if err := rt.launchRDLoss(kernel, 1, 1, distortionBuf, rateBuf, outBuf, lambda); err != nil {
		return backend.StepDispatchResult{}, err
	}
	out := []float32{0}
	if err := rt.downloadFloat32(out, outBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	result := cudaScalarStepResult(outputType, "__builtin_cuda_rate_distortion_loss", "rate_distortion_loss", "cuda_scalar", out[0])
	result.Metadata["lambda"] = lambda
	return result, nil
}

func (rt *deviceRuntime) runCrossEntropyStep(inputs []*backend.Tensor, outputType eosartifact.ValueType, plan cudaCrossEntropyPlan) (backend.StepDispatchResult, error) {
	if rt == nil || rt.ptr == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda runtime is not initialized")
	}
	if len(inputs) < 1 || inputs[0] == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda cross_entropy expects codes")
	}
	codes := inputs[0]
	var logits *backend.Tensor
	if len(inputs) > 1 {
		logits = inputs[1]
	}
	if len(codes.F32) != codes.Elements() {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda cross_entropy codes must be dense")
	}
	if err := validateCrossEntropyPlanInputs(codes, logits, plan); err != nil {
		return backend.StepDispatchResult{}, err
	}
	elements := len(codes.F32)
	if elements == 0 {
		return cudaScalarStepResult(outputType, "__builtin_cuda_cross_entropy", "cross_entropy_factorized", "cuda_reduction", 0), nil
	}
	kernel, err := rt.ensureCrossEntropyKernel()
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	codesBuf, err := rt.uploadFloat32(codes.F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(codesBuf)
	var logitsBuf C.CUdeviceptr
	logitsLen := 0
	if logits != nil && len(logits.F32) > 0 {
		logitsLen = len(logits.F32)
		logitsBuf, err = rt.uploadFloat32(logits.F32)
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		defer rt.freeBuffer(logitsBuf)
	}
	grid, block := reductionLaunchConfig(elements)
	partialsBuf, err := rt.allocFloat32(int(grid))
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer rt.freeBuffer(partialsBuf)
	codeRank, codeN, codeC, codeH, codeW := cudaTensorShapeArgs(codes)
	logitsN, logitsC, logitsH, logitsW := 0, 0, 0, 0
	if logits != nil {
		_, logitsN, logitsC, logitsH, logitsW = cudaTensorShapeArgs(logits)
	}
	if err := rt.launchCrossEntropyPartials(kernel, grid, block, codesBuf, logitsBuf, partialsBuf, elements, int(plan.mode), int(plan.layout), plan.levels, plan.bits, codeRank, codeN, codeC, codeH, codeW, logitsLen, logitsN, logitsC, logitsH, logitsW, int(plan.sigmaMode)); err != nil {
		return backend.StepDispatchResult{}, err
	}
	partials := make([]float32, int(grid))
	if err := rt.downloadFloat32(partials, partialsBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	value := sumFloat32(partials)
	result := cudaScalarStepResult(outputType, "__builtin_cuda_cross_entropy", "cross_entropy_factorized", "cuda_reduction", value)
	result.Metadata["cross_entropy_mode"] = cudaCrossEntropyModeName(plan.mode)
	result.Metadata["logits_layout"] = cudaCrossEntropyLayoutName(plan.layout)
	result.Metadata["levels"] = plan.levels
	result.Metadata["bits"] = plan.bits
	if plan.mode == cudaCrossEntropyLogNormal {
		result.Metadata["sigma_parameter"] = cudaSigmaModeName(plan.sigmaMode)
	}
	return result, nil
}

func (rt *deviceRuntime) quantizeBufferInPlace(ptr C.CUdeviceptr, elements, bits int) (bool, error) {
	if rt == nil || rt.ptr == nil || elements == 0 || bits <= 0 {
		return false, nil
	}
	maxAbs, err := rt.maxAbsFloat32(ptr, elements)
	if err != nil {
		return false, err
	}
	if maxAbs == 0 {
		return false, nil
	}
	levelsInt := (1 << uint(bits-1)) - 1
	if levelsInt <= 0 {
		return false, nil
	}
	levels := float32(levelsInt)
	scale := maxAbs / levels
	if scale == 0 {
		return false, nil
	}
	kernel, err := rt.ensureQuantizeKernel()
	if err != nil {
		return false, err
	}
	block := uint(128)
	grid := uint((elements + int(block) - 1) / int(block))
	var errStr *C.char
	if C.eosCudaLaunchQuantizeInPlace(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), ptr, C.int(elements), C.float(levels), C.float(scale), &errStr) != 0 {
		return false, cStringError(errStr)
	}
	return true, nil
}

func quantBitsForPreparedTensor(tensor *backend.Tensor) int {
	if tensor == nil {
		return 0
	}
	switch tensor.DType {
	case "q4":
		return 4
	case "q8":
		return 8
	default:
		return 0
	}
}

func (rt *deviceRuntime) freeBuffer(ptr C.CUdeviceptr) error {
	if ptr == 0 {
		return nil
	}
	var errStr *C.char
	if C.eosCudaMemFree(rt.ptr, ptr, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchRowWise(kernel *C.EosCudaKernel, grid, block C.uint, in0, out0 C.CUdeviceptr, rows, cols C.int) error {
	var errStr *C.char
	if C.eosCudaLaunchRowWise(rt.ptr, kernel, grid, block, in0, out0, rows, cols, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchRoPE(kernel *C.EosCudaKernel, grid, block C.uint, in0, out0 C.CUdeviceptr, rows, cols, seqLen C.int) error {
	var errStr *C.char
	if C.eosCudaLaunchRoPE(rt.ptr, kernel, grid, block, in0, out0, rows, cols, seqLen, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchElementWise(kernel *C.EosCudaKernel, grid, block C.uint, lhs, rhs, out0 C.CUdeviceptr, elements C.int) error {
	var errStr *C.char
	if C.eosCudaLaunchElementWise(rt.ptr, kernel, grid, block, lhs, rhs, out0, elements, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchAuxElementWise(kernel *auxKernel, grid, block uint, lhs, rhs, out0 C.CUdeviceptr, elements int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda auxiliary elementwise kernel is not initialized")
	}
	return rt.launchElementWise(kernel.ptr, C.uint(grid), C.uint(block), lhs, rhs, out0, C.int(elements))
}

func (rt *deviceRuntime) launchGDN(kernel *auxKernel, grid, block uint, input, beta, gamma, out0 C.CUdeviceptr, elements, channels, height, width, inverse int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda gdn kernel is not initialized")
	}
	var errStr *C.char
	if C.eosCudaLaunchGDN(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), input, beta, gamma, out0, C.int(elements), C.int(channels), C.int(height), C.int(width), C.int(inverse), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchConv2D(kernel *auxKernel, grid, block uint, input, weight, bias, out0 C.CUdeviceptr, elements int, cfg cudaConv2DConfig) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda conv2d kernel is not initialized")
	}
	hasBias := 0
	if cfg.hasBias {
		hasBias = 1
	}
	var errStr *C.char
	if C.eosCudaLaunchConv2D(
		rt.ptr,
		kernel.ptr,
		C.uint(grid),
		C.uint(block),
		input,
		weight,
		bias,
		out0,
		C.int(elements),
		C.int(cfg.inChannels),
		C.int(cfg.inHeight),
		C.int(cfg.inWidth),
		C.int(cfg.outChannels),
		C.int(cfg.outHeight),
		C.int(cfg.outWidth),
		C.int(cfg.inPerGroup),
		C.int(cfg.outPerGroup),
		C.int(cfg.kernelH),
		C.int(cfg.kernelW),
		C.int(cfg.strideH),
		C.int(cfg.strideW),
		C.int(cfg.padH),
		C.int(cfg.padW),
		C.int(cfg.dilationH),
		C.int(cfg.dilationW),
		C.int(hasBias),
		&errStr,
	) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchConv2DTranspose(kernel *auxKernel, grid, block uint, input, weight, bias, out0 C.CUdeviceptr, elements int, cfg cudaConv2DTransposeConfig) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda conv2d_transpose kernel is not initialized")
	}
	hasBias := 0
	if cfg.hasBias {
		hasBias = 1
	}
	var errStr *C.char
	if C.eosCudaLaunchConv2DTranspose(
		rt.ptr,
		kernel.ptr,
		C.uint(grid),
		C.uint(block),
		input,
		weight,
		bias,
		out0,
		C.int(elements),
		C.int(cfg.inChannels),
		C.int(cfg.inHeight),
		C.int(cfg.inWidth),
		C.int(cfg.outChannels),
		C.int(cfg.outHeight),
		C.int(cfg.outWidth),
		C.int(cfg.inPerGroup),
		C.int(cfg.outPerGroup),
		C.int(cfg.kernelH),
		C.int(cfg.kernelW),
		C.int(cfg.strideH),
		C.int(cfg.strideW),
		C.int(cfg.padH),
		C.int(cfg.padW),
		C.int(cfg.dilationH),
		C.int(cfg.dilationW),
		C.int(hasBias),
		&errStr,
	) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchTurboQEncode(kernel *auxKernel, grid, block uint, input, coords, norms, scratchWork, scratchRotated C.CUdeviceptr, spec *turboQDeviceSpec, vectors int, cfg cudaTurboQConfig) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda turboquant_encode kernel is not initialized")
	}
	if spec == nil {
		return fmt.Errorf("cuda turboquant spec is not initialized")
	}
	var errStr *C.char
	if C.eosCudaLaunchTurboQEncode(
		rt.ptr,
		kernel.ptr,
		C.uint(grid),
		C.uint(block),
		input,
		coords,
		norms,
		scratchWork,
		scratchRotated,
		spec.perm,
		spec.signs1,
		spec.signs2,
		spec.centroids,
		spec.boundaries,
		C.int(vectors),
		C.int(cfg.channels),
		C.int(cfg.height),
		C.int(cfg.width),
		C.int(1<<uint(cfg.bits)),
		&errStr,
	) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchTurboQDecode(kernel *auxKernel, grid, block uint, coords, norms, out0, scratchWork, scratchRotated C.CUdeviceptr, spec *turboQDeviceSpec, vectors int, cfg cudaTurboQConfig) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda turboquant_decode kernel is not initialized")
	}
	if spec == nil {
		return fmt.Errorf("cuda turboquant spec is not initialized")
	}
	var errStr *C.char
	if C.eosCudaLaunchTurboQDecode(
		rt.ptr,
		kernel.ptr,
		C.uint(grid),
		C.uint(block),
		coords,
		norms,
		out0,
		scratchWork,
		scratchRotated,
		spec.perm,
		spec.signs1,
		spec.signs2,
		spec.centroids,
		C.int(vectors),
		C.int(cfg.channels),
		C.int(cfg.height),
		C.int(cfg.width),
		C.int(1<<uint(cfg.bits)),
		&errStr,
	) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchSparseAttention(kernel *auxKernel, grid, block uint, query, key, value, out0 C.CUdeviceptr, cfg cudaSparseAttentionConfig) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda sparse_attention kernel is not initialized")
	}
	var errStr *C.char
	if C.eosCudaLaunchSparseAttention(
		rt.ptr,
		kernel.ptr,
		C.uint(grid),
		C.uint(block),
		query,
		key,
		value,
		out0,
		C.int(cfg.rank),
		C.int(cfg.kvLayout),
		C.int(cfg.batches),
		C.int(cfg.queryLen),
		C.int(cfg.keyLen),
		C.int(cfg.queryDim),
		C.int(cfg.valueDim),
		C.int(cfg.topK),
		&errStr,
	) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchTurboSparseAttention(kernel *auxKernel, grid, block uint, query, keyCoords, keyNorms, valueCoords, valueNorms, out0, keyScratchWork, keyScratchDecoded, valueScratchWork, valueScratchDecoded C.CUdeviceptr, keySpec, valueSpec *turboQDeviceSpec, cfg cudaTurboSparseAttentionConfig) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda turbo_sparse_attention kernel is not initialized")
	}
	if keySpec == nil || valueSpec == nil {
		return fmt.Errorf("cuda turbo_sparse_attention specs are not initialized")
	}
	var errStr *C.char
	if C.eosCudaLaunchTurboSparseAttention(
		rt.ptr,
		kernel.ptr,
		C.uint(grid),
		C.uint(block),
		query,
		keyCoords,
		keyNorms,
		valueCoords,
		valueNorms,
		out0,
		keyScratchWork,
		keyScratchDecoded,
		valueScratchWork,
		valueScratchDecoded,
		keySpec.perm,
		keySpec.signs1,
		keySpec.signs2,
		keySpec.centroids,
		valueSpec.perm,
		valueSpec.signs1,
		valueSpec.signs2,
		valueSpec.centroids,
		C.int(cfg.sparse.rank),
		C.int(cfg.sparse.batches),
		C.int(cfg.sparse.queryLen),
		C.int(cfg.sparse.keyLen),
		C.int(cfg.sparse.queryDim),
		C.int(cfg.sparse.valueDim),
		C.int(cfg.sparse.topK),
		C.int(1<<uint(cfg.key.bits)),
		C.int(1<<uint(cfg.value.bits)),
		C.int(cfg.routeBlockSize),
		C.int(cfg.routeTopBlocks),
		&errStr,
	) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchMSEPartials(kernel *auxKernel, grid, block uint, lhs, rhs, partials C.CUdeviceptr, elements int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda mse_loss kernel is not initialized")
	}
	var errStr *C.char
	if C.eosCudaLaunchMSEPartials(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), lhs, rhs, partials, C.int(elements), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchMSSSIMPartials(kernel *auxKernel, grid, block uint, lhs, rhs, partials C.CUdeviceptr, elements int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda ms_ssim_loss kernel is not initialized")
	}
	var errStr *C.char
	if C.eosCudaLaunchMSSSIMPartials(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), lhs, rhs, partials, C.int(elements), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchScalarSum(kernel *auxKernel, grid, block uint, values, out0 C.CUdeviceptr, count int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda scalar_add kernel is not initialized")
	}
	var errStr *C.char
	if C.eosCudaLaunchScalarSum(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), values, out0, C.int(count), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchRDLoss(kernel *auxKernel, grid, block uint, distortion, rate, out0 C.CUdeviceptr, lambda float32) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda rate_distortion_loss kernel is not initialized")
	}
	var errStr *C.char
	if C.eosCudaLaunchRDLoss(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), distortion, rate, out0, C.float(lambda), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchCrossEntropyPartials(kernel *auxKernel, grid, block uint, codes, logits, partials C.CUdeviceptr, elements, mode, layout, levels, bits, codeRank, codeN, codeC, codeH, codeW, logitsLen, logitsN, logitsC, logitsH, logitsW, sigmaMode int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda cross_entropy kernel is not initialized")
	}
	var errStr *C.char
	if C.eosCudaLaunchCrossEntropyPartials(
		rt.ptr,
		kernel.ptr,
		C.uint(grid),
		C.uint(block),
		codes,
		logits,
		partials,
		C.int(elements),
		C.int(mode),
		C.int(layout),
		C.int(levels),
		C.int(bits),
		C.int(codeRank),
		C.int(codeN),
		C.int(codeC),
		C.int(codeH),
		C.int(codeW),
		C.int(logitsLen),
		C.int(logitsN),
		C.int(logitsC),
		C.int(logitsH),
		C.int(logitsW),
		C.int(sigmaMode),
		&errStr,
	) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchUnary(kernel *C.EosCudaKernel, grid, block C.uint, in0, out0 C.CUdeviceptr, elements C.int) error {
	var errStr *C.char
	if C.eosCudaLaunchUnary(rt.ptr, kernel, grid, block, in0, out0, elements, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchScore(kernel *C.EosCudaKernel, grid, block C.uint, query, docs, out0 C.CUdeviceptr, rows, cols C.int) error {
	var errStr *C.char
	if C.eosCudaLaunchScore(rt.ptr, kernel, grid, block, query, docs, out0, rows, cols, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchOptimizerUpdate(kernel *auxKernel, grid, block uint, param, mom1, mom2, grad C.CUdeviceptr, elements, mode int, learningRate, weightDecay, beta1, beta2, corr1, corr2, epsilon, scale float32) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda optimizer kernel is not initialized")
	}
	var errStr *C.char
	if C.eosCudaLaunchOptimizerUpdate(
		rt.ptr,
		kernel.ptr,
		C.uint(grid),
		C.uint(block),
		param,
		mom1,
		mom2,
		grad,
		C.int(elements),
		C.int(mode),
		C.float(learningRate),
		C.float(weightDecay),
		C.float(beta1),
		C.float(beta2),
		C.float(corr1),
		C.float(corr2),
		C.float(epsilon),
		C.float(scale),
		&errStr,
	) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchAuxSoftmaxBackwardRows(kernel *auxKernel, grid, block uint, gradOut, probs, out0 C.CUdeviceptr, rows, cols int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda auxiliary softmax backward kernel is not initialized")
	}
	var errStr *C.char
	if C.eosCudaLaunchSoftmaxBackwardRows(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), gradOut, probs, out0, C.int(rows), C.int(cols), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

// launchAuxSoftmaxForwardRows runs the on-device forward row softmax in place
// over a rows×cols buffer on rt's stream (capturable).
func (rt *deviceRuntime) launchAuxSoftmaxForwardRows(kernel *auxKernel, grid, block uint, data C.CUdeviceptr, rows, cols int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda auxiliary softmax forward kernel is not initialized")
	}
	var errStr *C.char
	if C.eosCudaLaunchSoftmaxForwardRows(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), data, C.int(rows), C.int(cols), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

// launchAuxGeluForward runs the on-device forward GELU (out = gelu(src)) on rt's stream.
func (rt *deviceRuntime) launchAuxGeluForward(kernel *auxKernel, grid, block uint, src, dst C.CUdeviceptr, elements int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda auxiliary gelu forward kernel is not initialized")
	}
	var errStr *C.char
	if C.eosCudaLaunchGeluForward(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), src, dst, C.int(elements), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

// launchAuxLayerNormForwardRows runs the on-device forward row layernorm on rt's stream.
func (rt *deviceRuntime) launchAuxLayerNormForwardRows(kernel *auxKernel, grid, block uint, src, dst C.CUdeviceptr, rows, cols int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda auxiliary layernorm forward kernel is not initialized")
	}
	var errStr *C.char
	if C.eosCudaLaunchLayerNormForwardRows(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), src, dst, C.int(rows), C.int(cols), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

// launchAuxResidualAdd runs the on-device element-wise residual add (out = a+b) on rt's stream.
func (rt *deviceRuntime) launchAuxResidualAdd(kernel *auxKernel, grid, block uint, a, b, out0 C.CUdeviceptr, elements int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda auxiliary residual add kernel is not initialized")
	}
	var errStr *C.char
	if C.eosCudaLaunchResidualAdd(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), a, b, out0, C.int(elements), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchAuxLayerNormBackwardRows(kernel *auxKernel, grid, block uint, gradOut, normalized, pre, out0 C.CUdeviceptr, rows, cols int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda auxiliary layernorm backward kernel is not initialized")
	}
	var errStr *C.char
	if C.eosCudaLaunchLayerNormBackwardRows(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), gradOut, normalized, pre, out0, C.int(rows), C.int(cols), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchAuxContrastiveScores(kernel *auxKernel, grid, block uint, query, positive, queryNorms, positiveNorms, scores C.CUdeviceptr, queryRows, candidateRows, width int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda auxiliary contrastive score kernel is not initialized")
	}
	var errStr *C.char
	if C.eosCudaLaunchContrastiveScores(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), query, positive, queryNorms, positiveNorms, scores, C.int(queryRows), C.int(candidateRows), C.int(width), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchAuxInfoNCEScales(kernel *auxKernel, grid, block uint, scores, targetIndexes, scales, rowLoss, rowScore C.CUdeviceptr, queryRows, candidateRows int, temperature float32) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda auxiliary infonce scale kernel is not initialized")
	}
	var errStr *C.char
	if C.eosCudaLaunchInfoNCEScales(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), scores, targetIndexes, scales, rowLoss, rowScore, C.int(queryRows), C.int(candidateRows), C.float(temperature), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) launchAuxContrastiveGrad(kernel *auxKernel, grid, block uint, query, positive, queryNorms, positiveNorms, scores, scales, queryGrads, positiveGrads C.CUdeviceptr, queryRows, candidateRows, width int) error {
	if kernel == nil || kernel.ptr == nil {
		return fmt.Errorf("cuda auxiliary contrastive grad kernel is not initialized")
	}
	var errStr *C.char
	if C.eosCudaLaunchContrastiveGrad(rt.ptr, kernel.ptr, C.uint(grid), C.uint(block), query, positive, queryNorms, positiveNorms, scores, scales, queryGrads, positiveGrads, C.int(queryRows), C.int(candidateRows), C.int(width), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) matMulCublas(lhs, rhs, out0 C.CUdeviceptr, lhsRows, lhsCols, rhsRows, rhsCols C.int, transposeLeft, transposeRight bool) error {
	return rt.matMulCublasWithBeta(lhs, rhs, out0, lhsRows, lhsCols, rhsRows, rhsCols, transposeLeft, transposeRight, 0)
}

func (rt *deviceRuntime) matMulCublasWithBeta(lhs, rhs, out0 C.CUdeviceptr, lhsRows, lhsCols, rhsRows, rhsCols C.int, transposeLeft, transposeRight bool, beta float32) error {
	var errStr *C.char
	var left C.int
	var right C.int
	if transposeLeft {
		left = 1
	}
	if transposeRight {
		right = 1
	}
	if C.eosCudaMatMulCublasWithBeta(rt.ptr, lhs, rhs, out0, lhsRows, lhsCols, rhsRows, rhsCols, left, right, C.float(beta), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) matMulCublasWithBetaNoSync(lhs, rhs, out0 C.CUdeviceptr, lhsRows, lhsCols, rhsRows, rhsCols C.int, transposeLeft, transposeRight bool, beta float32) error {
	var errStr *C.char
	var left C.int
	var right C.int
	if transposeLeft {
		left = 1
	}
	if transposeRight {
		right = 1
	}
	if C.eosCudaMatMulCublasWithBetaNoSync(rt.ptr, lhs, rhs, out0, lhsRows, lhsCols, rhsRows, rhsCols, left, right, C.float(beta), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) matMulCublasStridedBatched(lhs, rhs, out0 C.CUdeviceptr, batches, lhsRows, lhsCols, rhsRows, rhsCols C.int, transposeLeft, transposeRight bool) error {
	var errStr *C.char
	var left C.int
	var right C.int
	if transposeLeft {
		left = 1
	}
	if transposeRight {
		right = 1
	}
	if C.eosCudaMatMulCublasStridedBatched(rt.ptr, lhs, rhs, out0, batches, lhsRows, lhsCols, rhsRows, rhsCols, left, right, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func cStringValue(value *C.char) string {
	if value == nil {
		return ""
	}
	out := C.GoString(value)
	C.eosCudaFreeCString(value)
	return out
}

// eosCudaGraphEnabled gates CUDA-graph capture/replay of stable, repeated
// GEMM regions. Read once from EOS_CUDA_GRAPH=1 so the default path is
// unchanged. Capture/replay requires stable device pointers (resident weights
// + named scratch); see project_manta_cuda_graph.
var eosCudaGraphEnabled = os.Getenv("EOS_CUDA_GRAPH") == "1"

// cudaGraph wraps an instantiated, replayable CUDA graph.
type cudaGraph struct {
	ptr *C.EosCudaGraph
}

// beginCapture puts rt's stream into capture mode. Warm up cuBLAS workspace
// for every GEMM shape BEFORE calling this — allocations are illegal during
// capture.
func (rt *deviceRuntime) beginCapture() error {
	var errStr *C.char
	if C.eosCudaBeginCapture(rt.ptr, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

// endCapture ends capture and instantiates the recorded graph.
func (rt *deviceRuntime) endCapture() (*cudaGraph, error) {
	var g *C.EosCudaGraph
	var errStr *C.char
	if C.eosCudaEndCapture(rt.ptr, &g, &errStr) != 0 {
		return nil, cStringError(errStr)
	}
	return &cudaGraph{ptr: g}, nil
}

// launchGraph replays a captured graph and synchronizes once.
func (rt *deviceRuntime) launchGraph(g *cudaGraph) error {
	if g == nil || g.ptr == nil {
		return errors.New("cuda: launch of nil graph")
	}
	var errStr *C.char
	if C.eosCudaGraphLaunch(rt.ptr, g.ptr, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (g *cudaGraph) destroy() {
	if g == nil || g.ptr == nil {
		return
	}
	C.eosCudaGraphDestroy(g.ptr)
	g.ptr = nil
}

func (rt *deviceRuntime) runBERTEmbeddingAffineLayerNorm(tokenEmbeddings, positionEmbeddings, tokenTypeEmbeddings, gamma, beta, inputIDs, tokenTypeIDs *backend.Tensor, epsilon float64) (*backend.Tensor, error) {
	if rt == nil || rt.ptr == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if tokenEmbeddings == nil || positionEmbeddings == nil || tokenTypeEmbeddings == nil || gamma == nil || beta == nil || inputIDs == nil || tokenTypeIDs == nil {
		return nil, fmt.Errorf("cuda bert embedding layernorm expects non-nil tensors")
	}
	if len(inputIDs.Shape) != 2 || !inputIDs.EqualShape(tokenTypeIDs) || inputIDs.DType != "i32" || tokenTypeIDs.DType != "i32" {
		return nil, fmt.Errorf("cuda bert embedding layernorm expects i32 input_ids/token_type_ids with matching [B,T] shape")
	}
	if len(tokenEmbeddings.Shape) != 2 || len(positionEmbeddings.Shape) != 2 || len(tokenTypeEmbeddings.Shape) != 2 {
		return nil, fmt.Errorf("cuda bert embedding layernorm expects rank-2 embedding tables")
	}
	for _, item := range []struct {
		name   string
		tensor *backend.Tensor
	}{
		{"cuda bert embedding layernorm token_embeddings", tokenEmbeddings},
		{"cuda bert embedding layernorm position_embeddings", positionEmbeddings},
		{"cuda bert embedding layernorm token_type_embeddings", tokenTypeEmbeddings},
		{"cuda bert embedding layernorm gamma", gamma},
		{"cuda bert embedding layernorm beta", beta},
	} {
		if err := validateCUDAF32CompatibleDType(item.tensor, item.name); err != nil {
			return nil, err
		}
	}
	batch, tokens := inputIDs.Shape[0], inputIDs.Shape[1]
	vocab, hidden := tokenEmbeddings.Shape[0], tokenEmbeddings.Shape[1]
	if batch <= 0 || tokens <= 0 || vocab <= 0 || hidden <= 0 {
		return nil, fmt.Errorf("cuda bert embedding layernorm shape must be positive")
	}
	if positionEmbeddings.Shape[1] != hidden || tokenTypeEmbeddings.Shape[1] != hidden || len(gamma.Shape) != 1 || gamma.Shape[0] != hidden || len(beta.Shape) != 1 || beta.Shape[0] != hidden {
		return nil, fmt.Errorf("cuda bert embedding layernorm hidden size mismatch")
	}
	rows, err := checkedProduct("cuda bert embedding layernorm rows", batch, tokens)
	if err != nil {
		return nil, err
	}
	outElements, err := checkedProduct("cuda bert embedding layernorm output", rows, hidden)
	if err != nil {
		return nil, err
	}
	for _, guard := range []struct {
		label string
		dims  []int
	}{
		{"cuda bert embedding layernorm row-hidden offset", []int{rows, hidden}},
		{"cuda bert embedding layernorm token embedding offset", []int{vocab, hidden}},
		{"cuda bert embedding layernorm position embedding offset", []int{positionEmbeddings.Shape[0], hidden}},
		{"cuda bert embedding layernorm token type embedding offset", []int{tokenTypeEmbeddings.Shape[0], hidden}},
	} {
		if err := checkedInt32Product(guard.label, guard.dims...); err != nil {
			return nil, err
		}
	}
	grid, block, err := checkedLaunch1D("cuda bert embedding layernorm", rows, 128)
	if err != nil {
		return nil, err
	}
	rowsArg, err := checkedCInt("cuda bert embedding layernorm rows", rows)
	if err != nil {
		return nil, err
	}
	tokensArg, err := checkedCInt("cuda bert embedding layernorm tokens", tokens)
	if err != nil {
		return nil, err
	}
	hiddenArg, err := checkedCInt("cuda bert embedding layernorm hidden", hidden)
	if err != nil {
		return nil, err
	}
	vocabArg, err := checkedCInt("cuda bert embedding layernorm vocab", vocab)
	if err != nil {
		return nil, err
	}
	positionsArg, err := checkedCInt("cuda bert embedding layernorm max positions", positionEmbeddings.Shape[0])
	if err != nil {
		return nil, err
	}
	typeVocabArg, err := checkedCInt("cuda bert embedding layernorm type vocab", tokenTypeEmbeddings.Shape[0])
	if err != nil {
		return nil, err
	}
	if inputIDs.Elements() != len(inputIDs.I32) || tokenTypeIDs.Elements() != len(tokenTypeIDs.I32) ||
		tokenEmbeddings.Elements() != len(tokenEmbeddings.F32) || positionEmbeddings.Elements() != len(positionEmbeddings.F32) ||
		tokenTypeEmbeddings.Elements() != len(tokenTypeEmbeddings.F32) || gamma.Elements() != len(gamma.F32) || beta.Elements() != len(beta.F32) {
		return nil, fmt.Errorf("cuda bert embedding layernorm tensor backing length mismatch")
	}
	if rt.bertEmbeddingKernel == nil {
		kernel, err := rt.compileAuxKernel(bertEmbeddingAffineLayerNormKernelSource, "manta_bert_embedding_affine_layernorm")
		if err != nil {
			return nil, err
		}
		rt.bertEmbeddingKernel = kernel
	}
	tokenEmbBuf, err := rt.uploadFloat32(tokenEmbeddings.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(tokenEmbBuf)
	positionEmbBuf, err := rt.uploadFloat32(positionEmbeddings.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(positionEmbBuf)
	tokenTypeEmbBuf, err := rt.uploadFloat32(tokenTypeEmbeddings.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(tokenTypeEmbBuf)
	gammaBuf, err := rt.uploadFloat32(gamma.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(gammaBuf)
	betaBuf, err := rt.uploadFloat32(beta.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(betaBuf)
	inputBuf, err := rt.uploadInt32(inputIDs.I32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(inputBuf)
	tokenTypeBuf, err := rt.uploadInt32(tokenTypeIDs.I32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(tokenTypeBuf)
	outBuf, err := rt.allocFloat32(outElements)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(outBuf)
	statusBuf, err := rt.uploadInt32([]int32{0})
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(statusBuf)
	var errStr *C.char
	if C.eosCudaLaunchBERTEmbeddingAffineLayerNorm(rt.ptr, rt.bertEmbeddingKernel.ptr, grid, block, tokenEmbBuf, positionEmbBuf, tokenTypeEmbBuf, gammaBuf, betaBuf, inputBuf, tokenTypeBuf, outBuf, statusBuf, rowsArg, tokensArg, hiddenArg, vocabArg, positionsArg, typeVocabArg, C.double(epsilon), &errStr) != 0 {
		return nil, cStringError(errStr)
	}
	status := []int32{0}
	if err := rt.downloadInt32(status, statusBuf); err != nil {
		return nil, err
	}
	if status[0] != 0 {
		return nil, fmt.Errorf("cuda bert embedding layernorm input id out of range")
	}
	out := make([]float32, outElements)
	if err := rt.downloadFloat32(out, outBuf); err != nil {
		return nil, err
	}
	return backend.NewTensorF32([]int{batch, tokens, hidden}, out), nil
}

func (rt *deviceRuntime) runBERTExactGELU(src *backend.Tensor) (*backend.Tensor, error) {
	if rt == nil || rt.ptr == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if err := validateCUDAF32CompatibleDType(src, "cuda bert exact gelu src"); err != nil {
		return nil, err
	}
	elements, err := checkedProduct("cuda bert exact gelu elements", src.Shape...)
	if err != nil {
		return nil, err
	}
	if err := checkedInt32Product("cuda bert exact gelu element offset", elements); err != nil {
		return nil, err
	}
	elementsArg, err := checkedCInt("cuda bert exact gelu elements", elements)
	if err != nil {
		return nil, err
	}
	grid, block, err := checkedLaunch1D("cuda bert exact gelu", elements, 256)
	if err != nil {
		return nil, err
	}
	if elements != len(src.F32) {
		return nil, fmt.Errorf("cuda bert exact gelu src tensor backing length mismatch")
	}
	if rt.bertExactGELUKernel == nil {
		kernel, err := rt.compileAuxKernel(bertExactGELUKernelSource, "manta_bert_exact_gelu")
		if err != nil {
			return nil, err
		}
		rt.bertExactGELUKernel = kernel
	}
	srcBuf, err := rt.uploadFloat32(src.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(srcBuf)
	outBuf, err := rt.allocFloat32(elements)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(outBuf)
	var errStr *C.char
	if C.eosCudaLaunchBERTExactGELU(rt.ptr, rt.bertExactGELUKernel.ptr, grid, block, srcBuf, outBuf, elementsArg, &errStr) != 0 {
		return nil, cStringError(errStr)
	}
	out := make([]float32, elements)
	if err := rt.downloadFloat32(out, outBuf); err != nil {
		return nil, err
	}
	return backend.NewTensorF32(src.Shape, out), nil
}

func (rt *deviceRuntime) runBERTResidualAffineLayerNorm(src, residual, gamma, beta *backend.Tensor, epsilon float64) (*backend.Tensor, error) {
	if rt == nil || rt.ptr == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if src == nil || residual == nil || gamma == nil || beta == nil || !src.EqualShape(residual) || len(src.Shape) < 2 {
		return nil, fmt.Errorf("cuda bert residual layernorm expects matching tensors with row shape")
	}
	for _, item := range []struct {
		name   string
		tensor *backend.Tensor
	}{
		{"cuda bert residual layernorm src", src},
		{"cuda bert residual layernorm residual", residual},
		{"cuda bert residual layernorm gamma", gamma},
		{"cuda bert residual layernorm beta", beta},
	} {
		if err := validateCUDAF32CompatibleDType(item.tensor, item.name); err != nil {
			return nil, err
		}
	}
	hidden := src.Shape[len(src.Shape)-1]
	if hidden <= 0 {
		return nil, fmt.Errorf("cuda bert residual layernorm hidden size must be positive")
	}
	elements, err := checkedProduct("cuda bert residual layernorm elements", src.Shape...)
	if err != nil {
		return nil, err
	}
	if err := checkedInt32Product("cuda bert residual layernorm row-hidden offset", src.Shape...); err != nil {
		return nil, err
	}
	rows := elements / hidden
	if rows <= 0 || elements != len(src.F32) || residual.Elements() != len(residual.F32) ||
		len(gamma.Shape) != 1 || gamma.Shape[0] != hidden || gamma.Elements() != len(gamma.F32) ||
		len(beta.Shape) != 1 || beta.Shape[0] != hidden || beta.Elements() != len(beta.F32) {
		return nil, fmt.Errorf("cuda bert residual layernorm shape mismatch")
	}
	rowsArg, err := checkedCInt("cuda bert residual layernorm rows", rows)
	if err != nil {
		return nil, err
	}
	hiddenArg, err := checkedCInt("cuda bert residual layernorm hidden", hidden)
	if err != nil {
		return nil, err
	}
	grid, block, err := checkedLaunch1D("cuda bert residual layernorm", rows, 128)
	if err != nil {
		return nil, err
	}
	if rt.bertResidualLayerNormKernel == nil {
		kernel, err := rt.compileAuxKernel(bertResidualAffineLayerNormKernelSource, "manta_bert_residual_affine_layernorm")
		if err != nil {
			return nil, err
		}
		rt.bertResidualLayerNormKernel = kernel
	}
	srcBuf, err := rt.uploadFloat32(src.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(srcBuf)
	residualBuf, err := rt.uploadFloat32(residual.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(residualBuf)
	gammaBuf, err := rt.uploadFloat32(gamma.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(gammaBuf)
	betaBuf, err := rt.uploadFloat32(beta.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(betaBuf)
	outBuf, err := rt.allocFloat32(len(src.F32))
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(outBuf)
	var errStr *C.char
	if C.eosCudaLaunchBERTResidualAffineLayerNorm(rt.ptr, rt.bertResidualLayerNormKernel.ptr, grid, block, srcBuf, residualBuf, gammaBuf, betaBuf, outBuf, rowsArg, hiddenArg, C.double(epsilon), &errStr) != 0 {
		return nil, cStringError(errStr)
	}
	out := make([]float32, len(src.F32))
	if err := rt.downloadFloat32(out, outBuf); err != nil {
		return nil, err
	}
	return backend.NewTensorF32(src.Shape, out), nil
}

func (rt *deviceRuntime) runBERTCLSL2(hiddenStates *backend.Tensor) (*backend.Tensor, error) {
	if rt == nil || rt.ptr == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if hiddenStates == nil || (hiddenStates.DType != "f32" && hiddenStates.DType != "f16") || len(hiddenStates.Shape) != 3 || hiddenStates.Elements() != len(hiddenStates.F32) {
		return nil, fmt.Errorf("cuda bert cls l2 expects f32-compatible hidden_states [B,T,H]")
	}
	batch, tokens, hidden := hiddenStates.Shape[0], hiddenStates.Shape[1], hiddenStates.Shape[2]
	if batch <= 0 || tokens <= 0 || hidden <= 0 {
		return nil, fmt.Errorf("cuda bert cls l2 shape must be positive")
	}
	outElements, err := checkedProduct("cuda bert cls l2 output", batch, hidden)
	if err != nil {
		return nil, err
	}
	if err := checkedInt32Product("cuda bert cls l2 hidden state offset", batch, tokens, hidden); err != nil {
		return nil, err
	}
	if err := checkedInt32Product("cuda bert cls l2 output offset", batch, hidden); err != nil {
		return nil, err
	}
	batchArg, err := checkedCInt("cuda bert cls l2 batch", batch)
	if err != nil {
		return nil, err
	}
	tokensArg, err := checkedCInt("cuda bert cls l2 tokens", tokens)
	if err != nil {
		return nil, err
	}
	hiddenArg, err := checkedCInt("cuda bert cls l2 hidden", hidden)
	if err != nil {
		return nil, err
	}
	grid, block, err := checkedLaunch1D("cuda bert cls l2", batch, 128)
	if err != nil {
		return nil, err
	}
	if rt.bertCLSL2Kernel == nil {
		kernel, err := rt.compileAuxKernel(bertCLSL2KernelSource, "manta_bert_cls_l2")
		if err != nil {
			return nil, err
		}
		rt.bertCLSL2Kernel = kernel
	}
	hiddenBuf, err := rt.uploadFloat32(hiddenStates.F32)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(hiddenBuf)
	outBuf, err := rt.allocFloat32(outElements)
	if err != nil {
		return nil, err
	}
	defer rt.freeBuffer(outBuf)
	var errStr *C.char
	if C.eosCudaLaunchBERTCLSL2(rt.ptr, rt.bertCLSL2Kernel.ptr, grid, block, hiddenBuf, outBuf, batchArg, tokensArg, hiddenArg, &errStr) != 0 {
		return nil, cStringError(errStr)
	}
	out := make([]float32, outElements)
	if err := rt.downloadFloat32(out, outBuf); err != nil {
		return nil, err
	}
	return backend.NewTensorF32([]int{batch, hidden}, out), nil
}

func (rt *deviceRuntime) runBERTCLSL2Device(hiddenStates, out C.CUdeviceptr, batch, tokens, hidden int) error {
	if rt == nil || rt.ptr == nil {
		return fmt.Errorf("cuda runtime is not initialized")
	}
	if batch <= 0 || tokens <= 0 || hidden <= 0 {
		return fmt.Errorf("cuda bert cls l2 device shape must be positive")
	}
	if err := checkedInt32Product("cuda bert cls l2 device hidden state offset", batch, tokens, hidden); err != nil {
		return err
	}
	if err := checkedInt32Product("cuda bert cls l2 device output offset", batch, hidden); err != nil {
		return err
	}
	batchArg, err := checkedCInt("cuda bert cls l2 device batch", batch)
	if err != nil {
		return err
	}
	tokensArg, err := checkedCInt("cuda bert cls l2 device tokens", tokens)
	if err != nil {
		return err
	}
	hiddenArg, err := checkedCInt("cuda bert cls l2 device hidden", hidden)
	if err != nil {
		return err
	}
	grid, block, err := checkedLaunch1D("cuda bert cls l2 device", batch, 128)
	if err != nil {
		return err
	}
	if rt.bertCLSL2Kernel == nil {
		kernel, err := rt.compileAuxKernel(bertCLSL2KernelSource, "manta_bert_cls_l2")
		if err != nil {
			return err
		}
		rt.bertCLSL2Kernel = kernel
	}
	var errStr *C.char
	if C.eosCudaLaunchBERTCLSL2(rt.ptr, rt.bertCLSL2Kernel.ptr, grid, block, hiddenStates, out, batchArg, tokensArg, hiddenArg, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

type bertCUDAOneLayerTransferStats struct {
	UploadedBytes                 int64
	DownloadedBytes               int64
	FinalDownloadedBytes          int64
	StatusDownloadedBytes         int64
	IntermediateDownloadedBytes   int64
	ResidentWeightBytesReferenced int64
}

type bertCUDAOneLayerPreflight struct {
	batch                         int
	tokens                        int
	rows                          int
	hidden                        int
	intermediate                  int
	hiddenElements                int
	intermediateElements          int
	residentWeightBytesReferenced int64
}

func (rt *deviceRuntime) bindBERTResidentTensor(name string, tensor *backend.Tensor) error {
	if rt == nil || rt.ptr == nil {
		return fmt.Errorf("cuda runtime is not initialized")
	}
	if name == "" {
		return fmt.Errorf("cuda bert resident tensor name is required")
	}
	if err := validateCUDAF32CompatibleTensor(tensor, "cuda bert resident tensor "+name); err != nil {
		return err
	}
	elements := len(tensor.F32)
	if resident, ok := rt.bertResidentTensors[name]; ok {
		if resident.elements == elements {
			if err := rt.copyFloat32ToBuffer(resident.ptr, tensor.F32); err != nil {
				return err
			}
			resident.shape = append(resident.shape[:0], tensor.Shape...)
			rt.bertResidentTensors[name] = resident
			return nil
		}
		_ = rt.freeBuffer(resident.ptr)
		delete(rt.bertResidentTensors, name)
	}
	ptr, err := rt.uploadFloat32(tensor.F32)
	if err != nil {
		return err
	}
	rt.bertResidentTensors[name] = residentTensor{ptr: ptr, shape: append([]int(nil), tensor.Shape...), elements: elements}
	return nil
}

func (rt *deviceRuntime) residentBERTTensor(name string) (residentTensor, error) {
	resident, ok := rt.bertResidentTensors[name]
	if !ok {
		return residentTensor{}, fmt.Errorf("cuda bert resident tensor %q is not bound", name)
	}
	return resident, nil
}

func (rt *deviceRuntime) ensureBERTBiasAddKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.bertBiasAddKernel != nil {
		return rt.bertBiasAddKernel, nil
	}
	kernel, err := rt.compileAuxKernel(bertBiasAddKernelSource, "manta_bert_bias_add")
	if err != nil {
		return nil, err
	}
	rt.bertBiasAddKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) ensureBERTAttentionContextKernel() (*auxKernel, error) {
	if rt == nil {
		return nil, fmt.Errorf("cuda runtime is not initialized")
	}
	if rt.bertAttentionContextKernel != nil {
		return rt.bertAttentionContextKernel, nil
	}
	kernel, err := rt.compileAuxKernel(bertAttentionContextKernelSource, "manta_bert_attention_context")
	if err != nil {
		return nil, err
	}
	rt.bertAttentionContextKernel = kernel
	return kernel, nil
}

func (rt *deviceRuntime) runBERTEmbeddingAffineLayerNormDevice(tokenEmbeddingsName, positionEmbeddingsName, tokenTypeEmbeddingsName, gammaName, betaName string, inputIDs, tokenTypeIDs, outBuf, statusBuf C.CUdeviceptr, batch, tokens int, epsilon float64) error {
	tokenEmbeddings, err := rt.residentBERTTensor(tokenEmbeddingsName)
	if err != nil {
		return err
	}
	positionEmbeddings, err := rt.residentBERTTensor(positionEmbeddingsName)
	if err != nil {
		return err
	}
	tokenTypeEmbeddings, err := rt.residentBERTTensor(tokenTypeEmbeddingsName)
	if err != nil {
		return err
	}
	gamma, err := rt.residentBERTTensor(gammaName)
	if err != nil {
		return err
	}
	beta, err := rt.residentBERTTensor(betaName)
	if err != nil {
		return err
	}
	if len(tokenEmbeddings.shape) != 2 || len(positionEmbeddings.shape) != 2 || len(tokenTypeEmbeddings.shape) != 2 || len(gamma.shape) != 1 || len(beta.shape) != 1 {
		return fmt.Errorf("cuda bert embedding resident tensors have invalid rank")
	}
	hidden := tokenEmbeddings.shape[1]
	if hidden <= 0 || positionEmbeddings.shape[1] != hidden || tokenTypeEmbeddings.shape[1] != hidden || gamma.shape[0] != hidden || beta.shape[0] != hidden {
		return fmt.Errorf("cuda bert embedding resident hidden size mismatch")
	}
	rows, err := checkedProduct("cuda bert embedding device rows", batch, tokens)
	if err != nil {
		return err
	}
	for _, guard := range []struct {
		label string
		dims  []int
	}{
		{"cuda bert embedding device row offset", []int{batch, tokens}},
		{"cuda bert embedding device output offset", []int{rows, hidden}},
		{"cuda bert embedding device token embedding offset", []int{tokenEmbeddings.shape[0], hidden}},
		{"cuda bert embedding device position embedding offset", []int{positionEmbeddings.shape[0], hidden}},
		{"cuda bert embedding device type embedding offset", []int{tokenTypeEmbeddings.shape[0], hidden}},
	} {
		if err := checkedInt32Product(guard.label, guard.dims...); err != nil {
			return err
		}
	}
	grid, block, err := checkedLaunch1D("cuda bert embedding device", rows, 128)
	if err != nil {
		return err
	}
	rowsArg, err := checkedCInt("cuda bert embedding device rows", rows)
	if err != nil {
		return err
	}
	tokensArg, err := checkedCInt("cuda bert embedding device tokens", tokens)
	if err != nil {
		return err
	}
	hiddenArg, err := checkedCInt("cuda bert embedding device hidden", hidden)
	if err != nil {
		return err
	}
	vocabArg, err := checkedCInt("cuda bert embedding device vocab", tokenEmbeddings.shape[0])
	if err != nil {
		return err
	}
	positionsArg, err := checkedCInt("cuda bert embedding device max positions", positionEmbeddings.shape[0])
	if err != nil {
		return err
	}
	typeVocabArg, err := checkedCInt("cuda bert embedding device type vocab", tokenTypeEmbeddings.shape[0])
	if err != nil {
		return err
	}
	if rt.bertEmbeddingKernel == nil {
		kernel, err := rt.compileAuxKernel(bertEmbeddingAffineLayerNormKernelSource, "manta_bert_embedding_affine_layernorm")
		if err != nil {
			return err
		}
		rt.bertEmbeddingKernel = kernel
	}
	var errStr *C.char
	if C.eosCudaLaunchBERTEmbeddingAffineLayerNorm(rt.ptr, rt.bertEmbeddingKernel.ptr, grid, block, tokenEmbeddings.ptr, positionEmbeddings.ptr, tokenTypeEmbeddings.ptr, gamma.ptr, beta.ptr, inputIDs, tokenTypeIDs, outBuf, statusBuf, rowsArg, tokensArg, hiddenArg, vocabArg, positionsArg, typeVocabArg, C.double(epsilon), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) runBERTBiasAddDevice(data C.CUdeviceptr, biasName string, rows, cols int) error {
	bias, err := rt.residentBERTTensor(biasName)
	if err != nil {
		return err
	}
	if len(bias.shape) != 1 || bias.shape[0] != cols {
		return fmt.Errorf("cuda bert bias %q shape %v does not match cols=%d", biasName, bias.shape, cols)
	}
	elements, err := checkedProduct("cuda bert bias add elements", rows, cols)
	if err != nil {
		return err
	}
	if err := checkedInt32Product("cuda bert bias add offset", rows, cols); err != nil {
		return err
	}
	grid, block, err := checkedLaunch1D("cuda bert bias add", elements, 256)
	if err != nil {
		return err
	}
	rowsArg, err := checkedCInt("cuda bert bias add rows", rows)
	if err != nil {
		return err
	}
	colsArg, err := checkedCInt("cuda bert bias add cols", cols)
	if err != nil {
		return err
	}
	kernel, err := rt.ensureBERTBiasAddKernel()
	if err != nil {
		return err
	}
	var errStr *C.char
	if C.eosCudaLaunchBERTBiasAdd(rt.ptr, kernel.ptr, grid, block, data, bias.ptr, rowsArg, colsArg, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) runBERTExactGELUDevice(src, dst C.CUdeviceptr, elements int) error {
	if err := checkedInt32Product("cuda bert exact gelu device element offset", elements); err != nil {
		return err
	}
	elementsArg, err := checkedCInt("cuda bert exact gelu device elements", elements)
	if err != nil {
		return err
	}
	grid, block, err := checkedLaunch1D("cuda bert exact gelu device", elements, 256)
	if err != nil {
		return err
	}
	if rt.bertExactGELUKernel == nil {
		kernel, err := rt.compileAuxKernel(bertExactGELUKernelSource, "manta_bert_exact_gelu")
		if err != nil {
			return err
		}
		rt.bertExactGELUKernel = kernel
	}
	var errStr *C.char
	if C.eosCudaLaunchBERTExactGELU(rt.ptr, rt.bertExactGELUKernel.ptr, grid, block, src, dst, elementsArg, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) runBERTResidualAffineLayerNormDevice(src, residual C.CUdeviceptr, gammaName, betaName string, out C.CUdeviceptr, rows, hidden int, epsilon float64) error {
	gamma, err := rt.residentBERTTensor(gammaName)
	if err != nil {
		return err
	}
	beta, err := rt.residentBERTTensor(betaName)
	if err != nil {
		return err
	}
	if len(gamma.shape) != 1 || gamma.shape[0] != hidden || len(beta.shape) != 1 || beta.shape[0] != hidden {
		return fmt.Errorf("cuda bert residual layernorm gamma/beta shape mismatch")
	}
	if err := checkedInt32Product("cuda bert residual layernorm device row-hidden offset", rows, hidden); err != nil {
		return err
	}
	grid, block, err := checkedLaunch1D("cuda bert residual layernorm device", rows, 128)
	if err != nil {
		return err
	}
	rowsArg, err := checkedCInt("cuda bert residual layernorm device rows", rows)
	if err != nil {
		return err
	}
	hiddenArg, err := checkedCInt("cuda bert residual layernorm device hidden", hidden)
	if err != nil {
		return err
	}
	if rt.bertResidualLayerNormKernel == nil {
		kernel, err := rt.compileAuxKernel(bertResidualAffineLayerNormKernelSource, "manta_bert_residual_affine_layernorm")
		if err != nil {
			return err
		}
		rt.bertResidualLayerNormKernel = kernel
	}
	var errStr *C.char
	if C.eosCudaLaunchBERTResidualAffineLayerNorm(rt.ptr, rt.bertResidualLayerNormKernel.ptr, grid, block, src, residual, gamma.ptr, beta.ptr, out, rowsArg, hiddenArg, C.double(epsilon), &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func (rt *deviceRuntime) runBERTAttentionContextDevice(query, key, value, attentionMask, out C.CUdeviceptr, batch, tokens, hidden, heads int) error {
	if heads <= 0 || hidden%heads != 0 {
		return fmt.Errorf("cuda bert attention hidden=%d must be divisible by heads=%d", hidden, heads)
	}
	if batch <= 0 {
		return fmt.Errorf("cuda bert attention batch=%d unsupported, want positive", batch)
	}
	if tokens <= 0 || tokens > 512 {
		return fmt.Errorf("cuda bert attention tokens=%d unsupported, want 1..512", tokens)
	}
	headDim := hidden / heads
	if headDim <= 0 || headDim > 32 {
		return fmt.Errorf("cuda bert attention head_dim=%d unsupported, want 1..32", headDim)
	}
	jobs, err := checkedProduct("cuda bert attention jobs", batch, tokens, heads)
	if err != nil {
		return err
	}
	for _, guard := range []struct {
		label string
		dims  []int
	}{
		{"cuda bert attention job offset", []int{batch, tokens, heads}},
		{"cuda bert attention hidden offset", []int{batch, tokens, hidden}},
		{"cuda bert attention head offset", []int{heads, headDim}},
	} {
		if err := checkedInt32Product(guard.label, guard.dims...); err != nil {
			return err
		}
	}
	grid, err := checkedCUint("cuda bert attention context grid", uint(jobs))
	if err != nil {
		return err
	}
	blockSize := bertAttentionContextBlockSize(tokens)
	block, err := checkedCUint("cuda bert attention context block", blockSize)
	if err != nil {
		return err
	}
	batchArg, err := checkedCInt("cuda bert attention batch", batch)
	if err != nil {
		return err
	}
	tokensArg, err := checkedCInt("cuda bert attention tokens", tokens)
	if err != nil {
		return err
	}
	hiddenArg, err := checkedCInt("cuda bert attention hidden", hidden)
	if err != nil {
		return err
	}
	headsArg, err := checkedCInt("cuda bert attention heads", heads)
	if err != nil {
		return err
	}
	headDimArg, err := checkedCInt("cuda bert attention head_dim", headDim)
	if err != nil {
		return err
	}
	kernel, err := rt.ensureBERTAttentionContextKernel()
	if err != nil {
		return err
	}
	var errStr *C.char
	if C.eosCudaLaunchBERTAttentionContext(rt.ptr, kernel.ptr, grid, block, query, key, value, attentionMask, out, batchArg, tokensArg, hiddenArg, headsArg, headDimArg, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func bertAttentionContextBlockSize(tokens int) uint {
	if tokens <= 256 {
		return 128
	}
	return 256
}

func (rt *deviceRuntime) runResidentBERTGEMM(lhs C.CUdeviceptr, lhsRows, lhsCols int, rightName string, out C.CUdeviceptr, transposeRight bool) error {
	resident, ok := rt.residentMatrices[rightName]
	if !ok {
		return fmt.Errorf("cuda bert GEMM resident matrix %q is not bound", rightName)
	}
	lhsRowsArg, err := checkedCInt("cuda bert GEMM lhs rows", lhsRows)
	if err != nil {
		return err
	}
	lhsColsArg, err := checkedCInt("cuda bert GEMM lhs cols", lhsCols)
	if err != nil {
		return err
	}
	rhsRowsArg, err := checkedCInt("cuda bert GEMM rhs rows", resident.rows)
	if err != nil {
		return err
	}
	rhsColsArg, err := checkedCInt("cuda bert GEMM rhs cols", resident.cols)
	if err != nil {
		return err
	}
	return rt.matMulCublasWithBetaNoSync(lhs, resident.ptr, out, lhsRowsArg, lhsColsArg, rhsRowsArg, rhsColsArg, false, transposeRight, 0)
}

func (rt *deviceRuntime) preflightBERTOneLayerResidentFixture(inputIDs, attentionMask, tokenTypeIDs *backend.Tensor, names map[string]string, heads int) (bertCUDAOneLayerPreflight, error) {
	var pre bertCUDAOneLayerPreflight
	if rt == nil || rt.ptr == nil {
		return pre, fmt.Errorf("cuda runtime is not initialized")
	}
	if inputIDs == nil || attentionMask == nil || tokenTypeIDs == nil || inputIDs.DType != "i32" || attentionMask.DType != "i32" || tokenTypeIDs.DType != "i32" || len(inputIDs.Shape) != 2 || !inputIDs.EqualShape(attentionMask) || !inputIDs.EqualShape(tokenTypeIDs) {
		return pre, fmt.Errorf("cuda bert one-layer fixture expects matching i32 [B,T] input tensors")
	}
	batch, tokens := inputIDs.Shape[0], inputIDs.Shape[1]
	if batch <= 0 || tokens <= 0 || inputIDs.Elements() != len(inputIDs.I32) || attentionMask.Elements() != len(attentionMask.I32) || tokenTypeIDs.Elements() != len(tokenTypeIDs.I32) {
		return pre, fmt.Errorf("cuda bert one-layer fixture input backing length mismatch")
	}
	rows, err := checkedProduct("cuda bert one-layer rows", batch, tokens)
	if err != nil {
		return pre, err
	}
	if err := checkedInt32Product("cuda bert one-layer input row offset", batch, tokens); err != nil {
		return pre, err
	}
	if _, _, err := checkedLaunch1D("cuda bert one-layer preflight rows", rows, 128); err != nil {
		return pre, err
	}

	requiredTensor := []string{
		"token_embeddings", "position_embeddings", "token_type_embeddings", "embedding_layernorm_weight", "embedding_layernorm_bias",
		"attention_query_bias", "attention_key_bias", "attention_value_bias", "attention_output_bias",
		"attention_layernorm_weight", "attention_layernorm_bias", "intermediate_bias", "output_bias", "output_layernorm_weight", "output_layernorm_bias",
	}
	requiredMatrix := []string{
		"attention_query_weight", "attention_key_weight", "attention_value_weight", "attention_output_weight", "intermediate_weight", "output_weight",
	}
	seenNames := map[string]string{}
	requireName := func(slot string) (string, error) {
		name := names[slot]
		if name == "" {
			return "", fmt.Errorf("cuda bert one-layer fixture missing resident name for %s", slot)
		}
		if previous, ok := seenNames[name]; ok {
			return "", fmt.Errorf("cuda bert one-layer fixture resident name %q is duplicated for %s and %s", name, previous, slot)
		}
		seenNames[name] = slot
		return name, nil
	}
	tensors := map[string]residentTensor{}
	for _, slot := range requiredTensor {
		name, err := requireName(slot)
		if err != nil {
			return pre, err
		}
		resident, ok := rt.bertResidentTensors[name]
		if !ok {
			if _, wrongKind := rt.residentMatrices[name]; wrongKind {
				return pre, fmt.Errorf("cuda bert one-layer fixture %s resident name %q is bound as matrix, want tensor", slot, name)
			}
			return pre, fmt.Errorf("cuda bert one-layer fixture %s resident name %q is not bound", slot, name)
		}
		if err := validateResidentTensorShapeProduct("cuda bert one-layer fixture "+slot, resident); err != nil {
			return pre, err
		}
		tensors[slot] = resident
		pre.residentWeightBytesReferenced += int64(resident.elements * 4)
	}
	matrices := map[string]residentMatrix{}
	for _, slot := range requiredMatrix {
		name, err := requireName(slot)
		if err != nil {
			return pre, err
		}
		resident, ok := rt.residentMatrices[name]
		if !ok {
			if _, wrongKind := rt.bertResidentTensors[name]; wrongKind {
				return pre, fmt.Errorf("cuda bert one-layer fixture %s resident name %q is bound as tensor, want matrix", slot, name)
			}
			return pre, fmt.Errorf("cuda bert one-layer fixture %s resident name %q is not bound", slot, name)
		}
		if err := validateResidentMatrixShapeProduct("cuda bert one-layer fixture "+slot, resident); err != nil {
			return pre, err
		}
		matrices[slot] = resident
		pre.residentWeightBytesReferenced += int64(resident.elements * 4)
	}

	tokenEmb := tensors["token_embeddings"]
	positionEmb := tensors["position_embeddings"]
	typeEmb := tensors["token_type_embeddings"]
	if len(tokenEmb.shape) != 2 || len(positionEmb.shape) != 2 || len(typeEmb.shape) != 2 {
		return pre, fmt.Errorf("cuda bert one-layer fixture embedding tensors must be rank-2")
	}
	hidden := tokenEmb.shape[1]
	if hidden <= 0 || positionEmb.shape[1] != hidden || typeEmb.shape[1] != hidden || positionEmb.shape[0] < tokens {
		return pre, fmt.Errorf("cuda bert one-layer fixture embedding shape mismatch token=%v position=%v type=%v tokens=%d", tokenEmb.shape, positionEmb.shape, typeEmb.shape, tokens)
	}
	if heads <= 0 || hidden%heads != 0 {
		return pre, fmt.Errorf("cuda bert one-layer fixture hidden=%d must be divisible by heads=%d", hidden, heads)
	}
	if typeEmb.shape[0] <= 0 || tokenEmb.shape[0] <= 0 {
		return pre, fmt.Errorf("cuda bert one-layer fixture embedding vocab sizes must be positive")
	}

	vectorShape := func(slot string, want int) error {
		resident := tensors[slot]
		if len(resident.shape) != 1 || resident.shape[0] != want || resident.elements != want {
			return fmt.Errorf("cuda bert one-layer fixture %s shape %v elements=%d, want [%d]", slot, resident.shape, resident.elements, want)
		}
		return nil
	}
	for _, slot := range []string{"embedding_layernorm_weight", "embedding_layernorm_bias", "attention_query_bias", "attention_key_bias", "attention_value_bias", "attention_output_bias", "attention_layernorm_weight", "attention_layernorm_bias", "output_bias", "output_layernorm_weight", "output_layernorm_bias"} {
		if err := vectorShape(slot, hidden); err != nil {
			return pre, err
		}
	}
	for _, slot := range []string{"attention_query_weight", "attention_key_weight", "attention_value_weight", "attention_output_weight"} {
		resident := matrices[slot]
		if resident.rows != hidden || resident.cols != hidden {
			return pre, fmt.Errorf("cuda bert one-layer fixture %s shape [%d,%d], want [%d,%d]", slot, resident.rows, resident.cols, hidden, hidden)
		}
	}
	intermediate := matrices["intermediate_weight"].rows
	if intermediate <= 0 || matrices["intermediate_weight"].cols != hidden {
		return pre, fmt.Errorf("cuda bert one-layer fixture intermediate_weight shape [%d,%d], want [I,%d]", matrices["intermediate_weight"].rows, matrices["intermediate_weight"].cols, hidden)
	}
	if err := vectorShape("intermediate_bias", intermediate); err != nil {
		return pre, err
	}
	outputWeight := matrices["output_weight"]
	if outputWeight.rows != hidden || outputWeight.cols != intermediate {
		return pre, fmt.Errorf("cuda bert one-layer fixture output_weight shape [%d,%d], want [%d,%d]", outputWeight.rows, outputWeight.cols, hidden, intermediate)
	}

	for i, tokenID := range inputIDs.I32 {
		if tokenID < 0 || int(tokenID) >= tokenEmb.shape[0] {
			return pre, fmt.Errorf("cuda bert one-layer fixture input_ids[%d]=%d out of range [0,%d)", i, tokenID, tokenEmb.shape[0])
		}
		positionID := i % tokens
		if positionID < 0 || positionID >= positionEmb.shape[0] {
			return pre, fmt.Errorf("cuda bert one-layer fixture implicit position_ids[%d]=%d out of range [0,%d)", i, positionID, positionEmb.shape[0])
		}
		tokenTypeID := tokenTypeIDs.I32[i]
		if tokenTypeID < 0 || int(tokenTypeID) >= typeEmb.shape[0] {
			return pre, fmt.Errorf("cuda bert one-layer fixture token_type_ids[%d]=%d out of range [0,%d)", i, tokenTypeID, typeEmb.shape[0])
		}
		mask := attentionMask.I32[i]
		if mask != 0 && mask != 1 {
			return pre, fmt.Errorf("cuda bert one-layer fixture attention_mask[%d]=%d is invalid, want 0 or 1", i, mask)
		}
	}

	hiddenElements, err := checkedProduct("cuda bert one-layer hidden elements", rows, hidden)
	if err != nil {
		return pre, err
	}
	intermediateElements, err := checkedProduct("cuda bert one-layer intermediate elements", rows, intermediate)
	if err != nil {
		return pre, err
	}
	attentionJobs, err := checkedProduct("cuda bert one-layer attention jobs", batch, tokens, heads)
	if err != nil {
		return pre, err
	}
	for _, guard := range []struct {
		label string
		dims  []int
	}{
		{"cuda bert one-layer hidden offset", []int{rows, hidden}},
		{"cuda bert one-layer intermediate offset", []int{rows, intermediate}},
		{"cuda bert one-layer attention jobs", []int{batch, tokens, heads}},
		{"cuda bert one-layer token embedding offset", []int{tokenEmb.shape[0], hidden}},
		{"cuda bert one-layer position embedding offset", []int{positionEmb.shape[0], hidden}},
		{"cuda bert one-layer type embedding offset", []int{typeEmb.shape[0], hidden}},
	} {
		if err := checkedInt32Product(guard.label, guard.dims...); err != nil {
			return pre, err
		}
	}
	for _, launch := range []struct {
		label    string
		elements int
		block    uint
	}{
		{"cuda bert one-layer embedding launch", rows, 128},
		{"cuda bert one-layer qkv bias launch", hiddenElements, 256},
		{"cuda bert one-layer attention launch", attentionJobs, 128},
		{"cuda bert one-layer gelu launch", intermediateElements, 256},
		{"cuda bert one-layer residual layernorm launch", rows, 128},
	} {
		if _, _, err := checkedLaunch1D(launch.label, launch.elements, launch.block); err != nil {
			return pre, err
		}
	}
	if _, err := checkedCUDABytes("cuda bert one-layer input upload bytes", len(inputIDs.I32)+len(attentionMask.I32)+len(tokenTypeIDs.I32)+1, 4); err != nil {
		return pre, err
	}
	if _, err := checkedCUDABytes("cuda bert one-layer hidden buffer bytes", hiddenElements, 4); err != nil {
		return pre, err
	}
	if _, err := checkedCUDABytes("cuda bert one-layer intermediate buffer bytes", intermediateElements, 4); err != nil {
		return pre, err
	}

	pre.batch = batch
	pre.tokens = tokens
	pre.rows = rows
	pre.hidden = hidden
	pre.intermediate = intermediate
	pre.hiddenElements = hiddenElements
	pre.intermediateElements = intermediateElements
	return pre, nil
}

func validateResidentTensorShapeProduct(label string, resident residentTensor) error {
	elements, err := checkedProduct(label+" shape product", resident.shape...)
	if err != nil {
		return err
	}
	if elements != resident.elements {
		return fmt.Errorf("%s shape %v has %d elements, resident backing has %d", label, resident.shape, elements, resident.elements)
	}
	return nil
}

func validateResidentMatrixShapeProduct(label string, resident residentMatrix) error {
	elements, err := checkedProduct(label+" shape product", resident.rows, resident.cols)
	if err != nil {
		return err
	}
	if elements != resident.elements {
		return fmt.Errorf("%s shape [%d,%d] has %d elements, resident backing has %d", label, resident.rows, resident.cols, elements, resident.elements)
	}
	return nil
}

type bertCUDAOneLayerFixtureOptions struct {
	InjectStatusBeforeDownload int32
}

func (rt *deviceRuntime) runBERTOneLayerResidentFixture(inputIDs, attentionMask, tokenTypeIDs *backend.Tensor, names map[string]string, heads int, epsilon float64) (*backend.Tensor, bertCUDAOneLayerTransferStats, error) {
	return rt.runBERTOneLayerResidentFixtureWithOptions(inputIDs, attentionMask, tokenTypeIDs, names, heads, epsilon, bertCUDAOneLayerFixtureOptions{})
}

func (rt *deviceRuntime) runBERTOneLayerResidentFixtureWithOptions(inputIDs, attentionMask, tokenTypeIDs *backend.Tensor, names map[string]string, heads int, epsilon float64, options bertCUDAOneLayerFixtureOptions) (*backend.Tensor, bertCUDAOneLayerTransferStats, error) {
	stats := bertCUDAOneLayerTransferStats{}
	pre, err := rt.preflightBERTOneLayerResidentFixture(inputIDs, attentionMask, tokenTypeIDs, names, heads)
	if err != nil {
		return nil, stats, err
	}
	stats.ResidentWeightBytesReferenced = pre.residentWeightBytesReferenced
	batch, tokens := pre.batch, pre.tokens
	rows, hidden, intermediate := pre.rows, pre.hidden, pre.intermediate
	hiddenElements, intermediateElements := pre.hiddenElements, pre.intermediateElements
	launched := false
	inputBuf, err := rt.uploadInt32(inputIDs.I32)
	if err != nil {
		return nil, stats, err
	}
	defer rt.freeBuffer(inputBuf)
	maskBuf, err := rt.uploadInt32(attentionMask.I32)
	if err != nil {
		return nil, stats, err
	}
	defer rt.freeBuffer(maskBuf)
	typeBuf, err := rt.uploadInt32(tokenTypeIDs.I32)
	if err != nil {
		return nil, stats, err
	}
	defer rt.freeBuffer(typeBuf)
	stats.UploadedBytes += int64((len(inputIDs.I32) + len(attentionMask.I32) + len(tokenTypeIDs.I32)) * 4)

	hiddenBuf, err := rt.allocFloat32(hiddenElements)
	if err != nil {
		return nil, stats, err
	}
	defer rt.freeBuffer(hiddenBuf)
	qBuf, err := rt.allocFloat32(hiddenElements)
	if err != nil {
		return nil, stats, err
	}
	defer rt.freeBuffer(qBuf)
	kBuf, err := rt.allocFloat32(hiddenElements)
	if err != nil {
		return nil, stats, err
	}
	defer rt.freeBuffer(kBuf)
	vBuf, err := rt.allocFloat32(hiddenElements)
	if err != nil {
		return nil, stats, err
	}
	defer rt.freeBuffer(vBuf)
	contextBuf, err := rt.allocFloat32(hiddenElements)
	if err != nil {
		return nil, stats, err
	}
	defer rt.freeBuffer(contextBuf)
	attentionProjectedBuf, err := rt.allocFloat32(hiddenElements)
	if err != nil {
		return nil, stats, err
	}
	defer rt.freeBuffer(attentionProjectedBuf)
	attentionLayerBuf, err := rt.allocFloat32(hiddenElements)
	if err != nil {
		return nil, stats, err
	}
	defer rt.freeBuffer(attentionLayerBuf)
	intermediateBuf, err := rt.allocFloat32(intermediateElements)
	if err != nil {
		return nil, stats, err
	}
	defer rt.freeBuffer(intermediateBuf)
	intermediateGELUBuf, err := rt.allocFloat32(intermediateElements)
	if err != nil {
		return nil, stats, err
	}
	defer rt.freeBuffer(intermediateGELUBuf)
	outputProjectedBuf, err := rt.allocFloat32(hiddenElements)
	if err != nil {
		return nil, stats, err
	}
	defer rt.freeBuffer(outputProjectedBuf)
	outputLayerBuf, err := rt.allocFloat32(hiddenElements)
	if err != nil {
		return nil, stats, err
	}
	defer rt.freeBuffer(outputLayerBuf)
	statusBuf, err := rt.uploadInt32([]int32{0})
	if err != nil {
		return nil, stats, err
	}
	defer rt.freeBuffer(statusBuf)
	stats.UploadedBytes += 4
	defer func() {
		if launched {
			_ = rt.synchronize()
		}
	}()

	if err := rt.runBERTEmbeddingAffineLayerNormDevice(names["token_embeddings"], names["position_embeddings"], names["token_type_embeddings"], names["embedding_layernorm_weight"], names["embedding_layernorm_bias"], inputBuf, typeBuf, hiddenBuf, statusBuf, batch, tokens, epsilon); err != nil {
		return nil, stats, err
	}
	launched = true
	for _, item := range []struct {
		weight string
		bias   string
		out    C.CUdeviceptr
	}{
		{"attention_query_weight", "attention_query_bias", qBuf},
		{"attention_key_weight", "attention_key_bias", kBuf},
		{"attention_value_weight", "attention_value_bias", vBuf},
	} {
		if err := rt.runResidentBERTGEMM(hiddenBuf, rows, hidden, names[item.weight], item.out, true); err != nil {
			return nil, stats, err
		}
		if err := rt.runBERTBiasAddDevice(item.out, names[item.bias], rows, hidden); err != nil {
			return nil, stats, err
		}
	}
	if err := rt.runBERTAttentionContextDevice(qBuf, kBuf, vBuf, maskBuf, contextBuf, batch, tokens, hidden, heads); err != nil {
		return nil, stats, err
	}
	if err := rt.runResidentBERTGEMM(contextBuf, rows, hidden, names["attention_output_weight"], attentionProjectedBuf, true); err != nil {
		return nil, stats, err
	}
	if err := rt.runBERTBiasAddDevice(attentionProjectedBuf, names["attention_output_bias"], rows, hidden); err != nil {
		return nil, stats, err
	}
	if err := rt.runBERTResidualAffineLayerNormDevice(attentionProjectedBuf, hiddenBuf, names["attention_layernorm_weight"], names["attention_layernorm_bias"], attentionLayerBuf, rows, hidden, epsilon); err != nil {
		return nil, stats, err
	}
	if err := rt.runResidentBERTGEMM(attentionLayerBuf, rows, hidden, names["intermediate_weight"], intermediateBuf, true); err != nil {
		return nil, stats, err
	}
	if err := rt.runBERTBiasAddDevice(intermediateBuf, names["intermediate_bias"], rows, intermediate); err != nil {
		return nil, stats, err
	}
	if err := rt.runBERTExactGELUDevice(intermediateBuf, intermediateGELUBuf, intermediateElements); err != nil {
		return nil, stats, err
	}
	if err := rt.runResidentBERTGEMM(intermediateGELUBuf, rows, intermediate, names["output_weight"], outputProjectedBuf, true); err != nil {
		return nil, stats, err
	}
	if err := rt.runBERTBiasAddDevice(outputProjectedBuf, names["output_bias"], rows, hidden); err != nil {
		return nil, stats, err
	}
	if err := rt.runBERTResidualAffineLayerNormDevice(outputProjectedBuf, attentionLayerBuf, names["output_layernorm_weight"], names["output_layernorm_bias"], outputLayerBuf, rows, hidden, epsilon); err != nil {
		return nil, stats, err
	}
	if err := rt.synchronize(); err != nil {
		return nil, stats, err
	}
	launched = false
	if options.InjectStatusBeforeDownload != 0 {
		if err := rt.copyInt32ToBuffer(statusBuf, []int32{options.InjectStatusBeforeDownload}); err != nil {
			return nil, stats, err
		}
		stats.UploadedBytes += 4
	}
	status := []int32{0}
	if err := rt.downloadInt32(status, statusBuf); err != nil {
		return nil, stats, err
	}
	stats.StatusDownloadedBytes = 4
	stats.DownloadedBytes += stats.StatusDownloadedBytes
	if status[0] != 0 {
		return nil, stats, fmt.Errorf("cuda bert one-layer fixture embedding status=%d", status[0])
	}
	out := make([]float32, hiddenElements)
	if err := rt.downloadFloat32(out, outputLayerBuf); err != nil {
		return nil, stats, err
	}
	stats.FinalDownloadedBytes = int64(hiddenElements * 4)
	stats.DownloadedBytes += stats.FinalDownloadedBytes
	return backend.NewTensorF32([]int{batch, tokens, hidden}, out), stats, nil
}

func (rt *deviceRuntime) preflightBGEFullEncoder(step eosartifact.Step, inputs []*backend.Tensor) (bertCUDAResidentWeightPlan, bertCUDAFullEncoderTransferStats, error) {
	stats := bertCUDAFullEncoderTransferStats{Layers: bgeSmallLayers}
	plan, _, err := planBGEPretrainedBERTResidentWeights(step, inputs)
	if err != nil {
		return bertCUDAResidentWeightPlan{}, stats, err
	}
	if len(inputs) < 3 {
		return bertCUDAResidentWeightPlan{}, stats, fmt.Errorf("cuda BGE full encoder expects token inputs")
	}
	batch, tokens := inputs[0].Shape[0], inputs[0].Shape[1]
	maxBatchTokens, err := bertCUDAMaxBatchTokensFromEnv(bgeSmallMaxPositions * 64)
	if err != nil {
		return bertCUDAResidentWeightPlan{}, stats, err
	}
	stats.MaxBatchTokens = maxBatchTokens
	rows, err := checkedProduct("cuda BGE full encoder batch tokens", batch, tokens)
	if err != nil {
		return bertCUDAResidentWeightPlan{}, stats, err
	}
	if batch <= 0 {
		return bertCUDAResidentWeightPlan{}, stats, fmt.Errorf("cuda BGE full encoder batch=%d unsupported for cooperative attention, want positive", batch)
	}
	if tokens <= 0 || tokens > 512 {
		return bertCUDAResidentWeightPlan{}, stats, fmt.Errorf("cuda BGE full encoder tokens=%d unsupported for cooperative attention, want 1..512", tokens)
	}
	if rows > maxBatchTokens {
		return bertCUDAResidentWeightPlan{}, stats, fmt.Errorf("cuda BGE full encoder batch tokens %d exceed EOS_BERT_CUDA_MAX_BATCH_TOKENS=%d", rows, maxBatchTokens)
	}
	for _, weight := range plan.Weights {
		switch weight.Role {
		case bertCUDAWeightDenseMatrix:
			resident, ok := rt.residentMatrices[weight.Name]
			if !ok {
				if _, wrongKind := rt.bertResidentTensors[weight.Name]; wrongKind {
					return bertCUDAResidentWeightPlan{}, stats, fmt.Errorf("cuda BGE full encoder %s is bound as tensor, want matrix", weight.Name)
				}
				return bertCUDAResidentWeightPlan{}, stats, fmt.Errorf("cuda BGE full encoder resident matrix %q is not bound", weight.Name)
			}
			if err := validateResidentMatrixShapeProduct("cuda BGE full encoder "+weight.Name, resident); err != nil {
				return bertCUDAResidentWeightPlan{}, stats, err
			}
			if resident.rows != weight.Shape[0] || resident.cols != weight.Shape[1] {
				return bertCUDAResidentWeightPlan{}, stats, fmt.Errorf("cuda BGE full encoder matrix %q shape [%d,%d], want %v", weight.Name, resident.rows, resident.cols, weight.Shape)
			}
			bytes, err := checkedCUDABytes("cuda BGE full encoder resident matrix bytes "+weight.Name, resident.elements, 4)
			if err != nil {
				return bertCUDAResidentWeightPlan{}, stats, err
			}
			stats.ResidentWeightBytesReferenced += int64(bytes)
		default:
			resident, ok := rt.bertResidentTensors[weight.Name]
			if !ok {
				if _, wrongKind := rt.residentMatrices[weight.Name]; wrongKind {
					return bertCUDAResidentWeightPlan{}, stats, fmt.Errorf("cuda BGE full encoder %s is bound as matrix, want tensor", weight.Name)
				}
				return bertCUDAResidentWeightPlan{}, stats, fmt.Errorf("cuda BGE full encoder resident tensor %q is not bound", weight.Name)
			}
			if err := validateResidentTensorShapeProduct("cuda BGE full encoder "+weight.Name, resident); err != nil {
				return bertCUDAResidentWeightPlan{}, stats, err
			}
			if len(resident.shape) != len(weight.Shape) {
				return bertCUDAResidentWeightPlan{}, stats, fmt.Errorf("cuda BGE full encoder tensor %q rank %d, want %d", weight.Name, len(resident.shape), len(weight.Shape))
			}
			for i := range weight.Shape {
				if resident.shape[i] != weight.Shape[i] {
					return bertCUDAResidentWeightPlan{}, stats, fmt.Errorf("cuda BGE full encoder tensor %q shape %v, want %v", weight.Name, resident.shape, weight.Shape)
				}
			}
			bytes, err := checkedCUDABytes("cuda BGE full encoder resident tensor bytes "+weight.Name, resident.elements, 4)
			if err != nil {
				return bertCUDAResidentWeightPlan{}, stats, err
			}
			stats.ResidentWeightBytesReferenced += int64(bytes)
		}
	}
	hiddenElements, err := checkedProduct("cuda BGE full encoder hidden elements", rows, bgeSmallHiddenSize)
	if err != nil {
		return bertCUDAResidentWeightPlan{}, stats, err
	}
	intermediateElements, err := checkedProduct("cuda BGE full encoder intermediate elements", rows, bgeSmallIntermediateSize)
	if err != nil {
		return bertCUDAResidentWeightPlan{}, stats, err
	}
	hiddenWorkspaceElements, err := checkedProduct("cuda BGE full encoder hidden workspace elements", hiddenElements, 9)
	if err != nil {
		return bertCUDAResidentWeightPlan{}, stats, err
	}
	intermediateWorkspaceElements, err := checkedProduct("cuda BGE full encoder intermediate workspace elements", intermediateElements, 2)
	if err != nil {
		return bertCUDAResidentWeightPlan{}, stats, err
	}
	finalElements, err := checkedProduct("cuda BGE full encoder final elements", batch, bgeSmallHiddenSize)
	if err != nil {
		return bertCUDAResidentWeightPlan{}, stats, err
	}
	workspaceElements := 0
	for _, addend := range []int{hiddenWorkspaceElements, intermediateWorkspaceElements, finalElements} {
		if workspaceElements > int(^uint(0)>>1)-addend {
			return bertCUDAResidentWeightPlan{}, stats, fmt.Errorf("cuda BGE full encoder workspace elements sum overflows int")
		}
		workspaceElements += addend
	}
	if workspaceBytes, err := checkedCUDABytes("cuda BGE full encoder workspace bytes", workspaceElements, 4); err != nil {
		return bertCUDAResidentWeightPlan{}, stats, err
	} else {
		stats.WorkspaceBytes = int64(workspaceBytes)
	}
	attentionJobs, err := checkedProduct("cuda BGE full encoder attention jobs", rows, bgeSmallHeads)
	if err != nil {
		return bertCUDAResidentWeightPlan{}, stats, err
	}
	for _, launch := range []struct {
		label    string
		elements int
		block    uint
	}{
		{"cuda BGE full encoder embedding launch", rows, 128},
		{"cuda BGE full encoder qkv bias launch", hiddenElements, 256},
		{"cuda BGE full encoder attention launch", attentionJobs, bertAttentionContextBlockSize(tokens)},
		{"cuda BGE full encoder gelu launch", intermediateElements, 256},
		{"cuda BGE full encoder residual layernorm launch", rows, 128},
		{"cuda BGE full encoder cls l2 launch", batch, 128},
	} {
		if _, _, err := checkedLaunch1D(launch.label, launch.elements, launch.block); err != nil {
			return bertCUDAResidentWeightPlan{}, stats, err
		}
	}
	return plan, stats, nil
}

func (rt *deviceRuntime) runBGEFullEncoderHidden(step eosartifact.Step, outputType eosartifact.ValueType, inputs []*backend.Tensor) (backend.StepDispatchResult, bertCUDAFullEncoderTransferStats, error) {
	runStart := time.Now()
	_, stats, err := rt.preflightBGEFullEncoder(step, inputs)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	inputIDs, attentionMask, tokenTypeIDs := inputs[0], inputs[1], inputs[2]
	batch, tokens := inputIDs.Shape[0], inputIDs.Shape[1]
	rows, err := checkedProduct("cuda BGE full encoder rows", batch, tokens)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	hiddenElements, err := checkedProduct("cuda BGE full encoder hidden elements", rows, bgeSmallHiddenSize)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	intermediateElements, err := checkedProduct("cuda BGE full encoder intermediate elements", rows, bgeSmallIntermediateSize)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	finalElements, err := checkedProduct("cuda BGE full encoder output elements", batch, bgeSmallHiddenSize)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}

	inputBuf, err := rt.uploadInt32(inputIDs.I32)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	defer rt.freeBuffer(inputBuf)
	maskBuf, err := rt.uploadInt32(attentionMask.I32)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	defer rt.freeBuffer(maskBuf)
	typeBuf, err := rt.uploadInt32(tokenTypeIDs.I32)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	defer rt.freeBuffer(typeBuf)
	statusBuf, err := rt.uploadInt32([]int32{0})
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	defer rt.freeBuffer(statusBuf)
	inputUploadElements := 0
	for _, addend := range []int{len(inputIDs.I32), len(attentionMask.I32), len(tokenTypeIDs.I32), 1} {
		if inputUploadElements > int(^uint(0)>>1)-addend {
			return backend.StepDispatchResult{}, stats, fmt.Errorf("cuda BGE full encoder input upload elements sum overflows int")
		}
		inputUploadElements += addend
	}
	inputUploadBytes, err := checkedCUDABytes("cuda BGE full encoder input upload bytes", inputUploadElements, 4)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	stats.InputUploadedBytes += int64(inputUploadBytes)
	stats.UploadedBytes += stats.InputUploadedBytes

	hiddenA, err := rt.matMulScratchFloat32("bge12_hidden_a", hiddenElements)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	hiddenB, err := rt.matMulScratchFloat32("bge12_hidden_b", hiddenElements)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	qBuf, err := rt.matMulScratchFloat32("bge12_q", hiddenElements)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	kBuf, err := rt.matMulScratchFloat32("bge12_k", hiddenElements)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	vBuf, err := rt.matMulScratchFloat32("bge12_v", hiddenElements)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	contextBuf, err := rt.matMulScratchFloat32("bge12_context", hiddenElements)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	attentionProjectedBuf, err := rt.matMulScratchFloat32("bge12_attention_projected", hiddenElements)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	attentionLayerBuf, err := rt.matMulScratchFloat32("bge12_attention_layer", hiddenElements)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	intermediateBuf, err := rt.matMulScratchFloat32("bge12_intermediate", intermediateElements)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	intermediateGELUBuf, err := rt.matMulScratchFloat32("bge12_intermediate_gelu", intermediateElements)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	outputProjectedBuf, err := rt.matMulScratchFloat32("bge12_output_projected", hiddenElements)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	finalBuf, err := rt.matMulScratchFloat32("bge12_final_embeddings", finalElements)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}

	launched := false
	defer func() {
		if launched {
			_ = rt.synchronize()
		}
	}()
	if err := rt.runBERTEmbeddingAffineLayerNormDevice(step.Inputs[3], step.Inputs[4], step.Inputs[5], step.Inputs[6], step.Inputs[7], inputBuf, typeBuf, hiddenA, statusBuf, batch, tokens, bgeSmallLayerNormEpsilon); err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	launched = true
	cur, next := hiddenA, hiddenB
	layerSlot := func(layer int, offset int) string {
		return step.Inputs[8+layer*16+offset]
	}
	for layer := 0; layer < bgeSmallLayers; layer++ {
		for _, item := range []struct {
			weightOffset int
			biasOffset   int
			out          C.CUdeviceptr
		}{
			{0, 1, qBuf},
			{2, 3, kBuf},
			{4, 5, vBuf},
		} {
			if err := rt.runResidentBERTGEMM(cur, rows, bgeSmallHiddenSize, layerSlot(layer, item.weightOffset), item.out, true); err != nil {
				return backend.StepDispatchResult{}, stats, err
			}
			if err := rt.runBERTBiasAddDevice(item.out, layerSlot(layer, item.biasOffset), rows, bgeSmallHiddenSize); err != nil {
				return backend.StepDispatchResult{}, stats, err
			}
		}
		if err := rt.runBERTAttentionContextDevice(qBuf, kBuf, vBuf, maskBuf, contextBuf, batch, tokens, bgeSmallHiddenSize, bgeSmallHeads); err != nil {
			return backend.StepDispatchResult{}, stats, err
		}
		if err := rt.runResidentBERTGEMM(contextBuf, rows, bgeSmallHiddenSize, layerSlot(layer, 6), attentionProjectedBuf, true); err != nil {
			return backend.StepDispatchResult{}, stats, err
		}
		if err := rt.runBERTBiasAddDevice(attentionProjectedBuf, layerSlot(layer, 7), rows, bgeSmallHiddenSize); err != nil {
			return backend.StepDispatchResult{}, stats, err
		}
		if err := rt.runBERTResidualAffineLayerNormDevice(attentionProjectedBuf, cur, layerSlot(layer, 8), layerSlot(layer, 9), attentionLayerBuf, rows, bgeSmallHiddenSize, bgeSmallLayerNormEpsilon); err != nil {
			return backend.StepDispatchResult{}, stats, err
		}
		if err := rt.runResidentBERTGEMM(attentionLayerBuf, rows, bgeSmallHiddenSize, layerSlot(layer, 10), intermediateBuf, true); err != nil {
			return backend.StepDispatchResult{}, stats, err
		}
		if err := rt.runBERTBiasAddDevice(intermediateBuf, layerSlot(layer, 11), rows, bgeSmallIntermediateSize); err != nil {
			return backend.StepDispatchResult{}, stats, err
		}
		if err := rt.runBERTExactGELUDevice(intermediateBuf, intermediateGELUBuf, intermediateElements); err != nil {
			return backend.StepDispatchResult{}, stats, err
		}
		if err := rt.runResidentBERTGEMM(intermediateGELUBuf, rows, bgeSmallIntermediateSize, layerSlot(layer, 12), outputProjectedBuf, true); err != nil {
			return backend.StepDispatchResult{}, stats, err
		}
		if err := rt.runBERTBiasAddDevice(outputProjectedBuf, layerSlot(layer, 13), rows, bgeSmallHiddenSize); err != nil {
			return backend.StepDispatchResult{}, stats, err
		}
		if err := rt.runBERTResidualAffineLayerNormDevice(outputProjectedBuf, attentionLayerBuf, layerSlot(layer, 14), layerSlot(layer, 15), next, rows, bgeSmallHiddenSize, bgeSmallLayerNormEpsilon); err != nil {
			return backend.StepDispatchResult{}, stats, err
		}
		cur, next = next, cur
	}
	if err := rt.runBERTCLSL2Device(cur, finalBuf, batch, tokens, bgeSmallHiddenSize); err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	if err := rt.synchronize(); err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	launched = false
	status := []int32{0}
	if err := rt.downloadInt32(status, statusBuf); err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	stats.StatusDownloadedBytes = 4
	stats.DownloadedBytes += 4
	if status[0] != 0 {
		return backend.StepDispatchResult{}, stats, fmt.Errorf("cuda BGE full encoder device status=%d", status[0])
	}
	out := make([]float32, finalElements)
	if err := rt.downloadFloat32(out, finalBuf); err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	finalDownloadedBytes, err := checkedCUDABytes("cuda BGE full encoder final download bytes", len(out), 4)
	if err != nil {
		return backend.StepDispatchResult{}, stats, err
	}
	stats.FinalDownloadedBytes = int64(finalDownloadedBytes)
	stats.DownloadedBytes += stats.FinalDownloadedBytes
	stats.RunNanos = time.Since(runStart).Nanoseconds()
	tensor := backend.NewTensorF32([]int{batch, bgeSmallHiddenSize}, out)
	if outputType.Tensor != nil {
		if outputType.Tensor.DType != "" && outputType.Tensor.DType != "f32" && outputType.Tensor.DType != "f16" {
			return backend.StepDispatchResult{}, stats, fmt.Errorf("cuda BGE full encoder output dtype %q is not f32-compatible", outputType.Tensor.DType)
		}
		if len(outputType.Tensor.Shape) != 0 && len(outputType.Tensor.Shape) != len(tensor.Shape) {
			return backend.StepDispatchResult{}, stats, fmt.Errorf("cuda BGE full encoder output rank %d does not match shape %v", len(outputType.Tensor.Shape), tensor.Shape)
		}
	}
	return backend.StepDispatchResult{
		Outputs:      []*backend.Tensor{tensor},
		VariantEntry: "__builtin_cuda_bge_small_12layer_hidden_resident",
		Metadata: map[string]any{
			"execution_mode":                      "pretrained_bert_cuda_hidden_resident_12layer",
			"selected_backend":                    "cuda",
			"full_device_execution":               false,
			"validated_device_encoder":            false,
			"device_encoder_contract":             pretrainedBERTCUDAFoundationContract,
			"device_encoder_contract_satisfied":   false,
			"hidden_gate":                         "EOS_BERT_CUDA_12LAYER_HIDDEN",
			"pooling":                             "cls",
			"normalization":                       "l2",
			"layers":                              bgeSmallLayers,
			"hidden":                              bgeSmallHiddenSize,
			"heads":                               bgeSmallHeads,
			"intermediate":                        bgeSmallIntermediateSize,
			"resident_uploaded_bytes":             stats.ResidentUploadedBytes,
			"input_uploaded_bytes":                stats.InputUploadedBytes,
			"uploaded_bytes":                      stats.UploadedBytes,
			"downloaded_bytes":                    stats.DownloadedBytes,
			"final_downloaded_bytes":              stats.FinalDownloadedBytes,
			"status_downloaded_bytes":             stats.StatusDownloadedBytes,
			"intermediate_downloaded_bytes":       stats.IntermediateDownloadedBytes,
			"resident_weight_bytes_referenced":    stats.ResidentWeightBytesReferenced,
			"resident_cache_hits":                 stats.ResidentCacheHits,
			"resident_cache_misses":               stats.ResidentCacheMisses,
			"resident_bind_nanos":                 stats.ResidentBindNanos,
			"run_nanos":                           stats.RunNanos,
			"cold_resident_bind":                  stats.ColdResidentBind,
			"warm_resident_cache":                 !stats.ColdResidentBind,
			"contract_fingerprint_sha256":         stats.ContractFingerprint,
			"weight_fingerprint_sha256":           stats.WeightFingerprint,
			"workspace_bytes":                     stats.WorkspaceBytes,
			"max_batch_tokens":                    stats.MaxBatchTokens,
			"attention_kernel_variant":            "cooperative_shared_logit_v2",
			"attention_kernel_block_size":         int(bertAttentionContextBlockSize(tokens)),
			"attention_kernel_max_tokens":         512,
			"opportunistic_device_ops_ignored":    true,
			"public_device_encoder_claim_blocked": true,
		},
	}, stats, nil
}

// runCapturedGEMMBatch executes a batch of stream GEMMs. On a cache hit for
// `key` it replays the captured graph (one launch + one sync, no per-GEMM
// launch/sync overhead). On a miss it runs the batch once synchronously (which
// both produces this call's result AND warms cuBLAS workspace for every shape),
// then captures the identical sequence for future replays. `issue` must enqueue
// the GEMMs on rt's stream without syncing; all operands must be
// stable-pointered buffers (resident weights + named scratch) whose contents
// are refreshed by the caller before each invocation.
func (rt *deviceRuntime) runCapturedGEMMBatch(key string, issue func() error) error {
	if rt.graphCache != nil {
		if g, ok := rt.graphCache[key]; ok {
			return rt.launchGraph(g)
		}
	}
	// Miss: warm cuBLAS workspace + produce this call's result synchronously.
	if err := issue(); err != nil {
		return err
	}
	if err := rt.synchronize(); err != nil {
		return err
	}
	// Capture the same sequence (recorded, not executed) for future replays.
	if err := rt.beginCapture(); err != nil {
		return err
	}
	captureErr := issue()
	g, endErr := rt.endCapture()
	if captureErr != nil {
		if endErr == nil {
			g.destroy()
		}
		return captureErr
	}
	if endErr != nil {
		return endErr
	}
	if rt.graphCache == nil {
		rt.graphCache = map[string]*cudaGraph{}
	}
	rt.graphCache[key] = g
	return nil
}

func cudaFloat32SliceBitEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// graphReplaySelfTest validates the CUDA-graph capture/replay primitive on a
// small deterministic GEMM. It asserts a captured graph (1) reproduces the
// direct (synced) cuBLAS result bit-for-bit, and (2) on replay recomputes
// against fresh contents of the same stable input buffer. This retires the
// cuBLAS-in-graph risk before capture is wired into the forward pass.
func (rt *deviceRuntime) graphReplaySelfTest() error {
	const m, k, n = 8, 16, 4
	a1 := make([]float32, m*k)
	a2 := make([]float32, m*k)
	b := make([]float32, k*n)
	for i := range a1 {
		a1[i] = float32((i%7)-3) * 0.25
	}
	for i := range a2 {
		a2[i] = float32((i%5)-2) * 0.5
	}
	for i := range b {
		b[i] = float32((i%4)-1) * 0.5
	}

	bPtr, err := rt.uploadMatMulScratchFloat32("graphtest_b", b)
	if err != nil {
		return fmt.Errorf("upload b: %w", err)
	}
	aPtr, err := rt.uploadMatMulScratchFloat32("graphtest_a", a1)
	if err != nil {
		return fmt.Errorf("upload a: %w", err)
	}
	outPtr, err := rt.matMulScratchFloat32("graphtest_out", m*n)
	if err != nil {
		return fmt.Errorf("alloc out: %w", err)
	}

	gemm := func() error {
		return rt.matMulCublasWithBeta(aPtr, bPtr, outPtr, C.int(m), C.int(k), C.int(k), C.int(n), false, false, 0)
	}
	gemmNoSync := func() error {
		return rt.matMulCublasWithBetaNoSync(aPtr, bPtr, outPtr, C.int(m), C.int(k), C.int(k), C.int(n), false, false, 0)
	}
	download := func() ([]float32, error) {
		dst := make([]float32, m*n)
		if err := rt.downloadFloat32(dst, outPtr); err != nil {
			return nil, err
		}
		return dst, nil
	}

	// 1) Direct (synced) result for a1 — also warms the cuBLAS workspace for
	// this shape so no allocation happens during capture.
	if err := gemm(); err != nil {
		return fmt.Errorf("direct gemm a1: %w", err)
	}
	direct1, err := download()
	if err != nil {
		return fmt.Errorf("download direct a1: %w", err)
	}

	// 2) Capture the identical GEMM, replay it, expect bit-identical output.
	if err := rt.beginCapture(); err != nil {
		return fmt.Errorf("begin capture: %w", err)
	}
	if err := gemmNoSync(); err != nil {
		return fmt.Errorf("capture gemm: %w", err)
	}
	g, err := rt.endCapture()
	if err != nil {
		return fmt.Errorf("end capture: %w", err)
	}
	defer g.destroy()
	if err := rt.launchGraph(g); err != nil {
		return fmt.Errorf("replay a1: %w", err)
	}
	graph1, err := download()
	if err != nil {
		return fmt.Errorf("download graph a1: %w", err)
	}
	if !cudaFloat32SliceBitEqual(direct1, graph1) {
		return fmt.Errorf("graph replay != direct for a1: direct=%v graph=%v", direct1, graph1)
	}

	// 3) Overwrite the stable input buffer with a2 and confirm replay
	// recomputes against the new contents (matches a fresh direct GEMM).
	if err := rt.copyFloat32ToBuffer(aPtr, a2); err != nil {
		return fmt.Errorf("overwrite a2: %w", err)
	}
	if err := gemm(); err != nil {
		return fmt.Errorf("direct gemm a2: %w", err)
	}
	direct2, err := download()
	if err != nil {
		return fmt.Errorf("download direct a2: %w", err)
	}
	if err := rt.launchGraph(g); err != nil {
		return fmt.Errorf("replay a2: %w", err)
	}
	graph2, err := download()
	if err != nil {
		return fmt.Errorf("download graph a2: %w", err)
	}
	if !cudaFloat32SliceBitEqual(direct2, graph2) {
		return fmt.Errorf("graph replay != direct for a2 (stale capture?): direct=%v graph=%v", direct2, graph2)
	}
	if cudaFloat32SliceBitEqual(direct1, direct2) {
		return errors.New("self-test inputs a1/a2 produced identical output; pick more distinct data")
	}
	return nil
}

// hostSoftmaxRows is the CPU reference for the forward row softmax, matching
// the trainer's softmaxRowsInPlace. Kept local to avoid a cycle with the
// runtime package.
func hostSoftmaxRows(data []float32, rows, cols int) {
	for row := 0; row < rows; row++ {
		base := row * cols
		maxVal := data[base]
		for col := 1; col < cols; col++ {
			if data[base+col] > maxVal {
				maxVal = data[base+col]
			}
		}
		var sum float32
		for col := 0; col < cols; col++ {
			e := float32(math.Exp(float64(data[base+col] - maxVal)))
			data[base+col] = e
			sum += e
		}
		if sum == 0 {
			continue
		}
		inv := 1 / sum
		for col := 0; col < cols; col++ {
			data[base+col] *= inv
		}
	}
}

// softmaxForwardSelfTest compiles + runs the on-device forward row softmax and
// asserts it matches the host reference within float tolerance. Validates the
// first device-resident forward activation kernel before it is wired into the
// forward pass.
func (rt *deviceRuntime) softmaxForwardSelfTest() error {
	kernel, err := rt.compileAuxKernel(forwardSoftmaxRowsKernelSource, "manta_softmax_forward_rows")
	if err != nil {
		return fmt.Errorf("compile forward softmax kernel: %w", err)
	}
	defer rt.destroyAuxKernel(kernel)

	const rows, cols = 6, 9
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32((i*3)%11-5) * 0.3
	}
	want := append([]float32(nil), data...)
	hostSoftmaxRows(want, rows, cols)

	buf, err := rt.uploadFloat32(data)
	if err != nil {
		return fmt.Errorf("upload: %w", err)
	}
	defer rt.freeBuffer(buf)
	block := uint(128)
	grid := uint((rows + int(block) - 1) / int(block))
	if err := rt.launchAuxSoftmaxForwardRows(kernel, grid, block, buf, rows, cols); err != nil {
		return fmt.Errorf("launch: %w", err)
	}
	got := make([]float32, rows*cols)
	if err := rt.downloadFloat32(got, buf); err != nil {
		return fmt.Errorf("download: %w", err)
	}
	for i := range want {
		if d := got[i] - want[i]; d > 1e-5 || d < -1e-5 {
			return fmt.Errorf("softmax mismatch at %d: device=%g host=%g", i, got[i], want[i])
		}
	}
	// Confirm each row normalized to ~1.
	for row := 0; row < rows; row++ {
		var sum float32
		for col := 0; col < cols; col++ {
			sum += got[row*cols+col]
		}
		if d := sum - 1; d > 1e-4 || d < -1e-4 {
			return fmt.Errorf("row %d sums to %g, want ~1", row, sum)
		}
	}
	return nil
}

func hostGeluForward(x float32) float32 {
	cubic := x * x * x
	inner := float32(0.7978845608) * (x + float32(0.044715)*cubic)
	return 0.5 * x * (1 + float32(math.Tanh(float64(inner))))
}

func hostLayerNormRows(dst, src []float32, rows, cols int) {
	for row := 0; row < rows; row++ {
		base := row * cols
		var mean float32
		for c := 0; c < cols; c++ {
			mean += src[base+c]
		}
		mean /= float32(cols)
		var variance float32
		for c := 0; c < cols; c++ {
			d := src[base+c] - mean
			variance += d * d
		}
		variance /= float32(cols)
		invStd := float32(1.0 / math.Sqrt(float64(variance)+1e-5))
		for c := 0; c < cols; c++ {
			dst[base+c] = (src[base+c] - mean) * invStd
		}
	}
}

func nearlyEqualF32(a, b []float32, tol float32) (int, bool) {
	if len(a) != len(b) {
		return -1, false
	}
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > tol {
			return i, false
		}
	}
	return -1, true
}

// forwardActivationKernelsSelfTest validates the on-device forward GELU,
// layernorm, and residual-add kernels against host references. These are the
// remaining device-resident forward activations (after softmax) needed to keep
// the whole forward pass on the GPU.
func (rt *deviceRuntime) forwardActivationKernelsSelfTest() error {
	const rows, cols = 6, 9
	n := rows * cols
	src := make([]float32, n)
	for i := range src {
		src[i] = float32((i*7)%13-6) * 0.2
	}
	block := uint(128)

	// GELU (element-wise).
	{
		kernel, err := rt.compileAuxKernel(forwardGeluKernelSource, "manta_gelu_forward")
		if err != nil {
			return fmt.Errorf("compile gelu: %w", err)
		}
		defer rt.destroyAuxKernel(kernel)
		want := make([]float32, n)
		for i := range src {
			want[i] = hostGeluForward(src[i])
		}
		srcBuf, err := rt.uploadFloat32(src)
		if err != nil {
			return fmt.Errorf("gelu upload: %w", err)
		}
		defer rt.freeBuffer(srcBuf)
		dstBuf, err := rt.allocFloat32(n)
		if err != nil {
			return fmt.Errorf("gelu alloc: %w", err)
		}
		defer rt.freeBuffer(dstBuf)
		grid := uint((n + int(block) - 1) / int(block))
		if err := rt.launchAuxGeluForward(kernel, grid, block, srcBuf, dstBuf, n); err != nil {
			return fmt.Errorf("gelu launch: %w", err)
		}
		got := make([]float32, n)
		if err := rt.downloadFloat32(got, dstBuf); err != nil {
			return fmt.Errorf("gelu download: %w", err)
		}
		if idx, ok := nearlyEqualF32(want, got, 1e-4); !ok {
			return fmt.Errorf("gelu mismatch at %d: host=%g device=%g", idx, want[idx], got[idx])
		}
	}

	// LayerNorm (per row).
	{
		kernel, err := rt.compileAuxKernel(forwardLayerNormRowsKernelSource, "manta_layernorm_forward_rows")
		if err != nil {
			return fmt.Errorf("compile layernorm: %w", err)
		}
		defer rt.destroyAuxKernel(kernel)
		want := make([]float32, n)
		hostLayerNormRows(want, src, rows, cols)
		srcBuf, err := rt.uploadFloat32(src)
		if err != nil {
			return fmt.Errorf("layernorm upload: %w", err)
		}
		defer rt.freeBuffer(srcBuf)
		dstBuf, err := rt.allocFloat32(n)
		if err != nil {
			return fmt.Errorf("layernorm alloc: %w", err)
		}
		defer rt.freeBuffer(dstBuf)
		grid := uint((rows + int(block) - 1) / int(block))
		if err := rt.launchAuxLayerNormForwardRows(kernel, grid, block, srcBuf, dstBuf, rows, cols); err != nil {
			return fmt.Errorf("layernorm launch: %w", err)
		}
		got := make([]float32, n)
		if err := rt.downloadFloat32(got, dstBuf); err != nil {
			return fmt.Errorf("layernorm download: %w", err)
		}
		if idx, ok := nearlyEqualF32(want, got, 1e-4); !ok {
			return fmt.Errorf("layernorm mismatch at %d: host=%g device=%g", idx, want[idx], got[idx])
		}
	}

	// Residual add (element-wise).
	{
		kernel, err := rt.compileAuxKernel(forwardResidualAddKernelSource, "manta_residual_add")
		if err != nil {
			return fmt.Errorf("compile residual: %w", err)
		}
		defer rt.destroyAuxKernel(kernel)
		bvals := make([]float32, n)
		for i := range bvals {
			bvals[i] = float32((i*5)%9-4) * 0.15
		}
		want := make([]float32, n)
		for i := range src {
			want[i] = src[i] + bvals[i]
		}
		aBuf, err := rt.uploadFloat32(src)
		if err != nil {
			return fmt.Errorf("residual upload a: %w", err)
		}
		defer rt.freeBuffer(aBuf)
		bBuf, err := rt.uploadFloat32(bvals)
		if err != nil {
			return fmt.Errorf("residual upload b: %w", err)
		}
		defer rt.freeBuffer(bBuf)
		outBuf, err := rt.allocFloat32(n)
		if err != nil {
			return fmt.Errorf("residual alloc: %w", err)
		}
		defer rt.freeBuffer(outBuf)
		grid := uint((n + int(block) - 1) / int(block))
		if err := rt.launchAuxResidualAdd(kernel, grid, block, aBuf, bBuf, outBuf, n); err != nil {
			return fmt.Errorf("residual launch: %w", err)
		}
		got := make([]float32, n)
		if err := rt.downloadFloat32(got, outBuf); err != nil {
			return fmt.Errorf("residual download: %w", err)
		}
		if idx, ok := nearlyEqualF32(want, got, 1e-6); !ok {
			return fmt.Errorf("residual mismatch at %d: host=%g device=%g", idx, want[idx], got[idx])
		}
	}
	return nil
}

func cStringError(value *C.char) error {
	if value == nil {
		return fmt.Errorf("cuda error")
	}
	defer C.eosCudaFreeCString(value)
	return errors.New(C.GoString(value))
}

func (rt *deviceRuntime) synchronize() error {
	var errStr *C.char
	if C.eosCudaSynchronize(rt.ptr, &errStr) != 0 {
		return cStringError(errStr)
	}
	return nil
}

func cudaEnvFlagEnabled(name string) bool {
	switch cudaEnv(name) {
	case "1", "true", "TRUE", "yes", "YES":
		return true
	default:
		return false
	}
}

func cudaEnv(name string) string {
	if value, ok := os.LookupEnv(name); ok {
		return value
	}
	return ""
}

func (rt *deviceRuntime) runMatMul(inputs []*backend.Tensor, outputType eosartifact.ValueType) (backend.StepDispatchResult, error) {
	return rt.runMatMulWithTranspose(inputs, outputType, false, false)
}

func (rt *deviceRuntime) bindMatMulRight(name string, tensor *backend.Tensor) error {
	if rt == nil || rt.ptr == nil {
		return fmt.Errorf("cuda runtime is not initialized")
	}
	if name == "" {
		return fmt.Errorf("cuda matmul binding name is required")
	}
	if tensor == nil || len(tensor.Shape) != 2 {
		return fmt.Errorf("cuda matmul binding %q must be a rank-2 tensor", name)
	}
	elements := len(tensor.F32)
	quantBits := quantBitsForPreparedTensor(tensor)
	start := time.Now()
	uploadedBytes := int64(elements * 4)
	if resident, ok := rt.residentMatrices[name]; ok {
		if resident.elements == elements {
			if err := rt.copyFloat32ToBuffer(resident.ptr, tensor.F32); err != nil {
				return err
			}
			quantizeStart := time.Now()
			ran, err := rt.quantizeBufferInPlace(resident.ptr, elements, quantBits)
			if err != nil {
				return err
			}
			if ran {
				rt.matMulStats.QuantizePasses++
				rt.matMulStats.QuantizedBytes += uploadedBytes
				rt.matMulStats.QuantizeNanos += time.Since(quantizeStart).Nanoseconds()
			}
			resident.rows = tensor.Shape[0]
			resident.cols = tensor.Shape[1]
			rt.residentMatrices[name] = resident
			rt.matMulStats.BindCalls++
			rt.matMulStats.UploadedBytes += uploadedBytes
			rt.matMulStats.BindNanos += time.Since(start).Nanoseconds()
			rt.matMulStats.BoundMatrices = int64(len(rt.residentMatrices))
			return nil
		}
		_ = rt.freeBuffer(resident.ptr)
		delete(rt.residentMatrices, name)
	}
	ptr, err := rt.uploadFloat32(tensor.F32)
	if err != nil {
		return err
	}
	quantizeStart := time.Now()
	ran, err := rt.quantizeBufferInPlace(ptr, elements, quantBits)
	if err != nil {
		_ = rt.freeBuffer(ptr)
		return err
	}
	if ran {
		rt.matMulStats.QuantizePasses++
		rt.matMulStats.QuantizedBytes += uploadedBytes
		rt.matMulStats.QuantizeNanos += time.Since(quantizeStart).Nanoseconds()
	}
	rt.residentMatrices[name] = residentMatrix{
		ptr:      ptr,
		rows:     tensor.Shape[0],
		cols:     tensor.Shape[1],
		elements: elements,
	}
	rt.matMulStats.BindCalls++
	rt.matMulStats.UploadedBytes += uploadedBytes
	rt.matMulStats.BindNanos += time.Since(start).Nanoseconds()
	rt.matMulStats.BoundMatrices = int64(len(rt.residentMatrices))
	return nil
}

func (rt *deviceRuntime) unbindMatMulRight(name string) error {
	if rt == nil || rt.ptr == nil {
		return fmt.Errorf("cuda runtime is not initialized")
	}
	resident, ok := rt.residentMatrices[name]
	if !ok {
		return nil
	}
	if err := rt.freeBuffer(resident.ptr); err != nil {
		return err
	}
	delete(rt.residentMatrices, name)
	rt.matMulStats.BoundMatrices = int64(len(rt.residentMatrices))
	return nil
}

func (rt *deviceRuntime) runMatMulWithBoundRights(lhs *backend.Tensor, rightNames []string, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) ([]backend.StepDispatchResult, error) {
	if lhs == nil {
		return nil, fmt.Errorf("cuda matmul lhs is nil")
	}
	if len(rightNames) == 0 {
		return nil, fmt.Errorf("cuda matmul requires at least one rhs binding")
	}
	type boundRightPlan struct {
		name            string
		resident        residentMatrix
		batches         int
		rows            int
		inner           int
		cols            int
		outShape        []int
		lhsRows         int
		lhsCols         int
		rhsRows         int
		rhsCols         int
		downloadedBytes int64
	}
	plans := make([]boundRightPlan, len(rightNames))
	for i, name := range rightNames {
		resident, ok := rt.residentMatrices[name]
		if !ok {
			return nil, fmt.Errorf("cuda matmul binding %q is not resident", name)
		}
		rhsTensor := &backend.Tensor{DType: "f32", Shape: []int{resident.rows, resident.cols}}
		batches, rows, inner, cols, rhsBatched, outShape, err := matmulLayout(lhs, rhsTensor)
		if transposeLeft || transposeRight {
			batches, rows, inner, cols, rhsBatched, outShape, err = matmulLayoutWithTranspose(lhs, rhsTensor, transposeLeft, transposeRight)
		}
		if err != nil {
			return nil, err
		}
		if rhsBatched {
			return nil, fmt.Errorf("cuda cached rhs matmul does not support batched rhs")
		}
		lhsRows, lhsCols := batches*rows, inner
		rhsRows, rhsCols := inner, cols
		if transposeLeft || transposeRight {
			lhsRows, lhsCols = matrixShape(lhs)
			rhsRows, rhsCols = resident.rows, resident.cols
		}
		plans[i] = boundRightPlan{
			name:            name,
			resident:        resident,
			batches:         batches,
			rows:            rows,
			inner:           inner,
			cols:            cols,
			outShape:        outShape,
			lhsRows:         lhsRows,
			lhsCols:         lhsCols,
			rhsRows:         rhsRows,
			rhsCols:         rhsCols,
			downloadedBytes: int64(batches * rows * cols * 4),
		}
	}

	compiled := cudaBuiltinMatMulCompiledKernel()
	runStart := time.Now()
	firstRunUploadedBytes := int64(len(lhs.F32) * 4)
	lhsBuf, err := rt.uploadMatMulScratchFloat32("matmul_lhs", lhs.F32)
	if err != nil {
		return nil, err
	}
	results := make([]backend.StepDispatchResult, len(plans))
	outHosts := make([][]float32, len(plans))
	outBufs := make([]C.CUdeviceptr, len(plans))
	runStarts := make([]time.Time, len(plans))
	runUploadedBytes := make([]int64, len(plans))
	for i, plan := range plans {
		outHost := make([]float32, plan.batches*plan.rows*plan.cols)
		outBuf, err := rt.matMulScratchFloat32(fmt.Sprintf("matmul_out_bound_right_%d", i), len(outHost))
		if err != nil {
			return nil, err
		}
		uploadedBytes := firstRunUploadedBytes
		if i > 0 {
			runStart = time.Now()
			uploadedBytes = 0
		}
		runStarts[i] = runStart
		runUploadedBytes[i] = uploadedBytes
		outHosts[i] = outHost
		outBufs[i] = outBuf
	}

	// Issue the bound-right GEMMs on the stream without syncing. Every operand
	// (lhsBuf, resident weights, out scratch) is a stable-pointered buffer whose
	// contents are refreshed per call, so the sequence is safe to capture once
	// and replay against fresh data.
	issueGEMMs := func() error {
		for i, plan := range plans {
			if err := rt.matMulCublasWithBetaNoSync(lhsBuf, plan.resident.ptr, outBufs[i], C.int(plan.lhsRows), C.int(plan.lhsCols), C.int(plan.rhsRows), C.int(plan.rhsCols), transposeLeft, transposeRight, 0); err != nil {
				return err
			}
		}
		return nil
	}

	if eosCudaGraphEnabled {
		// Key the captured graph by the exact operands (device pointers) and
		// shapes it records, so any scratch reallocation or shape change is a
		// natural cache miss that forces a fresh capture.
		key := fmt.Sprintf("boundright|lhs=%d|tl=%t|tr=%t", uint64(lhsBuf), transposeLeft, transposeRight)
		for i, plan := range plans {
			key += fmt.Sprintf("|%s:rhs=%d,out=%d,%dx%d*%dx%d", plan.name, uint64(plan.resident.ptr), uint64(outBufs[i]), plan.lhsRows, plan.lhsCols, plan.rhsRows, plan.rhsCols)
		}
		if err := rt.runCapturedGEMMBatch(key, issueGEMMs); err != nil {
			return nil, err
		}
	} else {
		if err := issueGEMMs(); err != nil {
			return nil, err
		}
		if err := rt.synchronize(); err != nil {
			return nil, err
		}
	}
	for i, plan := range plans {
		if err := rt.downloadFloat32(outHosts[i], outBufs[i]); err != nil {
			return nil, err
		}
		rt.recordMatMulRun(runStarts[i], runUploadedBytes[i], plan.downloadedBytes, false, true)
		results[i] = backend.StepDispatchResult{
			Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, plan.outShape, outHosts[i])},
			VariantEntry: compiled.Entry,
			SourceHash:   compiled.SourceHash,
			Metadata: map[string]any{
				"dispatch_mode":    "backend_native",
				"execution_mode":   "cuda_device",
				"device_execution": true,
				"launch_api":       "cublasSgemm",
				"launch_compiler":  "cublas",
				"backend_library":  "cublas",
				"transpose_left":   transposeLeft,
				"transpose_right":  transposeRight,
				"rhs_binding":      plan.name,
				"rhs_residency":    "device_resident",
				"coalesced_lhs":    true,
			},
		}
	}
	return results, nil
}

func (rt *deviceRuntime) runAccumulatedMatMulsWithBoundRights(lhsInputs []*backend.Tensor, rightNames []string, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	if len(lhsInputs) == 0 {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda accumulated matmul requires at least one lhs input")
	}
	if len(lhsInputs) != len(rightNames) {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda accumulated matmul lhs/right count mismatch: %d != %d", len(lhsInputs), len(rightNames))
	}
	type accumulatedBoundRightPlan struct {
		lhs             *backend.Tensor
		name            string
		resident        residentMatrix
		batches         int
		rows            int
		inner           int
		cols            int
		outShape        []int
		lhsRows         int
		lhsCols         int
		rhsRows         int
		rhsCols         int
		uploadedBytes   int64
		downloadedBytes int64
	}
	plans := make([]accumulatedBoundRightPlan, len(lhsInputs))
	var outShape []int
	var outElements int
	for i, lhs := range lhsInputs {
		if lhs == nil {
			return backend.StepDispatchResult{}, fmt.Errorf("cuda accumulated matmul lhs %d is nil", i)
		}
		name := rightNames[i]
		resident, ok := rt.residentMatrices[name]
		if !ok {
			return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul binding %q is not resident", name)
		}
		rhsTensor := &backend.Tensor{DType: "f32", Shape: []int{resident.rows, resident.cols}}
		batches, rows, inner, cols, rhsBatched, planOutShape, err := matmulLayout(lhs, rhsTensor)
		if transposeLeft || transposeRight {
			batches, rows, inner, cols, rhsBatched, planOutShape, err = matmulLayoutWithTranspose(lhs, rhsTensor, transposeLeft, transposeRight)
		}
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		if rhsBatched {
			return backend.StepDispatchResult{}, fmt.Errorf("cuda cached rhs matmul does not support batched rhs")
		}
		if i == 0 {
			outShape = append([]int(nil), planOutShape...)
			outElements = batches * rows * cols
		} else if !sameIntSlice(outShape, planOutShape) {
			return backend.StepDispatchResult{}, fmt.Errorf("cuda accumulated matmul output shape mismatch: %v != %v", planOutShape, outShape)
		}
		lhsRows, lhsCols := batches*rows, inner
		rhsRows, rhsCols := inner, cols
		if transposeLeft || transposeRight {
			lhsRows, lhsCols = matrixShape(lhs)
			rhsRows, rhsCols = resident.rows, resident.cols
		}
		plans[i] = accumulatedBoundRightPlan{
			lhs:             lhs,
			name:            name,
			resident:        resident,
			batches:         batches,
			rows:            rows,
			inner:           inner,
			cols:            cols,
			outShape:        planOutShape,
			lhsRows:         lhsRows,
			lhsCols:         lhsCols,
			rhsRows:         rhsRows,
			rhsCols:         rhsCols,
			uploadedBytes:   int64(len(lhs.F32) * 4),
			downloadedBytes: int64(batches * rows * cols * 4),
		}
	}

	compiled := cudaBuiltinMatMulCompiledKernel()
	runStart := time.Now()
	var runUploadedBytes int64
	outHost := make([]float32, outElements)
	outBuf, err := rt.matMulScratchFloat32("matmul_accum_out", len(outHost))
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	singleSync := !cudaEnvFlagEnabled("EOS_CUDA_DISABLE_ACCUMULATED_MATMUL_SINGLE_SYNC")
	bindings := make([]string, len(plans))
	for i, plan := range plans {
		runUploadedBytes += plan.uploadedBytes
		bindings[i] = plan.name
		lhsBuf, err := rt.uploadMatMulScratchFloat32("matmul_accum_lhs", plan.lhs.F32)
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		beta := float32(0)
		if i > 0 {
			beta = 1
		}
		if singleSync {
			err = rt.matMulCublasWithBetaNoSync(lhsBuf, plan.resident.ptr, outBuf, C.int(plan.lhsRows), C.int(plan.lhsCols), C.int(plan.rhsRows), C.int(plan.rhsCols), transposeLeft, transposeRight, beta)
		} else {
			err = rt.matMulCublasWithBeta(lhsBuf, plan.resident.ptr, outBuf, C.int(plan.lhsRows), C.int(plan.lhsCols), C.int(plan.rhsRows), C.int(plan.rhsCols), transposeLeft, transposeRight, beta)
		}
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
	}
	if singleSync {
		err = rt.synchronize()
	}
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	rt.recordMatMulRun(runStart, runUploadedBytes, int64(len(outHost)*4), false, true)
	return backend.StepDispatchResult{
		Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, outShape, outHost)},
		VariantEntry: compiled.Entry,
		SourceHash:   compiled.SourceHash,
		Metadata: map[string]any{
			"dispatch_mode":             "backend_native",
			"execution_mode":            "cuda_device",
			"device_execution":          true,
			"launch_api":                "cublasSgemmAccumulated",
			"launch_compiler":           "cublas",
			"backend_library":           "cublas",
			"transpose_left":            transposeLeft,
			"transpose_right":           transposeRight,
			"rhs_bindings":              bindings,
			"rhs_residency":             "device_resident",
			"accumulated_bound_rights":  true,
			"accumulated_cublas_calls":  len(plans),
			"accumulated_download_once": true,
			"accumulated_single_sync":   singleSync,
		},
	}, nil
}

func (rt *deviceRuntime) runMatMulsWithSharedLeft(lhs *backend.Tensor, rhsInputs []*backend.Tensor, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) ([]backend.StepDispatchResult, error) {
	if lhs == nil {
		return nil, fmt.Errorf("cuda matmul lhs is nil")
	}
	if len(rhsInputs) == 0 {
		return nil, fmt.Errorf("cuda matmul requires at least one rhs input")
	}
	type sharedLeftPlan struct {
		rhs             *backend.Tensor
		batches         int
		rows            int
		inner           int
		cols            int
		rhsBatched      bool
		outShape        []int
		lhsRows         int
		lhsCols         int
		rhsRows         int
		rhsCols         int
		downloadedBytes int64
	}
	plans := make([]sharedLeftPlan, len(rhsInputs))
	for i, rhs := range rhsInputs {
		if rhs == nil {
			return nil, fmt.Errorf("cuda matmul rhs %d is nil", i)
		}
		batches, rows, inner, cols, rhsBatched, outShape, err := matmulLayout(lhs, rhs)
		if transposeLeft || transposeRight {
			batches, rows, inner, cols, rhsBatched, outShape, err = matmulLayoutWithTranspose(lhs, rhs, transposeLeft, transposeRight)
		}
		if err != nil {
			return nil, err
		}
		lhsRows, lhsCols := batches*rows, inner
		rhsRows, rhsCols := inner, cols
		if transposeLeft || transposeRight || rhsBatched {
			lhsRows, lhsCols = matrixShape(lhs)
			rhsRows, rhsCols = matrixShape(rhs)
		}
		plans[i] = sharedLeftPlan{
			rhs:             rhs,
			batches:         batches,
			rows:            rows,
			inner:           inner,
			cols:            cols,
			rhsBatched:      rhsBatched,
			outShape:        outShape,
			lhsRows:         lhsRows,
			lhsCols:         lhsCols,
			rhsRows:         rhsRows,
			rhsCols:         rhsCols,
			downloadedBytes: int64(batches * rows * cols * 4),
		}
	}

	compiled := cudaBuiltinMatMulCompiledKernel()
	lhsBuf, err := rt.uploadMatMulScratchFloat32("matmul_lhs", lhs.F32)
	if err != nil {
		return nil, err
	}
	results := make([]backend.StepDispatchResult, len(plans))
	canDeferSync := true
	for _, plan := range plans {
		if plan.rhsBatched {
			canDeferSync = false
			break
		}
	}
	outHosts := make([][]float32, len(plans))
	outBufs := make([]C.CUdeviceptr, len(plans))
	runStarts := make([]time.Time, len(plans))
	runUploadedBytes := make([]int64, len(plans))
	launchAPIs := make([]string, len(plans))
	for i, plan := range plans {
		runStart := time.Now()
		uploadedBytes := int64(len(plan.rhs.F32) * 4)
		if i == 0 {
			uploadedBytes += int64(len(lhs.F32) * 4)
		}
		rhsBuf, err := rt.uploadMatMulScratchFloat32(fmt.Sprintf("matmul_rhs_shared_left_%d", i), plan.rhs.F32)
		if err != nil {
			return nil, err
		}
		outHost := make([]float32, plan.batches*plan.rows*plan.cols)
		outBuf, err := rt.matMulScratchFloat32(fmt.Sprintf("matmul_out_shared_left_%d", i), len(outHost))
		if err != nil {
			return nil, err
		}
		launchAPI := "cublasSgemm"
		if plan.rhsBatched {
			if err := rt.matMulCublasStridedBatched(lhsBuf, rhsBuf, outBuf, C.int(plan.batches), C.int(plan.lhsRows), C.int(plan.lhsCols), C.int(plan.rhsRows), C.int(plan.rhsCols), transposeLeft, transposeRight); err != nil {
				return nil, err
			}
			launchAPI = "cublasSgemmStridedBatched"
			if err := rt.downloadFloat32(outHost, outBuf); err != nil {
				return nil, err
			}
		} else {
			if canDeferSync {
				if err := rt.matMulCublasWithBetaNoSync(lhsBuf, rhsBuf, outBuf, C.int(plan.lhsRows), C.int(plan.lhsCols), C.int(plan.rhsRows), C.int(plan.rhsCols), transposeLeft, transposeRight, 0); err != nil {
					return nil, err
				}
			} else if err := rt.matMulCublas(lhsBuf, rhsBuf, outBuf, C.int(plan.lhsRows), C.int(plan.lhsCols), C.int(plan.rhsRows), C.int(plan.rhsCols), transposeLeft, transposeRight); err != nil {
				return nil, err
			}
			if !canDeferSync {
				if err := rt.downloadFloat32(outHost, outBuf); err != nil {
					return nil, err
				}
			}
		}
		outHosts[i] = outHost
		outBufs[i] = outBuf
		runStarts[i] = runStart
		runUploadedBytes[i] = uploadedBytes
		launchAPIs[i] = launchAPI
	}
	if canDeferSync {
		if err := rt.synchronize(); err != nil {
			return nil, err
		}
		for i := range plans {
			if err := rt.downloadFloat32(outHosts[i], outBufs[i]); err != nil {
				return nil, err
			}
		}
	}
	for i, plan := range plans {
		rt.recordMatMulRun(runStarts[i], runUploadedBytes[i], plan.downloadedBytes, false, false)
		results[i] = backend.StepDispatchResult{
			Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, plan.outShape, outHosts[i])},
			VariantEntry: compiled.Entry,
			SourceHash:   compiled.SourceHash,
			Metadata: map[string]any{
				"dispatch_mode":    "backend_native",
				"execution_mode":   "cuda_device",
				"device_execution": true,
				"launch_api":       launchAPIs[i],
				"launch_compiler":  "cublas",
				"backend_library":  "cublas",
				"transpose_left":   transposeLeft,
				"transpose_right":  transposeRight,
				"shared_lhs":       true,
			},
		}
	}
	return results, nil
}

func (rt *deviceRuntime) runMatMulWithBoundRight(lhs *backend.Tensor, rightName string, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	if lhs == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul lhs is nil")
	}
	resident, ok := rt.residentMatrices[rightName]
	if !ok {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul binding %q is not resident", rightName)
	}
	rhsTensor := &backend.Tensor{DType: "f32", Shape: []int{resident.rows, resident.cols}}
	batches, rows, inner, cols, rhsBatched, outShape, err := matmulLayout(lhs, rhsTensor)
	if transposeLeft || transposeRight {
		batches, rows, inner, cols, rhsBatched, outShape, err = matmulLayoutWithTranspose(lhs, rhsTensor, transposeLeft, transposeRight)
	}
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	if rhsBatched {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda cached rhs matmul does not support batched rhs")
	}
	runStart := time.Now()
	runUploadedBytes := int64(len(lhs.F32) * 4)
	runDownloadedBytes := int64(batches * rows * cols * 4)
	compiled := cudaBuiltinMatMulCompiledKernel()
	outHost := make([]float32, batches*rows*cols)
	lhsBuf, err := rt.uploadMatMulScratchFloat32("matmul_lhs", lhs.F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	outBuf, err := rt.matMulScratchFloat32("matmul_out", len(outHost))
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	lhsRows, lhsCols := batches*rows, inner
	rhsRows, rhsCols := inner, cols
	if transposeLeft || transposeRight {
		lhsRows, lhsCols = matrixShape(lhs)
		rhsRows, rhsCols = resident.rows, resident.cols
	}
	if err := rt.matMulCublas(lhsBuf, resident.ptr, outBuf, C.int(lhsRows), C.int(lhsCols), C.int(rhsRows), C.int(rhsCols), transposeLeft, transposeRight); err != nil {
		return backend.StepDispatchResult{}, err
	}
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	rt.recordMatMulRun(runStart, runUploadedBytes, runDownloadedBytes, false, true)
	return backend.StepDispatchResult{
		Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, outShape, outHost)},
		VariantEntry: compiled.Entry,
		SourceHash:   compiled.SourceHash,
		Metadata: map[string]any{
			"dispatch_mode":    "backend_native",
			"execution_mode":   "cuda_device",
			"device_execution": true,
			"launch_api":       "cublasSgemm",
			"launch_compiler":  "cublas",
			"backend_library":  "cublas",
			"transpose_left":   transposeLeft,
			"transpose_right":  transposeRight,
			"rhs_binding":      rightName,
			"rhs_residency":    "device_resident",
		},
	}, nil
}

func (rt *deviceRuntime) runMatMulWithBoundLeft(leftName string, rhs *backend.Tensor, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	if rhs == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul rhs is nil")
	}
	resident, ok := rt.residentMatrices[leftName]
	if !ok {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul binding %q is not resident", leftName)
	}
	lhsTensor := &backend.Tensor{DType: "f32", Shape: []int{resident.rows, resident.cols}}
	batches, rows, inner, cols, rhsBatched, outShape, err := matmulLayout(lhsTensor, rhs)
	if transposeLeft || transposeRight {
		batches, rows, inner, cols, rhsBatched, outShape, err = matmulLayoutWithTranspose(lhsTensor, rhs, transposeLeft, transposeRight)
	}
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	if rhsBatched || batches != 1 {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda cached lhs matmul does not support batched inputs")
	}
	runStart := time.Now()
	runUploadedBytes := int64(len(rhs.F32) * 4)
	runDownloadedBytes := int64(batches * rows * cols * 4)
	compiled := cudaBuiltinMatMulCompiledKernel()
	outHost := make([]float32, batches*rows*cols)
	rhsBuf, err := rt.uploadMatMulScratchFloat32("matmul_rhs", rhs.F32)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	outBuf, err := rt.matMulScratchFloat32("matmul_out", len(outHost))
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	lhsRows, lhsCols := resident.rows, resident.cols
	rhsRows, rhsCols := batches*inner, cols
	if transposeLeft || transposeRight {
		lhsRows, lhsCols = resident.rows, resident.cols
		rhsRows, rhsCols = matrixShape(rhs)
	}
	if err := rt.matMulCublas(resident.ptr, rhsBuf, outBuf, C.int(lhsRows), C.int(lhsCols), C.int(rhsRows), C.int(rhsCols), transposeLeft, transposeRight); err != nil {
		return backend.StepDispatchResult{}, err
	}
	if err := rt.downloadFloat32(outHost, outBuf); err != nil {
		return backend.StepDispatchResult{}, err
	}
	rt.recordMatMulRun(runStart, runUploadedBytes, runDownloadedBytes, true, false)
	return backend.StepDispatchResult{
		Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, outShape, outHost)},
		VariantEntry: compiled.Entry,
		SourceHash:   compiled.SourceHash,
		Metadata: map[string]any{
			"dispatch_mode":    "backend_native",
			"execution_mode":   "cuda_device",
			"device_execution": true,
			"launch_api":       "cublasSgemm",
			"launch_compiler":  "cublas",
			"backend_library":  "cublas",
			"transpose_left":   transposeLeft,
			"transpose_right":  transposeRight,
			"lhs_binding":      leftName,
			"lhs_residency":    "device_resident",
		},
	}, nil
}

func (rt *deviceRuntime) runMatMulWithTranspose(inputs []*backend.Tensor, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	if len(inputs) != 2 {
		return backend.StepDispatchResult{}, fmt.Errorf("matmul expects 2 inputs, got %d", len(inputs))
	}
	batches, rows, inner, cols, rhsBatched, outShape, err := matmulLayout(inputs[0], inputs[1])
	if transposeLeft || transposeRight {
		batches, rows, inner, cols, rhsBatched, outShape, err = matmulLayoutWithTranspose(inputs[0], inputs[1], transposeLeft, transposeRight)
	}
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	runStart := time.Now()
	runUploadedBytes := int64((len(inputs[0].F32) + len(inputs[1].F32)) * 4)
	runDownloadedBytes := int64(batches * rows * cols * 4)
	compiled := cudaBuiltinMatMulCompiledKernel()
	launchAPI := "cublasSgemm"
	outHost := make([]float32, batches*rows*cols)
	if !rhsBatched {
		lhsBuf, err := rt.uploadMatMulScratchFloat32("matmul_lhs", inputs[0].F32)
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		rhsBuf, err := rt.uploadMatMulScratchFloat32("matmul_rhs", inputs[1].F32)
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		outBuf, err := rt.matMulScratchFloat32("matmul_out", len(outHost))
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		lhsRows, lhsCols := batches*rows, inner
		rhsRows, rhsCols := inner, cols
		if transposeLeft || transposeRight {
			lhsRows, lhsCols = matrixShape(inputs[0])
			rhsRows, rhsCols = matrixShape(inputs[1])
		}
		if err := rt.matMulCublas(lhsBuf, rhsBuf, outBuf, C.int(lhsRows), C.int(lhsCols), C.int(rhsRows), C.int(rhsCols), transposeLeft, transposeRight); err != nil {
			return backend.StepDispatchResult{}, err
		}
		if err := rt.downloadFloat32(outHost, outBuf); err != nil {
			return backend.StepDispatchResult{}, err
		}
	} else {
		lhsRows, lhsCols := rows, inner
		rhsRows, rhsCols := inner, cols
		if transposeLeft || transposeRight {
			lhsRows, lhsCols = matrixShape(inputs[0])
			rhsRows, rhsCols = matrixShape(inputs[1])
		}
		lhsBuf, err := rt.uploadMatMulScratchFloat32("matmul_lhs", inputs[0].F32)
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		rhsBuf, err := rt.uploadMatMulScratchFloat32("matmul_rhs", inputs[1].F32)
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		outBuf, err := rt.matMulScratchFloat32("matmul_out", len(outHost))
		if err != nil {
			return backend.StepDispatchResult{}, err
		}
		if err := rt.matMulCublasStridedBatched(lhsBuf, rhsBuf, outBuf, C.int(batches), C.int(lhsRows), C.int(lhsCols), C.int(rhsRows), C.int(rhsCols), transposeLeft, transposeRight); err != nil {
			return backend.StepDispatchResult{}, err
		}
		if err := rt.downloadFloat32(outHost, outBuf); err != nil {
			return backend.StepDispatchResult{}, err
		}
		launchAPI = "cublasSgemmStridedBatched"
	}
	rt.recordMatMulRun(runStart, runUploadedBytes, runDownloadedBytes, false, false)
	return backend.StepDispatchResult{
		Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, outShape, outHost)},
		VariantEntry: compiled.Entry,
		SourceHash:   compiled.SourceHash,
		Metadata: map[string]any{
			"dispatch_mode":    "backend_native",
			"execution_mode":   "cuda_device",
			"device_execution": true,
			"launch_api":       launchAPI,
			"launch_compiler":  "cublas",
			"backend_library":  "cublas",
			"transpose_left":   transposeLeft,
			"transpose_right":  transposeRight,
		},
	}, nil
}

func cudaBuiltinMatMulCompiledKernel() backend.CompiledKernel {
	return backend.CompiledKernel{
		Name:       "__builtin_matmul",
		Backend:    eosartifact.BackendCUDA,
		Entry:      "cublas_sgemm",
		Source:     "library:cublas_sgemm",
		SourceHash: "library:cublas_sgemm",
		Meta: map[string]string{
			"library":      "cublas",
			"vector_width": "backend_selected",
			"memory":       "device_local",
		},
	}
}

func matmulLayout(lhs, rhs *backend.Tensor) (batches, rows, inner, cols int, rhsBatched bool, outShape []int, err error) {
	if lhs == nil || rhs == nil {
		return 0, 0, 0, 0, false, nil, fmt.Errorf("nil matmul input")
	}
	switch len(lhs.Shape) {
	case 2:
		if len(rhs.Shape) != 2 {
			return 0, 0, 0, 0, false, nil, fmt.Errorf("matmul rank-2 lhs requires rank-2 rhs tensor")
		}
		cols = rhs.Shape[1]
		if lhs.Shape[1] != rhs.Shape[0] {
			return 0, 0, 0, 0, false, nil, fmt.Errorf("matmul mismatch %v x %v", lhs.Shape, rhs.Shape)
		}
		return 1, lhs.Shape[0], lhs.Shape[1], cols, false, []int{lhs.Shape[0], rhs.Shape[1]}, nil
	case 3:
		switch len(rhs.Shape) {
		case 2:
			cols = rhs.Shape[1]
			if lhs.Shape[2] != rhs.Shape[0] {
				return 0, 0, 0, 0, false, nil, fmt.Errorf("matmul mismatch %v x %v", lhs.Shape, rhs.Shape)
			}
			return lhs.Shape[0], lhs.Shape[1], lhs.Shape[2], cols, false, []int{lhs.Shape[0], lhs.Shape[1], rhs.Shape[1]}, nil
		case 3:
			cols = rhs.Shape[2]
			if lhs.Shape[0] != rhs.Shape[0] || lhs.Shape[2] != rhs.Shape[1] {
				return 0, 0, 0, 0, false, nil, fmt.Errorf("matmul mismatch %v x %v", lhs.Shape, rhs.Shape)
			}
			return lhs.Shape[0], lhs.Shape[1], lhs.Shape[2], cols, true, []int{lhs.Shape[0], lhs.Shape[1], rhs.Shape[2]}, nil
		default:
			return 0, 0, 0, 0, false, nil, fmt.Errorf("matmul rank-3 lhs requires rank-2 or rank-3 rhs tensor")
		}
	default:
		return 0, 0, 0, 0, false, nil, fmt.Errorf("matmul expects rank-2 or rank-3 lhs tensor")
	}
}

func matmulLayoutWithTranspose(lhs, rhs *backend.Tensor, transposeLeft, transposeRight bool) (batches, rows, inner, cols int, rhsBatched bool, outShape []int, err error) {
	if lhs == nil || rhs == nil {
		return 0, 0, 0, 0, false, nil, fmt.Errorf("nil matmul input")
	}
	switch len(lhs.Shape) {
	case 2:
		if len(rhs.Shape) != 2 {
			return 0, 0, 0, 0, false, nil, fmt.Errorf("transposed rank-2 lhs requires rank-2 rhs tensor")
		}
		rows = lhs.Shape[0]
		inner = lhs.Shape[1]
		if transposeLeft {
			rows, inner = inner, rows
		}
		rhsInner := rhs.Shape[0]
		cols = rhs.Shape[1]
		if transposeRight {
			rhsInner, cols = cols, rhsInner
		}
		if inner != rhsInner {
			return 0, 0, 0, 0, false, nil, fmt.Errorf("matmul mismatch %v x %v with transpose_left=%t transpose_right=%t", lhs.Shape, rhs.Shape, transposeLeft, transposeRight)
		}
		return 1, rows, inner, cols, false, []int{rows, cols}, nil
	case 3:
		if len(rhs.Shape) != 3 {
			return 0, 0, 0, 0, false, nil, fmt.Errorf("transposed rank-3 lhs requires rank-3 rhs tensor")
		}
		if lhs.Shape[0] != rhs.Shape[0] {
			return 0, 0, 0, 0, false, nil, fmt.Errorf("matmul batch mismatch %v x %v", lhs.Shape, rhs.Shape)
		}
		rows = lhs.Shape[1]
		inner = lhs.Shape[2]
		if transposeLeft {
			rows, inner = inner, rows
		}
		rhsInner := rhs.Shape[1]
		cols = rhs.Shape[2]
		if transposeRight {
			rhsInner, cols = cols, rhsInner
		}
		if inner != rhsInner {
			return 0, 0, 0, 0, false, nil, fmt.Errorf("matmul mismatch %v x %v with transpose_left=%t transpose_right=%t", lhs.Shape, rhs.Shape, transposeLeft, transposeRight)
		}
		return lhs.Shape[0], rows, inner, cols, true, []int{lhs.Shape[0], rows, cols}, nil
	default:
		return 0, 0, 0, 0, false, nil, fmt.Errorf("transposed matmul expects rank-2 or rank-3 lhs tensor")
	}
}

func matrixShape(t *backend.Tensor) (rows, cols int) {
	if t == nil || len(t.Shape) < 2 {
		return 0, 0
	}
	return t.Shape[len(t.Shape)-2], t.Shape[len(t.Shape)-1]
}

func sameIntSlice(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func newStepOutputTensor(outputType eosartifact.ValueType, shape []int, data []float32) *backend.Tensor {
	dtype := "f16"
	if outputType.Tensor != nil && outputType.Tensor.DType != "" {
		dtype = outputType.Tensor.DType
	}
	switch dtype {
	case "f32":
		return &backend.Tensor{DType: "f32", Shape: append([]int(nil), shape...), F32: data}
	default:
		return &backend.Tensor{DType: "f16", Shape: append([]int(nil), shape...), F32: data}
	}
}

const (
	cudaReductionBlockSize = 256
	cudaMaxReductionBlocks = 1024
)

func reductionLaunchConfig(elements int) (grid, block uint) {
	if elements <= 0 {
		return 1, cudaReductionBlockSize
	}
	blocks := (elements + cudaReductionBlockSize - 1) / cudaReductionBlockSize
	if blocks < 1 {
		blocks = 1
	}
	if blocks > cudaMaxReductionBlocks {
		blocks = cudaMaxReductionBlocks
	}
	return uint(blocks), cudaReductionBlockSize
}

type turboQDeviceSpec struct {
	perm       C.CUdeviceptr
	signs1     C.CUdeviceptr
	signs2     C.CUdeviceptr
	centroids  C.CUdeviceptr
	boundaries C.CUdeviceptr
}

func (rt *deviceRuntime) uploadTurboQSpec(cfg cudaTurboQConfig) (*turboQDeviceSpec, error) {
	q := turboquant.NewHadamardWithSeed(cfg.channels, cfg.bits, cfg.seed)
	spec := q.Spec()
	if spec.RotationKind != "hadamard" {
		return nil, fmt.Errorf("cuda turboquant requires hadamard rotation spec")
	}
	levels := 1 << uint(cfg.bits)
	if len(spec.Perm) != cfg.channels || len(spec.Signs1) != cfg.channels || len(spec.Signs2) != cfg.channels {
		return nil, fmt.Errorf("cuda turboquant hadamard spec shape mismatch")
	}
	if len(spec.Centroids) != levels || len(spec.Boundaries) != levels-1 {
		return nil, fmt.Errorf("cuda turboquant codebook shape mismatch")
	}
	deviceSpec := &turboQDeviceSpec{}
	var err error
	deviceSpec.perm, err = rt.uploadFloat32(intsToFloat32(spec.Perm))
	if err != nil {
		deviceSpec.free(rt)
		return nil, err
	}
	deviceSpec.signs1, err = rt.uploadFloat32(spec.Signs1)
	if err != nil {
		deviceSpec.free(rt)
		return nil, err
	}
	deviceSpec.signs2, err = rt.uploadFloat32(spec.Signs2)
	if err != nil {
		deviceSpec.free(rt)
		return nil, err
	}
	deviceSpec.centroids, err = rt.uploadFloat32(spec.Centroids)
	if err != nil {
		deviceSpec.free(rt)
		return nil, err
	}
	deviceSpec.boundaries, err = rt.uploadFloat32(spec.Boundaries)
	if err != nil {
		deviceSpec.free(rt)
		return nil, err
	}
	return deviceSpec, nil
}

func (spec *turboQDeviceSpec) free(rt *deviceRuntime) {
	if spec == nil || rt == nil {
		return
	}
	_ = rt.freeBuffer(spec.perm)
	_ = rt.freeBuffer(spec.signs1)
	_ = rt.freeBuffer(spec.signs2)
	_ = rt.freeBuffer(spec.centroids)
	_ = rt.freeBuffer(spec.boundaries)
}

func intsToFloat32(values []int) []float32 {
	out := make([]float32, len(values))
	for i, value := range values {
		out[i] = float32(value)
	}
	return out
}

func turboQEncodeStepResult(cfg cudaTurboQConfig, coordsShape []int, coords []float32, normShape []int, norms []float32) backend.StepDispatchResult {
	return backend.StepDispatchResult{
		Outputs: []*backend.Tensor{
			newTurboQCoordsTensor(cfg.bits, coordsShape, coords),
			backend.NewTensorQNorm(normShape, norms),
		},
		VariantEntry: "__builtin_cuda_turboquant_encode",
		Metadata:     turboQStepMetadata("turboquant_encode", cfg),
	}
}

func turboQDecodeStepResult(outputType eosartifact.ValueType, cfg cudaTurboQConfig, shape []int, data []float32) backend.StepDispatchResult {
	return backend.StepDispatchResult{
		Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, shape, data)},
		VariantEntry: "__builtin_cuda_turboquant_decode",
		Metadata:     turboQStepMetadata("turboquant_decode", cfg),
	}
}

func newTurboQCoordsTensor(bits int, shape []int, data []float32) *backend.Tensor {
	switch bits {
	case 2:
		return backend.NewTensorQ2(shape, data)
	case 4:
		return backend.NewTensorQ4(shape, data)
	case 8:
		return backend.NewTensorQ8(shape, data)
	default:
		return backend.NewTensorF32(shape, data)
	}
}

func turboQStepMetadata(op string, cfg cudaTurboQConfig) map[string]any {
	return map[string]any{
		"dispatch_mode":    "backend_step",
		"device_execution": true,
		"execution_mode":   "cuda_device",
		"launch_api":       "cuda_driver",
		"launch_compiler":  "nvrtc",
		"op":               op,
		"bits":             cfg.bits,
		"seed":             cfg.seed,
		"rotation_kind":    "hadamard",
	}
}

func sparseAttentionOutputShape(cfg cudaSparseAttentionConfig) []int {
	if cfg.rank == 3 {
		return []int{cfg.batches, cfg.queryLen, cfg.valueDim}
	}
	return []int{cfg.queryLen, cfg.valueDim}
}

func sparseAttentionStepResult(outputType eosartifact.ValueType, cfg cudaSparseAttentionConfig, shape []int, data []float32) backend.StepDispatchResult {
	plan := backend.PlanSparseAttention(backend.SparseAttentionPlanInput{
		QueryLen: cfg.queryLen,
		KeyLen:   cfg.keyLen,
		QueryDim: cfg.queryDim,
		ValueDim: cfg.valueDim,
		TopK:     cfg.topK,
	})
	metadata := map[string]any{
		"dispatch_mode":    "backend_step",
		"device_execution": true,
		"execution_mode":   "cuda_device",
		"launch_api":       "cuda_driver",
		"launch_compiler":  "nvrtc",
		"op":               "sparse_attention",
	}
	for key, value := range plan.Metadata() {
		metadata[key] = value
	}
	return backend.StepDispatchResult{
		Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, shape, data)},
		VariantEntry: "__builtin_cuda_sparse_attention",
		Metadata:     metadata,
	}
}

func turboSparseAttentionStepResult(outputType eosartifact.ValueType, cfg cudaTurboSparseAttentionConfig, shape []int, data []float32) backend.StepDispatchResult {
	plan := backend.PlanSparseAttention(backend.SparseAttentionPlanInput{
		QueryLen:       cfg.sparse.queryLen,
		KeyLen:         cfg.sparse.keyLen,
		QueryDim:       cfg.sparse.queryDim,
		ValueDim:       cfg.sparse.valueDim,
		TopK:           cfg.sparse.topK,
		RouteBlockSize: cfg.routeBlockSize,
		RouteTopBlocks: cfg.routeTopBlocks,
	})
	metadata := map[string]any{
		"dispatch_mode":         "backend_step",
		"device_execution":      true,
		"execution_mode":        "cuda_device",
		"launch_api":            "cuda_driver",
		"launch_compiler":       "nvrtc",
		"op":                    "turbo_sparse_attention",
		"bits":                  cfg.key.bits,
		"seed":                  cfg.key.seed,
		"rotation_kind":         "hadamard",
		"kv_decode":             "cuda_turboquant_inline",
		"dense_kv_materialized": false,
		"fused":                 true,
		"scratch_scope":         "query_row",
	}
	for key, value := range plan.Metadata() {
		metadata[key] = value
	}
	result := backend.StepDispatchResult{
		Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, shape, data)},
		VariantEntry: "__builtin_cuda_turbo_sparse_attention",
		Metadata:     metadata,
	}
	if cfg.routeBlockSize > 0 && cfg.routeTopBlocks > 0 {
		result.Metadata["routing"] = "block_anchor"
	} else {
		result.Metadata["routing"] = "exact"
	}
	return result
}

func cudaConvStepResult(outputType eosartifact.ValueType, variant, op string, shape []int, data []float32, extra map[string]any) backend.StepDispatchResult {
	metadata := map[string]any{
		"dispatch_mode":    "backend_step",
		"device_execution": true,
		"execution_mode":   "cuda_device",
		"launch_api":       "cuda_driver",
		"launch_compiler":  "nvrtc",
		"op":               op,
	}
	for key, value := range extra {
		metadata[key] = value
	}
	return backend.StepDispatchResult{
		Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, shape, data)},
		VariantEntry: variant,
		Metadata:     metadata,
	}
}

func cfgMetadata(cfg cudaConv2DConfig) map[string]any {
	return map[string]any{
		"groups":     cfg.groups,
		"stride_h":   cfg.strideH,
		"stride_w":   cfg.strideW,
		"pad_h":      cfg.padH,
		"pad_w":      cfg.padW,
		"dilation_h": cfg.dilationH,
		"dilation_w": cfg.dilationW,
		"has_bias":   cfg.hasBias,
	}
}

func cfgTransposeMetadata(cfg cudaConv2DTransposeConfig) map[string]any {
	return map[string]any{
		"groups":           cfg.groups,
		"stride_h":         cfg.strideH,
		"stride_w":         cfg.strideW,
		"pad_h":            cfg.padH,
		"pad_w":            cfg.padW,
		"dilation_h":       cfg.dilationH,
		"dilation_w":       cfg.dilationW,
		"output_padding_h": cfg.outPadH,
		"output_padding_w": cfg.outPadW,
		"has_bias":         cfg.hasBias,
	}
}

func msSSIMLossFromMoments(sumA, sumB, sumAA, sumBB, sumAB float64, elements int) float64 {
	if elements <= 0 {
		return 0
	}
	denom := float64(elements)
	meanA := sumA / denom
	meanB := sumB / denom
	varA := sumAA/denom - meanA*meanA
	varB := sumBB/denom - meanB*meanB
	cov := sumAB/denom - meanA*meanB
	const c1 = 0.01 * 0.01
	const c2 = 0.03 * 0.03
	ssim := ((2*meanA*meanB + c1) * (2*cov + c2)) / ((meanA*meanA + meanB*meanB + c1) * (varA + varB + c2))
	if math.IsNaN(ssim) || math.IsInf(ssim, 0) {
		ssim = 0
	}
	loss := 1 - ssim
	if loss < 0 {
		return 0
	}
	if loss > 1 {
		return 1
	}
	return loss
}

func cudaScalarStepResult(outputType eosartifact.ValueType, variant, op, launchAPI string, value float32) backend.StepDispatchResult {
	return backend.StepDispatchResult{
		Outputs:      []*backend.Tensor{newStepOutputTensor(outputType, []int{1}, []float32{value})},
		VariantEntry: variant,
		Metadata: map[string]any{
			"dispatch_mode":    "backend_step",
			"device_execution": true,
			"execution_mode":   "cuda_device",
			"launch_api":       launchAPI,
			"launch_compiler":  "nvrtc",
			"op":               op,
		},
	}
}

func validateCrossEntropyPlanInputs(codes, logits *backend.Tensor, plan cudaCrossEntropyPlan) error {
	if codes == nil {
		return fmt.Errorf("cuda cross_entropy expects codes")
	}
	switch plan.mode {
	case cudaCrossEntropyLogNormal:
		if logits == nil {
			return fmt.Errorf("cuda cross_entropy log_normal expects norm params")
		}
		if len(codes.Shape) != 3 || len(logits.Shape) != 4 {
			return fmt.Errorf("cuda cross_entropy log_normal expects codes NHW and params N2HW")
		}
		if logits.Shape[0] != codes.Shape[0] || logits.Shape[1] < 2 || logits.Shape[2] != codes.Shape[1] || logits.Shape[3] != codes.Shape[2] {
			return fmt.Errorf("cuda cross_entropy norm param shape %v does not match codes %v", logits.Shape, codes.Shape)
		}
		if len(logits.F32) < logits.Elements() {
			return fmt.Errorf("cuda cross_entropy norm params must be dense")
		}
		if err := validateLogNormalParams(logits, plan.sigmaMode); err != nil {
			return err
		}
	case cudaCrossEntropyBitPlane:
		if plan.bits <= 0 {
			return fmt.Errorf("cuda cross_entropy bit-plane mode requires bits")
		}
		if plan.layout == cudaCrossEntropyLayoutNCHW && !supportsNCHWBitPair(codes, logits, plan.bits) {
			return fmt.Errorf("cuda cross_entropy bit-plane NCHW logits shape %v does not match codes %v", logitsShape(logits), codes.Shape)
		}
	case cudaCrossEntropyCategorical:
		if plan.levels <= 0 {
			return fmt.Errorf("cuda cross_entropy levels must be positive")
		}
		if plan.layout == cudaCrossEntropyLayoutNCHW && !supportsNCHWAlphabet(codes, logits, plan.levels) {
			return fmt.Errorf("cuda cross_entropy NCHW logits shape %v does not match codes %v", logitsShape(logits), codes.Shape)
		}
	default:
		return fmt.Errorf("cuda cross_entropy unsupported mode %d", plan.mode)
	}
	return nil
}

func validateLogNormalParams(params *backend.Tensor, sigmaMode cudaSigmaMode) error {
	channels, height, width := params.Shape[1], params.Shape[2], params.Shape[3]
	for n := 0; n < params.Shape[0]; n++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				muOffset := ((n*channels+0)*height+y)*width + x
				sigmaOffset := ((n*channels+1)*height+y)*width + x
				mu := params.F32[muOffset]
				rawSigma := params.F32[sigmaOffset]
				if math.IsNaN(float64(mu)) || math.IsInf(float64(mu), 0) {
					return fmt.Errorf("cuda cross_entropy invalid norm params at offset %d", muOffset)
				}
				if math.IsNaN(float64(rawSigma)) || math.IsInf(float64(rawSigma), 0) || (sigmaMode == cudaSigmaRaw && rawSigma <= 0) {
					return fmt.Errorf("cuda cross_entropy invalid norm params at offset %d", sigmaOffset)
				}
			}
		}
	}
	return nil
}

func cudaTensorShapeArgs(tensor *backend.Tensor) (rank, n, c, h, w int) {
	if tensor == nil {
		return 0, 0, 0, 0, 0
	}
	switch len(tensor.Shape) {
	case 4:
		return 4, tensor.Shape[0], tensor.Shape[1], tensor.Shape[2], tensor.Shape[3]
	case 3:
		return 3, tensor.Shape[0], 1, tensor.Shape[1], tensor.Shape[2]
	case 2:
		return 2, 1, tensor.Shape[0], 1, tensor.Shape[1]
	case 1:
		return 1, 1, 1, 1, tensor.Shape[0]
	default:
		width := tensor.Elements()
		if width < 1 {
			width = 1
		}
		return len(tensor.Shape), 1, 1, 1, width
	}
}

func logitsShape(tensor *backend.Tensor) []int {
	if tensor == nil {
		return nil
	}
	return tensor.Shape
}

func cudaCrossEntropyModeName(mode cudaCrossEntropyMode) string {
	switch mode {
	case cudaCrossEntropyCategorical:
		return "categorical"
	case cudaCrossEntropyBitPlane:
		return "bit-plane"
	case cudaCrossEntropyLogNormal:
		return "log_normal"
	default:
		return "unknown"
	}
}

func cudaCrossEntropyLayoutName(layout cudaCrossEntropyLayout) string {
	switch layout {
	case cudaCrossEntropyLayoutUniform:
		return "uniform"
	case cudaCrossEntropyLayoutGlobal:
		return "global"
	case cudaCrossEntropyLayoutFlat:
		return "flat"
	case cudaCrossEntropyLayoutNCHW:
		return "nchw"
	case cudaCrossEntropyLayoutSigmoidFallback:
		return "sigmoid_fallback"
	default:
		return "unknown"
	}
}

func cudaSigmaModeName(mode cudaSigmaMode) string {
	switch mode {
	case cudaSigmaRaw:
		return "raw"
	case cudaSigmaSoftplus:
		return "softplus"
	case cudaSigmaExp:
		return "exp"
	default:
		return "unknown"
	}
}

func firstIntValue(value any, fallback int) int {
	switch v := value.(type) {
	case int:
		return v
	case int32:
		return int(v)
	case int64:
		return int(v)
	case float64:
		return int(v)
	case string:
		var n int
		_, err := fmt.Sscanf(v, "%d", &n)
		if err == nil && n > 0 {
			return n
		}
	}
	return fallback
}
