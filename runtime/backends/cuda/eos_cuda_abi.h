#ifndef EOS_CUDA_ABI_H
#define EOS_CUDA_ABI_H

#include <cuda.h>
#include <cublas_v2.h>

// Optional CUDA-event control progress. Context counters describe the
// cuCtxSetCurrent attempt inside one wrapper; driver counters describe the
// event-create/record/query operation(s), including failed attempts.
typedef struct {
	int context_attempted;
	int context_succeeded;
	int driver_attempted;
	int driver_succeeded;
	int failure_stage;
} EosCudaProfileEventProgress;

// This header is the single source of truth for the CUDA runtime/kernel
// prefixes shared by the compact-forward, compact-train, and native cgo
// bridges. Keep field order and types stable: Go passes these pointers across
// translation units even though each bridge owns different helper functions.
typedef struct {
	CUcontext ctx;
	CUdevice device;
	int major;
	int minor;
	int primary_ctx;
	cublasHandle_t blas;
	CUstream stream;
	CUevent profile_k5_end;
	int profile_k5_end_recorded;
	EosCudaProfileEventProgress profile_k5_end_progress;
} EosCudaRuntime;

typedef struct {
	CUmodule module;
	CUfunction function;
} EosCudaKernel;

// Typed progress records are shared by the native bridge and the compact
// owner. Counts are prefixes: attempted_copies includes a failed driver copy,
// while completed_copies and device_copies include only successful copies.
typedef struct {
	int context_attempted;
	int context_succeeded;
	int attempted_copies;
	int completed_copies;
	int device_copies;
	int failure_stage;
	int status_copied;
	int pooled_copied;
	int active_copied;
	int status_value;
} EosCudaCompactTrainForwardReadbackProgress;

typedef struct {
	int context_attempted;
	int context_succeeded;
	int attempted_copies;
	int completed_copies;
	int device_copies;
	int failure_stage;
} EosCudaCompactTrainForwardInputUploadProgress;

#endif
