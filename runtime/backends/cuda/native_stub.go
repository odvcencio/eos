//go:build !linux || !cgo

package cuda

import (
	"fmt"
	"sync"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

type deviceRuntime struct {
	bgeFullEncoderMu          sync.Mutex
	residentMatrices          map[string]residentMatrix
	bertResidentTensors       map[string]residentTensor
	bertResidentCache         map[string]bertCUDAResidentBindingCache
	bertSelectedContractCache map[string]bertCUDASelectedContract
}

type residentMatrix struct {
	ptr      uintptr
	rows     int
	cols     int
	elements int
}

type residentTensor struct {
	shape    []int
	elements int
}

type residentOptimizerState struct {
	param      uintptr
	elements   int
	generation uint64
}

type optimizerResidentParameterToken struct{}

func (t *optimizerResidentParameterToken) OptimizerResidentParameterToken() {}

func (t *optimizerResidentParameterToken) CompactForwardResidentToken() {}

func (t *optimizerResidentParameterToken) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (t *optimizerResidentParameterToken) Generation() uint64 {
	return 0
}

func (t *optimizerResidentParameterToken) Alive() bool {
	return false
}

func (t *optimizerResidentParameterToken) lockCurrent() (residentOptimizerState, func(), error) {
	return residentOptimizerState{}, nil, fmt.Errorf("cuda optimizer resident token is unavailable without linux cgo")
}

func newDeviceRuntime() (*deviceRuntime, error) {
	return nil, nil
}

func (rt *deviceRuntime) close() {}

func (rt *deviceRuntime) matMulStatsSnapshot() backend.MatMulAcceleratorStats {
	return backend.MatMulAcceleratorStats{}
}

func (rt *deviceRuntime) attachDeviceExecution(prog *backend.NativeKernelProgram, kernel eosartifact.Kernel) error {
	if prog.LaunchConfig == nil {
		prog.LaunchConfig = map[string]any{}
	}
	prog.LaunchConfig["device_execution"] = false
	prog.LaunchConfig["execution_mode"] = "host_fallback"
	return nil
}

func (rt *deviceRuntime) runMatMul(inputs []*backend.Tensor, outputType eosartifact.ValueType) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (rt *deviceRuntime) runMatMulWithTranspose(inputs []*backend.Tensor, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (rt *deviceRuntime) runGDNStep(inputs []*backend.Tensor, outputType eosartifact.ValueType, inverse bool) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (rt *deviceRuntime) runConv2DStep(inputs []*backend.Tensor, outputType eosartifact.ValueType, cfg cudaConv2DConfig) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (rt *deviceRuntime) runConv2DTransposeStep(inputs []*backend.Tensor, outputType eosartifact.ValueType, cfg cudaConv2DTransposeConfig) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (rt *deviceRuntime) runTurboQEncodeStep(inputs []*backend.Tensor, outputType eosartifact.ValueType, cfg cudaTurboQConfig) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (rt *deviceRuntime) runTurboQDecodeStep(inputs []*backend.Tensor, outputType eosartifact.ValueType, cfg cudaTurboQConfig) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (rt *deviceRuntime) runSparseAttentionStep(inputs []*backend.Tensor, outputType eosartifact.ValueType, cfg cudaSparseAttentionConfig) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (rt *deviceRuntime) runTurboSparseAttentionStep(inputs []*backend.Tensor, outputType eosartifact.ValueType, cfg cudaTurboSparseAttentionConfig) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (rt *deviceRuntime) runMSELossStep(inputs []*backend.Tensor, outputType eosartifact.ValueType) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (rt *deviceRuntime) runMSSSIMLossStep(inputs []*backend.Tensor, outputType eosartifact.ValueType) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (rt *deviceRuntime) runScalarAddStep(inputs []*backend.Tensor, outputType eosartifact.ValueType) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (rt *deviceRuntime) runRDLossStep(inputs []*backend.Tensor, outputType eosartifact.ValueType, lambda float32) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (rt *deviceRuntime) runCrossEntropyStep(inputs []*backend.Tensor, outputType eosartifact.ValueType, plan cudaCrossEntropyPlan) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (rt *deviceRuntime) bindMatMulRight(name string, tensor *backend.Tensor) error {
	return nil
}

func (rt *deviceRuntime) bindBERTResidentTensor(name string, tensor *backend.Tensor) error {
	return nil
}

func (rt *deviceRuntime) unbindMatMulRight(name string) error {
	return nil
}

func (rt *deviceRuntime) runMatMulWithBoundRight(lhs *backend.Tensor, rightName string, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (rt *deviceRuntime) runMatMulWithBoundRights(lhs *backend.Tensor, rightNames []string, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) ([]backend.StepDispatchResult, error) {
	return nil, nil
}

func (rt *deviceRuntime) runMatMulsWithSharedLeft(lhs *backend.Tensor, rhs []*backend.Tensor, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) ([]backend.StepDispatchResult, error) {
	return nil, nil
}

func (rt *deviceRuntime) runAccumulatedMatMulsWithBoundRights(lhs []*backend.Tensor, rightNames []string, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (rt *deviceRuntime) runMatMulWithBoundLeft(leftName string, rhs *backend.Tensor, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	return backend.StepDispatchResult{}, nil
}

func (rt *deviceRuntime) runBGEFullEncoderHidden(step eosartifact.Step, outputType eosartifact.ValueType, inputs []*backend.Tensor) (backend.StepDispatchResult, bertCUDAFullEncoderTransferStats, error) {
	return backend.StepDispatchResult{}, bertCUDAFullEncoderTransferStats{}, nil
}
