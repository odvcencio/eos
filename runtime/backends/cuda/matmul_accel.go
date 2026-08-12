package cuda

import (
	"fmt"
	"sync"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

type matMulAccelerator struct {
	device  *deviceRuntime
	mu      sync.Mutex
	bridged map[string]*optimizerResidentParameterToken
}

func init() {
	backend.RegisterMatMulAccelerator(eosartifact.BackendCUDA, NewMatMulAccelerator)
}

// NewMatMulAccelerator exposes the CUDA backend's library-backed matmul fast path.
func NewMatMulAccelerator() (backend.MatMulAccelerator, error) {
	device, err := newDeviceRuntime()
	if err != nil {
		return nil, err
	}
	if device == nil {
		return nil, nil
	}
	return &matMulAccelerator{device: device, bridged: map[string]*optimizerResidentParameterToken{}}, nil
}

func (a *matMulAccelerator) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (a *matMulAccelerator) RunMatMul(inputs []*backend.Tensor, outputType eosartifact.ValueType) (backend.StepDispatchResult, error) {
	if a == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	return a.device.runMatMul(inputs, outputType)
}

func (a *matMulAccelerator) RunMatMulWithTranspose(inputs []*backend.Tensor, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	if a == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	return a.device.runMatMulWithTranspose(inputs, outputType, transposeLeft, transposeRight)
}

func (a *matMulAccelerator) BindMatrix(name string, tensor *backend.Tensor) error {
	if a == nil {
		return fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	if a.bridged[name] != nil {
		delete(a.device.residentMatrices, name)
		delete(a.bridged, name)
	}
	return a.device.bindMatMulRight(name, tensor)
}

func (a *matMulAccelerator) UnbindMatrix(name string) error {
	if a == nil {
		return fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	if a.bridged[name] != nil {
		delete(a.device.residentMatrices, name)
		delete(a.bridged, name)
		return nil
	}
	return a.device.unbindMatMulRight(name)
}

func (a *matMulAccelerator) RunMatMulWithBoundLeft(leftName string, rhs *backend.Tensor, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	if a == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	unlocks, err := a.lockResidentBridgeTokens([]string{leftName})
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer unlockResidentBridgeTokens(unlocks)
	return a.device.runMatMulWithBoundLeft(leftName, rhs, outputType, transposeLeft, transposeRight)
}

func (a *matMulAccelerator) RunMatMulWithBoundRight(lhs *backend.Tensor, rightName string, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	if a == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	unlocks, err := a.lockResidentBridgeTokens([]string{rightName})
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer unlockResidentBridgeTokens(unlocks)
	return a.device.runMatMulWithBoundRight(lhs, rightName, outputType, transposeLeft, transposeRight)
}

func (a *matMulAccelerator) RunMatMulWithBoundRights(lhs *backend.Tensor, rightNames []string, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) ([]backend.StepDispatchResult, error) {
	if a == nil {
		return nil, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return nil, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	unlocks, err := a.lockResidentBridgeTokens(rightNames)
	if err != nil {
		return nil, err
	}
	defer unlockResidentBridgeTokens(unlocks)
	return a.device.runMatMulWithBoundRights(lhs, rightNames, outputType, transposeLeft, transposeRight)
}

func (a *matMulAccelerator) RunMatMulsWithSharedLeft(lhs *backend.Tensor, rhs []*backend.Tensor, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) ([]backend.StepDispatchResult, error) {
	if a == nil {
		return nil, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return nil, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	return a.device.runMatMulsWithSharedLeft(lhs, rhs, outputType, transposeLeft, transposeRight)
}

func (a *matMulAccelerator) RunAccumulatedMatMulsWithBoundRights(lhs []*backend.Tensor, rightNames []string, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (backend.StepDispatchResult, error) {
	if a == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return backend.StepDispatchResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	unlocks, err := a.lockResidentBridgeTokens(rightNames)
	if err != nil {
		return backend.StepDispatchResult{}, err
	}
	defer unlockResidentBridgeTokens(unlocks)
	return a.device.runAccumulatedMatMulsWithBoundRights(lhs, rightNames, outputType, transposeLeft, transposeRight)
}

// RunAttentionBlockResident implements backend.AttentionResidentAccelerator.
// See deviceRuntime.runAttentionBlockResident for the device-resident chain.
func (a *matMulAccelerator) RunAttentionBlockResident(req backend.AttentionResidentRequest) (backend.AttentionResidentResult, error) {
	if a == nil {
		return backend.AttentionResidentResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return backend.AttentionResidentResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	return a.device.runAttentionBlockResident(req)
}

// BeginAttentionResidentTrainStep implements backend.AttentionResidentTrainAccelerator.
func (a *matMulAccelerator) BeginAttentionResidentTrainStep(stepID uint64) error {
	if a == nil {
		return fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	return a.device.beginAttentionResidentTrainStep(stepID)
}

// RunAttentionBlockResidentTrainForward implements backend.AttentionResidentTrainAccelerator.
// See deviceRuntime.runAttentionBlockResidentTrainForward for the device-resident chain.
func (a *matMulAccelerator) RunAttentionBlockResidentTrainForward(req backend.AttentionResidentTrainForwardRequest) (backend.AttentionResidentTrainForwardResult, error) {
	if a == nil {
		return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return backend.AttentionResidentTrainForwardResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	return a.device.runAttentionBlockResidentTrainForward(req)
}

// RunAttentionBlockResidentTrainBackward implements backend.AttentionResidentTrainAccelerator.
// See deviceRuntime.runAttentionBlockResidentTrainBackward for the device-resident chain.
func (a *matMulAccelerator) RunAttentionBlockResidentTrainBackward(req backend.AttentionResidentTrainBackwardRequest) (backend.AttentionResidentTrainBackwardResult, error) {
	if a == nil {
		return backend.AttentionResidentTrainBackwardResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return backend.AttentionResidentTrainBackwardResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	return a.device.runAttentionBlockResidentTrainBackward(req)
}

// EndAttentionResidentTrainStep implements backend.AttentionResidentTrainAccelerator.
func (a *matMulAccelerator) EndAttentionResidentTrainStep(stepID uint64) error {
	if a == nil {
		return fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	return a.device.endOrAbortAttentionResidentTrainStep(stepID)
}

// AbortAttentionResidentTrainStep implements backend.AttentionResidentTrainAccelerator.
func (a *matMulAccelerator) AbortAttentionResidentTrainStep(stepID uint64) error {
	if a == nil {
		return fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	return a.device.endOrAbortAttentionResidentTrainStep(stepID)
}

// ReleaseAttentionResidentTrainHandle implements backend.AttentionResidentTrainAccelerator.
func (a *matMulAccelerator) ReleaseAttentionResidentTrainHandle(handle backend.AttentionResidentTrainHandle) error {
	if a == nil {
		return fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	return a.device.releaseAttentionResidentTrainHandle(handle)
}

// RunFFNBlockResidentTrainForward implements backend.FFNResidentTrainAccelerator.
// See deviceRuntime.runFFNBlockResidentTrainForward for the device-resident
// chain. FFN handles share attention's Begin/End/AbortAttentionResidentTrainStep
// step lifecycle (see ffn_resident_train_accel_linux.go), so no separate
// Begin/End/Abort wrapper is needed here.
func (a *matMulAccelerator) RunFFNBlockResidentTrainForward(req backend.FFNResidentTrainForwardRequest) (backend.FFNResidentTrainForwardResult, error) {
	if a == nil {
		return backend.FFNResidentTrainForwardResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return backend.FFNResidentTrainForwardResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	return a.device.runFFNBlockResidentTrainForward(req)
}

// RunFFNBlockResidentTrainBackward implements backend.FFNResidentTrainAccelerator.
// See deviceRuntime.runFFNBlockResidentTrainBackward for the device-resident chain.
func (a *matMulAccelerator) RunFFNBlockResidentTrainBackward(req backend.FFNResidentTrainBackwardRequest) (backend.FFNResidentTrainBackwardResult, error) {
	if a == nil {
		return backend.FFNResidentTrainBackwardResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return backend.FFNResidentTrainBackwardResult{}, fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	return a.device.runFFNBlockResidentTrainBackward(req)
}

// ReleaseFFNResidentTrainHandle implements backend.FFNResidentTrainAccelerator.
func (a *matMulAccelerator) ReleaseFFNResidentTrainHandle(handle backend.FFNResidentTrainHandle) error {
	if a == nil {
		return fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	return a.device.releaseFFNResidentTrainHandle(handle)
}

func (a *matMulAccelerator) Stats() backend.MatMulAcceleratorStats {
	if a == nil {
		return backend.MatMulAcceleratorStats{}
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return backend.MatMulAcceleratorStats{}
	}
	return a.device.matMulStatsSnapshot()
}

func (a *matMulAccelerator) Close() {
	if a == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return
	}
	for name := range a.bridged {
		delete(a.device.residentMatrices, name)
		delete(a.bridged, name)
	}
	a.device.close()
	a.device = nil
}

func (a *matMulAccelerator) lockResidentBridgeTokens(names []string) ([]func(), error) {
	if len(names) == 0 || a == nil {
		return nil, nil
	}
	unlocks := make([]func(), 0, len(names))
	seen := map[string]bool{}
	for _, name := range names {
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		token := a.bridged[name]
		if token == nil {
			continue
		}
		state, unlock, err := token.lockCurrent()
		if err != nil {
			unlockResidentBridgeTokens(unlocks)
			return nil, err
		}
		if resident, ok := a.device.residentMatrices[name]; !ok || resident.ptr != state.param || resident.elements != state.elements {
			unlock()
			unlockResidentBridgeTokens(unlocks)
			return nil, fmt.Errorf("cuda matmul resident binding %q no longer matches optimizer token", name)
		}
		unlocks = append(unlocks, unlock)
	}
	return unlocks, nil
}

func unlockResidentBridgeTokens(unlocks []func()) {
	for i := len(unlocks) - 1; i >= 0; i-- {
		if unlocks[i] != nil {
			unlocks[i]()
		}
	}
}
