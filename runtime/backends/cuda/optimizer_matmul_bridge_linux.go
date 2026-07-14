//go:build linux && cgo

package cuda

import (
	"fmt"
	"time"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

func (a *matMulAccelerator) BindMatrixFromResident(name string, tensor *backend.Tensor, ref backend.OptimizerResidentParameter) error {
	if a == nil {
		return fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a == nil || a.device == nil {
		return fmt.Errorf("cuda matmul accelerator is not initialized")
	}
	if ref.Backend != eosartifact.BackendCUDA {
		return fmt.Errorf("cuda matmul resident binding %q has backend %q", name, ref.Backend)
	}
	token, ok := ref.Token.(*optimizerResidentParameterToken)
	if !ok || token == nil {
		return fmt.Errorf("cuda matmul resident binding %q has invalid optimizer token", name)
	}
	state, unlock, err := token.lockCurrent()
	if err != nil {
		return err
	}
	defer unlock()
	if state.param == 0 {
		return fmt.Errorf("cuda matmul resident binding %q has invalid device parameter", name)
	}
	if tensor == nil || len(tensor.Shape) != 2 {
		return fmt.Errorf("cuda matmul binding %q must be a rank-2 tensor", name)
	}
	if ref.Elements != len(tensor.F32) {
		return fmt.Errorf("cuda matmul resident binding %q elements %d does not match tensor size %d", name, ref.Elements, len(tensor.F32))
	}
	if existing, ok := a.device.residentMatrices[name]; ok && a.bridged[name] == nil {
		_ = a.device.freeBuffer(existing.ptr)
	}
	start := time.Now()
	a.device.residentMatrices[name] = residentMatrix{
		ptr:      state.param,
		rows:     tensor.Shape[0],
		cols:     tensor.Shape[1],
		elements: ref.Elements,
	}
	if a.bridged == nil {
		a.bridged = map[string]*optimizerResidentParameterToken{}
	}
	a.bridged[name] = token
	a.device.matMulStats.BindCalls++
	a.device.matMulStats.BindNanos += time.Since(start).Nanoseconds()
	a.device.matMulStats.BoundMatrices = int64(len(a.device.residentMatrices))
	return nil
}
