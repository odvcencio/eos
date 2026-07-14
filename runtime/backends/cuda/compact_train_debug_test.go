//go:build linux && cgo

package cuda

import (
	"fmt"

	"m31labs.dev/eos/runtime/backend"
)

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
