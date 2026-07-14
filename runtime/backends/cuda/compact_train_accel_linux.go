//go:build linux && cgo

package cuda

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#include <cuda.h>
#include <stdint.h>
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
	arena        *compactTrainArena
	nextHandleID uint64
	closed       bool
}

type compactTrainArena struct {
	shape      backend.CompactForwardShape
	generation uint64
	live       bool
	token      *compactTrainHandleToken

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
	modelScratch        []C.CUdeviceptr
	ffnScratch          []C.CUdeviceptr
	layers              []compactTrainLayerArena
	bytes               int64
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
	backend    eosartifact.BackendKind
	generation uint64
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
func (t *compactTrainHandleToken) Alive() bool {
	return t != nil && t.alive.Load()
}

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
	return &CompactTrainAccelerator{CompactForwardAccelerator: base}, nil
}

func (a *CompactTrainAccelerator) Backend() eosartifact.BackendKind { return eosartifact.BackendCUDA }

func (a *CompactTrainAccelerator) Close() {
	if a == nil || a.CompactForwardAccelerator == nil {
		return
	}
	a.mu.Lock()
	a.closed = true
	a.releaseArenaLocked()
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
	return nil
}

func (a *CompactTrainAccelerator) BeginCompactTrainStep(uint64, []backend.CompactForwardResidentRef) error {
	return fmt.Errorf("cuda compact resident train backward is unsupported in this forward-handle slice")
}

func (a *CompactTrainAccelerator) EndCompactTrainStep(uint64) error {
	return fmt.Errorf("cuda compact resident train backward is unsupported in this forward-handle slice")
}

func (a *CompactTrainAccelerator) RunCompactTrainBackward(backend.CompactTrainBackwardRequest) (backend.CompactTrainBackwardResult, error) {
	return backend.CompactTrainBackwardResult{}, fmt.Errorf("cuda compact resident train backward is unsupported in this forward-handle slice")
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

func (a *CompactTrainAccelerator) runCompactTrainForwardLocked(req backend.CompactTrainForwardRequest) (backend.CompactTrainForwardResult, error) {
	if err := a.preflightCompactTrainForwardLocked(req); err != nil {
		return backend.CompactTrainForwardResult{}, err
	}
	if a.arena != nil && a.arena.live {
		return backend.CompactTrainForwardResult{}, fmt.Errorf("cuda compact train arena already has a live handle")
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
	arena.generation++
	shape := req.Shape
	geluFast, _ := compactForwardGELUFast(req.GELUMode)
	launchesBefore := a.CompactForwardAccelerator.stats.KernelLaunches
	syncsBefore := a.CompactForwardAccelerator.stats.KernelSynchronizations
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
	uploaded := int64((len(tokensFlat) + len(masksFlat) + len(req.Roles) + 1) * 4)
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
	status := []int32{0}
	if err := a.device.downloadInt32(status, arena.status); err != nil {
		return backend.CompactTrainForwardResult{}, err
	}
	if status[0] != 0 {
		return backend.CompactTrainForwardResult{}, fmt.Errorf("cuda compact train gather status %d", status[0])
	}
	pooled := make([]float32, B*O)
	if err := a.device.downloadFloat32(pooled, arena.finalPooled); err != nil {
		return backend.CompactTrainForwardResult{}, err
	}
	activeCounts := make([]int32, B)
	if err := a.device.downloadInt32(activeCounts, arena.active); err != nil {
		return backend.CompactTrainForwardResult{}, err
	}
	token := &compactTrainHandleToken{backend: eosartifact.BackendCUDA, generation: arena.generation, id: a.nextHandleID + 1}
	token.alive.Store(true)
	a.nextHandleID = token.id
	arena.token = token
	arena.live = true
	pooledBytes := int64(len(pooled) * 4)
	statusBytes := int64(4)
	activeBytes := int64(len(activeCounts) * 4)
	_, packedFloats, _ := compactForwardPackedLayout(shape)
	a.stats.HandlesCreated++
	a.stats.LiveHandles++
	a.stats.UploadedBytes += uploaded
	a.stats.DownloadedBytes += pooledBytes + statusBytes + activeBytes
	a.stats.PooledDownloadedBytes += pooledBytes
	a.stats.StatusDownloadedBytes += statusBytes + activeBytes
	a.stats.PackedBytesAvoided += int64(packedFloats * 4)
	forwardLaunches := a.CompactForwardAccelerator.stats.KernelLaunches - launchesBefore
	forwardSyncs := a.CompactForwardAccelerator.stats.KernelSynchronizations - syncsBefore
	a.stats.KernelLaunches += forwardLaunches
	a.stats.KernelSynchronizations += forwardSyncs
	a.stats.LastShape = shape
	a.stats.LastForwardLaunches = forwardLaunches
	a.stats.LastForwardSyncs = forwardSyncs
	a.stats.ActivationArenaBytes = arena.bytes
	a.stats.WorkspaceArenaBytes = 0
	return backend.CompactTrainForwardResult{
		Handle: backend.CompactTrainHandle{
			Backend:    eosartifact.BackendCUDA,
			Token:      token,
			Shape:      shape,
			Generation: token.generation,
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
	token, ok := handle.Token.(*compactTrainHandleToken)
	if !ok || token == nil {
		return fmt.Errorf("cuda compact train handle has invalid token")
	}
	if handle.Backend != eosartifact.BackendCUDA || token.Backend() != eosartifact.BackendCUDA {
		return fmt.Errorf("cuda compact train handle backend mismatch")
	}
	if a.arena == nil || a.arena.token != token || a.arena.generation != handle.Generation || token.Generation() != handle.Generation {
		return fmt.Errorf("cuda compact train handle is stale")
	}
	if !token.alive.CompareAndSwap(true, false) {
		return fmt.Errorf("cuda compact train handle already released")
	}
	a.arena.live = false
	a.stats.HandlesReleased++
	a.stats.LiveHandles--
	return nil
}

func (a *CompactTrainAccelerator) prepareArenaLocked(shape backend.CompactForwardShape) (*compactTrainArena, error) {
	if a.arena != nil && a.arena.shape == shape {
		return a.arena, nil
	}
	a.releaseArenaLocked()
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
	a.arena = arena
	a.stats.ActivationArenaBytes = arena.bytes
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

func (a *CompactTrainAccelerator) releaseArenaLocked() {
	if a.arena != nil {
		a.freeArena(a.arena)
		a.arena = nil
	}
	a.stats.ActivationArenaBytes = 0
	a.stats.LiveHandles = 0
}

func (a *CompactTrainAccelerator) freeArena(arena *compactTrainArena) {
	if a == nil || a.device == nil || arena == nil {
		return
	}
	for _, ptr := range []C.CUdeviceptr{arena.tokens, arena.masks, arena.roles, arena.status, arena.input, arena.active, arena.finalNorm, arena.outputRows, arena.preProjectionPooled, arena.finalPooled} {
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
