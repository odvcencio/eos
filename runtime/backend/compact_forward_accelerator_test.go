package backend

import (
	"fmt"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
)

type compactForwardAcceleratorForTest struct{}

func (compactForwardAcceleratorForTest) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (compactForwardAcceleratorForTest) RunCompactForward(CompactForwardRequest) (CompactForwardResult, error) {
	return CompactForwardResult{}, fmt.Errorf("compact forward test accelerator is not executable")
}

func isolateCompactForwardFactoriesForTest(t *testing.T, factories []compactForwardAcceleratorFactory) {
	t.Helper()
	prev := append([]compactForwardAcceleratorFactory(nil), compactForwardAcceleratorFactories...)
	compactForwardAcceleratorFactories = append([]compactForwardAcceleratorFactory(nil), factories...)
	t.Cleanup(func() {
		compactForwardAcceleratorFactories = prev
	})
}

func TestNewPreferredCompactForwardAcceleratorEmptyRegistryIsNoop(t *testing.T) {
	isolateCompactForwardFactoriesForTest(t, nil)

	accel, kind, err := NewPreferredCompactForwardAccelerator(eosartifact.BackendCUDA)
	if err != nil {
		t.Fatalf("empty registry error = %v, want nil", err)
	}
	if accel != nil || kind != "" {
		t.Fatalf("empty registry accel/kind = %T/%q, want nil/empty", accel, kind)
	}
}

func TestNewPreferredCompactForwardAcceleratorPropagatesRegisteredFactoryError(t *testing.T) {
	isolateCompactForwardFactoriesForTest(t, []compactForwardAcceleratorFactory{
		{
			kind: eosartifact.BackendCUDA,
			factory: func() (CompactForwardAccelerator, error) {
				return nil, fmt.Errorf("forced compact factory failure")
			},
		},
	})

	accel, kind, err := NewPreferredCompactForwardAccelerator(eosartifact.BackendCUDA)
	if err == nil {
		t.Fatalf("factory error = nil, accel/kind = %T/%q", accel, kind)
	}
	if !strings.Contains(err.Error(), "forced compact factory failure") {
		t.Fatalf("factory error = %v, want forced compact factory failure", err)
	}
	if accel != nil || kind != "" {
		t.Fatalf("failed factory accel/kind = %T/%q, want nil/empty", accel, kind)
	}
}

func TestNewPreferredCompactForwardAcceleratorFallsBackToLaterMatchingFactory(t *testing.T) {
	calls := 0
	isolateCompactForwardFactoriesForTest(t, []compactForwardAcceleratorFactory{
		{
			kind: eosartifact.BackendCUDA,
			factory: func() (CompactForwardAccelerator, error) {
				calls++
				return nil, fmt.Errorf("first compact factory unavailable")
			},
		},
		{
			kind: eosartifact.BackendCUDA,
			factory: func() (CompactForwardAccelerator, error) {
				calls++
				return compactForwardAcceleratorForTest{}, nil
			},
		},
	})

	accel, kind, err := NewPreferredCompactForwardAccelerator(eosartifact.BackendCUDA)
	if err != nil {
		t.Fatalf("later matching factory error = %v", err)
	}
	if calls != 2 {
		t.Fatalf("factory calls = %d, want first failure then second success", calls)
	}
	if accel == nil || kind != eosartifact.BackendCUDA {
		t.Fatalf("later matching factory accel/kind = %T/%q, want cuda accel", accel, kind)
	}
}
