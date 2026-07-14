package backend

import (
	"fmt"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
)

type compactTrainAcceleratorForTest struct{}

func (compactTrainAcceleratorForTest) Backend() eosartifact.BackendKind {
	return eosartifact.BackendCUDA
}

func (compactTrainAcceleratorForTest) BeginCompactTrainStep(uint64, []CompactForwardResidentRef) error {
	return fmt.Errorf("compact train test accelerator is not executable")
}

func (compactTrainAcceleratorForTest) RunCompactTrainForward(CompactTrainForwardRequest) (CompactTrainForwardResult, error) {
	return CompactTrainForwardResult{}, fmt.Errorf("compact train test accelerator is not executable")
}

func (compactTrainAcceleratorForTest) RunCompactTrainBackward(CompactTrainBackwardRequest) (CompactTrainBackwardResult, error) {
	return CompactTrainBackwardResult{}, fmt.Errorf("compact train test accelerator is not executable")
}

func (compactTrainAcceleratorForTest) EndCompactTrainStep(uint64) error {
	return fmt.Errorf("compact train test accelerator is not executable")
}

func (compactTrainAcceleratorForTest) AbortCompactTrainStep(uint64) error {
	return fmt.Errorf("compact train test accelerator is not executable")
}

func (compactTrainAcceleratorForTest) ReleaseCompactTrainHandle(CompactTrainHandle) error {
	return fmt.Errorf("compact train test accelerator is not executable")
}

func isolateCompactTrainFactoriesForTest(t *testing.T, factories []compactTrainAcceleratorFactory) {
	t.Helper()
	prev := append([]compactTrainAcceleratorFactory(nil), compactTrainAcceleratorFactories...)
	compactTrainAcceleratorFactories = append([]compactTrainAcceleratorFactory(nil), factories...)
	t.Cleanup(func() {
		compactTrainAcceleratorFactories = prev
	})
}

func TestNewPreferredCompactTrainAcceleratorEmptyRegistryIsNoop(t *testing.T) {
	isolateCompactTrainFactoriesForTest(t, nil)

	accel, kind, err := NewPreferredCompactTrainAccelerator(eosartifact.BackendCUDA)
	if err != nil {
		t.Fatalf("empty registry error = %v, want nil", err)
	}
	if accel != nil || kind != "" {
		t.Fatalf("empty registry accel/kind = %T/%q, want nil/empty", accel, kind)
	}
}

func TestNewPreferredCompactTrainAcceleratorPropagatesRegisteredFactoryError(t *testing.T) {
	isolateCompactTrainFactoriesForTest(t, []compactTrainAcceleratorFactory{
		{
			kind: eosartifact.BackendCUDA,
			factory: func() (CompactTrainAccelerator, error) {
				return nil, fmt.Errorf("forced compact train factory failure")
			},
		},
	})

	accel, kind, err := NewPreferredCompactTrainAccelerator(eosartifact.BackendCUDA)
	if err == nil {
		t.Fatalf("factory error = nil, accel/kind = %T/%q", accel, kind)
	}
	if !strings.Contains(err.Error(), "forced compact train factory failure") {
		t.Fatalf("factory error = %v, want forced compact train factory failure", err)
	}
	if accel != nil || kind != "" {
		t.Fatalf("failed factory accel/kind = %T/%q, want nil/empty", accel, kind)
	}
}

func TestNewPreferredCompactTrainAcceleratorFallsBackToLaterMatchingFactory(t *testing.T) {
	calls := 0
	isolateCompactTrainFactoriesForTest(t, []compactTrainAcceleratorFactory{
		{
			kind: eosartifact.BackendCUDA,
			factory: func() (CompactTrainAccelerator, error) {
				calls++
				return nil, fmt.Errorf("first compact train factory unavailable")
			},
		},
		{
			kind: eosartifact.BackendCUDA,
			factory: func() (CompactTrainAccelerator, error) {
				calls++
				return compactTrainAcceleratorForTest{}, nil
			},
		},
	})

	accel, kind, err := NewPreferredCompactTrainAccelerator(eosartifact.BackendCUDA)
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
