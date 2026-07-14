package eosruntime

import (
	"os"
	"strings"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

const compactPackedForwardEnv = "EOS_TRAIN_ENABLE_COMPACT_PACKED_FORWARD"
const compactResidentTrainEnv = "EOS_TRAIN_ENABLE_COMPACT_RESIDENT_TRAIN"

func compactPackedForwardEnabled() bool {
	return strings.TrimSpace(os.Getenv(compactPackedForwardEnv)) == "1"
}

func compactResidentTrainEnabled() bool {
	return strings.TrimSpace(os.Getenv(compactResidentTrainEnv)) == "1"
}

var newTrainerCompactForwardAccelerator = defaultTrainerCompactForwardAccelerator
var newTrainerCompactTrainAccelerator = defaultTrainerCompactTrainAccelerator

func defaultTrainerCompactForwardAccelerator() (backend.CompactForwardAccelerator, eosartifact.BackendKind, error) {
	if !compactPackedForwardEnabled() {
		return nil, "", nil
	}
	accel, kind, err := backend.NewPreferredCompactForwardAccelerator(eosartifact.BackendCUDA)
	if err != nil {
		return nil, "", err
	}
	if accel == nil {
		return nil, "", nil
	}
	return accel, kind, nil
}

func defaultTrainerCompactTrainAccelerator() (backend.CompactTrainAccelerator, eosartifact.BackendKind, error) {
	if !compactResidentTrainEnabled() {
		return nil, "", nil
	}
	accel, kind, err := backend.NewPreferredCompactTrainAccelerator(eosartifact.BackendCUDA)
	if err != nil {
		return nil, "", err
	}
	if accel == nil {
		return nil, "", nil
	}
	return accel, kind, nil
}
