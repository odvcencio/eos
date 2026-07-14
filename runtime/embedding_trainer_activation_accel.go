package eosruntime

import (
	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

type trainerActivationAccelMode struct {
	fullBackward    bool
	softmaxBackward bool
}

func trainerActivationAccelModeFromEnv() trainerActivationAccelMode {
	if trainEnvFlagEnabled("EOS_TRAIN_DISABLE_ACTIVATION_ACCEL") {
		return trainerActivationAccelMode{}
	}
	full := trainEnvFlagEnabled("EOS_TRAIN_ENABLE_ACTIVATION_ACCEL")
	softmax := full || trainEnvFlagEnabled("EOS_TRAIN_ENABLE_SOFTMAX_BACKWARD_ACCEL")
	if trainEnvFlagEnabled("EOS_TRAIN_DISABLE_SOFTMAX_BACKWARD_ACCEL") {
		softmax = false
	}
	return trainerActivationAccelMode{
		fullBackward:    full,
		softmaxBackward: softmax,
	}
}

var newTrainerActivationAccelerator = defaultTrainerActivationAccelerator

func defaultTrainerActivationAccelerator() (backend.ActivationAccelerator, eosartifact.BackendKind, trainerActivationAccelMode, error) {
	mode := trainerActivationAccelModeFromEnv()
	if !mode.fullBackward && !mode.softmaxBackward {
		return nil, "", mode, nil
	}
	accel, backendKind, err := backend.NewPreferredActivationAccelerator(eosartifact.BackendCUDA, eosartifact.BackendMetal)
	return accel, backendKind, mode, err
}

func (t *EmbeddingTrainer) fullActivationBackwardAccelEnabled() bool {
	return t != nil && t.activationAccel != nil && t.activationAccelFull
}

func (t *EmbeddingTrainer) softmaxBackwardAccelEnabled() bool {
	return t != nil && t.activationAccel != nil && t.softmaxBackwardAccel
}
