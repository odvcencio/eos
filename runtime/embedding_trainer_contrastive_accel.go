package eosruntime

import (
	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

var newTrainerContrastiveAccelerator = defaultTrainerContrastiveAccelerator

func defaultTrainerContrastiveAccelerator() (backend.ContrastiveAccelerator, eosartifact.BackendKind, error) {
	return backend.NewPreferredContrastiveAccelerator(eosartifact.BackendCUDA, eosartifact.BackendMetal)
}
