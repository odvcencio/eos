package eos

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	prismvalidate "m31labs.dev/prism/validate"
)

// PrismBackend maps an Eos artifact backend to Prism's generic backend source
// family. The string values intentionally match, but this function keeps the
// public schemas decoupled.
func PrismBackend(kind BackendKind) (prismvalidate.Backend, bool) {
	switch kind {
	case BackendCUDA:
		return prismvalidate.BackendCUDA, true
	case BackendMetal:
		return prismvalidate.BackendMetal, true
	case BackendVulkan:
		return prismvalidate.BackendVulkan, true
	case BackendDirectML:
		return prismvalidate.BackendDirectML, true
	case BackendWebGPU:
		return prismvalidate.BackendWebGPU, true
	default:
		return "", false
	}
}

// PrismSourceForKernelVariant converts an Eos kernel variant to Prism's
// backend-neutral validation descriptor.
func PrismSourceForKernelVariant(kernelName string, variant KernelVariant) (prismvalidate.Source, error) {
	backend, ok := PrismBackend(variant.Backend)
	if !ok {
		return prismvalidate.Source{}, fmt.Errorf("unsupported backend %q", variant.Backend)
	}
	return prismvalidate.Source{
		Name:    kernelName,
		Backend: backend,
		Entry:   variant.Entry,
		Source:  variant.Source,
	}, nil
}

// ValidateKernelVariantSource checks that a kernel variant's source is shaped
// for its declared backend and entrypoint.
func ValidateKernelVariantSource(kernelName string, variant KernelVariant) error {
	src, err := PrismSourceForKernelVariant(kernelName, variant)
	if err != nil {
		return err
	}
	return prismvalidate.CheckSource(src)
}

// ValidateKernelVariant validates source plus any typed ABI/offline image
// attached to a variant. Runtime loaders use this single gate so stale source,
// ABI, or binary metadata cannot silently reach a native driver.
func ValidateKernelVariant(kernelName string, variant KernelVariant) error {
	if err := ValidateKernelVariantSource(kernelName, variant); err != nil {
		return err
	}
	if err := validateKernelABI(kernelName, variant); err != nil {
		return err
	}
	if variant.ABI != nil {
		sum := sha256.Sum256([]byte(variant.Source))
		if !strings.EqualFold(variant.ABI.SourceSHA256, hex.EncodeToString(sum[:])) {
			return fmt.Errorf("kernel %q variant ABI source hash does not match source", kernelName)
		}
	}
	return validateKernelBinary(kernelName, variant)
}

// ValidateKernelSources checks every backend-specific kernel source in mod.
func ValidateKernelSources(mod *Module) error {
	if mod == nil {
		return fmt.Errorf("nil module")
	}
	for _, kernel := range mod.Kernels {
		for _, variant := range kernel.Variants {
			if err := ValidateKernelVariantSource(kernel.Name, variant); err != nil {
				return err
			}
		}
	}
	return nil
}
