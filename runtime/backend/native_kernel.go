package backend

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"

	eosartifact "m31labs.dev/eos/artifact/eos"
)

// KernelLaunchArg describes the runtime-visible portion of a generated
// kernel's launch ABI. Built-in execution identifiers (for example Metal's
// thread_position_in_grid) are intentionally omitted: the backend supplies
// those values as part of dispatch rather than through the buffer argument
// vector.
type KernelLaunchArg struct {
	Name   string
	Kind   string
	Access string
}

// KernelLaunchContract is the backend-neutral launch shape derived from the
// kernel body. The ABI fingerprint is stable for a given backend/shape and is
// useful as a cache key for future generic launch bridges and fixed-shape
// graph replay.
type KernelLaunchContract struct {
	Version     string
	Backend     eosartifact.BackendKind
	Shape       string
	Fingerprint string
	Args        []KernelLaunchArg
}

// ExpectedKernelLaunchContract returns the contract for the generated CUDA
// and Metal kernel families currently dispatched directly by the runtime.
// Kernels outside those families remain on the existing host/backend path and
// deliberately do not receive a guessed argument contract.
func ExpectedKernelLaunchContract(kind eosartifact.BackendKind, kernel eosartifact.Kernel) (KernelLaunchContract, bool) {
	if kind != eosartifact.BackendCUDA && kind != eosartifact.BackendMetal {
		return KernelLaunchContract{}, false
	}
	if len(kernel.Body) < 2 || kernel.Body[len(kernel.Body)-1].Op != "return" {
		return KernelLaunchContract{}, false
	}
	var shape string
	var args []KernelLaunchArg
	switch kernel.Body[0].Op {
	case "normalize", "rmsnorm", "layernorm", "softmax":
		shape = "row_wise"
		args = []KernelLaunchArg{
			{Name: "in0", Kind: "pointer", Access: "read"},
			{Name: "out0", Kind: "pointer", Access: "write"},
			{Name: "rows", Kind: "value", Access: "value"},
			{Name: "cols", Kind: "value", Access: "value"},
		}
	case "rope":
		shape = "rope"
		args = []KernelLaunchArg{
			{Name: "in0", Kind: "pointer", Access: "read"},
			{Name: "out0", Kind: "pointer", Access: "write"},
			{Name: "rows", Kind: "value", Access: "value"},
			{Name: "cols", Kind: "value", Access: "value"},
			{Name: "seq_len", Kind: "value", Access: "value"},
		}
	case "binary_add", "binary_sub", "binary_mul", "binary_div":
		shape = "elementwise_binary"
		args = []KernelLaunchArg{
			{Name: "lhs", Kind: "pointer", Access: "read"},
			{Name: "rhs", Kind: "pointer", Access: "read"},
			{Name: "out0", Kind: "pointer", Access: "write"},
			{Name: "elements", Kind: "value", Access: "value"},
		}
	case "dequant", "gelu":
		shape = "elementwise_unary"
		args = []KernelLaunchArg{
			{Name: "in0", Kind: "pointer", Access: "read"},
			{Name: "out0", Kind: "pointer", Access: "write"},
			{Name: "elements", Kind: "value", Access: "value"},
		}
	case "dot", "cosine", "l2_distance":
		shape = "row_score"
		args = []KernelLaunchArg{
			{Name: "query", Kind: "pointer", Access: "read"},
			{Name: "docs", Kind: "pointer", Access: "read"},
			{Name: "out0", Kind: "pointer", Access: "write"},
			{Name: "rows", Kind: "value", Access: "value"},
			{Name: "cols", Kind: "value", Access: "value"},
		}
	default:
		return KernelLaunchContract{}, false
	}
	contract := KernelLaunchContract{
		Version: eosartifact.KernelABIVersion,
		Backend: kind,
		Shape:   shape,
		Args:    args,
	}
	contract.Fingerprint = kernelLaunchContractFingerprint(contract)
	return contract, true
}

func kernelLaunchContractFingerprint(contract KernelLaunchContract) string {
	var b strings.Builder
	b.WriteString(contract.Version)
	b.WriteByte('|')
	b.WriteString(string(contract.Backend))
	b.WriteByte('|')
	b.WriteString(contract.Shape)
	for _, arg := range contract.Args {
		b.WriteByte('|')
		b.WriteString(arg.Name)
		b.WriteByte(':')
		b.WriteString(arg.Kind)
		b.WriteByte(':')
		b.WriteString(arg.Access)
	}
	sum := sha256.Sum256([]byte(b.String()))
	return hex.EncodeToString(sum[:])
}

// ValidateKernelLaunchContract checks that the typed source ABI agrees with
// the arguments the backend launcher will actually pass. A nil ABI is a
// legacy source-backed artifact and remains loadable, but is reported as
// unverified in LaunchConfig; newly generated artifacts always carry an ABI.
func ValidateKernelLaunchContract(kind eosartifact.BackendKind, kernel eosartifact.Kernel, compiled CompiledKernel) error {
	contract, ok := ExpectedKernelLaunchContract(kind, kernel)
	if !ok || compiled.ABI == nil {
		return nil
	}
	args := make([]eosartifact.KernelABIArg, 0, len(compiled.ABI.Args))
	for _, arg := range compiled.ABI.Args {
		if kind == eosartifact.BackendMetal && strings.HasPrefix(arg.Location, "builtin:") {
			continue
		}
		args = append(args, arg)
	}
	if len(args) != len(contract.Args) {
		return fmt.Errorf("kernel %q launch ABI argument count %d does not match %s contract count %d", kernel.Name, len(args), contract.Shape, len(contract.Args))
	}
	for i, want := range contract.Args {
		got := args[i]
		if got.Name != want.Name {
			return fmt.Errorf("kernel %q launch ABI argument %d is %q, want %q", kernel.Name, i, got.Name, want.Name)
		}
		argKind := got.Kind
		if argKind == "" {
			argKind = inferredKernelABIArgKind(got)
		}
		if argKind != want.Kind {
			return fmt.Errorf("kernel %q launch ABI argument %q kind %q, want %q", kernel.Name, got.Name, argKind, want.Kind)
		}
		if got.Access != "" && got.Access != want.Access {
			return fmt.Errorf("kernel %q launch ABI argument %q access %q, want %q", kernel.Name, got.Name, got.Access, want.Access)
		}
		if argKind == "pointer" {
			if got.AddressSpace == "" && kind != eosartifact.BackendMetal {
				return fmt.Errorf("kernel %q launch ABI pointer argument %q has no address space", kernel.Name, got.Name)
			}
			if got.Location == "" {
				return fmt.Errorf("kernel %q launch ABI pointer argument %q has no location", kernel.Name, got.Name)
			}
		} else if got.AddressSpace != "" && got.AddressSpace != "constant" {
			return fmt.Errorf("kernel %q launch ABI scalar argument %q has unexpected address space %q", kernel.Name, got.Name, got.AddressSpace)
		}
	}
	return nil
}

func inferredKernelABIArgKind(arg eosartifact.KernelABIArg) string {
	if strings.Contains(arg.Type, "*") || (arg.AddressSpace != "" && !strings.HasPrefix(arg.Location, "builtin:")) {
		return "pointer"
	}
	return "value"
}

// CompileNativeKernelProgram builds a backend-owned kernel program from a
// compiled variant plus the backend-neutral kernel body. The program still
// executes the supported v0 ops numerically for now, but the launch path is
// owned by the selected backend rather than the generic plan executor.
func CompileNativeKernelProgram(kind eosartifact.BackendKind, kernel eosartifact.Kernel, compiled CompiledKernel) (NativeKernelProgram, error) {
	if kernel.Name == "" {
		return NativeKernelProgram{}, fmt.Errorf("kernel name is required")
	}
	if compiled.Entry == "" {
		return NativeKernelProgram{}, fmt.Errorf("kernel %q compiled entry is required", kernel.Name)
	}
	if err := validateCompiledKernelSource(kind, compiled); err != nil {
		return NativeKernelProgram{}, err
	}
	contract, hasContract := ExpectedKernelLaunchContract(kind, kernel)
	if hasContract {
		if err := ValidateKernelLaunchContract(kind, kernel, compiled); err != nil {
			return NativeKernelProgram{}, err
		}
	}
	config := nativeLaunchConfig(kind, kernel, compiled)
	if hasContract {
		config["launch_contract_version"] = contract.Version
		config["launch_contract_shape"] = contract.Shape
		config["launch_contract_fingerprint"] = contract.Fingerprint
		config["launch_arg_count"] = len(contract.Args)
		if compiled.ABI == nil {
			config["launch_abi_status"] = "legacy_unverified"
		} else {
			config["launch_abi_status"] = "validated"
		}
	} else {
		config["launch_abi_status"] = "not_applicable"
	}
	fallback := func(inputs []*Tensor) ([]*Tensor, error) {
		return executeKernel(kernel, inputs)
	}
	return NativeKernelProgram{
		Compiled:       compiled,
		LaunchContract: contract,
		LaunchConfig:   config,
		Fallback:       fallback,
		Run:            fallback,
	}, nil
}

func validateCompiledKernelSource(kind eosartifact.BackendKind, compiled CompiledKernel) error {
	return eosartifact.ValidateKernelVariant(compiled.Name, eosartifact.KernelVariant{
		Backend: kind,
		Entry:   compiled.Entry,
		Source:  compiled.Source,
		ABI:     compiled.ABI,
		Binary:  compiled.Binary,
	})
}

func nativeLaunchConfig(kind eosartifact.BackendKind, kernel eosartifact.Kernel, compiled CompiledKernel) map[string]any {
	config := map[string]any{
		"dispatch_mode":       "backend_native",
		"dispatch_backend":    string(kind),
		"device_execution":    false,
		"launch_entry":        compiled.Entry,
		"launch_tile":         compiled.Meta["tile"],
		"launch_tile_2d":      compiled.Meta["tile_2d"],
		"launch_vector_width": compiled.Meta["vector_width"],
		"launch_memory":       compiled.Meta["memory"],
		"launch_subgroup":     compiled.Meta["subgroup"],
		"launch_subgroup_2d":  compiled.Meta["subgroup_2d"],
		"launch_halo":         compiled.Meta["halo"],
	}
	if compiled.Binary != nil && compiled.Binary.Format != "" {
		config["launch_compiler"] = "offline_" + compiled.Binary.Format
	} else {
		config["launch_compiler"] = "nvrtc"
	}
	tile := firstTileSize(kernel, compiled)
	grid := "1d"
	if len(kernel.Hints.Tile2D) > 0 || compiled.Meta["tile_2d"] != "" {
		grid = "2d"
	}
	switch kind {
	case eosartifact.BackendCUDA:
		config["launch_api"] = "cuLaunchKernel"
		config["launch_grid"] = grid
		config["launch_block_size"] = tile
		config["launch_shared_bytes"] = estimatedSharedBytes(kernel, tile)
	case eosartifact.BackendMetal:
		config["launch_api"] = "dispatchThreadgroups"
		config["launch_grid"] = grid
		config["launch_threadgroup_size"] = tile
		config["launch_threadgroup_memory_bytes"] = estimatedSharedBytes(kernel, tile)
	case eosartifact.BackendVulkan:
		config["launch_api"] = "vkCmdDispatch"
		config["launch_grid"] = grid
		config["launch_workgroup_size"] = tile
		config["launch_shared_bytes"] = estimatedSharedBytes(kernel, tile)
	case eosartifact.BackendDirectML:
		config["launch_api"] = "IDMLCommandRecorder::RecordDispatch"
		config["launch_grid"] = grid
		config["launch_threadgroup_size"] = tile
		config["launch_temporary_resource_bytes"] = estimatedSharedBytes(kernel, tile)
	case eosartifact.BackendWebGPU:
		config["launch_api"] = "GPUComputePassEncoder.dispatchWorkgroups"
		config["launch_grid"] = grid
		config["launch_workgroup_size"] = tile
		config["launch_workgroup_memory_bytes"] = estimatedSharedBytes(kernel, tile)
	}
	return config
}

func firstTileSize(kernel eosartifact.Kernel, compiled CompiledKernel) int {
	if len(kernel.Hints.Tile) > 0 && kernel.Hints.Tile[0] > 0 {
		return kernel.Hints.Tile[0]
	}
	if raw := compiled.Meta["tile"]; raw != "" {
		raw = strings.TrimPrefix(raw, "[")
		raw = strings.TrimSuffix(raw, "]")
		if raw != "" {
			parts := strings.Split(raw, ",")
			if n, err := strconv.Atoi(strings.TrimSpace(parts[0])); err == nil && n > 0 {
				return n
			}
		}
	}
	return 1
}

func estimatedSharedBytes(kernel eosartifact.Kernel, tile int) int {
	if tile <= 0 {
		tile = 1
	}
	if kernel.Hints.Memory != "workgroup_local" {
		return 0
	}
	width := kernel.Hints.VectorWidth
	if width <= 0 {
		width = 1
	}
	valueCount := len(kernel.Inputs) + len(kernel.Outputs)
	if valueCount == 0 {
		valueCount = 1
	}
	return tile * width * valueCount * 4
}
