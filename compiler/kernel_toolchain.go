package compiler

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	eosartifact "m31labs.dev/eos/artifact/eos"
)

// KernelToolchainOptions selects an offline compiler for one backend. The
// compiler is deliberately invoked by an explicit build action; normal EOS
// compilation remains hermetic and source-backed when no toolchain is asked
// for. CUDA PTX is the first portable image format; cubin is supported by the
// same API once a concrete SM target is selected.
type KernelToolchainOptions struct {
	Backend eosartifact.BackendKind
	Tool    string
	Arch    string
	Format  string
}

// CompileKernelVariants compiles every variant for opts.Backend and attaches
// the resulting offline image to the module. It validates the source/ABI
// contract before invoking an external compiler and validates the sealed
// binary hash before returning.
func CompileKernelVariants(ctx context.Context, mod *eosartifact.Module, opts KernelToolchainOptions) error {
	if mod == nil {
		return fmt.Errorf("nil module")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if opts.Backend != eosartifact.BackendCUDA && opts.Backend != eosartifact.BackendMetal {
		return fmt.Errorf("offline kernel compilation is unsupported for backend %q", opts.Backend)
	}
	if !mod.SupportsBackend(opts.Backend) {
		return fmt.Errorf("module %q does not support backend %q", mod.Name, opts.Backend)
	}
	if err := mod.Validate(); err != nil {
		return fmt.Errorf("validate module before offline compilation: %w", err)
	}
	for i := range mod.Kernels {
		for j := range mod.Kernels[i].Variants {
			variant := &mod.Kernels[i].Variants[j]
			if variant.Backend != opts.Backend {
				continue
			}
			binary, err := compileKernelVariant(ctx, *variant, opts)
			if err != nil {
				return fmt.Errorf("kernel %q: %w", mod.Kernels[i].Name, err)
			}
			variant.Binary = binary
		}
	}
	if err := mod.Validate(); err != nil {
		return fmt.Errorf("validate module after offline compilation: %w", err)
	}
	return nil
}

func compileKernelVariant(ctx context.Context, variant eosartifact.KernelVariant, opts KernelToolchainOptions) (*eosartifact.KernelBinary, error) {
	if err := eosartifact.ValidateKernelVariant(variant.Entry, variant); err != nil {
		return nil, err
	}
	switch opts.Backend {
	case eosartifact.BackendCUDA:
		return compileCUDAKernelVariant(ctx, variant, opts)
	case eosartifact.BackendMetal:
		return compileMetalKernelVariant(ctx, variant, opts)
	default:
		return nil, fmt.Errorf("unsupported offline backend %q", opts.Backend)
	}
}

func compileCUDAKernelVariant(ctx context.Context, variant eosartifact.KernelVariant, opts KernelToolchainOptions) (*eosartifact.KernelBinary, error) {
	tool := opts.Tool
	if tool == "" {
		tool = os.Getenv("EOS_NVCC")
	}
	if tool == "" {
		tool = "nvcc"
	}
	arch, err := normalizeCUDAArch(opts.Arch)
	if err != nil {
		return nil, err
	}
	format := opts.Format
	if format == "" {
		format = "ptx"
	}
	if format != "ptx" && format != "cubin" && format != "fatbin" {
		return nil, fmt.Errorf("CUDA offline format %q is unsupported", format)
	}
	if format == "fatbin" {
		return nil, fmt.Errorf("CUDA offline format %q is reserved until multi-arch fatbin packing is wired", format)
	}

	outputFlag := "--ptx"
	archFlag := "compute_" + arch
	if format == "cubin" {
		outputFlag = "--cubin"
		archFlag = "sm_" + arch
	}
	binary, err := runKernelCompiler(ctx, tool, ".cu", variant.Source, outputFlag, "-arch="+archFlag)
	if err != nil {
		return nil, fmt.Errorf("run %s for %s: %w", tool, variant.Entry, err)
	}
	return kernelBinary(format, arch, filepath.Base(tool), variant.Source, binary), nil
}

func compileMetalKernelVariant(ctx context.Context, variant eosartifact.KernelVariant, opts KernelToolchainOptions) (*eosartifact.KernelBinary, error) {
	if runtime.GOOS != "darwin" {
		return nil, fmt.Errorf("Metal offline compilation requires darwin, got %s", runtime.GOOS)
	}
	tool := opts.Tool
	if tool == "" {
		tool = "xcrun"
	}
	format := opts.Format
	if format == "" {
		format = "metallib"
	}
	if format != "air" && format != "metallib" {
		return nil, fmt.Errorf("Metal offline format %q is unsupported", format)
	}
	dir, err := os.MkdirTemp("", "eos-metal-toolchain-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	sourcePath := filepath.Join(dir, "kernel.metal")
	airPath := filepath.Join(dir, "kernel.air")
	if err := os.WriteFile(sourcePath, []byte(variant.Source), 0o600); err != nil {
		return nil, err
	}
	if err := runTool(ctx, tool, "-sdk", "macosx", "metal", "-c", sourcePath, "-o", airPath); err != nil {
		return nil, fmt.Errorf("compile %s with %s: %w", variant.Entry, tool, err)
	}
	outputPath := airPath
	if format == "metallib" {
		outputPath = filepath.Join(dir, "kernel.metallib")
		if err := runTool(ctx, tool, "-sdk", "macosx", "metallib", airPath, "-o", outputPath); err != nil {
			return nil, fmt.Errorf("package %s with %s: %w", variant.Entry, tool, err)
		}
	}
	data, err := os.ReadFile(outputPath)
	if err != nil {
		return nil, fmt.Errorf("read Metal compiler output: %w", err)
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("Metal compiler produced an empty %s image", format)
	}
	return kernelBinary(format, opts.Arch, filepath.Base(tool), variant.Source, data), nil
}

func runKernelCompiler(ctx context.Context, tool, extension, source, outputFlag string, extraArgs ...string) ([]byte, error) {
	dir, err := os.MkdirTemp("", "eos-kernel-toolchain-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	sourcePath := filepath.Join(dir, "kernel"+extension)
	outputPath := filepath.Join(dir, "kernel.out")
	if err := os.WriteFile(sourcePath, []byte(source), 0o600); err != nil {
		return nil, err
	}
	args := []string{}
	args = append(args, extraArgs...)
	args = append(args, outputFlag, sourcePath, "-o", outputPath)
	var log bytes.Buffer
	if err := runToolWithLog(ctx, tool, &log, args...); err != nil {
		return nil, fmt.Errorf("%w: %s", err, strings.TrimSpace(log.String()))
	}
	binary, err := os.ReadFile(outputPath)
	if err != nil {
		return nil, fmt.Errorf("read compiler output: %w (log: %s)", err, strings.TrimSpace(log.String()))
	}
	if len(binary) == 0 {
		return nil, fmt.Errorf("compiler produced an empty image (log: %s)", strings.TrimSpace(log.String()))
	}
	return binary, nil
}

func runTool(ctx context.Context, tool string, args ...string) error {
	var log bytes.Buffer
	if err := runToolWithLog(ctx, tool, &log, args...); err != nil {
		return fmt.Errorf("%w: %s", err, strings.TrimSpace(log.String()))
	}
	return nil
}

func runToolWithLog(ctx context.Context, tool string, log *bytes.Buffer, args ...string) error {
	cmd := exec.CommandContext(ctx, tool, args...)
	cmd.Stdout = log
	cmd.Stderr = log
	return cmd.Run()
}

func kernelBinary(format, arch, toolchain, source string, data []byte) *eosartifact.KernelBinary {
	sourceSum := sha256.Sum256([]byte(source))
	sum := sha256.Sum256(data)
	return &eosartifact.KernelBinary{
		Format:       format,
		Arch:         arch,
		Toolchain:    toolchain,
		SourceSHA256: hex.EncodeToString(sourceSum[:]),
		SHA256:       hex.EncodeToString(sum[:]),
		Data:         append([]byte(nil), data...),
	}
}

func normalizeCUDAArch(raw string) (string, error) {
	arch := strings.TrimSpace(raw)
	arch = strings.TrimPrefix(arch, "sm_")
	arch = strings.TrimPrefix(arch, "compute_")
	if arch == "" {
		return "", fmt.Errorf("CUDA offline compilation requires --cuda-arch (for example sm_90)")
	}
	if _, err := strconv.Atoi(arch); err != nil {
		return "", fmt.Errorf("CUDA architecture %q must be numeric (for example sm_90)", raw)
	}
	return arch, nil
}
