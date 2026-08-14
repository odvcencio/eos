package compiler

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
)

func TestNormalizeCUDAArch(t *testing.T) {
	for _, raw := range []string{"90", "sm_90", "compute_90"} {
		if got, err := normalizeCUDAArch(raw); err != nil || got != "90" {
			t.Fatalf("normalizeCUDAArch(%q) = %q, %v", raw, got, err)
		}
	}
	for _, raw := range []string{"", "sm_xx"} {
		if _, err := normalizeCUDAArch(raw); err == nil {
			t.Fatalf("normalizeCUDAArch(%q) unexpectedly succeeded", raw)
		}
	}
}

func TestCompileKernelVariantsWithOfflineCompiler(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell fixture requires POSIX execution")
	}
	tool := filepath.Join(t.TempDir(), "fake-nvcc")
	script := "#!/bin/sh\nout=''\nprev=''\nfor arg in \"$@\"; do\n  if [ \"$prev\" = \"-o\" ]; then out=\"$arg\"; fi\n  prev=\"$arg\"\ndone\nprintf 'fake-ptx' > \"$out\"\n"
	if err := os.WriteFile(tool, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	bundle, err := Build(nil, Options{ModuleName: "offline_probe", Preset: PresetTinyEmbed})
	if err != nil {
		t.Fatal(err)
	}
	if err := CompileKernelVariants(context.Background(), bundle.Artifact, KernelToolchainOptions{
		Backend: eosartifact.BackendCUDA,
		Tool:    tool,
		Arch:    "sm_90",
	}); err != nil {
		t.Fatal(err)
	}
	for _, kernel := range bundle.Artifact.Kernels {
		for _, variant := range kernel.Variants {
			if variant.Backend != eosartifact.BackendCUDA {
				continue
			}
			if variant.Binary == nil || variant.Binary.Format != "ptx" || string(variant.Binary.Data) != "fake-ptx" {
				t.Fatalf("offline binary = %+v", variant.Binary)
			}
		}
	}
}

func TestCompileKernelVariantsRejectsUnsupportedBackend(t *testing.T) {
	bundle, err := Build(nil, Options{ModuleName: "offline_probe", Preset: PresetTinyEmbed})
	if err != nil {
		t.Fatal(err)
	}
	err = CompileKernelVariants(context.Background(), bundle.Artifact, KernelToolchainOptions{Backend: eosartifact.BackendWebGPU})
	if err == nil || !strings.Contains(err.Error(), "unsupported") {
		t.Fatalf("error = %v, want unsupported backend", err)
	}
}
