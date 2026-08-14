package eosruntime

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/compiler"
	"m31labs.dev/eos/runtime/backend"
	"m31labs.dev/eos/runtime/backends/cuda"
	"m31labs.dev/eos/runtime/backends/directml"
	"m31labs.dev/eos/runtime/backends/metal"
	"m31labs.dev/eos/runtime/backends/vulkan"
	"m31labs.dev/eos/runtime/backends/webgpu"
)

type stubBackend struct {
	kind         eosartifact.BackendKind
	capabilities []string
	loads        int
	// failErr, when set, makes Load return this error verbatim instead of
	// succeeding -- used to simulate a rejecting backend such as CUDA's
	// stale-RoPE-ABI guard (runtime/backends/cuda/native_linux.go's
	// validateRoPEKernelABI) without needing real GPU hardware.
	failErr error
}

func (b *stubBackend) Kind() eosartifact.BackendKind { return b.kind }

func (b *stubBackend) Capabilities() []string {
	return append([]string(nil), b.capabilities...)
}

func (b *stubBackend) CanLoad(mod *eosartifact.Module) bool {
	return mod != nil && mod.SupportsBackend(b.kind)
}

func (b *stubBackend) Load(_ context.Context, _ *eosartifact.Module, _ map[string]backend.WeightBinding) (backend.Executor, error) {
	b.loads++
	if b.failErr != nil {
		return nil, b.failErr
	}
	return stubExecutor{kind: b.kind}, nil
}

// alwaysRejectBackend rejects every module via CanLoad, so Load must never
// be reachable -- used to test strict-mode gating on the CanLoad rejection
// path (as opposed to a Load() call failure).
type alwaysRejectBackend struct {
	kind  eosartifact.BackendKind
	loads int
}

func (b *alwaysRejectBackend) Kind() eosartifact.BackendKind { return b.kind }

func (b *alwaysRejectBackend) CanLoad(*eosartifact.Module) bool { return false }

func (b *alwaysRejectBackend) Load(context.Context, *eosartifact.Module, map[string]backend.WeightBinding) (backend.Executor, error) {
	b.loads++
	return nil, fmt.Errorf("alwaysRejectBackend.Load should never be called (CanLoad is false)")
}

// captureStderr redirects os.Stderr for the duration of fn and returns
// everything written to it, mirroring cmd/eos/main_test.go's
// captureRunStderrAndOutput os.Pipe idiom for asserting on warning/log
// lines written directly to stderr.
func captureStderr(t *testing.T, fn func()) string {
	t.Helper()
	orig := os.Stderr
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("create stderr pipe: %v", err)
	}
	os.Stderr = writer
	defer func() {
		os.Stderr = orig
	}()
	fn()
	if err := writer.Close(); err != nil {
		t.Fatalf("close stderr writer: %v", err)
	}
	data, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("read stderr capture: %v", err)
	}
	if err := reader.Close(); err != nil {
		t.Fatalf("close stderr reader: %v", err)
	}
	return string(data)
}

type stubExecutor struct {
	kind eosartifact.BackendKind
}

func (e stubExecutor) Backend() eosartifact.BackendKind { return e.kind }

func (e stubExecutor) Run(_ context.Context, _ backend.Request) (backend.Result, error) {
	return backend.Result{}, fmt.Errorf("stub executor does not run")
}

type cacheKeyStubBackend struct {
	stubBackend
	cacheKeys []string
}

func (b *cacheKeyStubBackend) Load(_ context.Context, _ *eosartifact.Module, _ map[string]backend.WeightBinding) (backend.Executor, error) {
	return nil, fmt.Errorf("unexpected uncached load")
}

func (b *cacheKeyStubBackend) LoadWithCacheKey(_ context.Context, _ *eosartifact.Module, _ map[string]backend.WeightBinding, cacheKey string) (backend.Executor, error) {
	b.loads++
	b.cacheKeys = append(b.cacheKeys, cacheKey)
	return stubExecutor{kind: b.kind}, nil
}

func TestLoadRejectsMissingWeightBindings(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	_, err = rt.Load(context.Background(), bundle.Artifact, WithWeight("token_embedding", backend.NewTensorF16([]int{3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
	})))
	if err == nil {
		t.Fatal("expected missing weight binding error")
	}
	if !strings.Contains(err.Error(), "projection") {
		t.Fatalf("expected missing projection binding, got %v", err)
	}
}

func TestLoadAcceptsTinyEmbedBindings(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(
		context.Background(),
		bundle.Artifact,
		tinyEmbedWeights()...,
	)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got := prog.Backend(); got == "" {
		t.Fatal("expected selected backend")
	}
}

func TestLoadFallsBackWhenBackendMissingCapabilities(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_candidates", Preset: compiler.PresetTinyPackedCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	cudaOnly := &stubBackend{kind: eosartifact.BackendCUDA}
	metalCapable := &stubBackend{
		kind:         eosartifact.BackendMetal,
		capabilities: []string{eosartifact.CapabilityCandidatePack},
	}
	rt := New(cudaOnly, metalCapable)
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got := prog.Backend(); got != eosartifact.BackendMetal {
		t.Fatalf("backend = %q, want %q", got, eosartifact.BackendMetal)
	}
	if cudaOnly.loads != 0 {
		t.Fatalf("unexpected CUDA load attempts = %d", cudaOnly.loads)
	}
	if metalCapable.loads != 1 {
		t.Fatalf("metal load attempts = %d, want 1", metalCapable.loads)
	}
}

func TestLoadReportsMissingBackendCapabilities(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_candidates", Preset: compiler.PresetTinyPackedCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(&stubBackend{kind: eosartifact.BackendCUDA}, &stubBackend{kind: eosartifact.BackendMetal})
	_, err = rt.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err == nil {
		t.Fatal("expected missing capability error")
	}
	if !strings.Contains(err.Error(), eosartifact.CapabilityCandidatePack) {
		t.Fatalf("expected candidate_pack in error, got %v", err)
	}
}

// abiGuardStyleError mimics the CUDA stale-RoPE-ABI guard's error text
// (validateRoPEKernelABI in runtime/backends/cuda/native_linux.go, added by
// f0782712) so tests can assert it survives fallback/strict-mode handling
// verbatim without requiring real CUDA hardware.
func abiGuardStyleError() error {
	return errors.New(`stale rope kernel ABI for kernel "embed_layer0" (compiled before seq_len fix): recompile/reseal the module with a current eos binary (init-model --bootstrap-from <old.mll>)`)
}

func TestLoadWarnsOnFallbackAfterLoadFailure(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	abiErr := abiGuardStyleError()
	cudaFails := &stubBackend{kind: eosartifact.BackendCUDA, failErr: abiErr}
	metalOK := &stubBackend{kind: eosartifact.BackendMetal}
	rt := New(cudaFails, metalOK)

	var prog *Program
	stderr := captureStderr(t, func() {
		var loadErr error
		prog, loadErr = rt.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
		if loadErr != nil {
			t.Fatalf("load: %v", loadErr)
		}
	})

	if got := prog.Backend(); got != eosartifact.BackendMetal {
		t.Fatalf("backend = %q, want %q", got, eosartifact.BackendMetal)
	}
	if cudaFails.loads != 1 {
		t.Fatalf("cuda load attempts = %d, want 1", cudaFails.loads)
	}
	if metalOK.loads != 1 {
		t.Fatalf("metal load attempts = %d, want 1", metalOK.loads)
	}

	lines := strings.Split(strings.TrimRight(stderr, "\n"), "\n")
	if len(lines) != 1 {
		t.Fatalf("stderr lines = %d, want 1: %q", len(lines), stderr)
	}
	warning := lines[0]
	if !strings.Contains(warning, string(eosartifact.BackendCUDA)) {
		t.Fatalf("warning missing failed backend name: %q", warning)
	}
	if !strings.Contains(warning, abiErr.Error()) {
		t.Fatalf("warning does not preserve underlying error verbatim: %q", warning)
	}
	if !strings.Contains(warning, string(eosartifact.BackendMetal)) {
		t.Fatalf("warning missing selected fallback backend name: %q", warning)
	}
	if !strings.Contains(warning, EnvRequireBackend) {
		t.Fatalf("warning missing strict-mode escape hatch mention: %q", warning)
	}
}

func TestLoadWarnsPerFailedBackendAcrossMultipleFallbacks(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	cudaErr := errors.New("cuda device init failed: no CUDA-capable device is detected")
	metalErr := errors.New("metal device init failed: no Metal-capable device is detected")
	cudaFails := &stubBackend{kind: eosartifact.BackendCUDA, failErr: cudaErr}
	metalFails := &stubBackend{kind: eosartifact.BackendMetal, failErr: metalErr}
	vulkanOK := &stubBackend{kind: eosartifact.BackendVulkan}
	rt := New(cudaFails, metalFails, vulkanOK)

	var prog *Program
	stderr := captureStderr(t, func() {
		var loadErr error
		prog, loadErr = rt.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
		if loadErr != nil {
			t.Fatalf("load: %v", loadErr)
		}
	})

	if got := prog.Backend(); got != eosartifact.BackendVulkan {
		t.Fatalf("backend = %q, want %q", got, eosartifact.BackendVulkan)
	}

	lines := strings.Split(strings.TrimRight(stderr, "\n"), "\n")
	if len(lines) != 2 {
		t.Fatalf("stderr lines = %d, want 2: %q", len(lines), stderr)
	}
	if !strings.Contains(lines[0], string(eosartifact.BackendCUDA)) || !strings.Contains(lines[0], cudaErr.Error()) || !strings.Contains(lines[0], string(eosartifact.BackendVulkan)) {
		t.Fatalf("first warning malformed: %q", lines[0])
	}
	if !strings.Contains(lines[1], string(eosartifact.BackendMetal)) || !strings.Contains(lines[1], metalErr.Error()) || !strings.Contains(lines[1], string(eosartifact.BackendVulkan)) {
		t.Fatalf("second warning malformed: %q", lines[1])
	}
}

func TestLoadStrictModeOptionSurfacesLoadErrorInsteadOfFallback(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	abiErr := abiGuardStyleError()
	cudaFails := &stubBackend{kind: eosartifact.BackendCUDA, failErr: abiErr}
	metalOK := &stubBackend{kind: eosartifact.BackendMetal}
	rt := New(cudaFails, metalOK)

	opts := append(append([]LoadOption{}, tinyEmbedWeights()...), WithRequireBackend(eosartifact.BackendCUDA))
	var loadErr error
	stderr := captureStderr(t, func() {
		_, loadErr = rt.Load(context.Background(), bundle.Artifact, opts...)
	})
	if loadErr == nil {
		t.Fatal("expected strict-mode load error, got nil")
	}
	if !errors.Is(loadErr, abiErr) {
		t.Fatalf("expected error chain to wrap abiErr, got %v", loadErr)
	}
	if !strings.Contains(loadErr.Error(), abiErr.Error()) {
		t.Fatalf("error does not preserve underlying message verbatim: %v", loadErr)
	}
	if !strings.Contains(loadErr.Error(), string(eosartifact.BackendCUDA)) {
		t.Fatalf("error missing required backend name: %v", loadErr)
	}
	if cudaFails.loads != 1 {
		t.Fatalf("cuda load attempts = %d, want 1", cudaFails.loads)
	}
	if metalOK.loads != 0 {
		t.Fatalf("metal load attempts = %d, want 0 (fallback must not be attempted in strict mode)", metalOK.loads)
	}
	if stderr != "" {
		t.Fatalf("expected no fallback warning in strict mode, got %q", stderr)
	}
}

func TestLoadStrictModeSurfacesCanLoadRejectionInsteadOfFallback(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	cudaRejects := &alwaysRejectBackend{kind: eosartifact.BackendCUDA}
	metalOK := &stubBackend{kind: eosartifact.BackendMetal}
	rt := New(cudaRejects, metalOK)

	opts := append(append([]LoadOption{}, tinyEmbedWeights()...), WithRequireBackend(eosartifact.BackendCUDA))
	_, err = rt.Load(context.Background(), bundle.Artifact, opts...)
	if err == nil {
		t.Fatal("expected strict-mode CanLoad rejection error, got nil")
	}
	if !strings.Contains(err.Error(), "unsupported backend") {
		t.Fatalf("expected unsupported backend reason, got %v", err)
	}
	if !strings.Contains(err.Error(), string(eosartifact.BackendCUDA)) {
		t.Fatalf("error missing required backend name: %v", err)
	}
	if cudaRejects.loads != 0 {
		t.Fatalf("cuda Load attempts = %d, want 0 (CanLoad already false)", cudaRejects.loads)
	}
	if metalOK.loads != 0 {
		t.Fatalf("metal load attempts = %d, want 0 (fallback must not be attempted in strict mode)", metalOK.loads)
	}
}

func TestLoadStrictModeRejectsUnregisteredRequiredBackend(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	metalOK := &stubBackend{kind: eosartifact.BackendMetal}
	rt := New(metalOK)

	opts := append(append([]LoadOption{}, tinyEmbedWeights()...), WithRequireBackend(eosartifact.BackendCUDA))
	_, err = rt.Load(context.Background(), bundle.Artifact, opts...)
	if err == nil {
		t.Fatal("expected error for unregistered required backend")
	}
	if !strings.Contains(err.Error(), "not registered") {
		t.Fatalf("expected 'not registered' error, got %v", err)
	}
	if !strings.Contains(err.Error(), string(eosartifact.BackendMetal)) {
		t.Fatalf("error should list registered backends, got %v", err)
	}
	if metalOK.loads != 0 {
		t.Fatalf("metal load attempts = %d, want 0", metalOK.loads)
	}
}

func TestLoadStrictModeViaEnvVarSurfacesErrorInsteadOfFallback(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	abiErr := abiGuardStyleError()
	cudaFails := &stubBackend{kind: eosartifact.BackendCUDA, failErr: abiErr}
	metalOK := &stubBackend{kind: eosartifact.BackendMetal}
	rt := New(cudaFails, metalOK)

	t.Setenv(EnvRequireBackend, "CUDA") // exercise case-insensitive matching too
	_, err = rt.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
	if err == nil {
		t.Fatal("expected strict-mode load error via env var, got nil")
	}
	if !errors.Is(err, abiErr) {
		t.Fatalf("expected error chain to wrap abiErr, got %v", err)
	}
	if metalOK.loads != 0 {
		t.Fatalf("metal load attempts = %d, want 0 (fallback must not be attempted with %s set)", metalOK.loads, EnvRequireBackend)
	}
}

func TestLoadStrictModeOptionTakesPrecedenceOverEnvVar(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	abiErr := abiGuardStyleError()
	cudaFails := &stubBackend{kind: eosartifact.BackendCUDA, failErr: abiErr}
	metalOK := &stubBackend{kind: eosartifact.BackendMetal}
	rt := New(cudaFails, metalOK)

	// Env says "require metal" but the explicit option says "require cuda";
	// the option must win, so the cuda failure must surface instead of
	// silently accepting the metal load that the env var alone would allow.
	t.Setenv(EnvRequireBackend, "metal")
	opts := append(append([]LoadOption{}, tinyEmbedWeights()...), WithRequireBackend(eosartifact.BackendCUDA))
	_, err = rt.Load(context.Background(), bundle.Artifact, opts...)
	if err == nil {
		t.Fatal("expected strict-mode load error from option, got nil")
	}
	if !errors.Is(err, abiErr) {
		t.Fatalf("expected error chain to wrap abiErr (option must win over env var), got %v", err)
	}
	if metalOK.loads != 0 {
		t.Fatalf("metal load attempts = %d, want 0", metalOK.loads)
	}
}

func TestLoadSuccessfulFirstBackendEmitsNoFallbackWarning(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	var prog *Program
	stderr := captureStderr(t, func() {
		var loadErr error
		prog, loadErr = rt.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
		if loadErr != nil {
			t.Fatalf("load: %v", loadErr)
		}
	})
	if got := prog.Backend(); got != eosartifact.BackendCUDA {
		t.Fatalf("backend = %q, want %q", got, eosartifact.BackendCUDA)
	}
	if stderr != "" {
		t.Fatalf("expected no warning output on successful first-backend load, got %q", stderr)
	}
}

func TestLoadSingleBackendPathEmitsNoFallbackWarning(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(metal.New())
	var prog *Program
	stderr := captureStderr(t, func() {
		var loadErr error
		prog, loadErr = rt.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
		if loadErr != nil {
			t.Fatalf("load: %v", loadErr)
		}
	})
	if got := prog.Backend(); got != eosartifact.BackendMetal {
		t.Fatalf("backend = %q, want %q", got, eosartifact.BackendMetal)
	}
	if stderr != "" {
		t.Fatalf("expected no warning output on single-backend load, got %q", stderr)
	}
}

func TestLoadFileAcceptsSerializedArtifact(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	path := filepath.Join(t.TempDir(), "tiny_embed.mll")
	if err := eosartifact.WriteFile(path, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.LoadFile(context.Background(), path, tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load file: %v", err)
	}
	if got := prog.Backend(); got == "" {
		t.Fatal("expected selected backend from file load")
	}
}

func TestLoadFileUsesSiblingPackageManifestCacheKey(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "tiny_embed.mll")
	if err := eosartifact.WriteFile(path, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}

	manifest, err := BuildPackageManifest(PackageEmbedding, bundle.Artifact, map[string]string{
		"artifact": path,
	})
	if err != nil {
		t.Fatalf("build package manifest: %v", err)
	}
	manifestPath := DefaultPackageManifestPath(path)
	if err := manifest.WriteFile(manifestPath); err != nil {
		t.Fatalf("write package manifest: %v", err)
	}

	backend := &cacheKeyStubBackend{stubBackend: stubBackend{kind: eosartifact.BackendCUDA}}
	rt := New(backend)
	loadOnce := func() error {
		_, err := rt.LoadFile(context.Background(), path, tinyEmbedWeights()...)
		return err
	}
	if err := loadOnce(); err != nil {
		t.Fatalf("first load file: %v", err)
	}
	if err := loadOnce(); err != nil {
		t.Fatalf("second load file: %v", err)
	}

	wantKey := manifest.CacheKey()
	if wantKey == "" {
		t.Fatal("expected non-empty cache key")
	}
	if backend.loads != 2 {
		t.Fatalf("cache-key loads = %d, want 2", backend.loads)
	}
	if len(backend.cacheKeys) != 2 {
		t.Fatalf("cache key calls = %d, want 2", len(backend.cacheKeys))
	}
	for i, got := range backend.cacheKeys {
		if got != wantKey {
			t.Fatalf("cache key call %d = %q, want %q", i, got, wantKey)
		}
	}
}

func TestRunHonorsLazyStagedMemoryPlan(t *testing.T) {
	mod := newLazyStagedParamModule()
	weights := map[string]*backend.Tensor{
		"used":   backend.NewTensorF16([]int{2, 2}, []float32{1, 2, 3, 4}),
		"unused": backend.NewTensorF16([]int{2, 2}, []float32{9, 8, 7, 6}),
	}
	plan := NewMemoryPlan(mod, weights, MemoryPlanOptions{DeviceBudgetBytes: 1})

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(
		context.Background(),
		mod,
		WithWeight("used", weights["used"]),
		WithWeight("unused", weights["unused"]),
		WithMemoryPlan(plan),
	)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{Entry: "serve", Inputs: map[string]any{}})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	output, ok := result.Outputs["result"]
	if !ok {
		t.Fatalf("missing result output: %+v", result.Outputs)
	}
	tensor, ok := output.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("output data type = %T, want *backend.Tensor", output.Data)
	}
	assertTensorClose(t, tensor, []int{2, 2}, []float32{1, 2, 3, 4})
	if got := result.Metadata["params_eager_materialized"]; got != "0" {
		t.Fatalf("params_eager_materialized = %q, want 0", got)
	}
	if got := result.Metadata["params_lazy_materialized"]; got != "1" {
		t.Fatalf("params_lazy_materialized = %q, want 1", got)
	}
	if got := result.Metadata["params_released"]; got != "1" {
		t.Fatalf("params_released = %q, want 1", got)
	}
	if got := result.Metadata["params_unused_for_entry"]; got != "1" {
		t.Fatalf("params_unused_for_entry = %q, want 1", got)
	}
	if got := result.Metadata["param_materialization"]; got != "lazy_on_demand" {
		t.Fatalf("param_materialization = %q, want lazy_on_demand", got)
	}
}

func TestRunTinyEmbedEntryPoint(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  "embed",
		Inputs: map[string]any{"tokens": backend.NewTensorI32([]int{2}, []int32{0, 2})},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	output, ok := result.Outputs["embeddings"]
	if !ok {
		t.Fatalf("missing embeddings output: %+v", result.Outputs)
	}
	if output.Type.Kind != eosartifact.ValueTensor || output.Type.Tensor == nil {
		t.Fatalf("unexpected output type: %+v", output.Type)
	}
	if output.Type.Tensor.DType != "f16" {
		t.Fatalf("output dtype = %q, want f16", output.Type.Tensor.DType)
	}
	if got := strings.Join(output.Type.Tensor.Shape, ","); got != "2,2" {
		t.Fatalf("output shape = %q, want 2,2", got)
	}
	if output.Producer != "kernel:l2_normalize" {
		t.Fatalf("output producer = %q, want kernel:l2_normalize", output.Producer)
	}
	tensor, ok := output.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("output data type = %T, want *backend.Tensor", output.Data)
	}
	assertTensorClose(t, tensor, []int{2, 2}, []float32{
		1, 0,
		0.70710677, 0.70710677,
	})
	if output.Metadata["variant_entry"] != "l2_normalize_cuda" {
		t.Fatalf("variant_entry = %v, want l2_normalize_cuda", output.Metadata["variant_entry"])
	}
	if output.Metadata["dispatch_mode"] != "backend_native" {
		t.Fatalf("dispatch_mode = %v, want backend_native", output.Metadata["dispatch_mode"])
	}
	if output.Metadata["launch_api"] != "cuLaunchKernel" {
		t.Fatalf("launch_api = %v, want cuLaunchKernel", output.Metadata["launch_api"])
	}
	if output.Metadata["launch_block_size"] != 128 {
		t.Fatalf("launch_block_size = %v, want 128", output.Metadata["launch_block_size"])
	}
	if got := result.Metadata["compiled_kernels"]; got != "1" {
		t.Fatalf("compiled_kernels = %q, want 1", got)
	}
	if got := result.Metadata["kernel_dispatch"]; got != "backend_native" {
		t.Fatalf("kernel_dispatch = %q, want backend_native", got)
	}
	if got := result.Metadata["entrypoint"]; got != "embed" {
		t.Fatalf("entrypoint metadata = %q, want embed", got)
	}
	if got := len(result.Trace); got != len(bundle.Artifact.Steps) {
		t.Fatalf("trace len = %d, want %d", got, len(bundle.Artifact.Steps))
	}
	if result.Trace[len(result.Trace)-2].Variant != "l2_normalize_cuda" {
		t.Fatalf("trace variant = %q, want l2_normalize_cuda", result.Trace[len(result.Trace)-2].Variant)
	}
}

func TestPortableGPUBackendsLoadMLLArtifactsWithHostFallback(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	cases := []struct {
		name      string
		rt        *Runtime
		backend   eosartifact.BackendKind
		entry     string
		launchAPI string
	}{
		{name: "vulkan", rt: New(vulkan.New()), backend: eosartifact.BackendVulkan, entry: "l2_normalize_vulkan", launchAPI: "vkCmdDispatch"},
		{name: "directml", rt: New(directml.New()), backend: eosartifact.BackendDirectML, entry: "l2_normalize_directml", launchAPI: "IDMLCommandRecorder::RecordDispatch"},
		{name: "webgpu", rt: New(webgpu.New()), backend: eosartifact.BackendWebGPU, entry: "l2_normalize_webgpu", launchAPI: "GPUComputePassEncoder.dispatchWorkgroups"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			prog, err := tc.rt.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			if got := prog.Backend(); got != tc.backend {
				t.Fatalf("backend = %q, want %q", got, tc.backend)
			}
			result, err := prog.Run(context.Background(), backend.Request{
				Entry:  "embed",
				Inputs: map[string]any{"tokens": backend.NewTensorI32([]int{2}, []int32{0, 2})},
			})
			if err != nil {
				t.Fatalf("run: %v", err)
			}
			output := result.Outputs["embeddings"]
			if output.Metadata["variant_entry"] != tc.entry {
				t.Fatalf("variant_entry = %v, want %s", output.Metadata["variant_entry"], tc.entry)
			}
			if output.Metadata["launch_api"] != tc.launchAPI {
				t.Fatalf("launch_api = %v, want %s", output.Metadata["launch_api"], tc.launchAPI)
			}
			if output.Metadata["execution_mode"] != "host_fallback" {
				t.Fatalf("execution_mode = %v, want host_fallback", output.Metadata["execution_mode"])
			}
			tensor := output.Data.(*backend.Tensor)
			assertTensorClose(t, tensor, []int{2, 2}, []float32{
				1, 0,
				0.70710677, 0.70710677,
			})
		})
	}
}

func TestRunTinyDecodeEntryPoint(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_decode", Preset: compiler.PresetTinyDecode})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, WithWeight("wq", backend.NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0, 1,
	})))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	cache := backend.NewKVCache(backend.NewTensorF16([]int{2, 2}, []float32{
		0, 0,
		0, 0,
	}))
	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "decode_step",
		Inputs: map[string]any{
			"x": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			}),
			"cache": cache,
		},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	output, ok := result.Outputs["logits"]
	if !ok {
		t.Fatalf("missing logits output: %+v", result.Outputs)
	}
	if output.Producer != "kernel:softmax" {
		t.Fatalf("output producer = %q, want kernel:softmax", output.Producer)
	}
	tensor, ok := output.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("output data type = %T, want *backend.Tensor", output.Data)
	}
	if output.Metadata["variant_entry"] != "softmax_cuda" {
		t.Fatalf("variant_entry = %v, want softmax_cuda", output.Metadata["variant_entry"])
	}
	if output.Metadata["launch_api"] != "cuLaunchKernel" {
		t.Fatalf("launch_api = %v, want cuLaunchKernel", output.Metadata["launch_api"])
	}
	if output.Metadata["launch_block_size"] != 64 {
		t.Fatalf("launch_block_size = %v, want 64", output.Metadata["launch_block_size"])
	}
	if output.Metadata["launch_memory"] != "workgroup_local" {
		t.Fatalf("launch_memory = %v, want workgroup_local", output.Metadata["launch_memory"])
	}
	for r := 0; r < tensor.Shape[0]; r++ {
		rowSum := float32(0)
		for c := 0; c < tensor.Shape[1]; c++ {
			rowSum += tensor.F32[r*tensor.Shape[1]+c]
		}
		assertClose(t, rowSum, 1, 0.0005)
	}
	if got := result.Metadata["backend"]; got != "cuda" {
		t.Fatalf("backend metadata = %q, want cuda", got)
	}
	if got := result.Metadata["status"]; got != "hybrid" {
		t.Fatalf("status metadata = %q, want hybrid", got)
	}
	if got := result.Metadata["step_count"]; got != "7" {
		t.Fatalf("step_count metadata = %q, want 7", got)
	}
	if got := result.Metadata["compiled_kernels"]; got != "3" {
		t.Fatalf("compiled_kernels = %q, want 3", got)
	}
	if got := result.Trace[len(result.Trace)-2].Variant; got != "softmax_cuda" {
		t.Fatalf("softmax trace variant = %q, want softmax_cuda", got)
	}
	if cache.Value == nil {
		t.Fatal("expected kv cache mutation")
	}
	assertTensorClose(t, cache.Value, []int{2, 2}, []float32{
		1, 0,
		-0.84147096, 0.5403023,
	})
}

func TestRunTinyScoreEntryPoint(t *testing.T) {
	if !cudaNativeRuntimeTestsAvailable {
		t.Skip("CUDA native runtime dispatch requires linux cgo")
	}
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_score", Preset: compiler.PresetTinyScore})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyScoreWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  "score",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	output, ok := result.Outputs["scores"]
	if !ok {
		t.Fatalf("missing scores output: %+v", result.Outputs)
	}
	if output.Producer != "kernel:cosine" {
		t.Fatalf("output producer = %q, want kernel:cosine", output.Producer)
	}
	if output.Type.Kind != eosartifact.ValueTensor || output.Type.Tensor == nil || output.Type.Tensor.DType != "f32" {
		t.Fatalf("unexpected output type: %+v", output.Type)
	}
	tensor, ok := output.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("output data type = %T, want *backend.Tensor", output.Data)
	}
	assertTensorClose(t, tensor, []int{2}, []float32{1, 0})
	if output.Metadata["variant_entry"] != "cosine_cuda" {
		t.Fatalf("variant_entry = %v, want cosine_cuda", output.Metadata["variant_entry"])
	}
	if output.Metadata["device_execution"] != true {
		t.Fatalf("device_execution = %v, want true", output.Metadata["device_execution"])
	}
	if got := result.Metadata["compiled_kernels"]; got != "1" {
		t.Fatalf("compiled_kernels = %q, want 1", got)
	}
	if got := result.Metadata["entrypoint"]; got != "score" {
		t.Fatalf("entrypoint metadata = %q, want score", got)
	}
}

func TestRunTinyRerankEntryPoint(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_rerank", Preset: compiler.PresetTinyRerank})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  "rerank",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	topIDs, ok := result.Outputs["top_ids"]
	if !ok {
		t.Fatalf("missing top_ids output: %+v", result.Outputs)
	}
	if topIDs.Producer != "topk:topk" {
		t.Fatalf("top_ids producer = %q, want topk:topk", topIDs.Producer)
	}
	tensor, ok := topIDs.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("top_ids data type = %T, want *backend.Tensor", topIDs.Data)
	}
	assertTensorI32(t, tensor, []int{2}, []int32{0, 2})
	topScores, ok := result.Outputs["top_scores"]
	if !ok {
		t.Fatalf("missing top_scores output: %+v", result.Outputs)
	}
	if topScores.Producer != "gather:gather" {
		t.Fatalf("top_scores producer = %q, want gather:gather", topScores.Producer)
	}
	scoreTensor, ok := topScores.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("top_scores data type = %T, want *backend.Tensor", topScores.Data)
	}
	assertTensorClose(t, scoreTensor, []int{2}, []float32{1, 0.70710677})
	if got := result.Metadata["entrypoint"]; got != "rerank" {
		t.Fatalf("entrypoint metadata = %q, want rerank", got)
	}
	if got := len(result.Trace); got != 4 {
		t.Fatalf("trace len = %d, want 4", got)
	}
	if result.Trace[1].Kind != eosartifact.StepTopK {
		t.Fatalf("trace[1].kind = %q, want topk", result.Trace[1].Kind)
	}
	if result.Trace[2].Kind != eosartifact.StepGather {
		t.Fatalf("trace[2].kind = %q, want gather", result.Trace[2].Kind)
	}
}

func TestRunTinySelectEntryPoint(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_select", Preset: compiler.PresetTinySelect})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  "select_scores",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	output, ok := result.Outputs["top_scores"]
	if !ok {
		t.Fatalf("missing top_scores output: %+v", result.Outputs)
	}
	if output.Producer != "gather:gather" {
		t.Fatalf("output producer = %q, want gather:gather", output.Producer)
	}
	tensor, ok := output.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("output data type = %T, want *backend.Tensor", output.Data)
	}
	assertTensorClose(t, tensor, []int{2}, []float32{1, 0.70710677})
	if got := len(result.Trace); got != 4 {
		t.Fatalf("trace len = %d, want 4", got)
	}
	if result.Trace[2].Kind != eosartifact.StepGather {
		t.Fatalf("trace[2].kind = %q, want gather", result.Trace[2].Kind)
	}
}

func TestRunTinyRetrieveEntryPoint(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_retrieve", Preset: compiler.PresetTinyRetrieve})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  "retrieve",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	topIDs, ok := result.Outputs["top_ids"]
	if !ok {
		t.Fatalf("missing top_ids output: %+v", result.Outputs)
	}
	topScores, ok := result.Outputs["top_scores"]
	if !ok {
		t.Fatalf("missing top_scores output: %+v", result.Outputs)
	}
	topDocs, ok := result.Outputs["top_docs"]
	if !ok {
		t.Fatalf("missing top_docs output: %+v", result.Outputs)
	}

	idTensor := topIDs.Data.(*backend.Tensor)
	scoreTensor := topScores.Data.(*backend.Tensor)
	docTensor := topDocs.Data.(*backend.Tensor)
	assertTensorI32(t, idTensor, []int{2}, []int32{0, 2})
	assertTensorClose(t, scoreTensor, []int{2}, []float32{1, 0.70710677})
	assertTensorClose(t, docTensor, []int{2, 2}, []float32{
		1, 0,
		1, 1,
	})
	if docTensor.DType != "q4" {
		t.Fatalf("top_docs dtype = %q, want q4", docTensor.DType)
	}
	if got := len(result.Trace); got != 5 {
		t.Fatalf("trace len = %d, want 5", got)
	}
	if result.Trace[3].Kind != eosartifact.StepGather {
		t.Fatalf("trace[3].kind = %q, want gather", result.Trace[3].Kind)
	}
}

func TestRunTinyEmbedEntryPointOnMetal(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  "embed",
		Inputs: map[string]any{"tokens": backend.NewTensorI32([]int{2}, []int32{0, 2})},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if got := prog.Backend(); got != eosartifact.BackendMetal {
		t.Fatalf("backend = %q, want metal", got)
	}
	output := result.Outputs["embeddings"]
	if output.Metadata["variant_entry"] != "l2_normalize_metal" {
		t.Fatalf("variant_entry = %v, want l2_normalize_metal", output.Metadata["variant_entry"])
	}
	if output.Metadata["dispatch_mode"] != "backend_native" {
		t.Fatalf("dispatch_mode = %v, want backend_native", output.Metadata["dispatch_mode"])
	}
	if output.Metadata["launch_api"] != "dispatchThreadgroups" {
		t.Fatalf("launch_api = %v, want dispatchThreadgroups", output.Metadata["launch_api"])
	}
	if output.Metadata["launch_threadgroup_size"] != 128 {
		t.Fatalf("launch_threadgroup_size = %v, want 128", output.Metadata["launch_threadgroup_size"])
	}
	if result.Trace[len(result.Trace)-2].Variant != "l2_normalize_metal" {
		t.Fatalf("trace variant = %q, want l2_normalize_metal", result.Trace[len(result.Trace)-2].Variant)
	}
	tensor, ok := output.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("metal output data type = %T, want *backend.Tensor", output.Data)
	}
	assertTensorClose(t, tensor, []int{2, 2}, []float32{
		1, 0,
		0.70710677, 0.70710677,
	})
}

func TestRunTinyEmbedParityAcrossBackends(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	cudaRuntime := New(cuda.New())
	cudaProg, err := cudaRuntime.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	cudaResult, err := cudaProg.Run(context.Background(), backend.Request{
		Entry:  "embed",
		Inputs: map[string]any{"tokens": backend.NewTensorI32([]int{2}, []int32{0, 2})},
	})
	if err != nil {
		t.Fatalf("run cuda: %v", err)
	}

	metalRuntime := New(metal.New())
	metalProg, err := metalRuntime.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	metalResult, err := metalProg.Run(context.Background(), backend.Request{
		Entry:  "embed",
		Inputs: map[string]any{"tokens": backend.NewTensorI32([]int{2}, []int32{0, 2})},
	})
	if err != nil {
		t.Fatalf("run metal: %v", err)
	}

	cudaTensor := cudaResult.Outputs["embeddings"].Data.(*backend.Tensor)
	metalTensor := metalResult.Outputs["embeddings"].Data.(*backend.Tensor)
	assertTensorClose(t, cudaTensor, metalTensor.Shape, metalTensor.F32)
}

func TestRunTinyScoreParityAcrossBackends(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_score", Preset: compiler.PresetTinyScore})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	cudaRuntime := New(cuda.New())
	cudaProg, err := cudaRuntime.Load(context.Background(), bundle.Artifact, tinyScoreWeights()...)
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	cudaResult, err := cudaProg.Run(context.Background(), backend.Request{
		Entry:  "score",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run cuda: %v", err)
	}

	metalRuntime := New(metal.New())
	metalProg, err := metalRuntime.Load(context.Background(), bundle.Artifact, tinyScoreWeights()...)
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	metalResult, err := metalProg.Run(context.Background(), backend.Request{
		Entry:  "score",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run metal: %v", err)
	}

	cudaTensor := cudaResult.Outputs["scores"].Data.(*backend.Tensor)
	metalTensor := metalResult.Outputs["scores"].Data.(*backend.Tensor)
	assertTensorClose(t, cudaTensor, metalTensor.Shape, metalTensor.F32)
	if metalResult.Outputs["scores"].Metadata["variant_entry"] != "cosine_metal" {
		t.Fatalf("metal variant_entry = %v, want cosine_metal", metalResult.Outputs["scores"].Metadata["variant_entry"])
	}
}

func TestRunTinyRerankParityAcrossBackends(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_rerank", Preset: compiler.PresetTinyRerank})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	cudaRuntime := New(cuda.New())
	cudaProg, err := cudaRuntime.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	cudaResult, err := cudaProg.Run(context.Background(), backend.Request{
		Entry:  "rerank",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run cuda: %v", err)
	}

	metalRuntime := New(metal.New())
	metalProg, err := metalRuntime.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	metalResult, err := metalProg.Run(context.Background(), backend.Request{
		Entry:  "rerank",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run metal: %v", err)
	}

	cudaTensor := cudaResult.Outputs["top_ids"].Data.(*backend.Tensor)
	metalTensor := metalResult.Outputs["top_ids"].Data.(*backend.Tensor)
	assertTensorI32(t, cudaTensor, metalTensor.Shape, metalTensor.I32)
	cudaScores := cudaResult.Outputs["top_scores"].Data.(*backend.Tensor)
	metalScores := metalResult.Outputs["top_scores"].Data.(*backend.Tensor)
	assertTensorClose(t, cudaScores, metalScores.Shape, metalScores.F32)
}

func TestRunTinySelectParityAcrossBackends(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_select", Preset: compiler.PresetTinySelect})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	cudaRuntime := New(cuda.New())
	cudaProg, err := cudaRuntime.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	cudaResult, err := cudaProg.Run(context.Background(), backend.Request{
		Entry:  "select_scores",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run cuda: %v", err)
	}

	metalRuntime := New(metal.New())
	metalProg, err := metalRuntime.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	metalResult, err := metalProg.Run(context.Background(), backend.Request{
		Entry:  "select_scores",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run metal: %v", err)
	}

	cudaTensor := cudaResult.Outputs["top_scores"].Data.(*backend.Tensor)
	metalTensor := metalResult.Outputs["top_scores"].Data.(*backend.Tensor)
	assertTensorClose(t, cudaTensor, metalTensor.Shape, metalTensor.F32)
}

func TestRunTinyRetrieveParityAcrossBackends(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_retrieve", Preset: compiler.PresetTinyRetrieve})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	cudaRuntime := New(cuda.New())
	cudaProg, err := cudaRuntime.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	cudaResult, err := cudaProg.Run(context.Background(), backend.Request{
		Entry:  "retrieve",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run cuda: %v", err)
	}

	metalRuntime := New(metal.New())
	metalProg, err := metalRuntime.Load(context.Background(), bundle.Artifact, tinyRerankWeights()...)
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	metalResult, err := metalProg.Run(context.Background(), backend.Request{
		Entry:  "retrieve",
		Inputs: map[string]any{"query": backend.NewTensorF16([]int{2}, []float32{1, 0})},
	})
	if err != nil {
		t.Fatalf("run metal: %v", err)
	}

	cudaIDs := cudaResult.Outputs["top_ids"].Data.(*backend.Tensor)
	metalIDs := metalResult.Outputs["top_ids"].Data.(*backend.Tensor)
	assertTensorI32(t, cudaIDs, metalIDs.Shape, metalIDs.I32)
	cudaScores := cudaResult.Outputs["top_scores"].Data.(*backend.Tensor)
	metalScores := metalResult.Outputs["top_scores"].Data.(*backend.Tensor)
	assertTensorClose(t, cudaScores, metalScores.Shape, metalScores.F32)
	cudaDocs := cudaResult.Outputs["top_docs"].Data.(*backend.Tensor)
	metalDocs := metalResult.Outputs["top_docs"].Data.(*backend.Tensor)
	assertTensorClose(t, cudaDocs, metalDocs.Shape, metalDocs.F32)
	if metalDocs.DType != "q4" {
		t.Fatalf("metal top_docs dtype = %q, want q4", metalDocs.DType)
	}
}

func TestRunTinyCandidatesEntryPoint(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_candidates", Preset: compiler.PresetTinyCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "rerank_candidates",
		Inputs: map[string]any{
			"query":         backend.NewTensorF16([]int{2}, []float32{1, 0}),
			"docs":          backend.NewTensorQ4([]int{3, 2}, []float32{1, 0, 0, 1, 1, 1}),
			"candidate_ids": backend.NewTensorI64([]int{3}, []int64{1001, 2002, 3003}),
		},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	topCandidateIDs := result.Outputs["top_candidate_ids"].Data.(*backend.Tensor)
	topScores := result.Outputs["top_scores"].Data.(*backend.Tensor)
	topDocs := result.Outputs["top_docs"].Data.(*backend.Tensor)
	assertTensorI64(t, topCandidateIDs, []int{2}, []int64{1001, 3003})
	assertTensorClose(t, topScores, []int{2}, []float32{1, 0.70710677})
	assertTensorClose(t, topDocs, []int{2, 2}, []float32{
		1, 0,
		1, 1,
	})
	if topDocs.DType != "q4" {
		t.Fatalf("top_docs dtype = %q, want q4", topDocs.DType)
	}
}

func TestRunTinyCandidatesParityAcrossBackends(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_candidates", Preset: compiler.PresetTinyCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	cudaRuntime := New(cuda.New())
	cudaProg, err := cudaRuntime.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	cudaResult, err := cudaProg.Run(context.Background(), backend.Request{
		Entry: "rerank_candidates",
		Inputs: map[string]any{
			"query":         backend.NewTensorF16([]int{2}, []float32{1, 0}),
			"docs":          backend.NewTensorQ4([]int{3, 2}, []float32{1, 0, 0, 1, 1, 1}),
			"candidate_ids": backend.NewTensorI64([]int{3}, []int64{1001, 2002, 3003}),
		},
	})
	if err != nil {
		t.Fatalf("run cuda: %v", err)
	}

	metalRuntime := New(metal.New())
	metalProg, err := metalRuntime.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	metalResult, err := metalProg.Run(context.Background(), backend.Request{
		Entry: "rerank_candidates",
		Inputs: map[string]any{
			"query":         backend.NewTensorF16([]int{2}, []float32{1, 0}),
			"docs":          backend.NewTensorQ4([]int{3, 2}, []float32{1, 0, 0, 1, 1, 1}),
			"candidate_ids": backend.NewTensorI64([]int{3}, []int64{1001, 2002, 3003}),
		},
	})
	if err != nil {
		t.Fatalf("run metal: %v", err)
	}

	cudaIDs := cudaResult.Outputs["top_candidate_ids"].Data.(*backend.Tensor)
	metalIDs := metalResult.Outputs["top_candidate_ids"].Data.(*backend.Tensor)
	assertTensorI64(t, cudaIDs, metalIDs.Shape, metalIDs.I64)
	cudaScores := cudaResult.Outputs["top_scores"].Data.(*backend.Tensor)
	metalScores := metalResult.Outputs["top_scores"].Data.(*backend.Tensor)
	assertTensorClose(t, cudaScores, metalScores.Shape, metalScores.F32)
	cudaDocs := cudaResult.Outputs["top_docs"].Data.(*backend.Tensor)
	metalDocs := metalResult.Outputs["top_docs"].Data.(*backend.Tensor)
	assertTensorClose(t, cudaDocs, metalDocs.Shape, metalDocs.F32)
	if metalDocs.DType != "q4" {
		t.Fatalf("metal top_docs dtype = %q, want q4", metalDocs.DType)
	}
}

func TestRunTinyBatchCandidatesEntryPoint(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_batch_candidates", Preset: compiler.PresetTinyBatchCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "rerank_candidates_batch",
		Inputs: map[string]any{
			"queries": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			}),
			"docs": backend.NewTensorQ4([]int{2, 3, 2}, []float32{
				1, 0,
				0, 1,
				1, 1,
				0, 1,
				1, 0,
				1, 1,
			}),
			"candidate_ids": backend.NewTensorI64([]int{2, 3}, []int64{
				1001, 2002, 3003,
				4004, 5005, 6006,
			}),
		},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	topCandidateIDs := result.Outputs["top_candidate_ids"].Data.(*backend.Tensor)
	topScores := result.Outputs["top_scores"].Data.(*backend.Tensor)
	topDocs := result.Outputs["top_docs"].Data.(*backend.Tensor)
	assertTensorI64(t, topCandidateIDs, []int{2, 2}, []int64{
		1001, 3003,
		4004, 6006,
	})
	assertTensorClose(t, topScores, []int{2, 2}, []float32{
		1, 0.70710677,
		1, 0.70710677,
	})
	assertTensorClose(t, topDocs, []int{2, 2, 2}, []float32{
		1, 0,
		1, 1,
		0, 1,
		1, 1,
	})
	if topDocs.DType != "q4" {
		t.Fatalf("top_docs dtype = %q, want q4", topDocs.DType)
	}
}

func TestRunTinyBatchCandidatesParityAcrossBackends(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_batch_candidates", Preset: compiler.PresetTinyBatchCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	inputs := map[string]any{
		"queries": backend.NewTensorF16([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
		"docs": backend.NewTensorQ4([]int{2, 3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
			0, 1,
			1, 0,
			1, 1,
		}),
		"candidate_ids": backend.NewTensorI64([]int{2, 3}, []int64{
			1001, 2002, 3003,
			4004, 5005, 6006,
		}),
	}

	cudaRuntime := New(cuda.New())
	cudaProg, err := cudaRuntime.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	cudaResult, err := cudaProg.Run(context.Background(), backend.Request{
		Entry:  "rerank_candidates_batch",
		Inputs: inputs,
	})
	if err != nil {
		t.Fatalf("run cuda: %v", err)
	}

	metalRuntime := New(metal.New())
	metalProg, err := metalRuntime.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	metalResult, err := metalProg.Run(context.Background(), backend.Request{
		Entry:  "rerank_candidates_batch",
		Inputs: inputs,
	})
	if err != nil {
		t.Fatalf("run metal: %v", err)
	}

	cudaIDs := cudaResult.Outputs["top_candidate_ids"].Data.(*backend.Tensor)
	metalIDs := metalResult.Outputs["top_candidate_ids"].Data.(*backend.Tensor)
	assertTensorI64(t, cudaIDs, metalIDs.Shape, metalIDs.I64)
	cudaScores := cudaResult.Outputs["top_scores"].Data.(*backend.Tensor)
	metalScores := metalResult.Outputs["top_scores"].Data.(*backend.Tensor)
	assertTensorClose(t, cudaScores, metalScores.Shape, metalScores.F32)
	cudaDocs := cudaResult.Outputs["top_docs"].Data.(*backend.Tensor)
	metalDocs := metalResult.Outputs["top_docs"].Data.(*backend.Tensor)
	assertTensorClose(t, cudaDocs, metalDocs.Shape, metalDocs.F32)
	if metalDocs.DType != "q4" {
		t.Fatalf("metal top_docs dtype = %q, want q4", metalDocs.DType)
	}
}

func TestRunTinyPackedCandidatesEntryPoint(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_packed_candidates", Preset: compiler.PresetTinyPackedCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "rerank_candidates_packed",
		Inputs: map[string]any{
			"query":         backend.NewTensorF16([]int{2}, []float32{1, 0}),
			"docs":          backend.NewTensorQ4([]int{3, 2}, []float32{1, 0, 0, 1, 1, 1}),
			"candidate_ids": backend.NewTensorI64([]int{3}, []int64{1001, 2002, 3003}),
		},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	output := result.Outputs["candidates"]
	if output.Type.Kind != eosartifact.ValueCandidatePack || output.Type.CandidatePack == nil {
		t.Fatalf("output kind = %+v, want candidate_pack", output.Type)
	}
	pack, ok := output.Data.(*backend.CandidatePack)
	if !ok {
		t.Fatalf("output data type = %T, want *backend.CandidatePack", output.Data)
	}
	assertTensorI64(t, pack.IDs, []int{2}, []int64{1001, 3003})
	assertTensorClose(t, pack.Scores, []int{2}, []float32{1, 0.70710677})
	assertTensorClose(t, pack.Docs, []int{2, 2}, []float32{
		1, 0,
		1, 1,
	})
	if pack.Docs.DType != "q4" {
		t.Fatalf("packed docs dtype = %q, want q4", pack.Docs.DType)
	}
}

func TestRunTinyPackedCandidatesParityAcrossBackends(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_packed_candidates", Preset: compiler.PresetTinyPackedCandidates})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	inputs := map[string]any{
		"query":         backend.NewTensorF16([]int{2}, []float32{1, 0}),
		"docs":          backend.NewTensorQ4([]int{3, 2}, []float32{1, 0, 0, 1, 1, 1}),
		"candidate_ids": backend.NewTensorI64([]int{3}, []int64{1001, 2002, 3003}),
	}

	cudaRuntime := New(cuda.New())
	cudaProg, err := cudaRuntime.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	cudaResult, err := cudaProg.Run(context.Background(), backend.Request{
		Entry:  "rerank_candidates_packed",
		Inputs: inputs,
	})
	if err != nil {
		t.Fatalf("run cuda: %v", err)
	}

	metalRuntime := New(metal.New())
	metalProg, err := metalRuntime.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	metalResult, err := metalProg.Run(context.Background(), backend.Request{
		Entry:  "rerank_candidates_packed",
		Inputs: inputs,
	})
	if err != nil {
		t.Fatalf("run metal: %v", err)
	}

	cudaPack := cudaResult.Outputs["candidates"].Data.(*backend.CandidatePack)
	metalPack := metalResult.Outputs["candidates"].Data.(*backend.CandidatePack)
	assertTensorI64(t, cudaPack.IDs, metalPack.IDs.Shape, metalPack.IDs.I64)
	assertTensorClose(t, cudaPack.Scores, metalPack.Scores.Shape, metalPack.Scores.F32)
	assertTensorClose(t, cudaPack.Docs, metalPack.Docs.Shape, metalPack.Docs.F32)
}

func TestRunBatchedPackedCandidatesEntryPoint(t *testing.T) {
	src := []byte(`
pipeline rerank_candidates_packed_batch(queries: f16[Q, D], docs: q4[Q, N, D], candidate_ids: i64[Q, N]) -> candidate_pack[Q, 2, D] {
    let scores = cosine(queries, docs)
    let top_indices = topk(scores, 2)
    let top_candidate_ids = gather(candidate_ids, top_indices)
    let top_scores = gather(scores, top_indices)
    let top_docs = gather(docs, top_indices)
    return pack_candidates(top_candidate_ids, top_scores, top_docs)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "packed_batch"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "rerank_candidates_packed_batch",
		Inputs: map[string]any{
			"queries": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			}),
			"docs": backend.NewTensorQ4([]int{2, 3, 2}, []float32{
				1, 0,
				0, 1,
				1, 1,
				0, 1,
				1, 0,
				1, 1,
			}),
			"candidate_ids": backend.NewTensorI64([]int{2, 3}, []int64{
				1001, 2002, 3003,
				4004, 5005, 6006,
			}),
		},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	output := result.Outputs["result"]
	if output.Type.Kind != eosartifact.ValueCandidatePack || output.Type.CandidatePack == nil {
		t.Fatalf("output kind = %+v, want candidate_pack", output.Type)
	}
	pack := output.Data.(*backend.CandidatePack)
	assertTensorI64(t, pack.IDs, []int{2, 2}, []int64{
		1001, 3003,
		4004, 6006,
	})
	assertTensorClose(t, pack.Scores, []int{2, 2}, []float32{
		1, 0.70710677,
		1, 0.70710677,
	})
	assertTensorClose(t, pack.Docs, []int{2, 2, 2}, []float32{
		1, 0,
		1, 1,
		0, 1,
		1, 1,
	})
}

func TestRunBatchedScoreFallsBackToHost(t *testing.T) {
	src := []byte(`
pipeline score_batch(queries: f16[Q, D], docs: q4[Q, N, D]) -> (scores: f32[Q, N]) {
    return cosine(queries, docs)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "score_batch"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "score_batch",
		Inputs: map[string]any{
			"queries": backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			}),
			"docs": backend.NewTensorQ4([]int{2, 3, 2}, []float32{
				1, 0,
				0, 1,
				1, 1,
				0, 1,
				1, 0,
				1, 1,
			}),
		},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	output := result.Outputs["scores"]
	tensor := output.Data.(*backend.Tensor)
	assertTensorClose(t, tensor, []int{2, 3}, []float32{
		1, 0, 0.70710677,
		1, 0, 0.70710677,
	})
	if output.Metadata["device_execution"] != false {
		t.Fatalf("device_execution = %v, want false", output.Metadata["device_execution"])
	}
	if output.Metadata["execution_mode"] != "host_fallback" {
		t.Fatalf("execution_mode = %v, want host_fallback", output.Metadata["execution_mode"])
	}
	if output.Metadata["launch_api"] != "host_reference" {
		t.Fatalf("launch_api = %v, want host_reference", output.Metadata["launch_api"])
	}
}

func TestRunDirectQuantizedBuiltinScoreOps(t *testing.T) {
	if !cudaNativeRuntimeTestsAvailable {
		t.Skip("CUDA native runtime dispatch requires linux cgo")
	}
	cases := []struct {
		name     string
		op       string
		want     []float32
		producer string
	}{
		{name: "dot", op: "dot", want: []float32{1, 0, 1}, producer: "kernel:dot"},
		{name: "cosine", op: "cosine", want: []float32{1, 0, 0.70710677}, producer: "kernel:cosine"},
		{name: "l2_distance", op: "l2_distance", want: []float32{0, 1.4142135, 1}, producer: "kernel:l2_distance"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			src := []byte(`
pipeline score(query: f16[D], docs: q4[N, D]) -> f32[N] {
    return ` + tc.op + `(query, docs)
}
`)

			bundle, err := compiler.Build(src, compiler.Options{ModuleName: tc.name})
			if err != nil {
				t.Fatalf("build: %v", err)
			}

			rt := New(cuda.New())
			prog, err := rt.Load(context.Background(), bundle.Artifact)
			if err != nil {
				t.Fatalf("load: %v", err)
			}

			result, err := prog.Run(context.Background(), backend.Request{
				Entry: "score",
				Inputs: map[string]any{
					"query": backend.NewTensorF16([]int{2}, []float32{1, 0}),
					"docs": backend.NewTensorQ4([]int{3, 2}, []float32{
						1, 0,
						0, 1,
						1, 1,
					}),
				},
			})
			if err != nil {
				t.Fatalf("run: %v", err)
			}

			output, ok := result.Outputs["scores"]
			if !ok {
				t.Fatalf("missing scores output: %+v", result.Outputs)
			}
			if output.Producer != tc.producer {
				t.Fatalf("output producer = %q, want %q", output.Producer, tc.producer)
			}
			tensor, ok := output.Data.(*backend.Tensor)
			if !ok {
				t.Fatalf("output data type = %T, want *backend.Tensor", output.Data)
			}
			assertTensorClose(t, tensor, []int{3}, tc.want)
			if output.Metadata["device_execution"] != true {
				t.Fatalf("device_execution = %v, want true", output.Metadata["device_execution"])
			}
		})
	}
}

func TestRunRejectsMissingEntryInput(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	_, err = prog.Run(context.Background(), backend.Request{Entry: "embed", Inputs: map[string]any{}})
	if err == nil {
		t.Fatal("expected missing input error")
	}
	if !strings.Contains(err.Error(), "tokens") {
		t.Fatalf("expected tokens missing error, got %v", err)
	}
}

func TestLoadRejectsInconsistentParamShapes(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New())
	_, err = rt.Load(
		context.Background(),
		bundle.Artifact,
		WithWeight("token_embedding", backend.NewTensorF16([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		})),
		WithWeight("projection", backend.NewTensorF16([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		})),
	)
	if err == nil {
		t.Fatal("expected param shape mismatch")
	}
	if !strings.Contains(err.Error(), "symbol \"D\" mismatch") {
		t.Fatalf("expected D mismatch, got %v", err)
	}
}

func TestRunRejectsInputShapeMismatch(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_decode", Preset: compiler.PresetTinyDecode})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New())
	prog, err := rt.Load(
		context.Background(),
		bundle.Artifact,
		WithWeight("wq", backend.NewTensorF16([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		})),
	)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	_, err = prog.Run(context.Background(), backend.Request{
		Entry: "decode_step",
		Inputs: map[string]any{
			"x": backend.NewTensorF16([]int{2, 3}, []float32{
				1, 0, 0,
				0, 1, 0,
			}),
			"cache": backend.NewKVCache(nil),
		},
	})
	if err == nil {
		t.Fatal("expected input shape mismatch")
	}
	if !strings.Contains(err.Error(), "symbol \"D\" mismatch") {
		t.Fatalf("expected D mismatch, got %v", err)
	}
}

func TestRunIdentityPipelineUsesAliasSteps(t *testing.T) {
	src := []byte(`
pipeline identity(x: f16[T, D]) -> f16[T, D] {
    let forwarded = x
    return forwarded
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "identity"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	input := backend.NewTensorF16([]int{2, 2}, []float32{
		1, 2,
		3, 4,
	})
	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  "identity",
		Inputs: map[string]any{"x": input},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	output, ok := result.Outputs["result"]
	if !ok {
		t.Fatalf("missing result output: %+v", result.Outputs)
	}
	tensor, ok := output.Data.(*backend.Tensor)
	if !ok {
		t.Fatalf("output data type = %T, want *backend.Tensor", output.Data)
	}
	assertTensorClose(t, tensor, []int{2, 2}, []float32{
		1, 2,
		3, 4,
	})
	if output.Producer != "input:x" {
		t.Fatalf("output producer = %q, want input:x", output.Producer)
	}
	if got := len(result.Trace); got != 3 {
		t.Fatalf("trace len = %d, want 3", got)
	}
	if result.Trace[0].Kind != eosartifact.StepAlias || result.Trace[1].Kind != eosartifact.StepAlias {
		t.Fatalf("expected alias trace steps, got %+v", result.Trace)
	}
}

func tinyEmbedWeights() []LoadOption {
	return []LoadOption{
		WithWeight("token_embedding", backend.NewTensorF16([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		})),
		WithWeight("projection", backend.NewTensorF16([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		})),
	}
}

func tinyScoreWeights() []LoadOption {
	return []LoadOption{
		WithWeight("docs", backend.NewTensorQ4([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		})),
	}
}

func tinyRerankWeights() []LoadOption {
	return []LoadOption{
		WithWeight("docs", backend.NewTensorQ4([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		})),
	}
}

func newLazyStagedParamModule() *eosartifact.Module {
	mod := eosartifact.NewModule("lazy_params")
	valueType := eosartifact.ValueType{
		Kind: eosartifact.ValueTensor,
		Tensor: &eosartifact.TensorType{
			DType: "f16",
			Shape: []string{"T", "D"},
		},
	}
	mod.Params = []eosartifact.Param{
		{Name: "used", Type: valueType, Binding: "used"},
		{Name: "unused", Type: valueType, Binding: "unused"},
	}
	mod.EntryPoints = []eosartifact.EntryPoint{
		{
			Name: "serve",
			Kind: eosartifact.EntryPointPipeline,
			Outputs: []eosartifact.ValueBinding{
				{Name: "result", Type: valueType},
			},
		},
	}
	mod.Buffers = []eosartifact.Buffer{
		{Name: "result", DType: "f16", Shape: []string{"T", "D"}},
	}
	mod.Steps = []eosartifact.Step{
		{
			Entry:   "serve",
			Kind:    eosartifact.StepAlias,
			Name:    "forward_used",
			Inputs:  []string{"used"},
			Outputs: []string{"result"},
		},
		{
			Entry:   "serve",
			Kind:    eosartifact.StepReturn,
			Name:    "return_result",
			Outputs: []string{"result"},
		},
	}
	return mod
}

func assertTensorClose(t *testing.T, tensor *backend.Tensor, wantShape []int, want []float32) {
	t.Helper()
	if tensor == nil {
		t.Fatal("tensor is nil")
	}
	if len(tensor.Shape) != len(wantShape) {
		t.Fatalf("tensor rank = %d, want %d", len(tensor.Shape), len(wantShape))
	}
	for i := range wantShape {
		if tensor.Shape[i] != wantShape[i] {
			t.Fatalf("tensor shape[%d] = %d, want %d", i, tensor.Shape[i], wantShape[i])
		}
	}
	if len(tensor.F32) != len(want) {
		t.Fatalf("tensor values len = %d, want %d", len(tensor.F32), len(want))
	}
	for i, got := range tensor.F32 {
		assertClose(t, got, want[i], 0.0005)
	}
}

func assertTensorI32(t *testing.T, tensor *backend.Tensor, wantShape []int, want []int32) {
	t.Helper()
	if tensor == nil {
		t.Fatal("tensor is nil")
	}
	if len(tensor.Shape) != len(wantShape) {
		t.Fatalf("tensor rank = %d, want %d", len(tensor.Shape), len(wantShape))
	}
	for i := range wantShape {
		if tensor.Shape[i] != wantShape[i] {
			t.Fatalf("tensor shape[%d] = %d, want %d", i, tensor.Shape[i], wantShape[i])
		}
	}
	if len(tensor.I32) != len(want) {
		t.Fatalf("tensor values len = %d, want %d", len(tensor.I32), len(want))
	}
	for i, got := range tensor.I32 {
		if got != want[i] {
			t.Fatalf("tensor[%d] = %d, want %d", i, got, want[i])
		}
	}
}

func assertTensorI64(t *testing.T, tensor *backend.Tensor, wantShape []int, want []int64) {
	t.Helper()
	if tensor == nil {
		t.Fatal("tensor is nil")
	}
	if len(tensor.Shape) != len(wantShape) {
		t.Fatalf("tensor rank = %d, want %d", len(tensor.Shape), len(wantShape))
	}
	for i := range wantShape {
		if tensor.Shape[i] != wantShape[i] {
			t.Fatalf("tensor shape[%d] = %d, want %d", i, tensor.Shape[i], wantShape[i])
		}
	}
	if len(tensor.I64) != len(want) {
		t.Fatalf("tensor values len = %d, want %d", len(tensor.I64), len(want))
	}
	for i, got := range tensor.I64 {
		if got != want[i] {
			t.Fatalf("tensor[%d] = %d, want %d", i, got, want[i])
		}
	}
}

func assertClose(t *testing.T, got, want, tol float32) {
	t.Helper()
	diff := got - want
	if diff < -tol || diff > tol {
		t.Fatalf("value = %f, want %f (tol=%f)", got, want, tol)
	}
}
