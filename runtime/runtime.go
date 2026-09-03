package eosruntime

import (
	"context"
	"fmt"
	"os"
	"strings"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

// Runtime owns backend selection and module loading.
type Runtime struct {
	backends []backend.Backend
}

// LoadOption mutates runtime loading behavior.
type LoadOption func(*loadConfig)

type loadConfig struct {
	weights           map[string]backend.WeightBinding
	candidateMetadata map[int64]map[string]string
	memoryPlan        *MemoryPlan
	packageManifest   *PackageManifest
	postPoolTransform *AOQTGivensTransform
	requireBackend    eosartifact.BackendKind
}

// EnvRequireBackend, when set to a backend kind (e.g. "cuda", "metal",
// "vulkan", "directml", "webgpu"), pins Load to that backend and disables
// silent multi-backend fallback: if the required backend rejects the
// module for any reason -- CanLoad returns false, a required capability is
// missing, or Load itself errors (e.g. the CUDA stale-RoPE-ABI guard added
// in f0782712 rejecting a pre-fix artifact) -- Load returns that rejection
// immediately instead of trying the next registered backend. This is the
// CLI-reachable escape hatch for eval/serving entry points (cmd/eos,
// cmd/retrievaldump) that must not silently drift onto a slower or
// semantically different backend, such as the CPU-backed metal stub on
// Linux (runtime/backends/metal/native_stub.go): export
// EOS_REQUIRE_BACKEND=cuda before invoking the eos CLI, or pass
// WithRequireBackend(eosartifact.BackendCUDA) programmatically.
// WithRequireBackend takes precedence over the env var when both are set.
const EnvRequireBackend = "EOS_REQUIRE_BACKEND"

// WithRequireBackend forces Load to return a backend's rejection instead of
// silently falling back to a later candidate when kind rejects the module.
// See EnvRequireBackend for the equivalent env var (checked when this
// option is not supplied) and for the full fail-loud rationale.
func WithRequireBackend(kind eosartifact.BackendKind) LoadOption {
	return func(cfg *loadConfig) {
		cfg.requireBackend = kind
	}
}

// Program is a loaded executable module.
type Program struct {
	module            *eosartifact.Module
	executor          backend.Executor
	candidateMetadata map[int64]map[string]string
	memoryPlan        *MemoryPlan
}

// New constructs a runtime from registered backends.
func New(backends ...backend.Backend) *Runtime {
	cp := append([]backend.Backend(nil), backends...)
	return &Runtime{backends: cp}
}

// Load selects a compatible backend and prepares the program.
//
// Backend selection falls back candidate by candidate: if backends[0]
// rejects the module, backends[1] is tried, and so on. A rejected
// candidate is recorded in reasons for the final aggregate error in case
// every candidate rejects the module. When a fallback candidate ultimately
// succeeds, every earlier candidate whose Load call actually failed (as
// opposed to being merely unsupported or missing a capability) is reported
// with a stderr warning naming the failed backend, its verbatim error, and
// the backend Load fell back to -- see warnBackendFallback. To refuse
// fallback entirely and surface a rejection instead, set EnvRequireBackend
// or pass WithRequireBackend.
func (rt *Runtime) Load(ctx context.Context, mod *eosartifact.Module, opts ...LoadOption) (*Program, error) {
	if mod == nil {
		return nil, fmt.Errorf("nil module")
	}
	if err := mod.Validate(); err != nil {
		return nil, err
	}
	cfg := loadConfig{weights: map[string]backend.WeightBinding{}}
	for _, opt := range opts {
		opt(&cfg)
	}
	if err := applyMemoryPlanToWeights(mod, cfg.weights, cfg.memoryPlan); err != nil {
		return nil, err
	}
	if err := validateParamBindings(mod, cfg.weights); err != nil {
		return nil, err
	}

	requireBackend := cfg.requireBackend
	if requireBackend == "" {
		if v := strings.TrimSpace(os.Getenv(EnvRequireBackend)); v != "" {
			requireBackend = eosartifact.BackendKind(strings.ToLower(v))
		}
	}
	if requireBackend != "" && !rt.hasBackend(requireBackend) {
		return nil, fmt.Errorf("eos-runtime: required backend %q is not registered in this runtime (see %s / WithRequireBackend); registered backends: %s", requireBackend, EnvRequireBackend, strings.Join(rt.backendKinds(), ", "))
	}

	var reasons []string
	var loadFailures []backendLoadFailure
	for _, candidate := range rt.backends {
		kind := candidate.Kind()
		strict := requireBackend != "" && kind == requireBackend
		if !candidate.CanLoad(mod) {
			if strict {
				return nil, fmt.Errorf("eos-runtime: required backend %q rejected module %q: unsupported backend (refusing to fall back; see %s / WithRequireBackend)", kind, mod.Name, EnvRequireBackend)
			}
			reasons = append(reasons, fmt.Sprintf("%s: unsupported backend", kind))
			continue
		}
		if missing := missingBackendCapabilities(candidate, mod); len(missing) > 0 {
			if strict {
				return nil, fmt.Errorf("eos-runtime: required backend %q rejected module %q: missing capabilities [%s] (refusing to fall back; see %s / WithRequireBackend)", kind, mod.Name, strings.Join(missing, ", "), EnvRequireBackend)
			}
			reasons = append(reasons, fmt.Sprintf("%s: missing capabilities [%s]", kind, strings.Join(missing, ", ")))
			continue
		}
		exec, err := loadBackendExecutor(ctx, candidate, mod, cfg.weights, cfg.packageManifest)
		if err != nil {
			if strict {
				return nil, fmt.Errorf("eos-runtime: required backend %q failed to load module %q: %w (refusing to fall back; see %s / WithRequireBackend)", kind, mod.Name, err, EnvRequireBackend)
			}
			reasons = append(reasons, fmt.Sprintf("%s: %v", kind, err))
			loadFailures = append(loadFailures, backendLoadFailure{kind: kind, err: err})
			continue
		}
		warnBackendFallback(mod, loadFailures, kind)
		return &Program{
			module:            mod,
			executor:          exec,
			candidateMetadata: cloneCandidateMetadataLookup(cfg.candidateMetadata),
			memoryPlan:        cloneMemoryPlan(cfg.memoryPlan),
		}, nil
	}
	if len(reasons) == 0 {
		return nil, fmt.Errorf("no compatible backend for module %q", mod.Name)
	}
	return nil, fmt.Errorf("no compatible backend for module %q: %s", mod.Name, strings.Join(reasons, "; "))
}

// hasBackend reports whether kind is among rt's registered backends.
func (rt *Runtime) hasBackend(kind eosartifact.BackendKind) bool {
	for _, candidate := range rt.backends {
		if candidate.Kind() == kind {
			return true
		}
	}
	return false
}

// backendKinds lists rt's registered backend kinds, in registration order.
func (rt *Runtime) backendKinds() []string {
	kinds := make([]string, 0, len(rt.backends))
	for _, candidate := range rt.backends {
		kinds = append(kinds, string(candidate.Kind()))
	}
	return kinds
}

// backendLoadFailure records a candidate backend whose Load call returned
// an error during Runtime.Load's fallback search.
type backendLoadFailure struct {
	kind eosartifact.BackendKind
	err  error
}

// warnBackendFallback emits one stderr line per backend whose Load call
// failed before Load fell back to selected. Each line names the failed
// backend, preserves its error verbatim -- e.g. the CUDA stale-RoPE-ABI
// guard's "recompile/reseal the module" message -- and names the backend
// that was ultimately selected, so a Load rejection is never silently
// swallowed by the fallback path. Backends skipped only because CanLoad
// returned false or a capability was missing are routine multi-backend
// routing, not failures, so they are not warned about here (they still
// appear in the aggregate error if every candidate is rejected). Callers
// that must not tolerate any fallback should use EnvRequireBackend /
// WithRequireBackend instead of relying on this warning.
func warnBackendFallback(mod *eosartifact.Module, failures []backendLoadFailure, selected eosartifact.BackendKind) {
	for _, failure := range failures {
		fmt.Fprintf(os.Stderr, "eos-runtime: WARNING backend %q failed to load module %q: %v -- falling back to backend %q (set %s=%s to fail instead of falling back)\n",
			failure.kind, mod.Name, failure.err, selected, EnvRequireBackend, failure.kind)
	}
}

// LoadFile reads a serialized .mll artifact and loads it.
func (rt *Runtime) LoadFile(ctx context.Context, path string, opts ...LoadOption) (*Program, error) {
	mod, err := eosartifact.ReadFile(path)
	if err != nil {
		return nil, err
	}
	manifestPath := ResolvePackageManifestPath(path)
	if _, err := os.Stat(manifestPath); err == nil {
		manifest, err := ReadPackageManifestFile(manifestPath)
		if err != nil {
			return nil, err
		}
		opts = append(opts, WithPackageManifest(manifest))
	}
	return rt.Load(ctx, mod, opts...)
}

// Backend reports the selected backend.
func (p *Program) Backend() eosartifact.BackendKind {
	if p == nil || p.executor == nil {
		return ""
	}
	return p.executor.Backend()
}

// Run executes the loaded program.
func (p *Program) Run(ctx context.Context, req backend.Request) (backend.Result, error) {
	if p == nil || p.executor == nil {
		return backend.Result{}, fmt.Errorf("program is not loaded")
	}
	return p.executor.Run(ctx, req)
}

func (p *Program) MemoryPlan() *MemoryPlan {
	if p == nil {
		return nil
	}
	return cloneMemoryPlan(p.memoryPlan)
}

// WithWeight binds a Eos param to runtime-managed data.
func WithWeight(name string, data any) LoadOption {
	return func(cfg *loadConfig) {
		if cfg.weights == nil {
			cfg.weights = map[string]backend.WeightBinding{}
		}
		cfg.weights[name] = backend.WeightBinding{Name: name, Data: data}
	}
}

// WithCandidateMetadata binds external candidate metadata by candidate id.
func WithCandidateMetadata(metadata map[int64]map[string]string) LoadOption {
	return func(cfg *loadConfig) {
		cfg.candidateMetadata = cloneCandidateMetadataLookup(metadata)
	}
}

func WithMemoryPlan(plan MemoryPlan) LoadOption {
	return func(cfg *loadConfig) {
		cfg.memoryPlan = cloneMemoryPlan(&plan)
	}
}

func WithPackageManifest(manifest PackageManifest) LoadOption {
	return func(cfg *loadConfig) {
		cp := manifest
		cfg.packageManifest = &cp
	}
}

func WithPostPoolTransform(transform AOQTGivensTransform) LoadOption {
	return func(cfg *loadConfig) {
		cp := transform
		cfg.postPoolTransform = &cp
	}
}

func validateParamBindings(mod *eosartifact.Module, weights map[string]backend.WeightBinding) error {
	bindings := map[string]int{}
	for _, param := range mod.Params {
		weight, ok := weights[param.Name]
		if !ok {
			return fmt.Errorf("missing weight binding for param %q", param.Name)
		}
		if _, _, err := backend.PreviewValueWithBindings(param.Type, weight.Data, bindings); err != nil {
			return fmt.Errorf("param %q: %w", param.Name, err)
		}
	}
	return nil
}

func applyMemoryPlanToWeights(mod *eosartifact.Module, weights map[string]backend.WeightBinding, plan *MemoryPlan) error {
	if mod == nil || plan == nil {
		return nil
	}
	if plan.ModuleName != "" && plan.ModuleName != mod.Name {
		return fmt.Errorf("memory plan module %q does not match module %q", plan.ModuleName, mod.Name)
	}
	planned := make(map[string]WeightMemoryPlan, len(plan.Weights))
	for _, item := range plan.Weights {
		planned[item.Name] = item
	}
	for _, param := range mod.Params {
		weight, ok := weights[param.Name]
		if !ok {
			continue
		}
		item, ok := planned[param.Name]
		if !ok {
			return fmt.Errorf("memory plan missing weight %q", param.Name)
		}
		weight.Residency = string(item.Residency)
		weight.AccessCount = item.AccessCount
		weights[param.Name] = weight
	}
	return nil
}

func cloneCandidateMetadataLookup(in map[int64]map[string]string) map[int64]map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[int64]map[string]string, len(in))
	for id, meta := range in {
		out[id] = cloneCandidateMetadata(meta)
	}
	return out
}

func cloneCandidateMetadata(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func missingBackendCapabilities(candidate backend.Backend, mod *eosartifact.Module) []string {
	if candidate == nil || mod == nil {
		return nil
	}
	provider, ok := candidate.(backend.CapabilityProvider)
	if !ok {
		return eosartifact.MissingCapabilities(mod.Requirements.Capabilities, nil)
	}
	return eosartifact.MissingCapabilities(mod.Requirements.Capabilities, provider.Capabilities())
}

type cacheKeyLoader interface {
	LoadWithCacheKey(ctx context.Context, mod *eosartifact.Module, weights map[string]backend.WeightBinding, cacheKey string) (backend.Executor, error)
}

func loadBackendExecutor(ctx context.Context, candidate backend.Backend, mod *eosartifact.Module, weights map[string]backend.WeightBinding, manifest *PackageManifest) (backend.Executor, error) {
	cacheKey := ""
	if manifest != nil {
		cacheKey = manifest.CacheKey()
	}
	if loader, ok := candidate.(cacheKeyLoader); ok {
		return loader.LoadWithCacheKey(ctx, mod, weights, cacheKey)
	}
	return candidate.Load(ctx, mod, weights)
}
