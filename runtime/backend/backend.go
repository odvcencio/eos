package backend

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"

	eosartifact "m31labs.dev/eos/artifact/eos"
)

// Request is an execution request for a loaded program.
type Request struct {
	Entry  string
	Inputs map[string]any
}

// Value is a typed runtime value flowing through the symbolic executor.
type Value struct {
	Type     eosartifact.ValueType
	Data     any
	Producer string
	Inputs   []string
	Metadata map[string]any
}

// TraceStep records one executed plan step.
type TraceStep struct {
	Entry   string
	Kind    eosartifact.StepKind
	Name    string
	Kernel  string
	Variant string
	Inputs  []string
	Outputs []string
}

// WeightBinding attaches an external param binding to runtime-managed weight data.
type WeightBinding struct {
	Name        string
	Data        any
	Residency   string
	AccessCount int
}

// CompiledKernel is a backend-selected variant cached at load time.
type CompiledKernel struct {
	Name       string
	Backend    eosartifact.BackendKind
	Entry      string
	Source     string
	SourceHash string
	Meta       map[string]string
}

// NativeKernelProgram is a backend-owned compiled kernel program.
type NativeKernelProgram struct {
	Compiled     CompiledKernel
	LaunchConfig map[string]any
	Fallback     func(inputs []*Tensor) ([]*Tensor, error)
	Run          func(inputs []*Tensor) ([]*Tensor, error)
}

// KernelDispatchResult is the result of dispatching a compiled backend kernel.
type KernelDispatchResult struct {
	Outputs      []*Tensor
	VariantEntry string
	SourceHash   string
	Metadata     map[string]any
}

// StepDispatchResult is the result of dispatching a backend-owned non-kernel step.
type StepDispatchResult struct {
	Outputs      []*Tensor
	VariantEntry string
	SourceHash   string
	Metadata     map[string]any
}

// OptimizerUpdateConfig describes one parameter update.
type OptimizerUpdateConfig struct {
	Optimizer    string
	Step         int
	LearningRate float32
	WeightDecay  float32
	Beta1        float32
	Beta2        float32
	Epsilon      float32
	Scale        float32
	DeferSync    bool
}

// MatMulAcceleratorStats summarizes backend-owned matmul prep, residency, and run activity.
type MatMulAcceleratorStats struct {
	BindCalls          int64
	UploadedBytes      int64
	QuantizePasses     int64
	QuantizedBytes     int64
	BindNanos          int64
	QuantizeNanos      int64
	BoundMatrices      int64
	RunCalls           int64
	BoundLeftCalls     int64
	BoundRightCalls    int64
	RunUploadedBytes   int64
	RunDownloadedBytes int64
	RunNanos           int64
}

// OptimizerAcceleratorStats summarizes backend-owned optimizer update activity.
type OptimizerAcceleratorStats struct {
	LogicalSteps           int64
	TensorUpdateCalls      int64
	UpdateCalls            int64
	DeferredSyncUpdates    int64
	SyncCalls              int64
	ForcedSyncCalls        int64
	LastForcedSyncReason   string
	UploadedBytes          int64
	DownloadedBytes        int64
	UploadedBytesPerStep   float64
	DownloadedBytesPerStep float64
	UpdateNanos            int64
	SyncNanos              int64
	ResidentParams         int64
}

// OptimizerResidentParameter is a backend-owned device parameter reference.
// It carries a live owner/generation token so matching backend accelerators can
// reject stale bindings after optimizer replacement, reallocation, or close.
type OptimizerResidentParameter struct {
	Backend  eosartifact.BackendKind
	Token    OptimizerResidentParameterToken
	Elements int
}

// OptimizerResidentParameterToken is implemented by backend-private resident
// parameter tokens. The runtime treats it as opaque liveness metadata; backend
// packages may type-assert to their private token implementation.
type OptimizerResidentParameterToken interface {
	OptimizerResidentParameterToken()
	Backend() eosartifact.BackendKind
	Generation() uint64
	Alive() bool
}

// ActivationAcceleratorStats summarizes backend-owned activation backward activity.
type ActivationAcceleratorStats struct {
	BindCalls              int64
	GELUBackwardCalls      int64
	SoftmaxBackwardCalls   int64
	LayerNormBackwardCalls int64
	UploadedBytes          int64
	DownloadedBytes        int64
	RunNanos               int64
	BoundTensors           int64
}

// ContrastiveLossConfig describes one backend-owned contrastive loss/gradient run.
type ContrastiveLossConfig struct {
	Temperature float32
}

// ContrastiveAcceleratorStats summarizes backend-owned contrastive loss activity.
type ContrastiveAcceleratorStats struct {
	RunCalls        int64
	UploadedBytes   int64
	DownloadedBytes int64
	RunNanos        int64
}

// CompactForwardAcceleratorStats summarizes backend-owned compact forward
// activity. Negative values are not used; unavailable stats are represented by
// a nil stats provider at the trainer/profile layer, distinct from a zeroed
// available provider.
type CompactForwardAcceleratorStats struct {
	RunCalls                    int64
	UnhandledCalls              int64
	UploadedBytes               int64
	DownloadedBytes             int64
	StatusDownloadedBytes       int64
	PackedDownloads             int64
	PackedBytes                 int64
	IntermediateD2H             int64
	IntermediateDownloadedBytes int64
	KernelLaunches              int64
	KernelSynchronizations      int64
	RunNanos                    int64
	LastPackedFloats            int
	LastPackedBytes             int64
	LastUploadBytes             int64
	LastDownloadBytes           int64
	LastStatusDownloadedBytes   int64
	LastKernelLaunches          int64
	LastKernelSynchronizations  int64
}

// CompactTrainAcceleratorStats summarizes backend-owned compact resident train
// activity. A nil stats provider means unavailable; a zero-valued available
// stats struct means the implementation exists but has not run.
type CompactTrainAcceleratorStats struct {
	ForwardCalls               int64
	BackwardCalls              int64
	HandlesCreated             int64
	HandlesReleased            int64
	LiveHandles                int64
	GradientZeroCalls          int64
	ResidentGradBytes          int64
	ActivationArenaBytes       int64
	WorkspaceArenaBytes        int64
	UploadedBytes              int64
	DownloadedBytes            int64
	PooledDownloadedBytes      int64
	GradPooledUploadedBytes    int64
	StatusDownloadedBytes      int64
	PackedBytesAvoided         int64
	HostGradUploadBytesAvoided int64
	KernelLaunches             int64
	KernelSynchronizations     int64
	GraphCaptures              int64
	GraphReplays               int64
	FallbackOrUnhandled        int64
	ForwardNanos               int64
	BackwardNanos              int64
	OptimizerResidentGradNanos int64
	LastShape                  CompactForwardShape
	LastForwardLaunches        int64
	LastBackwardLaunches       int64
	LastForwardSyncs           int64
	LastBackwardSyncs          int64
}

// ContrastiveGradResult contains pooled embedding gradients and unnormalized row metrics.
type ContrastiveGradResult struct {
	QueryGrads    *Tensor
	PositiveGrads *Tensor
	LossSum       float32
	ScoreSum      float32
}

const CompactForwardPackedStateVersion = 1
const CompactTrainStateVersion = 1

const (
	CompactForwardGELUExact = "exact"
	CompactForwardGELUFast  = "fast"
)

// CompactForwardShape describes one exact-length compact-forward bucket.
// Tokens, masks, and roles are supplied by the caller; packed state data is
// exclusively float32 tensors needed by host backward reconstruction.
type CompactForwardShape struct {
	Batch               int
	Tokens              int
	ModelDim            int
	FFNDim              int
	Heads               int
	HeadDim             int
	Layers              int
	OutputDim           int
	HasOutputProjection bool
}

// CompactForwardResidentRef carries backend-neutral liveness metadata for a
// resident compact-forward parameter. Accelerators may use these opaque owner
// and generation refs for preflight without leaking backend-specific types into
// the runtime ABI.
type CompactForwardResidentRef struct {
	Name     string
	Backend  eosartifact.BackendKind
	Token    CompactForwardResidentToken
	Elements int
}

// CompactForwardResidentToken is implemented by backend-private resident
// parameter tokens. The runtime only depends on liveness and generation checks;
// backend packages may type-assert to their private token implementation.
type CompactForwardResidentToken interface {
	CompactForwardResidentToken()
	Backend() eosartifact.BackendKind
	Generation() uint64
	Alive() bool
}

// CompactForwardStateSpan names one contiguous slice inside a packed compact
// forward result.
type CompactForwardStateSpan struct {
	Name   string
	Offset int
	Len    int
}

// CompactForwardPackedStateLayout is the versioned ABI for reconstructing host
// compact forward state views from one packed float32 backing.
type CompactForwardPackedStateLayout struct {
	Version int
	Shape   CompactForwardShape
	Spans   []CompactForwardStateSpan
}

// CompactForwardRequest asks an accelerator to run a compact bucket. The
// concrete backend owns how resident parameters are represented; this contract
// intentionally avoids CUDA-specific types.
type CompactForwardRequest struct {
	Shape        CompactForwardShape
	Tokens       [][]int32
	Masks        [][]int32
	Roles        []int32
	ResidentRefs []CompactForwardResidentRef
	// GELUMode selects the host-compatible FFN activation mode. Empty means
	// CompactForwardGELUExact for backward compatibility with existing callers.
	GELUMode string
}

// CompactForwardResult returns a packed state buffer and the exact layout used
// to describe it. Accelerators transfer ownership of Data to the caller and
// must not mutate, recycle, or pool that backing until the caller has finished
// with the result. Callers must treat Data as immutable and validate the layout
// against the request shape before slicing it. Host reconstruction defensively
// copies Data once so reconstructed states remain stable if a backend violates
// the no-reuse contract.
type CompactForwardResult struct {
	Layout CompactForwardPackedStateLayout
	Data   []float32
}

// CompactTrainForwardRequest asks a resident-train accelerator to run compact
// forward while keeping backward-required activations device-resident. The host
// ABI exposes only pooled outputs and liveness handles.
type CompactTrainForwardRequest struct {
	Shape        CompactForwardShape
	Tokens       [][]int32
	Masks        [][]int32
	Roles        []int32
	ResidentRefs []CompactForwardResidentRef
	GELUMode     string
	StepID       uint64
}

type CompactTrainHandle struct {
	Backend    eosartifact.BackendKind
	Token      CompactTrainHandleToken
	Shape      CompactForwardShape
	Generation uint64
	StepID     uint64
}

type CompactTrainHandleToken interface {
	CompactTrainHandleToken()
	Backend() eosartifact.BackendKind
	Generation() uint64
	StepID() uint64
	Alive() bool
}

type CompactTrainForwardResult struct {
	Handle       CompactTrainHandle
	Pooled       *Tensor
	ActiveCounts []int32
}

type CompactTrainBackwardRequest struct {
	Handle     CompactTrainHandle
	GradPooled *Tensor
}

type CompactTrainBackwardResult struct {
	ResidentGradRefs []ResidentGradientRef
}

type ResidentGradientRef struct {
	Name       string
	Backend    eosartifact.BackendKind
	Token      ResidentGradientToken
	Elements   int
	Generation uint64
	StepID     uint64
}

type ResidentGradientToken interface {
	ResidentGradientToken()
	Backend() eosartifact.BackendKind
	Generation() uint64
	Alive() bool
}

// CompactForwardLayerConfig names one compact transformer layer's resident
// parameters for accelerator configuration.
type CompactForwardLayerConfig struct {
	AttentionQ string
	AttentionK string
	AttentionV string
	AttentionO string
	FFNUp      string
	FFNDown    string
}

// MatMulAccelerator exposes a backend-owned matmul fast path for non-plan callers.
type MatMulAccelerator interface {
	Backend() eosartifact.BackendKind
	RunMatMul(inputs []*Tensor, outputType eosartifact.ValueType) (StepDispatchResult, error)
	RunMatMulWithTranspose(inputs []*Tensor, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (StepDispatchResult, error)
	BindMatrix(name string, tensor *Tensor) error
	UnbindMatrix(name string) error
	RunMatMulWithBoundLeft(leftName string, rhs *Tensor, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (StepDispatchResult, error)
	RunMatMulWithBoundRight(lhs *Tensor, rightName string, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (StepDispatchResult, error)
	Stats() MatMulAcceleratorStats
	Close()
}

// MultiBoundRightMatMulAccelerator optionally coalesces several resident-right matmuls
// that share the same left input. It preserves each bound right-hand tensor's own
// residency and quantization state while avoiding repeated LHS uploads.
type MultiBoundRightMatMulAccelerator interface {
	RunMatMulWithBoundRights(lhs *Tensor, rightNames []string, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) ([]StepDispatchResult, error)
}

// SharedLeftMatMulAccelerator optionally coalesces several matmuls that share
// the same left input but use different non-resident right inputs.
type SharedLeftMatMulAccelerator interface {
	RunMatMulsWithSharedLeft(lhs *Tensor, rhs []*Tensor, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) ([]StepDispatchResult, error)
}

// AccumulatedBoundRightMatMulAccelerator optionally coalesces several
// resident-right matmuls with distinct left inputs into one accumulated output.
type AccumulatedBoundRightMatMulAccelerator interface {
	RunAccumulatedMatMulsWithBoundRights(lhs []*Tensor, rightNames []string, outputType eosartifact.ValueType, transposeLeft, transposeRight bool) (StepDispatchResult, error)
}

// CompactForwardAccelerator optionally executes compact forward for an
// exact-length bucket and returns the versioned packed state ABI consumed by
// host reconstruction/backward.
type CompactForwardAccelerator interface {
	Backend() eosartifact.BackendKind
	RunCompactForward(req CompactForwardRequest) (CompactForwardResult, error)
}

// CompactTrainAccelerator optionally executes compact resident train. Forward
// may be implemented before backward, but selected training callers must fail
// closed when backward/resident-gradient functionality is unsupported.
type CompactTrainAccelerator interface {
	Backend() eosartifact.BackendKind
	BeginCompactTrainStep(stepID uint64, refs []CompactForwardResidentRef) error
	RunCompactTrainForward(req CompactTrainForwardRequest) (CompactTrainForwardResult, error)
	RunCompactTrainBackward(req CompactTrainBackwardRequest) (CompactTrainBackwardResult, error)
	EndCompactTrainStep(stepID uint64) error
	ReleaseCompactTrainHandle(handle CompactTrainHandle) error
}

// CompactForwardStatsProvider exposes compact-forward counters when the
// selected accelerator can report them.
type CompactForwardStatsProvider interface {
	Stats() CompactForwardAcceleratorStats
}

// CompactTrainStatsProvider exposes compact resident-train counters when the
// selected accelerator can report them.
type CompactTrainStatsProvider interface {
	CompactTrainStats() CompactTrainAcceleratorStats
}

// CompactForwardConfigurator configures model-specific compact-forward names.
type CompactForwardConfigurator interface {
	ConfigureCompactForward(layers []CompactForwardLayerConfig, tokenName, roleName, outputProjectionName string, useRoPE bool)
}

// CompactForwardResidentBinder binds compact-forward parameters from resident
// optimizer state.
type CompactForwardResidentBinder interface {
	BindCompactForwardResident(name string, tensor *Tensor, ref OptimizerResidentParameter) error
}

// CompactTrainResidentBinder binds compact resident-train parameters from
// resident optimizer state.
type CompactTrainResidentBinder interface {
	BindCompactTrainResident(name string, tensor *Tensor, ref OptimizerResidentParameter) error
}

// CompactForwardPreflight validates a request before it is selected for the
// fail-closed packed path.
type CompactForwardPreflight interface {
	PreflightCompactForward(req CompactForwardRequest) error
}

// CompactTrainPreflight validates a resident-train forward request before a
// fail-closed validation path selects it.
type CompactTrainPreflight interface {
	PreflightCompactTrainForward(req CompactTrainForwardRequest) error
}

// OptimizerAccelerator exposes a backend-owned optimizer update fast path.
type OptimizerAccelerator interface {
	Backend() eosartifact.BackendKind
	ApplyUpdate(name string, cfg OptimizerUpdateConfig, tensor, mom1, mom2, grad *Tensor) error
	SyncState(name string, tensor, mom1, mom2 *Tensor, includeMoments bool) error
	Stats() OptimizerAcceleratorStats
	Close()
}

type ResidentGradientOptimizerAccelerator interface {
	ApplyUpdateWithResidentGrad(name string, cfg OptimizerUpdateConfig, tensor, mom1, mom2 *Tensor, grad ResidentGradientRef) error
}

// ForcedOptimizerSyncAccelerator optionally records the semantic reason for a
// host-visible optimizer sync.
type ForcedOptimizerSyncAccelerator interface {
	SyncStateWithReason(name string, tensor, mom1, mom2 *Tensor, includeMoments bool, reason string) error
}

// ResidentOptimizerParameterProvider optionally exposes backend-owned parameter
// buffers for another accelerator from the same backend.
type ResidentOptimizerParameterProvider interface {
	ResidentParameter(name string) (OptimizerResidentParameter, bool)
}

// ResidentOptimizerParameterSeeder ensures a parameter is resident without
// applying an optimizer update.
type ResidentOptimizerParameterSeeder interface {
	EnsureResidentParameter(name string, tensor, mom1, mom2 *Tensor) error
}

// DeviceResidentMatrixBinder optionally binds a matrix directly from a
// backend-owned device buffer rather than from host tensor contents.
type DeviceResidentMatrixBinder interface {
	BindMatrixFromResident(name string, tensor *Tensor, ref OptimizerResidentParameter) error
}

// ActivationAccelerator exposes backend-owned elementwise training ops.
type ActivationAccelerator interface {
	Backend() eosartifact.BackendKind
	BindTensor(name string, tensor *Tensor) error
	UnbindTensor(name string) error
	RunGELUBackwardMul(gradOut, preAct *Tensor) (*Tensor, error)
	RunGELUBackwardMulWithBoundPreAct(gradOut *Tensor, preActName string) (*Tensor, error)
	RunSoftmaxBackwardRows(gradOut, probs *Tensor) (*Tensor, error)
	RunSoftmaxBackwardRowsWithBoundProbs(gradOut *Tensor, probsName string) (*Tensor, error)
	RunLayerNormBackwardRows(gradOut, normalized, pre *Tensor) (*Tensor, error)
	RunLayerNormBackwardRowsWithBoundInputs(gradOut *Tensor, normalizedName, preName string) (*Tensor, error)
	Stats() ActivationAcceleratorStats
	Close()
}

// ContrastiveAccelerator exposes backend-owned contrastive loss and pooled-gradient ops.
type ContrastiveAccelerator interface {
	Backend() eosartifact.BackendKind
	RunInfoNCE(query, positive *Tensor, cfg ContrastiveLossConfig) (ContrastiveGradResult, error)
	RunInfoNCEWithTargets(query, candidates *Tensor, targetIndexes []int, cfg ContrastiveLossConfig) (ContrastiveGradResult, error)
	Stats() ContrastiveAcceleratorStats
	Close()
}

var matMulAcceleratorFactories []matMulAcceleratorFactory
var optimizerAcceleratorFactories []optimizerAcceleratorFactory
var activationAcceleratorFactories []activationAcceleratorFactory
var contrastiveAcceleratorFactories []contrastiveAcceleratorFactory
var compactForwardAcceleratorFactories []compactForwardAcceleratorFactory
var compactTrainAcceleratorFactories []compactTrainAcceleratorFactory

type matMulAcceleratorFactory struct {
	kind    eosartifact.BackendKind
	factory func() (MatMulAccelerator, error)
}

type optimizerAcceleratorFactory struct {
	kind    eosartifact.BackendKind
	factory func() (OptimizerAccelerator, error)
}

type activationAcceleratorFactory struct {
	kind    eosartifact.BackendKind
	factory func() (ActivationAccelerator, error)
}

type contrastiveAcceleratorFactory struct {
	kind    eosartifact.BackendKind
	factory func() (ContrastiveAccelerator, error)
}

type compactForwardAcceleratorFactory struct {
	kind    eosartifact.BackendKind
	factory func() (CompactForwardAccelerator, error)
}

type compactTrainAcceleratorFactory struct {
	kind    eosartifact.BackendKind
	factory func() (CompactTrainAccelerator, error)
}

// KernelDispatcher executes a launch_kernel step through a backend-owned path.
type KernelDispatcher func(ctx context.Context, kernel eosartifact.Kernel, inputs []*Tensor) (KernelDispatchResult, error)

// StepDispatcher executes a backend-owned plan step such as library-backed matmul.
type StepDispatcher func(ctx context.Context, step eosartifact.Step, outputType eosartifact.ValueType, inputs []*Tensor) (StepDispatchResult, bool, error)

// Result is the execution response from a backend.
type Result struct {
	Outputs  map[string]Value
	Metadata map[string]string
	Trace    []TraceStep
}

// Executor runs a previously loaded Eos module.
type Executor interface {
	Backend() eosartifact.BackendKind
	Run(ctx context.Context, req Request) (Result, error)
}

// Backend loads Eos modules and returns executors.
type Backend interface {
	Kind() eosartifact.BackendKind
	CanLoad(mod *eosartifact.Module) bool
	Load(ctx context.Context, mod *eosartifact.Module, weights map[string]WeightBinding) (Executor, error)
}

// DeviceInjector is an optional capability a Backend may implement to adopt an
// externally-owned GPU device — one shared with a renderer (GoSX) or a
// render-coupled compute layer (Elio) — instead of creating its own. The handle
// is backend-specific (a WebGPU GPUDevice as a syscall/js value, a Metal
// device, …); pass nil to clear. This is the Eos side of the
// Eos↔Elio↔Selena device-sharing seam; it pairs with GoSX's
// jsgpu.Device.NativeDevice() / jsgpu.WrapDevice. It takes effect on the next
// uncached Load.
type DeviceInjector interface {
	SetExternalDevice(handle any)
}

// CapabilityProvider reports runtime features a backend can satisfy.
type CapabilityProvider interface {
	Capabilities() []string
}

// CompileVariants resolves backend-specific kernel variants at load time.
func CompileVariants(mod *eosartifact.Module, kind eosartifact.BackendKind) (map[string]CompiledKernel, error) {
	compiled := map[string]CompiledKernel{}
	if mod == nil {
		return compiled, nil
	}
	for _, kernel := range mod.Kernels {
		variant, ok := kernelVariantForBackend(kernel, kind)
		if !ok {
			return nil, fmt.Errorf("kernel %q missing variant for backend %q", kernel.Name, kind)
		}
		sum := sha256.Sum256([]byte(variant.Source))
		compiled[kernel.Name] = CompiledKernel{
			Name:       kernel.Name,
			Backend:    kind,
			Entry:      variant.Entry,
			Source:     variant.Source,
			SourceHash: hex.EncodeToString(sum[:]),
			Meta:       cloneStringMap(variant.Meta),
		}
	}
	return compiled, nil
}

func cloneStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

// RegisterMatMulAccelerator registers an optional backend-owned matmul fast path.
func RegisterMatMulAccelerator(kind eosartifact.BackendKind, factory func() (MatMulAccelerator, error)) {
	if factory == nil {
		return
	}
	matMulAcceleratorFactories = append(matMulAcceleratorFactories, matMulAcceleratorFactory{
		kind:    kind,
		factory: factory,
	})
}

// RegisterOptimizerAccelerator registers an optional backend-owned optimizer fast path.
func RegisterOptimizerAccelerator(kind eosartifact.BackendKind, factory func() (OptimizerAccelerator, error)) {
	if factory == nil {
		return
	}
	optimizerAcceleratorFactories = append(optimizerAcceleratorFactories, optimizerAcceleratorFactory{
		kind:    kind,
		factory: factory,
	})
}

// RegisterActivationAccelerator registers an optional backend-owned activation fast path.
func RegisterActivationAccelerator(kind eosartifact.BackendKind, factory func() (ActivationAccelerator, error)) {
	if factory == nil {
		return
	}
	activationAcceleratorFactories = append(activationAcceleratorFactories, activationAcceleratorFactory{
		kind:    kind,
		factory: factory,
	})
}

// RegisterContrastiveAccelerator registers an optional backend-owned contrastive fast path.
func RegisterContrastiveAccelerator(kind eosartifact.BackendKind, factory func() (ContrastiveAccelerator, error)) {
	if factory == nil {
		return
	}
	contrastiveAcceleratorFactories = append(contrastiveAcceleratorFactories, contrastiveAcceleratorFactory{
		kind:    kind,
		factory: factory,
	})
}

// RegisterCompactForwardAccelerator registers an optional backend-owned compact forward fast path.
func RegisterCompactForwardAccelerator(kind eosartifact.BackendKind, factory func() (CompactForwardAccelerator, error)) {
	if factory == nil {
		return
	}
	compactForwardAcceleratorFactories = append(compactForwardAcceleratorFactories, compactForwardAcceleratorFactory{
		kind:    kind,
		factory: factory,
	})
}

// RegisterCompactTrainAccelerator registers an optional backend-owned compact
// resident-train fast path.
func RegisterCompactTrainAccelerator(kind eosartifact.BackendKind, factory func() (CompactTrainAccelerator, error)) {
	if factory == nil {
		return
	}
	compactTrainAcceleratorFactories = append(compactTrainAcceleratorFactories, compactTrainAcceleratorFactory{
		kind:    kind,
		factory: factory,
	})
}

// NewPreferredMatMulAccelerator returns the first available registered accelerator.
func NewPreferredMatMulAccelerator(preferred ...eosartifact.BackendKind) (MatMulAccelerator, eosartifact.BackendKind, error) {
	for _, kind := range preferred {
		for _, candidate := range matMulAcceleratorFactories {
			if candidate.kind != kind {
				continue
			}
			accel, err := candidate.factory()
			if err != nil {
				continue
			}
			if accel != nil {
				return accel, kind, nil
			}
		}
	}
	return nil, "", nil
}

// NewPreferredContrastiveAccelerator returns the first available registered contrastive accelerator.
func NewPreferredContrastiveAccelerator(preferred ...eosartifact.BackendKind) (ContrastiveAccelerator, eosartifact.BackendKind, error) {
	for _, kind := range preferred {
		for _, candidate := range contrastiveAcceleratorFactories {
			if candidate.kind != kind {
				continue
			}
			accel, err := candidate.factory()
			if err != nil {
				continue
			}
			if accel != nil {
				return accel, kind, nil
			}
		}
	}
	return nil, "", nil
}

// NewPreferredOptimizerAccelerator returns the first available registered optimizer accelerator.
func NewPreferredOptimizerAccelerator(preferred ...eosartifact.BackendKind) (OptimizerAccelerator, eosartifact.BackendKind, error) {
	for _, kind := range preferred {
		for _, candidate := range optimizerAcceleratorFactories {
			if candidate.kind != kind {
				continue
			}
			accel, err := candidate.factory()
			if err != nil {
				continue
			}
			if accel != nil {
				return accel, kind, nil
			}
		}
	}
	return nil, "", nil
}

// NewPreferredActivationAccelerator returns the first available registered activation accelerator.
func NewPreferredActivationAccelerator(preferred ...eosartifact.BackendKind) (ActivationAccelerator, eosartifact.BackendKind, error) {
	for _, kind := range preferred {
		for _, candidate := range activationAcceleratorFactories {
			if candidate.kind != kind {
				continue
			}
			accel, err := candidate.factory()
			if err != nil {
				continue
			}
			if accel != nil {
				return accel, kind, nil
			}
		}
	}
	return nil, "", nil
}

// NewPreferredCompactForwardAccelerator returns the first available registered compact-forward accelerator.
func NewPreferredCompactForwardAccelerator(preferred ...eosartifact.BackendKind) (CompactForwardAccelerator, eosartifact.BackendKind, error) {
	var errs []error
	for _, kind := range preferred {
		for _, candidate := range compactForwardAcceleratorFactories {
			if candidate.kind != kind {
				continue
			}
			accel, err := candidate.factory()
			if err != nil {
				errs = append(errs, fmt.Errorf("%s: %w", kind, err))
				continue
			}
			if accel != nil {
				return accel, kind, nil
			}
		}
	}
	if len(errs) != 0 {
		return nil, "", fmt.Errorf("compact forward accelerator factory failed: %w", errors.Join(errs...))
	}
	return nil, "", nil
}

// NewPreferredCompactTrainAccelerator returns the first available registered
// compact resident-train accelerator.
func NewPreferredCompactTrainAccelerator(preferred ...eosartifact.BackendKind) (CompactTrainAccelerator, eosartifact.BackendKind, error) {
	var errs []error
	for _, kind := range preferred {
		for _, candidate := range compactTrainAcceleratorFactories {
			if candidate.kind != kind {
				continue
			}
			accel, err := candidate.factory()
			if err != nil {
				errs = append(errs, fmt.Errorf("%s: %w", kind, err))
				continue
			}
			if accel != nil {
				return accel, kind, nil
			}
		}
	}
	if len(errs) != 0 {
		return nil, "", fmt.Errorf("compact train accelerator factory failed: %w", errors.Join(errs...))
	}
	return nil, "", nil
}
