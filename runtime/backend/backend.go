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
	LogicalSteps                   int64
	TensorUpdateCalls              int64
	UpdateCalls                    int64
	ResidentGradUpdateCalls        int64
	DeferredSyncUpdates            int64
	SyncCalls                      int64
	ForcedSyncCalls                int64
	LastForcedSyncReason           string
	UploadedBytes                  int64
	ResidentGradUploadBytesAvoided int64
	DownloadedBytes                int64
	UploadedBytesPerStep           float64
	DownloadedBytesPerStep         float64
	UpdateNanos                    int64
	ResidentGradUpdateNanos        int64
	SyncNanos                      int64
	ResidentParams                 int64
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
	ForwardCalls                int64
	BackwardCalls               int64
	HandlesCreated              int64
	HandlesReleased             int64
	LiveHandles                 int64
	GradientZeroCalls           int64
	ResidentGradBytes           int64
	ActivationArenaBytes        int64
	WorkspaceArenaBytes         int64
	UploadedBytes               int64
	DownloadedBytes             int64
	PooledDownloadedBytes       int64
	GradPooledUploadedBytes     int64
	StatusDownloadedBytes       int64
	PackedBytesAvoided          int64
	HostGradUploadBytesAvoided  int64
	KernelLaunches              int64
	CublasGemmCalls             int64
	KernelSynchronizations      int64
	GraphCaptures               int64
	GraphReplays                int64
	FallbackOrUnhandled         int64
	ForwardNanos                int64
	BackwardNanos               int64
	OptimizerResidentGradNanos  int64
	LastShape                   CompactForwardShape
	LastForwardLaunches         int64
	LastBackwardLaunches        int64
	LastForwardCublasGemmCalls  int64
	LastBackwardCublasGemmCalls int64
	LastForwardSyncs            int64
	LastBackwardSyncs           int64
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

// AttentionResidentRequest asks an attention-resident accelerator to run one
// attention block for a uniform-length batch: Q/K/V projection, per-sequence
// scaled dot-product scores, forward softmax, and value-mix. Every step after
// the initial upload runs device-to-device; Query, Key, Value, Scores, and
// Mixed are each downloaded exactly once. QueryName/KeyName/ValueName must
// already be resident (see MatMulAccelerator.BindMatrix). Score scaling is
// expected to already be baked into the QueryName binding's weight data, so
// the accelerator never needs a separate device-side scale step.
type AttentionResidentRequest struct {
	Batch    int
	SeqLen   int
	ModelDim int
	// Input is [Batch*SeqLen, ModelDim], row-major, sequences concatenated in
	// batch order.
	Input     *Tensor
	QueryName string
	KeyName   string
	ValueName string
	// Mask optionally carries one length-SeqLen per-sequence key mask
	// (nonzero means "attend to this key position"), one entry per batch
	// sequence in the same order as Input. Nil or empty means unmasked
	// (AttentionMaskMode "none"): every softmax row normalizes over every
	// key, matching the accelerator's pre-S3(b) behavior exactly. A non-nil
	// Mask must have exactly Batch entries, each of length SeqLen.
	Mask [][]int32
}

// AttentionResidentResult returns every host-visible activation the stage
// (a) forward-only residency boundary requires: the caller's existing
// backward path still consumes Query/Key/Value/Scores/Mixed from the host.
type AttentionResidentResult struct {
	Query  *Tensor // [Batch*SeqLen, ModelDim]
	Key    *Tensor // [Batch*SeqLen, ModelDim]
	Value  *Tensor // [Batch*SeqLen, ModelDim]
	Scores *Tensor // [Batch*SeqLen, SeqLen], post-softmax
	Mixed  *Tensor // [Batch*SeqLen, ModelDim]
}

// AttentionResidentAccelerator optionally executes one attention block's
// Q/K/V projection, scaled dot-product scores, softmax, and value-mix
// without any intermediate host round trip: Q, K, V, scores, and the mixed
// output stay device-resident between the block's internal kernel calls, and
// only the initial input upload plus the final Q/K/V/scores/mixed downloads
// cross the host/device boundary. Backends that do not implement it leave
// callers on their existing per-call matmul path. Upload/download accounting
// flows through the same backend's MatMulAccelerator.Stats() counters rather
// than a separate stats surface.
type AttentionResidentAccelerator interface {
	Backend() eosartifact.BackendKind
	RunAttentionBlockResident(req AttentionResidentRequest) (AttentionResidentResult, error)
}

// AttentionResidentTrainHandle references one resident attention block's
// live Q/K/V/scores (and flattened input) device buffers, kept alive from
// RunAttentionBlockResidentTrainForward until the matching
// RunAttentionBlockResidentTrainBackward call (or an End/Abort/Release).
// It mirrors CompactTrainHandle's lifecycle pattern at attention-block
// granularity instead of whole-compact-forward granularity.
type AttentionResidentTrainHandle struct {
	Backend    eosartifact.BackendKind
	Token      AttentionResidentTrainHandleToken
	Batch      int
	SeqLen     int
	ModelDim   int
	Generation uint64
	StepID     uint64
}

// AttentionResidentTrainHandleToken is implemented by backend-private
// resident handle tokens. The runtime treats it as opaque liveness
// metadata; backend packages may type-assert to their private token
// implementation.
type AttentionResidentTrainHandleToken interface {
	AttentionResidentTrainHandleToken()
	Backend() eosartifact.BackendKind
	Generation() uint64
	StepID() uint64
	Alive() bool
}

// AttentionResidentTrainForwardRequest is AttentionResidentRequest plus the
// step this call belongs to and an optional per-sequence key mask.
type AttentionResidentTrainForwardRequest struct {
	Batch    int
	SeqLen   int
	ModelDim int
	// Input is [Batch*SeqLen, ModelDim], row-major, sequences concatenated in
	// batch order.
	Input     *Tensor
	QueryName string
	KeyName   string
	ValueName string
	// Mask optionally carries one length-SeqLen per-sequence key mask, one
	// entry per batch sequence in the same order as Input. Nil/empty means
	// unmasked, matching AttentionResidentRequest.Mask.
	Mask [][]int32
	// Scale is the attention score scale (for example 1/sqrt(model_dim));
	// unlike AttentionResidentRequest, Q/K here are the caller's unscaled
	// weight bindings, so the accelerator applies Scale itself inside the
	// scores/softmax and backward kernels (0 behaves as 1: an unscaled
	// caller need not set this field).
	Scale  float32
	StepID uint64
	// SkipUnreadDownload is S3(d)'s refcount-gated forward-download elision:
	// when true, the caller has proven this encoded sequence is referenced
	// exactly once for this training step AND backward is guaranteed to
	// reach it through the singular per-sequence resident path (see
	// EmbeddingTrainer.residentDownloadElisionSafe), so host code will never
	// read Query/Key/Value/Scores between this forward call and the
	// matching backward call. The accelerator must still compute and keep
	// every buffer device-resident -- backward's algebra is unchanged --
	// it only skips their D2H copy, leaving the corresponding
	// AttentionResidentTrainForwardResult fields nil. Mixed is unaffected:
	// the caller's forward continuation (the Wo projection) always reads
	// it, so it is always downloaded regardless of this flag.
	SkipUnreadDownload bool
}

// AttentionResidentTrainForwardResult returns the same host-visible
// activations AttentionResidentResult does, plus the handle
// RunAttentionBlockResidentTrainBackward needs to find this call's
// still-resident Q/K/V/scores/input buffers. Query/Key/Value/Scores are nil
// when the matching request set SkipUnreadDownload and the accelerator
// honored it; Mixed is always populated.
type AttentionResidentTrainForwardResult struct {
	Handle AttentionResidentTrainHandle
	Query  *Tensor // [Batch*SeqLen, ModelDim], nil when SkipUnreadDownload
	Key    *Tensor // [Batch*SeqLen, ModelDim], nil when SkipUnreadDownload
	Value  *Tensor // [Batch*SeqLen, ModelDim], nil when SkipUnreadDownload
	Scores *Tensor // [Batch*SeqLen, SeqLen], post-softmax, nil when SkipUnreadDownload
	Mixed  *Tensor // [Batch*SeqLen, ModelDim]
}

// AttentionResidentTrainBackwardRequest asks a resident-train accelerator to
// backpropagate through one attention block's Q/K/V projection, scores,
// softmax, and value-mix, given the upstream gradient at Mixed (computed by
// the caller's existing host backward through Wo and any attention residual/
// layer norm). GradMixed rows must be in the same batch order the matching
// forward call used.
type AttentionResidentTrainBackwardRequest struct {
	Handle    AttentionResidentTrainHandle
	GradMixed *Tensor // [Batch*SeqLen, ModelDim]
}

// AttentionResidentTrainBackwardResult returns the gradient flowing back
// into the block's input -- the Q/K/V-path contribution only; the caller
// still adds any attention-residual pass-through itself, matching how
// backpropAttentionSequences composes its own per-call matmul results.
// S3(d): the Wq/Wk/Wv weight-gradient contribution is no longer downloaded
// per call -- the accelerator accumulates it device-side across every
// backward call in the step instead (see
// FlushAttentionResidentTrainWeightGradients).
type AttentionResidentTrainBackwardResult struct {
	GradInput *Tensor // [Batch*SeqLen, ModelDim]
}

// AttentionResidentTrainAccelerator optionally executes one attention
// block's forward AND backward without any intermediate host round trip
// beyond the forward input/mask upload, the forward Q/K/V/scores/mixed
// download (unchanged from AttentionResidentAccelerator, minus whatever
// SkipUnreadDownload elides per S3(d)), the backward gradMixed upload, and
// the backward gradInput download. Q, K, V, scores, and the flattened input
// stay device-resident across the forward/backward boundary under the
// returned handle. Mirrors CompactTrainAccelerator's
// Begin/Forward/Backward/End/Abort/Release lifecycle at attention-block
// granularity: BeginAttentionResidentTrainStep must run before any
// forward/backward call for that step, and EndAttentionResidentTrainStep
// (success) or AbortAttentionResidentTrainStep (failure) must run after,
// releasing every handle still live for the step so a later step can never
// observe an earlier step's activations.
//
// S3(d) adds device-side Wq/Wk/Wv weight-gradient accumulation: every
// RunAttentionBlockResidentTrainBackward call in a step adds its
// contribution into a step-scoped device accumulator (instead of
// downloading its own copy), and FlushAttentionResidentTrainWeightGradients
// downloads the accumulated sum exactly once. Implementations that do not
// support accumulation may leave the accumulator empty and always return
// nil from Flush; callers must treat a nil Flush result as "nothing to add"
// rather than an error.
type AttentionResidentTrainAccelerator interface {
	Backend() eosartifact.BackendKind
	BeginAttentionResidentTrainStep(stepID uint64) error
	RunAttentionBlockResidentTrainForward(req AttentionResidentTrainForwardRequest) (AttentionResidentTrainForwardResult, error)
	RunAttentionBlockResidentTrainBackward(req AttentionResidentTrainBackwardRequest) (AttentionResidentTrainBackwardResult, error)
	// FlushAttentionResidentTrainWeightGradients downloads (once) and resets
	// to zero the step-scoped, device-accumulated sum of every
	// RunAttentionBlockResidentTrainBackward call's Wq/Wk/Wv weight-gradient
	// contribution so far this step. queryName/keyName/valueName identify
	// which resident weight bindings' accumulators to flush; a name with no
	// contributions this step returns a nil tensor for that slot and no
	// error. The caller merges the result additively into its own
	// accumulator (for example via addFloat32Slice) alongside every
	// non-resident path's contribution -- this call does not zero or
	// otherwise touch the caller's own accumulator.
	FlushAttentionResidentTrainWeightGradients(queryName, keyName, valueName string) (gradWq, gradWk, gradWv *Tensor, err error)
	EndAttentionResidentTrainStep(stepID uint64) error
	AbortAttentionResidentTrainStep(stepID uint64) error
	ReleaseAttentionResidentTrainHandle(handle AttentionResidentTrainHandle) error
}

// FFNResidentTrainHandle references one resident FFN block's live
// ffnHidden/activated (and flattened input) device buffers, kept alive from
// RunFFNBlockResidentTrainForward until the matching
// RunFFNBlockResidentTrainBackward call (or an explicit Release). It shares
// its step lifecycle with AttentionResidentTrainHandle: the same
// Begin/End/AbortAttentionResidentTrainStep calls that bound an attention
// block's handle lifetime also bound an FFN block's, so S3(c) needs no
// separate step-boundary API -- one flag, one step, both block kinds.
type FFNResidentTrainHandle struct {
	Backend    eosartifact.BackendKind
	Token      FFNResidentTrainHandleToken
	SeqLen     int
	InputDim   int
	HiddenDim  int
	OutputDim  int
	Generation uint64
	StepID     uint64
}

// FFNResidentTrainHandleToken is implemented by backend-private FFN
// resident handle tokens, mirroring AttentionResidentTrainHandleToken.
type FFNResidentTrainHandleToken interface {
	FFNResidentTrainHandleToken()
	Backend() eosartifact.BackendKind
	Generation() uint64
	StepID() uint64
	Alive() bool
}

// FFNResidentTrainForwardRequest asks a resident-train accelerator to run
// one FFN block -- hidden projection, GELU, output projection -- entirely
// device-to-device: input is [SeqLen, InputDim] (the caller's already
// attention-residual/layernorm'd state.hidden), HiddenWeightName/
// OutputWeightName must already be resident (see
// MatMulAccelerator.BindMatrix), and FastGELU selects the same fast/exact
// GELU variant the host path uses (fillGELUForward/fillGELUBackwardMul) so
// forward and backward stay bit-consistent with the flag-OFF reference.
type FFNResidentTrainForwardRequest struct {
	SeqLen           int
	InputDim         int
	HiddenDim        int
	OutputDim        int
	Input            *Tensor // [SeqLen, InputDim]
	HiddenWeightName string
	OutputWeightName string
	FastGELU         bool
	StepID           uint64
	// SkipUnreadDownload mirrors
	// AttentionResidentTrainForwardRequest.SkipUnreadDownload: true when the
	// caller has proven this encoded sequence is referenced exactly once
	// this step AND backward is guaranteed to reach it through the singular
	// per-sequence resident path, so host code will never read
	// FFNHidden/Activated between this forward call and the matching
	// backward call. The accelerator still computes and keeps every buffer
	// device-resident; it only skips their D2H copy, leaving the
	// corresponding FFNResidentTrainForwardResult fields nil. FFNOutput is
	// unaffected: the caller's forward continuation (residual + layer norm)
	// always reads it, so it is always downloaded regardless of this flag.
	SkipUnreadDownload bool
}

// FFNResidentTrainForwardResult returns the FFN block's host-visible
// activations -- the caller's existing host path still adds the FFN
// residual and optional layer norm on top of FFNOutput -- plus the handle
// RunFFNBlockResidentTrainBackward needs to find this call's still-resident
// ffnHidden/activated/input buffers. FFNHidden/Activated are nil when the
// matching request set SkipUnreadDownload and the accelerator honored it;
// FFNOutput is always populated.
type FFNResidentTrainForwardResult struct {
	Handle    FFNResidentTrainHandle
	FFNHidden *Tensor // [SeqLen, HiddenDim], pre-GELU, nil when SkipUnreadDownload
	Activated *Tensor // [SeqLen, HiddenDim], post-GELU, nil when SkipUnreadDownload
	FFNOutput *Tensor // [SeqLen, OutputDim]
}

// FFNResidentTrainBackwardRequest asks a resident-train accelerator to
// backpropagate through one FFN block given the upstream gradient at
// FFNOutput (computed by the caller's existing host backward through the
// FFN residual and optional layer norm, exactly as
// backpropProjectedFFNSequence does today before it reaches the output
// projection). GradFFNOutput rows must be in the same order the matching
// forward call used.
type FFNResidentTrainBackwardRequest struct {
	Handle        FFNResidentTrainHandle
	GradFFNOutput *Tensor // [SeqLen, OutputDim]
}

// FFNResidentTrainBackwardResult returns the gradient flowing back into the
// block's input (state.hidden) -- the hidden-projection/GELU/output-
// projection contribution only; the caller still adds any FFN-residual
// pass-through itself, matching how backpropProjectedFFNSequence composes
// its own result. GradInput is named to match
// AttentionResidentTrainBackwardResult.GradInput -- both mean "gradient
// flowing into this block's input" -- and deliberately not "GradHidden",
// which in the trainer package already names the Wup (hidden-projection)
// weight-gradient accumulator, not an activation gradient. S3(d): the
// hidden/output weight-gradient contribution is no longer downloaded per
// call -- see FlushFFNResidentTrainWeightGradients.
type FFNResidentTrainBackwardResult struct {
	GradInput *Tensor // [SeqLen, InputDim]
}

// FFNResidentTrainAccelerator optionally executes one FFN block's forward
// AND backward without any intermediate host round trip beyond the forward
// input upload, the forward ffnHidden/activated/ffnOutput download (minus
// whatever SkipUnreadDownload elides per S3(d)), the backward gradFFNOutput
// upload, and the backward gradInput download. It shares
// BeginAttentionResidentTrainStep/EndAttentionResidentTrainStep/
// AbortAttentionResidentTrainStep with AttentionResidentTrainAccelerator: a
// backend implementing both opens and closes exactly one step per training
// step, and that single step's End/Abort sweeps every live handle of either
// kind.
//
// S3(d) adds device-side hidden/output weight-gradient accumulation,
// mirroring AttentionResidentTrainAccelerator's Wq/Wk/Wv accumulation: every
// RunFFNBlockResidentTrainBackward call in a step adds its contribution
// into a step-scoped device accumulator, and
// FlushFFNResidentTrainWeightGradients downloads the accumulated sum
// exactly once.
type FFNResidentTrainAccelerator interface {
	Backend() eosartifact.BackendKind
	RunFFNBlockResidentTrainForward(req FFNResidentTrainForwardRequest) (FFNResidentTrainForwardResult, error)
	RunFFNBlockResidentTrainBackward(req FFNResidentTrainBackwardRequest) (FFNResidentTrainBackwardResult, error)
	// FlushFFNResidentTrainWeightGradients downloads (once) and resets to
	// zero the step-scoped, device-accumulated sum of every
	// RunFFNBlockResidentTrainBackward call's hidden/output weight-gradient
	// contribution so far this step. hiddenWeightName/outputWeightName
	// identify which resident weight bindings' accumulators to flush; a
	// name with no contributions this step returns a nil tensor for that
	// slot and no error. The caller merges the result additively into its
	// own accumulator, mirroring
	// AttentionResidentTrainAccelerator.FlushAttentionResidentTrainWeightGradients.
	FlushFFNResidentTrainWeightGradients(hiddenWeightName, outputWeightName string) (gradHiddenWeight, gradOutputWeight *Tensor, err error)
	ReleaseFFNResidentTrainHandle(handle FFNResidentTrainHandle) error
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
	AbortCompactTrainStep(stepID uint64) error
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

// OptimizerPreflightAccelerator validates a host-gradient optimizer update
// without launching work, uploading gradients, allocating resident buffers, or
// mutating optimizer state/statistics. Implementations must validate the same
// predictable tensor/config/resident-binding errors as ApplyUpdate.
type OptimizerPreflightAccelerator interface {
	PreflightApplyUpdate(name string, cfg OptimizerUpdateConfig, tensor, mom1, mom2, grad *Tensor) error
}

type ResidentGradientOptimizerAccelerator interface {
	ApplyUpdateWithResidentGrad(name string, cfg OptimizerUpdateConfig, tensor, mom1, mom2 *Tensor, grad ResidentGradientRef) error
}

// ResidentGradientOptimizerPreflightAccelerator validates a resident-gradient
// update without launching work, consuming the gradient, or mutating optimizer
// state or statistics. Implementations must use the same validation as Apply.
type ResidentGradientOptimizerPreflightAccelerator interface {
	PreflightApplyUpdateWithResidentGrad(name string, cfg OptimizerUpdateConfig, tensor, mom1, mom2 *Tensor, grad ResidentGradientRef) error
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
