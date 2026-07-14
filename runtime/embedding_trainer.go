package eosruntime

import (
	"context"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
	"m31labs.dev/turboquant"
)

const (
	ScoreSpectrumLossModeHardSoft         = "hard_soft"
	ScoreSpectrumLossModeRecovery         = "recovery"
	ScoreSpectrumLossModeHardSoftRecovery = "hard_soft_recovery"
)

// EmbeddingPairExample is one supervised pairwise training example.
type EmbeddingPairExample struct {
	Source      string
	LeftTokens  []int32
	RightTokens []int32
	LeftMask    []int32
	RightMask   []int32
	Target      float32
}

// EmbeddingTrainConfig controls the narrow pooled-embedder trainer.
type EmbeddingTrainConfig struct {
	LearningRate                   float32
	WeightDecay                    float32
	WeightBits                     int
	Optimizer                      string
	Beta1                          float32
	Beta2                          float32
	Epsilon                        float32
	ContrastiveLoss                string
	Temperature                    float32
	GroupedLossWeight              float32
	TeacherLossWeight              float32
	TeacherTemperature             float32
	TeacherSourceTemperatures      map[string]float32
	TeacherSourceWeights           map[string]float32
	MatryoshkaDims                 []int
	MatryoshkaWeights              []float32
	TurboQuantPrefixBits           []int
	TurboQuantPrefixObjectives     []TurboQuantPrefixObjective
	TurboQuantPrefixWeight         float32
	TurboQuantPrefixSeed           int64
	TurboQuantPrefixScoreMode      string
	TurboQuantCompactObjectives    []TurboQuantPrefixObjective
	TurboQuantRankMarginObjectives []TurboQuantPrefixObjective
	TurboQuantRankMargin           float32
	ScoreSpectrumLossMode          string
	ScoreSpectrumRecoveryWeight    float32
	ScoreSpectrumRecoveryMargin    float32
	ScoreSpectrumRecoveryTopK      int
	ScoreSpectrumRecoveryTau       float32
}

// TurboQuantPrefixObjective targets one quantized compact-prefix loss.
type TurboQuantPrefixObjective struct {
	Dim      int     `json:"dim"`
	BitWidth int     `json:"bit_width"`
	Weight   float32 `json:"weight"`
}

// EmbeddingTrainMetrics summarizes one training/eval batch.
type EmbeddingTrainMetrics struct {
	Loss         float32
	AverageScore float32
	BatchSize    int
	Movement     *EmbeddingTrainMovementMetrics
}

// EmbeddingTrainStatMetrics summarizes an aggregate vector diagnostic.
type EmbeddingTrainStatMetrics struct {
	L2Norm       float32
	MaxAbs       float32
	NonzeroCount int
	TotalCount   int
}

// EmbeddingTrainMovementMetrics summarizes optimizer-step movement diagnostics.
type EmbeddingTrainMovementMetrics struct {
	Gradient       EmbeddingTrainStatMetrics
	ParameterDelta EmbeddingTrainStatMetrics
}

// EmbeddingEvalMetrics summarizes pairwise eval quality plus optional
// BEIR-style retrieval metrics computed from a held-out corpus+qrels.
type EmbeddingEvalMetrics struct {
	Loss               float32
	AverageScore       float32
	PositiveMeanScore  float32
	NegativeMeanScore  float32
	PairAccuracy       float32
	ThresholdAccuracy  float32
	ScoreThreshold     float32
	ROCAUC             float32
	ScoreMargin        float32
	Top1Accuracy       float32
	Top5Accuracy       float32
	Top10Accuracy      float32
	MeanReciprocalRank float32
	MeanPositiveRank   float32
	// RetrievalNDCGAt10, RetrievalMAPAt100, and RetrievalRecallAt100 are
	// BEIR-style retrieval metrics over a held-out corpus+qrels, computed during
	// training when a retrieval eval set is configured. They are the deployment
	// metrics (unlike the pairwise metrics above, which saturate), so they can
	// drive -select-metric retrieval_ndcg, retrieval_map, and
	// retrieval_recall. Zero when no retrieval eval set is configured.
	RetrievalNDCGAt10    float32
	RetrievalMAPAt100    float32
	RetrievalRecallAt100 float32
	RetrievalEval        *RetrievalEvalMetrics `json:"retrieval_eval,omitempty"`
	PairCount            int
	PositiveCount        int
	NegativeCount        int
}

// EmbeddingScoreSpectrumEvalMetrics summarizes read-only row-local
// score-spectrum quality over tokenized examples.
type EmbeddingScoreSpectrumEvalMetrics struct {
	Loss                              float32
	AverageScore                      float32
	AnyPositiveTop1                   float32
	OriginalPositiveTop1              float32
	AlternateRelevantRecovery         float32
	BestPositiveHardestNegativeMargin float32
	TargetCrossEntropy                float32
	TargetKL                          float32
	RowCount                          int
	CandidateCount                    int
	AnyPositiveRowCount               int
	OriginalPositiveRowCount          int
	AlternateRecoveryRowCount         int
	MarginRowCount                    int
	TargetDistributionRowCount        int
}

// EmbeddingListwiseGeometryEvalMetrics summarizes read-only listwise teacher
// geometry quality over tokenized query/document matrix batches.
type EmbeddingListwiseGeometryEvalMetrics struct {
	Loss                  float32
	AverageScore          float32
	TeacherCrossEntropy   float32
	TeacherKL             float32
	TeacherTop1Agreement  float32
	AnyPositiveTop1       float32
	QueryCount            int
	DocumentCellCount     int
	BatchCount            int
	AnyPositiveQueryCount int
}

// EmbeddingForwardResidencyStats summarizes trainer-level bind suppression plus backend prep activity.
type EmbeddingForwardResidencyStats struct {
	BindSkips int64
	MatMul    backend.MatMulAcceleratorStats
}

type embeddingEvalScore struct {
	Score    float32
	Positive bool
}

type embeddingEvalRankScore struct {
	QueryKey string
	Score    float32
	Positive bool
}

// EmbeddingTrainer trains a pooled embedding model with quantization-aware forward passes.
type EmbeddingTrainer struct {
	module                  *eosartifact.Module
	manifest                EmbeddingManifest
	config                  EmbeddingTrainConfig
	memoryPlan              *MemoryPlan
	scoreSpectrumLineage    EmbeddingScoreSpectrumPolicy
	listwiseGeometryLineage EmbeddingListwiseGeometryPolicy
	// Retrieval-nDCG eval gate (optional). When enabled, each pairwise/contrastive
	// eval also embeds a held-out BEIR-style corpus with the current in-training
	// weights and computes nDCG@10 — the deployment metric — so training can
	// select/restore-best on it instead of the saturated pairwise gate.
	retrievalEvalRuntime    *Runtime
	retrievalEvalConfig     RetrievalEvalConfig
	retrievalEvalTokenizer  *TokenizerFile
	retrievalEvalEnabled    bool
	step                    int
	tokenParam              eosartifact.Param
	roleParam               eosartifact.Param
	attnQParam              eosartifact.Param
	attnKParam              eosartifact.Param
	attnVParam              eosartifact.Param
	attnOParam              eosartifact.Param
	hiddenParam             eosartifact.Param
	projParam               eosartifact.Param
	tokenEmbed              *backend.Tensor
	roleEmbed               *backend.Tensor
	attentionQuery          *backend.Tensor
	attentionKey            *backend.Tensor
	attentionValue          *backend.Tensor
	attentionOutput         *backend.Tensor
	hiddenProjection        *backend.Tensor
	projection              *backend.Tensor
	tokenMom1               *backend.Tensor
	tokenMom2               *backend.Tensor
	roleMom1                *backend.Tensor
	roleMom2                *backend.Tensor
	attnQMom1               *backend.Tensor
	attnQMom2               *backend.Tensor
	attnKMom1               *backend.Tensor
	attnKMom2               *backend.Tensor
	attnVMom1               *backend.Tensor
	attnVMom2               *backend.Tensor
	attnOMom1               *backend.Tensor
	attnOMom2               *backend.Tensor
	hiddenMom1              *backend.Tensor
	hiddenMom2              *backend.Tensor
	projMom1                *backend.Tensor
	projMom2                *backend.Tensor
	forwardMatMul           backend.MatMulAccelerator
	forwardBackend          eosartifact.BackendKind
	optimizerAccel          backend.OptimizerAccelerator
	optimizerBackend        eosartifact.BackendKind
	activationAccel         backend.ActivationAccelerator
	activationBackend       eosartifact.BackendKind
	activationAccelFull     bool
	softmaxBackwardAccel    bool
	contrastiveAccel        backend.ContrastiveAccelerator
	contrastiveBackend      eosartifact.BackendKind
	sequenceBindingID       int
	momentsDirty            bool
	compactOptimizerUpdates int64
	forwardCache            *embeddingForwardWeights
	compactState            *CompactEmbeddingTrainState
	compactForwardCache     *compactEmbeddingForwardWeights
	boundForward            embeddingForwardWeights
	forwardDirty            bool
	forwardNeedsBind        bool
	forwardBindSkips        int64
	vectorDistillPhases     EmbeddingVectorDistillPhaseTimers
	scratchF32              [][]float32
	// vectorDistillDefaultRoleWarned tracks whether FitVectorDistill has already
	// logged the one-time warning about rows falling back to the default role.
	vectorDistillDefaultRoleWarned bool
}

// SetScoreSpectrumLineage records score-spectrum provenance that must follow
// future training and inference package writes.
func (t *EmbeddingTrainer) SetScoreSpectrumLineage(policy EmbeddingScoreSpectrumPolicy) {
	if t == nil {
		return
	}
	policy.SourceArtifactHashes = normalizeScoreSpectrumSourceHashes(policy.SourceArtifactHashes)
	policy.AutoClearedObjectives = normalizeScoreSpectrumObjectiveNames(policy.AutoClearedObjectives)
	policy.IsolatedInheritedObjectives = normalizeScoreSpectrumObjectiveNames(policy.IsolatedInheritedObjectives)
	t.scoreSpectrumLineage = policy
}

// SetListwiseGeometryLineage records listwise geometry provenance that must
// follow future training and inference package writes.
func (t *EmbeddingTrainer) SetListwiseGeometryLineage(policy EmbeddingListwiseGeometryPolicy) {
	if t == nil {
		return
	}
	policy.SourceArtifactHashes = normalizeScoreSpectrumSourceHashes(policy.SourceArtifactHashes)
	policy.AutoClearedObjectives = normalizeScoreSpectrumObjectiveNames(policy.AutoClearedObjectives)
	policy.IsolatedInheritedObjectives = normalizeScoreSpectrumObjectiveNames(policy.IsolatedInheritedObjectives)
	t.listwiseGeometryLineage = policy
}

type embeddingSequenceState struct {
	tokens              []int32
	mask                []int32
	activeCount         int
	input               []float32
	inputBinding        string
	hidden              []float32
	hiddenBinding       string
	attnQ               []float32
	attnK               []float32
	attnV               []float32
	bindingPrefix       string
	attnQBinding        string
	attnKBinding        string
	attnVBinding        string
	attnScores          []float32
	attnScoresBinding   string
	attnMixed           []float32
	attnMixedBinding    string
	attnOutput          []float32
	attnResidual        []float32
	attnResidualBinding string
	ffnHidden           []float32
	ffnHiddenBinding    string
	activated           []float32
	activatedBinding    string
	ffnOutput           []float32
	ffnResidual         []float32
	ffnResidualBinding  string
	projected           []float32
	projectedBinding    string
	normalized          []float32
	pooled              []float32
	skipUnboundActAccel bool
}

type embeddingEncodedSequence struct {
	layers []*embeddingSequenceState
	pooled []float32
	tokens []int32
	role   int32
}

func (s *embeddingEncodedSequence) finalLayer() *embeddingSequenceState {
	if s == nil || len(s.layers) == 0 {
		return nil
	}
	return s.layers[len(s.layers)-1]
}

type embeddingForwardWeights struct {
	token   *backend.Tensor
	role    *backend.Tensor
	attnQ   *backend.Tensor
	attnK   *backend.Tensor
	attnV   *backend.Tensor
	attnO   *backend.Tensor
	hidden  *backend.Tensor
	proj    *backend.Tensor
	compact *compactEmbeddingForwardWeights
}

type compactEmbeddingForwardLayer struct {
	attnQName      string
	attnKName      string
	attnVName      string
	attnOName      string
	ffnUpName      string
	ffnDownName    string
	attnQ          *backend.Tensor
	attnK          *backend.Tensor
	attnV          *backend.Tensor
	attnO          *backend.Tensor
	ffnUp          *backend.Tensor
	ffnDown        *backend.Tensor
	attentionHeads int
	headDim        int
}

type compactEmbeddingForwardWeights struct {
	token                *backend.Tensor
	role                 *backend.Tensor
	layers               []compactEmbeddingForwardLayer
	outputProjectionName string
	outputProjection     *backend.Tensor
}

type compactEmbeddingGradLayer struct {
	attnQ   []float32
	attnK   []float32
	attnV   []float32
	attnO   []float32
	ffnUp   []float32
	ffnDown []float32
}

type compactEmbeddingGradState struct {
	token            []float32
	role             []float32
	layers           []compactEmbeddingGradLayer
	outputProjection []float32
}

type embeddingTrainerParams struct {
	token         eosartifact.Param
	role          eosartifact.Param
	attnQ         eosartifact.Param
	attnK         eosartifact.Param
	attnV         eosartifact.Param
	attnO         eosartifact.Param
	hidden        eosartifact.Param
	proj          eosartifact.Param
	attnEnabled   bool
	hiddenEnabled bool
}

type embeddingTrainerWeights struct {
	token            *backend.Tensor
	role             *backend.Tensor
	attentionQuery   *backend.Tensor
	attentionKey     *backend.Tensor
	attentionValue   *backend.Tensor
	attentionOutput  *backend.Tensor
	hiddenProjection *backend.Tensor
	projection       *backend.Tensor
}

// NewEmbeddingTrainer constructs the first native pooled-embedder trainer.
func NewEmbeddingTrainer(mod *eosartifact.Module, manifest EmbeddingManifest, weights map[string]*backend.Tensor, cfg EmbeddingTrainConfig) (*EmbeddingTrainer, error) {
	manifest = manifest.normalized()
	if err := manifest.ValidateModule(mod); err != nil {
		return nil, err
	}
	if err := manifest.ValidateLegacyEmbeddingTrainerSupported(); err != nil {
		return nil, err
	}
	params, err := resolveEmbeddingTrainerParams(mod, manifest)
	if err != nil {
		return nil, err
	}
	tensors, err := resolveEmbeddingTrainerWeights(params, manifest, weights)
	if err != nil {
		return nil, err
	}
	cfg = normalizedTrainConfig(cfg, params.token, params.attnQ, params.attnK, params.attnV, params.attnO, params.hidden, params.proj)
	cfg, err = normalizeMatryoshkaTrainConfig(cfg, tensors.projection.Shape[1])
	if err != nil {
		return nil, err
	}
	if err := validateTrainConfig(cfg); err != nil {
		return nil, err
	}
	accel, accelBackend, err := newTrainerMatMulAccelerator()
	if err != nil {
		return nil, err
	}
	optimizerAccel, optimizerBackend, err := newTrainerOptimizerAccelerator()
	if err != nil {
		if accel != nil {
			accel.Close()
		}
		return nil, err
	}
	activationAccel, activationBackend, activationMode, err := newTrainerActivationAccelerator()
	if err != nil {
		if accel != nil {
			accel.Close()
		}
		if optimizerAccel != nil {
			optimizerAccel.Close()
		}
		return nil, err
	}
	contrastiveAccel, contrastiveBackend, err := newTrainerContrastiveAccelerator()
	if err != nil {
		if accel != nil {
			accel.Close()
		}
		if optimizerAccel != nil {
			optimizerAccel.Close()
		}
		if activationAccel != nil {
			activationAccel.Close()
		}
		return nil, err
	}
	return &EmbeddingTrainer{
		module:               mod,
		manifest:             manifest,
		config:               cfg,
		memoryPlan:           nil,
		tokenParam:           params.token,
		roleParam:            params.role,
		attnQParam:           params.attnQ,
		attnKParam:           params.attnK,
		attnVParam:           params.attnV,
		attnOParam:           params.attnO,
		hiddenParam:          params.hidden,
		projParam:            params.proj,
		tokenEmbed:           tensorAsMasterF32(tensors.token),
		roleEmbed:            tensorAsMasterF32(tensors.role),
		attentionQuery:       tensorAsMasterF32(tensors.attentionQuery),
		attentionKey:         tensorAsMasterF32(tensors.attentionKey),
		attentionValue:       tensorAsMasterF32(tensors.attentionValue),
		attentionOutput:      tensorAsMasterF32(tensors.attentionOutput),
		hiddenProjection:     tensorAsMasterF32(tensors.hiddenProjection),
		projection:           tensorAsMasterF32(tensors.projection),
		tokenMom1:            zeroLikeMaster(tensors.token),
		tokenMom2:            zeroLikeMaster(tensors.token),
		roleMom1:             zeroLikeMaster(tensors.role),
		roleMom2:             zeroLikeMaster(tensors.role),
		attnQMom1:            zeroLikeMaster(tensors.attentionQuery),
		attnQMom2:            zeroLikeMaster(tensors.attentionQuery),
		attnKMom1:            zeroLikeMaster(tensors.attentionKey),
		attnKMom2:            zeroLikeMaster(tensors.attentionKey),
		attnVMom1:            zeroLikeMaster(tensors.attentionValue),
		attnVMom2:            zeroLikeMaster(tensors.attentionValue),
		attnOMom1:            zeroLikeMaster(tensors.attentionOutput),
		attnOMom2:            zeroLikeMaster(tensors.attentionOutput),
		hiddenMom1:           zeroLikeMaster(tensors.hiddenProjection),
		hiddenMom2:           zeroLikeMaster(tensors.hiddenProjection),
		projMom1:             zeroLikeMaster(tensors.projection),
		projMom2:             zeroLikeMaster(tensors.projection),
		forwardMatMul:        accel,
		forwardBackend:       accelBackend,
		optimizerAccel:       optimizerAccel,
		optimizerBackend:     optimizerBackend,
		activationAccel:      activationAccel,
		activationBackend:    activationBackend,
		activationAccelFull:  activationMode.fullBackward,
		softmaxBackwardAccel: activationMode.softmaxBackward,
		contrastiveAccel:     contrastiveAccel,
		contrastiveBackend:   contrastiveBackend,
	}, nil
}

func newCompactEmbeddingTrainerFromTrainState(mod *eosartifact.Module, state *CompactEmbeddingTrainState) (*EmbeddingTrainer, error) {
	if mod == nil {
		return nil, fmt.Errorf("nil module")
	}
	if state == nil {
		return nil, fmt.Errorf("compact train state is nil")
	}
	manifest := state.Manifest.normalizedForModule(mod)
	if manifest.ArchitectureVersion != EmbeddingArchitectureCompactTransformerV1 {
		return nil, fmt.Errorf("compact trainer requires architecture_version=%q", EmbeddingArchitectureCompactTransformerV1)
	}
	cfg := state.Config
	cfg = normalizedTrainConfig(cfg)
	accel, accelBackend, err := newTrainerMatMulAccelerator()
	if err != nil {
		return nil, err
	}
	optimizerAccel, optimizerBackend, err := newTrainerOptimizerAccelerator()
	if err != nil {
		if accel != nil {
			accel.Close()
		}
		return nil, err
	}
	activationAccel, activationBackend, activationMode, err := newTrainerActivationAccelerator()
	if err != nil {
		if accel != nil {
			accel.Close()
		}
		if optimizerAccel != nil {
			optimizerAccel.Close()
		}
		return nil, err
	}
	if accelBackend == "" {
		accelBackend = eosartifact.BackendKind("host")
	}
	if optimizerBackend == "" {
		optimizerBackend = eosartifact.BackendKind("host")
	}
	if activationBackend == "" {
		activationBackend = eosartifact.BackendKind("host")
	}
	return &EmbeddingTrainer{
		module:               mod,
		manifest:             manifest,
		config:               cfg,
		step:                 state.Step,
		compactState:         state,
		forwardMatMul:        accel,
		forwardBackend:       accelBackend,
		optimizerAccel:       optimizerAccel,
		optimizerBackend:     optimizerBackend,
		activationAccel:      activationAccel,
		activationBackend:    activationBackend,
		activationAccelFull:  activationMode.fullBackward,
		softmaxBackwardAccel: activationMode.softmaxBackward,
		forwardDirty:         true,
		forwardNeedsBind:     true,
	}, nil
}

func (t *EmbeddingTrainer) isCompactTrainer() bool {
	return t != nil && t.compactState != nil
}

func compactTrainingUnsupportedError() error {
	return fmt.Errorf("%s training updates are not supported yet", EmbeddingArchitectureCompactTransformerV1)
}

func resolveEmbeddingTrainerParams(mod *eosartifact.Module, manifest EmbeddingManifest) (embeddingTrainerParams, error) {
	tokenParam, err := requireTrainableEmbeddingParam(mod, manifest.TokenEmbeddingParam)
	if err != nil {
		return embeddingTrainerParams{}, err
	}
	roleParam, err := optionalRoleEmbeddingParam(mod, manifest)
	if err != nil {
		return embeddingTrainerParams{}, err
	}
	attnQParam, attnEnabled, err := optionalTrainableEmbeddingParam(mod, manifest.AttentionQueryParam)
	if err != nil {
		return embeddingTrainerParams{}, err
	}
	attnKParam, _, err := optionalTrainableEmbeddingParam(mod, manifest.AttentionKeyParam)
	if err != nil {
		return embeddingTrainerParams{}, err
	}
	attnVParam, _, err := optionalTrainableEmbeddingParam(mod, manifest.AttentionValueParam)
	if err != nil {
		return embeddingTrainerParams{}, err
	}
	attnOParam, _, err := optionalTrainableEmbeddingParam(mod, manifest.AttentionOutputParam)
	if err != nil {
		return embeddingTrainerParams{}, err
	}
	hiddenParam, hiddenEnabled, err := optionalTrainableEmbeddingParam(mod, manifest.HiddenProjectionParam)
	if err != nil {
		return embeddingTrainerParams{}, err
	}
	if (manifest.AttentionResidual || manifest.AttentionLayerNorm) && !attnEnabled {
		return embeddingTrainerParams{}, fmt.Errorf("attention residual/layernorm requires attention weights")
	}
	if (manifest.FFNResidual || manifest.FFNLayerNorm) && !hiddenEnabled {
		return embeddingTrainerParams{}, fmt.Errorf("ffn residual/layernorm requires hidden projection weights")
	}
	projParam, err := requireTrainableEmbeddingParam(mod, manifest.ProjectionParam)
	if err != nil {
		return embeddingTrainerParams{}, err
	}
	return embeddingTrainerParams{
		token:         tokenParam,
		role:          roleParam,
		attnQ:         attnQParam,
		attnK:         attnKParam,
		attnV:         attnVParam,
		attnO:         attnOParam,
		hidden:        hiddenParam,
		proj:          projParam,
		attnEnabled:   attnEnabled,
		hiddenEnabled: hiddenEnabled,
	}, nil
}

func resolveEmbeddingTrainerWeights(params embeddingTrainerParams, manifest EmbeddingManifest, weights map[string]*backend.Tensor) (embeddingTrainerWeights, error) {
	tokenEmbed, err := requireTrainingWeight(weights, params.token.Name)
	if err != nil {
		return embeddingTrainerWeights{}, err
	}
	projection, err := requireTrainingWeight(weights, params.proj.Name)
	if err != nil {
		return embeddingTrainerWeights{}, err
	}
	out := embeddingTrainerWeights{
		token:      tokenEmbed,
		projection: projection,
	}
	if manifest.roleConditioned() {
		out.role, err = requireTrainingWeight(weights, params.role.Name)
		if err != nil {
			return embeddingTrainerWeights{}, err
		}
	}
	if params.attnEnabled {
		out.attentionQuery, err = requireTrainingWeight(weights, params.attnQ.Name)
		if err != nil {
			return embeddingTrainerWeights{}, err
		}
		out.attentionKey, err = requireTrainingWeight(weights, params.attnK.Name)
		if err != nil {
			return embeddingTrainerWeights{}, err
		}
		out.attentionValue, err = requireTrainingWeight(weights, params.attnV.Name)
		if err != nil {
			return embeddingTrainerWeights{}, err
		}
		out.attentionOutput, err = requireTrainingWeight(weights, params.attnO.Name)
		if err != nil {
			return embeddingTrainerWeights{}, err
		}
	}
	if params.hiddenEnabled {
		out.hiddenProjection, err = requireTrainingWeight(weights, params.hidden.Name)
		if err != nil {
			return embeddingTrainerWeights{}, err
		}
	}
	if err := validateEmbeddingTrainerWeightShapes(params, manifest, out); err != nil {
		return embeddingTrainerWeights{}, err
	}
	return out, nil
}

func requireTrainingWeight(weights map[string]*backend.Tensor, name string) (*backend.Tensor, error) {
	tensor, ok := weights[name]
	if !ok || tensor == nil {
		return nil, fmt.Errorf("missing training weight %q", name)
	}
	return tensor, nil
}

func optionalRoleEmbeddingParam(mod *eosartifact.Module, manifest EmbeddingManifest) (eosartifact.Param, error) {
	if !manifest.roleConditioned() {
		return eosartifact.Param{}, nil
	}
	if manifest.RoleEmbeddingParam == "" {
		return eosartifact.Param{}, fmt.Errorf("role-conditioned manifest requires role_embedding_param")
	}
	return requireTrainableEmbeddingParam(mod, manifest.RoleEmbeddingParam)
}

func maxInt32(values ...int32) int32 {
	var out int32
	for _, value := range values {
		if value > out {
			out = value
		}
	}
	return out
}

func validateEmbeddingTrainerWeightShapes(params embeddingTrainerParams, manifest EmbeddingManifest, weights embeddingTrainerWeights) error {
	if len(weights.token.Shape) != 2 {
		return fmt.Errorf("training weight %q rank = %d, want 2", params.token.Name, len(weights.token.Shape))
	}
	if manifest.roleConditioned() {
		if len(weights.role.Shape) != 2 {
			return fmt.Errorf("training weight %q rank = %d, want 2", params.role.Name, len(weights.role.Shape))
		}
		if weights.role.Shape[1] != weights.token.Shape[1] {
			return fmt.Errorf("role embedding width %d does not match token embedding width %d", weights.role.Shape[1], weights.token.Shape[1])
		}
		maxRole := maxInt32(manifest.RawRoleIndex, manifest.QueryRoleIndex, manifest.DocumentRoleIndex)
		if int(maxRole) >= weights.role.Shape[0] {
			return fmt.Errorf("role embedding rows %d cannot cover role index %d", weights.role.Shape[0], maxRole)
		}
	}
	if params.attnEnabled {
		for name, tensor := range map[string]*backend.Tensor{
			params.attnQ.Name: weights.attentionQuery,
			params.attnK.Name: weights.attentionKey,
			params.attnV.Name: weights.attentionValue,
			params.attnO.Name: weights.attentionOutput,
		} {
			if len(tensor.Shape) != 2 {
				return fmt.Errorf("training weight %q rank = %d, want 2", name, len(tensor.Shape))
			}
			if tensor.Shape[0] != weights.token.Shape[1] || tensor.Shape[1] != weights.token.Shape[1] {
				return fmt.Errorf("attention weight %q shape %v does not match embedding width %d", name, tensor.Shape, weights.token.Shape[1])
			}
		}
	}
	if params.hiddenEnabled && len(weights.hiddenProjection.Shape) != 2 {
		return fmt.Errorf("training weight %q rank = %d, want 2", params.hidden.Name, len(weights.hiddenProjection.Shape))
	}
	if len(weights.projection.Shape) != 2 {
		return fmt.Errorf("training weight %q rank = %d, want 2", params.proj.Name, len(weights.projection.Shape))
	}
	if params.hiddenEnabled {
		if weights.token.Shape[1] != weights.hiddenProjection.Shape[0] {
			return fmt.Errorf("embedding/hidden projection shape mismatch %v x %v", weights.token.Shape, weights.hiddenProjection.Shape)
		}
		if weights.hiddenProjection.Shape[1] != weights.projection.Shape[0] {
			return fmt.Errorf("hidden/output projection shape mismatch %v x %v", weights.hiddenProjection.Shape, weights.projection.Shape)
		}
		if manifest.FFNResidual && weights.projection.Shape[1] != weights.token.Shape[1] {
			return fmt.Errorf("ffn residual requires projection output width %d to match hidden width %d", weights.projection.Shape[1], weights.token.Shape[1])
		}
	} else if weights.token.Shape[1] != weights.projection.Shape[0] {
		return fmt.Errorf("embedding/projection shape mismatch %v x %v", weights.token.Shape, weights.projection.Shape)
	}
	if vocab := manifest.Tokenizer.VocabSize; vocab > 0 && weights.token.Shape[0] < vocab {
		return fmt.Errorf("training token embedding rows %d are smaller than vocab_size %d", weights.token.Shape[0], vocab)
	}
	return nil
}

// Close releases any backend-owned trainer acceleration state.
func (t *EmbeddingTrainer) Close() {
	if t == nil {
		return
	}
	if t.forwardMatMul != nil {
		t.forwardMatMul.Close()
		t.forwardMatMul = nil
		t.forwardBackend = ""
	}
	if t.optimizerAccel != nil {
		t.optimizerAccel.Close()
		t.optimizerAccel = nil
		t.optimizerBackend = ""
	}
	if t.activationAccel != nil {
		t.activationAccel.Close()
		t.activationAccel = nil
		t.activationBackend = ""
	}
	if t.contrastiveAccel != nil {
		t.contrastiveAccel.Close()
		t.contrastiveAccel = nil
		t.contrastiveBackend = ""
	}
}

func (t *EmbeddingTrainer) MemoryPlan() *MemoryPlan {
	if t == nil {
		return nil
	}
	return cloneMemoryPlan(t.memoryPlan)
}

func (t *EmbeddingTrainer) syncOptimizerState(includeMoments bool) error {
	if t == nil || t.optimizerAccel == nil {
		return nil
	}
	if !includeMoments || !t.momentsDirty {
		return nil
	}
	bindings := []struct {
		name string
		t    *backend.Tensor
		m1   *backend.Tensor
		m2   *backend.Tensor
	}{
		{name: t.tokenParam.Name, t: t.tokenEmbed, m1: t.tokenMom1, m2: t.tokenMom2},
		{name: t.roleParam.Name, t: t.roleEmbed, m1: t.roleMom1, m2: t.roleMom2},
		{name: t.attnQParam.Name, t: t.attentionQuery, m1: t.attnQMom1, m2: t.attnQMom2},
		{name: t.attnKParam.Name, t: t.attentionKey, m1: t.attnKMom1, m2: t.attnKMom2},
		{name: t.attnVParam.Name, t: t.attentionValue, m1: t.attnVMom1, m2: t.attnVMom2},
		{name: t.attnOParam.Name, t: t.attentionOutput, m1: t.attnOMom1, m2: t.attnOMom2},
		{name: t.hiddenParam.Name, t: t.hiddenProjection, m1: t.hiddenMom1, m2: t.hiddenMom2},
		{name: t.projParam.Name, t: t.projection, m1: t.projMom1, m2: t.projMom2},
	}
	for _, binding := range bindings {
		if binding.name == "" || binding.t == nil {
			continue
		}
		if err := t.optimizerAccel.SyncState(binding.name, binding.t, binding.m1, binding.m2, includeMoments); err != nil {
			return err
		}
	}
	t.momentsDirty = false
	return nil
}

func (t *EmbeddingTrainer) primeForwardWeightResidency(attnQForward, attnKForward, attnVForward, attnOForward, hiddenForward, projForward *backend.Tensor) {
	if t == nil || t.forwardMatMul == nil {
		return
	}
	if !t.forwardNeedsBind &&
		t.boundForward.attnQ == attnQForward &&
		t.boundForward.attnK == attnKForward &&
		t.boundForward.attnV == attnVForward &&
		t.boundForward.attnO == attnOForward &&
		t.boundForward.hidden == hiddenForward &&
		t.boundForward.proj == projForward {
		t.forwardBindSkips++
		return
	}
	bindings := []struct {
		name   string
		tensor *backend.Tensor
	}{
		{name: t.attnQParam.Name, tensor: attnQForward},
		{name: t.attnKParam.Name, tensor: attnKForward},
		{name: t.attnVParam.Name, tensor: attnVForward},
		{name: t.attnOParam.Name, tensor: attnOForward},
		{name: t.hiddenParam.Name, tensor: hiddenForward},
		{name: t.projParam.Name, tensor: projForward},
	}
	for _, binding := range bindings {
		if binding.name == "" || binding.tensor == nil {
			continue
		}
		if err := t.forwardMatMul.BindMatrix(binding.name, binding.tensor); err != nil {
			t.Close()
			return
		}
	}
	t.boundForward = embeddingForwardWeights{
		attnQ:  attnQForward,
		attnK:  attnKForward,
		attnV:  attnVForward,
		attnO:  attnOForward,
		hidden: hiddenForward,
		proj:   projForward,
	}
	t.forwardNeedsBind = false
}

func (t *EmbeddingTrainer) primeCompactForwardWeightResidency(forward *compactEmbeddingForwardWeights) {
	if t == nil || t.forwardMatMul == nil || forward == nil {
		return
	}
	if !t.forwardNeedsBind && t.boundForward.compact == forward {
		t.forwardBindSkips++
		return
	}
	for _, layer := range forward.layers {
		bindings := []struct {
			name   string
			tensor *backend.Tensor
		}{
			{name: layer.attnQName, tensor: layer.attnQ},
			{name: layer.attnKName, tensor: layer.attnK},
			{name: layer.attnVName, tensor: layer.attnV},
			{name: layer.attnOName, tensor: layer.attnO},
			{name: layer.ffnUpName, tensor: layer.ffnUp},
			{name: layer.ffnDownName, tensor: layer.ffnDown},
		}
		for _, binding := range bindings {
			if binding.name == "" || binding.tensor == nil {
				continue
			}
			if err := t.forwardMatMul.BindMatrix(binding.name, binding.tensor); err != nil {
				t.Close()
				return
			}
		}
	}
	if forward.outputProjectionName != "" && forward.outputProjection != nil {
		if err := t.forwardMatMul.BindMatrix(forward.outputProjectionName, forward.outputProjection); err != nil {
			t.Close()
			return
		}
	}
	t.boundForward = embeddingForwardWeights{compact: forward}
	t.forwardNeedsBind = false
}

// ForwardResidencyStats reports trainer-level residency reuse plus backend matmul prep activity.
func (t *EmbeddingTrainer) ForwardResidencyStats() EmbeddingForwardResidencyStats {
	if t == nil {
		return EmbeddingForwardResidencyStats{}
	}
	stats := EmbeddingForwardResidencyStats{
		BindSkips: t.forwardBindSkips,
	}
	if t.forwardMatMul != nil {
		stats.MatMul = t.forwardMatMul.Stats()
	}
	return stats
}

// TrainProfile snapshots backend and residency activity for the current trainer state.
func (t *EmbeddingTrainer) TrainProfile() EmbeddingTrainProfile {
	if t == nil {
		return EmbeddingTrainProfile{Version: EmbeddingTrainProfileVersion}
	}
	profile := EmbeddingTrainProfile{
		Version:             EmbeddingTrainProfileVersion,
		Step:                t.step,
		ForwardBackend:      t.forwardBackend,
		OptimizerBackend:    t.optimizerBackend,
		ActivationBackend:   t.activationBackend,
		ContrastiveBackend:  t.contrastiveBackend,
		ForwardResidency:    t.ForwardResidencyStats(),
		VectorDistillPhases: t.vectorDistillPhases,
	}
	if t.optimizerAccel != nil {
		profile.Optimizer = t.optimizerAccel.Stats()
	}
	if t.isCompactTrainer() {
		profile.Optimizer.UpdateCalls = t.compactOptimizerUpdates
	} else {
		profile.Optimizer.UpdateCalls += t.compactOptimizerUpdates
	}
	if t.activationAccel != nil {
		profile.Activation = t.activationAccel.Stats()
	}
	if t.contrastiveAccel != nil {
		profile.Contrastive = t.contrastiveAccel.Stats()
	}
	return profile
}

func (t *EmbeddingTrainer) nextSequenceBindingPrefix() string {
	if t == nil {
		return ""
	}
	t.sequenceBindingID++
	return fmt.Sprintf("seq_%d", t.sequenceBindingID)
}

func (t *EmbeddingTrainer) bindSequenceTensor(state *embeddingSequenceState, slot string, tensor *backend.Tensor, bindMatMul, bindActivation bool) string {
	if t == nil || state == nil || tensor == nil {
		return ""
	}
	bindMatMul = bindMatMul && t.forwardMatMul != nil && sequenceMatMulBindingsEnabled()
	bindActivation = bindActivation && t.activationAccel != nil
	if !bindMatMul && !bindActivation {
		return ""
	}
	prefix := state.bindingPrefix
	if prefix == "" {
		prefix = t.nextSequenceBindingPrefix()
		state.bindingPrefix = prefix
	}
	name := prefix + "_" + slot
	if bindMatMul {
		if err := t.forwardMatMul.BindMatrix(name, tensor); err != nil {
			return ""
		}
	}
	if bindActivation {
		if err := t.activationAccel.BindTensor(name, tensor); err != nil {
			if bindMatMul {
				_ = t.forwardMatMul.UnbindMatrix(name)
			}
			return ""
		}
	}
	return name
}

func (t *EmbeddingTrainer) bindSoftmaxActivationForBatchedForward() bool {
	return !batchedBackwardEnabled() || !t.softmaxBackwardAccelEnabled()
}

func (t *EmbeddingTrainer) bindFullActivationForBatchedForward() bool {
	return !batchedBackwardEnabled() || !t.fullActivationBackwardAccelEnabled()
}

func (state *embeddingSequenceState) skipUnboundActivationBackward(bindingNames ...string) bool {
	if state == nil || !state.skipUnboundActAccel {
		return false
	}
	for _, name := range bindingNames {
		if name == "" {
			return true
		}
	}
	return false
}

func (t *EmbeddingTrainer) unbindSequenceTensor(name string) {
	if t == nil || name == "" {
		return
	}
	if t.forwardMatMul != nil {
		_ = t.forwardMatMul.UnbindMatrix(name)
	}
	if t.activationAccel != nil {
		_ = t.activationAccel.UnbindTensor(name)
	}
}

func (t *EmbeddingTrainer) releaseSequenceBindings(state *embeddingSequenceState) {
	if t == nil || state == nil {
		return
	}
	for _, name := range []string{
		state.inputBinding,
		state.hiddenBinding,
		state.attnQBinding,
		state.attnKBinding,
		state.attnVBinding,
		state.attnScoresBinding,
		state.attnMixedBinding,
		state.attnResidualBinding,
		state.ffnHiddenBinding,
		state.activatedBinding,
		state.ffnResidualBinding,
		state.projectedBinding,
	} {
		if name == "" {
			continue
		}
		t.unbindSequenceTensor(name)
	}
	state.inputBinding = ""
	state.hiddenBinding = ""
	state.attnQBinding = ""
	state.attnKBinding = ""
	state.attnVBinding = ""
	state.attnScoresBinding = ""
	state.attnMixedBinding = ""
	state.attnResidualBinding = ""
	state.ffnHiddenBinding = ""
	state.activatedBinding = ""
	state.ffnResidualBinding = ""
	state.projectedBinding = ""
	state.bindingPrefix = ""
}

func (t *EmbeddingTrainer) releaseEncodedSequenceBindings(seq *embeddingEncodedSequence) {
	if t == nil || seq == nil {
		return
	}
	for _, layer := range seq.layers {
		t.releaseSequenceBindings(layer)
	}
}

func (t *EmbeddingTrainer) releaseEncodedSequences(seqs []*embeddingEncodedSequence) {
	if t == nil {
		return
	}
	for _, seq := range seqs {
		t.releaseEncodedSequenceBindings(seq)
	}
}

// EvalBatch runs the quantization-aware forward path without mutating weights.
func (t *EmbeddingTrainer) EvalBatch(batch []EmbeddingPairExample) (EmbeddingTrainMetrics, error) {
	return t.runBatch(batch, false)
}

// configureRetrievalEval enables the retrieval-nDCG eval gate for this run. When
// rt and a complete BEIR-style cfg (corpus/queries/qrels) are supplied, every
// eval additionally computes nDCG@10 over that set with the current weights.
func (t *EmbeddingTrainer) configureRetrievalEval(rt *Runtime, cfg RetrievalEvalConfig, tok *TokenizerFile) {
	if t == nil {
		return
	}
	t.retrievalEvalRuntime = rt
	t.retrievalEvalConfig = cfg
	t.retrievalEvalTokenizer = tok
	t.retrievalEvalEnabled = rt != nil &&
		strings.TrimSpace(cfg.CorpusPath) != "" &&
		strings.TrimSpace(cfg.QueriesPath) != "" &&
		strings.TrimSpace(cfg.QrelsPath) != ""
}

// augmentRetrievalMetrics fills retrieval metrics by embedding the configured
// held-out corpus + queries with the trainer's current in-memory weights and
// scoring them against the qrels. No-op when the gate is not enabled.
func (t *EmbeddingTrainer) augmentRetrievalMetrics(metrics *EmbeddingEvalMetrics) error {
	if t == nil || metrics == nil || !t.retrievalEvalEnabled {
		return nil
	}
	// Mid-run, accelerated optimizers hold the live weights on device; sync
	// the host masters first (as Checkpoint does) or the exported weights are
	// stale and the retrieval score silently reflects an older model.
	if err := t.syncOptimizerState(true); err != nil {
		return fmt.Errorf("retrieval eval: sync weights: %w", err)
	}
	weights, err := t.ExportInferenceWeights()
	if err != nil {
		return fmt.Errorf("retrieval eval: export weights: %w", err)
	}
	opts := make([]LoadOption, 0, len(weights))
	for name, tensor := range weights {
		opts = append(opts, WithWeight(name, tensor))
	}
	model, err := t.retrievalEvalRuntime.LoadEmbedding(context.Background(), t.module, t.manifest, opts...)
	if err != nil {
		return fmt.Errorf("retrieval eval: load model: %w", err)
	}
	if t.retrievalEvalTokenizer != nil {
		if err := model.attachTokenizer(*t.retrievalEvalTokenizer); err != nil {
			return fmt.Errorf("retrieval eval: attach tokenizer: %w", err)
		}
	}
	result, err := EvaluateEmbeddingRetrieval(context.Background(), model, t.retrievalEvalConfig)
	if err != nil {
		return fmt.Errorf("retrieval eval: %w", err)
	}
	metrics.RetrievalNDCGAt10 = float32(result.Quality.NDCGAt10)
	metrics.RetrievalMAPAt100 = float32(result.Quality.MAPAt100)
	metrics.RetrievalRecallAt100 = float32(result.Quality.RecallAt100)
	metrics.RetrievalEval = &result
	return nil
}

// EvaluatePairs runs the forward path and returns ship-facing pairwise quality metrics.
func (t *EmbeddingTrainer) EvaluatePairs(batch []EmbeddingPairExample) (EmbeddingEvalMetrics, error) {
	if t == nil {
		return EmbeddingEvalMetrics{}, fmt.Errorf("embedding trainer is not initialized")
	}
	if len(batch) == 0 {
		return EmbeddingEvalMetrics{}, fmt.Errorf("evaluation batch is empty")
	}
	forward := t.prepareForwardWeights()
	t.primeForwardWeightResidency(
		forward.attnQ,
		forward.attnK,
		forward.attnV,
		forward.attnO,
		forward.hidden,
		forward.proj,
	)

	metrics := EmbeddingEvalMetrics{PairCount: len(batch)}
	scores := make([]embeddingEvalScore, 0, len(batch))
	rankScores := make([]embeddingEvalRankScore, 0, len(batch))
	if ok, err := t.evaluatePairsBatchedForward(batch, forward, &metrics, &scores, &rankScores); err != nil {
		return EmbeddingEvalMetrics{}, err
	} else if !ok {
		for i, example := range batch {
			score, loss, err := t.scoreExamplePair(example, forward)
			if err != nil {
				return EmbeddingEvalMetrics{}, fmt.Errorf("batch %d: %w", i, err)
			}
			accumulateEvalPairScore(&metrics, &scores, example.Target, score, loss)
			appendEvalRankScore(&rankScores, example, score)
		}
	}
	invPairs := float32(1) / float32(len(batch))
	metrics.Loss *= invPairs
	metrics.AverageScore *= invPairs
	metrics.PairAccuracy *= invPairs
	if metrics.PositiveCount > 0 {
		metrics.PositiveMeanScore /= float32(metrics.PositiveCount)
	}
	if metrics.NegativeCount > 0 {
		metrics.NegativeMeanScore /= float32(metrics.NegativeCount)
	}
	metrics.ScoreMargin = metrics.PositiveMeanScore - metrics.NegativeMeanScore
	finalizeEvalScoreMetrics(&metrics, scores)
	finalizeEvalRankMetrics(&metrics, rankScores)
	if err := t.augmentRetrievalMetrics(&metrics); err != nil {
		return EmbeddingEvalMetrics{}, err
	}
	return metrics, nil
}

func (t *EmbeddingTrainer) evaluatePairsBatchedForward(batch []EmbeddingPairExample, forward *embeddingForwardWeights, metrics *EmbeddingEvalMetrics, scores *[]embeddingEvalScore, rankScores *[]embeddingEvalRankScore) (bool, error) {
	if t.isCompactTrainer() {
		return false, nil
	}
	if t == nil || t.forwardMatMul == nil || forward == nil || metrics == nil || scores == nil || rankScores == nil || len(batch) == 0 || !batchedPairwiseEvalEnabled() {
		return false, nil
	}
	chunkSize := pairwiseEvalBatchSize(len(batch))
	for start := 0; start < len(batch); start += chunkSize {
		end := start + chunkSize
		if end > len(batch) {
			end = len(batch)
		}
		chunk := batch[start:end]
		lefts, rights, ok, err := t.tryEncodePairBatchBatchedForward(chunk, forward, false)
		if err != nil {
			return true, err
		}
		if !ok {
			return false, nil
		}
		var chunkErr error
		for i, example := range chunk {
			if i >= len(lefts) || i >= len(rights) || lefts[i] == nil || rights[i] == nil {
				chunkErr = fmt.Errorf("batch %d: encoder produced nil pair", start+i)
				break
			}
			score, _, _ := cosineGrad(lefts[i].pooled, rights[i].pooled)
			scale := score - example.Target
			accumulateEvalPairScore(metrics, scores, example.Target, score, 0.5*scale*scale)
			appendEvalRankScore(rankScores, example, score)
		}
		t.releaseEncodedSequences(lefts)
		t.releaseEncodedSequences(rights)
		if chunkErr != nil {
			return true, chunkErr
		}
	}
	return true, nil
}

func accumulateEvalPairScore(metrics *EmbeddingEvalMetrics, scores *[]embeddingEvalScore, target, score, loss float32) {
	if metrics == nil || scores == nil {
		return
	}
	metrics.Loss += loss
	metrics.AverageScore += score
	positive := target > 0
	*scores = append(*scores, embeddingEvalScore{Score: score, Positive: positive})
	if positive {
		metrics.PositiveCount++
		metrics.PositiveMeanScore += score
		if score > 0 {
			metrics.PairAccuracy++
		}
	} else {
		metrics.NegativeCount++
		metrics.NegativeMeanScore += score
		if score < 0 {
			metrics.PairAccuracy++
		}
	}
}

func finalizeEvalScoreMetrics(metrics *EmbeddingEvalMetrics, scores []embeddingEvalScore) {
	if metrics == nil || len(scores) == 0 || metrics.PositiveCount == 0 || metrics.NegativeCount == 0 {
		return
	}
	metrics.ThresholdAccuracy, metrics.ScoreThreshold = bestEvalScoreThresholdAccuracy(scores, metrics.PositiveCount, metrics.NegativeCount)
	metrics.ROCAUC = evalScoreROCAUC(scores, metrics.PositiveCount, metrics.NegativeCount)
}

func appendEvalRankScore(scores *[]embeddingEvalRankScore, example EmbeddingPairExample, score float32) {
	if scores == nil || len(example.LeftTokens) == 0 {
		return
	}
	*scores = append(*scores, embeddingEvalRankScore{
		QueryKey: embeddingBatchSequenceKey(example.LeftTokens, example.LeftMask, 0),
		Score:    score,
		Positive: example.Target > 0,
	})
}

func finalizeEvalRankMetrics(metrics *EmbeddingEvalMetrics, scores []embeddingEvalRankScore) {
	if metrics == nil || len(scores) == 0 {
		return
	}
	groups := make(map[string][]embeddingEvalRankScore)
	for _, score := range scores {
		if score.QueryKey == "" {
			continue
		}
		groups[score.QueryKey] = append(groups[score.QueryKey], score)
	}
	queryCount := 0
	for _, group := range groups {
		if len(group) == 0 {
			continue
		}
		bestPositive := float32(math.Inf(-1))
		hasPositive := false
		for _, score := range group {
			if score.Positive && (!hasPositive || score.Score > bestPositive) {
				bestPositive = score.Score
				hasPositive = true
			}
		}
		if !hasPositive {
			continue
		}
		rank := 1
		for _, score := range group {
			if !score.Positive && score.Score > bestPositive {
				rank++
			}
		}
		if rank == 1 {
			metrics.Top1Accuracy++
		}
		if rank <= 5 {
			metrics.Top5Accuracy++
		}
		if rank <= 10 {
			metrics.Top10Accuracy++
		}
		metrics.MeanReciprocalRank += 1 / float32(rank)
		metrics.MeanPositiveRank += float32(rank)
		queryCount++
	}
	if queryCount == 0 {
		return
	}
	invQueries := float32(1) / float32(queryCount)
	metrics.Top1Accuracy *= invQueries
	metrics.Top5Accuracy *= invQueries
	metrics.Top10Accuracy *= invQueries
	metrics.MeanReciprocalRank *= invQueries
	metrics.MeanPositiveRank *= invQueries
}

func bestEvalScoreThresholdAccuracy(scores []embeddingEvalScore, positiveCount, negativeCount int) (float32, float32) {
	if len(scores) == 0 {
		return 0, 0
	}
	ordered := append([]embeddingEvalScore(nil), scores...)
	sort.Slice(ordered, func(i, j int) bool {
		return ordered[i].Score > ordered[j].Score
	})
	bestCorrect := negativeCount
	bestThreshold := ordered[0].Score
	seenPositives := 0
	seenNegatives := 0
	for i := 0; i < len(ordered); {
		j := i + 1
		groupPositives := 0
		groupNegatives := 0
		for j <= len(ordered) {
			current := ordered[j-1]
			if current.Positive {
				groupPositives++
			} else {
				groupNegatives++
			}
			if j == len(ordered) || ordered[j].Score != ordered[i].Score {
				break
			}
			j++
		}
		seenPositives += groupPositives
		seenNegatives += groupNegatives
		correct := seenPositives + (negativeCount - seenNegatives)
		if correct > bestCorrect {
			bestCorrect = correct
			bestThreshold = ordered[i].Score
		}
		i = j
	}
	return float32(bestCorrect) / float32(len(ordered)), bestThreshold
}

func evalScoreROCAUC(scores []embeddingEvalScore, positiveCount, negativeCount int) float32 {
	if len(scores) == 0 || positiveCount == 0 || negativeCount == 0 {
		return 0
	}
	ordered := append([]embeddingEvalScore(nil), scores...)
	sort.Slice(ordered, func(i, j int) bool {
		return ordered[i].Score < ordered[j].Score
	})
	rankSumPositives := float64(0)
	for i := 0; i < len(ordered); {
		j := i + 1
		positivesInGroup := 0
		if ordered[i].Positive {
			positivesInGroup++
		}
		for j < len(ordered) && ordered[j].Score == ordered[i].Score {
			if ordered[j].Positive {
				positivesInGroup++
			}
			j++
		}
		averageRank := (float64(i+1) + float64(j)) * 0.5
		rankSumPositives += float64(positivesInGroup) * averageRank
		i = j
	}
	positives := float64(positiveCount)
	negatives := float64(negativeCount)
	auc := (rankSumPositives - positives*(positives+1)*0.5) / (positives * negatives)
	if auc < 0 {
		return 0
	}
	if auc > 1 {
		return 1
	}
	return float32(auc)
}

// EvaluateContrastive runs pairwise contrastive evaluation without re-encoding each expanded pair.
func (t *EmbeddingTrainer) EvaluateContrastive(batch []EmbeddingContrastiveExample) (EmbeddingEvalMetrics, error) {
	if t == nil {
		return EmbeddingEvalMetrics{}, fmt.Errorf("embedding trainer is not initialized")
	}
	if len(batch) == 0 {
		return EmbeddingEvalMetrics{}, fmt.Errorf("evaluation batch is empty")
	}
	forward := t.prepareForwardWeights()
	t.primeForwardWeightResidency(
		forward.attnQ,
		forward.attnK,
		forward.attnV,
		forward.attnO,
		forward.hidden,
		forward.proj,
	)

	queries, positives, err := t.encodeContrastiveBatch(batch, forward, false)
	if err != nil {
		return EmbeddingEvalMetrics{}, err
	}
	defer t.releaseEncodedSequences(queries)
	defer t.releaseEncodedSequences(positives)
	metrics := evaluateContrastiveEncodings(queries, positives, t.config)
	if err := t.augmentRetrievalMetrics(&metrics); err != nil {
		return EmbeddingEvalMetrics{}, err
	}
	return metrics, nil
}

// TrainStep runs one SGD-style update over a batch of pairwise examples.
func (t *EmbeddingTrainer) TrainStep(batch []EmbeddingPairExample) (EmbeddingTrainMetrics, error) {
	return t.runBatch(batch, true)
}

// TrainContrastiveStep runs one update over a contrastive batch without expanding it into repeated pair encodes.
func (t *EmbeddingTrainer) TrainContrastiveStep(batch []EmbeddingContrastiveExample) (EmbeddingTrainMetrics, error) {
	if t == nil {
		return EmbeddingTrainMetrics{}, fmt.Errorf("embedding trainer is not initialized")
	}
	if t.isCompactTrainer() {
		return t.runCompactContrastiveBatchUpdate(batch)
	}
	if len(batch) < 2 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("contrastive training batch needs at least 2 examples")
	}
	forward := t.prepareForwardWeights()
	t.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	queries, positives, err := t.encodeContrastiveBatch(batch, forward, true)
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	defer t.releaseEncodedSequences(queries)
	defer t.releaseEncodedSequences(positives)

	gradToken := make([]float32, len(t.tokenEmbed.F32))
	gradRole := make([]float32, tensorDataLen(t.roleEmbed))
	gradAttnQ := make([]float32, tensorDataLen(t.attentionQuery))
	gradAttnK := make([]float32, tensorDataLen(t.attentionKey))
	gradAttnV := make([]float32, tensorDataLen(t.attentionValue))
	gradAttnO := make([]float32, tensorDataLen(t.attentionOutput))
	gradHidden := make([]float32, len(t.hiddenProjectionData()))
	gradProj := make([]float32, len(t.projection.F32))
	queryGrads := make([][]float32, len(queries))
	positiveGrads := make([][]float32, len(positives))
	for i := range queries {
		queryGrads[i] = make([]float32, len(queries[i].pooled))
		positiveGrads[i] = make([]float32, len(positives[i].pooled))
	}

	totalLoss := float32(0)
	totalScore := float32(0)
	pairCount := len(batch) * len(batch)
	batchScale := float32(1) / float32(pairCount)
	lossScale := batchScale
	if embeddingUsesInfoNCELoss(t.config.ContrastiveLoss) {
		var ok bool
		if len(t.config.MatryoshkaDims) == 0 {
			totalLoss, totalScore, ok = t.tryInfoNCEContrastiveAccelerator(queries, positives, queryGrads, positiveGrads)
		}
		if !ok {
			totalLoss, totalScore = accumulateInfoNCEContrastiveGrads(queries, positives, t.config.Temperature, queryGrads, positiveGrads)
		}
		batchScale = float32(1) / float32(len(batch))
		lossScale = batchScale
	} else {
		totalLoss, totalScore = accumulatePairMSEContrastiveGrads(queries, positives, queryGrads, positiveGrads)
	}
	prefixLoss, prefixScore, prefixPairs := accumulateMatryoshkaContrastiveGrads(queries, positives, t.config, queryGrads, positiveGrads)
	turboPrefixLoss, turboPrefixScore, turboPrefixPairs := accumulateTurboQuantPrefixContrastiveGrads(queries, positives, t.config, queryGrads, positiveGrads)
	if prefixPairs+turboPrefixPairs > 0 {
		weightSum := matryoshkaWeightSum(t.config.MatryoshkaWeights) + turboQuantPrefixWeightSum(t.config)
		objectiveScale := float32(1) / (1 + weightSum)
		scaleEmbeddingGradBuffers(queryGrads, objectiveScale)
		scaleEmbeddingGradBuffers(positiveGrads, objectiveScale)
		totalLoss = totalLoss*objectiveScale + (prefixLoss+turboPrefixLoss)*objectiveScale
		totalScore += prefixScore + turboPrefixScore
		pairCount += prefixPairs + turboPrefixPairs
	}

	if !t.tryBackpropContrastiveBatch(
		queries,
		positives,
		queryGrads,
		positiveGrads,
		forward.attnQ,
		forward.attnK,
		forward.attnV,
		forward.attnO,
		forward.hidden,
		forward.proj,
		gradToken,
		gradAttnQ,
		gradAttnK,
		gradAttnV,
		gradAttnO,
		gradHidden,
		gradProj,
	) {
		for i, query := range queries {
			inputGrad := t.backpropEncodedSequence(query, queryGrads[i], forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
			t.accumulateInputTokenGrad(query.tokens, inputGrad, gradToken)
			t.accumulateInputRoleGrad(query.role, len(query.tokens), inputGrad, gradRole)
		}
		for i, positive := range positives {
			inputGrad := t.backpropEncodedSequence(positive, positiveGrads[i], forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
			t.accumulateInputTokenGrad(positive.tokens, inputGrad, gradToken)
			t.accumulateInputRoleGrad(positive.role, len(positive.tokens), inputGrad, gradRole)
		}
	}

	t.step++
	t.applyOptimizerUpdate(t.tokenParam.Name, t.tokenEmbed, t.tokenMom1, t.tokenMom2, gradToken, batchScale)
	t.applyOptimizerUpdate(t.roleParam.Name, t.roleEmbed, t.roleMom1, t.roleMom2, gradRole, batchScale)
	t.applyOptimizerUpdate(t.attnQParam.Name, t.attentionQuery, t.attnQMom1, t.attnQMom2, gradAttnQ, batchScale)
	t.applyOptimizerUpdate(t.attnKParam.Name, t.attentionKey, t.attnKMom1, t.attnKMom2, gradAttnK, batchScale)
	t.applyOptimizerUpdate(t.attnVParam.Name, t.attentionValue, t.attnVMom1, t.attnVMom2, gradAttnV, batchScale)
	t.applyOptimizerUpdate(t.attnOParam.Name, t.attentionOutput, t.attnOMom1, t.attnOMom2, gradAttnO, batchScale)
	t.applyOptimizerUpdate(t.hiddenParam.Name, t.hiddenProjection, t.hiddenMom1, t.hiddenMom2, gradHidden, batchScale)
	t.applyOptimizerUpdate(t.projParam.Name, t.projection, t.projMom1, t.projMom2, gradProj, batchScale)
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * lossScale,
		AverageScore: totalScore / float32(pairCount),
		BatchSize:    pairCount,
	}, nil
}

func (t *EmbeddingTrainer) runCompactContrastiveBatchUpdate(batch []EmbeddingContrastiveExample) (EmbeddingTrainMetrics, error) {
	if t == nil || t.compactState == nil {
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact embedding trainer is not initialized")
	}
	if len(batch) < 2 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("contrastive training batch needs at least 2 examples")
	}
	if len(t.config.MatryoshkaDims) > 0 ||
		len(t.config.TurboQuantPrefixBits) > 0 ||
		len(t.config.TurboQuantPrefixObjectives) > 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact_transformer_v1 contrastive training supports pair-MSE and InfoNCE only")
	}
	forward := t.prepareForwardWeights()
	if forward == nil || forward.compact == nil {
		return EmbeddingTrainMetrics{}, fmt.Errorf("missing compact forward weights")
	}
	queries, positives, err := t.encodeContrastiveBatch(batch, forward, false)
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	defer t.releaseEncodedSequences(queries)
	defer t.releaseEncodedSequences(positives)

	grads := newCompactEmbeddingGradState(t.compactState)
	queryGrads := make([][]float32, len(queries))
	positiveGrads := make([][]float32, len(positives))
	for i := range queries {
		queryGrads[i] = make([]float32, len(queries[i].pooled))
		positiveGrads[i] = make([]float32, len(positives[i].pooled))
	}

	pairCount := len(batch) * len(batch)
	batchScale := float32(1) / float32(pairCount)
	lossScale := batchScale
	var totalLoss, totalScore float32
	if embeddingUsesInfoNCELoss(t.config.ContrastiveLoss) {
		totalLoss, totalScore = accumulateInfoNCEContrastiveGrads(queries, positives, t.config.Temperature, queryGrads, positiveGrads)
		batchScale = float32(1) / float32(len(batch))
		lossScale = batchScale
	} else {
		totalLoss, totalScore = accumulatePairMSEContrastiveGrads(queries, positives, queryGrads, positiveGrads)
	}
	for i, query := range queries {
		if err := t.backpropCompactEncodedSequence(query, queryGrads[i], forward.compact, grads); err != nil {
			return EmbeddingTrainMetrics{}, fmt.Errorf("query %d: %w", i, err)
		}
	}
	for i, positive := range positives {
		if err := t.backpropCompactEncodedSequence(positive, positiveGrads[i], forward.compact, grads); err != nil {
			return EmbeddingTrainMetrics{}, fmt.Errorf("positive %d: %w", i, err)
		}
	}
	t.applyCompactOptimizerUpdates(grads, batchScale)
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * lossScale,
		AverageScore: totalScore / float32(pairCount),
		BatchSize:    pairCount,
	}, nil
}

// TrainHardNegativeContrastiveStep runs one InfoNCE update over query-positive-negative batches.
func (t *EmbeddingTrainer) TrainHardNegativeContrastiveStep(batch []EmbeddingHardNegativeExample) (EmbeddingTrainMetrics, error) {
	if t == nil {
		return EmbeddingTrainMetrics{}, fmt.Errorf("embedding trainer is not initialized")
	}
	if t.isCompactTrainer() {
		return t.runCompactHardNegativeContrastiveBatchUpdate(batch)
	}
	if len(batch) == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("hard-negative training batch is empty")
	}
	queryInputs := make([]embeddingSequenceInput, len(batch))
	candidateInputs := make([]embeddingSequenceInput, 0, len(batch)*2)
	targetIndexes := make([]int, len(batch))
	candidateSpans := make([]embeddingCandidateSpan, len(batch))
	teacherScores := make([][]float32, len(batch))
	for i, example := range batch {
		queryInputs[i] = embeddingSequenceInput{
			tokens: example.QueryTokens,
			mask:   example.QueryMask,
			role:   t.queryRoleIndex(),
			label:  fmt.Sprintf("batch %d query", i),
		}
		targetIndexes[i] = len(candidateInputs)
		candidateSpans[i].Start = len(candidateInputs)
		candidateInputs = append(candidateInputs, embeddingSequenceInput{
			tokens: example.PositiveTokens,
			mask:   example.PositiveMask,
			role:   t.documentRoleIndex(),
			label:  fmt.Sprintf("batch %d positive", i),
		})
		for j, tokens := range example.NegativeTokens {
			var mask []int32
			if j < len(example.NegativeMasks) {
				mask = example.NegativeMasks[j]
			}
			candidateInputs = append(candidateInputs, embeddingSequenceInput{
				tokens: tokens,
				mask:   mask,
				role:   t.documentRoleIndex(),
				label:  fmt.Sprintf("batch %d negative %d", i, j),
			})
		}
		candidateSpans[i].End = len(candidateInputs)
		if len(example.TeacherScores) > 0 {
			if len(example.TeacherScores) != candidateSpans[i].End-candidateSpans[i].Start {
				return EmbeddingTrainMetrics{}, fmt.Errorf("hard-negative teacher_scores length %d does not match candidate count %d for batch %d", len(example.TeacherScores), candidateSpans[i].End-candidateSpans[i].Start, i)
			}
			teacherScores[i] = example.TeacherScores
		}
	}
	if len(candidateInputs) < 2 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("hard-negative training batch needs at least two candidate documents")
	}
	for _, target := range targetIndexes {
		if target < 0 || target >= len(candidateInputs) {
			return EmbeddingTrainMetrics{}, fmt.Errorf("hard-negative target index %d is outside %d candidates", target, len(candidateInputs))
		}
	}

	forward := t.prepareForwardWeights()
	t.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	allInputs := make([]embeddingSequenceInput, 0, len(queryInputs)+len(candidateInputs))
	allInputs = append(allInputs, queryInputs...)
	allInputs = append(allInputs, candidateInputs...)
	encoded, err := t.encodeSequenceInputs(allInputs, forward, true)
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	defer t.releaseEncodedSequences(encoded)
	queries := encoded[:len(queryInputs)]
	candidates := encoded[len(queryInputs):]

	gradToken := make([]float32, len(t.tokenEmbed.F32))
	gradRole := make([]float32, tensorDataLen(t.roleEmbed))
	gradAttnQ := make([]float32, tensorDataLen(t.attentionQuery))
	gradAttnK := make([]float32, tensorDataLen(t.attentionKey))
	gradAttnV := make([]float32, tensorDataLen(t.attentionValue))
	gradAttnO := make([]float32, tensorDataLen(t.attentionOutput))
	gradHidden := make([]float32, len(t.hiddenProjectionData()))
	gradProj := make([]float32, len(t.projection.F32))
	queryGrads := make([][]float32, len(queries))
	candidateGrads := make([][]float32, len(candidates))
	for i := range queries {
		queryGrads[i] = make([]float32, len(queries[i].pooled))
	}
	for i := range candidates {
		candidateGrads[i] = make([]float32, len(candidates[i].pooled))
	}

	totalLoss := float32(0)
	totalScore := float32(0)
	if t.config.ContrastiveLoss == "grouped_infonce" {
		totalLoss, totalScore = accumulateGroupedInfoNCEHardNegativeGrads(queries, candidates, candidateSpans, t.config.Temperature, queryGrads, candidateGrads)
	} else if t.config.ContrastiveLoss == "hybrid_infonce" {
		var ok bool
		globalLoss := float32(0)
		globalScore := float32(0)
		if len(t.config.MatryoshkaDims) == 0 {
			globalLoss, globalScore, ok = t.tryInfoNCEHardNegativeAccelerator(queries, candidates, targetIndexes, queryGrads, candidateGrads)
		}
		if !ok {
			globalLoss, globalScore = accumulateInfoNCEHardNegativeGrads(queries, candidates, targetIndexes, t.config.Temperature, queryGrads, candidateGrads)
		}
		groupedQueryGrads := newEmbeddingPooledGradBuffers(queries)
		groupedCandidateGrads := newEmbeddingPooledGradBuffers(candidates)
		groupedLoss, groupedScore := accumulateGroupedInfoNCEHardNegativeGrads(queries, candidates, candidateSpans, t.config.Temperature, groupedQueryGrads, groupedCandidateGrads)
		groupedWeight := effectiveGroupedLossWeight(t.config.ContrastiveLoss, t.config.GroupedLossWeight)
		globalScale := float32(1) / (1 + groupedWeight)
		groupedScale := groupedWeight / (1 + groupedWeight)
		scaleEmbeddingGradBuffers(queryGrads, globalScale)
		scaleEmbeddingGradBuffers(candidateGrads, globalScale)
		addScaledEmbeddingGradBuffers(queryGrads, groupedQueryGrads, groupedScale)
		addScaledEmbeddingGradBuffers(candidateGrads, groupedCandidateGrads, groupedScale)
		totalLoss = globalLoss*globalScale + groupedLoss*groupedScale
		totalScore = globalScore + groupedScore
	} else {
		var ok bool
		if len(t.config.MatryoshkaDims) == 0 {
			totalLoss, totalScore, ok = t.tryInfoNCEHardNegativeAccelerator(queries, candidates, targetIndexes, queryGrads, candidateGrads)
		}
		if !ok {
			totalLoss, totalScore = accumulateInfoNCEHardNegativeGrads(queries, candidates, targetIndexes, t.config.Temperature, queryGrads, candidateGrads)
		}
	}
	teacherPairCount := 0
	teacherWeight := t.config.TeacherLossWeight
	if teacherWeight > 0 {
		teacherQueryGrads := newEmbeddingPooledGradBuffers(queries)
		teacherCandidateGrads := newEmbeddingPooledGradBuffers(candidates)
		teacherTemperatures := make([]float32, len(batch))
		teacherSourceWeights := make([]float32, len(batch))
		for i, example := range batch {
			teacherTemperatures[i] = hardNegativeTeacherTemperature(t.config.TeacherSourceTemperatures, example.Source, t.config.TeacherTemperature)
			teacherSourceWeights[i] = hardNegativeTeacherWeight(t.config.TeacherSourceWeights, example.Source)
		}
		teacherLoss, teacherScore, pairs := accumulateTeacherDistributionHardNegativeGrads(queries, candidates, candidateSpans, teacherScores, teacherTemperatures, teacherSourceWeights, t.config.Temperature, t.config.TeacherTemperature, teacherQueryGrads, teacherCandidateGrads)
		if pairs > 0 {
			baseScale := float32(1) / (1 + teacherWeight)
			teacherScale := teacherWeight / (1 + teacherWeight)
			scaleEmbeddingGradBuffers(queryGrads, baseScale)
			scaleEmbeddingGradBuffers(candidateGrads, baseScale)
			addScaledEmbeddingGradBuffers(queryGrads, teacherQueryGrads, teacherScale)
			addScaledEmbeddingGradBuffers(candidateGrads, teacherCandidateGrads, teacherScale)
			totalLoss = totalLoss*baseScale + teacherLoss*teacherScale
			totalScore += teacherScore
			teacherPairCount = pairs
		}
	}
	pairCount := hardNegativeCandidatePairCount(len(queries), len(candidates), candidateSpans, t.config.ContrastiveLoss) + teacherPairCount
	prefixLoss, prefixScore, prefixPairs := accumulateMatryoshkaHardNegativeGrads(queries, candidates, targetIndexes, candidateSpans, t.config, queryGrads, candidateGrads)
	turboPrefixLoss, turboPrefixScore, turboPrefixPairs := accumulateTurboQuantPrefixHardNegativeGrads(queries, candidates, targetIndexes, candidateSpans, t.config, queryGrads, candidateGrads)
	compactLoss, compactScore, compactPairs := accumulateTurboQuantCompactHardNegativeGrads(queries, candidates, candidateSpans, teacherScores, t.config, queryGrads, candidateGrads)
	rankLoss, rankScore, rankPairs := accumulateTurboQuantRankMarginHardNegativeGrads(queries, candidates, candidateSpans, teacherScores, t.config, queryGrads, candidateGrads)
	if prefixPairs+turboPrefixPairs+compactPairs+rankPairs > 0 {
		weightSum := matryoshkaWeightSum(t.config.MatryoshkaWeights) + turboQuantPrefixWeightSum(t.config) + turboQuantCompactWeightSum(t.config) + turboQuantRankMarginWeightSum(t.config)
		objectiveScale := float32(1) / (1 + weightSum)
		scaleEmbeddingGradBuffers(queryGrads, objectiveScale)
		scaleEmbeddingGradBuffers(candidateGrads, objectiveScale)
		totalLoss = totalLoss*objectiveScale + (prefixLoss+turboPrefixLoss+compactLoss+rankLoss)*objectiveScale
		totalScore += prefixScore + turboPrefixScore + compactScore + rankScore
		pairCount += prefixPairs + turboPrefixPairs + compactPairs + rankPairs
	}
	if !t.tryBackpropContrastiveBatch(
		queries,
		candidates,
		queryGrads,
		candidateGrads,
		forward.attnQ,
		forward.attnK,
		forward.attnV,
		forward.attnO,
		forward.hidden,
		forward.proj,
		gradToken,
		gradAttnQ,
		gradAttnK,
		gradAttnV,
		gradAttnO,
		gradHidden,
		gradProj,
	) {
		for i, query := range queries {
			inputGrad := t.backpropEncodedSequence(query, queryGrads[i], forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
			t.accumulateInputTokenGrad(query.tokens, inputGrad, gradToken)
			t.accumulateInputRoleGrad(query.role, len(query.tokens), inputGrad, gradRole)
		}
		for i, candidate := range candidates {
			inputGrad := t.backpropEncodedSequence(candidate, candidateGrads[i], forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
			t.accumulateInputTokenGrad(candidate.tokens, inputGrad, gradToken)
			t.accumulateInputRoleGrad(candidate.role, len(candidate.tokens), inputGrad, gradRole)
		}
	}

	batchScale := float32(1) / float32(len(queries))
	t.step++
	t.applyOptimizerUpdate(t.tokenParam.Name, t.tokenEmbed, t.tokenMom1, t.tokenMom2, gradToken, batchScale)
	t.applyOptimizerUpdate(t.roleParam.Name, t.roleEmbed, t.roleMom1, t.roleMom2, gradRole, batchScale)
	t.applyOptimizerUpdate(t.attnQParam.Name, t.attentionQuery, t.attnQMom1, t.attnQMom2, gradAttnQ, batchScale)
	t.applyOptimizerUpdate(t.attnKParam.Name, t.attentionKey, t.attnKMom1, t.attnKMom2, gradAttnK, batchScale)
	t.applyOptimizerUpdate(t.attnVParam.Name, t.attentionValue, t.attnVMom1, t.attnVMom2, gradAttnV, batchScale)
	t.applyOptimizerUpdate(t.attnOParam.Name, t.attentionOutput, t.attnOMom1, t.attnOMom2, gradAttnO, batchScale)
	t.applyOptimizerUpdate(t.hiddenParam.Name, t.hiddenProjection, t.hiddenMom1, t.hiddenMom2, gradHidden, batchScale)
	t.applyOptimizerUpdate(t.projParam.Name, t.projection, t.projMom1, t.projMom2, gradProj, batchScale)
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * batchScale,
		AverageScore: totalScore / float32(pairCount),
		BatchSize:    pairCount,
	}, nil
}

func (t *EmbeddingTrainer) runCompactHardNegativeContrastiveBatchUpdate(batch []EmbeddingHardNegativeExample) (EmbeddingTrainMetrics, error) {
	if t == nil || t.compactState == nil {
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact embedding trainer is not initialized")
	}
	if len(batch) == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("hard-negative training batch is empty")
	}
	if len(t.config.MatryoshkaDims) > 0 ||
		len(t.config.TurboQuantPrefixBits) > 0 ||
		len(t.config.TurboQuantPrefixObjectives) > 0 ||
		len(t.config.TurboQuantCompactObjectives) > 0 ||
		len(t.config.TurboQuantRankMarginObjectives) > 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact_transformer_v1 hard-negative training supports InfoNCE and grouped InfoNCE only")
	}
	queryInputs := make([]embeddingSequenceInput, len(batch))
	candidateInputs := make([]embeddingSequenceInput, 0, len(batch)*2)
	targetIndexes := make([]int, len(batch))
	candidateSpans := make([]embeddingCandidateSpan, len(batch))
	teacherScores := make([][]float32, len(batch))
	for i, example := range batch {
		queryInputs[i] = embeddingSequenceInput{
			tokens: example.QueryTokens,
			mask:   example.QueryMask,
			role:   t.queryRoleIndex(),
			label:  fmt.Sprintf("batch %d query", i),
		}
		targetIndexes[i] = len(candidateInputs)
		candidateSpans[i].Start = len(candidateInputs)
		candidateInputs = append(candidateInputs, embeddingSequenceInput{
			tokens: example.PositiveTokens,
			mask:   example.PositiveMask,
			role:   t.documentRoleIndex(),
			label:  fmt.Sprintf("batch %d positive", i),
		})
		for j, tokens := range example.NegativeTokens {
			var mask []int32
			if j < len(example.NegativeMasks) {
				mask = example.NegativeMasks[j]
			}
			candidateInputs = append(candidateInputs, embeddingSequenceInput{
				tokens: tokens,
				mask:   mask,
				role:   t.documentRoleIndex(),
				label:  fmt.Sprintf("batch %d negative %d", i, j),
			})
		}
		candidateSpans[i].End = len(candidateInputs)
		if len(example.TeacherScores) > 0 {
			if len(example.TeacherScores) != candidateSpans[i].End-candidateSpans[i].Start {
				return EmbeddingTrainMetrics{}, fmt.Errorf("hard-negative teacher_scores length %d does not match candidate count %d for batch %d", len(example.TeacherScores), candidateSpans[i].End-candidateSpans[i].Start, i)
			}
			teacherScores[i] = example.TeacherScores
		}
	}
	if len(candidateInputs) < 2 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("hard-negative training batch needs at least two candidate documents")
	}
	for _, target := range targetIndexes {
		if target < 0 || target >= len(candidateInputs) {
			return EmbeddingTrainMetrics{}, fmt.Errorf("hard-negative target index %d is outside %d candidates", target, len(candidateInputs))
		}
	}

	forward := t.prepareForwardWeights()
	if forward == nil || forward.compact == nil {
		return EmbeddingTrainMetrics{}, fmt.Errorf("missing compact forward weights")
	}
	allInputs := make([]embeddingSequenceInput, 0, len(queryInputs)+len(candidateInputs))
	allInputs = append(allInputs, queryInputs...)
	allInputs = append(allInputs, candidateInputs...)
	encoded, err := t.encodeSequenceInputs(allInputs, forward, true)
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	defer t.releaseEncodedSequences(encoded)
	queries := encoded[:len(queryInputs)]
	candidates := encoded[len(queryInputs):]

	grads := newCompactEmbeddingGradState(t.compactState)
	queryGrads := make([][]float32, len(queries))
	candidateGrads := make([][]float32, len(candidates))
	for i := range queries {
		queryGrads[i] = make([]float32, len(queries[i].pooled))
	}
	for i := range candidates {
		candidateGrads[i] = make([]float32, len(candidates[i].pooled))
	}

	var totalLoss, totalScore float32
	switch t.config.ContrastiveLoss {
	case "grouped_infonce":
		totalLoss, totalScore = accumulateGroupedInfoNCEHardNegativeGrads(queries, candidates, candidateSpans, t.config.Temperature, queryGrads, candidateGrads)
	case "", "pair_mse", "infonce":
		totalLoss, totalScore = accumulateInfoNCEHardNegativeGrads(queries, candidates, targetIndexes, t.config.Temperature, queryGrads, candidateGrads)
	default:
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact_transformer_v1 hard-negative training does not support contrastive_loss %q", t.config.ContrastiveLoss)
	}

	teacherPairCount := 0
	teacherWeight := t.config.TeacherLossWeight
	if teacherWeight > 0 {
		teacherQueryGrads := newEmbeddingPooledGradBuffers(queries)
		teacherCandidateGrads := newEmbeddingPooledGradBuffers(candidates)
		teacherTemperatures := make([]float32, len(batch))
		teacherSourceWeights := make([]float32, len(batch))
		for i, example := range batch {
			teacherTemperatures[i] = hardNegativeTeacherTemperature(t.config.TeacherSourceTemperatures, example.Source, t.config.TeacherTemperature)
			teacherSourceWeights[i] = hardNegativeTeacherWeight(t.config.TeacherSourceWeights, example.Source)
		}
		teacherLoss, teacherScore, pairs := accumulateTeacherDistributionHardNegativeGrads(queries, candidates, candidateSpans, teacherScores, teacherTemperatures, teacherSourceWeights, t.config.Temperature, t.config.TeacherTemperature, teacherQueryGrads, teacherCandidateGrads)
		if pairs > 0 {
			baseScale := float32(1) / (1 + teacherWeight)
			teacherScale := teacherWeight / (1 + teacherWeight)
			scaleEmbeddingGradBuffers(queryGrads, baseScale)
			scaleEmbeddingGradBuffers(candidateGrads, baseScale)
			addScaledEmbeddingGradBuffers(queryGrads, teacherQueryGrads, teacherScale)
			addScaledEmbeddingGradBuffers(candidateGrads, teacherCandidateGrads, teacherScale)
			totalLoss = totalLoss*baseScale + teacherLoss*teacherScale
			totalScore += teacherScore
			teacherPairCount = pairs
		}
	}

	for i, query := range queries {
		if err := t.backpropCompactEncodedSequence(query, queryGrads[i], forward.compact, grads); err != nil {
			return EmbeddingTrainMetrics{}, fmt.Errorf("query %d: %w", i, err)
		}
	}
	for i, candidate := range candidates {
		if err := t.backpropCompactEncodedSequence(candidate, candidateGrads[i], forward.compact, grads); err != nil {
			return EmbeddingTrainMetrics{}, fmt.Errorf("candidate %d: %w", i, err)
		}
	}

	batchScale := float32(1) / float32(len(queries))
	t.applyCompactOptimizerUpdates(grads, batchScale)
	pairCount := hardNegativeCandidatePairCount(len(queries), len(candidates), candidateSpans, t.config.ContrastiveLoss) + teacherPairCount
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * batchScale,
		AverageScore: totalScore / float32(pairCount),
		BatchSize:    pairCount,
	}, nil
}

// TrainScoreSpectrumStep runs one row-local ranked-candidate score-spectrum update.
func (t *EmbeddingTrainer) TrainScoreSpectrumStep(batch []EmbeddingScoreSpectrumExample) (EmbeddingTrainMetrics, error) {
	if t == nil {
		return EmbeddingTrainMetrics{}, fmt.Errorf("embedding trainer is not initialized")
	}
	if t.isCompactTrainer() {
		return t.runCompactScoreSpectrumBatchUpdate(batch)
	}
	if len(batch) == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("score-spectrum training batch is empty")
	}
	if err := validateScoreSpectrumTrainerConfig(t.config); err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	canonicalBatch, err := canonicalizeTokenizedScoreSpectrumExamples(batch)
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	queryInputs := make([]embeddingSequenceInput, len(canonicalBatch))
	candidateInputs := make([]embeddingSequenceInput, 0, len(canonicalBatch)*2)
	candidateSpans := make([]embeddingCandidateSpan, len(canonicalBatch))
	for i, example := range canonicalBatch {
		if err := validateTokenizedScoreSpectrumShape(example); err != nil {
			return EmbeddingTrainMetrics{}, fmt.Errorf("score-spectrum batch %d: %w", i, err)
		}
		queryInputs[i] = embeddingSequenceInput{
			tokens: example.QueryTokens,
			mask:   example.QueryMask,
			role:   t.queryRoleIndex(),
			label:  fmt.Sprintf("batch %d query", i),
		}
		candidateSpans[i].Start = len(candidateInputs)
		for j, tokens := range example.CandidateTokens {
			var mask []int32
			if j < len(example.CandidateMasks) {
				mask = example.CandidateMasks[j]
			}
			candidateInputs = append(candidateInputs, embeddingSequenceInput{
				tokens: tokens,
				mask:   mask,
				role:   t.documentRoleIndex(),
				label:  fmt.Sprintf("batch %d candidate %d", i, j),
			})
		}
		candidateSpans[i].End = len(candidateInputs)
	}

	forward := t.prepareForwardWeights()
	t.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	allInputs := make([]embeddingSequenceInput, 0, len(queryInputs)+len(candidateInputs))
	allInputs = append(allInputs, queryInputs...)
	allInputs = append(allInputs, candidateInputs...)
	encoded, err := t.encodeSequenceInputs(allInputs, forward, true)
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	defer t.releaseEncodedSequences(encoded)
	queries := encoded[:len(queryInputs)]
	candidates := encoded[len(queryInputs):]

	gradToken := make([]float32, len(t.tokenEmbed.F32))
	gradRole := make([]float32, tensorDataLen(t.roleEmbed))
	gradAttnQ := make([]float32, tensorDataLen(t.attentionQuery))
	gradAttnK := make([]float32, tensorDataLen(t.attentionKey))
	gradAttnV := make([]float32, tensorDataLen(t.attentionValue))
	gradAttnO := make([]float32, tensorDataLen(t.attentionOutput))
	gradHidden := make([]float32, len(t.hiddenProjectionData()))
	gradProj := make([]float32, len(t.projection.F32))
	queryGrads := make([][]float32, len(queries))
	candidateGrads := make([][]float32, len(candidates))
	for i := range queries {
		queryGrads[i] = make([]float32, len(queries[i].pooled))
	}
	for i := range candidates {
		candidateGrads[i] = make([]float32, len(candidates[i].pooled))
	}

	totalLoss, totalScore, pairCount, err := accumulateScoreSpectrumGrads(queries, candidates, candidateSpans, canonicalBatch, t.config, queryGrads, candidateGrads)
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	if pairCount == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("score-spectrum training batch has no usable candidates")
	}
	if !t.tryBackpropContrastiveBatch(
		queries,
		candidates,
		queryGrads,
		candidateGrads,
		forward.attnQ,
		forward.attnK,
		forward.attnV,
		forward.attnO,
		forward.hidden,
		forward.proj,
		gradToken,
		gradAttnQ,
		gradAttnK,
		gradAttnV,
		gradAttnO,
		gradHidden,
		gradProj,
	) {
		for i, query := range queries {
			inputGrad := t.backpropEncodedSequence(query, queryGrads[i], forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
			t.accumulateInputTokenGrad(query.tokens, inputGrad, gradToken)
			t.accumulateInputRoleGrad(query.role, len(query.tokens), inputGrad, gradRole)
		}
		for i, candidate := range candidates {
			inputGrad := t.backpropEncodedSequence(candidate, candidateGrads[i], forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
			t.accumulateInputTokenGrad(candidate.tokens, inputGrad, gradToken)
			t.accumulateInputRoleGrad(candidate.role, len(candidate.tokens), inputGrad, gradRole)
		}
	}

	batchScale := float32(1) / float32(len(queries))
	t.step++
	t.applyOptimizerUpdate(t.tokenParam.Name, t.tokenEmbed, t.tokenMom1, t.tokenMom2, gradToken, batchScale)
	t.applyOptimizerUpdate(t.roleParam.Name, t.roleEmbed, t.roleMom1, t.roleMom2, gradRole, batchScale)
	t.applyOptimizerUpdate(t.attnQParam.Name, t.attentionQuery, t.attnQMom1, t.attnQMom2, gradAttnQ, batchScale)
	t.applyOptimizerUpdate(t.attnKParam.Name, t.attentionKey, t.attnKMom1, t.attnKMom2, gradAttnK, batchScale)
	t.applyOptimizerUpdate(t.attnVParam.Name, t.attentionValue, t.attnVMom1, t.attnVMom2, gradAttnV, batchScale)
	t.applyOptimizerUpdate(t.attnOParam.Name, t.attentionOutput, t.attnOMom1, t.attnOMom2, gradAttnO, batchScale)
	t.applyOptimizerUpdate(t.hiddenParam.Name, t.hiddenProjection, t.hiddenMom1, t.hiddenMom2, gradHidden, batchScale)
	t.applyOptimizerUpdate(t.projParam.Name, t.projection, t.projMom1, t.projMom2, gradProj, batchScale)
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * batchScale,
		AverageScore: totalScore / float32(pairCount),
		BatchSize:    pairCount,
	}, nil
}

func (t *EmbeddingTrainer) runCompactScoreSpectrumBatchUpdate(batch []EmbeddingScoreSpectrumExample) (EmbeddingTrainMetrics, error) {
	if t == nil || t.compactState == nil {
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact embedding trainer is not initialized")
	}
	if len(batch) == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("score-spectrum training batch is empty")
	}
	if len(t.config.MatryoshkaDims) > 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact_transformer_v1 score-spectrum training does not support matryoshka objectives")
	}
	if len(t.config.TurboQuantPrefixBits) > 0 || len(t.config.TurboQuantPrefixObjectives) > 0 || len(turboQuantPrefixObjectivesForConfig(t.config)) > 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact_transformer_v1 score-spectrum training does not support turboquant prefix objectives")
	}
	if len(t.config.TurboQuantCompactObjectives) > 0 || len(turboQuantCompactObjectivesForConfig(t.config)) > 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact_transformer_v1 score-spectrum training does not support turboquant compact objectives")
	}
	if len(t.config.TurboQuantRankMarginObjectives) > 0 || len(turboQuantRankMarginObjectivesForConfig(t.config)) > 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact_transformer_v1 score-spectrum training does not support turboquant rank-margin objectives")
	}
	if err := validateScoreSpectrumTrainerConfig(t.config); err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	canonicalBatch, err := canonicalizeTokenizedScoreSpectrumExamples(batch)
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	queryInputs, candidateInputs, candidateSpans, err := scoreSpectrumSequenceInputs(canonicalBatch, t.queryRoleIndex(), t.documentRoleIndex())
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}

	forward := t.prepareForwardWeights()
	if forward == nil || forward.compact == nil {
		return EmbeddingTrainMetrics{}, fmt.Errorf("missing compact forward weights")
	}
	allInputs := make([]embeddingSequenceInput, 0, len(queryInputs)+len(candidateInputs))
	allInputs = append(allInputs, queryInputs...)
	allInputs = append(allInputs, candidateInputs...)
	encoded, err := t.encodeSequenceInputs(allInputs, forward, true)
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	defer t.releaseEncodedSequences(encoded)
	queries := encoded[:len(queryInputs)]
	candidates := encoded[len(queryInputs):]

	grads := newCompactEmbeddingGradState(t.compactState)
	queryGrads := make([][]float32, len(queries))
	candidateGrads := make([][]float32, len(candidates))
	for i := range queries {
		queryGrads[i] = make([]float32, len(queries[i].pooled))
	}
	for i := range candidates {
		candidateGrads[i] = make([]float32, len(candidates[i].pooled))
	}

	totalLoss, totalScore, pairCount, err := accumulateScoreSpectrumGrads(queries, candidates, candidateSpans, canonicalBatch, t.config, queryGrads, candidateGrads)
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	if pairCount == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("score-spectrum training batch has no usable candidates")
	}
	for i, query := range queries {
		if err := t.backpropCompactEncodedSequence(query, queryGrads[i], forward.compact, grads); err != nil {
			return EmbeddingTrainMetrics{}, fmt.Errorf("query %d: %w", i, err)
		}
	}
	for i, candidate := range candidates {
		if err := t.backpropCompactEncodedSequence(candidate, candidateGrads[i], forward.compact, grads); err != nil {
			return EmbeddingTrainMetrics{}, fmt.Errorf("candidate %d: %w", i, err)
		}
	}

	batchScale := float32(1) / float32(len(queries))
	t.applyCompactOptimizerUpdates(grads, batchScale)
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * batchScale,
		AverageScore: totalScore / float32(pairCount),
		BatchSize:    pairCount,
	}, nil
}

// TrainListwiseGeometryStep runs one update over row-local query/document teacher geometry batches.
func (t *EmbeddingTrainer) TrainListwiseGeometryStep(batch []EmbeddingTokenizedListwiseGeometryBatch) (EmbeddingTrainMetrics, error) {
	return t.TrainListwiseGeometryStepWithDiagnostics(batch, false)
}

func (t *EmbeddingTrainer) TrainListwiseGeometryStepWithDiagnostics(batch []EmbeddingTokenizedListwiseGeometryBatch, diagnostics bool) (EmbeddingTrainMetrics, error) {
	if t == nil {
		return EmbeddingTrainMetrics{}, fmt.Errorf("embedding trainer is not initialized")
	}
	if t.isCompactTrainer() {
		return t.runCompactListwiseGeometryBatchUpdate(batch, diagnostics)
	}
	if len(batch) == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("listwise geometry training batch is empty")
	}
	if err := validateListwiseGeometryTrainerConfig(t.config); err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	queryInputs, documentInputs, spans, err := listwiseGeometrySequenceInputs(batch, t.queryRoleIndex(), t.documentRoleIndex())
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	forward := t.prepareForwardWeights()
	t.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	allInputs := make([]embeddingSequenceInput, 0, len(queryInputs)+len(documentInputs))
	allInputs = append(allInputs, queryInputs...)
	allInputs = append(allInputs, documentInputs...)
	encoded, err := t.encodeSequenceInputs(allInputs, forward, true)
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	defer t.releaseEncodedSequences(encoded)
	queries := encoded[:len(queryInputs)]
	documents := encoded[len(queryInputs):]

	gradToken := make([]float32, len(t.tokenEmbed.F32))
	gradRole := make([]float32, tensorDataLen(t.roleEmbed))
	gradAttnQ := make([]float32, tensorDataLen(t.attentionQuery))
	gradAttnK := make([]float32, tensorDataLen(t.attentionKey))
	gradAttnV := make([]float32, tensorDataLen(t.attentionValue))
	gradAttnO := make([]float32, tensorDataLen(t.attentionOutput))
	gradHidden := make([]float32, len(t.hiddenProjectionData()))
	gradProj := make([]float32, len(t.projection.F32))
	queryGrads := make([][]float32, len(queries))
	documentGrads := make([][]float32, len(documents))
	for i := range queries {
		queryGrads[i] = make([]float32, len(queries[i].pooled))
	}
	for i := range documents {
		documentGrads[i] = make([]float32, len(documents[i].pooled))
	}

	totalLoss, totalScore, pairCount, queryCount, err := accumulateListwiseGeometryGrads(queries, documents, spans, batch, t.config.Temperature, queryGrads, documentGrads)
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	if pairCount == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("listwise geometry training batch has no usable query-document pairs")
	}
	if queryCount == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("listwise geometry training batch has no usable queries")
	}
	if !t.tryBackpropContrastiveBatch(
		queries,
		documents,
		queryGrads,
		documentGrads,
		forward.attnQ,
		forward.attnK,
		forward.attnV,
		forward.attnO,
		forward.hidden,
		forward.proj,
		gradToken,
		gradAttnQ,
		gradAttnK,
		gradAttnV,
		gradAttnO,
		gradHidden,
		gradProj,
	) {
		for i, query := range queries {
			inputGrad := t.backpropEncodedSequence(query, queryGrads[i], forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
			t.accumulateInputTokenGrad(query.tokens, inputGrad, gradToken)
			t.accumulateInputRoleGrad(query.role, len(query.tokens), inputGrad, gradRole)
		}
		for i, document := range documents {
			inputGrad := t.backpropEncodedSequence(document, documentGrads[i], forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
			t.accumulateInputTokenGrad(document.tokens, inputGrad, gradToken)
			t.accumulateInputRoleGrad(document.role, len(document.tokens), inputGrad, gradRole)
		}
	}

	batchScale := float32(1) / float32(queryCount)
	var movement *EmbeddingTrainMovementMetrics
	var beforeUpdate []embeddingTrainTensorSnapshot
	if diagnostics {
		movement = &EmbeddingTrainMovementMetrics{
			Gradient: aggregateScaledGradientStats(batchScale, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj),
		}
		beforeUpdate = snapshotEmbeddingTrainTensors(t.tokenEmbed, t.attentionQuery, t.attentionKey, t.attentionValue, t.attentionOutput, t.hiddenProjection, t.projection)
	}
	t.step++
	t.applyOptimizerUpdate(t.tokenParam.Name, t.tokenEmbed, t.tokenMom1, t.tokenMom2, gradToken, batchScale)
	t.applyOptimizerUpdate(t.roleParam.Name, t.roleEmbed, t.roleMom1, t.roleMom2, gradRole, batchScale)
	t.applyOptimizerUpdate(t.attnQParam.Name, t.attentionQuery, t.attnQMom1, t.attnQMom2, gradAttnQ, batchScale)
	t.applyOptimizerUpdate(t.attnKParam.Name, t.attentionKey, t.attnKMom1, t.attnKMom2, gradAttnK, batchScale)
	t.applyOptimizerUpdate(t.attnVParam.Name, t.attentionValue, t.attnVMom1, t.attnVMom2, gradAttnV, batchScale)
	t.applyOptimizerUpdate(t.attnOParam.Name, t.attentionOutput, t.attnOMom1, t.attnOMom2, gradAttnO, batchScale)
	t.applyOptimizerUpdate(t.hiddenParam.Name, t.hiddenProjection, t.hiddenMom1, t.hiddenMom2, gradHidden, batchScale)
	t.applyOptimizerUpdate(t.projParam.Name, t.projection, t.projMom1, t.projMom2, gradProj, batchScale)
	if movement != nil {
		movement.ParameterDelta = aggregateParameterDeltaStats(beforeUpdate)
	}
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * batchScale,
		AverageScore: totalScore / float32(pairCount),
		BatchSize:    pairCount,
		Movement:     movement,
	}, nil
}

func (t *EmbeddingTrainer) runCompactListwiseGeometryBatchUpdate(batch []EmbeddingTokenizedListwiseGeometryBatch, diagnostics bool) (EmbeddingTrainMetrics, error) {
	if t == nil || t.compactState == nil {
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact embedding trainer is not initialized")
	}
	if len(batch) == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("listwise geometry training batch is empty")
	}
	if len(t.config.MatryoshkaDims) > 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact_transformer_v1 listwise geometry training does not support matryoshka objectives")
	}
	if len(t.config.TurboQuantPrefixBits) > 0 || len(t.config.TurboQuantPrefixObjectives) > 0 || len(turboQuantPrefixObjectivesForConfig(t.config)) > 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact_transformer_v1 listwise geometry training does not support turboquant prefix objectives")
	}
	if len(t.config.TurboQuantCompactObjectives) > 0 || len(turboQuantCompactObjectivesForConfig(t.config)) > 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact_transformer_v1 listwise geometry training does not support turboquant compact objectives")
	}
	if len(t.config.TurboQuantRankMarginObjectives) > 0 || len(turboQuantRankMarginObjectivesForConfig(t.config)) > 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact_transformer_v1 listwise geometry training does not support turboquant rank-margin objectives")
	}
	if err := validateListwiseGeometryTrainerConfig(t.config); err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	queryInputs, documentInputs, spans, err := listwiseGeometrySequenceInputs(batch, t.queryRoleIndex(), t.documentRoleIndex())
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	forward := t.prepareForwardWeights()
	if forward == nil || forward.compact == nil {
		return EmbeddingTrainMetrics{}, fmt.Errorf("missing compact forward weights")
	}
	allInputs := make([]embeddingSequenceInput, 0, len(queryInputs)+len(documentInputs))
	allInputs = append(allInputs, queryInputs...)
	allInputs = append(allInputs, documentInputs...)
	encoded, err := t.encodeSequenceInputs(allInputs, forward, true)
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	defer t.releaseEncodedSequences(encoded)
	queries := encoded[:len(queryInputs)]
	documents := encoded[len(queryInputs):]

	grads := newCompactEmbeddingGradState(t.compactState)
	queryGrads := make([][]float32, len(queries))
	documentGrads := make([][]float32, len(documents))
	for i := range queries {
		queryGrads[i] = make([]float32, len(queries[i].pooled))
	}
	for i := range documents {
		documentGrads[i] = make([]float32, len(documents[i].pooled))
	}

	totalLoss, totalScore, pairCount, queryCount, err := accumulateListwiseGeometryGrads(queries, documents, spans, batch, t.config.Temperature, queryGrads, documentGrads)
	if err != nil {
		return EmbeddingTrainMetrics{}, err
	}
	if pairCount == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("listwise geometry training batch has no usable query-document pairs")
	}
	if queryCount == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("listwise geometry training batch has no usable queries")
	}
	for i, query := range queries {
		if err := t.backpropCompactEncodedSequence(query, queryGrads[i], forward.compact, grads); err != nil {
			return EmbeddingTrainMetrics{}, fmt.Errorf("query %d: %w", i, err)
		}
	}
	for i, document := range documents {
		if err := t.backpropCompactEncodedSequence(document, documentGrads[i], forward.compact, grads); err != nil {
			return EmbeddingTrainMetrics{}, fmt.Errorf("document %d: %w", i, err)
		}
	}

	batchScale := float32(1) / float32(queryCount)
	var movement *EmbeddingTrainMovementMetrics
	var beforeUpdate []embeddingTrainTensorSnapshot
	if diagnostics {
		movement = &EmbeddingTrainMovementMetrics{
			Gradient: aggregateScaledGradientStats(batchScale, compactEmbeddingGradSlices(grads)...),
		}
		beforeUpdate = snapshotCompactEmbeddingTrainTensors(t.compactState)
	}
	t.applyCompactOptimizerUpdates(grads, batchScale)
	if movement != nil {
		movement.ParameterDelta = aggregateParameterDeltaStats(beforeUpdate)
	}
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * batchScale,
		AverageScore: totalScore / float32(pairCount),
		BatchSize:    pairCount,
		Movement:     movement,
	}, nil
}

// EvaluateScoreSpectrum scores tokenized score-spectrum examples without
// applying optimizer updates or consulting train-policy gates.
func (t *EmbeddingTrainer) EvaluateScoreSpectrum(examples []EmbeddingScoreSpectrumExample) (EmbeddingScoreSpectrumEvalMetrics, error) {
	return t.EvaluateScoreSpectrumBatched(examples, len(examples))
}

// EvaluateScoreSpectrumBatched scores tokenized score-spectrum examples in
// bounded row chunks without applying optimizer updates or consulting
// train-policy gates.
func (t *EmbeddingTrainer) EvaluateScoreSpectrumBatched(examples []EmbeddingScoreSpectrumExample, batchSize int) (EmbeddingScoreSpectrumEvalMetrics, error) {
	if t == nil {
		return EmbeddingScoreSpectrumEvalMetrics{}, fmt.Errorf("embedding trainer is not initialized")
	}
	if len(examples) == 0 {
		return EmbeddingScoreSpectrumEvalMetrics{}, fmt.Errorf("score-spectrum eval dataset is empty")
	}
	if batchSize <= 0 || batchSize > len(examples) {
		batchSize = len(examples)
	}
	canonicalExamples, err := canonicalizeTokenizedScoreSpectrumExamples(examples)
	if err != nil {
		return EmbeddingScoreSpectrumEvalMetrics{}, err
	}
	forward := t.prepareForwardWeights()
	t.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	var aggregate EmbeddingScoreSpectrumEvalMetrics
	for start := 0; start < len(canonicalExamples); start += batchSize {
		end := start + batchSize
		if end > len(canonicalExamples) {
			end = len(canonicalExamples)
		}
		chunkMetrics, err := t.evaluateScoreSpectrumCanonicalBatch(canonicalExamples[start:end], forward)
		if err != nil {
			return EmbeddingScoreSpectrumEvalMetrics{}, fmt.Errorf("score-spectrum eval rows %d-%d: %w", start, end-1, err)
		}
		mergeScoreSpectrumEvalMetrics(&aggregate, chunkMetrics)
	}
	normalizeScoreSpectrumEvalMetrics(&aggregate)
	return aggregate, nil
}

func (t *EmbeddingTrainer) evaluateScoreSpectrumCanonicalBatch(canonicalExamples []EmbeddingScoreSpectrumExample, forward *embeddingForwardWeights) (EmbeddingScoreSpectrumEvalMetrics, error) {
	if len(canonicalExamples) == 0 {
		return EmbeddingScoreSpectrumEvalMetrics{}, fmt.Errorf("score-spectrum eval batch is empty")
	}
	queryInputs, candidateInputs, candidateSpans, err := scoreSpectrumSequenceInputs(canonicalExamples, t.queryRoleIndex(), t.documentRoleIndex())
	if err != nil {
		return EmbeddingScoreSpectrumEvalMetrics{}, err
	}
	allInputs := make([]embeddingSequenceInput, 0, len(queryInputs)+len(candidateInputs))
	allInputs = append(allInputs, queryInputs...)
	allInputs = append(allInputs, candidateInputs...)
	encoded, err := t.encodeSequenceInputs(allInputs, forward, false)
	if err != nil {
		return EmbeddingScoreSpectrumEvalMetrics{}, err
	}
	defer t.releaseEncodedSequences(encoded)
	queries := encoded[:len(queryInputs)]
	candidates := encoded[len(queryInputs):]
	return evaluateScoreSpectrumEncodings(queries, candidates, candidateSpans, canonicalExamples, t.config.Temperature)
}

// EvaluateListwiseGeometry scores tokenized listwise geometry batches without
// applying optimizer updates or consulting train-policy gates.
func (t *EmbeddingTrainer) EvaluateListwiseGeometry(batches []EmbeddingTokenizedListwiseGeometryBatch) (EmbeddingListwiseGeometryEvalMetrics, error) {
	return t.EvaluateListwiseGeometryBatched(batches, len(batches))
}

// EvaluateListwiseGeometryBatched scores tokenized listwise geometry batches in
// bounded row chunks without applying optimizer updates or consulting
// train-policy gates.
func (t *EmbeddingTrainer) EvaluateListwiseGeometryBatched(batches []EmbeddingTokenizedListwiseGeometryBatch, batchSize int) (EmbeddingListwiseGeometryEvalMetrics, error) {
	if t == nil {
		return EmbeddingListwiseGeometryEvalMetrics{}, fmt.Errorf("embedding trainer is not initialized")
	}
	if len(batches) == 0 {
		return EmbeddingListwiseGeometryEvalMetrics{}, fmt.Errorf("listwise geometry eval dataset is empty")
	}
	if batchSize <= 0 || batchSize > len(batches) {
		batchSize = len(batches)
	}
	if err := validateListwiseGeometryTrainerConfig(t.config); err != nil {
		return EmbeddingListwiseGeometryEvalMetrics{}, err
	}
	forward := t.prepareForwardWeights()
	t.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	var aggregate EmbeddingListwiseGeometryEvalMetrics
	for start := 0; start < len(batches); start += batchSize {
		end := start + batchSize
		if end > len(batches) {
			end = len(batches)
		}
		chunkMetrics, err := t.evaluateListwiseGeometryBatch(batches[start:end], forward)
		if err != nil {
			return EmbeddingListwiseGeometryEvalMetrics{}, fmt.Errorf("listwise geometry eval batches %d-%d: %w", start, end-1, err)
		}
		mergeListwiseGeometryEvalMetrics(&aggregate, chunkMetrics)
	}
	normalizeListwiseGeometryEvalMetrics(&aggregate)
	return aggregate, nil
}

func (t *EmbeddingTrainer) evaluateListwiseGeometryBatch(batches []EmbeddingTokenizedListwiseGeometryBatch, forward *embeddingForwardWeights) (EmbeddingListwiseGeometryEvalMetrics, error) {
	if len(batches) == 0 {
		return EmbeddingListwiseGeometryEvalMetrics{}, fmt.Errorf("listwise geometry eval batch is empty")
	}
	queryInputs, documentInputs, spans, err := listwiseGeometrySequenceInputs(batches, t.queryRoleIndex(), t.documentRoleIndex())
	if err != nil {
		return EmbeddingListwiseGeometryEvalMetrics{}, err
	}
	allInputs := make([]embeddingSequenceInput, 0, len(queryInputs)+len(documentInputs))
	allInputs = append(allInputs, queryInputs...)
	allInputs = append(allInputs, documentInputs...)
	encoded, err := t.encodeSequenceInputs(allInputs, forward, false)
	if err != nil {
		return EmbeddingListwiseGeometryEvalMetrics{}, err
	}
	defer t.releaseEncodedSequences(encoded)
	queries := encoded[:len(queryInputs)]
	documents := encoded[len(queryInputs):]
	return evaluateListwiseGeometryEncodings(queries, documents, spans, batches, t.config.Temperature)
}

func mergeListwiseGeometryEvalMetrics(dst *EmbeddingListwiseGeometryEvalMetrics, chunk EmbeddingListwiseGeometryEvalMetrics) {
	dst.Loss += chunk.Loss * float32(chunk.QueryCount)
	dst.AverageScore += chunk.AverageScore * float32(chunk.DocumentCellCount)
	dst.TeacherCrossEntropy += chunk.TeacherCrossEntropy * float32(chunk.QueryCount)
	dst.TeacherKL += chunk.TeacherKL * float32(chunk.QueryCount)
	dst.TeacherTop1Agreement += chunk.TeacherTop1Agreement * float32(chunk.QueryCount)
	dst.AnyPositiveTop1 += chunk.AnyPositiveTop1 * float32(chunk.AnyPositiveQueryCount)
	dst.QueryCount += chunk.QueryCount
	dst.DocumentCellCount += chunk.DocumentCellCount
	dst.BatchCount += chunk.BatchCount
	dst.AnyPositiveQueryCount += chunk.AnyPositiveQueryCount
}

func normalizeListwiseGeometryEvalMetrics(metrics *EmbeddingListwiseGeometryEvalMetrics) {
	if metrics.QueryCount > 0 {
		invQueries := float32(1) / float32(metrics.QueryCount)
		metrics.Loss *= invQueries
		metrics.TeacherCrossEntropy *= invQueries
		metrics.TeacherKL *= invQueries
		metrics.TeacherTop1Agreement *= invQueries
	}
	if metrics.DocumentCellCount > 0 {
		metrics.AverageScore /= float32(metrics.DocumentCellCount)
	}
	if metrics.AnyPositiveQueryCount > 0 {
		metrics.AnyPositiveTop1 /= float32(metrics.AnyPositiveQueryCount)
	}
}

func mergeScoreSpectrumEvalMetrics(dst *EmbeddingScoreSpectrumEvalMetrics, chunk EmbeddingScoreSpectrumEvalMetrics) {
	dst.Loss += chunk.Loss * float32(chunk.RowCount)
	dst.AverageScore += chunk.AverageScore * float32(chunk.CandidateCount)
	dst.AnyPositiveTop1 += chunk.AnyPositiveTop1 * float32(chunk.AnyPositiveRowCount)
	dst.OriginalPositiveTop1 += chunk.OriginalPositiveTop1 * float32(chunk.OriginalPositiveRowCount)
	dst.AlternateRelevantRecovery += chunk.AlternateRelevantRecovery * float32(chunk.AlternateRecoveryRowCount)
	dst.BestPositiveHardestNegativeMargin += chunk.BestPositiveHardestNegativeMargin * float32(chunk.MarginRowCount)
	dst.TargetCrossEntropy += chunk.TargetCrossEntropy * float32(chunk.TargetDistributionRowCount)
	dst.TargetKL += chunk.TargetKL * float32(chunk.TargetDistributionRowCount)
	dst.RowCount += chunk.RowCount
	dst.CandidateCount += chunk.CandidateCount
	dst.AnyPositiveRowCount += chunk.AnyPositiveRowCount
	dst.OriginalPositiveRowCount += chunk.OriginalPositiveRowCount
	dst.AlternateRecoveryRowCount += chunk.AlternateRecoveryRowCount
	dst.MarginRowCount += chunk.MarginRowCount
	dst.TargetDistributionRowCount += chunk.TargetDistributionRowCount
}

func normalizeScoreSpectrumEvalMetrics(metrics *EmbeddingScoreSpectrumEvalMetrics) {
	if metrics.RowCount > 0 {
		metrics.Loss /= float32(metrics.RowCount)
	}
	if metrics.CandidateCount > 0 {
		metrics.AverageScore /= float32(metrics.CandidateCount)
	}
	if metrics.AnyPositiveRowCount > 0 {
		metrics.AnyPositiveTop1 /= float32(metrics.AnyPositiveRowCount)
	}
	if metrics.OriginalPositiveRowCount > 0 {
		metrics.OriginalPositiveTop1 /= float32(metrics.OriginalPositiveRowCount)
	}
	if metrics.AlternateRecoveryRowCount > 0 {
		metrics.AlternateRelevantRecovery /= float32(metrics.AlternateRecoveryRowCount)
	}
	if metrics.MarginRowCount > 0 {
		metrics.BestPositiveHardestNegativeMargin /= float32(metrics.MarginRowCount)
	}
	if metrics.TargetDistributionRowCount > 0 {
		metrics.TargetCrossEntropy /= float32(metrics.TargetDistributionRowCount)
		metrics.TargetKL /= float32(metrics.TargetDistributionRowCount)
	}
}

// ExportInferenceWeights returns runtime-loadable weights in the module's declared dtypes.
func (t *EmbeddingTrainer) ExportInferenceWeights() (map[string]*backend.Tensor, error) {
	if t == nil {
		return nil, fmt.Errorf("embedding trainer is not initialized")
	}
	if t.isCompactTrainer() {
		return t.exportCompactInferenceWeights()
	}
	out := map[string]*backend.Tensor{}
	token, err := exportTensorForParam(t.tokenParam, t.tokenEmbed)
	if err != nil {
		return nil, err
	}
	if t.roleEmbed != nil {
		role, err := exportTensorForParam(t.roleParam, t.roleEmbed)
		if err != nil {
			return nil, err
		}
		out[t.roleParam.Name] = role
	}
	if t.attentionQuery != nil {
		query, err := exportTensorForParam(t.attnQParam, t.attentionQuery)
		if err != nil {
			return nil, err
		}
		key, err := exportTensorForParam(t.attnKParam, t.attentionKey)
		if err != nil {
			return nil, err
		}
		value, err := exportTensorForParam(t.attnVParam, t.attentionValue)
		if err != nil {
			return nil, err
		}
		output, err := exportTensorForParam(t.attnOParam, t.attentionOutput)
		if err != nil {
			return nil, err
		}
		out[t.attnQParam.Name] = query
		out[t.attnKParam.Name] = key
		out[t.attnVParam.Name] = value
		out[t.attnOParam.Name] = output
	}
	if t.hiddenProjection != nil {
		hidden, err := exportTensorForParam(t.hiddenParam, t.hiddenProjection)
		if err != nil {
			return nil, err
		}
		out[t.hiddenParam.Name] = hidden
	}
	proj, err := exportTensorForParam(t.projParam, t.projection)
	if err != nil {
		return nil, err
	}
	out[t.tokenParam.Name] = token
	out[t.projParam.Name] = proj
	return out, nil
}

func (t *EmbeddingTrainer) exportCompactInferenceWeights() (map[string]*backend.Tensor, error) {
	if t == nil || t.compactState == nil {
		return nil, fmt.Errorf("compact embedding trainer is not initialized")
	}
	out := map[string]*backend.Tensor{}
	add := func(item CompactEmbeddingTrainTensor) error {
		if item.Name == "" || item.Tensor == nil {
			return nil
		}
		param, err := requireModuleTensorParam(t.module, item.Name)
		if err != nil {
			return err
		}
		tensor, err := exportTensorForParam(param, item.Tensor)
		if err != nil {
			return err
		}
		out[item.Name] = tensor
		return nil
	}
	if err := add(t.compactState.TokenEmbedding); err != nil {
		return nil, err
	}
	if t.compactState.RoleEmbedding != nil {
		if err := add(*t.compactState.RoleEmbedding); err != nil {
			return nil, err
		}
	}
	for _, layer := range t.compactState.Layers {
		for _, item := range []CompactEmbeddingTrainTensor{
			layer.AttentionQuery,
			layer.AttentionKey,
			layer.AttentionValue,
			layer.AttentionOutput,
			layer.FFNUp,
			layer.FFNDown,
		} {
			if err := add(item); err != nil {
				return nil, err
			}
		}
	}
	if t.compactState.OutputProjection != nil {
		if err := add(*t.compactState.OutputProjection); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func requireModuleTensorParam(mod *eosartifact.Module, name string) (eosartifact.Param, error) {
	if mod == nil {
		return eosartifact.Param{}, fmt.Errorf("nil module")
	}
	for _, param := range mod.Params {
		if param.Name != name {
			continue
		}
		if param.Type.Kind != eosartifact.ValueTensor || param.Type.Tensor == nil {
			return eosartifact.Param{}, fmt.Errorf("param %q is not a tensor weight", name)
		}
		return param, nil
	}
	return eosartifact.Param{}, fmt.Errorf("module is missing tensor param %q", name)
}

// ExportLoadOptions adapts the current trained weights to runtime load options.
func (t *EmbeddingTrainer) ExportLoadOptions() ([]LoadOption, error) {
	weights, err := t.ExportInferenceWeights()
	if err != nil {
		return nil, err
	}
	return NewWeightFile(weights).LoadOptions(), nil
}

// ExportWeightFile builds a serialized weight file for package loading.
func (t *EmbeddingTrainer) ExportWeightFile() (WeightFile, error) {
	weights, err := t.ExportInferenceWeights()
	if err != nil {
		return WeightFile{}, err
	}
	return NewWeightFile(weights), nil
}

// RenameEmbeddingModel updates the module and embedding manifest identity before rewriting a package.
func (t *EmbeddingTrainer) RenameEmbeddingModel(name string) error {
	if t == nil {
		return fmt.Errorf("embedding trainer is not initialized")
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return fmt.Errorf("embedding model name is required")
	}
	if t.module != nil {
		t.module.Name = name
	}
	t.manifest.Name = name
	return nil
}

// SetEmbeddingTokenizerMaxSequence updates the embedding tokenizer contract before rewriting a package.
func (t *EmbeddingTrainer) SetEmbeddingTokenizerMaxSequence(maxSeq int) error {
	if t == nil {
		return fmt.Errorf("embedding trainer is not initialized")
	}
	if maxSeq < 0 {
		return fmt.Errorf("embedding tokenizer max sequence must be non-negative")
	}
	if maxSeq == 0 {
		return nil
	}
	t.manifest.Tokenizer.MaxSequence = maxSeq
	return nil
}

// WriteEmbeddingPackage writes a packaged embedding model to sibling artifact, manifest, and weight files.
func (t *EmbeddingTrainer) WriteEmbeddingPackage(artifactPath string) (EmbeddingPackagePaths, error) {
	if t == nil {
		return EmbeddingPackagePaths{}, fmt.Errorf("embedding trainer is not initialized")
	}
	if err := eosartifact.WriteFile(artifactPath, t.module); err != nil {
		return EmbeddingPackagePaths{}, err
	}
	manifestPath := DefaultEmbeddingManifestPath(artifactPath)
	if err := t.manifest.WriteFile(manifestPath); err != nil {
		return EmbeddingPackagePaths{}, err
	}
	weightFile, err := t.ExportWeightFile()
	if err != nil {
		return EmbeddingPackagePaths{}, err
	}
	weightPath := DefaultWeightFilePath(artifactPath)
	if err := weightFile.WriteFile(weightPath); err != nil {
		return EmbeddingPackagePaths{}, err
	}
	memoryPlan := NewMemoryPlan(t.module, weightFile.Weights, MemoryPlanOptions{})
	memoryPlanPath := DefaultMemoryPlanPath(artifactPath)
	if err := memoryPlan.WriteFile(memoryPlanPath); err != nil {
		return EmbeddingPackagePaths{}, err
	}
	t.memoryPlan = cloneMemoryPlan(&memoryPlan)
	packageFiles := map[string]string{
		"artifact":           artifactPath,
		"embedding_manifest": manifestPath,
		"weights":            weightPath,
		"memory_plan":        memoryPlanPath,
	}
	tokenizerPath := DefaultTokenizerPath(artifactPath)
	if _, err := os.Stat(tokenizerPath); err == nil {
		packageFiles["tokenizer"] = tokenizerPath
	}
	packageManifest, err := BuildPackageManifest(PackageEmbedding, t.module, packageFiles)
	if err != nil {
		return EmbeddingPackagePaths{}, err
	}
	packageManifest.ScoreSpectrum = packageScoreSpectrumPolicy(t.scoreSpectrumLineage)
	packageManifest.ListwiseGeometry = packageListwiseGeometryPolicy(t.listwiseGeometryLineage)
	packageManifestPath := DefaultPackageManifestPath(artifactPath)
	if err := packageManifest.WriteFile(packageManifestPath); err != nil {
		return EmbeddingPackagePaths{}, err
	}
	return EmbeddingPackagePaths{
		ArtifactPath:        artifactPath,
		ManifestPath:        manifestPath,
		TokenizerPath:       tokenizerPath,
		WeightFilePath:      weightPath,
		MemoryPlanPath:      memoryPlanPath,
		PackageManifestPath: packageManifestPath,
	}, nil
}

// WriteTrainingPackage writes a packaged native-training embedder including trainer config and checkpoint state.
func (t *EmbeddingTrainer) WriteTrainingPackage(artifactPath string) (EmbeddingTrainPackagePaths, error) {
	if t == nil {
		return EmbeddingTrainPackagePaths{}, fmt.Errorf("embedding trainer is not initialized")
	}
	if err := eosartifact.WriteFile(artifactPath, t.module); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	embeddingManifestPath := DefaultEmbeddingManifestPath(artifactPath)
	if err := t.manifest.WriteFile(embeddingManifestPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	weightFile, err := t.ExportWeightFile()
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	weightPath := DefaultWeightFilePath(artifactPath)
	if err := weightFile.WriteFile(weightPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	memoryPlan := NewMemoryPlan(t.module, weightFile.Weights, MemoryPlanOptions{})
	memoryPlanPath := DefaultMemoryPlanPath(artifactPath)
	if err := memoryPlan.WriteFile(memoryPlanPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	t.memoryPlan = cloneMemoryPlan(&memoryPlan)
	trainManifest := EmbeddingTrainManifest{
		Name:             t.manifest.Name,
		Embedding:        t.manifest,
		Config:           t.config,
		ScoreSpectrum:    t.scoreSpectrumLineage,
		ListwiseGeometry: t.listwiseGeometryLineage,
	}
	trainManifestPath := DefaultEmbeddingTrainManifestPath(artifactPath)
	if err := trainManifest.WriteFile(trainManifestPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	checkpoint, err := t.Checkpoint()
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	checkpointPath := DefaultEmbeddingCheckpointPath(artifactPath)
	if err := checkpoint.WriteFile(checkpointPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	trainProfile := t.TrainProfile()
	trainProfilePath := DefaultEmbeddingTrainProfilePath(artifactPath)
	if err := trainProfile.WriteFile(trainProfilePath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	packageFiles := map[string]string{
		"artifact":           artifactPath,
		"embedding_manifest": embeddingManifestPath,
		"weights":            weightPath,
		"memory_plan":        memoryPlanPath,
		"train_manifest":     trainManifestPath,
		"checkpoint":         checkpointPath,
		"train_profile":      trainProfilePath,
	}
	tokenizerPath := DefaultTokenizerPath(artifactPath)
	if _, err := os.Stat(tokenizerPath); err == nil {
		packageFiles["tokenizer"] = tokenizerPath
	}
	packageManifest, err := BuildPackageManifest(PackageTraining, t.module, packageFiles)
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	packageManifest.ScoreSpectrum = packageScoreSpectrumPolicy(t.scoreSpectrumLineage)
	packageManifest.ListwiseGeometry = packageListwiseGeometryPolicy(t.listwiseGeometryLineage)
	packageManifestPath := DefaultPackageManifestPath(artifactPath)
	if err := packageManifest.WriteFile(packageManifestPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	return EmbeddingTrainPackagePaths{
		ArtifactPath:          artifactPath,
		EmbeddingManifestPath: embeddingManifestPath,
		TokenizerPath:         tokenizerPath,
		WeightFilePath:        weightPath,
		MemoryPlanPath:        memoryPlanPath,
		TrainManifestPath:     trainManifestPath,
		CheckpointPath:        checkpointPath,
		TrainProfilePath:      trainProfilePath,
		PackageManifestPath:   packageManifestPath,
	}, nil
}

func (t *EmbeddingTrainer) runBatch(batch []EmbeddingPairExample, update bool) (EmbeddingTrainMetrics, error) {
	if t == nil {
		return EmbeddingTrainMetrics{}, fmt.Errorf("embedding trainer is not initialized")
	}
	if len(batch) == 0 {
		return EmbeddingTrainMetrics{}, fmt.Errorf("training batch is empty")
	}
	forward := t.prepareForwardWeights()
	t.primeForwardWeightResidency(forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj)
	if t.isCompactTrainer() {
		if update {
			return t.runCompactPairBatchUpdate(batch, forward)
		}
		totalLoss := float32(0)
		totalScore := float32(0)
		for i, example := range batch {
			score, loss, err := t.scoreExamplePair(example, forward)
			if err != nil {
				return EmbeddingTrainMetrics{}, fmt.Errorf("batch %d: %w", i, err)
			}
			totalLoss += loss
			totalScore += score
		}
		batchScale := float32(1) / float32(len(batch))
		return EmbeddingTrainMetrics{
			Loss:         totalLoss * batchScale,
			AverageScore: totalScore * batchScale,
			BatchSize:    len(batch),
		}, nil
	}
	if metrics, ok, err := t.tryRunPairBatchBatched(batch, forward, update); ok || err != nil {
		return metrics, err
	}
	gradToken := make([]float32, len(t.tokenEmbed.F32))
	gradRole := make([]float32, tensorDataLen(t.roleEmbed))
	gradAttnQ := make([]float32, tensorDataLen(t.attentionQuery))
	gradAttnK := make([]float32, tensorDataLen(t.attentionKey))
	gradAttnV := make([]float32, tensorDataLen(t.attentionValue))
	gradAttnO := make([]float32, tensorDataLen(t.attentionOutput))
	gradHidden := make([]float32, len(t.hiddenProjectionData()))
	gradProj := make([]float32, len(t.projection.F32))

	totalLoss := float32(0)
	totalScore := float32(0)
	for i, example := range batch {
		left, right, err := t.encodeExamplePair(example, forward.token, forward.role, forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, update)
		if err != nil {
			return EmbeddingTrainMetrics{}, fmt.Errorf("batch %d: %w", i, err)
		}
		score, gradLeft, gradRight := cosineGrad(left.pooled, right.pooled)
		scale := score - example.Target
		totalLoss += 0.5 * scale * scale
		totalScore += score
		for j := range gradLeft {
			gradLeft[j] *= scale
			gradRight[j] *= scale
		}
		leftInputGrad := t.backpropEncodedSequence(left, gradLeft, forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
		rightInputGrad := t.backpropEncodedSequence(right, gradRight, forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
		t.accumulateInputTokenGrad(left.tokens, leftInputGrad, gradToken)
		t.accumulateInputTokenGrad(right.tokens, rightInputGrad, gradToken)
		t.accumulateInputRoleGrad(left.role, len(left.tokens), leftInputGrad, gradRole)
		t.accumulateInputRoleGrad(right.role, len(right.tokens), rightInputGrad, gradRole)
		t.releaseEncodedSequenceBindings(left)
		t.releaseEncodedSequenceBindings(right)
	}

	batchScale := float32(1) / float32(len(batch))
	if update {
		t.step++
		t.applyOptimizerUpdate(t.tokenParam.Name, t.tokenEmbed, t.tokenMom1, t.tokenMom2, gradToken, batchScale)
		t.applyOptimizerUpdate(t.roleParam.Name, t.roleEmbed, t.roleMom1, t.roleMom2, gradRole, batchScale)
		t.applyOptimizerUpdate(t.attnQParam.Name, t.attentionQuery, t.attnQMom1, t.attnQMom2, gradAttnQ, batchScale)
		t.applyOptimizerUpdate(t.attnKParam.Name, t.attentionKey, t.attnKMom1, t.attnKMom2, gradAttnK, batchScale)
		t.applyOptimizerUpdate(t.attnVParam.Name, t.attentionValue, t.attnVMom1, t.attnVMom2, gradAttnV, batchScale)
		t.applyOptimizerUpdate(t.attnOParam.Name, t.attentionOutput, t.attnOMom1, t.attnOMom2, gradAttnO, batchScale)
		t.applyOptimizerUpdate(t.hiddenParam.Name, t.hiddenProjection, t.hiddenMom1, t.hiddenMom2, gradHidden, batchScale)
		t.applyOptimizerUpdate(t.projParam.Name, t.projection, t.projMom1, t.projMom2, gradProj, batchScale)
	}
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * batchScale,
		AverageScore: totalScore * batchScale,
		BatchSize:    len(batch),
	}, nil
}

func (t *EmbeddingTrainer) runCompactPairBatchUpdate(batch []EmbeddingPairExample, forward *embeddingForwardWeights) (EmbeddingTrainMetrics, error) {
	if t == nil || t.compactState == nil {
		return EmbeddingTrainMetrics{}, fmt.Errorf("compact embedding trainer is not initialized")
	}
	if forward == nil || forward.compact == nil {
		return EmbeddingTrainMetrics{}, fmt.Errorf("missing compact forward weights")
	}
	grads := newCompactEmbeddingGradState(t.compactState)
	totalLoss := float32(0)
	totalScore := float32(0)
	for i, example := range batch {
		left, right, err := t.encodeExamplePair(example, nil, nil, nil, nil, nil, nil, nil, nil, false)
		if err != nil {
			return EmbeddingTrainMetrics{}, fmt.Errorf("batch %d: %w", i, err)
		}
		score, gradLeft, gradRight := cosineGrad(left.pooled, right.pooled)
		scale := score - example.Target
		totalLoss += 0.5 * scale * scale
		totalScore += score
		for j := range gradLeft {
			gradLeft[j] *= scale
			gradRight[j] *= scale
		}
		if err := t.backpropCompactEncodedSequence(left, gradLeft, forward.compact, grads); err != nil {
			return EmbeddingTrainMetrics{}, fmt.Errorf("batch %d left: %w", i, err)
		}
		if err := t.backpropCompactEncodedSequence(right, gradRight, forward.compact, grads); err != nil {
			return EmbeddingTrainMetrics{}, fmt.Errorf("batch %d right: %w", i, err)
		}
	}

	batchScale := float32(1) / float32(len(batch))
	t.applyCompactOptimizerUpdates(grads, batchScale)
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * batchScale,
		AverageScore: totalScore * batchScale,
		BatchSize:    len(batch),
	}, nil
}

func newCompactEmbeddingGradState(state *CompactEmbeddingTrainState) *compactEmbeddingGradState {
	if state == nil {
		return &compactEmbeddingGradState{}
	}
	grads := &compactEmbeddingGradState{
		token:  make([]float32, tensorDataLen(state.TokenEmbedding.Tensor)),
		layers: make([]compactEmbeddingGradLayer, len(state.Layers)),
	}
	if state.RoleEmbedding != nil {
		grads.role = make([]float32, tensorDataLen(state.RoleEmbedding.Tensor))
	}
	for i, layer := range state.Layers {
		grads.layers[i] = compactEmbeddingGradLayer{
			attnQ:   make([]float32, tensorDataLen(layer.AttentionQuery.Tensor)),
			attnK:   make([]float32, tensorDataLen(layer.AttentionKey.Tensor)),
			attnV:   make([]float32, tensorDataLen(layer.AttentionValue.Tensor)),
			attnO:   make([]float32, tensorDataLen(layer.AttentionOutput.Tensor)),
			ffnUp:   make([]float32, tensorDataLen(layer.FFNUp.Tensor)),
			ffnDown: make([]float32, tensorDataLen(layer.FFNDown.Tensor)),
		}
	}
	if state.OutputProjection != nil {
		grads.outputProjection = make([]float32, tensorDataLen(state.OutputProjection.Tensor))
	}
	return grads
}

func compactEmbeddingGradSlices(grads *compactEmbeddingGradState) [][]float32 {
	if grads == nil {
		return nil
	}
	slices := [][]float32{grads.token}
	if len(grads.role) > 0 {
		slices = append(slices, grads.role)
	}
	for i := range grads.layers {
		layer := grads.layers[i]
		slices = append(slices, layer.attnQ, layer.attnK, layer.attnV, layer.attnO, layer.ffnUp, layer.ffnDown)
	}
	if len(grads.outputProjection) > 0 {
		slices = append(slices, grads.outputProjection)
	}
	return slices
}

func snapshotCompactEmbeddingTrainTensors(state *CompactEmbeddingTrainState) []embeddingTrainTensorSnapshot {
	if state == nil {
		return nil
	}
	tensors := []*backend.Tensor{state.TokenEmbedding.Tensor}
	if state.RoleEmbedding != nil {
		tensors = append(tensors, state.RoleEmbedding.Tensor)
	}
	for i := range state.Layers {
		layer := state.Layers[i]
		tensors = append(
			tensors,
			layer.AttentionQuery.Tensor,
			layer.AttentionKey.Tensor,
			layer.AttentionValue.Tensor,
			layer.AttentionOutput.Tensor,
			layer.FFNUp.Tensor,
			layer.FFNDown.Tensor,
		)
	}
	if state.OutputProjection != nil {
		tensors = append(tensors, state.OutputProjection.Tensor)
	}
	return snapshotEmbeddingTrainTensors(tensors...)
}

func (t *EmbeddingTrainer) backpropCompactEncodedSequence(seq *embeddingEncodedSequence, gradPooled []float32, forward *compactEmbeddingForwardWeights, grads *compactEmbeddingGradState) error {
	if seq == nil || len(seq.layers) == 0 {
		return fmt.Errorf("compact encoded sequence has no layers")
	}
	if forward == nil || len(forward.layers) != len(seq.layers) {
		return fmt.Errorf("compact forward layer count %d does not match encoded layer count %d", compactForwardLayerCount(forward), len(seq.layers))
	}
	gradHidden, err := t.backpropCompactFinalOutput(seq, gradPooled, forward.outputProjection, grads.outputProjection)
	if err != nil {
		return err
	}
	for layerIndex := len(seq.layers) - 1; layerIndex >= 0; layerIndex-- {
		gradHidden = t.backpropCompactLayer(seq.layers[layerIndex], gradHidden, forward.layers[layerIndex], &grads.layers[layerIndex])
	}
	t.accumulateCompactInputGrads(seq.tokens, seq.role, gradHidden, grads)
	return nil
}

func compactForwardLayerCount(forward *compactEmbeddingForwardWeights) int {
	if forward == nil {
		return 0
	}
	return len(forward.layers)
}

func (t *EmbeddingTrainer) backpropCompactFinalOutput(seq *embeddingEncodedSequence, gradPooled []float32, outputProjection *backend.Tensor, gradOutputProjection []float32) ([]float32, error) {
	last := seq.finalLayer()
	if last == nil || len(last.tokens) == 0 {
		return nil, fmt.Errorf("compact final output requires a non-empty final layer")
	}
	seqLen := len(last.tokens)
	if len(last.projected)%seqLen != 0 {
		return nil, fmt.Errorf("compact final hidden size %d does not divide sequence length %d", len(last.projected), seqLen)
	}
	modelDim := len(last.projected) / seqLen
	outDim := modelDim
	if outputProjection != nil {
		if len(outputProjection.Shape) != 2 || outputProjection.Shape[0] != modelDim {
			return nil, fmt.Errorf("output_projection shape %v, want [%d O]", outputProjection.Shape, modelDim)
		}
		outDim = outputProjection.Shape[1]
	}
	if len(gradPooled) != outDim {
		return nil, fmt.Errorf("compact pooled gradient width %d does not match output width %d", len(gradPooled), outDim)
	}
	if last.activeCount <= 0 {
		return nil, fmt.Errorf("compact final output has zero active rows")
	}

	normalized := make([]float32, len(last.projected))
	norms := make([]float32, seqLen)
	for row := 0; row < seqLen; row++ {
		base := row * modelDim
		norm := vectorNorm(last.projected[base : base+modelDim])
		norms[row] = norm
		if norm == 0 {
			continue
		}
		for col := 0; col < modelDim; col++ {
			normalized[base+col] = last.projected[base+col] / norm
		}
	}

	gradOutputRows := make([]float32, seqLen*outDim)
	invActive := 1 / float32(last.activeCount)
	for row := 0; row < seqLen; row++ {
		if last.mask[row] == 0 {
			continue
		}
		base := row * outDim
		for col := 0; col < outDim; col++ {
			gradOutputRows[base+col] = gradPooled[col] * invActive
		}
	}

	gradNormalized := gradOutputRows
	if outputProjection != nil {
		gradProjStep := make([]float32, modelDim*outDim)
		if out, ok := t.tryTrainerMatMul(normalized, seqLen, modelDim, gradOutputRows, seqLen, outDim, true, false); ok {
			copy(gradProjStep, out)
		} else {
			fillHostMatMulTranspose(normalized, seqLen, modelDim, gradOutputRows, seqLen, outDim, true, false, gradProjStep)
		}
		addFloat32Slice(gradOutputProjection, gradProjStep)
		gradNormalized = make([]float32, seqLen*modelDim)
		if out, ok := t.tryTrainerMatMul(gradOutputRows, seqLen, outDim, outputProjection.F32, modelDim, outDim, false, true); ok {
			copy(gradNormalized, out)
		} else {
			fillHostMatMulTranspose(gradOutputRows, seqLen, outDim, outputProjection.F32, modelDim, outDim, false, true, gradNormalized)
		}
	}

	gradHidden := make([]float32, seqLen*modelDim)
	for row := 0; row < seqLen; row++ {
		norm := norms[row]
		if norm == 0 {
			continue
		}
		base := row * modelDim
		dotNG := float32(0)
		for col := 0; col < modelDim; col++ {
			dotNG += normalized[base+col] * gradNormalized[base+col]
		}
		for col := 0; col < modelDim; col++ {
			gradHidden[base+col] = (gradNormalized[base+col] - normalized[base+col]*dotNG) / norm
		}
	}
	return gradHidden, nil
}

func (t *EmbeddingTrainer) backpropCompactLayer(state *embeddingSequenceState, gradProjected []float32, layer compactEmbeddingForwardLayer, gradLayer *compactEmbeddingGradLayer) []float32 {
	if state == nil || len(state.tokens) == 0 {
		return nil
	}
	seqLen := len(state.tokens)
	d := compactLayerModelDim(layer)
	h := compactLayerFFNDim(layer)
	if d <= 0 || h <= 0 || len(gradProjected) != seqLen*d {
		return make([]float32, len(state.input))
	}
	heads, headDim, ok := compactLayerAttentionLayout(layer, d)
	if !ok {
		return make([]float32, len(state.input))
	}
	if len(state.attnScores) != heads*seqLen*seqLen {
		return make([]float32, len(state.input))
	}

	gradFFNResidual := make([]float32, seqLen*d)
	if out, ok := t.tryLayerNormBackwardRows(gradProjected, state.projected, state.ffnResidual, seqLen, d, state.projectedBinding, state.ffnResidualBinding); ok {
		copy(gradFFNResidual, out)
	} else {
		for row := 0; row < seqLen; row++ {
			base := row * d
			backwardLayerNormRow(
				gradFFNResidual[base:base+d],
				gradProjected[base:base+d],
				state.projected[base:base+d],
				state.ffnResidual[base:base+d],
			)
		}
	}
	gradFFNOutput := gradFFNResidual
	gradHidden := append([]float32(nil), gradFFNResidual...)

	gradFFNDownStep := make([]float32, h*d)
	if out, ok := t.tryTrainerMatMul(state.activated, seqLen, h, gradFFNOutput, seqLen, d, true, false); ok {
		copy(gradFFNDownStep, out)
	} else {
		fillHostMatMulTranspose(state.activated, seqLen, h, gradFFNOutput, seqLen, d, true, false, gradFFNDownStep)
	}
	addFloat32Slice(gradLayer.ffnDown, gradFFNDownStep)
	gradActivatedPre := make([]float32, seqLen*h)
	if out, ok := t.tryTrainerMatMul(gradFFNOutput, seqLen, d, forwardMatMulHostData(layer.ffnDown), h, d, false, true); ok {
		copy(gradActivatedPre, out)
	} else {
		fillHostMatMulTranspose(gradFFNOutput, seqLen, d, forwardMatMulHostData(layer.ffnDown), h, d, false, true, gradActivatedPre)
	}
	gradActivated := make([]float32, seqLen*h)
	fastGELU := fastGELUEnabled()
	usedGELUAccel := false
	if !fastGELU {
		if out, ok := t.tryGELUBackwardMul(gradActivatedPre, state.ffnHidden, seqLen, h, state.ffnHiddenBinding); ok {
			copy(gradActivated, out)
			usedGELUAccel = true
		}
	}
	if !usedGELUAccel {
		for row := 0; row < seqLen; row++ {
			base := row * h
			fillGELUBackwardMul(gradActivated[base:base+h], gradActivatedPre[base:base+h], state.ffnHidden[base:base+h], fastGELU)
		}
	}
	gradFFNUpStep := make([]float32, d*h)
	if out, ok := t.tryTrainerMatMul(state.hidden, seqLen, d, gradActivated, seqLen, h, true, false); ok {
		copy(gradFFNUpStep, out)
	} else {
		fillHostMatMulTranspose(state.hidden, seqLen, d, gradActivated, seqLen, h, true, false, gradFFNUpStep)
	}
	addFloat32Slice(gradLayer.ffnUp, gradFFNUpStep)
	gradHiddenFromFFN := make([]float32, seqLen*d)
	if out, ok := t.tryTrainerMatMul(gradActivated, seqLen, h, forwardMatMulHostData(layer.ffnUp), d, h, false, true); ok {
		copy(gradHiddenFromFFN, out)
	} else {
		fillHostMatMulTranspose(gradActivated, seqLen, h, forwardMatMulHostData(layer.ffnUp), d, h, false, true, gradHiddenFromFFN)
	}
	addFloat32Slice(gradHidden, gradHiddenFromFFN)

	gradAttnResidual := make([]float32, seqLen*d)
	if out, ok := t.tryLayerNormBackwardRows(gradHidden, state.hidden, state.attnResidual, seqLen, d, state.hiddenBinding, state.attnResidualBinding); ok {
		copy(gradAttnResidual, out)
	} else {
		for row := 0; row < seqLen; row++ {
			base := row * d
			backwardLayerNormRow(
				gradAttnResidual[base:base+d],
				gradHidden[base:base+d],
				state.hidden[base:base+d],
				state.attnResidual[base:base+d],
			)
		}
	}
	gradAttnOutput := gradAttnResidual
	gradInput := append([]float32(nil), gradAttnResidual...)

	gradAttnOStep := make([]float32, d*d)
	if out, ok := t.tryTrainerMatMul(state.attnMixed, seqLen, d, gradAttnOutput, seqLen, d, true, false); ok {
		copy(gradAttnOStep, out)
	} else {
		fillHostMatMulTranspose(state.attnMixed, seqLen, d, gradAttnOutput, seqLen, d, true, false, gradAttnOStep)
	}
	addFloat32Slice(gradLayer.attnO, gradAttnOStep)
	gradMixed := make([]float32, seqLen*d)
	if out, ok := t.tryTrainerMatMul(gradAttnOutput, seqLen, d, forwardMatMulHostData(layer.attnO), d, d, false, true); ok {
		copy(gradMixed, out)
	} else {
		fillHostMatMulTranspose(gradAttnOutput, seqLen, d, forwardMatMulHostData(layer.attnO), d, d, false, true, gradMixed)
	}

	gradV := make([]float32, seqLen*d)
	gradQ := make([]float32, seqLen*d)
	gradK := make([]float32, seqLen*d)
	scale := float32(1 / math.Sqrt(float64(headDim)))
	for head := 0; head < heads; head++ {
		headOffset := head * headDim
		scoreBase := head * seqLen * seqLen
		probs := state.attnScores[scoreBase : scoreBase+seqLen*seqLen]
		gradScores := make([]float32, seqLen*seqLen)
		for query := 0; query < seqLen; query++ {
			queryGradBase := query*d + headOffset
			scoreRowBase := query * seqLen
			for key := 0; key < seqLen; key++ {
				keyValueBase := key*d + headOffset
				sum := float32(0)
				for col := 0; col < headDim; col++ {
					sum += gradMixed[queryGradBase+col] * state.attnV[keyValueBase+col]
				}
				gradScores[scoreRowBase+key] = sum
			}
		}
		gradPreSoftmax := make([]float32, seqLen*seqLen)
		if out, ok := t.trySoftmaxBackwardRows(gradScores, probs, seqLen, seqLen, state.attnScoresBinding); ok {
			copy(gradPreSoftmax, out)
		} else {
			for row := 0; row < seqLen; row++ {
				base := row * seqLen
				backwardSoftmaxRow(gradPreSoftmax[base:base+seqLen], gradScores[base:base+seqLen], probs[base:base+seqLen])
			}
		}
		scaleFloat32Slice(gradPreSoftmax, scale)
		for query := 0; query < seqLen; query++ {
			queryBase := query*d + headOffset
			scoreRowBase := query * seqLen
			for key := 0; key < seqLen; key++ {
				keyBase := key*d + headOffset
				prob := probs[scoreRowBase+key]
				preGrad := gradPreSoftmax[scoreRowBase+key]
				for col := 0; col < headDim; col++ {
					gradV[keyBase+col] += prob * gradMixed[queryBase+col]
					gradQ[queryBase+col] += preGrad * state.attnK[keyBase+col]
					gradK[keyBase+col] += preGrad * state.attnQ[queryBase+col]
				}
			}
		}
	}

	gradAttnQStep := make([]float32, d*d)
	if out, ok := t.tryTrainerMatMul(state.input, seqLen, d, gradQ, seqLen, d, true, false); ok {
		copy(gradAttnQStep, out)
	} else {
		fillHostMatMulTranspose(state.input, seqLen, d, gradQ, seqLen, d, true, false, gradAttnQStep)
	}
	addFloat32Slice(gradLayer.attnQ, gradAttnQStep)
	gradAttnKStep := make([]float32, d*d)
	if out, ok := t.tryTrainerMatMul(state.input, seqLen, d, gradK, seqLen, d, true, false); ok {
		copy(gradAttnKStep, out)
	} else {
		fillHostMatMulTranspose(state.input, seqLen, d, gradK, seqLen, d, true, false, gradAttnKStep)
	}
	addFloat32Slice(gradLayer.attnK, gradAttnKStep)
	gradAttnVStep := make([]float32, d*d)
	if out, ok := t.tryTrainerMatMul(state.input, seqLen, d, gradV, seqLen, d, true, false); ok {
		copy(gradAttnVStep, out)
	} else {
		fillHostMatMulTranspose(state.input, seqLen, d, gradV, seqLen, d, true, false, gradAttnVStep)
	}
	addFloat32Slice(gradLayer.attnV, gradAttnVStep)

	gradInputStep := make([]float32, seqLen*d)
	if out, ok := t.tryTrainerMatMul(gradQ, seqLen, d, forwardMatMulHostData(layer.attnQ), d, d, false, true); ok {
		copy(gradInputStep, out)
	} else {
		fillHostMatMulTranspose(gradQ, seqLen, d, forwardMatMulHostData(layer.attnQ), d, d, false, true, gradInputStep)
	}
	addFloat32Slice(gradInput, gradInputStep)
	for i := range gradInputStep {
		gradInputStep[i] = 0
	}
	if out, ok := t.tryTrainerMatMul(gradK, seqLen, d, forwardMatMulHostData(layer.attnK), d, d, false, true); ok {
		copy(gradInputStep, out)
	} else {
		fillHostMatMulTranspose(gradK, seqLen, d, forwardMatMulHostData(layer.attnK), d, d, false, true, gradInputStep)
	}
	addFloat32Slice(gradInput, gradInputStep)
	for i := range gradInputStep {
		gradInputStep[i] = 0
	}
	if out, ok := t.tryTrainerMatMul(gradV, seqLen, d, forwardMatMulHostData(layer.attnV), d, d, false, true); ok {
		copy(gradInputStep, out)
	} else {
		fillHostMatMulTranspose(gradV, seqLen, d, forwardMatMulHostData(layer.attnV), d, d, false, true, gradInputStep)
	}
	addFloat32Slice(gradInput, gradInputStep)
	return gradInput
}

func (t *EmbeddingTrainer) accumulateCompactInputGrads(tokens []int32, role int32, gradInput []float32, grads *compactEmbeddingGradState) {
	if t == nil || t.compactState == nil || grads == nil || t.compactState.TokenEmbedding.Tensor == nil || len(t.compactState.TokenEmbedding.Tensor.Shape) != 2 {
		return
	}
	d := t.compactState.TokenEmbedding.Tensor.Shape[1]
	vocab := t.compactState.TokenEmbedding.Tensor.Shape[0]
	if d == 0 || vocab == 0 || len(gradInput) != len(tokens)*d {
		return
	}
	if t.manifest.PositionEncoding == EmbeddingPositionEncodingRoPE {
		gradInput = append([]float32(nil), gradInput...)
		applyRoPETransposeToRowsInPlace(gradInput, len(tokens), d)
	}
	accumulateTokenGrad(tokens, gradInput, grads.token, d, vocab)
	if t.compactState.RoleEmbedding == nil || t.compactState.RoleEmbedding.Tensor == nil || len(grads.role) == 0 || len(t.compactState.RoleEmbedding.Tensor.Shape) != 2 {
		return
	}
	rows := t.compactState.RoleEmbedding.Tensor.Shape[0]
	if role < 0 || int(role) >= rows {
		return
	}
	roleBase := int(role) * d
	if roleBase+d > len(grads.role) {
		return
	}
	for row := range tokens {
		inputBase := row * d
		for col := 0; col < d; col++ {
			grads.role[roleBase+col] += gradInput[inputBase+col]
		}
	}
}

func (t *EmbeddingTrainer) applyCompactOptimizerUpdates(grads *compactEmbeddingGradState, scale float32) {
	if t == nil || t.compactState == nil || grads == nil {
		return
	}
	t.step++
	t.compactState.Step = t.step
	t.compactOptimizerUpdates++
	apply := func(item *CompactEmbeddingTrainTensor, grad []float32) {
		if item == nil || item.Tensor == nil {
			return
		}
		if len(grad) != len(item.Tensor.F32) {
			return
		}
		if item.Moment1 == nil {
			item.Moment1 = zeroLikeMaster(item.Tensor)
		}
		if item.Moment2 == nil {
			item.Moment2 = zeroLikeMaster(item.Tensor)
		}
		t.applyOptimizerUpdate(item.Name, item.Tensor, item.Moment1, item.Moment2, grad, scale)
	}
	apply(&t.compactState.TokenEmbedding, grads.token)
	if t.compactState.RoleEmbedding != nil {
		apply(t.compactState.RoleEmbedding, grads.role)
	}
	for i := range t.compactState.Layers {
		layer := &t.compactState.Layers[i]
		gradLayer := grads.layers[i]
		apply(&layer.AttentionQuery, gradLayer.attnQ)
		apply(&layer.AttentionKey, gradLayer.attnK)
		apply(&layer.AttentionValue, gradLayer.attnV)
		apply(&layer.AttentionOutput, gradLayer.attnO)
		apply(&layer.FFNUp, gradLayer.ffnUp)
		apply(&layer.FFNDown, gradLayer.ffnDown)
	}
	if t.compactState.OutputProjection != nil {
		apply(t.compactState.OutputProjection, grads.outputProjection)
	}
	t.compactState.Config = t.config
}

func (t *EmbeddingTrainer) tryRunPairBatchBatched(batch []EmbeddingPairExample, forward *embeddingForwardWeights, update bool) (EmbeddingTrainMetrics, bool, error) {
	if t.isCompactTrainer() {
		return EmbeddingTrainMetrics{}, false, nil
	}
	if t == nil || forward == nil || len(batch) == 0 {
		return EmbeddingTrainMetrics{}, false, nil
	}
	if update {
		if !batchedPairwiseTrainEnabled() {
			return EmbeddingTrainMetrics{}, false, nil
		}
	} else if !batchedPairwiseEvalEnabled() {
		return EmbeddingTrainMetrics{}, false, nil
	}
	lefts, rights, ok, err := t.tryEncodePairBatchBatchedForward(batch, forward, update)
	if err != nil || !ok {
		return EmbeddingTrainMetrics{}, ok, err
	}
	defer t.releaseEncodedSequences(lefts)
	defer t.releaseEncodedSequences(rights)

	gradToken := make([]float32, len(t.tokenEmbed.F32))
	gradRole := make([]float32, tensorDataLen(t.roleEmbed))
	gradAttnQ := make([]float32, tensorDataLen(t.attentionQuery))
	gradAttnK := make([]float32, tensorDataLen(t.attentionKey))
	gradAttnV := make([]float32, tensorDataLen(t.attentionValue))
	gradAttnO := make([]float32, tensorDataLen(t.attentionOutput))
	gradHidden := make([]float32, len(t.hiddenProjectionData()))
	gradProj := make([]float32, len(t.projection.F32))
	leftGrads := make([][]float32, len(lefts))
	rightGrads := make([][]float32, len(rights))

	totalLoss := float32(0)
	totalScore := float32(0)
	for i, example := range batch {
		if i >= len(lefts) || i >= len(rights) || lefts[i] == nil || rights[i] == nil {
			return EmbeddingTrainMetrics{}, true, fmt.Errorf("batch %d: encoder produced nil pair", i)
		}
		score, gradLeft, gradRight := cosineGrad(lefts[i].pooled, rights[i].pooled)
		scale := score - example.Target
		totalLoss += 0.5 * scale * scale
		totalScore += score
		for j := range gradLeft {
			gradLeft[j] *= scale
			gradRight[j] *= scale
		}
		leftGrads[i] = gradLeft
		rightGrads[i] = gradRight
	}

	if update {
		if !t.tryBackpropContrastiveBatch(
			lefts,
			rights,
			leftGrads,
			rightGrads,
			forward.attnQ,
			forward.attnK,
			forward.attnV,
			forward.attnO,
			forward.hidden,
			forward.proj,
			gradToken,
			gradAttnQ,
			gradAttnK,
			gradAttnV,
			gradAttnO,
			gradHidden,
			gradProj,
		) {
			for i, left := range lefts {
				inputGrad := t.backpropEncodedSequence(left, leftGrads[i], forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
				t.accumulateInputTokenGrad(left.tokens, inputGrad, gradToken)
				t.accumulateInputRoleGrad(left.role, len(left.tokens), inputGrad, gradRole)
			}
			for i, right := range rights {
				inputGrad := t.backpropEncodedSequence(right, rightGrads[i], forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
				t.accumulateInputTokenGrad(right.tokens, inputGrad, gradToken)
				t.accumulateInputRoleGrad(right.role, len(right.tokens), inputGrad, gradRole)
			}
		}

		batchScale := float32(1) / float32(len(batch))
		t.step++
		t.applyOptimizerUpdate(t.tokenParam.Name, t.tokenEmbed, t.tokenMom1, t.tokenMom2, gradToken, batchScale)
		t.applyOptimizerUpdate(t.roleParam.Name, t.roleEmbed, t.roleMom1, t.roleMom2, gradRole, batchScale)
		t.applyOptimizerUpdate(t.attnQParam.Name, t.attentionQuery, t.attnQMom1, t.attnQMom2, gradAttnQ, batchScale)
		t.applyOptimizerUpdate(t.attnKParam.Name, t.attentionKey, t.attnKMom1, t.attnKMom2, gradAttnK, batchScale)
		t.applyOptimizerUpdate(t.attnVParam.Name, t.attentionValue, t.attnVMom1, t.attnVMom2, gradAttnV, batchScale)
		t.applyOptimizerUpdate(t.attnOParam.Name, t.attentionOutput, t.attnOMom1, t.attnOMom2, gradAttnO, batchScale)
		t.applyOptimizerUpdate(t.hiddenParam.Name, t.hiddenProjection, t.hiddenMom1, t.hiddenMom2, gradHidden, batchScale)
		t.applyOptimizerUpdate(t.projParam.Name, t.projection, t.projMom1, t.projMom2, gradProj, batchScale)
	}

	batchScale := float32(1) / float32(len(batch))
	return EmbeddingTrainMetrics{
		Loss:         totalLoss * batchScale,
		AverageScore: totalScore * batchScale,
		BatchSize:    len(batch),
	}, true, nil
}

func (t *EmbeddingTrainer) scoreExamplePair(example EmbeddingPairExample, forward *embeddingForwardWeights) (float32, float32, error) {
	if forward == nil {
		return 0, 0, fmt.Errorf("missing forward weights")
	}
	left, right, err := t.encodeExamplePair(example, forward.token, forward.role, forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, false)
	if err != nil {
		return 0, 0, err
	}
	defer t.releaseEncodedSequenceBindings(left)
	defer t.releaseEncodedSequenceBindings(right)
	score, _, _ := cosineGrad(left.pooled, right.pooled)
	scale := score - example.Target
	return score, 0.5 * scale * scale, nil
}

func (t *EmbeddingTrainer) encodeContrastiveBatch(batch []EmbeddingContrastiveExample, forward *embeddingForwardWeights, captureBindings bool) ([]*embeddingEncodedSequence, []*embeddingEncodedSequence, error) {
	if forward == nil {
		return nil, nil, fmt.Errorf("missing forward weights")
	}
	if queries, positives, ok, err := t.tryEncodeContrastiveBatchBatchedForward(batch, forward, captureBindings); ok || err != nil {
		return queries, positives, err
	}
	queries := make([]*embeddingEncodedSequence, 0, len(batch))
	positives := make([]*embeddingEncodedSequence, 0, len(batch))
	for i, example := range batch {
		queryMask, err := t.prepareMask(example.QueryTokens, example.QueryMask)
		if err != nil {
			t.releaseEncodedSequences(queries)
			t.releaseEncodedSequences(positives)
			return nil, nil, fmt.Errorf("batch %d query: %w", i, err)
		}
		positiveMask, err := t.prepareMask(example.PositiveTokens, example.PositiveMask)
		if err != nil {
			t.releaseEncodedSequences(queries)
			t.releaseEncodedSequences(positives)
			return nil, nil, fmt.Errorf("batch %d positive: %w", i, err)
		}
		query, err := t.encodeSequence(example.QueryTokens, queryMask, t.queryRoleIndex(), forward.token, forward.role, forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, captureBindings)
		if err != nil {
			t.releaseEncodedSequences(queries)
			t.releaseEncodedSequences(positives)
			return nil, nil, fmt.Errorf("batch %d query: %w", i, err)
		}
		positive, err := t.encodeSequence(example.PositiveTokens, positiveMask, t.documentRoleIndex(), forward.token, forward.role, forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, captureBindings)
		if err != nil {
			t.releaseEncodedSequenceBindings(query)
			t.releaseEncodedSequences(queries)
			t.releaseEncodedSequences(positives)
			return nil, nil, fmt.Errorf("batch %d positive: %w", i, err)
		}
		queries = append(queries, query)
		positives = append(positives, positive)
	}
	return queries, positives, nil
}

type embeddingBatchSequence struct {
	tokens  []int32
	mask    []int32
	role    int32
	current []float32
	encoded *embeddingEncodedSequence
}

type embeddingBatchSequenceSlot struct {
	sequence *embeddingBatchSequence
}

type embeddingSequenceInput struct {
	tokens []int32
	mask   []int32
	role   int32
	label  string
}

func embeddingBatchSequenceKey(tokens, mask []int32, role int32) string {
	var b strings.Builder
	b.Grow((len(tokens)+len(mask))*4 + 5)
	writeInt32KeyPart(&b, tokens)
	b.WriteByte('|')
	writeInt32KeyPart(&b, mask)
	b.WriteByte('|')
	writeInt32KeyPart(&b, []int32{role})
	return b.String()
}

func writeInt32KeyPart(b *strings.Builder, values []int32) {
	for _, value := range values {
		b.WriteByte(byte(value))
		b.WriteByte(byte(value >> 8))
		b.WriteByte(byte(value >> 16))
		b.WriteByte(byte(value >> 24))
	}
}

func (t *EmbeddingTrainer) cachedEmbeddingBatchSequence(cache map[string]*embeddingBatchSequence, sequences *[]*embeddingBatchSequence, groups map[int][]embeddingBatchSequenceSlot, lengths *[]int, tokens, mask []int32, role int32, forward *embeddingForwardWeights) (*embeddingBatchSequence, error) {
	key := embeddingBatchSequenceKey(tokens, mask, role)
	if sequence, ok := cache[key]; ok {
		return sequence, nil
	}
	sequence, err := t.newEmbeddingBatchSequence(tokens, mask, role, forward)
	if err != nil {
		return nil, err
	}
	cache[key] = sequence
	*sequences = append(*sequences, sequence)
	seqLen := len(sequence.tokens)
	if _, ok := groups[seqLen]; !ok {
		*lengths = append(*lengths, seqLen)
	}
	groups[seqLen] = append(groups[seqLen], embeddingBatchSequenceSlot{sequence: sequence})
	return sequence, nil
}

func (t *EmbeddingTrainer) encodeSequenceInputs(inputs []embeddingSequenceInput, forward *embeddingForwardWeights, captureBindings bool) ([]*embeddingEncodedSequence, error) {
	if forward == nil {
		return nil, fmt.Errorf("missing forward weights")
	}
	if seqs, ok, err := t.tryEncodeSequenceInputsBatchedForward(inputs, forward, captureBindings); ok || err != nil {
		return seqs, err
	}
	out := make([]*embeddingEncodedSequence, 0, len(inputs))
	for i, input := range inputs {
		mask, err := t.prepareMask(input.tokens, input.mask)
		if err != nil {
			t.releaseEncodedSequences(out)
			return nil, fmt.Errorf("%s: %w", embeddingSequenceInputLabel(input, i), err)
		}
		seq, err := t.encodeSequence(input.tokens, mask, input.role, forward.token, forward.role, forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, captureBindings)
		if err != nil {
			t.releaseEncodedSequences(out)
			return nil, fmt.Errorf("%s: %w", embeddingSequenceInputLabel(input, i), err)
		}
		out = append(out, seq)
	}
	return out, nil
}

func (t *EmbeddingTrainer) tryEncodeSequenceInputsBatchedForward(inputs []embeddingSequenceInput, forward *embeddingForwardWeights, captureBindings bool) ([]*embeddingEncodedSequence, bool, error) {
	if t.isCompactTrainer() {
		return nil, false, nil
	}
	if t == nil || t.forwardMatMul == nil || forward == nil || len(inputs) == 0 || !batchedContrastiveForwardEnabled() {
		return nil, false, nil
	}
	sequences := make([]*embeddingBatchSequence, 0, len(inputs))
	out := make([]*embeddingEncodedSequence, len(inputs))
	groups := map[int][]embeddingBatchSequenceSlot{}
	lengths := make([]int, 0)
	sequenceCache := map[string]*embeddingBatchSequence{}
	for i, input := range inputs {
		mask, err := t.prepareMask(input.tokens, input.mask)
		if err != nil {
			return nil, true, fmt.Errorf("%s: %w", embeddingSequenceInputLabel(input, i), err)
		}
		seq, err := t.cachedEmbeddingBatchSequence(sequenceCache, &sequences, groups, &lengths, input.tokens, mask, input.role, forward)
		if err != nil {
			t.releaseBatchEncodedSequences(sequences)
			return nil, true, fmt.Errorf("%s: %w", embeddingSequenceInputLabel(input, i), err)
		}
		out[i] = seq.encoded
	}
	if len(sequences) == 0 {
		return nil, false, nil
	}
	if err := t.encodeBatchSequencesByLength(sequences, groups, lengths, forward, captureBindings); err != nil {
		t.releaseBatchEncodedSequences(sequences)
		return nil, true, err
	}
	return out, true, nil
}

func embeddingSequenceInputLabel(input embeddingSequenceInput, index int) string {
	if input.label != "" {
		return input.label
	}
	return fmt.Sprintf("sequence %d", index)
}

func (t *EmbeddingTrainer) tryEncodeContrastiveBatchBatchedForward(batch []EmbeddingContrastiveExample, forward *embeddingForwardWeights, captureBindings bool) ([]*embeddingEncodedSequence, []*embeddingEncodedSequence, bool, error) {
	if t.isCompactTrainer() {
		return nil, nil, false, nil
	}
	if t == nil || t.forwardMatMul == nil || forward == nil || len(batch) == 0 || !batchedContrastiveForwardEnabled() {
		return nil, nil, false, nil
	}
	sequences := make([]*embeddingBatchSequence, 0, len(batch)*2)
	queries := make([]*embeddingEncodedSequence, len(batch))
	positives := make([]*embeddingEncodedSequence, len(batch))
	groups := map[int][]embeddingBatchSequenceSlot{}
	lengths := make([]int, 0)
	sequenceCache := map[string]*embeddingBatchSequence{}
	for i, example := range batch {
		queryMask, err := t.prepareMask(example.QueryTokens, example.QueryMask)
		if err != nil {
			return nil, nil, true, fmt.Errorf("batch %d query: %w", i, err)
		}
		positiveMask, err := t.prepareMask(example.PositiveTokens, example.PositiveMask)
		if err != nil {
			return nil, nil, true, fmt.Errorf("batch %d positive: %w", i, err)
		}
		query, err := t.cachedEmbeddingBatchSequence(sequenceCache, &sequences, groups, &lengths, example.QueryTokens, queryMask, t.queryRoleIndex(), forward)
		if err != nil {
			t.releaseBatchEncodedSequences(sequences)
			return nil, nil, true, fmt.Errorf("batch %d query: %w", i, err)
		}
		positive, err := t.cachedEmbeddingBatchSequence(sequenceCache, &sequences, groups, &lengths, example.PositiveTokens, positiveMask, t.documentRoleIndex(), forward)
		if err != nil {
			t.releaseBatchEncodedSequences(sequences)
			return nil, nil, true, fmt.Errorf("batch %d positive: %w", i, err)
		}
		queries[i] = query.encoded
		positives[i] = positive.encoded
	}
	if len(sequences) == 0 {
		return nil, nil, false, nil
	}
	if err := t.encodeBatchSequencesByLength(sequences, groups, lengths, forward, captureBindings); err != nil {
		t.releaseBatchEncodedSequences(sequences)
		return nil, nil, true, err
	}
	return queries, positives, true, nil
}

func (t *EmbeddingTrainer) tryEncodePairBatchBatchedForward(batch []EmbeddingPairExample, forward *embeddingForwardWeights, captureBindings bool) ([]*embeddingEncodedSequence, []*embeddingEncodedSequence, bool, error) {
	if t.isCompactTrainer() {
		return nil, nil, false, nil
	}
	if t == nil || t.forwardMatMul == nil || forward == nil || len(batch) == 0 || !batchedContrastiveForwardEnabled() {
		return nil, nil, false, nil
	}
	sequences := make([]*embeddingBatchSequence, 0, len(batch)*2)
	lefts := make([]*embeddingEncodedSequence, len(batch))
	rights := make([]*embeddingEncodedSequence, len(batch))
	groups := map[int][]embeddingBatchSequenceSlot{}
	lengths := make([]int, 0)
	sequenceCache := map[string]*embeddingBatchSequence{}
	for i, example := range batch {
		leftMask, err := t.prepareMask(example.LeftTokens, example.LeftMask)
		if err != nil {
			return nil, nil, true, fmt.Errorf("batch %d left: %w", i, err)
		}
		rightMask, err := t.prepareMask(example.RightTokens, example.RightMask)
		if err != nil {
			return nil, nil, true, fmt.Errorf("batch %d right: %w", i, err)
		}
		left, err := t.cachedEmbeddingBatchSequence(sequenceCache, &sequences, groups, &lengths, example.LeftTokens, leftMask, t.rawRoleIndex(), forward)
		if err != nil {
			t.releaseBatchEncodedSequences(sequences)
			return nil, nil, true, fmt.Errorf("batch %d left: %w", i, err)
		}
		right, err := t.cachedEmbeddingBatchSequence(sequenceCache, &sequences, groups, &lengths, example.RightTokens, rightMask, t.rawRoleIndex(), forward)
		if err != nil {
			t.releaseBatchEncodedSequences(sequences)
			return nil, nil, true, fmt.Errorf("batch %d right: %w", i, err)
		}
		lefts[i] = left.encoded
		rights[i] = right.encoded
	}
	if len(sequences) == 0 {
		return nil, nil, false, nil
	}
	if err := t.encodeBatchSequencesByLength(sequences, groups, lengths, forward, captureBindings); err != nil {
		t.releaseBatchEncodedSequences(sequences)
		return nil, nil, true, err
	}
	return lefts, rights, true, nil
}

func (t *EmbeddingTrainer) encodeBatchSequencesByLength(sequences []*embeddingBatchSequence, groups map[int][]embeddingBatchSequenceSlot, lengths []int, forward *embeddingForwardWeights, captureBindings bool) error {
	if len(sequences) == 0 {
		return nil
	}
	if forward == nil {
		return fmt.Errorf("missing forward weights")
	}
	for layer := 0; layer < t.encoderRepeats(); layer++ {
		for _, seqLen := range lengths {
			slots := groups[seqLen]
			states := make([]*embeddingSequenceState, len(slots))
			for i, slot := range slots {
				state, err := newEmbeddingSequenceState(slot.sequence.tokens, slot.sequence.mask, slot.sequence.current, forward.hidden, forward.proj)
				if err != nil {
					for _, state := range states {
						t.releaseSequenceBindings(state)
					}
					return err
				}
				if captureBindings {
					d := stateWidth(forward.hidden, forward.proj)
					state.inputBinding = t.bindSequenceTensor(state, "input", tensorF32View([]int{len(state.tokens), d}, state.input), true, false)
				}
				states[i] = state
			}
			if err := t.encodeBatchedLayerStates(states, forward.attnQ, forward.attnK, forward.attnV, forward.attnO, forward.hidden, forward.proj, captureBindings); err != nil {
				for _, state := range states {
					t.releaseSequenceBindings(state)
				}
				return err
			}
			for i, state := range states {
				sequence := slots[i].sequence
				sequence.encoded.layers = append(sequence.encoded.layers, state)
				sequence.current = state.projected
			}
		}
	}
	for _, sequence := range sequences {
		if len(sequence.encoded.layers) == 0 {
			return fmt.Errorf("encoder produced zero layers")
		}
		sequence.encoded.pooled = append([]float32(nil), sequence.encoded.layers[len(sequence.encoded.layers)-1].pooled...)
	}
	return nil
}

func batchedContrastiveForwardEnabled() bool {
	if trainEnvFlagEnabled("EOS_TRAIN_DISABLE_BATCHED_FORWARD") {
		return false
	}
	switch trainEnv("EOS_TRAIN_BATCHED_FORWARD") {
	case "0", "false", "FALSE", "no", "NO":
		return false
	default:
		return true
	}
}

func batchedPairwiseEvalEnabled() bool {
	if !batchedContrastiveForwardEnabled() || trainEnvFlagEnabled("EOS_TRAIN_DISABLE_BATCHED_PAIR_EVAL") {
		return false
	}
	switch trainEnv("EOS_TRAIN_BATCHED_PAIR_EVAL") {
	case "0", "false", "FALSE", "no", "NO":
		return false
	default:
		return true
	}
}

func batchedPairwiseTrainEnabled() bool {
	if !batchedContrastiveForwardEnabled() || trainEnvFlagEnabled("EOS_TRAIN_DISABLE_BATCHED_PAIR_TRAIN") {
		return false
	}
	switch trainEnv("EOS_TRAIN_BATCHED_PAIR_TRAIN") {
	case "0", "false", "FALSE", "no", "NO":
		return false
	default:
		return true
	}
}

func pairwiseEvalBatchSize(total int) int {
	if total <= 0 {
		return 0
	}
	size := 512
	if raw := trainEnv("EOS_TRAIN_PAIR_EVAL_BATCH_SIZE"); raw != "" {
		var parsed int
		if _, err := fmt.Sscanf(raw, "%d", &parsed); err == nil && parsed > 0 {
			size = parsed
		}
	}
	if size > total {
		size = total
	}
	return size
}

func sequenceMatMulBindingsEnabled() bool {
	if trainEnvFlagEnabled("EOS_TRAIN_DISABLE_SEQUENCE_MATMUL_BINDINGS") {
		return false
	}
	return trainEnvFlagEnabled("EOS_TRAIN_ENABLE_SEQUENCE_MATMUL_BINDINGS")
}

func qkvMultiBoundRightEnabled() bool {
	if trainEnvFlagEnabled("EOS_TRAIN_DISABLE_QKV_MULTI_BOUND") {
		return false
	}
	switch trainEnv("EOS_TRAIN_QKV_MULTI_BOUND") {
	case "0", "false", "FALSE", "no", "NO":
		return false
	default:
		return true
	}
}

func sharedLeftMatMulEnabled() bool {
	if trainEnvFlagEnabled("EOS_TRAIN_DISABLE_SHARED_LEFT_MATMUL") {
		return false
	}
	switch trainEnv("EOS_TRAIN_SHARED_LEFT_MATMUL") {
	case "0", "false", "FALSE", "no", "NO":
		return false
	default:
		return true
	}
}

func concatenatedSharedLeftMatMulEnabled() bool {
	if !sharedLeftMatMulEnabled() || trainEnvFlagEnabled("EOS_TRAIN_DISABLE_CONCAT_SHARED_LEFT_MATMUL") {
		return false
	}
	switch trainEnv("EOS_TRAIN_CONCAT_SHARED_LEFT_MATMUL") {
	case "0", "false", "FALSE", "no", "NO":
		return false
	default:
		return true
	}
}

func combinedAttentionVKGradMatMulEnabled() bool {
	if trainEnvFlagEnabled("EOS_TRAIN_DISABLE_COMBINED_ATTENTION_VK_GRAD") {
		return false
	}
	switch trainEnv("EOS_TRAIN_COMBINED_ATTENTION_VK_GRAD") {
	case "0", "false", "FALSE", "no", "NO":
		return false
	default:
		return true
	}
}

func accumulatedAttentionInputGradMatMulEnabled() bool {
	if trainEnvFlagEnabled("EOS_TRAIN_DISABLE_ACCUMULATED_ATTENTION_INPUT_GRAD") {
		return false
	}
	switch trainEnv("EOS_TRAIN_ACCUMULATED_ATTENTION_INPUT_GRAD") {
	case "0", "false", "FALSE", "no", "NO":
		return false
	default:
		return true
	}
}

func batchedBackwardEnabled() bool {
	if trainEnvFlagEnabled("EOS_TRAIN_DISABLE_BATCHED_BACKWARD") {
		return false
	}
	switch trainEnv("EOS_TRAIN_BATCHED_BACKWARD") {
	case "0", "false", "FALSE", "no", "NO":
		return false
	default:
		return true
	}
}

func fastGELUEnabled() bool {
	return trainEnvFlagEnabled("EOS_TRAIN_ENABLE_FAST_GELU")
}

func activationAccelMaxElements() int {
	limit := 1 << 20
	if raw := trainEnv("EOS_TRAIN_ACTIVATION_ACCEL_MAX_ELEMENTS"); raw != "" {
		var parsed int
		if _, err := fmt.Sscanf(raw, "%d", &parsed); err == nil {
			if parsed <= 0 {
				return 0
			}
			limit = parsed
		}
	}
	return limit
}

func activationAccelElementsAllowed(rows, cols int) bool {
	if rows <= 0 || cols <= 0 {
		return false
	}
	limit := activationAccelMaxElements()
	return limit <= 0 || rows <= limit/cols
}

func trainEnvFlagEnabled(name string) bool {
	switch trainEnv(name) {
	case "1", "true", "TRUE", "yes", "YES":
		return true
	default:
		return false
	}
}

func trainEnv(name string) string {
	if value, ok := os.LookupEnv(name); ok {
		return value
	}
	return ""
}

func (t *EmbeddingTrainer) newEmbeddingBatchSequence(tokens, mask []int32, role int32, forward *embeddingForwardWeights) (*embeddingBatchSequence, error) {
	if forward == nil {
		return nil, fmt.Errorf("missing forward weights")
	}
	input, err := embeddingInputForTokens(forward.token, tokens)
	if err != nil {
		return nil, err
	}
	if err := addRoleEmbeddingToInput(input, forward.role, role, len(tokens)); err != nil {
		return nil, err
	}
	if err := applyEmbeddingPositionEncoding(input, len(tokens), forward.token.Shape[1], t.manifest.PositionEncoding); err != nil {
		return nil, err
	}
	encoded := &embeddingEncodedSequence{
		layers: make([]*embeddingSequenceState, 0, t.encoderRepeats()),
		tokens: append([]int32(nil), tokens...),
		role:   role,
	}
	return &embeddingBatchSequence{
		tokens:  append([]int32(nil), tokens...),
		mask:    append([]int32(nil), mask...),
		role:    role,
		current: input,
		encoded: encoded,
	}, nil
}

func (t *EmbeddingTrainer) releaseBatchEncodedSequences(sequences []*embeddingBatchSequence) {
	for _, sequence := range sequences {
		if sequence != nil {
			t.releaseEncodedSequenceBindings(sequence.encoded)
		}
	}
}

func (t *EmbeddingTrainer) encodeBatchedLayerStates(states []*embeddingSequenceState, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection *backend.Tensor, captureBindings bool) error {
	if len(states) == 0 {
		return nil
	}
	if projection == nil || len(projection.Shape) != 2 {
		return fmt.Errorf("projection must be rank-2")
	}
	d := stateWidth(hiddenProjection, projection)
	h := 0
	if hiddenProjection != nil {
		if len(hiddenProjection.Shape) != 2 {
			return fmt.Errorf("hidden projection must be rank-2")
		}
		h = hiddenProjection.Shape[1]
	}
	e := projection.Shape[1]
	seqLen := len(states[0].tokens)
	bindSoftmaxActivation := t.bindSoftmaxActivationForBatchedForward()
	bindFullActivation := t.bindFullActivationForBatchedForward()
	skipUnboundActAccel := captureBindings && (!bindSoftmaxActivation || !bindFullActivation)
	if seqLen == 0 {
		return fmt.Errorf("tokens are empty")
	}
	for i, state := range states {
		if state == nil {
			return fmt.Errorf("batch state %d is nil", i)
		}
		if len(state.tokens) != seqLen || len(state.mask) != seqLen {
			return fmt.Errorf("batch state %d has inconsistent sequence length", i)
		}
		if len(state.input) != seqLen*d {
			return fmt.Errorf("batch state %d input size %d does not match tokens=%d width=%d", i, len(state.input), seqLen, d)
		}
		state.skipUnboundActAccel = skipUnboundActAccel
	}

	if attentionQuery != nil && attentionKey != nil && attentionValue != nil && attentionOutput != nil {
		if !t.fillBatchedForwardQKVMatMul(states, seqLen, d, attentionQuery, attentionKey, attentionValue) {
			t.fillBatchedForwardWeightMatMul(states, seqLen, d, func(state *embeddingSequenceState) []float32 {
				return state.input
			}, t.attnQParam.Name, attentionQuery, d, func(state *embeddingSequenceState) []float32 {
				return state.attnQ
			}, func(state *embeddingSequenceState, data []float32) {
				state.attnQ = data
			})
			t.fillBatchedForwardWeightMatMul(states, seqLen, d, func(state *embeddingSequenceState) []float32 {
				return state.input
			}, t.attnKParam.Name, attentionKey, d, func(state *embeddingSequenceState) []float32 {
				return state.attnK
			}, func(state *embeddingSequenceState, data []float32) {
				state.attnK = data
			})
			t.fillBatchedForwardWeightMatMul(states, seqLen, d, func(state *embeddingSequenceState) []float32 {
				return state.input
			}, t.attnVParam.Name, attentionValue, d, func(state *embeddingSequenceState) []float32 {
				return state.attnV
			}, func(state *embeddingSequenceState, data []float32) {
				state.attnV = data
			})
		}
		for _, state := range states {
			if captureBindings {
				state.attnQBinding = t.bindSequenceTensor(state, "q", tensorF32View([]int{seqLen, d}, state.attnQ), true, false)
				state.attnKBinding = t.bindSequenceTensor(state, "k", tensorF32View([]int{seqLen, d}, state.attnK), true, false)
				state.attnVBinding = t.bindSequenceTensor(state, "v", tensorF32View([]int{seqLen, d}, state.attnV), true, false)
			}
		}
		batchedScores, batchedScoresOK := t.tryBatchedAttentionScores(states, seqLen, d)
		scoreScale := t.attentionScoreScale(d)
		for i, state := range states {
			if batchedScoresOK {
				state.attnScores = batchedScores[i]
			} else {
				kt := transpose2DData(state.attnK, seqLen, d)
				var (
					scores   []float32
					matmulOK bool
				)
				if captureBindings {
					scores, matmulOK = t.tryTrainerMatMulBoundRight(state.attnQ, seqLen, d, state.attnKBinding, tensorF32View([]int{seqLen, d}, state.attnK), false, true)
				} else {
					scores, matmulOK = t.tryTrainerMatMul(state.attnQ, seqLen, d, state.attnK, seqLen, d, false, true)
				}
				if matmulOK {
					state.attnScores = scores
				} else {
					fillHostMatMul(state.attnQ, seqLen, d, kt, seqLen, state.attnScores)
				}
			}
			scaleFloat32Slice(state.attnScores, scoreScale)
			softmaxAttentionScoresInPlace(state.attnScores, seqLen, state.mask, t.manifest.AttentionMaskMode)
			if captureBindings {
				state.attnScoresBinding = t.bindSequenceTensor(state, "scores", tensorF32View([]int{seqLen, seqLen}, state.attnScores), true, t.softmaxBackwardAccelEnabled() && bindSoftmaxActivation)
			}
		}
		batchedMixed, batchedMixedOK := t.tryBatchedAttentionMixed(states, seqLen, d)
		for i, state := range states {
			if batchedMixedOK {
				state.attnMixed = batchedMixed[i]
			} else {
				var (
					mixed    []float32
					matmulOK bool
				)
				if captureBindings {
					mixed, matmulOK = t.tryTrainerMatMulBoundRight(state.attnScores, seqLen, seqLen, state.attnVBinding, tensorF32View([]int{seqLen, d}, state.attnV), false, false)
				} else {
					mixed, matmulOK = t.tryTrainerMatMul(state.attnScores, seqLen, seqLen, state.attnV, seqLen, d, false, false)
				}
				if matmulOK {
					state.attnMixed = mixed
				} else {
					fillHostMatMul(state.attnScores, seqLen, seqLen, state.attnV, d, state.attnMixed)
				}
			}
			if captureBindings {
				state.attnMixedBinding = t.bindSequenceTensor(state, "mixed", tensorF32View([]int{seqLen, d}, state.attnMixed), true, false)
			}
		}

		t.fillBatchedForwardWeightMatMul(states, seqLen, d, func(state *embeddingSequenceState) []float32 {
			return state.attnMixed
		}, t.attnOParam.Name, attentionOutput, d, func(state *embeddingSequenceState) []float32 {
			return state.attnOutput
		}, func(state *embeddingSequenceState, data []float32) {
			state.attnOutput = data
		})
		for _, state := range states {
			if t.attentionResidualEnabled() || t.attentionLayerNormEnabled() {
				for i := range state.attnOutput {
					value := state.attnOutput[i]
					if t.attentionResidualEnabled() {
						value += state.input[i]
					}
					state.attnResidual[i] = value
				}
				if t.attentionLayerNormEnabled() {
					for row := range state.tokens {
						base := row * d
						layerNormRow(state.hidden[base:base+d], state.attnResidual[base:base+d])
					}
					if captureBindings {
						state.attnResidualBinding = t.bindSequenceTensor(state, "attn_residual", tensorF32View([]int{seqLen, d}, state.attnResidual), false, t.fullActivationBackwardAccelEnabled() && bindFullActivation)
					}
				} else {
					copy(state.hidden, state.attnResidual)
				}
			} else {
				copy(state.hidden, state.attnOutput)
			}
		}
	} else {
		for _, state := range states {
			copy(state.hidden, state.input)
		}
	}

	for _, state := range states {
		if captureBindings {
			state.hiddenBinding = t.bindSequenceTensor(state, "hidden", tensorF32View([]int{seqLen, d}, state.hidden), true, t.fullActivationBackwardAccelEnabled() && t.attentionLayerNormEnabled() && bindFullActivation)
		}
	}

	if hiddenProjection != nil {
		t.fillBatchedForwardWeightMatMul(states, seqLen, d, func(state *embeddingSequenceState) []float32 {
			return state.hidden
		}, t.hiddenParam.Name, hiddenProjection, h, func(state *embeddingSequenceState) []float32 {
			return state.ffnHidden
		}, func(state *embeddingSequenceState, data []float32) {
			state.ffnHidden = data
		})
		for _, state := range states {
			if captureBindings {
				state.ffnHiddenBinding = t.bindSequenceTensor(state, "ffn_hidden", tensorF32View([]int{seqLen, h}, state.ffnHidden), false, t.fullActivationBackwardAccelEnabled() && bindFullActivation)
			}
			fillGELUForward(state.activated, state.ffnHidden, fastGELUEnabled())
			if captureBindings {
				state.activatedBinding = t.bindSequenceTensor(state, "activated", tensorF32View([]int{seqLen, h}, state.activated), true, false)
			}
		}
		t.fillBatchedForwardWeightMatMul(states, seqLen, h, func(state *embeddingSequenceState) []float32 {
			return state.activated
		}, t.projParam.Name, projection, e, func(state *embeddingSequenceState) []float32 {
			return state.ffnOutput
		}, func(state *embeddingSequenceState, data []float32) {
			state.ffnOutput = data
		})
		for _, state := range states {
			if t.ffnResidualEnabled() || t.ffnLayerNormEnabled() {
				for i := range state.ffnOutput {
					value := state.ffnOutput[i]
					if t.ffnResidualEnabled() {
						value += state.hidden[i]
					}
					state.ffnResidual[i] = value
				}
				if t.ffnLayerNormEnabled() {
					for row := range state.tokens {
						base := row * e
						layerNormRow(state.projected[base:base+e], state.ffnResidual[base:base+e])
					}
					if captureBindings {
						state.ffnResidualBinding = t.bindSequenceTensor(state, "ffn_residual", tensorF32View([]int{seqLen, e}, state.ffnResidual), false, t.fullActivationBackwardAccelEnabled() && bindFullActivation)
						state.projectedBinding = t.bindSequenceTensor(state, "projected", tensorF32View([]int{seqLen, e}, state.projected), false, t.fullActivationBackwardAccelEnabled() && bindFullActivation)
					}
				} else {
					copy(state.projected, state.ffnResidual)
				}
			} else {
				copy(state.projected, state.ffnOutput)
			}
		}
	} else {
		t.fillBatchedForwardWeightMatMul(states, seqLen, d, func(state *embeddingSequenceState) []float32 {
			return state.hidden
		}, t.projParam.Name, projection, e, func(state *embeddingSequenceState) []float32 {
			return state.projected
		}, func(state *embeddingSequenceState, data []float32) {
			state.projected = data
		})
	}

	for _, state := range states {
		if err := finalizeEncodedStatePooling(state, e); err != nil {
			return err
		}
	}
	return nil
}

func trainerF32TensorValueType() eosartifact.ValueType {
	return eosartifact.ValueType{
		Kind: eosartifact.ValueTensor,
		Tensor: &eosartifact.TensorType{
			DType: "f32",
		},
	}
}

func (t *EmbeddingTrainer) fillBatchedForwardQKVMatMul(states []*embeddingSequenceState, rows, width int, attentionQuery, attentionKey, attentionValue *backend.Tensor) bool {
	if t == nil || t.forwardMatMul == nil || len(states) == 0 || rows == 0 || width == 0 {
		return false
	}
	if !qkvMultiBoundRightEnabled() {
		return false
	}
	multi, ok := t.forwardMatMul.(backend.MultiBoundRightMatMulAccelerator)
	if !ok {
		return false
	}
	if attentionQuery == nil || attentionKey == nil || attentionValue == nil {
		return false
	}
	if len(attentionQuery.Shape) != 2 || len(attentionKey.Shape) != 2 || len(attentionValue.Shape) != 2 {
		return false
	}
	for _, tensor := range []*backend.Tensor{attentionQuery, attentionKey, attentionValue} {
		if tensor.Shape[0] != width || tensor.Shape[1] != width {
			return false
		}
	}
	perInput := rows * width
	batchedLHS := t.scratchFloat32(3, len(states)*perInput)
	for i, state := range states {
		if state == nil || len(state.input) != perInput || len(state.attnQ) != perInput || len(state.attnK) != perInput || len(state.attnV) != perInput {
			return false
		}
		copy(batchedLHS[i*perInput:(i+1)*perInput], state.input)
	}
	results, err := multi.RunMatMulWithBoundRights(
		tensorF32View([]int{len(states) * rows, width}, batchedLHS),
		[]string{t.attnQParam.Name, t.attnKParam.Name, t.attnVParam.Name},
		trainerF32TensorValueType(),
		false,
		false,
	)
	if err != nil || len(results) != 3 {
		return false
	}
	outs := make([][]float32, len(results))
	for i, result := range results {
		if len(result.Outputs) != 1 || result.Outputs[0] == nil {
			return false
		}
		out := result.Outputs[0].F32
		if len(out) != len(states)*perInput {
			return false
		}
		outs[i] = out
	}
	for i, state := range states {
		state.attnQ = outs[0][i*perInput : (i+1)*perInput]
		state.attnK = outs[1][i*perInput : (i+1)*perInput]
		state.attnV = outs[2][i*perInput : (i+1)*perInput]
	}
	return true
}

func (t *EmbeddingTrainer) fillBatchedForwardWeightMatMul(states []*embeddingSequenceState, rows, inner int, lhs func(*embeddingSequenceState) []float32, rhsName string, rhs *backend.Tensor, cols int, dst func(*embeddingSequenceState) []float32, assign func(*embeddingSequenceState, []float32)) {
	if len(states) == 0 || rhs == nil || rows == 0 || inner == 0 || cols == 0 {
		return
	}
	if t != nil && t.forwardMatMul != nil {
		perInput := rows * inner
		batchedLHS := t.scratchFloat32(3, len(states)*perInput)
		valid := true
		for i, state := range states {
			data := lhs(state)
			if len(data) != perInput {
				valid = false
				break
			}
			copy(batchedLHS[i*perInput:(i+1)*perInput], data)
		}
		if valid {
			if out, ok := t.tryForwardWeightMatMul(batchedLHS, len(states)*rows, inner, rhsName, rhs, cols); ok && len(out) == len(states)*rows*cols {
				for i, state := range states {
					view := out[i*rows*cols : (i+1)*rows*cols]
					if assign != nil {
						assign(state, view)
					} else {
						copy(dst(state), view)
					}
				}
				return
			}
		}
	}
	rhsData := forwardMatMulHostData(rhs)
	for _, state := range states {
		left := lhs(state)
		out := dst(state)
		if len(left) != rows*inner || len(out) != rows*cols {
			continue
		}
		fillHostMatMul(left, rows, inner, rhsData, cols, out)
	}
}

func (t *EmbeddingTrainer) tryBatchedAttentionScores(states []*embeddingSequenceState, seqLen, d int) ([][]float32, bool) {
	if len(states) < 2 || seqLen == 0 || d == 0 {
		return nil, false
	}
	queries := make([][]float32, len(states))
	keys := make([][]float32, len(states))
	for i, state := range states {
		if state == nil || len(state.attnQ) != seqLen*d || len(state.attnK) != seqLen*d {
			return nil, false
		}
		queries[i] = state.attnQ
		keys[i] = state.attnK
	}
	return t.tryTrainerBatchedMatMulTranspose(queries, seqLen, d, keys, seqLen, d, false, true)
}

func (t *EmbeddingTrainer) tryBatchedAttentionMixed(states []*embeddingSequenceState, seqLen, d int) ([][]float32, bool) {
	if len(states) < 2 || seqLen == 0 || d == 0 {
		return nil, false
	}
	scores := make([][]float32, len(states))
	values := make([][]float32, len(states))
	for i, state := range states {
		if state == nil || len(state.attnScores) != seqLen*seqLen || len(state.attnV) != seqLen*d {
			return nil, false
		}
		scores[i] = state.attnScores
		values[i] = state.attnV
	}
	return t.tryTrainerBatchedMatMul(scores, seqLen, seqLen, values, seqLen, d)
}

func finalizeEncodedStatePooling(state *embeddingSequenceState, e int) error {
	if state == nil {
		return fmt.Errorf("encoder state is nil")
	}
	for i := range state.normalized {
		state.normalized[i] = 0
	}
	for i := range state.pooled {
		state.pooled[i] = 0
	}
	state.activeCount = 0
	for row := range state.tokens {
		projectedBase := row * e
		norm := vectorNorm(state.projected[projectedBase : projectedBase+e])
		if norm == 0 {
			copy(state.normalized[projectedBase:projectedBase+e], state.projected[projectedBase:projectedBase+e])
		} else {
			for col := 0; col < e; col++ {
				state.normalized[projectedBase+col] = state.projected[projectedBase+col] / norm
			}
		}
		if state.mask[row] == 0 {
			continue
		}
		state.activeCount++
		for col := 0; col < e; col++ {
			state.pooled[col] += state.normalized[projectedBase+col]
		}
	}
	if state.activeCount == 0 {
		return fmt.Errorf("sequence mask selects zero tokens")
	}
	inv := 1 / float32(state.activeCount)
	for i := range state.pooled {
		state.pooled[i] *= inv
	}
	return nil
}

type contrastivePooledMatrix struct {
	rows  int
	width int
	data  []float32
	norms []float32
}

const (
	TurboQuantPrefixScoreModeReconstructCosine = "reconstruct_cosine"
	TurboQuantPrefixScoreModePreparedIP        = "prepared_ip"
)

func newContrastivePooledMatrix(seqs []*embeddingEncodedSequence) contrastivePooledMatrix {
	matrix := contrastivePooledMatrix{
		rows:  len(seqs),
		norms: make([]float32, len(seqs)),
	}
	if len(seqs) == 0 || seqs[0] == nil || len(seqs[0].pooled) == 0 {
		return matrix
	}
	matrix.width = len(seqs[0].pooled)
	matrix.data = make([]float32, matrix.rows*matrix.width)
	for row, seq := range seqs {
		if seq == nil || len(seq.pooled) != matrix.width {
			continue
		}
		values := matrix.row(row)
		copy(values, seq.pooled)
		matrix.norms[row] = vectorNorm(values)
	}
	return matrix
}

func newContrastivePrefixPooledMatrix(seqs []*embeddingEncodedSequence, dim int) contrastivePooledMatrix {
	matrix := contrastivePooledMatrix{
		rows:  len(seqs),
		norms: make([]float32, len(seqs)),
	}
	if len(seqs) == 0 || dim <= 0 {
		return matrix
	}
	matrix.width = dim
	matrix.data = make([]float32, matrix.rows*matrix.width)
	for row, seq := range seqs {
		if seq == nil || len(seq.pooled) < matrix.width {
			continue
		}
		values := matrix.row(row)
		copy(values, seq.pooled[:matrix.width])
		matrix.norms[row] = vectorNorm(values)
	}
	return matrix
}

func newTurboQuantDequantizedPrefixPooledMatrix(seqs []*embeddingEncodedSequence, dim, bitWidth int, seed int64) contrastivePooledMatrix {
	matrix := contrastivePooledMatrix{
		rows:  len(seqs),
		norms: make([]float32, len(seqs)),
	}
	if len(seqs) == 0 || dim <= 0 {
		return matrix
	}
	matrix.width = dim
	matrix.data = make([]float32, matrix.rows*matrix.width)
	q := turboquant.NewIPWithSeed(dim, bitWidth, seed)
	for row, seq := range seqs {
		if seq == nil || len(seq.pooled) < matrix.width {
			continue
		}
		values := matrix.row(row)
		dequantized := q.Dequantize(q.Quantize(seq.pooled[:matrix.width]))
		copy(values, dequantized)
		matrix.norms[row] = vectorNorm(values)
	}
	return matrix
}

type turboQuantPreparedPrefixMatrix struct {
	rows        int
	width       int
	raw         []float32
	normalized  []float32
	rawNorms    []float32
	quantized   []turboquant.IPQuantized
	dequantized []float32
	prepared    []turboquant.PreparedQuery
}

func newTurboQuantPreparedPrefixMatrix(seqs []*embeddingEncodedSequence, dim, bitWidth int, seed int64, prepareQueries bool) turboQuantPreparedPrefixMatrix {
	matrix := turboQuantPreparedPrefixMatrix{
		rows:     len(seqs),
		rawNorms: make([]float32, len(seqs)),
	}
	if len(seqs) == 0 || dim <= 0 {
		return matrix
	}
	matrix.width = dim
	matrix.raw = make([]float32, matrix.rows*matrix.width)
	matrix.normalized = make([]float32, matrix.rows*matrix.width)
	q := turboquant.NewIPWithSeed(dim, bitWidth, seed)
	if prepareQueries {
		matrix.prepared = make([]turboquant.PreparedQuery, matrix.rows)
	} else {
		matrix.quantized = make([]turboquant.IPQuantized, matrix.rows)
		matrix.dequantized = make([]float32, matrix.rows*matrix.width)
	}
	for row, seq := range seqs {
		normalized := matrix.normalizedRow(row)
		if seq != nil && len(seq.pooled) >= matrix.width {
			raw := matrix.rawRow(row)
			copy(raw, seq.pooled[:matrix.width])
			norm := vectorNorm(raw)
			matrix.rawNorms[row] = norm
			if norm > 0 {
				inv := 1 / norm
				for i, v := range raw {
					normalized[i] = v * inv
				}
			}
		}
		// Even nil, short, or zero-norm rows need valid TurboQuant objects because
		// later scoring loops index every row position.
		if prepareQueries {
			matrix.prepared[row] = q.PrepareQuery(normalized)
		} else {
			qx := q.Quantize(normalized)
			matrix.quantized[row] = qx
			copy(matrix.dequantizedRow(row), q.Dequantize(qx))
		}
	}
	return matrix
}

func (m turboQuantPreparedPrefixMatrix) rawRow(index int) []float32 {
	if index < 0 || index >= m.rows || m.width == 0 {
		return nil
	}
	base := index * m.width
	return m.raw[base : base+m.width]
}

func (m turboQuantPreparedPrefixMatrix) normalizedRow(index int) []float32 {
	if index < 0 || index >= m.rows || m.width == 0 {
		return nil
	}
	base := index * m.width
	return m.normalized[base : base+m.width]
}

func (m turboQuantPreparedPrefixMatrix) dequantizedRow(index int) []float32 {
	if index < 0 || index >= m.rows || m.width == 0 {
		return nil
	}
	base := index * m.width
	return m.dequantized[base : base+m.width]
}

func turboQuantPreparedPrefixScore(q *turboquant.IPQuantizer, normalizedCandidate, normalizedQuery []float32) float32 {
	if q == nil || len(normalizedCandidate) != q.Dim() || len(normalizedQuery) != q.Dim() {
		return 0
	}
	return q.InnerProductPrepared(q.Quantize(normalizedCandidate), q.PrepareQuery(normalizedQuery))
}

func accumulateNormalizedPrefixSTEGrad(raw, normalized []float32, rawNorm float32, upstream []float32, scale float32, grad []float32) {
	if len(raw) != len(normalized) || len(raw) != len(upstream) || len(grad) < len(raw) || rawNorm == 0 || scale == 0 {
		return
	}
	dot := float32(0)
	for i := range raw {
		dot += normalized[i] * upstream[i]
	}
	invNorm := 1 / rawNorm
	for i := range raw {
		grad[i] += scale * (upstream[i] - normalized[i]*dot) * invNorm
	}
}

func (m contrastivePooledMatrix) row(index int) []float32 {
	if index < 0 || index >= m.rows || m.width == 0 {
		return nil
	}
	base := index * m.width
	return m.data[base : base+m.width]
}

func cosineScoreWithNorms(left, right []float32, leftNorm, rightNorm float32) float32 {
	if len(left) != len(right) || len(left) == 0 || leftNorm == 0 || rightNorm == 0 {
		return 0
	}
	dot := float32(0)
	for i := range left {
		dot += left[i] * right[i]
	}
	return dot / (leftNorm * rightNorm)
}

func accumulateCosineGradFromScore(left, right []float32, leftNorm, rightNorm, score, scale float32, gradLeft, gradRight []float32) {
	if len(left) != len(right) || len(left) == 0 || leftNorm == 0 || rightNorm == 0 || len(gradLeft) < len(left) || len(gradRight) < len(right) {
		return
	}
	denom := leftNorm * rightNorm
	leftScale := score / (leftNorm * leftNorm)
	rightScale := score / (rightNorm * rightNorm)
	for i := range left {
		gradLeft[i] += scale * (right[i]/denom - left[i]*leftScale)
		gradRight[i] += scale * (left[i]/denom - right[i]*rightScale)
	}
}

func evaluateContrastiveEncodings(queries, positives []*embeddingEncodedSequence, cfg EmbeddingTrainConfig) EmbeddingEvalMetrics {
	metrics := EmbeddingEvalMetrics{
		PairCount: len(queries) * len(positives),
	}
	infoNCELoss := float32(0)
	queryMatrix := newContrastivePooledMatrix(queries)
	positiveMatrix := newContrastivePooledMatrix(positives)
	rowScores := make([]float32, len(positives))
	rowProbs := make([]float32, len(positives))
	scores := make([]embeddingEvalScore, 0, metrics.PairCount)
	for i := range queries {
		query := queryMatrix.row(i)
		queryNorm := queryMatrix.norms[i]
		for j := range positives {
			target := float32(-1)
			if i == j {
				target = 1
			}
			score := cosineScoreWithNorms(query, positiveMatrix.row(j), queryNorm, positiveMatrix.norms[j])
			scores = append(scores, embeddingEvalScore{Score: score, Positive: target > 0})
			rowScores[j] = score
			scale := score - target
			if !embeddingUsesInfoNCELoss(cfg.ContrastiveLoss) {
				metrics.Loss += 0.5 * scale * scale
			}
			metrics.AverageScore += score
			if target > 0 {
				metrics.PositiveCount++
				metrics.PositiveMeanScore += score
				if score > 0 {
					metrics.PairAccuracy++
				}
			} else {
				metrics.NegativeCount++
				metrics.NegativeMeanScore += score
				if score < 0 {
					metrics.PairAccuracy++
				}
			}
		}
		if len(rowScores) > 0 && i < len(rowScores) {
			best := 0
			for j := 1; j < len(rowScores); j++ {
				if rowScores[j] > rowScores[best] {
					best = j
				}
			}
			if best == i {
				metrics.Top1Accuracy++
			}
			rank := 1
			targetScore := rowScores[i]
			for j, score := range rowScores {
				if j != i && score > targetScore {
					rank++
				}
			}
			if rank <= 5 {
				metrics.Top5Accuracy++
			}
			if rank <= 10 {
				metrics.Top10Accuracy++
			}
			metrics.MeanReciprocalRank += 1 / float32(rank)
			metrics.MeanPositiveRank += float32(rank)
		}
		if embeddingUsesInfoNCELoss(cfg.ContrastiveLoss) {
			infoNCELoss += infoNCERowProbsAndLossInto(rowScores, i, cfg.Temperature, rowProbs)
		}
	}
	if metrics.PairCount == 0 {
		return metrics
	}
	invPairs := float32(1) / float32(metrics.PairCount)
	if embeddingUsesInfoNCELoss(cfg.ContrastiveLoss) && len(queries) > 0 {
		metrics.Loss = infoNCELoss / float32(len(queries))
	} else {
		metrics.Loss *= invPairs
	}
	metrics.AverageScore *= invPairs
	metrics.PairAccuracy *= invPairs
	if metrics.PositiveCount > 0 {
		metrics.PositiveMeanScore /= float32(metrics.PositiveCount)
	}
	if metrics.NegativeCount > 0 {
		metrics.NegativeMeanScore /= float32(metrics.NegativeCount)
	}
	if len(queries) > 0 {
		invRows := float32(1) / float32(len(queries))
		metrics.Top1Accuracy *= invRows
		metrics.Top5Accuracy *= invRows
		metrics.Top10Accuracy *= invRows
		metrics.MeanReciprocalRank *= invRows
		metrics.MeanPositiveRank *= invRows
	}
	metrics.ScoreMargin = metrics.PositiveMeanScore - metrics.NegativeMeanScore
	finalizeEvalScoreMetrics(&metrics, scores)
	return metrics
}

func (t *EmbeddingTrainer) tryInfoNCEContrastiveAccelerator(queries, positives []*embeddingEncodedSequence, queryGrads, positiveGrads [][]float32) (float32, float32, bool) {
	if t == nil || t.contrastiveAccel == nil || len(queries) == 0 || len(queries) != len(positives) {
		return 0, 0, false
	}
	queryMatrix := newContrastivePooledMatrix(queries)
	positiveMatrix := newContrastivePooledMatrix(positives)
	if queryMatrix.rows == 0 || queryMatrix.width == 0 || positiveMatrix.rows != queryMatrix.rows || positiveMatrix.width != queryMatrix.width {
		return 0, 0, false
	}
	result, err := t.contrastiveAccel.RunInfoNCE(
		tensorF32View([]int{queryMatrix.rows, queryMatrix.width}, queryMatrix.data),
		tensorF32View([]int{positiveMatrix.rows, positiveMatrix.width}, positiveMatrix.data),
		backend.ContrastiveLossConfig{Temperature: t.config.Temperature},
	)
	if err != nil || result.QueryGrads == nil || result.PositiveGrads == nil {
		return 0, 0, false
	}
	if result.QueryGrads.Rank() != 2 || result.PositiveGrads.Rank() != 2 ||
		result.QueryGrads.Shape[0] != queryMatrix.rows || result.QueryGrads.Shape[1] != queryMatrix.width ||
		result.PositiveGrads.Shape[0] != positiveMatrix.rows || result.PositiveGrads.Shape[1] != positiveMatrix.width {
		return 0, 0, false
	}
	for row := range queryGrads {
		copy(queryGrads[row], result.QueryGrads.F32[row*queryMatrix.width:(row+1)*queryMatrix.width])
		copy(positiveGrads[row], result.PositiveGrads.F32[row*positiveMatrix.width:(row+1)*positiveMatrix.width])
	}
	return result.LossSum, result.ScoreSum, true
}

func (t *EmbeddingTrainer) tryInfoNCEHardNegativeAccelerator(queries, candidates []*embeddingEncodedSequence, targetIndexes []int, queryGrads, candidateGrads [][]float32) (float32, float32, bool) {
	if t == nil || t.contrastiveAccel == nil || len(queries) == 0 || len(candidates) < 2 || len(targetIndexes) != len(queries) {
		return 0, 0, false
	}
	queryMatrix := newContrastivePooledMatrix(queries)
	candidateMatrix := newContrastivePooledMatrix(candidates)
	if queryMatrix.rows == 0 || queryMatrix.width == 0 || candidateMatrix.rows < 2 || candidateMatrix.width != queryMatrix.width {
		return 0, 0, false
	}
	result, err := t.contrastiveAccel.RunInfoNCEWithTargets(
		tensorF32View([]int{queryMatrix.rows, queryMatrix.width}, queryMatrix.data),
		tensorF32View([]int{candidateMatrix.rows, candidateMatrix.width}, candidateMatrix.data),
		targetIndexes,
		backend.ContrastiveLossConfig{Temperature: t.config.Temperature},
	)
	if err != nil || result.QueryGrads == nil || result.PositiveGrads == nil {
		return 0, 0, false
	}
	if result.QueryGrads.Rank() != 2 || result.PositiveGrads.Rank() != 2 ||
		result.QueryGrads.Shape[0] != queryMatrix.rows || result.QueryGrads.Shape[1] != queryMatrix.width ||
		result.PositiveGrads.Shape[0] != candidateMatrix.rows || result.PositiveGrads.Shape[1] != candidateMatrix.width {
		return 0, 0, false
	}
	for row := range queryGrads {
		copy(queryGrads[row], result.QueryGrads.F32[row*queryMatrix.width:(row+1)*queryMatrix.width])
	}
	for row := range candidateGrads {
		copy(candidateGrads[row], result.PositiveGrads.F32[row*candidateMatrix.width:(row+1)*candidateMatrix.width])
	}
	return result.LossSum, result.ScoreSum, true
}

func accumulatePairMSEContrastiveGrads(queries, positives []*embeddingEncodedSequence, queryGrads, positiveGrads [][]float32) (float32, float32) {
	totalLoss := float32(0)
	totalScore := float32(0)
	queryMatrix := newContrastivePooledMatrix(queries)
	positiveMatrix := newContrastivePooledMatrix(positives)
	for i := range queries {
		query := queryMatrix.row(i)
		queryNorm := queryMatrix.norms[i]
		for j := range positives {
			target := float32(-1)
			if i == j {
				target = 1
			}
			score := cosineScoreWithNorms(query, positiveMatrix.row(j), queryNorm, positiveMatrix.norms[j])
			scale := score - target
			totalLoss += 0.5 * scale * scale
			totalScore += score
			accumulateCosineGradFromScore(query, positiveMatrix.row(j), queryNorm, positiveMatrix.norms[j], score, scale, queryGrads[i], positiveGrads[j])
		}
	}
	return totalLoss, totalScore
}

func accumulateInfoNCEContrastiveGrads(queries, positives []*embeddingEncodedSequence, temperature float32, queryGrads, positiveGrads [][]float32) (float32, float32) {
	totalLoss := float32(0)
	totalScore := float32(0)
	if temperature <= 0 {
		temperature = 0.05
	}
	queryMatrix := newContrastivePooledMatrix(queries)
	positiveMatrix := newContrastivePooledMatrix(positives)
	rowScores := make([]float32, len(positives))
	rowProbs := make([]float32, len(positives))
	for i := range queries {
		query := queryMatrix.row(i)
		queryNorm := queryMatrix.norms[i]
		for j := range positives {
			score := cosineScoreWithNorms(query, positiveMatrix.row(j), queryNorm, positiveMatrix.norms[j])
			rowScores[j] = score
			totalScore += score
		}
		rowLoss := infoNCERowProbsAndLossInto(rowScores, i, temperature, rowProbs)
		totalLoss += rowLoss
		for j, prob := range rowProbs {
			target := float32(0)
			if i == j {
				target = 1
			}
			scale := (prob - target) / temperature
			accumulateCosineGradFromScore(query, positiveMatrix.row(j), queryNorm, positiveMatrix.norms[j], rowScores[j], scale, queryGrads[i], positiveGrads[j])
		}
	}
	return totalLoss, totalScore
}

func accumulatePrefixInfoNCEContrastiveGrads(queries, positives []*embeddingEncodedSequence, dim int, temperature float32, queryGrads, positiveGrads [][]float32) (float32, float32) {
	totalLoss := float32(0)
	totalScore := float32(0)
	if dim <= 0 {
		return 0, 0
	}
	if temperature <= 0 {
		temperature = 0.05
	}
	queryMatrix := newContrastivePrefixPooledMatrix(queries, dim)
	positiveMatrix := newContrastivePrefixPooledMatrix(positives, dim)
	if queryMatrix.width == 0 || positiveMatrix.width != queryMatrix.width {
		return 0, 0
	}
	rowScores := make([]float32, len(positives))
	rowProbs := make([]float32, len(positives))
	for i := range queries {
		query := queryMatrix.row(i)
		queryNorm := queryMatrix.norms[i]
		for j := range positives {
			score := cosineScoreWithNorms(query, positiveMatrix.row(j), queryNorm, positiveMatrix.norms[j])
			rowScores[j] = score
			totalScore += score
		}
		rowLoss := infoNCERowProbsAndLossInto(rowScores, i, temperature, rowProbs)
		totalLoss += rowLoss
		for j, prob := range rowProbs {
			target := float32(0)
			if i == j {
				target = 1
			}
			scale := (prob - target) / temperature
			accumulateCosineGradFromScore(query, positiveMatrix.row(j), queryNorm, positiveMatrix.norms[j], rowScores[j], scale, queryGrads[i], positiveGrads[j])
		}
	}
	return totalLoss, totalScore
}

func accumulateTurboQuantPrefixInfoNCEContrastiveGrads(queries, positives []*embeddingEncodedSequence, dim, bitWidth int, seed int64, temperature float32, queryGrads, positiveGrads [][]float32) (float32, float32) {
	totalLoss := float32(0)
	totalScore := float32(0)
	if dim <= 0 {
		return 0, 0
	}
	if temperature <= 0 {
		temperature = 0.05
	}
	queryMatrix := newContrastivePrefixPooledMatrix(queries, dim)
	positiveMatrix := newTurboQuantDequantizedPrefixPooledMatrix(positives, dim, bitWidth, seed)
	if queryMatrix.width == 0 || positiveMatrix.width != queryMatrix.width {
		return 0, 0
	}
	rowScores := make([]float32, len(positives))
	rowProbs := make([]float32, len(positives))
	for i := range queries {
		query := queryMatrix.row(i)
		queryNorm := queryMatrix.norms[i]
		for j := range positives {
			score := cosineScoreWithNorms(query, positiveMatrix.row(j), queryNorm, positiveMatrix.norms[j])
			rowScores[j] = score
			totalScore += score
		}
		rowLoss := infoNCERowProbsAndLossInto(rowScores, i, temperature, rowProbs)
		totalLoss += rowLoss
		for j, prob := range rowProbs {
			target := float32(0)
			if i == j {
				target = 1
			}
			scale := (prob - target) / temperature
			accumulateCosineGradFromScore(query, positiveMatrix.row(j), queryNorm, positiveMatrix.norms[j], rowScores[j], scale, queryGrads[i], positiveGrads[j])
		}
	}
	return totalLoss, totalScore
}

func accumulateTurboQuantPreparedIPPrefixInfoNCEContrastiveGrads(queries, positives []*embeddingEncodedSequence, dim, bitWidth int, seed int64, temperature float32, queryGrads, positiveGrads [][]float32) (float32, float32) {
	totalLoss := float32(0)
	totalScore := float32(0)
	if dim <= 0 {
		return 0, 0
	}
	if temperature <= 0 {
		temperature = 0.05
	}
	queryMatrix := newTurboQuantPreparedPrefixMatrix(queries, dim, bitWidth, seed, true)
	positiveMatrix := newTurboQuantPreparedPrefixMatrix(positives, dim, bitWidth, seed, false)
	if queryMatrix.width == 0 || positiveMatrix.width != queryMatrix.width {
		return 0, 0
	}
	q := turboquant.NewIPWithSeed(dim, bitWidth, seed)
	rowScores := make([]float32, len(positives))
	rowProbs := make([]float32, len(positives))
	for i := range queries {
		for j := range positives {
			score := q.InnerProductPrepared(positiveMatrix.quantized[j], queryMatrix.prepared[i])
			rowScores[j] = score
			totalScore += score
		}
		rowLoss := infoNCERowProbsAndLossInto(rowScores, i, temperature, rowProbs)
		totalLoss += rowLoss
		queryRaw := queryMatrix.rawRow(i)
		queryNormalized := queryMatrix.normalizedRow(i)
		for j, prob := range rowProbs {
			target := float32(0)
			if i == j {
				target = 1
			}
			scale := (prob - target) / temperature
			accumulateNormalizedPrefixSTEGrad(queryRaw, queryNormalized, queryMatrix.rawNorms[i], positiveMatrix.dequantizedRow(j), scale, queryGrads[i])
			accumulateNormalizedPrefixSTEGrad(positiveMatrix.rawRow(j), positiveMatrix.normalizedRow(j), positiveMatrix.rawNorms[j], queryNormalized, scale, positiveGrads[j])
		}
	}
	return totalLoss, totalScore
}

func accumulateMatryoshkaContrastiveGrads(queries, positives []*embeddingEncodedSequence, cfg EmbeddingTrainConfig, queryGrads, positiveGrads [][]float32) (float32, float32, int) {
	if len(cfg.MatryoshkaDims) == 0 {
		return 0, 0, 0
	}
	totalLoss := float32(0)
	totalScore := float32(0)
	pairCount := 0
	for i, dim := range cfg.MatryoshkaDims {
		weight := float32(1)
		if i < len(cfg.MatryoshkaWeights) {
			weight = cfg.MatryoshkaWeights[i]
		}
		if weight <= 0 {
			continue
		}
		dimQueryGrads := newEmbeddingPooledGradBuffers(queries)
		dimPositiveGrads := newEmbeddingPooledGradBuffers(positives)
		loss, score := accumulatePrefixInfoNCEContrastiveGrads(queries, positives, dim, cfg.Temperature, dimQueryGrads, dimPositiveGrads)
		addScaledEmbeddingGradBuffers(queryGrads, dimQueryGrads, weight)
		addScaledEmbeddingGradBuffers(positiveGrads, dimPositiveGrads, weight)
		totalLoss += loss * weight
		totalScore += score
		pairCount += len(queries) * len(positives)
	}
	return totalLoss, totalScore, pairCount
}

func accumulateTurboQuantPrefixContrastiveGrads(queries, positives []*embeddingEncodedSequence, cfg EmbeddingTrainConfig, queryGrads, positiveGrads [][]float32) (float32, float32, int) {
	objectives := turboQuantPrefixObjectivesForConfig(cfg)
	if len(objectives) == 0 {
		return 0, 0, 0
	}
	seed := effectiveTurboQuantPrefixSeed(cfg.TurboQuantPrefixSeed)
	scoreMode, _ := normalizeTurboQuantPrefixScoreMode(cfg.TurboQuantPrefixScoreMode)
	totalLoss := float32(0)
	totalScore := float32(0)
	pairCount := 0
	for _, objective := range objectives {
		if objective.Weight <= 0 {
			continue
		}
		dimQueryGrads := newEmbeddingPooledGradBuffers(queries)
		dimPositiveGrads := newEmbeddingPooledGradBuffers(positives)
		var loss, score float32
		if scoreMode == TurboQuantPrefixScoreModePreparedIP {
			loss, score = accumulateTurboQuantPreparedIPPrefixInfoNCEContrastiveGrads(queries, positives, objective.Dim, objective.BitWidth, seed, cfg.Temperature, dimQueryGrads, dimPositiveGrads)
		} else {
			loss, score = accumulateTurboQuantPrefixInfoNCEContrastiveGrads(queries, positives, objective.Dim, objective.BitWidth, seed, cfg.Temperature, dimQueryGrads, dimPositiveGrads)
		}
		addScaledEmbeddingGradBuffers(queryGrads, dimQueryGrads, objective.Weight)
		addScaledEmbeddingGradBuffers(positiveGrads, dimPositiveGrads, objective.Weight)
		totalLoss += loss * objective.Weight
		totalScore += score
		pairCount += len(queries) * len(positives)
	}
	return totalLoss, totalScore, pairCount
}

func accumulateInfoNCEHardNegativeGrads(queries, candidates []*embeddingEncodedSequence, targetIndexes []int, temperature float32, queryGrads, candidateGrads [][]float32) (float32, float32) {
	totalLoss := float32(0)
	totalScore := float32(0)
	if temperature <= 0 {
		temperature = 0.05
	}
	queryMatrix := newContrastivePooledMatrix(queries)
	candidateMatrix := newContrastivePooledMatrix(candidates)
	rowScores := make([]float32, len(candidates))
	rowProbs := make([]float32, len(candidates))
	for i := range queries {
		query := queryMatrix.row(i)
		queryNorm := queryMatrix.norms[i]
		for j := range candidates {
			score := cosineScoreWithNorms(query, candidateMatrix.row(j), queryNorm, candidateMatrix.norms[j])
			rowScores[j] = score
			totalScore += score
		}
		targetIndex := -1
		if i < len(targetIndexes) {
			targetIndex = targetIndexes[i]
		}
		rowLoss := infoNCERowProbsAndLossInto(rowScores, targetIndex, temperature, rowProbs)
		totalLoss += rowLoss
		for j, prob := range rowProbs {
			target := float32(0)
			if j == targetIndex {
				target = 1
			}
			scale := (prob - target) / temperature
			accumulateCosineGradFromScore(query, candidateMatrix.row(j), queryNorm, candidateMatrix.norms[j], rowScores[j], scale, queryGrads[i], candidateGrads[j])
		}
	}
	return totalLoss, totalScore
}

func accumulatePrefixInfoNCEHardNegativeGrads(queries, candidates []*embeddingEncodedSequence, targetIndexes []int, dim int, temperature float32, queryGrads, candidateGrads [][]float32) (float32, float32) {
	totalLoss := float32(0)
	totalScore := float32(0)
	if dim <= 0 {
		return 0, 0
	}
	if temperature <= 0 {
		temperature = 0.05
	}
	queryMatrix := newContrastivePrefixPooledMatrix(queries, dim)
	candidateMatrix := newContrastivePrefixPooledMatrix(candidates, dim)
	if queryMatrix.width == 0 || candidateMatrix.width != queryMatrix.width {
		return 0, 0
	}
	rowScores := make([]float32, len(candidates))
	rowProbs := make([]float32, len(candidates))
	for i := range queries {
		query := queryMatrix.row(i)
		queryNorm := queryMatrix.norms[i]
		for j := range candidates {
			score := cosineScoreWithNorms(query, candidateMatrix.row(j), queryNorm, candidateMatrix.norms[j])
			rowScores[j] = score
			totalScore += score
		}
		targetIndex := -1
		if i < len(targetIndexes) {
			targetIndex = targetIndexes[i]
		}
		rowLoss := infoNCERowProbsAndLossInto(rowScores, targetIndex, temperature, rowProbs)
		totalLoss += rowLoss
		for j, prob := range rowProbs {
			target := float32(0)
			if j == targetIndex {
				target = 1
			}
			scale := (prob - target) / temperature
			accumulateCosineGradFromScore(query, candidateMatrix.row(j), queryNorm, candidateMatrix.norms[j], rowScores[j], scale, queryGrads[i], candidateGrads[j])
		}
	}
	return totalLoss, totalScore
}

func accumulateTurboQuantPrefixInfoNCEHardNegativeGrads(queries, candidates []*embeddingEncodedSequence, targetIndexes []int, dim, bitWidth int, seed int64, temperature float32, queryGrads, candidateGrads [][]float32) (float32, float32) {
	totalLoss := float32(0)
	totalScore := float32(0)
	if dim <= 0 {
		return 0, 0
	}
	if temperature <= 0 {
		temperature = 0.05
	}
	queryMatrix := newContrastivePrefixPooledMatrix(queries, dim)
	candidateMatrix := newTurboQuantDequantizedPrefixPooledMatrix(candidates, dim, bitWidth, seed)
	if queryMatrix.width == 0 || candidateMatrix.width != queryMatrix.width {
		return 0, 0
	}
	rowScores := make([]float32, len(candidates))
	rowProbs := make([]float32, len(candidates))
	for i := range queries {
		query := queryMatrix.row(i)
		queryNorm := queryMatrix.norms[i]
		for j := range candidates {
			score := cosineScoreWithNorms(query, candidateMatrix.row(j), queryNorm, candidateMatrix.norms[j])
			rowScores[j] = score
			totalScore += score
		}
		targetIndex := -1
		if i < len(targetIndexes) {
			targetIndex = targetIndexes[i]
		}
		rowLoss := infoNCERowProbsAndLossInto(rowScores, targetIndex, temperature, rowProbs)
		totalLoss += rowLoss
		for j, prob := range rowProbs {
			target := float32(0)
			if j == targetIndex {
				target = 1
			}
			scale := (prob - target) / temperature
			accumulateCosineGradFromScore(query, candidateMatrix.row(j), queryNorm, candidateMatrix.norms[j], rowScores[j], scale, queryGrads[i], candidateGrads[j])
		}
	}
	return totalLoss, totalScore
}

func accumulateTurboQuantPreparedIPPrefixInfoNCEHardNegativeGrads(queries, candidates []*embeddingEncodedSequence, targetIndexes []int, dim, bitWidth int, seed int64, temperature float32, queryGrads, candidateGrads [][]float32) (float32, float32) {
	totalLoss := float32(0)
	totalScore := float32(0)
	if dim <= 0 {
		return 0, 0
	}
	if temperature <= 0 {
		temperature = 0.05
	}
	queryMatrix := newTurboQuantPreparedPrefixMatrix(queries, dim, bitWidth, seed, true)
	candidateMatrix := newTurboQuantPreparedPrefixMatrix(candidates, dim, bitWidth, seed, false)
	if queryMatrix.width == 0 || candidateMatrix.width != queryMatrix.width {
		return 0, 0
	}
	q := turboquant.NewIPWithSeed(dim, bitWidth, seed)
	rowScores := make([]float32, len(candidates))
	rowProbs := make([]float32, len(candidates))
	for i := range queries {
		for j := range candidates {
			score := q.InnerProductPrepared(candidateMatrix.quantized[j], queryMatrix.prepared[i])
			rowScores[j] = score
			totalScore += score
		}
		targetIndex := -1
		if i < len(targetIndexes) {
			targetIndex = targetIndexes[i]
		}
		rowLoss := infoNCERowProbsAndLossInto(rowScores, targetIndex, temperature, rowProbs)
		totalLoss += rowLoss
		queryRaw := queryMatrix.rawRow(i)
		queryNormalized := queryMatrix.normalizedRow(i)
		for j, prob := range rowProbs {
			target := float32(0)
			if j == targetIndex {
				target = 1
			}
			scale := (prob - target) / temperature
			accumulateNormalizedPrefixSTEGrad(queryRaw, queryNormalized, queryMatrix.rawNorms[i], candidateMatrix.dequantizedRow(j), scale, queryGrads[i])
			accumulateNormalizedPrefixSTEGrad(candidateMatrix.rawRow(j), candidateMatrix.normalizedRow(j), candidateMatrix.rawNorms[j], queryNormalized, scale, candidateGrads[j])
		}
	}
	return totalLoss, totalScore
}

type embeddingCandidateSpan struct {
	Start int
	End   int
}

func accumulateGroupedInfoNCEHardNegativeGrads(queries, candidates []*embeddingEncodedSequence, candidateSpans []embeddingCandidateSpan, temperature float32, queryGrads, candidateGrads [][]float32) (float32, float32) {
	totalLoss := float32(0)
	totalScore := float32(0)
	if temperature <= 0 {
		temperature = 0.05
	}
	queryMatrix := newContrastivePooledMatrix(queries)
	candidateMatrix := newContrastivePooledMatrix(candidates)
	maxCandidates := 0
	for i := range queries {
		span := groupedCandidateSpan(candidateSpans, i, len(candidates))
		if n := span.End - span.Start; n > maxCandidates {
			maxCandidates = n
		}
	}
	rowScores := make([]float32, maxCandidates)
	rowProbs := make([]float32, maxCandidates)
	for i := range queries {
		span := groupedCandidateSpan(candidateSpans, i, len(candidates))
		if span.End-span.Start < 2 {
			continue
		}
		query := queryMatrix.row(i)
		queryNorm := queryMatrix.norms[i]
		scores := rowScores[:span.End-span.Start]
		probs := rowProbs[:span.End-span.Start]
		for j := span.Start; j < span.End; j++ {
			local := j - span.Start
			score := cosineScoreWithNorms(query, candidateMatrix.row(j), queryNorm, candidateMatrix.norms[j])
			scores[local] = score
			totalScore += score
		}
		rowLoss := infoNCERowProbsAndLossInto(scores, 0, temperature, probs)
		totalLoss += rowLoss
		for local, prob := range probs {
			target := float32(0)
			if local == 0 {
				target = 1
			}
			j := span.Start + local
			scale := (prob - target) / temperature
			accumulateCosineGradFromScore(query, candidateMatrix.row(j), queryNorm, candidateMatrix.norms[j], scores[local], scale, queryGrads[i], candidateGrads[j])
		}
	}
	return totalLoss, totalScore
}

func accumulateScoreSpectrumGrads(queries, candidates []*embeddingEncodedSequence, candidateSpans []embeddingCandidateSpan, examples []EmbeddingScoreSpectrumExample, cfg EmbeddingTrainConfig, queryGrads, candidateGrads [][]float32) (float32, float32, int, error) {
	totalLoss := float32(0)
	totalScore := float32(0)
	pairCount := 0
	queryMatrix := newContrastivePooledMatrix(queries)
	candidateMatrix := newContrastivePooledMatrix(candidates)
	maxCandidates := 0
	for i := range queries {
		span := groupedCandidateSpan(candidateSpans, i, len(candidates))
		if n := span.End - span.Start; n > maxCandidates {
			maxCandidates = n
		}
	}
	rowScores := make([]float32, maxCandidates)
	for i := range queries {
		span := groupedCandidateSpan(candidateSpans, i, len(candidates))
		candidateCount := span.End - span.Start
		if candidateCount <= 0 {
			return 0, 0, 0, fmt.Errorf("score-spectrum row %d has no candidates", i)
		}
		if i >= len(examples) {
			return 0, 0, 0, fmt.Errorf("score-spectrum row %d missing example metadata", i)
		}
		example := examples[i]
		hardWeight, softWeight := scoreSpectrumEffectiveLossWeights(example.HardLossWeight, example.SoftLossWeight)
		if !scoreSpectrumLossModeIncludesHardSoft(cfg.ScoreSpectrumLossMode) {
			hardWeight, softWeight = 0, 0
		}
		recoveryWeight := scoreSpectrumEffectiveRecoveryWeightForExample(cfg.ScoreSpectrumRecoveryWeight, example)
		query := queryMatrix.row(i)
		queryNorm := queryMatrix.norms[i]
		scores := rowScores[:candidateCount]
		for j := span.Start; j < span.End; j++ {
			local := j - span.Start
			score := cosineScoreWithNorms(query, candidateMatrix.row(j), queryNorm, candidateMatrix.norms[j])
			scores[local] = score
			totalScore += score
		}
		loss, err := scoreSpectrumLossAndGrad(scores, example.PositiveIndexes, example.HardNegativeEligible, example.TargetProbabilities, cfg.Temperature, hardWeight, softWeight, scoreSpectrumRecoveryLossOptions{
			Enabled: scoreSpectrumLossModeIncludesRecovery(cfg.ScoreSpectrumLossMode) && recoveryWeight > 0,
			Weight:  recoveryWeight,
			Margin:  cfg.ScoreSpectrumRecoveryMargin,
			TopK:    cfg.ScoreSpectrumRecoveryTopK,
			Tau:     cfg.ScoreSpectrumRecoveryTau,
		})
		if err != nil {
			return 0, 0, 0, fmt.Errorf("score-spectrum row %d: %w", i, err)
		}
		totalLoss += loss.Loss
		pairCount += candidateCount
		for local, scale := range loss.Grad {
			if scale == 0 {
				continue
			}
			j := span.Start + local
			accumulateCosineGradFromScore(query, candidateMatrix.row(j), queryNorm, candidateMatrix.norms[j], scores[local], scale, queryGrads[i], candidateGrads[j])
		}
	}
	return totalLoss, totalScore, pairCount, nil
}

func accumulateListwiseGeometryGrads(queries, documents []*embeddingEncodedSequence, spans []embeddingCandidateSpan, batches []EmbeddingTokenizedListwiseGeometryBatch, temperature float32, queryGrads, documentGrads [][]float32) (float32, float32, int, int, error) {
	totalLoss := float32(0)
	totalScore := float32(0)
	pairCount := 0
	totalQueryCount := 0
	queryMatrix := newContrastivePooledMatrix(queries)
	documentMatrix := newContrastivePooledMatrix(documents)
	queryOffset := 0
	for i, batch := range batches {
		if i >= len(spans) {
			return 0, 0, 0, 0, fmt.Errorf("listwise geometry batch %d missing document span", i)
		}
		span := groupedCandidateSpan(spans, i, len(documents))
		queryCount := len(batch.QueryTokens)
		docCount := span.End - span.Start
		if queryCount <= 0 || docCount <= 0 {
			return 0, 0, 0, 0, fmt.Errorf("listwise geometry batch %d has empty query/document shape", i)
		}
		if queryOffset+queryCount > len(queries) {
			return 0, 0, 0, 0, fmt.Errorf("listwise geometry batch %d query span exceeds encodings", i)
		}
		student := make([][]float32, queryCount)
		for qi := 0; qi < queryCount; qi++ {
			globalQuery := queryOffset + qi
			query := queryMatrix.row(globalQuery)
			queryNorm := queryMatrix.norms[globalQuery]
			student[qi] = make([]float32, docCount)
			for localDoc := 0; localDoc < docCount; localDoc++ {
				globalDoc := span.Start + localDoc
				score := cosineScoreWithNorms(query, documentMatrix.row(globalDoc), queryNorm, documentMatrix.norms[globalDoc])
				student[qi][localDoc] = score
				totalScore += score
			}
		}
		loss, err := EmbeddingListwiseGeometryLossAndGrad(student, batch.TeacherSimilarity, temperature)
		if err != nil {
			return 0, 0, 0, 0, fmt.Errorf("listwise geometry batch %d: %w", i, err)
		}
		totalLoss += loss.Loss * float32(queryCount)
		pairCount += queryCount * docCount
		totalQueryCount += queryCount
		for qi := 0; qi < queryCount; qi++ {
			globalQuery := queryOffset + qi
			query := queryMatrix.row(globalQuery)
			queryNorm := queryMatrix.norms[globalQuery]
			for localDoc, scale := range loss.Grad[qi] {
				if scale == 0 {
					continue
				}
				globalDoc := span.Start + localDoc
				accumulateCosineGradFromScore(query, documentMatrix.row(globalDoc), queryNorm, documentMatrix.norms[globalDoc], student[qi][localDoc], scale*float32(queryCount), queryGrads[globalQuery], documentGrads[globalDoc])
			}
		}
		queryOffset += queryCount
	}
	return totalLoss, totalScore, pairCount, totalQueryCount, nil
}

func evaluateListwiseGeometryEncodings(queries, documents []*embeddingEncodedSequence, spans []embeddingCandidateSpan, batches []EmbeddingTokenizedListwiseGeometryBatch, temperature float32) (EmbeddingListwiseGeometryEvalMetrics, error) {
	var metrics EmbeddingListwiseGeometryEvalMetrics
	metrics.BatchCount = len(batches)
	queryMatrix := newContrastivePooledMatrix(queries)
	documentMatrix := newContrastivePooledMatrix(documents)
	queryOffset := 0
	for i, batch := range batches {
		if i >= len(spans) {
			return EmbeddingListwiseGeometryEvalMetrics{}, fmt.Errorf("listwise geometry batch %d missing document span", i)
		}
		span := groupedCandidateSpan(spans, i, len(documents))
		queryCount := len(batch.QueryTokens)
		docCount := span.End - span.Start
		if queryCount <= 0 || docCount <= 0 {
			return EmbeddingListwiseGeometryEvalMetrics{}, fmt.Errorf("listwise geometry batch %d has empty query/document shape", i)
		}
		if queryOffset+queryCount > len(queries) {
			return EmbeddingListwiseGeometryEvalMetrics{}, fmt.Errorf("listwise geometry batch %d query span exceeds encodings", i)
		}
		student := make([][]float32, queryCount)
		for qi := 0; qi < queryCount; qi++ {
			globalQuery := queryOffset + qi
			query := queryMatrix.row(globalQuery)
			queryNorm := queryMatrix.norms[globalQuery]
			student[qi] = make([]float32, docCount)
			for localDoc := 0; localDoc < docCount; localDoc++ {
				globalDoc := span.Start + localDoc
				score := cosineScoreWithNorms(query, documentMatrix.row(globalDoc), queryNorm, documentMatrix.norms[globalDoc])
				student[qi][localDoc] = score
				metrics.AverageScore += score
			}
		}
		loss, err := EmbeddingListwiseGeometryLossAndGrad(student, batch.TeacherSimilarity, temperature)
		if err != nil {
			return EmbeddingListwiseGeometryEvalMetrics{}, fmt.Errorf("listwise geometry batch %d: %w", i, err)
		}
		queryPositiveDocs := listwiseGeometryQueryPositiveDocSets(batch)
		for qi := 0; qi < queryCount; qi++ {
			rowCE := float32(0)
			teacherEntropy := float32(0)
			for j, p := range loss.TeacherProbs[qi] {
				if p > 0 {
					prob := loss.StudentProbs[qi][j]
					if prob < 1e-12 {
						prob = 1e-12
					}
					rowCE -= p * float32(math.Log(float64(prob)))
					teacherEntropy -= p * float32(math.Log(float64(p)))
				}
			}
			metrics.TeacherCrossEntropy += rowCE
			metrics.Loss += rowCE
			metrics.TeacherKL += rowCE - teacherEntropy
			teacherTop := maxFloat32Index(batch.TeacherSimilarity[qi])
			studentTop := maxFloat32Index(student[qi])
			if studentTop == teacherTop {
				metrics.TeacherTop1Agreement++
			}
			positiveDocs := queryPositiveDocs[listwiseGeometryQueryID(batch, qi)]
			if len(positiveDocs) > 0 {
				metrics.AnyPositiveQueryCount++
				if positiveDocs[listwiseGeometryDocumentID(batch, studentTop)] {
					metrics.AnyPositiveTop1++
				}
			}
		}
		metrics.QueryCount += queryCount
		metrics.DocumentCellCount += queryCount * docCount
		queryOffset += queryCount
	}
	normalizeListwiseGeometryEvalMetrics(&metrics)
	return metrics, nil
}

func listwiseGeometrySequenceInputs(batches []EmbeddingTokenizedListwiseGeometryBatch, queryRole, documentRole int32) ([]embeddingSequenceInput, []embeddingSequenceInput, []embeddingCandidateSpan, error) {
	queryInputs := []embeddingSequenceInput{}
	documentInputs := []embeddingSequenceInput{}
	documentSpans := make([]embeddingCandidateSpan, len(batches))
	for i, batch := range batches {
		if err := validateTokenizedListwiseGeometryBatch(batch); err != nil {
			return nil, nil, nil, fmt.Errorf("listwise geometry batch %d: %w", i, err)
		}
		for j, tokens := range batch.QueryTokens {
			var mask []int32
			if j < len(batch.QueryMasks) {
				mask = batch.QueryMasks[j]
			}
			queryInputs = append(queryInputs, embeddingSequenceInput{
				tokens: tokens,
				mask:   mask,
				role:   queryRole,
				label:  fmt.Sprintf("listwise geometry batch %d query %d", i, j),
			})
		}
		documentSpans[i].Start = len(documentInputs)
		for j, tokens := range batch.DocumentTokens {
			var mask []int32
			if j < len(batch.DocumentMasks) {
				mask = batch.DocumentMasks[j]
			}
			documentInputs = append(documentInputs, embeddingSequenceInput{
				tokens: tokens,
				mask:   mask,
				role:   documentRole,
				label:  fmt.Sprintf("listwise geometry batch %d document %d", i, j),
			})
		}
		documentSpans[i].End = len(documentInputs)
	}
	return queryInputs, documentInputs, documentSpans, nil
}

func listwiseGeometryQueryPositiveDocSets(batch EmbeddingTokenizedListwiseGeometryBatch) map[string]map[string]bool {
	out := map[string]map[string]bool{}
	for _, example := range batch.Examples {
		queryID := strings.TrimSpace(example.QueryID)
		docID := strings.TrimSpace(example.PositiveDocID)
		if queryID == "" || docID == "" {
			continue
		}
		set := out[queryID]
		if set == nil {
			set = map[string]bool{}
			out[queryID] = set
		}
		set[docID] = true
	}
	return out
}

func listwiseGeometryQueryID(batch EmbeddingTokenizedListwiseGeometryBatch, index int) string {
	if index >= 0 && index < len(batch.QueryIDs) {
		return strings.TrimSpace(batch.QueryIDs[index])
	}
	return ""
}

func listwiseGeometryDocumentID(batch EmbeddingTokenizedListwiseGeometryBatch, index int) string {
	if index >= 0 && index < len(batch.DocumentIDs) {
		return strings.TrimSpace(batch.DocumentIDs[index])
	}
	return ""
}

func maxFloat32Index(values []float32) int {
	best := 0
	for i, value := range values {
		if i == 0 || value > values[best] {
			best = i
		}
	}
	return best
}

func validateTokenizedListwiseGeometryBatch(batch EmbeddingTokenizedListwiseGeometryBatch) error {
	if len(batch.QueryTokens) == 0 {
		return fmt.Errorf("query_tokens are empty")
	}
	if len(batch.DocumentTokens) == 0 {
		return fmt.Errorf("document_tokens are empty")
	}
	if len(batch.QueryIDs) > 0 && len(batch.QueryIDs) != len(batch.QueryTokens) {
		return fmt.Errorf("query_ids length %d does not match query_tokens length %d", len(batch.QueryIDs), len(batch.QueryTokens))
	}
	if len(batch.DocumentIDs) > 0 && len(batch.DocumentIDs) != len(batch.DocumentTokens) {
		return fmt.Errorf("document_ids length %d does not match document_tokens length %d", len(batch.DocumentIDs), len(batch.DocumentTokens))
	}
	if len(batch.QueryMasks) > 0 && len(batch.QueryMasks) != len(batch.QueryTokens) {
		return fmt.Errorf("query_masks length %d does not match query_tokens length %d", len(batch.QueryMasks), len(batch.QueryTokens))
	}
	if len(batch.DocumentMasks) > 0 && len(batch.DocumentMasks) != len(batch.DocumentTokens) {
		return fmt.Errorf("document_masks length %d does not match document_tokens length %d", len(batch.DocumentMasks), len(batch.DocumentTokens))
	}
	for i, tokens := range batch.QueryTokens {
		if len(tokens) == 0 {
			return fmt.Errorf("query_tokens[%d] are empty", i)
		}
		if len(batch.QueryMasks) > i && len(batch.QueryMasks[i]) > 0 && len(batch.QueryMasks[i]) != len(tokens) {
			return fmt.Errorf("query_masks[%d] length %d does not match query_tokens[%d] length %d", i, len(batch.QueryMasks[i]), i, len(tokens))
		}
	}
	for i, tokens := range batch.DocumentTokens {
		if len(tokens) == 0 {
			return fmt.Errorf("document_tokens[%d] are empty", i)
		}
		if len(batch.DocumentMasks) > i && len(batch.DocumentMasks[i]) > 0 && len(batch.DocumentMasks[i]) != len(tokens) {
			return fmt.Errorf("document_masks[%d] length %d does not match document_tokens[%d] length %d", i, len(batch.DocumentMasks[i]), i, len(tokens))
		}
	}
	return validateEmbeddingListwiseGeometryMatrix(batch.TeacherSimilarity, len(batch.QueryTokens), len(batch.DocumentTokens), "teacher_similarity")
}

func validateListwiseGeometryTrainerConfig(cfg EmbeddingTrainConfig) error {
	if len(cfg.MatryoshkaDims) > 0 {
		return fmt.Errorf("listwise geometry training does not support matryoshka objectives in v1")
	}
	if len(cfg.TurboQuantPrefixBits) > 0 || len(cfg.TurboQuantPrefixObjectives) > 0 || len(turboQuantPrefixObjectivesForConfig(cfg)) > 0 {
		return fmt.Errorf("listwise geometry training does not support turboquant prefix objectives in v1")
	}
	if len(cfg.TurboQuantCompactObjectives) > 0 || len(turboQuantCompactObjectivesForConfig(cfg)) > 0 {
		return fmt.Errorf("listwise geometry training does not support turboquant compact objectives in v1")
	}
	if len(cfg.TurboQuantRankMarginObjectives) > 0 || len(turboQuantRankMarginObjectivesForConfig(cfg)) > 0 {
		return fmt.Errorf("listwise geometry training does not support turboquant rank-margin objectives in v1")
	}
	if cfg.Temperature <= 0 || math.IsNaN(float64(cfg.Temperature)) || math.IsInf(float64(cfg.Temperature), 0) {
		return fmt.Errorf("temperature must be finite and positive")
	}
	return nil
}

func scoreSpectrumSequenceInputs(examples []EmbeddingScoreSpectrumExample, queryRole, documentRole int32) ([]embeddingSequenceInput, []embeddingSequenceInput, []embeddingCandidateSpan, error) {
	examples, err := canonicalizeTokenizedScoreSpectrumExamples(examples)
	if err != nil {
		return nil, nil, nil, err
	}
	queryInputs := make([]embeddingSequenceInput, len(examples))
	candidateInputs := make([]embeddingSequenceInput, 0, len(examples)*2)
	candidateSpans := make([]embeddingCandidateSpan, len(examples))
	for i, example := range examples {
		if err := validateTokenizedScoreSpectrumShape(example); err != nil {
			return nil, nil, nil, fmt.Errorf("score-spectrum row %d: %w", i, err)
		}
		queryInputs[i] = embeddingSequenceInput{
			tokens: example.QueryTokens,
			mask:   example.QueryMask,
			role:   queryRole,
			label:  fmt.Sprintf("score-spectrum row %d query", i),
		}
		candidateSpans[i].Start = len(candidateInputs)
		for j, tokens := range example.CandidateTokens {
			var mask []int32
			if j < len(example.CandidateMasks) {
				mask = example.CandidateMasks[j]
			}
			candidateInputs = append(candidateInputs, embeddingSequenceInput{
				tokens: tokens,
				mask:   mask,
				role:   documentRole,
				label:  fmt.Sprintf("score-spectrum row %d candidate %d", i, j),
			})
		}
		candidateSpans[i].End = len(candidateInputs)
	}
	return queryInputs, candidateInputs, candidateSpans, nil
}

func canonicalizeTokenizedScoreSpectrumExamples(examples []EmbeddingScoreSpectrumExample) ([]EmbeddingScoreSpectrumExample, error) {
	canonical := make([]EmbeddingScoreSpectrumExample, len(examples))
	for i, example := range examples {
		clean, err := canonicalizeTokenizedScoreSpectrumExample(example)
		if err != nil {
			return nil, fmt.Errorf("score-spectrum row %d: %w", i, err)
		}
		canonical[i] = clean
	}
	return canonical, nil
}

func evaluateScoreSpectrumEncodings(queries, candidates []*embeddingEncodedSequence, candidateSpans []embeddingCandidateSpan, examples []EmbeddingScoreSpectrumExample, temperature float32) (EmbeddingScoreSpectrumEvalMetrics, error) {
	var metrics EmbeddingScoreSpectrumEvalMetrics
	metrics.RowCount = len(examples)
	queryMatrix := newContrastivePooledMatrix(queries)
	candidateMatrix := newContrastivePooledMatrix(candidates)
	maxCandidates := 0
	for i := range queries {
		span := groupedCandidateSpan(candidateSpans, i, len(candidates))
		if n := span.End - span.Start; n > maxCandidates {
			maxCandidates = n
		}
	}
	rowScores := make([]float32, maxCandidates)
	modelProbs := make([]float32, maxCandidates)
	for i := range queries {
		span := groupedCandidateSpan(candidateSpans, i, len(candidates))
		candidateCount := span.End - span.Start
		if candidateCount <= 0 {
			return EmbeddingScoreSpectrumEvalMetrics{}, fmt.Errorf("score-spectrum row %d has no candidates", i)
		}
		if i >= len(examples) {
			return EmbeddingScoreSpectrumEvalMetrics{}, fmt.Errorf("score-spectrum row %d missing example metadata", i)
		}
		example := examples[i]
		positiveSet, err := scoreSpectrumPositiveSet(candidateCount, example.PositiveIndexes)
		if err != nil {
			return EmbeddingScoreSpectrumEvalMetrics{}, fmt.Errorf("score-spectrum row %d: %w", i, err)
		}
		query := queryMatrix.row(i)
		queryNorm := queryMatrix.norms[i]
		scores := rowScores[:candidateCount]
		topIndex := 0
		for j := span.Start; j < span.End; j++ {
			local := j - span.Start
			score := cosineScoreWithNorms(query, candidateMatrix.row(j), queryNorm, candidateMatrix.norms[j])
			scores[local] = score
			metrics.AverageScore += score
			metrics.CandidateCount++
			if local == 0 || score > scores[topIndex] {
				topIndex = local
			}
		}

		hardWeight, softWeight := scoreSpectrumEffectiveLossWeights(example.HardLossWeight, example.SoftLossWeight)
		loss, err := scoreSpectrumLossAndGrad(scores, example.PositiveIndexes, example.HardNegativeEligible, example.TargetProbabilities, temperature, hardWeight, softWeight, scoreSpectrumRecoveryLossOptions{})
		if err != nil {
			return EmbeddingScoreSpectrumEvalMetrics{}, fmt.Errorf("score-spectrum row %d: %w", i, err)
		}
		metrics.Loss += loss.Loss

		if len(example.PositiveIndexes) > 0 {
			metrics.AnyPositiveRowCount++
			if positiveSet[topIndex] {
				metrics.AnyPositiveTop1++
			}
		}
		if example.SelectedPositiveIndex != nil {
			metrics.OriginalPositiveRowCount++
			if topIndex == *example.SelectedPositiveIndex {
				metrics.OriginalPositiveTop1++
			}
			if len(example.PositiveIndexes) > 1 {
				metrics.AlternateRecoveryRowCount++
				if positiveSet[topIndex] && topIndex != *example.SelectedPositiveIndex {
					metrics.AlternateRelevantRecovery++
				}
			}
		}

		bestPositive := float32(math.Inf(-1))
		hardestNegative := float32(math.Inf(-1))
		for j, score := range scores {
			if positiveSet[j] {
				if score > bestPositive {
					bestPositive = score
				}
				continue
			}
			if j < len(example.HardNegativeEligible) && example.HardNegativeEligible[j] && score > hardestNegative {
				hardestNegative = score
			}
		}
		if !math.IsInf(float64(bestPositive), -1) && !math.IsInf(float64(hardestNegative), -1) {
			metrics.BestPositiveHardestNegativeMargin += bestPositive - hardestNegative
			metrics.MarginRowCount++
		}

		if len(example.TargetProbabilities) == candidateCount {
			probs := modelProbs[:candidateCount]
			softmaxScoresInto(scores, temperature, probs)
			for j, target := range example.TargetProbabilities {
				if target == 0 {
					continue
				}
				prob := probs[j]
				if prob < 1e-12 {
					prob = 1e-12
				}
				metrics.TargetCrossEntropy -= target * float32(math.Log(float64(prob)))
				metrics.TargetKL += target * float32(math.Log(float64(target/prob)))
			}
			metrics.TargetDistributionRowCount++
		}
	}
	normalizeScoreSpectrumEvalMetrics(&metrics)
	return metrics, nil
}

func scoreSpectrumEffectiveLossWeights(hardWeight, softWeight float32) (float32, float32) {
	// Missing row weights default to the v1 hard-only objective. Rows that set
	// either weight explicitly preserve the dataset values, including zeros.
	if hardWeight == 0 && softWeight == 0 {
		return 1, 0
	}
	return hardWeight, softWeight
}

func scoreSpectrumEffectiveRecoveryWeight(globalWeight, rowWeight float32) float32 {
	// A missing per-row weight keeps the global recovery objective weight.
	// Positive row weights scale that global weight, so datasets can emphasize
	// recovery rows with values such as 2/1/1 without disabling default rows.
	if rowWeight > 0 {
		return globalWeight * rowWeight
	}
	return globalWeight
}

func scoreSpectrumEffectiveRecoveryWeightForExample(globalWeight float32, example EmbeddingScoreSpectrumExample) float32 {
	// Soft-only rows can explicitly disable recovery by carrying a zero recovery
	// weight with a positive soft objective and no hard objective. Legacy rows
	// with omitted weights still inherit the global recovery setting.
	if example.RecoveryLossWeight == 0 && example.HardLossWeight == 0 && example.SoftLossWeight > 0 {
		return 0
	}
	return scoreSpectrumEffectiveRecoveryWeight(globalWeight, example.RecoveryLossWeight)
}

func validateScoreSpectrumTrainerConfig(cfg EmbeddingTrainConfig) error {
	if len(cfg.MatryoshkaDims) > 0 {
		return fmt.Errorf("score-spectrum training does not support matryoshka objectives in v1")
	}
	if len(cfg.TurboQuantPrefixBits) > 0 || len(cfg.TurboQuantPrefixObjectives) > 0 || len(turboQuantPrefixObjectivesForConfig(cfg)) > 0 {
		return fmt.Errorf("score-spectrum training does not support turboquant prefix objectives in v1")
	}
	if len(cfg.TurboQuantCompactObjectives) > 0 || len(turboQuantCompactObjectivesForConfig(cfg)) > 0 {
		return fmt.Errorf("score-spectrum training does not support turboquant compact objectives in v1")
	}
	if len(cfg.TurboQuantRankMarginObjectives) > 0 || len(turboQuantRankMarginObjectivesForConfig(cfg)) > 0 {
		return fmt.Errorf("score-spectrum training does not support turboquant rank-margin objectives in v1")
	}
	if err := validateScoreSpectrumRecoveryConfig(cfg.ScoreSpectrumLossMode, cfg.ScoreSpectrumRecoveryWeight, cfg.ScoreSpectrumRecoveryMargin, cfg.ScoreSpectrumRecoveryTopK, cfg.ScoreSpectrumRecoveryTau); err != nil {
		return err
	}
	return nil
}

func validateTokenizedScoreSpectrumShape(example EmbeddingScoreSpectrumExample) error {
	if len(example.QueryTokens) == 0 {
		return fmt.Errorf("query_tokens are empty")
	}
	if len(example.QueryMask) > 0 && len(example.QueryMask) != len(example.QueryTokens) {
		return fmt.Errorf("query_mask length %d does not match query_tokens length %d", len(example.QueryMask), len(example.QueryTokens))
	}
	if len(example.CandidateTokens) == 0 {
		return fmt.Errorf("candidate_tokens are empty")
	}
	if len(example.CandidateIDs) > 0 && len(example.CandidateIDs) != len(example.CandidateTokens) {
		return fmt.Errorf("candidate_ids length %d does not match candidate_tokens length %d", len(example.CandidateIDs), len(example.CandidateTokens))
	}
	if len(example.CandidateMasks) > 0 && len(example.CandidateMasks) != len(example.CandidateTokens) {
		return fmt.Errorf("candidate_masks length %d does not match candidate_tokens length %d", len(example.CandidateMasks), len(example.CandidateTokens))
	}
	for i, tokens := range example.CandidateTokens {
		if len(tokens) == 0 {
			return fmt.Errorf("candidate_tokens[%d] are empty", i)
		}
		if len(example.CandidateMasks) > i && len(example.CandidateMasks[i]) > 0 && len(example.CandidateMasks[i]) != len(tokens) {
			return fmt.Errorf("candidate_masks[%d] length %d does not match candidate_tokens[%d] length %d", i, len(example.CandidateMasks[i]), i, len(tokens))
		}
	}
	if err := validateScoreSpectrumRecoveryLossWeight(example.RecoveryLossWeight); err != nil {
		return err
	}
	positiveIndexes, err := canonicalizeScoreSpectrumPositiveIndexes(len(example.CandidateTokens), example.PositiveIndexes, example.SelectedPositiveIndex)
	if err != nil {
		return err
	}
	return validateScoreSpectrumLabelsAndProbabilities(len(example.CandidateTokens), positiveIndexes, example.SelectedPositiveIndex, example.HardNegativeEligible, example.TargetProbabilities)
}

func accumulatePrefixGroupedInfoNCEHardNegativeGrads(queries, candidates []*embeddingEncodedSequence, candidateSpans []embeddingCandidateSpan, dim int, temperature float32, queryGrads, candidateGrads [][]float32) (float32, float32) {
	totalLoss := float32(0)
	totalScore := float32(0)
	if dim <= 0 {
		return 0, 0
	}
	if temperature <= 0 {
		temperature = 0.05
	}
	queryMatrix := newContrastivePrefixPooledMatrix(queries, dim)
	candidateMatrix := newContrastivePrefixPooledMatrix(candidates, dim)
	if queryMatrix.width == 0 || candidateMatrix.width != queryMatrix.width {
		return 0, 0
	}
	maxCandidates := 0
	for i := range queries {
		span := groupedCandidateSpan(candidateSpans, i, len(candidates))
		if n := span.End - span.Start; n > maxCandidates {
			maxCandidates = n
		}
	}
	rowScores := make([]float32, maxCandidates)
	rowProbs := make([]float32, maxCandidates)
	for i := range queries {
		span := groupedCandidateSpan(candidateSpans, i, len(candidates))
		if span.End-span.Start < 2 {
			continue
		}
		query := queryMatrix.row(i)
		queryNorm := queryMatrix.norms[i]
		scores := rowScores[:span.End-span.Start]
		probs := rowProbs[:span.End-span.Start]
		for j := span.Start; j < span.End; j++ {
			local := j - span.Start
			score := cosineScoreWithNorms(query, candidateMatrix.row(j), queryNorm, candidateMatrix.norms[j])
			scores[local] = score
			totalScore += score
		}
		rowLoss := infoNCERowProbsAndLossInto(scores, 0, temperature, probs)
		totalLoss += rowLoss
		for local, prob := range probs {
			target := float32(0)
			if local == 0 {
				target = 1
			}
			j := span.Start + local
			scale := (prob - target) / temperature
			accumulateCosineGradFromScore(query, candidateMatrix.row(j), queryNorm, candidateMatrix.norms[j], scores[local], scale, queryGrads[i], candidateGrads[j])
		}
	}
	return totalLoss, totalScore
}

func accumulateTurboQuantPrefixGroupedInfoNCEHardNegativeGrads(queries, candidates []*embeddingEncodedSequence, candidateSpans []embeddingCandidateSpan, dim, bitWidth int, seed int64, temperature float32, queryGrads, candidateGrads [][]float32) (float32, float32) {
	totalLoss := float32(0)
	totalScore := float32(0)
	if dim <= 0 {
		return 0, 0
	}
	if temperature <= 0 {
		temperature = 0.05
	}
	queryMatrix := newContrastivePrefixPooledMatrix(queries, dim)
	candidateMatrix := newTurboQuantDequantizedPrefixPooledMatrix(candidates, dim, bitWidth, seed)
	if queryMatrix.width == 0 || candidateMatrix.width != queryMatrix.width {
		return 0, 0
	}
	maxCandidates := 0
	for i := range queries {
		span := groupedCandidateSpan(candidateSpans, i, len(candidates))
		if n := span.End - span.Start; n > maxCandidates {
			maxCandidates = n
		}
	}
	rowScores := make([]float32, maxCandidates)
	rowProbs := make([]float32, maxCandidates)
	for i := range queries {
		span := groupedCandidateSpan(candidateSpans, i, len(candidates))
		if span.End-span.Start < 2 {
			continue
		}
		query := queryMatrix.row(i)
		queryNorm := queryMatrix.norms[i]
		scores := rowScores[:span.End-span.Start]
		probs := rowProbs[:span.End-span.Start]
		for j := span.Start; j < span.End; j++ {
			local := j - span.Start
			score := cosineScoreWithNorms(query, candidateMatrix.row(j), queryNorm, candidateMatrix.norms[j])
			scores[local] = score
			totalScore += score
		}
		rowLoss := infoNCERowProbsAndLossInto(scores, 0, temperature, probs)
		totalLoss += rowLoss
		for local, prob := range probs {
			target := float32(0)
			if local == 0 {
				target = 1
			}
			j := span.Start + local
			scale := (prob - target) / temperature
			accumulateCosineGradFromScore(query, candidateMatrix.row(j), queryNorm, candidateMatrix.norms[j], scores[local], scale, queryGrads[i], candidateGrads[j])
		}
	}
	return totalLoss, totalScore
}

func accumulateTurboQuantPreparedIPPrefixGroupedInfoNCEHardNegativeGrads(queries, candidates []*embeddingEncodedSequence, candidateSpans []embeddingCandidateSpan, dim, bitWidth int, seed int64, temperature float32, queryGrads, candidateGrads [][]float32) (float32, float32) {
	totalLoss := float32(0)
	totalScore := float32(0)
	if dim <= 0 {
		return 0, 0
	}
	if temperature <= 0 {
		temperature = 0.05
	}
	queryMatrix := newTurboQuantPreparedPrefixMatrix(queries, dim, bitWidth, seed, true)
	candidateMatrix := newTurboQuantPreparedPrefixMatrix(candidates, dim, bitWidth, seed, false)
	if queryMatrix.width == 0 || candidateMatrix.width != queryMatrix.width {
		return 0, 0
	}
	q := turboquant.NewIPWithSeed(dim, bitWidth, seed)
	maxCandidates := 0
	for i := range queries {
		span := groupedCandidateSpan(candidateSpans, i, len(candidates))
		if n := span.End - span.Start; n > maxCandidates {
			maxCandidates = n
		}
	}
	rowScores := make([]float32, maxCandidates)
	rowProbs := make([]float32, maxCandidates)
	for i := range queries {
		span := groupedCandidateSpan(candidateSpans, i, len(candidates))
		if span.End-span.Start < 2 {
			continue
		}
		scores := rowScores[:span.End-span.Start]
		probs := rowProbs[:span.End-span.Start]
		for j := span.Start; j < span.End; j++ {
			local := j - span.Start
			score := q.InnerProductPrepared(candidateMatrix.quantized[j], queryMatrix.prepared[i])
			scores[local] = score
			totalScore += score
		}
		rowLoss := infoNCERowProbsAndLossInto(scores, 0, temperature, probs)
		totalLoss += rowLoss
		queryRaw := queryMatrix.rawRow(i)
		queryNormalized := queryMatrix.normalizedRow(i)
		for local, prob := range probs {
			target := float32(0)
			if local == 0 {
				target = 1
			}
			j := span.Start + local
			scale := (prob - target) / temperature
			accumulateNormalizedPrefixSTEGrad(queryRaw, queryNormalized, queryMatrix.rawNorms[i], candidateMatrix.dequantizedRow(j), scale, queryGrads[i])
			accumulateNormalizedPrefixSTEGrad(candidateMatrix.rawRow(j), candidateMatrix.normalizedRow(j), candidateMatrix.rawNorms[j], queryNormalized, scale, candidateGrads[j])
		}
	}
	return totalLoss, totalScore
}

func accumulateMatryoshkaHardNegativeGrads(queries, candidates []*embeddingEncodedSequence, targetIndexes []int, candidateSpans []embeddingCandidateSpan, cfg EmbeddingTrainConfig, queryGrads, candidateGrads [][]float32) (float32, float32, int) {
	if len(cfg.MatryoshkaDims) == 0 {
		return 0, 0, 0
	}
	totalLoss := float32(0)
	totalScore := float32(0)
	pairCount := 0
	for i, dim := range cfg.MatryoshkaDims {
		weight := float32(1)
		if i < len(cfg.MatryoshkaWeights) {
			weight = cfg.MatryoshkaWeights[i]
		}
		if weight <= 0 {
			continue
		}
		dimQueryGrads := newEmbeddingPooledGradBuffers(queries)
		dimCandidateGrads := newEmbeddingPooledGradBuffers(candidates)
		loss := float32(0)
		score := float32(0)
		pairs := 0
		switch cfg.ContrastiveLoss {
		case "grouped_infonce":
			loss, score = accumulatePrefixGroupedInfoNCEHardNegativeGrads(queries, candidates, candidateSpans, dim, cfg.Temperature, dimQueryGrads, dimCandidateGrads)
			pairs = hardNegativeCandidatePairCount(len(queries), len(candidates), candidateSpans, "grouped_infonce")
		case "hybrid_infonce":
			globalQueryGrads := newEmbeddingPooledGradBuffers(queries)
			globalCandidateGrads := newEmbeddingPooledGradBuffers(candidates)
			globalLoss, globalScore := accumulatePrefixInfoNCEHardNegativeGrads(queries, candidates, targetIndexes, dim, cfg.Temperature, globalQueryGrads, globalCandidateGrads)
			groupedQueryGrads := newEmbeddingPooledGradBuffers(queries)
			groupedCandidateGrads := newEmbeddingPooledGradBuffers(candidates)
			groupedLoss, groupedScore := accumulatePrefixGroupedInfoNCEHardNegativeGrads(queries, candidates, candidateSpans, dim, cfg.Temperature, groupedQueryGrads, groupedCandidateGrads)
			groupedWeight := effectiveGroupedLossWeight(cfg.ContrastiveLoss, cfg.GroupedLossWeight)
			globalScale := float32(1) / (1 + groupedWeight)
			groupedScale := groupedWeight / (1 + groupedWeight)
			addScaledEmbeddingGradBuffers(dimQueryGrads, globalQueryGrads, globalScale)
			addScaledEmbeddingGradBuffers(dimCandidateGrads, globalCandidateGrads, globalScale)
			addScaledEmbeddingGradBuffers(dimQueryGrads, groupedQueryGrads, groupedScale)
			addScaledEmbeddingGradBuffers(dimCandidateGrads, groupedCandidateGrads, groupedScale)
			loss = globalLoss*globalScale + groupedLoss*groupedScale
			score = globalScore + groupedScore
			pairs = hardNegativeCandidatePairCount(len(queries), len(candidates), candidateSpans, "hybrid_infonce")
		default:
			loss, score = accumulatePrefixInfoNCEHardNegativeGrads(queries, candidates, targetIndexes, dim, cfg.Temperature, dimQueryGrads, dimCandidateGrads)
			pairs = hardNegativeCandidatePairCount(len(queries), len(candidates), candidateSpans, "infonce")
		}
		addScaledEmbeddingGradBuffers(queryGrads, dimQueryGrads, weight)
		addScaledEmbeddingGradBuffers(candidateGrads, dimCandidateGrads, weight)
		totalLoss += loss * weight
		totalScore += score
		pairCount += pairs
	}
	return totalLoss, totalScore, pairCount
}

func accumulateTurboQuantPrefixHardNegativeGrads(queries, candidates []*embeddingEncodedSequence, targetIndexes []int, candidateSpans []embeddingCandidateSpan, cfg EmbeddingTrainConfig, queryGrads, candidateGrads [][]float32) (float32, float32, int) {
	objectives := turboQuantPrefixObjectivesForConfig(cfg)
	if len(objectives) == 0 {
		return 0, 0, 0
	}
	seed := effectiveTurboQuantPrefixSeed(cfg.TurboQuantPrefixSeed)
	scoreMode, _ := normalizeTurboQuantPrefixScoreMode(cfg.TurboQuantPrefixScoreMode)
	totalLoss := float32(0)
	totalScore := float32(0)
	pairCount := 0
	for _, objective := range objectives {
		if objective.Weight <= 0 {
			continue
		}
		dimQueryGrads := newEmbeddingPooledGradBuffers(queries)
		dimCandidateGrads := newEmbeddingPooledGradBuffers(candidates)
		loss := float32(0)
		score := float32(0)
		pairs := 0
		switch cfg.ContrastiveLoss {
		case "grouped_infonce":
			if scoreMode == TurboQuantPrefixScoreModePreparedIP {
				loss, score = accumulateTurboQuantPreparedIPPrefixGroupedInfoNCEHardNegativeGrads(queries, candidates, candidateSpans, objective.Dim, objective.BitWidth, seed, cfg.Temperature, dimQueryGrads, dimCandidateGrads)
			} else {
				loss, score = accumulateTurboQuantPrefixGroupedInfoNCEHardNegativeGrads(queries, candidates, candidateSpans, objective.Dim, objective.BitWidth, seed, cfg.Temperature, dimQueryGrads, dimCandidateGrads)
			}
			pairs = hardNegativeCandidatePairCount(len(queries), len(candidates), candidateSpans, "grouped_infonce")
		case "hybrid_infonce":
			globalQueryGrads := newEmbeddingPooledGradBuffers(queries)
			globalCandidateGrads := newEmbeddingPooledGradBuffers(candidates)
			var globalLoss, globalScore float32
			if scoreMode == TurboQuantPrefixScoreModePreparedIP {
				globalLoss, globalScore = accumulateTurboQuantPreparedIPPrefixInfoNCEHardNegativeGrads(queries, candidates, targetIndexes, objective.Dim, objective.BitWidth, seed, cfg.Temperature, globalQueryGrads, globalCandidateGrads)
			} else {
				globalLoss, globalScore = accumulateTurboQuantPrefixInfoNCEHardNegativeGrads(queries, candidates, targetIndexes, objective.Dim, objective.BitWidth, seed, cfg.Temperature, globalQueryGrads, globalCandidateGrads)
			}
			groupedQueryGrads := newEmbeddingPooledGradBuffers(queries)
			groupedCandidateGrads := newEmbeddingPooledGradBuffers(candidates)
			var groupedLoss, groupedScore float32
			if scoreMode == TurboQuantPrefixScoreModePreparedIP {
				groupedLoss, groupedScore = accumulateTurboQuantPreparedIPPrefixGroupedInfoNCEHardNegativeGrads(queries, candidates, candidateSpans, objective.Dim, objective.BitWidth, seed, cfg.Temperature, groupedQueryGrads, groupedCandidateGrads)
			} else {
				groupedLoss, groupedScore = accumulateTurboQuantPrefixGroupedInfoNCEHardNegativeGrads(queries, candidates, candidateSpans, objective.Dim, objective.BitWidth, seed, cfg.Temperature, groupedQueryGrads, groupedCandidateGrads)
			}
			groupedWeight := effectiveGroupedLossWeight(cfg.ContrastiveLoss, cfg.GroupedLossWeight)
			globalScale := float32(1) / (1 + groupedWeight)
			groupedScale := groupedWeight / (1 + groupedWeight)
			addScaledEmbeddingGradBuffers(dimQueryGrads, globalQueryGrads, globalScale)
			addScaledEmbeddingGradBuffers(dimCandidateGrads, globalCandidateGrads, globalScale)
			addScaledEmbeddingGradBuffers(dimQueryGrads, groupedQueryGrads, groupedScale)
			addScaledEmbeddingGradBuffers(dimCandidateGrads, groupedCandidateGrads, groupedScale)
			loss = globalLoss*globalScale + groupedLoss*groupedScale
			score = globalScore + groupedScore
			pairs = hardNegativeCandidatePairCount(len(queries), len(candidates), candidateSpans, "hybrid_infonce")
		default:
			if scoreMode == TurboQuantPrefixScoreModePreparedIP {
				loss, score = accumulateTurboQuantPreparedIPPrefixInfoNCEHardNegativeGrads(queries, candidates, targetIndexes, objective.Dim, objective.BitWidth, seed, cfg.Temperature, dimQueryGrads, dimCandidateGrads)
			} else {
				loss, score = accumulateTurboQuantPrefixInfoNCEHardNegativeGrads(queries, candidates, targetIndexes, objective.Dim, objective.BitWidth, seed, cfg.Temperature, dimQueryGrads, dimCandidateGrads)
			}
			pairs = hardNegativeCandidatePairCount(len(queries), len(candidates), candidateSpans, "infonce")
		}
		addScaledEmbeddingGradBuffers(queryGrads, dimQueryGrads, objective.Weight)
		addScaledEmbeddingGradBuffers(candidateGrads, dimCandidateGrads, objective.Weight)
		totalLoss += loss * objective.Weight
		totalScore += score
		pairCount += pairs
	}
	return totalLoss, totalScore, pairCount
}

func accumulateTurboQuantRankMarginHardNegativeGrads(queries, candidates []*embeddingEncodedSequence, candidateSpans []embeddingCandidateSpan, teacherScores [][]float32, cfg EmbeddingTrainConfig, queryGrads, candidateGrads [][]float32) (float32, float32, int) {
	objectives := turboQuantRankMarginObjectivesForConfig(cfg)
	if len(objectives) == 0 {
		return 0, 0, 0
	}
	seed := effectiveTurboQuantPrefixSeed(cfg.TurboQuantPrefixSeed)
	scoreMode, _ := normalizeTurboQuantPrefixScoreMode(cfg.TurboQuantPrefixScoreMode)
	margin := effectiveTurboQuantRankMargin(cfg.TurboQuantRankMargin)
	totalLoss := float32(0)
	totalScore := float32(0)
	pairCount := 0
	for _, objective := range objectives {
		if objective.Weight <= 0 {
			continue
		}
		dimQueryGrads := newEmbeddingPooledGradBuffers(queries)
		dimCandidateGrads := newEmbeddingPooledGradBuffers(candidates)
		var loss, score float32
		var pairs int
		if scoreMode == TurboQuantPrefixScoreModePreparedIP {
			loss, score, pairs = accumulateTurboQuantPreparedIPRankMarginHardNegativeGrads(queries, candidates, candidateSpans, teacherScores, objective.Dim, objective.BitWidth, seed, margin, dimQueryGrads, dimCandidateGrads)
		} else {
			loss, score, pairs = accumulateTurboQuantRankMarginReconstructCosineHardNegativeGrads(queries, candidates, candidateSpans, teacherScores, objective.Dim, objective.BitWidth, seed, margin, dimQueryGrads, dimCandidateGrads)
		}
		addScaledEmbeddingGradBuffers(queryGrads, dimQueryGrads, objective.Weight)
		addScaledEmbeddingGradBuffers(candidateGrads, dimCandidateGrads, objective.Weight)
		totalLoss += loss * objective.Weight
		totalScore += score
		pairCount += pairs
	}
	return totalLoss, totalScore, pairCount
}

func accumulateTurboQuantCompactHardNegativeGrads(queries, candidates []*embeddingEncodedSequence, candidateSpans []embeddingCandidateSpan, teacherScores [][]float32, cfg EmbeddingTrainConfig, queryGrads, candidateGrads [][]float32) (float32, float32, int) {
	objectives := turboQuantCompactObjectivesForConfig(cfg)
	if len(objectives) == 0 {
		return 0, 0, 0
	}
	seed := effectiveTurboQuantPrefixSeed(cfg.TurboQuantPrefixSeed)
	totalLoss := float32(0)
	totalScore := float32(0)
	pairCount := 0
	for _, objective := range objectives {
		if objective.Weight <= 0 {
			continue
		}
		dimQueryGrads := newEmbeddingPooledGradBuffers(queries)
		dimCandidateGrads := newEmbeddingPooledGradBuffers(candidates)
		loss, score, pairs := accumulateTurboQuantPreparedIPCompactHardNegativeGrads(queries, candidates, candidateSpans, teacherScores, objective.Dim, objective.BitWidth, seed, cfg.Temperature, cfg.TeacherTemperature, dimQueryGrads, dimCandidateGrads)
		addScaledEmbeddingGradBuffers(queryGrads, dimQueryGrads, objective.Weight)
		addScaledEmbeddingGradBuffers(candidateGrads, dimCandidateGrads, objective.Weight)
		totalLoss += loss * objective.Weight
		totalScore += score
		pairCount += pairs
	}
	return totalLoss, totalScore, pairCount
}

func accumulateTurboQuantPreparedIPCompactHardNegativeGrads(queries, candidates []*embeddingEncodedSequence, candidateSpans []embeddingCandidateSpan, teacherScores [][]float32, dim, bitWidth int, seed int64, modelTemperature, teacherTemperature float32, queryGrads, candidateGrads [][]float32) (float32, float32, int) {
	totalLoss := float32(0)
	totalScore := float32(0)
	pairCount := 0
	if dim <= 0 {
		return 0, 0, 0
	}
	if modelTemperature <= 0 {
		modelTemperature = 0.05
	}
	if teacherTemperature <= 0 {
		teacherTemperature = 1
	}
	queryMatrix := newTurboQuantPreparedPrefixMatrix(queries, dim, bitWidth, seed, true)
	candidateMatrix := newTurboQuantPreparedPrefixMatrix(candidates, dim, bitWidth, seed, false)
	if queryMatrix.width == 0 || candidateMatrix.width != queryMatrix.width {
		return 0, 0, 0
	}
	maxCandidates := 0
	for i := range queries {
		span := groupedCandidateSpan(candidateSpans, i, len(candidates))
		if n := span.End - span.Start; n > maxCandidates {
			maxCandidates = n
		}
	}
	if maxCandidates < 2 {
		return 0, 0, 0
	}
	q := turboquant.NewIPWithSeed(dim, bitWidth, seed)
	modelScores := make([]float32, maxCandidates)
	modelProbs := make([]float32, maxCandidates)
	teacherProbs := make([]float32, maxCandidates)
	for i := range queries {
		span := groupedCandidateSpan(candidateSpans, i, len(candidates))
		count := span.End - span.Start
		if count < 2 {
			continue
		}
		scores := modelScores[:count]
		for j := span.Start; j < span.End; j++ {
			local := j - span.Start
			score := q.InnerProductPrepared(candidateMatrix.quantized[j], queryMatrix.prepared[i])
			scores[local] = score
			totalScore += score
		}
		model := modelProbs[:count]
		softmaxScoresInto(scores, modelTemperature, model)
		target := teacherProbs[:count]
		useTeacher := i < len(teacherScores) && len(teacherScores[i]) == count
		if useTeacher {
			softmaxScoresInto(teacherScores[i], teacherTemperature, target)
		}
		queryRaw := queryMatrix.rawRow(i)
		queryNormalized := queryMatrix.normalizedRow(i)
		for local, prob := range model {
			targetProb := float32(0)
			if useTeacher {
				targetProb = target[local]
			} else if local == 0 {
				targetProb = 1
			}
			if prob < 1e-12 {
				prob = 1e-12
			}
			totalLoss -= targetProb * float32(math.Log(float64(prob)))
			j := span.Start + local
			scale := (model[local] - targetProb) / modelTemperature
			accumulateNormalizedPrefixSTEGrad(queryRaw, queryNormalized, queryMatrix.rawNorms[i], candidateMatrix.dequantizedRow(j), scale, queryGrads[i])
			accumulateNormalizedPrefixSTEGrad(candidateMatrix.rawRow(j), candidateMatrix.normalizedRow(j), candidateMatrix.rawNorms[j], queryNormalized, scale, candidateGrads[j])
		}
		pairCount += count
	}
	return totalLoss, totalScore, pairCount
}

func accumulateTurboQuantRankMarginReconstructCosineHardNegativeGrads(queries, candidates []*embeddingEncodedSequence, candidateSpans []embeddingCandidateSpan, teacherScores [][]float32, dim, bitWidth int, seed int64, margin float32, queryGrads, candidateGrads [][]float32) (float32, float32, int) {
	totalLoss := float32(0)
	totalScore := float32(0)
	pairCount := 0
	if dim <= 0 {
		return 0, 0, 0
	}
	queryMatrix := newContrastivePrefixPooledMatrix(queries, dim)
	candidateMatrix := newTurboQuantDequantizedPrefixPooledMatrix(candidates, dim, bitWidth, seed)
	if queryMatrix.width == 0 || candidateMatrix.width != queryMatrix.width {
		return 0, 0, 0
	}
	for i := range queries {
		span := groupedCandidateSpan(candidateSpans, i, len(candidates))
		if span.End-span.Start < 2 {
			continue
		}
		query := queryMatrix.row(i)
		queryNorm := queryMatrix.norms[i]
		posIndex := span.Start
		posScore := cosineScoreWithNorms(query, candidateMatrix.row(posIndex), queryNorm, candidateMatrix.norms[posIndex])
		bestIndex := -1
		bestScore := float32(0)
		for j := span.Start + 1; j < span.End; j++ {
			if !turboQuantRankMarginTeacherEligible(teacherScores, i, j-span.Start) {
				continue
			}
			score := cosineScoreWithNorms(query, candidateMatrix.row(j), queryNorm, candidateMatrix.norms[j])
			if bestIndex < 0 || score > bestScore {
				bestIndex = j
				bestScore = score
			}
		}
		if bestIndex < 0 {
			continue
		}
		qMargin := posScore - bestScore
		totalScore += qMargin
		pairCount++
		violation := margin - qMargin
		if violation <= 0 {
			continue
		}
		totalLoss += 0.5 * violation * violation
		accumulateCosineGradFromScore(query, candidateMatrix.row(posIndex), queryNorm, candidateMatrix.norms[posIndex], posScore, -violation, queryGrads[i], candidateGrads[posIndex])
		accumulateCosineGradFromScore(query, candidateMatrix.row(bestIndex), queryNorm, candidateMatrix.norms[bestIndex], bestScore, violation, queryGrads[i], candidateGrads[bestIndex])
	}
	return totalLoss, totalScore, pairCount
}

func accumulateTurboQuantPreparedIPRankMarginHardNegativeGrads(queries, candidates []*embeddingEncodedSequence, candidateSpans []embeddingCandidateSpan, teacherScores [][]float32, dim, bitWidth int, seed int64, margin float32, queryGrads, candidateGrads [][]float32) (float32, float32, int) {
	totalLoss := float32(0)
	totalScore := float32(0)
	pairCount := 0
	if dim <= 0 {
		return 0, 0, 0
	}
	queryMatrix := newTurboQuantPreparedPrefixMatrix(queries, dim, bitWidth, seed, true)
	candidateMatrix := newTurboQuantPreparedPrefixMatrix(candidates, dim, bitWidth, seed, false)
	if queryMatrix.width == 0 || candidateMatrix.width != queryMatrix.width {
		return 0, 0, 0
	}
	q := turboquant.NewIPWithSeed(dim, bitWidth, seed)
	for i := range queries {
		span := groupedCandidateSpan(candidateSpans, i, len(candidates))
		if span.End-span.Start < 2 {
			continue
		}
		posIndex := span.Start
		posScore := q.InnerProductPrepared(candidateMatrix.quantized[posIndex], queryMatrix.prepared[i])
		bestIndex := -1
		bestScore := float32(0)
		for j := span.Start + 1; j < span.End; j++ {
			if !turboQuantRankMarginTeacherEligible(teacherScores, i, j-span.Start) {
				continue
			}
			score := q.InnerProductPrepared(candidateMatrix.quantized[j], queryMatrix.prepared[i])
			if bestIndex < 0 || score > bestScore {
				bestIndex = j
				bestScore = score
			}
		}
		if bestIndex < 0 {
			continue
		}
		qMargin := posScore - bestScore
		totalScore += qMargin
		pairCount++
		violation := margin - qMargin
		if violation <= 0 {
			continue
		}
		totalLoss += 0.5 * violation * violation
		queryRaw := queryMatrix.rawRow(i)
		queryNormalized := queryMatrix.normalizedRow(i)
		accumulateNormalizedPrefixSTEGrad(queryRaw, queryNormalized, queryMatrix.rawNorms[i], candidateMatrix.dequantizedRow(posIndex), -violation, queryGrads[i])
		accumulateNormalizedPrefixSTEGrad(candidateMatrix.rawRow(posIndex), candidateMatrix.normalizedRow(posIndex), candidateMatrix.rawNorms[posIndex], queryNormalized, -violation, candidateGrads[posIndex])
		accumulateNormalizedPrefixSTEGrad(queryRaw, queryNormalized, queryMatrix.rawNorms[i], candidateMatrix.dequantizedRow(bestIndex), violation, queryGrads[i])
		accumulateNormalizedPrefixSTEGrad(candidateMatrix.rawRow(bestIndex), candidateMatrix.normalizedRow(bestIndex), candidateMatrix.rawNorms[bestIndex], queryNormalized, violation, candidateGrads[bestIndex])
	}
	return totalLoss, totalScore, pairCount
}

func turboQuantRankMarginTeacherEligible(teacherScores [][]float32, row, localCandidate int) bool {
	if row < 0 || row >= len(teacherScores) || len(teacherScores[row]) == 0 {
		return true
	}
	if localCandidate <= 0 || localCandidate >= len(teacherScores[row]) {
		return false
	}
	return teacherScores[row][localCandidate] < teacherScores[row][0]
}

func accumulateTeacherDistributionHardNegativeGrads(queries, candidates []*embeddingEncodedSequence, candidateSpans []embeddingCandidateSpan, teacherScores [][]float32, teacherTemperatures []float32, teacherSourceWeights []float32, modelTemperature, teacherTemperature float32, queryGrads, candidateGrads [][]float32) (float32, float32, int) {
	totalLoss := float32(0)
	totalScore := float32(0)
	pairCount := 0
	if modelTemperature <= 0 {
		modelTemperature = 0.05
	}
	if teacherTemperature <= 0 {
		teacherTemperature = 1
	}
	queryMatrix := newContrastivePooledMatrix(queries)
	candidateMatrix := newContrastivePooledMatrix(candidates)
	maxCandidates := 0
	for i := range queries {
		if i >= len(teacherScores) || len(teacherScores[i]) == 0 {
			continue
		}
		span := groupedCandidateSpan(candidateSpans, i, len(candidates))
		if n := span.End - span.Start; n > maxCandidates {
			maxCandidates = n
		}
	}
	if maxCandidates < 2 {
		return 0, 0, 0
	}
	modelScores := make([]float32, maxCandidates)
	modelProbs := make([]float32, maxCandidates)
	teacherProbs := make([]float32, maxCandidates)
	for i := range queries {
		if i >= len(teacherScores) || len(teacherScores[i]) == 0 {
			continue
		}
		exampleWeight := float32(1)
		if i < len(teacherSourceWeights) {
			exampleWeight = teacherSourceWeights[i]
		}
		if exampleWeight <= 0 {
			continue
		}
		span := groupedCandidateSpan(candidateSpans, i, len(candidates))
		count := span.End - span.Start
		if count < 2 || len(teacherScores[i]) != count {
			continue
		}
		query := queryMatrix.row(i)
		queryNorm := queryMatrix.norms[i]
		scores := modelScores[:count]
		for j := span.Start; j < span.End; j++ {
			local := j - span.Start
			score := cosineScoreWithNorms(query, candidateMatrix.row(j), queryNorm, candidateMatrix.norms[j])
			scores[local] = score
			totalScore += exampleWeight * score
		}
		model := modelProbs[:count]
		teacher := teacherProbs[:count]
		exampleTeacherTemperature := teacherTemperature
		if i < len(teacherTemperatures) && teacherTemperatures[i] > 0 {
			exampleTeacherTemperature = teacherTemperatures[i]
		}
		softmaxScoresInto(scores, modelTemperature, model)
		softmaxScoresInto(teacherScores[i], exampleTeacherTemperature, teacher)
		for local, target := range teacher {
			prob := model[local]
			if prob < 1e-12 {
				prob = 1e-12
			}
			totalLoss -= exampleWeight * target * float32(math.Log(float64(prob)))
			j := span.Start + local
			scale := exampleWeight * (model[local] - target) / modelTemperature
			accumulateCosineGradFromScore(query, candidateMatrix.row(j), queryNorm, candidateMatrix.norms[j], scores[local], scale, queryGrads[i], candidateGrads[j])
		}
		pairCount += count
	}
	return totalLoss, totalScore, pairCount
}

func groupedCandidateSpan(spans []embeddingCandidateSpan, index int, candidateCount int) embeddingCandidateSpan {
	if index >= 0 && index < len(spans) {
		span := spans[index]
		if span.Start >= 0 && span.Start < span.End && span.End <= candidateCount {
			return span
		}
	}
	return embeddingCandidateSpan{Start: 0, End: candidateCount}
}

func hardNegativeCandidatePairCount(queryCount, candidateCount int, spans []embeddingCandidateSpan, loss string) int {
	if queryCount <= 0 || candidateCount <= 0 {
		return 0
	}
	if loss != "grouped_infonce" && loss != "hybrid_infonce" {
		return queryCount * candidateCount
	}
	groupedTotal := 0
	for i := 0; i < queryCount; i++ {
		span := groupedCandidateSpan(spans, i, candidateCount)
		if span.End-span.Start >= 2 {
			groupedTotal += span.End - span.Start
		}
	}
	if loss == "hybrid_infonce" {
		return queryCount*candidateCount + groupedTotal
	}
	return groupedTotal
}

func newEmbeddingPooledGradBuffers(seqs []*embeddingEncodedSequence) [][]float32 {
	grads := make([][]float32, len(seqs))
	for i := range seqs {
		grads[i] = make([]float32, len(seqs[i].pooled))
	}
	return grads
}

func scaleEmbeddingGradBuffers(grads [][]float32, scale float32) {
	if scale == 1 {
		return
	}
	for _, row := range grads {
		for i := range row {
			row[i] *= scale
		}
	}
}

func addScaledEmbeddingGradBuffers(dst, src [][]float32, scale float32) {
	if scale == 0 {
		return
	}
	for i := range dst {
		if i >= len(src) {
			return
		}
		for j := range dst[i] {
			if j >= len(src[i]) {
				break
			}
			dst[i][j] += src[i][j] * scale
		}
	}
}

func infoNCERowLoss(scores []float32, targetIndex int, temperature float32) float32 {
	_, loss := infoNCERowProbsAndLoss(scores, targetIndex, temperature)
	return loss
}

func infoNCERowProbsAndLoss(scores []float32, targetIndex int, temperature float32) ([]float32, float32) {
	probs := make([]float32, len(scores))
	loss := infoNCERowProbsAndLossInto(scores, targetIndex, temperature, probs)
	return probs, loss
}

func infoNCERowProbsAndLossInto(scores []float32, targetIndex int, temperature float32, probs []float32) float32 {
	if len(scores) == 0 || targetIndex < 0 || targetIndex >= len(scores) {
		for i := range probs {
			probs[i] = 0
		}
		return 0
	}
	if temperature <= 0 {
		temperature = 0.05
	}
	if len(probs) < len(scores) {
		return 0
	}
	probs = probs[:len(scores)]
	maxLogit := scores[0] / temperature
	for _, score := range scores[1:] {
		logit := score / temperature
		if logit > maxLogit {
			maxLogit = logit
		}
	}
	sum := float32(0)
	for i, score := range scores {
		value := float32(math.Exp(float64(score/temperature - maxLogit)))
		probs[i] = value
		sum += value
	}
	if sum == 0 {
		return 0
	}
	invSum := 1 / sum
	for i := range probs {
		probs[i] *= invSum
	}
	prob := probs[targetIndex]
	if prob < 1e-12 {
		prob = 1e-12
	}
	return -float32(math.Log(float64(prob)))
}

func softmaxScoresInto(scores []float32, temperature float32, probs []float32) {
	if len(scores) == 0 || len(probs) < len(scores) {
		for i := range probs {
			probs[i] = 0
		}
		return
	}
	if temperature <= 0 {
		temperature = 1
	}
	probs = probs[:len(scores)]
	maxLogit := scores[0] / temperature
	for _, score := range scores[1:] {
		logit := score / temperature
		if logit > maxLogit {
			maxLogit = logit
		}
	}
	sum := float32(0)
	for i, score := range scores {
		value := float32(math.Exp(float64(score/temperature - maxLogit)))
		probs[i] = value
		sum += value
	}
	if sum == 0 {
		for i := range probs {
			probs[i] = 0
		}
		return
	}
	invSum := 1 / sum
	for i := range probs {
		probs[i] *= invSum
	}
}

func (t *EmbeddingTrainer) prepareForwardWeights() *embeddingForwardWeights {
	if t == nil {
		return nil
	}
	if t.isCompactTrainer() {
		return &embeddingForwardWeights{compact: t.prepareCompactForwardWeights()}
	}
	if t.forwardCache != nil && !t.forwardDirty {
		return t.forwardCache
	}
	if t.forwardCache == nil {
		t.forwardCache = &embeddingForwardWeights{}
	}
	t.forwardCache.token = refreshForwardTensorForParam(t.tokenParam, t.tokenEmbed, t.config.WeightBits, t.forwardCache.token)
	t.forwardCache.role = refreshForwardTensorForParam(t.roleParam, t.roleEmbed, t.config.WeightBits, t.forwardCache.role)
	t.forwardCache.attnQ = refreshForwardMatMulTensorForParam(t.attnQParam, t.attentionQuery, t.forwardCache.attnQ)
	t.forwardCache.attnK = refreshForwardMatMulTensorForParam(t.attnKParam, t.attentionKey, t.forwardCache.attnK)
	t.forwardCache.attnV = refreshForwardMatMulTensorForParam(t.attnVParam, t.attentionValue, t.forwardCache.attnV)
	t.forwardCache.attnO = refreshForwardMatMulTensorForParam(t.attnOParam, t.attentionOutput, t.forwardCache.attnO)
	t.forwardCache.hidden = refreshForwardMatMulTensorForParam(t.hiddenParam, t.hiddenProjection, t.forwardCache.hidden)
	t.forwardCache.proj = refreshForwardMatMulTensorForParam(t.projParam, t.projection, t.forwardCache.proj)
	t.forwardDirty = false
	return t.forwardCache
}

func (t *EmbeddingTrainer) invalidateForwardWeights() {
	if t == nil {
		return
	}
	t.forwardDirty = true
	t.forwardNeedsBind = true
}

func (t *EmbeddingTrainer) prepareCompactForwardWeights() *compactEmbeddingForwardWeights {
	if t == nil || t.compactState == nil {
		return nil
	}
	if t.compactForwardCache != nil && !t.forwardDirty {
		t.primeCompactForwardWeightResidency(t.compactForwardCache)
		return t.compactForwardCache
	}
	state := t.compactState
	forward := &compactEmbeddingForwardWeights{
		token:  tensorAsMasterF32(state.TokenEmbedding.Tensor),
		layers: make([]compactEmbeddingForwardLayer, len(state.Layers)),
	}
	if state.RoleEmbedding != nil {
		forward.role = tensorAsMasterF32(state.RoleEmbedding.Tensor)
	}
	for i, layer := range state.Layers {
		forward.layers[i] = compactEmbeddingForwardLayer{
			attnQName:      layer.AttentionQuery.Name,
			attnKName:      layer.AttentionKey.Name,
			attnVName:      layer.AttentionValue.Name,
			attnOName:      layer.AttentionOutput.Name,
			ffnUpName:      layer.FFNUp.Name,
			ffnDownName:    layer.FFNDown.Name,
			attnQ:          tensorAsMasterF32(layer.AttentionQuery.Tensor),
			attnK:          tensorAsMasterF32(layer.AttentionKey.Tensor),
			attnV:          tensorAsMasterF32(layer.AttentionValue.Tensor),
			attnO:          tensorAsMasterF32(layer.AttentionOutput.Tensor),
			ffnUp:          tensorAsMasterF32(layer.FFNUp.Tensor),
			ffnDown:        tensorAsMasterF32(layer.FFNDown.Tensor),
			attentionHeads: state.Manifest.AttentionHeads,
			headDim:        state.Manifest.HeadDim,
		}
	}
	if state.OutputProjection != nil {
		forward.outputProjectionName = state.OutputProjection.Name
		forward.outputProjection = tensorAsMasterF32(state.OutputProjection.Tensor)
	}
	t.compactForwardCache = forward
	t.forwardDirty = false
	t.primeCompactForwardWeightResidency(forward)
	return forward
}

func (t *EmbeddingTrainer) encodeExamplePair(example EmbeddingPairExample, tokenForward, roleForward, attnQForward, attnKForward, attnVForward, attnOForward, hiddenForward, projForward *backend.Tensor, captureBindings bool) (*embeddingEncodedSequence, *embeddingEncodedSequence, error) {
	leftMask, err := t.prepareMask(example.LeftTokens, example.LeftMask)
	if err != nil {
		return nil, nil, fmt.Errorf("left: %w", err)
	}
	rightMask, err := t.prepareMask(example.RightTokens, example.RightMask)
	if err != nil {
		return nil, nil, fmt.Errorf("right: %w", err)
	}
	left, err := t.encodeSequence(example.LeftTokens, leftMask, t.rawRoleIndex(), tokenForward, roleForward, attnQForward, attnKForward, attnVForward, attnOForward, hiddenForward, projForward, captureBindings)
	if err != nil {
		return nil, nil, fmt.Errorf("left: %w", err)
	}
	right, err := t.encodeSequence(example.RightTokens, rightMask, t.rawRoleIndex(), tokenForward, roleForward, attnQForward, attnKForward, attnVForward, attnOForward, hiddenForward, projForward, captureBindings)
	if err != nil {
		t.releaseEncodedSequenceBindings(left)
		return nil, nil, fmt.Errorf("right: %w", err)
	}
	return left, right, nil
}

func (t *EmbeddingTrainer) prepareMask(tokens, mask []int32) ([]int32, error) {
	if err := t.validateTokenSequence(tokens); err != nil {
		return nil, err
	}
	if len(mask) == 0 {
		mask = make([]int32, len(tokens))
		for i := range mask {
			mask[i] = 1
		}
		return mask, nil
	}
	if len(mask) != len(tokens) {
		return nil, fmt.Errorf("mask length %d does not match token length %d", len(mask), len(tokens))
	}
	active := 0
	for _, v := range mask {
		if v != 0 {
			active++
		}
	}
	if active == 0 {
		return nil, fmt.Errorf("mask selects zero tokens")
	}
	return append([]int32(nil), mask...), nil
}

func (t *EmbeddingTrainer) rawRoleIndex() int32 {
	if t == nil || !t.manifest.roleConditioned() {
		return 0
	}
	return t.manifest.RawRoleIndex
}

func (t *EmbeddingTrainer) queryRoleIndex() int32 {
	if t == nil || !t.manifest.roleConditioned() {
		return 0
	}
	return t.manifest.QueryRoleIndex
}

func (t *EmbeddingTrainer) documentRoleIndex() int32 {
	if t == nil || !t.manifest.roleConditioned() {
		return 0
	}
	return t.manifest.DocumentRoleIndex
}

func (t *EmbeddingTrainer) validateTokenSequence(tokens []int32) error {
	if len(tokens) == 0 {
		return fmt.Errorf("tokens are empty")
	}
	if limit := t.manifest.Tokenizer.MaxSequence; limit > 0 && len(tokens) > limit {
		return fmt.Errorf("token sequence length %d exceeds max_sequence %d", len(tokens), limit)
	}
	if vocab := t.manifest.Tokenizer.VocabSize; vocab > 0 {
		for i, tok := range tokens {
			if tok < 0 || int(tok) >= vocab {
				return fmt.Errorf("token %d value %d is outside vocab_size %d", i, tok, vocab)
			}
		}
	}
	return nil
}

func (t *EmbeddingTrainer) encodeSequence(tokens, mask []int32, role int32, tokenEmbed, roleEmbed, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection *backend.Tensor, captureBindings bool) (*embeddingEncodedSequence, error) {
	if t.isCompactTrainer() {
		return t.encodeCompactSequence(tokens, mask, role, t.prepareCompactForwardWeights())
	}
	input, err := embeddingInputForTokens(tokenEmbed, tokens)
	if err != nil {
		return nil, err
	}
	if err := addRoleEmbeddingToInput(input, roleEmbed, role, len(tokens)); err != nil {
		return nil, err
	}
	if err := applyEmbeddingPositionEncoding(input, len(tokens), tokenEmbed.Shape[1], t.manifest.PositionEncoding); err != nil {
		return nil, err
	}
	encoded := &embeddingEncodedSequence{
		layers: make([]*embeddingSequenceState, 0, t.encoderRepeats()),
		pooled: nil,
		tokens: append([]int32(nil), tokens...),
		role:   role,
	}
	current := input
	for layer := 0; layer < t.encoderRepeats(); layer++ {
		state, err := t.encodeLayer(tokens, mask, current, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection, captureBindings)
		if err != nil {
			t.releaseEncodedSequenceBindings(encoded)
			return nil, err
		}
		encoded.layers = append(encoded.layers, state)
		current = state.projected
	}
	if len(encoded.layers) == 0 {
		return nil, fmt.Errorf("encoder produced zero layers")
	}
	encoded.pooled = append([]float32(nil), encoded.layers[len(encoded.layers)-1].pooled...)
	return encoded, nil
}

func (t *EmbeddingTrainer) encodeLayer(tokens, mask []int32, input []float32, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection *backend.Tensor, captureBindings bool) (*embeddingSequenceState, error) {
	d := projection.Shape[0]
	h := 0
	if hiddenProjection != nil {
		d = hiddenProjection.Shape[0]
		h = hiddenProjection.Shape[1]
	}
	e := projection.Shape[1]
	state, err := newEmbeddingSequenceState(tokens, mask, input, hiddenProjection, projection)
	if err != nil {
		return nil, err
	}
	if captureBindings {
		state.inputBinding = t.bindSequenceTensor(state, "input", tensorF32View([]int{len(tokens), d}, state.input), true, false)
	}
	if attentionQuery != nil && attentionKey != nil && attentionValue != nil && attentionOutput != nil {
		q, ok := t.tryForwardWeightMatMul(state.input, len(tokens), d, t.attnQParam.Name, attentionQuery, d)
		if ok {
			copy(state.attnQ, q)
		} else {
			attnQData := forwardMatMulHostData(attentionQuery)
			fillHostMatMul(state.input, len(tokens), d, attnQData, d, state.attnQ)
		}
		if captureBindings {
			state.attnQBinding = t.bindSequenceTensor(state, "q", tensorF32View([]int{len(tokens), d}, state.attnQ), true, false)
		}
		k, ok := t.tryForwardWeightMatMul(state.input, len(tokens), d, t.attnKParam.Name, attentionKey, d)
		if ok {
			copy(state.attnK, k)
		} else {
			attnKData := forwardMatMulHostData(attentionKey)
			fillHostMatMul(state.input, len(tokens), d, attnKData, d, state.attnK)
		}
		if captureBindings {
			state.attnKBinding = t.bindSequenceTensor(state, "k", tensorF32View([]int{len(tokens), d}, state.attnK), true, false)
		}
		v, ok := t.tryForwardWeightMatMul(state.input, len(tokens), d, t.attnVParam.Name, attentionValue, d)
		if ok {
			copy(state.attnV, v)
		} else {
			attnVData := forwardMatMulHostData(attentionValue)
			fillHostMatMul(state.input, len(tokens), d, attnVData, d, state.attnV)
		}
		if captureBindings {
			state.attnVBinding = t.bindSequenceTensor(state, "v", tensorF32View([]int{len(tokens), d}, state.attnV), true, false)
		}
		var (
			scores   []float32
			mixed    []float32
			matmulOK bool
		)
		if captureBindings {
			scores, matmulOK = t.tryTrainerMatMulBoundRight(state.attnQ, len(tokens), d, state.attnKBinding, tensorF32View([]int{len(tokens), d}, state.attnK), false, true)
		} else {
			scores, matmulOK = t.tryTrainerMatMul(state.attnQ, len(tokens), d, state.attnK, len(tokens), d, false, true)
		}
		if matmulOK {
			copy(state.attnScores, scores)
		} else {
			kt := transpose2DData(state.attnK, len(tokens), d)
			fillHostMatMul(state.attnQ, len(tokens), d, kt, len(tokens), state.attnScores)
		}
		scaleFloat32Slice(state.attnScores, t.attentionScoreScale(d))
		softmaxAttentionScoresInPlace(state.attnScores, len(tokens), mask, t.manifest.AttentionMaskMode)
		if captureBindings {
			state.attnScoresBinding = t.bindSequenceTensor(state, "scores", tensorF32View([]int{len(tokens), len(tokens)}, state.attnScores), true, t.softmaxBackwardAccelEnabled())
		}
		if captureBindings {
			mixed, matmulOK = t.tryTrainerMatMulBoundRight(state.attnScores, len(tokens), len(tokens), state.attnVBinding, tensorF32View([]int{len(tokens), d}, state.attnV), false, false)
		} else {
			mixed, matmulOK = t.tryTrainerMatMul(state.attnScores, len(tokens), len(tokens), state.attnV, len(tokens), d, false, false)
		}
		if matmulOK {
			copy(state.attnMixed, mixed)
		} else {
			fillHostMatMul(state.attnScores, len(tokens), len(tokens), state.attnV, d, state.attnMixed)
		}
		if captureBindings {
			state.attnMixedBinding = t.bindSequenceTensor(state, "mixed", tensorF32View([]int{len(tokens), d}, state.attnMixed), true, false)
		}
		output, ok := t.tryForwardWeightMatMul(state.attnMixed, len(tokens), d, t.attnOParam.Name, attentionOutput, d)
		if ok {
			copy(state.attnOutput, output)
		} else {
			attnOData := forwardMatMulHostData(attentionOutput)
			fillHostMatMul(state.attnMixed, len(tokens), d, attnOData, d, state.attnOutput)
		}
		if t.attentionResidualEnabled() || t.attentionLayerNormEnabled() {
			for i := range state.attnOutput {
				value := state.attnOutput[i]
				if t.attentionResidualEnabled() {
					value += state.input[i]
				}
				state.attnResidual[i] = value
			}
			if t.attentionLayerNormEnabled() {
				for row := range tokens {
					base := row * d
					layerNormRow(state.hidden[base:base+d], state.attnResidual[base:base+d])
				}
				if captureBindings {
					state.attnResidualBinding = t.bindSequenceTensor(state, "attn_residual", tensorF32View([]int{len(tokens), d}, state.attnResidual), false, t.fullActivationBackwardAccelEnabled())
				}
			} else {
				copy(state.hidden, state.attnResidual)
			}
		} else {
			copy(state.hidden, state.attnOutput)
		}
	} else {
		copy(state.hidden, state.input)
	}
	if captureBindings {
		state.hiddenBinding = t.bindSequenceTensor(state, "hidden", tensorF32View([]int{len(tokens), d}, state.hidden), true, t.fullActivationBackwardAccelEnabled() && t.attentionLayerNormEnabled())
	}
	if hiddenProjection != nil {
		ffnHidden, ok := t.tryForwardWeightMatMul(state.hidden, len(tokens), d, t.hiddenParam.Name, hiddenProjection, h)
		if ok {
			copy(state.ffnHidden, ffnHidden)
		} else {
			hiddenData := forwardMatMulHostData(hiddenProjection)
			fillHostMatMul(state.hidden, len(tokens), d, hiddenData, h, state.ffnHidden)
		}
		if captureBindings {
			state.ffnHiddenBinding = t.bindSequenceTensor(state, "ffn_hidden", tensorF32View([]int{len(tokens), h}, state.ffnHidden), false, t.fullActivationBackwardAccelEnabled())
		}
		fillGELUForward(state.activated, state.ffnHidden, fastGELUEnabled())
		if captureBindings {
			state.activatedBinding = t.bindSequenceTensor(state, "activated", tensorF32View([]int{len(tokens), h}, state.activated), true, false)
		}
		projected, ok := t.tryForwardWeightMatMul(state.activated, len(tokens), h, t.projParam.Name, projection, e)
		if ok {
			copy(state.ffnOutput, projected)
		} else {
			projData := forwardMatMulHostData(projection)
			fillHostMatMul(state.activated, len(tokens), h, projData, e, state.ffnOutput)
		}
		if t.ffnResidualEnabled() || t.ffnLayerNormEnabled() {
			for i := range state.ffnOutput {
				value := state.ffnOutput[i]
				if t.ffnResidualEnabled() {
					value += state.hidden[i]
				}
				state.ffnResidual[i] = value
			}
			if t.ffnLayerNormEnabled() {
				for row := range tokens {
					base := row * e
					layerNormRow(state.projected[base:base+e], state.ffnResidual[base:base+e])
				}
				if captureBindings {
					state.ffnResidualBinding = t.bindSequenceTensor(state, "ffn_residual", tensorF32View([]int{len(tokens), e}, state.ffnResidual), false, t.fullActivationBackwardAccelEnabled())
					state.projectedBinding = t.bindSequenceTensor(state, "projected", tensorF32View([]int{len(tokens), e}, state.projected), false, t.fullActivationBackwardAccelEnabled())
				}
			} else {
				copy(state.projected, state.ffnResidual)
			}
		} else {
			copy(state.projected, state.ffnOutput)
		}
	} else {
		projected, ok := t.tryForwardWeightMatMul(state.hidden, len(tokens), d, t.projParam.Name, projection, e)
		if ok {
			copy(state.projected, projected)
		} else {
			projData := forwardMatMulHostData(projection)
			fillHostMatMul(state.hidden, len(tokens), d, projData, e, state.projected)
		}
	}
	for row := range tokens {
		projectedBase := row * e
		norm := vectorNorm(state.projected[projectedBase : projectedBase+e])
		if norm == 0 {
			copy(state.normalized[projectedBase:projectedBase+e], state.projected[projectedBase:projectedBase+e])
		} else {
			for col := 0; col < e; col++ {
				state.normalized[projectedBase+col] = state.projected[projectedBase+col] / norm
			}
		}
		if mask[row] == 0 {
			continue
		}
		state.activeCount++
		for col := 0; col < e; col++ {
			state.pooled[col] += state.normalized[projectedBase+col]
		}
	}
	if state.activeCount == 0 {
		return nil, fmt.Errorf("sequence mask selects zero tokens")
	}
	inv := 1 / float32(state.activeCount)
	for i := range state.pooled {
		state.pooled[i] *= inv
	}
	return state, nil
}

func (t *EmbeddingTrainer) encodeCompactSequence(tokens, mask []int32, role int32, forward *compactEmbeddingForwardWeights) (*embeddingEncodedSequence, error) {
	if forward == nil {
		return nil, fmt.Errorf("missing compact forward weights")
	}
	if forward.token == nil || len(forward.token.Shape) != 2 {
		return nil, fmt.Errorf("compact token embedding must be rank-2")
	}
	d := forward.token.Shape[1]
	input, err := embeddingInputForTokens(forward.token, tokens)
	if err != nil {
		return nil, err
	}
	if err := addRoleEmbeddingToInput(input, forward.role, role, len(tokens)); err != nil {
		return nil, err
	}
	if err := applyEmbeddingPositionEncoding(input, len(tokens), d, t.manifest.PositionEncoding); err != nil {
		return nil, err
	}
	encoded := &embeddingEncodedSequence{
		layers: make([]*embeddingSequenceState, 0, len(forward.layers)),
		tokens: append([]int32(nil), tokens...),
		role:   role,
	}
	current := input
	for i, layer := range forward.layers {
		state, err := t.encodeCompactLayer(tokens, mask, current, layer)
		if err != nil {
			return nil, fmt.Errorf("compact layer %d: %w", i, err)
		}
		encoded.layers = append(encoded.layers, state)
		current = state.projected
	}
	if len(encoded.layers) == 0 {
		return nil, fmt.Errorf("encoder produced zero layers")
	}
	outputRows, err := t.compactFinalOutputRows(current, len(tokens), d, forward.outputProjection)
	if err != nil {
		return nil, err
	}
	pooled, active, err := meanPoolRows(outputRows, len(tokens), compactOutputWidth(d, forward.outputProjection), mask)
	if err != nil {
		return nil, err
	}
	last := encoded.layers[len(encoded.layers)-1]
	last.activeCount = active
	last.pooled = append(last.pooled[:0], pooled...)
	encoded.pooled = append([]float32(nil), pooled...)
	return encoded, nil
}

func (t *EmbeddingTrainer) encodeCompactLayer(tokens, mask []int32, input []float32, layer compactEmbeddingForwardLayer) (*embeddingSequenceState, error) {
	d := compactLayerModelDim(layer)
	h := compactLayerFFNDim(layer)
	if d <= 0 || h <= 0 {
		return nil, fmt.Errorf("invalid compact layer dimensions")
	}
	if len(input) != len(tokens)*d {
		return nil, fmt.Errorf("compact layer input size %d does not match tokens=%d width=%d", len(input), len(tokens), d)
	}
	if err := validateCompactForwardLayer(layer, d, h); err != nil {
		return nil, err
	}
	heads, headDim, err := validateCompactLayerAttentionLayout(layer, d)
	if err != nil {
		return nil, err
	}
	state := &embeddingSequenceState{
		tokens:       append([]int32(nil), tokens...),
		mask:         append([]int32(nil), mask...),
		input:        input,
		hidden:       make([]float32, len(tokens)*d),
		attnQ:        make([]float32, len(tokens)*d),
		attnK:        make([]float32, len(tokens)*d),
		attnV:        make([]float32, len(tokens)*d),
		attnScores:   make([]float32, heads*len(tokens)*len(tokens)),
		attnMixed:    make([]float32, len(tokens)*d),
		attnOutput:   make([]float32, len(tokens)*d),
		attnResidual: make([]float32, len(tokens)*d),
		ffnHidden:    make([]float32, len(tokens)*h),
		activated:    make([]float32, len(tokens)*h),
		ffnOutput:    make([]float32, len(tokens)*d),
		ffnResidual:  make([]float32, len(tokens)*d),
		projected:    make([]float32, len(tokens)*d),
		normalized:   make([]float32, len(tokens)*d),
		pooled:       make([]float32, d),
	}
	if !t.fillCompactForwardQKVMatMul(state, len(tokens), d, layer) {
		if out, ok := t.tryForwardWeightMatMul(input, len(tokens), d, layer.attnQName, layer.attnQ, d); ok {
			copy(state.attnQ, out)
		} else {
			fillHostMatMul(input, len(tokens), d, forwardMatMulHostData(layer.attnQ), d, state.attnQ)
		}
		if out, ok := t.tryForwardWeightMatMul(input, len(tokens), d, layer.attnKName, layer.attnK, d); ok {
			copy(state.attnK, out)
		} else {
			fillHostMatMul(input, len(tokens), d, forwardMatMulHostData(layer.attnK), d, state.attnK)
		}
		if out, ok := t.tryForwardWeightMatMul(input, len(tokens), d, layer.attnVName, layer.attnV, d); ok {
			copy(state.attnV, out)
		} else {
			fillHostMatMul(input, len(tokens), d, forwardMatMulHostData(layer.attnV), d, state.attnV)
		}
	}
	fillCompactMultiHeadAttention(state, len(tokens), d, heads, headDim, mask)
	if out, ok := t.tryForwardWeightMatMul(state.attnMixed, len(tokens), d, layer.attnOName, layer.attnO, d); ok {
		copy(state.attnOutput, out)
	} else {
		fillHostMatMul(state.attnMixed, len(tokens), d, forwardMatMulHostData(layer.attnO), d, state.attnOutput)
	}
	for i := range state.attnOutput {
		state.attnResidual[i] = state.attnOutput[i] + input[i]
	}
	for row := range tokens {
		base := row * d
		layerNormRow(state.hidden[base:base+d], state.attnResidual[base:base+d])
	}
	if out, ok := t.tryForwardWeightMatMul(state.hidden, len(tokens), d, layer.ffnUpName, layer.ffnUp, h); ok {
		copy(state.ffnHidden, out)
	} else {
		fillHostMatMul(state.hidden, len(tokens), d, forwardMatMulHostData(layer.ffnUp), h, state.ffnHidden)
	}
	fillGELUForward(state.activated, state.ffnHidden, fastGELUEnabled())
	if out, ok := t.tryForwardWeightMatMul(state.activated, len(tokens), h, layer.ffnDownName, layer.ffnDown, d); ok {
		copy(state.ffnOutput, out)
	} else {
		fillHostMatMul(state.activated, len(tokens), h, forwardMatMulHostData(layer.ffnDown), d, state.ffnOutput)
	}
	for i := range state.ffnOutput {
		state.ffnResidual[i] = state.ffnOutput[i] + state.hidden[i]
	}
	for row := range tokens {
		base := row * d
		layerNormRow(state.projected[base:base+d], state.ffnResidual[base:base+d])
	}
	return state, nil
}

func (t *EmbeddingTrainer) fillCompactForwardQKVMatMul(state *embeddingSequenceState, rows, width int, layer compactEmbeddingForwardLayer) bool {
	if t == nil || t.forwardMatMul == nil || state == nil || rows == 0 || width == 0 || !qkvMultiBoundRightEnabled() {
		return false
	}
	multi, ok := t.forwardMatMul.(backend.MultiBoundRightMatMulAccelerator)
	if !ok {
		return false
	}
	if layer.attnQName == "" || layer.attnKName == "" || layer.attnVName == "" {
		return false
	}
	for _, tensor := range []*backend.Tensor{layer.attnQ, layer.attnK, layer.attnV} {
		if tensor == nil || len(tensor.Shape) != 2 || tensor.Shape[0] != width || tensor.Shape[1] != width {
			return false
		}
	}
	perMatrix := rows * width
	if len(state.input) != perMatrix || len(state.attnQ) != perMatrix || len(state.attnK) != perMatrix || len(state.attnV) != perMatrix {
		return false
	}
	results, err := multi.RunMatMulWithBoundRights(
		tensorF32View([]int{rows, width}, state.input),
		[]string{layer.attnQName, layer.attnKName, layer.attnVName},
		trainerF32TensorValueType(),
		false,
		false,
	)
	if err != nil || len(results) != 3 {
		return false
	}
	targets := [][]float32{state.attnQ, state.attnK, state.attnV}
	names := []string{layer.attnQName, layer.attnKName, layer.attnVName}
	for i, result := range results {
		if len(result.Outputs) != 1 || result.Outputs[0] == nil || len(result.Outputs[0].F32) != perMatrix {
			return false
		}
		if !matMulResultMatchesRightBinding(result, names[i]) {
			return false
		}
		copy(targets[i], result.Outputs[0].F32)
	}
	return true
}

func matMulResultMatchesRightBinding(result backend.StepDispatchResult, name string) bool {
	if name == "" || result.Metadata == nil {
		return true
	}
	got, ok := result.Metadata["rhs_binding"]
	if !ok {
		return true
	}
	binding, ok := got.(string)
	return ok && binding == name
}

func compactLayerModelDim(layer compactEmbeddingForwardLayer) int {
	if layer.attnQ != nil && len(layer.attnQ.Shape) == 2 {
		return layer.attnQ.Shape[0]
	}
	return 0
}

func compactLayerFFNDim(layer compactEmbeddingForwardLayer) int {
	if layer.ffnUp != nil && len(layer.ffnUp.Shape) == 2 {
		return layer.ffnUp.Shape[1]
	}
	return 0
}

func validateCompactForwardLayer(layer compactEmbeddingForwardLayer, d, h int) error {
	for name, tensor := range map[string]*backend.Tensor{
		"attn_q": layer.attnQ,
		"attn_k": layer.attnK,
		"attn_v": layer.attnV,
		"attn_o": layer.attnO,
	} {
		if tensor == nil || len(tensor.Shape) != 2 || tensor.Shape[0] != d || tensor.Shape[1] != d {
			return fmt.Errorf("%s shape %v, want [%d %d]", name, tensorShapeForError(tensor), d, d)
		}
	}
	if layer.ffnUp == nil || len(layer.ffnUp.Shape) != 2 || layer.ffnUp.Shape[0] != d || layer.ffnUp.Shape[1] != h {
		return fmt.Errorf("ffn_up shape %v, want [%d %d]", tensorShapeForError(layer.ffnUp), d, h)
	}
	if layer.ffnDown == nil || len(layer.ffnDown.Shape) != 2 || layer.ffnDown.Shape[0] != h || layer.ffnDown.Shape[1] != d {
		return fmt.Errorf("ffn_down shape %v, want [%d %d]", tensorShapeForError(layer.ffnDown), h, d)
	}
	return nil
}

func validateCompactLayerAttentionLayout(layer compactEmbeddingForwardLayer, modelDim int) (int, int, error) {
	heads := layer.attentionHeads
	if heads <= 0 {
		heads = 1
	}
	headDim := layer.headDim
	if headDim <= 0 && modelDim%heads == 0 {
		headDim = modelDim / heads
	}
	if heads <= 0 || headDim <= 0 || heads*headDim != modelDim {
		return 0, 0, fmt.Errorf("attention_heads=%d head_dim=%d do not match model_dim=%d", heads, headDim, modelDim)
	}
	return heads, headDim, nil
}

func compactLayerAttentionLayout(layer compactEmbeddingForwardLayer, modelDim int) (int, int, bool) {
	heads, headDim, err := validateCompactLayerAttentionLayout(layer, modelDim)
	return heads, headDim, err == nil
}

func fillCompactMultiHeadAttention(state *embeddingSequenceState, seqLen, modelDim, heads, headDim int, mask []int32) {
	scale := float32(1 / math.Sqrt(float64(headDim)))
	for head := 0; head < heads; head++ {
		headOffset := head * headDim
		scoreBase := head * seqLen * seqLen
		scores := state.attnScores[scoreBase : scoreBase+seqLen*seqLen]
		for query := 0; query < seqLen; query++ {
			queryBase := query*modelDim + headOffset
			scoreRowBase := query * seqLen
			for key := 0; key < seqLen; key++ {
				keyBase := key*modelDim + headOffset
				sum := float32(0)
				for col := 0; col < headDim; col++ {
					sum += state.attnQ[queryBase+col] * state.attnK[keyBase+col]
				}
				scores[scoreRowBase+key] = sum * scale
			}
		}
		softmaxAttentionScoresInPlace(scores, seqLen, mask, EmbeddingAttentionMaskModeKey)
		for query := 0; query < seqLen; query++ {
			queryMixedBase := query*modelDim + headOffset
			scoreRowBase := query * seqLen
			for col := 0; col < headDim; col++ {
				sum := float32(0)
				for key := 0; key < seqLen; key++ {
					sum += scores[scoreRowBase+key] * state.attnV[key*modelDim+headOffset+col]
				}
				state.attnMixed[queryMixedBase+col] = sum
			}
		}
	}
}

func tensorShapeForError(t *backend.Tensor) []int {
	if t == nil {
		return nil
	}
	return t.Shape
}

func compactOutputWidth(modelDim int, outputProjection *backend.Tensor) int {
	if outputProjection != nil && len(outputProjection.Shape) == 2 {
		return outputProjection.Shape[1]
	}
	return modelDim
}

func (t *EmbeddingTrainer) compactFinalOutputRows(hidden []float32, rows, modelDim int, outputProjection *backend.Tensor) ([]float32, error) {
	if len(hidden) != rows*modelDim {
		return nil, fmt.Errorf("compact final hidden size %d does not match rows=%d width=%d", len(hidden), rows, modelDim)
	}
	normalized := make([]float32, len(hidden))
	for row := 0; row < rows; row++ {
		base := row * modelDim
		norm := vectorNorm(hidden[base : base+modelDim])
		if norm == 0 {
			copy(normalized[base:base+modelDim], hidden[base:base+modelDim])
		} else {
			for col := 0; col < modelDim; col++ {
				normalized[base+col] = hidden[base+col] / norm
			}
		}
	}
	if outputProjection == nil {
		return normalized, nil
	}
	if len(outputProjection.Shape) != 2 || outputProjection.Shape[0] != modelDim {
		return nil, fmt.Errorf("output_projection shape %v, want [%d O]", outputProjection.Shape, modelDim)
	}
	out := make([]float32, rows*outputProjection.Shape[1])
	outputProjectionName := ""
	if t != nil && t.compactForwardCache != nil && t.compactForwardCache.outputProjection == outputProjection {
		outputProjectionName = t.compactForwardCache.outputProjectionName
	}
	if accelerated, ok := t.tryForwardWeightMatMul(normalized, rows, modelDim, outputProjectionName, outputProjection, outputProjection.Shape[1]); ok {
		copy(out, accelerated)
	} else {
		fillHostMatMul(normalized, rows, modelDim, forwardMatMulHostData(outputProjection), outputProjection.Shape[1], out)
	}
	return out, nil
}

func meanPoolRows(rowsData []float32, rows, width int, mask []int32) ([]float32, int, error) {
	if len(rowsData) != rows*width {
		return nil, 0, fmt.Errorf("pool rows size %d does not match rows=%d width=%d", len(rowsData), rows, width)
	}
	if len(mask) != rows {
		return nil, 0, fmt.Errorf("mask length %d does not match token length %d", len(mask), rows)
	}
	pooled := make([]float32, width)
	active := 0
	for row := 0; row < rows; row++ {
		if mask[row] == 0 {
			continue
		}
		active++
		base := row * width
		for col := 0; col < width; col++ {
			pooled[col] += rowsData[base+col]
		}
	}
	if active == 0 {
		return nil, 0, fmt.Errorf("sequence mask selects zero tokens")
	}
	inv := 1 / float32(active)
	for i := range pooled {
		pooled[i] *= inv
	}
	return pooled, active, nil
}

func embeddingInputForTokens(tokenEmbed *backend.Tensor, tokens []int32) ([]float32, error) {
	if tokenEmbed == nil || len(tokenEmbed.Shape) != 2 {
		return nil, fmt.Errorf("token embedding must be rank-2")
	}
	d := tokenEmbed.Shape[1]
	input := make([]float32, len(tokens)*d)
	for row, tok := range tokens {
		if tok < 0 || int(tok) >= tokenEmbed.Shape[0] {
			return nil, fmt.Errorf("token %d is out of range", tok)
		}
		copy(input[row*d:(row+1)*d], tokenEmbed.F32[int(tok)*d:(int(tok)+1)*d])
	}
	return input, nil
}

func addRoleEmbeddingToInput(input []float32, roleEmbed *backend.Tensor, role int32, rows int) error {
	if roleEmbed == nil {
		return nil
	}
	if len(roleEmbed.Shape) != 2 {
		return fmt.Errorf("role embedding rank = %d, want 2", len(roleEmbed.Shape))
	}
	d := roleEmbed.Shape[1]
	if rows*d != len(input) {
		return fmt.Errorf("role embedding width %d does not match input shape rows=%d values=%d", d, rows, len(input))
	}
	if role < 0 || int(role) >= roleEmbed.Shape[0] {
		return fmt.Errorf("role index %d is outside role embedding rows %d", role, roleEmbed.Shape[0])
	}
	roleBase := int(role) * d
	for row := 0; row < rows; row++ {
		base := row * d
		for col := 0; col < d; col++ {
			input[base+col] += roleEmbed.F32[roleBase+col]
		}
	}
	return nil
}

func applyEmbeddingPositionEncoding(input []float32, rows, width int, mode string) error {
	switch mode {
	case "", EmbeddingPositionEncodingNone:
		return nil
	case EmbeddingPositionEncodingRoPE:
		applyRoPEToRowsInPlace(input, rows, width)
		return nil
	default:
		return fmt.Errorf("unsupported position_encoding %q", mode)
	}
}

func applyRoPEToRowsInPlace(data []float32, rows, cols int) {
	if rows <= 0 || cols <= 1 {
		return
	}
	for row := 0; row < rows; row++ {
		base := row * cols
		for col := 0; col+1 < cols; col += 2 {
			theta := float64(row) / math.Pow(10000, float64(col)/float64(cols))
			cosTheta := float32(math.Cos(theta))
			sinTheta := float32(math.Sin(theta))
			x0 := data[base+col]
			x1 := data[base+col+1]
			data[base+col] = x0*cosTheta - x1*sinTheta
			data[base+col+1] = x0*sinTheta + x1*cosTheta
		}
	}
}

func applyRoPETransposeToRowsInPlace(data []float32, rows, cols int) {
	if rows <= 0 || cols <= 1 {
		return
	}
	for row := 0; row < rows; row++ {
		base := row * cols
		for col := 0; col+1 < cols; col += 2 {
			theta := float64(row) / math.Pow(10000, float64(col)/float64(cols))
			cosTheta := float32(math.Cos(theta))
			sinTheta := float32(math.Sin(theta))
			x0 := data[base+col]
			x1 := data[base+col+1]
			data[base+col] = x0*cosTheta + x1*sinTheta
			data[base+col+1] = -x0*sinTheta + x1*cosTheta
		}
	}
}

func newEmbeddingSequenceState(tokens, mask []int32, input []float32, hiddenProjection, projection *backend.Tensor) (*embeddingSequenceState, error) {
	if projection == nil || len(projection.Shape) != 2 {
		return nil, fmt.Errorf("projection must be rank-2")
	}
	d := stateWidth(hiddenProjection, projection)
	if len(input) != len(tokens)*d {
		return nil, fmt.Errorf("encoder layer input size %d does not match tokens=%d width=%d", len(input), len(tokens), d)
	}
	h := 0
	if hiddenProjection != nil {
		h = hiddenProjection.Shape[1]
	}
	e := projection.Shape[1]
	state := &embeddingSequenceState{
		tokens:       append([]int32(nil), tokens...),
		mask:         append([]int32(nil), mask...),
		input:        input,
		hidden:       make([]float32, len(tokens)*d),
		attnQ:        make([]float32, len(tokens)*d),
		attnK:        make([]float32, len(tokens)*d),
		attnV:        make([]float32, len(tokens)*d),
		attnScores:   make([]float32, len(tokens)*len(tokens)),
		attnMixed:    make([]float32, len(tokens)*d),
		attnOutput:   make([]float32, len(tokens)*d),
		attnResidual: make([]float32, len(tokens)*d),
		ffnHidden:    make([]float32, len(tokens)*h),
		activated:    make([]float32, len(tokens)*h),
		ffnOutput:    make([]float32, len(tokens)*e),
		ffnResidual:  make([]float32, len(tokens)*e),
		projected:    make([]float32, len(tokens)*e),
		normalized:   make([]float32, len(tokens)*e),
		pooled:       make([]float32, e),
		activeCount:  0,
	}
	return state, nil
}

func stateWidth(hiddenProjection, projection *backend.Tensor) int {
	if hiddenProjection != nil {
		return hiddenProjection.Shape[0]
	}
	return projection.Shape[0]
}

func tensorF32View(shape []int, data []float32) *backend.Tensor {
	return &backend.Tensor{
		DType: "f32",
		Shape: append([]int(nil), shape...),
		F32:   data,
	}
}

func (t *EmbeddingTrainer) scratchFloat32(slot, elements int) []float32 {
	if elements <= 0 {
		return nil
	}
	if t == nil {
		return make([]float32, elements)
	}
	for len(t.scratchF32) <= slot {
		t.scratchF32 = append(t.scratchF32, nil)
	}
	buf := t.scratchF32[slot]
	if cap(buf) < elements {
		buf = make([]float32, elements)
	} else {
		buf = buf[:elements]
	}
	t.scratchF32[slot] = buf
	return buf
}

func (t *EmbeddingTrainer) flattenFixedFloat32MatricesScratch(slot int, matrices [][]float32, perMatrix int) ([]float32, bool) {
	if len(matrices) == 0 || perMatrix <= 0 {
		return nil, false
	}
	if out, ok := contiguousFixedFloat32Matrices(matrices, perMatrix); ok {
		return out, true
	}
	out := t.scratchFloat32(slot, len(matrices)*perMatrix)
	if !flattenFixedFloat32MatricesInto(out, matrices, perMatrix) {
		return nil, false
	}
	return out, true
}

func flattenFixedFloat32Matrices(matrices [][]float32, perMatrix int) ([]float32, bool) {
	if len(matrices) == 0 || perMatrix <= 0 {
		return nil, false
	}
	if out, ok := contiguousFixedFloat32Matrices(matrices, perMatrix); ok {
		return out, true
	}
	out := make([]float32, len(matrices)*perMatrix)
	if !flattenFixedFloat32MatricesInto(out, matrices, perMatrix) {
		return nil, false
	}
	return out, true
}

func contiguousFixedFloat32Matrices(matrices [][]float32, perMatrix int) ([]float32, bool) {
	if len(matrices) == 0 || perMatrix <= 0 {
		return nil, false
	}
	total := len(matrices) * perMatrix
	first := matrices[0]
	if len(first) != perMatrix || cap(first) < total {
		return nil, false
	}
	out := first[:total]
	for i, matrix := range matrices {
		if len(matrix) != perMatrix || &out[i*perMatrix] != &matrix[0] {
			return nil, false
		}
	}
	return out, true
}

func flattenFixedFloat32MatricesInto(out []float32, matrices [][]float32, perMatrix int) bool {
	if len(out) != len(matrices)*perMatrix {
		return false
	}
	for i, matrix := range matrices {
		if len(matrix) != perMatrix {
			return false
		}
		copy(out[i*perMatrix:(i+1)*perMatrix], matrix)
	}
	return true
}

func splitFloat32Views(out []float32, parts int) ([][]float32, bool) {
	if parts <= 0 || len(out)%parts != 0 {
		return nil, false
	}
	perMatrix := len(out) / parts
	result := make([][]float32, parts)
	for i := range result {
		result[i] = out[i*perMatrix : (i+1)*perMatrix]
	}
	return result, true
}

func (t *EmbeddingTrainer) tryForwardMatMul(lhsData []float32, rows, inner int, rhs *backend.Tensor, cols int) ([]float32, bool) {
	if rhs == nil {
		return nil, false
	}
	return t.tryTrainerMatMul(lhsData, rows, inner, rhs.F32, rhs.Shape[0], rhs.Shape[1], false, false)
}

func (t *EmbeddingTrainer) tryForwardWeightMatMul(lhsData []float32, rows, inner int, rhsName string, rhs *backend.Tensor, cols int) ([]float32, bool) {
	if rhs == nil {
		return nil, false
	}
	return t.tryTrainerMatMulBoundRight(lhsData, rows, inner, rhsName, rhs, false, false)
}

func (t *EmbeddingTrainer) tryTrainerMatMul(lhsData []float32, lhsRows, lhsCols int, rhsData []float32, rhsRows, rhsCols int, transposeLeft, transposeRight bool) ([]float32, bool) {
	if t == nil || t.forwardMatMul == nil || lhsRows == 0 || lhsCols == 0 || rhsRows == 0 || rhsCols == 0 {
		return nil, false
	}
	outRows, outCols, ok := trainerMatMulShape(lhsRows, lhsCols, rhsRows, rhsCols, transposeLeft, transposeRight)
	if !ok {
		return nil, false
	}
	result, err := t.forwardMatMul.RunMatMulWithTranspose(
		[]*backend.Tensor{
			tensorF32View([]int{lhsRows, lhsCols}, lhsData),
			tensorF32View([]int{rhsRows, rhsCols}, rhsData),
		},
		eosartifact.ValueType{
			Kind: eosartifact.ValueTensor,
			Tensor: &eosartifact.TensorType{
				DType: "f32",
			},
		},
		transposeLeft,
		transposeRight,
	)
	if err != nil || len(result.Outputs) != 1 || result.Outputs[0] == nil {
		return nil, false
	}
	out := result.Outputs[0].F32
	if len(out) != outRows*outCols {
		return nil, false
	}
	return out, true
}

func (t *EmbeddingTrainer) tryTrainerBatchedMatMul(lhsMatrices [][]float32, lhsRows, lhsCols int, rhsMatrices [][]float32, rhsRows, rhsCols int) ([][]float32, bool) {
	return t.tryTrainerBatchedMatMulTranspose(lhsMatrices, lhsRows, lhsCols, rhsMatrices, rhsRows, rhsCols, false, false)
}

func (t *EmbeddingTrainer) tryTrainerBatchedMatMulTranspose(lhsMatrices [][]float32, lhsRows, lhsCols int, rhsMatrices [][]float32, rhsRows, rhsCols int, transposeLeft, transposeRight bool) ([][]float32, bool) {
	if t == nil || t.forwardMatMul == nil || len(lhsMatrices) == 0 || len(lhsMatrices) != len(rhsMatrices) || lhsRows == 0 || lhsCols == 0 || rhsRows == 0 || rhsCols == 0 {
		return nil, false
	}
	outRows, outCols, ok := trainerMatMulShape(lhsRows, lhsCols, rhsRows, rhsCols, transposeLeft, transposeRight)
	if !ok {
		return nil, false
	}
	batches := len(lhsMatrices)
	lhsBatch, ok := t.flattenFixedFloat32MatricesScratch(0, lhsMatrices, lhsRows*lhsCols)
	if !ok {
		return nil, false
	}
	rhsBatch, ok := t.flattenFixedFloat32MatricesScratch(1, rhsMatrices, rhsRows*rhsCols)
	if !ok {
		return nil, false
	}
	result, err := t.forwardMatMul.RunMatMulWithTranspose(
		[]*backend.Tensor{
			tensorF32View([]int{batches, lhsRows, lhsCols}, lhsBatch),
			tensorF32View([]int{batches, rhsRows, rhsCols}, rhsBatch),
		},
		eosartifact.ValueType{
			Kind: eosartifact.ValueTensor,
			Tensor: &eosartifact.TensorType{
				DType: "f32",
			},
		},
		transposeLeft,
		transposeRight,
	)
	if err != nil || len(result.Outputs) != 1 || result.Outputs[0] == nil {
		return nil, false
	}
	perMatrix := outRows * outCols
	out := result.Outputs[0].F32
	if len(out) != batches*perMatrix {
		return nil, false
	}
	return splitFloat32Views(out, batches)
}

func (t *EmbeddingTrainer) tryTrainerMatMulBoundLeft(lhsName string, lhs, rhs *backend.Tensor, transposeLeft, transposeRight bool) ([]float32, bool) {
	if lhs == nil || rhs == nil {
		return nil, false
	}
	if t != nil && t.forwardMatMul != nil && lhsName != "" {
		outRows, outCols, ok := trainerMatMulShape(lhs.Shape[0], lhs.Shape[1], rhs.Shape[0], rhs.Shape[1], transposeLeft, transposeRight)
		if ok {
			result, err := t.forwardMatMul.RunMatMulWithBoundLeft(
				lhsName,
				rhs,
				eosartifact.ValueType{
					Kind: eosartifact.ValueTensor,
					Tensor: &eosartifact.TensorType{
						DType: "f32",
					},
				},
				transposeLeft,
				transposeRight,
			)
			if err == nil && len(result.Outputs) == 1 && result.Outputs[0] != nil {
				out := result.Outputs[0].F32
				if len(out) == outRows*outCols {
					return out, true
				}
			}
		}
	}
	return t.tryTrainerMatMul(lhs.F32, lhs.Shape[0], lhs.Shape[1], rhs.F32, rhs.Shape[0], rhs.Shape[1], transposeLeft, transposeRight)
}

func (t *EmbeddingTrainer) tryTrainerMatMulBoundRight(lhsData []float32, lhsRows, lhsCols int, rhsName string, rhs *backend.Tensor, transposeLeft, transposeRight bool) ([]float32, bool) {
	if rhs == nil {
		return nil, false
	}
	if t != nil && t.forwardMatMul != nil && rhsName != "" {
		outRows, outCols, ok := trainerMatMulShape(lhsRows, lhsCols, rhs.Shape[0], rhs.Shape[1], transposeLeft, transposeRight)
		if ok {
			result, err := t.forwardMatMul.RunMatMulWithBoundRight(
				tensorF32View([]int{lhsRows, lhsCols}, lhsData),
				rhsName,
				eosartifact.ValueType{
					Kind: eosartifact.ValueTensor,
					Tensor: &eosartifact.TensorType{
						DType: "f32",
					},
				},
				transposeLeft,
				transposeRight,
			)
			if err == nil && len(result.Outputs) == 1 && result.Outputs[0] != nil {
				out := result.Outputs[0].F32
				if len(out) == outRows*outCols {
					return out, true
				}
			}
		}
	}
	return t.tryTrainerMatMul(lhsData, lhsRows, lhsCols, rhs.F32, rhs.Shape[0], rhs.Shape[1], transposeLeft, transposeRight)
}

func fillHostMatMul(lhs []float32, rows, inner int, rhs []float32, cols int, out []float32) {
	fillHostMatMulTranspose(lhs, rows, inner, rhs, inner, cols, false, false, out)
}

func fillHostMatMulTranspose(lhs []float32, lhsRows, lhsCols int, rhs []float32, rhsRows, rhsCols int, transposeLeft, transposeRight bool, out []float32) {
	rows, inner, cols, ok := trainerMatMulDims(lhsRows, lhsCols, rhsRows, rhsCols, transposeLeft, transposeRight)
	if !ok {
		for i := range out {
			out[i] = 0
		}
		return
	}
	for row := 0; row < rows; row++ {
		outBase := row * cols
		for col := 0; col < cols; col++ {
			sum := float32(0)
			for kk := 0; kk < inner; kk++ {
				sum += trainerMatMulAt(lhs, lhsRows, lhsCols, row, kk, transposeLeft) * trainerMatMulAt(rhs, rhsRows, rhsCols, kk, col, transposeRight)
			}
			out[outBase+col] = sum
		}
	}
}

func addFloat32Slice(dst, src []float32) {
	limit := len(dst)
	if len(src) < limit {
		limit = len(src)
	}
	for i := 0; i < limit; i++ {
		dst[i] += src[i]
	}
}

func trainerMatMulShape(lhsRows, lhsCols, rhsRows, rhsCols int, transposeLeft, transposeRight bool) (rows, cols int, ok bool) {
	rows, _, cols, ok = trainerMatMulDims(lhsRows, lhsCols, rhsRows, rhsCols, transposeLeft, transposeRight)
	return rows, cols, ok
}

func trainerMatMulDims(lhsRows, lhsCols, rhsRows, rhsCols int, transposeLeft, transposeRight bool) (rows, inner, cols int, ok bool) {
	rows = lhsRows
	inner = lhsCols
	if transposeLeft {
		rows, inner = lhsCols, lhsRows
	}
	rhsInner := rhsRows
	cols = rhsCols
	if transposeRight {
		rhsInner, cols = rhsCols, rhsRows
	}
	if rows <= 0 || inner <= 0 || rhsInner <= 0 || cols <= 0 || inner != rhsInner {
		return 0, 0, 0, false
	}
	return rows, inner, cols, true
}

func trainerMatMulAt(data []float32, rows, cols, row, col int, transpose bool) float32 {
	if transpose {
		return data[col*cols+row]
	}
	return data[row*cols+col]
}

func requireTrainableEmbeddingParam(mod *eosartifact.Module, name string) (eosartifact.Param, error) {
	for _, param := range mod.Params {
		if param.Name != name {
			continue
		}
		if param.Type.Kind != eosartifact.ValueTensor || param.Type.Tensor == nil {
			return eosartifact.Param{}, fmt.Errorf("param %q is not a tensor", name)
		}
		if len(param.Type.Tensor.Shape) != 2 {
			return eosartifact.Param{}, fmt.Errorf("param %q rank = %d, want 2", name, len(param.Type.Tensor.Shape))
		}
		if !param.Trainable {
			return eosartifact.Param{}, fmt.Errorf("param %q is not marked @trainable", name)
		}
		return param, nil
	}
	return eosartifact.Param{}, fmt.Errorf("missing param %q", name)
}

func optionalTrainableEmbeddingParam(mod *eosartifact.Module, name string) (eosartifact.Param, bool, error) {
	if name == "" {
		return eosartifact.Param{}, false, nil
	}
	param, err := requireTrainableEmbeddingParam(mod, name)
	if err != nil {
		return eosartifact.Param{}, false, err
	}
	return param, true, nil
}

func normalizedTrainConfig(cfg EmbeddingTrainConfig, params ...eosartifact.Param) EmbeddingTrainConfig {
	if cfg.LearningRate == 0 {
		cfg.LearningRate = 0.05
	}
	if cfg.WeightBits == 0 {
		for _, param := range params {
			cfg.WeightBits = maxInt(cfg.WeightBits, paramQuantBits(param))
		}
	}
	if cfg.Optimizer == "" {
		cfg.Optimizer = "adamw"
	}
	if cfg.Beta1 == 0 {
		cfg.Beta1 = 0.9
	}
	if cfg.Beta2 == 0 {
		cfg.Beta2 = 0.999
	}
	if cfg.Epsilon == 0 {
		cfg.Epsilon = 1e-8
	}
	if cfg.ContrastiveLoss == "" {
		cfg.ContrastiveLoss = "pair_mse"
	}
	if cfg.Temperature == 0 {
		cfg.Temperature = 0.05
	}
	if cfg.TeacherTemperature == 0 {
		cfg.TeacherTemperature = 1
	}
	cfg = normalizedScoreSpectrumTrainConfig(cfg)
	cfg.TeacherSourceTemperatures = normalizeHardNegativeTeacherTemperatures(cfg.TeacherSourceTemperatures)
	cfg.TeacherSourceWeights = normalizeHardNegativeTeacherWeights(cfg.TeacherSourceWeights)
	cfg.GroupedLossWeight = effectiveGroupedLossWeight(cfg.ContrastiveLoss, cfg.GroupedLossWeight)
	if dims, weights, err := normalizeMatryoshkaDimsAndWeights(cfg.MatryoshkaDims, cfg.MatryoshkaWeights, 0); err == nil {
		cfg.MatryoshkaDims = dims
		cfg.MatryoshkaWeights = weights
	}
	if bits, err := normalizeTurboQuantPrefixBits(cfg.TurboQuantPrefixBits); err == nil {
		cfg.TurboQuantPrefixBits = bits
	}
	if objectives, err := normalizeTurboQuantPrefixObjectives(cfg.TurboQuantPrefixObjectives, cfg.MatryoshkaDims, 0); err == nil {
		cfg.TurboQuantPrefixObjectives = objectives
	}
	if objectives, err := normalizeTurboQuantPrefixObjectives(cfg.TurboQuantRankMarginObjectives, cfg.MatryoshkaDims, 0); err == nil {
		cfg.TurboQuantRankMarginObjectives = objectives
	}
	if objectives, err := normalizeTurboQuantPrefixObjectives(cfg.TurboQuantCompactObjectives, cfg.MatryoshkaDims, 0); err == nil {
		cfg.TurboQuantCompactObjectives = objectives
	}
	if len(cfg.TurboQuantRankMarginObjectives) > 0 && cfg.TurboQuantRankMargin == 0 {
		cfg.TurboQuantRankMargin = effectiveTurboQuantRankMargin(cfg.TurboQuantRankMargin)
	}
	if len(cfg.TurboQuantPrefixBits) > 0 || len(cfg.TurboQuantPrefixObjectives) > 0 || len(cfg.TurboQuantCompactObjectives) > 0 || len(cfg.TurboQuantRankMarginObjectives) > 0 {
		if cfg.TurboQuantPrefixWeight == 0 {
			if len(cfg.TurboQuantPrefixBits) > 0 {
				cfg.TurboQuantPrefixWeight = 1
			}
		}
		cfg.TurboQuantPrefixSeed = effectiveTurboQuantPrefixSeed(cfg.TurboQuantPrefixSeed)
	}
	if len(cfg.TurboQuantPrefixBits) > 0 || len(cfg.TurboQuantPrefixObjectives) > 0 || len(cfg.TurboQuantRankMarginObjectives) > 0 {
		if mode, err := normalizeTurboQuantPrefixScoreMode(cfg.TurboQuantPrefixScoreMode); err == nil {
			cfg.TurboQuantPrefixScoreMode = mode
		}
	}
	return cfg
}

func normalizedScoreSpectrumTrainConfig(cfg EmbeddingTrainConfig) EmbeddingTrainConfig {
	mode, err := normalizeScoreSpectrumLossMode(cfg.ScoreSpectrumLossMode)
	if err == nil {
		cfg.ScoreSpectrumLossMode = mode
	}
	if cfg.ScoreSpectrumRecoveryWeight == 0 && scoreSpectrumLossModeIncludesRecovery(cfg.ScoreSpectrumLossMode) {
		cfg.ScoreSpectrumRecoveryWeight = 1
	}
	if cfg.ScoreSpectrumRecoveryTopK == 0 {
		cfg.ScoreSpectrumRecoveryTopK = 4
	}
	if cfg.ScoreSpectrumRecoveryTau == 0 {
		if cfg.Temperature > 0 {
			cfg.ScoreSpectrumRecoveryTau = cfg.Temperature
		} else {
			cfg.ScoreSpectrumRecoveryTau = 0.05
		}
	}
	return cfg
}

func normalizeScoreSpectrumLossMode(mode string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(strings.ReplaceAll(mode, "-", "_"))) {
	case "", ScoreSpectrumLossModeHardSoft:
		return ScoreSpectrumLossModeHardSoft, nil
	case ScoreSpectrumLossModeRecovery:
		return ScoreSpectrumLossModeRecovery, nil
	case ScoreSpectrumLossModeHardSoftRecovery:
		return ScoreSpectrumLossModeHardSoftRecovery, nil
	default:
		return "", fmt.Errorf("unsupported score_spectrum_loss_mode %q (supported: %s, %s, %s)", mode, ScoreSpectrumLossModeHardSoft, ScoreSpectrumLossModeRecovery, ScoreSpectrumLossModeHardSoftRecovery)
	}
}

func NormalizeScoreSpectrumLossModeForCLI(mode string) (string, error) {
	return normalizeScoreSpectrumLossMode(mode)
}

func scoreSpectrumLossModeIncludesHardSoft(mode string) bool {
	mode, err := normalizeScoreSpectrumLossMode(mode)
	if err != nil {
		return false
	}
	return mode == ScoreSpectrumLossModeHardSoft || mode == ScoreSpectrumLossModeHardSoftRecovery
}

func scoreSpectrumLossModeIncludesRecovery(mode string) bool {
	mode, err := normalizeScoreSpectrumLossMode(mode)
	if err != nil {
		return false
	}
	return mode == ScoreSpectrumLossModeRecovery || mode == ScoreSpectrumLossModeHardSoftRecovery
}

func normalizeMatryoshkaTrainConfig(cfg EmbeddingTrainConfig, embeddingDim int) (EmbeddingTrainConfig, error) {
	dims, weights, err := normalizeMatryoshkaDimsAndWeights(cfg.MatryoshkaDims, cfg.MatryoshkaWeights, embeddingDim)
	if err != nil {
		return cfg, err
	}
	cfg.MatryoshkaDims = dims
	cfg.MatryoshkaWeights = weights
	bits, err := normalizeTurboQuantPrefixBits(cfg.TurboQuantPrefixBits)
	if err != nil {
		return cfg, err
	}
	cfg.TurboQuantPrefixBits = bits
	objectives, err := normalizeTurboQuantPrefixObjectives(cfg.TurboQuantPrefixObjectives, cfg.MatryoshkaDims, embeddingDim)
	if err != nil {
		return cfg, err
	}
	cfg.TurboQuantPrefixObjectives = objectives
	rankObjectives, err := normalizeTurboQuantPrefixObjectives(cfg.TurboQuantRankMarginObjectives, cfg.MatryoshkaDims, embeddingDim)
	if err != nil {
		return cfg, err
	}
	cfg.TurboQuantRankMarginObjectives = rankObjectives
	compactObjectives, err := normalizeTurboQuantPrefixObjectives(cfg.TurboQuantCompactObjectives, cfg.MatryoshkaDims, embeddingDim)
	if err != nil {
		return cfg, err
	}
	cfg.TurboQuantCompactObjectives = compactObjectives
	if len(cfg.TurboQuantPrefixBits) > 0 && len(cfg.TurboQuantPrefixObjectives) > 0 {
		return cfg, fmt.Errorf("turboquant_prefix_objectives is mutually exclusive with turboquant_prefix_bits")
	}
	if len(cfg.TurboQuantPrefixObjectives) > 0 && cfg.TurboQuantPrefixWeight != 0 {
		return cfg, fmt.Errorf("turboquant_prefix_weight must be zero when turboquant_prefix_objectives is set")
	}
	if len(cfg.TurboQuantRankMarginObjectives) > 0 && cfg.TurboQuantRankMargin == 0 {
		cfg.TurboQuantRankMargin = effectiveTurboQuantRankMargin(cfg.TurboQuantRankMargin)
	}
	if len(cfg.TurboQuantPrefixBits) > 0 || len(cfg.TurboQuantPrefixObjectives) > 0 || len(cfg.TurboQuantCompactObjectives) > 0 || len(cfg.TurboQuantRankMarginObjectives) > 0 {
		if cfg.TurboQuantPrefixWeight == 0 {
			if len(cfg.TurboQuantPrefixBits) > 0 {
				cfg.TurboQuantPrefixWeight = 1
			}
		}
		cfg.TurboQuantPrefixSeed = effectiveTurboQuantPrefixSeed(cfg.TurboQuantPrefixSeed)
	}
	if len(cfg.TurboQuantPrefixBits) > 0 || len(cfg.TurboQuantPrefixObjectives) > 0 || len(cfg.TurboQuantRankMarginObjectives) > 0 {
		mode, err := normalizeTurboQuantPrefixScoreMode(cfg.TurboQuantPrefixScoreMode)
		if err != nil {
			return cfg, err
		}
		cfg.TurboQuantPrefixScoreMode = mode
	}
	return cfg, nil
}

func normalizeMatryoshkaDimsAndWeights(dims []int, weights []float32, embeddingDim int) ([]int, []float32, error) {
	if len(dims) == 0 {
		return nil, nil, nil
	}
	if len(weights) != 0 && len(weights) != len(dims) {
		return nil, nil, fmt.Errorf("matryoshka_weights length %d must match matryoshka_dims length %d", len(weights), len(dims))
	}
	type weightedDim struct {
		dim    int
		weight float32
	}
	items := make([]weightedDim, 0, len(dims))
	seen := map[int]bool{}
	for i, dim := range dims {
		if dim <= 0 {
			return nil, nil, fmt.Errorf("matryoshka_dims[%d] must be positive", i)
		}
		if embeddingDim > 0 && dim > embeddingDim {
			return nil, nil, fmt.Errorf("matryoshka_dims[%d]=%d exceeds embedding dimension %d", i, dim, embeddingDim)
		}
		if embeddingDim > 0 && dim == embeddingDim {
			continue
		}
		if seen[dim] {
			continue
		}
		weight := float32(1)
		if len(weights) > 0 {
			weight = weights[i]
		}
		if weight <= 0 || math.IsNaN(float64(weight)) || math.IsInf(float64(weight), 0) {
			return nil, nil, fmt.Errorf("matryoshka_weights[%d] must be finite and positive", i)
		}
		seen[dim] = true
		items = append(items, weightedDim{dim: dim, weight: weight})
	}
	sort.Slice(items, func(i, j int) bool { return items[i].dim < items[j].dim })
	outDims := make([]int, 0, len(items))
	outWeights := make([]float32, 0, len(items))
	for _, item := range items {
		outDims = append(outDims, item.dim)
		outWeights = append(outWeights, item.weight)
	}
	if len(outDims) == 0 {
		return nil, nil, nil
	}
	return outDims, outWeights, nil
}

func trainerEmbeddingDim(t *EmbeddingTrainer) int {
	if t == nil || t.projection == nil || len(t.projection.Shape) != 2 {
		return 0
	}
	return t.projection.Shape[1]
}

func matryoshkaPairMultiplier(dims []int) int {
	if len(dims) == 0 {
		return 1
	}
	return 1 + len(dims)
}

func compactPrefixObjectiveMultiplier(cfg EmbeddingTrainRunConfig) int {
	multiplier := 1 + len(cfg.MatryoshkaDims)
	if len(cfg.TurboQuantPrefixObjectives) > 0 {
		for _, objective := range cfg.TurboQuantPrefixObjectives {
			if objective.Weight > 0 {
				multiplier++
			}
		}
	} else if len(cfg.TurboQuantPrefixBits) > 0 {
		multiplier += len(cfg.MatryoshkaDims) * len(cfg.TurboQuantPrefixBits)
	}
	return multiplier
}

func turboQuantRankMarginObjectiveCount(objectives []TurboQuantPrefixObjective) int {
	count := 0
	for _, objective := range objectives {
		if objective.Weight > 0 {
			count++
		}
	}
	return count
}

func matryoshkaWeightSum(weights []float32) float32 {
	if len(weights) == 0 {
		return 0
	}
	sum := float32(0)
	for _, weight := range weights {
		if weight > 0 {
			sum += weight
		}
	}
	return sum
}

func normalizeTurboQuantPrefixBits(bits []int) ([]int, error) {
	if len(bits) == 0 {
		return nil, nil
	}
	out := make([]int, 0, len(bits))
	seen := map[int]bool{}
	for i, bitWidth := range bits {
		if bitWidth < 2 || bitWidth > 8 {
			return nil, fmt.Errorf("turboquant_prefix_bits[%d]=%d outside supported range 2..8", i, bitWidth)
		}
		if seen[bitWidth] {
			continue
		}
		seen[bitWidth] = true
		out = append(out, bitWidth)
	}
	sort.Ints(out)
	return out, nil
}

// ParseTurboQuantPrefixObjectives parses CLI/manifest compact-prefix objectives.
func ParseTurboQuantPrefixObjectives(raw string) ([]TurboQuantPrefixObjective, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	parts := strings.Split(raw, ",")
	objectives := make([]TurboQuantPrefixObjective, 0, len(parts))
	seen := map[[2]int]bool{}
	for i, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			return nil, fmt.Errorf("turboquant_prefix_objectives[%d] is blank", i)
		}
		dimBit, weightRaw, ok := strings.Cut(part, "=")
		if !ok || strings.TrimSpace(dimBit) == "" || strings.TrimSpace(weightRaw) == "" {
			return nil, fmt.Errorf("turboquant_prefix_objectives[%d] must use dim:bit=weight syntax", i)
		}
		dimRaw, bitRaw, ok := strings.Cut(dimBit, ":")
		if !ok || strings.TrimSpace(dimRaw) == "" || strings.TrimSpace(bitRaw) == "" {
			return nil, fmt.Errorf("turboquant_prefix_objectives[%d] must use dim:bit=weight syntax", i)
		}
		dim, err := strconv.Atoi(strings.TrimSpace(dimRaw))
		if err != nil {
			return nil, fmt.Errorf("parse turboquant_prefix_objectives[%d] dim %q: %w", i, strings.TrimSpace(dimRaw), err)
		}
		if dim <= 0 {
			return nil, fmt.Errorf("turboquant_prefix_objectives[%d] dim must be positive", i)
		}
		bitWidth, err := strconv.Atoi(strings.TrimSpace(bitRaw))
		if err != nil {
			return nil, fmt.Errorf("parse turboquant_prefix_objectives[%d] bit %q: %w", i, strings.TrimSpace(bitRaw), err)
		}
		if bitWidth < 2 || bitWidth > 8 {
			return nil, fmt.Errorf("turboquant_prefix_objectives[%d] bit_width=%d outside supported range 2..8", i, bitWidth)
		}
		weight64, err := strconv.ParseFloat(strings.TrimSpace(weightRaw), 32)
		if err != nil {
			return nil, fmt.Errorf("parse turboquant_prefix_objectives[%d] weight %q: %w", i, strings.TrimSpace(weightRaw), err)
		}
		if weight64 < 0 || math.IsNaN(weight64) || math.IsInf(weight64, 0) {
			return nil, fmt.Errorf("turboquant_prefix_objectives[%d] weight must be finite and non-negative", i)
		}
		key := [2]int{dim, bitWidth}
		if seen[key] {
			return nil, fmt.Errorf("duplicate turboquant_prefix_objectives entry for dim=%d bit_width=%d", dim, bitWidth)
		}
		seen[key] = true
		objectives = append(objectives, TurboQuantPrefixObjective{Dim: dim, BitWidth: bitWidth, Weight: float32(weight64)})
	}
	return normalizeTurboQuantPrefixObjectiveOrder(objectives), nil
}

func FormatTurboQuantPrefixObjectives(objectives []TurboQuantPrefixObjective) string {
	if len(objectives) == 0 {
		return ""
	}
	objectives = normalizeTurboQuantPrefixObjectiveOrder(objectives)
	parts := make([]string, 0, len(objectives))
	for _, objective := range objectives {
		parts = append(parts, strconv.Itoa(objective.Dim)+":"+strconv.Itoa(objective.BitWidth)+"="+strconv.FormatFloat(float64(objective.Weight), 'g', -1, 32))
	}
	return strings.Join(parts, ",")
}

func normalizeTurboQuantPrefixObjectiveOrder(objectives []TurboQuantPrefixObjective) []TurboQuantPrefixObjective {
	out := append([]TurboQuantPrefixObjective(nil), objectives...)
	sort.Slice(out, func(i, j int) bool {
		if out[i].Dim != out[j].Dim {
			return out[i].Dim < out[j].Dim
		}
		return out[i].BitWidth < out[j].BitWidth
	})
	return out
}

func normalizeTurboQuantPrefixObjectives(objectives []TurboQuantPrefixObjective, matryoshkaDims []int, embeddingDim int) ([]TurboQuantPrefixObjective, error) {
	if len(objectives) == 0 {
		return nil, nil
	}
	allowedDims := map[int]bool{}
	for _, dim := range matryoshkaDims {
		allowedDims[dim] = true
	}
	if embeddingDim > 0 {
		allowedDims[embeddingDim] = true
	}
	seen := map[[2]int]bool{}
	out := make([]TurboQuantPrefixObjective, 0, len(objectives))
	for i, objective := range objectives {
		if objective.Dim <= 0 {
			return nil, fmt.Errorf("turboquant_prefix_objectives[%d] dim must be positive", i)
		}
		if objective.Dim < 2 {
			return nil, fmt.Errorf("turboquant_prefix_objectives[%d] dim=%d must be at least 2", i, objective.Dim)
		}
		if len(allowedDims) > 0 && !allowedDims[objective.Dim] {
			return nil, fmt.Errorf("turboquant_prefix_objectives[%d] dim=%d must exist in matryoshka_dims", i, objective.Dim)
		}
		if objective.BitWidth < 2 || objective.BitWidth > 8 {
			return nil, fmt.Errorf("turboquant_prefix_objectives[%d] bit_width=%d outside supported range 2..8", i, objective.BitWidth)
		}
		if objective.Weight < 0 || math.IsNaN(float64(objective.Weight)) || math.IsInf(float64(objective.Weight), 0) {
			return nil, fmt.Errorf("turboquant_prefix_objectives[%d] weight must be finite and non-negative", i)
		}
		key := [2]int{objective.Dim, objective.BitWidth}
		if seen[key] {
			return nil, fmt.Errorf("duplicate turboquant_prefix_objectives entry for dim=%d bit_width=%d", objective.Dim, objective.BitWidth)
		}
		seen[key] = true
		out = append(out, objective)
	}
	return normalizeTurboQuantPrefixObjectiveOrder(out), nil
}

func turboQuantPrefixObjectivesForConfig(cfg EmbeddingTrainConfig) []TurboQuantPrefixObjective {
	if len(cfg.TurboQuantPrefixObjectives) > 0 {
		return append([]TurboQuantPrefixObjective(nil), cfg.TurboQuantPrefixObjectives...)
	}
	if len(cfg.MatryoshkaDims) == 0 || len(cfg.TurboQuantPrefixBits) == 0 {
		return nil
	}
	weight := effectiveTurboQuantPrefixWeight(cfg)
	if weight <= 0 {
		return nil
	}
	objectives := make([]TurboQuantPrefixObjective, 0, len(cfg.MatryoshkaDims)*len(cfg.TurboQuantPrefixBits))
	for _, dim := range cfg.MatryoshkaDims {
		for _, bitWidth := range cfg.TurboQuantPrefixBits {
			objectives = append(objectives, TurboQuantPrefixObjective{Dim: dim, BitWidth: bitWidth, Weight: weight})
		}
	}
	return objectives
}

func turboQuantRankMarginObjectivesForConfig(cfg EmbeddingTrainConfig) []TurboQuantPrefixObjective {
	if len(cfg.TurboQuantRankMarginObjectives) == 0 {
		return nil
	}
	return append([]TurboQuantPrefixObjective(nil), cfg.TurboQuantRankMarginObjectives...)
}

func turboQuantCompactObjectivesForConfig(cfg EmbeddingTrainConfig) []TurboQuantPrefixObjective {
	if len(cfg.TurboQuantCompactObjectives) == 0 {
		return nil
	}
	return append([]TurboQuantPrefixObjective(nil), cfg.TurboQuantCompactObjectives...)
}

func effectiveTurboQuantRankMargin(margin float32) float32 {
	if margin == 0 {
		return 0.02
	}
	return margin
}

func effectiveTurboQuantPrefixWeight(cfg EmbeddingTrainConfig) float32 {
	if len(cfg.TurboQuantPrefixBits) == 0 {
		return 0
	}
	if cfg.TurboQuantPrefixWeight <= 0 {
		return 1
	}
	return cfg.TurboQuantPrefixWeight
}

func effectiveTurboQuantPrefixSeed(seed int64) int64 {
	if seed == 0 {
		return DefaultTurboQuantMultiVectorQuantizerSeed
	}
	return seed
}

func normalizeTurboQuantPrefixScoreMode(mode string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(strings.ReplaceAll(mode, "-", "_"))) {
	case "":
		return TurboQuantPrefixScoreModeReconstructCosine, nil
	case TurboQuantPrefixScoreModeReconstructCosine:
		return TurboQuantPrefixScoreModeReconstructCosine, nil
	case TurboQuantPrefixScoreModePreparedIP:
		return TurboQuantPrefixScoreModePreparedIP, nil
	default:
		return "", fmt.Errorf("unsupported turboquant_prefix_score_mode %q (supported: %s, %s; CLI accepts prepared-ip)", mode, TurboQuantPrefixScoreModeReconstructCosine, TurboQuantPrefixScoreModePreparedIP)
	}
}

func NormalizeTurboQuantPrefixScoreModeForCLI(mode string) (string, error) {
	return normalizeTurboQuantPrefixScoreMode(mode)
}

func turboQuantPrefixWeightSum(cfg EmbeddingTrainConfig) float32 {
	if len(cfg.TurboQuantPrefixObjectives) > 0 {
		sum := float32(0)
		for _, objective := range cfg.TurboQuantPrefixObjectives {
			if objective.Weight > 0 {
				sum += objective.Weight
			}
		}
		return sum
	}
	weight := effectiveTurboQuantPrefixWeight(cfg)
	if weight <= 0 || len(cfg.MatryoshkaDims) == 0 || len(cfg.TurboQuantPrefixBits) == 0 {
		return 0
	}
	return weight * float32(len(cfg.MatryoshkaDims)*len(cfg.TurboQuantPrefixBits))
}

func turboQuantRankMarginWeightSum(cfg EmbeddingTrainConfig) float32 {
	sum := float32(0)
	for _, objective := range turboQuantRankMarginObjectivesForConfig(cfg) {
		if objective.Weight > 0 {
			sum += objective.Weight
		}
	}
	return sum
}

func turboQuantCompactWeightSum(cfg EmbeddingTrainConfig) float32 {
	sum := float32(0)
	for _, objective := range turboQuantCompactObjectivesForConfig(cfg) {
		if objective.Weight > 0 {
			sum += objective.Weight
		}
	}
	return sum
}

func validateTrainConfig(cfg EmbeddingTrainConfig) error {
	cfg = normalizedScoreSpectrumTrainConfig(cfg)
	switch cfg.ContrastiveLoss {
	case "pair_mse", "infonce", "grouped_infonce", "hybrid_infonce":
	default:
		return fmt.Errorf("unsupported contrastive_loss %q", cfg.ContrastiveLoss)
	}
	if cfg.Temperature <= 0 {
		return fmt.Errorf("temperature must be positive")
	}
	if cfg.GroupedLossWeight < 0 {
		return fmt.Errorf("grouped_loss_weight must be non-negative")
	}
	if cfg.TeacherLossWeight < 0 {
		return fmt.Errorf("teacher_loss_weight must be non-negative")
	}
	if cfg.TeacherTemperature <= 0 {
		return fmt.Errorf("teacher_temperature must be positive")
	}
	for source, temp := range cfg.TeacherSourceTemperatures {
		if strings.TrimSpace(source) == "" {
			return fmt.Errorf("teacher_source_temperatures has an empty source")
		}
		if temp <= 0 {
			return fmt.Errorf("teacher_source_temperatures[%s] must be positive", source)
		}
	}
	for source, weight := range cfg.TeacherSourceWeights {
		if strings.TrimSpace(source) == "" {
			return fmt.Errorf("teacher_source_weights has an empty source")
		}
		if weight < 0 || math.IsNaN(float64(weight)) || math.IsInf(float64(weight), 0) {
			return fmt.Errorf("teacher_source_weights[%s] must be finite and non-negative", source)
		}
	}
	if len(cfg.MatryoshkaWeights) != 0 && len(cfg.MatryoshkaWeights) != len(cfg.MatryoshkaDims) {
		return fmt.Errorf("matryoshka_weights length %d must match matryoshka_dims length %d", len(cfg.MatryoshkaWeights), len(cfg.MatryoshkaDims))
	}
	for i, dim := range cfg.MatryoshkaDims {
		if dim <= 0 {
			return fmt.Errorf("matryoshka_dims[%d] must be positive", i)
		}
		if i > 0 && dim <= cfg.MatryoshkaDims[i-1] {
			return fmt.Errorf("matryoshka_dims must be strictly increasing")
		}
	}
	for i, weight := range cfg.MatryoshkaWeights {
		if weight <= 0 || math.IsNaN(float64(weight)) || math.IsInf(float64(weight), 0) {
			return fmt.Errorf("matryoshka_weights[%d] must be finite and positive", i)
		}
	}
	if len(cfg.TurboQuantPrefixBits) > 0 && len(cfg.TurboQuantPrefixObjectives) > 0 {
		return fmt.Errorf("turboquant_prefix_objectives is mutually exclusive with turboquant_prefix_bits")
	}
	if len(cfg.TurboQuantPrefixObjectives) > 0 && cfg.TurboQuantPrefixWeight != 0 {
		return fmt.Errorf("turboquant_prefix_weight must be zero when turboquant_prefix_objectives is set")
	}
	if len(cfg.TurboQuantPrefixBits) > 0 {
		if len(cfg.MatryoshkaDims) == 0 {
			return fmt.Errorf("turboquant compact-prefix objectives require matryoshka_dims")
		}
		for i, dim := range cfg.MatryoshkaDims {
			if dim < 2 {
				return fmt.Errorf("matryoshka_dims[%d]=%d must be at least 2 when turboquant compact-prefix objectives are enabled", i, dim)
			}
		}
	}
	if len(cfg.TurboQuantPrefixBits) > 0 {
		for i, bitWidth := range cfg.TurboQuantPrefixBits {
			if bitWidth < 2 || bitWidth > 8 {
				return fmt.Errorf("turboquant_prefix_bits[%d]=%d outside supported range 2..8", i, bitWidth)
			}
		}
		if cfg.TurboQuantPrefixWeight < 0 || math.IsNaN(float64(cfg.TurboQuantPrefixWeight)) || math.IsInf(float64(cfg.TurboQuantPrefixWeight), 0) {
			return fmt.Errorf("turboquant_prefix_weight must be finite and non-negative")
		}
		if _, err := normalizeTurboQuantPrefixScoreMode(cfg.TurboQuantPrefixScoreMode); err != nil {
			return err
		}
	}
	if len(cfg.TurboQuantPrefixObjectives) > 0 {
		if _, err := normalizeTurboQuantPrefixObjectives(cfg.TurboQuantPrefixObjectives, cfg.MatryoshkaDims, 0); err != nil {
			return err
		}
		if _, err := normalizeTurboQuantPrefixScoreMode(cfg.TurboQuantPrefixScoreMode); err != nil {
			return err
		}
	}
	if len(cfg.TurboQuantRankMarginObjectives) > 0 {
		if _, err := normalizeTurboQuantPrefixObjectives(cfg.TurboQuantRankMarginObjectives, cfg.MatryoshkaDims, 0); err != nil {
			return err
		}
		if cfg.TurboQuantRankMargin < 0 || math.IsNaN(float64(cfg.TurboQuantRankMargin)) || math.IsInf(float64(cfg.TurboQuantRankMargin), 0) {
			return fmt.Errorf("turboquant_rank_margin must be finite and non-negative")
		}
		if _, err := normalizeTurboQuantPrefixScoreMode(cfg.TurboQuantPrefixScoreMode); err != nil {
			return err
		}
	}
	if len(cfg.TurboQuantCompactObjectives) > 0 {
		if _, err := normalizeTurboQuantPrefixObjectives(cfg.TurboQuantCompactObjectives, cfg.MatryoshkaDims, 0); err != nil {
			return err
		}
		if _, err := normalizeTurboQuantPrefixScoreMode(TurboQuantPrefixScoreModePreparedIP); err != nil {
			return err
		}
	}
	if err := validateScoreSpectrumRecoveryConfig(cfg.ScoreSpectrumLossMode, cfg.ScoreSpectrumRecoveryWeight, cfg.ScoreSpectrumRecoveryMargin, cfg.ScoreSpectrumRecoveryTopK, cfg.ScoreSpectrumRecoveryTau); err != nil {
		return err
	}
	return nil
}

func validateScoreSpectrumRecoveryConfig(mode string, weight, margin float32, topK int, tau float32) error {
	normalizedMode, err := normalizeScoreSpectrumLossMode(mode)
	if err != nil {
		return err
	}
	if weight < 0 || math.IsNaN(float64(weight)) || math.IsInf(float64(weight), 0) {
		return fmt.Errorf("score_spectrum_recovery_weight must be finite and non-negative")
	}
	if margin < 0 || math.IsNaN(float64(margin)) || math.IsInf(float64(margin), 0) {
		return fmt.Errorf("score_spectrum_recovery_margin must be finite and non-negative")
	}
	if topK <= 0 {
		return fmt.Errorf("score_spectrum_recovery_top_k must be positive")
	}
	if tau <= 0 || math.IsNaN(float64(tau)) || math.IsInf(float64(tau), 0) {
		return fmt.Errorf("score_spectrum_recovery_tau must be finite and positive")
	}
	if scoreSpectrumLossModeIncludesRecovery(normalizedMode) && weight == 0 {
		return fmt.Errorf("score_spectrum_recovery_weight must be positive when recovery loss is enabled")
	}
	return nil
}

func embeddingUsesInfoNCELoss(loss string) bool {
	return loss == "infonce" || loss == "grouped_infonce" || loss == "hybrid_infonce"
}

func effectiveGroupedLossWeight(loss string, weight float32) float32 {
	switch loss {
	case "hybrid_infonce":
		if weight <= 0 {
			return 0.25
		}
		return weight
	case "grouped_infonce":
		if weight <= 0 {
			return 1
		}
		return weight
	default:
		return 0
	}
}

func paramQuantBits(param eosartifact.Param) int {
	if param.Type.Tensor == nil {
		return 0
	}
	switch param.Type.Tensor.DType {
	case "q4":
		return 4
	case "q8":
		return 8
	default:
		return 0
	}
}

func tensorAsMasterF32(t *backend.Tensor) *backend.Tensor {
	if t == nil {
		return nil
	}
	return backend.NewTensorF32(t.Shape, append([]float32(nil), t.F32...))
}

func zeroLikeMaster(t *backend.Tensor) *backend.Tensor {
	if t == nil {
		return nil
	}
	return backend.NewTensorF32(t.Shape, make([]float32, len(t.F32)))
}

func forwardTensorForParam(param eosartifact.Param, master *backend.Tensor, bits int) *backend.Tensor {
	return refreshForwardTensorForParam(param, master, bits, nil)
}

func refreshForwardMatMulTensorForParam(param eosartifact.Param, master *backend.Tensor, dst *backend.Tensor) *backend.Tensor {
	if master == nil {
		return nil
	}
	dtype := "f32"
	if param.Type.Tensor != nil && param.Type.Tensor.DType != "" {
		dtype = param.Type.Tensor.DType
	}
	if dst == nil {
		dst = &backend.Tensor{}
	}
	elements := len(master.F32)
	if len(dst.F32) != elements {
		dst.F32 = make([]float32, elements)
	}
	copy(dst.F32, master.F32)
	dst.DType = dtype
	dst.Shape = append(dst.Shape[:0], master.Shape...)
	dst.I32 = nil
	dst.I64 = nil
	return dst
}

func refreshForwardTensorForParam(param eosartifact.Param, master *backend.Tensor, bits int, dst *backend.Tensor) *backend.Tensor {
	if master == nil {
		return nil
	}
	bits = clampQuantBits(bits, paramQuantBits(param))
	dtype := param.Type.Tensor.DType
	if dtype == "" {
		dtype = "f32"
	}
	if dst == nil {
		dst = &backend.Tensor{}
	}
	elements := len(master.F32)
	if len(dst.F32) != elements {
		dst.F32 = make([]float32, elements)
	}
	copy(dst.F32, master.F32)
	if bits > 0 {
		fakeQuantizeDataInPlace(dst.F32, bits)
	}
	dst.DType = dtype
	dst.Shape = append(dst.Shape[:0], master.Shape...)
	dst.I32 = nil
	dst.I64 = nil
	return dst
}

func forwardMatMulHostData(rhs *backend.Tensor) []float32 {
	if rhs == nil {
		return nil
	}
	data := append([]float32(nil), rhs.F32...)
	switch rhs.DType {
	case "q4":
		return fakeQuantizeData(data, 4)
	case "q8":
		return fakeQuantizeData(data, 8)
	default:
		return data
	}
}

func exportTensorForParam(param eosartifact.Param, master *backend.Tensor) (*backend.Tensor, error) {
	if master == nil {
		return nil, fmt.Errorf("missing master tensor for %q", param.Name)
	}
	data := append([]float32(nil), master.F32...)
	switch param.Type.Tensor.DType {
	case "q4":
		return backend.NewTensorQ4(master.Shape, fakeQuantizeData(data, 4)), nil
	case "q8":
		return backend.NewTensorQ8(master.Shape, fakeQuantizeData(data, 8)), nil
	case "f16":
		return backend.NewTensorF16(master.Shape, data), nil
	case "f32":
		return backend.NewTensorF32(master.Shape, data), nil
	default:
		return nil, fmt.Errorf("unsupported export dtype %q for %q", param.Type.Tensor.DType, param.Name)
	}
}

func clampQuantBits(primary, fallback int) int {
	if primary > 0 {
		return primary
	}
	return fallback
}

func fakeQuantizeData(data []float32, bits int) []float32 {
	if bits <= 0 || len(data) == 0 {
		return data
	}
	fakeQuantizeDataInPlace(data, bits)
	return data
}

func fakeQuantizeDataInPlace(data []float32, bits int) {
	if bits <= 0 || len(data) == 0 {
		return
	}
	maxAbs := float32(0)
	for _, v := range data {
		abs := float32(math.Abs(float64(v)))
		if abs > maxAbs {
			maxAbs = abs
		}
	}
	if maxAbs == 0 {
		return
	}
	levelsInt := (1 << uint(bits-1)) - 1
	levels := float32(levelsInt)
	if levels <= 0 {
		return
	}
	scale := maxAbs / levels
	if scale == 0 {
		return
	}
	for i, v := range data {
		q := float32(math.Round(float64(v / scale)))
		if q > levels {
			q = levels
		}
		if q < -levels {
			q = -levels
		}
		data[i] = q * scale
	}
}

func pooledGradToProjected(state *embeddingSequenceState, gradPooled []float32, e int) []float32 {
	gradProjected := make([]float32, len(state.projected))
	if state == nil || state.activeCount == 0 {
		return gradProjected
	}
	invCount := 1 / float32(state.activeCount)
	for row := range state.tokens {
		if state.mask[row] == 0 {
			continue
		}
		projectedBase := row * e
		norm := vectorNorm(state.projected[projectedBase : projectedBase+e])
		if norm == 0 {
			continue
		}
		dotNG := float32(0)
		for col := 0; col < e; col++ {
			dotNG += state.normalized[projectedBase+col] * gradPooled[col] * invCount
		}
		for col := 0; col < e; col++ {
			gradProjected[projectedBase+col] = (gradPooled[col]*invCount - state.normalized[projectedBase+col]*dotNG) / norm
		}
	}
	return gradProjected
}

func (t *EmbeddingTrainer) backpropProjectedSequence(state *embeddingSequenceState, gradProjected []float32, projection *backend.Tensor, gradProj []float32, d, e int) []float32 {
	gradInput := make([]float32, len(state.hidden))
	if state == nil {
		return gradInput
	}
	seqLen := len(state.tokens)
	gradProjStep := make([]float32, d*e)
	if out, ok := t.tryTrainerMatMulBoundLeft(
		state.hiddenBinding,
		tensorF32View([]int{seqLen, d}, state.hidden),
		tensorF32View([]int{seqLen, e}, gradProjected),
		true,
		false,
	); ok {
		copy(gradProjStep, out)
	} else {
		fillHostMatMulTranspose(state.hidden, seqLen, d, gradProjected, seqLen, e, true, false, gradProjStep)
	}
	addFloat32Slice(gradProj, gradProjStep)
	if out, ok := t.tryTrainerMatMulBoundRight(gradProjected, seqLen, e, t.projParam.Name, projection, false, true); ok {
		copy(gradInput, out)
	} else {
		projData := forwardMatMulHostData(projection)
		fillHostMatMulTranspose(gradProjected, seqLen, e, projData, d, e, false, true, gradInput)
	}
	return gradInput
}

func (t *EmbeddingTrainer) backpropProjectedFFNSequence(state *embeddingSequenceState, gradProjected []float32, hiddenProjection, projection *backend.Tensor, gradHidden, gradProj []float32, d, h, e int) []float32 {
	gradInput := make([]float32, len(state.hidden))
	if state == nil {
		return gradInput
	}
	seqLen := len(state.tokens)
	gradOutputMatrix := make([]float32, seqLen*e)
	for row := range state.tokens {
		if state.mask[row] == 0 {
			continue
		}
		projectedBase := row * e
		gradOutput := gradOutputMatrix[projectedBase : projectedBase+e]
		copy(gradOutput, gradProjected[projectedBase:projectedBase+e])
	}
	if t.ffnLayerNormEnabled() {
		accelerated := false
		if !state.skipUnboundActivationBackward(state.projectedBinding, state.ffnResidualBinding) {
			if out, ok := t.tryLayerNormBackwardRows(gradOutputMatrix, state.projected, state.ffnResidual, seqLen, e, state.projectedBinding, state.ffnResidualBinding); ok {
				copy(gradOutputMatrix, out)
				accelerated = true
			}
		}
		if !accelerated {
			for row := range state.tokens {
				if state.mask[row] == 0 {
					continue
				}
				projectedBase := row * e
				backwardLayerNormRow(
					gradOutputMatrix[projectedBase:projectedBase+e],
					gradOutputMatrix[projectedBase:projectedBase+e],
					state.projected[projectedBase:projectedBase+e],
					state.ffnResidual[projectedBase:projectedBase+e],
				)
			}
		}
	}
	gradProjStep := make([]float32, h*e)
	if out, ok := t.tryTrainerMatMulBoundLeft(
		state.activatedBinding,
		tensorF32View([]int{seqLen, h}, state.activated),
		tensorF32View([]int{seqLen, e}, gradOutputMatrix),
		true,
		false,
	); ok {
		copy(gradProjStep, out)
	} else {
		fillHostMatMulTranspose(state.activated, seqLen, h, gradOutputMatrix, seqLen, e, true, false, gradProjStep)
	}
	addFloat32Slice(gradProj, gradProjStep)
	gradActivatedPre := make([]float32, seqLen*h)
	if out, ok := t.tryTrainerMatMulBoundRight(gradOutputMatrix, seqLen, e, t.projParam.Name, projection, false, true); ok {
		copy(gradActivatedPre, out)
	} else {
		projData := forwardMatMulHostData(projection)
		fillHostMatMulTranspose(gradOutputMatrix, seqLen, e, projData, h, e, false, true, gradActivatedPre)
	}
	gradActivated := make([]float32, seqLen*h)
	acceleratedGELU := false
	fastGELU := fastGELUEnabled()
	if !fastGELU && !state.skipUnboundActivationBackward(state.ffnHiddenBinding) {
		if out, ok := t.tryGELUBackwardMul(gradActivatedPre, state.ffnHidden, seqLen, h, state.ffnHiddenBinding); ok {
			copy(gradActivated, out)
			acceleratedGELU = true
		}
	}
	if !acceleratedGELU {
		for row := range state.tokens {
			if state.mask[row] == 0 {
				continue
			}
			ffnBase := row * h
			fillGELUBackwardMul(gradActivated[ffnBase:ffnBase+h], gradActivatedPre[ffnBase:ffnBase+h], state.ffnHidden[ffnBase:ffnBase+h], fastGELU)
		}
	}
	gradHiddenStep := make([]float32, d*h)
	if out, ok := t.tryTrainerMatMulBoundLeft(
		state.hiddenBinding,
		tensorF32View([]int{seqLen, d}, state.hidden),
		tensorF32View([]int{seqLen, h}, gradActivated),
		true,
		false,
	); ok {
		copy(gradHiddenStep, out)
	} else {
		fillHostMatMulTranspose(state.hidden, seqLen, d, gradActivated, seqLen, h, true, false, gradHiddenStep)
	}
	addFloat32Slice(gradHidden, gradHiddenStep)
	if out, ok := t.tryTrainerMatMulBoundRight(gradActivated, seqLen, h, t.hiddenParam.Name, hiddenProjection, false, true); ok {
		copy(gradInput, out)
	} else {
		hiddenData := forwardMatMulHostData(hiddenProjection)
		fillHostMatMulTranspose(gradActivated, seqLen, h, hiddenData, d, h, false, true, gradInput)
	}
	if t.ffnResidualEnabled() {
		addFloat32Slice(gradInput, gradOutputMatrix)
	}
	return gradInput
}

func (t *EmbeddingTrainer) backpropEncodedSequence(seq *embeddingEncodedSequence, gradPooled []float32, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection *backend.Tensor, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj []float32) []float32 {
	if seq == nil || len(seq.layers) == 0 {
		return nil
	}
	gradProjected := pooledGradToProjected(seq.layers[len(seq.layers)-1], gradPooled, t.projection.Shape[1])
	for layer := len(seq.layers) - 1; layer >= 0; layer-- {
		gradProjected = t.backpropEncodedLayer(seq.layers[layer], gradProjected, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
	}
	return gradProjected
}

type embeddingBackpropItem struct {
	seq           *embeddingEncodedSequence
	gradPooled    []float32
	gradProjected []float32
}

func (t *EmbeddingTrainer) tryBackpropContrastiveBatch(queries, positives []*embeddingEncodedSequence, queryGrads, positiveGrads [][]float32, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection *backend.Tensor, gradToken, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj []float32) bool {
	if t == nil || !batchedBackwardEnabled() || hiddenProjection == nil || projection == nil {
		return false
	}
	if t.roleEmbed != nil {
		return false
	}
	items := make([]embeddingBackpropItem, 0, len(queries)+len(positives))
	for i, seq := range queries {
		if i >= len(queryGrads) || seq == nil || len(seq.layers) == 0 {
			return false
		}
		items = append(items, embeddingBackpropItem{seq: seq, gradPooled: queryGrads[i]})
	}
	for i, seq := range positives {
		if i >= len(positiveGrads) || seq == nil || len(seq.layers) == 0 {
			return false
		}
		items = append(items, embeddingBackpropItem{seq: seq, gradPooled: positiveGrads[i]})
	}
	if len(items) == 0 {
		return false
	}
	layerCount := len(items[0].seq.layers)
	for i := range items {
		if len(items[i].seq.layers) != layerCount {
			return false
		}
		items[i].gradProjected = pooledGradToProjected(items[i].seq.layers[layerCount-1], items[i].gradPooled, t.projection.Shape[1])
	}
	d := t.tokenEmbed.Shape[1]
	h := hiddenProjection.Shape[1]
	e := t.projection.Shape[1]
	for layer := layerCount - 1; layer >= 0; layer-- {
		groups := map[int][]int{}
		lengths := make([]int, 0, len(items))
		for i := range items {
			state := items[i].seq.layers[layer]
			if state == nil || len(state.tokens) == 0 {
				return false
			}
			seqLen := len(state.tokens)
			if _, ok := groups[seqLen]; !ok {
				lengths = append(lengths, seqLen)
			}
			groups[seqLen] = append(groups[seqLen], i)
		}

		for _, seqLen := range lengths {
			indexes := groups[seqLen]
			if len(indexes) == 1 {
				idx := indexes[0]
				items[idx].gradProjected = t.backpropEncodedLayer(items[idx].seq.layers[layer], items[idx].gradProjected, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
				continue
			}

			states := make([]*embeddingSequenceState, len(indexes))
			gradProjected := make([][]float32, len(indexes))
			for groupIndex, itemIndex := range indexes {
				states[groupIndex] = items[itemIndex].seq.layers[layer]
				gradProjected[groupIndex] = items[itemIndex].gradProjected
			}

			gradInput, ok := t.backpropProjectedFFNSequences(states, gradProjected, hiddenProjection, projection, gradHidden, gradProj, d, h, e)
			if !ok {
				for _, idx := range indexes {
					items[idx].gradProjected = t.backpropEncodedLayer(items[idx].seq.layers[layer], items[idx].gradProjected, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
				}
				continue
			}
			if t.attentionEnabled() {
				var ok bool
				gradInput, ok = t.backpropAttentionSequences(states, gradInput, attentionQuery, attentionKey, attentionValue, attentionOutput, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, d)
				if !ok {
					for _, idx := range indexes {
						items[idx].gradProjected = t.backpropEncodedLayer(items[idx].seq.layers[layer], items[idx].gradProjected, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj)
					}
					continue
				}
			}
			for groupIndex, itemIndex := range indexes {
				items[itemIndex].gradProjected = gradInput[groupIndex]
			}
		}
	}
	for i := range items {
		t.accumulateInputTokenGrad(items[i].seq.tokens, items[i].gradProjected, gradToken)
	}
	return true
}

func (t *EmbeddingTrainer) backpropProjectedFFNSequences(states []*embeddingSequenceState, gradProjected [][]float32, hiddenProjection, projection *backend.Tensor, gradHidden, gradProj []float32, d, h, e int) ([][]float32, bool) {
	if len(states) == 0 || len(states) != len(gradProjected) || hiddenProjection == nil || projection == nil {
		return nil, false
	}
	seqLen := len(states[0].tokens)
	if seqLen == 0 {
		return nil, false
	}
	for i, state := range states {
		if state == nil || len(state.tokens) != seqLen || len(state.hidden) != seqLen*d || len(state.activated) != seqLen*h || len(gradProjected[i]) != seqLen*e {
			return nil, false
		}
	}

	gradOutputMatrices := make([][]float32, len(states))
	gradActivatedPreMatrices := make([][]float32, len(states))
	activatedMatrices := make([][]float32, len(states))
	batchActivations := t.fullActivationBackwardAccelEnabled()
	var (
		projectedMatrices   [][]float32
		ffnResidualMatrices [][]float32
		ffnHiddenMatrices   [][]float32
	)
	if batchActivations {
		projectedMatrices = make([][]float32, len(states))
		ffnResidualMatrices = make([][]float32, len(states))
		ffnHiddenMatrices = make([][]float32, len(states))
	}
	for i, state := range states {
		gradOutputMatrix := make([]float32, seqLen*e)
		for row := range state.tokens {
			if state.mask[row] == 0 {
				continue
			}
			projectedBase := row * e
			copy(gradOutputMatrix[projectedBase:projectedBase+e], gradProjected[i][projectedBase:projectedBase+e])
		}
		gradOutputMatrices[i] = gradOutputMatrix
		activatedMatrices[i] = state.activated
		if batchActivations {
			projectedMatrices[i] = state.projected
			ffnResidualMatrices[i] = state.ffnResidual
			ffnHiddenMatrices[i] = state.ffnHidden
		}
	}
	if t.ffnLayerNormEnabled() {
		batchedLayerNorm := false
		if batchActivations {
			if out, ok := t.tryBatchedLayerNormBackwardRows(gradOutputMatrices, projectedMatrices, ffnResidualMatrices, seqLen, e); ok {
				gradOutputMatrices = out
				batchedLayerNorm = true
			}
		}
		if !batchedLayerNorm {
			for i, state := range states {
				gradOutputMatrix := gradOutputMatrices[i]
				for row := range state.tokens {
					if state.mask[row] == 0 {
						continue
					}
					projectedBase := row * e
					backwardLayerNormRow(
						gradOutputMatrix[projectedBase:projectedBase+e],
						gradOutputMatrix[projectedBase:projectedBase+e],
						state.projected[projectedBase:projectedBase+e],
						state.ffnResidual[projectedBase:projectedBase+e],
					)
				}
			}
		}
	}

	if out, ok := t.tryAccumulatedTransposeMatMul(activatedMatrices, gradOutputMatrices, seqLen, h, e); ok {
		addFloat32Slice(gradProj, out)
	} else {
		for i, state := range states {
			gradProjStep := make([]float32, h*e)
			if out, ok := t.tryTrainerMatMulBoundLeft(
				state.activatedBinding,
				tensorF32View([]int{seqLen, h}, state.activated),
				tensorF32View([]int{seqLen, e}, gradOutputMatrices[i]),
				true,
				false,
			); ok {
				copy(gradProjStep, out)
			} else {
				fillHostMatMulTranspose(state.activated, seqLen, h, gradOutputMatrices[i], seqLen, e, true, false, gradProjStep)
			}
			addFloat32Slice(gradProj, gradProjStep)
		}
	}

	if out, ok := t.tryBatchedBoundRightMatMul(gradOutputMatrices, seqLen, e, t.projParam.Name, projection, false, true); ok {
		gradActivatedPreMatrices = out
	} else {
		projData := forwardMatMulHostData(projection)
		for i := range states {
			gradActivatedPre := make([]float32, seqLen*h)
			fillHostMatMulTranspose(gradOutputMatrices[i], seqLen, e, projData, h, e, false, true, gradActivatedPre)
			gradActivatedPreMatrices[i] = gradActivatedPre
		}
	}

	gradActivatedMatrices := make([][]float32, len(states))
	hiddenMatrices := make([][]float32, len(states))
	batchedGELU := false
	fastGELU := fastGELUEnabled()
	if batchActivations && !fastGELU {
		if out, ok := t.tryBatchedGELUBackwardMul(gradActivatedPreMatrices, ffnHiddenMatrices, seqLen, h); ok {
			gradActivatedMatrices = out
			batchedGELU = true
		}
	}
	if !batchedGELU {
		for i, state := range states {
			gradActivated := make([]float32, seqLen*h)
			for row := range state.tokens {
				if state.mask[row] == 0 {
					continue
				}
				ffnBase := row * h
				fillGELUBackwardMul(gradActivated[ffnBase:ffnBase+h], gradActivatedPreMatrices[i][ffnBase:ffnBase+h], state.ffnHidden[ffnBase:ffnBase+h], fastGELU)
			}
			gradActivatedMatrices[i] = gradActivated
		}
	}
	for i, state := range states {
		hiddenMatrices[i] = state.hidden
	}

	if out, ok := t.tryAccumulatedTransposeMatMul(hiddenMatrices, gradActivatedMatrices, seqLen, d, h); ok {
		addFloat32Slice(gradHidden, out)
	} else {
		for i, state := range states {
			gradHiddenStep := make([]float32, d*h)
			if out, ok := t.tryTrainerMatMulBoundLeft(
				state.hiddenBinding,
				tensorF32View([]int{seqLen, d}, state.hidden),
				tensorF32View([]int{seqLen, h}, gradActivatedMatrices[i]),
				true,
				false,
			); ok {
				copy(gradHiddenStep, out)
			} else {
				fillHostMatMulTranspose(state.hidden, seqLen, d, gradActivatedMatrices[i], seqLen, h, true, false, gradHiddenStep)
			}
			addFloat32Slice(gradHidden, gradHiddenStep)
		}
	}

	gradInputs := make([][]float32, len(states))
	if out, ok := t.tryBatchedBoundRightMatMul(gradActivatedMatrices, seqLen, h, t.hiddenParam.Name, hiddenProjection, false, true); ok {
		gradInputs = out
	} else {
		hiddenData := forwardMatMulHostData(hiddenProjection)
		for i := range states {
			gradInput := make([]float32, seqLen*d)
			fillHostMatMulTranspose(gradActivatedMatrices[i], seqLen, h, hiddenData, d, h, false, true, gradInput)
			gradInputs[i] = gradInput
		}
	}
	if t.ffnResidualEnabled() {
		for i := range states {
			addFloat32Slice(gradInputs[i], gradOutputMatrices[i])
		}
	}
	return gradInputs, true
}

func (t *EmbeddingTrainer) tryBatchedBoundRightMatMul(lhsMatrices [][]float32, rows, cols int, rhsName string, rhs *backend.Tensor, transposeLeft, transposeRight bool) ([][]float32, bool) {
	if len(lhsMatrices) == 0 || rows == 0 || cols == 0 || rhs == nil {
		return nil, false
	}
	totalRows := len(lhsMatrices) * rows
	batched, ok := t.flattenFixedFloat32MatricesScratch(0, lhsMatrices, rows*cols)
	if !ok {
		return nil, false
	}
	out, ok := t.tryTrainerMatMulBoundRight(batched, totalRows, cols, rhsName, rhs, transposeLeft, transposeRight)
	if !ok || len(out)%len(lhsMatrices) != 0 {
		return nil, false
	}
	return splitFloat32Views(out, len(lhsMatrices))
}

func (t *EmbeddingTrainer) tryAccumulatedTransposeMatMul(lhsMatrices, rhsMatrices [][]float32, rows, lhsCols, rhsCols int) ([]float32, bool) {
	if len(lhsMatrices) == 0 || len(lhsMatrices) != len(rhsMatrices) || rows == 0 || lhsCols == 0 || rhsCols == 0 {
		return nil, false
	}
	totalRows := len(lhsMatrices) * rows
	lhsBatch, ok := t.flattenFixedFloat32MatricesScratch(0, lhsMatrices, rows*lhsCols)
	if !ok {
		return nil, false
	}
	rhsBatch, ok := t.flattenFixedFloat32MatricesScratch(1, rhsMatrices, rows*rhsCols)
	if !ok {
		return nil, false
	}
	return t.tryTrainerMatMul(lhsBatch, totalRows, lhsCols, rhsBatch, totalRows, rhsCols, true, false)
}

func (t *EmbeddingTrainer) trySharedLeftAccumulatedTransposeMatMuls(lhsMatrices [][]float32, rhsMatrixSets [][][]float32, rows, lhsCols, rhsCols int) ([][]float32, bool) {
	if t == nil || t.forwardMatMul == nil || !sharedLeftMatMulEnabled() || len(lhsMatrices) == 0 || len(rhsMatrixSets) < 2 || rows == 0 || lhsCols == 0 || rhsCols == 0 {
		return nil, false
	}
	if out, ok := t.tryConcatenatedSharedLeftAccumulatedTransposeMatMuls(lhsMatrices, rhsMatrixSets, rows, lhsCols, rhsCols); ok {
		return out, true
	}
	shared, ok := t.forwardMatMul.(backend.SharedLeftMatMulAccelerator)
	if !ok {
		return nil, false
	}
	totalRows := len(lhsMatrices) * rows
	lhsBatch, ok := t.flattenFixedFloat32MatricesScratch(0, lhsMatrices, rows*lhsCols)
	if !ok {
		return nil, false
	}
	rhsTensors := make([]*backend.Tensor, len(rhsMatrixSets))
	for i, rhsMatrices := range rhsMatrixSets {
		if len(rhsMatrices) != len(lhsMatrices) {
			return nil, false
		}
		rhsBatch, ok := flattenFixedFloat32Matrices(rhsMatrices, rows*rhsCols)
		if !ok {
			return nil, false
		}
		rhsTensors[i] = tensorF32View([]int{totalRows, rhsCols}, rhsBatch)
	}
	results, err := shared.RunMatMulsWithSharedLeft(
		tensorF32View([]int{totalRows, lhsCols}, lhsBatch),
		rhsTensors,
		trainerF32TensorValueType(),
		true,
		false,
	)
	if err != nil || len(results) != len(rhsMatrixSets) {
		return nil, false
	}
	out := make([][]float32, len(results))
	for i, result := range results {
		if len(result.Outputs) != 1 || result.Outputs[0] == nil {
			return nil, false
		}
		data := result.Outputs[0].F32
		if len(data) != lhsCols*rhsCols {
			return nil, false
		}
		out[i] = data
	}
	return out, true
}

func (t *EmbeddingTrainer) tryConcatenatedSharedLeftAccumulatedTransposeMatMuls(lhsMatrices [][]float32, rhsMatrixSets [][][]float32, rows, lhsCols, rhsCols int) ([][]float32, bool) {
	if t == nil || !concatenatedSharedLeftMatMulEnabled() || len(lhsMatrices) == 0 || len(rhsMatrixSets) < 2 || rows == 0 || lhsCols == 0 || rhsCols == 0 {
		return nil, false
	}
	totalRows := len(lhsMatrices) * rows
	lhsBatch, ok := t.flattenFixedFloat32MatricesScratch(0, lhsMatrices, rows*lhsCols)
	if !ok {
		return nil, false
	}
	combinedCols := len(rhsMatrixSets) * rhsCols
	rhsBatch := t.scratchFloat32(1, totalRows*combinedCols)
	for setIndex, rhsMatrices := range rhsMatrixSets {
		if len(rhsMatrices) != len(lhsMatrices) {
			return nil, false
		}
		for matrixIndex, rhsMatrix := range rhsMatrices {
			if len(rhsMatrix) != rows*rhsCols {
				return nil, false
			}
			for row := 0; row < rows; row++ {
				srcBase := row * rhsCols
				dstBase := (matrixIndex*rows+row)*combinedCols + setIndex*rhsCols
				copy(rhsBatch[dstBase:dstBase+rhsCols], rhsMatrix[srcBase:srcBase+rhsCols])
			}
		}
	}
	out, ok := t.tryTrainerMatMul(lhsBatch, totalRows, lhsCols, rhsBatch, totalRows, combinedCols, true, false)
	if !ok || len(out) != lhsCols*combinedCols {
		return nil, false
	}
	results := make([][]float32, len(rhsMatrixSets))
	for setIndex := range results {
		results[setIndex] = make([]float32, lhsCols*rhsCols)
	}
	for row := 0; row < lhsCols; row++ {
		srcBase := row * combinedCols
		for setIndex := range results {
			dstBase := row * rhsCols
			chunk := out[srcBase+setIndex*rhsCols : srcBase+(setIndex+1)*rhsCols]
			copy(results[setIndex][dstBase:dstBase+rhsCols], chunk)
		}
	}
	return results, true
}

func (t *EmbeddingTrainer) tryCombinedAttentionValueKeyGradMatMul(attnScoreMatrices, gradMixedMatrices, gradPreSoftmaxMatrices, attnQMatrices [][]float32, seqLen, d int) ([][]float32, [][]float32, bool) {
	if t == nil || !combinedAttentionVKGradMatMulEnabled() || len(attnScoreMatrices) == 0 || len(attnScoreMatrices) != len(gradMixedMatrices) || len(attnScoreMatrices) != len(gradPreSoftmaxMatrices) || len(attnScoreMatrices) != len(attnQMatrices) || seqLen == 0 || d == 0 {
		return nil, nil, false
	}
	batches := len(attnScoreMatrices)
	lhsMatrices := make([][]float32, 0, batches*2)
	rhsMatrices := make([][]float32, 0, batches*2)
	lhsMatrices = append(lhsMatrices, attnScoreMatrices...)
	lhsMatrices = append(lhsMatrices, gradPreSoftmaxMatrices...)
	rhsMatrices = append(rhsMatrices, gradMixedMatrices...)
	rhsMatrices = append(rhsMatrices, attnQMatrices...)
	out, ok := t.tryTrainerBatchedMatMulTranspose(lhsMatrices, seqLen, seqLen, rhsMatrices, seqLen, d, true, false)
	if !ok || len(out) != batches*2 {
		return nil, nil, false
	}
	return out[:batches], out[batches:], true
}

func (t *EmbeddingTrainer) tryAccumulatedAttentionInputGradMatMul(gradQMatrices, gradKMatrices, gradVMatrices [][]float32, seqLen, d int, attentionQuery, attentionKey, attentionValue *backend.Tensor) ([][]float32, bool) {
	if t == nil || t.forwardMatMul == nil || !accumulatedAttentionInputGradMatMulEnabled() || len(gradQMatrices) == 0 || len(gradQMatrices) != len(gradKMatrices) || len(gradQMatrices) != len(gradVMatrices) || seqLen == 0 || d == 0 {
		return nil, false
	}
	accumulated, ok := t.forwardMatMul.(backend.AccumulatedBoundRightMatMulAccelerator)
	if !ok {
		return nil, false
	}
	for _, tensor := range []*backend.Tensor{attentionQuery, attentionKey, attentionValue} {
		if tensor == nil || len(tensor.Shape) != 2 || tensor.Shape[0] != d || tensor.Shape[1] != d {
			return nil, false
		}
	}
	perMatrix := seqLen * d
	gradQBatch, ok := t.flattenFixedFloat32MatricesScratch(0, gradQMatrices, perMatrix)
	if !ok {
		return nil, false
	}
	gradKBatch, ok := t.flattenFixedFloat32MatricesScratch(1, gradKMatrices, perMatrix)
	if !ok {
		return nil, false
	}
	gradVBatch, ok := t.flattenFixedFloat32MatricesScratch(2, gradVMatrices, perMatrix)
	if !ok {
		return nil, false
	}
	totalRows := len(gradQMatrices) * seqLen
	result, err := accumulated.RunAccumulatedMatMulsWithBoundRights(
		[]*backend.Tensor{
			tensorF32View([]int{totalRows, d}, gradQBatch),
			tensorF32View([]int{totalRows, d}, gradKBatch),
			tensorF32View([]int{totalRows, d}, gradVBatch),
		},
		[]string{t.attnQParam.Name, t.attnKParam.Name, t.attnVParam.Name},
		trainerF32TensorValueType(),
		false,
		true,
	)
	if err != nil || len(result.Outputs) != 1 || result.Outputs[0] == nil {
		return nil, false
	}
	out := result.Outputs[0].F32
	if len(out) != len(gradQMatrices)*perMatrix {
		return nil, false
	}
	return splitFloat32Views(out, len(gradQMatrices))
}

func (t *EmbeddingTrainer) backpropAttentionSequences(states []*embeddingSequenceState, gradHiddenMatrices [][]float32, attentionQuery, attentionKey, attentionValue, attentionOutput *backend.Tensor, gradAttnQ, gradAttnK, gradAttnV, gradAttnO []float32, d int) ([][]float32, bool) {
	if len(states) == 0 || len(states) != len(gradHiddenMatrices) || attentionQuery == nil || attentionKey == nil || attentionValue == nil || attentionOutput == nil {
		return nil, false
	}
	seqLen := len(states[0].tokens)
	if seqLen == 0 {
		return nil, false
	}
	for i, state := range states {
		if state == nil || len(state.tokens) != seqLen || len(state.input) != seqLen*d || len(gradHiddenMatrices[i]) != seqLen*d {
			return nil, false
		}
	}

	gradAttnOutputs := make([][]float32, len(states))
	gradResidualInputs := make([][]float32, len(states))
	attnMixedMatrices := make([][]float32, len(states))
	batchActivations := t.fullActivationBackwardAccelEnabled()
	var (
		attnHiddenMatrices   [][]float32
		attnResidualMatrices [][]float32
	)
	if batchActivations {
		attnHiddenMatrices = make([][]float32, len(states))
		attnResidualMatrices = make([][]float32, len(states))
	}
	for i, state := range states {
		gradAttnOutput := gradHiddenMatrices[i]
		gradAttnOutputs[i] = gradAttnOutput
		attnMixedMatrices[i] = state.attnMixed
		if batchActivations {
			attnHiddenMatrices[i] = state.hidden
			attnResidualMatrices[i] = state.attnResidual
		}
	}
	if t.attentionLayerNormEnabled() {
		batchedLayerNorm := false
		if batchActivations {
			if out, ok := t.tryBatchedLayerNormBackwardRows(gradAttnOutputs, attnHiddenMatrices, attnResidualMatrices, seqLen, d); ok {
				gradAttnOutputs = out
				batchedLayerNorm = true
			}
		}
		if !batchedLayerNorm {
			for i, state := range states {
				gradAttnOutput := gradAttnOutputs[i]
				for row := 0; row < seqLen; row++ {
					base := row * d
					backwardLayerNormRow(
						gradAttnOutput[base:base+d],
						gradHiddenMatrices[i][base:base+d],
						state.hidden[base:base+d],
						state.attnResidual[base:base+d],
					)
				}
			}
		}
	}
	if t.attentionResidualEnabled() {
		for i := range gradResidualInputs {
			gradResidualInputs[i] = gradAttnOutputs[i]
		}
	}

	if out, ok := t.tryAccumulatedTransposeMatMul(attnMixedMatrices, gradAttnOutputs, seqLen, d, d); ok {
		addFloat32Slice(gradAttnO, out)
	} else {
		for i, state := range states {
			gradAttnOStep := make([]float32, d*d)
			if out, ok := t.tryTrainerMatMulBoundLeft(
				state.attnMixedBinding,
				tensorF32View([]int{seqLen, d}, state.attnMixed),
				tensorF32View([]int{seqLen, d}, gradAttnOutputs[i]),
				true,
				false,
			); ok {
				copy(gradAttnOStep, out)
			} else {
				fillHostMatMulTranspose(state.attnMixed, seqLen, d, gradAttnOutputs[i], seqLen, d, true, false, gradAttnOStep)
			}
			addFloat32Slice(gradAttnO, gradAttnOStep)
		}
	}

	gradMixedMatrices, ok := t.tryBatchedBoundRightMatMul(gradAttnOutputs, seqLen, d, t.attnOParam.Name, attentionOutput, false, true)
	if !ok {
		attnOData := forwardMatMulHostData(attentionOutput)
		gradMixedMatrices = make([][]float32, len(states))
		for i := range states {
			gradMixed := make([]float32, seqLen*d)
			fillHostMatMulTranspose(gradAttnOutputs[i], seqLen, d, attnOData, d, d, false, true, gradMixed)
			gradMixedMatrices[i] = gradMixed
		}
	}

	gradQMatrices := make([][]float32, len(states))
	gradKMatrices := make([][]float32, len(states))
	gradVMatrices := make([][]float32, len(states))
	inputMatrices := make([][]float32, len(states))
	attnScoreMatrices := make([][]float32, len(states))
	attnQMatrices := make([][]float32, len(states))
	attnKMatrices := make([][]float32, len(states))
	attnVMatrices := make([][]float32, len(states))
	for i, state := range states {
		attnScoreMatrices[i] = state.attnScores
		attnQMatrices[i] = state.attnQ
		attnKMatrices[i] = state.attnK
		attnVMatrices[i] = state.attnV
		inputMatrices[i] = state.input
	}

	gradScoresMatrices, gradScoresOK := t.tryTrainerBatchedMatMulTranspose(gradMixedMatrices, seqLen, d, attnVMatrices, seqLen, d, false, true)
	if !gradScoresOK {
		gradScoresMatrices = make([][]float32, len(states))
		for i, state := range states {
			gradScoresFlat := make([]float32, seqLen*seqLen)
			if out, ok := t.tryTrainerMatMul(gradMixedMatrices[i], seqLen, d, state.attnV, seqLen, d, false, true); ok {
				copy(gradScoresFlat, out)
			} else {
				fillHostMatMulTranspose(gradMixedMatrices[i], seqLen, d, state.attnV, seqLen, d, false, true, gradScoresFlat)
			}
			gradScoresMatrices[i] = gradScoresFlat
		}
	}

	gradPreSoftmaxMatrices, gradPreSoftmaxOK := t.tryBatchedSoftmaxBackwardRows(gradScoresMatrices, attnScoreMatrices, seqLen, seqLen)
	if !gradPreSoftmaxOK {
		gradPreSoftmaxMatrices = make([][]float32, len(states))
		for i, state := range states {
			gradPreSoftmaxFlat := make([]float32, seqLen*seqLen)
			if !state.skipUnboundActivationBackward(state.attnScoresBinding) {
				if out, ok := t.trySoftmaxBackwardRows(gradScoresMatrices[i], state.attnScores, seqLen, seqLen, state.attnScoresBinding); ok {
					copy(gradPreSoftmaxFlat, out)
					gradPreSoftmaxMatrices[i] = gradPreSoftmaxFlat
					continue
				}
			}
			for row := 0; row < seqLen; row++ {
				rowScores := state.attnScores[row*seqLen : (row+1)*seqLen]
				gradScores := gradScoresMatrices[i][row*seqLen : (row+1)*seqLen]
				backwardSoftmaxRow(gradPreSoftmaxFlat[row*seqLen:(row+1)*seqLen], gradScores, rowScores)
			}
			gradPreSoftmaxMatrices[i] = gradPreSoftmaxFlat
		}
	}
	scoreScale := t.attentionScoreScale(d)
	for i := range gradPreSoftmaxMatrices {
		scaleFloat32Slice(gradPreSoftmaxMatrices[i], scoreScale)
	}

	batchedGradV, batchedGradK, combinedVKOK := t.tryCombinedAttentionValueKeyGradMatMul(attnScoreMatrices, gradMixedMatrices, gradPreSoftmaxMatrices, attnQMatrices, seqLen, d)
	gradVOK := combinedVKOK
	gradKOK := combinedVKOK
	if !combinedVKOK {
		batchedGradV, gradVOK = t.tryTrainerBatchedMatMulTranspose(attnScoreMatrices, seqLen, seqLen, gradMixedMatrices, seqLen, d, true, false)
		batchedGradK, gradKOK = t.tryTrainerBatchedMatMulTranspose(gradPreSoftmaxMatrices, seqLen, seqLen, attnQMatrices, seqLen, d, true, false)
	}
	batchedGradQ, gradQOK := t.tryTrainerBatchedMatMulTranspose(gradPreSoftmaxMatrices, seqLen, seqLen, attnKMatrices, seqLen, d, false, false)
	for i, state := range states {
		gradMixed := gradMixedMatrices[i]
		gradPreSoftmaxFlat := gradPreSoftmaxMatrices[i]
		var gradQ, gradK, gradV []float32
		if gradVOK {
			gradV = batchedGradV[i]
		} else if out, ok := t.tryTrainerMatMulBoundLeft(
			state.attnScoresBinding,
			tensorF32View([]int{seqLen, seqLen}, state.attnScores),
			tensorF32View([]int{seqLen, d}, gradMixed),
			true,
			false,
		); ok {
			gradV = out
		} else {
			gradV = make([]float32, seqLen*d)
			fillHostMatMulTranspose(state.attnScores, seqLen, seqLen, gradMixed, seqLen, d, true, false, gradV)
		}
		if gradQOK {
			gradQ = batchedGradQ[i]
		} else if out, ok := t.tryTrainerMatMul(gradPreSoftmaxFlat, seqLen, seqLen, state.attnK, seqLen, d, false, false); ok {
			gradQ = out
		} else {
			gradQ = make([]float32, seqLen*d)
			fillHostMatMulTranspose(gradPreSoftmaxFlat, seqLen, seqLen, state.attnK, seqLen, d, false, false, gradQ)
		}
		if gradKOK {
			gradK = batchedGradK[i]
		} else if out, ok := t.tryTrainerMatMul(gradPreSoftmaxFlat, seqLen, seqLen, state.attnQ, seqLen, d, true, false); ok {
			gradK = out
		} else {
			gradK = make([]float32, seqLen*d)
			fillHostMatMulTranspose(gradPreSoftmaxFlat, seqLen, seqLen, state.attnQ, seqLen, d, true, false, gradK)
		}
		gradQMatrices[i] = gradQ
		gradKMatrices[i] = gradK
		gradVMatrices[i] = gradV
	}

	if sharedGradWeights, ok := t.trySharedLeftAccumulatedTransposeMatMuls(inputMatrices, [][][]float32{gradQMatrices, gradKMatrices, gradVMatrices}, seqLen, d, d); ok {
		addFloat32Slice(gradAttnQ, sharedGradWeights[0])
		addFloat32Slice(gradAttnK, sharedGradWeights[1])
		addFloat32Slice(gradAttnV, sharedGradWeights[2])
	} else {
		if out, ok := t.tryAccumulatedTransposeMatMul(inputMatrices, gradQMatrices, seqLen, d, d); ok {
			addFloat32Slice(gradAttnQ, out)
		} else {
			for i, state := range states {
				gradAttnQStep := make([]float32, d*d)
				if out, ok := t.tryTrainerMatMulBoundLeft(
					state.inputBinding,
					tensorF32View([]int{seqLen, d}, state.input),
					tensorF32View([]int{seqLen, d}, gradQMatrices[i]),
					true,
					false,
				); ok {
					copy(gradAttnQStep, out)
				} else {
					fillHostMatMulTranspose(state.input, seqLen, d, gradQMatrices[i], seqLen, d, true, false, gradAttnQStep)
				}
				addFloat32Slice(gradAttnQ, gradAttnQStep)
			}
		}
		if out, ok := t.tryAccumulatedTransposeMatMul(inputMatrices, gradKMatrices, seqLen, d, d); ok {
			addFloat32Slice(gradAttnK, out)
		} else {
			for i, state := range states {
				gradAttnKStep := make([]float32, d*d)
				if out, ok := t.tryTrainerMatMulBoundLeft(
					state.inputBinding,
					tensorF32View([]int{seqLen, d}, state.input),
					tensorF32View([]int{seqLen, d}, gradKMatrices[i]),
					true,
					false,
				); ok {
					copy(gradAttnKStep, out)
				} else {
					fillHostMatMulTranspose(state.input, seqLen, d, gradKMatrices[i], seqLen, d, true, false, gradAttnKStep)
				}
				addFloat32Slice(gradAttnK, gradAttnKStep)
			}
		}
		if out, ok := t.tryAccumulatedTransposeMatMul(inputMatrices, gradVMatrices, seqLen, d, d); ok {
			addFloat32Slice(gradAttnV, out)
		} else {
			for i, state := range states {
				gradAttnVStep := make([]float32, d*d)
				if out, ok := t.tryTrainerMatMulBoundLeft(
					state.inputBinding,
					tensorF32View([]int{seqLen, d}, state.input),
					tensorF32View([]int{seqLen, d}, gradVMatrices[i]),
					true,
					false,
				); ok {
					copy(gradAttnVStep, out)
				} else {
					fillHostMatMulTranspose(state.input, seqLen, d, gradVMatrices[i], seqLen, d, true, false, gradAttnVStep)
				}
				addFloat32Slice(gradAttnV, gradAttnVStep)
			}
		}
	}

	if accumulatedGradInputs, ok := t.tryAccumulatedAttentionInputGradMatMul(gradQMatrices, gradKMatrices, gradVMatrices, seqLen, d, attentionQuery, attentionKey, attentionValue); ok {
		gradInputs := make([][]float32, len(states))
		for i := range states {
			gradInput := accumulatedGradInputs[i]
			addFloat32Slice(gradInput, gradResidualInputs[i])
			gradInputs[i] = gradInput
		}
		return gradInputs, true
	}

	gradQInputs, okQ := t.tryBatchedBoundRightMatMul(gradQMatrices, seqLen, d, t.attnQParam.Name, attentionQuery, false, true)
	gradKInputs, okK := t.tryBatchedBoundRightMatMul(gradKMatrices, seqLen, d, t.attnKParam.Name, attentionKey, false, true)
	gradVInputs, okV := t.tryBatchedBoundRightMatMul(gradVMatrices, seqLen, d, t.attnVParam.Name, attentionValue, false, true)
	gradInputs := make([][]float32, len(states))
	if !okQ || !okK || !okV {
		attnQData := forwardMatMulHostData(attentionQuery)
		attnKData := forwardMatMulHostData(attentionKey)
		attnVData := forwardMatMulHostData(attentionValue)
		for i := range states {
			gradInput := make([]float32, seqLen*d)
			gradInputStep := make([]float32, seqLen*d)
			if okQ {
				addFloat32Slice(gradInput, gradQInputs[i])
			} else {
				fillHostMatMulTranspose(gradQMatrices[i], seqLen, d, attnQData, d, d, false, true, gradInputStep)
				addFloat32Slice(gradInput, gradInputStep)
			}
			if okK {
				addFloat32Slice(gradInput, gradKInputs[i])
			} else {
				for j := range gradInputStep {
					gradInputStep[j] = 0
				}
				fillHostMatMulTranspose(gradKMatrices[i], seqLen, d, attnKData, d, d, false, true, gradInputStep)
				addFloat32Slice(gradInput, gradInputStep)
			}
			if okV {
				addFloat32Slice(gradInput, gradVInputs[i])
			} else {
				for j := range gradInputStep {
					gradInputStep[j] = 0
				}
				fillHostMatMulTranspose(gradVMatrices[i], seqLen, d, attnVData, d, d, false, true, gradInputStep)
				addFloat32Slice(gradInput, gradInputStep)
			}
			addFloat32Slice(gradInput, gradResidualInputs[i])
			gradInputs[i] = gradInput
		}
		return gradInputs, true
	}
	for i := range states {
		gradInput := make([]float32, seqLen*d)
		addFloat32Slice(gradInput, gradQInputs[i])
		addFloat32Slice(gradInput, gradKInputs[i])
		addFloat32Slice(gradInput, gradVInputs[i])
		addFloat32Slice(gradInput, gradResidualInputs[i])
		gradInputs[i] = gradInput
	}
	return gradInputs, true
}

func (t *EmbeddingTrainer) backpropEncodedLayer(state *embeddingSequenceState, gradProjected []float32, attentionQuery, attentionKey, attentionValue, attentionOutput, hiddenProjection, projection *backend.Tensor, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, gradHidden, gradProj []float32) []float32 {
	if hiddenProjection != nil {
		gradInput := t.backpropProjectedFFNSequence(state, gradProjected, hiddenProjection, projection, gradHidden, gradProj, t.tokenEmbed.Shape[1], hiddenProjection.Shape[1], t.projection.Shape[1])
		if t.attentionEnabled() {
			return t.backpropAttentionSequence(state, gradInput, attentionQuery, attentionKey, attentionValue, attentionOutput, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, t.tokenEmbed.Shape[1])
		}
		return gradInput
	}
	gradInput := t.backpropProjectedSequence(state, gradProjected, projection, gradProj, t.tokenEmbed.Shape[1], t.projection.Shape[1])
	if t.attentionEnabled() {
		return t.backpropAttentionSequence(state, gradInput, attentionQuery, attentionKey, attentionValue, attentionOutput, gradAttnQ, gradAttnK, gradAttnV, gradAttnO, t.tokenEmbed.Shape[1])
	}
	return gradInput
}

func (t *EmbeddingTrainer) backpropAttentionSequence(state *embeddingSequenceState, gradHidden []float32, attentionQuery, attentionKey, attentionValue, attentionOutput *backend.Tensor, gradAttnQ, gradAttnK, gradAttnV, gradAttnO []float32, d int) []float32 {
	gradInput := make([]float32, len(state.input))
	if state == nil || len(state.tokens) == 0 || attentionQuery == nil || attentionKey == nil || attentionValue == nil || attentionOutput == nil {
		return gradInput
	}
	seqLen := len(state.tokens)
	gradAttnOutput := make([]float32, seqLen*d)
	gradResidualInput := make([]float32, seqLen*d)
	for row := 0; row < seqLen; row++ {
		base := row * d
		copy(gradAttnOutput[base:base+d], gradHidden[base:base+d])
		if t.attentionResidualEnabled() {
			copy(gradResidualInput[base:base+d], gradAttnOutput[base:base+d])
		}
	}
	if t.attentionLayerNormEnabled() {
		accelerated := false
		if !state.skipUnboundActivationBackward(state.hiddenBinding, state.attnResidualBinding) {
			if out, ok := t.tryLayerNormBackwardRows(gradAttnOutput, state.hidden, state.attnResidual, seqLen, d, state.hiddenBinding, state.attnResidualBinding); ok {
				copy(gradAttnOutput, out)
				if t.attentionResidualEnabled() {
					copy(gradResidualInput, out)
				}
				accelerated = true
			}
		}
		if !accelerated {
			for row := 0; row < seqLen; row++ {
				base := row * d
				backwardLayerNormRow(
					gradAttnOutput[base:base+d],
					gradHidden[base:base+d],
					state.hidden[base:base+d],
					state.attnResidual[base:base+d],
				)
				if t.attentionResidualEnabled() {
					copy(gradResidualInput[base:base+d], gradAttnOutput[base:base+d])
				}
			}
		}
	}
	gradMixed := make([]float32, seqLen*d)
	gradAttnOStep := make([]float32, d*d)
	if out, ok := t.tryTrainerMatMulBoundLeft(
		state.attnMixedBinding,
		tensorF32View([]int{seqLen, d}, state.attnMixed),
		tensorF32View([]int{seqLen, d}, gradAttnOutput),
		true,
		false,
	); ok {
		copy(gradAttnOStep, out)
	} else {
		fillHostMatMulTranspose(state.attnMixed, seqLen, d, gradAttnOutput, seqLen, d, true, false, gradAttnOStep)
	}
	addFloat32Slice(gradAttnO, gradAttnOStep)
	if out, ok := t.tryTrainerMatMulBoundRight(gradAttnOutput, seqLen, d, t.attnOParam.Name, attentionOutput, false, true); ok {
		copy(gradMixed, out)
	} else {
		attnOData := forwardMatMulHostData(attentionOutput)
		fillHostMatMulTranspose(gradAttnOutput, seqLen, d, attnOData, d, d, false, true, gradMixed)
	}

	gradQ := make([]float32, seqLen*d)
	gradK := make([]float32, seqLen*d)
	gradV := make([]float32, seqLen*d)
	gradScoresFlat := make([]float32, seqLen*seqLen)
	if out, ok := t.tryTrainerMatMul(gradMixed, seqLen, d, state.attnV, seqLen, d, false, true); ok {
		copy(gradScoresFlat, out)
	} else {
		fillHostMatMulTranspose(gradMixed, seqLen, d, state.attnV, seqLen, d, false, true, gradScoresFlat)
	}
	gradPreSoftmaxFlat := make([]float32, seqLen*seqLen)
	acceleratedSoftmax := false
	if !state.skipUnboundActivationBackward(state.attnScoresBinding) {
		if out, ok := t.trySoftmaxBackwardRows(gradScoresFlat, state.attnScores, seqLen, seqLen, state.attnScoresBinding); ok {
			copy(gradPreSoftmaxFlat, out)
			acceleratedSoftmax = true
		}
	}
	if !acceleratedSoftmax {
		for i := 0; i < seqLen; i++ {
			rowScores := state.attnScores[i*seqLen : (i+1)*seqLen]
			gradScores := gradScoresFlat[i*seqLen : (i+1)*seqLen]
			backwardSoftmaxRow(gradPreSoftmaxFlat[i*seqLen:(i+1)*seqLen], gradScores, rowScores)
		}
	}
	scaleFloat32Slice(gradPreSoftmaxFlat, t.attentionScoreScale(d))
	if out, ok := t.tryTrainerMatMulBoundLeft(
		state.attnScoresBinding,
		tensorF32View([]int{seqLen, seqLen}, state.attnScores),
		tensorF32View([]int{seqLen, d}, gradMixed),
		true,
		false,
	); ok {
		addFloat32Slice(gradV, out)
	} else {
		gradVStep := make([]float32, seqLen*d)
		fillHostMatMulTranspose(state.attnScores, seqLen, seqLen, gradMixed, seqLen, d, true, false, gradVStep)
		addFloat32Slice(gradV, gradVStep)
	}
	if out, ok := t.tryTrainerMatMul(gradPreSoftmaxFlat, seqLen, seqLen, state.attnK, seqLen, d, false, false); ok {
		copy(gradQ, out)
	} else {
		fillHostMatMulTranspose(gradPreSoftmaxFlat, seqLen, seqLen, state.attnK, seqLen, d, false, false, gradQ)
	}
	if out, ok := t.tryTrainerMatMul(gradPreSoftmaxFlat, seqLen, seqLen, state.attnQ, seqLen, d, true, false); ok {
		copy(gradK, out)
	} else {
		fillHostMatMulTranspose(gradPreSoftmaxFlat, seqLen, seqLen, state.attnQ, seqLen, d, true, false, gradK)
	}

	gradAttnQStep := make([]float32, d*d)
	if out, ok := t.tryTrainerMatMulBoundLeft(
		state.inputBinding,
		tensorF32View([]int{seqLen, d}, state.input),
		tensorF32View([]int{seqLen, d}, gradQ),
		true,
		false,
	); ok {
		copy(gradAttnQStep, out)
	} else {
		fillHostMatMulTranspose(state.input, seqLen, d, gradQ, seqLen, d, true, false, gradAttnQStep)
	}
	addFloat32Slice(gradAttnQ, gradAttnQStep)
	gradAttnKStep := make([]float32, d*d)
	if out, ok := t.tryTrainerMatMulBoundLeft(
		state.inputBinding,
		tensorF32View([]int{seqLen, d}, state.input),
		tensorF32View([]int{seqLen, d}, gradK),
		true,
		false,
	); ok {
		copy(gradAttnKStep, out)
	} else {
		fillHostMatMulTranspose(state.input, seqLen, d, gradK, seqLen, d, true, false, gradAttnKStep)
	}
	addFloat32Slice(gradAttnK, gradAttnKStep)
	gradAttnVStep := make([]float32, d*d)
	if out, ok := t.tryTrainerMatMulBoundLeft(
		state.inputBinding,
		tensorF32View([]int{seqLen, d}, state.input),
		tensorF32View([]int{seqLen, d}, gradV),
		true,
		false,
	); ok {
		copy(gradAttnVStep, out)
	} else {
		fillHostMatMulTranspose(state.input, seqLen, d, gradV, seqLen, d, true, false, gradAttnVStep)
	}
	addFloat32Slice(gradAttnV, gradAttnVStep)
	if out, ok := t.tryTrainerMatMulBoundRight(gradQ, seqLen, d, t.attnQParam.Name, attentionQuery, false, true); ok {
		addFloat32Slice(gradInput, out)
	} else {
		attnQData := forwardMatMulHostData(attentionQuery)
		gradInputStep := make([]float32, seqLen*d)
		fillHostMatMulTranspose(gradQ, seqLen, d, attnQData, d, d, false, true, gradInputStep)
		addFloat32Slice(gradInput, gradInputStep)
	}
	if out, ok := t.tryTrainerMatMulBoundRight(gradK, seqLen, d, t.attnKParam.Name, attentionKey, false, true); ok {
		addFloat32Slice(gradInput, out)
	} else {
		attnKData := forwardMatMulHostData(attentionKey)
		gradInputStep := make([]float32, seqLen*d)
		fillHostMatMulTranspose(gradK, seqLen, d, attnKData, d, d, false, true, gradInputStep)
		addFloat32Slice(gradInput, gradInputStep)
	}
	if out, ok := t.tryTrainerMatMulBoundRight(gradV, seqLen, d, t.attnVParam.Name, attentionValue, false, true); ok {
		addFloat32Slice(gradInput, out)
	} else {
		attnVData := forwardMatMulHostData(attentionValue)
		gradInputStep := make([]float32, seqLen*d)
		fillHostMatMulTranspose(gradV, seqLen, d, attnVData, d, d, false, true, gradInputStep)
		addFloat32Slice(gradInput, gradInputStep)
	}
	for i := range gradInput {
		gradInput[i] += gradResidualInput[i]
	}
	return gradInput
}

func (t *EmbeddingTrainer) accumulateInputTokenGrad(tokens []int32, gradInput, gradToken []float32) {
	if t == nil || t.tokenEmbed == nil || len(t.tokenEmbed.Shape) != 2 {
		return
	}
	d := t.tokenEmbed.Shape[1]
	vocab := t.tokenEmbed.Shape[0]
	if d == 0 || vocab == 0 {
		return
	}
	if t.manifest.PositionEncoding == EmbeddingPositionEncodingRoPE {
		gradInput = append([]float32(nil), gradInput...)
		applyRoPETransposeToRowsInPlace(gradInput, len(tokens), d)
	}
	accumulateTokenGrad(tokens, gradInput, gradToken, d, vocab)
}

func (t *EmbeddingTrainer) accumulateInputRoleGrad(role int32, tokenCount int, gradInput, gradRole []float32) {
	if t == nil || t.roleEmbed == nil || len(t.roleEmbed.Shape) != 2 || len(gradRole) == 0 {
		return
	}
	d := t.roleEmbed.Shape[1]
	rows := t.roleEmbed.Shape[0]
	if d == 0 || rows == 0 || role < 0 || int(role) >= rows {
		return
	}
	if t.manifest.PositionEncoding == EmbeddingPositionEncodingRoPE {
		gradInput = append([]float32(nil), gradInput...)
		applyRoPETransposeToRowsInPlace(gradInput, tokenCount, d)
	}
	roleBase := int(role) * d
	for row := 0; row < tokenCount; row++ {
		inputBase := row * d
		for col := 0; col < d; col++ {
			gradRole[roleBase+col] += gradInput[inputBase+col]
		}
	}
}

func accumulateTokenGrad(tokens []int32, gradInput, gradToken []float32, d, vocab int) {
	if d == 0 || vocab == 0 {
		return
	}
	for row, tok := range tokens {
		tokenBase := int(tok) * d
		if tokenBase < 0 || tokenBase+d > vocab*d {
			continue
		}
		rowBase := row * d
		for i := 0; i < d; i++ {
			gradToken[tokenBase+i] += gradInput[rowBase+i]
		}
	}
}

func backwardSoftmaxRow(dX, dOut, probs []float32) {
	dot := float32(0)
	for i := range probs {
		dot += dOut[i] * probs[i]
	}
	for i := range probs {
		dX[i] = probs[i] * (dOut[i] - dot)
	}
}

func softmaxAttentionScoresInPlace(data []float32, seqLen int, mask []int32, mode string) {
	switch mode {
	case EmbeddingAttentionMaskModeKey:
		softmaxRowsMaskedColumnsInPlace(data, seqLen, seqLen, mask)
	default:
		softmaxRowsInPlace(data, seqLen, seqLen)
	}
}

func transpose2DData(data []float32, rows, cols int) []float32 {
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[c*rows+r] = data[r*cols+c]
		}
	}
	return out
}

func softmaxRowsMaskedColumnsInPlace(data []float32, rows, cols int, mask []int32) {
	if len(mask) != cols {
		softmaxRowsInPlace(data, rows, cols)
		return
	}
	for row := 0; row < rows; row++ {
		base := row * cols
		maxVal := float32(math.Inf(-1))
		active := false
		for col := 0; col < cols; col++ {
			if mask[col] == 0 {
				continue
			}
			if !active || data[base+col] > maxVal {
				maxVal = data[base+col]
			}
			active = true
		}
		if !active {
			for col := 0; col < cols; col++ {
				data[base+col] = 0
			}
			continue
		}
		sum := float32(0)
		for col := 0; col < cols; col++ {
			if mask[col] == 0 {
				data[base+col] = 0
				continue
			}
			value := float32(math.Exp(float64(data[base+col] - maxVal)))
			data[base+col] = value
			sum += value
		}
		if sum == 0 {
			continue
		}
		inv := 1 / sum
		for col := 0; col < cols; col++ {
			if mask[col] != 0 {
				data[base+col] *= inv
			}
		}
	}
}

func softmaxRowsInPlace(data []float32, rows, cols int) {
	for row := 0; row < rows; row++ {
		base := row * cols
		maxVal := data[base]
		for col := 1; col < cols; col++ {
			if data[base+col] > maxVal {
				maxVal = data[base+col]
			}
		}
		sum := float32(0)
		for col := 0; col < cols; col++ {
			value := float32(math.Exp(float64(data[base+col] - maxVal)))
			data[base+col] = value
			sum += value
		}
		if sum == 0 {
			continue
		}
		inv := 1 / sum
		for col := 0; col < cols; col++ {
			data[base+col] *= inv
		}
	}
}

func layerNormRow(dst, src []float32) {
	if len(dst) != len(src) || len(src) == 0 {
		return
	}
	mean := float32(0)
	for _, value := range src {
		mean += value
	}
	mean /= float32(len(src))
	variance := float32(0)
	for _, value := range src {
		centered := value - mean
		variance += centered * centered
	}
	variance /= float32(len(src))
	invStd := float32(1.0 / math.Sqrt(float64(variance)+1e-5))
	for i, value := range src {
		dst[i] = (value - mean) * invStd
	}
}

func backwardLayerNormRow(dst, gradOut, normalized, pre []float32) {
	if len(dst) != len(gradOut) || len(gradOut) != len(normalized) || len(normalized) != len(pre) || len(pre) == 0 {
		return
	}
	mean := float32(0)
	for _, value := range pre {
		mean += value
	}
	mean /= float32(len(pre))
	variance := float32(0)
	for _, value := range pre {
		centered := value - mean
		variance += centered * centered
	}
	variance /= float32(len(pre))
	invStd := float32(1.0 / math.Sqrt(float64(variance)+1e-5))
	sumGrad := float32(0)
	sumGradNorm := float32(0)
	for i := range gradOut {
		sumGrad += gradOut[i]
		sumGradNorm += gradOut[i] * normalized[i]
	}
	n := float32(len(pre))
	for i := range gradOut {
		dst[i] = (invStd / n) * (n*gradOut[i] - sumGrad - normalized[i]*sumGradNorm)
	}
}

func cosineGrad(left, right []float32) (float32, []float32, []float32) {
	gradLeft := make([]float32, len(left))
	gradRight := make([]float32, len(right))
	if len(left) != len(right) || len(left) == 0 {
		return 0, gradLeft, gradRight
	}
	dot := float32(0)
	leftNormSq := float32(0)
	rightNormSq := float32(0)
	for i := range left {
		dot += left[i] * right[i]
		leftNormSq += left[i] * left[i]
		rightNormSq += right[i] * right[i]
	}
	leftNorm := float32(math.Sqrt(float64(leftNormSq)))
	rightNorm := float32(math.Sqrt(float64(rightNormSq)))
	if leftNorm == 0 || rightNorm == 0 {
		return 0, gradLeft, gradRight
	}
	denom := leftNorm * rightNorm
	score := dot / denom
	leftScale := score / (leftNorm * leftNorm)
	rightScale := score / (rightNorm * rightNorm)
	for i := range left {
		gradLeft[i] = right[i]/denom - left[i]*leftScale
		gradRight[i] = left[i]/denom - right[i]*rightScale
	}
	return score, gradLeft, gradRight
}

func (t *EmbeddingTrainer) optimizerUpdateConfig(scale float32) backend.OptimizerUpdateConfig {
	return backend.OptimizerUpdateConfig{
		Optimizer:    t.config.Optimizer,
		Step:         t.step,
		LearningRate: t.config.LearningRate,
		WeightDecay:  t.config.WeightDecay,
		Beta1:        t.config.Beta1,
		Beta2:        t.config.Beta2,
		Epsilon:      t.config.Epsilon,
		Scale:        scale,
	}
}

func (t *EmbeddingTrainer) applyOptimizerUpdate(name string, tensor, mom1, mom2 *backend.Tensor, grad []float32, scale float32) {
	if tensor == nil {
		return
	}
	if t != nil && t.optimizerAccel != nil && len(grad) == len(tensor.F32) {
		err := t.optimizerAccel.ApplyUpdate(
			name,
			t.optimizerUpdateConfig(scale),
			tensor,
			mom1,
			mom2,
			tensorF32View(tensor.Shape, grad),
		)
		if err == nil {
			t.momentsDirty = true
			t.invalidateForwardWeights()
			return
		}
	}
	applyOptimizerUpdate(t.config, t.step, tensor, mom1, mom2, grad, scale)
	if t != nil {
		t.invalidateForwardWeights()
	}
}

type embeddingTrainTensorSnapshot struct {
	tensor *backend.Tensor
	before []float32
}

func aggregateScaledGradientStats(scale float32, grads ...[]float32) EmbeddingTrainStatMetrics {
	var stats EmbeddingTrainStatMetrics
	sumSquares := float64(0)
	for _, grad := range grads {
		stats.TotalCount += len(grad)
		for _, raw := range grad {
			v := raw * scale
			if v == 0 {
				continue
			}
			abs := float32(math.Abs(float64(v)))
			if abs > stats.MaxAbs {
				stats.MaxAbs = abs
			}
			stats.NonzeroCount++
			sumSquares += float64(v) * float64(v)
		}
	}
	stats.L2Norm = float32(math.Sqrt(sumSquares))
	return stats
}

func snapshotEmbeddingTrainTensors(tensors ...*backend.Tensor) []embeddingTrainTensorSnapshot {
	snapshots := make([]embeddingTrainTensorSnapshot, 0, len(tensors))
	for _, tensor := range tensors {
		if tensor == nil {
			continue
		}
		snapshots = append(snapshots, embeddingTrainTensorSnapshot{
			tensor: tensor,
			before: append([]float32(nil), tensor.F32...),
		})
	}
	return snapshots
}

func aggregateParameterDeltaStats(snapshots []embeddingTrainTensorSnapshot) EmbeddingTrainStatMetrics {
	var stats EmbeddingTrainStatMetrics
	sumSquares := float64(0)
	for _, snapshot := range snapshots {
		if snapshot.tensor == nil {
			continue
		}
		n := len(snapshot.before)
		if len(snapshot.tensor.F32) < n {
			n = len(snapshot.tensor.F32)
		}
		stats.TotalCount += n
		for i := 0; i < n; i++ {
			delta := snapshot.tensor.F32[i] - snapshot.before[i]
			if delta == 0 {
				continue
			}
			abs := float32(math.Abs(float64(delta)))
			if abs > stats.MaxAbs {
				stats.MaxAbs = abs
			}
			stats.NonzeroCount++
			sumSquares += float64(delta) * float64(delta)
		}
	}
	stats.L2Norm = float32(math.Sqrt(sumSquares))
	return stats
}

func mergeEmbeddingTrainMovementMetrics(dst *EmbeddingTrainMovementMetrics, src *EmbeddingTrainMovementMetrics) {
	if dst == nil || src == nil {
		return
	}
	dst.Gradient = mergeEmbeddingTrainStatMetrics(dst.Gradient, src.Gradient)
	dst.ParameterDelta = mergeEmbeddingTrainStatMetrics(dst.ParameterDelta, src.ParameterDelta)
}

func mergeEmbeddingTrainStatMetrics(a, b EmbeddingTrainStatMetrics) EmbeddingTrainStatMetrics {
	sumSquares := float64(a.L2Norm)*float64(a.L2Norm) + float64(b.L2Norm)*float64(b.L2Norm)
	if b.MaxAbs > a.MaxAbs {
		a.MaxAbs = b.MaxAbs
	}
	a.NonzeroCount += b.NonzeroCount
	a.TotalCount += b.TotalCount
	a.L2Norm = float32(math.Sqrt(sumSquares))
	return a
}

func (t *EmbeddingTrainer) tryGELUBackwardMul(gradOut, preAct []float32, rows, cols int, preActBinding string) ([]float32, bool) {
	if !t.fullActivationBackwardAccelEnabled() || rows == 0 || cols == 0 {
		return nil, false
	}
	if preActBinding == "" && !activationAccelElementsAllowed(rows, cols) {
		return nil, false
	}
	gradTensor := tensorF32View([]int{rows, cols}, gradOut)
	var (
		result *backend.Tensor
		err    error
	)
	if preActBinding != "" {
		result, err = t.activationAccel.RunGELUBackwardMulWithBoundPreAct(gradTensor, preActBinding)
	} else {
		result, err = t.activationAccel.RunGELUBackwardMul(
			gradTensor,
			tensorF32View([]int{rows, cols}, preAct),
		)
	}
	if err != nil || result == nil {
		return nil, false
	}
	out := result.F32
	if len(out) != len(gradOut) {
		return nil, false
	}
	return out, true
}

func (t *EmbeddingTrainer) trySoftmaxBackwardRows(gradOut, probs []float32, rows, cols int, probsBinding string) ([]float32, bool) {
	if !t.softmaxBackwardAccelEnabled() || rows == 0 || cols == 0 {
		return nil, false
	}
	if probsBinding == "" && !activationAccelElementsAllowed(rows, cols) {
		return nil, false
	}
	gradTensor := tensorF32View([]int{rows, cols}, gradOut)
	var (
		result *backend.Tensor
		err    error
	)
	if probsBinding != "" {
		result, err = t.activationAccel.RunSoftmaxBackwardRowsWithBoundProbs(gradTensor, probsBinding)
	} else {
		result, err = t.activationAccel.RunSoftmaxBackwardRows(
			gradTensor,
			tensorF32View([]int{rows, cols}, probs),
		)
	}
	if err != nil || result == nil {
		return nil, false
	}
	out := result.F32
	if len(out) != len(gradOut) {
		return nil, false
	}
	return out, true
}

func (t *EmbeddingTrainer) tryBatchedSoftmaxBackwardRows(gradOutMatrices, probsMatrices [][]float32, rows, cols int) ([][]float32, bool) {
	if !t.softmaxBackwardAccelEnabled() || len(gradOutMatrices) == 0 || len(gradOutMatrices) != len(probsMatrices) || rows == 0 || cols == 0 {
		return nil, false
	}
	if !activationAccelElementsAllowed(len(gradOutMatrices)*rows, cols) {
		return nil, false
	}
	perMatrix := rows * cols
	gradOut, ok := t.flattenFixedFloat32MatricesScratch(0, gradOutMatrices, perMatrix)
	if !ok {
		return nil, false
	}
	probs, ok := t.flattenFixedFloat32MatricesScratch(1, probsMatrices, perMatrix)
	if !ok {
		return nil, false
	}
	out, ok := t.trySoftmaxBackwardRows(gradOut, probs, len(gradOutMatrices)*rows, cols, "")
	if !ok || len(out) != len(gradOut) {
		return nil, false
	}
	return splitFloat32Views(out, len(gradOutMatrices))
}

func (t *EmbeddingTrainer) tryBatchedGELUBackwardMul(gradOutMatrices, preActMatrices [][]float32, rows, cols int) ([][]float32, bool) {
	if !t.fullActivationBackwardAccelEnabled() || len(gradOutMatrices) == 0 || len(gradOutMatrices) != len(preActMatrices) || rows == 0 || cols == 0 {
		return nil, false
	}
	if !activationAccelElementsAllowed(len(gradOutMatrices)*rows, cols) {
		return nil, false
	}
	perMatrix := rows * cols
	gradOut, ok := t.flattenFixedFloat32MatricesScratch(0, gradOutMatrices, perMatrix)
	if !ok {
		return nil, false
	}
	preAct, ok := t.flattenFixedFloat32MatricesScratch(1, preActMatrices, perMatrix)
	if !ok {
		return nil, false
	}
	out, ok := t.tryGELUBackwardMul(gradOut, preAct, len(gradOutMatrices)*rows, cols, "")
	if !ok || len(out) != len(gradOut) {
		return nil, false
	}
	return splitFloat32Views(out, len(gradOutMatrices))
}

func (t *EmbeddingTrainer) tryBatchedLayerNormBackwardRows(gradOutMatrices, normalizedMatrices, preMatrices [][]float32, rows, cols int) ([][]float32, bool) {
	if !t.fullActivationBackwardAccelEnabled() || len(gradOutMatrices) == 0 || len(gradOutMatrices) != len(normalizedMatrices) || len(gradOutMatrices) != len(preMatrices) || rows == 0 || cols == 0 {
		return nil, false
	}
	if !activationAccelElementsAllowed(len(gradOutMatrices)*rows, cols) {
		return nil, false
	}
	perMatrix := rows * cols
	gradOut, ok := t.flattenFixedFloat32MatricesScratch(0, gradOutMatrices, perMatrix)
	if !ok {
		return nil, false
	}
	normalized, ok := t.flattenFixedFloat32MatricesScratch(1, normalizedMatrices, perMatrix)
	if !ok {
		return nil, false
	}
	pre, ok := t.flattenFixedFloat32MatricesScratch(2, preMatrices, perMatrix)
	if !ok {
		return nil, false
	}
	out, ok := t.tryLayerNormBackwardRows(gradOut, normalized, pre, len(gradOutMatrices)*rows, cols, "", "")
	if !ok || len(out) != len(gradOut) {
		return nil, false
	}
	return splitFloat32Views(out, len(gradOutMatrices))
}

func (t *EmbeddingTrainer) tryLayerNormBackwardRows(gradOut, normalized, pre []float32, rows, cols int, normalizedBinding, preBinding string) ([]float32, bool) {
	if !t.fullActivationBackwardAccelEnabled() || rows == 0 || cols == 0 {
		return nil, false
	}
	if (normalizedBinding == "" || preBinding == "") && !activationAccelElementsAllowed(rows, cols) {
		return nil, false
	}
	gradTensor := tensorF32View([]int{rows, cols}, gradOut)
	var (
		result *backend.Tensor
		err    error
	)
	if normalizedBinding != "" && preBinding != "" {
		result, err = t.activationAccel.RunLayerNormBackwardRowsWithBoundInputs(gradTensor, normalizedBinding, preBinding)
	} else {
		result, err = t.activationAccel.RunLayerNormBackwardRows(
			gradTensor,
			tensorF32View([]int{rows, cols}, normalized),
			tensorF32View([]int{rows, cols}, pre),
		)
	}
	if err != nil || result == nil {
		return nil, false
	}
	out := result.F32
	if len(out) != len(gradOut) {
		return nil, false
	}
	return out, true
}

func applyOptimizerUpdate(cfg EmbeddingTrainConfig, step int, tensor, mom1, mom2 *backend.Tensor, grad []float32, scale float32) {
	if tensor == nil {
		return
	}
	switch cfg.Optimizer {
	case "", "sgd":
		for i := range tensor.F32 {
			g := grad[i] * scale
			if cfg.WeightDecay != 0 {
				g += cfg.WeightDecay * tensor.F32[i]
			}
			tensor.F32[i] -= cfg.LearningRate * g
		}
	default:
		applyAdamWUpdate(cfg, step, tensor, mom1, mom2, grad, scale)
	}
}

func applyAdamWUpdate(cfg EmbeddingTrainConfig, step int, tensor, mom1, mom2 *backend.Tensor, grad []float32, scale float32) {
	if tensor == nil || mom1 == nil || mom2 == nil {
		return
	}
	beta1Pow := float32(math.Pow(float64(cfg.Beta1), float64(step)))
	beta2Pow := float32(math.Pow(float64(cfg.Beta2), float64(step)))
	corr1 := float32(1) - beta1Pow
	corr2 := float32(1) - beta2Pow
	for i := range tensor.F32 {
		g := grad[i] * scale
		if cfg.WeightDecay != 0 {
			g += cfg.WeightDecay * tensor.F32[i]
		}
		mom1.F32[i] = cfg.Beta1*mom1.F32[i] + (1-cfg.Beta1)*g
		mom2.F32[i] = cfg.Beta2*mom2.F32[i] + (1-cfg.Beta2)*g*g
		mHat := mom1.F32[i]
		vHat := mom2.F32[i]
		if corr1 != 0 {
			mHat /= corr1
		}
		if corr2 != 0 {
			vHat /= corr2
		}
		tensor.F32[i] -= cfg.LearningRate * (mHat / (float32(math.Sqrt(float64(vHat))) + cfg.Epsilon))
	}
}

func vectorNorm(v []float32) float32 {
	sum := float64(0)
	for _, x := range v {
		sum += float64(x * x)
	}
	return float32(math.Sqrt(sum))
}

func geluForward(x float32) float32 {
	cubic := x * x * x
	inner := float32(0.7978845608) * (x + float32(0.044715)*cubic)
	return 0.5 * x * (1 + float32(math.Tanh(float64(inner))))
}

func geluForwardMode(x float32, fast bool) float32 {
	if !fast {
		return geluForward(x)
	}
	return geluForwardFast(x)
}

func fillGELUForward(dst, src []float32, fast bool) {
	if fast {
		for i, value := range src {
			dst[i] = geluForwardFast(value)
		}
		return
	}
	for i, value := range src {
		dst[i] = geluForward(value)
	}
}

func geluForwardFast(x float32) float32 {
	cubic := x * x * x
	inner := float32(0.7978845608) * (x + float32(0.044715)*cubic)
	return 0.5 * x * (1 + fastTanh(inner))
}

func geluBackward(x float32) float32 {
	cubic := x * x * x
	inner := float32(0.7978845608) * (x + float32(0.044715)*cubic)
	tanhInner := float32(math.Tanh(float64(inner)))
	sech2 := 1 - tanhInner*tanhInner
	innerGrad := float32(0.7978845608) * (1 + float32(3*0.044715)*x*x)
	return 0.5*(1+tanhInner) + 0.5*x*sech2*innerGrad
}

func geluBackwardMode(x float32, fast bool) float32 {
	if !fast {
		return geluBackward(x)
	}
	return geluBackwardFast(x)
}

func fillGELUBackwardMul(dst, gradOut, preAct []float32, fast bool) {
	if fast {
		for i, value := range preAct {
			dst[i] = gradOut[i] * geluBackwardFast(value)
		}
		return
	}
	for i, value := range preAct {
		dst[i] = gradOut[i] * geluBackward(value)
	}
}

func geluBackwardFast(x float32) float32 {
	cubic := x * x * x
	inner := float32(0.7978845608) * (x + float32(0.044715)*cubic)
	tanhInner := fastTanh(inner)
	innerGrad := float32(0.7978845608) * (1 + float32(3*0.044715)*x*x)
	return 0.5*(1+tanhInner) + 0.5*x*fastTanhDerivative(inner)*innerGrad
}

func fastTanh(x float32) float32 {
	if x >= 3 {
		return 1
	}
	if x <= -3 {
		return -1
	}
	x2 := x * x
	return x * (27 + x2) / (27 + 9*x2)
}

func fastTanhDerivative(x float32) float32 {
	if x >= 3 || x <= -3 {
		return 0
	}
	x2 := x * x
	diff := x2 - 9
	den := 3 + x2
	return (diff * diff) / (9 * den * den)
}

func (t *EmbeddingTrainer) hiddenProjectionData() []float32 {
	if t == nil || t.hiddenProjection == nil {
		return nil
	}
	return t.hiddenProjection.F32
}

func (t *EmbeddingTrainer) attentionEnabled() bool {
	return t != nil && t.attentionQuery != nil && t.attentionKey != nil && t.attentionValue != nil && t.attentionOutput != nil
}

func (t *EmbeddingTrainer) attentionResidualEnabled() bool {
	return t != nil && t.attentionEnabled() && t.manifest.AttentionResidual
}

func (t *EmbeddingTrainer) attentionLayerNormEnabled() bool {
	return t != nil && t.attentionEnabled() && t.manifest.AttentionLayerNorm
}

func (t *EmbeddingTrainer) attentionScoreScale(keyWidth int) float32 {
	if t == nil || keyWidth <= 0 {
		return 1
	}
	switch t.manifest.AttentionScoreScale {
	case EmbeddingAttentionScoreScaleKeyDimRSQ:
		return float32(1 / math.Sqrt(float64(keyWidth)))
	default:
		return 1
	}
}

func scaleFloat32Slice(values []float32, scale float32) {
	if scale == 1 {
		return
	}
	for i := range values {
		values[i] *= scale
	}
}

func (t *EmbeddingTrainer) ffnResidualEnabled() bool {
	return t != nil && t.hiddenProjection != nil && t.manifest.FFNResidual
}

func (t *EmbeddingTrainer) ffnLayerNormEnabled() bool {
	return t != nil && t.hiddenProjection != nil && t.manifest.FFNLayerNorm
}

func (t *EmbeddingTrainer) encoderRepeats() int {
	if t == nil || t.manifest.EncoderRepeats <= 0 {
		return 1
	}
	return t.manifest.EncoderRepeats
}

func tensorDataLen(t *backend.Tensor) int {
	if t == nil {
		return 0
	}
	return len(t.F32)
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
