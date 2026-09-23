package eosruntime

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sort"
	"strings"
	"sync"

	"m31labs.dev/turboquant"
)

type AOQTSidecarTrainConfig struct {
	Dim                             int
	Stages                          int
	PairingSeed                     int64
	WorkplanSeed                    int64
	OptimizerMode                   string
	ForwardConsistencyProbeRequired bool
	// ForwardConsistencyProbeOnly runs the required V7-r2 probe and returns
	// before any optimizer step. It is separate from
	// ForwardConsistencyProbeRequired so a probe-only run cannot fall through
	// into optimizer telemetry or package production.
	ForwardConsistencyProbeOnly                bool
	ForwardConsistencyProbeFoldID              string
	ForwardConsistencyProbeSplitManifestSHA256 string
	// ActualDirectionProbeRequired enables the V7-r3 deterministic actual
	// endpoint sweep. It is separate from the V7-r2 sign-consistency probe so
	// the older sign-authority contract cannot be silently relabeled.
	ActualDirectionProbeRequired            bool
	ActualDirectionProbeOnly                bool
	ActualDirectionProbeFoldID              string
	ActualDirectionProbeSplitManifestSHA256 string
	// ActualCoordinateProbeRequired enables the preregistered V7-r4 scalar
	// endpoint probe.  It is intentionally separate from the V7-r3 actual
	// direction search so the r3 contract cannot be silently relabeled.
	ActualCoordinateProbeRequired            bool
	ActualCoordinateProbeOnly                bool
	ActualCoordinateProbeFoldID              string
	ActualCoordinateProbeSplitManifestSHA256 string
	// ActualCoordinateTrainRequired enables the V7-r5 full trainer.  The
	// canonical V7-r4 receipt is an authorization/input pin; r5 never falls
	// back to Adam or the legacy STE proposal path.
	ActualCoordinateTrainRequired        bool
	ActualCoordinateTrainR4Receipt       *AOQTSidecarActualCoordinateProbeReceipt
	ActualCoordinateTrainR4ReceiptPath   string
	ActualCoordinateTrainR4ReceiptSHA256 string
	// CanonicalR4Receipt* are source-compatible aliases for callers that use
	// the shorter design terminology.
	CanonicalR4Receipt         *AOQTSidecarActualCoordinateProbeReceipt
	CanonicalR4ReceiptPath     string
	CanonicalR4ReceiptSHA256   string
	PlanOnly                   bool
	MaxSteps                   int
	LearningRate               float32
	Beta1                      float32
	Beta2                      float32
	Epsilon                    float32
	AngleCap                   float32
	MaxAngleCap                float32
	CaptureProposalReceipts    bool
	TrainingContract           string
	CandidateEligibilityPolicy *AOQTSidecarCandidateEligibilityPolicy
}

type AOQTSidecarWorkPlan struct {
	Dim                               int     `json:"dim"`
	Stages                            int     `json:"stages"`
	PairsPerStage                     int     `json:"pairs_per_stage"`
	AngleCount                        int     `json:"angle_count"`
	RowCount                          int     `json:"row_count"`
	PairCount                         int     `json:"pair_count"`
	StepCount                         int     `json:"step_count"`
	PairingSeed                       int64   `json:"pairing_seed"`
	WorkplanSeed                      int64   `json:"workplan_seed"`
	LearningRate                      float32 `json:"learning_rate"`
	OptimizerMode                     string  `json:"optimizer_mode,omitempty"`
	ForwardConsistencyProbeRequired   bool    `json:"forward_consistency_probe_required,omitempty"`
	ForwardConsistencyProbeOnly       bool    `json:"forward_consistency_probe_only,omitempty"`
	ActualDirectionProbeRequired      bool    `json:"actual_direction_probe_required,omitempty"`
	ActualDirectionProbeOnly          bool    `json:"actual_direction_probe_only,omitempty"`
	ActualCoordinateProbeRequired     bool    `json:"actual_coordinate_probe_required,omitempty"`
	ActualCoordinateProbeOnly         bool    `json:"actual_coordinate_probe_only,omitempty"`
	ActualCoordinateTrainRequired     bool    `json:"actual_coordinate_train_required,omitempty"`
	ActualCoordinateFullTrainRequired bool    `json:"actual_coordinate_full_train_required,omitempty"`
	PairingsSHA256                    string  `json:"pairings_sha256"`
	PlanOnly                          bool    `json:"plan_only"`
}

type AOQTSidecarTrainSummary struct {
	Plan                              AOQTSidecarWorkPlan                        `json:"plan"`
	ObjectiveContract                 AOQTSidecarObjectiveContract               `json:"objective_contract"`
	Steps                             int                                        `json:"steps"`
	InitialLoss                       float32                                    `json:"initial_loss"`
	FinalLoss                         float32                                    `json:"final_loss"`
	InitialObjectiveComponents        AOQTSidecarObjectiveComponents             `json:"initial_objective_components"`
	FinalObjectiveComponents          AOQTSidecarObjectiveComponents             `json:"final_objective_components"`
	InitialObjectiveActivation        AOQTSidecarObjectiveActivation             `json:"initial_objective_activation"`
	FinalObjectiveActivation          AOQTSidecarObjectiveActivation             `json:"final_objective_activation"`
	AngleL2                           float32                                    `json:"angle_l2"`
	AngleMaxAbs                       float32                                    `json:"angle_max_abs"`
	AnglesSHA256                      string                                     `json:"angles_sha256"`
	DenseMaxAbsDelta                  float64                                    `json:"dense_max_abs_delta"`
	OptimizerDiagnostics              *AOQTSidecarOptimizerDiagnostics           `json:"optimizer_diagnostics,omitempty"`
	OptimizerDiagnosticsSHA256        string                                     `json:"optimizer_diagnostics_sha256,omitempty"`
	QualityClaim                      bool                                       `json:"quality_claim"`
	TrainingContract                  string                                     `json:"training_contract,omitempty"`
	CandidateEligibilityPolicy        *AOQTSidecarCandidateEligibilityPolicy     `json:"candidate_eligibility_policy,omitempty"`
	OptimizerMode                     string                                     `json:"optimizer_mode,omitempty"`
	ForwardConsistencyProbeRequired   bool                                       `json:"forward_consistency_probe_required,omitempty"`
	ForwardConsistencyProbeOnly       bool                                       `json:"forward_consistency_probe_only,omitempty"`
	ActualDirectionProbeRequired      bool                                       `json:"actual_direction_probe_required,omitempty"`
	ActualDirectionProbeOnly          bool                                       `json:"actual_direction_probe_only,omitempty"`
	ActualCoordinateProbeRequired     bool                                       `json:"actual_coordinate_probe_required,omitempty"`
	ActualCoordinateProbeOnly         bool                                       `json:"actual_coordinate_probe_only,omitempty"`
	ActualCoordinateTrainRequired     bool                                       `json:"actual_coordinate_train_required,omitempty"`
	ActualCoordinateFullTrainRequired bool                                       `json:"actual_coordinate_full_train_required,omitempty"`
	ForwardConsistencyProbe           *AOQTSidecarForwardConsistencyProbeReceipt `json:"forward_consistency_probe,omitempty"`
	ForwardConsistencyProbeSHA256     string                                     `json:"forward_consistency_probe_sha256,omitempty"`
	ActualDirectionProbe              *AOQTSidecarActualDirectionProbeReceipt    `json:"actual_direction_probe,omitempty"`
	ActualDirectionProbeSHA256        string                                     `json:"actual_direction_probe_sha256,omitempty"`
	ActualCoordinateProbe             *AOQTSidecarActualCoordinateProbeReceipt   `json:"actual_coordinate_probe,omitempty"`
	ActualCoordinateProbeSHA256       string                                     `json:"actual_coordinate_probe_sha256,omitempty"`
	ActualCoordinateTrain             *AOQTSidecarActualCoordinateTrainReceipt   `json:"actual_coordinate_train,omitempty"`
	ActualCoordinateTrainSHA256       string                                     `json:"actual_coordinate_train_sha256,omitempty"`
}

type AOQTSidecarOptimizerDiagnostics struct {
	OptimizerMode                 string                                   `json:"optimizer_mode,omitempty"`
	DevOnly                       bool                                     `json:"dev_only,omitempty"`
	PlannedSteps                  int                                      `json:"planned_steps"`
	AttemptedSteps                int                                      `json:"attempted_steps"`
	AcceptedSteps                 int                                      `json:"accepted_steps"`
	ProposalAttempts              int                                      `json:"proposal_attempts"`
	AcceptedProposals             int                                      `json:"accepted_proposals"`
	RejectedProposals             int                                      `json:"rejected_proposals"`
	Backtracks                    int                                      `json:"backtracks"`
	MaxAttemptsPerStep            int                                      `json:"max_attempts_per_step"`
	ExhaustedSteps                int                                      `json:"exhausted_steps"`
	AdamProposalAttempts          int                                      `json:"adam_proposal_attempts,omitempty"`
	AdamAcceptedProposals         int                                      `json:"adam_accepted_proposals,omitempty"`
	AdamRejectedProposals         int                                      `json:"adam_rejected_proposals,omitempty"`
	CoordinateProposalAttempts    int                                      `json:"coordinate_proposal_attempts,omitempty"`
	CoordinateAcceptedProposals   int                                      `json:"coordinate_accepted_proposals,omitempty"`
	CoordinateRejectedProposals   int                                      `json:"coordinate_rejected_proposals,omitempty"`
	CoordinateSearchPlanCount     int                                      `json:"coordinate_search_plan_count,omitempty"`
	CoordinateTopAngles           int                                      `json:"coordinate_top_angles,omitempty"`
	CoordinateMagnitudeCount      int                                      `json:"coordinate_magnitude_count,omitempty"`
	CoordinateBlockCount          int                                      `json:"coordinate_block_count,omitempty"`
	CoordinateSearchStrategy      string                                   `json:"coordinate_search_strategy,omitempty"`
	CoordinateSearchOrderingHash  string                                   `json:"coordinate_search_ordering_sha256,omitempty"`
	CoordinateSearchLearningRate  float32                                  `json:"coordinate_search_learning_rate,omitempty"`
	CoordinateSearchAudit         string                                   `json:"coordinate_search_audit,omitempty"`
	CoordinateSearchAuditChain    string                                   `json:"coordinate_search_audit_chain,omitempty"`
	CoordinateSearchHashChain     string                                   `json:"coordinate_search_hash_chain,omitempty"`
	TrustRegionRadius             float32                                  `json:"trust_region_radius,omitempty"`
	TrustRegionProposalAttempts   int                                      `json:"trust_region_proposal_attempts,omitempty"`
	TrustRegionAcceptedBlocks     int                                      `json:"trust_region_accepted_blocks,omitempty"`
	TrustRegionAcceptedSingles    int                                      `json:"trust_region_accepted_singles,omitempty"`
	TrustRegionMaxMovedAngles     int                                      `json:"trust_region_max_moved_angles,omitempty"`
	TrustRegionMaxMagnitude       float32                                  `json:"trust_region_max_magnitude,omitempty"`
	TrustRegionAcceptedMagnitude  float32                                  `json:"trust_region_accepted_magnitude,omitempty"`
	TrustRegionMovedAngleIndices  *[]int                                   `json:"trust_region_moved_angle_indices,omitempty"`
	TrustRegionAcceptedSingleRule string                                   `json:"trust_region_accepted_single_coordinate_rule,omitempty"`
	RejectionDiagnostics          AOQTSidecarOptimizerRejectionDiagnostics `json:"rejection_diagnostics,omitempty"`
	ProposalReceipts              *AOQTSidecarOptimizerProposalReceipts    `json:"proposal_receipts,omitempty"`
}

type AOQTSidecarOptimizerProposalReceipts []AOQTSidecarOptimizerProposalReceipt

// AOQTSidecarOptimizerProposalReceipt is bounded, row-independent evidence
// for one evaluated transactional proposal. It intentionally contains no row,
// query, document, or vector payloads. The finite flags make a non-finite
// candidate explicit without attempting to marshal NaN/Inf into JSON.
type AOQTSidecarOptimizerProposalReceipt struct {
	Ordinal                   int                            `json:"ordinal"`
	Kind                      string                         `json:"kind"`
	Accepted                  bool                           `json:"accepted"`
	Reason                    string                         `json:"reason"`
	AngleMoved                bool                           `json:"angle_moved"`
	ActivationChecked         bool                           `json:"activation_checked"`
	ActivationSufficient      bool                           `json:"activation_sufficient"`
	BaselineLoss              float32                        `json:"baseline_loss"`
	BaselineLossFinite        bool                           `json:"baseline_loss_finite"`
	CandidateLoss             float32                        `json:"candidate_loss"`
	CandidateLossFinite       bool                           `json:"candidate_loss_finite"`
	BaselineComponents        AOQTSidecarObjectiveComponents `json:"baseline_components"`
	BaselineComponentsFinite  bool                           `json:"baseline_components_finite"`
	CandidateComponents       AOQTSidecarObjectiveComponents `json:"candidate_components"`
	CandidateComponentsFinite bool                           `json:"candidate_components_finite"`
	CandidateActivation       AOQTSidecarObjectiveActivation `json:"candidate_activation"`
	TotalDelta                float32                        `json:"total_delta"`
	TotalDeltaFinite          bool                           `json:"total_delta_finite"`
	ComponentDeltas           AOQTSidecarObjectiveComponents `json:"component_deltas"`
	ComponentDeltasFinite     bool                           `json:"component_deltas_finite"`
	ProposalMagnitude         float32                        `json:"proposal_magnitude,omitempty"`
	ProposalBlockSize         int                            `json:"proposal_block_size,omitempty"`
	CoordinateSearchStrategy  string                         `json:"coordinate_search_strategy,omitempty"`
	MovedAngleCount           int                            `json:"moved_angle_count,omitempty"`
	MovedAngleIndices         []int                          `json:"moved_angle_indices,omitempty"`
	DirectionSource           string                         `json:"direction_source,omitempty"`
	EndpointOrdinal           int                            `json:"endpoint_ordinal,omitempty"`
	// V7-r3 coordinate receipts carry the exact probe evidence that authorized
	// the full proposal. These fields remain absent for Adam and non-r3
	// receipts; metrics validation binds them back to the selected endpoint.
	EndpointHashSHA256         string `json:"endpoint_hash_sha256,omitempty"`
	RequestedDirectionSHA256   string `json:"requested_direction_sha256,omitempty"`
	ActualDirectionSHA256      string `json:"actual_direction_sha256,omitempty"`
	ActualDirectionProbeSHA256 string `json:"actual_direction_probe_sha256,omitempty"`
}

type AOQTSidecarOptimizerRejectionDiagnostics struct {
	CandidateEvaluations        int                                          `json:"candidate_evaluations,omitempty"`
	ReasonCounts                AOQTSidecarOptimizerRejectionReasonCounts    `json:"reason_counts,omitempty"`
	ComponentRegressionCounts   AOQTSidecarObjectiveComponentRejectionCounts `json:"component_regression_counts,omitempty"`
	LossDelta                   AOQTSidecarOptimizerDeltaStats               `json:"loss_delta,omitempty"`
	ComponentDeltas             AOQTSidecarObjectiveComponentDeltaStats      `json:"component_deltas,omitempty"`
	DominantReason              string                                       `json:"dominant_reason,omitempty"`
	DominantComponentRegression string                                       `json:"dominant_component_regression,omitempty"`
}

type AOQTSidecarOptimizerRejectionReasonCounts struct {
	NoAngleMovement       int `json:"no_angle_movement,omitempty"`
	NonFiniteLoss         int `json:"non_finite_loss,omitempty"`
	LossIncrease          int `json:"loss_increase,omitempty"`
	InvalidComponents     int `json:"invalid_components,omitempty"`
	LossComponentMismatch int `json:"loss_component_mismatch,omitempty"`
	InactiveObjective     int `json:"inactive_objective,omitempty"`
	NoQ3GainImprovement   int `json:"no_q3_gain_improvement,omitempty"`
	ComponentRegression   int `json:"component_regression,omitempty"`
}

type AOQTSidecarObjectiveComponentRejectionCounts struct {
	Q3Gain          int `json:"q3_gain,omitempty"`
	Q3OrderGuard    int `json:"q3_order_guard,omitempty"`
	Q3ScoreDistill  int `json:"q3_score_distill,omitempty"`
	Q5OrderGuard    int `json:"q5_order_guard,omitempty"`
	Q5ScoreDistill  int `json:"q5_score_distill,omitempty"`
	NFBoundaryGuard int `json:"nf_boundary_guard,omitempty"`
}

type AOQTSidecarOptimizerDeltaStats struct {
	Count int     `json:"count,omitempty"`
	Min   float32 `json:"min,omitempty"`
	Max   float32 `json:"max,omitempty"`
	Sum   float32 `json:"sum,omitempty"`
}

type AOQTSidecarObjectiveComponentDeltaStats struct {
	Q3Gain          AOQTSidecarOptimizerDeltaStats `json:"q3_gain,omitempty"`
	Q3OrderGuard    AOQTSidecarOptimizerDeltaStats `json:"q3_order_guard,omitempty"`
	Q3ScoreDistill  AOQTSidecarOptimizerDeltaStats `json:"q3_score_distill,omitempty"`
	Q5OrderGuard    AOQTSidecarOptimizerDeltaStats `json:"q5_order_guard,omitempty"`
	Q5ScoreDistill  AOQTSidecarOptimizerDeltaStats `json:"q5_score_distill,omitempty"`
	NFBoundaryGuard AOQTSidecarOptimizerDeltaStats `json:"nf_boundary_guard,omitempty"`
}

type AOQTSidecarObjectiveInput struct {
	Row        AOQTSidecarCalibrationRow
	Query      []float32
	Candidates [][]float32
}

type AOQTSidecarObjectiveResult struct {
	Loss           float32
	Components     AOQTSidecarObjectiveComponents
	QueryGrad      []float32
	CandidateGrads [][]float32
	Activation     AOQTSidecarObjectiveActivation
}

type AOQTSidecarObjectiveComponents struct {
	Q3Gain          float32 `json:"q3_gain"`
	Q3OrderGuard    float32 `json:"q3_order_guard"`
	Q3ScoreDistill  float32 `json:"q3_score_distill"`
	Q5OrderGuard    float32 `json:"q5_order_guard"`
	Q5ScoreDistill  float32 `json:"q5_score_distill"`
	NFBoundaryGuard float32 `json:"nf_boundary_guard"`
}

func (c *AOQTSidecarObjectiveComponents) Add(other AOQTSidecarObjectiveComponents) {
	c.Q3Gain += other.Q3Gain
	c.Q3OrderGuard += other.Q3OrderGuard
	c.Q3ScoreDistill += other.Q3ScoreDistill
	c.Q5OrderGuard += other.Q5OrderGuard
	c.Q5ScoreDistill += other.Q5ScoreDistill
	c.NFBoundaryGuard += other.NFBoundaryGuard
}

func (c AOQTSidecarObjectiveComponents) Sum() float32 {
	return c.Q3Gain + c.Q3OrderGuard + c.Q3ScoreDistill + c.Q5OrderGuard + c.Q5ScoreDistill + c.NFBoundaryGuard
}

type AOQTSidecarObjectiveActivation struct {
	Q3GainEligiblePairs      int `json:"q3_gain_eligible_pairs"`
	Q3GainContributingPairs  int `json:"q3_gain_contributing_pairs"`
	Q3OrderGuardPairs        int `json:"q3_order_guard_pairs"`
	Q3OrderGuardContributing int `json:"q3_order_guard_contributing_pairs"`
	// Score-distill counts are structural prepared-candidate coverage counts;
	// they do not assert that the corresponding numeric loss is non-zero.
	Q3ScoreDistillCount         int `json:"q3_score_distill_count"`
	Q5OrderGuardPairs           int `json:"q5_order_guard_pairs"`
	Q5OrderGuardContributing    int `json:"q5_order_guard_contributing_pairs"`
	Q5ScoreDistillCount         int `json:"q5_score_distill_count"`
	NFBoundaryGuardPairs        int `json:"nf_boundary_guard_pairs"`
	NFBoundaryGuardContributing int `json:"nf_boundary_guard_contributing_pairs"`
}

func (a *AOQTSidecarObjectiveActivation) Add(other AOQTSidecarObjectiveActivation) {
	a.Q3GainEligiblePairs += other.Q3GainEligiblePairs
	a.Q3GainContributingPairs += other.Q3GainContributingPairs
	a.Q3OrderGuardPairs += other.Q3OrderGuardPairs
	a.Q3OrderGuardContributing += other.Q3OrderGuardContributing
	a.Q3ScoreDistillCount += other.Q3ScoreDistillCount
	a.Q5OrderGuardPairs += other.Q5OrderGuardPairs
	a.Q5OrderGuardContributing += other.Q5OrderGuardContributing
	a.Q5ScoreDistillCount += other.Q5ScoreDistillCount
	a.NFBoundaryGuardPairs += other.NFBoundaryGuardPairs
	a.NFBoundaryGuardContributing += other.NFBoundaryGuardContributing
}

func validateAOQTObjectiveActivation(activation AOQTSidecarObjectiveActivation, label string) error {
	for _, item := range []struct {
		name  string
		value int
	}{
		{"q3_gain_eligible_pairs", activation.Q3GainEligiblePairs},
		{"q3_gain_contributing_pairs", activation.Q3GainContributingPairs},
		{"q3_order_guard_pairs", activation.Q3OrderGuardPairs},
		{"q3_order_guard_contributing_pairs", activation.Q3OrderGuardContributing},
		{"q3_score_distill_count", activation.Q3ScoreDistillCount},
		{"q5_order_guard_pairs", activation.Q5OrderGuardPairs},
		{"q5_order_guard_contributing_pairs", activation.Q5OrderGuardContributing},
		{"q5_score_distill_count", activation.Q5ScoreDistillCount},
		{"nf_boundary_guard_pairs", activation.NFBoundaryGuardPairs},
		{"nf_boundary_guard_contributing_pairs", activation.NFBoundaryGuardContributing},
	} {
		if item.value < 0 {
			return fmt.Errorf("%s.%s must be non-negative", label, item.name)
		}
	}
	return nil
}

type AOQTSidecarVectorObjective interface {
	EvaluateAOQT(input AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error)
}

type AOQTSidecarPreparedIPObjectiveContractProvider interface {
	AOQTPreparedIPObjectiveConfig() AOQTSidecarPreparedIPObjectiveConfig
}

type AOQTSidecarPreparedIPObjectiveConfig struct {
	Dim              int
	TurboQuantSeed   int64
	GainBit          int
	Q3GuardBit       int
	Q5GuardBit       int
	GainCutoff       int
	GainTau          float32
	GainMargin       float32
	GuardTau         float32
	GuardMargin      float32
	ScoreDistillTau  float32
	NFBoundarySource string
}

func (cfg AOQTSidecarPreparedIPObjectiveConfig) ObjectiveContract(weights AOQTSidecarRowWeights) AOQTSidecarObjectiveContract {
	cfg = normalizedAOQTPreparedIPObjectiveConfig(cfg)
	return AOQTSidecarObjectiveContract{
		Dim:              cfg.Dim,
		TurboQuantSeed:   cfg.TurboQuantSeed,
		GainBit:          cfg.GainBit,
		Q3GuardBit:       cfg.Q3GuardBit,
		Q5GuardBit:       cfg.Q5GuardBit,
		ScoreSurface:     AOQTSidecarPreparedIPScoreSurface,
		GainCutoff:       cfg.GainCutoff,
		GainTau:          cfg.GainTau,
		GainMargin:       cfg.GainMargin,
		GuardTau:         cfg.GuardTau,
		GuardMargin:      cfg.GuardMargin,
		ScoreDistillTau:  cfg.ScoreDistillTau,
		NFBoundarySource: cfg.NFBoundarySource,
		WeightSums:       weights,
	}
}

type AOQTSidecarPreparedIPObjective struct {
	config    AOQTSidecarPreparedIPObjectiveConfig
	workspace *aoqtPreparedIPObjectiveWorkspace
}

func NewAOQTSidecarPreparedIPObjective(cfg AOQTSidecarPreparedIPObjectiveConfig) (AOQTSidecarPreparedIPObjective, error) {
	cfg = normalizedAOQTPreparedIPObjectiveConfig(cfg)
	if err := validateAOQTPreparedIPObjectiveConfig(cfg); err != nil {
		return AOQTSidecarPreparedIPObjective{}, err
	}
	return AOQTSidecarPreparedIPObjective{config: cfg, workspace: newAOQTPreparedIPObjectiveWorkspace(cfg)}, nil
}

func (o AOQTSidecarPreparedIPObjective) AOQTPreparedIPObjectiveConfig() AOQTSidecarPreparedIPObjectiveConfig {
	return o.config
}

func (o AOQTSidecarPreparedIPObjective) EvaluateAOQT(input AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error) {
	cfg := o.config
	if err := validateAOQTPreparedIPObjectiveConfig(cfg); err != nil {
		return AOQTSidecarObjectiveResult{}, err
	}
	if len(input.Query) != cfg.Dim {
		return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT prepared-IP query dim = %d, want %d", len(input.Query), cfg.Dim)
	}
	if len(input.Candidates) == 0 {
		return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT prepared-IP candidates are required")
	}
	row := input.Row
	n := len(input.Candidates)
	if len(row.CandidateDocIDs) != n || len(row.QrelGains) != n {
		return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT prepared-IP row metadata length mismatch")
	}
	if len(row.AnchorScores.Q3) != n || len(row.AnchorScores.Q5) != n || len(row.AnchorRanks.Q3) != n || len(row.AnchorRanks.Q5) != n {
		return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT prepared-IP anchor q3/q5 metadata length mismatch")
	}
	for i, candidate := range input.Candidates {
		if len(candidate) != cfg.Dim {
			return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT prepared-IP candidate %d dim = %d, want %d", i, len(candidate), cfg.Dim)
		}
	}
	// In-package tests may install a deterministic evaluator seam on the
	// private workspace to exercise transactional failure/rollback paths while
	// retaining the exact concrete objective type required by protected-v2.
	// Constructors never populate this hook, and external packages cannot set
	// the unexported workspace field or substitute the concrete evaluator.
	if o.workspace != nil && o.workspace.evaluateOverride != nil {
		return o.workspace.evaluateOverride(input)
	}
	result := AOQTSidecarObjectiveResult{
		QueryGrad:      make([]float32, cfg.Dim),
		CandidateGrads: make([][]float32, n),
	}
	for i := range result.CandidateGrads {
		result.CandidateGrads[i] = make([]float32, cfg.Dim)
	}
	q3Gain := newAOQTPreparedIPSurfaceWithWorkspace(o.workspace, input.Query, input.Candidates, cfg.Dim, cfg.GainBit, cfg.TurboQuantSeed)
	if row.Weights.Q3Gain > 0 {
		loss, err := topkLambdaNDCGLossAndGrad(q3Gain.scores, row.QrelGains, row.CandidateDocIDs, cfg.GainCutoff, cfg.GainTau, cfg.GainMargin, row.EligiblePairMask)
		if err != nil {
			return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT q3 prepared-IP LambdaNDCG: %w", err)
		}
		if loss.EligiblePairs == 0 || loss.ContributingPairs == 0 {
			return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT q3_gain weight %.9g has no active contributing prepared-IP LambdaNDCG pairs", row.Weights.Q3Gain)
		}
		scale := row.Weights.Q3Gain
		component := loss.Loss * scale
		result.Loss += component
		result.Components.Q3Gain += component
		result.Activation.Q3GainEligiblePairs += loss.EligiblePairs
		result.Activation.Q3GainContributingPairs += loss.ContributingPairs
		q3Gain.accumulateScoreGrads(loss.Grad, scale, result.QueryGrad, result.CandidateGrads)
	}
	q3Guard := q3Gain
	if cfg.Q3GuardBit != cfg.GainBit {
		q3Guard = newAOQTPreparedIPSurfaceWithWorkspace(o.workspace, input.Query, input.Candidates, cfg.Dim, cfg.Q3GuardBit, cfg.TurboQuantSeed)
	}
	if row.Weights.Q3OrderGuard > 0 {
		loss, grads, pairs, contributing := aoqtAnchorOrderGuardLossAndGrad(q3Guard.scores, row.AnchorRanks.Q3, row.EligiblePairMask, cfg.GuardTau, cfg.GuardMargin)
		if pairs == 0 || contributing == 0 {
			return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT q3_order_guard weight %.9g has no active contributing prepared-IP guard pairs", row.Weights.Q3OrderGuard)
		}
		scale := row.Weights.Q3OrderGuard
		component := loss * scale
		result.Loss += component
		result.Components.Q3OrderGuard += component
		result.Activation.Q3OrderGuardPairs += pairs
		result.Activation.Q3OrderGuardContributing += contributing
		q3Guard.accumulateScoreGrads(grads, scale, result.QueryGrad, result.CandidateGrads)
	}
	if row.Weights.Q3ScoreDistill > 0 {
		loss, grads, count := aoqtCenteredScoreDistillLossAndGrad(q3Guard.scores, row.AnchorScores.Q3, cfg.ScoreDistillTau)
		if count == 0 {
			return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT q3_score_distill weight %.9g has no prepared-IP score coverage", row.Weights.Q3ScoreDistill)
		}
		scale := row.Weights.Q3ScoreDistill
		component := loss * scale
		result.Loss += component
		result.Components.Q3ScoreDistill += component
		result.Activation.Q3ScoreDistillCount += count
		q3Guard.accumulateScoreGrads(grads, scale, result.QueryGrad, result.CandidateGrads)
	}
	if row.Weights.NFBoundaryGuard > 0 {
		loss, grads, pairs, contributing := aoqtNFBoundaryGuardLossAndGrad(q3Guard.scores, row.AnchorRanks.Q3, row.CandidateSources, cfg.NFBoundarySource, row.EligiblePairMask, cfg.GuardTau, cfg.GuardMargin)
		if pairs == 0 || contributing == 0 {
			return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT nf_boundary_guard weight %.9g has no active contributing prepared-IP guard pairs", row.Weights.NFBoundaryGuard)
		}
		scale := row.Weights.NFBoundaryGuard
		component := loss * scale
		result.Loss += component
		result.Components.NFBoundaryGuard += component
		result.Activation.NFBoundaryGuardPairs += pairs
		result.Activation.NFBoundaryGuardContributing += contributing
		q3Guard.accumulateScoreGrads(grads, scale, result.QueryGrad, result.CandidateGrads)
	}
	if row.Weights.Q5OrderGuard > 0 || row.Weights.Q5ScoreDistill > 0 {
		q5 := newAOQTPreparedIPSurfaceWithWorkspace(o.workspace, input.Query, input.Candidates, cfg.Dim, cfg.Q5GuardBit, cfg.TurboQuantSeed)
		if row.Weights.Q5OrderGuard > 0 {
			loss, grads, pairs, contributing := aoqtAnchorOrderGuardLossAndGrad(q5.scores, row.AnchorRanks.Q5, row.EligiblePairMask, cfg.GuardTau, cfg.GuardMargin)
			if pairs == 0 || contributing == 0 {
				return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT q5_order_guard weight %.9g has no active contributing prepared-IP guard pairs", row.Weights.Q5OrderGuard)
			}
			scale := row.Weights.Q5OrderGuard
			component := loss * scale
			result.Loss += component
			result.Components.Q5OrderGuard += component
			result.Activation.Q5OrderGuardPairs += pairs
			result.Activation.Q5OrderGuardContributing += contributing
			q5.accumulateScoreGrads(grads, scale, result.QueryGrad, result.CandidateGrads)
		}
		if row.Weights.Q5ScoreDistill > 0 {
			loss, grads, count := aoqtCenteredScoreDistillLossAndGrad(q5.scores, row.AnchorScores.Q5, cfg.ScoreDistillTau)
			if count == 0 {
				return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT q5_score_distill weight %.9g has no prepared-IP score coverage", row.Weights.Q5ScoreDistill)
			}
			scale := row.Weights.Q5ScoreDistill
			component := loss * scale
			result.Loss += component
			result.Components.Q5ScoreDistill += component
			result.Activation.Q5ScoreDistillCount += count
			q5.accumulateScoreGrads(grads, scale, result.QueryGrad, result.CandidateGrads)
		}
	}
	if !isFinite32(result.Loss) {
		return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT prepared-IP loss must be finite")
	}
	return result, nil
}

type AOQTSidecarTrainer struct {
	config            AOQTSidecarTrainConfig
	eligibilityPolicy AOQTSidecarCandidateEligibilityPolicy
	topology          AOQTGivensTransform
	angles            []float32
	adamM             []float32
	adamV             []float32
	step              int
}

func NewAOQTSidecarTrainer(cfg AOQTSidecarTrainConfig) (*AOQTSidecarTrainer, error) {
	cfg = normalizedAOQTSidecarTrainConfig(cfg)
	eligibilityPolicy, err := AOQTSidecarTrainingContractPolicy(cfg.TrainingContract, cfg.CandidateEligibilityPolicy)
	if err != nil {
		return nil, err
	}
	if err := validateAOQTSidecarTrainConfig(cfg); err != nil {
		return nil, err
	}
	topology, err := NewAOQTGivensIdentityTopology(cfg.Dim, cfg.Stages, cfg.PairingSeed, cfg.AngleCap)
	if err != nil {
		return nil, err
	}
	angleCount := countAOQTAngles(topology)
	return &AOQTSidecarTrainer{
		config:            cfg,
		eligibilityPolicy: eligibilityPolicy,
		topology:          topology,
		angles:            make([]float32, angleCount),
		adamM:             make([]float32, angleCount),
		adamV:             make([]float32, angleCount),
	}, nil
}

func NewAOQTGivensIdentityTopology(dim, stages int, seed int64, angleCap float32) (AOQTGivensTransform, error) {
	if dim != AOQTSidecarDim {
		return AOQTGivensTransform{}, fmt.Errorf("AOQT topology dim = %d, want %d", dim, AOQTSidecarDim)
	}
	if stages != AOQTSidecarStages {
		return AOQTGivensTransform{}, fmt.Errorf("AOQT topology stages = %d, want %d", stages, AOQTSidecarStages)
	}
	if seed == 0 {
		return AOQTGivensTransform{}, fmt.Errorf("AOQT topology seed is required")
	}
	if angleCap <= 0 {
		angleCap = AOQTSidecarDefaultAngleCap
	}
	if angleCap > AOQTSidecarHardMaxAngleCap {
		return AOQTGivensTransform{}, fmt.Errorf("AOQT angle_cap %.8g exceeds hard max %.8g", angleCap, AOQTSidecarHardMaxAngleCap)
	}
	rng := rand.New(rand.NewSource(seed))
	transform := AOQTGivensTransform{
		Version:  AOQTTransformVersion,
		Kind:     EmbeddingPostPoolTransformAOQTGivens,
		Dim:      dim,
		Seed:     seed,
		AngleCap: angleCap,
		Stages:   make([]AOQTStage, stages),
	}
	for stage := range transform.Stages {
		perm := rng.Perm(dim)
		pairs := make([][2]int, dim/2)
		for i := range pairs {
			a, b := perm[2*i], perm[2*i+1]
			if a > b {
				a, b = b, a
			}
			pairs[i] = [2]int{a, b}
		}
		transform.Stages[stage] = AOQTStage{
			Pairs:  pairs,
			Angles: make([]float32, len(pairs)),
		}
	}
	if err := transform.Validate(); err != nil {
		return AOQTGivensTransform{}, err
	}
	pairings, err := transform.PairingsSHA256()
	if err != nil {
		return AOQTGivensTransform{}, err
	}
	angles, err := transform.AnglesSHA256()
	if err != nil {
		return AOQTGivensTransform{}, err
	}
	orth, err := transform.OrthogonalityFrobeniusPerDim()
	if err != nil {
		return AOQTGivensTransform{}, err
	}
	transform.Audit = AOQTAuditInfo{
		PairingsSHA256: pairings,
		AnglesSHA256:   angles,
		Orthogonality:  &orth,
	}
	return transform, nil
}

func (t *AOQTSidecarTrainer) Plan(set AOQTSidecarCalibrationSet) (AOQTSidecarWorkPlan, error) {
	if t == nil {
		return AOQTSidecarWorkPlan{}, fmt.Errorf("AOQT trainer is nil")
	}
	if err := set.Validate(); err != nil {
		return AOQTSidecarWorkPlan{}, err
	}
	pairings, err := t.topology.PairingsSHA256()
	if err != nil {
		return AOQTSidecarWorkPlan{}, err
	}
	if set.Manifest.Topology.PairingsSHA256 != pairings {
		return AOQTSidecarWorkPlan{}, fmt.Errorf("AOQT trainer pairings do not match manifest topology")
	}
	var pairCount int
	for _, row := range set.Rows {
		pairCount += countEligibleAOQTPairs(row)
	}
	steps := t.config.MaxSteps
	if t.config.PlanOnly || t.config.ForwardConsistencyProbeOnly || t.config.ActualDirectionProbeOnly || t.config.ActualCoordinateProbeOnly {
		steps = 0
	}
	return AOQTSidecarWorkPlan{
		Dim:                               t.config.Dim,
		Stages:                            t.config.Stages,
		PairsPerStage:                     AOQTSidecarPairsPerStage,
		AngleCount:                        len(t.angles),
		RowCount:                          len(set.Rows),
		PairCount:                         pairCount,
		StepCount:                         steps,
		PairingSeed:                       t.config.PairingSeed,
		WorkplanSeed:                      t.config.WorkplanSeed,
		LearningRate:                      t.config.LearningRate,
		OptimizerMode:                     t.config.OptimizerMode,
		ForwardConsistencyProbeRequired:   t.config.ForwardConsistencyProbeRequired,
		ForwardConsistencyProbeOnly:       t.config.ForwardConsistencyProbeOnly,
		ActualDirectionProbeRequired:      t.config.ActualDirectionProbeRequired,
		ActualDirectionProbeOnly:          t.config.ActualDirectionProbeOnly,
		ActualCoordinateProbeRequired:     t.config.ActualCoordinateProbeRequired,
		ActualCoordinateProbeOnly:         t.config.ActualCoordinateProbeOnly,
		ActualCoordinateTrainRequired:     t.config.ActualCoordinateTrainRequired || isAOQTV7R5ActualCoordinateMode(t.config.OptimizerMode),
		ActualCoordinateFullTrainRequired: t.config.ActualCoordinateTrainRequired || isAOQTV7R5ActualCoordinateMode(t.config.OptimizerMode),
		PairingsSHA256:                    pairings,
		PlanOnly:                          t.config.PlanOnly,
	}, nil
}

func (t *AOQTSidecarTrainer) Fit(set AOQTSidecarCalibrationSet, objective AOQTSidecarVectorObjective) (AOQTSidecarTrainSummary, error) {
	if t == nil {
		return AOQTSidecarTrainSummary{}, fmt.Errorf("AOQT trainer is nil")
	}
	plan, err := t.Plan(set)
	if err != nil {
		return AOQTSidecarTrainSummary{}, err
	}
	summary := AOQTSidecarTrainSummary{
		Plan:                              plan,
		ObjectiveContract:                 set.Manifest.ObjectiveContract,
		QualityClaim:                      false,
		TrainingContract:                  set.Manifest.TrainingContract,
		CandidateEligibilityPolicy:        cloneAOQTSidecarCandidateEligibilityPolicy(set.Manifest.CandidateEligibilityPolicy),
		OptimizerMode:                     t.config.OptimizerMode,
		ForwardConsistencyProbeRequired:   t.config.ForwardConsistencyProbeRequired,
		ForwardConsistencyProbeOnly:       t.config.ForwardConsistencyProbeOnly,
		ActualDirectionProbeRequired:      t.config.ActualDirectionProbeRequired,
		ActualDirectionProbeOnly:          t.config.ActualDirectionProbeOnly,
		ActualCoordinateProbeRequired:     t.config.ActualCoordinateProbeRequired,
		ActualCoordinateProbeOnly:         t.config.ActualCoordinateProbeOnly,
		ActualCoordinateTrainRequired:     t.config.ActualCoordinateTrainRequired || isAOQTV7R5ActualCoordinateMode(t.config.OptimizerMode),
		ActualCoordinateFullTrainRequired: t.config.ActualCoordinateTrainRequired || isAOQTV7R5ActualCoordinateMode(t.config.OptimizerMode),
	}
	manifestPolicy, err := AOQTSidecarTrainingContractPolicy(set.Manifest.TrainingContract, set.Manifest.CandidateEligibilityPolicy)
	if err != nil {
		if isAOQTV7R5ActualCoordinateMode(t.config.OptimizerMode) {
			if diagnostics, receipt, initErr := t.initializeV7R5FailureArtifacts(); initErr == nil {
				return t.finishV7R5Failure(summary, diagnostics, receipt, err)
			}
		}
		return AOQTSidecarTrainSummary{}, err
	}
	if !aoqtCandidateEligibilityPoliciesEqual(manifestPolicy, t.eligibilityPolicy) {
		failure := fmt.Errorf("AOQT trainer eligibility policy does not match calibration manifest training contract")
		if isAOQTV7R5ActualCoordinateMode(t.config.OptimizerMode) {
			if diagnostics, receipt, initErr := t.initializeV7R5FailureArtifacts(); initErr == nil {
				return t.finishV7R5Failure(summary, diagnostics, receipt, failure)
			}
		}
		return summary, failure
	}
	// Route r5 through its fail-closed initializer before inspecting the
	// objective.  Objective factories are allowed to fail (including a nil
	// factory), but those failures still need a bound failed r5 receipt and
	// diagnostics witness from the runner.
	if isAOQTV7R5ActualCoordinateMode(t.config.OptimizerMode) {
		return t.fitV7R5ActualCoordinate(set, objective, summary)
	}
	if t.config.PlanOnly {
		angles, err := t.Transform().AnglesSHA256()
		if err != nil {
			return summary, err
		}
		summary.AnglesSHA256 = angles
		return summary, nil
	}
	if objective == nil {
		return summary, fmt.Errorf("AOQT objective is required for non-plan training")
	}
	if err := validateAOQTFitObjectiveContract(set.Manifest.ObjectiveContract, objective); err != nil {
		return summary, err
	}
	if len(aoqtActiveProtectedComponentNames(set.Manifest.ObjectiveContract.WeightSums)) == 0 {
		return summary, fmt.Errorf("AOQT protected-v2 non-plan training requires at least one active protected objective component")
	}
	// The contract check above restricts production protected-v2 fits to the
	// prepared-IP objective. Component gradients below are isolated by masking
	// row weights, while the aggregate full-objective gradient remains the
	// transactional proposal/evaluation authority.
	diagnostics := AOQTSidecarOptimizerDiagnostics{
		PlannedSteps:       t.config.MaxSteps,
		MaxAttemptsPerStep: aoqtTransactionalMaxAttemptsPerStep,
		OptimizerMode:      t.config.OptimizerMode,
		DevOnly:            isAOQTV7DevOptimizerMode(t.config.OptimizerMode),
	}
	if t.config.CaptureProposalReceipts {
		receipts := make(AOQTSidecarOptimizerProposalReceipts, 0)
		diagnostics.ProposalReceipts = &receipts
	}
	refreshOptimizerDiagnostics := func() {
		summary.OptimizerDiagnostics = &diagnostics
		if hash, hashErr := diagnostics.SHA256(); hashErr == nil {
			summary.OptimizerDiagnosticsSHA256 = hash
		}
	}
	// The diagnostics pointer is installed before the optional r2 probe so a
	// normal dev probe failure can be serialized as a complete, fail-closed
	// artifact. Probe-only mode intentionally carries no optimizer diagnostics.
	if !t.config.ForwardConsistencyProbeOnly && !t.config.ActualDirectionProbeOnly && !t.config.ActualCoordinateProbeOnly {
		refreshOptimizerDiagnostics()
	}
	if t.config.ForwardConsistencyProbeRequired {
		receipt, probeErr := t.runForwardConsistencyProbe(set, objective)
		if probeErr != nil && receipt.FailureReason == "" {
			prefix := "probe_error: "
			if receipt.SelectionSeedSHA256 == "" {
				prefix = "precondition: "
			}
			receipt.FailureReason = prefix + probeErr.Error()
		}
		summary.ForwardConsistencyProbe = &receipt
		if probeErr != nil {
			if !t.config.ForwardConsistencyProbeOnly {
				refreshOptimizerDiagnostics()
			}
			return summary, probeErr
		}
		probeSHA, hashErr := receipt.SHA256()
		if hashErr != nil {
			if !t.config.ForwardConsistencyProbeOnly {
				refreshOptimizerDiagnostics()
			}
			return summary, hashErr
		}
		summary.ForwardConsistencyProbeSHA256 = probeSHA
		if t.config.ForwardConsistencyProbeOnly {
			// There is deliberately no optimizer path in probe-only mode. The
			// runner consumes the bound receipt and does not serialize this
			// in-memory summary as metrics or dev evidence.
			return summary, nil
		}
	}
	if t.config.ActualDirectionProbeRequired {
		receipt, probeErr := t.runActualDirectionProbe(set, objective)
		if probeErr != nil && receipt.FailureReason == "" {
			prefix := "probe_error: "
			if receipt.SelectionSeedSHA256 == "" {
				prefix = "precondition: "
			}
			receipt.FailureReason = prefix + probeErr.Error()
		}
		summary.ActualDirectionProbe = &receipt
		if probeErr != nil {
			if !t.config.ActualDirectionProbeOnly {
				refreshOptimizerDiagnostics()
			}
			return summary, probeErr
		}
		probeSHA, hashErr := receipt.SHA256()
		if hashErr != nil {
			if !t.config.ActualDirectionProbeOnly {
				refreshOptimizerDiagnostics()
			}
			return summary, hashErr
		}
		summary.ActualDirectionProbeSHA256 = probeSHA
		if t.config.ActualDirectionProbeOnly {
			// The runner serializes only the bound receipt. No metrics, transform,
			// proposal, or package artifact is produced in this mode.
			return summary, nil
		}
	}
	if t.config.ActualCoordinateProbeRequired {
		receipt, probeErr := t.runActualCoordinateProbe(set, objective)
		if probeErr != nil && receipt.FailureReason == "" {
			prefix := "probe_error: "
			if receipt.SelectionSeedSHA256 == "" {
				prefix = "precondition: "
			}
			receipt.FailureReason = prefix + probeErr.Error()
		}
		summary.ActualCoordinateProbe = &receipt
		if probeErr != nil {
			return summary, probeErr
		}
		probeSHA, hashErr := receipt.SHA256()
		if hashErr != nil {
			return summary, hashErr
		}
		summary.ActualCoordinateProbeSHA256 = probeSHA
		if t.config.ActualCoordinateProbeOnly {
			// Probe-only mode deliberately stops before optimizer telemetry,
			// metrics, transform, dev evidence, or package publication.
			return summary, nil
		}
	}
	var initialSet bool
	for step := 0; step < t.config.MaxSteps; step++ {
		rows := deterministicAOQTRowOrder(set.Rows, t.config.WorkplanSeed, step)
		loss, grad, activation, components, err := t.lossAndAngleGrad(rows, objective)
		if err != nil {
			refreshOptimizerDiagnostics()
			return summary, err
		}
		q3GainGrad, err := t.q3GainOnlyAngleGrad(rows, objective)
		if err != nil {
			refreshOptimizerDiagnostics()
			return summary, err
		}
		protectedGradients, err := t.protectedComponentAngleGrads(rows, objective, set.Manifest.ObjectiveContract.WeightSums)
		if err != nil {
			refreshOptimizerDiagnostics()
			return summary, err
		}
		if !initialSet {
			summary.InitialLoss = loss
			summary.InitialObjectiveActivation = activation
			summary.InitialObjectiveComponents = components
			initialSet = true
		}
		baseline := aoqtStepEvaluation{
			loss:       loss,
			activation: activation,
			components: components,
		}
		evaluate := func() (aoqtStepEvaluation, error) {
			loss, _, activation, components, err := t.lossAndAngleGrad(rows, objective)
			return aoqtStepEvaluation{loss: loss, activation: activation, components: components}, err
		}
		var accepted bool
		if t.config.OptimizerMode == AOQTSidecarOptimizerModeV7R3DevActualDirection {
			accepted, err = t.acceptTransactionalActualDirectionStep(set, objective, grad, q3GainGrad, baseline, evaluate, set.Manifest.ObjectiveContract.WeightSums, &diagnostics, protectedGradients)
		} else {
			accepted, err = t.acceptTransactionalAdamStep(grad, q3GainGrad, baseline, evaluate, set.Manifest.ObjectiveContract.WeightSums, &diagnostics, protectedGradients)
		}
		if err != nil {
			refreshOptimizerDiagnostics()
			return summary, err
		}
		diagnostics.AttemptedSteps++
		if !accepted {
			diagnostics.ExhaustedSteps++
			if diagnostics.AcceptedSteps == 0 {
				refreshOptimizerDiagnostics()
				return summary, fmt.Errorf("AOQT transactional optimizer accepted zero safe steps after %d proposal attempts; %s", diagnostics.ProposalAttempts, diagnostics.RejectionSummary())
			}
			break
		}
		diagnostics.AcceptedSteps++
		summary.Steps++
	}
	summary.OptimizerDiagnostics = &diagnostics
	diagnosticsSHA, err := diagnostics.SHA256()
	if err != nil {
		refreshOptimizerDiagnostics()
		return summary, err
	}
	summary.OptimizerDiagnosticsSHA256 = diagnosticsSHA
	// Reuse step-zero's deterministic row order for final telemetry so
	// float32 aggregate components remain comparable to the initial summary.
	finalRows := deterministicAOQTRowOrder(set.Rows, t.config.WorkplanSeed, 0)
	finalLoss, _, finalActivation, finalComponents, err := t.lossAndAngleGrad(finalRows, objective)
	if err != nil {
		refreshOptimizerDiagnostics()
		return summary, err
	}
	summary.FinalLoss = finalLoss
	summary.FinalObjectiveActivation = finalActivation
	summary.FinalObjectiveComponents = finalComponents
	finalTransform := t.Transform()
	angles, err := finalTransform.AnglesSHA256()
	if err != nil {
		refreshOptimizerDiagnostics()
		return summary, err
	}
	summary.AnglesSHA256 = angles
	summary.AngleL2, summary.AngleMaxAbs = aoqtAngleStats(t.angles)
	dense, err := DenseInvariantMaxAbsDelta(finalTransform, set.Rows)
	if err != nil {
		refreshOptimizerDiagnostics()
		return summary, err
	}
	summary.DenseMaxAbsDelta = dense
	return summary, nil
}

type aoqtStepEvaluation struct {
	loss       float32
	activation AOQTSidecarObjectiveActivation
	components AOQTSidecarObjectiveComponents
}

// aoqtProtectedAngleGradient is an angle-space gradient for one of the
// non-q3-gain objective components. It is used only while constructing the
// sparse proposal tail; transactional evaluation remains the authority on
// the actual candidate objective and gates.
type aoqtProtectedAngleGradient struct {
	Name string
	Grad []float32
}

type aoqtOptimizerState struct {
	angles []float32
	adamM  []float32
	adamV  []float32
	step   int
}

const (
	aoqtTransactionalMaxAttemptsPerStep                = 80
	aoqtTransactionalAdamMaxAttemptsPerStep            = 4
	aoqtTransactionalCoordinateTopAngles               = 8
	aoqtTransactionalCoordinateMagnitudeCount          = 6
	aoqtTransactionalCoordinateBlockMagnitudeCount     = 4
	aoqtTransactionalCoordinateReverseMagnitudeCount   = 2
	aoqtTransactionalCoordinateMicroTailMagnitudeCount = 3
	aoqtCoordinateSearchStrategyLegacyQ3GainPrimary    = "q3_gain_primary_fine_tail_v1"
	aoqtCoordinateSearchStrategyProtectedConeMicroTail = "q3_gain_protected_cone_micro_tail_v2"
	aoqtCoordinateSearchStrategyV7R3ActualDirection    = "actual_direction_probe_v1"
	aoqtCoordinateSearchStrategyV7R4ActualCoordinate   = "actual_coordinate_probe_v1"
	AOQTSidecarOptimizerModeV7DevTrustRegion           = "aoqt-v7-dev-trust-region-v1"
	AOQTSidecarOptimizerModeV7R2DevTrustRegion         = "aoqt-v7-r2-dev-trust-region-v1"
	AOQTSidecarOptimizerModeV7R3DevActualDirection     = "aoqt-v7-r3-dev-actual-direction-v1"
	AOQTSidecarOptimizerModeV7R4DevActualCoordinate    = "aoqt-v7-r4-dev-actual-coordinate-v1"
	// V7-r5 is the explicitly authorized full-training mode. Keep the two
	// earlier draft spellings as aliases so old callers fail closed through the
	// same dev-only path while newly emitted plans use the preregistered name.
	AOQTSidecarOptimizerModeV7R5DevActualCoordinateTrain  = "aoqt-v7-r5-dev-actual-coordinate-train-v1"
	AOQTSidecarOptimizerModeV7R5DevActualCoordinate       = AOQTSidecarOptimizerModeV7R5DevActualCoordinateTrain
	AOQTSidecarOptimizerModeV7R5DevActualCoordinateFull   = "aoqt-v7-r5-dev-actual-coordinate-full-v1"
	AOQTSidecarOptimizerModeV7R5DevActualCoordinateLegacy = "aoqt-v7-r5-dev-actual-coordinate-v1"
	AOQTSidecarV7AcceptedSingleCoordinateRule             = "aoqt-v7-dev-preregistered-accepted-single-coordinate-v1"
	aoqtCoordinateSearchStrategyV7DevTrustRegion          = "q3_gain_protected_cone_trust_region_dev_v1"
	aoqtTransactionalLossEpsilon                          = float32(1e-7)
	aoqtTransactionalQ3ImprovementMinMagnitude            = float32(0)
)

const (
	AOQTV7R2ForwardConsistencyProbeMagnitude       = float32(0.0003125)
	AOQTV7R2ForwardConsistencyProbeRowCount        = 8
	AOQTV7R2ForwardConsistencyProbeTopAngles       = 8
	AOQTV7R2ForwardConsistencyProbeCoordinateCount = AOQTSidecarDim * AOQTSidecarStages / 2
	// V7-r2 uses an exact sign comparison. Zero is the only neutral value;
	// there is no implicit numerical tolerance around zero.
	AOQTV7R2ForwardConsistencyProbeSignPolicy          = "exact-sign-zero-neutral-v1"
	AOQTV7R2ForwardConsistencyProbeSignNearZeroEpsilon = float32(0)
	AOQTV7R3ActualDirectionProbeRowCount               = 8
	AOQTV7R3ActualDirectionProbeTopAngles              = 8
	AOQTV7R3ActualDirectionProbeMagnitudeCount         = 6
	AOQTV7R3ActualDirectionProbeMaxFullCandidates      = 2
	AOQTV7R3ActualDirectionProbeEndpointCount          = 132
	AOQTV7R3ActualDirectionProbeLearningRate           = float32(0.01)
	AOQTV7R3ActualDirectionProbeDirectionSource        = "actual_probe"
	AOQTV7R4ActualCoordinateProbeRowCount              = 12
	AOQTV7R4ActualCoordinateProbeTopAngles             = 32
	AOQTV7R4ActualCoordinateProbeMagnitudeCount        = 6
	AOQTV7R4ActualCoordinateProbeEndpointCount         = 384
	AOQTV7R4ActualCoordinateProbeLearningRate          = float32(0.01)
	AOQTV7R4ActualCoordinateProbeDirectionSource       = "actual_coordinate_probe"
)

func isAOQTV7DevOptimizerMode(mode string) bool {
	return mode == AOQTSidecarOptimizerModeV7DevTrustRegion || mode == AOQTSidecarOptimizerModeV7R2DevTrustRegion || mode == AOQTSidecarOptimizerModeV7R3DevActualDirection || mode == AOQTSidecarOptimizerModeV7R4DevActualCoordinate || isAOQTV7R5ActualCoordinateMode(mode)
}

func isAOQTV7R5ActualCoordinateMode(mode string) bool {
	// V7-r5 is a preregistered authorization surface.  Do not silently
	// broaden it with draft/legacy spellings: the runner, receipts, and
	// fail-closed artifacts must all name the exact contract mode.
	return mode == AOQTSidecarOptimizerModeV7R5DevActualCoordinateTrain
}

var aoqtTransactionalCoordinateBlockSizes = []int{2, 4, 8}

func (d AOQTSidecarOptimizerDiagnostics) SHA256() (string, error) {
	data, err := json.Marshal(d)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func (d AOQTSidecarOptimizerDiagnostics) RejectionSummary() string {
	rejections := d.RejectionDiagnostics
	if rejections.CandidateEvaluations == 0 {
		return "rejection diagnostics: no evaluated candidate proposals"
	}
	rejections.finalizeDominantFields()
	parts := []string{
		fmt.Sprintf("dominant_reason=%s(%d/%d)", rejections.DominantReason, rejections.reasonCount(rejections.DominantReason), rejections.CandidateEvaluations),
		fmt.Sprintf("reason_counts=%s", rejections.ReasonCounts.summaryString()),
	}
	if rejections.DominantComponentRegression != "" {
		parts = append(parts, fmt.Sprintf("dominant_component_regression=%s(%d)", rejections.DominantComponentRegression, rejections.componentRegressionCount(rejections.DominantComponentRegression)))
	} else {
		parts = append(parts, "dominant_component_regression=none")
	}
	if name, stats, ok := rejections.ComponentDeltas.dominantPositiveMean(); ok {
		parts = append(parts, fmt.Sprintf("dominant_component_delta=%s(mean=%s max=%s)", name, formatAOQTDiagnosticFloat(stats.Mean()), formatAOQTDiagnosticFloat(stats.Max)))
	}
	if rejections.LossDelta.Count > 0 {
		parts = append(parts, fmt.Sprintf("loss_delta_mean=%s", formatAOQTDiagnosticFloat(rejections.LossDelta.Mean())))
	}
	return "rejection diagnostics: " + strings.Join(parts, "; ")
}

func (d *AOQTSidecarOptimizerDiagnostics) RecordRejection(decision aoqtTransactionalProposalDecision, baseline, candidate aoqtStepEvaluation) {
	if d == nil || decision.accepted {
		return
	}
	d.RejectionDiagnostics.record(decision, baseline, candidate)
}

const (
	aoqtProposalKindAdam       = "adam"
	aoqtProposalKindCoordinate = "coordinate"
	aoqtProposalAcceptedReason = "accepted"
)

type aoqtProposalEvidence struct {
	magnitude                  float32
	blockSize                  int
	movedAngleIndices          []int
	directionSource            string
	endpointOrdinal            int
	endpointHashSHA256         string
	requestedDirectionSHA256   string
	actualDirectionSHA256      string
	actualDirectionProbeSHA256 string
}

// RecordProposal appends one bounded receipt for an actually evaluated
// proposal and preserves the existing aggregate rejection diagnostics.
func (d *AOQTSidecarOptimizerDiagnostics) RecordProposal(ordinal int, kind string, decision aoqtTransactionalProposalDecision, baseline, candidate aoqtStepEvaluation, evidence ...aoqtProposalEvidence) {
	if d == nil {
		return
	}
	var proposalEvidence aoqtProposalEvidence
	if len(evidence) > 0 {
		proposalEvidence = evidence[0]
	}
	if d.ProposalReceipts != nil {
		*d.ProposalReceipts = append(*d.ProposalReceipts, newAOQTProposalReceipt(ordinal, kind, decision, baseline, candidate, proposalEvidence, d.CoordinateSearchStrategy))
	}
	if kind == aoqtProposalKindCoordinate && isAOQTV7DevOptimizerMode(d.OptimizerMode) && !isAOQTV7R5ActualCoordinateMode(d.OptimizerMode) {
		d.recordTrustRegionProposal(proposalEvidence, decision.accepted)
	}
	if !decision.accepted {
		d.RejectionDiagnostics.record(decision, baseline, candidate)
	}
}

func newAOQTProposalReceipt(ordinal int, kind string, decision aoqtTransactionalProposalDecision, baseline, candidate aoqtStepEvaluation, evidence aoqtProposalEvidence, coordinateSearchStrategy string) AOQTSidecarOptimizerProposalReceipt {
	receipt := AOQTSidecarOptimizerProposalReceipt{
		Ordinal:                    ordinal,
		Kind:                       kind,
		Accepted:                   decision.accepted,
		Reason:                     string(decision.reason),
		AngleMoved:                 decision.angleMoved,
		ActivationChecked:          decision.activationChecked,
		ActivationSufficient:       decision.activationSufficient,
		CandidateActivation:        candidate.activation,
		ProposalMagnitude:          evidence.magnitude,
		ProposalBlockSize:          evidence.blockSize,
		MovedAngleCount:            len(evidence.movedAngleIndices),
		MovedAngleIndices:          append([]int(nil), evidence.movedAngleIndices...),
		DirectionSource:            evidence.directionSource,
		EndpointOrdinal:            evidence.endpointOrdinal,
		EndpointHashSHA256:         evidence.endpointHashSHA256,
		RequestedDirectionSHA256:   evidence.requestedDirectionSHA256,
		ActualDirectionSHA256:      evidence.actualDirectionSHA256,
		ActualDirectionProbeSHA256: evidence.actualDirectionProbeSHA256,
	}
	if kind == aoqtProposalKindCoordinate {
		receipt.CoordinateSearchStrategy = coordinateSearchStrategy
	}
	if receipt.Accepted {
		receipt.Reason = aoqtProposalAcceptedReason
	}
	receipt.BaselineLoss, receipt.BaselineLossFinite = aoqtFiniteReceiptFloat(baseline.loss)
	receipt.CandidateLoss, receipt.CandidateLossFinite = aoqtFiniteReceiptFloat(candidate.loss)
	receipt.BaselineComponents, receipt.BaselineComponentsFinite = aoqtFiniteReceiptComponents(baseline.components)
	receipt.CandidateComponents, receipt.CandidateComponentsFinite = aoqtFiniteReceiptComponents(candidate.components)
	if receipt.BaselineLossFinite && receipt.CandidateLossFinite {
		receipt.TotalDelta = candidate.loss - baseline.loss
		receipt.TotalDeltaFinite = isFinite32(receipt.TotalDelta)
		if !receipt.TotalDeltaFinite {
			receipt.TotalDelta = 0
		}
	}
	if receipt.BaselineComponentsFinite && receipt.CandidateComponentsFinite {
		receipt.ComponentDeltas = aoqtObjectiveComponentDelta(baseline.components, candidate.components)
		receipt.ComponentDeltasFinite = aoqtFiniteReceiptComponentsValue(receipt.ComponentDeltas)
		if !receipt.ComponentDeltasFinite {
			receipt.ComponentDeltas = AOQTSidecarObjectiveComponents{}
		}
	}
	return receipt
}

func aoqtObjectiveComponentDelta(baseline, candidate AOQTSidecarObjectiveComponents) AOQTSidecarObjectiveComponents {
	return AOQTSidecarObjectiveComponents{
		Q3Gain:          candidate.Q3Gain - baseline.Q3Gain,
		Q3OrderGuard:    candidate.Q3OrderGuard - baseline.Q3OrderGuard,
		Q3ScoreDistill:  candidate.Q3ScoreDistill - baseline.Q3ScoreDistill,
		Q5OrderGuard:    candidate.Q5OrderGuard - baseline.Q5OrderGuard,
		Q5ScoreDistill:  candidate.Q5ScoreDistill - baseline.Q5ScoreDistill,
		NFBoundaryGuard: candidate.NFBoundaryGuard - baseline.NFBoundaryGuard,
	}
}

func (d *AOQTSidecarOptimizerDiagnostics) recordTrustRegionProposal(evidence aoqtProposalEvidence, accepted bool) {
	if d == nil {
		return
	}
	d.TrustRegionProposalAttempts++
	if evidence.magnitude > d.TrustRegionMaxMagnitude {
		d.TrustRegionMaxMagnitude = evidence.magnitude
	}
	if !accepted {
		return
	}
	moved := len(evidence.movedAngleIndices)
	if moved > d.TrustRegionMaxMovedAngles {
		d.TrustRegionMaxMovedAngles = moved
	}
	if moved > 1 {
		d.TrustRegionAcceptedBlocks++
	} else {
		d.TrustRegionAcceptedSingles++
	}
	if evidence.magnitude > d.TrustRegionAcceptedMagnitude {
		d.TrustRegionAcceptedMagnitude = evidence.magnitude
	}
	for _, index := range evidence.movedAngleIndices {
		if index < 0 {
			continue
		}
		if d.TrustRegionMovedAngleIndices == nil {
			indices := make([]int, 0, len(evidence.movedAngleIndices))
			d.TrustRegionMovedAngleIndices = &indices
		}
		indices := *d.TrustRegionMovedAngleIndices
		insert := sort.SearchInts(indices, index)
		if insert < len(indices) && indices[insert] == index {
			continue
		}
		indices = append(indices, 0)
		copy(indices[insert+1:], indices[insert:])
		indices[insert] = index
		d.TrustRegionMovedAngleIndices = &indices
	}
}

func aoqtFiniteReceiptFloat(value float32) (float32, bool) {
	if !isFinite32(value) {
		return 0, false
	}
	return value, true
}

func aoqtFiniteReceiptComponents(components AOQTSidecarObjectiveComponents) (AOQTSidecarObjectiveComponents, bool) {
	if !isFinite32(components.Q3Gain) || !isFinite32(components.Q3OrderGuard) || !isFinite32(components.Q3ScoreDistill) || !isFinite32(components.Q5OrderGuard) || !isFinite32(components.Q5ScoreDistill) || !isFinite32(components.NFBoundaryGuard) {
		return AOQTSidecarObjectiveComponents{}, false
	}
	return components, true
}

func (d *AOQTSidecarOptimizerRejectionDiagnostics) record(decision aoqtTransactionalProposalDecision, baseline, candidate aoqtStepEvaluation) {
	d.CandidateEvaluations++
	d.ReasonCounts.add(decision.reason)
	d.LossDelta.Record(candidate.loss - baseline.loss)
	d.ComponentDeltas.Record(baseline.components, candidate.components)
	for _, regression := range decision.componentRegressions {
		d.ComponentRegressionCounts.add(regression.name)
	}
	d.finalizeDominantFields()
}

func (d *AOQTSidecarOptimizerRejectionDiagnostics) finalizeDominantFields() {
	d.DominantReason = d.ReasonCounts.dominant()
	d.DominantComponentRegression = d.ComponentRegressionCounts.dominant()
}

func (c *AOQTSidecarOptimizerRejectionReasonCounts) add(reason aoqtTransactionalRejectionReason) {
	switch reason {
	case aoqtRejectionNoAngleMovement:
		c.NoAngleMovement++
	case aoqtRejectionNonFiniteLoss:
		c.NonFiniteLoss++
	case aoqtRejectionLossIncrease:
		c.LossIncrease++
	case aoqtRejectionInvalidComponents:
		c.InvalidComponents++
	case aoqtRejectionLossComponentMismatch:
		c.LossComponentMismatch++
	case aoqtRejectionInactiveObjective:
		c.InactiveObjective++
	case aoqtRejectionNoQ3GainImprovement:
		c.NoQ3GainImprovement++
	case aoqtRejectionComponentRegression:
		c.ComponentRegression++
	}
}

func (c AOQTSidecarOptimizerRejectionReasonCounts) count(reason string) int {
	switch reason {
	case string(aoqtRejectionNoAngleMovement):
		return c.NoAngleMovement
	case string(aoqtRejectionNonFiniteLoss):
		return c.NonFiniteLoss
	case string(aoqtRejectionLossIncrease):
		return c.LossIncrease
	case string(aoqtRejectionInvalidComponents):
		return c.InvalidComponents
	case string(aoqtRejectionLossComponentMismatch):
		return c.LossComponentMismatch
	case string(aoqtRejectionInactiveObjective):
		return c.InactiveObjective
	case string(aoqtRejectionNoQ3GainImprovement):
		return c.NoQ3GainImprovement
	case string(aoqtRejectionComponentRegression):
		return c.ComponentRegression
	default:
		return 0
	}
}

func (d AOQTSidecarOptimizerRejectionDiagnostics) reasonCount(reason string) int {
	return d.ReasonCounts.count(reason)
}

func (c AOQTSidecarOptimizerRejectionReasonCounts) dominant() string {
	var bestName string
	var bestCount int
	for _, item := range []struct {
		name  aoqtTransactionalRejectionReason
		count int
	}{
		{aoqtRejectionNoAngleMovement, c.NoAngleMovement},
		{aoqtRejectionNonFiniteLoss, c.NonFiniteLoss},
		{aoqtRejectionLossIncrease, c.LossIncrease},
		{aoqtRejectionInvalidComponents, c.InvalidComponents},
		{aoqtRejectionLossComponentMismatch, c.LossComponentMismatch},
		{aoqtRejectionInactiveObjective, c.InactiveObjective},
		{aoqtRejectionNoQ3GainImprovement, c.NoQ3GainImprovement},
		{aoqtRejectionComponentRegression, c.ComponentRegression},
	} {
		if item.count > bestCount {
			bestName = string(item.name)
			bestCount = item.count
		}
	}
	return bestName
}

func (c AOQTSidecarOptimizerRejectionReasonCounts) summaryString() string {
	parts := make([]string, 0, 8)
	for _, item := range []struct {
		name  aoqtTransactionalRejectionReason
		count int
	}{
		{aoqtRejectionNoAngleMovement, c.NoAngleMovement},
		{aoqtRejectionNonFiniteLoss, c.NonFiniteLoss},
		{aoqtRejectionLossIncrease, c.LossIncrease},
		{aoqtRejectionInvalidComponents, c.InvalidComponents},
		{aoqtRejectionLossComponentMismatch, c.LossComponentMismatch},
		{aoqtRejectionInactiveObjective, c.InactiveObjective},
		{aoqtRejectionNoQ3GainImprovement, c.NoQ3GainImprovement},
		{aoqtRejectionComponentRegression, c.ComponentRegression},
	} {
		if item.count > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", item.name, item.count))
		}
	}
	if len(parts) == 0 {
		return "none"
	}
	return strings.Join(parts, ",")
}

func (c *AOQTSidecarObjectiveComponentRejectionCounts) add(name string) {
	switch name {
	case "q3_gain":
		c.Q3Gain++
	case "q3_order_guard":
		c.Q3OrderGuard++
	case "q3_score_distill":
		c.Q3ScoreDistill++
	case "q5_order_guard":
		c.Q5OrderGuard++
	case "q5_score_distill":
		c.Q5ScoreDistill++
	case "nf_boundary_guard":
		c.NFBoundaryGuard++
	}
}

func (c AOQTSidecarObjectiveComponentRejectionCounts) count(name string) int {
	switch name {
	case "q3_gain":
		return c.Q3Gain
	case "q3_order_guard":
		return c.Q3OrderGuard
	case "q3_score_distill":
		return c.Q3ScoreDistill
	case "q5_order_guard":
		return c.Q5OrderGuard
	case "q5_score_distill":
		return c.Q5ScoreDistill
	case "nf_boundary_guard":
		return c.NFBoundaryGuard
	default:
		return 0
	}
}

func (d AOQTSidecarOptimizerRejectionDiagnostics) componentRegressionCount(name string) int {
	return d.ComponentRegressionCounts.count(name)
}

func (c AOQTSidecarObjectiveComponentRejectionCounts) dominant() string {
	var bestName string
	var bestCount int
	for _, item := range []struct {
		name  string
		count int
	}{
		{"q3_gain", c.Q3Gain},
		{"q3_order_guard", c.Q3OrderGuard},
		{"q3_score_distill", c.Q3ScoreDistill},
		{"q5_order_guard", c.Q5OrderGuard},
		{"q5_score_distill", c.Q5ScoreDistill},
		{"nf_boundary_guard", c.NFBoundaryGuard},
	} {
		if item.count > bestCount {
			bestName = item.name
			bestCount = item.count
		}
	}
	return bestName
}

func (s *AOQTSidecarOptimizerDeltaStats) Record(delta float32) {
	if !isFinite32(delta) {
		return
	}
	if s.Count == 0 || delta < s.Min {
		s.Min = delta
	}
	if s.Count == 0 || delta > s.Max {
		s.Max = delta
	}
	s.Count++
	s.Sum += delta
}

func (s AOQTSidecarOptimizerDeltaStats) Mean() float32 {
	if s.Count == 0 {
		return 0
	}
	return s.Sum / float32(s.Count)
}

func (d *AOQTSidecarObjectiveComponentDeltaStats) Record(baseline, candidate AOQTSidecarObjectiveComponents) {
	d.Q3Gain.Record(candidate.Q3Gain - baseline.Q3Gain)
	d.Q3OrderGuard.Record(candidate.Q3OrderGuard - baseline.Q3OrderGuard)
	d.Q3ScoreDistill.Record(candidate.Q3ScoreDistill - baseline.Q3ScoreDistill)
	d.Q5OrderGuard.Record(candidate.Q5OrderGuard - baseline.Q5OrderGuard)
	d.Q5ScoreDistill.Record(candidate.Q5ScoreDistill - baseline.Q5ScoreDistill)
	d.NFBoundaryGuard.Record(candidate.NFBoundaryGuard - baseline.NFBoundaryGuard)
}

func (d AOQTSidecarObjectiveComponentDeltaStats) dominantPositiveMean() (string, AOQTSidecarOptimizerDeltaStats, bool) {
	var bestName string
	var best AOQTSidecarOptimizerDeltaStats
	var ok bool
	for _, item := range []struct {
		name  string
		stats AOQTSidecarOptimizerDeltaStats
	}{
		{"q3_gain", d.Q3Gain},
		{"q3_order_guard", d.Q3OrderGuard},
		{"q3_score_distill", d.Q3ScoreDistill},
		{"q5_order_guard", d.Q5OrderGuard},
		{"q5_score_distill", d.Q5ScoreDistill},
		{"nf_boundary_guard", d.NFBoundaryGuard},
	} {
		if item.stats.Count == 0 {
			continue
		}
		if !ok || item.stats.Mean() > best.Mean() {
			bestName = item.name
			best = item.stats
			ok = true
		}
	}
	return bestName, best, ok
}

// validateAOQTRejectionDiagnostics validates the nested failure evidence that
// is included in both successful optimizer metrics and fail-closed output.
// CandidateEvaluations and the reason histogram count only rejected proposals
// (accepted proposals are not recorded here); finite delta statistics may be
// smaller when the corresponding rejected candidate was non-finite.
func validateAOQTRejectionDiagnostics(diagnostics AOQTSidecarOptimizerRejectionDiagnostics, rejectedProposals int, label string) error {
	if rejectedProposals < 0 {
		return fmt.Errorf("%s rejected_proposals must be non-negative", label)
	}
	if diagnostics.CandidateEvaluations < 0 {
		return fmt.Errorf("%s candidate_evaluations must be non-negative", label)
	}
	if diagnostics.CandidateEvaluations != rejectedProposals {
		return fmt.Errorf("%s candidate_evaluations = %d, want rejected_proposals %d", label, diagnostics.CandidateEvaluations, rejectedProposals)
	}
	reasons := []struct {
		name  string
		value int
	}{
		{"no_angle_movement", diagnostics.ReasonCounts.NoAngleMovement},
		{"non_finite_loss", diagnostics.ReasonCounts.NonFiniteLoss},
		{"loss_increase", diagnostics.ReasonCounts.LossIncrease},
		{"invalid_components", diagnostics.ReasonCounts.InvalidComponents},
		{"loss_component_mismatch", diagnostics.ReasonCounts.LossComponentMismatch},
		{"inactive_objective", diagnostics.ReasonCounts.InactiveObjective},
		{"no_q3_gain_improvement", diagnostics.ReasonCounts.NoQ3GainImprovement},
		{"component_regression", diagnostics.ReasonCounts.ComponentRegression},
	}
	reasonTotal := int64(0)
	for _, reason := range reasons {
		if reason.value < 0 {
			return fmt.Errorf("%s reason_counts.%s must be non-negative", label, reason.name)
		}
		reasonTotal += int64(reason.value)
	}
	if reasonTotal != int64(rejectedProposals) {
		return fmt.Errorf("%s reason_counts sum = %d, want rejected_proposals %d", label, reasonTotal, rejectedProposals)
	}
	components := []struct {
		name  string
		value int
	}{
		{"q3_gain", diagnostics.ComponentRegressionCounts.Q3Gain},
		{"q3_order_guard", diagnostics.ComponentRegressionCounts.Q3OrderGuard},
		{"q3_score_distill", diagnostics.ComponentRegressionCounts.Q3ScoreDistill},
		{"q5_order_guard", diagnostics.ComponentRegressionCounts.Q5OrderGuard},
		{"q5_score_distill", diagnostics.ComponentRegressionCounts.Q5ScoreDistill},
		{"nf_boundary_guard", diagnostics.ComponentRegressionCounts.NFBoundaryGuard},
	}
	componentTotal := int64(0)
	for _, component := range components {
		if component.value < 0 {
			return fmt.Errorf("%s component_regression_counts.%s must be non-negative", label, component.name)
		}
		if component.value > rejectedProposals {
			return fmt.Errorf("%s component_regression_counts.%s = %d exceeds rejected_proposals %d", label, component.name, component.value, rejectedProposals)
		}
		componentTotal += int64(component.value)
	}
	if diagnostics.ReasonCounts.ComponentRegression == 0 && componentTotal != 0 {
		return fmt.Errorf("%s component_regression_counts are non-zero without component_regression rejections", label)
	}
	if diagnostics.ReasonCounts.ComponentRegression > 0 && componentTotal < int64(diagnostics.ReasonCounts.ComponentRegression) {
		return fmt.Errorf("%s component_regression_counts sum = %d, below component_regression rejections %d", label, componentTotal, diagnostics.ReasonCounts.ComponentRegression)
	}
	if want := diagnostics.ReasonCounts.dominant(); diagnostics.DominantReason != want {
		return fmt.Errorf("%s dominant_reason = %q, want %q from reason_counts", label, diagnostics.DominantReason, want)
	}
	if want := diagnostics.ComponentRegressionCounts.dominant(); diagnostics.DominantComponentRegression != want {
		return fmt.Errorf("%s dominant_component_regression = %q, want %q from component_regression_counts", label, diagnostics.DominantComponentRegression, want)
	}
	validateStats := func(stats AOQTSidecarOptimizerDeltaStats, name string) error {
		if stats.Count < 0 {
			return fmt.Errorf("%s %s.count must be non-negative", label, name)
		}
		if stats.Count > diagnostics.CandidateEvaluations {
			return fmt.Errorf("%s %s.count = %d exceeds candidate_evaluations %d", label, name, stats.Count, diagnostics.CandidateEvaluations)
		}
		if !isFinite32(stats.Min) || !isFinite32(stats.Max) || !isFinite32(stats.Sum) {
			return fmt.Errorf("%s %s statistics must be finite", label, name)
		}
		if stats.Count == 0 {
			if stats.Min != 0 || stats.Max != 0 || stats.Sum != 0 {
				return fmt.Errorf("%s %s statistics must be zero when count is zero", label, name)
			}
			return nil
		}
		if stats.Min > stats.Max {
			return fmt.Errorf("%s %s.min = %.9g exceeds max %.9g", label, name, stats.Min, stats.Max)
		}
		lower := float64(stats.Min) * float64(stats.Count)
		upper := float64(stats.Max) * float64(stats.Count)
		sum := float64(stats.Sum)
		tolerance := 1e-4 * math.Max(1, math.Max(math.Abs(lower), math.Max(math.Abs(upper), math.Abs(sum))))
		if sum < lower-tolerance || sum > upper+tolerance {
			return fmt.Errorf("%s %s.sum = %.9g is outside [%.9g, %.9g]", label, name, stats.Sum, lower, upper)
		}
		return nil
	}
	if err := validateStats(diagnostics.LossDelta, "loss_delta"); err != nil {
		return err
	}
	for _, item := range []struct {
		name  string
		stats AOQTSidecarOptimizerDeltaStats
	}{
		{"component_deltas.q3_gain", diagnostics.ComponentDeltas.Q3Gain},
		{"component_deltas.q3_order_guard", diagnostics.ComponentDeltas.Q3OrderGuard},
		{"component_deltas.q3_score_distill", diagnostics.ComponentDeltas.Q3ScoreDistill},
		{"component_deltas.q5_order_guard", diagnostics.ComponentDeltas.Q5OrderGuard},
		{"component_deltas.q5_score_distill", diagnostics.ComponentDeltas.Q5ScoreDistill},
		{"component_deltas.nf_boundary_guard", diagnostics.ComponentDeltas.NFBoundaryGuard},
	} {
		if err := validateStats(item.stats, item.name); err != nil {
			return err
		}
	}
	return nil
}

func validateAOQTOptimizerProposalReceipts(diagnostics AOQTSidecarOptimizerDiagnostics, label string) error {
	if diagnostics.ProposalReceipts == nil {
		return nil
	}
	receipts := *diagnostics.ProposalReceipts
	if len(receipts) == 0 {
		if diagnostics.ProposalAttempts != 0 {
			return fmt.Errorf("%s proposal_receipts are empty with %d proposal_attempts", label, diagnostics.ProposalAttempts)
		}
		return nil
	}
	if diagnostics.ProposalAttempts <= 0 {
		return fmt.Errorf("%s proposal_receipts require positive proposal_attempts", label)
	}
	if len(receipts) != diagnostics.ProposalAttempts {
		return fmt.Errorf("%s proposal_receipts = %d, want proposal_attempts %d", label, len(receipts), diagnostics.ProposalAttempts)
	}
	if diagnostics.AttemptedSteps <= 0 || diagnostics.MaxAttemptsPerStep <= 0 {
		return fmt.Errorf("%s proposal_receipts require positive attempted_steps and max_attempts_per_step", label)
	}
	maxReceipts := int64(diagnostics.AttemptedSteps) * int64(diagnostics.MaxAttemptsPerStep)
	if int64(len(receipts)) > maxReceipts {
		return fmt.Errorf("%s proposal_receipts = %d exceeds actual max attempts %d", label, len(receipts), maxReceipts)
	}

	var (
		adamAttempts, coordinateAttempts int
		adamAccepted, coordinateAccepted int
		accepted, rejected               int
		trustRegionAcceptedBlocks        int
		trustRegionAcceptedSingles       int
		trustRegionMaxMovedAngles        int
		trustRegionMovedAngleSet         []int
		reasonCounts                     AOQTSidecarOptimizerRejectionReasonCounts
	)
	for i, receipt := range receipts {
		if receipt.Ordinal != i+1 {
			return fmt.Errorf("%s proposal_receipts[%d].ordinal = %d, want %d", label, i, receipt.Ordinal, i+1)
		}
		switch receipt.Kind {
		case aoqtProposalKindAdam:
			adamAttempts++
		case aoqtProposalKindCoordinate:
			coordinateAttempts++
		default:
			return fmt.Errorf("%s proposal_receipts[%d].kind %q is unsupported", label, i, receipt.Kind)
		}
		if diagnostics.OptimizerMode == AOQTSidecarOptimizerModeV7R3DevActualDirection {
			if receipt.Kind == aoqtProposalKindCoordinate {
				if receipt.DirectionSource != AOQTV7R3ActualDirectionProbeDirectionSource || receipt.EndpointOrdinal <= 0 || receipt.EndpointOrdinal > AOQTV7R3ActualDirectionProbeEndpointCount {
					return fmt.Errorf("%s proposal_receipts[%d] V7-r3 coordinate proposal must bind actual_probe direction_source and endpoint ordinal", label, i)
				}
				provenanceCount := 0
				for _, value := range []string{receipt.EndpointHashSHA256, receipt.RequestedDirectionSHA256, receipt.ActualDirectionSHA256, receipt.ActualDirectionProbeSHA256} {
					if value != "" {
						provenanceCount++
					}
				}
				if receipt.Accepted && provenanceCount != 4 {
					return fmt.Errorf("%s proposal_receipts[%d] accepted V7-r3 coordinate proposal must carry complete endpoint provenance", label, i)
				}
				if provenanceCount != 0 && provenanceCount != 4 {
					return fmt.Errorf("%s proposal_receipts[%d] V7-r3 coordinate endpoint provenance is incomplete", label, i)
				}
				if provenanceCount == 4 {
					for name, value := range map[string]string{
						"endpoint_hash_sha256":          receipt.EndpointHashSHA256,
						"requested_direction_sha256":    receipt.RequestedDirectionSHA256,
						"actual_direction_sha256":       receipt.ActualDirectionSHA256,
						"actual_direction_probe_sha256": receipt.ActualDirectionProbeSHA256,
					} {
						if err := validateAOQTSHA256(value, fmt.Sprintf("%s proposal_receipts[%d] %s", label, i, name)); err != nil {
							return err
						}
					}
				}
			} else if receipt.DirectionSource != "" || receipt.EndpointOrdinal != 0 {
				return fmt.Errorf("%s proposal_receipts[%d] V7-r3 Adam proposal must not carry actual-probe direction metadata", label, i)
			}
			if receipt.Kind == aoqtProposalKindAdam && (receipt.EndpointHashSHA256 != "" || receipt.RequestedDirectionSHA256 != "" || receipt.ActualDirectionSHA256 != "" || receipt.ActualDirectionProbeSHA256 != "") {
				return fmt.Errorf("%s proposal_receipts[%d] V7-r3 Adam proposal must not carry endpoint provenance", label, i)
			}
		} else if receipt.DirectionSource != "" || receipt.EndpointOrdinal != 0 {
			return fmt.Errorf("%s proposal_receipts[%d] non-r3 proposal must not carry actual-probe direction metadata", label, i)
		} else if receipt.EndpointHashSHA256 != "" || receipt.RequestedDirectionSHA256 != "" || receipt.ActualDirectionSHA256 != "" || receipt.ActualDirectionProbeSHA256 != "" {
			return fmt.Errorf("%s proposal_receipts[%d] non-r3 proposal must not carry endpoint provenance", label, i)
		}
		switch receipt.Reason {
		case aoqtProposalAcceptedReason:
			if !receipt.Accepted {
				return fmt.Errorf("%s proposal_receipts[%d] marks a rejected proposal as accepted", label, i)
			}
		case string(aoqtRejectionNoAngleMovement), string(aoqtRejectionNonFiniteLoss), string(aoqtRejectionLossIncrease), string(aoqtRejectionInvalidComponents), string(aoqtRejectionLossComponentMismatch), string(aoqtRejectionInactiveObjective), string(aoqtRejectionNoQ3GainImprovement), string(aoqtRejectionComponentRegression):
			if receipt.Accepted {
				return fmt.Errorf("%s proposal_receipts[%d] marks a rejected reason %q as accepted", label, i, receipt.Reason)
			}
			reasonCounts.add(aoqtTransactionalRejectionReason(receipt.Reason))
		default:
			return fmt.Errorf("%s proposal_receipts[%d].reason %q is unsupported", label, i, receipt.Reason)
		}
		if receipt.Accepted {
			accepted++
			if receipt.Kind == aoqtProposalKindAdam {
				adamAccepted++
			} else {
				coordinateAccepted++
				if isAOQTV7DevOptimizerMode(diagnostics.OptimizerMode) && !isAOQTV7R5ActualCoordinateMode(diagnostics.OptimizerMode) {
					if receipt.MovedAngleCount <= 0 {
						return fmt.Errorf("%s proposal_receipts[%d] accepted V7 trust-region coordinate proposal must record moved angles", label, i)
					}
					if receipt.MovedAngleCount > trustRegionMaxMovedAngles {
						trustRegionMaxMovedAngles = receipt.MovedAngleCount
					}
					if receipt.MovedAngleCount > 1 {
						trustRegionAcceptedBlocks++
					} else {
						trustRegionAcceptedSingles++
					}
					for _, index := range receipt.MovedAngleIndices {
						insert := sort.SearchInts(trustRegionMovedAngleSet, index)
						if insert < len(trustRegionMovedAngleSet) && trustRegionMovedAngleSet[insert] == index {
							continue
						}
						trustRegionMovedAngleSet = append(trustRegionMovedAngleSet, 0)
						copy(trustRegionMovedAngleSet[insert+1:], trustRegionMovedAngleSet[insert:])
						trustRegionMovedAngleSet[insert] = index
					}
				}
			}
		} else {
			rejected++
		}
		if !receipt.BaselineLossFinite || !isFinite32(receipt.BaselineLoss) {
			return fmt.Errorf("%s proposal_receipts[%d] baseline loss evidence is not finite", label, i)
		}
		if !receipt.BaselineComponentsFinite || !aoqtFiniteReceiptComponentsValue(receipt.BaselineComponents) {
			return fmt.Errorf("%s proposal_receipts[%d] baseline component evidence is not finite", label, i)
		}
		if receipt.CandidateLossFinite {
			if !isFinite32(receipt.CandidateLoss) {
				return fmt.Errorf("%s proposal_receipts[%d] candidate loss is marked finite but is not finite", label, i)
			}
		} else {
			if receipt.CandidateLoss != 0 || (receipt.Reason != string(aoqtRejectionNonFiniteLoss) && receipt.Reason != string(aoqtRejectionNoAngleMovement)) {
				return fmt.Errorf("%s proposal_receipts[%d] unavailable candidate loss evidence is inconsistent", label, i)
			}
		}
		if receipt.CandidateComponentsFinite {
			if !aoqtFiniteReceiptComponentsValue(receipt.CandidateComponents) {
				return fmt.Errorf("%s proposal_receipts[%d] candidate components are marked finite but are not finite", label, i)
			}
		} else if !aoqtReceiptComponentsZero(receipt.CandidateComponents) {
			return fmt.Errorf("%s proposal_receipts[%d] unavailable candidate component evidence must be zero", label, i)
		}
		if receipt.TotalDeltaFinite {
			if !isFinite32(receipt.TotalDelta) || !receipt.BaselineLossFinite || !receipt.CandidateLossFinite || receipt.TotalDelta != receipt.CandidateLoss-receipt.BaselineLoss {
				return fmt.Errorf("%s proposal_receipts[%d] total_delta is inconsistent with loss evidence", label, i)
			}
		} else if receipt.TotalDelta != 0 {
			return fmt.Errorf("%s proposal_receipts[%d] unavailable total_delta must be zero", label, i)
		}
		if receipt.ComponentDeltasFinite {
			if !aoqtFiniteReceiptComponentsValue(receipt.ComponentDeltas) || !receipt.BaselineComponentsFinite || !receipt.CandidateComponentsFinite || receipt.ComponentDeltas != aoqtObjectiveComponentDelta(receipt.BaselineComponents, receipt.CandidateComponents) {
				return fmt.Errorf("%s proposal_receipts[%d] component_deltas are inconsistent with component evidence", label, i)
			}
		} else if !aoqtReceiptComponentsZero(receipt.ComponentDeltas) {
			return fmt.Errorf("%s proposal_receipts[%d] unavailable component_deltas must be zero", label, i)
		}
		if err := validateAOQTObjectiveActivation(receipt.CandidateActivation, fmt.Sprintf("%s proposal_receipts[%d].candidate_activation", label, i)); err != nil {
			return err
		}
		if receipt.ProposalMagnitude < 0 || !isFinite32(receipt.ProposalMagnitude) {
			return fmt.Errorf("%s proposal_receipts[%d].proposal_magnitude must be finite and non-negative", label, i)
		}
		if receipt.ProposalBlockSize < 0 {
			return fmt.Errorf("%s proposal_receipts[%d].proposal_block_size must be non-negative", label, i)
		}
		if receipt.MovedAngleCount != len(receipt.MovedAngleIndices) {
			return fmt.Errorf("%s proposal_receipts[%d].moved_angle_count = %d, want moved_angle_indices length %d", label, i, receipt.MovedAngleCount, len(receipt.MovedAngleIndices))
		}
		for j, index := range receipt.MovedAngleIndices {
			if index < 0 {
				return fmt.Errorf("%s proposal_receipts[%d].moved_angle_indices[%d] must be non-negative", label, i, j)
			}
			if j > 0 && index <= receipt.MovedAngleIndices[j-1] {
				return fmt.Errorf("%s proposal_receipts[%d].moved_angle_indices must be strictly increasing", label, i)
			}
		}
		if receipt.ActivationSufficient && !receipt.ActivationChecked {
			return fmt.Errorf("%s proposal_receipts[%d] activation_sufficient requires activation_checked", label, i)
		}
		if receipt.Accepted {
			if !receipt.AngleMoved || !receipt.ActivationChecked || !receipt.ActivationSufficient || !receipt.CandidateLossFinite || !receipt.CandidateComponentsFinite {
				return fmt.Errorf("%s proposal_receipts[%d] accepted evidence is incomplete", label, i)
			}
		} else {
			if !receipt.AngleMoved && receipt.Reason != string(aoqtRejectionNoAngleMovement) {
				return fmt.Errorf("%s proposal_receipts[%d] angle movement evidence is inconsistent", label, i)
			}
			if receipt.Reason == string(aoqtRejectionNoAngleMovement) && receipt.AngleMoved {
				return fmt.Errorf("%s proposal_receipts[%d] no-angle reason has angle_moved=true", label, i)
			}
			if receipt.Reason == string(aoqtRejectionInactiveObjective) && (!receipt.ActivationChecked || receipt.ActivationSufficient) {
				return fmt.Errorf("%s proposal_receipts[%d] inactive-objective activation evidence is inconsistent", label, i)
			}
			if (receipt.Reason == string(aoqtRejectionNoQ3GainImprovement) || receipt.Reason == string(aoqtRejectionComponentRegression)) && (!receipt.ActivationChecked || !receipt.ActivationSufficient) {
				return fmt.Errorf("%s proposal_receipts[%d] post-activation rejection evidence is incomplete", label, i)
			}
		}
	}
	if adamAttempts != diagnostics.AdamProposalAttempts || coordinateAttempts != diagnostics.CoordinateProposalAttempts {
		return fmt.Errorf("%s proposal_receipts kind accounting mismatch", label)
	}
	if adamAccepted != diagnostics.AdamAcceptedProposals || coordinateAccepted != diagnostics.CoordinateAcceptedProposals {
		return fmt.Errorf("%s proposal_receipts accepted kind accounting mismatch", label)
	}
	if accepted != diagnostics.AcceptedProposals || rejected != diagnostics.RejectedProposals {
		return fmt.Errorf("%s proposal_receipts accepted/rejected accounting mismatch", label)
	}
	if isAOQTV7DevOptimizerMode(diagnostics.OptimizerMode) && !isAOQTV7R5ActualCoordinateMode(diagnostics.OptimizerMode) {
		if trustRegionAcceptedBlocks != diagnostics.TrustRegionAcceptedBlocks || trustRegionAcceptedSingles != diagnostics.TrustRegionAcceptedSingles {
			return fmt.Errorf("%s proposal_receipts trust-region accepted block/single accounting mismatch", label)
		}
		if trustRegionMaxMovedAngles != diagnostics.TrustRegionMaxMovedAngles {
			return fmt.Errorf("%s proposal_receipts trust_region_max_moved_angles mismatch", label)
		}
		if diagnostics.TrustRegionMovedAngleIndices == nil {
			if len(trustRegionMovedAngleSet) != 0 {
				return fmt.Errorf("%s proposal_receipts trust_region_moved_angle_indices missing accepted moved angles", label)
			}
		} else if !slices.Equal(*diagnostics.TrustRegionMovedAngleIndices, trustRegionMovedAngleSet) {
			return fmt.Errorf("%s proposal_receipts trust_region_moved_angle_indices mismatch", label)
		}
	}
	if reasonCounts != diagnostics.RejectionDiagnostics.ReasonCounts {
		return fmt.Errorf("%s proposal_receipts reason accounting mismatch", label)
	}
	return nil
}

func aoqtFiniteReceiptComponentsValue(components AOQTSidecarObjectiveComponents) bool {
	_, ok := aoqtFiniteReceiptComponents(components)
	return ok
}

func aoqtReceiptComponentsZero(components AOQTSidecarObjectiveComponents) bool {
	return components == (AOQTSidecarObjectiveComponents{})
}

// validateAOQTOptimizerPathDiagnostics is shared by metrics and fail-closed
// validation so top-level/path counters and their nested rejection evidence
// cannot drift between the two output formats.
func validateAOQTOptimizerPathDiagnostics(diagnostics AOQTSidecarOptimizerDiagnostics, label string) error {
	for _, item := range []struct {
		name  string
		value int
	}{
		{"planned_steps", diagnostics.PlannedSteps},
		{"attempted_steps", diagnostics.AttemptedSteps},
		{"accepted_steps", diagnostics.AcceptedSteps},
		{"proposal_attempts", diagnostics.ProposalAttempts},
		{"accepted_proposals", diagnostics.AcceptedProposals},
		{"rejected_proposals", diagnostics.RejectedProposals},
		{"backtracks", diagnostics.Backtracks},
		{"max_attempts_per_step", diagnostics.MaxAttemptsPerStep},
		{"exhausted_steps", diagnostics.ExhaustedSteps},
		{"adam_proposal_attempts", diagnostics.AdamProposalAttempts},
		{"adam_accepted_proposals", diagnostics.AdamAcceptedProposals},
		{"adam_rejected_proposals", diagnostics.AdamRejectedProposals},
		{"coordinate_proposal_attempts", diagnostics.CoordinateProposalAttempts},
		{"coordinate_accepted_proposals", diagnostics.CoordinateAcceptedProposals},
		{"coordinate_rejected_proposals", diagnostics.CoordinateRejectedProposals},
		{"coordinate_search_plan_count", diagnostics.CoordinateSearchPlanCount},
		{"coordinate_top_angles", diagnostics.CoordinateTopAngles},
		{"coordinate_magnitude_count", diagnostics.CoordinateMagnitudeCount},
		{"coordinate_block_count", diagnostics.CoordinateBlockCount},
		{"trust_region_proposal_attempts", diagnostics.TrustRegionProposalAttempts},
		{"trust_region_accepted_blocks", diagnostics.TrustRegionAcceptedBlocks},
		{"trust_region_accepted_singles", diagnostics.TrustRegionAcceptedSingles},
		{"trust_region_max_moved_angles", diagnostics.TrustRegionMaxMovedAngles},
	} {
		if item.value < 0 {
			return fmt.Errorf("%s %s must be non-negative", label, item.name)
		}
	}
	switch diagnostics.OptimizerMode {
	case "":
		if diagnostics.DevOnly || diagnostics.TrustRegionRadius != 0 || diagnostics.TrustRegionProposalAttempts != 0 || diagnostics.TrustRegionAcceptedBlocks != 0 || diagnostics.TrustRegionAcceptedSingles != 0 || diagnostics.TrustRegionMaxMovedAngles != 0 || diagnostics.TrustRegionMaxMagnitude != 0 || diagnostics.TrustRegionAcceptedMagnitude != 0 || diagnostics.TrustRegionMovedAngleIndices != nil || diagnostics.TrustRegionAcceptedSingleRule != "" {
			return fmt.Errorf("%s dev trust-region fields require explicit dev optimizer mode", label)
		}
	case AOQTSidecarOptimizerModeV7DevTrustRegion, AOQTSidecarOptimizerModeV7R2DevTrustRegion, AOQTSidecarOptimizerModeV7R3DevActualDirection:
		if !diagnostics.DevOnly {
			return fmt.Errorf("%s dev_only is required for V7 dev trust-region mode", label)
		}
		if diagnostics.TrustRegionAcceptedSingleRule != "" && diagnostics.TrustRegionAcceptedSingleRule != AOQTSidecarV7AcceptedSingleCoordinateRule {
			return fmt.Errorf("%s trust_region_accepted_single_coordinate_rule is unsupported", label)
		}
		if diagnostics.TrustRegionRadius < 0 || diagnostics.TrustRegionMaxMagnitude < 0 || diagnostics.TrustRegionAcceptedMagnitude < 0 || !isFinite32(diagnostics.TrustRegionRadius) || !isFinite32(diagnostics.TrustRegionMaxMagnitude) || !isFinite32(diagnostics.TrustRegionAcceptedMagnitude) {
			return fmt.Errorf("%s trust-region magnitudes must be finite and non-negative", label)
		}
	default:
		return fmt.Errorf("%s optimizer_mode %q is unsupported", label, diagnostics.OptimizerMode)
	}
	if diagnostics.TrustRegionMovedAngleIndices != nil {
		for i, index := range *diagnostics.TrustRegionMovedAngleIndices {
			if index < 0 {
				return fmt.Errorf("%s trust_region_moved_angle_indices[%d] must be non-negative", label, i)
			}
			if i > 0 && index <= (*diagnostics.TrustRegionMovedAngleIndices)[i-1] {
				return fmt.Errorf("%s trust_region_moved_angle_indices must be strictly increasing", label)
			}
		}
	}
	hasPathDiagnostics := diagnostics.AdamProposalAttempts != 0 ||
		diagnostics.AdamAcceptedProposals != 0 ||
		diagnostics.AdamRejectedProposals != 0 ||
		diagnostics.CoordinateProposalAttempts != 0 ||
		diagnostics.CoordinateAcceptedProposals != 0 ||
		diagnostics.CoordinateRejectedProposals != 0
	if !hasPathDiagnostics {
		if diagnostics.ProposalAttempts != 0 || diagnostics.AcceptedProposals != 0 || diagnostics.RejectedProposals != 0 || diagnostics.Backtracks != 0 {
			return fmt.Errorf("%s top-level proposal accounting requires optimizer path diagnostics", label)
		}
		if diagnostics.CoordinateSearchPlanCount != 0 || diagnostics.CoordinateSearchStrategy != "" || diagnostics.CoordinateSearchOrderingHash != "" || diagnostics.CoordinateSearchLearningRate != 0 || diagnostics.CoordinateSearchAudit != "" || diagnostics.CoordinateSearchAuditChain != "" || diagnostics.CoordinateSearchHashChain != "" || diagnostics.CoordinateTopAngles != 0 || diagnostics.CoordinateMagnitudeCount != 0 || diagnostics.CoordinateBlockCount != 0 {
			return fmt.Errorf("%s coordinate audit is present without optimizer path diagnostics", label)
		}
		if err := validateAOQTOptimizerProposalReceipts(diagnostics, label); err != nil {
			return err
		}
		return validateAOQTRejectionDiagnostics(diagnostics.RejectionDiagnostics, 0, label+" rejection diagnostics")
	}
	if diagnostics.AdamProposalAttempts != diagnostics.AdamAcceptedProposals+diagnostics.AdamRejectedProposals {
		return fmt.Errorf("%s adam proposal accounting mismatch", label)
	}
	if diagnostics.CoordinateProposalAttempts != diagnostics.CoordinateAcceptedProposals+diagnostics.CoordinateRejectedProposals {
		return fmt.Errorf("%s coordinate proposal accounting mismatch", label)
	}
	if diagnostics.AdamProposalAttempts+diagnostics.CoordinateProposalAttempts != diagnostics.ProposalAttempts {
		return fmt.Errorf("%s optimizer-path proposal accounting mismatch", label)
	}
	if diagnostics.AdamAcceptedProposals+diagnostics.CoordinateAcceptedProposals != diagnostics.AcceptedProposals {
		return fmt.Errorf("%s optimizer-path accepted proposal accounting mismatch", label)
	}
	if diagnostics.AdamRejectedProposals+diagnostics.CoordinateRejectedProposals != diagnostics.RejectedProposals {
		return fmt.Errorf("%s optimizer-path rejected proposal accounting mismatch", label)
	}
	if diagnostics.CoordinateSearchPlanCount > diagnostics.AttemptedSteps {
		return fmt.Errorf("%s coordinate_search_plan_count = %d exceeds attempted_steps %d", label, diagnostics.CoordinateSearchPlanCount, diagnostics.AttemptedSteps)
	}
	if diagnostics.CoordinateSearchPlanCount == 0 && (diagnostics.CoordinateProposalAttempts != 0 || diagnostics.CoordinateSearchStrategy != "" || diagnostics.CoordinateSearchOrderingHash != "" || diagnostics.CoordinateSearchLearningRate != 0 || diagnostics.CoordinateSearchAudit != "" || diagnostics.CoordinateSearchAuditChain != "" || diagnostics.CoordinateSearchHashChain != "" || diagnostics.CoordinateTopAngles != 0 || diagnostics.CoordinateMagnitudeCount != 0 || diagnostics.CoordinateBlockCount != 0) {
		return fmt.Errorf("%s coordinate audit is present without coordinate plan history", label)
	}
	if err := validateAOQTOptimizerProposalReceipts(diagnostics, label); err != nil {
		return err
	}
	return validateAOQTRejectionDiagnostics(diagnostics.RejectionDiagnostics, diagnostics.RejectedProposals, label+" rejection diagnostics")
}

func formatAOQTDiagnosticFloat(value float32) string {
	return fmt.Sprintf("%.9g", value)
}

func (t *AOQTSidecarTrainer) acceptTransactionalAdamStep(grad, q3GainGrad []float32, baseline aoqtStepEvaluation, evaluate func() (aoqtStepEvaluation, error), weights AOQTSidecarRowWeights, diagnostics *AOQTSidecarOptimizerDiagnostics, protectedGradientSets ...[]aoqtProtectedAngleGradient) (bool, error) {
	if evaluate == nil {
		return false, fmt.Errorf("AOQT transactional optimizer evaluator is required")
	}
	if diagnostics == nil {
		return false, fmt.Errorf("AOQT transactional optimizer diagnostics are required")
	}
	if !isFinite32(baseline.loss) {
		return false, fmt.Errorf("AOQT transactional optimizer baseline loss must be finite")
	}
	if err := validateAOQTObjectiveComponents(baseline.components, "AOQT transactional optimizer baseline components"); err != nil {
		return false, err
	}
	if err := validateAOQTLossMatchesComponents(baseline.loss, baseline.components, "AOQT transactional optimizer baseline"); err != nil {
		return false, err
	}
	if len(q3GainGrad) != len(grad) {
		return false, fmt.Errorf("AOQT q3-gain coordinate gradient count = %d, want %d", len(q3GainGrad), len(grad))
	}
	protectedGradients, protectedSetProvided, err := aoqtParseProtectedGradientSets(protectedGradientSets)
	if err != nil {
		return false, err
	}
	if protectedSetProvided {
		if err := validateAOQTProtectedGradientSet(protectedGradients, weights, len(grad)); err != nil {
			return false, err
		}
	}
	state := t.snapshotOptimizerState()
	scale := float32(1)
	attemptsThisStep := 0
	for attempt := 0; attempt < aoqtTransactionalAdamMaxAttemptsPerStep && attemptsThisStep < aoqtTransactionalMaxAttemptsPerStep; attempt++ {
		t.restoreOptimizerState(state)
		diagnostics.ProposalAttempts++
		diagnostics.AdamProposalAttempts++
		attemptsThisStep++
		if err := t.applyAdamScaled(grad, scale); err != nil {
			t.restoreOptimizerState(state)
			return false, err
		}
		candidate, err := evaluate()
		if err != nil {
			t.restoreOptimizerState(state)
			return false, err
		}
		decision := t.evaluateTransactionalProposal(state, baseline, candidate, weights)
		var evidence aoqtProposalEvidence
		if isAOQTV7DevOptimizerMode(diagnostics.OptimizerMode) {
			direction := make([]float32, len(t.angles))
			for i := range direction {
				direction[i] = t.angles[i] - state.angles[i]
			}
			evidence.magnitude = aoqtMaxAbsFloat32(direction)
			evidence.movedAngleIndices = aoqtMovedAngleIndices(direction)
			evidence.blockSize = len(evidence.movedAngleIndices)
		}
		diagnostics.RecordProposal(diagnostics.ProposalAttempts, aoqtProposalKindAdam, decision, baseline, candidate, evidence)
		if decision.accepted {
			diagnostics.AdamAcceptedProposals++
			diagnostics.AcceptedProposals++
			return true, nil
		}
		diagnostics.RejectedProposals++
		diagnostics.AdamRejectedProposals++
		diagnostics.Backtracks++
		scale *= 0.5
	}
	accepted, err := t.acceptTransactionalCoordinateStep(grad, q3GainGrad, state, baseline, evaluate, weights, diagnostics, attemptsThisStep, protectedGradientSets...)
	if err != nil || accepted {
		return accepted, err
	}
	t.restoreOptimizerState(state)
	return false, nil
}

func (t *AOQTSidecarTrainer) acceptTransactionalCoordinateStep(grad, q3GainGrad []float32, state aoqtOptimizerState, baseline aoqtStepEvaluation, evaluate func() (aoqtStepEvaluation, error), weights AOQTSidecarRowWeights, diagnostics *AOQTSidecarOptimizerDiagnostics, attemptsThisStep int, protectedGradientSets ...[]aoqtProtectedAngleGradient) (bool, error) {
	protectedGradients, protectedSetProvided, err := aoqtParseProtectedGradientSets(protectedGradientSets)
	if err != nil {
		t.restoreOptimizerState(state)
		return false, err
	}
	if protectedSetProvided {
		if err := validateAOQTProtectedGradientSet(protectedGradients, weights, len(grad)); err != nil {
			t.restoreOptimizerState(state)
			return false, err
		}
	}
	plan, err := newAOQTCoordinateSearchPlanForMode(q3GainGrad, grad, t.config.LearningRate, t.config.AngleCap, t.config.OptimizerMode, protectedGradients)
	if err != nil {
		t.restoreOptimizerState(state)
		return false, err
	}
	if len(plan.Order) == 0 || attemptsThisStep >= aoqtTransactionalMaxAttemptsPerStep {
		t.restoreOptimizerState(state)
		return false, nil
	}
	diagnostics.CoordinateSearchPlanCount++
	diagnostics.CoordinateTopAngles = len(plan.Order)
	diagnostics.CoordinateMagnitudeCount = len(plan.Magnitudes)
	diagnostics.CoordinateBlockCount = len(plan.BlockSizes)
	diagnostics.CoordinateSearchStrategy = plan.Strategy
	diagnostics.CoordinateSearchLearningRate = plan.LearningRate
	diagnostics.TrustRegionRadius = plan.TrustRegionRadius
	audit, err := plan.coordinateSearchAuditJSON()
	if err != nil {
		t.restoreOptimizerState(state)
		return false, err
	}
	diagnostics.CoordinateSearchAudit = string(audit)
	auditChain, err := appendAOQTCoordinateSearchAuditChain(diagnostics.CoordinateSearchAuditChain, string(audit))
	if err != nil {
		t.restoreOptimizerState(state)
		return false, err
	}
	hashChain, err := appendAOQTCoordinateSearchHashChain(diagnostics.CoordinateSearchHashChain, string(audit))
	if err != nil {
		t.restoreOptimizerState(state)
		return false, err
	}
	var linkedHashes []string
	if err := strictUnmarshalAOQT([]byte(hashChain), &linkedHashes); err != nil || len(linkedHashes) == 0 {
		t.restoreOptimizerState(state)
		if err != nil {
			return false, fmt.Errorf("AOQT coordinate search hash chain is invalid after append: %w", err)
		}
		return false, fmt.Errorf("AOQT coordinate search hash chain is empty after append")
	}
	diagnostics.CoordinateSearchOrderingHash = linkedHashes[len(linkedHashes)-1]
	diagnostics.CoordinateSearchAuditChain = auditChain
	diagnostics.CoordinateSearchHashChain = hashChain
	tryProposal := func(evidence aoqtProposalEvidence, apply func()) (bool, error) {
		if attemptsThisStep >= aoqtTransactionalMaxAttemptsPerStep {
			return false, nil
		}
		t.restoreOptimizerState(state)
		apply()
		if err := t.ProjectAngles(); err != nil {
			t.restoreOptimizerState(state)
			return false, err
		}
		direction := make([]float32, len(t.angles))
		for i := range direction {
			direction[i] = t.angles[i] - state.angles[i]
		}
		evidence.movedAngleIndices = aoqtMovedAngleIndices(direction)
		if len(plan.ProtectedComponents) > 0 {
			if !aoqtActualDirectionInProtectedCone(direction, q3GainGrad, protectedGradients) {
				t.restoreOptimizerState(state)
				return false, nil
			}
		}
		diagnostics.ProposalAttempts++
		diagnostics.CoordinateProposalAttempts++
		attemptsThisStep++
		candidate, err := evaluate()
		if err != nil {
			t.restoreOptimizerState(state)
			return false, err
		}
		decision := t.evaluateTransactionalProposal(state, baseline, candidate, weights)
		diagnostics.RecordProposal(diagnostics.ProposalAttempts, aoqtProposalKindCoordinate, decision, baseline, candidate, evidence)
		if decision.accepted {
			diagnostics.CoordinateAcceptedProposals++
			diagnostics.AcceptedProposals++
			return true, nil
		}
		diagnostics.RejectedProposals++
		diagnostics.CoordinateRejectedProposals++
		diagnostics.Backtracks++
		return false, nil
	}
	for _, magnitude := range plan.BlockMagnitudes {
		for _, blockSize := range plan.BlockSizes {
			if len(plan.ProtectedComponents) > 0 && !aoqtCoordinateBlockDirectionInCone(plan.Order[:blockSize], q3GainGrad, protectedGradients) {
				continue
			}
			indices := aoqtCoordinateRankIndices(plan.Order[:blockSize])
			accepted, err := tryProposal(aoqtProposalEvidence{magnitude: magnitude, blockSize: blockSize, movedAngleIndices: indices}, func() {
				for _, ranked := range plan.Order[:blockSize] {
					t.angles[ranked.index] += ranked.primaryDirection * magnitude
				}
			})
			if err != nil || accepted {
				return accepted, err
			}
		}
	}
	for _, ranked := range plan.Order {
		if len(plan.ProtectedComponents) > 0 && ranked.protectedConflictCount > 0 {
			continue
		}
		direction := ranked.primaryDirection
		for _, magnitude := range plan.Magnitudes {
			accepted, err := tryProposal(aoqtProposalEvidence{magnitude: magnitude, blockSize: 1, movedAngleIndices: []int{ranked.index}}, func() {
				t.angles[ranked.index] += direction * magnitude
			})
			if err != nil || accepted {
				return accepted, err
			}
			if attemptsThisStep >= aoqtTransactionalMaxAttemptsPerStep {
				t.restoreOptimizerState(state)
				return false, nil
			}
		}
	}
	if len(plan.ProtectedComponents) > 0 {
		t.restoreOptimizerState(state)
		return false, nil
	}
	for _, ranked := range plan.Order {
		direction := -aoqtCoordinateSearchPrimaryDirection(q3GainGrad[ranked.index])
		for _, magnitude := range plan.ReverseMagnitudes {
			accepted, err := tryProposal(aoqtProposalEvidence{magnitude: magnitude, blockSize: 1, movedAngleIndices: []int{ranked.index}}, func() {
				t.angles[ranked.index] += direction * magnitude
			})
			if err != nil || accepted {
				return accepted, err
			}
			if attemptsThisStep >= aoqtTransactionalMaxAttemptsPerStep {
				t.restoreOptimizerState(state)
				return false, nil
			}
		}
	}
	t.restoreOptimizerState(state)
	return false, nil
}

func (t *AOQTSidecarTrainer) acceptsTransactionalProposal(state aoqtOptimizerState, baseline, candidate aoqtStepEvaluation, weights AOQTSidecarRowWeights) bool {
	return t.evaluateTransactionalProposal(state, baseline, candidate, weights).accepted
}

func (t *AOQTSidecarTrainer) evaluateTransactionalProposal(state aoqtOptimizerState, baseline, candidate aoqtStepEvaluation, weights AOQTSidecarRowWeights) aoqtTransactionalProposalDecision {
	if !aoqtAnglesMoved(state.angles, t.angles) {
		return aoqtTransactionalProposalDecision{reason: aoqtRejectionNoAngleMovement}
	}
	decision := aoqtEvaluateTransactionalStepWithPolicy(baseline, candidate, weights, t.eligibilityPolicy)
	decision.angleMoved = true
	return decision
}

func aoqtAcceptsTransactionalStep(baseline, candidate aoqtStepEvaluation, weights AOQTSidecarRowWeights) bool {
	return aoqtEvaluateTransactionalStep(baseline, candidate, weights).accepted
}

type aoqtTransactionalRejectionReason string

const (
	aoqtRejectionNoAngleMovement       aoqtTransactionalRejectionReason = "no_angle_movement"
	aoqtRejectionNonFiniteLoss         aoqtTransactionalRejectionReason = "non_finite_loss"
	aoqtRejectionLossIncrease          aoqtTransactionalRejectionReason = "loss_increase"
	aoqtRejectionInvalidComponents     aoqtTransactionalRejectionReason = "invalid_components"
	aoqtRejectionLossComponentMismatch aoqtTransactionalRejectionReason = "loss_component_mismatch"
	aoqtRejectionInactiveObjective     aoqtTransactionalRejectionReason = "inactive_objective"
	aoqtRejectionNoQ3GainImprovement   aoqtTransactionalRejectionReason = "no_q3_gain_improvement"
	aoqtRejectionComponentRegression   aoqtTransactionalRejectionReason = "component_regression"
)

type aoqtTransactionalProposalDecision struct {
	accepted             bool
	reason               aoqtTransactionalRejectionReason
	componentRegressions []aoqtComponentRegression
	angleMoved           bool
	activationChecked    bool
	activationSufficient bool
}

func aoqtEvaluateTransactionalStep(baseline, candidate aoqtStepEvaluation, weights AOQTSidecarRowWeights) aoqtTransactionalProposalDecision {
	return aoqtEvaluateTransactionalStepWithPolicy(baseline, candidate, weights, AOQTSidecarDefaultCandidateEligibilityPolicy())
}

func aoqtEvaluateTransactionalStepWithPolicy(baseline, candidate aoqtStepEvaluation, weights AOQTSidecarRowWeights, policy AOQTSidecarCandidateEligibilityPolicy) aoqtTransactionalProposalDecision {
	if !isFinite32(candidate.loss) {
		return aoqtTransactionalProposalDecision{reason: aoqtRejectionNonFiniteLoss}
	}
	if candidate.loss-baseline.loss > aoqtTransactionalLossEpsilon {
		return aoqtTransactionalProposalDecision{reason: aoqtRejectionLossIncrease}
	}
	if err := validateAOQTObjectiveComponents(candidate.components, "AOQT transactional optimizer candidate components"); err != nil {
		return aoqtTransactionalProposalDecision{reason: aoqtRejectionInvalidComponents}
	}
	if err := validateAOQTLossMatchesComponents(candidate.loss, candidate.components, "AOQT transactional optimizer candidate"); err != nil {
		return aoqtTransactionalProposalDecision{reason: aoqtRejectionLossComponentMismatch}
	}
	if err := validateAOQTActiveObjectiveContributions(weights, candidate.activation); err != nil {
		return aoqtTransactionalProposalDecision{
			reason:               aoqtRejectionInactiveObjective,
			activationChecked:    true,
			activationSufficient: false,
		}
	}
	if candidate.components.Q3Gain >= baseline.components.Q3Gain-aoqtTransactionalQ3ImprovementMinMagnitude {
		return aoqtTransactionalProposalDecision{
			reason:               aoqtRejectionNoQ3GainImprovement,
			activationChecked:    true,
			activationSufficient: true,
		}
	}
	policy = normalizedAOQTSidecarCandidateEligibilityPolicy(policy)
	regressions := aoqtComponentRegressions(baseline.components, candidate.components, policy)
	if len(regressions) > 0 {
		return aoqtTransactionalProposalDecision{
			reason:               aoqtRejectionComponentRegression,
			componentRegressions: regressions,
			activationChecked:    true,
			activationSufficient: true,
		}
	}
	return aoqtTransactionalProposalDecision{
		accepted:             true,
		activationChecked:    true,
		activationSufficient: true,
	}
}

type aoqtCoordinateSearchRank struct {
	index                             int
	abs                               float32
	guardConflict                     bool
	aggregateAbs                      float32
	aggregateGradient                 float32
	primaryDirection                  float32
	protectedConflictCount            int
	protectedMaxDirectionalDerivative float32
	protectedDirectionalDerivatives   []float32
}

type aoqtCoordinateSearchPlan struct {
	Strategy                 string
	LearningRate             float32
	Order                    []aoqtCoordinateSearchRank
	FullOrder                []aoqtCoordinateSearchRank
	FullRankSourceCount      int
	FullRankSourceSHA256     string
	Magnitudes               []float32
	BlockMagnitudes          []float32
	ReverseMagnitudes        []float32
	MicroTailMagnitudes      []float32
	BlockSizes               []int
	ProtectedComponents      []string
	TrustRegionRadius        float32
	DevOnly                  bool
	ActualDirectionAuthority string
	GlobalSigns              []int
	EndpointCount            int
}

func newAOQTCoordinateSearchPlan(q3GainGrad, aggregateGrad []float32, learningRate float32, protectedGradientSets ...[]aoqtProtectedAngleGradient) (aoqtCoordinateSearchPlan, error) {
	return newAOQTCoordinateSearchPlanForMode(q3GainGrad, aggregateGrad, learningRate, AOQTSidecarDefaultAngleCap, "", protectedGradientSets...)
}

func newAOQTCoordinateSearchPlanForMode(q3GainGrad, aggregateGrad []float32, learningRate, angleCap float32, optimizerMode string, protectedGradientSets ...[]aoqtProtectedAngleGradient) (aoqtCoordinateSearchPlan, error) {
	switch optimizerMode {
	case "", AOQTSidecarOptimizerModeV7DevTrustRegion, AOQTSidecarOptimizerModeV7R2DevTrustRegion, AOQTSidecarOptimizerModeV7R3DevActualDirection, AOQTSidecarOptimizerModeV7R4DevActualCoordinate:
	default:
		return aoqtCoordinateSearchPlan{}, fmt.Errorf("AOQT coordinate search optimizer_mode %q is unsupported", optimizerMode)
	}
	protectedGradients, _, err := aoqtParseProtectedGradientSets(protectedGradientSets)
	if err != nil {
		return aoqtCoordinateSearchPlan{}, err
	}
	order, err := rankedAOQTCoordinateSearchAngles(q3GainGrad, aggregateGrad, protectedGradients)
	if err != nil {
		return aoqtCoordinateSearchPlan{}, err
	}
	top := aoqtTransactionalCoordinateTopAngles
	if len(order) < top {
		top = len(order)
	}
	fullOrder, err := rankedAOQTCoordinateSearchAnglesAll(q3GainGrad, aggregateGrad, protectedGradients)
	if err != nil {
		return aoqtCoordinateSearchPlan{}, err
	}
	protectedComponents := aoqtProtectedGradientNames(protectedGradients)
	if len(protectedComponents) > 0 {
		// Protected-v2 audit always carries the complete deterministic top-k
		// prefix, including zero-gradient ranks when necessary. This makes the
		// advertised rank source independently auditable and prevents a producer
		// from silently shortening the protected probe order to a convenient
		// subset.
		top = aoqtTransactionalCoordinateTopAngles
		if len(fullOrder) < top {
			top = len(fullOrder)
		}
		order = append([]aoqtCoordinateSearchRank(nil), fullOrder[:top]...)
	} else {
		order = append([]aoqtCoordinateSearchRank(nil), order[:top]...)
	}
	fullRankSourceSHA256, err := aoqtCoordinateSearchRankSourceSHA256(fullOrder, protectedComponents)
	if err != nil {
		return aoqtCoordinateSearchPlan{}, err
	}
	magnitudes := aoqtCoordinateSearchMagnitudes(learningRate)
	if len(magnitudes) == 0 {
		return aoqtCoordinateSearchPlan{}, fmt.Errorf("AOQT coordinate search requires at least one finite positive magnitude")
	}
	trustRegionRadius := float32(0)
	strategy := aoqtCoordinateSearchStrategyLegacyQ3GainPrimary
	blockMagnitudes := aoqtCoordinateSearchMagnitudePrefix(magnitudes, aoqtTransactionalCoordinateBlockMagnitudeCount)
	reverseMagnitudes := aoqtCoordinateSearchMagnitudePrefix(magnitudes, aoqtTransactionalCoordinateReverseMagnitudeCount)
	microTailMagnitudes := []float32(nil)
	if len(protectedComponents) > 0 {
		if len(magnitudes) != aoqtTransactionalCoordinateMagnitudeCount {
			return aoqtCoordinateSearchPlan{}, fmt.Errorf("AOQT protected coordinate search requires the full intended schedule of %d finite positive magnitudes, got %d", aoqtTransactionalCoordinateMagnitudeCount, len(magnitudes))
		}
		microTailMagnitudes = aoqtCoordinateSearchMicroTail(magnitudes, aoqtTransactionalCoordinateMicroTailMagnitudeCount)
		if len(microTailMagnitudes) != aoqtTransactionalCoordinateMicroTailMagnitudeCount {
			return aoqtCoordinateSearchPlan{}, fmt.Errorf("AOQT protected coordinate search requires exactly %d finite positive micro-tail magnitudes, got %d", aoqtTransactionalCoordinateMicroTailMagnitudeCount, len(microTailMagnitudes))
		}
		if optimizerMode == AOQTSidecarOptimizerModeV7R3DevActualDirection {
			strategy = aoqtCoordinateSearchStrategyV7R3ActualDirection
			trustRegionRadius = learningRate
			if learningRate != AOQTV7R3ActualDirectionProbeLearningRate {
				return aoqtCoordinateSearchPlan{}, fmt.Errorf("AOQT V7-r3 actual-direction search requires learning rate exactly 0.01")
			}
			magnitudes = aoqtV7R3ActualDirectionMagnitudes()
			blockMagnitudes = append([]float32(nil), magnitudes...)
			microTailMagnitudes = append([]float32(nil), magnitudes...)
			reverseMagnitudes = nil
		} else if isAOQTV7DevOptimizerMode(optimizerMode) {
			if angleCap <= 0 || !isFinite32(angleCap) || angleCap > AOQTSidecarHardMaxAngleCap {
				return aoqtCoordinateSearchPlan{}, fmt.Errorf("AOQT dev trust-region angle cap must be finite, positive, and <= hard max %.8g", AOQTSidecarHardMaxAngleCap)
			}
			strategy = aoqtCoordinateSearchStrategyV7DevTrustRegion
			trustRegionRadius = learningRate
			if trustRegionRadius > angleCap {
				trustRegionRadius = angleCap
			}
			if trustRegionRadius <= 0 || !isFinite32(trustRegionRadius) {
				return aoqtCoordinateSearchPlan{}, fmt.Errorf("AOQT dev trust-region radius must be finite and positive")
			}
			magnitudes = aoqtCoordinateSearchMagnitudes(trustRegionRadius)
			if len(magnitudes) != aoqtTransactionalCoordinateMagnitudeCount {
				return aoqtCoordinateSearchPlan{}, fmt.Errorf("AOQT dev trust-region requires the full intended schedule of %d finite positive magnitudes, got %d", aoqtTransactionalCoordinateMagnitudeCount, len(magnitudes))
			}
			microTailMagnitudes = aoqtCoordinateSearchMicroTail(magnitudes, aoqtTransactionalCoordinateMicroTailMagnitudeCount)
			blockMagnitudes = append([]float32(nil), magnitudes...)
		} else {
			strategy = aoqtCoordinateSearchStrategyProtectedConeMicroTail
			// The production protected strategy is deliberately a q3-descent-only
			// tail. Reverse probes are first-order q3 ascent and therefore cannot
			// be in the requested q3/protected cone.
			blockMagnitudes = append([]float32(nil), microTailMagnitudes...)
			magnitudes = append([]float32(nil), microTailMagnitudes...)
		}
		reverseMagnitudes = nil
	}
	return aoqtCoordinateSearchPlan{
		Strategy:             strategy,
		LearningRate:         learningRate,
		Order:                order,
		FullOrder:            fullOrder,
		FullRankSourceCount:  len(q3GainGrad),
		FullRankSourceSHA256: fullRankSourceSHA256,
		Magnitudes:           magnitudes,
		BlockMagnitudes:      blockMagnitudes,
		ReverseMagnitudes:    reverseMagnitudes,
		MicroTailMagnitudes:  microTailMagnitudes,
		BlockSizes:           aoqtCoordinateSearchBlockSizes(top),
		ProtectedComponents:  protectedComponents,
		TrustRegionRadius:    trustRegionRadius,
		DevOnly:              isAOQTV7DevOptimizerMode(optimizerMode),
		ActualDirectionAuthority: func() string {
			if optimizerMode == AOQTSidecarOptimizerModeV7R3DevActualDirection {
				return AOQTV7R3ActualDirectionProbeDirectionSource
			}
			return ""
		}(),
		GlobalSigns: func() []int {
			if optimizerMode == AOQTSidecarOptimizerModeV7R3DevActualDirection {
				return []int{1, -1}
			}
			return nil
		}(),
		EndpointCount: func() int {
			if optimizerMode == AOQTSidecarOptimizerModeV7R3DevActualDirection {
				return AOQTV7R3ActualDirectionProbeEndpointCount
			}
			return 0
		}(),
	}, nil
}

func rankedAOQTCoordinateSearchAngles(q3GainGrad, aggregateGrad []float32, protectedGradientSets ...[]aoqtProtectedAngleGradient) ([]aoqtCoordinateSearchRank, error) {
	protectedGradients, _, err := aoqtParseProtectedGradientSets(protectedGradientSets)
	if err != nil {
		return nil, err
	}
	return rankedAOQTCoordinateSearchAnglesInternal(q3GainGrad, aggregateGrad, protectedGradients, false)
}

func rankedAOQTCoordinateSearchAnglesAll(q3GainGrad, aggregateGrad []float32, protectedGradientSets ...[]aoqtProtectedAngleGradient) ([]aoqtCoordinateSearchRank, error) {
	protectedGradients, _, err := aoqtParseProtectedGradientSets(protectedGradientSets)
	if err != nil {
		return nil, err
	}
	return rankedAOQTCoordinateSearchAnglesInternal(q3GainGrad, aggregateGrad, protectedGradients, true)
}

func rankedAOQTCoordinateSearchAnglesInternal(q3GainGrad, aggregateGrad []float32, protectedGradients []aoqtProtectedAngleGradient, includeZero bool) ([]aoqtCoordinateSearchRank, error) {
	if len(q3GainGrad) != len(aggregateGrad) {
		return nil, fmt.Errorf("AOQT coordinate search aggregate gradient count = %d, want %d", len(aggregateGrad), len(q3GainGrad))
	}
	if err := validateAOQTProtectedGradientNames(protectedGradients); err != nil {
		return nil, err
	}
	for _, protected := range protectedGradients {
		if len(protected.Grad) != len(q3GainGrad) {
			return nil, fmt.Errorf("AOQT coordinate search protected component %q gradient count = %d, want %d", protected.Name, len(protected.Grad), len(q3GainGrad))
		}
		for i, value := range protected.Grad {
			if !isFinite32(value) {
				return nil, fmt.Errorf("AOQT coordinate search protected component %q gradient %d is not finite", protected.Name, i)
			}
		}
	}
	order := make([]aoqtCoordinateSearchRank, 0, len(q3GainGrad))
	for i, q3g := range q3GainGrad {
		if !isFinite32(q3g) {
			return nil, fmt.Errorf("AOQT coordinate search q3_gain gradient %d is not finite", i)
		}
		ag := aggregateGrad[i]
		if !isFinite32(ag) {
			return nil, fmt.Errorf("AOQT coordinate search aggregate gradient %d is not finite", i)
		}
		abs := float32(math.Abs(float64(q3g)))
		if abs == 0 && !includeZero {
			continue
		}
		primaryDirection := float32(0)
		if abs > 0 {
			primaryDirection = aoqtCoordinateSearchPrimaryDirection(q3g)
		}
		protectedDirectionalDerivatives := make([]float32, len(protectedGradients))
		protectedConflictCount := 0
		var protectedMaxDirectionalDerivative float32
		for pi, protected := range protectedGradients {
			derivative := protected.Grad[i] * primaryDirection
			protectedDirectionalDerivatives[pi] = derivative
			if derivative > 0 {
				protectedConflictCount++
				if derivative > protectedMaxDirectionalDerivative {
					protectedMaxDirectionalDerivative = derivative
				}
			}
		}
		order = append(order, aoqtCoordinateSearchRank{
			index:                             i,
			abs:                               abs,
			guardConflict:                     q3g*ag < 0 || protectedConflictCount > 0,
			aggregateAbs:                      float32(math.Abs(float64(ag))),
			aggregateGradient:                 ag,
			primaryDirection:                  primaryDirection,
			protectedConflictCount:            protectedConflictCount,
			protectedMaxDirectionalDerivative: protectedMaxDirectionalDerivative,
			protectedDirectionalDerivatives:   protectedDirectionalDerivatives,
		})
	}
	sort.SliceStable(order, func(i, j int) bool {
		return aoqtCoordinateSearchRankLess(order[i], order[j], len(protectedGradients) > 0)
	})
	return order, nil
}

func aoqtCoordinateSearchRankLess(a, b aoqtCoordinateSearchRank, protected bool) bool {
	if a.abs == 0 && b.abs != 0 {
		return false
	}
	if a.abs != 0 && b.abs == 0 {
		return true
	}
	if protected {
		if a.protectedConflictCount != b.protectedConflictCount {
			return a.protectedConflictCount < b.protectedConflictCount
		}
		if a.protectedMaxDirectionalDerivative != b.protectedMaxDirectionalDerivative {
			return a.protectedMaxDirectionalDerivative < b.protectedMaxDirectionalDerivative
		}
	}
	if a.abs == b.abs {
		if a.guardConflict != b.guardConflict {
			return !a.guardConflict
		}
		if a.aggregateAbs != b.aggregateAbs {
			return a.aggregateAbs > b.aggregateAbs
		}
		return a.index < b.index
	}
	return a.abs > b.abs
}

// aoqtCoordinateSearchRankSourceSHA256 hashes the complete deterministic rank
// source that is serialized in the audit. Keeping the source record
// self-contained lets validators recompute the digest from the payload rather
// than accepting an opaque producer-supplied hash.
func aoqtCoordinateSearchRankSourceSHA256(fullOrder []aoqtCoordinateSearchRank, protectedComponents []string) (string, error) {
	auditOrder := make([]aoqtCoordinateSearchAuditOrder, len(fullOrder))
	for i, item := range fullOrder {
		auditOrder[i] = aoqtCoordinateSearchAuditOrderFromRank(item)
	}
	source := struct {
		Version             string                           `json:"version"`
		Count               int                              `json:"count"`
		ProtectedComponents []string                         `json:"protected_components"`
		Ranks               []aoqtCoordinateSearchAuditOrder `json:"ranks"`
	}{
		Version:             "aoqt_protected_rank_source_v2",
		Count:               len(auditOrder),
		ProtectedComponents: append([]string(nil), protectedComponents...),
		Ranks:               auditOrder,
	}
	data, err := json.Marshal(source)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func aoqtParseProtectedGradientSets(gradientSets [][]aoqtProtectedAngleGradient) ([]aoqtProtectedAngleGradient, bool, error) {
	switch len(gradientSets) {
	case 0:
		// Omitted protected gradients preserve the historical q3-only caller
		// contract. Fit always supplies exactly one set, including an empty set
		// when no protected component has positive manifest weight.
		return nil, false, nil
	case 1:
		return gradientSets[0], true, nil
	default:
		return nil, false, fmt.Errorf("AOQT protected gradient sets = %d, want at most one explicit set", len(gradientSets))
	}
}

func aoqtProtectedGradientNames(gradients []aoqtProtectedAngleGradient) []string {
	if len(gradients) == 0 {
		return nil
	}
	names := make([]string, len(gradients))
	for i, gradient := range gradients {
		names[i] = gradient.Name
	}
	return names
}

func aoqtActiveProtectedComponentNames(weights AOQTSidecarRowWeights) []string {
	components := []struct {
		name   string
		weight float32
	}{
		{"q3_order_guard", weights.Q3OrderGuard},
		{"q3_score_distill", weights.Q3ScoreDistill},
		{"q5_order_guard", weights.Q5OrderGuard},
		{"q5_score_distill", weights.Q5ScoreDistill},
		{"nf_boundary_guard", weights.NFBoundaryGuard},
	}
	names := make([]string, 0, len(components))
	for _, component := range components {
		if component.weight > 0 {
			names = append(names, component.name)
		}
	}
	return names
}

func aoqtStringSlicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func validateAOQTProtectedGradientSet(gradients []aoqtProtectedAngleGradient, weights AOQTSidecarRowWeights, dimension int) error {
	if err := validateAOQTProtectedGradientNames(gradients); err != nil {
		return err
	}
	wantNames := aoqtActiveProtectedComponentNames(weights)
	gotNames := aoqtProtectedGradientNames(gradients)
	if !aoqtStringSlicesEqual(gotNames, wantNames) {
		return fmt.Errorf("AOQT protected gradient components = %v, want active components %v", gotNames, wantNames)
	}
	for _, gradient := range gradients {
		if len(gradient.Grad) != dimension {
			return fmt.Errorf("AOQT protected component %q gradient count = %d, want %d", gradient.Name, len(gradient.Grad), dimension)
		}
		for i, value := range gradient.Grad {
			if !isFinite32(value) {
				return fmt.Errorf("AOQT protected component %q gradient %d is not finite", gradient.Name, i)
			}
		}
	}
	return nil
}

func validateAOQTProtectedGradientNames(gradients []aoqtProtectedAngleGradient) error {
	seen := make(map[string]struct{}, len(gradients))
	for _, gradient := range gradients {
		if strings.TrimSpace(gradient.Name) == "" {
			return fmt.Errorf("AOQT protected gradient component name is required")
		}
		if _, ok := seen[gradient.Name]; ok {
			return fmt.Errorf("AOQT protected gradient component %q is duplicated", gradient.Name)
		}
		seen[gradient.Name] = struct{}{}
		switch gradient.Name {
		case "q3_order_guard", "q3_score_distill", "q5_order_guard", "q5_score_distill", "nf_boundary_guard":
		default:
			return fmt.Errorf("AOQT protected gradient component %q is not a recognized protected component", gradient.Name)
		}
	}
	return nil
}

func aoqtCoordinateSearchMicroTail(magnitudes []float32, count int) []float32 {
	if count <= 0 || len(magnitudes) == 0 {
		return nil
	}
	if count > len(magnitudes) {
		count = len(magnitudes)
	}
	start := len(magnitudes) - count
	return append([]float32(nil), magnitudes[start:]...)
}

// aoqtCoordinateBlockDirectionInCone checks the exact first-order invariant
// used by the proposal selector. For a loss gradient p and angle direction d,
// p·d <= 0 is the protected non-regression half-space. The q3-gain gradient
// must additionally satisfy g3·d < 0. The full objective is still evaluated
// after this proposal-only filter, so these checks never replace acceptance.
func aoqtCoordinateBlockDirectionInCone(order []aoqtCoordinateSearchRank, q3GainGrad []float32, protected []aoqtProtectedAngleGradient) bool {
	if len(order) == 0 {
		return false
	}
	for _, gradient := range protected {
		if len(gradient.Grad) != len(q3GainGrad) {
			return false
		}
		for _, value := range gradient.Grad {
			if !isFinite32(value) {
				return false
			}
		}
	}
	q3Derivative := float64(0)
	for _, ranked := range order {
		if ranked.index < 0 || ranked.index >= len(q3GainGrad) || !isFinite32(q3GainGrad[ranked.index]) || !isFinite32(ranked.primaryDirection) {
			return false
		}
		q3Derivative += float64(q3GainGrad[ranked.index]) * float64(ranked.primaryDirection)
	}
	if !(q3Derivative < 0) {
		return false
	}
	for _, gradient := range protected {
		derivative := float64(0)
		for _, ranked := range order {
			derivative += float64(gradient.Grad[ranked.index]) * float64(ranked.primaryDirection)
		}
		if math.IsNaN(derivative) || math.IsInf(derivative, 0) || derivative > 0 {
			return false
		}
	}
	return true
}

func aoqtDirectionInProtectedCone(direction []float32, protected []aoqtProtectedAngleGradient) bool {
	for _, value := range direction {
		if !isFinite32(value) {
			return false
		}
	}
	for _, gradient := range protected {
		if len(gradient.Grad) != len(direction) {
			return false
		}
		derivative := float64(0)
		for i, value := range direction {
			if !isFinite32(gradient.Grad[i]) {
				return false
			}
			derivative += float64(value) * float64(gradient.Grad[i])
		}
		if derivative > 0 {
			return false
		}
	}
	return true
}

// aoqtActualDirectionInProtectedCone rechecks the first-order proposal
// invariant after angle projection. Projection can clip a requested delta to
// zero or otherwise change it; only the applied delta may be used to admit a
// protected coordinate candidate. q3 gain requires strict descent, while each
// protected component permits only non-positive first-order change.
func aoqtActualDirectionInProtectedCone(direction, q3GainGrad []float32, protected []aoqtProtectedAngleGradient) bool {
	if len(direction) != len(q3GainGrad) || len(direction) == 0 {
		return false
	}
	q3Derivative := float64(0)
	for i, value := range direction {
		if !isFinite32(value) || !isFinite32(q3GainGrad[i]) {
			return false
		}
		q3Derivative += float64(value) * float64(q3GainGrad[i])
	}
	if !(q3Derivative < 0) || math.IsNaN(q3Derivative) || math.IsInf(q3Derivative, 0) {
		return false
	}
	for _, gradient := range protected {
		if len(gradient.Grad) != len(direction) {
			return false
		}
		derivative := float64(0)
		for i, value := range direction {
			if !isFinite32(gradient.Grad[i]) {
				return false
			}
			derivative += float64(value) * float64(gradient.Grad[i])
		}
		if math.IsNaN(derivative) || math.IsInf(derivative, 0) || derivative > 0 {
			return false
		}
	}
	return true
}

func aoqtCoordinateSearchDirections(gradient float32) [2]float32 {
	primary := aoqtCoordinateSearchPrimaryDirection(gradient)
	return [2]float32{primary, -primary}
}

func aoqtCoordinateSearchPrimaryDirection(gradient float32) float32 {
	if gradient < 0 {
		return 1
	}
	return -1
}

func aoqtCoordinateSearchMagnitudes(learningRate float32) []float32 {
	magnitudes := make([]float32, 0, aoqtTransactionalCoordinateMagnitudeCount)
	magnitude := learningRate
	for i := 0; i < aoqtTransactionalCoordinateMagnitudeCount; i++ {
		if magnitude > 0 && isFinite32(magnitude) {
			magnitudes = append(magnitudes, magnitude)
		}
		magnitude *= 0.5
	}
	return magnitudes
}

func aoqtCoordinateSearchMagnitudePrefix(magnitudes []float32, count int) []float32 {
	if count > len(magnitudes) {
		count = len(magnitudes)
	}
	if count <= 0 {
		return nil
	}
	return append([]float32(nil), magnitudes[:count]...)
}

func aoqtCoordinateSearchBlockSizes(top int) []int {
	blockSizes := make([]int, 0, len(aoqtTransactionalCoordinateBlockSizes))
	for _, size := range aoqtTransactionalCoordinateBlockSizes {
		if size > 1 && size <= top {
			blockSizes = append(blockSizes, size)
		}
	}
	return blockSizes
}

type aoqtCoordinateSearchAuditOrder struct {
	Index                             int       `json:"index"`
	Abs                               float32   `json:"abs"`
	GuardConflict                     bool      `json:"guard_conflict"`
	AggregateAbs                      float32   `json:"aggregate_abs"`
	AggregateGradient                 float32   `json:"aggregate_gradient"`
	PrimaryDirection                  float32   `json:"primary_direction"`
	ProtectedConflictCount            int       `json:"protected_conflict_count"`
	ProtectedMaxDirectionalDerivative float32   `json:"protected_max_directional_derivative"`
	ProtectedDirectionalDerivatives   []float32 `json:"protected_directional_derivatives"`
}

type aoqtCoordinateSearchAuditPayload struct {
	Strategy                 string                           `json:"strategy"`
	LearningRate             float32                          `json:"learning_rate"`
	TrustRegionRadius        float32                          `json:"trust_region_radius,omitempty"`
	DevOnly                  bool                             `json:"dev_only,omitempty"`
	ProtectedComponents      []string                         `json:"protected_components"`
	Order                    []aoqtCoordinateSearchAuditOrder `json:"order"`
	FullOrder                []aoqtCoordinateSearchAuditOrder `json:"full_order"`
	FullOrderSHA256          string                           `json:"full_order_sha256"`
	FullRankSourceCount      int                              `json:"full_rank_source_count"`
	FullRankSourceSHA256     string                           `json:"full_rank_source_sha256"`
	Magnitudes               []float32                        `json:"magnitudes"`
	BlockMagnitudes          []float32                        `json:"block_magnitudes"`
	ReverseMagnitudes        []float32                        `json:"reverse_magnitudes"`
	MicroTailMagnitudes      []float32                        `json:"micro_tail_magnitudes"`
	BlockSizes               []int                            `json:"block_sizes"`
	ActualDirectionAuthority string                           `json:"actual_direction_authority,omitempty"`
	GlobalSigns              []int                            `json:"global_signs,omitempty"`
	EndpointCount            int                              `json:"endpoint_count,omitempty"`
}

func aoqtCoordinateSearchAuditOrderFromRank(item aoqtCoordinateSearchRank) aoqtCoordinateSearchAuditOrder {
	return aoqtCoordinateSearchAuditOrder{
		Index:                             item.index,
		Abs:                               item.abs,
		GuardConflict:                     item.guardConflict,
		AggregateAbs:                      item.aggregateAbs,
		AggregateGradient:                 item.aggregateGradient,
		PrimaryDirection:                  item.primaryDirection,
		ProtectedConflictCount:            item.protectedConflictCount,
		ProtectedMaxDirectionalDerivative: item.protectedMaxDirectionalDerivative,
		ProtectedDirectionalDerivatives:   append([]float32(nil), item.protectedDirectionalDerivatives...),
	}
}

func (plan aoqtCoordinateSearchPlan) coordinateSearchAuditPayload() aoqtCoordinateSearchAuditPayload {
	payload := aoqtCoordinateSearchAuditPayload{
		Strategy:                 plan.Strategy,
		LearningRate:             plan.LearningRate,
		TrustRegionRadius:        plan.TrustRegionRadius,
		DevOnly:                  plan.DevOnly,
		ProtectedComponents:      append([]string(nil), plan.ProtectedComponents...),
		Order:                    make([]aoqtCoordinateSearchAuditOrder, len(plan.Order)),
		FullOrder:                make([]aoqtCoordinateSearchAuditOrder, len(plan.FullOrder)),
		FullRankSourceCount:      plan.FullRankSourceCount,
		FullRankSourceSHA256:     plan.FullRankSourceSHA256,
		Magnitudes:               append([]float32(nil), plan.Magnitudes...),
		BlockMagnitudes:          append([]float32(nil), plan.BlockMagnitudes...),
		ReverseMagnitudes:        append([]float32(nil), plan.ReverseMagnitudes...),
		MicroTailMagnitudes:      append([]float32(nil), plan.MicroTailMagnitudes...),
		BlockSizes:               append([]int(nil), plan.BlockSizes...),
		ActualDirectionAuthority: plan.ActualDirectionAuthority,
		GlobalSigns:              append([]int(nil), plan.GlobalSigns...),
		EndpointCount:            plan.EndpointCount,
	}
	for i, item := range plan.Order {
		payload.Order[i] = aoqtCoordinateSearchAuditOrderFromRank(item)
	}
	for i, item := range plan.FullOrder {
		payload.FullOrder[i] = aoqtCoordinateSearchAuditOrderFromRank(item)
	}
	fullOrderJSON, err := json.Marshal(payload.FullOrder)
	if err == nil {
		sum := sha256.Sum256(fullOrderJSON)
		payload.FullOrderSHA256 = hex.EncodeToString(sum[:])
	}
	return payload
}

func (plan aoqtCoordinateSearchPlan) coordinateSearchAuditJSON() ([]byte, error) {
	return json.Marshal(plan.coordinateSearchAuditPayload())
}

func appendAOQTCoordinateSearchAuditChain(existing, item string) (string, error) {
	chain := make([]string, 0, 1)
	if strings.TrimSpace(existing) != "" {
		if err := strictUnmarshalAOQT([]byte(existing), &chain); err != nil {
			return "", fmt.Errorf("AOQT coordinate search audit chain is invalid: %w", err)
		}
		if len(chain) == 0 {
			return "", fmt.Errorf("AOQT coordinate search audit chain must preserve a non-empty history")
		}
		canonical, err := json.Marshal(chain)
		if err != nil {
			return "", err
		}
		if string(canonical) != existing {
			return "", fmt.Errorf("AOQT coordinate search audit chain is not canonical JSON")
		}
	}
	chain = append(chain, item)
	data, err := json.Marshal(chain)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func appendAOQTCoordinateSearchHashChain(existing, audit string) (string, error) {
	chain := make([]string, 0, 1)
	if strings.TrimSpace(existing) != "" {
		if err := strictUnmarshalAOQT([]byte(existing), &chain); err != nil {
			return "", fmt.Errorf("AOQT coordinate search hash chain is invalid: %w", err)
		}
		if len(chain) == 0 {
			return "", fmt.Errorf("AOQT coordinate search hash chain must preserve a non-empty history")
		}
		canonical, err := json.Marshal(chain)
		if err != nil {
			return "", err
		}
		if string(canonical) != existing {
			return "", fmt.Errorf("AOQT coordinate search hash chain is not canonical JSON")
		}
	}
	previous := ""
	if len(chain) > 0 {
		previous = chain[len(chain)-1]
	}
	linked := sha256AOQTCoordinateSearchChainEntry(previous, audit)
	chain = append(chain, linked)
	data, err := json.Marshal(chain)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func sha256AOQTCoordinateSearchChainEntry(previous, audit string) string {
	hash := sha256.New()
	_, _ = hash.Write([]byte(previous))
	_, _ = hash.Write([]byte{0})
	_, _ = hash.Write([]byte(audit))
	return hex.EncodeToString(hash.Sum(nil))
}

func (plan aoqtCoordinateSearchPlan) SHA256() (string, error) {
	data, err := plan.coordinateSearchAuditJSON()
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func aoqtAnglesMoved(before, after []float32) bool {
	if len(before) != len(after) {
		return true
	}
	for i := range before {
		if before[i] != after[i] {
			return true
		}
	}
	return false
}

func aoqtMovedAngleIndices(direction []float32) []int {
	indices := make([]int, 0)
	for i, delta := range direction {
		if delta != 0 {
			indices = append(indices, i)
		}
	}
	return indices
}

func aoqtMaxAbsFloat32(values []float32) float32 {
	var maxAbs float32
	for _, value := range values {
		if abs := float32(math.Abs(float64(value))); abs > maxAbs {
			maxAbs = abs
		}
	}
	return maxAbs
}

func aoqtCoordinateRankIndices(order []aoqtCoordinateSearchRank) []int {
	indices := make([]int, len(order))
	for i, ranked := range order {
		indices[i] = ranked.index
	}
	return indices
}

func (t *AOQTSidecarTrainer) snapshotOptimizerState() aoqtOptimizerState {
	return aoqtOptimizerState{
		angles: append([]float32(nil), t.angles...),
		adamM:  append([]float32(nil), t.adamM...),
		adamV:  append([]float32(nil), t.adamV...),
		step:   t.step,
	}
}

func (t *AOQTSidecarTrainer) restoreOptimizerState(state aoqtOptimizerState) {
	copy(t.angles, state.angles)
	copy(t.adamM, state.adamM)
	copy(t.adamV, state.adamV)
	t.step = state.step
}

func validateAOQTFitObjectiveContract(contract AOQTSidecarObjectiveContract, objective AOQTSidecarVectorObjective) error {
	// A public interface or package-private marker is insufficient here: an
	// external wrapper can embed the exported prepared objective and inherit
	// every marker while overriding EvaluateAOQT. Require the exact concrete
	// prepared-IP implementation (value or pointer), so the evaluator and its
	// masked-gradient semantics cannot be substituted through embedding.
	var prepared AOQTSidecarPreparedIPObjective
	switch typed := objective.(type) {
	case AOQTSidecarPreparedIPObjective:
		prepared = typed
	case *AOQTSidecarPreparedIPObjective:
		if typed == nil {
			return fmt.Errorf("AOQT protected-v2 non-plan training requires a non-nil prepared-IP objective")
		}
		prepared = *typed
	default:
		return fmt.Errorf("AOQT protected-v2 non-plan training requires the concrete prepared-IP masked-gradient objective")
	}
	provider, ok := any(prepared).(AOQTSidecarPreparedIPObjectiveContractProvider)
	if !ok {
		return fmt.Errorf("AOQT non-plan training requires objective contract provider")
	}
	got := provider.AOQTPreparedIPObjectiveConfig().ObjectiveContract(contract.WeightSums)
	if !aoqtObjectiveContractsEqual(got, contract) {
		return fmt.Errorf("AOQT objective config does not match calibration manifest objective_contract")
	}
	return nil
}

func aoqtObjectiveContractsEqual(a, b AOQTSidecarObjectiveContract) bool {
	return a.Dim == b.Dim &&
		a.TurboQuantSeed == b.TurboQuantSeed &&
		a.GainBit == b.GainBit &&
		a.Q3GuardBit == b.Q3GuardBit &&
		a.Q5GuardBit == b.Q5GuardBit &&
		a.ScoreSurface == b.ScoreSurface &&
		a.GainCutoff == b.GainCutoff &&
		float32Near(a.GainTau, b.GainTau, 1e-8) &&
		float32Near(a.GainMargin, b.GainMargin, 1e-8) &&
		float32Near(a.GuardTau, b.GuardTau, 1e-8) &&
		float32Near(a.GuardMargin, b.GuardMargin, 1e-8) &&
		float32Near(a.ScoreDistillTau, b.ScoreDistillTau, 1e-8) &&
		a.NFBoundarySource == b.NFBoundarySource &&
		aoqtRowWeightsNearEqual(a.WeightSums, b.WeightSums)
}

func (t *AOQTSidecarTrainer) Transform() AOQTGivensTransform {
	transform := t.trainingTransformSnapshot()
	angles, err := transform.AnglesSHA256()
	if err == nil {
		transform.Audit.AnglesSHA256 = angles
	}
	orth, err := transform.OrthogonalityFrobeniusPerDim()
	if err == nil {
		transform.Audit.Orthogonality = &orth
	}
	return transform
}

func (t *AOQTSidecarTrainer) trainingTransformSnapshot() AOQTGivensTransform {
	transform := t.topology
	transform.Stages = append([]AOQTStage(nil), t.topology.Stages...)
	offset := 0
	for si := range transform.Stages {
		transform.Stages[si].Pairs = append([][2]int(nil), t.topology.Stages[si].Pairs...)
		transform.Stages[si].Angles = append([]float32(nil), t.angles[offset:offset+len(transform.Stages[si].Pairs)]...)
		offset += len(transform.Stages[si].Pairs)
	}
	return transform
}

func (t *AOQTSidecarTrainer) Angles() []float32 {
	return append([]float32(nil), t.angles...)
}

func (t *AOQTSidecarTrainer) ProjectAngles() error {
	if t == nil {
		return fmt.Errorf("AOQT trainer is nil")
	}
	return projectAOQTAngles(t.angles, t.config.AngleCap)
}

func (t *AOQTSidecarTrainer) SetAnglesForTest(angles []float32) error {
	if len(angles) != len(t.angles) {
		return fmt.Errorf("AOQT angle count = %d, want %d", len(angles), len(t.angles))
	}
	copy(t.angles, angles)
	return t.ProjectAngles()
}

func (t *AOQTSidecarTrainer) lossAndAngleGrad(rows []AOQTSidecarCalibrationRow, objective AOQTSidecarVectorObjective) (float32, []float32, AOQTSidecarObjectiveActivation, AOQTSidecarObjectiveComponents, error) {
	totalGrad := make([]float32, len(t.angles))
	totalLoss := float32(0)
	var totalActivation AOQTSidecarObjectiveActivation
	var totalComponents AOQTSidecarObjectiveComponents
	transform := t.trainingTransformSnapshot()
	runtime, err := newAOQTGivensTrainingRuntime(transform)
	if err != nil {
		return 0, nil, totalActivation, totalComponents, err
	}
	for _, row := range rows {
		query, candidates, err := transformAOQTRowWithRuntime(runtime, row)
		if err != nil {
			return 0, nil, totalActivation, totalComponents, err
		}
		result, err := objective.EvaluateAOQT(AOQTSidecarObjectiveInput{
			Row:        aoqtObjectiveRowView(row),
			Query:      append([]float32(nil), query...),
			Candidates: cloneAOQTVectors(candidates),
		})
		if err != nil {
			return 0, nil, totalActivation, totalComponents, err
		}
		if !isFinite32(result.Loss) {
			return 0, nil, totalActivation, totalComponents, fmt.Errorf("AOQT objective loss must be finite")
		}
		if err := validateAOQTObjectiveActivation(result.Activation, "AOQT objective activation"); err != nil {
			return 0, nil, totalActivation, totalComponents, err
		}
		if err := validateAOQTObjectiveComponents(result.Components, "AOQT objective components"); err != nil {
			return 0, nil, totalActivation, totalComponents, err
		}
		if err := validateAOQTLossMatchesComponents(result.Loss, result.Components, "AOQT objective"); err != nil {
			return 0, nil, totalActivation, totalComponents, err
		}
		if len(result.QueryGrad) != runtime.dim {
			return 0, nil, totalActivation, totalComponents, fmt.Errorf("AOQT objective query grad dim = %d, want %d", len(result.QueryGrad), runtime.dim)
		}
		if len(result.CandidateGrads) != len(row.CandidateVectors) {
			return 0, nil, totalActivation, totalComponents, fmt.Errorf("AOQT objective candidate grad count = %d, want %d", len(result.CandidateGrads), len(row.CandidateVectors))
		}
		totalLoss += result.Loss
		totalActivation.Add(result.Activation)
		totalComponents.Add(result.Components)
		if err := accumulateAOQTVectorAngleGradWithRuntime(runtime, row.QueryVector, result.QueryGrad, totalGrad); err != nil {
			return 0, nil, totalActivation, totalComponents, err
		}
		for i, grad := range result.CandidateGrads {
			if len(grad) != runtime.dim {
				return 0, nil, totalActivation, totalComponents, fmt.Errorf("AOQT objective candidate grad %d dim = %d, want %d", i, len(grad), runtime.dim)
			}
			if err := accumulateAOQTVectorAngleGradWithRuntime(runtime, row.CandidateVectors[i], grad, totalGrad); err != nil {
				return 0, nil, totalActivation, totalComponents, err
			}
		}
	}
	return totalLoss, totalGrad, totalActivation, totalComponents, nil
}

func (t *AOQTSidecarTrainer) q3GainOnlyAngleGrad(rows []AOQTSidecarCalibrationRow, objective AOQTSidecarVectorObjective) ([]float32, error) {
	grad, err := t.componentOnlyAngleGrad(rows, objective, "q3_gain")
	if err != nil {
		return nil, err
	}
	return grad, nil
}

func (t *AOQTSidecarTrainer) protectedComponentAngleGrads(rows []AOQTSidecarCalibrationRow, objective AOQTSidecarVectorObjective, weights AOQTSidecarRowWeights) ([]aoqtProtectedAngleGradient, error) {
	components := aoqtActiveProtectedComponentNames(weights)
	gradients := make([]aoqtProtectedAngleGradient, 0, len(components))
	for _, component := range components {
		grad, err := t.componentOnlyAngleGrad(rows, objective, component)
		if err != nil {
			return nil, fmt.Errorf("AOQT protected component %s angle gradient: %w", component, err)
		}
		gradients = append(gradients, aoqtProtectedAngleGradient{Name: component, Grad: grad})
	}
	return gradients, nil
}

func (t *AOQTSidecarTrainer) componentOnlyAngleGrad(rows []AOQTSidecarCalibrationRow, objective AOQTSidecarVectorObjective, component string) ([]float32, error) {
	componentRows := make([]AOQTSidecarCalibrationRow, len(rows))
	for i, row := range rows {
		row.Weights = aoqtOnlyComponentWeights(row.Weights, component)
		componentRows[i] = row
	}
	_, grad, _, _, err := t.lossAndAngleGrad(componentRows, objective)
	if err != nil {
		return nil, err
	}
	return grad, nil
}

func aoqtOnlyComponentWeights(weights AOQTSidecarRowWeights, component string) AOQTSidecarRowWeights {
	switch component {
	case "q3_gain":
		return AOQTSidecarRowWeights{Q3Gain: weights.Q3Gain}
	case "q3_order_guard":
		return AOQTSidecarRowWeights{Q3OrderGuard: weights.Q3OrderGuard}
	case "q3_score_distill":
		return AOQTSidecarRowWeights{Q3ScoreDistill: weights.Q3ScoreDistill}
	case "q5_order_guard":
		return AOQTSidecarRowWeights{Q5OrderGuard: weights.Q5OrderGuard}
	case "q5_score_distill":
		return AOQTSidecarRowWeights{Q5ScoreDistill: weights.Q5ScoreDistill}
	case "nf_boundary_guard":
		return AOQTSidecarRowWeights{NFBoundaryGuard: weights.NFBoundaryGuard}
	default:
		return AOQTSidecarRowWeights{}
	}
}

func validateAOQTObjectiveComponents(components AOQTSidecarObjectiveComponents, label string) error {
	for _, item := range []struct {
		name  string
		value float32
	}{
		{"q3_gain", components.Q3Gain},
		{"q3_order_guard", components.Q3OrderGuard},
		{"q3_score_distill", components.Q3ScoreDistill},
		{"q5_order_guard", components.Q5OrderGuard},
		{"q5_score_distill", components.Q5ScoreDistill},
		{"nf_boundary_guard", components.NFBoundaryGuard},
	} {
		if !isFinite32(item.value) || item.value < 0 {
			return fmt.Errorf("%s.%s must be finite and non-negative", label, item.name)
		}
	}
	return nil
}

func validateAOQTLossMatchesComponents(loss float32, components AOQTSidecarObjectiveComponents, label string) error {
	if !isFinite32(loss) || loss < 0 {
		return fmt.Errorf("%s loss must be finite and non-negative", label)
	}
	sum := components.Sum()
	tolerance := float32(1e-5)
	if abs := float32(math.Abs(float64(loss))); abs > 1 {
		tolerance *= abs
	}
	if abs := float32(math.Abs(float64(sum))); abs > 1 && abs*1e-5 > tolerance {
		tolerance = abs * 1e-5
	}
	if float32(math.Abs(float64(loss-sum))) > tolerance {
		return fmt.Errorf("%s loss %.9g must equal component sum %.9g within %.9g", label, loss, sum, tolerance)
	}
	return nil
}

func (t *AOQTSidecarTrainer) applyAdam(grad []float32) error {
	return t.applyAdamScaled(grad, 1)
}

func (t *AOQTSidecarTrainer) applyAdamScaled(grad []float32, scale float32) error {
	if len(grad) != len(t.angles) {
		return fmt.Errorf("AOQT gradient count = %d, want %d", len(grad), len(t.angles))
	}
	if scale <= 0 || !isFinite32(scale) {
		return fmt.Errorf("AOQT Adam proposal scale must be finite and positive")
	}
	t.step++
	b1, b2 := t.config.Beta1, t.config.Beta2
	lr := t.config.LearningRate
	eps := t.config.Epsilon
	for i, g := range grad {
		if !isFinite32(g) {
			return fmt.Errorf("AOQT gradient %d is not finite", i)
		}
		t.adamM[i] = b1*t.adamM[i] + (1-b1)*g
		t.adamV[i] = b2*t.adamV[i] + (1-b2)*g*g
		mhat := t.adamM[i] / (1 - float32(math.Pow(float64(b1), float64(t.step))))
		vhat := t.adamV[i] / (1 - float32(math.Pow(float64(b2), float64(t.step))))
		t.angles[i] -= scale * lr * mhat / (float32(math.Sqrt(float64(vhat))) + eps)
	}
	return t.ProjectAngles()
}

func transformAOQTRow(transform AOQTGivensTransform, row AOQTSidecarCalibrationRow) ([]float32, [][]float32, error) {
	runtime, err := newValidatedAOQTGivensRuntime(transform)
	if err != nil {
		return nil, nil, err
	}
	return transformAOQTRowWithRuntime(runtime, row)
}

func newAOQTGivensTrainingRuntime(transform AOQTGivensTransform) (aoqtGivensRuntime, error) {
	if err := validateAOQTGivensTrainingSnapshot(transform); err != nil {
		return aoqtGivensRuntime{}, err
	}
	return newAOQTGivensRuntimeAfterValidation(transform), nil
}

func transformAOQTRowWithRuntime(runtime aoqtGivensRuntime, row AOQTSidecarCalibrationRow) ([]float32, [][]float32, error) {
	query, err := runtime.applyVector(row.QueryVector)
	if err != nil {
		return nil, nil, err
	}
	candidates := make([][]float32, len(row.CandidateVectors))
	for i := range row.CandidateVectors {
		candidates[i], err = runtime.applyVector(row.CandidateVectors[i])
		if err != nil {
			return nil, nil, fmt.Errorf("candidate %d: %w", i, err)
		}
	}
	return query, candidates, nil
}

func accumulateAOQTVectorAngleGrad(transform AOQTGivensTransform, input, outputGrad []float32, angleGrad []float32) error {
	runtime, err := newAOQTGivensTrainingRuntime(transform)
	if err != nil {
		return err
	}
	return accumulateAOQTVectorAngleGradWithRuntime(runtime, input, outputGrad, angleGrad)
}

func accumulateAOQTVectorAngleGradWithRuntime(runtime aoqtGivensRuntime, input, outputGrad []float32, angleGrad []float32) error {
	if len(input) != runtime.dim || len(outputGrad) != runtime.dim {
		return fmt.Errorf("AOQT vector/grad dims must match transform dim")
	}
	if err := validateAOQTFiniteVector(outputGrad, "AOQT output grad"); err != nil {
		return err
	}
	if len(angleGrad) != runtime.angleCount {
		return fmt.Errorf("AOQT gradient count = %d, want %d", len(angleGrad), runtime.angleCount)
	}
	vec, err := runtime.applyFiniteVector(input, "AOQT input")
	if err != nil {
		return err
	}
	grad := append([]float32(nil), outputGrad...)
	angleIndex := runtime.angleCount
	for si := len(runtime.stages) - 1; si >= 0; si-- {
		stage := runtime.stages[si]
		for pi := len(stage.pairs) - 1; pi >= 0; pi-- {
			angleIndex--
			pair := stage.pairs[pi]
			a, b := pair[0], pair[1]
			c, s := stage.cos[pi], stage.sin[pi]
			pa, pb := vec[a], vec[b]
			x := c*pa + s*pb
			y := -s*pa + c*pb
			ga, gb := grad[a], grad[b]
			angleGrad[angleIndex] += ga*(-s*x-c*y) + gb*(c*x-s*y)
			grad[a] = c*ga + s*gb
			grad[b] = -s*ga + c*gb
			vec[a], vec[b] = x, y
		}
	}
	return nil
}

func DenseInvariantMaxAbsDelta(transform AOQTGivensTransform, rows []AOQTSidecarCalibrationRow) (float64, error) {
	runtime, err := newValidatedAOQTGivensRuntime(transform)
	if err != nil {
		return 0, err
	}
	var maxDelta float64
	for _, row := range rows {
		query, candidates, err := transformAOQTRowWithRuntime(runtime, row)
		if err != nil {
			return 0, err
		}
		for i, candidate := range candidates {
			before := dotAOQT(row.QueryVector, row.CandidateVectors[i])
			declaredDelta := math.Abs(float64(before - row.AnchorScores.Dense[i]))
			if declaredDelta > float64(AOQTSidecarDenseAnchorTolerance) {
				return 0, fmt.Errorf("AOQT row %q anchor_scores.dense[%d] does not match pre-transform dot", row.RowID, i)
			}
			after := dotAOQT(query, candidate)
			delta := math.Abs(float64(before - after))
			if delta > maxDelta {
				maxDelta = delta
			}
		}
	}
	return maxDelta, nil
}

func normalizedAOQTSidecarTrainConfig(cfg AOQTSidecarTrainConfig) AOQTSidecarTrainConfig {
	if cfg.Dim == 0 {
		cfg.Dim = AOQTSidecarDim
	}
	if cfg.Stages == 0 {
		cfg.Stages = AOQTSidecarStages
	}
	if cfg.AngleCap == 0 {
		cfg.AngleCap = AOQTSidecarDefaultAngleCap
	}
	if cfg.MaxAngleCap == 0 {
		cfg.MaxAngleCap = AOQTSidecarHardMaxAngleCap
	}
	if cfg.LearningRate == 0 {
		cfg.LearningRate = 1e-3
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
	return cfg
}

func validateAOQTSidecarTrainConfig(cfg AOQTSidecarTrainConfig) error {
	if cfg.Dim != AOQTSidecarDim {
		return fmt.Errorf("AOQT train dim = %d, want %d", cfg.Dim, AOQTSidecarDim)
	}
	if cfg.Stages != AOQTSidecarStages {
		return fmt.Errorf("AOQT train stages = %d, want %d", cfg.Stages, AOQTSidecarStages)
	}
	if cfg.PairingSeed == 0 {
		return fmt.Errorf("AOQT train pairing seed is required")
	}
	if cfg.WorkplanSeed == 0 {
		return fmt.Errorf("AOQT train workplan seed is required")
	}
	switch cfg.OptimizerMode {
	case "", AOQTSidecarOptimizerModeV7DevTrustRegion, AOQTSidecarOptimizerModeV7R2DevTrustRegion, AOQTSidecarOptimizerModeV7R3DevActualDirection, AOQTSidecarOptimizerModeV7R4DevActualCoordinate, AOQTSidecarOptimizerModeV7R5DevActualCoordinateTrain:
	default:
		return fmt.Errorf("AOQT optimizer_mode %q is unsupported", cfg.OptimizerMode)
	}
	r5AuthorizationRequested := cfg.ActualCoordinateTrainRequired || cfg.ActualCoordinateTrainR4Receipt != nil || cfg.CanonicalR4Receipt != nil || strings.TrimSpace(cfg.ActualCoordinateTrainR4ReceiptPath) != "" || strings.TrimSpace(cfg.CanonicalR4ReceiptPath) != "" || strings.TrimSpace(cfg.ActualCoordinateTrainR4ReceiptSHA256) != "" || strings.TrimSpace(cfg.CanonicalR4ReceiptSHA256) != ""
	if r5AuthorizationRequested && !isAOQTV7R5ActualCoordinateMode(cfg.OptimizerMode) {
		return fmt.Errorf("AOQT V7-r5 actual-coordinate training authorization requires optimizer_mode=%s", AOQTSidecarOptimizerModeV7R5DevActualCoordinateTrain)
	}
	if cfg.OptimizerMode == AOQTSidecarOptimizerModeV7R2DevTrustRegion && !cfg.ForwardConsistencyProbeRequired {
		return fmt.Errorf("AOQT V7-r2 optimizer mode requires the forward-consistency probe")
	}
	if cfg.OptimizerMode == AOQTSidecarOptimizerModeV7R3DevActualDirection && cfg.ForwardConsistencyProbeRequired {
		return fmt.Errorf("AOQT V7-r3 actual-direction mode cannot use the V7-r2 forward-consistency probe")
	}
	if cfg.ForwardConsistencyProbeOnly && !cfg.ForwardConsistencyProbeRequired {
		return fmt.Errorf("AOQT forward-consistency probe-only mode requires the forward-consistency probe")
	}
	if cfg.ForwardConsistencyProbeOnly && cfg.OptimizerMode != AOQTSidecarOptimizerModeV7R2DevTrustRegion {
		return fmt.Errorf("AOQT forward-consistency probe-only mode requires optimizer_mode=%s", AOQTSidecarOptimizerModeV7R2DevTrustRegion)
	}
	if cfg.ActualDirectionProbeOnly && !cfg.ActualDirectionProbeRequired {
		return fmt.Errorf("AOQT actual-direction probe-only mode requires the actual-direction probe")
	}
	if cfg.ActualDirectionProbeOnly && cfg.OptimizerMode != AOQTSidecarOptimizerModeV7R3DevActualDirection {
		return fmt.Errorf("AOQT actual-direction probe-only mode requires optimizer_mode=%s", AOQTSidecarOptimizerModeV7R3DevActualDirection)
	}
	if cfg.ActualDirectionProbeRequired && cfg.OptimizerMode != AOQTSidecarOptimizerModeV7R3DevActualDirection {
		return fmt.Errorf("AOQT actual-direction probe requires optimizer_mode=%s", AOQTSidecarOptimizerModeV7R3DevActualDirection)
	}
	if cfg.OptimizerMode == AOQTSidecarOptimizerModeV7R4DevActualCoordinate && !cfg.ActualCoordinateProbeOnly {
		return fmt.Errorf("AOQT V7-r4 actual-coordinate mode is probe-only until training is authorized")
	}
	if cfg.ActualCoordinateProbeOnly && !cfg.ActualCoordinateProbeRequired {
		return fmt.Errorf("AOQT actual-coordinate probe-only mode requires the actual-coordinate probe")
	}
	if cfg.ActualCoordinateProbeOnly && cfg.OptimizerMode != AOQTSidecarOptimizerModeV7R4DevActualCoordinate {
		return fmt.Errorf("AOQT actual-coordinate probe-only mode requires optimizer_mode=%s", AOQTSidecarOptimizerModeV7R4DevActualCoordinate)
	}
	if cfg.ActualCoordinateProbeRequired && cfg.OptimizerMode != AOQTSidecarOptimizerModeV7R4DevActualCoordinate {
		return fmt.Errorf("AOQT actual-coordinate probe requires optimizer_mode=%s", AOQTSidecarOptimizerModeV7R4DevActualCoordinate)
	}
	if cfg.OptimizerMode == AOQTSidecarOptimizerModeV7R4DevActualCoordinate && (cfg.ForwardConsistencyProbeRequired || cfg.ActualDirectionProbeRequired) {
		return fmt.Errorf("AOQT V7-r4 actual-coordinate mode cannot combine with an older probe")
	}
	if isAOQTV7R5ActualCoordinateMode(cfg.OptimizerMode) {
		if cfg.PlanOnly {
			return fmt.Errorf("AOQT V7-r5 actual-coordinate training is not available in plan-only mode")
		}
		if cfg.ActualCoordinateProbeRequired || cfg.ActualCoordinateProbeOnly || cfg.ForwardConsistencyProbeRequired || cfg.ActualDirectionProbeRequired {
			return fmt.Errorf("AOQT V7-r5 actual-coordinate training cannot combine with an older probe")
		}
		if cfg.MaxSteps != AOQTV7R5ActualCoordinateTrainMaxSteps {
			return fmt.Errorf("AOQT V7-r5 actual-coordinate training requires max_steps exactly %d", AOQTV7R5ActualCoordinateTrainMaxSteps)
		}
		if cfg.LearningRate != AOQTV7R4ActualCoordinateProbeLearningRate {
			return fmt.Errorf("AOQT V7-r5 actual-coordinate training requires learning_rate exactly 0.01")
		}
		if cfg.ActualCoordinateTrainR4Receipt == nil && cfg.CanonicalR4Receipt == nil {
			return fmt.Errorf("AOQT V7-r5 actual-coordinate training requires a canonical V7-r4 receipt")
		}
		if err := validateAOQTR5AliasConsistency("canonical r4 receipt path", cfg.ActualCoordinateTrainR4ReceiptPath, cfg.CanonicalR4ReceiptPath); err != nil {
			return err
		}
		if err := validateAOQTR5AliasConsistency("canonical r4 receipt sha256", cfg.ActualCoordinateTrainR4ReceiptSHA256, cfg.CanonicalR4ReceiptSHA256); err != nil {
			return err
		}
		if strings.TrimSpace(cfg.ActualCoordinateTrainR4ReceiptPath) == "" && strings.TrimSpace(cfg.CanonicalR4ReceiptPath) == "" {
			return fmt.Errorf("AOQT V7-r5 actual-coordinate training requires the canonical V7-r4 receipt path")
		}
		pinned := cfg.ActualCoordinateTrainR4ReceiptSHA256
		if pinned == "" {
			pinned = cfg.CanonicalR4ReceiptSHA256
		}
		if pinned != AOQTV7R5CanonicalR4ReceiptSHA256 {
			return fmt.Errorf("AOQT V7-r5 canonical r4 receipt sha256 must equal pinned authorization")
		}
		if strings.TrimSpace(cfg.ActualCoordinateProbeFoldID) == "" {
			return fmt.Errorf("AOQT V7-r5 actual-coordinate training requires a fold id")
		}
		if err := validateAOQTSHA256(cfg.ActualCoordinateProbeSplitManifestSHA256, "AOQT V7-r5 split manifest sha256"); err != nil {
			return err
		}
	}
	if cfg.ForwardConsistencyProbeRequired {
		if cfg.OptimizerMode != AOQTSidecarOptimizerModeV7R2DevTrustRegion {
			return fmt.Errorf("AOQT forward-consistency probe requires optimizer_mode=%s", AOQTSidecarOptimizerModeV7R2DevTrustRegion)
		}
		if cfg.PlanOnly {
			return fmt.Errorf("AOQT forward-consistency probe is not available in plan-only mode")
		}
		if cfg.LearningRate != float32(0.01) {
			return fmt.Errorf("AOQT V7-r2 forward-consistency probe requires learning_rate exactly 0.01")
		}
		if strings.TrimSpace(cfg.ForwardConsistencyProbeFoldID) == "" {
			return fmt.Errorf("AOQT forward-consistency probe requires a fold id")
		}
		if err := validateAOQTSHA256(cfg.ForwardConsistencyProbeSplitManifestSHA256, "AOQT forward-consistency probe split manifest sha256"); err != nil {
			return err
		}
	}
	if cfg.ActualDirectionProbeRequired {
		if cfg.PlanOnly {
			return fmt.Errorf("AOQT actual-direction probe is not available in plan-only mode")
		}
		if cfg.LearningRate != AOQTV7R3ActualDirectionProbeLearningRate {
			return fmt.Errorf("AOQT V7-r3 actual-direction probe requires learning_rate exactly 0.01")
		}
		if strings.TrimSpace(cfg.ActualDirectionProbeFoldID) == "" {
			return fmt.Errorf("AOQT actual-direction probe requires a fold id")
		}
		if err := validateAOQTSHA256(cfg.ActualDirectionProbeSplitManifestSHA256, "AOQT actual-direction probe split manifest sha256"); err != nil {
			return err
		}
	}
	if cfg.ActualCoordinateProbeRequired {
		if cfg.PlanOnly {
			return fmt.Errorf("AOQT actual-coordinate probe is not available in plan-only mode")
		}
		if cfg.LearningRate != AOQTV7R4ActualCoordinateProbeLearningRate {
			return fmt.Errorf("AOQT V7-r4 actual-coordinate probe requires learning_rate exactly 0.01")
		}
		if strings.TrimSpace(cfg.ActualCoordinateProbeFoldID) == "" {
			return fmt.Errorf("AOQT actual-coordinate probe requires a fold id")
		}
		if err := validateAOQTSHA256(cfg.ActualCoordinateProbeSplitManifestSHA256, "AOQT actual-coordinate probe split manifest sha256"); err != nil {
			return err
		}
	}
	if cfg.AngleCap <= 0 || !isFinite32(cfg.AngleCap) {
		return fmt.Errorf("AOQT angle cap must be finite and positive")
	}
	if cfg.MaxAngleCap <= 0 || !isFinite32(cfg.MaxAngleCap) {
		return fmt.Errorf("AOQT max angle cap must be finite and positive")
	}
	if cfg.AngleCap > AOQTSidecarDefaultAngleCap {
		return fmt.Errorf("AOQT angle cap %.8g exceeds default cap %.8g", cfg.AngleCap, AOQTSidecarDefaultAngleCap)
	}
	if cfg.AngleCap > cfg.MaxAngleCap || cfg.MaxAngleCap > AOQTSidecarHardMaxAngleCap {
		return fmt.Errorf("AOQT angle caps exceed hard max %.8g", AOQTSidecarHardMaxAngleCap)
	}
	if !cfg.PlanOnly && !cfg.ForwardConsistencyProbeOnly && !cfg.ActualDirectionProbeOnly && !cfg.ActualCoordinateProbeOnly && cfg.MaxSteps <= 0 {
		return fmt.Errorf("AOQT non-plan training requires max steps")
	}
	if (cfg.PlanOnly || cfg.ForwardConsistencyProbeOnly || cfg.ActualDirectionProbeOnly || cfg.ActualCoordinateProbeOnly) && cfg.MaxSteps < 0 {
		return fmt.Errorf("AOQT plan max steps must be non-negative")
	}
	if cfg.LearningRate <= 0 || !isFinite32(cfg.LearningRate) {
		return fmt.Errorf("AOQT learning rate must be finite and positive")
	}
	if cfg.Beta1 <= 0 || cfg.Beta1 >= 1 || cfg.Beta2 <= 0 || cfg.Beta2 >= 1 {
		return fmt.Errorf("AOQT Adam betas must be in (0,1)")
	}
	if cfg.Epsilon <= 0 || !isFinite32(cfg.Epsilon) {
		return fmt.Errorf("AOQT Adam epsilon must be finite and positive")
	}
	return nil
}

func deterministicAOQTRowOrder(rows []AOQTSidecarCalibrationRow, seed int64, step int) []AOQTSidecarCalibrationRow {
	ordered := append([]AOQTSidecarCalibrationRow(nil), rows...)
	rng := rand.New(rand.NewSource(seed + int64(step)*7919))
	rng.Shuffle(len(ordered), func(i, j int) {
		ordered[i], ordered[j] = ordered[j], ordered[i]
	})
	return ordered
}

func aoqtObjectiveRowView(row AOQTSidecarCalibrationRow) AOQTSidecarCalibrationRow {
	view := row
	view.QueryVector = nil
	view.CandidateVectors = nil
	view.SplitProof.ExclusionIdentities = append([]string(nil), row.SplitProof.ExclusionIdentities...)
	view.CandidateDocIDs = append([]string(nil), row.CandidateDocIDs...)
	view.CandidateVectorIDs = append([]string(nil), row.CandidateVectorIDs...)
	view.CandidateVectorSHA256 = append([]string(nil), row.CandidateVectorSHA256...)
	view.QrelGains = append([]float32(nil), row.QrelGains...)
	view.CandidateSources = append([]string(nil), row.CandidateSources...)
	view.EligiblePairMask = append([][]bool(nil), row.EligiblePairMask...)
	for i := range view.EligiblePairMask {
		view.EligiblePairMask[i] = append([]bool(nil), row.EligiblePairMask[i]...)
	}
	view.AnchorScores.Dense = append([]float32(nil), row.AnchorScores.Dense...)
	view.AnchorScores.Q3 = append([]float32(nil), row.AnchorScores.Q3...)
	view.AnchorScores.Q5 = append([]float32(nil), row.AnchorScores.Q5...)
	view.AnchorRanks.Dense = append([]int(nil), row.AnchorRanks.Dense...)
	view.AnchorRanks.Q3 = append([]int(nil), row.AnchorRanks.Q3...)
	view.AnchorRanks.Q5 = append([]int(nil), row.AnchorRanks.Q5...)
	view.Extra = cloneAOQTRawMessageMap(row.Extra)
	return view
}

func validateAOQTGivensTrainingSnapshot(transform AOQTGivensTransform) error {
	return transform.validateStructure()
}

type aoqtPreparedIPSurface struct {
	dim                  int
	scores               []float32
	queryRaw             []float32
	queryNormalized      []float32
	queryNorm            float32
	candidateRaw         [][]float32
	candidateNormalized  [][]float32
	candidateNorms       []float32
	candidateDequantized [][]float32
}

type aoqtPreparedIPObjectiveWorkspace struct {
	mu               sync.Mutex
	dim              int
	seed             int64
	surfaces         map[int]*aoqtPreparedIPSurfaceWorkspace
	evaluateOverride func(AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error)
}

type aoqtPreparedIPSurfaceWorkspace struct {
	quantizer *turboquant.IPQuantizer
	prepared  turboquant.PreparedQuery
	quantized turboquant.IPQuantized
}

func newAOQTPreparedIPObjectiveWorkspace(cfg AOQTSidecarPreparedIPObjectiveConfig) *aoqtPreparedIPObjectiveWorkspace {
	return &aoqtPreparedIPObjectiveWorkspace{
		dim:      cfg.Dim,
		seed:     cfg.TurboQuantSeed,
		surfaces: make(map[int]*aoqtPreparedIPSurfaceWorkspace, 3),
	}
}

func newAOQTPreparedIPSurface(query []float32, candidates [][]float32, dim, bitWidth int, seed int64) aoqtPreparedIPSurface {
	return newAOQTPreparedIPSurfaceWithWorkspace(nil, query, candidates, dim, bitWidth, seed)
}

func newAOQTPreparedIPSurfaceWithWorkspace(workspace *aoqtPreparedIPObjectiveWorkspace, query []float32, candidates [][]float32, dim, bitWidth int, seed int64) aoqtPreparedIPSurface {
	if workspace == nil {
		q := turboquant.NewIPWithSeed(dim, bitWidth, seed)
		prepared := q.PrepareQuery(query)
		return newAOQTPreparedIPSurfaceWithQuantizer(q, prepared, nil, query, candidates, dim)
	}
	workspace.mu.Lock()
	defer workspace.mu.Unlock()
	state := workspace.surfaceWorkspace(dim, bitWidth, seed)
	state.quantizer.PrepareQueryToTrusted(&state.prepared, query)
	return newAOQTPreparedIPSurfaceWithQuantizer(state.quantizer, state.prepared, &state.quantized, query, candidates, dim)
}

func (w *aoqtPreparedIPObjectiveWorkspace) surfaceWorkspace(dim, bitWidth int, seed int64) *aoqtPreparedIPSurfaceWorkspace {
	if w.dim != dim || w.seed != seed {
		w.dim = dim
		w.seed = seed
		w.surfaces = make(map[int]*aoqtPreparedIPSurfaceWorkspace, 3)
	}
	if state := w.surfaces[bitWidth]; state != nil {
		return state
	}
	q := turboquant.NewIPWithSeed(dim, bitWidth, seed)
	state := &aoqtPreparedIPSurfaceWorkspace{
		quantizer: q,
		prepared:  q.AllocPreparedQuery(),
		quantized: turboquant.AllocIPQuantized(dim, bitWidth),
	}
	w.surfaces[bitWidth] = state
	return state
}

func newAOQTPreparedIPSurfaceWithQuantizer(q *turboquant.IPQuantizer, prepared turboquant.PreparedQuery, quantized *turboquant.IPQuantized, query []float32, candidates [][]float32, dim int) aoqtPreparedIPSurface {
	surface := aoqtPreparedIPSurface{
		dim:                  dim,
		scores:               make([]float32, len(candidates)),
		queryRaw:             append([]float32(nil), query...),
		queryNormalized:      append([]float32(nil), query...),
		queryNorm:            vectorNorm(query),
		candidateRaw:         make([][]float32, len(candidates)),
		candidateNormalized:  make([][]float32, len(candidates)),
		candidateNorms:       make([]float32, len(candidates)),
		candidateDequantized: make([][]float32, len(candidates)),
	}
	for i, candidate := range candidates {
		surface.candidateRaw[i] = append([]float32(nil), candidate...)
		surface.candidateNormalized[i] = append([]float32(nil), candidate...)
		surface.candidateNorms[i] = vectorNorm(candidate)
		qx := turboquant.IPQuantized{}
		if quantized == nil {
			qx = q.Quantize(surface.candidateRaw[i])
		} else {
			q.QuantizeTo(quantized, surface.candidateRaw[i])
			qx = *quantized
		}
		surface.candidateDequantized[i] = q.Dequantize(qx)
		if quantized == nil {
			surface.scores[i] = q.InnerProductPrepared(qx, prepared)
		} else {
			surface.scores[i] = q.InnerProductPreparedTrusted(qx, prepared)
		}
	}
	return surface
}

func (s aoqtPreparedIPSurface) accumulateScoreGrads(scoreGrads []float32, scale float32, queryGrad []float32, candidateGrads [][]float32) {
	if scale == 0 || len(scoreGrads) != len(s.scores) {
		return
	}
	for i, grad := range scoreGrads {
		if grad == 0 {
			continue
		}
		accumulateNormalizedPrefixSTEGrad(s.queryRaw, s.queryNormalized, s.queryNorm, s.candidateDequantized[i], grad*scale, queryGrad)
		accumulateNormalizedPrefixSTEGrad(s.candidateRaw[i], s.candidateNormalized[i], s.candidateNorms[i], s.queryNormalized, grad*scale, candidateGrads[i])
	}
}

func normalizedAOQTVector(vec []float32) []float32 {
	out := make([]float32, len(vec))
	norm := vectorNorm(vec)
	if norm == 0 {
		return out
	}
	inv := 1 / norm
	for i, v := range vec {
		out[i] = v * inv
	}
	return out
}

func aoqtAnchorOrderGuardLossAndGrad(scores []float32, ranks []int, eligiblePairs [][]bool, tau, margin float32) (float32, []float32, int, int) {
	grads := make([]float32, len(scores))
	if len(scores) == 0 || len(ranks) != len(scores) {
		return 0, grads, 0, 0
	}
	var loss float32
	var pairs int
	var contributing int
	for high := range scores {
		for low := range scores {
			if high == low || ranks[high] >= ranks[low] {
				continue
			}
			if len(eligiblePairs) > 0 && !eligiblePairs[high][low] {
				continue
			}
			z := (scores[low] - scores[high] + margin) / tau
			pairScale := sigmoid32(z) / tau
			loss += softplus32(z)
			grads[high] -= pairScale
			grads[low] += pairScale
			pairs++
			if pairScale != 0 {
				contributing++
			}
		}
	}
	if pairs == 0 {
		return 0, grads, 0, 0
	}
	inv := 1 / float32(pairs)
	for i := range grads {
		grads[i] *= inv
	}
	return loss * inv, grads, pairs, contributing
}

func aoqtNFBoundaryGuardLossAndGrad(scores []float32, ranks []int, sources []string, boundarySource string, eligiblePairs [][]bool, tau, margin float32) (float32, []float32, int, int) {
	grads := make([]float32, len(scores))
	if len(scores) == 0 || len(ranks) != len(scores) || len(sources) != len(scores) {
		return 0, grads, 0, 0
	}
	var loss float32
	var pairs int
	var contributing int
	for boundary := range scores {
		if sources[boundary] != boundarySource {
			continue
		}
		for other := range scores {
			if boundary == other || ranks[boundary] >= ranks[other] {
				continue
			}
			if len(eligiblePairs) > 0 && !eligiblePairs[boundary][other] {
				continue
			}
			z := (scores[other] - scores[boundary] + margin) / tau
			pairScale := sigmoid32(z) / tau
			loss += softplus32(z)
			grads[boundary] -= pairScale
			grads[other] += pairScale
			pairs++
			if pairScale != 0 {
				contributing++
			}
		}
	}
	if pairs == 0 {
		return 0, grads, 0, 0
	}
	inv := 1 / float32(pairs)
	for i := range grads {
		grads[i] *= inv
	}
	return loss * inv, grads, pairs, contributing
}

func aoqtCenteredScoreDistillLossAndGrad(scores, anchors []float32, tau float32) (float32, []float32, int) {
	grads := make([]float32, len(scores))
	if len(scores) == 0 || len(scores) != len(anchors) {
		return 0, grads, 0
	}
	var scoreMean, anchorMean float32
	for i := range scores {
		scoreMean += scores[i]
		anchorMean += anchors[i]
	}
	invN := 1 / float32(len(scores))
	scoreMean *= invN
	anchorMean *= invN
	var loss float32
	for i := range scores {
		residual := ((scores[i] - scoreMean) - (anchors[i] - anchorMean)) / tau
		loss += 0.5 * residual * residual * invN
		grads[i] = residual * invN / tau
	}
	return loss, grads, len(scores)
}

func normalizedAOQTPreparedIPObjectiveConfig(cfg AOQTSidecarPreparedIPObjectiveConfig) AOQTSidecarPreparedIPObjectiveConfig {
	if cfg.Dim == 0 {
		cfg.Dim = AOQTSidecarDim
	}
	if cfg.GainBit == 0 {
		cfg.GainBit = AOQTSidecarDefaultGainBit
	}
	if cfg.Q3GuardBit == 0 {
		cfg.Q3GuardBit = AOQTSidecarDefaultGuardBit3
	}
	if cfg.Q5GuardBit == 0 {
		cfg.Q5GuardBit = AOQTSidecarDefaultGuardBit5
	}
	if cfg.GainCutoff == 0 {
		cfg.GainCutoff = 10
	}
	if cfg.GainTau == 0 {
		cfg.GainTau = 0.05
	}
	if cfg.GuardTau == 0 {
		cfg.GuardTau = 0.05
	}
	if cfg.ScoreDistillTau == 0 {
		cfg.ScoreDistillTau = 1
	}
	if cfg.NFBoundarySource == "" {
		cfg.NFBoundarySource = "nf_boundary80_120"
	}
	return cfg
}

func validateAOQTPreparedIPObjectiveConfig(cfg AOQTSidecarPreparedIPObjectiveConfig) error {
	if cfg.Dim != AOQTSidecarDim {
		return fmt.Errorf("AOQT prepared-IP dim = %d, want %d", cfg.Dim, AOQTSidecarDim)
	}
	if cfg.TurboQuantSeed == 0 {
		return fmt.Errorf("AOQT prepared-IP turboquant seed is required")
	}
	for _, item := range []struct {
		name string
		bit  int
	}{
		{"gain_bit", cfg.GainBit},
		{"q3_guard_bit", cfg.Q3GuardBit},
		{"q5_guard_bit", cfg.Q5GuardBit},
	} {
		if item.bit < 2 || item.bit > 8 {
			return fmt.Errorf("AOQT prepared-IP %s = %d, want 2..8", item.name, item.bit)
		}
	}
	if cfg.GainCutoff <= 0 {
		return fmt.Errorf("AOQT prepared-IP gain cutoff must be positive")
	}
	for _, item := range []struct {
		name  string
		value float32
	}{
		{"gain_tau", cfg.GainTau},
		{"guard_tau", cfg.GuardTau},
		{"score_distill_tau", cfg.ScoreDistillTau},
		{"gain_margin", cfg.GainMargin},
		{"guard_margin", cfg.GuardMargin},
	} {
		if !isFinite32(item.value) {
			return fmt.Errorf("AOQT prepared-IP %s must be finite", item.name)
		}
	}
	if cfg.GainTau <= 0 || cfg.GuardTau <= 0 || cfg.ScoreDistillTau <= 0 {
		return fmt.Errorf("AOQT prepared-IP taus must be positive")
	}
	return nil
}

func cloneAOQTVectors(in [][]float32) [][]float32 {
	out := make([][]float32, len(in))
	for i := range in {
		out[i] = append([]float32(nil), in[i]...)
	}
	return out
}

func projectAOQTAngles(angles []float32, cap float32) error {
	if cap <= 0 || !isFinite32(cap) {
		return fmt.Errorf("AOQT angle cap must be finite and positive")
	}
	for i, angle := range angles {
		if !isFinite32(angle) {
			return fmt.Errorf("AOQT angle %d is not finite", i)
		}
		if angle > cap {
			angles[i] = cap
		} else if angle < -cap {
			angles[i] = -cap
		}
	}
	return nil
}

func countAOQTAngles(transform AOQTGivensTransform) int {
	var count int
	for _, stage := range transform.Stages {
		count += len(stage.Pairs)
	}
	return count
}

func countEligibleAOQTPairs(row AOQTSidecarCalibrationRow) int {
	if len(row.EligiblePairMask) == 0 {
		n := len(row.CandidateDocIDs)
		return n * (n - 1)
	}
	var count int
	for i := range row.EligiblePairMask {
		for j := range row.EligiblePairMask[i] {
			if row.EligiblePairMask[i][j] {
				count++
			}
		}
	}
	return count
}

func aoqtAngleStats(angles []float32) (float32, float32) {
	var sq float64
	var maxAbs float32
	for _, angle := range angles {
		abs := float32(math.Abs(float64(angle)))
		if abs > maxAbs {
			maxAbs = abs
		}
		sq += float64(angle * angle)
	}
	return float32(math.Sqrt(sq)), maxAbs
}

func dotAOQT(a, b []float32) float32 {
	var out float32
	for i := range a {
		out += a[i] * b[i]
	}
	return out
}
