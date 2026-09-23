package eosruntime

// V8 is a research-only constrained-basis screen. It preserves the D384
// AOQT serving surface and explores a bounded sequential sparse support
// derived from a q3-only gradient after protected-gradient projection. It has
// no package, metrics promotion, heldout, or release path.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"sort"
	"strings"
)

const (
	AOQTSidecarV8ConstrainedBasisScreenSchema        = "eos.aoqt.v8_constrained_basis_screen.v1"
	AOQTSidecarOptimizerModeV8ConstrainedBasisScreen = "aoqt-v8-constrained-basis-v1"

	AOQTV8ConstrainedBasisSeed                      = "aoqt-v8-constrained-basis-seed-v1"
	AOQTV8ConstrainedBasisVersion                   = "aoqt-v8-sequential-projected-sparse-q3-support-v1"
	AOQTV8ConstrainedBasisScheduleVersion           = "aoqt-v8-ranks-4-8-16-x-four-magnitudes-x-two-signs-v1"
	AOQTV8ConstrainedBasisRankingVersion            = "lexicographic_full_fold_feasibility_then_total_delta_then_q3_gain_v1"
	AOQTV8ConstrainedBasisPolicyVersion             = AOQTV7R6ProgressiveScreenPolicyVersion
	AOQTV8ConstrainedBasisFullWeighting             = AOQTV7R6ProgressiveScreenFullWeightingVersion
	AOQTV8ConstrainedBasisDenseInvariantDeferred    = "deferred_to_dense_train_verification_v1"
	AOQTV8ConstrainedBasisDenseInvariantProof       = "proved_by_endpoint_dense_max_abs_delta_v1"
	AOQTV8ConstrainedBasisDenseInvariantProofSchema = "eos.aoqt.v8_dense_invariant_endpoint_proof.v1"
	AOQTV8ConstrainedBasisDenseInvariantRejection   = "dense_invariant_max_abs_delta"

	AOQTV8CanonicalV7R6ReceiptFileSHA256 = "41857a7d14ca87c589f3a51aab097a2fd5f6911cb8eb30ed1728ece1b792f63d"

	AOQTV8ConstrainedBasisCandidateCap      = 24
	AOQTV8ConstrainedBasisRankCount         = 3
	AOQTV8ConstrainedBasisMagnitudeCount    = 4
	AOQTV8ConstrainedBasisGlobalSignCount   = 2
	AOQTV8ConstrainedBasisObjectiveEvalMax  = 25 // one full-fold baseline plus 24 candidates.
	AOQTV8ConstrainedBasisExpectedDim       = 384
	AOQTV8ConstrainedBasisExpectedAngleSize = AOQTSidecarAngleCount
)

func AOQTV8ConstrainedBasisRanks() []int { return []int{4, 8, 16} }

func AOQTV8ConstrainedBasisMagnitudes() []float32 {
	return []float32{0.005, 0.0025, 0.00125, 0.000625}
}

func AOQTV8ConstrainedBasisGlobalSigns() []int { return []int{1, -1} }

type AOQTSidecarV8ConstrainedBasisScreenConfig struct {
	Set *AOQTSidecarCalibrationSet

	ManifestPath                        string
	RowsJSONLPath                       string
	PreflightJSONPath                   string
	SplitManifestPath                   string
	FoldID                              string
	OutputReceiptJSONPath               string
	ExpectedManifestSHA256              string
	ExpectedRowsSHA256                  string
	ExpectedPreflightSHA256             string
	ExpectedSplitManifestSHA256         string
	ExpectedAnchorArtifactSHA256        string
	ExpectedAnchorPackageManifestSHA256 string
	ExpectedAnchorEmbeddingSpaceID      string
	ExpectedCompatibilityDigest         string
	ExpectedSourceArtifactHashes        []string
	ExpectedVectorCacheHashes           []string

	CanonicalR4ReceiptPath   string
	CanonicalR4ReceiptSHA256 string
	CanonicalR5ReceiptPath   string
	CanonicalR5ReceiptSHA256 string

	CanonicalV7R6ReceiptPath       string
	CanonicalV7R6ReceiptFileSHA256 string

	Mode      string
	ProbeOnly bool
}

type AOQTSidecarV8ConstrainedBasisScreenBinding struct {
	Inputs                         AOQTSidecarRunMetricInputs     `json:"inputs"`
	IOReport                       AOQTSidecarCalibrationIOReport `json:"io_report"`
	PreflightPath                  string                         `json:"preflight_path"`
	PreflightSHA256                string                         `json:"preflight_sha256"`
	SplitBinding                   *AOQTSidecarDevSplitBinding    `json:"split_binding"`
	FoldID                         string                         `json:"fold_id"`
	CanonicalR4ReceiptPath         string                         `json:"canonical_r4_receipt_path"`
	CanonicalR4ReceiptSHA256       string                         `json:"canonical_r4_receipt_sha256"`
	CanonicalR4ReceiptFileSHA256   string                         `json:"canonical_r4_receipt_file_sha256"`
	CanonicalR5ReceiptPath         string                         `json:"canonical_r5_receipt_path"`
	CanonicalR5ReceiptSHA256       string                         `json:"canonical_r5_receipt_sha256"`
	CanonicalR5ReceiptFileSHA256   string                         `json:"canonical_r5_receipt_file_sha256"`
	CanonicalV7R6ReceiptPath       string                         `json:"canonical_v7_r6_receipt_path"`
	CanonicalV7R6ReceiptFileSHA256 string                         `json:"canonical_v7_r6_receipt_file_sha256"`
}

type AOQTSidecarV8BasisVector struct {
	Ordinal              int       `json:"ordinal"`
	Kind                 string    `json:"kind"`
	Rank                 int       `json:"rank"`
	Direction            []float32 `json:"direction"`
	DirectionSHA256      string    `json:"direction_sha256"`
	SelectedIndices      []int     `json:"selected_indices"`
	SelectedSigns        []int     `json:"selected_signs"`
	SourceGradientSHA256 string    `json:"source_gradient_sha256"`
}

type AOQTSidecarV8ConstrainedBasisEndpoint struct {
	Ordinal                   int                            `json:"ordinal"`
	BasisOrdinal              int                            `json:"basis_ordinal"`
	Rank                      int                            `json:"rank"`
	Magnitude                 float32                        `json:"magnitude"`
	GlobalSign                int                            `json:"global_sign"`
	RequestedDirectionSHA256  string                         `json:"requested_direction_sha256"`
	ActualDirectionSHA256     string                         `json:"actual_direction_sha256"`
	ActualDirectionMaxAbs     float32                        `json:"actual_direction_max_abs"`
	ActualDirection           []float32                      `json:"actual_direction"`
	AngleMoved                bool                           `json:"angle_moved"`
	BaselineLoss              float32                        `json:"baseline_loss"`
	BaselineLossFinite        bool                           `json:"baseline_loss_finite"`
	CandidateLoss             float32                        `json:"candidate_loss"`
	CandidateLossFinite       bool                           `json:"candidate_loss_finite"`
	BaselineComponents        AOQTSidecarObjectiveComponents `json:"baseline_components"`
	BaselineComponentsFinite  bool                           `json:"baseline_components_finite"`
	CandidateComponents       AOQTSidecarObjectiveComponents `json:"candidate_components"`
	CandidateComponentsFinite bool                           `json:"candidate_components_finite"`
	ComponentDeltas           AOQTSidecarObjectiveComponents `json:"component_deltas"`
	ComponentDeltasFinite     bool                           `json:"component_deltas_finite"`
	CandidateActivation       AOQTSidecarObjectiveActivation `json:"candidate_activation"`
	TotalDelta                float32                        `json:"total_delta"`
	TotalDeltaFinite          bool                           `json:"total_delta_finite"`
	Q3GainEligible            bool                           `json:"q3_gain_eligible"`
	ProtectedEligible         bool                           `json:"protected_eligible"`
	TransactionEligible       bool                           `json:"transaction_eligible"`
	Reason                    string                         `json:"reason"`
	DenseInvariantProved      bool                           `json:"dense_invariant_proved"`
	DenseInvariantEligible    bool                           `json:"dense_invariant_eligible"`
	DenseInvariantMaxAbsDelta float64                        `json:"dense_invariant_max_abs_delta"`
	DenseInvariantProofSHA256 string                         `json:"dense_invariant_proof_sha256,omitempty"`
	CandidateError            string                         `json:"candidate_error,omitempty"`
	RankOrder                 int                            `json:"rank_order"`
}

type AOQTSidecarV8DenseInvariantEndpointProof struct {
	Schema                    string                        `json:"schema"`
	Version                   string                        `json:"version"`
	EndpointOrdinal           int                           `json:"endpoint_ordinal"`
	EndpointEvidence          AOQTSidecarV8EndpointEvidence `json:"endpoint_evidence"`
	ManifestSHA256            string                        `json:"manifest_sha256"`
	RowsSHA256                string                        `json:"rows_sha256"`
	RowIDSHA256               string                        `json:"row_id_sha256"`
	RowCount                  int                           `json:"row_count"`
	FoldID                    string                        `json:"fold_id"`
	FullFoldWeightingVersion  string                        `json:"full_fold_weighting_version"`
	HTFactorsApplied          bool                          `json:"ht_factors_applied"`
	CandidatePairingsSHA256   string                        `json:"candidate_pairings_sha256"`
	CandidateAnglesSHA256     string                        `json:"candidate_angles_sha256"`
	DenseMaxAbsDeltaTolerance float64                       `json:"dense_max_abs_delta_tolerance"`
	DenseInvariantMaxAbsDelta float64                       `json:"dense_invariant_max_abs_delta"`
	DenseInvariantEligible    bool                          `json:"dense_invariant_eligible"`
}

type AOQTSidecarV8EndpointEvidence struct {
	Ordinal                  int     `json:"ordinal"`
	BasisOrdinal             int     `json:"basis_ordinal"`
	Rank                     int     `json:"rank"`
	Magnitude                float32 `json:"magnitude"`
	GlobalSign               int     `json:"global_sign"`
	RequestedDirectionSHA256 string  `json:"requested_direction_sha256"`
	ActualDirectionSHA256    string  `json:"actual_direction_sha256"`
	ActualDirectionMaxAbs    float32 `json:"actual_direction_max_abs"`
	AngleMoved               bool    `json:"angle_moved"`
	BaselineLoss             float32 `json:"baseline_loss"`
	CandidateLoss            float32 `json:"candidate_loss"`
}

type AOQTSidecarV8ConstrainedBasisScreenReceipt struct {
	Schema                            string                                     `json:"schema"`
	Mode                              string                                     `json:"mode"`
	Required                          bool                                       `json:"required"`
	ProbeOnly                         bool                                       `json:"probe_only"`
	Passed                            bool                                       `json:"passed"`
	FailureReason                     string                                     `json:"failure_reason,omitempty"`
	BasisVersion                      string                                     `json:"basis_version"`
	ScheduleVersion                   string                                     `json:"schedule_version"`
	RankingVersion                    string                                     `json:"ranking_version"`
	PolicyVersion                     string                                     `json:"policy_version"`
	SelectionSeedSHA256               string                                     `json:"selection_seed_sha256"`
	CoordinateCount                   int                                        `json:"coordinate_count"`
	Dim                               int                                        `json:"dim"`
	Ranks                             []int                                      `json:"ranks"`
	Magnitudes                        []float32                                  `json:"magnitudes"`
	GlobalSigns                       []int                                      `json:"global_signs"`
	CandidateCap                      int                                        `json:"candidate_cap"`
	ObjectiveEvaluationCount          int                                        `json:"objective_evaluation_count"`
	FullFoldRowCount                  int                                        `json:"full_fold_row_count"`
	FullFoldWeightingVersion          string                                     `json:"full_fold_weighting_version"`
	HTFactorsApplied                  bool                                       `json:"ht_factors_applied"`
	BaselineLoss                      float32                                    `json:"baseline_loss"`
	BaselineLossFinite                bool                                       `json:"baseline_loss_finite"`
	BaselineComponents                AOQTSidecarObjectiveComponents             `json:"baseline_components"`
	BaselineActivation                AOQTSidecarObjectiveActivation             `json:"baseline_activation"`
	Q3GradientSHA256                  string                                     `json:"q3_gradient_sha256"`
	ProtectedGradientNames            []string                                   `json:"protected_gradient_names"`
	ProtectedGradientMatrixSHA256     string                                     `json:"protected_gradient_matrix_sha256"`
	BasisSHA256                       string                                     `json:"basis_sha256"`
	DirectionScheduleSHA256           string                                     `json:"direction_schedule_sha256"`
	CanonicalObjectiveContractSHA256  string                                     `json:"canonical_objective_contract_sha256"`
	CanonicalEligibilityPolicySHA256  string                                     `json:"canonical_eligibility_policy_sha256"`
	CanonicalV7R6ReceiptFileSHA256    string                                     `json:"canonical_v7_r6_receipt_file_sha256"`
	CanonicalV7R6TerminalReceiptBound bool                                       `json:"canonical_v7_r6_terminal_receipt_bound"`
	DenseInvariantStatus              string                                     `json:"dense_invariant_status"`
	DenseInvariantProofSHA256         string                                     `json:"dense_invariant_proof_sha256,omitempty"`
	DenseMaxAbsDeltaTolerance         float64                                    `json:"dense_max_abs_delta_tolerance"`
	DenseInvariantDeferred            bool                                       `json:"dense_invariant_deferred"`
	GradientEvidenceDeferred          bool                                       `json:"gradient_evidence_deferred"`
	FullVerificationCompleted         bool                                       `json:"full_verification_completed"`
	FeasibleEndpointCount             int                                        `json:"feasible_endpoint_count"`
	SelectedEndpointOrdinals          []int                                      `json:"selected_endpoint_ordinals"`
	EndpointHashChain                 string                                     `json:"endpoint_hash_chain"`
	EndpointHashChainTailSHA256       string                                     `json:"endpoint_hash_chain_tail_sha256"`
	Basis                             []AOQTSidecarV8BasisVector                 `json:"basis"`
	Endpoints                         []AOQTSidecarV8ConstrainedBasisEndpoint    `json:"endpoints"`
	Binding                           AOQTSidecarV8ConstrainedBasisScreenBinding `json:"binding"`
	QualityClaim                      bool                                       `json:"quality_claim"`
	ReleaseClaim                      bool                                       `json:"release_claim"`
	OfficialClaim                     bool                                       `json:"official_claim"`
	OfficialHeldoutGate               bool                                       `json:"official_heldout_gate"`
	CommercialClaim                   bool                                       `json:"commercial_claim"`
	FreeOpenUseClaim                  bool                                       `json:"free_open_use_claim"`
}

type AOQTSidecarV8ConstrainedBasisScreenResult struct {
	Receipt         AOQTSidecarV8ConstrainedBasisScreenReceipt
	ReceiptSHA256   string
	ReceiptJSONPath string
}

func (r AOQTSidecarV8ConstrainedBasisScreenReceipt) SHA256() (string, error) {
	return aoqtCanonicalSHA256(r)
}

func (r AOQTSidecarV8ConstrainedBasisScreenReceipt) Validate() error {
	return r.validate(false)
}

func (r AOQTSidecarV8ConstrainedBasisScreenReceipt) ValidateBound() error {
	if r.CanonicalV7R6ReceiptFileSHA256 != AOQTV8CanonicalV7R6ReceiptFileSHA256 ||
		r.Binding.CanonicalV7R6ReceiptFileSHA256 != AOQTV8CanonicalV7R6ReceiptFileSHA256 {
		return fmt.Errorf("AOQT V8 constrained-basis canonical V7-r6 terminal receipt is not the pinned authorization")
	}
	if err := r.validate(true); err != nil {
		return err
	}
	if r.Binding.SplitBinding == nil {
		return fmt.Errorf("AOQT V8 constrained-basis split binding is required")
	}
	if err := r.Binding.SplitBinding.Validate(); err != nil {
		return fmt.Errorf("AOQT V8 constrained-basis split binding: %w", err)
	}
	if err := validateAOQTMetricInputs(r.Binding.Inputs); err != nil {
		return fmt.Errorf("AOQT V8 constrained-basis binding inputs: %w", err)
	}
	if r.Binding.IOReport.Schema != AOQTSidecarCalibrationIOReportSchema ||
		r.Binding.IOReport.ManifestSHA256 != r.Binding.Inputs.DatasetManifestSHA256 {
		return fmt.Errorf("AOQT V8 constrained-basis IO report does not match bound inputs")
	}
	if r.Binding.FoldID != r.Binding.SplitBinding.FoldID {
		return fmt.Errorf("AOQT V8 constrained-basis fold id does not match split binding")
	}
	for _, item := range []struct {
		name  string
		value string
	}{
		{"preflight sha256", r.Binding.PreflightSHA256},
		{"r4 receipt object sha256", r.Binding.CanonicalR4ReceiptSHA256},
		{"r4 receipt file sha256", r.Binding.CanonicalR4ReceiptFileSHA256},
		{"r5 receipt object sha256", r.Binding.CanonicalR5ReceiptSHA256},
		{"r5 receipt file sha256", r.Binding.CanonicalR5ReceiptFileSHA256},
		{"v7-r6 receipt file sha256", r.Binding.CanonicalV7R6ReceiptFileSHA256},
	} {
		if err := validateAOQTSHA256(item.value, "AOQT V8 constrained-basis "+item.name); err != nil {
			return err
		}
	}
	if r.Binding.CanonicalV7R6ReceiptFileSHA256 != AOQTV8CanonicalV7R6ReceiptFileSHA256 {
		return fmt.Errorf("AOQT V8 constrained-basis canonical V7-r6 terminal receipt is not the pinned authorization")
	}
	if r.Binding.CanonicalR4ReceiptFileSHA256 != AOQTV7R6CanonicalR4ReceiptFileSHA256 || r.Binding.CanonicalR5ReceiptFileSHA256 != AOQTV7R6CanonicalR5ReceiptFileSHA256 {
		return fmt.Errorf("AOQT V8 constrained-basis canonical r4/r5 receipts are not pinned")
	}
	return rereadAOQTV8ConstrainedBasisBinding(r)
}

func (r AOQTSidecarV8ConstrainedBasisScreenReceipt) validate(requireBinding bool) error {
	if r.Schema != AOQTSidecarV8ConstrainedBasisScreenSchema {
		return fmt.Errorf("AOQT V8 constrained-basis schema %q is unsupported, want %q", r.Schema, AOQTSidecarV8ConstrainedBasisScreenSchema)
	}
	if r.Mode != AOQTSidecarOptimizerModeV8ConstrainedBasisScreen {
		return fmt.Errorf("AOQT V8 constrained-basis mode %q is unsupported", r.Mode)
	}
	if !r.Required || !r.ProbeOnly {
		return fmt.Errorf("AOQT V8 constrained-basis receipt must be required and probe_only")
	}
	if r.QualityClaim || r.ReleaseClaim || r.OfficialClaim || r.OfficialHeldoutGate || r.CommercialClaim || r.FreeOpenUseClaim {
		return fmt.Errorf("AOQT V8 constrained-basis claims and official heldout gate must be false")
	}
	if r.BasisVersion != AOQTV8ConstrainedBasisVersion || r.ScheduleVersion != AOQTV8ConstrainedBasisScheduleVersion || r.RankingVersion != AOQTV8ConstrainedBasisRankingVersion || r.PolicyVersion != AOQTV8ConstrainedBasisPolicyVersion {
		return fmt.Errorf("AOQT V8 constrained-basis contract version binding is unsupported")
	}
	if r.Dim != AOQTV8ConstrainedBasisExpectedDim || r.CoordinateCount != AOQTV8ConstrainedBasisExpectedAngleSize {
		return fmt.Errorf("AOQT V8 constrained-basis must preserve D384 and 1536-angle AOQT serving format")
	}
	if !slices.Equal(r.Ranks, AOQTV8ConstrainedBasisRanks()) || !aoqtFloat32SlicesEqual(r.Magnitudes, AOQTV8ConstrainedBasisMagnitudes()) || !slices.Equal(r.GlobalSigns, AOQTV8ConstrainedBasisGlobalSigns()) {
		return fmt.Errorf("AOQT V8 constrained-basis schedule values are not canonical")
	}
	if r.CandidateCap != AOQTV8ConstrainedBasisCandidateCap || len(r.Endpoints) > AOQTV8ConstrainedBasisCandidateCap || r.ObjectiveEvaluationCount > AOQTV8ConstrainedBasisObjectiveEvalMax {
		return fmt.Errorf("AOQT V8 constrained-basis candidate/evaluation cap exceeded")
	}
	directionScheduleSHA, err := aoqtCanonicalSHA256(struct {
		Version    string    `json:"version"`
		Ranks      []int     `json:"ranks"`
		Magnitudes []float32 `json:"magnitudes"`
		Signs      []int     `json:"signs"`
		Cap        int       `json:"cap"`
	}{AOQTV8ConstrainedBasisScheduleVersion, AOQTV8ConstrainedBasisRanks(), AOQTV8ConstrainedBasisMagnitudes(), AOQTV8ConstrainedBasisGlobalSigns(), AOQTV8ConstrainedBasisCandidateCap})
	if err != nil {
		return err
	}
	if r.DirectionScheduleSHA256 != directionScheduleSHA {
		return fmt.Errorf("AOQT V8 constrained-basis direction schedule hash mismatch")
	}
	if r.ObjectiveEvaluationCount != 1+len(r.Endpoints) {
		return fmt.Errorf("AOQT V8 constrained-basis objective evaluation accounting must be baseline plus endpoints")
	}
	if r.FullFoldRowCount != AOQTV7R6ProgressiveScreenPopulationTotal || r.FullFoldWeightingVersion != AOQTV8ConstrainedBasisFullWeighting || r.HTFactorsApplied {
		return fmt.Errorf("AOQT V8 constrained-basis full-fold authority must be all 10,942 rows, unweighted")
	}
	if !r.BaselineLossFinite || !isFinite32(r.BaselineLoss) || !aoqtFiniteReceiptComponentsValue(r.BaselineComponents) || !aoqtV7R6PositiveActivation(r.BaselineActivation) {
		return fmt.Errorf("AOQT V8 constrained-basis baseline evidence is invalid")
	}
	if err := validateAOQTLossMatchesComponents(r.BaselineLoss, r.BaselineComponents, "AOQT V8 constrained-basis baseline"); err != nil {
		return err
	}
	policy := aoqtV7R6CanonicalEligibilityPolicy()
	policySHA, err := aoqtActualCoordinatePolicySHA256(policy)
	if err != nil {
		return err
	}
	if r.CanonicalEligibilityPolicySHA256 != policySHA {
		return fmt.Errorf("AOQT V8 constrained-basis eligibility policy must be the canonical joint-budget policy")
	}
	for _, item := range []struct{ name, value string }{
		{"selection_seed_sha256", r.SelectionSeedSHA256},
		{"q3_gradient_sha256", r.Q3GradientSHA256},
		{"protected_gradient_matrix_sha256", r.ProtectedGradientMatrixSHA256},
		{"basis_sha256", r.BasisSHA256},
		{"direction_schedule_sha256", r.DirectionScheduleSHA256},
		{"canonical_objective_contract_sha256", r.CanonicalObjectiveContractSHA256},
		{"canonical_v7_r6_receipt_file_sha256", r.CanonicalV7R6ReceiptFileSHA256},
		{"endpoint_hash_chain_tail_sha256", r.EndpointHashChainTailSHA256},
	} {
		if err := validateAOQTSHA256(item.value, "AOQT V8 constrained-basis "+item.name); err != nil {
			return err
		}
	}
	if r.CanonicalV7R6ReceiptFileSHA256 != AOQTV8CanonicalV7R6ReceiptFileSHA256 || !r.CanonicalV7R6TerminalReceiptBound {
		return fmt.Errorf("AOQT V8 constrained-basis must bind the canonical terminal V7-r6 receipt")
	}
	denseProofBound := false
	if r.DenseInvariantStatus == AOQTV8ConstrainedBasisDenseInvariantDeferred {
		if !r.DenseInvariantDeferred || r.DenseInvariantProofSHA256 != "" {
			return fmt.Errorf("AOQT V8 constrained-basis deferred dense invariant fields are inconsistent")
		}
	} else if r.DenseInvariantStatus == AOQTV8ConstrainedBasisDenseInvariantProof {
		if !requireBinding {
			return fmt.Errorf("AOQT V8 constrained-basis proved dense invariant requires bound validation over rows")
		}
		if r.DenseInvariantDeferred {
			return fmt.Errorf("AOQT V8 constrained-basis proved dense invariant cannot be deferred")
		}
		if r.GradientEvidenceDeferred || !r.FullVerificationCompleted {
			return fmt.Errorf("AOQT V8 constrained-basis proved dense invariant requires completed gradient and full verification evidence")
		}
		if len(r.Endpoints) != AOQTV8ConstrainedBasisCandidateCap || r.ObjectiveEvaluationCount != 1+AOQTV8ConstrainedBasisCandidateCap {
			return fmt.Errorf("AOQT V8 constrained-basis proved receipt must include the exact canonical endpoint schedule")
		}
		if err := validateAOQTV8CanonicalEndpointSchedule(r.Endpoints); err != nil {
			return err
		}
		if err := validateAOQTSHA256(r.DenseInvariantProofSHA256, "AOQT V8 constrained-basis dense invariant proof sha256"); err != nil {
			return err
		}
		denseProofBound = true
	} else {
		return fmt.Errorf("AOQT V8 constrained-basis dense invariant status is unsupported")
	}
	if !isFinite64(r.DenseMaxAbsDeltaTolerance) || r.DenseMaxAbsDeltaTolerance != policy.DenseMaxAbsDeltaTolerance {
		return fmt.Errorf("AOQT V8 constrained-basis dense tolerance must exactly match canonical V7-r6 policy")
	}
	basisSHA, err := aoqtCanonicalSHA256(r.Basis)
	if err != nil || basisSHA != r.BasisSHA256 {
		return fmt.Errorf("AOQT V8 constrained-basis basis hash mismatch")
	}
	if len(r.Basis) != AOQTV8ConstrainedBasisRankCount {
		return fmt.Errorf("AOQT V8 constrained-basis must record exactly three basis vectors")
	}
	basisByOrdinal := map[int]AOQTSidecarV8BasisVector{}
	for i, basis := range r.Basis {
		if basis.Ordinal != i+1 || basis.Kind != "sequential_projected_sparse_q3_support" || basis.Rank != r.Ranks[i] {
			return fmt.Errorf("AOQT V8 constrained-basis vector %d metadata is invalid", i)
		}
		if basis.SourceGradientSHA256 != r.Q3GradientSHA256 {
			return fmt.Errorf("AOQT V8 constrained-basis vector %d source gradient hash must match receipt q3 gradient hash", basis.Ordinal)
		}
		if err := validateAOQTV8BasisVector(basis, r.CoordinateCount); err != nil {
			return err
		}
		basisByOrdinal[basis.Ordinal] = basis
	}
	if err := validateAOQTV8EndpointHashChain(r); err != nil {
		return err
	}
	ranking := aoqtV8RankEndpoints(r.Endpoints, policy)
	rankAt := map[int]int{}
	for i, ordinal := range ranking {
		rankAt[ordinal] = i + 1
	}
	feasible := 0
	selected := make([]int, 0, len(r.Endpoints))
	for i, endpoint := range r.Endpoints {
		if err := validateAOQTV8EndpointDenseProofEnvelope(endpoint, r.DenseInvariantStatus, r.DenseMaxAbsDeltaTolerance); err != nil {
			return err
		}
		if err := validateAOQTV8Endpoint(endpoint, i, r.BaselineLoss, r.BaselineComponents, policy, basisByOrdinal, denseProofBound); err != nil {
			return err
		}
		if endpoint.RankOrder != rankAt[endpoint.Ordinal] {
			return fmt.Errorf("AOQT V8 constrained-basis endpoint %d rank_order does not match recomputed ranking", endpoint.Ordinal)
		}
		if endpoint.TransactionEligible {
			feasible++
			selected = append(selected, endpoint.Ordinal)
		}
	}
	if r.FeasibleEndpointCount != feasible || !reflect.DeepEqual(r.SelectedEndpointOrdinals, selected) {
		return fmt.Errorf("AOQT V8 constrained-basis selected/feasible endpoint accounting does not match recomputed policy")
	}
	wantPassed := denseProofBound && feasible > 0
	if r.Passed != wantPassed {
		return fmt.Errorf("AOQT V8 constrained-basis passed state does not match full proof evidence")
	}
	if !r.Passed && strings.TrimSpace(r.FailureReason) == "" {
		return fmt.Errorf("AOQT V8 constrained-basis failed receipt requires failure_reason")
	}
	if r.Passed && r.FailureReason != "" {
		return fmt.Errorf("AOQT V8 constrained-basis passed receipt has failure_reason")
	}
	if requireBinding {
		if r.Binding.IOReport.ManifestPath == "" || r.Binding.IOReport.RowsJSONLPath == "" || r.Binding.PreflightPath == "" ||
			r.Binding.CanonicalR4ReceiptPath == "" || r.Binding.CanonicalR5ReceiptPath == "" || r.Binding.CanonicalV7R6ReceiptPath == "" {
			return fmt.Errorf("AOQT V8 constrained-basis bound receipt requires all source paths")
		}
	}
	return nil
}

func validateAOQTV8BasisVector(basis AOQTSidecarV8BasisVector, coordinateCount int) error {
	if len(basis.Direction) != coordinateCount {
		return fmt.Errorf("AOQT V8 constrained-basis vector %d direction length = %d, want %d", basis.Ordinal, len(basis.Direction), coordinateCount)
	}
	if len(basis.SelectedIndices) != basis.Rank || len(basis.SelectedSigns) != basis.Rank {
		return fmt.Errorf("AOQT V8 constrained-basis vector %d rank support is invalid", basis.Ordinal)
	}
	seen := map[int]bool{}
	nonzero := 0
	maxAbs := float32(0)
	for i, value := range basis.Direction {
		if !isFinite32(value) {
			return fmt.Errorf("AOQT V8 constrained-basis vector %d direction[%d] is not finite", basis.Ordinal, i)
		}
		abs := float32(math.Abs(float64(value)))
		if abs > 0 {
			nonzero++
		}
		if abs > maxAbs {
			maxAbs = abs
		}
	}
	if nonzero != basis.Rank || maxAbs <= 0 || maxAbs > 1+1e-6 {
		return fmt.Errorf("AOQT V8 constrained-basis vector %d support/maxabs is invalid", basis.Ordinal)
	}
	for i, index := range basis.SelectedIndices {
		if index < 0 || index >= coordinateCount || seen[index] {
			return fmt.Errorf("AOQT V8 constrained-basis vector %d selected index is invalid", basis.Ordinal)
		}
		seen[index] = true
		sign := basis.SelectedSigns[i]
		if sign != 1 && sign != -1 {
			return fmt.Errorf("AOQT V8 constrained-basis vector %d selected sign is invalid", basis.Ordinal)
		}
		got := 0
		if basis.Direction[index] > 0 {
			got = 1
		} else if basis.Direction[index] < 0 {
			got = -1
		}
		if got != sign {
			return fmt.Errorf("AOQT V8 constrained-basis vector %d selected sign does not match direction", basis.Ordinal)
		}
	}
	hash, err := aoqtActualDirectionVectorSHA256(basis.Direction)
	if err != nil || hash != basis.DirectionSHA256 {
		return fmt.Errorf("AOQT V8 constrained-basis vector %d direction hash mismatch", basis.Ordinal)
	}
	if err := validateAOQTSHA256(basis.SourceGradientSHA256, "AOQT V8 constrained-basis source gradient sha256"); err != nil {
		return err
	}
	return nil
}

func validateAOQTV8EndpointDenseProofEnvelope(endpoint AOQTSidecarV8ConstrainedBasisEndpoint, denseStatus string, tolerance float64) error {
	switch denseStatus {
	case AOQTV8ConstrainedBasisDenseInvariantDeferred:
		if endpoint.DenseInvariantProved || endpoint.DenseInvariantEligible || endpoint.DenseInvariantMaxAbsDelta != 0 || endpoint.DenseInvariantProofSHA256 != "" {
			return fmt.Errorf("AOQT V8 constrained-basis endpoint %d dense proof must be empty while receipt dense proof is deferred", endpoint.Ordinal)
		}
	case AOQTV8ConstrainedBasisDenseInvariantProof:
		if !endpoint.DenseInvariantProved {
			return fmt.Errorf("AOQT V8 constrained-basis endpoint %d dense invariant proof is required", endpoint.Ordinal)
		}
		if !isFinite64(endpoint.DenseInvariantMaxAbsDelta) || endpoint.DenseInvariantMaxAbsDelta < 0 {
			return fmt.Errorf("AOQT V8 constrained-basis endpoint %d dense invariant max abs delta is invalid", endpoint.Ordinal)
		}
		if endpoint.DenseInvariantEligible != (endpoint.DenseInvariantMaxAbsDelta <= tolerance) {
			return fmt.Errorf("AOQT V8 constrained-basis endpoint %d dense invariant eligibility does not match tolerance", endpoint.Ordinal)
		}
		if err := validateAOQTSHA256(endpoint.DenseInvariantProofSHA256, "AOQT V8 constrained-basis endpoint dense invariant proof sha256"); err != nil {
			return err
		}
	default:
		return fmt.Errorf("AOQT V8 constrained-basis dense invariant status is unsupported")
	}
	return nil
}

func validateAOQTV8Endpoint(endpoint AOQTSidecarV8ConstrainedBasisEndpoint, index int, baselineLoss float32, baselineComponents AOQTSidecarObjectiveComponents, policy AOQTSidecarCandidateEligibilityPolicy, basis map[int]AOQTSidecarV8BasisVector, denseProofBound bool) error {
	if endpoint.Ordinal != index+1 {
		return fmt.Errorf("AOQT V8 constrained-basis endpoint ordinal %d is not sequential", endpoint.Ordinal)
	}
	basisVector, ok := basis[endpoint.BasisOrdinal]
	if !ok || basisVector.Rank != endpoint.Rank {
		return fmt.Errorf("AOQT V8 constrained-basis endpoint %d basis binding is invalid", endpoint.Ordinal)
	}
	if !containsFloat32AOQT(endpoint.Magnitude, AOQTV8ConstrainedBasisMagnitudes()) || !containsInt(endpoint.GlobalSign, AOQTV8ConstrainedBasisGlobalSigns()) {
		return fmt.Errorf("AOQT V8 constrained-basis endpoint %d schedule binding is invalid", endpoint.Ordinal)
	}
	if len(endpoint.ActualDirection) != AOQTV8ConstrainedBasisExpectedAngleSize {
		return fmt.Errorf("AOQT V8 constrained-basis endpoint %d actual direction length is invalid", endpoint.Ordinal)
	}
	requested := make([]float32, len(basisVector.Direction))
	for i, value := range basisVector.Direction {
		requested[i] = value * endpoint.Magnitude * float32(endpoint.GlobalSign)
	}
	requestedSHA, err := aoqtActualDirectionVectorSHA256(requested)
	if err != nil || requestedSHA != endpoint.RequestedDirectionSHA256 {
		return fmt.Errorf("AOQT V8 constrained-basis endpoint %d requested direction hash mismatch", endpoint.Ordinal)
	}
	actualSHA, err := aoqtActualDirectionVectorSHA256(endpoint.ActualDirection)
	if err != nil || actualSHA != endpoint.ActualDirectionSHA256 {
		return fmt.Errorf("AOQT V8 constrained-basis endpoint %d actual direction hash mismatch", endpoint.Ordinal)
	}
	if max := aoqtMaxAbsFloat32(endpoint.ActualDirection); max != endpoint.ActualDirectionMaxAbs {
		return fmt.Errorf("AOQT V8 constrained-basis endpoint %d actual direction maxabs mismatch", endpoint.Ordinal)
	}
	if endpoint.AngleMoved != (endpoint.ActualDirectionMaxAbs != 0) {
		return fmt.Errorf("AOQT V8 constrained-basis endpoint %d angle_moved does not match actual direction", endpoint.Ordinal)
	}
	if !slices.Equal(endpoint.ActualDirection, requested) {
		return fmt.Errorf("AOQT V8 constrained-basis endpoint %d actual direction must exactly match requested scheduled direction", endpoint.Ordinal)
	}
	if endpoint.BaselineLoss != baselineLoss || endpoint.BaselineComponents != baselineComponents || !endpoint.BaselineLossFinite || !endpoint.BaselineComponentsFinite {
		return fmt.Errorf("AOQT V8 constrained-basis endpoint %d baseline does not match receipt baseline", endpoint.Ordinal)
	}
	if !endpoint.CandidateLossFinite || !isFinite32(endpoint.CandidateLoss) || !endpoint.CandidateComponentsFinite || !aoqtFiniteReceiptComponentsValue(endpoint.CandidateComponents) || !endpoint.TotalDeltaFinite {
		return fmt.Errorf("AOQT V8 constrained-basis endpoint %d candidate evidence is not finite", endpoint.Ordinal)
	}
	if err := validateAOQTLossMatchesComponents(endpoint.CandidateLoss, endpoint.CandidateComponents, fmt.Sprintf("AOQT V8 constrained-basis endpoint %d candidate", endpoint.Ordinal)); err != nil {
		return err
	}
	wantDelta := aoqtObjectiveComponentDelta(endpoint.BaselineComponents, endpoint.CandidateComponents)
	if endpoint.ComponentDeltas != wantDelta || !endpoint.ComponentDeltasFinite {
		return fmt.Errorf("AOQT V8 constrained-basis endpoint %d component deltas mismatch", endpoint.Ordinal)
	}
	if endpoint.TotalDelta != endpoint.CandidateLoss-endpoint.BaselineLoss {
		return fmt.Errorf("AOQT V8 constrained-basis endpoint %d total delta mismatch", endpoint.Ordinal)
	}
	wantQ3 := endpoint.CandidateComponents.Q3Gain < endpoint.BaselineComponents.Q3Gain-aoqtTransactionalQ3ImprovementMinMagnitude
	wantProtected := len(aoqtComponentRegressions(endpoint.BaselineComponents, endpoint.CandidateComponents, policy)) == 0
	decision := aoqtEvaluateTransactionalStepWithPolicy(
		aoqtStepEvaluation{loss: endpoint.BaselineLoss, activation: AOQTSidecarObjectiveActivation{}, components: endpoint.BaselineComponents},
		aoqtStepEvaluation{loss: endpoint.CandidateLoss, activation: endpoint.CandidateActivation, components: endpoint.CandidateComponents},
		AOQTSidecarRowWeights{Q3Gain: 1, Q3OrderGuard: 1, Q3ScoreDistill: 1, Q5OrderGuard: 1, Q5ScoreDistill: 1, NFBoundaryGuard: 1},
		policy,
	)
	if !endpoint.AngleMoved {
		if endpoint.TransactionEligible {
			return fmt.Errorf("AOQT V8 constrained-basis endpoint %d cannot be transaction eligible without angle movement", endpoint.Ordinal)
		}
		if endpoint.Reason != string(aoqtRejectionNoAngleMovement) {
			return fmt.Errorf("AOQT V8 constrained-basis endpoint %d reason %q does not match no-angle movement", endpoint.Ordinal, endpoint.Reason)
		}
		return nil
	}
	wantTransaction := decision.accepted
	if denseProofBound {
		wantTransaction = wantTransaction && endpoint.DenseInvariantEligible
	}
	if endpoint.Q3GainEligible != wantQ3 || endpoint.ProtectedEligible != wantProtected || endpoint.TransactionEligible != wantTransaction {
		return fmt.Errorf("AOQT V8 constrained-basis endpoint %d policy flags do not match recomputed evidence", endpoint.Ordinal)
	}
	wantReason := string(decision.reason)
	if decision.accepted {
		if denseProofBound && !endpoint.DenseInvariantEligible {
			wantReason = AOQTV8ConstrainedBasisDenseInvariantRejection
		} else {
			wantReason = aoqtProposalAcceptedReason
		}
	}
	if endpoint.Reason != wantReason {
		return fmt.Errorf("AOQT V8 constrained-basis endpoint %d reason %q does not match recomputed %q", endpoint.Ordinal, endpoint.Reason, wantReason)
	}
	return nil
}

func validateAOQTV8EndpointHashChain(r AOQTSidecarV8ConstrainedBasisScreenReceipt) error {
	var chain []string
	if err := strictUnmarshalAOQT([]byte(r.EndpointHashChain), &chain); err != nil {
		return fmt.Errorf("AOQT V8 constrained-basis endpoint hash chain decode: %w", err)
	}
	if len(chain) != len(r.Endpoints) {
		return fmt.Errorf("AOQT V8 constrained-basis endpoint hash chain length mismatch")
	}
	previous := ""
	for i, endpoint := range r.Endpoints {
		payload, err := json.Marshal(endpoint)
		if err != nil {
			return err
		}
		previous = sha256AOQTCoordinateSearchChainEntry(previous, string(payload))
		if chain[i] != previous {
			return fmt.Errorf("AOQT V8 constrained-basis endpoint hash chain entry %d mismatch", i)
		}
	}
	if previous != r.EndpointHashChainTailSHA256 {
		return fmt.Errorf("AOQT V8 constrained-basis endpoint hash chain tail mismatch")
	}
	return nil
}

func rereadAOQTV8ConstrainedBasisBinding(r AOQTSidecarV8ConstrainedBasisScreenReceipt) error {
	binding := r.Binding
	manifestData, err := os.ReadFile(binding.IOReport.ManifestPath)
	if err != nil {
		return fmt.Errorf("AOQT V8 constrained-basis bound manifest reread: %w", err)
	}
	var manifest AOQTSidecarCalibrationManifest
	if err := strictUnmarshalAOQT(manifestData, &manifest); err != nil {
		return fmt.Errorf("AOQT V8 constrained-basis bound manifest is invalid: %w", err)
	}
	manifestSHA, err := AOQTSidecarManifestSHA256(manifest)
	if err != nil {
		return err
	}
	if manifestSHA != binding.IOReport.ManifestSHA256 || manifestSHA != binding.Inputs.DatasetManifestSHA256 {
		return fmt.Errorf("AOQT V8 constrained-basis bound manifest hash mismatch")
	}
	rowsData, err := os.ReadFile(binding.IOReport.RowsJSONLPath)
	if err != nil {
		return fmt.Errorf("AOQT V8 constrained-basis bound rows reread: %w", err)
	}
	rowsSHA := sha256BytesAOQT(rowsData)
	if rowsSHA != binding.IOReport.RowsSHA256 || rowsSHA != binding.SplitBinding.MaterializedRowsSHA256 {
		return fmt.Errorf("AOQT V8 constrained-basis bound rows hash mismatch")
	}
	rows, err := parseAOQTCalibrationRowsJSONL(rowsData, binding.IOReport.RowsJSONLPath)
	if err != nil {
		return err
	}
	set := AOQTSidecarCalibrationSet{Manifest: manifest, Rows: rows}
	if err := set.Validate(); err != nil {
		return fmt.Errorf("AOQT V8 constrained-basis bound set validation: %w", err)
	}
	if len(rows) != r.FullFoldRowCount || len(rows) != AOQTV7R6ProgressiveScreenPopulationTotal {
		return fmt.Errorf("AOQT V8 constrained-basis reread full-fold row count mismatch")
	}
	if rowIDs := aoqtV7R6OrderedRowIDSHA256(aoqtCalibrationRowIDs(rows)); rowIDs != binding.SplitBinding.MaterializedRowIDSHA256 {
		return fmt.Errorf("AOQT V8 constrained-basis row-id hash mismatch")
	}
	expectedReport, err := aoqtV7R6IOReportForSet(set, binding.IOReport.ManifestPath, binding.IOReport.RowsJSONLPath, rowsSHA)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(expectedReport, binding.IOReport) {
		return fmt.Errorf("AOQT V8 constrained-basis IO report does not match reread manifest/rows")
	}
	preflightData, err := os.ReadFile(binding.PreflightPath)
	if err != nil {
		return fmt.Errorf("AOQT V8 constrained-basis bound preflight reread: %w", err)
	}
	preflightSHA := sha256BytesAOQT(preflightData)
	if preflightSHA != binding.PreflightSHA256 || preflightSHA != binding.SplitBinding.MaterializedPreflightSHA256 {
		return fmt.Errorf("AOQT V8 constrained-basis preflight hash mismatch")
	}
	var preflight AOQTSidecarMaterializePreflight
	if err := strictUnmarshalAOQT(preflightData, &preflight); err != nil {
		return fmt.Errorf("AOQT V8 constrained-basis preflight is invalid: %w", err)
	}
	if err := validateAOQTFailClosedPreflight(preflight, binding.PreflightPath, expectedReport, binding.Inputs, manifest.Topology, manifest.ObjectiveContract, manifest.LegalGates); err != nil {
		return fmt.Errorf("AOQT V8 constrained-basis preflight binding: %w", err)
	}
	if binding.SplitBinding.FoldID != binding.FoldID ||
		binding.SplitBinding.MaterializedManifestSHA256 != manifestSHA ||
		binding.SplitBinding.MaterializedRowsSHA256 != rowsSHA ||
		binding.SplitBinding.MaterializedPreflightSHA256 != preflightSHA ||
		binding.SplitBinding.MaterializedRowCount != len(rows) {
		return fmt.Errorf("AOQT V8 constrained-basis split binding does not match reread inputs")
	}
	r4ObjectSHA, err := rereadAOQTV8PinnedR4Receipt(binding.CanonicalR4ReceiptPath, binding.CanonicalR4ReceiptFileSHA256)
	if err != nil {
		return err
	}
	if r4ObjectSHA != binding.CanonicalR4ReceiptSHA256 {
		return fmt.Errorf("AOQT V8 constrained-basis canonical r4 receipt object sha256 mismatch")
	}
	r5ObjectSHA, err := rereadAOQTV8PinnedR5Receipt(binding.CanonicalR5ReceiptPath, binding.CanonicalR5ReceiptFileSHA256)
	if err != nil {
		return err
	}
	if r5ObjectSHA != binding.CanonicalR5ReceiptSHA256 {
		return fmt.Errorf("AOQT V8 constrained-basis canonical r5 receipt object sha256 mismatch")
	}
	v7r6Data, err := os.ReadFile(binding.CanonicalV7R6ReceiptPath)
	if err != nil {
		return fmt.Errorf("AOQT V8 constrained-basis canonical V7-r6 receipt reread: %w", err)
	}
	v7r6SHA := sha256BytesAOQT(v7r6Data)
	if v7r6SHA != binding.CanonicalV7R6ReceiptFileSHA256 || v7r6SHA != r.CanonicalV7R6ReceiptFileSHA256 || v7r6SHA != AOQTV8CanonicalV7R6ReceiptFileSHA256 {
		return fmt.Errorf("AOQT V8 constrained-basis canonical V7-r6 receipt hash mismatch")
	}
	var v7r6 AOQTSidecarV7R6ProgressiveScreenReceipt
	if err := strictUnmarshalAOQT(v7r6Data, &v7r6); err != nil {
		return fmt.Errorf("AOQT V8 constrained-basis canonical V7-r6 receipt decode: %w", err)
	}
	if err := v7r6.ValidateBound(); err != nil {
		return fmt.Errorf("AOQT V8 constrained-basis canonical V7-r6 receipt validation: %w", err)
	}
	if v7r6.Passed {
		return fmt.Errorf("AOQT V8 constrained-basis canonical V7-r6 receipt must be terminal negative")
	}
	if err := validateAOQTV8V7R6Compatibility(v7r6, binding, r.FullFoldRowCount, r.FullFoldWeightingVersion, r.HTFactorsApplied); err != nil {
		return err
	}
	contractSHA, err := aoqtActualCoordinateObjectiveContractSHA256(manifest.ObjectiveContract)
	if err != nil {
		return err
	}
	if r.CanonicalObjectiveContractSHA256 != contractSHA {
		return fmt.Errorf("AOQT V8 constrained-basis objective contract hash does not match reread fold manifest")
	}
	policy, err := AOQTSidecarTrainingContractPolicy(manifest.TrainingContract, manifest.CandidateEligibilityPolicy)
	if err != nil {
		return err
	}
	policySHA, err := aoqtActualCoordinatePolicySHA256(policy)
	if err != nil {
		return err
	}
	if r.CanonicalEligibilityPolicySHA256 != policySHA {
		return fmt.Errorf("AOQT V8 constrained-basis eligibility policy hash does not match reread fold manifest")
	}
	selectionSeed, err := aoqtV8SelectionSeed(set, binding.FoldID, r.Q3GradientSHA256, r.ProtectedGradientMatrixSHA256, r.BasisSHA256, v7r6SHA)
	if err != nil {
		return err
	}
	if selectionSeed != r.SelectionSeedSHA256 {
		return fmt.Errorf("AOQT V8 constrained-basis selection seed hash mismatch")
	}
	if r.DenseInvariantStatus == AOQTV8ConstrainedBasisDenseInvariantProof {
		trainer, err := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{Dim: manifest.Topology.Dim, Stages: manifest.Topology.Stages, PairingSeed: manifest.Topology.Seed, WorkplanSeed: 1, OptimizerMode: AOQTSidecarOptimizerModeV7R4DevActualCoordinate, ActualCoordinateProbeRequired: true, ActualCoordinateProbeOnly: true, ActualCoordinateProbeFoldID: binding.FoldID, ActualCoordinateProbeSplitManifestSHA256: binding.SplitBinding.SplitManifestSHA256, LearningRate: AOQTV7R4ActualCoordinateProbeLearningRate, AngleCap: AOQTSidecarDefaultAngleCap, MaxAngleCap: AOQTSidecarHardMaxAngleCap, TrainingContract: manifest.TrainingContract, CandidateEligibilityPolicy: cloneAOQTSidecarCandidateEligibilityPolicy(manifest.CandidateEligibilityPolicy)})
		if err != nil {
			return err
		}
		if _, err := trainer.Plan(set); err != nil {
			return err
		}
		objective, err := objectiveFromAOQTContract(manifest.ObjectiveContract)
		if err != nil {
			return err
		}
		if err := validateAOQTV8BoundBasisEvidence(r, trainer, rows, objective, manifest.ObjectiveContract.WeightSums); err != nil {
			return err
		}
		if err := validateAOQTV8BoundObjectiveReplay(r, trainer, set, objective, manifest.ObjectiveContract.WeightSums); err != nil {
			return err
		}
		if err := validateAOQTV8BoundDenseProofs(r, set, binding, trainer.topology, r.DenseMaxAbsDeltaTolerance); err != nil {
			return err
		}
	}
	return nil
}

func validateAOQTV8BoundBasisEvidence(r AOQTSidecarV8ConstrainedBasisScreenReceipt, trainer *AOQTSidecarTrainer, rows []AOQTSidecarCalibrationRow, objective AOQTSidecarVectorObjective, weights AOQTSidecarRowWeights) error {
	baselineLoss, _, baselineActivation, baselineComponents, err := trainer.lossAndAngleGrad(rows, objective)
	if err != nil {
		return err
	}
	if r.BaselineLoss != baselineLoss || !r.BaselineLossFinite || r.BaselineComponents != baselineComponents || r.BaselineActivation != baselineActivation {
		return fmt.Errorf("AOQT V8 constrained-basis baseline objective evidence does not match bound replay")
	}
	q3Grad, err := trainer.q3GainOnlyAngleGrad(rows, objective)
	if err != nil {
		return err
	}
	protected, err := trainer.protectedComponentAngleGrads(rows, objective, weights)
	if err != nil {
		return err
	}
	basis, q3SHA, protectedNames, protectedSHA, basisSHA, err := aoqtV8ConstructBasis(q3Grad, protected, AOQTV8ConstrainedBasisRanks())
	if err != nil {
		return err
	}
	if r.Q3GradientSHA256 != q3SHA || r.ProtectedGradientMatrixSHA256 != protectedSHA || r.BasisSHA256 != basisSHA ||
		!slices.Equal(r.ProtectedGradientNames, protectedNames) || !reflect.DeepEqual(r.Basis, basis) {
		return fmt.Errorf("AOQT V8 constrained-basis gradient/basis evidence does not match bound replay")
	}
	return nil
}

type aoqtV8EndpointScheduleItem struct {
	Ordinal      int
	BasisOrdinal int
	Rank         int
	Magnitude    float32
	GlobalSign   int
}

func aoqtV8CanonicalEndpointSchedule() ([]aoqtV8EndpointScheduleItem, error) {
	out := make([]aoqtV8EndpointScheduleItem, 0, AOQTV8ConstrainedBasisCandidateCap)
	ordinal := 1
	for basisIndex, rank := range AOQTV8ConstrainedBasisRanks() {
		for _, magnitude := range AOQTV8ConstrainedBasisMagnitudes() {
			for _, sign := range AOQTV8ConstrainedBasisGlobalSigns() {
				if len(out) >= AOQTV8ConstrainedBasisCandidateCap {
					break
				}
				out = append(out, aoqtV8EndpointScheduleItem{Ordinal: ordinal, BasisOrdinal: basisIndex + 1, Rank: rank, Magnitude: magnitude, GlobalSign: sign})
				ordinal++
			}
		}
	}
	if len(out) != AOQTV8ConstrainedBasisCandidateCap {
		return nil, fmt.Errorf("AOQT V8 constrained-basis canonical endpoint schedule has %d entries, want %d", len(out), AOQTV8ConstrainedBasisCandidateCap)
	}
	return out, nil
}

func validateAOQTV8CanonicalEndpointSchedule(endpoints []AOQTSidecarV8ConstrainedBasisEndpoint) error {
	schedule, err := aoqtV8CanonicalEndpointSchedule()
	if err != nil {
		return err
	}
	if len(endpoints) != len(schedule) {
		return fmt.Errorf("AOQT V8 constrained-basis proved receipt endpoint count = %d, want canonical %d", len(endpoints), len(schedule))
	}
	seen := map[int]bool{}
	for i, endpoint := range endpoints {
		want := schedule[i]
		if seen[endpoint.Ordinal] {
			return fmt.Errorf("AOQT V8 constrained-basis duplicate endpoint ordinal %d", endpoint.Ordinal)
		}
		seen[endpoint.Ordinal] = true
		if endpoint.Ordinal != want.Ordinal ||
			endpoint.BasisOrdinal != want.BasisOrdinal ||
			endpoint.Rank != want.Rank ||
			endpoint.Magnitude != want.Magnitude ||
			endpoint.GlobalSign != want.GlobalSign {
			return fmt.Errorf("AOQT V8 constrained-basis endpoint %d does not match canonical schedule position %d", endpoint.Ordinal, i+1)
		}
	}
	return nil
}

func validateAOQTV8BoundObjectiveReplay(r AOQTSidecarV8ConstrainedBasisScreenReceipt, trainer *AOQTSidecarTrainer, set AOQTSidecarCalibrationSet, objective AOQTSidecarVectorObjective, weights AOQTSidecarRowWeights) error {
	baseline := aoqtStepEvaluation{loss: r.BaselineLoss, activation: r.BaselineActivation, components: r.BaselineComponents}
	expected := make([]AOQTSidecarV8ConstrainedBasisEndpoint, len(r.Endpoints))
	schedule, err := aoqtV8CanonicalEndpointSchedule()
	if err != nil {
		return err
	}
	if len(schedule) != len(r.Endpoints) {
		return fmt.Errorf("AOQT V8 constrained-basis objective replay requires canonical endpoint count")
	}
	for i, item := range schedule {
		if item.BasisOrdinal <= 0 || item.BasisOrdinal > len(r.Basis) {
			return fmt.Errorf("AOQT V8 constrained-basis schedule basis ordinal is invalid for replay")
		}
		expected[i] = trainer.evaluateV8ConstrainedBasisEndpoint(set.Rows, objective, baseline, weights, r.Basis[item.BasisOrdinal-1], item.Ordinal, item.Magnitude, item.GlobalSign)
		expected[i].RankOrder = r.Endpoints[i].RankOrder
	}
	proofSHA, err := applyAOQTV8DenseProofsToEndpoints(expected, set, r.Binding, trainer.topology, r.DenseMaxAbsDeltaTolerance)
	if err != nil {
		return err
	}
	if proofSHA != r.DenseInvariantProofSHA256 {
		return fmt.Errorf("AOQT V8 constrained-basis objective replay dense aggregate hash mismatch")
	}
	for i := range r.Endpoints {
		if err := compareAOQTV8EndpointReplay(r.Endpoints[i], expected[i]); err != nil {
			return err
		}
	}
	return nil
}

func compareAOQTV8EndpointReplay(got, want AOQTSidecarV8ConstrainedBasisEndpoint) error {
	if got.Ordinal != want.Ordinal ||
		got.BasisOrdinal != want.BasisOrdinal ||
		got.Rank != want.Rank ||
		got.Magnitude != want.Magnitude ||
		got.GlobalSign != want.GlobalSign ||
		got.RequestedDirectionSHA256 != want.RequestedDirectionSHA256 ||
		got.ActualDirectionSHA256 != want.ActualDirectionSHA256 ||
		got.ActualDirectionMaxAbs != want.ActualDirectionMaxAbs ||
		!slices.Equal(got.ActualDirection, want.ActualDirection) ||
		got.AngleMoved != want.AngleMoved ||
		got.BaselineLoss != want.BaselineLoss ||
		got.BaselineLossFinite != want.BaselineLossFinite ||
		got.CandidateLoss != want.CandidateLoss ||
		got.CandidateLossFinite != want.CandidateLossFinite ||
		got.BaselineComponents != want.BaselineComponents ||
		got.BaselineComponentsFinite != want.BaselineComponentsFinite ||
		got.CandidateComponents != want.CandidateComponents ||
		got.CandidateComponentsFinite != want.CandidateComponentsFinite ||
		got.ComponentDeltas != want.ComponentDeltas ||
		got.ComponentDeltasFinite != want.ComponentDeltasFinite ||
		got.CandidateActivation != want.CandidateActivation ||
		got.TotalDelta != want.TotalDelta ||
		got.TotalDeltaFinite != want.TotalDeltaFinite ||
		got.Q3GainEligible != want.Q3GainEligible ||
		got.ProtectedEligible != want.ProtectedEligible ||
		got.TransactionEligible != want.TransactionEligible ||
		got.Reason != want.Reason ||
		got.CandidateError != want.CandidateError ||
		got.RankOrder != want.RankOrder ||
		got.DenseInvariantProved != want.DenseInvariantProved ||
		got.DenseInvariantEligible != want.DenseInvariantEligible ||
		got.DenseInvariantMaxAbsDelta != want.DenseInvariantMaxAbsDelta ||
		got.DenseInvariantProofSHA256 != want.DenseInvariantProofSHA256 {
		return fmt.Errorf("AOQT V8 constrained-basis endpoint %d objective replay evidence mismatch", got.Ordinal)
	}
	return nil
}

func aoqtV8EndpointEvidence(endpoint AOQTSidecarV8ConstrainedBasisEndpoint) AOQTSidecarV8EndpointEvidence {
	return AOQTSidecarV8EndpointEvidence{
		Ordinal:                  endpoint.Ordinal,
		BasisOrdinal:             endpoint.BasisOrdinal,
		Rank:                     endpoint.Rank,
		Magnitude:                endpoint.Magnitude,
		GlobalSign:               endpoint.GlobalSign,
		RequestedDirectionSHA256: endpoint.RequestedDirectionSHA256,
		ActualDirectionSHA256:    endpoint.ActualDirectionSHA256,
		ActualDirectionMaxAbs:    endpoint.ActualDirectionMaxAbs,
		AngleMoved:               endpoint.AngleMoved,
		BaselineLoss:             endpoint.BaselineLoss,
		CandidateLoss:            endpoint.CandidateLoss,
	}
}

func aoqtV8CandidateTransformFromDirection(topology AOQTGivensTransform, direction []float32) (AOQTGivensTransform, error) {
	if len(direction) != countAOQTAngles(topology) {
		return AOQTGivensTransform{}, fmt.Errorf("AOQT V8 constrained-basis dense proof direction count = %d, want %d", len(direction), countAOQTAngles(topology))
	}
	transform := topology
	transform.Stages = append([]AOQTStage(nil), topology.Stages...)
	offset := 0
	for i := range transform.Stages {
		pairCount := len(topology.Stages[i].Pairs)
		transform.Stages[i].Pairs = append([][2]int(nil), topology.Stages[i].Pairs...)
		transform.Stages[i].Angles = append([]float32(nil), direction[offset:offset+pairCount]...)
		offset += pairCount
	}
	anglesSHA, err := transform.AnglesSHA256()
	if err != nil {
		return AOQTGivensTransform{}, err
	}
	pairingsSHA, err := transform.PairingsSHA256()
	if err != nil {
		return AOQTGivensTransform{}, err
	}
	orth, err := transform.OrthogonalityFrobeniusPerDim()
	if err != nil {
		return AOQTGivensTransform{}, err
	}
	transform.Audit = AOQTAuditInfo{PairingsSHA256: pairingsSHA, AnglesSHA256: anglesSHA, Orthogonality: &orth}
	return transform, nil
}

func aoqtV8DenseInvariantEndpointProof(endpoint AOQTSidecarV8ConstrainedBasisEndpoint, set AOQTSidecarCalibrationSet, binding AOQTSidecarV8ConstrainedBasisScreenBinding, topology AOQTGivensTransform, tolerance float64) (AOQTSidecarV8DenseInvariantEndpointProof, string, error) {
	if tolerance <= 0 || !isFinite64(tolerance) {
		return AOQTSidecarV8DenseInvariantEndpointProof{}, "", fmt.Errorf("AOQT V8 constrained-basis dense proof tolerance is invalid")
	}
	if len(set.Rows) == 0 {
		return AOQTSidecarV8DenseInvariantEndpointProof{}, "", fmt.Errorf("AOQT V8 constrained-basis dense proof requires rows")
	}
	actualSHA, err := aoqtActualDirectionVectorSHA256(endpoint.ActualDirection)
	if err != nil {
		return AOQTSidecarV8DenseInvariantEndpointProof{}, "", err
	}
	if actualSHA != endpoint.ActualDirectionSHA256 {
		return AOQTSidecarV8DenseInvariantEndpointProof{}, "", fmt.Errorf("AOQT V8 constrained-basis endpoint %d dense proof actual direction hash mismatch", endpoint.Ordinal)
	}
	transform, err := aoqtV8CandidateTransformFromDirection(topology, endpoint.ActualDirection)
	if err != nil {
		return AOQTSidecarV8DenseInvariantEndpointProof{}, "", err
	}
	pairingsSHA, err := transform.PairingsSHA256()
	if err != nil {
		return AOQTSidecarV8DenseInvariantEndpointProof{}, "", err
	}
	anglesSHA, err := transform.AnglesSHA256()
	if err != nil {
		return AOQTSidecarV8DenseInvariantEndpointProof{}, "", err
	}
	maxDelta, err := DenseInvariantMaxAbsDelta(transform, set.Rows)
	if err != nil {
		return AOQTSidecarV8DenseInvariantEndpointProof{}, "", err
	}
	proof := AOQTSidecarV8DenseInvariantEndpointProof{
		Schema:                    AOQTV8ConstrainedBasisDenseInvariantProofSchema,
		Version:                   AOQTV8ConstrainedBasisDenseInvariantProof,
		EndpointOrdinal:           endpoint.Ordinal,
		EndpointEvidence:          aoqtV8EndpointEvidence(endpoint),
		ManifestSHA256:            binding.IOReport.ManifestSHA256,
		RowsSHA256:                binding.IOReport.RowsSHA256,
		RowIDSHA256:               set.Manifest.RowIDSHA256,
		RowCount:                  len(set.Rows),
		FoldID:                    binding.FoldID,
		FullFoldWeightingVersion:  AOQTV8ConstrainedBasisFullWeighting,
		HTFactorsApplied:          false,
		CandidatePairingsSHA256:   pairingsSHA,
		CandidateAnglesSHA256:     anglesSHA,
		DenseMaxAbsDeltaTolerance: tolerance,
		DenseInvariantMaxAbsDelta: maxDelta,
		DenseInvariantEligible:    maxDelta <= tolerance,
	}
	hash, err := aoqtCanonicalSHA256(proof)
	if err != nil {
		return AOQTSidecarV8DenseInvariantEndpointProof{}, "", err
	}
	return proof, hash, nil
}

func aoqtV8DenseInvariantProofAggregateSHA(binding AOQTSidecarV8ConstrainedBasisScreenBinding, endpoints []AOQTSidecarV8ConstrainedBasisEndpoint, tolerance float64) (string, error) {
	type endpointProofBinding struct {
		Ordinal                   int     `json:"ordinal"`
		DenseInvariantEligible    bool    `json:"dense_invariant_eligible"`
		DenseInvariantMaxAbsDelta float64 `json:"dense_invariant_max_abs_delta"`
		DenseInvariantProofSHA256 string  `json:"dense_invariant_proof_sha256"`
		ActualDirectionSHA256     string  `json:"actual_direction_sha256"`
		TransactionEligible       bool    `json:"transaction_eligible"`
	}
	proofs := make([]endpointProofBinding, len(endpoints))
	for i, endpoint := range endpoints {
		proofs[i] = endpointProofBinding{
			Ordinal:                   endpoint.Ordinal,
			DenseInvariantEligible:    endpoint.DenseInvariantEligible,
			DenseInvariantMaxAbsDelta: endpoint.DenseInvariantMaxAbsDelta,
			DenseInvariantProofSHA256: endpoint.DenseInvariantProofSHA256,
			ActualDirectionSHA256:     endpoint.ActualDirectionSHA256,
			TransactionEligible:       endpoint.TransactionEligible,
		}
	}
	return aoqtCanonicalSHA256(struct {
		Version                   string                 `json:"version"`
		ManifestSHA256            string                 `json:"manifest_sha256"`
		RowsSHA256                string                 `json:"rows_sha256"`
		RowCount                  int                    `json:"row_count"`
		FoldID                    string                 `json:"fold_id"`
		DenseMaxAbsDeltaTolerance float64                `json:"dense_max_abs_delta_tolerance"`
		EndpointProofs            []endpointProofBinding `json:"endpoint_proofs"`
	}{AOQTV8ConstrainedBasisDenseInvariantProof, binding.IOReport.ManifestSHA256, binding.IOReport.RowsSHA256, binding.IOReport.RowCount, binding.FoldID, tolerance, proofs})
}

func applyAOQTV8DenseProofsToEndpoints(endpoints []AOQTSidecarV8ConstrainedBasisEndpoint, set AOQTSidecarCalibrationSet, binding AOQTSidecarV8ConstrainedBasisScreenBinding, topology AOQTGivensTransform, tolerance float64) (string, error) {
	for i := range endpoints {
		proof, proofSHA, err := aoqtV8DenseInvariantEndpointProof(endpoints[i], set, binding, topology, tolerance)
		if err != nil {
			return "", err
		}
		endpoints[i].DenseInvariantProved = true
		endpoints[i].DenseInvariantEligible = proof.DenseInvariantEligible
		endpoints[i].DenseInvariantMaxAbsDelta = proof.DenseInvariantMaxAbsDelta
		endpoints[i].DenseInvariantProofSHA256 = proofSHA
		if !endpoints[i].AngleMoved {
			endpoints[i].TransactionEligible = false
			endpoints[i].Reason = string(aoqtRejectionNoAngleMovement)
		} else if endpoints[i].TransactionEligible && !endpoints[i].DenseInvariantEligible {
			endpoints[i].TransactionEligible = false
			endpoints[i].Reason = AOQTV8ConstrainedBasisDenseInvariantRejection
		}
	}
	return aoqtV8DenseInvariantProofAggregateSHA(binding, endpoints, tolerance)
}

func validateAOQTV8BoundDenseProofs(r AOQTSidecarV8ConstrainedBasisScreenReceipt, set AOQTSidecarCalibrationSet, binding AOQTSidecarV8ConstrainedBasisScreenBinding, topology AOQTGivensTransform, tolerance float64) error {
	recomputed := append([]AOQTSidecarV8ConstrainedBasisEndpoint(nil), r.Endpoints...)
	for i := range recomputed {
		proof, proofSHA, err := aoqtV8DenseInvariantEndpointProof(recomputed[i], set, binding, topology, tolerance)
		if err != nil {
			return err
		}
		if !recomputed[i].DenseInvariantProved || recomputed[i].DenseInvariantProofSHA256 != proofSHA {
			return fmt.Errorf("AOQT V8 constrained-basis endpoint %d dense invariant proof hash mismatch", recomputed[i].Ordinal)
		}
		if recomputed[i].DenseInvariantMaxAbsDelta != proof.DenseInvariantMaxAbsDelta || recomputed[i].DenseInvariantEligible != proof.DenseInvariantEligible {
			return fmt.Errorf("AOQT V8 constrained-basis endpoint %d dense invariant proof evidence mismatch", recomputed[i].Ordinal)
		}
	}
	aggregate, err := aoqtV8DenseInvariantProofAggregateSHA(binding, recomputed, tolerance)
	if err != nil {
		return err
	}
	if aggregate != r.DenseInvariantProofSHA256 {
		return fmt.Errorf("AOQT V8 constrained-basis dense invariant aggregate proof hash mismatch")
	}
	return nil
}

func rereadAOQTV8PinnedR4Receipt(path, gotSHA string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("AOQT V8 constrained-basis canonical r4 receipt reread: %w", err)
	}
	if sha := sha256BytesAOQT(data); sha != gotSHA || sha != AOQTV7R6CanonicalR4ReceiptFileSHA256 {
		return "", fmt.Errorf("AOQT V8 constrained-basis canonical r4 receipt file sha256 mismatch")
	}
	var receipt AOQTSidecarActualCoordinateProbeReceipt
	if err := strictUnmarshalAOQT(data, &receipt); err != nil {
		return "", fmt.Errorf("AOQT V8 constrained-basis canonical r4 receipt decode: %w", err)
	}
	if err := receipt.ValidateBound(); err != nil {
		return "", fmt.Errorf("AOQT V8 constrained-basis canonical r4 receipt validation: %w", err)
	}
	return receipt.SHA256()
}

func rereadAOQTV8PinnedR5Receipt(path, gotSHA string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("AOQT V8 constrained-basis canonical r5 receipt reread: %w", err)
	}
	if sha := sha256BytesAOQT(data); sha != gotSHA || sha != AOQTV7R6CanonicalR5ReceiptFileSHA256 {
		return "", fmt.Errorf("AOQT V8 constrained-basis canonical r5 receipt file sha256 mismatch")
	}
	var receipt AOQTSidecarActualCoordinateTrainReceipt
	if err := strictUnmarshalAOQT(data, &receipt); err != nil {
		return "", fmt.Errorf("AOQT V8 constrained-basis canonical r5 receipt decode: %w", err)
	}
	if err := receipt.ValidateBound(); err != nil {
		return "", fmt.Errorf("AOQT V8 constrained-basis canonical r5 receipt validation: %w", err)
	}
	return receipt.SHA256()
}

func loadBoundAOQTV8CanonicalV7R6Receipt(path, expectedFileSHA string) (AOQTSidecarV7R6ProgressiveScreenReceipt, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return AOQTSidecarV7R6ProgressiveScreenReceipt{}, fmt.Errorf("AOQT V8 constrained-basis canonical V7-r6 receipt reread: %w", err)
	}
	if got := sha256BytesAOQT(data); got != expectedFileSHA || got != AOQTV8CanonicalV7R6ReceiptFileSHA256 {
		return AOQTSidecarV7R6ProgressiveScreenReceipt{}, fmt.Errorf("AOQT V8 constrained-basis canonical V7-r6 receipt hash mismatch")
	}
	var receipt AOQTSidecarV7R6ProgressiveScreenReceipt
	if err := strictUnmarshalAOQT(data, &receipt); err != nil {
		return AOQTSidecarV7R6ProgressiveScreenReceipt{}, fmt.Errorf("AOQT V8 constrained-basis canonical V7-r6 receipt decode: %w", err)
	}
	if err := receipt.ValidateBound(); err != nil {
		return AOQTSidecarV7R6ProgressiveScreenReceipt{}, fmt.Errorf("AOQT V8 constrained-basis canonical V7-r6 receipt validation: %w", err)
	}
	if receipt.Passed {
		return AOQTSidecarV7R6ProgressiveScreenReceipt{}, fmt.Errorf("AOQT V8 constrained-basis canonical V7-r6 receipt must be terminal negative")
	}
	return receipt, nil
}

func validateAOQTV8V7R6Compatibility(v7r6 AOQTSidecarV7R6ProgressiveScreenReceipt, binding AOQTSidecarV8ConstrainedBasisScreenBinding, fullFoldRowCount int, fullWeightingVersion string, htFactorsApplied bool) error {
	if !reflect.DeepEqual(v7r6.Binding.Inputs, binding.Inputs) ||
		!reflect.DeepEqual(v7r6.Binding.IOReport, binding.IOReport) ||
		v7r6.Binding.PreflightPath != binding.PreflightPath ||
		v7r6.Binding.PreflightSHA256 != binding.PreflightSHA256 ||
		!reflect.DeepEqual(v7r6.Binding.SplitBinding, binding.SplitBinding) ||
		v7r6.Binding.FoldID != binding.FoldID {
		return fmt.Errorf("AOQT V8 constrained-basis canonical V7-r6 input/fold/preflight binding mismatch")
	}
	if v7r6.FullVerification.RowCount != fullFoldRowCount ||
		v7r6.FullVerification.WeightingVersion != fullWeightingVersion ||
		v7r6.FullVerification.HTFactorsApplied != htFactorsApplied {
		return fmt.Errorf("AOQT V8 constrained-basis canonical V7-r6 full-fold authority mismatch")
	}
	if v7r6.Binding.CanonicalR4ReceiptPath != binding.CanonicalR4ReceiptPath ||
		v7r6.Binding.CanonicalR4ReceiptSHA256 != binding.CanonicalR4ReceiptSHA256 ||
		v7r6.Binding.CanonicalR4ReceiptFileSHA256 != binding.CanonicalR4ReceiptFileSHA256 ||
		v7r6.Binding.CanonicalR5ReceiptPath != binding.CanonicalR5ReceiptPath ||
		v7r6.Binding.CanonicalR5ReceiptSHA256 != binding.CanonicalR5ReceiptSHA256 ||
		v7r6.Binding.CanonicalR5ReceiptFileSHA256 != binding.CanonicalR5ReceiptFileSHA256 {
		return fmt.Errorf("AOQT V8 constrained-basis canonical V7-r6 r4/r5 receipt binding mismatch")
	}
	return nil
}

func aoqtCalibrationRowIDs(rows []AOQTSidecarCalibrationRow) []string {
	ids := make([]string, len(rows))
	for i, row := range rows {
		ids[i] = row.RowID
	}
	return ids
}

func RunAOQTV8ConstrainedBasisScreen(cfg AOQTSidecarV8ConstrainedBasisScreenConfig) (AOQTSidecarV8ConstrainedBasisScreenResult, error) {
	result := AOQTSidecarV8ConstrainedBasisScreenResult{}
	if cfg.Mode == "" {
		cfg.Mode = AOQTSidecarOptimizerModeV8ConstrainedBasisScreen
	}
	if cfg.Mode != AOQTSidecarOptimizerModeV8ConstrainedBasisScreen || !cfg.ProbeOnly {
		return result, fmt.Errorf("AOQT V8 constrained-basis requires mode=%s and probe-only=true", AOQTSidecarOptimizerModeV8ConstrainedBasisScreen)
	}
	if cfg.CanonicalV7R6ReceiptFileSHA256 == "" {
		cfg.CanonicalV7R6ReceiptFileSHA256 = AOQTV8CanonicalV7R6ReceiptFileSHA256
	}
	if cfg.CanonicalV7R6ReceiptFileSHA256 != AOQTV8CanonicalV7R6ReceiptFileSHA256 {
		return result, fmt.Errorf("AOQT V8 constrained-basis requires canonical V7-r6 terminal receipt sha256 %s", AOQTV8CanonicalV7R6ReceiptFileSHA256)
	}
	if strings.TrimSpace(cfg.CanonicalV7R6ReceiptPath) == "" {
		return result, fmt.Errorf("AOQT V8 constrained-basis requires canonical V7-r6 terminal receipt path")
	}
	v7r6Receipt, err := loadBoundAOQTV8CanonicalV7R6Receipt(cfg.CanonicalV7R6ReceiptPath, cfg.CanonicalV7R6ReceiptFileSHA256)
	if err != nil {
		return result, err
	}
	v7cfg := AOQTSidecarV7R6ProgressiveScreenConfig{
		Set:                                 cfg.Set,
		ManifestPath:                        cfg.ManifestPath,
		RowsJSONLPath:                       cfg.RowsJSONLPath,
		PreflightJSONPath:                   cfg.PreflightJSONPath,
		SplitManifestPath:                   cfg.SplitManifestPath,
		FoldID:                              cfg.FoldID,
		ExpectedManifestSHA256:              cfg.ExpectedManifestSHA256,
		ExpectedRowsSHA256:                  cfg.ExpectedRowsSHA256,
		ExpectedPreflightSHA256:             cfg.ExpectedPreflightSHA256,
		ExpectedSplitManifestSHA256:         cfg.ExpectedSplitManifestSHA256,
		ExpectedAnchorArtifactSHA256:        cfg.ExpectedAnchorArtifactSHA256,
		ExpectedAnchorPackageManifestSHA256: cfg.ExpectedAnchorPackageManifestSHA256,
		ExpectedAnchorEmbeddingSpaceID:      cfg.ExpectedAnchorEmbeddingSpaceID,
		ExpectedCompatibilityDigest:         cfg.ExpectedCompatibilityDigest,
		ExpectedSourceArtifactHashes:        append([]string(nil), cfg.ExpectedSourceArtifactHashes...),
		ExpectedVectorCacheHashes:           append([]string(nil), cfg.ExpectedVectorCacheHashes...),
		CanonicalR4ReceiptPath:              cfg.CanonicalR4ReceiptPath,
		CanonicalR4ReceiptSHA256:            cfg.CanonicalR4ReceiptSHA256,
		CanonicalR5ReceiptPath:              cfg.CanonicalR5ReceiptPath,
		CanonicalR5ReceiptSHA256:            cfg.CanonicalR5ReceiptSHA256,
		Mode:                                AOQTSidecarOptimizerModeV7R6ProgressiveScreen,
		ProbeOnly:                           true,
		FullVerification:                    true,
	}
	set, ioReport, _, preflightSHA, split, r4, r5, r4FileSHA, r5FileSHA, err := aoqtV7R6ResolveInputs(v7cfg)
	if err != nil {
		return result, err
	}
	if r4 == nil || r5 == nil {
		return result, fmt.Errorf("AOQT V8 constrained-basis requires bound r4 and r5 receipts")
	}
	r4ObjectSHA, err := r4.SHA256()
	if err != nil {
		return result, err
	}
	r5ObjectSHA, err := r5.SHA256()
	if err != nil {
		return result, err
	}
	earlyBinding := AOQTSidecarV8ConstrainedBasisScreenBinding{
		Inputs:                         AOQTSidecarRunMetricInputs{AnchorArtifactSHA256: set.Manifest.AnchorArtifactSHA256, AnchorPackageManifestSHA256: set.Manifest.AnchorPackageManifestSHA256, AnchorEmbeddingSpaceID: set.Manifest.AnchorEmbeddingSpaceID, DatasetManifestSHA256: ioReport.ManifestSHA256, QrelsSHA256ByDataset: cloneAOQTStringMap(set.Manifest.QrelsSHA256ByDataset), CompatibilityDigest: set.Manifest.CompatibilityDigest},
		IOReport:                       ioReport,
		PreflightPath:                  cfg.PreflightJSONPath,
		PreflightSHA256:                preflightSHA,
		SplitBinding:                   cloneAOQTDevSplitBinding(split),
		FoldID:                         split.FoldID,
		CanonicalR4ReceiptPath:         cfg.CanonicalR4ReceiptPath,
		CanonicalR4ReceiptSHA256:       r4ObjectSHA,
		CanonicalR4ReceiptFileSHA256:   r4FileSHA,
		CanonicalR5ReceiptPath:         cfg.CanonicalR5ReceiptPath,
		CanonicalR5ReceiptSHA256:       r5ObjectSHA,
		CanonicalR5ReceiptFileSHA256:   r5FileSHA,
		CanonicalV7R6ReceiptPath:       cfg.CanonicalV7R6ReceiptPath,
		CanonicalV7R6ReceiptFileSHA256: cfg.CanonicalV7R6ReceiptFileSHA256,
	}
	if err := validateAOQTV8V7R6Compatibility(v7r6Receipt, earlyBinding, len(set.Rows), AOQTV8ConstrainedBasisFullWeighting, false); err != nil {
		return result, err
	}
	if err := set.Validate(); err != nil {
		return result, fmt.Errorf("AOQT V8 constrained-basis calibration set: %w", err)
	}
	if set.Manifest.Topology.Dim != AOQTV8ConstrainedBasisExpectedDim {
		return result, fmt.Errorf("AOQT V8 constrained-basis requires D384 topology")
	}
	if split == nil {
		return result, fmt.Errorf("AOQT V8 constrained-basis requires a bound fold/split input")
	}
	if err := aoqtV7R6RejectOfficialOverlap(set.Rows); err != nil {
		return result, fmt.Errorf("%s", strings.Replace(err.Error(), "AOQT V7-r6", "AOQT V8 constrained-basis", 1))
	}
	if len(set.Rows) != AOQTV7R6ProgressiveScreenPopulationTotal {
		return result, fmt.Errorf("AOQT V8 constrained-basis full fold row count = %d, want %d", len(set.Rows), AOQTV7R6ProgressiveScreenPopulationTotal)
	}
	objective, err := objectiveFromAOQTContract(set.Manifest.ObjectiveContract)
	if err != nil {
		return result, err
	}
	policy, err := AOQTSidecarTrainingContractPolicy(set.Manifest.TrainingContract, set.Manifest.CandidateEligibilityPolicy)
	if err != nil {
		return result, err
	}
	trainer, err := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{Dim: set.Manifest.Topology.Dim, Stages: set.Manifest.Topology.Stages, PairingSeed: set.Manifest.Topology.Seed, WorkplanSeed: 1, OptimizerMode: AOQTSidecarOptimizerModeV7R4DevActualCoordinate, ActualCoordinateProbeRequired: true, ActualCoordinateProbeOnly: true, ActualCoordinateProbeFoldID: split.FoldID, ActualCoordinateProbeSplitManifestSHA256: split.SplitManifestSHA256, LearningRate: AOQTV7R4ActualCoordinateProbeLearningRate, AngleCap: AOQTSidecarDefaultAngleCap, MaxAngleCap: AOQTSidecarHardMaxAngleCap, TrainingContract: set.Manifest.TrainingContract, CandidateEligibilityPolicy: cloneAOQTSidecarCandidateEligibilityPolicy(set.Manifest.CandidateEligibilityPolicy)})
	if err != nil {
		return result, err
	}
	if _, err := trainer.Plan(set); err != nil {
		return result, err
	}
	rows := append([]AOQTSidecarCalibrationRow(nil), set.Rows...)
	baselineLoss, _, baselineActivation, baselineComponents, err := trainer.lossAndAngleGrad(rows, objective)
	if err != nil {
		return result, err
	}
	q3Grad, err := trainer.q3GainOnlyAngleGrad(rows, objective)
	if err != nil {
		return result, err
	}
	protected, err := trainer.protectedComponentAngleGrads(rows, objective, set.Manifest.ObjectiveContract.WeightSums)
	if err != nil {
		return result, err
	}
	basis, q3SHA, protectedNames, protectedSHA, basisSHA, err := aoqtV8ConstructBasis(q3Grad, protected, AOQTV8ConstrainedBasisRanks())
	if err != nil {
		return result, err
	}
	directionScheduleSHA, err := aoqtCanonicalSHA256(struct {
		Version    string    `json:"version"`
		Ranks      []int     `json:"ranks"`
		Magnitudes []float32 `json:"magnitudes"`
		Signs      []int     `json:"signs"`
		Cap        int       `json:"cap"`
	}{AOQTV8ConstrainedBasisScheduleVersion, AOQTV8ConstrainedBasisRanks(), AOQTV8ConstrainedBasisMagnitudes(), AOQTV8ConstrainedBasisGlobalSigns(), AOQTV8ConstrainedBasisCandidateCap})
	if err != nil {
		return result, err
	}
	endpoints := make([]AOQTSidecarV8ConstrainedBasisEndpoint, 0, AOQTV8ConstrainedBasisCandidateCap)
	baseline := aoqtStepEvaluation{loss: baselineLoss, activation: baselineActivation, components: baselineComponents}
	schedule, err := aoqtV8CanonicalEndpointSchedule()
	if err != nil {
		return result, err
	}
	for _, item := range schedule {
		endpoint := trainer.evaluateV8ConstrainedBasisEndpoint(rows, objective, baseline, set.Manifest.ObjectiveContract.WeightSums, basis[item.BasisOrdinal-1], item.Ordinal, item.Magnitude, item.GlobalSign)
		endpoints = append(endpoints, endpoint)
	}
	denseProofSHA, err := applyAOQTV8DenseProofsToEndpoints(endpoints, set, earlyBinding, trainer.topology, policy.DenseMaxAbsDeltaTolerance)
	if err != nil {
		return result, err
	}
	ranking := aoqtV8RankEndpoints(endpoints, policy)
	rankAt := map[int]int{}
	for i, endpointOrdinal := range ranking {
		rankAt[endpointOrdinal] = i + 1
	}
	feasible := 0
	selected := make([]int, 0, len(endpoints))
	for i := range endpoints {
		endpoints[i].RankOrder = rankAt[endpoints[i].Ordinal]
		if endpoints[i].TransactionEligible {
			feasible++
			selected = append(selected, endpoints[i].Ordinal)
		}
	}
	chainData, chainTail, err := aoqtV8EndpointHashChain(endpoints)
	if err != nil {
		return result, err
	}
	contractSHA, err := aoqtActualCoordinateObjectiveContractSHA256(set.Manifest.ObjectiveContract)
	if err != nil {
		return result, err
	}
	policySHA, err := aoqtActualCoordinatePolicySHA256(policy)
	if err != nil {
		return result, err
	}
	selectionSeed, err := aoqtV8SelectionSeed(set, split.FoldID, q3SHA, protectedSHA, basisSHA, cfg.CanonicalV7R6ReceiptFileSHA256)
	if err != nil {
		return result, err
	}
	receipt := AOQTSidecarV8ConstrainedBasisScreenReceipt{
		Schema:                            AOQTSidecarV8ConstrainedBasisScreenSchema,
		Mode:                              cfg.Mode,
		Required:                          true,
		ProbeOnly:                         true,
		Passed:                            feasible > 0,
		BasisVersion:                      AOQTV8ConstrainedBasisVersion,
		ScheduleVersion:                   AOQTV8ConstrainedBasisScheduleVersion,
		RankingVersion:                    AOQTV8ConstrainedBasisRankingVersion,
		PolicyVersion:                     AOQTV8ConstrainedBasisPolicyVersion,
		SelectionSeedSHA256:               selectionSeed,
		CoordinateCount:                   len(trainer.angles),
		Dim:                               set.Manifest.Topology.Dim,
		Ranks:                             AOQTV8ConstrainedBasisRanks(),
		Magnitudes:                        AOQTV8ConstrainedBasisMagnitudes(),
		GlobalSigns:                       AOQTV8ConstrainedBasisGlobalSigns(),
		CandidateCap:                      AOQTV8ConstrainedBasisCandidateCap,
		ObjectiveEvaluationCount:          1 + len(endpoints),
		FullFoldRowCount:                  len(rows),
		FullFoldWeightingVersion:          AOQTV8ConstrainedBasisFullWeighting,
		HTFactorsApplied:                  false,
		BaselineLoss:                      baselineLoss,
		BaselineLossFinite:                isFinite32(baselineLoss),
		BaselineComponents:                baselineComponents,
		BaselineActivation:                baselineActivation,
		Q3GradientSHA256:                  q3SHA,
		ProtectedGradientNames:            protectedNames,
		ProtectedGradientMatrixSHA256:     protectedSHA,
		BasisSHA256:                       basisSHA,
		DirectionScheduleSHA256:           directionScheduleSHA,
		CanonicalObjectiveContractSHA256:  contractSHA,
		CanonicalEligibilityPolicySHA256:  policySHA,
		CanonicalV7R6ReceiptFileSHA256:    cfg.CanonicalV7R6ReceiptFileSHA256,
		CanonicalV7R6TerminalReceiptBound: true,
		DenseInvariantStatus:              AOQTV8ConstrainedBasisDenseInvariantProof,
		DenseInvariantProofSHA256:         denseProofSHA,
		DenseMaxAbsDeltaTolerance:         policy.DenseMaxAbsDeltaTolerance,
		DenseInvariantDeferred:            false,
		GradientEvidenceDeferred:          false,
		FullVerificationCompleted:         true,
		FeasibleEndpointCount:             feasible,
		SelectedEndpointOrdinals:          selected,
		EndpointHashChain:                 string(chainData),
		EndpointHashChainTailSHA256:       chainTail,
		Basis:                             basis,
		Endpoints:                         endpoints,
		Binding:                           earlyBinding,
	}
	if !receipt.Passed {
		receipt.FailureReason = "no dense-invariant-safe V8 endpoint satisfied the transaction policy"
	}
	if cfg.PreflightJSONPath != "" || cfg.CanonicalR4ReceiptPath != "" || cfg.CanonicalR5ReceiptPath != "" || cfg.CanonicalV7R6ReceiptPath != "" {
		if err := receipt.ValidateBound(); err != nil {
			return result, err
		}
	} else if err := receipt.Validate(); err != nil {
		return result, err
	}
	if strings.TrimSpace(cfg.OutputReceiptJSONPath) != "" {
		if err := WriteAOQTV8ConstrainedBasisScreenReceipt(cfg.OutputReceiptJSONPath, receipt); err != nil {
			return result, err
		}
		result.ReceiptJSONPath = cfg.OutputReceiptJSONPath
	}
	receiptSHA, err := receipt.SHA256()
	if err != nil {
		return result, err
	}
	result.Receipt, result.ReceiptSHA256 = receipt, receiptSHA
	return result, nil
}

func (t *AOQTSidecarTrainer) evaluateV8ConstrainedBasisEndpoint(rows []AOQTSidecarCalibrationRow, objective AOQTSidecarVectorObjective, baseline aoqtStepEvaluation, weights AOQTSidecarRowWeights, basis AOQTSidecarV8BasisVector, ordinal int, magnitude float32, globalSign int) AOQTSidecarV8ConstrainedBasisEndpoint {
	endpoint := AOQTSidecarV8ConstrainedBasisEndpoint{
		Ordinal: ordinal, BasisOrdinal: basis.Ordinal, Rank: basis.Rank, Magnitude: magnitude, GlobalSign: globalSign,
		BaselineLoss: baseline.loss, BaselineLossFinite: isFinite32(baseline.loss), BaselineComponents: baseline.components,
		BaselineComponentsFinite: aoqtFiniteReceiptComponentsValue(baseline.components), ActualDirection: make([]float32, len(t.angles)),
	}
	requested := make([]float32, len(t.angles))
	for i, value := range basis.Direction {
		requested[i] = value * magnitude * float32(globalSign)
	}
	endpoint.RequestedDirectionSHA256, _ = aoqtActualDirectionVectorSHA256(requested)
	state := t.snapshotOptimizerState()
	defer t.restoreOptimizerState(state)
	for i, delta := range requested {
		if delta != 0 {
			t.angles[i] += delta
		}
	}
	if err := t.ProjectAngles(); err != nil {
		endpoint.CandidateError = err.Error()
		endpoint.Reason = "probe_error"
		return endpoint
	}
	for i := range endpoint.ActualDirection {
		endpoint.ActualDirection[i] = t.angles[i] - state.angles[i]
	}
	endpoint.ActualDirectionMaxAbs = aoqtMaxAbsFloat32(endpoint.ActualDirection)
	endpoint.ActualDirectionSHA256, _ = aoqtActualDirectionVectorSHA256(endpoint.ActualDirection)
	endpoint.AngleMoved = endpoint.ActualDirectionMaxAbs != 0
	candidateLoss, _, activation, components, err := t.lossAndAngleGrad(rows, objective)
	if err != nil {
		endpoint.CandidateError = err.Error()
		endpoint.Reason = "nonfinite_loss"
		return endpoint
	}
	endpoint.CandidateLoss = candidateLoss
	endpoint.CandidateLossFinite = isFinite32(candidateLoss)
	endpoint.CandidateComponents = components
	endpoint.CandidateComponentsFinite = aoqtFiniteReceiptComponentsValue(components)
	endpoint.CandidateActivation = activation
	if endpoint.BaselineLossFinite && endpoint.CandidateLossFinite {
		endpoint.TotalDelta = candidateLoss - baseline.loss
		endpoint.TotalDeltaFinite = isFinite32(endpoint.TotalDelta)
	}
	if endpoint.BaselineComponentsFinite && endpoint.CandidateComponentsFinite {
		endpoint.ComponentDeltas = aoqtObjectiveComponentDelta(baseline.components, components)
		endpoint.ComponentDeltasFinite = aoqtFiniteReceiptComponentsValue(endpoint.ComponentDeltas)
	}
	endpoint.Q3GainEligible = endpoint.ComponentDeltasFinite && components.Q3Gain < baseline.components.Q3Gain-aoqtTransactionalQ3ImprovementMinMagnitude
	decision := aoqtEvaluateTransactionalStepWithPolicy(baseline, aoqtStepEvaluation{loss: candidateLoss, activation: activation, components: components}, weights, t.eligibilityPolicy)
	endpoint.ProtectedEligible = len(aoqtComponentRegressions(baseline.components, components, t.eligibilityPolicy)) == 0
	if !endpoint.AngleMoved {
		endpoint.Reason = string(aoqtRejectionNoAngleMovement)
	} else if decision.accepted {
		endpoint.TransactionEligible = true
		endpoint.Reason = aoqtProposalAcceptedReason
	} else {
		endpoint.Reason = string(decision.reason)
	}
	return endpoint
}

func aoqtV8ConstructBasis(q3Grad []float32, protected []aoqtProtectedAngleGradient, ranks []int) ([]AOQTSidecarV8BasisVector, string, []string, string, string, error) {
	if len(q3Grad) != AOQTV8ConstrainedBasisExpectedAngleSize {
		return nil, "", nil, "", "", fmt.Errorf("AOQT V8 q3 gradient length = %d, want %d", len(q3Grad), AOQTV8ConstrainedBasisExpectedAngleSize)
	}
	q3SHA, err := aoqtCanonicalSHA256(q3Grad)
	if err != nil {
		return nil, "", nil, "", "", err
	}
	protectedNames := aoqtProtectedGradientNames(protected)
	protectedSHA, err := aoqtCanonicalSHA256(struct {
		Names     []string                     `json:"names"`
		Gradients []aoqtProtectedAngleGradient `json:"gradients"`
	}{protectedNames, protected})
	if err != nil {
		return nil, "", nil, "", "", err
	}
	residual := make([]float32, len(q3Grad))
	for i, v := range q3Grad {
		residual[i] = -v
	}
	for _, guard := range protected {
		residual = aoqtV8ProjectAway(residual, guard.Grad)
	}
	order := make([]int, len(residual))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool {
		ai, aj := float32(math.Abs(float64(residual[order[i]]))), float32(math.Abs(float64(residual[order[j]])))
		if ai != aj {
			return ai > aj
		}
		return order[i] < order[j]
	})
	basis := make([]AOQTSidecarV8BasisVector, 0, len(ranks))
	for i, rank := range ranks {
		if rank <= 0 || rank > len(order) {
			return nil, "", nil, "", "", fmt.Errorf("AOQT V8 invalid basis rank %d", rank)
		}
		direction := make([]float32, len(residual))
		selected := make([]int, 0, rank)
		maxAbs := float32(0)
		for _, index := range order {
			if residual[index] == 0 {
				continue
			}
			selected = append(selected, index)
			abs := float32(math.Abs(float64(residual[index])))
			if abs > maxAbs {
				maxAbs = abs
			}
			if len(selected) == rank {
				break
			}
		}
		if len(selected) != rank || maxAbs == 0 {
			return nil, "", nil, "", "", fmt.Errorf("AOQT V8 cannot construct nonzero rank-%d basis", rank)
		}
		signs := make([]int, 0, len(selected))
		for _, index := range selected {
			direction[index] = residual[index] / maxAbs
			if direction[index] >= 0 {
				signs = append(signs, 1)
			} else {
				signs = append(signs, -1)
			}
		}
		dirSHA, err := aoqtActualDirectionVectorSHA256(direction)
		if err != nil {
			return nil, "", nil, "", "", err
		}
		basis = append(basis, AOQTSidecarV8BasisVector{Ordinal: i + 1, Kind: "sequential_projected_sparse_q3_support", Rank: rank, Direction: direction, DirectionSHA256: dirSHA, SelectedIndices: selected, SelectedSigns: signs, SourceGradientSHA256: q3SHA})
	}
	basisSHA, err := aoqtCanonicalSHA256(basis)
	if err != nil {
		return nil, "", nil, "", "", err
	}
	return basis, q3SHA, protectedNames, protectedSHA, basisSHA, nil
}

func aoqtV8ProjectAway(vector, guard []float32) []float32 {
	if len(vector) != len(guard) {
		return append([]float32(nil), vector...)
	}
	var dot, norm float64
	for i := range vector {
		dot += float64(vector[i] * guard[i])
		norm += float64(guard[i] * guard[i])
	}
	if norm == 0 {
		return append([]float32(nil), vector...)
	}
	scale := dot / norm
	out := make([]float32, len(vector))
	for i := range vector {
		out[i] = vector[i] - float32(scale)*guard[i]
	}
	return out
}

func aoqtV8RankEndpoints(endpoints []AOQTSidecarV8ConstrainedBasisEndpoint, policy AOQTSidecarCandidateEligibilityPolicy) []int {
	order := make([]int, len(endpoints))
	for i := range endpoints {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool {
		a, b := endpoints[order[i]], endpoints[order[j]]
		if a.TransactionEligible != b.TransactionEligible {
			return a.TransactionEligible
		}
		if a.TotalDelta != b.TotalDelta {
			return a.TotalDelta < b.TotalDelta
		}
		if a.ComponentDeltas.Q3Gain != b.ComponentDeltas.Q3Gain {
			return a.ComponentDeltas.Q3Gain < b.ComponentDeltas.Q3Gain
		}
		return a.Ordinal < b.Ordinal
	})
	ranking := make([]int, len(order))
	for i, index := range order {
		ranking[i] = endpoints[index].Ordinal
	}
	_ = policy
	return ranking
}

func aoqtV8EndpointHashChain(endpoints []AOQTSidecarV8ConstrainedBasisEndpoint) ([]byte, string, error) {
	chain := make([]string, 0, len(endpoints))
	previous := ""
	for _, endpoint := range endpoints {
		payload, err := json.Marshal(endpoint)
		if err != nil {
			return nil, "", err
		}
		previous = sha256AOQTCoordinateSearchChainEntry(previous, string(payload))
		chain = append(chain, previous)
	}
	data, err := json.Marshal(chain)
	return data, previous, err
}

func aoqtV8SelectionSeed(set AOQTSidecarCalibrationSet, foldID, q3SHA, protectedSHA, basisSHA, v7r6SHA string) (string, error) {
	manifestSHA, err := AOQTSidecarManifestSHA256(set.Manifest)
	if err != nil {
		return "", err
	}
	payload := strings.Join([]string{AOQTV8ConstrainedBasisSeed, foldID, manifestSHA, set.Manifest.RowIDSHA256, q3SHA, protectedSHA, basisSHA, v7r6SHA}, "\x00")
	sum := sha256.Sum256([]byte(payload))
	return hex.EncodeToString(sum[:]), nil
}

func containsFloat32AOQT(value float32, values []float32) bool {
	for _, candidate := range values {
		if candidate == value {
			return true
		}
	}
	return false
}

func WriteAOQTV8ConstrainedBasisScreenReceipt(path string, receipt AOQTSidecarV8ConstrainedBasisScreenReceipt) error {
	if receipt.Binding.IOReport.ManifestPath != "" || receipt.Binding.IOReport.RowsJSONLPath != "" || receipt.Binding.PreflightPath != "" ||
		receipt.Binding.CanonicalR4ReceiptPath != "" || receipt.Binding.CanonicalR5ReceiptPath != "" || receipt.Binding.CanonicalV7R6ReceiptPath != "" {
		if err := receipt.ValidateBound(); err != nil {
			return err
		}
	} else {
		if err := receipt.Validate(); err != nil {
			return err
		}
	}
	if strings.TrimSpace(path) == "" {
		return fmt.Errorf("AOQT V8 constrained-basis receipt output path is required")
	}
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		if os.IsExist(err) {
			return fmt.Errorf("AOQT V8 constrained-basis receipt output %q already exists", path)
		}
		return err
	}
	ok := false
	defer func() {
		_ = f.Close()
		if !ok {
			_ = os.Remove(path)
		}
	}()
	if _, err := f.Write(data); err != nil {
		return err
	}
	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	ok = true
	return nil
}
