package eosruntime

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
	AOQTSidecarV9ActiveSetNullspaceScreenSchema        = "eos.aoqt.v9_active_set_nullspace_screen.v1"
	AOQTSidecarOptimizerModeV9ActiveSetNullspaceScreen = "aoqt-v9-active-set-nullspace-v1"
	AOQTV9CanonicalV8ReceiptFileSHA256                 = "4bb0cdf9e756340b3c308336313034104a188bb16dc7b32394ceeaa51769d07b"

	AOQTV9ActiveSetNullspaceSeed                   = "aoqt-v9-active-set-nullspace-seed-v1"
	AOQTV9ActiveSetNullspaceScheduleVersion        = "aoqt-v9-ranked-active-set-directions-x-four-magnitudes-v1"
	AOQTV9ActiveSetNullspaceRankingVersion         = AOQTV8ConstrainedBasisRankingVersion
	AOQTV9ActiveSetNullspacePolicyVersion          = AOQTV8ConstrainedBasisPolicyVersion
	AOQTV9ActiveSetNullspaceFullWeighting          = AOQTV8ConstrainedBasisFullWeighting
	AOQTV9ActiveSetNullspaceDenseProof             = "proved_by_endpoint_dense_max_abs_delta_v1"
	AOQTV9ActiveSetNullspaceDenseProofSchema       = "eos.aoqt.v9_dense_invariant_endpoint_proof.v1"
	AOQTV9ActiveSetNullspaceDenseRejection         = AOQTV8ConstrainedBasisDenseInvariantRejection
	AOQTV9ActiveSetNullspaceDerivativeSafetyReject = "derivative_safety"
	AOQTV9ActiveSetNullspaceDirectionCap           = 6
	AOQTV9ActiveSetNullspaceCandidateCap           = 24
	AOQTV9ActiveSetNullspaceObjectiveEvalMax       = 25
	AOQTV9ActiveSetNullspaceExpectedDim            = AOQTV8ConstrainedBasisExpectedDim
	AOQTV9ActiveSetNullspaceExpectedAngleSize      = AOQTV8ConstrainedBasisExpectedAngleSize
)

func AOQTV9ActiveSetNullspaceMagnitudes() []float32 {
	values := AOQTV9NullspaceMagnitudes()
	out := make([]float32, len(values))
	for i, value := range values {
		out[i] = float32(value)
	}
	return out
}

type AOQTSidecarV9ActiveSetNullspaceScreenConfig struct {
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

	CanonicalV8ReceiptPath       string
	CanonicalV8ReceiptFileSHA256 string

	Mode      string
	ProbeOnly bool
}

type AOQTSidecarV9ActiveSetNullspaceScreenBinding struct {
	Inputs                       AOQTSidecarRunMetricInputs     `json:"inputs"`
	IOReport                     AOQTSidecarCalibrationIOReport `json:"io_report"`
	PreflightPath                string                         `json:"preflight_path"`
	PreflightSHA256              string                         `json:"preflight_sha256"`
	SplitBinding                 *AOQTSidecarDevSplitBinding    `json:"split_binding"`
	FoldID                       string                         `json:"fold_id"`
	CanonicalR4ReceiptPath       string                         `json:"canonical_r4_receipt_path"`
	CanonicalR4ReceiptSHA256     string                         `json:"canonical_r4_receipt_sha256"`
	CanonicalR4ReceiptFileSHA256 string                         `json:"canonical_r4_receipt_file_sha256"`
	CanonicalR5ReceiptPath       string                         `json:"canonical_r5_receipt_path"`
	CanonicalR5ReceiptSHA256     string                         `json:"canonical_r5_receipt_sha256"`
	CanonicalR5ReceiptFileSHA256 string                         `json:"canonical_r5_receipt_file_sha256"`
	CanonicalV8ReceiptPath       string                         `json:"canonical_v8_receipt_path"`
	CanonicalV8ReceiptFileSHA256 string                         `json:"canonical_v8_receipt_file_sha256"`
}

type AOQTSidecarV9DirectionEvidence struct {
	Schema                   string                     `json:"schema"`
	Mode                     string                     `json:"mode"`
	BasisVersion             string                     `json:"basis_version"`
	ConeDerivativeVersion    string                     `json:"cone_derivative_version"`
	CanonicalV8ReceiptSHA256 string                     `json:"canonical_v8_receipt_sha256"`
	DirectionCap             int                        `json:"direction_cap"`
	CandidateCap             int                        `json:"candidate_cap"`
	ObjectiveEvaluationMax   int                        `json:"objective_evaluation_max"`
	Magnitudes               []float64                  `json:"magnitudes"`
	Candidates               []AOQTV9NullspaceCandidate `json:"candidates"`
}

type AOQTSidecarV9ActiveSetNullspaceEndpoint struct {
	Ordinal                    int                            `json:"ordinal"`
	DirectionRankOrder         int                            `json:"direction_rank_order"`
	DirectionOrdinal           int                            `json:"direction_ordinal"`
	Magnitude                  float32                        `json:"magnitude"`
	RequestedDirectionSHA256   string                         `json:"requested_direction_sha256"`
	ActualDirectionSHA256      string                         `json:"actual_direction_sha256"`
	ActualDirectionMaxAbs      float32                        `json:"actual_direction_max_abs"`
	ActualDirection            []float32                      `json:"actual_direction"`
	ActualQ3Derivative         float64                        `json:"actual_q3_derivative"`
	ActualProtectedDerivatives []float64                      `json:"actual_protected_derivatives"`
	DerivativeEligible         bool                           `json:"derivative_eligible"`
	AngleMoved                 bool                           `json:"angle_moved"`
	BaselineLoss               float32                        `json:"baseline_loss"`
	BaselineLossFinite         bool                           `json:"baseline_loss_finite"`
	CandidateLoss              float32                        `json:"candidate_loss"`
	CandidateLossFinite        bool                           `json:"candidate_loss_finite"`
	BaselineComponents         AOQTSidecarObjectiveComponents `json:"baseline_components"`
	BaselineComponentsFinite   bool                           `json:"baseline_components_finite"`
	CandidateComponents        AOQTSidecarObjectiveComponents `json:"candidate_components"`
	CandidateComponentsFinite  bool                           `json:"candidate_components_finite"`
	ComponentDeltas            AOQTSidecarObjectiveComponents `json:"component_deltas"`
	ComponentDeltasFinite      bool                           `json:"component_deltas_finite"`
	CandidateActivation        AOQTSidecarObjectiveActivation `json:"candidate_activation"`
	TotalDelta                 float32                        `json:"total_delta"`
	TotalDeltaFinite           bool                           `json:"total_delta_finite"`
	Q3GainEligible             bool                           `json:"q3_gain_eligible"`
	ProtectedEligible          bool                           `json:"protected_eligible"`
	TransactionEligible        bool                           `json:"transaction_eligible"`
	Reason                     string                         `json:"reason"`
	DenseInvariantProved       bool                           `json:"dense_invariant_proved"`
	DenseInvariantEligible     bool                           `json:"dense_invariant_eligible"`
	DenseInvariantMaxAbsDelta  float64                        `json:"dense_invariant_max_abs_delta"`
	DenseInvariantProofSHA256  string                         `json:"dense_invariant_proof_sha256,omitempty"`
	CandidateError             string                         `json:"candidate_error,omitempty"`
	RankOrder                  int                            `json:"rank_order"`
}

type AOQTSidecarV9DenseInvariantEndpointProof struct {
	Schema                    string                                     `json:"schema"`
	Version                   string                                     `json:"version"`
	EndpointOrdinal           int                                        `json:"endpoint_ordinal"`
	EndpointEvidence          AOQTSidecarV9DenseInvariantEndpointSummary `json:"endpoint_evidence"`
	ManifestSHA256            string                                     `json:"manifest_sha256"`
	RowsSHA256                string                                     `json:"rows_sha256"`
	RowIDSHA256               string                                     `json:"row_id_sha256"`
	RowCount                  int                                        `json:"row_count"`
	FoldID                    string                                     `json:"fold_id"`
	FullFoldWeightingVersion  string                                     `json:"full_fold_weighting_version"`
	HTFactorsApplied          bool                                       `json:"ht_factors_applied"`
	CandidatePairingsSHA256   string                                     `json:"candidate_pairings_sha256"`
	CandidateAnglesSHA256     string                                     `json:"candidate_angles_sha256"`
	DenseMaxAbsDeltaTolerance float64                                    `json:"dense_max_abs_delta_tolerance"`
	DenseInvariantMaxAbsDelta float64                                    `json:"dense_invariant_max_abs_delta"`
	DenseInvariantEligible    bool                                       `json:"dense_invariant_eligible"`
}

type AOQTSidecarV9DenseInvariantEndpointSummary struct {
	Ordinal                  int     `json:"ordinal"`
	DirectionRankOrder       int     `json:"direction_rank_order"`
	DirectionOrdinal         int     `json:"direction_ordinal"`
	Magnitude                float32 `json:"magnitude"`
	RequestedDirectionSHA256 string  `json:"requested_direction_sha256"`
	ActualDirectionSHA256    string  `json:"actual_direction_sha256"`
	ActualDirectionMaxAbs    float32 `json:"actual_direction_max_abs"`
	AngleMoved               bool    `json:"angle_moved"`
	BaselineLoss             float32 `json:"baseline_loss"`
	CandidateLoss            float32 `json:"candidate_loss"`
}

type AOQTSidecarV9ActiveSetNullspaceScreenReceipt struct {
	Schema                           string                                       `json:"schema"`
	Mode                             string                                       `json:"mode"`
	Required                         bool                                         `json:"required"`
	ProbeOnly                        bool                                         `json:"probe_only"`
	Passed                           bool                                         `json:"passed"`
	FailureReason                    string                                       `json:"failure_reason,omitempty"`
	BasisVersion                     string                                       `json:"basis_version"`
	ConeDerivativeVersion            string                                       `json:"cone_derivative_version"`
	ScheduleVersion                  string                                       `json:"schedule_version"`
	RankingVersion                   string                                       `json:"ranking_version"`
	PolicyVersion                    string                                       `json:"policy_version"`
	SelectionSeedSHA256              string                                       `json:"selection_seed_sha256"`
	CoordinateCount                  int                                          `json:"coordinate_count"`
	Dim                              int                                          `json:"dim"`
	DirectionCap                     int                                          `json:"direction_cap"`
	Magnitudes                       []float32                                    `json:"magnitudes"`
	CandidateCap                     int                                          `json:"candidate_cap"`
	ObjectiveEvaluationCount         int                                          `json:"objective_evaluation_count"`
	FullFoldRowCount                 int                                          `json:"full_fold_row_count"`
	FullFoldWeightingVersion         string                                       `json:"full_fold_weighting_version"`
	HTFactorsApplied                 bool                                         `json:"ht_factors_applied"`
	BaselineLoss                     float32                                      `json:"baseline_loss"`
	BaselineLossFinite               bool                                         `json:"baseline_loss_finite"`
	BaselineComponents               AOQTSidecarObjectiveComponents               `json:"baseline_components"`
	BaselineActivation               AOQTSidecarObjectiveActivation               `json:"baseline_activation"`
	ObjectiveWeightSums              AOQTSidecarRowWeights                        `json:"objective_weight_sums"`
	Q3GradientSHA256                 string                                       `json:"q3_gradient_sha256"`
	ProtectedGradientNames           []string                                     `json:"protected_gradient_names"`
	ProtectedGradientMatrixSHA256    string                                       `json:"protected_gradient_matrix_sha256"`
	DirectionEvidenceSHA256          string                                       `json:"direction_evidence_sha256"`
	DirectionScheduleSHA256          string                                       `json:"direction_schedule_sha256"`
	CanonicalObjectiveContractSHA256 string                                       `json:"canonical_objective_contract_sha256"`
	CanonicalEligibilityPolicySHA256 string                                       `json:"canonical_eligibility_policy_sha256"`
	CanonicalV8ReceiptFileSHA256     string                                       `json:"canonical_v8_receipt_file_sha256"`
	CanonicalV8TerminalReceiptBound  bool                                         `json:"canonical_v8_terminal_receipt_bound"`
	DenseInvariantStatus             string                                       `json:"dense_invariant_status"`
	DenseInvariantProofSHA256        string                                       `json:"dense_invariant_proof_sha256"`
	DenseMaxAbsDeltaTolerance        float64                                      `json:"dense_max_abs_delta_tolerance"`
	GradientEvidenceDeferred         bool                                         `json:"gradient_evidence_deferred"`
	FullVerificationCompleted        bool                                         `json:"full_verification_completed"`
	FeasibleEndpointCount            int                                          `json:"feasible_endpoint_count"`
	SelectedEndpointOrdinals         []int                                        `json:"selected_endpoint_ordinals"`
	EndpointHashChain                string                                       `json:"endpoint_hash_chain"`
	EndpointHashChainTailSHA256      string                                       `json:"endpoint_hash_chain_tail_sha256"`
	DirectionEvidence                AOQTSidecarV9DirectionEvidence               `json:"direction_evidence"`
	Endpoints                        []AOQTSidecarV9ActiveSetNullspaceEndpoint    `json:"endpoints"`
	Binding                          AOQTSidecarV9ActiveSetNullspaceScreenBinding `json:"binding"`
	QualityClaim                     bool                                         `json:"quality_claim"`
	ReleaseClaim                     bool                                         `json:"release_claim"`
	OfficialClaim                    bool                                         `json:"official_claim"`
	OfficialHeldoutGate              bool                                         `json:"official_heldout_gate"`
	CommercialClaim                  bool                                         `json:"commercial_claim"`
	FreeOpenUseClaim                 bool                                         `json:"free_open_use_claim"`
}

type AOQTSidecarV9ActiveSetNullspaceScreenResult struct {
	Receipt         AOQTSidecarV9ActiveSetNullspaceScreenReceipt
	ReceiptSHA256   string
	ReceiptJSONPath string
}

func (e AOQTSidecarV9DirectionEvidence) Validate() error {
	if e.Schema != AOQTSidecarV9ActiveSetNullspaceScreenSchema {
		return fmt.Errorf("AOQT V9 active-set nullspace schema %q is unsupported", e.Schema)
	}
	if e.Mode != AOQTSidecarOptimizerModeV9ActiveSetNullspaceScreen {
		return fmt.Errorf("AOQT V9 active-set nullspace mode %q is unsupported", e.Mode)
	}
	if e.BasisVersion != AOQTV9NullspaceBasisVersion || e.ConeDerivativeVersion != AOQTV9ConeDerivativeVersion {
		return fmt.Errorf("AOQT V9 active-set nullspace math version binding is unsupported")
	}
	if e.CanonicalV8ReceiptSHA256 != AOQTV9CanonicalV8ReceiptFileSHA256 {
		return fmt.Errorf("AOQT V9 active-set nullspace canonical V8 receipt is not the pinned authorization")
	}
	if e.DirectionCap != AOQTV9ActiveSetNullspaceDirectionCap || e.CandidateCap != AOQTV9ActiveSetNullspaceCandidateCap || e.ObjectiveEvaluationMax != AOQTV9ActiveSetNullspaceObjectiveEvalMax {
		return fmt.Errorf("AOQT V9 active-set nullspace cap binding is unsupported")
	}
	if !float64SlicesEqualAOQTV9(e.Magnitudes, AOQTV9NullspaceMagnitudes()) {
		return fmt.Errorf("AOQT V9 active-set nullspace magnitudes are not canonical")
	}
	if len(e.Candidates) == 0 || len(e.Candidates) > AOQTV9ActiveSetNullspaceDirectionCap {
		return fmt.Errorf("AOQT V9 active-set nullspace candidate direction count is out of bounds")
	}
	for i, candidate := range e.Candidates {
		if err := validateAOQTV9NullspaceCandidate(candidate, i); err != nil {
			return err
		}
	}
	return nil
}

func (r AOQTSidecarV9ActiveSetNullspaceScreenReceipt) SHA256() (string, error) {
	return aoqtCanonicalSHA256(r)
}

func (r AOQTSidecarV9ActiveSetNullspaceScreenReceipt) Validate() error {
	return r.validate(false)
}

func (r AOQTSidecarV9ActiveSetNullspaceScreenReceipt) ValidateBound() error {
	if r.CanonicalV8ReceiptFileSHA256 != AOQTV9CanonicalV8ReceiptFileSHA256 || r.Binding.CanonicalV8ReceiptFileSHA256 != AOQTV9CanonicalV8ReceiptFileSHA256 {
		return fmt.Errorf("AOQT V9 active-set nullspace canonical V8 receipt is not the pinned authorization")
	}
	if err := r.validate(true); err != nil {
		return err
	}
	return rereadAOQTV9ActiveSetNullspaceBinding(r)
}

func (r AOQTSidecarV9ActiveSetNullspaceScreenReceipt) validate(requireBinding bool) error {
	if r.Schema != AOQTSidecarV9ActiveSetNullspaceScreenSchema || r.Mode != AOQTSidecarOptimizerModeV9ActiveSetNullspaceScreen || !r.Required || !r.ProbeOnly {
		return fmt.Errorf("AOQT V9 active-set nullspace receipt must use the required probe-only V9 schema and mode")
	}
	if r.QualityClaim || r.ReleaseClaim || r.OfficialClaim || r.OfficialHeldoutGate || r.CommercialClaim || r.FreeOpenUseClaim {
		return fmt.Errorf("AOQT V9 active-set nullspace claims and official heldout gate must be false")
	}
	if r.BasisVersion != AOQTV9NullspaceBasisVersion || r.ConeDerivativeVersion != AOQTV9ConeDerivativeVersion || r.ScheduleVersion != AOQTV9ActiveSetNullspaceScheduleVersion || r.RankingVersion != AOQTV9ActiveSetNullspaceRankingVersion || r.PolicyVersion != AOQTV9ActiveSetNullspacePolicyVersion {
		return fmt.Errorf("AOQT V9 active-set nullspace contract version binding is unsupported")
	}
	if r.Dim != AOQTV9ActiveSetNullspaceExpectedDim || r.CoordinateCount != AOQTV9ActiveSetNullspaceExpectedAngleSize {
		return fmt.Errorf("AOQT V9 active-set nullspace must preserve D384 and 1536-angle AOQT serving format")
	}
	if r.DirectionCap != AOQTV9ActiveSetNullspaceDirectionCap || r.CandidateCap != AOQTV9ActiveSetNullspaceCandidateCap || len(r.Endpoints) > AOQTV9ActiveSetNullspaceCandidateCap || r.ObjectiveEvaluationCount > AOQTV9ActiveSetNullspaceObjectiveEvalMax {
		return fmt.Errorf("AOQT V9 active-set nullspace direction/candidate/evaluation cap exceeded")
	}
	if !aoqtFloat32SlicesEqual(r.Magnitudes, AOQTV9ActiveSetNullspaceMagnitudes()) {
		return fmt.Errorf("AOQT V9 active-set nullspace magnitudes are not canonical")
	}
	if r.ObjectiveEvaluationCount != 1+len(r.Endpoints) {
		return fmt.Errorf("AOQT V9 active-set nullspace objective evaluation accounting must be baseline plus endpoints")
	}
	if r.FullFoldRowCount != AOQTV7R6ProgressiveScreenPopulationTotal || r.FullFoldWeightingVersion != AOQTV9ActiveSetNullspaceFullWeighting || r.HTFactorsApplied {
		return fmt.Errorf("AOQT V9 active-set nullspace full-fold authority must be all 10,942 rows, unweighted")
	}
	if !r.BaselineLossFinite || !isFinite32(r.BaselineLoss) || !aoqtFiniteReceiptComponentsValue(r.BaselineComponents) || !aoqtV7R6PositiveActivation(r.BaselineActivation) {
		return fmt.Errorf("AOQT V9 active-set nullspace baseline evidence is invalid")
	}
	if err := validateAOQTLossMatchesComponents(r.BaselineLoss, r.BaselineComponents, "AOQT V9 active-set nullspace baseline"); err != nil {
		return err
	}
	policy := aoqtV7R6CanonicalEligibilityPolicy()
	policySHA, err := aoqtActualCoordinatePolicySHA256(policy)
	if err != nil {
		return err
	}
	if r.CanonicalEligibilityPolicySHA256 != policySHA {
		return fmt.Errorf("AOQT V9 active-set nullspace eligibility policy must be canonical")
	}
	if r.CanonicalV8ReceiptFileSHA256 != AOQTV9CanonicalV8ReceiptFileSHA256 || !r.CanonicalV8TerminalReceiptBound {
		return fmt.Errorf("AOQT V9 active-set nullspace must bind the canonical terminal V8 receipt")
	}
	if r.DenseInvariantStatus != AOQTV9ActiveSetNullspaceDenseProof || r.GradientEvidenceDeferred || !r.FullVerificationCompleted {
		return fmt.Errorf("AOQT V9 active-set nullspace requires exact completed gradient/objective/dense proof evidence")
	}
	if !isFinite64(r.DenseMaxAbsDeltaTolerance) || r.DenseMaxAbsDeltaTolerance != policy.DenseMaxAbsDeltaTolerance {
		return fmt.Errorf("AOQT V9 active-set nullspace dense tolerance must exactly match canonical policy")
	}
	for _, item := range []struct{ name, value string }{
		{"selection_seed_sha256", r.SelectionSeedSHA256},
		{"q3_gradient_sha256", r.Q3GradientSHA256},
		{"protected_gradient_matrix_sha256", r.ProtectedGradientMatrixSHA256},
		{"direction_evidence_sha256", r.DirectionEvidenceSHA256},
		{"direction_schedule_sha256", r.DirectionScheduleSHA256},
		{"canonical_objective_contract_sha256", r.CanonicalObjectiveContractSHA256},
		{"canonical_v8_receipt_file_sha256", r.CanonicalV8ReceiptFileSHA256},
		{"dense_invariant_proof_sha256", r.DenseInvariantProofSHA256},
		{"endpoint_hash_chain_tail_sha256", r.EndpointHashChainTailSHA256},
	} {
		if err := validateAOQTSHA256(item.value, "AOQT V9 active-set nullspace "+item.name); err != nil {
			return err
		}
	}
	if err := r.DirectionEvidence.Validate(); err != nil {
		return err
	}
	directionSHA, err := aoqtCanonicalSHA256(r.DirectionEvidence)
	if err != nil || directionSHA != r.DirectionEvidenceSHA256 {
		return fmt.Errorf("AOQT V9 active-set nullspace direction evidence hash mismatch")
	}
	scheduleSHA, err := aoqtV9DirectionScheduleSHA256(len(r.DirectionEvidence.Candidates))
	if err != nil || scheduleSHA != r.DirectionScheduleSHA256 {
		return fmt.Errorf("AOQT V9 active-set nullspace direction schedule hash mismatch")
	}
	if err := validateAOQTV9CanonicalEndpointSchedule(r.Endpoints, len(r.DirectionEvidence.Candidates)); err != nil {
		return err
	}
	if err := validateAOQTV9EndpointHashChain(r); err != nil {
		return err
	}
	ranking := aoqtV9RankEndpoints(r.Endpoints)
	rankAt := map[int]int{}
	for i, ordinal := range ranking {
		rankAt[ordinal] = i + 1
	}
	feasible := 0
	selected := make([]int, 0, len(r.Endpoints))
	for i, endpoint := range r.Endpoints {
		if err := validateAOQTV9Endpoint(endpoint, i, r.BaselineLoss, r.BaselineComponents, r.ObjectiveWeightSums, policy, r.DirectionEvidence.Candidates, r.DenseMaxAbsDeltaTolerance); err != nil {
			return err
		}
		if endpoint.RankOrder != rankAt[endpoint.Ordinal] {
			return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d rank_order does not match recomputed ranking", endpoint.Ordinal)
		}
		if endpoint.TransactionEligible {
			feasible++
			selected = append(selected, endpoint.Ordinal)
		}
	}
	if r.FeasibleEndpointCount != feasible || !reflect.DeepEqual(r.SelectedEndpointOrdinals, selected) {
		return fmt.Errorf("AOQT V9 active-set nullspace selected/feasible endpoint accounting does not match recomputed policy")
	}
	if r.Passed != (feasible > 0) {
		return fmt.Errorf("AOQT V9 active-set nullspace passed state does not match proof evidence")
	}
	if !r.Passed && strings.TrimSpace(r.FailureReason) == "" {
		return fmt.Errorf("AOQT V9 active-set nullspace failed receipt requires failure_reason")
	}
	if r.Passed && r.FailureReason != "" {
		return fmt.Errorf("AOQT V9 active-set nullspace passed receipt has failure_reason")
	}
	if requireBinding && (r.Binding.IOReport.ManifestPath == "" || r.Binding.IOReport.RowsJSONLPath == "" || r.Binding.PreflightPath == "" || r.Binding.CanonicalR4ReceiptPath == "" || r.Binding.CanonicalR5ReceiptPath == "" || r.Binding.CanonicalV8ReceiptPath == "") {
		return fmt.Errorf("AOQT V9 active-set nullspace bound receipt requires all source paths")
	}
	return nil
}

func validateAOQTV9NullspaceCandidate(candidate AOQTV9NullspaceCandidate, index int) error {
	if candidate.RankOrder != index+1 {
		return fmt.Errorf("AOQT V9 active-set nullspace candidate %d rank_order mismatch", candidate.Ordinal)
	}
	if !candidate.Cone.Accepted || candidate.PrimaryDerivative >= 0 || !finiteAOQTV9(candidate.PrimaryDerivative) {
		return fmt.Errorf("AOQT V9 active-set nullspace candidate %d is not q3-descent accepted", candidate.Ordinal)
	}
	if candidate.Cone.PrimaryDerivative != candidate.PrimaryDerivative || !slices.Equal(candidate.Cone.ProtectedDerivatives, candidate.ProtectedDerivatives) || candidate.Cone.ProtectedMaxDerivative != candidate.ProtectedMaxDerivative {
		return fmt.Errorf("AOQT V9 active-set nullspace candidate %d cone evidence mismatch", candidate.Ordinal)
	}
	if candidate.ProtectedMaxDerivative > AOQTV9ProtectedDerivativeTolerance || !finiteAOQTV9(candidate.ProtectedMaxDerivative) {
		return fmt.Errorf("AOQT V9 active-set nullspace candidate %d has unsafe protected derivative", candidate.Ordinal)
	}
	if len(candidate.Direction) != AOQTV9ActiveSetNullspaceExpectedAngleSize || !aoqtV9AllFinite(candidate.Direction) || candidate.DirectionMaxAbs != aoqtV9MaxAbs(candidate.Direction) || candidate.DirectionMaxAbs <= 0 {
		return fmt.Errorf("AOQT V9 active-set nullspace candidate %d direction evidence is invalid", candidate.Ordinal)
	}
	return nil
}

func validateAOQTV9Endpoint(endpoint AOQTSidecarV9ActiveSetNullspaceEndpoint, index int, baselineLoss float32, baselineComponents AOQTSidecarObjectiveComponents, weights AOQTSidecarRowWeights, policy AOQTSidecarCandidateEligibilityPolicy, directions []AOQTV9NullspaceCandidate, tolerance float64) error {
	if endpoint.Ordinal != index+1 {
		return fmt.Errorf("AOQT V9 active-set nullspace endpoint ordinal %d is not sequential", endpoint.Ordinal)
	}
	if endpoint.DirectionRankOrder <= 0 || endpoint.DirectionRankOrder > len(directions) {
		return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d direction rank order is invalid", endpoint.Ordinal)
	}
	direction := directions[endpoint.DirectionRankOrder-1]
	if endpoint.DirectionOrdinal != direction.Ordinal {
		return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d direction ordinal binding mismatch", endpoint.Ordinal)
	}
	if !containsFloat32AOQT(endpoint.Magnitude, AOQTV9ActiveSetNullspaceMagnitudes()) {
		return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d magnitude is not canonical", endpoint.Ordinal)
	}
	requested := aoqtV9ScaleDirection32(direction.Direction, endpoint.Magnitude)
	requestedSHA, err := aoqtActualDirectionVectorSHA256(requested)
	if err != nil || endpoint.RequestedDirectionSHA256 != requestedSHA {
		return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d requested direction hash mismatch", endpoint.Ordinal)
	}
	actualSHA, err := aoqtActualDirectionVectorSHA256(endpoint.ActualDirection)
	if err != nil || endpoint.ActualDirectionSHA256 != actualSHA {
		return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d actual direction hash mismatch", endpoint.Ordinal)
	}
	if endpoint.ActualDirectionMaxAbs != aoqtMaxAbsFloat32(endpoint.ActualDirection) || endpoint.AngleMoved != (endpoint.ActualDirectionMaxAbs != 0) {
		return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d actual movement flags mismatch", endpoint.Ordinal)
	}
	if !finiteAOQTV9(endpoint.ActualQ3Derivative) {
		return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d actual q3 derivative is not finite", endpoint.Ordinal)
	}
	if len(endpoint.ActualProtectedDerivatives) != len(direction.ProtectedDerivatives) {
		return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d actual protected derivative count mismatch", endpoint.Ordinal)
	}
	protectedSafe := true
	for i, value := range endpoint.ActualProtectedDerivatives {
		if !finiteAOQTV9(value) {
			return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d actual protected derivative[%d] is not finite", endpoint.Ordinal, i)
		}
		if value > AOQTV9ProtectedDerivativeTolerance {
			protectedSafe = false
		}
	}
	wantDerivative := endpoint.AngleMoved && endpoint.ActualQ3Derivative < 0 && protectedSafe
	if endpoint.DerivativeEligible != wantDerivative {
		return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d derivative eligibility does not match actual derivatives", endpoint.Ordinal)
	}
	if endpoint.BaselineLoss != baselineLoss || endpoint.BaselineComponents != baselineComponents || !endpoint.BaselineLossFinite || !endpoint.BaselineComponentsFinite {
		return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d baseline does not match receipt baseline", endpoint.Ordinal)
	}
	if endpoint.CandidateError != "" {
		if endpoint.TransactionEligible {
			return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d with candidate error cannot be transaction eligible", endpoint.Ordinal)
		}
		if endpoint.Reason != "probe_error" && endpoint.Reason != "nonfinite_loss" {
			return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d candidate error reason %q is unsupported", endpoint.Ordinal, endpoint.Reason)
		}
		if !endpoint.DenseInvariantProved || !finiteAOQTV9(endpoint.DenseInvariantMaxAbsDelta) || endpoint.DenseInvariantMaxAbsDelta < 0 || endpoint.DenseInvariantEligible != (endpoint.DenseInvariantMaxAbsDelta <= tolerance) {
			return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d dense invariant proof envelope is invalid", endpoint.Ordinal)
		}
		if err := validateAOQTSHA256(endpoint.DenseInvariantProofSHA256, "AOQT V9 active-set nullspace endpoint dense invariant proof sha256"); err != nil {
			return err
		}
		return nil
	}
	if !endpoint.CandidateLossFinite || !isFinite32(endpoint.CandidateLoss) || !endpoint.CandidateComponentsFinite || !aoqtFiniteReceiptComponentsValue(endpoint.CandidateComponents) || !endpoint.TotalDeltaFinite {
		return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d candidate evidence is not finite", endpoint.Ordinal)
	}
	if err := validateAOQTLossMatchesComponents(endpoint.CandidateLoss, endpoint.CandidateComponents, fmt.Sprintf("AOQT V9 active-set nullspace endpoint %d candidate", endpoint.Ordinal)); err != nil {
		return err
	}
	if endpoint.ComponentDeltas != aoqtObjectiveComponentDelta(endpoint.BaselineComponents, endpoint.CandidateComponents) || !endpoint.ComponentDeltasFinite || endpoint.TotalDelta != endpoint.CandidateLoss-endpoint.BaselineLoss {
		return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d objective delta mismatch", endpoint.Ordinal)
	}
	if !endpoint.DenseInvariantProved || !finiteAOQTV9(endpoint.DenseInvariantMaxAbsDelta) || endpoint.DenseInvariantMaxAbsDelta < 0 || endpoint.DenseInvariantEligible != (endpoint.DenseInvariantMaxAbsDelta <= tolerance) {
		return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d dense invariant proof envelope is invalid", endpoint.Ordinal)
	}
	if err := validateAOQTSHA256(endpoint.DenseInvariantProofSHA256, "AOQT V9 active-set nullspace endpoint dense invariant proof sha256"); err != nil {
		return err
	}
	wantQ3 := endpoint.CandidateComponents.Q3Gain < endpoint.BaselineComponents.Q3Gain-aoqtTransactionalQ3ImprovementMinMagnitude
	wantProtected := len(aoqtComponentRegressions(endpoint.BaselineComponents, endpoint.CandidateComponents, policy)) == 0
	decision := aoqtEvaluateTransactionalStepWithPolicy(aoqtStepEvaluation{loss: endpoint.BaselineLoss, components: endpoint.BaselineComponents}, aoqtStepEvaluation{loss: endpoint.CandidateLoss, activation: endpoint.CandidateActivation, components: endpoint.CandidateComponents}, weights, policy)
	wantTransaction := endpoint.AngleMoved && wantDerivative && decision.accepted && endpoint.DenseInvariantEligible
	if endpoint.Q3GainEligible != wantQ3 || endpoint.ProtectedEligible != wantProtected || endpoint.TransactionEligible != wantTransaction {
		return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d policy flags do not match recomputed evidence", endpoint.Ordinal)
	}
	wantReason := string(decision.reason)
	if !endpoint.AngleMoved {
		wantReason = string(aoqtRejectionNoAngleMovement)
	} else if !endpoint.DerivativeEligible {
		wantReason = AOQTV9ActiveSetNullspaceDerivativeSafetyReject
	} else if decision.accepted && !endpoint.DenseInvariantEligible {
		wantReason = AOQTV9ActiveSetNullspaceDenseRejection
	} else if decision.accepted {
		wantReason = aoqtProposalAcceptedReason
	}
	if endpoint.Reason != wantReason {
		return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d reason %q does not match recomputed %q", endpoint.Ordinal, endpoint.Reason, wantReason)
	}
	return nil
}

func RunAOQTV9ActiveSetNullspaceScreen(cfg AOQTSidecarV9ActiveSetNullspaceScreenConfig) (AOQTSidecarV9ActiveSetNullspaceScreenResult, error) {
	result := AOQTSidecarV9ActiveSetNullspaceScreenResult{}
	if cfg.Mode == "" {
		cfg.Mode = AOQTSidecarOptimizerModeV9ActiveSetNullspaceScreen
	}
	if cfg.Mode != AOQTSidecarOptimizerModeV9ActiveSetNullspaceScreen || !cfg.ProbeOnly {
		return result, fmt.Errorf("AOQT V9 active-set nullspace requires mode=%s and probe-only=true", AOQTSidecarOptimizerModeV9ActiveSetNullspaceScreen)
	}
	if cfg.CanonicalV8ReceiptFileSHA256 == "" {
		cfg.CanonicalV8ReceiptFileSHA256 = AOQTV9CanonicalV8ReceiptFileSHA256
	}
	if cfg.CanonicalV8ReceiptFileSHA256 != AOQTV9CanonicalV8ReceiptFileSHA256 {
		return result, fmt.Errorf("AOQT V9 active-set nullspace requires canonical V8 terminal receipt sha256 %s", AOQTV9CanonicalV8ReceiptFileSHA256)
	}
	if strings.TrimSpace(cfg.CanonicalV8ReceiptPath) == "" {
		return result, fmt.Errorf("AOQT V9 active-set nullspace requires canonical V8 terminal receipt path")
	}
	v8Receipt, v8FileSHA, err := loadBoundAOQTV9CanonicalV8Receipt(cfg.CanonicalV8ReceiptPath, cfg.CanonicalV8ReceiptFileSHA256)
	if err != nil {
		return result, err
	}
	set, ioReport, preflightSHA, split, r4, r5, r4FileSHA, r5FileSHA, err := aoqtV9ResolveInputs(cfg)
	if err != nil {
		return result, err
	}
	r4ObjectSHA, err := r4.SHA256()
	if err != nil {
		return result, err
	}
	r5ObjectSHA, err := r5.SHA256()
	if err != nil {
		return result, err
	}
	binding := AOQTSidecarV9ActiveSetNullspaceScreenBinding{Inputs: AOQTSidecarRunMetricInputs{AnchorArtifactSHA256: set.Manifest.AnchorArtifactSHA256, AnchorPackageManifestSHA256: set.Manifest.AnchorPackageManifestSHA256, AnchorEmbeddingSpaceID: set.Manifest.AnchorEmbeddingSpaceID, DatasetManifestSHA256: ioReport.ManifestSHA256, QrelsSHA256ByDataset: cloneAOQTStringMap(set.Manifest.QrelsSHA256ByDataset), CompatibilityDigest: set.Manifest.CompatibilityDigest}, IOReport: ioReport, PreflightPath: cfg.PreflightJSONPath, PreflightSHA256: preflightSHA, SplitBinding: cloneAOQTDevSplitBinding(split), FoldID: split.FoldID, CanonicalR4ReceiptPath: cfg.CanonicalR4ReceiptPath, CanonicalR4ReceiptSHA256: r4ObjectSHA, CanonicalR4ReceiptFileSHA256: r4FileSHA, CanonicalR5ReceiptPath: cfg.CanonicalR5ReceiptPath, CanonicalR5ReceiptSHA256: r5ObjectSHA, CanonicalR5ReceiptFileSHA256: r5FileSHA, CanonicalV8ReceiptPath: cfg.CanonicalV8ReceiptPath, CanonicalV8ReceiptFileSHA256: v8FileSHA}
	if err := validateAOQTV9V8Compatibility(v8Receipt, binding, len(set.Rows), AOQTV9ActiveSetNullspaceFullWeighting, false); err != nil {
		return result, err
	}
	receipt, err := buildAOQTV9ActiveSetNullspaceReceipt(set, binding, v8FileSHA)
	if err != nil {
		return result, err
	}
	if strings.TrimSpace(cfg.OutputReceiptJSONPath) != "" {
		if err := WriteAOQTV9ActiveSetNullspaceScreenReceipt(cfg.OutputReceiptJSONPath, receipt); err != nil {
			return result, err
		}
		result.ReceiptJSONPath = cfg.OutputReceiptJSONPath
	} else if err := receipt.ValidateBound(); err != nil {
		return result, err
	}
	receiptSHA, err := receipt.SHA256()
	if err != nil {
		return result, err
	}
	result.Receipt, result.ReceiptSHA256 = receipt, receiptSHA
	return result, nil
}

func buildAOQTV9ActiveSetNullspaceReceipt(set AOQTSidecarCalibrationSet, binding AOQTSidecarV9ActiveSetNullspaceScreenBinding, v8FileSHA string) (AOQTSidecarV9ActiveSetNullspaceScreenReceipt, error) {
	if err := set.Validate(); err != nil {
		return AOQTSidecarV9ActiveSetNullspaceScreenReceipt{}, fmt.Errorf("AOQT V9 active-set nullspace calibration set: %w", err)
	}
	if set.Manifest.Topology.Dim != AOQTV9ActiveSetNullspaceExpectedDim || len(set.Rows) != AOQTV7R6ProgressiveScreenPopulationTotal {
		return AOQTSidecarV9ActiveSetNullspaceScreenReceipt{}, fmt.Errorf("AOQT V9 active-set nullspace requires D384 and full 10,942-row fold")
	}
	if err := aoqtV7R6RejectOfficialOverlap(set.Rows); err != nil {
		return AOQTSidecarV9ActiveSetNullspaceScreenReceipt{}, fmt.Errorf("%s", strings.Replace(err.Error(), "AOQT V7-r6", "AOQT V9 active-set nullspace", 1))
	}
	objective, err := objectiveFromAOQTContract(set.Manifest.ObjectiveContract)
	if err != nil {
		return AOQTSidecarV9ActiveSetNullspaceScreenReceipt{}, err
	}
	policy, err := AOQTSidecarTrainingContractPolicy(set.Manifest.TrainingContract, set.Manifest.CandidateEligibilityPolicy)
	if err != nil {
		return AOQTSidecarV9ActiveSetNullspaceScreenReceipt{}, err
	}
	trainer, err := newAOQTV9ScreenTrainer(set, binding)
	if err != nil {
		return AOQTSidecarV9ActiveSetNullspaceScreenReceipt{}, err
	}
	rows := append([]AOQTSidecarCalibrationRow(nil), set.Rows...)
	baselineLoss, _, baselineActivation, baselineComponents, err := trainer.lossAndAngleGrad(rows, objective)
	if err != nil {
		return AOQTSidecarV9ActiveSetNullspaceScreenReceipt{}, err
	}
	q3Grad, err := trainer.q3GainOnlyAngleGrad(rows, objective)
	if err != nil {
		return AOQTSidecarV9ActiveSetNullspaceScreenReceipt{}, err
	}
	protected, err := trainer.protectedComponentAngleGrads(rows, objective, set.Manifest.ObjectiveContract.WeightSums)
	if err != nil {
		return AOQTSidecarV9ActiveSetNullspaceScreenReceipt{}, err
	}
	directionEvidence, q3SHA, protectedNames, protectedSHA, directionSHA, err := aoqtV9BuildDirectionEvidenceBound(q3Grad, protected)
	if err != nil {
		return AOQTSidecarV9ActiveSetNullspaceScreenReceipt{}, err
	}
	scheduleSHA, err := aoqtV9DirectionScheduleSHA256(len(directionEvidence.Candidates))
	if err != nil {
		return AOQTSidecarV9ActiveSetNullspaceScreenReceipt{}, err
	}
	endpoints := make([]AOQTSidecarV9ActiveSetNullspaceEndpoint, 0, len(directionEvidence.Candidates)*len(AOQTV9ActiveSetNullspaceMagnitudes()))
	baseline := aoqtStepEvaluation{loss: baselineLoss, activation: baselineActivation, components: baselineComponents}
	ordinal := 1
	for _, direction := range directionEvidence.Candidates {
		for _, magnitude := range AOQTV9ActiveSetNullspaceMagnitudes() {
			if len(endpoints) >= AOQTV9ActiveSetNullspaceCandidateCap {
				break
			}
			endpoints = append(endpoints, trainer.evaluateV9ActiveSetNullspaceEndpoint(rows, objective, baseline, set.Manifest.ObjectiveContract.WeightSums, q3Grad, protected, direction, ordinal, magnitude))
			ordinal++
		}
	}
	denseProofSHA, err := applyAOQTV9DenseProofsToEndpoints(endpoints, set, binding, trainer.topology, policy.DenseMaxAbsDeltaTolerance)
	if err != nil {
		return AOQTSidecarV9ActiveSetNullspaceScreenReceipt{}, err
	}
	applyAOQTV9EndpointRanking(endpoints)
	feasible, selected := aoqtV9EndpointSelection(endpoints)
	chainData, chainTail, err := aoqtV9EndpointHashChain(endpoints)
	if err != nil {
		return AOQTSidecarV9ActiveSetNullspaceScreenReceipt{}, err
	}
	contractSHA, err := aoqtActualCoordinateObjectiveContractSHA256(set.Manifest.ObjectiveContract)
	if err != nil {
		return AOQTSidecarV9ActiveSetNullspaceScreenReceipt{}, err
	}
	policySHA, err := aoqtActualCoordinatePolicySHA256(policy)
	if err != nil {
		return AOQTSidecarV9ActiveSetNullspaceScreenReceipt{}, err
	}
	selectionSeed, err := aoqtV9SelectionSeed(set, binding.FoldID, q3SHA, protectedSHA, directionSHA, v8FileSHA)
	if err != nil {
		return AOQTSidecarV9ActiveSetNullspaceScreenReceipt{}, err
	}
	receipt := AOQTSidecarV9ActiveSetNullspaceScreenReceipt{Schema: AOQTSidecarV9ActiveSetNullspaceScreenSchema, Mode: AOQTSidecarOptimizerModeV9ActiveSetNullspaceScreen, Required: true, ProbeOnly: true, Passed: feasible > 0, BasisVersion: AOQTV9NullspaceBasisVersion, ConeDerivativeVersion: AOQTV9ConeDerivativeVersion, ScheduleVersion: AOQTV9ActiveSetNullspaceScheduleVersion, RankingVersion: AOQTV9ActiveSetNullspaceRankingVersion, PolicyVersion: AOQTV9ActiveSetNullspacePolicyVersion, SelectionSeedSHA256: selectionSeed, CoordinateCount: countAOQTAngles(trainer.topology), Dim: set.Manifest.Topology.Dim, DirectionCap: AOQTV9ActiveSetNullspaceDirectionCap, Magnitudes: AOQTV9ActiveSetNullspaceMagnitudes(), CandidateCap: AOQTV9ActiveSetNullspaceCandidateCap, ObjectiveEvaluationCount: 1 + len(endpoints), FullFoldRowCount: len(rows), FullFoldWeightingVersion: AOQTV9ActiveSetNullspaceFullWeighting, HTFactorsApplied: false, BaselineLoss: baselineLoss, BaselineLossFinite: isFinite32(baselineLoss), BaselineComponents: baselineComponents, BaselineActivation: baselineActivation, ObjectiveWeightSums: set.Manifest.ObjectiveContract.WeightSums, Q3GradientSHA256: q3SHA, ProtectedGradientNames: protectedNames, ProtectedGradientMatrixSHA256: protectedSHA, DirectionEvidenceSHA256: directionSHA, DirectionScheduleSHA256: scheduleSHA, CanonicalObjectiveContractSHA256: contractSHA, CanonicalEligibilityPolicySHA256: policySHA, CanonicalV8ReceiptFileSHA256: v8FileSHA, CanonicalV8TerminalReceiptBound: true, DenseInvariantStatus: AOQTV9ActiveSetNullspaceDenseProof, DenseInvariantProofSHA256: denseProofSHA, DenseMaxAbsDeltaTolerance: policy.DenseMaxAbsDeltaTolerance, GradientEvidenceDeferred: false, FullVerificationCompleted: true, FeasibleEndpointCount: feasible, SelectedEndpointOrdinals: selected, EndpointHashChain: string(chainData), EndpointHashChainTailSHA256: chainTail, DirectionEvidence: directionEvidence, Endpoints: endpoints, Binding: binding}
	if !receipt.Passed {
		receipt.FailureReason = "no dense-invariant-safe V9 active-set nullspace endpoint satisfied the transaction policy"
	}
	return receipt, nil
}

func newAOQTV9ScreenTrainer(set AOQTSidecarCalibrationSet, binding AOQTSidecarV9ActiveSetNullspaceScreenBinding) (*AOQTSidecarTrainer, error) {
	trainer, err := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{Dim: set.Manifest.Topology.Dim, Stages: set.Manifest.Topology.Stages, PairingSeed: set.Manifest.Topology.Seed, WorkplanSeed: 1, OptimizerMode: AOQTSidecarOptimizerModeV7R4DevActualCoordinate, ActualCoordinateProbeRequired: true, ActualCoordinateProbeOnly: true, ActualCoordinateProbeFoldID: binding.FoldID, ActualCoordinateProbeSplitManifestSHA256: binding.SplitBinding.SplitManifestSHA256, LearningRate: AOQTV7R4ActualCoordinateProbeLearningRate, AngleCap: AOQTSidecarDefaultAngleCap, MaxAngleCap: AOQTSidecarHardMaxAngleCap, TrainingContract: set.Manifest.TrainingContract, CandidateEligibilityPolicy: cloneAOQTSidecarCandidateEligibilityPolicy(set.Manifest.CandidateEligibilityPolicy)})
	if err != nil {
		return nil, err
	}
	if _, err := trainer.Plan(set); err != nil {
		return nil, err
	}
	return trainer, nil
}

func AOQTV9BuildDirectionEvidence(q3Gradient []float32, protected []aoqtProtectedAngleGradient) (AOQTSidecarV9DirectionEvidence, error) {
	evidence, _, _, _, _, err := aoqtV9BuildDirectionEvidenceBound(q3Gradient, protected)
	return evidence, err
}

func aoqtV9BuildDirectionEvidenceBound(q3Gradient []float32, protected []aoqtProtectedAngleGradient) (AOQTSidecarV9DirectionEvidence, string, []string, string, string, error) {
	q3SHA, err := aoqtCanonicalSHA256(q3Gradient)
	if err != nil {
		return AOQTSidecarV9DirectionEvidence{}, "", nil, "", "", err
	}
	protectedNames := aoqtProtectedGradientNames(protected)
	protectedSHA, err := aoqtCanonicalSHA256(struct {
		Names     []string                     `json:"names"`
		Gradients []aoqtProtectedAngleGradient `json:"gradients"`
	}{protectedNames, protected})
	if err != nil {
		return AOQTSidecarV9DirectionEvidence{}, "", nil, "", "", err
	}
	rows := make([][]float64, len(protected))
	for i, gradient := range protected {
		rows[i] = float32ToFloat64AOQTV9(gradient.Grad)
	}
	candidates, err := AOQTV9BuildNullspaceCandidates(float32ToFloat64AOQTV9(q3Gradient), rows)
	if err != nil {
		return AOQTSidecarV9DirectionEvidence{}, "", nil, "", "", err
	}
	if len(candidates) > AOQTV9ActiveSetNullspaceDirectionCap {
		candidates = candidates[:AOQTV9ActiveSetNullspaceDirectionCap]
	}
	evidence := AOQTSidecarV9DirectionEvidence{Schema: AOQTSidecarV9ActiveSetNullspaceScreenSchema, Mode: AOQTSidecarOptimizerModeV9ActiveSetNullspaceScreen, BasisVersion: AOQTV9NullspaceBasisVersion, ConeDerivativeVersion: AOQTV9ConeDerivativeVersion, CanonicalV8ReceiptSHA256: AOQTV9CanonicalV8ReceiptFileSHA256, DirectionCap: AOQTV9ActiveSetNullspaceDirectionCap, CandidateCap: AOQTV9ActiveSetNullspaceCandidateCap, ObjectiveEvaluationMax: AOQTV9ActiveSetNullspaceObjectiveEvalMax, Magnitudes: AOQTV9NullspaceMagnitudes(), Candidates: candidates}
	if err := evidence.Validate(); err != nil {
		return AOQTSidecarV9DirectionEvidence{}, "", nil, "", "", err
	}
	evidenceSHA, err := aoqtCanonicalSHA256(evidence)
	if err != nil {
		return AOQTSidecarV9DirectionEvidence{}, "", nil, "", "", err
	}
	return evidence, q3SHA, protectedNames, protectedSHA, evidenceSHA, nil
}

func (t *AOQTSidecarTrainer) evaluateV9ActiveSetNullspaceEndpoint(rows []AOQTSidecarCalibrationRow, objective AOQTSidecarVectorObjective, baseline aoqtStepEvaluation, weights AOQTSidecarRowWeights, q3Grad []float32, protected []aoqtProtectedAngleGradient, direction AOQTV9NullspaceCandidate, ordinal int, magnitude float32) AOQTSidecarV9ActiveSetNullspaceEndpoint {
	endpoint := AOQTSidecarV9ActiveSetNullspaceEndpoint{Ordinal: ordinal, DirectionRankOrder: direction.RankOrder, DirectionOrdinal: direction.Ordinal, Magnitude: magnitude, BaselineLoss: baseline.loss, BaselineLossFinite: isFinite32(baseline.loss), BaselineComponents: baseline.components, BaselineComponentsFinite: aoqtFiniteReceiptComponentsValue(baseline.components), ActualDirection: make([]float32, len(t.angles))}
	requested := aoqtV9ScaleDirection32(direction.Direction, magnitude)
	endpoint.RequestedDirectionSHA256, _ = aoqtActualDirectionVectorSHA256(requested)
	state := t.snapshotOptimizerState()
	defer t.restoreOptimizerState(state)
	for i, delta := range requested {
		t.angles[i] += delta
	}
	if err := t.ProjectAngles(); err != nil {
		endpoint.ActualDirectionSHA256, _ = aoqtActualDirectionVectorSHA256(endpoint.ActualDirection)
		endpoint.ActualProtectedDerivatives = make([]float64, len(protected))
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
	endpoint.ActualQ3Derivative = aoqtV9Dot(float32ToFloat64AOQTV9(q3Grad), float32ToFloat64AOQTV9(endpoint.ActualDirection))
	endpoint.ActualProtectedDerivatives = aoqtV9ActualProtectedDerivatives(protected, endpoint.ActualDirection)
	endpoint.DerivativeEligible = endpoint.AngleMoved && endpoint.ActualQ3Derivative < 0 && aoqtV9DerivativesNonPositive(endpoint.ActualProtectedDerivatives)
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
	endpoint.ProtectedEligible = len(aoqtComponentRegressions(baseline.components, components, t.eligibilityPolicy)) == 0
	decision := aoqtEvaluateTransactionalStepWithPolicy(baseline, aoqtStepEvaluation{loss: candidateLoss, activation: activation, components: components}, weights, t.eligibilityPolicy)
	if !endpoint.AngleMoved {
		endpoint.Reason = string(aoqtRejectionNoAngleMovement)
	} else if !endpoint.DerivativeEligible {
		endpoint.Reason = AOQTV9ActiveSetNullspaceDerivativeSafetyReject
	} else if decision.accepted {
		endpoint.TransactionEligible = true
		endpoint.Reason = aoqtProposalAcceptedReason
	} else {
		endpoint.Reason = string(decision.reason)
	}
	return endpoint
}

func validateAOQTV9BoundObjectiveReplay(r AOQTSidecarV9ActiveSetNullspaceScreenReceipt, trainer *AOQTSidecarTrainer, set AOQTSidecarCalibrationSet, objective AOQTSidecarVectorObjective, q3Grad []float32, protected []aoqtProtectedAngleGradient) error {
	baseline := aoqtStepEvaluation{loss: r.BaselineLoss, activation: r.BaselineActivation, components: r.BaselineComponents}
	expected := make([]AOQTSidecarV9ActiveSetNullspaceEndpoint, 0, len(r.Endpoints))
	ordinal := 1
	for _, direction := range r.DirectionEvidence.Candidates {
		for _, magnitude := range AOQTV9ActiveSetNullspaceMagnitudes() {
			if len(expected) >= AOQTV9ActiveSetNullspaceCandidateCap || len(expected) >= len(r.Endpoints) {
				break
			}
			expected = append(expected, trainer.evaluateV9ActiveSetNullspaceEndpoint(set.Rows, objective, baseline, set.Manifest.ObjectiveContract.WeightSums, q3Grad, protected, direction, ordinal, magnitude))
			ordinal++
		}
	}
	proofSHA, err := applyAOQTV9DenseProofsToEndpoints(expected, set, r.Binding, trainer.topology, r.DenseMaxAbsDeltaTolerance)
	if err != nil {
		return err
	}
	if proofSHA != r.DenseInvariantProofSHA256 {
		return fmt.Errorf("AOQT V9 active-set nullspace objective replay dense aggregate hash mismatch")
	}
	applyAOQTV9EndpointRanking(expected)
	if len(expected) != len(r.Endpoints) {
		return fmt.Errorf("AOQT V9 active-set nullspace objective replay endpoint count mismatch")
	}
	for i := range r.Endpoints {
		if !reflect.DeepEqual(r.Endpoints[i], expected[i]) {
			return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d objective replay evidence mismatch", r.Endpoints[i].Ordinal)
		}
	}
	return nil
}

func applyAOQTV9DenseProofsToEndpoints(endpoints []AOQTSidecarV9ActiveSetNullspaceEndpoint, set AOQTSidecarCalibrationSet, binding AOQTSidecarV9ActiveSetNullspaceScreenBinding, topology AOQTGivensTransform, tolerance float64) (string, error) {
	for i := range endpoints {
		proof, proofSHA, err := aoqtV9DenseInvariantEndpointProof(endpoints[i], set, binding, topology, tolerance)
		if err != nil {
			return "", err
		}
		endpoints[i].DenseInvariantProved = true
		endpoints[i].DenseInvariantEligible = proof.DenseInvariantEligible
		endpoints[i].DenseInvariantMaxAbsDelta = proof.DenseInvariantMaxAbsDelta
		endpoints[i].DenseInvariantProofSHA256 = proofSHA
		if endpoints[i].TransactionEligible && !endpoints[i].DenseInvariantEligible {
			endpoints[i].TransactionEligible = false
			endpoints[i].Reason = AOQTV9ActiveSetNullspaceDenseRejection
		}
	}
	return aoqtV9DenseInvariantProofAggregateSHA(binding, endpoints, tolerance)
}

func aoqtV9DenseInvariantEndpointProof(endpoint AOQTSidecarV9ActiveSetNullspaceEndpoint, set AOQTSidecarCalibrationSet, binding AOQTSidecarV9ActiveSetNullspaceScreenBinding, topology AOQTGivensTransform, tolerance float64) (AOQTSidecarV9DenseInvariantEndpointProof, string, error) {
	transform, err := aoqtV8CandidateTransformFromDirection(topology, endpoint.ActualDirection)
	if err != nil {
		return AOQTSidecarV9DenseInvariantEndpointProof{}, "", err
	}
	pairingsSHA, err := transform.PairingsSHA256()
	if err != nil {
		return AOQTSidecarV9DenseInvariantEndpointProof{}, "", err
	}
	anglesSHA, err := transform.AnglesSHA256()
	if err != nil {
		return AOQTSidecarV9DenseInvariantEndpointProof{}, "", err
	}
	maxDelta, err := DenseInvariantMaxAbsDelta(transform, set.Rows)
	if err != nil {
		return AOQTSidecarV9DenseInvariantEndpointProof{}, "", err
	}
	proof := AOQTSidecarV9DenseInvariantEndpointProof{Schema: AOQTV9ActiveSetNullspaceDenseProofSchema, Version: AOQTV9ActiveSetNullspaceDenseProof, EndpointOrdinal: endpoint.Ordinal, EndpointEvidence: aoqtV9EndpointDenseSummary(endpoint), ManifestSHA256: binding.IOReport.ManifestSHA256, RowsSHA256: binding.IOReport.RowsSHA256, RowIDSHA256: set.Manifest.RowIDSHA256, RowCount: len(set.Rows), FoldID: binding.FoldID, FullFoldWeightingVersion: AOQTV9ActiveSetNullspaceFullWeighting, HTFactorsApplied: false, CandidatePairingsSHA256: pairingsSHA, CandidateAnglesSHA256: anglesSHA, DenseMaxAbsDeltaTolerance: tolerance, DenseInvariantMaxAbsDelta: maxDelta, DenseInvariantEligible: maxDelta <= tolerance}
	hash, err := aoqtCanonicalSHA256(proof)
	return proof, hash, err
}

func aoqtV9DenseInvariantProofAggregateSHA(binding AOQTSidecarV9ActiveSetNullspaceScreenBinding, endpoints []AOQTSidecarV9ActiveSetNullspaceEndpoint, tolerance float64) (string, error) {
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
		proofs[i] = endpointProofBinding{Ordinal: endpoint.Ordinal, DenseInvariantEligible: endpoint.DenseInvariantEligible, DenseInvariantMaxAbsDelta: endpoint.DenseInvariantMaxAbsDelta, DenseInvariantProofSHA256: endpoint.DenseInvariantProofSHA256, ActualDirectionSHA256: endpoint.ActualDirectionSHA256, TransactionEligible: endpoint.TransactionEligible}
	}
	return aoqtCanonicalSHA256(struct {
		Version                   string                 `json:"version"`
		ManifestSHA256            string                 `json:"manifest_sha256"`
		RowsSHA256                string                 `json:"rows_sha256"`
		RowCount                  int                    `json:"row_count"`
		FoldID                    string                 `json:"fold_id"`
		DenseMaxAbsDeltaTolerance float64                `json:"dense_max_abs_delta_tolerance"`
		EndpointProofs            []endpointProofBinding `json:"endpoint_proofs"`
	}{AOQTV9ActiveSetNullspaceDenseProof, binding.IOReport.ManifestSHA256, binding.IOReport.RowsSHA256, binding.IOReport.RowCount, binding.FoldID, tolerance, proofs})
}

func aoqtV9EndpointDenseSummary(endpoint AOQTSidecarV9ActiveSetNullspaceEndpoint) AOQTSidecarV9DenseInvariantEndpointSummary {
	return AOQTSidecarV9DenseInvariantEndpointSummary{Ordinal: endpoint.Ordinal, DirectionRankOrder: endpoint.DirectionRankOrder, DirectionOrdinal: endpoint.DirectionOrdinal, Magnitude: endpoint.Magnitude, RequestedDirectionSHA256: endpoint.RequestedDirectionSHA256, ActualDirectionSHA256: endpoint.ActualDirectionSHA256, ActualDirectionMaxAbs: endpoint.ActualDirectionMaxAbs, AngleMoved: endpoint.AngleMoved, BaselineLoss: endpoint.BaselineLoss, CandidateLoss: endpoint.CandidateLoss}
}

func loadBoundAOQTV9CanonicalV8Receipt(path, expectedFileSHA string) (AOQTSidecarV8ConstrainedBasisScreenReceipt, string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return AOQTSidecarV8ConstrainedBasisScreenReceipt{}, "", fmt.Errorf("AOQT V9 active-set nullspace canonical V8 receipt reread: %w", err)
	}
	fileSHA := sha256BytesAOQT(data)
	if fileSHA != expectedFileSHA || fileSHA != AOQTV9CanonicalV8ReceiptFileSHA256 {
		return AOQTSidecarV8ConstrainedBasisScreenReceipt{}, "", fmt.Errorf("AOQT V9 active-set nullspace canonical V8 receipt hash mismatch")
	}
	var receipt AOQTSidecarV8ConstrainedBasisScreenReceipt
	if err := strictUnmarshalAOQT(data, &receipt); err != nil {
		return AOQTSidecarV8ConstrainedBasisScreenReceipt{}, "", fmt.Errorf("AOQT V9 active-set nullspace canonical V8 receipt decode: %w", err)
	}
	if err := receipt.ValidateBound(); err != nil {
		return AOQTSidecarV8ConstrainedBasisScreenReceipt{}, "", fmt.Errorf("AOQT V9 active-set nullspace canonical V8 receipt validation: %w", err)
	}
	if receipt.Passed {
		return AOQTSidecarV8ConstrainedBasisScreenReceipt{}, "", fmt.Errorf("AOQT V9 active-set nullspace canonical V8 receipt must be terminal negative")
	}
	return receipt, fileSHA, nil
}

func validateAOQTV9V8Compatibility(v8 AOQTSidecarV8ConstrainedBasisScreenReceipt, binding AOQTSidecarV9ActiveSetNullspaceScreenBinding, fullFoldRowCount int, fullWeightingVersion string, htFactorsApplied bool) error {
	if !reflect.DeepEqual(v8.Binding.Inputs, binding.Inputs) || !reflect.DeepEqual(v8.Binding.IOReport, binding.IOReport) || v8.Binding.PreflightPath != binding.PreflightPath || v8.Binding.PreflightSHA256 != binding.PreflightSHA256 || !reflect.DeepEqual(v8.Binding.SplitBinding, binding.SplitBinding) || v8.Binding.FoldID != binding.FoldID {
		return fmt.Errorf("AOQT V9 active-set nullspace canonical V8 input/fold/preflight binding mismatch")
	}
	if v8.FullFoldRowCount != fullFoldRowCount || v8.FullFoldWeightingVersion != fullWeightingVersion || v8.HTFactorsApplied != htFactorsApplied {
		return fmt.Errorf("AOQT V9 active-set nullspace canonical V8 full-fold authority mismatch")
	}
	if v8.Binding.CanonicalR4ReceiptPath != binding.CanonicalR4ReceiptPath || v8.Binding.CanonicalR4ReceiptSHA256 != binding.CanonicalR4ReceiptSHA256 || v8.Binding.CanonicalR4ReceiptFileSHA256 != binding.CanonicalR4ReceiptFileSHA256 || v8.Binding.CanonicalR5ReceiptPath != binding.CanonicalR5ReceiptPath || v8.Binding.CanonicalR5ReceiptSHA256 != binding.CanonicalR5ReceiptSHA256 || v8.Binding.CanonicalR5ReceiptFileSHA256 != binding.CanonicalR5ReceiptFileSHA256 {
		return fmt.Errorf("AOQT V9 active-set nullspace canonical V8 r4/r5 receipt binding mismatch")
	}
	return nil
}

func rereadAOQTV9ActiveSetNullspaceBinding(r AOQTSidecarV9ActiveSetNullspaceScreenReceipt) error {
	binding := r.Binding
	manifestData, err := os.ReadFile(binding.IOReport.ManifestPath)
	if err != nil {
		return fmt.Errorf("AOQT V9 active-set nullspace bound manifest reread: %w", err)
	}
	var manifest AOQTSidecarCalibrationManifest
	if err := strictUnmarshalAOQT(manifestData, &manifest); err != nil {
		return fmt.Errorf("AOQT V9 active-set nullspace bound manifest is invalid: %w", err)
	}
	manifestSHA, err := AOQTSidecarManifestSHA256(manifest)
	if err != nil {
		return err
	}
	rowsData, err := os.ReadFile(binding.IOReport.RowsJSONLPath)
	if err != nil {
		return fmt.Errorf("AOQT V9 active-set nullspace bound rows reread: %w", err)
	}
	rowsSHA := sha256BytesAOQT(rowsData)
	rows, err := parseAOQTCalibrationRowsJSONL(rowsData, binding.IOReport.RowsJSONLPath)
	if err != nil {
		return err
	}
	set := AOQTSidecarCalibrationSet{Manifest: manifest, Rows: rows}
	if manifestSHA != binding.IOReport.ManifestSHA256 || rowsSHA != binding.IOReport.RowsSHA256 || len(rows) != r.FullFoldRowCount {
		return fmt.Errorf("AOQT V9 active-set nullspace bound manifest/rows mismatch")
	}
	expectedReport, err := aoqtV7R6IOReportForSet(set, binding.IOReport.ManifestPath, binding.IOReport.RowsJSONLPath, rowsSHA)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(expectedReport, binding.IOReport) {
		return fmt.Errorf("AOQT V9 active-set nullspace IO report does not match reread manifest/rows")
	}
	preflightData, err := os.ReadFile(binding.PreflightPath)
	if err != nil {
		return fmt.Errorf("AOQT V9 active-set nullspace bound preflight reread: %w", err)
	}
	preflightSHA := sha256BytesAOQT(preflightData)
	if preflightSHA != binding.PreflightSHA256 {
		return fmt.Errorf("AOQT V9 active-set nullspace preflight hash mismatch")
	}
	var preflight AOQTSidecarMaterializePreflight
	if err := strictUnmarshalAOQT(preflightData, &preflight); err != nil {
		return fmt.Errorf("AOQT V9 active-set nullspace preflight is invalid: %w", err)
	}
	if err := validateAOQTFailClosedPreflight(preflight, binding.PreflightPath, expectedReport, binding.Inputs, manifest.Topology, manifest.ObjectiveContract, manifest.LegalGates); err != nil {
		return fmt.Errorf("AOQT V9 active-set nullspace preflight binding: %w", err)
	}
	v8Receipt, v8FileSHA, err := loadBoundAOQTV9CanonicalV8Receipt(binding.CanonicalV8ReceiptPath, binding.CanonicalV8ReceiptFileSHA256)
	if err != nil {
		return err
	}
	if err := validateAOQTV9V8Compatibility(v8Receipt, binding, r.FullFoldRowCount, r.FullFoldWeightingVersion, r.HTFactorsApplied); err != nil {
		return err
	}
	contractSHA, err := aoqtActualCoordinateObjectiveContractSHA256(manifest.ObjectiveContract)
	if err != nil {
		return err
	}
	if r.CanonicalObjectiveContractSHA256 != contractSHA {
		return fmt.Errorf("AOQT V9 active-set nullspace objective contract hash does not match reread fold manifest")
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
		return fmt.Errorf("AOQT V9 active-set nullspace eligibility policy hash does not match reread fold manifest")
	}
	objective, err := objectiveFromAOQTContract(manifest.ObjectiveContract)
	if err != nil {
		return err
	}
	trainer, err := newAOQTV9ScreenTrainer(set, binding)
	if err != nil {
		return err
	}
	baselineLoss, _, baselineActivation, baselineComponents, err := trainer.lossAndAngleGrad(rows, objective)
	if err != nil {
		return err
	}
	q3Grad, err := trainer.q3GainOnlyAngleGrad(rows, objective)
	if err != nil {
		return err
	}
	protected, err := trainer.protectedComponentAngleGrads(rows, objective, manifest.ObjectiveContract.WeightSums)
	if err != nil {
		return err
	}
	directionEvidence, q3SHA, protectedNames, protectedSHA, directionSHA, err := aoqtV9BuildDirectionEvidenceBound(q3Grad, protected)
	if err != nil {
		return err
	}
	if r.BaselineLoss != baselineLoss || r.BaselineActivation != baselineActivation || r.BaselineComponents != baselineComponents || r.ObjectiveWeightSums != manifest.ObjectiveContract.WeightSums || r.Q3GradientSHA256 != q3SHA || !slices.Equal(r.ProtectedGradientNames, protectedNames) || r.ProtectedGradientMatrixSHA256 != protectedSHA || r.DirectionEvidenceSHA256 != directionSHA || !reflect.DeepEqual(r.DirectionEvidence, directionEvidence) {
		return fmt.Errorf("AOQT V9 active-set nullspace gradient/direction evidence does not match bound replay")
	}
	selectionSeed, err := aoqtV9SelectionSeed(set, binding.FoldID, q3SHA, protectedSHA, directionSHA, v8FileSHA)
	if err != nil {
		return err
	}
	if selectionSeed != r.SelectionSeedSHA256 {
		return fmt.Errorf("AOQT V9 active-set nullspace selection seed hash mismatch")
	}
	if err := validateAOQTV9BoundObjectiveReplay(r, trainer, set, objective, q3Grad, protected); err != nil {
		return err
	}
	return nil
}

func aoqtV9ResolveInputs(cfg AOQTSidecarV9ActiveSetNullspaceScreenConfig) (AOQTSidecarCalibrationSet, AOQTSidecarCalibrationIOReport, string, *AOQTSidecarDevSplitBinding, *AOQTSidecarActualCoordinateProbeReceipt, *AOQTSidecarActualCoordinateTrainReceipt, string, string, error) {
	v7cfg := AOQTSidecarV7R6ProgressiveScreenConfig{Set: cfg.Set, ManifestPath: cfg.ManifestPath, RowsJSONLPath: cfg.RowsJSONLPath, PreflightJSONPath: cfg.PreflightJSONPath, SplitManifestPath: cfg.SplitManifestPath, FoldID: cfg.FoldID, ExpectedManifestSHA256: cfg.ExpectedManifestSHA256, ExpectedRowsSHA256: cfg.ExpectedRowsSHA256, ExpectedPreflightSHA256: cfg.ExpectedPreflightSHA256, ExpectedSplitManifestSHA256: cfg.ExpectedSplitManifestSHA256, ExpectedAnchorArtifactSHA256: cfg.ExpectedAnchorArtifactSHA256, ExpectedAnchorPackageManifestSHA256: cfg.ExpectedAnchorPackageManifestSHA256, ExpectedAnchorEmbeddingSpaceID: cfg.ExpectedAnchorEmbeddingSpaceID, ExpectedCompatibilityDigest: cfg.ExpectedCompatibilityDigest, ExpectedSourceArtifactHashes: append([]string(nil), cfg.ExpectedSourceArtifactHashes...), ExpectedVectorCacheHashes: append([]string(nil), cfg.ExpectedVectorCacheHashes...), CanonicalR4ReceiptPath: cfg.CanonicalR4ReceiptPath, CanonicalR4ReceiptSHA256: cfg.CanonicalR4ReceiptSHA256, CanonicalR5ReceiptPath: cfg.CanonicalR5ReceiptPath, CanonicalR5ReceiptSHA256: cfg.CanonicalR5ReceiptSHA256, Mode: AOQTSidecarOptimizerModeV7R6ProgressiveScreen, ProbeOnly: true, FullVerification: true}
	set, ioReport, _, preflightSHA, split, r4, r5, r4FileSHA, r5FileSHA, err := aoqtV7R6ResolveInputs(v7cfg)
	if err != nil {
		return AOQTSidecarCalibrationSet{}, AOQTSidecarCalibrationIOReport{}, "", nil, nil, nil, "", "", err
	}
	if split == nil || r4 == nil || r5 == nil {
		return AOQTSidecarCalibrationSet{}, AOQTSidecarCalibrationIOReport{}, "", nil, nil, nil, "", "", fmt.Errorf("AOQT V9 active-set nullspace requires bound split, r4, and r5 receipts")
	}
	return set, ioReport, preflightSHA, split, r4, r5, r4FileSHA, r5FileSHA, nil
}

func validateAOQTV9CanonicalEndpointSchedule(endpoints []AOQTSidecarV9ActiveSetNullspaceEndpoint, directionCount int) error {
	want := 0
	for rankOrder := 1; rankOrder <= directionCount; rankOrder++ {
		for _, magnitude := range AOQTV9ActiveSetNullspaceMagnitudes() {
			if want >= AOQTV9ActiveSetNullspaceCandidateCap {
				break
			}
			if want >= len(endpoints) {
				return fmt.Errorf("AOQT V9 active-set nullspace omitted endpoint from canonical schedule")
			}
			endpoint := endpoints[want]
			if endpoint.Ordinal != want+1 || endpoint.DirectionRankOrder != rankOrder || endpoint.Magnitude != magnitude {
				return fmt.Errorf("AOQT V9 active-set nullspace endpoint %d does not match canonical schedule position %d", endpoint.Ordinal, want+1)
			}
			want++
		}
	}
	if len(endpoints) != want {
		return fmt.Errorf("AOQT V9 active-set nullspace duplicate or extra endpoint outside canonical schedule")
	}
	return nil
}

func aoqtV9RankEndpoints(endpoints []AOQTSidecarV9ActiveSetNullspaceEndpoint) []int {
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
	return ranking
}

func applyAOQTV9EndpointRanking(endpoints []AOQTSidecarV9ActiveSetNullspaceEndpoint) {
	ranking := aoqtV9RankEndpoints(endpoints)
	rankAt := map[int]int{}
	for i, ordinal := range ranking {
		rankAt[ordinal] = i + 1
	}
	for i := range endpoints {
		endpoints[i].RankOrder = rankAt[endpoints[i].Ordinal]
	}
}

func aoqtV9EndpointSelection(endpoints []AOQTSidecarV9ActiveSetNullspaceEndpoint) (int, []int) {
	selected := make([]int, 0, len(endpoints))
	for _, endpoint := range endpoints {
		if endpoint.TransactionEligible {
			selected = append(selected, endpoint.Ordinal)
		}
	}
	return len(selected), selected
}

func aoqtV9EndpointHashChain(endpoints []AOQTSidecarV9ActiveSetNullspaceEndpoint) ([]byte, string, error) {
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

func validateAOQTV9EndpointHashChain(r AOQTSidecarV9ActiveSetNullspaceScreenReceipt) error {
	var chain []string
	if err := strictUnmarshalAOQT([]byte(r.EndpointHashChain), &chain); err != nil {
		return fmt.Errorf("AOQT V9 active-set nullspace endpoint hash chain decode: %w", err)
	}
	if len(chain) != len(r.Endpoints) {
		return fmt.Errorf("AOQT V9 active-set nullspace endpoint hash chain length mismatch")
	}
	previous := ""
	for i, endpoint := range r.Endpoints {
		payload, err := json.Marshal(endpoint)
		if err != nil {
			return err
		}
		previous = sha256AOQTCoordinateSearchChainEntry(previous, string(payload))
		if chain[i] != previous {
			return fmt.Errorf("AOQT V9 active-set nullspace endpoint hash chain entry %d mismatch", i)
		}
	}
	if previous != r.EndpointHashChainTailSHA256 {
		return fmt.Errorf("AOQT V9 active-set nullspace endpoint hash chain tail mismatch")
	}
	return nil
}

func aoqtV9DirectionScheduleSHA256(directionCount int) (string, error) {
	return aoqtCanonicalSHA256(struct {
		Version        string    `json:"version"`
		DirectionCap   int       `json:"direction_cap"`
		DirectionCount int       `json:"direction_count"`
		Magnitudes     []float32 `json:"magnitudes"`
		CandidateCap   int       `json:"candidate_cap"`
	}{AOQTV9ActiveSetNullspaceScheduleVersion, AOQTV9ActiveSetNullspaceDirectionCap, directionCount, AOQTV9ActiveSetNullspaceMagnitudes(), AOQTV9ActiveSetNullspaceCandidateCap})
}

func aoqtV9SelectionSeed(set AOQTSidecarCalibrationSet, foldID, q3SHA, protectedSHA, directionSHA, v8SHA string) (string, error) {
	manifestSHA, err := AOQTSidecarManifestSHA256(set.Manifest)
	if err != nil {
		return "", err
	}
	payload := strings.Join([]string{AOQTV9ActiveSetNullspaceSeed, foldID, manifestSHA, set.Manifest.RowIDSHA256, q3SHA, protectedSHA, directionSHA, v8SHA}, "\x00")
	sum := sha256.Sum256([]byte(payload))
	return hex.EncodeToString(sum[:]), nil
}

func aoqtV9ScaleDirection32(direction []float64, magnitude float32) []float32 {
	out := make([]float32, len(direction))
	for i, value := range direction {
		out[i] = float32(value) * magnitude
	}
	return out
}

func aoqtV9ActualProtectedDerivatives(protected []aoqtProtectedAngleGradient, actual []float32) []float64 {
	out := make([]float64, len(protected))
	actual64 := float32ToFloat64AOQTV9(actual)
	for i, gradient := range protected {
		out[i] = aoqtV9Dot(float32ToFloat64AOQTV9(gradient.Grad), actual64)
	}
	return out
}

func aoqtV9DerivativesNonPositive(values []float64) bool {
	for _, value := range values {
		if !finiteAOQTV9(value) || value > AOQTV9ProtectedDerivativeTolerance {
			return false
		}
	}
	return true
}

func float32ToFloat64AOQTV9(values []float32) []float64 {
	out := make([]float64, len(values))
	for i, value := range values {
		out[i] = float64(value)
	}
	return out
}

func finiteAOQTV9(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func float64SlicesEqualAOQTV9(a, b []float64) bool {
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

func WriteAOQTV9ActiveSetNullspaceScreenReceipt(path string, receipt AOQTSidecarV9ActiveSetNullspaceScreenReceipt) error {
	if receipt.Binding.IOReport.ManifestPath != "" || receipt.Binding.IOReport.RowsJSONLPath != "" || receipt.Binding.PreflightPath != "" || receipt.Binding.CanonicalV8ReceiptPath != "" {
		if err := receipt.ValidateBound(); err != nil {
			return err
		}
	} else if err := receipt.Validate(); err != nil {
		return err
	}
	if strings.TrimSpace(path) == "" {
		return fmt.Errorf("AOQT V9 active-set nullspace receipt output path is required")
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
			return fmt.Errorf("AOQT V9 active-set nullspace receipt output %q already exists", path)
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
