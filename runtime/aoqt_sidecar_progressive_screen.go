package eosruntime

// V7-r6 is a train-only, probe-only screen.  It deliberately has no optimizer
// path and no package/metrics output path.  The screen consumes the already
// authorized V7-r4 endpoint set, binds the V7-r5 receipt and materialized fold,
// and evaluates only the fixed 31 endpoint ordinals on two deterministic
// Horvitz--Thompson samples, followed by an optional full-fold top-K replay.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
)

const (
	AOQTSidecarV7R6ProgressiveScreenSchema                 = "eos.aoqt.v7_r6_progressive_screen.v1"
	AOQTSidecarOptimizerModeV7R6ProgressiveScreen          = "aoqt-v7-r6-progressive-screen-v1"
	AOQTSidecarOptimizerModeV7R6ProgressiveTrainOnlyScreen = AOQTSidecarOptimizerModeV7R6ProgressiveScreen

	// The seed is a contract constant. Fold/manifest provenance is mixed into the
	// ordering key, so the same fold reconstructs the same samples while receipt
	// outcome changes cannot silently reshuffle the population.
	AOQTV7R6ProgressiveScreenSeed                  = "aoqt-v7-r6-progressive-screen-seed-v1"
	AOQTV7R6ProgressiveScreenStratificationVersion = "aoqt-v7-r6-semantic-seven-strata-v1"
	AOQTV7R6ProgressiveScreenWeightingVersion      = "horvitz_thompson_without_replacement_v1"
	AOQTV7R6ProgressiveScreenScheduleVersion       = "aoqt-v7-r6-r4-endpoints-schedule-v1"
	AOQTV7R6ProgressiveScreenRankingVersion        = "lexicographic_feasibility_violation_then_stage_delta_no_budget_tradeoff_v3"
	AOQTV7R6ProgressiveScreenPolicyVersion         = "r4-bound-policy-and-six-component-positive-activation-v1"
	AOQTV7R6ProgressiveScreenDenseGateStatus       = "deferred_to_full_train_verification_v1"
	AOQTV7R6CanonicalR4ReceiptFileSHA256           = "ae0950589be98ba35cadb8ec9097963019a7eeeaf3951db400b4c08986203844"
	AOQTV7R6CanonicalR5ReceiptFileSHA256           = "30e1fe48788ba8c23172ad01e60c336e0e2df10fc5efd0e61c4d4e7255ba36d7"
	AOQTV7R6CanonicalR5ReceiptSHA256               = "9d2dc038f50d20c0672c01e1f0fd174dd477d9691d26acf7c222445a88a271a0"

	// The two progressive stages each use one identity baseline plus the fixed
	// 31-endpoint schedule.  Full-objective top-K verification adds one fresh
	// identity baseline and one candidate evaluation for each of K=6 selected
	// endpoints.  A caller may explicitly stop after the two stage receipts;
	// that short form is accounted for as 64 rather than being mislabeled 71.
	AOQTV7R6ProgressiveScreenStageObjectiveEvaluations     = 32
	AOQTV7R6ProgressiveScreenFullObjectiveEvaluations      = 7
	AOQTV7R6ProgressiveScreenStageOnlyObjectiveEvaluations = 64
	AOQTV7R6ProgressiveScreenObjectiveEvaluations          = 71 // stage-only 64 + full top-K 7.
	AOQTV7R6ProgressiveScreenFullWeightingVersion          = "unweighted_complete_fold_v1"

	// These are the exact materialized fold populations bound by the Luna
	// design.  The constructor is intentionally fail-closed if a fold does not
	// contain exactly these seven semantic populations.
	AOQTV7R6ProgressiveScreenPopulationTotal = 10942

	AOQTV7R6ProgressiveScreenS512          = 512
	AOQTV7R6ProgressiveScreenS1024         = 1024
	AOQTV7R6ProgressiveScreenSelectedK     = 6
	AOQTV7R6ProgressiveScreenStratumCount  = 7
	AOQTV7R6ProgressiveScreenEndpointCount = 31
)

// Fixed per-stratum quotas.  S1024 is the larger prefix of the same ordered
// semantic populations, so S512 is a strict subset of S1024.  The quotas are
// intentionally asymmetric: the NFCorpus boundary workload is oversampled,
// while the six dataset/quantization strata retain their preregistered caps.
var aoqtV7R6ProgressiveScreenQuotas512 = [...]int{146, 148, 39, 39, 22, 22, 96}
var aoqtV7R6ProgressiveScreenQuotas1024 = [...]int{293, 295, 79, 78, 44, 43, 192}

var aoqtV7R6ProgressiveScreenPopulations = [...]int{3575, 3600, 959, 950, 533, 532, 793}

// These labels are intentionally stable.  Stratum membership is semantic and
// disjoint: three datasets x q3/q5 top-10 source, plus the NFCorpus q3
// rank-80..120 boundary source.  Outcomes never influence membership.
var aoqtV7R6ProgressiveScreenStratumNames = [...]string{
	"fiqa_q3_top10",
	"fiqa_q5_top10",
	"nfcorpus_q3_top10",
	"nfcorpus_q5_top10",
	"scifact_q3_top10",
	"scifact_q5_top10",
	"nfcorpus_q3_nf80_120",
}

type aoqtV7R6ProgressiveScreenStratumSpec struct {
	name       string
	dataset    string
	rowPrefix  string
	source     string
	guard      string
	boundary   bool
	population int
	quota512   int
	quota1024  int
}

var aoqtV7R6ProgressiveScreenStratumSpecs = [...]aoqtV7R6ProgressiveScreenStratumSpec{
	{name: "fiqa_q3_top10", dataset: "fiqa", rowPrefix: "fiqa.q3.top10.", source: "q3_top10", guard: "top10_guard", population: 3575, quota512: 146, quota1024: 293},
	{name: "fiqa_q5_top10", dataset: "fiqa", rowPrefix: "fiqa.q5.top10.", source: "q5_top10", guard: "top10_guard", population: 3600, quota512: 148, quota1024: 295},
	{name: "nfcorpus_q3_top10", dataset: "nfcorpus", rowPrefix: "nfcorpus.q3.top10.", source: "q3_top10", guard: "top10_guard", population: 959, quota512: 39, quota1024: 79},
	{name: "nfcorpus_q5_top10", dataset: "nfcorpus", rowPrefix: "nfcorpus.q5.top10.", source: "q5_top10", guard: "top10_guard", population: 950, quota512: 39, quota1024: 78},
	{name: "scifact_q3_top10", dataset: "scifact", rowPrefix: "scifact.q3.top10.", source: "q3_top10", guard: "top10_guard", population: 533, quota512: 22, quota1024: 44},
	{name: "scifact_q5_top10", dataset: "scifact", rowPrefix: "scifact.q5.top10.", source: "q5_top10", guard: "top10_guard", population: 532, quota512: 22, quota1024: 43},
	{name: "nfcorpus_q3_nf80_120", dataset: "nfcorpus", rowPrefix: "nfcorpus.q3.nf80_120.", source: "nf_boundary80_120", guard: "nf_boundary80_120_guard", boundary: true, population: 793, quota512: 96, quota1024: 192},
}

var aoqtV7R6CanonicalEligibleEndpointOrdinals = [...]int{
	8, 39, 41, 51, 53, 73, 75, 80, 81, 83, 94, 125, 139, 141, 143,
	165, 167, 216, 241, 244, 246, 248, 283, 294, 308, 314, 318, 320,
	321, 337, 339,
}

func aoqtV7R6OrderedRowIDSHA256(rowIDs []string) string {
	h := sha256.New()
	for _, id := range rowIDs {
		_, _ = h.Write([]byte(id))
		_, _ = h.Write([]byte{'\n'})
	}
	return hex.EncodeToString(h.Sum(nil))
}

// Public aliases keep the contract discoverable for callers that use either
// the full design name or the shorter AOQT screen terminology.
const (
	AOQTSidecarProgressiveScreenSchema        = AOQTSidecarV7R6ProgressiveScreenSchema
	AOQTSidecarProgressiveScreenOptimizerMode = AOQTSidecarOptimizerModeV7R6ProgressiveScreen
	AOQTV7R6ProgressiveScreenSample512        = AOQTV7R6ProgressiveScreenS512
	AOQTV7R6ProgressiveScreenSample1024       = AOQTV7R6ProgressiveScreenS1024
	AOQTV7R6ProgressiveScreenK                = AOQTV7R6ProgressiveScreenSelectedK
)

func AOQTV7R6CanonicalEligibleEndpointOrdinals() []int {
	return append([]int(nil), aoqtV7R6CanonicalEligibleEndpointOrdinals[:]...)
}

func AOQTV7R6ProgressiveScreenStratumNames() []string {
	return append([]string(nil), aoqtV7R6ProgressiveScreenStratumNames[:]...)
}

func AOQTV7R6ProgressiveScreenQuotas512() []int {
	return append([]int(nil), aoqtV7R6ProgressiveScreenQuotas512[:]...)
}

func AOQTV7R6ProgressiveScreenQuotas1024() []int {
	return append([]int(nil), aoqtV7R6ProgressiveScreenQuotas1024[:]...)
}

// AOQTV7R6ProgressiveScreenPopulations returns the exact materialized-fold
// population expected for each semantic stratum, in the same order as the
// quota and sample metadata helpers.
func AOQTV7R6ProgressiveScreenPopulations() []int {
	return append([]int(nil), aoqtV7R6ProgressiveScreenPopulations[:]...)
}

// AOQTSidecarV7R6ProgressiveScreenConfig is accepted by both the in-memory
// test/runtime API and the file-backed CLI API.  Set may be supplied directly
// for bounded unit tests; file paths are required for a real bound screen.
type AOQTSidecarV7R6ProgressiveScreenConfig struct {
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

	CanonicalR4Receipt       *AOQTSidecarActualCoordinateProbeReceipt
	CanonicalR4ReceiptPath   string
	CanonicalR4ReceiptSHA256 string
	CanonicalR5Receipt       *AOQTSidecarActualCoordinateTrainReceipt
	CanonicalR5ReceiptPath   string
	CanonicalR5ReceiptSHA256 string

	// Mode and probe-only are explicit to prevent accidental use as a training
	// fallback.  Empty Mode is normalized to the canonical screen mode.
	Mode      string
	ProbeOnly bool

	// FullVerification explicitly enables the fixed-K full-objective replay.
	// The default probe command performs it (71 evaluations); callers that need
	// the receipt-only form set StopAfterReceipts, which is still probe-only and
	// is accounted for as 64 evaluations.
	FullVerification  bool
	StopAfterReceipts bool
}

// Short alias retained for source compatibility with screen-oriented callers.
type AOQTSidecarProgressiveScreenConfig = AOQTSidecarV7R6ProgressiveScreenConfig

type AOQTSidecarV7R6ProgressiveScreenStratum struct {
	Name                      string   `json:"name"`
	Index                     int      `json:"index"`
	Population                int      `json:"population"`
	QuotaS512                 int      `json:"quota_s512"`
	QuotaS1024                int      `json:"quota_s1024"`
	InclusionProbabilityS512  float64  `json:"inclusion_probability_s512"`
	InclusionProbabilityS1024 float64  `json:"inclusion_probability_s1024"`
	RowIDSHA256               string   `json:"row_ids_sha256"`
	OrderedRowIDSHA256        string   `json:"ordered_row_ids_sha256"`
	SampleRowIDs              []string `json:"sample_row_ids,omitempty"`
}

type AOQTSidecarV7R6ProgressiveScreenSample struct {
	Name           string                                    `json:"name"`
	RowCount       int                                       `json:"row_count"`
	RowIDs         []string                                  `json:"row_ids"`
	RowIDSHA256    string                                    `json:"row_ids_sha256"`
	Strata         []AOQTSidecarV7R6ProgressiveScreenStratum `json:"strata"`
	StrictSubsetOf string                                    `json:"strict_subset_of,omitempty"`
}

type AOQTSidecarV7R6ProgressiveScreenWeight struct {
	Stratum               string  `json:"stratum"`
	Population            int     `json:"population"`
	SampleCount           int     `json:"sample_count"`
	InclusionProbability  float64 `json:"inclusion_probability"`
	HorvitzThompsonFactor float64 `json:"horvitz_thompson_factor"`
}

type AOQTSidecarV7R6ProgressiveScreenStageEndpoint struct {
	BaselineLoss          float32                        `json:"baseline_loss"`
	BaselineLossFinite    bool                           `json:"baseline_loss_finite"`
	CandidateLoss         float32                        `json:"candidate_loss"`
	CandidateLossFinite   bool                           `json:"candidate_loss_finite"`
	BaselineComponents    AOQTSidecarObjectiveComponents `json:"baseline_components"`
	CandidateComponents   AOQTSidecarObjectiveComponents `json:"candidate_components"`
	ComponentDeltas       AOQTSidecarObjectiveComponents `json:"component_deltas"`
	CandidateActivation   AOQTSidecarObjectiveActivation `json:"candidate_activation"`
	BaselineActivation    AOQTSidecarObjectiveActivation `json:"baseline_activation"`
	TotalDelta            float32                        `json:"total_delta"`
	TotalDeltaFinite      bool                           `json:"total_delta_finite"`
	ActualDirectionSHA256 string                         `json:"actual_direction_sha256"`
	ActualDirectionMaxAbs float32                        `json:"actual_direction_max_abs"`
	ActualDirection       []float32                      `json:"actual_direction"`
	AngleMoved            bool                           `json:"angle_moved"`
	Q3GainEligible        bool                           `json:"q3_gain_eligible"`
	ProtectedEligible     bool                           `json:"protected_eligible"`
	TransactionEligible   bool                           `json:"transaction_eligible"`
	Reason                string                         `json:"reason"`
	CandidateError        string                         `json:"candidate_error,omitempty"`
}

type AOQTSidecarV7R6ProgressiveScreenEndpoint struct {
	Ordinal                        int                                           `json:"ordinal"`
	Kind                           string                                        `json:"kind"`
	CoordinateIndex                int                                           `json:"coordinate_index"`
	AnalyticDirection              int                                           `json:"analytic_direction"`
	Indices                        []int                                         `json:"indices"`
	Directions                     []int                                         `json:"directions"`
	GlobalSign                     int                                           `json:"global_sign"`
	Magnitude                      float32                                       `json:"magnitude"`
	RequestedDirectionSHA256       string                                        `json:"requested_direction_sha256"`
	CanonicalActualDirectionSHA256 string                                        `json:"canonical_actual_direction_sha256"`
	S512                           AOQTSidecarV7R6ProgressiveScreenStageEndpoint `json:"s512"`
	S1024                          AOQTSidecarV7R6ProgressiveScreenStageEndpoint `json:"s1024"`
	RankS512                       int                                           `json:"rank_s512"`
	RankS1024                      int                                           `json:"rank_s1024"`
}

type AOQTSidecarV7R6ProgressiveScreenStage struct {
	Name                       string                                   `json:"name"`
	Sample                     AOQTSidecarV7R6ProgressiveScreenSample   `json:"sample"`
	Weights                    []AOQTSidecarV7R6ProgressiveScreenWeight `json:"weights"`
	BaselineLoss               float32                                  `json:"baseline_loss"`
	BaselineLossFinite         bool                                     `json:"baseline_loss_finite"`
	BaselineComponents         AOQTSidecarObjectiveComponents           `json:"baseline_components"`
	BaselineActivation         AOQTSidecarObjectiveActivation           `json:"baseline_activation"`
	BaselineWeightedActivation AOQTSidecarV7R6WeightedActivation        `json:"baseline_weighted_activation"`
	ObjectiveEvaluationCount   int                                      `json:"objective_evaluation_count"`
	EndpointOrdinals           []int                                    `json:"endpoint_ordinals"`
	Ranking                    []int                                    `json:"ranking"`
	RankingSHA256              string                                   `json:"ranking_sha256"`
}

// AOQTSidecarV7R6ProgressiveScreenFullObjectiveEndpoint is the independent
// full-objective replay for one selected endpoint.  It deliberately retains
// the same complete six-component evidence as a stage endpoint instead of
// substituting a q3-only or scalar-budget score.
type AOQTSidecarV7R6ProgressiveScreenFullObjectiveEndpoint struct {
	Ordinal         int                                           `json:"ordinal"`
	Evidence        AOQTSidecarV7R6ProgressiveScreenStageEndpoint `json:"evidence"`
	Feasible        bool                                          `json:"feasible"`
	DenseGateStatus string                                        `json:"dense_gate_status"`
}

// AOQTSidecarV7R6ProgressiveScreenFullVerification records the optional
// fixed-K full-objective replay.  Dense preservation is intentionally not
// claimed here: that gate belongs to the later full-training verification.
type AOQTSidecarV7R6ProgressiveScreenFullVerification struct {
	Enabled                  bool                                                    `json:"enabled"`
	Completed                bool                                                    `json:"completed"`
	ObjectiveEvaluationCount int                                                     `json:"objective_evaluation_count"`
	RowCount                 int                                                     `json:"row_count"`
	WeightingVersion         string                                                  `json:"weighting_version"`
	HTFactorsApplied         bool                                                    `json:"ht_factors_applied"`
	EndpointOrdinals         []int                                                   `json:"endpoint_ordinals"`
	BaselineLoss             float32                                                 `json:"baseline_loss"`
	BaselineLossFinite       bool                                                    `json:"baseline_loss_finite"`
	BaselineComponents       AOQTSidecarObjectiveComponents                          `json:"baseline_components"`
	BaselineActivation       AOQTSidecarObjectiveActivation                          `json:"baseline_activation"`
	Endpoints                []AOQTSidecarV7R6ProgressiveScreenFullObjectiveEndpoint `json:"endpoints"`
	DenseGateStatus          string                                                  `json:"dense_gate_status"`
	DenseGateDeferred        bool                                                    `json:"dense_gate_deferred"`
}

type AOQTSidecarV7R6WeightedActivation struct {
	Q3GainEligiblePairs         float64 `json:"q3_gain_eligible_pairs"`
	Q3GainContributingPairs     float64 `json:"q3_gain_contributing_pairs"`
	Q3OrderGuardPairs           float64 `json:"q3_order_guard_pairs"`
	Q3OrderGuardContributing    float64 `json:"q3_order_guard_contributing_pairs"`
	Q3ScoreDistillCount         float64 `json:"q3_score_distill_count"`
	Q5OrderGuardPairs           float64 `json:"q5_order_guard_pairs"`
	Q5OrderGuardContributing    float64 `json:"q5_order_guard_contributing_pairs"`
	Q5ScoreDistillCount         float64 `json:"q5_score_distill_count"`
	NFBoundaryGuardPairs        float64 `json:"nf_boundary_guard_pairs"`
	NFBoundaryGuardContributing float64 `json:"nf_boundary_guard_contributing_pairs"`
}

type AOQTSidecarV7R6ProgressiveScreenBinding struct {
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
}

type AOQTSidecarV7R6ProgressiveScreenReceipt struct {
	Schema                           string                                           `json:"schema"`
	Mode                             string                                           `json:"mode"`
	Required                         bool                                             `json:"required"`
	ProbeOnly                        bool                                             `json:"probe_only"`
	Passed                           bool                                             `json:"passed"`
	FailureReason                    string                                           `json:"failure_reason,omitempty"`
	StratificationVersion            string                                           `json:"stratification_version"`
	WeightingVersion                 string                                           `json:"weighting_version"`
	ScheduleVersion                  string                                           `json:"schedule_version"`
	RankingVersion                   string                                           `json:"ranking_version"`
	PolicyVersion                    string                                           `json:"policy_version"`
	SelectionSeedSHA256              string                                           `json:"selection_seed_sha256"`
	R4EndpointSetSHA256              string                                           `json:"r4_endpoint_set_sha256"`
	R4EligibleEndpointOrdinals       []int                                            `json:"r4_eligible_endpoint_ordinals"`
	R5ReceiptSHA256                  string                                           `json:"r5_receipt_sha256"`
	FoldInputRowIDSHA256             string                                           `json:"fold_input_row_ids_sha256"`
	FoldInputRowsSHA256              string                                           `json:"fold_input_rows_sha256"`
	CanonicalR4ScheduleSHA256        string                                           `json:"canonical_r4_schedule_sha256"`
	CanonicalR4RankingSHA256         string                                           `json:"canonical_r4_ranking_sha256"`
	CanonicalR4PolicySHA256          string                                           `json:"canonical_r4_policy_sha256"`
	CanonicalObjectiveContractSHA256 string                                           `json:"canonical_objective_contract_sha256"`
	CanonicalEligibilityPolicySHA256 string                                           `json:"canonical_eligibility_policy_sha256"`
	StageS512                        AOQTSidecarV7R6ProgressiveScreenStage            `json:"stage_s512"`
	StageS1024                       AOQTSidecarV7R6ProgressiveScreenStage            `json:"stage_s1024"`
	FullVerification                 AOQTSidecarV7R6ProgressiveScreenFullVerification `json:"full_verification"`
	Endpoints                        []AOQTSidecarV7R6ProgressiveScreenEndpoint       `json:"endpoints"`
	SelectedEndpointOrdinals         []int                                            `json:"selected_endpoint_ordinals"`
	SelectionK                       int                                              `json:"selection_k"`
	ObjectiveEvaluationCount         int                                              `json:"objective_evaluation_count"`
	SelectionFeasible                bool                                             `json:"selection_feasible"`
	SelectionFeasibleCount           int                                              `json:"selection_feasible_count"`
	SelectionFallbackUsed            bool                                             `json:"selection_fallback_used"`
	FullVerificationFeasible         bool                                             `json:"full_verification_feasible"`
	FullVerificationFeasibleCount    int                                              `json:"full_verification_feasible_count"`
	DenseGateStatus                  string                                           `json:"dense_gate_status"`
	DenseGateDeferred                bool                                             `json:"dense_gate_deferred"`
	DenseMaxAbsDeltaTolerance        float64                                          `json:"dense_max_abs_delta_tolerance"`
	Binding                          AOQTSidecarV7R6ProgressiveScreenBinding          `json:"binding"`
	QualityClaim                     bool                                             `json:"quality_claim"`
	ReleaseClaim                     bool                                             `json:"release_claim"`
	OfficialClaim                    bool                                             `json:"official_claim"`
	OfficialHeldoutGate              bool                                             `json:"official_heldout_gate"`
	CommercialClaim                  bool                                             `json:"commercial_claim"`
}

// Short aliases.
type AOQTSidecarProgressiveScreenReceipt = AOQTSidecarV7R6ProgressiveScreenReceipt
type AOQTSidecarProgressiveScreenStage = AOQTSidecarV7R6ProgressiveScreenStage
type AOQTSidecarProgressiveScreenStageEndpoint = AOQTSidecarV7R6ProgressiveScreenStageEndpoint
type AOQTSidecarProgressiveScreenEndpoint = AOQTSidecarV7R6ProgressiveScreenEndpoint
type AOQTSidecarProgressiveScreenStratum = AOQTSidecarV7R6ProgressiveScreenStratum
type AOQTSidecarProgressiveScreenWeight = AOQTSidecarV7R6ProgressiveScreenWeight
type AOQTSidecarProgressiveScreenBinding = AOQTSidecarV7R6ProgressiveScreenBinding
type AOQTSidecarProgressiveScreenWeightedActivation = AOQTSidecarV7R6WeightedActivation
type AOQTSidecarProgressiveScreenFullObjectiveEndpoint = AOQTSidecarV7R6ProgressiveScreenFullObjectiveEndpoint
type AOQTSidecarProgressiveScreenFullVerification = AOQTSidecarV7R6ProgressiveScreenFullVerification

type AOQTSidecarV7R6ProgressiveScreenResult struct {
	Receipt         AOQTSidecarV7R6ProgressiveScreenReceipt
	ReceiptSHA256   string
	ReceiptJSONPath string
}

type AOQTSidecarProgressiveScreenResult = AOQTSidecarV7R6ProgressiveScreenResult

type aoqtV7R6EndpointPolicyEvidence struct {
	Q3GainEligible      bool
	ProtectedEligible   bool
	TransactionEligible bool
	Reason              string
}

func (r AOQTSidecarV7R6ProgressiveScreenReceipt) SHA256() (string, error) {
	return aoqtCanonicalSHA256(r)
}

func (r AOQTSidecarV7R6ProgressiveScreenReceipt) Validate() error {
	return r.validate(false)
}

func (r AOQTSidecarV7R6ProgressiveScreenReceipt) ValidateBound() error {
	if err := r.validate(true); err != nil {
		return err
	}
	if r.Binding.SplitBinding == nil {
		return fmt.Errorf("AOQT V7-r6 split binding is required")
	}
	if err := r.Binding.SplitBinding.Validate(); err != nil {
		return fmt.Errorf("AOQT V7-r6 split binding: %w", err)
	}
	if err := validateAOQTMetricInputs(r.Binding.Inputs); err != nil {
		return fmt.Errorf("AOQT V7-r6 binding inputs: %w", err)
	}
	if r.Binding.IOReport.Schema != AOQTSidecarCalibrationIOReportSchema ||
		r.Binding.IOReport.ManifestSHA256 != r.Binding.Inputs.DatasetManifestSHA256 ||
		r.Binding.IOReport.RowsSHA256 != r.FoldInputRowsSHA256 {
		return fmt.Errorf("AOQT V7-r6 IO report does not match bound fold inputs")
	}
	if r.Binding.FoldID != r.Binding.SplitBinding.FoldID {
		return fmt.Errorf("AOQT V7-r6 fold id does not match split binding")
	}
	if r.Binding.IOReport.ManifestPath == "" || r.Binding.IOReport.RowsJSONLPath == "" {
		return fmt.Errorf("AOQT V7-r6 bound IO report source paths are required")
	}
	if r.Binding.PreflightPath == "" || r.Binding.PreflightSHA256 == "" {
		return fmt.Errorf("AOQT V7-r6 bound preflight path/hash are required")
	}
	if err := validateAOQTSHA256(r.Binding.PreflightSHA256, "AOQT V7-r6 preflight sha256"); err != nil {
		return err
	}
	for _, item := range []struct {
		name  string
		value string
	}{
		{"canonical r4 receipt hash", r.Binding.CanonicalR4ReceiptSHA256},
		{"canonical r4 receipt file hash", r.Binding.CanonicalR4ReceiptFileSHA256},
		{"canonical r5 receipt hash", r.Binding.CanonicalR5ReceiptSHA256},
		{"canonical r5 receipt file hash", r.Binding.CanonicalR5ReceiptFileSHA256},
	} {
		if err := validateAOQTSHA256(item.value, "AOQT V7-r6 "+item.name); err != nil {
			return err
		}
	}
	if r.Binding.CanonicalR4ReceiptFileSHA256 != AOQTV7R6CanonicalR4ReceiptFileSHA256 {
		return fmt.Errorf("AOQT V7-r6 canonical r4 receipt file is not the pinned authorization")
	}
	if r.Binding.CanonicalR5ReceiptFileSHA256 != AOQTV7R6CanonicalR5ReceiptFileSHA256 || r.Binding.CanonicalR5ReceiptSHA256 != AOQTV7R6CanonicalR5ReceiptSHA256 {
		return fmt.Errorf("AOQT V7-r6 canonical r5 receipt is not the pinned authorization")
	}
	return rereadAOQTV7R6Binding(r)
}

func (r AOQTSidecarV7R6ProgressiveScreenReceipt) validate(requireBinding bool) error {
	if r.Schema != AOQTSidecarV7R6ProgressiveScreenSchema {
		return fmt.Errorf("AOQT V7-r6 receipt schema %q is unsupported, want %q", r.Schema, AOQTSidecarV7R6ProgressiveScreenSchema)
	}
	if r.Mode != AOQTSidecarOptimizerModeV7R6ProgressiveScreen {
		return fmt.Errorf("AOQT V7-r6 mode %q is unsupported", r.Mode)
	}
	if !r.Required || !r.ProbeOnly {
		return fmt.Errorf("AOQT V7-r6 receipt must be required and probe_only")
	}
	if r.QualityClaim || r.ReleaseClaim || r.OfficialClaim || r.OfficialHeldoutGate || r.CommercialClaim {
		return fmt.Errorf("AOQT V7-r6 claims and official heldout gate must be false")
	}
	if r.StratificationVersion != AOQTV7R6ProgressiveScreenStratificationVersion || r.WeightingVersion != AOQTV7R6ProgressiveScreenWeightingVersion || r.ScheduleVersion != AOQTV7R6ProgressiveScreenScheduleVersion || r.RankingVersion != AOQTV7R6ProgressiveScreenRankingVersion || r.PolicyVersion != AOQTV7R6ProgressiveScreenPolicyVersion {
		return fmt.Errorf("AOQT V7-r6 contract version binding is unsupported")
	}
	policy := aoqtV7R6CanonicalEligibilityPolicy()
	if err := validateAOQTSidecarCandidateEligibilityPolicy(policy); err != nil {
		return err
	}
	canonicalPolicySHA, err := aoqtActualCoordinatePolicySHA256(policy)
	if err != nil {
		return err
	}
	if r.DenseGateStatus != AOQTV7R6ProgressiveScreenDenseGateStatus || !r.DenseGateDeferred || !isFinite64(r.DenseMaxAbsDeltaTolerance) || r.DenseMaxAbsDeltaTolerance <= 0 {
		return fmt.Errorf("AOQT V7-r6 dense gate must remain explicitly deferred")
	}
	if err := validateAOQTSHA256(r.SelectionSeedSHA256, "AOQT V7-r6 selection seed sha256"); err != nil {
		return err
	}
	if err := validateAOQTSHA256(r.R4EndpointSetSHA256, "AOQT V7-r6 r4 endpoint set sha256"); err != nil {
		return err
	}
	if err := validateAOQTSHA256(r.R5ReceiptSHA256, "AOQT V7-r6 r5 receipt sha256"); err != nil {
		return err
	}
	for name, value := range map[string]string{
		"fold input row ids sha256":           r.FoldInputRowIDSHA256,
		"fold input rows sha256":              r.FoldInputRowsSHA256,
		"canonical r4 schedule sha256":        r.CanonicalR4ScheduleSHA256,
		"canonical r4 ranking sha256":         r.CanonicalR4RankingSHA256,
		"canonical r4 policy sha256":          r.CanonicalR4PolicySHA256,
		"canonical objective contract sha256": r.CanonicalObjectiveContractSHA256,
		"canonical eligibility policy sha256": r.CanonicalEligibilityPolicySHA256,
	} {
		if err := validateAOQTSHA256(value, "AOQT V7-r6 "+name); err != nil {
			return err
		}
	}
	if r.CanonicalEligibilityPolicySHA256 != canonicalPolicySHA {
		return fmt.Errorf("AOQT V7-r6 eligibility policy must be the canonical joint-budget policy")
	}
	if !reflect.DeepEqual(r.R4EligibleEndpointOrdinals, aoqtV7R6CanonicalEligibleEndpointOrdinals[:]) {
		return fmt.Errorf("AOQT V7-r6 eligible endpoint ordinals do not match canonical exact 31")
	}
	if r.SelectionK != AOQTV7R6ProgressiveScreenSelectedK || (r.ObjectiveEvaluationCount != AOQTV7R6ProgressiveScreenStageOnlyObjectiveEvaluations && r.ObjectiveEvaluationCount != AOQTV7R6ProgressiveScreenObjectiveEvaluations) {
		return fmt.Errorf("AOQT V7-r6 selection/evaluation accounting is not fixed")
	}
	if err := validateAOQTV7R6Sample(r.StageS512.Sample, false); err != nil {
		return err
	}
	if err := validateAOQTV7R6Sample(r.StageS1024.Sample, true); err != nil {
		return err
	}
	if len(r.StageS512.Sample.RowIDs) >= len(r.StageS1024.Sample.RowIDs) || !isStrictStringSubset(r.StageS512.Sample.RowIDs, r.StageS1024.Sample.RowIDs) {
		return fmt.Errorf("AOQT V7-r6 S512 must be a strict subset of S1024")
	}
	if r.StageS512.ObjectiveEvaluationCount != AOQTV7R6ProgressiveScreenStageObjectiveEvaluations || r.StageS1024.ObjectiveEvaluationCount != AOQTV7R6ProgressiveScreenStageObjectiveEvaluations {
		return fmt.Errorf("AOQT V7-r6 stage objective evaluation accounting is invalid")
	}
	if err := validateAOQTV7R6Stage(r.StageS512, false); err != nil {
		return err
	}
	if err := validateAOQTV7R6Stage(r.StageS1024, true); err != nil {
		return err
	}
	if len(r.Endpoints) != AOQTV7R6ProgressiveScreenEndpointCount || len(r.SelectedEndpointOrdinals) != AOQTV7R6ProgressiveScreenSelectedK {
		return fmt.Errorf("AOQT V7-r6 endpoint/selection counts are invalid")
	}
	seen := map[int]bool{}
	rank512At := make(map[int]int, len(r.StageS512.Ranking))
	for i, ordinal := range r.StageS512.Ranking {
		rank512At[ordinal] = i + 1
	}
	rank1024At := make(map[int]int, len(r.StageS1024.Ranking))
	for i, ordinal := range r.StageS1024.Ranking {
		rank1024At[ordinal] = i + 1
	}
	for i, endpoint := range r.Endpoints {
		if err := validateAOQTV7R6Endpoint(endpoint, i, policy); err != nil {
			return err
		}
		if seen[endpoint.Ordinal] {
			return fmt.Errorf("AOQT V7-r6 duplicate endpoint ordinal %d", endpoint.Ordinal)
		}
		seen[endpoint.Ordinal] = true
		if endpoint.RankS512 != rank512At[endpoint.Ordinal] || endpoint.RankS1024 != rank1024At[endpoint.Ordinal] {
			return fmt.Errorf("AOQT V7-r6 endpoint %d rank fields do not match stage rankings", endpoint.Ordinal)
		}
		if endpoint.S512.BaselineLoss != r.StageS512.BaselineLoss || endpoint.S512.BaselineComponents != r.StageS512.BaselineComponents || !reflect.DeepEqual(endpoint.S512.BaselineActivation, r.StageS512.BaselineActivation) {
			return fmt.Errorf("AOQT V7-r6 endpoint %d S512 baseline does not match stage baseline", endpoint.Ordinal)
		}
		if endpoint.S1024.BaselineLoss != r.StageS1024.BaselineLoss || endpoint.S1024.BaselineComponents != r.StageS1024.BaselineComponents || !reflect.DeepEqual(endpoint.S1024.BaselineActivation, r.StageS1024.BaselineActivation) {
			return fmt.Errorf("AOQT V7-r6 endpoint %d S1024 baseline does not match stage baseline", endpoint.Ordinal)
		}
	}
	recomputedRank512, err := aoqtV7R6RankEndpointsWithPolicy(r.Endpoints, "S512", policy)
	if err != nil {
		return err
	}
	recomputedRank1024, err := aoqtV7R6RankEndpointsWithPolicy(r.Endpoints, "S1024", policy)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(r.StageS512.Ranking, recomputedRank512) || !reflect.DeepEqual(r.StageS1024.Ranking, recomputedRank1024) {
		return fmt.Errorf("AOQT V7-r6 persisted stage ranking does not match recomputed endpoint evidence")
	}
	recomputedRank512SHA, err := aoqtV7R6RankingSHA(recomputedRank512)
	if err != nil {
		return err
	}
	recomputedRank1024SHA, err := aoqtV7R6RankingSHA(recomputedRank1024)
	if err != nil {
		return err
	}
	if r.StageS512.RankingSHA256 != recomputedRank512SHA || r.StageS1024.RankingSHA256 != recomputedRank1024SHA {
		return fmt.Errorf("AOQT V7-r6 persisted ranking hash does not match recomputed endpoint evidence")
	}
	for _, endpoint := range r.Endpoints {
		if endpoint.RankS512 != indexOfIntAOQT(recomputedRank512, endpoint.Ordinal)+1 || endpoint.RankS1024 != indexOfIntAOQT(recomputedRank1024, endpoint.Ordinal)+1 {
			return fmt.Errorf("AOQT V7-r6 endpoint %d rank fields do not match recomputed ranking", endpoint.Ordinal)
		}
	}
	for _, ordinal := range r.SelectedEndpointOrdinals {
		if !seen[ordinal] {
			return fmt.Errorf("AOQT V7-r6 selected endpoint ordinal %d is not present", ordinal)
		}
	}
	selectionFeasibleCount := 0
	byOrdinal := make(map[int]AOQTSidecarV7R6ProgressiveScreenEndpoint, len(r.Endpoints))
	for _, endpoint := range r.Endpoints {
		byOrdinal[endpoint.Ordinal] = endpoint
	}
	if err := validateAOQTV7R6FullVerification(r.FullVerification, r.SelectedEndpointOrdinals, r.ObjectiveEvaluationCount == AOQTV7R6ProgressiveScreenObjectiveEvaluations, policy, byOrdinal); err != nil {
		return err
	}
	for _, ordinal := range r.SelectedEndpointOrdinals {
		if aoqtV7R6EndpointFeasibleWithPolicy(byOrdinal[ordinal].S1024, policy) {
			selectionFeasibleCount++
		}
	}
	selectionFeasible := selectionFeasibleCount == AOQTV7R6ProgressiveScreenSelectedK
	if r.SelectionFeasible != selectionFeasible {
		return fmt.Errorf("AOQT V7-r6 selection_feasible does not match endpoint feasibility")
	}
	if r.SelectionFeasibleCount != selectionFeasibleCount || r.SelectionFallbackUsed != !selectionFeasible {
		return fmt.Errorf("AOQT V7-r6 fixed-K feasibility/fallback accounting does not match endpoint evidence")
	}
	fullVerificationFeasibleCount := 0
	if r.ObjectiveEvaluationCount == AOQTV7R6ProgressiveScreenObjectiveEvaluations {
		for _, endpoint := range r.FullVerification.Endpoints {
			if aoqtV7R6EndpointFeasibleWithPolicy(endpoint.Evidence, policy) {
				fullVerificationFeasibleCount++
			}
		}
	}
	fullVerificationFeasible := r.ObjectiveEvaluationCount == AOQTV7R6ProgressiveScreenObjectiveEvaluations && fullVerificationFeasibleCount == AOQTV7R6ProgressiveScreenSelectedK
	if r.FullVerificationFeasible != fullVerificationFeasible || r.FullVerificationFeasibleCount != fullVerificationFeasibleCount {
		return fmt.Errorf("AOQT V7-r6 full-verification feasibility accounting does not match endpoint evidence")
	}
	wantPassed := selectionFeasible
	if r.ObjectiveEvaluationCount == AOQTV7R6ProgressiveScreenObjectiveEvaluations {
		// S1024 feasibility is diagnostic for the fixed-K fallback. The
		// authoritative full fold decides pass/fail for the selected six, even
		// when one or more of them was an S1024-infeasible fallback.
		wantPassed = r.FullVerification.Completed && fullVerificationFeasible
	}
	if r.Passed != wantPassed {
		return fmt.Errorf("AOQT V7-r6 passed state does not match screen execution/selection evidence")
	}
	if !reflect.DeepEqual(r.SelectedEndpointOrdinals, r.StageS1024.Ranking[:AOQTV7R6ProgressiveScreenSelectedK]) {
		return fmt.Errorf("AOQT V7-r6 selected endpoints must be the fixed-K S1024 ranking prefix")
	}
	if r.Passed {
		if r.FailureReason != "" {
			return fmt.Errorf("AOQT V7-r6 passed receipt has failure_reason")
		}
	} else if strings.TrimSpace(r.FailureReason) == "" {
		return fmt.Errorf("AOQT V7-r6 failed receipt requires failure_reason")
	}
	if requireBinding {
		if r.Binding.SplitBinding == nil {
			return fmt.Errorf("AOQT V7-r6 bound receipt requires split binding")
		}
		if r.Binding.CanonicalR4ReceiptPath == "" || r.Binding.CanonicalR5ReceiptPath == "" {
			return fmt.Errorf("AOQT V7-r6 bound receipt requires r4/r5 receipt paths")
		}
	}
	return nil
}

func validateAOQTV7R6Stage(stage AOQTSidecarV7R6ProgressiveScreenStage, large bool) error {
	if stage.Name != map[bool]string{false: "S512", true: "S1024"}[large] {
		return fmt.Errorf("AOQT V7-r6 stage name is invalid")
	}
	if (!large && stage.Sample.StrictSubsetOf != "S1024") || (large && stage.Sample.StrictSubsetOf != "") {
		return fmt.Errorf("AOQT V7-r6 %s strict-subset marker is invalid", stage.Name)
	}
	if !stage.BaselineLossFinite || !isFinite32(stage.BaselineLoss) || !aoqtFiniteReceiptComponentsValue(stage.BaselineComponents) || !aoqtV7R6PositiveActivation(stage.BaselineActivation) {
		return fmt.Errorf("AOQT V7-r6 %s baseline evidence is invalid", stage.Name)
	}
	if err := validateAOQTLossMatchesComponents(stage.BaselineLoss, stage.BaselineComponents, "AOQT V7-r6 "+stage.Name+" baseline"); err != nil {
		return err
	}
	if len(stage.Weights) != AOQTV7R6ProgressiveScreenStratumCount {
		return fmt.Errorf("AOQT V7-r6 %s must record all seven stratum weights", stage.Name)
	}
	for i, weight := range stage.Weights {
		if weight.Stratum != aoqtV7R6ProgressiveScreenStratumNames[i] || weight.Population != stage.Sample.Strata[i].Population {
			return fmt.Errorf("AOQT V7-r6 %s weight %d does not bind its stratum", stage.Name, i)
		}
		sampleCount := stage.Sample.Strata[i].QuotaS512
		probability := stage.Sample.Strata[i].InclusionProbabilityS512
		if large {
			sampleCount = stage.Sample.Strata[i].QuotaS1024
			probability = stage.Sample.Strata[i].InclusionProbabilityS1024
		}
		factor, err := AOQTV7R6ProgressiveScreenHorvitzThompsonFactor(weight.Population, sampleCount)
		if err != nil || weight.SampleCount != sampleCount || !isFinite64(weight.InclusionProbability) || math.Abs(weight.InclusionProbability-probability) > 1e-12 || !isFinite64(weight.HorvitzThompsonFactor) || math.Abs(weight.HorvitzThompsonFactor-factor) > 1e-12 {
			return fmt.Errorf("AOQT V7-r6 %s weight %d is not the exact Horvitz-Thompson factor", stage.Name, i)
		}
	}
	if !aoqtV7R6WeightedActivationFinitePositive(stage.BaselineWeightedActivation) {
		return fmt.Errorf("AOQT V7-r6 %s weighted baseline activation is not positive for all six components", stage.Name)
	}
	if len(stage.EndpointOrdinals) != AOQTV7R6ProgressiveScreenEndpointCount || !reflect.DeepEqual(stage.EndpointOrdinals, aoqtV7R6CanonicalEligibleEndpointOrdinals[:]) {
		return fmt.Errorf("AOQT V7-r6 %s endpoint schedule is not canonical", stage.Name)
	}
	if len(stage.Ranking) != AOQTV7R6ProgressiveScreenEndpointCount {
		return fmt.Errorf("AOQT V7-r6 %s ranking count is invalid", stage.Name)
	}
	seen := map[int]bool{}
	for _, ordinal := range stage.Ranking {
		if !containsInt(ordinal, aoqtV7R6CanonicalEligibleEndpointOrdinals[:]) || seen[ordinal] {
			return fmt.Errorf("AOQT V7-r6 %s ranking is not a permutation", stage.Name)
		}
		seen[ordinal] = true
	}
	want, err := aoqtV7R6RankingSHA(stage.Ranking)
	if err != nil || want != stage.RankingSHA256 {
		return fmt.Errorf("AOQT V7-r6 %s ranking hash mismatch", stage.Name)
	}
	return nil
}

func aoqtV7R6EndpointFeasible(endpoint AOQTSidecarV7R6ProgressiveScreenStageEndpoint) bool {
	return aoqtV7R6EndpointFeasibleWithPolicy(endpoint, aoqtV7R6CanonicalEligibilityPolicy())
}

func aoqtV7R6EndpointFeasibleWithPolicy(endpoint AOQTSidecarV7R6ProgressiveScreenStageEndpoint, policy AOQTSidecarCandidateEligibilityPolicy) bool {
	evidence, err := aoqtV7R6RecomputeEndpointPolicy(endpoint, policy)
	return err == nil && evidence.TransactionEligible
}

func aoqtV7R6CanonicalEligibilityPolicy() AOQTSidecarCandidateEligibilityPolicy {
	return AOQTSidecarScoreDistillJointBudgetV1Policy()
}

func validateAOQTV7R6FullVerification(full AOQTSidecarV7R6ProgressiveScreenFullVerification, selected []int, required bool, policy AOQTSidecarCandidateEligibilityPolicy, endpointByOrdinal map[int]AOQTSidecarV7R6ProgressiveScreenEndpoint) error {
	if !full.DenseGateDeferred || full.DenseGateStatus != AOQTV7R6ProgressiveScreenDenseGateStatus {
		return fmt.Errorf("AOQT V7-r6 full verification must preserve deferred dense gate")
	}
	if !required {
		if full.Enabled || full.Completed || full.ObjectiveEvaluationCount != 0 || full.RowCount != 0 || full.WeightingVersion != "" || full.HTFactorsApplied || len(full.EndpointOrdinals) != 0 || len(full.Endpoints) != 0 {
			return fmt.Errorf("AOQT V7-r6 stage-only receipt cannot contain full verification evidence")
		}
		return nil
	}
	policy = normalizedAOQTSidecarCandidateEligibilityPolicy(policy)
	if !full.Enabled || !full.Completed || full.ObjectiveEvaluationCount != AOQTV7R6ProgressiveScreenFullObjectiveEvaluations || full.RowCount != AOQTV7R6ProgressiveScreenPopulationTotal || full.WeightingVersion != AOQTV7R6ProgressiveScreenFullWeightingVersion || full.HTFactorsApplied || len(full.EndpointOrdinals) != AOQTV7R6ProgressiveScreenSelectedK || len(full.Endpoints) != AOQTV7R6ProgressiveScreenSelectedK || !reflect.DeepEqual(full.EndpointOrdinals, selected) {
		return fmt.Errorf("AOQT V7-r6 full verification must contain exactly fixed-K evidence")
	}
	if !full.BaselineLossFinite || !isFinite32(full.BaselineLoss) || !aoqtFiniteReceiptComponentsValue(full.BaselineComponents) || !aoqtV7R6PositiveActivation(full.BaselineActivation) {
		return fmt.Errorf("AOQT V7-r6 full verification baseline evidence is invalid")
	}
	if err := validateAOQTLossMatchesComponents(full.BaselineLoss, full.BaselineComponents, "AOQT V7-r6 full verification baseline"); err != nil {
		return err
	}
	for i, endpoint := range full.Endpoints {
		canonicalEndpoint, hasCanonicalEndpoint := endpointByOrdinal[endpoint.Ordinal]
		if endpoint.Ordinal != selected[i] || endpoint.DenseGateStatus != AOQTV7R6ProgressiveScreenDenseGateStatus {
			return fmt.Errorf("AOQT V7-r6 full verification endpoint %d metadata is invalid", i)
		}
		if endpoint.Evidence.BaselineLoss != full.BaselineLoss || endpoint.Evidence.BaselineComponents != full.BaselineComponents || !reflect.DeepEqual(endpoint.Evidence.BaselineActivation, full.BaselineActivation) {
			return fmt.Errorf("AOQT V7-r6 full verification endpoint %d baseline does not match", i)
		}
		expectedActualSHA := ""
		if hasCanonicalEndpoint {
			expectedActualSHA = canonicalEndpoint.CanonicalActualDirectionSHA256
		}
		recomputed, err := validateAOQTV7R6StageEndpointEvidence(endpoint.Evidence, policy, expectedActualSHA, fmt.Sprintf("AOQT V7-r6 full verification endpoint %d", i))
		if err != nil {
			return err
		}
		if endpoint.Feasible != recomputed.TransactionEligible {
			return fmt.Errorf("AOQT V7-r6 full verification endpoint %d feasible flag does not match recomputed policy", i)
		}
	}
	return nil
}

func validateAOQTV7R6Sample(sample AOQTSidecarV7R6ProgressiveScreenSample, large bool) error {
	wantCount := AOQTV7R6ProgressiveScreenS512
	wantName := "S512"
	if large {
		wantCount = AOQTV7R6ProgressiveScreenS1024
		wantName = "S1024"
	}
	if sample.Name != wantName || sample.RowCount != wantCount || len(sample.RowIDs) != wantCount {
		return fmt.Errorf("AOQT V7-r6 %s sample row count is invalid", sample.Name)
	}
	if sample.RowIDSHA256 != aoqtRowIDSHA256(sample.RowIDs) {
		return fmt.Errorf("AOQT V7-r6 %s sample row id hash mismatch", sample.Name)
	}
	seen := map[string]bool{}
	for _, id := range sample.RowIDs {
		if id == "" || seen[id] {
			return fmt.Errorf("AOQT V7-r6 %s sample row ids are not unique", sample.Name)
		}
		seen[id] = true
	}
	if !sort.StringsAreSorted(sample.RowIDs) {
		return fmt.Errorf("AOQT V7-r6 %s sample row ids must be sorted", sample.Name)
	}
	if len(sample.Strata) != AOQTV7R6ProgressiveScreenStratumCount {
		return fmt.Errorf("AOQT V7-r6 %s must record all seven strata", sample.Name)
	}
	seenByStratum := map[string]bool{}
	union := make([]string, 0, len(sample.RowIDs))
	for i, stratum := range sample.Strata {
		spec := aoqtV7R6ProgressiveScreenStratumSpecs[i]
		if stratum.Index != i || stratum.Name != spec.name || stratum.Population != spec.population {
			return fmt.Errorf("AOQT V7-r6 %s stratum %d is invalid", sample.Name, i)
		}
		wantQuota := spec.quota512
		if large {
			wantQuota = spec.quota1024
		}
		if stratum.QuotaS512 != spec.quota512 || stratum.QuotaS1024 != spec.quota1024 {
			return fmt.Errorf("AOQT V7-r6 %s stratum %d quota is not fixed", sample.Name, i)
		}
		if wantQuota <= 0 || stratum.Population < wantQuota {
			return fmt.Errorf("AOQT V7-r6 %s stratum %d population cannot satisfy quota", sample.Name, i)
		}
		p := stratum.InclusionProbabilityS512
		if large {
			p = stratum.InclusionProbabilityS1024
		}
		if !isFinite64(p) || p <= 0 || p > 1 || math.Abs(p-float64(wantQuota)/float64(stratum.Population)) > 1e-12 {
			return fmt.Errorf("AOQT V7-r6 %s stratum %d inclusion probability is invalid", sample.Name, i)
		}
		if len(stratum.SampleRowIDs) != wantQuota || stratum.RowIDSHA256 != aoqtRowIDSHA256(stratum.SampleRowIDs) {
			return fmt.Errorf("AOQT V7-r6 %s stratum %d sample membership/hash is invalid", sample.Name, i)
		}
		for _, id := range stratum.SampleRowIDs {
			if strings.TrimSpace(id) == "" || seenByStratum[id] {
				return fmt.Errorf("AOQT V7-r6 %s stratum %d sample row ids are not unique", sample.Name, i)
			}
			seenByStratum[id] = true
			union = append(union, id)
		}
		for _, value := range []string{stratum.RowIDSHA256, stratum.OrderedRowIDSHA256} {
			if err := validateAOQTSHA256(value, "AOQT V7-r6 stratum row hash"); err != nil {
				return err
			}
		}
	}
	sort.Strings(union)
	if !reflect.DeepEqual(union, sample.RowIDs) {
		return fmt.Errorf("AOQT V7-r6 %s stratum memberships do not match sample row ids", sample.Name)
	}
	return nil
}

func validateAOQTV7R6Endpoint(endpoint AOQTSidecarV7R6ProgressiveScreenEndpoint, index int, policy AOQTSidecarCandidateEligibilityPolicy) error {
	if endpoint.Ordinal != aoqtV7R6CanonicalEligibleEndpointOrdinals[index] {
		return fmt.Errorf("AOQT V7-r6 endpoint[%d] ordinal %d is not canonical eligible ordinal", index, endpoint.Ordinal)
	}
	if endpoint.Kind != "coordinate" || endpoint.CoordinateIndex < 0 || endpoint.CoordinateIndex >= AOQTSidecarAngleCount || endpoint.AnalyticDirection != -1 && endpoint.AnalyticDirection != 1 || endpoint.GlobalSign != -1 && endpoint.GlobalSign != 1 || len(endpoint.Indices) != 1 || len(endpoint.Directions) != 1 || endpoint.Indices[0] != endpoint.CoordinateIndex || endpoint.Directions[0] != endpoint.AnalyticDirection*endpoint.GlobalSign {
		return fmt.Errorf("AOQT V7-r6 endpoint[%d] coordinate metadata is invalid", index)
	}
	if !isFinite32(endpoint.Magnitude) || endpoint.Magnitude <= 0 {
		return fmt.Errorf("AOQT V7-r6 endpoint[%d] magnitude is invalid", index)
	}
	for name, value := range map[string]string{"requested direction": endpoint.RequestedDirectionSHA256, "canonical actual direction": endpoint.CanonicalActualDirectionSHA256} {
		if err := validateAOQTSHA256(value, "AOQT V7-r6 endpoint "+name); err != nil {
			return err
		}
	}
	requested := make([]float32, AOQTSidecarAngleCount)
	requested[endpoint.CoordinateIndex] = endpoint.Magnitude * float32(endpoint.Directions[0])
	requestedSHA, err := aoqtActualCoordinateVectorSHA256(requested)
	if err != nil || requestedSHA != endpoint.RequestedDirectionSHA256 {
		return fmt.Errorf("AOQT V7-r6 endpoint[%d] requested direction hash does not match coordinate metadata", index)
	}
	if endpoint.RankS512 <= 0 || endpoint.RankS512 > AOQTV7R6ProgressiveScreenEndpointCount || endpoint.RankS1024 <= 0 || endpoint.RankS1024 > AOQTV7R6ProgressiveScreenEndpointCount {
		return fmt.Errorf("AOQT V7-r6 endpoint[%d] rank is invalid", index)
	}
	for _, stage := range []struct {
		name  string
		value AOQTSidecarV7R6ProgressiveScreenStageEndpoint
	}{
		{"S512", endpoint.S512}, {"S1024", endpoint.S1024},
	} {
		if _, err := validateAOQTV7R6StageEndpointEvidence(stage.value, policy, endpoint.CanonicalActualDirectionSHA256, fmt.Sprintf("AOQT V7-r6 endpoint[%d] %s", index, stage.name)); err != nil {
			return err
		}
	}
	return nil
}

func validateAOQTV7R6StageEndpointEvidence(endpoint AOQTSidecarV7R6ProgressiveScreenStageEndpoint, policy AOQTSidecarCandidateEligibilityPolicy, expectedActualDirectionSHA256, label string) (aoqtV7R6EndpointPolicyEvidence, error) {
	if !endpoint.BaselineLossFinite || !endpoint.CandidateLossFinite || !endpoint.TotalDeltaFinite || !isFinite32(endpoint.BaselineLoss) || !isFinite32(endpoint.CandidateLoss) || !isFinite32(endpoint.TotalDelta) {
		return aoqtV7R6EndpointPolicyEvidence{}, fmt.Errorf("%s loss evidence is not finite", label)
	}
	if endpoint.TotalDelta != endpoint.CandidateLoss-endpoint.BaselineLoss || endpoint.ComponentDeltas != aoqtObjectiveComponentDelta(endpoint.BaselineComponents, endpoint.CandidateComponents) {
		return aoqtV7R6EndpointPolicyEvidence{}, fmt.Errorf("%s delta evidence mismatch", label)
	}
	if err := validateAOQTObjectiveActivation(endpoint.BaselineActivation, label+" baseline activation"); err != nil {
		return aoqtV7R6EndpointPolicyEvidence{}, err
	}
	if err := validateAOQTObjectiveActivation(endpoint.CandidateActivation, label+" candidate activation"); err != nil {
		return aoqtV7R6EndpointPolicyEvidence{}, err
	}
	if err := validateAOQTLossMatchesComponents(endpoint.BaselineLoss, endpoint.BaselineComponents, label+" baseline"); err != nil {
		return aoqtV7R6EndpointPolicyEvidence{}, err
	}
	if err := validateAOQTLossMatchesComponents(endpoint.CandidateLoss, endpoint.CandidateComponents, label+" candidate"); err != nil {
		return aoqtV7R6EndpointPolicyEvidence{}, err
	}
	if len(endpoint.ActualDirection) != AOQTSidecarAngleCount || !isFinite32(endpoint.ActualDirectionMaxAbs) || endpoint.ActualDirectionMaxAbs < 0 {
		return aoqtV7R6EndpointPolicyEvidence{}, fmt.Errorf("%s actual direction length/max-abs is invalid", label)
	}
	for i, value := range endpoint.ActualDirection {
		if !isFinite32(value) {
			return aoqtV7R6EndpointPolicyEvidence{}, fmt.Errorf("%s actual direction[%d] is not finite", label, i)
		}
	}
	actualSHA, err := aoqtActualCoordinateVectorSHA256(endpoint.ActualDirection)
	if err != nil || actualSHA != endpoint.ActualDirectionSHA256 || endpoint.ActualDirectionMaxAbs != aoqtMaxAbsFloat32(endpoint.ActualDirection) || endpoint.AngleMoved != (endpoint.ActualDirectionMaxAbs != 0) {
		return aoqtV7R6EndpointPolicyEvidence{}, fmt.Errorf("%s actual direction hash/max-abs mismatch", label)
	}
	if expectedActualDirectionSHA256 != "" && endpoint.ActualDirectionSHA256 != expectedActualDirectionSHA256 {
		return aoqtV7R6EndpointPolicyEvidence{}, fmt.Errorf("%s actual direction is not bound to canonical r4", label)
	}
	if !endpoint.BaselineActivationPositive() || !endpoint.CandidateActivationPositive() {
		return aoqtV7R6EndpointPolicyEvidence{}, fmt.Errorf("%s activation is not positive for all six components", label)
	}
	recomputed, err := aoqtV7R6RecomputeEndpointPolicy(endpoint, policy)
	if err != nil {
		return aoqtV7R6EndpointPolicyEvidence{}, err
	}
	if endpoint.Q3GainEligible != recomputed.Q3GainEligible {
		return aoqtV7R6EndpointPolicyEvidence{}, fmt.Errorf("%s q3_gain_eligible flag does not match recomputed policy", label)
	}
	if endpoint.ProtectedEligible != recomputed.ProtectedEligible {
		return aoqtV7R6EndpointPolicyEvidence{}, fmt.Errorf("%s protected_eligible flag does not match recomputed policy", label)
	}
	if endpoint.TransactionEligible != recomputed.TransactionEligible {
		return aoqtV7R6EndpointPolicyEvidence{}, fmt.Errorf("%s transaction_eligible flag does not match recomputed policy", label)
	}
	if endpoint.Reason != recomputed.Reason {
		return aoqtV7R6EndpointPolicyEvidence{}, fmt.Errorf("%s reason %q does not match recomputed policy reason %q", label, endpoint.Reason, recomputed.Reason)
	}
	return recomputed, nil
}

func aoqtV7R6RecomputeEndpointPolicy(endpoint AOQTSidecarV7R6ProgressiveScreenStageEndpoint, policy AOQTSidecarCandidateEligibilityPolicy) (aoqtV7R6EndpointPolicyEvidence, error) {
	policy = normalizedAOQTSidecarCandidateEligibilityPolicy(policy)
	if err := validateAOQTSidecarCandidateEligibilityPolicy(policy); err != nil {
		return aoqtV7R6EndpointPolicyEvidence{}, err
	}
	if err := validateAOQTObjectiveComponents(endpoint.BaselineComponents, "AOQT V7-r6 endpoint baseline components"); err != nil {
		return aoqtV7R6EndpointPolicyEvidence{}, err
	}
	baseline := aoqtStepEvaluation{loss: endpoint.BaselineLoss, activation: endpoint.BaselineActivation, components: endpoint.BaselineComponents}
	candidate := aoqtStepEvaluation{loss: endpoint.CandidateLoss, activation: endpoint.CandidateActivation, components: endpoint.CandidateComponents}
	q3GainEligible := endpoint.ComponentDeltas == aoqtObjectiveComponentDelta(endpoint.BaselineComponents, endpoint.CandidateComponents) && endpoint.CandidateComponents.Q3Gain < endpoint.BaselineComponents.Q3Gain-aoqtTransactionalQ3ImprovementMinMagnitude
	protectedEligible := endpoint.ComponentDeltas == aoqtObjectiveComponentDelta(endpoint.BaselineComponents, endpoint.CandidateComponents) && aoqtFiniteReceiptComponentsValue(endpoint.ComponentDeltas) && len(aoqtComponentRegressions(endpoint.BaselineComponents, endpoint.CandidateComponents, policy)) == 0
	if !endpoint.AngleMoved {
		return aoqtV7R6EndpointPolicyEvidence{Q3GainEligible: q3GainEligible, ProtectedEligible: protectedEligible, Reason: string(aoqtRejectionNoAngleMovement)}, nil
	}
	decision := aoqtEvaluateTransactionalStepWithPolicy(baseline, candidate, aoqtV7R6AllComponentWeights(), policy)
	if decision.accepted {
		return aoqtV7R6EndpointPolicyEvidence{Q3GainEligible: q3GainEligible, ProtectedEligible: protectedEligible, TransactionEligible: true, Reason: aoqtProposalAcceptedReason}, nil
	}
	return aoqtV7R6EndpointPolicyEvidence{Q3GainEligible: q3GainEligible, ProtectedEligible: protectedEligible, Reason: string(decision.reason)}, nil
}

func aoqtV7R6AllComponentWeights() AOQTSidecarRowWeights {
	return AOQTSidecarRowWeights{Q3Gain: 1, Q3OrderGuard: 1, Q3ScoreDistill: 1, Q5OrderGuard: 1, Q5ScoreDistill: 1, NFBoundaryGuard: 1}
}

func (a AOQTSidecarV7R6ProgressiveScreenStageEndpoint) BaselineActivationPositive() bool {
	return a.BaselineActivation.Q3GainEligiblePairs > 0 && a.BaselineActivation.Q3GainContributingPairs > 0 && a.BaselineActivation.Q3OrderGuardPairs > 0 && a.BaselineActivation.Q3OrderGuardContributing > 0 && a.BaselineActivation.Q3ScoreDistillCount > 0 && a.BaselineActivation.Q5OrderGuardPairs > 0 && a.BaselineActivation.Q5OrderGuardContributing > 0 && a.BaselineActivation.Q5ScoreDistillCount > 0 && a.BaselineActivation.NFBoundaryGuardPairs > 0 && a.BaselineActivation.NFBoundaryGuardContributing > 0
}

func (a AOQTSidecarV7R6ProgressiveScreenStageEndpoint) CandidateActivationPositive() bool {
	return a.CandidateActivation.Q3GainEligiblePairs > 0 && a.CandidateActivation.Q3GainContributingPairs > 0 && a.CandidateActivation.Q3OrderGuardPairs > 0 && a.CandidateActivation.Q3OrderGuardContributing > 0 && a.CandidateActivation.Q3ScoreDistillCount > 0 && a.CandidateActivation.Q5OrderGuardPairs > 0 && a.CandidateActivation.Q5OrderGuardContributing > 0 && a.CandidateActivation.Q5ScoreDistillCount > 0 && a.CandidateActivation.NFBoundaryGuardPairs > 0 && a.CandidateActivation.NFBoundaryGuardContributing > 0
}

func aoqtV7R6WeightedActivationFinitePositive(a AOQTSidecarV7R6WeightedActivation) bool {
	for _, value := range []float64{
		a.Q3GainEligiblePairs, a.Q3GainContributingPairs,
		a.Q3OrderGuardPairs, a.Q3OrderGuardContributing,
		a.Q3ScoreDistillCount,
		a.Q5OrderGuardPairs, a.Q5OrderGuardContributing,
		a.Q5ScoreDistillCount,
		a.NFBoundaryGuardPairs, a.NFBoundaryGuardContributing,
	} {
		if !isFinite64(value) || value <= 0 {
			return false
		}
	}
	return true
}

func isFinite64(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

func isStrictStringSubset(small, large []string) bool {
	a, b := map[string]bool{}, map[string]bool{}
	for _, id := range small {
		a[id] = true
	}
	for _, id := range large {
		b[id] = true
	}
	if len(a) >= len(b) {
		return false
	}
	for id := range a {
		if !b[id] {
			return false
		}
	}
	return true
}

func indexOfIntAOQT(values []int, needle int) int {
	for i, value := range values {
		if value == needle {
			return i
		}
	}
	return -1
}

func aoqtV7R6OrderingKey(seed, stratum, rowID string) string {
	sum := sha256.Sum256([]byte(seed + "\x00" + stratum + "\x00" + rowID))
	return hex.EncodeToString(sum[:])
}

func aoqtV7R6ClassifyRow(row AOQTSidecarCalibrationRow) (int, error) {
	source := ""
	if len(row.CandidateSources) == 0 {
		return -1, fmt.Errorf("AOQT V7-r6 row %q has no candidate source semantics", row.RowID)
	}
	for i, candidateSource := range row.CandidateSources {
		if strings.TrimSpace(candidateSource) == "" {
			return -1, fmt.Errorf("AOQT V7-r6 row %q candidate source %d is empty", row.RowID, i)
		}
		if source == "" {
			source = candidateSource
		} else if candidateSource != source {
			return -1, fmt.Errorf("AOQT V7-r6 row %q mixes semantic candidate sources", row.RowID)
		}
	}
	for i, spec := range aoqtV7R6ProgressiveScreenStratumSpecs {
		// The row-id prefix is part of the semantic contract.  Checking it in
		// addition to the materialized dataset/guard/source fields prevents a
		// forged row from moving between strata by changing only one metadata
		// field.  Prefixes are disjoint, so membership remains deterministic and
		// does not depend on objective outcomes.
		if !strings.HasPrefix(row.RowID, spec.rowPrefix) || row.Dataset != spec.dataset || source != spec.source || row.GuardClass != spec.guard {
			continue
		}
		if spec.boundary && row.Weights.NFBoundaryGuard <= 0 {
			return -1, fmt.Errorf("AOQT V7-r6 row %q boundary stratum lacks positive nf boundary weight", row.RowID)
		}
		if !spec.boundary && row.Weights.NFBoundaryGuard != 0 {
			return -1, fmt.Errorf("AOQT V7-r6 row %q top10 stratum carries NF boundary weight", row.RowID)
		}
		return i, nil
	}
	return -1, fmt.Errorf("AOQT V7-r6 row %q does not match a canonical semantic stratum (dataset=%q source=%q guard=%q)", row.RowID, row.Dataset, source, row.GuardClass)
}

// ConstructAOQTV7R6ProgressiveScreenSamples deterministically partitions rows
// into the seven preregistered semantic strata, orders each stratum by a fixed
// SHA-256 key, and takes exact quota prefixes. It does not inspect outcomes or
// evaluate an objective, and therefore is safe to call before any probe work.
func ConstructAOQTV7R6ProgressiveScreenSamples(rows []AOQTSidecarCalibrationRow, seeds ...string) (AOQTSidecarV7R6ProgressiveScreenSamples, error) {
	seed := AOQTV7R6ProgressiveScreenSeed
	if len(seeds) > 1 {
		return AOQTSidecarV7R6ProgressiveScreenSamples{}, fmt.Errorf("AOQT V7-r6 sample construction accepts at most one seed")
	}
	if len(seeds) == 1 {
		seed = seeds[0]
	}
	if strings.TrimSpace(seed) == "" {
		return AOQTSidecarV7R6ProgressiveScreenSamples{}, fmt.Errorf("AOQT V7-r6 sample seed is required")
	}
	if len(rows) < AOQTV7R6ProgressiveScreenS1024 {
		return AOQTSidecarV7R6ProgressiveScreenSamples{}, fmt.Errorf("AOQT V7-r6 sampling requires at least %d rows", AOQTV7R6ProgressiveScreenS1024)
	}
	type item struct {
		row AOQTSidecarCalibrationRow
		key string
	}
	strata := make([][]item, AOQTV7R6ProgressiveScreenStratumCount)
	seen := map[string]bool{}
	for _, row := range rows {
		if strings.TrimSpace(row.RowID) == "" {
			return AOQTSidecarV7R6ProgressiveScreenSamples{}, fmt.Errorf("AOQT V7-r6 row id is required")
		}
		if seen[row.RowID] {
			return AOQTSidecarV7R6ProgressiveScreenSamples{}, fmt.Errorf("AOQT V7-r6 duplicate row id %q", row.RowID)
		}
		seen[row.RowID] = true
		i, err := aoqtV7R6ClassifyRow(row)
		if err != nil {
			return AOQTSidecarV7R6ProgressiveScreenSamples{}, err
		}
		strata[i] = append(strata[i], item{row: row, key: aoqtV7R6OrderingKey(seed, aoqtV7R6ProgressiveScreenStratumNames[i], row.RowID)})
	}
	var sample512, sample1024 AOQTSidecarV7R6ProgressiveScreenSample
	sample512.Name, sample1024.Name = "S512", "S1024"
	sample512.RowCount, sample1024.RowCount = AOQTV7R6ProgressiveScreenS512, AOQTV7R6ProgressiveScreenS1024
	for i := range strata {
		sort.SliceStable(strata[i], func(a, b int) bool {
			if strata[i][a].key != strata[i][b].key {
				return strata[i][a].key < strata[i][b].key
			}
			return strata[i][a].row.RowID < strata[i][b].row.RowID
		})
		spec := aoqtV7R6ProgressiveScreenStratumSpecs[i]
		q512, q1024 := spec.quota512, spec.quota1024
		if len(strata[i]) != spec.population {
			return AOQTSidecarV7R6ProgressiveScreenSamples{}, fmt.Errorf("AOQT V7-r6 semantic stratum %s population %d does not match preregistered population %d", spec.name, len(strata[i]), spec.population)
		}
		if len(strata[i]) < q1024 {
			return AOQTSidecarV7R6ProgressiveScreenSamples{}, fmt.Errorf("AOQT V7-r6 semantic stratum %s population %d is below quota %d", spec.name, len(strata[i]), q1024)
		}
		allIDs := make([]string, len(strata[i]))
		for j := range strata[i] {
			allIDs[j] = strata[i][j].row.RowID
		}
		st512 := make([]string, q512)
		st1024 := make([]string, q1024)
		copy(st512, allIDs[:q512])
		copy(st1024, allIDs[:q1024])
		s512 := AOQTSidecarV7R6ProgressiveScreenStratum{Name: spec.name, Index: i, Population: len(strata[i]), QuotaS512: q512, QuotaS1024: q1024, InclusionProbabilityS512: float64(q512) / float64(len(strata[i])), InclusionProbabilityS1024: float64(q1024) / float64(len(strata[i])), RowIDSHA256: aoqtRowIDSHA256(st512), OrderedRowIDSHA256: aoqtV7R6OrderedRowIDSHA256(allIDs), SampleRowIDs: append([]string(nil), st512...)}
		s1024 := s512
		s1024.RowIDSHA256 = aoqtRowIDSHA256(st1024)
		s1024.SampleRowIDs = append([]string(nil), st1024...)
		sample512.Strata = append(sample512.Strata, s512)
		sample1024.Strata = append(sample1024.Strata, s1024)
		sample512.RowIDs = append(sample512.RowIDs, st512...)
		sample1024.RowIDs = append(sample1024.RowIDs, st1024...)
	}
	sort.Strings(sample512.RowIDs)
	sort.Strings(sample1024.RowIDs)
	sample512.RowIDSHA256, sample1024.RowIDSHA256 = aoqtRowIDSHA256(sample512.RowIDs), aoqtRowIDSHA256(sample1024.RowIDs)
	sample512.StrictSubsetOf = "S1024"
	if !isStrictStringSubset(sample512.RowIDs, sample1024.RowIDs) {
		return AOQTSidecarV7R6ProgressiveScreenSamples{}, fmt.Errorf("AOQT V7-r6 S512 is not a strict subset of S1024")
	}
	return AOQTSidecarV7R6ProgressiveScreenSamples{S512: sample512, S1024: sample1024}, nil
}

// ConstructAOQTSidecarV7R6ProgressiveScreenSamples is a compatibility spelling
// for callers that include the sidecar namespace in the function name.
func ConstructAOQTSidecarV7R6ProgressiveScreenSamples(rows []AOQTSidecarCalibrationRow, seeds ...string) (AOQTSidecarV7R6ProgressiveScreenSamples, error) {
	return ConstructAOQTV7R6ProgressiveScreenSamples(rows, seeds...)
}

// The plural return type is exported for tests and callers that only need the
// sample construction phase.
type AOQTSidecarV7R6ProgressiveScreenSamples struct {
	S512  AOQTSidecarV7R6ProgressiveScreenSample
	S1024 AOQTSidecarV7R6ProgressiveScreenSample
}

type AOQTSidecarProgressiveScreenSamples = AOQTSidecarV7R6ProgressiveScreenSamples

func AOQTV7R6ProgressiveScreenStratumForRow(rowID string) string {
	for i, spec := range aoqtV7R6ProgressiveScreenStratumSpecs {
		if strings.HasPrefix(rowID, spec.rowPrefix) {
			return aoqtV7R6ProgressiveScreenStratumNames[i]
		}
	}
	return ""
}

// AOQTV7R6ProgressiveScreenStratumForCalibrationRow applies the complete
// semantic classifier (row-id prefix, dataset, guard, source, and boundary
// weight) and returns an empty string for rows that are not eligible for the
// preregistered screen.
func AOQTV7R6ProgressiveScreenStratumForCalibrationRow(row AOQTSidecarCalibrationRow) string {
	i, err := aoqtV7R6ClassifyRow(row)
	if err != nil {
		return ""
	}
	return aoqtV7R6ProgressiveScreenStratumNames[i]
}

func AOQTV7R6ProgressiveScreenHorvitzThompsonFactor(population, sampleCount int) (float64, error) {
	if population <= 0 || sampleCount <= 0 || sampleCount > population {
		return 0, fmt.Errorf("AOQT V7-r6 HT population/sample counts are invalid")
	}
	return float64(population) / float64(sampleCount), nil
}

// AOQTV7R6ProgressiveScreenHTWeights returns the exact per-stratum weights
// recorded for a stage.  It is intentionally derived only from the declared
// static populations and quotas; objective outcomes cannot influence it.
func AOQTV7R6ProgressiveScreenHTWeights(sample AOQTSidecarV7R6ProgressiveScreenSample) ([]AOQTSidecarV7R6ProgressiveScreenWeight, error) {
	if sample.Name != "S512" && sample.Name != "S1024" {
		return nil, fmt.Errorf("AOQT V7-r6 HT weights require stage S512 or S1024")
	}
	weights := make([]AOQTSidecarV7R6ProgressiveScreenWeight, 0, len(sample.Strata))
	for i, stratum := range sample.Strata {
		if i >= len(aoqtV7R6ProgressiveScreenStratumNames) || stratum.Name != aoqtV7R6ProgressiveScreenStratumNames[i] {
			return nil, fmt.Errorf("AOQT V7-r6 HT weight stratum %d is not canonical", i)
		}
		n := stratum.QuotaS512
		p := stratum.InclusionProbabilityS512
		if sample.Name == "S1024" {
			n = stratum.QuotaS1024
			p = stratum.InclusionProbabilityS1024
		}
		factor, err := AOQTV7R6ProgressiveScreenHorvitzThompsonFactor(stratum.Population, n)
		if err != nil {
			return nil, err
		}
		if !isFinite64(p) || math.Abs(p-float64(n)/float64(stratum.Population)) > 1e-12 {
			return nil, fmt.Errorf("AOQT V7-r6 HT weight stratum %s has invalid inclusion probability", stratum.Name)
		}
		weights = append(weights, AOQTSidecarV7R6ProgressiveScreenWeight{Stratum: stratum.Name, Population: stratum.Population, SampleCount: n, InclusionProbability: p, HorvitzThompsonFactor: factor})
	}
	if len(weights) != AOQTV7R6ProgressiveScreenStratumCount {
		return nil, fmt.Errorf("AOQT V7-r6 HT weights require all seven strata")
	}
	return weights, nil
}

func aoqtV7R6PositiveActivation(a AOQTSidecarObjectiveActivation) bool {
	return a.Q3GainEligiblePairs > 0 && a.Q3GainContributingPairs > 0 && a.Q3OrderGuardPairs > 0 && a.Q3OrderGuardContributing > 0 && a.Q3ScoreDistillCount > 0 && a.Q5OrderGuardPairs > 0 && a.Q5OrderGuardContributing > 0 && a.Q5ScoreDistillCount > 0 && a.NFBoundaryGuardPairs > 0 && a.NFBoundaryGuardContributing > 0
}

func aoqtV7R6SampleRows(rows []AOQTSidecarCalibrationRow, sample AOQTSidecarV7R6ProgressiveScreenSample) ([]AOQTSidecarCalibrationRow, error) {
	byID := make(map[string]AOQTSidecarCalibrationRow, len(rows))
	for _, row := range rows {
		byID[row.RowID] = row
	}
	selected := make([]AOQTSidecarCalibrationRow, 0, len(sample.RowIDs))
	for _, id := range sample.RowIDs {
		row, ok := byID[id]
		if !ok {
			return nil, fmt.Errorf("AOQT V7-r6 sample row %q is not in fold input", id)
		}
		selected = append(selected, row)
	}
	return selected, nil
}

func aoqtV7R6ApplyHTWeights(rows []AOQTSidecarCalibrationRow, sample AOQTSidecarV7R6ProgressiveScreenSample) ([]AOQTSidecarCalibrationRow, []AOQTSidecarV7R6ProgressiveScreenWeight, error) {
	weights, err := AOQTV7R6ProgressiveScreenHTWeights(sample)
	if err != nil {
		return nil, nil, err
	}
	pop, n := map[string]int{}, map[string]int{}
	for _, weight := range weights {
		pop[weight.Stratum], n[weight.Stratum] = weight.Population, weight.SampleCount
	}
	for i := range rows {
		stratum, err := aoqtV7R6ClassifyRow(rows[i])
		if err != nil {
			return nil, nil, err
		}
		name := aoqtV7R6ProgressiveScreenStratumNames[stratum]
		factor, err := AOQTV7R6ProgressiveScreenHorvitzThompsonFactor(pop[name], n[name])
		if err != nil {
			return nil, nil, err
		}
		rows[i].Weights.Q3Gain *= float32(factor)
		rows[i].Weights.Q3OrderGuard *= float32(factor)
		rows[i].Weights.Q3ScoreDistill *= float32(factor)
		rows[i].Weights.Q5OrderGuard *= float32(factor)
		rows[i].Weights.Q5ScoreDistill *= float32(factor)
		rows[i].Weights.NFBoundaryGuard *= float32(factor)
	}
	return rows, weights, nil
}

func aoqtV7R6EndpointSetSHA(endpoints []AOQTSidecarActualCoordinateProbeEndpoint) (string, error) {
	return aoqtCanonicalSHA256(endpoints)
}

// AOQTV7R6CanonicalR4EndpointSetSHA256 binds the exact eligible projection of
// the canonical r4 endpoint set.  The full r4 receipt remains bound separately
// by its object/file hashes; this digest gives the r6 receipt a small explicit
// witness for the 31 endpoint metadata and actual-direction source.
func AOQTV7R6CanonicalR4EndpointSetSHA256(r4 AOQTSidecarActualCoordinateProbeReceipt) (string, error) {
	if len(r4.Endpoints) != 384 || r4.EndpointCount != 384 {
		return "", fmt.Errorf("AOQT V7-r6 canonical r4 endpoint set must contain exactly 384 endpoints")
	}
	if !reflect.DeepEqual(r4.SelectedEndpointOrdinals, aoqtV7R6CanonicalEligibleEndpointOrdinals[:]) {
		return "", fmt.Errorf("AOQT V7-r6 canonical r4 selected ordinals are not the exact 31")
	}
	selected := make([]AOQTSidecarActualCoordinateProbeEndpoint, 0, len(aoqtV7R6CanonicalEligibleEndpointOrdinals))
	seen := make(map[int]bool, len(r4.Endpoints))
	for _, endpoint := range r4.Endpoints {
		if endpoint.Ordinal <= 0 || endpoint.Ordinal > len(r4.Endpoints) || seen[endpoint.Ordinal] {
			return "", fmt.Errorf("AOQT V7-r6 canonical r4 endpoint ordinal %d is duplicated or out of range", endpoint.Ordinal)
		}
		seen[endpoint.Ordinal] = true
	}
	for _, ordinal := range aoqtV7R6CanonicalEligibleEndpointOrdinals {
		endpoint := r4EndpointByOrdinal(r4.Endpoints, ordinal)
		if endpoint.Ordinal != ordinal || !endpoint.TransactionEligible {
			return "", fmt.Errorf("AOQT V7-r6 canonical r4 endpoint %d is not eligible", ordinal)
		}
		selected = append(selected, endpoint)
	}
	return aoqtV7R6EndpointSetSHA(selected)
}

func aoqtV7R6SelectionSeed(r4Support, r5Outcome string, set AOQTSidecarCalibrationSet, foldID string) (string, error) {
	manifestSHA, err := AOQTSidecarManifestSHA256(set.Manifest)
	if err != nil {
		return "", err
	}
	objectiveSHA, err := aoqtActualCoordinateObjectiveContractSHA256(set.Manifest.ObjectiveContract)
	if err != nil {
		return "", err
	}
	// Sampling is fixed before endpoint outcomes exist. The first argument is
	// the canonical r4 support identity (the eligible endpoint-set digest),
	// while the legacy second argument is deliberately ignored: an r5 training
	// outcome mutation must never reshuffle the screen population.
	_ = r5Outcome
	payload := strings.Join([]string{
		AOQTV7R6ProgressiveScreenSeed,
		AOQTV7R6ProgressiveScreenStratificationVersion,
		AOQTV7R6ProgressiveScreenScheduleVersion,
		AOQTV7R6ProgressiveScreenRankingVersion,
		AOQTV7R6ProgressiveScreenPolicyVersion,
		foldID,
		manifestSHA,
		set.Manifest.RowIDSHA256,
		objectiveSHA,
		r4Support,
	}, "\x00")
	sum := sha256.Sum256([]byte(payload))
	return hex.EncodeToString(sum[:]), nil
}

func aoqtV7R6EndpointFromCanonical(endpoint AOQTSidecarActualCoordinateProbeEndpoint) AOQTSidecarV7R6ProgressiveScreenEndpoint {
	return AOQTSidecarV7R6ProgressiveScreenEndpoint{Ordinal: endpoint.Ordinal, Kind: endpoint.Kind, CoordinateIndex: endpoint.CoordinateIndex, AnalyticDirection: endpoint.AnalyticDirection, Indices: append([]int(nil), endpoint.Indices...), Directions: append([]int(nil), endpoint.Directions...), GlobalSign: endpoint.GlobalSign, Magnitude: endpoint.Magnitude, RequestedDirectionSHA256: endpoint.RequestedDirectionSHA256, CanonicalActualDirectionSHA256: endpoint.ActualDirectionSHA256}
}

func aoqtV7R6StageEndpoint(endpoint AOQTSidecarActualCoordinateProbeEndpoint, baseline aoqtStepEvaluation) AOQTSidecarV7R6ProgressiveScreenStageEndpoint {
	return AOQTSidecarV7R6ProgressiveScreenStageEndpoint{BaselineLoss: endpoint.BaselineLoss, BaselineLossFinite: endpoint.BaselineLossFinite, CandidateLoss: endpoint.CandidateLoss, CandidateLossFinite: endpoint.CandidateLossFinite, BaselineComponents: endpoint.BaselineComponents, CandidateComponents: endpoint.CandidateComponents, ComponentDeltas: endpoint.ComponentDeltas, CandidateActivation: endpoint.CandidateActivation, BaselineActivation: baseline.activation, TotalDelta: endpoint.TotalDelta, TotalDeltaFinite: endpoint.TotalDeltaFinite, ActualDirectionSHA256: endpoint.ActualDirectionSHA256, ActualDirectionMaxAbs: endpoint.ActualDirectionMaxAbs, ActualDirection: append([]float32(nil), endpoint.ActualDirection...), AngleMoved: endpoint.AngleMoved, Q3GainEligible: endpoint.Q3GainEligible, ProtectedEligible: endpoint.ProtectedEligible, TransactionEligible: endpoint.TransactionEligible, Reason: endpoint.Reason, CandidateError: endpoint.CandidateError}
}

func aoqtV7R6EvaluateBaseline(rows []AOQTSidecarCalibrationRow, sample AOQTSidecarV7R6ProgressiveScreenSample, objective AOQTSidecarVectorObjective, trainer *AOQTSidecarTrainer) (float32, AOQTSidecarObjectiveActivation, AOQTSidecarObjectiveComponents, AOQTSidecarV7R6WeightedActivation, error) {
	var (
		totalLoss       float32
		totalActivation AOQTSidecarObjectiveActivation
		totalComponents AOQTSidecarObjectiveComponents
		weighted        AOQTSidecarV7R6WeightedActivation
	)
	factors := make(map[string]float64, len(sample.Strata))
	for _, stratum := range sample.Strata {
		n := stratum.QuotaS512
		if sample.Name == "S1024" {
			n = stratum.QuotaS1024
		}
		factor, err := AOQTV7R6ProgressiveScreenHorvitzThompsonFactor(stratum.Population, n)
		if err != nil {
			return 0, totalActivation, totalComponents, weighted, err
		}
		factors[stratum.Name] = factor
	}
	runtime, err := newAOQTGivensTrainingRuntime(trainer.trainingTransformSnapshot())
	if err != nil {
		return 0, totalActivation, totalComponents, weighted, err
	}
	for _, row := range rows {
		query, candidates, err := transformAOQTRowWithRuntime(runtime, row)
		if err != nil {
			return 0, totalActivation, totalComponents, weighted, err
		}
		result, err := objective.EvaluateAOQT(AOQTSidecarObjectiveInput{Row: aoqtObjectiveRowView(row), Query: query, Candidates: candidates})
		if err != nil {
			return 0, totalActivation, totalComponents, weighted, err
		}
		if !isFinite32(result.Loss) {
			return 0, totalActivation, totalComponents, weighted, fmt.Errorf("AOQT V7-r6 baseline objective loss must be finite")
		}
		if err := validateAOQTObjectiveActivation(result.Activation, "AOQT V7-r6 baseline objective activation"); err != nil {
			return 0, totalActivation, totalComponents, weighted, err
		}
		if err := validateAOQTObjectiveComponents(result.Components, "AOQT V7-r6 baseline objective components"); err != nil {
			return 0, totalActivation, totalComponents, weighted, err
		}
		if err := validateAOQTLossMatchesComponents(result.Loss, result.Components, "AOQT V7-r6 baseline objective"); err != nil {
			return 0, totalActivation, totalComponents, weighted, err
		}
		totalLoss += result.Loss
		totalActivation.Add(result.Activation)
		totalComponents.Add(result.Components)
		stratum, err := aoqtV7R6ClassifyRow(row)
		if err != nil {
			return 0, totalActivation, totalComponents, weighted, err
		}
		factor := factors[aoqtV7R6ProgressiveScreenStratumNames[stratum]]
		if factor <= 0 || !isFinite64(factor) {
			return 0, totalActivation, totalComponents, weighted, fmt.Errorf("AOQT V7-r6 baseline has no finite HT factor for stratum %q", aoqtV7R6ProgressiveScreenStratumNames[stratum])
		}
		weighted.Q3GainEligiblePairs += float64(result.Activation.Q3GainEligiblePairs) * factor
		weighted.Q3GainContributingPairs += float64(result.Activation.Q3GainContributingPairs) * factor
		weighted.Q3OrderGuardPairs += float64(result.Activation.Q3OrderGuardPairs) * factor
		weighted.Q3OrderGuardContributing += float64(result.Activation.Q3OrderGuardContributing) * factor
		weighted.Q3ScoreDistillCount += float64(result.Activation.Q3ScoreDistillCount) * factor
		weighted.Q5OrderGuardPairs += float64(result.Activation.Q5OrderGuardPairs) * factor
		weighted.Q5OrderGuardContributing += float64(result.Activation.Q5OrderGuardContributing) * factor
		weighted.Q5ScoreDistillCount += float64(result.Activation.Q5ScoreDistillCount) * factor
		weighted.NFBoundaryGuardPairs += float64(result.Activation.NFBoundaryGuardPairs) * factor
		weighted.NFBoundaryGuardContributing += float64(result.Activation.NFBoundaryGuardContributing) * factor
	}
	return totalLoss, totalActivation, totalComponents, weighted, nil
}

// aoqtV7R6EvaluateUnweightedBaseline is reserved for the optional full-stage
// replay. It consumes the complete 10,942-row fold and supplies unit factors;
// no sample quota or HT expansion is applied to this authority evaluation.
func aoqtV7R6EvaluateUnweightedBaseline(rows []AOQTSidecarCalibrationRow, objective AOQTSidecarVectorObjective, trainer *AOQTSidecarTrainer) (float32, AOQTSidecarObjectiveActivation, AOQTSidecarObjectiveComponents, error) {
	if len(rows) != AOQTV7R6ProgressiveScreenPopulationTotal {
		return 0, AOQTSidecarObjectiveActivation{}, AOQTSidecarObjectiveComponents{}, fmt.Errorf("AOQT V7-r6 full verification requires exactly %d fold rows, got %d", AOQTV7R6ProgressiveScreenPopulationTotal, len(rows))
	}
	counts := make([]int, AOQTV7R6ProgressiveScreenStratumCount)
	for _, row := range rows {
		i, err := aoqtV7R6ClassifyRow(row)
		if err != nil {
			return 0, AOQTSidecarObjectiveActivation{}, AOQTSidecarObjectiveComponents{}, err
		}
		counts[i]++
	}
	for i, want := range aoqtV7R6ProgressiveScreenPopulations {
		if counts[i] != want {
			return 0, AOQTSidecarObjectiveActivation{}, AOQTSidecarObjectiveComponents{}, fmt.Errorf("AOQT V7-r6 full verification stratum %q population %d does not match %d", aoqtV7R6ProgressiveScreenStratumNames[i], counts[i], want)
		}
	}
	unitSample := AOQTSidecarV7R6ProgressiveScreenSample{Name: "S1024", Strata: make([]AOQTSidecarV7R6ProgressiveScreenStratum, 0, AOQTV7R6ProgressiveScreenStratumCount)}
	for i, spec := range aoqtV7R6ProgressiveScreenStratumSpecs {
		unitSample.Strata = append(unitSample.Strata, AOQTSidecarV7R6ProgressiveScreenStratum{
			Name: spec.name, Index: i, Population: spec.population,
			QuotaS512: spec.population, QuotaS1024: spec.population,
			InclusionProbabilityS512: 1, InclusionProbabilityS1024: 1,
		})
	}
	loss, activation, components, _, err := aoqtV7R6EvaluateBaseline(rows, unitSample, objective, trainer)
	return loss, activation, components, err
}

type aoqtV7R6ViolationRank struct {
	count  int
	excess float32
}

// aoqtV7R6EndpointViolationRank is used only to order the fixed-K fallback
// tail. Feasible endpoints always rank first. For infeasible endpoints the
// preregistered hard gates are ordered by violation count, then aggregate
// excess over the corresponding strict/allowed threshold, and only then by
// observed loss delta. This keeps a spectacular but unsafe loss improvement
// from displacing a candidate that is closer to satisfying the transaction
// policy.
func aoqtV7R6EndpointViolationRank(endpoint AOQTSidecarV7R6ProgressiveScreenStageEndpoint, policy AOQTSidecarCandidateEligibilityPolicy) aoqtV7R6ViolationRank {
	if aoqtV7R6EndpointFeasibleWithPolicy(endpoint, policy) {
		return aoqtV7R6ViolationRank{}
	}
	policy = normalizedAOQTSidecarCandidateEligibilityPolicy(policy)
	rank := aoqtV7R6ViolationRank{}
	add := func(excess float32) {
		rank.count++
		if math.IsNaN(float64(excess)) || math.IsInf(float64(excess), 0) {
			rank.excess = float32(math.Inf(1))
			return
		}
		if excess > 0 && !math.IsInf(float64(rank.excess), 1) {
			rank.excess += excess
		}
	}
	if !endpoint.AngleMoved {
		add(1)
	}
	if !endpoint.CandidateLossFinite || !isFinite32(endpoint.CandidateLoss) {
		add(1)
	}
	if !endpoint.CandidateLossFinite || !endpoint.BaselineLossFinite || !endpoint.TotalDeltaFinite || !isFinite32(endpoint.TotalDelta) {
		add(1)
	} else if endpoint.TotalDelta > aoqtTransactionalLossEpsilon {
		add(endpoint.TotalDelta - aoqtTransactionalLossEpsilon)
	}
	if !aoqtFiniteReceiptComponentsValue(endpoint.CandidateComponents) || !aoqtFiniteReceiptComponentsValue(endpoint.BaselineComponents) || !aoqtFiniteReceiptComponentsValue(endpoint.ComponentDeltas) {
		add(1)
	}
	if !endpoint.Q3GainEligible {
		excess := endpoint.ComponentDeltas.Q3Gain + aoqtTransactionalQ3ImprovementMinMagnitude
		if excess < 0 {
			excess = 0
		}
		add(excess)
	}
	regressionCount := 0
	for _, item := range []struct {
		delta   float32
		allowed float32
	}{
		{endpoint.ComponentDeltas.Q3OrderGuard, policy.Q3OrderGuardAllowedLossIncrease},
		{endpoint.ComponentDeltas.Q3ScoreDistill, policy.Q3ScoreDistillAllowedLossIncrease},
		{endpoint.ComponentDeltas.Q5OrderGuard, policy.Q5OrderGuardAllowedLossIncrease},
		{endpoint.ComponentDeltas.Q5ScoreDistill, policy.Q5ScoreDistillAllowedLossIncrease},
		{endpoint.ComponentDeltas.NFBoundaryGuard, policy.NFBoundaryGuardAllowedLossIncrease},
	} {
		if !isFinite32(item.delta) || item.delta > item.allowed+1e-7 {
			excess := item.delta - item.allowed
			if !isFinite32(excess) || excess < 0 {
				excess = 1
			}
			add(excess)
			regressionCount++
		}
	}
	if !endpoint.ProtectedEligible && regressionCount == 0 {
		// Preserve a deterministic violation witness if a receipt carries an
		// inconsistent aggregate protected flag without component deltas.
		add(1)
	}
	activation := endpoint.CandidateActivation
	if activation.Q3GainEligiblePairs <= 0 || activation.Q3GainContributingPairs <= 0 {
		add(1)
	}
	if activation.Q3OrderGuardPairs <= 0 || activation.Q3OrderGuardContributing <= 0 {
		add(1)
	}
	if activation.Q3ScoreDistillCount <= 0 {
		add(1)
	}
	if activation.Q5OrderGuardPairs <= 0 || activation.Q5OrderGuardContributing <= 0 {
		add(1)
	}
	if activation.Q5ScoreDistillCount <= 0 {
		add(1)
	}
	if activation.NFBoundaryGuardPairs <= 0 || activation.NFBoundaryGuardContributing <= 0 {
		add(1)
	}
	if rank.count == 0 {
		add(1)
	}
	return rank
}

func aoqtV7R6RankEndpointsWithPolicy(endpoints []AOQTSidecarV7R6ProgressiveScreenEndpoint, stage string, policy AOQTSidecarCandidateEligibilityPolicy) ([]int, error) {
	if stage != "S512" && stage != "S1024" {
		return nil, fmt.Errorf("AOQT V7-r6 rank stage %q is unsupported", stage)
	}
	type item struct {
		ordinal   int
		delta     float32
		feasible  bool
		violation aoqtV7R6ViolationRank
	}
	items := make([]item, 0, len(endpoints))
	for _, endpoint := range endpoints {
		value := endpoint.S512.TotalDelta
		if stage == "S1024" {
			value = endpoint.S1024.TotalDelta
		}
		if !isFinite32(value) {
			return nil, fmt.Errorf("AOQT V7-r6 %s rank encountered nonfinite delta", stage)
		}
		stageEndpoint := endpoint.S512
		if stage == "S1024" {
			stageEndpoint = endpoint.S1024
		}
		items = append(items, item{ordinal: endpoint.Ordinal, delta: value, feasible: aoqtV7R6EndpointFeasibleWithPolicy(stageEndpoint, policy), violation: aoqtV7R6EndpointViolationRank(stageEndpoint, policy)})
	}
	sort.SliceStable(items, func(i, j int) bool {
		if items[i].feasible != items[j].feasible {
			return items[i].feasible
		}
		if !items[i].feasible && items[i].violation.count != items[j].violation.count {
			return items[i].violation.count < items[j].violation.count
		}
		if !items[i].feasible && items[i].violation.excess != items[j].violation.excess {
			return items[i].violation.excess < items[j].violation.excess
		}
		if items[i].delta != items[j].delta {
			return items[i].delta < items[j].delta
		}
		return items[i].ordinal < items[j].ordinal
	})
	order := make([]int, len(items))
	for i := range items {
		order[i] = items[i].ordinal
	}
	return order, nil
}

func aoqtV7R6RankEndpoints(endpoints []AOQTSidecarV7R6ProgressiveScreenEndpoint, stage string) ([]int, error) {
	return aoqtV7R6RankEndpointsWithPolicy(endpoints, stage, aoqtV7R6CanonicalEligibilityPolicy())
}

// RankAOQTV7R6ProgressiveScreenEndpoints exposes the preregistered ranking
// rule for deterministic tests and audit tooling.  It sorts each stage's
// actual loss delta ascending (improvement first), then ordinal, but only
// after feasible transactional candidates are placed ahead of infeasible
// diagnostics. No budget or cross-component scalar is introduced.
func RankAOQTV7R6ProgressiveScreenEndpoints(endpoints []AOQTSidecarV7R6ProgressiveScreenEndpoint, stage string) ([]int, error) {
	return aoqtV7R6RankEndpoints(endpoints, stage)
}

func aoqtV7R6RankingSHA(order []int) (string, error) {
	return aoqtCanonicalSHA256(struct {
		Version string `json:"version"`
		Order   []int  `json:"order"`
	}{AOQTV7R6ProgressiveScreenRankingVersion, append([]int(nil), order...)})
}

// RunAOQTV7R6ProgressiveScreen executes a screen in memory.  It refuses to
// construct a trainer or objective unless the mode is explicitly probe-only,
// and it never calls Fit or writes metrics/package artifacts.
func RunAOQTV7R6ProgressiveScreen(cfg AOQTSidecarV7R6ProgressiveScreenConfig) (AOQTSidecarV7R6ProgressiveScreenResult, error) {
	result := AOQTSidecarV7R6ProgressiveScreenResult{}
	if cfg.Mode == "" {
		cfg.Mode = AOQTSidecarOptimizerModeV7R6ProgressiveScreen
	}
	if cfg.Mode != AOQTSidecarOptimizerModeV7R6ProgressiveScreen || !cfg.ProbeOnly {
		return result, fmt.Errorf("AOQT V7-r6 requires mode=%s and probe-only=true", AOQTSidecarOptimizerModeV7R6ProgressiveScreen)
	}
	set, ioReport, preflight, preflightSHA, split, r4, r5, r4FileSHA, r5FileSHA, err := aoqtV7R6ResolveInputs(cfg)
	if err != nil {
		return result, err
	}
	if err := set.Validate(); err != nil {
		return result, fmt.Errorf("AOQT V7-r6 calibration set: %w", err)
	}
	if err := aoqtV7R6RejectOfficialOverlap(set.Rows); err != nil {
		return result, err
	}
	if split == nil {
		return result, fmt.Errorf("AOQT V7-r6 requires a bound fold/split input")
	}
	if r4 == nil || r5 == nil {
		return result, fmt.Errorf("AOQT V7-r6 requires bound r4 and r5 receipts")
	}
	if err := validateAOQTV7R6CanonicalReceipts(*r4, *r5, set, split); err != nil {
		return result, err
	}
	r4SHA, err := r4.SHA256()
	if err != nil {
		return result, err
	}
	r5SHA, err := r5.SHA256()
	if err != nil {
		return result, err
	}
	endpointSetSHA, err := AOQTV7R6CanonicalR4EndpointSetSHA256(*r4)
	if err != nil {
		return result, err
	}
	selectionSeed, err := aoqtV7R6SelectionSeed(endpointSetSHA, r5SHA, set, split.FoldID)
	if err != nil {
		return result, err
	}
	samples, err := ConstructAOQTV7R6ProgressiveScreenSamples(set.Rows, selectionSeed)
	if err != nil {
		return result, err
	}
	if err := aoqtV7R6RequireSixComponents(set, samples); err != nil {
		return result, err
	}
	objective, err := objectiveFromAOQTContract(set.Manifest.ObjectiveContract)
	if err != nil {
		return result, err
	}
	policy, err := AOQTSidecarTrainingContractPolicy(set.Manifest.TrainingContract, set.Manifest.CandidateEligibilityPolicy)
	if err != nil {
		return result, err
	}
	policySHA, err := aoqtActualCoordinatePolicySHA256(policy)
	if err != nil {
		return result, err
	}
	contractSHA, err := aoqtActualCoordinateObjectiveContractSHA256(set.Manifest.ObjectiveContract)
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
	canonicalEndpoints := make([]AOQTSidecarActualCoordinateProbeEndpoint, 0, len(aoqtV7R6CanonicalEligibleEndpointOrdinals))
	for _, ordinal := range aoqtV7R6CanonicalEligibleEndpointOrdinals {
		canonicalEndpoints = append(canonicalEndpoints, r4EndpointByOrdinal(r4.Endpoints, ordinal))
	}
	stage512, outcomes512, err := aoqtV7R6EvaluateStage(trainer, objective, policy, set, samples.S512, canonicalEndpoints)
	if err != nil {
		return result, err
	}
	// New identity trainer for S1024: each stage must start at exactly the
	// same identity baseline, not at a projected endpoint from S512.
	trainer, err = NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{Dim: set.Manifest.Topology.Dim, Stages: set.Manifest.Topology.Stages, PairingSeed: set.Manifest.Topology.Seed, WorkplanSeed: 1, OptimizerMode: AOQTSidecarOptimizerModeV7R4DevActualCoordinate, ActualCoordinateProbeRequired: true, ActualCoordinateProbeOnly: true, ActualCoordinateProbeFoldID: split.FoldID, ActualCoordinateProbeSplitManifestSHA256: split.SplitManifestSHA256, LearningRate: AOQTV7R4ActualCoordinateProbeLearningRate, AngleCap: AOQTSidecarDefaultAngleCap, MaxAngleCap: AOQTSidecarHardMaxAngleCap, TrainingContract: set.Manifest.TrainingContract, CandidateEligibilityPolicy: cloneAOQTSidecarCandidateEligibilityPolicy(set.Manifest.CandidateEligibilityPolicy)})
	if err != nil {
		return result, err
	}
	stage1024, outcomes1024, err := aoqtV7R6EvaluateStage(trainer, objective, policy, set, samples.S1024, canonicalEndpoints)
	if err != nil {
		return result, err
	}
	rank512, err := aoqtV7R6RankEndpointsWithPolicy(outcomes512, "S512", policy)
	if err != nil {
		return result, err
	}
	rank1024, err := aoqtV7R6RankEndpointsWithPolicy(outcomes1024, "S1024", policy)
	if err != nil {
		return result, err
	}
	rank512SHA, err := aoqtV7R6RankingSHA(rank512)
	if err != nil {
		return result, err
	}
	rank1024SHA, err := aoqtV7R6RankingSHA(rank1024)
	if err != nil {
		return result, err
	}
	rankAt := map[int]int{}
	for i, ordinal := range rank512 {
		rankAt[ordinal] = i + 1
	}
	for i := range outcomes512 {
		outcomes512[i].RankS512 = rankAt[outcomes512[i].Ordinal]
	}
	rankAt = map[int]int{}
	for i, ordinal := range rank1024 {
		rankAt[ordinal] = i + 1
	}
	for i := range outcomes512 {
		outcomes512[i].RankS1024 = rankAt[outcomes512[i].Ordinal]
	}
	// The selected list is a fixed-K prefix of the large-stage order.  No
	// scalar budget/utility tradeoff is applied; both independent stage orders
	// remain in the receipt for reversal analysis.
	selected := append([]int(nil), rank1024[:AOQTV7R6ProgressiveScreenSelectedK]...)
	for i := range outcomes512 {
		outcomes512[i].S1024 = outcomes1024[i].S1024
	}
	byOrdinal := make(map[int]AOQTSidecarV7R6ProgressiveScreenEndpoint, len(outcomes512))
	for _, endpoint := range outcomes512 {
		byOrdinal[endpoint.Ordinal] = endpoint
	}
	selectionFeasibleCount := 0
	for _, ordinal := range selected {
		if aoqtV7R6EndpointFeasibleWithPolicy(byOrdinal[ordinal].S1024, policy) {
			selectionFeasibleCount++
		}
	}
	selectionFeasible := selectionFeasibleCount == AOQTV7R6ProgressiveScreenSelectedK
	runFullVerification := !cfg.StopAfterReceipts
	if cfg.FullVerification {
		runFullVerification = true
	}
	fullVerification := AOQTSidecarV7R6ProgressiveScreenFullVerification{DenseGateStatus: AOQTV7R6ProgressiveScreenDenseGateStatus, DenseGateDeferred: true}
	objectiveEvaluationCount := AOQTV7R6ProgressiveScreenStageOnlyObjectiveEvaluations
	if runFullVerification {
		fullTrainer, trainerErr := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{Dim: set.Manifest.Topology.Dim, Stages: set.Manifest.Topology.Stages, PairingSeed: set.Manifest.Topology.Seed, WorkplanSeed: 1, OptimizerMode: AOQTSidecarOptimizerModeV7R4DevActualCoordinate, ActualCoordinateProbeRequired: true, ActualCoordinateProbeOnly: true, ActualCoordinateProbeFoldID: split.FoldID, ActualCoordinateProbeSplitManifestSHA256: split.SplitManifestSHA256, LearningRate: AOQTV7R4ActualCoordinateProbeLearningRate, AngleCap: AOQTSidecarDefaultAngleCap, MaxAngleCap: AOQTSidecarHardMaxAngleCap, TrainingContract: set.Manifest.TrainingContract, CandidateEligibilityPolicy: cloneAOQTSidecarCandidateEligibilityPolicy(set.Manifest.CandidateEligibilityPolicy)})
		if trainerErr != nil {
			return result, trainerErr
		}
		if _, trainerErr = fullTrainer.Plan(set); trainerErr != nil {
			return result, trainerErr
		}
		fullVerification, err = aoqtV7R6EvaluateFullVerification(fullTrainer, objective, policy, set, canonicalEndpoints, selected)
		if err != nil {
			return result, err
		}
		objectiveEvaluationCount = AOQTV7R6ProgressiveScreenObjectiveEvaluations
	}
	fullVerificationFeasibleCount := 0
	if runFullVerification {
		for _, endpoint := range fullVerification.Endpoints {
			if endpoint.Feasible {
				fullVerificationFeasibleCount++
			}
		}
	}
	fullVerificationFeasible := runFullVerification && fullVerificationFeasibleCount == AOQTV7R6ProgressiveScreenSelectedK
	passed := selectionFeasible
	if runFullVerification {
		// S1024 feasibility is diagnostic for the fixed-K fallback. The
		// authoritative full fold decides pass/fail for the selected six, even
		// when one or more of them was an S1024-infeasible fallback.
		passed = fullVerification.Completed && fullVerificationFeasible
	}
	failureReason := ""
	if !passed {
		if runFullVerification && !fullVerification.Completed {
			failureReason = "full-objective top-K verification did not complete"
		} else if runFullVerification && !fullVerificationFeasible {
			failureReason = fmt.Sprintf("full-objective top-K contains %d/%d feasible endpoints", fullVerificationFeasibleCount, AOQTV7R6ProgressiveScreenSelectedK)
		} else if !selectionFeasible {
			failureReason = fmt.Sprintf("fixed-K S1024 prefix contains %d/%d feasible endpoints; infeasible endpoints were retained as fixed-K fallback", selectionFeasibleCount, AOQTV7R6ProgressiveScreenSelectedK)
		}
	}
	inputs := AOQTSidecarRunMetricInputs{AnchorArtifactSHA256: set.Manifest.AnchorArtifactSHA256, AnchorPackageManifestSHA256: set.Manifest.AnchorPackageManifestSHA256, AnchorEmbeddingSpaceID: set.Manifest.AnchorEmbeddingSpaceID, DatasetManifestSHA256: ioReport.ManifestSHA256, QrelsSHA256ByDataset: cloneAOQTStringMap(set.Manifest.QrelsSHA256ByDataset), CompatibilityDigest: set.Manifest.CompatibilityDigest}
	receipt := AOQTSidecarV7R6ProgressiveScreenReceipt{Schema: AOQTSidecarV7R6ProgressiveScreenSchema, Mode: cfg.Mode, Required: true, ProbeOnly: true, Passed: passed, FailureReason: failureReason, StratificationVersion: AOQTV7R6ProgressiveScreenStratificationVersion, WeightingVersion: AOQTV7R6ProgressiveScreenWeightingVersion, ScheduleVersion: AOQTV7R6ProgressiveScreenScheduleVersion, RankingVersion: AOQTV7R6ProgressiveScreenRankingVersion, PolicyVersion: AOQTV7R6ProgressiveScreenPolicyVersion, SelectionSeedSHA256: selectionSeed, R4EndpointSetSHA256: endpointSetSHA, R4EligibleEndpointOrdinals: AOQTV7R6CanonicalEligibleEndpointOrdinals(), R5ReceiptSHA256: r5SHA, FoldInputRowIDSHA256: set.Manifest.RowIDSHA256, FoldInputRowsSHA256: ioReport.RowsSHA256, CanonicalR4ScheduleSHA256: r4.ScheduleSHA256, CanonicalR4RankingSHA256: r4.RankingSHA256, CanonicalR4PolicySHA256: r4.EligibilityPolicySHA256, CanonicalObjectiveContractSHA256: contractSHA, CanonicalEligibilityPolicySHA256: policySHA, StageS512: stage512, StageS1024: stage1024, FullVerification: fullVerification, Endpoints: outcomes512, SelectedEndpointOrdinals: selected, SelectionK: AOQTV7R6ProgressiveScreenSelectedK, ObjectiveEvaluationCount: objectiveEvaluationCount, SelectionFeasible: selectionFeasible, SelectionFeasibleCount: selectionFeasibleCount, SelectionFallbackUsed: !selectionFeasible, FullVerificationFeasible: fullVerificationFeasible, FullVerificationFeasibleCount: fullVerificationFeasibleCount, DenseGateStatus: AOQTV7R6ProgressiveScreenDenseGateStatus, DenseGateDeferred: true, DenseMaxAbsDeltaTolerance: policy.DenseMaxAbsDeltaTolerance, QualityClaim: false, ReleaseClaim: false, OfficialClaim: false, OfficialHeldoutGate: false, CommercialClaim: false}
	receipt.StageS512.Ranking, receipt.StageS512.RankingSHA256 = rank512, rank512SHA
	receipt.StageS1024.Ranking, receipt.StageS1024.RankingSHA256 = rank1024, rank1024SHA
	receipt.Binding = AOQTSidecarV7R6ProgressiveScreenBinding{Inputs: inputs, IOReport: ioReport, PreflightPath: cfg.PreflightJSONPath, PreflightSHA256: preflightSHA, SplitBinding: cloneAOQTDevSplitBinding(split), FoldID: split.FoldID, CanonicalR4ReceiptPath: cfg.CanonicalR4ReceiptPath, CanonicalR4ReceiptSHA256: r4SHA, CanonicalR4ReceiptFileSHA256: r4FileSHA, CanonicalR5ReceiptPath: cfg.CanonicalR5ReceiptPath, CanonicalR5ReceiptSHA256: r5SHA, CanonicalR5ReceiptFileSHA256: r5FileSHA}
	// Preserve preflight as a consumed provenance artifact even though screen
	// output intentionally has no metrics output.  The variable is otherwise
	// unused in the in-memory API.
	_ = preflight
	_ = policy
	if cfg.PreflightJSONPath != "" || cfg.CanonicalR4ReceiptPath != "" || cfg.CanonicalR5ReceiptPath != "" {
		if err := receipt.ValidateBound(); err != nil {
			return result, err
		}
	} else if err := receipt.Validate(); err != nil {
		return result, err
	}
	if strings.TrimSpace(cfg.OutputReceiptJSONPath) != "" {
		if err := writeAOQTV7R6ProgressiveScreenReceipt(cfg.OutputReceiptJSONPath, receipt); err != nil {
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

// Alias entry point.
func RunAOQTSidecarProgressiveScreen(cfg AOQTSidecarProgressiveScreenConfig) (AOQTSidecarProgressiveScreenResult, error) {
	return RunAOQTV7R6ProgressiveScreen(cfg)
}

func r4EndpointByOrdinal(endpoints []AOQTSidecarActualCoordinateProbeEndpoint, ordinal int) AOQTSidecarActualCoordinateProbeEndpoint {
	for _, endpoint := range endpoints {
		if endpoint.Ordinal == ordinal {
			return endpoint
		}
	}
	return AOQTSidecarActualCoordinateProbeEndpoint{}
}

func aoqtV7R6EvaluateStage(trainer *AOQTSidecarTrainer, objective AOQTSidecarVectorObjective, policy AOQTSidecarCandidateEligibilityPolicy, set AOQTSidecarCalibrationSet, sample AOQTSidecarV7R6ProgressiveScreenSample, canonical []AOQTSidecarActualCoordinateProbeEndpoint) (AOQTSidecarV7R6ProgressiveScreenStage, []AOQTSidecarV7R6ProgressiveScreenEndpoint, error) {
	rows, err := aoqtV7R6SampleRows(set.Rows, sample)
	if err != nil {
		return AOQTSidecarV7R6ProgressiveScreenStage{}, nil, err
	}
	rows, weights, err := aoqtV7R6ApplyHTWeights(rows, sample)
	if err != nil {
		return AOQTSidecarV7R6ProgressiveScreenStage{}, nil, err
	}
	baselineLoss, baselineActivation, baselineComponents, weighted, err := aoqtV7R6EvaluateBaseline(rows, sample, objective, trainer)
	if err != nil {
		return AOQTSidecarV7R6ProgressiveScreenStage{}, nil, err
	}
	if !isFinite32(baselineLoss) || !aoqtFiniteReceiptComponentsValue(baselineComponents) || !aoqtV7R6PositiveActivation(baselineActivation) {
		return AOQTSidecarV7R6ProgressiveScreenStage{}, nil, fmt.Errorf("AOQT V7-r6 %s baseline requires finite loss and positive activation for all six components", sample.Name)
	}
	baseline := aoqtStepEvaluation{loss: baselineLoss, activation: baselineActivation, components: baselineComponents}
	stage := AOQTSidecarV7R6ProgressiveScreenStage{Name: sample.Name, Sample: sample, Weights: weights, BaselineLoss: baselineLoss, BaselineLossFinite: true, BaselineComponents: baselineComponents, BaselineActivation: baselineActivation, ObjectiveEvaluationCount: AOQTV7R6ProgressiveScreenStageObjectiveEvaluations}
	outcomes := make([]AOQTSidecarV7R6ProgressiveScreenEndpoint, 0, len(canonical))
	for _, source := range canonical {
		endpoint := aoqtV7R6EndpointFromCanonical(source)
		spec := aoqtActualCoordinateEndpointSpec{ordinal: source.Ordinal, coordinate: source.CoordinateIndex, analyticDirection: source.AnalyticDirection, magnitude: source.Magnitude, globalSign: source.GlobalSign}
		evaluated := trainer.evaluateActualCoordinateEndpoint(rows, objective, baseline, set.Manifest.ObjectiveContract.WeightSums, spec)
		if evaluated.ActualDirectionSHA256 != source.ActualDirectionSHA256 {
			return AOQTSidecarV7R6ProgressiveScreenStage{}, nil, fmt.Errorf("AOQT V7-r6 %s endpoint %d actual direction changed from canonical r4", sample.Name, source.Ordinal)
		}
		endpoint.S512 = aoqtV7R6StageEndpoint(evaluated, baseline)
		endpoint.S1024 = endpoint.S512
		if !endpoint.S512.BaselineActivationPositive() || !endpoint.S512.CandidateActivationPositive() {
			return AOQTSidecarV7R6ProgressiveScreenStage{}, nil, fmt.Errorf("AOQT V7-r6 %s endpoint %d loses positive six-component activation", sample.Name, source.Ordinal)
		}
		outcomes = append(outcomes, endpoint)
		stage.EndpointOrdinals = append(stage.EndpointOrdinals, endpoint.Ordinal)
	}
	stage.BaselineWeightedActivation = weighted
	_ = policy
	return stage, outcomes, nil
}

func aoqtV7R6EvaluateFullVerification(trainer *AOQTSidecarTrainer, objective AOQTSidecarVectorObjective, policy AOQTSidecarCandidateEligibilityPolicy, set AOQTSidecarCalibrationSet, canonical []AOQTSidecarActualCoordinateProbeEndpoint, selected []int) (AOQTSidecarV7R6ProgressiveScreenFullVerification, error) {
	full := AOQTSidecarV7R6ProgressiveScreenFullVerification{
		Enabled:                  true,
		ObjectiveEvaluationCount: AOQTV7R6ProgressiveScreenFullObjectiveEvaluations,
		RowCount:                 len(set.Rows),
		WeightingVersion:         AOQTV7R6ProgressiveScreenFullWeightingVersion,
		HTFactorsApplied:         false,
		EndpointOrdinals:         append([]int(nil), selected...),
		DenseGateStatus:          AOQTV7R6ProgressiveScreenDenseGateStatus,
		DenseGateDeferred:        true,
	}
	if len(selected) != AOQTV7R6ProgressiveScreenSelectedK {
		return full, fmt.Errorf("AOQT V7-r6 full verification requires exactly %d selected endpoints", AOQTV7R6ProgressiveScreenSelectedK)
	}
	// Full verification is deliberately not another sample estimate. It is the
	// unweighted prepared-IP objective over the complete bound fold.
	rows := append([]AOQTSidecarCalibrationRow(nil), set.Rows...)
	var err error
	full.BaselineLoss, full.BaselineActivation, full.BaselineComponents, err = aoqtV7R6EvaluateUnweightedBaseline(rows, objective, trainer)
	if err != nil {
		return full, err
	}
	if !isFinite32(full.BaselineLoss) {
		return full, fmt.Errorf("AOQT V7-r6 full verification baseline is non-finite")
	}
	full.BaselineLossFinite = true
	if !aoqtV7R6PositiveActivation(full.BaselineActivation) || !aoqtFiniteReceiptComponentsValue(full.BaselineComponents) {
		return full, fmt.Errorf("AOQT V7-r6 full verification baseline lacks positive six-component activation")
	}
	byOrdinal := make(map[int]AOQTSidecarActualCoordinateProbeEndpoint, len(canonical))
	for _, endpoint := range canonical {
		byOrdinal[endpoint.Ordinal] = endpoint
	}
	baseline := aoqtStepEvaluation{loss: full.BaselineLoss, activation: full.BaselineActivation, components: full.BaselineComponents}
	for _, ordinal := range selected {
		source, ok := byOrdinal[ordinal]
		if !ok {
			return full, fmt.Errorf("AOQT V7-r6 full verification endpoint %d is not canonical", ordinal)
		}
		evaluated := trainer.evaluateActualCoordinateEndpoint(rows, objective, baseline, set.Manifest.ObjectiveContract.WeightSums, aoqtActualCoordinateEndpointSpec{ordinal: source.Ordinal, coordinate: source.CoordinateIndex, analyticDirection: source.AnalyticDirection, magnitude: source.Magnitude, globalSign: source.GlobalSign})
		if evaluated.ActualDirectionSHA256 != source.ActualDirectionSHA256 {
			return full, fmt.Errorf("AOQT V7-r6 full verification endpoint %d actual direction changed from canonical r4", ordinal)
		}
		evidence := aoqtV7R6StageEndpoint(evaluated, baseline)
		if !evidence.BaselineActivationPositive() || !evidence.CandidateActivationPositive() || !evidence.TotalDeltaFinite {
			return full, fmt.Errorf("AOQT V7-r6 full verification endpoint %d lacks finite positive objective evidence", ordinal)
		}
		full.Endpoints = append(full.Endpoints, AOQTSidecarV7R6ProgressiveScreenFullObjectiveEndpoint{Ordinal: ordinal, Evidence: evidence, Feasible: aoqtV7R6EndpointFeasibleWithPolicy(evidence, policy), DenseGateStatus: AOQTV7R6ProgressiveScreenDenseGateStatus})
	}
	full.Completed = true
	return full, nil
}

func aoqtV7R6RequireSixComponents(set AOQTSidecarCalibrationSet, samples AOQTSidecarV7R6ProgressiveScreenSamples) error {
	required := aoqtV7R4ActualCoordinateRequiredComponents()
	for _, sample := range []AOQTSidecarV7R6ProgressiveScreenSample{samples.S512, samples.S1024} {
		rows, err := aoqtV7R6SampleRows(set.Rows, sample)
		if err != nil {
			return err
		}
		seen := map[string]bool{}
		for _, row := range rows {
			for name := range aoqtActualCoordinateRowComponents(row, set.Manifest.ObjectiveContract) {
				seen[name] = true
			}
			if row.SplitProof.Split != "train" || !row.SplitProof.TrainOnly {
				return fmt.Errorf("AOQT V7-r6 %s has non-train row %q", sample.Name, row.RowID)
			}
		}
		for _, name := range required {
			if !seen[name] {
				return fmt.Errorf("AOQT V7-r6 %s cannot activate required component %q", sample.Name, name)
			}
		}
	}
	return nil
}

func validateAOQTV7R6CanonicalReceipts(r4 AOQTSidecarActualCoordinateProbeReceipt, r5 AOQTSidecarActualCoordinateTrainReceipt, set AOQTSidecarCalibrationSet, split *AOQTSidecarDevSplitBinding) error {
	if err := r4.Validate(); err != nil {
		return fmt.Errorf("AOQT V7-r6 canonical r4 receipt: %w", err)
	}
	if !r4.Passed || len(r4.Endpoints) == 0 {
		return fmt.Errorf("AOQT V7-r6 canonical r4 receipt must be a passed endpoint set")
	}
	if err := r5.Validate(); err != nil {
		return fmt.Errorf("AOQT V7-r6 canonical r5 receipt: %w", err)
	}
	if r5.CanonicalR4ReceiptSHA256 != AOQTV7R6CanonicalR4ReceiptFileSHA256 {
		return fmt.Errorf("AOQT V7-r6 r5 receipt does not carry pinned canonical r4 hash")
	}
	r5SHA, err := r5.SHA256()
	if err != nil {
		return fmt.Errorf("AOQT V7-r6 canonical r5 receipt hash: %w", err)
	}
	if r5SHA != AOQTV7R6CanonicalR5ReceiptSHA256 {
		return fmt.Errorf("AOQT V7-r6 canonical r5 receipt object sha256 %s is not the pinned authorization", r5SHA)
	}
	if r4.Binding.SplitBinding == nil || !reflect.DeepEqual(r4.Binding.SplitBinding, split) {
		return fmt.Errorf("AOQT V7-r6 r4 fold binding mismatch")
	}
	if r5.Binding.SplitBinding == nil || !reflect.DeepEqual(r5.Binding.SplitBinding, split) {
		return fmt.Errorf("AOQT V7-r6 r5 fold binding mismatch")
	}
	if !reflect.DeepEqual(r5.Binding.Inputs, r4.Binding.Inputs) || !reflect.DeepEqual(r5.Binding.IOReport, r4.Binding.IOReport) || r5.Binding.PreflightPath != r4.Binding.PreflightPath || r5.Binding.PreflightSHA256 != r4.Binding.PreflightSHA256 {
		return fmt.Errorf("AOQT V7-r6 r4/r5 input binding mismatch")
	}
	if r5.Binding.CanonicalR4ReceiptSHA256 != "" && r5.Binding.CanonicalR4ReceiptSHA256 != AOQTV7R6CanonicalR4ReceiptFileSHA256 {
		return fmt.Errorf("AOQT V7-r6 r5 canonical r4 authorization mismatch")
	}
	manifestSHA, err := AOQTSidecarManifestSHA256(set.Manifest)
	if err != nil {
		return err
	}
	if r4.Binding.Inputs.DatasetManifestSHA256 != manifestSHA || r5.Binding.Inputs.DatasetManifestSHA256 != manifestSHA {
		return fmt.Errorf("AOQT V7-r6 r4/r5 manifest hash does not match fold input")
	}
	contractSHA, err := aoqtActualCoordinateObjectiveContractSHA256(set.Manifest.ObjectiveContract)
	if err != nil {
		return err
	}
	policy, err := AOQTSidecarTrainingContractPolicy(set.Manifest.TrainingContract, set.Manifest.CandidateEligibilityPolicy)
	if err != nil {
		return err
	}
	policySHA, err := aoqtActualCoordinatePolicySHA256(policy)
	if err != nil {
		return err
	}
	if r4.ObjectiveContractSHA256 != contractSHA || r4.EligibilityPolicySHA256 != policySHA || r4.Binding.ObjectiveContractSHA256 != contractSHA || r4.Binding.EligibilityPolicySHA256 != policySHA || r5.Binding.ObjectiveContractSHA256 != contractSHA || r5.Binding.EligibilityPolicySHA256 != policySHA {
		return fmt.Errorf("AOQT V7-r6 r4/r5 objective or eligibility policy hash does not match fold input")
	}
	if r4.Binding.IOReport.RowsSHA256 != "" && r5.Binding.IOReport.RowsSHA256 != "" && r4.Binding.IOReport.RowsSHA256 != r5.Binding.IOReport.RowsSHA256 {
		return fmt.Errorf("AOQT V7-r6 r4/r5 rows hash mismatch")
	}
	if len(r4.Endpoints) != 384 || r4.EndpointCount != 384 {
		return fmt.Errorf("AOQT V7-r6 canonical r4 endpoint set must contain 384 endpoints")
	}
	if _, err := AOQTV7R6CanonicalR4EndpointSetSHA256(r4); err != nil {
		return fmt.Errorf("AOQT V7-r6 canonical r4 endpoint set: %w", err)
	}
	return nil
}

// rereadAOQTV7R6Binding is the authoritative provenance check for a persisted
// screen receipt.  Receipt self-digests are useful diagnostics, but they are
// not authority: every source file is re-read, hashed, parsed, and compared to
// the receipt's binding before the sample and endpoint-set hashes are trusted.
// This function is intentionally provenance-only; it does not silently replay
// 71 objectives during a receipt reread. Receipt.Validate performs structural
// evidence checks, while the execution path records the actual stage/full
// objective results from fresh identity trainers.
func rereadAOQTV7R6Binding(r AOQTSidecarV7R6ProgressiveScreenReceipt) error {
	binding := r.Binding
	manifestData, err := os.ReadFile(binding.IOReport.ManifestPath)
	if err != nil {
		return fmt.Errorf("AOQT V7-r6 bound manifest reread: %w", err)
	}
	var manifest AOQTSidecarCalibrationManifest
	if err := strictUnmarshalAOQT(manifestData, &manifest); err != nil {
		return fmt.Errorf("AOQT V7-r6 bound manifest is invalid: %w", err)
	}
	if err := manifest.Validate(); err != nil {
		return fmt.Errorf("AOQT V7-r6 bound manifest validation: %w", err)
	}
	manifestSHA, err := AOQTSidecarManifestSHA256(manifest)
	if err != nil {
		return fmt.Errorf("AOQT V7-r6 bound manifest canonical hash: %w", err)
	}
	if manifestSHA != binding.Inputs.DatasetManifestSHA256 || manifestSHA != binding.IOReport.ManifestSHA256 {
		return fmt.Errorf("AOQT V7-r6 bound manifest canonical sha256 mismatch")
	}
	rowsData, err := os.ReadFile(binding.IOReport.RowsJSONLPath)
	if err != nil {
		return fmt.Errorf("AOQT V7-r6 bound rows reread: %w", err)
	}
	rowsSHA := sha256BytesAOQT(rowsData)
	if rowsSHA != binding.IOReport.RowsSHA256 || rowsSHA != r.FoldInputRowsSHA256 {
		return fmt.Errorf("AOQT V7-r6 bound rows sha256 mismatch")
	}
	rows, err := parseAOQTCalibrationRowsJSONL(rowsData, binding.IOReport.RowsJSONLPath)
	if err != nil {
		return fmt.Errorf("AOQT V7-r6 bound rows are invalid: %w", err)
	}
	set := AOQTSidecarCalibrationSet{Manifest: manifest, Rows: rows}
	if err := set.Validate(); err != nil {
		return fmt.Errorf("AOQT V7-r6 bound calibration set validation: %w", err)
	}
	if err := aoqtV7R6RejectOfficialOverlap(rows); err != nil {
		return err
	}
	if manifest.RowIDSHA256 != r.FoldInputRowIDSHA256 || manifest.RowCount != len(rows) || binding.IOReport.RowCount != len(rows) {
		return fmt.Errorf("AOQT V7-r6 bound row identity/count does not match receipt")
	}
	contractSHA, err := aoqtActualCoordinateObjectiveContractSHA256(manifest.ObjectiveContract)
	if err != nil {
		return err
	}
	if r.CanonicalObjectiveContractSHA256 != contractSHA {
		return fmt.Errorf("AOQT V7-r6 objective contract hash does not match reread fold manifest")
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
		return fmt.Errorf("AOQT V7-r6 eligibility policy hash does not match reread fold manifest")
	}
	wantIO, err := aoqtV7R6IOReportForSet(set, binding.IOReport.ManifestPath, binding.IOReport.RowsJSONLPath, rowsSHA)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(wantIO, binding.IOReport) {
		return fmt.Errorf("AOQT V7-r6 bound IO report does not match reread calibration inputs")
	}

	preflightCfg := AOQTSidecarTrainRunnerConfig{PreflightJSONPath: binding.PreflightPath, ExpectedPreflightSHA256: binding.PreflightSHA256}
	preflight, preflightSHA, err := loadAndValidateAOQTTrainPreflight(preflightCfg, set, binding.IOReport)
	if err != nil {
		return fmt.Errorf("AOQT V7-r6 bound preflight validation: %w", err)
	}
	if preflightSHA != binding.PreflightSHA256 {
		return fmt.Errorf("AOQT V7-r6 bound preflight sha256 mismatch")
	}
	if binding.SplitBinding == nil {
		return fmt.Errorf("AOQT V7-r6 bound split binding is required")
	}
	splitCfg := AOQTSidecarTrainRunnerConfig{
		SplitManifestPath:           binding.SplitBinding.SplitManifestPath,
		ExpectedSplitManifestSHA256: binding.SplitBinding.SplitManifestFileSHA256,
		FoldID:                      binding.FoldID,
	}
	split, err := validateAOQTV7DevSplitBinding(splitCfg, set, binding.IOReport, preflight, preflightSHA)
	if err != nil {
		return fmt.Errorf("AOQT V7-r6 bound split validation: %w", err)
	}
	if split == nil || !reflect.DeepEqual(split, binding.SplitBinding) {
		return fmt.Errorf("AOQT V7-r6 bound split binding does not match reread split manifest")
	}

	r4Data, err := os.ReadFile(binding.CanonicalR4ReceiptPath)
	if err != nil {
		return fmt.Errorf("AOQT V7-r6 bound r4 receipt reread: %w", err)
	}
	r4FileSHA := sha256BytesAOQT(r4Data)
	if r4FileSHA != binding.CanonicalR4ReceiptFileSHA256 || r4FileSHA != AOQTV7R6CanonicalR4ReceiptFileSHA256 {
		return fmt.Errorf("AOQT V7-r6 bound r4 receipt file sha256 mismatch")
	}
	var r4 AOQTSidecarActualCoordinateProbeReceipt
	if err := strictUnmarshalAOQT(r4Data, &r4); err != nil {
		return fmt.Errorf("AOQT V7-r6 bound r4 receipt is invalid: %w", err)
	}
	if err := r4.ValidateBound(); err != nil {
		return fmt.Errorf("AOQT V7-r6 bound r4 receipt validation: %w", err)
	}
	r4SHA, err := r4.SHA256()
	if err != nil {
		return err
	}
	if r4SHA != binding.CanonicalR4ReceiptSHA256 || !reflect.DeepEqual(r4.Binding.SplitBinding, binding.SplitBinding) || !reflect.DeepEqual(r4.Binding.Inputs, binding.Inputs) || !reflect.DeepEqual(r4.Binding.IOReport, binding.IOReport) || r4.Binding.PreflightPath != binding.PreflightPath || r4.Binding.PreflightSHA256 != binding.PreflightSHA256 {
		return fmt.Errorf("AOQT V7-r6 bound r4 receipt provenance does not match screen binding")
	}

	r5Data, err := os.ReadFile(binding.CanonicalR5ReceiptPath)
	if err != nil {
		return fmt.Errorf("AOQT V7-r6 bound r5 receipt reread: %w", err)
	}
	r5FileSHA := sha256BytesAOQT(r5Data)
	if r5FileSHA != binding.CanonicalR5ReceiptFileSHA256 {
		return fmt.Errorf("AOQT V7-r6 bound r5 receipt file sha256 mismatch")
	}
	var r5 AOQTSidecarActualCoordinateTrainReceipt
	if err := strictUnmarshalAOQT(r5Data, &r5); err != nil {
		return fmt.Errorf("AOQT V7-r6 bound r5 receipt is invalid: %w", err)
	}
	if err := r5.ValidateBound(); err != nil {
		return fmt.Errorf("AOQT V7-r6 bound r5 receipt validation: %w", err)
	}
	r5SHA, err := r5.SHA256()
	if err != nil {
		return err
	}
	if r5SHA != binding.CanonicalR5ReceiptSHA256 || !reflect.DeepEqual(r5.Binding.SplitBinding, binding.SplitBinding) || !reflect.DeepEqual(r5.Binding.Inputs, binding.Inputs) || !reflect.DeepEqual(r5.Binding.IOReport, binding.IOReport) || r5.Binding.PreflightPath != binding.PreflightPath || r5.Binding.PreflightSHA256 != binding.PreflightSHA256 {
		return fmt.Errorf("AOQT V7-r6 bound r5 receipt provenance does not match screen binding")
	}
	if r5SHA != AOQTV7R6CanonicalR5ReceiptSHA256 || r5FileSHA != AOQTV7R6CanonicalR5ReceiptFileSHA256 {
		return fmt.Errorf("AOQT V7-r6 bound r5 receipt is not the pinned authorization")
	}
	if r5.CanonicalR4ReceiptSHA256 != AOQTV7R6CanonicalR4ReceiptFileSHA256 || r5.CanonicalR4ReceiptFileSHA256 != r4FileSHA {
		return fmt.Errorf("AOQT V7-r6 bound r5 receipt does not bind canonical r4")
	}
	if err := validateAOQTV7R6BoundContractHashes(r, manifest, r4, r5); err != nil {
		return err
	}
	if r.CanonicalR4ScheduleSHA256 != r4.ScheduleSHA256 || r.CanonicalR4RankingSHA256 != r4.RankingSHA256 || r.CanonicalR4PolicySHA256 != r4.EligibilityPolicySHA256 {
		return fmt.Errorf("AOQT V7-r6 canonical r4 schedule/ranking/policy binding mismatch")
	}
	endpointSetSHA, err := AOQTV7R6CanonicalR4EndpointSetSHA256(r4)
	if err != nil {
		return err
	}
	if endpointSetSHA != r.R4EndpointSetSHA256 {
		return fmt.Errorf("AOQT V7-r6 canonical r4 endpoint-set hash mismatch")
	}
	if err := validateAOQTV7R6ReceiptAgainstCanonicalR4Endpoints(r, r4); err != nil {
		return err
	}
	selectionSeed, err := aoqtV7R6SelectionSeed(endpointSetSHA, r5SHA, set, binding.FoldID)
	if err != nil {
		return err
	}
	if selectionSeed != r.SelectionSeedSHA256 {
		return fmt.Errorf("AOQT V7-r6 selection seed hash mismatch")
	}
	samples, err := ConstructAOQTV7R6ProgressiveScreenSamples(set.Rows, selectionSeed)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(samples.S512, r.StageS512.Sample) || !reflect.DeepEqual(samples.S1024, r.StageS1024.Sample) {
		return fmt.Errorf("AOQT V7-r6 persisted samples do not match deterministic fold reconstruction")
	}
	return nil
}

func validateAOQTV7R6BoundContractHashes(r AOQTSidecarV7R6ProgressiveScreenReceipt, manifest AOQTSidecarCalibrationManifest, r4 AOQTSidecarActualCoordinateProbeReceipt, r5 AOQTSidecarActualCoordinateTrainReceipt) error {
	contractSHA, err := aoqtActualCoordinateObjectiveContractSHA256(manifest.ObjectiveContract)
	if err != nil {
		return err
	}
	if r.CanonicalObjectiveContractSHA256 != contractSHA {
		return fmt.Errorf("AOQT V7-r6 objective contract hash does not match reread fold manifest")
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
		return fmt.Errorf("AOQT V7-r6 eligibility policy hash does not match reread fold manifest")
	}
	if r4.ObjectiveContractSHA256 != contractSHA || r4.Binding.ObjectiveContractSHA256 != contractSHA || r5.Binding.ObjectiveContractSHA256 != contractSHA {
		return fmt.Errorf("AOQT V7-r6 canonical r4/r5 objective contract does not match reread fold manifest")
	}
	if r4.EligibilityPolicySHA256 != policySHA || r4.Binding.EligibilityPolicySHA256 != policySHA || r5.Binding.EligibilityPolicySHA256 != policySHA {
		return fmt.Errorf("AOQT V7-r6 canonical r4/r5 eligibility policy does not match reread fold manifest")
	}
	return nil
}

func validateAOQTV7R6ReceiptAgainstCanonicalR4Endpoints(r AOQTSidecarV7R6ProgressiveScreenReceipt, r4 AOQTSidecarActualCoordinateProbeReceipt) error {
	byOrdinal := make(map[int]AOQTSidecarActualCoordinateProbeEndpoint, len(r4.Endpoints))
	for _, endpoint := range r4.Endpoints {
		byOrdinal[endpoint.Ordinal] = endpoint
	}
	screenByOrdinal := make(map[int]AOQTSidecarV7R6ProgressiveScreenEndpoint, len(r.Endpoints))
	for _, endpoint := range r.Endpoints {
		source, ok := byOrdinal[endpoint.Ordinal]
		if !ok {
			return fmt.Errorf("AOQT V7-r6 endpoint %d is absent from canonical r4 receipt", endpoint.Ordinal)
		}
		if err := validateAOQTV7R6EndpointMetadataMatchesR4(endpoint, source); err != nil {
			return err
		}
		if err := validateAOQTV7R6EvidenceActualDirectionMatchesR4(endpoint.S512, source, fmt.Sprintf("AOQT V7-r6 endpoint %d S512", endpoint.Ordinal)); err != nil {
			return err
		}
		if err := validateAOQTV7R6EvidenceActualDirectionMatchesR4(endpoint.S1024, source, fmt.Sprintf("AOQT V7-r6 endpoint %d S1024", endpoint.Ordinal)); err != nil {
			return err
		}
		screenByOrdinal[endpoint.Ordinal] = endpoint
	}
	for i, selected := range r.SelectedEndpointOrdinals {
		source, ok := byOrdinal[selected]
		if !ok {
			return fmt.Errorf("AOQT V7-r6 selected endpoint %d is absent from canonical r4 receipt", selected)
		}
		endpoint, ok := screenByOrdinal[selected]
		if !ok {
			return fmt.Errorf("AOQT V7-r6 selected endpoint %d is absent from screen evidence", selected)
		}
		if err := validateAOQTV7R6EndpointMetadataMatchesR4(endpoint, source); err != nil {
			return err
		}
		if i >= len(r.FullVerification.Endpoints) {
			continue
		}
		full := r.FullVerification.Endpoints[i]
		if full.Ordinal != selected {
			return fmt.Errorf("AOQT V7-r6 full verification endpoint %d does not match selected ordinal %d", full.Ordinal, selected)
		}
		if err := validateAOQTV7R6EvidenceActualDirectionMatchesR4(full.Evidence, source, fmt.Sprintf("AOQT V7-r6 full verification endpoint %d", selected)); err != nil {
			return err
		}
	}
	return nil
}

func validateAOQTV7R6EndpointMetadataMatchesR4(endpoint AOQTSidecarV7R6ProgressiveScreenEndpoint, source AOQTSidecarActualCoordinateProbeEndpoint) error {
	if endpoint.Ordinal != source.Ordinal ||
		endpoint.Kind != source.Kind ||
		endpoint.CoordinateIndex != source.CoordinateIndex ||
		endpoint.AnalyticDirection != source.AnalyticDirection ||
		endpoint.GlobalSign != source.GlobalSign ||
		endpoint.Magnitude != source.Magnitude ||
		endpoint.RequestedDirectionSHA256 != source.RequestedDirectionSHA256 ||
		endpoint.CanonicalActualDirectionSHA256 != source.ActualDirectionSHA256 ||
		!reflect.DeepEqual(endpoint.Indices, source.Indices) ||
		!reflect.DeepEqual(endpoint.Directions, source.Directions) {
		return fmt.Errorf("AOQT V7-r6 endpoint %d metadata does not match canonical r4", endpoint.Ordinal)
	}
	return nil
}

func validateAOQTV7R6EvidenceActualDirectionMatchesR4(evidence AOQTSidecarV7R6ProgressiveScreenStageEndpoint, source AOQTSidecarActualCoordinateProbeEndpoint, label string) error {
	if evidence.ActualDirectionSHA256 != source.ActualDirectionSHA256 ||
		evidence.ActualDirectionMaxAbs != source.ActualDirectionMaxAbs ||
		len(evidence.ActualDirection) != len(source.ActualDirection) ||
		len(evidence.ActualDirection) != AOQTSidecarAngleCount ||
		!reflect.DeepEqual(evidence.ActualDirection, source.ActualDirection) {
		return fmt.Errorf("%s actual direction does not match canonical r4 endpoint evidence", label)
	}
	return nil
}

func aoqtV7R6IOReportForSet(set AOQTSidecarCalibrationSet, manifestPath, rowsPath, rowsSHA string) (AOQTSidecarCalibrationIOReport, error) {
	manifestSHA, err := AOQTSidecarManifestSHA256(set.Manifest)
	if err != nil {
		return AOQTSidecarCalibrationIOReport{}, err
	}
	if strings.TrimSpace(rowsSHA) == "" {
		rowsSHA, err = aoqtV7R6CanonicalRowsSHA256(set.Rows)
		if err != nil {
			return AOQTSidecarCalibrationIOReport{}, err
		}
	}
	return AOQTSidecarCalibrationIOReport{
		Schema:                      AOQTSidecarCalibrationIOReportSchema,
		ManifestPath:                manifestPath,
		RowsJSONLPath:               rowsPath,
		ManifestSHA256:              manifestSHA,
		RowsSHA256:                  rowsSHA,
		AnchorArtifactSHA256:        set.Manifest.AnchorArtifactSHA256,
		AnchorPackageManifestSHA256: set.Manifest.AnchorPackageManifestSHA256,
		AnchorEmbeddingSpaceID:      set.Manifest.AnchorEmbeddingSpaceID,
		CompatibilityDigest:         set.Manifest.CompatibilityDigest,
		TurboQuantSeed:              set.Manifest.TurboQuantSeed,
		Topology:                    set.Manifest.Topology,
		ObjectiveContract:           set.Manifest.ObjectiveContract,
		LegalGates:                  set.Manifest.LegalGates,
		RowCount:                    len(set.Rows),
		CandidateCount:              countAOQTCandidates(set.Rows),
		PairCount:                   countAOQTPairs(set.Rows),
		TrainingContract:            set.Manifest.TrainingContract,
		CandidateEligibilityPolicy:  cloneAOQTSidecarCandidateEligibilityPolicy(set.Manifest.CandidateEligibilityPolicy),
	}, nil
}

// Direct in-memory callers do not have a JSONL byte stream to hash.  Use the
// same newline-delimited JSON representation used by the materializer so the
// receipt still carries a deterministic rows-content digest; file-backed
// callers pass the raw file hash and never take this fallback.
func aoqtV7R6CanonicalRowsSHA256(rows []AOQTSidecarCalibrationRow) (string, error) {
	h := sha256.New()
	for _, row := range rows {
		data, err := json.Marshal(row)
		if err != nil {
			return "", err
		}
		if _, err := h.Write(data); err != nil {
			return "", err
		}
		if _, err := h.Write([]byte{'\n'}); err != nil {
			return "", err
		}
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func aoqtV7R6RejectOfficialOverlap(rows []AOQTSidecarCalibrationRow) error {
	for _, row := range rows {
		values := []string{row.RowID, row.Dataset, row.QueryID, row.QueryVectorID, row.GuardClass, row.SourceArtifactHash, row.QrelsSHA256, row.CompatibilityDigest}
		values = append(values, row.CandidateDocIDs...)
		values = append(values, row.CandidateVectorIDs...)
		values = append(values, row.CandidateSources...)
		for _, value := range values {
			if strings.Contains(strings.ToLower(value), "official") {
				return fmt.Errorf("AOQT V7-r6 official overlap in row %q", row.RowID)
			}
		}
		if len(row.Extra) > 0 {
			data, err := json.Marshal(row.Extra)
			if err != nil {
				return fmt.Errorf("AOQT V7-r6 row %q extra metadata cannot be inspected: %w", row.RowID, err)
			}
			if strings.Contains(strings.ToLower(string(data)), "official") {
				return fmt.Errorf("AOQT V7-r6 official overlap in row %q extra metadata", row.RowID)
			}
		}
	}
	return nil
}

func aoqtV7R6ResolveInputs(cfg AOQTSidecarV7R6ProgressiveScreenConfig) (AOQTSidecarCalibrationSet, AOQTSidecarCalibrationIOReport, AOQTSidecarMaterializePreflight, string, *AOQTSidecarDevSplitBinding, *AOQTSidecarActualCoordinateProbeReceipt, *AOQTSidecarActualCoordinateTrainReceipt, string, string, error) {
	var set AOQTSidecarCalibrationSet
	var ioReport AOQTSidecarCalibrationIOReport
	var preflight AOQTSidecarMaterializePreflight
	var preflightSHA string
	var split *AOQTSidecarDevSplitBinding
	var err error
	if cfg.Set != nil {
		set = *cfg.Set
		ioReport, err = aoqtV7R6IOReportForSet(set, cfg.ManifestPath, cfg.RowsJSONLPath, cfg.ExpectedRowsSHA256)
		if err != nil {
			return set, ioReport, preflight, preflightSHA, split, nil, nil, "", "", err
		}
	} else {
		expectedTopology, e := expectedAOQTSidecarTrainTopology()
		if e != nil {
			return set, ioReport, preflight, preflightSHA, split, nil, nil, "", "", e
		}
		set, ioReport, err = LoadAOQTSidecarCalibrationSet(AOQTSidecarCalibrationIOConfig{ManifestPath: cfg.ManifestPath, RowsJSONLPath: cfg.RowsJSONLPath, ExpectedManifestSHA256: cfg.ExpectedManifestSHA256, ExpectedRowsSHA256: cfg.ExpectedRowsSHA256, ExpectedAnchorArtifactSHA256: cfg.ExpectedAnchorArtifactSHA256, ExpectedAnchorPackageManifestSHA256: cfg.ExpectedAnchorPackageManifestSHA256, ExpectedAnchorEmbeddingSpaceID: cfg.ExpectedAnchorEmbeddingSpaceID, ExpectedCompatibilityDigest: cfg.ExpectedCompatibilityDigest, ExpectedTurboQuantSeed: AOQTSidecarMaterializerQuantSeed, ExpectedTopology: expectedTopology, ExpectedSourceArtifactHashes: cfg.ExpectedSourceArtifactHashes, ExpectedVectorCacheHashes: cfg.ExpectedVectorCacheHashes})
		if err != nil {
			return set, ioReport, preflight, preflightSHA, split, nil, nil, "", "", err
		}
	}
	if cfg.PreflightJSONPath != "" {
		runnerCfg := AOQTSidecarTrainRunnerConfig{PreflightJSONPath: cfg.PreflightJSONPath, ExpectedPreflightSHA256: cfg.ExpectedPreflightSHA256}
		preflight, preflightSHA, err = loadAndValidateAOQTTrainPreflight(runnerCfg, set, ioReport)
		if err != nil {
			return set, ioReport, preflight, preflightSHA, split, nil, nil, "", "", err
		}
	} else {
		preflightSHA = strings.Repeat("0", 64)
	}
	if cfg.SplitManifestPath != "" {
		runnerCfg := AOQTSidecarTrainRunnerConfig{SplitManifestPath: cfg.SplitManifestPath, ExpectedSplitManifestSHA256: cfg.ExpectedSplitManifestSHA256, FoldID: cfg.FoldID}
		split, err = validateAOQTV7DevSplitBinding(runnerCfg, set, ioReport, preflight, preflightSHA)
		if err != nil {
			return set, ioReport, preflight, preflightSHA, split, nil, nil, "", "", err
		}
	} else if cfg.CanonicalR4Receipt != nil && cfg.CanonicalR4Receipt.Binding.SplitBinding != nil {
		split = cloneAOQTDevSplitBinding(cfg.CanonicalR4Receipt.Binding.SplitBinding)
	}
	r4 := cfg.CanonicalR4Receipt
	r5 := cfg.CanonicalR5Receipt
	r4FileSHA, r5FileSHA := "", ""
	if cfg.CanonicalR4ReceiptPath != "" {
		data, e := os.ReadFile(cfg.CanonicalR4ReceiptPath)
		if e != nil {
			return set, ioReport, preflight, preflightSHA, split, nil, nil, "", "", e
		}
		r4FileSHA = sha256BytesAOQT(data)
		if r4FileSHA != AOQTV7R6CanonicalR4ReceiptFileSHA256 {
			return set, ioReport, preflight, preflightSHA, split, nil, nil, r4FileSHA, "", fmt.Errorf("AOQT V7-r6 canonical r4 receipt is not the pinned authorization")
		}
		var value AOQTSidecarActualCoordinateProbeReceipt
		if e := strictUnmarshalAOQT(data, &value); e != nil {
			return set, ioReport, preflight, preflightSHA, split, nil, nil, r4FileSHA, "", e
		}
		r4 = &value
		if err := value.ValidateBound(); err != nil {
			return set, ioReport, preflight, preflightSHA, split, nil, nil, r4FileSHA, "", fmt.Errorf("AOQT V7-r6 canonical r4 bound validation: %w", err)
		}
		if cfg.CanonicalR4ReceiptSHA256 != "" && cfg.CanonicalR4ReceiptSHA256 != r4FileSHA {
			return set, ioReport, preflight, preflightSHA, split, nil, nil, r4FileSHA, "", fmt.Errorf("AOQT V7-r6 canonical r4 receipt raw sha256 mismatch")
		}
	}
	if cfg.CanonicalR5ReceiptPath != "" {
		if strings.TrimSpace(cfg.CanonicalR5ReceiptSHA256) == "" {
			return set, ioReport, preflight, preflightSHA, split, nil, nil, r4FileSHA, "", fmt.Errorf("AOQT V7-r6 canonical r5 receipt raw sha256 is required")
		}
		if cfg.CanonicalR5ReceiptSHA256 != AOQTV7R6CanonicalR5ReceiptFileSHA256 {
			return set, ioReport, preflight, preflightSHA, split, nil, nil, r4FileSHA, "", fmt.Errorf("AOQT V7-r6 canonical r5 receipt raw sha256 is not the pinned authorization")
		}
		data, e := os.ReadFile(cfg.CanonicalR5ReceiptPath)
		if e != nil {
			return set, ioReport, preflight, preflightSHA, split, nil, nil, r4FileSHA, "", e
		}
		r5FileSHA = sha256BytesAOQT(data)
		var value AOQTSidecarActualCoordinateTrainReceipt
		if e := strictUnmarshalAOQT(data, &value); e != nil {
			return set, ioReport, preflight, preflightSHA, split, nil, nil, r4FileSHA, r5FileSHA, e
		}
		r5 = &value
		if err := value.ValidateBound(); err != nil {
			return set, ioReport, preflight, preflightSHA, split, nil, nil, r4FileSHA, r5FileSHA, fmt.Errorf("AOQT V7-r6 canonical r5 bound validation: %w", err)
		}
		if cfg.CanonicalR5ReceiptSHA256 != "" && cfg.CanonicalR5ReceiptSHA256 != r5FileSHA {
			return set, ioReport, preflight, preflightSHA, split, nil, nil, r4FileSHA, r5FileSHA, fmt.Errorf("AOQT V7-r6 canonical r5 receipt raw sha256 mismatch")
		}
	}
	return set, ioReport, preflight, preflightSHA, split, r4, r5, r4FileSHA, r5FileSHA, nil
}

func writeAOQTV7R6ProgressiveScreenReceipt(path string, receipt AOQTSidecarV7R6ProgressiveScreenReceipt) error {
	if err := receipt.Validate(); err != nil {
		return err
	}
	if strings.TrimSpace(path) == "" {
		return fmt.Errorf("AOQT V7-r6 receipt output path is required")
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
			return fmt.Errorf("AOQT V7-r6 receipt output %q already exists", path)
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

// Exported writer keeps file-backed tests from relying on the CLI package.
func WriteAOQTV7R6ProgressiveScreenReceipt(path string, receipt AOQTSidecarV7R6ProgressiveScreenReceipt) error {
	return writeAOQTV7R6ProgressiveScreenReceipt(path, receipt)
}
