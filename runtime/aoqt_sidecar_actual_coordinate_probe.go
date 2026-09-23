package eosruntime

// V7-r4 is a deliberately narrow, dev-only probe. Analytic gradients rank
// scalar support only; every eligibility decision below is made by evaluating
// the actual projected prepared-IP objective at the endpoint.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"slices"
	"sort"
	"strconv"
	"strings"
)

const AOQTSidecarActualCoordinateProbeSchema = "eos.aoqt.v7_r4_actual_coordinate_probe.v1"

// AOQTV7R4ScoreDistillActivationSemantics is deliberately distinct from the
// objective's numeric score-distillation loss.  A score-distill row is covered
// structurally when the row has at least two prepared candidates and a
// positive score-distill weight; the loss may legitimately be zero when the
// prepared scores already equal the anchor scores.
const AOQTV7R4ScoreDistillActivationSemantics = "structural_candidate_count_v1"

// Validation of a bound receipt intentionally replays the baseline and every
// endpoint.  Keep the cost explicit in the receipt so an auditor can see that
// this is a correctness replay, not a cheaper hash-only check.
const (
	AOQTV7R4ValidationReplayEndpointCount            = AOQTV7R4ActualCoordinateProbeEndpointCount
	AOQTV7R4ValidationReplayObjectiveEvaluationCount = AOQTV7R4ValidationReplayEndpointCount + 1 // baseline + endpoints
	AOQTV7R4ValidationReplayCost                     = "baseline_plus_384_prepared_ip_endpoint_evaluations"
)

type AOQTSidecarActualCoordinateProbeBinding struct {
	Inputs                  AOQTSidecarRunMetricInputs     `json:"inputs"`
	IOReport                AOQTSidecarCalibrationIOReport `json:"io_report"`
	PreflightPath           string                         `json:"preflight_path"`
	PreflightSHA256         string                         `json:"preflight_sha256"`
	SplitBinding            *AOQTSidecarDevSplitBinding    `json:"split_binding,omitempty"`
	ObjectiveContractSHA256 string                         `json:"objective_contract_sha256"`
	EligibilityPolicySHA256 string                         `json:"eligibility_policy_sha256"`
	ScheduleSHA256          string                         `json:"schedule_sha256"`
	RankingSHA256           string                         `json:"ranking_sha256"`
	CoverageSHA256          string                         `json:"coverage_sha256"`
}

// AOQTSidecarActualCoordinateProbeCoverage records the train-only, static
// coverage proof separately from the objective's runtime activation. Row
// selection uses only these fields, never objective outcomes.
type AOQTSidecarActualCoordinateProbeCoverage struct {
	RequiredComponents         []string                       `json:"required_components"`
	StaticComponentRowCounts   map[string]int                 `json:"static_component_row_counts"`
	SelectedComponentRowCounts map[string]int                 `json:"selected_component_row_counts"`
	SelectedDatasetRowCounts   map[string]int                 `json:"selected_dataset_row_counts"`
	BaselineActivation         AOQTSidecarObjectiveActivation `json:"baseline_activation"`
	// Score-distill coverage is structural candidate-count coverage, not a
	// requirement that the numeric score-distill loss be non-zero.
	ScoreDistillActivationSemantics string `json:"score_distill_activation_semantics"`
	AllTrainOnly                    bool   `json:"all_train_only"`
	Possible                        bool   `json:"possible"`
}

// AOQTSidecarActualCoordinateProbeEndpoint is one scalar, projected endpoint.
// ActualDirection is retained as a full angle-space vector so projection and
// cap behavior are auditable; exactly one requested coordinate is nonzero.
type AOQTSidecarActualCoordinateProbeEndpoint struct {
	Ordinal                   int                            `json:"ordinal"`
	Kind                      string                         `json:"kind"`
	CoordinateIndex           int                            `json:"coordinate_index"`
	AnalyticDirection         int                            `json:"analytic_direction"`
	Indices                   []int                          `json:"indices"`
	Directions                []int                          `json:"directions"`
	GlobalSign                int                            `json:"global_sign"`
	Magnitude                 float32                        `json:"magnitude"`
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
	CandidateActivation       AOQTSidecarObjectiveActivation `json:"candidate_activation"`
	TotalDelta                float32                        `json:"total_delta"`
	TotalDeltaFinite          bool                           `json:"total_delta_finite"`
	// LossDelta aliases total_delta in the explicit loss diagnostic surface;
	// retaining both names keeps the receipt self-describing for downstream
	// audit tooling.
	LossDelta             float32                        `json:"loss_delta"`
	LossDeltaFinite       bool                           `json:"loss_delta_finite"`
	LossDiagnostic        string                         `json:"loss_diagnostic"`
	ComponentDeltas       AOQTSidecarObjectiveComponents `json:"component_deltas"`
	ComponentDeltasFinite bool                           `json:"component_deltas_finite"`
	Q3GainEligible        bool                           `json:"q3_gain_eligible"`
	ProtectedEligible     bool                           `json:"protected_eligible"`
	TransactionEligible   bool                           `json:"transaction_eligible"`
	Reason                string                         `json:"reason"`
	CandidateError        string                         `json:"candidate_error,omitempty"`
}

type AOQTSidecarActualCoordinateProbeReceipt struct {
	Schema                                   string                                     `json:"schema"`
	Required                                 bool                                       `json:"required"`
	Passed                                   bool                                       `json:"passed"`
	FailureReason                            string                                     `json:"failure_reason,omitempty"`
	StratificationVersion                    string                                     `json:"stratification_version"`
	SubsetRowCount                           int                                        `json:"subset_row_count"`
	SubsetRowIDs                             []string                                   `json:"subset_row_ids"`
	SubsetRowIDSHA256                        string                                     `json:"subset_row_ids_sha256"`
	SelectionSeedSHA256                      string                                     `json:"selection_seed_sha256"`
	Coverage                                 AOQTSidecarActualCoordinateProbeCoverage   `json:"coverage"`
	CoverageSHA256                           string                                     `json:"coverage_sha256"`
	CoordinateCount                          int                                        `json:"coordinate_count"`
	TopCoordinateCount                       int                                        `json:"top_coordinate_count"`
	BlockSizes                               []int                                      `json:"block_sizes"`
	Magnitudes                               []float32                                  `json:"magnitudes"`
	GlobalSigns                              []int                                      `json:"global_signs"`
	EndpointCount                            int                                        `json:"endpoint_count"`
	EligibleEndpointCount                    int                                        `json:"eligible_endpoint_count"`
	SelectedEndpointOrdinals                 []int                                      `json:"selected_endpoint_ordinals"`
	RankingSHA256                            string                                     `json:"ranking_sha256"`
	ScheduleSHA256                           string                                     `json:"schedule_sha256"`
	ObjectiveContractSHA256                  string                                     `json:"objective_contract_sha256"`
	EligibilityPolicySHA256                  string                                     `json:"eligibility_policy_sha256"`
	EndpointHashChain                        string                                     `json:"endpoint_hash_chain"`
	EndpointHashChainTailSHA256              string                                     `json:"endpoint_hash_chain_tail_sha256"`
	ValidationReplayEndpointCount            int                                        `json:"validation_replay_endpoint_count"`
	ValidationReplayObjectiveEvaluationCount int                                        `json:"validation_replay_objective_evaluation_count"`
	ValidationReplayCost                     string                                     `json:"validation_replay_cost"`
	Endpoints                                []AOQTSidecarActualCoordinateProbeEndpoint `json:"endpoints"`
	Binding                                  AOQTSidecarActualCoordinateProbeBinding    `json:"binding"`
	QualityClaim                             bool                                       `json:"quality_claim"`
	ReleaseClaim                             bool                                       `json:"release_claim"`
	OfficialClaim                            bool                                       `json:"official_claim"`
	OfficialHeldoutGate                      bool                                       `json:"official_heldout_gate"`
	CommercialClaim                          bool                                       `json:"commercial_claim"`
}

type aoqtActualCoordinateEndpointSpec struct {
	ordinal           int
	coordinate        int
	analyticDirection int
	magnitude         float32
	globalSign        int
}

type aoqtActualCoordinateSearch struct {
	Receipt AOQTSidecarActualCoordinateProbeReceipt
}

func aoqtV7R4ActualCoordinateMagnitudes() []float32 {
	return []float32{0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125}
}

func aoqtV7R4ActualCoordinateGlobalSigns() []int { return []int{1, -1} }

func aoqtV7R4ActualCoordinateBlockSizes() []int { return []int{} }

func sha256AOQTActualCoordinateSelectionSeed(workplanSeed int64, foldID, splitManifestSHA256, manifestSHA256, rowIDSHA256 string) string {
	payload := strings.Join([]string{"aoqt-v7-r4-actual-coordinate-subset-v1", strconv.FormatInt(workplanSeed, 10), foldID, splitManifestSHA256, manifestSHA256, rowIDSHA256}, "\x00")
	sum := sha256.Sum256([]byte(payload))
	return hex.EncodeToString(sum[:])
}

func aoqtActualCoordinatePolicySHA256(policy AOQTSidecarCandidateEligibilityPolicy) (string, error) {
	return aoqtCanonicalSHA256(normalizedAOQTSidecarCandidateEligibilityPolicy(policy))
}

func aoqtActualCoordinateObjectiveContractSHA256(contract AOQTSidecarObjectiveContract) (string, error) {
	return aoqtCanonicalSHA256(contract)
}

func aoqtActualCoordinateScheduleSHA256(magnitudes []float32, signs, blocks []int) (string, error) {
	return aoqtCanonicalSHA256(struct {
		Version       string    `json:"version"`
		TopAngles     int       `json:"top_angles"`
		Magnitudes    []float32 `json:"magnitudes"`
		GlobalSigns   []int     `json:"global_signs"`
		BlockSizes    []int     `json:"block_sizes"`
		EndpointCount int       `json:"endpoint_count"`
	}{"aoqt-v7-r4-actual-coordinate-schedule-v1", AOQTV7R4ActualCoordinateProbeTopAngles, append([]float32(nil), magnitudes...), append([]int(nil), signs...), append([]int(nil), blocks...), AOQTV7R4ActualCoordinateProbeEndpointCount})
}

func aoqtActualCoordinateRankingSHA256(order []aoqtCoordinateSearchRank, protected []string) (string, error) {
	return aoqtCoordinateSearchRankSourceSHA256(order, protected)
}

func aoqtActualCoordinateVectorSHA256(direction []float32) (string, error) {
	return aoqtCanonicalSHA256(direction)
}

func aoqtActualCoordinateCoverageSHA256(coverage AOQTSidecarActualCoordinateProbeCoverage) (string, error) {
	return aoqtCanonicalSHA256(coverage)
}

func (r AOQTSidecarActualCoordinateProbeReceipt) SHA256() (string, error) {
	return aoqtCanonicalSHA256(r)
}

func (r AOQTSidecarActualCoordinateProbeReceipt) Validate() error { return r.validate(false) }

func (r AOQTSidecarActualCoordinateProbeReceipt) ValidateBound() error {
	if err := r.validate(true); err != nil {
		return err
	}
	if err := validateAOQTMetricInputs(r.Binding.Inputs); err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe binding inputs: %w", err)
	}
	if r.Binding.IOReport.Schema != AOQTSidecarCalibrationIOReportSchema || r.Binding.IOReport.ManifestSHA256 != r.Binding.Inputs.DatasetManifestSHA256 || r.Binding.IOReport.RowsSHA256 == "" {
		return fmt.Errorf("AOQT actual-coordinate probe binding io_report provenance does not match inputs")
	}
	if err := validateAOQTSHA256(r.Binding.IOReport.RowsSHA256, "AOQT actual-coordinate probe binding rows sha256"); err != nil {
		return err
	}
	if strings.TrimSpace(r.Binding.IOReport.ManifestPath) == "" || strings.TrimSpace(r.Binding.IOReport.RowsJSONLPath) == "" || strings.TrimSpace(r.Binding.PreflightPath) == "" {
		return fmt.Errorf("AOQT actual-coordinate probe binding source paths are required")
	}
	if err := validateAOQTSHA256(r.Binding.PreflightSHA256, "AOQT actual-coordinate probe binding preflight sha256"); err != nil {
		return err
	}
	if r.Binding.SplitBinding == nil {
		return fmt.Errorf("AOQT actual-coordinate probe split binding is required")
	}
	if err := r.Binding.SplitBinding.Validate(); err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe split binding: %w", err)
	}
	if r.Binding.SplitBinding.MaterializedManifestSHA256 != r.Binding.Inputs.DatasetManifestSHA256 || r.Binding.SplitBinding.MaterializedRowsSHA256 != r.Binding.IOReport.RowsSHA256 || r.Binding.SplitBinding.MaterializedPreflightSHA256 != r.Binding.PreflightSHA256 || r.Binding.SplitBinding.MaterializedRowCount != r.Binding.IOReport.RowCount {
		return fmt.Errorf("AOQT actual-coordinate probe split binding does not match IO inputs")
	}
	for _, item := range []struct {
		name string
		got  string
		want string
	}{{"objective contract", r.Binding.ObjectiveContractSHA256, r.ObjectiveContractSHA256}, {"eligibility policy", r.Binding.EligibilityPolicySHA256, r.EligibilityPolicySHA256}, {"schedule", r.Binding.ScheduleSHA256, r.ScheduleSHA256}, {"ranking", r.Binding.RankingSHA256, r.RankingSHA256}, {"coverage", r.Binding.CoverageSHA256, r.CoverageSHA256}} {
		if item.want == "" {
			if item.got != "" {
				return fmt.Errorf("AOQT actual-coordinate probe binding %s sha256 is present without receipt evidence", item.name)
			}
			continue
		}
		if err := validateAOQTSHA256(item.got, "AOQT actual-coordinate probe binding "+item.name+" sha256"); err != nil {
			return err
		}
		if item.got != item.want {
			return fmt.Errorf("AOQT actual-coordinate probe binding %s sha256 does not match receipt", item.name)
		}
	}
	return rereadAOQTActualCoordinateProbeBinding(r)
}

func (r AOQTSidecarActualCoordinateProbeReceipt) validate(requireBinding bool) error {
	if r.Schema != AOQTSidecarActualCoordinateProbeSchema {
		return fmt.Errorf("AOQT actual-coordinate probe schema %q is unsupported, want %q", r.Schema, AOQTSidecarActualCoordinateProbeSchema)
	}
	if !r.Required {
		return fmt.Errorf("AOQT actual-coordinate probe receipt must be required")
	}
	if r.QualityClaim || r.ReleaseClaim || r.OfficialClaim || r.OfficialHeldoutGate || r.CommercialClaim {
		return fmt.Errorf("AOQT actual-coordinate probe claims and official heldout gate must be false")
	}
	if !r.Passed && len(r.Endpoints) == 0 && (strings.HasPrefix(r.FailureReason, "precondition:") || strings.HasPrefix(r.FailureReason, "coverage:") || strings.HasPrefix(r.FailureReason, "activation:") || strings.HasPrefix(r.FailureReason, "probe_error:")) {
		if requireBinding && r.Binding.SplitBinding == nil {
			return fmt.Errorf("AOQT actual-coordinate probe bound failed receipt requires split binding")
		}
		return nil
	}
	if r.StratificationVersion != "aoqt-v7-r4-train-only-stratified-before-outcomes-v1" {
		return fmt.Errorf("AOQT actual-coordinate probe stratification version is unsupported")
	}
	if len(r.Magnitudes) != AOQTV7R4ActualCoordinateProbeMagnitudeCount || !aoqtFloat32SlicesEqual(r.Magnitudes, aoqtV7R4ActualCoordinateMagnitudes()) {
		return fmt.Errorf("AOQT actual-coordinate probe magnitudes must be the fixed six-value schedule")
	}
	if !slices.Equal(r.GlobalSigns, aoqtV7R4ActualCoordinateGlobalSigns()) || len(r.BlockSizes) != 0 {
		return fmt.Errorf("AOQT actual-coordinate probe requires both signs and no blocks")
	}
	if r.EndpointCount != AOQTV7R4ActualCoordinateProbeEndpointCount || len(r.Endpoints) != r.EndpointCount {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint count = %d/%d, want exactly %d", r.EndpointCount, len(r.Endpoints), AOQTV7R4ActualCoordinateProbeEndpointCount)
	}
	if r.ValidationReplayEndpointCount != AOQTV7R4ValidationReplayEndpointCount || r.ValidationReplayObjectiveEvaluationCount != AOQTV7R4ValidationReplayObjectiveEvaluationCount || r.ValidationReplayCost != AOQTV7R4ValidationReplayCost {
		return fmt.Errorf("AOQT actual-coordinate probe validation replay accounting = endpoints=%d evaluations=%d cost=%q, want endpoints=%d evaluations=%d cost=%q", r.ValidationReplayEndpointCount, r.ValidationReplayObjectiveEvaluationCount, r.ValidationReplayCost, AOQTV7R4ValidationReplayEndpointCount, AOQTV7R4ValidationReplayObjectiveEvaluationCount, AOQTV7R4ValidationReplayCost)
	}
	if r.EligibleEndpointCount < 0 || r.EligibleEndpointCount > r.EndpointCount || len(r.SelectedEndpointOrdinals) != r.EligibleEndpointCount {
		return fmt.Errorf("AOQT actual-coordinate probe eligible endpoint accounting is invalid")
	}
	if err := validateAOQTSHA256(r.SelectionSeedSHA256, "AOQT actual-coordinate probe selection seed sha256"); err != nil {
		return err
	}
	if r.SubsetRowCount != AOQTV7R4ActualCoordinateProbeRowCount || len(r.SubsetRowIDs) != r.SubsetRowCount || !slices.IsSorted(r.SubsetRowIDs) {
		return fmt.Errorf("AOQT actual-coordinate probe subset must contain exactly twelve sorted rows")
	}
	for i, id := range r.SubsetRowIDs {
		if strings.TrimSpace(id) == "" || (i > 0 && id == r.SubsetRowIDs[i-1]) {
			return fmt.Errorf("AOQT actual-coordinate probe subset row ids must be non-empty and unique")
		}
	}
	if r.SubsetRowIDSHA256 != aoqtRowIDSHA256(r.SubsetRowIDs) {
		return fmt.Errorf("AOQT actual-coordinate probe subset row id hash mismatch")
	}
	if r.CoordinateCount <= 0 || r.TopCoordinateCount != AOQTV7R4ActualCoordinateProbeTopAngles || r.CoordinateCount < r.TopCoordinateCount {
		return fmt.Errorf("AOQT actual-coordinate probe coordinate schedule is invalid")
	}
	for _, item := range []struct{ name, value string }{{"schedule_sha256", r.ScheduleSHA256}, {"ranking_sha256", r.RankingSHA256}, {"objective_contract_sha256", r.ObjectiveContractSHA256}, {"eligibility_policy_sha256", r.EligibilityPolicySHA256}, {"coverage_sha256", r.CoverageSHA256}} {
		if err := validateAOQTSHA256(item.value, "AOQT actual-coordinate probe "+item.name); err != nil {
			return err
		}
	}
	wantSchedule, err := aoqtActualCoordinateScheduleSHA256(r.Magnitudes, r.GlobalSigns, r.BlockSizes)
	if err != nil || wantSchedule != r.ScheduleSHA256 {
		return fmt.Errorf("AOQT actual-coordinate probe schedule digest does not bind fixed schedule")
	}
	wantCoverage, err := aoqtActualCoordinateCoverageSHA256(r.Coverage)
	if err != nil || wantCoverage != r.CoverageSHA256 {
		return fmt.Errorf("AOQT actual-coordinate probe coverage digest mismatch")
	}
	if !slices.Equal(r.Coverage.RequiredComponents, aoqtV7R4ActualCoordinateRequiredComponents()) {
		return fmt.Errorf("AOQT actual-coordinate probe coverage required components are not the six canonical components")
	}
	if r.Coverage.ScoreDistillActivationSemantics != AOQTV7R4ScoreDistillActivationSemantics {
		return fmt.Errorf("AOQT actual-coordinate probe coverage score-distill activation semantics = %q, want %q", r.Coverage.ScoreDistillActivationSemantics, AOQTV7R4ScoreDistillActivationSemantics)
	}
	if !r.Coverage.Possible || !r.Coverage.AllTrainOnly || len(r.Coverage.RequiredComponents) == 0 {
		return fmt.Errorf("AOQT actual-coordinate probe coverage is not complete")
	}
	if err := validateAOQTActualCoordinateProbeEndpointHashChain(r); err != nil {
		return err
	}
	selected := make(map[int]struct{}, len(r.SelectedEndpointOrdinals))
	for _, ordinal := range r.SelectedEndpointOrdinals {
		if ordinal <= 0 || ordinal > r.EndpointCount {
			return fmt.Errorf("AOQT actual-coordinate probe selected endpoint ordinal %d is out of range", ordinal)
		}
		if _, ok := selected[ordinal]; ok {
			return fmt.Errorf("AOQT actual-coordinate probe selected endpoint ordinal %d is duplicated", ordinal)
		}
		selected[ordinal] = struct{}{}
	}
	eligible := make([]int, 0)
	for i, endpoint := range r.Endpoints {
		if err := validateAOQTActualCoordinateProbeEndpoint(endpoint, i, r); err != nil {
			return err
		}
		if endpoint.TransactionEligible {
			eligible = append(eligible, endpoint.Ordinal)
		}
	}
	if !slices.Equal(eligible, r.SelectedEndpointOrdinals) {
		return fmt.Errorf("AOQT actual-coordinate probe selected endpoint ordering is not deterministic")
	}
	if r.Passed {
		if strings.TrimSpace(r.FailureReason) != "" {
			return fmt.Errorf("AOQT actual-coordinate probe passed receipt must not have failure_reason")
		}
	} else if strings.TrimSpace(r.FailureReason) == "" {
		return fmt.Errorf("AOQT actual-coordinate probe failed receipt requires failure_reason")
	}
	if requireBinding && r.Binding.SplitBinding == nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound receipt requires split binding")
	}
	return nil
}

func validateAOQTActualCoordinateProbeEndpointHashChain(r AOQTSidecarActualCoordinateProbeReceipt) error {
	var chain []string
	if err := strictUnmarshalAOQT([]byte(r.EndpointHashChain), &chain); err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint hash chain is invalid: %w", err)
	}
	if len(chain) != len(r.Endpoints) || len(chain) == 0 {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint hash chain length = %d, want %d", len(chain), len(r.Endpoints))
	}
	canonical, err := json.Marshal(chain)
	if err != nil || string(canonical) != r.EndpointHashChain {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint hash chain is not canonical JSON")
	}
	previous := ""
	for i, endpoint := range r.Endpoints {
		payload, err := json.Marshal(endpoint)
		if err != nil {
			return err
		}
		want := sha256AOQTCoordinateSearchChainEntry(previous, string(payload))
		if chain[i] != want {
			return fmt.Errorf("AOQT actual-coordinate probe endpoint hash chain[%d] does not bind endpoint/predecessor", i)
		}
		previous = chain[i]
	}
	if r.EndpointHashChainTailSHA256 != previous {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint hash chain tail mismatch")
	}
	return nil
}

func validateAOQTActualCoordinateProbeEndpoint(endpoint AOQTSidecarActualCoordinateProbeEndpoint, ordinal int, receipt AOQTSidecarActualCoordinateProbeReceipt) error {
	if endpoint.Ordinal != ordinal+1 || endpoint.Kind != "coordinate" || endpoint.CoordinateIndex < 0 || endpoint.CoordinateIndex >= receipt.CoordinateCount {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] ordinal/coordinate is invalid", ordinal)
	}
	if len(endpoint.Indices) != 1 || len(endpoint.Directions) != 1 || endpoint.Indices[0] != endpoint.CoordinateIndex {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] must contain exactly one scalar coordinate", ordinal)
	}
	if endpoint.AnalyticDirection != -1 && endpoint.AnalyticDirection != 1 {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] analytic direction must be -1 or 1", ordinal)
	}
	if endpoint.GlobalSign != -1 && endpoint.GlobalSign != 1 {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] global sign must be -1 or 1", ordinal)
	}
	if endpoint.Directions[0] != endpoint.AnalyticDirection*endpoint.GlobalSign {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] effective direction mismatch", ordinal)
	}
	if !isFinite32(endpoint.Magnitude) || endpoint.Magnitude <= 0 || !slices.Contains(receipt.Magnitudes, endpoint.Magnitude) {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] magnitude is outside fixed schedule", ordinal)
	}
	requested := make([]float32, receipt.CoordinateCount)
	requested[endpoint.CoordinateIndex] = endpoint.Magnitude * float32(endpoint.Directions[0])
	requestedSHA, err := aoqtActualCoordinateVectorSHA256(requested)
	if err != nil || requestedSHA != endpoint.RequestedDirectionSHA256 {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] requested direction hash mismatch", ordinal)
	}
	if len(endpoint.ActualDirection) != receipt.CoordinateCount {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] actual direction length = %d, want %d", ordinal, len(endpoint.ActualDirection), receipt.CoordinateCount)
	}
	for i, value := range endpoint.ActualDirection {
		if !isFinite32(value) {
			return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] actual direction[%d] is not finite", ordinal, i)
		}
	}
	actualSHA, err := aoqtActualCoordinateVectorSHA256(endpoint.ActualDirection)
	if err != nil || actualSHA != endpoint.ActualDirectionSHA256 {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] actual direction hash mismatch", ordinal)
	}
	if endpoint.ActualDirectionMaxAbs < 0 || !isFinite32(endpoint.ActualDirectionMaxAbs) || endpoint.ActualDirectionMaxAbs != aoqtMaxAbsFloat32(endpoint.ActualDirection) {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] actual direction max abs mismatch", ordinal)
	}
	if endpoint.BaselineLossFinite && !isFinite32(endpoint.BaselineLoss) {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] baseline loss is non-finite", ordinal)
	}
	if !endpoint.BaselineLossFinite && endpoint.BaselineLoss != 0 {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] unavailable baseline loss must be zero", ordinal)
	}
	if endpoint.CandidateLossFinite && !isFinite32(endpoint.CandidateLoss) {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] candidate loss is non-finite", ordinal)
	}
	if !endpoint.CandidateLossFinite && endpoint.CandidateLoss != 0 && endpoint.CandidateError == "" {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] unavailable candidate loss is inconsistent", ordinal)
	}
	if endpoint.BaselineComponentsFinite && !aoqtFiniteReceiptComponentsValue(endpoint.BaselineComponents) {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] baseline components are non-finite", ordinal)
	}
	if endpoint.CandidateComponentsFinite && !aoqtFiniteReceiptComponentsValue(endpoint.CandidateComponents) {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] candidate components are non-finite", ordinal)
	}
	if endpoint.TotalDeltaFinite && (!endpoint.BaselineLossFinite || !endpoint.CandidateLossFinite || endpoint.TotalDelta != endpoint.CandidateLoss-endpoint.BaselineLoss) {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] total delta mismatch", ordinal)
	}
	if endpoint.LossDeltaFinite != endpoint.TotalDeltaFinite || endpoint.LossDelta != endpoint.TotalDelta {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] loss delta mismatch", ordinal)
	}
	if strings.TrimSpace(endpoint.LossDiagnostic) == "" {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] loss diagnostic is required", ordinal)
	}
	if endpoint.ComponentDeltasFinite && (!endpoint.BaselineComponentsFinite || !endpoint.CandidateComponentsFinite || endpoint.ComponentDeltas != aoqtObjectiveComponentDelta(endpoint.BaselineComponents, endpoint.CandidateComponents)) {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] component delta mismatch", ordinal)
	}
	if endpoint.Q3GainEligible && (!endpoint.BaselineComponentsFinite || !endpoint.CandidateComponentsFinite || !(endpoint.CandidateComponents.Q3Gain < endpoint.BaselineComponents.Q3Gain-aoqtTransactionalQ3ImprovementMinMagnitude)) {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] q3 eligibility is inconsistent", ordinal)
	}
	if endpoint.TransactionEligible && (!endpoint.AngleMoved || !endpoint.CandidateLossFinite || !endpoint.CandidateComponentsFinite || !endpoint.Q3GainEligible || !endpoint.ProtectedEligible) {
		return fmt.Errorf("AOQT actual-coordinate probe endpoint[%d] eligible evidence is incomplete", ordinal)
	}
	return nil
}

func (t *AOQTSidecarTrainer) runActualCoordinateProbe(set AOQTSidecarCalibrationSet, objective AOQTSidecarVectorObjective) (AOQTSidecarActualCoordinateProbeReceipt, error) {
	search, err := t.actualCoordinateSearch(set, objective)
	if search.Receipt.Schema == "" {
		search.Receipt.Schema = AOQTSidecarActualCoordinateProbeSchema
		search.Receipt.Required = true
	}
	if err != nil && search.Receipt.FailureReason == "" {
		prefix := "probe_error: "
		if search.Receipt.SelectionSeedSHA256 == "" {
			prefix = "precondition: "
		}
		search.Receipt.FailureReason = prefix + err.Error()
	}
	return search.Receipt, err
}

func aoqtActualCoordinateComponentNames(contract AOQTSidecarObjectiveContract) []string {
	_ = contract
	// V7-r4 is preregistered against the full protected-v2 objective surface;
	// a contract with one of these components inert must fail closed rather
	// than silently turning the probe into a smaller experiment.
	return aoqtV7R4ActualCoordinateRequiredComponents()
}

func aoqtV7R4ActualCoordinateRequiredComponents() []string {
	return []string{"q3_gain", "q3_order_guard", "q3_score_distill", "q5_order_guard", "q5_score_distill", "nf_boundary_guard"}
}

func aoqtActualCoordinateRowComponents(row AOQTSidecarCalibrationRow, contract AOQTSidecarObjectiveContract) map[string]bool {
	components := make(map[string]bool)
	if row.Weights.Q3Gain > 0 && countAOQTGainEligiblePairs(row.QrelGains, row.EligiblePairMask) > 0 {
		components["q3_gain"] = true
	}
	if row.Weights.Q3OrderGuard > 0 && countAOQTRankOrderedPairs(row.AnchorRanks.Q3, row.EligiblePairMask) > 0 {
		components["q3_order_guard"] = true
	}
	// Score-distill activation is structural coverage: two or more prepared
	// candidates make the score surface evaluable even if the eventual numeric
	// distillation loss is exactly zero.
	if row.Weights.Q3ScoreDistill > 0 && len(row.CandidateDocIDs) >= 2 {
		components["q3_score_distill"] = true
	}
	if row.Weights.Q5OrderGuard > 0 && countAOQTRankOrderedPairs(row.AnchorRanks.Q5, row.EligiblePairMask) > 0 {
		components["q5_order_guard"] = true
	}
	if row.Weights.Q5ScoreDistill > 0 && len(row.CandidateDocIDs) >= 2 {
		components["q5_score_distill"] = true
	}
	if row.Weights.NFBoundaryGuard > 0 && countAOQTNFBoundaryGuardPairs(row.AnchorRanks.Q3, row.CandidateSources, contract.NFBoundarySource, row.EligiblePairMask) > 0 {
		components["nf_boundary_guard"] = true
	}
	return components
}

func aoqtActualCoordinateRowRank(seed, rowID string) string {
	sum := sha256.Sum256([]byte(seed + "\x00" + rowID))
	return hex.EncodeToString(sum[:])
}

func selectAOQTActualCoordinateRows(rows []AOQTSidecarCalibrationRow, contract AOQTSidecarObjectiveContract, selectionSeed string) ([]AOQTSidecarCalibrationRow, AOQTSidecarActualCoordinateProbeCoverage, error) {
	required := aoqtActualCoordinateComponentNames(contract)
	staticCounts := make(map[string]int)
	allTrainOnly := true
	for _, row := range rows {
		if row.SplitProof.Split != "train" || !row.SplitProof.TrainOnly {
			allTrainOnly = false
		}
		for component := range aoqtActualCoordinateRowComponents(row, contract) {
			staticCounts[component]++
		}
	}
	coverage := AOQTSidecarActualCoordinateProbeCoverage{
		RequiredComponents:              append([]string(nil), required...),
		StaticComponentRowCounts:        staticCounts,
		SelectedComponentRowCounts:      make(map[string]int),
		SelectedDatasetRowCounts:        make(map[string]int),
		ScoreDistillActivationSemantics: AOQTV7R4ScoreDistillActivationSemantics,
		AllTrainOnly:                    allTrainOnly,
	}
	if len(rows) < AOQTV7R4ActualCoordinateProbeRowCount {
		return nil, coverage, fmt.Errorf("AOQT actual-coordinate probe requires at least %d train rows, got %d", AOQTV7R4ActualCoordinateProbeRowCount, len(rows))
	}
	if !allTrainOnly {
		return nil, coverage, fmt.Errorf("AOQT actual-coordinate probe requires every selected row to carry train-only split proof")
	}
	for _, component := range required {
		if staticCounts[component] == 0 {
			return nil, coverage, fmt.Errorf("AOQT actual-coordinate probe cannot activate required component %q", component)
		}
	}
	candidates := append([]AOQTSidecarCalibrationRow(nil), rows...)
	sort.SliceStable(candidates, func(i, j int) bool {
		a, b := aoqtActualCoordinateRowRank(selectionSeed, candidates[i].RowID), aoqtActualCoordinateRowRank(selectionSeed, candidates[j].RowID)
		if a != b {
			return a < b
		}
		return candidates[i].RowID < candidates[j].RowID
	})
	selected := make([]AOQTSidecarCalibrationRow, 0, AOQTV7R4ActualCoordinateProbeRowCount)
	used := make(map[string]bool)
	covered := make(map[string]bool)
	for len(selected) < AOQTV7R4ActualCoordinateProbeRowCount {
		best := -1
		bestMissing := -1
		bestRank := ""
		for i, row := range candidates {
			if used[row.RowID] {
				continue
			}
			components := aoqtActualCoordinateRowComponents(row, contract)
			missing := 0
			for _, component := range required {
				if components[component] && !covered[component] {
					missing++
				}
			}
			rank := aoqtActualCoordinateRowRank(selectionSeed, row.RowID)
			if missing > bestMissing || (missing == bestMissing && (best < 0 || rank < bestRank)) {
				best, bestMissing, bestRank = i, missing, rank
			}
		}
		if best < 0 {
			break
		}
		row := candidates[best]
		used[row.RowID] = true
		selected = append(selected, row)
		for component := range aoqtActualCoordinateRowComponents(row, contract) {
			covered[component] = true
		}
	}
	if len(selected) != AOQTV7R4ActualCoordinateProbeRowCount {
		return selected, coverage, fmt.Errorf("AOQT actual-coordinate probe selected %d rows, want %d", len(selected), AOQTV7R4ActualCoordinateProbeRowCount)
	}
	for _, component := range required {
		if !covered[component] {
			return selected, coverage, fmt.Errorf("AOQT actual-coordinate probe selected subset cannot activate required component %q", component)
		}
	}
	sort.Slice(selected, func(i, j int) bool { return selected[i].RowID < selected[j].RowID })
	for _, row := range selected {
		coverage.SelectedDatasetRowCounts[row.Dataset]++
		for component := range aoqtActualCoordinateRowComponents(row, contract) {
			coverage.SelectedComponentRowCounts[component]++
		}
	}
	coverage.Possible = true
	return selected, coverage, nil
}

func (t *AOQTSidecarTrainer) actualCoordinateSearch(set AOQTSidecarCalibrationSet, objective AOQTSidecarVectorObjective) (aoqtActualCoordinateSearch, error) {
	search := aoqtActualCoordinateSearch{Receipt: AOQTSidecarActualCoordinateProbeReceipt{
		Schema:                                   AOQTSidecarActualCoordinateProbeSchema,
		Required:                                 true,
		StratificationVersion:                    "aoqt-v7-r4-train-only-stratified-before-outcomes-v1",
		BlockSizes:                               aoqtV7R4ActualCoordinateBlockSizes(),
		Magnitudes:                               aoqtV7R4ActualCoordinateMagnitudes(),
		GlobalSigns:                              aoqtV7R4ActualCoordinateGlobalSigns(),
		EndpointCount:                            AOQTV7R4ActualCoordinateProbeEndpointCount,
		ValidationReplayEndpointCount:            AOQTV7R4ValidationReplayEndpointCount,
		ValidationReplayObjectiveEvaluationCount: AOQTV7R4ValidationReplayObjectiveEvaluationCount,
		ValidationReplayCost:                     AOQTV7R4ValidationReplayCost,
		QualityClaim:                             false, ReleaseClaim: false, OfficialClaim: false,
		OfficialHeldoutGate: false, CommercialClaim: false,
	}}
	if t == nil {
		return search, fmt.Errorf("AOQT actual-coordinate probe trainer is nil")
	}
	if err := set.Validate(); err != nil {
		return search, err
	}
	if objective == nil {
		return search, fmt.Errorf("AOQT actual-coordinate probe objective is required")
	}
	if t.config.OptimizerMode != AOQTSidecarOptimizerModeV7R4DevActualCoordinate && !isAOQTV7R5ActualCoordinateMode(t.config.OptimizerMode) {
		return search, fmt.Errorf("AOQT actual-coordinate search requires optimizer_mode=%s or the authorized V7-r5 training mode", AOQTSidecarOptimizerModeV7R4DevActualCoordinate)
	}
	if !t.config.ActualCoordinateProbeOnly && !isAOQTV7R5ActualCoordinateMode(t.config.OptimizerMode) {
		return search, fmt.Errorf("AOQT actual-coordinate probe is probe-only until training is authorized")
	}
	if t.config.LearningRate != AOQTV7R4ActualCoordinateProbeLearningRate {
		return search, fmt.Errorf("AOQT actual-coordinate probe requires learning_rate exactly 0.01")
	}
	if err := validateAOQTFitObjectiveContract(set.Manifest.ObjectiveContract, objective); err != nil {
		return search, err
	}
	manifestSHA, err := AOQTSidecarManifestSHA256(set.Manifest)
	if err != nil {
		return search, err
	}
	selectionSeed := sha256AOQTActualCoordinateSelectionSeed(t.config.WorkplanSeed, t.config.ActualCoordinateProbeFoldID, t.config.ActualCoordinateProbeSplitManifestSHA256, manifestSHA, set.Manifest.RowIDSHA256)
	search.Receipt.SelectionSeedSHA256 = selectionSeed
	rows, coverage, err := selectAOQTActualCoordinateRows(set.Rows, set.Manifest.ObjectiveContract, selectionSeed)
	search.Receipt.SubsetRowCount = len(rows)
	search.Receipt.SubsetRowIDs = make([]string, len(rows))
	for i, row := range rows {
		search.Receipt.SubsetRowIDs[i] = row.RowID
	}
	search.Receipt.SubsetRowIDSHA256 = aoqtRowIDSHA256(search.Receipt.SubsetRowIDs)
	search.Receipt.Coverage = coverage
	if err != nil {
		search.Receipt.FailureReason = "coverage: " + err.Error()
		return search, err
	}
	search.Receipt.CoordinateCount = len(t.angles)
	search.Receipt.TopCoordinateCount = AOQTV7R4ActualCoordinateProbeTopAngles
	policySHA, err := aoqtActualCoordinatePolicySHA256(t.eligibilityPolicy)
	if err != nil {
		return search, err
	}
	contractSHA, err := aoqtActualCoordinateObjectiveContractSHA256(set.Manifest.ObjectiveContract)
	if err != nil {
		return search, err
	}
	scheduleSHA, err := aoqtActualCoordinateScheduleSHA256(search.Receipt.Magnitudes, search.Receipt.GlobalSigns, search.Receipt.BlockSizes)
	if err != nil {
		return search, err
	}
	search.Receipt.EligibilityPolicySHA256 = policySHA
	search.Receipt.ObjectiveContractSHA256 = contractSHA
	search.Receipt.ScheduleSHA256 = scheduleSHA
	baselineLoss, _, baselineActivation, baselineComponents, err := t.lossAndAngleGrad(rows, objective)
	if err != nil {
		return search, err
	}
	search.Receipt.Coverage.BaselineActivation = baselineActivation
	search.Receipt.CoverageSHA256, err = aoqtActualCoordinateCoverageSHA256(search.Receipt.Coverage)
	if err != nil {
		return search, err
	}
	if err := validateAOQTActiveObjectiveContributions(set.Manifest.ObjectiveContract.WeightSums, baselineActivation); err != nil {
		search.Receipt.FailureReason = "activation: " + err.Error()
		return search, err
	}
	if !isFinite32(baselineLoss) || !aoqtFiniteReceiptComponentsValue(baselineComponents) {
		return search, fmt.Errorf("AOQT actual-coordinate probe baseline objective is non-finite")
	}
	q3Gradient, err := t.q3GainOnlyAngleGrad(rows, objective)
	if err != nil {
		return search, err
	}
	protectedGradients, err := t.protectedComponentAngleGrads(rows, objective, set.Manifest.ObjectiveContract.WeightSums)
	if err != nil {
		return search, err
	}
	_, fullGradient, _, _, err := t.lossAndAngleGrad(rows, objective)
	if err != nil {
		return search, err
	}
	order, err := rankedAOQTCoordinateSearchAnglesAll(q3Gradient, fullGradient, protectedGradients)
	if err != nil {
		return search, err
	}
	if len(order) < AOQTV7R4ActualCoordinateProbeTopAngles {
		return search, fmt.Errorf("AOQT actual-coordinate probe rank count = %d, want at least %d", len(order), AOQTV7R4ActualCoordinateProbeTopAngles)
	}
	order = append([]aoqtCoordinateSearchRank(nil), order[:AOQTV7R4ActualCoordinateProbeTopAngles]...)
	for i := range order {
		if order[i].primaryDirection == 0 {
			order[i].primaryDirection = -1
		}
	}
	protectedNames := aoqtProtectedGradientNames(protectedGradients)
	search.Receipt.RankingSHA256, err = aoqtActualCoordinateRankingSHA256(order, protectedNames)
	if err != nil {
		return search, err
	}
	baseline := aoqtStepEvaluation{loss: baselineLoss, activation: baselineActivation, components: baselineComponents}
	search.Receipt.Endpoints = make([]AOQTSidecarActualCoordinateProbeEndpoint, 0, AOQTV7R4ActualCoordinateProbeEndpointCount)
	hadEndpointError := false
	ordinal := 0
	for _, rank := range order {
		for _, magnitude := range search.Receipt.Magnitudes {
			for _, globalSign := range search.Receipt.GlobalSigns {
				ordinal++
				endpoint := t.evaluateActualCoordinateEndpoint(rows, objective, baseline, set.Manifest.ObjectiveContract.WeightSums, aoqtActualCoordinateEndpointSpec{ordinal: ordinal, coordinate: rank.index, analyticDirection: int(rank.primaryDirection), magnitude: magnitude, globalSign: globalSign})
				if endpoint.CandidateError != "" {
					hadEndpointError = true
				}
				search.Receipt.Endpoints = append(search.Receipt.Endpoints, endpoint)
			}
		}
	}
	if len(search.Receipt.Endpoints) != AOQTV7R4ActualCoordinateProbeEndpointCount {
		return search, fmt.Errorf("AOQT actual-coordinate probe generated %d endpoints, want exactly %d", len(search.Receipt.Endpoints), AOQTV7R4ActualCoordinateProbeEndpointCount)
	}
	for _, endpoint := range search.Receipt.Endpoints {
		if endpoint.TransactionEligible {
			search.Receipt.SelectedEndpointOrdinals = append(search.Receipt.SelectedEndpointOrdinals, endpoint.Ordinal)
		}
	}
	search.Receipt.EligibleEndpointCount = len(search.Receipt.SelectedEndpointOrdinals)
	chain := make([]string, 0, len(search.Receipt.Endpoints))
	previous := ""
	for _, endpoint := range search.Receipt.Endpoints {
		payload, err := json.Marshal(endpoint)
		if err != nil {
			return search, err
		}
		previous = sha256AOQTCoordinateSearchChainEntry(previous, string(payload))
		chain = append(chain, previous)
	}
	chainData, err := json.Marshal(chain)
	if err != nil {
		return search, err
	}
	search.Receipt.EndpointHashChain = string(chainData)
	search.Receipt.EndpointHashChainTailSHA256 = previous
	if hadEndpointError {
		search.Receipt.FailureReason = "probe_error: nonfinite or invalid actual coordinate endpoint evaluation"
		return search, fmt.Errorf("AOQT V7-r4 actual-coordinate endpoint probe failed closed")
	}
	search.Receipt.Passed = true
	return search, nil
}

func (t *AOQTSidecarTrainer) evaluateActualCoordinateEndpoint(rows []AOQTSidecarCalibrationRow, objective AOQTSidecarVectorObjective, baseline aoqtStepEvaluation, weights AOQTSidecarRowWeights, spec aoqtActualCoordinateEndpointSpec) AOQTSidecarActualCoordinateProbeEndpoint {
	endpoint := AOQTSidecarActualCoordinateProbeEndpoint{
		Ordinal: spec.ordinal, Kind: "coordinate", CoordinateIndex: spec.coordinate, AnalyticDirection: spec.analyticDirection,
		Indices: []int{spec.coordinate}, Directions: []int{spec.analyticDirection * spec.globalSign},
		GlobalSign: spec.globalSign, Magnitude: spec.magnitude,
		BaselineLoss: baseline.loss, BaselineLossFinite: isFinite32(baseline.loss), BaselineComponents: baseline.components,
		BaselineComponentsFinite: aoqtFiniteReceiptComponentsValue(baseline.components), ActualDirection: make([]float32, len(t.angles)),
		LossDiagnostic: "pending",
	}
	requested := make([]float32, len(t.angles))
	requested[spec.coordinate] = spec.magnitude * float32(endpoint.Directions[0])
	endpoint.RequestedDirectionSHA256, _ = aoqtActualCoordinateVectorSHA256(requested)
	state := t.snapshotOptimizerState()
	defer t.restoreOptimizerState(state)
	t.angles[spec.coordinate] += requested[spec.coordinate]
	if err := t.ProjectAngles(); err != nil {
		endpoint.CandidateError = err.Error()
		endpoint.Reason = "probe_error"
		endpoint.LossDiagnostic = endpoint.Reason
		endpoint.LossDelta = endpoint.TotalDelta
		endpoint.LossDeltaFinite = endpoint.TotalDeltaFinite
		endpoint.ActualDirectionSHA256, _ = aoqtActualCoordinateVectorSHA256(endpoint.ActualDirection)
		return endpoint
	}
	for i := range endpoint.ActualDirection {
		endpoint.ActualDirection[i] = t.angles[i] - state.angles[i]
	}
	endpoint.ActualDirectionMaxAbs = aoqtMaxAbsFloat32(endpoint.ActualDirection)
	endpoint.ActualDirectionSHA256, _ = aoqtActualCoordinateVectorSHA256(endpoint.ActualDirection)
	endpoint.AngleMoved = endpoint.ActualDirectionMaxAbs != 0
	candidateLoss, _, activation, components, err := t.lossAndAngleGrad(rows, objective)
	if err != nil {
		endpoint.CandidateError = err.Error()
		endpoint.Reason = "nonfinite_loss"
		endpoint.LossDiagnostic = endpoint.Reason
		endpoint.LossDelta = endpoint.TotalDelta
		endpoint.LossDeltaFinite = endpoint.TotalDeltaFinite
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
	endpoint.ProtectedEligible = endpoint.ComponentDeltasFinite && len(aoqtComponentRegressions(baseline.components, components, t.eligibilityPolicy)) == 0
	decision := aoqtEvaluateTransactionalStepWithPolicy(baseline, aoqtStepEvaluation{loss: candidateLoss, activation: activation, components: components}, weights, t.eligibilityPolicy)
	if !endpoint.AngleMoved {
		endpoint.Reason = string(aoqtRejectionNoAngleMovement)
	} else if decision.accepted {
		endpoint.TransactionEligible = true
		endpoint.Reason = aoqtProposalAcceptedReason
	} else {
		endpoint.Reason = string(decision.reason)
	}
	endpoint.LossDelta = endpoint.TotalDelta
	endpoint.LossDeltaFinite = endpoint.TotalDeltaFinite
	endpoint.LossDiagnostic = endpoint.Reason
	return endpoint
}

func rereadAOQTActualCoordinateProbeBinding(r AOQTSidecarActualCoordinateProbeReceipt) error {
	binding := r.Binding
	manifestData, err := os.ReadFile(binding.IOReport.ManifestPath)
	if err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound manifest reread: %w", err)
	}
	var manifest AOQTSidecarCalibrationManifest
	if err := strictUnmarshalAOQT(manifestData, &manifest); err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound manifest is invalid: %w", err)
	}
	if err := manifest.Validate(); err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound manifest validation: %w", err)
	}
	manifestSHA, err := AOQTSidecarManifestSHA256(manifest)
	if err != nil || manifestSHA != binding.Inputs.DatasetManifestSHA256 || manifestSHA != binding.IOReport.ManifestSHA256 {
		return fmt.Errorf("AOQT actual-coordinate probe bound manifest sha256 mismatch")
	}
	rowsData, err := os.ReadFile(binding.IOReport.RowsJSONLPath)
	if err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound rows reread: %w", err)
	}
	if rowsSHA := sha256BytesAOQT(rowsData); rowsSHA != binding.IOReport.RowsSHA256 {
		return fmt.Errorf("AOQT actual-coordinate probe bound rows sha256 mismatch")
	}
	rows, err := parseAOQTCalibrationRowsJSONL(rowsData, binding.IOReport.RowsJSONLPath)
	if err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound rows are invalid: %w", err)
	}
	if len(rows) != binding.IOReport.RowCount || len(rows) != manifest.RowCount {
		return fmt.Errorf("AOQT actual-coordinate probe bound row count mismatch")
	}
	set := AOQTSidecarCalibrationSet{Manifest: manifest, Rows: rows}
	if err := set.Validate(); err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound calibration set validation: %w", err)
	}
	expectedInputs := AOQTSidecarRunMetricInputs{AnchorArtifactSHA256: manifest.AnchorArtifactSHA256, AnchorPackageManifestSHA256: manifest.AnchorPackageManifestSHA256, AnchorEmbeddingSpaceID: manifest.AnchorEmbeddingSpaceID, DatasetManifestSHA256: manifestSHA, QrelsSHA256ByDataset: cloneAOQTStringMap(manifest.QrelsSHA256ByDataset), CompatibilityDigest: manifest.CompatibilityDigest}
	if !reflect.DeepEqual(binding.Inputs, expectedInputs) {
		return fmt.Errorf("AOQT actual-coordinate probe bound inputs changed from canonical manifest provenance")
	}
	expectedReport := AOQTSidecarCalibrationIOReport{Schema: AOQTSidecarCalibrationIOReportSchema, ManifestPath: binding.IOReport.ManifestPath, RowsJSONLPath: binding.IOReport.RowsJSONLPath, ManifestSHA256: manifestSHA, RowsSHA256: binding.IOReport.RowsSHA256, AnchorArtifactSHA256: manifest.AnchorArtifactSHA256, AnchorPackageManifestSHA256: manifest.AnchorPackageManifestSHA256, AnchorEmbeddingSpaceID: manifest.AnchorEmbeddingSpaceID, CompatibilityDigest: manifest.CompatibilityDigest, TurboQuantSeed: manifest.TurboQuantSeed, Topology: manifest.Topology, ObjectiveContract: manifest.ObjectiveContract, LegalGates: manifest.LegalGates, RowCount: len(rows), CandidateCount: countAOQTCandidates(rows), PairCount: countAOQTPairs(rows), TrainingContract: manifest.TrainingContract, CandidateEligibilityPolicy: cloneAOQTSidecarCandidateEligibilityPolicy(manifest.CandidateEligibilityPolicy)}
	if !reflect.DeepEqual(binding.IOReport, expectedReport) {
		return fmt.Errorf("AOQT actual-coordinate probe bound IO report changed from canonical manifest provenance")
	}
	preflightData, err := os.ReadFile(binding.PreflightPath)
	if err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound preflight reread: %w", err)
	}
	if sha256BytesAOQT(preflightData) != binding.PreflightSHA256 {
		return fmt.Errorf("AOQT actual-coordinate probe bound preflight sha256 mismatch")
	}
	var preflight AOQTSidecarMaterializePreflight
	if err := strictUnmarshalAOQT(preflightData, &preflight); err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound preflight is invalid: %w", err)
	}
	if err := validateAOQTFailClosedPreflight(preflight, binding.PreflightPath, expectedReport, expectedInputs, manifest.Topology, manifest.ObjectiveContract, manifest.LegalGates); err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound preflight validation: %w", err)
	}
	actualBinding, err := validateAOQTV7DevSplitBinding(AOQTSidecarTrainRunnerConfig{SplitManifestPath: binding.SplitBinding.SplitManifestPath, ExpectedSplitManifestSHA256: binding.SplitBinding.SplitManifestFileSHA256, FoldID: binding.SplitBinding.FoldID}, set, expectedReport, preflight, binding.PreflightSHA256)
	if err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound split manifest validation: %w", err)
	}
	if actualBinding == nil || !reflect.DeepEqual(actualBinding, binding.SplitBinding) {
		return fmt.Errorf("AOQT actual-coordinate probe bound split manifest binding changed from canonical split provenance")
	}
	selectionSeed := sha256AOQTActualCoordinateSelectionSeed(binding.IOReport.TurboQuantSeed, binding.SplitBinding.FoldID, binding.SplitBinding.SplitManifestSHA256, manifestSHA, manifest.RowIDSHA256)
	if r.SelectionSeedSHA256 != "" && r.SelectionSeedSHA256 != selectionSeed {
		return fmt.Errorf("AOQT actual-coordinate probe selection seed does not match authoritative bound inputs")
	}
	// A fail-closed receipt may stop before deterministic subset construction
	// (for example, impossible static coverage or inactive baseline). Its
	// provenance is still checked above, but there is no honest subset/hash to
	// reconstruct.
	if !r.Passed && len(r.Endpoints) == 0 {
		return nil
	}
	policy, err := AOQTSidecarTrainingContractPolicy(manifest.TrainingContract, manifest.CandidateEligibilityPolicy)
	if err != nil {
		return err
	}
	wantPolicy, err := aoqtActualCoordinatePolicySHA256(policy)
	if err != nil || wantPolicy != r.EligibilityPolicySHA256 {
		return fmt.Errorf("AOQT actual-coordinate probe bound eligibility policy hash mismatch")
	}
	wantContract, err := aoqtActualCoordinateObjectiveContractSHA256(manifest.ObjectiveContract)
	if err != nil || wantContract != r.ObjectiveContractSHA256 {
		return fmt.Errorf("AOQT actual-coordinate probe bound objective contract hash mismatch")
	}
	wantSchedule, err := aoqtActualCoordinateScheduleSHA256(aoqtV7R4ActualCoordinateMagnitudes(), aoqtV7R4ActualCoordinateGlobalSigns(), aoqtV7R4ActualCoordinateBlockSizes())
	if err != nil || wantSchedule != r.ScheduleSHA256 {
		return fmt.Errorf("AOQT actual-coordinate probe bound schedule hash mismatch")
	}
	if selectionSeed != r.SelectionSeedSHA256 {
		return fmt.Errorf("AOQT actual-coordinate probe bound selection seed changed from canonical provenance")
	}
	selectedRows, coverage, selectErr := selectAOQTActualCoordinateRows(rows, manifest.ObjectiveContract, selectionSeed)
	if selectErr != nil || len(selectedRows) != len(r.SubsetRowIDs) {
		return fmt.Errorf("AOQT actual-coordinate probe bound deterministic subset is unavailable")
	}
	ids := make([]string, len(selectedRows))
	for i, row := range selectedRows {
		ids[i] = row.RowID
	}
	if !slices.Equal(ids, r.SubsetRowIDs) || !slices.Equal(coverage.RequiredComponents, aoqtV7R4ActualCoordinateRequiredComponents()) || coverage.Possible != r.Coverage.Possible || coverage.AllTrainOnly != r.Coverage.AllTrainOnly || coverage.ScoreDistillActivationSemantics != AOQTV7R4ScoreDistillActivationSemantics || !reflect.DeepEqual(coverage.StaticComponentRowCounts, r.Coverage.StaticComponentRowCounts) || !reflect.DeepEqual(coverage.SelectedComponentRowCounts, r.Coverage.SelectedComponentRowCounts) || !reflect.DeepEqual(coverage.SelectedDatasetRowCounts, r.Coverage.SelectedDatasetRowCounts) {
		return fmt.Errorf("AOQT actual-coordinate probe bound deterministic subset/coverage changed from canonical provenance")
	}
	objective, err := objectiveFromAOQTContract(manifest.ObjectiveContract)
	if err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound objective: %w", err)
	}
	trainer, err := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{
		Dim:                                      manifest.Topology.Dim,
		Stages:                                   manifest.Topology.Stages,
		PairingSeed:                              manifest.Topology.Seed,
		WorkplanSeed:                             binding.IOReport.TurboQuantSeed,
		OptimizerMode:                            AOQTSidecarOptimizerModeV7R4DevActualCoordinate,
		ActualCoordinateProbeRequired:            true,
		ActualCoordinateProbeOnly:                true,
		ActualCoordinateProbeFoldID:              binding.SplitBinding.FoldID,
		ActualCoordinateProbeSplitManifestSHA256: binding.SplitBinding.SplitManifestSHA256,
		MaxSteps:                                 0,
		LearningRate:                             AOQTV7R4ActualCoordinateProbeLearningRate,
		AngleCap:                                 AOQTSidecarDefaultAngleCap,
		MaxAngleCap:                              AOQTSidecarHardMaxAngleCap,
		TrainingContract:                         manifest.TrainingContract,
		CandidateEligibilityPolicy:               cloneAOQTSidecarCandidateEligibilityPolicy(manifest.CandidateEligibilityPolicy),
	})
	if err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound trainer config: %w", err)
	}
	if _, err := trainer.Plan(set); err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound topology/config: %w", err)
	}
	if r.CoordinateCount != len(trainer.angles) {
		return fmt.Errorf("AOQT actual-coordinate probe coordinate count = %d, want authoritative count %d", r.CoordinateCount, len(trainer.angles))
	}

	// The receipt contains objective outcomes, not just hashes of outcomes.
	// Recompute the baseline from the authoritative prepared-IP objective before
	// rebuilding the analytic ranking. This is intentionally a second objective
	// evaluation relative to probe generation: bound validation must not trust a
	// self-consistent receipt that has had its loss, components, activation, or
	// eligibility fields edited and rehashed.
	baselineLoss, _, baselineActivation, baselineComponents, err := trainer.lossAndAngleGrad(selectedRows, objective)
	if err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound baseline replay: %w", err)
	}
	if !isFinite32(baselineLoss) || !aoqtFiniteReceiptComponentsValue(baselineComponents) {
		return fmt.Errorf("AOQT actual-coordinate probe bound baseline replay is non-finite")
	}
	if err := validateAOQTActiveObjectiveContributions(manifest.ObjectiveContract.WeightSums, baselineActivation); err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound baseline activation: %w", err)
	}
	coverage.BaselineActivation = baselineActivation
	if !reflect.DeepEqual(coverage, r.Coverage) {
		return fmt.Errorf("AOQT actual-coordinate probe bound coverage, including baseline activation, does not match authoritative replay")
	}
	wantCoverageSHA, err := aoqtActualCoordinateCoverageSHA256(coverage)
	if err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound coverage replay hash: %w", err)
	}
	if wantCoverageSHA != r.CoverageSHA256 || wantCoverageSHA != r.Binding.CoverageSHA256 {
		return fmt.Errorf("AOQT actual-coordinate probe bound coverage hash does not match authoritative replay")
	}

	q3Gradient, err := trainer.q3GainOnlyAngleGrad(selectedRows, objective)
	if err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound q3 ranking gradient: %w", err)
	}
	protectedGradients, err := trainer.protectedComponentAngleGrads(selectedRows, objective, manifest.ObjectiveContract.WeightSums)
	if err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound protected ranking gradient: %w", err)
	}
	_, fullGradient, _, _, err := trainer.lossAndAngleGrad(selectedRows, objective)
	if err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound ranking gradient: %w", err)
	}
	order, err := rankedAOQTCoordinateSearchAnglesAll(q3Gradient, fullGradient, protectedGradients)
	if err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound ranking: %w", err)
	}
	if len(order) < AOQTV7R4ActualCoordinateProbeTopAngles {
		return fmt.Errorf("AOQT actual-coordinate probe bound ranking count = %d, want at least %d", len(order), AOQTV7R4ActualCoordinateProbeTopAngles)
	}
	order = append([]aoqtCoordinateSearchRank(nil), order[:AOQTV7R4ActualCoordinateProbeTopAngles]...)
	for i := range order {
		if order[i].primaryDirection == 0 {
			order[i].primaryDirection = -1
		}
	}
	wantRanking, err := aoqtActualCoordinateRankingSHA256(order, aoqtProtectedGradientNames(protectedGradients))
	if err != nil || wantRanking != r.RankingSHA256 {
		return fmt.Errorf("AOQT actual-coordinate probe ranking hash does not match authoritative bound objective/config")
	}
	if err := validateAOQTActualCoordinateProbeEndpointRanking(r, order); err != nil {
		return err
	}

	// Replay every fixed scalar endpoint through the same projection and
	// prepared-IP evaluator used by the probe. Comparing the complete endpoint
	// structs covers actual projected direction, candidate loss/component sum,
	// activation counts, candidate errors, all eligibility booleans/reasons, and
	// the explicit direction hashes. A caller cannot bypass this by recomputing
	// endpoint self hashes or the endpoint chain after tampering.
	baseline := aoqtStepEvaluation{loss: baselineLoss, activation: baselineActivation, components: baselineComponents}
	replayedEndpoints := make([]AOQTSidecarActualCoordinateProbeEndpoint, 0, AOQTV7R4ActualCoordinateProbeEndpointCount)
	for rank, ranked := range order {
		for _, magnitude := range aoqtV7R4ActualCoordinateMagnitudes() {
			for _, globalSign := range aoqtV7R4ActualCoordinateGlobalSigns() {
				ordinal := len(replayedEndpoints) + 1
				replayed := trainer.evaluateActualCoordinateEndpoint(selectedRows, objective, baseline, manifest.ObjectiveContract.WeightSums, aoqtActualCoordinateEndpointSpec{
					ordinal: ordinal, coordinate: ranked.index, analyticDirection: int(ranked.primaryDirection), magnitude: magnitude, globalSign: globalSign,
				})
				if ordinal > len(r.Endpoints) || !reflect.DeepEqual(replayed, r.Endpoints[ordinal-1]) {
					return fmt.Errorf("AOQT actual-coordinate probe bound endpoint replay mismatch at ordinal %d (rank %d)", ordinal, rank)
				}
				replayedEndpoints = append(replayedEndpoints, replayed)
			}
		}
	}
	if len(replayedEndpoints) != AOQTV7R4ValidationReplayEndpointCount {
		return fmt.Errorf("AOQT actual-coordinate probe bound endpoint replay count = %d, want %d", len(replayedEndpoints), AOQTV7R4ValidationReplayEndpointCount)
	}
	replayedSelected := make([]int, 0, len(replayedEndpoints))
	hadEndpointError := false
	for _, endpoint := range replayedEndpoints {
		if endpoint.CandidateError != "" {
			hadEndpointError = true
		}
		if endpoint.TransactionEligible {
			replayedSelected = append(replayedSelected, endpoint.Ordinal)
		}
	}
	if !slices.Equal(replayedSelected, r.SelectedEndpointOrdinals) || len(replayedSelected) != r.EligibleEndpointCount {
		return fmt.Errorf("AOQT actual-coordinate probe bound eligibility replay does not match selected endpoint accounting")
	}
	if r.Passed == hadEndpointError {
		return fmt.Errorf("AOQT actual-coordinate probe bound passed state does not match replay candidate errors")
	}
	if hadEndpointError {
		const wantFailure = "probe_error: nonfinite or invalid actual coordinate endpoint evaluation"
		if r.FailureReason != wantFailure {
			return fmt.Errorf("AOQT actual-coordinate probe bound failure reason %q does not match replay %q", r.FailureReason, wantFailure)
		}
	} else if r.FailureReason != "" {
		return fmt.Errorf("AOQT actual-coordinate probe bound successful replay has failure reason %q", r.FailureReason)
	}
	replayChain := make([]string, 0, len(replayedEndpoints))
	previous := ""
	for _, endpoint := range replayedEndpoints {
		payload, err := json.Marshal(endpoint)
		if err != nil {
			return fmt.Errorf("AOQT actual-coordinate probe bound endpoint replay encoding: %w", err)
		}
		previous = sha256AOQTCoordinateSearchChainEntry(previous, string(payload))
		replayChain = append(replayChain, previous)
	}
	replayChainData, err := json.Marshal(replayChain)
	if err != nil {
		return fmt.Errorf("AOQT actual-coordinate probe bound endpoint replay chain encoding: %w", err)
	}
	if string(replayChainData) != r.EndpointHashChain || previous != r.EndpointHashChainTailSHA256 {
		return fmt.Errorf("AOQT actual-coordinate probe bound endpoint hash chain/tail does not match authoritative replay")
	}
	return nil
}

func validateAOQTActualCoordinateProbeEndpointRanking(r AOQTSidecarActualCoordinateProbeReceipt, order []aoqtCoordinateSearchRank) error {
	if len(order) != AOQTV7R4ActualCoordinateProbeTopAngles || len(r.Endpoints) != AOQTV7R4ActualCoordinateProbeEndpointCount {
		return fmt.Errorf("AOQT actual-coordinate probe ranking endpoint validation requires the complete fixed schedule")
	}
	const endpointsPerRank = AOQTV7R4ActualCoordinateProbeMagnitudeCount * 2
	for rank, ranked := range order {
		base := rank * endpointsPerRank
		for offset := 0; offset < endpointsPerRank; offset++ {
			endpoint := r.Endpoints[base+offset]
			if endpoint.CoordinateIndex != ranked.index || endpoint.AnalyticDirection != int(ranked.primaryDirection) || endpoint.Indices[0] != ranked.index || endpoint.Directions[0] != int(ranked.primaryDirection)*endpoint.GlobalSign {
				return fmt.Errorf("AOQT actual-coordinate probe endpoint ranking[%d] does not match authoritative analytic ranking", rank)
			}
		}
	}
	return nil
}
