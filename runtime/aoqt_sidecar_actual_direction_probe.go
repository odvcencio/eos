package eosruntime

// V7-r3 actual-direction search is intentionally separate from V7-r2. The
// analytic gradients provide only a deterministic ranking/direction prior;
// projected angle deltas and actual objective evaluations decide eligibility.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"slices"
	"sort"
	"strings"
)

const AOQTSidecarActualDirectionProbeSchema = "eos.aoqt.v7_r3_actual_direction_probe.v1"

type AOQTSidecarActualDirectionProbeBinding struct {
	Inputs                  AOQTSidecarRunMetricInputs     `json:"inputs"`
	IOReport                AOQTSidecarCalibrationIOReport `json:"io_report"`
	PreflightPath           string                         `json:"preflight_path"`
	PreflightSHA256         string                         `json:"preflight_sha256"`
	SplitBinding            *AOQTSidecarDevSplitBinding    `json:"split_binding,omitempty"`
	ObjectiveContractSHA256 string                         `json:"objective_contract_sha256"`
	EligibilityPolicySHA256 string                         `json:"eligibility_policy_sha256"`
	ScheduleSHA256          string                         `json:"schedule_sha256"`
	RankingSHA256           string                         `json:"ranking_sha256"`
}

// AOQTSidecarActualDirectionProbeEndpoint is bounded, row-independent
// evidence for one endpoint. It contains angle deltas only, never row or
// vector payloads.
type AOQTSidecarActualDirectionProbeEndpoint struct {
	Ordinal                   int                            `json:"ordinal"`
	Kind                      string                         `json:"kind"`
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
	ComponentDeltas           AOQTSidecarObjectiveComponents `json:"component_deltas"`
	ComponentDeltasFinite     bool                           `json:"component_deltas_finite"`
	Q3GainEligible            bool                           `json:"q3_gain_eligible"`
	ProtectedEligible         bool                           `json:"protected_eligible"`
	TransactionEligible       bool                           `json:"transaction_eligible"`
	FullTransactionCandidate  bool                           `json:"full_transaction_candidate"`
	FullTransactionOrdinal    int                            `json:"full_transaction_ordinal,omitempty"`
	Reason                    string                         `json:"reason"`
	CandidateError            string                         `json:"candidate_error,omitempty"`
}

type AOQTSidecarActualDirectionProbeReceipt struct {
	Schema                        string                                    `json:"schema"`
	Required                      bool                                      `json:"required"`
	Passed                        bool                                      `json:"passed"`
	FailureReason                 string                                    `json:"failure_reason,omitempty"`
	SubsetRowCount                int                                       `json:"subset_row_count"`
	SubsetRowIDs                  []string                                  `json:"subset_row_ids"`
	SubsetRowIDSHA256             string                                    `json:"subset_row_ids_sha256"`
	SelectionSeedSHA256           string                                    `json:"selection_seed_sha256"`
	CoordinateCount               int                                       `json:"coordinate_count"`
	TopCoordinateCount            int                                       `json:"top_coordinate_count"`
	BlockSizes                    []int                                     `json:"block_sizes"`
	Magnitudes                    []float32                                 `json:"magnitudes"`
	GlobalSigns                   []int                                     `json:"global_signs"`
	EndpointCount                 int                                       `json:"endpoint_count"`
	FullTransactionCandidateLimit int                                       `json:"full_transaction_candidate_limit"`
	FullTransactionCandidateCount int                                       `json:"full_transaction_candidate_count"`
	SelectedEndpointOrdinals      []int                                     `json:"selected_endpoint_ordinals"`
	RankingSHA256                 string                                    `json:"ranking_sha256"`
	ScheduleSHA256                string                                    `json:"schedule_sha256"`
	ObjectiveContractSHA256       string                                    `json:"objective_contract_sha256"`
	EligibilityPolicySHA256       string                                    `json:"eligibility_policy_sha256"`
	EndpointHashChain             string                                    `json:"endpoint_hash_chain"`
	EndpointHashChainTailSHA256   string                                    `json:"endpoint_hash_chain_tail_sha256"`
	Endpoints                     []AOQTSidecarActualDirectionProbeEndpoint `json:"endpoints"`
	Binding                       AOQTSidecarActualDirectionProbeBinding    `json:"binding"`
	QualityClaim                  bool                                      `json:"quality_claim"`
	ReleaseClaim                  bool                                      `json:"release_claim"`
	OfficialClaim                 bool                                      `json:"official_claim"`
	OfficialHeldoutGate           bool                                      `json:"official_heldout_gate"`
	CommercialClaim               bool                                      `json:"commercial_claim"`
}

type aoqtActualDirectionEndpointSpec struct {
	ordinal                  int
	kind                     string
	ranks                    []aoqtCoordinateSearchRank
	magnitude                float32
	globalSign               int
	endpointHashSHA256       string
	requestedDirectionSHA256 string
	actualDirectionSHA256    string
	probeReceiptSHA256       string
}

type aoqtActualDirectionSearch struct {
	Receipt       AOQTSidecarActualDirectionProbeReceipt
	Specs         []aoqtActualDirectionEndpointSpec
	SelectedSpecs []aoqtActualDirectionEndpointSpec
}

func aoqtV7R3ActualDirectionMagnitudes() []float32 {
	return []float32{0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125}
}

func aoqtV7R3ActualDirectionGlobalSigns() []int { return []int{1, -1} }

func aoqtV7R3ActualDirectionBlockSizes() []int { return []int{2, 4, 8} }

func sha256AOQTActualDirectionSelectionSeed(workplanSeed int64, foldID, splitManifestSHA256, manifestSHA256, rowIDSHA256 string) string {
	payload := strings.Join([]string{"aoqt-v7-r3-actual-direction-subset-v1", fmt.Sprintf("%d", workplanSeed), foldID, splitManifestSHA256, manifestSHA256, rowIDSHA256}, "\x00")
	sum := sha256.Sum256([]byte(payload))
	return hex.EncodeToString(sum[:])
}

func aoqtCanonicalSHA256(value any) (string, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func aoqtActualDirectionPolicySHA256(policy AOQTSidecarCandidateEligibilityPolicy) (string, error) {
	return aoqtCanonicalSHA256(normalizedAOQTSidecarCandidateEligibilityPolicy(policy))
}

func aoqtActualDirectionObjectiveContractSHA256(contract AOQTSidecarObjectiveContract) (string, error) {
	return aoqtCanonicalSHA256(contract)
}

func aoqtActualDirectionScheduleSHA256(magnitudes []float32, signs, blocks []int) (string, error) {
	return aoqtCanonicalSHA256(struct {
		Version       string    `json:"version"`
		TopAngles     int       `json:"top_angles"`
		Magnitudes    []float32 `json:"magnitudes"`
		GlobalSigns   []int     `json:"global_signs"`
		BlockSizes    []int     `json:"block_sizes"`
		EndpointCount int       `json:"endpoint_count"`
	}{"aoqt-v7-r3-actual-direction-schedule-v1", AOQTV7R3ActualDirectionProbeTopAngles, append([]float32(nil), magnitudes...), append([]int(nil), signs...), append([]int(nil), blocks...), AOQTV7R3ActualDirectionProbeEndpointCount})
}

func aoqtActualDirectionRankingSHA256(order []aoqtCoordinateSearchRank, protected []string) (string, error) {
	return aoqtCoordinateSearchRankSourceSHA256(order, protected)
}

func aoqtActualDirectionVectorSHA256(direction []float32) (string, error) {
	return aoqtCanonicalSHA256(direction)
}

func (r AOQTSidecarActualDirectionProbeReceipt) SHA256() (string, error) {
	return aoqtCanonicalSHA256(r)
}

func (r AOQTSidecarActualDirectionProbeReceipt) Validate() error { return r.validate(false) }

func (r AOQTSidecarActualDirectionProbeReceipt) ValidateBound() error {
	if err := r.validate(true); err != nil {
		return err
	}
	if err := validateAOQTMetricInputs(r.Binding.Inputs); err != nil {
		return fmt.Errorf("AOQT actual-direction probe binding inputs: %w", err)
	}
	if r.Binding.IOReport.Schema != AOQTSidecarCalibrationIOReportSchema {
		return fmt.Errorf("AOQT actual-direction probe binding io_report schema %q is unsupported", r.Binding.IOReport.Schema)
	}
	if r.Binding.IOReport.ManifestSHA256 != r.Binding.Inputs.DatasetManifestSHA256 || r.Binding.IOReport.RowsSHA256 == "" {
		return fmt.Errorf("AOQT actual-direction probe binding io_report provenance does not match inputs")
	}
	if err := validateAOQTSHA256(r.Binding.IOReport.RowsSHA256, "AOQT actual-direction probe binding rows sha256"); err != nil {
		return err
	}
	if strings.TrimSpace(r.Binding.IOReport.ManifestPath) == "" || strings.TrimSpace(r.Binding.IOReport.RowsJSONLPath) == "" || strings.TrimSpace(r.Binding.PreflightPath) == "" {
		return fmt.Errorf("AOQT actual-direction probe binding source paths are required")
	}
	if err := validateAOQTSHA256(r.Binding.PreflightSHA256, "AOQT actual-direction probe binding preflight sha256"); err != nil {
		return err
	}
	if r.Binding.SplitBinding == nil {
		return fmt.Errorf("AOQT actual-direction probe split binding is required")
	}
	if err := r.Binding.SplitBinding.Validate(); err != nil {
		return fmt.Errorf("AOQT actual-direction probe split binding: %w", err)
	}
	if r.Binding.SplitBinding.MaterializedManifestSHA256 != r.Binding.Inputs.DatasetManifestSHA256 || r.Binding.SplitBinding.MaterializedRowsSHA256 != r.Binding.IOReport.RowsSHA256 || r.Binding.SplitBinding.MaterializedPreflightSHA256 != r.Binding.PreflightSHA256 || r.Binding.SplitBinding.MaterializedRowCount != r.Binding.IOReport.RowCount {
		return fmt.Errorf("AOQT actual-direction probe split binding does not match IO inputs")
	}
	for _, item := range []struct {
		name string
		got  string
		want string
	}{{"objective contract", r.Binding.ObjectiveContractSHA256, r.ObjectiveContractSHA256}, {"eligibility policy", r.Binding.EligibilityPolicySHA256, r.EligibilityPolicySHA256}, {"schedule", r.Binding.ScheduleSHA256, r.ScheduleSHA256}, {"ranking", r.Binding.RankingSHA256, r.RankingSHA256}} {
		if item.want == "" {
			if item.got != "" {
				return fmt.Errorf("AOQT actual-direction probe binding %s sha256 is present without receipt evidence", item.name)
			}
			continue
		}
		if item.want != "" {
			if err := validateAOQTSHA256(item.got, "AOQT actual-direction probe binding "+item.name+" sha256"); err != nil {
				return err
			}
			if item.got != item.want {
				return fmt.Errorf("AOQT actual-direction probe binding %s sha256 does not match receipt", item.name)
			}
		}
	}
	// The binding fields above are only claims. The reread path reconstructs
	// the canonical IO report, preflight, split binding, objective/policy
	// digests, deterministic subset, and analytic ranking from source files and
	// fixed runner configuration. This is deliberately independent of the
	// receipt's self-reported hashes, so changing receipt and binding together
	// cannot forge a valid bound receipt.
	return rereadAOQTActualDirectionProbeBinding(r)
}

func (r AOQTSidecarActualDirectionProbeReceipt) validate(requireBinding bool) error {
	if r.Schema != AOQTSidecarActualDirectionProbeSchema {
		return fmt.Errorf("AOQT actual-direction probe schema %q is unsupported, want %q", r.Schema, AOQTSidecarActualDirectionProbeSchema)
	}
	if !r.Required {
		return fmt.Errorf("AOQT actual-direction probe receipt must be required")
	}
	if r.QualityClaim || r.ReleaseClaim || r.OfficialClaim || r.OfficialHeldoutGate || r.CommercialClaim {
		return fmt.Errorf("AOQT actual-direction probe claims and official heldout gate must be false")
	}
	if !r.Passed && (strings.HasPrefix(r.FailureReason, "precondition:") || (strings.HasPrefix(r.FailureReason, "probe_error:") && len(r.Endpoints) == 0)) {
		// A failed precondition or a probe failure before endpoint construction
		// still gets a provenance-bound, fail-closed receipt. There is no honest
		// fixed schedule or hash chain to validate in that case.
		return nil
	}
	if len(r.Magnitudes) != AOQTV7R3ActualDirectionProbeMagnitudeCount || !aoqtFloat32SlicesEqual(r.Magnitudes, aoqtV7R3ActualDirectionMagnitudes()) {
		return fmt.Errorf("AOQT actual-direction probe magnitudes must be the fixed six-value schedule")
	}
	if !slices.Equal(r.GlobalSigns, aoqtV7R3ActualDirectionGlobalSigns()) {
		return fmt.Errorf("AOQT actual-direction probe global signs = %v, want %v", r.GlobalSigns, aoqtV7R3ActualDirectionGlobalSigns())
	}
	if !slices.Equal(r.BlockSizes, aoqtV7R3ActualDirectionBlockSizes()) {
		return fmt.Errorf("AOQT actual-direction probe block sizes = %v, want %v", r.BlockSizes, aoqtV7R3ActualDirectionBlockSizes())
	}
	if r.EndpointCount != AOQTV7R3ActualDirectionProbeEndpointCount || len(r.Endpoints) != r.EndpointCount {
		return fmt.Errorf("AOQT actual-direction probe endpoint count = %d/%d, want exactly %d", r.EndpointCount, len(r.Endpoints), AOQTV7R3ActualDirectionProbeEndpointCount)
	}
	if r.FullTransactionCandidateLimit != AOQTV7R3ActualDirectionProbeMaxFullCandidates || r.FullTransactionCandidateCount < 0 || r.FullTransactionCandidateCount > r.FullTransactionCandidateLimit || len(r.SelectedEndpointOrdinals) != r.FullTransactionCandidateCount {
		return fmt.Errorf("AOQT actual-direction probe full transaction candidate accounting is invalid")
	}
	if err := validateAOQTSHA256(r.SelectionSeedSHA256, "AOQT actual-direction probe selection seed sha256"); err != nil {
		return err
	}
	if r.SubsetRowCount != AOQTV7R3ActualDirectionProbeRowCount || len(r.SubsetRowIDs) != r.SubsetRowCount || !slices.IsSorted(r.SubsetRowIDs) {
		return fmt.Errorf("AOQT actual-direction probe subset must contain exactly eight sorted rows")
	}
	for i, id := range r.SubsetRowIDs {
		if strings.TrimSpace(id) == "" || (i > 0 && id == r.SubsetRowIDs[i-1]) {
			return fmt.Errorf("AOQT actual-direction probe subset row ids must be non-empty and unique")
		}
	}
	if r.SubsetRowIDSHA256 != aoqtRowIDSHA256(r.SubsetRowIDs) {
		return fmt.Errorf("AOQT actual-direction probe subset row id hash mismatch")
	}
	if r.CoordinateCount <= 0 || r.TopCoordinateCount != AOQTV7R3ActualDirectionProbeTopAngles || r.CoordinateCount < r.TopCoordinateCount {
		return fmt.Errorf("AOQT actual-direction probe coordinate schedule is invalid")
	}
	for _, item := range []struct{ name, value string }{
		{"schedule_sha256", r.ScheduleSHA256}, {"ranking_sha256", r.RankingSHA256},
		{"objective_contract_sha256", r.ObjectiveContractSHA256}, {"eligibility_policy_sha256", r.EligibilityPolicySHA256},
	} {
		if err := validateAOQTSHA256(item.value, "AOQT actual-direction probe "+item.name); err != nil {
			return err
		}
	}
	wantSchedule, err := aoqtActualDirectionScheduleSHA256(r.Magnitudes, r.GlobalSigns, r.BlockSizes)
	if err != nil || wantSchedule != r.ScheduleSHA256 {
		return fmt.Errorf("AOQT actual-direction probe schedule digest does not bind fixed schedule")
	}
	if err := validateAOQTActualDirectionProbeEndpointHashChain(r); err != nil {
		return err
	}
	selected := make(map[int]struct{}, len(r.SelectedEndpointOrdinals))
	for _, ordinal := range r.SelectedEndpointOrdinals {
		if ordinal <= 0 || ordinal > r.EndpointCount {
			return fmt.Errorf("AOQT actual-direction probe selected endpoint ordinal %d is out of range", ordinal)
		}
		if _, ok := selected[ordinal]; ok {
			return fmt.Errorf("AOQT actual-direction probe selected endpoint ordinal %d is duplicated", ordinal)
		}
		selected[ordinal] = struct{}{}
	}
	for i, endpoint := range r.Endpoints {
		if err := validateAOQTActualDirectionEndpoint(endpoint, i, r); err != nil {
			return err
		}
		_, isSelected := selected[endpoint.Ordinal]
		if endpoint.FullTransactionCandidate != isSelected {
			return fmt.Errorf("AOQT actual-direction probe endpoint[%d] selected-candidate flag is inconsistent", i)
		}
		if endpoint.FullTransactionCandidate {
			if endpoint.FullTransactionOrdinal <= 0 || endpoint.FullTransactionOrdinal > r.FullTransactionCandidateCount {
				return fmt.Errorf("AOQT actual-direction probe endpoint[%d] full transaction ordinal is invalid", i)
			}
		} else if endpoint.FullTransactionOrdinal != 0 {
			return fmt.Errorf("AOQT actual-direction probe endpoint[%d] unselected endpoint has full transaction ordinal", i)
		}
	}
	for rank, ordinal := range r.SelectedEndpointOrdinals {
		endpoint := r.Endpoints[ordinal-1]
		if !endpoint.FullTransactionCandidate || endpoint.FullTransactionOrdinal != rank+1 {
			return fmt.Errorf("AOQT actual-direction probe selected endpoint ordinal %d has inconsistent candidate rank", ordinal)
		}
	}
	eligible := make([]int, 0, r.FullTransactionCandidateCount)
	for i, endpoint := range r.Endpoints {
		if endpoint.TransactionEligible {
			if !endpoint.TotalDeltaFinite || !endpoint.ComponentDeltasFinite {
				return fmt.Errorf("AOQT actual-direction probe endpoint[%d] eligible evidence is non-finite", i)
			}
			eligible = append(eligible, i)
		}
	}
	sort.SliceStable(eligible, func(i, j int) bool {
		a, b := r.Endpoints[eligible[i]], r.Endpoints[eligible[j]]
		if a.TotalDelta != b.TotalDelta {
			return a.TotalDelta < b.TotalDelta
		}
		if a.ComponentDeltas.Q3Gain != b.ComponentDeltas.Q3Gain {
			return a.ComponentDeltas.Q3Gain < b.ComponentDeltas.Q3Gain
		}
		return a.Ordinal < b.Ordinal
	})
	if len(eligible) > r.FullTransactionCandidateLimit {
		eligible = eligible[:r.FullTransactionCandidateLimit]
	}
	wantSelected := make([]int, len(eligible))
	for i, index := range eligible {
		wantSelected[i] = r.Endpoints[index].Ordinal
	}
	if !slices.Equal(wantSelected, r.SelectedEndpointOrdinals) {
		return fmt.Errorf("AOQT actual-direction probe selected endpoint ordering is not deterministic")
	}
	if err := validateAOQTActualDirectionProbeEndpointSchedule(r); err != nil {
		return err
	}
	if r.Passed {
		if strings.TrimSpace(r.FailureReason) != "" {
			return fmt.Errorf("AOQT actual-direction probe passed receipt must not have failure_reason")
		}
	} else if strings.TrimSpace(r.FailureReason) == "" {
		return fmt.Errorf("AOQT actual-direction probe failed receipt requires failure_reason")
	}
	if requireBinding && r.Binding.SplitBinding == nil {
		return fmt.Errorf("AOQT actual-direction probe bound receipt requires split binding")
	}
	return nil
}

// validateAOQTActualDirectionProbeEndpointSchedule binds the receipt to the
// fixed endpoint ordering emitted by actualDirectionSearch. The rank digest
// remains the provenance for the analytic ranking itself; this check prevents
// a producer from relabeling a valid hash chain with a different sign,
// magnitude, coordinate, or block schedule.
func validateAOQTActualDirectionProbeEndpointSchedule(r AOQTSidecarActualDirectionProbeReceipt) error {
	if len(r.Endpoints) != AOQTV7R3ActualDirectionProbeEndpointCount {
		return fmt.Errorf("AOQT actual-direction probe schedule validation requires %d endpoints", AOQTV7R3ActualDirectionProbeEndpointCount)
	}
	if len(r.Magnitudes) != AOQTV7R3ActualDirectionProbeMagnitudeCount || len(r.GlobalSigns) != 2 {
		return fmt.Errorf("AOQT actual-direction probe schedule arrays are invalid")
	}
	const endpointsPerRank = AOQTV7R3ActualDirectionProbeMagnitudeCount * 2
	primaryIndices := make([]int, AOQTV7R3ActualDirectionProbeTopAngles)
	primaryDirections := make([]int, AOQTV7R3ActualDirectionProbeTopAngles)
	for rank := 0; rank < AOQTV7R3ActualDirectionProbeTopAngles; rank++ {
		base := rank * endpointsPerRank
		coordinate := r.Endpoints[base]
		if coordinate.Kind != "coordinate" || len(coordinate.Indices) != 1 || len(coordinate.Directions) != 1 {
			return fmt.Errorf("AOQT actual-direction probe coordinate rank %d is not a canonical coordinate group", rank)
		}
		primaryIndices[rank] = coordinate.Indices[0]
		primaryDirections[rank] = coordinate.Directions[0]
		for offset := 0; offset < endpointsPerRank; offset++ {
			endpoint := r.Endpoints[base+offset]
			wantMagnitude := r.Magnitudes[offset/2]
			wantSign := r.GlobalSigns[offset%2]
			if endpoint.Kind != "coordinate" || len(endpoint.Indices) != 1 || endpoint.Indices[0] != primaryIndices[rank] || endpoint.Magnitude != wantMagnitude || endpoint.GlobalSign != wantSign || endpoint.Directions[0] != primaryDirections[rank]*wantSign {
				return fmt.Errorf("AOQT actual-direction probe coordinate schedule mismatch at endpoint %d", endpoint.Ordinal)
			}
		}
	}
	base := AOQTV7R3ActualDirectionProbeTopAngles * endpointsPerRank
	for blockIndex, blockSize := range r.BlockSizes {
		for offset := 0; offset < endpointsPerRank; offset++ {
			endpoint := r.Endpoints[base+blockIndex*endpointsPerRank+offset]
			wantMagnitude := r.Magnitudes[offset/2]
			wantSign := r.GlobalSigns[offset%2]
			if endpoint.Kind != "block" || len(endpoint.Indices) != blockSize || endpoint.Magnitude != wantMagnitude || endpoint.GlobalSign != wantSign {
				return fmt.Errorf("AOQT actual-direction probe block schedule mismatch at endpoint %d", endpoint.Ordinal)
			}
			for i := 0; i < blockSize; i++ {
				if endpoint.Indices[i] != primaryIndices[i] || endpoint.Directions[i] != primaryDirections[i]*wantSign {
					return fmt.Errorf("AOQT actual-direction probe block coordinate schedule mismatch at endpoint %d", endpoint.Ordinal)
				}
			}
		}
	}
	return nil
}

func validateAOQTActualDirectionProbeEndpointHashChain(r AOQTSidecarActualDirectionProbeReceipt) error {
	var chain []string
	if err := strictUnmarshalAOQT([]byte(r.EndpointHashChain), &chain); err != nil {
		return fmt.Errorf("AOQT actual-direction probe endpoint hash chain is invalid: %w", err)
	}
	if len(chain) != len(r.Endpoints) || len(chain) == 0 {
		return fmt.Errorf("AOQT actual-direction probe endpoint hash chain length = %d, want %d", len(chain), len(r.Endpoints))
	}
	canonical, err := json.Marshal(chain)
	if err != nil || string(canonical) != r.EndpointHashChain {
		return fmt.Errorf("AOQT actual-direction probe endpoint hash chain is not canonical JSON")
	}
	previous := ""
	for i, endpoint := range r.Endpoints {
		payload, err := json.Marshal(endpoint)
		if err != nil {
			return err
		}
		want := sha256AOQTCoordinateSearchChainEntry(previous, string(payload))
		if chain[i] != want {
			return fmt.Errorf("AOQT actual-direction probe endpoint hash chain[%d] does not bind endpoint/predecessor", i)
		}
		previous = chain[i]
	}
	if r.EndpointHashChainTailSHA256 != previous {
		return fmt.Errorf("AOQT actual-direction probe endpoint hash chain tail mismatch")
	}
	return nil
}

func validateAOQTActualDirectionEndpoint(endpoint AOQTSidecarActualDirectionProbeEndpoint, ordinal int, receipt AOQTSidecarActualDirectionProbeReceipt) error {
	if endpoint.Ordinal != ordinal+1 {
		return fmt.Errorf("AOQT actual-direction probe endpoint[%d].ordinal = %d, want %d", ordinal, endpoint.Ordinal, ordinal+1)
	}
	if endpoint.Kind != "coordinate" && endpoint.Kind != "block" {
		return fmt.Errorf("AOQT actual-direction probe endpoint[%d] kind %q is unsupported", ordinal, endpoint.Kind)
	}
	if endpoint.Kind == "coordinate" && len(endpoint.Indices) != 1 {
		return fmt.Errorf("AOQT actual-direction probe endpoint[%d] coordinate count = %d, want 1", ordinal, len(endpoint.Indices))
	}
	if endpoint.Kind == "block" && !slices.Contains(receipt.BlockSizes, len(endpoint.Indices)) {
		return fmt.Errorf("AOQT actual-direction probe endpoint[%d] block size is unsupported", ordinal)
	}
	if len(endpoint.Indices) != len(endpoint.Directions) || len(endpoint.Indices) == 0 {
		return fmt.Errorf("AOQT actual-direction probe endpoint[%d] indices/directions mismatch", ordinal)
	}
	if endpoint.GlobalSign != 1 && endpoint.GlobalSign != -1 {
		return fmt.Errorf("AOQT actual-direction probe endpoint[%d] global sign must be -1 or 1", ordinal)
	}
	if !isFinite32(endpoint.Magnitude) || endpoint.Magnitude <= 0 || !slices.Contains(receipt.Magnitudes, endpoint.Magnitude) {
		return fmt.Errorf("AOQT actual-direction probe endpoint[%d] magnitude is outside fixed schedule", ordinal)
	}
	seen := make(map[int]struct{}, len(endpoint.Indices))
	for i, index := range endpoint.Indices {
		if index < 0 || index >= receipt.CoordinateCount || (endpoint.Directions[i] != -1 && endpoint.Directions[i] != 1) {
			return fmt.Errorf("AOQT actual-direction probe endpoint[%d] index/direction is invalid", ordinal)
		}
		if _, ok := seen[index]; ok {
			return fmt.Errorf("AOQT actual-direction probe endpoint[%d] repeats coordinate %d", ordinal, index)
		}
		seen[index] = struct{}{}
	}
	requested := make([]float32, receipt.CoordinateCount)
	for i, index := range endpoint.Indices {
		requested[index] = endpoint.Magnitude * float32(endpoint.Directions[i])
	}
	requestedSHA, err := aoqtActualDirectionVectorSHA256(requested)
	if err != nil || requestedSHA != endpoint.RequestedDirectionSHA256 {
		return fmt.Errorf("AOQT actual-direction probe endpoint[%d] requested direction hash mismatch", ordinal)
	}
	if len(endpoint.ActualDirection) != receipt.CoordinateCount {
		return fmt.Errorf("AOQT actual-direction probe endpoint[%d] actual direction length = %d, want %d", ordinal, len(endpoint.ActualDirection), receipt.CoordinateCount)
	}
	for i, value := range endpoint.ActualDirection {
		if !isFinite32(value) {
			return fmt.Errorf("AOQT actual-direction probe endpoint[%d] actual direction[%d] is not finite", ordinal, i)
		}
	}
	actualSHA, err := aoqtActualDirectionVectorSHA256(endpoint.ActualDirection)
	if err != nil || actualSHA != endpoint.ActualDirectionSHA256 {
		return fmt.Errorf("AOQT actual-direction probe endpoint[%d] actual direction hash mismatch", ordinal)
	}
	if endpoint.ActualDirectionMaxAbs < 0 || !isFinite32(endpoint.ActualDirectionMaxAbs) || endpoint.ActualDirectionMaxAbs != aoqtMaxAbsFloat32(endpoint.ActualDirection) {
		return fmt.Errorf("AOQT actual-direction probe endpoint[%d] actual direction max abs mismatch", ordinal)
	}
	if endpoint.BaselineLossFinite && !isFinite32(endpoint.BaselineLoss) {
		return fmt.Errorf("AOQT actual-direction probe endpoint[%d] baseline loss is non-finite", ordinal)
	}
	if !endpoint.BaselineLossFinite && endpoint.BaselineLoss != 0 {
		return fmt.Errorf("AOQT actual-direction probe endpoint[%d] unavailable baseline loss must be zero", ordinal)
	}
	if endpoint.CandidateLossFinite && !isFinite32(endpoint.CandidateLoss) {
		return fmt.Errorf("AOQT actual-direction probe endpoint[%d] candidate loss is non-finite", ordinal)
	}
	if !endpoint.CandidateLossFinite && endpoint.CandidateLoss != 0 && endpoint.CandidateError == "" {
		return fmt.Errorf("AOQT actual-direction probe endpoint[%d] unavailable candidate loss is inconsistent", ordinal)
	}
	if endpoint.BaselineComponentsFinite && !aoqtFiniteReceiptComponentsValue(endpoint.BaselineComponents) {
		return fmt.Errorf("AOQT actual-direction probe endpoint[%d] baseline components are non-finite", ordinal)
	}
	if endpoint.CandidateComponentsFinite && !aoqtFiniteReceiptComponentsValue(endpoint.CandidateComponents) {
		return fmt.Errorf("AOQT actual-direction probe endpoint[%d] candidate components are non-finite", ordinal)
	}
	if endpoint.TotalDeltaFinite && (!endpoint.BaselineLossFinite || !endpoint.CandidateLossFinite || endpoint.TotalDelta != endpoint.CandidateLoss-endpoint.BaselineLoss) {
		return fmt.Errorf("AOQT actual-direction probe endpoint[%d] total delta mismatch", ordinal)
	}
	if endpoint.ComponentDeltasFinite && (!endpoint.BaselineComponentsFinite || !endpoint.CandidateComponentsFinite || endpoint.ComponentDeltas != aoqtObjectiveComponentDelta(endpoint.BaselineComponents, endpoint.CandidateComponents)) {
		return fmt.Errorf("AOQT actual-direction probe endpoint[%d] component delta mismatch", ordinal)
	}
	if endpoint.TransactionEligible && (!endpoint.AngleMoved || !endpoint.CandidateLossFinite || !endpoint.CandidateComponentsFinite || !endpoint.Q3GainEligible || !endpoint.ProtectedEligible) {
		return fmt.Errorf("AOQT actual-direction probe endpoint[%d] eligible evidence is incomplete", ordinal)
	}
	return nil
}

func (t *AOQTSidecarTrainer) runActualDirectionProbe(set AOQTSidecarCalibrationSet, objective AOQTSidecarVectorObjective) (AOQTSidecarActualDirectionProbeReceipt, error) {
	search, err := t.actualDirectionSearch(set, objective)
	if search.Receipt.Schema == "" {
		search.Receipt.Schema = AOQTSidecarActualDirectionProbeSchema
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

func (t *AOQTSidecarTrainer) actualDirectionSearch(set AOQTSidecarCalibrationSet, objective AOQTSidecarVectorObjective) (aoqtActualDirectionSearch, error) {
	search := aoqtActualDirectionSearch{Receipt: AOQTSidecarActualDirectionProbeReceipt{
		Schema:                        AOQTSidecarActualDirectionProbeSchema,
		Required:                      true,
		BlockSizes:                    aoqtV7R3ActualDirectionBlockSizes(),
		Magnitudes:                    aoqtV7R3ActualDirectionMagnitudes(),
		GlobalSigns:                   aoqtV7R3ActualDirectionGlobalSigns(),
		EndpointCount:                 AOQTV7R3ActualDirectionProbeEndpointCount,
		FullTransactionCandidateLimit: AOQTV7R3ActualDirectionProbeMaxFullCandidates,
		QualityClaim:                  false, ReleaseClaim: false, OfficialClaim: false,
		OfficialHeldoutGate: false, CommercialClaim: false,
	}}
	if t == nil {
		return search, fmt.Errorf("AOQT actual-direction probe trainer is nil")
	}
	if err := set.Validate(); err != nil {
		return search, err
	}
	if objective == nil {
		return search, fmt.Errorf("AOQT actual-direction probe objective is required")
	}
	if t.config.OptimizerMode != AOQTSidecarOptimizerModeV7R3DevActualDirection {
		return search, fmt.Errorf("AOQT actual-direction probe requires optimizer_mode=%s", AOQTSidecarOptimizerModeV7R3DevActualDirection)
	}
	if t.config.LearningRate != AOQTV7R3ActualDirectionProbeLearningRate {
		return search, fmt.Errorf("AOQT actual-direction probe requires learning_rate exactly 0.01")
	}
	if err := validateAOQTFitObjectiveContract(set.Manifest.ObjectiveContract, objective); err != nil {
		return search, err
	}
	if len(set.Rows) < AOQTV7R3ActualDirectionProbeRowCount {
		search.Receipt.FailureReason = fmt.Sprintf("precondition: AOQT actual-direction probe requires at least %d train rows, got %d", AOQTV7R3ActualDirectionProbeRowCount, len(set.Rows))
		return search, fmt.Errorf("AOQT actual-direction probe requires at least %d train rows, got %d", AOQTV7R3ActualDirectionProbeRowCount, len(set.Rows))
	}
	manifestSHA, err := AOQTSidecarManifestSHA256(set.Manifest)
	if err != nil {
		return search, err
	}
	selectionSeed := sha256AOQTActualDirectionSelectionSeed(t.config.WorkplanSeed, t.config.ActualDirectionProbeFoldID, t.config.ActualDirectionProbeSplitManifestSHA256, manifestSHA, set.Manifest.RowIDSHA256)
	search.Receipt.SelectionSeedSHA256 = selectionSeed
	rows := selectAOQTForwardConsistencyRows(set.Rows, selectionSeed)
	search.Receipt.SubsetRowCount = len(rows)
	search.Receipt.SubsetRowIDs = make([]string, len(rows))
	for i, row := range rows {
		search.Receipt.SubsetRowIDs[i] = row.RowID
	}
	search.Receipt.SubsetRowIDSHA256 = aoqtRowIDSHA256(search.Receipt.SubsetRowIDs)
	search.Receipt.CoordinateCount = len(t.angles)
	search.Receipt.TopCoordinateCount = AOQTV7R3ActualDirectionProbeTopAngles
	policySHA, err := aoqtActualDirectionPolicySHA256(t.eligibilityPolicy)
	if err != nil {
		return search, err
	}
	contractSHA, err := aoqtActualDirectionObjectiveContractSHA256(set.Manifest.ObjectiveContract)
	if err != nil {
		return search, err
	}
	scheduleSHA, err := aoqtActualDirectionScheduleSHA256(search.Receipt.Magnitudes, search.Receipt.GlobalSigns, search.Receipt.BlockSizes)
	if err != nil {
		return search, err
	}
	search.Receipt.EligibilityPolicySHA256 = policySHA
	search.Receipt.ObjectiveContractSHA256 = contractSHA
	search.Receipt.ScheduleSHA256 = scheduleSHA
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
	if len(order) < AOQTV7R3ActualDirectionProbeTopAngles {
		return search, fmt.Errorf("AOQT actual-direction probe rank count = %d, want at least %d", len(order), AOQTV7R3ActualDirectionProbeTopAngles)
	}
	order = append([]aoqtCoordinateSearchRank(nil), order[:AOQTV7R3ActualDirectionProbeTopAngles]...)
	// A zero analytic gradient has no sign authority. Keep the endpoint
	// schedule total and deterministic by assigning the fixed negative prior;
	// both global signs are still evaluated and actual eligibility remains the
	// only authority.
	for i := range order {
		if order[i].primaryDirection == 0 {
			order[i].primaryDirection = -1
		}
	}
	protectedNames := aoqtProtectedGradientNames(protectedGradients)
	search.Receipt.RankingSHA256, err = aoqtActualDirectionRankingSHA256(order, protectedNames)
	if err != nil {
		return search, err
	}
	baselineLoss, _, baselineActivation, baselineComponents, err := t.lossAndAngleGrad(rows, objective)
	if err != nil {
		return search, err
	}
	if !isFinite32(baselineLoss) || !aoqtFiniteReceiptComponentsValue(baselineComponents) {
		return search, fmt.Errorf("AOQT actual-direction probe baseline objective is non-finite")
	}
	baseline := aoqtStepEvaluation{loss: baselineLoss, activation: baselineActivation, components: baselineComponents}
	state := t.snapshotOptimizerState()
	defer t.restoreOptimizerState(state)
	appendSpecs := func(specs *[]aoqtActualDirectionEndpointSpec, kind string, ranks []aoqtCoordinateSearchRank) {
		for _, magnitude := range search.Receipt.Magnitudes {
			for _, globalSign := range search.Receipt.GlobalSigns {
				*specs = append(*specs, aoqtActualDirectionEndpointSpec{kind: kind, ranks: append([]aoqtCoordinateSearchRank(nil), ranks...), magnitude: magnitude, globalSign: globalSign})
			}
		}
	}
	specs := make([]aoqtActualDirectionEndpointSpec, 0, AOQTV7R3ActualDirectionProbeEndpointCount)
	for _, rank := range order {
		appendSpecs(&specs, "coordinate", []aoqtCoordinateSearchRank{rank})
	}
	for _, blockSize := range search.Receipt.BlockSizes {
		appendSpecs(&specs, "block", order[:blockSize])
	}
	if len(specs) != AOQTV7R3ActualDirectionProbeEndpointCount {
		return search, fmt.Errorf("AOQT actual-direction probe generated %d endpoints, want exactly %d", len(specs), AOQTV7R3ActualDirectionProbeEndpointCount)
	}
	search.Specs = specs
	search.Receipt.Endpoints = make([]AOQTSidecarActualDirectionProbeEndpoint, 0, len(specs))
	hadEndpointError := false
	for ordinal := range specs {
		specs[ordinal].ordinal = ordinal + 1
		endpoint := t.evaluateActualDirectionEndpoint(rows, objective, baseline, set.Manifest.ObjectiveContract.WeightSums, *&specs[ordinal])
		if endpoint.CandidateError != "" {
			hadEndpointError = true
		}
		search.Receipt.Endpoints = append(search.Receipt.Endpoints, endpoint)
	}
	selected := make([]int, 0, len(search.Receipt.Endpoints))
	for i, endpoint := range search.Receipt.Endpoints {
		if endpoint.TransactionEligible {
			selected = append(selected, i)
		}
	}
	sort.SliceStable(selected, func(i, j int) bool {
		a, b := search.Receipt.Endpoints[selected[i]], search.Receipt.Endpoints[selected[j]]
		if a.TotalDelta != b.TotalDelta {
			return a.TotalDelta < b.TotalDelta
		}
		if a.ComponentDeltas.Q3Gain != b.ComponentDeltas.Q3Gain {
			return a.ComponentDeltas.Q3Gain < b.ComponentDeltas.Q3Gain
		}
		return a.Ordinal < b.Ordinal
	})
	if len(selected) > AOQTV7R3ActualDirectionProbeMaxFullCandidates {
		selected = selected[:AOQTV7R3ActualDirectionProbeMaxFullCandidates]
	}
	for rank, index := range selected {
		search.Receipt.Endpoints[index].FullTransactionCandidate = true
		search.Receipt.Endpoints[index].FullTransactionOrdinal = rank + 1
		search.Receipt.SelectedEndpointOrdinals = append(search.Receipt.SelectedEndpointOrdinals, search.Receipt.Endpoints[index].Ordinal)
		search.SelectedSpecs = append(search.SelectedSpecs, specs[index])
	}
	search.Receipt.FullTransactionCandidateCount = len(selected)
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
		search.Receipt.FailureReason = "nonfinite or invalid actual endpoint evaluation"
		return search, fmt.Errorf("AOQT V7-r3 actual-direction endpoint probe failed closed")
	}
	search.Receipt.Passed = true
	var probeHashChain []string
	if err := strictUnmarshalAOQT([]byte(search.Receipt.EndpointHashChain), &probeHashChain); err != nil {
		return search, fmt.Errorf("AOQT actual-direction probe endpoint hash chain internal decode: %w", err)
	}
	probeReceiptSHA256, err := search.Receipt.SHA256()
	if err != nil {
		return search, fmt.Errorf("AOQT actual-direction probe receipt hash: %w", err)
	}
	for i := range search.SelectedSpecs {
		endpointOrdinal := search.SelectedSpecs[i].ordinal
		if endpointOrdinal <= 0 || endpointOrdinal > len(search.Receipt.Endpoints) {
			return search, fmt.Errorf("AOQT actual-direction probe selected endpoint ordinal %d is out of range", endpointOrdinal)
		}
		endpoint := search.Receipt.Endpoints[endpointOrdinal-1]
		search.SelectedSpecs[i].endpointHashSHA256 = probeHashChain[endpointOrdinal-1]
		search.SelectedSpecs[i].requestedDirectionSHA256 = endpoint.RequestedDirectionSHA256
		search.SelectedSpecs[i].actualDirectionSHA256 = endpoint.ActualDirectionSHA256
		search.SelectedSpecs[i].probeReceiptSHA256 = probeReceiptSHA256
	}
	return search, nil
}

func (t *AOQTSidecarTrainer) evaluateActualDirectionEndpoint(rows []AOQTSidecarCalibrationRow, objective AOQTSidecarVectorObjective, baseline aoqtStepEvaluation, weights AOQTSidecarRowWeights, spec aoqtActualDirectionEndpointSpec) AOQTSidecarActualDirectionProbeEndpoint {
	endpoint := AOQTSidecarActualDirectionProbeEndpoint{
		Ordinal: spec.ordinal, Kind: spec.kind, GlobalSign: spec.globalSign, Magnitude: spec.magnitude,
		BaselineLoss: baseline.loss, BaselineLossFinite: isFinite32(baseline.loss), BaselineComponents: baseline.components,
		BaselineComponentsFinite: aoqtFiniteReceiptComponentsValue(baseline.components), ActualDirection: make([]float32, len(t.angles)),
	}
	for _, rank := range spec.ranks {
		endpoint.Indices = append(endpoint.Indices, rank.index)
		endpoint.Directions = append(endpoint.Directions, int(rank.primaryDirection)*spec.globalSign)
		endpoint.ActualDirection[rank.index] = rank.primaryDirection * spec.magnitude * float32(spec.globalSign)
	}
	requested := append([]float32(nil), endpoint.ActualDirection...)
	endpoint.RequestedDirectionSHA256, _ = aoqtActualDirectionVectorSHA256(requested)
	state := t.snapshotOptimizerState()
	defer t.restoreOptimizerState(state)
	for i, delta := range endpoint.ActualDirection {
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

func aoqtDifference(after, before []float32) []float32 {
	if len(after) != len(before) {
		return nil
	}
	delta := make([]float32, len(after))
	for i := range after {
		delta[i] = after[i] - before[i]
	}
	return delta
}

func rereadAOQTActualDirectionProbeBinding(r AOQTSidecarActualDirectionProbeReceipt) error {
	binding := r.Binding
	manifestData, err := os.ReadFile(binding.IOReport.ManifestPath)
	if err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound manifest reread: %w", err)
	}
	var manifest AOQTSidecarCalibrationManifest
	if err := strictUnmarshalAOQT(manifestData, &manifest); err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound manifest is invalid: %w", err)
	}
	if err := manifest.Validate(); err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound manifest validation: %w", err)
	}
	manifestSHA, err := AOQTSidecarManifestSHA256(manifest)
	if err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound manifest hash: %w", err)
	}
	if manifestSHA != binding.Inputs.DatasetManifestSHA256 || manifestSHA != binding.IOReport.ManifestSHA256 {
		return fmt.Errorf("AOQT actual-direction probe bound manifest sha256 mismatch")
	}

	rowsData, err := os.ReadFile(binding.IOReport.RowsJSONLPath)
	if err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound rows reread: %w", err)
	}
	if rowsSHA := sha256BytesAOQT(rowsData); rowsSHA != binding.IOReport.RowsSHA256 {
		return fmt.Errorf("AOQT actual-direction probe bound rows sha256 mismatch")
	}
	rows, err := parseAOQTCalibrationRowsJSONL(rowsData, binding.IOReport.RowsJSONLPath)
	if err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound rows are invalid: %w", err)
	}
	if len(rows) != binding.IOReport.RowCount || len(rows) != manifest.RowCount {
		return fmt.Errorf("AOQT actual-direction probe bound row count mismatch")
	}
	set := AOQTSidecarCalibrationSet{Manifest: manifest, Rows: rows}
	if err := set.Validate(); err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound calibration set validation: %w", err)
	}

	expectedInputs := AOQTSidecarRunMetricInputs{
		AnchorArtifactSHA256:        manifest.AnchorArtifactSHA256,
		AnchorPackageManifestSHA256: manifest.AnchorPackageManifestSHA256,
		AnchorEmbeddingSpaceID:      manifest.AnchorEmbeddingSpaceID,
		DatasetManifestSHA256:       manifestSHA,
		QrelsSHA256ByDataset:        cloneAOQTStringMap(manifest.QrelsSHA256ByDataset),
		CompatibilityDigest:         manifest.CompatibilityDigest,
	}
	if !reflect.DeepEqual(binding.Inputs, expectedInputs) {
		return fmt.Errorf("AOQT actual-direction probe bound inputs changed from canonical manifest provenance")
	}
	expectedReport := AOQTSidecarCalibrationIOReport{
		Schema:                      AOQTSidecarCalibrationIOReportSchema,
		ManifestPath:                binding.IOReport.ManifestPath,
		RowsJSONLPath:               binding.IOReport.RowsJSONLPath,
		ManifestSHA256:              manifestSHA,
		RowsSHA256:                  binding.IOReport.RowsSHA256,
		AnchorArtifactSHA256:        manifest.AnchorArtifactSHA256,
		AnchorPackageManifestSHA256: manifest.AnchorPackageManifestSHA256,
		AnchorEmbeddingSpaceID:      manifest.AnchorEmbeddingSpaceID,
		CompatibilityDigest:         manifest.CompatibilityDigest,
		TurboQuantSeed:              manifest.TurboQuantSeed,
		Topology:                    manifest.Topology,
		ObjectiveContract:           manifest.ObjectiveContract,
		LegalGates:                  manifest.LegalGates,
		RowCount:                    len(rows),
		CandidateCount:              countAOQTCandidates(rows),
		PairCount:                   countAOQTPairs(rows),
		TrainingContract:            manifest.TrainingContract,
		CandidateEligibilityPolicy:  cloneAOQTSidecarCandidateEligibilityPolicy(manifest.CandidateEligibilityPolicy),
	}
	if !reflect.DeepEqual(binding.IOReport, expectedReport) {
		return fmt.Errorf("AOQT actual-direction probe bound IO report changed from canonical manifest provenance")
	}

	preflightData, err := os.ReadFile(binding.PreflightPath)
	if err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound preflight reread: %w", err)
	}
	if sha256BytesAOQT(preflightData) != binding.PreflightSHA256 {
		return fmt.Errorf("AOQT actual-direction probe bound preflight sha256 mismatch")
	}
	var preflight AOQTSidecarMaterializePreflight
	if err := strictUnmarshalAOQT(preflightData, &preflight); err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound preflight is invalid: %w", err)
	}
	if err := validateAOQTFailClosedPreflight(preflight, binding.PreflightPath, expectedReport, expectedInputs, manifest.Topology, manifest.ObjectiveContract, manifest.LegalGates); err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound preflight validation: %w", err)
	}

	actualBinding, err := validateAOQTV7DevSplitBinding(AOQTSidecarTrainRunnerConfig{
		SplitManifestPath:           binding.SplitBinding.SplitManifestPath,
		ExpectedSplitManifestSHA256: binding.SplitBinding.SplitManifestFileSHA256,
		FoldID:                      binding.SplitBinding.FoldID,
	}, set, expectedReport, preflight, binding.PreflightSHA256)
	if err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound split manifest validation: %w", err)
	}
	if actualBinding == nil || !reflect.DeepEqual(actualBinding, binding.SplitBinding) {
		return fmt.Errorf("AOQT actual-direction probe bound split manifest binding changed from canonical split provenance")
	}

	policy, err := AOQTSidecarTrainingContractPolicy(manifest.TrainingContract, manifest.CandidateEligibilityPolicy)
	if err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound eligibility policy: %w", err)
	}
	wantPolicySHA, err := aoqtActualDirectionPolicySHA256(policy)
	if err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound eligibility policy hash: %w", err)
	}
	wantContractSHA, err := aoqtActualDirectionObjectiveContractSHA256(manifest.ObjectiveContract)
	if err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound objective contract hash: %w", err)
	}
	wantScheduleSHA, err := aoqtActualDirectionScheduleSHA256(aoqtV7R3ActualDirectionMagnitudes(), aoqtV7R3ActualDirectionGlobalSigns(), aoqtV7R3ActualDirectionBlockSizes())
	if err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound schedule hash: %w", err)
	}
	for _, item := range []struct {
		name string
		got  string
		want string
	}{
		{"objective contract", r.ObjectiveContractSHA256, wantContractSHA},
		{"eligibility policy", r.EligibilityPolicySHA256, wantPolicySHA},
		{"schedule", r.ScheduleSHA256, wantScheduleSHA},
	} {
		if item.got == "" {
			if !actualDirectionProbeReceiptHasEarlyFailure(r) {
				return fmt.Errorf("AOQT actual-direction probe bound %s hash is missing", item.name)
			}
			continue
		}
		if item.got != item.want {
			return fmt.Errorf("AOQT actual-direction probe bound %s hash does not match authoritative input/config", item.name)
		}
	}

	selectionSeed := sha256AOQTActualDirectionSelectionSeed(
		binding.IOReport.TurboQuantSeed,
		binding.SplitBinding.FoldID,
		binding.SplitBinding.SplitManifestSHA256,
		manifestSHA,
		manifest.RowIDSHA256,
	)
	if r.SelectionSeedSHA256 != "" && r.SelectionSeedSHA256 != selectionSeed {
		return fmt.Errorf("AOQT actual-direction probe selection seed does not match authoritative bound inputs")
	}
	if r.SelectionSeedSHA256 != "" {
		expectedRows := selectAOQTForwardConsistencyRows(rows, selectionSeed)
		expectedIDs := make([]string, len(expectedRows))
		for i, row := range expectedRows {
			expectedIDs[i] = row.RowID
		}
		if !slices.Equal(r.SubsetRowIDs, expectedIDs) {
			return fmt.Errorf("AOQT actual-direction probe selected row ids do not match authoritative deterministic subset")
		}
	}
	if actualDirectionProbeReceiptHasEarlyFailure(r) && len(r.Endpoints) == 0 {
		return nil
	}

	objective, err := objectiveFromAOQTContract(manifest.ObjectiveContract)
	if err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound objective: %w", err)
	}
	trainer, err := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{
		Dim:                                     manifest.Topology.Dim,
		Stages:                                  manifest.Topology.Stages,
		PairingSeed:                             manifest.Topology.Seed,
		WorkplanSeed:                            binding.IOReport.TurboQuantSeed,
		OptimizerMode:                           AOQTSidecarOptimizerModeV7R3DevActualDirection,
		ActualDirectionProbeRequired:            true,
		ActualDirectionProbeFoldID:              binding.SplitBinding.FoldID,
		ActualDirectionProbeSplitManifestSHA256: binding.SplitBinding.SplitManifestSHA256,
		MaxSteps:                                1,
		LearningRate:                            AOQTV7R3ActualDirectionProbeLearningRate,
		AngleCap:                                AOQTSidecarDefaultAngleCap,
		MaxAngleCap:                             AOQTSidecarHardMaxAngleCap,
		TrainingContract:                        manifest.TrainingContract,
		CandidateEligibilityPolicy:              cloneAOQTSidecarCandidateEligibilityPolicy(manifest.CandidateEligibilityPolicy),
	})
	if err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound trainer config: %w", err)
	}
	if _, err := trainer.Plan(set); err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound topology/config: %w", err)
	}
	probeRows := selectAOQTForwardConsistencyRows(rows, selectionSeed)
	q3Gradient, err := trainer.q3GainOnlyAngleGrad(probeRows, objective)
	if err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound q3 ranking gradient: %w", err)
	}
	protectedGradients, err := trainer.protectedComponentAngleGrads(probeRows, objective, manifest.ObjectiveContract.WeightSums)
	if err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound protected ranking gradient: %w", err)
	}
	_, fullGradient, _, _, err := trainer.lossAndAngleGrad(probeRows, objective)
	if err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound ranking gradient: %w", err)
	}
	order, err := rankedAOQTCoordinateSearchAnglesAll(q3Gradient, fullGradient, protectedGradients)
	if err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound ranking: %w", err)
	}
	if len(order) < AOQTV7R3ActualDirectionProbeTopAngles {
		return fmt.Errorf("AOQT actual-direction probe bound ranking count = %d, want at least %d", len(order), AOQTV7R3ActualDirectionProbeTopAngles)
	}
	order = append([]aoqtCoordinateSearchRank(nil), order[:AOQTV7R3ActualDirectionProbeTopAngles]...)
	for i := range order {
		if order[i].primaryDirection == 0 {
			order[i].primaryDirection = -1
		}
	}
	protectedNames := aoqtProtectedGradientNames(protectedGradients)
	wantRankingSHA, err := aoqtActualDirectionRankingSHA256(order, protectedNames)
	if err != nil {
		return fmt.Errorf("AOQT actual-direction probe bound ranking hash: %w", err)
	}
	if r.RankingSHA256 != wantRankingSHA {
		return fmt.Errorf("AOQT actual-direction probe ranking hash does not match authoritative bound objective/config")
	}
	if r.CoordinateCount != len(trainer.angles) {
		return fmt.Errorf("AOQT actual-direction probe coordinate count = %d, want authoritative count %d", r.CoordinateCount, len(trainer.angles))
	}
	if err := validateAOQTActualDirectionProbeEndpointRanking(r, order); err != nil {
		return err
	}
	return nil
}

func actualDirectionProbeReceiptHasEarlyFailure(r AOQTSidecarActualDirectionProbeReceipt) bool {
	return !r.Passed && (strings.HasPrefix(r.FailureReason, "precondition:") || (strings.HasPrefix(r.FailureReason, "probe_error:") && len(r.Endpoints) == 0))
}

func validateAOQTActualDirectionProbeEndpointRanking(r AOQTSidecarActualDirectionProbeReceipt, order []aoqtCoordinateSearchRank) error {
	if len(order) != AOQTV7R3ActualDirectionProbeTopAngles || len(r.Endpoints) != AOQTV7R3ActualDirectionProbeEndpointCount {
		return fmt.Errorf("AOQT actual-direction probe ranking endpoint validation requires the complete fixed schedule")
	}
	const endpointsPerRank = AOQTV7R3ActualDirectionProbeMagnitudeCount * 2
	for rank, ranked := range order {
		endpoint := r.Endpoints[rank*endpointsPerRank]
		if len(endpoint.Indices) != 1 || endpoint.Indices[0] != ranked.index || endpoint.Directions[0] != int(ranked.primaryDirection) {
			return fmt.Errorf("AOQT actual-direction probe endpoint ranking[%d] does not match authoritative analytic ranking", rank)
		}
	}
	base := AOQTV7R3ActualDirectionProbeTopAngles * endpointsPerRank
	for blockIndex, blockSize := range r.BlockSizes {
		for offset := 0; offset < endpointsPerRank; offset++ {
			endpoint := r.Endpoints[base+blockIndex*endpointsPerRank+offset]
			for i := 0; i < blockSize; i++ {
				if endpoint.Indices[i] != order[i].index || endpoint.Directions[i] != int(order[i].primaryDirection)*endpoint.GlobalSign {
					return fmt.Errorf("AOQT actual-direction probe block ranking[%d] does not match authoritative analytic ranking", blockIndex)
				}
			}
		}
	}
	return nil
}

// validateAOQTActualDirectionProposalProvenance binds every accepted r3 full
// proposal back to the exact endpoint that authorized it. The endpoint hash
// chain and direction hashes are redundant by design: the chain commits the
// complete endpoint, while the explicit direction hashes make state-alignment
// drift visible at the proposal receipt itself.
func validateAOQTActualDirectionProposalProvenance(probe AOQTSidecarActualDirectionProbeReceipt, probeSHA string, diagnostics AOQTSidecarOptimizerDiagnostics, label string) error {
	if diagnostics.OptimizerMode != AOQTSidecarOptimizerModeV7R3DevActualDirection || diagnostics.ProposalReceipts == nil {
		return nil
	}
	if !probe.Passed && len(probe.Endpoints) == 0 {
		return nil
	}
	if strings.TrimSpace(probeSHA) == "" {
		return fmt.Errorf("%s actual-direction probe receipt hash is required for proposal provenance", label)
	}
	if err := validateAOQTSHA256(probeSHA, label+" actual-direction probe receipt hash"); err != nil {
		return err
	}
	var chain []string
	if err := strictUnmarshalAOQT([]byte(probe.EndpointHashChain), &chain); err != nil {
		return fmt.Errorf("%s actual-direction endpoint hash chain: %w", label, err)
	}
	if len(chain) != len(probe.Endpoints) {
		return fmt.Errorf("%s actual-direction endpoint hash chain length = %d, want %d", label, len(chain), len(probe.Endpoints))
	}
	for index, receipt := range *diagnostics.ProposalReceipts {
		if receipt.Kind != aoqtProposalKindCoordinate {
			if receipt.EndpointHashSHA256 != "" || receipt.RequestedDirectionSHA256 != "" || receipt.ActualDirectionSHA256 != "" || receipt.ActualDirectionProbeSHA256 != "" {
				return fmt.Errorf("%s proposal_receipts[%d] non-coordinate proposal carries actual-direction endpoint provenance", label, index)
			}
			continue
		}
		if receipt.EndpointOrdinal <= 0 || receipt.EndpointOrdinal > len(probe.Endpoints) {
			return fmt.Errorf("%s proposal_receipts[%d] endpoint ordinal %d is outside probe", label, index, receipt.EndpointOrdinal)
		}
		populated := receipt.EndpointHashSHA256 != "" || receipt.RequestedDirectionSHA256 != "" || receipt.ActualDirectionSHA256 != "" || receipt.ActualDirectionProbeSHA256 != ""
		if !receipt.Accepted && !populated {
			continue
		}
		if !populated {
			return fmt.Errorf("%s proposal_receipts[%d] accepted coordinate proposal is missing endpoint provenance", label, index)
		}
		for name, value := range map[string]string{
			"endpoint_hash_sha256":          receipt.EndpointHashSHA256,
			"requested_direction_sha256":    receipt.RequestedDirectionSHA256,
			"actual_direction_sha256":       receipt.ActualDirectionSHA256,
			"actual_direction_probe_sha256": receipt.ActualDirectionProbeSHA256,
		} {
			if err := validateAOQTSHA256(value, fmt.Sprintf("%s proposal_receipts[%d] %s", label, index, name)); err != nil {
				return err
			}
		}
		endpoint := probe.Endpoints[receipt.EndpointOrdinal-1]
		if !endpoint.FullTransactionCandidate || !endpoint.TransactionEligible {
			return fmt.Errorf("%s proposal_receipts[%d] endpoint %d is not a selected full-transaction candidate", label, index, receipt.EndpointOrdinal)
		}
		if receipt.EndpointHashSHA256 != chain[receipt.EndpointOrdinal-1] {
			return fmt.Errorf("%s proposal_receipts[%d] endpoint hash does not bind selected probe endpoint", label, index)
		}
		if receipt.RequestedDirectionSHA256 != endpoint.RequestedDirectionSHA256 || receipt.ActualDirectionSHA256 != endpoint.ActualDirectionSHA256 {
			return fmt.Errorf("%s proposal_receipts[%d] direction hashes do not bind selected probe endpoint", label, index)
		}
		if receipt.ActualDirectionProbeSHA256 != probeSHA {
			return fmt.Errorf("%s proposal_receipts[%d] probe receipt hash does not bind actual-direction probe", label, index)
		}
		if receipt.ProposalMagnitude != endpoint.ActualDirectionMaxAbs || receipt.ProposalBlockSize != len(endpoint.Indices) {
			return fmt.Errorf("%s proposal_receipts[%d] proposal magnitude/block does not bind selected endpoint actual direction", label, index)
		}
		wantMoved := aoqtMovedAngleIndices(endpoint.ActualDirection)
		if receipt.MovedAngleCount != len(wantMoved) || !slices.Equal(receipt.MovedAngleIndices, wantMoved) {
			return fmt.Errorf("%s proposal_receipts[%d] moved-angle evidence does not bind selected endpoint actual direction", label, index)
		}
	}
	return nil
}

// acceptTransactionalActualDirectionStep keeps the existing Adam path and
// replaces only the r3 coordinate fallback. It performs no more than two full
// transaction evaluations after the 132 endpoint ranking pass.
func (t *AOQTSidecarTrainer) acceptTransactionalActualDirectionStep(set AOQTSidecarCalibrationSet, objective AOQTSidecarVectorObjective, grad, q3GainGrad []float32, baseline aoqtStepEvaluation, evaluate func() (aoqtStepEvaluation, error), weights AOQTSidecarRowWeights, diagnostics *AOQTSidecarOptimizerDiagnostics, protectedGradients []aoqtProtectedAngleGradient) (bool, error) {
	if diagnostics == nil || evaluate == nil {
		return false, fmt.Errorf("AOQT V7-r3 actual-direction transactional inputs are required")
	}
	if len(q3GainGrad) != len(grad) {
		return false, fmt.Errorf("AOQT V7-r3 q3-gain gradient count = %d, want %d", len(q3GainGrad), len(grad))
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
		direction := aoqtDifference(t.angles, state.angles)
		evidence := aoqtProposalEvidence{magnitude: aoqtMaxAbsFloat32(direction), movedAngleIndices: aoqtMovedAngleIndices(direction)}
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
	// The probe is defined at the exact state that existed before the Adam
	// attempts. The final rejected Adam proposal remains resident after the
	// loop above, so restore the transactional baseline before deriving the
	// analytic ranking and evaluating the actual endpoints.
	t.restoreOptimizerState(state)
	search, err := t.actualDirectionSearch(set, objective)
	if err != nil {
		t.restoreOptimizerState(state)
		return false, err
	}
	plan, err := newAOQTCoordinateSearchPlanForMode(q3GainGrad, grad, t.config.LearningRate, t.config.AngleCap, t.config.OptimizerMode, protectedGradients)
	if err != nil {
		t.restoreOptimizerState(state)
		return false, err
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
	diagnostics.CoordinateSearchAuditChain, err = appendAOQTCoordinateSearchAuditChain(diagnostics.CoordinateSearchAuditChain, string(audit))
	if err != nil {
		t.restoreOptimizerState(state)
		return false, err
	}
	diagnostics.CoordinateSearchHashChain, err = appendAOQTCoordinateSearchHashChain(diagnostics.CoordinateSearchHashChain, string(audit))
	if err != nil {
		t.restoreOptimizerState(state)
		return false, err
	}
	var hashes []string
	if err := strictUnmarshalAOQT([]byte(diagnostics.CoordinateSearchHashChain), &hashes); err != nil || len(hashes) == 0 {
		t.restoreOptimizerState(state)
		if err != nil {
			return false, err
		}
		return false, fmt.Errorf("AOQT V7-r3 coordinate hash chain is empty")
	}
	diagnostics.CoordinateSearchOrderingHash = hashes[len(hashes)-1]
	for _, spec := range search.SelectedSpecs {
		if attemptsThisStep >= aoqtTransactionalMaxAttemptsPerStep {
			break
		}
		t.restoreOptimizerState(state)
		for _, rank := range spec.ranks {
			t.angles[rank.index] += rank.primaryDirection * spec.magnitude * float32(spec.globalSign)
		}
		if err := t.ProjectAngles(); err != nil {
			t.restoreOptimizerState(state)
			return false, err
		}
		direction := aoqtDifference(t.angles, state.angles)
		if len(aoqtMovedAngleIndices(direction)) == 0 {
			continue
		}
		attemptsThisStep++
		diagnostics.ProposalAttempts++
		diagnostics.CoordinateProposalAttempts++
		candidate, err := evaluate()
		if err != nil {
			t.restoreOptimizerState(state)
			return false, err
		}
		decision := t.evaluateTransactionalProposal(state, baseline, candidate, weights)
		evidence := aoqtProposalEvidence{
			magnitude:                  aoqtMaxAbsFloat32(direction),
			blockSize:                  len(spec.ranks),
			movedAngleIndices:          aoqtMovedAngleIndices(direction),
			directionSource:            AOQTV7R3ActualDirectionProbeDirectionSource,
			endpointOrdinal:            spec.ordinal,
			endpointHashSHA256:         spec.endpointHashSHA256,
			requestedDirectionSHA256:   spec.requestedDirectionSHA256,
			actualDirectionSHA256:      spec.actualDirectionSHA256,
			actualDirectionProbeSHA256: spec.probeReceiptSHA256,
		}
		diagnostics.RecordProposal(diagnostics.ProposalAttempts, aoqtProposalKindCoordinate, decision, baseline, candidate, evidence)
		if decision.accepted {
			diagnostics.CoordinateAcceptedProposals++
			diagnostics.AcceptedProposals++
			return true, nil
		}
		diagnostics.RejectedProposals++
		diagnostics.CoordinateRejectedProposals++
		diagnostics.Backtracks++
	}
	t.restoreOptimizerState(state)
	return false, nil
}
