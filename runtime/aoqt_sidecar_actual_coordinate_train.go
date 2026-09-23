package eosruntime

// V7-r5 is the first authorized full trainer for the actual-coordinate
// surface.  It is deliberately a separate path from the historical Adam /
// protected-cone fallback: the r4 search is used as an actual endpoint oracle,
// at most two endpoints are selected deterministically, and the unchanged
// full calibration set is the only transaction authority.

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"sort"
	"strings"
)

const (
	AOQTSidecarActualCoordinateTrainSchema            = "eos.aoqt.v7_r5_actual_coordinate_train.v1"
	AOQTSidecarActualCoordinateFullTrainSchema        = AOQTSidecarActualCoordinateTrainSchema
	AOQTV7R5ActualCoordinateTrainMaxSteps             = 2
	AOQTV7R5ActualCoordinateTrainMaxCandidatesPerStep = 2
	AOQTV7R5ActualCoordinateTrainFullRowCount         = 10942
	AOQTV7R5ActualCoordinateTrainSearchStrategy       = "actual_coordinate_full_train_v1"
	AOQTV7R5ActualCoordinateTrainStateVersion         = "aoqt-v7-r5-actual-coordinate-state-v1"
	AOQTV7R5ActualCoordinateTrainSelectionVersion     = "total_delta_q3_ordinal_v1"
	// This is the raw-file SHA-256 of the root-authorized, bound V7-r4
	// receipt. It is an authorization pin, not a caller-provided suggestion.
	AOQTV7R5CanonicalR4ReceiptSHA256 = "ae0950589be98ba35cadb8ec9097963019a7eeeaf3951db400b4c08986203844"
)

// Public aliases keep the contract names discoverable alongside the V7-r5
// prefixed constants used by the implementation.
const (
	AOQTSidecarActualCoordinateTrainMaxSteps             = AOQTV7R5ActualCoordinateTrainMaxSteps
	AOQTSidecarActualCoordinateTrainMaxCandidatesPerStep = AOQTV7R5ActualCoordinateTrainMaxCandidatesPerStep
	AOQTSidecarActualCoordinateTrainFullRowCount         = AOQTV7R5ActualCoordinateTrainFullRowCount
	AOQTSidecarActualCoordinateTrainCanonicalR4SHA256    = AOQTV7R5CanonicalR4ReceiptSHA256
)

// AOQTSidecarActualCoordinateTrainBinding binds the full trainer to the same
// materialized inputs as the canonical r4 receipt.  The r4 receipt itself is
// pinned separately because it is the authorization artifact, not merely a
// convenient diagnostic.
type AOQTSidecarActualCoordinateTrainBinding struct {
	Inputs                       AOQTSidecarRunMetricInputs     `json:"inputs"`
	IOReport                     AOQTSidecarCalibrationIOReport `json:"io_report"`
	PreflightPath                string                         `json:"preflight_path"`
	PreflightSHA256              string                         `json:"preflight_sha256"`
	SplitBinding                 *AOQTSidecarDevSplitBinding    `json:"split_binding,omitempty"`
	ObjectiveContractSHA256      string                         `json:"objective_contract_sha256"`
	EligibilityPolicySHA256      string                         `json:"eligibility_policy_sha256"`
	ScheduleSHA256               string                         `json:"schedule_sha256"`
	RankingSHA256                string                         `json:"ranking_sha256"`
	CoverageSHA256               string                         `json:"coverage_sha256"`
	CanonicalR4ReceiptPath       string                         `json:"canonical_r4_receipt_path"`
	CanonicalR4ReceiptSHA256     string                         `json:"canonical_r4_receipt_sha256"`
	CanonicalR4ReceiptFileSHA256 string                         `json:"canonical_r4_receipt_file_sha256"`
}

// AOQTSidecarActualCoordinateTrainState is a compact state witness.  The
// state hash covers the complete optimizer state (including the zero Adam
// buffers) so a candidate cannot be evaluated from a stale or partially
// restored state.  AnglesSHA256 remains a useful transform-level witness.
type AOQTSidecarActualCoordinateTrainState struct {
	StateSHA256   string `json:"state_sha256"`
	AnglesSHA256  string `json:"angles_sha256"`
	OptimizerStep int    `json:"optimizer_step"`
	AngleCount    int    `json:"angle_count"`
}

type aoqtV7R5StatePayload struct {
	Version       string    `json:"version"`
	Angles        []float32 `json:"angles"`
	AdamM         []float32 `json:"adam_m"`
	AdamV         []float32 `json:"adam_v"`
	OptimizerStep int       `json:"optimizer_step"`
}

func aoqtV7R5OptimizerStateSHA256(state aoqtOptimizerState) (string, error) {
	return aoqtCanonicalSHA256(aoqtV7R5StatePayload{
		Version:       AOQTV7R5ActualCoordinateTrainStateVersion,
		Angles:        append([]float32(nil), state.angles...),
		AdamM:         append([]float32(nil), state.adamM...),
		AdamV:         append([]float32(nil), state.adamV...),
		OptimizerStep: state.step,
	})
}

func (t *AOQTSidecarTrainer) aoqtV7R5StateEvidence() (AOQTSidecarActualCoordinateTrainState, error) {
	if t == nil {
		return AOQTSidecarActualCoordinateTrainState{}, fmt.Errorf("AOQT V7-r5 trainer is nil")
	}
	state := t.snapshotOptimizerState()
	stateSHA, err := aoqtV7R5OptimizerStateSHA256(state)
	if err != nil {
		return AOQTSidecarActualCoordinateTrainState{}, err
	}
	transform := t.Transform()
	anglesSHA, err := transform.AnglesSHA256()
	if err != nil {
		return AOQTSidecarActualCoordinateTrainState{}, err
	}
	return AOQTSidecarActualCoordinateTrainState{
		StateSHA256: stateSHA, AnglesSHA256: anglesSHA,
		OptimizerStep: state.step, AngleCount: len(state.angles),
	}, nil
}

// AOQTSidecarActualCoordinateTrainCandidate records both the r4 endpoint
// ranking evidence and the authoritative full-row transaction outcome.
type AOQTSidecarActualCoordinateTrainCandidate struct {
	Ordinal                       int                                      `json:"ordinal"`
	EndpointOrdinal               int                                      `json:"endpoint_ordinal"`
	SelectionRank                 int                                      `json:"selection_rank"`
	TotalDelta                    float32                                  `json:"total_delta"`
	Q3Delta                       float32                                  `json:"q3_delta"`
	Endpoint                      AOQTSidecarActualCoordinateProbeEndpoint `json:"endpoint"`
	StateBefore                   AOQTSidecarActualCoordinateTrainState    `json:"state_before"`
	CandidateState                AOQTSidecarActualCoordinateTrainState    `json:"candidate_state"`
	RestoredStateSHA256           string                                   `json:"restored_state_sha256"`
	FullBaselineLoss              float32                                  `json:"full_baseline_loss"`
	FullBaselineLossFinite        bool                                     `json:"full_baseline_loss_finite"`
	FullCandidateLoss             float32                                  `json:"full_candidate_loss"`
	FullCandidateLossFinite       bool                                     `json:"full_candidate_loss_finite"`
	FullBaselineComponents        AOQTSidecarObjectiveComponents           `json:"full_baseline_components"`
	FullCandidateComponents       AOQTSidecarObjectiveComponents           `json:"full_candidate_components"`
	FullBaselineComponentsFinite  bool                                     `json:"full_baseline_components_finite"`
	FullCandidateComponentsFinite bool                                     `json:"full_candidate_components_finite"`
	FullCandidateActivation       AOQTSidecarObjectiveActivation           `json:"full_candidate_activation"`
	FullTotalDelta                float32                                  `json:"full_total_delta"`
	FullTotalDeltaFinite          bool                                     `json:"full_total_delta_finite"`
	FullComponentDeltas           AOQTSidecarObjectiveComponents           `json:"full_component_deltas"`
	FullComponentDeltasFinite     bool                                     `json:"full_component_deltas_finite"`
	Q3GainEligible                bool                                     `json:"q3_gain_eligible"`
	ProtectedEligible             bool                                     `json:"protected_eligible"`
	TransactionEligible           bool                                     `json:"transaction_eligible"`
	Accepted                      bool                                     `json:"accepted"`
	Committed                     bool                                     `json:"committed"`
	Reason                        string                                   `json:"reason"`
	CandidateError                string                                   `json:"candidate_error,omitempty"`
}

type AOQTSidecarActualCoordinateTrainSearch struct {
	Step                     int                                     `json:"step"`
	StateBefore              AOQTSidecarActualCoordinateTrainState   `json:"state_before"`
	Receipt                  AOQTSidecarActualCoordinateProbeReceipt `json:"receipt"`
	ReceiptSHA256            string                                  `json:"receipt_sha256"`
	SelectedEndpointOrdinals []int                                   `json:"selected_endpoint_ordinals"`
	SelectionVersion         string                                  `json:"selection_version"`
	SelectionMaxCandidates   int                                     `json:"selection_max_candidates"`
}

type AOQTSidecarActualCoordinateTrainStep struct {
	Step                     int                                         `json:"step"`
	StateBefore              AOQTSidecarActualCoordinateTrainState       `json:"state_before"`
	Search                   AOQTSidecarActualCoordinateTrainSearch      `json:"search"`
	SelectedEndpointOrdinals []int                                       `json:"selected_endpoint_ordinals"`
	Candidates               []AOQTSidecarActualCoordinateTrainCandidate `json:"candidates"`
	Committed                bool                                        `json:"committed"`
	CommittedEndpointOrdinal int                                         `json:"committed_endpoint_ordinal,omitempty"`
	StateAfter               AOQTSidecarActualCoordinateTrainState       `json:"state_after"`
}

type AOQTSidecarActualCoordinateTrainReceipt struct {
	Schema                       string                                  `json:"schema"`
	Required                     bool                                    `json:"required"`
	Passed                       bool                                    `json:"passed"`
	FailureReason                string                                  `json:"failure_reason,omitempty"`
	MaxSteps                     int                                     `json:"max_steps"`
	Steps                        int                                     `json:"steps"`
	AttemptedSteps               int                                     `json:"attempted_steps"`
	AcceptedSteps                int                                     `json:"accepted_steps"`
	FullRowCount                 int                                     `json:"full_row_count"`
	FullCandidateEvaluationCount int                                     `json:"full_candidate_evaluation_count"`
	SearchCount                  int                                     `json:"search_count"`
	IdentityStateSHA256          string                                  `json:"identity_state_sha256"`
	InitialState                 AOQTSidecarActualCoordinateTrainState   `json:"initial_state"`
	FinalState                   AOQTSidecarActualCoordinateTrainState   `json:"final_state"`
	CanonicalR4ReceiptPath       string                                  `json:"canonical_r4_receipt_path"`
	CanonicalR4ReceiptSHA256     string                                  `json:"canonical_r4_receipt_sha256"`
	CanonicalR4ReceiptFileSHA256 string                                  `json:"canonical_r4_receipt_file_sha256"`
	StepHashChain                string                                  `json:"step_hash_chain"`
	StepHashChainTailSHA256      string                                  `json:"step_hash_chain_tail_sha256"`
	SearchStrategy               string                                  `json:"search_strategy"`
	SelectionVersion             string                                  `json:"selection_version"`
	StepsEvidence                []AOQTSidecarActualCoordinateTrainStep  `json:"steps_evidence"`
	Binding                      AOQTSidecarActualCoordinateTrainBinding `json:"binding"`
	QualityClaim                 bool                                    `json:"quality_claim"`
	ReleaseClaim                 bool                                    `json:"release_claim"`
	OfficialClaim                bool                                    `json:"official_claim"`
	OfficialHeldoutGate          bool                                    `json:"official_heldout_gate"`
	CommercialClaim              bool                                    `json:"commercial_claim"`
}

// Alias used by callers that spell the artifact "full train".
type AOQTSidecarActualCoordinateFullTrainReceipt = AOQTSidecarActualCoordinateTrainReceipt

func (r AOQTSidecarActualCoordinateTrainReceipt) SHA256() (string, error) {
	return aoqtCanonicalSHA256(r)
}

func (r AOQTSidecarActualCoordinateTrainReceipt) Validate() error {
	if r.Schema != AOQTSidecarActualCoordinateTrainSchema {
		return fmt.Errorf("AOQT V7-r5 receipt schema %q is unsupported, want %q", r.Schema, AOQTSidecarActualCoordinateTrainSchema)
	}
	if !r.Required {
		return fmt.Errorf("AOQT V7-r5 receipt must be required")
	}
	if r.QualityClaim || r.ReleaseClaim || r.OfficialClaim || r.OfficialHeldoutGate || r.CommercialClaim {
		return fmt.Errorf("AOQT V7-r5 receipt claims must be false")
	}
	if r.MaxSteps != AOQTV7R5ActualCoordinateTrainMaxSteps {
		return fmt.Errorf("AOQT V7-r5 max_steps = %d, want exactly %d", r.MaxSteps, AOQTV7R5ActualCoordinateTrainMaxSteps)
	}
	if r.FullRowCount != AOQTV7R5ActualCoordinateTrainFullRowCount || r.FullCandidateEvaluationCount < 0 || r.SearchCount < 0 {
		return fmt.Errorf("AOQT V7-r5 full row/search accounting = rows:%d candidates:%d searches:%d; want exactly %d rows", r.FullRowCount, r.FullCandidateEvaluationCount, r.SearchCount, AOQTV7R5ActualCoordinateTrainFullRowCount)
	}
	for _, item := range []struct{ name, value string }{
		{"identity_state_sha256", r.IdentityStateSHA256},
		{"canonical_r4_receipt_sha256", r.CanonicalR4ReceiptSHA256},
	} {
		if err := validateAOQTSHA256(item.value, "AOQT V7-r5 "+item.name); err != nil {
			return err
		}
	}
	if strings.TrimSpace(r.CanonicalR4ReceiptFileSHA256) != "" {
		if err := validateAOQTSHA256(r.CanonicalR4ReceiptFileSHA256, "AOQT V7-r5 canonical r4 receipt file sha256"); err != nil {
			return err
		}
	}
	if r.CanonicalR4ReceiptSHA256 != AOQTV7R5CanonicalR4ReceiptSHA256 {
		return fmt.Errorf("AOQT V7-r5 canonical r4 receipt sha256 = %q, want pinned %q", r.CanonicalR4ReceiptSHA256, AOQTV7R5CanonicalR4ReceiptSHA256)
	}
	if r.SearchStrategy != AOQTV7R5ActualCoordinateTrainSearchStrategy || r.SelectionVersion != AOQTV7R5ActualCoordinateTrainSelectionVersion {
		return fmt.Errorf("AOQT V7-r5 search/selection contract is unsupported")
	}
	for _, item := range []struct {
		name  string
		state AOQTSidecarActualCoordinateTrainState
	}{
		{"initial", r.InitialState},
		{"final", r.FinalState},
	} {
		if err := validateAOQTSHA256(item.state.StateSHA256, "AOQT V7-r5 "+item.name+" state sha256"); err != nil {
			return err
		}
		if err := validateAOQTSHA256(item.state.AnglesSHA256, "AOQT V7-r5 "+item.name+" angles sha256"); err != nil {
			return err
		}
		if item.state.AngleCount <= 0 || item.state.OptimizerStep < 0 {
			return fmt.Errorf("AOQT V7-r5 %s state dimensions/step are invalid", item.name)
		}
	}
	if r.InitialState.StateSHA256 != r.IdentityStateSHA256 {
		return fmt.Errorf("AOQT V7-r5 initial state does not equal identity state")
	}
	if r.FinalState.AngleCount != r.InitialState.AngleCount || r.FinalState.OptimizerStep < r.InitialState.OptimizerStep {
		return fmt.Errorf("AOQT V7-r5 state angle counts are invalid")
	}
	if r.AttemptedSteps != len(r.StepsEvidence) || r.SearchCount != len(r.StepsEvidence) || r.Steps != r.AcceptedSteps || r.AcceptedSteps < 0 || r.AttemptedSteps < r.AcceptedSteps || r.AttemptedSteps > r.MaxSteps || r.FullCandidateEvaluationCount > r.AttemptedSteps*AOQTV7R5ActualCoordinateTrainMaxCandidatesPerStep {
		return fmt.Errorf("AOQT V7-r5 step accounting is inconsistent")
	}
	chain, err := aoqtV7R5DecodeStepChain(r.StepHashChain)
	if err != nil {
		return err
	}
	if len(chain) != len(r.StepsEvidence) {
		return fmt.Errorf("AOQT V7-r5 step hash chain length = %d, want %d", len(chain), len(r.StepsEvidence))
	}
	if len(chain) == 0 {
		if strings.TrimSpace(r.StepHashChainTailSHA256) != "" {
			if err := validateAOQTSHA256(r.StepHashChainTailSHA256, "AOQT V7-r5 step hash chain tail sha256"); err != nil {
				return err
			}
		}
		if r.Passed || r.AttemptedSteps != 0 || r.Steps != 0 || !strings.HasPrefix(r.FailureReason, "precondition:") {
			return fmt.Errorf("AOQT V7-r5 empty step hash chain is only valid for a failed precondition")
		}
		return nil
	}
	previous := ""
	fullEvaluations := 0
	for i, step := range r.StepsEvidence {
		if step.Step != i || step.StateBefore.StateSHA256 == "" || step.StateAfter.StateSHA256 == "" {
			return fmt.Errorf("AOQT V7-r5 step %d index/state evidence is invalid", i)
		}
		if i > 0 && !reflect.DeepEqual(step.StateBefore, r.StepsEvidence[i-1].StateAfter) {
			return fmt.Errorf("AOQT V7-r5 step %d does not start at prior after-state", i)
		}
		if step.Search.Step != step.Step || !reflect.DeepEqual(step.Search.StateBefore, step.StateBefore) || step.Search.SelectionVersion != r.SelectionVersion || step.Search.SelectionMaxCandidates != AOQTV7R5ActualCoordinateTrainMaxCandidatesPerStep {
			return fmt.Errorf("AOQT V7-r5 step %d search state evidence is stale", i)
		}
		if err := step.Search.Receipt.Validate(); err != nil {
			return fmt.Errorf("AOQT V7-r5 step %d r4 search receipt: %w", i, err)
		}
		searchSHA, err := step.Search.Receipt.SHA256()
		if err != nil || searchSHA != step.Search.ReceiptSHA256 {
			return fmt.Errorf("AOQT V7-r5 step %d search receipt sha256 mismatch", i)
		}
		// The selected list is the deterministic max-two prefix from the fresh
		// r4 search.  Candidate evidence may be a shorter prefix because the
		// first accepted transaction stops the step immediately; an accepted
		// second candidate therefore still has rejected evidence for candidate 1.
		if len(step.SelectedEndpointOrdinals) > AOQTV7R5ActualCoordinateTrainMaxCandidatesPerStep || len(step.Candidates) > len(step.SelectedEndpointOrdinals) {
			return fmt.Errorf("AOQT V7-r5 step %d selected/candidate accounting is invalid", i)
		}
		wantSelected := aoqtV7R5SortSelectedEndpoints(step.Search.Receipt)
		wantSelectedOrdinals := make([]int, len(wantSelected))
		for j := range wantSelected {
			wantSelectedOrdinals[j] = wantSelected[j].Ordinal
		}
		if !reflect.DeepEqual(step.SelectedEndpointOrdinals, wantSelectedOrdinals) {
			return fmt.Errorf("AOQT V7-r5 step %d selected endpoints do not match fresh search", i)
		}
		for j, candidate := range step.Candidates {
			if candidate.Ordinal != j+1 || candidate.SelectionRank != j || candidate.EndpointOrdinal != step.SelectedEndpointOrdinals[j] {
				return fmt.Errorf("AOQT V7-r5 step %d candidate %d ordinal/selection mismatch", i, j)
			}
			if !reflect.DeepEqual(candidate.StateBefore, step.StateBefore) {
				return fmt.Errorf("AOQT V7-r5 step %d candidate %d state-before mismatch", i, j)
			}
			if !containsInt(candidate.EndpointOrdinal, step.Search.SelectedEndpointOrdinals) {
				return fmt.Errorf("AOQT V7-r5 step %d candidate %d is not selected by search", i, j)
			}
			var searchEndpoint *AOQTSidecarActualCoordinateProbeEndpoint
			for k := range step.Search.Receipt.Endpoints {
				if step.Search.Receipt.Endpoints[k].Ordinal == candidate.EndpointOrdinal {
					searchEndpoint = &step.Search.Receipt.Endpoints[k]
					break
				}
			}
			if searchEndpoint == nil || !reflect.DeepEqual(candidate.Endpoint, *searchEndpoint) {
				return fmt.Errorf("AOQT V7-r5 step %d candidate %d endpoint evidence mismatch", i, j)
			}
			if !candidate.FullBaselineLossFinite || !isFinite32(candidate.FullBaselineLoss) {
				return fmt.Errorf("AOQT V7-r5 step %d candidate %d full loss evidence is not finite", i, j)
			}
			if !candidate.FullBaselineComponentsFinite || !aoqtFiniteReceiptComponentsValue(candidate.FullBaselineComponents) {
				return fmt.Errorf("AOQT V7-r5 step %d candidate %d full component evidence is not finite", i, j)
			}
			if candidate.CandidateError != "" {
				if candidate.Accepted || candidate.Committed || candidate.FullCandidateLossFinite || candidate.FullCandidateComponentsFinite || candidate.RestoredStateSHA256 != candidate.StateBefore.StateSHA256 {
					return fmt.Errorf("AOQT V7-r5 step %d candidate %d error/rollback evidence is invalid", i, j)
				}
				continue
			}
			fullEvaluations++
			if !candidate.FullCandidateLossFinite || !isFinite32(candidate.FullCandidateLoss) || !candidate.FullCandidateComponentsFinite || !aoqtFiniteReceiptComponentsValue(candidate.FullCandidateComponents) || !candidate.FullTotalDeltaFinite || !isFinite32(candidate.FullTotalDelta) || !candidate.FullComponentDeltasFinite || !aoqtFiniteReceiptComponentsValue(candidate.FullComponentDeltas) {
				return fmt.Errorf("AOQT V7-r5 step %d candidate %d full candidate evidence is not finite", i, j)
			}
			if candidate.FullTotalDelta != candidate.FullCandidateLoss-candidate.FullBaselineLoss || candidate.FullComponentDeltas != aoqtObjectiveComponentDelta(candidate.FullBaselineComponents, candidate.FullCandidateComponents) {
				return fmt.Errorf("AOQT V7-r5 step %d candidate %d full delta evidence mismatch", i, j)
			}
			if candidate.TotalDelta != candidate.Endpoint.TotalDelta || candidate.Q3Delta != candidate.Endpoint.ComponentDeltas.Q3Gain {
				return fmt.Errorf("AOQT V7-r5 step %d candidate %d r4 ranking evidence mismatch", i, j)
			}
			if candidate.TransactionEligible != candidate.Accepted || candidate.Accepted != candidate.Committed {
				return fmt.Errorf("AOQT V7-r5 step %d candidate %d accepted/committed mismatch", i, j)
			}
			if err := validateAOQTSHA256(candidate.CandidateState.StateSHA256, "AOQT V7-r5 candidate state sha256"); err != nil {
				return err
			}
			if err := validateAOQTSHA256(candidate.CandidateState.AnglesSHA256, "AOQT V7-r5 candidate angles sha256"); err != nil {
				return err
			}
			if !candidate.Accepted && candidate.RestoredStateSHA256 != candidate.StateBefore.StateSHA256 {
				return fmt.Errorf("AOQT V7-r5 step %d candidate %d rejected candidate was not exactly rolled back", i, j)
			}
		}
		accepted := 0
		for _, candidate := range step.Candidates {
			if candidate.Accepted {
				accepted++
			}
		}
		if accepted > 1 || (step.Committed && accepted != 1) || (!step.Committed && accepted != 0) {
			return fmt.Errorf("AOQT V7-r5 step %d commit accounting is invalid", i)
		}
		committedEndpointOrdinal := 0
		for _, candidate := range step.Candidates {
			if candidate.Accepted {
				committedEndpointOrdinal = candidate.EndpointOrdinal
				break
			}
		}
		if step.Committed && (committedEndpointOrdinal == 0 || step.CommittedEndpointOrdinal != committedEndpointOrdinal) {
			return fmt.Errorf("AOQT V7-r5 step %d committed endpoint mismatch", i)
		}
		if !step.Committed && step.CommittedEndpointOrdinal != 0 {
			return fmt.Errorf("AOQT V7-r5 step %d rejected step has a committed endpoint", i)
		}
		if !step.Committed && !reflect.DeepEqual(step.StateAfter, step.StateBefore) {
			return fmt.Errorf("AOQT V7-r5 step %d zero-safe step did not exactly restore state", i)
		}
		if step.Committed && step.StateAfter.OptimizerStep != step.StateBefore.OptimizerStep+1 {
			return fmt.Errorf("AOQT V7-r5 step %d committed state did not advance exactly one optimizer step", i)
		}
		if step.StateAfter.AngleCount != step.StateBefore.AngleCount {
			return fmt.Errorf("AOQT V7-r5 step %d changed the angle count", i)
		}
		if step.Committed {
			for _, candidate := range step.Candidates {
				if candidate.Accepted && !reflect.DeepEqual(candidate.CandidateState, step.StateAfter) {
					return fmt.Errorf("AOQT V7-r5 step %d committed candidate state does not match step after-state", i)
				}
			}
		}
		payload, err := json.Marshal(step)
		if err != nil {
			return err
		}
		want := sha256AOQTCoordinateSearchChainEntry(previous, string(payload))
		if chain[i] != want {
			return fmt.Errorf("AOQT V7-r5 step hash chain[%d] mismatch", i)
		}
		previous = chain[i]
	}
	if previous != r.StepHashChainTailSHA256 {
		return fmt.Errorf("AOQT V7-r5 step hash chain tail mismatch")
	}
	acceptedSteps := 0
	for _, step := range r.StepsEvidence {
		if step.Committed {
			acceptedSteps++
		}
	}
	if acceptedSteps != r.AcceptedSteps {
		return fmt.Errorf("AOQT V7-r5 accepted step count does not match step evidence")
	}
	if fullEvaluations != r.FullCandidateEvaluationCount {
		return fmt.Errorf("AOQT V7-r5 full candidate evaluation count = %d, evidence count = %d", r.FullCandidateEvaluationCount, fullEvaluations)
	}
	if len(r.StepsEvidence) > 0 && r.FinalState.StateSHA256 != r.StepsEvidence[len(r.StepsEvidence)-1].StateAfter.StateSHA256 {
		return fmt.Errorf("AOQT V7-r5 final state does not match step evidence")
	}
	if r.Passed {
		if r.AttemptedSteps != r.MaxSteps || r.AcceptedSteps != r.MaxSteps || strings.TrimSpace(r.FailureReason) != "" {
			return fmt.Errorf("AOQT V7-r5 passed receipt requires all max steps accepted and no failure")
		}
	} else if strings.TrimSpace(r.FailureReason) == "" {
		return fmt.Errorf("AOQT V7-r5 failed receipt requires failure_reason")
	}
	return nil
}

func containsInt(value int, values []int) bool {
	for _, candidate := range values {
		if candidate == value {
			return true
		}
	}
	return false
}

func aoqtV7R5DecodeStepChain(data string) ([]string, error) {
	var chain []string
	if err := strictUnmarshalAOQT([]byte(data), &chain); err != nil {
		return nil, fmt.Errorf("AOQT V7-r5 step hash chain is invalid: %w", err)
	}
	canonical, err := json.Marshal(chain)
	if err != nil || string(canonical) != data {
		return nil, fmt.Errorf("AOQT V7-r5 step hash chain is not canonical JSON")
	}
	for _, hash := range chain {
		if err := validateAOQTSHA256(hash, "AOQT V7-r5 step hash chain entry"); err != nil {
			return nil, err
		}
	}
	return chain, nil
}

func aoqtV7R5CanonicalR4Receipt(t *AOQTSidecarTrainer) *AOQTSidecarActualCoordinateProbeReceipt {
	if t == nil {
		return nil
	}
	if t.config.ActualCoordinateTrainR4Receipt != nil {
		return t.config.ActualCoordinateTrainR4Receipt
	}
	return t.config.CanonicalR4Receipt
}

func aoqtV7R5CanonicalR4ReceiptPath(t *AOQTSidecarTrainer) string {
	if t == nil {
		return ""
	}
	if strings.TrimSpace(t.config.ActualCoordinateTrainR4ReceiptPath) != "" {
		return t.config.ActualCoordinateTrainR4ReceiptPath
	}
	return t.config.CanonicalR4ReceiptPath
}

func aoqtV7R5CanonicalR4ReceiptSHA256(t *AOQTSidecarTrainer) string {
	if t == nil {
		return ""
	}
	if strings.TrimSpace(t.config.ActualCoordinateTrainR4ReceiptSHA256) != "" {
		return t.config.ActualCoordinateTrainR4ReceiptSHA256
	}
	return t.config.CanonicalR4ReceiptSHA256
}

func (r AOQTSidecarActualCoordinateTrainReceipt) ValidateBound() error {
	if err := r.Validate(); err != nil {
		return err
	}
	if r.Binding.SplitBinding == nil {
		return fmt.Errorf("AOQT V7-r5 bound receipt requires split binding")
	}
	if err := r.Binding.SplitBinding.Validate(); err != nil {
		return err
	}
	if err := validateAOQTMetricInputs(r.Binding.Inputs); err != nil {
		return fmt.Errorf("AOQT V7-r5 bound receipt inputs: %w", err)
	}
	if r.Binding.IOReport.Schema != AOQTSidecarCalibrationIOReportSchema || r.Binding.IOReport.ManifestSHA256 != r.Binding.Inputs.DatasetManifestSHA256 {
		return fmt.Errorf("AOQT V7-r5 bound receipt IO report provenance does not match inputs")
	}
	if r.Binding.IOReport.RowCount != AOQTV7R5ActualCoordinateTrainFullRowCount {
		return fmt.Errorf("AOQT V7-r5 bound receipt IO report row_count = %d, want exactly %d", r.Binding.IOReport.RowCount, AOQTV7R5ActualCoordinateTrainFullRowCount)
	}
	if r.Binding.SplitBinding != nil && (r.Binding.SplitBinding.TrainRowCount != AOQTV7R5ActualCoordinateTrainFullRowCount || r.Binding.SplitBinding.MaterializedRowCount != AOQTV7R5ActualCoordinateTrainFullRowCount) {
		return fmt.Errorf("AOQT V7-r5 bound receipt split row counts must be exactly %d", AOQTV7R5ActualCoordinateTrainFullRowCount)
	}
	if err := validateAOQTSHA256(r.Binding.IOReport.RowsSHA256, "AOQT V7-r5 bound rows sha256"); err != nil {
		return err
	}
	if strings.TrimSpace(r.Binding.IOReport.ManifestPath) == "" || strings.TrimSpace(r.Binding.IOReport.RowsJSONLPath) == "" || strings.TrimSpace(r.Binding.PreflightPath) == "" {
		return fmt.Errorf("AOQT V7-r5 bound receipt source paths are required")
	}
	if err := validateAOQTSHA256(r.Binding.PreflightSHA256, "AOQT V7-r5 bound preflight sha256"); err != nil {
		return err
	}
	if err := validateAOQTSHA256(r.Binding.CanonicalR4ReceiptSHA256, "AOQT V7-r5 bound canonical r4 sha256"); err != nil {
		return err
	}
	if r.Binding.CanonicalR4ReceiptSHA256 != r.CanonicalR4ReceiptSHA256 || r.Binding.CanonicalR4ReceiptPath != r.CanonicalR4ReceiptPath {
		return fmt.Errorf("AOQT V7-r5 bound canonical r4 pin does not match receipt")
	}
	data, err := os.ReadFile(r.CanonicalR4ReceiptPath)
	if err != nil {
		return fmt.Errorf("AOQT V7-r5 canonical r4 receipt reread: %w", err)
	}
	fileSHA := sha256BytesAOQT(data)
	if strings.TrimSpace(r.CanonicalR4ReceiptFileSHA256) == "" || fileSHA != r.CanonicalR4ReceiptFileSHA256 {
		return fmt.Errorf("AOQT V7-r5 canonical r4 receipt file sha256 mismatch")
	}
	if r.Binding.CanonicalR4ReceiptFileSHA256 != r.CanonicalR4ReceiptFileSHA256 {
		return fmt.Errorf("AOQT V7-r5 bound canonical r4 file sha256 does not match receipt")
	}
	var canonical AOQTSidecarActualCoordinateProbeReceipt
	if err := strictUnmarshalAOQT(data, &canonical); err != nil {
		return fmt.Errorf("AOQT V7-r5 canonical r4 receipt is invalid: %w", err)
	}
	if err := canonical.ValidateBound(); err != nil {
		return fmt.Errorf("AOQT V7-r5 canonical r4 ValidateBound: %w", err)
	}
	if fileSHA != AOQTV7R5CanonicalR4ReceiptSHA256 || r.CanonicalR4ReceiptSHA256 != AOQTV7R5CanonicalR4ReceiptSHA256 {
		return fmt.Errorf("AOQT V7-r5 canonical r4 receipt sha256 is not the pinned authorization")
	}
	// The canonical receipt is the immutable source authorization.  Requiring
	// the outer binding to equal its already-validated provenance prevents a
	// caller from rehashing nested searches against a different path/data set.
	if !reflect.DeepEqual(r.Binding.Inputs, canonical.Binding.Inputs) || !reflect.DeepEqual(r.Binding.IOReport, canonical.Binding.IOReport) || r.Binding.PreflightPath != canonical.Binding.PreflightPath || r.Binding.PreflightSHA256 != canonical.Binding.PreflightSHA256 || !reflect.DeepEqual(r.Binding.SplitBinding, canonical.Binding.SplitBinding) {
		return fmt.Errorf("AOQT V7-r5 bound receipt source binding does not match canonical r4 provenance")
	}
	if r.Binding.ObjectiveContractSHA256 != canonical.Binding.ObjectiveContractSHA256 || r.Binding.EligibilityPolicySHA256 != canonical.Binding.EligibilityPolicySHA256 || r.Binding.ScheduleSHA256 != canonical.Binding.ScheduleSHA256 || r.Binding.RankingSHA256 != canonical.Binding.RankingSHA256 || r.Binding.CoverageSHA256 != canonical.Binding.CoverageSHA256 {
		return fmt.Errorf("AOQT V7-r5 bound receipt contract/schedule binding does not match canonical r4 provenance")
	}
	for i := range r.StepsEvidence {
		step := &r.StepsEvidence[i]
		if err := validateAOQTV7R5SearchBinding(step.Search.Receipt, r.Binding); err != nil {
			return fmt.Errorf("AOQT V7-r5 step %d search receipt: %w", i, err)
		}
	}
	return nil
}

// validateAOQTV7R5SearchBinding checks the immutable source binding on each
// nested r4 receipt against the already-authorized outer binding.  This is an
// independent comparison: the nested receipt's self-digests are not treated
// as authority.  In particular, changing an IO report, contract/policy,
// coverage, schedule, or ranking and then rehashing the nested receipt still
// fails because every authoritative field and digest must equal the outer
// r4-bound values.
func validateAOQTV7R5SearchBinding(search AOQTSidecarActualCoordinateProbeReceipt, outer AOQTSidecarActualCoordinateTrainBinding) error {
	if err := search.Validate(); err != nil {
		return err
	}
	if !reflect.DeepEqual(search.Binding.Inputs, outer.Inputs) {
		return fmt.Errorf("is not bound to the full training metric inputs")
	}
	if err := validateAOQTV7R5IOReportBinding(search.Binding.IOReport, outer.IOReport); err != nil {
		return fmt.Errorf("IO report is not authoritative: %w", err)
	}
	if search.Binding.PreflightPath != outer.PreflightPath || search.Binding.PreflightSHA256 != outer.PreflightSHA256 {
		return fmt.Errorf("preflight binding does not match full training inputs")
	}
	if search.Binding.SplitBinding == nil || outer.SplitBinding == nil || !reflect.DeepEqual(search.Binding.SplitBinding, outer.SplitBinding) {
		return fmt.Errorf("split binding does not match full training inputs")
	}
	for _, item := range []struct {
		name       string
		searchHash string
		bindHash   string
		want       string
	}{
		{"objective contract", search.ObjectiveContractSHA256, search.Binding.ObjectiveContractSHA256, outer.ObjectiveContractSHA256},
		{"eligibility policy", search.EligibilityPolicySHA256, search.Binding.EligibilityPolicySHA256, outer.EligibilityPolicySHA256},
		{"schedule", search.ScheduleSHA256, search.Binding.ScheduleSHA256, outer.ScheduleSHA256},
		{"ranking", search.RankingSHA256, search.Binding.RankingSHA256, outer.RankingSHA256},
		{"coverage", search.CoverageSHA256, search.Binding.CoverageSHA256, outer.CoverageSHA256},
	} {
		if err := validateAOQTSHA256(item.searchHash, "AOQT V7-r5 nested "+item.name); err != nil {
			return err
		}
		if item.searchHash != item.bindHash || item.bindHash != item.want {
			return fmt.Errorf("%s digest does not match authoritative r4 binding", item.name)
		}
	}
	return nil
}

// validateAOQTV7R5IOReportBinding deliberately enumerates the complete IO
// report surface before the final structural comparison.  A path/hash-only
// check is insufficient: a coordinated forgery can preserve those values
// while changing counts, topology, legal gates, objective, or training
// policy, then recompute the nested receipt hashes.
func validateAOQTV7R5IOReportBinding(got, want AOQTSidecarCalibrationIOReport) error {
	if got.Schema != AOQTSidecarCalibrationIOReportSchema || want.Schema != AOQTSidecarCalibrationIOReportSchema || got.Schema != want.Schema {
		return fmt.Errorf("schema mismatch")
	}
	for _, item := range []struct {
		name string
		got  string
		want string
	}{
		{"manifest path", got.ManifestPath, want.ManifestPath},
		{"rows path", got.RowsJSONLPath, want.RowsJSONLPath},
		{"manifest sha256", got.ManifestSHA256, want.ManifestSHA256},
		{"rows sha256", got.RowsSHA256, want.RowsSHA256},
		{"anchor artifact sha256", got.AnchorArtifactSHA256, want.AnchorArtifactSHA256},
		{"anchor package manifest sha256", got.AnchorPackageManifestSHA256, want.AnchorPackageManifestSHA256},
		{"anchor embedding space id", got.AnchorEmbeddingSpaceID, want.AnchorEmbeddingSpaceID},
		{"compatibility digest", got.CompatibilityDigest, want.CompatibilityDigest},
		{"training contract", got.TrainingContract, want.TrainingContract},
	} {
		if item.got != item.want {
			return fmt.Errorf("%s mismatch", item.name)
		}
	}
	if got.TurboQuantSeed != want.TurboQuantSeed {
		return fmt.Errorf("turboquant seed mismatch")
	}
	if got.RowCount != want.RowCount || got.CandidateCount != want.CandidateCount || got.PairCount != want.PairCount {
		return fmt.Errorf("row/candidate/pair counts mismatch")
	}
	if !reflect.DeepEqual(got.Topology, want.Topology) {
		return fmt.Errorf("topology mismatch")
	}
	if !reflect.DeepEqual(got.ObjectiveContract, want.ObjectiveContract) {
		return fmt.Errorf("objective contract mismatch")
	}
	if !reflect.DeepEqual(got.LegalGates, want.LegalGates) {
		return fmt.Errorf("legal gates mismatch")
	}
	if !reflect.DeepEqual(got.CandidateEligibilityPolicy, want.CandidateEligibilityPolicy) {
		return fmt.Errorf("candidate eligibility policy mismatch")
	}
	for _, item := range []struct {
		name  string
		value string
	}{
		{"manifest sha256", got.ManifestSHA256},
		{"rows sha256", got.RowsSHA256},
		{"anchor artifact sha256", got.AnchorArtifactSHA256},
		{"anchor package manifest sha256", got.AnchorPackageManifestSHA256},
	} {
		if err := validateAOQTSHA256(item.value, "AOQT V7-r5 authoritative IO report "+item.name); err != nil {
			return err
		}
	}
	if !reflect.DeepEqual(got, want) {
		return fmt.Errorf("complete IO report topology/contracts/legal-gates/policy mismatch")
	}
	return nil
}

func aoqtV7R5SortSelectedEndpoints(receipt AOQTSidecarActualCoordinateProbeReceipt) []AOQTSidecarActualCoordinateProbeEndpoint {
	selected := make([]AOQTSidecarActualCoordinateProbeEndpoint, 0, len(receipt.Endpoints))
	for _, endpoint := range receipt.Endpoints {
		if endpoint.TransactionEligible {
			selected = append(selected, endpoint)
		}
	}
	sort.SliceStable(selected, func(i, j int) bool {
		if selected[i].TotalDelta != selected[j].TotalDelta {
			return selected[i].TotalDelta < selected[j].TotalDelta
		}
		if selected[i].ComponentDeltas.Q3Gain != selected[j].ComponentDeltas.Q3Gain {
			return selected[i].ComponentDeltas.Q3Gain < selected[j].ComponentDeltas.Q3Gain
		}
		return selected[i].Ordinal < selected[j].Ordinal
	})
	if len(selected) > AOQTV7R5ActualCoordinateTrainMaxCandidatesPerStep {
		selected = selected[:AOQTV7R5ActualCoordinateTrainMaxCandidatesPerStep]
	}
	return selected
}

func aoqtV7R5SearchReceiptSHA256(receipt AOQTSidecarActualCoordinateProbeReceipt) (string, error) {
	receipt.Binding = AOQTSidecarActualCoordinateProbeBinding{}
	return receipt.SHA256()
}

func aoqtV7R5StepHashChain(steps []AOQTSidecarActualCoordinateTrainStep) (string, string, error) {
	chain := make([]string, 0, len(steps))
	previous := ""
	for _, step := range steps {
		payload, err := json.Marshal(step)
		if err != nil {
			return "", "", err
		}
		previous = sha256AOQTCoordinateSearchChainEntry(previous, string(payload))
		chain = append(chain, previous)
	}
	data, err := json.Marshal(chain)
	if err != nil {
		return "", "", err
	}
	return string(data), previous, nil
}

// initializeV7R5FailureArtifacts creates the two fail-closed artifacts before
// any r5 precondition is inspected.  In particular, objective construction,
// canonical-receipt validation, identity checks, and full-row checks must not
// be able to return a summary with a missing receipt or diagnostics witness.
// The values that are authoritative for a bound receipt are filled in by the
// runner after materialization; the structural fields here are deliberately
// fixed by the r5 contract so even an early failed receipt remains valid.
func (t *AOQTSidecarTrainer) initializeV7R5FailureArtifacts() (AOQTSidecarOptimizerDiagnostics, AOQTSidecarActualCoordinateTrainReceipt, error) {
	if t == nil {
		return AOQTSidecarOptimizerDiagnostics{}, AOQTSidecarActualCoordinateTrainReceipt{}, fmt.Errorf("AOQT V7-r5 trainer is nil")
	}
	identity, err := t.aoqtV7R5StateEvidence()
	if err != nil {
		// Preserve a valid failure witness even when the mutable state itself is
		// malformed (for example, a non-finite angle makes the normal canonical
		// state hash unavailable). The failure reason is emitted by the caller;
		// these hashes only provide a stable, auditable placeholder witness.
		state := t.snapshotOptimizerState()
		angleCount := len(state.angles)
		if angleCount == 0 {
			angleCount = 1
		}
		identity = AOQTSidecarActualCoordinateTrainState{
			StateSHA256:  sha256BytesAOQT([]byte("aoqt-v7-r5-invalid-state:" + err.Error())),
			AnglesSHA256: sha256BytesAOQT([]byte("aoqt-v7-r5-invalid-angles:" + err.Error())),
			AngleCount:   angleCount,
		}
	}
	canonicalPin := aoqtV7R5CanonicalR4ReceiptSHA256(t)
	// A malformed caller-supplied pin is itself a precondition failure. Keep
	// the emitted failed receipt structurally valid so the failure can still be
	// audited; the fit path reports the actual pin mismatch separately.
	if canonicalPin != AOQTV7R5CanonicalR4ReceiptSHA256 {
		canonicalPin = AOQTV7R5CanonicalR4ReceiptSHA256
	}
	diagnostics := AOQTSidecarOptimizerDiagnostics{
		PlannedSteps:                 AOQTV7R5ActualCoordinateTrainMaxSteps,
		MaxAttemptsPerStep:           AOQTV7R5ActualCoordinateTrainMaxCandidatesPerStep,
		OptimizerMode:                t.config.OptimizerMode,
		DevOnly:                      true,
		CoordinateSearchStrategy:     AOQTV7R5ActualCoordinateTrainSearchStrategy,
		CoordinateSearchLearningRate: t.config.LearningRate,
		CoordinateTopAngles:          AOQTV7R4ActualCoordinateProbeTopAngles,
		CoordinateMagnitudeCount:     AOQTV7R4ActualCoordinateProbeMagnitudeCount,
		CoordinateBlockCount:         0,
	}
	receipts := make(AOQTSidecarOptimizerProposalReceipts, 0, AOQTV7R5ActualCoordinateTrainMaxSteps*AOQTV7R5ActualCoordinateTrainMaxCandidatesPerStep)
	diagnostics.ProposalReceipts = &receipts
	receipt := AOQTSidecarActualCoordinateTrainReceipt{
		Schema: AOQTSidecarActualCoordinateTrainSchema, Required: true,
		MaxSteps:            AOQTV7R5ActualCoordinateTrainMaxSteps,
		FullRowCount:        AOQTV7R5ActualCoordinateTrainFullRowCount,
		IdentityStateSHA256: identity.StateSHA256, InitialState: identity, FinalState: identity,
		CanonicalR4ReceiptPath: aoqtV7R5CanonicalR4ReceiptPath(t), CanonicalR4ReceiptSHA256: canonicalPin,
		SearchStrategy:   AOQTV7R5ActualCoordinateTrainSearchStrategy,
		SelectionVersion: AOQTV7R5ActualCoordinateTrainSelectionVersion,
	}
	return diagnostics, receipt, nil
}

func (t *AOQTSidecarTrainer) fitV7R5ActualCoordinate(set AOQTSidecarCalibrationSet, objective AOQTSidecarVectorObjective, summary AOQTSidecarTrainSummary) (AOQTSidecarTrainSummary, error) {
	diagnostics, r5, initErr := t.initializeV7R5FailureArtifacts()
	if initErr != nil {
		return summary, initErr
	}
	if !isAOQTV7R5ActualCoordinateMode(t.config.OptimizerMode) {
		return t.finishV7R5Failure(summary, diagnostics, r5, fmt.Errorf("AOQT V7-r5 actual-coordinate trainer mode is required"))
	}
	if t.config.MaxSteps != AOQTV7R5ActualCoordinateTrainMaxSteps {
		return t.finishV7R5Failure(summary, diagnostics, r5, fmt.Errorf("AOQT V7-r5 actual-coordinate trainer requires --max-steps exactly %d", AOQTV7R5ActualCoordinateTrainMaxSteps))
	}
	if objective == nil {
		return t.finishV7R5Failure(summary, diagnostics, r5, fmt.Errorf("AOQT objective is required for non-plan training"))
	}
	if err := validateAOQTFitObjectiveContract(set.Manifest.ObjectiveContract, objective); err != nil {
		return t.finishV7R5Failure(summary, diagnostics, r5, err)
	}
	if len(aoqtActiveProtectedComponentNames(set.Manifest.ObjectiveContract.WeightSums)) == 0 {
		return t.finishV7R5Failure(summary, diagnostics, r5, fmt.Errorf("AOQT protected-v2 non-plan training requires at least one active protected objective component"))
	}
	canonical := aoqtV7R5CanonicalR4Receipt(t)
	if canonical == nil {
		return t.finishV7R5Failure(summary, diagnostics, r5, fmt.Errorf("AOQT V7-r5 actual-coordinate trainer requires canonical V7-r4 receipt"))
	}
	if err := canonical.Validate(); err != nil {
		return t.finishV7R5Failure(summary, diagnostics, r5, fmt.Errorf("AOQT V7-r5 canonical r4 receipt: %w", err))
	}
	if !canonical.Passed {
		return t.finishV7R5Failure(summary, diagnostics, r5, fmt.Errorf("AOQT V7-r5 canonical r4 receipt must be passed"))
	}
	canonicalPin := aoqtV7R5CanonicalR4ReceiptSHA256(t)
	if canonicalPin != AOQTV7R5CanonicalR4ReceiptSHA256 {
		return t.finishV7R5Failure(summary, diagnostics, r5, fmt.Errorf("AOQT V7-r5 canonical r4 receipt sha256 pin mismatch"))
	}
	canonicalPath := aoqtV7R5CanonicalR4ReceiptPath(t)
	if strings.TrimSpace(canonicalPath) == "" {
		return t.finishV7R5Failure(summary, diagnostics, r5, fmt.Errorf("AOQT V7-r5 canonical r4 receipt path is required for file-hash authorization"))
	}
	canonicalData, readErr := os.ReadFile(canonicalPath)
	if readErr != nil {
		return t.finishV7R5Failure(summary, diagnostics, r5, fmt.Errorf("AOQT V7-r5 canonical r4 receipt reread: %w", readErr))
	}
	if fileSHA := sha256BytesAOQT(canonicalData); fileSHA != canonicalPin {
		return t.finishV7R5Failure(summary, diagnostics, r5, fmt.Errorf("AOQT V7-r5 canonical r4 receipt file sha256 = %s, want pinned %s", fileSHA, canonicalPin))
	}
	if canonical.Binding.SplitBinding != nil {
		if err := canonical.ValidateBound(); err != nil {
			return t.finishV7R5Failure(summary, diagnostics, r5, err)
		}
	}
	identity, err := t.aoqtV7R5StateEvidence()
	if err != nil {
		return t.finishV7R5Failure(summary, diagnostics, r5, err)
	}
	if identity.OptimizerStep != 0 || identity.AnglesSHA256 == "" || !aoqtV7R5StateIsIdentity(t.snapshotOptimizerState()) {
		return t.finishV7R5Failure(summary, diagnostics, r5, fmt.Errorf("AOQT V7-r5 trainer must start at identity state"))
	}
	fullRows := set.Rows
	if len(fullRows) != AOQTV7R5ActualCoordinateTrainFullRowCount {
		return t.finishV7R5Failure(summary, diagnostics, r5, fmt.Errorf("AOQT V7-r5 full transaction requires exactly %d rows, got %d", AOQTV7R5ActualCoordinateTrainFullRowCount, len(fullRows)))
	}
	r5.CanonicalR4ReceiptPath = canonicalPath
	r5.CanonicalR4ReceiptSHA256 = canonicalPin
	r5.IdentityStateSHA256, r5.InitialState, r5.FinalState = identity.StateSHA256, identity, identity
	r5.CanonicalR4ReceiptFileSHA256 = sha256BytesAOQT(canonicalData)
	fullLoss, _, fullActivation, fullComponents, err := t.lossAndAngleGrad(fullRows, objective)
	if err != nil {
		return t.finishV7R5Failure(summary, diagnostics, r5, err)
	}
	if err := validateAOQTActiveObjectiveContributions(set.Manifest.ObjectiveContract.WeightSums, fullActivation); err != nil {
		return t.finishV7R5Failure(summary, diagnostics, r5, err)
	}
	if !isFinite32(fullLoss) || !aoqtFiniteReceiptComponentsValue(fullComponents) {
		return t.finishV7R5Failure(summary, diagnostics, r5, fmt.Errorf("AOQT V7-r5 full baseline is non-finite"))
	}
	summary.InitialLoss, summary.InitialObjectiveActivation, summary.InitialObjectiveComponents = fullLoss, fullActivation, fullComponents
	searchForStep := func(step int) (AOQTSidecarActualCoordinateTrainSearch, error) {
		before, stateErr := t.aoqtV7R5StateEvidence()
		if stateErr != nil {
			return AOQTSidecarActualCoordinateTrainSearch{}, stateErr
		}
		search, searchErr := t.actualCoordinateSearch(set, objective)
		if searchErr != nil {
			return AOQTSidecarActualCoordinateTrainSearch{}, searchErr
		}
		searchSHA, hashErr := aoqtV7R5SearchReceiptSHA256(search.Receipt)
		if hashErr != nil {
			return AOQTSidecarActualCoordinateTrainSearch{}, hashErr
		}
		selected := aoqtV7R5SortSelectedEndpoints(search.Receipt)
		selectedOrdinals := make([]int, len(selected))
		for i := range selected {
			selectedOrdinals[i] = selected[i].Ordinal
		}
		return AOQTSidecarActualCoordinateTrainSearch{
			Step: step, StateBefore: before, Receipt: search.Receipt, ReceiptSHA256: searchSHA,
			SelectedEndpointOrdinals: selectedOrdinals,
			SelectionVersion:         AOQTV7R5ActualCoordinateTrainSelectionVersion,
			SelectionMaxCandidates:   AOQTV7R5ActualCoordinateTrainMaxCandidatesPerStep,
		}, nil
	}
	for step := 0; step < AOQTV7R5ActualCoordinateTrainMaxSteps; step++ {
		stateBefore, stateErr := t.aoqtV7R5StateEvidence()
		if stateErr != nil {
			return t.finishV7R5Failure(summary, diagnostics, r5, stateErr)
		}
		stepState := t.snapshotOptimizerState()
		search, searchErr := searchForStep(step)
		if searchErr != nil {
			return t.finishV7R5Failure(summary, diagnostics, r5, searchErr)
		}
		diagnostics.CoordinateSearchPlanCount++
		if step == 0 {
			canonicalReplay := *canonical
			canonicalReplay.Binding = AOQTSidecarActualCoordinateProbeBinding{}
			if !reflect.DeepEqual(search.Receipt, canonicalReplay) {
				return t.finishV7R5Failure(summary, diagnostics, r5, fmt.Errorf("AOQT V7-r5 step0 fresh r4 replay does not exactly match canonical receipt"))
			}
		}
		selected := aoqtV7R5SortSelectedEndpoints(search.Receipt)
		stepEvidence := AOQTSidecarActualCoordinateTrainStep{
			Step: step, StateBefore: stateBefore, Search: search,
			SelectedEndpointOrdinals: append([]int(nil), search.SelectedEndpointOrdinals...),
		}
		baseline := aoqtStepEvaluation{loss: fullLoss, activation: fullActivation, components: fullComponents}
		for index, endpoint := range selected {
			t.restoreOptimizerState(stepState)
			candidate := AOQTSidecarActualCoordinateTrainCandidate{
				Ordinal: index + 1, EndpointOrdinal: endpoint.Ordinal, SelectionRank: index,
				TotalDelta: endpoint.TotalDelta, Q3Delta: endpoint.ComponentDeltas.Q3Gain,
				Endpoint: endpoint, StateBefore: stateBefore,
				FullBaselineLoss: fullLoss, FullBaselineLossFinite: isFinite32(fullLoss),
				FullBaselineComponents:       fullComponents,
				FullBaselineComponentsFinite: aoqtFiniteReceiptComponentsValue(fullComponents),
			}
			if endpoint.CoordinateIndex < 0 || endpoint.CoordinateIndex >= len(t.angles) || len(endpoint.Directions) != 1 {
				candidate.CandidateError = "invalid_endpoint_coordinate"
				candidate.Reason = candidate.CandidateError
				t.restoreOptimizerState(stepState)
				if restored, restoreErr := t.aoqtV7R5StateEvidence(); restoreErr == nil {
					candidate.RestoredStateSHA256 = restored.StateSHA256
				}
				stepEvidence.Candidates = append(stepEvidence.Candidates, candidate)
				continue
			}
			t.angles[endpoint.CoordinateIndex] += endpoint.Magnitude * float32(endpoint.Directions[0])
			if projectErr := t.ProjectAngles(); projectErr != nil {
				candidate.CandidateError = projectErr.Error()
				candidate.Reason = "projection_error"
				t.restoreOptimizerState(stepState)
				if restored, restoreErr := t.aoqtV7R5StateEvidence(); restoreErr == nil {
					candidate.RestoredStateSHA256 = restored.StateSHA256
				}
				stepEvidence.Candidates = append(stepEvidence.Candidates, candidate)
				continue
			}
			candidate.CandidateState, err = t.aoqtV7R5StateEvidence()
			if err != nil {
				t.restoreOptimizerState(stepState)
				return t.finishV7R5Failure(summary, diagnostics, r5, err)
			}
			candidateLoss, _, candidateActivation, candidateComponents, candidateErr := t.lossAndAngleGrad(fullRows, objective)
			if candidateErr != nil {
				candidate.CandidateError = candidateErr.Error()
				candidate.Reason = "candidate_error"
				t.restoreOptimizerState(stepState)
				if restored, restoreErr := t.aoqtV7R5StateEvidence(); restoreErr == nil {
					candidate.RestoredStateSHA256 = restored.StateSHA256
				}
				stepEvidence.Candidates = append(stepEvidence.Candidates, candidate)
				continue
			}
			candidate.FullCandidateLoss, candidate.FullCandidateLossFinite = candidateLoss, isFinite32(candidateLoss)
			candidate.FullCandidateComponents, candidate.FullCandidateComponentsFinite = candidateComponents, aoqtFiniteReceiptComponentsValue(candidateComponents)
			candidate.FullCandidateActivation = candidateActivation
			candidate.FullTotalDelta, candidate.FullTotalDeltaFinite = candidateLoss-fullLoss, isFinite32(candidateLoss-fullLoss)
			candidate.FullComponentDeltas = aoqtObjectiveComponentDelta(fullComponents, candidateComponents)
			candidate.FullComponentDeltasFinite = candidate.FullBaselineComponentsFinite && candidate.FullCandidateComponentsFinite && aoqtFiniteReceiptComponentsValue(candidate.FullComponentDeltas)
			candidate.Q3GainEligible = candidate.FullComponentDeltasFinite && candidateComponents.Q3Gain < fullComponents.Q3Gain
			candidate.ProtectedEligible = candidate.FullComponentDeltasFinite && len(aoqtComponentRegressions(fullComponents, candidateComponents, t.eligibilityPolicy)) == 0
			candidateEval := aoqtStepEvaluation{loss: candidateLoss, activation: candidateActivation, components: candidateComponents}
			decision := t.evaluateTransactionalProposal(stepState, baseline, candidateEval, set.Manifest.ObjectiveContract.WeightSums)
			candidate.TransactionEligible, candidate.Accepted = decision.accepted, decision.accepted
			candidate.Reason = string(decision.reason)
			if decision.accepted {
				candidate.Reason, candidate.Committed = aoqtProposalAcceptedReason, true
			}
			diagnostics.ProposalAttempts++
			diagnostics.CoordinateProposalAttempts++
			if decision.accepted {
				diagnostics.AcceptedProposals++
				diagnostics.CoordinateAcceptedProposals++
			} else {
				diagnostics.RejectedProposals++
				diagnostics.CoordinateRejectedProposals++
				diagnostics.Backtracks++
			}
			diagnostics.RecordProposal(diagnostics.ProposalAttempts, aoqtProposalKindCoordinate, decision, baseline, candidateEval, aoqtProposalEvidence{
				magnitude: endpoint.Magnitude, blockSize: 1,
				movedAngleIndices: aoqtMovedAngleIndices(aoqtDifference(t.angles, stepState.angles)),
			})
			if decision.accepted {
				t.step = stepState.step + 1
				candidate.CandidateState, _ = t.aoqtV7R5StateEvidence()
				stepEvidence.Candidates = append(stepEvidence.Candidates, candidate)
				stepEvidence.Committed, stepEvidence.CommittedEndpointOrdinal = true, endpoint.Ordinal
				break
			}
			t.restoreOptimizerState(stepState)
			if restored, restoreErr := t.aoqtV7R5StateEvidence(); restoreErr == nil {
				candidate.RestoredStateSHA256 = restored.StateSHA256
			}
			stepEvidence.Candidates = append(stepEvidence.Candidates, candidate)
		}
		diagnostics.AttemptedSteps++
		if !stepEvidence.Committed {
			t.restoreOptimizerState(stepState)
			stepEvidence.StateAfter = stateBefore
			diagnostics.ExhaustedSteps++
			r5.StepsEvidence = append(r5.StepsEvidence, stepEvidence)
			r5.AttemptedSteps, r5.AcceptedSteps, r5.Steps = len(r5.StepsEvidence), summary.Steps, summary.Steps
			r5.FullCandidateEvaluationCount, r5.SearchCount = diagnostics.ProposalAttempts, len(r5.StepsEvidence)
			return t.finishV7R5Failure(summary, diagnostics, r5, fmt.Errorf("AOQT V7-r5 zero-safe step: no full-row candidate accepted"))
		}
		diagnostics.AcceptedSteps++
		summary.Steps++
		stepEvidence.StateAfter, err = t.aoqtV7R5StateEvidence()
		if err != nil {
			return t.finishV7R5Failure(summary, diagnostics, r5, err)
		}
		r5.StepsEvidence = append(r5.StepsEvidence, stepEvidence)
		fullLoss, _, fullActivation, fullComponents, err = t.lossAndAngleGrad(fullRows, objective)
		if err != nil {
			return t.finishV7R5Failure(summary, diagnostics, r5, err)
		}
	}
	r5.AttemptedSteps, r5.AcceptedSteps, r5.Steps = len(r5.StepsEvidence), summary.Steps, summary.Steps
	r5.FullCandidateEvaluationCount, r5.SearchCount = diagnostics.ProposalAttempts, len(r5.StepsEvidence)
	r5.FinalState, err = t.aoqtV7R5StateEvidence()
	if err != nil {
		return t.finishV7R5Failure(summary, diagnostics, r5, err)
	}
	r5.StepHashChain, r5.StepHashChainTailSHA256, err = aoqtV7R5StepHashChain(r5.StepsEvidence)
	if err != nil {
		return t.finishV7R5Failure(summary, diagnostics, r5, err)
	}
	r5.Passed = true
	summary.FinalLoss, summary.FinalObjectiveActivation, summary.FinalObjectiveComponents = fullLoss, fullActivation, fullComponents
	finalTransform := t.Transform()
	summary.AnglesSHA256, err = finalTransform.AnglesSHA256()
	if err != nil {
		return t.finishV7R5Failure(summary, diagnostics, r5, err)
	}
	summary.AngleL2, summary.AngleMaxAbs = aoqtAngleStats(t.angles)
	summary.DenseMaxAbsDelta, err = DenseInvariantMaxAbsDelta(finalTransform, set.Rows)
	if err != nil {
		return t.finishV7R5Failure(summary, diagnostics, r5, err)
	}
	summary.OptimizerDiagnostics = &diagnostics
	summary.OptimizerDiagnosticsSHA256, err = diagnostics.SHA256()
	if err != nil {
		return t.finishV7R5Failure(summary, diagnostics, r5, err)
	}
	summary.ActualCoordinateTrain = &r5
	summary.ActualCoordinateTrainSHA256, err = r5.SHA256()
	if err != nil {
		return t.finishV7R5Failure(summary, diagnostics, r5, err)
	}
	return summary, nil
}

func aoqtV7R5StateIsIdentity(state aoqtOptimizerState) bool {
	if state.step != 0 {
		return false
	}
	for _, value := range append(append(append([]float32(nil), state.angles...), state.adamM...), state.adamV...) {
		if value != 0 {
			return false
		}
	}
	return true
}

func (t *AOQTSidecarTrainer) finishV7R5Failure(summary AOQTSidecarTrainSummary, diagnostics AOQTSidecarOptimizerDiagnostics, receipt AOQTSidecarActualCoordinateTrainReceipt, failure error) (AOQTSidecarTrainSummary, error) {
	if failure == nil {
		failure = fmt.Errorf("AOQT V7-r5 training failed")
	}
	// Keep this helper safe for callers that fail before the normal r5 setup
	// block (for example an objective factory failure in the runner).  The
	// initializer supplies all structural fields and the current identity
	// witness; durable step evidence, when any, is retained from the caller.
	if receipt.Schema == "" || receipt.IdentityStateSHA256 == "" || receipt.InitialState.StateSHA256 == "" {
		if seededDiagnostics, seededReceipt, seedErr := t.initializeV7R5FailureArtifacts(); seedErr == nil {
			seededReceipt.StepsEvidence = receipt.StepsEvidence
			seededReceipt.CanonicalR4ReceiptFileSHA256 = receipt.CanonicalR4ReceiptFileSHA256
			seededReceipt.Binding = receipt.Binding
			receipt = seededReceipt
			if diagnostics.ProposalReceipts == nil {
				diagnostics = seededDiagnostics
			}
		}
	}
	if receipt.Schema == "" {
		receipt.Schema = AOQTSidecarActualCoordinateTrainSchema
	}
	receipt.Required = true
	receipt.MaxSteps = AOQTV7R5ActualCoordinateTrainMaxSteps
	receipt.FullRowCount = AOQTV7R5ActualCoordinateTrainFullRowCount
	if receipt.CanonicalR4ReceiptSHA256 != AOQTV7R5CanonicalR4ReceiptSHA256 {
		receipt.CanonicalR4ReceiptSHA256 = AOQTV7R5CanonicalR4ReceiptSHA256
	}
	receipt.SearchStrategy = AOQTV7R5ActualCoordinateTrainSearchStrategy
	receipt.SelectionVersion = AOQTV7R5ActualCoordinateTrainSelectionVersion
	receipt.FailureReason = failure.Error()
	if len(receipt.StepsEvidence) == 0 && !strings.HasPrefix(receipt.FailureReason, "precondition:") {
		receipt.FailureReason = "precondition: " + receipt.FailureReason
	}
	receipt.Passed = false
	// Counts describe durable evidence, not work that was started and then
	// abandoned by the failing operation.  Recompute all related counters from
	// the retained step/candidate evidence so receipt and diagnostics cannot
	// disagree on attempted, completed, or exhausted work.
	acceptedSteps := 0
	exhaustedSteps := 0
	acceptedProposals := 0
	rejectedProposals := 0
	for _, step := range receipt.StepsEvidence {
		if step.Committed {
			acceptedSteps++
		} else {
			exhaustedSteps++
		}
		for _, candidate := range step.Candidates {
			if candidate.CandidateError != "" {
				continue
			}
			if candidate.Accepted {
				acceptedProposals++
			} else {
				rejectedProposals++
			}
		}
	}
	summary.Steps = acceptedSteps
	receipt.AttemptedSteps = len(receipt.StepsEvidence)
	receipt.AcceptedSteps = acceptedSteps
	receipt.Steps = acceptedSteps
	receipt.FullCandidateEvaluationCount = acceptedProposals + rejectedProposals
	receipt.SearchCount = len(receipt.StepsEvidence)
	if finalState, stateErr := t.aoqtV7R5StateEvidence(); stateErr == nil {
		receipt.FinalState = finalState
	}
	receipt.StepHashChain, receipt.StepHashChainTailSHA256, _ = aoqtV7R5StepHashChain(receipt.StepsEvidence)
	// A search can fail after being started but before its step evidence is
	// durable (for example, the step-0 canonical replay guard).  Do not count
	// that abandoned search as an attempted optimizer step in the fail-closed
	// diagnostics.
	if diagnostics.ProposalReceipts == nil {
		receipts := make(AOQTSidecarOptimizerProposalReceipts, 0)
		diagnostics.ProposalReceipts = &receipts
	}
	if len(*diagnostics.ProposalReceipts) > receipt.FullCandidateEvaluationCount {
		*diagnostics.ProposalReceipts = (*diagnostics.ProposalReceipts)[:receipt.FullCandidateEvaluationCount]
	}
	// Rebuild rejection diagnostics from the retained proposal receipts. This
	// matters when a failure interrupts a step after a proposal was recorded
	// but before that step became durable: abandoned proposals were truncated
	// above and must not remain in the rejection histogram/statistics.
	var rejection AOQTSidecarOptimizerRejectionDiagnostics
	for _, proposal := range *diagnostics.ProposalReceipts {
		if proposal.Accepted {
			continue
		}
		rejection.CandidateEvaluations++
		rejection.ReasonCounts.add(aoqtTransactionalRejectionReason(proposal.Reason))
		if proposal.BaselineLossFinite && proposal.CandidateLossFinite {
			rejection.LossDelta.Record(proposal.CandidateLoss - proposal.BaselineLoss)
		}
		if proposal.BaselineComponentsFinite && proposal.CandidateComponentsFinite {
			rejection.ComponentDeltas.Record(proposal.BaselineComponents, proposal.CandidateComponents)
			for _, regression := range aoqtComponentRegressions(proposal.BaselineComponents, proposal.CandidateComponents, t.eligibilityPolicy) {
				rejection.ComponentRegressionCounts.add(regression.name)
			}
		}
	}
	rejection.finalizeDominantFields()
	diagnostics.RejectionDiagnostics = rejection
	diagnostics.PlannedSteps = AOQTV7R5ActualCoordinateTrainMaxSteps
	diagnostics.MaxAttemptsPerStep = AOQTV7R5ActualCoordinateTrainMaxCandidatesPerStep
	diagnostics.AttemptedSteps = receipt.AttemptedSteps
	diagnostics.AcceptedSteps = acceptedSteps
	diagnostics.ExhaustedSteps = exhaustedSteps
	diagnostics.ProposalAttempts = receipt.FullCandidateEvaluationCount
	diagnostics.AcceptedProposals = acceptedProposals
	diagnostics.RejectedProposals = rejectedProposals
	diagnostics.Backtracks = rejectedProposals
	diagnostics.CoordinateSearchPlanCount = receipt.SearchCount
	diagnostics.CoordinateProposalAttempts = receipt.FullCandidateEvaluationCount
	diagnostics.CoordinateAcceptedProposals = acceptedProposals
	diagnostics.CoordinateRejectedProposals = rejectedProposals
	diagnostics.OptimizerMode = t.config.OptimizerMode
	diagnostics.DevOnly = true
	diagnosticsSHA, _ := diagnostics.SHA256()
	summary.OptimizerDiagnostics = &diagnostics
	summary.OptimizerDiagnosticsSHA256 = diagnosticsSHA
	summary.ActualCoordinateTrain = &receipt
	summary.ActualCoordinateTrainSHA256, _ = receipt.SHA256()
	return summary, failure
}
