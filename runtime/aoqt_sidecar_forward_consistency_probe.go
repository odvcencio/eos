package eosruntime

// The V7-r2 probe is deliberately kept in the runtime package with the
// trainer.  It is a train-only diagnostic: it never evaluates a proxy,
// held-out split, or a candidate package and it does not alter the default
// optimizer path.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"slices"
	"sort"
	"strconv"
	"strings"
)

const AOQTSidecarForwardConsistencyProbeSchema = "eos.aoqt.v7_r2_forward_consistency_probe.v1"

// AOQTSidecarForwardConsistencyProbeBinding is the exact provenance binding
// attached by the runner.  The trainer can produce an unbound receipt for
// synthetic tests; a successful runner result requires the complete binding.
type AOQTSidecarForwardConsistencyProbeBinding struct {
	Inputs          AOQTSidecarRunMetricInputs     `json:"inputs"`
	IOReport        AOQTSidecarCalibrationIOReport `json:"io_report"`
	PreflightPath   string                         `json:"preflight_path"`
	PreflightSHA256 string                         `json:"preflight_sha256"`
	SplitBinding    *AOQTSidecarDevSplitBinding    `json:"split_binding,omitempty"`
}

type AOQTSidecarForwardConsistencyProbeSignCheck struct {
	Component                     string  `json:"component"`
	AnalyticDirectionalDerivative float32 `json:"analytic_directional_derivative"`
	AnalyticSign                  int     `json:"analytic_sign"`
	ActualDelta                   float32 `json:"actual_delta"`
	ActualSign                    int     `json:"actual_sign"`
	SignMatch                     bool    `json:"sign_match"`
}

type AOQTSidecarForwardConsistencyProbeProposal struct {
	Kind                string                                        `json:"kind"`
	Magnitude           float32                                       `json:"magnitude"`
	Indices             []int                                         `json:"indices"`
	Directions          []int                                         `json:"directions"`
	BaselineLoss        float32                                       `json:"baseline_loss"`
	CandidateLoss       float32                                       `json:"candidate_loss"`
	TotalDelta          float32                                       `json:"total_delta"`
	BaselineComponents  AOQTSidecarObjectiveComponents                `json:"baseline_components"`
	CandidateComponents AOQTSidecarObjectiveComponents                `json:"candidate_components"`
	ComponentDeltas     AOQTSidecarObjectiveComponents                `json:"component_deltas"`
	Checks              []AOQTSidecarForwardConsistencyProbeSignCheck `json:"checks"`
	Passed              bool                                          `json:"passed"`
}

type AOQTSidecarForwardConsistencyProbeReceipt struct {
	Schema              string                                       `json:"schema"`
	Required            bool                                         `json:"required"`
	Passed              bool                                         `json:"passed"`
	FailureReason       string                                       `json:"failure_reason,omitempty"`
	Magnitude           float32                                      `json:"magnitude"`
	SignPolicy          string                                       `json:"sign_policy"`
	SignNearZeroEpsilon float32                                      `json:"sign_near_zero_epsilon"`
	SubsetRowCount      int                                          `json:"subset_row_count"`
	SubsetRowIDs        []string                                     `json:"subset_row_ids"`
	SubsetRowIDSHA256   string                                       `json:"subset_row_ids_sha256"`
	SelectionSeedSHA256 string                                       `json:"selection_seed_sha256"`
	CoordinateCount     int                                          `json:"coordinate_count"`
	TopCoordinateCount  int                                          `json:"top_coordinate_count"`
	ProtectedComponents []string                                     `json:"protected_components"`
	BlockSizes          []int                                        `json:"block_sizes"`
	CoordinateProposals []AOQTSidecarForwardConsistencyProbeProposal `json:"coordinate_proposals"`
	BlockProposals      []AOQTSidecarForwardConsistencyProbeProposal `json:"block_proposals"`
	Binding             AOQTSidecarForwardConsistencyProbeBinding    `json:"binding"`
	QualityClaim        bool                                         `json:"quality_claim"`
	ReleaseClaim        bool                                         `json:"release_claim"`
	OfficialClaim       bool                                         `json:"official_claim"`
	OfficialHeldoutGate bool                                         `json:"official_heldout_gate"`
}

// SHA256 returns the canonical receipt digest used by the summary and metrics
// artifacts.  encoding/json sorts map keys, so the result is stable across
// repeated runs with the same inputs.
func (r AOQTSidecarForwardConsistencyProbeReceipt) SHA256() (string, error) {
	data, err := json.Marshal(r)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func (r AOQTSidecarForwardConsistencyProbeReceipt) Validate() error {
	return r.validate(false)
}

func (r AOQTSidecarForwardConsistencyProbeReceipt) ValidateBound() error {
	if err := r.validate(true); err != nil {
		return err
	}
	if err := validateAOQTMetricInputs(r.Binding.Inputs); err != nil {
		return fmt.Errorf("AOQT forward-consistency probe binding inputs: %w", err)
	}
	if r.Binding.IOReport.Schema != AOQTSidecarCalibrationIOReportSchema {
		return fmt.Errorf("AOQT forward-consistency probe binding io_report schema %q is unsupported", r.Binding.IOReport.Schema)
	}
	if strings.TrimSpace(r.Binding.IOReport.ManifestPath) == "" || strings.TrimSpace(r.Binding.IOReport.RowsJSONLPath) == "" {
		return fmt.Errorf("AOQT forward-consistency probe binding io_report paths are required")
	}
	if r.Binding.IOReport.ManifestSHA256 != r.Binding.Inputs.DatasetManifestSHA256 {
		return fmt.Errorf("AOQT forward-consistency probe binding io_report manifest sha256 must match inputs")
	}
	if err := validateAOQTSHA256(r.Binding.IOReport.RowsSHA256, "AOQT forward-consistency probe binding io_report rows sha256"); err != nil {
		return err
	}
	if r.Binding.IOReport.AnchorArtifactSHA256 != r.Binding.Inputs.AnchorArtifactSHA256 ||
		r.Binding.IOReport.AnchorPackageManifestSHA256 != r.Binding.Inputs.AnchorPackageManifestSHA256 ||
		r.Binding.IOReport.AnchorEmbeddingSpaceID != r.Binding.Inputs.AnchorEmbeddingSpaceID ||
		r.Binding.IOReport.CompatibilityDigest != r.Binding.Inputs.CompatibilityDigest {
		return fmt.Errorf("AOQT forward-consistency probe binding io_report provenance must match inputs")
	}
	if err := validateAOQTSHA256(r.Binding.PreflightSHA256, "AOQT forward-consistency probe preflight sha256"); err != nil {
		return err
	}
	if strings.TrimSpace(r.Binding.PreflightPath) == "" {
		return fmt.Errorf("AOQT forward-consistency probe preflight path is required")
	}
	if r.Binding.SplitBinding == nil {
		return fmt.Errorf("AOQT forward-consistency probe split binding is required")
	}
	if err := r.Binding.SplitBinding.Validate(); err != nil {
		return fmt.Errorf("AOQT forward-consistency probe split binding: %w", err)
	}
	if r.Binding.SplitBinding.MaterializedManifestSHA256 != r.Binding.Inputs.DatasetManifestSHA256 {
		return fmt.Errorf("AOQT forward-consistency probe split binding manifest sha256 must match inputs")
	}
	if r.Binding.SplitBinding.MaterializedRowsSHA256 != r.Binding.IOReport.RowsSHA256 || r.Binding.SplitBinding.MaterializedRowCount != r.Binding.IOReport.RowCount {
		return fmt.Errorf("AOQT forward-consistency probe split binding rows/count must match IO report")
	}
	if r.Binding.SplitBinding.MaterializedPreflightSHA256 != r.Binding.PreflightSHA256 {
		return fmt.Errorf("AOQT forward-consistency probe split binding preflight sha256 must match binding")
	}
	if r.Binding.SplitBinding.MaterializedRowIDSHA256 == "" {
		return fmt.Errorf("AOQT forward-consistency probe split binding materialized row ids sha256 is required")
	}
	if err := validateAOQTSHA256(r.Binding.SplitBinding.MaterializedRowIDSHA256, "AOQT forward-consistency probe materialized row ids sha256"); err != nil {
		return err
	}
	if r.Binding.IOReport.TurboQuantSeed == 0 {
		return fmt.Errorf("AOQT forward-consistency probe binding io_report turboquant seed is required")
	}
	expectedSeed := sha256AOQTForwardConsistencySelectionSeed(
		r.Binding.IOReport.TurboQuantSeed,
		r.Binding.SplitBinding.FoldID,
		r.Binding.SplitBinding.SplitManifestSHA256,
		r.Binding.Inputs.DatasetManifestSHA256,
		r.Binding.SplitBinding.MaterializedRowIDSHA256,
	)
	if r.SelectionSeedSHA256 != expectedSeed {
		return fmt.Errorf("AOQT forward-consistency probe selection seed does not match bound inputs")
	}
	if err := rereadAOQTForwardConsistencyProbeBinding(r); err != nil {
		return err
	}
	active := sortedAOQTProtectedComponentNames(aoqtActiveProtectedComponentNames(r.Binding.IOReport.ObjectiveContract.WeightSums))
	if !slices.Equal(r.ProtectedComponents, active) {
		return fmt.Errorf("AOQT forward-consistency probe protected components %v do not match active objective contract %v", r.ProtectedComponents, active)
	}
	return nil
}

func probeReceiptHasEarlyFailure(r AOQTSidecarForwardConsistencyProbeReceipt) bool {
	return !r.Passed && (strings.HasPrefix(r.FailureReason, "precondition:") || strings.HasPrefix(r.FailureReason, "probe_error:"))
}

// rereadAOQTForwardConsistencyProbeBinding closes the TOCTOU gap between the
// runner's initial input load and receipt serialization. The receipt is only
// authentic while every bound source file still hashes to the values used by
// the probe.
func rereadAOQTForwardConsistencyProbeBinding(r AOQTSidecarForwardConsistencyProbeReceipt) error {
	binding := r.Binding
	manifestData, err := os.ReadFile(binding.IOReport.ManifestPath)
	if err != nil {
		return fmt.Errorf("AOQT forward-consistency probe bound manifest reread: %w", err)
	}
	var manifest AOQTSidecarCalibrationManifest
	if err := strictUnmarshalAOQT(manifestData, &manifest); err != nil {
		return fmt.Errorf("AOQT forward-consistency probe bound manifest is invalid: %w", err)
	}
	manifestSHA, err := AOQTSidecarManifestSHA256(manifest)
	if err != nil {
		return fmt.Errorf("AOQT forward-consistency probe bound manifest hash: %w", err)
	}
	if manifestSHA != binding.Inputs.DatasetManifestSHA256 || manifestSHA != binding.IOReport.ManifestSHA256 {
		return fmt.Errorf("AOQT forward-consistency probe bound manifest sha256 mismatch")
	}
	if err := manifest.Validate(); err != nil {
		return fmt.Errorf("AOQT forward-consistency probe bound manifest validation: %w", err)
	}
	if !aoqtObjectiveContractsEqual(manifest.ObjectiveContract, binding.IOReport.ObjectiveContract) {
		return fmt.Errorf("AOQT forward-consistency probe bound manifest objective contract mismatch")
	}

	rowsData, err := os.ReadFile(binding.IOReport.RowsJSONLPath)
	if err != nil {
		return fmt.Errorf("AOQT forward-consistency probe bound rows reread: %w", err)
	}
	if rowsSHA := sha256BytesAOQT(rowsData); rowsSHA != binding.IOReport.RowsSHA256 {
		return fmt.Errorf("AOQT forward-consistency probe bound rows sha256 mismatch")
	}
	rows, err := parseAOQTCalibrationRowsJSONL(rowsData, binding.IOReport.RowsJSONLPath)
	if err != nil {
		return fmt.Errorf("AOQT forward-consistency probe bound rows are invalid: %w", err)
	}
	if len(rows) != binding.IOReport.RowCount || len(rows) != manifest.RowCount {
		return fmt.Errorf("AOQT forward-consistency probe bound row count mismatch")
	}
	set := AOQTSidecarCalibrationSet{Manifest: manifest, Rows: rows}
	if err := set.Validate(); err != nil {
		return fmt.Errorf("AOQT forward-consistency probe bound calibration set validation: %w", err)
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
		return fmt.Errorf("AOQT forward-consistency probe bound IO report changed")
	}
	rowIDs := make([]string, 0, len(rows))
	for _, row := range rows {
		rowIDs = append(rowIDs, row.RowID)
	}
	if got := aoqtRowIDSHA256(rowIDs); got != binding.SplitBinding.MaterializedRowIDSHA256 || got != manifest.RowIDSHA256 {
		return fmt.Errorf("AOQT forward-consistency probe bound row-id hash mismatch")
	}
	if got, err := sha256JSONAOQT(rowIDs); err != nil || got != binding.SplitBinding.TrainRowIDSHA256 {
		if err != nil {
			return fmt.Errorf("AOQT forward-consistency probe bound train row-id hash: %w", err)
		}
		return fmt.Errorf("AOQT forward-consistency probe bound train row-id hash mismatch")
	}
	rowSet := make(map[string]struct{}, len(rowIDs))
	for _, id := range rowIDs {
		rowSet[id] = struct{}{}
	}
	if !probeReceiptHasEarlyFailure(r) {
		for _, id := range r.SubsetRowIDs {
			if _, ok := rowSet[id]; !ok {
				return fmt.Errorf("AOQT forward-consistency probe selected row id %q is outside bound row-id set", id)
			}
		}
		expectedRows := selectAOQTForwardConsistencyRows(rows, r.SelectionSeedSHA256)
		expectedIDs := make([]string, len(expectedRows))
		for i, row := range expectedRows {
			expectedIDs[i] = row.RowID
		}
		if !slices.Equal(r.SubsetRowIDs, expectedIDs) {
			return fmt.Errorf("AOQT forward-consistency probe selected row ids do not match the bound deterministic subset")
		}
	}

	preflightData, err := os.ReadFile(binding.PreflightPath)
	if err != nil {
		return fmt.Errorf("AOQT forward-consistency probe bound preflight reread: %w", err)
	}
	if got := sha256BytesAOQT(preflightData); got != binding.PreflightSHA256 {
		return fmt.Errorf("AOQT forward-consistency probe bound preflight sha256 mismatch")
	}
	var preflight AOQTSidecarMaterializePreflight
	if err := strictUnmarshalAOQT(preflightData, &preflight); err != nil {
		return fmt.Errorf("AOQT forward-consistency probe bound preflight is invalid: %w", err)
	}
	if err := validateAOQTFailClosedPreflight(preflight, binding.PreflightPath, binding.IOReport, binding.Inputs, binding.IOReport.Topology, binding.IOReport.ObjectiveContract, binding.IOReport.LegalGates); err != nil {
		return fmt.Errorf("AOQT forward-consistency probe bound preflight validation: %w", err)
	}

	splitData, err := os.ReadFile(binding.SplitBinding.SplitManifestPath)
	if err != nil {
		return fmt.Errorf("AOQT forward-consistency probe bound split manifest reread: %w", err)
	}
	if got := sha256BytesAOQT(splitData); got != binding.SplitBinding.SplitManifestFileSHA256 || got != binding.SplitBinding.SplitManifestSHA256 {
		return fmt.Errorf("AOQT forward-consistency probe bound split manifest sha256 mismatch")
	}
	actualBinding, err := validateAOQTV7DevSplitBinding(AOQTSidecarTrainRunnerConfig{
		SplitManifestPath:           binding.SplitBinding.SplitManifestPath,
		ExpectedSplitManifestSHA256: binding.SplitBinding.SplitManifestFileSHA256,
		FoldID:                      binding.SplitBinding.FoldID,
	}, set, binding.IOReport, preflight, binding.PreflightSHA256)
	if err != nil {
		return fmt.Errorf("AOQT forward-consistency probe bound split manifest validation: %w", err)
	}
	if actualBinding == nil || !reflect.DeepEqual(actualBinding, binding.SplitBinding) {
		return fmt.Errorf("AOQT forward-consistency probe bound split manifest binding changed")
	}
	return nil
}

func (r AOQTSidecarForwardConsistencyProbeReceipt) validate(requireBinding bool) error {
	if r.Schema != AOQTSidecarForwardConsistencyProbeSchema {
		return fmt.Errorf("AOQT forward-consistency probe schema %q is unsupported, want %q", r.Schema, AOQTSidecarForwardConsistencyProbeSchema)
	}
	if !r.Required {
		return fmt.Errorf("AOQT forward-consistency probe receipt must be required")
	}
	if !isFinite32(r.Magnitude) || r.Magnitude != AOQTV7R2ForwardConsistencyProbeMagnitude {
		return fmt.Errorf("AOQT forward-consistency probe magnitude = %.9g, want %.9g", r.Magnitude, AOQTV7R2ForwardConsistencyProbeMagnitude)
	}
	if r.SignPolicy != AOQTV7R2ForwardConsistencyProbeSignPolicy {
		return fmt.Errorf("AOQT forward-consistency probe sign policy %q is unsupported, want %q", r.SignPolicy, AOQTV7R2ForwardConsistencyProbeSignPolicy)
	}
	if !isFinite32(r.SignNearZeroEpsilon) || r.SignNearZeroEpsilon != AOQTV7R2ForwardConsistencyProbeSignNearZeroEpsilon {
		return fmt.Errorf("AOQT forward-consistency probe sign near-zero epsilon = %.9g, want exact %.9g", r.SignNearZeroEpsilon, AOQTV7R2ForwardConsistencyProbeSignNearZeroEpsilon)
	}
	if probeReceiptHasEarlyFailure(r) {
		// A failed precondition (for example, a materialized fixture with fewer
		// than eight train rows) still gets a bound, fail-closed receipt. There
		// is no honest subset or sign table to validate in that case, but the
		// active protected-component contract remains bound below.
		if err := validateAOQTForwardConsistencyProbeProtectedComponents(r.ProtectedComponents, "AOQT forward-consistency probe"); err != nil {
			return err
		}
		if r.QualityClaim || r.ReleaseClaim || r.OfficialClaim || r.OfficialHeldoutGate {
			return fmt.Errorf("AOQT forward-consistency probe claims and official heldout gate must be false")
		}
		return nil
	}
	if r.SubsetRowCount != AOQTV7R2ForwardConsistencyProbeRowCount {
		return fmt.Errorf("AOQT forward-consistency probe subset row count = %d, want %d", r.SubsetRowCount, AOQTV7R2ForwardConsistencyProbeRowCount)
	}
	if len(r.SubsetRowIDs) != r.SubsetRowCount || !slices.IsSorted(r.SubsetRowIDs) {
		return fmt.Errorf("AOQT forward-consistency probe subset row ids must be sorted and contain exactly %d rows", r.SubsetRowCount)
	}
	for i, id := range r.SubsetRowIDs {
		if strings.TrimSpace(id) == "" || (i > 0 && id == r.SubsetRowIDs[i-1]) {
			return fmt.Errorf("AOQT forward-consistency probe subset row ids must be non-empty and unique")
		}
	}
	if r.SubsetRowIDSHA256 != aoqtRowIDSHA256(r.SubsetRowIDs) {
		return fmt.Errorf("AOQT forward-consistency probe subset row ids sha256 does not bind row ids")
	}
	if err := validateAOQTSHA256(r.SelectionSeedSHA256, "AOQT forward-consistency probe selection seed sha256"); err != nil {
		return err
	}
	if r.CoordinateCount != AOQTV7R2ForwardConsistencyProbeCoordinateCount {
		return fmt.Errorf("AOQT forward-consistency probe coordinate count = %d, want exact D384 count %d", r.CoordinateCount, AOQTV7R2ForwardConsistencyProbeCoordinateCount)
	}
	if r.TopCoordinateCount != AOQTV7R2ForwardConsistencyProbeTopAngles {
		return fmt.Errorf("AOQT forward-consistency probe top coordinate count = %d, want %d", r.TopCoordinateCount, AOQTV7R2ForwardConsistencyProbeTopAngles)
	}
	wantBlocks := []int{2, 4, 8}
	if !slices.Equal(r.BlockSizes, wantBlocks) {
		return fmt.Errorf("AOQT forward-consistency probe block sizes = %v, want %v", r.BlockSizes, wantBlocks)
	}
	if len(r.CoordinateProposals) != r.TopCoordinateCount || len(r.BlockProposals) != len(r.BlockSizes) {
		return fmt.Errorf("AOQT forward-consistency probe proposal counts = %d/%d, want %d/%d", len(r.CoordinateProposals), len(r.BlockProposals), r.TopCoordinateCount, len(r.BlockSizes))
	}
	if r.QualityClaim || r.ReleaseClaim || r.OfficialClaim || r.OfficialHeldoutGate {
		return fmt.Errorf("AOQT forward-consistency probe claims and official heldout gate must be false")
	}
	if err := validateAOQTForwardConsistencyProbeProtectedComponents(r.ProtectedComponents, "AOQT forward-consistency probe"); err != nil {
		return err
	}
	expectedComponents := append([]string{"q3_gain"}, r.ProtectedComponents...)
	seenCoordinates := make(map[int]struct{}, len(r.CoordinateProposals))
	for i, proposal := range r.CoordinateProposals {
		if err := validateAOQTForwardConsistencyProposal(proposal, "coordinate", 1, i, r.CoordinateCount, r.Magnitude, expectedComponents); err != nil {
			return err
		}
		index := proposal.Indices[0]
		if _, ok := seenCoordinates[index]; ok {
			return fmt.Errorf("AOQT forward-consistency probe repeats top coordinate index %d", index)
		}
		seenCoordinates[index] = struct{}{}
	}
	for i, proposal := range r.BlockProposals {
		if err := validateAOQTForwardConsistencyProposal(proposal, "block", r.BlockSizes[i], i, r.CoordinateCount, r.Magnitude, expectedComponents); err != nil {
			return err
		}
	}
	for blockIndex, blockSize := range r.BlockSizes {
		block := r.BlockProposals[blockIndex]
		for index := 0; index < blockSize; index++ {
			if block.Indices[index] != r.CoordinateProposals[index].Indices[0] || block.Directions[index] != r.CoordinateProposals[index].Directions[0] {
				return fmt.Errorf("AOQT forward-consistency block proposal[%d] does not bind top-coordinate prefix", blockIndex)
			}
		}
	}
	if r.Passed {
		if strings.TrimSpace(r.FailureReason) != "" {
			return fmt.Errorf("AOQT forward-consistency probe passed receipt must not have failure_reason")
		}
		for _, proposal := range append(append([]AOQTSidecarForwardConsistencyProbeProposal(nil), r.CoordinateProposals...), r.BlockProposals...) {
			if !proposal.Passed {
				return fmt.Errorf("AOQT forward-consistency probe passed receipt contains a failed proposal")
			}
		}
	} else if strings.TrimSpace(r.FailureReason) == "" {
		return fmt.Errorf("AOQT forward-consistency probe failed receipt requires failure_reason")
	}
	if requireBinding {
		// The detailed binding checks live in ValidateBound so the unbound
		// synthetic trainer receipt remains useful in package-local tests.
		return nil
	}
	return nil
}

func validateAOQTForwardConsistencyProbeProtectedComponents(components []string, label string) error {
	if len(components) == 0 || !slices.Equal(components, sortedAOQTProtectedComponentNames(components)) {
		return fmt.Errorf("%s protected components must be non-empty and canonical", label)
	}
	for _, component := range components {
		if aoqtProtectedComponentOrder(component) >= 100 {
			return fmt.Errorf("%s protected component %q is unsupported", label, component)
		}
	}
	return nil
}

func validateAOQTForwardConsistencyProposal(proposal AOQTSidecarForwardConsistencyProbeProposal, wantKind string, wantCount, ordinal, coordinateCount int, magnitude float32, expectedComponents []string) error {
	if proposal.Kind != wantKind {
		return fmt.Errorf("AOQT forward-consistency %s proposal[%d] kind = %q", wantKind, ordinal, proposal.Kind)
	}
	if !isFinite32(proposal.Magnitude) || proposal.Magnitude != magnitude {
		return fmt.Errorf("AOQT forward-consistency %s proposal[%d] magnitude is not fixed", wantKind, ordinal)
	}
	if len(proposal.Indices) != wantCount || len(proposal.Directions) != wantCount {
		return fmt.Errorf("AOQT forward-consistency %s proposal[%d] indices/directions = %d/%d, want %d", wantKind, ordinal, len(proposal.Indices), len(proposal.Directions), wantCount)
	}
	seen := make(map[int]struct{}, len(proposal.Indices))
	for i, index := range proposal.Indices {
		if index < 0 || index >= coordinateCount {
			return fmt.Errorf("AOQT forward-consistency %s proposal[%d] index %d outside angle count %d", wantKind, ordinal, index, coordinateCount)
		}
		if _, ok := seen[index]; ok {
			return fmt.Errorf("AOQT forward-consistency %s proposal[%d] repeats index %d", wantKind, ordinal, index)
		}
		seen[index] = struct{}{}
		if proposal.Directions[i] != -1 && proposal.Directions[i] != 1 {
			return fmt.Errorf("AOQT forward-consistency %s proposal[%d] direction[%d] must be -1 or 1", wantKind, ordinal, i)
		}
	}
	for _, value := range []struct {
		name  string
		value float32
	}{
		{"baseline_loss", proposal.BaselineLoss},
		{"candidate_loss", proposal.CandidateLoss},
		{"total_delta", proposal.TotalDelta},
		{"baseline_q3_gain", proposal.BaselineComponents.Q3Gain},
		{"candidate_q3_gain", proposal.CandidateComponents.Q3Gain},
		{"delta_q3_gain", proposal.ComponentDeltas.Q3Gain},
	} {
		if !isFinite32(value.value) {
			return fmt.Errorf("AOQT forward-consistency %s proposal[%d] %s must be finite", wantKind, ordinal, value.name)
		}
	}
	if proposal.TotalDelta != proposal.CandidateLoss-proposal.BaselineLoss {
		return fmt.Errorf("AOQT forward-consistency %s proposal[%d] total delta mismatch", wantKind, ordinal)
	}
	if proposal.ComponentDeltas != aoqtObjectiveComponentDelta(proposal.BaselineComponents, proposal.CandidateComponents) {
		return fmt.Errorf("AOQT forward-consistency %s proposal[%d] component delta mismatch", wantKind, ordinal)
	}
	if err := validateAOQTObjectiveComponents(proposal.BaselineComponents, fmt.Sprintf("AOQT forward-consistency %s proposal[%d] baseline components", wantKind, ordinal)); err != nil {
		return err
	}
	if err := validateAOQTObjectiveComponents(proposal.CandidateComponents, fmt.Sprintf("AOQT forward-consistency %s proposal[%d] candidate components", wantKind, ordinal)); err != nil {
		return err
	}
	if err := validateAOQTObjectiveComponents(proposal.ComponentDeltas, fmt.Sprintf("AOQT forward-consistency %s proposal[%d] component deltas", wantKind, ordinal)); err != nil {
		// Deltas are allowed to be negative; only finiteness is required here.
		for _, value := range []float32{proposal.ComponentDeltas.Q3Gain, proposal.ComponentDeltas.Q3OrderGuard, proposal.ComponentDeltas.Q3ScoreDistill, proposal.ComponentDeltas.Q5OrderGuard, proposal.ComponentDeltas.Q5ScoreDistill, proposal.ComponentDeltas.NFBoundaryGuard} {
			if !isFinite32(value) {
				return err
			}
		}
	}
	if !slices.Equal(probeCheckNames(proposal.Checks), expectedComponents) {
		return fmt.Errorf("AOQT forward-consistency %s proposal[%d] checks do not bind q3/protected components", wantKind, ordinal)
	}
	allMatch := true
	for checkIndex, check := range proposal.Checks {
		if strings.TrimSpace(check.Component) == "" || !isFinite32(check.AnalyticDirectionalDerivative) || !isFinite32(check.ActualDelta) {
			return fmt.Errorf("AOQT forward-consistency %s proposal[%d] check[%d] is incomplete", wantKind, ordinal, checkIndex)
		}
		if (check.AnalyticSign < -1 || check.AnalyticSign > 1) || (check.ActualSign < -1 || check.ActualSign > 1) {
			return fmt.Errorf("AOQT forward-consistency %s proposal[%d] check[%d] sign must be -1, 0, or 1", wantKind, ordinal, checkIndex)
		}
		if check.AnalyticSign != signAOQTFloat32(check.AnalyticDirectionalDerivative) || check.ActualSign != signAOQTFloat32(check.ActualDelta) {
			return fmt.Errorf("AOQT forward-consistency %s proposal[%d] check[%d] sign does not bind value", wantKind, ordinal, checkIndex)
		}
		if check.SignMatch != (check.AnalyticSign == check.ActualSign) {
			return fmt.Errorf("AOQT forward-consistency %s proposal[%d] check[%d] sign_match is inconsistent", wantKind, ordinal, checkIndex)
		}
		allMatch = allMatch && check.SignMatch
	}
	if proposal.Passed != allMatch {
		return fmt.Errorf("AOQT forward-consistency %s proposal[%d] passed does not bind checks", wantKind, ordinal)
	}
	return nil
}

func probeCheckNames(checks []AOQTSidecarForwardConsistencyProbeSignCheck) []string {
	names := make([]string, len(checks))
	for i, check := range checks {
		names[i] = check.Component
	}
	return names
}

func sortedAOQTProtectedComponentNames(names []string) []string {
	ordered := append([]string(nil), names...)
	sort.SliceStable(ordered, func(i, j int) bool {
		return aoqtProtectedComponentOrder(ordered[i]) < aoqtProtectedComponentOrder(ordered[j])
	})
	return ordered
}

func aoqtProtectedComponentOrder(name string) int {
	for i, candidate := range []string{"q3_order_guard", "q3_score_distill", "q5_order_guard", "q5_score_distill", "nf_boundary_guard"} {
		if name == candidate {
			return i
		}
	}
	return 100
}

func signAOQTFloat32(value float32) int {
	if value < 0 {
		return -1
	}
	if value > 0 {
		return 1
	}
	return 0
}

func (t *AOQTSidecarTrainer) runForwardConsistencyProbe(set AOQTSidecarCalibrationSet, objective AOQTSidecarVectorObjective) (AOQTSidecarForwardConsistencyProbeReceipt, error) {
	receipt := AOQTSidecarForwardConsistencyProbeReceipt{
		Schema:              AOQTSidecarForwardConsistencyProbeSchema,
		Required:            true,
		Magnitude:           AOQTV7R2ForwardConsistencyProbeMagnitude,
		SignPolicy:          AOQTV7R2ForwardConsistencyProbeSignPolicy,
		SignNearZeroEpsilon: AOQTV7R2ForwardConsistencyProbeSignNearZeroEpsilon,
		BlockSizes:          []int{2, 4, 8},
		QualityClaim:        false,
		ReleaseClaim:        false,
		OfficialClaim:       false,
		OfficialHeldoutGate: false,
	}
	if err := set.Validate(); err != nil {
		return receipt, err
	}
	if objective == nil {
		return receipt, fmt.Errorf("AOQT forward-consistency probe objective is required")
	}
	if t.config.OptimizerMode != AOQTSidecarOptimizerModeV7R2DevTrustRegion {
		return receipt, fmt.Errorf("AOQT forward-consistency probe requires optimizer_mode=%s", AOQTSidecarOptimizerModeV7R2DevTrustRegion)
	}
	if t.config.LearningRate != float32(0.01) {
		return receipt, fmt.Errorf("AOQT forward-consistency probe requires learning_rate exactly 0.01")
	}
	if err := validateAOQTFitObjectiveContract(set.Manifest.ObjectiveContract, objective); err != nil {
		return receipt, err
	}
	receipt.ProtectedComponents = sortedAOQTProtectedComponentNames(aoqtActiveProtectedComponentNames(set.Manifest.ObjectiveContract.WeightSums))
	manifestSHA, err := AOQTSidecarManifestSHA256(set.Manifest)
	if err != nil {
		return receipt, err
	}
	receipt.Binding.Inputs = AOQTSidecarRunMetricInputs{
		AnchorArtifactSHA256:        set.Manifest.AnchorArtifactSHA256,
		AnchorPackageManifestSHA256: set.Manifest.AnchorPackageManifestSHA256,
		AnchorEmbeddingSpaceID:      set.Manifest.AnchorEmbeddingSpaceID,
		DatasetManifestSHA256:       manifestSHA,
		QrelsSHA256ByDataset:        cloneAOQTStringMap(set.Manifest.QrelsSHA256ByDataset),
		CompatibilityDigest:         set.Manifest.CompatibilityDigest,
	}
	selectionSeed := sha256AOQTForwardConsistencySelectionSeed(t.config.WorkplanSeed, t.config.ForwardConsistencyProbeFoldID, t.config.ForwardConsistencyProbeSplitManifestSHA256, manifestSHA, set.Manifest.RowIDSHA256)
	receipt.SelectionSeedSHA256 = selectionSeed
	if len(set.Rows) < AOQTV7R2ForwardConsistencyProbeRowCount {
		receipt.FailureReason = fmt.Sprintf("precondition: AOQT forward-consistency probe requires at least %d train rows, got %d", AOQTV7R2ForwardConsistencyProbeRowCount, len(set.Rows))
		return receipt, fmt.Errorf("AOQT forward-consistency probe requires at least %d train rows, got %d", AOQTV7R2ForwardConsistencyProbeRowCount, len(set.Rows))
	}
	rows := selectAOQTForwardConsistencyRows(set.Rows, selectionSeed)
	receipt.SubsetRowCount = len(rows)
	receipt.SubsetRowIDs = make([]string, len(rows))
	for i, row := range rows {
		receipt.SubsetRowIDs[i] = row.RowID
	}
	receipt.SubsetRowIDSHA256 = aoqtRowIDSHA256(receipt.SubsetRowIDs)
	receipt.CoordinateCount = len(t.angles)
	receipt.TopCoordinateCount = AOQTV7R2ForwardConsistencyProbeTopAngles

	q3Gradient, err := t.q3GainOnlyAngleGrad(rows, objective)
	if err != nil {
		return receipt, err
	}
	protectedGradients, err := t.protectedComponentAngleGrads(rows, objective, set.Manifest.ObjectiveContract.WeightSums)
	if err != nil {
		return receipt, err
	}
	_, fullGradient, _, _, err := t.lossAndAngleGrad(rows, objective)
	if err != nil {
		return receipt, err
	}
	plan, err := newAOQTCoordinateSearchPlanForMode(q3Gradient, fullGradient, t.config.LearningRate, t.config.AngleCap, t.config.OptimizerMode, protectedGradients)
	if err != nil {
		return receipt, err
	}
	if len(plan.Order) < AOQTV7R2ForwardConsistencyProbeTopAngles {
		return receipt, fmt.Errorf("AOQT forward-consistency probe coordinate order has %d entries, want at least %d", len(plan.Order), AOQTV7R2ForwardConsistencyProbeTopAngles)
	}
	plan.Order = plan.Order[:AOQTV7R2ForwardConsistencyProbeTopAngles]
	receipt.ProtectedComponents = append([]string(nil), plan.ProtectedComponents...)
	state := t.snapshotOptimizerState()
	defer t.restoreOptimizerState(state)
	baselineLoss, _, _, baselineComponents, err := t.lossAndAngleGrad(rows, objective)
	if err != nil {
		return receipt, err
	}
	for ordinal, rank := range plan.Order {
		proposal, proposalErr := t.forwardConsistencyProposal(rows, objective, baselineLoss, baselineComponents, "coordinate", []aoqtCoordinateSearchRank{rank}, q3Gradient, protectedGradients)
		receipt.CoordinateProposals = append(receipt.CoordinateProposals, proposal)
		if proposalErr != nil {
			return receipt, proposalErr
		}
		if !proposal.Passed && receipt.FailureReason == "" {
			receipt.FailureReason = fmt.Sprintf("coordinate proposal %d sign disagreement", ordinal)
		}
	}
	for _, blockSize := range receipt.BlockSizes {
		block := append([]aoqtCoordinateSearchRank(nil), plan.Order[:blockSize]...)
		proposal, proposalErr := t.forwardConsistencyProposal(rows, objective, baselineLoss, baselineComponents, "block", block, q3Gradient, protectedGradients)
		receipt.BlockProposals = append(receipt.BlockProposals, proposal)
		if proposalErr != nil {
			return receipt, proposalErr
		}
		if !proposal.Passed && receipt.FailureReason == "" {
			receipt.FailureReason = fmt.Sprintf("block size %d sign disagreement", blockSize)
		}
	}
	if receipt.FailureReason != "" {
		return receipt, fmt.Errorf("AOQT V7-r2 forward-consistency probe failed closed: %s", receipt.FailureReason)
	}
	receipt.Passed = true
	return receipt, nil
}

func (t *AOQTSidecarTrainer) forwardConsistencyProposal(rows []AOQTSidecarCalibrationRow, objective AOQTSidecarVectorObjective, baselineLoss float32, baselineComponents AOQTSidecarObjectiveComponents, kind string, ranks []aoqtCoordinateSearchRank, q3Gradient []float32, protectedGradients []aoqtProtectedAngleGradient) (AOQTSidecarForwardConsistencyProbeProposal, error) {
	proposal := AOQTSidecarForwardConsistencyProbeProposal{
		Kind:               kind,
		Magnitude:          AOQTV7R2ForwardConsistencyProbeMagnitude,
		Indices:            make([]int, len(ranks)),
		Directions:         make([]int, len(ranks)),
		BaselineLoss:       baselineLoss,
		BaselineComponents: baselineComponents,
	}
	direction := make([]float32, len(t.angles))
	for i, rank := range ranks {
		proposal.Indices[i] = rank.index
		proposal.Directions[i] = int(rank.primaryDirection)
		direction[rank.index] = rank.primaryDirection * AOQTV7R2ForwardConsistencyProbeMagnitude
	}
	state := t.snapshotOptimizerState()
	defer t.restoreOptimizerState(state)
	for i, index := range proposal.Indices {
		if index < 0 || index >= len(t.angles) || proposal.Directions[i] == 0 {
			return proposal, fmt.Errorf("AOQT forward-consistency probe proposal has an invalid top-coordinate direction")
		}
	}
	for i, delta := range direction {
		if delta != 0 {
			t.angles[i] += delta
		}
	}
	if err := t.ProjectAngles(); err != nil {
		return proposal, err
	}
	candidateLoss, _, _, candidateComponents, err := t.lossAndAngleGrad(rows, objective)
	if err != nil {
		return proposal, err
	}
	proposal.CandidateLoss = candidateLoss
	proposal.CandidateComponents = candidateComponents
	proposal.TotalDelta = candidateLoss - baselineLoss
	proposal.ComponentDeltas = aoqtObjectiveComponentDelta(baselineComponents, candidateComponents)
	checks := make([]AOQTSidecarForwardConsistencyProbeSignCheck, 0, 1+len(protectedGradients))
	checks = append(checks, AOQTSidecarForwardConsistencyProbeSignCheck{
		Component:                     "q3_gain",
		AnalyticDirectionalDerivative: dotAOQTFloat32(q3Gradient, direction),
		ActualDelta:                   proposal.ComponentDeltas.Q3Gain,
	})
	for _, protected := range protectedGradients {
		checks = append(checks, AOQTSidecarForwardConsistencyProbeSignCheck{
			Component:                     protected.Name,
			AnalyticDirectionalDerivative: dotAOQTFloat32(protected.Grad, direction),
			ActualDelta:                   aoqtNamedComponentValue(proposal.ComponentDeltas, protected.Name),
		})
	}
	proposal.Checks = checks
	proposal.Passed = true
	for i := range proposal.Checks {
		proposal.Checks[i].AnalyticSign = signAOQTFloat32(proposal.Checks[i].AnalyticDirectionalDerivative)
		proposal.Checks[i].ActualSign = signAOQTFloat32(proposal.Checks[i].ActualDelta)
		proposal.Checks[i].SignMatch = proposal.Checks[i].AnalyticSign == proposal.Checks[i].ActualSign
		proposal.Passed = proposal.Passed && proposal.Checks[i].SignMatch
	}
	return proposal, nil
}

func dotAOQTFloat32(left, right []float32) float32 {
	if len(left) != len(right) {
		return float32(math.NaN())
	}
	var sum float32
	for i := range left {
		sum += left[i] * right[i]
	}
	return sum
}

func aoqtNamedComponentValue(components AOQTSidecarObjectiveComponents, name string) float32 {
	switch name {
	case "q3_gain":
		return components.Q3Gain
	case "q3_order_guard":
		return components.Q3OrderGuard
	case "q3_score_distill":
		return components.Q3ScoreDistill
	case "q5_order_guard":
		return components.Q5OrderGuard
	case "q5_score_distill":
		return components.Q5ScoreDistill
	case "nf_boundary_guard":
		return components.NFBoundaryGuard
	default:
		return float32(math.NaN())
	}
}

func sha256AOQTForwardConsistencySelectionSeed(workplanSeed int64, foldID, splitManifestSHA256, manifestSHA256, rowIDSHA256 string) string {
	payload := strings.Join([]string{
		"aoqt-v7-r2-forward-consistency-subset-v1",
		strconv.FormatInt(workplanSeed, 10),
		foldID,
		splitManifestSHA256,
		manifestSHA256,
		rowIDSHA256,
	}, "\x00")
	sum := sha256.Sum256([]byte(payload))
	return hex.EncodeToString(sum[:])
}

func selectAOQTForwardConsistencyRows(rows []AOQTSidecarCalibrationRow, selectionSeed string) []AOQTSidecarCalibrationRow {
	type candidate struct {
		row  AOQTSidecarCalibrationRow
		hash string
	}
	candidates := make([]candidate, len(rows))
	for i, row := range rows {
		sum := sha256.Sum256([]byte(selectionSeed + "\x00" + row.RowID))
		candidates[i] = candidate{row: row, hash: hex.EncodeToString(sum[:])}
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].hash != candidates[j].hash {
			return candidates[i].hash < candidates[j].hash
		}
		return candidates[i].row.RowID < candidates[j].row.RowID
	})
	selected := make([]AOQTSidecarCalibrationRow, AOQTV7R2ForwardConsistencyProbeRowCount)
	for i := range selected {
		selected[i] = candidates[i].row
	}
	sort.Slice(selected, func(i, j int) bool { return selected[i].RowID < selected[j].RowID })
	return selected
}
