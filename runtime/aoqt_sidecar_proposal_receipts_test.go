package eosruntime

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func TestAOQTOptimizerProposalReceiptsCaptureBoundedDecisionEvidence(t *testing.T) {
	trainer := newTinyAOQTTrainer(t, false, 288)
	grad := make([]float32, AOQTSidecarAngleCount)
	grad[0] = 1
	weights := AOQTSidecarRowWeights{Q3Gain: 1}
	diagnostics := AOQTSidecarOptimizerDiagnostics{
		PlannedSteps:       1,
		MaxAttemptsPerStep: aoqtTransactionalMaxAttemptsPerStep,
	}
	receipts := make(AOQTSidecarOptimizerProposalReceipts, 0)
	diagnostics.ProposalReceipts = &receipts

	accepted, err := trainer.acceptTransactionalAdamStep(grad, grad, aoqtSafeStepEvaluation(1, weights), func() (aoqtStepEvaluation, error) {
		if math.Abs(float64(trainer.angles[0])) > 0.0075 {
			return aoqtSafeStepEvaluation(2, weights), nil
		}
		return aoqtSafeStepEvaluation(0.5, weights), nil
	}, weights, &diagnostics)
	if err != nil {
		t.Fatalf("transactional step: %v", err)
	}
	if !accepted {
		t.Fatalf("safe Adam proposal was not accepted")
	}
	diagnostics.AttemptedSteps = 1
	receiptCount := 0
	if diagnostics.ProposalReceipts != nil {
		receiptCount = len(*diagnostics.ProposalReceipts)
	}
	if diagnostics.ProposalReceipts == nil || receiptCount != diagnostics.ProposalAttempts || receiptCount != 2 {
		t.Fatalf("proposal receipts = %d, attempts = %d, want two exact Adam receipts", receiptCount, diagnostics.ProposalAttempts)
	}
	first, second := (*diagnostics.ProposalReceipts)[0], (*diagnostics.ProposalReceipts)[1]
	if first.Ordinal != 1 || first.Kind != aoqtProposalKindAdam || first.Accepted || first.Reason != string(aoqtRejectionLossIncrease) {
		t.Fatalf("first receipt = %+v, want rejected Adam loss-increase receipt", first)
	}
	if second.Ordinal != 2 || second.Kind != aoqtProposalKindAdam || !second.Accepted || second.Reason != aoqtProposalAcceptedReason {
		t.Fatalf("second receipt = %+v, want accepted Adam receipt", second)
	}
	if !first.BaselineLossFinite || !first.CandidateLossFinite || first.BaselineLoss != 1 || first.CandidateLoss != 2 || first.BaselineComponents.Q3Gain != 1 || first.CandidateComponents.Q3Gain != 2 {
		t.Fatalf("first receipt objective evidence = %+v, want exact baseline/candidate values", first)
	}
	if !second.AngleMoved || !second.ActivationChecked || !second.ActivationSufficient || !second.CandidateComponentsFinite {
		t.Fatalf("second receipt gate evidence = %+v, want complete accepted evidence", second)
	}
	if err := validateAOQTOptimizerPathDiagnostics(diagnostics, "receipt test"); err != nil {
		t.Fatalf("receipt diagnostics validation: %v", err)
	}

	withReceiptsHash, err := diagnostics.SHA256()
	if err != nil {
		t.Fatalf("receipt diagnostics hash: %v", err)
	}
	withoutReceipts := diagnostics
	withoutReceipts.ProposalReceipts = nil
	withoutReceiptsHash, err := withoutReceipts.SHA256()
	if err != nil {
		t.Fatalf("legacy diagnostics hash: %v", err)
	}
	if withReceiptsHash == withoutReceiptsHash {
		t.Fatalf("proposal receipts were not bound into diagnostics hash")
	}

	bounded := diagnostics
	bounded.MaxAttemptsPerStep = 1
	if err := validateAOQTOptimizerPathDiagnostics(bounded, "receipt bound test"); err == nil || !strings.Contains(err.Error(), "exceeds actual max attempts") {
		t.Fatalf("receipt bound validation error = %v, want actual max-attempt rejection", err)
	}
}

func TestAOQTOptimizerProposalReceiptsCaptureCoordinateKind(t *testing.T) {
	trainer := newTinyAOQTTrainer(t, false, 289)
	grad := make([]float32, AOQTSidecarAngleCount)
	grad[0] = 1
	weights := AOQTSidecarRowWeights{Q3Gain: 1}
	diagnostics := AOQTSidecarOptimizerDiagnostics{
		PlannedSteps:       1,
		MaxAttemptsPerStep: aoqtTransactionalMaxAttemptsPerStep,
	}
	receipts := make(AOQTSidecarOptimizerProposalReceipts, 0)
	diagnostics.ProposalReceipts = &receipts

	accepted, err := trainer.acceptTransactionalAdamStep(grad, grad, aoqtSafeStepEvaluation(1, weights), func() (aoqtStepEvaluation, error) {
		if trainer.angles[0] > 0 {
			return aoqtSafeStepEvaluation(0.5, weights), nil
		}
		return aoqtSafeStepEvaluation(2, weights), nil
	}, weights, &diagnostics)
	if err != nil {
		t.Fatalf("transactional coordinate fallback: %v", err)
	}
	if !accepted || diagnostics.CoordinateAcceptedProposals != 1 {
		t.Fatalf("accepted = %v, coordinate accepted = %d, want one coordinate acceptance", accepted, diagnostics.CoordinateAcceptedProposals)
	}
	diagnostics.AttemptedSteps = 1
	receiptCount := 0
	if diagnostics.ProposalReceipts != nil {
		receiptCount = len(*diagnostics.ProposalReceipts)
	}
	if diagnostics.ProposalReceipts == nil || receiptCount != diagnostics.ProposalAttempts || receiptCount <= aoqtTransactionalAdamMaxAttemptsPerStep {
		t.Fatalf("proposal receipts = %d, attempts = %d, want Adam plus coordinate receipts", receiptCount, diagnostics.ProposalAttempts)
	}
	for i, receipt := range (*diagnostics.ProposalReceipts)[:aoqtTransactionalAdamMaxAttemptsPerStep] {
		if receipt.Kind != aoqtProposalKindAdam {
			t.Fatalf("receipt %d kind = %q, want Adam", i, receipt.Kind)
		}
	}
	last := (*diagnostics.ProposalReceipts)[len(*diagnostics.ProposalReceipts)-1]
	if last.Kind != aoqtProposalKindCoordinate || !last.Accepted || last.Reason != aoqtProposalAcceptedReason {
		t.Fatalf("last receipt = %+v, want accepted coordinate receipt", last)
	}
	if err := validateAOQTOptimizerPathDiagnostics(diagnostics, "coordinate receipt test"); err != nil {
		t.Fatalf("coordinate receipt diagnostics validation: %v", err)
	}
}

func TestAOQTOptimizerProposalReceiptsPreserveLegacyDiagnostics(t *testing.T) {
	legacyJSON := []byte(`{"planned_steps":0,"attempted_steps":0,"accepted_steps":0,"proposal_attempts":0,"accepted_proposals":0,"rejected_proposals":0,"backtracks":0,"max_attempts_per_step":0,"exhausted_steps":0,"rejection_diagnostics":{}}`)
	var legacy AOQTSidecarOptimizerDiagnostics
	if err := json.Unmarshal(legacyJSON, &legacy); err != nil {
		t.Fatalf("unmarshal legacy diagnostics: %v", err)
	}
	if err := validateAOQTOptimizerPathDiagnostics(legacy, "legacy receipt diagnostics"); err != nil {
		t.Fatalf("legacy diagnostics rejected: %v", err)
	}
	raw, err := json.Marshal(legacy)
	if err != nil {
		t.Fatalf("marshal legacy diagnostics: %v", err)
	}
	if strings.Contains(string(raw), "proposal_receipts") {
		t.Fatalf("legacy diagnostics unexpectedly serialized absent proposal receipts: %s", raw)
	}
	firstHash, err := legacy.SHA256()
	if err != nil {
		t.Fatalf("legacy diagnostics hash: %v", err)
	}
	var roundTrip AOQTSidecarOptimizerDiagnostics
	if err := json.Unmarshal(raw, &roundTrip); err != nil {
		t.Fatalf("round-trip legacy diagnostics: %v", err)
	}
	secondHash, err := roundTrip.SHA256()
	if err != nil {
		t.Fatalf("round-trip legacy diagnostics hash: %v", err)
	}
	if firstHash != secondHash {
		t.Fatalf("legacy diagnostics hash changed after round trip: %s vs %s", firstHash, secondHash)
	}
}

func TestAOQTOptimizerProposalReceiptsMarkNonFiniteEvidenceUnavailable(t *testing.T) {
	weights := AOQTSidecarRowWeights{Q3Gain: 1}
	diagnostics := AOQTSidecarOptimizerDiagnostics{
		PlannedSteps:          1,
		AttemptedSteps:        1,
		MaxAttemptsPerStep:    aoqtTransactionalMaxAttemptsPerStep,
		ProposalAttempts:      1,
		RejectedProposals:     1,
		Backtracks:            1,
		AdamProposalAttempts:  1,
		AdamRejectedProposals: 1,
	}
	receipts := make(AOQTSidecarOptimizerProposalReceipts, 0, 1)
	diagnostics.ProposalReceipts = &receipts
	baseline := aoqtSafeStepEvaluation(1, weights)
	candidate := aoqtStepEvaluation{
		loss:       float32(math.NaN()),
		components: AOQTSidecarObjectiveComponents{Q3Gain: float32(math.NaN())},
	}
	diagnostics.RecordProposal(1, aoqtProposalKindAdam, aoqtTransactionalProposalDecision{
		reason:     aoqtRejectionNonFiniteLoss,
		angleMoved: true,
	}, baseline, candidate)
	if diagnostics.ProposalReceipts == nil || len(*diagnostics.ProposalReceipts) != 1 {
		t.Fatalf("non-finite proposal receipts = %#v, want one receipt", diagnostics.ProposalReceipts)
	}
	receipt := (*diagnostics.ProposalReceipts)[0]
	if receipt.CandidateLossFinite || receipt.CandidateLoss != 0 || receipt.CandidateComponentsFinite || receipt.CandidateComponents != (AOQTSidecarObjectiveComponents{}) {
		t.Fatalf("non-finite receipt = %+v, want explicit unavailable numeric evidence", receipt)
	}
	if _, err := json.Marshal(diagnostics); err != nil {
		t.Fatalf("marshal non-finite receipt diagnostics: %v", err)
	}
	if err := validateAOQTOptimizerPathDiagnostics(diagnostics, "non-finite receipt test"); err != nil {
		t.Fatalf("non-finite receipt diagnostics validation: %v", err)
	}
}
