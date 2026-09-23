package eosruntime

import (
	"strings"
	"testing"
)

func TestAOQTV9ActiveSetNullspaceBuildDirectionEvidenceBindsScreenContract(t *testing.T) {
	q3 := make([]float32, AOQTV9ActiveSetNullspaceExpectedAngleSize)
	q3[5] = -2
	protected := []aoqtProtectedAngleGradient{
		{Name: "q3_order_guard", Grad: make([]float32, AOQTV9ActiveSetNullspaceExpectedAngleSize)},
		{Name: "q3_score_distill", Grad: make([]float32, AOQTV9ActiveSetNullspaceExpectedAngleSize)},
		{Name: "q5_order_guard", Grad: make([]float32, AOQTV9ActiveSetNullspaceExpectedAngleSize)},
		{Name: "q5_score_distill", Grad: make([]float32, AOQTV9ActiveSetNullspaceExpectedAngleSize)},
		{Name: "nf_boundary_guard", Grad: make([]float32, AOQTV9ActiveSetNullspaceExpectedAngleSize)},
	}
	evidence, err := AOQTV9BuildDirectionEvidence(q3, protected)
	if err != nil {
		t.Fatalf("build V9 direction evidence: %v", err)
	}
	if err := evidence.Validate(); err != nil {
		t.Fatalf("validate V9 direction evidence: %v", err)
	}
	if len(evidence.Candidates) == 0 || len(evidence.Candidates) > AOQTV9ActiveSetNullspaceDirectionCap {
		t.Fatalf("candidate count = %d, want within cap", len(evidence.Candidates))
	}
	if got := len(AOQTV9ActiveSetNullspaceMagnitudes()) * len(evidence.Candidates); got > AOQTV9ActiveSetNullspaceCandidateCap {
		t.Fatalf("endpoint schedule count = %d, cap %d", got, AOQTV9ActiveSetNullspaceCandidateCap)
	}
}

func TestAOQTV9ActiveSetNullspaceDirectionEvidenceRejectsV8HashTamper(t *testing.T) {
	q3 := make([]float32, AOQTV9ActiveSetNullspaceExpectedAngleSize)
	q3[5] = -2
	protected := make([]aoqtProtectedAngleGradient, 5)
	for i := range protected {
		protected[i] = aoqtProtectedAngleGradient{Name: []string{"q3_order_guard", "q3_score_distill", "q5_order_guard", "q5_score_distill", "nf_boundary_guard"}[i], Grad: make([]float32, AOQTV9ActiveSetNullspaceExpectedAngleSize)}
	}
	evidence, err := AOQTV9BuildDirectionEvidence(q3, protected)
	if err != nil {
		t.Fatalf("build V9 direction evidence: %v", err)
	}
	evidence.CanonicalV8ReceiptSHA256 = strings.Repeat("0", 64)
	if err := evidence.Validate(); err == nil || !strings.Contains(err.Error(), "canonical V8 receipt") {
		t.Fatalf("V8 hash tamper error = %v, want canonical V8 rejection", err)
	}
}

func TestRunAOQTV9ActiveSetNullspaceRequiresCanonicalV8PathBeforeInputResolve(t *testing.T) {
	_, err := RunAOQTV9ActiveSetNullspaceScreen(AOQTSidecarV9ActiveSetNullspaceScreenConfig{
		Mode:                  AOQTSidecarOptimizerModeV9ActiveSetNullspaceScreen,
		ProbeOnly:             true,
		OutputReceiptJSONPath: "unused.receipt.json",
		ManifestPath:          "missing-manifest.json",
		RowsJSONLPath:         "missing-rows.jsonl",
		PreflightJSONPath:     "missing-preflight.json",
	})
	if err == nil || !strings.Contains(err.Error(), "canonical V8 terminal receipt path") {
		t.Fatalf("empty V8 path error = %v, want direct API path guard before input resolution", err)
	}
}

func TestAOQTV9ActiveSetNullspaceReceiptValidatesSyntheticProvedEnvelope(t *testing.T) {
	receipt := aoqtV9TestReceipt(t)
	if err := receipt.Validate(); err != nil {
		t.Fatalf("honest V9 receipt validation: %v", err)
	}
	if !receipt.Passed || receipt.ObjectiveEvaluationCount != 1+len(receipt.Endpoints) || len(receipt.Endpoints) > AOQTV9ActiveSetNullspaceCandidateCap {
		t.Fatalf("fixture accounting invalid: passed=%v evals=%d endpoints=%d", receipt.Passed, receipt.ObjectiveEvaluationCount, len(receipt.Endpoints))
	}
}

func TestAOQTV9ActiveSetNullspaceRejectsDerivativeTamper(t *testing.T) {
	receipt := aoqtV9TestReceipt(t)
	receipt.Endpoints[0].ActualQ3Derivative = 0
	receipt.EndpointHashChain, receipt.EndpointHashChainTailSHA256 = aoqtV9TestEndpointChain(t, receipt.Endpoints)
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "derivative eligibility") {
		t.Fatalf("derivative tamper error = %v, want derivative eligibility rejection", err)
	}
}

func TestAOQTV9ActiveSetNullspaceRejectsDirectionTamper(t *testing.T) {
	receipt := aoqtV9TestReceipt(t)
	receipt.DirectionEvidence.Candidates[0].Direction[5] = 0
	directionSHA, err := aoqtCanonicalSHA256(receipt.DirectionEvidence)
	if err != nil {
		t.Fatalf("direction evidence sha: %v", err)
	}
	receipt.DirectionEvidenceSHA256 = directionSHA
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "direction evidence is invalid") {
		t.Fatalf("direction tamper error = %v, want invalid direction rejection", err)
	}
}

func TestAOQTV9ActiveSetNullspaceRejectsOmittedDuplicatedReorderedEndpoint(t *testing.T) {
	receipt := aoqtV9TestReceipt(t)
	omitted := receipt
	omitted.Endpoints = omitted.Endpoints[:len(omitted.Endpoints)-1]
	omitted.ObjectiveEvaluationCount = 1 + len(omitted.Endpoints)
	omitted.EndpointHashChain, omitted.EndpointHashChainTailSHA256 = aoqtV9TestEndpointChain(t, omitted.Endpoints)
	if err := omitted.Validate(); err == nil || !strings.Contains(err.Error(), "omitted endpoint") {
		t.Fatalf("omitted endpoint error = %v, want schedule rejection", err)
	}

	duplicated := receipt
	duplicated.Endpoints = append([]AOQTSidecarV9ActiveSetNullspaceEndpoint(nil), receipt.Endpoints...)
	duplicated.Endpoints[1] = duplicated.Endpoints[0]
	duplicated.EndpointHashChain, duplicated.EndpointHashChainTailSHA256 = aoqtV9TestEndpointChain(t, duplicated.Endpoints)
	if err := duplicated.Validate(); err == nil || !strings.Contains(err.Error(), "canonical schedule position") {
		t.Fatalf("duplicated endpoint error = %v, want schedule rejection", err)
	}

	reordered := receipt
	reordered.Endpoints = append([]AOQTSidecarV9ActiveSetNullspaceEndpoint(nil), receipt.Endpoints...)
	reordered.Endpoints[0], reordered.Endpoints[1] = reordered.Endpoints[1], reordered.Endpoints[0]
	reordered.EndpointHashChain, reordered.EndpointHashChainTailSHA256 = aoqtV9TestEndpointChain(t, reordered.Endpoints)
	if err := reordered.Validate(); err == nil || !strings.Contains(err.Error(), "canonical schedule position") {
		t.Fatalf("reordered endpoint error = %v, want schedule rejection", err)
	}
}

func TestAOQTV9ActiveSetNullspaceRejectsFalseSafetyFlagsDenseClaimTolerance(t *testing.T) {
	receipt := aoqtV9TestReceipt(t)
	falseFlag := receipt
	falseFlag.Endpoints = append([]AOQTSidecarV9ActiveSetNullspaceEndpoint(nil), receipt.Endpoints...)
	falseFlag.Endpoints[0].ProtectedEligible = false
	falseFlag.EndpointHashChain, falseFlag.EndpointHashChainTailSHA256 = aoqtV9TestEndpointChain(t, falseFlag.Endpoints)
	if err := falseFlag.Validate(); err == nil || !strings.Contains(err.Error(), "policy flags") {
		t.Fatalf("false policy flag error = %v, want policy flag rejection", err)
	}

	claim := receipt
	claim.QualityClaim = true
	if err := claim.Validate(); err == nil || !strings.Contains(err.Error(), "claims") {
		t.Fatalf("claim tamper error = %v, want claims rejection", err)
	}

	tolerance := receipt
	tolerance.DenseMaxAbsDeltaTolerance *= 2
	if err := tolerance.Validate(); err == nil || !strings.Contains(err.Error(), "dense tolerance") {
		t.Fatalf("tolerance tamper error = %v, want dense tolerance rejection", err)
	}
}

func TestAOQTV9ActiveSetNullspaceRejectsObjectiveWeightTamper(t *testing.T) {
	receipt := aoqtV9TestReceipt(t)
	receipt.ObjectiveWeightSums.Q5ScoreDistill = 0
	for i := range receipt.Endpoints {
		receipt.Endpoints[i].CandidateActivation.Q5ScoreDistillCount = 0
	}
	receipt.EndpointHashChain, receipt.EndpointHashChainTailSHA256 = aoqtV9TestEndpointChain(t, receipt.Endpoints)
	if err := receipt.Validate(); err != nil {
		t.Fatalf("zero-weight inactive component fixture should validate: %v", err)
	}
	receipt.ObjectiveWeightSums.Q5ScoreDistill = 1
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "policy flags") {
		t.Fatalf("weight tamper error = %v, want policy replay rejection", err)
	}
}

func TestAOQTV9ActiveSetNullspaceAllowsRejectedDerivativeEndpoint(t *testing.T) {
	receipt := aoqtV9TestReceipt(t)
	receipt.Endpoints[0].ActualProtectedDerivatives[0] = AOQTV9ProtectedDerivativeTolerance * 2
	receipt.Endpoints[0].DerivativeEligible = false
	receipt.Endpoints[0].TransactionEligible = false
	receipt.Endpoints[0].Reason = AOQTV9ActiveSetNullspaceDerivativeSafetyReject
	aoqtV9RefreshTestReceiptEndpoints(t, &receipt)
	if err := receipt.Validate(); err != nil {
		t.Fatalf("derivative-rejected endpoint should remain valid terminal evidence: %v", err)
	}
}

func TestAOQTV9ActiveSetNullspaceAllowsNoAngleRejectedEndpoint(t *testing.T) {
	receipt := aoqtV9TestReceipt(t)
	receipt.Endpoints[0].ActualDirection = make([]float32, AOQTV9ActiveSetNullspaceExpectedAngleSize)
	sha, err := aoqtActualDirectionVectorSHA256(receipt.Endpoints[0].ActualDirection)
	if err != nil {
		t.Fatalf("zero actual direction sha: %v", err)
	}
	receipt.Endpoints[0].ActualDirectionSHA256 = sha
	receipt.Endpoints[0].ActualDirectionMaxAbs = 0
	receipt.Endpoints[0].ActualQ3Derivative = 0
	receipt.Endpoints[0].ActualProtectedDerivatives = make([]float64, len(receipt.Endpoints[0].ActualProtectedDerivatives))
	receipt.Endpoints[0].DerivativeEligible = false
	receipt.Endpoints[0].AngleMoved = false
	receipt.Endpoints[0].TransactionEligible = false
	receipt.Endpoints[0].Reason = string(aoqtRejectionNoAngleMovement)
	aoqtV9RefreshTestReceiptEndpoints(t, &receipt)
	if err := receipt.Validate(); err != nil {
		t.Fatalf("no-angle rejected endpoint should remain valid terminal evidence: %v", err)
	}
}

func TestAOQTV9ActiveSetNullspaceEndpointAllowsTinyPositiveProtectedResidual(t *testing.T) {
	receipt := aoqtV9TestReceipt(t)
	receipt.Endpoints[0].ActualProtectedDerivatives[0] = AOQTV9ProtectedDerivativeTolerance / 2
	receipt.EndpointHashChain, receipt.EndpointHashChainTailSHA256 = aoqtV9TestEndpointChain(t, receipt.Endpoints)
	if err := receipt.Validate(); err != nil {
		t.Fatalf("tiny positive protected residual should validate under tolerance: %v", err)
	}
}

func TestAOQTV9ActiveSetNullspaceRejectsDenseProofOrderingTamper(t *testing.T) {
	receipt := aoqtV9TestReceipt(t)
	receipt.Endpoints[0].RankOrder, receipt.Endpoints[1].RankOrder = receipt.Endpoints[1].RankOrder, receipt.Endpoints[0].RankOrder
	receipt.EndpointHashChain, receipt.EndpointHashChainTailSHA256 = aoqtV9TestEndpointChain(t, receipt.Endpoints)
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "rank_order") {
		t.Fatalf("ranking tamper error = %v, want rank_order rejection", err)
	}
}

func TestAOQTV9ActiveSetNullspaceValidateBoundRejectsV8HashTamper(t *testing.T) {
	receipt := aoqtV9TestReceipt(t)
	receipt.Binding.CanonicalV8ReceiptFileSHA256 = strings.Repeat("5", 64)
	if err := receipt.ValidateBound(); err == nil || !strings.Contains(err.Error(), "canonical V8 receipt") {
		t.Fatalf("bound V8 hash tamper error = %v, want pinned receipt rejection", err)
	}
}

func aoqtV9TestReceipt(t *testing.T) AOQTSidecarV9ActiveSetNullspaceScreenReceipt {
	t.Helper()
	q3 := make([]float32, AOQTV9ActiveSetNullspaceExpectedAngleSize)
	q3[5] = -2
	protected := []aoqtProtectedAngleGradient{
		{Name: "q3_order_guard", Grad: make([]float32, AOQTV9ActiveSetNullspaceExpectedAngleSize)},
		{Name: "q3_score_distill", Grad: make([]float32, AOQTV9ActiveSetNullspaceExpectedAngleSize)},
		{Name: "q5_order_guard", Grad: make([]float32, AOQTV9ActiveSetNullspaceExpectedAngleSize)},
		{Name: "q5_score_distill", Grad: make([]float32, AOQTV9ActiveSetNullspaceExpectedAngleSize)},
		{Name: "nf_boundary_guard", Grad: make([]float32, AOQTV9ActiveSetNullspaceExpectedAngleSize)},
	}
	evidence, q3SHA, protectedNames, protectedSHA, evidenceSHA, err := aoqtV9BuildDirectionEvidenceBound(q3, protected)
	if err != nil {
		t.Fatalf("direction evidence: %v", err)
	}
	scheduleSHA, err := aoqtV9DirectionScheduleSHA256(len(evidence.Candidates))
	if err != nil {
		t.Fatalf("schedule sha: %v", err)
	}
	policy := aoqtV7R6CanonicalEligibilityPolicy()
	policySHA, err := aoqtActualCoordinatePolicySHA256(policy)
	if err != nil {
		t.Fatalf("policy sha: %v", err)
	}
	baseline := AOQTSidecarObjectiveComponents{Q3Gain: 1, Q3OrderGuard: 1, Q3ScoreDistill: 1, Q5OrderGuard: 1, Q5ScoreDistill: 1, NFBoundaryGuard: 1}
	candidate := AOQTSidecarObjectiveComponents{Q3Gain: 0.9, Q3OrderGuard: 1, Q3ScoreDistill: 1, Q5OrderGuard: 1, Q5ScoreDistill: 1, NFBoundaryGuard: 1}
	activation := AOQTSidecarObjectiveActivation{Q3GainEligiblePairs: 1, Q3GainContributingPairs: 1, Q3OrderGuardPairs: 1, Q3OrderGuardContributing: 1, Q3ScoreDistillCount: 1, Q5OrderGuardPairs: 1, Q5OrderGuardContributing: 1, Q5ScoreDistillCount: 1, NFBoundaryGuardPairs: 1, NFBoundaryGuardContributing: 1}
	endpoints := make([]AOQTSidecarV9ActiveSetNullspaceEndpoint, 0, len(AOQTV9ActiveSetNullspaceMagnitudes()))
	for i, magnitude := range AOQTV9ActiveSetNullspaceMagnitudes() {
		direction := evidence.Candidates[0]
		actual := aoqtV9ScaleDirection32(direction.Direction, magnitude)
		sha, err := aoqtActualDirectionVectorSHA256(actual)
		if err != nil {
			t.Fatalf("direction sha: %v", err)
		}
		delta := aoqtObjectiveComponentDelta(baseline, candidate)
		endpoint := AOQTSidecarV9ActiveSetNullspaceEndpoint{
			Ordinal: i + 1, DirectionRankOrder: 1, DirectionOrdinal: direction.Ordinal, Magnitude: magnitude,
			RequestedDirectionSHA256: sha, ActualDirectionSHA256: sha, ActualDirectionMaxAbs: aoqtMaxAbsFloat32(actual), ActualDirection: actual,
			ActualQ3Derivative: float64(magnitude) * direction.PrimaryDerivative, ActualProtectedDerivatives: make([]float64, len(direction.ProtectedDerivatives)), DerivativeEligible: true, AngleMoved: true,
			BaselineLoss: baseline.Sum(), BaselineLossFinite: true, CandidateLoss: candidate.Sum(), CandidateLossFinite: true,
			BaselineComponents: baseline, BaselineComponentsFinite: true, CandidateComponents: candidate, CandidateComponentsFinite: true,
			ComponentDeltas: delta, ComponentDeltasFinite: true, CandidateActivation: activation, TotalDelta: candidate.Sum() - baseline.Sum(), TotalDeltaFinite: true,
			Q3GainEligible: true, ProtectedEligible: true, TransactionEligible: true, Reason: aoqtProposalAcceptedReason,
			DenseInvariantProved: true, DenseInvariantEligible: true, DenseInvariantMaxAbsDelta: 0, DenseInvariantProofSHA256: strings.Repeat("d", 64),
		}
		endpoints = append(endpoints, endpoint)
	}
	applyAOQTV9EndpointRanking(endpoints)
	chain, tail := aoqtV9TestEndpointChain(t, endpoints)
	return AOQTSidecarV9ActiveSetNullspaceScreenReceipt{
		Schema: AOQTSidecarV9ActiveSetNullspaceScreenSchema, Mode: AOQTSidecarOptimizerModeV9ActiveSetNullspaceScreen, Required: true, ProbeOnly: true, Passed: true,
		BasisVersion: AOQTV9NullspaceBasisVersion, ConeDerivativeVersion: AOQTV9ConeDerivativeVersion, ScheduleVersion: AOQTV9ActiveSetNullspaceScheduleVersion, RankingVersion: AOQTV9ActiveSetNullspaceRankingVersion, PolicyVersion: AOQTV9ActiveSetNullspacePolicyVersion,
		SelectionSeedSHA256: strings.Repeat("1", 64), CoordinateCount: AOQTV9ActiveSetNullspaceExpectedAngleSize, Dim: AOQTV9ActiveSetNullspaceExpectedDim, DirectionCap: AOQTV9ActiveSetNullspaceDirectionCap, Magnitudes: AOQTV9ActiveSetNullspaceMagnitudes(), CandidateCap: AOQTV9ActiveSetNullspaceCandidateCap,
		ObjectiveEvaluationCount: 1 + len(endpoints), FullFoldRowCount: AOQTV7R6ProgressiveScreenPopulationTotal, FullFoldWeightingVersion: AOQTV9ActiveSetNullspaceFullWeighting, HTFactorsApplied: false, BaselineLoss: baseline.Sum(), BaselineLossFinite: true, BaselineComponents: baseline, BaselineActivation: activation,
		ObjectiveWeightSums: AOQTSidecarRowWeights{Q3Gain: 1, Q3OrderGuard: 1, Q3ScoreDistill: 1, Q5OrderGuard: 1, Q5ScoreDistill: 1, NFBoundaryGuard: 1},
		Q3GradientSHA256:    q3SHA, ProtectedGradientNames: protectedNames, ProtectedGradientMatrixSHA256: protectedSHA, DirectionEvidenceSHA256: evidenceSHA, DirectionScheduleSHA256: scheduleSHA, CanonicalObjectiveContractSHA256: strings.Repeat("2", 64), CanonicalEligibilityPolicySHA256: policySHA,
		CanonicalV8ReceiptFileSHA256: AOQTV9CanonicalV8ReceiptFileSHA256, CanonicalV8TerminalReceiptBound: true, DenseInvariantStatus: AOQTV9ActiveSetNullspaceDenseProof, DenseInvariantProofSHA256: strings.Repeat("3", 64), DenseMaxAbsDeltaTolerance: policy.DenseMaxAbsDeltaTolerance,
		GradientEvidenceDeferred: false, FullVerificationCompleted: true, FeasibleEndpointCount: len(endpoints), SelectedEndpointOrdinals: []int{1, 2, 3, 4}, EndpointHashChain: chain, EndpointHashChainTailSHA256: tail, DirectionEvidence: evidence, Endpoints: endpoints,
	}
}

func aoqtV9TestEndpointChain(t *testing.T, endpoints []AOQTSidecarV9ActiveSetNullspaceEndpoint) (string, string) {
	t.Helper()
	data, tail, err := aoqtV9EndpointHashChain(endpoints)
	if err != nil {
		t.Fatalf("endpoint chain: %v", err)
	}
	return string(data), tail
}

func aoqtV9RefreshTestReceiptEndpoints(t *testing.T, receipt *AOQTSidecarV9ActiveSetNullspaceScreenReceipt) {
	t.Helper()
	applyAOQTV9EndpointRanking(receipt.Endpoints)
	receipt.FeasibleEndpointCount, receipt.SelectedEndpointOrdinals = aoqtV9EndpointSelection(receipt.Endpoints)
	receipt.Passed = receipt.FeasibleEndpointCount > 0
	if receipt.Passed {
		receipt.FailureReason = ""
	} else {
		receipt.FailureReason = "no dense-invariant-safe V9 active-set nullspace endpoint satisfied the transaction policy"
	}
	receipt.EndpointHashChain, receipt.EndpointHashChainTailSHA256 = aoqtV9TestEndpointChain(t, receipt.Endpoints)
}
