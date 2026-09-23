package eosruntime

import (
	"os"
	"strings"
	"testing"
)

func TestAOQTV8ConstrainedBasisReceiptValidatesFailClosed(t *testing.T) {
	receipt := aoqtV8TestReceipt(t)
	if receipt.Passed {
		t.Fatal("fixture should be fail-closed while dense invariant is deferred")
	}
	if err := receipt.Validate(); err != nil {
		t.Fatalf("honest V8 constrained-basis receipt validation: %v", err)
	}
	if receipt.ObjectiveEvaluationCount != 1+len(receipt.Endpoints) || len(receipt.Endpoints) > AOQTV8ConstrainedBasisCandidateCap {
		t.Fatalf("fixture accounting escaped cap: evals=%d endpoints=%d", receipt.ObjectiveEvaluationCount, len(receipt.Endpoints))
	}
}

func TestAOQTV8ConstrainedBasisRejectsDeferredDensePass(t *testing.T) {
	receipt := aoqtV8TestReceipt(t)
	receipt.Passed = true
	receipt.FailureReason = ""
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "passed state") {
		t.Fatalf("deferred dense pass error = %v, want passed-state rejection", err)
	}
}

func TestAOQTV8ConstrainedBasisRejectsProvedDenseWithoutVerifier(t *testing.T) {
	receipt := aoqtV8TestReceipt(t)
	receipt.DenseInvariantStatus = AOQTV8ConstrainedBasisDenseInvariantProof
	receipt.DenseInvariantDeferred = false
	receipt.DenseInvariantProofSHA256 = strings.Repeat("6", 64)
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "requires bound validation") {
		t.Fatalf("proved dense plain validation error = %v, want bound-validation rejection", err)
	}
}

func TestAOQTV8ConstrainedBasisRejectsCandidateCapEscape(t *testing.T) {
	receipt := aoqtV8TestReceipt(t)
	for len(receipt.Endpoints) <= AOQTV8ConstrainedBasisCandidateCap {
		next := receipt.Endpoints[len(receipt.Endpoints)%2]
		next.Ordinal = len(receipt.Endpoints) + 1
		next.RankOrder = next.Ordinal
		receipt.Endpoints = append(receipt.Endpoints, next)
	}
	receipt.ObjectiveEvaluationCount = 1 + len(receipt.Endpoints)
	receipt.EndpointHashChain, receipt.EndpointHashChainTailSHA256 = aoqtV8TestEndpointChain(t, receipt.Endpoints)
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "cap exceeded") {
		t.Fatalf("cap escape error = %v, want cap rejection", err)
	}
}

func TestAOQTV8ConstrainedBasisRejectsForgedRanking(t *testing.T) {
	receipt := aoqtV8TestReceipt(t)
	receipt.Endpoints[0].RankOrder, receipt.Endpoints[1].RankOrder = receipt.Endpoints[1].RankOrder, receipt.Endpoints[0].RankOrder
	receipt.EndpointHashChain, receipt.EndpointHashChainTailSHA256 = aoqtV8TestEndpointChain(t, receipt.Endpoints)
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "rank_order") {
		t.Fatalf("forged ranking error = %v, want rank_order rejection", err)
	}
}

func TestAOQTV8ConstrainedBasisRejectsEndpointHashTamper(t *testing.T) {
	receipt := aoqtV8TestReceipt(t)
	receipt.EndpointHashChainTailSHA256 = strings.Repeat("f", 64)
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "tail") {
		t.Fatalf("hash-chain tamper error = %v, want tail rejection", err)
	}
}

func TestAOQTV8ConstrainedBasisRejectsBasisSourceGradientTamper(t *testing.T) {
	receipt := aoqtV8TestReceipt(t)
	receipt.Basis[0].SourceGradientSHA256 = strings.Repeat("9", 64)
	basisSHA, err := aoqtCanonicalSHA256(receipt.Basis)
	if err != nil {
		t.Fatalf("basis sha: %v", err)
	}
	receipt.BasisSHA256 = basisSHA
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "source gradient hash") {
		t.Fatalf("basis source-gradient tamper error = %v, want q3 source binding rejection", err)
	}
}

func TestAOQTV8ConstrainedBasisRejectsNoAngleForgedEligibility(t *testing.T) {
	receipt := aoqtV8TestReceipt(t)
	receipt.Endpoints[0].ActualDirection = make([]float32, AOQTV8ConstrainedBasisExpectedAngleSize)
	zeroSHA, err := aoqtActualDirectionVectorSHA256(receipt.Endpoints[0].ActualDirection)
	if err != nil {
		t.Fatalf("zero direction sha: %v", err)
	}
	receipt.Endpoints[0].ActualDirectionSHA256 = zeroSHA
	receipt.Endpoints[0].ActualDirectionMaxAbs = 0
	receipt.Endpoints[0].AngleMoved = false
	receipt.Endpoints[0].TransactionEligible = true
	receipt.Endpoints[0].Reason = string(aoqtRejectionNoAngleMovement)
	receipt.EndpointHashChain, receipt.EndpointHashChainTailSHA256 = aoqtV8TestEndpointChain(t, receipt.Endpoints)
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "actual direction must exactly match requested") {
		t.Fatalf("no-angle forged eligibility error = %v, want scheduled-direction rejection", err)
	}
}

func TestAOQTV8ConstrainedBasisRejectsRequestedActualDirectionMismatch(t *testing.T) {
	receipt := aoqtV8TestReceipt(t)
	receipt.Endpoints[0].ActualDirection = append([]float32(nil), receipt.Endpoints[0].ActualDirection...)
	receipt.Endpoints[0].ActualDirection[0] += 0.0001
	actualSHA, err := aoqtActualDirectionVectorSHA256(receipt.Endpoints[0].ActualDirection)
	if err != nil {
		t.Fatalf("actual direction sha: %v", err)
	}
	receipt.Endpoints[0].ActualDirectionSHA256 = actualSHA
	receipt.Endpoints[0].ActualDirectionMaxAbs = aoqtMaxAbsFloat32(receipt.Endpoints[0].ActualDirection)
	receipt.EndpointHashChain, receipt.EndpointHashChainTailSHA256 = aoqtV8TestEndpointChain(t, receipt.Endpoints)
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "actual direction must exactly match requested") {
		t.Fatalf("requested/actual mismatch error = %v, want exact scheduled direction rejection", err)
	}
}

func TestAOQTV8ConstrainedBasisRejectsForgedZeroDirectionAngleMoved(t *testing.T) {
	receipt := aoqtV8TestReceipt(t)
	receipt.Endpoints[0].ActualDirection = make([]float32, AOQTV8ConstrainedBasisExpectedAngleSize)
	zeroSHA, err := aoqtActualDirectionVectorSHA256(receipt.Endpoints[0].ActualDirection)
	if err != nil {
		t.Fatalf("zero direction sha: %v", err)
	}
	receipt.Endpoints[0].ActualDirectionSHA256 = zeroSHA
	receipt.Endpoints[0].ActualDirectionMaxAbs = 0
	receipt.Endpoints[0].AngleMoved = true
	receipt.Endpoints[0].TransactionEligible = false
	receipt.EndpointHashChain, receipt.EndpointHashChainTailSHA256 = aoqtV8TestEndpointChain(t, receipt.Endpoints)
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "angle_moved") {
		t.Fatalf("forged zero-direction angle_moved error = %v, want angle_moved rejection", err)
	}
}

func TestAOQTV8ConstrainedBasisRejectsToleranceInflation(t *testing.T) {
	receipt := aoqtV8TestReceipt(t)
	receipt.DenseMaxAbsDeltaTolerance = receipt.DenseMaxAbsDeltaTolerance * 2
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "canonical V7-r6 policy") {
		t.Fatalf("tolerance inflation error = %v, want canonical tolerance rejection", err)
	}
}

func TestAOQTV8ConstrainedBasisRejectsProvedOmittedEndpoint(t *testing.T) {
	receipt := aoqtV8ProvedEnvelopeReceipt(t)
	receipt.Endpoints = receipt.Endpoints[:len(receipt.Endpoints)-1]
	receipt.ObjectiveEvaluationCount = 1 + len(receipt.Endpoints)
	if err := receipt.validate(true); err == nil || !strings.Contains(err.Error(), "exact canonical endpoint schedule") {
		t.Fatalf("omitted endpoint error = %v, want canonical schedule rejection", err)
	}
}

func TestAOQTV8ConstrainedBasisRejectsProvedDuplicatedEndpoint(t *testing.T) {
	receipt := aoqtV8ProvedEnvelopeReceipt(t)
	receipt.Endpoints[1] = receipt.Endpoints[0]
	if err := receipt.validate(true); err == nil || !strings.Contains(err.Error(), "duplicate endpoint ordinal") {
		t.Fatalf("duplicated endpoint error = %v, want duplicate rejection", err)
	}
}

func TestAOQTV8ConstrainedBasisRejectsProvedReorderedEndpoint(t *testing.T) {
	receipt := aoqtV8ProvedEnvelopeReceipt(t)
	receipt.Endpoints[0], receipt.Endpoints[1] = receipt.Endpoints[1], receipt.Endpoints[0]
	if err := receipt.validate(true); err == nil || !strings.Contains(err.Error(), "canonical schedule position") {
		t.Fatalf("reordered endpoint error = %v, want schedule-position rejection", err)
	}
}

func TestAOQTV8ConstrainedBasisRejectsProvedCompletionMetadataTamper(t *testing.T) {
	receipt := aoqtV8ProvedEnvelopeReceipt(t)
	receipt.FullVerificationCompleted = false
	if err := receipt.validate(true); err == nil || !strings.Contains(err.Error(), "completed gradient and full verification") {
		t.Fatalf("completion metadata error = %v, want completion metadata rejection", err)
	}
	receipt = aoqtV8ProvedEnvelopeReceipt(t)
	receipt.GradientEvidenceDeferred = true
	if err := receipt.validate(true); err == nil || !strings.Contains(err.Error(), "completed gradient and full verification") {
		t.Fatalf("gradient metadata error = %v, want completion metadata rejection", err)
	}
}

func TestAOQTV8ConstrainedBasisDenseIneligibleMovedEndpoint(t *testing.T) {
	basis := aoqtV8TestBasis(t)[0]
	baseline := AOQTSidecarObjectiveComponents{Q3Gain: 1, Q3OrderGuard: 1, Q3ScoreDistill: 1, Q5OrderGuard: 1, Q5ScoreDistill: 1, NFBoundaryGuard: 1}
	candidate := AOQTSidecarObjectiveComponents{Q3Gain: 0.9, Q3OrderGuard: 1, Q3ScoreDistill: 1, Q5OrderGuard: 1, Q5ScoreDistill: 1, NFBoundaryGuard: 1}
	activation := AOQTSidecarObjectiveActivation{Q3GainEligiblePairs: 1, Q3GainContributingPairs: 1, Q3OrderGuardPairs: 1, Q3OrderGuardContributing: 1, Q3ScoreDistillCount: 1, Q5OrderGuardPairs: 1, Q5OrderGuardContributing: 1, Q5ScoreDistillCount: 1, NFBoundaryGuardPairs: 1, NFBoundaryGuardContributing: 1}
	endpoint := aoqtV8TestEndpoint(t, 1, basis, 0.00125, 1, baseline, candidate, activation)
	policy := aoqtV7R6CanonicalEligibilityPolicy()
	endpoint.DenseInvariantProved = true
	endpoint.DenseInvariantEligible = false
	endpoint.DenseInvariantMaxAbsDelta = policy.DenseMaxAbsDeltaTolerance * 2
	endpoint.DenseInvariantProofSHA256 = strings.Repeat("d", 64)
	endpoint.TransactionEligible = false
	endpoint.Reason = AOQTV8ConstrainedBasisDenseInvariantRejection
	if err := validateAOQTV8EndpointDenseProofEnvelope(endpoint, AOQTV8ConstrainedBasisDenseInvariantProof, policy.DenseMaxAbsDeltaTolerance); err != nil {
		t.Fatalf("dense envelope should accept ineligible evidence: %v", err)
	}
	if err := validateAOQTV8Endpoint(endpoint, 0, baseline.Sum(), baseline, policy, map[int]AOQTSidecarV8BasisVector{basis.Ordinal: basis}, true); err != nil {
		t.Fatalf("dense-ineligible moved endpoint should validate as transaction false: %v", err)
	}
	endpoint.TransactionEligible = true
	if err := validateAOQTV8Endpoint(endpoint, 0, baseline.Sum(), baseline, policy, map[int]AOQTSidecarV8BasisVector{basis.Ordinal: basis}, true); err == nil || !strings.Contains(err.Error(), "policy flags") {
		t.Fatalf("forged dense-ineligible transaction error = %v, want policy flag rejection", err)
	}
}

func TestAOQTV8ConstrainedBasisValidateBoundRejectsV7R6HashTamper(t *testing.T) {
	receipt := aoqtV8TestReceipt(t)
	receipt.Binding.CanonicalV7R6ReceiptFileSHA256 = strings.Repeat("5", 64)
	if err := receipt.ValidateBound(); err == nil || !strings.Contains(err.Error(), "canonical V7-r6 terminal receipt") {
		t.Fatalf("bound V7-r6 hash tamper error = %v, want pinned receipt rejection", err)
	}
}

func TestAOQTV8V7R6CompatibilityRejectsWrongPreflightBeforeObjectiveWork(t *testing.T) {
	binding := AOQTSidecarV8ConstrainedBasisScreenBinding{
		Inputs:                       AOQTSidecarRunMetricInputs{AnchorArtifactSHA256: strings.Repeat("1", 64), AnchorPackageManifestSHA256: strings.Repeat("2", 64), AnchorEmbeddingSpaceID: "space", DatasetManifestSHA256: strings.Repeat("3", 64), CompatibilityDigest: strings.Repeat("4", 64)},
		IOReport:                     AOQTSidecarCalibrationIOReport{RowsSHA256: strings.Repeat("5", 64)},
		PreflightPath:                "preflight-a.json",
		PreflightSHA256:              strings.Repeat("6", 64),
		SplitBinding:                 &AOQTSidecarDevSplitBinding{FoldID: "fold-0"},
		FoldID:                       "fold-0",
		CanonicalR4ReceiptPath:       "r4.json",
		CanonicalR4ReceiptSHA256:     strings.Repeat("7", 64),
		CanonicalR4ReceiptFileSHA256: AOQTV7R6CanonicalR4ReceiptFileSHA256,
		CanonicalR5ReceiptPath:       "r5.json",
		CanonicalR5ReceiptSHA256:     strings.Repeat("8", 64),
		CanonicalR5ReceiptFileSHA256: AOQTV7R6CanonicalR5ReceiptFileSHA256,
	}
	v7 := AOQTSidecarV7R6ProgressiveScreenReceipt{
		Binding:          AOQTSidecarV7R6ProgressiveScreenBinding{Inputs: binding.Inputs, IOReport: binding.IOReport, PreflightPath: binding.PreflightPath, PreflightSHA256: strings.Repeat("9", 64), SplitBinding: cloneAOQTDevSplitBinding(binding.SplitBinding), FoldID: binding.FoldID, CanonicalR4ReceiptPath: binding.CanonicalR4ReceiptPath, CanonicalR4ReceiptSHA256: binding.CanonicalR4ReceiptSHA256, CanonicalR4ReceiptFileSHA256: binding.CanonicalR4ReceiptFileSHA256, CanonicalR5ReceiptPath: binding.CanonicalR5ReceiptPath, CanonicalR5ReceiptSHA256: binding.CanonicalR5ReceiptSHA256, CanonicalR5ReceiptFileSHA256: binding.CanonicalR5ReceiptFileSHA256},
		FullVerification: AOQTSidecarV7R6ProgressiveScreenFullVerification{RowCount: AOQTV7R6ProgressiveScreenPopulationTotal, WeightingVersion: AOQTV8ConstrainedBasisFullWeighting, HTFactorsApplied: false},
	}
	if err := validateAOQTV8V7R6Compatibility(v7, binding, AOQTV7R6ProgressiveScreenPopulationTotal, AOQTV8ConstrainedBasisFullWeighting, false); err == nil || !strings.Contains(err.Error(), "preflight") {
		t.Fatalf("wrong preflight compatibility error = %v, want preflight mismatch before objective work", err)
	}
}

func TestAOQTV8ConstrainedBasisValidateBoundRejectsV7R6BindingMismatch(t *testing.T) {
	receipt := aoqtV8RealBindingReceiptOrSkip(t)
	receipt.Binding.PreflightSHA256 = strings.Repeat("7", 64)
	if err := receipt.ValidateBound(); err == nil || !strings.Contains(err.Error(), "preflight") {
		t.Fatalf("wrong preflight binding error = %v, want preflight/V7-r6 binding rejection", err)
	}
}

func TestAOQTV8ConstructBasisBindsQ3OnlySourceHash(t *testing.T) {
	q3Only := make([]float32, AOQTV8ConstrainedBasisExpectedAngleSize)
	q3Only[9], q3Only[3], q3Only[7], q3Only[1] = 4, -3, 2, -1
	fullLoss := append([]float32(nil), q3Only...)
	fullLoss[0] = 99
	basis, q3SHA, _, _, _, err := aoqtV8ConstructBasis(q3Only, nil, []int{4})
	if err != nil {
		t.Fatalf("construct basis: %v", err)
	}
	wantQ3SHA, err := aoqtCanonicalSHA256(q3Only)
	if err != nil {
		t.Fatalf("q3 sha: %v", err)
	}
	fullSHA, err := aoqtCanonicalSHA256(fullLoss)
	if err != nil {
		t.Fatalf("full sha: %v", err)
	}
	if q3SHA != wantQ3SHA || q3SHA == fullSHA {
		t.Fatalf("q3 source hash = %s, want q3-only %s and not full-loss %s", q3SHA, wantQ3SHA, fullSHA)
	}
	for _, vector := range basis {
		if vector.SourceGradientSHA256 != q3SHA {
			t.Fatalf("basis source hash = %s, want q3 hash %s", vector.SourceGradientSHA256, q3SHA)
		}
	}
}

func TestRunAOQTV8ConstrainedBasisRequiresCanonicalV7R6PathBeforeInputResolve(t *testing.T) {
	_, err := RunAOQTV8ConstrainedBasisScreen(AOQTSidecarV8ConstrainedBasisScreenConfig{
		Mode:                   AOQTSidecarOptimizerModeV8ConstrainedBasisScreen,
		ProbeOnly:              true,
		OutputReceiptJSONPath:  "unused.receipt.json",
		ManifestPath:           "missing-manifest.json",
		RowsJSONLPath:          "missing-rows.jsonl",
		PreflightJSONPath:      "missing-preflight.json",
		SplitManifestPath:      "missing-split.json",
		CanonicalR4ReceiptPath: "missing-r4.json",
		CanonicalR5ReceiptPath: "missing-r5.json",
	})
	if err == nil || !strings.Contains(err.Error(), "canonical V7-r6 terminal receipt path") {
		t.Fatalf("empty V7-r6 path error = %v, want direct API path guard before input resolution", err)
	}
}

func TestAOQTV8DenseInvariantProofSyntheticNoAngleAndTamper(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 9901)
	topology, err := NewAOQTGivensIdentityTopology(set.Manifest.Topology.Dim, set.Manifest.Topology.Stages, set.Manifest.Topology.Seed, AOQTSidecarDefaultAngleCap)
	if err != nil {
		t.Fatalf("topology: %v", err)
	}
	direction := make([]float32, countAOQTAngles(topology))
	directionSHA, err := aoqtActualDirectionVectorSHA256(direction)
	if err != nil {
		t.Fatalf("direction sha: %v", err)
	}
	binding := AOQTSidecarV8ConstrainedBasisScreenBinding{
		IOReport: AOQTSidecarCalibrationIOReport{
			ManifestSHA256: strings.Repeat("a", 64),
			RowsSHA256:     strings.Repeat("b", 64),
			RowCount:       len(set.Rows),
		},
		FoldID: "fold-synth",
	}
	endpoints := []AOQTSidecarV8ConstrainedBasisEndpoint{{
		Ordinal:                  1,
		BasisOrdinal:             1,
		Rank:                     4,
		Magnitude:                0.00125,
		GlobalSign:               1,
		RequestedDirectionSHA256: directionSHA,
		ActualDirectionSHA256:    directionSHA,
		ActualDirection:          direction,
		ActualDirectionMaxAbs:    0,
		AngleMoved:               false,
		BaselineLoss:             1,
		CandidateLoss:            0.9,
		TransactionEligible:      true,
		Reason:                   aoqtProposalAcceptedReason,
	}}
	tolerance := 5e-4
	aggregate, err := applyAOQTV8DenseProofsToEndpoints(endpoints, set, binding, topology, tolerance)
	if err != nil {
		t.Fatalf("apply dense proofs: %v", err)
	}
	if !endpoints[0].DenseInvariantProved || !endpoints[0].DenseInvariantEligible || endpoints[0].DenseInvariantMaxAbsDelta != 0 {
		t.Fatalf("no-angle dense proof = proved:%v eligible:%v delta:%g, want proved eligible zero", endpoints[0].DenseInvariantProved, endpoints[0].DenseInvariantEligible, endpoints[0].DenseInvariantMaxAbsDelta)
	}
	if endpoints[0].TransactionEligible || endpoints[0].Reason != string(aoqtRejectionNoAngleMovement) {
		t.Fatalf("no-angle endpoint transaction = %v reason=%q, want ineligible no-angle", endpoints[0].TransactionEligible, endpoints[0].Reason)
	}
	receipt := AOQTSidecarV8ConstrainedBasisScreenReceipt{
		DenseInvariantStatus:      AOQTV8ConstrainedBasisDenseInvariantProof,
		DenseInvariantProofSHA256: aggregate,
		Endpoints:                 endpoints,
	}
	if err := validateAOQTV8BoundDenseProofs(receipt, set, binding, topology, tolerance); err != nil {
		t.Fatalf("validate synthetic dense proof: %v", err)
	}
	tampered := receipt
	tampered.Endpoints = append([]AOQTSidecarV8ConstrainedBasisEndpoint(nil), receipt.Endpoints...)
	tampered.Endpoints[0].DenseInvariantProofSHA256 = strings.Repeat("c", 64)
	if err := validateAOQTV8BoundDenseProofs(tampered, set, binding, topology, tolerance); err == nil || !strings.Contains(err.Error(), "proof hash mismatch") {
		t.Fatalf("tampered dense proof error = %v, want proof hash mismatch", err)
	}
}

func TestAOQTV8BoundObjectiveReplayRejectsForgedComponentsWithValidDenseProof(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 9902)
	trainer, err := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{
		Dim:                                      set.Manifest.Topology.Dim,
		Stages:                                   set.Manifest.Topology.Stages,
		PairingSeed:                              set.Manifest.Topology.Seed,
		WorkplanSeed:                             1,
		OptimizerMode:                            AOQTSidecarOptimizerModeV7R4DevActualCoordinate,
		ActualCoordinateProbeRequired:            true,
		ActualCoordinateProbeOnly:                true,
		ActualCoordinateProbeFoldID:              "fold-synth",
		ActualCoordinateProbeSplitManifestSHA256: strings.Repeat("e", 64),
		LearningRate:                             AOQTV7R4ActualCoordinateProbeLearningRate,
		TrainingContract:                         set.Manifest.TrainingContract,
		CandidateEligibilityPolicy:               cloneAOQTSidecarCandidateEligibilityPolicy(set.Manifest.CandidateEligibilityPolicy),
	})
	if err != nil {
		t.Fatalf("trainer: %v", err)
	}
	if _, err := trainer.Plan(set); err != nil {
		t.Fatalf("plan: %v", err)
	}
	objective, err := objectiveFromAOQTContract(set.Manifest.ObjectiveContract)
	if err != nil {
		t.Fatalf("objective: %v", err)
	}
	baselineLoss, _, baselineActivation, baselineComponents, err := trainer.lossAndAngleGrad(set.Rows, objective)
	if err != nil {
		t.Fatalf("baseline: %v", err)
	}
	q3Grad := make([]float32, countAOQTAngles(trainer.topology))
	for i := range q3Grad {
		q3Grad[i] = float32(i%7+1) * 0.001
	}
	basis, _, _, _, _, err := aoqtV8ConstructBasis(q3Grad, nil, AOQTV8ConstrainedBasisRanks())
	if err != nil {
		t.Fatalf("basis: %v", err)
	}
	schedule, err := aoqtV8CanonicalEndpointSchedule()
	if err != nil {
		t.Fatalf("schedule: %v", err)
	}
	endpoints := make([]AOQTSidecarV8ConstrainedBasisEndpoint, 0, len(schedule))
	for _, item := range schedule {
		endpoints = append(endpoints, trainer.evaluateV8ConstrainedBasisEndpoint(set.Rows, objective, aoqtStepEvaluation{loss: baselineLoss, activation: baselineActivation, components: baselineComponents}, set.Manifest.ObjectiveContract.WeightSums, basis[item.BasisOrdinal-1], item.Ordinal, item.Magnitude, item.GlobalSign))
	}
	binding := AOQTSidecarV8ConstrainedBasisScreenBinding{
		IOReport: AOQTSidecarCalibrationIOReport{
			ManifestSHA256: strings.Repeat("a", 64),
			RowsSHA256:     strings.Repeat("b", 64),
			RowCount:       len(set.Rows),
		},
		FoldID: "fold-synth",
	}
	tolerance := aoqtV7R6CanonicalEligibilityPolicy().DenseMaxAbsDeltaTolerance
	aggregate, err := applyAOQTV8DenseProofsToEndpoints(endpoints, set, binding, trainer.topology, tolerance)
	if err != nil {
		t.Fatalf("dense proof: %v", err)
	}
	receipt := AOQTSidecarV8ConstrainedBasisScreenReceipt{
		BaselineLoss:              baselineLoss,
		BaselineLossFinite:        true,
		BaselineComponents:        baselineComponents,
		BaselineActivation:        baselineActivation,
		DenseMaxAbsDeltaTolerance: tolerance,
		DenseInvariantProofSHA256: aggregate,
		Basis:                     basis,
		Endpoints:                 endpoints,
		Binding:                   binding,
	}
	tampered := receipt
	tampered.Endpoints = append([]AOQTSidecarV8ConstrainedBasisEndpoint(nil), receipt.Endpoints...)
	tampered.Endpoints[0].CandidateComponents.Q3Gain += 0.25
	if err := validateAOQTV8BoundDenseProofs(tampered, set, binding, trainer.topology, tolerance); err != nil {
		t.Fatalf("dense proof should remain valid before objective replay: %v", err)
	}
	if err := validateAOQTV8BoundObjectiveReplay(tampered, trainer, set, objective, set.Manifest.ObjectiveContract.WeightSums); err == nil || !strings.Contains(err.Error(), "objective replay evidence mismatch") {
		t.Fatalf("forged objective replay error = %v, want objective replay mismatch", err)
	}
}

func aoqtV8ProvedEnvelopeReceipt(t *testing.T) AOQTSidecarV8ConstrainedBasisScreenReceipt {
	t.Helper()
	receipt := aoqtV8TestReceipt(t)
	baseline := receipt.BaselineComponents
	activation := receipt.BaselineActivation
	schedule, err := aoqtV8CanonicalEndpointSchedule()
	if err != nil {
		t.Fatalf("schedule: %v", err)
	}
	endpoints := make([]AOQTSidecarV8ConstrainedBasisEndpoint, 0, len(schedule))
	for _, item := range schedule {
		endpoint := aoqtV8TestEndpoint(t, item.Ordinal, receipt.Basis[item.BasisOrdinal-1], item.Magnitude, item.GlobalSign, baseline, AOQTSidecarObjectiveComponents{Q3Gain: 0.9, Q3OrderGuard: 1, Q3ScoreDistill: 1, Q5OrderGuard: 1, Q5ScoreDistill: 1, NFBoundaryGuard: 1}, activation)
		endpoint.DenseInvariantProved = true
		endpoint.DenseInvariantEligible = true
		endpoint.DenseInvariantMaxAbsDelta = 0
		endpoint.DenseInvariantProofSHA256 = strings.Repeat("e", 64)
		endpoints = append(endpoints, endpoint)
	}
	policy := aoqtV7R6CanonicalEligibilityPolicy()
	ranking := aoqtV8RankEndpoints(endpoints, policy)
	rankAt := map[int]int{}
	for i, ordinal := range ranking {
		rankAt[ordinal] = i + 1
	}
	selected := make([]int, 0, len(endpoints))
	for i := range endpoints {
		endpoints[i].RankOrder = rankAt[endpoints[i].Ordinal]
		if endpoints[i].TransactionEligible {
			selected = append(selected, endpoints[i].Ordinal)
		}
	}
	chain, tail := aoqtV8TestEndpointChain(t, endpoints)
	receipt.Passed = len(selected) > 0
	receipt.FailureReason = ""
	receipt.DenseInvariantStatus = AOQTV8ConstrainedBasisDenseInvariantProof
	receipt.DenseInvariantProofSHA256 = strings.Repeat("6", 64)
	receipt.DenseInvariantDeferred = false
	receipt.GradientEvidenceDeferred = false
	receipt.FullVerificationCompleted = true
	receipt.ObjectiveEvaluationCount = 1 + AOQTV8ConstrainedBasisCandidateCap
	receipt.FeasibleEndpointCount = len(selected)
	receipt.SelectedEndpointOrdinals = selected
	receipt.EndpointHashChain = chain
	receipt.EndpointHashChainTailSHA256 = tail
	receipt.Endpoints = endpoints
	return receipt
}

func aoqtV8TestReceipt(t *testing.T) AOQTSidecarV8ConstrainedBasisScreenReceipt {
	t.Helper()
	basis := aoqtV8TestBasis(t)
	basisSHA, err := aoqtCanonicalSHA256(basis)
	if err != nil {
		t.Fatalf("basis sha: %v", err)
	}
	policy := aoqtV7R6CanonicalEligibilityPolicy()
	policySHA, err := aoqtActualCoordinatePolicySHA256(policy)
	if err != nil {
		t.Fatalf("policy sha: %v", err)
	}
	scheduleSHA, err := aoqtCanonicalSHA256(struct {
		Version    string    `json:"version"`
		Ranks      []int     `json:"ranks"`
		Magnitudes []float32 `json:"magnitudes"`
		Signs      []int     `json:"signs"`
		Cap        int       `json:"cap"`
	}{AOQTV8ConstrainedBasisScheduleVersion, AOQTV8ConstrainedBasisRanks(), AOQTV8ConstrainedBasisMagnitudes(), AOQTV8ConstrainedBasisGlobalSigns(), AOQTV8ConstrainedBasisCandidateCap})
	if err != nil {
		t.Fatalf("schedule sha: %v", err)
	}
	baseline := AOQTSidecarObjectiveComponents{Q3Gain: 1, Q3OrderGuard: 1, Q3ScoreDistill: 1, Q5OrderGuard: 1, Q5ScoreDistill: 1, NFBoundaryGuard: 1}
	activation := AOQTSidecarObjectiveActivation{Q3GainEligiblePairs: 1, Q3GainContributingPairs: 1, Q3OrderGuardPairs: 1, Q3OrderGuardContributing: 1, Q3ScoreDistillCount: 1, Q5OrderGuardPairs: 1, Q5OrderGuardContributing: 1, Q5ScoreDistillCount: 1, NFBoundaryGuardPairs: 1, NFBoundaryGuardContributing: 1}
	endpoints := []AOQTSidecarV8ConstrainedBasisEndpoint{
		aoqtV8TestEndpoint(t, 1, basis[0], 0.00125, 1, baseline, AOQTSidecarObjectiveComponents{Q3Gain: 0.9, Q3OrderGuard: 1, Q3ScoreDistill: 1, Q5OrderGuard: 1, Q5ScoreDistill: 1, NFBoundaryGuard: 1}, activation),
		aoqtV8TestEndpoint(t, 2, basis[1], 0.00125, 1, baseline, AOQTSidecarObjectiveComponents{Q3Gain: 1, Q3OrderGuard: 1, Q3ScoreDistill: 1, Q5OrderGuard: 1, Q5ScoreDistill: 1, NFBoundaryGuard: 1}, activation),
	}
	endpoints[0].RankOrder = 1
	endpoints[1].RankOrder = 2
	chain, tail := aoqtV8TestEndpointChain(t, endpoints)
	return AOQTSidecarV8ConstrainedBasisScreenReceipt{
		Schema:                            AOQTSidecarV8ConstrainedBasisScreenSchema,
		Mode:                              AOQTSidecarOptimizerModeV8ConstrainedBasisScreen,
		Required:                          true,
		ProbeOnly:                         true,
		Passed:                            false,
		FailureReason:                     "dense invariant proof deferred; no V8 candidate is acceptable without dense proof",
		BasisVersion:                      AOQTV8ConstrainedBasisVersion,
		ScheduleVersion:                   AOQTV8ConstrainedBasisScheduleVersion,
		RankingVersion:                    AOQTV8ConstrainedBasisRankingVersion,
		PolicyVersion:                     AOQTV8ConstrainedBasisPolicyVersion,
		SelectionSeedSHA256:               strings.Repeat("1", 64),
		CoordinateCount:                   AOQTV8ConstrainedBasisExpectedAngleSize,
		Dim:                               AOQTV8ConstrainedBasisExpectedDim,
		Ranks:                             AOQTV8ConstrainedBasisRanks(),
		Magnitudes:                        AOQTV8ConstrainedBasisMagnitudes(),
		GlobalSigns:                       AOQTV8ConstrainedBasisGlobalSigns(),
		CandidateCap:                      AOQTV8ConstrainedBasisCandidateCap,
		ObjectiveEvaluationCount:          1 + len(endpoints),
		FullFoldRowCount:                  AOQTV7R6ProgressiveScreenPopulationTotal,
		FullFoldWeightingVersion:          AOQTV8ConstrainedBasisFullWeighting,
		HTFactorsApplied:                  false,
		BaselineLoss:                      baseline.Sum(),
		BaselineLossFinite:                true,
		BaselineComponents:                baseline,
		BaselineActivation:                activation,
		Q3GradientSHA256:                  strings.Repeat("2", 64),
		ProtectedGradientNames:            []string{"q3_order_guard", "q3_score_distill", "q5_order_guard", "q5_score_distill", "nf_boundary_guard"},
		ProtectedGradientMatrixSHA256:     strings.Repeat("3", 64),
		BasisSHA256:                       basisSHA,
		DirectionScheduleSHA256:           scheduleSHA,
		CanonicalObjectiveContractSHA256:  strings.Repeat("4", 64),
		CanonicalEligibilityPolicySHA256:  policySHA,
		CanonicalV7R6ReceiptFileSHA256:    AOQTV8CanonicalV7R6ReceiptFileSHA256,
		CanonicalV7R6TerminalReceiptBound: true,
		DenseInvariantStatus:              AOQTV8ConstrainedBasisDenseInvariantDeferred,
		DenseMaxAbsDeltaTolerance:         policy.DenseMaxAbsDeltaTolerance,
		DenseInvariantDeferred:            true,
		GradientEvidenceDeferred:          false,
		FullVerificationCompleted:         true,
		FeasibleEndpointCount:             1,
		SelectedEndpointOrdinals:          []int{1},
		EndpointHashChain:                 chain,
		EndpointHashChainTailSHA256:       tail,
		Basis:                             basis,
		Endpoints:                         endpoints,
	}
}

func aoqtV8TestBasis(t *testing.T) []AOQTSidecarV8BasisVector {
	t.Helper()
	out := make([]AOQTSidecarV8BasisVector, 0, AOQTV8ConstrainedBasisRankCount)
	for i, rank := range AOQTV8ConstrainedBasisRanks() {
		direction := make([]float32, AOQTV8ConstrainedBasisExpectedAngleSize)
		indices := make([]int, 0, rank)
		signs := make([]int, 0, rank)
		for j := 0; j < rank; j++ {
			index := i*64 + j
			direction[index] = 1
			indices = append(indices, index)
			signs = append(signs, 1)
		}
		sha, err := aoqtActualDirectionVectorSHA256(direction)
		if err != nil {
			t.Fatalf("direction sha: %v", err)
		}
		out = append(out, AOQTSidecarV8BasisVector{Ordinal: i + 1, Kind: "sequential_projected_sparse_q3_support", Rank: rank, Direction: direction, DirectionSHA256: sha, SelectedIndices: indices, SelectedSigns: signs, SourceGradientSHA256: strings.Repeat("2", 64)})
	}
	return out
}

func aoqtV8TestEndpoint(t *testing.T, ordinal int, basis AOQTSidecarV8BasisVector, magnitude float32, sign int, baseline, candidate AOQTSidecarObjectiveComponents, activation AOQTSidecarObjectiveActivation) AOQTSidecarV8ConstrainedBasisEndpoint {
	t.Helper()
	direction := make([]float32, len(basis.Direction))
	for i, value := range basis.Direction {
		direction[i] = value * magnitude * float32(sign)
	}
	sha, err := aoqtActualDirectionVectorSHA256(direction)
	if err != nil {
		t.Fatalf("endpoint direction sha: %v", err)
	}
	delta := aoqtObjectiveComponentDelta(baseline, candidate)
	totalDelta := candidate.Sum() - baseline.Sum()
	policy := aoqtV7R6CanonicalEligibilityPolicy()
	decision := aoqtEvaluateTransactionalStepWithPolicy(
		aoqtStepEvaluation{loss: baseline.Sum(), activation: activation, components: baseline},
		aoqtStepEvaluation{loss: candidate.Sum(), activation: activation, components: candidate},
		AOQTSidecarRowWeights{Q3Gain: 1, Q3OrderGuard: 1, Q3ScoreDistill: 1, Q5OrderGuard: 1, Q5ScoreDistill: 1, NFBoundaryGuard: 1},
		policy,
	)
	reason := string(decision.reason)
	if decision.accepted {
		reason = aoqtProposalAcceptedReason
	}
	return AOQTSidecarV8ConstrainedBasisEndpoint{
		Ordinal: ordinal, BasisOrdinal: basis.Ordinal, Rank: basis.Rank, Magnitude: magnitude, GlobalSign: sign,
		RequestedDirectionSHA256: sha, ActualDirectionSHA256: sha, ActualDirectionMaxAbs: aoqtMaxAbsFloat32(direction), ActualDirection: direction, AngleMoved: true,
		BaselineLoss: baseline.Sum(), BaselineLossFinite: true, BaselineComponents: baseline, BaselineComponentsFinite: true,
		CandidateLoss: candidate.Sum(), CandidateLossFinite: true, CandidateComponents: candidate, CandidateComponentsFinite: true,
		ComponentDeltas: delta, ComponentDeltasFinite: true, CandidateActivation: activation, TotalDelta: totalDelta, TotalDeltaFinite: true,
		Q3GainEligible:      candidate.Q3Gain < baseline.Q3Gain-aoqtTransactionalQ3ImprovementMinMagnitude,
		ProtectedEligible:   len(aoqtComponentRegressions(baseline, candidate, policy)) == 0,
		TransactionEligible: decision.accepted,
		Reason:              reason,
	}
}

func aoqtV8TestEndpointChain(t *testing.T, endpoints []AOQTSidecarV8ConstrainedBasisEndpoint) (string, string) {
	t.Helper()
	data, tail, err := aoqtV8EndpointHashChain(endpoints)
	if err != nil {
		t.Fatalf("endpoint chain: %v", err)
	}
	return string(data), tail
}

func aoqtV8RealBindingReceiptOrSkip(t *testing.T) AOQTSidecarV8ConstrainedBasisScreenReceipt {
	t.Helper()
	path := ".tiller/scratch/codex/aoqt-v7-r6-real/fold-0-r4/progressive-screen.receipt.json"
	data, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("canonical V7-r6 receipt unavailable: %v", err)
	}
	if got := sha256BytesAOQT(data); got != AOQTV8CanonicalV7R6ReceiptFileSHA256 {
		t.Skipf("canonical V7-r6 receipt hash = %s, want %s", got, AOQTV8CanonicalV7R6ReceiptFileSHA256)
	}
	var v7 AOQTSidecarV7R6ProgressiveScreenReceipt
	if err := strictUnmarshalAOQT(data, &v7); err != nil {
		t.Fatalf("decode V7-r6 receipt: %v", err)
	}
	if err := v7.ValidateBound(); err != nil {
		t.Skipf("canonical V7-r6 bound validation unavailable: %v", err)
	}
	manifestData, err := os.ReadFile(v7.Binding.IOReport.ManifestPath)
	if err != nil {
		t.Fatalf("read bound manifest: %v", err)
	}
	var manifest AOQTSidecarCalibrationManifest
	if err := strictUnmarshalAOQT(manifestData, &manifest); err != nil {
		t.Fatalf("decode bound manifest: %v", err)
	}
	rowsData, err := os.ReadFile(v7.Binding.IOReport.RowsJSONLPath)
	if err != nil {
		t.Fatalf("read bound rows: %v", err)
	}
	rows, err := parseAOQTCalibrationRowsJSONL(rowsData, v7.Binding.IOReport.RowsJSONLPath)
	if err != nil {
		t.Fatalf("decode bound rows: %v", err)
	}
	set := AOQTSidecarCalibrationSet{Manifest: manifest, Rows: rows}
	receipt := aoqtV8TestReceipt(t)
	receipt.FullFoldRowCount = v7.FullVerification.RowCount
	receipt.FullFoldWeightingVersion = v7.FullVerification.WeightingVersion
	receipt.HTFactorsApplied = v7.FullVerification.HTFactorsApplied
	receipt.CanonicalObjectiveContractSHA256 = v7.CanonicalObjectiveContractSHA256
	receipt.CanonicalEligibilityPolicySHA256 = v7.CanonicalEligibilityPolicySHA256
	selectionSeed, err := aoqtV8SelectionSeed(set, v7.Binding.FoldID, receipt.Q3GradientSHA256, receipt.ProtectedGradientMatrixSHA256, receipt.BasisSHA256, AOQTV8CanonicalV7R6ReceiptFileSHA256)
	if err != nil {
		t.Fatalf("selection seed: %v", err)
	}
	receipt.SelectionSeedSHA256 = selectionSeed
	receipt.Binding = AOQTSidecarV8ConstrainedBasisScreenBinding{
		Inputs:                         v7.Binding.Inputs,
		IOReport:                       v7.Binding.IOReport,
		PreflightPath:                  v7.Binding.PreflightPath,
		PreflightSHA256:                v7.Binding.PreflightSHA256,
		SplitBinding:                   cloneAOQTDevSplitBinding(v7.Binding.SplitBinding),
		FoldID:                         v7.Binding.FoldID,
		CanonicalR4ReceiptPath:         v7.Binding.CanonicalR4ReceiptPath,
		CanonicalR4ReceiptSHA256:       v7.Binding.CanonicalR4ReceiptSHA256,
		CanonicalR4ReceiptFileSHA256:   v7.Binding.CanonicalR4ReceiptFileSHA256,
		CanonicalR5ReceiptPath:         v7.Binding.CanonicalR5ReceiptPath,
		CanonicalR5ReceiptSHA256:       v7.Binding.CanonicalR5ReceiptSHA256,
		CanonicalR5ReceiptFileSHA256:   v7.Binding.CanonicalR5ReceiptFileSHA256,
		CanonicalV7R6ReceiptPath:       path,
		CanonicalV7R6ReceiptFileSHA256: AOQTV8CanonicalV7R6ReceiptFileSHA256,
	}
	return receipt
}
