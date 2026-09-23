package eosruntime

import (
	"reflect"
	"strings"
	"testing"
)

func TestAOQTV7R6ProgressiveScreenHonestReceiptValidates(t *testing.T) {
	if err := aoqtV7R6TestReceipt(t).Validate(); err != nil {
		t.Fatalf("honest r6 receipt validation: %v", err)
	}
}

func TestAOQTV7R6ProgressiveScreenExportedRankUsesJointBudgetPolicy(t *testing.T) {
	jointBudgeted := aoqtV7R6TestEvidence(t, AOQTSidecarObjectiveComponents{Q3Gain: 0.4, Q3OrderGuard: 1, Q3ScoreDistill: 1.00005, Q5OrderGuard: 1, Q5ScoreDistill: 1, NFBoundaryGuard: 1}, 8)
	zeroBudgeted := aoqtV7R6TestEvidence(t, AOQTSidecarObjectiveComponents{Q3Gain: 0.9, Q3OrderGuard: 1, Q3ScoreDistill: 1, Q5OrderGuard: 1, Q5ScoreDistill: 1, NFBoundaryGuard: 1}, 39)
	endpoints := []AOQTSidecarV7R6ProgressiveScreenEndpoint{{Ordinal: 8, S512: jointBudgeted}, {Ordinal: 39, S512: zeroBudgeted}}
	got, err := RankAOQTV7R6ProgressiveScreenEndpoints(endpoints, "S512")
	if err != nil || !reflect.DeepEqual(got, []int{8, 39}) {
		t.Fatalf("exported rank = %v/%v, want joint-budget-feasible endpoint first", got, err)
	}
	defaultRank, err := aoqtV7R6RankEndpointsWithPolicy(endpoints, "S512", AOQTSidecarDefaultCandidateEligibilityPolicy())
	if err != nil || !reflect.DeepEqual(defaultRank, []int{39, 8}) {
		t.Fatalf("legacy-default rank fixture = %v/%v, want opposite order", defaultRank, err)
	}
}

func TestAOQTV7R6ProgressiveScreenRejectsForgedRankingAndSelectedPrefix(t *testing.T) {
	receipt := aoqtV7R6TestReceipt(t)
	receipt.StageS1024.Ranking = append([]int(nil), receipt.StageS1024.Ranking...)
	receipt.StageS1024.Ranking[0], receipt.StageS1024.Ranking[1] = receipt.StageS1024.Ranking[1], receipt.StageS1024.Ranking[0]
	var err error
	receipt.StageS1024.RankingSHA256, err = aoqtV7R6RankingSHA(receipt.StageS1024.Ranking)
	if err != nil {
		t.Fatalf("forged ranking sha: %v", err)
	}
	rankAt := map[int]int{}
	for i, ordinal := range receipt.StageS1024.Ranking {
		rankAt[ordinal] = i + 1
	}
	for i := range receipt.Endpoints {
		receipt.Endpoints[i].RankS1024 = rankAt[receipt.Endpoints[i].Ordinal]
	}
	receipt.SelectedEndpointOrdinals = append([]int(nil), receipt.StageS1024.Ranking[:AOQTV7R6ProgressiveScreenSelectedK]...)
	receipt.FullVerification.EndpointOrdinals = append([]int(nil), receipt.SelectedEndpointOrdinals...)
	byOrdinal := map[int]AOQTSidecarV7R6ProgressiveScreenEndpoint{}
	for _, endpoint := range receipt.Endpoints {
		byOrdinal[endpoint.Ordinal] = endpoint
	}
	receipt.FullVerification.Endpoints = receipt.FullVerification.Endpoints[:0]
	for _, ordinal := range receipt.SelectedEndpointOrdinals {
		receipt.FullVerification.Endpoints = append(receipt.FullVerification.Endpoints, AOQTSidecarV7R6ProgressiveScreenFullObjectiveEndpoint{Ordinal: ordinal, Evidence: byOrdinal[ordinal].S1024, Feasible: true, DenseGateStatus: AOQTV7R6ProgressiveScreenDenseGateStatus})
	}
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "recomputed endpoint evidence") {
		t.Fatalf("forged self-consistent S1024 ranking error = %v, want recomputed ranking rejection", err)
	}
}

func TestAOQTV7R6ProgressiveScreenRejectsForgedS512Ranking(t *testing.T) {
	receipt := aoqtV7R6TestReceipt(t)
	receipt.StageS512.Ranking = append([]int(nil), receipt.StageS512.Ranking...)
	receipt.StageS512.Ranking[0], receipt.StageS512.Ranking[1] = receipt.StageS512.Ranking[1], receipt.StageS512.Ranking[0]
	var err error
	receipt.StageS512.RankingSHA256, err = aoqtV7R6RankingSHA(receipt.StageS512.Ranking)
	if err != nil {
		t.Fatalf("forged ranking sha: %v", err)
	}
	rankAt := map[int]int{}
	for i, ordinal := range receipt.StageS512.Ranking {
		rankAt[ordinal] = i + 1
	}
	for i := range receipt.Endpoints {
		receipt.Endpoints[i].RankS512 = rankAt[receipt.Endpoints[i].Ordinal]
	}
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "recomputed endpoint evidence") {
		t.Fatalf("forged self-consistent S512 ranking error = %v, want recomputed ranking rejection", err)
	}
}

func TestAOQTV7R6ProgressiveScreenRejectsBoundObjectiveContractHashTamper(t *testing.T) {
	receipt := aoqtV7R6TestReceipt(t)
	manifest := aoqtV7R6TestManifest()
	contractSHA, err := aoqtActualCoordinateObjectiveContractSHA256(manifest.ObjectiveContract)
	if err != nil {
		t.Fatalf("contract sha: %v", err)
	}
	policySHA, err := aoqtActualCoordinatePolicySHA256(aoqtV7R6CanonicalEligibilityPolicy())
	if err != nil {
		t.Fatalf("policy sha: %v", err)
	}
	receipt.CanonicalObjectiveContractSHA256 = contractSHA
	receipt.CanonicalEligibilityPolicySHA256 = policySHA
	r4 := AOQTSidecarActualCoordinateProbeReceipt{ObjectiveContractSHA256: contractSHA, EligibilityPolicySHA256: policySHA}
	r4.Binding.ObjectiveContractSHA256 = contractSHA
	r4.Binding.EligibilityPolicySHA256 = policySHA
	r5 := AOQTSidecarActualCoordinateTrainReceipt{}
	r5.Binding.ObjectiveContractSHA256 = contractSHA
	r5.Binding.EligibilityPolicySHA256 = policySHA
	if err := validateAOQTV7R6BoundContractHashes(receipt, manifest, r4, r5); err != nil {
		t.Fatalf("honest contract binding: %v", err)
	}
	receipt.CanonicalObjectiveContractSHA256 = strings.Repeat("9", 64)
	if err := validateAOQTV7R6BoundContractHashes(receipt, manifest, r4, r5); err == nil || !strings.Contains(err.Error(), "objective contract hash") {
		t.Fatalf("tampered objective contract error = %v, want objective hash rejection", err)
	}
}

func TestAOQTV7R6ProgressiveScreenRejectsForgedPolicyViolations(t *testing.T) {
	cases := []struct {
		name       string
		candidate  AOQTSidecarObjectiveComponents
		wantReason string
	}{
		{"q5 order", AOQTSidecarObjectiveComponents{Q3Gain: 0.4, Q3OrderGuard: 0.9, Q3ScoreDistill: 0.9, Q5OrderGuard: 1.2, Q5ScoreDistill: 0.9, NFBoundaryGuard: 0.9}, string(aoqtRejectionComponentRegression)},
		{"nf boundary", AOQTSidecarObjectiveComponents{Q3Gain: 0.4, Q3OrderGuard: 0.9, Q3ScoreDistill: 0.9, Q5OrderGuard: 0.9, Q5ScoreDistill: 0.9, NFBoundaryGuard: 1.2}, string(aoqtRejectionComponentRegression)},
		{"total loss", AOQTSidecarObjectiveComponents{Q3Gain: 0.9, Q3OrderGuard: 1.1, Q3ScoreDistill: 1.1, Q5OrderGuard: 1.1, Q5ScoreDistill: 1.1, NFBoundaryGuard: 1.1}, string(aoqtRejectionLossIncrease)},
		{"no q3 gain", AOQTSidecarObjectiveComponents{Q3Gain: 1, Q3OrderGuard: 0.8, Q3ScoreDistill: 0.8, Q5OrderGuard: 0.8, Q5ScoreDistill: 0.8, NFBoundaryGuard: 0.8}, string(aoqtRejectionNoQ3GainImprovement)},
		{"score budget", AOQTSidecarObjectiveComponents{Q3Gain: 0.4, Q3OrderGuard: 0.9, Q3ScoreDistill: 1.001, Q5OrderGuard: 0.9, Q5ScoreDistill: 0.9, NFBoundaryGuard: 0.9}, string(aoqtRejectionComponentRegression)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			evidence := aoqtV7R6TestEvidence(t, tc.candidate, 0)
			if evidence.TransactionEligible || evidence.Reason != tc.wantReason {
				t.Fatalf("fixture transaction=%t reason=%q, want false/%q", evidence.TransactionEligible, evidence.Reason, tc.wantReason)
			}
			evidence.Q3GainEligible = true
			evidence.ProtectedEligible = true
			evidence.TransactionEligible = true
			evidence.Reason = aoqtProposalAcceptedReason
			if _, err := validateAOQTV7R6StageEndpointEvidence(evidence, aoqtV7R6CanonicalEligibilityPolicy(), evidence.ActualDirectionSHA256, "forged "+tc.name); err == nil {
				t.Fatalf("forged %s policy violation unexpectedly validated", tc.name)
			}
		})
	}
}

func TestAOQTV7R6ProgressiveScreenRejectsForgedFullFeasibleFlag(t *testing.T) {
	receipt := aoqtV7R6TestReceipt(t)
	receipt.FullVerification.Endpoints = append([]AOQTSidecarV7R6ProgressiveScreenFullObjectiveEndpoint(nil), receipt.FullVerification.Endpoints...)
	receipt.FullVerification.Endpoints[0].Feasible = false
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "feasible flag") {
		t.Fatalf("forged full feasible flag error = %v, want recomputed feasible rejection", err)
	}
}

func TestAOQTV7R6ProgressiveScreenRejectsFullDirectionAndMetadataTamperAgainstR4(t *testing.T) {
	receipt := aoqtV7R6TestReceipt(t)
	r4 := aoqtV7R6TestR4ForReceipt(receipt)
	if err := validateAOQTV7R6ReceiptAgainstCanonicalR4Endpoints(receipt, r4); err != nil {
		t.Fatalf("honest r4 binding: %v", err)
	}
	tampered := receipt
	tampered.FullVerification.Endpoints = append([]AOQTSidecarV7R6ProgressiveScreenFullObjectiveEndpoint(nil), receipt.FullVerification.Endpoints...)
	tampered.FullVerification.Endpoints[0].Evidence.ActualDirectionSHA256 = strings.Repeat("b", 64)
	if err := validateAOQTV7R6ReceiptAgainstCanonicalR4Endpoints(tampered, r4); err == nil {
		t.Fatal("forged full actual direction hash unexpectedly matched r4")
	}
	tampered = receipt
	tampered.FullVerification.Endpoints = append([]AOQTSidecarV7R6ProgressiveScreenFullObjectiveEndpoint(nil), receipt.FullVerification.Endpoints...)
	tampered.FullVerification.Endpoints[0].Evidence.ActualDirection = append([]float32(nil), receipt.FullVerification.Endpoints[0].Evidence.ActualDirection...)
	tampered.FullVerification.Endpoints[0].Evidence.ActualDirection[0] += 0.001
	if err := validateAOQTV7R6ReceiptAgainstCanonicalR4Endpoints(tampered, r4); err == nil {
		t.Fatal("forged full actual direction vector unexpectedly matched r4")
	}
	tampered = receipt
	tampered.FullVerification.Endpoints = append([]AOQTSidecarV7R6ProgressiveScreenFullObjectiveEndpoint(nil), receipt.FullVerification.Endpoints...)
	tampered.FullVerification.Endpoints[0].Evidence.ActualDirectionMaxAbs += 0.001
	if err := validateAOQTV7R6ReceiptAgainstCanonicalR4Endpoints(tampered, r4); err == nil {
		t.Fatal("forged full actual direction maxabs unexpectedly matched r4")
	}
	tampered = receipt
	tampered.Endpoints = append([]AOQTSidecarV7R6ProgressiveScreenEndpoint(nil), receipt.Endpoints...)
	tampered.Endpoints[0].CoordinateIndex++
	if err := validateAOQTV7R6ReceiptAgainstCanonicalR4Endpoints(tampered, r4); err == nil {
		t.Fatal("forged selected endpoint metadata unexpectedly matched r4")
	}
}

func TestAOQTV7R6ProgressiveScreenSamplesAndRowClassification(t *testing.T) {
	rows := aoqtV7R6SemanticTestRows()
	samples, err := ConstructAOQTV7R6ProgressiveScreenSamples(rows, "sample-seed")
	if err != nil {
		t.Fatalf("construct samples: %v", err)
	}
	if samples.S512.RowCount != AOQTV7R6ProgressiveScreenS512 || samples.S1024.RowCount != AOQTV7R6ProgressiveScreenS1024 || !isStrictStringSubset(samples.S512.RowIDs, samples.S1024.RowIDs) {
		t.Fatalf("sample shape = %d/%d", samples.S512.RowCount, samples.S1024.RowCount)
	}
	for i, spec := range aoqtV7R6ProgressiveScreenStratumSpecs {
		row := rows[0]
		row.RowID = spec.rowPrefix + "qid-real-form"
		row.Dataset = spec.dataset
		row.GuardClass = spec.guard
		row.CandidateSources = []string{spec.source, spec.source}
		row.Weights.NFBoundaryGuard = 0
		if spec.boundary {
			row.Weights.NFBoundaryGuard = 0.75
		}
		if got, err := aoqtV7R6ClassifyRow(row); err != nil || got != i {
			t.Fatalf("real row form %q classified as %d/%v, want %d", row.RowID, got, err, i)
		}
		row.Dataset = "wrong-dataset"
		if _, err := aoqtV7R6ClassifyRow(row); err == nil {
			t.Fatalf("metadata-forged row %q unexpectedly classified", row.RowID)
		}
	}
}

func aoqtV7R6SemanticTestRows() []AOQTSidecarCalibrationRow {
	rows := make([]AOQTSidecarCalibrationRow, 0, AOQTV7R6ProgressiveScreenPopulationTotal)
	for i, spec := range aoqtV7R6ProgressiveScreenStratumSpecs {
		for j := 0; j < spec.population; j++ {
			row := AOQTSidecarCalibrationRow{
				RowID:            spec.rowPrefix + "qid-" + intStringAOQT(i) + "-" + intStringAOQT(j),
				Dataset:          spec.dataset,
				CandidateSources: []string{spec.source, spec.source},
				GuardClass:       spec.guard,
				Weights:          AOQTSidecarRowWeights{Q3Gain: 1, Q3OrderGuard: 1, Q3ScoreDistill: 1, Q5OrderGuard: 1, Q5ScoreDistill: 1},
			}
			if spec.boundary {
				row.Weights.NFBoundaryGuard = 0.75
			}
			rows = append(rows, row)
		}
	}
	return rows
}

func intStringAOQT(value int) string {
	if value == 0 {
		return "0"
	}
	var buf [32]byte
	i := len(buf)
	for value > 0 {
		i--
		buf[i] = byte('0' + value%10)
		value /= 10
	}
	return string(buf[i:])
}

func aoqtV7R6TestReceipt(t *testing.T) AOQTSidecarV7R6ProgressiveScreenReceipt {
	t.Helper()
	samples, err := ConstructAOQTV7R6ProgressiveScreenSamples(aoqtV7R6SemanticTestRows(), "receipt-seed")
	if err != nil {
		t.Fatalf("construct receipt samples: %v", err)
	}
	policy := aoqtV7R6CanonicalEligibilityPolicy()
	policySHA, err := aoqtActualCoordinatePolicySHA256(policy)
	if err != nil {
		t.Fatalf("policy sha: %v", err)
	}
	endpoints := make([]AOQTSidecarV7R6ProgressiveScreenEndpoint, 0, AOQTV7R6ProgressiveScreenEndpointCount)
	ranking := AOQTV7R6CanonicalEligibleEndpointOrdinals()
	for i, ordinal := range ranking {
		endpoints = append(endpoints, aoqtV7R6TestScreenEndpoint(t, ordinal, i))
		endpoints[i].RankS512 = i + 1
		endpoints[i].RankS1024 = i + 1
	}
	rankingSHA, err := aoqtV7R6RankingSHA(ranking)
	if err != nil {
		t.Fatalf("ranking sha: %v", err)
	}
	selected := append([]int(nil), ranking[:AOQTV7R6ProgressiveScreenSelectedK]...)
	full := AOQTSidecarV7R6ProgressiveScreenFullVerification{
		Enabled: true, Completed: true, ObjectiveEvaluationCount: AOQTV7R6ProgressiveScreenFullObjectiveEvaluations,
		RowCount: AOQTV7R6ProgressiveScreenPopulationTotal, WeightingVersion: AOQTV7R6ProgressiveScreenFullWeightingVersion,
		EndpointOrdinals: append([]int(nil), selected...), BaselineLoss: 6, BaselineLossFinite: true,
		BaselineComponents: aoqtV7R6TestBaselineComponents(), BaselineActivation: aoqtV7R6TestActivation(),
		DenseGateStatus: AOQTV7R6ProgressiveScreenDenseGateStatus, DenseGateDeferred: true,
	}
	for i, ordinal := range selected {
		full.Endpoints = append(full.Endpoints, AOQTSidecarV7R6ProgressiveScreenFullObjectiveEndpoint{Ordinal: ordinal, Evidence: endpoints[i].S1024, Feasible: true, DenseGateStatus: AOQTV7R6ProgressiveScreenDenseGateStatus})
	}
	return AOQTSidecarV7R6ProgressiveScreenReceipt{
		Schema: AOQTSidecarV7R6ProgressiveScreenSchema, Mode: AOQTSidecarOptimizerModeV7R6ProgressiveScreen, Required: true, ProbeOnly: true, Passed: true,
		StratificationVersion: AOQTV7R6ProgressiveScreenStratificationVersion, WeightingVersion: AOQTV7R6ProgressiveScreenWeightingVersion, ScheduleVersion: AOQTV7R6ProgressiveScreenScheduleVersion, RankingVersion: AOQTV7R6ProgressiveScreenRankingVersion, PolicyVersion: AOQTV7R6ProgressiveScreenPolicyVersion,
		SelectionSeedSHA256: strings.Repeat("1", 64), R4EndpointSetSHA256: strings.Repeat("2", 64), R4EligibleEndpointOrdinals: AOQTV7R6CanonicalEligibleEndpointOrdinals(), R5ReceiptSHA256: strings.Repeat("3", 64),
		FoldInputRowIDSHA256: strings.Repeat("4", 64), FoldInputRowsSHA256: strings.Repeat("5", 64), CanonicalR4ScheduleSHA256: strings.Repeat("6", 64), CanonicalR4RankingSHA256: strings.Repeat("7", 64), CanonicalR4PolicySHA256: policySHA,
		CanonicalObjectiveContractSHA256: strings.Repeat("8", 64), CanonicalEligibilityPolicySHA256: policySHA,
		StageS512: aoqtV7R6TestStage(t, "S512", samples.S512, ranking, rankingSHA), StageS1024: aoqtV7R6TestStage(t, "S1024", samples.S1024, ranking, rankingSHA),
		FullVerification: full, Endpoints: endpoints, SelectedEndpointOrdinals: selected, SelectionK: AOQTV7R6ProgressiveScreenSelectedK, ObjectiveEvaluationCount: AOQTV7R6ProgressiveScreenObjectiveEvaluations,
		SelectionFeasible: true, SelectionFeasibleCount: AOQTV7R6ProgressiveScreenSelectedK, FullVerificationFeasible: true, FullVerificationFeasibleCount: AOQTV7R6ProgressiveScreenSelectedK,
		DenseGateStatus: AOQTV7R6ProgressiveScreenDenseGateStatus, DenseGateDeferred: true, DenseMaxAbsDeltaTolerance: policy.DenseMaxAbsDeltaTolerance,
	}
}

func aoqtV7R6TestStage(t *testing.T, name string, sample AOQTSidecarV7R6ProgressiveScreenSample, ranking []int, rankingSHA string) AOQTSidecarV7R6ProgressiveScreenStage {
	t.Helper()
	weights, err := AOQTV7R6ProgressiveScreenHTWeights(sample)
	if err != nil {
		t.Fatalf("%s weights: %v", name, err)
	}
	return AOQTSidecarV7R6ProgressiveScreenStage{Name: name, Sample: sample, Weights: weights, BaselineLoss: 6, BaselineLossFinite: true, BaselineComponents: aoqtV7R6TestBaselineComponents(), BaselineActivation: aoqtV7R6TestActivation(), BaselineWeightedActivation: aoqtV7R6TestWeightedActivation(), ObjectiveEvaluationCount: AOQTV7R6ProgressiveScreenStageObjectiveEvaluations, EndpointOrdinals: AOQTV7R6CanonicalEligibleEndpointOrdinals(), Ranking: append([]int(nil), ranking...), RankingSHA256: rankingSHA}
}

func aoqtV7R6TestScreenEndpoint(t *testing.T, ordinal, coord int) AOQTSidecarV7R6ProgressiveScreenEndpoint {
	t.Helper()
	evidence := aoqtV7R6TestEvidence(t, AOQTSidecarObjectiveComponents{Q3Gain: 0.5, Q3OrderGuard: 1, Q3ScoreDistill: 1, Q5OrderGuard: 1, Q5ScoreDistill: 1, NFBoundaryGuard: 1}, coord)
	requestedSHA, err := aoqtActualCoordinateVectorSHA256(evidence.ActualDirection)
	if err != nil {
		t.Fatalf("requested direction sha: %v", err)
	}
	return AOQTSidecarV7R6ProgressiveScreenEndpoint{Ordinal: ordinal, Kind: "coordinate", CoordinateIndex: coord, AnalyticDirection: 1, Indices: []int{coord}, Directions: []int{1}, GlobalSign: 1, Magnitude: evidence.ActualDirectionMaxAbs, RequestedDirectionSHA256: requestedSHA, CanonicalActualDirectionSHA256: evidence.ActualDirectionSHA256, S512: evidence, S1024: evidence}
}

func aoqtV7R6TestEvidence(t *testing.T, candidate AOQTSidecarObjectiveComponents, coord int) AOQTSidecarV7R6ProgressiveScreenStageEndpoint {
	t.Helper()
	actual := make([]float32, AOQTSidecarAngleCount)
	actual[coord] = 0.01
	actualSHA, err := aoqtActualCoordinateVectorSHA256(actual)
	if err != nil {
		t.Fatalf("actual direction sha: %v", err)
	}
	evidence := AOQTSidecarV7R6ProgressiveScreenStageEndpoint{BaselineLoss: 6, BaselineLossFinite: true, CandidateLoss: candidate.Sum(), CandidateLossFinite: true, BaselineComponents: aoqtV7R6TestBaselineComponents(), CandidateComponents: candidate, ComponentDeltas: aoqtObjectiveComponentDelta(aoqtV7R6TestBaselineComponents(), candidate), BaselineActivation: aoqtV7R6TestActivation(), CandidateActivation: aoqtV7R6TestActivation(), TotalDelta: candidate.Sum() - 6, TotalDeltaFinite: true, ActualDirectionSHA256: actualSHA, ActualDirectionMaxAbs: 0.01, ActualDirection: actual, AngleMoved: true}
	policy, err := aoqtV7R6RecomputeEndpointPolicy(evidence, aoqtV7R6CanonicalEligibilityPolicy())
	if err != nil {
		t.Fatalf("policy recompute: %v", err)
	}
	evidence.Q3GainEligible, evidence.ProtectedEligible, evidence.TransactionEligible, evidence.Reason = policy.Q3GainEligible, policy.ProtectedEligible, policy.TransactionEligible, policy.Reason
	return evidence
}

func aoqtV7R6TestBaselineComponents() AOQTSidecarObjectiveComponents {
	return AOQTSidecarObjectiveComponents{Q3Gain: 1, Q3OrderGuard: 1, Q3ScoreDistill: 1, Q5OrderGuard: 1, Q5ScoreDistill: 1, NFBoundaryGuard: 1}
}

func aoqtV7R6TestActivation() AOQTSidecarObjectiveActivation {
	return AOQTSidecarObjectiveActivation{Q3GainEligiblePairs: 1, Q3GainContributingPairs: 1, Q3OrderGuardPairs: 1, Q3OrderGuardContributing: 1, Q3ScoreDistillCount: 1, Q5OrderGuardPairs: 1, Q5OrderGuardContributing: 1, Q5ScoreDistillCount: 1, NFBoundaryGuardPairs: 1, NFBoundaryGuardContributing: 1}
}

func aoqtV7R6TestWeightedActivation() AOQTSidecarV7R6WeightedActivation {
	return AOQTSidecarV7R6WeightedActivation{Q3GainEligiblePairs: 1, Q3GainContributingPairs: 1, Q3OrderGuardPairs: 1, Q3OrderGuardContributing: 1, Q3ScoreDistillCount: 1, Q5OrderGuardPairs: 1, Q5OrderGuardContributing: 1, Q5ScoreDistillCount: 1, NFBoundaryGuardPairs: 1, NFBoundaryGuardContributing: 1}
}

func aoqtV7R6TestManifest() AOQTSidecarCalibrationManifest {
	policy := aoqtV7R6CanonicalEligibilityPolicy()
	return AOQTSidecarCalibrationManifest{ObjectiveContract: AOQTSidecarObjectiveContract{}, TrainingContract: AOQTSidecarScoreDistillJointBudgetV1Contract, CandidateEligibilityPolicy: &policy}
}

func aoqtV7R6TestR4ForReceipt(receipt AOQTSidecarV7R6ProgressiveScreenReceipt) AOQTSidecarActualCoordinateProbeReceipt {
	r4 := AOQTSidecarActualCoordinateProbeReceipt{Endpoints: make([]AOQTSidecarActualCoordinateProbeEndpoint, 384)}
	for i := range r4.Endpoints {
		r4.Endpoints[i].Ordinal = i + 1
	}
	for _, endpoint := range receipt.Endpoints {
		r4.Endpoints[endpoint.Ordinal-1] = AOQTSidecarActualCoordinateProbeEndpoint{Ordinal: endpoint.Ordinal, Kind: endpoint.Kind, CoordinateIndex: endpoint.CoordinateIndex, AnalyticDirection: endpoint.AnalyticDirection, Indices: append([]int(nil), endpoint.Indices...), Directions: append([]int(nil), endpoint.Directions...), GlobalSign: endpoint.GlobalSign, Magnitude: endpoint.Magnitude, RequestedDirectionSHA256: endpoint.RequestedDirectionSHA256, ActualDirectionSHA256: endpoint.CanonicalActualDirectionSHA256, ActualDirectionMaxAbs: endpoint.S1024.ActualDirectionMaxAbs, ActualDirection: append([]float32(nil), endpoint.S1024.ActualDirection...)}
	}
	return r4
}
