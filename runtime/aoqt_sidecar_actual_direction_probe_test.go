package eosruntime

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func newAOQTV7R3ProbeTestTrainer(t *testing.T, seed int64) *AOQTSidecarTrainer {
	t.Helper()
	trainer, err := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{
		PairingSeed:                             seed,
		WorkplanSeed:                            seed + 1000,
		OptimizerMode:                           AOQTSidecarOptimizerModeV7R3DevActualDirection,
		ActualDirectionProbeRequired:            true,
		ActualDirectionProbeFoldID:              "fold-0",
		ActualDirectionProbeSplitManifestSHA256: hex64("probe-split"),
		MaxSteps:                                1,
		LearningRate:                            AOQTV7R3ActualDirectionProbeLearningRate,
		CaptureProposalReceipts:                 true,
	})
	if err != nil {
		t.Fatalf("new V7-r3 trainer: %v", err)
	}
	return trainer
}

func TestAOQTV7R3ActualDirectionProbeDeterministic132Endpoints(t *testing.T) {
	set := eightRowAOQTCalibrationSetForProbe(t, 701)
	objective := syntheticForwardConsistencyObjectiveForSet(t, set, false)
	a := newAOQTV7R3ProbeTestTrainer(t, 701)
	b := newAOQTV7R3ProbeTestTrainer(t, 701)
	receiptA, err := a.runActualDirectionProbe(set, objective)
	if err != nil {
		t.Fatalf("actual-direction probe: %v", err)
	}
	receiptB, err := b.runActualDirectionProbe(set, objective)
	if err != nil {
		t.Fatalf("repeat actual-direction probe: %v", err)
	}
	if !receiptA.Passed || !receiptB.Passed {
		t.Fatalf("probe receipts are not passed: %+v / %+v", receiptA, receiptB)
	}
	if !reflect.DeepEqual(receiptA, receiptB) {
		t.Fatalf("actual-direction probe receipts are not deterministic")
	}
	if receiptA.EndpointCount != AOQTV7R3ActualDirectionProbeEndpointCount || len(receiptA.Endpoints) != 132 {
		t.Fatalf("endpoint count = %d/%d, want exactly 132", receiptA.EndpointCount, len(receiptA.Endpoints))
	}
	if receiptA.FullTransactionCandidateCount > AOQTV7R3ActualDirectionProbeMaxFullCandidates {
		t.Fatalf("full candidate count = %d, want <= 2", receiptA.FullTransactionCandidateCount)
	}
	if err := receiptA.Validate(); err != nil {
		t.Fatalf("validate actual-direction receipt: %v", err)
	}
	for _, endpoint := range receiptA.Endpoints {
		if endpoint.BaselineLoss != receiptA.Endpoints[0].BaselineLoss || endpoint.BaselineComponents != receiptA.Endpoints[0].BaselineComponents {
			t.Fatalf("endpoint %d baseline differs from endpoint 1", endpoint.Ordinal)
		}
	}
	if got, err := receiptA.SHA256(); err != nil || got == "" {
		t.Fatalf("actual-direction receipt hash = %q/%v", got, err)
	}
}

func TestAOQTV7R3ActualDirectionProbeHashChainTamperAndNoOp(t *testing.T) {
	set := eightRowAOQTCalibrationSetForProbe(t, 702)
	objective := syntheticForwardConsistencyObjectiveForSet(t, set, false)
	trainer := newAOQTV7R3ProbeTestTrainer(t, 702)
	receipt, err := trainer.runActualDirectionProbe(set, objective)
	if err != nil {
		t.Fatalf("actual-direction probe: %v", err)
	}
	tampered := receipt
	tampered.Endpoints = append([]AOQTSidecarActualDirectionProbeEndpoint(nil), receipt.Endpoints...)
	tampered.Endpoints[0].Magnitude = receipt.Endpoints[0].Magnitude * 0.5
	if err := tampered.Validate(); err == nil {
		t.Fatalf("tampered endpoint unexpectedly validated")
	}

	noOp := preparedAOQTObjectiveForSet(t, set)
	noOp.workspace.evaluateOverride = func(input AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error) {
		result := AOQTSidecarObjectiveResult{
			QueryGrad:      make([]float32, len(input.Query)),
			CandidateGrads: make([][]float32, len(input.Candidates)),
			Activation:     aoqtActiveObjectiveForWeights(input.Row.Weights, len(input.Candidates)),
		}
		for i, candidate := range input.Candidates {
			result.CandidateGrads[i] = make([]float32, len(candidate))
		}
		return result, nil
	}
	noOpTrainer := newAOQTV7R3ProbeTestTrainer(t, 702)
	noOpReceipt, err := noOpTrainer.runActualDirectionProbe(set, noOp)
	if err != nil {
		t.Fatalf("no-op actual-direction probe: %v", err)
	}
	if !noOpReceipt.Passed || noOpReceipt.FullTransactionCandidateCount != 0 {
		t.Fatalf("no-op receipt = passed %t candidates %d, want passed with zero candidates", noOpReceipt.Passed, noOpReceipt.FullTransactionCandidateCount)
	}
	if err := noOpReceipt.Validate(); err != nil {
		t.Fatalf("validate no-op actual-direction receipt: %v", err)
	}
}

func TestAOQTV7R3ActualDirectionProbeNonFiniteFailsClosed(t *testing.T) {
	set := eightRowAOQTCalibrationSetForProbe(t, 703)
	objective := preparedAOQTObjectiveForSet(t, set)
	objective.workspace.evaluateOverride = func(input AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error) {
		result := AOQTSidecarObjectiveResult{
			Loss:           float32(math.NaN()),
			QueryGrad:      make([]float32, len(input.Query)),
			CandidateGrads: make([][]float32, len(input.Candidates)),
			Activation:     aoqtActiveObjectiveForWeights(input.Row.Weights, len(input.Candidates)),
		}
		for i, candidate := range input.Candidates {
			result.CandidateGrads[i] = make([]float32, len(candidate))
		}
		return result, nil
	}
	trainer := newAOQTV7R3ProbeTestTrainer(t, 703)
	summary, err := trainer.Fit(set, objective)
	if err == nil || summary.ActualDirectionProbe == nil || summary.ActualDirectionProbe.Passed {
		t.Fatalf("non-finite actual-direction fit = receipt=%+v err=%v; want failed receipt", summary.ActualDirectionProbe, err)
	}
	if receiptErr := summary.ActualDirectionProbe.Validate(); receiptErr != nil {
		t.Fatalf("non-finite early-failure receipt should validate as fail-closed evidence: %v", receiptErr)
	}
}

func TestAOQTV7R3ActualDirectionFitEmitsActualProbeReceipts(t *testing.T) {
	set := eightRowAOQTCalibrationSetForProbe(t, 704)
	trainer := newAOQTV7R3ProbeTestTrainer(t, 704)
	summary, err := trainer.Fit(set, syntheticForwardConsistencyObjectiveForSet(t, set, false))
	if err != nil {
		t.Fatalf("V7-r3 fit: %v", err)
	}
	if summary.ActualDirectionProbe == nil || !summary.ActualDirectionProbe.Passed {
		t.Fatalf("V7-r3 fit actual-direction probe = %+v", summary.ActualDirectionProbe)
	}
	if summary.ActualDirectionProbeSHA256 == "" {
		t.Fatalf("V7-r3 fit did not emit actual-direction probe hash")
	}
	if summary.OptimizerDiagnostics == nil {
		t.Fatalf("V7-r3 optimizer diagnostics are missing")
	}
	if err := validateAOQTOptimizerDiagnostics(summary.Plan, summary); err != nil {
		t.Fatalf("V7-r3 optimizer diagnostics provenance: %v", err)
	}
	if summary.OptimizerDiagnostics.ProposalReceipts != nil {
		for _, receipt := range *summary.OptimizerDiagnostics.ProposalReceipts {
			if receipt.Kind == aoqtProposalKindCoordinate {
				if summary.OptimizerDiagnostics.CoordinateSearchStrategy != aoqtCoordinateSearchStrategyV7R3ActualDirection {
					t.Fatalf("V7-r3 coordinate strategy = %q", summary.OptimizerDiagnostics.CoordinateSearchStrategy)
				}
				if receipt.DirectionSource != AOQTV7R3ActualDirectionProbeDirectionSource || receipt.EndpointOrdinal <= 0 {
					t.Fatalf("V7-r3 coordinate receipt missing actual-probe binding: %+v", receipt)
				}
			}
		}
	}
}

func TestAOQTV7R3FallbackUsesProbeBaselineAndBindsAcceptedEndpoint(t *testing.T) {
	set := eightRowAOQTCalibrationSetForProbe(t, 705)
	objective := syntheticForwardConsistencyObjectiveForSet(t, set, false)
	trainer := newAOQTV7R3ProbeTestTrainer(t, 705)
	baselineLoss, aggregateGrad, activation, baselineComponents, err := trainer.lossAndAngleGrad(set.Rows, objective)
	if err != nil {
		t.Fatalf("baseline objective: %v", err)
	}
	q3GainGrad, err := trainer.q3GainOnlyAngleGrad(set.Rows, objective)
	if err != nil {
		t.Fatalf("q3 objective gradient: %v", err)
	}
	protectedGradients, err := trainer.protectedComponentAngleGrads(set.Rows, objective, set.Manifest.ObjectiveContract.WeightSums)
	if err != nil {
		t.Fatalf("protected objective gradients: %v", err)
	}
	search, err := trainer.actualDirectionSearch(set, objective)
	if err != nil {
		t.Fatalf("actual-direction search: %v", err)
	}
	if len(search.SelectedSpecs) == 0 {
		t.Skip("synthetic objective produced no eligible full-transaction endpoint")
	}
	baseline := aoqtStepEvaluation{loss: baselineLoss, activation: activation, components: baselineComponents}
	stateBefore := trainer.snapshotOptimizerState()
	receipts := make(AOQTSidecarOptimizerProposalReceipts, 0)
	diagnostics := AOQTSidecarOptimizerDiagnostics{
		OptimizerMode:      AOQTSidecarOptimizerModeV7R3DevActualDirection,
		DevOnly:            true,
		PlannedSteps:       1,
		MaxAttemptsPerStep: aoqtTransactionalMaxAttemptsPerStep,
		ProposalReceipts:   &receipts,
	}
	callbackCalls := 0
	accepted, err := trainer.acceptTransactionalActualDirectionStep(set, objective, aggregateGrad, q3GainGrad, baseline, func() (aoqtStepEvaluation, error) {
		callbackCalls++
		candidate := baseline
		if callbackCalls <= aoqtTransactionalAdamMaxAttemptsPerStep {
			candidate.loss += 1
			candidate.components.Q3Gain += 1
		} else {
			candidate.loss -= 1
			candidate.components.Q3Gain -= 1
		}
		return candidate, nil
	}, set.Manifest.ObjectiveContract.WeightSums, &diagnostics, protectedGradients)
	if err != nil {
		t.Fatalf("actual-direction fallback: %v", err)
	}
	if !accepted || diagnostics.CoordinateAcceptedProposals != 1 {
		t.Fatalf("fallback accepted=%t diagnostics=%+v, want one accepted coordinate proposal", accepted, diagnostics)
	}
	var acceptedReceipt *AOQTSidecarOptimizerProposalReceipt
	for i := range receipts {
		if receipts[i].Accepted {
			acceptedReceipt = &receipts[i]
			break
		}
	}
	if acceptedReceipt == nil {
		t.Fatal("accepted coordinate proposal receipt is missing")
	}
	probeSHA, err := search.Receipt.SHA256()
	if err != nil {
		t.Fatalf("probe hash: %v", err)
	}
	if acceptedReceipt.ActualDirectionProbeSHA256 != probeSHA {
		t.Fatalf("proposal probe hash = %q, want %q", acceptedReceipt.ActualDirectionProbeSHA256, probeSHA)
	}
	endpoint := search.Receipt.Endpoints[acceptedReceipt.EndpointOrdinal-1]
	var chain []string
	if err := strictUnmarshalAOQT([]byte(search.Receipt.EndpointHashChain), &chain); err != nil {
		t.Fatalf("decode probe hash chain: %v", err)
	}
	if acceptedReceipt.EndpointHashSHA256 != chain[acceptedReceipt.EndpointOrdinal-1] || acceptedReceipt.RequestedDirectionSHA256 != endpoint.RequestedDirectionSHA256 || acceptedReceipt.ActualDirectionSHA256 != endpoint.ActualDirectionSHA256 {
		t.Fatalf("accepted proposal provenance = %+v, want selected endpoint %+v", acceptedReceipt, endpoint)
	}
	if err := validateAOQTActualDirectionProposalProvenance(search.Receipt, probeSHA, diagnostics, "fallback test"); err != nil {
		t.Fatalf("accepted proposal provenance validation: %v", err)
	}
	forgedDiagnostics := diagnostics
	forgedReceipts := append(AOQTSidecarOptimizerProposalReceipts(nil), receipts...)
	forgedDiagnostics.ProposalReceipts = &forgedReceipts
	for i := range forgedReceipts {
		if forgedReceipts[i].Accepted {
			forgedReceipts[i].EndpointHashSHA256 = strings.Repeat("d", 64)
			break
		}
	}
	if err := validateAOQTActualDirectionProposalProvenance(search.Receipt, probeSHA, forgedDiagnostics, "fallback forgery test"); err == nil {
		t.Fatal("forged accepted endpoint provenance unexpectedly validated")
	}
	if trainer.step != stateBefore.step || !float32SlicesEqual(trainer.adamM, stateBefore.adamM) || !float32SlicesEqual(trainer.adamV, stateBefore.adamV) {
		t.Fatalf("fallback changed Adam state despite accepted full proposal")
	}
}

func TestAOQTV7R3RunnerPreconditionWritesOnlyDevFailureDiagnostics(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	splitManifestPath := writeV7DevSplitManifestForTrainRunnerTest(t, trainCfg, "fold-0", nil)
	trainCfg.PlanOnly = false
	trainCfg.MaxSteps = 1
	trainCfg.LearningRate = AOQTV7R3ActualDirectionProbeLearningRate
	trainCfg.OptimizerMode = AOQTSidecarOptimizerModeV7R3DevActualDirection
	trainCfg.RequireActualDirectionProbe = true
	trainCfg.AllowDevOnly = true
	trainCfg.DevOutputDir = t.TempDir()
	trainCfg.SplitManifestPath = splitManifestPath
	trainCfg.ExpectedSplitManifestSHA256 = mustSHA256FileAOQTTest(t, splitManifestPath)
	trainCfg.FoldID = "fold-0"
	result, err := runAOQTSidecarTraining(trainCfg, func(contract AOQTSidecarObjectiveContract) (AOQTSidecarVectorObjective, error) {
		objective, objectiveErr := objectiveFromAOQTContract(contract)
		if objectiveErr != nil {
			return nil, objectiveErr
		}
		objective.workspace.evaluateOverride = func(input AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error) {
			result := AOQTSidecarObjectiveResult{
				Loss:           float32(math.NaN()),
				QueryGrad:      make([]float32, len(input.Query)),
				CandidateGrads: make([][]float32, len(input.Candidates)),
				Activation:     aoqtActiveObjectiveForWeights(input.Row.Weights, len(input.Candidates)),
			}
			for i, candidate := range input.Candidates {
				result.CandidateGrads[i] = make([]float32, len(candidate))
			}
			return result, nil
		}
		return objective, nil
	})
	if err == nil || result.DevFailureDiagnostics == nil {
		t.Fatalf("V7-r3 precondition result=%+v err=%v, want fail-closed diagnostics", result, err)
	}
	if result.DevFailureDiagnostics.ActualDirectionProbe == nil || result.DevFailureDiagnostics.ActualDirectionProbe.Passed {
		t.Fatalf("V7-r3 precondition probe = %+v, want failed receipt", result.DevFailureDiagnostics.ActualDirectionProbe)
	}
	if !strings.Contains(result.DevFailureDiagnostics.FailureKind, "actual_direction") {
		t.Fatalf("V7-r3 failure kind = %q", result.DevFailureDiagnostics.FailureKind)
	}
	if err := result.DevFailureDiagnostics.Validate(); err != nil {
		t.Fatalf("V7-r3 precondition diagnostics invalid: %v", err)
	}
	if _, err := os.Stat(AOQTDevFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath)); err != nil {
		t.Fatalf("V7-r3 dev fail-closed diagnostics missing: %v", err)
	}
	if _, err := os.Stat(filepath.Join(trainCfg.DevOutputDir, "aoqt-sidecar-dev-evidence.json")); !os.IsNotExist(err) {
		t.Fatalf("V7-r3 precondition wrote dev evidence: %v", err)
	}
}

func TestAOQTV7R3ProbeOnlyWritesOnlyBoundReceipt(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	splitManifestPath := writeV7DevSplitManifestForTrainRunnerTest(t, trainCfg, "fold-0", nil)
	receiptPath := filepath.Join(t.TempDir(), "actual-direction-probe.json")
	trainCfg.MetricsJSONPath = filepath.Join(t.TempDir(), "must-not-be-written.json")
	trainCfg.PlanOnly = false
	trainCfg.MaxSteps = 0
	trainCfg.LearningRate = AOQTV7R3ActualDirectionProbeLearningRate
	trainCfg.OptimizerMode = AOQTSidecarOptimizerModeV7R3DevActualDirection
	trainCfg.RequireActualDirectionProbe = true
	trainCfg.ActualDirectionProbeOnly = true
	trainCfg.ActualDirectionProbeReceiptJSONPath = receiptPath
	trainCfg.AllowDevOnly = true
	trainCfg.DevOutputDir = ""
	trainCfg.SplitManifestPath = splitManifestPath
	trainCfg.ExpectedSplitManifestSHA256 = mustSHA256FileAOQTTest(t, splitManifestPath)
	trainCfg.FoldID = "fold-0"
	result, err := runAOQTSidecarTraining(trainCfg, aoqtRunnerObjectiveFromContract)
	if err != nil {
		t.Fatalf("V7-r3 probe-only run: %v", err)
	}
	if result.ActualDirectionProbe == nil || !result.ActualDirectionProbe.Passed {
		t.Fatalf("V7-r3 probe-only receipt = %+v", result.ActualDirectionProbe)
	}
	if result.ActualDirectionProbeReceiptJSONPath != receiptPath || result.ActualDirectionProbeSHA256 == "" {
		t.Fatalf("V7-r3 probe-only result path/hash = %q/%q", result.ActualDirectionProbeReceiptJSONPath, result.ActualDirectionProbeSHA256)
	}
	if err := result.ActualDirectionProbe.ValidateBound(); err != nil {
		t.Fatalf("V7-r3 bound probe receipt: %v", err)
	}
	mutate := func(name string, fn func(*AOQTSidecarActualDirectionProbeReceipt)) {
		t.Helper()
		data, err := json.Marshal(*result.ActualDirectionProbe)
		if err != nil {
			t.Fatalf("marshal receipt %s: %v", name, err)
		}
		var tampered AOQTSidecarActualDirectionProbeReceipt
		if err := strictUnmarshalAOQT(data, &tampered); err != nil {
			t.Fatalf("clone receipt %s: %v", name, err)
		}
		fn(&tampered)
		if err := tampered.ValidateBound(); err == nil {
			t.Fatalf("coordinated receipt/binding forgery %s unexpectedly validated", name)
		}
	}
	mutate("ranking-hash", func(r *AOQTSidecarActualDirectionProbeReceipt) {
		r.RankingSHA256 = strings.Repeat("a", 64)
		r.Binding.RankingSHA256 = r.RankingSHA256
	})
	mutate("objective-contract-and-hash", func(r *AOQTSidecarActualDirectionProbeReceipt) {
		r.Binding.IOReport.ObjectiveContract.GainMargin += 0.001
		contractSHA, err := aoqtActualDirectionObjectiveContractSHA256(r.Binding.IOReport.ObjectiveContract)
		if err != nil {
			t.Fatalf("hash forged objective contract: %v", err)
		}
		r.ObjectiveContractSHA256 = contractSHA
		r.Binding.ObjectiveContractSHA256 = contractSHA
	})
	mutate("eligibility-policy-and-hash", func(r *AOQTSidecarActualDirectionProbeReceipt) {
		policy := AOQTSidecarDefaultCandidateEligibilityPolicy()
		policy.Q3GainAllowedLossIncrease = 0.001
		r.Binding.IOReport.CandidateEligibilityPolicy = &policy
		policySHA, err := aoqtActualDirectionPolicySHA256(policy)
		if err != nil {
			t.Fatalf("hash forged eligibility policy: %v", err)
		}
		r.EligibilityPolicySHA256 = policySHA
		r.Binding.EligibilityPolicySHA256 = policySHA
	})
	mutate("selection-seed", func(r *AOQTSidecarActualDirectionProbeReceipt) {
		r.SelectionSeedSHA256 = strings.Repeat("b", 64)
	})
	mutate("schedule-hash", func(r *AOQTSidecarActualDirectionProbeReceipt) {
		r.ScheduleSHA256 = strings.Repeat("c", 64)
		r.Binding.ScheduleSHA256 = r.ScheduleSHA256
	})
	mutate("split-binding-fold", func(r *AOQTSidecarActualDirectionProbeReceipt) {
		r.Binding.SplitBinding.FoldID = "forged-fold"
	})
	if _, err := os.Stat(receiptPath); err != nil {
		t.Fatalf("V7-r3 receipt output missing: %v", err)
	}
	for _, path := range []string{trainCfg.MetricsJSONPath, AOQTFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath), AOQTDevFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath), AOQTPostFitFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath)} {
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("V7-r3 probe-only wrote nonreceipt artifact %q: %v", path, err)
		}
	}
}
