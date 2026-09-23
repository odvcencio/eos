package eosruntime

import (
	"fmt"
	"math"
	"reflect"
	"testing"
)

func TestAOQTV7R2ForwardConsistencyProbeSyntheticAgreementAndDeterminism(t *testing.T) {
	set := eightRowAOQTCalibrationSetForProbe(t, 601)
	objective := syntheticForwardConsistencyObjectiveForSet(t, set, false)
	newTrainer := func() *AOQTSidecarTrainer {
		trainer, err := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{
			PairingSeed:                                601,
			WorkplanSeed:                               1601,
			OptimizerMode:                              AOQTSidecarOptimizerModeV7R2DevTrustRegion,
			ForwardConsistencyProbeRequired:            true,
			ForwardConsistencyProbeFoldID:              "fold-0",
			ForwardConsistencyProbeSplitManifestSHA256: hex64("probe-split"),
			MaxSteps:                1,
			LearningRate:            0.01,
			CaptureProposalReceipts: true,
		})
		if err != nil {
			t.Fatalf("new V7-r2 trainer: %v", err)
		}
		return trainer
	}
	a := newTrainer()
	b := newTrainer()
	receiptA, err := a.runForwardConsistencyProbe(set, objective)
	if err != nil {
		t.Fatalf("agreement probe: %v", err)
	}
	receiptB, err := b.runForwardConsistencyProbe(set, objective)
	if err != nil {
		t.Fatalf("repeat agreement probe: %v", err)
	}
	if !receiptA.Passed || !receiptB.Passed {
		t.Fatalf("agreement receipts = %+v / %+v", receiptA, receiptB)
	}
	if !reflect.DeepEqual(receiptA, receiptB) {
		t.Fatalf("probe receipts are not deterministic")
	}
	if got, err := receiptA.SHA256(); err != nil || got == "" {
		t.Fatalf("probe receipt hash = %q/%v", got, err)
	}
	if err := receiptA.Validate(); err != nil {
		t.Fatalf("validate agreement receipt: %v", err)
	}
	if len(receiptA.CoordinateProposals) != AOQTV7R2ForwardConsistencyProbeTopAngles || len(receiptA.BlockProposals) != 3 {
		t.Fatalf("proposal counts = %d/%d, want 8/3", len(receiptA.CoordinateProposals), len(receiptA.BlockProposals))
	}
}

func TestAOQTV7R2ForwardConsistencyProbeSyntheticDisagreementFailsClosed(t *testing.T) {
	set := eightRowAOQTCalibrationSetForProbe(t, 602)
	objective := syntheticForwardConsistencyObjectiveForSet(t, set, true)
	trainer, err := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{
		PairingSeed:                                602,
		WorkplanSeed:                               1602,
		OptimizerMode:                              AOQTSidecarOptimizerModeV7R2DevTrustRegion,
		ForwardConsistencyProbeRequired:            true,
		ForwardConsistencyProbeFoldID:              "fold-0",
		ForwardConsistencyProbeSplitManifestSHA256: hex64("probe-split"),
		MaxSteps:     1,
		LearningRate: 0.01,
	})
	if err != nil {
		t.Fatalf("new V7-r2 trainer: %v", err)
	}
	receipt, err := trainer.runForwardConsistencyProbe(set, objective)
	if err == nil || receipt.Passed || receipt.FailureReason == "" {
		t.Fatalf("disagreement probe = %+v, err=%v; want fail-closed receipt", receipt, err)
	}
	if err := receipt.Validate(); err != nil {
		t.Fatalf("validate disagreement receipt: %v", err)
	}
	if receipt.CoordinateProposals[0].Passed {
		t.Fatalf("first disagreement proposal unexpectedly passed: %+v", receipt.CoordinateProposals[0])
	}
}

func TestAOQTV7R2ForwardConsistencyProbeReceiptRejectsContractDriftAndNonFiniteSigns(t *testing.T) {
	set := eightRowAOQTCalibrationSetForProbe(t, 603)
	trainer, err := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{
		PairingSeed:                                603,
		WorkplanSeed:                               1603,
		OptimizerMode:                              AOQTSidecarOptimizerModeV7R2DevTrustRegion,
		ForwardConsistencyProbeRequired:            true,
		ForwardConsistencyProbeFoldID:              "fold-0",
		ForwardConsistencyProbeSplitManifestSHA256: hex64("probe-split"),
		MaxSteps:     1,
		LearningRate: 0.01,
	})
	if err != nil {
		t.Fatalf("new V7-r2 trainer: %v", err)
	}
	receipt, err := trainer.runForwardConsistencyProbe(set, syntheticForwardConsistencyObjectiveForSet(t, set, false))
	if err != nil {
		t.Fatalf("run synthetic probe: %v", err)
	}
	if err := receipt.Validate(); err != nil {
		t.Fatalf("validate baseline receipt: %v", err)
	}
	for name, mutate := range map[string]func(*AOQTSidecarForwardConsistencyProbeReceipt){
		"coordinate-count": func(r *AOQTSidecarForwardConsistencyProbeReceipt) { r.CoordinateCount-- },
		"sign-policy":      func(r *AOQTSidecarForwardConsistencyProbeReceipt) { r.SignPolicy = "silent-tolerance" },
		"near-zero-epsilon": func(r *AOQTSidecarForwardConsistencyProbeReceipt) {
			r.SignNearZeroEpsilon = float32(1e-6)
		},
		"non-finite-sign-data": func(r *AOQTSidecarForwardConsistencyProbeReceipt) {
			r.CoordinateProposals[0].Checks[0].AnalyticDirectionalDerivative = float32(math.NaN())
		},
	} {
		t.Run(name, func(t *testing.T) {
			tampered := receipt
			tampered.CoordinateProposals = append([]AOQTSidecarForwardConsistencyProbeProposal(nil), receipt.CoordinateProposals...)
			tampered.BlockProposals = append([]AOQTSidecarForwardConsistencyProbeProposal(nil), receipt.BlockProposals...)
			tampered.CoordinateProposals[0].Checks = append([]AOQTSidecarForwardConsistencyProbeSignCheck(nil), receipt.CoordinateProposals[0].Checks...)
			mutate(&tampered)
			if err := tampered.Validate(); err == nil {
				t.Fatalf("tampered receipt unexpectedly validated: %+v", tampered)
			}
		})
	}
}

func eightRowAOQTCalibrationSetForProbe(t *testing.T, seed int64) AOQTSidecarCalibrationSet {
	t.Helper()
	base := tinyAOQTCalibrationSet(t, seed)
	rows := make([]AOQTSidecarCalibrationRow, 8)
	for i := range rows {
		rows[i] = base.Rows[0]
		rows[i].RowID = fmt.Sprintf("row-%04d", i+1)
		rows[i].QueryID = fmt.Sprintf("q-%04d", i+1)
		rows[i].QueryVectorID = fmt.Sprintf("qv-%04d", i+1)
	}
	base.Rows = rows
	base.Manifest.RowCount = len(rows)
	rowIDs := make([]string, len(rows))
	for i := range rows {
		rowIDs[i] = rows[i].RowID
	}
	base.Manifest.RowIDSHA256 = aoqtRowIDSHA256(rowIDs)
	base.Manifest.ObjectiveContract = tinyAOQTObjectiveConfig(base.Manifest.TurboQuantSeed).ObjectiveContract(sumAOQTRowWeights(rows))
	if err := base.Validate(); err != nil {
		t.Fatalf("eight-row probe fixture: %v", err)
	}
	return base
}

func syntheticForwardConsistencyObjectiveForSet(t *testing.T, set AOQTSidecarCalibrationSet, inconsistent bool) AOQTSidecarPreparedIPObjective {
	t.Helper()
	objective := preparedAOQTObjectiveForSet(t, set)
	objective.workspace.evaluateOverride = func(input AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error) {
		const baseValue = float32(100)
		const slope = float32(10)
		componentSlope := slope
		gradientSlope := slope
		if inconsistent {
			componentSlope = -slope
		}
		result := AOQTSidecarObjectiveResult{
			QueryGrad:      make([]float32, len(input.Query)),
			CandidateGrads: make([][]float32, len(input.Candidates)),
		}
		for i, candidate := range input.Candidates {
			result.CandidateGrads[i] = make([]float32, len(candidate))
		}
		value := baseValue + componentSlope*input.Query[0]
		for _, item := range []struct {
			weight float32
			set    func(*AOQTSidecarObjectiveComponents, float32)
		}{
			{input.Row.Weights.Q3Gain, func(c *AOQTSidecarObjectiveComponents, value float32) { c.Q3Gain += value }},
			{input.Row.Weights.Q3OrderGuard, func(c *AOQTSidecarObjectiveComponents, value float32) { c.Q3OrderGuard += value }},
			{input.Row.Weights.Q3ScoreDistill, func(c *AOQTSidecarObjectiveComponents, value float32) { c.Q3ScoreDistill += value }},
			{input.Row.Weights.Q5OrderGuard, func(c *AOQTSidecarObjectiveComponents, value float32) { c.Q5OrderGuard += value }},
			{input.Row.Weights.Q5ScoreDistill, func(c *AOQTSidecarObjectiveComponents, value float32) { c.Q5ScoreDistill += value }},
			{input.Row.Weights.NFBoundaryGuard, func(c *AOQTSidecarObjectiveComponents, value float32) { c.NFBoundaryGuard += value }},
		} {
			if item.weight <= 0 {
				continue
			}
			component := item.weight * value
			item.set(&result.Components, component)
			result.Loss += component
			result.QueryGrad[0] += item.weight * gradientSlope
		}
		result.Activation = aoqtActiveObjectiveForWeights(input.Row.Weights, len(input.Candidates))
		return result, nil
	}
	return objective
}
