package eosruntime

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"
)

// TestAOQTSidecarSummaryUsesStepZeroOrder keeps initial and final telemetry
// on the same deterministic row order. The real prepared-IP objective and Fit
// path are used; only the calibration rows are synthetic and in-memory.
func TestAOQTSidecarSummaryUsesStepZeroOrder(t *testing.T) {
	const seed int64 = 5581486560434873699
	const rowCount = 8

	for _, maxSteps := range []int{1, 2} {
		t.Run(fmt.Sprintf("max_steps_%d", maxSteps), func(t *testing.T) {
			set := tinyAOQTCalibrationSet(t, seed)
			set.Rows = syntheticAOQTOrderRowsForSummaryTest(t, set.Rows[0], rowCount, seed+1)
			for i := range set.Rows {
				set.Rows[i].Weights = AOQTSidecarRowWeights{
					Q3Gain:         1,
					Q3OrderGuard:   0.5,
					Q3ScoreDistill: 0.25,
					Q5OrderGuard:   0.5,
					Q5ScoreDistill: 0.25,
				}
			}
			set.Manifest.RowCount = len(set.Rows)
			rowIDs := make([]string, len(set.Rows))
			for i, row := range set.Rows {
				rowIDs[i] = row.RowID
			}
			set.Manifest.RowIDSHA256 = aoqtRowIDSHA256(rowIDs)
			set.Manifest.ObjectiveContract = tinyAOQTObjectiveConfig(set.Manifest.TurboQuantSeed).ObjectiveContract(sumAOQTRowWeights(set.Rows))
			if err := set.Validate(); err != nil {
				t.Fatalf("synthetic Fit fixture invalid: %v", err)
			}

			stepZeroRows := deterministicAOQTRowOrder(set.Rows, seed+1000, 0)
			initialTrainer, err := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{
				PairingSeed:  seed,
				WorkplanSeed: seed + 1000,
				MaxSteps:     maxSteps,
				LearningRate: 0.001,
			})
			if err != nil {
				t.Fatalf("initial trainer: %v", err)
			}
			initialObjective := preparedAOQTObjectiveForSet(t, set)
			initialLoss, _, initialActivation, initialComponents, err := initialTrainer.lossAndAngleGrad(stepZeroRows, initialObjective)
			if err != nil {
				t.Fatalf("recompute initial step-zero evaluation: %v", err)
			}

			trainer, err := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{
				PairingSeed:             seed,
				WorkplanSeed:            seed + 1000,
				MaxSteps:                maxSteps,
				LearningRate:            0.001,
				CaptureProposalReceipts: true,
			})
			if err != nil {
				t.Fatalf("Fit trainer: %v", err)
			}
			fitObjective := preparedAOQTObjectiveForSet(t, set)
			summary, err := trainer.Fit(set, fitObjective)
			if err != nil {
				t.Fatalf("Fit failed before summary comparison: %v; diagnostics=%+v", err, summary.OptimizerDiagnostics)
			}
			if summary.OptimizerDiagnostics == nil {
				t.Fatal("Fit diagnostics are nil")
			}
			diagnostics := summary.OptimizerDiagnostics
			t.Logf("max_steps=%d actual_summary_steps=%d attempted_steps=%d accepted_steps=%d proposal_attempts=%d", maxSteps, summary.Steps, diagnostics.AttemptedSteps, diagnostics.AcceptedSteps, diagnostics.ProposalAttempts)
			if summary.Steps != maxSteps || diagnostics.AttemptedSteps != maxSteps || diagnostics.AcceptedSteps != maxSteps {
				t.Fatalf("Fit step accounting = summary steps %d attempted %d accepted %d, want all %d", summary.Steps, diagnostics.AttemptedSteps, diagnostics.AcceptedSteps, maxSteps)
			}
			if summary.InitialLoss != initialLoss || summary.InitialObjectiveActivation != initialActivation || summary.InitialObjectiveComponents != initialComponents {
				t.Fatalf("initial summary is not recomputable from step-zero order: got loss=%.9g components=%+v activation=%+v want loss=%.9g components=%+v activation=%+v", summary.InitialLoss, summary.InitialObjectiveComponents, summary.InitialObjectiveActivation, initialLoss, initialComponents, initialActivation)
			}

			finalLoss, _, finalActivation, finalStepZeroComponents, err := trainer.lossAndAngleGrad(stepZeroRows, fitObjective)
			if err != nil {
				t.Fatalf("recompute final step-zero evaluation: %v", err)
			}
			if summary.FinalLoss != finalLoss || summary.FinalObjectiveActivation != finalActivation || summary.FinalObjectiveComponents != finalStepZeroComponents {
				t.Fatalf("final summary is not recomputable from same step-zero order: got loss=%.9g components=%+v activation=%+v want loss=%.9g components=%+v activation=%+v", summary.FinalLoss, summary.FinalObjectiveComponents, summary.FinalObjectiveActivation, finalLoss, finalStepZeroComponents, finalActivation)
			}
		})
	}
}

func TestAOQTSidecarJointPolicyRejectsQ5OrderGuardRegressionAtZeroMargin(t *testing.T) {
	jointPolicy := AOQTSidecarScoreDistillJointBudgetV1Policy()
	metrics := safeTinyAOQTCandidateMetrics(t, tinyAOQTCalibrationSet(t, 188), AOQTSidecarObjectiveActivation{
		Q3GainEligiblePairs: 2, Q3GainContributingPairs: 2,
		Q3OrderGuardPairs: 2, Q3OrderGuardContributing: 2,
		Q3ScoreDistillCount: 3, Q5OrderGuardPairs: 2,
		Q5OrderGuardContributing: 2, Q5ScoreDistillCount: 3,
	})
	metrics.TrainingContract = AOQTSidecarScoreDistillJointBudgetV1Contract
	metrics.CandidateEligibilityPolicy = &jointPolicy
	metrics.Summary.TrainingContract = AOQTSidecarScoreDistillJointBudgetV1Contract
	metrics.Summary.CandidateEligibilityPolicy = &jointPolicy
	initial := metrics.Summary.InitialObjectiveComponents
	final := initial
	final.Q3Gain -= 0.5
	final.Q5OrderGuard += 0.001
	metrics.Summary.FinalObjectiveComponents = final
	metrics.Summary.InitialLoss = initial.Sum()
	metrics.Summary.FinalLoss = final.Sum()
	if err := metrics.Validate(); err != nil {
		t.Fatalf("structurally valid joint q5 regression fixture: %v", err)
	}
	if err := ValidateAOQTSidecarCandidateEligibility(metrics, jointPolicy); err == nil || !strings.Contains(err.Error(), "q5_order_guard") {
		t.Fatalf("joint q5 order-guard regression error = %v, want zero-margin q5 rejection", err)
	}
}

func syntheticAOQTOrderRowsForSummaryTest(t *testing.T, base AOQTSidecarCalibrationRow, count int, seed int64) []AOQTSidecarCalibrationRow {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	rows := make([]AOQTSidecarCalibrationRow, count)
	for i := range rows {
		row := cloneAOQTCalibrationSet(AOQTSidecarCalibrationSet{Rows: []AOQTSidecarCalibrationRow{base}}).Rows[0]
		query := aoqtRandomVec(rng)
		candidates := [][]float32{aoqtRandomVec(rng), aoqtRandomVec(rng), append([]float32(nil), query...)}
		row.RowID = fmt.Sprintf("summary-order-%04d", i)
		row.QueryID = fmt.Sprintf("summary-q-%04d", i)
		row.QueryVectorID = fmt.Sprintf("summary-qv-%04d", i)
		row.QueryVector = query
		row.QueryVectorSHA256 = aoqtVectorSHA256(query)
		row.CandidateVectors = candidates
		row.CandidateVectorSHA256 = make([]string, len(candidates))
		for j := range candidates {
			row.CandidateVectorSHA256[j] = aoqtVectorSHA256(candidates[j])
		}
		row.AnchorScores.Dense = denseAOQTScores(query, candidates)
		row.AnchorScores.Q3 = preparedAOQTScores(query, candidates, AOQTSidecarDefaultGuardBit3, 77)
		row.AnchorScores.Q5 = preparedAOQTScores(query, candidates, AOQTSidecarDefaultGuardBit5, 77)
		row.AnchorRanks.Dense = ranksAOQT(row.CandidateDocIDs, row.AnchorScores.Dense)
		row.AnchorRanks.Q3 = ranksAOQT(row.CandidateDocIDs, row.AnchorScores.Q3)
		row.AnchorRanks.Q5 = ranksAOQT(row.CandidateDocIDs, row.AnchorScores.Q5)
		rows[i] = row
	}
	return rows
}
