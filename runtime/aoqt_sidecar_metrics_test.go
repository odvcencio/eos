package eosruntime

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"testing"
)

func TestAOQTCandidateEligibilityRequiresAngleMovementAndQ3Improvement(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 85)
	enableTinyAOQTNFGuard(t, &set)
	activation := AOQTSidecarObjectiveActivation{
		Q3GainEligiblePairs:         2,
		Q3GainContributingPairs:     2,
		Q3OrderGuardPairs:           2,
		Q3OrderGuardContributing:    2,
		Q3ScoreDistillCount:         3,
		Q5OrderGuardPairs:           2,
		Q5OrderGuardContributing:    2,
		Q5ScoreDistillCount:         3,
		NFBoundaryGuardPairs:        1,
		NFBoundaryGuardContributing: 1,
	}
	metrics := safeTinyAOQTCandidateMetrics(t, set, activation)

	for _, tc := range []struct {
		name   string
		mutate func(*AOQTSidecarRunMetrics)
		want   string
	}{
		{
			name: "no-angle-movement",
			mutate: func(m *AOQTSidecarRunMetrics) {
				m.Summary.AngleL2 = 0
				m.Summary.AngleMaxAbs = 0
			},
			want: "angle_max_abs must be non-zero",
		},
		{
			name: "no-q3-gain-improvement",
			mutate: func(m *AOQTSidecarRunMetrics) {
				m.Summary.FinalObjectiveComponents.Q3Gain = m.Summary.InitialObjectiveComponents.Q3Gain
				m.Summary.FinalLoss = m.Summary.FinalObjectiveComponents.Sum()
			},
			want: "q3_gain component must strictly improve",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bad := metrics
			tc.mutate(&bad)
			err := ValidateAOQTSidecarCandidateEligibility(bad, AOQTSidecarCandidateEligibilityPolicy{RequireObjectiveActivation: true})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestAOQTScoreDistillBudgetContractPreservesLegacyZeroBudget(t *testing.T) {
	baseline := aoqtStepEvaluation{
		loss:       6,
		activation: AOQTSidecarObjectiveActivation{Q3GainEligiblePairs: 1, Q3GainContributingPairs: 1, Q3ScoreDistillCount: 1, Q5ScoreDistillCount: 1},
		components: AOQTSidecarObjectiveComponents{Q3Gain: 1, Q3ScoreDistill: 1, Q5ScoreDistill: 1, Q3OrderGuard: 1, Q5OrderGuard: 1, NFBoundaryGuard: 1},
	}
	candidate := baseline
	candidate.components.Q3Gain = 0.5
	candidate.components.Q3ScoreDistill += 5e-6
	candidate.components.Q5ScoreDistill += 5e-6
	candidate.loss = candidate.components.Sum()
	weights := AOQTSidecarRowWeights{Q3Gain: 1, Q3ScoreDistill: 1, Q5ScoreDistill: 1}
	if decision := aoqtEvaluateTransactionalStep(baseline, candidate, weights); decision.accepted || decision.reason != aoqtRejectionComponentRegression {
		t.Fatalf("legacy zero-budget decision = %+v, want component-regression rejection", decision)
	}
	lanePolicy := AOQTSidecarScoreDistillBudgetLaneAPolicy()
	resolved, err := AOQTSidecarTrainingContractPolicy(AOQTSidecarScoreDistillBudgetLaneAContract, &lanePolicy)
	if err != nil {
		t.Fatalf("resolve laneA contract: %v", err)
	}
	if decision := aoqtEvaluateTransactionalStepWithPolicy(baseline, candidate, weights, resolved); !decision.accepted {
		t.Fatalf("laneA within-budget decision = %+v, want acceptance", decision)
	}
	trainer, err := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{
		PairingSeed:                1,
		WorkplanSeed:               2,
		PlanOnly:                   true,
		TrainingContract:           AOQTSidecarScoreDistillBudgetLaneAContract,
		CandidateEligibilityPolicy: &lanePolicy,
	})
	if err != nil {
		t.Fatalf("laneA trainer: %v", err)
	}
	state := trainer.snapshotOptimizerState()
	trainer.angles[0] = 1e-4
	if decision := trainer.evaluateTransactionalProposal(state, baseline, candidate, weights); !decision.accepted {
		t.Fatalf("laneA trainer wiring decision = %+v, want acceptance", decision)
	}

	over := candidate
	over.components.Q3ScoreDistill = baseline.components.Q3ScoreDistill + 2e-5
	over.loss = over.components.Sum()
	if decision := aoqtEvaluateTransactionalStepWithPolicy(baseline, over, weights, resolved); decision.accepted || decision.reason != aoqtRejectionComponentRegression {
		t.Fatalf("laneA above-budget decision = %+v, want component-regression rejection", decision)
	}
	otherGuard := candidate
	otherGuard.components.Q3OrderGuard = baseline.components.Q3OrderGuard + 2e-6
	otherGuard.loss = otherGuard.components.Sum()
	if decision := aoqtEvaluateTransactionalStepWithPolicy(baseline, otherGuard, weights, resolved); decision.accepted || decision.reason != aoqtRejectionComponentRegression {
		t.Fatalf("laneA other-guard decision = %+v, want component-regression rejection", decision)
	}
}

func TestAOQTScoreDistillBudgetContractRejectsMislabelledProvenance(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 186)
	activation := AOQTSidecarObjectiveActivation{
		Q3GainEligiblePairs: 2, Q3GainContributingPairs: 2,
		Q3OrderGuardPairs: 2, Q3OrderGuardContributing: 2,
		Q3ScoreDistillCount: 3, Q5OrderGuardPairs: 2,
		Q5OrderGuardContributing: 2, Q5ScoreDistillCount: 3,
	}
	metrics := safeTinyAOQTCandidateMetrics(t, set, activation)
	policy := AOQTSidecarScoreDistillBudgetLaneAPolicy()
	metrics.TrainingContract = AOQTSidecarScoreDistillBudgetLaneAContract
	metrics.CandidateEligibilityPolicy = &policy
	metrics.Summary.TrainingContract = AOQTSidecarScoreDistillBudgetLaneAContract
	metrics.Summary.CandidateEligibilityPolicy = &policy
	if err := metrics.Validate(); err != nil {
		t.Fatalf("valid laneA metrics: %v", err)
	}
	if err := ValidateAOQTSidecarCandidateEligibility(metrics, policy); err != nil {
		t.Fatalf("valid laneA candidate: %v", err)
	}
	bad := metrics
	bad.Summary.CandidateEligibilityPolicy = &AOQTSidecarCandidateEligibilityPolicy{
		DenseMaxAbsDeltaTolerance:         policy.DenseMaxAbsDeltaTolerance,
		AngleMaxAbsCap:                    policy.AngleMaxAbsCap,
		RequireObjectiveActivation:        true,
		Q3ScoreDistillAllowedLossIncrease: 2e-5,
		Q5ScoreDistillAllowedLossIncrease: 1e-5,
	}
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "does not match its preregistered q3/q5 score-distill budgets") {
		t.Fatalf("mislabelled budget validation error = %v, want preregistered-policy rejection", err)
	}
	if err := ValidateAOQTSidecarCandidateEligibility(metrics, AOQTSidecarDefaultCandidateEligibilityPolicy()); err == nil || !strings.Contains(err.Error(), "does not match metrics training contract") {
		t.Fatalf("wrong caller budget error = %v, want contract mismatch", err)
	}
}

func TestAOQTCandidateEligibilityEnforcesTransactionalTotalLossEpsilon(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 187)
	activation := AOQTSidecarObjectiveActivation{
		Q3GainEligiblePairs: 2, Q3GainContributingPairs: 2,
		Q3OrderGuardPairs: 2, Q3OrderGuardContributing: 2,
		Q3ScoreDistillCount: 3, Q5OrderGuardPairs: 2,
		Q5OrderGuardContributing: 2, Q5ScoreDistillCount: 3,
	}
	metrics := safeTinyAOQTCandidateMetrics(t, set, activation)
	lanePolicy := AOQTSidecarScoreDistillBudgetLaneAPolicy()
	metrics.TrainingContract = AOQTSidecarScoreDistillBudgetLaneAContract
	metrics.CandidateEligibilityPolicy = &lanePolicy
	metrics.Summary.TrainingContract = AOQTSidecarScoreDistillBudgetLaneAContract
	metrics.Summary.CandidateEligibilityPolicy = &lanePolicy
	initial := metrics.Summary.InitialObjectiveComponents

	bad := metrics
	badFinal := initial
	badFinal.Q3Gain -= float32(1e-6)
	badFinal.Q3ScoreDistill += float32(9e-6)
	badFinal.Q5ScoreDistill += float32(9e-6)
	bad.Summary.FinalObjectiveComponents = badFinal
	bad.Summary.InitialLoss = initial.Sum()
	bad.Summary.FinalLoss = badFinal.Sum()
	if delta := bad.Summary.FinalLoss - bad.Summary.InitialLoss; delta <= aoqtTransactionalLossEpsilon {
		t.Fatalf("regression fixture total loss delta = %.9g, want > epsilon %.9g", delta, aoqtTransactionalLossEpsilon)
	}
	if err := bad.Validate(); err != nil {
		t.Fatalf("regression fixture metrics: %v", err)
	}
	if err := ValidateAOQTSidecarCandidateEligibility(bad, lanePolicy); err == nil || !strings.Contains(err.Error(), "total loss increased") {
		t.Fatalf("total-loss regression error = %v, want transactional epsilon rejection", err)
	}

	good := metrics
	goodFinal := initial
	goodFinal.Q3Gain -= float32(1e-6)
	goodFinal.Q3ScoreDistill += float32(1e-7)
	goodFinal.Q5ScoreDistill += float32(1e-7)
	good.Summary.FinalObjectiveComponents = goodFinal
	good.Summary.InitialLoss = initial.Sum()
	good.Summary.FinalLoss = goodFinal.Sum()
	if delta := good.Summary.FinalLoss - good.Summary.InitialLoss; delta > aoqtTransactionalLossEpsilon {
		t.Fatalf("non-regression fixture total loss delta = %.9g, want <= epsilon %.9g", delta, aoqtTransactionalLossEpsilon)
	}
	if err := good.Validate(); err != nil {
		t.Fatalf("non-regression fixture metrics: %v", err)
	}
	if err := ValidateAOQTSidecarCandidateEligibility(good, lanePolicy); err != nil {
		t.Fatalf("total-loss delta within transactional epsilon: %v", err)
	}
}

func TestAOQTOptimizerDiagnosticsRequireKnownCoordinateSearchStrategy(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 89)
	enableTinyAOQTNFGuard(t, &set)
	activation := AOQTSidecarObjectiveActivation{
		Q3GainEligiblePairs:         2,
		Q3GainContributingPairs:     2,
		Q3OrderGuardPairs:           2,
		Q3OrderGuardContributing:    2,
		Q3ScoreDistillCount:         3,
		Q5OrderGuardPairs:           2,
		Q5OrderGuardContributing:    2,
		Q5ScoreDistillCount:         3,
		NFBoundaryGuardPairs:        1,
		NFBoundaryGuardContributing: 1,
	}
	metrics := safeTinyAOQTCandidateMetrics(t, set, activation)
	seedCoordinateDiagnosticsForMetricsTest(t, &metrics, aoqtCoordinateSearchStrategyProtectedConeMicroTail)
	if err := metrics.Validate(); err != nil {
		t.Fatalf("valid coordinate diagnostics: %v", err)
	}

	for _, tc := range []struct {
		name     string
		strategy string
	}{
		{name: "missing", strategy: ""},
		{name: "wrong", strategy: "aggregate_abs_v0"},
		{name: "superseded-q3-only", strategy: "q3_gain_primary_fine_tail_v1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bad := metrics
			diagnostics := *bad.Summary.OptimizerDiagnostics
			diagnostics.CoordinateSearchStrategy = tc.strategy
			bad.Summary.OptimizerDiagnostics = &diagnostics
			refreshAOQTDiagnosticsSHAForMetricsTest(t, &bad)
			err := bad.Validate()
			if err == nil || !strings.Contains(err.Error(), "coordinate_search_strategy") || !strings.Contains(err.Error(), aoqtCoordinateSearchStrategyProtectedConeMicroTail) {
				t.Fatalf("error = %v, want coordinate strategy rejection", err)
			}
		})
	}
}

func TestAOQTProtectedCoordinateDiagnosticsAreStrictlyBound(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 90)
	enableTinyAOQTNFGuard(t, &set)
	activation := AOQTSidecarObjectiveActivation{
		Q3GainEligiblePairs:         2,
		Q3GainContributingPairs:     2,
		Q3OrderGuardPairs:           2,
		Q3OrderGuardContributing:    2,
		Q3ScoreDistillCount:         3,
		Q5OrderGuardPairs:           2,
		Q5OrderGuardContributing:    2,
		Q5ScoreDistillCount:         3,
		NFBoundaryGuardPairs:        1,
		NFBoundaryGuardContributing: 1,
	}
	metrics := safeTinyAOQTCandidateMetrics(t, set, activation)
	seedCoordinateDiagnosticsForMetricsTest(t, &metrics, aoqtCoordinateSearchStrategyProtectedConeMicroTail)
	for _, tc := range []struct {
		name   string
		mutate func(*AOQTSidecarOptimizerDiagnostics)
		want   string
	}{
		{
			name: "arbitrary-hash",
			mutate: func(d *AOQTSidecarOptimizerDiagnostics) {
				d.CoordinateSearchOrderingHash = hex64("opaque-hash")
			},
			want: "does not match hash chain tail",
		},
		{
			name: "wrong-count",
			mutate: func(d *AOQTSidecarOptimizerDiagnostics) {
				d.CoordinateMagnitudeCount = 1
			},
			want: "exact micro-tail count",
		},
		{
			name: "missing-active-component",
			mutate: func(d *AOQTSidecarOptimizerDiagnostics) {
				var payload aoqtCoordinateSearchAuditPayload
				if err := strictUnmarshalAOQT([]byte(d.CoordinateSearchAudit), &payload); err != nil {
					t.Fatalf("decode audit: %v", err)
				}
				payload.ProtectedComponents = payload.ProtectedComponents[:len(payload.ProtectedComponents)-1]
				bindAOQTCoordinateAuditForMetricsTest(t, d, payload)
			},
			want: "protected components",
		},
		{
			name: "chain-truncation",
			mutate: func(d *AOQTSidecarOptimizerDiagnostics) {
				d.CoordinateSearchAuditChain = "[]"
			},
			want: "chain must be non-empty",
		},
		{
			name: "plan-count-chain-mismatch",
			mutate: func(d *AOQTSidecarOptimizerDiagnostics) {
				d.CoordinateSearchPlanCount = 2
			},
			want: "plan_count",
		},
		{
			name: "guard-conflict-semantic-binding",
			mutate: func(d *AOQTSidecarOptimizerDiagnostics) {
				var payload aoqtCoordinateSearchAuditPayload
				if err := strictUnmarshalAOQT([]byte(d.CoordinateSearchAudit), &payload); err != nil {
					t.Fatalf("decode audit: %v", err)
				}
				payload.Order[0].GuardConflict = true
				bindAOQTCoordinateAuditForMetricsTest(t, d, payload)
			},
			want: "guard conflict",
		},
		{
			name: "full-order-hash-binding",
			mutate: func(d *AOQTSidecarOptimizerDiagnostics) {
				var payload aoqtCoordinateSearchAuditPayload
				if err := strictUnmarshalAOQT([]byte(d.CoordinateSearchAudit), &payload); err != nil {
					t.Fatalf("decode audit: %v", err)
				}
				payload.FullOrder[0].Abs *= 2
				bindAOQTCoordinateAuditForMetricsTest(t, d, payload)
			},
			want: "full order sha256",
		},
		{
			name: "full-order-truncation-rebuilt",
			mutate: func(d *AOQTSidecarOptimizerDiagnostics) {
				var payload aoqtCoordinateSearchAuditPayload
				if err := strictUnmarshalAOQT([]byte(d.CoordinateSearchAudit), &payload); err != nil {
					t.Fatalf("decode audit: %v", err)
				}
				payload.FullOrder = payload.FullOrder[:1]
				fullOrderJSON, err := json.Marshal(payload.FullOrder)
				if err != nil {
					t.Fatalf("marshal truncated full order: %v", err)
				}
				sum := sha256.Sum256(fullOrderJSON)
				payload.FullOrderSHA256 = hex.EncodeToString(sum[:])
				bindAOQTCoordinateAuditForMetricsTest(t, d, payload)
			},
			want: "full order count",
		},
		{
			name: "full-rank-source-binding",
			mutate: func(d *AOQTSidecarOptimizerDiagnostics) {
				var payload aoqtCoordinateSearchAuditPayload
				if err := strictUnmarshalAOQT([]byte(d.CoordinateSearchAudit), &payload); err != nil {
					t.Fatalf("decode audit: %v", err)
				}
				payload.FullRankSourceSHA256 = hex64("tampered-rank-source")
				bindAOQTCoordinateAuditForMetricsTest(t, d, payload)
			},
			want: "full rank source sha256",
		},
		{
			name: "full-rank-source-recomputed-tamper",
			mutate: func(d *AOQTSidecarOptimizerDiagnostics) {
				var payload aoqtCoordinateSearchAuditPayload
				if err := strictUnmarshalAOQT([]byte(d.CoordinateSearchAudit), &payload); err != nil {
					t.Fatalf("decode audit: %v", err)
				}
				payload.FullOrder[0].Abs *= 2
				fullOrderJSON, err := json.Marshal(payload.FullOrder)
				if err != nil {
					t.Fatalf("marshal tampered full order: %v", err)
				}
				sum := sha256.Sum256(fullOrderJSON)
				payload.FullOrderSHA256 = hex.EncodeToString(sum[:])
				bindAOQTCoordinateAuditForMetricsTest(t, d, payload)
			},
			want: "full rank source sha256",
		},
		{
			name: "learning-rate-binding",
			mutate: func(d *AOQTSidecarOptimizerDiagnostics) {
				var payload aoqtCoordinateSearchAuditPayload
				if err := strictUnmarshalAOQT([]byte(d.CoordinateSearchAudit), &payload); err != nil {
					t.Fatalf("decode audit: %v", err)
				}
				payload.LearningRate *= 2
				bindAOQTCoordinateAuditForMetricsTest(t, d, payload)
			},
			want: "learning_rate",
		},
		{
			name: "top-order-truncation",
			mutate: func(d *AOQTSidecarOptimizerDiagnostics) {
				var payload aoqtCoordinateSearchAuditPayload
				if err := strictUnmarshalAOQT([]byte(d.CoordinateSearchAudit), &payload); err != nil {
					t.Fatalf("decode audit: %v", err)
				}
				payload.Order = payload.Order[:1]
				payload.BlockSizes = nil
				d.CoordinateTopAngles = 1
				d.CoordinateBlockCount = 0
				bindAOQTCoordinateAuditForMetricsTest(t, d, payload)
			},
			want: "exact top-order count",
		},
		{
			name: "empty-blocks",
			mutate: func(d *AOQTSidecarOptimizerDiagnostics) {
				var payload aoqtCoordinateSearchAuditPayload
				if err := strictUnmarshalAOQT([]byte(d.CoordinateSearchAudit), &payload); err != nil {
					t.Fatalf("decode audit: %v", err)
				}
				payload.BlockSizes = nil
				bindAOQTCoordinateAuditForMetricsTest(t, d, payload)
			},
			want: "protected block sizes",
		},
		{
			name: "rejection-reason-count",
			mutate: func(d *AOQTSidecarOptimizerDiagnostics) {
				d.RejectionDiagnostics.ReasonCounts.LossIncrease++
			},
			want: "reason_counts sum",
		},
		{
			name: "rejection-component-count",
			mutate: func(d *AOQTSidecarOptimizerDiagnostics) {
				d.RejectionDiagnostics.ComponentRegressionCounts.Q3ScoreDistill = 1
			},
			want: "component_regression_counts are non-zero",
		},
		{
			name: "rejection-delta-count",
			mutate: func(d *AOQTSidecarOptimizerDiagnostics) {
				d.RejectionDiagnostics.LossDelta.Count = -1
			},
			want: "loss_delta.count must be non-negative",
		},
		{
			name: "rejection-delta-sum",
			mutate: func(d *AOQTSidecarOptimizerDiagnostics) {
				d.RejectionDiagnostics.LossDelta.Sum = 2
			},
			want: "loss_delta.sum",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bad := metrics
			diagnostics := *bad.Summary.OptimizerDiagnostics
			tc.mutate(&diagnostics)
			bad.Summary.OptimizerDiagnostics = &diagnostics
			refreshAOQTDiagnosticsSHAForMetricsTest(t, &bad)
			err := bad.Validate()
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want strict protected audit rejection containing %q", err, tc.want)
			}
		})
	}
	legacy := metrics
	legacy.Schema = "eos.q3_aoqt_sidecar_metrics.v1"
	if err := legacy.Validate(); err == nil || !strings.Contains(err.Error(), AOQTSidecarMetricsSchema) {
		t.Fatalf("legacy metrics schema error = %v, want explicit v2 incompatibility", err)
	}
}

func TestAOQTV2MetricsRejectsLegacyQ3OnlyAdamSummary(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 96)
	metrics := safeTinyAOQTCandidateMetrics(t, set, AOQTSidecarObjectiveActivation{
		Q3GainEligiblePairs:     2,
		Q3GainContributingPairs: 2,
	})
	q3OnlyContract := tinyAOQTObjectiveConfig(set.Manifest.TurboQuantSeed).ObjectiveContract(AOQTSidecarRowWeights{
		Q3Gain: metrics.ObjectiveContract.WeightSums.Q3Gain,
	})
	metrics.ObjectiveContract = q3OnlyContract
	metrics.Summary.ObjectiveContract = q3OnlyContract
	refreshAOQTDiagnosticsSHAForMetricsTest(t, &metrics)
	if err := metrics.Validate(); err == nil || !strings.Contains(err.Error(), "active protected objective component") {
		t.Fatalf("q3-only v2 metrics error = %v, want unconditional protected-component rejection", err)
	}
}

func TestAOQTOptimizerDiagnosticsRejectStrippedPathAccounting(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 97)
	enableTinyAOQTNFGuard(t, &set)
	metrics := safeTinyAOQTCandidateMetrics(t, set, AOQTSidecarObjectiveActivation{
		Q3GainEligiblePairs:         2,
		Q3GainContributingPairs:     2,
		Q3OrderGuardPairs:           2,
		Q3OrderGuardContributing:    2,
		Q3ScoreDistillCount:         3,
		Q5OrderGuardPairs:           2,
		Q5OrderGuardContributing:    2,
		Q5ScoreDistillCount:         3,
		NFBoundaryGuardPairs:        1,
		NFBoundaryGuardContributing: 1,
	})
	seedCoordinateDiagnosticsForMetricsTest(t, &metrics, aoqtCoordinateSearchStrategyProtectedConeMicroTail)
	diagnostics := *metrics.Summary.OptimizerDiagnostics
	diagnostics.AdamProposalAttempts = 0
	diagnostics.AdamAcceptedProposals = 0
	diagnostics.AdamRejectedProposals = 0
	diagnostics.CoordinateProposalAttempts = 0
	diagnostics.CoordinateAcceptedProposals = 0
	diagnostics.CoordinateRejectedProposals = 0
	diagnostics.CoordinateSearchPlanCount = 0
	diagnostics.CoordinateTopAngles = 0
	diagnostics.CoordinateMagnitudeCount = 0
	diagnostics.CoordinateBlockCount = 0
	diagnostics.CoordinateSearchStrategy = ""
	diagnostics.CoordinateSearchOrderingHash = ""
	diagnostics.CoordinateSearchLearningRate = 0
	diagnostics.CoordinateSearchAudit = ""
	diagnostics.CoordinateSearchAuditChain = ""
	diagnostics.CoordinateSearchHashChain = ""
	metrics.Summary.OptimizerDiagnostics = &diagnostics
	refreshAOQTDiagnosticsSHAForMetricsTest(t, &metrics)
	if err := metrics.Validate(); err == nil || !strings.Contains(err.Error(), "top-level proposal accounting requires optimizer path diagnostics") {
		t.Fatalf("stripped optimizer path error = %v, want fail-closed top-level accounting rejection", err)
	}
}

func TestAOQTOptimizerDiagnosticsRejectNegativePathCounter(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 99)
	metrics := safeTinyAOQTCandidateMetrics(t, set, AOQTSidecarObjectiveActivation{
		Q3GainEligiblePairs:      2,
		Q3GainContributingPairs:  2,
		Q3OrderGuardPairs:        2,
		Q3OrderGuardContributing: 2,
		Q3ScoreDistillCount:      3,
		Q5OrderGuardPairs:        2,
		Q5OrderGuardContributing: 2,
		Q5ScoreDistillCount:      3,
	})
	diagnostics := *metrics.Summary.OptimizerDiagnostics
	diagnostics.AdamProposalAttempts = -1
	metrics.Summary.OptimizerDiagnostics = &diagnostics
	refreshAOQTDiagnosticsSHAForMetricsTest(t, &metrics)
	if err := metrics.Validate(); err == nil || !strings.Contains(err.Error(), "adam_proposal_attempts must be non-negative") {
		t.Fatalf("negative Adam counter error = %v, want explicit non-negative rejection", err)
	}
}

func TestAOQTOptimizerDiagnosticsRejectIncompleteNonExhaustedPlan(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 100)
	metrics := safeTinyAOQTCandidateMetrics(t, set, AOQTSidecarObjectiveActivation{
		Q3GainEligiblePairs:      2,
		Q3GainContributingPairs:  2,
		Q3OrderGuardPairs:        2,
		Q3OrderGuardContributing: 2,
		Q3ScoreDistillCount:      3,
		Q5OrderGuardPairs:        2,
		Q5OrderGuardContributing: 2,
		Q5ScoreDistillCount:      3,
	})
	bad := metrics
	bad.Plan.StepCount++
	bad.Summary.Plan.StepCount++
	diagnostics := *bad.Summary.OptimizerDiagnostics
	diagnostics.PlannedSteps++
	bad.Summary.OptimizerDiagnostics = &diagnostics
	refreshAOQTDiagnosticsSHAForMetricsTest(t, &bad)
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "want planned_steps") {
		t.Fatalf("incomplete non-exhausted plan error = %v, want planned/attempted termination rejection", err)
	}
}

func TestAOQTMetricsRejectNegativeActivationCounts(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 101)
	metrics := safeTinyAOQTCandidateMetrics(t, set, AOQTSidecarObjectiveActivation{
		Q3GainEligiblePairs:      2,
		Q3GainContributingPairs:  2,
		Q3OrderGuardPairs:        2,
		Q3OrderGuardContributing: 2,
		Q3ScoreDistillCount:      3,
		Q5OrderGuardPairs:        2,
		Q5OrderGuardContributing: 2,
		Q5ScoreDistillCount:      3,
	})
	metrics.Summary.InitialObjectiveActivation.Q3GainEligiblePairs = -1
	if err := metrics.Validate(); err == nil || !strings.Contains(err.Error(), "initial_objective_activation.q3_gain_eligible_pairs must be non-negative") {
		t.Fatalf("negative activation error = %v, want explicit non-negative rejection", err)
	}
}
func bindAOQTCoordinateAuditForMetricsTest(t *testing.T, diagnostics *AOQTSidecarOptimizerDiagnostics, payload aoqtCoordinateSearchAuditPayload) {
	t.Helper()
	audit, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal coordinate audit: %v", err)
	}
	diagnostics.CoordinateSearchAudit = string(audit)
	diagnostics.CoordinateSearchAuditChain, err = appendAOQTCoordinateSearchAuditChain("", string(audit))
	if err != nil {
		t.Fatalf("coordinate audit chain: %v", err)
	}
	diagnostics.CoordinateSearchHashChain, err = appendAOQTCoordinateSearchHashChain("", string(audit))
	if err != nil {
		t.Fatalf("coordinate hash chain: %v", err)
	}
	var hashes []string
	if err := strictUnmarshalAOQT([]byte(diagnostics.CoordinateSearchHashChain), &hashes); err != nil || len(hashes) == 0 {
		t.Fatalf("coordinate hash chain decode: %v", err)
	}
	diagnostics.CoordinateSearchOrderingHash = hashes[len(hashes)-1]
}

func seedCoordinateDiagnosticsForMetricsTest(t *testing.T, metrics *AOQTSidecarRunMetrics, strategy string) {
	t.Helper()
	diagnostics := *metrics.Summary.OptimizerDiagnostics
	diagnostics.ProposalAttempts = diagnostics.AcceptedSteps + 1
	diagnostics.AcceptedProposals = diagnostics.AcceptedSteps
	diagnostics.RejectedProposals = 1
	diagnostics.Backtracks = 1
	diagnostics.AdamProposalAttempts = diagnostics.AcceptedSteps
	diagnostics.AdamAcceptedProposals = diagnostics.AcceptedSteps
	diagnostics.CoordinateProposalAttempts = 1
	diagnostics.CoordinateRejectedProposals = 1
	diagnostics.RejectionDiagnostics.CandidateEvaluations = 1
	diagnostics.RejectionDiagnostics.ReasonCounts.LossIncrease = 1
	diagnostics.RejectionDiagnostics.LossDelta.Record(1)
	diagnostics.RejectionDiagnostics.ComponentDeltas.Record(
		AOQTSidecarObjectiveComponents{Q3Gain: 1, Q3OrderGuard: 1, Q3ScoreDistill: 1, Q5OrderGuard: 1, Q5ScoreDistill: 1, NFBoundaryGuard: 1},
		AOQTSidecarObjectiveComponents{Q3Gain: 2, Q3OrderGuard: 2, Q3ScoreDistill: 2, Q5OrderGuard: 2, Q5ScoreDistill: 2, NFBoundaryGuard: 2},
	)
	diagnostics.RejectionDiagnostics.finalizeDominantFields()
	diagnostics.CoordinateSearchPlanCount = 1
	diagnostics.CoordinateSearchStrategy = strategy
	diagnostics.CoordinateSearchLearningRate = 0.01
	protectedComponents := aoqtActiveProtectedComponentNames(metrics.ObjectiveContract.WeightSums)
	q3GainGrad := make([]float32, metrics.Plan.AngleCount)
	q3GainGrad[0] = 1
	q3GainGrad[1] = -0.5
	aggregateGrad := append([]float32(nil), q3GainGrad...)
	protected := make([]aoqtProtectedAngleGradient, 0, len(protectedComponents))
	for _, name := range protectedComponents {
		protected = append(protected, aoqtProtectedAngleGradient{Name: name, Grad: make([]float32, metrics.Plan.AngleCount)})
	}
	auditPlan, err := newAOQTCoordinateSearchPlan(q3GainGrad, aggregateGrad, 0.01, protected)
	if err != nil {
		t.Fatalf("coordinate audit plan: %v", err)
	}
	auditPlan.Strategy = strategy
	diagnostics.CoordinateTopAngles = len(auditPlan.Order)
	diagnostics.CoordinateMagnitudeCount = len(auditPlan.Magnitudes)
	diagnostics.CoordinateBlockCount = len(auditPlan.BlockSizes)
	audit, err := auditPlan.coordinateSearchAuditJSON()
	if err != nil {
		t.Fatalf("coordinate audit: %v", err)
	}
	diagnostics.CoordinateSearchAudit = string(audit)
	diagnostics.CoordinateSearchAuditChain, err = appendAOQTCoordinateSearchAuditChain("", string(audit))
	if err != nil {
		t.Fatalf("coordinate audit chain: %v", err)
	}
	diagnostics.CoordinateSearchHashChain, err = appendAOQTCoordinateSearchHashChain("", string(audit))
	if err != nil {
		t.Fatalf("coordinate hash chain: %v", err)
	}
	var hashes []string
	if err := strictUnmarshalAOQT([]byte(diagnostics.CoordinateSearchHashChain), &hashes); err != nil || len(hashes) == 0 {
		t.Fatalf("coordinate hash chain decode: %v", err)
	}
	diagnostics.CoordinateSearchOrderingHash = hashes[len(hashes)-1]
	metrics.Summary.OptimizerDiagnostics = &diagnostics
	refreshAOQTDiagnosticsSHAForMetricsTest(t, metrics)
}

func refreshAOQTDiagnosticsSHAForMetricsTest(t *testing.T, metrics *AOQTSidecarRunMetrics) {
	t.Helper()
	sha, err := metrics.Summary.OptimizerDiagnostics.SHA256()
	if err != nil {
		t.Fatalf("diagnostics sha: %v", err)
	}
	metrics.Summary.OptimizerDiagnosticsSHA256 = sha
	metrics.Summary.Steps = metrics.Summary.OptimizerDiagnostics.AcceptedSteps
}
