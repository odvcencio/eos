package eosruntime

import (
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
	diagnostics.CoordinateSearchPlanCount = 1
	diagnostics.CoordinateTopAngles = 1
	diagnostics.CoordinateMagnitudeCount = aoqtTransactionalCoordinateMicroTailMagnitudeCount
	diagnostics.CoordinateBlockCount = 0
	diagnostics.CoordinateSearchStrategy = strategy
	diagnostics.CoordinateSearchLearningRate = 0.01
	protectedComponents := aoqtActiveProtectedComponentNames(metrics.ObjectiveContract.WeightSums)
	microTail := []float32{0.00125, 0.000625, 0.0003125}
	auditPlan := aoqtCoordinateSearchPlan{
		Strategy:     strategy,
		LearningRate: 0.01,
		Order: []aoqtCoordinateSearchRank{{
			index:                           0,
			abs:                             1,
			aggregateAbs:                    1,
			aggregateGradient:               1,
			primaryDirection:                -1,
			protectedDirectionalDerivatives: make([]float32, len(protectedComponents)),
		}},
		FullOrder: []aoqtCoordinateSearchRank{{
			index:                           0,
			abs:                             1,
			aggregateAbs:                    1,
			aggregateGradient:               1,
			primaryDirection:                -1,
			protectedDirectionalDerivatives: make([]float32, len(protectedComponents)),
		}},
		Magnitudes:          microTail,
		BlockMagnitudes:     microTail,
		MicroTailMagnitudes: microTail,
		BlockSizes:          nil,
		ProtectedComponents: protectedComponents,
	}
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
