package eosruntime

import (
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
	seedCoordinateDiagnosticsForMetricsTest(t, &metrics, aoqtCoordinateSearchStrategyQ3GainPrimary)
	if err := metrics.Validate(); err != nil {
		t.Fatalf("valid coordinate diagnostics: %v", err)
	}

	for _, tc := range []struct {
		name     string
		strategy string
	}{
		{name: "missing", strategy: ""},
		{name: "wrong", strategy: "aggregate_abs_v0"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bad := metrics
			diagnostics := *bad.Summary.OptimizerDiagnostics
			diagnostics.CoordinateSearchStrategy = tc.strategy
			bad.Summary.OptimizerDiagnostics = &diagnostics
			refreshAOQTDiagnosticsSHAForMetricsTest(t, &bad)
			err := bad.Validate()
			if err == nil || !strings.Contains(err.Error(), "coordinate_search_strategy") || !strings.Contains(err.Error(), aoqtCoordinateSearchStrategyQ3GainPrimary) {
				t.Fatalf("error = %v, want coordinate strategy rejection", err)
			}
		})
	}
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
	diagnostics.CoordinateTopAngles = 1
	diagnostics.CoordinateMagnitudeCount = 1
	diagnostics.CoordinateBlockCount = 0
	diagnostics.CoordinateSearchStrategy = strategy
	diagnostics.CoordinateSearchOrderingHash = hex64("coordinate-ordering")
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
