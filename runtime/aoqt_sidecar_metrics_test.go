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
