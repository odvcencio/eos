package eosruntime

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAOQTCalibrationIOLoadsStrictManifestAndRows(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 81)
	manifestPath, rowsPath := writeAOQTCalibrationIOFixture(t, set)
	manifestSHA, rowsSHA := mustSHA256FileAOQTTest(t, manifestPath), mustSHA256FileAOQTTest(t, rowsPath)

	loaded, report, err := LoadAOQTSidecarCalibrationSet(AOQTSidecarCalibrationIOConfig{
		ManifestPath:                        manifestPath,
		RowsJSONLPath:                       rowsPath,
		ExpectedManifestSHA256:              manifestSHA,
		ExpectedRowsSHA256:                  rowsSHA,
		ExpectedAnchorArtifactSHA256:        set.Manifest.AnchorArtifactSHA256,
		ExpectedAnchorPackageManifestSHA256: set.Manifest.AnchorPackageManifestSHA256,
		ExpectedAnchorEmbeddingSpaceID:      set.Manifest.AnchorEmbeddingSpaceID,
		ExpectedCompatibilityDigest:         set.Manifest.CompatibilityDigest,
		ExpectedTurboQuantSeed:              set.Manifest.TurboQuantSeed,
		ExpectedTopology:                    set.Manifest.Topology,
		ExpectedSourceArtifactHashes:        append([]string(nil), set.Manifest.SourceArtifactHashes...),
		ExpectedVectorCacheHashes:           append([]string(nil), set.Manifest.VectorCacheHashes...),
	})
	if err != nil {
		t.Fatalf("load calibration set: %v", err)
	}
	if report.Schema != AOQTSidecarCalibrationIOReportSchema || report.ManifestSHA256 != manifestSHA || report.RowsSHA256 != rowsSHA {
		t.Fatalf("unexpected IO report: %+v", report)
	}
	if report.RowCount != len(set.Rows) || report.CandidateCount != len(set.Rows[0].CandidateDocIDs) || report.PairCount != countEligibleAOQTPairs(set.Rows[0]) {
		t.Fatalf("unexpected IO counts: %+v", report)
	}
	if loaded.Manifest.CompatibilityDigest != set.Manifest.CompatibilityDigest || loaded.Rows[0].RowID != set.Rows[0].RowID {
		t.Fatalf("loaded calibration set lost manifest/row identity")
	}
}

func TestAOQTCalibrationIORejectsMalformedAndHashMismatch(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 82)
	manifestPath, rowsPath := writeAOQTCalibrationIOFixture(t, set)
	if _, _, err := LoadAOQTSidecarCalibrationSet(AOQTSidecarCalibrationIOConfig{
		ManifestPath:           manifestPath,
		RowsJSONLPath:          rowsPath,
		ExpectedManifestSHA256: hex64("wrong-manifest"),
	}); err == nil || !strings.Contains(err.Error(), "manifest sha256") {
		t.Fatalf("manifest hash error = %v, want mismatch rejection", err)
	}

	badRowsPath := filepath.Join(t.TempDir(), "rows.jsonl")
	if err := os.WriteFile(badRowsPath, []byte(`{"schema":"eos.q3_aoqt_sidecar_row.v1","unknown":true}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	badRowsSHA := mustSHA256FileAOQTTest(t, badRowsPath)
	if _, _, err := LoadAOQTSidecarCalibrationSet(AOQTSidecarCalibrationIOConfig{
		ManifestPath:                        manifestPath,
		RowsJSONLPath:                       badRowsPath,
		ExpectedManifestSHA256:              mustSHA256FileAOQTTest(t, manifestPath),
		ExpectedRowsSHA256:                  badRowsSHA,
		ExpectedAnchorArtifactSHA256:        set.Manifest.AnchorArtifactSHA256,
		ExpectedAnchorPackageManifestSHA256: set.Manifest.AnchorPackageManifestSHA256,
		ExpectedAnchorEmbeddingSpaceID:      set.Manifest.AnchorEmbeddingSpaceID,
		ExpectedCompatibilityDigest:         set.Manifest.CompatibilityDigest,
		ExpectedTurboQuantSeed:              set.Manifest.TurboQuantSeed,
		ExpectedTopology:                    set.Manifest.Topology,
		ExpectedSourceArtifactHashes:        append([]string(nil), set.Manifest.SourceArtifactHashes...),
		ExpectedVectorCacheHashes:           append([]string(nil), set.Manifest.VectorCacheHashes...),
	}); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("malformed rows error = %v, want strict JSON rejection", err)
	}

	if _, _, err := LoadAOQTSidecarCalibrationSet(AOQTSidecarCalibrationIOConfig{
		ManifestPath:                        manifestPath,
		RowsJSONLPath:                       rowsPath,
		ExpectedManifestSHA256:              mustSHA256FileAOQTTest(t, manifestPath),
		ExpectedRowsSHA256:                  mustSHA256FileAOQTTest(t, rowsPath),
		ExpectedAnchorArtifactSHA256:        set.Manifest.AnchorArtifactSHA256,
		ExpectedAnchorPackageManifestSHA256: hex64("wrong-package"),
		ExpectedAnchorEmbeddingSpaceID:      set.Manifest.AnchorEmbeddingSpaceID,
		ExpectedCompatibilityDigest:         set.Manifest.CompatibilityDigest,
		ExpectedTurboQuantSeed:              set.Manifest.TurboQuantSeed,
		ExpectedTopology:                    set.Manifest.Topology,
		ExpectedSourceArtifactHashes:        append([]string(nil), set.Manifest.SourceArtifactHashes...),
		ExpectedVectorCacheHashes:           append([]string(nil), set.Manifest.VectorCacheHashes...),
	}); err == nil || !strings.Contains(err.Error(), "anchor_package_manifest_sha256") {
		t.Fatalf("provenance error = %v, want anchor package binding rejection", err)
	}
}

func TestAOQTCalibrationIORejectsDuplicateKeysAndMissingBindings(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 85)
	manifestPath, rowsPath := writeAOQTCalibrationIOFixture(t, set)
	rowsSHA := mustSHA256FileAOQTTest(t, rowsPath)

	duplicateManifest := filepath.Join(t.TempDir(), "manifest.json")
	if err := os.WriteFile(duplicateManifest, []byte(`{"schema":"eos.q3_aoqt_sidecar_manifest.v1","extra":{"x":1,"x":2}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := LoadAOQTSidecarCalibrationSet(AOQTSidecarCalibrationIOConfig{
		ManifestPath:           duplicateManifest,
		RowsJSONLPath:          rowsPath,
		ExpectedManifestSHA256: mustSHA256FileAOQTTest(t, duplicateManifest),
		ExpectedRowsSHA256:     rowsSHA,
	}); err == nil || !strings.Contains(err.Error(), "duplicate object key") {
		t.Fatalf("duplicate manifest key error = %v, want duplicate-key rejection", err)
	}

	duplicateRows := filepath.Join(t.TempDir(), "rows.jsonl")
	if err := os.WriteFile(duplicateRows, []byte(`{"schema":"eos.q3_aoqt_sidecar_row.v1","anchor_scores":{"q3":[1],"q3":[2]}}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := LoadAOQTSidecarCalibrationSet(AOQTSidecarCalibrationIOConfig{
		ManifestPath:           manifestPath,
		RowsJSONLPath:          duplicateRows,
		ExpectedManifestSHA256: mustSHA256FileAOQTTest(t, manifestPath),
		ExpectedRowsSHA256:     mustSHA256FileAOQTTest(t, duplicateRows),
	}); err == nil || !strings.Contains(err.Error(), "duplicate object key") {
		t.Fatalf("duplicate row key error = %v, want duplicate-key rejection", err)
	}

	if _, _, err := LoadAOQTSidecarCalibrationSet(AOQTSidecarCalibrationIOConfig{
		ManifestPath:           manifestPath,
		RowsJSONLPath:          rowsPath,
		ExpectedManifestSHA256: mustSHA256FileAOQTTest(t, manifestPath),
		ExpectedRowsSHA256:     rowsSHA,
	}); err == nil || !strings.Contains(err.Error(), "expected anchor_artifact_sha256 is required") {
		t.Fatalf("missing IO binding error = %v, want all high-level bindings required", err)
	}
}

func TestAOQTPreparedIPObjectiveReportsSeparateComponents(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 83)
	enableTinyAOQTNFGuard(t, &set)
	objective, err := NewAOQTSidecarPreparedIPObjective(tinyAOQTObjectiveConfig(set.Manifest.TurboQuantSeed))
	if err != nil {
		t.Fatalf("objective: %v", err)
	}
	trainer := newTinyAOQTTrainer(t, false, set.Manifest.Topology.Seed)
	angles := trainer.Angles()
	angles[0] = 0.01
	if err := trainer.SetAnglesForTest(angles); err != nil {
		t.Fatalf("set AOQT angle: %v", err)
	}
	query, candidates, err := transformAOQTRow(trainer.Transform(), set.Rows[0])
	if err != nil {
		t.Fatalf("transform row: %v", err)
	}
	result, err := objective.EvaluateAOQT(AOQTSidecarObjectiveInput{
		Row:        aoqtObjectiveRowView(set.Rows[0]),
		Query:      query,
		Candidates: cloneAOQTVectors(candidates),
	})
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	components := result.Components
	if components.Q3Gain <= 0 || components.Q3OrderGuard <= 0 || components.Q3ScoreDistill <= 0 ||
		components.Q5OrderGuard <= 0 || components.Q5ScoreDistill <= 0 || components.NFBoundaryGuard <= 0 {
		t.Fatalf("components = %+v, want every synthetic objective component active", components)
	}
	sum := components.Q3Gain + components.Q3OrderGuard + components.Q3ScoreDistill + components.Q5OrderGuard + components.Q5ScoreDistill + components.NFBoundaryGuard
	if !float32Near(sum, result.Loss, 1e-6) {
		t.Fatalf("component sum = %.9g, loss = %.9g", sum, result.Loss)
	}
}

func TestAOQTCandidateEligibilityRejectsUnsafeOutputsAndAllowsSafePass(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 84)
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
	if err := ValidateAOQTSidecarCandidateEligibility(metrics, AOQTSidecarCandidateEligibilityPolicy{RequireObjectiveActivation: true}); err != nil {
		t.Fatalf("safe candidate eligibility: %v", err)
	}

	for _, tc := range []struct {
		name   string
		mutate func(*AOQTSidecarRunMetrics)
		want   string
	}{
		{
			name: "dense",
			mutate: func(m *AOQTSidecarRunMetrics) {
				m.Summary.DenseMaxAbsDelta = 0.01
			},
			want: "dense_max_abs_delta",
		},
		{
			name: "angle",
			mutate: func(m *AOQTSidecarRunMetrics) {
				m.Summary.AngleMaxAbs = AOQTSidecarDefaultAngleCap * 2
			},
			want: "angle_max_abs",
		},
		{
			name: "inert",
			mutate: func(m *AOQTSidecarRunMetrics) {
				m.Summary.FinalObjectiveActivation.Q5OrderGuardContributing = 0
			},
			want: "q5_order_guard objective is inactive",
		},
		{
			name: "regression",
			mutate: func(m *AOQTSidecarRunMetrics) {
				m.Summary.FinalObjectiveComponents.NFBoundaryGuard = m.Summary.InitialObjectiveComponents.NFBoundaryGuard + 0.1
				m.Summary.FinalLoss = m.Summary.FinalObjectiveComponents.Sum()
			},
			want: "nf_boundary_guard component regressed",
		},
		{
			name: "dense-nan",
			mutate: func(m *AOQTSidecarRunMetrics) {
				m.Summary.DenseMaxAbsDelta = math.NaN()
			},
			want: "dense_max_abs_delta must be finite and non-negative",
		},
		{
			name: "dense-negative",
			mutate: func(m *AOQTSidecarRunMetrics) {
				m.Summary.DenseMaxAbsDelta = -1
			},
			want: "dense_max_abs_delta must be finite and non-negative",
		},
		{
			name: "loss-accounting",
			mutate: func(m *AOQTSidecarRunMetrics) {
				m.Summary.FinalLoss = m.Summary.FinalObjectiveComponents.Sum() + 0.25
			},
			want: "must equal component sum",
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

func writeAOQTCalibrationIOFixture(t *testing.T, set AOQTSidecarCalibrationSet) (string, string) {
	t.Helper()
	dir := t.TempDir()
	manifestPath := filepath.Join(dir, "manifest.json")
	rowsPath := filepath.Join(dir, "rows.jsonl")
	if err := writeJSONFileAOQT(manifestPath, set.Manifest); err != nil {
		t.Fatal(err)
	}
	if err := writeAOQTCalibrationRows(rowsPath, set.Rows); err != nil {
		t.Fatal(err)
	}
	return manifestPath, rowsPath
}

func mustSHA256FileAOQTTest(t *testing.T, path string) string {
	t.Helper()
	sum, err := sha256FileAOQT(path)
	if err != nil {
		t.Fatal(err)
	}
	return sum
}

func enableTinyAOQTNFGuard(t *testing.T, set *AOQTSidecarCalibrationSet) {
	t.Helper()
	set.Rows[0].Weights.NFBoundaryGuard = 0.75
	set.Rows[0].EligiblePairMask[2][1] = true
	set.Manifest.ObjectiveContract = tinyAOQTObjectiveConfig(set.Manifest.TurboQuantSeed).ObjectiveContract(sumAOQTRowWeights(set.Rows))
	if err := set.Validate(); err != nil {
		t.Fatalf("NF-enabled fixture invalid: %v", err)
	}
}

func safeTinyAOQTCandidateMetrics(t *testing.T, set AOQTSidecarCalibrationSet, activation AOQTSidecarObjectiveActivation) AOQTSidecarRunMetrics {
	t.Helper()
	trainer := newTinyAOQTTrainer(t, false, set.Manifest.Topology.Seed)
	plan, err := trainer.Plan(set)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	anglesSHA, err := trainer.Transform().AnglesSHA256()
	if err != nil {
		t.Fatalf("angles sha: %v", err)
	}
	summary := AOQTSidecarTrainSummary{
		Plan:                       plan,
		ObjectiveContract:          set.Manifest.ObjectiveContract,
		Steps:                      plan.StepCount,
		InitialLoss:                6,
		FinalLoss:                  3,
		InitialObjectiveComponents: AOQTSidecarObjectiveComponents{Q3Gain: 1, Q3OrderGuard: 1, Q3ScoreDistill: 1, Q5OrderGuard: 1, Q5ScoreDistill: 1, NFBoundaryGuard: 1},
		FinalObjectiveComponents:   AOQTSidecarObjectiveComponents{Q3Gain: 0.5, Q3OrderGuard: 0.5, Q3ScoreDistill: 0.5, Q5OrderGuard: 0.5, Q5ScoreDistill: 0.5, NFBoundaryGuard: 0.5},
		InitialObjectiveActivation: activation,
		FinalObjectiveActivation:   activation,
		AngleL2:                    0.01,
		AngleMaxAbs:                0.01,
		AnglesSHA256:               anglesSHA,
		DenseMaxAbsDelta:           0,
	}
	metrics, err := NewAOQTSidecarRunMetrics(set, summary)
	if err != nil {
		t.Fatalf("metrics: %v", err)
	}
	return metrics
}
