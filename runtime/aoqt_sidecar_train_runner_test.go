package eosruntime

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAOQTSidecarTrainRunnerAcceptsCanonicalMaterializerManifestDigest(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	preflight, err := MaterializeAOQTSidecarCalibration(materializerCfg)
	if err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	rawManifestSHA := mustSHA256FileAOQTTest(t, materializerCfg.ManifestJSONPath)
	if rawManifestSHA == preflight.CalibrationManifestSHA256 {
		t.Fatalf("fixture raw manifest SHA unexpectedly equals canonical SHA %s", rawManifestSHA)
	}

	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	result, err := RunAOQTSidecarTraining(trainCfg)
	if err != nil {
		t.Fatalf("run AOQT sidecar train preflight: %v", err)
	}
	if result.IOReport.ManifestSHA256 != preflight.CalibrationManifestSHA256 {
		t.Fatalf("IO manifest SHA = %s, want preflight canonical SHA %s", result.IOReport.ManifestSHA256, preflight.CalibrationManifestSHA256)
	}
	if result.Metrics.Inputs.DatasetManifestSHA256 != preflight.CalibrationManifestSHA256 {
		t.Fatalf("metrics dataset manifest SHA = %s, want canonical SHA %s", result.Metrics.Inputs.DatasetManifestSHA256, preflight.CalibrationManifestSHA256)
	}
	if _, err := os.Lstat(AOQTFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath)); !os.IsNotExist(err) {
		t.Fatalf("successful plan-only run fail-closed diagnostics state = %v, want absent", err)
	}
}

func TestAOQTSidecarTrainRunnerRejectsSemanticManifestTamper(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	manifest := readAOQTManifestForTrainRunnerTest(t, materializerCfg.ManifestJSONPath)
	manifest.CreatedAtUTC = "2030-01-01T00:00:00Z"
	if err := writeJSONFileAOQT(materializerCfg.ManifestJSONPath, manifest); err != nil {
		t.Fatal(err)
	}

	if _, err := RunAOQTSidecarTraining(trainCfg); err == nil || !strings.Contains(err.Error(), "AOQT calibration manifest sha256") {
		t.Fatalf("semantic manifest tamper error = %v, want manifest digest rejection", err)
	}
}

func TestAOQTSidecarTrainRunnerRejectsPreflightManifestDigestDrift(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	preflight, err := MaterializeAOQTSidecarCalibration(materializerCfg)
	if err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	rawManifestSHA := mustSHA256FileAOQTTest(t, materializerCfg.ManifestJSONPath)
	if rawManifestSHA == preflight.CalibrationManifestSHA256 {
		t.Fatalf("fixture raw manifest SHA unexpectedly equals canonical SHA %s", rawManifestSHA)
	}
	preflight.CalibrationManifestSHA256 = rawManifestSHA
	if err := writeJSONFileAOQT(materializerCfg.PreflightJSONPath, preflight); err != nil {
		t.Fatal(err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)

	if _, err := RunAOQTSidecarTraining(trainCfg); err == nil || !strings.Contains(err.Error(), "preflight calibration manifest sha256 mismatch") {
		t.Fatalf("preflight manifest digest drift error = %v, want preflight mismatch rejection", err)
	}
}

func TestAOQTSidecarTrainRunnerRejectsRawPreflightTamper(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	raw, err := os.ReadFile(trainCfg.PreflightJSONPath)
	if err != nil {
		t.Fatalf("read preflight JSON: %v", err)
	}
	if err := os.WriteFile(trainCfg.PreflightJSONPath, append(raw, []byte(" \n")...), 0o644); err != nil {
		t.Fatalf("tamper preflight JSON: %v", err)
	}
	trainCfg.PlanOnly = false
	trainCfg.MaxSteps = 1
	trainCfg.OutputArtifactPath = filepath.Join(t.TempDir(), "candidate.mll")
	called := false
	_, err = runAOQTSidecarTraining(trainCfg, func(AOQTSidecarObjectiveContract) (AOQTSidecarVectorObjective, error) {
		called = true
		return nil, fmt.Errorf("objective factory must not run after raw preflight tamper")
	})
	if err == nil || !strings.Contains(err.Error(), "preflight sha256 mismatch") {
		t.Fatalf("raw preflight tamper error = %v, want raw SHA rejection", err)
	}
	if called {
		t.Fatalf("objective factory ran after raw preflight tamper")
	}
	if _, err := os.Lstat(trainCfg.MetricsJSONPath); !os.IsNotExist(err) {
		t.Fatalf("metrics output state = %v, want absent", err)
	}
	if _, err := os.Lstat(AOQTFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath)); !os.IsNotExist(err) {
		t.Fatalf("fail-closed diagnostics state = %v, want absent", err)
	}
}

func TestAOQTSidecarTrainRunnerRejectsZeroAcceptedWithoutPackage(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	trainCfg.PlanOnly = false
	trainCfg.MaxSteps = 1
	trainCfg.OutputArtifactPath = filepath.Join(t.TempDir(), "candidate.mll")
	objectiveFactory := func(contract AOQTSidecarObjectiveContract) (AOQTSidecarVectorObjective, error) {
		objective, err := objectiveFromAOQTContract(contract)
		if err != nil {
			return nil, err
		}
		stateful := &statefulRejectingAOQTObjective{config: objective.AOQTPreparedIPObjectiveConfig()}
		objective.workspace.evaluateOverride = stateful.EvaluateAOQT
		return objective, nil
	}

	result, err := runAOQTSidecarTraining(trainCfg, objectiveFactory)
	if err == nil || !strings.Contains(err.Error(), "accepted zero safe steps") || !strings.Contains(err.Error(), "rejection diagnostics: dominant_reason=") {
		t.Fatalf("run AOQT sidecar training error = %v, want zero-accepted failure with rejection diagnostics", err)
	}
	if result.FailureDiagnostics == nil {
		t.Fatalf("runner did not return fail-closed diagnostics")
	}
	if err := result.FailureDiagnostics.Validate(); err != nil {
		t.Fatalf("returned fail-closed diagnostics invalid: %v", err)
	}
	if result.FailureDiagnostics.QualityClaim || result.FailureDiagnostics.Summary.Steps != 0 || result.FailureDiagnostics.Summary.OptimizerDiagnostics.AcceptedSteps != 0 {
		t.Fatalf("failure diagnostics not fail-closed: %+v", result.FailureDiagnostics)
	}
	diagnosticsPath := AOQTFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath)
	data, err := os.ReadFile(diagnosticsPath)
	if err != nil {
		t.Fatalf("read fail-closed diagnostics: %v", err)
	}
	var diagnostics AOQTSidecarFailClosedDiagnostics
	if err := strictUnmarshalAOQT(data, &diagnostics); err != nil {
		t.Fatalf("decode fail-closed diagnostics: %v\n%s", err, data)
	}
	if err := diagnostics.Validate(); err != nil {
		t.Fatalf("written fail-closed diagnostics invalid: %v", err)
	}
	if diagnostics.Schema != AOQTSidecarFailClosedDiagnosticsSchema || diagnostics.FailureKind != "zero_accepted_safe_steps" {
		t.Fatalf("unexpected fail-closed diagnostics identity: %+v", diagnostics)
	}
	if diagnostics.Summary.OptimizerDiagnosticsSHA256 != result.FailureDiagnostics.Summary.OptimizerDiagnosticsSHA256 {
		t.Fatalf("written diagnostics hash mismatch: %s vs %s", diagnostics.Summary.OptimizerDiagnosticsSHA256, result.FailureDiagnostics.Summary.OptimizerDiagnosticsSHA256)
	}
	if diagnostics.IOReport != result.IOReport {
		t.Fatalf("written diagnostics did not preserve the runner IO report")
	}
	if diagnostics.PreflightPath != trainCfg.PreflightJSONPath || diagnostics.PreflightSHA256 != trainCfg.ExpectedPreflightSHA256 {
		t.Fatalf("written diagnostics preflight binding = (%q,%q), want (%q,%q)", diagnostics.PreflightPath, diagnostics.PreflightSHA256, trainCfg.PreflightJSONPath, trainCfg.ExpectedPreflightSHA256)
	}
	if diagnostics.Preflight.RowCount != result.Preflight.RowCount || diagnostics.Preflight.CandidateCount != result.Preflight.CandidateCount || diagnostics.Preflight.PairCount != result.Preflight.PairCount || diagnostics.Preflight.PlanSHA256 != result.Preflight.PlanSHA256 || diagnostics.Preflight.CalibrationManifestSHA256 != result.Preflight.CalibrationManifestSHA256 {
		t.Fatalf("written diagnostics did not preserve preflight workload/provenance: %+v vs %+v", diagnostics.Preflight, result.Preflight)
	}
	if optimizer := diagnostics.Summary.OptimizerDiagnostics; optimizer == nil || optimizer.CoordinateSearchPlanCount == 0 || optimizer.CoordinateSearchAudit == "" {
		t.Fatalf("zero-safe fixture did not produce genuine protected coordinate audit: %+v", optimizer)
	}
	t.Run("v2-q3-only-contract-is-rejected", func(t *testing.T) {
		tampered := diagnostics
		q3Only := tampered.ObjectiveContract
		q3Only.WeightSums = AOQTSidecarRowWeights{Q3Gain: q3Only.WeightSums.Q3Gain}
		tampered.ObjectiveContract = q3Only
		tampered.Summary.ObjectiveContract = q3Only
		optimizer := *tampered.Summary.OptimizerDiagnostics
		sha, err := optimizer.SHA256()
		if err != nil {
			t.Fatalf("recompute q3-only optimizer hash: %v", err)
		}
		tampered.Summary.OptimizerDiagnosticsSHA256 = sha
		if err := tampered.Validate(); err == nil || !strings.Contains(err.Error(), "active protected objective component") {
			t.Fatalf("q3-only fail-closed diagnostics error = %v, want active-protected rejection", err)
		}
	})
	if !aoqtStringMapsEqual(diagnostics.Preflight.InputSHA256, result.Preflight.InputSHA256) || !aoqtStringMapsEqual(diagnostics.Preflight.QrelsSHA256ByDataset, result.Preflight.QrelsSHA256ByDataset) {
		t.Fatalf("written diagnostics did not preserve preflight input/qrels hashes")
	}
	if err := writeAOQTFailClosedDiagnosticsFile(diagnosticsPath, diagnostics); err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("exclusive fail-closed diagnostics rewrite error = %v, want existing-output rejection", err)
	}
	afterRewriteAttempt, err := os.ReadFile(diagnosticsPath)
	if err != nil {
		t.Fatalf("read fail-closed diagnostics after exclusive rewrite attempt: %v", err)
	}
	if !bytes.Equal(afterRewriteAttempt, data) {
		t.Fatalf("exclusive rewrite attempt changed prior fail-closed diagnostics")
	}
	for name, mutate := range map[string]func(*AOQTSidecarFailClosedDiagnostics){
		"io workload count": func(tampered *AOQTSidecarFailClosedDiagnostics) {
			tampered.IOReport.RowCount++
		},
		"io rows hash binding": func(tampered *AOQTSidecarFailClosedDiagnostics) {
			tampered.IOReport.RowsSHA256 = strings.Repeat("b", 64)
		},
		"preflight workload count": func(tampered *AOQTSidecarFailClosedDiagnostics) {
			tampered.Preflight.PairCount++
		},
		"final objective component": func(tampered *AOQTSidecarFailClosedDiagnostics) {
			tampered.Summary.FinalObjectiveComponents.Q3Gain++
		},
		"objective contract": func(tampered *AOQTSidecarFailClosedDiagnostics) {
			tampered.ObjectiveContract.GainBit++
		},
		"preflight qrels provenance": func(tampered *AOQTSidecarFailClosedDiagnostics) {
			tampered.Preflight.QrelsSHA256ByDataset["fiqa"] = strings.Repeat("a", 64)
		},
		"invalid preflight plan hash": func(tampered *AOQTSidecarFailClosedDiagnostics) {
			tampered.Preflight.PlanSHA256 = "not-a-sha256"
		},
		"preflight path binding": func(tampered *AOQTSidecarFailClosedDiagnostics) {
			tampered.PreflightPath += ".tampered"
		},
		"preflight raw hash binding": func(tampered *AOQTSidecarFailClosedDiagnostics) {
			tampered.PreflightSHA256 = strings.Repeat("a", 64)
		},
		"negative activation count": func(tampered *AOQTSidecarFailClosedDiagnostics) {
			tampered.Summary.InitialObjectiveActivation.Q3GainEligiblePairs = -1
		},
	} {
		t.Run(name, func(t *testing.T) {
			tampered := diagnostics
			tampered.Inputs.QrelsSHA256ByDataset = cloneAOQTStringMap(diagnostics.Inputs.QrelsSHA256ByDataset)
			tampered.Preflight.QrelsSHA256ByDataset = cloneAOQTStringMap(diagnostics.Preflight.QrelsSHA256ByDataset)
			mutate(&tampered)
			if err := tampered.Validate(); err == nil {
				t.Fatalf("tampered fail-closed diagnostics unexpectedly validated: %+v", tampered)
			}
		})
	}
	if _, err := NewAOQTSidecarFailClosedDiagnostics(AOQTSidecarCalibrationSet{}, AOQTSidecarCalibrationIOReport{}, result.FailureDiagnostics.Summary, fmt.Errorf("legacy constructor must not synthesize provenance")); err == nil || !strings.Contains(err.Error(), "authentic materializer preflight") {
		t.Fatalf("legacy fail-closed diagnostics constructor error = %v, want authentic-preflight rejection", err)
	}
	if _, err := os.Lstat(trainCfg.MetricsJSONPath); err == nil {
		t.Fatalf("runner wrote metrics output %q despite zero accepted safe steps", trainCfg.MetricsJSONPath)
	} else if !os.IsNotExist(err) {
		t.Fatalf("stat metrics output: %v", err)
	}
	for role, candidate := range aoqtCandidatePathMap(aoqtCandidateOutputPaths(trainCfg.OutputArtifactPath, true)) {
		if _, err := os.Lstat(candidate); err == nil {
			t.Fatalf("runner wrote %s output %q despite zero accepted safe steps", role, candidate)
		} else if !os.IsNotExist(err) {
			t.Fatalf("stat %s output: %v", role, err)
		}
	}
	mutateOptimizer := func(tampered *AOQTSidecarFailClosedDiagnostics, mutate func(*AOQTSidecarOptimizerDiagnostics)) {
		optimizer := *tampered.Summary.OptimizerDiagnostics
		mutate(&optimizer)
		tampered.Summary.OptimizerDiagnostics = &optimizer
		sha, err := optimizer.SHA256()
		if err != nil {
			t.Fatalf("tampered optimizer diagnostics sha: %v", err)
		}
		tampered.Summary.OptimizerDiagnosticsSHA256 = sha
	}
	for name, mutate := range map[string]func(*AOQTSidecarOptimizerDiagnostics){
		"optimizer-path-strip": func(optimizer *AOQTSidecarOptimizerDiagnostics) {
			optimizer.AdamProposalAttempts = 0
			optimizer.AdamAcceptedProposals = 0
			optimizer.AdamRejectedProposals = 0
			optimizer.CoordinateProposalAttempts = 0
			optimizer.CoordinateAcceptedProposals = 0
			optimizer.CoordinateRejectedProposals = 0
			optimizer.CoordinateSearchPlanCount = 0
			optimizer.CoordinateTopAngles = 0
			optimizer.CoordinateMagnitudeCount = 0
			optimizer.CoordinateBlockCount = 0
			optimizer.CoordinateSearchStrategy = ""
			optimizer.CoordinateSearchOrderingHash = ""
			optimizer.CoordinateSearchLearningRate = 0
			optimizer.CoordinateSearchAudit = ""
			optimizer.CoordinateSearchAuditChain = ""
			optimizer.CoordinateSearchHashChain = ""
		},
		"coordinate-audit-chain": func(optimizer *AOQTSidecarOptimizerDiagnostics) {
			optimizer.CoordinateSearchAuditChain = "[]"
		},
		"coordinate-plan-count": func(optimizer *AOQTSidecarOptimizerDiagnostics) {
			optimizer.CoordinateSearchPlanCount = 0
			optimizer.CoordinateSearchAudit = ""
			optimizer.CoordinateSearchAuditChain = ""
			optimizer.CoordinateSearchHashChain = ""
		},
		"coordinate-magnitude-count": func(optimizer *AOQTSidecarOptimizerDiagnostics) {
			optimizer.CoordinateMagnitudeCount = 1
		},
		"coordinate-schedule": func(optimizer *AOQTSidecarOptimizerDiagnostics) {
			var payload aoqtCoordinateSearchAuditPayload
			if err := strictUnmarshalAOQT([]byte(optimizer.CoordinateSearchAudit), &payload); err != nil {
				t.Fatalf("decode protected coordinate audit: %v", err)
			}
			payload.MicroTailMagnitudes[0] *= 2
			audit, err := json.Marshal(payload)
			if err != nil {
				t.Fatalf("marshal protected coordinate audit: %v", err)
			}
			optimizer.CoordinateSearchAudit = string(audit)
			optimizer.CoordinateSearchAuditChain, err = appendAOQTCoordinateSearchAuditChain("", string(audit))
			if err != nil {
				t.Fatalf("append protected coordinate audit chain: %v", err)
			}
			optimizer.CoordinateSearchHashChain, err = appendAOQTCoordinateSearchHashChain("", string(audit))
			if err != nil {
				t.Fatalf("append protected coordinate hash chain: %v", err)
			}
			var hashes []string
			if err := strictUnmarshalAOQT([]byte(optimizer.CoordinateSearchHashChain), &hashes); err != nil || len(hashes) == 0 {
				t.Fatalf("decode protected coordinate hash chain: %v", err)
			}
			optimizer.CoordinateSearchOrderingHash = hashes[len(hashes)-1]
		},
		"negative-adam-counter": func(optimizer *AOQTSidecarOptimizerDiagnostics) {
			optimizer.AdamProposalAttempts = -1
		},
		"rejection-reason-count": func(optimizer *AOQTSidecarOptimizerDiagnostics) {
			optimizer.RejectionDiagnostics.ReasonCounts.LossIncrease++
		},
		"rejection-component-count": func(optimizer *AOQTSidecarOptimizerDiagnostics) {
			optimizer.RejectionDiagnostics.ComponentRegressionCounts.Q3ScoreDistill = 1
		},
		"rejection-delta-stats": func(optimizer *AOQTSidecarOptimizerDiagnostics) {
			optimizer.RejectionDiagnostics.LossDelta.Sum = 2
		},
	} {
		t.Run("protected-"+name, func(t *testing.T) {
			tampered := diagnostics
			mutateOptimizer(&tampered, mutate)
			if err := tampered.Validate(); err == nil {
				t.Fatalf("tampered protected fail-closed diagnostics unexpectedly validated")
			}
		})
	}
}

func TestAOQTSidecarTrainRunnerRejectsDiagnosticsCandidatePathCollision(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	trainCfg.PlanOnly = false
	trainCfg.MaxSteps = 1
	trainCfg.OutputArtifactPath = AOQTFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath)
	called := false
	_, err := runAOQTSidecarTraining(trainCfg, func(AOQTSidecarObjectiveContract) (AOQTSidecarVectorObjective, error) {
		called = true
		return nil, fmt.Errorf("objective factory must not run when outputs collide")
	})
	if err == nil || !strings.Contains(err.Error(), "output path collision") {
		t.Fatalf("diagnostics/candidate path collision error = %v, want collision rejection", err)
	}
	if called {
		t.Fatalf("objective factory ran despite diagnostics/candidate path collision")
	}
	if _, err := os.Lstat(trainCfg.MetricsJSONPath); !os.IsNotExist(err) {
		t.Fatalf("metrics output state = %v, want absent", err)
	}
	if _, err := os.Lstat(AOQTFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath)); !os.IsNotExist(err) {
		t.Fatalf("fail-closed diagnostics state = %v, want absent", err)
	}
}

func TestAOQTSidecarTrainRunnerRejectsCandidateSiblingPathCollisions(t *testing.T) {
	output := filepath.Join(t.TempDir(), "candidate.mll")
	for role, path := range aoqtCandidatePathMap(aoqtCandidateOutputPaths(output, true)) {
		t.Run(role, func(t *testing.T) {
			cfg := AOQTSidecarTrainRunnerConfig{
				MetricsJSONPath:    path,
				OutputArtifactPath: output,
			}
			if err := validateAOQTTrainRunnerOutputPathDistinctness(cfg); err == nil || !strings.Contains(err.Error(), "output path collision") {
				t.Fatalf("candidate sibling collision error = %v, want collision rejection", err)
			}
		})
	}
	inputDir := t.TempDir()
	inputPaths := map[string]string{
		"manifest-input":  filepath.Join(inputDir, "manifest.json"),
		"rows-input":      filepath.Join(inputDir, "rows.jsonl"),
		"preflight-input": filepath.Join(inputDir, "preflight.json"),
	}
	for role, inputPath := range inputPaths {
		t.Run(role, func(t *testing.T) {
			cfg := AOQTSidecarTrainRunnerConfig{
				ManifestPath:       inputPaths["manifest-input"],
				RowsJSONLPath:      inputPaths["rows-input"],
				PreflightJSONPath:  inputPaths["preflight-input"],
				MetricsJSONPath:    filepath.Join(inputDir, "metrics.json"),
				OutputArtifactPath: inputPath,
			}
			if err := validateAOQTTrainRunnerOutputPathDistinctness(cfg); err == nil || !strings.Contains(err.Error(), "output path collision") {
				t.Fatalf("output/input collision error = %v, want collision rejection", err)
			}
		})
	}
	resolvedOutput := filepath.Join(inputDir, "manifest.json")
	aliasOutput := filepath.Join(inputDir, "candidate-alias.mll")
	if err := os.WriteFile(resolvedOutput, []byte("placeholder\n"), 0o644); err != nil {
		t.Fatalf("write output/input symlink target: %v", err)
	}
	if err := os.Symlink(resolvedOutput, aliasOutput); err != nil {
		t.Fatalf("create output/input symlink alias: %v", err)
	}
	if err := validateAOQTTrainRunnerOutputPathDistinctness(AOQTSidecarTrainRunnerConfig{
		ManifestPath:       resolvedOutput,
		RowsJSONLPath:      inputPaths["rows-input"],
		PreflightJSONPath:  inputPaths["preflight-input"],
		MetricsJSONPath:    filepath.Join(inputDir, "metrics-alias.json"),
		OutputArtifactPath: aliasOutput,
	}); err == nil || !strings.Contains(err.Error(), "output path collision") {
		t.Fatalf("symlink output/input collision error = %v, want canonical collision rejection", err)
	}
}

func TestAOQTSidecarTrainRunnerRejectsPreexistingFailClosedDiagnostics(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	trainCfg.PlanOnly = false
	trainCfg.MaxSteps = 1
	trainCfg.OutputArtifactPath = filepath.Join(t.TempDir(), "candidate.mll")
	diagnosticsPath := AOQTFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath)
	sentinel := []byte("previous fail-closed evidence\n")
	if err := os.WriteFile(diagnosticsPath, sentinel, 0o644); err != nil {
		t.Fatalf("write existing fail-closed diagnostics: %v", err)
	}
	called := false
	_, err := runAOQTSidecarTraining(trainCfg, func(AOQTSidecarObjectiveContract) (AOQTSidecarVectorObjective, error) {
		called = true
		return nil, fmt.Errorf("objective factory must not run when fail-closed diagnostics output exists")
	})
	if err == nil || !strings.Contains(err.Error(), "fail-closed diagnostics output") || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("preexisting fail-closed diagnostics error = %v, want protected-output rejection", err)
	}
	if called {
		t.Fatalf("objective factory ran despite protected fail-closed diagnostics output")
	}
	got, err := os.ReadFile(diagnosticsPath)
	if err != nil {
		t.Fatalf("read existing fail-closed diagnostics: %v", err)
	}
	if !bytes.Equal(got, sentinel) {
		t.Fatalf("preexisting fail-closed diagnostics changed: %q", got)
	}
	if _, err := os.Lstat(trainCfg.MetricsJSONPath); !os.IsNotExist(err) {
		t.Fatalf("metrics output state = %v, want absent", err)
	}
}

func TestAOQTSidecarTrainRunnerRejectsStaleFailClosedDiagnosticsBeforeSuccessOutput(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	diagnosticsPath := AOQTFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath)
	sentinel := []byte("stale fail-closed evidence\n")
	if err := os.WriteFile(diagnosticsPath, sentinel, 0o644); err != nil {
		t.Fatalf("write stale fail-closed diagnostics: %v", err)
	}
	result, err := RunAOQTSidecarTraining(trainCfg)
	if err == nil || !strings.Contains(err.Error(), "fail-closed diagnostics output") || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("stale fail-closed diagnostics error = %v, want success-output protection", err)
	}
	if result.Metrics.Schema != "" || result.PackageResult != nil {
		t.Fatalf("runner returned successful output alongside stale diagnostics: %+v", result)
	}
	if _, err := os.Lstat(trainCfg.MetricsJSONPath); !os.IsNotExist(err) {
		t.Fatalf("metrics output state = %v, want absent", err)
	}
	got, err := os.ReadFile(diagnosticsPath)
	if err != nil {
		t.Fatalf("read stale fail-closed diagnostics: %v", err)
	}
	if !bytes.Equal(got, sentinel) {
		t.Fatalf("stale fail-closed diagnostics changed: %q", got)
	}
}

func TestAOQTSidecarTrainRunnerRejectsPreexistingMetricsBeforeFailure(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	trainCfg.PlanOnly = false
	trainCfg.MaxSteps = 1
	trainCfg.OutputArtifactPath = filepath.Join(t.TempDir(), "candidate.mll")
	sentinel := []byte("prior successful metrics\n")
	if err := os.WriteFile(trainCfg.MetricsJSONPath, sentinel, 0o644); err != nil {
		t.Fatalf("write metrics sentinel: %v", err)
	}
	called := false
	_, err := runAOQTSidecarTraining(trainCfg, func(AOQTSidecarObjectiveContract) (AOQTSidecarVectorObjective, error) {
		called = true
		return nil, fmt.Errorf("objective factory must not run with preexisting metrics")
	})
	if err == nil || !strings.Contains(err.Error(), "metrics output") || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("preexisting metrics error = %v, want stale-success rejection", err)
	}
	if called {
		t.Fatalf("objective factory ran despite preexisting metrics sentinel")
	}
	got, err := os.ReadFile(trainCfg.MetricsJSONPath)
	if err != nil {
		t.Fatalf("read metrics sentinel: %v", err)
	}
	if !bytes.Equal(got, sentinel) {
		t.Fatalf("metrics sentinel changed after rejected run: %q", got)
	}
	if _, err := os.Lstat(AOQTFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath)); !os.IsNotExist(err) {
		t.Fatalf("fail-closed diagnostics state = %v, want absent", err)
	}
}

func TestAOQTSidecarTrainRunnerRollsBackMetricsAfterPackageFailure(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	anchor := writeTinyAOQTAnchorPackage(t)
	anchorSHA, _, err := fileHash(anchor)
	if err != nil {
		t.Fatalf("hash tiny anchor: %v", err)
	}
	packageSHA, _, err := fileHash(DefaultPackageManifestPath(anchor))
	if err != nil {
		t.Fatalf("hash tiny anchor package manifest: %v", err)
	}
	materializerCfg.AnchorArtifactPath = anchor
	materializerCfg.AnchorArtifactSHA256 = anchorSHA
	materializerCfg.PackageManifestSHA256 = packageSHA
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	trainCfg.PlanOnly = false
	trainCfg.MaxSteps = 1
	trainCfg.OutputArtifactPath = filepath.Join(t.TempDir(), "candidate.mll")

	oldWriter := writeAOQTCandidatePackageManifestExclusive
	writeAOQTCandidatePackageManifestExclusive = func(manifest PackageManifest, path string, cleanup *aoqtCandidateCleanup) error {
		data, err := encodePackageManifestMLL(manifest)
		if err != nil {
			return err
		}
		if err := writeAOQTExclusiveFile(path, data, 0o644, cleanup); err != nil {
			return err
		}
		return fmt.Errorf("injected runner package publication failure")
	}
	defer func() { writeAOQTCandidatePackageManifestExclusive = oldWriter }()
	if _, err := runAOQTSidecarTraining(trainCfg, aoqtRunnerObjectiveFromContract); err == nil || !strings.Contains(err.Error(), "injected runner package publication failure") {
		t.Fatalf("injected package publication error = %v, want transactional rollback", err)
	}
	writeAOQTCandidatePackageManifestExclusive = oldWriter
	for role, path := range aoqtCandidatePathMap(aoqtCandidateOutputPaths(trainCfg.OutputArtifactPath, true)) {
		if path == "" {
			continue
		}
		if _, err := os.Lstat(path); !os.IsNotExist(err) {
			t.Fatalf("candidate %s residue after package failure at %s: %v", role, path, err)
		}
	}
	if _, err := os.Lstat(trainCfg.MetricsJSONPath); !os.IsNotExist(err) {
		t.Fatalf("metrics residue after package failure = %v, want absent", err)
	}
	if _, err := os.Lstat(AOQTFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath)); !os.IsNotExist(err) {
		t.Fatalf("fail-closed residue after package failure = %v, want absent", err)
	}

	result, err := runAOQTSidecarTraining(trainCfg, aoqtRunnerObjectiveFromContract)
	if err != nil {
		t.Fatalf("fresh run after package failure: %v", err)
	}
	if result.PackageResult == nil {
		t.Fatalf("fresh run after package failure returned no package result")
	}
	if _, err := os.Lstat(trainCfg.MetricsJSONPath); err != nil {
		t.Fatalf("fresh run metrics output: %v", err)
	}
	for role, path := range aoqtCandidatePathMap(result.PackageResult.Paths) {
		if path == "" {
			continue
		}
		if _, err := os.Lstat(path); err != nil {
			t.Fatalf("fresh run candidate %s output: %v", role, err)
		}
	}
}

func aoqtTrainRunnerConfigFromMaterializerFixture(t *testing.T, materializerCfg AOQTSidecarMaterializeConfig) AOQTSidecarTrainRunnerConfig {
	t.Helper()
	manifest := readAOQTManifestForTrainRunnerTest(t, materializerCfg.ManifestJSONPath)
	manifestSHA := mustAOQTManifestSHA256Test(t, manifest)
	return AOQTSidecarTrainRunnerConfig{
		ManifestPath:                        materializerCfg.ManifestJSONPath,
		RowsJSONLPath:                       materializerCfg.RowJSONLPath,
		PreflightJSONPath:                   materializerCfg.PreflightJSONPath,
		MetricsJSONPath:                     filepath.Join(t.TempDir(), "metrics.json"),
		ExpectedManifestSHA256:              manifestSHA,
		ExpectedRowsSHA256:                  mustSHA256FileAOQTTest(t, materializerCfg.RowJSONLPath),
		ExpectedPreflightSHA256:             mustSHA256FileAOQTTest(t, materializerCfg.PreflightJSONPath),
		ExpectedAnchorArtifactSHA256:        manifest.AnchorArtifactSHA256,
		ExpectedAnchorPackageManifestSHA256: manifest.AnchorPackageManifestSHA256,
		ExpectedAnchorEmbeddingSpaceID:      manifest.AnchorEmbeddingSpaceID,
		ExpectedCompatibilityDigest:         manifest.CompatibilityDigest,
		ExpectedSourceArtifactHashes:        append([]string(nil), manifest.SourceArtifactHashes...),
		ExpectedVectorCacheHashes:           append([]string(nil), manifest.VectorCacheHashes...),
		PlanOnly:                            true,
		AllowResearchOnly:                   true,
	}
}

func readAOQTManifestForTrainRunnerTest(t *testing.T, path string) AOQTSidecarCalibrationManifest {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var manifest AOQTSidecarCalibrationManifest
	if err := strictUnmarshalAOQT(data, &manifest); err != nil {
		t.Fatal(err)
	}
	return manifest
}
