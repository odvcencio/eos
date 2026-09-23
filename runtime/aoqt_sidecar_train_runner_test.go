package eosruntime

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
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
	if result.Metrics.Plan.OptimizerMode != "" || result.Metrics.Summary.OptimizerMode != "" || result.DevEvidence != nil || result.PackageResult != nil {
		t.Fatalf("default plan-only runner was not production-compatible: metrics=%+v dev=%+v package=%+v", result.Metrics, result.DevEvidence, result.PackageResult)
	}
	if _, err := os.Lstat(AOQTFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath)); !os.IsNotExist(err) {
		t.Fatalf("successful plan-only run fail-closed diagnostics state = %v, want absent", err)
	}
}

func TestAOQTSidecarTrainRunnerRejectsUnauthorizedV7DevMode(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	trainCfg.PlanOnly = false
	trainCfg.MaxSteps = 1
	trainCfg.OptimizerMode = AOQTSidecarOptimizerModeV7DevTrustRegion
	trainCfg.DevOutputDir = t.TempDir()
	called := false
	_, err := runAOQTSidecarTraining(trainCfg, func(AOQTSidecarObjectiveContract) (AOQTSidecarVectorObjective, error) {
		called = true
		return nil, fmt.Errorf("objective factory must not run without dev authorization")
	})
	if err == nil || !strings.Contains(err.Error(), "requires explicit dev-only authorization") {
		t.Fatalf("unauthorized dev optimizer error = %v, want dev authorization rejection", err)
	}
	if called {
		t.Fatalf("objective factory ran despite missing dev authorization")
	}
	if _, err := os.Lstat(trainCfg.MetricsJSONPath); !os.IsNotExist(err) {
		t.Fatalf("metrics output state = %v, want absent", err)
	}
}

func TestAOQTSidecarTrainRunnerWritesV7DevEvidenceWithoutPackage(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	splitManifestPath := writeV7DevSplitManifestForTrainRunnerTest(t, trainCfg, "fold-0", nil)
	trainCfg.PlanOnly = false
	trainCfg.MaxSteps = 1
	trainCfg.OptimizerMode = AOQTSidecarOptimizerModeV7DevTrustRegion
	trainCfg.AllowDevOnly = true
	trainCfg.DevOutputDir = t.TempDir()
	trainCfg.SplitManifestPath = splitManifestPath
	trainCfg.ExpectedSplitManifestSHA256 = mustSHA256FileAOQTTest(t, splitManifestPath)
	trainCfg.FoldID = "fold-0"

	result, err := runAOQTSidecarTraining(trainCfg, aoqtRunnerObjectiveFromContract)
	if err != nil {
		t.Fatalf("run V7 dev AOQT sidecar training: %v", err)
	}
	if result.PackageResult != nil {
		t.Fatalf("dev-only runner returned candidate package result: %+v", result.PackageResult)
	}
	if result.DevEvidence == nil {
		t.Fatalf("dev-only runner did not return dev evidence")
	}
	if result.Metrics.Plan.OptimizerMode != AOQTSidecarOptimizerModeV7DevTrustRegion || result.Metrics.Summary.OptimizerMode != AOQTSidecarOptimizerModeV7DevTrustRegion {
		t.Fatalf("dev optimizer mode not recorded in metrics: plan=%q summary=%q", result.Metrics.Plan.OptimizerMode, result.Metrics.Summary.OptimizerMode)
	}
	if result.Metrics.Summary.OptimizerDiagnostics == nil || !result.Metrics.Summary.OptimizerDiagnostics.DevOnly {
		t.Fatalf("dev-only optimizer diagnostics not recorded: %+v", result.Metrics.Summary.OptimizerDiagnostics)
	}
	if err := result.DevEvidence.Validate(); err != nil {
		t.Fatalf("dev evidence invalid: %v", err)
	}
	if result.DevEvidence.SplitBinding == nil || result.DevEvidence.SplitBinding.SplitManifestSHA256 != trainCfg.ExpectedSplitManifestSHA256 || result.DevEvidence.SplitBinding.FoldID != trainCfg.FoldID {
		t.Fatalf("dev split binding not recorded: %+v", result.DevEvidence.SplitBinding)
	}
	if result.Metrics.SplitBinding == nil || !reflect.DeepEqual(*result.Metrics.SplitBinding, *result.DevEvidence.SplitBinding) {
		t.Fatalf("metrics/dev evidence split binding mismatch: metrics=%+v dev=%+v", result.Metrics.SplitBinding, result.DevEvidence.SplitBinding)
	}
	if result.Metrics.SplitBinding.TrainRowCount != result.Metrics.Plan.RowCount || result.Metrics.SplitBinding.MaterializedRowCount != result.Metrics.Plan.RowCount {
		t.Fatalf("split binding row counts = train %d materialized %d plan %d", result.Metrics.SplitBinding.TrainRowCount, result.Metrics.SplitBinding.MaterializedRowCount, result.Metrics.Plan.RowCount)
	}
	if result.Metrics.SplitBinding.MaterializedRowsSHA256 != trainCfg.ExpectedRowsSHA256 || result.Metrics.SplitBinding.MaterializedPreflightSHA256 != trainCfg.ExpectedPreflightSHA256 {
		t.Fatalf("split binding materialized hashes not recorded: %+v", result.Metrics.SplitBinding)
	}
	for _, path := range []string{trainCfg.MetricsJSONPath, result.DevEvidence.TransformPath, filepath.Join(trainCfg.DevOutputDir, "aoqt-sidecar-dev-evidence.json")} {
		if _, err := os.Lstat(path); err != nil {
			t.Fatalf("expected dev artifact %q: %v", path, err)
		}
	}
	var transform AOQTGivensTransform
	transformData, err := os.ReadFile(result.DevEvidence.TransformPath)
	if err != nil {
		t.Fatalf("read dev transform: %v", err)
	}
	if err := strictUnmarshalAOQT(transformData, &transform); err != nil {
		t.Fatalf("decode dev transform: %v", err)
	}
	if got, err := transform.AnglesSHA256(); err != nil || got != result.Metrics.Summary.AnglesSHA256 {
		t.Fatalf("dev transform angles sha = %q, %v; want metrics %q", got, err, result.Metrics.Summary.AnglesSHA256)
	}
	for role, candidate := range aoqtCandidatePathMap(aoqtCandidateOutputPaths(filepath.Join(trainCfg.DevOutputDir, "candidate.mll"), true)) {
		if _, err := os.Lstat(candidate); err == nil {
			t.Fatalf("dev-only runner wrote candidate %s output %q", role, candidate)
		} else if !os.IsNotExist(err) {
			t.Fatalf("stat candidate %s output: %v", role, err)
		}
	}
}

func TestAOQTSidecarTrainRunnerWritesDevFailureDiagnosticsWithoutPackage(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	splitManifestPath := writeV7DevSplitManifestForTrainRunnerTest(t, trainCfg, "fold-0", nil)
	trainCfg.PlanOnly = false
	trainCfg.MaxSteps = 1
	trainCfg.OptimizerMode = AOQTSidecarOptimizerModeV7DevTrustRegion
	trainCfg.AllowDevOnly = true
	trainCfg.DevOutputDir = t.TempDir()
	trainCfg.SplitManifestPath = splitManifestPath
	trainCfg.ExpectedSplitManifestSHA256 = mustSHA256FileAOQTTest(t, splitManifestPath)
	trainCfg.FoldID = "fold-0"
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
	if err == nil {
		t.Fatal("dev failure fixture unexpectedly succeeded")
	}
	if result.DevFailureDiagnostics == nil {
		t.Fatalf("dev failure diagnostics missing: %v", err)
	}
	diagnostics := result.DevFailureDiagnostics
	if diagnostics.QualityClaim || diagnostics.ReleaseClaim || diagnostics.OfficialClaim || diagnostics.OfficialHeldoutGate || diagnostics.CommercialClaim {
		t.Fatalf("dev failure claims are not fail-closed: %+v", diagnostics)
	}
	if err := diagnostics.Validate(); err != nil {
		t.Fatalf("dev failure diagnostics invalid: %v", err)
	}
	optimizer := diagnostics.Summary.OptimizerDiagnostics
	if optimizer == nil || optimizer.ProposalReceipts == nil || len(*optimizer.ProposalReceipts) != optimizer.ProposalAttempts {
		t.Fatalf("dev failure proposal receipts = %+v, want every attempt", optimizer)
	}
	for i, receipt := range *optimizer.ProposalReceipts {
		if !receipt.TotalDeltaFinite || !receipt.ComponentDeltasFinite {
			t.Fatalf("receipt[%d] omitted finite deltas: %+v", i, receipt)
		}
	}
	path := AOQTDevFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read dev failure diagnostics: %v", err)
	}
	var decoded AOQTSidecarDevFailClosedDiagnostics
	if err := strictUnmarshalAOQT(data, &decoded); err != nil {
		t.Fatalf("decode dev failure diagnostics: %v", err)
	}
	if err := decoded.Validate(); err != nil {
		t.Fatalf("decoded dev failure diagnostics invalid: %v", err)
	}
	if decoded.IOReport != result.IOReport || !reflect.DeepEqual(decoded.Preflight, result.Preflight) || decoded.SplitBinding == nil {
		t.Fatalf("dev failure diagnostics lost exact input/split binding")
	}
	if _, err := os.Lstat(trainCfg.MetricsJSONPath); !os.IsNotExist(err) {
		t.Fatalf("metrics output state = %v, want absent on failure", err)
	}
	if _, err := os.Lstat(trainCfg.OutputArtifactPath); !os.IsNotExist(err) {
		t.Fatalf("candidate package output state = %v, want absent", err)
	}
}

func TestAOQTSidecarTrainRunnerV7R2ProbePreconditionFailsClosed(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	splitManifestPath := writeV7DevSplitManifestForTrainRunnerTest(t, trainCfg, "fold-0", nil)
	trainCfg.PlanOnly = false
	trainCfg.MaxSteps = 1
	trainCfg.LearningRate = 0.01
	trainCfg.OptimizerMode = AOQTSidecarOptimizerModeV7R2DevTrustRegion
	trainCfg.RequireForwardConsistencyProbe = true
	trainCfg.AllowDevOnly = true
	trainCfg.DevOutputDir = t.TempDir()
	trainCfg.SplitManifestPath = splitManifestPath
	trainCfg.ExpectedSplitManifestSHA256 = mustSHA256FileAOQTTest(t, splitManifestPath)
	trainCfg.FoldID = "fold-0"
	result, err := runAOQTSidecarTraining(trainCfg, aoqtRunnerObjectiveFromContract)
	if err == nil || result.DevFailureDiagnostics == nil {
		t.Fatalf("V7-r2 precondition result=%+v err=%v, want fail-closed diagnostics", result, err)
	}
	if result.DevFailureDiagnostics.ForwardConsistencyProbe == nil || result.DevFailureDiagnostics.ForwardConsistencyProbe.Passed {
		t.Fatalf("V7-r2 precondition probe receipt = %+v, want failed receipt", result.DevFailureDiagnostics.ForwardConsistencyProbe)
	}
	if strings.TrimSpace(result.DevFailureDiagnostics.ForwardConsistencyProbe.FailureReason) == "" {
		t.Fatalf("V7-r2 probe failure reason is empty")
	}
	if err := result.DevFailureDiagnostics.Validate(); err != nil {
		t.Fatalf("V7-r2 precondition diagnostics invalid: %v", err)
	}
	if _, err := os.Stat(AOQTDevFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath)); err != nil {
		t.Fatalf("V7-r2 dev fail-closed diagnostics missing: %v", err)
	}
}

func TestAOQTSidecarTrainRunnerV7R2PassedProbeOptimizerFailureIsDevTrainingFailed(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	splitManifestPath := writeV7DevSplitManifestForTrainRunnerTest(t, trainCfg, "fold-0", nil)
	trainCfg.PlanOnly = false
	trainCfg.MaxSteps = 1
	trainCfg.LearningRate = 0.01
	trainCfg.OptimizerMode = AOQTSidecarOptimizerModeV7R2DevTrustRegion
	trainCfg.RequireForwardConsistencyProbe = true
	trainCfg.AllowDevOnly = true
	trainCfg.DevOutputDir = t.TempDir()
	trainCfg.SplitManifestPath = splitManifestPath
	trainCfg.ExpectedSplitManifestSHA256 = mustSHA256FileAOQTTest(t, splitManifestPath)
	trainCfg.FoldID = "fold-0"

	manifest := readAOQTManifestForTrainRunnerTest(t, trainCfg.ManifestPath)
	topology, err := expectedAOQTSidecarTrainTopology()
	if err != nil {
		t.Fatalf("expected AOQT topology: %v", err)
	}
	set, _, err := LoadAOQTSidecarCalibrationSet(AOQTSidecarCalibrationIOConfig{
		ManifestPath:                        trainCfg.ManifestPath,
		RowsJSONLPath:                       trainCfg.RowsJSONLPath,
		ExpectedManifestSHA256:              trainCfg.ExpectedManifestSHA256,
		ExpectedRowsSHA256:                  trainCfg.ExpectedRowsSHA256,
		ExpectedAnchorArtifactSHA256:        trainCfg.ExpectedAnchorArtifactSHA256,
		ExpectedAnchorPackageManifestSHA256: trainCfg.ExpectedAnchorPackageManifestSHA256,
		ExpectedAnchorEmbeddingSpaceID:      trainCfg.ExpectedAnchorEmbeddingSpaceID,
		ExpectedCompatibilityDigest:         trainCfg.ExpectedCompatibilityDigest,
		ExpectedTurboQuantSeed:              AOQTSidecarMaterializerQuantSeed,
		ExpectedTopology:                    topology,
		ExpectedSourceArtifactHashes:        manifest.SourceArtifactHashes,
		ExpectedVectorCacheHashes:           manifest.VectorCacheHashes,
	})
	if err != nil {
		t.Fatalf("load AOQT fixture: %v", err)
	}

	// Count the exact objective calls consumed by the probe. The runner uses
	// the same deterministic trainer inputs, so the next objective call is
	// unambiguously an optimizer failure rather than a probe failure.
	newProbeObjective := func() AOQTSidecarPreparedIPObjective {
		objective, err := objectiveFromAOQTContract(set.Manifest.ObjectiveContract)
		if err != nil {
			t.Fatalf("new probe objective: %v", err)
		}
		objective.workspace.evaluateOverride = func(input AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error) {
			const slope = float32(10)
			value := float32(100)
			for _, coordinate := range input.Query {
				value += slope * coordinate
			}
			weightSum := input.Row.Weights.Q3Gain + input.Row.Weights.Q3OrderGuard + input.Row.Weights.Q3ScoreDistill + input.Row.Weights.Q5OrderGuard + input.Row.Weights.Q5ScoreDistill + input.Row.Weights.NFBoundaryGuard
			result := AOQTSidecarObjectiveResult{
				QueryGrad:  make([]float32, len(input.Query)),
				Components: AOQTSidecarObjectiveComponents{},
				Activation: aoqtActiveObjectiveForWeights(input.Row.Weights, len(input.Candidates)),
			}
			for i := range result.QueryGrad {
				result.QueryGrad[i] = slope * weightSum
			}
			for i := range input.Candidates {
				result.CandidateGrads = append(result.CandidateGrads, make([]float32, len(input.Candidates[i])))
			}
			for _, item := range []struct {
				weight float32
				set    func(*AOQTSidecarObjectiveComponents, float32)
			}{
				{input.Row.Weights.Q3Gain, func(c *AOQTSidecarObjectiveComponents, v float32) { c.Q3Gain += v }},
				{input.Row.Weights.Q3OrderGuard, func(c *AOQTSidecarObjectiveComponents, v float32) { c.Q3OrderGuard += v }},
				{input.Row.Weights.Q3ScoreDistill, func(c *AOQTSidecarObjectiveComponents, v float32) { c.Q3ScoreDistill += v }},
				{input.Row.Weights.Q5OrderGuard, func(c *AOQTSidecarObjectiveComponents, v float32) { c.Q5OrderGuard += v }},
				{input.Row.Weights.Q5ScoreDistill, func(c *AOQTSidecarObjectiveComponents, v float32) { c.Q5ScoreDistill += v }},
				{input.Row.Weights.NFBoundaryGuard, func(c *AOQTSidecarObjectiveComponents, v float32) { c.NFBoundaryGuard += v }},
			} {
				component := item.weight * value
				item.set(&result.Components, component)
				result.Loss += component
			}
			return result, nil
		}
		return objective
	}
	probeObjective := newProbeObjective()
	probeCalls := 0
	probeEvaluate := probeObjective.workspace.evaluateOverride
	probeObjective.workspace.evaluateOverride = func(input AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error) {
		probeCalls++
		return probeEvaluate(input)
	}
	probeTrainer, err := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{
		PairingSeed:                                AOQTSidecarMaterializerTopologySeed,
		WorkplanSeed:                               AOQTSidecarMaterializerQuantSeed,
		OptimizerMode:                              AOQTSidecarOptimizerModeV7R2DevTrustRegion,
		ForwardConsistencyProbeRequired:            true,
		ForwardConsistencyProbeFoldID:              trainCfg.FoldID,
		ForwardConsistencyProbeSplitManifestSHA256: trainCfg.ExpectedSplitManifestSHA256,
		MaxSteps:                   1,
		LearningRate:               0.01,
		TrainingContract:           set.Manifest.TrainingContract,
		CandidateEligibilityPolicy: cloneAOQTSidecarCandidateEligibilityPolicy(set.Manifest.CandidateEligibilityPolicy),
	})
	if err != nil {
		t.Fatalf("new probe trainer: %v", err)
	}
	probeReceipt, err := probeTrainer.runForwardConsistencyProbe(set, probeObjective)
	if err != nil || !probeReceipt.Passed || probeCalls == 0 {
		t.Fatalf("synthetic probe setup receipt=%+v calls=%d err=%v", probeReceipt, probeCalls, err)
	}

	result, err := runAOQTSidecarTraining(trainCfg, func(AOQTSidecarObjectiveContract) (AOQTSidecarVectorObjective, error) {
		objective := newProbeObjective()
		calls := 0
		evaluate := objective.workspace.evaluateOverride
		objective.workspace.evaluateOverride = func(input AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error) {
			calls++
			if calls > probeCalls {
				return AOQTSidecarObjectiveResult{}, fmt.Errorf("forced optimizer failure after passed probe")
			}
			return evaluate(input)
		}
		return objective, nil
	})
	if err == nil {
		t.Fatal("passed-probe optimizer failure unexpectedly succeeded")
	}
	if result.DevFailureDiagnostics == nil || result.DevFailureDiagnostics.ForwardConsistencyProbe == nil {
		t.Fatalf("passed-probe optimizer failure diagnostics = %+v, want bound probe diagnostics", result.DevFailureDiagnostics)
	}
	diagnostics := result.DevFailureDiagnostics
	if !diagnostics.ForwardConsistencyProbe.Passed {
		t.Fatalf("optimizer failure was classified with a failed probe: %+v", diagnostics.ForwardConsistencyProbe)
	}
	if diagnostics.FailureKind != "dev_training_failed" {
		t.Fatalf("failure kind = %q, want dev_training_failed", diagnostics.FailureKind)
	}
	if err := diagnostics.Validate(); err != nil {
		t.Fatalf("passed-probe optimizer failure diagnostics invalid: %v", err)
	}
	data, err := os.ReadFile(AOQTDevFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath))
	if err != nil {
		t.Fatalf("read passed-probe optimizer failure diagnostics: %v", err)
	}
	var decoded AOQTSidecarDevFailClosedDiagnostics
	if err := strictUnmarshalAOQT(data, &decoded); err != nil {
		t.Fatalf("decode passed-probe optimizer failure diagnostics: %v", err)
	}
	if decoded.FailureKind != "dev_training_failed" || !decoded.ForwardConsistencyProbe.Passed {
		t.Fatalf("written failure diagnostics = %+v, want dev_training_failed with passed probe", decoded)
	}
}

func TestAOQTSidecarTrainRunnerV7R2ProbeOnlyWritesOnlyBoundReceipt(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	splitManifestPath := writeV7DevSplitManifestForTrainRunnerTest(t, trainCfg, "fold-0", nil)
	receiptPath := filepath.Join(t.TempDir(), "v7-r2-probe.receipt.json")
	trainCfg.PlanOnly = false
	trainCfg.MaxSteps = 99 // ignored by probe-only mode; no optimizer step is legal.
	trainCfg.LearningRate = 0.01
	trainCfg.OptimizerMode = AOQTSidecarOptimizerModeV7R2DevTrustRegion
	trainCfg.RequireForwardConsistencyProbe = true
	trainCfg.ForwardConsistencyProbeOnly = true
	trainCfg.ForwardConsistencyProbeReceiptJSONPath = receiptPath
	trainCfg.AllowDevOnly = true
	trainCfg.DevOutputDir = ""
	trainCfg.SplitManifestPath = splitManifestPath
	trainCfg.ExpectedSplitManifestSHA256 = mustSHA256FileAOQTTest(t, splitManifestPath)
	trainCfg.FoldID = "fold-0"

	result, err := runAOQTSidecarTraining(trainCfg, aoqtRunnerObjectiveFromContract)
	if err == nil || result.ForwardConsistencyProbe == nil || result.ForwardConsistencyProbe.Passed {
		t.Fatalf("probe-only precondition result=%+v err=%v, want failed receipt", result, err)
	}
	if result.ForwardConsistencyProbeReceiptJSONPath != receiptPath {
		t.Fatalf("probe-only receipt path = %q, want %q", result.ForwardConsistencyProbeReceiptJSONPath, receiptPath)
	}
	data, readErr := os.ReadFile(receiptPath)
	if readErr != nil {
		t.Fatalf("read probe-only receipt: %v", readErr)
	}
	var receipt AOQTSidecarForwardConsistencyProbeReceipt
	if err := strictUnmarshalAOQT(data, &receipt); err != nil {
		t.Fatalf("decode probe-only receipt: %v", err)
	}
	if err := receipt.ValidateBound(); err != nil {
		t.Fatalf("probe-only receipt validation: %v", err)
	}
	if receipt.Passed || strings.TrimSpace(receipt.FailureReason) == "" {
		t.Fatalf("probe-only receipt = %+v, want failed receipt", receipt)
	}
	t.Run("bound-selection-and-contract-tamper", func(t *testing.T) {
		mutate := func(name string, fn func(*AOQTSidecarForwardConsistencyProbeReceipt)) {
			t.Helper()
			tampered := receipt
			tampered.SubsetRowIDs = append([]string(nil), receipt.SubsetRowIDs...)
			tampered.ProtectedComponents = append([]string(nil), receipt.ProtectedComponents...)
			fn(&tampered)
			if err := tampered.ValidateBound(); err == nil {
				t.Fatalf("tampered %s receipt unexpectedly validated", name)
			}
		}
		mutate("selected-row-membership", func(r *AOQTSidecarForwardConsistencyProbeReceipt) {
			r.SubsetRowIDs[0] = "forged-row-id"
			r.SubsetRowIDSHA256 = aoqtRowIDSHA256(r.SubsetRowIDs)
		})
		mutate("selection-seed", func(r *AOQTSidecarForwardConsistencyProbeReceipt) {
			r.SelectionSeedSHA256 = strings.Repeat("a", 64)
		})
		mutate("bound-io-seed", func(r *AOQTSidecarForwardConsistencyProbeReceipt) {
			r.Binding.IOReport.TurboQuantSeed++
			r.SelectionSeedSHA256 = sha256AOQTForwardConsistencySelectionSeed(
				r.Binding.IOReport.TurboQuantSeed,
				r.Binding.SplitBinding.FoldID,
				r.Binding.SplitBinding.SplitManifestSHA256,
				r.Binding.Inputs.DatasetManifestSHA256,
				r.Binding.SplitBinding.MaterializedRowIDSHA256,
			)
		})
		mutate("protected-components", func(r *AOQTSidecarForwardConsistencyProbeReceipt) {
			r.ProtectedComponents = []string{"q3_order_guard"}
		})
	})
	if result.Metrics.Schema != "" || result.DevEvidence != nil || result.DevFailureDiagnostics != nil || result.PackageResult != nil {
		t.Fatalf("probe-only returned non-receipt artifacts: metrics=%+v dev=%+v failure=%+v package=%+v", result.Metrics, result.DevEvidence, result.DevFailureDiagnostics, result.PackageResult)
	}
	for _, path := range []string{
		trainCfg.MetricsJSONPath,
		AOQTDevFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath),
		AOQTFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath),
		filepath.Join(trainCfg.DevOutputDir, "aoqt-sidecar-dev-evidence.json"),
	} {
		if path == "" || path == ".dev.failclosed.json" || path == ".failclosed.json" {
			continue
		}
		if _, statErr := os.Lstat(path); !os.IsNotExist(statErr) {
			t.Fatalf("probe-only wrote forbidden artifact %q: %v", path, statErr)
		}
	}
}

func TestAOQTSidecarTrainRunnerRejectsForgedV7DevSplitManifestPayload(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	splitManifestPath := writeV7DevSplitManifestForTrainRunnerTest(t, trainCfg, "fold-0", func(manifest map[string]any) {
		manifest["seed"] = "forged-after-payload-hash"
	})
	trainCfg.PlanOnly = false
	trainCfg.MaxSteps = 1
	trainCfg.OptimizerMode = AOQTSidecarOptimizerModeV7DevTrustRegion
	trainCfg.AllowDevOnly = true
	trainCfg.DevOutputDir = t.TempDir()
	trainCfg.SplitManifestPath = splitManifestPath
	trainCfg.ExpectedSplitManifestSHA256 = mustSHA256FileAOQTTest(t, splitManifestPath)
	trainCfg.FoldID = "fold-0"
	called := false
	_, err := runAOQTSidecarTraining(trainCfg, func(AOQTSidecarObjectiveContract) (AOQTSidecarVectorObjective, error) {
		called = true
		return nil, fmt.Errorf("objective factory must not run after split manifest forgery")
	})
	if err == nil || !strings.Contains(err.Error(), "provenance.manifest_sha256 does not bind split payload") {
		t.Fatalf("forged split manifest error = %v, want payload binding rejection", err)
	}
	if called {
		t.Fatalf("objective factory ran after forged split manifest")
	}
}

func TestAOQTSidecarTrainRunnerRejectsUnknownV7DevSplitFold(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	splitManifestPath := writeV7DevSplitManifestForTrainRunnerTest(t, trainCfg, "fold-0", nil)
	trainCfg.PlanOnly = false
	trainCfg.MaxSteps = 1
	trainCfg.OptimizerMode = AOQTSidecarOptimizerModeV7DevTrustRegion
	trainCfg.AllowDevOnly = true
	trainCfg.DevOutputDir = t.TempDir()
	trainCfg.SplitManifestPath = splitManifestPath
	trainCfg.ExpectedSplitManifestSHA256 = mustSHA256FileAOQTTest(t, splitManifestPath)
	trainCfg.FoldID = "fold-1"
	called := false
	_, err := runAOQTSidecarTraining(trainCfg, func(AOQTSidecarObjectiveContract) (AOQTSidecarVectorObjective, error) {
		called = true
		return nil, fmt.Errorf("objective factory must not run after wrong fold")
	})
	if err == nil || !strings.Contains(err.Error(), "does not contain fold") {
		t.Fatalf("wrong fold error = %v, want fold rejection", err)
	}
	if called {
		t.Fatalf("objective factory ran after wrong fold")
	}
}

func TestAOQTSidecarTrainRunnerRejectsWrongV7DevTrainRows(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	splitManifestPath := writeV7DevSplitManifestForTrainRunnerTest(t, trainCfg, "fold-0", func(manifest map[string]any) {
		fold := manifest["folds"].([]any)[0].(map[string]any)
		train := fold["train"].(map[string]any)
		rows := append([]string(nil), train["row_ids"].([]string)...)
		rows[len(rows)-1] += ".forged"
		train["row_ids"] = rows
		train["row_ids_sha256"] = mustSHA256JSONAOQT(rows)
		provenance := manifest["provenance"].(map[string]any)
		delete(provenance, "manifest_sha256")
		payloadSHA, err := aoqtV7SplitPayloadSHA256(manifest)
		if err != nil {
			t.Fatalf("rehash tampered split manifest: %v", err)
		}
		provenance["manifest_sha256"] = payloadSHA
	})
	trainCfg.PlanOnly = false
	trainCfg.MaxSteps = 1
	trainCfg.OptimizerMode = AOQTSidecarOptimizerModeV7DevTrustRegion
	trainCfg.AllowDevOnly = true
	trainCfg.DevOutputDir = t.TempDir()
	trainCfg.SplitManifestPath = splitManifestPath
	trainCfg.ExpectedSplitManifestSHA256 = mustSHA256FileAOQTTest(t, splitManifestPath)
	trainCfg.FoldID = "fold-0"
	called := false
	_, err := runAOQTSidecarTraining(trainCfg, func(AOQTSidecarObjectiveContract) (AOQTSidecarVectorObjective, error) {
		called = true
		return nil, fmt.Errorf("objective factory must not run after wrong train rows")
	})
	if err == nil || !strings.Contains(err.Error(), "train row_ids do not exactly match materialized calibration rows") {
		t.Fatalf("wrong train rows error = %v, want row binding rejection", err)
	}
	if called {
		t.Fatalf("objective factory ran after wrong train rows")
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

func writeV7DevSplitManifestForTrainRunnerTest(t *testing.T, trainCfg AOQTSidecarTrainRunnerConfig, foldID string, mutate func(map[string]any)) string {
	t.Helper()
	rowsData, err := os.ReadFile(trainCfg.RowsJSONLPath)
	if err != nil {
		t.Fatalf("read rows JSONL: %v", err)
	}
	rows, err := parseAOQTCalibrationRowsJSONL(rowsData, trainCfg.RowsJSONLPath)
	if err != nil {
		t.Fatalf("parse rows JSONL: %v", err)
	}
	rowIDs := make([]string, 0, len(rows))
	qidsByDataset := map[string][]string{}
	seenQIDs := map[string]map[string]bool{}
	for _, row := range rows {
		rowIDs = append(rowIDs, row.RowID)
		if seenQIDs[row.Dataset] == nil {
			seenQIDs[row.Dataset] = map[string]bool{}
		}
		if !seenQIDs[row.Dataset][row.QueryID] {
			seenQIDs[row.Dataset][row.QueryID] = true
			qidsByDataset[row.Dataset] = append(qidsByDataset[row.Dataset], row.QueryID)
		}
	}
	sort.Strings(rowIDs)
	for dataset := range qidsByDataset {
		sort.Strings(qidsByDataset[dataset])
	}
	qidCounts := map[string]int{}
	for dataset, qids := range qidsByDataset {
		qidCounts[dataset] = len(qids)
	}
	train := map[string]any{
		"row_count":              len(rowIDs),
		"row_ids":                rowIDs,
		"row_ids_sha256":         mustSHA256JSONAOQT(rowIDs),
		"qid_count_by_dataset":   qidCounts,
		"qids_by_dataset":        qidsByDataset,
		"qids_by_dataset_sha256": mustSHA256JSONAOQT(qidsByDataset),
	}
	devQIDs := map[string][]string{}
	dev := map[string]any{
		"row_count":              0,
		"row_ids":                []string{},
		"row_ids_sha256":         mustSHA256JSONAOQT([]string{}),
		"qid_count_by_dataset":   map[string]int{},
		"qids_by_dataset":        devQIDs,
		"qids_by_dataset_sha256": mustSHA256JSONAOQT(devQIDs),
	}
	manifest := map[string]any{
		"schema":                        AOQTV7DevSplitManifestSchema,
		"mode":                          "plan_only_split_manifest",
		"actual_training_ran":           false,
		"actual_eval_ran":               false,
		"actual_official_data_eval_ran": false,
		"seed":                          "go-runner-test",
		"fold_count":                    1,
		"reserve_fraction":              json.Number("0.0"),
		"source_plan": map[string]any{
			"sha256":         strings.Repeat("a", 64),
			"rows_sha256":    strings.Repeat("b", 64),
			"row_ids_sha256": strings.Repeat("c", 64),
		},
		"official_qid_registry": map[string]any{
			"manifest_sha256": strings.Repeat("d", 64),
			"source_sha256":   strings.Repeat("e", 64),
		},
		"folds": []any{
			map[string]any{
				"name":  foldID,
				"train": train,
				"dev":   dev,
			},
		},
		"provenance": map[string]any{},
	}
	provenance := manifest["provenance"].(map[string]any)
	payloadSHA, err := aoqtV7SplitPayloadSHA256(manifest)
	if err != nil {
		t.Fatalf("hash split manifest payload: %v", err)
	}
	provenance["manifest_sha256"] = payloadSHA
	if mutate != nil {
		mutate(manifest)
	}
	path := filepath.Join(t.TempDir(), "split.json")
	if err := writeJSONFileAOQT(path, manifest); err != nil {
		t.Fatalf("write split manifest: %v", err)
	}
	return path
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
