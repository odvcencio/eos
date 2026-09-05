package eosruntime

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

type aoqtPostFitFixture struct {
	cfg          AOQTSidecarTrainRunnerConfig
	set          AOQTSidecarCalibrationSet
	ioReport     AOQTSidecarCalibrationIOReport
	preflight    AOQTSidecarMaterializePreflight
	preflightSHA string
	metrics      AOQTSidecarRunMetrics
	policy       AOQTSidecarCandidateEligibilityPolicy
}

func newAOQTPostFitFixture(t *testing.T) aoqtPostFitFixture {
	t.Helper()
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize synthetic AOQT fixture: %v", err)
	}
	cfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	cfg.PlanOnly = false
	cfg.MaxSteps = 1
	cfg.OutputArtifactPath = filepath.Join(t.TempDir(), "candidate.mll")
	manifest := readAOQTManifestForTrainRunnerTest(t, materializerCfg.ManifestJSONPath)
	topology, err := expectedAOQTSidecarTrainTopology()
	if err != nil {
		t.Fatalf("expected synthetic AOQT topology: %v", err)
	}
	set, ioReport, err := LoadAOQTSidecarCalibrationSet(AOQTSidecarCalibrationIOConfig{
		ManifestPath:                        cfg.ManifestPath,
		RowsJSONLPath:                       cfg.RowsJSONLPath,
		ExpectedManifestSHA256:              cfg.ExpectedManifestSHA256,
		ExpectedRowsSHA256:                  cfg.ExpectedRowsSHA256,
		ExpectedAnchorArtifactSHA256:        cfg.ExpectedAnchorArtifactSHA256,
		ExpectedAnchorPackageManifestSHA256: cfg.ExpectedAnchorPackageManifestSHA256,
		ExpectedAnchorEmbeddingSpaceID:      cfg.ExpectedAnchorEmbeddingSpaceID,
		ExpectedCompatibilityDigest:         cfg.ExpectedCompatibilityDigest,
		ExpectedTurboQuantSeed:              AOQTSidecarMaterializerQuantSeed,
		ExpectedTopology:                    topology,
		ExpectedSourceArtifactHashes:        manifest.SourceArtifactHashes,
		ExpectedVectorCacheHashes:           manifest.VectorCacheHashes,
	})
	if err != nil {
		t.Fatalf("load synthetic AOQT fixture: %v", err)
	}
	preflight, preflightSHA, err := loadAndValidateAOQTTrainPreflight(cfg, set, ioReport)
	if err != nil {
		t.Fatalf("validate synthetic AOQT preflight: %v", err)
	}
	trainer, err := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{
		PairingSeed:                AOQTSidecarMaterializerTopologySeed,
		WorkplanSeed:               AOQTSidecarMaterializerQuantSeed,
		PlanOnly:                   false,
		MaxSteps:                   cfg.MaxSteps,
		LearningRate:               cfg.LearningRate,
		AngleCap:                   AOQTSidecarDefaultAngleCap,
		MaxAngleCap:                AOQTSidecarHardMaxAngleCap,
		CaptureProposalReceipts:    true,
		TrainingContract:           set.Manifest.TrainingContract,
		CandidateEligibilityPolicy: cloneAOQTSidecarCandidateEligibilityPolicy(set.Manifest.CandidateEligibilityPolicy),
	})
	if err != nil {
		t.Fatalf("new synthetic AOQT trainer: %v", err)
	}
	objective, err := objectiveFromAOQTContract(set.Manifest.ObjectiveContract)
	if err != nil {
		t.Fatalf("new synthetic AOQT objective: %v", err)
	}
	summary, err := trainer.Fit(set, objective)
	if err != nil {
		t.Fatalf("fit synthetic accepted-step fixture: %v", err)
	}
	metrics, err := NewAOQTSidecarRunMetrics(set, summary)
	if err != nil {
		t.Fatalf("build synthetic metrics: %v", err)
	}
	policy, err := AOQTSidecarTrainingContractPolicy(metrics.TrainingContract, metrics.CandidateEligibilityPolicy)
	if err != nil {
		t.Fatalf("resolve synthetic eligibility policy: %v", err)
	}
	if err := ValidateAOQTSidecarCandidateEligibility(metrics, policy); err != nil {
		t.Fatalf("synthetic baseline metrics unexpectedly ineligible: %v", err)
	}
	if metrics.Summary.OptimizerDiagnostics == nil || metrics.Summary.OptimizerDiagnostics.ProposalReceipts == nil || len(*metrics.Summary.OptimizerDiagnostics.ProposalReceipts) == 0 {
		t.Fatalf("synthetic metrics lack optimizer proposal receipts: %+v", metrics.Summary.OptimizerDiagnostics)
	}
	return aoqtPostFitFixture{cfg: cfg, set: set, ioReport: ioReport, preflight: preflight, preflightSHA: preflightSHA, metrics: metrics, policy: policy}
}

func mutateAOQTPostFitQ5Regression(metrics *AOQTSidecarRunMetrics) {
	metrics.Summary.InitialObjectiveComponents.Q3Gain += 1e-3
	metrics.Summary.InitialLoss = metrics.Summary.InitialObjectiveComponents.Sum()
	metrics.Summary.FinalObjectiveComponents.Q5OrderGuard += 1e-3
	metrics.Summary.FinalLoss = metrics.Summary.FinalObjectiveComponents.Sum()
}

func cloneAOQTPostFitDiagnostics(t *testing.T, source AOQTSidecarPostFitFailClosedDiagnostics) AOQTSidecarPostFitFailClosedDiagnostics {
	t.Helper()
	data, err := json.Marshal(source)
	if err != nil {
		t.Fatalf("marshal post-fit diagnostics clone: %v", err)
	}
	var clone AOQTSidecarPostFitFailClosedDiagnostics
	if err := strictUnmarshalAOQT(data, &clone); err != nil {
		t.Fatalf("unmarshal post-fit diagnostics clone: %v", err)
	}
	return clone
}

func TestAOQTPostFitDiagnosticsPreserveAcceptedMetricsAndReceipts(t *testing.T) {
	fixture := newAOQTPostFitFixture(t)
	mutateAOQTPostFitQ5Regression(&fixture.metrics)
	failure := ValidateAOQTSidecarCandidateEligibility(fixture.metrics, fixture.policy)
	if failure == nil || !strings.Contains(failure.Error(), "q5_order_guard component regressed") {
		t.Fatalf("mutated q5 eligibility error = %v, want unchanged q5 guard rejection", failure)
	}
	diagnostics, err := newAOQTPostFitFailClosedDiagnostics(fixture.metrics, fixture.ioReport, fixture.preflight, fixture.cfg.PreflightJSONPath, fixture.preflightSHA, fixture.policy, failure)
	if err != nil {
		t.Fatalf("construct post-fit diagnostics: %v", err)
	}
	if err := diagnostics.Validate(); err != nil {
		t.Fatalf("constructed post-fit diagnostics invalid: %v", err)
	}
	if diagnostics.Schema != AOQTSidecarPostFitFailClosedDiagnosticsSchema || diagnostics.FailureKind != AOQTSidecarPostFitCandidateEligibilityFailureKind || diagnostics.QualityClaim || diagnostics.CandidatePublished {
		t.Fatalf("post-fit diagnostics identity/flags = %+v", diagnostics)
	}
	if diagnostics.Error != failure.Error() || !reflect.DeepEqual(diagnostics.Metrics.Summary, fixture.metrics.Summary) {
		t.Fatalf("post-fit diagnostics did not preserve exact failure/summary")
	}
	if diagnostics.Metrics.Summary.OptimizerDiagnosticsSHA256 != fixture.metrics.Summary.OptimizerDiagnosticsSHA256 || !reflect.DeepEqual(diagnostics.Metrics.Summary.OptimizerDiagnostics, fixture.metrics.Summary.OptimizerDiagnostics) {
		t.Fatalf("post-fit diagnostics did not preserve optimizer receipts/hash")
	}
	if diagnostics.IOReport != fixture.ioReport || !reflect.DeepEqual(diagnostics.Preflight, fixture.preflight) || diagnostics.PreflightPath != fixture.cfg.PreflightJSONPath || diagnostics.PreflightSHA256 != fixture.preflightSHA {
		t.Fatalf("post-fit diagnostics did not preserve authentic IO/preflight bindings")
	}
	path := AOQTPostFitFailClosedDiagnosticsPath(fixture.cfg.MetricsJSONPath)
	if err := writeAOQTPostFitFailClosedDiagnosticsFile(path, diagnostics); err != nil {
		t.Fatalf("write post-fit diagnostics: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read post-fit diagnostics: %v", err)
	}
	var roundTrip AOQTSidecarPostFitFailClosedDiagnostics
	if err := strictUnmarshalAOQT(data, &roundTrip); err != nil {
		t.Fatalf("decode post-fit diagnostics: %v", err)
	}
	if err := roundTrip.Validate(); err != nil {
		t.Fatalf("round-trip post-fit diagnostics invalid: %v", err)
	}
	if _, err := os.Lstat(fixture.cfg.MetricsJSONPath); !os.IsNotExist(err) {
		t.Fatalf("normal metrics output state = %v, want absent", err)
	}
	for _, candidatePath := range aoqtCandidatePathMap(aoqtCandidateOutputPaths(fixture.cfg.OutputArtifactPath, true)) {
		if candidatePath == "" {
			continue
		}
		if _, err := os.Lstat(candidatePath); !os.IsNotExist(err) {
			t.Fatalf("candidate output %q state = %v, want absent", candidatePath, err)
		}
	}
}

func TestAOQTPostFitDiagnosticsRejectsPlanZeroReceiptsAndFakeFailure(t *testing.T) {
	fixture := newAOQTPostFitFixture(t)
	mutateAOQTPostFitQ5Regression(&fixture.metrics)
	failure := ValidateAOQTSidecarCandidateEligibility(fixture.metrics, fixture.policy)
	valid, err := newAOQTPostFitFailClosedDiagnostics(fixture.metrics, fixture.ioReport, fixture.preflight, fixture.cfg.PreflightJSONPath, fixture.preflightSHA, fixture.policy, failure)
	if err != nil {
		t.Fatalf("construct baseline post-fit diagnostics: %v", err)
	}
	for name, mutate := range map[string]func(*AOQTSidecarPostFitFailClosedDiagnostics){
		"wrong-schema":        func(d *AOQTSidecarPostFitFailClosedDiagnostics) { d.Schema = "wrong" },
		"wrong-kind":          func(d *AOQTSidecarPostFitFailClosedDiagnostics) { d.FailureKind = "zero_accepted_safe_steps" },
		"quality-claim":       func(d *AOQTSidecarPostFitFailClosedDiagnostics) { d.QualityClaim = true },
		"candidate-published": func(d *AOQTSidecarPostFitFailClosedDiagnostics) { d.CandidatePublished = true },
		"io-hash":             func(d *AOQTSidecarPostFitFailClosedDiagnostics) { d.IOReport.RowsSHA256 = strings.Repeat("a", 64) },
		"preflight-hash":      func(d *AOQTSidecarPostFitFailClosedDiagnostics) { d.PreflightSHA256 = strings.Repeat("b", 64) },
		"contract":            func(d *AOQTSidecarPostFitFailClosedDiagnostics) { d.IOReport.TrainingContract = "tampered-contract" },
		"nil-receipts": func(d *AOQTSidecarPostFitFailClosedDiagnostics) {
			d.Metrics.Summary.OptimizerDiagnostics.ProposalReceipts = nil
		},
		"plan-only": func(d *AOQTSidecarPostFitFailClosedDiagnostics) {
			d.Metrics.Plan.PlanOnly = true
			d.Metrics.Summary.Plan.PlanOnly = true
			d.Metrics.Plan.StepCount = 0
			d.Metrics.Summary.Plan.StepCount = 0
			d.Metrics.Summary.Steps = 0
			d.Metrics.Summary.OptimizerDiagnostics = nil
			d.Metrics.Summary.OptimizerDiagnosticsSHA256 = ""
		},
	} {
		t.Run(name, func(t *testing.T) {
			tampered := cloneAOQTPostFitDiagnostics(t, valid)
			mutate(&tampered)
			if err := tampered.Validate(); err == nil {
				t.Fatalf("tampered post-fit diagnostics unexpectedly validated")
			}
		})
	}
	if _, err := newAOQTPostFitFailClosedDiagnostics(fixture.metrics, fixture.ioReport, fixture.preflight, fixture.cfg.PreflightJSONPath, fixture.preflightSHA, fixture.policy, fmt.Errorf("fake failure")); err == nil || !strings.Contains(err.Error(), "does not match unchanged validator") {
		t.Fatalf("fake failure construction error = %v, want eligibility reproduction rejection", err)
	}
}

func TestAOQTPostFitDiagnosticsWriterIsExclusiveAndCollisionBound(t *testing.T) {
	fixture := newAOQTPostFitFixture(t)
	mutateAOQTPostFitQ5Regression(&fixture.metrics)
	failure := ValidateAOQTSidecarCandidateEligibility(fixture.metrics, fixture.policy)
	diagnostics, err := newAOQTPostFitFailClosedDiagnostics(fixture.metrics, fixture.ioReport, fixture.preflight, fixture.cfg.PreflightJSONPath, fixture.preflightSHA, fixture.policy, failure)
	if err != nil {
		t.Fatalf("construct post-fit diagnostics: %v", err)
	}
	path := AOQTPostFitFailClosedDiagnosticsPath(fixture.cfg.MetricsJSONPath)
	sentinel := []byte("existing post-fit sentinel\n")
	if err := os.WriteFile(path, sentinel, 0o644); err != nil {
		t.Fatalf("write sentinel: %v", err)
	}
	if err := writeAOQTPostFitFailClosedDiagnosticsFile(path, diagnostics); err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("exclusive writer error = %v, want collision", err)
	}
	got, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(got, sentinel) {
		t.Fatalf("sentinel after collision = %q (%v), want unchanged", got, err)
	}
	if err := ensureAOQTPostFitFailClosedDiagnosticsAbsent(fixture.cfg.MetricsJSONPath); err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("preflight post-fit collision = %v, want rejection", err)
	}
	if err := validateAOQTTrainRunnerOutputPathDistinctness(AOQTSidecarTrainRunnerConfig{
		MetricsJSONPath:    fixture.cfg.MetricsJSONPath,
		OutputArtifactPath: path,
		ManifestPath:       fixture.cfg.ManifestPath,
		RowsJSONLPath:      fixture.cfg.RowsJSONLPath,
		PreflightJSONPath:  fixture.cfg.PreflightJSONPath,
	}); err == nil || !strings.Contains(err.Error(), "output path collision") {
		t.Fatalf("metrics/post-fit output collision = %v, want rejection", err)
	}
}

func TestAOQTPostFitDiagnosticsPathReservedBeforeTrainingForPlanAndNonPlan(t *testing.T) {
	for _, planOnly := range []bool{true, false} {
		t.Run(map[bool]string{true: "plan-only", false: "non-plan"}[planOnly], func(t *testing.T) {
			fixture := newAOQTPostFitFixture(t)
			fixture.cfg.PlanOnly = planOnly
			if planOnly {
				fixture.cfg.OutputArtifactPath = ""
				fixture.cfg.MaxSteps = 0
			}
			path := AOQTPostFitFailClosedDiagnosticsPath(fixture.cfg.MetricsJSONPath)
			sentinel := []byte("reserved post-fit sentinel\n")
			if err := os.WriteFile(path, sentinel, 0o644); err != nil {
				t.Fatalf("write reserved-path sentinel: %v", err)
			}
			called := false
			_, err := runAOQTSidecarTraining(fixture.cfg, func(AOQTSidecarObjectiveContract) (AOQTSidecarVectorObjective, error) {
				called = true
				return nil, fmt.Errorf("objective factory must not run after reserved post-fit path")
			})
			if err == nil || !strings.Contains(err.Error(), "post-fit fail-closed diagnostics output") || !strings.Contains(err.Error(), "already exists") {
				t.Fatalf("reserved post-fit path error = %v, want pre-training collision", err)
			}
			if called {
				t.Fatalf("objective factory ran after reserved post-fit path")
			}
			got, readErr := os.ReadFile(path)
			if readErr != nil || !bytes.Equal(got, sentinel) {
				t.Fatalf("reserved post-fit sentinel after rejection = %q (%v), want unchanged", got, readErr)
			}
		})
	}
}

func TestAOQTSidecarTrainRunnerPostFitFailurePublishesOnlyDistinctArtifact(t *testing.T) {
	fixture := newAOQTPostFitFixture(t)
	if err := validateAOQTSidecarTrainRunnerConfig(fixture.cfg); err != nil {
		t.Fatalf("synthetic runner config rejected before post-fit path check: %v", err)
	}
	postFitPath := AOQTPostFitFailClosedDiagnosticsPath(fixture.cfg.MetricsJSONPath)
	if postFitPath == fixture.cfg.MetricsJSONPath || postFitPath == fixture.cfg.OutputArtifactPath {
		t.Fatalf("post-fit diagnostics path is not distinct: %q", postFitPath)
	}
}

// TestAOQTSidecarTrainRunnerPostFitFailure exercises the actual runner branch
// with the concrete prepared-IP objective.  The override only changes the
// synthetic first/final telemetry evaluations; it cannot bypass the trainer's
// concrete-objective check or the unchanged eligibility validator.
func TestAOQTSidecarTrainRunnerPostFitFailure(t *testing.T) {
	fixture := newAOQTPostFitFixture(t)
	rowCount := len(fixture.set.Rows)
	if rowCount == 0 {
		t.Fatal("synthetic post-fit fixture has no rows")
	}

	var baselineCalls int
	baselineObjective, err := objectiveFromAOQTContract(fixture.set.Manifest.ObjectiveContract)
	if err != nil {
		t.Fatalf("build baseline objective: %v", err)
	}
	baselineSource, err := objectiveFromAOQTContract(fixture.set.Manifest.ObjectiveContract)
	if err != nil {
		t.Fatalf("build baseline source objective: %v", err)
	}
	baselineObjective.workspace.evaluateOverride = func(input AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error) {
		baselineCalls++
		return baselineSource.EvaluateAOQT(input)
	}
	baselineTrainer, err := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{
		PairingSeed:             AOQTSidecarMaterializerTopologySeed,
		WorkplanSeed:            AOQTSidecarMaterializerQuantSeed,
		PlanOnly:                false,
		MaxSteps:                fixture.cfg.MaxSteps,
		LearningRate:            fixture.cfg.LearningRate,
		AngleCap:                AOQTSidecarDefaultAngleCap,
		MaxAngleCap:             AOQTSidecarHardMaxAngleCap,
		CaptureProposalReceipts: true,
		TrainingContract:        fixture.set.Manifest.TrainingContract,
		CandidateEligibilityPolicy: cloneAOQTSidecarCandidateEligibilityPolicy(
			fixture.set.Manifest.CandidateEligibilityPolicy,
		),
	})
	if err != nil {
		t.Fatalf("build baseline trainer: %v", err)
	}
	baselineSummary, err := baselineTrainer.Fit(fixture.set, baselineObjective)
	if err != nil {
		t.Fatalf("baseline synthetic fit: %v", err)
	}
	if baselineSummary.OptimizerDiagnostics == nil || baselineSummary.Steps <= 0 {
		t.Fatalf("baseline synthetic fit did not accept a step: %+v", baselineSummary)
	}
	if baselineCalls <= rowCount*2 {
		t.Fatalf("baseline objective calls = %d, want initial and final evaluation ranges", baselineCalls)
	}
	finalDelta := float32(1e-3)
	initialDelta := baselineSummary.FinalLoss - baselineSummary.InitialLoss + finalDelta + float32(1e-3)
	if initialDelta <= 0 || !isFinite32(initialDelta) {
		t.Fatalf("synthetic initial balancing delta = %.9g is invalid", initialDelta)
	}

	var alteredCalls int
	objectiveFactory := func(contract AOQTSidecarObjectiveContract) (AOQTSidecarVectorObjective, error) {
		objective, err := objectiveFromAOQTContract(contract)
		if err != nil {
			return nil, err
		}
		source, err := objectiveFromAOQTContract(contract)
		if err != nil {
			return nil, err
		}
		objective.workspace.evaluateOverride = func(input AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error) {
			alteredCalls++
			result, err := source.EvaluateAOQT(input)
			if err != nil {
				return AOQTSidecarObjectiveResult{}, err
			}
			switch {
			case alteredCalls <= rowCount:
				result.Components.Q3Gain += initialDelta / float32(rowCount)
				result.Loss = result.Components.Sum()
			case alteredCalls > baselineCalls-rowCount:
				result.Components.Q5OrderGuard += finalDelta / float32(rowCount)
				result.Loss = result.Components.Sum()
			}
			return result, nil
		}
		return objective, nil
	}

	result, err := runAOQTSidecarTraining(fixture.cfg, objectiveFactory)
	if err == nil || !strings.Contains(err.Error(), "q5_order_guard component regressed") {
		t.Fatalf("runner post-fit error = %v, want q5 eligibility rejection", err)
	}
	if alteredCalls != baselineCalls {
		t.Fatalf("runner objective calls = %d, baseline calls = %d; final-evaluation seam moved", alteredCalls, baselineCalls)
	}
	if !reflect.DeepEqual(result.Metrics, AOQTSidecarRunMetrics{}) {
		t.Fatalf("runner failure returned normal metrics instead of zero value: %+v", result.Metrics)
	}
	if result.FailureDiagnostics != nil || result.PackageResult != nil || result.PostFitFailureDiagnostics == nil {
		t.Fatalf("runner post-fit result channels = failure=%v package=%v postfit=%v", result.FailureDiagnostics != nil, result.PackageResult != nil, result.PostFitFailureDiagnostics != nil)
	}
	if err := result.PostFitFailureDiagnostics.Validate(); err != nil {
		t.Fatalf("runner returned invalid post-fit diagnostics: %v", err)
	}
	if result.PostFitFailureDiagnostics.Error != err.Error() {
		t.Fatalf("runner diagnostic error = %q, runner error = %q", result.PostFitFailureDiagnostics.Error, err.Error())
	}
	postFitPath := AOQTPostFitFailClosedDiagnosticsPath(fixture.cfg.MetricsJSONPath)
	data, err := os.ReadFile(postFitPath)
	if err != nil {
		t.Fatalf("read runner post-fit diagnostics: %v", err)
	}
	var written AOQTSidecarPostFitFailClosedDiagnostics
	if err := strictUnmarshalAOQT(data, &written); err != nil {
		t.Fatalf("decode runner post-fit diagnostics: %v", err)
	}
	if err := written.Validate(); err != nil {
		t.Fatalf("written runner post-fit diagnostics invalid: %v", err)
	}
	if _, err := os.Lstat(fixture.cfg.MetricsJSONPath); !os.IsNotExist(err) {
		t.Fatalf("runner normal metrics output state = %v, want absent", err)
	}
	for _, candidatePath := range aoqtCandidatePathMap(aoqtCandidateOutputPaths(fixture.cfg.OutputArtifactPath, true)) {
		if candidatePath == "" {
			continue
		}
		if _, err := os.Lstat(candidatePath); !os.IsNotExist(err) {
			t.Fatalf("runner candidate output %q state = %v, want absent", candidatePath, err)
		}
	}
}
