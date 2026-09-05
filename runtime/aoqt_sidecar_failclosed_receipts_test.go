package eosruntime

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type mixedRejectingAOQTObjective struct {
	config               AOQTSidecarPreparedIPObjectiveConfig
	rowsPerEvaluation    int
	preambleCalls        int
	calls                int
	candidateEvaluations int
}

func (o *mixedRejectingAOQTObjective) AOQTPreparedIPObjectiveConfig() AOQTSidecarPreparedIPObjectiveConfig {
	return o.config
}

func (o *mixedRejectingAOQTObjective) EvaluateAOQT(input AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error) {
	o.calls++
	queryGrad := make([]float32, len(input.Query))
	for i := range queryGrad {
		queryGrad[i] = 1
	}
	candidateGrads := make([][]float32, len(input.Candidates))
	for i := range input.Candidates {
		candidateGrads[i] = make([]float32, len(input.Candidates[i]))
	}
	components := AOQTSidecarObjectiveComponents{
		Q3Gain:         10,
		Q3OrderGuard:   1,
		Q3ScoreDistill: 1,
		Q5OrderGuard:   1,
		Q5ScoreDistill: 1,
	}
	if o.calls > o.preambleCalls {
		candidateOrdinal := (o.calls-o.preambleCalls-1)/o.rowsPerEvaluation + 1
		if candidateOrdinal > o.candidateEvaluations {
			o.candidateEvaluations = candidateOrdinal
		}
		switch {
		case candidateOrdinal >= 1 && candidateOrdinal <= 16:
			components.Q3Gain = 11
		case candidateOrdinal == 17:
			// Preserve q3_gain and all components: no q3 improvement.
		case candidateOrdinal >= 18 && candidateOrdinal <= 37:
			components.Q3Gain = 9
			components.Q3ScoreDistill = 2
		}
	}
	return AOQTSidecarObjectiveResult{
		Loss:           components.Sum(),
		Components:     components,
		QueryGrad:      queryGrad,
		CandidateGrads: candidateGrads,
		Activation:     aoqtActiveObjectiveForWeights(input.Row.Weights, len(input.Candidates)),
	}, nil
}

func TestAOQTSidecarTrainRunnerFailClosedPreserves37ProposalReceipts(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	laneAPolicy := AOQTSidecarScoreDistillBudgetLaneAPolicy()
	materializerCfg.TrainingContract = AOQTSidecarScoreDistillBudgetLaneAContract
	materializerCfg.CandidateEligibilityPolicy = &laneAPolicy
	preflight, err := MaterializeAOQTSidecarCalibration(materializerCfg)
	if err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	trainCfg.PlanOnly = false
	trainCfg.MaxSteps = 1
	trainCfg.OutputArtifactPath = filepath.Join(t.TempDir(), "candidate.mll")
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("get test working directory: %v", err)
	}
	trainCfg.ManifestPath, err = filepath.Rel(cwd, trainCfg.ManifestPath)
	if err != nil {
		t.Fatalf("make manifest path relative: %v", err)
	}
	trainCfg.RowsJSONLPath, err = filepath.Rel(cwd, trainCfg.RowsJSONLPath)
	if err != nil {
		t.Fatalf("make rows path relative: %v", err)
	}
	trainCfg.PreflightJSONPath, err = filepath.Rel(cwd, trainCfg.PreflightJSONPath)
	if err != nil {
		t.Fatalf("make preflight path relative: %v", err)
	}
	var mixed *mixedRejectingAOQTObjective
	objectiveFactory := func(contract AOQTSidecarObjectiveContract) (AOQTSidecarVectorObjective, error) {
		objective, err := objectiveFromAOQTContract(contract)
		if err != nil {
			return nil, err
		}
		mixed = &mixedRejectingAOQTObjective{
			config:            objective.AOQTPreparedIPObjectiveConfig(),
			rowsPerEvaluation: preflight.RowCount,
			preambleCalls:     preflight.RowCount * (2 + len(aoqtActiveProtectedComponentNames(contract.WeightSums))),
		}
		objective.workspace.evaluateOverride = mixed.EvaluateAOQT
		return objective, nil
	}

	result, err := runAOQTSidecarTraining(trainCfg, objectiveFactory)
	if err == nil || !strings.Contains(err.Error(), "accepted zero safe steps") {
		t.Fatalf("run error = %v, want zero-safe-step failure (rows=%d objective=%v)", err, result.IOReport.RowCount, mixed)
	}
	if mixed == nil || mixed.candidateEvaluations != 37 {
		t.Fatalf("candidate objective evaluations = %v, want 37 proposals (rows=%d)", mixed, result.IOReport.RowCount)
	}
	if result.FailureDiagnostics == nil {
		t.Fatalf("runner did not return fail-closed diagnostics: %v", err)
	}
	diagnostics := *result.FailureDiagnostics
	if validateErr := diagnostics.Validate(); validateErr != nil {
		t.Fatalf("returned fail-closed diagnostics invalid: %v", validateErr)
	}
	optimizer := diagnostics.Summary.OptimizerDiagnostics
	if optimizer == nil || optimizer.ProposalAttempts != 37 || optimizer.RejectedProposals != 37 {
		t.Fatalf("optimizer accounting = %+v, want 37 rejected proposals", optimizer)
	}
	if optimizer.ProposalReceipts == nil || len(*optimizer.ProposalReceipts) != 37 {
		t.Fatalf("proposal receipts = %v, want 37 receipts", optimizer.ProposalReceipts)
	}
	if optimizer.RejectionDiagnostics.ReasonCounts.LossIncrease != 16 || optimizer.RejectionDiagnostics.ReasonCounts.NoQ3GainImprovement != 1 || optimizer.RejectionDiagnostics.ReasonCounts.ComponentRegression != 20 {
		t.Fatalf("reason counts = %+v, want loss=16 no-q3=1 component=20", optimizer.RejectionDiagnostics.ReasonCounts)
	}
	if (*optimizer.ProposalReceipts)[36].Ordinal != 37 || (*optimizer.ProposalReceipts)[36].Reason != string(aoqtRejectionComponentRegression) {
		t.Fatalf("last proposal receipt = %+v, want ordinal 37 component regression", (*optimizer.ProposalReceipts)[36])
	}

	diagnosticsPath := AOQTFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath)
	data, readErr := os.ReadFile(diagnosticsPath)
	if readErr != nil {
		t.Fatalf("read fail-closed diagnostics: %v", readErr)
	}
	var written AOQTSidecarFailClosedDiagnostics
	if unmarshalErr := strictUnmarshalAOQT(data, &written); unmarshalErr != nil {
		t.Fatalf("decode fail-closed diagnostics: %v", unmarshalErr)
	}
	if written.Summary.OptimizerDiagnostics == nil || written.Summary.OptimizerDiagnostics.ProposalReceipts == nil || len(*written.Summary.OptimizerDiagnostics.ProposalReceipts) != 37 {
		t.Fatalf("written proposal receipts were not preserved: %v", written.Summary.OptimizerDiagnostics)
	}
	if !strings.Contains(string(data), `"proposal_receipts"`) {
		t.Fatalf("written diagnostics JSON omitted proposal_receipts")
	}
}

func TestAOQTSidecarTrainRunnerSurfacesFailClosedWriterError(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize AOQT sidecar calibration: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	trainCfg.PlanOnly = false
	trainCfg.MaxSteps = 1
	trainCfg.OutputArtifactPath = filepath.Join(t.TempDir(), "candidate.mll")
	metricsParent := t.TempDir()
	trainCfg.MetricsJSONPath = filepath.Join(metricsParent, "metrics.json")
	objectiveFactory := func(contract AOQTSidecarObjectiveContract) (AOQTSidecarVectorObjective, error) {
		// Create the obstruction after runner preflight has checked output
		// absence. This makes the writer failure deterministic for root and
		// non-root test users alike.
		if err := os.WriteFile(AOQTFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath), []byte("obstruction\n"), 0o644); err != nil {
			return nil, err
		}
		objective, err := objectiveFromAOQTContract(contract)
		if err != nil {
			return nil, err
		}
		stateful := &statefulRejectingAOQTObjective{config: objective.AOQTPreparedIPObjectiveConfig()}
		objective.workspace.evaluateOverride = stateful.EvaluateAOQT
		return objective, nil
	}

	result, err := runAOQTSidecarTraining(trainCfg, objectiveFactory)
	if err == nil || !strings.Contains(err.Error(), "failed to write AOQT fail-closed diagnostics") || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("writer error = %v, want surfaced fail-closed writer failure", err)
	}
	if result.FailureDiagnostics == nil {
		t.Fatalf("writer failure discarded fail-closed diagnostics in memory")
	}
	obstructionPath := AOQTFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath)
	obstruction, statErr := os.ReadFile(obstructionPath)
	if statErr != nil || string(obstruction) != "obstruction\n" {
		t.Fatalf("fail-closed diagnostics obstruction after writer failure = %q (%v), want preserved file", obstruction, statErr)
	}
}

func TestAOQTFailClosedCanonicalPathBinding(t *testing.T) {
	dir := t.TempDir()
	declaredPath := filepath.Join(dir, "declared.json")
	actualPath := filepath.Join(dir, "actual.json")
	if err := os.WriteFile(declaredPath, []byte("declared\n"), 0o644); err != nil {
		t.Fatalf("write declared path: %v", err)
	}
	if err := os.WriteFile(actualPath, []byte("actual\n"), 0o644); err != nil {
		t.Fatalf("write actual path: %v", err)
	}
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("get test working directory: %v", err)
	}
	relativeActual, err := filepath.Rel(cwd, declaredPath)
	if err != nil {
		t.Fatalf("make equivalent relative path: %v", err)
	}
	if err := validateAOQTPathBinding(relativeActual, declaredPath, "declared", "actual"); err != nil {
		t.Fatalf("equivalent relative/absolute paths rejected: %v", err)
	}
	if err := validateAOQTPathBinding(declaredPath, actualPath, "declared", "actual"); err == nil || !strings.Contains(err.Error(), "must bind") {
		t.Fatalf("distinct paths accepted: %v", err)
	}
}
