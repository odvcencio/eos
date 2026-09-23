package eosruntime

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func r4ProbeSet(t *testing.T, seed int64) AOQTSidecarCalibrationSet {
	t.Helper()
	base := tinyAOQTCalibrationSet(t, seed)
	rows := make([]AOQTSidecarCalibrationRow, AOQTV7R4ActualCoordinateProbeRowCount)
	for i := range rows {
		rows[i] = cloneAOQTCalibrationSet(base).Rows[0]
		rows[i].RowID = fmt.Sprintf("row-%04d", i+1)
		rows[i].QueryID = fmt.Sprintf("q-%04d", i+1)
		rows[i].QueryVectorID = fmt.Sprintf("qv-%04d", i+1)
		rows[i].Weights.NFBoundaryGuard = 0
	}
	// Keep one static NF-boundary stratum while leaving the other five
	// objective strata available in every row.
	rows[0].Weights.NFBoundaryGuard = 0.75
	rows[0].EligiblePairMask[2][1] = true
	manifest := base.Manifest
	manifest.RowCount = len(rows)
	rowIDs := make([]string, len(rows))
	for i := range rows {
		rowIDs[i] = rows[i].RowID
	}
	manifest.RowIDSHA256 = aoqtRowIDSHA256(rowIDs)
	manifest.ObjectiveContract = tinyAOQTObjectiveConfig(manifest.TurboQuantSeed).ObjectiveContract(sumAOQTRowWeights(rows))
	set := AOQTSidecarCalibrationSet{Manifest: manifest, Rows: rows}
	if err := set.Validate(); err != nil {
		t.Fatalf("r4 fixture invalid: %v", err)
	}
	return set
}

func newR4ProbeTrainer(t *testing.T, set AOQTSidecarCalibrationSet) *AOQTSidecarTrainer {
	t.Helper()
	trainer, err := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{
		PairingSeed:                              1,
		WorkplanSeed:                             2,
		OptimizerMode:                            AOQTSidecarOptimizerModeV7R4DevActualCoordinate,
		ActualCoordinateProbeRequired:            true,
		ActualCoordinateProbeOnly:                true,
		ActualCoordinateProbeFoldID:              "fold-0",
		ActualCoordinateProbeSplitManifestSHA256: hex64("r4-split"),
		LearningRate:                             AOQTV7R4ActualCoordinateProbeLearningRate,
		MaxSteps:                                 0,
		TrainingContract:                         set.Manifest.TrainingContract,
		CandidateEligibilityPolicy:               cloneAOQTSidecarCandidateEligibilityPolicy(set.Manifest.CandidateEligibilityPolicy),
	})
	if err != nil {
		t.Fatalf("new r4 trainer: %v", err)
	}
	return trainer
}

func r4SyntheticObjective(t *testing.T, set AOQTSidecarCalibrationSet, improving bool, active bool) AOQTSidecarPreparedIPObjective {
	t.Helper()
	return preparedAOQTObjectiveWithEvaluatorForSet(t, set, func(input AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error) {
		value := float32(1)
		grad := float32(0)
		if improving {
			value += input.Query[0] * 0.1
			grad = 0.1
		}
		weights := input.Row.Weights
		components := AOQTSidecarObjectiveComponents{
			Q3Gain:          weights.Q3Gain * value,
			Q3OrderGuard:    weights.Q3OrderGuard * 2,
			Q3ScoreDistill:  weights.Q3ScoreDistill * 2,
			Q5OrderGuard:    weights.Q5OrderGuard * 2,
			Q5ScoreDistill:  weights.Q5ScoreDistill * 2,
			NFBoundaryGuard: weights.NFBoundaryGuard * 2,
		}
		activation := aoqtActiveObjectiveForWeights(weights, len(input.Candidates))
		if !active {
			activation = AOQTSidecarObjectiveActivation{}
		}
		candidateGrads := make([][]float32, len(input.Candidates))
		for i := range candidateGrads {
			candidateGrads[i] = make([]float32, len(input.Query))
		}
		return AOQTSidecarObjectiveResult{
			Loss:           components.Sum(),
			Components:     components,
			QueryGrad:      append([]float32{grad * weights.Q3Gain}, make([]float32, len(input.Query)-1)...),
			CandidateGrads: candidateGrads,
			Activation:     activation,
		}, nil
	})
}

func TestAOQTV7R4ActualCoordinateCoverageAndExactSchedule(t *testing.T) {
	set := r4ProbeSet(t, 901)
	trainer := newR4ProbeTrainer(t, set)
	objective := r4SyntheticObjective(t, set, true, true)
	receipt, err := trainer.runActualCoordinateProbe(set, objective)
	if err != nil {
		t.Fatalf("r4 probe: %v", err)
	}
	if !receipt.Passed || receipt.SubsetRowCount != 12 || receipt.EndpointCount != 384 || len(receipt.Endpoints) != 384 {
		t.Fatalf("receipt cardinality = passed=%t rows=%d endpoints=%d/%d", receipt.Passed, receipt.SubsetRowCount, receipt.EndpointCount, len(receipt.Endpoints))
	}
	if len(receipt.BlockSizes) != 0 || receipt.TopCoordinateCount != 32 || len(receipt.Coverage.RequiredComponents) != 6 {
		t.Fatalf("r4 schedule/coverage = blocks=%v top=%d required=%v", receipt.BlockSizes, receipt.TopCoordinateCount, receipt.Coverage.RequiredComponents)
	}
	if receipt.Coverage.ScoreDistillActivationSemantics != AOQTV7R4ScoreDistillActivationSemantics {
		t.Fatalf("score-distill coverage semantics = %q, want %q", receipt.Coverage.ScoreDistillActivationSemantics, AOQTV7R4ScoreDistillActivationSemantics)
	}
	for _, component := range receipt.Coverage.RequiredComponents {
		if receipt.Coverage.SelectedComponentRowCounts[component] == 0 {
			t.Fatalf("component %q was not covered", component)
		}
	}
	if err := receipt.Validate(); err != nil {
		t.Fatalf("validate r4 receipt: %v", err)
	}
}

func TestAOQTV7R4ActualCoordinateDeterministic(t *testing.T) {
	set := r4ProbeSet(t, 902)
	first := newR4ProbeTrainer(t, set)
	second := newR4ProbeTrainer(t, set)
	firstReceipt, err := first.runActualCoordinateProbe(set, r4SyntheticObjective(t, set, true, true))
	if err != nil {
		t.Fatalf("first r4 probe: %v", err)
	}
	secondReceipt, err := second.runActualCoordinateProbe(set, r4SyntheticObjective(t, set, true, true))
	if err != nil {
		t.Fatalf("second r4 probe: %v", err)
	}
	firstHash, err := firstReceipt.SHA256()
	if err != nil {
		t.Fatal(err)
	}
	secondHash, err := secondReceipt.SHA256()
	if err != nil {
		t.Fatal(err)
	}
	if firstHash != secondHash {
		t.Fatalf("r4 receipt hash changed: %s != %s", firstHash, secondHash)
	}
}

func TestAOQTV7R4ActualCoordinateTamperAndEligibility(t *testing.T) {
	set := r4ProbeSet(t, 903)
	trainer := newR4ProbeTrainer(t, set)
	receipt, err := trainer.runActualCoordinateProbe(set, r4SyntheticObjective(t, set, true, true))
	if err != nil {
		t.Fatalf("eligible r4 probe: %v", err)
	}
	if receipt.EligibleEndpointCount == 0 {
		t.Fatal("eligible objective produced no eligible endpoints")
	}
	tampered := receipt
	tampered.Endpoints = append([]AOQTSidecarActualCoordinateProbeEndpoint(nil), receipt.Endpoints...)
	tampered.Endpoints[0].Magnitude = 0.02
	if err := tampered.Validate(); err == nil {
		t.Fatal("tampered r4 endpoint unexpectedly validated")
	}
}

func TestAOQTV7R4ActualCoordinateNoEligibleAndActivationFailClosed(t *testing.T) {
	set := r4ProbeSet(t, 904)
	noEligible := newR4ProbeTrainer(t, set)
	receipt, err := noEligible.runActualCoordinateProbe(set, r4SyntheticObjective(t, set, false, true))
	if err != nil {
		t.Fatalf("no-eligible r4 probe: %v", err)
	}
	if !receipt.Passed || receipt.EligibleEndpointCount != 0 {
		t.Fatalf("no-eligible receipt = passed=%t eligible=%d", receipt.Passed, receipt.EligibleEndpointCount)
	}
	if err := receipt.Validate(); err != nil {
		t.Fatalf("validate no-eligible receipt: %v", err)
	}
	activationFail := newR4ProbeTrainer(t, set)
	if _, err := activationFail.runActualCoordinateProbe(set, r4SyntheticObjective(t, set, true, false)); err == nil {
		t.Fatal("inactive r4 objective unexpectedly ran")
	}
}

func TestAOQTV7R4ActualCoordinateCoverageImpossible(t *testing.T) {
	set := r4ProbeSet(t, 905)
	rows := append([]AOQTSidecarCalibrationRow(nil), set.Rows...)
	for i := range rows {
		rows[i].Weights.NFBoundaryGuard = 0.75
		for j := range rows[i].CandidateSources {
			rows[i].CandidateSources[j] = "broad_anchor"
		}
	}
	_, _, err := selectAOQTActualCoordinateRows(rows, set.Manifest.ObjectiveContract, hex64("r4-selection"))
	if err == nil {
		t.Fatal("coverage-impossible r4 selection unexpectedly succeeded")
	}
}

func TestAOQTV7R4ActualCoordinateRunnerWritesOnlyReceipt(t *testing.T) {
	materializerCfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(materializerCfg); err != nil {
		t.Fatalf("materialize fixture: %v", err)
	}
	trainCfg := aoqtTrainRunnerConfigFromMaterializerFixture(t, materializerCfg)
	splitManifestPath := writeV7DevSplitManifestForTrainRunnerTest(t, trainCfg, "fold-0", nil)
	receiptPath := filepath.Join(t.TempDir(), "v7-r4-probe.receipt.json")
	trainCfg.PlanOnly = false
	trainCfg.MaxSteps = 99
	trainCfg.LearningRate = 0.01
	trainCfg.OptimizerMode = AOQTSidecarOptimizerModeV7R4DevActualCoordinate
	trainCfg.RequireActualCoordinateProbe = true
	trainCfg.ActualCoordinateProbeOnly = true
	trainCfg.ActualCoordinateProbeReceiptJSONPath = receiptPath
	trainCfg.AllowDevOnly = true
	trainCfg.DevOutputDir = ""
	trainCfg.SplitManifestPath = splitManifestPath
	trainCfg.ExpectedSplitManifestSHA256 = mustSHA256FileAOQTTest(t, splitManifestPath)
	trainCfg.FoldID = "fold-0"
	result, err := runAOQTSidecarTraining(trainCfg, aoqtRunnerObjectiveFromContract)
	if err != nil || result.ActualCoordinateProbe == nil || !result.ActualCoordinateProbe.Passed || result.ActualCoordinateProbe.EndpointCount != AOQTV7R4ActualCoordinateProbeEndpointCount {
		t.Fatalf("r4 runner result=%+v err=%v, want completed probe receipt", result, err)
	}
	data, readErr := os.ReadFile(receiptPath)
	if readErr != nil {
		t.Fatalf("read r4 receipt: %v", readErr)
	}
	var receipt AOQTSidecarActualCoordinateProbeReceipt
	if err := strictUnmarshalAOQT(data, &receipt); err != nil {
		t.Fatalf("decode r4 receipt: %v", err)
	}
	if err := receipt.ValidateBound(); err != nil {
		t.Fatalf("validate bound r4 receipt: %v", err)
	}

	// Bound validation must not accept a receipt whose internal hashes are
	// updated after changing outcome evidence. Keep all tamper variants rooted
	// in this one generated receipt so the test exercises the authoritative
	// reread/replay path without running another probe.
	cloneReceipt := func() AOQTSidecarActualCoordinateProbeReceipt {
		out := receipt
		out.Endpoints = append([]AOQTSidecarActualCoordinateProbeEndpoint(nil), receipt.Endpoints...)
		for i := range out.Endpoints {
			out.Endpoints[i].Indices = append([]int(nil), receipt.Endpoints[i].Indices...)
			out.Endpoints[i].Directions = append([]int(nil), receipt.Endpoints[i].Directions...)
			out.Endpoints[i].ActualDirection = append([]float32(nil), receipt.Endpoints[i].ActualDirection...)
		}
		out.Coverage.RequiredComponents = append([]string(nil), receipt.Coverage.RequiredComponents...)
		out.Coverage.StaticComponentRowCounts = cloneAOQTIntMap(receipt.Coverage.StaticComponentRowCounts)
		out.Coverage.SelectedComponentRowCounts = cloneAOQTIntMap(receipt.Coverage.SelectedComponentRowCounts)
		out.Coverage.SelectedDatasetRowCounts = cloneAOQTIntMap(receipt.Coverage.SelectedDatasetRowCounts)
		out.SelectedEndpointOrdinals = append([]int(nil), receipt.SelectedEndpointOrdinals...)
		return out
	}
	rechain := func(out *AOQTSidecarActualCoordinateProbeReceipt) {
		chain := make([]string, 0, len(out.Endpoints))
		previous := ""
		for _, endpoint := range out.Endpoints {
			payload, err := json.Marshal(endpoint)
			if err != nil {
				t.Fatalf("marshal tampered endpoint: %v", err)
			}
			previous = sha256AOQTCoordinateSearchChainEntry(previous, string(payload))
			chain = append(chain, previous)
		}
		data, err := json.Marshal(chain)
		if err != nil {
			t.Fatalf("marshal tampered chain: %v", err)
		}
		out.EndpointHashChain = string(data)
		out.EndpointHashChainTailSHA256 = previous
	}
	tests := []struct {
		name   string
		mutate func(*AOQTSidecarActualCoordinateProbeReceipt)
	}{
		{
			name: "candidate components and loss",
			mutate: func(out *AOQTSidecarActualCoordinateProbeReceipt) {
				endpoint := &out.Endpoints[0]
				delta := float32(1e-6)
				endpoint.CandidateComponents.Q3ScoreDistill += delta
				endpoint.CandidateLoss += delta
				endpoint.ComponentDeltas.Q3ScoreDistill += delta
				endpoint.TotalDelta += delta
				endpoint.LossDelta = endpoint.TotalDelta
			},
		},
		{
			name: "eligibility reason",
			mutate: func(out *AOQTSidecarActualCoordinateProbeReceipt) {
				out.Endpoints[0].Reason = "tampered_reason"
			},
		},
		{
			name: "candidate error",
			mutate: func(out *AOQTSidecarActualCoordinateProbeReceipt) {
				out.Endpoints[0].CandidateError = "tampered_candidate_error"
			},
		},
		{
			name: "projected direction",
			mutate: func(out *AOQTSidecarActualCoordinateProbeReceipt) {
				endpoint := &out.Endpoints[0]
				endpoint.ActualDirection[0] += 1e-6
				endpoint.ActualDirectionMaxAbs = aoqtMaxAbsFloat32(endpoint.ActualDirection)
				endpoint.ActualDirectionSHA256, _ = aoqtActualCoordinateVectorSHA256(endpoint.ActualDirection)
			},
		},
		{
			name: "coverage counts",
			mutate: func(out *AOQTSidecarActualCoordinateProbeReceipt) {
				out.Coverage.StaticComponentRowCounts["q3_gain"]++
				out.CoverageSHA256, _ = aoqtActualCoordinateCoverageSHA256(out.Coverage)
				out.Binding.CoverageSHA256 = out.CoverageSHA256
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tampered := cloneReceipt()
			tc.mutate(&tampered)
			if tc.name != "coverage counts" {
				rechain(&tampered)
			}
			if err := tampered.ValidateBound(); err == nil {
				t.Fatalf("tampered bound receipt unexpectedly validated")
			}
		})
	}
	if !receipt.Passed || result.Metrics.Schema != "" || result.PackageResult != nil || result.DevEvidence != nil {
		t.Fatalf("r4 runner emitted unexpected artifacts/claim: receipt=%+v result=%+v", receipt, result)
	}
	for _, path := range []string{trainCfg.MetricsJSONPath, AOQTDevFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath), AOQTFailClosedDiagnosticsPath(trainCfg.MetricsJSONPath), filepath.Join(trainCfg.DevOutputDir, "aoqt-sidecar-dev-evidence.json")} {
		if path == "" || path == ".dev.failclosed.json" || path == ".failclosed.json" {
			continue
		}
		if _, statErr := os.Lstat(path); !os.IsNotExist(statErr) {
			t.Fatalf("r4 runner wrote forbidden artifact %q: %v", path, statErr)
		}
	}
}
