package eosruntime

import (
	"strings"
	"testing"
)

func TestAOQTV7R5SortSelectedEndpoints(t *testing.T) {
	receipt := AOQTSidecarActualCoordinateProbeReceipt{
		Endpoints: []AOQTSidecarActualCoordinateProbeEndpoint{
			{Ordinal: 4, TransactionEligible: true, TotalDelta: -0.5, ComponentDeltas: AOQTSidecarObjectiveComponents{Q3Gain: 0.2}},
			{Ordinal: 2, TransactionEligible: true, TotalDelta: -0.5, ComponentDeltas: AOQTSidecarObjectiveComponents{Q3Gain: 0.1}},
			{Ordinal: 3, TransactionEligible: true, TotalDelta: -0.5, ComponentDeltas: AOQTSidecarObjectiveComponents{Q3Gain: 0.1}},
			{Ordinal: 1, TransactionEligible: false, TotalDelta: -10},
		},
	}
	selected := aoqtV7R5SortSelectedEndpoints(receipt)
	if len(selected) != 2 {
		t.Fatalf("selected %d endpoints, want 2", len(selected))
	}
	if selected[0].Ordinal != 2 || selected[1].Ordinal != 3 {
		t.Fatalf("selected ordinals = %v, want [2 3]", []int{selected[0].Ordinal, selected[1].Ordinal})
	}
}

func TestAOQTV7R5FailedPreconditionReceiptValidates(t *testing.T) {
	const identity = "0000000000000000000000000000000000000000000000000000000000000000"
	const angles = "0000000000000000000000000000000000000000000000000000000000000000"
	receipt := AOQTSidecarActualCoordinateTrainReceipt{
		Schema:                   AOQTSidecarActualCoordinateTrainSchema,
		Required:                 true,
		FailureReason:            "precondition: canonical r4 replay unavailable",
		MaxSteps:                 AOQTV7R5ActualCoordinateTrainMaxSteps,
		FullRowCount:             AOQTV7R5ActualCoordinateTrainFullRowCount,
		IdentityStateSHA256:      identity,
		InitialState:             AOQTSidecarActualCoordinateTrainState{StateSHA256: identity, AnglesSHA256: angles, AngleCount: 1},
		FinalState:               AOQTSidecarActualCoordinateTrainState{StateSHA256: identity, AnglesSHA256: angles, AngleCount: 1},
		CanonicalR4ReceiptSHA256: AOQTV7R5CanonicalR4ReceiptSHA256,
		StepHashChain:            "[]",
		SearchStrategy:           AOQTV7R5ActualCoordinateTrainSearchStrategy,
		SelectionVersion:         AOQTV7R5ActualCoordinateTrainSelectionVersion,
	}
	if err := receipt.Validate(); err != nil {
		t.Fatalf("failed precondition receipt should validate: %v", err)
	}
}

func TestAOQTV7R5ReceiptRequiresFullCalibrationSet(t *testing.T) {
	receipt := AOQTSidecarActualCoordinateTrainReceipt{
		Schema:                   AOQTSidecarActualCoordinateTrainSchema,
		Required:                 true,
		MaxSteps:                 AOQTV7R5ActualCoordinateTrainMaxSteps,
		FullRowCount:             AOQTV7R5ActualCoordinateTrainFullRowCount - 1,
		CanonicalR4ReceiptSHA256: AOQTV7R5CanonicalR4ReceiptSHA256,
		SearchStrategy:           AOQTV7R5ActualCoordinateTrainSearchStrategy,
		SelectionVersion:         AOQTV7R5ActualCoordinateTrainSelectionVersion,
	}
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "full row/search accounting") {
		t.Fatalf("Validate() error = %v, want full-row-count rejection", err)
	}
}

func TestAOQTV7R5RunnerModePinsCanonicalReceipt(t *testing.T) {
	cfg := AOQTSidecarTrainRunnerConfig{
		OptimizerMode:                      AOQTSidecarOptimizerModeV7R5DevActualCoordinateTrain,
		AllowDevOnly:                       true,
		DevOutputDir:                       "/tmp/aoqt-r5-test",
		MaxSteps:                           AOQTV7R5ActualCoordinateTrainMaxSteps,
		LearningRate:                       0.01,
		SplitManifestPath:                  "/tmp/split.json",
		ExpectedSplitManifestSHA256:        "0000000000000000000000000000000000000000000000000000000000000000",
		FoldID:                             "fold0",
		ActualCoordinateTrainR4ReceiptPath: "/tmp/canonical-r4.json",
		ExpectedCanonicalR4ReceiptSHA256:   "bad-pin",
	}
	if err := validateAOQTTrainRunnerDevMode(cfg); err == nil || !strings.Contains(err.Error(), "canonical V7-r4 receipt sha256 pin") {
		t.Fatalf("validateAOQTTrainRunnerDevMode() error = %v, want pinned receipt rejection", err)
	}
}

func TestAOQTV7R5NestedR4BindingRejectsCoordinatedForgery(t *testing.T) {
	const (
		manifestPath  = "/authorized/manifest.json"
		rowsPath      = "/authorized/rows.jsonl"
		preflightPath = "/authorized/preflight.json"
	)
	inputs := AOQTSidecarRunMetricInputs{
		AnchorArtifactSHA256:        hex64("anchor"),
		AnchorPackageManifestSHA256: hex64("package"),
		AnchorEmbeddingSpaceID:      "eos-d384-anchor",
		DatasetManifestSHA256:       hex64("manifest"),
		QrelsSHA256ByDataset:        map[string]string{"toy": hex64("qrels")},
		CompatibilityDigest:         hex64("compatibility"),
	}
	ioReport := AOQTSidecarCalibrationIOReport{
		Schema:                      AOQTSidecarCalibrationIOReportSchema,
		ManifestPath:                manifestPath,
		RowsJSONLPath:               rowsPath,
		ManifestSHA256:              inputs.DatasetManifestSHA256,
		RowsSHA256:                  hex64("rows"),
		AnchorArtifactSHA256:        inputs.AnchorArtifactSHA256,
		AnchorPackageManifestSHA256: inputs.AnchorPackageManifestSHA256,
		AnchorEmbeddingSpaceID:      inputs.AnchorEmbeddingSpaceID,
		CompatibilityDigest:         inputs.CompatibilityDigest,
		TurboQuantSeed:              17,
		Topology:                    AOQTSidecarTopologyBinding{Kind: "givens", Dim: 384, Stages: 4, PairsPerStage: 192, AngleCount: 768, Seed: 19, PairingsSHA256: hex64("pairings")},
		ObjectiveContract:           AOQTSidecarObjectiveContract{Dim: 384, TurboQuantSeed: 17, ScoreSurface: "prepared"},
		LegalGates:                  AOQTSidecarLegalGates{ResearchTrainAllowed: true},
		RowCount:                    10942,
		CandidateCount:              123,
		PairCount:                   456,
		TrainingContract:            "aoqt-trainonly-score-distill-joint-budget-v1",
		CandidateEligibilityPolicy:  &AOQTSidecarCandidateEligibilityPolicy{DenseMaxAbsDeltaTolerance: 1e-6},
	}
	split := &AOQTSidecarDevSplitBinding{Schema: "split"}
	outer := AOQTSidecarActualCoordinateTrainBinding{
		Inputs: inputs, IOReport: ioReport, PreflightPath: preflightPath, PreflightSHA256: hex64("preflight"), SplitBinding: split,
		ObjectiveContractSHA256: hex64("objective"), EligibilityPolicySHA256: hex64("eligibility"), ScheduleSHA256: hex64("schedule"), RankingSHA256: hex64("ranking"), CoverageSHA256: hex64("coverage"),
	}
	base := AOQTSidecarActualCoordinateProbeReceipt{
		Schema: AOQTSidecarActualCoordinateProbeSchema, Required: true, Passed: false, FailureReason: "precondition: no search evidence",
		ObjectiveContractSHA256: outer.ObjectiveContractSHA256, EligibilityPolicySHA256: outer.EligibilityPolicySHA256,
		ScheduleSHA256: outer.ScheduleSHA256, RankingSHA256: outer.RankingSHA256, CoverageSHA256: outer.CoverageSHA256,
		Binding: AOQTSidecarActualCoordinateProbeBinding{
			Inputs: inputs, IOReport: ioReport, PreflightPath: preflightPath, PreflightSHA256: outer.PreflightSHA256, SplitBinding: split,
			ObjectiveContractSHA256: outer.ObjectiveContractSHA256, EligibilityPolicySHA256: outer.EligibilityPolicySHA256,
			ScheduleSHA256: outer.ScheduleSHA256, RankingSHA256: outer.RankingSHA256, CoverageSHA256: outer.CoverageSHA256,
		},
	}
	if err := validateAOQTV7R5SearchBinding(base, outer); err != nil {
		t.Fatalf("authoritative nested binding rejected: %v", err)
	}
	for _, mutate := range []struct {
		name   string
		mutate func(*AOQTSidecarActualCoordinateProbeReceipt)
	}{
		{"io-count", func(r *AOQTSidecarActualCoordinateProbeReceipt) { r.Binding.IOReport.RowCount++ }},
		{"io-topology", func(r *AOQTSidecarActualCoordinateProbeReceipt) { r.Binding.IOReport.Topology.Dim++ }},
		{"io-legal-gate", func(r *AOQTSidecarActualCoordinateProbeReceipt) {
			r.Binding.IOReport.LegalGates.ReleaseTrainAllowed = true
		}},
		{"io-objective", func(r *AOQTSidecarActualCoordinateProbeReceipt) { r.Binding.IOReport.ObjectiveContract.GainBit++ }},
		{"io-training-policy", func(r *AOQTSidecarActualCoordinateProbeReceipt) { r.Binding.IOReport.TrainingContract = "forged" }},
		{"coverage-rehash", func(r *AOQTSidecarActualCoordinateProbeReceipt) {
			r.CoverageSHA256 = hex64("forged-coverage")
			r.Binding.CoverageSHA256 = r.CoverageSHA256
		}},
		{"ranking-rehash", func(r *AOQTSidecarActualCoordinateProbeReceipt) {
			r.RankingSHA256 = hex64("forged-ranking")
			r.Binding.RankingSHA256 = r.RankingSHA256
		}},
	} {
		t.Run(mutate.name, func(t *testing.T) {
			forged := base
			forged.Binding = base.Binding
			forged.Binding.IOReport = base.Binding.IOReport
			if base.Binding.IOReport.CandidateEligibilityPolicy != nil {
				policy := *base.Binding.IOReport.CandidateEligibilityPolicy
				forged.Binding.IOReport.CandidateEligibilityPolicy = &policy
			}
			mutate.mutate(&forged)
			if err := validateAOQTV7R5SearchBinding(forged, outer); err == nil {
				t.Fatal("coordinated nested forgery unexpectedly validated")
			}
		})
	}
}

func TestAOQTV7R5ObjectiveFailureInitializesReceiptAndDiagnostics(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 906)
	canonical := &AOQTSidecarActualCoordinateProbeReceipt{}
	trainer, err := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{
		PairingSeed:                              906,
		WorkplanSeed:                             907,
		OptimizerMode:                            AOQTSidecarOptimizerModeV7R5DevActualCoordinateTrain,
		ActualCoordinateTrainRequired:            true,
		ActualCoordinateTrainR4Receipt:           canonical,
		ActualCoordinateTrainR4ReceiptPath:       "/authorized/canonical-r4.json",
		ActualCoordinateTrainR4ReceiptSHA256:     AOQTV7R5CanonicalR4ReceiptSHA256,
		ActualCoordinateProbeFoldID:              "fold-0",
		ActualCoordinateProbeSplitManifestSHA256: hex64("split"),
		MaxSteps:                                 AOQTV7R5ActualCoordinateTrainMaxSteps,
		LearningRate:                             0.01,
		TrainingContract:                         set.Manifest.TrainingContract,
		CandidateEligibilityPolicy:               cloneAOQTSidecarCandidateEligibilityPolicy(set.Manifest.CandidateEligibilityPolicy),
	})
	if err != nil {
		t.Fatalf("new r5 trainer: %v", err)
	}
	summary, fitErr := trainer.Fit(set, nil)
	if fitErr == nil {
		t.Fatal("nil objective unexpectedly succeeded")
	}
	if summary.ActualCoordinateTrain == nil || summary.OptimizerDiagnostics == nil {
		t.Fatalf("failed r5 fit omitted receipt/diagnostics: receipt=%v diagnostics=%v", summary.ActualCoordinateTrain != nil, summary.OptimizerDiagnostics != nil)
	}
	if err := summary.ActualCoordinateTrain.Validate(); err != nil {
		t.Fatalf("failed r5 receipt is invalid: %v", err)
	}
	if err := validateAOQTV7R5OptimizerDiagnostics(summary.Plan, summary, *summary.OptimizerDiagnostics); err != nil {
		t.Fatalf("failed r5 diagnostics are invalid: %v", err)
	}
	if summary.ActualCoordinateTrain.AttemptedSteps != 0 || summary.ActualCoordinateTrain.AcceptedSteps != 0 || summary.ActualCoordinateTrain.SearchCount != 0 || summary.ActualCoordinateTrain.FullCandidateEvaluationCount != 0 {
		t.Fatalf("nil-objective failure reported work: %+v", summary.ActualCoordinateTrain)
	}
}

func TestAOQTV7R5CanonicalPreconditionFailureInitializesReceipt(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 908)
	canonical := &AOQTSidecarActualCoordinateProbeReceipt{
		Schema: AOQTSidecarActualCoordinateProbeSchema, Required: true,
		FailureReason: "precondition: canonical r4 was not passed",
	}
	trainer, err := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{
		PairingSeed:                              908,
		WorkplanSeed:                             909,
		OptimizerMode:                            AOQTSidecarOptimizerModeV7R5DevActualCoordinateTrain,
		ActualCoordinateTrainRequired:            true,
		ActualCoordinateTrainR4Receipt:           canonical,
		ActualCoordinateTrainR4ReceiptPath:       "/authorized/canonical-r4.json",
		ActualCoordinateTrainR4ReceiptSHA256:     AOQTV7R5CanonicalR4ReceiptSHA256,
		ActualCoordinateProbeFoldID:              "fold-0",
		ActualCoordinateProbeSplitManifestSHA256: hex64("split-canonical-failure"),
		MaxSteps:                                 AOQTV7R5ActualCoordinateTrainMaxSteps,
		LearningRate:                             0.01,
		TrainingContract:                         set.Manifest.TrainingContract,
		CandidateEligibilityPolicy:               cloneAOQTSidecarCandidateEligibilityPolicy(set.Manifest.CandidateEligibilityPolicy),
	})
	if err != nil {
		t.Fatalf("new r5 trainer: %v", err)
	}
	summary, fitErr := trainer.Fit(set, r4SyntheticObjective(t, set, true, true))
	if fitErr == nil {
		t.Fatal("failed canonical receipt unexpectedly allowed training")
	}
	if summary.ActualCoordinateTrain == nil || summary.OptimizerDiagnostics == nil {
		t.Fatalf("canonical precondition failure omitted receipt/diagnostics")
	}
	if err := summary.ActualCoordinateTrain.Validate(); err != nil {
		t.Fatalf("canonical precondition receipt is invalid: %v", err)
	}
	if err := validateAOQTV7R5OptimizerDiagnostics(summary.Plan, summary, *summary.OptimizerDiagnostics); err != nil {
		t.Fatalf("canonical precondition diagnostics are invalid: %v", err)
	}
	if summary.ActualCoordinateTrain.AttemptedSteps != 0 || summary.ActualCoordinateTrain.AcceptedSteps != 0 || summary.ActualCoordinateTrain.SearchCount != 0 {
		t.Fatalf("canonical precondition reported work: %+v", summary.ActualCoordinateTrain)
	}
}
