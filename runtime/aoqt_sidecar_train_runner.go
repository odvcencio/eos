package eosruntime

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"sort"
	"strings"
)

type AOQTSidecarTrainRunnerConfig struct {
	ManifestPath                        string
	RowsJSONLPath                       string
	PreflightJSONPath                   string
	MetricsJSONPath                     string
	OutputArtifactPath                  string
	ExpectedManifestSHA256              string
	ExpectedRowsSHA256                  string
	ExpectedPreflightSHA256             string
	ExpectedAnchorArtifactSHA256        string
	ExpectedAnchorPackageManifestSHA256 string
	ExpectedAnchorEmbeddingSpaceID      string
	ExpectedCompatibilityDigest         string
	ExpectedSourceArtifactHashes        []string
	ExpectedVectorCacheHashes           []string
	PlanOnly                            bool
	AllowResearchOnly                   bool
	MaxSteps                            int
	LearningRate                        float32
}

type AOQTSidecarTrainRunnerResult struct {
	IOReport           AOQTSidecarCalibrationIOReport
	Preflight          AOQTSidecarMaterializePreflight
	Metrics            AOQTSidecarRunMetrics
	PackageResult      *AOQTSidecarCandidatePackageResult
	FailureDiagnostics *AOQTSidecarFailClosedDiagnostics
}

type aoqtRunnerObjectiveFactory func(AOQTSidecarObjectiveContract) (AOQTSidecarVectorObjective, error)

// v2 binds fail-closed optimizer evidence to the protected-cone coordinate
// audit contract. Older v1 diagnostics are intentionally not relabeled.
const AOQTSidecarFailClosedDiagnosticsSchema = "eos.q3_aoqt_sidecar_failclosed_diagnostics.v2"

type AOQTSidecarFailClosedDiagnostics struct {
	Schema            string                          `json:"schema"`
	FailureKind       string                          `json:"failure_kind"`
	Error             string                          `json:"error"`
	RejectionSummary  string                          `json:"rejection_summary"`
	Plan              AOQTSidecarWorkPlan             `json:"plan"`
	Inputs            AOQTSidecarRunMetricInputs      `json:"inputs"`
	IOReport          AOQTSidecarCalibrationIOReport  `json:"io_report"`
	Preflight         AOQTSidecarMaterializePreflight `json:"preflight"`
	PreflightPath     string                          `json:"preflight_path"`
	PreflightSHA256   string                          `json:"preflight_sha256"`
	Topology          AOQTSidecarTopologyBinding      `json:"topology"`
	ObjectiveContract AOQTSidecarObjectiveContract    `json:"objective_contract"`
	LegalGates        AOQTSidecarLegalGates           `json:"legal_gates"`
	Summary           AOQTSidecarTrainSummary         `json:"summary"`
	QualityClaim      bool                            `json:"quality_claim"`
}

func RunAOQTSidecarTraining(cfg AOQTSidecarTrainRunnerConfig) (AOQTSidecarTrainRunnerResult, error) {
	return runAOQTSidecarTraining(cfg, aoqtRunnerObjectiveFromContract)
}

func runAOQTSidecarTraining(cfg AOQTSidecarTrainRunnerConfig, objectiveFactory aoqtRunnerObjectiveFactory) (AOQTSidecarTrainRunnerResult, error) {
	if err := validateAOQTSidecarTrainRunnerConfig(cfg); err != nil {
		return AOQTSidecarTrainRunnerResult{}, err
	}
	topology, err := expectedAOQTSidecarTrainTopology()
	if err != nil {
		return AOQTSidecarTrainRunnerResult{}, err
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
		ExpectedSourceArtifactHashes:        append([]string(nil), cfg.ExpectedSourceArtifactHashes...),
		ExpectedVectorCacheHashes:           append([]string(nil), cfg.ExpectedVectorCacheHashes...),
	})
	if err != nil {
		return AOQTSidecarTrainRunnerResult{}, err
	}
	preflight, preflightSHA256, err := loadAndValidateAOQTTrainPreflight(cfg, set, ioReport)
	if err != nil {
		return AOQTSidecarTrainRunnerResult{}, err
	}
	trainer, err := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{
		PairingSeed:  AOQTSidecarMaterializerTopologySeed,
		WorkplanSeed: AOQTSidecarMaterializerQuantSeed,
		PlanOnly:     cfg.PlanOnly,
		MaxSteps:     cfg.MaxSteps,
		LearningRate: cfg.LearningRate,
		AngleCap:     AOQTSidecarDefaultAngleCap,
		MaxAngleCap:  AOQTSidecarHardMaxAngleCap,
	})
	if err != nil {
		return AOQTSidecarTrainRunnerResult{}, err
	}
	var objective AOQTSidecarVectorObjective
	if !cfg.PlanOnly {
		if objectiveFactory == nil {
			return AOQTSidecarTrainRunnerResult{}, fmt.Errorf("AOQT objective factory is required for non-plan training")
		}
		objective, err = objectiveFactory(set.Manifest.ObjectiveContract)
		if err != nil {
			return AOQTSidecarTrainRunnerResult{}, err
		}
	}
	summary, err := trainer.Fit(set, objective)
	if err != nil {
		result := AOQTSidecarTrainRunnerResult{IOReport: ioReport, Preflight: preflight}
		if !cfg.PlanOnly {
			diagnostics, diagnosticsErr := newAOQTFailClosedDiagnostics(set, ioReport, preflight, cfg.PreflightJSONPath, preflightSHA256, summary, err)
			if diagnosticsErr == nil {
				result.FailureDiagnostics = &diagnostics
				if writeErr := writeAOQTFailClosedDiagnosticsFile(AOQTFailClosedDiagnosticsPath(cfg.MetricsJSONPath), diagnostics); writeErr != nil {
					return result, fmt.Errorf("%w; failed to write AOQT fail-closed diagnostics: %v", err, writeErr)
				}
			}
		}
		return result, err
	}
	metrics, err := NewAOQTSidecarRunMetrics(set, summary)
	if err != nil {
		return AOQTSidecarTrainRunnerResult{}, err
	}
	if !cfg.PlanOnly {
		if err := ValidateAOQTSidecarCandidateEligibility(metrics, AOQTSidecarCandidateEligibilityPolicy{RequireObjectiveActivation: true}); err != nil {
			return AOQTSidecarTrainRunnerResult{}, err
		}
	}
	if err := writeAOQTRunMetricsFile(cfg.MetricsJSONPath, metrics); err != nil {
		return AOQTSidecarTrainRunnerResult{}, err
	}
	result := AOQTSidecarTrainRunnerResult{IOReport: ioReport, Preflight: preflight, Metrics: metrics}
	if !cfg.PlanOnly {
		packageResult, err := WriteAOQTSidecarCandidatePackage(AOQTSidecarCandidatePackageConfig{
			AnchorArtifactPath:                  set.Manifest.AnchorArtifactPath,
			OutputArtifactPath:                  cfg.OutputArtifactPath,
			ExpectedAnchorArtifactSHA256:        set.Manifest.AnchorArtifactSHA256,
			ExpectedAnchorPackageManifestSHA256: set.Manifest.AnchorPackageManifestSHA256,
			AnchorEmbeddingSpaceID:              set.Manifest.AnchorEmbeddingSpaceID,
			DatasetManifestSHA256:               metrics.Inputs.DatasetManifestSHA256,
			QrelsSHA256ByDataset:                cloneAOQTStringMap(set.Manifest.QrelsSHA256ByDataset),
			CompatibilityDigest:                 set.Manifest.CompatibilityDigest,
			Transform:                           trainer.Transform(),
		})
		if err != nil {
			return AOQTSidecarTrainRunnerResult{}, err
		}
		result.PackageResult = &packageResult
	}
	return result, nil
}

func aoqtRunnerObjectiveFromContract(contract AOQTSidecarObjectiveContract) (AOQTSidecarVectorObjective, error) {
	objective, err := objectiveFromAOQTContract(contract)
	if err != nil {
		return nil, err
	}
	return objective, nil
}

func validateAOQTSidecarTrainRunnerConfig(cfg AOQTSidecarTrainRunnerConfig) error {
	if !cfg.AllowResearchOnly {
		return fmt.Errorf("AOQT sidecar training requires explicit research-only train authorization")
	}
	for _, item := range []struct {
		name  string
		value string
	}{
		{"manifest", cfg.ManifestPath},
		{"rows", cfg.RowsJSONLPath},
		{"preflight", cfg.PreflightJSONPath},
		{"metrics-json", cfg.MetricsJSONPath},
		{"expected-manifest-sha256", cfg.ExpectedManifestSHA256},
		{"expected-rows-sha256", cfg.ExpectedRowsSHA256},
		{"expected-preflight-sha256", cfg.ExpectedPreflightSHA256},
		{"expected-anchor-artifact-sha256", cfg.ExpectedAnchorArtifactSHA256},
		{"expected-anchor-package-manifest-sha256", cfg.ExpectedAnchorPackageManifestSHA256},
		{"anchor-embedding-space-id", cfg.ExpectedAnchorEmbeddingSpaceID},
		{"compatibility-digest", cfg.ExpectedCompatibilityDigest},
	} {
		if strings.TrimSpace(item.value) == "" {
			return fmt.Errorf("AOQT sidecar training requires --%s", item.name)
		}
	}
	if err := ensureAOQTFailClosedDiagnosticsAbsent(cfg.MetricsJSONPath); err != nil {
		return err
	}
	if cfg.PlanOnly {
		if strings.TrimSpace(cfg.OutputArtifactPath) != "" {
			return fmt.Errorf("AOQT sidecar plan-only writes metrics only; --output is not allowed")
		}
		if cfg.MaxSteps < 0 {
			return fmt.Errorf("AOQT sidecar max steps must be non-negative")
		}
	} else {
		if strings.TrimSpace(cfg.OutputArtifactPath) == "" {
			return fmt.Errorf("AOQT sidecar non-plan training requires --output")
		}
		if cfg.MaxSteps <= 0 {
			return fmt.Errorf("AOQT sidecar non-plan training requires positive --max-steps")
		}
		if err := validateAOQTTrainRunnerOutputPathDistinctness(cfg); err != nil {
			return err
		}
		if err := ensureAOQTOutputArtifactAbsent(cfg.OutputArtifactPath); err != nil {
			return err
		}
	}
	return nil
}

func expectedAOQTSidecarTrainTopology() (AOQTSidecarTopologyBinding, error) {
	transform, err := NewAOQTGivensIdentityTopology(AOQTSidecarDim, AOQTSidecarStages, AOQTSidecarMaterializerTopologySeed, AOQTSidecarDefaultAngleCap)
	if err != nil {
		return AOQTSidecarTopologyBinding{}, err
	}
	pairings, err := transform.PairingsSHA256()
	if err != nil {
		return AOQTSidecarTopologyBinding{}, err
	}
	return AOQTSidecarTopologyBinding{
		Kind:           AOQTTopologyKindGivensV1,
		Dim:            AOQTSidecarDim,
		Stages:         AOQTSidecarStages,
		PairsPerStage:  AOQTSidecarPairsPerStage,
		AngleCount:     AOQTSidecarAngleCount,
		Seed:           AOQTSidecarMaterializerTopologySeed,
		PairingsSHA256: pairings,
	}, nil
}

func loadAndValidateAOQTTrainPreflight(cfg AOQTSidecarTrainRunnerConfig, set AOQTSidecarCalibrationSet, ioReport AOQTSidecarCalibrationIOReport) (AOQTSidecarMaterializePreflight, string, error) {
	if err := validateAOQTSHA256(cfg.ExpectedPreflightSHA256, "AOQT sidecar expected preflight sha256"); err != nil {
		return AOQTSidecarMaterializePreflight{}, "", err
	}
	data, err := os.ReadFile(cfg.PreflightJSONPath)
	if err != nil {
		return AOQTSidecarMaterializePreflight{}, "", err
	}
	preflightSHA256 := sha256BytesAOQT(data)
	if preflightSHA256 != cfg.ExpectedPreflightSHA256 {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("AOQT sidecar preflight sha256 mismatch")
	}
	var preflight AOQTSidecarMaterializePreflight
	if err := strictUnmarshalAOQT(data, &preflight); err != nil {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("%s: invalid AOQT materializer preflight JSON: %w", cfg.PreflightJSONPath, err)
	}
	if preflight.Schema != AOQTSidecarMaterializerPreflightSchema {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("AOQT sidecar preflight schema %q is not supported", preflight.Schema)
	}
	if preflight.QualityClaim || !preflight.ResearchOnly || preflight.ActualTrainingRan || preflight.ActualEvalRan || preflight.ActualVectorExportRan {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("AOQT sidecar preflight must be research-only and record no training/eval/vector export execution")
	}
	if preflight.TurboQuantSeed != AOQTSidecarMaterializerQuantSeed {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("AOQT sidecar preflight turboquant seed = %d, want %d", preflight.TurboQuantSeed, AOQTSidecarMaterializerQuantSeed)
	}
	if preflight.TopologySeed != AOQTSidecarMaterializerTopologySeed {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("AOQT sidecar preflight topology seed = %d, want %d", preflight.TopologySeed, AOQTSidecarMaterializerTopologySeed)
	}
	if preflight.RowCount != len(set.Rows) || preflight.CandidateCount != ioReport.CandidateCount || preflight.PairCount != ioReport.PairCount {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("AOQT sidecar preflight workload counts do not match calibration IO")
	}
	if preflight.CalibrationManifestSHA256 != ioReport.ManifestSHA256 {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("AOQT sidecar preflight calibration manifest sha256 mismatch")
	}
	if !aoqtObjectiveContractsEqual(preflight.ObjectiveContract, set.Manifest.ObjectiveContract) {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("AOQT sidecar preflight objective contract mismatch")
	}
	if preflight.LegalGates != set.Manifest.LegalGates {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("AOQT sidecar preflight legal gates mismatch")
	}
	return preflight, preflightSHA256, nil
}

func objectiveFromAOQTContract(contract AOQTSidecarObjectiveContract) (AOQTSidecarPreparedIPObjective, error) {
	return NewAOQTSidecarPreparedIPObjective(AOQTSidecarPreparedIPObjectiveConfig{
		Dim:              contract.Dim,
		TurboQuantSeed:   contract.TurboQuantSeed,
		GainBit:          contract.GainBit,
		Q3GuardBit:       contract.Q3GuardBit,
		Q5GuardBit:       contract.Q5GuardBit,
		GainCutoff:       contract.GainCutoff,
		GainTau:          contract.GainTau,
		GainMargin:       contract.GainMargin,
		GuardTau:         contract.GuardTau,
		GuardMargin:      contract.GuardMargin,
		ScoreDistillTau:  contract.ScoreDistillTau,
		NFBoundarySource: contract.NFBoundarySource,
	})
}

func validateAOQTTrainRunnerOutputPathDistinctness(cfg AOQTSidecarTrainRunnerConfig) error {
	paths := map[string]string{
		"metrics":                 cfg.MetricsJSONPath,
		"fail-closed diagnostics": AOQTFailClosedDiagnosticsPath(cfg.MetricsJSONPath),
	}
	for role, path := range aoqtCandidatePathMap(aoqtCandidateOutputPaths(cfg.OutputArtifactPath, true)) {
		paths[role] = path
	}
	roles := make([]string, 0, len(paths))
	for role := range paths {
		roles = append(roles, role)
	}
	sort.Strings(roles)
	seen := make(map[string]string, len(paths))
	for _, role := range roles {
		path := paths[role]
		if strings.TrimSpace(path) == "" {
			continue
		}
		canonical, err := filepath.Abs(filepath.Clean(path))
		if err != nil {
			return fmt.Errorf("resolve AOQT sidecar %s output path %q: %w", role, path, err)
		}
		if prior, ok := seen[canonical]; ok {
			return fmt.Errorf("AOQT sidecar output path collision: %s %q conflicts with %s %q", role, path, prior, paths[prior])
		}
		seen[canonical] = role
	}
	return nil
}

// NewAOQTSidecarFailClosedDiagnostics is retained for source compatibility,
// but cannot construct evidence safely without the authentic raw preflight
// file and hash. The training runner is the only supported constructor.
func NewAOQTSidecarFailClosedDiagnostics(set AOQTSidecarCalibrationSet, ioReport AOQTSidecarCalibrationIOReport, summary AOQTSidecarTrainSummary, failure error) (AOQTSidecarFailClosedDiagnostics, error) {
	return AOQTSidecarFailClosedDiagnostics{}, fmt.Errorf("AOQT fail-closed diagnostics require authentic materializer preflight provenance; construct them through the training runner")
}

func newAOQTFailClosedDiagnostics(set AOQTSidecarCalibrationSet, ioReport AOQTSidecarCalibrationIOReport, preflight AOQTSidecarMaterializePreflight, preflightPath, preflightSHA256 string, summary AOQTSidecarTrainSummary, failure error) (AOQTSidecarFailClosedDiagnostics, error) {
	if failure == nil {
		return AOQTSidecarFailClosedDiagnostics{}, fmt.Errorf("AOQT fail-closed diagnostics require a failure")
	}
	if err := set.Validate(); err != nil {
		return AOQTSidecarFailClosedDiagnostics{}, err
	}
	if summary.OptimizerDiagnostics == nil || summary.OptimizerDiagnostics.AcceptedSteps != 0 {
		return AOQTSidecarFailClosedDiagnostics{}, fmt.Errorf("AOQT fail-closed diagnostics require zero accepted optimizer steps")
	}
	manifestHash, err := AOQTSidecarManifestSHA256(set.Manifest)
	if err != nil {
		return AOQTSidecarFailClosedDiagnostics{}, err
	}
	diagnostics := AOQTSidecarFailClosedDiagnostics{
		Schema:           AOQTSidecarFailClosedDiagnosticsSchema,
		FailureKind:      "zero_accepted_safe_steps",
		Error:            failure.Error(),
		RejectionSummary: summary.OptimizerDiagnostics.RejectionSummary(),
		Plan:             summary.Plan,
		Inputs: AOQTSidecarRunMetricInputs{
			AnchorArtifactSHA256:        set.Manifest.AnchorArtifactSHA256,
			AnchorPackageManifestSHA256: set.Manifest.AnchorPackageManifestSHA256,
			AnchorEmbeddingSpaceID:      set.Manifest.AnchorEmbeddingSpaceID,
			DatasetManifestSHA256:       manifestHash,
			QrelsSHA256ByDataset:        cloneAOQTStringMap(set.Manifest.QrelsSHA256ByDataset),
			CompatibilityDigest:         set.Manifest.CompatibilityDigest,
		},
		IOReport:          ioReport,
		Preflight:         preflight,
		PreflightPath:     preflightPath,
		PreflightSHA256:   preflightSHA256,
		Topology:          set.Manifest.Topology,
		ObjectiveContract: set.Manifest.ObjectiveContract,
		LegalGates:        set.Manifest.LegalGates,
		Summary:           summary,
		QualityClaim:      false,
	}
	return diagnostics, diagnostics.Validate()
}

func (d AOQTSidecarFailClosedDiagnostics) Validate() error {
	if d.Schema != AOQTSidecarFailClosedDiagnosticsSchema {
		return fmt.Errorf("AOQT fail-closed diagnostics schema %q is not supported, want %q", d.Schema, AOQTSidecarFailClosedDiagnosticsSchema)
	}
	if d.QualityClaim || d.Summary.QualityClaim {
		return fmt.Errorf("AOQT fail-closed diagnostics quality_claim must be false")
	}
	if d.FailureKind != "zero_accepted_safe_steps" {
		return fmt.Errorf("AOQT fail-closed diagnostics failure_kind %q is not supported", d.FailureKind)
	}
	if strings.TrimSpace(d.Error) == "" {
		return fmt.Errorf("AOQT fail-closed diagnostics error is required")
	}
	if strings.TrimSpace(d.RejectionSummary) == "" {
		return fmt.Errorf("AOQT fail-closed diagnostics rejection_summary is required")
	}
	if strings.TrimSpace(d.PreflightPath) == "" {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight_path is required")
	}
	if err := validateAOQTSHA256(d.PreflightSHA256, "AOQT fail-closed diagnostics preflight_sha256"); err != nil {
		return err
	}
	if err := validateAOQTFailClosedPreflightRawBinding(d.Preflight, d.PreflightPath, d.PreflightSHA256); err != nil {
		return err
	}
	if d.Plan != d.Summary.Plan {
		return fmt.Errorf("AOQT fail-closed diagnostics plan must exactly match summary.plan")
	}
	if d.Plan.PlanOnly {
		return fmt.Errorf("AOQT fail-closed diagnostics require a non-plan optimizer run")
	}
	if d.Summary.Steps != 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics summary.steps = %d, want 0", d.Summary.Steps)
	}
	if d.Plan.RowCount <= 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics plan.row_count must be positive")
	}
	if d.Plan.PairCount <= 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics plan.pair_count must be positive")
	}
	if d.Plan.StepCount <= 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics plan.step_count must be positive")
	}
	if err := d.Topology.Validate(); err != nil {
		return err
	}
	if err := validateAOQTMetricsPlanTopology(d.Plan, d.Topology); err != nil {
		return err
	}
	if !aoqtObjectiveContractsEqual(d.ObjectiveContract, d.Summary.ObjectiveContract) {
		return fmt.Errorf("AOQT fail-closed diagnostics objective_contract must exactly match summary.objective_contract")
	}
	if err := validateAOQTFailClosedObjectiveContract(d.ObjectiveContract, d.Topology); err != nil {
		return err
	}
	if err := validateAOQTResearchOnlyLegalGates(d.LegalGates, "AOQT fail-closed diagnostics"); err != nil {
		return err
	}
	if d.Summary.OptimizerDiagnostics == nil {
		return fmt.Errorf("AOQT fail-closed diagnostics optimizer diagnostics are required")
	}
	optimizer := *d.Summary.OptimizerDiagnostics
	if optimizer.PlannedSteps != d.Plan.StepCount {
		return fmt.Errorf("AOQT fail-closed diagnostics planned_steps = %d, want plan step_count %d", optimizer.PlannedSteps, d.Plan.StepCount)
	}
	if optimizer.AttemptedSteps != 1 {
		return fmt.Errorf("AOQT fail-closed diagnostics attempted_steps = %d, want 1", optimizer.AttemptedSteps)
	}
	if optimizer.AcceptedSteps != 0 || optimizer.AcceptedProposals != 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics must have zero accepted steps and proposals")
	}
	if optimizer.ExhaustedSteps != 1 {
		return fmt.Errorf("AOQT fail-closed diagnostics exhausted_steps = %d, want 1", optimizer.ExhaustedSteps)
	}
	if optimizer.MaxAttemptsPerStep != aoqtTransactionalMaxAttemptsPerStep {
		return fmt.Errorf("AOQT fail-closed diagnostics max_attempts_per_step = %d, want %d", optimizer.MaxAttemptsPerStep, aoqtTransactionalMaxAttemptsPerStep)
	}
	if optimizer.ProposalAttempts != optimizer.RejectedProposals || optimizer.Backtracks != optimizer.RejectedProposals {
		return fmt.Errorf("AOQT fail-closed diagnostics rejected proposal accounting mismatch")
	}
	if optimizer.ProposalAttempts <= 0 || optimizer.ProposalAttempts > optimizer.MaxAttemptsPerStep {
		return fmt.Errorf("AOQT fail-closed diagnostics proposal_attempts = %d outside expected range", optimizer.ProposalAttempts)
	}
	if optimizer.RejectedProposals < 0 || optimizer.Backtracks < 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics rejected proposal counts must be non-negative")
	}
	if optimizer.RejectionDiagnostics.CandidateEvaluations != optimizer.ProposalAttempts {
		return fmt.Errorf("AOQT fail-closed diagnostics candidate_evaluations = %d, want proposal_attempts %d", optimizer.RejectionDiagnostics.CandidateEvaluations, optimizer.ProposalAttempts)
	}
	if err := validateAOQTFailClosedOptimizerPathDiagnostics(d.Plan, d.ObjectiveContract.WeightSums, optimizer); err != nil {
		return err
	}
	if d.RejectionSummary != optimizer.RejectionSummary() {
		return fmt.Errorf("AOQT fail-closed diagnostics rejection_summary does not match optimizer diagnostics")
	}
	if !strings.Contains(d.Error, d.RejectionSummary) {
		return fmt.Errorf("AOQT fail-closed diagnostics error does not contain rejection_summary")
	}
	if strings.TrimSpace(d.Summary.OptimizerDiagnosticsSHA256) == "" {
		return fmt.Errorf("AOQT fail-closed diagnostics optimizer diagnostics sha256 is required")
	}
	if err := validateAOQTSHA256(d.Summary.OptimizerDiagnosticsSHA256, "AOQT fail-closed diagnostics optimizer diagnostics sha256"); err != nil {
		return err
	}
	got, err := optimizer.SHA256()
	if err != nil {
		return err
	}
	if got != d.Summary.OptimizerDiagnosticsSHA256 {
		return fmt.Errorf("AOQT fail-closed diagnostics optimizer diagnostics sha256 mismatch")
	}
	if err := validateAOQTObjectiveComponents(d.Summary.InitialObjectiveComponents, "AOQT fail-closed diagnostics summary.initial_objective_components"); err != nil {
		return err
	}
	if err := validateAOQTLossMatchesComponents(d.Summary.InitialLoss, d.Summary.InitialObjectiveComponents, "AOQT fail-closed diagnostics summary.initial"); err != nil {
		return err
	}
	if err := validateAOQTObjectiveComponents(d.Summary.FinalObjectiveComponents, "AOQT fail-closed diagnostics summary.final_objective_components"); err != nil {
		return err
	}
	if err := validateAOQTLossMatchesComponents(d.Summary.FinalLoss, d.Summary.FinalObjectiveComponents, "AOQT fail-closed diagnostics summary.final"); err != nil {
		return err
	}
	if err := validateAOQTMetricInputs(d.Inputs); err != nil {
		return err
	}
	if err := validateAOQTFailClosedIOReport(d.IOReport, d.Plan, d.Inputs, d.Topology, d.ObjectiveContract, d.LegalGates); err != nil {
		return err
	}
	if err := validateAOQTFailClosedPreflight(d.Preflight, d.PreflightPath, d.IOReport, d.Inputs, d.Topology, d.ObjectiveContract, d.LegalGates); err != nil {
		return err
	}
	return nil
}

func validateAOQTFailClosedPreflightRawBinding(preflight AOQTSidecarMaterializePreflight, path, expectedSHA256 string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight raw file %q: %w", path, err)
	}
	if got := sha256BytesAOQT(raw); got != expectedSHA256 {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight raw sha256 mismatch")
	}
	var loaded AOQTSidecarMaterializePreflight
	if err := strictUnmarshalAOQT(raw, &loaded); err != nil {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight raw file %q is invalid: %w", path, err)
	}
	if !reflect.DeepEqual(loaded, preflight) {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight does not match raw file %q", path)
	}
	return nil
}

func validateAOQTFailClosedOptimizerPathDiagnostics(plan AOQTSidecarWorkPlan, weights AOQTSidecarRowWeights, diagnostics AOQTSidecarOptimizerDiagnostics) error {
	for _, item := range []struct {
		name  string
		value int
	}{
		{"adam_proposal_attempts", diagnostics.AdamProposalAttempts},
		{"adam_accepted_proposals", diagnostics.AdamAcceptedProposals},
		{"adam_rejected_proposals", diagnostics.AdamRejectedProposals},
		{"coordinate_proposal_attempts", diagnostics.CoordinateProposalAttempts},
		{"coordinate_accepted_proposals", diagnostics.CoordinateAcceptedProposals},
		{"coordinate_rejected_proposals", diagnostics.CoordinateRejectedProposals},
	} {
		if item.value < 0 {
			return fmt.Errorf("AOQT fail-closed diagnostics %s must be non-negative", item.name)
		}
	}
	hasPathDiagnostics := diagnostics.AdamProposalAttempts != 0 ||
		diagnostics.AdamAcceptedProposals != 0 ||
		diagnostics.AdamRejectedProposals != 0 ||
		diagnostics.CoordinateProposalAttempts != 0 ||
		diagnostics.CoordinateAcceptedProposals != 0 ||
		diagnostics.CoordinateRejectedProposals != 0
	if !hasPathDiagnostics {
		if diagnostics.CoordinateSearchPlanCount != 0 || diagnostics.CoordinateSearchStrategy != "" || diagnostics.CoordinateSearchOrderingHash != "" || diagnostics.CoordinateSearchLearningRate != 0 || diagnostics.CoordinateSearchAudit != "" || diagnostics.CoordinateSearchAuditChain != "" || diagnostics.CoordinateSearchHashChain != "" || diagnostics.CoordinateTopAngles != 0 || diagnostics.CoordinateMagnitudeCount != 0 || diagnostics.CoordinateBlockCount != 0 {
			return fmt.Errorf("AOQT fail-closed diagnostics coordinate audit is present without optimizer path diagnostics")
		}
		return nil
	}
	if diagnostics.CoordinateSearchPlanCount < 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics coordinate_search_plan_count must be non-negative")
	}
	if diagnostics.CoordinateSearchPlanCount > diagnostics.AttemptedSteps {
		return fmt.Errorf("AOQT fail-closed diagnostics coordinate_search_plan_count = %d exceeds attempted_steps %d", diagnostics.CoordinateSearchPlanCount, diagnostics.AttemptedSteps)
	}
	if diagnostics.CoordinateSearchPlanCount == 0 && (diagnostics.CoordinateProposalAttempts != 0 || diagnostics.CoordinateSearchStrategy != "" || diagnostics.CoordinateSearchOrderingHash != "" || diagnostics.CoordinateSearchLearningRate != 0 || diagnostics.CoordinateSearchAudit != "" || diagnostics.CoordinateSearchAuditChain != "" || diagnostics.CoordinateSearchHashChain != "" || diagnostics.CoordinateTopAngles != 0 || diagnostics.CoordinateMagnitudeCount != 0 || diagnostics.CoordinateBlockCount != 0) {
		return fmt.Errorf("AOQT fail-closed diagnostics coordinate audit is present without coordinate plan history")
	}
	if diagnostics.AdamProposalAttempts != diagnostics.AdamAcceptedProposals+diagnostics.AdamRejectedProposals {
		return fmt.Errorf("AOQT fail-closed diagnostics adam proposal accounting mismatch")
	}
	if diagnostics.CoordinateProposalAttempts != diagnostics.CoordinateAcceptedProposals+diagnostics.CoordinateRejectedProposals {
		return fmt.Errorf("AOQT fail-closed diagnostics coordinate proposal accounting mismatch")
	}
	if diagnostics.AdamProposalAttempts+diagnostics.CoordinateProposalAttempts != diagnostics.ProposalAttempts {
		return fmt.Errorf("AOQT fail-closed diagnostics optimizer-path proposal accounting mismatch")
	}
	if diagnostics.AdamAcceptedProposals+diagnostics.CoordinateAcceptedProposals != diagnostics.AcceptedProposals {
		return fmt.Errorf("AOQT fail-closed diagnostics optimizer-path accepted proposal accounting mismatch")
	}
	if diagnostics.AdamRejectedProposals+diagnostics.CoordinateRejectedProposals != diagnostics.RejectedProposals {
		return fmt.Errorf("AOQT fail-closed diagnostics optimizer-path rejected proposal accounting mismatch")
	}
	if diagnostics.CoordinateSearchPlanCount > 0 {
		if diagnostics.CoordinateSearchStrategy != aoqtCoordinateSearchStrategyProtectedConeMicroTail {
			return fmt.Errorf("AOQT fail-closed diagnostics coordinate_search_strategy = %q, want %q", diagnostics.CoordinateSearchStrategy, aoqtCoordinateSearchStrategyProtectedConeMicroTail)
		}
		if diagnostics.CoordinateTopAngles <= 0 || diagnostics.CoordinateTopAngles > aoqtTransactionalCoordinateTopAngles {
			return fmt.Errorf("AOQT fail-closed diagnostics coordinate_top_angles = %d outside expected range", diagnostics.CoordinateTopAngles)
		}
		if diagnostics.CoordinateMagnitudeCount != aoqtTransactionalCoordinateMicroTailMagnitudeCount {
			return fmt.Errorf("AOQT fail-closed diagnostics protected coordinate_magnitude_count = %d, want exact micro-tail count %d", diagnostics.CoordinateMagnitudeCount, aoqtTransactionalCoordinateMicroTailMagnitudeCount)
		}
		if !isFinite32(diagnostics.CoordinateSearchLearningRate) || diagnostics.CoordinateSearchLearningRate <= 0 {
			return fmt.Errorf("AOQT fail-closed diagnostics coordinate_search_learning_rate must be finite and positive")
		}
		if diagnostics.CoordinateBlockCount < 0 || diagnostics.CoordinateBlockCount > len(aoqtTransactionalCoordinateBlockSizes) {
			return fmt.Errorf("AOQT fail-closed diagnostics coordinate_block_count = %d outside expected range", diagnostics.CoordinateBlockCount)
		}
		if err := validateAOQTProtectedCoordinateSearchAudit(plan, weights, diagnostics); err != nil {
			return fmt.Errorf("AOQT fail-closed diagnostics protected coordinate audit: %w", err)
		}
	}
	return nil
}

func validateAOQTFailClosedObjectiveContract(contract AOQTSidecarObjectiveContract, topology AOQTSidecarTopologyBinding) error {
	manifest := AOQTSidecarCalibrationManifest{
		Dim:            topology.Dim,
		TurboQuantSeed: contract.TurboQuantSeed,
		QuantSurfaces: []AOQTSidecarQuantSurface{
			{BitWidth: contract.GainBit, Seed: contract.TurboQuantSeed, ScoreSurface: contract.ScoreSurface, PreparedQuery: true},
			{BitWidth: contract.Q3GuardBit, Seed: contract.TurboQuantSeed, ScoreSurface: contract.ScoreSurface, PreparedQuery: true},
			{BitWidth: contract.Q5GuardBit, Seed: contract.TurboQuantSeed, ScoreSurface: contract.ScoreSurface, PreparedQuery: true},
		},
	}
	if err := contract.Validate(manifest); err != nil {
		return fmt.Errorf("AOQT fail-closed diagnostics objective contract: %w", err)
	}
	return nil
}

func validateAOQTFailClosedIOReport(report AOQTSidecarCalibrationIOReport, plan AOQTSidecarWorkPlan, inputs AOQTSidecarRunMetricInputs, topology AOQTSidecarTopologyBinding, objective AOQTSidecarObjectiveContract, gates AOQTSidecarLegalGates) error {
	if report.Schema != AOQTSidecarCalibrationIOReportSchema {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report schema %q is not supported, want %q", report.Schema, AOQTSidecarCalibrationIOReportSchema)
	}
	for _, field := range []struct {
		name  string
		value string
	}{
		{"manifest_path", report.ManifestPath},
		{"rows_jsonl_path", report.RowsJSONLPath},
		{"manifest_sha256", report.ManifestSHA256},
		{"rows_sha256", report.RowsSHA256},
		{"anchor_artifact_sha256", report.AnchorArtifactSHA256},
		{"anchor_package_manifest_sha256", report.AnchorPackageManifestSHA256},
		{"anchor_embedding_space_id", report.AnchorEmbeddingSpaceID},
		{"compatibility_digest", report.CompatibilityDigest},
	} {
		if strings.TrimSpace(field.value) == "" {
			return fmt.Errorf("AOQT fail-closed diagnostics IO report %s is required", field.name)
		}
	}
	for _, item := range []struct {
		value string
		label string
	}{
		{report.ManifestSHA256, "AOQT fail-closed diagnostics IO report manifest_sha256"},
		{report.RowsSHA256, "AOQT fail-closed diagnostics IO report rows_sha256"},
		{report.AnchorArtifactSHA256, "AOQT fail-closed diagnostics IO report anchor_artifact_sha256"},
		{report.AnchorPackageManifestSHA256, "AOQT fail-closed diagnostics IO report anchor_package_manifest_sha256"},
		{report.CompatibilityDigest, "AOQT fail-closed diagnostics IO report compatibility_digest"},
	} {
		if err := validateAOQTSHA256(item.value, item.label); err != nil {
			return err
		}
	}
	rowsSHA256, err := sha256FileAOQT(report.RowsJSONLPath)
	if err != nil {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report rows file %q: %w", report.RowsJSONLPath, err)
	}
	if rowsSHA256 != report.RowsSHA256 {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report rows_sha256 does not match rows file")
	}
	if report.ManifestSHA256 != inputs.DatasetManifestSHA256 {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report manifest_sha256 must match inputs.dataset_manifest_sha256")
	}
	if report.AnchorArtifactSHA256 != inputs.AnchorArtifactSHA256 || report.AnchorPackageManifestSHA256 != inputs.AnchorPackageManifestSHA256 || report.AnchorEmbeddingSpaceID != inputs.AnchorEmbeddingSpaceID || report.CompatibilityDigest != inputs.CompatibilityDigest {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report provenance must exactly match inputs")
	}
	if report.Topology != topology {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report topology must exactly match topology")
	}
	if !aoqtObjectiveContractsEqual(report.ObjectiveContract, objective) {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report objective_contract must exactly match objective_contract")
	}
	if report.LegalGates != gates {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report legal_gates must exactly match legal_gates")
	}
	if report.TurboQuantSeed != objective.TurboQuantSeed {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report turboquant_seed = %d, want objective contract seed %d", report.TurboQuantSeed, objective.TurboQuantSeed)
	}
	if report.RowCount <= 0 || report.CandidateCount <= 0 || report.PairCount <= 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report workload counts must be positive")
	}
	if report.RowCount != plan.RowCount || report.PairCount != plan.PairCount {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report workload counts must match plan")
	}
	return nil
}

func validateAOQTFailClosedPreflight(preflight AOQTSidecarMaterializePreflight, preflightPath string, report AOQTSidecarCalibrationIOReport, inputs AOQTSidecarRunMetricInputs, topology AOQTSidecarTopologyBinding, objective AOQTSidecarObjectiveContract, gates AOQTSidecarLegalGates) error {
	if preflight.Schema != AOQTSidecarMaterializerPreflightSchema {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight schema %q is not supported, want %q", preflight.Schema, AOQTSidecarMaterializerPreflightSchema)
	}
	if preflight.QualityClaim || !preflight.ResearchOnly || preflight.ActualTrainingRan || preflight.ActualEvalRan || preflight.ActualVectorExportRan {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight must be research-only and record no training/eval/vector export execution")
	}
	if err := validateAOQTSHA256(preflight.PlanSHA256, "AOQT fail-closed diagnostics preflight plan_sha256"); err != nil {
		return err
	}
	if len(preflight.InputSHA256) == 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight input_sha256 is required")
	}
	for path, sum := range preflight.InputSHA256 {
		if strings.TrimSpace(path) == "" {
			return fmt.Errorf("AOQT fail-closed diagnostics preflight input_sha256 path is required")
		}
		if err := validateAOQTSHA256(sum, "AOQT fail-closed diagnostics preflight input_sha256"); err != nil {
			return err
		}
	}
	planHashBound := false
	for _, sum := range preflight.InputSHA256 {
		if sum == preflight.PlanSHA256 {
			planHashBound = true
			break
		}
	}
	if !planHashBound {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight plan_sha256 is not bound by input_sha256")
	}
	if preflight.RowCount <= 0 || preflight.CandidateCount <= 0 || preflight.PairCount <= 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight workload counts must be positive")
	}
	if preflight.RowCount != report.RowCount || preflight.CandidateCount != report.CandidateCount || preflight.PairCount != report.PairCount {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight workload counts must match IO report")
	}
	if preflight.TurboQuantSeed != objective.TurboQuantSeed {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight turboquant_seed = %d, want objective contract seed %d", preflight.TurboQuantSeed, objective.TurboQuantSeed)
	}
	if preflight.TopologySeed != topology.Seed {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight topology_seed = %d, want topology seed %d", preflight.TopologySeed, topology.Seed)
	}
	if preflight.QuantScoreSurface != objective.ScoreSurface {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight quant_score_surface = %q, want objective score surface %q", preflight.QuantScoreSurface, objective.ScoreSurface)
	}
	if preflight.CalibrationManifestSHA256 != report.ManifestSHA256 || preflight.CalibrationManifestSHA256 != inputs.DatasetManifestSHA256 {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight calibration manifest sha256 must match IO report and inputs")
	}
	if !aoqtStringMapsEqual(preflight.QrelsSHA256ByDataset, inputs.QrelsSHA256ByDataset) {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight qrels_sha256_by_dataset must exactly match inputs")
	}
	if preflight.LegalGates != gates {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight legal_gates must exactly match legal_gates")
	}
	if !aoqtObjectiveContractsEqual(preflight.ObjectiveContract, objective) {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight objective_contract must exactly match objective_contract")
	}
	if !slices.Equal(preflight.SourceSemantics, []string{"q3_top10", "q5_top10", "nf_boundary80_120"}) {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight source_semantics must bind q3_top10,q5_top10,nf_boundary80_120")
	}
	for _, key := range []string{"rows_jsonl", "manifest_json", "preflight_json"} {
		if strings.TrimSpace(preflight.Outputs[key]) == "" {
			return fmt.Errorf("AOQT fail-closed diagnostics preflight outputs.%s is required", key)
		}
	}
	if preflight.Outputs["rows_jsonl"] != report.RowsJSONLPath || preflight.Outputs["manifest_json"] != report.ManifestPath {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight outputs must bind IO report paths")
	}
	if preflight.Outputs["preflight_json"] != preflightPath {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight outputs must bind preflight path")
	}
	return nil
}

func aoqtStringMapsEqual(left, right map[string]string) bool {
	if len(left) != len(right) {
		return false
	}
	for key, value := range left {
		if right[key] != value {
			return false
		}
	}
	return true
}

func validateAOQTMetricInputs(inputs AOQTSidecarRunMetricInputs) error {
	for _, field := range []struct {
		name  string
		value string
	}{
		{"anchor_artifact_sha256", inputs.AnchorArtifactSHA256},
		{"anchor_package_manifest_sha256", inputs.AnchorPackageManifestSHA256},
		{"anchor_embedding_space_id", inputs.AnchorEmbeddingSpaceID},
		{"dataset_manifest_sha256", inputs.DatasetManifestSHA256},
		{"compatibility_digest", inputs.CompatibilityDigest},
	} {
		if field.value == "" {
			return fmt.Errorf("AOQT run inputs.%s is required", field.name)
		}
	}
	if err := validateAOQTSHA256(inputs.AnchorArtifactSHA256, "AOQT run inputs.anchor_artifact_sha256"); err != nil {
		return err
	}
	if err := validateAOQTSHA256(inputs.AnchorPackageManifestSHA256, "AOQT run inputs.anchor_package_manifest_sha256"); err != nil {
		return err
	}
	if err := validateAOQTSHA256(inputs.DatasetManifestSHA256, "AOQT run inputs.dataset_manifest_sha256"); err != nil {
		return err
	}
	if err := validateAOQTSHA256(inputs.CompatibilityDigest, "AOQT run inputs.compatibility_digest"); err != nil {
		return err
	}
	if len(inputs.QrelsSHA256ByDataset) == 0 {
		return fmt.Errorf("AOQT run inputs.qrels_sha256_by_dataset is required")
	}
	for dataset, sum := range inputs.QrelsSHA256ByDataset {
		if dataset == "" {
			return fmt.Errorf("AOQT run inputs qrels dataset name is required")
		}
		if err := validateAOQTSHA256(sum, "AOQT run inputs.qrels_sha256_by_dataset"); err != nil {
			return err
		}
	}
	return nil
}

func AOQTFailClosedDiagnosticsPath(metricsPath string) string {
	return metricsPath + ".failclosed.json"
}

func writeAOQTFailClosedDiagnosticsFile(path string, diagnostics AOQTSidecarFailClosedDiagnostics) error {
	if err := diagnostics.Validate(); err != nil {
		return err
	}
	data, err := json.MarshalIndent(diagnostics, "", "  ")
	if err != nil {
		return fmt.Errorf("encode AOQT fail-closed diagnostics JSON: %w", err)
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create AOQT fail-closed diagnostics parent: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".aoqt-failclosed-*")
	if err != nil {
		return fmt.Errorf("create AOQT fail-closed diagnostics temporary file: %w", err)
	}
	tmpPath := tmp.Name()
	cleanup := func() {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
	}
	defer cleanup()
	if err := tmp.Chmod(0o644); err != nil {
		return fmt.Errorf("set AOQT fail-closed diagnostics temporary permissions: %w", err)
	}
	if _, err := tmp.Write(data); err != nil {
		return fmt.Errorf("write AOQT fail-closed diagnostics temporary file: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("sync AOQT fail-closed diagnostics temporary file: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close AOQT fail-closed diagnostics temporary file: %w", err)
	}
	if err := os.Link(tmpPath, path); err != nil {
		if os.IsExist(err) {
			return fmt.Errorf("AOQT fail-closed diagnostics output %q already exists", path)
		}
		return fmt.Errorf("create AOQT fail-closed diagnostics output %q exclusively: %w", path, err)
	}
	return nil
}

func writeAOQTRunMetricsFile(path string, metrics AOQTSidecarRunMetrics) error {
	data, err := json.MarshalIndent(metrics, "", "  ")
	if err != nil {
		return fmt.Errorf("encode AOQT metrics JSON: %w", err)
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create AOQT metrics parent: %w", err)
	}
	return os.WriteFile(path, data, 0o644)
}

func ensureAOQTOutputArtifactAbsent(path string) error {
	for role, candidate := range aoqtCandidatePathMap(aoqtCandidateOutputPaths(path, true)) {
		if candidate == "" {
			continue
		}
		if _, err := os.Lstat(candidate); err == nil {
			return fmt.Errorf("AOQT sidecar output %s %q already exists", role, candidate)
		} else if !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}

func ensureAOQTFailClosedDiagnosticsAbsent(metricsPath string) error {
	path := AOQTFailClosedDiagnosticsPath(metricsPath)
	if _, err := os.Lstat(path); err == nil {
		return fmt.Errorf("AOQT sidecar fail-closed diagnostics output %q already exists", path)
	} else if !os.IsNotExist(err) {
		return err
	}
	return nil
}
