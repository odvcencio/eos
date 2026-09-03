package eosruntime

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
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
	IOReport      AOQTSidecarCalibrationIOReport
	Preflight     AOQTSidecarMaterializePreflight
	Metrics       AOQTSidecarRunMetrics
	PackageResult *AOQTSidecarCandidatePackageResult
}

func RunAOQTSidecarTraining(cfg AOQTSidecarTrainRunnerConfig) (AOQTSidecarTrainRunnerResult, error) {
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
	preflight, err := loadAndValidateAOQTTrainPreflight(cfg, set, ioReport)
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
		objective, err = objectiveFromAOQTContract(set.Manifest.ObjectiveContract)
		if err != nil {
			return AOQTSidecarTrainRunnerResult{}, err
		}
	}
	summary, err := trainer.Fit(set, objective)
	if err != nil {
		return AOQTSidecarTrainRunnerResult{}, err
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

func loadAndValidateAOQTTrainPreflight(cfg AOQTSidecarTrainRunnerConfig, set AOQTSidecarCalibrationSet, ioReport AOQTSidecarCalibrationIOReport) (AOQTSidecarMaterializePreflight, error) {
	data, err := os.ReadFile(cfg.PreflightJSONPath)
	if err != nil {
		return AOQTSidecarMaterializePreflight{}, err
	}
	if got := sha256BytesAOQT(data); got != cfg.ExpectedPreflightSHA256 {
		return AOQTSidecarMaterializePreflight{}, fmt.Errorf("AOQT sidecar preflight sha256 mismatch")
	}
	var preflight AOQTSidecarMaterializePreflight
	if err := strictUnmarshalAOQT(data, &preflight); err != nil {
		return AOQTSidecarMaterializePreflight{}, fmt.Errorf("%s: invalid AOQT materializer preflight JSON: %w", cfg.PreflightJSONPath, err)
	}
	if preflight.Schema != AOQTSidecarMaterializerPreflightSchema {
		return AOQTSidecarMaterializePreflight{}, fmt.Errorf("AOQT sidecar preflight schema %q is not supported", preflight.Schema)
	}
	if preflight.QualityClaim || !preflight.ResearchOnly || preflight.ActualTrainingRan || preflight.ActualEvalRan || preflight.ActualVectorExportRan {
		return AOQTSidecarMaterializePreflight{}, fmt.Errorf("AOQT sidecar preflight must be research-only and record no training/eval/vector export execution")
	}
	if preflight.TurboQuantSeed != AOQTSidecarMaterializerQuantSeed {
		return AOQTSidecarMaterializePreflight{}, fmt.Errorf("AOQT sidecar preflight turboquant seed = %d, want %d", preflight.TurboQuantSeed, AOQTSidecarMaterializerQuantSeed)
	}
	if preflight.TopologySeed != AOQTSidecarMaterializerTopologySeed {
		return AOQTSidecarMaterializePreflight{}, fmt.Errorf("AOQT sidecar preflight topology seed = %d, want %d", preflight.TopologySeed, AOQTSidecarMaterializerTopologySeed)
	}
	if preflight.RowCount != len(set.Rows) || preflight.CandidateCount != ioReport.CandidateCount || preflight.PairCount != ioReport.PairCount {
		return AOQTSidecarMaterializePreflight{}, fmt.Errorf("AOQT sidecar preflight workload counts do not match calibration IO")
	}
	if preflight.CalibrationManifestSHA256 != ioReport.ManifestSHA256 {
		return AOQTSidecarMaterializePreflight{}, fmt.Errorf("AOQT sidecar preflight calibration manifest sha256 mismatch")
	}
	if !aoqtObjectiveContractsEqual(preflight.ObjectiveContract, set.Manifest.ObjectiveContract) {
		return AOQTSidecarMaterializePreflight{}, fmt.Errorf("AOQT sidecar preflight objective contract mismatch")
	}
	if preflight.LegalGates != set.Manifest.LegalGates {
		return AOQTSidecarMaterializePreflight{}, fmt.Errorf("AOQT sidecar preflight legal gates mismatch")
	}
	return preflight, nil
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
