package eosruntime

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

const AOQTSidecarMetricsSchema = "eos.q3_aoqt_sidecar_metrics.v1"

type AOQTSidecarRunMetrics struct {
	Schema            string                       `json:"schema"`
	Plan              AOQTSidecarWorkPlan          `json:"plan"`
	Inputs            AOQTSidecarRunMetricInputs   `json:"inputs"`
	Topology          AOQTSidecarTopologyBinding   `json:"topology"`
	ObjectiveContract AOQTSidecarObjectiveContract `json:"objective_contract"`
	LegalGates        AOQTSidecarLegalGates        `json:"legal_gates"`
	Summary           AOQTSidecarTrainSummary      `json:"summary"`
	QualityClaim      bool                         `json:"quality_claim"`
	ReservedStage     string                       `json:"reserved_stage,omitempty"`
}

type AOQTSidecarRunMetricInputs struct {
	AnchorArtifactSHA256        string            `json:"anchor_artifact_sha256"`
	AnchorPackageManifestSHA256 string            `json:"anchor_package_manifest_sha256"`
	AnchorEmbeddingSpaceID      string            `json:"anchor_embedding_space_id"`
	DatasetManifestSHA256       string            `json:"dataset_manifest_sha256"`
	QrelsSHA256ByDataset        map[string]string `json:"qrels_sha256_by_dataset"`
	CompatibilityDigest         string            `json:"compatibility_digest"`
}

func NewAOQTSidecarRunMetrics(set AOQTSidecarCalibrationSet, summary AOQTSidecarTrainSummary) (AOQTSidecarRunMetrics, error) {
	if err := set.Validate(); err != nil {
		return AOQTSidecarRunMetrics{}, err
	}
	if err := validateAOQTSidecarSummaryPlan(summary, set.Manifest, set.Rows); err != nil {
		return AOQTSidecarRunMetrics{}, err
	}
	manifestHash, err := AOQTSidecarManifestSHA256(set.Manifest)
	if err != nil {
		return AOQTSidecarRunMetrics{}, err
	}
	metrics := AOQTSidecarRunMetrics{
		Schema: AOQTSidecarMetricsSchema,
		Plan:   summary.Plan,
		Inputs: AOQTSidecarRunMetricInputs{
			AnchorArtifactSHA256:        set.Manifest.AnchorArtifactSHA256,
			AnchorPackageManifestSHA256: set.Manifest.AnchorPackageManifestSHA256,
			AnchorEmbeddingSpaceID:      set.Manifest.AnchorEmbeddingSpaceID,
			DatasetManifestSHA256:       manifestHash,
			QrelsSHA256ByDataset:        cloneAOQTStringMap(set.Manifest.QrelsSHA256ByDataset),
			CompatibilityDigest:         set.Manifest.CompatibilityDigest,
		},
		Topology:          set.Manifest.Topology,
		ObjectiveContract: set.Manifest.ObjectiveContract,
		LegalGates:        set.Manifest.LegalGates,
		Summary:           summary,
		QualityClaim:      false,
		ReservedStage:     "Stage2B score-spectrum TurboQuant STE adapter",
	}
	return metrics, metrics.Validate()
}

func (m AOQTSidecarRunMetrics) Validate() error {
	if m.Schema != AOQTSidecarMetricsSchema {
		return fmt.Errorf("AOQT metrics schema %q is not supported, want %q", m.Schema, AOQTSidecarMetricsSchema)
	}
	if m.QualityClaim {
		return fmt.Errorf("AOQT metrics quality_claim must be false for Stage2 runtime")
	}
	if m.Summary.QualityClaim {
		return fmt.Errorf("AOQT summary quality_claim must be false for Stage2 runtime")
	}
	if m.Plan != m.Summary.Plan {
		return fmt.Errorf("AOQT metrics plan must exactly match summary.plan")
	}
	if err := m.Topology.Validate(); err != nil {
		return err
	}
	if err := validateAOQTMetricsPlanTopology(m.Plan, m.Topology); err != nil {
		return err
	}
	if !aoqtObjectiveContractsEqual(m.ObjectiveContract, m.Summary.ObjectiveContract) {
		return fmt.Errorf("AOQT metrics objective_contract must exactly match summary.objective_contract")
	}
	if err := m.ObjectiveContract.Validate(AOQTSidecarCalibrationManifest{
		Dim:            m.Topology.Dim,
		TurboQuantSeed: m.ObjectiveContract.TurboQuantSeed,
		QuantSurfaces: []AOQTSidecarQuantSurface{
			{BitWidth: m.ObjectiveContract.GainBit, Seed: m.ObjectiveContract.TurboQuantSeed, ScoreSurface: m.ObjectiveContract.ScoreSurface, PreparedQuery: true},
			{BitWidth: m.ObjectiveContract.Q3GuardBit, Seed: m.ObjectiveContract.TurboQuantSeed, ScoreSurface: m.ObjectiveContract.ScoreSurface, PreparedQuery: true},
			{BitWidth: m.ObjectiveContract.Q5GuardBit, Seed: m.ObjectiveContract.TurboQuantSeed, ScoreSurface: m.ObjectiveContract.ScoreSurface, PreparedQuery: true},
		},
	}); err != nil {
		return err
	}
	if m.Plan.RowCount <= 0 {
		return fmt.Errorf("AOQT metrics plan.row_count must be positive")
	}
	if m.Plan.PairCount <= 0 {
		return fmt.Errorf("AOQT metrics plan.pair_count must be positive")
	}
	if m.Plan.PlanOnly {
		if m.Plan.StepCount != 0 || m.Summary.Steps != 0 {
			return fmt.Errorf("AOQT plan-only metrics must have zero planned and completed steps")
		}
	} else if m.Summary.Steps != m.Plan.StepCount {
		return fmt.Errorf("AOQT metrics summary.steps = %d, want plan.step_count %d", m.Summary.Steps, m.Plan.StepCount)
	}
	if err := validateAOQTResearchOnlyLegalGates(m.LegalGates, "AOQT metrics"); err != nil {
		return err
	}
	for _, field := range []struct {
		name  string
		value string
	}{
		{"anchor_artifact_sha256", m.Inputs.AnchorArtifactSHA256},
		{"anchor_package_manifest_sha256", m.Inputs.AnchorPackageManifestSHA256},
		{"anchor_embedding_space_id", m.Inputs.AnchorEmbeddingSpaceID},
		{"dataset_manifest_sha256", m.Inputs.DatasetManifestSHA256},
		{"compatibility_digest", m.Inputs.CompatibilityDigest},
	} {
		if field.value == "" {
			return fmt.Errorf("AOQT metrics inputs.%s is required", field.name)
		}
	}
	if err := validateAOQTSHA256(m.Inputs.AnchorArtifactSHA256, "AOQT metrics inputs.anchor_artifact_sha256"); err != nil {
		return err
	}
	if err := validateAOQTSHA256(m.Inputs.AnchorPackageManifestSHA256, "AOQT metrics inputs.anchor_package_manifest_sha256"); err != nil {
		return err
	}
	if err := validateAOQTSHA256(m.Inputs.DatasetManifestSHA256, "AOQT metrics inputs.dataset_manifest_sha256"); err != nil {
		return err
	}
	if err := validateAOQTSHA256(m.Inputs.CompatibilityDigest, "AOQT metrics inputs.compatibility_digest"); err != nil {
		return err
	}
	if len(m.Inputs.QrelsSHA256ByDataset) == 0 {
		return fmt.Errorf("AOQT metrics inputs.qrels_sha256_by_dataset is required")
	}
	for dataset, sum := range m.Inputs.QrelsSHA256ByDataset {
		if dataset == "" {
			return fmt.Errorf("AOQT metrics qrels dataset name is required")
		}
		if err := validateAOQTSHA256(sum, "AOQT metrics inputs.qrels_sha256_by_dataset"); err != nil {
			return err
		}
	}
	return nil
}

func validateAOQTSidecarSummaryPlan(summary AOQTSidecarTrainSummary, manifest AOQTSidecarCalibrationManifest, rows []AOQTSidecarCalibrationRow) error {
	if err := validateAOQTMetricsPlanTopology(summary.Plan, manifest.Topology); err != nil {
		return err
	}
	if summary.Plan.RowCount != len(rows) {
		return fmt.Errorf("AOQT summary plan row_count = %d, want calibration rows %d", summary.Plan.RowCount, len(rows))
	}
	var pairCount int
	for _, row := range rows {
		pairCount += countEligibleAOQTPairs(row)
	}
	if summary.Plan.PairCount != pairCount {
		return fmt.Errorf("AOQT summary plan pair_count = %d, want calibration pair count %d", summary.Plan.PairCount, pairCount)
	}
	if summary.Plan.PlanOnly {
		if summary.Plan.StepCount != 0 || summary.Steps != 0 {
			return fmt.Errorf("AOQT summary plan-only run must have zero planned and completed steps")
		}
	} else if summary.Steps != summary.Plan.StepCount {
		return fmt.Errorf("AOQT summary steps = %d, want plan step_count %d", summary.Steps, summary.Plan.StepCount)
	}
	return nil
}

func validateAOQTMetricsPlanTopology(plan AOQTSidecarWorkPlan, topology AOQTSidecarTopologyBinding) error {
	if plan.Dim != topology.Dim {
		return fmt.Errorf("AOQT plan dim = %d, want topology dim %d", plan.Dim, topology.Dim)
	}
	if plan.Stages != topology.Stages {
		return fmt.Errorf("AOQT plan stages = %d, want topology stages %d", plan.Stages, topology.Stages)
	}
	if plan.PairsPerStage != topology.PairsPerStage {
		return fmt.Errorf("AOQT plan pairs_per_stage = %d, want topology pairs_per_stage %d", plan.PairsPerStage, topology.PairsPerStage)
	}
	if plan.AngleCount != topology.AngleCount {
		return fmt.Errorf("AOQT plan angle_count = %d, want topology angle_count %d", plan.AngleCount, topology.AngleCount)
	}
	if plan.PairingSeed != topology.Seed {
		return fmt.Errorf("AOQT plan pairing_seed = %d, want topology seed %d", plan.PairingSeed, topology.Seed)
	}
	if plan.PairingsSHA256 != topology.PairingsSHA256 {
		return fmt.Errorf("AOQT plan pairings_sha256 mismatch with topology")
	}
	return nil
}

func AOQTSidecarManifestSHA256(manifest AOQTSidecarCalibrationManifest) (string, error) {
	data, err := json.Marshal(manifest)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func cloneAOQTStringMap(in map[string]string) map[string]string {
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
