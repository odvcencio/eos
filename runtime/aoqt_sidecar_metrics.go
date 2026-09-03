package eosruntime

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

const AOQTSidecarMetricsSchema = "eos.q3_aoqt_sidecar_metrics.v1"

type AOQTSidecarRunMetrics struct {
	Schema        string                     `json:"schema"`
	Plan          AOQTSidecarWorkPlan        `json:"plan"`
	Inputs        AOQTSidecarRunMetricInputs `json:"inputs"`
	Topology      AOQTSidecarTopologyBinding `json:"topology"`
	LegalGates    AOQTSidecarLegalGates      `json:"legal_gates"`
	Summary       AOQTSidecarTrainSummary    `json:"summary"`
	QualityClaim  bool                       `json:"quality_claim"`
	ReservedStage string                     `json:"reserved_stage,omitempty"`
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
		Topology:      set.Manifest.Topology,
		LegalGates:    set.Manifest.LegalGates,
		Summary:       summary,
		QualityClaim:  false,
		ReservedStage: "Stage2B score-spectrum TurboQuant STE adapter",
	}
	return metrics, metrics.Validate()
}

func (m AOQTSidecarRunMetrics) Validate() error {
	if m.Schema != AOQTSidecarMetricsSchema {
		return fmt.Errorf("AOQT metrics schema %q is not supported, want %q", m.Schema, AOQTSidecarMetricsSchema)
	}
	if m.QualityClaim {
		return fmt.Errorf("AOQT metrics quality_claim must be false for Stage2A")
	}
	if m.Summary.QualityClaim {
		return fmt.Errorf("AOQT summary quality_claim must be false for Stage2A")
	}
	if err := m.Topology.Validate(); err != nil {
		return err
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
