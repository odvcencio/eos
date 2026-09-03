package eosruntime

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"strings"
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

type AOQTSidecarCandidateEligibilityPolicy struct {
	DenseMaxAbsDeltaTolerance          float64 `json:"dense_max_abs_delta_tolerance"`
	AngleMaxAbsCap                     float32 `json:"angle_max_abs_cap"`
	RequireObjectiveActivation         bool    `json:"require_objective_activation"`
	Q3GainAllowedLossIncrease          float32 `json:"q3_gain_allowed_loss_increase"`
	Q3OrderGuardAllowedLossIncrease    float32 `json:"q3_order_guard_allowed_loss_increase"`
	Q3ScoreDistillAllowedLossIncrease  float32 `json:"q3_score_distill_allowed_loss_increase"`
	Q5OrderGuardAllowedLossIncrease    float32 `json:"q5_order_guard_allowed_loss_increase"`
	Q5ScoreDistillAllowedLossIncrease  float32 `json:"q5_score_distill_allowed_loss_increase"`
	NFBoundaryGuardAllowedLossIncrease float32 `json:"nf_boundary_guard_allowed_loss_increase"`
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
		if m.Summary.OptimizerDiagnostics != nil || strings.TrimSpace(m.Summary.OptimizerDiagnosticsSHA256) != "" {
			return fmt.Errorf("AOQT plan-only metrics must not include optimizer diagnostics")
		}
	} else if err := validateAOQTOptimizerDiagnostics(m.Plan, m.Summary); err != nil {
		return err
	}
	if err := validateAOQTResearchOnlyLegalGates(m.LegalGates, "AOQT metrics"); err != nil {
		return err
	}
	if err := validateAOQTObjectiveComponents(m.Summary.InitialObjectiveComponents, "AOQT metrics summary.initial_objective_components"); err != nil {
		return err
	}
	if err := validateAOQTObjectiveComponents(m.Summary.FinalObjectiveComponents, "AOQT metrics summary.final_objective_components"); err != nil {
		return err
	}
	if err := validateAOQTLossMatchesComponents(m.Summary.InitialLoss, m.Summary.InitialObjectiveComponents, "AOQT metrics summary.initial"); err != nil {
		return err
	}
	if err := validateAOQTLossMatchesComponents(m.Summary.FinalLoss, m.Summary.FinalObjectiveComponents, "AOQT metrics summary.final"); err != nil {
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

func ValidateAOQTSidecarCandidateEligibility(metrics AOQTSidecarRunMetrics, policy AOQTSidecarCandidateEligibilityPolicy) error {
	policy = normalizedAOQTSidecarCandidateEligibilityPolicy(policy)
	if err := validateAOQTSidecarCandidateEligibilityPolicy(policy); err != nil {
		return err
	}
	if err := metrics.Validate(); err != nil {
		return err
	}
	if metrics.Plan.PlanOnly || metrics.Summary.Steps == 0 {
		return fmt.Errorf("AOQT candidate eligibility requires a non-plan training summary")
	}
	if metrics.Summary.OptimizerDiagnostics != nil && metrics.Summary.OptimizerDiagnostics.AcceptedSteps <= 0 {
		return fmt.Errorf("AOQT candidate eligibility requires at least one accepted safe optimizer step")
	}
	if metrics.Summary.QualityClaim || metrics.QualityClaim {
		return fmt.Errorf("AOQT candidate eligibility cannot be based on a quality claim")
	}
	if !isFinite32(metrics.Summary.InitialLoss) || !isFinite32(metrics.Summary.FinalLoss) {
		return fmt.Errorf("AOQT candidate eligibility losses must be finite")
	}
	if metrics.Summary.DenseMaxAbsDelta < 0 || math.IsNaN(metrics.Summary.DenseMaxAbsDelta) || math.IsInf(metrics.Summary.DenseMaxAbsDelta, 0) {
		return fmt.Errorf("AOQT candidate dense_max_abs_delta must be finite and non-negative")
	}
	if metrics.Summary.DenseMaxAbsDelta > policy.DenseMaxAbsDeltaTolerance {
		return fmt.Errorf("AOQT candidate dense_max_abs_delta %.9g exceeds tolerance %.9g", metrics.Summary.DenseMaxAbsDelta, policy.DenseMaxAbsDeltaTolerance)
	}
	if strings.TrimSpace(metrics.Summary.AnglesSHA256) == "" {
		return fmt.Errorf("AOQT candidate angle audit hash is required")
	}
	if err := validateAOQTSHA256(metrics.Summary.AnglesSHA256, "AOQT candidate angles_sha256"); err != nil {
		return err
	}
	if !isFinite32(metrics.Summary.AngleL2) || metrics.Summary.AngleL2 < 0 {
		return fmt.Errorf("AOQT candidate angle_l2 must be finite and non-negative")
	}
	if !isFinite32(metrics.Summary.AngleMaxAbs) || metrics.Summary.AngleMaxAbs < 0 {
		return fmt.Errorf("AOQT candidate angle_max_abs must be finite and non-negative")
	}
	if metrics.Summary.AngleMaxAbs > policy.AngleMaxAbsCap+1e-7 {
		return fmt.Errorf("AOQT candidate angle_max_abs %.9g exceeds cap %.9g", metrics.Summary.AngleMaxAbs, policy.AngleMaxAbsCap)
	}
	if policy.RequireObjectiveActivation {
		if err := validateAOQTActiveObjectiveContributions(metrics.ObjectiveContract.WeightSums, metrics.Summary.FinalObjectiveActivation); err != nil {
			return err
		}
	}
	return validateAOQTAllowedComponentRegressions(metrics.Summary.InitialObjectiveComponents, metrics.Summary.FinalObjectiveComponents, policy)
}

func normalizedAOQTSidecarCandidateEligibilityPolicy(policy AOQTSidecarCandidateEligibilityPolicy) AOQTSidecarCandidateEligibilityPolicy {
	if policy.DenseMaxAbsDeltaTolerance == 0 {
		policy.DenseMaxAbsDeltaTolerance = 5e-4
	}
	if policy.AngleMaxAbsCap == 0 {
		policy.AngleMaxAbsCap = AOQTSidecarDefaultAngleCap
	}
	policy.RequireObjectiveActivation = true
	return policy
}

func validateAOQTSidecarCandidateEligibilityPolicy(policy AOQTSidecarCandidateEligibilityPolicy) error {
	if policy.DenseMaxAbsDeltaTolerance <= 0 || math.IsNaN(policy.DenseMaxAbsDeltaTolerance) || math.IsInf(policy.DenseMaxAbsDeltaTolerance, 0) {
		return fmt.Errorf("AOQT candidate dense_max_abs_delta_tolerance must be finite and positive")
	}
	if policy.AngleMaxAbsCap <= 0 || !isFinite32(policy.AngleMaxAbsCap) || policy.AngleMaxAbsCap > AOQTSidecarHardMaxAngleCap {
		return fmt.Errorf("AOQT candidate angle_max_abs_cap must be finite, positive, and <= hard cap %.9g", AOQTSidecarHardMaxAngleCap)
	}
	for _, item := range []struct {
		name  string
		value float32
	}{
		{"q3_gain_allowed_loss_increase", policy.Q3GainAllowedLossIncrease},
		{"q3_order_guard_allowed_loss_increase", policy.Q3OrderGuardAllowedLossIncrease},
		{"q3_score_distill_allowed_loss_increase", policy.Q3ScoreDistillAllowedLossIncrease},
		{"q5_order_guard_allowed_loss_increase", policy.Q5OrderGuardAllowedLossIncrease},
		{"q5_score_distill_allowed_loss_increase", policy.Q5ScoreDistillAllowedLossIncrease},
		{"nf_boundary_guard_allowed_loss_increase", policy.NFBoundaryGuardAllowedLossIncrease},
	} {
		if !isFinite32(item.value) || item.value < 0 {
			return fmt.Errorf("AOQT candidate %s must be finite and non-negative", item.name)
		}
	}
	return nil
}

func validateAOQTActiveObjectiveContributions(weights AOQTSidecarRowWeights, activation AOQTSidecarObjectiveActivation) error {
	if weights.Q3Gain > 0 && (activation.Q3GainEligiblePairs == 0 || activation.Q3GainContributingPairs == 0) {
		return fmt.Errorf("AOQT candidate q3_gain objective is inactive or non-contributing")
	}
	if weights.Q3OrderGuard > 0 && (activation.Q3OrderGuardPairs == 0 || activation.Q3OrderGuardContributing == 0) {
		return fmt.Errorf("AOQT candidate q3_order_guard objective is inactive or non-contributing")
	}
	if weights.Q3ScoreDistill > 0 && activation.Q3ScoreDistillCount == 0 {
		return fmt.Errorf("AOQT candidate q3_score_distill objective is inactive")
	}
	if weights.Q5OrderGuard > 0 && (activation.Q5OrderGuardPairs == 0 || activation.Q5OrderGuardContributing == 0) {
		return fmt.Errorf("AOQT candidate q5_order_guard objective is inactive or non-contributing")
	}
	if weights.Q5ScoreDistill > 0 && activation.Q5ScoreDistillCount == 0 {
		return fmt.Errorf("AOQT candidate q5_score_distill objective is inactive")
	}
	if weights.NFBoundaryGuard > 0 && (activation.NFBoundaryGuardPairs == 0 || activation.NFBoundaryGuardContributing == 0) {
		return fmt.Errorf("AOQT candidate nf_boundary_guard objective is inactive or non-contributing")
	}
	return nil
}

func validateAOQTAllowedComponentRegressions(initial, final AOQTSidecarObjectiveComponents, policy AOQTSidecarCandidateEligibilityPolicy) error {
	for _, item := range []struct {
		name    string
		initial float32
		final   float32
		allowed float32
	}{
		{"q3_gain", initial.Q3Gain, final.Q3Gain, policy.Q3GainAllowedLossIncrease},
		{"q3_order_guard", initial.Q3OrderGuard, final.Q3OrderGuard, policy.Q3OrderGuardAllowedLossIncrease},
		{"q3_score_distill", initial.Q3ScoreDistill, final.Q3ScoreDistill, policy.Q3ScoreDistillAllowedLossIncrease},
		{"q5_order_guard", initial.Q5OrderGuard, final.Q5OrderGuard, policy.Q5OrderGuardAllowedLossIncrease},
		{"q5_score_distill", initial.Q5ScoreDistill, final.Q5ScoreDistill, policy.Q5ScoreDistillAllowedLossIncrease},
		{"nf_boundary_guard", initial.NFBoundaryGuard, final.NFBoundaryGuard, policy.NFBoundaryGuardAllowedLossIncrease},
	} {
		if item.final-item.initial > item.allowed+1e-7 {
			return fmt.Errorf("AOQT candidate %s component regressed by %.9g, allowed %.9g", item.name, item.final-item.initial, item.allowed)
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
		if summary.OptimizerDiagnostics != nil || strings.TrimSpace(summary.OptimizerDiagnosticsSHA256) != "" {
			return fmt.Errorf("AOQT summary plan-only run must not include optimizer diagnostics")
		}
	} else if err := validateAOQTOptimizerDiagnostics(summary.Plan, summary); err != nil {
		return err
	}
	return nil
}

func validateAOQTOptimizerDiagnostics(plan AOQTSidecarWorkPlan, summary AOQTSidecarTrainSummary) error {
	if summary.OptimizerDiagnostics == nil {
		return fmt.Errorf("AOQT optimizer diagnostics are required for non-plan AOQT summaries")
	}
	diagnostics := *summary.OptimizerDiagnostics
	if diagnostics.PlannedSteps != plan.StepCount {
		return fmt.Errorf("AOQT optimizer diagnostics planned_steps = %d, want plan step_count %d", diagnostics.PlannedSteps, plan.StepCount)
	}
	if diagnostics.MaxAttemptsPerStep != aoqtTransactionalMaxAttemptsPerStep {
		return fmt.Errorf("AOQT optimizer diagnostics max_attempts_per_step = %d, want %d", diagnostics.MaxAttemptsPerStep, aoqtTransactionalMaxAttemptsPerStep)
	}
	if diagnostics.AttemptedSteps <= 0 || diagnostics.AttemptedSteps > diagnostics.PlannedSteps {
		return fmt.Errorf("AOQT optimizer diagnostics attempted_steps = %d outside [1,%d]", diagnostics.AttemptedSteps, diagnostics.PlannedSteps)
	}
	if diagnostics.AcceptedSteps != summary.Steps {
		return fmt.Errorf("AOQT optimizer diagnostics accepted_steps = %d, want summary.steps %d", diagnostics.AcceptedSteps, summary.Steps)
	}
	if diagnostics.AcceptedProposals != diagnostics.AcceptedSteps {
		return fmt.Errorf("AOQT optimizer diagnostics accepted_proposals = %d, want accepted_steps %d", diagnostics.AcceptedProposals, diagnostics.AcceptedSteps)
	}
	if diagnostics.AcceptedSteps <= 0 {
		return fmt.Errorf("AOQT optimizer diagnostics accepted_steps must be positive")
	}
	if diagnostics.AcceptedSteps > diagnostics.AttemptedSteps {
		return fmt.Errorf("AOQT optimizer diagnostics accepted_steps exceeds attempted_steps")
	}
	if diagnostics.ProposalAttempts != diagnostics.AcceptedProposals+diagnostics.RejectedProposals {
		return fmt.Errorf("AOQT optimizer diagnostics proposal accounting mismatch")
	}
	if diagnostics.Backtracks != diagnostics.RejectedProposals {
		return fmt.Errorf("AOQT optimizer diagnostics backtracks = %d, want rejected_proposals %d", diagnostics.Backtracks, diagnostics.RejectedProposals)
	}
	if diagnostics.ProposalAttempts < diagnostics.AttemptedSteps || diagnostics.ProposalAttempts > diagnostics.AttemptedSteps*diagnostics.MaxAttemptsPerStep {
		return fmt.Errorf("AOQT optimizer diagnostics proposal_attempts = %d outside expected range", diagnostics.ProposalAttempts)
	}
	if diagnostics.ExhaustedSteps < 0 || diagnostics.ExhaustedSteps > 1 {
		return fmt.Errorf("AOQT optimizer diagnostics exhausted_steps = %d, want 0 or 1", diagnostics.ExhaustedSteps)
	}
	if diagnostics.ExhaustedSteps == 0 && diagnostics.AcceptedSteps != diagnostics.AttemptedSteps {
		return fmt.Errorf("AOQT optimizer diagnostics accepted_steps = %d, want attempted_steps %d when no step exhausted", diagnostics.AcceptedSteps, diagnostics.AttemptedSteps)
	}
	if diagnostics.ExhaustedSteps == 1 && diagnostics.AcceptedSteps+1 != diagnostics.AttemptedSteps {
		return fmt.Errorf("AOQT optimizer diagnostics attempted_steps must equal accepted_steps+1 when a step exhausts")
	}
	if strings.TrimSpace(summary.OptimizerDiagnosticsSHA256) == "" {
		return fmt.Errorf("AOQT optimizer diagnostics sha256 is required")
	}
	if err := validateAOQTSHA256(summary.OptimizerDiagnosticsSHA256, "AOQT optimizer diagnostics sha256"); err != nil {
		return err
	}
	got, err := diagnostics.SHA256()
	if err != nil {
		return err
	}
	if got != summary.OptimizerDiagnosticsSHA256 {
		return fmt.Errorf("AOQT optimizer diagnostics sha256 mismatch")
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
