package eosruntime

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
)

// v2 records the protected-cone coordinate audit and per-step plan history.
// v1 summaries are intentionally not relabeled as this stricter contract.
const AOQTSidecarMetricsSchema = "eos.q3_aoqt_sidecar_metrics.v2"

type AOQTSidecarRunMetrics struct {
	Schema                     string                                 `json:"schema"`
	Plan                       AOQTSidecarWorkPlan                    `json:"plan"`
	Inputs                     AOQTSidecarRunMetricInputs             `json:"inputs"`
	Topology                   AOQTSidecarTopologyBinding             `json:"topology"`
	ObjectiveContract          AOQTSidecarObjectiveContract           `json:"objective_contract"`
	LegalGates                 AOQTSidecarLegalGates                  `json:"legal_gates"`
	Summary                    AOQTSidecarTrainSummary                `json:"summary"`
	QualityClaim               bool                                   `json:"quality_claim"`
	ReservedStage              string                                 `json:"reserved_stage,omitempty"`
	TrainingContract           string                                 `json:"training_contract,omitempty"`
	CandidateEligibilityPolicy *AOQTSidecarCandidateEligibilityPolicy `json:"candidate_eligibility_policy,omitempty"`
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

// AOQTSidecarScoreDistillBudgetLaneAContract is an explicit train-only
// contract. It permits only the measured score-distill component budgets;
// strict q3 gain improvement, total-loss non-increase, dense/angle caps, and
// every other component guard remain unchanged. The allowance is in the
// existing AOQTSidecarObjectiveComponents units: each field is the aggregate
// row-weighted component sum, not a second per-row mean.
const AOQTSidecarScoreDistillBudgetLaneAContract = "aoqt-trainonly-score-distill-budget-laneA-v1"

func AOQTSidecarDefaultCandidateEligibilityPolicy() AOQTSidecarCandidateEligibilityPolicy {
	return normalizedAOQTSidecarCandidateEligibilityPolicy(AOQTSidecarCandidateEligibilityPolicy{})
}

func AOQTSidecarScoreDistillBudgetLaneAPolicy() AOQTSidecarCandidateEligibilityPolicy {
	policy := AOQTSidecarDefaultCandidateEligibilityPolicy()
	policy.Q3ScoreDistillAllowedLossIncrease = 1e-5
	policy.Q5ScoreDistillAllowedLossIncrease = 1e-5
	return policy
}

// AOQTSidecarTrainingContractPolicy resolves the policy bound to a manifest,
// preflight, summary, or metrics artifact. Empty fields mean the historical
// zero-budget contract. Positive budgets are accepted only under the named
// laneA contract, preventing a budget from being silently relabeled as the
// legacy policy or from being changed under a reused name.
func AOQTSidecarTrainingContractPolicy(name string, declared *AOQTSidecarCandidateEligibilityPolicy) (AOQTSidecarCandidateEligibilityPolicy, error) {
	originalName := name
	name = strings.TrimSpace(name)
	if originalName != name {
		return AOQTSidecarCandidateEligibilityPolicy{}, fmt.Errorf("AOQT training contract name must be canonical without surrounding whitespace")
	}
	if name == "" {
		defaultPolicy := AOQTSidecarDefaultCandidateEligibilityPolicy()
		if declared == nil {
			return defaultPolicy, nil
		}
		policy := normalizedAOQTSidecarCandidateEligibilityPolicy(*declared)
		if err := validateAOQTSidecarCandidateEligibilityPolicy(policy); err != nil {
			return AOQTSidecarCandidateEligibilityPolicy{}, err
		}
		if !aoqtCandidateEligibilityPoliciesEqual(policy, defaultPolicy) {
			return AOQTSidecarCandidateEligibilityPolicy{}, fmt.Errorf("legacy AOQT training contract must retain the default zero-budget eligibility policy")
		}
		return policy, nil
	}
	if name != AOQTSidecarScoreDistillBudgetLaneAContract {
		return AOQTSidecarCandidateEligibilityPolicy{}, fmt.Errorf("unsupported AOQT training contract %q", name)
	}
	if declared == nil {
		return AOQTSidecarCandidateEligibilityPolicy{}, fmt.Errorf("AOQT training contract %q requires an explicit candidate eligibility policy", name)
	}
	policy := normalizedAOQTSidecarCandidateEligibilityPolicy(*declared)
	if err := validateAOQTSidecarCandidateEligibilityPolicy(policy); err != nil {
		return AOQTSidecarCandidateEligibilityPolicy{}, err
	}
	want := AOQTSidecarScoreDistillBudgetLaneAPolicy()
	if !aoqtCandidateEligibilityPoliciesEqual(policy, want) {
		return AOQTSidecarCandidateEligibilityPolicy{}, fmt.Errorf("AOQT training contract %q policy does not match its preregistered q3/q5 score-distill budgets", name)
	}
	return policy, nil
}

func cloneAOQTSidecarCandidateEligibilityPolicy(policy *AOQTSidecarCandidateEligibilityPolicy) *AOQTSidecarCandidateEligibilityPolicy {
	if policy == nil {
		return nil
	}
	copy := *policy
	return &copy
}

func aoqtCandidateEligibilityPoliciesEqual(a, b AOQTSidecarCandidateEligibilityPolicy) bool {
	return a.DenseMaxAbsDeltaTolerance == b.DenseMaxAbsDeltaTolerance &&
		a.AngleMaxAbsCap == b.AngleMaxAbsCap &&
		a.RequireObjectiveActivation == b.RequireObjectiveActivation &&
		a.Q3GainAllowedLossIncrease == b.Q3GainAllowedLossIncrease &&
		a.Q3OrderGuardAllowedLossIncrease == b.Q3OrderGuardAllowedLossIncrease &&
		a.Q3ScoreDistillAllowedLossIncrease == b.Q3ScoreDistillAllowedLossIncrease &&
		a.Q5OrderGuardAllowedLossIncrease == b.Q5OrderGuardAllowedLossIncrease &&
		a.Q5ScoreDistillAllowedLossIncrease == b.Q5ScoreDistillAllowedLossIncrease &&
		a.NFBoundaryGuardAllowedLossIncrease == b.NFBoundaryGuardAllowedLossIncrease
}

func validateAOQTSidecarTrainingContractBinding(leftName string, leftPolicy *AOQTSidecarCandidateEligibilityPolicy, rightName string, rightPolicy *AOQTSidecarCandidateEligibilityPolicy) error {
	leftEffective, err := AOQTSidecarTrainingContractPolicy(leftName, leftPolicy)
	if err != nil {
		return fmt.Errorf("left contract: %w", err)
	}
	rightEffective, err := AOQTSidecarTrainingContractPolicy(rightName, rightPolicy)
	if err != nil {
		return fmt.Errorf("right contract: %w", err)
	}
	if strings.TrimSpace(leftName) != strings.TrimSpace(rightName) {
		return fmt.Errorf("contract name mismatch %q vs %q", leftName, rightName)
	}
	if !aoqtCandidateEligibilityPoliciesEqual(leftEffective, rightEffective) {
		return fmt.Errorf("candidate eligibility policy mismatch")
	}
	return nil
}

func NewAOQTSidecarRunMetrics(set AOQTSidecarCalibrationSet, summary AOQTSidecarTrainSummary) (AOQTSidecarRunMetrics, error) {
	if err := set.Validate(); err != nil {
		return AOQTSidecarRunMetrics{}, err
	}
	if err := validateAOQTSidecarSummaryPlan(summary, set.Manifest, set.Rows); err != nil {
		return AOQTSidecarRunMetrics{}, err
	}
	if err := validateAOQTSidecarTrainingContractBinding(set.Manifest.TrainingContract, set.Manifest.CandidateEligibilityPolicy, summary.TrainingContract, summary.CandidateEligibilityPolicy); err != nil {
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
		Topology:                   set.Manifest.Topology,
		ObjectiveContract:          set.Manifest.ObjectiveContract,
		LegalGates:                 set.Manifest.LegalGates,
		Summary:                    summary,
		QualityClaim:               false,
		ReservedStage:              "Stage2B score-spectrum TurboQuant STE adapter",
		TrainingContract:           set.Manifest.TrainingContract,
		CandidateEligibilityPolicy: cloneAOQTSidecarCandidateEligibilityPolicy(set.Manifest.CandidateEligibilityPolicy),
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
	if err := validateAOQTSidecarTrainingContractBinding(m.TrainingContract, m.CandidateEligibilityPolicy, m.Summary.TrainingContract, m.Summary.CandidateEligibilityPolicy); err != nil {
		return fmt.Errorf("AOQT metrics training contract: %w", err)
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
	if err := validateAOQTObjectiveActivation(m.Summary.InitialObjectiveActivation, "AOQT metrics summary.initial_objective_activation"); err != nil {
		return err
	}
	if err := validateAOQTObjectiveActivation(m.Summary.FinalObjectiveActivation, "AOQT metrics summary.final_objective_activation"); err != nil {
		return err
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
	declaredPolicy, err := AOQTSidecarTrainingContractPolicy(metrics.TrainingContract, metrics.CandidateEligibilityPolicy)
	if err != nil {
		return fmt.Errorf("AOQT candidate training contract: %w", err)
	}
	if !aoqtCandidateEligibilityPoliciesEqual(policy, declaredPolicy) {
		return fmt.Errorf("AOQT candidate eligibility policy does not match metrics training contract")
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
	if lossDelta := metrics.Summary.FinalLoss - metrics.Summary.InitialLoss; lossDelta > aoqtTransactionalLossEpsilon {
		return fmt.Errorf("AOQT candidate total loss increased by %.9g beyond transactional epsilon %.9g", lossDelta, aoqtTransactionalLossEpsilon)
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
	if metrics.Summary.AngleMaxAbs == 0 {
		return fmt.Errorf("AOQT candidate angle_max_abs must be non-zero")
	}
	if metrics.Summary.AngleMaxAbs > policy.AngleMaxAbsCap+1e-7 {
		return fmt.Errorf("AOQT candidate angle_max_abs %.9g exceeds cap %.9g", metrics.Summary.AngleMaxAbs, policy.AngleMaxAbsCap)
	}
	if metrics.Summary.FinalObjectiveComponents.Q3Gain >= metrics.Summary.InitialObjectiveComponents.Q3Gain {
		return fmt.Errorf("AOQT candidate q3_gain component must strictly improve")
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
	regressions := aoqtComponentRegressions(initial, final, policy)
	if len(regressions) > 0 {
		item := regressions[0]
		return fmt.Errorf("AOQT candidate %s component regressed by %.9g, allowed %.9g", item.name, item.delta, item.allowed)
	}
	return nil
}

type aoqtComponentRegression struct {
	name    string
	delta   float32
	allowed float32
}

func aoqtComponentRegressions(initial, final AOQTSidecarObjectiveComponents, policy AOQTSidecarCandidateEligibilityPolicy) []aoqtComponentRegression {
	var regressions []aoqtComponentRegression
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
		delta := item.final - item.initial
		if delta > item.allowed+1e-7 {
			regressions = append(regressions, aoqtComponentRegression{name: item.name, delta: delta, allowed: item.allowed})
		}
	}
	return regressions
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
	if len(aoqtActiveProtectedComponentNames(summary.ObjectiveContract.WeightSums)) == 0 {
		return fmt.Errorf("AOQT metrics v2 non-plan summaries require at least one active protected objective component")
	}
	diagnostics := *summary.OptimizerDiagnostics
	if err := validateAOQTOptimizerPathDiagnostics(diagnostics, "AOQT optimizer diagnostics"); err != nil {
		return err
	}
	for _, item := range []struct {
		name  string
		value int
	}{
		{"planned_steps", diagnostics.PlannedSteps},
		{"attempted_steps", diagnostics.AttemptedSteps},
		{"accepted_steps", diagnostics.AcceptedSteps},
		{"proposal_attempts", diagnostics.ProposalAttempts},
		{"accepted_proposals", diagnostics.AcceptedProposals},
		{"rejected_proposals", diagnostics.RejectedProposals},
		{"backtracks", diagnostics.Backtracks},
		{"max_attempts_per_step", diagnostics.MaxAttemptsPerStep},
		{"exhausted_steps", diagnostics.ExhaustedSteps},
		{"adam_proposal_attempts", diagnostics.AdamProposalAttempts},
		{"adam_accepted_proposals", diagnostics.AdamAcceptedProposals},
		{"adam_rejected_proposals", diagnostics.AdamRejectedProposals},
		{"coordinate_proposal_attempts", diagnostics.CoordinateProposalAttempts},
		{"coordinate_accepted_proposals", diagnostics.CoordinateAcceptedProposals},
		{"coordinate_rejected_proposals", diagnostics.CoordinateRejectedProposals},
		{"coordinate_search_plan_count", diagnostics.CoordinateSearchPlanCount},
		{"coordinate_top_angles", diagnostics.CoordinateTopAngles},
		{"coordinate_magnitude_count", diagnostics.CoordinateMagnitudeCount},
		{"coordinate_block_count", diagnostics.CoordinateBlockCount},
	} {
		if item.value < 0 {
			return fmt.Errorf("AOQT optimizer diagnostics %s must be non-negative", item.name)
		}
	}
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
	hasOptimizerPathDiagnostics := diagnostics.AdamProposalAttempts != 0 ||
		diagnostics.AdamAcceptedProposals != 0 ||
		diagnostics.AdamRejectedProposals != 0 ||
		diagnostics.CoordinateProposalAttempts != 0 ||
		diagnostics.CoordinateAcceptedProposals != 0 ||
		diagnostics.CoordinateRejectedProposals != 0
	if diagnostics.CoordinateSearchPlanCount < 0 {
		return fmt.Errorf("AOQT optimizer diagnostics coordinate_search_plan_count must be non-negative")
	}
	if diagnostics.CoordinateSearchPlanCount > diagnostics.AttemptedSteps {
		return fmt.Errorf("AOQT optimizer diagnostics coordinate_search_plan_count = %d exceeds attempted_steps %d", diagnostics.CoordinateSearchPlanCount, diagnostics.AttemptedSteps)
	}
	if diagnostics.CoordinateSearchPlanCount == 0 && (diagnostics.CoordinateProposalAttempts != 0 || diagnostics.CoordinateSearchStrategy != "" || diagnostics.CoordinateSearchOrderingHash != "" || diagnostics.CoordinateSearchLearningRate != 0 || diagnostics.CoordinateSearchAudit != "" || diagnostics.CoordinateSearchAuditChain != "" || diagnostics.CoordinateSearchHashChain != "" || diagnostics.CoordinateTopAngles != 0 || diagnostics.CoordinateMagnitudeCount != 0 || diagnostics.CoordinateBlockCount != 0) {
		return fmt.Errorf("AOQT optimizer diagnostics coordinate audit is present without coordinate proposals")
	}
	if diagnostics.CoordinateSearchPlanCount > 0 && !hasOptimizerPathDiagnostics {
		return fmt.Errorf("AOQT optimizer diagnostics coordinate plan history is present without optimizer path diagnostics")
	}
	if !hasOptimizerPathDiagnostics && (diagnostics.ProposalAttempts != 0 || diagnostics.AcceptedProposals != 0 || diagnostics.RejectedProposals != 0 || diagnostics.Backtracks != 0) {
		return fmt.Errorf("AOQT optimizer diagnostics top-level proposal accounting requires optimizer path diagnostics")
	}
	if hasOptimizerPathDiagnostics {
		if diagnostics.AdamProposalAttempts != diagnostics.AdamAcceptedProposals+diagnostics.AdamRejectedProposals {
			return fmt.Errorf("AOQT optimizer diagnostics adam proposal accounting mismatch")
		}
		if diagnostics.CoordinateProposalAttempts != diagnostics.CoordinateAcceptedProposals+diagnostics.CoordinateRejectedProposals {
			return fmt.Errorf("AOQT optimizer diagnostics coordinate proposal accounting mismatch")
		}
		if diagnostics.AdamProposalAttempts+diagnostics.CoordinateProposalAttempts != diagnostics.ProposalAttempts {
			return fmt.Errorf("AOQT optimizer diagnostics optimizer-path proposal accounting mismatch")
		}
		if diagnostics.AdamAcceptedProposals+diagnostics.CoordinateAcceptedProposals != diagnostics.AcceptedProposals {
			return fmt.Errorf("AOQT optimizer diagnostics optimizer-path accepted proposal accounting mismatch")
		}
		if diagnostics.AdamRejectedProposals+diagnostics.CoordinateRejectedProposals != diagnostics.RejectedProposals {
			return fmt.Errorf("AOQT optimizer diagnostics optimizer-path rejected proposal accounting mismatch")
		}
	}
	if diagnostics.CoordinateSearchPlanCount > 0 {
		if diagnostics.CoordinateSearchStrategy != aoqtCoordinateSearchStrategyProtectedConeMicroTail {
			return fmt.Errorf("AOQT optimizer diagnostics coordinate_search_strategy = %q, want %q", diagnostics.CoordinateSearchStrategy, aoqtCoordinateSearchStrategyProtectedConeMicroTail)
		}
		if diagnostics.CoordinateTopAngles <= 0 || diagnostics.CoordinateTopAngles > aoqtTransactionalCoordinateTopAngles {
			return fmt.Errorf("AOQT optimizer diagnostics coordinate_top_angles = %d outside expected range", diagnostics.CoordinateTopAngles)
		}
		if diagnostics.CoordinateMagnitudeCount != aoqtTransactionalCoordinateMicroTailMagnitudeCount {
			return fmt.Errorf("AOQT optimizer diagnostics protected coordinate_magnitude_count = %d, want exact micro-tail count %d", diagnostics.CoordinateMagnitudeCount, aoqtTransactionalCoordinateMicroTailMagnitudeCount)
		}
		if !isFinite32(diagnostics.CoordinateSearchLearningRate) || diagnostics.CoordinateSearchLearningRate <= 0 {
			return fmt.Errorf("AOQT optimizer diagnostics coordinate_search_learning_rate must be finite and positive")
		}
		if diagnostics.CoordinateBlockCount < 0 || diagnostics.CoordinateBlockCount > len(aoqtTransactionalCoordinateBlockSizes) {
			return fmt.Errorf("AOQT optimizer diagnostics coordinate_block_count = %d outside expected range", diagnostics.CoordinateBlockCount)
		}
		if err := validateAOQTProtectedCoordinateSearchAudit(plan, summary.ObjectiveContract.WeightSums, diagnostics); err != nil {
			return err
		}
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
	if diagnostics.ExhaustedSteps == 0 && diagnostics.AttemptedSteps != diagnostics.PlannedSteps {
		return fmt.Errorf("AOQT optimizer diagnostics attempted_steps = %d, want planned_steps %d when no step exhausted", diagnostics.AttemptedSteps, diagnostics.PlannedSteps)
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

func validateAOQTProtectedCoordinateSearchAudit(plan AOQTSidecarWorkPlan, weights AOQTSidecarRowWeights, diagnostics AOQTSidecarOptimizerDiagnostics) error {
	if !isFinite32(plan.LearningRate) || plan.LearningRate <= 0 {
		return fmt.Errorf("AOQT optimizer diagnostics work plan learning_rate must be finite and positive")
	}
	audits, err := decodeAOQTCoordinateSearchStringChain(diagnostics.CoordinateSearchAuditChain, "coordinate_search_audit_chain")
	if err != nil {
		return err
	}
	hashes, err := decodeAOQTCoordinateSearchStringChain(diagnostics.CoordinateSearchHashChain, "coordinate_search_hash_chain")
	if err != nil {
		return err
	}
	if len(audits) != len(hashes) || len(audits) == 0 {
		return fmt.Errorf("AOQT optimizer diagnostics coordinate audit/hash chain lengths = %d/%d, want equal non-zero chains", len(audits), len(hashes))
	}
	if diagnostics.CoordinateSearchPlanCount != len(audits) {
		return fmt.Errorf("AOQT optimizer diagnostics coordinate_search_plan_count = %d, audit chain length = %d", diagnostics.CoordinateSearchPlanCount, len(audits))
	}
	previousHash := ""
	var chainLearningRate float32
	for i, rawAudit := range audits {
		var payload aoqtCoordinateSearchAuditPayload
		if err := strictUnmarshalAOQT([]byte(rawAudit), &payload); err != nil {
			return fmt.Errorf("AOQT optimizer diagnostics coordinate audit[%d] is invalid: %w", i, err)
		}
		canonical, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		if string(canonical) != rawAudit {
			return fmt.Errorf("AOQT optimizer diagnostics coordinate audit[%d] is not canonical JSON", i)
		}
		// Check the immutable per-plan rate relation before the full payload
		// verifier. This keeps multi-step history tampering distinguishable from
		// a single-entry plan-rate mismatch while both remain fail-closed.
		if !isFinite32(payload.LearningRate) || payload.LearningRate <= 0 {
			return fmt.Errorf("AOQT optimizer diagnostics coordinate audit[%d] learning_rate must be finite and positive", i)
		}
		if i == 0 {
			chainLearningRate = payload.LearningRate
		} else if payload.LearningRate != chainLearningRate {
			return fmt.Errorf("AOQT optimizer diagnostics coordinate audit[%d] learning_rate = %.9g differs from fixed chain learning_rate %.9g", i, payload.LearningRate, chainLearningRate)
		}
		if err := validateAOQTProtectedCoordinateSearchAuditPayload(payload, plan.AngleCount, plan.LearningRate, weights); err != nil {
			return fmt.Errorf("AOQT optimizer diagnostics coordinate audit[%d]: %w", i, err)
		}
		wantHash := sha256AOQTCoordinateSearchChainEntry(previousHash, rawAudit)
		if hashes[i] != wantHash {
			return fmt.Errorf("AOQT optimizer diagnostics coordinate hash chain[%d] does not bind audit payload/predecessor", i)
		}
		previousHash = hashes[i]
	}
	if diagnostics.CoordinateSearchAudit != audits[len(audits)-1] {
		return fmt.Errorf("AOQT optimizer diagnostics final coordinate audit does not match audit chain tail")
	}
	if diagnostics.CoordinateSearchOrderingHash != hashes[len(hashes)-1] {
		return fmt.Errorf("AOQT optimizer diagnostics final coordinate ordering hash does not match hash chain tail")
	}
	var final aoqtCoordinateSearchAuditPayload
	if err := strictUnmarshalAOQT([]byte(audits[len(audits)-1]), &final); err != nil {
		return fmt.Errorf("AOQT optimizer diagnostics final coordinate audit is invalid: %w", err)
	}
	if len(final.Order) != diagnostics.CoordinateTopAngles {
		return fmt.Errorf("AOQT optimizer diagnostics coordinate_top_angles = %d, audit order count = %d", diagnostics.CoordinateTopAngles, len(final.Order))
	}
	if len(final.Magnitudes) != diagnostics.CoordinateMagnitudeCount {
		return fmt.Errorf("AOQT optimizer diagnostics coordinate_magnitude_count = %d, audit magnitude count = %d", diagnostics.CoordinateMagnitudeCount, len(final.Magnitudes))
	}
	if len(final.BlockSizes) != diagnostics.CoordinateBlockCount {
		return fmt.Errorf("AOQT optimizer diagnostics coordinate_block_count = %d, audit block count = %d", diagnostics.CoordinateBlockCount, len(final.BlockSizes))
	}
	if final.LearningRate != diagnostics.CoordinateSearchLearningRate {
		return fmt.Errorf("AOQT optimizer diagnostics coordinate_search_learning_rate = %.9g, audit learning_rate = %.9g", diagnostics.CoordinateSearchLearningRate, final.LearningRate)
	}
	return nil
}

func decodeAOQTCoordinateSearchStringChain(raw, label string) ([]string, error) {
	if strings.TrimSpace(raw) == "" {
		return nil, fmt.Errorf("AOQT optimizer diagnostics %s is required", label)
	}
	var chain []string
	if err := strictUnmarshalAOQT([]byte(raw), &chain); err != nil {
		return nil, fmt.Errorf("AOQT optimizer diagnostics %s is invalid: %w", label, err)
	}
	if len(chain) == 0 {
		return nil, fmt.Errorf("AOQT optimizer diagnostics %s must be non-empty", label)
	}
	canonical, err := json.Marshal(chain)
	if err != nil {
		return nil, err
	}
	if string(canonical) != raw {
		return nil, fmt.Errorf("AOQT optimizer diagnostics %s is not canonical JSON", label)
	}
	return chain, nil
}

func validateAOQTProtectedCoordinateSearchAuditPayload(payload aoqtCoordinateSearchAuditPayload, angleCount int, expectedLearningRate float32, weights AOQTSidecarRowWeights) error {
	if payload.Strategy != aoqtCoordinateSearchStrategyProtectedConeMicroTail {
		return fmt.Errorf("strategy = %q, want protected strategy %q", payload.Strategy, aoqtCoordinateSearchStrategyProtectedConeMicroTail)
	}
	if !isFinite32(payload.LearningRate) || payload.LearningRate <= 0 {
		return fmt.Errorf("learning_rate must be finite and positive")
	}
	if !isFinite32(expectedLearningRate) || expectedLearningRate <= 0 {
		return fmt.Errorf("work plan learning_rate must be finite and positive")
	}
	if payload.LearningRate != expectedLearningRate {
		return fmt.Errorf("learning_rate = %.9g does not match work plan learning_rate %.9g", payload.LearningRate, expectedLearningRate)
	}
	wantComponents := aoqtActiveProtectedComponentNames(weights)
	if len(wantComponents) == 0 {
		return fmt.Errorf("protected strategy requires at least one active protected component")
	}
	if !aoqtStringSlicesEqual(payload.ProtectedComponents, wantComponents) {
		return fmt.Errorf("protected components = %v, want active components %v", payload.ProtectedComponents, wantComponents)
	}
	if len(payload.Order) <= 0 || len(payload.Order) > aoqtTransactionalCoordinateTopAngles {
		return fmt.Errorf("order count = %d outside [1,%d]", len(payload.Order), aoqtTransactionalCoordinateTopAngles)
	}
	if len(payload.FullOrder) < len(payload.Order) {
		return fmt.Errorf("full order count = %d is smaller than top order count %d", len(payload.FullOrder), len(payload.Order))
	}
	if payload.FullRankSourceCount != angleCount {
		return fmt.Errorf("full rank source count = %d, want complete angle count %d", payload.FullRankSourceCount, angleCount)
	}
	if err := validateAOQTSHA256(payload.FullRankSourceSHA256, "AOQT protected full rank source sha256"); err != nil {
		return err
	}
	if len(payload.FullOrder) != payload.FullRankSourceCount {
		return fmt.Errorf("full order count = %d, want complete full rank source count %d", len(payload.FullOrder), payload.FullRankSourceCount)
	}
	wantTopOrder := aoqtTransactionalCoordinateTopAngles
	if len(payload.FullOrder) < wantTopOrder {
		wantTopOrder = len(payload.FullOrder)
	}
	if len(payload.Order) != wantTopOrder {
		return fmt.Errorf("order count = %d, want exact top-order count %d from full rank source", len(payload.Order), wantTopOrder)
	}
	fullOrderJSON, err := json.Marshal(payload.FullOrder)
	if err != nil {
		return err
	}
	fullOrderSum := sha256.Sum256(fullOrderJSON)
	if payload.FullOrderSHA256 != hex.EncodeToString(fullOrderSum[:]) {
		return fmt.Errorf("full order sha256 does not bind full rank source")
	}
	fullRanks, err := validateAOQTProtectedCoordinateSearchAuditOrder(payload.FullOrder, angleCount, wantComponents, true)
	if err != nil {
		return fmt.Errorf("full order: %w", err)
	}
	fullRankSource, err := aoqtCoordinateSearchRankSourceSHA256(fullRanks, wantComponents)
	if err != nil {
		return err
	}
	if payload.FullRankSourceSHA256 != fullRankSource {
		return fmt.Errorf("full rank source sha256 does not bind complete rank source")
	}
	expectedMagnitudes := aoqtCoordinateSearchMagnitudes(payload.LearningRate)
	if len(expectedMagnitudes) != aoqtTransactionalCoordinateMagnitudeCount {
		return fmt.Errorf("protected learning_rate produces %d finite positive magnitudes, want full intended schedule of %d", len(expectedMagnitudes), aoqtTransactionalCoordinateMagnitudeCount)
	}
	expectedMicroTail := aoqtCoordinateSearchMicroTail(expectedMagnitudes, aoqtTransactionalCoordinateMicroTailMagnitudeCount)
	if len(expectedMicroTail) != aoqtTransactionalCoordinateMicroTailMagnitudeCount || !aoqtFloat32SlicesEqual(expectedMicroTail, payload.MicroTailMagnitudes) {
		return fmt.Errorf("protected learning_rate does not bind the exact micro-tail schedule")
	}
	if len(payload.Magnitudes) != aoqtTransactionalCoordinateMicroTailMagnitudeCount || len(payload.MicroTailMagnitudes) != aoqtTransactionalCoordinateMicroTailMagnitudeCount {
		return fmt.Errorf("protected micro-tail magnitude counts = %d/%d, want %d", len(payload.Magnitudes), len(payload.MicroTailMagnitudes), aoqtTransactionalCoordinateMicroTailMagnitudeCount)
	}
	if !aoqtFloat32SlicesEqual(payload.Magnitudes, payload.MicroTailMagnitudes) {
		return fmt.Errorf("protected magnitudes do not exactly match micro-tail magnitudes")
	}
	if !aoqtFloat32SlicesEqual(payload.BlockMagnitudes, payload.MicroTailMagnitudes) {
		return fmt.Errorf("protected block magnitudes do not exactly match micro-tail magnitudes")
	}
	if len(payload.ReverseMagnitudes) != 0 {
		return fmt.Errorf("protected reverse magnitude count = %d, want 0", len(payload.ReverseMagnitudes))
	}
	for i, magnitude := range payload.MicroTailMagnitudes {
		if !isFinite32(magnitude) || magnitude <= 0 {
			return fmt.Errorf("protected micro-tail magnitude[%d] must be finite and positive", i)
		}
		if i > 0 && magnitude != payload.MicroTailMagnitudes[i-1]*0.5 {
			return fmt.Errorf("protected micro-tail magnitude[%d] = %.9g is not the exact half of magnitude[%d]", i, magnitude, i-1)
		}
	}
	wantBlockSizes := aoqtCoordinateSearchBlockSizes(len(payload.Order))
	if !aoqtIntSlicesEqual(payload.BlockSizes, wantBlockSizes) {
		return fmt.Errorf("protected block sizes = %v, want %v", payload.BlockSizes, wantBlockSizes)
	}
	_, err = validateAOQTProtectedCoordinateSearchAuditOrder(payload.Order, angleCount, wantComponents, true)
	if err != nil {
		return fmt.Errorf("top order: %w", err)
	}
	topJSON, err := json.Marshal(payload.Order)
	if err != nil {
		return err
	}
	prefixJSON, err := json.Marshal(payload.FullOrder[:len(payload.Order)])
	if err != nil {
		return err
	}
	if string(topJSON) != string(prefixJSON) {
		return fmt.Errorf("top order is not the prefix of the full rank source")
	}
	ordered := append([]aoqtCoordinateSearchRank(nil), fullRanks...)
	sort.SliceStable(ordered, func(i, j int) bool {
		return aoqtCoordinateSearchRankLess(ordered[i], ordered[j], true)
	})
	for i := range fullRanks {
		if ordered[i].index != fullRanks[i].index {
			return fmt.Errorf("full order is not deterministic protected ranking at position %d: got index %d, want %d", i, fullRanks[i].index, ordered[i].index)
		}
	}
	return nil
}

func validateAOQTProtectedCoordinateSearchAuditOrder(items []aoqtCoordinateSearchAuditOrder, angleCount int, wantComponents []string, allowZero bool) ([]aoqtCoordinateSearchRank, error) {
	if len(items) == 0 {
		return nil, fmt.Errorf("order must be non-empty")
	}
	seen := make(map[int]struct{}, len(items))
	ranks := make([]aoqtCoordinateSearchRank, len(items))
	for i, item := range items {
		if item.Index < 0 || item.Index >= angleCount {
			return nil, fmt.Errorf("order[%d] index = %d outside [0,%d)", i, item.Index, angleCount)
		}
		if _, ok := seen[item.Index]; ok {
			return nil, fmt.Errorf("order[%d] repeats index %d", i, item.Index)
		}
		seen[item.Index] = struct{}{}
		if !isFinite32(item.Abs) || item.Abs < 0 {
			return nil, fmt.Errorf("order[%d] abs must be finite and non-negative", i)
		}
		if item.Abs == 0 && !allowZero {
			return nil, fmt.Errorf("order[%d] abs must be finite and positive", i)
		}
		if !isFinite32(item.AggregateAbs) || item.AggregateAbs < 0 {
			return nil, fmt.Errorf("order[%d] aggregate_abs must be finite and non-negative", i)
		}
		if !isFinite32(item.AggregateGradient) {
			return nil, fmt.Errorf("order[%d] aggregate_gradient must be finite", i)
		}
		if item.AggregateAbs != float32(math.Abs(float64(item.AggregateGradient))) {
			return nil, fmt.Errorf("order[%d] aggregate_abs = %.9g, want |aggregate_gradient| %.9g", i, item.AggregateAbs, math.Abs(float64(item.AggregateGradient)))
		}
		if item.Abs > 0 && item.PrimaryDirection != -1 && item.PrimaryDirection != 1 {
			return nil, fmt.Errorf("order[%d] primary_direction = %.9g, want -1 or 1", i, item.PrimaryDirection)
		}
		if item.Abs == 0 && item.PrimaryDirection != 0 {
			return nil, fmt.Errorf("order[%d] zero-abs primary_direction = %.9g, want 0", i, item.PrimaryDirection)
		}
		if len(item.ProtectedDirectionalDerivatives) != len(wantComponents) {
			return nil, fmt.Errorf("order[%d] protected derivative count = %d, want %d", i, len(item.ProtectedDirectionalDerivatives), len(wantComponents))
		}
		conflicts := 0
		var maxAscent float32
		for pi, derivative := range item.ProtectedDirectionalDerivatives {
			if !isFinite32(derivative) {
				return nil, fmt.Errorf("order[%d] protected derivative[%d] is not finite", i, pi)
			}
			if derivative > 0 {
				conflicts++
				if derivative > maxAscent {
					maxAscent = derivative
				}
			}
		}
		if item.ProtectedConflictCount != conflicts {
			return nil, fmt.Errorf("order[%d] protected conflict count = %d, want %d", i, item.ProtectedConflictCount, conflicts)
		}
		if item.ProtectedMaxDirectionalDerivative != maxAscent {
			return nil, fmt.Errorf("order[%d] protected max derivative = %.9g, want %.9g", i, item.ProtectedMaxDirectionalDerivative, maxAscent)
		}
		if item.Abs == 0 && (item.ProtectedConflictCount != 0 || item.ProtectedMaxDirectionalDerivative != 0 || item.GuardConflict) {
			return nil, fmt.Errorf("order[%d] zero-abs rank has non-zero conflict metadata", i)
		}
		q3Gradient := -item.PrimaryDirection * item.Abs
		wantGuardConflict := q3Gradient*item.AggregateGradient < 0 || conflicts > 0
		if item.GuardConflict != wantGuardConflict {
			return nil, fmt.Errorf("order[%d] guard conflict = %t, want %t from aggregate/protected derivatives", i, item.GuardConflict, wantGuardConflict)
		}
		ranks[i] = aoqtCoordinateSearchRank{
			index:                             item.Index,
			abs:                               item.Abs,
			guardConflict:                     item.GuardConflict,
			aggregateAbs:                      item.AggregateAbs,
			aggregateGradient:                 item.AggregateGradient,
			primaryDirection:                  item.PrimaryDirection,
			protectedConflictCount:            item.ProtectedConflictCount,
			protectedMaxDirectionalDerivative: item.ProtectedMaxDirectionalDerivative,
			protectedDirectionalDerivatives:   append([]float32(nil), item.ProtectedDirectionalDerivatives...),
		}
	}
	return ranks, nil
}

func aoqtFloat32SlicesEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func aoqtIntSlicesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
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
	if !isFinite32(plan.LearningRate) || plan.LearningRate <= 0 {
		return fmt.Errorf("AOQT plan learning_rate must be finite and positive")
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
