package eosruntime

import (
	"fmt"
	"math"
	"math/rand"

	"m31labs.dev/turboquant"
)

type AOQTSidecarTrainConfig struct {
	Dim          int
	Stages       int
	PairingSeed  int64
	WorkplanSeed int64
	PlanOnly     bool
	MaxSteps     int
	LearningRate float32
	Beta1        float32
	Beta2        float32
	Epsilon      float32
	AngleCap     float32
	MaxAngleCap  float32
}

type AOQTSidecarWorkPlan struct {
	Dim            int    `json:"dim"`
	Stages         int    `json:"stages"`
	PairsPerStage  int    `json:"pairs_per_stage"`
	AngleCount     int    `json:"angle_count"`
	RowCount       int    `json:"row_count"`
	PairCount      int    `json:"pair_count"`
	StepCount      int    `json:"step_count"`
	PairingSeed    int64  `json:"pairing_seed"`
	WorkplanSeed   int64  `json:"workplan_seed"`
	PairingsSHA256 string `json:"pairings_sha256"`
	PlanOnly       bool   `json:"plan_only"`
}

type AOQTSidecarTrainSummary struct {
	Plan                       AOQTSidecarWorkPlan            `json:"plan"`
	ObjectiveContract          AOQTSidecarObjectiveContract   `json:"objective_contract"`
	Steps                      int                            `json:"steps"`
	InitialLoss                float32                        `json:"initial_loss"`
	FinalLoss                  float32                        `json:"final_loss"`
	InitialObjectiveComponents AOQTSidecarObjectiveComponents `json:"initial_objective_components"`
	FinalObjectiveComponents   AOQTSidecarObjectiveComponents `json:"final_objective_components"`
	InitialObjectiveActivation AOQTSidecarObjectiveActivation `json:"initial_objective_activation"`
	FinalObjectiveActivation   AOQTSidecarObjectiveActivation `json:"final_objective_activation"`
	AngleL2                    float32                        `json:"angle_l2"`
	AngleMaxAbs                float32                        `json:"angle_max_abs"`
	AnglesSHA256               string                         `json:"angles_sha256"`
	DenseMaxAbsDelta           float64                        `json:"dense_max_abs_delta"`
	QualityClaim               bool                           `json:"quality_claim"`
}

type AOQTSidecarObjectiveInput struct {
	Row        AOQTSidecarCalibrationRow
	Query      []float32
	Candidates [][]float32
}

type AOQTSidecarObjectiveResult struct {
	Loss           float32
	Components     AOQTSidecarObjectiveComponents
	QueryGrad      []float32
	CandidateGrads [][]float32
	Activation     AOQTSidecarObjectiveActivation
}

type AOQTSidecarObjectiveComponents struct {
	Q3Gain          float32 `json:"q3_gain"`
	Q3OrderGuard    float32 `json:"q3_order_guard"`
	Q3ScoreDistill  float32 `json:"q3_score_distill"`
	Q5OrderGuard    float32 `json:"q5_order_guard"`
	Q5ScoreDistill  float32 `json:"q5_score_distill"`
	NFBoundaryGuard float32 `json:"nf_boundary_guard"`
}

func (c *AOQTSidecarObjectiveComponents) Add(other AOQTSidecarObjectiveComponents) {
	c.Q3Gain += other.Q3Gain
	c.Q3OrderGuard += other.Q3OrderGuard
	c.Q3ScoreDistill += other.Q3ScoreDistill
	c.Q5OrderGuard += other.Q5OrderGuard
	c.Q5ScoreDistill += other.Q5ScoreDistill
	c.NFBoundaryGuard += other.NFBoundaryGuard
}

func (c AOQTSidecarObjectiveComponents) Sum() float32 {
	return c.Q3Gain + c.Q3OrderGuard + c.Q3ScoreDistill + c.Q5OrderGuard + c.Q5ScoreDistill + c.NFBoundaryGuard
}

type AOQTSidecarObjectiveActivation struct {
	Q3GainEligiblePairs         int `json:"q3_gain_eligible_pairs"`
	Q3GainContributingPairs     int `json:"q3_gain_contributing_pairs"`
	Q3OrderGuardPairs           int `json:"q3_order_guard_pairs"`
	Q3OrderGuardContributing    int `json:"q3_order_guard_contributing_pairs"`
	Q3ScoreDistillCount         int `json:"q3_score_distill_count"`
	Q5OrderGuardPairs           int `json:"q5_order_guard_pairs"`
	Q5OrderGuardContributing    int `json:"q5_order_guard_contributing_pairs"`
	Q5ScoreDistillCount         int `json:"q5_score_distill_count"`
	NFBoundaryGuardPairs        int `json:"nf_boundary_guard_pairs"`
	NFBoundaryGuardContributing int `json:"nf_boundary_guard_contributing_pairs"`
}

func (a *AOQTSidecarObjectiveActivation) Add(other AOQTSidecarObjectiveActivation) {
	a.Q3GainEligiblePairs += other.Q3GainEligiblePairs
	a.Q3GainContributingPairs += other.Q3GainContributingPairs
	a.Q3OrderGuardPairs += other.Q3OrderGuardPairs
	a.Q3OrderGuardContributing += other.Q3OrderGuardContributing
	a.Q3ScoreDistillCount += other.Q3ScoreDistillCount
	a.Q5OrderGuardPairs += other.Q5OrderGuardPairs
	a.Q5OrderGuardContributing += other.Q5OrderGuardContributing
	a.Q5ScoreDistillCount += other.Q5ScoreDistillCount
	a.NFBoundaryGuardPairs += other.NFBoundaryGuardPairs
	a.NFBoundaryGuardContributing += other.NFBoundaryGuardContributing
}

type AOQTSidecarVectorObjective interface {
	EvaluateAOQT(input AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error)
}

type AOQTSidecarPreparedIPObjectiveContractProvider interface {
	AOQTPreparedIPObjectiveConfig() AOQTSidecarPreparedIPObjectiveConfig
}

type AOQTSidecarPreparedIPObjectiveConfig struct {
	Dim              int
	TurboQuantSeed   int64
	GainBit          int
	Q3GuardBit       int
	Q5GuardBit       int
	GainCutoff       int
	GainTau          float32
	GainMargin       float32
	GuardTau         float32
	GuardMargin      float32
	ScoreDistillTau  float32
	NFBoundarySource string
}

func (cfg AOQTSidecarPreparedIPObjectiveConfig) ObjectiveContract(weights AOQTSidecarRowWeights) AOQTSidecarObjectiveContract {
	cfg = normalizedAOQTPreparedIPObjectiveConfig(cfg)
	return AOQTSidecarObjectiveContract{
		Dim:              cfg.Dim,
		TurboQuantSeed:   cfg.TurboQuantSeed,
		GainBit:          cfg.GainBit,
		Q3GuardBit:       cfg.Q3GuardBit,
		Q5GuardBit:       cfg.Q5GuardBit,
		ScoreSurface:     AOQTSidecarPreparedIPScoreSurface,
		GainCutoff:       cfg.GainCutoff,
		GainTau:          cfg.GainTau,
		GainMargin:       cfg.GainMargin,
		GuardTau:         cfg.GuardTau,
		GuardMargin:      cfg.GuardMargin,
		ScoreDistillTau:  cfg.ScoreDistillTau,
		NFBoundarySource: cfg.NFBoundarySource,
		WeightSums:       weights,
	}
}

type AOQTSidecarPreparedIPObjective struct {
	config AOQTSidecarPreparedIPObjectiveConfig
}

func NewAOQTSidecarPreparedIPObjective(cfg AOQTSidecarPreparedIPObjectiveConfig) (AOQTSidecarPreparedIPObjective, error) {
	cfg = normalizedAOQTPreparedIPObjectiveConfig(cfg)
	if err := validateAOQTPreparedIPObjectiveConfig(cfg); err != nil {
		return AOQTSidecarPreparedIPObjective{}, err
	}
	return AOQTSidecarPreparedIPObjective{config: cfg}, nil
}

func (o AOQTSidecarPreparedIPObjective) AOQTPreparedIPObjectiveConfig() AOQTSidecarPreparedIPObjectiveConfig {
	return o.config
}

func (o AOQTSidecarPreparedIPObjective) EvaluateAOQT(input AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error) {
	cfg := o.config
	if err := validateAOQTPreparedIPObjectiveConfig(cfg); err != nil {
		return AOQTSidecarObjectiveResult{}, err
	}
	if len(input.Query) != cfg.Dim {
		return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT prepared-IP query dim = %d, want %d", len(input.Query), cfg.Dim)
	}
	if len(input.Candidates) == 0 {
		return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT prepared-IP candidates are required")
	}
	row := input.Row
	n := len(input.Candidates)
	if len(row.CandidateDocIDs) != n || len(row.QrelGains) != n {
		return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT prepared-IP row metadata length mismatch")
	}
	if len(row.AnchorScores.Q3) != n || len(row.AnchorScores.Q5) != n || len(row.AnchorRanks.Q3) != n || len(row.AnchorRanks.Q5) != n {
		return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT prepared-IP anchor q3/q5 metadata length mismatch")
	}
	for i, candidate := range input.Candidates {
		if len(candidate) != cfg.Dim {
			return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT prepared-IP candidate %d dim = %d, want %d", i, len(candidate), cfg.Dim)
		}
	}
	result := AOQTSidecarObjectiveResult{
		QueryGrad:      make([]float32, cfg.Dim),
		CandidateGrads: make([][]float32, n),
	}
	for i := range result.CandidateGrads {
		result.CandidateGrads[i] = make([]float32, cfg.Dim)
	}
	q3Gain := newAOQTPreparedIPSurface(input.Query, input.Candidates, cfg.Dim, cfg.GainBit, cfg.TurboQuantSeed)
	if row.Weights.Q3Gain > 0 {
		loss, err := topkLambdaNDCGLossAndGrad(q3Gain.scores, row.QrelGains, row.CandidateDocIDs, cfg.GainCutoff, cfg.GainTau, cfg.GainMargin, row.EligiblePairMask)
		if err != nil {
			return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT q3 prepared-IP LambdaNDCG: %w", err)
		}
		if loss.EligiblePairs == 0 || loss.ContributingPairs == 0 {
			return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT q3_gain weight %.9g has no active contributing prepared-IP LambdaNDCG pairs", row.Weights.Q3Gain)
		}
		scale := row.Weights.Q3Gain
		component := loss.Loss * scale
		result.Loss += component
		result.Components.Q3Gain += component
		result.Activation.Q3GainEligiblePairs += loss.EligiblePairs
		result.Activation.Q3GainContributingPairs += loss.ContributingPairs
		q3Gain.accumulateScoreGrads(loss.Grad, scale, result.QueryGrad, result.CandidateGrads)
	}
	q3Guard := q3Gain
	if cfg.Q3GuardBit != cfg.GainBit {
		q3Guard = newAOQTPreparedIPSurface(input.Query, input.Candidates, cfg.Dim, cfg.Q3GuardBit, cfg.TurboQuantSeed)
	}
	if row.Weights.Q3OrderGuard > 0 {
		loss, grads, pairs, contributing := aoqtAnchorOrderGuardLossAndGrad(q3Guard.scores, row.AnchorRanks.Q3, row.EligiblePairMask, cfg.GuardTau, cfg.GuardMargin)
		if pairs == 0 || contributing == 0 {
			return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT q3_order_guard weight %.9g has no active contributing prepared-IP guard pairs", row.Weights.Q3OrderGuard)
		}
		scale := row.Weights.Q3OrderGuard
		component := loss * scale
		result.Loss += component
		result.Components.Q3OrderGuard += component
		result.Activation.Q3OrderGuardPairs += pairs
		result.Activation.Q3OrderGuardContributing += contributing
		q3Guard.accumulateScoreGrads(grads, scale, result.QueryGrad, result.CandidateGrads)
	}
	if row.Weights.Q3ScoreDistill > 0 {
		loss, grads, count := aoqtCenteredScoreDistillLossAndGrad(q3Guard.scores, row.AnchorScores.Q3, cfg.ScoreDistillTau)
		if count == 0 {
			return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT q3_score_distill weight %.9g has no prepared-IP score coverage", row.Weights.Q3ScoreDistill)
		}
		scale := row.Weights.Q3ScoreDistill
		component := loss * scale
		result.Loss += component
		result.Components.Q3ScoreDistill += component
		result.Activation.Q3ScoreDistillCount += count
		q3Guard.accumulateScoreGrads(grads, scale, result.QueryGrad, result.CandidateGrads)
	}
	if row.Weights.NFBoundaryGuard > 0 {
		loss, grads, pairs, contributing := aoqtNFBoundaryGuardLossAndGrad(q3Guard.scores, row.AnchorRanks.Q3, row.CandidateSources, cfg.NFBoundarySource, row.EligiblePairMask, cfg.GuardTau, cfg.GuardMargin)
		if pairs == 0 || contributing == 0 {
			return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT nf_boundary_guard weight %.9g has no active contributing prepared-IP guard pairs", row.Weights.NFBoundaryGuard)
		}
		scale := row.Weights.NFBoundaryGuard
		component := loss * scale
		result.Loss += component
		result.Components.NFBoundaryGuard += component
		result.Activation.NFBoundaryGuardPairs += pairs
		result.Activation.NFBoundaryGuardContributing += contributing
		q3Guard.accumulateScoreGrads(grads, scale, result.QueryGrad, result.CandidateGrads)
	}
	if row.Weights.Q5OrderGuard > 0 || row.Weights.Q5ScoreDistill > 0 {
		q5 := newAOQTPreparedIPSurface(input.Query, input.Candidates, cfg.Dim, cfg.Q5GuardBit, cfg.TurboQuantSeed)
		if row.Weights.Q5OrderGuard > 0 {
			loss, grads, pairs, contributing := aoqtAnchorOrderGuardLossAndGrad(q5.scores, row.AnchorRanks.Q5, row.EligiblePairMask, cfg.GuardTau, cfg.GuardMargin)
			if pairs == 0 || contributing == 0 {
				return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT q5_order_guard weight %.9g has no active contributing prepared-IP guard pairs", row.Weights.Q5OrderGuard)
			}
			scale := row.Weights.Q5OrderGuard
			component := loss * scale
			result.Loss += component
			result.Components.Q5OrderGuard += component
			result.Activation.Q5OrderGuardPairs += pairs
			result.Activation.Q5OrderGuardContributing += contributing
			q5.accumulateScoreGrads(grads, scale, result.QueryGrad, result.CandidateGrads)
		}
		if row.Weights.Q5ScoreDistill > 0 {
			loss, grads, count := aoqtCenteredScoreDistillLossAndGrad(q5.scores, row.AnchorScores.Q5, cfg.ScoreDistillTau)
			if count == 0 {
				return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT q5_score_distill weight %.9g has no prepared-IP score coverage", row.Weights.Q5ScoreDistill)
			}
			scale := row.Weights.Q5ScoreDistill
			component := loss * scale
			result.Loss += component
			result.Components.Q5ScoreDistill += component
			result.Activation.Q5ScoreDistillCount += count
			q5.accumulateScoreGrads(grads, scale, result.QueryGrad, result.CandidateGrads)
		}
	}
	if !isFinite32(result.Loss) {
		return AOQTSidecarObjectiveResult{}, fmt.Errorf("AOQT prepared-IP loss must be finite")
	}
	return result, nil
}

type AOQTSidecarTrainer struct {
	config   AOQTSidecarTrainConfig
	topology AOQTGivensTransform
	angles   []float32
	adamM    []float32
	adamV    []float32
	step     int
}

func NewAOQTSidecarTrainer(cfg AOQTSidecarTrainConfig) (*AOQTSidecarTrainer, error) {
	cfg = normalizedAOQTSidecarTrainConfig(cfg)
	if err := validateAOQTSidecarTrainConfig(cfg); err != nil {
		return nil, err
	}
	topology, err := NewAOQTGivensIdentityTopology(cfg.Dim, cfg.Stages, cfg.PairingSeed, cfg.AngleCap)
	if err != nil {
		return nil, err
	}
	angleCount := countAOQTAngles(topology)
	return &AOQTSidecarTrainer{
		config:   cfg,
		topology: topology,
		angles:   make([]float32, angleCount),
		adamM:    make([]float32, angleCount),
		adamV:    make([]float32, angleCount),
	}, nil
}

func NewAOQTGivensIdentityTopology(dim, stages int, seed int64, angleCap float32) (AOQTGivensTransform, error) {
	if dim != AOQTSidecarDim {
		return AOQTGivensTransform{}, fmt.Errorf("AOQT topology dim = %d, want %d", dim, AOQTSidecarDim)
	}
	if stages != AOQTSidecarStages {
		return AOQTGivensTransform{}, fmt.Errorf("AOQT topology stages = %d, want %d", stages, AOQTSidecarStages)
	}
	if seed == 0 {
		return AOQTGivensTransform{}, fmt.Errorf("AOQT topology seed is required")
	}
	if angleCap <= 0 {
		angleCap = AOQTSidecarDefaultAngleCap
	}
	if angleCap > AOQTSidecarHardMaxAngleCap {
		return AOQTGivensTransform{}, fmt.Errorf("AOQT angle_cap %.8g exceeds hard max %.8g", angleCap, AOQTSidecarHardMaxAngleCap)
	}
	rng := rand.New(rand.NewSource(seed))
	transform := AOQTGivensTransform{
		Version:  AOQTTransformVersion,
		Kind:     EmbeddingPostPoolTransformAOQTGivens,
		Dim:      dim,
		Seed:     seed,
		AngleCap: angleCap,
		Stages:   make([]AOQTStage, stages),
	}
	for stage := range transform.Stages {
		perm := rng.Perm(dim)
		pairs := make([][2]int, dim/2)
		for i := range pairs {
			a, b := perm[2*i], perm[2*i+1]
			if a > b {
				a, b = b, a
			}
			pairs[i] = [2]int{a, b}
		}
		transform.Stages[stage] = AOQTStage{
			Pairs:  pairs,
			Angles: make([]float32, len(pairs)),
		}
	}
	if err := transform.Validate(); err != nil {
		return AOQTGivensTransform{}, err
	}
	pairings, err := transform.PairingsSHA256()
	if err != nil {
		return AOQTGivensTransform{}, err
	}
	angles, err := transform.AnglesSHA256()
	if err != nil {
		return AOQTGivensTransform{}, err
	}
	orth, err := transform.OrthogonalityFrobeniusPerDim()
	if err != nil {
		return AOQTGivensTransform{}, err
	}
	transform.Audit = AOQTAuditInfo{
		PairingsSHA256: pairings,
		AnglesSHA256:   angles,
		Orthogonality:  &orth,
	}
	return transform, nil
}

func (t *AOQTSidecarTrainer) Plan(set AOQTSidecarCalibrationSet) (AOQTSidecarWorkPlan, error) {
	if t == nil {
		return AOQTSidecarWorkPlan{}, fmt.Errorf("AOQT trainer is nil")
	}
	if err := set.Validate(); err != nil {
		return AOQTSidecarWorkPlan{}, err
	}
	pairings, err := t.topology.PairingsSHA256()
	if err != nil {
		return AOQTSidecarWorkPlan{}, err
	}
	if set.Manifest.Topology.PairingsSHA256 != pairings {
		return AOQTSidecarWorkPlan{}, fmt.Errorf("AOQT trainer pairings do not match manifest topology")
	}
	var pairCount int
	for _, row := range set.Rows {
		pairCount += countEligibleAOQTPairs(row)
	}
	steps := t.config.MaxSteps
	if t.config.PlanOnly {
		steps = 0
	}
	return AOQTSidecarWorkPlan{
		Dim:            t.config.Dim,
		Stages:         t.config.Stages,
		PairsPerStage:  AOQTSidecarPairsPerStage,
		AngleCount:     len(t.angles),
		RowCount:       len(set.Rows),
		PairCount:      pairCount,
		StepCount:      steps,
		PairingSeed:    t.config.PairingSeed,
		WorkplanSeed:   t.config.WorkplanSeed,
		PairingsSHA256: pairings,
		PlanOnly:       t.config.PlanOnly,
	}, nil
}

func (t *AOQTSidecarTrainer) Fit(set AOQTSidecarCalibrationSet, objective AOQTSidecarVectorObjective) (AOQTSidecarTrainSummary, error) {
	if t == nil {
		return AOQTSidecarTrainSummary{}, fmt.Errorf("AOQT trainer is nil")
	}
	plan, err := t.Plan(set)
	if err != nil {
		return AOQTSidecarTrainSummary{}, err
	}
	summary := AOQTSidecarTrainSummary{Plan: plan, ObjectiveContract: set.Manifest.ObjectiveContract, QualityClaim: false}
	if t.config.PlanOnly {
		angles, err := t.Transform().AnglesSHA256()
		if err != nil {
			return summary, err
		}
		summary.AnglesSHA256 = angles
		return summary, nil
	}
	if objective == nil {
		return summary, fmt.Errorf("AOQT objective is required for non-plan training")
	}
	if err := validateAOQTFitObjectiveContract(set.Manifest.ObjectiveContract, objective); err != nil {
		return summary, err
	}
	var initialSet bool
	for step := 0; step < t.config.MaxSteps; step++ {
		rows := deterministicAOQTRowOrder(set.Rows, t.config.WorkplanSeed, step)
		loss, grad, activation, components, err := t.lossAndAngleGrad(rows, objective)
		if err != nil {
			return summary, err
		}
		if !initialSet {
			summary.InitialLoss = loss
			summary.InitialObjectiveActivation = activation
			summary.InitialObjectiveComponents = components
			initialSet = true
		}
		if err := t.applyAdam(grad); err != nil {
			return summary, err
		}
		summary.Steps++
	}
	finalLoss, _, finalActivation, finalComponents, err := t.lossAndAngleGrad(set.Rows, objective)
	if err != nil {
		return summary, err
	}
	summary.FinalLoss = finalLoss
	summary.FinalObjectiveActivation = finalActivation
	summary.FinalObjectiveComponents = finalComponents
	finalTransform := t.Transform()
	angles, err := finalTransform.AnglesSHA256()
	if err != nil {
		return summary, err
	}
	summary.AnglesSHA256 = angles
	summary.AngleL2, summary.AngleMaxAbs = aoqtAngleStats(t.angles)
	dense, err := DenseInvariantMaxAbsDelta(finalTransform, set.Rows)
	if err != nil {
		return summary, err
	}
	summary.DenseMaxAbsDelta = dense
	return summary, nil
}

func validateAOQTFitObjectiveContract(contract AOQTSidecarObjectiveContract, objective AOQTSidecarVectorObjective) error {
	provider, ok := objective.(AOQTSidecarPreparedIPObjectiveContractProvider)
	if !ok {
		return fmt.Errorf("AOQT non-plan training requires objective contract provider")
	}
	got := provider.AOQTPreparedIPObjectiveConfig().ObjectiveContract(contract.WeightSums)
	if !aoqtObjectiveContractsEqual(got, contract) {
		return fmt.Errorf("AOQT objective config does not match calibration manifest objective_contract")
	}
	return nil
}

func aoqtObjectiveContractsEqual(a, b AOQTSidecarObjectiveContract) bool {
	return a.Dim == b.Dim &&
		a.TurboQuantSeed == b.TurboQuantSeed &&
		a.GainBit == b.GainBit &&
		a.Q3GuardBit == b.Q3GuardBit &&
		a.Q5GuardBit == b.Q5GuardBit &&
		a.ScoreSurface == b.ScoreSurface &&
		a.GainCutoff == b.GainCutoff &&
		float32Near(a.GainTau, b.GainTau, 1e-8) &&
		float32Near(a.GainMargin, b.GainMargin, 1e-8) &&
		float32Near(a.GuardTau, b.GuardTau, 1e-8) &&
		float32Near(a.GuardMargin, b.GuardMargin, 1e-8) &&
		float32Near(a.ScoreDistillTau, b.ScoreDistillTau, 1e-8) &&
		a.NFBoundarySource == b.NFBoundarySource &&
		aoqtRowWeightsNearEqual(a.WeightSums, b.WeightSums)
}

func (t *AOQTSidecarTrainer) Transform() AOQTGivensTransform {
	transform := t.topology
	transform.Stages = append([]AOQTStage(nil), t.topology.Stages...)
	offset := 0
	for si := range transform.Stages {
		transform.Stages[si].Pairs = append([][2]int(nil), t.topology.Stages[si].Pairs...)
		transform.Stages[si].Angles = append([]float32(nil), t.angles[offset:offset+len(transform.Stages[si].Pairs)]...)
		offset += len(transform.Stages[si].Pairs)
	}
	angles, err := transform.AnglesSHA256()
	if err == nil {
		transform.Audit.AnglesSHA256 = angles
	}
	orth, err := transform.OrthogonalityFrobeniusPerDim()
	if err == nil {
		transform.Audit.Orthogonality = &orth
	}
	return transform
}

func (t *AOQTSidecarTrainer) Angles() []float32 {
	return append([]float32(nil), t.angles...)
}

func (t *AOQTSidecarTrainer) ProjectAngles() error {
	if t == nil {
		return fmt.Errorf("AOQT trainer is nil")
	}
	return projectAOQTAngles(t.angles, t.config.AngleCap)
}

func (t *AOQTSidecarTrainer) SetAnglesForTest(angles []float32) error {
	if len(angles) != len(t.angles) {
		return fmt.Errorf("AOQT angle count = %d, want %d", len(angles), len(t.angles))
	}
	copy(t.angles, angles)
	return t.ProjectAngles()
}

func (t *AOQTSidecarTrainer) lossAndAngleGrad(rows []AOQTSidecarCalibrationRow, objective AOQTSidecarVectorObjective) (float32, []float32, AOQTSidecarObjectiveActivation, AOQTSidecarObjectiveComponents, error) {
	totalGrad := make([]float32, len(t.angles))
	totalLoss := float32(0)
	var totalActivation AOQTSidecarObjectiveActivation
	var totalComponents AOQTSidecarObjectiveComponents
	transform := t.Transform()
	for _, row := range rows {
		query, candidates, err := transformAOQTRow(transform, row)
		if err != nil {
			return 0, nil, totalActivation, totalComponents, err
		}
		result, err := objective.EvaluateAOQT(AOQTSidecarObjectiveInput{
			Row:        aoqtObjectiveRowView(row),
			Query:      append([]float32(nil), query...),
			Candidates: cloneAOQTVectors(candidates),
		})
		if err != nil {
			return 0, nil, totalActivation, totalComponents, err
		}
		if !isFinite32(result.Loss) {
			return 0, nil, totalActivation, totalComponents, fmt.Errorf("AOQT objective loss must be finite")
		}
		if err := validateAOQTObjectiveComponents(result.Components, "AOQT objective components"); err != nil {
			return 0, nil, totalActivation, totalComponents, err
		}
		if err := validateAOQTLossMatchesComponents(result.Loss, result.Components, "AOQT objective"); err != nil {
			return 0, nil, totalActivation, totalComponents, err
		}
		if len(result.QueryGrad) != transform.Dim {
			return 0, nil, totalActivation, totalComponents, fmt.Errorf("AOQT objective query grad dim = %d, want %d", len(result.QueryGrad), transform.Dim)
		}
		if len(result.CandidateGrads) != len(row.CandidateVectors) {
			return 0, nil, totalActivation, totalComponents, fmt.Errorf("AOQT objective candidate grad count = %d, want %d", len(result.CandidateGrads), len(row.CandidateVectors))
		}
		totalLoss += result.Loss
		totalActivation.Add(result.Activation)
		totalComponents.Add(result.Components)
		if err := accumulateAOQTVectorAngleGrad(transform, row.QueryVector, result.QueryGrad, totalGrad); err != nil {
			return 0, nil, totalActivation, totalComponents, err
		}
		for i, grad := range result.CandidateGrads {
			if len(grad) != transform.Dim {
				return 0, nil, totalActivation, totalComponents, fmt.Errorf("AOQT objective candidate grad %d dim = %d, want %d", i, len(grad), transform.Dim)
			}
			if err := accumulateAOQTVectorAngleGrad(transform, row.CandidateVectors[i], grad, totalGrad); err != nil {
				return 0, nil, totalActivation, totalComponents, err
			}
		}
	}
	return totalLoss, totalGrad, totalActivation, totalComponents, nil
}

func validateAOQTObjectiveComponents(components AOQTSidecarObjectiveComponents, label string) error {
	for _, item := range []struct {
		name  string
		value float32
	}{
		{"q3_gain", components.Q3Gain},
		{"q3_order_guard", components.Q3OrderGuard},
		{"q3_score_distill", components.Q3ScoreDistill},
		{"q5_order_guard", components.Q5OrderGuard},
		{"q5_score_distill", components.Q5ScoreDistill},
		{"nf_boundary_guard", components.NFBoundaryGuard},
	} {
		if !isFinite32(item.value) || item.value < 0 {
			return fmt.Errorf("%s.%s must be finite and non-negative", label, item.name)
		}
	}
	return nil
}

func validateAOQTLossMatchesComponents(loss float32, components AOQTSidecarObjectiveComponents, label string) error {
	if !isFinite32(loss) || loss < 0 {
		return fmt.Errorf("%s loss must be finite and non-negative", label)
	}
	sum := components.Sum()
	tolerance := float32(1e-5)
	if abs := float32(math.Abs(float64(loss))); abs > 1 {
		tolerance *= abs
	}
	if abs := float32(math.Abs(float64(sum))); abs > 1 && abs*1e-5 > tolerance {
		tolerance = abs * 1e-5
	}
	if float32(math.Abs(float64(loss-sum))) > tolerance {
		return fmt.Errorf("%s loss %.9g must equal component sum %.9g within %.9g", label, loss, sum, tolerance)
	}
	return nil
}

func (t *AOQTSidecarTrainer) applyAdam(grad []float32) error {
	if len(grad) != len(t.angles) {
		return fmt.Errorf("AOQT gradient count = %d, want %d", len(grad), len(t.angles))
	}
	t.step++
	b1, b2 := t.config.Beta1, t.config.Beta2
	lr := t.config.LearningRate
	eps := t.config.Epsilon
	for i, g := range grad {
		if !isFinite32(g) {
			return fmt.Errorf("AOQT gradient %d is not finite", i)
		}
		t.adamM[i] = b1*t.adamM[i] + (1-b1)*g
		t.adamV[i] = b2*t.adamV[i] + (1-b2)*g*g
		mhat := t.adamM[i] / (1 - float32(math.Pow(float64(b1), float64(t.step))))
		vhat := t.adamV[i] / (1 - float32(math.Pow(float64(b2), float64(t.step))))
		t.angles[i] -= lr * mhat / (float32(math.Sqrt(float64(vhat))) + eps)
	}
	return t.ProjectAngles()
}

func transformAOQTRow(transform AOQTGivensTransform, row AOQTSidecarCalibrationRow) ([]float32, [][]float32, error) {
	query, err := transform.ApplyVector(row.QueryVector)
	if err != nil {
		return nil, nil, err
	}
	candidates := make([][]float32, len(row.CandidateVectors))
	for i := range row.CandidateVectors {
		candidates[i], err = transform.ApplyVector(row.CandidateVectors[i])
		if err != nil {
			return nil, nil, fmt.Errorf("candidate %d: %w", i, err)
		}
	}
	return query, candidates, nil
}

func accumulateAOQTVectorAngleGrad(transform AOQTGivensTransform, input, outputGrad []float32, angleGrad []float32) error {
	if len(input) != transform.Dim || len(outputGrad) != transform.Dim {
		return fmt.Errorf("AOQT vector/grad dims must match transform dim")
	}
	if err := validateAOQTFiniteVector(outputGrad, "AOQT output grad"); err != nil {
		return err
	}
	activations, err := aoqtForwardActivations(transform, input)
	if err != nil {
		return err
	}
	grad := append([]float32(nil), outputGrad...)
	angleIndex := countAOQTAngles(transform)
	for si := len(transform.Stages) - 1; si >= 0; si-- {
		stage := transform.Stages[si]
		for pi := len(stage.Pairs) - 1; pi >= 0; pi-- {
			angleIndex--
			pair := stage.Pairs[pi]
			a, b := pair[0], pair[1]
			x := activations[angleIndex][a]
			y := activations[angleIndex][b]
			theta := float64(stage.Angles[pi])
			c, s := float32(math.Cos(theta)), float32(math.Sin(theta))
			ga, gb := grad[a], grad[b]
			angleGrad[angleIndex] += ga*(-s*x-c*y) + gb*(c*x-s*y)
			grad[a] = c*ga + s*gb
			grad[b] = -s*ga + c*gb
		}
	}
	return nil
}

func aoqtForwardActivations(transform AOQTGivensTransform, input []float32) ([][]float32, error) {
	if err := validateAOQTGivensTrainingSnapshot(transform); err != nil {
		return nil, err
	}
	if len(input) != transform.Dim {
		return nil, fmt.Errorf("AOQT input dim = %d, want %d", len(input), transform.Dim)
	}
	if err := validateAOQTFiniteVector(input, "AOQT input"); err != nil {
		return nil, err
	}
	vec := append([]float32(nil), input...)
	activations := make([][]float32, 0, countAOQTAngles(transform))
	for _, stage := range transform.Stages {
		for i, pair := range stage.Pairs {
			activations = append(activations, append([]float32(nil), vec...))
			a, b := pair[0], pair[1]
			theta := float64(stage.Angles[i])
			c, s := float32(math.Cos(theta)), float32(math.Sin(theta))
			x, y := vec[a], vec[b]
			vec[a] = c*x - s*y
			vec[b] = s*x + c*y
		}
	}
	return activations, nil
}

func DenseInvariantMaxAbsDelta(transform AOQTGivensTransform, rows []AOQTSidecarCalibrationRow) (float64, error) {
	var maxDelta float64
	for _, row := range rows {
		query, candidates, err := transformAOQTRow(transform, row)
		if err != nil {
			return 0, err
		}
		for i, candidate := range candidates {
			before := dotAOQT(row.QueryVector, row.CandidateVectors[i])
			declaredDelta := math.Abs(float64(before - row.AnchorScores.Dense[i]))
			if declaredDelta > float64(AOQTSidecarDenseAnchorTolerance) {
				return 0, fmt.Errorf("AOQT row %q anchor_scores.dense[%d] does not match pre-transform dot", row.RowID, i)
			}
			after := dotAOQT(query, candidate)
			delta := math.Abs(float64(before - after))
			if delta > maxDelta {
				maxDelta = delta
			}
		}
	}
	return maxDelta, nil
}

func normalizedAOQTSidecarTrainConfig(cfg AOQTSidecarTrainConfig) AOQTSidecarTrainConfig {
	if cfg.Dim == 0 {
		cfg.Dim = AOQTSidecarDim
	}
	if cfg.Stages == 0 {
		cfg.Stages = AOQTSidecarStages
	}
	if cfg.AngleCap == 0 {
		cfg.AngleCap = AOQTSidecarDefaultAngleCap
	}
	if cfg.MaxAngleCap == 0 {
		cfg.MaxAngleCap = AOQTSidecarHardMaxAngleCap
	}
	if cfg.LearningRate == 0 {
		cfg.LearningRate = 1e-3
	}
	if cfg.Beta1 == 0 {
		cfg.Beta1 = 0.9
	}
	if cfg.Beta2 == 0 {
		cfg.Beta2 = 0.999
	}
	if cfg.Epsilon == 0 {
		cfg.Epsilon = 1e-8
	}
	return cfg
}

func validateAOQTSidecarTrainConfig(cfg AOQTSidecarTrainConfig) error {
	if cfg.Dim != AOQTSidecarDim {
		return fmt.Errorf("AOQT train dim = %d, want %d", cfg.Dim, AOQTSidecarDim)
	}
	if cfg.Stages != AOQTSidecarStages {
		return fmt.Errorf("AOQT train stages = %d, want %d", cfg.Stages, AOQTSidecarStages)
	}
	if cfg.PairingSeed == 0 {
		return fmt.Errorf("AOQT train pairing seed is required")
	}
	if cfg.WorkplanSeed == 0 {
		return fmt.Errorf("AOQT train workplan seed is required")
	}
	if cfg.AngleCap <= 0 || !isFinite32(cfg.AngleCap) {
		return fmt.Errorf("AOQT angle cap must be finite and positive")
	}
	if cfg.MaxAngleCap <= 0 || !isFinite32(cfg.MaxAngleCap) {
		return fmt.Errorf("AOQT max angle cap must be finite and positive")
	}
	if cfg.AngleCap > AOQTSidecarDefaultAngleCap {
		return fmt.Errorf("AOQT angle cap %.8g exceeds default cap %.8g", cfg.AngleCap, AOQTSidecarDefaultAngleCap)
	}
	if cfg.AngleCap > cfg.MaxAngleCap || cfg.MaxAngleCap > AOQTSidecarHardMaxAngleCap {
		return fmt.Errorf("AOQT angle caps exceed hard max %.8g", AOQTSidecarHardMaxAngleCap)
	}
	if !cfg.PlanOnly && cfg.MaxSteps <= 0 {
		return fmt.Errorf("AOQT non-plan training requires max steps")
	}
	if cfg.PlanOnly && cfg.MaxSteps < 0 {
		return fmt.Errorf("AOQT plan max steps must be non-negative")
	}
	if cfg.LearningRate <= 0 || !isFinite32(cfg.LearningRate) {
		return fmt.Errorf("AOQT learning rate must be finite and positive")
	}
	if cfg.Beta1 <= 0 || cfg.Beta1 >= 1 || cfg.Beta2 <= 0 || cfg.Beta2 >= 1 {
		return fmt.Errorf("AOQT Adam betas must be in (0,1)")
	}
	if cfg.Epsilon <= 0 || !isFinite32(cfg.Epsilon) {
		return fmt.Errorf("AOQT Adam epsilon must be finite and positive")
	}
	return nil
}

func deterministicAOQTRowOrder(rows []AOQTSidecarCalibrationRow, seed int64, step int) []AOQTSidecarCalibrationRow {
	ordered := append([]AOQTSidecarCalibrationRow(nil), rows...)
	rng := rand.New(rand.NewSource(seed + int64(step)*7919))
	rng.Shuffle(len(ordered), func(i, j int) {
		ordered[i], ordered[j] = ordered[j], ordered[i]
	})
	return ordered
}

func aoqtObjectiveRowView(row AOQTSidecarCalibrationRow) AOQTSidecarCalibrationRow {
	view := row
	view.QueryVector = nil
	view.CandidateVectors = nil
	view.SplitProof.ExclusionIdentities = append([]string(nil), row.SplitProof.ExclusionIdentities...)
	view.CandidateDocIDs = append([]string(nil), row.CandidateDocIDs...)
	view.CandidateVectorIDs = append([]string(nil), row.CandidateVectorIDs...)
	view.CandidateVectorSHA256 = append([]string(nil), row.CandidateVectorSHA256...)
	view.QrelGains = append([]float32(nil), row.QrelGains...)
	view.CandidateSources = append([]string(nil), row.CandidateSources...)
	view.EligiblePairMask = append([][]bool(nil), row.EligiblePairMask...)
	for i := range view.EligiblePairMask {
		view.EligiblePairMask[i] = append([]bool(nil), row.EligiblePairMask[i]...)
	}
	view.AnchorScores.Dense = append([]float32(nil), row.AnchorScores.Dense...)
	view.AnchorScores.Q3 = append([]float32(nil), row.AnchorScores.Q3...)
	view.AnchorScores.Q5 = append([]float32(nil), row.AnchorScores.Q5...)
	view.AnchorRanks.Dense = append([]int(nil), row.AnchorRanks.Dense...)
	view.AnchorRanks.Q3 = append([]int(nil), row.AnchorRanks.Q3...)
	view.AnchorRanks.Q5 = append([]int(nil), row.AnchorRanks.Q5...)
	view.Extra = cloneAOQTRawMessageMap(row.Extra)
	return view
}

func validateAOQTGivensTrainingSnapshot(transform AOQTGivensTransform) error {
	if transform.Version == "" {
		return fmt.Errorf("AOQT transform version is required")
	}
	if transform.Version != AOQTTransformVersion {
		return fmt.Errorf("AOQT transform version %q is not supported, want %q", transform.Version, AOQTTransformVersion)
	}
	if transform.Kind != EmbeddingPostPoolTransformAOQTGivens {
		return fmt.Errorf("AOQT transform kind %q is not supported, want %q", transform.Kind, EmbeddingPostPoolTransformAOQTGivens)
	}
	if transform.Dim <= 0 {
		return fmt.Errorf("AOQT transform dim must be positive")
	}
	if len(transform.Stages) == 0 {
		return fmt.Errorf("AOQT transform must contain at least one stage")
	}
	for stageIndex, stage := range transform.Stages {
		if len(stage.Pairs) == 0 {
			return fmt.Errorf("AOQT stage %d has no pairs", stageIndex)
		}
		if len(stage.Angles) != len(stage.Pairs) {
			return fmt.Errorf("AOQT stage %d angle count = %d, want %d", stageIndex, len(stage.Angles), len(stage.Pairs))
		}
		seen := map[int]bool{}
		for pairIndex, pair := range stage.Pairs {
			a, b := pair[0], pair[1]
			if a < 0 || a >= transform.Dim || b < 0 || b >= transform.Dim {
				return fmt.Errorf("AOQT stage %d pair %d = [%d %d] outside dim %d", stageIndex, pairIndex, a, b, transform.Dim)
			}
			if a == b {
				return fmt.Errorf("AOQT stage %d pair %d repeats coordinate %d", stageIndex, pairIndex, a)
			}
			if seen[a] || seen[b] {
				return fmt.Errorf("AOQT stage %d coordinate appears in more than one pair", stageIndex)
			}
			seen[a], seen[b] = true, true
			angle := stage.Angles[pairIndex]
			if math.IsNaN(float64(angle)) || math.IsInf(float64(angle), 0) {
				return fmt.Errorf("AOQT stage %d angle %d is not finite", stageIndex, pairIndex)
			}
			if transform.AngleCap > 0 && float32(math.Abs(float64(angle))) > transform.AngleCap+1e-7 {
				return fmt.Errorf("AOQT stage %d angle %d exceeds angle_cap", stageIndex, pairIndex)
			}
		}
	}
	return nil
}

type aoqtPreparedIPSurface struct {
	dim                  int
	scores               []float32
	queryRaw             []float32
	queryNormalized      []float32
	queryNorm            float32
	candidateRaw         [][]float32
	candidateNormalized  [][]float32
	candidateNorms       []float32
	candidateDequantized [][]float32
}

func newAOQTPreparedIPSurface(query []float32, candidates [][]float32, dim, bitWidth int, seed int64) aoqtPreparedIPSurface {
	q := turboquant.NewIPWithSeed(dim, bitWidth, seed)
	surface := aoqtPreparedIPSurface{
		dim:                  dim,
		scores:               make([]float32, len(candidates)),
		queryRaw:             append([]float32(nil), query...),
		queryNormalized:      append([]float32(nil), query...),
		queryNorm:            vectorNorm(query),
		candidateRaw:         make([][]float32, len(candidates)),
		candidateNormalized:  make([][]float32, len(candidates)),
		candidateNorms:       make([]float32, len(candidates)),
		candidateDequantized: make([][]float32, len(candidates)),
	}
	prepared := q.PrepareQuery(surface.queryRaw)
	for i, candidate := range candidates {
		surface.candidateRaw[i] = append([]float32(nil), candidate...)
		surface.candidateNormalized[i] = append([]float32(nil), candidate...)
		surface.candidateNorms[i] = vectorNorm(candidate)
		qx := q.Quantize(surface.candidateRaw[i])
		surface.candidateDequantized[i] = q.Dequantize(qx)
		surface.scores[i] = q.InnerProductPrepared(qx, prepared)
	}
	return surface
}

func (s aoqtPreparedIPSurface) accumulateScoreGrads(scoreGrads []float32, scale float32, queryGrad []float32, candidateGrads [][]float32) {
	if scale == 0 || len(scoreGrads) != len(s.scores) {
		return
	}
	for i, grad := range scoreGrads {
		if grad == 0 {
			continue
		}
		accumulateNormalizedPrefixSTEGrad(s.queryRaw, s.queryNormalized, s.queryNorm, s.candidateDequantized[i], grad*scale, queryGrad)
		accumulateNormalizedPrefixSTEGrad(s.candidateRaw[i], s.candidateNormalized[i], s.candidateNorms[i], s.queryNormalized, grad*scale, candidateGrads[i])
	}
}

func normalizedAOQTVector(vec []float32) []float32 {
	out := make([]float32, len(vec))
	norm := vectorNorm(vec)
	if norm == 0 {
		return out
	}
	inv := 1 / norm
	for i, v := range vec {
		out[i] = v * inv
	}
	return out
}

func aoqtAnchorOrderGuardLossAndGrad(scores []float32, ranks []int, eligiblePairs [][]bool, tau, margin float32) (float32, []float32, int, int) {
	grads := make([]float32, len(scores))
	if len(scores) == 0 || len(ranks) != len(scores) {
		return 0, grads, 0, 0
	}
	var loss float32
	var pairs int
	var contributing int
	for high := range scores {
		for low := range scores {
			if high == low || ranks[high] >= ranks[low] {
				continue
			}
			if len(eligiblePairs) > 0 && !eligiblePairs[high][low] {
				continue
			}
			z := (scores[low] - scores[high] + margin) / tau
			pairScale := sigmoid32(z) / tau
			loss += softplus32(z)
			grads[high] -= pairScale
			grads[low] += pairScale
			pairs++
			if pairScale != 0 {
				contributing++
			}
		}
	}
	if pairs == 0 {
		return 0, grads, 0, 0
	}
	inv := 1 / float32(pairs)
	for i := range grads {
		grads[i] *= inv
	}
	return loss * inv, grads, pairs, contributing
}

func aoqtNFBoundaryGuardLossAndGrad(scores []float32, ranks []int, sources []string, boundarySource string, eligiblePairs [][]bool, tau, margin float32) (float32, []float32, int, int) {
	grads := make([]float32, len(scores))
	if len(scores) == 0 || len(ranks) != len(scores) || len(sources) != len(scores) {
		return 0, grads, 0, 0
	}
	var loss float32
	var pairs int
	var contributing int
	for boundary := range scores {
		if sources[boundary] != boundarySource {
			continue
		}
		for other := range scores {
			if boundary == other || ranks[boundary] >= ranks[other] {
				continue
			}
			if len(eligiblePairs) > 0 && !eligiblePairs[boundary][other] {
				continue
			}
			z := (scores[other] - scores[boundary] + margin) / tau
			pairScale := sigmoid32(z) / tau
			loss += softplus32(z)
			grads[boundary] -= pairScale
			grads[other] += pairScale
			pairs++
			if pairScale != 0 {
				contributing++
			}
		}
	}
	if pairs == 0 {
		return 0, grads, 0, 0
	}
	inv := 1 / float32(pairs)
	for i := range grads {
		grads[i] *= inv
	}
	return loss * inv, grads, pairs, contributing
}

func aoqtCenteredScoreDistillLossAndGrad(scores, anchors []float32, tau float32) (float32, []float32, int) {
	grads := make([]float32, len(scores))
	if len(scores) == 0 || len(scores) != len(anchors) {
		return 0, grads, 0
	}
	var scoreMean, anchorMean float32
	for i := range scores {
		scoreMean += scores[i]
		anchorMean += anchors[i]
	}
	invN := 1 / float32(len(scores))
	scoreMean *= invN
	anchorMean *= invN
	var loss float32
	for i := range scores {
		residual := ((scores[i] - scoreMean) - (anchors[i] - anchorMean)) / tau
		loss += 0.5 * residual * residual * invN
		grads[i] = residual * invN / tau
	}
	return loss, grads, len(scores)
}

func normalizedAOQTPreparedIPObjectiveConfig(cfg AOQTSidecarPreparedIPObjectiveConfig) AOQTSidecarPreparedIPObjectiveConfig {
	if cfg.Dim == 0 {
		cfg.Dim = AOQTSidecarDim
	}
	if cfg.GainBit == 0 {
		cfg.GainBit = AOQTSidecarDefaultGainBit
	}
	if cfg.Q3GuardBit == 0 {
		cfg.Q3GuardBit = AOQTSidecarDefaultGuardBit3
	}
	if cfg.Q5GuardBit == 0 {
		cfg.Q5GuardBit = AOQTSidecarDefaultGuardBit5
	}
	if cfg.GainCutoff == 0 {
		cfg.GainCutoff = 10
	}
	if cfg.GainTau == 0 {
		cfg.GainTau = 0.05
	}
	if cfg.GuardTau == 0 {
		cfg.GuardTau = 0.05
	}
	if cfg.ScoreDistillTau == 0 {
		cfg.ScoreDistillTau = 1
	}
	if cfg.NFBoundarySource == "" {
		cfg.NFBoundarySource = "nf_boundary80_120"
	}
	return cfg
}

func validateAOQTPreparedIPObjectiveConfig(cfg AOQTSidecarPreparedIPObjectiveConfig) error {
	if cfg.Dim != AOQTSidecarDim {
		return fmt.Errorf("AOQT prepared-IP dim = %d, want %d", cfg.Dim, AOQTSidecarDim)
	}
	if cfg.TurboQuantSeed == 0 {
		return fmt.Errorf("AOQT prepared-IP turboquant seed is required")
	}
	for _, item := range []struct {
		name string
		bit  int
	}{
		{"gain_bit", cfg.GainBit},
		{"q3_guard_bit", cfg.Q3GuardBit},
		{"q5_guard_bit", cfg.Q5GuardBit},
	} {
		if item.bit < 2 || item.bit > 8 {
			return fmt.Errorf("AOQT prepared-IP %s = %d, want 2..8", item.name, item.bit)
		}
	}
	if cfg.GainCutoff <= 0 {
		return fmt.Errorf("AOQT prepared-IP gain cutoff must be positive")
	}
	for _, item := range []struct {
		name  string
		value float32
	}{
		{"gain_tau", cfg.GainTau},
		{"guard_tau", cfg.GuardTau},
		{"score_distill_tau", cfg.ScoreDistillTau},
		{"gain_margin", cfg.GainMargin},
		{"guard_margin", cfg.GuardMargin},
	} {
		if !isFinite32(item.value) {
			return fmt.Errorf("AOQT prepared-IP %s must be finite", item.name)
		}
	}
	if cfg.GainTau <= 0 || cfg.GuardTau <= 0 || cfg.ScoreDistillTau <= 0 {
		return fmt.Errorf("AOQT prepared-IP taus must be positive")
	}
	return nil
}

func cloneAOQTVectors(in [][]float32) [][]float32 {
	out := make([][]float32, len(in))
	for i := range in {
		out[i] = append([]float32(nil), in[i]...)
	}
	return out
}

func projectAOQTAngles(angles []float32, cap float32) error {
	if cap <= 0 || !isFinite32(cap) {
		return fmt.Errorf("AOQT angle cap must be finite and positive")
	}
	for i, angle := range angles {
		if !isFinite32(angle) {
			return fmt.Errorf("AOQT angle %d is not finite", i)
		}
		if angle > cap {
			angles[i] = cap
		} else if angle < -cap {
			angles[i] = -cap
		}
	}
	return nil
}

func countAOQTAngles(transform AOQTGivensTransform) int {
	var count int
	for _, stage := range transform.Stages {
		count += len(stage.Pairs)
	}
	return count
}

func countEligibleAOQTPairs(row AOQTSidecarCalibrationRow) int {
	if len(row.EligiblePairMask) == 0 {
		n := len(row.CandidateDocIDs)
		return n * (n - 1)
	}
	var count int
	for i := range row.EligiblePairMask {
		for j := range row.EligiblePairMask[i] {
			if row.EligiblePairMask[i][j] {
				count++
			}
		}
	}
	return count
}

func aoqtAngleStats(angles []float32) (float32, float32) {
	var sq float64
	var maxAbs float32
	for _, angle := range angles {
		abs := float32(math.Abs(float64(angle)))
		if abs > maxAbs {
			maxAbs = abs
		}
		sq += float64(angle * angle)
	}
	return float32(math.Sqrt(sq)), maxAbs
}

func dotAOQT(a, b []float32) float32 {
	var out float32
	for i := range a {
		out += a[i] * b[i]
	}
	return out
}
