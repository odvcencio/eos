package eosruntime

import (
	"fmt"
	"math"
	"math/rand"
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
	Plan             AOQTSidecarWorkPlan `json:"plan"`
	Steps            int                 `json:"steps"`
	InitialLoss      float32             `json:"initial_loss"`
	FinalLoss        float32             `json:"final_loss"`
	AngleL2          float32             `json:"angle_l2"`
	AngleMaxAbs      float32             `json:"angle_max_abs"`
	AnglesSHA256     string              `json:"angles_sha256"`
	DenseMaxAbsDelta float64             `json:"dense_max_abs_delta"`
	QualityClaim     bool                `json:"quality_claim"`
}

type AOQTSidecarObjectiveInput struct {
	Row        AOQTSidecarCalibrationRow
	Query      []float32
	Candidates [][]float32
}

type AOQTSidecarObjectiveResult struct {
	Loss           float32
	QueryGrad      []float32
	CandidateGrads [][]float32
}

type AOQTSidecarVectorObjective interface {
	EvaluateAOQT(input AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error)
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
	summary := AOQTSidecarTrainSummary{Plan: plan, QualityClaim: false}
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
	var initialSet bool
	for step := 0; step < t.config.MaxSteps; step++ {
		rows := deterministicAOQTRowOrder(set.Rows, t.config.WorkplanSeed, step)
		loss, grad, err := t.lossAndAngleGrad(rows, objective)
		if err != nil {
			return summary, err
		}
		if !initialSet {
			summary.InitialLoss = loss
			initialSet = true
		}
		if err := t.applyAdam(grad); err != nil {
			return summary, err
		}
		summary.Steps++
	}
	finalLoss, _, err := t.lossAndAngleGrad(set.Rows, objective)
	if err != nil {
		return summary, err
	}
	summary.FinalLoss = finalLoss
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

func (t *AOQTSidecarTrainer) lossAndAngleGrad(rows []AOQTSidecarCalibrationRow, objective AOQTSidecarVectorObjective) (float32, []float32, error) {
	totalGrad := make([]float32, len(t.angles))
	totalLoss := float32(0)
	transform := t.Transform()
	for _, row := range rows {
		query, candidates, err := transformAOQTRow(transform, row)
		if err != nil {
			return 0, nil, err
		}
		result, err := objective.EvaluateAOQT(AOQTSidecarObjectiveInput{
			Row:        aoqtObjectiveRowView(row),
			Query:      append([]float32(nil), query...),
			Candidates: cloneAOQTVectors(candidates),
		})
		if err != nil {
			return 0, nil, err
		}
		if !isFinite32(result.Loss) {
			return 0, nil, fmt.Errorf("AOQT objective loss must be finite")
		}
		if len(result.QueryGrad) != transform.Dim {
			return 0, nil, fmt.Errorf("AOQT objective query grad dim = %d, want %d", len(result.QueryGrad), transform.Dim)
		}
		if len(result.CandidateGrads) != len(row.CandidateVectors) {
			return 0, nil, fmt.Errorf("AOQT objective candidate grad count = %d, want %d", len(result.CandidateGrads), len(row.CandidateVectors))
		}
		totalLoss += result.Loss
		if err := accumulateAOQTVectorAngleGrad(transform, row.QueryVector, result.QueryGrad, totalGrad); err != nil {
			return 0, nil, err
		}
		for i, grad := range result.CandidateGrads {
			if len(grad) != transform.Dim {
				return 0, nil, fmt.Errorf("AOQT objective candidate grad %d dim = %d, want %d", i, len(grad), transform.Dim)
			}
			if err := accumulateAOQTVectorAngleGrad(transform, row.CandidateVectors[i], grad, totalGrad); err != nil {
				return 0, nil, err
			}
		}
	}
	return totalLoss, totalGrad, nil
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
	if err := transform.Validate(); err != nil {
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
	return view
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
