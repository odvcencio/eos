package eosruntime

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"

	"m31labs.dev/turboquant"
)

func TestAOQTStage2ATopologyAndPlanAreDeterministic(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 11)
	trainerA := newTinyAOQTTrainer(t, true, 11)
	trainerB := newTinyAOQTTrainer(t, true, 11)
	planA, err := trainerA.Plan(set)
	if err != nil {
		t.Fatalf("plan A: %v", err)
	}
	planB, err := trainerB.Plan(set)
	if err != nil {
		t.Fatalf("plan B: %v", err)
	}
	if planA != planB {
		t.Fatalf("plans differ:\n%+v\n%+v", planA, planB)
	}
	if planA.AngleCount != AOQTSidecarAngleCount || planA.PairsPerStage != AOQTSidecarPairsPerStage || planA.StepCount != 0 || !planA.PlanOnly {
		t.Fatalf("unexpected plan: %+v", planA)
	}
	if got := len(trainerA.Angles()); got != AOQTSidecarAngleCount {
		t.Fatalf("angle count = %d, want %d", got, AOQTSidecarAngleCount)
	}
}

func TestAOQTStage2AIdentityDenseInvariant(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 17)
	trainer := newTinyAOQTTrainer(t, true, 17)
	delta, err := DenseInvariantMaxAbsDelta(trainer.Transform(), set.Rows)
	if err != nil {
		t.Fatalf("dense invariant audit: %v", err)
	}
	if delta != 0 {
		t.Fatalf("identity dense delta = %.12g, want 0", delta)
	}
}

func TestAOQTStage2AOnlyAnglesMutateAndCapsProject(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 23)
	trainer := newTinyAOQTTrainer(t, false, 23)
	before := cloneAOQTCalibrationSet(set)
	summary, err := trainer.Fit(set, toyAOQTObjective{})
	if err != nil {
		t.Fatalf("fit: %v", err)
	}
	if summary.Steps != 2 {
		t.Fatalf("steps = %d, want 2", summary.Steps)
	}
	if len(trainer.Angles()) != AOQTSidecarAngleCount {
		t.Fatalf("trainer angle state shape changed")
	}
	if before.Rows[0].QueryVector[0] != set.Rows[0].QueryVector[0] || before.Rows[0].CandidateVectors[0][0] != set.Rows[0].CandidateVectors[0][0] {
		t.Fatalf("frozen row vectors mutated")
	}
	for i, angle := range trainer.Angles() {
		if float32(math.Abs(float64(angle))) > AOQTSidecarDefaultAngleCap+1e-7 {
			t.Fatalf("angle %d = %.9g exceeds cap", i, angle)
		}
	}
	tooLarge := make([]float32, AOQTSidecarAngleCount)
	for i := range tooLarge {
		tooLarge[i] = AOQTSidecarDefaultAngleCap * 3
	}
	if err := trainer.SetAnglesForTest(tooLarge); err != nil {
		t.Fatalf("set projected angles: %v", err)
	}
	for i, angle := range trainer.Angles() {
		if angle != AOQTSidecarDefaultAngleCap {
			t.Fatalf("projected angle %d = %.9g, want cap %.9g", i, angle, AOQTSidecarDefaultAngleCap)
		}
	}
}

func TestAOQTStage2AReproducibleToyUpdate(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 31)
	a := newTinyAOQTTrainer(t, false, 31)
	b := newTinyAOQTTrainer(t, false, 31)
	sa, err := a.Fit(set, toyAOQTObjective{})
	if err != nil {
		t.Fatalf("fit A: %v", err)
	}
	sb, err := b.Fit(set, toyAOQTObjective{})
	if err != nil {
		t.Fatalf("fit B: %v", err)
	}
	if sa.AnglesSHA256 != sb.AnglesSHA256 {
		t.Fatalf("angle hashes differ: %s vs %s", sa.AnglesSHA256, sb.AnglesSHA256)
	}
	angles := a.Angles()
	var nonzero bool
	for _, angle := range angles {
		if angle != 0 {
			nonzero = true
			break
		}
	}
	if !nonzero {
		t.Fatalf("toy objective did not move any AOQT angles")
	}
}

func TestAOQTStage2AValidatorsFailClosed(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 41)
	bad := set
	bad.Manifest.LegalGates.CommercialUseAllowed = true
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "commercial_use_allowed=false") {
		t.Fatalf("commercial gate error = %v, want fail closed", err)
	}
	bad = cloneAOQTCalibrationSet(set)
	bad.Rows[0].QrelsSHA256 = hex64("other")
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "qrels_sha256 mismatch") {
		t.Fatalf("qrels mismatch error = %v, want mismatch", err)
	}
	bad = cloneAOQTCalibrationSet(set)
	bad.Rows[0].CandidateVectors[0] = bad.Rows[0].CandidateVectors[0][:383]
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "dim = 383") {
		t.Fatalf("bad vector dim error = %v, want dim failure", err)
	}
	bad = cloneAOQTCalibrationSet(set)
	bad.Rows[0].EligiblePairMask = [][]bool{{true}}
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "eligible_pair_mask rows") {
		t.Fatalf("bad mask error = %v, want mask failure", err)
	}
	bad = cloneAOQTCalibrationSet(set)
	bad.Rows[0].QueryVector = nil
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "query_vector dim = 0") {
		t.Fatalf("missing query vector error = %v, want required D384 vector", err)
	}
	bad = cloneAOQTCalibrationSet(set)
	bad.Rows[0].AnchorScores.Dense[0] += 0.01
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "anchor_scores.dense[0]") {
		t.Fatalf("bad dense anchor score error = %v, want dense-dot mismatch", err)
	}
	bad = cloneAOQTCalibrationSet(set)
	bad.Manifest.SplitProof.Split = "dev"
	bad.Rows[0].SplitProof = bad.Manifest.SplitProof
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "split=train") {
		t.Fatalf("bad split proof error = %v, want train-only rejection", err)
	}
	bad = cloneAOQTCalibrationSet(set)
	bad.Manifest.SplitProof.ExclusionIdentities = []string{"dev", "reserve", "test", "official", "proxy"}
	bad.Rows[0].SplitProof = bad.Manifest.SplitProof
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "dev4") {
		t.Fatalf("missing exclusion identity error = %v, want dev4 rejection", err)
	}
	bad = cloneAOQTCalibrationSet(set)
	bad.Manifest.QuantSurfaces[0].ScoreSurface = "turboquant.PreparedIP"
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), AOQTSidecarPreparedIPScoreSurface) {
		t.Fatalf("bad score surface error = %v, want exact surface rejection", err)
	}
	bad = cloneAOQTCalibrationSet(set)
	bad.Rows[0].CandidateVectorSHA256[0] = hex64("wrong")
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "candidate_vector_sha256[0]") {
		t.Fatalf("bad vector hash error = %v, want vector hash rejection", err)
	}
	bad = cloneAOQTCalibrationSet(set)
	bad.Rows[0].SourceArtifactHash = hex64("undeclared")
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "source_artifact_hash") {
		t.Fatalf("bad source hash error = %v, want manifest provenance rejection", err)
	}
	cfg := AOQTSidecarTrainConfig{PairingSeed: 1, WorkplanSeed: 2, AngleCap: 0.041, PlanOnly: true}
	if _, err := NewAOQTSidecarTrainer(cfg); err == nil || !strings.Contains(err.Error(), "exceeds default cap") {
		t.Fatalf("bad cap error = %v, want cap failure", err)
	}
}

func TestAOQTStage2AObjectiveInputIsImmutableVectorFreeRowView(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 47)
	before := cloneAOQTCalibrationSet(set)
	trainer := newTinyAOQTTrainer(t, false, 47)
	if _, err := trainer.Fit(set, mutatingAOQTObjective{}); err != nil {
		t.Fatalf("fit mutating objective: %v", err)
	}
	if before.Rows[0].QueryVector[0] != set.Rows[0].QueryVector[0] || before.Rows[0].CandidateVectors[0][0] != set.Rows[0].CandidateVectors[0][0] {
		t.Fatalf("mutating objective altered calibration vectors")
	}
	if before.Rows[0].CandidateDocIDs[0] != set.Rows[0].CandidateDocIDs[0] || before.Rows[0].AnchorScores.Dense[0] != set.Rows[0].AnchorScores.Dense[0] {
		t.Fatalf("mutating objective altered row metadata")
	}
	if string(before.Rows[0].Extra["note"]) != string(set.Rows[0].Extra["note"]) || string(before.Rows[0].SplitProof.ExclusionIdentities[0]) != string(set.Rows[0].SplitProof.ExclusionIdentities[0]) {
		t.Fatalf("mutating objective altered nested row metadata")
	}
}

func TestAOQTStage2AMetricsBindInputsAndRejectClaims(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 53)
	trainer := newTinyAOQTTrainer(t, true, 53)
	summary, err := trainer.Fit(set, nil)
	if err != nil {
		t.Fatalf("plan-only fit: %v", err)
	}
	metrics, err := NewAOQTSidecarRunMetrics(set, summary)
	if err != nil {
		t.Fatalf("metrics: %v", err)
	}
	if metrics.Inputs.QrelsSHA256ByDataset["toy"] != set.Manifest.QrelsSHA256ByDataset["toy"] {
		t.Fatalf("metrics qrels binding mismatch")
	}
	metrics.QualityClaim = true
	if err := metrics.Validate(); err == nil || !strings.Contains(err.Error(), "quality_claim must be false") {
		t.Fatalf("quality claim error = %v, want rejection", err)
	}
}

func TestAOQTStage2AAngleGradientMatchesFiniteDifference(t *testing.T) {
	transform := AOQTGivensTransform{
		Version:  AOQTTransformVersion,
		Kind:     EmbeddingPostPoolTransformAOQTGivens,
		Dim:      AOQTSidecarDim,
		Seed:     61,
		AngleCap: AOQTSidecarDefaultAngleCap,
		Stages: []AOQTStage{{
			Pairs:  [][2]int{{0, 1}},
			Angles: []float32{0.012},
		}},
	}
	x := make([]float32, AOQTSidecarDim)
	upstream := make([]float32, AOQTSidecarDim)
	pair := transform.Stages[0].Pairs[0]
	x[pair[0]] = 0.7
	x[pair[1]] = -0.2
	upstream[pair[0]] = -0.3
	upstream[pair[1]] = 0.5
	got := make([]float32, countAOQTAngles(transform))
	if err := accumulateAOQTVectorAngleGrad(transform, x, upstream, got); err != nil {
		t.Fatalf("angle grad: %v", err)
	}
	eps := float32(1e-3)
	plusAngles := []float32{transform.Stages[0].Angles[0]}
	plusAngles[0] += eps
	minusAngles := []float32{transform.Stages[0].Angles[0]}
	minusAngles[0] -= eps
	plusTransform := transformWithAngles(t, transform, plusAngles)
	minusTransform := transformWithAngles(t, transform, minusAngles)
	plusVec, err := plusTransform.ApplyVector(x)
	if err != nil {
		t.Fatalf("plus apply: %v", err)
	}
	minusVec, err := minusTransform.ApplyVector(x)
	if err != nil {
		t.Fatalf("minus apply: %v", err)
	}
	fd := (dotAOQT(plusVec, upstream) - dotAOQT(minusVec, upstream)) / (2 * eps)
	if math.Abs(float64(got[0]-fd)) > 2e-4 {
		t.Fatalf("angle grad = %.9g, finite diff %.9g", got[0], fd)
	}
}

func TestAOQTStage2BPreparedIPObjectiveMatchesTurboQuantSurface(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 67)
	row := set.Rows[0]
	query := append([]float32(nil), row.QueryVector...)
	candidates := cloneAOQTVectors(row.CandidateVectors)
	objective, err := NewAOQTSidecarPreparedIPObjective(AOQTSidecarPreparedIPObjectiveConfig{
		TurboQuantSeed: set.Manifest.TurboQuantSeed,
	})
	if err != nil {
		t.Fatalf("objective: %v", err)
	}
	result, err := objective.EvaluateAOQT(AOQTSidecarObjectiveInput{
		Row:        aoqtObjectiveRowView(row),
		Query:      query,
		Candidates: candidates,
	})
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if result.Loss <= 0 {
		t.Fatalf("loss = %.9g, want positive synthetic q3/q5 objective", result.Loss)
	}
	if len(result.QueryGrad) != AOQTSidecarDim || len(result.CandidateGrads) != len(candidates) {
		t.Fatalf("unexpected gradient shapes")
	}
	var gradL2 float64
	for _, v := range result.QueryGrad {
		gradL2 += float64(v * v)
	}
	for _, grad := range result.CandidateGrads {
		for _, v := range grad {
			gradL2 += float64(v * v)
		}
	}
	if gradL2 == 0 {
		t.Fatalf("prepared-IP objective produced zero vector gradient")
	}
	surface := newAOQTPreparedIPSurface(query, candidates, AOQTSidecarDim, AOQTSidecarDefaultGainBit, set.Manifest.TurboQuantSeed)
	q := turboquant.NewIPWithSeed(AOQTSidecarDim, AOQTSidecarDefaultGainBit, set.Manifest.TurboQuantSeed)
	prepared := q.PrepareQuery(normalizedAOQTVector(query))
	for i, candidate := range candidates {
		want := q.InnerProductPrepared(q.Quantize(normalizedAOQTVector(candidate)), prepared)
		if math.Abs(float64(surface.scores[i]-want)) > 1e-7 {
			t.Fatalf("surface score[%d] = %.9g, want TurboQuant prepared-IP %.9g", i, surface.scores[i], want)
		}
	}
}

func TestAOQTStage2BPreparedIPSurfaceBindsRawUnitVectorContract(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 68)
	row := set.Rows[0]
	query := append([]float32(nil), row.QueryVector...)
	candidates := cloneAOQTVectors(row.CandidateVectors)
	surface := newAOQTPreparedIPSurface(query, candidates, AOQTSidecarDim, AOQTSidecarDefaultGainBit, set.Manifest.TurboQuantSeed)
	q := turboquant.NewIPWithSeed(AOQTSidecarDim, AOQTSidecarDefaultGainBit, set.Manifest.TurboQuantSeed)
	preparedRaw := q.PrepareQuery(query)
	preparedUnit := q.PrepareQuery(normalizedAOQTVector(query))
	for i, candidate := range candidates {
		raw := q.InnerProductPrepared(q.Quantize(candidate), preparedRaw)
		unit := q.InnerProductPrepared(q.Quantize(normalizedAOQTVector(candidate)), preparedUnit)
		if math.Abs(float64(surface.scores[i]-raw)) > 1e-7 {
			t.Fatalf("surface score[%d] = %.9g, want raw prepared-IP %.9g", i, surface.scores[i], raw)
		}
		if math.Abs(float64(raw-unit)) > 1e-7 {
			t.Fatalf("raw/unit prepared-IP score[%d] diverged %.9g vs %.9g for unit-vector row", i, raw, unit)
		}
	}
}

func TestAOQTStage2BPositiveGuardsFailClosedWhenInert(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 69)
	objective, err := NewAOQTSidecarPreparedIPObjective(tinyAOQTObjectiveConfig(set.Manifest.TurboQuantSeed))
	if err != nil {
		t.Fatalf("objective: %v", err)
	}
	for _, tc := range []struct {
		name      string
		weights   AOQTSidecarRowWeights
		wantError string
	}{
		{
			name:      "q3",
			weights:   AOQTSidecarRowWeights{Q3OrderGuard: 1},
			wantError: "q3_order_guard weight",
		},
		{
			name:      "q5",
			weights:   AOQTSidecarRowWeights{Q5OrderGuard: 1},
			wantError: "q5_order_guard weight",
		},
		{
			name:      "nf",
			weights:   AOQTSidecarRowWeights{NFBoundaryGuard: 1},
			wantError: "nf_boundary_guard weight",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			row := aoqtObjectiveRowView(set.Rows[0])
			row.Weights = tc.weights
			row.EligiblePairMask = [][]bool{
				{false, false, false},
				{false, false, false},
				{false, false, false},
			}
			_, err := objective.EvaluateAOQT(AOQTSidecarObjectiveInput{
				Row:        row,
				Query:      append([]float32(nil), set.Rows[0].QueryVector...),
				Candidates: cloneAOQTVectors(set.Rows[0].CandidateVectors),
			})
			if err == nil || !strings.Contains(err.Error(), tc.wantError) || !strings.Contains(err.Error(), "no active contributing") {
				t.Fatalf("error = %v, want inert positive %s guard rejection", err, tc.name)
			}
		})
	}
}

func TestAOQTStage2BValidQ5AndNFGuardsActivate(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 70)
	set.Rows[0].Weights.NFBoundaryGuard = 0.75
	set.Rows[0].EligiblePairMask[2][1] = true
	set.Manifest.ObjectiveContract = tinyAOQTObjectiveConfig(set.Manifest.TurboQuantSeed).ObjectiveContract(sumAOQTRowWeights(set.Rows))
	if err := set.Validate(); err != nil {
		t.Fatalf("fixture with q5/nf guards invalid: %v", err)
	}
	objective, err := NewAOQTSidecarPreparedIPObjective(tinyAOQTObjectiveConfig(set.Manifest.TurboQuantSeed))
	if err != nil {
		t.Fatalf("objective: %v", err)
	}
	result, err := objective.EvaluateAOQT(AOQTSidecarObjectiveInput{
		Row:        aoqtObjectiveRowView(set.Rows[0]),
		Query:      append([]float32(nil), set.Rows[0].QueryVector...),
		Candidates: cloneAOQTVectors(set.Rows[0].CandidateVectors),
	})
	if err != nil {
		t.Fatalf("evaluate valid q5/nf fixture: %v", err)
	}
	if result.Activation.Q5OrderGuardPairs == 0 || result.Activation.Q5OrderGuardContributing == 0 {
		t.Fatalf("q5 guard activation = %+v, want active contributing q5 guard", result.Activation)
	}
	if result.Activation.Q5ScoreDistillCount != len(set.Rows[0].CandidateDocIDs) {
		t.Fatalf("q5 score distill count = %d, want %d", result.Activation.Q5ScoreDistillCount, len(set.Rows[0].CandidateDocIDs))
	}
	if result.Activation.NFBoundaryGuardPairs == 0 || result.Activation.NFBoundaryGuardContributing == 0 {
		t.Fatalf("NF guard activation = %+v, want active contributing NF guard", result.Activation)
	}
}

func TestAOQTStage2BPreparedIPObjectiveMovesAnglesDeterministically(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 71)
	objective, err := NewAOQTSidecarPreparedIPObjective(AOQTSidecarPreparedIPObjectiveConfig{
		TurboQuantSeed: set.Manifest.TurboQuantSeed,
	})
	if err != nil {
		t.Fatalf("objective: %v", err)
	}
	a := newTinyAOQTTrainer(t, false, 71)
	b := newTinyAOQTTrainer(t, false, 71)
	sa, err := a.Fit(set, objective)
	if err != nil {
		t.Fatalf("fit A: %v", err)
	}
	sb, err := b.Fit(set, objective)
	if err != nil {
		t.Fatalf("fit B: %v", err)
	}
	if sa.AnglesSHA256 != sb.AnglesSHA256 {
		t.Fatalf("angle hashes differ: %s vs %s", sa.AnglesSHA256, sb.AnglesSHA256)
	}
	var moved bool
	for _, angle := range a.Angles() {
		if angle != 0 {
			moved = true
			break
		}
	}
	if !moved {
		t.Fatalf("prepared-IP objective did not move AOQT angles")
	}
}

func TestAOQTStage2BOrderGuardOpposesUnsafeMovement(t *testing.T) {
	scores := []float32{0.1, 0.5, 0.2}
	ranks := []int{1, 3, 2}
	loss, grads, pairs, contributing := aoqtAnchorOrderGuardLossAndGrad(scores, ranks, nil, 0.05, 0.01)
	if loss <= 0 {
		t.Fatalf("guard loss = %.9g, want active violation", loss)
	}
	if pairs == 0 || contributing == 0 {
		t.Fatalf("guard activation pairs=%d contributing=%d, want active coverage", pairs, contributing)
	}
	if grads[0] >= 0 {
		t.Fatalf("guard grad for anchor-best doc = %.9g, want negative so descent raises it", grads[0])
	}
	if grads[1] <= 0 {
		t.Fatalf("guard grad for unsafe promoted doc = %.9g, want positive so descent lowers it", grads[1])
	}
}

func TestAOQTStage2BValidatorsRejectRankAndMetricsPlanDefects(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 73)
	bad := cloneAOQTCalibrationSet(set)
	bad.Rows[0].AnchorRanks.Q3 = []int{1, 1, 3}
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "unique 1..3 permutation") {
		t.Fatalf("duplicate rank error = %v, want unique permutation rejection", err)
	}
	trainer := newTinyAOQTTrainer(t, true, 73)
	summary, err := trainer.Fit(set, nil)
	if err != nil {
		t.Fatalf("plan-only fit: %v", err)
	}
	summary.Plan.RowCount++
	if _, err := NewAOQTSidecarRunMetrics(set, summary); err == nil || !strings.Contains(err.Error(), "row_count") {
		t.Fatalf("bad metrics plan error = %v, want row_count rejection", err)
	}
	summary, err = trainer.Fit(set, nil)
	if err != nil {
		t.Fatalf("plan-only fit 2: %v", err)
	}
	metrics, err := NewAOQTSidecarRunMetrics(set, summary)
	if err != nil {
		t.Fatalf("metrics: %v", err)
	}
	metrics.Summary.Plan.PairingsSHA256 = hex64("different-plan")
	if err := metrics.Validate(); err == nil || !strings.Contains(err.Error(), "plan must exactly match summary.plan") {
		t.Fatalf("metrics plan mismatch error = %v, want rejection", err)
	}
}

func TestAOQTStage2BObjectiveContractMismatchesFailClosed(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 74)
	for _, tc := range []struct {
		name string
		cfg  AOQTSidecarPreparedIPObjectiveConfig
	}{
		{
			name: "seed",
			cfg:  AOQTSidecarPreparedIPObjectiveConfig{TurboQuantSeed: set.Manifest.TurboQuantSeed + 1},
		},
		{
			name: "gain-bit",
			cfg:  AOQTSidecarPreparedIPObjectiveConfig{TurboQuantSeed: set.Manifest.TurboQuantSeed, GainBit: 4},
		},
		{
			name: "q5-bit",
			cfg:  AOQTSidecarPreparedIPObjectiveConfig{TurboQuantSeed: set.Manifest.TurboQuantSeed, Q5GuardBit: 4},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			objective, err := NewAOQTSidecarPreparedIPObjective(tc.cfg)
			if err != nil {
				t.Fatalf("objective: %v", err)
			}
			if _, err := newTinyAOQTTrainer(t, false, 74).Fit(set, objective); err == nil || !strings.Contains(err.Error(), "objective config does not match") {
				t.Fatalf("fit error = %v, want objective/manifest contract rejection", err)
			}
		})
	}
	bad := cloneAOQTCalibrationSet(set)
	bad.Manifest.ObjectiveContract.GainBit = 4
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "bit_width 4 is not declared") {
		t.Fatalf("manifest objective bit error = %v, want undeclared quant surface rejection", err)
	}
}

func TestAOQTStage2BMetricsExposeAndValidateObjectiveProvenance(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 75)
	trainer := newTinyAOQTTrainer(t, true, 75)
	summary, err := trainer.Fit(set, nil)
	if err != nil {
		t.Fatalf("plan-only fit: %v", err)
	}
	metrics, err := NewAOQTSidecarRunMetrics(set, summary)
	if err != nil {
		t.Fatalf("metrics: %v", err)
	}
	if !aoqtObjectiveContractsEqual(metrics.ObjectiveContract, set.Manifest.ObjectiveContract) {
		t.Fatalf("metrics objective contract = %+v, want manifest contract %+v", metrics.ObjectiveContract, set.Manifest.ObjectiveContract)
	}
	if metrics.ObjectiveContract.TurboQuantSeed != set.Manifest.TurboQuantSeed || metrics.ObjectiveContract.GainBit != AOQTSidecarDefaultGainBit || metrics.ObjectiveContract.Q5GuardBit != AOQTSidecarDefaultGuardBit5 {
		t.Fatalf("metrics objective provenance not bound to configured TQ seed/bits: %+v", metrics.ObjectiveContract)
	}
	if metrics.ObjectiveContract.ScoreSurface != AOQTSidecarPreparedIPScoreSurface {
		t.Fatalf("metrics score surface = %q, want %q", metrics.ObjectiveContract.ScoreSurface, AOQTSidecarPreparedIPScoreSurface)
	}
	metrics.ObjectiveContract.TurboQuantSeed++
	if err := metrics.Validate(); err == nil || !strings.Contains(err.Error(), "objective_contract must exactly match") {
		t.Fatalf("mutated metrics objective error = %v, want provenance mismatch rejection", err)
	}
}

func TestAOQTStage2BValidatorsRejectNonUnitVectors(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 76)
	bad := cloneAOQTCalibrationSet(set)
	bad.Rows[0].QueryVector[0] *= 0.5
	bad.Rows[0].QueryVectorSHA256 = aoqtVectorSHA256(bad.Rows[0].QueryVector)
	for i := range bad.Rows[0].CandidateVectors {
		bad.Rows[0].AnchorScores.Dense[i] = dotAOQT(bad.Rows[0].QueryVector, bad.Rows[0].CandidateVectors[i])
	}
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "query_vector norm") {
		t.Fatalf("non-unit query error = %v, want unit-vector rejection", err)
	}
	bad = cloneAOQTCalibrationSet(set)
	bad.Rows[0].CandidateVectors[0][0] *= 0.5
	bad.Rows[0].CandidateVectorSHA256[0] = aoqtVectorSHA256(bad.Rows[0].CandidateVectors[0])
	bad.Rows[0].AnchorScores.Dense[0] = dotAOQT(bad.Rows[0].QueryVector, bad.Rows[0].CandidateVectors[0])
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "candidate_vectors[0] norm") {
		t.Fatalf("non-unit candidate error = %v, want unit-vector rejection", err)
	}
}

type toyAOQTObjective struct{}

func (toyAOQTObjective) EvaluateAOQT(input AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error) {
	queryGrad := make([]float32, len(input.Query))
	candidateGrads := make([][]float32, len(input.Candidates))
	var loss float32
	for i, candidate := range input.Candidates {
		candidateGrads[i] = make([]float32, len(candidate))
		want := input.Row.QrelGains[i]
		got := input.Query[0]*candidate[1] - input.Query[1]*candidate[0]
		diff := got - want
		loss += 0.5 * diff * diff
		queryGrad[0] += diff * candidate[1]
		queryGrad[1] -= diff * candidate[0]
		candidateGrads[i][1] += diff * input.Query[0]
		candidateGrads[i][0] -= diff * input.Query[1]
	}
	return AOQTSidecarObjectiveResult{Loss: loss, QueryGrad: queryGrad, CandidateGrads: candidateGrads}, nil
}

func newTinyAOQTTrainer(t *testing.T, planOnly bool, seed int64) *AOQTSidecarTrainer {
	t.Helper()
	trainer, err := NewAOQTSidecarTrainer(AOQTSidecarTrainConfig{
		PairingSeed:  seed,
		WorkplanSeed: seed + 1000,
		PlanOnly:     planOnly,
		MaxSteps:     2,
		LearningRate: 0.01,
	})
	if err != nil {
		t.Fatalf("new trainer: %v", err)
	}
	return trainer
}

func tinyAOQTCalibrationSet(t *testing.T, seed int64) AOQTSidecarCalibrationSet {
	t.Helper()
	topology, err := NewAOQTGivensIdentityTopology(AOQTSidecarDim, AOQTSidecarStages, seed, AOQTSidecarDefaultAngleCap)
	if err != nil {
		t.Fatalf("topology: %v", err)
	}
	pairings, err := topology.PairingsSHA256()
	if err != nil {
		t.Fatalf("pairings: %v", err)
	}
	rng := rand.New(rand.NewSource(seed))
	query := aoqtRandomVec(rng)
	candidates := [][]float32{aoqtRandomVec(rng), aoqtRandomVec(rng), aoqtRandomVec(rng)}
	rowID := "row-0001"
	qrels := hex64("qrels")
	compat := hex64("compat")
	splitProof := tinyAOQTSplitProof()
	rows := []AOQTSidecarCalibrationRow{{
		Schema:                AOQTSidecarRowSchema,
		RowID:                 rowID,
		Dataset:               "toy",
		QueryID:               "q1",
		QueryVectorID:         "qv1",
		QueryVectorSHA256:     aoqtVectorSHA256(query),
		QueryVector:           query,
		CandidateDocIDs:       []string{"d1", "d2", "d3"},
		CandidateVectorIDs:    []string{"dv1", "dv2", "dv3"},
		CandidateVectorSHA256: []string{aoqtVectorSHA256(candidates[0]), aoqtVectorSHA256(candidates[1]), aoqtVectorSHA256(candidates[2])},
		CandidateVectors:      candidates,
		QrelGains:             []float32{1, 0, 0.25},
		CandidateSources:      []string{"frontier_top10", "broad_anchor", "nf_boundary80_120"},
		EligiblePairMask: [][]bool{
			{false, true, true},
			{false, false, true},
			{true, false, false},
		},
		GuardClass: "frontier_top10",
		AnchorScores: AOQTSidecarAnchorScores{
			Dense: []float32{dotAOQT(query, candidates[0]), dotAOQT(query, candidates[1]), dotAOQT(query, candidates[2])},
			Q3:    []float32{0.3, 0.1, 0.2},
			Q5:    []float32{0.31, 0.11, 0.21},
		},
		AnchorRanks: AOQTSidecarAnchorRanks{
			Dense: []int{1, 3, 2},
			Q3:    []int{1, 3, 2},
			Q5:    []int{1, 3, 2},
		},
		Weights: AOQTSidecarRowWeights{
			Q3Gain:         1,
			Q3OrderGuard:   0.5,
			Q3ScoreDistill: 0.25,
			Q5OrderGuard:   0.5,
			Q5ScoreDistill: 0.25,
		},
		SourceArtifactHash:  hex64("source"),
		QrelsSHA256:         qrels,
		SplitProof:          splitProof,
		CompatibilityDigest: compat,
		LegalGates:          aoqtResearchOnlyGates(),
		Extra:               map[string]json.RawMessage{"note": json.RawMessage(`{"fixture":true}`)},
	}}
	manifest := AOQTSidecarCalibrationManifest{
		Schema:                      AOQTSidecarManifestSchema,
		AnchorArtifactSHA256:        hex64("anchor"),
		AnchorPackageManifestSHA256: hex64("pkg"),
		AnchorEmbeddingSpaceID:      "eos-d384-anchor",
		Dim:                         AOQTSidecarDim,
		Topology: AOQTSidecarTopologyBinding{
			Kind:           AOQTTopologyKindGivensV1,
			Dim:            AOQTSidecarDim,
			Stages:         AOQTSidecarStages,
			PairsPerStage:  AOQTSidecarPairsPerStage,
			AngleCount:     AOQTSidecarAngleCount,
			Seed:           seed,
			PairingsSHA256: pairings,
		},
		TurboQuantSeed: 77,
		QuantSurfaces: []AOQTSidecarQuantSurface{
			{BitWidth: 3, Seed: 77, ScoreSurface: AOQTSidecarPreparedIPScoreSurface, PreparedQuery: true},
			{BitWidth: 5, Seed: 77, ScoreSurface: AOQTSidecarPreparedIPScoreSurface, PreparedQuery: true},
		},
		QrelsSHA256ByDataset: map[string]string{"toy": qrels},
		SplitProof:           splitProof,
		SourceArtifactHashes: []string{hex64("source")},
		VectorCacheHashes:    []string{hex64("cache")},
		RowCount:             len(rows),
		RowIDSHA256:          aoqtRowIDSHA256([]string{rowID}),
		CompatibilityDigest:  compat,
		LegalGates:           aoqtResearchOnlyGates(),
		Extra:                map[string]json.RawMessage{"fixture": json.RawMessage(`true`)},
	}
	manifest.ObjectiveContract = tinyAOQTObjectiveConfig(manifest.TurboQuantSeed).ObjectiveContract(sumAOQTRowWeights(rows))
	set := AOQTSidecarCalibrationSet{Manifest: manifest, Rows: rows}
	if err := set.Validate(); err != nil {
		t.Fatalf("fixture invalid: %v", err)
	}
	return set
}

type mutatingAOQTObjective struct{}

func (mutatingAOQTObjective) EvaluateAOQT(input AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error) {
	if len(input.Row.QueryVector) != 0 || len(input.Row.CandidateVectors) != 0 {
		return AOQTSidecarObjectiveResult{}, fmt.Errorf("objective row view exposed calibration vectors")
	}
	input.Row.CandidateDocIDs[0] = "mutated-doc"
	input.Row.AnchorScores.Dense[0] = 99
	input.Row.SplitProof.ExclusionIdentities[0] = "mutated-split"
	input.Row.Extra["note"][0] = 'x'
	input.Query[0] = 99
	input.Candidates[0][0] = 99
	queryGrad := make([]float32, len(input.Query))
	candidateGrads := make([][]float32, len(input.Candidates))
	for i := range candidateGrads {
		candidateGrads[i] = make([]float32, len(input.Candidates[i]))
	}
	return AOQTSidecarObjectiveResult{Loss: 0, QueryGrad: queryGrad, CandidateGrads: candidateGrads}, nil
}

func (toyAOQTObjective) AOQTPreparedIPObjectiveConfig() AOQTSidecarPreparedIPObjectiveConfig {
	return tinyAOQTObjectiveConfig(77)
}

func (mutatingAOQTObjective) AOQTPreparedIPObjectiveConfig() AOQTSidecarPreparedIPObjectiveConfig {
	return tinyAOQTObjectiveConfig(77)
}

func tinyAOQTObjectiveConfig(seed int64) AOQTSidecarPreparedIPObjectiveConfig {
	return AOQTSidecarPreparedIPObjectiveConfig{TurboQuantSeed: seed}
}

func tinyAOQTSplitProof() AOQTSidecarTrainOnlySplitProof {
	return AOQTSidecarTrainOnlySplitProof{
		Split:               "train",
		TrainOnly:           true,
		ProofSHA256:         hex64("split-proof"),
		ExclusionIdentities: append([]string(nil), requiredAOQTSplitExclusionIdentities...),
	}
}

func aoqtResearchOnlyGates() AOQTSidecarLegalGates {
	return AOQTSidecarLegalGates{ResearchTrainAllowed: true}
}

func aoqtRandomVec(rng *rand.Rand) []float32 {
	out := make([]float32, AOQTSidecarDim)
	for i := range out {
		out[i] = float32(rng.NormFloat64() * 0.01)
	}
	norm := vectorNorm(out)
	if norm != 0 {
		inv := 1 / norm
		for i := range out {
			out[i] *= inv
		}
	}
	return out
}

func hex64(label string) string {
	sum := sha256.Sum256([]byte(label))
	return hex.EncodeToString(sum[:])
}

func transformWithAngles(t *testing.T, base AOQTGivensTransform, angles []float32) AOQTGivensTransform {
	t.Helper()
	out := base
	out.Stages = append([]AOQTStage(nil), base.Stages...)
	offset := 0
	for i := range out.Stages {
		out.Stages[i].Pairs = append([][2]int(nil), base.Stages[i].Pairs...)
		out.Stages[i].Angles = append([]float32(nil), angles[offset:offset+len(out.Stages[i].Pairs)]...)
		offset += len(out.Stages[i].Pairs)
	}
	out.Audit.AnglesSHA256 = ""
	out.Audit.Orthogonality = nil
	if err := out.Validate(); err != nil {
		t.Fatalf("transform with angles: %v", err)
	}
	return out
}

func cloneAOQTCalibrationSet(in AOQTSidecarCalibrationSet) AOQTSidecarCalibrationSet {
	out := in
	out.Manifest.QrelsSHA256ByDataset = cloneAOQTStringMap(in.Manifest.QrelsSHA256ByDataset)
	out.Manifest.QuantSurfaces = append([]AOQTSidecarQuantSurface(nil), in.Manifest.QuantSurfaces...)
	out.Manifest.SplitProof.ExclusionIdentities = append([]string(nil), in.Manifest.SplitProof.ExclusionIdentities...)
	out.Manifest.SourceArtifactHashes = append([]string(nil), in.Manifest.SourceArtifactHashes...)
	out.Manifest.VectorCacheHashes = append([]string(nil), in.Manifest.VectorCacheHashes...)
	out.Manifest.Extra = cloneAOQTRawMessageMap(in.Manifest.Extra)
	out.Rows = append([]AOQTSidecarCalibrationRow(nil), in.Rows...)
	for i := range out.Rows {
		out.Rows[i].CandidateDocIDs = append([]string(nil), in.Rows[i].CandidateDocIDs...)
		out.Rows[i].CandidateVectorIDs = append([]string(nil), in.Rows[i].CandidateVectorIDs...)
		out.Rows[i].CandidateVectorSHA256 = append([]string(nil), in.Rows[i].CandidateVectorSHA256...)
		out.Rows[i].QrelGains = append([]float32(nil), in.Rows[i].QrelGains...)
		out.Rows[i].CandidateSources = append([]string(nil), in.Rows[i].CandidateSources...)
		out.Rows[i].SplitProof.ExclusionIdentities = append([]string(nil), in.Rows[i].SplitProof.ExclusionIdentities...)
		out.Rows[i].AnchorScores.Dense = append([]float32(nil), in.Rows[i].AnchorScores.Dense...)
		out.Rows[i].AnchorScores.Q3 = append([]float32(nil), in.Rows[i].AnchorScores.Q3...)
		out.Rows[i].AnchorScores.Q5 = append([]float32(nil), in.Rows[i].AnchorScores.Q5...)
		out.Rows[i].AnchorRanks.Dense = append([]int(nil), in.Rows[i].AnchorRanks.Dense...)
		out.Rows[i].AnchorRanks.Q3 = append([]int(nil), in.Rows[i].AnchorRanks.Q3...)
		out.Rows[i].AnchorRanks.Q5 = append([]int(nil), in.Rows[i].AnchorRanks.Q5...)
		out.Rows[i].QueryVector = append([]float32(nil), in.Rows[i].QueryVector...)
		out.Rows[i].CandidateVectors = append([][]float32(nil), in.Rows[i].CandidateVectors...)
		for j := range out.Rows[i].CandidateVectors {
			out.Rows[i].CandidateVectors[j] = append([]float32(nil), in.Rows[i].CandidateVectors[j]...)
		}
		out.Rows[i].EligiblePairMask = append([][]bool(nil), in.Rows[i].EligiblePairMask...)
		for j := range out.Rows[i].EligiblePairMask {
			out.Rows[i].EligiblePairMask[j] = append([]bool(nil), in.Rows[i].EligiblePairMask[j]...)
		}
		out.Rows[i].Extra = cloneAOQTRawMessageMap(in.Rows[i].Extra)
	}
	return out
}
