package eosruntime

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"sync"
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
	bad.Rows[0].AnchorScores.Q3[0] += 0.01
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "anchor_scores.q3[0]") {
		t.Fatalf("bad q3 anchor score error = %v, want prepared-IP recompute mismatch", err)
	}
	bad = cloneAOQTCalibrationSet(set)
	bad.Rows[0].AnchorRanks.Q5[0], bad.Rows[0].AnchorRanks.Q5[1] = bad.Rows[0].AnchorRanks.Q5[1], bad.Rows[0].AnchorRanks.Q5[0]
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "anchor_ranks.q5") {
		t.Fatalf("bad q5 anchor rank error = %v, want prepared-IP rank recompute mismatch", err)
	}
	bad = cloneAOQTCalibrationSet(set)
	bad.Rows[0].CandidateVectors[1] = append([]float32(nil), bad.Rows[0].CandidateVectors[0]...)
	bad.Rows[0].CandidateVectorSHA256[1] = aoqtVectorSHA256(bad.Rows[0].CandidateVectors[1])
	bad.Rows[0].AnchorScores.Dense = denseAOQTScores(bad.Rows[0].QueryVector, bad.Rows[0].CandidateVectors)
	bad.Rows[0].AnchorScores.Q3 = preparedAOQTScores(bad.Rows[0].QueryVector, bad.Rows[0].CandidateVectors, bad.Manifest.ObjectiveContract.Q3GuardBit, bad.Manifest.TurboQuantSeed)
	bad.Rows[0].AnchorScores.Q5 = preparedAOQTScores(bad.Rows[0].QueryVector, bad.Rows[0].CandidateVectors, bad.Manifest.ObjectiveContract.Q5GuardBit, bad.Manifest.TurboQuantSeed)
	bad.Rows[0].AnchorRanks.Dense = ranksAOQT(bad.Rows[0].CandidateDocIDs, bad.Rows[0].AnchorScores.Dense)
	bad.Rows[0].AnchorRanks.Q3 = ranksAOQT(bad.Rows[0].CandidateDocIDs, bad.Rows[0].AnchorScores.Q3)
	bad.Rows[0].AnchorRanks.Q5 = ranksAOQT(bad.Rows[0].CandidateDocIDs, bad.Rows[0].AnchorScores.Q5)
	bad.Rows[0].AnchorRanks.Q3[0], bad.Rows[0].AnchorRanks.Q3[1] = bad.Rows[0].AnchorRanks.Q3[1], bad.Rows[0].AnchorRanks.Q3[0]
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "doc-id-asc tie policy") {
		t.Fatalf("bad q3 tie-policy rank error = %v, want exact tie policy rejection", err)
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

func TestAOQTStage2ARejectsObjectiveLossComponentMismatch(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 49)
	trainer := newTinyAOQTTrainer(t, false, 49)
	if _, err := trainer.Fit(set, unaccountedAOQTObjective{}); err == nil || !strings.Contains(err.Error(), "must equal component sum") {
		t.Fatalf("unaccounted objective error = %v, want component accounting rejection", err)
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

func TestAOQTStage2AAngleGradientStreamingMatchesSnapshotReference(t *testing.T) {
	transform, err := NewAOQTGivensIdentityTopology(AOQTSidecarDim, AOQTSidecarStages, 62, AOQTSidecarDefaultAngleCap)
	if err != nil {
		t.Fatalf("topology: %v", err)
	}
	rng := rand.New(rand.NewSource(6201))
	for si := range transform.Stages {
		for ai := range transform.Stages[si].Angles {
			transform.Stages[si].Angles[ai] = float32((rng.Float64()*2 - 1) * float64(AOQTSidecarDefaultAngleCap) * 0.75)
		}
	}
	input := randomFloat32Vector(rng, AOQTSidecarDim)
	upstream := randomFloat32Vector(rng, AOQTSidecarDim)
	got := make([]float32, countAOQTAngles(transform))
	if err := accumulateAOQTVectorAngleGrad(transform, input, upstream, got); err != nil {
		t.Fatalf("streaming angle grad: %v", err)
	}
	want, err := snapshotAOQTVectorAngleGradReference(transform, input, upstream)
	if err != nil {
		t.Fatalf("snapshot reference angle grad: %v", err)
	}
	for i := range got {
		if delta := math.Abs(float64(got[i] - want[i])); delta > 2e-5 {
			t.Fatalf("angle grad[%d] delta = %.12g, got %.9g want %.9g", i, delta, got[i], want[i])
		}
	}
}

func TestAOQTStage2ATrainingHotPathAvoidsFullAuditPerVector(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 63)
	trainer := newTinyAOQTTrainer(t, false, 63)
	var validateCalls int
	var orthogonalityCalls int
	oldValidateHook := aoqtGivensValidateHookForTest
	oldOrthogonalityHook := aoqtGivensOrthogonalityHookForTest
	aoqtGivensValidateHookForTest = func() { validateCalls++ }
	aoqtGivensOrthogonalityHookForTest = func() { orthogonalityCalls++ }
	t.Cleanup(func() {
		aoqtGivensValidateHookForTest = oldValidateHook
		aoqtGivensOrthogonalityHookForTest = oldOrthogonalityHook
	})

	if _, _, _, _, err := trainer.lossAndAngleGrad(set.Rows, toyAOQTObjective{}); err != nil {
		t.Fatalf("loss and angle grad: %v", err)
	}
	if validateCalls != 0 || orthogonalityCalls != 0 {
		t.Fatalf("training hot path validation calls = validate:%d orthogonality:%d, want no full audit validation per vector", validateCalls, orthogonalityCalls)
	}

	if _, err := trainer.Transform().ApplyVector(set.Rows[0].QueryVector); err != nil {
		t.Fatalf("strict public AOQT apply: %v", err)
	}
	if validateCalls == 0 || orthogonalityCalls == 0 {
		t.Fatalf("test hooks did not observe strict public validation: validate:%d orthogonality:%d", validateCalls, orthogonalityCalls)
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
		if math.Abs(float64(surface.scores[i]-want)) > 1e-6 {
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
		if math.Abs(float64(surface.scores[i]-raw)) > 1e-6 {
			t.Fatalf("surface score[%d] = %.9g, want raw prepared-IP %.9g", i, surface.scores[i], raw)
		}
		if math.Abs(float64(raw-unit)) > 1e-6 {
			t.Fatalf("raw/unit prepared-IP score[%d] diverged %.9g vs %.9g for unit-vector row", i, raw, unit)
		}
	}
}

func TestAOQTStage2BPreparedIPObjectiveWorkspaceMatchesUncachedReference(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 681)
	cfg := tinyAOQTObjectiveConfig(set.Manifest.TurboQuantSeed)
	cached, err := NewAOQTSidecarPreparedIPObjective(cfg)
	if err != nil {
		t.Fatalf("cached objective: %v", err)
	}
	uncached := AOQTSidecarPreparedIPObjective{config: normalizedAOQTPreparedIPObjectiveConfig(cfg)}
	inputs := []AOQTSidecarObjectiveInput{
		aoqtObjectiveInputForRow(set.Rows[0]),
		aoqtObjectiveInputForRow(tinyAOQTCalibrationSet(t, 682).Rows[0]),
	}
	for i, input := range inputs {
		want, err := uncached.EvaluateAOQT(input)
		if err != nil {
			t.Fatalf("uncached evaluate input %d: %v", i, err)
		}
		first, err := cached.EvaluateAOQT(input)
		if err != nil {
			t.Fatalf("cached first evaluate input %d: %v", i, err)
		}
		if cached.workspace == nil || len(cached.workspace.surfaces) != 2 {
			t.Fatalf("cached workspace surfaces after input %d = %d, want q3 and q5", i, len(cached.workspace.surfaces))
		}
		q3State := cached.workspace.surfaces[AOQTSidecarDefaultGainBit]
		q5State := cached.workspace.surfaces[AOQTSidecarDefaultGuardBit5]
		second, err := cached.EvaluateAOQT(input)
		if err != nil {
			t.Fatalf("cached second evaluate input %d: %v", i, err)
		}
		if cached.workspace.surfaces[AOQTSidecarDefaultGainBit] != q3State || cached.workspace.surfaces[AOQTSidecarDefaultGuardBit5] != q5State {
			t.Fatalf("cached objective rebuilt prepared-IP workspace after input %d", i)
		}
		assertAOQTObjectiveResultsEqual(t, fmt.Sprintf("cached first input %d", i), first, want)
		assertAOQTObjectiveResultsEqual(t, fmt.Sprintf("cached second input %d", i), second, want)
	}
}

func TestAOQTStage2BPreparedIPObjectiveWorkspaceConcurrentCopiesMatchReference(t *testing.T) {
	setA := tinyAOQTCalibrationSet(t, 683)
	setB := tinyAOQTCalibrationSet(t, 684)
	cfg := tinyAOQTObjectiveConfig(setA.Manifest.TurboQuantSeed)
	shared, err := NewAOQTSidecarPreparedIPObjective(cfg)
	if err != nil {
		t.Fatalf("shared objective: %v", err)
	}
	uncached := AOQTSidecarPreparedIPObjective{config: normalizedAOQTPreparedIPObjectiveConfig(cfg)}
	inputs := []AOQTSidecarObjectiveInput{
		aoqtObjectiveInputForRow(setA.Rows[0]),
		aoqtObjectiveInputForRow(setB.Rows[0]),
	}
	wants := make([]AOQTSidecarObjectiveResult, len(inputs))
	for i, input := range inputs {
		wants[i], err = uncached.EvaluateAOQT(input)
		if err != nil {
			t.Fatalf("uncached evaluate input %d: %v", i, err)
		}
	}

	const workers = 8
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for i := 0; i < workers; i++ {
		i := i
		wg.Add(1)
		go func(objective AOQTSidecarPreparedIPObjective) {
			defer wg.Done()
			inputIndex := i % len(inputs)
			got, err := objective.EvaluateAOQT(inputs[inputIndex])
			if err != nil {
				errs <- fmt.Errorf("cached concurrent evaluate %d: %w", i, err)
				return
			}
			if err := compareAOQTObjectiveResults(got, wants[inputIndex]); err != nil {
				errs <- fmt.Errorf("cached concurrent evaluate %d input %d: %w", i, inputIndex, err)
			}
		}(shared)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
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

func TestAOQTStage2BPreparedIPObjectiveAcceptsBoundedQ3SafeStep(t *testing.T) {
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
	if sa.OptimizerDiagnostics == nil || sb.OptimizerDiagnostics == nil {
		t.Fatalf("optimizer diagnostics missing after prepared-IP fit")
	}
	if *sa.OptimizerDiagnostics != *sb.OptimizerDiagnostics || sa.OptimizerDiagnosticsSHA256 != sb.OptimizerDiagnosticsSHA256 {
		t.Fatalf("prepared-IP diagnostics differ: %+v/%s vs %+v/%s", *sa.OptimizerDiagnostics, sa.OptimizerDiagnosticsSHA256, *sb.OptimizerDiagnostics, sb.OptimizerDiagnosticsSHA256)
	}
	if sa.OptimizerDiagnostics.AcceptedSteps != sa.Steps || sa.Steps == 0 {
		t.Fatalf("diagnostics = %+v summary steps=%d, want accepted safe prepared-IP step", sa.OptimizerDiagnostics, sa.Steps)
	}
	if sa.FinalObjectiveComponents.Q3Gain >= sa.InitialObjectiveComponents.Q3Gain {
		t.Fatalf("q3_gain final %.9g initial %.9g, want strict improvement", sa.FinalObjectiveComponents.Q3Gain, sa.InitialObjectiveComponents.Q3Gain)
	}
	if sa.AngleMaxAbs == 0 || sa.AngleL2 == 0 {
		t.Fatalf("angle stats l2=%.9g max=%.9g, want nonidentity movement", sa.AngleL2, sa.AngleMaxAbs)
	}
	metrics, err := NewAOQTSidecarRunMetrics(set, sa)
	if err != nil {
		t.Fatalf("metrics: %v", err)
	}
	if err := ValidateAOQTSidecarCandidateEligibility(metrics, AOQTSidecarCandidateEligibilityPolicy{RequireObjectiveActivation: true}); err != nil {
		t.Fatalf("candidate eligibility: %v", err)
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

func TestAOQTTransactionalStepRollsBackExactStateWhenUnsafe(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 77)
	trainer := newTinyAOQTTrainer(t, false, 77)
	grad := make([]float32, AOQTSidecarAngleCount)
	grad[0] = 1
	before := trainer.snapshotOptimizerState()
	diagnostics := AOQTSidecarOptimizerDiagnostics{
		PlannedSteps:       1,
		MaxAttemptsPerStep: aoqtTransactionalMaxAttemptsPerStep,
	}

	accepted, err := trainer.acceptTransactionalAdamStep(grad, grad, aoqtSafeStepEvaluation(1, set.Manifest.ObjectiveContract.WeightSums), func() (aoqtStepEvaluation, error) {
		return aoqtSafeStepEvaluation(2, set.Manifest.ObjectiveContract.WeightSums), nil
	}, set.Manifest.ObjectiveContract.WeightSums, &diagnostics)
	if err != nil {
		t.Fatalf("transactional step: %v", err)
	}
	if accepted {
		t.Fatalf("unsafe proposal accepted")
	}
	if trainer.step != before.step || !float32SlicesEqual(trainer.angles, before.angles) || !float32SlicesEqual(trainer.adamM, before.adamM) || !float32SlicesEqual(trainer.adamV, before.adamV) {
		t.Fatalf("optimizer state was not restored exactly after rejected proposals")
	}
	wantAttempts := aoqtTransactionalAdamMaxAttemptsPerStep + aoqtTransactionalCoordinateMagnitudeCount + aoqtTransactionalCoordinateReverseMagnitudeCount
	if diagnostics.ProposalAttempts != wantAttempts || diagnostics.RejectedProposals != wantAttempts || diagnostics.Backtracks != wantAttempts {
		t.Fatalf("diagnostics = %+v, want every available single-coordinate proposal rejected/backtracked", diagnostics)
	}
	if diagnostics.AdamProposalAttempts != aoqtTransactionalAdamMaxAttemptsPerStep || diagnostics.AdamRejectedProposals != aoqtTransactionalAdamMaxAttemptsPerStep || diagnostics.CoordinateProposalAttempts != aoqtTransactionalCoordinateMagnitudeCount+aoqtTransactionalCoordinateReverseMagnitudeCount || diagnostics.CoordinateRejectedProposals != aoqtTransactionalCoordinateMagnitudeCount+aoqtTransactionalCoordinateReverseMagnitudeCount {
		t.Fatalf("optimizer-path diagnostics = %+v, want Adam plus single-coordinate rejection accounting", diagnostics)
	}
	if diagnostics.RejectionDiagnostics.CandidateEvaluations != wantAttempts || diagnostics.RejectionDiagnostics.ReasonCounts.LossIncrease != wantAttempts {
		t.Fatalf("rejection diagnostics = %+v, want every exact proposal counted as loss_increase", diagnostics.RejectionDiagnostics)
	}
	if diagnostics.RejectionDiagnostics.DominantReason != string(aoqtRejectionLossIncrease) {
		t.Fatalf("dominant rejection = %q, want %q", diagnostics.RejectionDiagnostics.DominantReason, aoqtRejectionLossIncrease)
	}
	if diagnostics.RejectionDiagnostics.LossDelta.Count != wantAttempts || diagnostics.RejectionDiagnostics.LossDelta.Min < 0.999 || diagnostics.RejectionDiagnostics.LossDelta.Max > 1.001 {
		t.Fatalf("loss deltas = %+v, want aggregate +1 deltas for rejected proposals", diagnostics.RejectionDiagnostics.LossDelta)
	}
}

func TestAOQTTransactionalStepAcceptsSmallerScale(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 78)
	trainer := newTinyAOQTTrainer(t, false, 78)
	grad := make([]float32, AOQTSidecarAngleCount)
	grad[0] = 1
	diagnostics := AOQTSidecarOptimizerDiagnostics{
		PlannedSteps:       1,
		MaxAttemptsPerStep: aoqtTransactionalMaxAttemptsPerStep,
	}

	accepted, err := trainer.acceptTransactionalAdamStep(grad, grad, aoqtSafeStepEvaluation(1, set.Manifest.ObjectiveContract.WeightSums), func() (aoqtStepEvaluation, error) {
		if math.Abs(float64(trainer.angles[0])) > 0.0075 {
			return aoqtSafeStepEvaluation(2, set.Manifest.ObjectiveContract.WeightSums), nil
		}
		return aoqtSafeStepEvaluation(0.5, set.Manifest.ObjectiveContract.WeightSums), nil
	}, set.Manifest.ObjectiveContract.WeightSums, &diagnostics)
	if err != nil {
		t.Fatalf("transactional step: %v", err)
	}
	if !accepted {
		t.Fatalf("smaller safe proposal was rejected")
	}
	if diagnostics.ProposalAttempts != 2 || diagnostics.AcceptedProposals != 1 || diagnostics.RejectedProposals != 1 || diagnostics.Backtracks != 1 {
		t.Fatalf("diagnostics = %+v, want one backtrack then acceptance", diagnostics)
	}
	if diagnostics.AdamProposalAttempts != 2 || diagnostics.AdamAcceptedProposals != 1 || diagnostics.AdamRejectedProposals != 1 || diagnostics.CoordinateProposalAttempts != 0 {
		t.Fatalf("optimizer-path diagnostics = %+v, want Adam-only acceptance", diagnostics)
	}
	if got := math.Abs(float64(trainer.angles[0])); got < 0.0049 || got > 0.0051 {
		t.Fatalf("accepted angle magnitude = %.9g, want half-scale Adam proposal", got)
	}
	if trainer.step != 1 {
		t.Fatalf("trainer step = %d, want accepted Adam step", trainer.step)
	}
}

func TestAOQTTransactionalStepFallsBackToCoordinateAfterAdamRejection(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 81)
	trainer := newTinyAOQTTrainer(t, false, 81)
	grad := make([]float32, AOQTSidecarAngleCount)
	grad[0] = 1
	diagnostics := AOQTSidecarOptimizerDiagnostics{
		PlannedSteps:       1,
		MaxAttemptsPerStep: aoqtTransactionalMaxAttemptsPerStep,
	}
	var observedAngles []float32

	accepted, err := trainer.acceptTransactionalAdamStep(grad, grad, aoqtSafeStepEvaluation(1, set.Manifest.ObjectiveContract.WeightSums), func() (aoqtStepEvaluation, error) {
		angle := trainer.angles[0]
		observedAngles = append(observedAngles, angle)
		if angle > 0 {
			return aoqtSafeStepEvaluation(0.5, set.Manifest.ObjectiveContract.WeightSums), nil
		}
		return aoqtSafeStepEvaluation(2, set.Manifest.ObjectiveContract.WeightSums), nil
	}, set.Manifest.ObjectiveContract.WeightSums, &diagnostics)
	if err != nil {
		t.Fatalf("transactional step: %v", err)
	}
	if !accepted {
		t.Fatalf("coordinate fallback did not accept safe opposite-sign proposal")
	}
	if len(observedAngles) != 11 {
		t.Fatalf("observed %d proposals %v, want 4 Adam rejects + 7 coordinate probes", len(observedAngles), observedAngles)
	}
	for i, angle := range observedAngles[:10] {
		if angle >= 0 {
			t.Fatalf("proposal %d angle = %.9g, want negative rejected proposal before opposite sign", i, angle)
		}
	}
	if observedAngles[10] <= 0 {
		t.Fatalf("accepted proposal angle = %.9g, want opposite positive coordinate sign", observedAngles[10])
	}
	if diagnostics.ProposalAttempts != 11 || diagnostics.AcceptedProposals != 1 || diagnostics.RejectedProposals != 10 || diagnostics.Backtracks != 10 {
		t.Fatalf("diagnostics = %+v, want ten rejects and one coordinate acceptance", diagnostics)
	}
	if diagnostics.AdamProposalAttempts != aoqtTransactionalAdamMaxAttemptsPerStep || diagnostics.AdamAcceptedProposals != 0 || diagnostics.AdamRejectedProposals != aoqtTransactionalAdamMaxAttemptsPerStep {
		t.Fatalf("adam diagnostics = %+v, want all Adam probes rejected", diagnostics)
	}
	if diagnostics.CoordinateProposalAttempts != 7 || diagnostics.CoordinateAcceptedProposals != 1 || diagnostics.CoordinateRejectedProposals != 6 {
		t.Fatalf("coordinate diagnostics = %+v, want seventh coordinate probe accepted", diagnostics)
	}
	if diagnostics.CoordinateTopAngles != 1 || diagnostics.CoordinateMagnitudeCount != aoqtTransactionalCoordinateMagnitudeCount {
		t.Fatalf("coordinate search bounds = %+v, want one active angle and configured magnitudes", diagnostics)
	}
	if err := validateAOQTSHA256(diagnostics.CoordinateSearchOrderingHash, "coordinate ordering hash"); err != nil {
		t.Fatalf("coordinate ordering hash invalid: %v", err)
	}
	if got := trainer.angles[0]; got < 0.0099 || got > 0.0101 {
		t.Fatalf("accepted coordinate angle = %.9g, want +learning_rate", got)
	}
}

func TestAOQTTransactionalCoordinateFallbackUsesFineTailBeforeBoundedReverse(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 84)
	trainer := newTinyAOQTTrainer(t, false, 84)
	grad := make([]float32, AOQTSidecarAngleCount)
	for i := 0; i < aoqtTransactionalCoordinateTopAngles; i++ {
		grad[i] = float32(aoqtTransactionalCoordinateTopAngles - i)
	}
	diagnostics := AOQTSidecarOptimizerDiagnostics{
		PlannedSteps:       1,
		MaxAttemptsPerStep: aoqtTransactionalMaxAttemptsPerStep,
	}
	var proposals [][]float32

	accepted, err := trainer.acceptTransactionalAdamStep(grad, grad, aoqtSafeStepEvaluation(1, set.Manifest.ObjectiveContract.WeightSums), func() (aoqtStepEvaluation, error) {
		proposals = append(proposals, append([]float32(nil), trainer.angles...))
		return aoqtSafeStepEvaluation(2, set.Manifest.ObjectiveContract.WeightSums), nil
	}, set.Manifest.ObjectiveContract.WeightSums, &diagnostics)
	if err != nil {
		t.Fatalf("transactional step: %v", err)
	}
	if accepted {
		t.Fatalf("accepted intentionally unsafe proposal")
	}
	if len(proposals) != aoqtTransactionalMaxAttemptsPerStep {
		t.Fatalf("proposals = %d, want bounded exact %d attempts", len(proposals), aoqtTransactionalMaxAttemptsPerStep)
	}
	if diagnostics.AdamProposalAttempts != aoqtTransactionalAdamMaxAttemptsPerStep || diagnostics.CoordinateProposalAttempts != aoqtTransactionalMaxAttemptsPerStep-aoqtTransactionalAdamMaxAttemptsPerStep {
		t.Fatalf("diagnostics = %+v, want Adam plus full coordinate budget accounting", diagnostics)
	}
	magnitudes := aoqtCoordinateSearchMagnitudes(trainer.config.LearningRate)
	blockMagnitudes := aoqtCoordinateSearchMagnitudePrefix(magnitudes, aoqtTransactionalCoordinateBlockMagnitudeCount)
	reverseMagnitudes := aoqtCoordinateSearchMagnitudePrefix(magnitudes, aoqtTransactionalCoordinateReverseMagnitudeCount)
	offset := aoqtTransactionalAdamMaxAttemptsPerStep
	blockSizes := aoqtCoordinateSearchBlockSizes(aoqtTransactionalCoordinateTopAngles)
	for _, magnitude := range blockMagnitudes {
		for _, blockSize := range blockSizes {
			requireAOQTChangedAngles(t, proposals[offset], blockSize, -magnitude, "block")
			offset++
		}
	}
	for index := 0; index < aoqtTransactionalCoordinateTopAngles; index++ {
		for _, magnitude := range magnitudes {
			requireAOQTChangedAngle(t, proposals[offset], index, -magnitude, "primary")
			offset++
		}
	}
	for index := 0; index < aoqtTransactionalCoordinateTopAngles; index++ {
		for _, magnitude := range reverseMagnitudes {
			requireAOQTChangedAngle(t, proposals[offset], index, magnitude, "reverse")
			offset++
		}
	}
	if offset != len(proposals) {
		t.Fatalf("checked %d proposals, captured %d", offset, len(proposals))
	}
}

func TestAOQTTransactionalStepTriesSafeTopCoordinateBlock(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 83)
	trainer := newTinyAOQTTrainer(t, false, 83)
	grad := make([]float32, AOQTSidecarAngleCount)
	for i := 0; i < aoqtTransactionalCoordinateTopAngles; i++ {
		grad[i] = float32(aoqtTransactionalCoordinateTopAngles - i)
	}
	grad[aoqtTransactionalCoordinateTopAngles] = 0.5
	diagnostics := AOQTSidecarOptimizerDiagnostics{
		PlannedSteps:       1,
		MaxAttemptsPerStep: aoqtTransactionalMaxAttemptsPerStep,
	}

	accepted, err := trainer.acceptTransactionalAdamStep(grad, grad, aoqtSafeStepEvaluation(1, set.Manifest.ObjectiveContract.WeightSums), func() (aoqtStepEvaluation, error) {
		for i := 0; i < aoqtTransactionalCoordinateTopAngles; i++ {
			if trainer.angles[i] >= -0.0099 {
				return aoqtSafeStepEvaluation(2, set.Manifest.ObjectiveContract.WeightSums), nil
			}
		}
		if trainer.angles[aoqtTransactionalCoordinateTopAngles] != 0 {
			return aoqtSafeStepEvaluation(2, set.Manifest.ObjectiveContract.WeightSums), nil
		}
		return aoqtSafeStepEvaluation(0.5, set.Manifest.ObjectiveContract.WeightSums), nil
	}, set.Manifest.ObjectiveContract.WeightSums, &diagnostics)
	if err != nil {
		t.Fatalf("transactional step: %v", err)
	}
	if !accepted {
		t.Fatalf("top-coordinate block proposal was not accepted")
	}
	if diagnostics.AdamProposalAttempts != aoqtTransactionalAdamMaxAttemptsPerStep || diagnostics.AdamAcceptedProposals != 0 {
		t.Fatalf("adam diagnostics = %+v, want all Adam probes rejected before block search", diagnostics)
	}
	if diagnostics.CoordinateProposalAttempts != 3 || diagnostics.CoordinateAcceptedProposals != 1 || diagnostics.CoordinateRejectedProposals != 2 {
		t.Fatalf("coordinate diagnostics = %+v, want third block probe accepted", diagnostics)
	}
	if diagnostics.CoordinateTopAngles != aoqtTransactionalCoordinateTopAngles || diagnostics.CoordinateMagnitudeCount != aoqtTransactionalCoordinateMagnitudeCount || diagnostics.CoordinateBlockCount != len(aoqtTransactionalCoordinateBlockSizes) {
		t.Fatalf("coordinate search plan diagnostics = %+v, want full top/block/magnitude plan", diagnostics)
	}
	for i := 0; i < aoqtTransactionalCoordinateTopAngles; i++ {
		if got := trainer.angles[i]; got < -0.0101 || got > -0.0099 {
			t.Fatalf("angle %d = %.9g, want block descent step", i, got)
		}
	}
	if got := trainer.angles[aoqtTransactionalCoordinateTopAngles]; got != 0 {
		t.Fatalf("angle outside top block = %.9g, want untouched", got)
	}
}

func TestAOQTTransactionalCoordinateFallbackUsesQ3GainPrimaryGradient(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 87)
	trainer := newTinyAOQTTrainer(t, false, 87)
	aggregateGrad := make([]float32, AOQTSidecarAngleCount)
	q3GainGrad := make([]float32, AOQTSidecarAngleCount)
	aggregateGrad[0] = 100
	q3GainGrad[1] = -3
	diagnostics := AOQTSidecarOptimizerDiagnostics{
		PlannedSteps:       1,
		MaxAttemptsPerStep: aoqtTransactionalMaxAttemptsPerStep,
	}
	var observedAngles []float32

	accepted, err := trainer.acceptTransactionalAdamStep(aggregateGrad, q3GainGrad, aoqtSafeStepEvaluation(1, set.Manifest.ObjectiveContract.WeightSums), func() (aoqtStepEvaluation, error) {
		observedAngles = append(observedAngles, trainer.angles[1])
		if trainer.angles[1] > 0 && trainer.angles[0] == 0 {
			return aoqtSafeStepEvaluation(0.5, set.Manifest.ObjectiveContract.WeightSums), nil
		}
		return aoqtSafeStepEvaluation(2, set.Manifest.ObjectiveContract.WeightSums), nil
	}, set.Manifest.ObjectiveContract.WeightSums, &diagnostics)
	if err != nil {
		t.Fatalf("transactional step: %v", err)
	}
	if !accepted {
		t.Fatalf("coordinate fallback did not follow q3_gain-primary gradient")
	}
	if len(observedAngles) != 5 {
		t.Fatalf("observed %d proposals %v, want 4 aggregate Adam rejects then first q3 coordinate acceptance", len(observedAngles), observedAngles)
	}
	for i, angle := range observedAngles[:4] {
		if angle != 0 {
			t.Fatalf("Adam proposal %d moved q3-only coordinate angle = %.9g, want untouched", i, angle)
		}
	}
	if observedAngles[4] <= 0 {
		t.Fatalf("accepted q3-primary coordinate angle = %.9g, want positive q3 direction", observedAngles[4])
	}
	if trainer.angles[0] != 0 {
		t.Fatalf("aggregate-only angle = %.9g, want restored before q3 coordinate acceptance", trainer.angles[0])
	}
	if diagnostics.CoordinateProposalAttempts != 1 || diagnostics.CoordinateAcceptedProposals != 1 || diagnostics.CoordinateRejectedProposals != 0 {
		t.Fatalf("coordinate diagnostics = %+v, want first q3 coordinate accepted", diagnostics)
	}
	if diagnostics.CoordinateSearchStrategy != aoqtCoordinateSearchStrategyLegacyQ3GainPrimary {
		t.Fatalf("coordinate search strategy = %q, want legacy %q", diagnostics.CoordinateSearchStrategy, aoqtCoordinateSearchStrategyLegacyQ3GainPrimary)
	}
}

func TestAOQTTransactionalStepRejectsNoOpSafeCandidate(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 82)
	trainer := newTinyAOQTTrainer(t, false, 82)
	before := trainer.snapshotOptimizerState()
	grad := make([]float32, AOQTSidecarAngleCount)
	diagnostics := AOQTSidecarOptimizerDiagnostics{
		PlannedSteps:       1,
		MaxAttemptsPerStep: aoqtTransactionalMaxAttemptsPerStep,
	}

	accepted, err := trainer.acceptTransactionalAdamStep(grad, grad, aoqtSafeStepEvaluation(1, set.Manifest.ObjectiveContract.WeightSums), func() (aoqtStepEvaluation, error) {
		return aoqtSafeStepEvaluation(0.5, set.Manifest.ObjectiveContract.WeightSums), nil
	}, set.Manifest.ObjectiveContract.WeightSums, &diagnostics)
	if err != nil {
		t.Fatalf("transactional step: %v", err)
	}
	if accepted {
		t.Fatalf("accepted no-op safe candidate")
	}
	if trainer.step != before.step || !float32SlicesEqual(trainer.angles, before.angles) || !float32SlicesEqual(trainer.adamM, before.adamM) || !float32SlicesEqual(trainer.adamV, before.adamV) {
		t.Fatalf("optimizer state was not restored exactly after no-op proposals")
	}
	if diagnostics.ProposalAttempts != aoqtTransactionalAdamMaxAttemptsPerStep || diagnostics.AcceptedProposals != 0 || diagnostics.RejectedProposals != aoqtTransactionalAdamMaxAttemptsPerStep || diagnostics.CoordinateProposalAttempts != 0 {
		t.Fatalf("diagnostics = %+v, want bounded Adam-only no-op rejection", diagnostics)
	}
	if diagnostics.RejectionDiagnostics.CandidateEvaluations != aoqtTransactionalAdamMaxAttemptsPerStep || diagnostics.RejectionDiagnostics.ReasonCounts.NoAngleMovement != aoqtTransactionalAdamMaxAttemptsPerStep {
		t.Fatalf("rejection diagnostics = %+v, want Adam no-op proposals diagnosed without coordinate fallback", diagnostics.RejectionDiagnostics)
	}
}

func TestAOQTTransactionalRejectionDiagnosticsClassifyGates(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 86)
	weights := AOQTSidecarRowWeights{Q3Gain: 1}

	for _, tc := range []struct {
		name        string
		baseline    aoqtStepEvaluation
		candidate   aoqtStepEvaluation
		wantReason  aoqtTransactionalRejectionReason
		wantRegress string
	}{
		{
			name: "no-q3-gain-improvement",
			baseline: aoqtStepEvaluation{
				loss:       2,
				components: AOQTSidecarObjectiveComponents{Q3Gain: 1, Q3OrderGuard: 1},
				activation: aoqtActiveObjectiveForWeights(weights, 3),
			},
			candidate: aoqtStepEvaluation{
				loss:       1.9,
				components: AOQTSidecarObjectiveComponents{Q3Gain: 1, Q3OrderGuard: 0.9},
				activation: aoqtActiveObjectiveForWeights(weights, 3),
			},
			wantReason: aoqtRejectionNoQ3GainImprovement,
		},
		{
			name: "component-regression",
			baseline: aoqtStepEvaluation{
				loss:       2,
				components: AOQTSidecarObjectiveComponents{Q3Gain: 1, Q3ScoreDistill: 1},
				activation: aoqtActiveObjectiveForWeights(weights, 3),
			},
			candidate: aoqtStepEvaluation{
				loss:       1.9,
				components: AOQTSidecarObjectiveComponents{Q3Gain: 0.5, Q3ScoreDistill: 1.4},
				activation: aoqtActiveObjectiveForWeights(weights, 3),
			},
			wantReason:  aoqtRejectionComponentRegression,
			wantRegress: "q3_score_distill",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			trainer := newTinyAOQTTrainer(t, false, set.Manifest.Topology.Seed)
			grad := make([]float32, AOQTSidecarAngleCount)
			grad[0] = 1
			diagnostics := AOQTSidecarOptimizerDiagnostics{
				PlannedSteps:       1,
				MaxAttemptsPerStep: aoqtTransactionalMaxAttemptsPerStep,
			}

			accepted, err := trainer.acceptTransactionalAdamStep(grad, grad, tc.baseline, func() (aoqtStepEvaluation, error) {
				return tc.candidate, nil
			}, weights, &diagnostics)
			if err != nil {
				t.Fatalf("transactional step: %v", err)
			}
			if accepted {
				t.Fatalf("accepted intentionally unsafe %s proposal", tc.name)
			}
			if diagnostics.RejectionDiagnostics.CandidateEvaluations != diagnostics.ProposalAttempts {
				t.Fatalf("rejection evaluations = %d, proposal attempts = %d", diagnostics.RejectionDiagnostics.CandidateEvaluations, diagnostics.ProposalAttempts)
			}
			if diagnostics.RejectionDiagnostics.DominantReason != string(tc.wantReason) || diagnostics.RejectionDiagnostics.ReasonCounts.count(string(tc.wantReason)) != diagnostics.ProposalAttempts {
				t.Fatalf("rejection diagnostics = %+v, want all proposals rejected by %s", diagnostics.RejectionDiagnostics, tc.wantReason)
			}
			if tc.wantRegress != "" && diagnostics.RejectionDiagnostics.DominantComponentRegression != tc.wantRegress {
				t.Fatalf("dominant component regression = %q, want %q in %+v", diagnostics.RejectionDiagnostics.DominantComponentRegression, tc.wantRegress, diagnostics.RejectionDiagnostics.ComponentRegressionCounts)
			}
			if diagnostics.RejectionDiagnostics.ComponentDeltas.Q3Gain.Count != diagnostics.ProposalAttempts {
				t.Fatalf("component deltas = %+v, want exact proposal aggregate deltas", diagnostics.RejectionDiagnostics.ComponentDeltas)
			}
		})
	}
}

func TestAOQTCoordinateSearchOrderingIsDeterministic(t *testing.T) {
	q3GainGrad := make([]float32, 8)
	aggregateGrad := make([]float32, 8)
	q3GainGrad[0] = 1
	q3GainGrad[1] = -3
	q3GainGrad[2] = 1
	q3GainGrad[4] = -2
	q3GainGrad[5] = 2
	copy(aggregateGrad, q3GainGrad)
	order, err := rankedAOQTCoordinateSearchAngles(q3GainGrad, aggregateGrad)
	if err != nil {
		t.Fatalf("rank coordinates: %v", err)
	}
	var got []int
	top := aoqtTransactionalCoordinateTopAngles
	if len(order) < top {
		top = len(order)
	}
	for _, item := range order[:top] {
		got = append(got, item.index)
	}
	want := []int{1, 4, 5, 0, 2}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("coordinate order = %v, want abs-desc/index-stable %v", got, want)
	}
	if dirs := aoqtCoordinateSearchDirections(q3GainGrad[1]); dirs != [2]float32{1, -1} {
		t.Fatalf("negative-gradient directions = %v, want positive then negative", dirs)
	}
	if dirs := aoqtCoordinateSearchDirections(q3GainGrad[0]); dirs != [2]float32{-1, 1} {
		t.Fatalf("positive-gradient directions = %v, want negative then positive", dirs)
	}
	magnitudes := []float32{0.01, 0.005}
	plan := aoqtCoordinateSearchPlan{
		Strategy:          aoqtCoordinateSearchStrategyLegacyQ3GainPrimary,
		Order:             order[:top],
		Magnitudes:        magnitudes,
		BlockMagnitudes:   aoqtCoordinateSearchMagnitudePrefix(magnitudes, 1),
		ReverseMagnitudes: aoqtCoordinateSearchMagnitudePrefix(magnitudes, 1),
		BlockSizes:        aoqtCoordinateSearchBlockSizes(top),
	}
	if fmt.Sprint(plan.BlockSizes) != fmt.Sprint([]int{2, 4}) {
		t.Fatalf("block sizes = %v, want deterministic applicable prefix blocks", plan.BlockSizes)
	}
	ha, err := plan.SHA256()
	if err != nil {
		t.Fatalf("hash A: %v", err)
	}
	hb, err := plan.SHA256()
	if err != nil {
		t.Fatalf("hash B: %v", err)
	}
	if ha != hb {
		t.Fatalf("ordering hash is not deterministic: %s vs %s", ha, hb)
	}
	if err := validateAOQTSHA256(ha, "coordinate ordering hash"); err != nil {
		t.Fatalf("ordering hash invalid: %v", err)
	}
	blockChanged := plan
	blockChanged.BlockMagnitudes = []float32{magnitudes[1]}
	hc, err := blockChanged.SHA256()
	if err != nil {
		t.Fatalf("hash C: %v", err)
	}
	reverseChanged := plan
	reverseChanged.ReverseMagnitudes = []float32{magnitudes[1]}
	hd, err := reverseChanged.SHA256()
	if err != nil {
		t.Fatalf("hash D: %v", err)
	}
	if ha == hc || ha == hd {
		t.Fatalf("ordering hash must bind block/reverse magnitudes: base=%s block=%s reverse=%s", ha, hc, hd)
	}
}

func TestAOQTCoordinateSearchOrderingUsesQ3GainPrimaryTieBreaks(t *testing.T) {
	q3GainGrad := []float32{0, 2, -2, 2, -2, 1}
	aggregateGrad := []float32{100, 0.25, -10, -50, -5, 200}
	order, err := rankedAOQTCoordinateSearchAngles(q3GainGrad, aggregateGrad)
	if err != nil {
		t.Fatalf("rank coordinates: %v", err)
	}
	var got []int
	for _, item := range order {
		got = append(got, item.index)
	}
	want := []int{2, 4, 1, 3, 5}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("coordinate order = %v, want q3-abs then non-conflict/aggregate/index tie-breaks %v", got, want)
	}
	if !order[3].guardConflict {
		t.Fatalf("rank for index 3 should record aggregate/q3 guard conflict")
	}
	if order[0].aggregateAbs != 10 {
		t.Fatalf("rank aggregate abs = %.9g, want plan hash to bind aggregate tie-break", order[0].aggregateAbs)
	}
}

func TestAOQTProtectedConeMicroTailPlanIsDeterministicAndAudited(t *testing.T) {
	q3GainGrad := []float32{3, -2, 2, -1, 1, -4, 0, 0}
	aggregateGrad := []float32{3, -2, -8, -1, 5, -4, 0, 0}
	protected := []aoqtProtectedAngleGradient{
		{Name: "q3_order_guard", Grad: []float32{-1, -1, 1, -1, 1, -1, 0, 0}},
		{Name: "q3_score_distill", Grad: []float32{-2, -1, 1, -1, 1, -2, 0, 0}},
		{Name: "q5_order_guard", Grad: []float32{-1, 0, 1, 0, 1, -1, 0, 0}},
		{Name: "q5_score_distill", Grad: []float32{-1, -1, 1, -1, 1, -1, 0, 0}},
		{Name: "nf_boundary_guard", Grad: []float32{-1, -1, 1, -1, 1, -1, 0, 0}},
	}
	a, err := newAOQTCoordinateSearchPlan(q3GainGrad, aggregateGrad, 0.01, protected)
	if err != nil {
		t.Fatalf("plan A: %v", err)
	}
	b, err := newAOQTCoordinateSearchPlan(q3GainGrad, aggregateGrad, 0.01, protected)
	if err != nil {
		t.Fatalf("plan B: %v", err)
	}
	if a.Strategy != aoqtCoordinateSearchStrategyProtectedConeMicroTail {
		t.Fatalf("strategy = %q, want %q", a.Strategy, aoqtCoordinateSearchStrategyProtectedConeMicroTail)
	}
	wantTail := []float32{0.00125, 0.000625, 0.0003125}
	if !float32SlicesEqual(a.Magnitudes, wantTail) || !float32SlicesEqual(a.BlockMagnitudes, wantTail) || !float32SlicesEqual(a.MicroTailMagnitudes, wantTail) {
		t.Fatalf("micro-tail magnitudes = %v/%v/%v, want %v", a.Magnitudes, a.BlockMagnitudes, a.MicroTailMagnitudes, wantTail)
	}
	if len(a.ReverseMagnitudes) != 0 {
		t.Fatalf("reverse magnitudes = %v, want q3-descent-only tail", a.ReverseMagnitudes)
	}
	if fmt.Sprint(a.ProtectedComponents) != fmt.Sprint([]string{"q3_order_guard", "q3_score_distill", "q5_order_guard", "q5_score_distill", "nf_boundary_guard"}) {
		t.Fatalf("protected components = %v, want fixed component order", a.ProtectedComponents)
	}
	ha, err := a.SHA256()
	if err != nil {
		t.Fatalf("plan A hash: %v", err)
	}
	hb, err := b.SHA256()
	if err != nil {
		t.Fatalf("plan B hash: %v", err)
	}
	if ha != hb {
		t.Fatalf("protected plan hash is not deterministic: %s vs %s", ha, hb)
	}
	mutated := a
	mutated.ProtectedComponents = append([]string(nil), a.ProtectedComponents...)
	mutated.ProtectedComponents[0] = "tampered_component"
	hc, err := mutated.SHA256()
	if err != nil {
		t.Fatalf("mutated plan hash: %v", err)
	}
	if ha == hc {
		t.Fatalf("plan hash does not bind protected component identity: %s", ha)
	}
}

func TestAOQTCoordinateSearchPlanKeepsLegacyProvenanceWithoutProtectedSet(t *testing.T) {
	q3GainGrad := []float32{3, -2, 1, 0}
	aggregateGrad := append([]float32(nil), q3GainGrad...)
	legacy, err := newAOQTCoordinateSearchPlan(q3GainGrad, aggregateGrad, 0.01)
	if err != nil {
		t.Fatalf("legacy plan: %v", err)
	}
	if legacy.Strategy != aoqtCoordinateSearchStrategyLegacyQ3GainPrimary {
		t.Fatalf("legacy strategy = %q, want %q", legacy.Strategy, aoqtCoordinateSearchStrategyLegacyQ3GainPrimary)
	}
	if len(legacy.Magnitudes) != aoqtTransactionalCoordinateMagnitudeCount || len(legacy.BlockMagnitudes) != aoqtTransactionalCoordinateBlockMagnitudeCount || len(legacy.ReverseMagnitudes) != aoqtTransactionalCoordinateReverseMagnitudeCount || len(legacy.MicroTailMagnitudes) != 0 || len(legacy.ProtectedComponents) != 0 {
		t.Fatalf("legacy schedule/protected audit = %+v, want historical full schedule and no protected set", legacy)
	}
	protected := []aoqtProtectedAngleGradient{{Name: "q3_score_distill", Grad: []float32{-1, 0, -1, 0}}}
	protectedPlan, err := newAOQTCoordinateSearchPlan(q3GainGrad, aggregateGrad, 0.01, protected)
	if err != nil {
		t.Fatalf("protected plan: %v", err)
	}
	if protectedPlan.Strategy != aoqtCoordinateSearchStrategyProtectedConeMicroTail || len(protectedPlan.Magnitudes) != aoqtTransactionalCoordinateMicroTailMagnitudeCount || len(protectedPlan.ReverseMagnitudes) != 0 {
		t.Fatalf("protected strategy/schedule = %+v, want exact protected micro-tail", protectedPlan)
	}
}

func TestAOQTProtectedConeMicroTailRequiresThreeFiniteMagnitudes(t *testing.T) {
	protected := []aoqtProtectedAngleGradient{{Name: "q3_score_distill", Grad: []float32{-1}}}
	_, err := newAOQTCoordinateSearchPlan([]float32{1}, []float32{1}, math.SmallestNonzeroFloat32, protected)
	if err == nil || !strings.Contains(err.Error(), "exactly 3 finite positive micro-tail magnitudes") {
		t.Fatalf("subnormal learning-rate plan error = %v, want exact protected micro-tail rejection", err)
	}
}

func TestAOQTProtectedGradientSetsRejectMultipleOrIncompleteSets(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 94)
	enableTinyAOQTNFGuard(t, &set)
	q3GainGrad := make([]float32, AOQTSidecarAngleCount)
	q3GainGrad[0] = 1
	aggregateGrad := append([]float32(nil), q3GainGrad...)
	one := []aoqtProtectedAngleGradient{{Name: "q3_order_guard", Grad: make([]float32, AOQTSidecarAngleCount)}}
	if _, err := newAOQTCoordinateSearchPlan(q3GainGrad, aggregateGrad, 0.01, one, one); err == nil || !strings.Contains(err.Error(), "at most one explicit set") {
		t.Fatalf("multiple protected gradient sets error = %v, want fail-closed variadic rejection", err)
	}
	unknown := []aoqtProtectedAngleGradient{{Name: "q3_score_distill_typo", Grad: make([]float32, AOQTSidecarAngleCount)}}
	if _, err := newAOQTCoordinateSearchPlan(q3GainGrad, aggregateGrad, 0.01, unknown); err == nil || !strings.Contains(err.Error(), "not a recognized protected component") {
		t.Fatalf("unknown protected component error = %v, want fail-closed component validation", err)
	}
	trainer := newTinyAOQTTrainer(t, false, 94)
	stateBefore := trainer.snapshotOptimizerState()
	evaluateCalls := 0
	diagnostics := AOQTSidecarOptimizerDiagnostics{PlannedSteps: 1, MaxAttemptsPerStep: aoqtTransactionalMaxAttemptsPerStep}
	accepted, err := trainer.acceptTransactionalAdamStep(aggregateGrad, q3GainGrad, aoqtSafeStepEvaluation(1, set.Manifest.ObjectiveContract.WeightSums), func() (aoqtStepEvaluation, error) {
		evaluateCalls++
		return aoqtSafeStepEvaluation(0.5, set.Manifest.ObjectiveContract.WeightSums), nil
	}, set.Manifest.ObjectiveContract.WeightSums, &diagnostics, one)
	if err == nil || !strings.Contains(err.Error(), "want active components") {
		t.Fatalf("incomplete protected gradient set error = %v, want active-component validation", err)
	}
	if accepted || evaluateCalls != 0 {
		t.Fatalf("incomplete protected set accepted=%t evaluate_calls=%d, want rejected before proposal evaluation", accepted, evaluateCalls)
	}
	if trainer.step != stateBefore.step || !float32SlicesEqual(trainer.angles, stateBefore.angles) || !float32SlicesEqual(trainer.adamM, stateBefore.adamM) || !float32SlicesEqual(trainer.adamV, stateBefore.adamV) {
		t.Fatalf("incomplete protected set mutated optimizer state")
	}
}

func TestAOQTCoordinateSearchAuditChainRejectsTruncatedHistory(t *testing.T) {
	if _, err := appendAOQTCoordinateSearchAuditChain("[]", "next-audit"); err == nil || !strings.Contains(err.Error(), "preserve a non-empty history") {
		t.Fatalf("truncated coordinate audit chain error = %v, want fail-closed history rejection", err)
	}
}

func TestAOQTProtectedConeSelectorRejectsFirstOrderProtectedAscent(t *testing.T) {
	q3GainGrad := []float32{1, -1, 0}
	aggregateGrad := append([]float32(nil), q3GainGrad...)
	protected := []aoqtProtectedAngleGradient{{
		Name: "q3_score_distill",
		Grad: []float32{-1, -1, 0},
	}}
	order, err := rankedAOQTCoordinateSearchAngles(q3GainGrad, aggregateGrad, protected)
	if err != nil {
		t.Fatalf("rank protected coordinates: %v", err)
	}
	if len(order) != 2 || order[0].index != 1 || order[1].index != 0 {
		t.Fatalf("protected coordinate order = %+v, want safe index 1 before conflicting index 0", order)
	}
	var first, second aoqtCoordinateSearchRank
	for _, ranked := range order {
		switch ranked.index {
		case 0:
			first = ranked
		case 1:
			second = ranked
		}
	}
	if first.protectedConflictCount != 1 || len(first.protectedDirectionalDerivatives) != 1 || first.protectedDirectionalDerivatives[0] <= 0 {
		t.Fatalf("conflicting coordinate rank = %+v, want positive protected directional derivative", first)
	}
	if second.protectedConflictCount != 0 || second.protectedDirectionalDerivatives[0] >= 0 {
		t.Fatalf("safe coordinate rank = %+v, want non-positive protected derivative", second)
	}
	if aoqtCoordinateBlockDirectionInCone([]aoqtCoordinateSearchRank{first}, q3GainGrad, protected) {
		t.Fatalf("protected-ascent coordinate was admitted to the first-order cone")
	}
	if !aoqtCoordinateBlockDirectionInCone([]aoqtCoordinateSearchRank{second}, q3GainGrad, protected) {
		t.Fatalf("protected-descent coordinate was rejected from the first-order cone")
	}
	if !aoqtDirectionInProtectedCone([]float32{0, 1, 0}, protected) || aoqtDirectionInProtectedCone([]float32{-1, 0, 0}, protected) {
		t.Fatalf("generic protected-cone check did not reject positive directional derivative")
	}
	if !aoqtCoordinateBlockDirectionInCone([]aoqtCoordinateSearchRank{first, second}, q3GainGrad, protected) {
		t.Fatalf("balanced block with zero protected derivative was rejected")
	}
	plan, err := newAOQTCoordinateSearchPlan(q3GainGrad, aggregateGrad, 0.01, protected)
	if err != nil {
		t.Fatalf("protected plan: %v", err)
	}
	hBefore, err := plan.SHA256()
	if err != nil {
		t.Fatalf("protected plan hash: %v", err)
	}
	plan.Order[0].protectedDirectionalDerivatives[0] = -0.5
	hAfter, err := plan.SHA256()
	if err != nil {
		t.Fatalf("mutated protected plan hash: %v", err)
	}
	if hBefore == hAfter {
		t.Fatalf("protected plan hash does not bind directional derivative audit")
	}
}

func TestAOQTProtectedConeMicroTailSkipsConflictingCoordinatesAndRollsBack(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 90)
	trainer := newTinyAOQTTrainer(t, false, 90)
	q3GainGrad := make([]float32, AOQTSidecarAngleCount)
	q3GainGrad[0] = 1
	q3GainGrad[1] = -0.5
	aggregateGrad := append([]float32(nil), q3GainGrad...)
	protected := make([]aoqtProtectedAngleGradient, 0, 5)
	for _, name := range aoqtActiveProtectedComponentNames(set.Manifest.ObjectiveContract.WeightSums) {
		protected = append(protected, aoqtProtectedAngleGradient{
			Name: name,
			Grad: func() []float32 {
				grad := make([]float32, AOQTSidecarAngleCount)
				grad[0] = -2
				grad[1] = -1
				return grad
			}(),
		})
	}
	before := trainer.snapshotOptimizerState()
	diagnostics := AOQTSidecarOptimizerDiagnostics{
		PlannedSteps:       1,
		MaxAttemptsPerStep: aoqtTransactionalMaxAttemptsPerStep,
	}
	var proposals [][]float32
	accepted, err := trainer.acceptTransactionalAdamStep(aggregateGrad, q3GainGrad, aoqtSafeStepEvaluation(1, set.Manifest.ObjectiveContract.WeightSums), func() (aoqtStepEvaluation, error) {
		proposals = append(proposals, append([]float32(nil), trainer.angles...))
		return aoqtSafeStepEvaluation(2, set.Manifest.ObjectiveContract.WeightSums), nil
	}, set.Manifest.ObjectiveContract.WeightSums, &diagnostics, protected)
	if err != nil {
		t.Fatalf("protected transactional step: %v", err)
	}
	if accepted {
		t.Fatalf("accepted intentionally unsafe candidate")
	}
	if len(proposals) != aoqtTransactionalAdamMaxAttemptsPerStep+aoqtTransactionalCoordinateMicroTailMagnitudeCount {
		t.Fatalf("evaluated proposals = %d, want Adam plus safe-coordinate micro-tail %d", len(proposals), aoqtTransactionalAdamMaxAttemptsPerStep+aoqtTransactionalCoordinateMicroTailMagnitudeCount)
	}
	for i, direction := range proposals[aoqtTransactionalAdamMaxAttemptsPerStep:] {
		if direction[0] != 0 {
			t.Fatalf("micro-tail proposal %d moved conflicting angle 0 = %.9g", i, direction[0])
		}
		if direction[1] <= 0 || !aoqtDirectionInProtectedCone(direction, protected) {
			t.Fatalf("micro-tail proposal %d direction is outside protected cone: angle1=%.9g", i, direction[1])
		}
	}
	if trainer.step != before.step || !float32SlicesEqual(trainer.angles, before.angles) || !float32SlicesEqual(trainer.adamM, before.adamM) || !float32SlicesEqual(trainer.adamV, before.adamV) {
		t.Fatalf("protected micro-tail rejection did not restore optimizer state")
	}
	if diagnostics.CoordinateSearchStrategy != aoqtCoordinateSearchStrategyProtectedConeMicroTail || diagnostics.CoordinateMagnitudeCount != aoqtTransactionalCoordinateMicroTailMagnitudeCount {
		t.Fatalf("protected coordinate diagnostics = %+v, want protected micro-tail strategy/count", diagnostics)
	}
	if diagnostics.CoordinateProposalAttempts != aoqtTransactionalCoordinateMicroTailMagnitudeCount || diagnostics.CoordinateAcceptedProposals != 0 || diagnostics.CoordinateRejectedProposals != aoqtTransactionalCoordinateMicroTailMagnitudeCount {
		t.Fatalf("protected coordinate accounting = %+v, want only safe coordinate probes", diagnostics)
	}
}

func TestAOQTPreparedIPProtectedGradientsDecomposeFullGradientAndAuditFallback(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 91)
	enableTinyAOQTNFGuard(t, &set)
	objective, err := NewAOQTSidecarPreparedIPObjective(tinyAOQTObjectiveConfig(set.Manifest.TurboQuantSeed))
	if err != nil {
		t.Fatalf("objective: %v", err)
	}
	trainer := newTinyAOQTTrainer(t, false, 91)
	fullLoss, aggregateGrad, activation, components, err := trainer.lossAndAngleGrad(set.Rows, objective)
	if err != nil {
		t.Fatalf("full objective gradient: %v", err)
	}
	q3GainGrad, err := trainer.q3GainOnlyAngleGrad(set.Rows, objective)
	if err != nil {
		t.Fatalf("q3 gain gradient: %v", err)
	}
	protected, err := trainer.protectedComponentAngleGrads(set.Rows, objective, set.Manifest.ObjectiveContract.WeightSums)
	if err != nil {
		t.Fatalf("protected gradients: %v", err)
	}
	wantNames := []string{"q3_order_guard", "q3_score_distill", "q5_order_guard", "q5_score_distill", "nf_boundary_guard"}
	if !aoqtStringSlicesEqual(aoqtProtectedGradientNames(protected), wantNames) {
		t.Fatalf("prepared protected gradient names = %v, want %v", aoqtProtectedGradientNames(protected), wantNames)
	}
	decomposed := append([]float32(nil), q3GainGrad...)
	for _, gradient := range protected {
		if len(gradient.Grad) != len(decomposed) {
			t.Fatalf("prepared protected %s gradient length = %d, want %d", gradient.Name, len(gradient.Grad), len(decomposed))
		}
		for i, value := range gradient.Grad {
			if !isFinite32(value) {
				t.Fatalf("prepared protected %s gradient[%d] = %.9g, want finite", gradient.Name, i, value)
			}
			decomposed[i] += value
		}
	}
	maxGradientDelta := float32(0)
	for i := range aggregateGrad {
		delta := float32(math.Abs(float64(aggregateGrad[i] - decomposed[i])))
		if delta > maxGradientDelta {
			maxGradientDelta = delta
		}
	}
	if maxGradientDelta > 2e-4 {
		t.Fatalf("prepared full gradient decomposition max delta = %.9g, want <= 2e-4", maxGradientDelta)
	}
	baseline := aoqtStepEvaluation{loss: fullLoss, activation: activation, components: components}
	before := trainer.snapshotOptimizerState()
	diagnostics := AOQTSidecarOptimizerDiagnostics{PlannedSteps: 1, MaxAttemptsPerStep: aoqtTransactionalMaxAttemptsPerStep}
	var proposals [][]float32
	accepted, err := trainer.acceptTransactionalAdamStep(aggregateGrad, q3GainGrad, baseline, func() (aoqtStepEvaluation, error) {
		proposals = append(proposals, append([]float32(nil), trainer.angles...))
		return baseline, nil
	}, set.Manifest.ObjectiveContract.WeightSums, &diagnostics, protected)
	if err != nil {
		t.Fatalf("prepared protected fallback: %v", err)
	}
	if accepted {
		t.Fatalf("same-objective proposals unexpectedly accepted")
	}
	if len(proposals) <= aoqtTransactionalAdamMaxAttemptsPerStep {
		t.Fatalf("prepared fallback proposals = %d, want coordinate fallback after Adam", len(proposals))
	}
	for i, proposal := range proposals[aoqtTransactionalAdamMaxAttemptsPerStep:] {
		direction := make([]float32, len(proposal))
		for j := range proposal {
			direction[j] = proposal[j] - before.angles[j]
		}
		q3Derivative := float64(0)
		for j, value := range direction {
			q3Derivative += float64(q3GainGrad[j]) * float64(value)
		}
		if !(q3Derivative < 0) || !aoqtDirectionInProtectedCone(direction, protected) {
			t.Fatalf("prepared coordinate proposal %d is outside q3/protected cone: q3 derivative %.9g", i, q3Derivative)
		}
	}
	plan, err := trainer.Plan(set)
	if err != nil {
		t.Fatalf("prepared fallback plan: %v", err)
	}
	if diagnostics.CoordinateSearchPlanCount != 1 {
		t.Fatalf("prepared fallback plan count = %d, want one audited coordinate fallback", diagnostics.CoordinateSearchPlanCount)
	}
	if err := validateAOQTProtectedCoordinateSearchAudit(plan, set.Manifest.ObjectiveContract.WeightSums, diagnostics); err != nil {
		t.Fatalf("prepared fallback audit: %v", err)
	}
	if trainer.step != before.step || !float32SlicesEqual(trainer.angles, before.angles) || !float32SlicesEqual(trainer.adamM, before.adamM) || !float32SlicesEqual(trainer.adamV, before.adamV) {
		t.Fatalf("prepared protected rejection did not restore optimizer state")
	}
}

func TestAOQTPreparedIPProtectedBalancedBlockCanBeAccepted(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 92)
	trainer := newTinyAOQTTrainer(t, false, 92)
	q3GainGrad := make([]float32, AOQTSidecarAngleCount)
	q3GainGrad[0] = 1
	q3GainGrad[1] = -1
	aggregateGrad := append([]float32(nil), q3GainGrad...)
	protected := make([]aoqtProtectedAngleGradient, 0, len(aoqtActiveProtectedComponentNames(set.Manifest.ObjectiveContract.WeightSums)))
	for _, name := range aoqtActiveProtectedComponentNames(set.Manifest.ObjectiveContract.WeightSums) {
		gradient := make([]float32, AOQTSidecarAngleCount)
		gradient[0] = -1
		gradient[1] = -1
		protected = append(protected, aoqtProtectedAngleGradient{Name: name, Grad: gradient})
	}
	baseline := aoqtSafeStepEvaluation(1, set.Manifest.ObjectiveContract.WeightSums)
	diagnostics := AOQTSidecarOptimizerDiagnostics{PlannedSteps: 1, MaxAttemptsPerStep: aoqtTransactionalMaxAttemptsPerStep}
	var proposals [][]float32
	callbackCalls := 0
	accepted, err := trainer.acceptTransactionalAdamStep(aggregateGrad, q3GainGrad, baseline, func() (aoqtStepEvaluation, error) {
		callbackCalls++
		proposals = append(proposals, append([]float32(nil), trainer.angles...))
		if callbackCalls > aoqtTransactionalAdamMaxAttemptsPerStep && trainer.angles[0] != 0 && trainer.angles[1] != 0 {
			return aoqtSafeStepEvaluation(0.5, set.Manifest.ObjectiveContract.WeightSums), nil
		}
		return aoqtSafeStepEvaluation(2, set.Manifest.ObjectiveContract.WeightSums), nil
	}, set.Manifest.ObjectiveContract.WeightSums, &diagnostics, protected)
	if err != nil {
		t.Fatalf("balanced protected block fallback: %v", err)
	}
	if !accepted {
		t.Fatalf("balanced protected block was not accepted")
	}
	if len(proposals) != aoqtTransactionalAdamMaxAttemptsPerStep+1 {
		t.Fatalf("balanced block proposals = %d, want Adam plus first accepted block", len(proposals))
	}
	block := proposals[aoqtTransactionalAdamMaxAttemptsPerStep]
	if block[0] >= 0 || block[1] <= 0 || !aoqtDirectionInProtectedCone(block, protected) {
		t.Fatalf("accepted balanced block = [%.9g %.9g], want q3/protected-cone directions", block[0], block[1])
	}
	if diagnostics.CoordinateSearchPlanCount != 1 || diagnostics.CoordinateProposalAttempts != 1 || diagnostics.CoordinateAcceptedProposals != 1 || diagnostics.CoordinateRejectedProposals != 0 {
		t.Fatalf("balanced block diagnostics = %+v, want one accepted audited coordinate block", diagnostics)
	}
	plan, err := newAOQTCoordinateSearchPlan(q3GainGrad, aggregateGrad, trainer.config.LearningRate, protected)
	if err != nil {
		t.Fatalf("balanced block plan: %v", err)
	}
	if !aoqtCoordinateBlockDirectionInCone(plan.Order[:2], q3GainGrad, protected) {
		t.Fatalf("balanced block plan was not in protected cone")
	}
}

func TestAOQTProtectedCoordinatePlanHistoryRetainsMultipleFallbacks(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 93)
	trainer := newTinyAOQTTrainer(t, false, 93)
	q3GainGrad := make([]float32, AOQTSidecarAngleCount)
	q3GainGrad[0] = 1
	aggregateGrad := append([]float32(nil), q3GainGrad...)
	protected := make([]aoqtProtectedAngleGradient, 0, len(aoqtActiveProtectedComponentNames(set.Manifest.ObjectiveContract.WeightSums)))
	for _, name := range aoqtActiveProtectedComponentNames(set.Manifest.ObjectiveContract.WeightSums) {
		protected = append(protected, aoqtProtectedAngleGradient{Name: name, Grad: make([]float32, AOQTSidecarAngleCount)})
	}
	diagnostics := AOQTSidecarOptimizerDiagnostics{PlannedSteps: 2, MaxAttemptsPerStep: aoqtTransactionalMaxAttemptsPerStep}
	baseline := aoqtSafeStepEvaluation(1, set.Manifest.ObjectiveContract.WeightSums)
	for step := 0; step < 2; step++ {
		accepted, err := trainer.acceptTransactionalAdamStep(aggregateGrad, q3GainGrad, baseline, func() (aoqtStepEvaluation, error) {
			return aoqtSafeStepEvaluation(2, set.Manifest.ObjectiveContract.WeightSums), nil
		}, set.Manifest.ObjectiveContract.WeightSums, &diagnostics, protected)
		if err != nil {
			t.Fatalf("fallback %d: %v", step, err)
		}
		if accepted {
			t.Fatalf("fallback %d unexpectedly accepted unsafe candidate", step)
		}
	}
	if diagnostics.CoordinateSearchPlanCount != 2 {
		t.Fatalf("coordinate search plan count = %d, want two fallback plans", diagnostics.CoordinateSearchPlanCount)
	}
	audits, err := decodeAOQTCoordinateSearchStringChain(diagnostics.CoordinateSearchAuditChain, "coordinate_search_audit_chain")
	if err != nil {
		t.Fatalf("audit history: %v", err)
	}
	hashes, err := decodeAOQTCoordinateSearchStringChain(diagnostics.CoordinateSearchHashChain, "coordinate_search_hash_chain")
	if err != nil {
		t.Fatalf("hash history: %v", err)
	}
	if len(audits) != 2 || len(hashes) != 2 {
		t.Fatalf("coordinate history lengths = %d/%d, want two per-step entries", len(audits), len(hashes))
	}
	plan, err := trainer.Plan(set)
	if err != nil {
		t.Fatalf("history plan: %v", err)
	}
	if err := validateAOQTProtectedCoordinateSearchAudit(plan, set.Manifest.ObjectiveContract.WeightSums, diagnostics); err != nil {
		t.Fatalf("history audit validation: %v", err)
	}
	tampered := diagnostics
	var tamperedHashes []string
	if err := strictUnmarshalAOQT([]byte(tampered.CoordinateSearchHashChain), &tamperedHashes); err != nil {
		t.Fatalf("decode tampered hash history: %v", err)
	}
	tamperedHashes[0] = strings.Repeat("a", 64)
	tamperedHashChain, err := json.Marshal(tamperedHashes)
	if err != nil {
		t.Fatalf("marshal tampered hash history: %v", err)
	}
	tampered.CoordinateSearchHashChain = string(tamperedHashChain)
	if err := validateAOQTProtectedCoordinateSearchAudit(plan, set.Manifest.ObjectiveContract.WeightSums, tampered); err == nil || !strings.Contains(err.Error(), "bind audit payload/predecessor") {
		t.Fatalf("tampered predecessor hash history error = %v, want linked-chain rejection", err)
	}
}

func TestAOQTQ3GainOnlyAngleGradMatchesAggregateForQ3OnlyObjective(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 88)
	for i := range set.Rows {
		set.Rows[i].Weights = AOQTSidecarRowWeights{Q3Gain: set.Rows[i].Weights.Q3Gain}
	}
	set.Manifest.ObjectiveContract = tinyAOQTObjectiveConfig(set.Manifest.TurboQuantSeed).ObjectiveContract(sumAOQTRowWeights(set.Rows))
	trainer := newTinyAOQTTrainer(t, false, 88)
	_, aggregateGrad, _, _, err := trainer.lossAndAngleGrad(set.Rows, toyAOQTObjective{})
	if err != nil {
		t.Fatalf("aggregate q3-only grad: %v", err)
	}
	q3GainGrad, err := trainer.q3GainOnlyAngleGrad(set.Rows, toyAOQTObjective{})
	if err != nil {
		t.Fatalf("q3-only grad: %v", err)
	}
	if !float32SlicesEqual(aggregateGrad, q3GainGrad) {
		t.Fatalf("q3-only gradient differs from aggregate gradient for q3-only objective")
	}
}

func TestAOQTProtectedV2FitRejectsInactiveProtectedObjective(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 95)
	for i := range set.Rows {
		set.Rows[i].Weights = AOQTSidecarRowWeights{Q3Gain: set.Rows[i].Weights.Q3Gain}
	}
	set.Manifest.ObjectiveContract = tinyAOQTObjectiveConfig(set.Manifest.TurboQuantSeed).ObjectiveContract(sumAOQTRowWeights(set.Rows))
	if err := set.Validate(); err != nil {
		t.Fatalf("q3-only fixture invalid: %v", err)
	}
	trainer := newTinyAOQTTrainer(t, false, 95)
	if _, err := trainer.Fit(set, toyAOQTObjective{}); err == nil || !strings.Contains(err.Error(), "protected-v2 non-plan training requires") {
		t.Fatalf("q3-only non-plan fit error = %v, want explicit protected-v2 fail-closed policy", err)
	}
}

func TestAOQTTransactionalFitRejectsZeroAcceptedAndRestoresState(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 79)
	trainer := newTinyAOQTTrainer(t, false, 79)
	before := trainer.snapshotOptimizerState()
	objective := &statefulRejectingAOQTObjective{config: tinyAOQTObjectiveConfig(set.Manifest.TurboQuantSeed)}

	summary, err := trainer.Fit(set, objective)
	if err == nil || !strings.Contains(err.Error(), "accepted zero safe steps") || !strings.Contains(err.Error(), "dominant_reason=loss_increase(") {
		t.Fatalf("fit error = %v, want zero-accepted transactional failure with bounded rejection diagnostics", err)
	}
	if summary.OptimizerDiagnostics == nil || summary.OptimizerDiagnostics.AcceptedSteps != 0 || summary.OptimizerDiagnostics.ProposalAttempts <= 0 || summary.OptimizerDiagnostics.ProposalAttempts > aoqtTransactionalMaxAttemptsPerStep {
		t.Fatalf("diagnostics = %+v, want zero accepted and bounded proposals", summary.OptimizerDiagnostics)
	}
	attempts := summary.OptimizerDiagnostics.ProposalAttempts
	if summary.OptimizerDiagnostics.RejectionDiagnostics.CandidateEvaluations != attempts || summary.OptimizerDiagnostics.RejectionDiagnostics.ReasonCounts.LossIncrease != attempts {
		t.Fatalf("rejection diagnostics = %+v, want exact exhausted proposal accounting for %d proposals", summary.OptimizerDiagnostics.RejectionDiagnostics, attempts)
	}
	if trainer.step != before.step || !float32SlicesEqual(trainer.angles, before.angles) || !float32SlicesEqual(trainer.adamM, before.adamM) || !float32SlicesEqual(trainer.adamV, before.adamV) {
		t.Fatalf("optimizer state was not restored after zero accepted fit")
	}
}

func TestAOQTTransactionalDiagnosticsHashIsDeterministic(t *testing.T) {
	set := tinyAOQTCalibrationSet(t, 80)
	a := newTinyAOQTTrainer(t, false, 80)
	b := newTinyAOQTTrainer(t, false, 80)
	sa, err := a.Fit(set, toyAOQTObjective{})
	if err != nil {
		t.Fatalf("fit A: %v", err)
	}
	sb, err := b.Fit(set, toyAOQTObjective{})
	if err != nil {
		t.Fatalf("fit B: %v", err)
	}
	if sa.OptimizerDiagnostics == nil || sb.OptimizerDiagnostics == nil {
		t.Fatalf("optimizer diagnostics missing")
	}
	if *sa.OptimizerDiagnostics != *sb.OptimizerDiagnostics || sa.OptimizerDiagnosticsSHA256 != sb.OptimizerDiagnosticsSHA256 {
		t.Fatalf("diagnostics differ: %+v/%s vs %+v/%s", *sa.OptimizerDiagnostics, sa.OptimizerDiagnosticsSHA256, *sb.OptimizerDiagnostics, sb.OptimizerDiagnosticsSHA256)
	}
	if sa.OptimizerDiagnostics.AcceptedSteps != sa.Steps || sa.OptimizerDiagnostics.AcceptedSteps == 0 {
		t.Fatalf("diagnostics accepted steps = %d summary steps = %d", sa.OptimizerDiagnostics.AcceptedSteps, sa.Steps)
	}
	if got, err := sa.OptimizerDiagnostics.SHA256(); err != nil || got != sa.OptimizerDiagnosticsSHA256 {
		t.Fatalf("diagnostics sha = %s/%v, want %s", got, err, sa.OptimizerDiagnosticsSHA256)
	}
}

type toyAOQTObjective struct{}

type unaccountedAOQTObjective struct{}

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
	return AOQTSidecarObjectiveResult{Loss: loss, Components: AOQTSidecarObjectiveComponents{Q3Gain: loss}, QueryGrad: queryGrad, CandidateGrads: candidateGrads, Activation: aoqtActiveObjectiveForWeights(input.Row.Weights, len(input.Candidates))}, nil
}

func (unaccountedAOQTObjective) EvaluateAOQT(input AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error) {
	queryGrad := make([]float32, len(input.Query))
	candidateGrads := make([][]float32, len(input.Candidates))
	for i := range input.Candidates {
		candidateGrads[i] = make([]float32, len(input.Candidates[i]))
	}
	return AOQTSidecarObjectiveResult{Loss: 1, QueryGrad: queryGrad, CandidateGrads: candidateGrads}, nil
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
	candidates := [][]float32{aoqtRandomVec(rng), aoqtRandomVec(rng), append([]float32(nil), query...)}
	rowID := "row-0001"
	qrels := hex64("qrels")
	compat := hex64("compat")
	splitProof := tinyAOQTSplitProof()
	turboQuantSeed := int64(77)
	docIDs := []string{"d1", "d2", "d3"}
	denseScores := []float32{dotAOQT(query, candidates[0]), dotAOQT(query, candidates[1]), dotAOQT(query, candidates[2])}
	q3Scores := preparedAOQTScores(query, candidates, AOQTSidecarDefaultGuardBit3, turboQuantSeed)
	q5Scores := preparedAOQTScores(query, candidates, AOQTSidecarDefaultGuardBit5, turboQuantSeed)
	rows := []AOQTSidecarCalibrationRow{{
		Schema:                AOQTSidecarRowSchema,
		RowID:                 rowID,
		Dataset:               "toy",
		QueryID:               "q1",
		QueryVectorID:         "qv1",
		QueryVectorSHA256:     aoqtVectorSHA256(query),
		QueryVector:           query,
		CandidateDocIDs:       docIDs,
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
			Dense: denseScores,
			Q3:    q3Scores,
			Q5:    q5Scores,
		},
		AnchorRanks: AOQTSidecarAnchorRanks{
			Dense: ranksAOQT(docIDs, denseScores),
			Q3:    ranksAOQT(docIDs, q3Scores),
			Q5:    ranksAOQT(docIDs, q5Scores),
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
		TurboQuantSeed: turboQuantSeed,
		QuantSurfaces: []AOQTSidecarQuantSurface{
			{BitWidth: 3, Seed: turboQuantSeed, ScoreSurface: AOQTSidecarPreparedIPScoreSurface, PreparedQuery: true},
			{BitWidth: 5, Seed: turboQuantSeed, ScoreSurface: AOQTSidecarPreparedIPScoreSurface, PreparedQuery: true},
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
	result, err := (toyAOQTObjective{}).EvaluateAOQT(input)
	if err != nil {
		return AOQTSidecarObjectiveResult{}, err
	}
	input.Row.CandidateDocIDs[0] = "mutated-doc"
	input.Row.AnchorScores.Dense[0] = 99
	input.Row.SplitProof.ExclusionIdentities[0] = "mutated-split"
	input.Row.Extra["note"][0] = 'x'
	input.Query[0] = 99
	input.Candidates[0][0] = 99
	return result, nil
}

func (toyAOQTObjective) AOQTPreparedIPObjectiveConfig() AOQTSidecarPreparedIPObjectiveConfig {
	return tinyAOQTObjectiveConfig(77)
}

func (unaccountedAOQTObjective) AOQTPreparedIPObjectiveConfig() AOQTSidecarPreparedIPObjectiveConfig {
	return tinyAOQTObjectiveConfig(77)
}

func (mutatingAOQTObjective) AOQTPreparedIPObjectiveConfig() AOQTSidecarPreparedIPObjectiveConfig {
	return tinyAOQTObjectiveConfig(77)
}

func tinyAOQTObjectiveConfig(seed int64) AOQTSidecarPreparedIPObjectiveConfig {
	return AOQTSidecarPreparedIPObjectiveConfig{TurboQuantSeed: seed}
}

func aoqtObjectiveInputForRow(row AOQTSidecarCalibrationRow) AOQTSidecarObjectiveInput {
	return AOQTSidecarObjectiveInput{
		Row:        aoqtObjectiveRowView(row),
		Query:      append([]float32(nil), row.QueryVector...),
		Candidates: cloneAOQTVectors(row.CandidateVectors),
	}
}

func assertAOQTObjectiveResultsEqual(t *testing.T, label string, got, want AOQTSidecarObjectiveResult) {
	t.Helper()
	if err := compareAOQTObjectiveResults(got, want); err != nil {
		t.Fatalf("%s: %v", label, err)
	}
}

func compareAOQTObjectiveResults(got, want AOQTSidecarObjectiveResult) error {
	if got.Loss != want.Loss || got.Components != want.Components || got.Activation != want.Activation {
		return fmt.Errorf("result mismatch:\ngot  loss %.9g components %+v activation %+v\nwant loss %.9g components %+v activation %+v", got.Loss, got.Components, got.Activation, want.Loss, want.Components, want.Activation)
	}
	if len(got.QueryGrad) != len(want.QueryGrad) {
		return fmt.Errorf("query grad length = %d, want %d", len(got.QueryGrad), len(want.QueryGrad))
	}
	for i := range got.QueryGrad {
		if got.QueryGrad[i] != want.QueryGrad[i] {
			return fmt.Errorf("query grad[%d] = %.9g, want %.9g", i, got.QueryGrad[i], want.QueryGrad[i])
		}
	}
	if len(got.CandidateGrads) != len(want.CandidateGrads) {
		return fmt.Errorf("candidate grad count = %d, want %d", len(got.CandidateGrads), len(want.CandidateGrads))
	}
	for i := range got.CandidateGrads {
		if len(got.CandidateGrads[i]) != len(want.CandidateGrads[i]) {
			return fmt.Errorf("candidate grad[%d] length = %d, want %d", i, len(got.CandidateGrads[i]), len(want.CandidateGrads[i]))
		}
		for j := range got.CandidateGrads[i] {
			if got.CandidateGrads[i][j] != want.CandidateGrads[i][j] {
				return fmt.Errorf("candidate grad[%d][%d] = %.9g, want %.9g", i, j, got.CandidateGrads[i][j], want.CandidateGrads[i][j])
			}
		}
	}
	return nil
}

type statefulRejectingAOQTObjective struct {
	config AOQTSidecarPreparedIPObjectiveConfig
	calls  int
}

func (o *statefulRejectingAOQTObjective) AOQTPreparedIPObjectiveConfig() AOQTSidecarPreparedIPObjectiveConfig {
	return o.config
}

func (o *statefulRejectingAOQTObjective) EvaluateAOQT(input AOQTSidecarObjectiveInput) (AOQTSidecarObjectiveResult, error) {
	o.calls++
	queryGrad := make([]float32, len(input.Query))
	for i := range queryGrad {
		// A dense upstream vector guarantees a non-zero angle direction for
		// the identity/unit-vector fixtures, allowing runner tests to exercise
		// the protected coordinate fallback rather than only Adam no-ops.
		queryGrad[i] = 1
	}
	candidateGrads := make([][]float32, len(input.Candidates))
	for i := range input.Candidates {
		candidateGrads[i] = make([]float32, len(input.Candidates[i]))
	}
	loss := float32(1)
	if o.calls > 1 {
		loss = 2
	}
	return AOQTSidecarObjectiveResult{
		Loss:           loss,
		Components:     AOQTSidecarObjectiveComponents{Q3Gain: loss},
		QueryGrad:      queryGrad,
		CandidateGrads: candidateGrads,
		Activation:     aoqtActiveObjectiveForWeights(input.Row.Weights, len(input.Candidates)),
	}, nil
}

func aoqtSafeStepEvaluation(loss float32, weights AOQTSidecarRowWeights) aoqtStepEvaluation {
	return aoqtStepEvaluation{
		loss:       loss,
		components: AOQTSidecarObjectiveComponents{Q3Gain: loss},
		activation: aoqtActiveObjectiveForWeights(weights, 3),
	}
}

func aoqtActiveObjectiveForWeights(weights AOQTSidecarRowWeights, candidates int) AOQTSidecarObjectiveActivation {
	if candidates < 2 {
		candidates = 2
	}
	activation := AOQTSidecarObjectiveActivation{}
	if weights.Q3Gain > 0 {
		activation.Q3GainEligiblePairs = 1
		activation.Q3GainContributingPairs = 1
	}
	if weights.Q3OrderGuard > 0 {
		activation.Q3OrderGuardPairs = 1
		activation.Q3OrderGuardContributing = 1
	}
	if weights.Q3ScoreDistill > 0 {
		activation.Q3ScoreDistillCount = candidates
	}
	if weights.Q5OrderGuard > 0 {
		activation.Q5OrderGuardPairs = 1
		activation.Q5OrderGuardContributing = 1
	}
	if weights.Q5ScoreDistill > 0 {
		activation.Q5ScoreDistillCount = candidates
	}
	if weights.NFBoundaryGuard > 0 {
		activation.NFBoundaryGuardPairs = 1
		activation.NFBoundaryGuardContributing = 1
	}
	return activation
}

func float32SlicesEqual(a, b []float32) bool {
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

func requireAOQTChangedAngles(t *testing.T, angles []float32, wantCount int, wantValue float32, context string) {
	t.Helper()
	changed := 0
	for i, angle := range angles[:aoqtTransactionalCoordinateTopAngles] {
		if i < wantCount {
			if math.Abs(float64(angle-wantValue)) > 1e-7 {
				t.Fatalf("%s angle %d = %.9g, want %.9g", context, i, angle, wantValue)
			}
			changed++
			continue
		}
		if angle != 0 {
			t.Fatalf("%s angle %d = %.9g, want untouched", context, i, angle)
		}
	}
	if changed != wantCount {
		t.Fatalf("%s changed %d angles, want %d", context, changed, wantCount)
	}
}

func requireAOQTChangedAngle(t *testing.T, angles []float32, wantIndex int, wantValue float32, context string) {
	t.Helper()
	for i, angle := range angles[:aoqtTransactionalCoordinateTopAngles] {
		if i == wantIndex {
			if math.Abs(float64(angle-wantValue)) > 1e-7 {
				t.Fatalf("%s angle %d = %.9g, want %.9g", context, i, angle, wantValue)
			}
			continue
		}
		if angle != 0 {
			t.Fatalf("%s angle %d = %.9g, want untouched", context, i, angle)
		}
	}
}

func snapshotAOQTVectorAngleGradReference(transform AOQTGivensTransform, input, outputGrad []float32) ([]float32, error) {
	if err := validateAOQTGivensTrainingSnapshot(transform); err != nil {
		return nil, err
	}
	if len(input) != transform.Dim || len(outputGrad) != transform.Dim {
		return nil, fmt.Errorf("AOQT vector/grad dims must match transform dim")
	}
	if err := validateAOQTFiniteVector(input, "AOQT input"); err != nil {
		return nil, err
	}
	if err := validateAOQTFiniteVector(outputGrad, "AOQT output grad"); err != nil {
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
	angleGrad := make([]float32, countAOQTAngles(transform))
	grad := append([]float32(nil), outputGrad...)
	angleIndex := len(angleGrad)
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
	return angleGrad, nil
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
