package eosruntime

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"
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
	}
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
	input.Query[0] = 99
	input.Candidates[0][0] = 99
	queryGrad := make([]float32, len(input.Query))
	candidateGrads := make([][]float32, len(input.Candidates))
	for i := range candidateGrads {
		candidateGrads[i] = make([]float32, len(input.Candidates[i]))
	}
	return AOQTSidecarObjectiveResult{Loss: 0, QueryGrad: queryGrad, CandidateGrads: candidateGrads}, nil
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
	}
	return out
}
